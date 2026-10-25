"""r9u 数据集（2026-10-10，docs/laya-r22-plan-2026-10-10.md）：r9s 全部题按 **f9lf** 重渲染 + 2026-10-09 线上题（第 470~487 局）。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9u.py            # 写数据集 + 重标清单
  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9u.py --check    # 只渲染、打印统计与内容 md5（逐字确定性检查）

输出：
- datasets/r9u/points.jsonl       题面 f9lf（R22 训练 / 评估用）
- datasets/r9u_f9l/points.jsonl   同一批 id、同一划分、同一选项键，题面 f9l（线上 R17 看到的文本，用于 R17 对照）
- datasets/r9u/{info.json, new_ids.json, matches.json, relabel.json, relabel_train_ids.json, relabel_test_ids.json}

规则：
- 旧题：id / 划分 / 选项键都不动（按各自 render_cfg 渲染，ALLOW_NEW_KEYS 后限回原选项键）；渲染失败的题两个版本一起丢。
- 新题：data/laya/iterations/batch2_matches.jsonl 里 match 470~487、没有 note 的局的 laya_points.jsonl（全部题型），剥掉学生答案；
  按线上当时的选项限回（laya_policy._restrict 删掉的不加回来）；swap 训练题补回 bench（同 r9s）。
  第 485~487 局整局进新测试集 test_r9u_new（全部题都标，作"最新测试集"）；其余进 train（只标受新事实影响的题）。
- 重标清单（新事实会改变判断的题，只在 f9lf 文本上判，不看旧标签）：
    A 刷新：商店题的 refresh 选项写了 "can't afford a hero"（付费刷新后剩 <3 能量，或免费刷新但手里 <3 能量）；
    B 发英雄牌：题里的候选卡（商店 / N 选 1 / 天赋 / 用牌 / 拍卖）有 grants_pool 卡池（局面里出现 "your plan"）；
    C 专属天赋：天赋题选项里出现 "= Ali lv6 EXCLUSIVE talent"。
  旧题只在已有标签的题里挑（没标签的题训练用不上）；新训练题挑 A/B/C；test_r9u_new 全部。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_r9s as BR  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

VER = "f9lf"
OLD_VER = "f9l"                      # 线上 R17
BASE = "r9s"
OLD_LABELS = PROJECT_DIR / "data" / "laya" / "labels" / "r9s_all" / "labels_all.jsonl"
MATCH_RANGE = (470, 487)
TEST_MATCHES = {485, 486, 487}
TEST_SPLIT = "test_r9u_new"
STUDENT_KEYS = BR.STUDENT_KEYS + ("model",)
GRANT_RE = re.compile(r"your plan|pool \? \(tier")
EXCL_TAG = "= Ali lv6 EXCLUSIVE talent"
AFFORD_TAG = "can't afford a hero"


def render(p: dict, ver: str, keys: list | None) -> dict | None:
    """按 ver 渲染一道题，限回 keys（None = 不限）。失败返回 None。"""
    F.ACTIVE = ver
    BR.apply_cfg(p.get("render_cfg"))
    r = RR.rerender(dict(p))
    if r is None:
        return None
    if keys is not None and list(r["question"]["criteria"]) != list(keys):
        if not BR.restrict_to(r, list(keys)) or list(r["question"]["criteria"]) != list(keys):
            return None
    r["feature_version"] = ver
    return r


def tags_of(r: dict) -> list[str]:
    crit = r["question"]["criteria"]
    out = []
    if r["kind"] == "shop" and AFFORD_TAG in (crit.get("refresh") or ""):
        out.append("A_refresh_cant_afford")
    if r["kind"] in ("shop", "card_pick", "talent", "use_card", "auction") and GRANT_RE.search(r["state"]):
        out.append("B_grant_pool")
    if r["kind"] == "talent" and any(EXCL_TAG in v for v in crit.values()):
        out.append("C_exclusive_talent")
    return out


def new_points(stats: collections.Counter) -> list[dict]:
    """第 470~487 局的线上题。"""
    runs, match_no = [], {}
    for line in (PROJECT_DIR / "data" / "laya" / "iterations" / "batch2_matches.jsonl").open(encoding="utf-8"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if MATCH_RANGE[0] <= (m.get("match") or 0) <= MATCH_RANGE[1] and not m.get("note") and m.get("run"):
            runs.append(m["run"])
            match_no[m["run"]] = m["match"]
    cfgs = BR.run_render_cfgs()
    out = []
    for d in sorted(runs):
        f = PROJECT_DIR / "data" / "logs" / d / "laya_points.jsonl"
        if not f.exists():
            stats["run_without_points"] += 1
            continue
        rows = []
        for line in f.open(encoding="utf-8"):
            try:
                rows.append(dict(json.loads(line), run=d))
            except ValueError:
                stats["online_bad_json"] += 1
        log_cfg = {k: v for k, v in (cfgs.get(d) or {}).items() if k != "log"} or None
        F.ACTIVE = OLD_VER                  # 按线上原文（f9l）反推当时的路线配置
        RR.ALLOW_NEW_KEYS = True
        cfg, hit = BR.pick_cfg(rows, log_cfg)
        for p in rows:
            if not p.get("question") or p.get("kind") == "eco":
                continue
            for k in STUDENT_KEYS:
                p.pop(k, None)
            p["id"] = BR.pid(d, p.get("ts"), p["kind"])
            p["source"] = "online"
            p["render_cfg"] = cfg
            p["match"] = f"r9u_m{match_no[d]:03d}"
            p["match_no"] = match_no[d]
            p["split"] = TEST_SPLIT if match_no[d] in TEST_MATCHES else "train"
            stats["online_raw:" + p["kind"]] += 1
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    st = collections.Counter()
    labels = set()
    for line in OLD_LABELS.open(encoding="utf-8"):
        if line.strip():
            labels.add(json.loads(line)["id"])
    base = [json.loads(l) for l in (ds / BASE / "points.jsonl").open(encoding="utf-8")]
    base_ids = {p["id"] for p in base}
    t0 = time.time()

    new_rows, old_rows = [], []      # (f9lf, f9l) 成对
    # ---- 旧题：按原选项键重渲染（按 run 排序，复用 states.jsonl 缓存）
    for p in sorted(base, key=lambda p: (p.get("run") or "", p["id"])):
        keys = list(p["options"])
        a = render(p, VER, keys)
        b = render(p, OLD_VER, keys)
        if a is None or b is None:
            st["base_dropped:" + p["kind"]] += 1
            continue
        new_rows.append(a)
        old_rows.append(b)
    st["base_kept"] = len(new_rows)
    # ---- 新题
    seen, seen_shop = set(), set()
    for p in sorted(new_points(st), key=lambda p: (p["match"], p.get("ts") or 0)):
        if p["id"] in base_ids:
            st["new_id_in_base"] += 1
            continue
        online_keys = list(p["question"]["criteria"])
        keys = online_keys
        if p["kind"] == "swap" and p["split"] == "train" and "bench" not in keys:
            full = render(p, VER, None)                 # 补回 bench（同 r9s）
            if full is not None and "bench" in full["question"]["criteria"]:
                keys = [k for k in full["question"]["criteria"] if k in set(online_keys) | {"bench"}]
                st["new_swap_bench_added"] += 1
        a = render(p, VER, keys)
        b = render(p, OLD_VER, keys)
        if a is None or b is None:
            st["new_dropped:" + p["kind"]] += 1
            continue
        sig = (p["match"], p.get("round"), p["kind"], json.dumps(a["question"]["criteria"], ensure_ascii=False, sort_keys=True))
        if sig in seen:
            st["new_dup_same_round_same_options:" + p["kind"]] += 1
            continue
        if p["kind"] == "shop":
            sk = (p["match"], p.get("round"), BR.shop_key(p), tuple(a["question"]["criteria"]))
            if sk in seen_shop:
                st["new_dup_same_round_same_shop"] += 1
                continue
            seen_shop.add(sk)
        seen.add(sig)
        st["new_kept:" + p["split"] + ":" + p["kind"]] += 1
        new_rows.append(a)
        old_rows.append(b)
    BR.apply_cfg(None)
    F.ACTIVE = VER
    new_ids = [r["id"] for r in new_rows if r["id"] not in base_ids]
    assert len({r["id"] for r in new_rows}) == len(new_rows)

    # ---- 重标清单
    relabel = {}
    for r in new_rows:
        tg = tags_of(r)
        is_new = r["id"] not in base_ids
        if r["split"] == TEST_SPLIT:
            relabel[r["id"]] = tg or ["T_latest_test_all"]
        elif tg and (r["id"] in labels or is_new):
            relabel[r["id"]] = tg
    rows_by = {r["id"]: r for r in new_rows}
    tr = sorted(i for i in relabel if rows_by[i]["split"] == "train")
    te = sorted(i for i in relabel if rows_by[i]["split"] != "train")
    for i in relabel:
        for t in relabel[i]:
            st[f"relabel:{'train' if rows_by[i]['split'] == 'train' else 'test'}:{rows_by[i]['kind']}:{t}"] += 1
    st["relabel_train"], st["relabel_test"] = len(tr), len(te)

    h = hashlib.md5()
    for r in new_rows:
        h.update(json.dumps([r["id"], r["state"], r["question"]], ensure_ascii=False, sort_keys=True).encode())
    summary = {"points": len(new_rows), "new": len(new_ids), "md5_f9lf": h.hexdigest(),
               "secs": round(time.time() - t0), "stats": dict(sorted(st.items())),
               "by_split": dict(collections.Counter(r["split"] for r in new_rows))}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    for name, rows in (("r9u", new_rows), ("r9u_f9l", old_rows)):
        (ds / name).mkdir(parents=True, exist_ok=True)
        with (ds / name / "points.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (ds / name / "info.json").write_text(json.dumps(dict(summary, base=BASE, feature_version=VER if name == "r9u" else OLD_VER,
                                                             built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                             matches=list(MATCH_RANGE), test_matches=sorted(TEST_MATCHES)),
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    od = ds / "r9u"
    (od / "new_ids.json").write_text(json.dumps(new_ids), encoding="utf-8")
    (od / "relabel.json").write_text(json.dumps(relabel, ensure_ascii=False, indent=0), encoding="utf-8")
    (od / "relabel_train_ids.json").write_text(json.dumps(tr), encoding="utf-8")
    (od / "relabel_test_ids.json").write_text(json.dumps(te), encoding="utf-8")
    (od / "matches.json").write_text(json.dumps(sorted({r["match"] for r in new_rows if r["id"] in set(new_ids)})),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
