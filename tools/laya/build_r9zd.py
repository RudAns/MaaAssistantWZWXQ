"""r9zd 数据集（2026-10-14，复制自 build_r9zb.py）：r9zb 全部题按 **f9lp** 重渲染 + 第 606~627 局线上题。

f9lp = 流派事实（标注规范第 52 节 g9za）：
  最低配套行补"缺的人在不在眼前商店里、一次买齐多少能量、钱和空位够就写 COMPLETES"；
  新增 FLOW OPTIONS 行（agent/sim/flows.flow_rank：官网阿离阵容按主C 合并 + 路线最低配套 + 遗志海月，前 3 个流派已有 / 缺 / 商店里有）。
设计目标（2026-10-14）：能判定"核心配件场上有一个、商店里有两个，买了就全部凑齐"这类事实；推荐阵容是最终目标，中间按现实往不同流派靠。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zd.py            # 写数据集 + 重标清单
  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zd.py --check    # 只渲染、打印统计

输出：
- datasets/r9zd/points.jsonl        题面 f9lp（R29 训练 / 评估）
- datasets/r9zd_f9lm/points.jsonl   同一批 id / 划分 / 选项键，题面 f9lm（线上 R27 看到的文本，R27 对照）
- datasets/r9zd/{info.json, new_ids.json, matches.json, relabel.json, relabel_*_ids.json, teacher_*_ids.json}

重标清单（g9za）：新局（606~627）全部 Laya 题（carry / swap 除外，线上由局面网络答）；旧题里：
    P_complete  状态行写着 COMPLETES（现在买就凑齐 最低配套 / 某流派核心）
    Q_flow      FLOW OPTIONS 第一个流派的主C 不是现任主C、且这个流派已有 ≥ 一半
旧训练题按标签封顶抽样，旧测试集抽 200。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_r9s as BR  # noqa: E402
import build_r9za as BA  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

VER = "f9lp"
OLD_VER = "f9lm"                     # 线上 R27
BASE = "r9zb"
MATCH_RANGE = (606, 627)
TEST_MATCHES = {617, 625}
TEST_SPLIT = "test_r9zd_new"
EXTRA_RUNS: dict = {}
STUDENT_KEYS = BR.STUDENT_KEYS + ("model",)
LAYA_KINDS = BA.LAYA_KINDS
NEW_TAGS = ("COMPLETES", "FLOW OPTIONS")


def render(p: dict, ver: str, keys: list | None) -> dict | None:
    return BA.render(p, ver, keys)


def tags_of_m(r: dict) -> list[str]:
    """f9lp 新事实会改变判断的题（只看 f9lp 文本）。"""
    if r["kind"] not in LAYA_KINDS:
        return []
    st = r.get("state") or ""
    out = []
    if "COMPLETES" in st:
        out.append("P_complete")
    i = st.find("FLOW OPTIONS")
    if i >= 0:
        first = st[i:].split(": ", 1)[-1].split(" | ")[0]          # "X flow: have a/b (...)"
        name = first.split(" flow:")[0].strip()
        try:
            have_part = first.split("have ", 1)[1]
            a, b = have_part.split(" ")[0].split("/")
            half = int(a) * 2 >= int(b)
            owned = have_part.split("(", 1)[1].split(")")[0].split(", ")
        except (IndexError, ValueError):
            half, owned = False, []
        carry = ""
        for key in ("current carry", "future carry: "):            # 现任 / 未来主C 写在这两种行里
            j = st.find(key)
            if j >= 0:
                carry += st[j:j + 40] + " "
        # 真有转型选项：排第一的流派不是现任主C 的，它的最终主C 已经在我方（场上 / 手里），且核心已有一半
        if half and name and name in owned and name not in carry:
            out.append("Q_flow")
    return out


def new_points(stats: collections.Counter) -> list[dict]:
    """第 588~605 局的线上题。"""
    runs, match_no = [], {}
    for line in (PROJECT_DIR / "data" / "laya" / "iterations" / "batch2_matches.jsonl").open(encoding="utf-8"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        mn = m.get("match") or 0
        if not (MATCH_RANGE[0] <= mn <= MATCH_RANGE[1]) or not m.get("run"):
            continue
        if mn in EXTRA_RUNS:
            for d in EXTRA_RUNS[mn]:
                runs.append(d)
                match_no[d] = mn
        elif not m.get("note"):
            runs.append(m["run"])
            match_no[m["run"]] = mn
    cfgs = BR.run_render_cfgs()
    out = []
    for d in sorted(dict.fromkeys(runs)):
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
        F.ACTIVE = OLD_VER                  # 按线上原文（f9lm）反推当时的路线配置
        RR.ALLOW_NEW_KEYS = True
        cfg, hit = BR.pick_cfg(rows, log_cfg)
        stats[f"cfg_hit:{d}"] = hit
        for p in rows:
            if not p.get("question") or p.get("kind") == "eco":
                continue
            for k in STUDENT_KEYS:
                p.pop(k, None)
            p["id"] = BR.pid(d, p.get("ts"), p["kind"])
            p["source"] = "online"
            p["render_cfg"] = cfg
            p["match"] = f"r9zd_m{match_no[d]:03d}"
            p["match_no"] = match_no[d]
            p["split"] = TEST_SPLIT if match_no[d] in TEST_MATCHES else "train"
            stats["online_raw:" + p["kind"]] += 1
            out.append(p)
    return out


# 省 token 的重标清单（协调方 2026-10-13 定，目标训练 + 测试合计约 100 批）：
#   carry / swap 不进这轮 Laya 重标（线上由局面网络回答，Laya 的这两题沿用旧标签）；
#   旧训练题按标签抽样封顶（固定种子、按局分层）；新局（588~605）非 carry/swap 训练题全收；
#   测试：test_r9zb_new 非 carry/swap 全收，旧测试集里带新标签的抽 TEST_OLD_CAP 题。
SKIP_KINDS = ("carry", "swap")
CAPS = (("P_complete", 500), ("Q_flow", 400))
TEST_OLD_CAP = 200


def _stratified(ids: list, rows_by: dict, cap: int, rng: random.Random) -> list:
    """按局分层抽样：每局内部打乱，轮流从各局取，直到 cap。"""
    by = collections.defaultdict(list)
    for i in sorted(ids):
        r = rows_by[i]
        by[str(r.get("match") or r.get("run") or "?")].append(i)
    keys = sorted(by)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(by[k])
    out = []
    while len(out) < cap and any(by.values()):
        for k in keys:
            if by[k] and len(out) < cap:
                out.append(by[k].pop())
    return out


def select_relabel(relabel: dict, rows_by: dict, new_set: set, seed: int) -> tuple[list, list, dict]:
    rng = random.Random(seed)
    tr, te = set(), set()
    cand = {i: t for i, t in relabel.items() if rows_by[i]["kind"] not in SKIP_KINDS}
    for i, t in cand.items():
        if i in new_set:
            (te if rows_by[i]["split"] == TEST_SPLIT else tr).add(i)
    old_tr = {i: t for i, t in cand.items() if i not in new_set and rows_by[i]["split"] == "train"}
    for tag, cap in CAPS:
        ids = [i for i, t in old_tr.items() if tag in t and i not in tr]
        tr |= set(_stratified(ids, rows_by, cap, rng))
    old_te = [i for i in cand if i not in new_set and rows_by[i]["split"] != "train"]
    te |= set(_stratified(old_te, rows_by, TEST_OLD_CAP, rng))
    st = collections.Counter()
    for i in tr | te:
        for t in relabel[i]:
            st[f"{'train' if i in tr else 'test'}:{rows_by[i]['kind']}:{t}"] += 1
    return sorted(tr), sorted(te), dict(sorted(st.items()))


def reselect(seed: int) -> int:
    """不重渲染：读已建好的 r9zd，重写 teacher_*_ids.json（省 token 版重标清单）。"""
    od = PROJECT_DIR / "data" / "laya" / "datasets" / "r9zd"
    rows_by = {}
    for line in (od / "points.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        rows_by[r["id"]] = r
    relabel = json.loads((od / "relabel.json").read_text(encoding="utf-8"))
    new_set = set(json.loads((od / "new_ids.json").read_text(encoding="utf-8")))
    tr, te, st = select_relabel(relabel, rows_by, new_set, seed)
    (od / "teacher_train_ids.json").write_text(json.dumps(tr), encoding="utf-8")
    (od / "teacher_test_ids.json").write_text(json.dumps(te), encoding="utf-8")
    print(json.dumps({"teacher_train": len(tr), "teacher_test": len(te), "by": st}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reselect", action="store_true", help="只按省 token 规则重选重标清单（不重渲染）")
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    if args.reselect:
        return reselect(args.seed)
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    st = collections.Counter()
    base = [json.loads(l) for l in (ds / BASE / "points.jsonl").open(encoding="utf-8")]
    base_ids = {p["id"] for p in base}
    t0 = time.time()

    new_rows, old_rows = [], []      # (f9lp, f9lm) 成对
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
    new_set = set(new_ids)
    assert len({r["id"] for r in new_rows}) == len(new_rows)

    # ---- 重标清单：新局全部 Laya 题 + 旧题里 f9lp 新事实出现的题
    relabel = {}
    for r in new_rows:
        if r["kind"] not in LAYA_KINDS:
            continue
        tg = tags_of_m(r)
        if r["id"] in new_set:
            relabel[r["id"]] = tg + (["T_latest_test_all"] if r["split"] == TEST_SPLIT else ["S_new_match"])
        elif tg:
            relabel[r["id"]] = tg
    rows_by = {r["id"]: r for r in new_rows}
    tr = sorted(i for i in relabel if rows_by[i]["split"] == "train")
    te = sorted(i for i in relabel if rows_by[i]["split"] != "train")
    for i in relabel:
        for t in relabel[i]:
            st[f"relabel:{'train' if rows_by[i]['split'] == 'train' else 'test'}:{rows_by[i]['kind']}:{t}"] += 1
    st["relabel_train"], st["relabel_test"] = len(tr), len(te)
    # 旧测试集里也会出现新标签（test_r9za_new / test_r9z_new …）：一并重标，评估时新旧学生看同一版规范标签
    st["relabel_old_test_splits"] = len([i for i in te if i not in new_set])

    h = hashlib.md5()
    for r in new_rows:
        h.update(json.dumps([r["id"], r["state"], r["question"]], ensure_ascii=False, sort_keys=True).encode())
    summary = {"points": len(new_rows), "new": len(new_ids), "md5_f9lp": h.hexdigest(),
               "secs": round(time.time() - t0), "stats": dict(sorted(st.items())),
               "by_split": dict(collections.Counter(r["split"] for r in new_rows)),
               "new_by_split": dict(collections.Counter(rows_by[i]["split"] for i in new_ids))}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    for name, rows in (("r9zd", new_rows), ("r9zd_f9lm", old_rows)):
        (ds / name).mkdir(parents=True, exist_ok=True)
        with (ds / name / "points.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (ds / name / "info.json").write_text(json.dumps(dict(summary, base=BASE, feature_version=VER if name == "r9zd" else OLD_VER,
                                                             built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                             matches=list(MATCH_RANGE), test_matches=sorted(TEST_MATCHES)),
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    od = ds / "r9zd"
    (od / "new_ids.json").write_text(json.dumps(new_ids), encoding="utf-8")
    (od / "relabel.json").write_text(json.dumps(relabel, ensure_ascii=False, indent=0), encoding="utf-8")
    (od / "relabel_train_ids.json").write_text(json.dumps(tr), encoding="utf-8")
    (od / "relabel_test_ids.json").write_text(json.dumps(te), encoding="utf-8")
    ttr, tte, _ = select_relabel(relabel, rows_by, new_set, args.seed)       # 老师实际要标的（省 token 版）
    (od / "teacher_train_ids.json").write_text(json.dumps(ttr), encoding="utf-8")
    (od / "teacher_test_ids.json").write_text(json.dumps(tte), encoding="utf-8")
    (od / "matches.json").write_text(json.dumps(sorted({r["match"] for r in new_rows if r["id"] in new_set})),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
