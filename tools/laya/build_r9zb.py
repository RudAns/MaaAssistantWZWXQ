"""r9zb 数据集（2026-10-13，复制自 build_r9za.py）：r9za 全部题按 **f9lm** 重渲染 + 第 588~605 局线上题。

f9lm = 2026-10-13 理想打法的事实（规范第 50 节 g9y）：
  临时主C（current carry X is TEMPORARY / copy of TEMPORARY carry / target is TEMPORARY carry）、
  候选路线核心卡（candidate route cores 行 + CORE of route R (own k)）、FINAL carry settled、
  天赋点名派系时 our 【X】 heroes: board a, hand b, candidate-route cores c、卡面带赠送牌效果（given_skill）。
注意：data/cards.json 2026-10-13 重爬（11 张数值改动 + 20 张补回的赠送牌），旧题重渲染时 f9lk 文本也会随卡库变（抽样约 4%，只因卡面数值）。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zb.py            # 写数据集 + 重标清单
  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zb.py --check    # 只渲染、打印统计

输出：
- datasets/r9zb/points.jsonl        题面 f9lm（R27 训练 / 评估）
- datasets/r9zb_f9lk/points.jsonl   同一批 id / 划分 / 选项键，题面 f9lk（线上 R25b 看到的文本，用于 R25b 对照）
- datasets/r9zb/{info.json, new_ids.json, matches.json, relabel.json, relabel_train_ids.json, relabel_test_ids.json,
                 teacher_train_ids.json, teacher_test_ids.json}

规则：
- 旧题（r9za）：id / 划分 / 选项键都不动；渲染失败的两个版本一起丢。
- 新题：batch2_matches.jsonl 里第 588~605 局（第 588 局带 note、中途重启，两个运行目录 live_1791831489 / live_1791831957
  都收，见 data/logs/agent_one_588.log / agent_one_588b.log），剥掉学生答案；按线上当时的选项限回；swap 训练题补回 bench。
  第 600、604 局整局进测试集 test_r9zb_new。
- relabel.json = 全部受影响的题；teacher_*_ids.json = 老师实际标的（省 token 版，见 SKIP_KINDS / CAPS，`--reselect` 只重选不重渲染）。
- 重标清单（g9y）：新局（588~605）全部 Laya 题；旧题里 f9lm 新标签出现的题：
    P_temp_opt   选项里有 TEMPORARY（copy of TEMPORARY carry / target is TEMPORARY carry）
    P_temp_carry carry / swap 题且状态行写现任主C 是 TEMPORARY（选项里不打标签，但"临时换临时"要按新规范标）
    Q_core       选项里有 CORE of route
    R_faction    天赋 / N 选 1 选项里有 our 【X】 heroes
  （FINAL carry settled / candidate route cores 状态行几乎每题都有，不单独入选）
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

VER = "f9lm"
OLD_VER = "f9lk"                     # 线上 R25b
BASE = "r9za"
MATCH_RANGE = (588, 605)
TEST_MATCHES = {600, 604}
TEST_SPLIT = "test_r9zb_new"
EXTRA_RUNS = {588: ["live_1791831489", "live_1791831957"]}   # 第 588 局中途修 bug 重启续打（batch2 带 note，原脚本会跳过）
STUDENT_KEYS = BR.STUDENT_KEYS + ("model",)
LAYA_KINDS = BA.LAYA_KINDS
NEW_TAGS = ("TEMPORARY", "CORE of route", "our 【")     # assemble_r9zb 的覆盖检查也用这组


def render(p: dict, ver: str, keys: list | None) -> dict | None:
    return BA.render(p, ver, keys)


def tags_of_m(r: dict) -> list[str]:
    """f9lm 新事实会改变判断的题（只看 f9lm 文本）。"""
    if r["kind"] not in LAYA_KINDS:
        return []
    st = r.get("state") or ""
    crit = r["question"]["criteria"]
    out = []
    if any("TEMPORARY" in v for v in crit.values()):
        out.append("P_temp_opt")
    if r["kind"] in ("carry", "swap") and "is TEMPORARY (t1" in st:
        out.append("P_temp_carry")
    if any("CORE of route" in v for v in crit.values()):
        out.append("Q_core")
    if r["kind"] in ("talent", "card_pick") and any("our 【" in v for v in crit.values()):
        out.append("R_faction")
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
        F.ACTIVE = OLD_VER                  # 按线上原文（f9lk）反推当时的路线配置
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
            p["match"] = f"r9zb_m{match_no[d]:03d}"
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
CAPS = (("P_temp_opt", 400), ("Q_core", 500), ("R_faction", 300))
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
    """不重渲染：读已建好的 r9zb，重写 teacher_*_ids.json（省 token 版重标清单）。"""
    od = PROJECT_DIR / "data" / "laya" / "datasets" / "r9zb"
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
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()
    if args.reselect:
        return reselect(args.seed)
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    st = collections.Counter()
    base = [json.loads(l) for l in (ds / BASE / "points.jsonl").open(encoding="utf-8")]
    base_ids = {p["id"] for p in base}
    t0 = time.time()

    new_rows, old_rows = [], []      # (f9lm, f9lk) 成对
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

    # ---- 重标清单：新局全部 Laya 题 + 旧题里 f9lm 新标签出现的题（新标签必须全覆盖，R25 教训）
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
    summary = {"points": len(new_rows), "new": len(new_ids), "md5_f9lm": h.hexdigest(),
               "secs": round(time.time() - t0), "stats": dict(sorted(st.items())),
               "by_split": dict(collections.Counter(r["split"] for r in new_rows)),
               "new_by_split": dict(collections.Counter(rows_by[i]["split"] for i in new_ids))}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    for name, rows in (("r9zb", new_rows), ("r9zb_f9lk", old_rows)):
        (ds / name).mkdir(parents=True, exist_ok=True)
        with (ds / name / "points.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (ds / name / "info.json").write_text(json.dumps(dict(summary, base=BASE, feature_version=VER if name == "r9zb" else OLD_VER,
                                                             built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                             matches=list(MATCH_RANGE), test_matches=sorted(TEST_MATCHES)),
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    od = ds / "r9zb"
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
