"""r9zl 数据集（2026-10-15，复制自 build_r9zk.py）：r9zk 全部题按 **f9lu** 重渲染 + 第 642 局起（测试）。重标清单由 reselect_r9zl.py 定。
旧说明：

f9lt：6 级实测概率、触发类牌合规目标、"消除最低 N 名"写实际对象、本回合限定免费刷新。规范 54.5 / 54.6（g9ze）。
旧说明：

f9ls = R30 / R30b 用牌题退步的修正：燃料行只算能用在主C 身上的效果牌、主C 没到手不写、变身 / 移除 / 摧毁装备的用牌题不写、
燃料件在手里不算稳定来源；锁店选项列出非英雄牌。对照文本 r9zj_f9lp = 线上 R29 看到的。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zj.py
重标：f9ls 与 f9lr 文本不同的题（Z_f9ls）+ 新局全部 Laya 题。
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

VER = "f9lu"
PREV_VER = "f9lt"                   # 上一版题面（r9zh 的老师标签按它标的）
OLD_VER = "f9lp"                     # 线上 R29
BASE = "r9zk"
MATCH_RANGE = (642, 670)
TEST_MATCHES = set(range(642, 671))
TEST_SPLIT = "test_r9zl_new"
EXTRA_RUNS: dict = {642: ["live_1792047112", "live_1792047504"]}   # 第 642 局中途重启，前后两段都收
STUDENT_KEYS = BR.STUDENT_KEYS + ("model",)
LAYA_KINDS = BA.LAYA_KINDS
NEW_TAGS = ("condition met", "free slots ?", "S_junk")


def render(p: dict, ver: str, keys: list | None) -> dict | None:
    return BA.render(p, ver, keys)


import re as _re

_JUNK_RE = _re.compile(r"not in ROUTE")


def swap_junk(r: dict) -> bool:
    """换人题：等着的人是路线成员 / 最低配套 / 引擎零件，场上有 ≤3 级、不在路线、不是坦克 / 引擎零件的单位（第 631 局那类）。"""
    if r["kind"] != "swap":
        return False
    cr = r["question"]["criteria"]
    b = cr.get("bench", "") + " " + r["question"].get("instructions", "")
    if not ("ROUTE member" in b or "MIN ENGINE" in b or "ENGINE PART" in b):
        return False
    for k, v in cr.items():
        if not k.startswith("s") or not _JUNK_RE.search(v) or "TANK" in v or "tank" in v or "ENGINE PART" in v:
            continue
        m = _re.search(r"lv ?(\d+)", v)
        if m and int(m.group(1)) <= 3:
            return True
    return False


def tags_of_q(a: dict, b: dict, p: dict | None = None) -> list[str]:
    """f9ls：和上一版（f9lr）文本不同的题 = Z_f9ls（p = f9lr 渲染）。"""
    if p is not None and a["kind"] in LAYA_KINDS:
        if json.dumps([a.get("state"), a["question"]], ensure_ascii=False, sort_keys=True) !=                 json.dumps([p.get("state"), p["question"]], ensure_ascii=False, sort_keys=True):
            return ["Z_f9lu"]
        return G9ZE(a)
    return tags_of_q_old(a, b)


def G9ZE(a: dict) -> list[str]:
    """g9ze 54.5 / 54.6 点到的商店题（文本没变也要按新裁决重标）。"""
    if a["kind"] != "shop":
        return []
    t = json.dumps([a.get("state"), a["question"]], ensure_ascii=False)
    out = []
    if "only merges help" in t or "finds would only wait" in t:
        out.append("E_endgame")
    if "can replace filler" in t and ("MIN ENGINE piece missing" in t or "ROUTE stage unit" in t):
        out.append("E_engine_filler")
    return out


def tags_of_q_old(a: dict, b: dict) -> list[str]:
    """f9lr 新事实出现的题（只看 f9lr 文本 a）。"""
    ta = json.dumps([a.get("state"), a["question"]], ensure_ascii=False)
    if a["kind"] == "swap":
        return ["S_idle"] if "idle: 0 heroes with own" in ta else []
    if a["kind"] not in LAYA_KINDS:
        return []
    out = []
    if "idle: 0 heroes with own" in ta:
        out.append("R_idle")
    if "CARRY FUEL" in ta and ("steady carry FUEL" in ta or "steady sources: none" in ta):
        out.append("F_fuel")
    if "FULL + " in ta:
        out.append("W_full")
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
        elif not m.get("note") or str(m.get("note")).startswith(("R29", "R31")):
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
            p["match"] = f"r9zl_m{match_no[d]:03d}"
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
CAPS = (("Z_f9lt", 600), ("E_engine_filler", 250), ("E_endgame", 350))
TEST_OLD_CAP = 150
SWAP_TRAIN_CAP, SWAP_TEST_CAP = 250, 80


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


def select_swap(relabel: dict, rows_by: dict, new_set: set, seed: int) -> tuple[list, list]:
    """换人题（局面网络用）：S_junk 的题 + 新局全部换人题；训练 / 测试分别封顶（按局分层）。"""
    rng = random.Random(seed + 1)
    sw = {i: t for i, t in relabel.items() if rows_by[i]["kind"] == "swap"}
    tr_new = [i for i in sw if i in new_set and rows_by[i]["split"] == "train"]
    te_new = [i for i in sw if i in new_set and rows_by[i]["split"] != "train"]
    tr_old = [i for i, t in sw.items() if i not in new_set and "S_idle" in t and rows_by[i]["split"] == "train"]
    te_old = [i for i, t in sw.items() if i not in new_set and "S_idle" in t and rows_by[i]["split"] != "train"]
    tr = sorted(set(tr_new) | set(_stratified(tr_old, rows_by, SWAP_TRAIN_CAP, rng)))
    te = sorted(set(te_new) | set(_stratified(te_old, rows_by, SWAP_TEST_CAP, rng)))
    return tr, te


def reselect(seed: int) -> int:
    """不重渲染：读已建好的 r9zd，重写 teacher_*_ids.json（省 token 版重标清单）。"""
    od = PROJECT_DIR / "data" / "laya" / "datasets" / "r9zl"
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

    new_rows, old_rows, prev_rows = [], [], []      # (f9ls, f9lp, f9lr) 成对
    for p in sorted(base, key=lambda p: (p.get("run") or "", p["id"])):
        keys = list(p["options"])
        a = render(p, VER, keys)
        b = render(p, OLD_VER, keys)
        c = render(p, PREV_VER, keys)
        if a is None or b is None or c is None:
            st["base_dropped:" + p["kind"]] += 1
            continue
        new_rows.append(a)
        old_rows.append(b)
        prev_rows.append(c)
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
        prev_rows.append(a)
    BR.apply_cfg(None)
    F.ACTIVE = VER
    new_ids = [r["id"] for r in new_rows if r["id"] not in base_ids]
    new_set = set(new_ids)
    assert len({r["id"] for r in new_rows}) == len(new_rows)

    # ---- 重标清单：新局全部 Laya 题 + 旧题里 f9lp 新事实出现的题
    relabel = {}
    for r, ro, rp in zip(new_rows, old_rows, prev_rows):
        if r["kind"] not in LAYA_KINDS and r["kind"] != "swap":
            continue
        tg = tags_of_q(r, ro, rp) if r["id"] not in new_set else []
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
    summary = {"points": len(new_rows), "new": len(new_ids), "md5_f9lu": h.hexdigest(),
               "secs": round(time.time() - t0), "stats": dict(sorted(st.items())),
               "by_split": dict(collections.Counter(r["split"] for r in new_rows)),
               "new_by_split": dict(collections.Counter(rows_by[i]["split"] for i in new_ids))}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    for name, rows in (("r9zl", new_rows), ("r9zl_f9lp", old_rows)):
        (ds / name).mkdir(parents=True, exist_ok=True)
        with (ds / name / "points.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (ds / name / "info.json").write_text(json.dumps(dict(summary, base=BASE, feature_version=VER if name == "r9zl" else OLD_VER,
                                                             built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                             matches=list(MATCH_RANGE), test_matches=sorted(TEST_MATCHES)),
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    od = ds / "r9zl"
    (od / "new_ids.json").write_text(json.dumps(new_ids), encoding="utf-8")
    (od / "relabel.json").write_text(json.dumps(relabel, ensure_ascii=False, indent=0), encoding="utf-8")
    (od / "relabel_train_ids.json").write_text(json.dumps(tr), encoding="utf-8")
    (od / "relabel_test_ids.json").write_text(json.dumps(te), encoding="utf-8")
    ttr, tte, _ = select_relabel(relabel, rows_by, new_set, args.seed)       # 老师实际要标的（省 token 版）
    (od / "teacher_train_ids.json").write_text(json.dumps(ttr), encoding="utf-8")
    (od / "teacher_test_ids.json").write_text(json.dumps(tte), encoding="utf-8")
    str_, ste = select_swap(relabel, rows_by, new_set, args.seed)
    (od / "teacher_swap_train_ids.json").write_text(json.dumps(str_), encoding="utf-8")
    (od / "teacher_swap_test_ids.json").write_text(json.dumps(ste), encoding="utf-8")
    print(json.dumps({"teacher_train": len(ttr), "teacher_test": len(tte), "swap_train": len(str_), "swap_test": len(ste)}))
    (od / "matches.json").write_text(json.dumps(sorted({r["match"] for r in new_rows if r["id"] in new_set})),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
