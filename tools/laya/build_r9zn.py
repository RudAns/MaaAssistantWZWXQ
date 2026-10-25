"""r9zn 数据集（2026-10-16，f9lw / g9zh）。

- r9zm 全部题按 **f9lw** 重渲染（R33 的 f9lv 修正全部保留：f9lw 只在 f9lv 上叠加）；
- 第 652 局起的线上题（测试 test_r9zn_new）；
- **离线新题**：sell_filler（没人等位时卖不卖占位件）+ 点兵点将的 use_timing（build_dataset.extract_run，SELL_FILLER_QUESTIONS）。
  线上从没出过这两类题，只能从各局 states.jsonl 还原。
重标清单（2026-10-16：要求先修掉所有已知问题、该重标的重标后再训练，避免下次训练结果里旧问题复现）：
  新局全部 Laya 题 + 换人题；带 f9lw 新标签的题按类封顶（W_*，见 TAGS）；sell_filler / use_timing 全部（封顶）。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zn.py [--check]
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
import build_r9zm as BM  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

VER = "f9lw"
PREV_VER = "f9lv"
BASE = "r9zm"
NAME = "r9zn"
MATCH_RANGE = (652, 700)
TEST_MATCHES = set(range(652, 701))
TEST_SPLIT = "test_r9zn_new"
OFFLINE_FROM = 560                     # 离线 sell_filler / use_timing 题从这局起（states.jsonl 有棋手等级与场上读数）
LAYA_KINDS = BA.LAYA_KINDS + ("sell_filler", "use_timing")
STUDENT_KEYS = BR.STUDENT_KEYS + ("model",)

# f9lw 新标签 → 重标类别（文本检索；a = f9lw 渲染）
TAGS = (
    ("W_maple_temp", lambda k, t: "MAPLE HOST on TEMPORARY carry" in t or "maple moves to the future carry" in t),
    ("W_filler", lambda k, t: "PAID OFF" in t or "PAID-OFF" in t),
    ("W_other_carry", lambda k, t: "OTHER LINEUP" in t),
    ("W_2nd_output", lambda k, t: "only parks gear" in t),
    ("W_dbdj", lambda k, t: k == "use_timing" and "copy of each board hero" in t),
    ("W_rift", lambda k, t: k == "use_card" and "rift fields" in t),
    ("W_chaos_talent", lambda k, t: k == "talent" and "AVOID:" in t),
    ("W_sell_filler", lambda k, t: k == "sell_filler"),
    ("W_basic_host", lambda k, t: k == "talent" and "rewards" in t and "BASIC-ATTACK" in t),
)
# 每类重标封顶（训练, 测试）。新局（652+）的题不受封顶，全标
CAPS = {"W_maple_temp": (220, 60), "W_filler": (260, 60), "W_other_carry": (300, 70), "W_2nd_output": (160, 40),
        "W_dbdj": (80, 20), "W_rift": (60, 15), "W_chaos_talent": (60, 15), "W_sell_filler": (320, 80),
        "W_basic_host": (0, 0)}


def render(p: dict, ver: str, keys) -> dict | None:
    return BA.render(p, ver, keys)


def tags(a: dict) -> list[str]:
    t = json.dumps([a.get("state"), a["question"]], ensure_ascii=False)
    return [name for name, f in TAGS if f(a["kind"], t)]


def online_new(st) -> list[dict]:
    BM.MATCH_RANGE, BM.TEST_MATCHES, BM.TEST_SPLIT = MATCH_RANGE, TEST_MATCHES, TEST_SPLIT
    out = BM.new_points(st)
    for p in out:
        p["match"] = p["match"].replace("r9zm_", NAME + "_")
    return out


def offline_points(st) -> list[dict]:
    """sell_filler / 点兵点将 use_timing：各局 states.jsonl 还原（线上没出过这两类题）。"""
    import build_dataset as BD
    BD.SELL_FILLER_QUESTIONS = True
    runs = {}
    for line in (PROJECT_DIR / "data/laya/iterations/batch2_matches.jsonl").open(encoding="utf-8"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if (m.get("match") or 0) >= OFFLINE_FROM and m.get("run"):
            runs[m["run"]] = m["match"]
    cfgs = BR.run_render_cfgs()
    out = []
    for d, mn in sorted(runs.items(), key=lambda x: x[1]):
        rd = PROJECT_DIR / "data/logs" / d
        if not (rd / "states.jsonl").exists():
            st["offline_no_states"] += 1
            continue
        rows = []
        f = rd / "laya_points.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        F.ACTIVE = BM.OLD_VER
        RR.ALLOW_NEW_KEYS = True
        cfg, _hit = BR.pick_cfg(rows, {k: v for k, v in (cfgs.get(d) or {}).items() if k != "log"} or None)
        BR.apply_cfg(cfg)
        F.ACTIVE = VER
        F._facts()
        try:
            pts = BD.extract_run(rd)
        except Exception as e:  # noqa: BLE001
            st["offline_err"] += 1
            print("extract_run 出错", d, type(e).__name__, e)
            continue
        for p in pts:
            if p["kind"] not in ("sell_filler", "use_timing"):
                continue
            p["run"] = d
            p["id"] = BR.pid(d, p.get("ts"), p["kind"])
            p["source"] = "offline"
            p["render_cfg"] = cfg
            p["match"] = f"{NAME}_m{mn:03d}"
            p["match_no"] = mn
            p["split"] = TEST_SPLIT if mn in TEST_MATCHES else "train"
            p["feature_version"] = VER
            st[f"offline:{p['split']}:{p['kind']}"] += 1
            out.append(p)
    BR.apply_cfg(None)
    return out


def _strat(ids, rows_by, cap, rng):
    return BM._stratified(ids, rows_by, cap, rng)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=20261016)
    args = ap.parse_args()
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data/laya/datasets"
    st = collections.Counter()
    t0 = time.time()
    base = [json.loads(l) for l in (ds / BASE / "points.jsonl").open(encoding="utf-8")]
    base_ids = {p["id"] for p in base}
    rows, changed = [], set()
    for p in sorted(base, key=lambda p: (p.get("run") or "", p["id"])):
        keys = list(p["options"])
        a = render(p, VER, keys)
        if a is None:
            st["base_dropped:" + p["kind"]] += 1
            continue
        c = render(p, PREV_VER, keys)
        if c is None or json.dumps([a.get("state"), a["question"]], ensure_ascii=False, sort_keys=True) != \
                json.dumps([c.get("state"), c["question"]], ensure_ascii=False, sort_keys=True):
            changed.add(a["id"])
        rows.append(a)
    st["base_kept"], st["base_changed_text"] = len(rows), len(changed)
    seen = set()
    for p in sorted(online_new(st), key=lambda p: (p["match"], p.get("ts") or 0)):
        if p["id"] in base_ids:
            continue
        keys = list(p["question"]["criteria"])
        a = render(p, VER, keys)
        if a is None:
            st["new_dropped:" + p["kind"]] += 1
            continue
        sig = (p["match"], p.get("round"), p["kind"], json.dumps(a["question"]["criteria"], ensure_ascii=False, sort_keys=True))
        if sig in seen:
            continue
        seen.add(sig)
        st[f"new_kept:{a['split']}:{a['kind']}"] += 1
        rows.append(a)
    have = {r["id"] for r in rows}
    for p in offline_points(st):
        if p["id"] in have:
            continue
        have.add(p["id"])
        rows.append(p)
    BR.apply_cfg(None)
    F.ACTIVE = VER
    new_ids = [r["id"] for r in rows if r["id"] not in base_ids]
    new_set = set(new_ids)
    rows_by = {r["id"]: r for r in rows}
    assert len(rows_by) == len(rows)

    # ---- 重标清单
    rng = random.Random(args.seed)
    tg = {r["id"]: tags(r) for r in rows if r["kind"] in LAYA_KINDS or r["kind"] == "swap"}
    for i, t in tg.items():
        for x in t:
            st[f"tag:{rows_by[i]['split'] == 'train' and 'train' or 'test'}:{x}"] += 1
    tr, te = set(), set()
    for i in new_set:                       # 新局线上题全标（离线新题按 W_sell_filler / W_dbdj 封顶）
        r = rows_by[i]
        if r.get("source") == "offline" or r["kind"] not in LAYA_KINDS + ("swap",):
            continue
        (te if r["split"] == TEST_SPLIT else tr).add(i)
    for name, (ctr, cte) in CAPS.items():
        idx_tr = [i for i, t in tg.items() if name in t and rows_by[i]["split"] == "train" and i not in tr]
        idx_te = [i for i, t in tg.items() if name in t and rows_by[i]["split"] != "train" and i not in te]
        tr |= set(_strat(idx_tr, rows_by, ctr, rng))
        te |= set(_strat(idx_te, rows_by, cte, rng))
    sel = collections.Counter()
    for i in tr | te:
        sel[f"{'train' if i in tr else 'test'}:{rows_by[i]['kind']}"] += 1
    up = sorted(i for i, t in tg.items() if any(x.startswith("W_") for x in t) and rows_by[i]["split"] == "train")

    h = hashlib.md5()
    for r in rows:
        h.update(json.dumps([r["id"], r["state"], r["question"]], ensure_ascii=False, sort_keys=True).encode())
    summary = {"points": len(rows), "new": len(new_ids), "md5": h.hexdigest(), "secs": round(time.time() - t0),
               "relabel_train": len(tr), "relabel_test": len(te), "selected": dict(sorted(sel.items())),
               "upsample_candidates": len(up), "stats": dict(sorted(st.items())),
               "by_split": dict(collections.Counter(r["split"] for r in rows))}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    od = ds / NAME
    od.mkdir(parents=True, exist_ok=True)
    with (od / "points.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (od / "info.json").write_text(json.dumps(dict(summary, base=BASE, feature_version=VER, built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                  matches=list(MATCH_RANGE), test_matches=sorted(TEST_MATCHES)),
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    (od / "new_ids.json").write_text(json.dumps(new_ids), encoding="utf-8")
    (od / "tags.json").write_text(json.dumps({i: t for i, t in tg.items() if t}, ensure_ascii=False), encoding="utf-8")
    (od / "changed_vs_f9lv.json").write_text(json.dumps(sorted(changed)), encoding="utf-8")
    (od / "teacher_train_ids.json").write_text(json.dumps(sorted(tr)), encoding="utf-8")
    (od / "teacher_test_ids.json").write_text(json.dumps(sorted(te)), encoding="utf-8")
    (od / "upsample_w_ids.json").write_text(json.dumps(up), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
