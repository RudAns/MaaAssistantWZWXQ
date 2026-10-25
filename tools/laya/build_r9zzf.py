"""r9zzf 数据集（2026-10-20 下午，f9z / g9zza = 54.28；复制自 build_r9zze.py）：r9zze 全部题按 f9z 重渲染 + 第 815 局起线上题（815~829 训练、830 起测试）。
更早：r9zze 数据集（2026-10-20，f9y / g9zz = 54.27；复制自 build_r9zzd.py）：r9zzd 全部题按 f9y 重渲染 + 第 805 局起线上题（805~810 训练、811 起测试）；定向重标 54.27。
更早：r9zzd 数据集（2026-10-19 夜，f9x / g9zy = 54.26；复制自 build_r9zzc.py）：r9zzc 全部题按 f9x 重渲染 + 第 789 局起线上题（789~795 训练、796 起测试）；定向重标 54.26。
更早：r9zzc 数据集（2026-10-19 傍晚，f9w / g9zx；复制自 build_r9zzb.py）：r9zzb 全部题按 f9w 重渲染 + 第 780 局起线上题（780~784 训练、785 起测试）；定向重标 54.25：低价值天赋、高攻速缺吸血、枫影无拘免控、特殊装备 N 选 1、路线 not_wanted / extra_wanted、阵营条件不满足。
更早：r9zzb 数据集（2026-10-19 午，f9v / g9zw；复制自 build_r9zz.py）：r9zz 全部题按 f9v 重渲染 + 第 758 局起线上题（758~773 训练、774 起测试）；定向重标 54.24：路线战绩 ours、明世隐同名 / 临时等级来源、刷新 1 能量、机器头只给木兰嬴政、核心引擎表补 5 条。
更早：r9zz 数据集（2026-10-19，f9u / g9zv；复制自 build_r9zy.py）：r9zy 全部题按 f9u 重渲染 + 第 738 局起线上题（r9zw / r9zy 的备注过滤只认到 R41，738~757 局 R42/R43 的线上题漏进训练，n=1 的 test_r9zy_new 就是这个）；738~757 训练、758 起测试；定向重标 54.23（古币机 / 机器头 / 裁决）。
更早：r9zy 数据集（2026-10-19，f9t / g9zt；复制自 build_r9zw.py）：r9zw 全部题按 f9t 重渲染 + 第 751 局起线上题；定向重标 54.20~54.22 + 10-19 卡面机制确认相关的题。
更早：r9zw 数据集（2026-10-18 傍晚，f9s / g9zq；复制自 build_r9zv.py）：r9zv 全部题按 f9s 重渲染 + 第 733 局起线上题；定向重标 g9zq（54.18：最终阵容唯一目标、路线定了不换、缺核心件多刷、太乙第 12 回合前低优先、竞速天赋、满员拍卖、不卖刚上场的人）。
更早：r9zv 数据集（2026-10-18，f9r / g9zo；复制自 build_r9zs.py）：r9zu（= r9zt 的 f9q 版）全部题按 f9r 重渲染 + 第 721 局起线上题；定向重标 g9zo（54.16）的题。
更早：r9zs 数据集（2026-10-17 晚，f9o / g9zm；复制自 build_r9zr.py）：r9zr 全部题按 f9o 重渲染 + 第 697 局起线上题；定向重标 g9zm（54.14）的题。
更早：r9zr 数据集（2026-10-17，f9n / g9zl；复制自 build_r9zq.py）：r9zq 全部题按 f9n 重渲染 + 第 685 局起线上题；定向重标 g9zl 裁决的冲突题。
更早：r9zq 数据集（2026-10-17，f9lz / g9zk；复制自 build_r9zp.py）：r9zp 全部题按 f9lz 重渲染 + 第 674 局起线上题 + 离线 use_timing / sell_hand 题。
更早：r9zp 数据集（2026-10-16 晚，f9ly / g9zj；复制自 build_r9zo.py）：r9zo 全部题按 f9ly 重渲染 + 第 666 局起线上题 + 离线 use_timing / sell_hand 题。
更早：r9zo 数据集（2026-10-16 上午，f9lx / g9zi；复制自 build_r9zn.py）。r9zn 全部题按 f9lx 重渲染 + 第 656 局起（R34）线上题 + 离线 sell_hand 题。
旧说明（r9zn，2026-10-16，f9lw / g9zh）。

- r9zm 全部题按 **f9lw** 重渲染（R33 的 f9lv 修正全部保留：f9lw 只在 f9lv 上叠加）；
- 第 652 局起的线上题（测试 test_r9zn_new）；
- **离线新题**：sell_filler（没人等位时卖不卖占位件）+ 点兵点将的 use_timing（build_dataset.extract_run，SELL_FILLER_QUESTIONS）。
  线上从没出过这两类题，只能从各局 states.jsonl 还原。
重标清单（2026-10-16：先修掉所有已知问题、该重标的重标，再重新训练）：
  新局全部 Laya 题 + 换人题；带 f9lw 新标签的题按类封顶（W_*，见 TAGS）；sell_filler / use_timing 全部（封顶）。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zn.py [--check]
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
import build_r9za as BA  # noqa: E402
import build_r9zm as BM  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

VER = "f9z"
PREV_VER = "f9y"
BASE = "r9zze"
NAME = "r9zzf"
MATCH_RANGE = (815, 900)
TEST_MATCHES = set(range(830, 901))
TEST_SPLIT = "test_r9zzf_new"
OFFLINE_FROM = 600                     # 离线 sell_filler / use_timing 题从这局起（states.jsonl 有棋手等级与场上读数）
LAYA_KINDS = BA.LAYA_KINDS + ("use_timing", "sell_hand")     # sell_filler 停用（g9zj 13）
STUDENT_KEYS = BR.STUDENT_KEYS + ("model",)

# f9ly 新标签 → 重标类别（文本检索；a = f9ly 渲染）
TAGS = (
    ("E_engine_f9z", lambda k, t: "MIN ENGINE" in t or "ROUTE core engine piece" in t),
    ("E_route_backfill", lambda k, t: k == "route"),
    ("E_carry_lottery", lambda k, t: "carry lottery" in t),
    ("E_zhulu_fuel", lambda k, t: "-only tactic card -> no fuel" in t),
    ("E_speedlock", lambda k, t: "only 花木兰/嬴政 -> MISMATCH" in t),
    ("E_special_place", lambda k, t: "needs special placement" in t),
    ("E_missing_shop", lambda k, t: "in shop now: lock keeps it" in t),
    ("E_talent_cond", lambda k, t: k == "talent" and ("conditional:" in t or "LOW VALUE:" in t)),
    ("E_missing_shop", lambda k, t: "in shop now: lock keeps it" in t),
    ("E_special_fit", lambda k, t: "for carry" in t and "MISMATCH" in t),
    ("E_awaken_lv6", lambda k, t: "AWAKENS core" in t),
    ("E_stale_maple", lambda k, t: "(stale:" in t),
    ("E_coin_machine", lambda k, t: "go into 古币机" in t),
)
CAPS = {"E_engine_f9z": (80, 20), "E_route_backfill": (30, 8), "E_carry_lottery": (20, 5), "E_zhulu_fuel": (20, 5), "E_speedlock": (20, 5), "E_special_place": (20, 5), "E_missing_shop": (40, 10), "E_talent_cond": (100, 25), "Z_changed": (150, 40)}


def render(p: dict, ver: str, keys) -> dict | None:
    return BA.render(p, ver, keys)


def tags(a: dict) -> list[str]:
    t = json.dumps([a.get("state"), a["question"]], ensure_ascii=False)
    return [name for name, f in TAGS if f(a["kind"], t)]


def online_new(st) -> list[dict]:
    """第 656 局起的线上题（R34 及以后）。"""
    BM.MATCH_RANGE, BM.TEST_MATCHES, BM.TEST_SPLIT = MATCH_RANGE, TEST_MATCHES, TEST_SPLIT
    BM.NOTE_PREFIXES = ("R29", "R31", "R32", "R34", "R35", "R36", "R37", "R38", "R39", "R40", "R41", "R42", "R43", "R44", "R45", "R46", "R47", "R48", "R49")
    out = BM.new_points(st)
    for p in out:
        p["match"] = p["match"].replace("r9zm_", NAME + "_")
    return out


def offline_points(st) -> list[dict]:
    """sell_filler / 点兵点将 use_timing：各局 states.jsonl 还原（线上没出过这两类题）。"""
    import build_dataset as BD
    BD.SELL_FILLER_QUESTIONS = True      # 点兵点将 use_timing 仍从这条开关出（sell_filler 题 f9lx 线上停用，照出但不进重标）
    BD.SELL_HAND_QUESTIONS = True
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
            if p["kind"] not in ("sell_filler", "use_timing", "sell_hand"):
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
    tg = {r["id"]: tags(r) + (["Z_changed"] if r["id"] in changed else []) for r in rows if r["kind"] in LAYA_KINDS or r["kind"] == "swap"}
    for i, t in tg.items():
        for x in t:
            st[f"tag:{rows_by[i]['split'] == 'train' and 'train' or 'test'}:{x}"] += 1
    tr, te = set(), set()
    per_match = collections.defaultdict(list)   # r9zv：新局线上题每局抽 35 道
    for i in sorted(new_set):
        r = rows_by[i]
        if r.get("source") == "offline" or r["kind"] not in LAYA_KINDS + ("swap",):
            continue
        per_match[r["match"]].append(i)
    for m, ids in per_match.items():
        rng.shuffle(ids)
        for i in ids[:35]:
            (te if rows_by[i]["split"] == TEST_SPLIT else tr).add(i)
    for name, (ctr, cte) in CAPS.items():
        idx_tr = [i for i, t in tg.items() if name in t and rows_by[i]["split"] == "train" and i not in tr]
        idx_te = [i for i, t in tg.items() if name in t and rows_by[i]["split"] != "train" and i not in te]
        tr |= set(_strat(idx_tr, rows_by, ctr, rng))
        te |= set(_strat(idx_te, rows_by, cte, rng))
    sel = collections.Counter()
    for i in tr | te:
        sel[f"{'train' if i in tr else 'test'}:{rows_by[i]['kind']}"] += 1
    up = sorted(i for i, t in tg.items() if any(x.startswith("E_") for x in t) and rows_by[i]["split"] == "train")

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
    (od / "changed_vs_prev.json").write_text(json.dumps(sorted(changed)), encoding="utf-8")
    (od / "teacher_train_ids.json").write_text(json.dumps(sorted(tr)), encoding="utf-8")
    (od / "teacher_test_ids.json").write_text(json.dumps(sorted(te)), encoding="utf-8")
    (od / "upsample_w_ids.json").write_text(json.dumps(up), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
