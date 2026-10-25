"""经济推演器核对（2026-10-16 午：推演器只用于辅助出题 → 写进题面前先核对它的预测准不准）。

拿 eco_roll_eval 的输出（每道经济题各选项的推演结果），和这道题之后实际发生的事比：
  1. 等级：推演"4 回合后棋手等级"（按实际走的那条路：本回合实际升了级 → levelup，否则 → roll_0~roll_all 的均值）vs 实际第 r+4 回合的等级；
  2. 推演分差能不能分辨：本回合实际升级 vs 没升级两组，实际 4 回合后的等级差，和推演的 levelup − roll 差同号吗。
  .venv/Scripts/python.exe -X utf8 tools/sim/eco_sim_validate.py data/sim/eco_roll_seed1.jsonl
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def levels_by_round(run):
    out = {}
    f = ROOT / "data/logs" / run / "states.jsonl"
    for l in f.open(encoding="utf-8"):
        try:
            s = json.loads(l)["state"]
        except Exception:  # noqa: BLE001
            continue
        r, lv = s.get("round_no") or 0, s.get("level") or 0
        if 0 < r < 40 and 0 < lv <= 6:
            out[r] = max(out.get(r, 0), lv)        # 一回合里看到的最高等级（回合末）
    return out


def owned_timeline(run):
    """[(round, ts, 拥有的英雄名集合)]：场上（rerender.board_timeline 逐帧合并后的场上）+ 手牌。"""
    sys.path.insert(0, str(ROOT / "tools" / "laya")); sys.path.insert(0, str(ROOT))
    import rerender as RR
    tl = RR.board_timeline(run)
    out, i = [], 0
    for l in (ROOT / "data/logs" / run / "states.jsonl").open(encoding="utf-8"):
        try:
            d = json.loads(l)
        except Exception:  # noqa: BLE001
            continue
        s, ts = d["state"], d["ts"]
        while i + 1 < len(tl) and tl[i + 1][0] <= ts:
            i += 1
        board = set((tl[i][1] or {}).keys()) if tl and tl[i][0] <= ts else set()
        out.append((s.get("round_no") or 0, ts, board | set(s.get("hand") or [])))
    return out


def acquisitions(rows):
    """推演"4 回合内拿到几个缺的人" vs 实际（出题时缺、之后 4 回合内出现在场上或手里）。"""
    cache, pairs = {}, []
    for r in rows:
        if "acq" not in r or not r.get("missing"):
            continue
        run, rd, ts = r["run"], r["round"], r.get("ts") or 0
        if run not in cache:
            cache[run] = owned_timeline(run)
        tl = cache[run]
        if not any(x[0] >= rd + 4 for x in tl):
            continue
        before = set().union(*[o for (q, t, o) in tl if t <= ts and q == rd] or [set()])
        after = set().union(*[o for (q, t, o) in tl if t > ts and q <= rd + 4] or [set()])
        miss = [m for m in r["missing"] if m not in before]
        got = sum(1 for m in miss if m in after)
        pred = st.mean(v for k, v in r["acq"].items() if k.startswith("roll"))   # 推演各刷新选项的均值（实际走法多数是刷新 / 停手）
        pairs.append((pred, got, rd, len(miss)))
    if not pairs:
        print("没有可比的题"); return
    e = [a - b for a, b, _, _ in pairs]
    print(f"拿缺的人：题数 {len(pairs)}，推演均值 {st.mean(a for a, *_ in pairs):.2f}，实际均值 {st.mean(b for _, b, *_ in pairs):.2f}，"
          f"偏差 {st.mean(e):+.2f}，平均绝对误差 {st.mean(abs(x) for x in e):.2f}")
    for lo, hi in ((1, 4), (5, 8), (9, 99)):
        sub = [(a, b) for a, b, rd, _ in pairs if lo <= rd <= hi]
        if sub:
            print(f"  第 {lo}~{hi} 回合：n={len(sub)} 推演 {st.mean(a for a, _ in sub):.2f} 实际 {st.mean(b for _, b in sub):.2f}")
    a = [p[0] for p in pairs]; b = [p[1] for p in pairs]
    ma, mb = st.mean(a), st.mean(b)
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    print(f"  相关系数 {sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else float('nan'):.2f}")


def main():
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    cache, pairs, grp = {}, [], defaultdict(list)
    for r in rows:
        run, rd = r["run"], r["round"]
        if run not in cache:
            cache[run] = levels_by_round(run)
        L = cache[run]
        if rd not in L or rd + 4 not in L or (rd - 1) not in L and rd > 1:
            continue
        up_now = L[rd] > (r.get("level") or L[rd])       # 本回合实际升级了
        rolls = [v for k, v in r["lv_at_h"].items() if k.startswith("roll")]
        pred = r["lv_at_h"]["levelup"] if up_now else st.mean(rolls)
        pairs.append((pred, L[rd + 4], rd))
        grp[up_now].append(L[rd + 4] - (r.get("level") or 0))
    err = [a - b for a, b, _ in pairs]
    print(f"题数 {len(pairs)}；4 回合后等级 推演 − 实际：均值 {st.mean(err):+.2f}，平均绝对误差 {st.mean(abs(e) for e in err):.2f}")
    for lo, hi in ((1, 4), (5, 8), (9, 99)):
        e = [a - b for a, b, rd in pairs if lo <= rd <= hi]
        if e:
            print(f"  第 {lo}~{hi} 回合：n={len(e)} 偏差 {st.mean(e):+.2f} 绝对误差 {st.mean(abs(x) for x in e):.2f}")
    if len(pairs) > 2:
        a = [p for p, _, _ in pairs]; b = [q for _, q, _ in pairs]
        ma, mb = st.mean(a), st.mean(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        print(f"相关系数 {cov / den if den else float('nan'):.2f}")
    for k, v in grp.items():
        print(f"本回合{'升了级' if k else '没升级'}：n={len(v)}，4 回合后比出题时高 {st.mean(v):.2f} 级")
    acquisitions(rows)


if __name__ == "__main__":
    main()
