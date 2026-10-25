"""实测：每回合结束时场上**最高英雄等级**、**英雄总等级**，按棋手等级分组看每回合涨多少。

  python tools/strategy/measure_growth.py

读 data/logs/live_*/states.jsonl 里的准备阶段帧；每个 (运行, 回合) 取该回合最后 10 帧的中位数（徽章 OCR 会读错）。
只用 2026-09-23 之后的运行（徽章判据改过、等级读数才可用）。
"""
from __future__ import annotations
import collections, glob, json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
per = {}
for f in sorted(glob.glob(str(ROOT / "data/logs/live_*/states.jsonl"))):
    run = Path(f).parent.name
    if int(run.split("_")[1]) < 1790100000:
        continue
    by_round = collections.defaultdict(list)
    for line in open(f, encoding="utf-8", errors="ignore"):
        try: st = json.loads(line)["state"]
        except (ValueError, KeyError): continue
        if st.get("phase") != "prepare" or not st.get("round_no"): continue
        lv = [v for v in (st.get("unit_levels") or {}).values() if v and v < 1000]
        if not lv: continue
        by_round[st["round_no"]].append((st.get("level") or 0, max(lv), sum(lv), len(lv)))
    rows = {}
    for r, v in by_round.items():
        v = v[-10:]
        rows[r] = tuple(statistics.median(x[i] for x in v) for i in range(4))
    if len(rows) >= 5:
        per[run] = rows
print(f"{len(per)} 局")
print("回合  棋手等级  最高等级(中位)  总等级(中位)  读到单位数   样本")
for r in range(1, 16):
    xs = [p[r] for p in per.values() if r in p]
    if len(xs) < 3: continue
    print(f"{r:3d}   {statistics.median(x[0] for x in xs):5.1f}   {statistics.median(x[1] for x in xs):8.1f}   "
          f"{statistics.median(x[2] for x in xs):8.1f}   {statistics.median(x[3] for x in xs):6.1f}   {len(xs)}")
# 每回合最高等级增量 vs 当回合棋手等级
gain = collections.defaultdict(list)
for p in per.values():
    for r in p:
        if r + 1 in p:
            gain[int(p[r][0])].append(p[r + 1][1] - p[r][1])
print("\n棋手等级 -> 下回合最高英雄等级增量（中位 / 均值 / n）")
for lv in sorted(gain):
    g = gain[lv]
    print(f"  {lv}: {statistics.median(g):5.1f} / {statistics.mean(g):5.1f} / {len(g)}")
