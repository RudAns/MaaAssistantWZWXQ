"""实测：往场上同名英雄身上**合成**一张，英雄等级涨多少；按卡牌品阶、觉醒前后分组。

  python tools/strategy/measure_merge.py

做法：states.jsonl 里执行过的 drag_card（英雄牌）且落点那一格之前就有等级读数（= 合成，不是上场），
取拖之前 3 秒内最后一次读数 L0、拖之后 1~12 秒内该格读数的中位数 L1，记 ΔL = L1 - L0。
徽章 OCR 会读错，所以只报中位数，并丢掉 ΔL<0 或 >60 的。
"""
from __future__ import annotations
import collections, glob, json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
cards = json.loads((ROOT / "data/cards.json").read_text(encoding="utf-8"))["cards"]
def ck(p): return f"{int(p[0])},{int(p[1])}"
out = collections.defaultdict(list)
for f in glob.glob(str(ROOT / "data/logs/live_*/states.jsonl")):
    rows = []
    for line in open(f, encoding="utf-8", errors="ignore"):
        try: rows.append(json.loads(line))
        except ValueError: pass
    for i, r in enumerate(rows):
        if not r.get("executed"): continue
        for a in r.get("actions") or []:
            if a.get("type") != "drag_card" or cards.get(a.get("piece"), {}).get("category") != "英雄牌": continue
            dst = a.get("dst")
            if not dst: continue
            # 落点可能被 board_drop 压到 y<=545，按最近的读数格匹配（半径 60）
            def lvl_at(st):
                best = None
                for k, v in (st.get("unit_levels") or {}).items():
                    x, y = map(int, k.split(","))
                    d = abs(x - dst[0]) + abs(y - dst[1])
                    if d < 60 and v and (best is None or d < best[0]): best = (d, v)
                return best[1] if best else None
            t0 = r["ts"]; l0 = None
            for j in range(i, max(-1, i - 30), -1):
                if t0 - rows[j]["ts"] > 3: break
                v = lvl_at(rows[j]["state"])
                if v: l0 = v; break
            if not l0: continue
            after = []
            for j in range(i + 1, min(len(rows), i + 60)):
                dt = rows[j]["ts"] - t0
                if dt > 12: break
                if dt >= 1:
                    v = lvl_at(rows[j]["state"])
                    if v: after.append(v)
            if len(after) < 2: continue
            d = statistics.median(after) - l0
            if 0 <= d <= 60:
                q = cards[a["piece"]].get("quality")
                out[q].append((d, l0, a["piece"]))
allv = [d for v in out.values() for d, *_ in v]
print(f"合成样本 {len(allv)} 个，ΔL 中位数 {statistics.median(allv) if allv else None}")
for q in sorted(out):
    ds = [d for d, *_ in out[q]]
    print(f"  {q} 阶: n={len(ds)} 中位 {statistics.median(ds)} 均值 {statistics.mean(ds):.1f}  分布 {collections.Counter(int(x) for x in ds).most_common(6)}")
