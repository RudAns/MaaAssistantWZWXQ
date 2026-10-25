"""等级来源账本（2026-09-29，提升水平方案的第 1 步：查清等级为什么滚不起来）。

  python tools/analyze_level_sources.py live_xxx [live_yyy ...]   # 不给参数 = 最近 15 局

对每局、每回合，拆主C（和全场）等级的变化：
  prep  = 本回合准备阶段内（第一帧 → 最后一帧）涨的：合成 / 加等级牌 / 枫叶舞 / 夺取 —— 我们的操作
  auto  = 上回合最后一帧 → 本回合第一帧涨的：战斗中与回合交替时的被动效果（牺牲、整备、开团、古币…）—— 引擎
同时列出本回合执行过的、目标是主C 的用牌动作，以及合成次数。
只用 states.jsonl 的 "board"（rules.board_facts，读得到的等级；读不到记 ?）。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs_default(n=15):
    ds = sorted(glob.glob(os.path.join(ROOT, "data", "logs", "live_*")), key=os.path.getmtime)
    return [d for d in ds if os.path.exists(os.path.join(d, "states.jsonl"))][-n - 1:-1]


def analyze(d: str) -> dict:
    per = defaultdict(lambda: {"first": None, "last": None, "carry": None, "cards": [], "merges": 0, "maple": 0})
    for line in open(os.path.join(d, "states.jsonl"), encoding="utf-8"):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        s = x.get("state") or {}
        r = s.get("round_no") or 0
        if not r:
            continue
        b = x.get("board")
        if isinstance(b, dict) and b and s.get("phase") == "prepare":
            lv = {n: (v or {}).get("lv") for n, v in b.items()}
            if any(isinstance(v, int) for v in lv.values()):
                if per[r]["first"] is None:
                    per[r]["first"] = lv
                per[r]["last"] = lv
            if x.get("carry"):
                per[r]["carry"] = x["carry"]
        if x.get("executed") and x.get("actions"):
            a = x["actions"][0]
            if a.get("type") == "drag_card":
                if a.get("merge"):
                    per[r]["merges"] += 1
                elif a.get("piece") == "枫叶舞":
                    per[r]["maple"] += 1
                elif a.get("kind") in ("效果牌", "天赋牌") and a.get("target"):
                    per[r]["cards"].append((a.get("piece"), a.get("target")))
    rows, prev_last = [], None
    for r in sorted(per):
        p = per[r]
        c = p["carry"]
        f, l = p["first"] or {}, p["last"] or {}
        cf, cl = f.get(c), l.get(c)
        cp = prev_last.get(c) if prev_last else None
        tot = lambda m: sum(v for v in (m or {}).values() if isinstance(v, int))  # noqa: E731
        rows.append({"r": r, "carry": c, "lv": cl,
                     "prep": (cl - cf) if isinstance(cl, int) and isinstance(cf, int) else None,
                     "auto": (cf - cp) if isinstance(cf, int) and isinstance(cp, int) else None,
                     "board": tot(l), "board_auto": (tot(f) - tot(prev_last)) if prev_last else None,
                     "to_carry": sum(1 for _, t in p["cards"] if t == c), "cards": len(p["cards"]),
                     "merges": p["merges"], "maple": p["maple"]})
        if p["last"]:
            prev_last = p["last"]
    return {"run": os.path.basename(d), "rows": rows}


def main() -> int:
    ds = [os.path.join(ROOT, "data", "logs", a) if not os.path.isdir(a) else a for a in sys.argv[1:]] or runs_default()
    agg = defaultdict(lambda: [0, 0, 0])      # r -> [prep 合计, auto 合计, 局数]
    for d in ds:
        res = analyze(d)
        print(f"== {res['run']}")
        for w in res["rows"]:
            print(f"  R{w['r']:>2} 主C {w['carry'] or '-':<5} lv {w['lv'] if w['lv'] is not None else '?':>3} | 准备期+{w['prep'] if w['prep'] is not None else '?':<3} "
                  f"回合间+{w['auto'] if w['auto'] is not None else '?':<3} | 全场 {w['board']:>3} (回合间+{w['board_auto'] if w['board_auto'] is not None else '?'}) "
                  f"| 用牌 {w['cards']}（给主C {w['to_carry']}） 合成 {w['merges']} 枫叶舞 {w['maple']}")
            if w["prep"] is not None and w["auto"] is not None:
                a = agg[w["r"]]
                a[0] += w["prep"]; a[1] += w["auto"]; a[2] += 1
    print("\n== 平均（有读数的局）：回合 / 主C 准备期涨 / 回合间涨")
    for r in sorted(agg):
        p, a, n = agg[r]
        print(f"  R{r:>2}: 准备期 +{p / n:.1f}  回合间 +{a / n:.1f}  (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
