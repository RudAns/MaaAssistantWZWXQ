"""读 flow_compare 的结果，找"三档战斗情景下都站得住"的转型：某个局面里，某流派在低 / 中 / 高三档的第 14 回合主C 等级
都比灵活打法高（且差距 ≥ margin）才算。只比养成（不估伤害）。

  .venv/Scripts/python.exe -X utf8 tools/sim/flow_robust.py <flowcmp_R4.json> [<flowcmp_R6.json> …] --margin 5
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.sim.flows import FLOWS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--margin", type=float, default=5.0)
    a = ap.parse_args()
    for fp in a.files:
        rows = json.loads(Path(fp).read_text(encoding="utf-8"))
        by = collections.defaultdict(dict)                 # run -> scen -> row
        for r in rows:
            by[r["run"]][r["scen"]] = r
        robust, any_win, n = collections.Counter(), collections.Counter(), 0
        examples = []
        for run, sc in by.items():
            if len(sc) < 3:
                continue
            n += 1
            keys = [k for k in next(iter(sc.values()))["res"] if k != "普通打法"]
            for k in keys:
                d = [sc[s]["res"][k]["lv14"] - sc[s]["res"]["普通打法"]["lv14"] for s in ("低", "中", "高")]
                if any(x >= a.margin for x in d):
                    any_win[k] += 1
                if all(x >= a.margin for x in d):
                    robust[k] += 1
                    r0 = sc["中"]
                    examples.append((run, FLOWS[k].name if k in FLOWS else k, r0["board"], r0["lord"], d))
        print(f"\n== {Path(fp).name}：{n} 个局面")
        print(f"  三档都比灵活打法高 ≥{a.margin:g} 级的（流派：局面数）：" + ("、".join(
            f"{FLOWS[k].name if k in FLOWS else k} {v}" for k, v in robust.most_common()) or "没有"))
        print("  至少一档高的（前 8）：" + "、".join(f"{FLOWS[k].name if k in FLOWS else k} {v}" for k, v in any_win.most_common(8)))
        for run, name, board, lord, d in examples[:8]:
            print(f"    {run} 棋手{lord} 场上{board[:6]} → {name}（低/中/高 +{d[0]:.0f}/+{d[1]:.0f}/+{d[2]:.0f}）")


if __name__ == "__main__":
    main()
