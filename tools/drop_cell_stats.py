"""统计每个上场落点的历史成败，写 data/drop_cell_stats.json（规则引擎挑落点时当先验）。

为什么（2026-09-23 第十局）：辅助"贴着主C 放"会挑离主C 最近的格子，连着两格都拖不进去。
全历史账本一统计，有的格子几乎从没成功过：(917,570) 42 次上场决策失败 41 次、(555,569) 21 次失败 20 次、
(955,502) 失败 83%。多半是最下面一行的落点为了避开手牌被压到 y<=545，正好落进上面那一格（常有人）。
这种"格子本身就坏"的信息单局里学不到（_cell_fails 每局清零），只能从历史里攒。

数据：data/logs/live_*/ledger.jsonl 里的「上场」（决策时记，含落点）和「拖拽失败」（回执）。
失败里混着"格子被占"的情况，所以只当**排序先验**，不当硬排除。

用法：.venv/Scripts/python.exe -X utf8 tools/drop_cell_stats.py
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT = PROJECT_DIR / "data" / "drop_cell_stats.json"


def main() -> int:
    tries: collections.Counter = collections.Counter()
    fails: collections.Counter = collections.Counter()
    # 只看最近 N 天（默认 7）：老数据里混着"最下一行落点压到手牌上"（09-23 才修）的失败，会把好格子判成坏格子
    # （2026-10-10：(917,570) 失败率 0.95，太乙真人因此没贴主C 守约）
    import time as _t
    days = float(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 7.0
    since = _t.time() - days * 86400
    for f in glob.glob(str(PROJECT_DIR / "data/logs/live_*/ledger.jsonl")):
        try:
            if int(Path(f).parent.name.split("_")[1]) < since:
                continue
        except (IndexError, ValueError):
            continue
        for line in open(f, encoding="utf-8"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "上场" and e.get("位置"):
                tries[e["位置"]] += 1
            elif e.get("event") == "拖拽失败" and e.get("落点"):
                fails[e["落点"]] += 1
    cells = {}
    for c, n in tries.items():
        f = min(fails.get(c, 0), n)
        # 拉普拉斯平滑：样本少的格子别被一两次失败判死
        cells[c] = {"tries": n, "fails": f, "fail_rate": round((f + 1) / (n + 2), 3)}
    OUT.write_text(json.dumps({"note": "上场落点历史失败率（含被占），只当排序先验",
                               "cells": cells}, ensure_ascii=False, indent=1), encoding="utf-8")
    for c, v in sorted(cells.items(), key=lambda kv: -kv[1]["fail_rate"])[:10]:
        print(c, v)
    print(f"写入 {OUT}（{len(cells)} 格）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
