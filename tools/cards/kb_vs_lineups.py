"""知识库一致性检查：官网推荐阵容（data/lineups.json，448 套）能不能被知识库的"套路"解释。

  python tools/cards/kb_vs_lineups.py

每套阵容取 英雄 + 推荐效果牌 + 推荐天赋，查知识库条目的 routes；
"主套路覆盖率" = 阵容里属于出现最多的那个套路的卡占比。覆盖率低的热门阵容 = 知识库可能漏了它依赖的机制，逐个人工看。
只看"机制套路"，去掉 找牌/防守/输出装/经济/单核 这类通用标签（每套阵容都有，不说明问题）。
"""
from __future__ import annotations

import ast
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERIC = {"找牌", "防守", "输出装", "经济", "单核", "召唤", "高阶流", "刷新", "人口", "棋手经验", "卖人", "装备流", "其他棋手"}


def lst(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x) if x else []
    except (ValueError, SyntaxError):
        return []


def main() -> int:
    kb = json.loads((ROOT / "data" / "card_kb.json").read_text(encoding="utf-8"))["entries"]
    L = json.loads((ROOT / "data" / "lineups.json").read_text(encoding="utf-8"))["lineups"]
    rows, unknown = [], collections.Counter()
    route_heat = collections.Counter()
    for x in L:
        use = int(x.get("use_num") or 0)
        cards = [h["name"] for h in lst(x.get("heroes")) if isinstance(h, dict)] + lst(x.get("effect_cards")) + lst(x.get("talent_cards"))
        rc = collections.Counter()
        for c in cards:
            if c not in kb:
                unknown[c] += use
                continue
            for r in set(kb[c]["routes"]) - GENERIC:
                rc[r] += 1
        if not cards:
            continue
        top, n = rc.most_common(1)[0] if rc else ("-", 0)
        second = rc.most_common(2)[1] if len(rc) > 1 else ("-", 0)
        rows.append((use, x.get("name"), str(x.get("lord_names")), len(cards), top, n / len(cards), second[0], x.get("carry")))
        for r, k in rc.items():
            if k >= 2:
                route_heat[r] += use
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows) or 1
    cov = sum(r[0] * r[5] for r in rows) / tot
    print(f"{len(rows)} 套阵容，热度加权的主套路覆盖率 {cov:.2f}")
    print("\n热度前 25 套：热度 | 阵容 | 棋手 | 卡数 | 主套路(占比) | 第二套路 | 主C")
    for use, nm, lords, n, top, share, sec, carry in rows[:25]:
        print(f"  {use:>8} | {nm} | {lords} | {n} | {top}({share:.0%}) | {sec} | {carry}")
    print("\n覆盖率 < 30% 的热门阵容（热度 > 20 万）——可能是知识库没解释的机制：")
    for use, nm, lords, n, top, share, sec, carry in rows:
        if use > 200000 and share < .3:
            print(f"  {use:>8} | {nm} | {lords} | 主套路 {top}({share:.0%})")
    print("\n套路热度（阵容里 ≥2 张该套路的卡，按使用次数加权）：")
    for r, u in route_heat.most_common():
        print(f"  {r}: {u / tot:.1%}")
    print("\n阵容里出现但知识库没有的卡：", unknown.most_common(15))
    return 0


if __name__ == "__main__":
    sys.exit(main())
