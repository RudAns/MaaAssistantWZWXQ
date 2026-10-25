"""商店的实测分布（2026-10-14）：推演器按它出商店（只用游戏自己的随机，不含我们的选择）。

  .venv/Scripts/python.exe -X utf8 tools/engine/build_shop_stats.py  → data/engine/shop_stats.json
输出（按棋手等级）：
  tier_odds[lv]   商店英雄格的品阶占比（实测：2 级是 60/40，不是攻略的 50/50）
  nonhero[lv]     非英雄格（效果牌 / 天赋牌 / 装备牌）的占比
  cards[lv]       非英雄格出现过的牌及次数；cost[牌] 价格中位数
  opp_take[tier][round]  对手持有的张数（每个英雄；= 总张数 − 详情面板"剩余数量" − 我方持有，只用没觉醒的单位，噪声大，只作量级）
另外核对：被禁派系的英雄在商店里出现过几次（6 万个英雄格里 4 次，基本是读错 → 禁用 = 不进牌库）。
"""
import collections
import glob
import json
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.decision import meta  # noqa: E402
from agent.sim.engine import FACTION_ALIAS, category, faction, quality  # noqa: E402


def main() -> int:
    files = sorted(glob.glob(str(ROOT / "data/logs/live_*/states.jsonl")))[-400:]
    tiers = collections.defaultdict(collections.Counter)
    nonhero = collections.defaultdict(lambda: [0, 0])
    cards = collections.defaultdict(collections.Counter)
    cost = collections.defaultdict(list)
    opp = collections.defaultdict(list)
    banned_hits, hero_slots, seen = 0, 0, set()
    for path in files:
        for l in open(path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            st = r.get("state") or {}
            if st.get("phase") != "prepare":
                continue
            R, lv = st.get("round_no"), st.get("level")
            for n, b in (r.get("board") or {}).items():          # 对手持有张数（只作量级）
                pl, ak = b.get("pool_left"), b.get("awk_left")
                if pl is None or not R or ak in (None, 0):
                    continue
                q = quality(n)
                tot = meta.POOL_COPIES.get(q)
                if tot and isinstance(pl, (int, float)):
                    opp[(q, R)].append(tot - pl - (1 + 3 - ak))
            if not st.get("shop_open") or not lv:
                continue
            shop = [x for x in (st.get("shop") or []) if isinstance(x, dict)]
            names = tuple(x.get("piece") for x in shop)
            key = (path, R, names)
            if key in seen:
                continue
            seen.add(key)
            ban = st.get("banned_faction")
            ban = FACTION_ALIAS.get(ban, ban) if ban else ""
            for x in shop:
                n = x.get("piece")
                c = category(n) if n else ""
                if not c:
                    continue
                nonhero[lv][1] += 1
                if c == "英雄牌":
                    tiers[lv][quality(n)] += 1
                    hero_slots += 1
                    banned_hits += bool(ban) and faction(n) == ban
                else:
                    nonhero[lv][0] += 1
                    cards[lv][n] += 1
                    if isinstance(x.get("cost"), (int, float)):
                        cost[n].append(x["cost"])
    out = {
        "n_shops": len(seen),
        "banned_check": {"banned_faction_hero_slots": banned_hits, "hero_slots": hero_slots},
        "tier_odds": {lv: {q: round(v / sum(c.values()), 4) for q, v in sorted(c.items()) if v / sum(c.values()) >= 0.003}
                      for lv, c in sorted(tiers.items())},
        "nonhero": {lv: round(a / max(1, b), 4) for lv, (a, b) in sorted(nonhero.items())},
        "cards": {lv: dict(c.most_common()) for lv, c in sorted(cards.items())},
        "cost": {n: S.median(v) for n, v in cost.items()},
        "opp_take": {q: {R: round(max(0.0, S.mean(opp[(q, R)])), 2) for R in range(1, 20) if len(opp[(q, R)]) >= 20}
                     for q in range(1, 6)},
    }
    (ROOT / "data/engine/shop_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("商店", out["n_shops"], "次；被禁派系英雄出现", banned_hits, "/", hero_slots)
    for lv in out["tier_odds"]:
        print(f"  棋手 {lv}：品阶 {out['tier_odds'][lv]}  非英雄格 {out['nonhero'].get(lv)}  非英雄种类 {len(out['cards'].get(lv, {}))}")
    print("  对手持有（每个英雄）", out["opp_take"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
