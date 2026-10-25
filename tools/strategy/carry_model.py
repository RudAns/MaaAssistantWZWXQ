"""**主C 量化评分模型**：游戏机制 × 商店品阶概率 × 数值（2026-09-24：根据游戏玩法和商店刷新的各种概率，
结合数值演算一个评分，再让 Laya 学习这些评分或由此产生的选择数据）。

  python tools/strategy/carry_model.py --sims 4000          # 算表 -> data/strategy/carry_model.json + 报告

三部分：
1. **到手概率（蒙特卡洛）**：按实测商店品阶概率（meta.TIER_ODDS）、公共牌库（每阶英雄数 POOL_HEROES × 每人张数 POOL_COPIES）、
   每回合收入（meta.energy_income）、棋手经验（每回合 +1、拍卖回合 +3；升级费 = 还差的经验，1 能量 1 点）、禁用一个阵营、
   7 个对手也从牌库里拿牌，模拟"锁定某个英雄当主C、按某种升级节奏打"到第 R 回合**拿到 ≥1 张的概率**和**期望张数**。
2. **强度指数**：主 agent 读技能的评分（data/hero_carry_reviewed.json，0~10）× 从技能文字里解析出的**成长倍率**
   （10/40/100 级强化里"提升至 X%"相对基础系数的倍数，以及"数量/范围/全场"类强化）。
3. **综合分**：主C 价值 = 强度 × 成型概率（前期看第 6 回合、后期看第 10 回合），并用官网阵容里玩家实际选主C 的频率检验排序。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.decision import meta  # noqa: E402

XP_NEED = {1: 7, 2: 13, 3: 17, 4: 23, 5: 27}          # 升级所需经验（经验条 N/M 的 M，实测）
SHOP_SLOTS = 4
BUY_COST = 3
REFRESH_COST = 2
OPPONENTS = 7
OPP_BUYS_PER_ROUND = 2                                   # 每个对手每回合从牌库拿走几张（估计值，可调）
ROUNDS = 14

# 升级节奏（第几回合达到几级）：standard = 规则里的 LEVEL_TARGET；fast = 攻略"速 3/速 4"；roll3 = 停 3 级搜牌（遗志流）
PLANS = {
    "standard": {2: 2, 4: 3, 7: 4, 10: 5, 13: 6},
    "fast": {2: 2, 3: 3, 5: 4, 8: 5, 11: 6},
    "roll3": {2: 2, 4: 3, 10: 4, 11: 5, 13: 6},
}


def load_heroes() -> dict[str, dict]:
    cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    return {n: c for n, c in cards.items() if c.get("category") == "英雄牌"}


def simulate(target: str, heroes: dict, plan: dict[int, int], sims: int, seed: int = 0) -> dict:
    """锁定 target 当主C：每回合先按节奏升级，剩下的钱刷新找它、见到就买。返回各回合 P(≥1 张) 与期望张数。"""
    rng = random.Random(seed)
    by_tier = collections.defaultdict(list)
    for n, c in heroes.items():
        by_tier[c.get("quality", 1)].append(n)
    factions = sorted({c.get("faction") for c in heroes.values() if c.get("faction") and c.get("faction") != "无阵营"})
    tq = heroes[target].get("quality", 1)
    got_by_round = [[0] * sims for _ in range(ROUNDS + 1)]
    for s in range(sims):
        banned = rng.choice(factions)
        if heroes[target].get("faction") == banned:
            banned = rng.choice([f for f in factions if f != banned])   # 被禁就不可能选它当主C，换一个被禁阵营
        pool = {n: meta.POOL_COPIES[c.get("quality", 1)] for n, c in heroes.items() if c.get("faction") != banned}
        gold = xp = 0
        level = 1
        copies = 0
        for r in range(1, ROUNDS + 1):
            gold += meta.energy_income(r)
            # 回合开始的自然经验（上回合战斗 +1、拍卖回合 +3）
            if r > 1:
                xp += meta.lord_xp_gain(r - 1)
            while level < 6 and xp >= XP_NEED.get(level, 99):
                xp -= XP_NEED[level]
                level += 1
            # 按节奏买经验升级
            want = max([lv for rr, lv in plan.items() if rr <= r] or [1])
            while level < min(want, 6):
                cost = XP_NEED[level] - xp
                if gold < cost:
                    break
                gold -= cost
                xp = 0
                level += 1
            odds = meta.TIER_ODDS.get(min(level, 6), {1: 1.0})
            # 对手拿牌（按同等级的品阶分布从牌库随机拿）
            for _ in range(OPPONENTS * OPP_BUYS_PER_ROUND):
                t = _draw_tier(rng, odds)
                cands = [n for n in by_tier[t] if pool.get(n, 0) > 0]
                if cands:
                    pool[rng.choice(cands)] -= 1
            # 自己：先看一次免费的回合开店，再用剩下的钱刷新（保留买一张的钱）
            first = True
            while first or gold >= REFRESH_COST + BUY_COST:
                if not first:
                    gold -= REFRESH_COST
                first = False
                for _ in range(SHOP_SLOTS):
                    t = _draw_tier(rng, odds)
                    cands = [n for n in by_tier[t] for _k in range(pool.get(n, 0))]
                    if not cands:
                        continue
                    h = rng.choice(cands)
                    if h == target and gold >= BUY_COST:
                        gold -= BUY_COST
                        pool[h] -= 1
                        copies += 1
                if tq > meta.MAX_TIER_BY_LEVEL.get(min(level, 6), 1):
                    break                       # 当前等级根本刷不出它：不白刷
            got_by_round[r][s] = copies
    out = {}
    for r in range(1, ROUNDS + 1):
        vals = got_by_round[r]
        out[r] = {"p_any": round(sum(1 for v in vals if v > 0) / sims, 3),
                  "e_copies": round(sum(vals) / sims, 2)}
    return out


def _draw_tier(rng, odds: dict[int, float]) -> int:
    x = rng.random()
    acc = 0.0
    for t, p in sorted(odds.items()):
        acc += p
        if x <= acc:
            return t
    return max(odds)


PCT_RE = re.compile(r"(\d+)%(?:物理|法术)攻击力")
UP_RE = re.compile(r"提升至(?:\d+\+)?(\d+)%")


def growth(c: dict) -> float:
    """从技能文字解析成长倍率：100 级时的伤害系数 / 基础系数，另加"数量/范围/全场"类强化。"""
    sk = (c.get("skills") or [{}])[0]
    base = [int(x) for x in PCT_RE.findall(sk.get("desc") or "")]
    b = max(base) if base else 100
    top = b
    extra = 1.0
    for e in sk.get("enhance") or []:
        d = e.get("desc") or ""
        ups = [int(x) for x in UP_RE.findall(d)]
        if ups:
            top = max(top, max(ups))
        if re.search(r"全场|所有敌人|整个战场|全体", d):
            extra *= 1.6
        elif re.search(r"数量|增加至\d+|\+\d+|额外|连续|范围", d):
            extra *= 1.15
    return round(min(4.0, (top / b) * extra), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--min-score", type=int, default=5, help="只模拟技能评分 ≥ 这个数的英雄（主C 候选）")
    args = ap.parse_args()
    heroes = load_heroes()
    reviewed = json.loads((ROOT / "data" / "hero_carry_reviewed.json").read_text(encoding="utf-8"))
    lineups = json.loads((ROOT / "data" / "lineups.json").read_text(encoding="utf-8"))
    L = list(lineups["lineups"].values()) if isinstance(lineups.get("lineups"), dict) else lineups["lineups"]
    carry_use = collections.Counter()
    ali_use = collections.Counter()
    for x in L:
        carry_use[x.get("carry")] += int(x.get("use_num") or 0)
        if "阿离" in str(x.get("lord_names") or ""):
            ali_use[x.get("carry")] += int(x.get("use_num") or 0)
    tot, tot_ali = sum(carry_use.values()) or 1, sum(ali_use.values()) or 1
    res = {}
    cands = [n for n, v in reviewed.items() if v["carry_score"] >= args.min_score]
    for i, n in enumerate(sorted(cands)):
        c = heroes[n]
        sims = {p: simulate(n, heroes, plan, args.sims, seed=i) for p, plan in PLANS.items()}
        g = growth(c)
        power = reviewed[n]["carry_score"] * g
        best_plan = max(PLANS, key=lambda p: sims[p][8]["p_any"])
        early = sims["standard"][6]["p_any"]
        mid = sims["standard"][10]["p_any"]
        res[n] = {"quality": c.get("quality"), "faction": c.get("faction"), "skill_score": reviewed[n]["carry_score"],
                  "growth": g, "power": round(power, 1),
                  "p_by_r6": early, "p_by_r10": mid, "e_copies_r10": sims["standard"][10]["e_copies"],
                  "best_plan_for_r8": best_plan, "sims": sims,
                  "value_early": round(power * early, 2), "value_mid": round(power * mid, 2),
                  "lineup_carry_share": round(carry_use.get(n, 0) / tot, 4),
                  "ali_carry_share": round(ali_use.get(n, 0) / tot_ali, 4)}
        print(f"{n:6s} {c.get('quality')}阶 技能{reviewed[n]['carry_score']} 成长×{g} 强度{power:5.1f} "
              f"| R6 到手 {early:.2f} R10 {mid:.2f} | 前期值 {res[n]['value_early']:5.1f} 中期值 {res[n]['value_mid']:5.1f} "
              f"| 阵容主C 占比 {res[n]['lineup_carry_share']:.3f} 阿离 {res[n]['ali_carry_share']:.3f}")
    out = ROOT / "data" / "strategy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "carry_model.json").write_text(json.dumps(
        {"params": {"sims": args.sims, "opponents": OPPONENTS, "opp_buys_per_round": OPP_BUYS_PER_ROUND,
                    "plans": PLANS, "xp_need": XP_NEED}, "heroes": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    # 检验：模型排序 vs 玩家实际选主C（Spearman）
    for key, name in (("lineup_carry_share", "全部阵容"), ("ali_carry_share", "阿离阵容")):
        for vk in ("value_early", "value_mid", "skill_score"):
            print(f"Spearman({vk} vs {name}主C 占比) = {_spearman([res[n][vk] for n in res], [res[n][key] for n in res]):.3f}")
    return 0


def _spearman(a, b) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** .5
    vb = sum((y - mb) ** 2 for y in rb) ** .5
    return cov / (va * vb) if va and vb else 0.0


if __name__ == "__main__":
    sys.exit(main())
