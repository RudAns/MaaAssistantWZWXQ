"""商店概率：一次看店见到某英雄、刷新 N 次、凑齐一组英雄的回合分布。

依据 meta.TIER_ODDS（实测各等级品阶概率）+ 公共牌池（每品阶英雄数 × 每位张数），每个槽位独立抽。
和 rules.route_eta / tools/route_difficulty.py 同一模型，但输入是"一组英雄"，不认路线名：
  - 已在场 / 手里的不用找；
  - 每回合看店 views 次（1 次自然 + 刷新），等级按 max(当前, 进度表) 往上走；
  - remaining（牌池剩余张数）不知道就按满池。
解析计算（不抽样），可直接给推演器 / 网络当特征。
"""
from __future__ import annotations

from agent.decision import meta

from .effects import db

SHOP_SLOTS = meta.SHOP_SLOTS


def quality_of(name: str) -> int:
    c = db().unit(name)
    return c.quality if c else 0


def p_view(name: str, level: int, remaining: int | None = None) -> float:
    """看一次商店（4 个槽位）至少出现一张 name 的概率。"""
    return meta.shop_hit_prob(quality_of(name), min(int(level or 1), meta.MAX_LORD_LEVEL), remaining)


def p_refresh(name: str, level: int, n_refresh: int, remaining: int | None = None) -> float:
    """刷新 n 次（共看 n 次新商店）至少见到一次的概率。"""
    return 1 - (1 - p_view(name, level, remaining)) ** max(0, n_refresh)


def expected_copies(name: str, level: int, views: int, remaining: int | None = None) -> float:
    """看 views 次商店，期望能见到几张（每个槽位独立，见到就算一张）。"""
    q = quality_of(name)
    if not q or q not in meta.POOL_COPIES:
        return 0.0
    rem = meta.POOL_COPIES[q] if remaining is None else max(0, remaining)
    p_slot = meta.tier_odds(q, min(level, meta.MAX_LORD_LEVEL)) * rem / (meta.POOL_HEROES[q] * meta.POOL_COPIES[q])
    return p_slot * SHOP_SLOTS * views


def eta_distribution(names, owned=(), round_no: int = 1, level: int = 1, views: float = 3.0,
                     max_rounds: int = 16, remaining: dict | None = None) -> dict:
    """凑齐 names（每个至少一张）的回合分布。

    返回 {"p_by_round": {回合: 到该回合末已凑齐的概率}, "median": 中位回合(凑不齐 99), "missing": [...],
          "each": {名字: 中位回合}}。各英雄独立（不同英雄互不影响，同一槽位竞争忽略）。
    """
    own = set(owned or ())
    need = [n for n in dict.fromkeys(names) if n and n not in own]
    out = {"missing": need, "p_by_round": {}, "median": round_no if not need else 99, "each": {}}
    if not need:
        return out
    miss = {n: 1.0 for n in need}
    for t in range(1, max_rounds + 1):
        rd = round_no + t - 1                 # 本回合剩下的看店也算一次机会
        lv = min(max(level, meta.level_target(rd)), meta.MAX_LORD_LEVEL)
        for n in need:
            miss[n] *= (1 - p_view(n, lv, (remaining or {}).get(n))) ** views
            if n not in out["each"] and 1 - miss[n] >= 0.5:
                out["each"][n] = rd
        prob = 1.0
        for n in need:
            prob *= 1 - miss[n]
        out["p_by_round"][rd] = prob
        if out["median"] == 99 and prob >= 0.5:
            out["median"] = rd
    for n in need:
        out["each"].setdefault(n, 99)
    return out


def eta(names, owned=(), round_no: int = 1, level: int = 1, views: float = 3.0) -> int:
    return eta_distribution(names, owned, round_no, level, views)["median"]
