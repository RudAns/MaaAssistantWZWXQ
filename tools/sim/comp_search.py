"""给定主C，用正向推演器比较不同配套阵容的养成结果（只比养成：等级 / 觉醒 / 开战时的临时等级，不估伤害）。

做法：从第 START 回合的典型局面出发（主C 已在场、绑枫叶舞；配套前几个已上场），之后每回合照默认打法（agent.sim.play.Player）走：
升棋手等级、天赋 / N 选 1 / 枫叶舞强化 / 拍卖事件、刷商店——但**商店只买阵容里的人**（合成或补空位），事件给的阵容外英雄直接卖牌。
每个阵容跑 n 局取中位数。

  .venv/Scripts/python.exe -X utf8 tools/sim/comp_search.py --carry 虞姬 --pool 少司缘,鬼谷子,大司命,东皇太一,太乙真人,敖隐,张良,云中君,项羽,明世隐,干将莫邪 --size 6 --n 24
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics as S
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.engine import board as B  # noqa: E402
from agent.sim import engine as E  # noqa: E402
from agent.sim.engine import Engine, card_text, category  # noqa: E402
from agent.sim.play import Player  # noqa: E402
from agent.sim.state import State  # noqa: E402

BACK = [(617, 496), (499, 500), (720, 500), (555, 569), (681, 571)]


class CompPlayer(Player):
    """阵容固定的打法：主C 不换，只买 / 上阵容里的人。"""

    def __init__(self, eng, comp, carry, **kw):
        self.comp, self.carry_name = list(comp), carry
        super().__init__(eng, **kw)

    def _sub(self, eng):
        return CompPlayer(eng, self.comp, self.carry_name, events=False, lookahead=0)

    def pick_carry(self):
        s = self.e.s
        c = s.unit(self.carry_name)
        if c:
            s.carry = c.name
            if s.maple != c.name:
                self.e.bind_maple(c)

    def play_hand(self):
        e, s = self.e, self.e.s
        for name in list(s.hand):
            if name in s.hand and category(name) in ("英雄牌", "召唤物"):
                if name not in self.comp and name != self.carry_name:
                    s.hand.remove(name)              # 阵容外英雄：卖牌换 1 能量
                    s.energy += 1
                    continue
                if not e.can_play_hero():
                    continue
                u = s.unit(name)
                if u is not None:
                    s.hand.remove(name)
                    e.merge(u)
                elif s.pop_used() < e.pop_cap():
                    s.hand.remove(name)
                    e.deploy(name, cell=_cell_for(e, name, self.carry_name))
        heroes = [n for n in s.hand if category(n) in ("英雄牌", "召唤物")]
        rest = [n for n in s.hand if n not in heroes]
        s.hand[:] = rest + heroes
        super().play_hand()                          # 其余牌（效果 / 装备）照默认打法；英雄牌这时只剩等空位的

    def prep(self):
        e, s = self.e, self.e.s
        tgt = B.meta.level_target(s.round)
        lv0 = s.lord_level
        while s.lord_level < min(tgt, B.MAX_LORD_LEVEL):
            cost = B.level_cost(s.lord_level, s.xp)
            if not cost or s.energy < cost or not e.spend(cost, src="升级"):
                break
            s.xp += cost
            e.level_ups()
        for _ in range(s.lord_level - lv0):
            self.event("talent")
        want = set(self.comp) | {self.carry_name}
        views = 0
        while True:
            for n in e.shop_view():
                if n and category(n) != "英雄牌":
                    self.buy_card(n)
                    continue
                if not n or n not in want or s.energy < B.HERO_COST + s.shop_extra_cost:
                    continue
                on = s.unit(n) is not None
                if on or s.pop_used() < e.pop_cap():
                    e.buy(n)
            self.play_hand()
            self.pick_carry()
            views += 1
            if views > 8 or s.energy < B.REFRESH_COST + B.HERO_COST or s.lord_level < tgt:
                break
            if not e.refresh_shop():
                break
        self.play_hand()
        self.pick_carry()


def _cell_for(e, name, carry):
    free = e.free_cells()
    if not free:
        return None
    if name == carry:
        for c in BACK:
            if c in free:
                return c
    cu = e.s.unit(carry)
    if cu is not None and cu.cell and "距离最近" in card_text(name):   # 明世隐 / 太乙真人这类"最近的英雄"：贴着主C
        return min(free, key=lambda c: B.hex_dist(cu.cell, c))
    return free[0]


def start_state(carry, comp, start_round, carry_lv, sup_lv, seed):
    s = State(round=start_round, lord_level=B.meta.level_target(start_round), energy=B.income(start_round) if hasattr(B, "income") else 8,
              lord_cards=["枫叶舞", "惊鸿游龙"])
    e = Engine(s, random.Random(seed))
    u = e.deploy(carry, cell=BACK[0], level=carry_lv)
    s.carry = carry
    e.bind_maple(u)
    cap = e.pop_cap()
    for n in comp[:cap - 1]:
        e.deploy(n, cell=_cell_for(e, n, carry), level=sup_lv)
    return s


def run_comp(carry, comp, n, start_round=5, end_round=14, carry_lv=9, sup_lv=4, lookahead=1, seed=0):
    rec = {R: {"carry": [], "carry_fight": [], "team": [], "awk": [], "kill": []} for R in (8, 10, 12, end_round)}
    for i in range(n):
        s = start_state(carry, comp, start_round, carry_lv, sup_lv, seed * 1000 + i)
        e = Engine(s, random.Random(seed * 7919 + i))
        p = CompPlayer(e, comp, carry, lookahead=lookahead, samples=1)
        fight = {}
        orig = e.fire

        def fire(u, timing, _o=orig, _s=s, _f=fight, **kw):
            r = _o(u, timing, **kw)
            if timing == "battle_start" and u.name == carry:
                _f[_s.round] = u.lv()               # 开战时的等级（含临时等级：敖隐 / 东皇太一 / 枫舞扶摇…）
            return r
        e.fire = fire
        while s.round < end_round:
            p.round()
            R = s.round
            if R in rec:
                c = s.unit(carry)
                rec[R]["carry"].append(c.level if c else 0)
                rec[R]["carry_fight"].append(fight.get(R - 1, 0))
                rec[R]["team"].append(sum(u.level for u in s.heroes()))
                rec[R]["awk"].append(1 if c and c.awakened else 0)
                rec[R]["kill"].append(sum(kw.get("n", 0) for (_r, ev, kw) in s.log
                                          if ev == "等级" and kw.get("unit") == carry and kw.get("src") in KILL_SRC))
    out = {}
    for R, d in rec.items():
        if not d["carry"]:
            continue
        o = {k: (S.median(v) if k != "awk" else sum(v) / len(v)) for k, v in d.items() if v}
        q = S.quantiles(d["carry"], n=4) if len(d["carry"]) >= 4 else [o["carry"]] * 3
        o["q25"], o["q75"] = q[0], q[2]
        out[R] = o
    return out


# 主C 等级里"靠击杀得来"的来源（击杀是外生抽样的，这部分对所有阵容一样，只是噪声；单独列出来）
KILL_SRC = {"翩跹舞步·蓄势"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carry", default="虞姬")
    ap.add_argument("--pool", default="少司缘,鬼谷子,大司命,东皇太一,太乙真人,敖隐,张良,云中君,项羽,明世隐,干将莫邪")
    ap.add_argument("--fixed", default="", help="一定在阵容里的人（逗号分隔）")
    ap.add_argument("--comps", default="", help="直接给阵容：用 ; 分隔，每个阵容用 , 分隔")
    ap.add_argument("--size", type=int, default=6, help="配套人数（不含主C）")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--scen", default="中", help="战斗结果情景（逗号分隔：低,中,高）；多档时按各档名次的最差值排序")
    a = ap.parse_args()
    if a.comps:
        comps = [tuple(x.split(",")) for x in a.comps.split(";") if x]
    else:
        fixed = [x for x in a.fixed.split(",") if x]
        pool = [x for x in a.pool.split(",") if x and x not in fixed]
        comps = [tuple(fixed) + c for c in itertools.combinations(pool, a.size - len(fixed))]
    scens = [x for x in a.scen.split(",") if x]
    print(f"{len(comps)} 个阵容 × {a.n} 局 × 情景 {scens}")
    res = []
    t0 = time.time()
    for k, comp in enumerate(comps):
        rr = {}
        for sc in scens:
            E.set_battle_scenario(sc)
            rr[sc] = run_comp(a.carry, comp, a.n, lookahead=a.lookahead)
        res.append((comp, rr))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(comps)}  {time.time() - t0:.0f}s", flush=True)
    E.set_battle_scenario("中")
    score = lambda r: r[12]["carry"] + 0.25 * r[12]["carry_fight"]  # noqa: E731
    rank = {}
    for sc in scens:
        order = sorted(range(len(res)), key=lambda i: -score(res[i][1][sc]))
        for pos, i in enumerate(order):
            rank.setdefault(i, {})[sc] = pos + 1
    idx = sorted(range(len(res)), key=lambda i: (max(rank[i].values()), sum(rank[i].values())))
    print(f"\n按各情景第 12 回合{a.carry}等级的名次排序（取最差的那档）；中位数 [四分位]；击杀 = 其中靠击杀得来的；开战 = 上回合开战时含临时等级：")
    for i in idx[:a.top]:
        comp, rr = res[i]
        print("  " + "、".join(comp) + "   名次 " + " ".join(f"{sc}{rank[i][sc]}" for sc in scens))
        for sc in scens:
            print(f"     [{sc}] " + " | ".join(f"R{R} {d['carry']:.0f}[{d['q25']:.0f}-{d['q75']:.0f}] 击杀{d['kill']:.0f} 开战{d['carry_fight']:.0f} 全队{d['team']:.0f} 觉醒{d['awk']:.0%}"
                                           for R, d in sorted(rr[sc].items())))
    if a.out:
        Path(a.out).write_text(json.dumps([{"comp": c, "res": r, "rank": rank[i]} for i, (c, r) in enumerate(res)], ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
