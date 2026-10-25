"""推演器第 4 层：**按真实规则做概率推演的养成账**（2026-10-14：只估计养成，不估计那些不准确的数值）。

一局 = 每回合：准备阶段（按默认打法花钱：升级 / 看店买牌 / 合成 / 上场 / 开锦囊）→ 战斗（只结算"击杀 / 胜利 / 牺牲才触发"
的养成效果，胜负用实测平均值）→ 回合结束（经验、收入、自动升级）→ 下回合开始（整备等卡面效果）。商店每格按
棋手等级抽品阶（meta.TIER_ODDS 实测）、再按该品阶剩余张数抽人；合成一张同名 +3 级（1452 次实测）、本体 + 3 张同名觉醒；
每合成 3 次送 1 个效果锦囊（品阶 = 棋手等级），锦囊里开出什么按历史 N 选 1 频率抽（data/engine/draw_stats.json）。
跑很多次取分布。**不估任何战力 / 伤害 / 胜率**。

  from agent.engine.rollout import Plan, rollout_many
  res = rollout_many(position, Plan(carry="百里守约", wanted=[...]), n=200, rounds_to=14)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import board as B
from .effects import db
from .project import Sim, PARAMS, dev_key
from .state import Position, make_unit

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def hero_pool() -> dict:
    """品阶 → 英雄名列表（英雄牌名册）。"""
    cs = json.loads((ROOT / "data/cards.json").read_text(encoding="utf-8"))
    cs = cs.get("cards", cs) if isinstance(cs, dict) else cs
    cs = cs if isinstance(cs, list) else list(cs.values())
    out: dict = {}
    seen = set()
    for x in cs:
        if x.get("category") == "英雄牌" and x.get("quality") in (1, 2, 3, 4, 5) and x["name"] not in seen:
            seen.add(x["name"])
            out.setdefault(int(x["quality"]), []).append(x["name"])
    return out


@lru_cache(maxsize=1)
def event_offers() -> dict:
    try:
        return json.loads((ROOT / "data/engine/event_offers.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


NEAREST_WHO = ("nearest_ally", "nearest_faction")


def feeds_nearest(name: str) -> bool:
    """卡面效果作用于"距离最近的（某派系）友军"的英雄（李元芳古币、明世隐转化、太乙复生…）：推演里站主C 旁边。"""
    c = db().unit(name)
    return bool(c and any(e.who in NEAREST_WHO for e in c.fx()))


@lru_cache(maxsize=1)
def draw_stats() -> dict:
    try:
        return json.loads((ROOT / "data/engine/draw_stats.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@dataclass
class Plan:
    """默认打法（推演里"以后每回合怎么花钱"）：
      carry      主C（他的同名优先买、加等级牌给他）；None = 推演里按养成最好的输出位
      wanted     想要的英雄（核心 / 配套），同名见到就买
      level      "schedule" = 按进度表升级（第 2/4/7/10 回合 2/3/4/5 级）；"hold3_until_yizhi" = 3 级停到开出遗志或 hold_rounds 回合；
                 "rush5" = 有钱就升直到 5 级
      hoard      这些英雄觉醒后的同名先捏在手里（伞海月的干将：到 3 级再一起合成）
      hold_rounds  3 级最多停到第几回合
      roll_reserve 看店时留多少钱（不低于这个就继续刷）"""
    carry: str | None = None
    wanted: list = field(default_factory=list)
    level: str = "schedule"
    hoard: list = field(default_factory=list)
    hold_rounds: int = 9
    roll_reserve: int = 0
    max_refresh: int = 10
    events: bool = True             # 天赋 / N 选 1 / 拍卖 / 枫叶舞强化（按实测频率与真实出现过的选项抽）


class Rollout:
    def __init__(self, pos: Position, plan: Plan, rng: random.Random, params: dict | None = None):
        self.p = pos
        self.plan = plan
        self.rng = rng
        self.sim = Sim(pos, dict({"invest_frac": 0.0}, **(params or {})))   # 投入由本层显式模拟，不用平均投入
        self.merges = 0
        self.jinnang = {}
        self.yizhi = False
        self.taken: dict = {}            # 我们买走的张数（牌池剩余 = 满池 − 我们 − 对手消耗）
        self.log = []
        if not pos.lord_cards:
            pos.lord_cards = ["枫叶舞"]                 # 阿离：枫叶舞绑主C（换棋手时换成该棋手的技能牌）

    # ------------------------------------------------------------ 发牌
    def _remaining(self, name: str, q: int) -> int:
        full = B.meta.POOL_COPIES.get(q, 6)
        opp = self.p.round * PARAMS.get("opp_take_per_round", 0.15)     # 对手平均每回合拿走的张数（估计，按日志里点开看到的剩余数校准）
        return max(0, int(round(full - self.taken.get(name, 0) - opp)))

    def draw_slot(self) -> str | None:
        lv = self.p.lord_level
        odds = B.meta.TIER_ODDS.get(lv) or {1: 1.0}
        r = self.rng.random()
        acc = 0.0
        q = max(odds)
        for t, pr in sorted(odds.items()):
            acc += pr
            if r <= acc:
                q = t
                break
        pool = hero_pool().get(q, [])
        ws = [self._remaining(n, q) for n in pool]
        tot = sum(ws)
        if tot <= 0:
            return None
        x = self.rng.random() * tot
        for n, w in zip(pool, ws):
            x -= w
            if x <= 0:
                return n
        return pool[-1]

    # ------------------------------------------------------------ 拿到一张英雄
    def gain(self, name: str) -> None:
        p = self.p
        self.taken[name] = self.taken.get(name, 0) + 1
        u = p.unit(name)
        if u is not None:
            if name in self.plan.hoard and u.is_awakened and p.lord_level < 3:
                if len(p.hand) < B.HAND_LIMIT:
                    p.hand.append(name)           # 伞海月：觉醒后的同名捏在手里，3 级再合成
                return
            self.sim.merge(u)
            self.merges += 1
            if self.merges % 3 == 0:
                self.open_jinnang(min(p.lord_level, 5))
            return
        if p.pop_used() < p.pop_cap:
            self.sim.deploy(name, cell=self._cell_for(name))
            if name == self.plan.carry:
                p.lord_bound = name
            return
        if name == self.plan.carry or name in self.plan.wanted:
            # 想要的人来了、场上满：卖掉一个不想要的补位件给他腾位（真人都这么打）
            junk = [x for x in p.heroes() if x.pop and x.name != self.plan.carry and x.name not in self.plan.wanted]
            if junk:
                v = min(junk, key=dev_key)
                p.units.remove(v)
                p.energy += 1
                self.sim.deploy(name, cell=v.cell)
                if name == self.plan.carry:
                    p.lord_bound = name
                return
        if len(p.hand) < B.HAND_LIMIT:
            p.hand.append(name)

    def _cell_for(self, name: str):
        """送"最近友军"等级的人站主C 旁边；其余按默认落点。"""
        from .project import best_cell
        car = self.p.unit(self.plan.carry) if self.plan.carry else None
        if car is not None and car.cell and feeds_nearest(name):
            return best_cell(self.p, make_unit(name, 1), near=car.cell)
        return None

    # ------------------------------------------------------------ 事件（选项按真实出现过的抽；推演里按"2 回合后主C 等级 + 全队等级"贪心选）
    def _sample_offer(self, kind: str):
        offs = (event_offers().get("offers") or {}).get(kind) or {}
        R = self.p.round
        for d in (0, 1, -1, 2, -2, 3, -3):
            lst = offs.get(str(R + d))
            if lst:
                return self.rng.choice(lst)["opts"]
        return None

    def _apply_pick(self, kind: str, name: str, q: Position, sim) -> None:
        from .options import Action, do
        if kind == "talent":
            do(q, Action("talent", "", {"name": name}), sim)
        elif kind == "lord_boost":
            q.lord_cards.append(name)
        else:
            c = db().get(name)
            if c and c.category == "英雄牌":
                u = q.unit(name)
                if u is not None:
                    sim.merge(u)
                elif q.pop_used() < q.pop_cap:
                    sim.deploy(name)
                elif len(q.hand) < B.HAND_LIMIT:
                    q.hand.append(name)
            else:
                do(q, Action("gain", "", {"name": name}), sim)

    def _greedy(self, kind: str, opts: list) -> str:
        best, bv = opts[0], -1e9
        for o in opts:
            q = self.p.clone()
            sim = Sim(q, dict(self.sim.P))
            try:
                self._apply_pick(kind, o, q, sim)
                for _ in range(2):
                    sim.step()
            except Exception:  # noqa: BLE001
                continue
            c = q.unit(self.plan.carry) if self.plan.carry else None
            v = (c.level if c else 0.0) + 0.2 * sum(u.level for u in q.heroes())
            if v > bv:
                best, bv = o, v
        return best

    def events(self, phase: str) -> None:
        if not self.plan.events:
            return
        rate = event_offers().get("rate") or {}
        R = str(self.p.round)
        kinds = ("card_pick", "lord_boost") if phase == "prep" else ("auction",)
        for kind in kinds:
            r = float((rate.get(kind) or {}).get(R, 0.0))
            n = int(r) + (1 if self.rng.random() < r - int(r) else 0)
            for _ in range(n):
                opts = self._sample_offer(kind)
                if opts:
                    pick = self._greedy(kind, opts)
                    self._apply_pick(kind, pick, self.p, self.sim)
                    if pick == "遗志":
                        self.yizhi = True
                    self.log.append((self.p.round, kind, pick))

    def talent_on_levelup(self, before: int) -> None:
        for _ in range(max(0, self.p.lord_level - before)):
            opts = self._sample_offer("talent")
            if opts:
                pick = self._greedy("talent", opts)
                self._apply_pick("talent", pick, self.p, self.sim)
                self.log.append((self.p.round, "talent", pick))

    def open_jinnang(self, tier: int) -> None:
        """开效果锦囊：按历史频率抽 3 张选项。遗志在里面就拿；否则 ~40% 是发英雄类（兑换券 / 征召令 / 呼朋唤友）→ 给一张该品阶英雄。"""
        self.jinnang[tier] = self.jinnang.get(tier, 0) + 1
        st = ((draw_stats().get("jinnang") or {}).get(str(tier)) or {})
        cards = st.get("cards") or {}
        if not cards:
            return
        names, ws = list(cards), list(cards.values())
        k = 3
        opts = set()
        for _ in range(20):
            opts.add(self.rng.choices(names, ws)[0])
            if len(opts) >= k:
                break
        if "遗志" in opts:
            self.yizhi = True
            return
        if self.plan.events:
            return                     # 锦囊开出的其他牌：已包含在 card_pick 的实测频率里
        if any(("兑换券" in o or "征召令" in o or "呼朋唤友" in o or "广纳良才" in o) for o in opts):
            pool = hero_pool().get(min(tier, 5), [])
            want = [w for w in self.plan.wanted if w in pool]
            pick = self.rng.choice(want) if want and self.rng.random() < 0.3 else (self.rng.choice(pool) if pool else None)
            if pick:
                self.gain(pick)

    # ------------------------------------------------------------ 准备阶段（默认打法）
    def want(self, name: str) -> bool:
        p = self.p
        if name == self.plan.carry or name in self.plan.wanted:
            return True
        return p.unit(name) is not None and name in (self.plan.wanted + [self.plan.carry or ""])

    def prep(self) -> None:
        p, plan = self.p, self.plan
        # 1) 升级
        tgt = B.meta.level_target(p.round)
        if plan.level == "rush5":
            tgt = 5
        elif plan.level == "hold3_until_yizhi" and p.lord_level >= 3 and not self.yizhi and p.round <= plan.hold_rounds:
            tgt = 3
        elif plan.level == "hold3_until_yizhi" and (self.yizhi or p.round > plan.hold_rounds):
            tgt = 5
        lv0 = p.lord_level
        while p.lord_level < min(tgt, B.MAX_LORD_LEVEL):
            cost = B.level_cost(p.lord_level, p.xp)
            if not cost or p.energy < cost:
                break
            p.energy -= cost
            p.xp += cost
            self.sim.level_ups()
        self.talent_on_levelup(lv0)
        self.events("prep")
        # 2) 看店：第一次免费，之后刷新
        views = 0
        while True:
            shop = [self.draw_slot() for _ in range(B.meta.SHOP_SLOTS)]
            for n in shop:
                if n is None or p.energy < B.HERO_COST:
                    continue
                u = p.unit(n)
                fill = u is None and p.pop_used() < p.pop_cap          # 空位先补人（空着必输）
                if self.want(n) or u is not None or fill:
                    if not fill and u is None and p.pop_used() >= p.pop_cap and len(p.hand) >= B.HAND_LIMIT:
                        continue
                    p.energy -= B.HERO_COST
                    self.gain(n)
            views += 1
            if views > plan.max_refresh or p.energy < B.REFRESH_COST + B.HERO_COST + plan.roll_reserve:
                break
            if plan.level != "rush5" and p.lord_level < tgt:
                break                                                  # 还要攒钱升级
            p.energy -= B.REFRESH_COST
        # 3) 手里的人：有空位就上；场上有同名就合成（捏着的等 3 级）
        for n in list(p.hand):
            u = p.unit(n)
            if u is not None and not (n in plan.hoard and u.is_awakened and p.lord_level < 3):
                p.hand.remove(n)
                self.taken[n] = self.taken.get(n, 0) - 1
                self.gain(n)
            elif u is None and p.pop_used() < p.pop_cap and B.meta.POOL_COPIES:
                p.hand.remove(n)
                self.sim.deploy(n)

    # ------------------------------------------------------------ 一回合
    def set_carry(self) -> None:
        """当前主C：计划指定的人在场就是他；否则在"计划想要的输出位"里取养得最好的（谁先到用谁），都没有就取场上养得最好的输出位。
        枫叶舞跟着绑主C（换绑的代价在决策层算，默认打法里只要主C 变了就换绑）。"""
        from .project import is_output
        p = self.p
        c = p.unit(self.plan.carry) if self.plan.carry else None
        if c is None:
            cands = [u for u in p.heroes() if u.can_carry and is_output(u)]
            pref = [u for u in cands if u.name in self.plan.wanted]
            c = max(pref or cands, key=dev_key, default=None)
        if c is not None:
            p.carry = c.name
            p.lord_bound = c.name

    def round(self) -> dict:
        self.set_carry()
        self.prep()
        self.set_carry()
        self.sim.run_battle()
        self.events("battle")
        lv0 = self.p.lord_level
        self.sim.round_end()
        self.talent_on_levelup(lv0)
        self.sim.round_start()
        p = self.p
        c = p.unit(p.carry) if p.carry else None
        return {"round": p.round, "lord": p.lord_level, "energy": p.energy,
                "carry": c.name if c else "", "carry_lv": c.level if c else 0.0,
                "carry_awk": (3 if c.is_awakened else c.awaken) if c else 0,
                "team_lv": sum(u.level for u in p.heroes()), "n_units": len(p.heroes()),
                "yizhi": self.yizhi, "jinnang": dict(self.jinnang), "merges": self.merges}


def rollout_many(pos: Position, plan: Plan, n: int = 200, rounds_to: int = 14, seed: int = 0,
                 params: dict | None = None) -> list:
    """从 pos 起按 plan 推到 rounds_to 回合，跑 n 次。返回 [每次的逐回合记录列表]。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        q = pos.clone()
        ro = Rollout(q, plan, rng, params)
        traj = []
        while q.round < rounds_to:
            traj.append(ro.round())
        out.append(traj)
    return out


def summarize(runs: list, rounds=(5, 8, 10, 12, 14)) -> dict:
    import statistics as S
    out = {}
    for R in rounds:
        rows = [next((x for x in tr if x["round"] == R), None) for tr in runs]
        rows = [x for x in rows if x]
        if not rows:
            continue
        out[R] = {"carry_lv_med": S.median(x["carry_lv"] for x in rows),
                  "carry_awk": sum(x["carry_awk"] >= 3 for x in rows) / len(rows),
                  "lord": S.median(x["lord"] for x in rows), "team_lv_med": S.median(x["team_lv"] for x in rows),
                  "yizhi": sum(x["yizhi"] for x in rows) / len(rows), "merges": S.median(x["merges"] for x in rows)}
    return out
