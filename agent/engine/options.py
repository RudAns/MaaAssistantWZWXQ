"""枚举可行动作 → 每个动作推演 → 价值 V 与分项 → ΔV（相对"停手"在同一视野下的推演）。

V（"主C 等级当量"，权重 VALUE_W 集中可调）：
  carry_lv   视野末主C 永久等级                       × 0.5
  carry_pow  视野末主C 战力（project.unit_power：品阶/等级/质变/装备/觉醒/攻速）× 3.0
  carry_awk  主C 觉醒进度（0~3，觉醒=词条翻倍）          × 4.0 / 级
  team_lv    其余英雄永久等级和                          × 0.15
  win        视野内各回合胜率之和（掉血的代理量）          × 10
  energy     视野末剩余能量                               × 0.5
  carry_eta  主C 不在场时的预计到手回合数（odds，手里 = 0）× -3
主C 不写死：动作若指定（换主C / 天赋送出可当主C 的单位），按动作；否则 = 局面设定，没有则价值最高的输出位。
没有"临时主C / 最终主C"标签：换主C = V(换成 X, 到对局结束) − V(维持, 到对局结束)，代价全部进推演：
  等级损失（手里有搬等级牌时按卡面公式搬过去）、枫叶舞换绑要卖宿主、装备扇形转移 0、换人回合战力空档。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from . import board as B
from . import odds
from .effects import db
from .project import dev_key, PARAMS, Sim, pick_carry, unit_power, is_output, best_cell
from .state import Position, Unit, make_unit

VALUE_W = {"carry_lv": 0.5, "carry_pow": 3.0, "carry_awk": 4.0, "team_lv": 0.15, "win": 10.0, "energy": 0.5, "carry_eta": -3.0,
           "bound_waste": -1.0, "gear_future": 1.0}
GEAR_FUTURE_V = 1.5        # 一件装备"推演视野之后"的价值：穿在会留到最后、且对口的人身上才算（估计值）
BOUND_ITEM_V = 4.0         # 一件绑定装备困在"会被换掉的人"身上 / 主C 用不上占一格：损失（估计值）
BOUND_GROWTH_V = 8.0       # 卡面带永久成长（每击败 / 每战斗）的绑定装备，损失加倍


def keep_prob(u, carry) -> float:
    """这个单位留到对局结束的概率（估计）：主C 1；给主C 产出过等级的高；品阶越高越可能留下。
    只用来折算"投在他身上、拿不回来"的东西（绑定装备），不改战斗推演。"""
    if u is carry:
        return 1.0
    feeds = bool(carry and (carry.src or {}).get(u.name, 0) > 0)
    return max(0.0, min(1.0, 0.2 + 0.12 * (u.quality or 1) + (0.4 if feeds else 0.0)))


def _gear_future(end, carry) -> float:
    """视野之后装备还值多少：对口才算，按持有者留到最后的概率折算。
    可转移装备在持有者被换掉后还能转给别人 —— 但只有阵容里还有对口的人才有用，这里按"场上对口的人里最可能留下的"取。"""
    from .project import equip_fits
    from .state import BOUND_EQUIPS
    tot = 0.0
    for u in end.units:
        for it in u.equips[:B.MAX_EQUIPS]:
            if it == "?":
                continue
            c = db().get(it)
            v = GEAR_FUTURE_V * (2 if c and (c.equip or {}).get("permanent_growth") else 1)
            if it in BOUND_EQUIPS:
                k = keep_prob(u, carry) if equip_fits(it, u) else 0.0
            else:
                k = max([keep_prob(x, carry) for x in end.units if x.can_equip and equip_fits(it, x)] or [0.0])
            tot += v * k
    return tot


def _bound_waste(end, carry) -> float:
    from .project import equip_fits
    from .state import BOUND_EQUIPS
    w = 0.0
    for u in end.units:
        for it in u.bound:
            if it not in BOUND_EQUIPS:          # 出生自带（双剑）属于英雄本身，不算浪费
                continue
            c = db().get(it)
            v = BOUND_GROWTH_V if c and (c.equip or {}).get("permanent_growth") else BOUND_ITEM_V
            if u is carry:
                w += 0.0 if equip_fits(it, u) else v
            else:
                w += (1 - keep_prob(u, carry)) * v
    return w
SELL_ENERGY = 1          # 卖英雄回多少能量（实测卖弈星手牌 +1；场上英雄按同值，待核对）
TRANSFER_RE = re.compile(r"(转移|保留给|分配给|移交).{0,12}等级|等级.{0,8}(转移|保留给|分配)|原等级")


@dataclass
class Action:
    kind: str                      # stop/buy/deploy/merge/sell/swap/levelup/refresh/use/talent/carry/auction
    label: str
    args: dict = field(default_factory=dict)
    horizon: int = 3


@dataclass
class Scored:
    action: Action
    V: float
    dV: float
    parts: dict
    dparts: dict
    note: str = ""


def card_cost(name: str) -> int:
    c = db().get(name)
    if c and c.category == "英雄牌":
        return B.HERO_COST
    raw = db().raw(name)
    try:
        return int(raw.get("cost") or 0)
    except Exception:
        return 0


def is_hero_card(name: str) -> bool:
    c = db().get(name)
    return bool(c and c.category == "英雄牌")


# ------------------------------------------------------------------ 价值
VALUE_MODE = "dev"          # v1 = 线性等级计分（旧）；v2 = 相对战力取对数（2026-10-14）
VALUE_W2 = {"log_pow": 10.0, "terminal": 10.0, "energy": 0.5, "carry_eta": -3.0, "bound_waste": -1.0, "gear_future": 1.0}
TERMINAL_ROUNDS = 4.0      # 视野之后的局面还影响几回合（打折后的等效回合数）


def team_power(p: Position) -> float:
    from .project import unit_power
    from .project import agg
    return agg(unit_power(u) for u in p.on_board()) or 1e-6


def lasting_power(p: Position) -> float:
    """视野末局面里"会留下来"的战力：会被换掉的人（keep_prob 低），身上的装备按留下的概率算，
    其余部分按"不带装备的他"算 —— 装备会转走，人会被同等级别的替补顶上，投在他身上的东西不会一直在。"""
    import copy as _c
    from .project import unit_power
    carry = pick_carry(p)
    from .project import agg
    vals = []
    for u in p.on_board():
        k = keep_prob(u, carry)
        if k >= 1.0 or not u.equips:
            vals.append(unit_power(u))
            continue
        bare = _c.copy(u)
        bare.equips = [x for x in u.equips if x in u.bound]
        vals.append(k * unit_power(u) + (1 - k) * unit_power(bare))
    return agg(vals) or 1e-6


DEV_KEYS = ("carry_lv", "carry_awk", "carry_gear", "core_on", "core_lv", "team_lv", "slots_empty", "energy",
            "wasted", "carry_eta")
DEV_W = {"carry_lv": 3.0, "carry_awk": 2.0, "carry_gear": 1.0, "core_on": 1.0, "core_lv": 1.0, "team_lv": 1.0,
         "slots_empty": -1.0, "energy": 0.05, "wasted": -1.0, "carry_eta": -1.0}
# ↑ 各项之间的权衡**没有校准**，只是把养成向量压成一个数方便看；真正的权衡交给局面网络（输入 = 下面这个养成向量）。


def value_dev(p0: Position, sim_end: Position, recs: list, carry_name: str = "") -> tuple[float, dict]:
    """**只看养成**的结果向量（推演到视野末）：
      carry_lv   主C 永久等级 log(1+L)          carry_awk  主C 觉醒进度 0~3
      carry_gear 主C 身上对口装备件数            core_on    核心阵容（喂主C / 主坦 / 路线成员 …）在场人数 —— 这里用"非主C 英雄里留到最后概率高的"
      core_lv    这些人的 log 等级和             team_lv    全队 log(1+等级) 和
      slots_empty 空人口（每回合平均）           energy     剩余能量（+ 手里英雄可卖的能量）
      wasted     投在会被换掉的人身上、拿不回来的东西（绑定装备等）  carry_eta 主C 还没到手要等几回合
    不含任何战力 / 伤害 / 胜率估计。"""
    import math
    from .project import equip_fits
    end = sim_end
    if carry_name:
        end.carry = carry_name
    c = pick_carry(end)
    heroes = [u for u in end.heroes() if u.presence == "prep"]
    keep = [u for u in heroes if u is not c and keep_prob(u, c) >= 0.6]
    parts = dict.fromkeys(DEV_KEYS, 0.0)
    if c:
        parts["carry_lv"] = math.log1p(max(0.0, c.level))
        parts["carry_awk"] = 3.0 if c.is_awakened else float(min(3, c.awaken))
        parts["carry_gear"] = float(sum(1 for it in c.equips[:B.MAX_EQUIPS] if equip_fits(it, c)))
    elif carry_name:
        parts["carry_eta"] = 0.0 if carry_name in end.hand else float(
            max(0, odds.eta([carry_name], (), end.round, end.lord_level) - end.round))
    parts["core_on"] = float(len(keep))
    parts["core_lv"] = sum(math.log1p(u.level) for u in keep)
    parts["team_lv"] = sum(math.log1p(u.level) for u in heroes)
    parts["slots_empty"] = float(max(0, end.pop_cap - end.pop_used()))
    parts["energy"] = float(end.energy) + SELL_ENERGY * sum(1 for h in end.hand if is_hero_card(h))
    parts["wasted"] = _bound_waste(end, c)
    V = sum(DEV_W[k] * v for k, v in parts.items())
    return V, parts


def value(p0: Position, sim_end: Position, recs: list, carry_name: str = "") -> tuple[float, dict]:
    if VALUE_MODE == "dev":
        return value_dev(p0, sim_end, recs, carry_name)
    if VALUE_MODE == "v2":
        return value_v2(p0, sim_end, recs, carry_name)
    return value_v1(p0, sim_end, recs, carry_name)


def value_v2(p0: Position, sim_end: Position, recs: list, carry_name: str = "") -> tuple[float, dict]:
    """价值 = Σ视野内每回合 log(我方有效战力 / 敌方战力) + 视野末局面的 log 战力比 × 余下等效回合 + 能量等。
    战力是乘法关系：+2 级对 10 级主C 是 +11%、对 69 级是 +2.6% —— 固定加级件（古币、李元芳）后期自动变弱，
    按比例生效的（降法力、复生、学识宝石 8%/回合）不缩水。不再单独给"主C 等级"线性计分。"""
    import math
    from .project import enemy_power
    end = sim_end
    if carry_name:
        end.carry = carry_name
    c = pick_carry(end)
    parts = {"log_pow": 0.0, "terminal": 0.0,
             "energy": float(end.energy) + SELL_ENERGY * sum(1 for h in end.hand if is_hero_card(h)),
             "carry_eta": 0.0, "bound_waste": 0.0, "gear_future": 0.0}
    parts["log_pow"] = sum(math.log(max(1e-6, r.power) / max(1e-6, r.enemy)) for r in recs)
    rest = max(0, PARAMS["game_end_round"] - end.round)
    parts["terminal"] = min(TERMINAL_ROUNDS, rest) * math.log(lasting_power(end) / max(1e-6, enemy_power(end.round)[0]))
    if not c and carry_name:
        in_hand = carry_name in end.hand
        parts["carry_eta"] = 0.0 if in_hand else float(max(0, odds.eta([carry_name], (), end.round, end.lord_level) - end.round))
    parts["bound_waste"] = _bound_waste(end, c)
    parts["gear_future"] = _gear_future(end, c)
    V = sum(VALUE_W2[k] * v for k, v in parts.items())
    return V, parts


def value_v1(p0: Position, sim_end: Position, recs: list, carry_name: str = "") -> tuple[float, dict]:
    end = sim_end
    if carry_name:
        end.carry = carry_name
    c = pick_carry(end)
    parts = {"carry_lv": 0.0, "carry_pow": 0.0, "carry_awk": 0.0, "team_lv": 0.0, "win": 0.0,
             "energy": float(end.energy) + SELL_ENERGY * sum(1 for h in end.hand if is_hero_card(h)),   # 手里的英雄至少值卖出的能量
             "carry_eta": 0.0, "bound_waste": 0.0, "gear_future": 0.0}
    if c:
        parts["carry_lv"] = c.level
        parts["carry_pow"] = unit_power(c, c.level)
        parts["carry_awk"] = 3.0 if c.is_awakened else float(min(3, c.awaken))
    elif carry_name:
        in_hand = carry_name in end.hand
        parts["carry_eta"] = 0.0 if in_hand else float(max(0, odds.eta([carry_name], (), end.round, end.lord_level) - end.round))
    parts["team_lv"] = sum(u.level for u in end.heroes() if u is not c)
    parts["win"] = sum(r.win_p for r in recs)
    parts["bound_waste"] = _bound_waste(end, c)
    parts["gear_future"] = _gear_future(end, c)
    V = sum(VALUE_W.get(k, 0.0) * v for k, v in parts.items())
    return V, parts


def evaluate(p: Position, horizon: int, carry_name: str = "", params: dict | None = None):
    sim = Sim(p, params)
    recs = [sim.step() for _ in range(horizon)]
    return value(p, p, recs, carry_name), recs


# ------------------------------------------------------------------ 动作执行（作用在 clone 上）
def _use_card(sim: Sim, name: str, target: Unit | None) -> bool:
    p = sim.p
    c = db().get(name)
    if not c:
        return False
    if c.category == "装备牌":
        if target is None or not target.can_equip or len(target.equips) >= B.MAX_EQUIPS:
            return False
        target.equips.append(name)
        from .state import BOUND_EQUIPS
        if name in BOUND_EQUIPS:
            target.bound.append(name)
        for e in c.effects:
            if e.event == "equip" and e.effect in ("level", "temp_level", "coin", "atk"):
                sim.apply(e, target, name, chosen=target)
        return True
    if c.category == "棋手技能牌":
        if target is None:
            return False
        p.lord_bound = target.name
        return True
    for e in c.fx():
        if e.event in ("use", "deploy", "other"):
            if e.effect == "hero_card":
                g = str(e.extra.get("card") or "")
                if g and (db().unit(g) and db().unit(g).category == "召唤物"):
                    u = make_unit(g, 1)
                    u.cell = best_cell(p, u)
                    if u.cell:
                        p.units.append(u)
                elif g:
                    p.hand.append(g)
                continue
            sim.apply(e, target, name, chosen=target)
    sim.react("react:effect_card", 1)
    return True


def _target_ok(name: str, t: Unit) -> bool:
    c = db().get(name)
    if not c:
        return False
    if c.category == "装备牌":
        from .project import equip_fits
        return t.can_equip and len(t.equips) < B.MAX_EQUIPS and name not in t.equips and equip_fits(name, t)
    if not t.counts_as_hero:
        return False
    for e in c.fx():
        if e.who == "chosen" and e.tfaction:
            from .normalize import norm_faction
            if t.faction != norm_faction(e.tfaction):
                return False
        if e.effect == "trigger" and e.who == "chosen":
            kinds = str(e.trigger_kind or "prepare").split("|")
            if not any(any(x.event == k for x in (t.fx.fx(t.is_awakened) if t.fx else [])) for k in kinds):
                return False
    return True


def _transfer_card(p: Position) -> Optional[str]:
    for h in p.hand:
        c = db().get(h)
        if c and c.category != "英雄牌" and any(TRANSFER_RE.search(e.raw or "") for e in c.effects):
            return h
    return None


def do(p: Position, a: Action, sim: Sim) -> bool:
    """把动作作用到局面 p（就地），不合法返回 False。"""
    k, g = a.kind, a.args
    if k == "stop":
        return True
    if k in ("buy", "auction"):
        name = g["name"]
        cost = g.get("cost", card_cost(name))
        if p.energy < cost:
            return False
        p.energy -= cost
        if k == "buy" and name in p.shop:
            p.shop.remove(name)
        if is_hero_card(name):
            u = p.unit(name)
            if u:
                sim.merge(u)
            elif p.pop_used() < p.pop_cap:
                sim.deploy(name)
            elif len(p.hand) < B.HAND_LIMIT:
                p.hand.append(name)
            else:
                return False
            return True
        if g.get("target"):
            return _use_card(sim, name, p.unit(g["target"]))
        if len(p.hand) >= B.HAND_LIMIT:
            return False
        p.hand.append(name)
        return True
    if k == "deploy":
        if g["name"] not in p.hand or p.pop_used() >= p.pop_cap:
            return False
        p.hand.remove(g["name"])
        if g.get("cell") and g["cell"] not in p.free_cells():
            return False
        return sim.deploy(g["name"], cell=g.get("cell")) is not None
    if k == "merge":
        u = p.unit(g["name"])
        if g["name"] not in p.hand or not u:
            return False
        p.hand.remove(g["name"])
        sim.merge(u)
        return True
    if k in ("sell", "swap"):
        u = p.unit(g["sell"])
        if not u or u.kind != "hero":
            return False
        _sell(p, u, sim)
        if k == "swap":
            if g["name"] not in p.hand:
                return False
            p.hand.remove(g["name"])
            cell = g.get("cell") or u.cell
            if cell not in p.free_cells():
                return False
            return sim.deploy(g["name"], cell=cell) is not None
        return True
    if k == "levelup":
        cost = B.level_cost(p.lord_level, p.xp)
        if not cost or p.energy < cost:
            return False
        p.energy -= cost
        p.xp += cost
        sim.level_ups()
        return True
    if k == "use":
        t = p.unit(g["target"]) if g.get("target") else None
        if g["name"] not in p.hand:
            return False
        if t is not None and not _target_ok(g["name"], t):
            return False
        if not _use_card(sim, g["name"], t):
            return False
        p.hand.remove(g["name"])
        return True
    if k == "talent":
        p.talents.append(g["name"])
        c = db().get(g["name"])
        if c and c.category == "装备牌":            # 天赋直接给装备：穿给最合适的人（主C 优先）
            ts = _rank_targets(p, g["name"])
            if ts:
                _use_card(sim, g["name"], ts[0])
            elif len(p.hand) < B.HAND_LIMIT:
                p.hand.append(g["name"])
            return True
        for e in (c.effects if c else []):
            if e.event == "use" and e.effect == "hero_card":
                gname = str(e.extra.get("card") or "")
                gu = db().unit(gname) if gname else None
                if gu and gu.category == "召唤物":           # 送出的单位直接上场（不占人口）
                    u = make_unit(gname, 1)
                    u.cell = best_cell(p, u)
                    if u.cell:
                        p.units.append(u)
                        sim.deploy_summons(u)
                elif gname:
                    _gain_hero(p, sim, gname)
            elif e.event == "use" and e.effect in ("level", "energy", "lord_xp", "coin", "pop", "refresh"):
                sim.apply(e, None, g["name"])
        if g.get("carry"):
            p.carry = g["carry"]
            if any(db().get(x) and db().get(x).category == "棋手技能牌" for x in p.lord_cards) and not p.lord_bound:
                p.lord_bound = g["carry"]
        return True
    if k == "gain":                          # 拿到一张牌（N 选 1 / 拍卖 / 送牌）：英雄合成或上场，装备穿给最合适的人，效果牌打给最合适的人
        name = g["name"]
        c = db().get(name)
        if not c:
            return False
        if is_hero_card(name):
            _gain_hero(p, sim, name)
            return True
        if c.category == "天赋牌":
            return do(p, Action("talent", "", {"name": name}), sim)
        if c.category in ("装备牌", "效果牌"):
            ts = _rank_targets(p, name)
            if ts and _use_card(sim, name, ts[0]):
                return True
        if len(p.hand) < B.HAND_LIMIT:
            p.hand.append(name)
        return True
    if k == "carry":
        sim_keep_bind[0] = bool(g.get("keep_bind"))
        try:
            return _switch_carry(p, g["name"], sim)
        finally:
            sim_keep_bind[0] = False
    if k == "move_equip":                    # 扇形轮盘转移，代价 0
        a_, b_ = p.unit(g["from"]), p.unit(g["to"])
        it = g["item"]
        if not a_ or not b_ or it not in a_.equips or it in a_.bound or not b_.can_equip                 or len(b_.equips) >= B.MAX_EQUIPS or (it in b_.equips and it != "?"):
            return False
        a_.equips.remove(it)
        b_.equips.append(it)
        return True
    if k == "swap_equip":                    # 对方满 3 件时互换
        a_, b_ = p.unit(g["a"]), p.unit(g["b"])
        ia, ib = g["item_a"], g["item_b"]
        if not a_ or not b_ or ia not in a_.equips or ib not in b_.equips or ia in a_.bound or ib in b_.bound:
            return False
        a_.equips[a_.equips.index(ia)] = ib
        b_.equips[b_.equips.index(ib)] = ia
        return True
    return False


def _gain_hero(p: Position, sim: Sim, name: str) -> None:
    """拿到一张英雄牌（天赋 / 效果牌送的）：场上有同名就合成，有空位就上场，否则进手牌（之后只值卖出的能量）。"""
    u = p.unit(name)
    if u and is_hero_card(name):
        sim.merge(u)
    elif is_hero_card(name) and p.pop_used() < p.pop_cap and sim.deploy(name) is not None:
        pass
    elif len(p.hand) < B.HAND_LIMIT:
        p.hand.append(name)


def _sell(p: Position, u: Unit, sim: Sim) -> None:
    sim.fire(u, "sell")
    p.units.remove(u)
    p.energy += SELL_ENERGY
    for e in u.equips:                      # 可转移装备退回手牌；绑定装备随人消失
        if e != "?" and e not in u.bound:
            p.hand.append(e)
    if p.lord_bound == u.name:
        p.lord_bound = ""
    if p.carry == u.name:
        p.carry = ""


sim_keep_bind = [False]


def _switch_carry(p: Position, name: str, sim: Sim) -> bool:
    new = p.unit(name)
    if not new or not new.can_carry:
        return False
    old = pick_carry(p)
    if old is new:
        return False
    # 搬等级：手里有搬等级牌就按卡面公式搬（变量 = 旧主C 的永久等级）
    tc = _transfer_card(p)
    if old is not None and tc:
        c = db().get(tc)
        moved = 0.0
        lv = old.level
        for e in c.effects:
            if e.effect == "level":
                v = e.value(sim.ctx(old, {"prev_holder_real_level": lv, "removed_unit_perm_level": lv,
                                          "removed_level": lv, "level_of_chosen": lv}))
                if v and v > 0:
                    moved = max(moved, v)
        if moved > 0:
            new.level += moved
            new.src[tc] = new.src.get(tc, 0.0) + moved
            old.level = max(1.0, old.level - moved)
            p.hand.remove(tc)
    # 装备扇形转移：代价 0
    if old is not None:
        for it in [x for x in old.equips if x not in old.bound]:
            if len(new.equips) >= B.MAX_EQUIPS or not new.can_equip or (it in new.equips and it != "?"):
                break
            old.equips.remove(it)
            new.equips.append(it)
    # 枫叶舞类"指定后绑定"的棋手技能：换绑要卖掉宿主
    if p.lord_bound and p.lord_bound != name and sim_keep_bind[0]:
        pass                                     # 只换主C、绑定牌不动（不卖宿主）
    elif p.lord_bound and p.lord_bound != name:
        host = p.unit(p.lord_bound)
        if host is not None and host.kind == "hero":
            _sell(p, host, sim)
        p.lord_bound = name
    elif any(db().get(x) and db().get(x).category == "棋手技能牌" for x in p.lord_cards):
        p.lord_bound = name
    p.carry = name
    return True


# ------------------------------------------------------------------ 枚举
def enumerate_actions(p: Position, talents: list | None = None, auction: list | None = None,
                      horizon: int = 3, max_use_targets: int = 4) -> list:
    acts = [Action("stop", "停手", horizon=horizon)]
    heroes = p.heroes()
    full = p.pop_used() >= p.pop_cap
    for name in dict.fromkeys(p.shop):
        cost = card_cost(name)
        if cost > p.energy:
            continue
        if is_hero_card(name) or (db().get(name) and db().get(name).category not in ("效果牌", "装备牌")):
            acts.append(Action("buy", f"买 {name}", {"name": name, "cost": cost}, horizon))
        else:
            for t in _rank_targets(p, name)[:2]:
                acts.append(Action("buy", f"买 {name} 给 {t.name}", {"name": name, "cost": cost, "target": t.name},
                                   horizon))
    for name in dict.fromkeys(p.hand):
        c = db().get(name)
        if not c:
            continue
        if c.category == "英雄牌":
            if p.unit(name):
                acts.append(Action("merge", f"合成 {name}", {"name": name}, horizon))
            elif not full:
                acts.append(Action("deploy", f"上场 {name}", {"name": name}, horizon))
            else:
                for u in sorted(heroes, key=unit_power)[:2]:
                    if u.name != p.carry:
                        acts.append(Action("swap", f"卖 {u.name} 换上 {name}", {"sell": u.name, "name": name}, horizon))
        elif c.category in ("效果牌", "装备牌", "天赋牌", "棋手技能牌"):
            for t in _rank_targets(p, name)[:max_use_targets]:
                acts.append(Action("use", f"{name} → {t.name}", {"name": name, "target": t.name}, horizon))
    cost = B.level_cost(p.lord_level, p.xp)
    if cost and cost <= p.energy:
        acts.append(Action("levelup", f"升级（{cost}）", {}, horizon))
    if p.energy >= B.REFRESH_COST or p.free_refresh > 0:
        acts.append(Action("refresh", "刷新", {}, horizon))
    for u in heroes:
        if u.kind == "hero" and u.name != (pick_carry(p).name if pick_carry(p) else "") and not full:
            pass
    rest = max(1, PARAMS["game_end_round"] - p.round)
    cur = pick_carry(p)
    if cur is not None and cur.can_equip:     # 可转移装备 → 主C（满 3 件就和主C 身上的互换）
        for u in p.units:
            if u is cur:
                continue
            for it in dict.fromkeys(x for x in u.equips if x not in u.bound and x != "?"):
                if it in cur.equips:
                    continue
                if len(cur.equips) < B.MAX_EQUIPS:
                    acts.append(Action("move_equip", f"装备 {it}：{u.name} → {cur.name}",
                                       {"from": u.name, "item": it, "to": cur.name}, horizon))
                else:
                    for jt in [x for x in cur.equips if x not in cur.bound and x != "?"][:1]:
                        acts.append(Action("swap_equip", f"装备互换 {it}({u.name}) ↔ {jt}({cur.name})",
                                           {"a": u.name, "item_a": it, "b": cur.name, "item_b": jt}, horizon))
    for u in p.units:
        if u.can_carry and is_output(u) and u is not cur and u.presence == "prep":
            acts.append(Action("carry", f"换主C → {u.name}", {"name": u.name}, min(rest, 10)))
            if p.lord_bound and p.lord_bound != u.name:     # 也试"只换主C、枫叶舞类绑定先不动"（不卖宿主）
                acts.append(Action("carry", f"换主C → {u.name}（绑定不动）", {"name": u.name, "keep_bind": True},
                                   min(rest, 10)))
    for t in talents or []:
        gift = _gift_carry(t)
        acts.append(Action("talent", f"天赋 {t}" + (f"（送出可当主C 的 {gift}）" if gift else ""),
                           {"name": t, **({"carry": gift} if gift else {})}, min(rest, 10) if gift else horizon))
    for lot in auction or []:
        n, bid = (lot if isinstance(lot, (list, tuple)) else (lot, 3))
        if bid <= p.energy:
            acts.append(Action("auction", f"拍 {n}（{bid}）", {"name": n, "cost": bid}, horizon))
    return acts


def _gift_carry(talent: str) -> str:
    c = db().get(talent)
    for e in (c.effects if c else []):
        g = str(e.extra.get("card") or "")
        if e.effect == "hero_card" and g:
            from .state import SUMMON_ATTRS
            if (SUMMON_ATTRS.get(g) or {}).get("can_carry"):
                return g
    return ""


def _rank_targets(p: Position, card: str) -> list:
    cands = [u for u in p.units if u.presence == "prep" and _target_ok(card, u)]
    cur = pick_carry(p)
    from .state import BOUND_EQUIPS
    if card in BOUND_EQUIPS:                    # 拿不下来的：给最可能留到最后的人
        return sorted(cands, key=lambda u: (u is not cur, -keep_prob(u, cur), tuple(-x for x in dev_key(u))))
    return sorted(cands, key=lambda u: (u is not cur, tuple(-x for x in dev_key(u))))


# ------------------------------------------------------------------ 评估
def score_options(p: Position, talents: list | None = None, auction: list | None = None, horizon: int = 3,
                  actions: list | None = None, params: dict | None = None, refresh_candidates: int = 3) -> list:
    acts = actions if actions is not None else enumerate_actions(p, talents, auction, horizon)
    base_cache: dict = {}

    def base(h):
        if h not in base_cache:
            q = p.clone()
            (V, parts), _ = evaluate(q, h, params=params)
            base_cache[h] = (V, parts)
        return base_cache[h]

    out = []
    acts = _expand_cells(p, acts)
    for a in acts:
        if a.kind == "refresh":
            out.append(_score_refresh(p, a, base, params, refresh_candidates))
            continue
        q = p.clone()
        sim = Sim(q, params)
        if not do(q, a, sim):
            continue
        carry_name = a.args.get("carry") or (a.args.get("name") if a.kind == "carry" else "")
        (V, parts), recs = evaluate(q, a.horizon, carry_name, params)
        V0, parts0 = base(a.horizon)
        # 同一视野下"维持现状"的主C 也要按动作指定的主C 名来比吗？不：基准就是维持现任
        out.append(Scored(a, V, V - V0, parts, {k: parts[k] - parts0.get(k, 0) for k in parts},
                          note=f"unmodeled {q.meta.get('unmodeled', 0)}"))
    best: dict = {}                              # 同一动作的不同落点只留最好的那个
    for sc in out:
        k = sc.action.args.get("_group")
        if k is None:
            continue
        if k not in best or sc.dV > best[k].dV:
            best[k] = sc
    out = [sc for sc in out if sc.action.args.get("_group") is None or best.get(sc.action.args["_group"]) is sc]
    out.sort(key=lambda s: -s.dV)
    return out


POS_RE = re.compile(r"最近|相邻|左右|周围|\d格|同一?排|身后|前方")


def _expand_cells(p: Position, acts: list) -> list:
    """上场 / 换人：卡面和站位有关（钟馗"左右 1 格获得牺牲"、明世隐"距离最近"…）的英雄，每个可用格子都试一次，取最好的。
    和站位无关的照旧用默认落点。"""
    out = []
    for a in acts:
        if a.kind not in ("swap", "deploy") or a.args.get("cell"):
            out.append(a)
            continue
        r = db().raw(a.args["name"]) or {}
        if not POS_RE.search((r.get("skill") or "") + (r.get("desc") or "")):
            out.append(a)
            continue
        cells = list(p.free_cells())
        if a.kind == "swap":
            u = p.unit(a.args["sell"])
            if u and u.cell:
                cells.append(u.cell)
        key = id(a)
        for c in cells:
            out.append(Action(a.kind, a.label + f" @{c}", {**a.args, "cell": c, "_group": key}, a.horizon))
    return out


def _score_refresh(p: Position, a: Action, base, params, k: int) -> Scored:
    """刷新 = 花 2（或免费）看一次新商店：Σ 见到 h 的概率 × max(0, ΔV(买 h))，h 取场上同名 + 主C 公式变量的产出方。"""
    D = db()
    cost = 0 if p.free_refresh > 0 else B.REFRESH_COST
    cands = [u.name for u in p.heroes() if u.kind == "hero"]
    cur = pick_carry(p)
    if cur:
        for var in D.formula_vars(cur.name):
            cands += [n for n in D.producers(var) if is_hero_card(n)][:4]
    cands = [c for c in dict.fromkeys(cands) if odds.p_view(c, p.lord_level) > 0]
    V0, parts0 = base(a.horizon)
    gains = []
    for n in cands:
        q = p.clone()
        q.energy -= cost
        sim = Sim(q, params)
        if not do(q, Action("buy", "", {"name": n, "cost": B.HERO_COST}), sim):
            continue
        (V, _), _ = evaluate(q, a.horizon, params=params)
        gains.append((odds.p_view(n, p.lord_level), V - V0, n))
    gains.sort(key=lambda x: -x[0] * max(0, x[1]))
    ev = 0.0
    miss = 1.0
    for pv, dv, n in gains[: max(k, 6)]:
        ev += miss * pv * max(0.0, dv)
        miss *= 1 - pv
    dE = -cost * VALUE_W["energy"]
    dV = ev + dE
    return Scored(a, V0 + dV, dV, {}, {"expected_buy": ev, "energy": dE},
                  note="候选 " + ", ".join(f"{n} p{pv:.2f} Δ{dv:+.1f}" for pv, dv, n in gains[:4]))


def best(p: Position, **kw) -> Scored | None:
    s = score_options(p, **kw)
    return s[0] if s else None


def explain(scored: list, n: int = 8) -> str:
    lines = []
    for s in scored[:n]:
        dp = " ".join(f"{k}{v:+.1f}" for k, v in s.dparts.items() if abs(v) >= 0.05)
        lines.append(f"ΔV {s.dV:+6.1f}  {s.action.label:<24} [{dp}] {s.note}")
    return "\n".join(lines)



def robust_scores(p: Position, n: int = 24, seed: int = 0, **kw) -> dict:
    """参数在 PARAM_RANGES 里随机抽 n 组，每组都算一遍，统计每个动作"是最优"的比例和平均 ΔV。
    返回 {label: {"p_best", "dV_mean", "dV_lo", "dV_hi", "action"}}，按 p_best 降序。
    结论稳（p_best 高）的题可以直接当标签；来回翻的交给老师，并记下是哪类参数在翻。"""
    import random
    from . import project as PJ
    rng = random.Random(seed)
    saved = dict(PJ.PARAMS)
    stats: dict = {}
    try:
        for i in range(n):
            if i:                                   # 第 0 组用默认参数
                for k, (lo, hi) in PJ.PARAM_RANGES.items():
                    PJ.PARAMS[k] = rng.uniform(lo, hi)
            sc = score_options(p, **kw)
            if not sc:
                continue
            groups = {}
            for x in sc:                            # 同一动作不同落点算一个
                key = x.action.label.split(" @")[0]
                if key not in groups or x.dV > groups[key].dV:
                    groups[key] = x
            best = max(groups.values(), key=lambda x: x.dV)
            for key, x in groups.items():
                st = stats.setdefault(key, {"best": 0, "dVs": [], "action": x.action})
                st["dVs"].append(x.dV)
                st["best"] += x is best
    finally:
        PJ.PARAMS.clear()
        PJ.PARAMS.update(saved)
    out = {}
    for key, st in stats.items():
        v = sorted(st["dVs"])
        out[key] = {"p_best": st["best"] / max(1, len(v)), "dV_mean": sum(v) / len(v),
                    "dV_lo": v[int(0.1 * (len(v) - 1))], "dV_hi": v[int(0.9 * (len(v) - 1))], "action": st["action"]}
    return dict(sorted(out.items(), key=lambda kv: (-kv[1]["p_best"], -kv[1]["dV_mean"])))
