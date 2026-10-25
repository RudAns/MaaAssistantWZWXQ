"""流派 = 手法：终点阵容 + 转型前提 + 执行步骤 + 放弃条件（2026-10-14：网上推荐的阵容是最终目标，
中间按实际情况往不同流派靠；海月流的转型时机是前期刷出很多干将、且刷出对海月有用的护盾核心装备）。

推演器用它回答："从这个局面开始投入这个流派，第几回合前成型的概率、成型时主C 几级、半路放弃亏多少"。
**要不要投入**是价值判断，不在这里做（交给局面网络 / Laya）；这里只负责"投入了就照步骤打"和"现在满足哪些前提"。

  from agent.sim.flows import FLOWS, FlowPlayer, flow_facts
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.engine import board as B
from .engine import card_text, category, impl
from .play import Player, rating
from .state import State


@dataclass
class Flow:
    key: str
    name: str
    reservoir: str                      # 前期攒等级的"仓库"（干将：1 阶、张数多、合成便宜、自带绑定装备）；"" = 直接养主C，没有转等级这一步
    final: str                          # 最终主C
    transfer_cards: tuple = ("遗志",)   # 手牌里的转等级手段（遗志：移除仓库，等级给场上随机 1 人 → 先清场）
    transfer_boosts: tuple = ("红叶染锋",)  # 棋手强化 / 天赋里的转等级手段（枫叶舞宿主离场留 50% 给下一个宿主）
    hold_level: int = 3                 # 还没拿到转等级手段时停在这一级（合成出这一阶锦囊；遗志只出现在 3 阶锦囊里：45 个里 17 个）
    hold_copies: bool = True            # 仓库觉醒后再买到的同名先捏在手里，到 hold_level 再一次性合成（2026-10-14 定的节奏）
    hold_until: int = 8                 # …最多停到第几回合，之后不等了
    roll_level: int = 5                 # 拿到手段后升到几级刷最终主C（最终主C 的品阶进池等级）
    abort_round: int = 12               # 到这回合（含）结束还没拿到最终主C → 放弃，回到普通打法
    keep_item_words: tuple = ()         # 卡面含这些词的装备留给最终主C，不给仓库
    end_comp: tuple = ()                # 终点阵容的其余人（成型后补）
    core: tuple = ()                    # 核心件（和主C 一起让等级滚起来的最小集合；成型前就买）
    want_cards: tuple = ()              # 这个流派要的效果牌 / 天赋（官网阵容写的推荐牌，如传承）：N 选 1 / 天赋 / 拍卖里见到就拿
    notes: str = ""

    @property
    def direct(self) -> bool:
        return not self.reservoir


FLOWS = {
    "haiyue": Flow(
        key="haiyue", name="遗志海月（阿离）", reservoir="干将莫邪", final="海月",
        keep_item_words=("护盾", "生命"), core=("扁鹊", "瑶", "太乙真人"),   # 配套：扁鹊炸盾、瑶分装备、太乙复生
        end_comp=("太乙真人", "瑶", "扁鹊", "云中君", "墨子"),
        notes="海月依旧是输出最高的主C（技能 + 枫之洪流无限施法 + 施法叠盾转生命）；牺牲只触发一次，有太乙时是可接受的代价。",
    ),
}


def _route_flows() -> dict:
    """data/route_min_engine.json 的每条路线 → 直接养主C 的流派（终点 = 核心 + 前期先买 + 后期补位）。"""
    import json
    from pathlib import Path
    from .engine import quality
    path = Path(__file__).resolve().parents[2] / "data/route_min_engine.json"
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for k, v in d.items():
        if not isinstance(v, dict) or not v.get("carry") or k.startswith("_"):
            continue
        carry = v["carry"]
        core = tuple(x["unit"] for x in (v.get("core_set") or []) if x.get("unit") and x["unit"] != carry)
        early = tuple(x for x in (v.get("early_priority") or []) if x != carry)
        late = tuple(x for x in (v.get("late_fill") or []) if x != carry)
        q = quality(carry) or 3
        out[f"route:{k}"] = Flow(key=f"route:{k}", name=v.get("name") or carry, reservoir="", final=carry,
                                 transfer_cards=(), transfer_boosts=(), hold_level=0, hold_until=0,
                                 roll_level=max(2, min(5, q)), abort_round=10,
                                 core=tuple(dict.fromkeys(core + early)), end_comp=tuple(dict.fromkeys(core + early + late)),
                                 notes="由 data/route_min_engine.json 生成：" + (v.get("name") or ""))
    return out


def _lineup_flows(min_lineups: int = 1) -> dict:
    """官网推荐阵容（阿离能用的，data/lineups.json）→ 每个主C 一个直接养主C 的流派：
    同一主C 的多套阵容合并，按使用人数的对数加权；核心件 = 前 6 回合运营阶段里加权出现 ≥40% 的人；
    终点 = 最终阵容里加权出现 ≥30% 的人；要的牌 = 推荐的效果牌 / 天赋（加权出现 ≥30%）。"""
    import math
    from collections import defaultdict
    from agent.decision import routes as R
    from .engine import quality
    by = defaultdict(list)
    for lu in R.ali_lineups():                           # ali_lineups 本身按使用人数 + key 排好序
        if lu.get("carry"):
            by[lu["carry"]].append(lu)
    out = {}
    for carry, lus in sorted(by.items()):
        if len(lus) < min_lineups:
            continue
        w = {id(lu): math.log10(max(10, lu.get("use_num") or 10)) for lu in lus}
        W = sum(w.values())
        def share(pred):
            c = defaultdict(float)
            for lu in lus:
                for x in sorted(set(pred(lu))):           # 排序：集合的遍历顺序随进程的字符串哈希变，题面必须逐字确定
                    c[x] += w[id(lu)]
            return {k: c[k] / W for k in sorted(c)}
        early = share(lambda lu: [h for st in lu.get("stages") or [] if (st.get("from_round") or 0) <= 6
                                  for h in (st.get("main") or [])])
        fin = share(R.final_team)
        cards = share(lambda lu: (lu.get("effect_cards") or []) + (lu.get("talent_cards") or []))
        core = tuple(k for k, v in sorted(early.items(), key=lambda x: (-x[1], x[0])) if v >= 0.4 and k != carry)
        end = tuple(k for k, v in sorted(fin.items(), key=lambda x: (-x[1], x[0])) if v >= 0.3 and k != carry)
        want = tuple(k for k, v in sorted(cards.items(), key=lambda x: (-x[1], x[0])) if v >= 0.3)
        q = quality(carry) or 3
        out[f"lineup:{carry}"] = Flow(key=f"lineup:{carry}", name=f"官网·{carry}（{len(lus)}套）", reservoir="", final=carry,
                                      transfer_cards=(), transfer_boosts=(), hold_level=0, hold_until=0,
                                      roll_level=max(2, min(5, q)), abort_round=10, core=core,
                                      end_comp=tuple(dict.fromkeys(core + end)), want_cards=want,
                                      notes=f"由官网 {len(lus)} 套阿离阵容合并生成（data/lineups.json）")
    return out


FLOWS.update(_route_flows())
FLOWS.update(_lineup_flows())


def _holds_transfer(s: State, f: Flow) -> str:
    """手里 / 身上可用的转等级手段。红叶染锋只对枫叶舞宿主生效（限一次）：宿主不是仓库时，换绑要先卖掉原宿主，
    那一卖就把红叶染锋用掉了 → 这时不算可用。"""
    for c in f.transfer_cards:
        if c in s.hand:
            return c
    for c in f.transfer_boosts:
        if (c in s.lord_cards or c in s.talents) and not s.flags.get(f"{c}_used") and (not s.maple or s.maple == f.reservoir):
            return c
    return ""


def keep_item(name: str, f: Flow) -> bool:
    return category(name) == "装备牌" and any(w in card_text(name) for w in f.keep_item_words)


def flow_pieces(f: Flow) -> tuple:
    """这个流派要凑的人：最终主C + 核心件（转等级型再加仓库）。"""
    return tuple(dict.fromkeys(((f.reservoir,) if f.reservoir else ()) + (f.final,) + tuple(f.core)))


def flow_rank(owned, shop=()) -> list:
    """各流派离成型还差什么（只用"我方有哪些英雄 + 眼前商店"，不推演，线上题面 / 局面网络特征共用这一份）：
    返回 [{final, key, pieces, have, missing, in_shop, buy_completes}]，按"现在买得齐 > 已有比例 > 已有件数"排序，同一最终主C 只留一条。"""
    owned = set(owned)
    shop = list(shop or [])
    rows = []
    for key in sorted(FLOWS):
        f = FLOWS[key]
        pcs = list(flow_pieces(f))
        have = [x for x in pcs if x in owned]
        if not have:
            continue
        miss = [x for x in pcs if x not in owned]
        left, ins = list(shop), []
        for x in miss:
            if x in left:
                ins.append(x)
                left.remove(x)
        rows.append({"final": f.final, "key": key, "pieces": pcs, "have": have, "missing": miss, "in_shop": ins,
                     "buy_completes": bool(miss) and len(ins) == len(miss)})
    # 先比"现在买得齐"，再比已有件数（只比比例会让两件套的流派——有一件就算一半——永远排第一），再比比例
    rows.sort(key=lambda r: (not r["buy_completes"], -len(r["have"]), -len(r["have"]) / len(r["pieces"]), r["key"]))
    out, seen = [], set()
    for r in rows:
        if r["final"] in seen:
            continue
        seen.add(r["final"])
        out.append(r)
    return out


def shop_completion(s: State, f: Flow) -> dict:
    """商店里眼前这几张能不能把缺的人补齐：缺谁、商店里有谁、一次买齐要多少钱、钱 / 手牌 / 人口够不够。"""
    from .engine import Engine
    have = {u.name for u in s.heroes()} | set(s.hand)
    missing = [x for x in flow_pieces(f) if x not in have]
    shop = list(s.shop or [])
    in_shop, left = [], list(shop)
    for x in missing:
        if x in left:
            in_shop.append(x)
            left.remove(x)
    cost = sum(Engine(s.clone(keep_log=False)).shop_cost(x) for x in in_shop)
    hand_room = B.HAND_LIMIT - len(s.hand)
    pop_room = B.pop_cap(s.lord_level) + s.pop_bonus - s.pop_used()
    fits = hand_room >= len(in_shop) and pop_room >= len(in_shop)     # 买得下、也放得上场（腾位要卖人，另算）
    return {"missing": missing, "in_shop": in_shop, "cost": cost, "pop_room": pop_room,
            "affordable": s.energy >= cost and fits,
            "completes_now": bool(missing) and len(in_shop) == len(missing) and s.energy >= cost and fits}


def flow_facts(s: State, f: Flow) -> dict:
    """转型前提的现状（只列事实，不下结论）。"""
    r = s.unit(f.reservoir) if f.reservoir else None
    sc = shop_completion(s, f)
    return {
        "pieces_have": [x for x in flow_pieces(f) if x not in sc["missing"]], "pieces_missing": sc["missing"],
        "missing_in_shop": sc["in_shop"], "buy_all_cost": sc["cost"], "pop_room": sc["pop_room"],
        "completes_if_buy_now": sc["completes_now"],
        "round": s.round, "lord_level": s.lord_level, "energy": s.energy,
        "reservoir_level": r.level if r else 0, "reservoir_awaken": r.awaken if r else 0,
        "reservoir_in_hand": s.hand.count(f.reservoir),
        "transfer_tool": _holds_transfer(s, f),
        "keep_items": [x for x in s.hand if keep_item(x, f)] + ([e for e in r.equips if keep_item(e, f)] if r else []),
        "final_on_board": s.unit(f.final) is not None, "final_in_hand": f.final in s.hand,
    }


class FlowPlayer(Player):
    """投入某个流派后的打法：按步骤走，不再每步按眼前养成重新选。放弃后退回普通打法（Player）。"""

    def __init__(self, eng, flow: Flow, **kw):
        self.flow = flow
        self.done = False          # 等级已转到最终主C
        self.aborted = False
        super().__init__(eng, **kw)

    def _lock_if_wanted(self, view: list, wanted: set) -> bool:
        """流派要的人就在这一眼商店里、但这回合买不起（或正在攒钱冲级）→ 锁店留到下回合，不再刷新。"""
        s = self.e.s
        have = {u.name for u in s.heroes()} | set(s.hand)
        left = [x for x in view if x and x in wanted and x not in have]
        if left and s.energy < B.HERO_COST + s.shop_extra_cost + (B.level_cost(s.lord_level, s.xp) or 0
                                                                    if (not self.done and _holds_transfer(s, self.flow)
                                                                        and s.lord_level < self.flow.roll_level) else 0):
            self.e.lock_shop([x for x in view if x and x not in have])
            return True
        return False

    def buy_card(self, name: str) -> bool:
        f, s = self.flow, self.e.s
        if self.active and not self.done and not f.direct and _holds_transfer(s, f) and s.lord_level < f.roll_level:
            return False                                  # 攒钱冲级期间不买
        return super().buy_card(name)

    def protected(self) -> set:
        return super().protected() | ({self.flow.reservoir, self.flow.final} if self.active else set())

    def _sub(self, eng):
        p = FlowPlayer(eng, self.flow, events=False, lookahead=0)
        p.done, p.aborted = self.done, self.aborted
        return p

    @property
    def active(self) -> bool:
        return not self.aborted

    # ------------------------------------------------------------ 选择：要的牌直接拿
    def choose(self, eng, kind, options, **ctx):
        f = self.flow
        if self.active and not self.done and not _holds_transfer(eng.s, f):
            for o in options:
                if o in f.transfer_cards or o in f.transfer_boosts:
                    return o
        if self.active and not self.done:
            for o in options:
                if o == f.final or (o == f.reservoir and o):
                    return o
        if self.active and f.want_cards:
            for o in options:
                if o in f.want_cards:
                    return o
        if self.active and f.direct:
            have = {u.name for u in eng.s.heroes()} | set(eng.s.hand)
            for o in options:
                if o in f.core and o not in have:
                    return o
        return super().choose(eng, kind, options, **ctx)

    # ------------------------------------------------------------ 主C / 枫叶舞
    def pick_carry(self):
        if not self.active:
            return super().pick_carry()
        s, f = self.e.s, self.flow
        if f.direct:
            fin = s.unit(f.final)
            if fin is None:
                return super().pick_carry()          # 主C 没到手：临时主C 照普通打法挑
            self.done = True
            s.carry = fin.name
            old = s.unit(s.maple) if s.maple else None
            if old is None:
                self.e.bind_maple(fin)
            elif old is not fin and (s.round >= 9 or old.level <= 2 * fin.level):
                self.e.sell(old)                     # 临时宿主卖掉换绑（太强的临时主C 先等主C 追上一半，最晚第 9 回合）
                self.e.bind_maple(fin)
            return
        want = f.final if self.done else f.reservoir
        u = s.unit(want)
        if u is None:
            return super().pick_carry() if not s.carry or not s.unit(s.carry) else None
        s.carry = u.name
        if s.maple != u.name and (not s.maple or not s.unit(s.maple)):
            self.e.bind_maple(u)

    # ------------------------------------------------------------ 手牌
    def play_hand(self):
        if not self.active:
            return super().play_hand()
        e, s, f = self.e, self.e.s, self.flow
        fin = s.unit(f.final)
        held = []
        res = s.unit(f.reservoir)
        if (f.hold_copies and not self.done and res is not None and res.awakened
                and s.lord_level < f.hold_level and not _holds_transfer(s, f)):
            while f.reservoir in s.hand:                # 1~2 级合成只出 1~2 阶锦囊（没有遗志）→ 先捏着
                s.hand.remove(f.reservoir)
                held.append(f.reservoir)
        for name in list(s.hand):                      # 留给最终主C 的装备：最终主C 在场才穿，否则捏着
            if keep_item(name, f):
                if fin is not None and self.done and e.equip(fin, name):
                    s.hand.remove(name)
                else:
                    held.append(name)
                    s.hand.remove(name)
        if f.final in s.hand and s.unit(f.final) is None:   # 最终主C：腾位也要上
            if s.pop_used() >= e.pop_cap():
                weak = [x for x in s.heroes() if x.pop and x.name not in (f.reservoir, f.final) and x.name not in f.core]
                if weak:
                    e.sell(min(weak, key=lambda x: (rating(x.name), x.level)))
            if s.pop_used() < e.pop_cap() and e.can_play_hero():
                s.hand.remove(f.final)
                e.deploy(f.final, cell=self.cell_for(f.final))
        try:
            super().play_hand()
        finally:
            s.hand.extend(held)

    # ------------------------------------------------------------ 转等级
    def transfer(self) -> bool:
        e, s, f = self.e, self.e.s, self.flow
        fin, res = s.unit(f.final), s.unit(f.reservoir)
        if fin is None:
            return False
        if res is None:                                  # 仓库没了（被卖 / 变身）：直接认最终主C
            self.done = True
            return True
        if "遗志" in f.transfer_cards and "遗志" in s.hand:
            for x in list(s.heroes()):                   # 清场：遗志的等级给场上随机 1 人
                if x is not fin and x is not res:
                    e.sell(x)
            lv = res.level
            s.hand.remove("遗志")
            e.use_card("遗志", res)
            e.note("流派转等级", flow=f.key, tool="遗志", level=lv, to=f.final)
        elif any(b in s.lord_cards or b in s.talents for b in f.transfer_boosts):
            lv = res.level
            if s.maple != res.name:
                return False
            e.sell(res)                                   # 枫叶舞宿主离场 → 红叶染锋留 50%
            e.note("流派转等级", flow=f.key, tool="红叶染锋", level=lv, to=f.final)
        else:
            return False
        if s.maple != fin.name:
            e.bind_maple(fin)
        s.carry = fin.name
        self.done = True
        return True

    # ------------------------------------------------------------ 准备阶段
    def _prep_direct(self):
        """直接养主C：等级按进度表；见到主C / 核心件就买；到主C 进池等级还没拿到 → 多刷几次找主C。"""
        e, s, f = self.e, self.e.s, self.flow
        have_final = s.unit(f.final) is not None or f.final in s.hand
        if not have_final and s.round > f.abort_round:
            self.aborted = True
            e.note("流派放弃", flow=f.key, round=s.round)
            return Player.prep(self)
        goal = B.meta.level_target(s.round)
        lv0 = s.lord_level
        while s.lord_level < min(goal, B.MAX_LORD_LEVEL):
            cost = B.level_cost(s.lord_level, s.xp)
            if not cost or s.energy < cost or not e.spend(cost, src="升级"):
                break
            s.xp += cost
            e.level_ups()
        for _ in range(s.lord_level - lv0):
            self.event("talent")
        want = {f.final} | set(f.core) | (set(f.end_comp) if have_final else set())
        rolling = not have_final and s.lord_level >= f.roll_level
        views = 0
        while True:
            view = e.shop_view()
            for n in view:
                if n and category(n) != "英雄牌":
                    self.buy_card(n)
                    continue
                if not n or s.energy < B.HERO_COST + s.shop_extra_cost:
                    continue
                on = s.unit(n) is not None
                fill = not on and s.pop_used() < e.pop_cap()
                if n in want or on or fill:
                    if not on and n not in want and len(s.hand) >= B.HAND_LIMIT - 1:
                        continue
                    e.buy(n)
            self.play_hand()
            self.pick_carry()
            if self._lock_if_wanted(view, {f.final} | set(f.core)):
                break
            views += 1
            have_final = s.unit(f.final) is not None or f.final in s.hand
            rolling = not have_final and s.lord_level >= f.roll_level
            if views > (12 if rolling else 8) or s.energy < B.REFRESH_COST + B.HERO_COST:
                break
            if not e.refresh_shop():
                break
        self.play_hand()
        self.pick_carry()

    def _pick_reservoir(self) -> None:
        """仓库不限于干将（2026-10-14：也可以是其他英雄，干将是 1 星英雄、前期比较容易获得）：
        干将不在场也不在手里时，用场上等级最高的 1~2 阶英雄（不是最终主C），选定后固定。"""
        import dataclasses
        from .engine import quality
        s, f = self.e.s, self.flow
        if f.direct or self.done or s.unit(f.reservoir) is not None or f.reservoir in s.hand:
            return
        cands = [u for u in s.heroes() if u.name != f.final and (quality(u.name) or 9) <= 2]
        if not cands:
            return
        sub = max(cands, key=lambda u: (u.level, u.awaken, u.name))
        self.flow = dataclasses.replace(f, reservoir=sub.name)
        self.e.note("流派仓库", flow=f.key, reservoir=sub.name)

    def prep(self):
        if not self.active:
            return super().prep()
        if self.flow.direct:
            return self._prep_direct()
        self._pick_reservoir()
        e, s, f = self.e, self.e.s, self.flow
        if not self.done and s.round > f.abort_round and s.unit(f.final) is None and f.final not in s.hand:
            self.aborted = True
            e.note("流派放弃", flow=f.key, round=s.round)
            return super().prep()
        tool = _holds_transfer(s, f)
        if self.done:
            goal = B.meta.level_target(s.round)
        elif tool:
            goal = f.roll_level
        elif s.round < f.hold_until:
            goal = min(f.hold_level, max(B.meta.level_target(s.round), 1))
        else:
            goal = f.roll_level
        lv0 = s.lord_level
        while s.lord_level < min(goal, B.MAX_LORD_LEVEL):
            cost = B.level_cost(s.lord_level, s.xp)
            if not cost or s.energy < cost or not e.spend(cost, src="升级"):
                break
            s.xp += cost
            e.level_ups()
        for _ in range(s.lord_level - lv0):
            self.event("talent")
        want = {f.reservoir, f.final} | (set(f.end_comp) if self.done else set())
        rolling = (not self.done) and s.lord_level >= f.roll_level and bool(tool)
        # 攒钱冲级：拿到手段、还没到刷最终主C 的等级 → 升级按钮要一次付清剩余经验，钱只留给升级；
        # 例外是仓库同名（合成继续给仓库加等级），且买完仍够下一次升级
        saving = (not self.done) and bool(tool) and s.lord_level < f.roll_level
        if saving:
            need = B.level_cost(s.lord_level, s.xp) or 0
            view = e.shop_view()
            for n in view:
                if n and category(n) != "英雄牌":
                    self.buy_card(n)
                    continue
                if n in (f.reservoir, f.final) and s.energy - B.HERO_COST - s.shop_extra_cost >= need:
                    e.buy(n)
            self._lock_if_wanted(view, {f.final})
            self.play_hand()
            self.pick_carry()
            if s.unit(f.final) is not None:
                self.transfer()
            return
        # 刷锦囊：停在 hold_level 等遗志时，锦囊按**所有英雄**的合成次数送（每 3 次 1 个，品阶 = 棋手等级）→
        # 钱花在"场上 / 手里已有的同名"上，多合成、多拿 3 阶锦囊
        farming = (not self.done) and (not tool) and s.lord_level >= f.hold_level and s.round < f.hold_until
        views = 0
        while True:
            view = e.shop_view()
            for n in view:
                if n and category(n) != "英雄牌":
                    self.buy_card(n)
                    continue
                if not n or s.energy < B.HERO_COST + s.shop_extra_cost:
                    continue
                on = s.unit(n) is not None
                fill = not on and s.pop_used() < e.pop_cap()
                pair = farming and n in s.hand
                if n in want or on or pair or (fill and not rolling):
                    if not on and n not in want and len(s.hand) >= B.HAND_LIMIT - 1:
                        continue
                    e.buy(n)
            self.play_hand()
            self.pick_carry()
            if not self.done and (s.unit(f.final) is not None) and tool:
                self.transfer()
            if self._lock_if_wanted(view, {f.final, f.reservoir} - {""}):
                break
            views += 1
            # 刷最终主C：能刷就刷（这是这一步的全部目的）；其余时候和普通打法一样最多看 8 次
            cap = 30 if (rolling or farming) and not self.done else 8
            if views > cap or s.energy < B.REFRESH_COST + B.HERO_COST:   # 留够买一张的钱，否则刷到了也买不起
                break
            if not rolling and s.lord_level < goal:
                break
            if not e.refresh_shop():
                break
        self.play_hand()
        self.pick_carry()
        if not self.done and s.unit(f.final) is not None and _holds_transfer(s, f):
            self.transfer()
