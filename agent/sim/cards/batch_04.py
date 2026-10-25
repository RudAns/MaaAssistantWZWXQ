"""第 4 批：以天赋牌为主（来客 / 精兵 / 古币类天赋 / 阿科米亚核心类 / 三分天赋…）+ 王维 + 破损古币。

2026-10-14 统一接口后（原来本模块给引擎打的补丁都并进了 engine.py）：
  - 阿科米亚核心：eng.s.core / eng.core_add（发 core_up）；开战分配由引擎 core_distribute 统一做，本批的
    能量改造 / 光棱塔 用类属性 core_no_share 让引擎不分配。
  - 升级所需经验修正（人才济济 -2）：引擎 level_ups 认持有牌的 xp_need_delta。
  - 获得牌的拦截（零钱兑换：古币 → 破损古币）：引擎 gain_card 认持有牌的 gain_redirect。
  - 手牌里的牌（险中求富 in_hand=True）/ 本回合打出过的牌（锦囊 → eng.attach_round，s.round_cards）由引擎 holders() 产出。
  - 被触发的登场（能量改造 / 锦囊额外登场 / 智能装甲额外整备）走 eng.trigger，"打出英雄牌"用 eng.is_card_use() 区分。
  - 天赋的"战斗开始时"：借 react_battle_start（每个单位一次）+ 每场一次的标记。
  - 多彩古币改写古币的主动使用：本模块重新注册"古币"（继承 pilot 的实现，只在有多彩古币天赋时改走变身）。

同名的"天赋 / 赠送牌"（超速成长、前浪的荣光、全军列阵、胜之不武）共用一个类：
  play.py 选天赋时先把天赋放进 talents 再 use_card(名字, None)。用 _talent_pick 区分：
  talents 里这个名字的个数多于已经发过牌的次数 = 这次是选天赋（发牌），否则 = 打出赠送牌。
  要目标的赠送牌（超速成长 / 前浪的荣光 / 全军列阵）有目标就一定是打出赠送牌。
"""
from __future__ import annotations

from functools import lru_cache

from agent.engine import board as B
from ..engine import card, card_db, category, faction, quality, has_keyword
from ..state import Unit

FACTIONS = ("河洛", "逐鹿", "大河流域", "日落海", "三分之地")
COIN_CARDS = ("古币", "精致古币")


# ============================================================ 辅助
def _in_hand(eng, name: str) -> bool:
    return name in eng.s.hand


def _on_board(eng, u) -> bool:
    return u is not None and u in eng.s.units


def _talent_pick(eng, name: str) -> bool:
    """这次 use_card(name) 是不是"选天赋"：talents 里 name 的个数多于已经结算过的次数（结算一次记一次）。"""
    k = "_天赋已发:" + name
    done = eng.s.flags.get(k, 0)
    if eng.s.talents.count(name) > done:
        eng.s.flags[k] = done + 1
        return True
    return False


def _add_xp(eng, n: int, src: str = "") -> None:
    eng.add_xp(n, src=src)


# ============================================================ 本回合整回合生效的牌（锦囊…）：engine.attach_round / s.round_cards
def _attach_round(eng, name: str) -> None:
    eng.attach_round(name)


@lru_cache(maxsize=1)
def _basic_equips() -> tuple:
    """基础装备 = 1 阶、卡面"可以铸造成…"的 8 件（风暴巨剑 / 雷鸣刃 / 速击之枪 / 大棒 / 守护者之铠 / 力量腰带 / 圣者法典 / 破碎圣杯）。"""
    return tuple(n for n, c in card_db().items()
                 if c.get("category") == "装备牌" and "可以铸造成" in (c.get("skill") or ""))


@lru_cache(maxsize=1)
def _sunset_equips() -> tuple:
    """【日落海】专属装备（卡面以"【日落海】专属装备"开头：星源装甲 / 星源狩枪 / 星源星徽 / 星源合金盾 / 月影之力）。"""
    return tuple(n for n, c in card_db().items()
                 if c.get("category") == "装备牌" and (c.get("skill") or "").startswith("【日落海】专属装备"))


@lru_cache(maxsize=1)
def _totems() -> tuple:
    return tuple(n for n, c in card_db().items()
                 if c.get("category") == "召唤物" and n.endswith("图腾") and "（" not in n)


def _gain_basic(eng, n: int = 1) -> None:
    for _ in range(n):
        if _basic_equips():
            eng.gain_card(eng.rng.choice(_basic_equips()))


def _total_level(eng) -> float:
    """英雄总等级（真实等级之和，不含临时等级）。待核对：游戏里的"总等级"是否算临时等级 / 召唤单位。"""
    return sum(u.level for u in eng.heroes())


def _hero_options(eng, tiers, fac=None, k: int = 3) -> list:
    """卡牌 k 选 1 的选项：每个选项先在 tiers 里随机一个品阶，再按牌库剩余抽，不重复。"""
    opts = []
    for _ in range(30):
        x = eng.draw_hero(eng.rng.choice(list(tiers)), fac)
        if x and x not in opts:
            opts.append(x)
        if len(opts) >= k:
            break
    return opts


def _pick_hero_card(eng, tiers, fac=None, kind: str = "hero") -> None:
    opts = _hero_options(eng, tiers, fac)
    pick = eng.choose(kind, opts, tiers=tuple(tiers), faction=fac)
    if pick:
        eng.take(pick)
        eng.gain_card(pick)


def _draw_hero_card(eng, tier=None, fac=None) -> None:
    for _ in range(10):
        x = eng.draw_hero(tier, fac)
        if x:
            eng.take(x)
            eng.gain_card(x)
            return


def _battle_once(eng, key: str) -> bool:
    """天赋的"战斗开始时"：借每个单位的 battle_start 事件，第一次返回 True（on_battle_end 里 _battle_reset）。"""
    k = "battle:" + key
    if eng.s.flags.get(k):
        return False
    eng.s.flags[k] = True
    return True


def _battle_reset(eng, key: str) -> None:
    eng.s.flags.pop("battle:" + key, None)


def _is_summon_unit(u) -> bool:
    """召唤单位：召唤物 / 视为不占人口的赠送单位（始祖熊灵…）；摆件（kind=object）不算。"""
    return u.kind == "summon" or (u.kind == "hero" and not u.pop)


def _spread_temp(eng, units: list, total: int, src: str) -> None:
    """把 total 点临时等级逐点随机分给 units。"""
    if not units or total <= 0:
        return
    got = {}
    for _ in range(int(total)):
        u = eng.rng.choice(units)
        got[id(u)] = got.get(id(u), 0) + 1
    for u in units:
        if got.get(id(u)):
            eng.temp(u, got[id(u)], src=src)


def _attack_count(eng) -> float:
    """一场战斗里英雄的普攻次数估计 = Σ eng.attacks(英雄)（攻速 × 20 秒）。"""
    return sum(eng.attacks(x) for x in eng.s.on_board() if x.kind == "hero")


def _place_hero(eng, name: str, level: float = 1.0):
    """把英雄直接放上场（召唤上场，不是打出英雄牌 → 不触发登场）。"""
    free = eng.free_cells()
    if not free:
        return None
    u = Unit(name=name, level=level, cell=free[0], kind="hero", pop=True, uid=eng.s.next_uid)
    eng.s.next_uid += 1
    eng.s.units.append(u)
    eng.note("召唤上场", unit=name, cell=free[0])
    return u


def _high_effect_card(eng):
    """高阶效果牌：按效果锦囊历史开出的 4~5 阶效果牌频率抽；没有统计时退回卡库 4~5 阶非棋手专属效果牌。"""
    try:
        from agent.engine.rollout import draw_stats
        st = {}
        for t in ("4", "5"):
            for n, w in (((draw_stats().get("jinnang") or {}).get(t) or {}).get("cards") or {}).items():
                st[n] = st.get(n, 0) + w
    except Exception:                                    # noqa: BLE001
        st = {}
    if st:
        return eng.rng.choices(list(st), list(st.values()))[0]
    names = [n for n, c in card_db().items() if c.get("category") == "效果牌" and (c.get("quality") or 0) >= 4
             and not c.get("lord") and "锦囊" not in n and "兑换券" not in n]
    return eng.rng.choice(names) if names else None


class _PrepLevelWatch:
    """"备战阶段中，英雄总等级到达 N 级时"类天赋：引擎没有"等级变化"事件，就在备战阶段会改等级的各个时机检查一次。"""
    def _check(self, eng):
        raise NotImplementedError

    def on_round_start(self, eng, holder):
        self._check(eng)

    def react_prepare(self, eng, holder, **kw):
        self._check(eng)

    def react_deploy(self, eng, holder, **kw):
        self._check(eng)

    def react_merge(self, eng, holder, **kw):
        self._check(eng)

    def react_effect_card(self, eng, holder, **kw):
        self._check(eng)

    def react_coin(self, eng, holder, **kw):
        self._check(eng)


# ============================================================ 经验 / 能量 / 经济
@card("三十而达")
class 三十而达(_PrepLevelWatch):
    """备战阶段中，英雄总等级到达30级时，获得9棋手经验。
    待核对：按"限一次"实现；总等级只算真实等级。"""
    def _check(self, eng):
        if eng.s.flags.get("三十而达_done") or _total_level(eng) < 30:
            return
        eng.s.flags["三十而达_done"] = True
        _add_xp(eng, 9, src="三十而达")

    def on_use(self, eng, t):
        self._check(eng)


@card("透支")
class 透支:
    """能量可用至-15。回合开始时，如果能量为负，本回合无法打出英雄牌。
    能量下限 eng.s.energy_floor = -15（engine.spend 认它）；本回合不能打英雄牌记 flags["round:no_hero_card"]
    （推演打法用 eng.can_play_hero() 判断）。"""
    def on_use(self, eng, t):
        eng.s.energy_floor = min(eng.s.energy_floor, -15)

    def on_round_start(self, eng, holder):
        eng.s.energy_floor = min(eng.s.energy_floor, -15)
        if eng.s.energy < 0:
            eng.s.flags["round:no_hero_card"] = True


@card("伤害转化器")
class 伤害转化器:
    """获得2能量。战斗胜利时，再获得2能量。"""
    def on_use(self, eng, t):
        eng.s.energy += 2

    def on_battle_end(self, eng, holder, win=False):
        if win:
            eng.s.energy += 2


@card("大买卖")
class 大买卖:
    """商店购买额外花费1能量，出售英雄牌额外获得1能量。
    商店购买 +1：eng.s.shop_extra_cost（engine.buy 扣）；出售场上英雄（退场事件）/ 出售手里的英雄牌（sell_card 事件）+1 能量。"""
    def on_use(self, eng, t):
        eng.s.shop_extra_cost += 1

    def react_sell(self, eng, holder, unit=None, **kw):
        if unit is not None and unit.kind == "hero":
            eng.s.energy += 1

    def react_sell_card(self, eng, holder, card="", **kw):
        if category(card) == "英雄牌":
            eng.s.energy += 1


@card("卧薪尝胆")
class 卧薪尝胆:
    """选取天赋后，当你输掉2场战斗时，获得9棋手经验。
    待核对：按"输满 2 场给一次"实现（不是每输 2 场都给）。"""
    def on_battle_end(self, eng, holder, win=False):
        if win or eng.s.flags.get("卧薪尝胆_done"):
            return
        n = eng.s.flags.get("卧薪尝胆_loss", 0) + 1
        eng.s.flags["卧薪尝胆_loss"] = n
        if n >= 2:
            eng.s.flags["卧薪尝胆_done"] = True
            _add_xp(eng, 9, src="卧薪尝胆")


@card("重大里程碑")
class 重大里程碑:
    """棋手到达5级时，获得8能量、1张装备箱和1张高阶效果牌。
    高阶效果牌按效果锦囊开出的 4~5 阶效果牌频率随机。
    待核对：选这个天赋时已经 ≥5 级 → 按立即给一次实现；"高阶效果牌"的具体范围。"""
    def _grant(self, eng):
        if eng.s.flags.get("重大里程碑_done"):
            return
        eng.s.flags["重大里程碑_done"] = True
        eng.s.energy += 8
        eng.gain_card("装备箱")
        c = _high_effect_card(eng)
        if c:
            eng.gain_card(c)

    def on_use(self, eng, t):
        if eng.s.lord_level >= 5:
            self._grant(eng)

    def react_lord_levelup(self, eng, holder, level=0, **kw):
        if level >= 5:
            self._grant(eng)


@card("一分钱一分货")
class 一分钱一分货:
    """获得1张天赋牌等价交换：扣除当前所有能量。根据扣除数量获得不同资源奖励。"""
    def on_use(self, eng, t):
        eng.gain_card("等价交换")


@card("等价交换")
class 等价交换:
    """扣除所有能量：＜2，获得2张英雄牌；2-4，获得1张基础装备牌；≥5，获得1张5阶兑换券，2张效果锦囊牌。
    英雄牌按棋手等级的商店品阶、牌库剩余抽；基础装备随机。
    待核对：效果锦囊的品阶（按合成送锦囊的规则取 min(棋手等级, 5)）；能量为负（透支）时按"扣到 0 以下不变、算 ＜2"。"""
    def on_use(self, eng, t):
        e = eng.s.energy
        if e > 0:
            eng.s.energy = 0
        if e < 2:
            for _ in range(2):
                _draw_hero_card(eng)
        elif e <= 4:
            _gain_basic(eng)
        else:
            eng.gain_card("5阶英雄兑换券")
            tier = min(eng.s.lord_level, 5)
            eng.gain_card(f"{tier}阶效果锦囊")
            eng.gain_card(f"{tier}阶效果锦囊")


@card("人才济济")
class 人才济济:
    """升级所需棋手经验-2，每次升级时获得1张天赋牌人才搜寻。
    经验减免走 xp_need_delta（引擎补丁里的 Engine.level_ups 认它；round_end / 买经验后的升级都按减免后的门槛判）。
    选天赋时如果经验已经够减免后的门槛，立即升级。"""
    def xp_need_delta(self, eng, holder):
        return -2

    def on_use(self, eng, t):
        eng.level_ups()

    def react_lord_levelup(self, eng, holder, level=0, **kw):
        eng.gain_card("人才搜寻")


@card("保持进步")
class 保持进步:
    """获得1张3阶兑换券。回合开始时，获得1棋手经验。满级后，改为获得1能量。"""
    def on_use(self, eng, t):
        eng.gain_card("3阶英雄兑换券")

    def on_round_start(self, eng, holder):
        if eng.s.lord_level >= B.MAX_LORD_LEVEL:
            eng.s.energy += 1
        else:
            _add_xp(eng, 1, src="保持进步")


@card("技高一筹")
class 技高一筹:
    """【逐鹿】英雄每释放6次技能，获得1能量，上限5能量。
    释放次数按 eng.casts(英雄)（卡库面板估的每场放技能次数）累计，余数跨场保留；上限 5 能量按整局累计。
    待核对：放技能次数是估的（没算被打回蓝 / 阵亡时间）。"""
    def on_battle_end(self, eng, holder, win=False):
        got = eng.s.flags.get("技高一筹_energy", 0)
        if got >= 5:
            return
        n = sum(eng.casts(u) for u in eng.s.on_board() if u.kind == "hero" and eng.is_faction(u, "逐鹿"))
        acc = eng.s.flags.get("技高一筹_casts", 0.0) + n
        e = min(int(acc // 6), 5 - got)
        eng.s.flags["技高一筹_casts"] = acc - 6 * int(acc // 6)
        eng.s.flags["技高一筹_energy"] = got + e
        eng.s.energy += e


# ============================================================ 等级
@card("超速成长")
class 超速成长:
    """获得1张天赋牌超速成长：指定1名英雄等级+15。
    （选天赋 = 发 1 张；打出赠送牌 = 指定目标 +15）"""
    def on_use(self, eng, t):
        if t is not None:
            eng.level(t, 15, src="超速成长")
            return
        if not _talent_pick(eng, "超速成长"):
            return False                               # 打出赠送牌却没指定目标
        eng.gain_card("超速成长")


@card("祝你好运")
class 祝你好运:
    """棋手升级时，获得1张天赋牌碰碰运气：指定1名英雄提升5-20等级，随机消耗1-5能量。"""
    def react_lord_levelup(self, eng, holder, level=0, **kw):
        eng.gain_card("碰碰运气")


@card("碰碰运气")
class 碰碰运气:
    """消耗0能量，指定英雄使其随机提升5-20级。（祝你好运送的牌：卡面费用随机 1~5 能量）
    待核对：费用是拿到牌时定的（卡面显示），这里在打出时才随机；能量不够返回 False（不能打）；
    等级 5~20 按整数均匀。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        cost = eng.rng.randint(1, 5)
        if not eng.spend(cost, src="碰碰运气"):
            return False
        eng.level(t, eng.rng.randint(5, 20), src="碰碰运气")


@card("临时增幅")
class 临时增幅:
    """战斗开始时，将20临时等级随机分配给己方英雄。每回合增加5临时等级。
    逐点随机分给场上英雄；池子从选天赋时的 20 起，之后每个回合开始 +5。
    待核对："随机分配"的粒度（逐点 / 随机切份）。"""
    def on_use(self, eng, t):
        eng.s.flags.setdefault("临时增幅", 20)

    def on_round_start(self, eng, holder):
        eng.s.flags["临时增幅"] = eng.s.flags.get("临时增幅", 15) + 5

    def react_battle_start(self, eng, holder, **kw):
        if not _battle_once(eng, "临时增幅"):
            return
        _spread_temp(eng, list(eng.heroes()), eng.s.flags.setdefault("临时增幅", 20), "临时增幅")

    def on_battle_end(self, eng, holder, win=False):
        _battle_reset(eng, "临时增幅")


@card("胜之不武")
class 胜之不武:
    """获得3张天赋牌胜之不武：下场战斗你的英雄+20临时等级，但对敌方棋手造成的伤害-10。
    （选天赋 = 发 3 张；打出赠送牌 = 立即给场上英雄 +20 临时等级，开战时再补给之后才上场的英雄；减伤是战斗效果）
    赠送牌不需要目标，用 _talent_pick 区分选天赋和打出赠送牌。
    待核对：同一回合打出多张是否叠加（按叠加实现）。"""
    def on_use(self, eng, t):
        s = eng.s
        if _talent_pick(eng, "胜之不武"):
            for _ in range(3):
                eng.gain_card("胜之不武")
            return
        s.flags["round:胜之不武"] = s.flags.get("round:胜之不武", 0) + 1
        got = s.flags.setdefault("round:胜之不武_got", {})
        for u in eng.heroes():
            eng.temp(u, 20, src="胜之不武")
            got[u.uid] = got.get(u.uid, 0) + 1

    def react_battle_start(self, eng, holder, **kw):
        n = eng.s.flags.get("round:胜之不武", 0)
        if not n or not _battle_once(eng, "胜之不武"):
            return
        got = eng.s.flags.setdefault("round:胜之不武_got", {})
        for u in eng.heroes():
            miss = n - got.get(u.uid, 0)
            if miss > 0:
                eng.temp(u, 20 * miss, src="胜之不武")
                got[u.uid] = n

    def on_battle_end(self, eng, holder, win=False):
        _battle_reset(eng, "胜之不武")
        eng.s.flags.pop("round:胜之不武", None)
        eng.s.flags.pop("round:胜之不武_got", None)


@card("崛起")
class 崛起:
    """获得1张马可波罗和1张百里守约。共战时，获得3临时等级。若战斗胜利，至多保留他们获得的20临时等级。
    共战 = 两人同时在场：开战各 +3 临时等级；赢了每人把身上的临时等级转成永久，最多 20。
    待核对："至多 20"是每人还是两人合计（按每人）；保留的是全部临时等级还是只限崛起给的（按全部）。"""
    PAIR = ("马可波罗", "百里守约")

    def on_use(self, eng, t):
        for n in self.PAIR:
            eng.take(n)
            eng.gain_card(n)

    def _pair(self, eng):
        us = [eng.s.unit(n) for n in self.PAIR]
        return us if all(u is not None and u.kind == "hero" for u in us) else None

    def react_battle_start(self, eng, holder, **kw):
        us = self._pair(eng)
        if not us or not _battle_once(eng, "崛起"):
            return
        for u in us:
            eng.temp(u, 3, src="崛起")

    def on_battle_end(self, eng, holder, win=False):
        _battle_reset(eng, "崛起")
        us = self._pair(eng)
        if not win or not us:
            return
        for u in us:
            keep = min(max(0.0, u.temp), 20.0)
            if keep > 0:
                eng.level(u, keep, src="崛起(保留临时等级)")


@card("协同一致")
class 协同一致:
    """回合开始时，无阵营英雄获得等于当前棋手等级的等级。
    无阵营 = 卡库 faction 为"无阵营"的英雄牌（弈星 / 牛魔 / 明世隐 / 太乙真人…），且没有被视为某派系。"""
    def on_round_start(self, eng, holder):
        for u in eng.heroes():
            if category(u.name) != "英雄牌" or faction(u.name) not in ("无阵营", ""):
                continue
            if any(f in u.keywords for f in FACTIONS):
                continue
            eng.level(u, eng.s.lord_level, src="协同一致")


@card("强力小弟")
class 强力小弟:
    """你的召唤单位等级+5。每回合开始时，再+5。
    召唤单位 = 召唤物 + 不占人口的赠送单位（始祖熊灵…）。
    待核对：选天赋之后才召唤出来的单位是否也先 +5（按"只加当时在场的"实现，之后靠每回合 +5）。"""
    def _bump(self, eng):
        for u in list(eng.s.units):
            if _is_summon_unit(u):
                eng.level(u, 5, src="强力小弟")

    def on_use(self, eng, t):
        self._bump(eng)

    def on_round_start(self, eng, holder):
        self._bump(eng)


@card("红豆之托")
class 红豆之托(_PrepLevelWatch):
    """获得1张【河洛】英雄王维。备战阶段，场上英雄总等级分别达到100/200/400级时，再获得1张王维。
    每个门槛各给一次；总等级只算真实等级（待核对）。"""
    STEPS = (100, 200, 400)

    def _check(self, eng):
        done = eng.s.flags.setdefault("红豆之托_steps", [])
        tot = _total_level(eng)
        for st in self.STEPS:
            if tot >= st and st not in done:
                done.append(st)
                eng.take("王维")
                eng.gain_card("王维")

    def on_use(self, eng, t):
        eng.take("王维")
        eng.gain_card("王维")
        self._check(eng)


@card("王维")
class 王维:
    """开团：获得空山人语和水穷云起各1张。（技能水穷云起是战斗效果）
    空山人语 / 水穷云起 是转瞬效果牌（卡面 2026-10-15 补录，见下方两张牌的实现）。觉醒翻倍由引擎执行两次。
    待核对：是否每场开团都给（按卡面每场都给）。"""
    def on_open(self, eng, u):
        # 2026-10-15 确认：开战拿到牌、数这场战斗的牺牲；下回合不用就消失；下次战斗拿新牌、重新计数
        eng.s.flags.pop("空山人语_bonus", None)
        eng.s.flags.pop("水穷云起_prep", None)
        eng.gain_card("空山人语")
        eng.gain_card("水穷云起")


@card("空山人语")
class 空山人语:
    """转瞬。指定1名英雄等级+1，每触发1次牺牲效果，额外+1。（卡面：2026-10-15 人工补录）
    按"持有期间累计"实现：在手里时每触发 1 次牺牲累计 +1，打出时一次给（王维开团给牌 → 同场牺牲、下回合打出）。
    待核对：累计是否只算持有期间（还是本局全部牺牲次数）。"""
    in_hand = True

    def react_sacrifice(self, eng, holder, unit=None, **kw):
        if holder is None and _in_hand(eng, "空山人语"):
            eng.s.flags["空山人语_bonus"] = eng.s.flags.get("空山人语_bonus", 0) + 1

    def on_use(self, eng, t):
        n = 1 + eng.s.flags.pop("空山人语_bonus", 0)
        if t is not None:
            eng.level(t, n, src="空山人语")


@card("水穷云起")
class 水穷云起:
    """转瞬。全体英雄的等级+1，每触发2次整备效果，额外+1。（卡面：2026-10-15 人工补录）
    按"持有期间累计"实现：在手里时每 2 次整备 +1（下回合开始的整备在打出前触发，所以通常能吃到一轮整备）。"""
    in_hand = True

    def react_prepare(self, eng, holder, unit=None, **kw):
        if holder is None and _in_hand(eng, "水穷云起"):
            eng.s.flags["水穷云起_prep"] = eng.s.flags.get("水穷云起_prep", 0) + 1

    def on_use(self, eng, t):
        n = 1 + eng.s.flags.pop("水穷云起_prep", 0) // 2
        for u in list(eng.s.heroes()):
            eng.level(u, n, src="水穷云起")


# ============================================================ 发牌（来客 / 精兵 / 其他）
def _visitor(hero):
    class _V:
        __doc__ = f"获得3能量以及1张{hero}。"

        def on_use(self, eng, t):
            eng.s.energy += 3
            eng.take(hero)
            eng.gain_card(hero)
    return _V


for _n, _h in (("河洛来客", "李信"), ("逐鹿来客", "蒙犽"), ("大河来客", "刘邦"),
               ("日落海来客", "雅典娜"), ("三分来客", "小乔")):
    card(_n)(_visitor(_h))


def _elite(fac, label):
    class _E:
        __doc__ = f"卡牌3选1（4阶或5阶的【{label}】英雄）。每个选项先随机 4/5 阶，再按牌库剩余抽。"

        def on_use(self, eng, t):
            _pick_hero_card(eng, (4, 5), fac)
    return _E


for _n, _f, _l in (("河洛精兵", "河洛", "河洛"), ("逐鹿精兵", "逐鹿", "逐鹿"), ("大河精兵", "大河流域", "大河流域"),
                   ("日落海精兵", "日落海", "日落海"), ("三分精兵", "三分之地", "三分")):
    card(_n)(_elite(_f, _l))


@card("紧急行动")
class 紧急行动:
    """获得10点棋手生命值、1张装备箱。（生命值上限 60，engine.add_hp）"""
    def on_use(self, eng, t):
        eng.add_hp(10, src="紧急行动")
        eng.gain_card("装备箱")


@card("王牌认证")
class 王牌认证:
    """战斗开始时，若你至少有3名4-5阶英雄，获得1张2-3阶效果锦囊。
    待核对："2-3 阶"按 2 / 3 阶各半随机；"至少有"按场上英雄算。"""
    def react_battle_start(self, eng, holder, **kw):
        if not _battle_once(eng, "王牌认证"):
            return
        if sum(1 for u in eng.heroes() if quality(u.name) >= 4) >= 3:
            eng.gain_card(f"{eng.rng.choice((2, 3))}阶效果锦囊")

    def on_battle_end(self, eng, holder, win=False):
        _battle_reset(eng, "王牌认证")


@card("前浪的荣光")
class 前浪的荣光:
    """获得2张天赋牌前浪的荣光：指定1名英雄，获得1张高1-2阶的同阵营英雄牌。
    （选天赋 = 发 2 张；打出 = 按目标派系、品阶 +1 或 +2（各半随机，超过 5 阶的不选）按牌库剩余抽 1 张）
    待核对：5 阶英雄做目标时无牌可给（返回 False）；无阵营英雄的"同阵营"按无阵营英雄算。"""
    def on_use(self, eng, t):
        if t is None:
            if not _talent_pick(eng, "前浪的荣光"):
                return False                           # 打出赠送牌却没指定目标
            eng.gain_card("前浪的荣光")
            eng.gain_card("前浪的荣光")
            return
        q, fac = quality(t.name), faction(t.name)
        tiers = [x for x in (q + 1, q + 2) if 1 <= x <= 5]
        if not tiers:
            return False
        eng.rng.shuffle(tiers)
        for tier in tiers:
            x = eng.draw_hero(tier, fac or None)
            if x:
                eng.take(x)
                eng.gain_card(x)
                return
        return False


@card("全军列阵")
class 全军列阵:
    """获得1张天赋牌全军列阵：指定1名英雄，召唤其同阵营英雄上场直到阵容人数上限（不超过商店可购品阶）。
    （选天赋 = 发 1 张；打出 = 按棋手等级的商店品阶概率、目标派系抽英雄直接放上场，到人口上限或没空格为止）
    待核对：召唤上场是否触发登场（按"不是打出英雄牌、不触发"实现）；上场等级按 1 级；同名已在场时是否改为合成（按重抽、不放同名）。"""
    def on_use(self, eng, t):
        if t is None:
            if not _talent_pick(eng, "全军列阵"):
                return False                           # 打出赠送牌却没指定目标
            eng.gain_card("全军列阵")
            return
        fac = faction(t.name) or None
        for _ in range(40):
            if eng.s.pop_used() >= eng.pop_cap() or not eng.free_cells():
                break
            x = eng.draw_hero(None, fac)
            if not x:
                break
            if eng.s.unit(x) is not None:              # 场上已有同名：重抽（不另放一个同名）
                continue
            eng.take(x)
            _place_hero(eng, x)


@card("贤士登门")
class 贤士登门:
    """随机获得1张【逐鹿】英雄牌。回合开始时，再获得1张。
    按棋手等级的商店品阶概率、牌库剩余抽。"""
    def on_use(self, eng, t):
        _draw_hero_card(eng, None, "逐鹿")

    def on_round_start(self, eng, holder):
        _draw_hero_card(eng, None, "逐鹿")


@card("乘胜追击")
class 乘胜追击:
    """战斗胜利时，获得1张高阶推演战术。"""
    def on_battle_end(self, eng, holder, win=False):
        if win:
            eng.gain_card("高阶推演战术")


@card("锦囊妙计")
class 锦囊妙计:
    """获得1张锦囊：使用后，本回合打出的登场英雄牌，会额外触发1次登场效果。（锦囊本身见下）"""
    def on_use(self, eng, t):
        eng.gain_card("锦囊")


@card("锦囊")
class 锦囊:
    """使用后，本回合打出的登场英雄牌，会额外触发1次登场效果。
    打出后进 s.round_cards（eng.attach_round，回合开始清），本回合每次**打出**英雄牌（上场 / 合成，eng.is_card_use()）
    且它有登场效果 → 额外触发 1 次登场（eng.trigger，算"被触发"，不会再连锁）；打出多张叠加。被别的牌触发的登场不算。"""
    def on_use(self, eng, t):
        _attach_round(eng, "锦囊")

    def react_deploy(self, eng, holder, unit=None, **kw):
        if unit is None or unit.kind != "hero" or not eng.is_card_use():
            return
        if not has_keyword(unit.name, "登场"):
            return
        eng.trigger(unit, "deploy")


@card("挂帅上阵")
class 挂帅上阵:
    """获得1张挂帅：指定1名【三分】英雄，使其等级等同于1名最近的【三分】英雄。（挂帅本身在另一批实现）"""
    def on_use(self, eng, t):
        eng.gain_card("挂帅")


# ============================================================ 险中求富
@card("富贵险中求")
class 富贵险中求:
    """获得1张天赋牌一刀穷一刀富：打出后，获得1张基础装备牌及1张险中求富（持有时获胜累积奖励，失败则消失）。"""
    def on_use(self, eng, t):
        eng.gain_card("一刀穷一刀富")


@card("一刀穷一刀富")
class 一刀穷一刀富:
    """获得1张基础装备牌及1张险中求富并开始记录胜场。"""
    def on_use(self, eng, t):
        _gain_basic(eng)
        eng.gain_card("险中求富")
        eng.s.flags["险中求富_wins"] = 0


@card("险中求富")
class 险中求富:
    """获得基础装备牌（当前:0件）。战斗失败销毁此牌并获得2能量。
    持有时（在手牌里，in_hand=True 由引擎补丁挂上）每场战斗结束：赢 = 累计 +1 件；输 = 销毁此牌、+2 能量、累计作废。
    打出 = 按累计件数获得基础装备。
    注意：0 胜时打出也合法（什么都不给），什么时候打由推演打法决定。"""
    in_hand = True

    def on_use(self, eng, t):
        n = eng.s.flags.pop("险中求富_wins", 0)
        _gain_basic(eng, n)

    def on_battle_end(self, eng, holder, win=False):
        s = eng.s
        if "险中求富" not in s.hand:
            return
        if win:
            s.flags["险中求富_wins"] = s.flags.get("险中求富_wins", 0) + 1
        else:
            s.hand.remove("险中求富")
            s.flags.pop("险中求富_wins", None)
            s.energy += 2
            eng.note("销毁", card="险中求富")


# ============================================================ 古币类
@card("多彩古币")
class 多彩古币:
    """获得1张古币。古币只能对非【河洛】英雄主动使用：随机变成其他同阶非【河洛】英雄，等级额外+7。
    选天赋时发 1 张古币。改写古币的部分在 transform()，由本模块重新注册的"古币"在有这个天赋时调用。
    待核对：变身后保留等级 / 装备 / 觉醒进度（按全保留）；"额外 +7"= 古币本身 +2 之外再 +7；
    "只能主动使用"= 李元芳 等被动使用古币不受影响。"""
    def on_use(self, eng, t):
        eng.gain_card("古币")

    @staticmethod
    def transform(eng, t) -> bool:
        if t is None or t not in eng.s.units or t.kind != "hero" or eng.is_faction(t, "河洛"):
            return False
        q = quality(t.name)
        names = [n for n, c in card_db().items() if c.get("category") == "英雄牌" and quality(n) == q
                 and faction(n) != "河洛" and n != t.name]
        if not names:
            return False
        ws = [eng.pool_left(n) for n in names]
        new = eng.rng.choices(names, ws if sum(ws) > 0 else None)[0]
        eng.take(new)
        eng.rename(t, new, src="多彩古币")
        eng.coin(t, 1, src="多彩古币", active=True)
        eng.level(t, 7, src="多彩古币")
        return True


@card("传家宝")
class 传家宝:
    """古币不再附带转瞬。每回合开始时，手牌中古币等级+1，若数量≥8张，则+2级。
    每张手牌古币各记一个加成（flags["传家宝_lv"]，从高到低），回合开始时每张 +1（手里 ≥8 张时 +2）；
    打出古币时用掉加成最高的那张。只认从手牌直接打出的古币（eng.coin(active=True)），李元芳 这类被动使用不算。
    "不再附带转瞬"：引擎目前不让转瞬牌过期，不用处理。
    待核对：精致古币算不算"古币"（按算）；同时持有多张时打出的是哪一张（按加成最高的）。"""
    def _sync(self, eng) -> list:
        n = sum(1 for x in eng.s.hand if x in COIN_CARDS)
        lst = sorted(eng.s.flags.setdefault("传家宝_lv", []), reverse=True)[:n]
        lst += [0] * (n - len(lst))
        eng.s.flags["传家宝_lv"] = lst
        return lst

    def on_round_start(self, eng, holder):
        lst = self._sync(eng)
        inc = 2 if len(lst) >= 8 else 1
        eng.s.flags["传家宝_lv"] = [x + inc for x in lst]

    def coin_bonus(self, eng, holder, target, fine, active=False):
        if not active:
            return 0
        lst = self._sync(eng)
        return lst.pop(0) if lst else 0


@card("古币翻新")
class 古币翻新:
    """获得1张古币。古币牌会额外提供2级。（含被动使用的古币；精致古币按 1 张古币牌 +2）"""
    def on_use(self, eng, t):
        eng.gain_card("古币")

    def coin_bonus(self, eng, holder, target, fine):
        return 2


@card("零钱兑换")
class 零钱兑换:
    """你获得的古币会自动兑换为2张无法提供等级的破损古币，精致古币则兑换为4张。
    获得时兑换：gain_redirect（引擎补丁里的 Engine.gain_card 认它）。
    待核对：选天赋前已经在手里的古币是否也兑换（按兑换，选天赋时一次换掉）；手牌放不下的破损古币丢弃。"""
    RATE = {"古币": 2, "精致古币": 4}

    def gain_redirect(self, eng, holder, name):
        k = self.RATE.get(name)
        return ["破损古币"] * k if k else None

    def on_use(self, eng, t):
        s = eng.s
        for name in self.RATE:
            while name in s.hand:
                s.hand.remove(name)
                eng.gain_card(name)                    # 走兑换


@card("破损古币")
class 破损古币:
    """转瞬 指定1名【河洛】英雄使用，无法提供等级。
    不加等级，但算一次"使用古币"（发 coin 事件：钱生战力 / 古币机 等计数照算）。"""
    def on_use(self, eng, t):
        if t is None or t not in eng.s.units or not eng.is_faction(t, "河洛"):
            return False
        eng.note("古币(破损)", unit=t.name)
        eng.emit("coin", target=t, fine=False)


@card("古币契约")
class 古币契约:
    """你的英雄每触发3次牺牲效果，获得1张古币。
    待核对：觉醒英雄的牺牲（效果翻倍）按触发 1 次计。"""
    def react_sacrifice(self, eng, holder, unit=None, **kw):
        if unit is None or unit.kind != "hero":
            return
        n = eng.s.flags.get("古币契约_n", 0) + 1
        if n >= 3:
            n -= 3
            eng.gain_card("古币")
        eng.s.flags["古币契约_n"] = n


@card("幸运硬币")
class 幸运硬币:
    """打出古币时有概率额外触发1-3次。
    "打出"= 从手牌直接打出的古币（coin 事件 active=True；李元芳 这类被动使用不算）。
    待核对：概率卡面没写、日志里也统计不出 —— PROB 填实测值后生效（None = 不触发，不硬塞一个数）；
    额外次数 1~3 按均匀；额外触发的每次都重新算古币加成（古币翻新 / 武则天…）。"""
    PROB = None

    def react_coin(self, eng, holder, target=None, fine=False, active=False, **kw):
        s = eng.s
        if self.PROB is None or target is None or s.flags.get("_幸运硬币_busy") or not active:
            return
        if eng.rng.random() >= self.PROB:
            return
        s.flags["_幸运硬币_busy"] = True
        try:
            eng.coin(target, eng.rng.randint(1, 3), src="幸运硬币", fine=fine)
        finally:
            s.flags.pop("_幸运硬币_busy", None)


@card("钱生战力")
class 钱生战力:
    """战斗开始时，随机4名【河洛】英雄获得累计使用古币数（选择此天赋后）的临时等级。
    古币使用次数含被动使用（李元芳…）和破损古币。"""
    def on_use(self, eng, t):
        eng.s.flags.setdefault("钱生战力_n", 0)

    def react_coin(self, eng, holder, **kw):
        eng.s.flags["钱生战力_n"] = eng.s.flags.get("钱生战力_n", 0) + 1

    def react_battle_start(self, eng, holder, **kw):
        n = eng.s.flags.get("钱生战力_n", 0)
        if not n or not _battle_once(eng, "钱生战力"):
            return
        for u in eng.pick([x for x in eng.heroes() if eng.is_faction(x, "河洛")], 4):
            eng.temp(u, n, src="钱生战力")

    def on_battle_end(self, eng, holder, win=False):
        _battle_reset(eng, "钱生战力")


# ============================================================ 阿科米亚核心 / 日落海
@card("能量改造")
class 能量改造:
    """阿科米亚核心不再分配等级。回合开始时，核心每拥有30级，将随机触发场上1名英雄的1次整备或登场效果（单回合最多触发100次）。
    类属性 core_no_share：持有时引擎 core_distribute 不分配。触发走 eng.trigger（被触发的登场不算"打出英雄牌"）。
    每次在有整备或登场效果的英雄里随机挑 1 人，两样都有的随机挑一样。
    待核对：随机范围是否包含没有整备 / 登场效果的英雄（按不包含）。"""
    core_no_share = True

    def on_use(self, eng, t):
        return None

    def on_round_start(self, eng, holder):
        n = min(100, int(eng.s.core // 30))
        for _ in range(n):
            cands = [u for u in eng.heroes() if has_keyword(u.name, "整备") or has_keyword(u.name, "登场")]
            if not cands:
                return
            u = eng.rng.choice(cands)
            ts = [x for x, kw in (("prepare", "整备"), ("deploy", "登场")) if has_keyword(u.name, kw)]
            eng.trigger(u, eng.rng.choice(ts))


@card("光棱塔")
class 光棱塔:
    """阿科米亚核心不再分配等级。每秒对1个随机敌方单位造成6次核心等级×1的法术伤害。
    类属性 core_no_share：持有时引擎不做核心开战分配；伤害是战斗效果，不影响养成。"""
    core_no_share = True

    def on_use(self, eng, t):
        return None


@card("场外供能")
class 场外供能:
    """回合开始时，阿科米亚核心获得棋手等级×3的等级。"""
    def on_round_start(self, eng, holder):
        eng.core_add(eng.s.lord_level * 3, src="场外供能")


@card("动能灌注")
class 动能灌注:
    """阿科米亚核心等级+10。你的英雄每合计攻击10次，核心等级+1。
    攻击次数按 Σ 攻速 × 20 秒估（同 pilot 因缘图腾），余数跨场累计。
    待核对：攻速加成 / 阵亡时间未计。"""
    def on_use(self, eng, t):
        eng.core_add(10, src="动能灌注")

    def on_battle_end(self, eng, holder, win=False):
        acc = eng.s.flags.get("动能灌注_att", 0.0) + _attack_count(eng)
        n = int(acc // 10)
        eng.s.flags["动能灌注_att"] = acc - 10 * n
        eng.core_add(n, src="动能灌注")


@card("积极整备")
class 积极整备:
    """你的英雄每触发5次整备效果，随机获得1件日落海专属装备，上限3件。
    待核对：觉醒英雄的整备（执行两次）按触发 1 次计。"""
    def react_prepare(self, eng, holder, unit=None, **kw):
        s = eng.s
        if unit is None or unit.kind != "hero" or not has_keyword(unit.name, "整备"):
            return
        if s.flags.get("积极整备_got", 0) >= 3:
            return
        n = s.flags.get("积极整备_n", 0) + 1
        if n >= 5:
            n -= 5
            if _sunset_equips():
                eng.gain_card(eng.rng.choice(_sunset_equips()))
                s.flags["积极整备_got"] = s.flags.get("积极整备_got", 0) + 1
        s.flags["积极整备_n"] = n


@card("智能装甲")
class 智能装甲:
    """随机获得1件【日落海专属装备】。我方英雄每穿戴1件【日落海专属装备】，其整备效果会额外触发1次。
    额外触发走 eng.trigger（会再发整备事件，积极整备等照常计数），用标记防止额外触发自己再触发。"""
    def on_use(self, eng, t):
        if _sunset_equips():
            eng.gain_card(eng.rng.choice(_sunset_equips()))

    def react_prepare(self, eng, holder, unit=None, **kw):
        s = eng.s
        if unit is None or unit.kind != "hero" or s.flags.get("_智能装甲_busy"):
            return
        if not has_keyword(unit.name, "整备"):
            return
        n = sum(1 for e in unit.equips if e in _sunset_equips())
        if not n:
            return
        s.flags["_智能装甲_busy"] = True
        try:
            for _ in range(n):
                eng.trigger(unit, "prepare")
        finally:
            s.flags.pop("_智能装甲_busy", None)


# ============================================================ 大河 / 图腾
@card("因缘际会")
class 因缘际会:
    """召唤1个1级的因缘图腾。（放在主C 旁边，没有主C 就第一个空格）"""
    def on_use(self, eng, t):
        near = eng.s.unit(eng.s.carry) if eng.s.carry else None
        eng.summon("因缘图腾", near=near, level=1)


@card("图腾启灵")
class 图腾启灵:
    """每回合随机召唤1个图腾，直到回合结束。
    回合开始时在先知图腾 / 因缘图腾 / 往生图腾里随机召唤 1 个（随机英雄旁边，battle_only：战斗结束由引擎移除；
    没经过战斗就到了下回合时，回合开始先收回上回合那个）。
    待核对：随机池是否包含其他图腾；图腾的回合内效果（先知图腾开团给玄微之种等）照常结算。"""
    def _clear(self, eng):
        uid = eng.s.flags.pop("图腾启灵_uid", None)
        if uid is None:
            return
        u = next((x for x in eng.s.units if x.uid == uid), None)
        if u is not None:
            eng.remove(u, src="图腾启灵(回合结束)")

    def on_round_start(self, eng, holder):
        self._clear(eng)
        if not _totems():
            return
        hs = eng.heroes()
        u = eng.summon(eng.rng.choice(_totems()), near=eng.rng.choice(hs) if hs else None, battle_only=True)
        if u is not None and u in eng.s.units:
            eng.s.flags["图腾启灵_uid"] = u.uid


@card("荆棘牢笼")
class 荆棘牢笼:
    """先知图腾获得额外效果：每5秒眩晕最远的1个敌人，持续2秒。如果敌人在眩晕期间阵亡，召唤1只与其等级相同的熊灵。
    战斗效果（控制 + 战斗中召唤熊灵，战后不留），不影响养成。"""


@card("坚不可摧")
class 坚不可摧:
    """战斗开始时，我方先知图腾、因缘图腾和往生图腾获得回合数x100的护盾。战斗效果：护盾，不影响养成。"""


@card("化为熊灵")
class 化为熊灵:
    """战斗中首名【大河】英雄阵亡时，召唤1只场上大河英雄等级之和的熊灵（不计算临时等级），并嘲讽周围2格敌人，持续4秒。
    战斗效果：战斗中召唤的熊灵战后不留（棋盘战后复原；若常驻，每场都会多一只），不影响养成。
    待核对：这只熊灵上场是否算"召唤物上场"（往生图腾吸收）——战斗中召唤的单位按不算。"""


@card("生命绽放")
class 生命绽放:
    """获得1张鬼谷子，玄微之种获得额外效果：每秒回复穿戴者3%生命值。（回血是战斗效果）"""
    def on_use(self, eng, t):
        eng.take("鬼谷子")
        eng.gain_card("鬼谷子")


# ============================================================ 河洛
@card("献祭之赐")
class 献祭之赐:
    """获得1张李信，李信卡牌效果提供的临时等级额外+3。若李信觉醒，敌方英雄触发牺牲时，也会触发己方李信的卡牌效果。
    李信："每当你拥有牺牲效果的英雄阵亡时，李信临时等级+7"（另一批实现）——这里在每次己方牺牲时给场上李信再 +3。
    敌方牺牲不建模（看不到对手）。待核对：觉醒李信（+14）时额外部分是否也翻倍（按翻倍 +6）。"""
    def on_use(self, eng, t):
        eng.take("李信")
        eng.gain_card("李信")

    def react_sacrifice(self, eng, holder, unit=None, **kw):
        if unit is None or unit.kind != "hero":
            return
        for u in eng.heroes():
            if u.name == "李信":
                eng.temp(u, 3 * (2 if u.awakened else 1), src="献祭之赐")


@card("最终守卫")
class 最终守卫:
    """你最后1名【河洛】英雄受到致命伤害时进入无敌状态，并提升50%攻击速度，持续1秒。战斗中你的英雄每触发过1次牺牲效果，无敌时间增加0.25秒。
    战斗效果：无敌 / 攻速，不影响养成。"""


# ============================================================ 逐鹿
@card("鲸吞")
class 鲸吞:
    """获得1张庄周，【逐鹿】英雄每释放6次技能，庄周会吞噬敌方等级最低的1名英雄，永久获得其30%的生命值上限（每场战斗最多吞噬1次）。若庄周觉醒，吞噬上限变为3次。
    只实现发庄周；吞噬加的是生命值上限（战斗属性），不影响养成。"""
    def on_use(self, eng, t):
        eng.take("庄周")
        eng.gain_card("庄周")


@card("逆袭之路")
class 逆袭之路:
    """当你排名不是第1或第2名时，你的所有单位伤害+10%，最大生命值+20%。战斗效果：伤害 / 生命，不影响养成。"""


@card("瞬影连斩")
class 瞬影连斩:
    """闪现英雄击败敌方单位时，会立刻闪现至最近的敌人处，并无敌0.5秒。战斗效果：位移 / 无敌，不影响养成。"""


@card("绝对领域")
class 绝对领域:
    """棋手每次受到的伤害减少3点。战斗效果：棋手减伤，不影响养成。"""


@card("小身体大能量")
class 小身体大能量:
    """战斗开始时，英雄模型缩小，攻速+60%，暴击率+40%，生命值上限-20%。战斗效果：攻速 / 暴击 / 生命，不影响养成。"""


# ============================================================ 三分
@card("乐善好施")
class 乐善好施:
    """你的英雄触发登场效果时，随机2名【三分】英雄等级+2。
    只在有"登场："效果的英雄登场时触发（每次登场事件 1 次）；随机 2 名可以包含登场的英雄本人。
    待核对：觉醒英雄的登场（执行两次）按 1 次计。"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if unit is None or unit.kind != "hero" or not has_keyword(unit.name, "登场"):
            return
        for u in eng.pick([x for x in eng.heroes() if eng.is_faction(x, "三分之地")], 2):
            eng.level(u, 2, src="乐善好施")


@card("三三得三")
class 三三得三:
    """战斗开始时，若你场上【三分】英雄人数刚好是3的倍数，他们分别获得33临时等级。"""
    def react_battle_start(self, eng, holder, **kw):
        if not _battle_once(eng, "三三得三"):
            return
        sf = [x for x in eng.heroes() if eng.is_faction(x, "三分之地")]
        if sf and len(sf) % 3 == 0:
            for u in sf:
                eng.temp(u, 33, src="三三得三")

    def on_battle_end(self, eng, holder, win=False):
        _battle_reset(eng, "三三得三")


@card("怒涛")
class 怒涛:
    """【三分】英雄获得牺牲：大乔灵体现身，对敌人施加1次巨型海浪打击。
    养成上的作用 = 让【三分】英雄拥有"牺牲"（牺牲反应器 / 古币契约 等因此触发更多）；海浪打击是战斗效果。"""
    def grant_keywords(self, eng, holder):
        for u in eng.heroes():
            if eng.is_faction(u, "三分之地"):
                u.keywords.add("牺牲")


@card("膨化生命")
class 膨化生命:
    """战斗开始时，你的【三分】英雄额外获得100%生命值上限（仅提升上限），且生命恢复和护盾效果提升20%。战斗效果：生命 / 护盾，不影响养成。"""


@card("一身是胆")
class 一身是胆:
    """获得1张赵云，赵云在释放技能前，会额外释放1个技能惊雷之龙，且赵云生命值每次经过70%/30%档位时，会恢复全部法力值。
    只实现发赵云；技能部分是战斗效果。"""
    def on_use(self, eng, t):
        eng.take("赵云")
        eng.gain_card("赵云")


@card("惊雷之龙")
class 惊雷之龙:
    """赵云向敌人冲锋，造成150%物攻的物理伤害，并获得50%物理吸血，持续2秒。战斗效果：伤害 / 吸血，不影响养成。"""


@card("卧龙威势")
class 卧龙威势:
    """获得1张诸葛亮，你的英雄释放技能时，诸葛亮周围1格的其他【三分】英雄获得10点法力值。
    只实现发诸葛亮；法力是战斗效果。"""
    def on_use(self, eng, t):
        eng.take("诸葛亮")
        eng.gain_card("诸葛亮")


@card("控制感应")
class 控制感应:
    """你的英雄施加控制效果时，你的【三分】英雄获得20点法术攻击力。战斗效果：法术攻击，不影响养成。"""


@card("搏战")
class 搏战:
    """你前两排的【三分】英雄生命值50%以下时，每秒恢复5%生命值，并提升50%攻速。战斗效果：回血 / 攻速，不影响养成。"""


# ============================================================ 古币（多彩古币改写主动使用）
from . import pilot as _pilot   # noqa: E402


@card("古币")
class 古币(_pilot.古币):
    """转瞬 指定1名【河洛】英雄等级+2。
    有天赋"多彩古币"时，主动使用只能对非【河洛】英雄：随机变成其他同阶非【河洛】英雄，等级额外+7（多彩古币.transform）。
    其余情况照 pilot 的实现；被动使用（李元芳 等直接调 eng.coin）不受影响。"""
    def on_use(self, eng, t):
        if "多彩古币" in eng.s.talents:
            return 多彩古币.transform(eng, t)
        return super().on_use(eng, t)
