"""第 3 批：天赋牌（经济 / 送英雄牌 / 送装备 / 等级触发 / 预言 / 拍卖重现…）及其赠送牌、召唤物。

约定（本模块内统一）：
- 天赋被选中：推演打法把天赋名放进 eng.s.talents，再调 on_use(eng, None)。
- "获得 N 张天赋牌 X"且赠送牌与天赋**同名**（五福临门 / 复印机 / 终极复印机 / 限制性进化 / 自适应武装 / 援军）：
  需要目标的牌按 target 区分（None = 选中天赋 → 发牌；有目标 = 打出赠送牌，同 pilot 的「加速成长」）；
  不需要目标的（五福临门）按 flags["gift:<名>"] 里还没打出的赠送牌张数区分。
- "打出英雄牌" = react_deploy 且 eng.is_card_use()（上场 / 合成 / 觉醒回手再打出；被唤醒 / 曹操这类触发的登场不算）。
- "战斗开始时"的天赋：用 react_battle_start（每个单位发一次）+ 本回合标记只结算一次。
- 战斗数值（伤害 / 护盾 / 攻速 / 法力 / 控制 / 棋手血量）不实现，只写 docstring。
"""
from __future__ import annotations

import json
from functools import lru_cache

from agent.engine import board as B
from ..engine import FACTION_ALIAS, ROOT, card, card_db, faction, has_keyword, quality

NO_FACTION = ("", "无阵营")
CN_NUM = {1: "一", 2: "二", 3: "三"}


# ============================================================ 辅助函数
def _once_round(eng, key: str) -> bool:
    """本回合第一次返回 True（"round:" 前缀的标记在回合开始被引擎清掉）。"""
    k = "round:" + key
    if eng.s.flags.get(k):
        return False
    eng.s.flags[k] = True
    return True


def _max_tier(eng) -> int:
    """商店能出的最高品阶 = 棋手等级（封顶 5）。"""
    return B.meta.MAX_TIER_BY_LEVEL.get(eng.s.lord_level, min(max(1, eng.s.lord_level), 5))


def _draw_tiers(eng, tiers, fac: str | None = None, pred=None):
    """在若干品阶里按牌库剩余张数抽 1 张英雄牌（fac 限派系，pred(卡名) 额外过滤）。"""
    fac = FACTION_ALIAS.get(fac, fac) if fac else None
    names = [n for n, c in card_db().items() if c.get("category") == "英雄牌" and quality(n) in tiers
             and (not fac or faction(n) == fac) and (pred is None or pred(n))]
    ws = [eng.pool_left(n) for n in names]
    if not names or sum(ws) <= 0:
        return None
    return eng.rng.choices(names, ws)[0]


def _gain_hero(eng, name) -> bool:
    """获得 1 张英雄牌：记牌库被拿走 + 进手牌。"""
    if not name:
        return False
    eng.take(name)
    eng.gain_card(name)
    return True


def _gain_gift(eng, name: str, n: int) -> None:
    """发 n 张赠送牌，并记下还没打出的张数（同名天赋 / 赠送牌靠它区分"选天赋"和"打出赠送牌"）。"""
    for _ in range(n):
        before = len(eng.s.hand)
        eng.gain_card(name)
        if len(eng.s.hand) > before:
            k = "gift:" + name
            eng.s.flags[k] = eng.s.flags.get(k, 0) + 1


def _use_gift(eng, name: str) -> None:
    k = "gift:" + name
    if eng.s.flags.get(k, 0) > 0:
        eng.s.flags[k] -= 1


def _played_hero(eng, unit) -> bool:
    """这次登场是"打出英雄牌"：英雄单位，且不是被别的效果触发的（eng.is_card_use）。"""
    return unit is not None and getattr(unit, "kind", "") == "hero" and eng.is_card_use()


def _lord_hp(eng, n: int) -> None:
    """棋手生命值变化（engine.add_hp，上限 60）。"""
    eng.add_hp(n)


def _add_xp(eng, n: int) -> None:
    eng.add_xp(n)


@lru_cache(maxsize=1)
def _basic_equips() -> tuple:
    """普通装备牌 = 8 件基础装备（穿上后四选一铸造成普通装备）。"""
    try:
        rec = json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8")).get("recipes") or {}
        return tuple(rec)
    except Exception:                                    # noqa: BLE001
        return ()


@lru_cache(maxsize=1)
def _recipes() -> dict:
    try:
        return json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8")).get("recipes") or {}
    except Exception:                                    # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def _equip_fit() -> dict:
    try:
        return json.loads((ROOT / "data/equip_relations.json").read_text(encoding="utf-8")).get("fit") or {}
    except Exception:                                    # noqa: BLE001
        return {}


def _random_basic(eng):
    b = _basic_equips()
    return eng.pick(list(b), 1)[0] if b else None


class _Passive:
    """持续效果的天赋：选中时没有立即效果（on_use 什么都不做，免得被当成"未实现"）。"""
    def on_use(self, eng, t):
        return None


# ============================================================ 经济 / 能量
@card("小电池包")
class 小电池包:
    """获得5能量。"""
    def on_use(self, eng, t):
        eng.s.energy += 5


@card("电池包")
class 电池包:
    """获得8能量。"""
    def on_use(self, eng, t):
        eng.s.energy += 8


@card("大电池包")
class 大电池包:
    """获得10能量。"""
    def on_use(self, eng, t):
        eng.s.energy += 10


@card("结伴")
class 结伴:
    """获得2能量以及2张3阶英雄兑换券。"""
    def on_use(self, eng, t):
        eng.s.energy += 2
        eng.gain_card("3阶英雄兑换券")
        eng.gain_card("3阶英雄兑换券")


@card("极限操作")
class 极限操作:
    """棋手生命值-15，获得15能量。（engine.add_hp）"""
    def on_use(self, eng, t):
        _lord_hp(eng, -15)
        eng.s.energy += 15


@card("存款利息")
class 存款利息:
    """获得3能量。回合结束时，你每有7能量，额外获得1能量。
    "回合结束" = engine.round_end 的 on_round_end（下回合收入到账之前）。"""
    def on_use(self, eng, t):
        eng.s.energy += 3

    def on_round_end(self, eng, holder):
        if holder is None:
            eng.s.energy += max(0, eng.s.energy) // 7


@card("整装待发")
class 整装待发:
    """若回合内没有打出英雄牌，下回合开始时，获得2能量。
    待核对：选天赋的那个回合也算（从选中的回合起判）；"打出英雄牌"含合成（打出同名英雄牌）。"""
    def on_use(self, eng, t):
        eng.s.flags["整装待发_pick"] = eng.s.round

    def react_deploy(self, eng, holder, unit=None, **kw):
        if _played_hero(eng, unit):
            eng.s.flags["整装待发_last"] = eng.s.round

    def on_round_start(self, eng, holder):
        prev = eng.s.round - 1
        if prev >= 1 and prev >= eng.s.flags.get("整装待发_pick", 0) and eng.s.flags.get("整装待发_last") != prev:
            eng.s.energy += 2


@card("常胜将军")
class 常胜将军(_Passive):
    """若你至少2连胜，战斗胜利时获得2能量和4棋手生命值。
    待核对：连胜从选中天赋起算、含本场（本场胜利后连胜 ≥2 即生效）。棋手血量 engine.add_hp。"""
    def on_battle_end(self, eng, holder, win=False):
        if holder is not None:
            return
        k = "常胜将军_streak"
        if not win:
            eng.s.flags[k] = 0
            return
        eng.s.flags[k] = eng.s.flags.get(k, 0) + 1
        if eng.s.flags[k] >= 2:
            eng.s.energy += 2
            _lord_hp(eng, 4)


@card("消费返利")
class 消费返利(_Passive):
    """每消耗15能量，获得1张加速成长：指定1名英雄等级+5。
    消耗能量 = engine.spend 发的 spend 事件（买牌 / 刷新 / 升级 / 可购买天赋都走 eng.spend）。"""
    def react_spend(self, eng, holder, n=0, **kw):
        if holder is not None:
            return
        acc = eng.s.flags.get("消费返利_acc", 0) + n
        while acc >= 15:
            acc -= 15
            eng.gain_card("加速成长")
        eng.s.flags["消费返利_acc"] = acc


@card("扩编名额")
class 扩编名额:
    """获得1个可购买天赋：花费8能量，可上阵人数+1。可购买2次。
    购买是推演打法的动作 —— 调本类的 buy(eng)（eng.spend 扣 8 能量且还有次数才成功）。"""
    def on_use(self, eng, t):
        eng.s.flags["扩编名额_left"] = eng.s.flags.get("扩编名额_left", 0) + 2

    def buy(self, eng) -> bool:
        if eng.s.flags.get("扩编名额_left", 0) <= 0 or not eng.spend(8, src="扩编名额"):
            return False
        eng.s.flags["扩编名额_left"] -= 1
        eng.s.pop_bonus += 1
        return True


@card("战术模组")
class 战术模组:
    """可上阵人数+1。"""
    def on_use(self, eng, t):
        eng.s.pop_bonus += 1


@card("节能商店")
class 节能商店(_Passive):
    """每次手动刷新商店后，总有1张商店卡牌所需能量-1。
    手动刷新 = engine.refresh_shop 发的 refresh 事件；引擎不模拟商店价格，只记本回合降价次数 flags["round:节能商店"]，给商店模块用。"""
    def react_refresh(self, eng, holder, **kw):
        if holder is None:
            eng.s.flags["round:节能商店"] = eng.s.flags.get("round:节能商店", 0) + 1


@card("破格")
class 破格:
    """回合开始商店自动刷新时，会出现1个高1阶英雄牌。
    接口缺口：商店生成不在本引擎里；只记 flags["破格"]=True，交给商店模块（高 1 阶 = 当前最高可刷品阶 +1，封顶 5，待核对）。"""
    def on_use(self, eng, t):
        eng.s.flags["破格"] = True


# ============================================================ 棋手经验 / 棋手等级
@card("里程碑")
class 里程碑:
    """棋手到达5级时，获得6能量、1张装备箱。
    待核对：选天赋时棋手已经 ≥5 级 —— 按"立即生效一次"实现；限一次。"""
    def _fire(self, eng):
        if eng.s.flags.get("里程碑_done"):
            return
        eng.s.flags["里程碑_done"] = True
        eng.s.energy += 6
        eng.gain_card("装备箱")

    def on_use(self, eng, t):
        if eng.s.lord_level >= 5:
            self._fire(eng)

    def react_lord_levelup(self, eng, holder, level=0, **kw):
        if holder is None and level >= 5:
            self._fire(eng)


@card("先苦后甜")
class 先苦后甜(_Passive):
    """选取天赋后，当你输掉2场战斗时，获得6棋手经验。（限一次：第 2 场败仗时结算）"""
    def on_battle_end(self, eng, holder, win=False):
        if holder is not None or win or eng.s.flags.get("先苦后甜_done"):
            return
        n = eng.s.flags.get("先苦后甜_lose", 0) + 1
        eng.s.flags["先苦后甜_lose"] = n
        if n >= 2:
            eng.s.flags["先苦后甜_done"] = True
            _add_xp(eng, 6)


class _LevelWatch:
    """等级触发（引擎没有"等级变化"事件）：在备战阶段的各个时机轮询检查。"""
    def _check(self, eng):
        raise NotImplementedError

    def on_round_start(self, eng, holder):
        if holder is None:
            self._check(eng)

    def react_prepare(self, eng, holder, **kw):
        if holder is None:
            self._check(eng)

    def react_deploy(self, eng, holder, **kw):
        if holder is None:
            self._check(eng)

    def react_merge(self, eng, holder, **kw):
        if holder is None:
            self._check(eng)

    def react_coin(self, eng, holder, **kw):
        if holder is None:
            self._check(eng)

    def react_effect_card(self, eng, holder, **kw):
        if holder is None:
            self._check(eng)

    def react_sell(self, eng, holder, **kw):
        if holder is None:
            self._check(eng)


@card("三十而立")
class 三十而立(_LevelWatch):
    """备战阶段中，英雄总等级到达30级时，获得6棋手经验。
    待核对："英雄总等级"按场上英雄（不含召唤物）的 永久 + 临时 等级求和；限一次；选天赋时已满 30 → 立即生效。
    接口缺口：没有"等级变化"事件，只在备战阶段的各时机（回合开始 / 整备 / 上场 / 合成 / 古币 / 效果牌 / 出售）检查。"""
    def on_use(self, eng, t):
        self._check(eng)

    def _check(self, eng):
        if eng.s.flags.get("三十而立_done"):
            return
        if sum(u.lv() for u in eng.heroes()) >= 30:
            eng.s.flags["三十而立_done"] = True
            _add_xp(eng, 6)


@card("成年礼")
class 成年礼(_LevelWatch):
    """己方英雄30级时，随机获得2张其阵营的英雄牌（已达30级的英雄，再升级1次可获得）。
    待核对：按永久等级判断；每名英雄限一次；"其阵营的英雄牌"按商店品阶概率 + 牌库剩余抽；无阵营英雄不发。
    接口缺口：没有"等级变化"事件 —— 备战各时机 + 战斗结束时轮询。"""
    def on_use(self, eng, t):
        for u in eng.heroes():                 # 选中时已满 30 级的：记下当前等级，再升一次才给
            if u.level >= 30:
                u.counters["成年礼_base"] = u.level

    def on_battle_end(self, eng, holder, win=False):
        if holder is None:
            self._check(eng)

    def _check(self, eng):
        for u in eng.heroes():
            if u.level < 30 or u.counters.get("成年礼_done"):
                continue
            base = u.counters.get("成年礼_base")
            if base is not None and u.level <= base:
                continue
            u.counters["成年礼_done"] = True
            fac = eng.faction_of(u)
            if fac in NO_FACTION:
                continue
            for _ in range(2):
                _gain_hero(eng, eng.draw_hero(None, fac))


# ============================================================ 送英雄牌
def _newcomer(fac):
    class _N:
        __doc__ = f"""获得2能量，以及随机1张2-3阶【{fac}】英雄牌。（2、3 阶合在一起按牌库剩余张数抽）"""

        def on_use(self, eng, t):
            eng.s.energy += 2
            _gain_hero(eng, _draw_tiers(eng, (2, 3), fac))
    return _N


for _nm, _fac in (("三分新援", "三分之地"), ("日落海新援", "日落海"), ("大河新援", "大河流域"),
                  ("逐鹿新援", "逐鹿"), ("河洛新援", "河洛")):
    card(_nm)(_newcomer(_fac))


@card("双人游")
class 双人游:
    """随机获得2张2-3阶英雄牌。"""
    def on_use(self, eng, t):
        for _ in range(2):
            _gain_hero(eng, _draw_tiers(eng, (2, 3)))


@card("五福临门")
class 五福临门:
    """获得1张天赋牌五福临门：随机获得5张英雄牌。
    选中天赋 → 给 1 张同名赠送牌；打出赠送牌 → 随机 5 张英雄牌（按棋手等级的商店品阶概率 + 牌库剩余，待核对）。
    同名区分：flags["gift:五福临门"] > 0 时视为打出赠送牌。"""
    def on_use(self, eng, t):
        if eng.s.flags.get("gift:五福临门", 0) > 0:
            _use_gift(eng, "五福临门")
            for _ in range(5):
                _gain_hero(eng, eng.draw_hero())
            return
        _gain_gift(eng, "五福临门", 1)


@card("补给线：开团")
class 补给线开团:
    """获得3张开团效果英雄牌。
    待核对：品阶不限，在所有带「开团」的英雄里按牌库剩余张数抽。"""
    def on_use(self, eng, t):
        for _ in range(3):
            _gain_hero(eng, _draw_tiers(eng, (1, 2, 3, 4, 5), pred=lambda n: has_keyword(n, "开团")))


@card("控能达人")
class 控能达人(_Passive):
    """战斗开始时，若当前能量≤3，随机获得1张3-4阶英雄牌。（每场战斗一次）"""
    def react_battle_start(self, eng, holder, **kw):
        if holder is None and _once_round(eng, "控能达人") and eng.s.energy <= 3:
            _gain_hero(eng, _draw_tiers(eng, (3, 4)))


@card("凯旋征召")
class 凯旋征召(_Passive):
    """战斗胜利时，获得1张你等级最高的英雄牌。（等级 = 永久等级；并列随机）"""
    def on_battle_end(self, eng, holder, win=False):
        if holder is None and win:
            h = eng.highest(eng.heroes(), 1)
            if h:
                _gain_hero(eng, h[0].name)


@card("人才市场")
class 人才市场:
    """获得3张天赋牌人才搜寻。"""
    def on_use(self, eng, t):
        for _ in range(3):
            eng.gain_card("人才搜寻")


@card("人才搜寻")
class 人才搜寻:
    """卡牌3选1（商店可购的最高品阶英雄）。（最高品阶 = 棋手等级，封顶 5；3 个不同的英雄按牌库剩余抽）"""
    def on_use(self, eng, t):
        tier = _max_tier(eng)
        opts = []
        for _ in range(20):
            x = eng.draw_hero(tier)
            if x and x not in opts:
                opts.append(x)
            if len(opts) >= 3:
                break
        pick = eng.choose("hero", opts, tier=tier)
        if pick:
            _gain_hero(eng, pick)


@card("二象性")
class 二象性:
    """获得1张天赋牌二象坍缩：随机生效1个获取英雄牌的效果（详情查看关联卡牌）。"""
    def on_use(self, eng, t):
        eng.gain_card("二象坍缩")


@card("二象坍缩")
class 二象坍缩:
    """指定1名英雄，获得3张同阶英雄牌，或2张同阵营的当前最高阶英雄牌。
    两个效果随机生效 1 个（各 50%）。"当前最高阶" = 商店可出的最高品阶（棋手等级，封顶 5）。
    待核对：目标无阵营时第二个效果怎么算 —— 按"只能生效第一个"实现；两个效果的概率是否均等。"""
    def on_use(self, eng, t):
        if t is None or t.kind != "hero":
            return False
        fac = eng.faction_of(t)
        if fac in NO_FACTION or eng.rng.random() < 0.5:
            for _ in range(3):
                _gain_hero(eng, eng.draw_hero(quality(t.name) or 1))
        else:
            for _ in range(2):
                _gain_hero(eng, eng.draw_hero(_max_tier(eng), fac))


@card("援军")
class 援军:
    """获得1张援军：指定1名英雄，随机获得4张同阵营的英雄牌。
    选中天赋（无目标）→ 给 1 张同名赠送牌；打出赠送牌（有目标）→ 4 张同阵营英雄牌。
    待核对：品阶按商店概率抽；目标是无阵营英雄时视为不合法。"""
    def on_use(self, eng, t):
        if t is None:
            _gain_gift(eng, "援军", 1)
            return
        fac = eng.faction_of(t)
        if t.kind != "hero" or fac in NO_FACTION:
            return False
        _use_gift(eng, "援军")
        for _ in range(4):
            _gain_hero(eng, eng.draw_hero(None, fac))


@card("复印机")
class 复印机:
    """获得2张天赋牌复印机：指定1名1-3阶英雄，获得该英雄牌。
    选中天赋（无目标）→ 2 张同名赠送牌；打出赠送牌（有目标）→ 获得目标的英雄牌。"""
    def on_use(self, eng, t):
        if t is None:
            _gain_gift(eng, "复印机", 2)
            return
        if t.kind != "hero" or not 1 <= quality(t.name) <= 3:
            return False
        _use_gift(eng, "复印机")
        _gain_hero(eng, t.name)


@card("终极复印机")
class 终极复印机:
    """获得2张天赋牌终极复印机：指定1名英雄，获得该英雄牌。
    选中天赋（无目标）→ 2 张同名赠送牌；打出赠送牌（有目标）→ 获得目标的英雄牌。"""
    def on_use(self, eng, t):
        if t is None:
            _gain_gift(eng, "终极复印机", 2)
            return
        if t.kind != "hero":
            return False
        _use_gift(eng, "终极复印机")
        _gain_hero(eng, t.name)


@card("限制性进化")
class 限制性进化:
    """获得2张天赋牌限制性进化：指定移除1名1~2阶英雄，获得高2阶的随机英雄牌，且上阵时为原等级+10。
    选中天赋（无目标）→ 2 张同名赠送牌；打出赠送牌（有目标）→ 移除目标，得到 1 张高 2 阶随机英雄牌（按牌库剩余抽）。
    "上阵时为原等级+10"：手牌没有逐张状态，记在 flags["限制性进化_pending"]（[卡名, 原等级]），
    该卡**新上场**（不是合成）时把等级补到 原等级+10。
    待核对："原等级"按永久等级；拿去合成同名时是否也生效（现在不生效）。"""
    def on_use(self, eng, t):
        if t is None:
            _gain_gift(eng, "限制性进化", 2)
            return
        if t.kind != "hero" or t not in eng.s.units or not 1 <= quality(t.name) <= 2:
            return False
        _use_gift(eng, "限制性进化")
        lv, q = t.level, quality(t.name)
        eng.remove(t, src="限制性进化")
        x = eng.draw_hero(q + 2)
        if x:
            _gain_hero(eng, x)
            eng.s.flags.setdefault("限制性进化_pending", []).append([x, lv])
        if "限制性进化" not in eng.s.talents:          # 没走选天赋直接打出的：挂一个标记，上场补等级靠它
            eng.s.talents.append("限制性进化")

    def react_deploy(self, eng, holder, unit=None, **kw):
        if holder is not None or not _played_hero(eng, unit) or unit.awaken > 0 or unit.counters.get("限制性进化"):
            return
        pend = eng.s.flags.get("限制性进化_pending") or []
        for i, (nm, lv) in enumerate(pend):
            if nm == unit.name:
                pend.pop(i)
                unit.counters["限制性进化"] = True
                eng.level(unit, lv + 10 - 1, src="限制性进化")      # 新上场 1 级 → 原等级 +10
                return


@card("如法炮制")
class 如法炮制(_Passive):
    """每回合首次打出英雄牌时，获得1张同阶的英雄牌（不可超过棋手等级）。（品阶 = min(打出的品阶, 棋手可刷最高品阶)）"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if holder is None and _played_hero(eng, unit) and _once_round(eng, "如法炮制"):
            tier = min(quality(unit.name) or 1, _max_tier(eng))
            _gain_hero(eng, eng.draw_hero(tier))


@card("同路人")
class 同路人(_Passive):
    """同一回合内，每打出4张同阵营英雄牌，随机获得1张该阵营英雄牌。
    待核对：送的牌品阶按商店概率抽；无阵营英雄不计数。"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if holder is not None or not _played_hero(eng, unit):
            return
        fac = eng.faction_of(unit)
        if fac in NO_FACTION:
            return
        k = "round:同路人:" + fac
        eng.s.flags[k] = eng.s.flags.get(k, 0) + 1
        if eng.s.flags[k] % 4 == 0:
            _gain_hero(eng, eng.draw_hero(None, fac))


@card("瓶中人")
class 瓶中人:
    """获得1张天赋牌契约瓶：获胜时，累积3张随机英雄牌；失败时，损失所累积的一半英雄牌。
    契约瓶拿在手里时，累积由本天赋在战斗结束时代为结算（手牌没有钩子）。"""
    def on_use(self, eng, t):
        eng.gain_card("契约瓶")

    def on_battle_end(self, eng, holder, win=False):
        if holder is None and "契约瓶" in eng.s.hand:
            契约瓶.accumulate(eng, win)


@card("契约瓶")
class 契约瓶:
    """获胜时，累积3张随机英雄牌；失败时，损失一半。打出后获得累积的英雄牌（当前0张）。
    累积张数记在 flags["契约瓶_acc"]；打出时按商店品阶概率 + 牌库剩余抽出具体的牌。
    待核对："损失一半"向下取整（7 张损失 3 张）；具体是哪些牌在打出时才决定。"""
    @staticmethod
    def accumulate(eng, win: bool) -> None:
        n = eng.s.flags.get("契约瓶_acc", 0)
        n = n + 3 if win else n - n // 2
        eng.s.flags["契约瓶_acc"] = n

    def on_use(self, eng, t):
        n = eng.s.flags.pop("契约瓶_acc", 0)
        for _ in range(n):
            _gain_hero(eng, eng.draw_hero())


# ============================================================ 加等级（打出英雄牌 / 战斗开始 / 出售）
@card("战争头牌")
class 战争头牌:
    """每回合获得1个品阶数字，打出相同品阶的英雄牌时，额外+4等级。
    待核对：品阶数字在 1~5 里均匀随机；选中天赋的回合立即给一个数字；每次打出（含合成）都生效。"""
    def _roll(self, eng):
        eng.s.flags["战争头牌_tier"] = eng.rng.randint(1, 5)

    def on_use(self, eng, t):
        self._roll(eng)

    def on_round_start(self, eng, holder):
        if holder is None:
            self._roll(eng)

    def react_deploy(self, eng, holder, unit=None, **kw):
        if holder is None and _played_hero(eng, unit) and quality(unit.name) == eng.s.flags.get("战争头牌_tier"):
            eng.level(unit, 4, src="战争头牌")


@card("中坚力量")
class 中坚力量(_Passive):
    """打出3阶英雄牌，额外+4级。（含合成时打出的同名牌）"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if holder is None and _played_hero(eng, unit) and quality(unit.name) == 3:
            eng.level(unit, 4, src="中坚力量")


@card("群英荟萃")
class 群英荟萃(_Passive):
    """回合中每打出3张不同品阶的英雄牌，全体英雄等级+3。
    待核对：凑齐 3 个不同品阶就结算一次并清空，重新开始凑。"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if holder is not None or not _played_hero(eng, unit):
            return
        k = "round:群英荟萃"
        tiers = eng.s.flags.setdefault(k, [])
        q = quality(unit.name)
        if q not in tiers:
            tiers.append(q)
        if len(tiers) >= 3:
            eng.s.flags[k] = []
            for h in eng.heroes():
                eng.level(h, 3, src="群英荟萃")


def _top_tier(eng):
    hs = [u for u in eng.s.on_board() if u.kind == "hero"]
    if not hs:
        return None
    m = max(quality(u.name) for u in hs)
    return eng.pick([u for u in hs if quality(u.name) == m], 1)[0]


@card("随机成长")
class 随机成长(_Passive):
    """战斗开始时，你品阶最高的1名英雄等级永久+4。（并列随机；每场战斗一次）"""
    def react_battle_start(self, eng, holder, **kw):
        if holder is None and _once_round(eng, "随机成长"):
            eng.level(_top_tier(eng), 4, src="随机成长")


@card("随机进化")
class 随机进化(_Passive):
    """战斗开始时，你品阶最高的1名英雄等级永久+6。（并列随机；每场战斗一次）"""
    def react_battle_start(self, eng, holder, **kw):
        if holder is None and _once_round(eng, "随机进化"):
            eng.level(_top_tier(eng), 6, src="随机进化")


def _abacus(n, nm):
    class _A(_Passive):
        __doc__ = f"""出售场上英雄时，随机1名英雄等级+{n}。（随机范围不含被卖掉的那个）"""

        def react_sell(self, eng, holder, unit=None, **kw):
            if holder is not None or not _played_hero(eng, unit):
                return
            r = eng.pick([x for x in eng.heroes() if x is not unit], 1)
            if r:
                eng.level(r[0], n, src=nm)
    return _A


card("银算盘")(_abacus(2, "银算盘"))
card("金算盘")(_abacus(3, "金算盘"))


# ============================================================ 移除换资源
@card("好聚好散")
class 好聚好散:
    """获得1张天赋牌体面离场：指定移除一名英雄，获得能量。此英雄0/1/2/3觉醒进度时，获得3/6/9/12能量。"""
    def on_use(self, eng, t):
        eng.gain_card("体面离场")


@card("体面离场")
class 体面离场:
    """指定移除一名英雄，获得能量。此英雄0/1/2/3觉醒进度时，获得3/6/9/12能量。（移除 ≠ 出售，不触发退场）"""
    def on_use(self, eng, t):
        if t is None or t.kind != "hero" or t not in eng.s.units:
            return False
        eng.s.energy += (3, 6, 9, 12)[max(0, min(3, t.awaken))]
        eng.remove(t, src="体面离场")


@card("一个响指")
class 一个响指:
    """获得1张天赋牌听天由命：消除等级最低的4个英雄，每名英雄会提供一定资源。"""
    def on_use(self, eng, t):
        eng.gain_card("听天由命")


@card("听天由命")
class 听天由命:
    """消除场上4名等级最低的英雄。每消除1名，获得3能量和1张同阶兑换券，1张简易协作。
    同阶兑换券 = 被消除英雄品阶的英雄兑换券。待核对：等级按永久等级比较，并列随机；不足 4 名就全部消除。"""
    def on_use(self, eng, t):
        for u in eng.lowest(eng.heroes(), 4):
            q = max(1, min(5, quality(u.name) or 1))
            eng.remove(u, src="听天由命")
            eng.s.energy += 3
            eng.gain_card(f"{q}阶英雄兑换券")
            eng.gain_card("简易协作")


# ============================================================ 装备
@card("自适应武装")
class 自适应武装:
    """获得1张天赋牌自适应武装：指定1名英雄，随机获得2张与其适配的普通装备牌。
    选中天赋（无目标）→ 1 张同名赠送牌；打出赠送牌（有目标）→ 2 张基础装备牌。
    "适配"：基础装备四选一成品里对该英雄的最高适配分（data/equip_relations.json）≥5 的算适配，没有就在 8 件里随机。
    待核对：游戏的"适配"判据；两张可以重复。"""
    def on_use(self, eng, t):
        if t is None:
            _gain_gift(eng, "自适应武装", 1)
            return
        if t.kind != "hero":
            return False
        _use_gift(eng, "自适应武装")
        fit = _equip_fit().get(t.name) or {}
        rec = _recipes()

        def best(b):
            return max([float((fit.get(x) or {}).get("score") or 0) for x in rec.get(b, [])] or [0.0])
        cands = [b for b in _basic_equips() if best(b) >= 5] or list(_basic_equips())
        for _ in range(2):
            if cands:
                eng.gain_card(eng.pick(cands, 1)[0])


@card("抢攻")
class 抢攻:
    """获得1张普通装备牌。之后5回合内，若对其他棋手累计造成20点伤害，获得10能量。
    普通装备牌 = 8 件基础装备里随机 1 张。
    接口缺口：引擎没有"对棋手造成伤害"，后半句不结算（记 flags["抢攻_until"] = 截止回合，给以后用）。"""
    def on_use(self, eng, t):
        eng.gain_card(_random_basic(eng))
        eng.s.flags["抢攻_until"] = eng.s.round + 5


@card("厉兵秣马")
class 厉兵秣马:
    """随机获得1张普通装备牌。接下来5个回合若你的棋手等级没有变化，再获得2张。
    待核对："接下来 5 个回合" = 选中之后的第 1~5 个回合，在第 6 个回合开始时结算；期间升过级就作废。"""
    def on_use(self, eng, t):
        eng.gain_card(_random_basic(eng))
        eng.s.flags["厉兵秣马"] = [eng.s.round, eng.s.lord_level]

    def on_round_start(self, eng, holder):
        st = eng.s.flags.get("厉兵秣马")
        if holder is not None or not st:
            return
        r0, lv0 = st
        if eng.s.lord_level != lv0:
            eng.s.flags["厉兵秣马"] = None
            return
        if eng.s.round >= r0 + 6:
            eng.s.flags["厉兵秣马"] = None
            for _ in range(2):
                eng.gain_card(_random_basic(eng))


@card("神秘英雄")
class 神秘英雄:
    """获得2张神秘英雄牌。手动刷新时，随机变为不低于当前棋手对应品阶的英雄牌。每回合有15%概率自动销毁并获得3能量。
    神秘英雄牌自己是 in_hand 的牌（第 7 批）：回合开始 15% 销毁 +3 能量、手动刷新（eng.refresh_shop）时变身，
    都由引擎的手牌钩子结算，本天赋只负责发牌。"""
    def on_use(self, eng, t):
        eng.gain_card("神秘英雄牌")
        eng.gain_card("神秘英雄牌")


# ============================================================ 预言
@card("预言家")
class 预言家:
    """分别获得1张天赋牌胜利预言与失败预言：预测战斗结果，正确可获得奖励。"""
    def on_use(self, eng, t):
        eng.gain_card("胜利预言")
        eng.gain_card("失败预言")


class _Prophecy:
    """打出后挂进 eng.s.talents 等下一场战斗结算，结算后摘掉（手牌 / 已打出的牌没有钩子）。"""
    name = ""
    want_win = True

    def on_use(self, eng, t):
        eng.s.talents.append(self.name)

    def on_battle_end(self, eng, holder, win=False):
        if holder is not None or self.name not in eng.s.talents:
            return
        eng.s.talents.remove(self.name)
        if bool(win) == self.want_win:
            self.reward(eng)

    def reward(self, eng):
        raise NotImplementedError


@card("胜利预言")
class 胜利预言(_Prophecy):
    """预测下场战斗胜利，预测成功，获得2张3阶英雄兑换券。"""
    name, want_win = "胜利预言", True

    def reward(self, eng):
        eng.gain_card("3阶英雄兑换券")
        eng.gain_card("3阶英雄兑换券")


@card("失败预言")
class 失败预言(_Prophecy):
    """预测下场战斗失败，预测成功，获得4能量。"""
    name, want_win = "失败预言", False

    def reward(self, eng):
        eng.s.energy += 4


# ============================================================ 拍卖
@card("拍卖专家")
class 拍卖专家(_Passive):
    """每次拍卖后，获得1张天赋牌拍卖重现：选择获得1张拍卖中你未获得的卡牌。
    第 k 次拍卖（第 4/8/12 回合开头）后给「拍卖重现：k」，在该回合开始时发。
    接口缺口：引擎没有拍卖阶段，用 on_round_start + 拍卖回合表近似。"""
    def on_round_start(self, eng, holder):
        if holder is not None:
            return
        rs = tuple(getattr(B.meta, "AUCTION_ROUNDS", (4, 8, 12)))
        if eng.s.round in rs:
            k = rs.index(eng.s.round) + 1
            if k in CN_NUM and not eng.s.flags.get(f"拍卖专家_{k}"):
                eng.s.flags[f"拍卖专家_{k}"] = True
                eng.gain_card(f"拍卖重现：{CN_NUM[k]}")


def _replay(k, cost):
    class _R:
        __doc__ = (f"{'消耗4能量，' if cost else ''}展示第{CN_NUM[k]}轮拍卖中你未获得的卡牌（最多5张），从中挑选1张。\n"
                   f"    接口缺口：拍品数据要由推演打法放进 flags[\"auction_unwon:{k}\"]（卡名列表）；没有数据时视为不能使用。")

        def on_use(self, eng, t):
            opts = list(eng.s.flags.get(f"auction_unwon:{k}") or [])[:5]
            if not opts or eng.s.energy < cost:
                return False
            pick = eng.choose("auction_replay", opts, round=k)
            if not pick:
                return False
            eng.s.energy -= cost
            if (card_db().get(pick) or {}).get("category") == "英雄牌":
                eng.take(pick)
            eng.gain_card(pick)
    return _R


card("拍卖重现：一")(_replay(1, 0))
card("拍卖重现：二")(_replay(2, 0))
card("拍卖重现：三")(_replay(3, 4))


# ============================================================ 场上物件
@card("擂台猛虎")
class 擂台猛虎:
    """获得1个猛虎擂台，擂台中的【三分】英雄获得30点物攻和法攻。每当敌人于擂台范围内被击败时，擂台吸收其灵魂，永久额外提供10点物攻和法攻。每吸收5次，擂台范围永久扩大1格，最多扩大3次。
    养成上只做"放一个猛虎擂台"（物件，不占人口）；攻击力加成是战斗效果。
    待核对：擂台是否占一个格子（现按占格物件实现，kind="object"）。"""
    def on_use(self, eng, t):
        if not any(u.name == "猛虎擂台" for u in eng.s.units):
            eng.deploy("猛虎擂台", kind="object", pop=False)


@card("猛虎擂台")
class 猛虎擂台:
    """为【三分】英雄增加物理和法术攻击力。
    战斗效果：擂台中的【三分】英雄 +30 物攻法攻，范围内击败敌人永久 +10，不影响养成。"""


@card("班叔的遗留物")
class 班叔的遗留物:
    """获得2级电磁弹簧：在指定位置构筑1个2级电磁弹簧（敌方踩中时会被弹至战场后排）。"""
    def on_use(self, eng, t):
        eng.gain_card("2级电磁弹簧")


@card("2级电磁弹簧")
class 二级电磁弹簧:
    """在指定位置构筑1个2级电磁弹簧。
    养成上只做"放一个电磁弹簧物件"（单位名「电磁弹簧」，counters["level"]=2）；把敌人弹到后排是战斗效果。
    待核对：弹簧是否占我方棋盘的格子（现按占格物件实现）；放在哪一格由推演打法决定（这里取第一个空格）。"""
    def on_use(self, eng, t):
        u = eng.deploy("电磁弹簧", kind="object", pop=False)
        if u is None:
            return False
        u.counters["level"] = 2


# ============================================================ 只有战斗效果
@card("防御领域")
class 防御领域:
    """棋手每次受到的伤害减少2点。
    战斗效果：减少棋手受到的伤害，不影响养成。"""


@card("狂铁·强化")
class 狂铁强化:
    """狂铁触发卡牌效果时，自身+10%攻速。
    战斗效果：攻速，不影响养成。"""


@card("微弱优势")
class 微弱优势:
    """打出觉醒英雄牌时，其获得22%暴击率和11%物理及法术吸血（每名英雄仅1次）。
    战斗效果：暴击 / 吸血，不影响养成。"""


@card("刻舟减蓝")
class 刻舟减蓝:
    """英雄20级后，法力值上限-10。
    战斗效果：法力上限，不影响养成。"""


@card("猪猪")
class 猪猪:
    """战斗开始时，敌方2名随机英雄变为小猪，持续3秒。
    战斗效果：控制，不影响养成。"""


@card("专注射击")
class 专注射击:
    """战斗开始时，你后两排英雄每次普攻获得3%攻速和3点物理攻击力，上限10层。
    战斗效果：攻速 / 攻击力，不影响养成。"""


@card("超专注射击")
class 超专注射击:
    """战斗开始时，你后两排英雄每次普攻获得5%攻速和5点物理攻击力，上限10层。
    战斗效果：攻速 / 攻击力，不影响养成。"""


@card("小小的助力")
class 小小的助力:
    """战斗开始时，你场上有不少于4名1-3阶英雄，则4-5阶英雄获得30%伤害加成和60%攻速。
    战斗效果：伤害加成 / 攻速，不影响养成。"""


@card("重装上阵")
class 重装上阵:
    """战斗开始时，你至少穿戴2件装备的英雄获得15%伤害加成和15%伤害减免。
    战斗效果：伤害加成 / 减免，不影响养成。"""


@card("前线堡垒")
class 前线堡垒:
    """战斗开始时，你第一排的英雄获得20点物理及法术防御。
    战斗效果：防御，不影响养成。"""


@card("强者自强")
class 强者自强:
    """战斗开始时，已觉醒英雄获得40点物理攻击力和40点法术攻击力。
    战斗效果：攻击力，不影响养成。"""


@card("破败之力")
class 破败之力:
    """战斗开始时，你的英雄普攻附带敌人最大生命值2%的法术伤害。
    战斗效果：伤害，不影响养成。"""


@card("虚构装备")
class 虚构装备:
    """战斗开始时，随机3名己方英雄临时获得1件普通装备及100点生命上限。
    战斗效果：临时装备 / 生命上限，只在战斗中存在，不影响养成。
    待核对：临时装备在战斗中是否算"穿戴装备件数"（影响芈月这类按装备数选目标的反应器）—— 现在不算。"""


@card("战前叫阵")
class 战前叫阵(_Passive):
    """战斗开始时，敌我双方等级最高英雄比拼等级：我方胜利，其英雄获得1次完美复生；我方失败，其变为猪猪3秒，获得30%最大生命值护盾。
    养成上只有"完美复生"有影响（多一条命 → 阵亡 / 牺牲触发次数变化）；变猪 / 护盾是战斗效果。
    接口缺口：引擎不知道敌方最高等级 —— 推演打法放进 flags["enemy_top_level"] 才比拼，没有就不结算。
    待核对：比拼用 永久 + 临时 等级；平局算我方失败。"""
    def react_battle_start(self, eng, holder, **kw):
        if holder is not None or not _once_round(eng, "战前叫阵"):
            return
        enemy = eng.s.flags.get("enemy_top_level")
        hs = [u for u in eng.s.on_board() if u.kind == "hero"]
        if enemy is None or not hs:
            return
        m = max(u.lv() for u in hs)
        top = eng.pick([u for u in hs if u.lv() == m], 1)[0]
        if m > enemy:
            top.counters["revive"] = top.counters.get("revive", 0) + 1


# ------------------------------------------------------------ 接口说明（2026-10-14 统一后）
# 已由引擎补上：打出英雄牌 vs 被触发的登场（eng.is_card_use）、回合结束 on_round_end（存款利息）、消耗能量 spend 事件
# （消费返利）、手动刷新 refresh 事件（节能商店）、手牌钩子（神秘英雄牌）、棋手生命值 State.hp / eng.add_hp。
# 仍未解决：没有"等级变化"事件（三十而立 / 成年礼 轮询）；商店生成与价格（破格、节能商店只记标记）；
# 拍卖阶段与拍品数据（拍卖专家用回合表近似、拍卖重现要 flags["auction_unwon:k"]）；敌方信息（战前叫阵要
# flags["enemy_top_level"]）；对其他棋手的伤害（抢攻后半句不结算）；限制性进化 / 契约瓶仍按名字记在 flags。
