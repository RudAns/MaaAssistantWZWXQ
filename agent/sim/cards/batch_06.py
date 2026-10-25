"""第 6 批：通用天赋（群星璀璨 … 无限进步）+ 棋手关联牌（小妲己 / 阿离强化 / 镜 / 乔汐 / 玉环 / 庄小鱼 / 马可 / 白歌 / 昭君）。

约定（本模块内）：
- 天赋牌被选中时，调用方把卡名放进 eng.s.talents 并调用 on_use(eng, None)。
  天赋送出的"同名赠送牌"（延迟满足 / 大明星 / 资深猎头）也用 on_use 打出 → 用 flags 区分"第一次 = 选天赋"和"之后 = 打出赠送牌"。
- 持续效果的天赋 / 强化牌，都用 _once 做"每回合只结算一次"保护：同一张卡同时出现在 talents 和 lord_cards 时不会结算两次。
- "战斗开始时 / 开战时"的天赋效果挂在 react_battle_start 上（每个单位都会发一次事件，用 _once 只执行第一次）；
  "开战前"（混沌裂隙吞噬、潮起潮落传送）挂在引擎的整场钩子 on_battle_begin（开团之前）。
- 2026-10-14 统一接口：棋手生命值 eng.s.hp / eng.add_hp；关键词 has_effect（engine）；觉醒进度 eng.awaken_step；
  转瞬复制记在手牌状态（eng.gain_card(..., transient=True)）；"打出英雄牌" eng.is_card_use()；购买事件 engine.buy；
  只在本场战斗存在的召唤物 eng.summon(battle_only=True)。
"""
from __future__ import annotations

import json
from functools import lru_cache

from agent.engine import board as B
from ..engine import LORD_HP_MAX, ROOT, REGISTRY, card, card_db, category, has_effect, impl, quality


# ============================================================ 辅助函数
def _once(eng, key: str) -> bool:
    """本回合第一次返回 True（round: 前缀的 flags 在回合开始时清空）。"""
    k = "round:" + key
    if eng.s.flags.get(k):
        return False
    eng.s.flags[k] = True
    return True


def _first_call(eng, key: str) -> bool:
    """整局第一次返回 True：用来区分"选中天赋"与"打出同名赠送牌"。"""
    k = key + ":talent_picked"
    if eng.s.flags.get(k):
        return False
    eng.s.flags[k] = True
    return True


def _persist(eng, name: str, where: str = "talents") -> None:
    """让持续效果有挂靠处：不在 talents / lord_cards 里就加进去（天赋牌进 talents，棋手关联效果牌进 lord_cards）。"""
    s = eng.s
    if name in s.talents or name in s.lord_cards:
        return
    (s.talents if where == "talents" else s.lord_cards).append(name)


def _has_effect(name: str, kw: str) -> bool:
    """英雄卡面拥有 kw 效果（engine.has_effect：只"提到"的反应器不算）。"""
    return has_effect(name, kw)


def _hero_names(tiers=None, pred=None) -> list:
    return [n for n, c in card_db().items() if c.get("category") == "英雄牌"
            and (tiers is None or quality(n) in tiers) and (pred is None or pred(n))]


def _draw_hero_in(eng, tiers, pred=None):
    """在给定品阶范围内按牌库剩余张数抽 1 张英雄（多个品阶合在一起按剩余张数加权）。"""
    names = _hero_names(tiers, pred)
    ws = [eng.pool_left(n) for n in names]
    if not names or sum(ws) <= 0:
        return None
    return eng.rng.choices(names, ws)[0]


def _draw_hero_shop(eng, pred=None):
    """按当前棋手等级的商店品阶概率先抽品阶（只在有符合条件的英雄的品阶里抽），再按牌库剩余抽英雄。"""
    odds = B.meta.TIER_ODDS.get(eng.s.lord_level) or {1: 1.0}
    ts = [t for t, w in odds.items() if w > 0 and any(eng.pool_left(n) > 0 for n in _hero_names([t], pred))]
    if not ts:
        return _draw_hero_in(eng, [1, 2, 3, 4, 5], pred)
    t = eng.rng.choices(ts, [odds[x] for x in ts])[0]
    return _draw_hero_in(eng, [t], pred)


def _gain_hero(eng, name, take=True) -> None:
    if not name:
        return
    if take:
        eng.take(name)
    eng.gain_card(name)


def _hero_opts(eng, tiers, k=3, pred=None) -> list:
    opts = []
    for _ in range(30):
        x = _draw_hero_in(eng, tiers, pred)
        if x and x not in opts:
            opts.append(x)
        if len(opts) >= k:
            break
    return opts


@lru_cache(maxsize=1)
def _basic_equips() -> list:
    try:
        d = json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8"))
        return list(d.get("recipes") or {})
    except Exception:  # noqa: BLE001
        return ["守护者之铠", "力量腰带", "圣者法典", "破碎圣杯", "大棒", "雷鸣刃", "速击之枪", "风暴巨剑"]


def _effect_opts(eng, tiers, k=3) -> list:
    """按历史出现频率（data/engine/draw_stats.json 的锦囊统计）抽 k 张不同的效果牌。"""
    from agent.engine.rollout import draw_stats
    js = draw_stats().get("jinnang") or {}
    st: dict = {}
    for t in tiers:
        for n, w in (((js.get(str(t)) or {}).get("cards")) or {}).items():
            st[n] = st.get(n, 0) + w
    if not st:
        return []
    names, ws = list(st), list(st.values())
    opts = []
    for _ in range(40):
        x = eng.rng.choices(names, ws)[0]
        if x not in opts:
            opts.append(x)
        if len(opts) >= k:
            break
    return opts


def _random_effect(eng):
    """随机 1 张效果牌：品阶按当前棋手等级的商店概率抽，再按锦囊统计抽卡。待核对：真实的随机池与品阶分布。"""
    odds = B.meta.TIER_ODDS.get(eng.s.lord_level) or {1: 1.0}
    t = eng.rng.choices(list(odds), list(odds.values()))[0]
    o = _effect_opts(eng, [t], 1)
    return o[0] if o else None


def _hand_heroes(eng) -> list:
    """手牌里的英雄牌下标。接口缺口：手牌只记卡名，不知道是不是觉醒卡 —— 一律当非觉醒。"""
    return [i for i, n in enumerate(eng.s.hand) if category(n) == "英雄牌"]


def _awaken_step(eng, u, n: int = 1, src: str = "") -> None:
    """觉醒进度 +n（engine.awaken_step：封顶 3，凑满 = 觉醒 = 回手再打出一次登场）。"""
    eng.awaken_step(u, n, src=src)


def _unit_by_uid(eng, uid):
    return next((x for x in eng.s.units if x.uid == uid), None)


def _cell_of(eng, t, kind: str):
    """"指定1个阵位"：target 可以是单位（取其格）或格子坐标；None 时交给推演打法选（默认：等级最低的英雄所在格，其次空格）。"""
    if t is not None and hasattr(t, "cell"):
        return t.cell
    if isinstance(t, (tuple, list)) and len(t) == 2:
        return B.cell_of(t)
    opts = [x.cell for x in sorted(eng.heroes(), key=lambda x: x.level) if x.cell] + eng.free_cells()
    return eng.choose(kind, opts)


def _register_aux(name: str):
    """注册本批卡面里出现、但卡库里没有独立条目的辅助牌（已有实现就不覆盖）。"""
    def deco(cls):
        if name not in REGISTRY:
            card(name)(cls)
        return cls
    return deco


# ============================================================ 辅助牌（卡库没有独立条目）
@_register_aux("4-5阶英雄兑换券")
class 四五阶英雄兑换券:
    """（群星璀璨送的）卡牌3选1（4-5阶英雄牌）。待核对：3 个选项按 4、5 阶合并后的牌库剩余加权抽。"""
    def on_use(self, eng, t):
        opts = _hero_opts(eng, [4, 5])
        pick = eng.choose("hero", opts, tier=45)
        if pick:
            _gain_hero(eng, pick)


@_register_aux("4-5阶效果锦囊")
class 四五阶效果锦囊:
    """（王牌部队送的）选择1张4-5阶效果牌。待核对：选项按 4、5 阶锦囊统计合并抽 3 张。"""
    def on_use(self, eng, t):
        pick = eng.choose("jinnang", _effect_opts(eng, [4, 5]), tier=45)
        if pick:
            eng.gain_card(pick)


# ============================================================ 通用天赋
@card("群星璀璨")
class 群星璀璨:
    """获得2张4-5阶英雄兑换券。当你场上至少有5名4-5阶英雄时，所有4-5阶英雄获得20临时等级。
    待核对：临时等级按"战斗开始时满足条件就给"结算（持续条件，准备阶段是否也显示未建模）。"""
    def on_use(self, eng, t):
        for _ in range(2):
            eng.gain_card("4-5阶英雄兑换券")

    def react_battle_start(self, eng, holder, **kw):
        if not _once(eng, "群星璀璨"):
            return
        hi = [x for x in eng.heroes() if quality(x.name) in (4, 5)]
        if len(hi) >= 5:
            for x in hi:
                eng.temp(x, 20, src="群星璀璨")


@card("特大电池包")
class 特大电池包:
    """获得12能量。"""
    def on_use(self, eng, t):
        eng.s.energy += 12


@card("强者恒强")
class 强者恒强:
    """战斗开始时，你的已觉醒英雄获得50点物理攻击力和50点法术攻击力。若所有己方英雄已觉醒，则再获得20%物理及法术吸血。
    战斗效果：攻击力 / 吸血，不影响养成。"""


@card("随机强化")
class 随机强化:
    """战斗开始时，你品阶最高的1名英雄等级永久+9。（品阶并列时随机 1 名）"""
    def react_battle_start(self, eng, holder, **kw):
        if not _once(eng, "随机强化"):
            return
        hs = eng.heroes()
        if not hs:
            return
        q = max(quality(x.name) for x in hs)
        t = eng.pick([x for x in hs if quality(x.name) == q], 1)[0]
        eng.level(t, 9, src="随机强化")


@card("无影手")
class 无影手:
    """回合开始时，获得3次仅本回合可用的免费商店刷新。
    战斗结束时收回没用掉的这 3 次；没经过战斗就进了下一回合（推演打法跳过战斗）时，在下一回合开始先收回再发。
    待核对：同时有永久免费刷新时，按"先用掉临时的"收回。
    基线记在不带 round: 前缀的 flags['无影手_base']（round: 前缀在回合开始时被引擎清掉，跨不过回合）。"""
    KEY = "无影手_base"

    def _expire(self, eng):
        s = eng.s
        base = s.flags.pop(self.KEY, None)
        if base is not None:
            s.free_refresh = min(s.free_refresh, base)

    def on_round_start(self, eng, holder):
        if not _once(eng, "无影手"):
            return
        self._expire(eng)                     # 上回合没经过战斗：剩下的临时刷新在这里过期
        s = eng.s
        s.flags[self.KEY] = s.free_refresh
        s.free_refresh += 3

    def on_battle_end(self, eng, holder, win=False):
        self._expire(eng)


@card("学霸")
class 学霸:
    """获得5能量。若你是第1个到达5/6级的棋手，则5级获得2张5阶英雄兑换券，6级获得1件天赋装备。失败则获得留级证明。
    "是不是第 1 个"看不到对手，交给推演打法：eng.choose("学霸_第一个", [False, True], level=) —— 默认 False（保守）。
    待核对：选中时已到 5/6 级是否当场判定（按当场判定当前等级实现）；天赋装备的候选（按卡库品阶 5 的装备、随机 3 件让打法选）；
    失败后是否还判 6 级（按失败即结束实现）。"""
    def on_use(self, eng, t):
        if not _first_call(eng, "学霸"):
            return False
        eng.s.energy += 5
        if eng.s.lord_level >= 5:
            self._check(eng, eng.s.lord_level)

    def react_lord_levelup(self, eng, holder, level=0):
        if "学霸" in eng.s.talents and level in (5, 6):
            self._check(eng, level)

    def _check(self, eng, level):
        f = eng.s.flags
        if f.get("学霸_done") or f.get(f"学霸_lv{level}"):
            return
        f[f"学霸_lv{level}"] = True
        if level == 6 and not f.get("学霸_lv5_ok"):
            return
        if eng.choose("学霸_第一个", [False, True], level=level):
            if level == 5:
                f["学霸_lv5_ok"] = True
                eng.gain_card("5阶英雄兑换券")
                eng.gain_card("5阶英雄兑换券")
            else:
                eqs = [n for n, c in card_db().items() if c.get("category") == "装备牌" and quality(n) == 5]
                pick = eng.choose("天赋装备", eng.pick(eqs, 3))
                if pick:
                    eng.gain_card(pick)
                f["学霸_done"] = True
        else:
            f["学霸_done"] = True
            _persist(eng, "留级证明")


@card("留级证明")
class 留级证明:
    """保持在棋手等级1时，每回合获得1能量。（回合开始时结算）"""
    def on_use(self, eng, t):
        _persist(eng, "留级证明")

    def on_round_start(self, eng, holder):
        if _once(eng, "留级证明") and eng.s.lord_level == 1:
            eng.s.energy += 1


@card("破败神威")
class 破败神威:
    """战斗开始时，你的英雄普攻附带敌人最大生命值3%的法术伤害。
    战斗效果：附加伤害，不影响养成。"""


@card("绝境逆袭")
class 绝境逆袭:
    """当你排名不是第1或第2名时，你的所有单位伤害+20%，最大生命值+25%。
    战斗效果：伤害 / 生命，不影响养成。"""


@card("买橘子去")
class 买橘子去:
    """获得1张天赋牌站这别动：指定1个阵位，每过2回合，获得1张此格上的英雄牌。"""
    def on_use(self, eng, t):
        eng.gain_card("站这别动")


@card("站这别动")
class 站这别动:
    """指定1个阵位，每过2回合，获得1张此格上的英雄牌。
    实现：记下（格子, 指定回合），在指定回合 +2、+4 … 的回合开始时，该格站着英雄就复制 1 张它的英雄牌（复制，不占牌库）。
    待核对：复制是否从公共牌库扣；"每过 2 回合"的起算点（按指定当回合起算）；召唤物站在格子上不算。"""
    def on_use(self, eng, t):
        cell = _cell_of(eng, t, "站这别动")
        if not cell:
            return False
        eng.s.flags.setdefault("站这别动_cells", []).append([cell, eng.s.round])
        _persist(eng, "站这别动")

    def on_round_start(self, eng, holder):
        if not _once(eng, "站这别动"):
            return
        for cell, r0 in eng.s.flags.get("站这别动_cells", []):
            d = eng.s.round - r0
            if d > 0 and d % 2 == 0:
                u = next((x for x in eng.heroes() if x.cell == tuple(cell)), None)
                if u:
                    eng.gain_card(u.name)


@card("战略中枢")
class 战略中枢:
    """可上阵人数+1，同时获得2张有福同享。"""
    def on_use(self, eng, t):
        eng.s.pop_bonus += 1
        eng.gain_card("有福同享")
        eng.gain_card("有福同享")


@card("虚构武装")
class 虚构武装:
    """战斗开始时，随机3名己方英雄临时获得2件普通装备及300点生命上限。
    战斗效果：临时装备 / 生命，不影响养成。待核对：临时装备是否计入"穿戴装备数"类效果（如芈月"穿戴装备最多"）。"""


@card("延迟满足")
class 延迟满足:
    """获得1张天赋牌延迟满足：指定1名英雄等级+10。3回合后，再+25。
    第一次 on_use = 选中天赋（送牌）；之后 = 打出赠送牌（要指定英雄）。+25 在"打出回合 +3"的回合开始时结算。
    待核对：目标届时已离场则作废。"""
    def on_use(self, eng, t):
        if _first_call(eng, "延迟满足"):
            eng.gain_card("延迟满足")
            return
        if t is None or t not in eng.s.units:
            return False
        eng.level(t, 10, src="延迟满足")
        eng.s.flags.setdefault("延迟满足_pending", []).append([t.uid, eng.s.round + 3])
        _persist(eng, "延迟满足")

    def on_round_start(self, eng, holder):
        if not _once(eng, "延迟满足"):
            return
        keep = []
        for uid, due in eng.s.flags.get("延迟满足_pending", []):
            if eng.s.round >= due:
                eng.level(_unit_by_uid(eng, uid), 25, src="延迟满足")
            else:
                keep.append([uid, due])
        eng.s.flags["延迟满足_pending"] = keep


@card("全副武装")
class 全副武装:
    """获得1张基础装备牌。战斗开始时，你至少穿戴2件装备的英雄获得15%伤害加成和减免。
    伤害加成 / 减免是战斗效果。待核对：基础装备是随机还是可选（按 8 件基础装备随机）。"""
    def on_use(self, eng, t):
        eng.gain_card(eng.pick(_basic_equips(), 1)[0])


@card("同心协力")
class 同心协力:
    """战斗开始时，若你至少有6名同阵营英雄，其获得50点物攻和法攻以及50%攻速。
    战斗效果：攻击 / 攻速，不影响养成。"""


@card("泉水")
class 泉水:
    """回合开始时，恢复3棋手生命值；若棋手生命值已满，改为获得4棋手经验。
    棋手生命值 eng.s.hp（上限 60）；经验到账后立即检查升级。"""
    def on_round_start(self, eng, holder):
        if not _once(eng, "泉水"):
            return
        if eng.s.hp < LORD_HP_MAX:
            eng.add_hp(3, src="泉水")
        else:
            eng.add_xp(4, src="泉水")


@card("三人游")
class 三人游:
    """随机获得3张4-5阶英雄牌。待核对：4、5 阶合并按牌库剩余张数加权抽。"""
    def on_use(self, eng, t):
        for _ in range(3):
            _gain_hero(eng, _draw_hero_in(eng, [4, 5]))


@card("扩编高峰")
class 扩编高峰:
    """获得1个可购买天赋：花费8能量，可上阵人数+1。可购买3次。
    没有"可购买天赋"的购买动作 —— 提供 buy(eng)（eng.spend 扣费），并在回合开始时问推演打法
    eng.choose("扩编高峰", [False, True], cost=8)（默认不买）。"""
    COST, TIMES, KEY = 8, 3, "扩编高峰_left"

    def on_use(self, eng, t):
        eng.s.flags[self.KEY] = eng.s.flags.get(self.KEY, 0) + self.TIMES

    def buy(self, eng) -> bool:
        s = eng.s
        if s.flags.get(self.KEY, 0) <= 0 or s.energy < self.COST or not eng.spend(self.COST, src="扩编高峰"):
            return False
        s.flags[self.KEY] -= 1
        s.pop_bonus += 1
        eng.note("购买天赋", card=self.KEY.split("_")[0])
        return True

    def on_round_start(self, eng, holder):
        if not _once(eng, self.KEY):
            return
        s = eng.s
        while s.flags.get(self.KEY, 0) > 0 and s.energy >= self.COST and \
                eng.choose(self.KEY.split("_")[0], [False, True], cost=self.COST):
            self.buy(eng)


def _supply(kw):
    class _S:
        def _give(self, eng):
            _gain_hero(eng, _draw_hero_shop(eng, lambda n: _has_effect(n, kw)))

        def on_use(self, eng, t):
            self._give(eng)

        def on_round_start(self, eng, holder):
            if _once(eng, f"补给线：{kw}"):
                self._give(eng)
    _S.__doc__ = (f"获得1张{kw}效果英雄牌。回合开始时，再获得1张。\n"
                  f"    英雄池 = 卡面带\"{kw}：\"效果条目的英雄；品阶按当前棋手等级的商店概率抽、再按牌库剩余抽。\n"
                  f"    待核对：品阶分布（是否跟商店等级走）。")
    return _S


for _kw in ("整备", "牺牲", "登场"):
    card(f"补给线：{_kw}")(_supply(_kw))


@card("中流砥柱")
class 中流砥柱:
    """打出3阶英雄牌，额外+6级。
    "打出" = 每次使用 3 阶英雄牌（上场、合成、觉醒回手再打出；eng.is_card_use()，被唤醒类触发的登场不算）。召唤物 / 随机上阵不算。"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if unit is not None and unit.kind == "hero" and unit in eng.s.units and quality(unit.name) == 3 \
                and eng.is_card_use():
            eng.level(unit, 6, src="中流砥柱")


@card("混沌次元")
class 混沌次元:
    """获得混沌裂隙：指定阵位召唤裂隙，开战前吞噬英雄，永久获得其双倍等级。开战时，召唤1名品阶≥商店等级的满装备英雄。
    （天赋：送 1 张混沌裂隙，效果在混沌裂隙里）"""
    def on_use(self, eng, t):
        eng.gain_card("混沌裂隙")


@card("混沌裂隙")
class 混沌裂隙:
    """指定阵位召唤裂隙，开战前吞噬该格的英雄，永久获得其双倍等级。开战时，召唤1名品阶≥商店等级的英雄（无卡牌效果），使之获得裂隙等级，并获得适配装备。
    规则事实（2026-10-16 确认）：裂隙不占人口、放下不能挪；**每场**战前把站在格上的英雄换成随机英雄（等级 = 其 2 倍）
    只打这一场，**战后原英雄回来、不会消失**；格上站谁可以随时换。（旧实现按"永久吞掉 + 等级累加"，错。）
    实现：flags['混沌裂隙'] = [{cell}]；on_battle_begin 把格上英雄暂时移出 units（记在 _rift_out），召唤 battle_only 的
    "混沌裂隙·X"（等级 2×真实等级，品阶在 [min(棋手等级,5), 5] 均匀抽）；on_battle_end 放回。格上没人 → 不召唤。"""
    def on_use(self, eng, t):
        cell = _cell_of(eng, t, "混沌裂隙")
        if not cell:
            return False
        eng.s.flags.setdefault("混沌裂隙", []).append({"cell": tuple(cell)})
        _persist(eng, "混沌裂隙")

    def on_battle_begin(self, eng, holder):
        if holder is not None or not _once(eng, "混沌裂隙"):
            return
        out = eng.s.flags.setdefault("_rift_out", [])
        for rift in eng.s.flags.get("混沌裂隙", []):
            v = next((x for x in eng.heroes() if x.cell == tuple(rift["cell"])), None)
            if v is None:
                continue
            eng.s.units.remove(v)                          # 只离开这一场战斗（不触发离场、不还牌库）
            out.append(v)
            eng.note("裂隙换人", unit=v.name, n=2 * v.level)
            lo = min(max(1, eng.s.lord_level), 5)
            name = _draw_hero_in(eng, [eng.rng.randint(lo, 5)])
            eng.summon(f"混沌裂隙·{name or '英雄'}", level=max(1.0, 2 * v.level), battle_only=True)

    def on_battle_end(self, eng, holder, **kw):
        if holder is not None:
            return
        for v in eng.s.flags.pop("_rift_out", []):
            if v not in eng.s.units:
                eng.s.units.append(v)


@card("小咖秀")
class 小咖秀:
    """战斗开始时，若你只有1-3阶英雄，随机获得1张场上的英雄牌。（复制，不占牌库。待核对）"""
    def react_battle_start(self, eng, holder, **kw):
        if not _once(eng, "小咖秀"):
            return
        hs = eng.heroes()
        if hs and all(1 <= quality(x.name) <= 3 for x in hs):
            eng.gain_card(eng.pick(hs, 1)[0].name)


@card("永不独行")
class 永不独行:
    """获得1张双人同行。回合开始时，若棋手不超过3级，再获得1张。
    待核对：卡库有"双人同行·2阶 / ·3阶"两张，送哪张没写 —— 按各 50% 随机。"""
    def _give(self, eng):
        eng.gain_card(eng.pick(["双人同行·2阶", "双人同行·3阶"], 1)[0])

    def on_use(self, eng, t):
        self._give(eng)

    def on_round_start(self, eng, holder):
        if _once(eng, "永不独行") and eng.s.lord_level <= 3:
            self._give(eng)


def _duo(tier):
    class _D:
        def on_use(self, eng, t):
            x = _draw_hero_in(eng, [tier], lambda n: eng.pool_left(n) >= 2) or _draw_hero_in(eng, [tier])
            if not x:
                return False
            _gain_hero(eng, x)
            _gain_hero(eng, x)
    _D.__doc__ = f"随机获得2张相同的{tier}阶英雄牌。（按牌库剩余抽 1 个英雄，拿 2 张，优先剩余 ≥2 张的）"
    return _D


card("双人同行·2阶")(_duo(2))
card("双人同行·3阶")(_duo(3))


@card("搏命")
class 搏命:
    """棋手生命值-20，获得25能量。（engine.add_hp）"""
    def on_use(self, eng, t):
        eng.add_hp(-20, src="搏命")
        eng.s.energy += 25


@card("紧急救援")
class 紧急救援:
    """获得15棋手生命值、1张装备箱、1张5阶兑换券。（生命值上限 60；5阶兑换券 = 5阶英雄兑换券）"""
    def on_use(self, eng, t):
        eng.add_hp(15, src="紧急救援")
        eng.gain_card("装备箱")
        eng.gain_card("5阶英雄兑换券")


@card("危险数字")
class 危险数字:
    """在商店每买3张牌，获得1能量；若其品阶相加等于10，随机再获得其中1张。
    购买 = engine.buy 发的 buy 事件。
    待核对：再获得的那张是否占牌库（按复制、不扣牌库）。"""
    def react_buy(self, eng, holder, card="", **kw):
        if not card:
            return
        buys = eng.s.flags.setdefault("危险数字_buys", [])
        buys.append(card)
        if len(buys) >= 3:
            got = buys[:3]
            del buys[:3]
            eng.s.energy += 1
            if sum(quality(n) for n in got) == 10:
                eng.gain_card(eng.pick(got, 1)[0])


@card("大明星")
class 大明星:
    """获得1张大明星：随机上阵1名不在场的等级为棋手等级x7的5阶英雄，其觉醒进度为1，此后每经过2回合则额外+1。
    第一次 on_use = 选中天赋（送牌）；之后 = 打出赠送牌。上阵按牌库剩余抽 5 阶（排除场上已有的名字）并扣牌库；
    觉醒进度在"上阵回合 +2、+4"的回合开始时各 +1，凑满 3 = 觉醒（见 _awaken_step）。
    待核对：上阵时是否扣公共牌库；人口满时能否上阵（引擎只查空格）；觉醒后是否回手重打。"""
    def on_use(self, eng, t):
        if _first_call(eng, "大明星"):
            eng.gain_card("大明星")
            return
        on = {x.name for x in eng.heroes()}
        name = _draw_hero_in(eng, [5], lambda n: n not in on)
        if not name:
            return False
        u = eng.deploy(name, level=float(eng.s.lord_level * 7))
        if u is None:
            return False
        eng.take(name)
        u.awaken = max(u.awaken, 1)
        eng.s.flags.setdefault("大明星_units", []).append([u.uid, eng.s.round])
        _persist(eng, "大明星")

    def on_round_start(self, eng, holder):
        if not _once(eng, "大明星"):
            return
        for uid, r0 in eng.s.flags.get("大明星_units", []):
            d = eng.s.round - r0
            if d > 0 and d % 2 == 0:
                _awaken_step(eng, _unit_by_uid(eng, uid), 1, src="大明星")


@card("王牌部队")
class 王牌部队:
    """战斗开始时，若你至少有3名4-5阶英雄，获得1张4-5阶效果锦囊。"""
    def react_battle_start(self, eng, holder, **kw):
        if not _once(eng, "王牌部队"):
            return
        if sum(1 for x in eng.heroes() if quality(x.name) in (4, 5)) >= 3:
            eng.gain_card("4-5阶效果锦囊")


@card("有舍有得")
class 有舍有得:
    """回合开始时，随机1名英雄觉醒进度+1，但英雄合成提升的等级-1。
    待核对：随机只在未觉醒的英雄里挑；合成 -1 = 合成后立刻 -1 级（引擎合成固定 +3）。"""
    def on_round_start(self, eng, holder):
        if not _once(eng, "有舍有得"):
            return
        c = [x for x in eng.heroes() if not x.awakened]
        if c:
            _awaken_step(eng, eng.pick(c, 1)[0], 1, src="有舍有得")

    def react_merge(self, eng, holder, unit=None, **kw):
        if unit is not None and unit in eng.s.units:
            eng.level(unit, -1, src="有舍有得")


@card("资深猎头")
class 资深猎头:
    """棋手4级时，获得1张天赋牌资深猎头：4阶卡牌2选1，并将选中的卡牌一分为四。
    第一次 on_use = 选中天赋（已 ≥4 级当场送牌，否则升到 4 级时送）；之后 = 打出赠送牌：
    2 个 4 阶英雄选项（按牌库剩余抽）→ 打法选 1 → 获得 4 张。
    待核对："4阶卡牌"是否含效果牌 / 装备（按英雄牌）；4 张里扣牌库几张（按 1 张，其余是"分出来的"）。"""
    def on_use(self, eng, t):
        f = eng.s.flags
        if _first_call(eng, "资深猎头"):
            if eng.s.lord_level >= 4:
                f["资深猎头_given"] = True
                eng.gain_card("资深猎头")
            return
        pick = eng.choose("hero", _hero_opts(eng, [4], k=2), tier=4, src="资深猎头")
        if not pick:
            return False
        eng.take(pick)
        for _ in range(4):
            eng.gain_card(pick)

    def react_lord_levelup(self, eng, holder, level=0):
        f = eng.s.flags
        if level >= 4 and f.get("资深猎头:talent_picked") and not f.get("资深猎头_given"):
            f["资深猎头_given"] = True
            eng.gain_card("资深猎头")


@card("无限进步")
class 无限进步:
    """获得1张3阶兑换券。回合开始时，获得2棋手经验。满级后，改为获得2能量。（3阶兑换券 = 3阶英雄兑换券；经验到账立即检查升级）"""
    def on_use(self, eng, t):
        eng.gain_card("3阶英雄兑换券")

    def on_round_start(self, eng, holder):
        if not _once(eng, "无限进步"):
            return
        if eng.s.lord_level >= B.MAX_LORD_LEVEL:
            eng.s.energy += 2
        else:
            eng.add_xp(2, src="无限进步")


# ============================================================ 小妲己
@card("心之引力")
class 心之引力:
    """获得4次临时刷新。且这4次刷新时，商店购买消耗-1。
    实现：免费刷新 +4，并记 flags['round:心之引力_折扣']=4。接口缺口：引擎不模拟商店购买价 / 临时刷新过期，由推演打法读 flags。
    待核对："临时"= 仅本回合。"""
    def on_use(self, eng, t):
        eng.s.free_refresh += 4
        eng.s.flags["round:心之引力_折扣"] = eng.s.flags.get("round:心之引力_折扣", 0) + 4


@card("魅力无限")
class 魅力无限:
    """每回合打出的第1张效果牌会触发2次。
    实现：每回合第一次"使用效果牌"事件时，再结算一次那张牌的 on_use（同一目标）；第二次结算不再发"使用效果牌"事件。
    打出魅力无限本身不占"第 1 张"。待核对：第二次是否也触发芈月等反应器。"""
    def on_use(self, eng, t):
        _persist(eng, "魅力无限", "lord_cards")

    def react_effect_card(self, eng, holder, card="", target=None, **kw):
        if card == "魅力无限" or not _once(eng, "魅力无限"):
            return
        c = impl(card)
        if c is not None and hasattr(c, "on_use"):
            c.on_use(eng, target)


# ============================================================ 阿离：枫叶舞强化牌（全是转移 / 战斗效果）
class _MapleBoost:
    def on_use(self, eng, t):
        name = type(self).names[0]
        if name not in eng.s.lord_cards:
            eng.s.lord_cards.append(name)


@card("枫灵涌动")
class 枫灵涌动(_MapleBoost):
    """转移后每秒获得15点法力值，持续4秒。战斗效果：法力，不影响养成。（强化牌：放进 lord_cards）"""


@card("有叶新生")
class 有叶新生(_MapleBoost):
    """转移后恢复20%最大生命值。战斗效果：回血，不影响养成。（强化牌：放进 lord_cards）"""


@card("枫佑永驻")
class 枫佑永驻(_MapleBoost):
    """护盾量增加且持续时间变为无限。战斗效果：护盾，不影响养成。（强化牌：放进 lord_cards）"""


@card("拈花一指")
class 拈花一指(_MapleBoost):
    """转移后眩晕附近的敌人2秒。战斗效果：控制，不影响养成。（强化牌：放进 lord_cards）"""


@card("红枫疾舞")
class 红枫疾舞(_MapleBoost):
    """转移后的4次普通攻击以最大速度进行。战斗效果：攻速，不影响养成。（强化牌：放进 lord_cards）"""


@card("枫华之刃")
class 枫华之刃(_MapleBoost):
    """转移后获得25%伤害加成。战斗效果：伤害，不影响养成。（强化牌：放进 lord_cards）"""


@card("枫影无拘")
class 枫影无拘(_MapleBoost):
    """转移后5秒内免疫受到的控制。战斗效果：免控，不影响养成。（强化牌：放进 lord_cards）"""


@card("斗转星移")
class 斗转星移(_MapleBoost):
    """转移后受到当前生命值的40%的伤害，并获得15%物理与法术吸血。战斗效果：扣血 / 吸血，不影响养成。（强化牌：放进 lord_cards）"""


@card("青叶绽蕊")
class 青叶绽蕊(_MapleBoost):
    """指定的英雄在拥有护盾时，每击败1名英雄则恢复10%最大生命值。战斗效果：回血，不影响养成。（强化牌：放进 lord_cards）"""


@card("层林遮影")
class 层林遮影(_MapleBoost):
    """转移后7秒内不被敌人优先攻击。战斗效果：仇恨，不影响养成。（强化牌：放进 lord_cards）"""


@card("一叶双生")
class 一叶双生(_MapleBoost):
    """转移时在原地创造1个具有30%最大生命值，阵亡时会眩晕周围敌人的分身，分身无法攻击且被创造时嘲讽周围敌人2秒。
    战斗效果：战斗中的分身 / 嘲讽 / 眩晕，不影响养成。（强化牌：放进 lord_cards）"""


@card("瞬步无影")
class 瞬步无影(_MapleBoost):
    """开战时获得1次转移次数。战斗效果：转移次数，不影响养成。（强化牌：放进 lord_cards）"""


@card("踏叶摘星")
class 踏叶摘星(_MapleBoost):
    """转移所需要的击败数减1。战斗效果：转移条件，不影响养成。（强化牌：放进 lord_cards）"""


# ============================================================ 镜
@card("万镜")
class 万镜:
    """随机复制场上4张不同的英雄牌。（场上不同名字的英雄随机取 4 个，各复制 1 张；不足 4 个全复制。待核对：复制不扣牌库）"""
    def on_use(self, eng, t):
        names = sorted({x.name for x in eng.heroes()})
        for n in eng.pick(names, 4):
            eng.gain_card(n)


# ============================================================ 乔汐
@card("潮汐的交响")
class 潮汐的交响:
    """场上随机5名英雄等级+8。（不重复；不足 5 名全加）"""
    def on_use(self, eng, t):
        for x in eng.pick(eng.heroes(), 5):
            eng.level(x, 8, src="潮汐的交响")


@card("潮起潮落")
class 潮起潮落:
    """开战时，随机传送1个临时等级100级的英雄进入战场（不含卡牌效果），该等级每回合+40。
    实现：每场战斗开战前（on_battle_begin）召唤"潮起潮落·X"（无卡牌效果、不占人口，battle_only：引擎战斗结束移除），
    临时等级 = 100 + 40×(当前回合 − 获得回合)。
    只影响"场上单位数"类的计数（如始祖熊灵、因缘图腾普攻数），本身是战斗单位。
    待核对：随机英雄的范围（按全部英雄按牌库剩余抽，不扣牌库）；"+40"从获得的下一回合起算。"""
    def on_use(self, eng, t):
        eng.s.flags.setdefault("潮起潮落_r0", eng.s.round)
        _persist(eng, "潮起潮落", "lord_cards")

    def on_battle_begin(self, eng, holder):
        if holder is not None or not _once(eng, "潮起潮落"):
            return
        r0 = eng.s.flags.setdefault("潮起潮落_r0", eng.s.round)
        name = _draw_hero_in(eng, [1, 2, 3, 4, 5])
        g = eng.summon(f"潮起潮落·{name or '英雄'}", battle_only=True)
        if g is not None and g in eng.s.units:
            eng.temp(g, 100.0 + 40.0 * max(0, eng.s.round - r0), src="潮起潮落")


# ============================================================ 玉环
@card("扫拂")
class 扫拂:
    """使场上等级最低的2名英雄获得等于棋手等级的等级。（并列随机）"""
    def on_use(self, eng, t):
        for x in eng.lowest(eng.heroes(), 2):
            eng.level(x, eng.s.lord_level, src="扫拂")


@card("吟揉")
class 吟揉:
    """获得1张随机的转瞬英雄牌。
    实现：按商店品阶概率 + 牌库剩余抽 1 张英雄，进手牌并记为转瞬（手牌状态 transient=True）。
    待核对：随机池范围；转瞬版是否扣牌库（按不扣）。"""
    def on_use(self, eng, t):
        x = _draw_hero_shop(eng)
        if x:
            eng.gain_card(x, transient=True)


@card("绞弦")
class 绞弦:
    """场上随机英雄等级+1。（随机 1 名）"""
    def on_use(self, eng, t):
        r = eng.pick(eng.heroes(), 1)
        if r:
            eng.level(r[0], 1, src="绞弦")


@card("余音绕梁")
class 余音绕梁:
    """获得3次惊鸿调演奏次数。
    接口缺口：玉环的琵琶 / 惊鸿调机制没有建模 —— 只记 flags['惊鸿调_次数'] += 3。"""
    def on_use(self, eng, t):
        eng.s.flags["惊鸿调_次数"] = eng.s.flags.get("惊鸿调_次数", 0) + 3


# ============================================================ 庄小鱼
@card("如梦似幻")
class 如梦似幻:
    """基于场上英雄、阵营（至少3名）获取一套梦幻手牌。
    待核对："一套梦幻手牌"的内容和张数卡面没写。按最自然的读法实现：
      选场上英雄数 ≥3 的阵营（多个时取人数最多的，并列交给 eng.choose("如梦似幻_阵营")），
      获得该阵营在场英雄的同名英雄牌各 1 张（等级高的优先，最多 5 张 —— 审定用法写 hand_room:5），扣牌库；
      牌库里某张已经没了就跳过。没有阵营满 3 名时不能用（返回 False）。"""
    MAX_CARDS = 5

    def on_use(self, eng, t):
        by: dict = {}
        for x in eng.heroes():
            f = eng.faction_of(x)
            if f and f != "无阵营":
                by.setdefault(f, []).append(x)
        facs = [f for f, us in by.items() if len(us) >= 3]
        if not facs:
            return False
        top = max(len(by[f]) for f in facs)
        fac = eng.choose("如梦似幻_阵营", [f for f in facs if len(by[f]) == top])
        got = []
        for u in sorted(by[fac], key=lambda x: -x.level):
            if len(got) >= self.MAX_CARDS:
                break
            if u.name in got or eng.pool_left(u.name) <= 0:
                continue
            _gain_hero(eng, u.name)
            got.append(u.name)
        eng.s.flags["梦幻手牌"] = eng.s.flags.get("梦幻手牌", 0) + 1
        eng.note("梦幻手牌", card="如梦似幻", faction=fac, cards=got)


# ============================================================ 马可
@card("红笺锦鲤符")
class 红笺锦鲤符:
    """指定1名英雄等级+4。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        eng.level(t, 4, src="红笺锦鲤符")


@card("狼牙部落的图腾雕像")
class 狼牙部落的图腾雕像:
    """抽取1张3阶及以下的英雄牌。待核对：1~3 阶合并按牌库剩余加权抽（不看商店等级），扣牌库。"""
    def on_use(self, eng, t):
        _gain_hero(eng, _draw_hero_in(eng, [1, 2, 3]))


@card("沙海甘泉")
class 沙海甘泉:
    """获得1能量。"""
    def on_use(self, eng, t):
        eng.s.energy += 1


@card("人鱼之泪")
class 人鱼之泪:
    """随机获得1张效果牌。（见 _random_effect，待核对随机池）"""
    def on_use(self, eng, t):
        x = _random_effect(eng)
        if x:
            eng.gain_card(x)


@card("神树彩陶")
class 神树彩陶:
    """场上随机4名英雄等级+2。（不重复；不足 4 名全加）"""
    def on_use(self, eng, t):
        for x in eng.pick(eng.heroes(), 4):
            eng.level(x, 2, src="神树彩陶")


@card("冰曜宝石镜")
class 冰曜宝石镜:
    """随机获得手牌中1张非觉醒英雄牌的复制。
    待核对：手牌不记觉醒与否（全当非觉醒）；手里没有英雄牌时视为不能用（返回 False）；复制不扣牌库。"""
    def on_use(self, eng, t):
        idx = _hand_heroes(eng)
        if not idx:
            return False
        eng.gain_card(eng.s.hand[eng.pick(idx, 1)[0]])


@card("马可的随身信物")
class 马可的随身信物:
    """随机获得4张效果牌。（见 _random_effect，待核对随机池）"""
    def on_use(self, eng, t):
        for _ in range(4):
            x = _random_effect(eng)
            if x:
                eng.gain_card(x)


@card("马可的魅力飞吻")
class 马可的魅力飞吻:
    """全体英雄等级+10。"""
    def on_use(self, eng, t):
        for x in eng.heroes():
            eng.level(x, 10, src="马可的魅力飞吻")


@card("马可的限量亲签")
class 马可的限量亲签:
    """获得4能量，1张装备牌。待核对：装备牌的范围（按 8 件基础装备随机）。"""
    def on_use(self, eng, t):
        eng.s.energy += 4
        eng.gain_card(eng.pick(_basic_equips(), 1)[0])


@card("马可的吸睛自拍")
class 马可的吸睛自拍:
    """随机获得3张不同的场上英雄牌。（场上不同名字的英雄随机取 3 个，各复制 1 张。待核对：复制不扣牌库）"""
    def on_use(self, eng, t):
        names = sorted({x.name for x in eng.heroes()})
        for n in eng.pick(names, 3):
            eng.gain_card(n)


@card("终极宝藏正是在下")
class 终极宝藏正是在下:
    """随机从3个终极宝藏中选择1个。
    待核对："终极宝藏"有哪些、各是什么效果卡库里没有 —— 只记 flags['终极宝藏'] 并写日志。接口缺口：马可的宝藏牌体系未建模。"""
    def on_use(self, eng, t):
        eng.s.flags["终极宝藏"] = eng.s.flags.get("终极宝藏", 0) + 1
        eng.note("未实现", card="终极宝藏正是在下")


@card("瓜分宝藏")
class 瓜分宝藏:
    """从2种英雄强化效果中选择1个。
    待核对：两种"英雄强化效果"的内容卡库里没有 —— 只记 flags['瓜分宝藏'] 并写日志。"""
    def on_use(self, eng, t):
        eng.s.flags["瓜分宝藏"] = eng.s.flags.get("瓜分宝藏", 0) + 1
        eng.note("未实现", card="瓜分宝藏")


# ============================================================ 白歌
@card("即兴创作")
class 即兴创作:
    """随机获得场上1名英雄的复制。（复制 = 获得 1 张该英雄牌，不扣牌库。待核对）"""
    def on_use(self, eng, t):
        r = eng.pick(eng.heroes(), 1)
        if r:
            eng.gain_card(r[0].name)


@card("灵感爆发")
class 灵感爆发:
    """释放3道剑气，斩击手牌中随机3张非觉醒英雄牌，将每张牌斩为2张转瞬版复制。
    实现：随机挑手牌里 3 张英雄牌（不足 3 张全挑），每张去掉原牌、换成 2 张转瞬版（手牌状态 transient=True）。
    待核对：手牌不记觉醒与否（全当非觉醒）；转瞬版不扣牌库；手牌上限截断。"""
    def on_use(self, eng, t):
        idx = _hand_heroes(eng)
        if not idx:
            return False
        chosen = sorted(eng.pick(idx, 3), reverse=True)
        names = [eng.s.hand[i] for i in chosen]
        for i in chosen:
            eng.s.hand.pop(i)
        for n in names:
            for _ in range(2):
                eng.gain_card(n, transient=True)


@card("奇才")
class 奇才:
    """指定1名英雄，获得4张该英雄牌。待核对：是否扣公共牌库（按复制、不扣）。"""
    def on_use(self, eng, t):
        if t is None or t.kind != "hero":
            return False
        for _ in range(4):
            eng.gain_card(t.name)


@card("惊才绝艳")
class 惊才绝艳:
    """获得1张奇才：指定1名英雄，获得4张该英雄牌。"""
    def on_use(self, eng, t):
        eng.gain_card("奇才")


# ============================================================ 昭君
@card("不朽坚冰")
class 不朽坚冰:
    """战斗开始时，你的全体英雄获得500点最大生命值、60点物理和法术攻击力。
    战斗效果：生命 / 攻击，不影响养成。"""
    def on_use(self, eng, t):
        _persist(eng, "不朽坚冰", "lord_cards")


@card("雪暴时刻")
class 雪暴时刻:
    """冻结的英雄，解冻后等级永久+20。
    解冻 = batch_07.thaw(eng, 英雄)（冰之心解冻入口）最后发的 thaw 事件 → react_thaw(unit=)。"""
    def on_use(self, eng, t):
        _persist(eng, "雪暴时刻", "lord_cards")

    def react_thaw(self, eng, holder, unit=None, **kw):
        if unit is not None and unit in eng.s.units:
            eng.level(unit, 20, src="雪暴时刻")


@card("寒潮之势")
class 寒潮之势:
    """战斗开始时，你等级最高的英雄获得1200点生命值、120点物理和法术攻击力。
    战斗效果：生命 / 攻击，不影响养成。"""
    def on_use(self, eng, t):
        _persist(eng, "寒潮之势", "lord_cards")
