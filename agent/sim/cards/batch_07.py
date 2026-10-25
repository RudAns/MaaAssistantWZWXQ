"""第 7 批：王昭君（冰之心）强化、鲁班（神匠）机关、明先生卦象、阵营天赋、经营类决策牌、香香弹药（战斗）、装备箱 / 能量转换 / 挂帅 / 神秘英雄牌。

约定（本批自定，引擎里没有的部分见各处"接口缺口"）：
- 一次性选到、之后持续生效的牌（冰封强化、机关强化、经营决策、卦象强化…）打出时放进 eng.s.lord_cards，
  靠 on_round_start / on_battle_end / react_* 持续结算（holder = None）。同名不重复放（待核对：能否叠加）。
- 冰之心"解冻"：本模块提供 thaw(eng, u)，按 lord_cards 里的冰封强化决定变成谁，并发 "thaw" 事件（react_thaw）。
- 2026-10-14 统一接口：棋手生命值 eng.s.hp / eng.add_hp；排名 eng.s.rank；淘汰事件 eliminated(n)；能量下限
  eng.s.energy_floor；回合结束 on_round_end；战斗存活 eng.survivors()（引擎记 s.battle_dead，复生后活下来的算存活）；
  觉醒进度 eng.awaken_step；"打出英雄牌" eng.is_card_use()；神秘英雄牌是 in_hand 的牌（回合开始 / 手动刷新由引擎调）。
"""
from __future__ import annotations

import math

from agent.engine import board as B
from ..engine import card, card_db, category, faction, quality, card_text, impl, FACTION_ALIAS


# ============================================================ 辅助函数（本批私有）
def _name(obj) -> str:
    return type(obj).names[0]


def _add_lord(eng, name: str) -> None:
    """持续生效的强化 / 决策：放进 lord_cards（不重复）。"""
    if name not in eng.s.lord_cards:
        eng.s.lord_cards.append(name)


class _Persist:
    """打出 = 进入 lord_cards 持续生效。"""
    def on_use(self, eng, t):
        _add_lord(eng, _name(self))


def _once(eng, key: str) -> bool:
    """本回合第一次返回 True（flags 里 round: 前缀每回合清空）。"""
    k = "round:" + key
    if eng.s.flags.get(k):
        return False
    eng.s.flags[k] = True
    return True


def _add_awaken(eng, u, n: int, src: str = "") -> None:
    """觉醒进度 +n（engine.awaken_step，凑满 3 = 觉醒 = 回手再打出一次）。"""
    eng.awaken_step(u, n, src=src)


def _draw_hero_min_tier(eng, tmin: int, fac: str | None = None, exclude: str = ""):
    """按牌库剩余张数，从品阶 ≥ tmin 的英雄里抽一张。"""
    from ..engine import faction as _fac
    names = [n for n, c in card_db().items() if c.get("category") == "英雄牌" and quality(n) >= tmin
             and n != exclude and (not fac or _fac(n) == FACTION_ALIAS.get(fac, fac))]
    ws = [eng.pool_left(n) for n in names]
    if not names or sum(ws) <= 0:
        return None
    return eng.rng.choices(names, ws)[0]


def _draw_hero_tier(eng, tier: int, fac: str | None = None, exclude: str = ""):
    for _ in range(20):
        x = eng.draw_hero(tier, fac)
        if x and x != exclude:
            return x
    return None


def _transform(eng, u, new: str, src: str = "") -> None:
    """英雄"变为"另一名英雄：继承等级 / 临时等级 / 装备 / 格子 / 觉醒进度（待核对：觉醒进度是否继承），不触发登场。"""
    if not new or u not in eng.s.units:
        return
    old = u.name
    eng.s.pool_taken[old] = max(0, eng.s.pool_taken.get(old, 0) - 1)
    eng.take(new)
    eng.rename(u, new, src=src)


def _equips(q_pred) -> list:
    return [n for n, c in card_db().items() if c.get("category") == "装备牌" and q_pred(n)]


def basic_equips() -> list:
    """基础装备：1 品阶、卡面"可以铸造成"（风暴巨剑 / 雷鸣刃 / 速击之枪 / 大棒 / 守护者之铠 / 力量腰带 / 圣者法典 / 破碎圣杯）。"""
    return _equips(lambda n: quality(n) == 1 and "铸造" in card_text(n))


def normal_equips() -> list:
    """普通装备（成品）：2 品阶。"""
    return _equips(lambda n: quality(n) == 2)


def special_equips() -> list:
    """特殊装备。待核对：按"品阶 ≥3 的装备"认（机关匣 / 贤者的庇护 / 星源… 都在这一档）。"""
    return _equips(lambda n: quality(n) >= 3)


def _gain_choice(eng, kind: str, pool: list, k: int = 3) -> None:
    opts = eng.pick(pool, k)
    pick = eng.choose(kind, opts)
    if pick:
        eng.gain_card(pick)


# ============================================================ 王昭君：冰之心（冻结 → 解冻）的强化
THAW_MODS = ["寒冰王座", "冰封结晶", "寒冰觉醒", "霜寒进化", "升华", "破冰裂变", "雪暴时刻"]   # 解冻类强化
FROST_BUFFS = ["冰心领域", "寒雨凝结", "寒潮共振"]                                         # 回合开始类增益


def thaw(eng, u) -> None:
    """冰之心解冻：默认"随机变为高 1 阶的英雄，继承等级"，再按 lord_cards 里的冰封强化改写：
      寒冰王座 → 变为 5 阶随机英雄；破冰裂变 → 高 2 阶；升华 → 高 1 阶；霜寒进化 → 同阵营高 1 阶；
      寒冰觉醒（且没有别的变身强化）→ 英雄不变；升华 / 寒冰觉醒 各觉醒进度 +2；冰封结晶 → 额外 1 张解冻后英雄品阶的兑换券；
      破冰裂变 → 下回合起冰之心每回合可用 2 次（flags["冰之心_每回合次数"]、["冰之心_2次起始回合"]）。
    最后发 "thaw" 事件（react_thaw(unit=u)，给雪暴时刻这类"解冻后"的牌）。
    接口约定：冰之心（未实现）解冻时必须调用本函数，不要自己变身再 emit("thaw") —— 强化改写的是解冻本身
    （"英雄不变""高 2 阶"），只发事件的话这些强化不会生效。
    待核对：多张变身类强化同时存在时谁优先（这里按 寒冰王座 > 破冰裂变 > 升华/霜寒进化 > 寒冰觉醒 > 默认）；
            已是 5 阶时"高 1 阶"按仍为 5 阶随机英雄（不同名）处理。"""
    if u is None or u not in eng.s.units:
        return
    lc = set(eng.s.lord_cards)
    q = quality(u.name)
    fac = None
    tier = q + 1
    keep = False
    if "寒冰王座" in lc:
        tier = 5
    elif "破冰裂变" in lc:
        tier = q + 2
    elif "升华" in lc or "霜寒进化" in lc:
        tier = q + 1
        if "霜寒进化" in lc and "升华" not in lc:
            fac = faction(u.name) or None
    elif "寒冰觉醒" in lc:
        keep = True
    if not keep:
        new = _draw_hero_tier(eng, min(5, max(1, tier)), fac, exclude=u.name)
        _transform(eng, u, new, src="冰之心解冻")
    n_aw = 2 * (("升华" in lc) + ("寒冰觉醒" in lc))
    if n_aw:
        _add_awaken(eng, u, n_aw, src="冰封强化")
    if "冰封结晶" in lc:
        eng.gain_card(f"{min(5, max(1, quality(u.name)))}阶英雄兑换券")
    if "破冰裂变" in lc:
        eng.s.flags["冰之心_每回合次数"] = 2
        eng.s.flags.setdefault("冰之心_2次起始回合", eng.s.round + 1)
    eng.emit("thaw", unit=u)


@card("寒冰王座")
class 寒冰王座(_Persist):
    """解冻后，英雄变为5阶的随机英雄。（冰封强化：进 lord_cards，结算在 thaw()）"""


@card("冰封结晶")
class 冰封结晶(_Persist):
    """解冻后，额外获得1张解冻英雄品阶的兑换券。（结算在 thaw()；待核对："解冻英雄品阶"取解冻后的品阶）"""


@card("寒冰觉醒")
class 寒冰觉醒(_Persist):
    """解冻后，英雄不变，觉醒进度额外+2。（结算在 thaw()）"""


@card("霜寒进化")
class 霜寒进化(_Persist):
    """解冻后，英雄变为同阵营高1阶的随机英雄。（结算在 thaw()）"""


@card("升华")
class 升华(_Persist):
    """解冻后，英雄变为高1阶的英雄，觉醒进度额外+2。（结算在 thaw()；待核对：卡面没写"随机"，按随机高 1 阶实现）"""


@card("破冰裂变")
class 破冰裂变(_Persist):
    """解冻后，英雄变为高2阶的英雄，下回合起，技能单回合可使用2次。（结算在 thaw()；"技能" = 冰之心）"""


@card("冰心领域")
class 冰心领域(_Persist):
    """回合开始时，场上友方英雄获得场上品阶总数35%的等级。
    待核对：取整方式（这里向下取整）；"场上品阶总数" = 场上英雄品阶之和。"""
    def on_round_start(self, eng, holder):
        hs = eng.heroes()
        n = math.floor(sum(quality(h.name) for h in hs) * 0.35)
        for h in hs:
            eng.level(h, n, src="冰心领域")


@card("寒雨凝结")
class 寒雨凝结(_Persist):
    """回合开始时，场上友方英雄获得主要阵营人数的等级。（主要阵营 = 场上人数最多的阵营；无阵营不算）"""
    def on_round_start(self, eng, holder):
        hs = eng.heroes()
        cnt: dict = {}
        for h in hs:
            f = eng.faction_of(h)
            if f and f != "无阵营":
                cnt[f] = cnt.get(f, 0) + 1
        n = max(cnt.values()) if cnt else 0
        for h in hs:
            eng.level(h, n, src="寒雨凝结")


@card("寒潮共振")
class 寒潮共振(_Persist):
    """回合开始时，场上友方英雄获得已觉醒英雄数量×1.5的等级。待核对：取整方式（这里向下取整）。"""
    def on_round_start(self, eng, holder):
        hs = eng.heroes()
        n = math.floor(sum(1 for h in hs if h.awakened) * 1.5)
        for h in hs:
            eng.level(h, n, src="寒潮共振")


@card("深度冻结")
class 深度冻结:
    """从随机3个强化效果中选择1个。
    待核对：强化池按"解冻类强化"（寒冰王座 / 冰封结晶 / 寒冰觉醒 / 霜寒进化 / 升华 / 破冰裂变 / 雪暴时刻）实现，排除已拥有的；
            卡面没写冻结目标，只做"选强化"。"""
    def on_use(self, eng, t):
        pool = [x for x in THAW_MODS if x not in eng.s.lord_cards]
        pick = eng.choose("frost_mod", eng.pick(pool, 3))
        if pick:
            _add_lord(eng, pick)


@card("冰雪秘藏")
class 冰雪秘藏:
    """从3个增益效果中选择1个。
    待核对：3 个增益按 冰心领域 / 寒雨凝结 / 寒潮共振（回合开始类增益，正好 3 个）实现。"""
    def on_use(self, eng, t):
        pick = eng.choose("frost_buff", list(FROST_BUFFS))
        if pick:
            _add_lord(eng, pick)


# ============================================================ 鲁班（神匠）：机关
MECHS = ["热能陷阱", "电磁弹簧", "等离子屏障"]
MECH_UPGRADES = ["热能陷阱强化", "电磁弹簧强化", "等离子屏障强化"]


def _place_mech(eng, name):
    """放置机关：作为 object 单位占一格、不占人口、1 级。待核对：机关是否占格（挡英雄落点）、放在哪一格（交给推演打法）。"""
    free = eng.free_cells()
    cell = eng.choose("mech_cell", free, mech=name) if free else None
    return eng.deploy(name, cell=cell, kind="object", pop=False)


@card("便携式机关")
class 便携式机关:
    """从热能陷阱、电磁弹簧、等离子屏障3种机关中选择1个。（选中的机关牌进手牌）"""
    def on_use(self, eng, t):
        pick = eng.choose("mech", list(MECHS))
        if pick:
            eng.gain_card(pick)


@card("热能陷阱")
class 热能陷阱:
    """可放置1个热能陷阱。敌方踩中时，对其造成真实伤害并眩晕。
    养成上：放一个机关（占格）；伤害 / 眩晕是战斗效果，不影响养成。"""
    def on_use(self, eng, t):
        if _place_mech(eng, "热能陷阱") is None:
            return False


@card("电磁弹簧")
class 电磁弹簧:
    """可放置1个电磁弹簧。敌方踩中时触发，将其弹至敌方战场后排，触发2次后消失。
    养成上：放一个机关（占格）；弹飞是战斗效果，不影响养成。"""
    def on_use(self, eng, t):
        if _place_mech(eng, "电磁弹簧") is None:
            return False


@card("等离子屏障")
class 等离子屏障:
    """可放置1个等离子屏障。可在横向展开阻碍敌方行动。
    养成上：放一个机关（占格）；阻挡是战斗效果，不影响养成。"""
    def on_use(self, eng, t):
        if _place_mech(eng, "等离子屏障") is None:
            return False


@card("构筑升级")
class 构筑升级:
    """指定场上1个机关升级其效果。（机关等级 +1；待核对：上限按 3 级，卡面"3级屏障 / 3级陷阱"）"""
    def on_use(self, eng, t):
        if t is None or t not in eng.s.units or t.name not in MECHS or t.level >= 3:
            return False
        eng.level(t, 1, src="构筑升级")


@card("成品方案")
class 成品方案:
    """从3种机关强化方案中选择1个。（热能陷阱强化 / 电磁弹簧强化 / 等离子屏障强化，进 lord_cards）"""
    def on_use(self, eng, t):
        pick = eng.choose("mech_upgrade", list(MECH_UPGRADES))
        if pick:
            _add_lord(eng, pick)


@card("等离子屏障强化")
class 等离子屏障强化(_Persist):
    """屏障生命值+100%，3级屏障会在战斗开始时为屏障左右1格的友军提供复生效果。
    养成上：复生多一条命（影响阵亡 / 牺牲触发次数）；屏障生命值是战斗效果。
    待核对："左右 1 格"按同一排、格距 1 的英雄。"""
    def react_battle_start(self, eng, holder, unit=None, **kw):
        if not _once(eng, "等离子屏障强化"):
            return
        for m in [x for x in eng.s.units if x.name == "等离子屏障" and x.level >= 3 and x.cell]:
            r = B.row_of(m.cell)
            for h in eng.heroes():
                if h.cell and B.row_of(h.cell) == r and eng.dist(m, h) == 1:
                    h.counters["revive"] = h.counters.get("revive", 0) + 1


@card("电磁弹簧强化")
class 电磁弹簧强化(_Persist):
    """弹簧被友军触发时，使友军获得50%生命值上限的磁力护盾，护盾存在期间该单位造成的伤害提升50%。
    战斗效果：护盾 / 增伤，不影响养成。"""


@card("热能陷阱强化")
class 热能陷阱强化(_Persist):
    """陷阱会延时2秒爆炸且爆炸范围增加1格，3级陷阱的冷却时间缩短至4秒。
    战斗效果：爆炸范围 / 冷却，不影响养成。"""


# ============================================================ 战斗开始复制 / 明先生卦象 / 阵营天赋
@card("驭势")
class 驭势(_Persist):
    """战斗开始时，将敌方场上所有英雄复制进你的手牌。
    待核对：看不到对手，敌方人数取同等级人口上限、每人按我方棋手等级的商店品阶抽（复制品不占牌库）；手牌上限照常截断。"""
    def react_battle_start(self, eng, holder, unit=None, **kw):
        if not _once(eng, "驭势"):
            return
        n = B.pop_cap(max(1, min(6, eng.s.lord_level)))
        for _ in range(n):
            x = eng.draw_hero(None)
            if x:
                eng.gain_card(x)


@card("神之一手")
class 神之一手:
    """战斗开始时，将敌方等级最高的英雄复制，临时加入我方战场。
    战斗效果：临时多一个战斗单位，不影响养成。"""


@card("飞龙在天")
class 飞龙在天(_Persist):
    """强化明先生的卦象。
    卦象·节：每回合额外获得4能量。
    卦象·丰：每回合获得8次临时刷新。
    当前卦象读 flags["卦象"]（"节" / "丰"，或 (名, 剩余回合)）—— 由先知卦象写入（接口缺口）。
    待核对：这里的数值是"强化后"的完整效果还是在原卦象上追加（这里按追加结算）；临时刷新记在 free_refresh。"""
    def on_round_start(self, eng, holder):
        g = eng.s.flags.get("卦象")
        if isinstance(g, (tuple, list)):
            g = g[0] if (len(g) < 2 or g[1] > 0) else None
        if g == "节":
            eng.s.energy += 4
        elif g == "丰":
            eng.s.free_refresh += 8


def _chosen_faction(eng) -> str:
    """"技能所选阵营"：读 flags["所选阵营"]（棋手技能写入，接口缺口）；没有就让推演打法选一个并记下。"""
    f = eng.s.flags.get("所选阵营")
    if not f:
        f = eng.choose("faction", ["河洛", "逐鹿", "大河流域", "日落海", "三分之地"])
        eng.s.flags["所选阵营"] = f
    return FACTION_ALIAS.get(f, f)


def _faction_talents(fac: str) -> list:
    shorts = {fac, fac.replace("流域", "").replace("之地", "")}
    return [n for n, c in card_db().items() if c.get("category") == "天赋牌" and "(赠送)" not in n
            and any(f"【{s}】" in card_text(n) for s in shorts)]


def gain_talent(eng, name: str) -> None:
    """获得一个天赋：进 talents；有 on_use 的立即结算（拿到时立即生效 / 送牌）。"""
    eng.s.talents.append(name)
    c = impl(name)
    if c is not None and hasattr(c, "on_use"):
        c.on_use(eng, None)
    eng.note("天赋", card=name)


@card("打成一片")
class 打成一片:
    """天赋3选1（技能选择阵营相关的天赋）。
    待核对：天赋池 = 卡面写到【所选阵营】的天赋牌，随机 3 个；所选阵营见 _chosen_faction。"""
    def on_use(self, eng, t):
        fac = _chosen_faction(eng)
        pool = [n for n in _faction_talents(fac) if n not in eng.s.talents]
        pick = eng.choose("talent", eng.pick(pool, 3), faction=fac)
        if pick:
            gain_talent(eng, pick)


@card("朋友多多")
class 朋友多多:
    """随机抽取4张技能所选阵营的英雄牌。（按棋手等级的商店品阶、牌库剩余抽）"""
    def on_use(self, eng, t):
        fac = _chosen_faction(eng)
        for _ in range(4):
            for _try in range(10):
                x = eng.draw_hero(None, fac)
                if x:
                    eng.take(x)
                    eng.gain_card(x)
                    break


@card("狂欢时刻")
class 狂欢时刻:
    """获得1张打成一片。"""
    def on_use(self, eng, t):
        eng.gain_card("打成一片")


# ============================================================ 经营类决策（进 lord_cards 持续生效）
@card("投资增值")
class 投资增值(_Persist):
    """棋手升级时获得2能量，满级后每回合获得2次临时刷新。（临时刷新记在 free_refresh）"""
    def react_lord_levelup(self, eng, holder, level=0, **kw):
        eng.s.energy += 2

    def on_round_start(self, eng, holder):
        if eng.s.lord_level >= B.MAX_LORD_LEVEL:
            eng.s.free_refresh += 2


@card("风险对冲")
class 风险对冲(_Persist):
    """战斗失败时额外获得2次刷新，每回合开始时恢复1棋手生命值。
    待核对："2 次刷新"是否只在本回合有效（这里记进 free_refresh）。"""
    def on_battle_end(self, eng, holder, win=False):
        if not win:
            eng.s.free_refresh += 2

    def on_round_start(self, eng, holder):
        eng.add_hp(1, src="风险对冲")


@card("固定收益")
class 固定收益(_Persist):
    """每回合额外获得1能量和1次临时刷新。（回合开始时结算）"""
    def on_round_start(self, eng, holder):
        eng.s.energy += 1
        eng.s.free_refresh += 1


@card("基层提拔")
class 基层提拔(_Persist):
    """打出1-3阶英雄牌时其等级+3。（打出 = 使用英雄牌的登场：上场 / 合成同名；被触发的登场不算，eng.is_card_use）"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if unit is not None and unit.kind == "hero" and 1 <= quality(unit.name) <= 3 and eng.is_card_use():
            eng.level(unit, 3, src="基层提拔")


@card("空降人才")
class 空降人才(_Persist):
    """本场比赛打出的前2张4-5阶英雄觉醒进度+2。
    "打出" = eng.is_card_use()（被触发的登场不算）。
    待核对：从拿到这张牌之后开始计数；已觉醒的不再计（避免觉醒后回手重打又算一张）。"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if unit is None or unit.kind != "hero" or quality(unit.name) < 4 or unit.awakened or not eng.is_card_use():
            return
        n = eng.s.flags.get("空降人才_n", 0)
        if n >= 2:
            return
        eng.s.flags["空降人才_n"] = n + 1
        _add_awaken(eng, unit, 2, src="空降人才")


@card("沉淀收益")
class 沉淀收益(_Persist):
    """回合开始时，获得0能量。每在当前棋手等级停留1回合，额外获得1能量，至多4能量。
    待核对：从拿到这张牌的回合开始计停留回合数。"""
    def on_round_start(self, eng, holder):
        f = eng.s.flags
        if f.get("沉淀收益_lv") == eng.s.lord_level:
            f["沉淀收益_stay"] = f.get("沉淀收益_stay", 0) + 1
        else:
            f["沉淀收益_lv"] = eng.s.lord_level
            f["沉淀收益_stay"] = 0
        eng.s.energy += min(4, f["沉淀收益_stay"])


@card("并购红利")
class 并购红利(_Persist):
    """合成英雄时，获得1能量，每回合限3次。"""
    def react_merge(self, eng, holder, unit=None, **kw):
        k = "round:并购红利_n"
        n = eng.s.flags.get(k, 0)
        if n < 3:
            eng.s.flags[k] = n + 1
            eng.s.energy += 1


@card("低压运营")
class 低压运营(_Persist):
    """回合开始时，每有15棋手生命值则场上所有英雄等级+1。（棋手生命值 eng.s.hp）"""
    def on_round_start(self, eng, holder):
        n = int(eng.s.hp) // 15
        for h in eng.heroes():
            eng.level(h, n, src="低压运营")


@card("资本杠杆")
class 资本杠杆(_Persist):
    """回合结束时每有7能量，获得1能量。（engine.round_end 的 on_round_end：下回合收入到账之前）"""
    def on_round_end(self, eng, holder):
        eng.s.energy += max(0, eng.s.energy) // 7


@card("资产清算")
class 资产清算(_Persist):
    """若你累计出售了10名场上的1-3阶英雄，获得10能量。（一次性）
    待核对："累计"从拿到这张牌之后开始数（之前卖过的引擎没记）。"""
    def react_sell(self, eng, holder, unit=None, **kw):
        f = eng.s.flags
        if f.get("资产清算_done") or unit is None or unit.kind != "hero" or not 1 <= quality(unit.name) <= 3:
            return
        f["资产清算_n"] = f.get("资产清算_n", 0) + 1
        if f["资产清算_n"] >= 10:
            f["资产清算_done"] = True
            eng.s.energy += 10


@card("胜场激励")
class 胜场激励(_Persist):
    """若你战斗胜利2次，获得10能量。（一次性；从拿到这张牌之后数）"""
    def on_battle_end(self, eng, holder, win=False):
        f = eng.s.flags
        if f.get("胜场激励_done") or not win:
            return
        f["胜场激励_n"] = f.get("胜场激励_n", 0) + 1
        if f["胜场激励_n"] >= 2:
            f["胜场激励_done"] = True
            eng.s.energy += 10


@card("分红")
class 分红(_Persist):
    """战斗结束时，你每存活1名英雄，获得1能量。（存活 = eng.survivors()：复生后活下来的也算存活）"""
    def on_battle_end(self, eng, holder, win=False):
        eng.s.energy += len(eng.survivors())


@card("人才储备")
class 人才储备(_Persist):
    """回合开始时，若棋手等级高于3，获得1张人才搜寻。"""
    def on_round_start(self, eng, holder):
        if eng.s.lord_level > 3:
            eng.gain_card("人才搜寻")


@card("投资回报")
class 投资回报(_Persist):
    """战斗开始时，若本回合累计打出了不少于8张英雄及效果牌，随机获得其中2张。
    打出 = 使用英雄牌的登场（含合成，eng.is_card_use）+ 效果牌使用事件，记在 flags["round:投资回报"]。待核对："其中 2 张"可以是同名。"""
    def _rec(self, eng, name):
        eng.s.flags.setdefault("round:投资回报", []).append(name)

    def react_deploy(self, eng, holder, unit=None, **kw):
        if unit is not None and unit.kind == "hero" and eng.is_card_use():
            self._rec(eng, unit.name)

    def react_effect_card(self, eng, holder, card="", target=None, **kw):
        if card:
            self._rec(eng, card)

    def react_battle_start(self, eng, holder, unit=None, **kw):
        if not _once(eng, "投资回报_fired"):
            return
        played = eng.s.flags.get("round:投资回报") or []
        if len(played) >= 8:
            for x in eng.pick(played, 2):
                eng.gain_card(x)


@card("首购激励")
class 首购激励(_Persist):
    """每回合购买的第1张牌会额外复制1张。
    购买 = engine.buy 发的 buy 事件（复制的那张不扣牌库）。"""
    def react_buy(self, eng, holder, card="", **kw):
        if card and _once(eng, "首购激励"):
            eng.gain_card(card)


@card("管理进修")
class 管理进修(_Persist):
    """获得3棋手经验，棋手升级时再次获得。（先挂上再加经验：这 3 点经验引起的升级也会再给 3 点）"""
    def on_use(self, eng, t):
        _add_lord(eng, "管理进修")
        eng.add_xp(3, src="管理进修")

    def react_lord_levelup(self, eng, holder, level=0, **kw):
        eng.s.xp += 3


@card("战略性收缩")
class 战略性收缩(_Persist):
    """若你战斗失败2次，获得9棋手经验。（一次性；从拿到这张牌之后数）"""
    def on_battle_end(self, eng, holder, win=False):
        f = eng.s.flags
        if f.get("战略性收缩_done") or win:
            return
        f["战略性收缩_n"] = f.get("战略性收缩_n", 0) + 1
        if f["战略性收缩_n"] >= 2:
            f["战略性收缩_done"] = True
            eng.add_xp(9, src="战略性收缩")


@card("短期企划")
class 短期企划(_Persist):
    """战斗开始时，你的英雄获得10临时等级。战斗结束时仍存活的英雄保留5级。（存活 = eng.survivors()）"""
    def react_battle_start(self, eng, holder, unit=None, **kw):
        if not _once(eng, "短期企划"):
            return
        for h in eng.heroes():
            eng.temp(h, 10, src="短期企划")

    def on_battle_end(self, eng, holder, win=False):
        for h in eng.survivors():
            eng.level(h, 5, src="短期企划")


@card("股权激励")
class 股权激励(_Persist):
    """回合开始时，排名最高的玩家获得1件特殊装备；第2高的玩家获得1件普通装备。
    只结算我方：排名读 eng.s.rank（None = 不知道，不结算）。
    待核对：发哪一件（这里在对应池里随机 1 件）；"特殊装备"的范围见 special_equips()。"""
    def on_round_start(self, eng, holder):
        r = eng.s.rank
        if r == 1:
            pool = special_equips()
        elif r == 2:
            pool = normal_equips()
        else:
            return
        if pool:
            eng.gain_card(eng.pick(pool, 1)[0])


@card("精英计划")
class 精英计划(_Persist):
    """战斗开始时，场上等级最高的友方英雄最大法力值-20。
    战斗效果：法力值，不影响养成。"""


@card("协同增效")
class 协同增效(_Persist):
    """英雄合成时等级额外+4。"""
    def react_merge(self, eng, holder, unit=None, **kw):
        if unit is not None and unit in eng.s.units:
            eng.level(unit, 4, src="协同增效")


@card("猎头挖角")
class 猎头挖角(_Persist):
    """战斗开始时，复制对方等级最高的英雄加入战斗。
    战斗效果：临时多一个战斗单位，不影响养成。"""


@card("基层强化")
class 基层强化(_Persist):
    """战斗开始时，1-3阶英雄伤害提升50%。
    战斗效果：伤害，不影响养成。"""


# ============================================================ 特权决策
PRIVILEGES = ["绝对压制", "贷款协议", "资源再分配"]


def _privileged(eng, name: str) -> bool:
    return eng.s.flags.get("特权_颁布") == name


@card("特权决策")
class 特权决策:
    """选择1个特权决策颁布，对所有玩家生效，并且自己额外享受特权效果。
    选项 = 卡面带"决策颁布者额外特权"的 3 张（绝对压制 / 贷款协议 / 资源再分配）；只模拟我方。"""
    def on_use(self, eng, t):
        pick = eng.choose("privilege", list(PRIVILEGES))
        if not pick:
            return
        eng.s.flags["特权_颁布"] = pick
        c = impl(pick)
        if c is not None:
            c.on_use(eng, None)


@card("绝对压制")
class 绝对压制(_Persist):
    """战斗开始时，你等级最高的英雄变成猪猪，持续3秒，并在本局对战中获得30%伤害减免。决策颁布者额外特权：你的该英雄额外获得20%伤害提升。
    战斗效果：变猪 / 减伤 / 增伤，不影响养成。"""


@card("贷款协议")
class 贷款协议(_Persist):
    """能量可用至-15。回合开始时若能量为负，则本回合获得的能量-2。决策颁布者额外特权：能量下限为-20且不受到惩罚。
    能量下限 eng.s.energy_floor（engine.spend 认它）。
    待核对："本回合获得的能量-2"按回合开始时直接扣 2 能量实现（收入在回合开始前已经到账）。"""
    def on_use(self, eng, t):
        _add_lord(eng, "贷款协议")
        eng.s.energy_floor = min(eng.s.energy_floor, -20 if _privileged(eng, "贷款协议") else -15)

    def on_round_start(self, eng, holder):
        if eng.s.energy < 0 and not _privileged(eng, "贷款协议"):
            eng.s.energy -= 2


@card("资源再分配")
class 资源再分配(_Persist):
    """有棋手被淘汰时，获得5能量以及1张基础装备牌。决策颁布者额外特权：改为获得特殊装备牌且额外恢复5棋手生命值。
    淘汰 = engine.eliminate(n) 发的 eliminated 事件，每淘汰 1 位结算 1 次。
    待核对：发哪一件（对应池里随机 1 件）。"""
    def react_eliminated(self, eng, holder, n=1, **kw):
        for _ in range(max(0, int(n))):
            eng.s.energy += 5
            priv = _privileged(eng, "资源再分配")
            pool = special_equips() if priv else basic_equips()
            if pool:
                eng.gain_card(eng.pick(pool, 1)[0])
            if priv:
                eng.add_hp(5, src="资源再分配")


# ============================================================ 香香（孙尚香棋手）攻击模式的弹药强化：全部是战斗效果
def _combat_only(name: str, text: str):
    cls = type(name, (_Persist,), {"__doc__": f"{text}\n    战斗效果：香香的攻击模式强化，不影响养成。"})
    card(name)(cls)
    return cls


for _n, _t in [
    ("补给弹匣", "若子弹击败敌方英雄，立即补充10发子弹。"),
    ("额外弹匣", "发射的炮弹数量+2。"),
    ("破甲弹头", "炮弹命中的目标会降低20点物理防御，可叠加。"),
    ("燃烧弹头", "炮弹命中后会留下一片燃烧区域，每秒对敌人造成物理伤害，持续10秒。"),
    ("制导弹头", "炮弹初始拥有最大爆炸范围且对同一目标发射。"),
    ("裂伤弹", "子弹附带目标当前生命值5%的额外物理伤害。"),
    ("干扰弹", "子弹命中的目标在5秒内沉默且受到的治疗效果降低70%。"),
    ("贯穿弹头", "狙击弹伤害不再因穿透而衰减。"),
    ("大号弹头", "狙击弹额外造成目标最大生命值60%的物理伤害，但无法穿透。"),
    ("凝滞弹头", "优先攻击敌方等级最高3名英雄中的随机1名，狙击弹命中的目标4秒内被缴械且禁锢。"),
    ("歼灭补给", "额外装填的狙击弹不再有数量限制。"),
    ("超级电池", "激光持续时间增加5秒。"),
    ("聚焦光束", "激光对同一目标发射超过2秒后，伤害+80%。"),
    ("殉爆模式", "激光命中过的目标在阵亡时会爆炸，对一定范围内的敌人造成物理伤害，并眩晕1秒。"),
    ("高能激发", "激光伤害+100%，但不会攻击生命值低于20%的目标。"),
]:
    _combat_only(_n, _t)


# ============================================================ 杂项：装备箱 / 能量转换 / 挂帅 / 神秘英雄牌
@card("装备箱")
class 装备箱:
    """使用：挑选1件装备（弹出装备N选1）。
    待核对：选项池与张数 —— 按"普通装备（2 品阶成品）随机 3 件"实现；选中的进手牌。"""
    def on_use(self, eng, t):
        _gain_choice(eng, "equip", normal_equips(), 3)


@card("能量转换")
class 能量转换:
    """消耗3能量，获得与棋手等级相对应的效果锦囊牌（不超过5阶）。（能量不足 3 不能用；eng.spend，认能量下限）"""
    def on_use(self, eng, t):
        if eng.s.energy < 3 or not eng.spend(3, src="能量转换"):
            return False
        eng.gain_card(f"{max(1, min(eng.s.lord_level, 5))}阶效果锦囊")


@card("挂帅")
class 挂帅:
    """指定1名【三分】英雄，使其等级等同于1名最近的【三分】英雄。
    待核对：按真实等级"等同"（对方更低时也会降下来，按卡面字面）；临时等级不算。"""
    def on_use(self, eng, t):
        if t is None or t not in eng.s.units or not eng.is_faction(t, "三分之地"):
            return False
        n = eng.nearest(t, lambda x: eng.is_faction(x, "三分之地"))
        if not n:
            return False
        eng.level(t, n[0].level - t.level, src="挂帅")


MYSTERY = "神秘英雄牌"


@card("神秘英雄牌")
class 神秘英雄牌:
    """手动刷新时，随机变为不低于当前棋手对应品阶的英雄牌。每回合有15%概率自动销毁并获得3能量。
    in_hand = True：拿在手里时引擎把它当持有牌 —— 回合开始（on_round_start）调 mystery_round_start，
    手动刷新（eng.refresh_shop 的 refresh 事件）调 mystery_on_refresh；同名多张一次处理完。
    待核对："当前棋手对应品阶" = min(棋手等级, 5)；变成的英雄从品阶 ≥ 该值的英雄里按牌库剩余抽（会拿走牌库）。不能直接打出。"""
    in_hand = True

    def on_use(self, eng, t):
        return False

    def on_round_start(self, eng, holder):
        n = mystery_round_start(eng)
        if n:
            eng.note("神秘英雄牌销毁", n=n, energy=3 * n)

    def react_refresh(self, eng, holder, **kw):
        mystery_on_refresh(eng)


def mystery_on_refresh(eng) -> int:
    """手动刷新：手里每张神秘英雄牌变成一张英雄牌。返回变了几张。"""
    n = 0
    hand = eng.s.hand
    for i, x in enumerate(list(hand)):
        if x != MYSTERY:
            continue
        h = _draw_hero_min_tier(eng, max(1, min(eng.s.lord_level, 5)))
        if h:
            eng.take(h)
            hand[i] = h
            eng.s.hand_meta.setdefault(h, []).append({"got": eng.s.round})
            eng.note("神秘英雄牌", to=h)
            n += 1
    return n


def mystery_round_start(eng) -> int:
    """回合开始：每张神秘英雄牌 15% 自动销毁并获得 3 能量。返回销毁几张。"""
    keep, n = [], 0
    for x in eng.s.hand:
        if x == MYSTERY and eng.rng.random() < 0.15:
            eng.s.energy += 3
            n += 1
        else:
            keep.append(x)
    eng.s.hand[:] = keep
    return n
