"""第 5 批（79 张）：3 阶天赋为主（派系援军、阿科米亚核心、古币机、羁绊送牌）+ 云中君的 6 件神兵装备 + 巨像的两只神兽。

只实现养成相关的效果；战斗效果写在 docstring 里。2026-10-14 统一接口后：
  - 阿科米亚核心 = eng.s.core / eng.core_add；开战分配由引擎 core_distribute 统一做（定向传输 = 类属性 core_direct_cap）。
  - 演武"获得的战术牌变高阶" = gain_redirect（引擎 gain_card 拦截），不再轮询。
  - 卖手牌 = engine.sell_card 发 sell_card 事件（丝绸之路"出售古币"）；淘汰 = engine.eliminate 发 eliminated 事件（拾荒者）。
  - "打出英雄牌"用 eng.is_card_use()；只在本场战斗存在的召唤物 eng.summon(battle_only=True)。
  - 复现"附加转瞬"记在手牌状态（eng.gain_card(..., transient=True)）。
仍未解决：等级变化没有事件（积少成多"每提升 10 级"轮询）；技能释放没有逐次事件（旋刃、穷兵黩武）。
"""
from __future__ import annotations

import math

from ..engine import card, card_db, faction, quality, has_keyword, FACTION_ALIAS

FACTIONS = ("河洛", "逐鹿", "大河流域", "日落海", "三分之地")

# 战术牌（逐鹿效果牌，带"战术"二字的 5 张，各有"高阶"版）
TACTICS = ["援护战术", "招募战术", "演练战术", "推演战术", "特训战术"]
# 火鹰装备 = 日落海专属装备（card_kb.json：资料里叫"火鹰装备"，与星源系列逐字对得上）
HUOYING = ["星源合金盾", "星源星徽", "星源狩枪", "星源装甲", "月影之力"]
# 神兵装备：卡库里没有明确列表。按本批 6 件"云中君主题、唯一、3 阶装备"认定（流云之刺 / 唤鹰骨笛 召唤云中君灵体）。
SHENBING = ["流云之刺", "翼苍穹盾", "掠魂爪", "永冻霜羽", "苍穹王权", "唤鹰骨笛"]
# 普通装备牌：取 8 件基础装备（data/equip_recipes.json）
BASE_EQUIPS = ["守护者之铠", "力量腰带", "圣者法典", "破碎圣杯", "大棒", "雷鸣刃", "速击之枪", "风暴巨剑"]


# ============================================================ 辅助
def fac_of(u) -> str:
    """单位所属派系（卡库派系；没有则看"视为【X】英雄"类关键词）。无阵营 / 空 → ""。"""
    f = faction(u.name)
    if f in FACTIONS:
        return f
    for k in u.keywords:
        if k in FACTIONS:
            return k
    return ""


def once_per_battle(eng, key: str) -> bool:
    """"战斗开始时"类天赋只结算一次：react_battle_start 每个单位都会触发，用本回合标记去重（round: 前缀回合开始清掉）。"""
    k = "round:" + key
    if eng.s.flags.get(k):
        return False
    eng.s.flags[k] = True
    return True


def gain_tactic(eng, force_high: bool = False) -> None:
    """获得 1 张随机战术牌（5 种等概率，待核对权重）。演武：棋手 5 级后变高阶。"""
    high = force_high or ("演武" in eng.s.talents and eng.s.lord_level >= 5)
    name = eng.pick(TACTICS, 1)[0]
    eng.gain_card(("高阶" + name) if high else name)


def draw_tiers(eng, tiers, fac=None):
    """从若干品阶的英雄里按牌库剩余张数抽 1 张（"随机 4-5 阶"这类：品阶不单独抽，直接按剩余张数加权，待核对）。"""
    fac = FACTION_ALIAS.get(fac, fac) if fac else None
    names = [n for n, c in card_db().items() if c.get("category") == "英雄牌" and quality(n) in tiers
             and (not fac or faction(n) == fac)]
    ws = [eng.pool_left(n) for n in names]
    if not names or sum(ws) <= 0:
        return None
    return eng.rng.choices(names, ws)[0]


def draw_by_lord(eng, fac=None):
    """按棋手等级的商店品阶概率抽 1 张英雄（与征召令同口径）。"""
    for _ in range(10):
        x = eng.draw_hero(None, fac)
        if x:
            return x
    return None


def give_hero(eng, name) -> None:
    """给 1 张英雄牌进手牌并记牌库被拿走。待核对：天赋指名赠送的英雄牌（李白 / 云中君…）是否也占公共牌池（按占）。"""
    if not name:
        return
    eng.take(name)
    eng.gain_card(name)


def is_hero_card(name: str) -> bool:
    return (card_db().get(name) or {}).get("category") == "英雄牌"


class _Talent:
    """天赋基类：拿到时没有立即效果（持续效果挂在 eng.s.talents，holder = None）。"""
    def on_use(self, eng, t):
        return None


class _GiftTalent(_Talent):
    """拿到时送指定英雄牌；后半句"共战时 …"是战斗效果。"""
    gifts: tuple = ()

    def on_use(self, eng, t):
        for n in self.gifts:
            give_hero(eng, n)


# ============================================================ 三分
@card("三分领袖")
class 三分领袖(_Talent):
    """获得1张装备牌三分旗帜：造成伤害的30%平摊治疗所有【三分】英雄（仅【三分】英雄可穿戴）。
    养成：拿到时得 1 张三分旗帜。旗帜本身是战斗效果（治疗）。"""
    def on_use(self, eng, t):
        eng.gain_card("三分旗帜")


@card("鬼神降临")
class 鬼神降临(_Talent):
    """你阵亡4名【三分】英雄后，召唤吕布，其等级为已阵亡【三分】英雄等级之和的一半。
    按"同一场战斗里阵亡第 4 名三分英雄时，战斗中召唤吕布（召唤单位，不占人口）"实现，每场至多 1 次；
    吕布在阵亡者附近空格出现（eng.summon battle_only：引擎战斗结束移除）。
    等级 = 已阵亡三分英雄等级之和的一半，向下取整。
    待核对：阵亡数是否跨战斗累计（按每场清零：阵亡只发生在战斗里，战后全员复活）；吕布战后是否留在棋盘上（按不留）；
    "等级"是否含临时等级（按含）。"""
    def react_death(self, eng, holder, unit=None, **kw):
        if holder is not None or unit is None or unit.kind != "hero" or fac_of(unit) != "三分之地":
            return
        f = eng.s.flags
        f["round:鬼神降临_n"] = f.get("round:鬼神降临_n", 0) + 1
        f["round:鬼神降临_lv"] = f.get("round:鬼神降临_lv", 0.0) + unit.lv()
        if f["round:鬼神降临_n"] == 4:
            lb = eng.summon("吕布", near=unit, level=max(1, math.floor(f["round:鬼神降临_lv"] / 2)), battle_only=True)
            if lb:
                eng.note("召唤", unit="吕布", level=lb.level, src="鬼神降临(战斗中)")


@card("三分强援")
class 三分强援(_Talent):
    """获得5能量和随机3张4-5阶【三分】英雄牌。（3 张可重复，按牌库剩余加权，待核对）"""
    def on_use(self, eng, t):
        eng.s.energy += 5
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (4, 5), "三分之地"))


@card("三分增援")
class 三分增援(_Talent):
    """随机获得3张3-4阶【三分】英雄牌。"""
    def on_use(self, eng, t):
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (3, 4), "三分之地"))


# ============================================================ 多阵营 / 通用
@card("四重奏")
class 四重奏(_Talent):
    """战斗开始时，若你场上有至少3个阵营，全体英雄获得5×棋手等级的临时等级。
    待核对：无阵营是否算一个阵营（按不算）；"全体英雄"只算场上英雄（不含召唤物）。"""
    def react_battle_start(self, eng, holder, **kw):
        if holder is not None or not once_per_battle(eng, "四重奏"):
            return
        facs = {fac_of(u) for u in eng.heroes()} - {""}
        if len(facs) >= 3:
            for u in eng.heroes():
                eng.temp(u, 5 * eng.s.lord_level, src="四重奏")


@card("千里相聚")
class 千里相聚(_Talent):
    """每打出3张无阵营英雄牌时，全体英雄等级+4。
    "打出英雄牌" = 上场或合成（引擎里两者都是一次 deploy 事件；觉醒回手再打出也算一次）；被效果触发的登场不算。"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if holder is not None or unit is None or unit.kind != "hero" or faction(unit.name) != "无阵营":
            return
        if not eng.is_card_use():                 # 被别的效果触发的登场（唤醒类）不是"打出英雄牌"
            return
        n = eng.s.flags.get("千里相聚_n", 0) + 1
        eng.s.flags["千里相聚_n"] = n
        if n % 3 == 0:
            for u in eng.heroes():
                eng.level(u, 4, src="千里相聚")


@card("慕名而来")
class 慕名而来(_Talent):
    """你的英雄每触发4次登场效果，随机获得1张英雄牌。
    只数卡面有登场效果的英雄；觉醒翻倍不算两次触发。随机英雄按棋手等级的商店品阶概率抽（待核对）。"""
    def react_deploy(self, eng, holder, unit=None, **kw):
        if holder is not None or unit is None or unit.kind != "hero" or not has_keyword(unit.name, "登场"):
            return
        n = eng.s.flags.get("慕名而来_n", 0) + 1
        eng.s.flags["慕名而来_n"] = n
        if n % 4 == 0:
            give_hero(eng, draw_by_lord(eng))


@card("超级蓄能")
class 超级蓄能(_Talent):
    """回合开始时，你的整备效果英雄会提升场上整备效果英雄数x2的等级。
    在整备结算之前加（引擎：天赋 on_round_start 先于整备）。"""
    def on_round_start(self, eng, holder):
        if holder is not None:
            return
        zb = [u for u in eng.heroes() if eng.has_kw(u, "整备")]
        for u in zb:
            eng.level(u, 2 * len(zb), src="超级蓄能")


@card("天兵")
class 天兵(_Talent):
    """获得1张5阶英雄牌。回合开始时，再获得1张。"""
    def on_use(self, eng, t):
        give_hero(eng, eng.draw_hero(5))

    def on_round_start(self, eng, holder):
        if holder is None:
            give_hero(eng, eng.draw_hero(5))


@card("点兵点将")
class 点兵点将:
    """获得1张天赋牌点兵点将：获得场上英雄牌各1张。
    赠送牌（同名）：获得场上所有英雄牌各1张。
    天赋和赠送牌同名、都没有目标：第一次使用视为拿到天赋（送 1 张），之后每次使用是赠送牌的效果。
    只给真正的英雄牌（召唤单位 / 视为英雄的赠送单位不给）。待核对：是否占公共牌池（按占）。"""
    def on_use(self, eng, t):
        if not eng.s.flags.get("点兵点将_picked"):
            eng.s.flags["点兵点将_picked"] = True
            eng.gain_card("点兵点将")
            return
        for u in list(eng.heroes()):
            if is_hero_card(u.name):
                give_hero(eng, u.name)


@card("拾荒者")
class 拾荒者(_Talent):
    """恢复10棋手生命值。之后每有1位棋手被淘汰，获得2张5阶英雄兑换券。
    生命值 engine.add_hp（上限 60）；淘汰 = engine.eliminate(n) 发的 eliminated 事件。"""
    def on_use(self, eng, t):
        eng.add_hp(10, src="拾荒者")

    def react_eliminated(self, eng, holder, n=1, **kw):
        if holder is not None:
            return
        for _ in range(2 * n):
            eng.gain_card("5阶英雄兑换券")


@card("守护灵")
class 守护灵(_Talent):
    """当你在战斗中仅剩1名英雄时，随机召唤1名满装备的5阶英雄（无卡牌效果）出战，并使之获得回合数x10的等级（至少为80级）。
    战斗效果：只在战斗中出现、无卡牌效果，不影响养成。"""


@card("生生不息")
class 生生不息(_Talent):
    """4/5阶的召唤单位获得完美复生。
    战斗效果为主（复生只多一条命）；按引擎的复生计数挂上（开团时给 4/5 阶召唤单位 +1 次复生）。"""
    def react_open(self, eng, holder, unit=None, **kw):
        if holder is None and unit is not None and unit.kind == "summon" and quality(unit.name) in (4, 5):
            unit.counters["revive"] = unit.counters.get("revive", 0) + 1


@card("升格")
class 升格(_Talent):
    """战斗中入场的召唤物存活8秒后，变为其召唤者（无卡牌效果）并保留等级，每场战斗最多5次。
    战斗效果：变身只在战斗中、无卡牌效果，不影响养成。"""


@card("狂野帷幕")
class 狂野帷幕(_Talent):
    """战斗开始时，所有非牺牲单位进入隐形状态，持续10秒。
    战斗效果：隐形，不影响养成。"""


@card("暴君归来")
class 暴君归来(_Talent):
    """你在战斗过程中复生的英雄，获得200%攻速、100%暴击率并无敌，持续1.5秒。
    战斗效果：攻速 / 暴击 / 无敌，不影响养成。"""


@card("水晶穹顶")
class 水晶穹顶(_Talent):
    """战斗开始时，穹顶会将区域内英雄的护盾吸收变为自身的护盾，保护区域内的友军不受伤害。
    战斗效果：护盾，不影响养成。"""


@card("四海来援")
class 四海来援(_Talent):
    """无阵营英雄获得牺牲：召唤1名与自身品阶和等级相同的英雄（无卡牌效果）参战。
    养成：让无阵营英雄拥有"牺牲"（牺牲反应器因此多触发）。召唤的英雄只在战斗中、无卡牌效果 → 只记日志。
    关键词每回合开始重算（grant_keywords），上场 / 开团时补一次（回合中途上场的人）。"""
    def _grant(self, eng):
        for u in eng.heroes():
            if faction(u.name) == "无阵营":
                u.keywords.add("牺牲")

    def grant_keywords(self, eng, holder):
        if holder is None:
            self._grant(eng)

    def react_deploy(self, eng, holder, **kw):
        if holder is None:
            self._grant(eng)

    def react_open(self, eng, holder, **kw):
        if holder is None:
            self._grant(eng)

    def react_sacrifice(self, eng, holder, unit=None, **kw):
        if holder is None and unit is not None and unit.kind == "hero" and faction(unit.name) == "无阵营":
            eng.note("召唤", unit=unit.name + "(无卡牌效果)", level=unit.lv(), src="四海来援(战斗中)")


# ============================================================ 逐鹿 / 战术
@card("意外之财")
class 意外之财(_Talent):
    """每合成4次【逐鹿】英雄，获得1张随机1-5阶效果锦囊牌。（品阶 1~5 等概率，待核对）"""
    def react_merge(self, eng, holder, unit=None, **kw):
        if holder is not None or unit is None or fac_of(unit) != "逐鹿":
            return
        n = eng.s.flags.get("意外之财_n", 0) + 1
        eng.s.flags["意外之财_n"] = n
        if n % 4 == 0:
            eng.gain_card(f"{eng.rng.randint(1, 5)}阶效果锦囊")


@card("令行如流")
class 令行如流(_Talent):
    """获得1张战术牌。回合开始时，再获得1张（棋手等级达到5级后，必定是高阶战术）。
    战术牌 = 援护 / 招募 / 演练 / 推演 / 特训战术，随机等概率。
    待核对：5 级之前是否也有概率给高阶（"必定"暗示有，但卡面没写概率 → 按只给普通）；拿到时那 1 张是否也适用 5 级规则（按适用）。"""
    def on_use(self, eng, t):
        gain_tactic(eng, force_high=eng.s.lord_level >= 5)

    def on_round_start(self, eng, holder):
        if holder is None:
            gain_tactic(eng, force_high=eng.s.lord_level >= 5)


@card("演武")
class 演武(_Talent):
    """棋手升到5级后，你获得的战术牌都会变为高阶战术。
    获得牌时拦截（gain_redirect，引擎 gain_card 认它）：棋手 ≥5 级时获得的普通战术牌改成同名"高阶"版。
    选天赋之前手里已有的普通战术牌不转换（卡面"你获得的"按"拿到天赋之后获得"理解）。"""
    def gain_redirect(self, eng, holder, name):
        if holder is None and eng.s.lord_level >= 5 and name in TACTICS:
            return ["高阶" + name]
        return None


@card("穷兵黩武")
class 穷兵黩武(_Talent):
    """【逐鹿】英雄释放技能时，召唤1个玄雍士兵，其等级与召唤英雄相同，最多同时存在3个。
    战斗效果：士兵只在战斗中出现（拿不到装备），不影响养成。接口缺口：没有"释放技能"事件。"""


@card("复现")
class 复现(_Talent):
    """每回合首次对【逐鹿】英雄使用的效果牌会返回手牌并附加转瞬。
    养成：本回合第一次对逐鹿英雄打出效果牌后，同名牌回到手牌。
    "附加转瞬"记在手牌状态 transient=True（eng.hand_meta）；引擎不在回合结束自动销毁，推演打法要在本回合内用掉它。"""
    def react_effect_card(self, eng, holder, card="", target=None, **kw):
        if holder is not None or target is None or fac_of(target) != "逐鹿":
            return
        if eng.s.flags.get("round:复现"):
            return
        eng.s.flags["round:复现"] = True
        eng.gain_card(card, transient=True)
        eng.note("转瞬", card=card, src="复现")


@card("罡气护体")
class 罡气护体(_Talent):
    """【逐鹿】英雄在技能释放期间，每1秒获得自身最大生命值5%的护盾，护盾持续3秒。
    战斗效果：护盾，不影响养成。"""


@card("春风化雨")
class 春风化雨(_Talent):
    """回合开始时，获得1张【逐鹿】英雄牌；若上场战斗胜利，再获得1张场上【逐鹿】英雄的同名牌。
    逐鹿英雄按棋手等级的商店品阶概率抽；"场上逐鹿英雄的同名牌"在场上逐鹿英雄里随机挑 1 名（待核对：是否有固定规则）。"""
    def on_round_start(self, eng, holder):
        if holder is not None:
            return
        give_hero(eng, draw_by_lord(eng, "逐鹿"))
        if eng.s.flags.get("春风化雨_win"):
            zl = [u for u in eng.heroes() if fac_of(u) == "逐鹿" and is_hero_card(u.name)]
            if zl:
                give_hero(eng, eng.pick(zl, 1)[0].name)
        eng.s.flags["春风化雨_win"] = False

    def on_battle_end(self, eng, holder, win=False):
        if holder is None:
            eng.s.flags["春风化雨_win"] = bool(win)


@card("逐鹿强援")
class 逐鹿强援(_Talent):
    """获得5能量和随机3张4-5阶【逐鹿】英雄牌。"""
    def on_use(self, eng, t):
        eng.s.energy += 5
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (4, 5), "逐鹿"))


@card("逐鹿增援")
class 逐鹿增援(_Talent):
    """随机获得3张3-4阶【逐鹿】英雄牌。"""
    def on_use(self, eng, t):
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (3, 4), "逐鹿"))


# ============================================================ 河洛 / 古币
@card("千金换酒")
class 千金换酒(_Talent):
    """获得1张李白，李白每获得12次古币，释放技能时会永久额外释放1次。
    养成：拿到时送 1 张李白。古币计数记在李白身上（counters["千金换酒_古币"] / ["额外释放"]），额外释放是战斗效果。"""
    def on_use(self, eng, t):
        give_hero(eng, "李白")

    def react_coin(self, eng, holder, target=None, fine=False, **kw):
        if holder is not None or target is None or target.name != "李白":
            return
        c = target.counters.get("千金换酒_古币", 0) + 1
        target.counters["千金换酒_古币"] = c
        target.counters["额外释放"] = c // 12


@card("丝绸之路")
class 丝绸之路(_Talent):
    """每出售4名场上英雄获得1张古币，每出售1张古币获得1张【河洛】英雄牌。
    河洛英雄按棋手等级的商店品阶概率抽（待核对）。
    卖手牌 = engine.sell_card 发的 sell_card 事件。"""
    def react_sell(self, eng, holder, unit=None, **kw):
        if holder is not None or unit is None or unit.kind != "hero":
            return
        n = eng.s.flags.get("丝绸之路_n", 0) + 1
        eng.s.flags["丝绸之路_n"] = n
        if n % 4 == 0:
            eng.gain_card("古币")

    def react_sell_card(self, eng, holder, card="", **kw):
        if holder is None and card == "古币":
            give_hero(eng, draw_by_lord(eng, "河洛"))


@card("自我奉献")
class 自我奉献(_Talent):
    """【河洛】英雄获得牺牲：所有友方单位+1临时等级，每次触发此牺牲效果时，临时等级额外+1。
    第 k 次触发给所有友方单位（含召唤物）+k 临时等级。待核对：k 是整局累计还是每场战斗重新计（按整局累计）。"""
    def _grant(self, eng):
        for u in eng.heroes():
            if fac_of(u) == "河洛":
                u.keywords.add("牺牲")

    def grant_keywords(self, eng, holder):
        if holder is None:
            self._grant(eng)

    def react_deploy(self, eng, holder, **kw):
        if holder is None:
            self._grant(eng)

    def react_open(self, eng, holder, **kw):
        if holder is None:
            self._grant(eng)

    def react_sacrifice(self, eng, holder, unit=None, **kw):
        if holder is not None or unit is None or unit.kind != "hero" or fac_of(unit) != "河洛":
            return
        k = eng.s.flags.get("自我奉献_n", 0) + 1
        eng.s.flags["自我奉献_n"] = k
        for u in eng.board_units():
            eng.temp(u, k, src="自我奉献")


@card("一起发财")
class 一起发财(_Talent):
    """使用古币时，其他每个阵营中各1名随机英雄等级+3。
    "其他阵营" = 被用古币那名英雄以外的阵营；无阵营不算阵营（待核对）。古币机 / 李元芳等"使用古币"同样触发。"""
    def react_coin(self, eng, holder, target=None, fine=False, **kw):
        if holder is not None or target is None:
            return
        own = fac_of(target) if target.kind == "hero" else ""
        groups: dict = {}
        for u in eng.heroes():
            f = fac_of(u)
            if f and f != own:
                groups.setdefault(f, []).append(u)
        for f, us in groups.items():
            eng.level(eng.pick(us, 1)[0], 3, src="一起发财")


@card("古币机")
class 古币机:
    """召唤古币机，每对其使用4张古币牌（精致古币视为2次），依次获得：随机5张河洛英雄牌；1件贤者的庇护。之后随机获得：1张超速成长、1张普通装备牌或8能量。
    天赋（holder=None）：拿到时召唤古币机（占一个格子、不上战斗、不占人口）。
    古币机单位（同名）：被用古币时计数，满 4 次给奖励。古币卡面要求目标是【河洛】→ 给古币机挂"河洛"关键词才能被指定
    （它是 object，不在 heroes() 里，不会被"距离最近的河洛英雄"之类选中）。
    待核对：5 张河洛英雄牌的品阶（按棋手等级的商店品阶概率）；"普通装备牌"按 8 件基础装备随机；之后三种奖励等概率。"""
    def on_use(self, eng, t):
        if t is not None:
            return False
        if any(u.name == "古币机" for u in eng.s.units):
            return
        u = eng.deploy("古币机", kind="object", pop=False)
        if u:
            u.keywords.add("河洛")

    def grant_keywords(self, eng, holder):
        if holder is not None and holder.name == "古币机":
            holder.keywords.add("河洛")

    def react_coin(self, eng, holder, target=None, fine=False, **kw):
        if holder is None or target is not holder:
            return
        c = holder.counters.get("coins", 0) + (2 if fine else 1)
        while c >= 4:
            c -= 4
            stage = holder.counters.get("stage", 0) + 1
            holder.counters["stage"] = stage
            if stage == 1:
                for _ in range(5):
                    give_hero(eng, draw_by_lord(eng, "河洛"))
            elif stage == 2:
                eng.gain_card("贤者的庇护")
            else:
                r = eng.rng.randrange(3)
                if r == 0:
                    eng.gain_card("超速成长")
                elif r == 1:
                    eng.gain_card(eng.pick(BASE_EQUIPS, 1)[0])
                else:
                    eng.s.energy += 8
        holder.counters["coins"] = c


@card("升值")
class 升值(_Talent):
    """获得2张古币。战斗胜利时，古币牌提供的等级永久+1。
    每赢一场，之后每张古币多给 1 级（累加）。待核对：精致古币是否 +2（按 +1）。"""
    def on_use(self, eng, t):
        eng.gain_card("古币")
        eng.gain_card("古币")

    def coin_bonus(self, eng, holder, target, fine):
        return eng.s.flags.get("升值_bonus", 0) if holder is None else 0

    def on_battle_end(self, eng, holder, win=False):
        if holder is None and win:
            eng.s.flags["升值_bonus"] = eng.s.flags.get("升值_bonus", 0) + 1


@card("河洛强援")
class 河洛强援(_Talent):
    """获得5能量和3张随机4-5阶【河洛】英雄牌。"""
    def on_use(self, eng, t):
        eng.s.energy += 5
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (4, 5), "河洛"))


@card("河洛增援")
class 河洛增援(_Talent):
    """随机获得3张3-4阶【河洛】英雄牌。"""
    def on_use(self, eng, t):
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (3, 4), "河洛"))


# ============================================================ 大河 / 图腾
@card("望梅止渴")
class 望梅止渴:
    """获得1张望梅止渴。回合开始时，场上每有2名【大河】英雄，再获得1张。
    赠送牌（同名）：指定1名英雄临时等级+3，战斗开始时，每穿1件装备额外+1级。若战斗胜利，将永久获得此等级。
    on_use：目标为 None = 拿到天赋（送 1 张）；有目标 = 赠送牌效果。
    赠送牌的后续结算（开战按装备加、胜利转永久）挂在天赋上（赠送牌只来自这个天赋）。
    待核对："额外+1级"是临时等级（按临时，战胜后一并转永久）。"""
    def on_use(self, eng, t):
        if t is None:
            eng.gain_card("望梅止渴")
            return
        eng.temp(t, 3, src="望梅止渴")
        t.counters["望梅止渴"] = t.counters.get("望梅止渴", 0) + 1

    def on_round_start(self, eng, holder):
        if holder is not None:
            return
        n = sum(1 for u in eng.heroes() if fac_of(u) == "大河流域") // 2
        for _ in range(n):
            eng.gain_card("望梅止渴")

    def react_battle_start(self, eng, holder, **kw):
        if holder is not None or not once_per_battle(eng, "望梅止渴"):
            return
        for u in eng.s.units:
            k = u.counters.get("望梅止渴", 0)
            if k:
                extra = k * len(u.equips)
                eng.temp(u, extra, src="望梅止渴(装备)")
                u.counters["望梅止渴_lv"] = 3 * k + extra

    def on_battle_end(self, eng, holder, win=False):
        if holder is not None:
            return
        for u in eng.s.units:
            k = u.counters.pop("望梅止渴", 0)
            lv = u.counters.pop("望梅止渴_lv", 3 * k)
            if k and win:
                eng.level(u, lv, src="望梅止渴(胜利)")


@card("命运图腾")
class 命运图腾(_Talent):
    """战斗开始时，若你至少有2名【大河】英雄，随机在其周围1格召唤命运图腾：消灭击败该图腾的敌人。
    天赋（holder=None）：每场战斗开始在随机 1 名大河英雄旁召唤命运图腾（battle_only，引擎战斗结束移除）。
    图腾同名单位挂的也是这个类，所以钩子都先判 holder is None。
    养成上的意义：它是"图腾"，会被灵祭升华（图腾被摧毁）等算到。
    待核对：图腾等级（按 1 级）；与开团（虞姬"场上每有 1 个图腾"）的先后（引擎里开团在前，算不到它）。"""
    def react_battle_start(self, eng, holder, **kw):
        if holder is not None or not once_per_battle(eng, "命运图腾"):
            return
        dh = [u for u in eng.heroes() if fac_of(u) == "大河流域"]
        if len(dh) >= 2:
            eng.summon("命运图腾", near=eng.pick(dh, 1)[0], battle_only=True)


@card("神明庇佑")
class 神明庇佑(_Talent):
    """战斗开始时，图腾周围1格的己方单位提高10%生命值上限和20%攻速，可叠加。
    战斗效果：生命值 / 攻速（站位：图腾围着主C），不影响养成。"""


@card("熊魂附甲")
class 熊魂附甲(_Talent):
    """你至少穿戴2件装备的【大河】英雄阵亡获得牺牲：召唤1只与该英雄等级相同的熊灵。
    养成：让穿 ≥2 件装备的大河英雄拥有"牺牲"；牺牲时在它附近召唤熊灵（召唤单位，等级 = 该英雄当时等级含临时等级），
    熊灵只在战斗中存在（data/engine/summon_units_v1.json presence=battle，拿不到装备；eng.summon battle_only，引擎战斗结束移除）。
    待核对："等级相同"是否含临时等级（按含）。"""
    def _grant(self, eng):
        for u in eng.heroes():
            if fac_of(u) == "大河流域" and len(u.equips) >= 2:
                u.keywords.add("牺牲")

    def grant_keywords(self, eng, holder):
        if holder is None:
            self._grant(eng)

    def react_open(self, eng, holder, **kw):
        if holder is None:
            self._grant(eng)

    def react_sacrifice(self, eng, holder, unit=None, **kw):
        if holder is None and unit is not None and fac_of(unit) == "大河流域" and len(unit.equips) >= 2:
            b = eng.summon("熊灵", near=unit, level=max(1.0, unit.lv()), battle_only=True)
            if b:
                eng.note("召唤", unit="熊灵", level=b.level, src="熊魂附甲(战斗中)")


@card("虚妄转生")
class 虚妄转生(_Talent):
    """召唤1个虚妄图腾：被摧毁时，变为场上存活的等级最高或品阶最高的【大河】英雄（无卡牌效果）。
    养成：拿到时在棋盘上召唤虚妄图腾（召唤物、不占人口、算图腾）。被摧毁后的变身只在战斗中、无卡牌效果。
    待核对：图腾等级（按 1 级）。"""
    def on_use(self, eng, t):
        eng.summon("虚妄图腾")


@card("灵祭升华")
class 灵祭升华(_Talent):
    """场上的图腾被摧毁时，等级最高的【大河】英雄获得图腾一半等级的临时等级。
    "被摧毁" = 战斗中图腾阵亡（引擎对召唤物也做阵亡判定）。待核对：一半向下取整；等级含图腾临时等级。"""
    def react_death(self, eng, holder, unit=None, **kw):
        if holder is not None or unit is None or "图腾" not in unit.name:
            return
        dh = [u for u in eng.heroes() if fac_of(u) == "大河流域"]
        if dh:
            t = max(dh, key=lambda x: x.lv())
            eng.temp(t, math.floor(unit.lv() / 2), src="灵祭升华")


@card("神鹰锻匠")
class 神鹰锻匠(_Talent):
    """获得1张云中君，战斗开始时，云中君会为距离最近的1个【大河】英雄添加1件神兵装备。若云中君觉醒，神兵至多添加3件。
    按"神兵留在身上（拿不下来）；每个云中君累计最多添加 1 件，觉醒后最多 3 件"实现；目标不含云中君自己，装备满 3 件就不加。
    神兵 = 流云之刺 / 翼苍穹盾 / 掠魂爪 / 永冻霜羽 / 苍穹王权 / 唤鹰骨笛（随机、不与目标已有的重复）。
    待核对：神兵装备的确切列表；战后是否保留；"至多 3 件"是每场 / 每人 / 累计。"""
    def on_use(self, eng, t):
        give_hero(eng, "云中君")

    def react_battle_start(self, eng, holder, **kw):
        if holder is not None or not once_per_battle(eng, "神鹰锻匠"):
            return
        for y in [u for u in eng.heroes() if u.name == "云中君"]:
            cap = 3 if y.awakened else 1
            if y.counters.get("神兵", 0) >= cap:
                continue
            t = eng.nearest(y, lambda x: fac_of(x) == "大河流域")
            if not t or len(t[0].equips) >= 3:
                continue
            opts = [e for e in SHENBING if e not in t[0].equips]
            if not opts:
                continue
            e = eng.pick(opts, 1)[0]
            if eng.equip(t[0], e, bound=True, forge=False, src="神鹰锻匠"):
                y.counters["神兵"] = y.counters.get("神兵", 0) + 1


@card("大河强援")
class 大河强援(_Talent):
    """获得5能量和随机3张4-5阶【大河流域】英雄牌。"""
    def on_use(self, eng, t):
        eng.s.energy += 5
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (4, 5), "大河流域"))


@card("大河增援")
class 大河增援(_Talent):
    """随机获得3张3-4阶【大河流域】英雄牌。"""
    def on_use(self, eng, t):
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (3, 4), "大河流域"))


# ============================================================ 神兵装备（云中君）
@card("流云之刺")
class 流云之刺:
    """唯一。攻击速度+30%。每对目标进行3次普通攻击，会召唤云中君灵体对其造成4%最大生命值的物理伤害并降低其5点护甲。
    战斗效果：攻速 / 伤害 / 减甲，不影响养成。"""


@card("翼苍穹盾")
class 翼苍穹盾:
    """唯一。物理和法术防御力+10。敌方每阵亡1个单位，再各提升10点，上限100点。
    战斗效果：双抗，不影响养成。"""


@card("掠魂爪")
class 掠魂爪:
    """唯一。物理和法术吸血+50%。普通攻击额外造成目标8%当前生命值的物理伤害。
    战斗效果：吸血 / 伤害，不影响养成。"""


@card("永冻霜羽")
class 永冻霜羽:
    """唯一。法术攻击力+100 。对每个目标首次造成法术伤害时，将冻结目标2秒。
    战斗效果：法攻 / 冻结，不影响养成。"""


@card("苍穹王权")
class 苍穹王权:
    """唯一。受到致命伤害时，免疫此次伤害，持续1秒。并对全部敌人造成100+其已损失30%生命值的法术伤害。
    战斗效果：免死 / 伤害，不影响养成。"""


@card("唤鹰骨笛")
class 唤鹰骨笛:
    """战斗开始后每8秒，召唤一名等级为自身一半的云中君，迅速冲向最远处的敌人。
    战斗效果：战斗中召唤的云中君只在战斗中存在，不影响养成。"""


# ============================================================ 日落海 / 阿科米亚核心
class _CoreTalent(_Talent):
    """核心相关天赋（开战分配由引擎 core_distribute 在开团之前统一做，这里不再分配）。"""


@card("核心裂痕")
class 核心裂痕(_CoreTalent):
    """获得1张盗火。回合开始时，再获得1张（转瞬。指定1名英雄，至多偷取阿科米亚核心20级）。"""
    def on_use(self, eng, t):
        eng.gain_card("盗火")

    def on_round_start(self, eng, holder):
        if holder is None:
            eng.gain_card("盗火")


@card("盗火")
class 盗火:
    """转瞬 指定1名英雄，至多偷取阿科米亚核心20级。
    核心等级减少、目标永久等级增加（取 min(20, 核心等级)）。核心等级 eng.s.core。"""
    def on_use(self, eng, t):
        if t is None or t not in eng.s.units:
            return False
        a = min(20.0, eng.s.core)
        if a > 0:
            eng.core_add(-a, src="盗火")
            eng.level(t, a, src="盗火")


@card("能源置换")
class 能源置换(_CoreTalent):
    """回合开始时，阿科米亚核心失去20级，获得10能量。每回合最多触发1次。
    待核对：核心不足 20 级时是否触发（按不触发）。"""
    def on_round_start(self, eng, holder):
        if holder is not None or eng.s.flags.get("round:能源置换"):
            return
        if eng.s.core >= 20:
            eng.s.flags["round:能源置换"] = True
            eng.core_add(-20, src="能源置换")
            eng.s.energy += 10


@card("积少成多")
class 积少成多(_CoreTalent):
    """【日落海】英雄每提升10级，阿科米亚核心获得4等级。
    每名日落海英雄记一个基准等级（counters["积少成多_base"]，拿到天赋时 = 当时等级，之后新来的 = 第一次看到时的等级），
    永久等级每超过基准 10 级，核心 +4、基准 +10；等级被夺走时基准随之下调（再涨回来也算"提升"）。
    接口缺口：没有"等级变化"事件，只能在回合开始 / 整备 / 上场 / 合成 / 用古币 / 用效果牌 / 战斗结束时轮询（基准记在单位上，不会漏算，只会晚算）。
    待核对：临时等级（含核心自己分出去的）是否计入（按不计，否则核心→临时等级→核心无限循环）。"""
    def _check(self, eng):
        for u in eng.heroes():
            if fac_of(u) != "日落海":
                continue
            base = u.counters.get("积少成多_base")
            if base is None:
                u.counters["积少成多_base"] = u.level
                continue
            base = min(base, u.level)
            steps = int((u.level - base) // 10)
            if steps > 0:
                eng.core_add(4 * steps, src="积少成多")
                base += 10 * steps
            u.counters["积少成多_base"] = base

    def on_use(self, eng, t):
        self._check(eng)

    def on_round_start(self, eng, holder):
        if holder is None:
            self._check(eng)

    def on_battle_end(self, eng, holder, win=False):
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


@card("能量汲取")
class 能量汲取(_CoreTalent):
    """出售场上英雄时，阿科米亚核心会获得其等级。（永久等级；备战期没有临时等级）"""
    def react_sell(self, eng, holder, unit=None, **kw):
        if holder is None and unit is not None and unit.kind == "hero":
            eng.core_add(unit.level, src="能量汲取")


@card("定向传输")
class 定向传输(_CoreTalent):
    """阿科米亚核心的等级会优先分配给穿戴装备最多的【日落海】英雄，最多分配200级。
    类属性 core_direct_cap = 200：引擎 core_distribute 先给穿装备最多的那名（至多 200，不受 50% 上限），剩下的再平均分。"""
    core_direct_cap = 200


@card("双重分配")
class 双重分配(_CoreTalent):
    """阿科米亚核心每分配1等级，还将随机分配1点属性：双攻/攻速/初始法力值中的随机1个（每场战斗随机）。
    战斗效果：属性，不影响养成。"""


@card("全域共鸣")
class 全域共鸣(_Talent):
    """随机获得1件火鹰装备，且火鹰装备的【日落海】专属效果对穿戴它的任意英雄生效。
    火鹰装备 = 日落海专属装备（星源合金盾 / 星源星徽 / 星源狩枪 / 星源装甲 / 月影之力）。
    "对任意英雄生效"是穿戴合法性（推演打法判断），记 flags["全域共鸣"]=True。"""
    def on_use(self, eng, t):
        eng.s.flags["全域共鸣"] = True
        eng.gain_card(eng.pick(HUOYING, 1)[0])


@card("日落海强援")
class 日落海强援(_Talent):
    """获得5能量和随机3张4-5阶【日落海】英雄牌。"""
    def on_use(self, eng, t):
        eng.s.energy += 5
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (4, 5), "日落海"))


@card("日落海增援")
class 日落海增援(_Talent):
    """随机获得3张3-4阶【日落海】英雄牌。"""
    def on_use(self, eng, t):
        for _ in range(3):
            give_hero(eng, draw_tiers(eng, (3, 4), "日落海"))


# ============================================================ 夺取
@card("摄人心魄")
class 摄人心魄(_Talent):
    """你的夺取效果英雄击败敌方英雄时，将永久夺取对方3级。（每击杀 1 名 +3 永久等级；对手掉级不在我方局面里）"""
    def react_kill(self, eng, holder, unit=None, n=1, **kw):
        if holder is None and unit is not None and eng.has_kw(unit, "夺取"):
            eng.level(unit, 3 * n, src="摄人心魄")


@card("控魂")
class 控魂(_Talent):
    """夺取效果英雄击败敌方英雄时，将其灵魂复活（无卡牌效果），满法力值作战，持续5秒。
    战斗效果：战斗中复活的灵魂无卡牌效果，不影响养成。"""


@card("双生相")
class 双生相(_Talent):
    """战斗开始时，第一排闪现英雄不再闪现，获得30%最大生命值上限以及10点双抗；后三排闪现英雄额外获得20%暴击率和40%暴击效果。
    战斗效果：生命 / 双抗 / 暴击，不影响养成。"""


# ============================================================ 羁绊送牌（共战）
@card("身外身")
class 身外身(_Talent):
    """获得1张孙悟空，孙悟空释放技能时，呼出2个猴毛化身进行1次攻击。战斗胜利时，化身永久+1。
    养成：送 1 张孙悟空。化身只在战斗中出现；"化身永久+1"记在孙悟空身上（counters["化身"]），是战斗强度。
    待核对："+1"是化身等级还是化身数量（只计数，不进养成）。"""
    def on_use(self, eng, t):
        give_hero(eng, "孙悟空")

    def on_battle_end(self, eng, holder, win=False):
        if holder is not None or not win:
            return
        for u in eng.heroes():
            if u.name == "孙悟空":
                u.counters["化身"] = u.counters.get("化身", 0) + 1


@card("兄妹")
class 兄妹(_GiftTalent):
    """获得1张铠和1张露娜。共战时，铠首次释放技能，露娜可以无限释放技能，持续7秒。露娜每次释放技能，铠提升40%攻速。
    养成：送铠、露娜各 1 张。共战效果是战斗效果。"""
    gifts = ("铠", "露娜")


@card("兽觉")
class 兽觉(_GiftTalent):
    """获得1张裴擒虎和1张云中君。共战时，每当敌人被击败，二人获得兽觉，普攻额外造成100%物理伤害，持续5秒。
    养成：送裴擒虎、云中君各 1 张。兽觉是战斗效果。"""
    gifts = ("裴擒虎", "云中君")


@card("巨像")
class 巨像(_GiftTalent):
    """获得1张米莱狄和1张蒙恬。二人在场时，召唤与米莱狄等级相同的蔚蓝神兽。二人等级之和达到100时，召唤与蒙恬等级相同的猩红神兽。
    养成：送米莱狄、蒙恬各 1 张；二人都在场 → 棋盘上有蔚蓝神兽（等级跟米莱狄同步）；二人永久等级和达到 100 → 再加猩红神兽
    （等级跟蒙恬同步，达到后和低于 100 也保留）；有一人不在场 → 两只神兽移除。
    在拿到天赋 / 回合开始 / 上场 / 合成 / 开团时同步。待核对：神兽等级是只在召唤时取一次还是一直跟随（按一直跟随）。"""
    gifts = ("米莱狄", "蒙恬")

    def _sync(self, eng):
        m = next((u for u in eng.heroes() if u.name == "米莱狄"), None)
        t = next((u for u in eng.heroes() if u.name == "蒙恬"), None)
        blue = eng.s.unit("蔚蓝神兽")
        red = eng.s.unit("猩红神兽")
        if not (m and t):
            for b in (blue, red):
                if b is not None:
                    eng.remove(b, src="巨像(二人不在场)")
            eng.s.flags["巨像_red"] = False
            return
        if blue is None:
            eng.summon("蔚蓝神兽", near=m, level=m.level)
        else:
            blue.level = m.level
        if m.level + t.level >= 100:
            eng.s.flags["巨像_red"] = True
        if eng.s.flags.get("巨像_red"):
            if red is None:
                eng.summon("猩红神兽", near=t, level=t.level)
            else:
                red.level = t.level

    def on_use(self, eng, t):
        super().on_use(eng, t)
        self._sync(eng)

    def on_round_start(self, eng, holder):
        if holder is None:
            self._sync(eng)

    def react_deploy(self, eng, holder, **kw):
        if holder is None:
            self._sync(eng)

    def react_merge(self, eng, holder, **kw):
        if holder is None:
            self._sync(eng)

    def react_open(self, eng, holder, **kw):
        if holder is None and once_per_battle(eng, "巨像"):
            self._sync(eng)


@card("猩红神兽")
class 猩红神兽:
    """战斗中，我方召唤物入场时使之获得猩红神兽之力。
    战斗效果：只作用于战斗中入场的召唤物，不影响养成（神兽本身由巨像召唤，见巨像）。"""


@card("蔚蓝神兽")
class 蔚蓝神兽:
    """战斗中我方单位入场时，蔚蓝神兽永久增加5点生命值。
    战斗效果：永久生命值是战斗属性，不涉及等级，不影响养成。"""


@card("舞力全开")
class 舞力全开(_Talent):
    """貂蝉、婉儿共战时二人共享法力值（不会变更），技能变为融合技，可对目标造成伤害及负面效果。
    战斗效果：融合技，不影响养成（不送牌）。"""


@card("起舞")
class 起舞(_GiftTalent):
    """获得1张貂蝉和1张上官婉儿。共战时，参与击败敌方英雄后，二人均+10点法术攻击力并+1级。
    养成：送貂蝉、上官婉儿各 1 张；二人都在场时，其中任一人每击杀 1 名，二人各 +1 永久等级（法攻是战斗属性）。
    待核对："参与击败"含助攻（引擎只记击杀者，助攻算不到）。"""
    gifts = ("貂蝉", "上官婉儿")

    def react_kill(self, eng, holder, unit=None, n=1, **kw):
        if holder is not None or unit is None or unit.name not in ("貂蝉", "上官婉儿"):
            return
        pair = [u for u in eng.s.on_board() if u.name in ("貂蝉", "上官婉儿")]
        if len({u.name for u in pair}) < 2:
            return
        for u in pair:
            eng.level(u, n, src="起舞")


@card("箭雨")
class 箭雨(_GiftTalent):
    """获得1张虞姬和1张公孙离。共战时，每合计攻击10次，将往敌方人群最密集处释放箭雨，各造成3次自身等级x2的法术伤害。
    养成：送虞姬、公孙离各 1 张。箭雨是战斗效果。"""
    gifts = ("虞姬", "公孙离")


@card("双生突袭")
class 双生突袭(_Talent):
    """马可波罗、百里守约共战，释放技能时创造分身朝敌人释放技能。
    战斗效果：分身释放技能，不影响养成（不送牌）。"""


@card("毒奶")
class 毒奶(_GiftTalent):
    """获得1张蔡文姬和1张扁鹊。共战时，扁鹊固定对蔡文姬释放技能，蔡文姬获得50%移速，持续在敌方附近游走并造成伤害（详见技能面板）。
    养成：送蔡文姬、扁鹊各 1 张。共战效果是战斗效果。"""
    gifts = ("蔡文姬", "扁鹊")


@card("逐星")
class 逐星(_GiftTalent):
    """获得1张李白和1张曜。共战时，李白释放技能后，曜同步模仿李白，3次后曜将二人的技能融会贯通，领悟更强大的招式（详见技能面板）。
    养成：送李白、曜各 1 张。融合招式是战斗效果。"""
    gifts = ("李白", "曜")


@card("宝盒")
class 宝盒(_GiftTalent):
    """获得1张露娜和1张孙悟空。共战时二者任意一方受到致死伤害时会暂时离场，持续6秒。在此期间另一方如未阵亡，则返回战场并恢复全部生命值及法力值。
    养成：送露娜、孙悟空各 1 张。暂离 / 复原是战斗效果。"""
    gifts = ("露娜", "孙悟空")


@card("旋刃")
class 旋刃(_GiftTalent):
    """获得1张李元芳和1张公孙离。共战时二人共享法力值（不会变更），技能变为融合技：旋刃（详见技能面板）。旋刃每造成2次伤害，李元芳等级永久+1。
    养成：送李元芳、公孙离各 1 张。
    待核对 / 接口缺口："旋刃每造成 2 次伤害李元芳永久 +1"是养成效果，但要知道一场战斗旋刃放几次、每次打几段 ——
    引擎没有技能释放事件，卡面也没给次数，不猜数值，暂不结算。"""
    gifts = ("李元芳", "公孙离")


# ============================================================ 天赋送的特殊装备
@card("根源之力")
class 根源之力:
    """获得根源之力：对敌方血量最少的单位造成法术伤害。
    装备（同名）：唯一。每4秒对生命值最低的敌方单位造成600%法术攻击力的法术伤害。
    天赋：拿到时得 1 张装备牌根源之力（on_use 目标为 None）。装备本身是战斗效果（伤害），不影响养成。"""
    def on_use(self, eng, t):
        if t is not None:
            return False
        eng.gain_card("根源之力")


@card("强敌克星")
class 强敌克星:
    """获得强敌克星：造成真实伤害，每普攻4次眩晕敌人。
    装备（同名）：唯一。造成伤害的15%转化为真实伤害。对目标每普通攻击4次，会眩晕目标1秒。
    天赋：拿到时得 1 张装备牌强敌克星（on_use 目标为 None）。装备本身是战斗效果（真伤 / 眩晕），不影响养成。"""
    def on_use(self, eng, t):
        if t is not None:
            return False
        eng.gain_card("强敌克星")
