"""第 0 批卡牌实现（data/engine/sim_batches.json batches[0]，79 张）：
阿离枫叶舞强化 / 秘技、1~5 阶英雄、阿科米亚核心件、图腾 / 召唤物、逐鹿战术牌、相关装备。

只实现养成（等级 / 临时等级 / 觉醒 / 装备 / 获得牌 / 能量 / 人口 / 召唤 / 移除 / 触发效果），战斗数值只在 docstring 写一句。
2026-10-14：原来在本模块补的接口缺口（核心等级 / 开战分配、被触发的登场、召唤事件、关键词判定、放技能次数、
棋手生命、夺取、枫叶舞强化挑选）都并进了 engine.py，这里直接调 eng.core_add / eng.trigger / eng.is_card_use /
eng.summon / has_effect / eng.casts / eng.s.hp / eng.steal / eng.pick_lord_boost。
"""
from __future__ import annotations

import math

from ..engine import BATTLE_SECONDS, LORD_HP_MAX, card, card_db, category, faction, has_effect, quality


# ============================================================ 本批共用的辅助函数
def m_of(u) -> int:
    """反应器 / 战斗结束这类引擎不会自动翻倍的钩子：持有者觉醒 → 效果 ×2。"""
    return 2 if (u is not None and u.awakened) else 1


def on_field(eng, u) -> bool:
    return u is not None and u in eng.s.units


def is_totem(u) -> bool:
    return "图腾" in u.name


def totems(eng) -> list:
    return [x for x in eng.s.units if is_totem(x)]


# ---------------------------------------------------------------- 阿离：枫叶舞
def maple_unit(eng):
    return eng.s.unit(eng.s.maple) if eng.s.maple else None


class _LordCard:
    """拿到 / 选中后进 lord_cards（被动生效）。"""
    def on_use(self, eng, t):
        eng.add_lord_card(self.names[0])


# ============================================================ 阿离：枫叶舞强化 / 特性 / 秘技
@card("枫舞扶摇")
class 枫舞扶摇(_LordCard):
    """每回合首次转移后获得5临时等级，枫叶舞英雄本局对战每击败1名英雄，此等级额外+1。
    实现：放进 lord_cards；每场战斗开始时给枫叶舞英雄 5 + 本局累计击败数 的临时等级。
    待核对：转移（枫叶舞保命转移）是战斗事件，引擎不模拟 —— 按"每场战斗都会转移一次、发生在开战时"实现；
    "本局对战"按整局累计击败数理解（不是本场）。保命 / 转移本身是战斗效果。"""
    def react_kill(self, eng, holder, unit=None, n=0):
        if unit is not None and unit.name == eng.s.maple:
            eng.s.flags["枫舞扶摇:kills"] = eng.s.flags.get("枫舞扶摇:kills", 0) + n

    def react_battle_start(self, eng, holder, unit=None):
        if holder is not None or eng.s.flags.get("round:枫舞扶摇"):
            return
        u = maple_unit(eng)
        if u is None:
            return
        eng.s.flags["round:枫舞扶摇"] = True
        eng.temp(u, 5 + eng.s.flags.get("枫舞扶摇:kills", 0), src="枫舞扶摇")


@card("翩跹馈赠")
class 翩跹馈赠(_LordCard):
    """转移后，枫叶舞英雄累积普攻15次时也会获得1次转移次数。
    战斗效果：多一次保命转移，不影响养成（枫叶舞特性，拿到后进 lord_cards）。"""


@card("枫之洪流")
class 枫之洪流(_LordCard):
    """枫叶舞英雄在首次转移后的3秒内，可以无限释放技能。
    战斗效果：转移后 3 秒无限放技能，不影响养成。待核对：放技能次数变多会让孙膑 / 刘禅这类"释放技能时"的卡多触发，引擎没有技能事件，未计。"""


@card("落英汲取")
class 落英汲取(_LordCard):
    """枫叶舞英雄转移后，对1格内每个敌人造成自身最大生命值20%的法术伤害，并恢复自身（3%×敌人数量）最大生命值。
    战斗效果：伤害与回血，不影响养成。"""


@card("翩跹舞步·余韵")
class 翩跹舞步余韵:
    """为枫叶舞挑选1张强化牌。
    实现：打出时进 lord_cards，并从强化牌池选 1 张。"""
    def on_use(self, eng, t):
        eng.add_lord_card("翩跹舞步·余韵")
        eng.pick_lord_boost()


@card("翩跹舞步·进化")
class 翩跹舞步进化:
    """为枫叶舞挑选1张强化牌。枫叶舞英雄获得本局击败敌方英雄数量×4点物理和法术攻击力。
    实现：打出时进 lord_cards + 选 1 张强化牌。攻击力加成是战斗效果，不影响养成。"""
    def on_use(self, eng, t):
        eng.add_lord_card("翩跹舞步·进化")
        eng.pick_lord_boost()


@card("翩跹舞步·流动")
class 翩跹舞步流动:
    """为枫叶舞挑选1张强化牌。枫叶舞英雄在转移后每击败3名敌方英雄，获得额外的1次转移次数。
    实现：打出时进 lord_cards + 选 1 张强化牌。额外转移次数是战斗效果，不影响养成。"""
    def on_use(self, eng, t):
        eng.add_lord_card("翩跹舞步·流动")
        eng.pick_lord_boost()


@card("翩跹舞步·蓄势")
class 翩跹舞步蓄势:
    """为枫叶舞挑选1张强化牌。枫叶舞英雄击败敌方英雄时永久等级+1。
    实现：打出时进 lord_cards + 选 1 张强化牌；之后枫叶舞英雄每击败 1 名敌方英雄 等级 +1。
    待核对：引擎的"击杀"不区分敌方英雄 / 召唤物，按全部算。"""
    def on_use(self, eng, t):
        eng.add_lord_card("翩跹舞步·蓄势")
        eng.pick_lord_boost()

    def react_kill(self, eng, holder, unit=None, n=0):
        if holder is None and unit is not None and n and unit.name == eng.s.maple and unit in eng.s.units:
            eng.level(unit, n, src="翩跹舞步·蓄势")


MAPLE_TRAITS: list = []   # 卡面重读 10-18：翩跹馈赠 / 枫之洪流 / 落英汲取 是「即刻起舞」的 3 个选项（官网 previewCards），不是惊鸿游龙的特性；4 项特性未知 → 不再白送
JIKE_OPTIONS = ["落英汲取", "枫之洪流", "翩跹馈赠"]   # 即刻起舞 previewCards（data/web/棋手牌.raw.json）
XIANWU = ["翩跹舞步·新生", "翩跹舞步·蓄势", "翩跹舞步·流动", "翩跹舞步·进化", "翩跹舞步·余韵"]


@card("惊鸿游龙")
class 惊鸿游龙:
    """枫叶舞指定的英雄击败3/7/11/15名敌方英雄，获得1张秘技牌翩跹舞步并为枫叶舞解锁1项特性（共4项）。
    实现：阿离秘技（天赋），拿到即进 lord_cards；枫叶舞英雄累计击败数（整局，换绑不清零）每跨过 3/7/11/15 一档：
    获得 1 张翩跹舞步（5 种里交给推演打法选）+ 解锁 1 项特性（进 lord_cards）。
    待核对：①"获得1张秘技牌翩跹舞步"是 5 种里随机还是可选，按可选实现；②4 项特性具体是哪 4 张未知 —— 原来按 翩跹馈赠 / 枫之洪流 / 落英汲取 实现，
    10-18 卡面重读确认这三张是即刻起舞的选项，已撤掉（MAPLE_TRAITS 为空 = 特性不建模）；③击败数换绑不清（10-19 规则确认）。"""
    STEPS = (3, 10, 21, 36)   # 逐档另计（3 / 再 7 / 再 11 / 再 15）—— 2026-10-14 按 200 局"获得翩跹舞步"的回合（中位 R3/R7/R11/R14）
                              # 和翩跹舞步·蓄势的回放残差（中期每回合 2~3 杀）两个探针一起定的

    def on_use(self, eng, t):
        eng.add_lord_card("惊鸿游龙")

    def react_kill(self, eng, holder, unit=None, n=0):
        if holder is not None or unit is None or not n or unit.name != eng.s.maple:
            return
        if "惊鸿游龙" not in eng.s.lord_cards and "惊鸿游龙" not in eng.s.talents:
            return
        f = eng.s.flags
        before = f.get("惊鸿游龙:kills", 0)
        after = before + n
        f["惊鸿游龙:kills"] = after
        for step in self.STEPS:
            if before < step <= after:
                card_name = eng.choose("翩跹舞步", list(XIANWU))
                if card_name:
                    eng.gain_card(card_name)
                left = [x for x in MAPLE_TRAITS if x not in eng.s.lord_cards]
                if left:
                    eng.add_lord_card(eng.choose("maple_trait", left))


@card("即刻起舞")
class 即刻起舞:
    """从3个枫叶舞强化中选择1项。
    3 个选项固定是 落英汲取 / 枫之洪流 / 翩跹馈赠（卡面重读 10-18：官网 previewCards），不是 15 张强化牌池随机 3 张。"""
    def on_use(self, eng, t):
        opts = [x for x in JIKE_OPTIONS if x not in eng.s.lord_cards]   # 依据：即刻起舞 previewCards
        sel = eng.choose("lord_boost", opts) if opts else None
        if sel:
            eng.add_lord_card(sel)
            eng.note("枫叶舞强化", card=sel)


# ============================================================ 1 阶英雄
@card("曜")
class 曜:
    """你的英雄合成提升的等级+1。
    实现：每当己方英雄合成，被合成的英雄额外 +1 级（曜觉醒 +2）。战斗技能不实现。"""
    def react_merge(self, eng, holder, unit=None):
        if on_field(eng, holder) and unit is not None and unit.kind == "hero":
            eng.level(unit, 1 * m_of(holder), src="曜")


@card("孙膑")
class 孙膑:
    """你的英雄每释放3次技能，该英雄等级永久+1。
    实现：每场战斗结束，场上每名英雄按估计的放技能次数（见 casts()）累计，每满 3 次 +1 级，余数跨回合保留；孙膑觉醒每满 3 次 +2。
    待核对：放技能次数是按面板估的（引擎没有技能事件）。"""
    def on_battle_end(self, eng, holder, win=False):
        if not on_field(eng, holder):
            return
        m = m_of(holder)
        for h in eng.heroes():
            acc = h.counters.get("孙膑:casts", 0) + eng.casts(h)
            k = acc // 3
            h.counters["孙膑:casts"] = acc - 3 * k
            if k:
                eng.level(h, k * m, src="孙膑")


@card("狂铁")
class 狂铁:
    """整备：阿科米亚核心的等级+2。（开战分配由引擎 core_distribute 统一做）"""
    def on_prepare(self, eng, u):
        eng.core_add(2, src="狂铁")


@card("孙尚香")
class 孙尚香:
    """登场：获得1次免费刷新。"""
    def on_deploy(self, eng, u):
        eng.s.free_refresh += 1


@card("双剑·雄")
class 双剑雄:
    """攻击速度+30%。若穿戴者为干将莫邪，其法力值上限降低至10。
    干将莫邪出生自带、拿不下来（pilot 干将莫邪 放进 bound）。战斗效果：攻速 / 法力上限，不影响养成。"""


@card("牛魔")
class 牛魔:
    """凯旋：获得1能量。"""
    def on_win(self, eng, u):
        eng.s.energy += 1


# ============================================================ 2 阶英雄
@card("亚连")
class 亚连:
    """整备：你穿戴装备最多的1名英雄等级+2，且阿科米亚核心等级+1。
    实现：件数并列时随机 1 名（含亚连自己）。"""
    def on_prepare(self, eng, u):
        hs = eng.heroes()
        if hs:
            mx = max(len(x.equips) for x in hs)
            t = eng.pick([x for x in hs if len(x.equips) == mx], 1)[0]
            eng.level(t, 2, src="亚连")
        eng.core_add(1, src="亚连")


@card("米莱狄")
class 米莱狄:
    """召唤单位获得 3临时等级。整备：该临时等级及阿科米亚核心等级额外+3。
    实现：米莱狄身上记一个"召唤单位临时等级"，初始 3；每次整备 +3 并让核心 +3（永久累计）。
    每场战斗开始时，场上己方召唤单位（召唤物 / 图腾）获得这么多临时等级。觉醒：初始值、每次整备的增量都 ×2（自己处理）。
    待核对：①"召唤单位"按场上所有己方召唤单位（不只米莱狄技能的机械仆从）；②战斗中才出现的召唤物（玄雍士兵等）没吃到；
    ③整备的累加是否跨回合保留 —— 按保留（"额外+3"读作永久成长）。机械仆从本身是战斗效果。"""
    awaken_custom = True

    def on_prepare(self, eng, u):
        m = m_of(u)
        u.counters["米莱狄:extra"] = u.counters.get("米莱狄:extra", 0) + 3 * m
        eng.core_add(3 * m, src="米莱狄")

    def on_battle_start(self, eng, u):
        n = 3 * m_of(u) + u.counters.get("米莱狄:extra", 0)
        for x in eng.s.on_board():
            if x.kind == "summon":
                eng.temp(x, n, src="米莱狄")


@card("项羽")
class 项羽:
    """牺牲：所有图腾等级永久+3且本回合你的召唤物免疫控制。
    实现：牺牲时场上所有图腾 +3 级。免疫控制是战斗效果。"""
    def on_sacrifice(self, eng, u):
        for x in totems(eng):
            eng.level(x, 3, src="项羽")


@card("鬼谷子")
class 鬼谷子:
    """在场时，召唤先知图腾。
    实现：上场时（场上还没有先知图腾）召唤 1 个先知图腾。技能"玄微子包裹友方临时等级+5，持续8秒"是战斗中的短时加成，不实现。
    待核对：鬼谷子离场后先知图腾是否消失（pilot 少司缘同样没处理）。"""
    def on_deploy(self, eng, u):
        if not any(x.name == "先知图腾" for x in eng.s.units):
            eng.summon("先知图腾", near=u)


@card("先知图腾")
class 先知图腾:
    """开团：距离最近的2名己方英雄获得装备玄微之种。
    实现：开团时给最近 2 名己方英雄（10-18 新卡面不限阵营）穿上玄微之种（eng.equip：满 3 件 / 已穿着的跳过），并当场给"牺牲"关键词；战斗结束收回。
    待核对：①玄微之种是否只在本场战斗存在、占不占 3 件装备的格子 —— 按"本场战斗、占格"实现；②觉醒（图腾不会觉醒）不涉及。"""
    def on_open(self, eng, u):
        given = u.counters.setdefault("先知图腾:given", [])
        for t in eng.nearest(u, k=2):        # 卡面 10-18 重爬：【大河流域】英雄 → 己方英雄（不限阵营）；满 3 件的拿不到（10-19 规则确认）
            if "玄微之种" in t.equips or not eng.equip(t, "玄微之种", src="先知图腾"):
                continue
            t.keywords.add("牺牲")
            given.append(t.uid)

    def on_battle_end(self, eng, holder, win=False):
        if holder is None:
            return
        for uid in holder.counters.pop("先知图腾:given", []):
            t = next((x for x in eng.s.units if x.uid == uid), None)
            if t is not None and "玄微之种" in t.equips:
                t.equips.remove("玄微之种")


@card("玄微之种")
class 玄微之种:
    """穿戴者获得牺牲：生命值百分比最低的己方英雄恢复200（+20%最大生命值）点生命值。
    实现：穿戴者获得"牺牲"关键词（会触发守约 / 李信 / 大司命这类牺牲反应器）。回血本身是战斗效果。"""
    def grant_keywords(self, eng, holder):
        if on_field(eng, holder):
            holder.keywords.add("牺牲")


@card("梦奇")
class 梦奇:
    """退场：获得1张 1阶英雄兑换券。败阵：提升英雄兑换券的品阶。
    实现：梦奇身上记兑换券品阶（初始 1）；每次败阵 +1（最高 5）；被出售时获得 1 张该品阶的英雄兑换券。
    觉醒（卡面）：退场得 2 张，败阵那句没变（仍 +1 阶）→ 自己处理觉醒，不让引擎把败阵执行两次。
    待核对：①"提升英雄兑换券的品阶"按"梦奇退场时给的那张券"理解，而不是手里已有的兑换券；②品阶上限按 5。"""
    awaken_custom = True     # 卡面：觉醒只把退场改成 2 张，败阵仍 +1 阶

    def on_lose(self, eng, u):
        u.counters["梦奇:tier"] = min(5, u.counters.get("梦奇:tier", 1) + 1)

    def on_sell(self, eng, u):
        for _ in range(2 if u.awakened else 1):           # 卡面觉醒：退场获得 2 张兑换券
            eng.gain_card(f"{u.counters.get('梦奇:tier', 1)}阶英雄兑换券")


@card("曹操")
class 曹操:
    """整备：随机触发1名【三分】英雄的登场效果。
    实现：从场上有"登场："效果的三分英雄里随机 1 名，触发其登场（不算使用英雄牌）。
    觉醒：随机 2 名（卡面）→ 不同的 2 名，自己处理（原来引擎执行两次各抽 1 名，可能抽到同一人）。"""
    awaken_custom = True     # 卡面觉醒：随机触发 2 名

    def on_prepare(self, eng, u):
        cands = [x for x in eng.heroes() if eng.is_faction(x, "三分之地") and has_effect(x.name, "登场")]
        for t in eng.pick(cands, 2 if u.awakened else 1):   # 卡面觉醒：2 名
            eng.trigger(t, "deploy")


@card("马可波罗")
class 马可波罗:
    """登场：自身及阿科米亚核心等级+2。"""
    def on_deploy(self, eng, u):
        eng.level(u, 2, src="马可波罗")
        eng.core_add(2, src="马可波罗")


@card("赵云")
class 赵云:
    """登场：自身及随机2名其他英雄等级+1。（觉醒：同样 2 名各 +2 —— 卡面；原来引擎执行两次会抽两轮、可能分给 4 个人）"""
    awaken_custom = True     # 卡面觉醒：+2（还是随机 2 名）

    def on_deploy(self, eng, u):
        m = m_of(u)
        eng.level(u, m, src="赵云")
        for t in eng.pick([x for x in eng.heroes() if x is not u], 2):
            eng.level(t, m, src="赵云")


@card("大乔")
class 大乔:
    """你的英雄每触发2次登场效果，大乔的法力值上限永久-5（觉醒：每 1 次，卡面）。
    战斗效果：降法力上限，不影响养成。"""


@card("公孙离")
class 公孙离:
    """夺取 开团：场上每有1名已觉醒英雄，攻击速度+15%。
    实现：夺取（上场 / 合成时抢周围 1 格随机 1 名己方英雄最多 10 级）。攻速是战斗效果。
    觉醒只把攻速 15% → 30%，"夺取"仍是 1 次（卡面重读 10-18）→ 自己处理觉醒，不让引擎夺两次。"""
    awaken_custom = True     # 卡面：觉醒文本里夺取只出现 1 次

    def on_deploy(self, eng, u):
        eng.steal(u, src="公孙离(夺取)")


@card("墨子")
class 墨子:
    """友方单位护盾被击破时，该单位生命值上限永久+30。
    战斗效果：生命上限成长，不影响养成。"""


@card("百里玄策")
class 百里玄策:
    """闪现 整备：你的闪现英雄上回合每击败过1名敌方英雄，随机1名闪现英雄等级+1（击败次数： 0）。
    实现：玄策在场时记下本场闪现英雄的击败数；下回合整备时每 1 次击败 → 随机 1 名闪现英雄 +1（逐次独立随机，可重复）。
    觉醒：每次击败 +2（自己处理，避免整备执行两次时击败数被消耗掉）。待核对：击杀不区分敌方英雄 / 召唤物。"""
    awaken_custom = True

    def react_kill(self, eng, holder, unit=None, n=0):
        if on_field(eng, holder) and unit is not None and eng.has_kw(unit, "闪现"):
            holder.counters["玄策:kills"] = holder.counters.get("玄策:kills", 0) + n

    def on_prepare(self, eng, u):
        k = u.counters.pop("玄策:kills", 0)
        flash = [x for x in eng.heroes() if eng.has_kw(x, "闪现")]
        if not flash:
            return
        for _ in range(int(k)):
            eng.level(eng.pick(flash, 1)[0], m_of(u), src="百里玄策")


@card("廉颇")
class 廉颇:
    """廉颇释放技能时，你的其他英雄获得 5点护盾值。每次使用战术牌会提升该护盾值。
    战斗效果：护盾，不影响养成。"""


@card("援护战术")
class 援护战术:
    """转瞬 全体【逐鹿】英雄本回合获得50（+25×战术牌打出次数，当前：75）点护盾。
    战斗效果：护盾，不影响养成。作为效果牌打出时仍会触发镜 / 白起 / 芈月 等"使用效果牌"的反应器。"""
    def on_use(self, eng, t):
        return None


@card("招募战术")
class 招募战术:
    """转瞬 随机获得1张【逐鹿】英雄牌。
    实现：按棋手等级的商店品阶概率 + 牌库剩余抽 1 张逐鹿英雄（与征召令同一口径）。待核对：品阶分布是否跟商店概率。"""
    def on_use(self, eng, t):
        for _ in range(10):
            x = eng.draw_hero(None, "逐鹿")
            if x:
                eng.take(x)
                eng.gain_card(x)
                return


@card("演练战术")
class 演练战术:
    """转瞬 所有英雄等级+1。"""
    def on_use(self, eng, t):
        for x in eng.heroes():
            eng.level(x, 1, src="演练战术")


@card("蒙恬")
class 蒙恬:
    """牺牲：召唤2名与自身等级相同的玄雍士兵。
    实现：牺牲时在身边召唤 2 名同等级玄雍士兵（battle_only：只在本场战斗存在，引擎战斗结束移除）；召唤发 summon 事件（往生图腾会吸收）。
    待核对：场上没空格时召唤不出来（引擎召唤要占格子）。技能里召唤的士兵是战斗效果，不实现。"""
    def on_sacrifice(self, eng, u):
        for _ in range(2):
            eng.summon("玄雍士兵", near=u, level=u.level, battle_only=True)


@card("玄雍士兵")
class 玄雍士兵:
    """“令则行，禁则止，无有所怠！”
    蒙恬牺牲召唤的战斗单位，没有养成效果（只普攻）。"""


@card("蔡文姬")
class 蔡文姬:
    """每当你使用【三分】英雄牌时，等级最低的其他3名【三分】英雄等级+1。
    实现：使用三分英雄牌（上场 / 合成 / 觉醒回手再打出；被曹操等触发的登场不算）时，除蔡文姬外等级最低的 3 名三分英雄 +1（蔡文姬觉醒 +2）。
    待核对："其他"按"蔡文姬以外"理解（刚打出的那名也可能被选中）。"""
    def react_deploy(self, eng, holder, unit=None):
        if not on_field(eng, holder) or unit is None or unit.kind != "hero" or not eng.is_card_use():
            return
        if not eng.is_faction(unit, "三分之地"):
            return
        for t in eng.lowest([x for x in eng.heroes() if x is not holder and eng.is_faction(x, "三分之地")], 3):
            eng.level(t, m_of(holder), src="蔡文姬")


# ============================================================ 3 阶英雄
@card("云中君")
class 云中君:
    """闪现 开团：你的英雄每穿戴1件装备，该英雄等级永久+1。
    实现：开团时每名英雄按当前装备件数 +N 级（觉醒由引擎执行两次）。待核对：开团时先知图腾 / 瑶临时给的装备是否算 —— 按开团结算顺序自然算。"""
    def on_open(self, eng, u):
        for h in eng.heroes():
            if h.equips:
                eng.level(h, len(h.equips), src="云中君")


@card("姬小满")
class 姬小满:
    """每当你出售4名场上英雄后，获得1次临时刷新。
    实现：姬小满在场时数出售的场上英雄（含她自己被卖那一次），每满 4 名 → free_refresh +1（觉醒：每满 2 名 +1，卡面）。
    待核对：①"临时刷新"按免费刷新次数记（是否本回合过期未处理）；②计数在姬小满不在场时是否保留 —— 记在她身上，离场即丢。"""
    def react_sell(self, eng, holder, unit=None):
        if not on_field(eng, holder) or unit is None or unit.kind != "hero":
            return
        c = holder.counters.get("姬小满:sold", 0) + 1
        need = 2 if holder.awakened else 4                 # 卡面觉醒：每出售 2 名得 1 次（不是每 4 名得 2 次）
        if c >= need:
            c -= need
            eng.s.free_refresh += 1
        holder.counters["姬小满:sold"] = c


@card("苍")
class 苍:
    """你的召唤单位阵亡时，会对周围敌人造成 30点物理伤害（觉醒 60，卡面 10-18 重爬）。每次召唤还会额外提升该伤害。
    战斗效果：伤害，不影响养成。"""


@card("白起")
class 白起:
    """每回合首次使用效果牌时，获得1张援护战术。
    实现：白起在场时，每回合第一次使用效果牌 → 获得 1 张援护战术（觉醒：1 张高阶援护战术，卡面）。援护战术本身也是效果牌，但已不是本回合首次。"""
    def react_effect_card(self, eng, holder, card="", target=None):
        if not on_field(eng, holder):
            return
        key = f"round:白起:{holder.uid}"
        if eng.s.flags.get(key):
            return
        eng.s.flags[key] = True
        eng.gain_card("高阶援护战术" if holder.awakened else "援护战术")   # 卡面觉醒：1 张高阶，不是 2 张普通


@card("阿轲")
class 阿轲:
    """闪现 登场：自身及随机1名英雄等级+1，棋手生命值每降低10点，重复1次。
    实现：重复次数 = 1 + (60 − 棋手生命值) // 10；每次自身 +1、随机 1 名其他英雄 +1（每次独立随机）。
    棋手生命值读 eng.s.hp（满血 60）。待核对："随机1名英雄"按不含阿轲自己。"""
    def on_deploy(self, eng, u):
        reps = 1 + max(0, LORD_HP_MAX - eng.s.hp) // 10
        for _ in range(reps):
            eng.level(u, 1, src="阿轲")
            o = eng.pick([x for x in eng.heroes() if x is not u], 1)
            if o:
                eng.level(o[0], 1, src="阿轲")


@card("马超")
class 马超:
    """己方施加控制效果时，马超获得15点法力值及5%暴击率。
    战斗效果：法力 / 暴击，不影响养成。"""


@card("裴擒虎")
class 裴擒虎:
    """闪现 整备：随机获得1名其他英雄等级数一半的临时等级，不可叠加。
    实现：整备时随机 1 名其他英雄，获得其真实等级一半（向下取整）的临时等级；同一回合再触发只保留较大的一次（不可叠加）。
    觉醒：一半 ×2（自己处理，避免两次随机被"不可叠加"吃掉）。待核对：①取整方式；②"等级数"按真实等级（不含临时等级）。"""
    awaken_custom = True

    def on_prepare(self, eng, u):
        o = eng.pick([x for x in eng.heroes() if x is not u], 1)
        if not o:
            return
        val = math.floor(o[0].level / 2) * m_of(u)
        rnd = eng.s.round
        old = u.counters.get("裴擒虎:temp", 0) if u.counters.get("裴擒虎:round") == rnd else 0
        if val > old:
            eng.temp(u, val - old, src="裴擒虎")
            u.counters["裴擒虎:temp"] = val
            u.counters["裴擒虎:round"] = rnd


@card("老夫子")
class 老夫子:
    """每回合首次合成时，获得1张招募战术。
    实现：老夫子在场时，每回合第一次有己方英雄合成 → 获得 1 张招募战术（觉醒：1 张高阶招募战术，卡面）。"""
    def react_merge(self, eng, holder, unit=None):
        if not on_field(eng, holder):
            return
        key = f"round:老夫子:{holder.uid}"
        if eng.s.flags.get(key):
            return
        eng.s.flags[key] = True
        eng.gain_card("高阶招募战术" if holder.awakened else "招募战术")   # 卡面觉醒：1 张高阶，不是 2 张普通


@card("刘禅")
class 刘禅:
    """你的英雄释放技能时，该英雄获得2临时等级。
    实现：开战时给每名英雄 2 × 本场估计放技能次数（casts()）的临时等级（觉醒由引擎执行两次）。
    待核对：临时等级其实是战斗中逐次获得的，这里开战一次性给足（影响司空震 / 明世隐 这类按临时等级结算的卡）。"""
    def on_battle_start(self, eng, u):
        for h in eng.heroes():
            c = eng.casts(h)
            if c:
                eng.temp(h, 2 * c, src="刘禅")


@card("猪八戒")
class 猪八戒:
    """夺取 每当3阶及以下英雄合成时，与其同阶的英雄等级+1。
    实现：①夺取（上场 / 合成时抢周围 1 格随机 1 名己方英雄最多 10 级）；②猪八戒在场时，品阶 ≤3 的英雄合成 → 场上同品阶英雄（含被合成的那名）各 +1（觉醒 +2）。
    待核对："与其同阶的英雄"是否包含被合成的本人。"""
    awaken_custom = True     # 卡面：觉醒只把 +1 改 +2，夺取仍是 1 次（react_merge 已按 m_of 翻倍）

    def on_deploy(self, eng, u):
        eng.steal(u, src="猪八戒(夺取)")

    def react_merge(self, eng, holder, unit=None):
        if not on_field(eng, holder) or unit is None:
            return
        q = quality(unit.name)
        if not q or q > 3:
            return
        for h in eng.heroes():
            if quality(h.name) == q:
                eng.level(h, m_of(holder), src="猪八戒")


@card("狄仁杰")
class 狄仁杰:
    """对狄仁杰使用古币时，其攻击速度永久+5%。
    战斗效果：攻速成长，不影响养成（古币本身的 +2 级由引擎 coin 结算）。"""


@card("张良")
class 张良:
    """每当你的单位触发开团效果时，场上随机3名己方英雄等级永久+1。
    实现：张良在场时，己方单位（本体或装备有"开团："效果）每开团一次 → 随机 3 名不同英雄 +1；
    张良觉醒：所有英雄 +1（卡面，不是随机 3 名 +2）。觉醒单位开团算 2 次（execs，10-19 规则确认）。"""
    def react_open(self, eng, holder, unit=None, execs=1):
        if not on_field(eng, holder) or unit is None or not eng.unit_has_effect(unit, "开团"):
            return
        for _ in range(execs):                              # 10-19 确认：觉醒英雄执行两次 = 计数 2 次
            for t in (eng.heroes() if holder.awakened else eng.pick(eng.heroes(), 3)):   # 卡面觉醒：所有英雄 +1
                eng.level(t, 1, src="张良")


@card("亚瑟")
class 亚瑟:
    """每当你使用【日落海】英雄牌时，阿科米亚核心等级+3。
    实现：使用日落海英雄牌（上场 / 合成 / 觉醒回手再打出，含亚瑟自己）→ 核心 +3（亚瑟觉醒 +6）。被触发的登场不算。"""
    def react_deploy(self, eng, holder, unit=None):
        if on_field(eng, holder) and unit is not None and unit.kind == "hero" and eng.is_card_use() \
                and eng.is_faction(unit, "日落海"):
            eng.core_add(3 * m_of(holder), src="亚瑟")


@card("貂蝉")
class 貂蝉:
    """登场：【三分】英雄等级+1。
    实现：场上所有三分英雄（含貂蝉自己）+1。"""
    def on_deploy(self, eng, u):
        for h in eng.heroes():
            if eng.is_faction(h, "三分之地"):
                eng.level(h, 1, src="貂蝉")


@card("安琪拉")
class 安琪拉:
    """整备：自身及阿科米亚核心等级+2。"""
    def on_prepare(self, eng, u):
        eng.level(u, 2, src="安琪拉")
        eng.core_add(2, src="安琪拉")


@card("瑶")
class 瑶:
    """开团：自身的首件普通装备分享给随机3名友方英雄（持续15秒）。
    战斗效果：装备分享只持续 15 秒，不影响养成。待核对：分享出去的装备若带"开团："效果，是否会让张良 / 韩信 / 云中君多触发 —— 未实现。"""


@card("镜")
class 镜:
    """闪现 每使用2张效果牌，等级+1。
    实现：镜在场时数己方使用的效果牌，每满 2 张镜 +1（觉醒 +2），余数保留在镜身上。"""
    def react_effect_card(self, eng, holder, card="", target=None):
        if not on_field(eng, holder):
            return
        c = holder.counters.get("镜:cards", 0) + 1
        if c >= 2:
            c -= 2
            eng.level(holder, m_of(holder), src="镜")
        holder.counters["镜:cards"] = c


@card("朵莉亚")
class 朵莉亚:
    """每当你的英雄触发整备效果时，随机3名英雄及阿科米亚核心等级+1。
    实现：朵莉亚在场时，己方英雄（本体或装备有"整备："效果）每触发一次整备 → 随机 3 名不同英雄 +1、核心 +1；
    朵莉亚觉醒：己方所有英雄 +1、核心 +1（卡面，不是随机 3 名 +2、核心 +2）。觉醒英雄的整备算 2 次（execs，10-19 规则确认）；
    被雅典娜 / 露娜随机触发的整备也算（10-19 规则确认）。"""
    def react_prepare(self, eng, holder, unit=None, execs=1):
        if not on_field(eng, holder) or unit is None or unit.kind != "hero" or not eng.unit_has_effect(unit, "整备"):
            return
        for _ in range(execs):                              # 10-19 确认：觉醒英雄执行两次 = 计数 2 次
            for t in (eng.heroes() if holder.awakened else eng.pick(eng.heroes(), 3)):   # 卡面觉醒：己方所有英雄 +1
                eng.level(t, 1, src="朵莉亚")
            eng.core_add(1, src="朵莉亚")                   # 卡面觉醒：核心仍 +1


# ============================================================ 4 阶英雄
@card("莱西奥")
class 莱西奥:
    """每当你出售2名场上英雄后，阿科米亚核心等级+1。
    实现：莱西奥在场时数出售的场上英雄（含他自己被卖），每满 2 名核心 +1（觉醒：每卖 1 名 +1，卡面）。"""
    def react_sell(self, eng, holder, unit=None):
        if not on_field(eng, holder) or unit is None or unit.kind != "hero":
            return
        c = holder.counters.get("莱西奥:sold", 0) + 1
        need = 1 if holder.awakened else 2                 # 卡面觉醒："每当你出售场上英雄后 +1"
        if c >= need:
            c -= need
            eng.core_add(1, src="莱西奥")
        holder.counters["莱西奥:sold"] = c


@card("上官婉儿")
class 上官婉儿:
    """整备：随机对3名不同【河洛】英雄使用1张古币。
    实现：场上河洛英雄（含自己）里随机 3 名不同的，各吃 1 张古币（经引擎 coin，武则天 / 李白 等加成照算）。
    觉醒：随机 6 名不同河洛各 1 张（卡面）—— 不是两轮各 3 名（会重复给同一人）。
    整备自动用掉的古币算"使用古币 / 效果牌"（10-19 规则确认）→ 发 effect_card。"""
    awaken_custom = True     # 卡面觉醒：6 名不同【河洛】英雄

    def on_prepare(self, eng, u):
        from .pilot import auto_coin
        for t in eng.pick([x for x in eng.heroes() if eng.is_faction(x, "河洛")], 6 if u.awakened else 3):
            auto_coin(eng, t, src="上官婉儿")


@card("艾琳")
class 艾琳:
    """开团：艾琳获得自身等级×1%的攻击速度。
    战斗效果：攻速，不影响养成。"""


@card("扁鹊")
class 扁鹊:
    """友方单位护盾被击破时，对周围敌人造成该英雄最大生命值10%的法术伤害（冷却1秒）。
    战斗效果：伤害，不影响养成。"""


@card("嬴政")
class 嬴政:
    """开团：法术攻击力+ 7。每次使用战术牌会提升该加成。
    战斗效果：法术攻击力，不影响养成。"""


@card("李信")
class 李信:
    """每当你拥有牺牲效果的英雄阵亡时，李信临时等级+7。
    实现：李信在场时，每发生一次牺牲（拥有牺牲的英雄阵亡，引擎发 sacrifice 事件）→ 李信临时等级 +7（觉醒 +14）。
    待核对：引擎 has_keyword 会因为李信卡面 keywords 列表里有"牺牲"而把李信本人当成拥有牺牲（接口缺口）。"""
    def react_sacrifice(self, eng, holder, unit=None):
        if on_field(eng, holder) and unit is not None:
            eng.temp(holder, 7 * m_of(holder), src="李信")


@card("李白")
class 李白:
    """闪现 古币牌会额外提供4临时等级。若目标为李白，该临时等级会额外+3。
    实现：李白在场时，每张古币（含精致古币）结算后目标再得 4 临时等级，目标是李白本人则 7（李白觉醒 ×2）。"""
    def react_coin(self, eng, holder, target=None, fine=False):
        if not on_field(eng, holder) or not on_field(eng, target):
            return
        n = 4 + (3 if target is holder else 0)
        eng.temp(target, n * m_of(holder), src="李白")


@card("雅典娜")
class 雅典娜:
    """开团：随机触发2名【日落海】英雄的整备效果。
    实现：从场上有"整备："效果的日落海英雄里随机 2 名不同的，触发其整备。护盾 / 免控是战斗效果。
    觉醒：随机 4 名不同的（卡面）→ 自己处理（原来引擎执行两次各抽 2 名，可能重复）。"""
    awaken_custom = True     # 卡面觉醒：随机触发 4 名

    def on_open(self, eng, u):
        cands = [x for x in eng.heroes() if eng.is_faction(x, "日落海") and eng.unit_has_effect(x, "整备")]
        for t in eng.pick(cands, 4 if u.awakened else 2):   # 卡面觉醒：4 名
            eng.trigger(t, "prepare")


@card("甄姬")
class 甄姬:
    """登场：获得1张其他的登场英雄牌。
    实现：从卡面有"登场："效果的英雄牌（甄姬除外）里抽：品阶按棋手等级的商店概率、同品阶按牌库剩余（10-19 规则确认：送英雄牌的品阶跟棋手等级 / 商店池走）。
    觉醒：从 3 张中选 1 张（卡面）—— 不是随机拿 2 张。"""
    awaken_custom = True     # 卡面觉醒：3 选 1

    def on_deploy(self, eng, u):
        from .batch_01 import _gain_heroes, _pick_heroes
        pred = lambda n: n != "甄姬" and has_effect(n, "登场")   # noqa: E731
        if u.awakened:
            _pick_heroes(eng, 3, 1, "hero", pred=pred)        # 卡面觉醒：从 3 张中选择 1 张
        else:
            _gain_heroes(eng, 1, pred)                        # 10-19 确认：品阶跟棋手等级 / 商店池


@card("小乔")
class 小乔:
    """每当你的英雄触发登场效果时，随机3名英雄等级+1。
    实现：小乔在场时，己方英雄（卡面有"登场："效果的）每登场一次（使用英雄牌或被曹操等触发都算，10-19 规则确认）→ 随机 3 名不同英雄 +1；
    小乔觉醒：所有英雄 +1（卡面，不是随机 3 名 +2）。觉醒英雄的登场算 2 次（execs，10-19 规则确认）。"""
    def react_deploy(self, eng, holder, unit=None, execs=1):
        if not on_field(eng, holder) or unit is None or unit.kind != "hero" or not has_effect(unit.name, "登场"):
            return
        for _ in range(execs):                              # 10-19 确认：觉醒英雄执行两次 = 计数 2 次
            for t in (eng.heroes() if holder.awakened else eng.pick(eng.heroes(), 3)):   # 卡面觉醒：所有英雄 +1
                eng.level(t, 1, src="小乔")


@card("大司命")
class 大司命:
    """在场时，召唤往生图腾。每当你触发牺牲效果时：往生图腾等级永久+2并获得50点法力值。
    实现：上场时（场上没有往生图腾）召唤 1 个；大司命在场时每次牺牲 → 往生图腾 +2（觉醒 +4）。法力值是战斗效果。
    待核对：①大司命离场后往生图腾是否消失；②"触发牺牲效果"按每次 sacrifice 事件算（含没有牺牲效果文本、只是被钟馗 / 玄微之种给了牺牲的人）。"""
    def on_deploy(self, eng, u):
        if not any(x.name == "往生图腾" for x in eng.s.units):
            eng.summon("往生图腾", near=u)

    def react_sacrifice(self, eng, holder, unit=None):
        if not on_field(eng, holder):
            return
        for t in eng.s.units:
            if t.name == "往生图腾":
                eng.level(t, 2 * m_of(holder), src="大司命")


@card("往生图腾", "往生图腾（山鬼形态）")
class 往生图腾:
    """大司命灵体以外的1-3阶己方召唤物上场时，吸收该单位使往生图腾等级永久+1并获得15点法力值。
    实现：引擎 eng.summon 召唤时发 summon 事件；1~3 阶己方召唤物上场 → 被移除、往生图腾 +1。山鬼形态是战斗中的变身（同一效果）。
    待核对：按卡面字面，1~3 阶的图腾（先知图腾 2 阶、因缘图腾）也会被吸收。法力值是战斗效果。"""
    def react_summon(self, eng, holder, unit=None):
        if not on_field(eng, holder) or unit is None or unit is holder or unit.kind != "summon":
            return
        if "灵体" in unit.name or unit.name.startswith("往生图腾"):
            return
        q = quality(unit.name)
        if not (1 <= q <= 3):
            return
        eng.remove(unit, src="往生图腾吸收")
        eng.level(holder, 1, src="往生图腾")


@card("庄周")
class 庄周:
    """登场：自身等级+2并获得1张招募战术。（觉醒：自身 +4 并获得 1 张高阶招募战术 —— 卡面，不是执行两次给 2 张普通）"""
    awaken_custom = True     # 卡面觉醒：+4 + 1 张高阶招募战术

    def on_deploy(self, eng, u):
        eng.level(u, 4 if u.awakened else 2, src="庄周")
        eng.gain_card("高阶招募战术" if u.awakened else "招募战术")


@card("刘邦")
class 刘邦:
    """开团：摧毁你所有的图腾，每摧毁1个，刘邦等级永久+5。
    2026-10-14 确认：每回合战斗开始时摧毁，战斗结束又回来 → 图腾不从场上移除，但**这场战斗里图腾不在**：
    因缘图腾这场不计普攻、敖隐这场拿不到图腾加成（flags["round:图腾已摧毁"]）。
    虞姬 / 东皇太一同样在开团时数图腾，结算先后卡面没写，按同时结算（照常数到）。"""
    def on_open(self, eng, u):
        n = len(totems(eng))
        if n:
            eng.level(u, 5 * n, src="刘邦")
            eng.s.flags["round:图腾已摧毁"] = True


@card("诸葛亮")
class 诸葛亮:
    """你的其他英雄释放技能时，诸葛亮获得15点法力值以及4临时等级。
    实现：开战时诸葛亮获得 4 × 其他英雄本场估计放技能次数之和（casts()）的临时等级（觉醒由引擎执行两次）。法力值是战斗效果。
    待核对：临时等级其实是战斗中逐次获得的，这里开战一次性给足。"""
    def on_battle_start(self, eng, u):
        n = sum(eng.casts(h) for h in eng.heroes() if h is not u)
        if n:
            eng.temp(u, 4 * n, src="诸葛亮")


@card("韩信")
class 韩信:
    """闪现 每当你的英雄触发整备或开团效果时，韩信等级永久+2。
    实现：韩信在场时，己方英雄（本体或装备有"整备："/"开团："效果）每触发一次整备或开团 → 韩信 +2（觉醒 +4）。
    觉醒单位的整备 / 开团算 2 次（execs）、被露娜 / 雅典娜随机触发的整备也算（10-19 规则确认）。"""
    def react_prepare(self, eng, holder, unit=None, execs=1):
        if on_field(eng, holder) and unit is not None and unit.kind == "hero" and eng.unit_has_effect(unit, "整备"):
            eng.level(holder, 2 * m_of(holder) * execs, src="韩信(整备)")   # 10-19 确认：觉醒执行两次 = 计数 2 次

    def react_open(self, eng, holder, unit=None, execs=1):
        if on_field(eng, holder) and unit is not None and unit.kind == "hero" and eng.unit_has_effect(unit, "开团"):
            eng.level(holder, 2 * m_of(holder) * execs, src="韩信(开团)")   # 10-19 确认：觉醒执行两次 = 计数 2 次


@card("蒙犽")
class 蒙犽:
    """登场：获得1张特训战术。（觉醒：1 张高阶特训战术 —— 卡面，不是 2 张普通。技能 12 枚导弹 / L100 20 枚是战斗效果）"""
    awaken_custom = True     # 卡面觉醒：1 张高阶特训战术

    def on_deploy(self, eng, u):
        eng.gain_card("高阶特训战术" if u.awakened else "特训战术")


# ============================================================ 5 阶英雄 / 召唤物
@card("东皇太一")
class 东皇太一:
    """图腾不会被优先攻击。开团：获得每个图腾等级数15%的临时等级。
    实现：开团时获得 场上所有图腾（含临时等级）等级之和 × 15% 的临时等级（觉醒由引擎执行两次）。图腾不被优先攻击是战斗效果。
    待核对：取整（这里不取整）、图腾等级是否含临时等级。"""
    def on_open(self, eng, u):
        tot = sum(t.lv() for t in totems(eng))
        if tot > 0:
            eng.temp(u, tot * 0.15, src="东皇太一")


@card("敖隐")
class 敖隐:
    """战斗中，每个图腾每4秒为敖隐提供16%攻击速度和4临时等级。
    实现：开战时一次性给 图腾数 × floor(BATTLE_SECONDS / 4) × 4 的临时等级（引擎常量 20 秒；觉醒由引擎执行两次）。攻速是战斗效果。
    待核对：战斗时长、图腾阵亡后不再提供未计。"""
    def on_battle_start(self, eng, u):
        n = 0 if eng.s.flags.get("round:图腾已摧毁") else len(totems(eng))    # 刘邦开团摧毁了图腾：这场没有图腾
        if n:
            eng.temp(u, n * int(BATTLE_SECONDS // 4) * 4, src="敖隐")


@card("机关傀儡")
class 机关傀儡:
    """获得等同于你场上【逐鹿】英雄或召唤单位中最高等级一半的临时等级。
    实现：开战时取场上逐鹿英雄 / 逐鹿召唤单位（不含自己）的最高真实等级，获得一半（向下取整）的临时等级。
    待核对：①"召唤单位"是否限逐鹿 —— 按"【逐鹿】英雄或【逐鹿】召唤单位"读；②取整、是否含临时等级。攻击是战斗效果。"""
    def on_battle_start(self, eng, u):
        cands = [x for x in eng.s.on_board() if x is not u and x.kind in ("hero", "summon") and eng.is_faction(x, "逐鹿")]
        if cands:
            mx = max(x.level for x in cands)
            eng.temp(u, math.floor(mx / 2), src="机关傀儡")


@card("花木兰")
class 花木兰:
    """本回合每使用5张古币牌，获得1次复生（总复生次数： 1）。 牺牲：临时等级+10。
    实现：①花木兰在场时数本回合的古币（含精致古币，不论目标、不论谁打出、整备自动用掉的也算），每满 5 张 → 花木兰复生 +1（觉醒不变；复生由引擎在阵亡判定时消耗）；
    ②牺牲：临时等级 +10（觉醒由引擎执行两次）。
    待核对：卡面"（总复生次数： 1）"像是自带 1 次复生的显示 —— 没有实现自带复生；古币计数是否只算对花木兰用的。"""
    def react_coin(self, eng, holder, target=None, fine=False):
        if not on_field(eng, holder):
            return
        rnd = eng.s.round
        if holder.counters.get("花木兰:round") != rnd:
            holder.counters["花木兰:round"] = rnd
            holder.counters["花木兰:coins"] = 0
        c = holder.counters["花木兰:coins"] + 1
        if c >= 5:
            c -= 5
            holder.counters["revive"] = holder.counters.get("revive", 0) + 1   # 卡面：觉醒复生那句没变（每 5 张 1 次），只有牺牲 10→20
        holder.counters["花木兰:coins"] = c

    def on_sacrifice(self, eng, u):
        eng.temp(u, 10, src="花木兰")


@card("司空震")
class 司空震:
    """你的其他英雄获得临时等级时，司空震获得一半效果（不超过自身真实等级的1倍）。
    实现：引擎 temp() 发 temp 事件 —— 其他英雄（kind == hero，不含召唤物、不含自己）每获得一次 n>0 的临时等级，
    司空震当场获得 n/2 的临时等级；本回合由此获得的合计不超过司空震真实等级。上场前别人已有的临时等级自然不算。
    觉醒 = 词条数字翻倍：一半 → 全部，1 倍 → 2 倍。
    待核对：①"1倍"上限是单次还是合计 —— 按本回合合计；②觉醒后上限是否也翻倍 —— 按翻倍。"""
    def react_temp(self, eng, holder, unit=None, n=0, src=""):
        if not on_field(eng, holder) or unit is None or unit is holder or unit.kind != "hero" or n <= 0:
            return
        m = m_of(holder)
        key = f"round:司空震:{holder.uid}:given"
        room = holder.level * m - eng.s.flags.get(key, 0.0)
        gain = min(n / 2 * m, room)
        if gain > 0:
            eng.s.flags[key] = eng.s.flags.get(key, 0.0) + gain
            eng.temp(holder, gain, src="司空震")


@card("姜子牙")
class 姜子牙:
    """你每次使用不在场的阵营以及无阵营的英雄牌时，姜子牙和随机3名其他不同阵营的英雄等级+1。
    实现：姜子牙在场时使用英雄牌（被触发的登场不算），若该牌无阵营，或场上没有其他同阵营英雄（合成时视为在场）→
    姜子牙 +1，并从其他英雄里按阵营分组、随机 3 个不同阵营各挑 1 名 +1（姜子牙觉醒 +2）。
    待核对：①"不同阵营"是 3 人彼此不同阵营（按这个实现；无阵营不算阵营、不参与，10-19 规则确认）还是与姜子牙不同；②姜子牙自己上场时是否触发（按触发：他无阵营）；
    ③觉醒回手再打出（engine.merge 里第二次 deploy）按新出牌、看场上其他人。"""
    def react_merge(self, eng, holder, unit=None):
        if on_field(eng, holder) and unit is not None:
            eng.s.flags["姜子牙:merging"] = unit.uid

    def react_deploy(self, eng, holder, unit=None):
        if not on_field(eng, holder) or unit is None or unit.kind != "hero" or not eng.is_card_use():
            return
        merging = eng.s.flags.pop("姜子牙:merging", None) == unit.uid
        fac = faction(unit.name)
        none_fac = fac in ("", "无阵营")
        if not none_fac:
            if merging:
                return
            if any(x is not unit and eng.is_faction(x, fac) for x in eng.heroes()):
                return
        m = m_of(holder)
        eng.level(holder, m, src="姜子牙")
        groups: dict = {}
        for x in eng.heroes():
            if x is holder:
                continue
            f = faction(x.name)
            if f in ("", "无阵营", "通用"):                  # 10-19 规则确认：无阵营不算一种阵营
                continue
            groups.setdefault(f, []).append(x)
        for g in eng.pick(list(groups), 3):
            eng.level(eng.pick(groups[g], 1)[0], m, src="姜子牙")
