"""试写的 ~35 张代表性卡（定接口用）：古币类、河洛牺牲、搬等级、加等级、锦囊 / 兑换券 / 征召令、阿离强化、熊灵、图腾。"""
from __future__ import annotations

import math

from agent.engine import board as B
from ..engine import card, category, faction, quality, has_keyword


# ============================================================ 古币
@card("古币")
class 古币:
    """转瞬 指定1名【河洛】英雄等级+2。（从手牌打出 = eng.coin(active=True)）"""
    def on_use(self, eng, t):
        if t is None or not eng.is_faction(t, "河洛"):
            return False
        eng.coin(t, 1, src="古币", active=True)


@card("精致古币")
class 精致古币:
    """转瞬 指定1名【河洛】英雄等级+4。（古币机"精致古币视为 2 次"）"""
    def on_use(self, eng, t):
        if t is None or not eng.is_faction(t, "河洛"):
            return False
        eng.coin(t, 1, src="精致古币", fine=True, active=True)


def auto_coin(eng, t, src: str, fine: bool = False) -> None:
    """英雄效果自动对 t 使用 1 张古币 / 精致古币（李元芳 / 程咬金 / 上官婉儿 的整备）。
    10-19 规则确认：整备自动用掉的古币算"使用古币 / 效果牌" → 除了 coin 事件，再发 effect_card（芈月 / 镜 / 白起 照算）。"""
    eng.coin(t, 1, src=src, fine=fine)
    eng.emit("effect_card", card="精致古币" if fine else "古币", target=t)


@card("李元芳")
class 李元芳:
    """整备：对距离最近的【河洛】英雄使用1张古币。（觉醒：1 张精致古币 —— 卡面，不是 2 张古币）
    场上没有别的河洛英雄时打给自己（2026-10-14 回放实测：独自在场 15 次里 12 次自己 +2）。"""
    awaken_custom = True     # 卡面觉醒：1 张精致古币（古币牌次数是 1 不是 2）

    def on_prepare(self, eng, u):
        t = eng.nearest(u, lambda x: eng.is_faction(x, "河洛")) or [u]
        if t:
            auto_coin(eng, t[0], src="李元芳", fine=u.awakened)   # 卡面觉醒：精致古币；10-19 确认：自动用掉的古币算使用效果牌


@card("程咬金")
class 程咬金:
    """整备：获得1张古币，对自身使用1张古币。（觉醒：精致古币 —— 卡面，不是 2 张古币）"""
    awaken_custom = True     # 卡面觉醒：获得并对自身使用 1 张精致古币

    def on_prepare(self, eng, u):
        auto_coin(eng, u, src="程咬金", fine=u.awakened)   # 卡面觉醒：精致古币；10-19 确认：自动用掉的古币算使用效果牌


@card("武则天")
class 武则天:
    """古币牌提升的等级+1。 登场：获得1张精致古币。（觉醒：古币牌提升 +2、登场 2 张精致古币 —— 卡面）"""
    def coin_bonus(self, eng, holder, target, fine):
        if holder is None or holder not in eng.s.units:
            return 0
        return 2 if holder.awakened else 1                 # 卡面觉醒：古币牌提升的等级 +2（原来恒 +1）

    def on_deploy(self, eng, u):
        eng.gain_card("精致古币")                            # 觉醒 2 张 = 引擎执行两次


@card("盾山")
class 盾山:
    """牺牲：获得1张古币。（觉醒：1 张精致古币 —— 卡面，不是 2 张古币）"""
    awaken_custom = True     # 卡面觉醒：1 张精致古币

    def on_sacrifice(self, eng, u):
        eng.gain_card("精致古币" if u.awakened else "古币")


# ============================================================ 河洛牺牲 / 临时转永久
@card("苏烈")
class 苏烈:
    """牺牲：苏烈的等级永久+2。"""
    def on_sacrifice(self, eng, u):
        eng.level(u, 2, src="苏烈")


@card("铠")
class 铠:
    """复生 牺牲：你的【河洛】英雄获得5临时等级。（复生：战斗中多一条命，由引擎在阵亡判定时消耗）
    觉醒（卡面）：完美复生、牺牲 10 临时等级。"完美复生"和复生的区别卡面 / 关键词表都没写 → 按 1 次复生算（原来引擎执行两次 = 2 次复生）。
    复生英雄第一次倒下就触发牺牲（10-19 规则确认，引擎 battle 已如此）。"""
    awaken_custom = True     # 卡面觉醒：完美复生（1 个），不是两次复生

    def on_open(self, eng, u):
        u.counters["revive"] = u.counters.get("revive", 0) + 1
        if u.awakened:
            u.counters["perfect_revive"] = True            # 卡面觉醒：完美复生（具体差别待确认，只做标记）

    def on_sacrifice(self, eng, u):
        for h in eng.heroes():
            if eng.is_faction(h, "河洛"):
                eng.temp(h, 10 if u.awakened else 5, src="铠")   # 卡面觉醒：10 临时等级


@card("钟馗")
class 钟馗:
    """自身及左右1格的英雄获得牺牲：击败者等级临时降低10%（每场战斗限1次）。
    养成上的作用 = 让自己和左右 1 格的人拥有"牺牲"（守约 / 李信 / 铠 等牺牲反应器因此触发更多）。击败者降级是战斗效果。"""
    def grant_keywords(self, eng, holder):
        if holder is None or holder not in eng.s.units or not holder.cell:
            return
        holder.keywords.add("牺牲")
        r = B.row_of(holder.cell)
        for x in eng.heroes():
            if x is not holder and x.cell and B.row_of(x.cell) == r and eng.dist(holder, x) <= 1:
                x.keywords.add("牺牲")


@card("明世隐")
class 明世隐:
    """距离最近的英雄，临时等级中的 2级可转化为永久等级。 登场：可转化的等级+2。
    待核对：转化在战斗结束时结算（临时等级清零前）；"登场 +2"每次登场（含合成）累加。"""
    def on_deploy(self, eng, u):
        u.counters["conv"] = u.counters.get("conv", 2) + 2

    def on_battle_end(self, eng, holder, win=False):
        if holder is None or holder not in eng.s.units:
            return
        t = eng.nearest(holder)
        if not t:
            return
        cap = holder.counters.get("conv", 2) * (2 if holder.awakened else 1)
        a = min(max(0.0, t[0].temp), cap)
        if a > 0:
            eng.level(t[0], a, src="明世隐(临时转永久)")


@card("太乙真人")
class 太乙真人:
    """开团：使距离最近的1名英雄获得复生。（复生只影响"死几次"，进而影响牺牲 / 阵亡触发）
    觉醒（卡面）：最近 1 名获得"完美复生" —— 不是两次复生；完美复生的差别未定义 → 按 1 次复生算。"""
    awaken_custom = True     # 卡面觉醒：完美复生（1 个），原来引擎执行两次 = 复生 +2

    def on_open(self, eng, u):
        t = eng.nearest(u)
        if t:
            t[0].counters["revive"] = t[0].counters.get("revive", 0) + 1
            if u.awakened:
                t[0].counters["perfect_revive"] = True     # 卡面觉醒：完美复生（具体差别待确认，只做标记）


@card("百里守约", "沈梦溪")
class 只改法力的主C:
    """百里守约：每当你拥有牺牲效果的英雄阵亡时，本回合法力值上限-20。沈梦溪：每对沈梦溪使用1张效果牌，本回合法力值上限-15。
    战斗效果（多放技能），不影响养成。"""


# ============================================================ 大河 / 图腾 / 熊灵
@card("虞姬")
class 虞姬:
    """开团：场上每有1个图腾，自身及随机2名不同【大河流域】英雄等级+1。
    觉醒（卡面）：自身及随机 4 名 —— 自身仍 +1/图腾。原来引擎执行两次 = 自身 +2/图腾、两轮各 2 名可能重复 → 自己处理。"""
    awaken_custom = True     # 卡面觉醒：随机 4 名，自身不翻倍

    def on_open(self, eng, u):
        n = sum(1 for x in eng.s.units if "图腾" in x.name)
        if not n:
            return
        eng.level(u, n, src="虞姬")
        k = 4 if u.awakened else 2                          # 卡面觉醒：随机 4 名不同【大河流域】英雄
        for t in eng.pick([x for x in eng.heroes() if x is not u and eng.is_faction(x, "大河流域")], k):
            eng.level(t, n, src="虞姬")


@card("少司缘")
class 少司缘:
    """在场时，召唤因缘图腾。"""
    def on_deploy(self, eng, u):
        if not any(x.name == "因缘图腾" for x in eng.s.units):
            eng.summon("因缘图腾", near=u)


@card("因缘图腾")
class 因缘图腾:
    """你的单位每合计进行25次普攻，所有己方英雄等级永久+1。
    普攻次数 = Σ 每个单位 eng.attacks(单位)（攻速 × BATTLE_SECONDS 20 秒；攻速取卡库面板）。待核对：攻速加成 / 阵亡时间未计。"""
    def on_battle_end(self, eng, holder, win=False):
        if holder is None or holder not in eng.s.units or eng.s.flags.get("round:图腾已摧毁"):
            return                                    # 刘邦开团把图腾摧毁了：这场战斗图腾不在，不计普攻（2026-10-14）
        att = sum(eng.attacks(x) for x in eng.s.on_board())
        acc = holder.counters.get("att", 0.0) + att
        n = int(acc // 25)
        holder.counters["att"] = acc - 25 * n
        for h in eng.heroes():
            eng.level(h, n, src="因缘图腾")


@card("始祖熊灵")
class 始祖熊灵天赋:
    """获得1张始祖熊灵，其被视为【大河】英雄且不占人口。战斗开始时，我方每比敌方多1个单位，始祖熊灵永久增加2级。
    （天赋：拿到时召唤熊灵；熊灵本体的成长在单位实现里）"""
    def on_use(self, eng, t):
        u = eng.deploy("始祖熊灵(赠送)", kind="hero", pop=False)
        if u:
            u.keywords.add("大河流域")


@card("始祖熊灵(赠送)")
class 始祖熊灵:
    """战斗开始时，我方每比敌方多1个单位，始祖熊灵永久增加2级。（被视为【大河】英雄、不占人口；可合成同名）
    "被视为【大河】英雄"用 grant_keywords 每回合重新赋予（引擎回合开始会清空 u.keywords）。
    待核对：敌方单位数取对手的人口上限（看不到对手，用同等级的人口上限）。"""
    def grant_keywords(self, eng, holder):
        if holder is not None and holder in eng.s.units:
            holder.keywords.add("大河流域")

    def on_battle_start(self, eng, u):
        ours = len(eng.s.on_board())
        enemy = B.pop_cap(max(1, min(6, eng.s.lord_level)))
        if ours > enemy:
            eng.level(u, 2 * (ours - enemy), src="始祖熊灵")


# ============================================================ 搬等级
@card("众星拱月")
class 众星拱月:
    """指定1名英雄，夺取其他英雄各10级。待核对：被夺取的人最低降到 1 级（不够 10 级就夺走全部真实等级 −1）。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        got = 0.0
        for x in eng.heroes():
            if x is t:
                continue
            a = min(10.0, max(0.0, x.level - 1))
            x.level -= a
            got += a
        eng.level(t, got, src="众星拱月")


@card("夺取之术")
class 夺取之术:
    """指定1名英雄，夺取最近1名己方英雄的等级（最多10级，不包含临时等级）。
    10-19 规则确认：只限周围 1 格（1 格外夺不到），被夺的人最低留 1 级 → 和关键词「夺取」同一口径（eng.steal：周围 1 格随机 1 名）。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        eng.steal(t, src="夺取之术")                       # 10-19 确认：只限周围 1 格、最低留 1 级


@card("遗志")
class 遗志:
    """指定场上1名英雄将其移除，并转移其等级（临时等级除外）给1名随机英雄。
    （随机 = 移除后场上剩下的英雄里均匀选；被移除算离场 → 枫叶舞宿主触发红叶染锋）
    10-19 规则确认：装备回手（eng.remove 把非绑定装备退回手牌）、英雄离场、不返能量（不走 eng.sell）。"""
    def on_use(self, eng, t):
        if t is None or t not in eng.s.units:
            return False
        lv = t.level
        eng.remove(t, src="遗志")
        rest = [x for x in eng.heroes()]
        if rest:
            eng.level(eng.pick(rest, 1)[0], lv, src="遗志")


@card("传承")
class 传承:
    """指定场上1名10级以上的英雄，将其等级随机分配给其他英雄中等级最低的3名（不会分配给同一英雄）。
    10-19 规则确认：平均分给等级最低的 3 名（召唤物不算英雄、不参与），原英雄留 1 级、留在场上。
    整数平均分：每人 total // n，余下的几级随机各 +1。"""
    def on_use(self, eng, t):
        if t is None or t.level < 10:
            return False
        lows = eng.lowest([x for x in eng.heroes() if x is not t], 3)
        if not lows:
            return
        total = int(t.level - 1)
        t.level = 1.0                                       # 10-19 确认：原英雄留 1 级
        base, rem = divmod(total, len(lows))                # 10-19 确认：平均分（原来随机切 3 份）
        extra = set(id(x) for x in eng.pick(lows, rem))
        for x in lows:
            eng.level(x, base + (1 if id(x) in extra else 0), src="传承")


@card("红叶染锋")
class 红叶染锋:
    """枫叶舞英雄离场后，其真实等级的50%会保留给下一个枫叶舞英雄，限一次。（强化牌：放进 lord_cards；结算在 engine.remove / bind_maple）"""
    def on_use(self, eng, t):
        if "红叶染锋" not in eng.s.lord_cards:
            eng.s.lord_cards.append("红叶染锋")


# ============================================================ 加等级 / 触发
@card("特训战术")
class 特训战术:
    """转瞬 指定1名英雄等级+4。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        eng.level(t, 4, src="特训战术")


@card("简易协作")
class 简易协作:
    """指定1名英雄等级+3，随机1名英雄等级+2。待核对："随机 1 名"是否可能是被指定的人 —— 按"场上任意 1 名"实现。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        eng.level(t, 3, src="简易协作")
        r = eng.pick(eng.heroes(), 1)
        if r:
            eng.level(r[0], 2, src="简易协作")


@card("精密协作")
class 精密协作:
    """指定1名英雄等级+4，随机1名英雄等级+3。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        eng.level(t, 4, src="精密协作")
        r = eng.pick(eng.heroes(), 1)
        if r:
            eng.level(r[0], 3, src="精密协作")


@card("有福同享")
class 有福同享:
    """指定1名英雄，场上和其品阶相同的英雄等级+4。（含被指定的人自己）"""
    def on_use(self, eng, t):
        if t is None:
            return False
        q = quality(t.name)
        for x in eng.heroes():
            if quality(x.name) == q:
                eng.level(x, 4, src="有福同享")


@card("临阵磨枪")
class 临阵磨枪:
    """指定1名英雄，触发其整备效果。（目标必须有"整备："条目；被触发的整备 = eng.trigger）"""
    def on_use(self, eng, t):
        if t is None or not has_keyword(t.name, "整备"):
            return False
        eng.trigger(t, "prepare")


@card("加速成长")
class 加速成长天赋:
    """获得2张天赋牌加速成长：指定1名英雄等级+5。（天赋牌：拿到时给 2 张赠送牌）"""
    def on_use(self, eng, t):
        if t is not None:                 # 赠送牌被指定目标打出
            eng.level(t, 5, src="加速成长")
            return
        eng.gain_card("加速成长")
        eng.gain_card("加速成长")


@card("扶摇直上")
class 扶摇直上:
    """棋手等级提升1。如果已满级，改为获得10能量。"""
    def on_use(self, eng, t):
        s = eng.s
        if s.lord_level >= B.MAX_LORD_LEVEL:
            s.energy += 10
            return
        s.lord_level += 1
        eng.emit("lord_levelup", level=s.lord_level)


@card("弈星")
class 弈星:
    """棋手升级时，棋手经验+2。"""
    def react_lord_levelup(self, eng, holder, level=0):
        if holder is not None and holder in eng.s.units:
            eng.s.xp += 2 * (2 if holder.awakened else 1)


@card("孙悟空")
class 孙悟空:
    """闪现，夺取 整备：拥有闪现效果的英雄等级+2。（夺取：上场 / 合成时夺取周围 1 格随机 1 名己方英雄最多 10 级）
    觉醒（卡面）：整备 +4，夺取仍是 1 次 → 自己处理觉醒（原来引擎把登场执行两次 = 夺两次）。"""
    awaken_custom = True     # 卡面：觉醒文本里夺取只出现 1 次

    def on_prepare(self, eng, u):
        for x in eng.heroes():
            if eng.has_kw(x, "闪现"):
                eng.level(x, 4 if u.awakened else 2, src="孙悟空")   # 卡面觉醒：+4

    def on_deploy(self, eng, u):
        eng.steal(u, src="孙悟空(夺取)")


@card("芈月")
class 芈月:
    """每当你使用效果牌时，己方穿戴装备最多的1名【逐鹿】英雄等级+1。"""
    def react_effect_card(self, eng, holder, card="", target=None):
        if holder is None or holder not in eng.s.units:
            return
        zl = [x for x in eng.heroes() if eng.is_faction(x, "逐鹿")]
        if zl:
            m = max(len(x.equips) for x in zl)
            t = eng.pick([x for x in zl if len(x.equips) == m], 1)[0]
            eng.level(t, 2 if holder.awakened else 1, src="芈月")


@card("西施")
class 西施:
    """整备：获得1张推演战术。（觉醒：1 张高阶推演战术 —— 卡面，不是 2 张普通）"""
    awaken_custom = True     # 卡面觉醒：1 张高阶推演战术

    def on_prepare(self, eng, u):
        eng.gain_card("高阶推演战术" if u.awakened else "推演战术")


@card("推演战术")
class 推演战术:
    """转瞬 指定1名【逐鹿】英雄，使其本回合法力值上限降低（每次使用都提升，上限 20）。战斗效果；作为效果牌会触发芈月等。"""
    def on_use(self, eng, t):
        if t is None or not eng.is_faction(t, "逐鹿"):
            return False


@card("干将莫邪")
class 干将莫邪:
    """干将莫邪在场时，获得普通装备双剑·雄。（觉醒：双剑·雄 + 双剑·雌；出生自带、拿不下来）"""
    awaken_custom = True

    def on_deploy(self, eng, u):
        for e in (["双剑·雄", "双剑·雌"] if u.awakened else ["双剑·雄"]):
            if e not in u.equips:
                eng.equip(u, e, bound=True, forge=False, force=True, src="干将莫邪(出生自带)")


@card("鲁班大师")
class 鲁班大师:
    """鲁班大师在场时，召唤1个等级与自身相同的机关傀儡。
    "等级与自身相同"按持续同步读（卡面重读 10-18）：每场战斗开始前把傀儡等级抬到大师当前等级（只抬不降：傀儡能吃效果卡，10-19 规则确认）。
    待核对：觉醒召「机关傀儡·改」卡库没有条目，仍按普通傀儡。"""
    def on_deploy(self, eng, u):
        if not any(x.name == "机关傀儡" for x in eng.s.units):
            eng.summon("机关傀儡", near=u, level=u.level)

    def on_battle_begin(self, eng, holder):
        if holder is None or holder not in eng.s.units:
            return
        for x in eng.s.units:
            if x.name == "机关傀儡" and x.level < holder.level:
                eng.level(x, holder.level - x.level, src="鲁班大师(等级同步)")   # 卡面：等级与自身相同


# ============================================================ 装备
@card("学识宝石")
class 学识宝石:
    """唯一 每回合开始时等级+8%（至少+3级）并减少2点法力值上限。"""
    def on_round_start(self, eng, holder):
        if holder is not None and holder in eng.s.units:
            eng.level(holder, max(3, math.floor(holder.level * 0.08)), src="学识宝石")


@card("百锻刀", "聚灵法杖")
class 成长神装:
    """唯一，无法拆卸。攻击力随击杀永久成长（战斗效果，不影响养成）。"""


# ============================================================ 阿离：枫叶舞强化
@card("翩跹舞步·新生")
class 翩跹舞步新生:
    """为枫叶舞挑选1张强化牌。枫叶舞英雄每回合等级+2，且转移后恢复10%最大生命值。
    打出时进 lord_cards 并从强化牌池选 1 张（与另外四张翩跹舞步一致，eng.pick_lord_boost）；之后回合开始枫叶舞英雄 +2。
    转移后回血是战斗效果。"""
    def on_use(self, eng, t):
        eng.add_lord_card("翩跹舞步·新生")
        eng.pick_lord_boost()

    def on_round_start(self, eng, holder):
        u = eng.s.unit(eng.s.maple) if eng.s.maple else None
        if u:
            eng.level(u, 2, src="翩跹舞步·新生")


@card("枫叶舞")
class 枫叶舞:
    """使用技能，指向1名英雄（绑定）。保命效果是战斗效果；绑定对象是强化牌（翩跹舞步·新生 / 红叶染锋…）的作用对象。"""


# ============================================================ 锦囊 / 兑换券 / 征召令
def _jinnang(tier):
    class _J:
        f"""{tier}阶效果锦囊：选择1张{tier}阶效果牌（N 选 1）。选项按历史出现频率抽 3 张（data/engine/draw_stats.json），由推演打法选。"""
        def on_use(self, eng, t):
            from agent.engine.rollout import draw_stats
            js = (draw_stats().get("jinnang") or {}).get(str(tier)) or {}
            st = js.get("cards") or {}
            if (js.get("n") or 0) < 10:
                # 实测样本太少（6 阶锦囊只见过 1 次）→ 这一阶的效果牌等概率（2026-10-14 确认：众星拱月是 6 阶效果牌，锦囊也能开出来）
                from agent.sim.engine import card_db, category, quality
                pool = [n for n in card_db() if category(n) == "效果牌" and quality(n) == tier]
                st = {n: 1 for n in pool} or st
            if not st:
                return
            names, ws = list(st), list(st.values())
            opts = []
            for _ in range(30):
                x = eng.rng.choices(names, ws)[0]
                if x not in opts:
                    opts.append(x)
                if len(opts) >= 3:
                    break
            pick = eng.choose("jinnang", opts, tier=tier)
            if pick:
                eng.gain_card(pick)
    return _J


for _t in range(1, 7):
    card(f"{_t}阶效果锦囊")(_jinnang(_t))


def _exchange(tier):
    class _E:
        f"""{tier}阶英雄兑换券：卡牌3选1（{tier}阶英雄牌），按牌库剩余抽。"""
        def on_use(self, eng, t):
            opts = []
            for _ in range(20):
                x = eng.draw_hero(tier)
                if x and x not in opts:
                    opts.append(x)
                if len(opts) >= 3:
                    break
            pick = eng.choose("hero", opts, tier=tier)
            if pick:
                eng.take(pick)
                eng.gain_card(pick)
    return _E


for _t in range(1, 6):
    card(f"{_t}阶英雄兑换券")(_exchange(_t))


def _levy(fac):
    class _L:
        f"""{fac}征召令：获得2张【{fac}】英雄牌（按棋手等级的商店品阶、牌库剩余抽）。"""
        def on_use(self, eng, t):
            for _ in range(2):
                for _try in range(10):
                    x = eng.draw_hero(None, fac)
                    if x:
                        eng.take(x)
                        eng.gain_card(x)
                        break
    return _L


for _f in ("河洛", "逐鹿", "大河流域", "日落海", "三分之地"):
    card(f"{_f.replace('流域', '').replace('之地', '')}征召令")(_levy(_f))
