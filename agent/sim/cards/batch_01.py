"""第 1 批卡牌实现（data/engine/sim_batches.json batches[1]，79 张）。

只实现养成相关效果（等级 / 临时等级 / 觉醒 / 装备 / 获得牌 / 能量 / 经验 / 人口 / 召唤 / 移除 / 触发别人的效果），
战斗数值只在 docstring 写一句。写卡规范见 README.md。

2026-10-14 统一接口：阿科米亚核心 = eng.s.core / eng.core_add（发 core_up，海诺用），开战分配由引擎做
（星源星徽 blocks_core）；穿戴限制由 engine.can_wear 读卡面；觉醒进度 eng.awaken_step；被触发的登场 / 整备 eng.trigger；
手牌持有回合 eng.take_meta；只在本场战斗存在的召唤物 eng.summon(battle_only=True)。
"""
from __future__ import annotations

import json
import math
import statistics
from functools import lru_cache

from agent.engine import board as B
from ..engine import ROOT, card, card_db, card_text, category, faction, has_effect, quality
from .pilot import _levy

# 「阵营」是否把"无阵营"也算作一个阵营（八方来援 / 阵营祝福 / 因材施教）。
# 待核对：卡面"阵营"按最自然的读法 = 河洛 / 逐鹿 / 大河流域 / 日落海 / 三分之地 五个，无阵营不算。
COUNT_NEUTRAL_AS_FACTION = False
NEUTRAL = ("无阵营", "", "通用")


# ============================================================ 通用辅助
def _alive(eng, holder) -> bool:
    return holder is not None and holder in eng.s.units


def _has_entry(name: str, kw: str) -> bool:
    """卡面有"kw："这一条效果，或第一行单独列出 kw（engine.has_effect）。"""
    return has_effect(name, kw)


_has_attr = _has_entry


def _hero_names(pred=lambda n: True) -> list:
    return [n for n, c in card_db().items() if c.get("category") == "英雄牌" and pred(n)]


def _real_factions(eng, units=None) -> dict:
    """场上英雄按阵营分组：{阵营: [单位…]}（无阵营默认不算，见 COUNT_NEUTRAL_AS_FACTION）。"""
    out: dict = {}
    for u in (units if units is not None else eng.heroes()):
        f = eng.faction_of(u)
        if f in NEUTRAL and not (COUNT_NEUTRAL_AS_FACTION and f == "无阵营"):
            continue
        out.setdefault(f, []).append(u)
    return out


def _draw_hero_where(eng, pred, tier=None, tries: int = 12):
    """按"棋手等级的商店品阶概率 + 牌库剩余"抽 1 张满足 pred 的英雄牌（pred 只看卡名）。
    某品阶没有满足条件的牌就重抽品阶；还抽不到就在本等级能出的品阶里合并抽。"""
    odds = B.meta.TIER_ODDS.get(eng.s.lord_level) or {1: 1.0}
    for _ in range(tries):
        t = tier
        if t is None:
            t = eng.rng.choices(list(odds), list(odds.values()))[0]
        names = [n for n in _hero_names(pred) if quality(n) == t]
        ws = [eng.pool_left(n) for n in names]
        if names and sum(ws) > 0:
            return eng.rng.choices(names, ws)[0]
        if tier is not None:
            break
    tiers = [tier] if tier is not None else list(odds)
    names = [n for n in _hero_names(pred) if quality(n) in tiers]
    ws = [eng.pool_left(n) for n in names]
    if names and sum(ws) > 0:
        return eng.rng.choices(names, ws)[0]
    return None


def _gain_heroes(eng, k: int, pred=lambda n: True, tier=None) -> int:
    got = 0
    for _ in range(k):
        x = _draw_hero_where(eng, pred, tier)
        if x:
            eng.take(x)
            eng.gain_card(x)
            got += 1
    return got


def _hero_options(eng, k: int, pred=lambda n: True, tier=None) -> list:
    """N 选 1 的选项：k 张不同的英雄牌。"""
    opts: list = []
    for _ in range(k * 10):
        x = _draw_hero_where(eng, pred, tier)
        if x and x not in opts:
            opts.append(x)
        if len(opts) >= k:
            break
    return opts


def _pick_heroes(eng, k: int, m: int, kind: str, pred=lambda n: True, tier=None) -> None:
    """卡牌 k 选 m（英雄牌）：推演打法逐张选，选中的进手牌、记牌库。"""
    opts = _hero_options(eng, k, pred, tier)
    for _ in range(m):
        p = eng.choose(kind, opts, k=k, m=m)
        if not p:
            break
        opts.remove(p)
        eng.take(p)
        eng.gain_card(p)


@lru_cache(maxsize=1)
def _base_equips() -> tuple:
    """8 件基础装备（拖上英雄弹四选一的那种），来自 data/equip_recipes.json。"""
    d = json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8"))
    return tuple(d.get("recipes", {}).keys())


@lru_cache(maxsize=1)
def _tactic_cards() -> tuple:
    """战术牌 = 【逐鹿】效果牌里名字带"战术"的 10 张（援护 / 招募 / 演练 / 推演 / 特训 及其高阶版）。
    孙尚香的"战术·X"是她专属的强化牌（无阵营），不算。"""
    return tuple(n for n, c in card_db().items()
                 if c.get("category") == "效果牌" and "战术" in n and faction(n) == "逐鹿")


@lru_cache(maxsize=1)
def _sunset_gear() -> tuple:
    """5 件【日落海】专属装备（卡面第一行写"【日落海】专属装备"）。"""
    return tuple(n for n, c in card_db().items()
                 if c.get("category") == "装备牌" and (c.get("skill") or "").startswith("【日落海】专属装备"))


@lru_cache(maxsize=1)
def _special_gear() -> tuple:
    """珍品宝箱的选项池：特殊装备 / 天赋装备。
    待核对：卡库没有"特殊 / 天赋装备"标记，按"品阶 ≥3 的装备牌、不含日落海专属（远洋宝藏的池）、不含干将的双剑"取。"""
    sun = set(_sunset_gear())
    return tuple(n for n, c in card_db().items()
                 if c.get("category") == "装备牌" and quality(n) >= 3 and n not in sun and "双剑" not in n)


def _destroy_equips(u) -> int:
    """摧毁 u 身上所有能拿下来的装备，返回摧毁件数。待核对：出生自带 / 无法拆卸（u.bound）的不摧毁、不计数。"""
    gone = [e for e in u.equips if e not in u.bound]
    u.equips = [e for e in u.equips if e in u.bound]
    return len(gone)


def awaken_step(eng, u, src: str = "") -> bool:
    """觉醒进度 +1（engine.awaken_step；已觉醒的不能再用 → False）。"""
    return eng.awaken_step(u, 1, src=src)


def totems(eng) -> int:
    return sum(1 for x in eng.s.units if "图腾" in x.name)


# ------------------------------------------------------------ 敌方最高等级（移花接木）
@lru_cache(maxsize=1)
def _enemy_max_by_round() -> dict:
    """data/engine/enemy_levels_all.json：每回合敌方最高等级的中位数（读数为 0 的不算）。"""
    try:
        d = json.loads((ROOT / "data/engine/enemy_levels_all.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    acc: dict = {}
    for per in d.values():
        if not isinstance(per, dict):
            continue
        for r, v in per.items():
            try:
                m = float((v or {}).get("enemy_max") or 0)
                r = int(r)
            except (TypeError, ValueError, AttributeError):
                continue
            if m > 0:
                acc.setdefault(r, []).append(m)
    return {r: statistics.median(v) for r, v in acc.items() if v}


def enemy_max_level(eng):
    tab = _enemy_max_by_round()
    if not tab:
        return None
    r = min(tab, key=lambda k: abs(k - eng.s.round))
    return tab[r]


# ============================================================ 英雄
@card("吕布")
class 吕布:
    """开团：等级最高的3名【三分】英雄获得20%物理和法术吸血，若目标为吕布，效果翻倍。
    战斗效果：吸血，不影响养成。"""


@card("海月")
class 海月:
    """牺牲：将敌方等级最高的英雄困入水月幻境控制4秒，并造成等同于海月生命值的真实伤害。
    战斗效果：控制 + 真实伤害，不影响养成（拥有"牺牲"，守约等牺牲反应器照常由引擎判定）。"""


@card("鲁班七号")
class 鲁班七号:
    """你的英雄每合成3次，随机获得1张战术牌。
    在场期间计数（鲁班七号自己合成也算）；觉醒：每 3 次得 1 张**高阶**战术牌（卡面，不是 2 张随机）。
    普通版从 5 张普通战术牌里均匀抽，觉醒版从 5 张高阶战术牌里均匀抽（卡面"战术牌" / "高阶战术牌"）。
    待核对：计数是否从鲁班七号上场起算（按此实现）；是否按品阶加权未知。"""
    def react_merge(self, eng, holder, unit=None, **kw):
        if not _alive(eng, holder):
            return
        c = holder.counters.get("merge3", 0) + 1
        if c >= 3:
            c = 0
            hi = holder.awakened                            # 卡面觉醒：1 张高阶战术牌（原来 2 张、可能抽到普通）
            pool = [n for n in _tactic_cards() if n.startswith("高阶") == hi] or list(_tactic_cards())
            eng.gain_card(eng.pick(pool, 1)[0])
        holder.counters["merge3"] = c


@card("海诺")
class 海诺:
    """每当阿科米亚核心等级提升时，核心等级额外+1。
    觉醒翻倍 = 额外 +2。额外这 1 级不算新的"提升"（eng.core_add(event=False)，不会再触发自己）。"""
    def react_core_up(self, eng, holder, n=0, **kw):
        if not _alive(eng, holder):
            return
        eng.core_add(2 if holder.awakened else 1, src="海诺", event=False)


@card("露娜")
class 露娜:
    """闪现
    登场：阿科米亚核心等级+2，并且随机触发2名英雄的整备效果。
    "随机 2 名英雄"取场上有"整备："效果的英雄（不含露娜自己），不同的 2 名（eng.trigger：被触发的整备）。
    觉醒（卡面）：核心 +4、随机触发 4 名 → 自己处理成不同的 4 名（原来引擎执行两次各抽 2 名，可能重复）。
    待核对：没有整备效果的英雄会不会被抽中（白触发）。"""
    awaken_custom = True     # 卡面觉醒：核心 +4、触发 4 名

    def on_deploy(self, eng, u):
        eng.core_add(4 if u.awakened else 2, src="露娜")    # 卡面觉醒：核心 +4
        cands = [x for x in eng.heroes() if x is not u and _has_entry(x.name, "整备")]
        for t in eng.pick(cands, 4 if u.awakened else 2):   # 卡面觉醒：4 名
            eng.trigger(t, "prepare")


@card("杨玉环")
class 杨玉环:
    """杨玉环在场时，每过3个回合获得1张弦音。
    退场：获得1张弦音。
    在场回合数从上场后第一个回合开始计（回合开始时 +1，每满 3 得 1 张）。
    觉醒（卡面）：两处都换成 1 张妙音（不限品阶）—— 不是 2 张弦音。
    待核对：上场当回合算不算第 1 个回合。"""
    awaken_custom = True     # 卡面觉醒：弦音 → 妙音，张数不变

    def on_round_start(self, eng, holder):
        if not _alive(eng, holder):
            return
        c = holder.counters.get("xianyin_rounds", 0) + 1
        if c >= 3:
            c = 0
            eng.gain_card("妙音" if holder.awakened else "弦音")   # 卡面觉醒：1 张妙音
        holder.counters["xianyin_rounds"] = c

    def on_sell(self, eng, u):
        eng.gain_card("妙音" if u.awakened else "弦音")             # 卡面觉醒：退场 1 张妙音


@card("周瑜")
class 周瑜:
    """登场：全体英雄等级+1。（含周瑜自己）"""
    def on_deploy(self, eng, u):
        for x in eng.heroes():
            eng.level(x, 1, src="周瑜")


# ============================================================ 觉醒进度
@card("弦音")
class 弦音:
    """指定1名1-3阶英雄，使其觉醒进度+1。（已觉醒的不能用）"""
    def on_use(self, eng, t):
        if t is None or not 1 <= quality(t.name) <= 3:
            return False
        return awaken_step(eng, t, src="弦音")


@card("妙音")
class 妙音:
    """指定1名英雄，使其觉醒进度+1。（已觉醒的不能用）"""
    def on_use(self, eng, t):
        return awaken_step(eng, t, src="妙音")


@card("天籁琴音")
class 天籁琴音:
    """获得3张妙音（妙音：指定1名英雄，使其觉醒进度+1）。"""
    def on_use(self, eng, t):
        for _ in range(3):
            eng.gain_card("妙音")


# ============================================================ 获得英雄牌
@card("吸纳人才")
class 吸纳人才:
    """卡牌4选1（英雄牌）。待核对：选项品阶按棋手等级的商店概率抽（按此实现）。"""
    def on_use(self, eng, t):
        _pick_heroes(eng, 4, 1, "hero")


@card("广纳良才")
class 广纳良才:
    """卡牌5选2（英雄牌）。待核对：选项品阶按棋手等级的商店概率抽。"""
    def on_use(self, eng, t):
        _pick_heroes(eng, 5, 2, "hero")


@card("高级人才")
class 高级人才:
    """随机获得1张特定品阶（棋手等级+1）的英雄牌。品阶封顶 5。"""
    def on_use(self, eng, t):
        _gain_heroes(eng, 1, tier=min(eng.s.lord_level + 1, 5))


def _tier_heroes(tier):
    class _T:
        f"""获得2张{tier}阶英雄牌。（按牌库剩余抽）"""
        def on_use(self, eng, t):
            _gain_heroes(eng, 2, tier=tier)
    _T.__doc__ = f"获得2张{tier}阶英雄牌。（按牌库剩余抽）"
    return _T


for _t in (2, 4, 5):
    card(f"{_t}阶英雄")(_tier_heroes(_t))


card("无阵营征召令")(_levy("无阵营"))
card("大河流域征召令")(_levy("大河流域"))


@card("图腾仪式")
class 图腾仪式:
    """随机获得1张与图腾相关的【大河流域】英雄牌。
    "与图腾相关" = 卡面提到"图腾"的大河英雄（项羽 / 鬼谷子 / 虞姬 / 少司缘 / 大司命 / 刘邦…）。
    待核对：是否受棋手等级的品阶限制 —— 按"不限品阶、按牌库剩余抽"实现。"""
    def on_use(self, eng, t):
        names = _hero_names(lambda n: faction(n) == "大河流域" and "图腾" in card_text(n))
        ws = [eng.pool_left(n) for n in names]
        if names and sum(ws) > 0:
            x = eng.rng.choices(names, ws)[0]
            eng.take(x)
            eng.gain_card(x)


@card("呼朋唤友")
class 呼朋唤友:
    """指定1名英雄，随机获得1张与其同阵营的英雄牌。
    待核对：品阶按棋手等级的商店概率（按此实现）；无阵营英雄的"同阵营" = 无阵营。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        f = eng.faction_of(t)
        _gain_heroes(eng, 1, pred=lambda n: faction(n) == f)


@card("整备高手")
class 整备高手:
    """获得2张具有整备效果的英雄牌。待核对：品阶按棋手等级的商店概率。"""
    def on_use(self, eng, t):
        _gain_heroes(eng, 2, pred=lambda n: _has_entry(n, "整备"))


@card("登场人物")
class 登场人物:
    """获得2张具有登场效果的英雄牌。待核对：品阶按棋手等级的商店概率。"""
    def on_use(self, eng, t):
        _gain_heroes(eng, 2, pred=lambda n: _has_entry(n, "登场"))


@card("牺牲品")
class 牺牲品:
    """获得2张具有牺牲效果的英雄牌。待核对：品阶按棋手等级的商店概率。"""
    def on_use(self, eng, t):
        _gain_heroes(eng, 2, pred=lambda n: _has_attr(n, "牺牲"))


@card("闪现突击者")
class 闪现突击者:
    """获得2张具有闪现效果的英雄牌。待核对：品阶按棋手等级的商店概率。"""
    def on_use(self, eng, t):
        _gain_heroes(eng, 2, pred=lambda n: _has_attr(n, "闪现"))


@card("开团号令")
class 开团号令:
    """获得2张具有开团效果的英雄牌。待核对：品阶按棋手等级的商店概率。"""
    def on_use(self, eng, t):
        _gain_heroes(eng, 2, pred=lambda n: _has_entry(n, "开团"))


@card("乱世")
class 乱世:
    """卡牌3选1（登场英雄牌）。待核对：品阶按棋手等级的商店概率。"""
    def on_use(self, eng, t):
        _pick_heroes(eng, 3, 1, "hero", pred=lambda n: _has_entry(n, "登场"))


@card("高阶招募战术")
class 高阶招募战术:
    """转瞬
    3选1【逐鹿】英雄牌。待核对：品阶按棋手等级的商店概率。"""
    def on_use(self, eng, t):
        _pick_heroes(eng, 3, 1, "hero", pred=lambda n: faction(n) == "逐鹿")


@card("求贤")
class 求贤:
    """指定1名英雄，发现与其同阵营的3张英雄牌，从中选择1张。待核对：品阶按棋手等级的商店概率。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        f = eng.faction_of(t)
        _pick_heroes(eng, 3, 1, "hero", pred=lambda n: faction(n) == f)


@card("照影")
class 照影:
    """随机获得1张场上英雄牌。（场上随机 1 名英雄的同名牌；召唤物不算）
    待核对：是否从公共牌库里扣（按扣实现）；牌库空了是否还能拿（按"能拿"实现）。"""
    def on_use(self, eng, t):
        hs = eng.heroes()
        if not hs:
            return False
        x = eng.pick(hs, 1)[0].name
        eng.take(x)
        eng.gain_card(x)


@card("久经沙场")
class 久经沙场:
    """指定1名已觉醒的【逐鹿】英雄，获得1张该英雄牌。待核对：是否从公共牌库里扣（按扣实现）。"""
    def on_use(self, eng, t):
        if t is None or not t.awakened or not eng.is_faction(t, "逐鹿"):
            return False
        eng.take(t.name)
        eng.gain_card(t.name)


# ============================================================ 经验 / 能量 / 刷新
@card("研读会")
class 研读会:
    """获得3棋手经验。（经验够了立即升级）"""
    def on_use(self, eng, t):
        eng.add_xp(3, src="研读会")


@card("经典研读会")
class 经典研读会:
    """获得4棋手经验。（经验够了立即升级）"""
    def on_use(self, eng, t):
        eng.add_xp(4, src="经典研读会")


@card("惊喜补贴")
class 惊喜补贴:
    """获得3次免费刷新。"""
    def on_use(self, eng, t):
        eng.s.free_refresh += 3


@card("超级补贴")
class 超级补贴:
    """获得4次免费刷新。"""
    def on_use(self, eng, t):
        eng.s.free_refresh += 4


@card("展翅腾飞")
class 展翅腾飞:
    """本回合刷新商店不消耗能量。
    eng.s.free_refresh_all = True（回合开始引擎清掉）；eng.refresh_shop() 认它。"""
    def on_use(self, eng, t):
        eng.s.free_refresh_all = True


@card("能量大爆炸")
class 能量大爆炸:
    """获得5能量。"""
    def on_use(self, eng, t):
        eng.s.energy += 5


# ============================================================ 加等级
@card("八方来援")
class 八方来援:
    """指定1名英雄等级+2。场上每有1种阵营，额外触发1次此效果。
    阵营数 = 场上英雄的不同阵营数。待核对：无阵营算不算一种阵营（按不算，见 COUNT_NEUTRAL_AS_FACTION）。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        for _ in range(1 + len(_real_factions(eng))):
            eng.level(t, 2, src="八方来援")


@card("武装之力")
class 武装之力:
    """所有英雄获得等同于自身装备数量的等级。（出生自带的装备也算件数）"""
    def on_use(self, eng, t):
        for x in eng.heroes():
            eng.level(x, len(x.equips), src="武装之力")


@card("换帅")
class 换帅:
    """指定1名【三分】英雄与其最近的【三分】英雄交换等级。（交换永久等级，临时等级不动）"""
    def on_use(self, eng, t):
        if t is None or not eng.is_faction(t, "三分之地"):
            return False
        o = eng.nearest(t, lambda x: eng.is_faction(x, "三分之地"))
        if not o:
            return False
        o = o[0]
        t.level, o.level = o.level, t.level
        eng.note("交换等级", a=t.name, b=o.name, src="换帅")


@card("高阶演练战术")
class 高阶演练战术:
    """转瞬
    所有英雄等级+2。"""
    def on_use(self, eng, t):
        for x in eng.heroes():
            eng.level(x, 2, src="高阶演练战术")


@card("高阶特训战术")
class 高阶特训战术:
    """转瞬
    指定1名英雄等级+8。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        eng.level(t, 8, src="高阶特训战术")


@card("阵营祝福")
class 阵营祝福:
    """各阵营中随机1名英雄等级+4。待核对：无阵营算不算一个阵营（按不算）。"""
    def on_use(self, eng, t):
        for f, us in _real_factions(eng).items():
            eng.level(eng.pick(us, 1)[0], 4, src="阵营祝福")


@card("扶倾济弱")
class 扶倾济弱:
    """使场上等级最低（临时等级除外）的英雄提升9级。（并列随机 1 名）"""
    def on_use(self, eng, t):
        lo = eng.lowest(eng.heroes(), 1)
        if not lo:
            return False
        eng.level(lo[0], 9, src="扶倾济弱")


@card("战前宣誓")
class 战前宣誓:
    """指定1名英雄，获得自身品阶×3的临时等级。（"自身" = 被指定的英雄）"""
    def on_use(self, eng, t):
        if t is None:
            return False
        eng.temp(t, quality(t.name) * 3, src="战前宣誓")


@card("逍遥自在")
class 逍遥自在:
    """所有无阵营英雄获得2×场上无阵营英雄数量的等级。"""
    def on_use(self, eng, t):
        ns = [x for x in eng.heroes() if eng.faction_of(x) == "无阵营"]
        for x in ns:
            eng.level(x, 2 * len(ns), src="逍遥自在")


@card("图腾祝福")
class 图腾祝福:
    """随机1名【大河流域】英雄等级+5。场上每有1个图腾，额外触发1次此效果。（每次独立随机）"""
    def on_use(self, eng, t):
        dh = [x for x in eng.heroes() if eng.is_faction(x, "大河流域")]
        if not dh:
            return False
        for _ in range(1 + totems(eng)):
            eng.level(eng.pick(dh, 1)[0], 5, src="图腾祝福")


@card("移花接木")
class 移花接木:
    """在下场战斗中，随机1名英雄获得敌方当前等级最高英雄的一半等级（包含临时等级）。
    实现：使用时（备战期）给随机 1 名英雄临时等级 = ⌊敌方最高等级 / 2⌋，临时等级在下场战斗结束清零。
    敌方最高等级取 data/engine/enemy_levels_all.json 同回合实测中位数（引擎没有敌方模型）。
    待核对：取整方式；"下场战斗"的随机目标是在开战时才定（按使用时定实现）。"""
    def on_use(self, eng, t):
        hs = eng.heroes()
        m = enemy_max_level(eng)
        if not hs or m is None:
            return
        eng.temp(eng.pick(hs, 1)[0], math.floor(m / 2), src="移花接木")


@card("众星闪耀")
class 众星闪耀:
    """全体己方英雄等级+10。每持有一回合，此效果增强5级。
    持有回合数 = 现在的回合 − 拿到这张牌的回合（手牌状态 eng.take_meta：同名先拿到的先用）。
    待核对："持有一回合"是否按回合开始计数（按回合号差实现）。"""
    def on_use(self, eng, t):
        s = eng.s
        got = eng.take_meta("众星闪耀").get("got", s.round)
        held = max(0, s.round - got)
        for x in eng.heroes():
            eng.level(x, 10 + 5 * held, src="众星闪耀")


@card("破境之证")
class 破境之证:
    """指定1名英雄立即提升20级，之后每回合提升10级。
    "之后每回合"挂在被指定的单位身上（u.counters["破境之证"] = 用过几张）；回合开始的结算由隐藏持续效果
    "破境之证(持续)"（放进 eng.s.talents）执行，单位离场即失效。
    10-19 确认的规则：每回合 +10 永久、跟英雄走、觉醒后保留（合成觉醒是同一单位，计数不丢）、卖掉消失 —— 与实现一致。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        eng.level(t, 20, src="破境之证")
        t.counters["破境之证"] = t.counters.get("破境之证", 0) + 1
        if "破境之证(持续)" not in eng.s.talents:
            eng.s.talents.append("破境之证(持续)")


@card("破境之证(持续)")
class 破境之证持续:
    """（内部持续效果，不是真卡）回合开始：每个被破境之证指定过的英雄等级 +10 × 张数。"""
    def on_round_start(self, eng, holder):
        for x in eng.heroes():
            k = x.counters.get("破境之证", 0)
            if k:
                eng.level(x, 10 * k, src="破境之证")


# ============================================================ 触发效果
@card("唤醒")
class 唤醒:
    """指定1名英雄，触发其登场效果。（目标必须有"登场："条目）"""
    def on_use(self, eng, t):
        if t is None or not _has_entry(t.name, "登场"):
            return False
        eng.trigger(t, "deploy")


@card("集体唤醒")
class 集体唤醒:
    """立即触发所有英雄的登场效果。（只触发有"登场："条目的英雄）"""
    def on_use(self, eng, t):
        for x in list(eng.heroes()):
            if _has_entry(x.name, "登场"):
                eng.trigger(x, "deploy")


@card("全军动员")
class 全军动员:
    """立即触发所有英雄的整备效果。（只触发有"整备："条目的英雄）"""
    def on_use(self, eng, t):
        for x in list(eng.heroes()):
            if _has_entry(x.name, "整备"):
                eng.trigger(x, "prepare")


# ============================================================ 阿科米亚核心
@card("跳级")
class 跳级:
    """使1名英雄夺取阿科米亚核心至多5级。
    被指定的英雄永久等级 + min(5, 核心等级)，核心等级减去同样数量。
    待核对：此前的说法是"核心本身只增不减"，但卡面写"夺取"—— 按夺取（核心减少）实现；核心为 0 时不能用。"""
    def on_use(self, eng, t):
        a = min(5.0, eng.s.core)
        if t is None or a <= 0:
            return False
        eng.core_add(-a, src="跳级")
        eng.level(t, a, src="跳级")


@card("助力核心")
class 助力核心:
    """阿科米亚核心等级+2，场上每有1名【日落海】英雄重复1次。（每次重复是一次独立的"提升"，海诺每次都额外 +1）"""
    def on_use(self, eng, t):
        n = sum(1 for x in eng.heroes() if eng.is_faction(x, "日落海"))
        for _ in range(1 + n):
            eng.core_add(2, src="助力核心")


# ============================================================ 装备 / 获得装备
@card("武装重构")
class 武装重构:
    """指定1名英雄，摧毁其所有装备，随机获得对应数量+1的基础装备。
    待核对：出生自带 / 无法拆卸的装备不摧毁、不计数；没有装备时也能用（得 1 件）。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        k = _destroy_equips(t)
        for _ in range(k + 1):
            eng.gain_card(eng.pick(_base_equips(), 1)[0])


@card("武装分解")
class 武装分解:
    """指定1名英雄，摧毁其所有装备：
    摧毁1件获得4能量；2件获得1张妙音；3件获得1张珍品宝箱。
    待核对：出生自带 / 无法拆卸的装备不摧毁、不计数；0 件不能用。"""
    def on_use(self, eng, t):
        if t is None:
            return False
        k = _destroy_equips(t)
        if k <= 0:
            return False
        if k == 1:
            eng.s.energy += 4
        elif k == 2:
            eng.gain_card("妙音")
        else:
            eng.gain_card("珍品宝箱")


@card("珍品宝箱")
class 珍品宝箱:
    """卡牌3选1（特殊装备或天赋装备）。选项池见 _special_gear（待核对）。"""
    def on_use(self, eng, t):
        opts = eng.pick(_special_gear(), 3)
        p = eng.choose("equip", opts)
        if p:
            eng.gain_card(p)


@card("精选武装")
class 精选武装:
    """卡牌4选1（基础装备牌）。"""
    def on_use(self, eng, t):
        p = eng.choose("equip", eng.pick(_base_equips(), 4))
        if p:
            eng.gain_card(p)


@card("军械库")
class 军械库:
    """随机获得3张基础装备牌。（每张独立随机，可重复）"""
    def on_use(self, eng, t):
        for _ in range(3):
            eng.gain_card(eng.pick(_base_equips(), 1)[0])


@card("远洋宝藏")
class 远洋宝藏:
    """从5件【日落海】专属装备中随机获得1件（不会重复获得）。（已获得的记在 eng.s.flags["远洋宝藏_got"]）"""
    def on_use(self, eng, t):
        got = eng.s.flags.setdefault("远洋宝藏_got", [])
        rest = [e for e in _sunset_gear() if e not in got]
        if not rest:
            return
        e = eng.pick(rest, 1)[0]
        got.append(e)
        eng.gain_card(e)


@card("星源装甲")
class 星源装甲:
    """【日落海】专属装备
    物理和法术吸血+20%。唯一被动：释放技能后的2秒内免疫控制。
    战斗效果：吸血 / 免控，不影响养成。"""


@card("星源狩枪")
class 星源狩枪:
    """【日落海】专属装备
    物理和法术攻击力+20%。唯一被动：法力值上限-20，且技能会直接击败生命值10%以下的敌人。
    战斗效果：攻击 / 法力 / 斩杀，不影响养成（斩杀会多击杀，击杀由引擎按实测平均抽，不单独建模）。"""


@card("星源星徽")
class 星源星徽:
    """【日落海】专属装备
    唯一被动：该英雄不会被阿科米亚核心分配等级。
    养成相关：引擎核心分配（engine.core_eligible）跳过穿戴者 —— 类属性 blocks_core。"""
    blocks_core = True


@card("星源合金盾")
class 星源合金盾:
    """【日落海】专属装备
    物理和法术防御力+20。唯一被动：受到的伤害-20%。
    战斗效果：防御 / 减伤，不影响养成。"""


@card("月影之力")
class 月影之力:
    """【日落海】专属装备
    生命值上限+15%。唯一被动：露娜释放2次技能后，将释放1次【月下无限连】（仅露娜穿戴时生效，每场战斗限1次）。
    战斗效果：生命 / 技能，不影响养成。"""


@card("度命之护")
class 度命之护:
    """获得1张特殊装备贤者的庇护。可穿戴给【河洛】英雄，穿戴者会在首次被击败的2秒后复生，仅恢复30%生命值。"""
    def on_use(self, eng, t):
        eng.gain_card("贤者的庇护")


@card("贤者的庇护")
class 贤者的庇护:
    """可穿戴给【河洛】英雄，穿戴者会在首次被击败的2秒后复生，仅恢复30%生命值。
    特殊装备（无四选一）。穿戴限制"可穿戴给【河洛】英雄"由 engine.can_wear 读卡面。
    养成相关：每场战斗多一条命（复生，影响阵亡 / 牺牲触发次数），开战时挂上，引擎战斗结束清掉。"""
    def on_battle_start(self, eng, u):
        u.counters["revive"] = u.counters.get("revive", 0) + 1


@card("机关秘典")
class 机关秘典:
    """获得1件特殊装备机关匣：可穿戴给【逐鹿】英雄，使其获得效果：战斗开始时，召唤2个"故障机器人"。"""
    def on_use(self, eng, t):
        eng.gain_card("机关匣")


@card("机关匣")
class 机关匣:
    """唯一。可穿戴给【逐鹿】英雄，使其获得效果：战斗开始时，召唤2个拥有你等级最高英雄一半等级的"故障机器人"。
    特殊装备，穿戴限制"可穿戴给【逐鹿】英雄"由 engine.can_wear 读卡面。
    战斗开始在穿戴者旁召唤 2 个故障机器人（battle_only：引擎战斗结束移除；不占人口）。
    待核对："一半等级"取整（按向下取整，至少 1）；"等级最高英雄"按永久 + 临时等级。"""
    def on_battle_start(self, eng, u):
        hs = eng.heroes()
        top = max((x.lv() for x in hs), default=1.0)
        lv = max(1.0, math.floor(top / 2))
        for _ in range(2):
            eng.summon("故障机器人", near=u, level=lv, battle_only=True)


@card("故障机器人")
class 故障机器人:
    """404 Error，出厂设置故障，无法攻击敌人。
    召唤物（机关匣开战召唤）。no_attack：引擎分配击杀时跳过它；作为单位算进"我方单位数"。"""
    no_attack = True


# ============================================================ 随机效果牌 / 战术牌
@card("发号施令")
class 发号施令:
    """随机获得2张战术牌。待核对：10 张逐鹿战术牌均匀随机（是否含高阶 / 是否按品阶加权未知）。"""
    def on_use(self, eng, t):
        for _ in range(2):
            eng.gain_card(eng.pick(_tactic_cards(), 1)[0])


@card("古币多多")
class 古币多多:
    """获得2张古币。"""
    def on_use(self, eng, t):
        eng.gain_card("古币")
        eng.gain_card("古币")


@card("神兵天降")
class 神兵天降:
    """获得4张5阶英雄兑换券。"""
    def on_use(self, eng, t):
        for _ in range(4):
            eng.gain_card("5阶英雄兑换券")


@card("呼风唤雨")
class 呼风唤雨:
    """随机获得4张5阶效果牌。
    选项池用 5 阶效果锦囊的实测出牌频率（data/engine/draw_stats.json jinnang["5"]），每张独立抽、可重复。
    待核对：真实的 5 阶效果牌池与权重。"""
    def on_use(self, eng, t):
        from agent.engine.rollout import draw_stats
        st = ((draw_stats().get("jinnang") or {}).get("5") or {}).get("cards") or {}
        if not st:
            return
        names, ws = list(st), list(st.values())
        for _ in range(4):
            eng.gain_card(eng.rng.choices(names, ws)[0])


@card("因材施教")
class 因材施教:
    """随机获得1张你场上主要阵营相关的效果牌。
    主要阵营 = 场上英雄最多的阵营（并列随机，无阵营不算）；"相关的效果牌" = 卡库里该阵营的效果牌（不含赠送牌），均匀随机。
    待核对：主要阵营的判定、选项池与权重；场上没有阵营英雄时无效果。"""
    def on_use(self, eng, t):
        g = _real_factions(eng)
        if not g:
            return
        m = max(len(v) for v in g.values())
        f = eng.pick([k for k, v in g.items() if len(v) == m], 1)[0]
        pool = [n for n, c in card_db().items()
                if c.get("category") == "效果牌" and faction(n) == f and "(赠送)" not in n]
        if pool:
            eng.gain_card(eng.pick(pool, 1)[0])


# ============================================================ 只有战斗效果的效果牌
@card("高阶援护战术")
class 高阶援护战术:
    """转瞬
    全体【逐鹿】英雄本回合获得100（+50×战术牌打出次数，当前：150）点护盾。
    战斗效果：护盾，不影响养成（作为效果牌会触发芈月等反应器）。"""
    def on_use(self, eng, t):
        return None


@card("高阶推演战术")
class 高阶推演战术:
    """转瞬
    指定1名【逐鹿】英雄，使其本回合法力值上限{推演高级战术:-10}（每次使用此牌都会提升效果，上限40点）。
    战斗效果：法力值上限，不影响养成。目标必须是【逐鹿】英雄。"""
    def on_use(self, eng, t):
        if t is None or not eng.is_faction(t, "逐鹿"):
            return False


@card("奇谋")
class 奇谋:
    """指定1名【三分】英雄，下场战斗开始时会立即释放技能。
    战斗效果：开局放技能，不影响养成。目标必须是【三分】英雄。"""
    def on_use(self, eng, t):
        if t is None or not eng.is_faction(t, "三分之地"):
            return False


@card("开局狂飙")
class 开局狂飙:
    """指定1名英雄，本回合开战后前15秒内，可以无限释放技能。
    战斗效果：无限放技能，不影响养成。"""
    def on_use(self, eng, t):
        if t is None:
            return False


# ============================================================ 接口说明
# 2026-10-14 原来列在这里的缺口已由引擎统一补上：State.core / core_add / core_distribute（星源星徽 blocks_core）、
# engine.equip / can_wear（穿戴限制读卡面）、hand_meta（众星闪耀持有回合）、summon(battle_only=True)、
# State.free_refresh_all（展翅腾飞，refresh_shop 认它）。仍未解决：破境之证"之后每回合"借隐藏持续效果
# "破境之证(持续)" 挂在 talents；移花接木用实测敌方最高等级（引擎没有敌方模型）。
