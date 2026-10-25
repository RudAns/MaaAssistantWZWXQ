"""第 4 批卡牌的独立测试：**只按 data/cards.json 卡面原文写**，没有读 agent/sim/cards/batch_04.py。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_batch_04.py
逐条打印 OK / FAIL（含原因），末尾打印通过数 / 失败数，有失败退出码 1。
没注册的卡记 FAIL "未实现"，不抛异常。

约定（照 agent/sim/play.py 的 _take）：选天赋 = 放进 s.talents，有 on_use 的再 eng.use_card(名字, None)。
"获得1张天赋牌X"给出的牌按卡名 X 或 "X(赠送)" 认，打出用 eng.use_card(X, 目标)。
测试单位尽量用 pilot / 其他批里**没有养成钩子**的"白板"英雄，避免别的卡的效果污染读数。

只有战斗效果、不影响养成的卡不写测试（列在 COMBAT_ONLY，运行时打印 SKIP）；
影响养成但现有接口搭不出局面的列在 NOT_TESTABLE（原因即"接口缺口"）。
"""
from __future__ import annotations

import contextlib
import json
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.engine import board as B                      # noqa: E402
import agent.sim.engine as E                             # noqa: E402
from agent.sim.engine import Engine, impl, card_db, quality, category, faction, card_text   # noqa: E402
import agent.sim.engine as _E                                                     # noqa: E402
_E.EQUIP_DROP_ROUNDS = ()      # 固定回合投放装备（开局 / 第 4/8/12/16 回合）是引擎规则，卡牌测试里关掉，免得混进手牌 / 装备数的断言
from agent.sim.state import State, Unit                  # noqa: E402

# ---------------------------------------------------------------- 只有战斗效果的卡（不测）
COMBAT_ONLY = [
    "逆袭之路",      # 伤害 / 生命值
    "绝对领域",      # 棋手受伤减免
    "小身体大能量",  # 攻速 / 暴击 / 生命
    "瞬影连斩",      # 闪现 / 无敌
    "坚不可摧",      # 图腾护盾
    "最终守卫",      # 无敌 / 攻速
    "膨化生命",      # 生命 / 护盾
    "控制感应",      # 法术攻击力
    "搏战",          # 回血 / 攻速
    "惊雷之龙",      # 赵云的技能
]
# 影响养成、但现有接口搭不出局面的卡（不测）
NOT_TESTABLE = {
    "荆棘牢笼": "眩晕期间敌人阵亡才召唤熊灵：引擎不模拟敌方单位",
}

RECIPES = json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8"))["recipes"]
BASIC_EQUIPS = set(RECIPES)                                # 8 件基础装备
SUNSET_EQUIPS = {n for n, c in card_db().items()
                 if c.get("category") == "装备牌" and (c.get("skill") or "").startswith("【日落海】专属装备")}
CORE_KEYS = ("core", "阿科米亚核心")                      # batch_01 / batch_00 各用了一个 key（接口缺口）


class Unimpl(Exception):
    pass


# ---------------------------------------------------------------- 搭局面
def mk(seed=0, **kw) -> Engine:
    return Engine(State(**kw), random.Random(seed))


def add(eng: Engine, name: str, cell_idx: int = 0, level: float = 1.0, temp: float = 0.0,
        kind: str = "hero", awaken: int = 0, equips=None) -> Unit:
    """直接放一个单位（不触发登场）。"""
    u = Unit(name=name, level=level, temp=temp, cell=B.CELLS[cell_idx], kind=kind,
             pop=(kind == "hero"), uid=eng.s.next_uid, awaken=awaken)
    eng.s.next_uid += 1
    if equips:
        u.equips = list(equips)
    eng.s.units.append(u)
    return u


def need(name: str):
    c = impl(name)
    if c is None:
        raise Unimpl("未实现")
    return c


def take(eng: Engine, name: str):
    """选天赋：放进 s.talents，有 on_use 的再结算一次（同 play.Player._take）。"""
    c = need(name)
    eng.s.talents.append(name)
    if hasattr(c, "on_use"):
        return eng.use_card(name, None)
    return None


def use(eng: Engine, name: str, target=None):
    need(name)
    return eng.use_card(name, target)


def cnt(eng: Engine, base: str) -> int:
    """手牌里 base 或 base(赠送) 的张数。"""
    return sum(1 for x in eng.s.hand if x == base or x.startswith(base + "("))


def gift(eng: Engine, base: str) -> str:
    """手牌里那张赠送牌的实际卡名。"""
    for x in eng.s.hand:
        if x == base or x.startswith(base + "("):
            return x
    raise AssertionError(f"手牌里没有 {base}：{eng.s.hand}")


def xp_total(s: State) -> int:
    """累计棋手经验（含已经用来升级的）。"""
    return s.xp + sum(B.XP_NEED[l] for l in range(1, s.lord_level))


def temps_since(eng: Engine, i0: int) -> dict:
    """log[i0:] 里各单位得到的临时等级总和（battle 结束会清零 temp，只能从日志读）。"""
    out = {}
    for (_, ev, kw) in eng.s.log[i0:]:
        if ev == "临时等级":
            out[kw.get("unit")] = out.get(kw.get("unit"), 0) + kw.get("n", 0)
    return out


def core_now(eng: Engine) -> float:
    return float(eng.s.core)                                # 2026-10-14 统一接口：核心等级 = State.core


def core_gain(eng: Engine, i0: int, before: float) -> float:
    """核心等级增量：先看日志 '核心等级' 事件（batch_00 / 01 都记了），没有再看 flags。"""
    g = sum(kw.get("n", 0) for (_, ev, kw) in eng.s.log[i0:] if ev == "核心等级")
    return g if g else core_now(eng) - before


@contextlib.contextmanager
def patched(**kw):
    """临时改引擎常量（BASE_WIN / DEATH_* …），强制胜负 / 阵亡。"""
    old = {k: getattr(E, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(E, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(E, k, v)


def force(win: bool):
    return patched(BASE_WIN=1.0 if win else 0.0)


def recorder(eng: Engine, pick=-1):
    calls = []

    def ch(e, kind, options, **ctx):
        calls.append((kind, list(options)))
        return options[pick]
    eng.chooser = ch
    return calls


def hero(name):
    assert category(name) == "英雄牌", f"{name} 不是英雄牌（测试数据有误）"
    return name


# 白板英雄（pilot / 其他批没有养成钩子）
HL_A, HL_B, HL_C = hero("沈梦溪"), hero("百里守约"), hero("狄仁杰")       # 河洛（守约自带牺牲关键词）
ZL_A, ZL_B = hero("墨子"), hero("廉颇")                                  # 逐鹿 2 阶
SF_A, SF_B, SF_C = hero("大乔"), hero("马超"), hero("吕布")              # 三分
WU_A, WU_B = hero("苍"), hero("海月")                                    # 无阵营
HI45 = [hero("扁鹊"), hero("嬴政"), hero("艾琳"), hero("吕布"), hero("海月")]   # 4~5 阶白板


# ================================================================ 棋手经验类
def t_三十而达():
    """备战阶段中，英雄总等级到达30级时，获得9棋手经验。（按一次性读）"""
    eng = mk(0)
    a = add(eng, HL_A, 0, level=10)
    add(eng, ZL_A, 1, level=10)
    take(eng, "三十而达")
    x0 = xp_total(eng.s)
    eng.round_start()
    assert xp_total(eng.s) == x0, f"总等级 20 时就给了经验 {xp_total(eng.s) - x0}"
    a.level = 20                                           # 总等级 30
    eng.round_start()
    assert xp_total(eng.s) - x0 == 9, f"总等级到 30 后经验 +{xp_total(eng.s) - x0}，应 +9"
    eng.round_start()
    assert xp_total(eng.s) - x0 == 9, f"再过一个备战阶段经验累计 +{xp_total(eng.s) - x0}，'到达'应只给一次"


def t_卧薪尝胆():
    """选取天赋后，当你输掉2场战斗时，获得9棋手经验。（累计输 2 场，不要求连续）"""
    eng = mk(0)
    add(eng, HL_A, 0)
    with force(False):
        eng.battle()                                       # 选天赋之前输的不算
    take(eng, "卧薪尝胆")
    x0 = xp_total(eng.s)
    with force(False):
        eng.battle()
    assert xp_total(eng.s) == x0, "选取后才输 1 场就给了经验"
    with force(True):
        eng.battle()
    with force(False):
        eng.battle()
    assert xp_total(eng.s) - x0 == 9, f"选取后累计输 2 场，经验 +{xp_total(eng.s) - x0}，应 +9"


def t_保持进步():
    """获得1张3阶兑换券。回合开始时，获得1棋手经验。满级后，改为获得1能量。"""
    eng = mk(0)
    add(eng, HL_A, 0)
    take(eng, "保持进步")
    n = cnt(eng, "3阶英雄兑换券") + cnt(eng, "3阶兑换券")
    assert n == 1 and len(eng.s.hand) == 1, f"手牌 {eng.s.hand}，应只有 1 张 3阶兑换券"
    x0, e0 = xp_total(eng.s), eng.s.energy
    eng.round_start()
    assert xp_total(eng.s) - x0 == 1, f"回合开始经验 +{xp_total(eng.s) - x0}，应 +1"
    assert eng.s.energy == e0, f"没满级却给了能量 {eng.s.energy - e0}"
    eng = mk(0, lord_level=B.MAX_LORD_LEVEL)
    take(eng, "保持进步")
    x0, e0 = eng.s.xp, eng.s.energy
    eng.round_start()
    assert eng.s.energy - e0 == 1, f"满级回合开始能量 +{eng.s.energy - e0}，应 +1"
    assert eng.s.xp == x0, f"满级还在加经验 {eng.s.xp - x0}"


def t_人才济济():
    """升级所需棋手经验-2，每次升级时获得1张天赋牌人才搜寻。"""
    eng = mk(0)
    take(eng, "人才济济")
    assert cnt(eng, "人才搜寻") == 0, f"选天赋时就给了人才搜寻 {eng.s.hand}"
    eng.s.xp += 5                                          # 1→2 原需 7，-2 = 5
    eng.level_ups()
    assert eng.s.lord_level == 2, f"攒 5 经验没升到 2 级（lord_level={eng.s.lord_level} xp={eng.s.xp}）—— '-2' 没生效"
    assert cnt(eng, "人才搜寻") == 1, f"升 1 级应得 1 张人才搜寻，手牌 {eng.s.hand}"
    eng = mk(1)
    take(eng, "人才济济")
    eng.s.xp += 40
    eng.level_ups()
    n = eng.s.lord_level - 1
    assert n >= 2 and cnt(eng, "人才搜寻") == n, f"升了 {n} 级，人才搜寻 {cnt(eng, '人才搜寻')} 张"


def t_祝你好运():
    """棋手升级时，获得1张天赋牌碰碰运气。"""
    eng = mk(0)
    take(eng, "祝你好运")
    assert cnt(eng, "碰碰运气") == 0, f"选天赋时就给了 {eng.s.hand}"
    eng.s.xp += 7
    eng.level_ups()
    assert eng.s.lord_level == 2 and cnt(eng, "碰碰运气") == 1, f"升 1 级后手牌 {eng.s.hand}"
    eng.s.xp += 13
    eng.level_ups()
    assert cnt(eng, "碰碰运气") == 2, f"升 2 级后手牌 {eng.s.hand}"


def t_碰碰运气():
    """消耗0能量，指定英雄使其随机提升5-20级。（祝你好运卡面说"随机消耗1-5能量"，两处冲突：能量只断言消耗 0~5）"""
    gains = []
    for seed in range(60):
        eng = mk(seed, energy=10)
        u = add(eng, HL_A, 0, level=3)
        ok = use(eng, "碰碰运气", u)
        assert ok is not False, "指定了英雄却返回 False"
        d = u.level - 3
        assert 5 <= d <= 20 and d == int(d), f"seed{seed} 提升 {d} 级（应 5~20 整数）"
        de = 10 - eng.s.energy
        assert 0 <= de <= 5, f"seed{seed} 能量变化 -{de}（卡面 0，或祝你好运说的 1~5）"
        gains.append(d)
    assert min(gains) < max(gains), f"60 次都是 {gains[0]}，不像随机"
    assert use(mk(0), "碰碰运气", None) is False, "没有目标应返回 False"


def t_重大里程碑():
    """棋手到达5级时，获得8能量、1张装备箱和1张高阶效果牌。（"高阶"按品阶 ≥3 的效果牌断言）"""
    eng = mk(0, lord_level=3)
    take(eng, "重大里程碑")
    e0 = eng.s.energy
    assert not eng.s.hand and eng.s.energy == e0, "选天赋时就给了东西"
    eng.s.xp += B.XP_NEED[3]
    eng.level_ups()
    assert eng.s.lord_level == 4 and not eng.s.hand and eng.s.energy == e0, f"到 4 级就触发了：{eng.s.hand}"
    eng.s.xp += B.XP_NEED[4]
    eng.level_ups()
    assert eng.s.lord_level == 5
    assert eng.s.energy - e0 == 8, f"到 5 级能量 +{eng.s.energy - e0}，应 +8"
    h = list(eng.s.hand)
    assert len(h) == 2 and h.count("装备箱") == 1, f"手牌 {h}，应是 装备箱 + 1 张高阶效果牌"
    other = [x for x in h if x != "装备箱"][0]
    assert category(other) == "效果牌" and quality(other) >= 3, f"另一张 {other}（{category(other)} {quality(other)}阶）不像高阶效果牌"


# ================================================================ 天赋牌（拿到给牌）
def t_超速成长():
    """获得1张天赋牌超速成长：指定1名英雄等级+15。"""
    eng = mk(0)
    u = add(eng, HL_A, 0, level=4)
    take(eng, "超速成长")
    assert cnt(eng, "超速成长") == 1 and len(eng.s.hand) == 1, f"手牌 {eng.s.hand}，应 1 张超速成长"
    assert u.level == 4, f"选天赋时就加了等级 {u.level}"
    ok = use(eng, gift(eng, "超速成长"), u)
    assert ok is not False and u.level == 19, f"打出后等级 {u.level}，应 4+15=19"


def _visitor(talent, card_name):
    def f():
        eng = mk(0, energy=4)
        take(eng, talent)
        assert eng.s.energy == 7, f"能量 {eng.s.energy}，应 4+3=7"
        assert eng.s.hand == [card_name], f"手牌 {eng.s.hand}，应是 [{card_name}]"
    f.__doc__ = f"获得3能量以及1张{card_name}。"
    return f


def _give_one(talent, card_name):
    def f():
        eng = mk(0)
        add(eng, HL_A, 0)
        take(eng, talent)
        assert cnt(eng, card_name) == 1 and len(eng.s.hand) == 1, f"手牌 {eng.s.hand}，应只有 1 张 {card_name}"
    f.__doc__ = f"获得1张{card_name}（其余是战斗效果）。"
    return f


def _elite(talent, fac):
    def f():
        for seed in range(8):
            eng = mk(seed, lord_level=2)
            calls = recorder(eng, pick=-1)
            take(eng, talent)
            assert len(calls) == 1, f"seed{seed} choose 被调用 {len(calls)} 次，应 1 次（3选1）"
            opts = calls[0][1]
            assert len(opts) == 3 and len(set(opts)) == 3, f"seed{seed} 选项 {opts}，应 3 张不同的牌"
            for o in opts:
                assert category(o) == "英雄牌" and quality(o) in (4, 5) and faction(o) == E.FACTION_ALIAS.get(fac, fac), \
                    f"seed{seed} 选项 {o}（{category(o)} {quality(o)}阶 {faction(o)}）不是 4~5 阶【{fac}】英雄"
            assert eng.s.hand == [opts[-1]], f"seed{seed} 选了 {opts[-1]}，手牌 {eng.s.hand}"
    f.__doc__ = f"卡牌3选1（4阶或5阶的【{fac}】英雄）。"
    return f


def t_紧急行动():
    """获得10点棋手生命值、1张装备箱。（生命值 State 没有，不测）"""
    eng = mk(0)
    take(eng, "紧急行动")
    assert eng.s.hand == ["装备箱"], f"手牌 {eng.s.hand}"


def t_富贵险中求():
    """获得1张天赋牌一刀穷一刀富。"""
    eng = mk(0)
    take(eng, "富贵险中求")
    assert cnt(eng, "一刀穷一刀富") == 1 and len(eng.s.hand) == 1, f"手牌 {eng.s.hand}"


def _xzqf_present(eng):
    return cnt(eng, "险中求富") + eng.s.talents.count("险中求富") + eng.s.lord_cards.count("险中求富")


def t_一刀穷一刀富():
    """获得1张基础装备牌及1张险中求富并开始记录胜场。"""
    for seed in range(5):
        eng = mk(seed)
        add(eng, HL_A, 0)
        use(eng, "一刀穷一刀富")
        basics = [x for x in eng.s.hand if x in BASIC_EQUIPS]
        assert len(basics) == 1, f"seed{seed} 基础装备 {basics}，手牌 {eng.s.hand}"
        assert _xzqf_present(eng) == 1, f"seed{seed} 没拿到险中求富：手牌 {eng.s.hand} 天赋 {eng.s.talents}"


def t_险中求富():
    """获得基础装备牌（当前:N件，持有时每胜 1 场累积 1 件）。战斗失败销毁此牌并获得2能量。"""
    eng = mk(0)
    add(eng, HL_A, 0)
    use(eng, "一刀穷一刀富")
    b0 = sum(1 for x in eng.s.hand if x in BASIC_EQUIPS)
    with force(True):
        eng.battle()
        eng.battle()
    assert sum(1 for x in eng.s.hand if x in BASIC_EQUIPS) == b0, "还没打出就给了基础装备"
    ok = use(eng, "险中求富")
    assert ok is not False, "打出险中求富返回 False"
    got = sum(1 for x in eng.s.hand if x in BASIC_EQUIPS) - b0
    assert got == 2, f"赢 2 场后打出，得到基础装备 {got} 件，应 2 件"
    # 失败：销毁 + 2 能量，累积的作废
    eng = mk(0)
    add(eng, HL_A, 0)
    use(eng, "一刀穷一刀富")
    b0 = sum(1 for x in eng.s.hand if x in BASIC_EQUIPS)
    with force(True):
        eng.battle()
    e0 = eng.s.energy
    with force(False):
        eng.battle()
    assert eng.s.energy - e0 == 2, f"失败后能量 +{eng.s.energy - e0}，应 +2"
    assert _xzqf_present(eng) == 0, f"失败后险中求富还在：手牌 {eng.s.hand} 天赋 {eng.s.talents}"
    assert sum(1 for x in eng.s.hand if x in BASIC_EQUIPS) == b0, "失败后不该再给累积的装备"


def t_前浪的荣光():
    """获得2张天赋牌前浪的荣光：指定1名英雄，获得1张高1-2阶的同阵营英雄牌。"""
    eng = mk(0)
    take(eng, "前浪的荣光")
    assert cnt(eng, "前浪的荣光") == 2 and len(eng.s.hand) == 2, f"手牌 {eng.s.hand}，应 2 张"
    tiers = set()
    for seed in range(30):
        for src in (HL_A, ZL_A):                            # 2 阶 河洛 / 逐鹿
            eng = mk(seed)
            u = add(eng, src, 0)
            ok = use(eng, "前浪的荣光", u)
            assert ok is not False, "指定英雄却返回 False"
            new = [x for x in eng.s.hand if category(x) == "英雄牌"]
            assert len(new) == 1, f"seed{seed} 应得 1 张英雄牌，手牌 {eng.s.hand}"
            n = new[0]
            assert faction(n) == faction(src), f"{n}（{faction(n)}）不是 {faction(src)}"
            assert quality(n) in (3, 4), f"{src}(2阶) 得到 {n}（{quality(n)}阶），应 3~4 阶"
            tiers.add(quality(n))
    assert tiers == {3, 4}, f"30 个种子只出过 {tiers} 阶"


def t_全军列阵():
    """获得1张天赋牌全军列阵：指定1名英雄，召唤其同阵营英雄上场直到阵容人数上限（不超过商店可购品阶）。"""
    eng = mk(0, lord_level=3)
    take(eng, "全军列阵")
    assert cnt(eng, "全军列阵") == 1 and len(eng.s.hand) == 1, f"手牌 {eng.s.hand}"
    maxq = max(B.meta.TIER_ODDS[3])
    for seed in range(6):
        eng = mk(seed, lord_level=3)
        u = add(eng, HL_A, 0)
        add(eng, ZL_A, 1)
        before = list(eng.s.units)
        ok = use(eng, "全军列阵", u)
        assert ok is not False, "指定英雄却返回 False"
        cap = eng.pop_cap()
        assert eng.s.pop_used() == cap, f"seed{seed} 人口 {eng.s.pop_used()}/{cap}，应填满"
        new = [x for x in eng.s.units if x not in before]
        assert len(new) == cap - 2, f"seed{seed} 新上 {len(new)} 人，应 {cap - 2}"
        for x in new:
            assert x.kind == "hero" and category(x.name) == "英雄牌", f"{x.name} 不是英雄"
            assert faction(x.name) == "河洛", f"{x.name}（{faction(x.name)}）不是河洛"
            assert quality(x.name) <= maxq, f"{x.name} {quality(x.name)}阶，3 级商店最高 {maxq} 阶"


def t_胜之不武():
    """获得3张天赋牌胜之不武：下场战斗你的英雄+20临时等级。"""
    eng = mk(0)
    take(eng, "胜之不武")
    assert cnt(eng, "胜之不武") == 3 and len(eng.s.hand) == 3, f"手牌 {eng.s.hand}，应 3 张"
    eng = mk(0)
    a, b = add(eng, HL_A, 0), add(eng, ZL_A, 1)
    tot = add(eng, "因缘图腾", 6, kind="summon")
    ok = use(eng, "胜之不武")
    assert ok is not False, "打出返回 False"
    i0 = len(eng.s.log)
    a_pre, b_pre = a.temp, b.temp
    eng.battle()
    got = temps_since(eng, i0)
    ga, gb = got.get(a.name, 0) + a_pre, got.get(b.name, 0) + b_pre
    assert ga == 20 and gb == 20, f"下场战斗临时等级 {a.name}={ga} {b.name}={gb}，应各 20"
    assert got.get(tot.name, 0) == 0, "召唤物也拿了临时等级"
    i1 = len(eng.s.log)
    eng.round_start()
    eng.battle()
    got = temps_since(eng, i1)
    assert got.get(a.name, 0) == 0, f"再下一场还在给 {got}"


def t_一分钱一分货():
    """获得1张天赋牌等价交换。"""
    eng = mk(0, energy=5)
    take(eng, "一分钱一分货")
    assert cnt(eng, "等价交换") == 1 and len(eng.s.hand) == 1, f"手牌 {eng.s.hand}"


def t_等价交换():
    """扣除所有能量：＜2，获得2张英雄牌；2-4，获得1张基础装备牌；≥5，获得1张5阶兑换券，2张效果锦囊牌。"""
    for e in (0, 1):
        eng = mk(e, energy=e)
        use(eng, "等价交换")
        h = eng.s.hand
        assert eng.s.energy == 0, f"能量 {e} 用后剩 {eng.s.energy}"
        assert len(h) == 2 and all(category(x) == "英雄牌" for x in h), f"能量 {e}：手牌 {h}，应 2 张英雄牌"
    for e in (2, 3, 4):
        eng = mk(e, energy=e)
        use(eng, "等价交换")
        h = eng.s.hand
        assert eng.s.energy == 0, f"能量 {e} 用后剩 {eng.s.energy}"
        assert len(h) == 1 and h[0] in BASIC_EQUIPS, f"能量 {e}：手牌 {h}，应 1 张基础装备"
    for e in (5, 12):
        eng = mk(e, energy=e)
        use(eng, "等价交换")
        h = eng.s.hand
        assert eng.s.energy == 0, f"能量 {e} 用后剩 {eng.s.energy}"
        q = [x for x in h if x in ("5阶英雄兑换券", "5阶兑换券")]
        j = [x for x in h if x.endswith("效果锦囊")]
        assert len(h) == 3 and len(q) == 1 and len(j) == 2, f"能量 {e}：手牌 {h}，应 1 张5阶兑换券 + 2 张效果锦囊"


def t_锦囊妙计():
    """获得1张锦囊。"""
    eng = mk(0)
    take(eng, "锦囊妙计")
    assert cnt(eng, "锦囊") == 1 and len(eng.s.hand) == 1, f"手牌 {eng.s.hand}"


def t_锦囊():
    """使用后，本回合打出的登场英雄牌，会额外触发1次登场效果。"""
    eng = mk(0)
    add(eng, HL_A, 0)
    ok = use(eng, "锦囊")
    assert ok is not False, "打出返回 False"
    z = eng.deploy(hero("庄周"))                            # 登场：自身等级+2并获得1张招募战术
    assert z is not None
    assert z.level == 5, f"庄周等级 {z.level}，登场触发 2 次应 1+2+2=5"
    assert cnt(eng, "招募战术") == 2, f"招募战术 {cnt(eng, '招募战术')} 张，应 2"
    eng.round_end()
    eng.round_start()
    eng.deploy(hero("蒙犽"))                                # 登场：获得1张特训战术
    assert cnt(eng, "特训战术") == 1, f"下一回合还在额外触发（特训战术 {cnt(eng, '特训战术')} 张）"


def t_挂帅上阵():
    """获得1张挂帅。"""
    eng = mk(0)
    take(eng, "挂帅上阵")
    assert cnt(eng, "挂帅") == 1 and len(eng.s.hand) == 1, f"手牌 {eng.s.hand}"


def t_挂帅():
    """指定1名【三分】英雄，使其等级等同于1名最近的【三分】英雄。"""
    eng = mk(0)
    t = add(eng, SF_A, 0, level=3)                          # (319,375)
    add(eng, SF_B, 1, level=20)                             # 相邻
    add(eng, SF_C, 6, level=50)                             # 同排最远
    ok = use(eng, "挂帅", t)
    assert ok is not False and t.level == 20, f"大乔等级 {t.level}，应等同最近的马超 20"
    eng = mk(0)
    h = add(eng, HL_A, 0, level=3)
    add(eng, SF_B, 1, level=20)
    ok = use(eng, "挂帅", h)
    assert ok is False and h.level == 3, "非三分英雄不该能被挂帅"


# ================================================================ 能量 / 出售
def t_伤害转化器():
    """获得2能量。战斗胜利时，再获得2能量。"""
    eng = mk(0, energy=3)
    add(eng, HL_A, 0)
    take(eng, "伤害转化器")
    assert eng.s.energy == 5, f"能量 {eng.s.energy}，应 3+2"
    with force(True):
        eng.battle()
    assert eng.s.energy == 7, f"胜利后能量 {eng.s.energy}，应 +2"
    with force(False):
        eng.battle()
    assert eng.s.energy == 7, f"失败也加了能量 {eng.s.energy}"
    with force(True):
        eng.battle()
    assert eng.s.energy == 9, f"再胜一场能量 {eng.s.energy}，应 9"


def t_大买卖():
    """出售英雄牌额外获得1能量。（商店购买 +1 能量：引擎没有购买接口，不测；出售按卖场上英雄测）"""
    eng = mk(0, energy=3)
    u = add(eng, HL_A, 0)
    take(eng, "大买卖")
    e0 = eng.s.energy
    eng.sell(u)
    assert eng.s.energy - e0 == 2, f"出售得 {eng.s.energy - e0} 能量，应 1+1=2"


# ================================================================ 等级 / 临时等级
def t_临时增幅():
    """战斗开始时，将20临时等级随机分配给己方英雄。每回合增加5临时等级。"""
    for seed in range(5):
        eng = mk(seed)
        hs = [add(eng, HL_A, 0), add(eng, ZL_A, 1), add(eng, SF_A, 2)]
        tot = add(eng, "因缘图腾", 6, kind="summon")
        take(eng, "临时增幅")
        i0 = len(eng.s.log)
        eng.battle()
        got = temps_since(eng, i0)
        s1 = sum(got.get(h.name, 0) for h in hs)
        assert s1 == 20, f"seed{seed} 第 1 场分了 {s1}，应 20（{got}）"
        assert got.get(tot.name, 0) == 0, "召唤物也分到了"
        eng.round_end()
        eng.round_start()
        i1 = len(eng.s.log)
        eng.battle()
        got = temps_since(eng, i1)
        s2 = sum(got.get(h.name, 0) for h in hs)
        assert s2 == 25, f"seed{seed} 下一回合分了 {s2}，应 25"


def t_协同一致():
    """回合开始时，无阵营英雄获得等于当前棋手等级的等级。"""
    eng = mk(0, lord_level=3)
    a, b = add(eng, WU_A, 0, level=2), add(eng, WU_B, 1, level=5)
    h = add(eng, HL_A, 2, level=4)
    assert faction(WU_A) == "无阵营" and faction(WU_B) == "无阵营"
    take(eng, "协同一致")
    assert (a.level, b.level, h.level) == (2, 5, 4), "选天赋时就加了等级"
    eng.round_start()
    assert a.level == 5 and b.level == 8, f"无阵营 {a.level}/{b.level}，应 +3（棋手 3 级）"
    assert h.level == 4, f"河洛英雄也加了 {h.level}"


def t_强力小弟():
    """你的召唤单位等级+5。每回合开始时，再+5。（按"选取时现有召唤物 +5，之后每回合开始 +5"读）"""
    eng = mk(0)
    s = add(eng, "因缘图腾", 6, kind="summon", level=1)
    h = add(eng, HL_A, 0, level=1)
    take(eng, "强力小弟")
    eng.round_start()
    assert s.level == 11, f"召唤物等级 {s.level}，应 1+5+5=11"
    assert h.level == 1, f"英雄也加了等级 {h.level}"
    eng.round_start()
    assert s.level == 16, f"第二个回合开始后 {s.level}，应 16"


def t_三三得三():
    """战斗开始时，若你场上【三分】英雄人数刚好是3的倍数，他们分别获得33临时等级。"""
    eng = mk(0)
    sf = [add(eng, SF_A, 0), add(eng, SF_B, 1), add(eng, SF_C, 2)]
    o = add(eng, ZL_A, 3)
    take(eng, "三三得三")
    i0 = len(eng.s.log)
    eng.battle()
    got = temps_since(eng, i0)
    assert all(got.get(x.name, 0) == 33 for x in sf), f"3 名三分各应 +33：{got}"
    assert got.get(o.name, 0) == 0, f"非三分也拿了 {got}"
    eng = mk(0)
    add(eng, SF_A, 0), add(eng, SF_B, 1)
    add(eng, ZL_A, 3)
    take(eng, "三三得三")
    i0 = len(eng.s.log)
    eng.battle()
    got = temps_since(eng, i0)
    assert not any(got.values()), f"只有 2 名三分却给了 {got}"


def t_乐善好施():
    """你的英雄触发登场效果时，随机2名【三分】英雄等级+2。"""
    for seed in range(6):
        eng = mk(seed)
        sf = [add(eng, SF_A, 0, level=1), add(eng, SF_B, 1, level=1), add(eng, SF_C, 2, level=1)]
        take(eng, "乐善好施")
        eng.deploy(hero("墨子"))                           # 没有登场效果
        assert all(x.level == 1 for x in sf), f"seed{seed} 墨子没有登场效果却触发了"
        eng.deploy(hero("庄周"))                           # 登场：自身等级+2…
        ups = sorted(x.level - 1 for x in sf)
        assert ups == [0, 2, 2], f"seed{seed} 三分英雄加级 {ups}，应恰好 2 名 +2"


def t_献祭之赐():
    """获得1张李信，李信卡牌效果提供的临时等级额外+3。（敌方牺牲那半句引擎不模拟敌方，不测）"""
    eng = mk(0)
    take(eng, "献祭之赐")
    assert eng.s.hand == ["李信"], f"手牌 {eng.s.hand}"
    for with_t, exp in ((False, 7), (True, 10)):
        eng = mk(0)
        lx = add(eng, hero("李信"), 0)
        su = add(eng, hero("苏烈"), 1)                     # 苏烈有牺牲
        if with_t:
            take(eng, "献祭之赐")
        eng.fire(su, "death")
        eng.fire(su, "sacrifice")
        assert lx.temp == exp, f"{'有' if with_t else '没有'}天赋：李信临时等级 {lx.temp}，应 {exp}" + \
            ("（基线不对，是李信的实现问题）" if not with_t else "")


def t_钱生战力():
    """战斗开始时，随机4名【河洛】英雄获得累计使用古币数（选择此天赋后）的临时等级。"""
    for seed in range(4):
        eng = mk(seed)
        hl = [add(eng, HL_A, 0), add(eng, HL_B, 1), add(eng, HL_C, 2),
              add(eng, hero("程咬金"), 3), add(eng, hero("李元芳"), 4)]
        o = add(eng, ZL_A, 5)
        eng.use_card("古币", hl[0])                         # 选天赋之前的不算
        take(eng, "钱生战力")
        for _ in range(3):
            eng.use_card("古币", hl[0])
        i0 = len(eng.s.log)
        eng.battle()
        got = temps_since(eng, i0)
        vals = sorted(got.get(x.name, 0) for x in hl)
        assert vals == [0, 3, 3, 3, 3], f"seed{seed} 河洛临时等级 {vals}，应 4 名各 +3"
        assert got.get(o.name, 0) == 0, "非河洛也拿了"


# ================================================================ 古币类
def t_多彩古币():
    """获得1张古币。古币只能对非【河洛】英雄主动使用：随机变成其他同阶非【河洛】英雄，等级额外+7。
    "额外+7"：可能是古币本来的 +2 再 +7（=+9），也可能只 +7 —— 两种都接受。"""
    eng = mk(0)
    take(eng, "多彩古币")
    assert eng.s.hand == ["古币"], f"手牌 {eng.s.hand}"
    seen = set()
    for seed in range(20):
        eng = mk(seed)
        take(eng, "多彩古币")
        u = add(eng, ZL_A, 3, level=5)                      # 墨子 2 阶 逐鹿
        cell = u.cell
        i0 = len(eng.s.log)
        ok = eng.use_card("古币", u)
        assert ok is not False, "对非河洛英雄使用古币返回 False"
        nu = next((x for x in eng.s.units if x.cell == cell), None)
        assert nu is not None, "原格子上没人了"
        assert category(nu.name) == "英雄牌" and quality(nu.name) == 2, f"变成 {nu.name}（{quality(nu.name)}阶），应同阶 2 阶"
        assert nu.name != ZL_A, "没有变（还是墨子）"
        assert faction(nu.name) != "河洛", f"变成了河洛英雄 {nu.name}"
        # 只数古币 / 多彩古币给的等级（变出来的英雄自己"每使用效果牌"之类的反应另算）
        own = sum(kw.get("n", 0) for (_, ev, kw) in eng.s.log[i0:]
                  if ev == "等级" and kw.get("src") in ("古币", "多彩古币") and kw.get("unit") in (ZL_A, nu.name))
        assert own in (7, 9), f"变成 {nu.name}，古币 / 多彩古币给了 {own} 级（总 5→{nu.level}），应 +7 或 +9"
        seen.add(nu.name)
    assert len(seen) >= 2, f"20 个种子只变出 {seen}，不像随机"
    eng = mk(0)
    take(eng, "多彩古币")
    h = add(eng, HL_A, 0, level=5)
    ok = eng.use_card("古币", h)
    assert ok is False and h.level == 5, "有多彩古币时古币不能主动用在河洛英雄身上"


def t_传家宝():
    """古币不再附带转瞬。每回合开始时，手牌中古币等级+1，若数量≥8张，则+2级。（古币本身 +2）"""
    eng = mk(0)
    h = add(eng, HL_A, 0, level=1)
    take(eng, "传家宝")
    eng.s.hand += ["古币"]
    eng.round_start()
    eng.use_card("古币", h)
    assert h.level == 1 + 3, f"持有 1 回合的古币给了 {h.level - 1} 级，应 2+1=3"
    eng = mk(0)
    h = add(eng, HL_A, 0, level=1)
    take(eng, "传家宝")
    eng.s.hand += ["古币"] * 8
    eng.round_start()
    eng.round_start()
    eng.use_card("古币", h)
    assert h.level == 1 + 6, f"手里 8 张、持有 2 回合的古币给了 {h.level - 1} 级，应 2+2+2=6"


def t_古币翻新():
    """获得1张古币。古币牌会额外提供2级。"""
    eng = mk(0)
    take(eng, "古币翻新")
    assert eng.s.hand == ["古币"], f"手牌 {eng.s.hand}"
    h = add(eng, HL_A, 0, level=1)
    eng.use_card("古币", h)
    assert h.level == 1 + 4, f"古币给了 {h.level - 1} 级，应 2+2=4"


def t_零钱兑换():
    """你获得的古币会自动兑换为2张无法提供等级的破损古币，精致古币则兑换为4张。"""
    eng = mk(0)
    take(eng, "零钱兑换")
    eng.gain_card("古币")
    assert cnt(eng, "古币") == 0 and cnt(eng, "破损古币") == 2, f"获得古币后手牌 {eng.s.hand}，应 2 张破损古币"
    eng.gain_card("精致古币")
    assert cnt(eng, "精致古币") == 0 and cnt(eng, "破损古币") == 6, f"再获得精致古币后手牌 {eng.s.hand}，应共 6 张破损古币"


def t_破损古币():
    """转瞬 指定1名【河洛】英雄使用，无法提供等级。"""
    eng = mk(0)
    h = add(eng, HL_A, 0, level=3)
    ok = use(eng, "破损古币", h)
    assert ok is not False, "对河洛英雄使用返回 False"
    assert h.level == 3, f"破损古币提供了等级 {h.level - 3}"
    z = add(eng, ZL_A, 1, level=3)
    assert use(eng, "破损古币", z) is False and z.level == 3, "非河洛英雄不该能用"


def t_古币契约():
    """你的英雄每触发3次牺牲效果，获得1张古币。"""
    eng = mk(0)
    su = add(eng, hero("苏烈"), 0)
    take(eng, "古币契约")
    for i in range(1, 7):
        eng.fire(su, "sacrifice")
        exp = i // 3
        assert cnt(eng, "古币") == exp, f"触发 {i} 次牺牲后古币 {cnt(eng, '古币')} 张，应 {exp}"


def t_幸运硬币():
    """打出古币时有概率额外触发1-3次。
    卡面没写概率（实现留 PROB 参数，None = 不触发、不硬塞一个数；待核对）。
    这里测机制：概率未定时临时设 0.5，要求有时额外触发、有时不触发，额外 1~3 次。"""
    c = need("幸运硬币")
    prob = getattr(c, "PROB", "无")
    old = prob
    if prob is None:
        c.PROB = 0.5
    try:
        ds = []
        for seed in range(300):
            eng = mk(seed)
            h = add(eng, HL_A, 0, level=1)
            take(eng, "幸运硬币")
            eng.use_card("古币", h)
            d = h.level - 1
            assert d in (2, 4, 6, 8), f"seed{seed} 古币给了 {d} 级，应 2×(1+0~3)"
            ds.append(d)
        assert max(ds) > 2, "300 次一次都没额外触发"
        assert min(ds) == 2, "300 次每次都额外触发（卡面是'有概率'）"
        assert {4, 6, 8} <= set(ds), f"额外次数只出现过 {sorted(set(ds))}，应 1~3 次都有"
        # 被动使用（直接 eng.coin，如李元芳）不算"打出"
        eng = mk(0)
        h = add(eng, HL_A, 0, level=1)
        take(eng, "幸运硬币")
        if prob is None:
            c.PROB = 1.0
        ly = add(eng, hero("李元芳"), 1)                    # 整备：对最近的河洛英雄使用 1 张古币（被动）
        eng.fire(ly, "prepare")
        assert h.level == 3, f"被动使用古币（李元芳整备）也额外触发了（{h.level - 1} 级）"
    finally:
        if old is None:
            c.PROB = None


# ================================================================ 召唤 / 图腾 / 核心
def t_因缘际会():
    """召唤1个1级的因缘图腾。"""
    eng = mk(0)
    add(eng, HL_A, 0)
    take(eng, "因缘际会")
    ts = [u for u in eng.s.units if u.name == "因缘图腾"]
    assert len(ts) == 1, f"场上因缘图腾 {len(ts)} 个"
    assert ts[0].level == 1 and ts[0].kind != "hero" and not ts[0].pop, \
        f"因缘图腾 level={ts[0].level} kind={ts[0].kind} pop={ts[0].pop}，应 1 级召唤物、不占人口"


def t_图腾启灵():
    """每回合随机召唤1个图腾，直到回合结束。"""
    names = set()
    for seed in range(12):
        eng = mk(seed)
        add(eng, HL_A, 0)
        take(eng, "图腾启灵")
        for r in range(3):
            eng.round_start()
            ts = [u for u in eng.s.units if "图腾" in u.name]
            assert len(ts) == 1, f"seed{seed} 第{r + 1}回合场上图腾 {[t.name for t in ts]}，应 1 个（上回合的应已消失）"
            assert ts[0].kind != "hero", f"{ts[0].name} 是 {ts[0].kind}，应是召唤物"
            assert category(ts[0].name) == "召唤物", f"{ts[0].name} 不是卡库里的召唤物"
            names.add(ts[0].name)
            eng.battle()
            eng.round_end()
    assert len(names) >= 2, f"12 局只召唤过 {names}，不像随机"


def t_化为熊灵():
    """战斗中首名【大河】英雄阵亡时，召唤1只场上大河英雄等级之和的熊灵（不计算临时等级），并嘲讽周围2格敌人，持续4秒。
    读法（与实现统一）：熊灵是战斗中召唤的，战后不留在棋盘上（棋盘战后复原；若常驻每场都会多一只）
    → 只影响战斗。这里断言：大河英雄阵亡的战斗之后，场上没有多出熊灵、英雄等级不变。"""
    hit = 0
    for seed in range(30):
        eng = mk(seed)
        y = add(eng, hero("瑶"), 0, level=10, temp=5)
        lb = add(eng, hero("刘邦"), 1, level=7)
        o = add(eng, HL_A, 2, level=30)                     # 非大河，不计入
        take(eng, "化为熊灵")
        deaths = []
        orig = eng.fire

        def rec(u, timing, **kw):
            if timing == "death":
                deaths.append(u)
            return orig(u, timing, **kw)
        eng.fire = rec
        before = set(id(u) for u in eng.s.units)
        lv0 = [(u.name, u.level) for u in eng.s.units]
        with patched(DEATH_WIN=1.0, DEATH_LOSE=1.0, FRONT_MULT=0.6, BACK_MULT=0.6):
            eng.battle()
        if any(u in (y, lb) for u in deaths):
            hit += 1
        new = [u.name for u in eng.s.units if id(u) not in before]
        assert not new, f"seed{seed} 战后场上多了 {new}（战斗中召唤的熊灵不该留下）"
        assert [(u.name, u.level) for u in eng.s.units] == lv0, f"seed{seed} 等级变了"
    assert hit >= 5, f"30 个种子只有 {hit} 次大河阵亡（测试本身需调参数）"


def t_场外供能():
    """回合开始时，阿科米亚核心获得棋手等级×3的等级。"""
    eng = mk(0, lord_level=4)
    add(eng, HL_A, 0)
    take(eng, "场外供能")
    c0, i0 = core_now(eng), len(eng.s.log)
    eng.round_start()
    g = core_gain(eng, i0, c0)
    assert g == 12, f"核心 +{g}，应 4×3=12（核心等级读 log '核心等级' 或 flags {CORE_KEYS}）"


def t_动能灌注():
    """阿科米亚核心等级+10。（攻击次数那半句引擎不模拟攻击，不测）"""
    eng = mk(0)
    add(eng, HL_A, 0)
    c0, i0 = core_now(eng), len(eng.s.log)
    take(eng, "动能灌注")
    g = core_gain(eng, i0, c0)
    assert g == 10, f"核心 +{g}，应 +10"


def t_能量改造():
    """回合开始时，核心每拥有30级，将随机触发场上1名英雄的1次整备或登场效果。（核心 60 级 → 额外 2 次；按 eng.fire 计数）"""
    for seed in range(5):
        eng = mk(seed)
        eng.s.core = 60
        add(eng, hero("程咬金"), 0)                         # 整备：对自身用古币
        add(eng, hero("庄周"), 1)                           # 登场：自身等级+2
        take(eng, "能量改造")
        calls = []
        orig = eng.fire

        def rec(u, timing, **kw):
            if timing in ("prepare", "deploy"):
                calls.append((u.name, timing))
            return orig(u, timing, **kw)
        eng.fire = rec
        eng.round_start()
        extra = len(calls) - len(eng.s.units)               # 回合开始本来就给每个单位一次整备
        assert extra == 2, f"seed{seed} 额外触发 {extra} 次，应 60//30=2（{calls}；核心等级放在 flags {CORE_KEYS}）"


def t_积极整备():
    """你的英雄每触发5次整备效果，随机获得1件日落海专属装备，上限3件。"""
    eng = mk(0)
    add(eng, hero("程咬金"), 0)                             # 有整备效果，且整备不进手牌
    take(eng, "积极整备")
    for r in range(1, 21):
        eng.round_start()
        got = [x for x in eng.s.hand if x in SUNSET_EQUIPS]
        exp = min(3, r // 5)
        assert len(got) == exp, f"整备 {r} 次后日落海专属装备 {got}，应 {exp} 件"


def t_智能装甲():
    """随机获得1件【日落海专属装备】。我方英雄每穿戴1件【日落海专属装备】，其整备效果会额外触发1次。"""
    seen = set()
    for seed in range(15):
        eng = mk(seed)
        take(eng, "智能装甲")
        h = eng.s.hand
        assert len(h) == 1 and h[0] in SUNSET_EQUIPS, f"seed{seed} 手牌 {h}，应 1 件日落海专属装备"
        seen.add(h[0])
    assert len(seen) >= 2, f"15 次只给过 {seen}"
    for n_eq, exp in ((0, 2), (1, 4), (2, 6)):
        eng = mk(0)
        cj = add(eng, hero("程咬金"), 0, level=1, equips=["星源装甲", "星源合金盾"][:n_eq])
        take(eng, "智能装甲")
        eng.s.hand.clear()
        eng.round_start()
        assert cj.level - 1 == exp, f"穿 {n_eq} 件日落海专属装备，程咬金整备给了 {cj.level - 1} 级，应 {exp}"


# ================================================================ 怒涛 / 王维 / 红豆之托 / 贤士登门 / 乘胜追击
def t_怒涛():
    """【三分】英雄获得牺牲（大乔灵体是战斗效果）。养成上的作用：三分英雄拥有'牺牲'。"""
    eng = mk(0)
    sf = add(eng, SF_A, 0)
    h = add(eng, HL_A, 1)
    assert not eng.has_kw(sf, "牺牲"), "前提：大乔本来没有牺牲"
    take(eng, "怒涛")
    eng.round_start()
    assert eng.has_kw(sf, "牺牲"), "三分英雄没获得牺牲"
    assert not eng.has_kw(h, "牺牲"), "非三分英雄也获得了牺牲"


def t_王维():
    """开团：获得空山人语和水穷云起各1张。（觉醒翻倍）"""
    eng = mk(0)
    w = add(eng, hero("王维"), 0)
    need("王维")
    eng.fire(w, "open")
    assert cnt(eng, "空山人语") == 1 and cnt(eng, "水穷云起") == 1, f"手牌 {eng.s.hand}"
    eng = mk(0)
    w = add(eng, hero("王维"), 0, awaken=3)
    eng.fire(w, "open")
    assert cnt(eng, "空山人语") == 2 and cnt(eng, "水穷云起") == 2, f"觉醒后手牌 {eng.s.hand}，应各 2 张"


def t_红豆之托():
    """获得1张【河洛】英雄王维。备战阶段，场上英雄总等级分别达到100/200/400级时，再获得1张王维。"""
    eng = mk(0)
    a = add(eng, HL_A, 0, level=30)
    add(eng, ZL_A, 1, level=20)
    take(eng, "红豆之托")
    assert cnt(eng, "王维") == 1, f"手牌 {eng.s.hand}"
    for lv, exp in ((30, 1), (80, 2), (80, 2), (180, 3), (300, 3), (380, 4), (900, 4)):
        a.level = lv                                       # 总等级 = lv + 20
        eng.round_start()
        assert cnt(eng, "王维") == exp, f"总等级 {lv + 20} 时王维 {cnt(eng, '王维')} 张，应 {exp}"


def t_贤士登门():
    """随机获得1张【逐鹿】英雄牌。回合开始时，再获得1张。（按每个回合开始都给读）"""
    eng = mk(0)
    take(eng, "贤士登门")
    h = eng.s.hand
    assert len(h) == 1 and category(h[0]) == "英雄牌" and faction(h[0]) == "逐鹿", f"手牌 {h}"
    eng.round_start()
    eng.round_start()
    h = eng.s.hand
    assert len(h) == 3 and all(category(x) == "英雄牌" and faction(x) == "逐鹿" for x in h), f"两个回合开始后手牌 {h}，应 3 张逐鹿英雄"


def t_乘胜追击():
    """战斗胜利时，获得1张高阶推演战术。"""
    eng = mk(0)
    add(eng, ZL_A, 0)
    take(eng, "乘胜追击")
    with force(False):
        eng.battle()
    assert cnt(eng, "高阶推演战术") == 0, f"失败也给了 {eng.s.hand}"
    with force(True):
        eng.battle()
    assert cnt(eng, "高阶推演战术") == 1, f"胜利后手牌 {eng.s.hand}"


def t_王牌认证():
    """战斗开始时，若你至少有3名4-5阶英雄，获得1张2-3阶效果锦囊。"""
    tiers = set()
    for seed in range(10):
        eng = mk(seed)
        for i, n in enumerate(HI45[:3]):
            add(eng, n, i)
        take(eng, "王牌认证")
        eng.battle()
        h = eng.s.hand
        assert len(h) == 1 and h[0] in ("2阶效果锦囊", "3阶效果锦囊"), f"seed{seed} 3 名 4~5 阶，手牌 {h}"
        tiers.add(h[0])
        eng.battle()
        assert len(eng.s.hand) == 2, f"seed{seed} 第二场后手牌 {eng.s.hand}，应每场 1 张"
    eng = mk(0)
    add(eng, HI45[0], 0), add(eng, HI45[1], 1)
    add(eng, WU_A, 2), add(eng, SF_B, 3)                    # 3 阶不算
    take(eng, "王牌认证")
    eng.battle()
    assert not eng.s.hand, f"只有 2 名 4~5 阶也给了 {eng.s.hand}"


def t_崛起():
    """获得1张马可波罗和1张百里守约。共战时，获得3临时等级。若战斗胜利，至多保留他们获得的20临时等级。
    读法：两人同时在场参战 → 各 +3 临时；胜利时把他们本场获得的临时等级转成永久（上限 20，按人还是合计有歧义，这里只到 3，不触上限）。"""
    eng = mk(0)
    take(eng, "崛起")
    assert sorted(eng.s.hand) == sorted(["马可波罗", "百里守约"]), f"手牌 {eng.s.hand}"
    for win in (False, True):
        eng = mk(0)
        m = add(eng, hero("马可波罗"), 0, level=5)
        b = add(eng, hero("百里守约"), 1, level=5)
        o = add(eng, ZL_A, 2, level=5)
        take(eng, "崛起")
        eng.s.hand.clear()
        i0 = len(eng.s.log)
        with force(win):
            eng.battle()
        got = temps_since(eng, i0)
        assert got.get(m.name, 0) == 3 and got.get(b.name, 0) == 3, f"共战临时等级 {got}，应两人各 3"
        assert got.get(o.name, 0) == 0, f"别人也拿了临时等级 {got}"
        exp = 8 if win else 5
        assert m.level == exp and b.level == exp, f"{'胜' if win else '负'}后等级 {m.level}/{b.level}，应 {exp}"
        assert o.level == 5
    eng = mk(0)                                            # 只有一人在场：不算共战
    m = add(eng, hero("马可波罗"), 0, level=5)
    take(eng, "崛起")
    i0 = len(eng.s.log)
    with force(True):
        eng.battle()
    assert temps_since(eng, i0).get(m.name, 0) == 0 and m.level == 5, "只有马可波罗在场也触发了共战"


# ================================================================ 运行
def t_透支():
    """能量可用至-15。回合开始时，如果能量为负，本回合无法打出英雄牌。（State.energy_floor / eng.spend / eng.can_play_hero）"""
    eng = mk(0, energy=2)
    assert not eng.spend(5), "没有透支时不能花到负数"
    take(eng, "透支")
    assert eng.spend(5) and eng.s.energy == -3, f"透支后能花到负数：{eng.s.energy}"
    assert not eng.spend(13), "下限 -15：-3 再花 13 = -16 不行"
    assert eng.spend(12) and eng.s.energy == -15, f"正好到 -15 可以：{eng.s.energy}"
    eng.round_start()
    assert not eng.can_play_hero(), "回合开始能量为负 → 本回合不能打英雄牌"
    eng.s.energy = 4
    eng.round_start()
    assert eng.can_play_hero(), "回合开始能量不为负 → 可以打英雄牌"


def t_技高一筹():
    """【逐鹿】英雄每释放6次技能，获得1能量，上限5能量。（放技能次数 eng.casts，测试用 counters["casts"] 指定）"""
    eng = mk(0, energy=0)
    take(eng, "技高一筹")
    a = add(eng, ZL_A, 0, level=5)
    b = add(eng, HL_A, 1, level=5)                          # 河洛：不算
    a.counters["casts"] = 4
    b.counters["casts"] = 50
    e0 = eng.s.energy
    eng.battle()
    assert eng.s.energy - e0 == 0, f"逐鹿放 4 次不够 6 次：+{eng.s.energy - e0}"
    eng.battle()
    assert eng.s.energy - e0 == 1, f"两场共 8 次 → +1（余数累计）：+{eng.s.energy - e0}"
    a.counters["casts"] = 60
    for _ in range(3):
        eng.battle()
    assert eng.s.energy - e0 == 5, f"整局上限 5 能量：+{eng.s.energy - e0}"


def t_光棱塔():
    """阿科米亚核心不再分配等级。（有光棱塔时开战核心分配不给临时等级；对照：没有时日落海英雄分到）"""
    for with_tower in (False, True):
        eng = mk(0)
        sun = add(eng, hero("艾琳"), 0, level=10)
        if with_tower:
            take(eng, "光棱塔")
        eng.s.core = 40
        i0 = len(eng.s.log)
        eng.battle()
        got = sum(kw.get("n", 0) for (_, ev, kw) in eng.s.log[i0:]
                  if ev == "临时等级" and str(kw.get("src", "")).startswith("阿科米亚核心"))
        if with_tower:
            assert got == 0, f"有光棱塔时核心不该分配：{got}"
        else:
            assert got == 20, f"对照：核心 40 → 单人上限 50% = 20：{got}"


TESTS = [
    ("透支", t_透支), ("技高一筹", t_技高一筹), ("光棱塔", t_光棱塔),
    ("三十而达", t_三十而达), ("卧薪尝胆", t_卧薪尝胆), ("保持进步", t_保持进步), ("人才济济", t_人才济济),
    ("祝你好运", t_祝你好运), ("碰碰运气", t_碰碰运气), ("重大里程碑", t_重大里程碑),
    ("超速成长", t_超速成长),
    ("河洛来客", _visitor("河洛来客", "李信")), ("逐鹿来客", _visitor("逐鹿来客", "蒙犽")),
    ("大河来客", _visitor("大河来客", "刘邦")), ("日落海来客", _visitor("日落海来客", "雅典娜")),
    ("三分来客", _visitor("三分来客", "小乔")),
    ("河洛精兵", _elite("河洛精兵", "河洛")), ("逐鹿精兵", _elite("逐鹿精兵", "逐鹿")),
    ("大河精兵", _elite("大河精兵", "大河流域")), ("日落海精兵", _elite("日落海精兵", "日落海")),
    ("三分精兵", _elite("三分精兵", "三分之地")),
    ("紧急行动", t_紧急行动), ("富贵险中求", t_富贵险中求), ("一刀穷一刀富", t_一刀穷一刀富), ("险中求富", t_险中求富),
    ("前浪的荣光", t_前浪的荣光), ("全军列阵", t_全军列阵), ("胜之不武", t_胜之不武),
    ("一分钱一分货", t_一分钱一分货), ("等价交换", t_等价交换), ("锦囊妙计", t_锦囊妙计), ("锦囊", t_锦囊),
    ("挂帅上阵", t_挂帅上阵),
    ("鲸吞", _give_one("鲸吞", "庄周")), ("生命绽放", _give_one("生命绽放", "鬼谷子")),
    ("一身是胆", _give_one("一身是胆", "赵云")), ("卧龙威势", _give_one("卧龙威势", "诸葛亮")),
    ("伤害转化器", t_伤害转化器), ("大买卖", t_大买卖),
    ("临时增幅", t_临时增幅), ("协同一致", t_协同一致), ("强力小弟", t_强力小弟), ("三三得三", t_三三得三),
    ("乐善好施", t_乐善好施), ("献祭之赐", t_献祭之赐), ("钱生战力", t_钱生战力),
    ("多彩古币", t_多彩古币), ("传家宝", t_传家宝), ("古币翻新", t_古币翻新), ("零钱兑换", t_零钱兑换),
    ("破损古币", t_破损古币), ("古币契约", t_古币契约), ("幸运硬币", t_幸运硬币),
    ("因缘际会", t_因缘际会), ("图腾启灵", t_图腾启灵), ("化为熊灵", t_化为熊灵),
    ("场外供能", t_场外供能), ("动能灌注", t_动能灌注), ("能量改造", t_能量改造),
    ("积极整备", t_积极整备), ("智能装甲", t_智能装甲),
    ("怒涛", t_怒涛), ("王维", t_王维), ("红豆之托", t_红豆之托), ("贤士登门", t_贤士登门), ("乘胜追击", t_乘胜追击),
    ("王牌认证", t_王牌认证), ("崛起", t_崛起),
    ("挂帅", t_挂帅),                                     # 挂帅上阵送的牌（不在批次表里）
]


def main() -> int:
    batch = json.loads((ROOT / "data/engine/sim_batches.json").read_text(encoding="utf-8"))["batches"][4]
    covered = {n for n, _ in TESTS} | set(COMBAT_ONLY) | set(NOT_TESTABLE)
    missing = [n for n in batch if n not in covered]
    extra = [n for n, _ in TESTS if n not in batch]
    ok = fail = 0
    for name, fn in TESTS:
        try:
            if name not in card_db():
                raise AssertionError("卡库里没有这张卡")
            need(name)
            fn()
            print(f"OK    {name}")
            ok += 1
        except Unimpl as e:
            print(f"FAIL  {name}: {e}")
            fail += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
            fail += 1
        except Exception as e:                             # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()
            print(f"FAIL  {name}: 异常 {type(e).__name__}: {e} @ {tb[-3].strip() if len(tb) >= 3 else ''}")
            fail += 1
    for n in COMBAT_ONLY:
        print(f"SKIP  {n}: 只有战斗效果")
    for n, why in NOT_TESTABLE.items():
        print(f"SKIP  {n}: {why}")
    if missing:
        print(f"注意：本批这些卡没有归类 {missing}")
    if extra:
        print(f"注意：这些测试的卡不在批次表里（赠送牌）{extra}")
    print(f"\n通过 {ok} / 失败 {fail}（共 {ok + fail} 条测试，跳过 {len(COMBAT_ONLY) + len(NOT_TESTABLE)} 张）")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
