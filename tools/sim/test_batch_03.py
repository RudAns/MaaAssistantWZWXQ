"""第 3 批卡牌（几乎全是天赋）的独立测试：**只按 data/cards.json 卡面原文写**，没有读 agent/sim/cards/batch_03.py。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_batch_03.py
逐条打印 OK / FAIL（含原因），末尾打印通过数 / 失败数，有失败退出码 1。
没注册的卡记 FAIL "未实现"，不抛异常。

约定（与 agent/sim/play.Player._take 一致）：
  选天赋 = s.talents.append(名) + 有 on_use 就 use_card(名, None)；
  天赋送出的同名牌（复印机 / 援军 / 自适应武装 / 限制性进化 …）= 带目标 use_card(名, 英雄)；
  "打出英雄牌" = eng.deploy / eng.merge（引擎对它们发 deploy 事件）。
测试用的英雄只挑"在本测试场景里不会触发任何钩子"的（运行时用 quiet() 检查，不看实现代码）。

只有战斗效果的卡列在 COMBAT_ONLY；影响养成但接口搭不出局面的列在 NOT_TESTABLE。
"""
from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.engine import board as B                                              # noqa: E402
from agent.sim.engine import Engine, impl, card_db, quality, category, faction, card_text   # noqa: E402
from agent.sim.state import State, Unit                                          # noqa: E402
import agent.sim.engine as _E                                                     # noqa: E402
_E.EQUIP_DROP_ROUNDS = ()      # 固定回合投放装备（第 4/8/12/16 回合）是引擎规则，单卡测试里关掉，免得混进"手里有几件装备"的断言

COMBAT_ONLY = [
    "防御领域", "狂铁·强化", "微弱优势", "刻舟减蓝", "猪猪", "专注射击", "超专注射击", "2级电磁弹簧",
    "小小的助力", "战前叫阵", "重装上阵", "前线堡垒", "强者自强", "破败之力", "猛虎擂台",
]
NOT_TESTABLE = {
    "破格": "回合开始商店自动刷新 —— sim 引擎没有商店 / 自动刷新时机",
    "节能商店": "手动刷新商店后某张卡降价 —— sim 引擎有刷新动作（eng.refresh_shop），但不模拟商店价格",
}

HOOK_PREFIX = ("on_", "react_", "coin_bonus", "grant_keywords")
DEPLOY_SAFE = {"on_sacrifice", "on_death", "on_kill", "on_win", "on_lose", "on_open", "on_battle_start",
               "on_sell", "on_battle_end"}          # 只上场 / 回合开始、不打仗时不会被触发的钩子


class Unimpl(Exception):
    pass


# ---------------------------------------------------------------- 搭局面
def mk(seed=0, **kw) -> Engine:
    return Engine(State(**kw), random.Random(seed))


def quiet(name: str, allow=()) -> str:
    """检查测试用英雄在场景里不会自己产生效果（有钩子但不在 allow 里 → 测试数据有误）。"""
    assert category(name) == "英雄牌", f"测试数据：{name} 不是英雄牌"
    c = impl(name)
    hooks = {a for a in dir(c) if a.startswith(HOOK_PREFIX)} if c else set()
    bad = hooks - set(allow)
    assert not bad, f"测试数据：{name} 有钩子 {sorted(bad)}，换一个英雄"
    return name


def add(eng: Engine, name: str, cell_idx: int, level: float = 1.0, kind: str = "hero", awaken: int = 0) -> Unit:
    """直接放一个单位（不触发登场）。"""
    u = Unit(name=name, level=level, cell=B.CELLS[cell_idx], kind=kind, pop=(kind == "hero"),
             uid=eng.s.next_uid, awaken=awaken)
    eng.s.next_uid += 1
    eng.s.units.append(u)
    return u


def need(name: str):
    c = impl(name)
    if c is None:
        raise Unimpl("未实现")
    return c


def pick_talent(eng: Engine, name: str):
    need(name)
    eng.s.talents.append(name)
    if hasattr(impl(name), "on_use"):
        eng.use_card(name, None)


def use(eng: Engine, name: str, target=None):
    c = need(name)
    if not hasattr(c, "on_use"):
        raise AssertionError(f"{name} 没有 on_use，无法使用")
    return eng.use_card(name, target)


def recorder(eng: Engine, pick=0):
    calls = []

    def ch(e, kind, options, **ctx):
        calls.append((kind, list(options)))
        return options[pick] if not isinstance(options[0], bool) else True
    eng.chooser = ch
    return calls


def heroes_in_hand(eng):
    return [x for x in eng.s.hand if category(x) == "英雄牌"]


def equips_in_hand(eng):
    return [x for x in eng.s.hand if category(x) == "装备牌"]


def eff_xp(s) -> int:
    """总经验（把已经升掉的级折回经验），实现里顺手 level_ups 也不影响判断。"""
    return s.xp + sum(B.XP_NEED[l] for l in range(1, s.lord_level))


def battle_seeds(want_win: bool, n=1, start=0):
    """找出第一场战斗结果为 want_win 的种子（空场打一场看结果；随机流只用于这次比较）。"""
    out, s = [], start
    while len(out) < n and s < 500:
        e = mk(s)
        if e.battle() == want_win:
            out.append(s)
        s += 1
    return out


# 场景用英雄（品阶, 阵营）
T1_HL = "苏烈"            # 1 河洛（只有牺牲）
T2_HL_A, T2_HL_B = "沈梦溪", "百里守约"   # 2 河洛
T2_ZL_A, T2_ZL_B = "墨子", "廉颇"         # 2 逐鹿
T3_SF = "马超"            # 3 三分
T3_HL = "狄仁杰"          # 3 河洛
T3_WZ = "苍"              # 3 无阵营
T4_ZL_A, T4_ZL_B = "扁鹊", "嬴政"         # 4 逐鹿
T5_SF = "吕布"            # 5 三分


# ================================================================ 直接给资源的天赋
def _energy_talent(name, n):
    def t():
        """获得 n 能量。"""
        eng = mk(0, energy=4)
        pick_talent(eng, name)
        assert eng.s.energy == 4 + n, f"能量 4→{eng.s.energy}，应 +{n}"
        assert not eng.s.hand, f"不该拿牌 {eng.s.hand}"
    return t


def _xinyuan(name, fac):
    def t():
        """获得2能量，以及随机1张2-3阶【fac】英雄牌。"""
        tiers = set()
        for seed in range(25):
            eng = mk(seed, energy=0, lord_level=1)
            pick_talent(eng, name)
            assert eng.s.energy == 2, f"seed{seed} 能量 0→{eng.s.energy}，应 +2"
            h = eng.s.hand
            assert len(h) == 1 and category(h[0]) == "英雄牌", f"seed{seed} 应得 1 张英雄牌，实际 {h}"
            assert quality(h[0]) in (2, 3), f"seed{seed} {h[0]} 是 {quality(h[0])} 阶（应 2-3 阶）"
            assert faction(h[0]) == fac, f"seed{seed} {h[0]} 阵营 {faction(h[0])}，应 {fac}"
            tiers.add(quality(h[0]))
        assert tiers == {2, 3}, f"25 次只出过 {tiers} 阶，不像 2-3 阶随机"
    return t


def t_双人游():
    """随机获得2张2-3阶英雄牌。"""
    for seed in range(15):
        eng = mk(seed, lord_level=1)
        pick_talent(eng, "双人游")
        h = eng.s.hand
        assert len(h) == 2 and all(category(x) == "英雄牌" and quality(x) in (2, 3) for x in h), \
            f"seed{seed} 应得 2 张 2-3 阶英雄牌，实际 {[(x, quality(x)) for x in h]}"


def t_结伴():
    """获得2能量以及2张3阶英雄兑换券。"""
    eng = mk(0, energy=1)
    pick_talent(eng, "结伴")
    assert eng.s.energy == 3, f"能量 1→{eng.s.energy}"
    assert sorted(eng.s.hand) == ["3阶英雄兑换券"] * 2, f"手牌 {eng.s.hand}"


def t_战术模组():
    """可上阵人数+1。"""
    eng = mk(0, lord_level=3)
    cap = eng.pop_cap()
    pick_talent(eng, "战术模组")
    assert eng.pop_cap() == cap + 1, f"人口上限 {cap}→{eng.pop_cap()}"


def t_极限操作():
    """棋手生命值-15（State 无生命值，不测），获得15能量。"""
    eng = mk(0, energy=2)
    pick_talent(eng, "极限操作")
    assert eng.s.energy == 17, f"能量 2→{eng.s.energy}"


def t_补给线开团():
    """获得3张开团效果英雄牌。"""
    for seed in range(10):
        eng = mk(seed, lord_level=3)
        pick_talent(eng, "补给线：开团")
        h = eng.s.hand
        assert len(h) == 3, f"seed{seed} 应得 3 张，实际 {h}"
        bad = [x for x in h if category(x) != "英雄牌" or "开团" not in card_text(x)]
        assert not bad, f"seed{seed} 这些不是开团英雄牌 {bad}"


def t_抢攻():
    """获得1张普通装备牌（之后的'对棋手造成伤害换能量'无法模拟，不测）。"""
    for seed in range(5):
        eng = mk(seed)
        pick_talent(eng, "抢攻")
        assert len(eng.s.hand) == 1 and len(equips_in_hand(eng)) == 1, f"seed{seed} 手牌 {eng.s.hand}"


def t_擂台猛虎():
    """获得1个猛虎擂台（召唤物）。增益是战斗效果，只测"拿到了擂台"（在场上或在手里）。"""
    eng = mk(0)
    pick_talent(eng, "擂台猛虎")
    on = [u for u in eng.s.units if u.name == "猛虎擂台"]
    inh = [x for x in eng.s.hand if x == "猛虎擂台"]
    assert len(on) + len(inh) == 1, f"应获得 1 个猛虎擂台：场上 {on} 手牌 {eng.s.hand}"
    for u in on:
        assert u.kind != "hero" and not u.pop, f"擂台不应是占人口的英雄 kind={u.kind} pop={u.pop}"


# ================================================================ 送一张天赋牌的天赋
def _gives(name, cards):
    def t():
        """选取后手牌里出现对应天赋牌。"""
        eng = mk(0)
        pick_talent(eng, name)
        assert sorted(eng.s.hand) == sorted(cards), f"手牌 {eng.s.hand}，应为 {cards}"
    return t


# ================================================================ 天赋牌（被送出的）
def t_二象坍缩():
    """指定1名英雄，获得3张同阶英雄牌，或2张同阵营的当前最高阶英雄牌（随机生效其一）。
    用棋手 5 级消除"当前最高阶"的歧义（商店最高品阶 = 卡库最高品阶 = 5）。"""
    need("二象坍缩")
    seen = set()
    for seed in range(30):
        eng = mk(seed, lord_level=5)
        t = add(eng, quiet(T1_HL, DEPLOY_SAFE), 3, level=5)
        ok = use(eng, "二象坍缩", t)
        assert ok is not False, "合法目标却返回 False"
        h = eng.s.hand
        assert h and all(category(x) == "英雄牌" for x in h), f"seed{seed} 手牌 {h}"
        if len(h) == 3 and all(quality(x) == 1 for x in h):
            seen.add("同阶")
        elif len(h) == 2 and all(quality(x) == 5 and faction(x) == "河洛" for x in h):
            seen.add("同阵营最高阶")
        else:
            raise AssertionError(f"seed{seed} 两种结果都不像：{[(x, quality(x), faction(x)) for x in h]}")
    assert seen == {"同阶", "同阵营最高阶"}, f"30 次只出现过 {seen}"
    eng = mk(0)
    assert use(eng, "二象坍缩", None) is False and not eng.s.hand, "没指定英雄应返回 False"


def t_体面离场():
    """指定移除一名英雄，获得能量。觉醒进度 0/1/2/3 时获得 3/6/9/12 能量。"""
    need("体面离场")
    for k, gain in enumerate((3, 6, 9, 12)):
        eng = mk(0, energy=1)
        t = add(eng, quiet(T2_HL_A), 3, level=10, awaken=k)
        o = add(eng, quiet(T2_ZL_A), 10, level=4)
        use(eng, "体面离场", t)
        assert t not in eng.s.units, f"进度{k}：被指定的英雄还在场上"
        assert o in eng.s.units and o.level == 4, "别的英雄不该受影响"
        assert eng.s.energy == 1 + gain, f"进度{k}：能量 1→{eng.s.energy}，应 +{gain}"
    eng = mk(0, energy=1)
    assert use(eng, "体面离场", None) is False and eng.s.energy == 1, "没指定英雄应返回 False 且不给能量"


def t_契约瓶():
    """获胜时累积3张随机英雄牌；失败时损失一半；打出后获得累积的英雄牌。
    走"选瓶中人 → 打几场 → 打出契约瓶"。'一半'取整有歧义：向下 / 向上取整都接受（但要前后一致）。"""
    need("瓶中人"), need("契约瓶")
    patterns = set()
    for seed in range(12):
        eng = mk(seed)
        pick_talent(eng, "瓶中人")
        assert "契约瓶" in eng.s.hand, f"选瓶中人没拿到契约瓶 {eng.s.hand}"
        lo = hi = 0
        res = []
        for _ in range(3):
            w = eng.battle()
            res.append(w)
            if w:
                lo, hi = lo + 3, hi + 3
            else:
                lo, hi = lo - (lo + 1) // 2, hi - hi // 2       # lo：损失向上取整；hi：损失向下取整
        patterns.add(tuple(res))
        before = len(heroes_in_hand(eng))
        assert before == 0, f"契约瓶打出前就拿到了英雄牌 {eng.s.hand}"
        use(eng, "契约瓶")
        got = heroes_in_hand(eng)
        assert len(got) in (lo, hi), f"seed{seed} 胜负 {res}：拿到 {len(got)} 张，应为 {lo}（损失向上取整）或 {hi}（向下取整）"
    assert any(p[0] for p in patterns) and any(not p[0] for p in patterns), f"种子没覆盖胜负 {patterns}"
    eng = mk(0)
    pick_talent(eng, "瓶中人")
    use(eng, "契约瓶")
    assert not heroes_in_hand(eng), f"没打过仗就打出，应得 0 张，实际 {eng.s.hand}"


def t_胜利预言():
    """预测下场战斗胜利，预测成功获得2张3阶英雄兑换券（失败没有；只管下一场）。"""
    need("胜利预言")
    seen = set()
    for want in (True, False):
        for seed in battle_seeds(want, 2):
            eng = mk(seed)
            use(eng, "胜利预言")
            assert not eng.s.hand, f"打出时就给了牌 {eng.s.hand}"
            w = eng.battle()
            seen.add(w)
            exp = ["3阶英雄兑换券"] * 2 if w else []
            assert eng.s.hand == exp, f"seed{seed} {'胜' if w else '负'}：手牌 {eng.s.hand}，应 {exp}"
            n = len(eng.s.hand)
            for _ in range(4):
                eng.battle()
            assert len(eng.s.hand) == n, f"只预测下一场，之后又给了牌 {eng.s.hand}"
    assert seen == {True, False}, f"测试自身：种子没覆盖胜负 {seen}"


def t_失败预言():
    """预测下场战斗失败，预测成功获得4能量。"""
    need("失败预言")
    seen = set()
    for want in (True, False):
        for seed in battle_seeds(want, 2):
            eng = mk(seed, energy=0)
            use(eng, "失败预言")
            assert eng.s.energy == 0, "打出时就给了能量"
            w = eng.battle()
            seen.add(w)
            exp = 0 if w else 4
            assert eng.s.energy == exp, f"seed{seed} {'胜' if w else '负'}：能量 {eng.s.energy}，应 {exp}"
            for _ in range(4):
                eng.battle()
            assert eng.s.energy == exp, f"只预测下一场，之后能量变成 {eng.s.energy}"
    assert seen == {True, False}, f"测试自身：种子没覆盖胜负 {seen}"


def t_五福临门():
    """天赋：获得1张天赋牌五福临门；那张牌：随机获得5张英雄牌。
    同名歧义：选天赋时直接给 5 张英雄牌也接受；手里出现'五福临门'就再打出一次。"""
    for seed in range(5):
        eng = mk(seed, lord_level=3)
        pick_talent(eng, "五福临门")
        if "五福临门" in eng.s.hand:
            assert not heroes_in_hand(eng), f"送牌的同时就给了英雄 {eng.s.hand}"
            eng.s.hand.remove("五福临门")
            use(eng, "五福临门")
        h = heroes_in_hand(eng)
        assert len(h) == 5 and len(eng.s.hand) == 5, f"seed{seed} 应得 5 张英雄牌，实际 {eng.s.hand}"


def _copier(name, max_tier):
    def t():
        """获得2张天赋牌；指定1名英雄（复印机限 1-3 阶），获得该英雄牌。"""
        eng = mk(0)
        pick_talent(eng, name)
        assert eng.s.hand == [name, name], f"选天赋应得 2 张 {name}，实际 {eng.s.hand}"
        eng.s.hand.clear()
        a = add(eng, quiet(T3_SF), 3, level=5)
        b = add(eng, quiet(T4_ZL_A), 10, level=5)
        assert use(eng, name, a) is not False, f"{a.name}(3阶) 是合法目标"
        assert eng.s.hand == [a.name], f"应得 1 张 {a.name}，实际 {eng.s.hand}"
        ok = use(eng, name, b)
        if max_tier >= 4:
            assert ok is not False and eng.s.hand == [a.name, b.name], f"终极复印机指定4阶：手牌 {eng.s.hand}"
        else:
            assert ok is False and eng.s.hand == [a.name], f"复印机不能指定 4 阶：返回 {ok}，手牌 {eng.s.hand}"
        assert a.level == 5 and b.level == 5 and len(eng.s.units) == 2, "场上英雄不该变"
    return t


def t_听天由命():
    """消除场上4名等级最低的英雄。每消除1名，获得3能量、1张同阶兑换券、1张简易协作。"""
    need("听天由命")
    eng = mk(0, energy=0)
    spec = [(T1_HL, 1), (T2_HL_A, 2), (T3_SF, 3), (T4_ZL_A, 4), (T5_SF, 10), (T2_ZL_A, 11)]
    us = [add(eng, quiet(n, DEPLOY_SAFE), i * 3, level=lv) for i, (n, lv) in enumerate(spec)]
    totem = add(eng, "因缘图腾", 27, level=1, kind="summon")
    use(eng, "听天由命")
    left = sorted(u.name for u in eng.s.heroes())
    assert left == sorted([T5_SF, T2_ZL_A]), f"应剩 {T5_SF}/{T2_ZL_A}，实际剩 {left}"
    assert totem in eng.s.units, "召唤物不是英雄，不该被消除"
    assert eng.s.energy == 12, f"消除 4 名应 +12 能量，实际 {eng.s.energy}"
    exp = sorted([f"{q}阶英雄兑换券" for q in (1, 2, 3, 4)] + ["简易协作"] * 4)
    assert sorted(eng.s.hand) == exp, f"手牌 {sorted(eng.s.hand)}，应 {exp}"
    eng = mk(0, energy=0)                                   # 不足 4 名：有几个消几个
    add(eng, quiet(T2_HL_A), 3, level=3)
    add(eng, quiet(T3_SF), 10, level=8)
    use(eng, "听天由命")
    assert not eng.s.heroes() and eng.s.energy == 6, f"2 名：剩 {eng.s.heroes()} 能量 {eng.s.energy}"
    assert sorted(eng.s.hand) == sorted(["2阶英雄兑换券", "3阶英雄兑换券", "简易协作", "简易协作"]), f"手牌 {eng.s.hand}"


def t_人才搜寻():
    """卡牌3选1（商店可购的最高品阶英雄）；商店最高品阶 = 棋手等级（封顶 5）。"""
    need("人才搜寻")
    for lv in (2, 3, 4):
        for seed in range(4):
            eng = mk(seed, lord_level=lv)
            calls = recorder(eng, pick=-1)
            use(eng, "人才搜寻")
            h = eng.s.hand
            assert len(h) == 1 and category(h[0]) == "英雄牌" and quality(h[0]) == lv, \
                f"{lv}级 seed{seed}：应得 1 张 {lv} 阶英雄，实际 {[(x, quality(x)) for x in h]}"
            if calls:
                opts = calls[-1][1]
                assert len(opts) == 3 and len(set(opts)) == 3, f"应是 3 选 1：{opts}"
                assert all(quality(x) == lv for x in opts), f"选项品阶不对 {[(x, quality(x)) for x in opts]}"
                assert h[0] == opts[-1], f"拿到 {h[0]}，打法选的是 {opts[-1]}"


def t_限制性进化():
    """获得2张；指定移除1名1~2阶英雄，获得高2阶的随机英雄牌，且上阵时为原等级+10。"""
    eng = mk(0)
    pick_talent(eng, "限制性进化")
    assert eng.s.hand == ["限制性进化"] * 2, f"选天赋应得 2 张，实际 {eng.s.hand}"
    for src, orig in ((T1_HL, 7), (T2_HL_A, 12)):
        for seed in range(4):
            eng = mk(seed, lord_level=3)
            pick_talent(eng, "限制性进化")
            eng.s.hand.clear()
            t = add(eng, quiet(src, DEPLOY_SAFE), 3, level=orig)
            assert use(eng, "限制性进化", t) is not False, f"{src}({quality(src)}阶) 是合法目标"
            assert t not in eng.s.units, f"{src} 应被移除"
            h = heroes_in_hand(eng)
            assert len(h) == 1 and quality(h[0]) == quality(src) + 2, \
                f"{src}：应得 1 张 {quality(src) + 2} 阶英雄，实际 {[(x, quality(x)) for x in h]}"
            eng.s.hand.remove(h[0])
            u = eng.deploy(h[0])
            assert u is not None, "上阵失败"
            ci = impl(h[0])
            silent = not ci or not any(a.startswith(("on_deploy", "react_")) for a in dir(ci))
            if silent:
                assert u.level == orig + 10, f"{h[0]} 上阵等级 {u.level}，应为原等级 {orig}+10"
            else:
                assert u.level >= orig + 10, f"{h[0]} 上阵等级 {u.level}，至少应为 {orig}+10"
    eng = mk(0)
    pick_talent(eng, "限制性进化")
    eng.s.hand.clear()
    t = add(eng, quiet(T3_SF), 3, level=9)
    ok = use(eng, "限制性进化", t)
    assert ok is False and t in eng.s.units and not eng.s.hand, f"3 阶不是合法目标：返回 {ok} 手牌 {eng.s.hand}"


def t_援军():
    """获得1张援军；指定1名英雄，随机获得4张同阵营的英雄牌。"""
    eng = mk(0)
    pick_talent(eng, "援军")
    assert eng.s.hand == ["援军"], f"选天赋应得 1 张援军，实际 {eng.s.hand}"
    for seed in range(5):
        eng = mk(seed, lord_level=3)
        t = add(eng, quiet(T1_HL, DEPLOY_SAFE), 3)
        use(eng, "援军", t)
        h = eng.s.hand
        assert len(h) == 4 and all(category(x) == "英雄牌" and faction(x) == "河洛" for x in h), \
            f"seed{seed} 应得 4 张河洛英雄牌，实际 {[(x, faction(x)) for x in h]}"


def t_自适应武装():
    """获得1张自适应武装；指定1名英雄，随机获得2张与其适配的普通装备牌（'适配'只测是装备牌）。"""
    eng = mk(0)
    pick_talent(eng, "自适应武装")
    assert eng.s.hand == ["自适应武装"], f"选天赋应得 1 张，实际 {eng.s.hand}"
    for seed in range(5):
        eng = mk(seed)
        t = add(eng, quiet(T3_SF), 3)
        use(eng, "自适应武装", t)
        assert len(eng.s.hand) == 2 and len(equips_in_hand(eng)) == 2, f"seed{seed} 手牌 {eng.s.hand}"
        assert not t.equips, "是获得装备牌（进手牌），不是直接穿上"


def t_拍卖重现():
    """一 / 二：从未获得的拍品里挑 1 张（拍卖内容 State 里没有，只测：最多得 1 张、不花能量）。
    三：消耗 4 能量再挑；能量不足 4 应不能用。
    接口缺口：State 没有"每轮拍卖未获得的拍品"，实现拒绝使用（返回 False）也接受，但那时不能扣钱。"""
    for n in ("拍卖重现：一", "拍卖重现：二"):
        eng = mk(0, energy=5)
        use(eng, n)
        assert eng.s.energy == 5, f"{n} 不该花能量：5→{eng.s.energy}"
        assert len(eng.s.hand) <= 1, f"{n} 最多挑 1 张：{eng.s.hand}"
    eng = mk(0, energy=10)
    ok = use(eng, "拍卖重现：三")
    if ok is False:                                          # State 里没有拍卖记录 → 无牌可展示，不能用：不该扣钱
        assert eng.s.energy == 10 and not eng.s.hand, f"三 用不了却扣了钱 / 给了牌：能量 {eng.s.energy} 手牌 {eng.s.hand}"
    else:
        assert eng.s.energy == 6, f"三：能量 10→{eng.s.energy}，应 -4"
        assert len(eng.s.hand) <= 1, f"三 最多挑 1 张：{eng.s.hand}"
    eng = mk(0, energy=3)
    ok = use(eng, "拍卖重现：三")
    assert ok is False and eng.s.energy == 3 and not eng.s.hand, f"能量 3 不够 4：返回 {ok} 能量 {eng.s.energy}"


# ================================================================ 持续生效的天赋
def t_里程碑():
    """棋手到达5级时，获得6能量、1张装备箱（只一次：5→6 不再给）。"""
    eng = mk(0, lord_level=3, xp=0, energy=0)
    pick_talent(eng, "里程碑")
    eng.s.xp = 17
    eng.level_ups()
    assert eng.s.lord_level == 4 and eng.s.energy == 0 and not eng.s.hand, f"到 4 级不该给：能量 {eng.s.energy} 手牌 {eng.s.hand}"
    eng.s.xp = 23
    eng.level_ups()
    assert eng.s.lord_level == 5, "测试自身：没升到 5 级"
    assert eng.s.energy == 6 and eng.s.hand == ["装备箱"], f"到 5 级：能量 {eng.s.energy} 手牌 {eng.s.hand}"
    eng.s.xp = 27
    eng.level_ups()
    assert eng.s.energy == 6 and eng.s.hand == ["装备箱"], f"5→6 级不该再给：能量 {eng.s.energy} 手牌 {eng.s.hand}"


def t_整装待发():
    """若回合内没有打出英雄牌，下回合开始时获得2能量。"""
    need("整装待发")

    def next_round(eng):
        """战斗 → 回合结束 → 下回合开始，返回扣掉回合收入后的能量变化。"""
        e0 = eng.s.energy
        eng.battle()
        eng.round_end()
        eng.round_start()
        return eng.s.energy - e0 - B.income(eng.s.round)

    eng = mk(0)
    pick_talent(eng, "整装待发")
    d = next_round(eng)
    assert d == 2, f"没打英雄牌：额外能量 {d}，应 +2"
    eng = mk(0)
    pick_talent(eng, "整装待发")
    eng.deploy(quiet(T1_HL, DEPLOY_SAFE))
    d = next_round(eng)
    assert d == 0, f"打过英雄牌：不该给能量，额外 {d}"
    d = next_round(eng)                                    # 再下一回合没打：照样给
    assert d == 2, f"第二回合没打英雄牌：额外 {d}，应 +2"


def t_先苦后甜():
    """选取天赋后，输掉2场战斗时，获得6棋手经验。
    歧义：是否每输 2 场都给 —— 第 4 场败时 6 或 12 都接受。"""
    need("先苦后甜")
    checked = False
    for seed in range(20):
        eng = mk(seed, lord_level=1, xp=0)
        pick_talent(eng, "先苦后甜")
        losses = 0
        for _ in range(12):
            if not eng.battle():
                losses += 1
            x = eff_xp(eng.s)
            if losses < 2:
                assert x == 0, f"seed{seed} 输 {losses} 场就给了经验 {x}"
            elif losses < 4:
                assert x == 6, f"seed{seed} 输 {losses} 场：经验 {x}，应 6"
            else:
                assert x in (6, 12, 18, 24, 30) and x <= 6 * (losses // 2), f"seed{seed} 输 {losses} 场：经验 {x}"
                checked = True
    assert checked, "测试自身：种子没输够 4 场"


def t_存款利息():
    """获得3能量。回合结束时，每有7能量额外获得1能量。
    引擎没有'回合结束'钩子：在 战斗→round_end→round_start 之间结算都接受（按收入前 / 后的能量算都行）。"""
    eng = mk(0, energy=0)
    pick_talent(eng, "存款利息")
    assert eng.s.energy == 3, f"选取：0→{eng.s.energy}，应 +3"
    eng.s.energy = 21
    eng.battle()
    eng.round_end()
    eng.round_start()
    inc = B.income(eng.s.round)
    extra = eng.s.energy - 21 - inc
    assert extra in (21 // 7, (21 + inc) // 7), f"21 能量：额外得 {extra}，应 3（或按收入后算 {(21 + inc) // 7}）"
    eng.s.energy = 6
    r = eng.s.round
    eng.battle()
    eng.round_end()
    eng.round_start()
    inc = B.income(r + 1)
    extra = eng.s.energy - 6 - inc
    assert extra in (0, (6 + inc) // 7), f"6 能量：额外得 {extra}"


def t_三十而立():
    """备战阶段中，英雄总等级到达30级时，获得6棋手经验（一次）。"""
    need("三十而立")
    eng = mk(0, lord_level=1, xp=0)
    add(eng, quiet(T2_HL_A), 3, level=10)
    add(eng, quiet(T2_ZL_A), 10, level=10)
    pick_talent(eng, "三十而立")
    eng.round_start()
    eng.battle()
    assert eff_xp(eng.s) == 0, f"总等级 20 就给了经验 {eff_xp(eng.s)}"
    eng.s.units[0].level = 16
    eng.s.units[1].level = 15
    eng.round_start()
    eng.battle()
    assert eff_xp(eng.s) == 6, f"总等级 31：经验 {eff_xp(eng.s)}，应 6"
    eng.s.units[0].level = 40
    for _ in range(2):
        eng.round_start()
        eng.battle()
    assert eff_xp(eng.s) == 6, f"只给一次：经验 {eff_xp(eng.s)}"


def t_战争头牌():
    """每回合获得1个品阶数字，打出相同品阶的英雄牌时额外+4等级（棋手 5 级，1~5 阶各打一张，恰好一张 +4）。"""
    need("战争头牌")
    names = [quiet(n, DEPLOY_SAFE) for n in (T1_HL, T2_HL_A, T3_SF, T4_ZL_A, T5_SF)]
    tiers = set()
    for seed in range(12):
        eng = mk(seed, lord_level=5)
        pick_talent(eng, "战争头牌")
        eng.round_start()
        us = [eng.deploy(n) for n in names]
        lv = [u.level for u in us]
        hit = [quality(u.name) for u in us if u.level == 5]
        assert sorted(lv) == [1, 1, 1, 1, 5], f"seed{seed} 各品阶上场等级 {dict(zip(names, lv))}，应恰好一张 5 级"
        tiers.add(hit[0])
    assert len(tiers) >= 2, f"12 个种子品阶数字都是 {tiers}，不像每回合随机"


def t_中坚力量():
    """打出3阶英雄牌，额外+4级。"""
    need("中坚力量")
    eng = mk(0)
    pick_talent(eng, "中坚力量")
    a = eng.deploy(quiet(T3_SF))
    b = eng.deploy(quiet(T2_HL_A))
    c = eng.deploy(quiet(T4_ZL_A))
    assert a.level == 5, f"3 阶上场 {a.level}，应 1+4"
    assert b.level == 1 and c.level == 1, f"非 3 阶不该加：{b.level} / {c.level}"


def t_如法炮制():
    """每回合首次打出英雄牌时，获得1张同阶的英雄牌（不可超过棋手等级）。"""
    need("如法炮制")
    for seed in range(4):
        eng = mk(seed, lord_level=5)
        pick_talent(eng, "如法炮制")
        eng.deploy(quiet(T3_SF))
        h = eng.s.hand
        assert len(h) == 1 and category(h[0]) == "英雄牌" and quality(h[0]) == 3, f"首次打 3 阶：手牌 {h}"
        eng.deploy(quiet(T3_HL))
        assert len(eng.s.hand) == 1, f"本回合第二次不该再给：{eng.s.hand}"
        eng.round_start()
        eng.deploy(quiet(T2_HL_A))
        assert len(eng.s.hand) == 2 and quality(eng.s.hand[1]) == 2, f"新回合首次打 2 阶：手牌 {eng.s.hand}"
    eng = mk(0, lord_level=2)                                # 不可超过棋手等级
    pick_talent(eng, "如法炮制")
    eng.deploy(quiet(T3_SF))
    bad = [x for x in eng.s.hand if quality(x) > 2]
    assert not bad and len(eng.s.hand) <= 1, f"棋手 2 级打 3 阶：拿到 {[(x, quality(x)) for x in eng.s.hand]}"


def t_同路人():
    """同一回合内，每打出4张同阵营英雄牌，随机获得1张该阵营英雄牌。"""
    need("同路人")
    hl = [quiet(n, DEPLOY_SAFE) for n in (T1_HL, T2_HL_A, T2_HL_B, T3_HL)]
    eng = mk(0, lord_level=3)
    pick_talent(eng, "同路人")
    for n in hl[:3]:
        eng.deploy(n)
    assert not eng.s.hand, f"3 张就给了 {eng.s.hand}"
    eng.deploy(quiet(T2_ZL_A))                             # 别的阵营不算
    assert not eng.s.hand, f"第 4 张是逐鹿，不该给 {eng.s.hand}"
    eng.deploy(hl[3])
    h = eng.s.hand
    assert len(h) == 1 and category(h[0]) == "英雄牌" and faction(h[0]) == "河洛", f"第 4 张河洛：手牌 {h}"
    eng = mk(0, lord_level=3)                                # 跨回合不累计
    pick_talent(eng, "同路人")
    eng.deploy(hl[0]); eng.deploy(hl[1])
    eng.round_start()
    eng.deploy(hl[2]); eng.deploy(hl[3])
    assert not eng.s.hand, f"跨回合 2+2 不该给 {eng.s.hand}"


def t_群英荟萃():
    """回合中每打出3张不同品阶的英雄牌，全体英雄等级+3。"""
    need("群英荟萃")
    eng = mk(0)
    pick_talent(eng, "群英荟萃")
    totem = add(eng, "因缘图腾", 27, kind="summon")
    a = eng.deploy(quiet(T1_HL, DEPLOY_SAFE))
    b = eng.deploy(quiet(T2_HL_A))
    c = eng.deploy(quiet(T2_ZL_A))
    assert [a.level, b.level, c.level] == [1, 1, 1], f"只有 2 种品阶就加了 {[a.level, b.level, c.level]}"
    d = eng.deploy(quiet(T3_SF))
    assert [a.level, b.level, c.level, d.level] == [4, 4, 4, 4], f"凑齐 1/2/3 阶：等级 {[a.level, b.level, c.level, d.level]}，应全 4"
    assert totem.level == 1, "召唤物不是英雄"
    eng = mk(0)                                              # 跨回合不累计
    pick_talent(eng, "群英荟萃")
    a = eng.deploy(quiet(T1_HL, DEPLOY_SAFE))
    b = eng.deploy(quiet(T2_HL_A))
    eng.round_start()
    d = eng.deploy(quiet(T3_SF))
    assert [a.level, b.level, d.level] == [1, 1, 1], f"跨回合不该触发 {[a.level, b.level, d.level]}"


def _算盘(name, n):
    def t():
        """出售场上英雄时，随机1名英雄等级+n（按剩下的英雄里选）。"""
        need(name)
        hit = set()
        for seed in range(10):
            eng = mk(seed)
            pick_talent(eng, name)
            sold = add(eng, quiet(T2_HL_A), 3, level=5)
            rest = [add(eng, quiet(T2_ZL_A), 10, level=5), add(eng, quiet(T3_SF), 13, level=5)]
            eng.sell(sold)
            gains = [u.level - 5 for u in rest]
            assert sorted(gains) == [0, n], f"seed{seed} 剩下两人涨了 {gains}，应恰好一人 +{n}"
            hit.add(gains.index(n))
        assert hit == {0, 1}, "10 次都给同一个人，不像随机"
    return t


def t_常胜将军():
    """若至少2连胜，战斗胜利时获得2能量（和4棋手生命值，不测）。
    歧义：'至少2连胜'含不含本场 —— 读法A：本场胜且连胜≥2（含本场）；读法B：本场胜且之前已连胜≥2。"""
    need("常胜将军")
    for seed in range(6):
        eng = mk(seed, energy=0)
        pick_talent(eng, "常胜将军")
        res, gains = [], []
        for _ in range(10):
            e0 = eng.s.energy
            res.append(eng.battle())
            gains.append(eng.s.energy - e0)

        def streak_before(i):
            k = 0
            for w in reversed(res[:i]):
                if not w:
                    break
                k += 1
            return k
        a = [2 if res[i] and streak_before(i) + 1 >= 2 else 0 for i in range(10)]
        b = [2 if res[i] and streak_before(i) >= 2 else 0 for i in range(10)]
        assert gains == a or gains == b, f"seed{seed} 胜负 {['胜' if w else '负' for w in res]} 能量 {gains}；读法A {a} 读法B {b}"


def t_凯旋征召():
    """战斗胜利时，获得1张你等级最高的英雄牌。"""
    need("凯旋征召")
    for want in (True, False):
        for seed in battle_seeds(want, 2):
            eng = mk(seed)
            pick_talent(eng, "凯旋征召")
            add(eng, quiet(T2_HL_A), 3, level=10)
            add(eng, quiet(T2_ZL_A), 10, level=4)
            w = eng.battle()
            exp = [T2_HL_A] if w else []
            assert eng.s.hand == exp, f"seed{seed} {'胜' if w else '负'}：手牌 {eng.s.hand}，应 {exp}"


def _随机成长(name, n):
    def t():
        """战斗开始时，你品阶最高的1名英雄等级永久+n（并列随机一人）。"""
        need(name)
        eng = mk(0)
        pick_talent(eng, name)
        lo = add(eng, quiet(T2_ZL_A), 3, level=1)
        hi = add(eng, quiet(T4_ZL_A), 10, level=1)
        eng.battle()
        assert hi.level == 1 + n and lo.level == 1, f"4阶 {hi.level}（应 {1 + n}），2阶 {lo.level}（应 1）"
        assert hi.temp == 0, "是永久等级"
        eng.round_start()                                    # 两场战斗之间隔一个回合开始
        eng.battle()
        assert hi.level == 1 + 2 * n, f"每场战斗都给：两场后 {hi.level}"
        picked = set()
        for seed in range(10):
            eng = mk(seed)
            pick_talent(eng, name)
            x = add(eng, quiet(T4_ZL_A), 3, level=1)
            y = add(eng, quiet(T4_ZL_B), 10, level=1)
            add(eng, quiet(T2_ZL_A), 13, level=1)
            eng.battle()
            g = sorted([x.level - 1, y.level - 1])
            assert g == [0, n], f"seed{seed} 两名 4 阶并列：涨了 {g}，应只给一人"
            picked.add(x.level > 1)
        assert picked == {True, False}, "并列时总给同一人，不像随机"
    return t


def t_控能达人():
    """战斗开始时，若当前能量≤3，随机获得1张3-4阶英雄牌。"""
    need("控能达人")
    tiers = set()
    for seed in range(15):
        eng = mk(seed, energy=3)
        pick_talent(eng, "控能达人")
        add(eng, quiet(T2_HL_A), 3)                          # 场上要有人（空场没有"战斗开始"）
        eng.battle()
        h = eng.s.hand
        assert len(h) == 1 and category(h[0]) == "英雄牌" and quality(h[0]) in (3, 4), f"seed{seed} 能量3：手牌 {h}"
        tiers.add(quality(h[0]))
    assert tiers == {3, 4}, f"15 次只出过 {tiers} 阶"
    eng = mk(0, energy=4)
    pick_talent(eng, "控能达人")
    add(eng, quiet(T2_HL_A), 3)
    eng.battle()
    assert not eng.s.hand, f"能量 4 不该给 {eng.s.hand}"


def t_神秘英雄():
    """获得2张神秘英雄牌；每回合有15%概率自动销毁并获得3能量（手动刷新变身不测：引擎无商店）。"""
    names = ("神秘英雄牌", "神秘英雄")
    eng = mk(0)
    pick_talent(eng, "神秘英雄")
    got = [x for x in eng.s.hand if x in names]
    assert len(got) == 2 and len(eng.s.hand) == 2, f"应得 2 张神秘英雄牌，实际 {eng.s.hand}"
    destroyed = trials = 0
    for seed in range(300):
        eng = mk(seed, energy=0)
        pick_talent(eng, "神秘英雄")
        before = sum(1 for x in eng.s.hand if x in names)
        eng.round_start()
        eng.battle()
        after = sum(1 for x in eng.s.hand if x in names)
        d = before - after
        assert eng.s.energy == 3 * d, f"seed{seed} 销毁 {d} 张，能量 {eng.s.energy}（应 {3 * d}）"
        destroyed += d
        trials += before
    rate = destroyed / trials
    assert 0.09 <= rate <= 0.21, f"一回合销毁率 {rate:.3f}（{destroyed}/{trials}），应约 0.15"


def t_厉兵秣马():
    """随机获得1张普通装备牌。接下来5个回合若棋手等级没有变化，再获得2张。
    结算时机歧义：第 5 或第 6 个回合交界处给都接受（到第 4 个回合还没给、之后只给一次）。"""
    need("厉兵秣马")

    def cycle(eng):
        eng.battle()
        eng.s.round += 1                                     # 不走 round_end：避免自动加经验升级
        eng.round_start()

    for seed in range(3):
        eng = mk(seed, lord_level=2)
        pick_talent(eng, "厉兵秣马")
        assert len(equips_in_hand(eng)) == 1 and len(eng.s.hand) == 1, f"选取应得 1 张装备，实际 {eng.s.hand}"
        counts = []
        for _ in range(8):
            cycle(eng)
            counts.append(len(equips_in_hand(eng)))
        assert counts[3] == 1, f"seed{seed} 第 4 个回合就给了：{counts}"
        assert counts[5] == 3 and counts[-1] == 3, f"seed{seed} 每回合装备数 {counts}，应在第 5/6 个回合变 3 张且只给一次"
        assert len(eng.s.hand) == 3, f"不该拿别的牌 {eng.s.hand}"
    eng = mk(0, lord_level=2)                                # 棋手等级变了：不再给
    pick_talent(eng, "厉兵秣马")
    for k in range(8):
        if k == 1:
            eng.s.lord_level += 1
            eng.emit("lord_levelup", level=eng.s.lord_level)
        cycle(eng)
    assert len(equips_in_hand(eng)) == 1, f"棋手升过级还给了：{eng.s.hand}"


def t_扩编名额():
    """获得1个可购买天赋：花费8能量，可上阵人数+1，可购买2次。
    接口缺口：引擎没有'购买天赋'动作 —— 实现有 buy(eng) 就直接调；否则让推演打法在回合开始时选"买"。"""
    c = need("扩编名额")
    eng = mk(0, energy=20, lord_level=3)
    cap = eng.pop_cap()
    pick_talent(eng, "扩编名额")
    assert eng.s.energy == 20 and eng.pop_cap() == cap, f"选取时不该自动扣钱 / 加人口：能量 {eng.s.energy} 上限 {eng.pop_cap()}"
    if hasattr(c, "buy"):
        r = [c.buy(eng) for _ in range(3)]
        assert r[2] is False, "只能买 2 次"
    else:
        recorder(eng)
        eng.round_start()
        eng.round_start()
    assert eng.s.energy == 4 and eng.pop_cap() == cap + 2, f"买 2 次：能量 20→{eng.s.energy}（应 4），人口上限 {cap}→{eng.pop_cap()}（应 +2）"
    eng = mk(0, energy=7, lord_level=3)                      # 钱不够不能买
    pick_talent(eng, "扩编名额")
    if hasattr(c, "buy"):
        assert c.buy(eng) is False
    else:
        recorder(eng)
        eng.round_start()
    assert eng.s.energy == 7 and eng.pop_cap() == cap, f"7 能量不够：能量 {eng.s.energy} 上限 {eng.pop_cap()}"


def t_成年礼():
    """己方英雄30级时，随机获得2张其阵营的英雄牌（已达30级的英雄，再升级1次可获得）。
    引擎没有'等级变化'事件：给实现在 回合开始 / 战斗 里检查的机会。"""
    need("成年礼")
    eng = mk(0, lord_level=3)
    a = add(eng, quiet(T5_SF), 3, level=35)                  # 选天赋时已 35 级（三分）
    b = add(eng, quiet(T2_HL_A), 10, level=25)               # 河洛
    pick_talent(eng, "成年礼")
    eng.round_start()
    eng.battle()
    assert not eng.s.hand, f"没人升级不该给：{eng.s.hand}"
    eng.level(b, 6, src="测试")                               # 25 → 31
    eng.round_start()
    eng.battle()
    h = heroes_in_hand(eng)
    assert len(h) == 2 and all(faction(x) == "河洛" for x in h), f"{b.name} 到 30 级：手牌 {[(x, faction(x)) for x in eng.s.hand]}，应 2 张河洛"
    eng.level(b, 3, src="测试")
    eng.round_start()
    eng.battle()
    assert len(eng.s.hand) == 2, f"{b.name} 再升级不该再给：{eng.s.hand}"
    eng.level(a, 1, src="测试")                               # 已达 30 级的人再升 1 次
    eng.round_start()
    eng.battle()
    new = eng.s.hand[2:]
    assert len(new) == 2 and all(faction(x) == "三分之地" for x in new), f"{a.name} 再升 1 级：新增 {[(x, faction(x)) for x in new]}，应 2 张三分"


def t_拍卖专家():
    """每次拍卖后，获得1张天赋牌拍卖重现（第 1/2/3 次拍卖 → 一/二/三）。
    接口缺口：引擎没有拍卖事件 —— 先按回合推进（拍卖在第 4/8/12 回合开头），拿不到再试 emit("auction")。"""
    need("拍卖专家")
    exp = ["拍卖重现：一", "拍卖重现：二", "拍卖重现：三"]
    eng = mk(0, lord_level=1)
    pick_talent(eng, "拍卖专家")
    seen = []
    for _ in range(13):
        eng.battle()
        eng.round_end()
        eng.round_start()
        seen.append(tuple(x for x in eng.s.hand if x.startswith("拍卖重现")))
    got = [x for x in eng.s.hand if x.startswith("拍卖重现")]
    if got:
        assert got == exp, f"按回合推进：拿到 {got}，应 {exp}"
        r4 = seen[2]                                           # 第 4 回合开始后
        assert r4 == ("拍卖重现：一",), f"第一次拍卖（第 4 回合开头）后应有 拍卖重现：一，实际 {r4}"
        return
    eng = mk(0)
    pick_talent(eng, "拍卖专家")
    for k, r in enumerate((4, 8, 12)):
        eng.s.round = r
        eng.emit("auction")
        got = [x for x in eng.s.hand if x.startswith("拍卖重现")]
        assert got == exp[:k + 1], f"按拍卖事件：第 {k + 1} 次拍卖后 {got}，应 {exp[:k + 1]}"


def t_虚构装备():
    """战斗开始时随机3名英雄临时获得1件普通装备（战斗效果）。只测'临时'：战后装备不残留、不进手牌。"""
    need("虚构装备")
    for seed in range(5):
        eng = mk(seed)
        pick_talent(eng, "虚构装备")
        us = [add(eng, quiet(n), i * 4) for i, n in enumerate((T2_HL_A, T2_ZL_A, T3_SF, T4_ZL_A))]
        eng.battle()
        assert all(not u.equips for u in us), f"seed{seed} 战后还穿着 {[u.equips for u in us]}"
        assert not eng.s.hand, f"seed{seed} 装备进了手牌 {eng.s.hand}"


# ================================================================ 运行
def t_消费返利():
    """每消耗15能量，获得1张加速成长。（消耗 = eng.spend：买牌 / 刷新 / 可购买天赋都走它；余数累计）"""
    eng = mk(0, energy=100)
    pick_talent(eng, "消费返利")
    assert "加速成长" not in eng.s.hand, "选天赋时不该给牌"
    for _ in range(4):
        assert eng.spend(3)
    assert eng.s.hand.count("加速成长") == 0, f"花 12 不到 15：{eng.s.hand}"
    eng.spend(3)
    assert eng.s.hand.count("加速成长") == 1, f"花 15 应得 1 张：{eng.s.hand}"
    eng.buy("墨子")                                        # 英雄牌 3 能量
    eng.s.free_refresh = 0
    for _ in range(6):
        eng.refresh_shop()                                  # 每次 2 能量
    assert eng.s.hand.count("加速成长") == 2, f"再花 3+12=15 应又得 1 张：{eng.s.hand}"
    assert eng.s.energy == 100 - 15 - 15, f"能量 {eng.s.energy}"


TESTS = [
    ("里程碑", t_里程碑), ("消费返利", t_消费返利),
    ("三分新援", _xinyuan("三分新援", "三分之地")), ("日落海新援", _xinyuan("日落海新援", "日落海")),
    ("大河新援", _xinyuan("大河新援", "大河流域")), ("逐鹿新援", _xinyuan("逐鹿新援", "逐鹿")),
    ("河洛新援", _xinyuan("河洛新援", "河洛")),
    ("整装待发", t_整装待发), ("先苦后甜", t_先苦后甜), ("存款利息", t_存款利息),
    ("二象性", _gives("二象性", ["二象坍缩"])), ("二象坍缩", t_二象坍缩),
    ("预言家", _gives("预言家", ["胜利预言", "失败预言"])), ("胜利预言", t_胜利预言), ("失败预言", t_失败预言),
    ("三十而立", t_三十而立),
    ("好聚好散", _gives("好聚好散", ["体面离场"])), ("体面离场", t_体面离场),
    ("瓶中人", _gives("瓶中人", ["契约瓶"])), ("契约瓶", t_契约瓶),
    ("战争头牌", t_战争头牌), ("随机成长", _随机成长("随机成长", 4)), ("双人游", t_双人游),
    ("班叔的遗留物", _gives("班叔的遗留物", ["2级电磁弹簧"])),
    ("银算盘", _算盘("银算盘", 2)), ("五福临门", t_五福临门),
    ("小电池包", _energy_talent("小电池包", 5)), ("复印机", _copier("复印机", 3)),
    ("一个响指", _gives("一个响指", ["听天由命"])), ("听天由命", t_听天由命),
    ("终极复印机", _copier("终极复印机", 5)), ("扩编名额", t_扩编名额), ("补给线：开团", t_补给线开团),
    ("常胜将军", t_常胜将军), ("中坚力量", t_中坚力量), ("如法炮制", t_如法炮制), ("凯旋征召", t_凯旋征召),
    ("金算盘", _算盘("金算盘", 3)), ("电池包", _energy_talent("电池包", 8)), ("结伴", t_结伴),
    ("人才市场", _gives("人才市场", ["人才搜寻"] * 3)), ("人才搜寻", t_人才搜寻), ("同路人", t_同路人),
    ("成年礼", t_成年礼), ("大电池包", _energy_talent("大电池包", 10)), ("限制性进化", t_限制性进化),
    ("援军", t_援军), ("群英荟萃", t_群英荟萃), ("自适应武装", t_自适应武装), ("虚构装备", t_虚构装备),
    ("战术模组", t_战术模组), ("极限操作", t_极限操作), ("控能达人", t_控能达人), ("抢攻", t_抢攻),
    ("神秘英雄", t_神秘英雄), ("随机进化", _随机成长("随机进化", 6)), ("厉兵秣马", t_厉兵秣马),
    ("擂台猛虎", t_擂台猛虎), ("拍卖专家", t_拍卖专家),
    ("拍卖重现：一/二/三", t_拍卖重现),
]
COVERED_EXTRA = {"拍卖重现：一", "拍卖重现：二", "拍卖重现：三"}


def main() -> int:
    batch = json.loads((ROOT / "data/engine/sim_batches.json").read_text(encoding="utf-8"))["batches"][3]
    covered = {n for n, _ in TESTS} | COVERED_EXTRA | set(COMBAT_ONLY) | set(NOT_TESTABLE)
    missing = [n for n in batch if n not in covered]
    ok = fail = 0
    for name, fn in TESTS:
        try:
            for n in (COVERED_EXTRA if "/" in name else [name]):
                if n not in card_db():
                    raise AssertionError(f"卡库里没有 {n}")
            fn()
            print(f"OK    {name}")
            ok += 1
        except (Unimpl, AssertionError) as e:
            print(f"FAIL  {name}: {e}")
            fail += 1
        except Exception as e:                                 # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()
            print(f"FAIL  {name}: 异常 {type(e).__name__}: {e} @ {tb[-3].strip() if len(tb) >= 3 else ''}")
            fail += 1
    for n in COMBAT_ONLY:
        print(f"SKIP  {n}: 只有战斗效果")
    for n, why in NOT_TESTABLE.items():
        print(f"SKIP  {n}: {why}")
    if missing:
        print(f"注意：本批这些卡没有归类 {missing}")
    print(f"\n通过 {ok} / 失败 {fail}（共 {ok + fail} 条测试，跳过 {len(COMBAT_ONLY) + len(NOT_TESTABLE)} 张）")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
