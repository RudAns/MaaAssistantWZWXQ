"""2026-10-19 推演器卡牌修正的回归测试：按 data/cards.json 卡面（10-18 晚重爬）+ 10-19 卡面机制确认。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_card_reread_1019.py
逐条打印 OK / FAIL，末尾打印通过数 / 失败数，有失败退出码 1。
依据：推演器和卡面不一致的条目清单。
"""
from __future__ import annotations

import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.engine import board as B                      # noqa: E402
import agent.sim.engine as _E                            # noqa: E402
from agent.sim.engine import Engine, impl                # noqa: E402
from agent.sim.state import State, Unit                  # noqa: E402

_E.EQUIP_DROP_ROUNDS = ()
C = B.CELLS
HL1, HL2, HL3, HL4 = "沈梦溪", "百里守约", "狄仁杰", "苏烈"   # 河洛（苏烈只有牺牲钩子）
ZL1, ZL2 = "墨子", "廉颇"                                # 逐鹿，只有战斗效果
NEU = "苍"                                               # 无阵营，只有战斗效果


def mk(seed=0, **kw) -> Engine:
    return Engine(State(**kw), random.Random(seed))


def add(eng, name, cell, level=1.0, kind="hero", awaken=0, equips=None) -> Unit:
    u = Unit(name=name, level=level, cell=C[cell], kind=kind, pop=(kind == "hero"), uid=eng.s.next_uid, awaken=awaken)
    eng.s.next_uid += 1
    if equips:
        u.equips = list(equips)
    eng.s.units.append(u)
    return u


def snap(eng):
    return {u.uid: u.level for u in eng.s.units}


def gain(eng, s0, u):
    return u.level - s0.get(u.uid, 0)


def recorder(eng, pick=0):
    calls = []

    def ch(e, kind, options, **ctx):
        calls.append((kind, list(options)))
        return options[pick] if options else None
    eng.chooser = ch
    return calls


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ------------------------------------------------------------------ 觉醒改成"所有英雄 +1"，觉醒单位触发算 2 次
@test
def 小乔_觉醒所有英雄加1_觉醒登场算两次():
    eng = mk(0)
    add(eng, "小乔", 0, level=5, awaken=3)
    fs = [add(eng, ZL1, i + 1, level=5) for i in range(4)]
    s0 = snap(eng)
    eng.deploy("孙尚香", cell=C[6])                    # 有"登场："、没觉醒 → 1 次
    assert all(gain(eng, s0, f) == 1 for f in fs), [gain(eng, s0, f) for f in fs]
    sx = add(eng, "孙尚香", 7, awaken=3)
    s0 = snap(eng)
    eng.trigger(sx, "deploy")                         # 觉醒英雄的登场 = 2 次（10-19）
    assert all(gain(eng, s0, f) == 2 for f in fs), [gain(eng, s0, f) for f in fs]


@test
def 小乔_未觉醒随机3名():
    eng = mk(1)
    add(eng, "小乔", 0, level=5)
    fs = [add(eng, ZL1, i + 1, level=5) for i in range(5)]
    s0 = snap(eng)
    sx = eng.deploy("孙尚香", cell=C[7])
    tot = sum(gain(eng, s0, u) for u in eng.heroes() if u is not sx) + (sx.level - 1)
    assert tot == 3, tot


@test
def 张良_觉醒所有英雄加1():
    eng = mk(0)
    add(eng, "张良", 0, level=5, awaken=3)
    y = add(eng, "嬴政", 1, level=5)
    fs = [add(eng, ZL1, i + 2, level=5) for i in range(3)]
    s0 = snap(eng)
    eng.fire(y, "open")
    assert all(gain(eng, s0, u) == 1 for u in eng.heroes()), [gain(eng, s0, u) for u in eng.heroes()]


@test
def 张良_觉醒开团单位算两次():
    eng = mk(0)
    add(eng, "张良", 0, level=5)
    y = add(eng, "嬴政", 1, level=5, awaken=3)
    [add(eng, ZL1, i + 2, level=5) for i in range(4)]
    s0 = snap(eng)
    eng.fire(y, "open")
    assert sum(gain(eng, s0, u) for u in eng.heroes()) == 6


@test
def 朵莉亚_觉醒全员加1核心加1():
    eng = mk(0)
    add(eng, "朵莉亚", 0, level=5, awaken=3)
    k = add(eng, "狂铁", 1, level=5)
    fs = [add(eng, ZL1, i + 2, level=5) for i in range(4)]
    s0, c0 = snap(eng), eng.s.core
    eng.fire(k, "prepare")
    assert all(gain(eng, s0, u) == 1 for u in eng.heroes()), [gain(eng, s0, u) for u in eng.heroes()]
    assert eng.s.core - c0 == 3, eng.s.core - c0      # 狂铁 +2、朵莉亚 +1


@test
def 韩信_觉醒整备算两次():
    eng = mk(0)
    h = add(eng, "韩信", 0, level=5)
    k = add(eng, "狂铁", 1, level=5, awaken=3)
    eng.fire(k, "prepare")
    assert h.level == 5 + 4, h.level


# ------------------------------------------------------------------ 觉醒给 1 张高阶牌
@test
def 庄周_觉醒加4与1张高阶招募():
    eng = mk(0)
    u = add(eng, "庄周", 0, level=5, awaken=3)
    eng.fire(u, "deploy")
    assert u.level == 9 and eng.s.hand == ["高阶招募战术"], (u.level, eng.s.hand)


@test
def 蒙犽_西施_老夫子_白起_觉醒高阶():
    eng = mk(0)
    u = add(eng, "蒙犽", 0, awaken=3)
    eng.fire(u, "deploy")
    assert eng.s.hand == ["高阶特训战术"], eng.s.hand
    eng = mk(0)
    u = add(eng, "西施", 0, awaken=3)
    eng.fire(u, "prepare")
    assert eng.s.hand == ["高阶推演战术"], eng.s.hand
    eng = mk(0)
    add(eng, "老夫子", 0, awaken=3)
    h = add(eng, ZL1, 1)
    eng.merge(h)
    assert eng.s.hand.count("高阶招募战术") == 1 and "招募战术" not in eng.s.hand, eng.s.hand
    eng = mk(0)
    add(eng, "白起", 0, awaken=3)
    h = add(eng, ZL1, 1)
    eng.use_card("特训战术", h)
    assert eng.s.hand == ["高阶援护战术"], eng.s.hand


@test
def 鲁班七号_觉醒每3次合成1张高阶():
    for seed in range(5):
        eng = mk(seed)
        add(eng, "鲁班七号", 0, awaken=3)
        h = add(eng, ZL1, 1)
        for _ in range(3):
            eng.merge(h)
        got = [x for x in eng.s.hand if "战术" in x]
        assert len(got) == 1 and got[0].startswith("高阶"), eng.s.hand


# ------------------------------------------------------------------ 古币类：觉醒 1 张精致古币；武则天觉醒 +2；自动用掉的古币算效果牌
@test
def 程咬金_觉醒精致古币一次_算效果牌():
    eng = mk(0)
    u = add(eng, "程咬金", 0, level=5, awaken=3)
    j = add(eng, "镜", 5, level=5)
    eng.fire(u, "prepare")
    assert u.level == 9, u.level                      # +4，不是两次 +2
    assert j.counters.get("镜:cards") == 1, j.counters   # 自动用的古币算使用效果牌（10-19）


@test
def 李元芳_觉醒精致古币():
    eng = mk(0)
    y = add(eng, "李元芳", 0, awaken=3)
    t = add(eng, HL1, 1, level=5)
    seen = []
    eng.chooser = None
    orig = eng.coin
    eng.coin = lambda tt, n=1, src="", fine=False, active=False: (seen.append(fine), orig(tt, n, src, fine, active))
    eng.fire(y, "prepare")
    assert seen == [True] and t.level == 9, (seen, t.level)


@test
def 盾山_觉醒牺牲得精致古币():
    eng = mk(0)
    u = add(eng, "盾山", 0, awaken=3)
    eng.fire(u, "sacrifice")
    assert eng.s.hand == ["精致古币"], eng.s.hand


@test
def 武则天_觉醒古币加2():
    eng = mk(0)
    add(eng, "武则天", 0, awaken=3)
    t = add(eng, HL1, 1, level=5)
    eng.use_card("古币", t)
    assert t.level == 5 + 2 + 2, t.level


@test
def 上官婉儿_觉醒6名不同河洛():
    for seed in range(5):
        eng = mk(seed)
        w = add(eng, "上官婉儿", 0, level=5, awaken=3)
        hs = [add(eng, n, i + 1, level=5) for i, n in enumerate((HL1, HL2, HL3, HL4))]
        s0 = snap(eng)
        eng.fire(w, "prepare")
        assert all(gain(eng, s0, u) == 2 for u in [w] + hs), [gain(eng, s0, u) for u in [w] + hs]


@test
def 花木兰_觉醒每5张仍1次复生():
    eng = mk(0)
    m = add(eng, "花木兰", 0, level=5, awaken=3)
    t = add(eng, HL1, 1)
    for _ in range(5):
        eng.use_card("古币", t)
    assert m.counters.get("revive") == 1, m.counters


# ------------------------------------------------------------------ 其他卡面数值
@test
def 杨玉环_觉醒妙音():
    eng = mk(0)
    u = add(eng, "杨玉环", 0, awaken=3)
    eng.sell(u)
    assert eng.s.hand == ["妙音"], eng.s.hand


@test
def 太乙与铠_觉醒完美复生只1次():
    eng = mk(0)
    ty = add(eng, "太乙真人", 0, awaken=3)
    t = add(eng, ZL1, 1)
    eng.fire(ty, "open")
    assert t.counters.get("revive") == 1, t.counters
    eng = mk(0)
    k = add(eng, "铠", 0, awaken=3)
    h = add(eng, HL1, 1)
    eng.fire(k, "open")
    assert k.counters.get("revive") == 1, k.counters
    eng.fire(k, "sacrifice")
    assert h.temp == 10, h.temp


@test
def 公孙离_孙悟空_猪八戒_觉醒只夺1次():
    for name in ("公孙离", "孙悟空", "猪八戒"):
        eng = mk(0)
        nb = add(eng, HL1, 8, level=40)
        u = add(eng, name, 9, awaken=3)
        eng.fire(u, "deploy")
        assert nb.level == 30, (name, nb.level)


@test
def 孙悟空_觉醒整备加4():
    eng = mk(0)
    u = add(eng, "孙悟空", 0, level=5, awaken=3)
    eng.fire(u, "prepare")
    assert u.level == 9, u.level


@test
def 姬小满_莱西奥_觉醒阈值():
    eng = mk(0)
    add(eng, "姬小满", 0, awaken=3)
    hs = [add(eng, ZL1, i + 1) for i in range(2)]
    for h in hs:
        eng.sell(h)
    assert eng.s.free_refresh == 1, eng.s.free_refresh
    eng = mk(0)
    add(eng, "莱西奥", 0, awaken=3)
    add(eng, "狂铁", 2)                                # 有日落海，核心才有意义
    h = add(eng, ZL1, 1)
    c0 = eng.s.core
    eng.sell(h)
    assert eng.s.core - c0 == 1, eng.s.core - c0


@test
def 梦奇_觉醒2张券_败阵仍加1阶():
    eng = mk(0)
    u = add(eng, "梦奇", 0, awaken=3)
    eng.fire(u, "lose")
    eng.sell(u)
    assert eng.s.hand == ["2阶英雄兑换券", "2阶英雄兑换券"], eng.s.hand


@test
def 甄姬_觉醒3选1():
    eng = mk(0, lord_level=5)
    calls = recorder(eng)
    u = add(eng, "甄姬", 0, awaken=3)
    eng.fire(u, "deploy")
    assert len(eng.s.hand) == 1 and calls and len(calls[0][1]) == 3, (eng.s.hand, calls)


@test
def 虞姬_觉醒自身不翻倍_随机4名():
    for seed in range(5):
        eng = mk(seed)
        y = add(eng, "虞姬", 0, level=5, awaken=3)
        add(eng, "先知图腾", 20, kind="summon")
        ds = [add(eng, "瑶", i + 1, level=5) for i in range(5)]
        s0 = snap(eng)
        eng.fire(y, "open")
        assert gain(eng, s0, y) == 1, gain(eng, s0, y)
        assert sorted(gain(eng, s0, x) for x in ds) == [0, 1, 1, 1, 1], [gain(eng, s0, x) for x in ds]


@test
def 雅典娜_觉醒4名不同():
    for seed in range(5):
        eng = mk(seed)
        a = add(eng, "雅典娜", 0, awaken=3)
        ts = [add(eng, n, i + 1) for i, n in enumerate(("狂铁", "安琪拉", "亚连", "米莱狄"))]
        hit = []
        orig = eng.trigger
        eng.trigger = lambda u, timing, **kw: (hit.append(u.uid), orig(u, timing, **kw))
        eng.fire(a, "open")
        assert sorted(hit) == sorted(t.uid for t in ts), hit


@test
def 传承_平均分最低3名留1级():
    eng = mk(0)
    src = add(eng, ZL1, 0, level=31)
    lows = [add(eng, ZL2, i + 1, level=i + 1) for i in range(3)]
    hi = add(eng, NEU, 5, level=50)
    s0 = snap(eng)
    eng.use_card("传承", src)
    assert src.level == 1 and [gain(eng, s0, x) for x in lows] == [10, 10, 10] and hi.level == 50, \
        (src.level, [gain(eng, s0, x) for x in lows], hi.level)


@test
def 夺取之术_只限周围1格():
    eng = mk(0)
    t = add(eng, ZL1, 9, level=5)
    nb = add(eng, ZL2, 8, level=30)
    far = add(eng, NEU, 27, level=50)
    eng.use_card("夺取之术", t)
    assert t.level == 15 and nb.level == 20 and far.level == 50, (t.level, nb.level, far.level)
    eng = mk(0)
    t = add(eng, ZL1, 9, level=5)
    far = add(eng, NEU, 27, level=50)
    eng.use_card("夺取之术", t)
    assert t.level == 5 and far.level == 50, (t.level, far.level)


@test
def 姜子牙_无阵营不算阵营():
    for seed in range(5):
        eng = mk(seed)
        j = add(eng, "姜子牙", 0, level=5)
        z = add(eng, ZL1, 1, level=5)
        n = add(eng, NEU, 2, level=5)
        u = eng.deploy("弈星", cell=C[3])
        assert j.level == 6 and z.level == 6 and n.level == 5 and u.level == 1, (j.level, z.level, n.level, u.level)


@test
def 化铁为金_变身后触发登场():
    eng = mk(0)
    t = add(eng, ZL1, 0, level=7)
    hit = []
    orig = eng.trigger
    eng.trigger = lambda u, timing, **kw: (hit.append((u.uid, timing)), orig(u, timing, **kw))
    ok = eng.use_card("化铁为金", t)
    assert ok and t.awakened and hit == [(t.uid, "deploy")], (ok, t.awakened, hit)


@test
def 召唤物能穿装备_图腾不能():
    eng = mk(0)
    p = add(eng, "机关傀儡", 0, kind="summon")
    tt = add(eng, "先知图腾", 1, kind="summon")
    assert eng.can_wear(p, "影牙") and not eng.can_wear(tt, "影牙")


@test
def 鲁班大师_傀儡等级同步():
    eng = mk(0)
    m = eng.deploy("鲁班大师", cell=C[0])
    pup = next(x for x in eng.s.units if x.name == "机关傀儡")
    eng.level(m, 20)
    impl("鲁班大师").on_battle_begin(eng, m)
    assert pup.level == m.level, (pup.level, m.level)


@test
def 即刻起舞_固定三选一_惊鸿游龙不送特性():
    eng = mk(0, lord_cards=["枫叶舞"])
    calls = recorder(eng)
    eng.use_card("即刻起舞", None)
    assert sorted(calls[0][1]) == sorted(["落英汲取", "枫之洪流", "翩跹馈赠"]), calls
    eng = mk(0, lord_cards=["枫叶舞", "惊鸿游龙"])
    recorder(eng)
    a = add(eng, HL1, 3, level=10)
    eng.bind_maple(a)
    eng.fire(a, "kill", n=40)
    assert not {"落英汲取", "枫之洪流", "翩跹馈赠"} & set(eng.s.lord_cards), eng.s.lord_cards


@test
def 先知图腾_不限阵营():
    eng = mk(0)
    t = add(eng, "先知图腾", 9, kind="summon")
    z = add(eng, ZL1, 8)
    eng.fire(t, "open")
    assert "玄微之种" in z.equips, z.equips


def main() -> int:
    ok = fail = 0
    for fn in TESTS:
        try:
            fn()
            ok += 1
            print(f"OK    {fn.__name__}")
        except Exception as e:                          # noqa: BLE001
            fail += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__} {e}")
            traceback.print_exc(limit=2)
    print(f"通过 {ok} / 失败 {fail}（共 {len(TESTS)} 条）")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
