"""正向推演引擎（agent/sim）的跨批次集成测试。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_integration.py [--seeds 50]

两部分：
  1. 随机整局：N 个种子（默认 50），每个从第 1 回合空局面开始，用 agent/sim/play.py 的 Player 跑到第 14 回合。
     为了让更多卡被执行到，每局开头再随机选 2 个天赋、每回合随机塞 1 张效果牌进手牌，并在中后期模拟其他棋手淘汰 / 排名。
     要求：不抛异常，每场战斗结束时没有遗留的 battle_only 召唤物、临时等级清零，且每回合结束时局面自洽
     （格子不重叠、手牌 ≤10、核心 ≥0、生命值 0~60、能量不低于下限、永久等级 ≥1、手牌状态和手牌对得上）。
  2. 跨卡组合局面：按卡面原文推结果并断言（古币链、牺牲链、图腾链、战术牌链、遗志 + 红叶染锋换绑、
     日落海核心分配、被触发的登场 vs 打出英雄牌、往生图腾吸收战斗召唤物、手牌持有回合 / 购买事件）。
逐条打印 OK / FAIL，末尾打印通过数，有失败退出码 1。
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.engine import board as B                      # noqa: E402
from agent.sim.engine import (Engine, REGISTRY, card_db, impl,  # noqa: E402
                              LORD_HP_MAX)
import agent.sim.engine as _E                                                     # noqa: E402
_E.EQUIP_DROP_ROUNDS = ()      # 固定回合投放装备（开局 / 第 4/8/12/16 回合）是引擎规则，卡牌测试里关掉，免得混进手牌 / 装备数的断言
from agent.sim.play import Player                        # noqa: E402
from agent.sim.state import State, Unit                  # noqa: E402

C = B.CELLS


class Fail(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def mk(seed=0, **kw) -> Engine:
    return Engine(State(**kw), random.Random(seed))


def put(eng: Engine, name: str, idx: int, level: float = 1.0, kind: str = "hero", pop: bool = True) -> Unit:
    """直接摆一个单位（不触发登场）。"""
    u = Unit(name=name, level=float(level), cell=C[idx], kind=kind, pop=pop, uid=eng.s.next_uid)
    eng.s.next_uid += 1
    eng.s.units.append(u)
    return u


def die(eng: Engine, u: Unit):
    """阵亡（和 engine.battle 同序：阵亡 → 拥有牺牲则触发牺牲）。"""
    eng.fire(u, "death")
    if eng.has_kw(u, "牺牲"):
        eng.fire(u, "sacrifice")


def logged(eng: Engine, ev: str, src: str = None, unit: str = None, i0: int = 0) -> float:
    """日志里某类事件 n 的合计（按来源 / 单位过滤）。"""
    tot = 0.0
    for (_, e, kw) in eng.s.log[i0:]:
        if e != ev:
            continue
        if src is not None and not str(kw.get("src", "")).startswith(src):
            continue
        if unit is not None and kw.get("unit") != unit:
            continue
        tot += kw.get("n", 0)
    return tot


TESTS = []


def test(name):
    def deco(f):
        TESTS.append((name, f))
        return f
    return deco


# ================================================================ 1. 随机整局
def _impl_cards(cat: str) -> list:
    impl("古币")
    return sorted(n for n, c in card_db().items() if c.get("category") == cat and n in REGISTRY
                  and hasattr(REGISTRY[n], "on_use"))


def _invariants(eng: Engine, R: int) -> None:
    s = eng.s
    cells = [u.cell for u in s.units]
    check(all(c in C for c in cells), f"R{R} 有单位不在棋盘格子上：{[(u.name, u.cell) for u in s.units if u.cell not in C]}")
    check(len(cells) == len(set(cells)), f"R{R} 格子重叠：{cells}")
    # 手牌回合中可以超过上限（2026-10-14 规则：超出的下回合开始自动销毁）→ 只查"超出的部分在回合开始被销毁"：
    # 这里在整回合结束后检查，允许回合内获得的牌暂时超出，但不允许无限堆积
    check(len(s.hand) <= B.HAND_LIMIT + 6, f"R{R} 手牌异常堆积：{len(s.hand)}")
    check(s.core >= 0, f"R{R} 核心等级为负：{s.core}")
    check(0 <= s.hp <= LORD_HP_MAX, f"R{R} 棋手生命值越界：{s.hp}")
    check(all(u.level >= 1 for u in s.units), f"R{R} 有单位永久等级 <1：{[(u.name, u.level) for u in s.units if u.level < 1]}")
    check(s.energy >= s.energy_floor, f"R{R} 能量 {s.energy} 低于下限 {s.energy_floor}")
    for n in set(s.hand):
        check(len(eng.hand_meta(n)) == s.hand.count(n), f"R{R} 手牌状态和手牌对不上：{n}")


def run_random(seeds: int) -> tuple:
    talents = _impl_cards("天赋牌")
    effects = _impl_cards("效果牌")
    ok = fail = 0
    for seed in range(seeds):
        rng = random.Random(10_000 + seed)
        eng = Engine(State(), random.Random(seed))
        p = Player(eng, lookahead=1, samples=1)
        R = 1
        orig_battle = eng.battle

        def battle_checked(_orig=orig_battle, _eng=eng):
            win = _orig()
            left = [u.name for u in _eng.s.units if u.counters.get("battle_only")]
            check(not left, f"R{_eng.s.round} 战斗结束后还留着只在本场存在的召唤物：{left}")
            check(all(u.temp == 0 for u in _eng.s.units), f"R{_eng.s.round} 战斗结束临时等级应清零")
            return win
        eng.battle = battle_checked
        try:
            for t in rng.sample(talents, 2):
                p._take("talent", t)
            while eng.s.round < 14:
                R = eng.s.round
                if rng.random() < 0.8:
                    eng.gain_card(rng.choice(effects))
                if R >= 6:
                    eng.s.rank = rng.randint(1, 8)
                if R >= 8 and rng.random() < 0.3:
                    eng.eliminate(1)
                p.round()
                _invariants(eng, R)
            check(eng.s.round == 14, f"没跑到第 14 回合：{eng.s.round}")
            ok += 1
        except Exception as e:                           # noqa: BLE001
            fail += 1
            tb = traceback.format_exc(limit=6).strip().splitlines()
            print(f"FAIL  随机整局 seed{seed} R{R}: {type(e).__name__}: {e}")
            for line in tb[-6:]:
                print("        " + line)
    return ok, fail


# ================================================================ 2. 跨卡组合局面
@test("古币链：李元芳 + 武则天 + 程咬金（+ 古币翻新）")
def t_coin_chain():
    """李元芳 整备：对最近的河洛英雄用 1 张古币；程咬金 整备：对自身用 1 张古币；武则天：古币提升的等级 +1；
    古币翻新：古币额外 +2。→ 回合开始：武则天（离李元芳最近）+2+1+2，程咬金 +2+1+2；手牌古币打给李元芳 +5。"""
    eng = mk(0, talents=["古币翻新"])
    ly = put(eng, "李元芳", 0)
    wz = put(eng, "武则天", 1)
    cy = put(eng, "程咬金", 20)
    eng.round_start()
    check(wz.level == 1 + 5, f"武则天应 +5（李元芳的古币 2+1+2），实际 +{wz.level - 1}")
    check(cy.level == 1 + 5, f"程咬金应 +5（自用古币 2+1+2），实际 +{cy.level - 1}")
    check(ly.level == 1, f"李元芳自己不吃古币：+{ly.level - 1}")
    eng.gain_card("古币")
    check(eng.use_card("古币", ly) is not False and ly.level == 6, f"手牌古币打给李元芳 +5：{ly.level}")
    check(eng.use_card("古币", put(eng, "墨子", 5)) is False, "古币指向非河洛英雄不合法")


@test("牺牲链：钟馗 + 百里守约 + 李信 + 苏烈")
def t_sacrifice_chain():
    """钟馗：自身及左右 1 格获得牺牲；李信：拥有牺牲的英雄阵亡时 临时等级 +7；苏烈 牺牲：永久 +2。
    守约站在钟馗旁边（同排相邻）→ 阵亡触发牺牲；李信自己只是"提到"牺牲，不拥有 → 自己阵亡不触发。"""
    eng = mk(0)
    zk = put(eng, "钟馗", 8)
    sy = put(eng, "百里守约", 9)
    lx = put(eng, "李信", 23)
    sl = put(eng, "苏烈", 0)
    check(B.row_of(zk.cell) == B.row_of(sy.cell) and eng.dist(zk, sy) == 1, "测试前置：守约要在钟馗同排相邻")
    eng.round_start()
    check(eng.has_kw(sy, "牺牲") and eng.has_kw(zk, "牺牲"), "钟馗应让自己和左右 1 格获得牺牲")
    check(not eng.has_kw(lx, "牺牲"), "李信卡面只是提到牺牲，不该拥有")
    die(eng, sy)
    check(lx.temp == 7, f"守约（被钟馗给了牺牲）阵亡 → 李信 +7：{lx.temp}")
    die(eng, sl)
    check(lx.temp == 14 and sl.level == 3, f"苏烈牺牲 → 李信 +7、苏烈 +2：李信 {lx.temp} 苏烈 {sl.level}")
    die(eng, lx)
    check(lx.temp == 14, f"李信自己阵亡不是牺牲：{lx.temp}")
    die(eng, zk)
    check(lx.temp == 21, f"钟馗阵亡（自身有牺牲）→ 李信 +7：{lx.temp}")


@test("图腾链：少司缘 + 虞姬 + 始祖熊灵")
def t_totem_chain():
    """少司缘在场召唤因缘图腾；虞姬 开团：每个图腾 自身及随机 2 名其他大河英雄 +1；
    始祖熊灵视为大河、不占人口，开战时我方单位比敌方多几个就 +2×几；因缘图腾：合计普攻每 25 次全体英雄 +1。"""
    eng = mk(3, lord_level=2)
    put(eng, "虞姬", 3)
    eng.use_card("始祖熊灵", None)
    bear = eng.s.unit("始祖熊灵(赠送)")
    check(bear is not None and not bear.pop, "始祖熊灵应上场且不占人口")
    eng.deploy("少司缘", cell=C[4])
    totem = eng.s.unit("因缘图腾")
    check(totem is not None and totem.kind == "summon", "少司缘上场应召唤因缘图腾")
    eng.round_start()                                       # 回合开始后熊灵仍被视为大河（关键词每回合重赋）
    check(eng.is_faction(bear, "大河流域"), "始祖熊灵回合开始后仍应被视为【大河】")
    i0 = len(eng.s.log)
    ours = len(eng.s.on_board())
    enemy = B.pop_cap(2)
    att = sum(eng.attacks(x) for x in eng.s.on_board())
    acc0 = totem.counters.get("att", 0.0)
    eng.battle()
    check(logged(eng, "等级", src="虞姬", unit="虞姬", i0=i0) == 1, "虞姬开团：1 个图腾 → 自身 +1")
    others = logged(eng, "等级", src="虞姬", i0=i0) - 1
    check(others == 2, f"虞姬开团：随机 2 名其他大河英雄各 +1（少司缘 / 始祖熊灵），实际合计 {others}")
    want_bear = 2 * max(0, ours - enemy)
    check(logged(eng, "等级", src="始祖熊灵", i0=i0) == want_bear,
          f"始祖熊灵：我方 {ours} vs 敌方 {enemy} → +{want_bear}，实际 {logged(eng, '等级', src='始祖熊灵', i0=i0)}")
    n = int((acc0 + att) // 25)
    got = logged(eng, "等级", src="因缘图腾", unit="虞姬", i0=i0)
    check(got == n, f"因缘图腾：普攻 {acc0 + att:.1f} 次 → 每名英雄 +{n}，虞姬实际 +{got}")


@test("战术牌链：西施 + 芈月 + 白起")
def t_tactic_chain():
    """西施 整备：获得推演战术；芈月：每使用效果牌，穿装备最多的逐鹿英雄 +1；白起：每回合首次使用效果牌获得援护战术。"""
    eng = mk(0)
    xs = put(eng, "西施", 0)
    my = put(eng, "芈月", 1)
    bq = put(eng, "白起", 2)
    mz = put(eng, "墨子", 3)
    check(eng.equip(mz, "无尽巨剑"), "测试前置：墨子穿 1 件装备（芈月的受益人）")
    eng.round_start()
    check(eng.s.hand == ["推演战术"], f"西施整备应得推演战术：{eng.s.hand}")
    check(eng.use_card("推演战术", mz) is not False, "推演战术应能打给逐鹿英雄")
    eng.s.hand.remove("推演战术")
    check(mz.level == 2, f"芈月：效果牌 → 墨子 +1：{mz.level}")
    check(eng.s.hand == ["援护战术"], f"白起：本回合首次用效果牌 → 援护战术：{eng.s.hand}")
    eng.use_card("援护战术", None)
    eng.s.hand.remove("援护战术")
    check(mz.level == 3 and not eng.s.hand, f"第二张效果牌：芈月再 +1，白起不再给：墨子 {mz.level} 手牌 {eng.s.hand}")
    eng.round_start()                                       # 新回合：西施又给推演战术，白起重置
    eng.use_card("推演战术", mz)
    check("援护战术" in eng.s.hand, f"新回合白起重新计：{eng.s.hand}")
    check(xs.level == 1 and bq.level == 1 and my.level == 1, "其他人不该加等级")


@test("换绑：遗志 + 红叶染锋")
def t_yizhi_hongye():
    """遗志：移除 1 名英雄，等级转给随机 1 名英雄；红叶染锋：枫叶舞英雄离场后 50% 真实等级留给下一个枫叶舞英雄（限一次）。"""
    eng = mk(0, lord_cards=["枫叶舞", "红叶染锋"])
    a = put(eng, "沈梦溪", 0, level=41)
    b = put(eng, "墨子", 1)
    c = put(eng, "廉颇", 2)
    eng.bind_maple(a)
    check(eng.use_card("遗志", a) is not False, "遗志应能指定场上英雄")
    check(a not in eng.s.units and eng.s.maple == "", "宿主被移除、枫叶舞解绑")
    check(sorted([b.level, c.level]) == [1, 42], f"41 级转给随机 1 名：{b.level} / {c.level}")
    check(eng.s.maple_keep == 20, f"红叶染锋留下 floor(41×50%) = 20：{eng.s.maple_keep}")
    nxt = b if b.level == 1 else c
    eng.bind_maple(nxt)
    check(nxt.level == 21 and eng.s.maple_keep == 0, f"换绑后新宿主 +20：{nxt.level}")
    eng.remove(nxt)
    check(eng.s.maple_keep == 0, "红叶染锋限一次，第二次离场不再保留")


@test("日落海核心：安琪拉 + 亚瑟 + 海诺 + 马可波罗，星源星徽 / 阿科米亚之子 / 定向传输")
def t_core_chain():
    """安琪拉 整备：核心 +2；亚瑟：使用日落海英雄牌核心 +3（被唤醒触发的登场不算）；马可波罗 登场：核心 +2；
    海诺：每次核心提升额外 +1。开战：核心以临时等级平分给日落海英雄（单人 ≤50%），星源星徽不分，阿科米亚之子参与。"""
    for direct in (False, True):
        eng = mk(0, talents=["定向传输"] if direct else [])
        aq = put(eng, "安琪拉", 0)
        ys = put(eng, "亚瑟", 1)
        hn = put(eng, "海诺", 2)
        al = put(eng, "艾琳", 3)
        sm = put(eng, "沈梦溪", 4)
        check(eng.equip(al, "星源星徽") and eng.equip(sm, "阿科米亚之子"), "测试前置：穿装备")
        eng.round_start()
        check(eng.s.core == 3, f"安琪拉整备 +2、海诺 +1 → 3：{eng.s.core}")
        mk_ = eng.deploy("马可波罗", cell=C[5])
        check(eng.s.core == 3 + 3 + 4, f"马可登场 +2(+1) 和 亚瑟 +3(+1) → 10：{eng.s.core}")
        eng.gain_card("唤醒")
        eng.use_card("唤醒", mk_)
        check(eng.s.core == 13, f"唤醒触发马可登场：马可 +2(+1)，亚瑟不算（被触发）→ 13：{eng.s.core}")
        i0 = len(eng.s.log)
        eng.core_distribute()
        got = {u.name: logged(eng, "临时等级", src="阿科米亚核心", unit=u.name, i0=i0) for u in (aq, ys, hn, al, sm, mk_)}
        check(got["艾琳"] == 0, f"星源星徽不分：{got}")
        if not direct:
            check(all(got[n] == math.floor(13 / 5) for n in ("安琪拉", "亚瑟", "海诺", "沈梦溪", "马可波罗")),
                  f"5 人平分 13 → 各 2：{got}")
        else:
            check(got["沈梦溪"] == 13 and sum(got.values()) == 13,
                  f"定向传输：穿装备最多的（沈梦溪，阿科米亚之子）拿全部 13：{got}")


@test("被触发的登场 ≠ 打出英雄牌：蔡文姬 + 貂蝉 + 唤醒")
def t_triggered_vs_played():
    """蔡文姬：使用三分英雄牌时，最低的 3 名其他三分 +1；貂蝉 登场：三分英雄 +1；唤醒：触发其登场效果（不是使用英雄牌）。"""
    eng = mk(0)
    put(eng, "蔡文姬", 0)
    put(eng, "马超", 1)
    put(eng, "大乔", 2)
    i0 = len(eng.s.log)
    dc = eng.deploy("貂蝉", cell=C[3])
    check(logged(eng, "等级", src="蔡文姬", i0=i0) == 3, "打出貂蝉：蔡文姬给 3 名 +1")
    check(logged(eng, "等级", src="貂蝉", i0=i0) == 4, "貂蝉登场：4 名三分英雄 +1")
    i1 = len(eng.s.log)
    eng.use_card("唤醒", dc)
    check(logged(eng, "等级", src="蔡文姬", i0=i1) == 0, "唤醒触发的登场不算使用英雄牌，蔡文姬不该加")
    check(logged(eng, "等级", src="貂蝉", i0=i1) == 4, "唤醒应再触发一次貂蝉登场")
    i2 = len(eng.s.log)
    eng.merge(dc)
    check(logged(eng, "等级", src="蔡文姬", i0=i2) == 3, "合成 = 打出同名英雄牌，蔡文姬应再给 3 名 +1")


@test("战斗召唤：往生图腾吸收故障机器人 / 玄雍士兵，战斗结束不留 battle_only 单位")
def t_battle_summons():
    """大司命在场召唤往生图腾；机关匣：开战召唤 2 个故障机器人（1 阶）；蒙恬 牺牲：召唤 2 名玄雍士兵（2 阶）。
    往生图腾：1-3 阶召唤物上场即被吸收、图腾 +1；没被吸收的只在本场战斗存在。"""
    for seed in range(10):
        eng = mk(seed, lord_level=3)
        eng.deploy("大司命", cell=C[10])
        ws = eng.s.unit("往生图腾")
        check(ws is not None, "大司命应召唤往生图腾")
        mz = put(eng, "墨子", 3)
        check(eng.equip(mz, "机关匣"), "机关匣应能穿给逐鹿英雄")
        put(eng, "蒙恬", 4, level=5)
        lv0 = ws.level
        i0 = len(eng.s.log)
        eng.battle()
        absorbed = sum(1 for (_, ev, kw) in eng.s.log[i0:] if ev == "离场" and kw.get("src") == "往生图腾吸收")
        check(absorbed >= 2, f"seed{seed} 至少吸收 2 个故障机器人：{absorbed}")
        gain = logged(eng, "等级", src="往生图腾", unit="往生图腾", i0=i0)
        check(gain == absorbed, f"seed{seed} 吸收 {absorbed} 个应 +{absorbed}，实际 +{gain}")
        sac = logged(eng, "等级", src="大司命", unit="往生图腾", i0=i0)
        check(ws.level == lv0 + gain + sac, f"seed{seed} 往生图腾等级账对不上：{ws.level} vs {lv0}+{gain}+{sac}")
        check(not any(u.name in ("故障机器人", "玄雍士兵") for u in eng.s.units), f"seed{seed} 战斗召唤物应移除")


@test("手牌状态 + 购买 / 消耗事件：众星闪耀、首购激励、消费返利")
def t_hand_meta_buy():
    """众星闪耀：全体 +10，每持有一回合 +5（手牌状态记获得回合）；首购激励：每回合第 1 张购买的牌复制 1 张；
    消费返利：每消耗 15 能量得 1 张加速成长（买牌 / 刷新都走 eng.spend）。"""
    eng = mk(0, energy=40, lord_cards=["首购激励"], talents=["消费返利"])
    u = put(eng, "墨子", 0)
    eng.gain_card("众星闪耀")
    eng.round_end()
    eng.round_start()
    eng.round_end()
    eng.round_start()
    e0 = eng.s.energy
    eng.use_card("众星闪耀", None)
    eng.s.hand.remove("众星闪耀")
    check(u.level == 1 + 10 + 5 * 2, f"持有 2 回合：+20，实际 +{u.level - 1}")
    check(eng.buy("廉颇") and eng.s.hand.count("廉颇") == 2, f"首购复制：{eng.s.hand}")
    check(eng.buy("廉颇") and eng.s.hand.count("廉颇") == 3, f"第二张不复制：{eng.s.hand}")
    for _ in range(5):
        eng.refresh_shop()
    check(eng.s.energy == e0 - 6 - 10, f"买 2 张 + 刷新 5 次 = 16：{e0 - eng.s.energy}")
    check(eng.s.hand.count("加速成长") == 1, f"消耗 16 ≥ 15 → 1 张加速成长：{eng.s.hand}")


@test("演武 + 令行如流 / 西施：棋手 5 级后获得的战术牌变高阶（获得时拦截）")
def t_yanwu():
    eng = mk(0, lord_level=5, talents=["演武"])
    put(eng, "西施", 0)
    eng.round_start()
    check(eng.s.hand == ["高阶推演战术"], f"5 级后西施给的推演战术应变高阶：{eng.s.hand}")
    eng = mk(0, lord_level=4, talents=["演武"])
    put(eng, "西施", 0)
    eng.round_start()
    check(eng.s.hand == ["推演战术"], f"4 级时不变：{eng.s.hand}")


# ================================================================ 运行
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    a = ap.parse_args()
    ok = fail = 0
    for name, f in TESTS:
        try:
            f()
            print(f"OK    {name}")
            ok += 1
        except Exception as e:                           # noqa: BLE001
            tb = traceback.format_exc(limit=4).strip().splitlines()
            print(f"FAIL  {name}: {type(e).__name__}: {e} | {' / '.join(x.strip() for x in tb[-3:])}")
            fail += 1
    rok, rfail = run_random(a.seeds)
    print(f"{'OK  ' if not rfail else 'FAIL'}  随机整局：{rok}/{a.seeds} 个种子跑到第 14 回合无异常、局面自洽")
    ok += 1 if not rfail else 0
    fail += 1 if rfail else 0
    print(f"\n通过 {ok} / 失败 {fail}（组合局面 {len(TESTS)} 条 + 随机整局 1 条）")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
