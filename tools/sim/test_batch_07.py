"""第 7 批卡牌的独立测试：只按 data/cards.json 卡面原文写，不看 agent/sim/cards/batch_07.py 的实现。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_batch_07.py
逐条打印 OK / FAIL / SKIP，末尾打印通过数 / 失败数 / 跳过数；有失败时退出码 1。

约定（测试这边的假设，实现若不同会在这里暴露出来）：
- 搭局面时英雄一律"静音"（把英雄牌的实现从 REGISTRY 里摘掉），只测被测这张卡，免得别的英雄的整备 / 登场掺进来。
- "持续生效"的效果牌（冰心领域、固定收益…）：先 use_card(卡名, None)；如果实现没有自己把卡挂进
  lord_cards / talents，测试替它放进 lord_cards（引擎只从 lord_cards / talents / 场上单位找回合开始与反应器）。
- 结算时机写不清的（"每回合"、"解冻后"），用"有这张卡 vs 没这张卡"的对照局面比差值，或者往后推几个回合看最终结果。
- 引擎缺的两处机制按 batch_07 对外约定的入口测（复核时补读了 batch_07 的接口说明，只读入口、不看结算细节）：
  解冻 = batch_07.thaw(eng, 英雄)（将来冰之心解冻时调用；强化改写的是解冻本身，不能挂在解冻后的 "thaw" 事件上）；
  神秘英雄牌 = 手牌里 in_hand 的牌：回合开始（eng.round_start）/ 手动刷新（eng.refresh_shop）由引擎结算（2026-10-14 统一接口）。
- 棋手生命值 State.hp、排名 State.rank、淘汰 eng.eliminate(n)、购买 eng.buy(卡名)（2026-10-14 统一接口）。
"""
from __future__ import annotations

import random
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.engine import board as B                      # noqa: E402
from agent.sim.engine import (Engine, REGISTRY, impl, card_db, category,   # noqa: E402
                              quality, faction)
import agent.sim.engine as _E                                                     # noqa: E402
_E.EQUIP_DROP_ROUNDS = ()      # 固定回合投放装备（开局 / 第 4/8/12/16 回合）是引擎规则，卡牌测试里关掉，免得混进手牌 / 装备数的断言
from agent.sim.state import State, Unit                  # noqa: E402

impl("古币")                                              # 触发全部卡牌模块导入（注册）
MUTED = {n: REGISTRY.pop(n) for n in list(REGISTRY) if category(n) == "英雄牌"}

# 昭君的解冻强化（深度冻结的候选）：本批 6 张 + 雪暴时刻（batch_06）
FREEZE_ENH = {"寒冰王座", "冰封结晶", "寒冰觉醒", "霜寒进化", "升华", "破冰裂变", "雪暴时刻"}
ICE_BUFFS = {"冰心领域", "寒雨凝结", "寒潮共振"}
MACHINES = {"热能陷阱", "电磁弹簧", "等离子屏障"}
MACHINE_ENH = {"热能陷阱强化", "电磁弹簧强化", "等离子屏障强化"}

# 只有战斗效果 → 不测
SKIP_COMBAT = {
    "热能陷阱强化": "延时爆炸 / 范围 / 冷却，纯战斗",
    "电磁弹簧强化": "磁力护盾 / 增伤，纯战斗",
    "神之一手": "战斗开始临时复制敌方英雄加入战斗，战后不留，纯战斗",
    "猎头挖角": "战斗开始复制对方英雄加入战斗，战后不留，纯战斗",
    "精英计划": "最大法力值 -20，纯战斗",
    "基层强化": "伤害提升，纯战斗",
    "绝对压制": "变猪 / 减伤 / 增伤，纯战斗",
    "补给弹匣": "子弹，纯战斗", "额外弹匣": "炮弹，纯战斗", "破甲弹头": "降防，纯战斗",
    "燃烧弹头": "燃烧伤害，纯战斗", "制导弹头": "炮弹范围，纯战斗", "裂伤弹": "额外伤害，纯战斗",
    "干扰弹": "沉默 / 减疗，纯战斗", "贯穿弹头": "穿透，纯战斗", "大号弹头": "额外伤害，纯战斗",
    "凝滞弹头": "缴械禁锢，纯战斗", "歼灭补给": "装填，纯战斗", "超级电池": "激光时长，纯战斗",
    "聚焦光束": "激光增伤，纯战斗", "殉爆模式": "爆炸眩晕，纯战斗", "高能激发": "激光增伤，纯战斗",
}
# 影响养成，但引擎里没有对应的量 / 事件，无法独立测试
SKIP_GAP: dict = {}


# ================================================================ 小工具
class Fail(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def heroes_of(tier=None, fac=None):
    out = []
    for n, c in card_db().items():
        if c.get("category") != "英雄牌":
            continue
        if tier is not None and quality(n) != tier:
            continue
        if fac is not None and faction(n) != fac:
            continue
        out.append(n)
    return sorted(out)


def hero(tier, fac=None, i=0):
    hs = heroes_of(tier, fac)
    check(len(hs) > i, f"卡库里找不到第 {i + 1} 个 {tier}阶 {fac or ''} 英雄")
    return hs[i]


def new_eng(seed=0, **kw) -> Engine:
    return Engine(State(**kw), random.Random(seed))


def put(eng, name, level=1.0, cell=None, awaken=0, kind="hero", pop=True) -> Unit:
    """直接摆一个单位（不触发登场），搭局面用。"""
    if cell is None:
        cell = eng.free_cells()[0]
    u = Unit(name=name, level=float(level), cell=cell, kind=kind, pop=pop, awaken=awaken, uid=eng.s.next_uid)
    eng.s.next_uid += 1
    eng.s.units.append(u)
    return u


def install(eng, name, target=None):
    """获得一张持续生效的效果牌：use_card；实现没挂进 lord_cards / talents 的，测试替它挂进 lord_cards。"""
    s = eng.s
    before = (list(s.lord_cards), list(s.talents))
    c = impl(name)
    if c is not None and hasattr(c, "on_use"):
        ok = eng.use_card(name, target)
        check(ok is not False, f"use_card({name}) 返回 False")
    if (list(s.lord_cards), list(s.talents)) == before and name not in s.lord_cards and name not in s.talents:
        s.lord_cards.append(name)


class Rec:
    """记录 eng.choose 的调用，选第 pick 个。"""
    def __init__(self, pick=0):
        self.pick = pick
        self.calls = []

    def __call__(self, eng, kind, options, **ctx):
        self.calls.append((kind, list(options), ctx))
        return options[min(self.pick, len(options) - 1)]


def obtained(eng, name) -> bool:
    s = eng.s
    return name in s.hand or name in s.lord_cards or name in s.talents or any(u.name == name for u in s.units)


def hand_gain(eng, hand0) -> list:
    """手牌里新增的牌（多重集差）。"""
    c = Counter(eng.s.hand)
    c.subtract(Counter(hand0))
    return [k for k, v in c.items() for _ in range(max(0, v))]


def track_deaths(eng) -> set:
    died = set()
    orig = eng.fire

    def fire(u, timing, **kw):
        if timing == "death":
            died.add(u.uid)
        return orig(u, timing, **kw)
    eng.fire = fire
    return died


def twin(seed=0, **kw):
    """对照局面：同种子的两套引擎（一套放被测卡，一套不放）。"""
    return new_eng(seed, **kw), new_eng(seed, **kw)


def cycle(eng, battle=False):
    if battle:
        eng.battle()
    eng.round_end()
    eng.round_start()


def base_name(n):
    return n.replace("(赠送)", "")


# ================================================================ 测试注册
TESTS = []


def test(card, desc):
    def deco(f):
        TESTS.append((card, desc, f))
        return f
    return deco


# ================================================================ 冰：解冻强化
def _thaw_runs(card, orig_name, orig_awaken, seeds=range(5)):
    """解冻强化的触发方式（接口约定，见 batch_07 模块说明）：
    强化牌先挂上（use_card(卡, None) → 进 lord_cards），解冻由冰之心结算 —— 冰之心还没实现，
    它解冻时要调用的入口是 batch_07.thaw(eng, 英雄)（默认"随机变高 1 阶、继承等级"，再按已挂的强化改写，最后发 "thaw" 事件）。
    强化改写的是解冻本身（"英雄不变""高 2 阶"…），所以不能挂在解冻之后的 "thaw" 事件上 —— 那是雪暴时刻这类"解冻后"牌用的。
    （最初版本试的是 emit("thaw") / 直接对英雄 use_card，两者都不是约定的入口，已改。）
    每个种子一套新局面，产出 (eng, 原单位uid, 手牌快照)。"""
    from agent.sim.cards import batch_07
    for seed in seeds:
        eng = new_eng(seed, round=3, lord_level=4)
        u = put(eng, orig_name, level=10, cell=(555, 569), awaken=orig_awaken)
        hand0 = list(eng.s.hand)
        install(eng, card)
        if impl("冰之心") is not None:
            eng.use_card("冰之心", u)
            eng.battle()                                  # 冰之心：战斗结束后解冻
        else:
            batch_07.thaw(eng, u)
        yield seed, eng, u.uid, hand0


def thaw_case(card, orig_name, pred, orig_awaken=0):
    """pred(场上英雄列表, 手牌新增) 每个种子解冻后都要成立。"""
    for seed, eng, uid, hand0 in _thaw_runs(card, orig_name, orig_awaken):
        if not pred(eng.s.heroes(), hand_gain(eng, hand0)):
            raise Fail(f"种子 {seed}: 场上={[(x.name, quality(base_name(x.name)), x.awaken, x.level) for x in eng.s.heroes()]} "
                       f"手牌新增={hand_gain(eng, hand0)}")


def _is_hero_tier(n, tier, fac=None):
    n = base_name(n)
    return category(n) == "英雄牌" and quality(n) == tier and (fac is None or faction(n) == fac)


def _one(us, o):
    """解冻后场上只有这一个英雄（原单位变身 / 不变），等级继承 10。"""
    return len(us) == 1 and us[0].level == 10


@test("寒冰王座", "2阶英雄解冻后变为5阶随机英雄（继承等级）")
def t_hbwz():
    o = hero(2, "逐鹿")
    thaw_case("寒冰王座", o, lambda us, hg: _one(us, o) and us[0].name != o and _is_hero_tier(us[0].name, 5))


@test("冰封结晶", "解冻后额外获得1张解冻英雄品阶的兑换券（待核对：取原 2 阶还是变身后 3 阶，两种都算对）")
def t_bfjj():
    o = hero(2, "逐鹿")
    thaw_case("冰封结晶", o, lambda us, hg: _one(us, o) and
              sorted(hg) in (["2阶英雄兑换券"], ["3阶英雄兑换券"]))


@test("寒冰觉醒", "解冻后英雄不变，觉醒进度 +2")
def t_hbjx():
    o = hero(2, "逐鹿")
    thaw_case("寒冰觉醒", o, lambda us, hg: _one(us, o) and us[0].name == o and us[0].awaken == 2)


@test("寒冰觉醒", "已有 1 进度的英雄解冻后进度 = 3（觉醒）")
def t_hbjx2():
    o = hero(2, "逐鹿")
    thaw_case("寒冰觉醒", o, lambda us, hg: len(us) == 1 and us[0].name == o and us[0].awaken >= 3 and us[0].awakened,
              orig_awaken=1)


@test("霜寒进化", "2阶【逐鹿】解冻后变为同阵营 3 阶随机英雄")
def t_shjh():
    o = hero(2, "逐鹿")
    thaw_case("霜寒进化", o, lambda us, hg: _one(us, o) and us[0].name != o and _is_hero_tier(us[0].name, 3, "逐鹿"))


@test("升华", "2阶英雄解冻后变为 3 阶英雄、觉醒进度 +2")
def t_sh():
    o = hero(2, "逐鹿")
    thaw_case("升华", o, lambda us, hg: _one(us, o) and us[0].name != o and _is_hero_tier(us[0].name, 3)
              and us[0].awaken == 2)


@test("破冰裂变", "2阶英雄解冻后变为 4 阶英雄；下回合起冰之心单回合可用 2 次")
def t_pblb():
    o = hero(2, "逐鹿")
    from agent.sim.cards import batch_07
    thaw_case("破冰裂变", o, lambda us, hg: _one(us, o) and us[0].name != o and _is_hero_tier(us[0].name, 4))
    eng = new_eng(0, round=3, lord_level=4)
    u = put(eng, o, level=10)
    install(eng, "破冰裂变")
    batch_07.thaw(eng, u)
    f = eng.s.flags
    check(f.get("冰之心_每回合次数") == 2 and f.get("冰之心_2次起始回合") == 4,
          f"冰之心次数标记不对：{ {k: v for k, v in f.items() if '冰之心' in k} }")


@test("寒冰王座", "没有任何强化时解冻 = 冰之心原文：随机变高 1 阶、继承等级（对照）")
def t_thaw_default():
    from agent.sim.cards import batch_07
    o = hero(2, "逐鹿")
    for seed in range(5):
        eng = new_eng(seed, round=3, lord_level=4)
        u = put(eng, o, level=10)
        batch_07.thaw(eng, u)
        us = eng.s.heroes()
        check(_one(us, o) and us[0].name != o and _is_hero_tier(us[0].name, 3),
              f"种子 {seed}: {[(x.name, x.level) for x in us]}")


@test("寒冰王座", "和雪暴时刻（batch_06，react_thaw）叠加：变 5 阶且等级 +20")
def t_thaw_snow():
    from agent.sim.cards import batch_07
    o = hero(2, "逐鹿")
    eng = new_eng(0, round=3, lord_level=4)
    u = put(eng, o, level=10)
    install(eng, "寒冰王座")
    install(eng, "雪暴时刻")
    batch_07.thaw(eng, u)
    us = eng.s.heroes()
    check(len(us) == 1 and _is_hero_tier(us[0].name, 5) and us[0].level == 30,
          f"{[(x.name, quality(x.name), x.level) for x in us]}")


@test("深度冻结", "从随机 3 个（不同的）解冻强化效果中选 1 个，选中的拿到手")
def t_sddj():
    for seed in range(5):
        eng = new_eng(seed, round=3, lord_level=4)
        rec = Rec(pick=1)
        eng.chooser = rec
        put(eng, hero(2, "逐鹿"))
        install(eng, "深度冻结")
        calls = [c for c in rec.calls if len(c[1]) == 3]
        check(calls, f"没有出现 3 个选项的选择（调用={rec.calls}）")
        opts = calls[0][1]
        check(len(set(opts)) == 3, f"3 个选项有重复：{opts}")
        check(set(opts) <= FREEZE_ENH, f"选项不全是解冻强化（{sorted(FREEZE_ENH)}）：{opts}")
        check(obtained(eng, opts[1]) or str(opts[1]) in str(eng.s.flags), f"选中的 {opts[1]} 没拿到")


@test("冰雪秘藏", "从 3 个增益效果（冰心领域 / 寒雨凝结 / 寒潮共振）中选 1 个")
def t_bxmc():
    eng = new_eng(0, round=3)
    rec = Rec(pick=2)
    eng.chooser = rec
    install(eng, "冰雪秘藏")
    calls = [c for c in rec.calls if len(c[1]) == 3]
    check(calls, f"没有出现 3 个选项的选择（调用={rec.calls}）")
    opts = calls[0][1]
    check(set(opts) == ICE_BUFFS, f"选项应为 {sorted(ICE_BUFFS)}，实际 {opts}")
    check(obtained(eng, opts[2]), f"选中的 {opts[2]} 没拿到 / 没挂上")


def _board_delta(build, card, rounds=1):
    eng = new_eng(0, round=3, lord_level=4)
    us = build(eng)
    lv0 = [u.level for u in us]
    install(eng, card)
    for _ in range(rounds):
        eng.round_start()
    return [u.level - a for u, a in zip(us, lv0)]


@test("冰心领域", "回合开始：每名英雄 + 场上品阶总数×35%（5+5+4+3+3=20 → +7）")
def t_bxly():
    def build(eng):
        return [put(eng, hero(5, None, 0)), put(eng, hero(5, None, 1)), put(eng, hero(4, None, 0)),
                put(eng, hero(3, None, 0)), put(eng, hero(3, None, 1))]
    d = _board_delta(build, "冰心领域")
    check(all(x == 7 for x in d), f"每人永久等级应 +7，实际 {d}")


@test("寒雨凝结", "回合开始：每名英雄 + 主要阵营人数（3 河洛 + 1 逐鹿 → +3）")
def t_hynj():
    def build(eng):
        return [put(eng, hero(2, "河洛", 0)), put(eng, hero(2, "河洛", 1)), put(eng, hero(3, "河洛", 0)),
                put(eng, hero(2, "逐鹿", 0))]
    d = _board_delta(build, "寒雨凝结")
    check(all(x == 3 for x in d), f"每人永久等级应 +3，实际 {d}")


@test("寒潮共振", "回合开始：每名英雄 + 已觉醒数×1.5（2 名觉醒 → +3），两回合累计 +6")
def t_hcgz():
    def build(eng):
        return [put(eng, hero(2, "逐鹿", 0), awaken=3), put(eng, hero(3, "河洛", 0), awaken=3),
                put(eng, hero(4, "逐鹿", 0))]
    d = _board_delta(build, "寒潮共振")
    check(all(x == 3 for x in d), f"每人永久等级应 +3，实际 {d}")
    d2 = _board_delta(build, "寒潮共振", rounds=2)
    check(all(x == 6 for x in d2), f"两回合应累计 +6，实际 {d2}")


@test("寒潮共振", "没有觉醒英雄时不加")
def t_hcgz0():
    d = _board_delta(lambda eng: [put(eng, hero(2, "逐鹿", 0)), put(eng, hero(3, "河洛", 0))], "寒潮共振")
    check(all(x == 0 for x in d), f"应 +0，实际 {d}")


# ================================================================ 机关
@test("便携式机关", "从 热能陷阱 / 电磁弹簧 / 等离子屏障 中选 1 个并拿到")
def t_bxsjg():
    eng = new_eng(0, round=3)
    rec = Rec(pick=1)
    eng.chooser = rec
    ok = eng.use_card("便携式机关", None)
    check(ok is not False, "use_card 返回 False")
    calls = [c for c in rec.calls if set(c[1]) == MACHINES]
    check(calls, f"没有出现 {sorted(MACHINES)} 三选一（调用={rec.calls}）")
    pick = calls[0][1][1]
    check(obtained(eng, pick), f"选中的 {pick} 没拿到 / 没放上场")


for _m in sorted(MACHINES):
    def _mk(m):
        @test(m, "放置 1 个机关：场上多一个同名单位，不占人口")
        def t():
            eng = new_eng(0, round=3)
            put(eng, hero(2, "逐鹿"))
            pop0 = eng.s.pop_used()
            ok = eng.use_card(m, None)
            check(ok is not False, "use_card 返回 False")
            us = [u for u in eng.s.units if u.name == m]
            check(len(us) == 1, f"场上应有 1 个 {m}，实际 {len(us)}")
            check(eng.s.pop_used() == pop0, f"机关不应占人口：{pop0} → {eng.s.pop_used()}")
            check(us[0].kind != "hero", f"机关不应算英雄（kind={us[0].kind}）")
        return t
    _mk(_m)


@test("构筑升级", "指定场上 1 个机关升级（等级 1 → 2）；指向英雄不合法")
def t_gzsj():
    eng = new_eng(0, round=3)
    m = put(eng, "等离子屏障", level=1, kind="object", pop=False)
    h = put(eng, hero(2, "逐鹿"), level=5)
    ok = eng.use_card("构筑升级", m)
    check(ok is not False, "对机关使用返回 False")
    check(m.level == 2, f"机关等级应 1→2，实际 {m.level}")
    ok2 = eng.use_card("构筑升级", h)
    check(ok2 is False, "指向英雄应返回 False（不是机关）")
    check(h.level == 5, f"英雄等级不应变化，实际 {h.level}")


@test("成品方案", "从 3 种机关强化方案中选 1 个并拿到")
def t_cpfa():
    eng = new_eng(0, round=3)
    rec = Rec(pick=0)
    eng.chooser = rec
    install(eng, "成品方案")
    calls = [c for c in rec.calls if set(c[1]) == MACHINE_ENH]
    check(calls, f"没有出现 {sorted(MACHINE_ENH)} 三选一（调用={rec.calls}）")
    pick = calls[0][1][0]
    check(obtained(eng, pick), f"选中的 {pick} 没拿到")


@test("等离子屏障强化", "3 级屏障：战斗开始时左右 1 格的友军获得复生；远处的不给；2 级屏障不给")
def t_dlzqh():
    for lv, expect in ((3, True), (2, False)):
        eng = new_eng(0, round=5)
        install(eng, "等离子屏障强化")
        bar = put(eng, "等离子屏障", level=lv, cell=(555, 569), kind="object", pop=False)
        left = put(eng, hero(2, "逐鹿", 0), cell=(433, 569))
        right = put(eng, hero(2, "逐鹿", 1), cell=(681, 571))
        far = put(eng, hero(3, "河洛", 0), cell=(1027, 572))
        for u in list(eng.s.units):
            eng.fire(u, "battle_start")
        got = [x.counters.get("revive", 0) for x in (left, right, far)]
        if expect:
            check(got[0] >= 1 and got[1] >= 1, f"左右两格应获得复生，实际 revive={got}")
            check(got[2] == 0, f"远处英雄不应获得复生，实际 revive={got}")
        else:
            check(all(g == 0 for g in got), f"2 级屏障不应给复生，实际 revive={got}")
        del bar


# ================================================================ 其他棋手的强化
@test("驭势", "战斗开始：敌方场上所有英雄复制进手牌（只复制一遍，全是英雄牌）")
def t_ys():
    for seed in range(3):
        eng = new_eng(seed, round=5, lord_level=3)
        put(eng, hero(2, "逐鹿", 0), cell=(433, 569))
        put(eng, hero(2, "逐鹿", 1), cell=(555, 569))
        put(eng, hero(3, "河洛", 0), cell=(681, 571))
        install(eng, "驭势")
        hand0 = list(eng.s.hand)
        eng.battle()
        g = hand_gain(eng, hand0)
        check(1 <= len(g) <= 7, f"应获得 1~7 张（敌方英雄数），实际 {len(g)}：{g}")
        check(all(category(base_name(n)) == "英雄牌" for n in g), f"复制来的应全是英雄牌：{g}")


@test("飞龙在天", "明先生卦象强化：卦象·节 每回合额外 +4 能量 / 卦象·丰 每回合 +8 临时刷新（试几种卦象约定）")
def t_flzt():
    tried = []
    setups = [
        ("无卦象标记", lambda e: None),
        ("lord_cards 有先知卦象", lambda e: e.s.lord_cards.append("先知卦象")),
        ("flags 卦象=节", lambda e: e.s.flags.__setitem__("卦象", "节")),
        ("flags 卦象=卦象·节", lambda e: e.s.flags.__setitem__("卦象", "卦象·节")),
        ("flags 卦象=丰", lambda e: e.s.flags.__setitem__("卦象", "丰")),
    ]
    for label, setup in setups:
        a, b = twin(0, round=3, lord="明先生")
        setup(a)
        setup(b)
        install(a, "飞龙在天")
        for _ in range(2):
            cycle(a)
            cycle(b)
        de = a.s.energy - b.s.energy
        dr = a.s.free_refresh - b.s.free_refresh
        if (de >= 8 and dr >= 0) or dr >= 16:
            return
        tried.append(f"{label}: 能量差 {de} 刷新差 {dr}")
    raise Fail("两回合应 +8 能量（节）或 +16 临时刷新（丰）；可能是接口缺口（没有卦象状态）—— " + "；".join(tried))


@test("打成一片", "天赋 3 选 1：选项都是天赋牌，选中的进天赋")
def t_dcyp():
    eng = new_eng(0, round=5, lord_level=3)
    rec = Rec(pick=1)
    eng.chooser = rec
    ok = eng.use_card("打成一片", None)
    check(ok is not False, "use_card 返回 False")
    calls = [c for c in rec.calls if len(c[1]) == 3]
    check(calls, f"没有出现 3 选 1（调用={rec.calls}）")
    opts = calls[0][1]
    check(all(category(base_name(o)) == "天赋牌" for o in opts), f"选项应全是天赋牌：{opts}")
    check(opts[1] in eng.s.talents or opts[1] in eng.s.hand, f"选中的天赋 {opts[1]} 没拿到")


@test("朋友多多", "随机获得 4 张同一阵营的英雄牌")
def t_pydd():
    for seed in range(3):
        eng = new_eng(seed, round=5, lord_level=3)
        hand0 = list(eng.s.hand)
        ok = eng.use_card("朋友多多", None)
        check(ok is not False, "use_card 返回 False")
        g = hand_gain(eng, hand0)
        check(len(g) == 4, f"应获得 4 张，实际 {len(g)}：{g}")
        check(all(category(base_name(n)) == "英雄牌" for n in g), f"应全是英雄牌：{g}")
        facs = {faction(base_name(n)) for n in g}
        check(len(facs) == 1, f"应同一阵营，实际 {facs}")


@test("狂欢时刻", "获得 1 张打成一片")
def t_khsk():
    eng = new_eng(0, round=5)
    hand0 = list(eng.s.hand)
    ok = eng.use_card("狂欢时刻", None)
    check(ok is not False, "use_card 返回 False")
    check(hand_gain(eng, hand0) == ["打成一片"], f"手牌应新增 [打成一片]，实际 {hand_gain(eng, hand0)}")


# ================================================================ 经济类
@test("投资增值", "棋手升级时 +2 能量")
def t_tzzz():
    eng = new_eng(0, round=3, lord_level=1, energy=0)
    install(eng, "投资增值")
    e0 = eng.s.energy
    eng.s.xp = B.XP_NEED[1]
    n = eng.level_ups()
    check(n == 1, f"应升 1 级，实际 {n}")
    check(eng.s.energy - e0 == 2, f"升级应 +2 能量，实际 +{eng.s.energy - e0}")


@test("投资增值", "满级后每回合 +2 临时刷新；未满级不给")
def t_tzzz2():
    for lv, want in ((B.MAX_LORD_LEVEL, 2), (B.MAX_LORD_LEVEL - 1, 0)):
        eng = new_eng(0, round=10, lord_level=lv, energy=0)
        install(eng, "投资增值")
        r0 = eng.s.free_refresh
        eng.round_start()
        check(eng.s.free_refresh - r0 == want, f"{lv} 级：临时刷新应 +{want}，实际 +{eng.s.free_refresh - r0}")


@test("风险对冲", "战斗失败 +2 次刷新，胜利不给")
def t_fxdc():
    eng = new_eng(1, round=5, lord_level=3)
    put(eng, hero(2, "逐鹿"), cell=(433, 569))
    install(eng, "风险对冲")
    seen = set()
    for _ in range(30):
        r0 = eng.s.free_refresh
        win = eng.battle()
        d = eng.s.free_refresh - r0
        check(d == (0 if win else 2), f"{'胜' if win else '败'}：刷新应 +{0 if win else 2}，实际 +{d}")
        seen.add(win)
    check(seen == {True, False}, "30 场里没同时出现胜负（换个种子）")


@test("固定收益", "每回合额外 +1 能量、+1 临时刷新（对照局面差值，两回合 +2 / +2）")
def t_gdsy():
    a, b = twin(0, round=3)
    install(a, "固定收益")
    for _ in range(2):
        cycle(a)
        cycle(b)
    check(a.s.energy - b.s.energy == 2, f"能量差应为 2，实际 {a.s.energy - b.s.energy}")
    check(a.s.free_refresh - b.s.free_refresh == 2, f"临时刷新差应为 2，实际 {a.s.free_refresh - b.s.free_refresh}")


@test("基层提拔", "打出 1-3 阶英雄牌时其等级 +3；4 阶不加")
def t_jctb():
    eng = new_eng(0, round=3)
    install(eng, "基层提拔")
    u2 = eng.deploy(hero(2, "逐鹿"))
    u3 = eng.deploy(hero(3, "河洛"))
    u4 = eng.deploy(hero(4, "逐鹿"))
    check(u2.level == 4, f"2阶上场应 1→4，实际 {u2.level}")
    check(u3.level == 4, f"3阶上场应 1→4，实际 {u3.level}")
    check(u4.level == 1, f"4阶上场不加，实际 {u4.level}")


@test("空降人才", "本局打出的前 2 张 4-5 阶英雄觉醒进度 +2，第 3 张不加，1-3 阶不占名额")
def t_jkrc():
    eng = new_eng(0, round=5, lord_level=5)
    install(eng, "空降人才")
    a = eng.deploy(hero(2, "逐鹿"))
    b = eng.deploy(hero(4, "逐鹿", 0))
    c = eng.deploy(hero(5, None, 0))
    d = eng.deploy(hero(4, "逐鹿", 1))
    got = [a.awaken, b.awaken, c.awaken, d.awaken]
    check(got == [0, 2, 2, 0], f"觉醒进度应为 [0,2,2,0]，实际 {got}")


@test("沉淀收益", "回合开始给能量：同一棋手等级每停留 1 回合 +1，至多 4；升级后重新计")
def t_cdsy():
    a, b = twin(0, round=2, lord_level=5, xp=0, energy=0)
    install(a, "沉淀收益")
    g = []
    for _ in range(6):
        ea, eb = a.s.energy, b.s.energy
        cycle(a)
        cycle(b)
        check(a.s.lord_level == 5, "测试局面不应升级")
        g.append((a.s.energy - ea) - (b.s.energy - eb))
    check(0 <= g[0] <= 1, f"第一回合应给 0~1，实际 {g}")
    check(all(0 <= y - x <= 1 for x, y in zip(g, g[1:])), f"每回合最多多给 1，实际 {g}")
    check(max(g) == 4 and g[-1] == 4, f"至多 4 且应涨到 4，实际 {g}")
    a.s.xp = b.s.xp = 1000
    ea, eb = a.s.energy, b.s.energy
    cycle(a)
    cycle(b)
    check(a.s.lord_level == 6, "这里应升到 6 级")
    d = (a.s.energy - ea) - (b.s.energy - eb)
    check(d <= 1, f"刚升级的回合应重新计（≤1），实际 {d}")


@test("并购红利", "合成英雄 +1 能量，每回合限 3 次；下回合重置")
def t_bghl():
    eng = new_eng(0, round=3, lord_level=4, energy=0)
    install(eng, "并购红利")
    u = put(eng, hero(2, "逐鹿"))
    e0 = eng.s.energy
    for _ in range(4):
        eng.merge(u)
    check(eng.s.energy - e0 == 3, f"本回合合成 4 次应 +3，实际 +{eng.s.energy - e0}")
    eng.round_start()
    e1 = eng.s.energy
    eng.merge(u)
    check(eng.s.energy - e1 == 1, f"下回合合成应再 +1，实际 +{eng.s.energy - e1}")


@test("低压运营", "回合开始：每 15 棋手生命值，全体英雄等级 +1（60 血 +4、29 血 +1）")
def t_dyyy():
    for hp, want in ((60, 4), (29, 1)):
        eng = new_eng(0, round=5)
        eng.s.hp = hp
        us = [put(eng, hero(2, "逐鹿", 0)), put(eng, hero(3, "河洛", 0))]
        install(eng, "低压运营")
        eng.round_start()
        d = [u.level - 1 for u in us]
        check(all(x == want for x in d), f"{hp} 血：每人应 +{want}，实际 {d}")


@test("资本杠杆", "回合结束时每有 7 能量 +1（15 能量 → +2；6 能量 → 0）")
def t_zbgg():
    for e, want in ((15, 2), (6, 0)):
        a, b = twin(0, round=1, energy=e)
        install(a, "资本杠杆")
        cycle(a, battle=True)
        cycle(b, battle=True)
        d = a.s.energy - b.s.energy
        check(d == want, f"回合结束 {e} 能量：应 +{want}，实际 +{d}（若是按收入到账后的能量算就会多）")


@test("资产清算", "累计出售 10 名场上 1-3 阶英雄 → +10 能量；4 阶不计")
def t_zcqs():
    eng = new_eng(0, round=5, energy=0)
    install(eng, "资产清算")
    for i in range(3):
        eng.sell(put(eng, hero(4, "逐鹿", i % 2)))
    for i in range(9):
        eng.sell(put(eng, hero(2 if i % 2 else 3, None, i % 3)))
    check(eng.s.energy == 12, f"卖 3 个 4 阶 + 9 个低阶：应只有出售所得 12，实际 {eng.s.energy}")
    eng.sell(put(eng, hero(2, "逐鹿")))
    check(eng.s.energy == 23, f"第 10 个低阶卖出后应 12+1+10=23，实际 {eng.s.energy}")


@test("胜场激励", "第 1 次胜利不给；第 2 次胜利 +10 能量")
def t_scjl():
    eng = new_eng(2, round=5, lord_level=3, energy=0)
    put(eng, hero(2, "逐鹿"), cell=(433, 569))
    install(eng, "胜场激励")
    wins = 0
    for _ in range(40):
        e0 = eng.s.energy
        win = eng.battle()
        d = eng.s.energy - e0
        wins += win
        if win and wins == 1:
            check(d == 0, f"第 1 次胜利不应给能量，实际 +{d}")
        elif win and wins == 2:
            check(d == 10, f"第 2 次胜利应 +10，实际 +{d}")
            return
        elif not win:
            check(d == 0, f"失败不应给能量，实际 +{d}")
    raise Fail("40 场没赢够 2 次")


@test("分红", "战斗结束：每存活 1 名英雄 +1 能量")
def t_fh():
    for seed in range(8):
        eng = new_eng(seed, round=5, lord_level=3, energy=0)
        us = [put(eng, hero(2, "逐鹿", 0), cell=(529, 372)), put(eng, hero(3, "河洛", 0), cell=(556, 430)),
              put(eng, hero(4, "逐鹿", 0), cell=(681, 571))]
        install(eng, "分红")
        died = track_deaths(eng)
        e0 = eng.s.energy
        eng.battle()
        alive = sum(1 for u in us if u.uid not in died)
        check(eng.s.energy - e0 == alive, f"种子 {seed}：存活 {alive} 人应 +{alive}，实际 +{eng.s.energy - e0}")


@test("人才储备", "回合开始：棋手等级高于 3 时获得 1 张人才搜寻；3 级不给")
def t_rccb():
    for lv, want in ((4, 1), (3, 0)):
        eng = new_eng(0, round=6, lord_level=lv)
        install(eng, "人才储备")
        hand0 = list(eng.s.hand)
        eng.round_start()
        g = hand_gain(eng, hand0)
        check(g.count("人才搜寻") == want and len(g) == want, f"{lv} 级：手牌新增应 {want} 张人才搜寻，实际 {g}")


@test("投资回报", "本回合打出 ≥8 张英雄 / 效果牌 → 战斗开始随机获得其中 2 张；7 张不给")
def t_tzhb():
    for n_play, want in ((8, 2), (7, 0)):
        eng = new_eng(0, round=5, lord_level=4)
        install(eng, "投资回报")
        eng.round_start()
        played = []
        heroes = [hero(2, "逐鹿", 0), hero(2, "逐鹿", 1), hero(3, "河洛", 0), hero(4, "逐鹿", 0), hero(2, "河洛", 0)]
        for h in heroes:
            eng.deploy(h)
            played.append(h)
        tgt = eng.s.heroes()[0]
        for _ in range(n_play - len(heroes)):
            ok = eng.use_card("特训战术", tgt)
            check(ok is not False, "特训战术 使用失败（测试搭局面用）")
            played.append("特训战术")
        hand0 = list(eng.s.hand)
        eng.battle()
        g = hand_gain(eng, hand0)
        check(len(g) == want, f"打出 {n_play} 张：应获得 {want} 张，实际 {g}")
        check(all(x in played for x in g), f"获得的牌应来自本回合打出的牌 {played}：{g}")


@test("管理进修", "获得 3 棋手经验；棋手升级时再获得 3")
def t_glxx():
    eng = new_eng(0, round=2, lord_level=1, xp=0)
    install(eng, "管理进修")
    check(eng.s.xp == 3, f"使用后经验应为 3，实际 {eng.s.xp}")
    eng.s.xp += B.XP_NEED[1] - 3
    eng.level_ups()
    check(eng.s.lord_level == 2, f"应升到 2 级，实际 {eng.s.lord_level}")
    check(eng.s.xp == 3, f"升级后应再得 3 经验（7-7+3=3），实际 {eng.s.xp}")


@test("战略性收缩", "第 1 次失败不给；第 2 次失败 +9 棋手经验")
def t_zlxss():
    eng = new_eng(3, round=5, lord_level=5, xp=0)
    put(eng, hero(2, "逐鹿"), cell=(433, 569))
    install(eng, "战略性收缩")
    losses = 0
    for _ in range(40):
        x0 = eng.s.xp
        win = eng.battle()
        d = eng.s.xp - x0
        if win:
            check(d == 0, f"胜利不应给经验，实际 +{d}")
            continue
        losses += 1
        if losses == 1:
            check(d == 0, f"第 1 次失败不应给经验，实际 +{d}")
        else:
            check(d == 9, f"第 2 次失败应 +9 经验，实际 +{d}")
            return
    raise Fail("40 场没输够 2 次")


@test("短期企划", "战斗开始全体 +10 临时等级；战斗结束存活的永久 +5，阵亡的不加")
def t_dqqh():
    for seed in range(8):
        eng = new_eng(seed, round=5, lord_level=3)
        us = [put(eng, hero(2, "逐鹿", 0), level=5, cell=(529, 372)),
              put(eng, hero(3, "河洛", 0), level=5, cell=(556, 430)),
              put(eng, hero(4, "逐鹿", 0), level=5, cell=(681, 571))]
        install(eng, "短期企划")
        died = track_deaths(eng)
        n0 = len(eng.s.log)
        eng.battle()
        temps = Counter(kw.get("unit") for (_, ev, kw) in eng.s.log[n0:] if ev == "临时等级" and kw.get("n") == 10)
        for u in us:
            check(temps[u.name] == 1, f"种子 {seed}：{u.name} 应获得一次 10 临时等级，实际 {temps[u.name]} 次")
            want = 5 if u.uid not in died else 0
            check(u.level - 5 == want, f"种子 {seed}：{u.name}（{'阵亡' if u.uid in died else '存活'}）应永久 +{want}，实际 +{u.level - 5}")
            check(u.temp == 0, f"战后临时等级应清零，实际 {u.temp}")


@test("协同增效", "英雄合成时等级额外 +4（合成本身 +3 → 共 +7）")
def t_xtzx():
    eng = new_eng(0, round=3)
    install(eng, "协同增效")
    u = put(eng, hero(2, "逐鹿"), level=5)
    eng.merge(u)
    check(u.level == 12, f"5 级合成应到 12，实际 {u.level}")


@test("贷款协议", "回合开始若能量为负，本回合获得的能量 -2；不为负不罚")
def t_dkxy():
    for e, want in ((-15, -2), (5, 0)):
        a, b = twin(0, round=1, energy=e)
        install(a, "贷款协议")
        cycle(a)
        cycle(b)
        d = a.s.energy - b.s.energy
        check(d == want, f"起始能量 {e}：应差 {want}，实际 {d}")


@test("特权决策", "选择 1 个特权决策颁布：选项都是效果牌，选中的生效（挂上）")
def t_tqjc():
    eng = new_eng(0, round=5)
    rec = Rec(pick=0)
    eng.chooser = rec
    install(eng, "特权决策")
    check(rec.calls, "没有出现选择")
    opts = rec.calls[0][1]
    check(all(category(o) == "效果牌" for o in opts), f"选项应是效果牌：{opts}")
    check(obtained(eng, opts[0]) or str(opts[0]) in str(eng.s.flags), f"选中的 {opts[0]} 没生效")


# ================================================================ 其他
@test("装备箱", "挑选 1 件装备（N 选 1，选项全是装备牌），进手牌")
def t_zbx():
    eng = new_eng(0, round=5, lord_level=3)
    rec = Rec(pick=1)
    eng.chooser = rec
    hand0 = list(eng.s.hand)
    ok = eng.use_card("装备箱", None)
    check(ok is not False, "use_card 返回 False")
    check(rec.calls, "没有出现装备 N 选 1")
    opts = rec.calls[0][1]
    check(len(opts) >= 2 and all(category(o) == "装备牌" for o in opts), f"选项应是 ≥2 件装备牌：{opts}")
    pick = opts[min(1, len(opts) - 1)]
    check(hand_gain(eng, hand0) == [pick], f"手牌应新增 [{pick}]，实际 {hand_gain(eng, hand0)}")


@test("能量转换", "消耗 3 能量，获得棋手等级对应的效果锦囊（3 级 → 3阶；6 级 → 5阶）；能量不够不能用")
def t_nlzh():
    for lv, want in ((3, "3阶效果锦囊"), (6, "5阶效果锦囊")):
        eng = new_eng(0, round=8, lord_level=lv, energy=5)
        hand0 = list(eng.s.hand)
        ok = eng.use_card("能量转换", None)
        check(ok is not False, "use_card 返回 False")
        check(eng.s.energy == 2, f"{lv} 级：能量应 5→2，实际 {eng.s.energy}")
        check(hand_gain(eng, hand0) == [want], f"{lv} 级：手牌应新增 [{want}]，实际 {hand_gain(eng, hand0)}")
    eng = new_eng(0, round=8, lord_level=3, energy=2)
    ok = eng.use_card("能量转换", None)
    check(ok is False and eng.s.energy == 2 and not eng.s.hand, f"能量 2 时不应能用（ok={ok} 能量={eng.s.energy} 手牌={eng.s.hand}）")


@test("挂帅", "指定【三分】英雄，等级等同于最近的【三分】英雄；指向非三分不合法")
def t_gs():
    eng = new_eng(0, round=5)
    t = put(eng, hero(2, "三分之地", 0), level=5, cell=(433, 569))
    near = put(eng, hero(3, "三分之地", 0), level=20, cell=(555, 569))
    far = put(eng, hero(5, "三分之地", 0), level=30, cell=(1027, 572))
    other = put(eng, hero(2, "逐鹿", 0), level=8, cell=(300, 569))
    ok = eng.use_card("挂帅", t)
    check(ok is not False, "use_card 返回 False")
    check(t.level == 20, f"目标应等同最近三分英雄的 20 级，实际 {t.level}")
    check(near.level == 20 and far.level == 30, f"别人等级不应变：{near.level} / {far.level}")
    ok2 = eng.use_card("挂帅", other)
    check(ok2 is False and other.level == 8, f"指向非三分应不合法（ok={ok2} 等级={other.level}）")


@test("神秘英雄牌", "在手牌里：每回合开始约 15% 概率销毁并 +3 能量（600 次统计）")
def t_smyx():
    """手牌里的 in_hand 牌由引擎在回合开始结算（2026-10-14 统一接口：只调 eng.round_start()）。"""
    N, gone = 600, 0
    for seed in range(N):
        eng = new_eng(seed, round=5, lord_level=3, energy=0)
        eng.s.hand.append("神秘英雄牌")
        eng.round_start()
        out = "神秘英雄牌" not in eng.s.hand
        check(eng.s.energy == (3 if out else 0), f"种子 {seed}：销毁={out} 能量={eng.s.energy}（销毁才 +3）")
        gone += out
    p = gone / N
    check(0.105 <= p <= 0.195, f"销毁比例应约 15%，实际 {p:.3f}")


@test("神秘英雄牌", "手动刷新时变为不低于棋手对应品阶的英雄牌（棋手 3 级 → 品阶 ≥3），不能直接打出")
def t_smyx_refresh():
    for seed in range(20):
        eng = new_eng(seed, round=5, lord_level=3, energy=0)
        eng.s.hand.extend(["神秘英雄牌", "神秘英雄牌"])
        check(eng.use_card("神秘英雄牌", None) is False, "神秘英雄牌不应能直接打出")
        eng.s.free_refresh = 1
        check(eng.refresh_shop(), "用免费刷新手动刷新应成功")
        check("神秘英雄牌" not in eng.s.hand and len(eng.s.hand) == 2, f"应两张都变：手牌={eng.s.hand}")
        for x in eng.s.hand:
            check(category(x) == "英雄牌" and quality(x) >= 3, f"种子 {seed}: 变成了 {x}（品阶 {quality(x)}）")


@test("首购激励", "每回合购买的第 1 张牌额外复制 1 张（eng.buy 发 buy 事件）；第 2 张不复制；下回合重新计")
def t_sgjl():
    eng = new_eng(0, round=3, lord_level=3, energy=30)
    install(eng, "首购激励")
    a, b = hero(2, "逐鹿", 0), hero(2, "逐鹿", 1)
    check(eng.buy(a), "买牌失败")
    check(eng.s.hand.count(a) == 2, f"第 1 张应复制：手牌 {eng.s.hand}")
    check(eng.buy(b) and eng.s.hand.count(b) == 1, f"第 2 张不复制：手牌 {eng.s.hand}")
    eng.round_start()
    check(eng.buy(b) and eng.s.hand.count(b) == 3, f"下回合第 1 张又复制：手牌 {eng.s.hand}")
    check(eng.s.energy == 30 - 3 * 3, f"只扣 3 张的钱（复制不要钱）：{eng.s.energy}")


@test("股权激励", "回合开始：排名第 1 得 1 件特殊装备，第 2 得 1 件普通装备，其他 / 不知道排名不给")
def t_gqjl():
    for rank, pred in ((1, lambda x: category(x) == "装备牌" and quality(x) >= 3),
                       (2, lambda x: category(x) == "装备牌" and quality(x) == 2),
                       (3, None), (None, None)):
        eng = new_eng(0, round=5, lord_level=4)
        install(eng, "股权激励")
        eng.s.rank = rank
        h0 = list(eng.s.hand)
        eng.round_start()
        g = hand_gain(eng, h0)
        if pred is None:
            check(not g, f"排名 {rank}：不该给，实际 {g}")
        else:
            check(len(g) == 1 and pred(g[0]), f"排名 {rank}：应得 1 件对应装备，实际 {g}（品阶 {[quality(x) for x in g]}）")


@test("资源再分配", "每有 1 位棋手被淘汰：+5 能量、1 张基础装备牌；颁布者改为特殊装备并 +5 棋手生命值")
def t_zyzfp():
    eng = new_eng(0, round=8, lord_level=5, energy=0)
    install(eng, "资源再分配")
    h0 = list(eng.s.hand)
    eng.eliminate(2)
    g = hand_gain(eng, h0)
    check(eng.s.energy == 10, f"淘汰 2 位应 +10 能量，实际 {eng.s.energy}")
    check(len(g) == 2 and all(category(x) == "装备牌" and quality(x) == 1 for x in g), f"应得 2 张基础装备：{g}")
    eng = new_eng(0, round=8, lord_level=5, energy=0, hp=40)
    eng.chooser = Rec(2)                                   # 特权决策三选一：绝对压制 / 贷款协议 / 资源再分配
    eng.use_card("特权决策", None)
    check(eng.s.flags.get("特权_颁布") == "资源再分配", f"测试前置：没选中资源再分配 {eng.s.flags.get('特权_颁布')}")
    h0 = list(eng.s.hand)
    eng.eliminate(1)
    g = hand_gain(eng, h0)
    check(eng.s.energy == 5 and eng.s.hp == 45, f"颁布者：能量 +5、生命 +5：能量 {eng.s.energy} 生命 {eng.s.hp}")
    check(len(g) == 1 and category(g[0]) == "装备牌" and quality(g[0]) >= 3, f"颁布者应得特殊装备：{g}")


# ================================================================ 运行
def main() -> int:
    ok = fail = skip = 0
    for name in sorted(SKIP_COMBAT):
        print(f"SKIP {name}: 只有战斗效果（{SKIP_COMBAT[name]}）")
        skip += 1
    for name in sorted(SKIP_GAP):
        print(f"SKIP {name}: 接口缺口 —— {SKIP_GAP[name]}")
        skip += 1
    for card_name, desc, f in TESTS:
        tag = f"{card_name}: {desc}"
        if impl(card_name) is None:
            print(f"FAIL {tag} —— 未实现")
            fail += 1
            continue
        try:
            f()
        except Fail as e:
            print(f"FAIL {tag} —— {e}")
            fail += 1
        except Exception as e:                           # noqa: BLE001
            tb = traceback.format_exc(limit=3).strip().splitlines()
            print(f"FAIL {tag} —— 异常 {type(e).__name__}: {e} | {' / '.join(tb[-3:])}")
            fail += 1
        else:
            print(f"OK   {tag}")
            ok += 1
    print(f"\n通过 {ok} / 失败 {fail} / 跳过 {skip}（测试 {len(TESTS)} 条）")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
