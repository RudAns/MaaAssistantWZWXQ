"""第 0 批卡牌的独立测试：**只按 data/cards.json 卡面原文写**，没有读 agent/sim/cards/batch_00.py。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_batch_00.py
逐条打印 OK / FAIL（含原因），末尾打印通过数 / 失败数，有失败退出码 1。
没注册的卡记 FAIL "未实现"，不抛异常。

只有战斗效果、不影响养成的卡不写测试（COMBAT_ONLY，运行时打印 SKIP）；
影响养成但引擎没有对应事件、搭不出局面的卡列在 NOT_TESTABLE。

约定（读接口得来，不是读实现）：
  - 阿科米亚核心等级：State.core（2026-10-14 引擎统一接口；原来各批次存在 flags 里，键名不统一）。
  - 放技能次数：eng.casts(u)，u.counters["casts"] 可以直接指定（孙膑的测试用它）。
  - 阵亡且拥有牺牲：照 engine.battle() 的顺序先 fire(death) 再 fire(sacrifice)。
  - "使用英雄牌" = eng.deploy / eng.merge（引擎里两者都会 fire deploy）。
  - 枫叶舞强化 / 翩跹舞步特性的"持有"= 放进 s.lord_cards（play.py 选 lord_boost 时就是这么放的）。
"""
from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.engine import board as B                      # noqa: E402
from agent.sim.engine import Engine, impl, card_db, quality, category, faction, card_text, has_keyword  # noqa: E402
import agent.sim.engine as _E                                                     # noqa: E402
_E.EQUIP_DROP_ROUNDS = ()      # 固定回合投放装备（开局 / 第 4/8/12/16 回合）是引擎规则，卡牌测试里关掉，免得混进手牌 / 装备数的断言
from agent.sim.state import State, Unit                  # noqa: E402

C = B.CELLS

# ---------------------------------------------------------------- 只有战斗效果的卡（不测）
COMBAT_ONLY = {
    "翩跹馈赠": "转移次数（战斗内）",
    "枫之洪流": "转移后 3 秒无限放技能（战斗内）",
    "落英汲取": "转移后伤害 / 回血（战斗内）",
    "双剑·雄": "攻速 / 法力上限",
    "大乔": "法力上限（只是数登场次数）",
    "墨子": "生命上限",
    "廉颇": "护盾",
    "马超": "法力 / 暴击",
    "苍": "伤害",
    "狄仁杰": "攻速",
    "艾琳": "攻速",
    "扁鹊": "伤害",
    "嬴政": "法术攻击",
    "刘禅": "放技能时的临时等级只在战斗中产生、战斗结束清零（引擎没有'释放技能'事件）",
    "诸葛亮": "法力 + 其他英雄放技能时的临时等级（同上，战斗内）",
    "瑶": "把首件装备分享给友军 15 秒（战斗内）",
    "玄雍士兵": "卡面只有台词，没有效果",
}
# 影响养成、但引擎没有对应事件，搭不出局面
NOT_TESTABLE = {
    "枫舞扶摇": "每回合首次转移后获得临时等级 —— '转移'是战斗内的枫叶舞动作，引擎没有这个事件；临时等级也只在战斗内",
}

ALI = json.loads((ROOT / "data/lord_cards.json").read_text(encoding="utf-8"))["阿离"]
ALI_BOOSTS = {x for x in ALI["related"] if not x.startswith("翩跹舞步") and x != "即刻起舞"}


GAP_NOTES: list = []          # 接口缺口导致没测的路径（运行结束时打印）


class Unimpl(Exception):
    pass


# ---------------------------------------------------------------- 搭局面
def mk(seed=0, **kw) -> Engine:
    return Engine(State(**kw), random.Random(seed))


def add(eng: Engine, name: str, cell_idx: int = 0, level: float = 1.0, temp: float = 0.0,
        kind: str = "hero", equips=None, awaken: int = 0) -> Unit:
    """直接放一个单位（不触发登场），最小局面用。"""
    u = Unit(name=name, level=level, temp=temp, cell=C[cell_idx], kind=kind,
             pop=(kind == "hero"), uid=eng.s.next_uid, awaken=awaken)
    eng.s.next_uid += 1
    if equips:
        u.equips = list(equips)
    eng.s.units.append(u)
    return u


def need(name: str):
    if name not in card_db():
        raise AssertionError(f"卡库里没有「{name}」")
    c = impl(name)
    if c is None:
        raise Unimpl("未实现")
    return c


def use(eng: Engine, name: str, target=None):
    need(name)
    return eng.use_card(name, target)


def snap(eng: Engine) -> dict:
    return {u.uid: u.level for u in eng.s.units}


def d(eng: Engine, before: dict, u: Unit) -> float:
    return u.level - before.get(u.uid, 0)


def core(eng: Engine):
    """阿科米亚核心等级（State.core）。"""
    return float(eng.s.core)


def core0(eng):
    v = core(eng)
    return 0.0 if v is None else v


def core_delta(eng, before):
    v = core(eng)
    return v - before


def track_fires(eng: Engine) -> list:
    """包一层 eng.fire，记录 (单位名, 时机, kw)。"""
    log = []
    orig = eng.fire

    def f(u, timing, **kw):
        log.append((u.name, timing, kw))
        return orig(u, timing, **kw)
    eng.fire = f
    return log


def temp_log(eng: Engine, name: str) -> float:
    return sum(kw.get("n", 0) for (_, ev, kw) in eng.s.log if ev == "临时等级" and kw.get("unit") == name)


def recorder(eng: Engine, pick=0):
    calls = []

    def ch(e, kind, options, **ctx):
        calls.append((kind, list(options)))
        return options[pick] if options else None
    eng.chooser = ch
    return calls


def kill(eng, u, n):
    eng.fire(u, "kill", n=n)


def die(eng, u):
    """阵亡（照 engine.battle 的顺序：阵亡 → 有牺牲则触发牺牲）。"""
    eng.fire(u, "death")
    if eng.has_kw(u, "牺牲"):
        eng.fire(u, "sacrifice")


def next_round(eng):
    eng.round_end()
    eng.round_start()


P_HL1, P_HL2 = "沈梦溪", "百里守约"       # 河洛 2 阶，pilot 里无养成钩子
P_ZL1, P_ZL2 = "墨子", "廉颇"             # 逐鹿 2 阶，只有战斗效果
P_SF1, P_SF2 = "马超", "大乔"             # 三分 3/2 阶，只有战斗效果
P_DH = "项羽"                              # 大河 2 阶，只有牺牲
P_RL = "艾琳"                              # 日落海 4 阶，开团攻速（战斗）
P_WU = "苍"                                # 无阵营 3 阶，战斗


# ================================================================ 阿离：枫叶舞强化 / 秘技
def _boost_choice(name, n_opts=None):
    def t():
        need(name)
        eng = mk(0, lord_cards=["枫叶舞"])
        calls = recorder(eng, 0)
        ok = eng.use_card(name, None)
        assert ok is not False, "使用返回 False"
        assert calls, "没有调用 eng.choose 让推演打法挑强化"
        kind, opts = calls[0]
        assert opts, "强化选项为空"
        bad = [o for o in opts if o not in ALI_BOOSTS]
        assert not bad, f"选项里有不是阿离枫叶舞强化的牌 {bad}"
        if n_opts is not None:
            assert len(opts) == n_opts, f"应 {n_opts} 选 1，实际 {len(opts)} 个选项 {opts}"
        pick = opts[0]
        assert pick in eng.s.lord_cards or pick in eng.s.hand, \
            f"选中的强化「{pick}」没进 lord_cards / 手牌：lord={eng.s.lord_cards} hand={eng.s.hand}"
    return t


def t_即刻起舞():
    """从3个枫叶舞强化中选择1项：选项固定是 落英汲取 / 枫之洪流 / 翩跹馈赠（官网 previewCards，卡面重读 10-18）。"""
    need("即刻起舞")
    eng = mk(0, lord_cards=["枫叶舞"])
    calls = recorder(eng, 0)
    ok = eng.use_card("即刻起舞", None)
    assert ok is not False, "使用返回 False"
    assert calls, "没有调用 eng.choose"
    _, opts = calls[0]
    assert sorted(opts) == sorted(["落英汲取", "枫之洪流", "翩跹馈赠"]), f"选项应是即刻起舞的 3 张专属：{opts}"
    assert opts[0] in eng.s.lord_cards, f"选中的没进 lord_cards：{eng.s.lord_cards}"


def t_翩跹舞步_蓄势_击杀():
    """（持有时）枫叶舞英雄击败敌方英雄时永久等级+1。"""
    need("翩跹舞步·蓄势")
    eng = mk(0, lord_cards=["枫叶舞", "翩跹舞步·蓄势"])
    a = add(eng, P_HL1, 3, level=10)
    b = add(eng, P_HL2, 4, level=10)
    eng.bind_maple(a)
    kill(eng, a, 2)
    assert a.level == 12, f"枫叶舞英雄击败 2 名应 +2，实际 {a.level - 10}"
    kill(eng, b, 3)
    assert b.level == 10 and a.level == 12, f"非枫叶舞英雄击杀不该加级：a={a.level} b={b.level}"


def t_惊鸿游龙():
    """枫叶舞指定的英雄击败 3/7/11/15 名敌方英雄，各获得 1 张秘技牌翩跹舞步（共 4 项）。
    2026-10-14 按实测改为逐档另计（累计 3/10/21/36），见 batch_00 惊鸿游龙.STEPS。"""
    need("惊鸿游龙")
    eng = mk(0, lord_cards=["枫叶舞", "惊鸿游龙"])
    recorder(eng, 0)
    a = add(eng, P_HL1, 3, level=10)
    b = add(eng, P_HL2, 4, level=10)
    eng.bind_maple(a)

    def n_dance():
        return sum(1 for x in eng.s.hand + eng.s.lord_cards if x.startswith("翩跹舞步"))
    kill(eng, b, 5)
    assert n_dance() == 0, "非枫叶舞英雄的击杀不该计数"
    steps = [(2, 0), (1, 1), (6, 1), (1, 2), (10, 2), (1, 3), (15, 4), (10, 4)]   # 累计 2/3/9/10/20/21/36/46
    for n, want in steps:
        kill(eng, a, n)
        assert n_dance() == want, f"累计击杀加 {n} 后应有 {want} 张翩跹舞步，实际 {n_dance()}（hand={eng.s.hand} lord={eng.s.lord_cards}）"


# ================================================================ 1~2 阶
def t_曜():
    """你的英雄合成提升的等级+1（觉醒翻倍）。"""
    need("曜")
    eng = mk(0)
    add(eng, "曜", 0, level=5)
    x = add(eng, P_ZL1, 3, level=5)
    eng.merge(x)
    assert x.level == 5 + 3 + 1, f"有曜时合成应 +4，实际 +{x.level - 5}"
    eng = mk(0)
    y = add(eng, "曜", 0, level=5)
    eng.merge(y)
    assert y.level == 5 + 4, f"曜自己合成也是'你的英雄'，应 +4，实际 +{y.level - 5}"
    eng = mk(0)
    add(eng, "曜", 0, level=5, awaken=3)
    x = add(eng, P_ZL1, 3, level=5)
    eng.merge(x)
    assert x.level == 5 + 3 + 2, f"觉醒曜应 +2 额外，实际 +{x.level - 5}"
    eng = mk(0)                                            # 没有曜：基准 +3
    x = add(eng, P_ZL1, 3, level=5)
    eng.merge(x)
    assert x.level == 8


def t_狂铁():
    """整备：阿科米亚核心的等级+2。"""
    need("狂铁")
    eng = mk(0)
    u = add(eng, "狂铁", 0)
    c0 = core0(eng)
    eng.fire(u, "prepare")
    assert core_delta(eng, c0) == 2, f"核心应 +2，实际 {core_delta(eng, c0)}"
    eng = mk(0)
    u = add(eng, "狂铁", 0, awaken=3)
    c0 = core0(eng)
    eng.fire(u, "prepare")
    assert core_delta(eng, c0) == 4, f"觉醒应 +4，实际 {core_delta(eng, c0)}"


def t_孙膑():
    """你的英雄每释放3次技能，该英雄等级永久+1。（放技能次数 = eng.casts，测试用 counters["casts"] 指定；觉醒 +2）"""
    need("孙膑")
    for awk, per in ((0, 1), (3, 2)):
        eng = mk(0)
        sb = add(eng, "孙膑", 0, level=5, awaken=awk)
        sb.counters["casts"] = 0
        a = add(eng, P_ZL1, 12, level=5)
        b = add(eng, P_ZL2, 13, level=5)
        a.counters["casts"] = 3
        b.counters["casts"] = 2
        eng.battle()
        assert a.level == 5 + per, f"觉醒{awk}：放 3 次技能应 +{per}，实际 +{a.level - 5}"
        assert b.level == 5, f"觉醒{awk}：放 2 次不够 3 次，不该加：+{b.level - 5}"
        eng.battle()
        assert b.level == 5 + per, f"觉醒{awk}：两场共 4 次应 +{per}（余数跨场累计），实际 +{b.level - 5}"
        assert a.level == 5 + 2 * per, f"觉醒{awk}：两场共 6 次应 +{2 * per}，实际 +{a.level - 5}"


def t_孙尚香():
    """登场：获得1次免费刷新。"""
    need("孙尚香")
    eng = mk(0)
    eng.deploy("孙尚香")
    assert eng.s.free_refresh == 1, f"free_refresh={eng.s.free_refresh}"


def t_牛魔():
    """凯旋：获得1能量。"""
    need("牛魔")
    eng = mk(0, energy=5)
    u = add(eng, "牛魔", 0)
    eng.fire(u, "lose")
    assert eng.s.energy == 5, "败阵不该给能量"
    eng.fire(u, "win")
    assert eng.s.energy == 6, f"凯旋应 +1，实际 {eng.s.energy - 5}"
    u.awaken = 3
    eng.fire(u, "win")
    assert eng.s.energy == 8, f"觉醒凯旋应 +2，实际 {eng.s.energy - 6}"


def t_亚连():
    """整备：你穿戴装备最多的1名英雄等级+2，且阿科米亚核心等级+1。"""
    need("亚连")
    eng = mk(0)
    y = add(eng, "亚连", 0)
    a = add(eng, P_ZL1, 3, level=5, equips=["红魔", "制裁之剑"])
    b = add(eng, P_ZL2, 4, level=5, equips=["红魔"])
    s0, c0 = snap(eng), core0(eng)
    eng.fire(y, "prepare")
    assert d(eng, s0, a) == 2 and d(eng, s0, b) == 0 and d(eng, s0, y) == 0, \
        f"应只给装备最多的 a +2：a{d(eng, s0, a)} b{d(eng, s0, b)} 亚连{d(eng, s0, y)}"
    assert core_delta(eng, c0) == 1, f"核心应 +1，实际 {core_delta(eng, c0)}"
    # 并列：恰好一人 +2
    for seed in range(6):
        eng = mk(seed)
        y = add(eng, "亚连", 0)
        a = add(eng, P_ZL1, 3, equips=["红魔", "制裁之剑"])
        b = add(eng, P_ZL2, 4, equips=["红魔", "制裁之剑"])
        s0 = snap(eng)
        eng.fire(y, "prepare")
        g = sorted([d(eng, s0, a), d(eng, s0, b)])
        assert g == [0, 2], f"seed{seed} 并列时应恰好一人 +2：{g}"


def t_米莱狄():
    """召唤单位获得3临时等级。整备：该临时等级及阿科米亚核心等级额外+3。"""
    need("米莱狄")
    eng = mk(0)
    m = add(eng, "米莱狄", 0)
    c0 = core0(eng)
    eng.fire(m, "prepare")
    assert core_delta(eng, c0) == 3, f"整备核心应 +3，实际 {core_delta(eng, c0)}"
    # 召唤单位的临时等级：一次整备之后的那场战斗 = 3 + 3
    eng = mk(0)
    add(eng, "米莱狄", 0)
    add(eng, "玄雍士兵", 5, kind="summon")
    h = add(eng, P_ZL1, 3)
    eng.round_start()
    eng.battle()
    got = temp_log(eng, "玄雍士兵")
    assert abs(got - 6) < 1e-9, f"召唤单位本回合应获得 3+3=6 临时等级，实际 {got}"
    assert temp_log(eng, h.name) == 0, "英雄不是召唤单位，不该拿临时等级"


def t_项羽():
    """牺牲：所有图腾等级永久+3（召唤物免疫控制是战斗效果）。"""
    need("项羽")
    eng = mk(0)
    x = add(eng, "项羽", 0)
    t1 = add(eng, "先知图腾", 5, kind="summon", level=4)
    t2 = add(eng, "因缘图腾", 6, kind="summon", level=2)
    s1 = add(eng, "玄雍士兵", 12, kind="summon", level=2)
    h = add(eng, P_HL1, 3)
    s0 = snap(eng)
    eng.fire(x, "sacrifice")
    assert d(eng, s0, t1) == 3 and d(eng, s0, t2) == 3, f"图腾应各 +3：{d(eng, s0, t1)}, {d(eng, s0, t2)}"
    assert d(eng, s0, s1) == 0 and d(eng, s0, h) == 0 and d(eng, s0, x) == 0, "非图腾不该加级"


def t_鬼谷子():
    """在场时，召唤先知图腾（只一个）。"""
    need("鬼谷子")
    eng = mk(0)
    g = eng.deploy("鬼谷子")
    ts = [u for u in eng.s.units if u.name == "先知图腾"]
    assert len(ts) == 1, f"应召唤 1 个先知图腾，实际 {len(ts)}"
    assert ts[0].kind == "summon" and not ts[0].pop, f"先知图腾应是不占人口的召唤物：kind={ts[0].kind} pop={ts[0].pop}"
    eng.merge(g)
    ts = [u for u in eng.s.units if u.name == "先知图腾"]
    assert len(ts) == 1, f"合成后（再次登场）仍应只有 1 个先知图腾，实际 {len(ts)}"


def t_先知图腾():
    """开团：距离最近的2名己方英雄获得装备玄微之种。（10-18 重爬：原来是【大河流域】英雄，现在不限阵营）"""
    need("先知图腾")
    eng = mk(0)
    t = add(eng, "先知图腾", 9, kind="summon")
    near1 = add(eng, P_DH, 10)              # 距离 1
    near2 = add(eng, "鬼谷子", 11)          # 距离 2
    far = add(eng, "东皇太一", 27)          # 距离 5
    other = add(eng, P_ZL1, 8)              # 逐鹿，距离 1
    assert eng.dist(t, near1) < eng.dist(t, near2) < eng.dist(t, far)
    eng.fire(t, "open")
    assert "玄微之种" in near1.equips and "玄微之种" in other.equips, \
        f"最近两名己方英雄（不限阵营）应获得玄微之种：{near1.equips} / {other.equips}"
    assert "玄微之种" not in near2.equips and "玄微之种" not in far.equips, "第 3 近及更远的不该获得"


def t_玄微之种():
    """穿戴者获得牺牲（回血是战斗效果）。① 穿着过回合 → 有牺牲；② 开团时被先知图腾给上 → 同一场战斗阵亡触发牺牲。"""
    need("玄微之种")
    eng = mk(0)
    u = add(eng, P_ZL1, 3, equips=["玄微之种"])
    eng.round_start()
    assert eng.has_kw(u, "牺牲"), "穿着玄微之种过回合后没有'牺牲'"
    deaths = sac = 0
    for seed in range(30):
        eng = mk(seed)
        add(eng, "先知图腾", 9, kind="summon")
        g = add(eng, "鬼谷子", 10)
        add(eng, "东皇太一", 11)
        fl = track_fires(eng)
        eng.battle()                                       # 种子是开团时给的，战后是否保留卡面没写，不查
        dd = [x for x in fl if x[0] == "鬼谷子" and x[1] == "death"]
        ss = [x for x in fl if x[0] == "鬼谷子" and x[1] == "sacrifice"]
        deaths += len(dd)
        sac += len(ss)
    assert deaths > 0, "30 场里鬼谷子一次没阵亡（测试需调种子）"
    assert sac == deaths, f"开团时拿到玄微之种的鬼谷子阵亡 {deaths} 次，只触发牺牲 {sac} 次（种子没在开团时生效 / 没给牺牲）"


def t_梦奇():
    """退场：获得1张1阶英雄兑换券。败阵：提升英雄兑换券的品阶。"""
    need("梦奇")
    eng = mk(0, energy=0)
    u = add(eng, "梦奇", 0)
    eng.sell(u)
    assert eng.s.hand == ["1阶英雄兑换券"], f"出售后手牌 {eng.s.hand}"
    eng = mk(0)
    u = add(eng, "梦奇", 0)
    eng.fire(u, "lose")
    eng.fire(u, "lose")
    eng.sell(u)
    assert eng.s.hand == ["3阶英雄兑换券"], f"败阵 2 次后出售应得 3阶兑换券，实际 {eng.s.hand}"
    eng = mk(0)
    u = add(eng, "梦奇", 0)
    eng.fire(u, "win")
    eng.sell(u)
    assert eng.s.hand == ["1阶英雄兑换券"], f"凯旋不该升品阶：{eng.s.hand}"
    eng = mk(0)
    u = add(eng, "梦奇", 0)
    for _ in range(8):
        eng.fire(u, "lose")
    eng.sell(u)
    assert len(eng.s.hand) == 1 and eng.s.hand[0] in card_db() and eng.s.hand[0].endswith("阶英雄兑换券") \
        and int(eng.s.hand[0][0]) <= 5, f"品阶应封顶在存在的兑换券（≤5阶）：{eng.s.hand}"


def t_曹操():
    """整备：随机触发1名【三分】英雄的登场效果。"""
    need("曹操")
    tot = 0
    for seed in range(40):
        eng = mk(seed)
        c = add(eng, "曹操", 0)
        add(eng, "孙尚香", 3)
        eng.fire(c, "prepare")
        assert eng.s.free_refresh in (0, 1), f"seed{seed} free_refresh={eng.s.free_refresh}"
        tot += eng.s.free_refresh
    assert tot >= 12, f"40 次整备只触发了孙尚香登场 {tot} 次"
    for seed in range(10):                                 # 非三分（阿轲 无阵营）的登场不会被触发
        eng = mk(seed)
        c = add(eng, "曹操", 0)
        a = add(eng, "阿轲", 3, level=5)
        h = add(eng, P_ZL1, 4, level=5)
        s0 = snap(eng)
        eng.fire(c, "prepare")
        assert all(v == 0 for v in (d(eng, s0, a), d(eng, s0, h), d(eng, s0, c))), "触发了非三分英雄的登场"


def t_马可波罗():
    """登场：自身及阿科米亚核心等级+2。"""
    need("马可波罗")
    eng = mk(0)
    c0 = core0(eng)
    u = eng.deploy("马可波罗")
    assert u.level == 3, f"登场后等级应 1+2=3，实际 {u.level}"
    assert core_delta(eng, c0) == 2, f"核心应 +2，实际 {core_delta(eng, c0)}"
    eng.merge(u)                                           # 合成 = 再次使用英雄牌 = 再登场
    assert u.level == 3 + 3 + 2, f"合成后应 +3+2，实际 {u.level}"
    assert core_delta(eng, c0) == 4, f"合成后核心累计应 +4，实际 {core_delta(eng, c0)}"


def t_赵云():
    """登场：自身及随机2名其他英雄等级+1。"""
    need("赵云")
    for seed in range(8):
        eng = mk(seed)
        os_ = [add(eng, P_ZL1, 3, level=5), add(eng, P_ZL2, 4, level=5), add(eng, P_HL1, 5, level=5)]
        s0 = snap(eng)
        z = eng.deploy("赵云")
        g = [d(eng, s0, o) for o in os_]
        assert z.level == 2, f"seed{seed} 赵云自身应 +1，实际 {z.level}"
        assert sorted(g) == [0, 1, 1], f"seed{seed} 应恰好 2 名其他英雄 +1：{g}"
    eng = mk(0)
    o = add(eng, P_ZL1, 3, level=5)
    eng.deploy("赵云")
    assert o.level == 6, "只有 1 名其他英雄时它应 +1"


def t_公孙离():
    """夺取：使用时夺取周围1格随机1名己方英雄的等级（最多10级）。合成也是使用。"""
    need("公孙离")
    for seed in range(6):
        eng = mk(seed)
        nb = add(eng, P_HL1, 8, level=15)                  # 与 C[9] 相邻
        far = add(eng, P_HL2, 27, level=20)
        u = eng.deploy("公孙离", cell=C[9])
        assert u.cell == C[9]
        assert u.level == 11, f"seed{seed} 公孙离应 1+10=11，实际 {u.level}"
        assert nb.level == 5 and far.level == 20, f"seed{seed} 邻居 {nb.level}（应 5）远处 {far.level}（应 20）"
    eng = mk(0)                                            # 两个邻居 → 只夺 1 人
    a = add(eng, P_HL1, 8, level=30)
    b = add(eng, P_HL2, 10, level=30)
    u = eng.deploy("公孙离", cell=C[9])
    assert sorted([a.level, b.level]) == [20, 30], f"只应夺取 1 名邻居：{a.level}, {b.level}"
    eng.merge(u)
    assert u.level == 11 + 3 + 10, f"合成 = 再次使用，应再夺 10 级：{u.level}"
    assert a.level + b.level == 40, f"两名邻居合计应被夺 20：{a.level}+{b.level}"
    eng = mk(0)                                            # 周围没人 → 不夺
    far = add(eng, P_HL2, 27, level=20)
    u = eng.deploy("公孙离", cell=C[9])
    assert u.level == 1 and far.level == 20


def t_百里玄策():
    """整备：你的闪现英雄上回合每击败过1名敌方英雄，随机1名闪现英雄等级+1。"""
    need("百里玄策")
    for seed in range(5):
        eng = mk(seed, round=3)
        x = add(eng, "百里玄策", 0, level=5)
        a = add(eng, "阿轲", 3, level=5)                    # 闪现
        h = add(eng, P_ZL1, 4, level=5)                    # 非闪现
        kill(eng, a, 2)
        kill(eng, x, 1)
        kill(eng, h, 4)                                    # 非闪现的击杀不算
        s0 = snap(eng)
        next_round(eng)
        g = d(eng, s0, x) + d(eng, s0, a)
        assert g == 3, f"seed{seed} 闪现英雄上回合击败 3 名，闪现英雄合计应 +3，实际 {g}"
        assert d(eng, s0, h) == 0, f"seed{seed} 非闪现英雄不该加级"
    eng = mk(0, round=3)                                   # 上回合没击杀 → 0
    x = add(eng, "百里玄策", 0, level=5)
    s0 = snap(eng)
    next_round(eng)
    assert d(eng, s0, x) == 0


def t_援护战术():
    """转瞬 全体逐鹿英雄护盾（战斗）。养成上：能正常打出（作为效果牌触发镜 / 白起 / 芈月…）。"""
    need("援护战术")
    eng = mk(0)
    add(eng, P_ZL1, 3)
    ok = eng.use_card("援护战术", None)
    assert ok is True, f"援护战术应能直接打出，use_card 返回 {ok}"


def t_招募战术():
    """转瞬 随机获得1张【逐鹿】英雄牌。"""
    need("招募战术")
    seen = set()
    for seed in range(20):
        eng = mk(seed, lord_level=3)
        ok = use(eng, "招募战术")
        assert ok is not False, "use_card 返回 False"
        h = eng.s.hand
        assert len(h) == 1, f"seed{seed} 应得 1 张牌，实际 {h}"
        assert category(h[0]) == "英雄牌" and faction(h[0]) == "逐鹿", f"seed{seed} 得到 {h[0]}（{category(h[0])}/{faction(h[0])}）"
        seen.add(h[0])
    assert len(seen) >= 2, f"20 次只得到 {seen}，不像随机"


def t_演练战术():
    """转瞬 所有英雄等级+1。"""
    need("演练战术")
    eng = mk(0)
    hs = [add(eng, P_ZL1, 3, level=5), add(eng, P_HL1, 4, level=2)]
    s1 = add(eng, "玄雍士兵", 5, kind="summon", level=3)
    s0 = snap(eng)
    ok = use(eng, "演练战术")
    assert ok is not False
    assert all(d(eng, s0, h) == 1 for h in hs), f"英雄应各 +1：{[d(eng, s0, h) for h in hs]}"
    assert d(eng, s0, s1) == 0, "召唤物不是英雄"


def t_蒙恬():
    """牺牲：召唤2名与自身等级相同的玄雍士兵。"""
    need("蒙恬")
    eng = mk(0)
    m = add(eng, "蒙恬", 0, level=7)
    eng.fire(m, "sacrifice")
    ss = [u for u in eng.s.units if u.name == "玄雍士兵"]
    assert len(ss) == 2, f"应召唤 2 名玄雍士兵，实际 {len(ss)}"
    assert all(u.level == 7 and u.kind == "summon" and not u.pop for u in ss), \
        f"玄雍士兵应 7 级、不占人口的召唤物：{[(u.level, u.kind, u.pop) for u in ss]}"


def t_蔡文姬():
    """每当你使用【三分】英雄牌时，等级最低的其他3名【三分】英雄等级+1。"""
    need("蔡文姬")
    eng = mk(0)
    cw = add(eng, "蔡文姬", 0, level=50)
    a = add(eng, P_SF1, 3, level=2)
    b = add(eng, P_SF2, 4, level=3)
    c = add(eng, "刘禅", 5, level=4)
    dd = add(eng, "貂蝉", 6, level=20)
    z = add(eng, P_ZL1, 7, level=1)                       # 逐鹿，等级最低但不是三分
    s0 = snap(eng)
    e = eng.deploy("诸葛亮", level=30)                    # 三分、无登场
    g = {u.name: d(eng, s0, u) for u in (cw, a, b, c, dd, z)}
    assert g[a.name] == 1 and g[b.name] == 1 and g[c.name] == 1, f"最低的 3 名三分应各 +1：{g}"
    assert g[dd.name] == 0 and g[cw.name] == 0 and g[z.name] == 0 and e.level == 30, f"其他人不该加级：{g} 诸葛亮 {e.level}"
    s0 = snap(eng)
    eng.deploy(P_ZL2, level=30)                           # 非三分
    assert all(d(eng, s0, u) == 0 for u in (a, b, c)), "使用非三分英雄牌不该触发"


def t_云中君():
    """开团：你的英雄每穿戴1件装备，该英雄等级永久+1。"""
    need("云中君")
    for aw in (0, 3):
        k = 2 if aw else 1
        eng = mk(0)
        y = add(eng, "云中君", 0, level=5, equips=["红魔"], awaken=aw)
        a = add(eng, P_ZL1, 3, level=5, equips=["红魔", "制裁之剑"])
        b = add(eng, P_ZL2, 4, level=5)
        s1 = add(eng, "玄雍士兵", 5, kind="summon", level=5, equips=["红魔"])
        s0 = snap(eng)
        eng.fire(y, "open")
        g = [d(eng, s0, u) for u in (y, a, b, s1)]
        assert g == [1 * k, 2 * k, 0, 0], f"{'觉醒' if aw else ''}云中君开团 应 {[k, 2 * k, 0, 0]}，实际 {g}"


def t_姬小满():
    """每当你出售4名场上英雄后，获得1次临时刷新。"""
    need("姬小满")

    def refresh(eng):
        extra = sum(v for k, v in eng.s.flags.items() if isinstance(k, str) and "刷新" in k
                    and isinstance(v, (int, float)) and not isinstance(v, bool))
        return eng.s.free_refresh + extra
    eng = mk(0)
    add(eng, "姬小满", 0)
    hs = [add(eng, P_ZL1, i + 1) for i in range(8)]
    for h in hs[:3]:
        eng.sell(h)
    assert refresh(eng) == 0, f"出售 3 名就给了刷新 {refresh(eng)}"
    eng.sell(hs[3])
    assert refresh(eng) == 1, f"出售 4 名应 +1 次刷新，实际 {refresh(eng)}"
    for h in hs[4:]:
        eng.sell(h)
    assert refresh(eng) == 2, f"出售 8 名应 +2 次刷新，实际 {refresh(eng)}"


def t_白起():
    """每回合首次使用效果牌时，获得1张援护战术。"""
    need("白起")
    eng = mk(0)
    add(eng, "白起", 0)
    h = add(eng, P_ZL1, 3)
    eng.use_card("特训战术", h)
    eng.use_card("特训战术", h)
    assert eng.s.hand.count("援护战术") == 1, f"同一回合用 2 张效果牌应只得 1 张援护战术：{eng.s.hand}"
    next_round(eng)
    eng.use_card("特训战术", h)
    assert eng.s.hand.count("援护战术") == 2, f"下一回合首次使用应再得 1 张：{eng.s.hand}"


def t_阿轲():
    """登场：自身及随机1名英雄等级+1（棋手生命值每降低10点重复 —— State 无棋手生命值，按满血 1 次）。"""
    need("阿轲")
    for seed in range(8):
        eng = mk(seed)
        os_ = [add(eng, P_ZL1, 3, level=5), add(eng, P_ZL2, 4, level=5), add(eng, P_HL1, 5, level=5)]
        s0 = snap(eng)
        a = eng.deploy("阿轲")
        tot = (a.level - 1) + sum(d(eng, s0, o) for o in os_)
        assert a.level >= 2, f"seed{seed} 阿轲自身应至少 +1：{a.level}"
        assert tot == 2, f"seed{seed} 合计应 +2（自身 1 + 随机 1 名 1），实际 {tot}"


def t_裴擒虎():
    """整备：随机获得1名其他英雄等级数一半的临时等级，不可叠加。"""
    need("裴擒虎")
    seen = set()
    for seed in range(12):
        eng = mk(seed)
        p = add(eng, "裴擒虎", 0, level=30)
        add(eng, P_ZL1, 3, level=10)
        add(eng, P_ZL2, 4, level=20)
        eng.fire(p, "prepare")
        assert p.temp in (5, 10), f"seed{seed} 临时等级 {p.temp}（应为 10/2 或 20/2）"
        eng.fire(p, "prepare")
        assert p.temp in (5, 10), f"seed{seed} 两次整备后临时等级 {p.temp}：不可叠加"
        assert p.level == 30, "不该加永久等级"
        seen.add(p.temp)
    assert seen == {5, 10}, f"12 个种子只出现 {seen}，不像随机"


def t_老夫子():
    """每回合首次合成时，获得1张招募战术。"""
    need("老夫子")
    eng = mk(0)
    add(eng, "老夫子", 0)
    h = add(eng, P_ZL1, 3)
    eng.merge(h)
    eng.merge(h)
    assert eng.s.hand.count("招募战术") == 1, f"同一回合合成 2 次应只得 1 张：{eng.s.hand}"
    next_round(eng)
    eng.merge(h)
    assert eng.s.hand.count("招募战术") == 2, f"下一回合首次合成应再得 1 张：{eng.s.hand}"


def t_猪八戒():
    """夺取；每当3阶及以下英雄合成时，与其同阶的英雄等级+1。"""
    need("猪八戒")
    eng = mk(0)                                            # 夺取
    nb = add(eng, P_HL1, 8, level=15)
    far = add(eng, P_HL2, 27, level=20)
    u = eng.deploy("猪八戒", cell=C[9])
    assert u.level == 11 and nb.level == 5 and far.level == 20, f"夺取：八戒 {u.level} 邻居 {nb.level} 远处 {far.level}"
    eng = mk(0)                                            # 2 阶合成
    add(eng, "猪八戒", 0, level=5)
    m = add(eng, P_ZL1, 3, level=5)                        # 2 阶
    l2 = add(eng, P_ZL2, 4, level=5)                       # 2 阶
    q3 = add(eng, P_SF1, 5, level=5)                       # 3 阶
    s0 = snap(eng)
    eng.merge(m)
    assert d(eng, s0, l2) == 1, f"同阶（2阶）英雄应 +1：{d(eng, s0, l2)}"
    assert d(eng, s0, q3) == 0, "不同阶不该加"
    assert d(eng, s0, m) in (3, 4), f"合成者自己 +{d(eng, s0, m)}（合成 3，是否算'同阶'有歧义）"
    eng = mk(0)                                            # 4 阶合成不触发
    add(eng, "猪八戒", 0, level=5)
    b4 = add(eng, "扁鹊", 3, level=5)
    o4 = add(eng, "嬴政", 4, level=5)
    s0 = snap(eng)
    eng.merge(b4)
    assert d(eng, s0, o4) == 0 and d(eng, s0, b4) == 3, f"4 阶合成不该触发：嬴政 +{d(eng, s0, o4)} 扁鹊 +{d(eng, s0, b4)}"


def t_张良():
    """每当你的单位触发开团效果时，场上随机3名己方英雄等级永久+1。"""
    need("张良")
    for seed in range(6):
        eng = mk(seed)
        z = add(eng, "张良", 0, level=5)
        y = add(eng, "嬴政", 1, level=5)                   # 有开团
        t = add(eng, "先知图腾", 2, kind="summon", level=5)
        hs = [z, y] + [add(eng, P_ZL1, 3 + i, level=5) for i in range(3)]
        s0 = snap(eng)
        eng.fire(y, "open")
        g = [d(eng, s0, h) for h in hs]
        assert sum(g) == 3 and max(g) == 1, f"seed{seed} 应随机 3 名英雄各 +1：{g}"
        assert d(eng, s0, t) == 0, "召唤物不是英雄"
        s0 = snap(eng)
        eng.fire(hs[2], "open")                            # 墨子：没有开团效果
        assert sum(d(eng, s0, h) for h in hs) == 0, "没有开团效果的单位不该触发"
        s0 = snap(eng)
        eng.fire(t, "open")                                # 召唤物也是"你的单位"
        g = [d(eng, s0, h) for h in hs]
        assert sum(g) == 3, f"seed{seed} 先知图腾（召唤单位）开团也应触发 +3：{g}"


def t_亚瑟():
    """每当你使用【日落海】英雄牌时，阿科米亚核心等级+3。"""
    need("亚瑟")
    eng = mk(0)
    add(eng, "亚瑟", 0)
    c0 = core0(eng)
    u = eng.deploy(P_RL)
    assert core_delta(eng, c0) == 3, f"使用日落海英雄牌核心应 +3，实际 {core_delta(eng, c0)}"
    eng.merge(u)
    assert core_delta(eng, c0) == 6, f"合成（再次使用）后累计应 +6，实际 {core_delta(eng, c0)}"
    eng.deploy(P_ZL1)
    assert core_delta(eng, c0) == 6, "使用非日落海英雄牌不该加核心"


def t_貂蝉():
    """登场：【三分】英雄等级+1。"""
    need("貂蝉")
    eng = mk(0)
    a = add(eng, P_SF1, 3, level=5)
    b = add(eng, P_SF2, 4, level=5)
    z = add(eng, P_ZL1, 5, level=5)
    s0 = snap(eng)
    u = eng.deploy("貂蝉")
    assert d(eng, s0, a) == 1 and d(eng, s0, b) == 1, "三分英雄应各 +1"
    assert d(eng, s0, z) == 0, "非三分不该加"
    assert u.level in (1, 2), f"貂蝉自己 {u.level}"


def t_安琪拉():
    """整备：自身及阿科米亚核心等级+2。"""
    need("安琪拉")
    eng = mk(0)
    u = add(eng, "安琪拉", 0, level=5)
    c0 = core0(eng)
    eng.fire(u, "prepare")
    assert u.level == 7, f"自身应 +2，实际 {u.level - 5}"
    assert core_delta(eng, c0) == 2, f"核心应 +2，实际 {core_delta(eng, c0)}"


def t_镜():
    """每使用2张效果牌，等级+1（觉醒翻倍）。"""
    need("镜")
    for aw, k in ((0, 1), (3, 2)):
        eng = mk(0)
        j = add(eng, "镜", 0, level=5, awaken=aw)
        h = add(eng, P_ZL1, 3)
        for i in range(1, 6):
            eng.use_card("特训战术", h)
            want = 5 + (i // 2) * k
            assert j.level == want, f"{'觉醒' if aw else ''}镜：用了 {i} 张效果牌应 {want} 级，实际 {j.level}"


def t_朵莉亚():
    """每当你的英雄触发整备效果时，随机3名英雄及阿科米亚核心等级+1。"""
    need("朵莉亚")
    for seed in range(6):
        eng = mk(seed)
        dl = add(eng, "朵莉亚", 0, level=5)
        k = add(eng, "狂铁", 1, level=5)                   # 整备：核心 +2
        hs = [dl, k] + [add(eng, P_ZL1, 3 + i, level=5) for i in range(3)]
        s0, c0 = snap(eng), core0(eng)
        eng.fire(k, "prepare")
        g = [d(eng, s0, h) for h in hs]
        assert sum(g) == 3 and max(g) == 1, f"seed{seed} 随机 3 名英雄应各 +1：{g}"
        assert core_delta(eng, c0) == 3, f"seed{seed} 核心应 +2（狂铁）+1（朵莉亚）=3，实际 {core_delta(eng, c0)}"
        s0 = snap(eng)
        eng.fire(hs[2], "prepare")                         # 墨子没有整备效果
        assert sum(d(eng, s0, h) for h in hs) == 0, "没有整备效果的英雄不该触发"


def t_莱西奥():
    """每当你出售2名场上英雄后，阿科米亚核心等级+1。"""
    need("莱西奥")
    eng = mk(0)
    add(eng, "莱西奥", 0)
    hs = [add(eng, P_ZL1, i + 1) for i in range(5)]
    c0 = core0(eng)
    eng.sell(hs[0])
    assert core0(eng) - c0 == 0, "出售 1 名不该加核心"
    eng.sell(hs[1])
    assert core_delta(eng, c0) == 1, f"出售 2 名核心应 +1，实际 {core_delta(eng, c0)}"
    eng.sell(hs[2])
    eng.sell(hs[3])
    eng.sell(hs[4])
    assert core_delta(eng, c0) == 2, f"出售 5 名核心应 +2，实际 {core_delta(eng, c0)}"


def t_上官婉儿():
    """整备：随机对3名不同【河洛】英雄使用1张古币（古币：河洛英雄等级+2）。"""
    need("上官婉儿")
    for seed in range(8):
        eng = mk(seed)
        w = add(eng, "上官婉儿", 0, level=5)
        hl = [w, add(eng, P_HL1, 3, level=5), add(eng, P_HL2, 4, level=5), add(eng, "苏烈", 5, level=5)]
        z = add(eng, P_ZL1, 6, level=5)
        s0 = snap(eng)
        eng.fire(w, "prepare")
        g = [d(eng, s0, h) for h in hl]
        assert sorted(g).count(2) == 3 and sum(g) == 6, f"seed{seed} 应 3 名不同河洛英雄各 +2：{g}"
        assert d(eng, s0, z) == 0, "非河洛不该加"


def t_李信():
    """每当你拥有牺牲效果的英雄阵亡时，李信临时等级+7。"""
    need("李信")
    eng = mk(0)
    l = add(eng, "李信", 0, level=5)
    s = add(eng, "苏烈", 3, level=5)                       # 有牺牲
    m = add(eng, P_ZL1, 4, level=5)                        # 无牺牲
    die(eng, s)
    assert l.temp == 7, f"牺牲英雄阵亡一次应 +7 临时等级，实际 {l.temp}"
    die(eng, m)
    assert l.temp == 7, f"无牺牲的英雄阵亡不该触发：{l.temp}"
    assert l.level == 5, "是临时等级，不是永久"


def t_李白():
    """古币牌会额外提供4临时等级。若目标为李白，该临时等级会额外+3。"""
    need("李白")
    eng = mk(0)
    lb = add(eng, "李白", 0, level=5)
    h = add(eng, P_HL1, 3, level=5)
    eng.use_card("古币", h)
    assert h.level == 7 and h.temp == 4, f"古币给沈梦溪：应 +2 级 +4 临时，实际 {h.level - 5} / {h.temp}"
    eng.use_card("古币", lb)
    assert lb.level == 7 and lb.temp == 7, f"古币给李白：应 +2 级 +7 临时，实际 {lb.level - 5} / {lb.temp}"


def t_雅典娜():
    """开团：随机触发2名【日落海】英雄的整备效果。"""
    need("雅典娜")
    eng = mk(0)
    a = add(eng, "雅典娜", 0, level=5)
    k = add(eng, "狂铁", 3, level=5)
    q = add(eng, "安琪拉", 4, level=5)
    s0, c0 = snap(eng), core0(eng)
    eng.fire(a, "open")
    assert core_delta(eng, c0) == 4, f"应触发狂铁（核心+2）和安琪拉（核心+2），核心实际 +{core_delta(eng, c0)}"
    assert d(eng, s0, q) == 2, f"安琪拉整备自身应 +2，实际 {d(eng, s0, q)}"
    for seed in range(8):                                  # 3 名日落海整备英雄 → 恰好触发 2 名不同的
        eng = mk(seed)
        a = add(eng, "雅典娜", 0, level=5)
        add(eng, "狂铁", 3)
        add(eng, "安琪拉", 4)
        add(eng, "亚连", 5)
        fl = track_fires(eng)
        eng.fire(a, "open")
        trig = [n for n, t, _ in fl if t == "prepare"]
        assert len(trig) == 2 and len(set(trig)) == 2, f"seed{seed} 应触发 2 名不同英雄的整备，实际 {trig}"
        assert "雅典娜" not in trig, f"seed{seed} 触发了自己（没有整备效果）"
    eng = mk(0)                                            # 非日落海不会被触发
    a = add(eng, "雅典娜", 0)
    x = add(eng, "上官婉儿", 3)
    fl = track_fires(eng)
    eng.fire(a, "open")
    assert not [t for t in fl if t[0] == "上官婉儿" and t[1] == "prepare"], "触发了非日落海英雄的整备"


def t_甄姬():
    """登场：获得1张其他的登场英雄牌。"""
    need("甄姬")
    seen = set()
    for seed in range(20):
        eng = mk(seed, lord_level=5)          # 10-19 确认：送英雄牌的品阶跟棋手等级 / 商店池走 → 1 级棋手只出 1 阶，测试放 5 级
        eng.deploy("甄姬")
        h = eng.s.hand
        assert len(h) == 1, f"seed{seed} 应得 1 张牌：{h}"
        n = h[0]
        assert category(n) == "英雄牌" and n != "甄姬", f"seed{seed} 得到 {n}"
        assert has_keyword(n, "登场") or "登场" in card_text(n), f"seed{seed} {n} 不是登场英雄"
        seen.add(n)
    assert len(seen) >= 2, f"20 次只得到 {seen}"
    eng = mk(0, lord_level=5)                 # 卡面觉醒：从 3 张中选择 1 张（只得 1 张）
    calls = recorder(eng, 0)
    eng.deploy("甄姬")
    eng.s.units[-1].awaken = 3
    eng.s.hand.clear()
    eng.fire(eng.s.units[-1], "deploy")
    assert len(eng.s.hand) == 1, f"觉醒甄姬登场应只得 1 张：{eng.s.hand}"
    assert any(len(o) == 3 for _, o in calls), f"觉醒甄姬应 3 选 1：{calls}"


def t_小乔():
    """每当你的英雄触发登场效果时，随机3名英雄等级+1。"""
    need("小乔")
    for seed in range(6):
        eng = mk(seed)
        x = add(eng, "小乔", 0, level=5)
        os_ = [x] + [add(eng, P_ZL1, 3 + i, level=5) for i in range(3)]
        s0 = snap(eng)
        u = eng.deploy("孙尚香")                           # 有登场效果
        g = [d(eng, s0, o) for o in os_] + [u.level - 1]
        assert sum(g) == 3 and max(g) == 1, f"seed{seed} 随机 3 名英雄应各 +1：{g}"
        s0 = snap(eng)
        v = eng.deploy(P_ZL2)                              # 没有登场效果
        g = [d(eng, s0, o) for o in os_] + [v.level - 1]
        assert sum(g) == 0, f"seed{seed} 没有登场效果的英雄上场不该触发：{g}"


def t_大司命():
    """在场时召唤往生图腾。每当你触发牺牲效果时：往生图腾等级永久+2。"""
    need("大司命")
    eng = mk(0)
    eng.deploy("大司命")
    ts = [u for u in eng.s.units if u.name.startswith("往生图腾")]
    assert len(ts) == 1 and ts[0].kind == "summon" and not ts[0].pop, f"应召唤 1 个往生图腾：{[(u.name, u.kind) for u in ts]}"
    t = ts[0]
    s = add(eng, "苏烈", 20, level=5)
    m = add(eng, P_ZL1, 21, level=5)
    l0 = t.level
    die(eng, s)
    assert t.level - l0 == 2, f"苏烈牺牲一次，往生图腾应 +2，实际 {t.level - l0}"
    die(eng, m)
    assert t.level - l0 == 2, "无牺牲的阵亡不该触发"


def _wangsheng(name):
    def t():
        """大司命灵体以外的1-3阶己方召唤物上场时，吸收该单位使往生图腾等级永久+1。
        两条路径都测：① 真实的出牌路径：蒙恬牺牲召唤 2 名玄雍士兵（2 阶召唤物）；② 直接 eng.summon（引擎召唤发 summon 事件）。"""
        need(name)
        need("蒙恬")
        eng = mk(0)
        t_ = add(eng, name, 0, kind="summon", level=5)
        m = add(eng, "蒙恬", 10, level=4)
        n0 = len(eng.s.units)
        die(eng, m)                                         # 牺牲：召唤 2 名玄雍士兵
        assert not any(u.name == "玄雍士兵" for u in eng.s.units),             f"2 阶召唤物上场应被吸收（离场）：{[u.name for u in eng.s.units]}"
        assert len(eng.s.units) == n0, "吸收后场上单位数应不变"
        assert t_.level == 7, f"吸收 2 名应 +2，实际 {t_.level - 5}"
        # 没有往生图腾时，同样的牺牲应留下 2 名士兵（证明上面的"消失"是被吸收，不是别的原因）
        eng = mk(0)
        m = add(eng, "蒙恬", 10, level=4)
        die(eng, m)
        assert sum(u.name == "玄雍士兵" for u in eng.s.units) == 2, "没有往生图腾时应召唤出 2 名玄雍士兵"
        # ② 直接调 eng.summon：1-3 阶召唤物被吸收、4 阶以上不吸收
        eng = mk(0)
        t_ = add(eng, name, 0, kind="summon", level=5)
        low = [n for n, c in card_db().items() if c.get("category") == "召唤物" and 1 <= quality(n) <= 3
               and "图腾" not in n and "灵体" not in n][0]
        eng.summon(low)
        assert not any(u.name == low for u in eng.s.units), f"直接召唤的 {low}（{quality(low)} 阶）应被吸收"
        assert t_.level == 6, f"吸收 1 名应 +1，实际 {t_.level - 5}"
        high = [n for n, c in card_db().items() if c.get("category") == "召唤物" and quality(n) >= 4]
        if high:
            eng.summon(high[0])
            assert any(u.name == high[0] for u in eng.s.units), f"{high[0]}（{quality(high[0])} 阶）不该被吸收"
            assert t_.level == 6, "4 阶以上召唤物不该让图腾升级"
    return t


def t_庄周():
    """登场：自身等级+2并获得1张招募战术。"""
    need("庄周")
    eng = mk(0)
    u = eng.deploy("庄周")
    assert u.level == 3, f"应 1+2=3，实际 {u.level}"
    assert eng.s.hand == ["招募战术"], f"手牌 {eng.s.hand}"


def t_刘邦():
    """开团：摧毁你所有的图腾，每摧毁1个，刘邦等级永久+5。"""
    need("刘邦")
    eng = mk(0)
    l = add(eng, "刘邦", 0, level=5)
    add(eng, "先知图腾", 3, kind="summon")
    add(eng, "因缘图腾", 4, kind="summon")
    sd = add(eng, "玄雍士兵", 5, kind="summon")
    eng.fire(l, "open")
    assert l.level == 15, f"摧毁 2 个图腾应 +10，实际 +{l.level - 5}"
    assert sd in eng.s.units, "非图腾召唤物不该被摧毁"
    # 图腾是战斗中被摧毁，战后是否回到棋盘卡面没写（战斗阵亡的单位战后都会回来），不查"图腾离场"
    eng = mk(0)
    l = add(eng, "刘邦", 0, level=5)
    eng.fire(l, "open")
    assert l.level == 5, "没有图腾不该加级"


def t_韩信():
    """每当你的英雄触发整备或开团效果时，韩信等级永久+2（觉醒翻倍）。"""
    need("韩信")
    for aw, k in ((0, 2), (3, 4)):
        eng = mk(0)
        h = add(eng, "韩信", 0, level=5, awaken=aw)
        kt = add(eng, "狂铁", 3)
        yz = add(eng, "嬴政", 4)
        mz = add(eng, P_ZL1, 5)
        eng.fire(kt, "prepare")
        assert h.level == 5 + k, f"{'觉醒' if aw else ''}韩信：整备触发应 +{k}，实际 +{h.level - 5}"
        eng.fire(yz, "open")
        assert h.level == 5 + 2 * k, f"开团触发后应累计 +{2 * k}，实际 +{h.level - 5}"
        eng.fire(mz, "prepare")
        eng.fire(mz, "open")
        assert h.level == 5 + 2 * k, "没有整备 / 开团效果的英雄不该触发"


def t_蒙犽():
    """登场：获得1张特训战术。"""
    need("蒙犽")
    eng = mk(0)
    eng.deploy("蒙犽")
    assert eng.s.hand == ["特训战术"], f"手牌 {eng.s.hand}"


def t_东皇太一():
    """开团：获得每个图腾等级数15%的临时等级。"""
    need("东皇太一")
    eng = mk(0)
    u = add(eng, "东皇太一", 0, level=5)
    add(eng, "先知图腾", 3, kind="summon", level=20)
    add(eng, "因缘图腾", 4, kind="summon", level=40)
    add(eng, "玄雍士兵", 5, kind="summon", level=100)
    eng.fire(u, "open")
    assert abs(u.temp - 9) < 1e-6, f"应得 (20+40)×15%=9 临时等级，实际 {u.temp}"
    assert u.level == 5


def t_敖隐():
    """战斗中，每个图腾每4秒为敖隐提供4临时等级（攻速是战斗效果）：没有图腾 0，2 个图腾 = 1 个的 2 倍。"""
    need("敖隐")
    res = []
    for n in (0, 1, 2):
        eng = mk(0)
        add(eng, "敖隐", 0, level=5)
        for i, tn in enumerate(["因缘图腾", "先知图腾"][:n]):
            add(eng, tn, 3 + i, kind="summon")
        eng.battle()
        res.append(temp_log(eng, "敖隐"))
    assert res[0] == 0, f"没有图腾却得了临时等级 {res[0]}"
    assert res[1] > 0 and res[1] % 4 == 0, f"1 个图腾应得 4 的倍数的临时等级：{res[1]}"
    assert abs(res[2] - 2 * res[1]) < 1e-9, f"2 个图腾应是 1 个的 2 倍：{res}"


def t_机关傀儡():
    """获得等同于场上【逐鹿】英雄中最高等级一半的临时等级。"""
    need("机关傀儡")
    eng = mk(0)
    p = add(eng, "机关傀儡", 0, kind="summon", level=3)
    add(eng, P_ZL1, 3, level=20)
    add(eng, P_ZL2, 4, level=10)
    add(eng, P_HL1, 5, level=50)                           # 河洛，不算
    eng.round_start()
    eng.battle()
    got = temp_log(eng, "机关傀儡")
    assert abs(got - 10) < 1e-9, f"应得 20/2=10 临时等级（每回合一次），实际 {got}"


def t_花木兰():
    """本回合每使用5张古币牌，获得1次复生；牺牲：临时等级+10。"""
    need("花木兰")
    eng = mk(0)
    m = add(eng, "花木兰", 0, level=5)
    eng.fire(m, "sacrifice")
    assert m.temp == 10 and m.level == 5, f"牺牲应 +10 临时等级：temp={m.temp} level={m.level}"
    rv = {}
    for n in (0, 5, 10):
        eng = mk(0)
        m = add(eng, "花木兰", 0, level=5)
        h = add(eng, P_HL1, 3)
        for _ in range(n):
            eng.use_card("古币", h)
        eng.fire(m, "open")
        eng.fire(m, "battle_start")
        rv[n] = m.counters.get("revive", 0)
    assert rv[5] - rv[0] == 1 and rv[10] - rv[0] == 2, f"每 5 张古币 +1 次复生：0/5/10 张 → {rv}"


def _skz_gain(eng) -> float:
    """日志里司空震拿到的临时等级合计（Engine.temp 会记 "临时等级" 日志）。"""
    return sum(kw.get("n", 0) for _, ev, kw in eng.s.log if ev == "临时等级" and kw.get("unit") == "司空震")


def t_司空震():
    """你的其他英雄获得临时等级时，司空震获得一半效果（不超过自身真实等级的1倍）。
    Engine.temp 不发事件（接口缺口），所以不要求"temp 一调用立刻分到"；只要求到这场战斗结束为止分到正确的量
    （按日志合计，因为战斗结束临时等级会清零）。"""
    need("司空震")
    eng = mk(0)
    s = add(eng, "司空震", 0, level=10)
    h = add(eng, P_ZL1, 3)
    sm = add(eng, "玄雍士兵", 4, kind="summon")
    eng.temp(h, 8, src="测试")
    eng.temp(sm, 8, src="测试")
    eng.temp(s, 5, src="测试")
    eng.battle()
    g = _skz_gain(eng) - 5
    assert g == 4, f"其他英雄 +8 → 司空震应 +4（召唤物、自己获得的不算），实际 {g}"
    eng = mk(0)
    s = add(eng, "司空震", 0, level=3)
    h = add(eng, P_ZL1, 3)
    eng.temp(h, 20, src="测试")
    eng.battle()
    g = _skz_gain(eng)
    assert g == 3, f"一半 10 应被真实等级 3 封顶，实际 {g}"
    # 司空震上场之前别人已经拿到的临时等级不算（"获得…时"只看之后）
    eng = mk(0)
    h = add(eng, P_ZL1, 3)
    eng.temp(h, 8, src="测试")
    eng.deploy("司空震")
    eng.battle()
    assert _skz_gain(eng) == 0, f"上场前别人已有的临时等级不该分到，实际 {_skz_gain(eng)}"


def t_姜子牙():
    """你每次使用不在场的阵营以及无阵营的英雄牌时，姜子牙和随机3名其他不同阵营的英雄等级+1。"""
    need("姜子牙")
    for seed in range(8):
        eng = mk(seed)
        j = add(eng, "姜子牙", 0, level=5)
        os_ = [add(eng, P_ZL1, 3, level=5), add(eng, P_HL1, 4, level=5),
               add(eng, P_DH, 5, level=5), add(eng, P_RL, 6, level=5)]
        s0 = snap(eng)
        n = eng.deploy(P_SF1)                              # 三分不在场
        allo = os_ + [n]
        g = [d(eng, s0, o) if o is not n else n.level - 1 for o in allo]
        assert d(eng, s0, j) == 1, f"seed{seed} 姜子牙应 +1：{d(eng, s0, j)}"
        assert sum(g) == 3 and max(g) == 1, f"seed{seed} 应随机 3 名其他英雄各 +1：{g}"
        facs = [faction(o.name) for o, x in zip(allo, g) if x]
        assert len(set(facs)) == 3, f"seed{seed} 3 名应是不同阵营：{facs}"
        s0 = snap(eng)
        eng.deploy(P_ZL2)                                  # 逐鹿已在场（墨子）
        assert d(eng, s0, j) == 0, f"seed{seed} 在场阵营的英雄牌不该触发"
        s0 = snap(eng)
        eng.deploy(P_WU)                                   # 无阵营
        assert d(eng, s0, j) == 1, f"seed{seed} 无阵营英雄牌应触发"


# ================================================================ 运行
TESTS = [
    ("翩跹舞步·余韵", _boost_choice("翩跹舞步·余韵")),
    ("翩跹舞步·进化", _boost_choice("翩跹舞步·进化")),
    ("翩跹舞步·流动", _boost_choice("翩跹舞步·流动")),
    ("翩跹舞步·蓄势", _boost_choice("翩跹舞步·蓄势")),
    ("翩跹舞步·蓄势(击杀+1)", t_翩跹舞步_蓄势_击杀),
    ("惊鸿游龙", t_惊鸿游龙),
    ("即刻起舞", t_即刻起舞),
    ("曜", t_曜), ("孙膑", t_孙膑), ("狂铁", t_狂铁), ("孙尚香", t_孙尚香), ("牛魔", t_牛魔), ("亚连", t_亚连), ("米莱狄", t_米莱狄),
    ("项羽", t_项羽), ("鬼谷子", t_鬼谷子), ("先知图腾", t_先知图腾), ("玄微之种", t_玄微之种), ("梦奇", t_梦奇),
    ("曹操", t_曹操), ("马可波罗", t_马可波罗), ("赵云", t_赵云), ("公孙离", t_公孙离), ("百里玄策", t_百里玄策),
    ("援护战术", t_援护战术), ("招募战术", t_招募战术), ("演练战术", t_演练战术), ("蒙恬", t_蒙恬), ("蔡文姬", t_蔡文姬),
    ("云中君", t_云中君), ("姬小满", t_姬小满), ("白起", t_白起), ("阿轲", t_阿轲), ("裴擒虎", t_裴擒虎),
    ("老夫子", t_老夫子), ("猪八戒", t_猪八戒), ("张良", t_张良), ("亚瑟", t_亚瑟), ("貂蝉", t_貂蝉),
    ("安琪拉", t_安琪拉), ("镜", t_镜), ("朵莉亚", t_朵莉亚), ("莱西奥", t_莱西奥), ("上官婉儿", t_上官婉儿),
    ("李信", t_李信), ("李白", t_李白), ("雅典娜", t_雅典娜), ("甄姬", t_甄姬), ("小乔", t_小乔), ("大司命", t_大司命),
    ("往生图腾", _wangsheng("往生图腾")), ("往生图腾（山鬼形态）", _wangsheng("往生图腾（山鬼形态）")),
    ("庄周", t_庄周), ("刘邦", t_刘邦), ("韩信", t_韩信), ("蒙犽", t_蒙犽), ("东皇太一", t_东皇太一),
    ("敖隐", t_敖隐), ("机关傀儡", t_机关傀儡), ("花木兰", t_花木兰), ("司空震", t_司空震), ("姜子牙", t_姜子牙),
]


def main() -> int:
    batch = json.loads((ROOT / "data/engine/sim_batches.json").read_text(encoding="utf-8"))["batches"][0]
    covered = {n.split("(")[0] for n, _ in TESTS} | set(COMBAT_ONLY) | set(NOT_TESTABLE)
    missing = [n for n in batch if n not in covered]
    ok = fail = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"OK    {name}")
            ok += 1
        except (Unimpl, AssertionError) as e:
            print(f"FAIL  {name}: {e}")
            fail += 1
        except Exception as e:                             # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()
            print(f"FAIL  {name}: 异常 {type(e).__name__}: {e} @ {tb[-3].strip() if len(tb) >= 3 else ''}")
            fail += 1
    for n, why in COMBAT_ONLY.items():
        print(f"SKIP  {n}: 只有战斗效果（{why}）")
    for n, why in NOT_TESTABLE.items():
        print(f"SKIP  {n}: {why}")
    for g in GAP_NOTES:
        print(f"SKIP  {g}")
    if missing:
        print(f"注意：本批这些卡没有归类 {missing}")
    print(f"\n通过 {ok} / 失败 {fail}（共 {ok + fail} 条测试，跳过 {len(COMBAT_ONLY) + len(NOT_TESTABLE)} 张）")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
