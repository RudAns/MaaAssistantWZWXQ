"""第 6 批卡牌的独立测试：**只按 data/cards.json 卡面原文写**，没有读 agent/sim/cards/batch_06.py。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_batch_06.py
逐条打印 OK / FAIL（含原因），末尾打印通过数 / 失败数，有失败退出码 1。
没注册的卡记 FAIL "未实现"，不抛异常。

调用约定（照 agent/sim/play.py 的推演打法）：
- 选中天赋：先 s.talents.append(名)，有 on_use 再 use_card(名, None)。
- 打出手牌：牌留在手上调用 use_card(名, 目标)，依次试 [候选目标…, None]，第一个不返回 False 的算打出，之后从手牌移除。
- 临时刷新记在 s.free_refresh（其他批次的约定）；"仅本回合"的也允许记在 flags 里（键名含"刷新"/"refresh"）。
- 可购买天赋：调实现类的 buy(eng)（第 3 批扩编名额的约定）。
- 商店购买事件：eng.emit("buy", card=名)（第 7 批的约定，引擎本身没有）。

只有战斗效果的卡不写测试（COMBAT_ONLY）；影响养成但搭不出局面的列在 NOT_TESTABLE。运行时都打印 SKIP。
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
from agent.sim.engine import Engine, impl, card_db, quality, category, has_keyword   # noqa: E402
import agent.sim.engine as _E                                                     # noqa: E402
_E.EQUIP_DROP_ROUNDS = ()      # 固定回合投放装备（开局 / 第 4/8/12/16 回合）是引擎规则，卡牌测试里关掉，免得混进手牌 / 装备数的断言
from agent.sim.state import State, Unit                  # noqa: E402

# ---------------------------------------------------------------- 只有战斗效果的卡（不测）
COMBAT_ONLY = [
    "强者恒强", "破败神威", "绝境逆袭", "虚构武装", "同心协力",
    "枫灵涌动", "有叶新生", "枫佑永驻", "拈花一指", "红枫疾舞", "枫华之刃", "枫影无拘", "斗转星移",
    "青叶绽蕊", "层林遮影", "一叶双生", "瞬步无影", "踏叶摘星",
    "潮起潮落", "不朽坚冰", "寒潮之势",
]
# 影响养成、但卡面 / 现有接口搭不出可断言局面的卡
NOT_TESTABLE = {
    "余音绕梁": "惊鸿调的效果取决于'记录的前 6 次指法'，卡面没给出，State 也没有演奏次数",
    "终极宝藏正是在下": "卡面没写 3 个终极宝藏各是什么",
    "瓜分宝藏": "卡面没写 2 种英雄强化效果各是什么",
}

RECIPES = json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8"))["recipes"]
BASE_EQUIPS = set(RECIPES)


class Unimpl(Exception):
    pass


# ---------------------------------------------------------------- 搭局面
def mk(seed=0, **kw) -> Engine:
    return Engine(State(**kw), random.Random(seed))


def add(eng: Engine, name: str, cell_idx: int = 0, level: float = 1.0, kind: str = "hero", awaken: int = 0) -> Unit:
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


def use(eng: Engine, name: str, target=None):
    need(name)
    return eng.use_card(name, target)


def take_talent(eng: Engine, name: str):
    """选中天赋（play.py 的做法）。"""
    c = need(name)
    eng.s.talents.append(name)
    if hasattr(c, "on_use"):
        eng.use_card(name, None)


def play_from_hand(eng: Engine, name: str, targets=()) -> bool:
    """打出手牌（play.py 的做法）：牌在手上时调用，依次试目标，成功后从手牌移除。"""
    need(name)
    assert name in eng.s.hand, f"手牌里没有 {name}：{eng.s.hand}"
    for t in list(targets) + [None]:
        if eng.use_card(name, t) is not False:
            if name in eng.s.hand:
                eng.s.hand.remove(name)
            return True
    return False


def next_round(eng: Engine):
    eng.round_end()
    eng.round_start()


def recorder(eng: Engine, pick=-1):
    calls = []

    def ch(e, kind, options, **ctx):
        calls.append((kind, list(options)))
        return options[pick]
    eng.chooser = ch
    return calls


def refreshes(eng: Engine) -> float:
    """可用的临时 / 免费刷新次数：free_refresh + flags 里键名含"刷新"/"refresh"的整数。"""
    n = eng.s.free_refresh
    for k, v in eng.s.flags.items():
        if ("刷新" in k or "refresh" in k.lower()) and isinstance(v, (int, float)) and not isinstance(v, bool):
            n += v
    return n


def temp_by_unit(eng: Engine) -> dict:
    out = {}
    for (_, ev, kw) in eng.s.log:
        if ev == "临时等级":
            out[kw.get("unit")] = out.get(kw.get("unit"), 0) + kw.get("n", 0)
    return out


HEROES = sorted([n for n, c in card_db().items() if c.get("category") == "英雄牌"], key=len, reverse=True)


def hero_base(name: str):
    """英雄牌（含"转瞬版"之类带后缀的复制）→ 英雄名；不是英雄牌返回 None。"""
    if category(name) == "英雄牌":
        return name
    for h in HEROES:
        if h in name:
            return h
    return None


def _hooks(c):
    return [a for a in dir(c) if a.startswith(("on_", "react_", "coin_bonus", "grant_keywords"))]


def plain(tier: int, k: int) -> list:
    """k 个"白板"英雄（没有任何养成钩子），按品阶取。"""
    out = [n for n in card_db() if category(n) == "英雄牌" and quality(n) == tier
           and (impl(n) is None or not _hooks(impl(n)))]
    assert len(out) >= k, f"{tier}阶白板英雄不够 {k} 个（测试数据有误）：{out}"
    return out[:k]


def is_exchange(name: str, tiers) -> bool:
    return "兑换券" in name and any(str(t) in name for t in tiers)


def is_jinnang(name: str, tiers) -> bool:
    return "效果锦囊" in name and any(str(t) in name for t in tiers)


# ================================================================ 天赋：经济 / 发牌
def t_群星璀璨():
    """获得2张4-5阶英雄兑换券。至少5名4-5阶英雄时，所有4-5阶英雄获得20临时等级（战斗开始时）。"""
    eng = mk(0)
    take_talent(eng, "群星璀璨")
    h = eng.s.hand
    assert len(h) == 2 and all(is_exchange(x, (4, 5)) for x in h), f"应得 2 张 4-5 阶英雄兑换券，手牌 {h}"
    hi = plain(4, 3) + plain(5, 2)
    lo = plain(2, 1)[0]
    for seed in range(4):
        eng = mk(seed)
        eng.s.talents.append("群星璀璨")
        for i, n in enumerate(hi + [lo]):
            add(eng, n, i * 3)
        eng.battle()
        t = temp_by_unit(eng)
        for n in hi:
            assert abs(t.get(n, 0) - 20) < 1e-9, f"seed{seed} 5 名 4-5 阶：{n} 临时等级 {t.get(n, 0)}，应 20"
        assert t.get(lo, 0) == 0, f"2阶英雄 {lo} 也拿了临时等级 {t.get(lo)}"
    eng = mk(0)                                            # 只有 4 名 4-5 阶 → 不触发
    eng.s.talents.append("群星璀璨")
    for i, n in enumerate(hi[:4] + [lo]):
        add(eng, n, i * 3)
    eng.battle()
    assert not any(temp_by_unit(eng).values()), f"只有 4 名 4-5 阶也触发了 {temp_by_unit(eng)}"


def t_特大电池包():
    eng = mk(0, energy=3)
    take_talent(eng, "特大电池包")
    assert eng.s.energy == 15, f"能量 3 → {eng.s.energy}，应 15"


def t_随机强化():
    """战斗开始时，品阶最高的1名英雄等级永久+9。"""
    a2, a3, a5 = plain(2, 1)[0], plain(3, 1)[0], plain(5, 1)[0]
    for seed in range(4):
        eng = mk(seed)
        eng.s.talents.append("随机强化")
        us = [add(eng, a2, 0), add(eng, a3, 3), add(eng, a5, 6)]
        eng.battle()
        assert [u.level for u in us] == [1, 1, 10], f"seed{seed} 等级 {[(u.name, u.level) for u in us]}，只有 5 阶应 +9"
    got = set()
    b5 = plain(5, 2)
    for seed in range(12):                                 # 并列最高：只给其中 1 名
        eng = mk(seed)
        eng.s.talents.append("随机强化")
        us = [add(eng, b5[0], 0), add(eng, b5[1], 3), add(eng, a2, 6)]
        eng.battle()
        lv = [u.level for u in us]
        assert sorted(lv) == [1, 1, 10] and lv[2] == 1, f"seed{seed} 并列最高时 {lv}，应恰好 1 名 5 阶 +9"
        got.add(lv.index(10))
    assert got == {0, 1}, f"并列时 12 个种子只选过 {got}，不像随机"


def t_无影手():
    """回合开始时，获得3次仅本回合可用的免费商店刷新。"""
    eng = mk(0)
    take_talent(eng, "无影手")
    r0 = refreshes(eng)
    eng.round_start()
    assert refreshes(eng) - r0 == 3, f"回合开始后刷新次数 {r0} → {refreshes(eng)}，应 +3"
    next_round(eng)                                        # 上回合没用完的过期
    assert refreshes(eng) - r0 == 3, f"第二回合开始后刷新次数 {refreshes(eng)}（上回合剩的应过期，仍是 {r0}+3）"


def t_学霸():
    """获得5能量（"第1个到达5/6级"需要对手信息，不测）。"""
    eng = mk(0, energy=2)
    take_talent(eng, "学霸")
    assert eng.s.energy == 7, f"能量 2 → {eng.s.energy}，应 7"


def t_留级证明():
    """保持在棋手等级1时，每回合获得1能量。（一整回合：回合开始→战斗→回合结束，扣掉回合收入）"""
    for lv, exp in ((1, 1), (2, 0)):
        eng = mk(0, lord_level=lv, xp=0)
        eng.s.talents.append("留级证明")
        e0 = eng.s.energy
        eng.round_start()
        eng.battle()
        eng.round_end()
        assert eng.s.lord_level == lv, "测试前提：这一回合棋手没升级"
        d = eng.s.energy - e0 - B.income(eng.s.round)
        assert d == exp, f"棋手 {lv} 级：一回合额外能量 {d}，应 {exp}"


def t_买橘子去():
    eng = mk(0)
    take_talent(eng, "买橘子去")
    assert eng.s.hand == ["站这别动"], f"手牌 {eng.s.hand}，应只有 1 张站这别动"


def t_站这别动():
    """指定1个阵位，每过2回合，获得1张此格上的英雄牌。（经由 买橘子去 拿到赠送牌，指定目标 = 该格上的英雄）"""
    a, b = plain(2, 2)
    eng = mk(0)
    take_talent(eng, "买橘子去")
    x = add(eng, a, 5)
    assert play_from_hand(eng, "站这别动", [x]), "站这别动 打不出去"
    next_round(eng)
    assert a not in eng.s.hand, f"才过 1 回合就得到了 {a}：{eng.s.hand}"
    next_round(eng)
    assert eng.s.hand.count(a) == 1, f"过 2 回合应得 1 张 {a}，手牌 {eng.s.hand}"
    cell = x.cell                                          # 换人：该格改站 b
    eng.s.units.remove(x)
    y = add(eng, b, 0)
    y.cell = cell
    next_round(eng)
    next_round(eng)
    assert eng.s.hand.count(b) == 1 and eng.s.hand.count(a) == 1, \
        f"过 4 回合（第 3~4 回合格上是 {b}）应各 1 张，手牌 {eng.s.hand}"


def t_战略中枢():
    eng = mk(0)
    take_talent(eng, "战略中枢")
    assert eng.s.pop_bonus == 1, f"pop_bonus {eng.s.pop_bonus}，应 1"
    assert sorted(eng.s.hand) == ["有福同享", "有福同享"], f"手牌 {eng.s.hand}"


def t_延迟满足():
    """获得1张天赋牌延迟满足：指定1名英雄等级+10。3回合后，再+25。"""
    a = plain(2, 1)[0]
    eng = mk(0)
    take_talent(eng, "延迟满足")
    assert eng.s.hand == ["延迟满足"], f"选中天赋后手牌 {eng.s.hand}，应 1 张延迟满足"
    x = add(eng, a, 3)
    assert play_from_hand(eng, "延迟满足", [x]), "赠送牌打不出去"
    assert x.level == 11, f"打出后等级 {x.level}，应 1+10"
    assert "延迟满足" not in eng.s.hand, f"打出后又多了一张：{eng.s.hand}"
    next_round(eng)
    next_round(eng)
    assert x.level == 11, f"才过 2 回合等级就变成 {x.level}"
    next_round(eng)
    assert x.level == 36, f"3 回合后等级 {x.level}，应 11+25"
    next_round(eng)
    next_round(eng)
    assert x.level == 36, f"+25 只给一次，现在 {x.level}"


def t_全副武装():
    eng = mk(0)
    take_talent(eng, "全副武装")
    h = eng.s.hand
    assert len(h) == 1 and h[0] in BASE_EQUIPS, f"应得 1 张基础装备牌，手牌 {h}"


def t_三人游():
    for seed in range(6):
        eng = mk(seed)
        take_talent(eng, "三人游")
        h = eng.s.hand
        assert len(h) == 3, f"seed{seed} 手牌 {h}，应 3 张"
        bad = [x for x in h if category(x) != "英雄牌" or quality(x) not in (4, 5)]
        assert not bad, f"seed{seed} 不是 4-5 阶英雄牌：{bad}"


def t_扩编高峰():
    """获得1个可购买天赋：花费8能量，可上阵人数+1。可购买3次。（购买 = buy(eng)）"""
    c = need("扩编高峰")
    if not hasattr(c, "buy"):
        raise AssertionError("没有购买接口 buy(eng)（第 3 批扩编名额的约定）")
    eng = mk(0, energy=30)
    take_talent(eng, "扩编高峰")
    assert eng.s.pop_bonus == 0 and eng.s.energy == 30, "选中天赋时就加了人口 / 扣了能量"
    res = [c.buy(eng) for _ in range(4)]
    assert eng.s.pop_bonus == 3, f"买 4 次后 pop_bonus {eng.s.pop_bonus}，应 3（限 3 次）结果 {res}"
    assert eng.s.energy == 6, f"能量 30 → {eng.s.energy}，应 30-24"
    eng = mk(0, energy=7)
    take_talent(eng, "扩编高峰")
    c.buy(eng)
    assert eng.s.pop_bonus == 0 and eng.s.energy == 7, "能量不够 8 也买成了"


def _supply(kw):
    def t():
        """获得1张{kw}效果英雄牌。回合开始时，再获得1张。"""
        name = f"补给线：{kw}"
        for seed in range(5):
            eng = mk(seed, lord_level=3)
            take_talent(eng, name)
            h = list(eng.s.hand)
            assert len(h) == 1, f"seed{seed} 选中后手牌 {h}，应 1 张"
            eng.round_start()
            h = list(eng.s.hand)
            assert len(h) == 2, f"seed{seed} 回合开始后手牌 {h}，应 2 张"
            # "X效果英雄牌"：卡库关键词表里有 X 或卡面写着「X：」（卡库标签把"每当…触发X时"的反应器也算进去）
            bad = [x for x in h if category(x) != "英雄牌" or not has_keyword(x, kw)]
            assert not bad, f"seed{seed} 不是{kw}效果英雄牌：{bad}"
    return t


def t_中流砥柱():
    """打出3阶英雄牌，额外+6级。（上场和合成都算打出）"""
    a3, b3 = plain(3, 2)
    a2 = plain(2, 1)[0]
    eng = mk(0)
    take_talent(eng, "中流砥柱")
    u = eng.deploy(a3)
    assert u is not None and u.level == 7, f"3阶上场后等级 {u and u.level}，应 1+6"
    v = eng.deploy(a2)
    assert v.level == 1, f"2阶上场也加了：{v.level}"
    eng.merge(u)
    assert u.level == 7 + 3 + 6, f"3阶合成后等级 {u.level}，应 7+3(合成)+6"


def t_混沌次元():
    eng = mk(0)
    take_talent(eng, "混沌次元")
    assert eng.s.hand == ["混沌裂隙"], f"手牌 {eng.s.hand}，应 1 张混沌裂隙"


def t_混沌裂隙():
    """指定阵位召唤裂隙，开战前吞噬该格的英雄，永久获得其双倍等级。开战时，召唤1名品阶≥商店等级的英雄。
    2026-10-16 确认的规则事实：只在这一场里换人，战后原英雄回来、不消失；每场都换。
    断言：战后原英雄还在场；出现等级 ≥ 2×10 的召唤英雄；召唤出的英雄品阶 ≥ 棋手等级；第二场照样换。"""
    a = plain(3, 1)[0]
    for seed in range(3):
        eng = mk(seed, lord_level=3)
        take_talent(eng, "混沌次元")
        x = add(eng, a, 9, level=10)
        assert play_from_hand(eng, "混沌裂隙", [x]), "混沌裂隙 打不出去"
        eng.s.log.clear()
        eng.battle()
        assert x in eng.s.units, f"seed{seed} {a} 战后没回来（裂隙只换这一场）"
        # 裂隙等级记在哪由实现定：场上单位等级 / 日志里任一事件的 n 或 level，取最大
        nums = [u.level for u in eng.s.units] + [0]
        for (_, ev, kw) in eng.s.log:
            nums += [v for k, v in kw.items() if k in ("n", "level") and isinstance(v, (int, float))]
        assert max(nums) >= 20, f"seed{seed} 没看到裂隙拿到 2×10 级：日志 {eng.s.log}"
        dep = [kw.get("unit") for (_, ev, kw) in eng.s.log if ev == "上场"] + [u.name for u in eng.s.units]
        hs = [hero_base(n) for n in dep if n and n != a and hero_base(n)]     # 召唤出同名英雄也算
        assert hs, f"seed{seed} 开战时没召唤英雄（上场记录 {dep}）"
        assert all(quality(n) >= 3 for n in hs), f"seed{seed} 召唤的英雄品阶低于商店等级 3：{hs}"
        eng.s.flags.pop("round:混沌裂隙", None)          # 下一回合（每回合一场）
        eng.s.log.clear()
        eng.battle()
        assert any(ev == "裂隙换人" for (_, ev, kw) in eng.s.log), f"seed{seed} 第二场没换人"
        assert x in eng.s.units


def t_小咖秀():
    """战斗开始时，若你只有1-3阶英雄，随机获得1张场上的英雄牌。"""
    low = plain(2, 2) + plain(3, 1)
    got = set()
    for seed in range(10):
        eng = mk(seed)
        eng.s.talents.append("小咖秀")
        for i, n in enumerate(low):
            add(eng, n, i * 4)
        eng.battle()
        h = eng.s.hand
        assert len(h) == 1 and h[0] in low, f"seed{seed} 手牌 {h}，应 1 张场上英雄"
        got.add(h[0])
    assert len(got) >= 2, f"10 个种子只拿过 {got}，不像随机"
    eng = mk(0)
    eng.s.talents.append("小咖秀")
    for i, n in enumerate(low + plain(4, 1)):
        add(eng, n, i * 4)
    eng.battle()
    assert not eng.s.hand, f"有 4 阶英雄也触发了：{eng.s.hand}"


def t_永不独行():
    """获得1张双人同行。回合开始时，若棋手不超过3级，再获得1张。"""
    for lv, exp in ((3, 2), (4, 1)):
        eng = mk(0, lord_level=lv)
        take_talent(eng, "永不独行")
        assert len(eng.s.hand) == 1, f"选中后手牌 {eng.s.hand}"
        eng.round_start()
        h = eng.s.hand
        assert len(h) == exp and all(x.startswith("双人同行") for x in h), f"棋手 {lv} 级回合开始后手牌 {h}，应 {exp} 张双人同行"


def _duo(tier):
    def t():
        names = set()
        for seed in range(8):
            eng = mk(seed)
            use(eng, f"双人同行·{tier}阶")
            h = eng.s.hand
            assert len(h) == 2 and h[0] == h[1], f"seed{seed} 应得 2 张相同的牌，手牌 {h}"
            assert category(h[0]) == "英雄牌" and quality(h[0]) == tier, f"seed{seed} {h[0]} 不是 {tier} 阶英雄牌"
            names.add(h[0])
        assert len(names) >= 2, f"8 个种子只给过 {names}，不像随机"
    return t


def t_搏命():
    eng = mk(0, energy=1)
    take_talent(eng, "搏命")
    assert eng.s.energy == 26, f"能量 1 → {eng.s.energy}，应 26"


def t_紧急救援():
    eng = mk(0)
    take_talent(eng, "紧急救援")
    h = eng.s.hand
    assert len(h) == 2, f"手牌 {h}，应 2 张（装备箱 + 5阶兑换券）"
    assert "装备箱" in h, f"没有装备箱：{h}"
    assert any(is_exchange(x, (5,)) for x in h), f"没有 5 阶兑换券：{h}"


def t_危险数字():
    """在商店每买3张牌，获得1能量；若其品阶相加等于10，随机再获得其中1张。（购买事件 emit("buy", card=)）"""
    c = need("危险数字")
    if not hasattr(c, "react_buy"):
        raise AssertionError("没有 react_buy（第 7 批的购买事件约定）")
    t3 = plain(3, 2)
    t4 = plain(4, 1)[0]
    t2 = plain(2, 3)
    got = set()
    for seed in range(8):
        eng = mk(seed, energy=0)
        eng.s.talents.append("危险数字")
        for n in (t3[0], t3[1]):
            eng.emit("buy", card=n)
        assert eng.s.energy == 0 and not eng.s.hand, f"seed{seed} 才买 2 张就给了东西"
        eng.emit("buy", card=t4)                           # 3+3+4 = 10
        assert eng.s.energy == 1, f"seed{seed} 买满 3 张能量 {eng.s.energy}，应 1"
        assert len(eng.s.hand) == 1 and eng.s.hand[0] in (t3[0], t3[1], t4), f"seed{seed} 品阶和 10 应再得其中 1 张：{eng.s.hand}"
        got.add(eng.s.hand[0])
        for n in t2:                                       # 2+2+2 = 6
            eng.emit("buy", card=n)
        assert eng.s.energy == 2 and len(eng.s.hand) == 1, f"seed{seed} 第二组 3 张：能量 {eng.s.energy} 手牌 {eng.s.hand}"
    assert len(got) >= 2, f"8 个种子只给过 {got}，不像随机"


def t_大明星():
    """获得1张大明星：随机上阵1名不在场的等级为棋手等级x7的5阶英雄，其觉醒进度为1，此后每经过2回合则额外+1。"""
    on5 = plain(5, 1)[0]
    names = set()
    for seed in range(5):
        eng = mk(seed, lord_level=3)
        old = add(eng, on5, 0, level=5)
        take_talent(eng, "大明星")
        assert eng.s.hand == ["大明星"], f"seed{seed} 选中后手牌 {eng.s.hand}"
        assert play_from_hand(eng, "大明星", [old]), "大明星 打不出去"
        new = [u for u in eng.s.units if u is not old]
        assert len(new) >= 1, f"seed{seed} 没有上阵新英雄"
        star = next((u for u in new if category(u.name) == "英雄牌" and quality(u.name) == 5), None)
        assert star is not None, f"seed{seed} 上阵的不是 5 阶英雄：{[u.name for u in new]}"
        assert star.name != on5, f"seed{seed} 上阵了在场的 {on5}"
        assert star.level >= 21, f"seed{seed} {star.name} 等级 {star.level}，应 3×7=21"
        assert star.awaken == 1, f"seed{seed} 觉醒进度 {star.awaken}，应 1"
        names.add(star.name)
        next_round(eng)
        assert star.awaken == 1, f"seed{seed} 过 1 回合觉醒进度就变成 {star.awaken}"
        next_round(eng)
        assert star.awaken == 2, f"seed{seed} 过 2 回合觉醒进度 {star.awaken}，应 2"
        next_round(eng)
        next_round(eng)
        assert star.awaken >= 3, f"seed{seed} 过 4 回合觉醒进度 {star.awaken}，应 3"
    assert len(names) >= 2, f"5 个种子只上过 {names}，不像随机"


def t_王牌部队():
    """战斗开始时，若你至少有3名4-5阶英雄，获得1张4-5阶效果锦囊。"""
    hi = plain(4, 2) + plain(5, 1)
    eng = mk(0)
    eng.s.talents.append("王牌部队")
    for i, n in enumerate(hi):
        add(eng, n, i * 4)
    eng.battle()
    h = eng.s.hand
    assert len(h) == 1 and is_jinnang(h[0], (4, 5)), f"3 名 4-5 阶：手牌 {h}，应 1 张 4-5 阶效果锦囊"
    next_round(eng)
    eng.battle()
    assert len(eng.s.hand) == 2, f"下一回合战斗后手牌 {eng.s.hand}，应再得 1 张"
    eng = mk(0)
    eng.s.talents.append("王牌部队")
    for i, n in enumerate(hi[:2] + plain(3, 1)):
        add(eng, n, i * 4)
    eng.battle()
    assert not eng.s.hand, f"只有 2 名 4-5 阶也给了：{eng.s.hand}"


def t_有舍有得():
    """回合开始时，随机1名英雄觉醒进度+1，但英雄合成提升的等级-1。"""
    a, b = plain(2, 2)
    got = set()
    for seed in range(8):
        eng = mk(seed)
        take_talent(eng, "有舍有得")
        us = [add(eng, a, 0), add(eng, b, 4)]
        add(eng, "因缘图腾", 8, kind="summon")
        eng.round_start()
        aw = [u.awaken for u in us]
        assert sorted(aw) == [0, 1], f"seed{seed} 觉醒进度 {aw}，应恰好 1 名 +1"
        got.add(aw.index(1))
    assert got == {0, 1}, f"8 个种子只选过 {got}，不像随机"
    eng = mk(0)
    take_talent(eng, "有舍有得")
    u = add(eng, a, 0, level=5)
    eng.merge(u)
    assert u.level == 5 + 3 - 1, f"合成后等级 {u.level}，应 5+3-1"


def t_资深猎头():
    """棋手4级时，获得1张天赋牌资深猎头：4阶卡牌2选1，并将选中的卡牌一分为四。"""
    eng = mk(0, lord_level=3, xp=0)
    take_talent(eng, "资深猎头")
    assert not eng.s.hand, f"棋手 3 级就给了牌：{eng.s.hand}"
    eng.s.xp = B.XP_NEED[3]
    eng.level_ups()
    assert eng.s.lord_level == 4
    assert eng.s.hand == ["资深猎头"], f"升到 4 级后手牌 {eng.s.hand}，应 1 张资深猎头"
    calls = recorder(eng, pick=-1)
    assert play_from_hand(eng, "资深猎头", []), "资深猎头 打不出去"
    two = [c for c in calls if len(c[1]) == 2]
    assert two, f"没有 2 选 1：{calls}"
    opts = two[-1][1]
    assert all(quality(x) == 4 for x in opts), f"选项不全是 4 阶：{opts}"
    pick = opts[-1]
    assert eng.s.hand.count(pick) == 4 and len(eng.s.hand) == 4, f"选了 {pick}，手牌 {eng.s.hand}，应 4 张"
    eng.s.xp = B.XP_NEED[4]
    eng.level_ups()
    assert len(eng.s.hand) == 4, f"升到 5 级又给了：{eng.s.hand}"


def t_无限进步():
    """获得1张3阶兑换券。回合开始时，获得2棋手经验。满级后，改为获得2能量。"""
    eng = mk(0, lord_level=2, xp=0)
    take_talent(eng, "无限进步")
    h = eng.s.hand
    assert len(h) == 1 and is_exchange(h[0], (3,)), f"手牌 {h}，应 1 张 3 阶兑换券"
    e0 = eng.s.energy
    eng.round_start()
    assert eng.s.xp == 2 and eng.s.energy == e0, f"回合开始：经验 {eng.s.xp}（应 2）能量 {e0}→{eng.s.energy}"
    eng = mk(0, lord_level=B.MAX_LORD_LEVEL, xp=0, energy=0)
    eng.s.talents.append("无限进步")
    eng.round_start()
    assert eng.s.energy == 2 and eng.s.xp == 0, f"满级回合开始：能量 {eng.s.energy}（应 2）经验 {eng.s.xp}（应 0）"


# ================================================================ 效果牌
def t_心之引力():
    """获得4次临时刷新（购买消耗 -1 不测：引擎没有商店扣费）。"""
    eng = mk(0)
    r0 = refreshes(eng)
    use(eng, "心之引力")
    assert refreshes(eng) - r0 == 4, f"刷新次数 {r0} → {refreshes(eng)}，应 +4"


def t_魅力无限():
    """每回合打出的第1张效果牌会触发2次。（用 pilot 的特训战术 +4 来测；魅力无限自己算不算"第 1 张"有歧义，跨回合再测）"""
    assert category("特训战术") == "效果牌"
    a = plain(2, 1)[0]
    eng = mk(0)
    x = add(eng, a, 0)
    use(eng, "魅力无限")
    next_round(eng)
    lv = x.level
    use(eng, "特训战术", x)
    assert x.level - lv == 8, f"本回合第 1 张效果牌：+{x.level - lv}，应 +8（触发 2 次）"
    lv = x.level
    use(eng, "特训战术", x)
    assert x.level - lv == 4, f"本回合第 2 张效果牌：+{x.level - lv}，应 +4"
    next_round(eng)
    lv = x.level
    use(eng, "特训战术", x)
    assert x.level - lv == 8, f"下回合第 1 张效果牌：+{x.level - lv}，应 +8"


def t_万镜():
    """随机复制场上4张不同的英雄牌。"""
    hs = plain(2, 3) + plain(3, 2)
    for seed in range(6):
        eng = mk(seed)
        for i, n in enumerate(hs):
            add(eng, n, i * 3)
        add(eng, "因缘图腾", 20, kind="summon")
        use(eng, "万镜")
        h = eng.s.hand
        assert len(h) == 4 and len(set(h)) == 4, f"seed{seed} 手牌 {h}，应 4 张不同"
        assert set(h) <= set(hs), f"seed{seed} 复制了场上没有的英雄 {set(h) - set(hs)}"


def t_潮汐的交响():
    """场上随机5名英雄等级+8。"""
    hs = plain(2, 3) + plain(3, 3)
    for seed in range(6):
        eng = mk(seed)
        us = [add(eng, n, i * 3) for i, n in enumerate(hs)]
        tt = add(eng, "因缘图腾", 20, kind="summon")
        use(eng, "潮汐的交响")
        lv = sorted(u.level for u in us)
        assert lv == [1, 9, 9, 9, 9, 9], f"seed{seed} 等级 {lv}，应 5 名 +8"
        assert tt.level == 1, "召唤物也加了"


def t_扫拂():
    """使场上等级最低的2名英雄获得等于棋手等级的等级。"""
    a, b, c = plain(2, 3)
    eng = mk(0, lord_level=4)
    ua, ub, uc = add(eng, a, 0, level=1), add(eng, b, 3, level=3), add(eng, c, 6, level=10)
    use(eng, "扫拂")
    assert (ua.level, ub.level, uc.level) == (5, 7, 10), f"等级 {(ua.level, ub.level, uc.level)}，应 (5,7,10)"


def t_吟揉():
    """获得1张随机的转瞬英雄牌。"""
    for seed in range(5):
        eng = mk(seed)
        use(eng, "吟揉")
        h = eng.s.hand
        assert len(h) == 1 and hero_base(h[0]), f"seed{seed} 手牌 {h}，应 1 张英雄牌（转瞬版）"


def t_绞弦():
    """场上随机英雄等级+1。"""
    hs = plain(2, 3)
    got = set()
    for seed in range(10):
        eng = mk(seed)
        us = [add(eng, n, i * 3) for i, n in enumerate(hs)]
        use(eng, "绞弦")
        d = [u.level - 1 for u in us]
        assert sorted(d) == [0, 0, 1], f"seed{seed} 增量 {d}，应恰好 1 名 +1"
        got.add(d.index(1))
    assert len(got) >= 2, f"10 个种子只选过 {got}，不像随机"


def t_如梦似幻():
    """基于场上英雄、阵营（至少3名）获取一套梦幻手牌。
    待核对："一套梦幻手牌"内容卡面没写，和实现统一成"该阵营在场英雄的同名英雄牌"：
    测 3 名同阵营 → 得到 ≥1 张牌、都是该阵营的英雄牌；不满 3 名不能用。"""
    db = card_db()
    by_f = {}
    for n in db:
        if category(n) == "英雄牌" and db[n].get("faction"):
            by_f.setdefault(db[n]["faction"], []).append(n)
    fac, ns = next((f, v) for f, v in by_f.items() if len(v) >= 3)
    eng = mk(0)
    for i, n in enumerate(ns[:3]):
        add(eng, n, i * 4)
    use(eng, "如梦似幻")
    assert len(eng.s.hand) >= 1, f"3 名【{fac}】英雄在场，什么也没得到"
    bad = [n for n in eng.s.hand if category(n) != "英雄牌" or db[n].get("faction") != fac]
    assert not bad, f"梦幻手牌里有不是【{fac}】英雄牌的：{bad}"
    eng2 = mk(0)
    for i, n in enumerate(ns[:2]):
        add(eng2, n, i * 4)
    assert use(eng2, "如梦似幻") is False and not eng2.s.hand, "只有 2 名同阵营英雄却能用"


def t_红笺锦鲤符():
    a = plain(2, 1)[0]
    eng = mk(0)
    x = add(eng, a, 0, level=3)
    use(eng, "红笺锦鲤符", x)
    assert x.level == 7, f"等级 {x.level}，应 3+4"
    assert use(eng, "红笺锦鲤符", None) is False, "没有目标却打出成功"


def t_狼牙部落的图腾雕像():
    """抽取1张3阶及以下的英雄牌。"""
    for seed in range(8):
        eng = mk(seed, lord_level=5)
        use(eng, "狼牙部落的图腾雕像")
        h = eng.s.hand
        assert len(h) == 1 and category(h[0]) == "英雄牌" and quality(h[0]) <= 3, f"seed{seed} 手牌 {h}"


def t_沙海甘泉():
    eng = mk(0, energy=4)
    use(eng, "沙海甘泉")
    assert eng.s.energy == 5, f"能量 {eng.s.energy}，应 5"


def t_人鱼之泪():
    for seed in range(5):
        eng = mk(seed)
        use(eng, "人鱼之泪")
        h = eng.s.hand
        assert len(h) == 1 and category(h[0]) == "效果牌", f"seed{seed} 手牌 {h}，应 1 张效果牌"


def t_神树彩陶():
    """场上随机4名英雄等级+2。"""
    hs = plain(2, 3) + plain(3, 2)
    for seed in range(6):
        eng = mk(seed)
        us = [add(eng, n, i * 3) for i, n in enumerate(hs)]
        use(eng, "神树彩陶")
        lv = sorted(u.level for u in us)
        assert lv == [1, 3, 3, 3, 3], f"seed{seed} 等级 {lv}，应 4 名 +2"


def t_冰曜宝石镜():
    """随机获得手牌中1张非觉醒英雄牌的复制。（手牌只存卡名，标不出觉醒，觉醒排除不测）"""
    a, b = plain(2, 2)
    got = set()
    for seed in range(10):
        eng = mk(seed)
        eng.s.hand = [a, "古币", b]
        use(eng, "冰曜宝石镜")
        h = eng.s.hand
        assert len(h) == 4, f"seed{seed} 手牌 {h}，应多 1 张"
        extra = list(h)
        for x in (a, "古币", b):
            extra.remove(x)
        assert extra[0] in (a, b), f"seed{seed} 复制出 {extra[0]}，应是手牌里的英雄牌"
        got.add(extra[0])
    assert got == {a, b}, f"10 个种子只复制过 {got}"
    eng = mk(0)
    eng.s.hand = ["古币"]
    use(eng, "冰曜宝石镜")
    assert eng.s.hand == ["古币"], f"手里没有英雄牌却得到了 {eng.s.hand}"


def t_马可的随身信物():
    for seed in range(4):
        eng = mk(seed)
        use(eng, "马可的随身信物")
        h = eng.s.hand
        assert len(h) == 4 and all(category(x) == "效果牌" for x in h), f"seed{seed} 手牌 {h}，应 4 张效果牌"


def t_马可的魅力飞吻():
    hs = plain(2, 3)
    eng = mk(0)
    us = [add(eng, n, i * 3, level=2 + i) for i, n in enumerate(hs)]
    tt = add(eng, "因缘图腾", 20, kind="summon")
    use(eng, "马可的魅力飞吻")
    assert [u.level for u in us] == [12, 13, 14], f"等级 {[u.level for u in us]}，应全体 +10"
    assert tt.level == 1, "召唤物也加了"


def t_马可的限量亲签():
    eng = mk(0, energy=1)
    use(eng, "马可的限量亲签")
    assert eng.s.energy == 5, f"能量 {eng.s.energy}，应 1+4"
    h = eng.s.hand
    assert len(h) == 1 and category(h[0]) == "装备牌", f"手牌 {h}，应 1 张装备牌"


def t_马可的吸睛自拍():
    """随机获得3张不同的场上英雄牌。"""
    hs = plain(2, 3) + plain(3, 1)
    for seed in range(6):
        eng = mk(seed)
        for i, n in enumerate(hs):
            add(eng, n, i * 3)
        use(eng, "马可的吸睛自拍")
        h = eng.s.hand
        assert len(h) == 3 and len(set(h)) == 3 and set(h) <= set(hs), f"seed{seed} 手牌 {h}，应 3 张不同的场上英雄"


def t_即兴创作():
    hs = plain(2, 2)
    got = set()
    for seed in range(10):
        eng = mk(seed)
        for i, n in enumerate(hs):
            add(eng, n, i * 3)
        add(eng, "因缘图腾", 20, kind="summon")
        use(eng, "即兴创作")
        h = eng.s.hand
        assert len(h) == 1 and h[0] in hs, f"seed{seed} 手牌 {h}，应 1 张场上英雄"
        got.add(h[0])
    assert got == set(hs), f"10 个种子只复制过 {got}"


def t_灵感爆发():
    """斩击手牌中随机3张非觉醒英雄牌，每张斩为2张转瞬版复制（= 被斩的英雄各变成 2 张）。"""
    hs = plain(2, 3) + plain(3, 1)
    for seed in range(6):
        eng = mk(seed)
        eng.s.hand = list(hs) + ["古币"]
        use(eng, "灵感爆发")
        h = eng.s.hand
        assert len(h) == 8, f"seed{seed} 手牌 {len(h)} 张：{h}，应 5-3+6=8"
        assert h.count("古币") == 1, f"seed{seed} 古币被动了：{h}"
        cnt = {}
        for x in h:
            if x == "古币":
                continue
            b = hero_base(x)
            assert b in hs, f"seed{seed} 手牌里出现 {x}"
            cnt[b] = cnt.get(b, 0) + 1
        assert sorted(cnt.values()) == [1, 2, 2, 2], f"seed{seed} 各英雄张数 {cnt}，应 3 名各 2 张 + 1 名 1 张"


def t_奇才():
    a = plain(3, 1)[0]
    eng = mk(0)
    x = add(eng, a, 0)
    use(eng, "奇才", x)
    assert eng.s.hand.count(a) == 4 and len(eng.s.hand) == 4, f"手牌 {eng.s.hand}，应 4 张 {a}"


def t_惊才绝艳():
    eng = mk(0)
    use(eng, "惊才绝艳")
    assert eng.s.hand == ["奇才"], f"手牌 {eng.s.hand}，应 1 张奇才"


# ================================================================ 运行
def t_泉水():
    """回合开始时，恢复3棋手生命值；若棋手生命值已满，改为获得4棋手经验。（State.hp，上限 60）"""
    eng = mk(0, hp=50, lord_level=1, xp=0)
    take_talent(eng, "泉水")
    eng.round_start()
    assert eng.s.hp == 53 and eng.s.xp == 0, f"没满血：生命 +3、不给经验：生命 {eng.s.hp} 经验 {eng.s.xp}"
    eng = mk(0, hp=59, lord_level=1, xp=0)
    take_talent(eng, "泉水")
    eng.round_start()
    assert eng.s.hp == 60, f"回复不超过上限 60：{eng.s.hp}"
    eng = mk(0, hp=60, lord_level=1, xp=0)
    take_talent(eng, "泉水")
    eng.round_start()
    assert eng.s.hp == 60 and eng.s.xp == 4, f"满血：改为 +4 经验：生命 {eng.s.hp} 经验 {eng.s.xp}"


def t_雪暴时刻():
    """冻结的英雄，解冻后等级永久+20。（解冻 = thaw 事件，冰之心解冻入口 batch_07.thaw 最后发它）"""
    eng = mk(0)
    use(eng, "雪暴时刻")
    if "雪暴时刻" not in eng.s.lord_cards and "雪暴时刻" not in eng.s.talents:
        eng.s.lord_cards.append("雪暴时刻")
    u = add(eng, "墨子", 3, level=10)                       # 只有战斗效果的英雄
    v = add(eng, "廉颇", 4, level=10)
    eng.emit("thaw", unit=u)
    assert u.level == 30 and v.level == 10, f"解冻的那名 +20、别人不变：{u.level} / {v.level}"


TESTS = [
    ("泉水", t_泉水), ("雪暴时刻", t_雪暴时刻), ("群星璀璨", t_群星璀璨), ("特大电池包", t_特大电池包), ("随机强化", t_随机强化), ("无影手", t_无影手),
    ("学霸", t_学霸), ("留级证明", t_留级证明), ("买橘子去", t_买橘子去), ("站这别动", t_站这别动),
    ("战略中枢", t_战略中枢), ("延迟满足", t_延迟满足), ("全副武装", t_全副武装), ("三人游", t_三人游),
    ("扩编高峰", t_扩编高峰), ("补给线：整备", _supply("整备")), ("补给线：牺牲", _supply("牺牲")),
    ("中流砥柱", t_中流砥柱), ("补给线：登场", _supply("登场")), ("混沌次元", t_混沌次元), ("混沌裂隙", t_混沌裂隙),
    ("小咖秀", t_小咖秀), ("永不独行", t_永不独行), ("双人同行·2阶", _duo(2)), ("双人同行·3阶", _duo(3)),
    ("搏命", t_搏命), ("紧急救援", t_紧急救援), ("危险数字", t_危险数字), ("大明星", t_大明星),
    ("王牌部队", t_王牌部队), ("有舍有得", t_有舍有得), ("资深猎头", t_资深猎头), ("无限进步", t_无限进步),
    ("心之引力", t_心之引力), ("魅力无限", t_魅力无限), ("万镜", t_万镜), ("潮汐的交响", t_潮汐的交响),
    ("扫拂", t_扫拂), ("吟揉", t_吟揉), ("绞弦", t_绞弦), ("如梦似幻", t_如梦似幻), ("红笺锦鲤符", t_红笺锦鲤符),
    ("狼牙部落的图腾雕像", t_狼牙部落的图腾雕像), ("沙海甘泉", t_沙海甘泉), ("人鱼之泪", t_人鱼之泪),
    ("神树彩陶", t_神树彩陶), ("冰曜宝石镜", t_冰曜宝石镜), ("马可的随身信物", t_马可的随身信物),
    ("马可的魅力飞吻", t_马可的魅力飞吻), ("马可的限量亲签", t_马可的限量亲签), ("马可的吸睛自拍", t_马可的吸睛自拍),
    ("即兴创作", t_即兴创作), ("灵感爆发", t_灵感爆发), ("奇才", t_奇才), ("惊才绝艳", t_惊才绝艳),
]


def main() -> int:
    batch = json.loads((ROOT / "data/engine/sim_batches.json").read_text(encoding="utf-8"))["batches"][6]
    covered = {n for n, _ in TESTS} | set(COMBAT_ONLY) | set(NOT_TESTABLE)
    missing = [n for n in batch if n not in covered]
    ok = fail = 0
    for name, fn in TESTS:
        try:
            if name not in card_db():
                raise AssertionError("卡库里没有这张卡")
            need(name)
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
