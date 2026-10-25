"""第 2 批卡牌的独立测试：**只按 data/cards.json 卡面原文写**，没有读 agent/sim/cards/batch_02.py。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_batch_02.py
逐条打印 OK / FAIL（含原因），末尾打印通过数 / 失败数，有失败退出码 1。
没注册的卡记 FAIL "未实现"，不抛异常。

只有战斗效果、不影响养成的卡不写测试（列在 COMBAT_ONLY，运行时打印 SKIP）。
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
from agent.sim.engine import Engine, impl, card_db, quality, category   # noqa: E402
import agent.sim.engine as _E                                                     # noqa: E402
_E.EQUIP_DROP_ROUNDS = ()      # 固定回合投放装备（开局 / 第 4/8/12/16 回合）是引擎规则，卡牌测试里关掉，免得混进手牌 / 装备数的断言
from agent.sim.state import State, Unit                  # noqa: E402

# ---------------------------------------------------------------- 只有战斗效果的卡（不测）
COMBAT_ONLY = [
    "名刀·司命", "纯净苍穹", "铸梦·逐风", "无尽巨剑", "制裁之剑", "贤者之书", "炽热支配者", "生命魔典",
    "御灵之纱", "冥王", "红魔", "魔律之刃", "不祥征兆", "极寒风暴", "近卫荣耀", "反伤重甲", "霸者重铠",
    "不死鸟之眼", "魔女斗篷", "红莲炎甲", "痛苦面具", "法源之杯", "破茧之衣", "聚灵水晶", "凝渊", "影牙",
    "破晓", "日渊", "虚无法杖", "寒霜侵袭", "自然法杖", "金色圣剑", "断念", "鼓舞之戒", "梦魇之牙",
    "三分旗帜", "怒神项链", "魂灭灵冠", "灵泉项链", "魂祭绝刃", "破阵战芒", "法潮灵杖", "末世", "光辉圣铠",
    "护命灵珠", "破妄之刃", "怒莲之刺", "噬神之书", "征服之剑", "盛华", "静谧之铃", "分身镜", "驱雷",
    "青莲剑谱", "银莲庇佑", "机器头", "磐石之心",
]
# 影响养成、但现有接口搭不出局面的卡（不测，原因见最终报告）
NOT_TESTABLE: dict = {}

RECIPES = json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8"))["recipes"]


class Unimpl(Exception):
    pass


# ---------------------------------------------------------------- 搭局面
def mk(seed=0, **kw) -> Engine:
    st = State(**kw)
    return Engine(st, random.Random(seed))


def add(eng: Engine, name: str, cell_idx: int = 0, level: float = 1.0, temp: float = 0.0,
        kind: str = "hero", equips=None) -> Unit:
    """直接放一个单位（不触发登场），最小局面用。"""
    u = Unit(name=name, level=level, temp=temp, cell=B.CELLS[cell_idx], kind=kind,
             pop=(kind == "hero"), uid=eng.s.next_uid)
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


def use(eng: Engine, name: str, target=None):
    need(name)
    return eng.use_card(name, target)


def recorder(eng: Engine, pick=-1):
    """装一个推演打法：记下每次 choose 的选项，选第 pick 个。"""
    calls = []

    def ch(e, kind, options, **ctx):
        calls.append((kind, list(options)))
        return options[pick]
    eng.chooser = ch
    return calls


def equip(eng: Engine, name: str, u: Unit):
    """穿装备：引擎统一接口 eng.equip(u, 名字)（3 件上限 / 唯一 / 穿戴限制 / 基础装备四选一铸造 / on_equip 都在里面）。"""
    need(name)
    assert eng.equip(u, name), f"eng.equip({u.name}, {name}) 失败"


def hero(name):
    assert category(name) == "英雄牌", f"{name} 不是英雄牌（测试数据有误）"
    return name


# 测试里用到的"白板"英雄：pilot 里注册为纯战斗（无任何养成钩子）
PLAIN_A, PLAIN_B = hero("沈梦溪"), hero("百里守约")


# ================================================================ 效果牌
def t_绝处逢生():
    """随机获得10张效果牌（棋手生命值变为1 —— State 无生命值字段，不测）。"""
    for seed in range(5):
        eng = mk(seed)
        use(eng, "绝处逢生")
        h = eng.s.hand
        assert len(h) == 10, f"seed{seed} 手牌应 +10，实际 {len(h)}: {h}"
        bad = [x for x in h if category(x) != "效果牌"]
        assert not bad, f"seed{seed} 获得了非效果牌 {bad}"


def t_化铁为金():
    """指定1名1-3阶英雄随机变为不在场的5阶英雄，觉醒并保留其当前等级（含临时等级）。"""
    x5 = hero("东皇太一")
    assert quality(x5) == 5
    seen = set()
    for seed in range(20):
        for src in ("苏烈", "马可波罗", "安琪拉"):          # 1 / 2 / 3 阶
            eng = mk(seed)
            t = add(eng, src, 3, level=10, temp=3)
            add(eng, x5, 10, level=5)
            cell = t.cell
            ok = use(eng, "化铁为金", t)
            assert ok is not False, f"{src}({quality(src)}阶) 是合法目标，却返回 False"
            nu = next((u for u in eng.s.units if u.cell == cell), None)
            assert nu is not None, "原格子上没有单位了"
            assert category(nu.name) == "英雄牌" and quality(nu.name) == 5, f"变成了 {nu.name}（{quality(nu.name)}阶）"
            assert nu.name != x5, f"变成了在场的 {x5}（卡面：不在场的5阶英雄）"
            assert nu.awakened, f"{nu.name} 没觉醒 awaken={nu.awaken}"
            assert nu.lv() >= 13 - 1e-9, f"等级没保留：原 10+临时3，现 level={nu.level} temp={nu.temp}"
            assert len([u for u in eng.s.units if u.kind == "hero"]) == 2, "场上英雄数变了"
            seen.add(nu.name)
    assert len(seen) >= 2, f"20 个种子只变出 {seen}，不像随机"
    # 4 阶不合法
    eng = mk(0)
    t = add(eng, hero("李信"), 3, level=10)
    assert quality("李信") == 4
    ok = use(eng, "化铁为金", t)
    assert ok is False and t.name == "李信" and t.level == 10, "4阶英雄不应能被化铁为金"


def t_超级复印机():
    """随机获得3张相同的场上英雄牌。"""
    names = set()
    for seed in range(20):
        eng = mk(seed)
        add(eng, PLAIN_A, 0)
        add(eng, PLAIN_B, 3)
        add(eng, "因缘图腾", 6, kind="summon")
        use(eng, "超级复印机")
        h = eng.s.hand
        assert len(h) == 3 and len(set(h)) == 1, f"seed{seed} 应得 3 张相同的牌，实际 {h}"
        assert h[0] in (PLAIN_A, PLAIN_B), f"得到的 {h[0]} 不是场上英雄（召唤物不算）"
        names.add(h[0])
    assert len(names) == 2, f"20 个种子只复印过 {names}，不像随机"
    eng = mk(0)                                            # 场上没有英雄 → 什么也得不到
    add(eng, "因缘图腾", 6, kind="summon")
    use(eng, "超级复印机")
    assert not eng.s.hand, f"场上没有英雄却得到了 {eng.s.hand}"


def t_一线生机():
    """获得2张名刀·司命。（棋手生命值首次降至0不淘汰 —— State 无生命值，不测）"""
    eng = mk(0)
    use(eng, "一线生机")
    assert eng.s.hand.count("名刀·司命") == 2, f"手牌 {eng.s.hand}"
    assert len(eng.s.hand) == 2, f"多拿了牌 {eng.s.hand}"


def t_能量超新星():
    """随机获得10-20能量。"""
    gains = []
    for seed in range(60):
        eng = mk(seed, energy=5)
        use(eng, "能量超新星")
        d = eng.s.energy - 5
        assert 10 <= d <= 20 and d == int(d), f"seed{seed} 获得 {d} 能量（应 10~20 整数）"
        gains.append(d)
    assert min(gains) < max(gains), f"60 次都是 {gains[0]}，不像随机"


def t_狂飙三倍速():
    """获得3张开局狂飙。"""
    eng = mk(0)
    use(eng, "狂飙三倍速")
    assert eng.s.hand == ["开局狂飙"] * 3, f"手牌 {eng.s.hand}"


def t_四人同行():
    """卡牌5选1，并将选中的卡牌一分为四（= 得到 4 张选中的牌）。"""
    for seed in range(5):
        eng = mk(seed)
        calls = recorder(eng, pick=-1)
        use(eng, "四人同行")
        h = eng.s.hand
        assert len(h) == 4 and len(set(h)) == 1, f"seed{seed} 应得 4 张相同的牌，实际 {h}"
        if calls:
            kind, opts = calls[0]
            assert len(opts) == 5, f"应是 5 选 1，实际选项 {opts}"
            assert len(set(opts)) == 5, f"5 个选项有重复 {opts}"
            assert h[0] == opts[-1], f"拿到的 {h[0]} 不是打法选中的 {opts[-1]}"


def t_新生力量():
    """可上阵人数+1。"""
    eng = mk(0, lord_level=3)
    cap0 = eng.pop_cap()
    use(eng, "新生力量")
    assert eng.s.pop_bonus == 1 and eng.pop_cap() == cap0 + 1, f"pop_bonus={eng.s.pop_bonus} cap {cap0}->{eng.pop_cap()}"
    use(eng, "新生力量")
    assert eng.pop_cap() == cap0 + 2, "两张应 +2"


def t_沧海拾遗():
    """随机获得1张其他6阶效果牌。"""
    got = set()
    for seed in range(30):
        eng = mk(seed)
        use(eng, "沧海拾遗")
        h = eng.s.hand
        assert len(h) == 1, f"seed{seed} 应得 1 张，实际 {h}"
        x = h[0]
        assert category(x) == "效果牌" and quality(x) == 6, f"{x} 不是 6 阶效果牌（{category(x)} {quality(x)}）"
        assert x != "沧海拾遗", "卡面：其他 6 阶效果牌，不能是自己"
        got.add(x)
    assert len(got) >= 3, f"30 次只出过 {got}，不像随机"


# ================================================================ 基础装备：穿上铸造成普通装备
def _forge(base):
    def t():
        """穿戴给英雄，可以铸造成1件普通装备（四选一，成品见 data/equip_recipes.json）。"""
        prods = set(RECIPES[base])
        for seed in range(4):
            eng = mk(seed)
            u = add(eng, PLAIN_A, 3)
            calls = recorder(eng, pick=seed % 4)
            equip(eng, base, u)
            assert base not in u.equips, f"基础装备 {base} 还挂在身上：{u.equips}"
            assert len(u.equips) == 1 and u.equips[0] in prods, f"铸造结果 {u.equips} 不是 {sorted(prods)} 之一"
            if calls:
                opts = calls[-1][1]
                assert set(opts) == prods, f"四选一选项 {opts} ≠ {sorted(prods)}"
                assert u.equips[0] == opts[seed % 4], f"铸出 {u.equips[0]}，打法选的是 {opts[seed % 4]}"
    t.__doc__ = f"{base}：" + t.__doc__
    return t


# ================================================================ 有养成效果的装备
def t_不屈意志():
    """战斗开始及之后每3秒获得 场上友军穿戴不屈意志数量×2 的临时等级；阵亡时，自身等级30%平均分配给穿戴不屈意志的友军。"""
    need("不屈意志")
    # 战斗开始：两人都穿 → 至少拿到一跳临时等级，且是 2 的倍数
    eng = mk(0)
    a = add(eng, PLAIN_A, 3, level=10, equips=["不屈意志"])
    b = add(eng, PLAIN_B, 10, level=10, equips=["不屈意志"])
    eng.fire(a, "battle_start")
    assert a.temp > 0, f"战斗开始没拿到临时等级 temp={a.temp}"
    assert abs(a.temp / 2 - round(a.temp / 2)) < 1e-9, f"临时等级应是 2 的倍数，实际 {a.temp}"
    assert a.level == 10, f"永久等级不该变 {a.level}"
    # 阵亡：20 级 → 6 级平分给另外两名穿戴者，不穿的不给
    eng = mk(0)
    a = add(eng, PLAIN_A, 3, level=20, equips=["不屈意志"])
    b = add(eng, PLAIN_B, 10, level=5, equips=["不屈意志"])
    c = add(eng, hero("苏烈"), 11, level=5, equips=["不屈意志"])
    d = add(eng, hero("安琪拉"), 12, level=5)
    eng.fire(a, "death")
    gb, gc = b.level - 5, c.level - 5
    assert abs(gb + gc - 6) < 1e-6 and abs(gb - gc) < 1e-6, f"应各 +3（共 20×30%=6），实际 {gb}/{gc}"
    assert d.level == 5, f"没穿不屈意志的也拿到了等级 {d.level}"
    # 只有自己穿：阵亡后没人可分
    eng = mk(0)
    a = add(eng, PLAIN_A, 3, level=20, equips=["不屈意志"])
    d = add(eng, PLAIN_B, 10, level=5)
    eng.fire(a, "death")
    assert d.level == 5, f"没有其他穿戴者时不该分等级，实际 {d.level}"


def t_捕能手套():
    """每击败1名英雄获得1能量；单回合击败不少于5名英雄时，获得能量转换。"""
    need("捕能手套")
    eng = mk(0, energy=3, lord_level=3)
    u = add(eng, PLAIN_A, 3, equips=["捕能手套"])
    eng.fire(u, "kill", n=2)
    assert eng.s.energy == 5, f"击败 2 名应 +2 能量，3→{eng.s.energy}"
    assert "能量转换" not in eng.s.hand, "不到 5 名不该获得能量转换"
    eng = mk(0, energy=3, lord_level=3)
    u = add(eng, PLAIN_A, 3, equips=["捕能手套"])
    eng.fire(u, "kill", n=4)
    assert eng.s.energy == 7 and not eng.s.hand, f"4 名：能量 {eng.s.energy}（应 7），手牌 {eng.s.hand}"
    eng = mk(0, energy=3, lord_level=3)
    u = add(eng, PLAIN_A, 3, equips=["捕能手套"])
    eng.fire(u, "kill", n=5)
    h = eng.s.hand
    if "能量转换" in h:                                   # 读法一：得到"能量转换"这张牌
        assert eng.s.energy == 8, f"5 名应 +5 能量，3→{eng.s.energy}"
        assert h.count("能量转换") == 1, f"能量转换应只给 1 张 {h}"
    else:                                                  # 读法二：直接结算转换（-3 能量换 3 阶锦囊）
        assert eng.s.energy == 5 and h == ["3阶效果锦囊"], f"5 名：既没拿到能量转换也没正确结算转换，能量 {eng.s.energy} 手牌 {h}"


def t_灵泉法杖():
    """获得闪现（养成上：闪现关键词让孙悟空等的"拥有闪现的英雄"效果生效）。"""
    need("灵泉法杖")
    eng = mk(0)
    u = add(eng, PLAIN_A, 3, level=5, equips=["灵泉法杖"])
    assert not eng.has_kw(u, "闪现") or "闪现" in u.keywords, "前提：沈梦溪本身不该有闪现"
    eng.round_start()                                      # 关键词每回合重算
    assert eng.has_kw(u, "闪现"), "穿灵泉法杖的人回合开始后没有闪现"
    # 联动：孙悟空 整备：拥有闪现效果的英雄等级+2
    eng = mk(0)
    u = add(eng, PLAIN_A, 3, level=5, equips=["灵泉法杖"])
    add(eng, hero("孙悟空"), 27, level=5)                  # 远离，不触发夺取（直接放置也不登场）
    v = add(eng, PLAIN_B, 13, level=5)
    eng.round_start()
    assert u.level == 7, f"孙悟空整备应给闪现的穿戴者 +2，实际 5→{u.level}"
    assert v.level == 5, f"没闪现的人不该涨 {v.level}"


# ================================================================ 天赋
def t_兵行诡招():
    """战斗开始时，己方每名英雄+2临时等级；战斗胜利时，获得1棋手生命值，临时等级额外+1。（棋手生命值不测：State 无生命值）
    "临时等级额外+1"有两种读法：
      A 本场胜利当场再 +1 临时等级（本场共 3，负 2）；
      B 每胜一场，以后战斗开始给的临时等级永久 +1（第 k 场给 2 + 之前胜场数）。
    临时等级战斗结束就清零，A 读法下这 +1 没有任何作用 → 统一按 B（实现 docstring 标了待核对）。"""
    need("兵行诡招")
    seen = set()
    for seed in range(6):
        eng = mk(seed)
        eng.s.talents.append("兵行诡招")
        hs = [add(eng, PLAIN_A, 3), add(eng, PLAIN_B, 10)]
        add(eng, "因缘图腾", 27, kind="summon")             # 召唤物不是英雄
        got_all, wins = [], []
        for k in range(6):
            eng.s.log.clear()
            win = eng.battle()
            wins.append(win)
            seen.add(win)
            got = [sum(kw.get("n", 0) for (_, ev, kw) in eng.s.log if ev == "临时等级" and kw.get("unit") == h.name)
                   for h in hs]
            assert got[0] == got[1], f"seed{seed} 第{k+1}场 两名英雄临时等级不同 {got}"
            tot = sum(kw.get("n", 0) for (_, ev, kw) in eng.s.log if ev == "临时等级" and kw.get("unit") == "因缘图腾")
            assert tot == 0, f"召唤物也拿了临时等级 {tot}"
            got_all.append(got[0])
        exp_a = [3 if w else 2 for w in wins]
        exp_b = [2 + sum(wins[:i]) for i in range(len(wins))]
        assert got_all == exp_b, f"seed{seed} 胜负 {['胜' if w else '负' for w in wins]}，每场临时等级 {got_all}；应 {exp_b}（读法A {exp_a}）"
    assert seen == {True, False}, f"种子没覆盖胜 / 负两种情况：{seen}（测试本身需调种子）"


def t_棋手生命值():
    """兵行诡招：生命 -10，每胜 +1，回复不超过 60；绝处逢生：生命变为 1；一线生机：第一次降到 0 保留 1，第二次不保。
    棋手生命值 = State.hp，改动走 eng.add_hp（引擎统一接口）。"""
    need("兵行诡招"); need("绝处逢生"); need("一线生机")

    def lord_hp(e):
        return e.s.hp

    def add_lord_hp(e, delta):
        return e.add_hp(delta)
    eng = mk(1)
    assert lord_hp(eng) == 60, f"初始生命应 60，实际 {lord_hp(eng)}"
    impl("兵行诡招").on_use(eng, None)
    assert lord_hp(eng) == 50, f"兵行诡招应 -10 → 50，实际 {lord_hp(eng)}"
    eng.s.talents.append("兵行诡招")
    add(eng, PLAIN_A, 3)
    for _ in range(30):
        eng.battle()
    assert lord_hp(eng) <= 60, f"回复超过上限 60：{lord_hp(eng)}"
    eng = mk(0)
    impl("绝处逢生").on_use(eng, None)
    assert lord_hp(eng) == 1, f"绝处逢生后生命应 1，实际 {lord_hp(eng)}"
    eng = mk(0)
    impl("一线生机").on_use(eng, None)
    add_lord_hp(eng, -100)
    assert lord_hp(eng) == 1, f"一线生机第一次降到 0 应保留 1，实际 {lord_hp(eng)}"
    add_lord_hp(eng, -5)
    assert lord_hp(eng) == 0, f"一线生机只保一次，第二次应到 0，实际 {lord_hp(eng)}"


def t_阿科米亚之子():
    """唯一。参与阿科米亚核心的等级分配。（非日落海英雄穿上后，开战核心分配也分给他）"""
    need("阿科米亚之子")
    eng = mk(0)
    sun = add(eng, hero("艾琳"), 3, level=10)                 # 日落海
    other = add(eng, PLAIN_A, 4, level=10)                     # 河洛
    eng.s.core = 60
    eng.core_distribute()
    assert sun.temp == 30 and other.temp == 0, f"没穿时只有日落海分到（min(60/1, 50%)=30）：{sun.temp} / {other.temp}"
    sun.temp = other.temp = 0
    equip(eng, "阿科米亚之子", other)
    eng.core_distribute()
    assert sun.temp == 30 and other.temp == 30, f"穿上后两人平分（各 30）：{sun.temp} / {other.temp}"
    sun.temp = other.temp = 0
    eng.round_start()                                          # 关键词每回合重算，仍然参与
    eng.core_distribute()
    assert other.temp == 30, f"回合开始后仍应参与分配：{other.temp}"


def t_穿戴限制():
    """穿戴限制（读卡面）：日渊限远程、青莲剑谱限近战、三分旗帜仅三分、破阵战芒仅 1-3 阶；最多 3 件；唯一不能重复；盛华无法拆卸。"""
    ranged = next(n for n, c in card_db().items() if c.get("category") == "英雄牌"
                  and int(((c.get("properties") or {}).get("attackDistance") or 0)) > 1)
    melee = next(n for n, c in card_db().items() if c.get("category") == "英雄牌"
                 and 0 < int(((c.get("properties") or {}).get("attackDistance") or 0)) <= 1)
    eng = mk(0)
    r = add(eng, ranged, 3)
    m = add(eng, melee, 4)
    assert eng.equip(r, "日渊") and not eng.equip(m, "日渊"), "日渊只有远程能穿"
    assert eng.equip(m, "青莲剑谱") and not eng.equip(r, "青莲剑谱"), "青莲剑谱只有近战能穿"
    sf = add(eng, hero("马超"), 5)                               # 三分 3 阶
    q5 = add(eng, hero("东皇太一"), 6)                            # 5 阶
    assert eng.equip(sf, "三分旗帜") and not eng.equip(q5, "三分旗帜"), "三分旗帜仅三分英雄"
    assert eng.equip(sf, "破阵战芒") and not eng.equip(q5, "破阵战芒"), "破阵战芒仅 1-3 阶"
    assert not eng.equip(r, "日渊"), "唯一装备同一人不能穿两件"
    eng.equip(r, "噬神之书")
    eng.equip(r, "征服之剑")
    assert len(r.equips) == 3 and not eng.equip(r, "盛华"), f"最多 3 件：{r.equips}"
    eng.equip(q5, "盛华")
    assert "盛华" in q5.bound, "盛华无法拆卸，应进 bound"
    eng.remove(q5)
    assert "盛华" not in eng.s.hand, "无法拆卸的装备离场不该回手"


# ================================================================ 运行
TESTS = [
    ("绝处逢生", t_绝处逢生), ("化铁为金", t_化铁为金), ("超级复印机", t_超级复印机), ("一线生机", t_一线生机),
    ("能量超新星", t_能量超新星), ("狂飙三倍速", t_狂飙三倍速), ("四人同行", t_四人同行), ("新生力量", t_新生力量),
    ("沧海拾遗", t_沧海拾遗),
] + [(b, _forge(b)) for b in ("风暴巨剑", "圣者法典", "雷鸣刃", "守护者之铠", "力量腰带", "破碎圣杯", "速击之枪", "大棒")] + [
    ("不屈意志", t_不屈意志), ("捕能手套", t_捕能手套), ("灵泉法杖", t_灵泉法杖), ("兵行诡招", t_兵行诡招), ("兵行诡招", t_棋手生命值),
    ("阿科米亚之子", t_阿科米亚之子), ("日渊", t_穿戴限制),
]


def main() -> int:
    batch = json.loads((ROOT / "data/engine/sim_batches.json").read_text(encoding="utf-8"))["batches"][2]
    covered = {n for n, _ in TESTS} | set(COMBAT_ONLY) | set(NOT_TESTABLE)
    missing = [n for n in batch if n not in covered]
    ok = fail = 0
    for name, fn in TESTS:
        try:
            if name not in card_db():
                raise AssertionError("卡库里没有这张卡")
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
    print(f"\n通过 {ok} / 失败 {fail}（共 {ok + fail} 条测试，跳过 {len(COMBAT_ONLY) + len(NOT_TESTABLE)} 张）")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
