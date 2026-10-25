"""第 5 批卡牌的独立测试：**只按 data/cards.json 卡面原文写**（天赋送的牌看 given_skill），没有读 agent/sim/cards/batch_05.py。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_batch_05.py
逐条打印 OK / FAIL（含原因），末尾打印通过数 / 失败数，有失败退出码 1。
没注册的卡记 FAIL "未实现"，不抛异常。

约定（照 agent/sim/play.py 的 _take）：选天赋 = 放进 s.talents，再 use_card(名字, None)（有 on_use 时）。
天赋送出的同名牌（点兵点将 / 望梅止渴）按 play.py 出牌方式打出：先试英雄目标。

几个通用手法：
- 白板英雄：按派系在卡库里挑"实现里没有任何时机钩子"的英雄（运行时判断，不读批次文件），避免别的卡效果掺进来。
- 对照组：真英雄（貂蝉 / 孙悟空 / 程咬金…）自身也有效果时，同一局面"有天赋 / 无天赋"各跑一次，比差值。
- 战斗探针：往 REGISTRY 注册一个测试用天赋，在开团 / 战斗开始反应和战斗结束时给全场拍快照（临时等级、装备、复生）。
- 阿科米亚核心：State 没有核心字段，用「盗火」反复偷（在克隆局面上）读出核心等级 —— 核心相关的测试互相依赖，失败信息里写明。
"""
from __future__ import annotations

import json
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.engine import board as B                                       # noqa: E402
from agent.sim.engine import (Engine, REGISTRY, impl, card_db, quality,    # noqa: E402
                              category, faction, has_keyword, card_text)
import agent.sim.engine as _E                                                     # noqa: E402
_E.EQUIP_DROP_ROUNDS = ()      # 固定回合投放装备（开局 / 第 4/8/12/16 回合）是引擎规则，卡牌测试里关掉，免得混进手牌 / 装备数的断言
from agent.sim.state import State, Unit                                   # noqa: E402

# ---------------------------------------------------------------- 只有战斗效果的卡（不测）
COMBAT_ONLY = {
    "穷兵黩武": "技能释放时召唤玄雍士兵（战斗中出现、战斗结束消失），引擎不模拟技能释放",
    "罡气护体": "护盾",
    "暴君归来": "攻速 / 暴击 / 无敌",
    "水晶穹顶": "护盾",
    "神明庇佑": "生命 / 攻速",
    "流云之刺": "攻速 / 伤害 / 护甲",
    "翼苍穹盾": "双抗",
    "掠魂爪": "吸血 / 伤害",
    "永冻霜羽": "法攻 / 冻结",
    "苍穹王权": "免死 / 伤害",
    "唤鹰骨笛": "战斗中召唤半等级云中君冲锋",
    "升格": "战斗中召唤物变身召唤者（战斗内）",
    "猩红神兽": "战斗中给召唤物加力",
    "蔚蓝神兽": "永久加生命值（不是等级）",
    "舞力全开": "共享法力 / 融合技",
    "双生突袭": "分身放技能",
    "双重分配": "核心分配时附带双攻 / 攻速 / 法力",
    "狂野帷幕": "隐形",
    "控魂": "复活敌方灵魂作战",
    "双生相": "生命 / 双抗 / 暴击",
    "根源之力": "获得一个法术伤害能力（卡库里没有单独的牌）",
    "强敌克星": "获得一个真实伤害 / 眩晕能力（卡库里没有单独的牌）",
    "守护灵": "战斗中仅剩 1 人时召唤满装 5 阶英雄出战",
}
# 影响养成、但现有接口搭不出局面的卡
NOT_TESTABLE: dict = {}

TACTICS = ["援护战术", "招募战术", "演练战术", "推演战术", "特训战术"]
HIGH_TACTICS = ["高阶" + x for x in TACTICS]
SHENBING = {"流云之刺", "翼苍穹盾", "掠魂爪", "永冻霜羽", "苍穹王权", "唤鹰骨笛"}   # 本批里云中君 / 鹰主题的 6 件，当作"神兵装备"
FAKE_EQ = ["测试装备甲", "测试装备乙", "测试装备丙"]                         # 卡库里没有的名字：只占件数，不带效果


class Unimpl(Exception):
    pass


# ---------------------------------------------------------------- 搭局面
def mk(seed=0, **kw) -> Engine:
    return Engine(State(**kw), random.Random(seed))


def add(eng: Engine, name: str, idx: int = 0, level: float = 1.0, kind: str = "hero",
        equips=None, awaken: int = 0, temp: float = 0.0) -> Unit:
    """直接放一个单位（不触发登场）。"""
    u = Unit(name=name, level=level, temp=temp, cell=B.CELLS[idx], kind=kind,
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


def pick(eng: Engine, name: str):
    """选天赋（同 play.py _take）：放进天赋栏，有 on_use 就以 target=None 打出一次。"""
    c = need(name)
    eng.s.talents.append(name)
    if hasattr(c, "on_use"):
        eng.use_card(name, None)


def play(eng: Engine, name: str, target=None):
    """从手牌打出一张（手里有就先拿掉，和真实出牌一样）。"""
    need(name)
    if name in eng.s.hand:
        eng.s.hand.remove(name)
    return eng.use_card(name, target)


def gained(before: list, after: list) -> list:
    return list((Counter(after) - Counter(before)).elements())


def _hooks(c) -> list:
    return [a for a in dir(c) if a.startswith(("on_", "react_")) or a in ("coin_bonus", "grant_keywords")]


def plain(fac: str, n: int = 1, avoid_kw=("牺牲", "登场", "整备", "夺取")) -> list:
    """该派系里实现没有任何钩子、卡面也没有指定关键词的英雄（按卡库顺序取前 n 个，不够就重复）。"""
    fac = {"大河": "大河流域", "三分": "三分之地"}.get(fac, fac)
    out = []
    for name, c in card_db().items():
        if c.get("category") != "英雄牌" or faction(name) != fac:
            continue
        i = impl(name)
        if i is not None and _hooks(i):
            continue
        if any(has_keyword(name, k) for k in avoid_kw):
            continue
        out.append(name)
    if not out:
        raise AssertionError(f"测试环境：【{fac}】里找不到白板英雄")
    return [out[k % len(out)] for k in range(n)]


def hero_cards(names, fac=None, tiers=None):
    for x in names:
        assert category(x) == "英雄牌", f"{x} 不是英雄牌（{category(x)}）"
        if fac:
            assert faction(x) == fac, f"{x} 不是【{fac}】（{faction(x)}）"
        if tiers:
            assert quality(x) in tiers, f"{x} 品阶 {quality(x)} 不在 {sorted(tiers)}"


# ---------------------------------------------------------------- 战斗探针
PROBE = "__测试探针__"


class _Probe:
    def __init__(self):
        self.snaps = []

    def _snap(self, eng, stage):
        self.snaps.append((stage, [(u.uid, u.name, u.level, u.temp, list(u.equips),
                                    u.counters.get("revive", 0), u.kind) for u in eng.s.units]))

    def react_open(self, eng, holder, **kw):
        self._snap(eng, "open")

    def react_battle_start(self, eng, holder, **kw):
        self._snap(eng, "bs")

    def on_battle_end(self, eng, holder, win=False):
        self._snap(eng, "end")

    def rows(self, uid):
        return [(st, r) for st, rows in self.snaps for r in rows if r[0] == uid]

    def at_end(self, uid):
        r = [r for st, r in self.rows(uid) if st == "end"]
        return r[-1] if r else None

    def names(self):
        return {r[1] for _, rows in self.snaps for r in rows}


def probe(eng: Engine) -> _Probe:
    impl("古币")                         # 确保卡牌已注册
    p = _Probe()
    REGISTRY[PROBE] = p
    eng.s.talents.append(PROBE)
    return p


def logged_deploys(eng, sub: str) -> list:
    return [kw.get("unit") for (_, ev, kw) in eng.s.log if ev == "上场" and sub in str(kw.get("unit", ""))]


# ---------------------------------------------------------------- 阿科米亚核心（用盗火读）
def core_level(eng: Engine) -> float:
    """在克隆局面上反复对一个白板英雄打「盗火」（每次至多偷 20），累计偷到的就是核心等级。"""
    need("盗火")
    e2 = Engine(eng.s.clone(), random.Random(0))
    idx = next(i for i, c in enumerate(B.CELLS) if c not in {u.cell for u in e2.s.units})
    h = add(e2, plain("逐鹿")[0], idx, level=1)
    tot = 0.0
    for k in range(40):
        lv = h.level
        e2.use_card("盗火", h)
        g = h.level - lv
        assert g <= 20 + 1e-9, f"盗火一次偷了 {g} 级（卡面至多 20）"
        if g <= 0:
            return tot
        tot += g
    raise AssertionError("盗火连偷 40 次仍有收获 —— 盗火没有从核心扣等级？")


def load_core(eng: Engine, n: int):
    """用「能量汲取」（出售场上英雄 → 核心获得其等级）把核心充到 n 级。"""
    need("能量汲取")
    if "能量汲取" not in eng.s.talents:
        pick(eng, "能量汲取")
    idx = next(i for i, c in enumerate(B.CELLS) if c not in {u.cell for u in eng.s.units})
    u = add(eng, plain("三分")[0], idx, level=n)
    eng.sell(u)


# ================================================================ 获得牌（天赋拿到时）
GIVES = {
    "三分领袖": ["三分旗帜"], "千金换酒": ["李白"], "身外身": ["孙悟空"], "兄妹": ["铠", "露娜"],
    "兽觉": ["裴擒虎", "云中君"], "巨像": ["米莱狄", "蒙恬"], "起舞": ["貂蝉", "上官婉儿"], "箭雨": ["虞姬", "公孙离"],
    "毒奶": ["蔡文姬", "扁鹊"], "逐星": ["李白", "曜"], "宝盒": ["露娜", "孙悟空"], "旋刃": ["李元芳", "公孙离"],
    "神鹰锻匠": ["云中君"], "核心裂痕": ["盗火"], "点兵点将": ["点兵点将"], "望梅止渴": ["望梅止渴"],
    "升值": ["古币", "古币"], "天兵": None,
}


def _gives(name, exp):
    def t():
        eng = mk(0)
        h0 = list(eng.s.hand)
        pick(eng, name)
        g = gained(h0, eng.s.hand)
        assert sorted(g) == sorted(exp), f"选天赋后手牌应获得 {exp}，实际 {g}"
    return t


def t_天兵():
    """获得1张5阶英雄牌。回合开始时，再获得1张。"""
    for seed in range(4):
        eng = mk(seed)
        h0 = list(eng.s.hand)
        pick(eng, "天兵")
        g = gained(h0, eng.s.hand)
        assert len(g) == 1, f"seed{seed} 选天赋应得 1 张，实际 {g}"
        hero_cards(g, tiers={5})
        h1 = list(eng.s.hand)
        eng.round_start()
        g = gained(h1, eng.s.hand)
        assert len(g) == 1, f"seed{seed} 回合开始应再得 1 张，实际 {g}"
        hero_cards(g, tiers={5})


def _reinforce(name, fac, energy, tiers):
    def t():
        for seed in range(5):
            eng = mk(seed)
            e0, h0 = eng.s.energy, list(eng.s.hand)
            pick(eng, name)
            g = gained(h0, eng.s.hand)
            assert eng.s.energy - e0 == energy, f"seed{seed} 能量应 +{energy}，实际 +{eng.s.energy - e0}"
            assert len(g) == 3, f"seed{seed} 应得 3 张英雄牌，实际 {g}"
            hero_cards(g, fac=fac, tiers=tiers)
    return t


# ================================================================ 回合开始 / 持续获得
def t_令行如流():
    """获得1张战术牌。回合开始时，再获得1张（棋手等级达到5级后，必定是高阶战术）。"""
    allt = set(TACTICS) | set(HIGH_TACTICS)
    for seed in range(4):
        eng = mk(seed, lord_level=3)
        h0 = list(eng.s.hand)
        pick(eng, "令行如流")
        g = gained(h0, eng.s.hand)
        assert len(g) == 1 and g[0] in allt, f"seed{seed} 选天赋应得 1 张战术牌，实际 {g}"
        h1 = list(eng.s.hand)
        eng.round_start()
        g = gained(h1, eng.s.hand)
        assert len(g) == 1 and g[0] in allt, f"seed{seed} 回合开始应再得 1 张战术牌，实际 {g}"
    for seed in range(8):
        eng = mk(seed, lord_level=5)
        pick(eng, "令行如流")
        h1 = list(eng.s.hand)
        eng.round_start()
        g = gained(h1, eng.s.hand)
        assert len(g) == 1 and g[0] in HIGH_TACTICS, f"seed{seed} 棋手 5 级回合开始应得高阶战术，实际 {g}"


def t_演武():
    """棋手升到5级后，你获得的战术牌都会变为高阶战术。（用西施"整备：获得1张推演战术"来产生战术牌）"""
    need("演武")
    for lv, exp in ((5, "高阶推演战术"), (4, "推演战术")):
        eng = mk(0, lord_level=lv)
        pick(eng, "演武")
        add(eng, "西施", 17)
        h0 = list(eng.s.hand)
        eng.round_start()
        g = gained(h0, eng.s.hand)
        assert g == [exp], f"棋手 {lv} 级时西施整备应得 [{exp}]，实际 {g}"


def t_核心裂痕_回合():
    """获得1张盗火。回合开始时，再获得1张。"""
    eng = mk(0)
    pick(eng, "核心裂痕")
    h1 = list(eng.s.hand)
    eng.round_start()
    g = gained(h1, eng.s.hand)
    assert g == ["盗火"], f"回合开始应再得 1 张盗火，实际 {g}"


def t_春风化雨():
    """回合开始时，获得1张【逐鹿】英雄牌；若上场战斗胜利，再获得1张场上【逐鹿】英雄的同名牌。
    胜利那张不论在战斗结束时给还是下回合开始时给，都算在"战斗 + 下回合开始"这一段里。"""
    zl = plain("逐鹿")[0]
    seen = set()
    for seed in range(10):
        eng = mk(seed)
        pick(eng, "春风化雨")
        h0 = list(eng.s.hand)
        eng.round_start()
        g = gained(h0, eng.s.hand)
        assert len(g) == 1, f"seed{seed} 第一次回合开始应得 1 张，实际 {g}"
        hero_cards(g, fac="逐鹿")
        add(eng, zl, 17, level=5)
        add(eng, plain("大河")[0], 3, level=5)
        h1 = list(eng.s.hand)
        win = eng.battle()
        seen.add(win)
        eng.s.round += 1
        eng.round_start()
        g = gained(h1, eng.s.hand)
        hero_cards(g, fac="逐鹿")
        assert len(g) == (2 if win else 1), f"seed{seed} {'胜' if win else '负'}后应得 {2 if win else 1} 张，实际 {g}"
        if win:
            assert zl in g, f"seed{seed} 胜利应得场上逐鹿英雄 {zl} 的同名牌，实际 {g}"
    assert seen == {True, False}, f"种子没覆盖胜负两种：{seen}"


def t_望梅止渴_回合():
    """回合开始时，场上每有2名【大河】英雄，再获得1张（望梅止渴）。"""
    for n_dh, exp in ((4, 2), (3, 1), (1, 0)):
        eng = mk(0)
        pick(eng, "望梅止渴")
        for k, nm in enumerate(plain("大河", n_dh)):
            add(eng, nm, k)
        add(eng, plain("逐鹿")[0], 20)
        h1 = list(eng.s.hand)
        eng.round_start()
        g = gained(h1, eng.s.hand)
        assert g == ["望梅止渴"] * exp, f"{n_dh} 名大河英雄时回合开始应得 {exp} 张望梅止渴，实际 {g}"


def t_望梅止渴_赠送牌():
    """赠送牌（given_skill）：指定1名英雄临时等级+3，战斗开始时，每穿1件装备额外+1级。若战斗胜利，将永久获得此等级。"""
    seen = set()
    for seed in range(10):
        eng = mk(seed)
        pick(eng, "望梅止渴")
        p = probe(eng)
        u = add(eng, plain("逐鹿")[0], 17, level=10, equips=FAKE_EQ[:2])
        ok = play(eng, "望梅止渴", u)
        assert ok is not False, "对英雄打出望梅止渴被拒"
        assert u.temp == 3, f"打出后临时等级应 +3，实际 {u.temp}"
        win = eng.battle()
        seen.add(win)
        r = p.at_end(u.uid)
        assert r and r[3] == 5, f"seed{seed} 战斗中临时等级应为 3+2件装备=5，实际 {r and r[3]}"
        assert u.level == (15 if win else 10), f"seed{seed} {'胜' if win else '负'}后等级应为 {15 if win else 10}，实际 {u.level}"
    assert seen == {True, False}, f"种子没覆盖胜负两种：{seen}"


# ================================================================ 战斗相关的养成
def t_四重奏():
    """战斗开始时，若你场上有至少3个阵营，全体英雄获得5×棋手等级的临时等级。"""
    for facs, exp in ((("逐鹿", "大河", "三分"), 15), (("逐鹿", "逐鹿", "大河"), 0)):
        eng = mk(0, lord_level=3)
        pick(eng, "四重奏")
        p = probe(eng)
        hs = [add(eng, plain(f)[0], i * 3) for i, f in enumerate(facs)]
        tot = add(eng, "因缘图腾", 27, kind="summon")
        eng.battle()
        for h in hs:
            r = p.at_end(h.uid)
            assert r and r[3] == exp, f"阵营 {facs}：{h.name} 战斗中临时等级应为 {exp}，实际 {r and r[3]}"
        r = p.at_end(tot.uid)
        assert r and r[3] == 0, f"召唤物不是英雄，不该拿临时等级：{r and r[3]}"


def t_意外之财():
    """每合成4次【逐鹿】英雄，获得1张随机1-5阶效果锦囊牌。（引擎本身每合成 3 次也送 1 个锦囊，扣掉它）"""
    def jn(h):
        return [x for x in h if x.endswith("阶效果锦囊")]
    for fac, merges, exp in (("逐鹿", 4, 1), ("逐鹿", 8, 2), ("逐鹿", 3, 0), ("大河", 4, 0)):
        eng = mk(0)
        pick(eng, "意外之财")
        u = add(eng, plain(fac)[0], 17)
        for _ in range(merges):
            eng.merge(u)
        engine_gift = merges // 3
        got = jn(eng.s.hand)
        assert len(got) == engine_gift + exp, f"合成【{fac}】{merges} 次：锦囊应 {engine_gift}(引擎)+{exp}(天赋)，实际 {got}"
        for x in got:
            assert 1 <= int(x[0]) <= 5, f"锦囊品阶超出 1-5：{x}"


def t_复现():
    """每回合首次对【逐鹿】英雄使用的效果牌会返回手牌并附加转瞬。（用特训战术：指定1名英雄等级+4）"""
    need("特训战术")
    eng = mk(0)
    pick(eng, "复现")
    zl = add(eng, plain("逐鹿")[0], 17, level=1)
    dh = add(eng, plain("大河")[0], 3, level=1)

    def cnt():
        return sum(1 for x in eng.s.hand if x.startswith("特训战术"))
    eng.s.hand.append("特训战术")
    play(eng, "特训战术", zl)
    assert zl.level == 5, f"特训战术应 +4：{zl.level}"
    assert cnt() == 1, f"本回合首次对逐鹿用的效果牌应回手，手牌 {eng.s.hand}"
    for x in list(eng.s.hand):
        if x.startswith("特训战术"):
            eng.s.hand.remove(x)
    eng.s.hand.append("特训战术")
    play(eng, "特训战术", zl)
    assert cnt() == 0, f"同回合第二次不该再回手，手牌 {eng.s.hand}"
    # 新回合：先对非逐鹿用一张（不回手、也不占"首次"），再对逐鹿用 → 回手
    eng.s.round += 1
    eng.round_start()
    eng.s.hand.append("特训战术")
    play(eng, "特训战术", dh)
    assert dh.level == 5 and cnt() == 0, f"对非逐鹿英雄用的不该回手，手牌 {eng.s.hand}"
    eng.s.hand.append("特训战术")
    play(eng, "特训战术", zl)
    assert cnt() == 1, f"新回合首次对逐鹿用的应回手，手牌 {eng.s.hand}"


def t_鬼神降临():
    """你阵亡4名【三分】英雄后，召唤吕布，其等级为已阵亡【三分】英雄等级之和的一半。（非三分阵亡不算）
    读法（与实现统一）：同一场战斗内累计阵亡数（阵亡只发生在战斗里）；吕布是战斗中召唤的单位。本测试 4 名都在同一场里阵亡，
    跨战斗是否累计、吕布战后留不留不断言（待核对）。"""
    need("鬼神降临")
    eng = mk(0)
    pick(eng, "鬼神降临")
    sf = [add(eng, plain("三分")[0], i, level=lv) for i, lv in enumerate((10, 20, 30, 40))]
    other = add(eng, plain("逐鹿")[0], 10, level=99)
    n0 = len(eng.s.units)

    def lvbu():
        return [u for u in eng.s.units[n0:] if u.name.startswith("吕布")]
    eng.fire(other, "death")
    for u in sf[:3]:
        eng.fire(u, "death")
    assert not lvbu(), f"只阵亡 3 名三分（+1 名逐鹿）就召唤了吕布"
    eng.fire(sf[3], "death")
    lb = lvbu()
    assert len(lb) == 1, f"阵亡第 4 名三分后应召唤 1 个吕布，实际 {[(u.name, u.level) for u in lb]}（日志上场 {logged_deploys(eng, '吕布')}）"
    assert lb[0].level == 50, f"吕布等级应为 (10+20+30+40)/2=50，实际 {lb[0].level}"


def t_四海来援():
    """无阵营英雄获得牺牲（召唤同品阶同等级的英雄参战 —— 战斗内，不测）。"""
    eng = mk(0)
    pick(eng, "四海来援")
    wz = add(eng, plain("无阵营")[0], 17)
    zl = add(eng, plain("逐鹿")[0], 3)
    eng.round_start()
    assert eng.has_kw(wz, "牺牲"), "无阵营英雄应获得牺牲"
    assert not eng.has_kw(zl, "牺牲"), "逐鹿英雄不该获得牺牲"


def t_命运图腾():
    """战斗开始时，若你至少有2名【大河】英雄，随机在其周围1格召唤命运图腾。"""
    for n_dh, exp in ((2, True), (1, False)):
        eng = mk(0)
        pick(eng, "命运图腾")
        p = probe(eng)
        dh = [add(eng, nm, idx) for nm, idx in zip(plain("大河", n_dh), (17, 3))]
        add(eng, plain("逐鹿")[0], 24)
        eng.battle()
        totems = [u for u in eng.s.units if "命运图腾" in u.name]
        hit = bool(totems or logged_deploys(eng, "命运图腾") or any("命运图腾" in n for n in p.names()))
        assert hit == exp, f"{n_dh} 名大河英雄：{'应' if exp else '不该'}召唤命运图腾"
        for t in totems:
            assert any(B.hex_dist(t.cell, d.cell) <= 1 for d in dh), f"命运图腾应在大河英雄周围 1 格：{t.cell}"
            assert not t.pop, "图腾不该占人口"


def t_自我奉献():
    """【河洛】英雄获得牺牲：所有友方单位+1临时等级，每次触发此牺牲效果时，临时等级额外+1。
    读法：第 k 次触发给 k 临时等级（同一场里递增）。"""
    eng = mk(0)
    pick(eng, "自我奉献")
    a, b = [add(eng, nm, i) for nm, i in zip(plain("河洛", 2), (17, 18))]
    zl = add(eng, plain("逐鹿")[0], 3)
    sm = add(eng, "因缘图腾", 27, kind="summon")
    eng.round_start()
    assert eng.has_kw(a, "牺牲") and eng.has_kw(b, "牺牲"), "河洛英雄应获得牺牲"
    assert not eng.has_kw(zl, "牺牲"), "逐鹿英雄不该获得牺牲"
    eng.fire(a, "sacrifice")
    assert (zl.temp, sm.temp, b.temp) == (1, 1, 1), f"第 1 次触发：友方各 +1 临时等级，实际 逐鹿{zl.temp} 召唤物{sm.temp} 河洛{b.temp}"
    eng.fire(b, "sacrifice")
    assert (zl.temp, sm.temp) == (3, 3), f"第 2 次触发应再 +2（共 3），实际 逐鹿{zl.temp} 召唤物{sm.temp}"


def t_熊魂附甲():
    """你至少穿戴2件装备的【大河】英雄阵亡获得牺牲：召唤1只与该英雄等级相同的熊灵。"""
    eng = mk(0)
    pick(eng, "熊魂附甲")
    n2, n1 = plain("大河", 2)
    a = add(eng, n2, 17, level=17, equips=FAKE_EQ[:2])
    b = add(eng, n1, 3, level=9, equips=FAKE_EQ[:1])
    eng.round_start()
    assert eng.has_kw(a, "牺牲"), "穿 2 件装备的大河英雄应获得牺牲"
    assert not eng.has_kw(b, "牺牲"), "只穿 1 件的不该获得牺牲"
    n0 = len(eng.s.units)
    eng.fire(a, "sacrifice")
    bears = [u for u in eng.s.units[n0:] if "熊灵" in u.name]
    assert len(bears) == 1, f"牺牲应召唤 1 只熊灵，实际 {[(u.name, u.level) for u in bears]}（日志上场 {logged_deploys(eng, '熊灵')}）"
    assert bears[0].level == 17, f"熊灵等级应与英雄相同 17，实际 {bears[0].level}"


def t_一起发财():
    """使用古币时，其他每个阵营中各1名随机英雄等级+3。（目标河洛 → 逐鹿/大河/三分/日落海各 1 名；同为河洛的不加）"""
    need("古币")
    for seed in range(5):
        eng = mk(seed)
        pick(eng, "一起发财")
        ta, hb = [add(eng, nm, i) for nm, i in zip(plain("河洛", 2), (0, 1))]
        zl = add(eng, plain("逐鹿")[0], 2)
        d1, d2 = [add(eng, nm, i) for nm, i in zip(plain("大河", 2), (3, 4))]
        sf = add(eng, plain("三分")[0], 5)
        rl = add(eng, plain("日落海")[0], 6)
        assert play(eng, "古币", ta) is not False, "古币对河洛英雄被拒"
        assert ta.level == 3, f"seed{seed} 目标应 +2（古币），实际 {ta.level - 1}"
        assert hb.level == 1, f"seed{seed} 同为河洛的不该加，实际 +{hb.level - 1}"
        for u in (zl, sf, rl):
            assert u.level == 4, f"seed{seed} {faction(u.name)} 唯一一名应 +3，实际 +{u.level - 1}"
        assert sorted((d1.level, d2.level)) == [1, 4], f"seed{seed} 大河两名里应恰好 1 名 +3，实际 {d1.level - 1}/{d2.level - 1}"


def t_古币机():
    """召唤古币机，每对其使用4张古币牌（精致古币视为2次），依次获得：随机5张河洛英雄牌；1件贤者的庇护。
    之后随机获得：1张超速成长、1张普通装备牌或8能量。"""
    need("古币")
    for seed in range(4):
        eng = mk(seed)
        pick(eng, "古币机")
        ms = [u for u in eng.s.units if "古币机" in u.name]
        assert len(ms) == 1, f"应召唤 1 个古币机，实际 {[u.name for u in eng.s.units]}"
        m = ms[0]
        assert not m.pop, "古币机不该占人口"
        rejected = []

        def coin(name):
            if eng.use_card(name, m) is False:
                rejected.append(name)
                eng.coin(m, 1, src=name, fine=(name == "精致古币"))     # 兜底：按"对它用了这张古币"继续测奖励
        h0 = list(eng.s.hand)
        for k in range(4):
            coin("古币")
            if k < 3:
                assert not gained(h0, eng.s.hand), f"seed{seed} 第 {k + 1} 张古币就给了奖励 {gained(h0, eng.s.hand)}"
        g = gained(h0, eng.s.hand)
        assert len(g) == 5, f"seed{seed} 4 张古币后应得 5 张河洛英雄牌，实际 {g}"
        hero_cards(g, fac="河洛")
        h1 = list(eng.s.hand)
        coin("精致古币")
        coin("精致古币")
        g = gained(h1, eng.s.hand)
        assert g == ["贤者的庇护"], f"seed{seed} 再 2 张精致古币(=4 次)应得 [贤者的庇护]，实际 {g}"
        h2, e2 = list(eng.s.hand), eng.s.energy
        for _ in range(4):
            coin("古币")
        g, de = gained(h2, eng.s.hand), eng.s.energy - e2
        ok = (g == ["超速成长"] and de == 0) or (len(g) == 1 and category(g[0]) == "装备牌" and de == 0) or (not g and de == 8)
        assert ok, f"seed{seed} 第三轮应得 超速成长 / 1 张普通装备 / 8 能量 之一，实际 牌{g} 能量+{de}"
        assert not rejected, f"古币 / 精致古币不能对古币机打出（use_card 返回 False）：{rejected}"


def t_虚妄转生():
    """召唤1个虚妄图腾（被摧毁时变身 —— 战斗内，不测）。"""
    eng = mk(0)
    pick(eng, "虚妄转生")
    ts = [u for u in eng.s.units if "虚妄图腾" in u.name]
    assert len(ts) == 1, f"应召唤 1 个虚妄图腾，实际 {[u.name for u in eng.s.units]}"
    assert not ts[0].pop and ts[0].kind != "hero", f"图腾应是不占人口的召唤物：kind={ts[0].kind} pop={ts[0].pop}"


def t_升值():
    """获得2张古币。战斗胜利时，古币牌提供的等级永久+1。（第 w 场胜利后古币给 2+w）"""
    need("古币")
    seen = set()
    for seed in range(6):
        eng = mk(seed)
        pick(eng, "升值")
        assert eng.s.hand.count("古币") == 2, f"应获得 2 张古币：{eng.s.hand}"
        u = add(eng, plain("河洛")[0], 17, level=10)
        wins = 0
        for _ in range(3):
            w = eng.battle()
            wins += w
            seen.add(w)
        lv = u.level
        assert play(eng, "古币", u) is not False, "古币被拒"
        assert u.level - lv == 2 + wins, f"seed{seed} 胜 {wins} 场后古币应 +{2 + wins}，实际 +{u.level - lv}"
    assert seen == {True, False}, f"种子没覆盖胜负两种：{seen}"


def t_灵祭升华():
    """场上的图腾被摧毁时，等级最高的【大河】英雄获得图腾一半等级的临时等级。"""
    eng = mk(0)
    pick(eng, "灵祭升华")
    lo, hi = [add(eng, nm, i, level=lv) for nm, i, lv in zip(plain("大河", 2), (17, 18), (10, 30))]
    zl = add(eng, plain("逐鹿")[0], 3, level=50)
    tot = add(eng, "因缘图腾", 27, kind="summon", level=20)
    eng.fire(tot, "death")
    assert hi.temp == 10, f"等级最高的大河英雄应得 20/2=10 临时等级，实际 {hi.temp}"
    assert lo.temp == 0 and zl.temp == 0, f"其他人不该得：大河低 {lo.temp} 逐鹿 {zl.temp}"
    assert hi.level == 30, "是临时等级，不该加永久等级"


def t_神鹰锻匠():
    """战斗开始时，云中君为距离最近的1个【大河】英雄添加1件神兵装备。若云中君觉醒，神兵至多添加3件。
    断言：最近的大河英雄战斗中出现神兵（未觉醒峰值恰 1 件、觉醒 1~3 件），别人没有。"""
    for awk, lo_, hi_ in ((0, 1, 1), (3, 1, 3)):
        eng = mk(0)
        pick(eng, "神鹰锻匠")
        p = probe(eng)
        add(eng, "云中君", 17, awaken=awk)
        near = add(eng, plain("大河")[0], 18)
        far = add(eng, plain("大河", 2)[1], 0)
        zl = add(eng, plain("逐鹿")[0], 16)
        for _ in range(2):
            eng.battle()

        def peak(u):
            return max([sum(1 for e in r[4] if e in SHENBING) for _, r in p.rows(u.uid)] + [0])

        def any_new(u):
            return {e for _, r in p.rows(u.uid) for e in r[4]}
        pn = peak(near)
        assert lo_ <= pn <= hi_, f"{'觉醒' if awk else '未觉醒'}云中君：最近大河英雄神兵峰值应在 {lo_}~{hi_}，实际 {pn}（见过的装备 {any_new(near)}）"
        assert not any_new(far) and not any_new(zl), f"其他人不该拿到装备：远处大河 {any_new(far)} 逐鹿 {any_new(zl)}"


def t_全域共鸣():
    """随机获得1件火鹰装备。（读法：火鹰装备 = 卡面写【日落海】专属的装备）"""
    for seed in range(5):
        eng = mk(seed)
        h0 = list(eng.s.hand)
        pick(eng, "全域共鸣")
        g = gained(h0, eng.s.hand)
        assert len(g) == 1 and category(g[0]) == "装备牌", f"seed{seed} 应得 1 件装备牌，实际 {g}"
        assert "日落海" in card_text(g[0]), f"seed{seed} {g[0]} 不是【日落海】专属装备"


def t_千里相聚():
    """每打出3张无阵营英雄牌时，全体英雄等级+4。（上场、合成都算打出英雄牌）"""
    wz = plain("无阵营", 2)
    eng = mk(0)
    pick(eng, "千里相聚")
    other = add(eng, plain("逐鹿")[0], 0, level=1)
    a = eng.deploy(wz[0], cell=B.CELLS[17])
    b = eng.deploy(wz[1], cell=B.CELLS[18])
    eng.deploy(plain("逐鹿")[0], cell=B.CELLS[19])               # 非无阵营不计数
    assert other.level == 1 and a.level == 1, f"才 2 张无阵营就加了等级：{other.level}/{a.level}"
    c = eng.deploy(wz[0], cell=B.CELLS[20])
    assert (other.level, a.level, b.level, c.level) == (5, 5, 5, 5), \
        f"第 3 张无阵营英雄牌打出后全体 +4，实际 {other.level}/{a.level}/{b.level}/{c.level}"
    # 合成也算打出：再合成 3 次 → 再 +4（a 自己另有合成 +3×3）
    for _ in range(3):
        eng.merge(b)
    assert other.level == 9, f"合成 3 张无阵营英雄牌也应触发 +4，旁人 5→{other.level}"


def t_起舞_共战():
    """共战时（貂蝉、上官婉儿都在场），参与击败敌方英雄后，二人均+1级（+10 法攻是战斗效果）。对照组扣掉英雄自身效果。"""
    need("起舞")

    def run(tal, who, n, both=True):
        eng = mk(0)
        if tal:
            pick(eng, "起舞")
        dc = add(eng, "貂蝉", 17, level=5)
        wr = add(eng, "上官婉儿", 18, level=5) if both else None
        eng.fire(dc if who == 0 else wr, "kill", n=n)
        return dc.level, (wr.level if wr else 0)
    for who, n, exp in ((0, 1, (1, 1)), (1, 1, (1, 1)), (0, 2, (2, 2))):
        a, b = run(True, who, n), run(False, who, n)
        d = (a[0] - b[0], a[1] - b[1])
        assert d == exp, f"{'貂蝉' if who == 0 else '婉儿'}击败 {n} 名：二人应各 +{exp[0]}，实际 {d}"
    a, b = run(True, 0, 1, both=False), run(False, 0, 1, both=False)
    assert a[0] == b[0], f"婉儿不在场时不该触发：貂蝉多了 {a[0] - b[0]}"


def t_巨像_神兽():
    """二人在场时，召唤与米莱狄等级相同的蔚蓝神兽。二人等级之和达到100时，召唤与蒙恬等级相同的猩红神兽。"""
    eng = mk(0)
    pick(eng, "巨像")
    for x in ("米莱狄", "蒙恬"):
        if x in eng.s.hand:
            eng.s.hand.remove(x)
    ml = eng.deploy("米莱狄", cell=B.CELLS[17], level=20)
    eng.round_start()
    assert not [u for u in eng.s.units if "神兽" in u.name], "只有米莱狄在场就召唤了神兽"
    mt = eng.deploy("蒙恬", cell=B.CELLS[18], level=30)
    eng.round_start()
    blue = [u for u in eng.s.units if u.name == "蔚蓝神兽"]
    assert len(blue) == 1, f"二人在场应召唤 1 只蔚蓝神兽，实际 {[u.name for u in eng.s.units]}"
    assert blue[0].level == 20, f"蔚蓝神兽等级应 = 米莱狄 20，实际 {blue[0].level}"
    assert not blue[0].pop, "神兽不该占人口"
    assert not [u for u in eng.s.units if u.name == "猩红神兽"], "等级和 50 就召唤了猩红神兽"
    ml.level, mt.level = 60.0, 45.0
    eng.round_start()
    red = [u for u in eng.s.units if u.name == "猩红神兽"]
    if not red:
        eng.battle()
        red = [u for u in eng.s.units if u.name == "猩红神兽"]
    assert len(red) == 1, f"等级和 105 应召唤 1 只猩红神兽，实际 {[u.name for u in eng.s.units]}"
    assert red[0].level == 45, f"猩红神兽等级应 = 蒙恬 45，实际 {red[0].level}"
    assert len([u for u in eng.s.units if u.name == "蔚蓝神兽"]) == 1, "蔚蓝神兽被重复召唤"


def t_点兵点将_赠送牌():
    """赠送牌（given_skill）：获得场上所有英雄牌各1张。（召唤物不算）"""
    eng = mk(0)
    pick(eng, "点兵点将")
    a = add(eng, plain("逐鹿")[0], 17)
    b = add(eng, plain("大河")[0], 18)
    add(eng, "因缘图腾", 27, kind="summon")
    h0 = [x for x in eng.s.hand if x != "点兵点将"]
    ok = play(eng, "点兵点将", a)
    assert ok is not False, "打出点兵点将被拒"
    g = gained(h0, eng.s.hand)
    assert sorted(g) == sorted([a.name, b.name]), f"应获得场上英雄各 1 张 {[a.name, b.name]}，实际 {g}"


def t_丝绸之路():
    """每出售4名场上英雄获得1张古币。（"每出售1张古币获得1张河洛英雄牌"：引擎没有出售手牌的接口，不测）"""
    for n, exp in ((3, 0), (4, 1), (8, 2)):
        eng = mk(0)
        pick(eng, "丝绸之路")
        us = [add(eng, plain("三分")[0], i) for i in range(n)]
        h0 = list(eng.s.hand)
        for u in us:
            eng.sell(u)
        g = gained(h0, eng.s.hand)
        assert g == ["古币"] * exp, f"出售 {n} 名英雄应得 {exp} 张古币，实际 {g}"


def t_慕名而来():
    """你的英雄每触发4次登场效果，随机获得1张英雄牌。（武则天 登场：获得1张精致古币；白板英雄没有登场效果）"""
    for name, n, exp in (("武则天", 4, 1), ("武则天", 3, 0), (plain("逐鹿")[0], 4, 0)):
        eng = mk(0)
        pick(eng, "慕名而来")
        h0 = list(eng.s.hand)
        for i in range(n):
            eng.deploy(name, cell=B.CELLS[10 + i])
        g = [x for x in gained(h0, eng.s.hand) if category(x) == "英雄牌"]
        assert len(g) == exp, f"{name} 登场 {n} 次应得 {exp} 张英雄牌，实际 {g}"


def t_超级蓄能():
    """回合开始时，你的整备效果英雄会提升场上整备效果英雄数x2的等级。（3 名整备英雄 → 各 +6；对照组扣掉整备本身）"""
    need("超级蓄能")

    def run(tal):
        eng = mk(0)
        if tal:
            pick(eng, "超级蓄能")
        us = [add(eng, "程咬金", 17, level=10), add(eng, "程咬金", 18, level=10), add(eng, "西施", 19, level=10),
              add(eng, plain("三分")[0], 3, level=10)]
        eng.round_start()
        return [u.level for u in us]
    a, b = run(True), run(False)
    d = [x - y for x, y in zip(a, b)]
    assert d == [6, 6, 6, 0], f"整备英雄各应 +3×2=6、非整备 0，实际差值 {d}"


def t_摄人心魄():
    """你的夺取效果英雄击败敌方英雄时，将永久夺取对方3级。（孙悟空有夺取；对照组扣掉自身效果）"""
    need("摄人心魄")

    def run(tal, name, n):
        eng = mk(0)
        if tal:
            pick(eng, "摄人心魄")
        u = add(eng, name, 17, level=10)
        eng.fire(u, "kill", n=n)
        return u.level
    for name, n, exp in (("孙悟空", 1, 3), ("孙悟空", 2, 6), (plain("逐鹿")[0], 2, 0)):
        d = run(True, name, n) - run(False, name, n)
        assert d == exp, f"{name} 击败 {n} 名：应 +{exp}，实际 +{d}"


def t_生生不息():
    """4/5阶的召唤单位获得完美复生。（复生 = 战斗中多一条命；看开战后 counters['revive']）"""
    eng = mk(0)
    pick(eng, "生生不息")
    p = probe(eng)
    h = add(eng, plain("逐鹿")[0], 3)
    s4 = add(eng, "蔚蓝神兽", 17, kind="summon")      # 召唤物 4 阶
    s2 = add(eng, "玄雍士兵", 18, kind="summon")      # 召唤物 2 阶
    eng.round_start()
    eng.battle()

    def rv(u):
        return max([r[5] for _, r in p.rows(u.uid)] + [0])
    assert rv(s4) >= 1, f"4 阶召唤物应获得复生，实际 revive={rv(s4)}"
    assert rv(s2) == 0 and rv(h) == 0, f"2 阶召唤物 / 英雄不该获得：{rv(s2)} / {rv(h)}"


# ================================================================ 阿科米亚核心
def t_盗火():
    """转瞬 指定1名英雄，至多偷取阿科米亚核心20级。（核心空时 0；用能量汲取把核心充到 30：20 → 10 → 0）"""
    eng = mk(0)
    u = add(eng, plain("逐鹿")[0], 17, level=5)
    play(eng, "盗火", u)
    assert u.level == 5, f"核心为 0 时不该偷到等级：{u.level}"
    load_core(eng, 30)
    for exp in (20, 10, 0):
        lv = u.level
        play(eng, "盗火", u)
        assert u.level - lv == exp, f"核心 30 依次偷应 20/10/0，这次实际 {u.level - lv}（依赖能量汲取充核心）"


def t_能量汲取():
    """出售场上英雄时，阿科米亚核心会获得其等级。（核心等级用盗火读出）"""
    eng = mk(0)
    load_core(eng, 30)
    c = core_level(eng)
    assert c == 30, f"出售 30 级英雄后核心应 30，读到 {c}"
    load_core(eng, 7)
    c = core_level(eng)
    assert c == 37, f"再出售 7 级后核心应 37，读到 {c}"


def t_能源置换():
    """回合开始时，阿科米亚核心失去20级，获得10能量。（核心 25 → 5，能量 +10）"""
    eng = mk(0)
    load_core(eng, 25)
    pick(eng, "能源置换")
    e0 = eng.s.energy
    eng.round_start()
    assert eng.s.energy - e0 == 10, f"回合开始应 +10 能量，实际 +{eng.s.energy - e0}"
    c = core_level(eng)
    assert c == 5, f"核心应 25-20=5，读到 {c}"


def t_积少成多():
    """【日落海】英雄每提升10级，阿科米亚核心获得4等级。（+20 级 → 核心 +8；+9 级 → 0）"""
    for up, exp in ((20, 8), (9, 0)):
        eng = mk(0)
        pick(eng, "积少成多")
        u = add(eng, plain("日落海")[0], 17, level=1)
        eng.round_start()
        base = core_level(eng)
        eng.level(u, up, src="测试")
        c = core_level(eng) - base
        if c == 0 and exp:
            eng.round_start()
            c = core_level(eng) - base
        assert c == exp, f"日落海英雄 +{up} 级，核心应 +{exp}，读到 +{c}"


def t_定向传输():
    """阿科米亚核心的等级会优先分配给穿戴装备最多的【日落海】英雄，最多分配200级。
    （核心 100 → 穿装备最多的那名日落海英雄开战拿到 100 临时等级，另一名 0；依赖引擎 / 别的卡实现核心分配）"""
    eng = mk(0)
    load_core(eng, 100)
    pick(eng, "定向传输")
    p = probe(eng)
    a, b = [add(eng, nm, i, level=10, equips=eq) for nm, i, eq in zip(plain("日落海", 2), (17, 18), (FAKE_EQ[:2], None))]
    eng.battle()
    ra, rb = p.at_end(a.uid), p.at_end(b.uid)
    assert ra and ra[3] == 100, f"穿 2 件装备的日落海英雄应拿到全部 100 临时等级，实际 {ra and ra[3]}"
    assert rb and rb[3] == 0, f"另一名不该分到，实际 {rb and rb[3]}"


def t_拾荒者():
    """恢复10棋手生命值。之后每有1位棋手被淘汰，获得2张5阶英雄兑换券。（State.hp / eng.eliminate）"""
    eng = mk(0, hp=40)
    h0 = list(eng.s.hand)
    pick(eng, "拾荒者")
    assert eng.s.hp == 50, f"生命应 40+10=50，实际 {eng.s.hp}"
    assert not gained(h0, eng.s.hand), f"选天赋时不该给牌：{gained(h0, eng.s.hand)}"
    eng.eliminate(2)
    g = gained(h0, eng.s.hand)
    assert sorted(g) == ["5阶英雄兑换券"] * 4, f"淘汰 2 位应得 4 张 5 阶英雄兑换券，实际 {g}"
    eng = mk(0, hp=55)
    pick(eng, "拾荒者")
    assert eng.s.hp == 60, f"生命上限 60：{eng.s.hp}"


# ================================================================ 运行
TESTS = [(f"{n}·获得", _gives(n, e)) for n, e in GIVES.items() if e is not None] + [
    ("天兵", t_天兵),
    ("三分强援", _reinforce("三分强援", "三分之地", 5, {4, 5})), ("三分增援", _reinforce("三分增援", "三分之地", 0, {3, 4})),
    ("日落海强援", _reinforce("日落海强援", "日落海", 5, {4, 5})), ("日落海增援", _reinforce("日落海增援", "日落海", 0, {3, 4})),
    ("大河强援", _reinforce("大河强援", "大河流域", 5, {4, 5})), ("大河增援", _reinforce("大河增援", "大河流域", 0, {3, 4})),
    ("逐鹿强援", _reinforce("逐鹿强援", "逐鹿", 5, {4, 5})), ("逐鹿增援", _reinforce("逐鹿增援", "逐鹿", 0, {3, 4})),
    ("河洛强援", _reinforce("河洛强援", "河洛", 5, {4, 5})), ("河洛增援", _reinforce("河洛增援", "河洛", 0, {3, 4})),
    ("令行如流", t_令行如流), ("演武", t_演武), ("核心裂痕·回合", t_核心裂痕_回合), ("春风化雨", t_春风化雨),
    ("望梅止渴·回合", t_望梅止渴_回合), ("望梅止渴·赠送牌", t_望梅止渴_赠送牌),
    ("四重奏", t_四重奏), ("意外之财", t_意外之财), ("复现", t_复现), ("鬼神降临", t_鬼神降临), ("四海来援", t_四海来援),
    ("命运图腾", t_命运图腾), ("自我奉献", t_自我奉献), ("熊魂附甲", t_熊魂附甲), ("一起发财", t_一起发财),
    ("古币机", t_古币机), ("虚妄转生", t_虚妄转生), ("升值", t_升值), ("灵祭升华", t_灵祭升华),
    ("神鹰锻匠·神兵", t_神鹰锻匠), ("全域共鸣", t_全域共鸣), ("千里相聚", t_千里相聚), ("起舞·共战", t_起舞_共战),
    ("巨像·神兽", t_巨像_神兽), ("点兵点将·赠送牌", t_点兵点将_赠送牌), ("丝绸之路", t_丝绸之路), ("慕名而来", t_慕名而来),
    ("超级蓄能", t_超级蓄能), ("摄人心魄", t_摄人心魄), ("生生不息", t_生生不息),
    ("拾荒者", t_拾荒者), ("盗火", t_盗火), ("能量汲取", t_能量汲取), ("能源置换", t_能源置换), ("积少成多", t_积少成多), ("定向传输", t_定向传输),
]


def main() -> int:
    batch = json.loads((ROOT / "data/engine/sim_batches.json").read_text(encoding="utf-8"))["batches"][5]
    covered = {n.split("·")[0] for n, _ in TESTS} | set(COMBAT_ONLY) | set(NOT_TESTABLE)
    missing = [n for n in batch if n not in covered]
    ok = fail = 0
    for name, fn in TESTS:
        try:
            if name.split("·")[0] not in card_db():
                raise AssertionError("卡库里没有这张卡")
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
        finally:
            REGISTRY.pop(PROBE, None)
    for n, why in COMBAT_ONLY.items():
        print(f"SKIP  {n}: 只有战斗效果（{why}）")
    for n, why in NOT_TESTABLE.items():
        print(f"SKIP  {n}: {why}")
    if missing:
        print(f"注意：本批这些卡没有归类 {missing}")
    print(f"\n通过 {ok} / 失败 {fail}（共 {ok + fail} 条测试，跳过 {len(COMBAT_ONLY) + len(NOT_TESTABLE)} 张）")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
