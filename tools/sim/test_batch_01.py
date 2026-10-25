"""第 1 批卡（data/engine/sim_batches.json 的 batches[1]）的独立测试 —— 只照卡面原文（data/cards.json）写，没读实现。

运行：.venv/Scripts/python.exe -X utf8 tools/sim/test_batch_01.py
逐条打印 OK / FAIL（含原因），末尾打印通过数 / 失败数；有失败时退出码 1。
没注册实现的卡记为 FAIL "未实现"，不中断。

约定：
- 搭局面时场上单位直接放进 s.units（不走 eng.deploy，避免触发它们自己的登场效果）；只有"被测卡本身的登场"才走 eng.deploy。
- 陪练英雄尽量选卡面只有战斗效果的（墨子 / 扁鹊 / 廉颇 逐鹿，沈梦溪 河洛，马超 / 诸葛亮 / 刘禅 三分，
  敖隐 / 瑶 大河，艾琳 / 姬小满 日落海，苍 / 海月 无阵营），整备 / 登场的触发对象用 pilot 里已写好的（西施 / 程咬金 / 武则天）。
- 阿科米亚核心：State / Engine 没有核心等级字段（接口缺口），这里按"名字里带 核心 / core / 阿科米亚 的数值"
  在 State 属性、s.flags、s.counters、Engine 属性、场上单位里找，比较使用前后的差值。
- 随机效果用多个种子跑，断言总量 / 范围。
"""
from __future__ import annotations

import random
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.engine import board as B                      # noqa: E402
from agent.sim.engine import Engine, impl, card_db, category, quality, faction, card_text  # noqa: E402
import agent.sim.engine as _E                                                     # noqa: E402
_E.EQUIP_DROP_ROUNDS = ()      # 固定回合投放装备（开局 / 第 4/8/12/16 回合）是引擎规则，卡牌测试里关掉，免得混进手牌 / 装备数的断言
from agent.sim.state import State, Unit                  # noqa: E402

DB = card_db()
BASE_EQUIPS = {"守护者之铠", "力量腰带", "圣者法典", "破碎圣杯", "大棒", "雷鸣刃", "速击之枪", "风暴巨剑"}
TACTICS = {n for n, c in DB.items() if n.endswith("战术") and c.get("category") == "效果牌"}
SUNSET_EXCLUSIVE = {"星源装甲", "星源狩枪", "星源星徽", "星源合金盾", "月影之力"}
SEEDS = range(6)


# ============================================================ 工具
class Fail(AssertionError):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def new_eng(seed: int = 0, **kw) -> Engine:
    st = State(**kw)
    eng = Engine(st, random.Random(seed))
    eng.calls = []

    def rec(e, kind, options, **ctx):
        e.calls.append((kind, list(options), ctx))
        return options[0]
    eng.chooser = rec
    return eng


def put(eng: Engine, name: str, level: float = 1.0, cell: int | None = None, awaken: int = 0,
        equips=None, kind: str = "hero", pop: bool = True) -> Unit:
    """直接把单位放上场（不触发登场）。cell = B.CELLS 的下标；None = 第一个空格。"""
    used = {u.cell for u in eng.s.units}
    c = B.CELLS[cell] if cell is not None else next(x for x in B.CELLS if x not in used)
    u = Unit(name=name, level=float(level), cell=c, awaken=awaken, equips=list(equips or []), kind=kind,
             pop=pop, uid=eng.s.next_uid)
    eng.s.next_uid += 1
    eng.s.units.append(u)
    return u


def gained(eng: Engine, before: list) -> list:
    """手牌里多出来的牌（多重集合差）。"""
    d = Counter(eng.s.hand) - Counter(before)
    return list(d.elements())


def use(eng: Engine, name: str, target=None):
    hand0 = list(eng.s.hand)
    r = eng.use_card(name, target)
    return r, gained(eng, hand0)


def kws(name: str) -> set:
    return {k.get("name") for k in (DB.get(name) or {}).get("keywords") or []}


def hero_has(name: str, kw: str) -> bool:
    return kw in kws(name) or kw in card_text(name)


def is_hero(name: str) -> bool:
    return category(name) == "英雄牌"


def total_xp(eng: Engine, lvl0: int) -> int:
    s = eng.s
    return s.xp + sum(B.XP_NEED[l] for l in range(lvl0, s.lord_level))


def _core_items(eng: Engine) -> dict:
    out = {}

    def ok_key(k):
        k = str(k)
        return not k.startswith("round:") and ("核心" in k or "core" in k.lower() or "阿科米亚" in k)

    def num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    for k, v in vars(eng.s).items():
        if ok_key(k) and num(v):
            out["s." + k] = v
    for k, v in (eng.s.flags or {}).items():
        if ok_key(k) and num(v):
            out["flags." + str(k)] = v
    cnt = getattr(eng.s, "counters", None)
    if isinstance(cnt, dict):
        for k, v in cnt.items():
            if ok_key(k) and num(v):
                out["counters." + str(k)] = v
    for k, v in vars(eng).items():
        if ok_key(k) and num(v):
            out["eng." + k] = v
    for u in eng.s.units:
        if "阿科米亚" in u.name or "核心" in u.name:
            out["unit." + u.name] = u.level
    return out


def core(eng: Engine) -> float:
    return float(sum(_core_items(eng).values()))


# ============================================================ 注册
TESTS = []


def test(*cards, desc=""):
    def deco(f):
        TESTS.append((cards, desc or f.__doc__ or f.__name__, f))
        return f
    return deco


SKIPPED = {
    "吕布": "开团：吸血（战斗效果）",
    "海月": "牺牲：控制 + 真实伤害（战斗效果）",
    "高阶援护战术": "护盾（战斗效果）",
    "星源装甲": "吸血 / 免控（战斗效果）",
    "星源狩枪": "攻击力 / 法力 / 斩杀（战斗效果）",
    "星源合金盾": "防御 / 减伤（战斗效果）",
    "月影之力": "生命值 / 露娜额外技能（战斗效果）",
    "高阶推演战术": "法力值上限（战斗效果）",

    "奇谋": "开战立即放技能（战斗效果）",
    "开局狂飙": "前 15 秒无限放技能（战斗效果）",
}


def _plain_heroes(fac: str, n: int = 2) -> list:
    """该派系里实现没有任何养成钩子的英雄（只拿来占位）。"""
    out = []
    for name, c in DB.items():
        if c.get("category") != "英雄牌" or faction(name) != fac:
            continue
        i = impl(name)
        if i is not None and any(a.startswith(("on_", "react_")) or a in ("coin_bonus", "grant_keywords") for a in dir(i)):
            continue
        out.append(name)
    check(len(out) >= n, f"【{fac}】里白板英雄不够 {n} 个：{out}")
    return out[:n]


@test("星源星徽", desc="穿戴者不被阿科米亚核心分配等级（核心 100、2 名日落海：穿的 0，没穿的 50）")
def t_xingyuan_xinghui():
    a_name, b_name = "艾琳", "雅典娜"                      # 日落海；这里只调 core_distribute，不触发它们的效果
    eng = new_eng()
    a = put(eng, a_name, level=10)
    b = put(eng, b_name, level=10)
    ok = eng.equip(a, "星源星徽")
    check(ok, "星源星徽应能穿给日落海英雄")
    eng.s.core = 100
    eng.core_distribute()
    check(a.temp == 0, f"穿星源星徽的不该分到核心等级：{a.temp}")
    check(b.temp == 50, f"另一名日落海英雄应分到 min(100/1, 50%) = 50：{b.temp}")
    eng = new_eng()
    a = put(eng, a_name, level=10)
    b = put(eng, b_name, level=10)
    eng.s.core = 100
    eng.core_distribute()
    check(a.temp == 50 and b.temp == 50, f"对照：都不穿时各 50：{a.temp} / {b.temp}")


@test("展翅腾飞", desc="本回合刷新商店不耗能量；下回合恢复收费")
def t_zhanchi():
    eng = new_eng(energy=5)
    r, _ = use(eng, "展翅腾飞")
    check(r is not False, "使用返回 False")
    for _ in range(4):
        check(eng.refresh_shop(), "本回合刷新应成功")
    check(eng.s.energy == 5 and eng.s.free_refresh == 0, f"本回合刷新不该花能量：能量 {eng.s.energy}")
    eng.round_start()
    check(eng.refresh_shop() and eng.s.energy == 5 - B.REFRESH_COST, f"下回合刷新应扣 {B.REFRESH_COST}：能量 {eng.s.energy}")


@test("机关匣", desc="故障机器人无法攻击：战斗里不分走击杀")
def t_bot_no_kill():
    for seed in range(20):
        eng = new_eng(seed, lord_level=6)
        m = put(eng, "墨子", level=10)
        eng.equip(m, "机关匣")
        fired = []
        orig = eng.fire

        def rec(u, timing, **kw):
            fired.append((u.name, timing))
            return orig(u, timing, **kw)
        eng.fire = rec
        eng.battle()
        check(not any(u.name == "故障机器人" for u in eng.s.units), "战斗结束故障机器人应移除")
        check(not any(n == "故障机器人" and t == "kill" for n, t in fired), "故障机器人不该有击杀")


# ============================================================ 英雄
@test("鲁班七号", desc="你的英雄每合成 3 次，随机获得 1 张战术牌")
def t_luban7():
    eng = new_eng()
    put(eng, "鲁班七号")
    m = put(eng, "墨子")
    eng.s.merges = 0
    eng.merge(m)
    eng.merge(m)
    t2 = [x for x in eng.s.hand if x in TACTICS]
    check(not t2, f"合成 2 次就给了战术牌：{t2}")
    eng.merge(m)
    t3 = [x for x in eng.s.hand if x in TACTICS]
    check(len(t3) == 1, f"合成 3 次应得 1 张战术牌，实际手牌 {eng.s.hand}")


@test("海诺", "助力核心", desc="每当核心等级提升时额外 +1（与同配置无海诺对比，差值 = 提升次数）")
def t_hainuo():
    a = new_eng()
    put(a, "海诺")
    put(a, "艾琳")
    c0 = core(a)
    a.use_card("助力核心")
    da = core(a) - c0
    b = new_eng()
    put(b, "姬小满")
    put(b, "艾琳")
    c0 = core(b)
    b.use_card("助力核心")
    db_ = core(b) - c0
    check(db_ > 0, f"前置：无海诺时助力核心没提升核心（找到的核心字段 {_core_items(b)}）")
    raises = db_ / 2
    check(da - db_ == raises, f"海诺应让每次 +2 各额外 +1：无海诺 +{db_}（{raises:g} 次提升），有海诺 +{da}")


@test("露娜", desc="登场：核心 +2，随机触发 2 名英雄的整备")
def t_luna():
    eng = new_eng()
    put(eng, "西施", cell=0)
    cyj = put(eng, "程咬金", cell=1)
    c0 = core(eng)
    hand0 = list(eng.s.hand)
    eng.deploy("露娜", cell=B.CELLS[2])
    check(core(eng) - c0 == 2, f"核心应 +2，实际 +{core(eng) - c0}（字段 {_core_items(eng)}）")
    check(cyj.level == 3, f"程咬金整备（自用古币 +2）应被触发：等级 {cyj.level}")
    g = gained(eng, hand0)
    check(g.count("推演战术") == 1, f"西施整备（得推演战术）应被触发：新增手牌 {g}")


@test("杨玉环", desc="在场每过 3 个回合得 1 张弦音（6 回合 = 2 张）")
def t_yyh_rounds():
    eng = new_eng()
    eng.deploy("杨玉环")
    n0 = eng.s.hand.count("弦音")
    for _ in range(6):
        eng.round_start()
        eng.battle()
        eng.round_end()
    n = eng.s.hand.count("弦音") - n0
    check(n == 2, f"6 个回合应得 2 张弦音，实际 {n}（手牌 {eng.s.hand}）")


@test("杨玉环", desc="退场（出售）：获得 1 张弦音")
def t_yyh_sell():
    eng = new_eng()
    u = put(eng, "杨玉环")
    hand0 = list(eng.s.hand)
    eng.sell(u)
    g = gained(eng, hand0)
    check(g.count("弦音") == 1, f"出售应得 1 张弦音，新增手牌 {g}")


@test("周瑜", desc="登场：全体英雄等级 +1（含自己）")
def t_zhouyu():
    eng = new_eng()
    a = put(eng, "墨子", level=1)
    b = put(eng, "扁鹊", level=5)
    z = eng.deploy("周瑜")
    check(a.level == 2 and b.level == 6, f"其他英雄应 +1：墨子 {a.level} 扁鹊 {b.level}")
    check(z.level == 2, f"周瑜自己也应 +1：{z.level}")


# ============================================================ 觉醒进度
@test("弦音", desc="指定 1 名 1-3 阶英雄觉醒进度 +1；4 阶不合法")
def t_xianyin():
    eng = new_eng()
    m = put(eng, "墨子", level=4)           # 2 阶
    r = eng.use_card("弦音", m)
    check(r is not False and m.awaken == 1, f"2 阶墨子觉醒进度应 +1：返回 {r}，进度 {m.awaken}")
    check(m.level == 4, f"卡面只加觉醒进度，等级不该变：{m.level}")
    b = put(eng, "扁鹊")                    # 4 阶
    r = eng.use_card("弦音", b)
    check(r is False and b.awaken == 0, f"4 阶扁鹊不合法：返回 {r}，进度 {b.awaken}")


@test("妙音", desc="指定 1 名英雄觉醒进度 +1（不限品阶）")
def t_miaoyin():
    eng = new_eng()
    b = put(eng, "扁鹊", level=4)
    r = eng.use_card("妙音", b)
    check(r is not False and b.awaken == 1, f"扁鹊觉醒进度应 +1：返回 {r}，进度 {b.awaken}")
    check(b.level == 4, f"等级不该变：{b.level}")


@test("天籁琴音", desc="获得 3 张妙音")
def t_tianlai():
    eng = new_eng()
    _, g = use(eng, "天籁琴音")
    check(g.count("妙音") == 3 and len(g) == 3, f"应得 3 张妙音：{g}")


@test("久经沙场", desc="指定已觉醒【逐鹿】英雄，获得 1 张该英雄牌")
def t_jiujing():
    eng = new_eng()
    m = put(eng, "墨子", awaken=3)
    r, g = use(eng, "久经沙场", m)
    check(r is not False and g == ["墨子"], f"应得 1 张墨子：返回 {r} 新增 {g}")
    l = put(eng, "廉颇", awaken=1)
    r, g = use(eng, "久经沙场", l)
    check(r is False and not g, f"未觉醒的廉颇不合法：返回 {r} 新增 {g}")
    s = put(eng, "沈梦溪", awaken=3)
    r, g = use(eng, "久经沙场", s)
    check(r is False and not g, f"已觉醒但不是逐鹿（沈梦溪）不合法：返回 {r} 新增 {g}")


# ============================================================ 棋手经验 / 能量 / 刷新
@test("研读会", desc="获得 3 棋手经验")
def t_yandu():
    eng = new_eng(lord_level=1, xp=0)
    eng.use_card("研读会")
    check(total_xp(eng, 1) == 3, f"经验应 +3：等级 {eng.s.lord_level} 经验 {eng.s.xp}")


@test("经典研读会", desc="获得 4 棋手经验")
def t_jingdian():
    eng = new_eng(lord_level=1, xp=0)
    eng.use_card("经典研读会")
    check(total_xp(eng, 1) == 4, f"经验应 +4：等级 {eng.s.lord_level} 经验 {eng.s.xp}")


@test("惊喜补贴", desc="获得 3 次免费刷新")
def t_jingxi():
    eng = new_eng()
    eng.use_card("惊喜补贴")
    check(eng.s.free_refresh == 3, f"免费刷新应 = 3：{eng.s.free_refresh}")


@test("超级补贴", desc="获得 4 次免费刷新")
def t_chaoji():
    eng = new_eng()
    eng.use_card("超级补贴")
    check(eng.s.free_refresh == 4, f"免费刷新应 = 4：{eng.s.free_refresh}")


@test("能量大爆炸", desc="获得 5 能量")
def t_nengliang():
    eng = new_eng(energy=3)
    eng.use_card("能量大爆炸")
    check(eng.s.energy == 8, f"能量应 3→8：{eng.s.energy}")


# ============================================================ 获得英雄牌
def _hero_cards(card, n, pred, why, seeds=SEEDS, setup=None, **st):
    for sd in seeds:
        eng = new_eng(sd, **st)
        tgt = setup(eng) if setup else None
        r, g = use(eng, card, tgt)
        check(r is not False, f"种子 {sd}：使用失败")
        check(len(g) == n, f"种子 {sd}：应得 {n} 张，实际 {g}")
        bad = [x for x in g if not (is_hero(x) and pred(x))]
        check(not bad, f"种子 {sd}：{bad} 不是{why}")


@test("吸纳人才", desc="卡牌 4 选 1（英雄牌）")
def t_xina():
    for sd in SEEDS:
        eng = new_eng(sd)
        r, g = use(eng, "吸纳人才")
        check(len(g) == 1 and is_hero(g[0]), f"种子 {sd}：应得 1 张英雄牌：{g}")
        opts = [c[1] for c in eng.calls]
        check(any(len(o) == 4 and all(is_hero(x) for x in o) for o in opts), f"种子 {sd}：应弹 4 个英雄选项：{opts}")


@test("广纳良才", desc="卡牌 5 选 2（英雄牌）")
def t_guangna():
    for sd in SEEDS:
        eng = new_eng(sd)
        r, g = use(eng, "广纳良才")
        check(len(g) == 2 and all(is_hero(x) for x in g), f"种子 {sd}：应得 2 张英雄牌：{g}")
        opts = [c[1] for c in eng.calls]
        check(any(len(o) == 5 and all(is_hero(x) for x in o) for o in opts), f"种子 {sd}：应弹 5 个英雄选项：{opts}")


@test("高级人才", desc="随机获得 1 张品阶 = 棋手等级 +1 的英雄牌")
def t_gaoji():
    for lv in (1, 2, 4):
        _hero_cards("高级人才", 1, lambda x, q=lv + 1: quality(x) == q, f"{lv + 1} 阶英雄", lord_level=lv)


@test("图腾仪式", desc="随机获得 1 张与图腾相关的【大河流域】英雄牌")
def t_tuteng_yishi():
    _hero_cards("图腾仪式", 1, lambda x: faction(x) == "大河流域" and "图腾" in card_text(x), "图腾相关的大河英雄")


@test("2阶英雄", desc="获得 2 张 2 阶英雄牌")
def t_hero2():
    _hero_cards("2阶英雄", 2, lambda x: quality(x) == 2, "2 阶英雄")


@test("4阶英雄", desc="获得 2 张 4 阶英雄牌")
def t_hero4():
    _hero_cards("4阶英雄", 2, lambda x: quality(x) == 4, "4 阶英雄")


@test("5阶英雄", desc="获得 2 张 5 阶英雄牌")
def t_hero5():
    _hero_cards("5阶英雄", 2, lambda x: quality(x) == 5, "5 阶英雄")


@test("无阵营征召令", desc="获得 2 张无阵营英雄牌")
def t_wuzhenying():
    _hero_cards("无阵营征召令", 2, lambda x: faction(x) == "无阵营", "无阵营英雄", lord_level=3)


@test("大河流域征召令", desc="获得 2 张【大河流域】英雄牌")
def t_dahe_zhao():
    _hero_cards("大河流域征召令", 2, lambda x: faction(x) == "大河流域", "大河英雄", lord_level=3)


@test("整备高手", desc="获得 2 张具有整备效果的英雄牌")
def t_zhengbei():
    _hero_cards("整备高手", 2, lambda x: hero_has(x, "整备"), "整备英雄", lord_level=4)


@test("登场人物", desc="获得 2 张具有登场效果的英雄牌")
def t_dengchang():
    _hero_cards("登场人物", 2, lambda x: hero_has(x, "登场"), "登场英雄", lord_level=4)


@test("牺牲品", desc="获得 2 张具有牺牲效果的英雄牌")
def t_xisheng():
    _hero_cards("牺牲品", 2, lambda x: hero_has(x, "牺牲"), "牺牲英雄", lord_level=4)


@test("闪现突击者", desc="获得 2 张具有闪现效果的英雄牌")
def t_shanxian():
    _hero_cards("闪现突击者", 2, lambda x: hero_has(x, "闪现"), "闪现英雄", lord_level=4)


@test("开团号令", desc="获得 2 张具有开团效果的英雄牌")
def t_kaituan():
    _hero_cards("开团号令", 2, lambda x: hero_has(x, "开团"), "开团英雄", lord_level=4)


@test("呼朋唤友", desc="指定 1 名英雄，随机获得 1 张同阵营英雄牌")
def t_hupeng():
    _hero_cards("呼朋唤友", 1, lambda x: faction(x) == "逐鹿", "逐鹿英雄（墨子的阵营）",
                setup=lambda e: put(e, "墨子"), lord_level=3)


@test("照影", desc="随机获得 1 张场上英雄牌")
def t_zhaoying():
    def setup(e):
        put(e, "墨子")
        put(e, "扁鹊")
    _hero_cards("照影", 1, lambda x: x in ("墨子", "扁鹊"), "场上英雄（墨子 / 扁鹊）", setup=setup)


@test("乱世", desc="卡牌 3 选 1（登场英雄牌）")
def t_luanshi():
    for sd in SEEDS:
        eng = new_eng(sd, lord_level=4)
        r, g = use(eng, "乱世")
        check(len(g) == 1 and is_hero(g[0]) and hero_has(g[0], "登场"), f"种子 {sd}：应得 1 张登场英雄：{g}")
        opts = [c[1] for c in eng.calls]
        check(any(len(o) == 3 and all(is_hero(x) and hero_has(x, "登场") for x in o) for o in opts),
              f"种子 {sd}：应弹 3 个登场英雄选项：{opts}")


@test("高阶招募战术", desc="转瞬 3 选 1【逐鹿】英雄牌")
def t_gj_zhaomu():
    for sd in SEEDS:
        eng = new_eng(sd, lord_level=4)
        r, g = use(eng, "高阶招募战术")
        check(len(g) == 1 and is_hero(g[0]) and faction(g[0]) == "逐鹿", f"种子 {sd}：应得 1 张逐鹿英雄：{g}")
        opts = [c[1] for c in eng.calls]
        check(any(len(o) == 3 and all(is_hero(x) and faction(x) == "逐鹿" for x in o) for o in opts),
              f"种子 {sd}：应弹 3 个逐鹿英雄选项：{opts}")


@test("求贤", desc="指定 1 名英雄，发现 3 张同阵营英雄牌选 1")
def t_qiuxian():
    for sd in SEEDS:
        eng = new_eng(sd, lord_level=4)
        s = put(eng, "沈梦溪")
        r, g = use(eng, "求贤", s)
        check(len(g) == 1 and is_hero(g[0]) and faction(g[0]) == "河洛", f"种子 {sd}：应得 1 张河洛英雄：{g}")
        opts = [c[1] for c in eng.calls]
        check(any(len(o) == 3 and all(is_hero(x) and faction(x) == "河洛" for x in o) for o in opts),
              f"种子 {sd}：应弹 3 个河洛英雄选项：{opts}")


# ============================================================ 获得其他牌
@test("古币多多", desc="获得 2 张古币")
def t_gubi():
    eng = new_eng()
    _, g = use(eng, "古币多多")
    check(sorted(g) == ["古币", "古币"], f"应得 2 张古币：{g}")


@test("发号施令", desc="随机获得 2 张战术牌")
def t_fahao():
    for sd in SEEDS:
        eng = new_eng(sd)
        _, g = use(eng, "发号施令")
        check(len(g) == 2 and all(x in TACTICS for x in g), f"种子 {sd}：应得 2 张战术牌：{g}")


@test("因材施教", desc="随机获得 1 张场上主要阵营相关的效果牌")
def t_yincai():
    for sd in SEEDS:
        eng = new_eng(sd, lord_level=4)
        for n in ("墨子", "廉颇", "扁鹊", "沈梦溪"):
            put(eng, n)
        _, g = use(eng, "因材施教")
        check(len(g) == 1 and category(g[0]) == "效果牌" and faction(g[0]) == "逐鹿",
              f"种子 {sd}：主要阵营是逐鹿（3 人），应得 1 张逐鹿效果牌：{g} "
              f"（{[(x, category(x), faction(x)) for x in g]}）")


@test("远洋宝藏", desc="从 5 件日落海专属装备中随机获得 1 件，不重复")
def t_yuanyang():
    eng = new_eng()
    got = []
    for _ in range(5):
        _, g = use(eng, "远洋宝藏")
        check(len(g) == 1 and g[0] in SUNSET_EXCLUSIVE, f"每次应得 1 件日落海专属装备：{g}")
        got += g
    check(len(set(got)) == 5, f"5 次应拿齐 5 件不重复：{got}")
    _, g = use(eng, "远洋宝藏")
    check(not g, f"5 件都拿过后不应再得：{g}")


@test("度命之护", desc="获得 1 张贤者的庇护")
def t_duming():
    eng = new_eng()
    _, g = use(eng, "度命之护")
    check(g == ["贤者的庇护"], f"应得贤者的庇护：{g}")


@test("机关秘典", desc="获得 1 件机关匣")
def t_jiguan_mi():
    eng = new_eng()
    _, g = use(eng, "机关秘典")
    check(g == ["机关匣"], f"应得机关匣：{g}")


@test("珍品宝箱", desc="卡牌 3 选 1（特殊装备或天赋装备）")
def t_zhenpin():
    for sd in SEEDS:
        eng = new_eng(sd)
        _, g = use(eng, "珍品宝箱")
        check(len(g) == 1 and category(g[0]) == "装备牌" and quality(g[0]) >= 3,
              f"种子 {sd}：应得 1 件特殊 / 天赋装备（装备牌、非基础 / 四选一）：{g}")
        opts = [c[1] for c in eng.calls]
        check(any(len(o) == 3 and all(category(x) == "装备牌" and quality(x) >= 3 for x in o) for o in opts),
              f"种子 {sd}：应弹 3 个特殊装备选项：{opts}")


@test("精选武装", desc="卡牌 4 选 1（基础装备牌）")
def t_jingxuan():
    for sd in SEEDS:
        eng = new_eng(sd)
        _, g = use(eng, "精选武装")
        check(len(g) == 1 and g[0] in BASE_EQUIPS, f"种子 {sd}：应得 1 件基础装备：{g}")
        opts = [c[1] for c in eng.calls]
        check(any(len(o) == 4 and all(x in BASE_EQUIPS for x in o) for o in opts), f"种子 {sd}：应弹 4 个基础装备选项：{opts}")


@test("军械库", desc="随机获得 3 张基础装备牌")
def t_junxie():
    for sd in SEEDS:
        eng = new_eng(sd)
        _, g = use(eng, "军械库")
        check(len(g) == 3 and all(x in BASE_EQUIPS for x in g), f"种子 {sd}：应得 3 件基础装备：{g}")


@test("神兵天降", desc="获得 4 张 5 阶英雄兑换券")
def t_shenbing():
    eng = new_eng()
    _, g = use(eng, "神兵天降")
    check(g == ["5阶英雄兑换券"] * 4, f"应得 4 张 5 阶英雄兑换券：{g}")


@test("呼风唤雨", desc="随机获得 4 张 5 阶效果牌")
def t_hufeng():
    for sd in SEEDS:
        eng = new_eng(sd)
        _, g = use(eng, "呼风唤雨")
        check(len(g) == 4 and all(category(x) == "效果牌" and quality(x) == 5 for x in g),
              f"种子 {sd}：应得 4 张 5 阶效果牌：{[(x, category(x), quality(x)) for x in g]}")


# ============================================================ 加等级
@test("八方来援", desc="指定英雄 +2，场上每有 1 种阵营额外触发 1 次（逐鹿 + 河洛 = 共 3 次）")
def t_bafang():
    eng = new_eng()
    m = put(eng, "墨子")
    s = put(eng, "沈梦溪")
    eng.use_card("八方来援", m)
    check(m.level == 7 and s.level == 1, f"墨子应 1→7（+2×3），沈梦溪不变：墨子 {m.level} 沈梦溪 {s.level}")


@test("武装之力", desc="所有英雄获得等同于自身装备数量的等级")
def t_wuzhuang_li():
    eng = new_eng()
    a = put(eng, "墨子", equips=["无尽巨剑", "制裁之剑"])
    b = put(eng, "扁鹊")
    c = put(eng, "沈梦溪", equips=["冥王"])
    eng.use_card("武装之力")
    check((a.level, b.level, c.level) == (3, 1, 2), f"应为 3/1/2：{a.level}/{b.level}/{c.level}")


@test("换帅", desc="指定【三分】英雄与最近的【三分】英雄交换等级")
def t_huanshuai():
    eng = new_eng()
    a = put(eng, "马超", level=10, cell=0)
    b = put(eng, "诸葛亮", level=3, cell=1)       # 紧挨着
    c = put(eng, "刘禅", level=7, cell=6)         # 远
    r = eng.use_card("换帅", a)
    check(r is not False and (a.level, b.level, c.level) == (3, 10, 7),
          f"马超 / 诸葛亮应交换、刘禅不变：{a.level}/{b.level}/{c.level}")
    m = put(eng, "墨子", level=5)
    r = eng.use_card("换帅", m)
    check(r is False and m.level == 5, f"非三分目标不合法：返回 {r} 墨子 {m.level}")


@test("助力核心", desc="核心 +2，场上每有 1 名【日落海】英雄重复 1 次（2 名日落海 = +6）")
def t_zhuli():
    eng = new_eng()
    put(eng, "艾琳")
    put(eng, "姬小满")
    put(eng, "墨子")
    c0 = core(eng)
    eng.use_card("助力核心")
    d = core(eng) - c0
    check(d == 6, f"核心应 +6（2×(1+2)），实际 +{d}（字段 {_core_items(eng)}）")


@test("跳级", "助力核心", desc="使 1 名英雄夺取核心至多 5 级（核心 6 → 拿 5；核心 2 → 拿 2）")
def t_tiaoji():
    for setup, label in ((("艾琳", "姬小满"), "核心多于 5"), ((), "核心少于 5")):
        eng = new_eng()
        for n in setup:
            put(eng, n)
        m = put(eng, "墨子", level=1)
        eng.use_card("助力核心")
        c0 = core(eng)
        check(c0 > 0, f"{label}：前置，助力核心没提升核心（字段 {_core_items(eng)}）")
        eng.use_card("跳级", m)
        want = min(5, c0)
        check(m.level - 1 == want, f"{label}：核心 {c0:g}，墨子应 +{want:g}，实际 +{m.level - 1:g}")
        check(c0 - core(eng) == want, f"{label}：核心应减 {want:g}，实际 {c0:g}→{core(eng):g}")


@test("高阶演练战术", desc="转瞬 所有英雄等级 +2")
def t_gj_yanlian():
    eng = new_eng()
    a = put(eng, "墨子")
    b = put(eng, "扁鹊", level=4)
    eng.use_card("高阶演练战术")
    check((a.level, b.level) == (3, 6), f"应各 +2：{a.level}/{b.level}")


@test("高阶特训战术", desc="转瞬 指定 1 名英雄等级 +8")
def t_gj_texun():
    eng = new_eng()
    a = put(eng, "墨子")
    b = put(eng, "扁鹊")
    eng.use_card("高阶特训战术", a)
    check((a.level, b.level) == (9, 1), f"墨子 +8、扁鹊不变：{a.level}/{b.level}")


@test("阵营祝福", desc="各阵营中随机 1 名英雄等级 +4")
def t_zhenying_zhufu():
    for sd in SEEDS:
        eng = new_eng(sd)
        a = put(eng, "墨子")
        b = put(eng, "廉颇")
        c = put(eng, "沈梦溪")
        eng.use_card("阵营祝福")
        zl = sorted([a.level - 1, b.level - 1])
        check(zl == [0, 4], f"种子 {sd}：逐鹿两人中恰 1 人 +4：墨子 {a.level} 廉颇 {b.level}")
        check(c.level == 5, f"种子 {sd}：河洛只有沈梦溪，应 +4：{c.level}")


@test("扶倾济弱", desc="场上等级最低（临时等级除外）的英雄提升 9 级")
def t_fuqing():
    eng = new_eng()
    a = put(eng, "墨子", level=5)
    b = put(eng, "扁鹊", level=3)
    b.temp = 10                               # 算上临时等级是 13，但不算临时等级它最低
    eng.use_card("扶倾济弱")
    check((a.level, b.level) == (5, 12), f"扁鹊（真实 3 级）应 +9：墨子 {a.level} 扁鹊 {b.level}")


@test("战前宣誓", desc="指定英雄获得 自身品阶×3 的临时等级")
def t_zhanqian():
    eng = new_eng()
    b = put(eng, "扁鹊", level=5)            # 4 阶
    eng.use_card("战前宣誓", b)
    check(b.temp == 12 and b.level == 5, f"扁鹊应临时 +12、永久不变：临时 {b.temp} 永久 {b.level}")


@test("逍遥自在", desc="所有无阵营英雄获得 2×场上无阵营英雄数 的等级")
def t_xiaoyao():
    eng = new_eng()
    a = put(eng, "苍")
    b = put(eng, "海月")
    c = put(eng, "墨子")
    eng.use_card("逍遥自在")
    check((a.level, b.level, c.level) == (5, 5, 1), f"两名无阵营各 +4、墨子不变：{a.level}/{b.level}/{c.level}")


@test("图腾祝福", desc="随机 1 名【大河】英雄 +5，场上每有 1 个图腾额外触发 1 次（1 个图腾 = 共 +10）")
def t_tuteng_zhufu():
    for sd in SEEDS:
        eng = new_eng(sd)
        a = put(eng, "敖隐")
        b = put(eng, "瑶")
        c = put(eng, "墨子")
        put(eng, "因缘图腾", kind="summon", pop=False)
        eng.use_card("图腾祝福")
        ga, gb = a.level - 1, b.level - 1
        check(ga + gb == 10 and ga % 5 == 0 and gb % 5 == 0, f"种子 {sd}：大河英雄合计应 +10（每次 5）：敖隐 +{ga} 瑶 +{gb}")
        check(c.level == 1, f"种子 {sd}：墨子不是大河，不该加：{c.level}")


@test("众星闪耀", desc="全体己方英雄等级 +10")
def t_zhongxing():
    eng = new_eng()
    a = put(eng, "墨子")
    b = put(eng, "扁鹊", level=3)
    eng.gain_card("众星闪耀")
    eng.use_card("众星闪耀")
    check((a.level, b.level) == (11, 13), f"应各 +10：{a.level}/{b.level}")


@test("众星闪耀", desc="每持有一回合增强 5 级（持有 1 回合后打出 = +15）")
def t_zhongxing_hold():
    eng = new_eng()
    a = put(eng, "墨子")
    eng.gain_card("众星闪耀")
    eng.round_end()
    eng.round_start()
    lv0 = a.level
    eng.use_card("众星闪耀")
    check(a.level - lv0 == 15, f"持有 1 回合后应 +15，实际 +{a.level - lv0:g}")


@test("破境之证", desc="指定英雄立即 +20，之后每回合 +10")
def t_pojing():
    eng = new_eng()
    a = put(eng, "墨子")
    b = put(eng, "扁鹊")
    eng.use_card("破境之证", a)
    check(a.level == 21, f"立即 +20：{a.level}")
    for i in (1, 2):
        eng.battle()
        eng.round_end()
        eng.round_start()
        check(a.level == 21 + 10 * i, f"第 {i} 个后续回合后应为 {21 + 10 * i}：{a.level}")
    check(b.level == 1, f"扁鹊不该加：{b.level}")


# ============================================================ 触发别人的效果
@test("唤醒", desc="指定 1 名英雄，触发其登场效果（武则天登场：得精致古币）")
def t_huanxing():
    eng = new_eng()
    w = put(eng, "武则天")
    _, g = use(eng, "唤醒", w)
    check(g == ["精致古币"], f"应触发武则天登场得 1 张精致古币：{g}")


@test("集体唤醒", desc="立即触发所有英雄的登场效果")
def t_jiti():
    eng = new_eng()
    put(eng, "武则天")
    put(eng, "武则天")
    put(eng, "墨子")
    _, g = use(eng, "集体唤醒")
    check(g.count("精致古币") == 2, f"两名武则天各触发一次登场，应得 2 张精致古币：{g}")


@test("全军动员", desc="立即触发所有英雄的整备效果")
def t_quanjun():
    eng = new_eng()
    put(eng, "西施")
    c = put(eng, "程咬金")
    _, g = use(eng, "全军动员")
    check(g.count("推演战术") == 1, f"西施整备应得推演战术：{g}")
    check(c.level == 3, f"程咬金整备（自用古币 +2）：{c.level}")


# ============================================================ 装备
@test("武装重构", desc="摧毁目标所有装备，随机获得 数量+1 的基础装备")
def t_chonggou():
    for sd in SEEDS:
        eng = new_eng(sd)
        m = put(eng, "墨子", equips=["无尽巨剑", "制裁之剑"])
        _, g = use(eng, "武装重构", m)
        check(not m.equips, f"种子 {sd}：装备应被摧毁：{m.equips}")
        check(len(g) == 3 and all(x in BASE_EQUIPS for x in g), f"种子 {sd}：应得 3 件基础装备（被摧毁的不回手）：{g}")


@test("武装分解", desc="摧毁目标所有装备：1 件 → 4 能量；2 件 → 妙音；3 件 → 珍品宝箱")
def t_fenjie():
    eng = new_eng(energy=0)
    m = put(eng, "墨子", equips=["无尽巨剑"])
    _, g = use(eng, "武装分解", m)
    check(not m.equips and eng.s.energy == 4, f"1 件：装备清空、能量 +4：装备 {m.equips} 能量 {eng.s.energy}")
    check("无尽巨剑" not in g, f"1 件：被摧毁的装备不该回手：{g}")

    eng = new_eng(energy=0)
    m = put(eng, "墨子", equips=["无尽巨剑", "制裁之剑"])
    _, g = use(eng, "武装分解", m)
    check(not m.equips and "妙音" in g, f"2 件：装备清空、得妙音：装备 {m.equips} 新增 {g}")
    check(not ({"无尽巨剑", "制裁之剑"} & set(g)), f"2 件：被摧毁的装备不该回手：{g}")

    eng = new_eng(energy=0)
    m = put(eng, "墨子", equips=["无尽巨剑", "制裁之剑", "冥王"])
    _, g = use(eng, "武装分解", m)
    check(not m.equips and "珍品宝箱" in g, f"3 件：装备清空、得珍品宝箱：装备 {m.equips} 新增 {g}")


@test("贤者的庇护", desc="穿戴者每场战斗获得 1 次复生（首次被击败后复生）")
def t_xianzhe():
    eng = new_eng()
    s = put(eng, "沈梦溪", equips=["贤者的庇护"])
    c = impl("贤者的庇护")
    if hasattr(c, "on_equip"):
        c.on_equip(eng, s)
    for i in (1, 2):
        eng.fire(s, "open")
        eng.fire(s, "battle_start")
        check(s.counters.get("revive", 0) == 1, f"第 {i} 场：开战后应有 1 次复生：{s.counters}")
        s.counters.pop("revive", None)        # 模拟战斗结束（引擎在 battle() 末尾清掉复生）


@test("机关匣", desc="穿戴的逐鹿英雄：战斗开始时召唤 2 个 等级 = 我方最高英雄一半 的故障机器人")
def t_jiguanxia():
    eng = new_eng()
    m = put(eng, "墨子", equips=["机关匣"])
    put(eng, "扁鹊", level=20)
    eng.fire(m, "open")
    eng.fire(m, "battle_start")
    bots = [u for u in eng.s.units if u.name == "故障机器人"]
    check(len(bots) == 2, f"应召唤 2 个故障机器人：{[(u.name, u.level) for u in eng.s.units]}")
    check(all(u.level == 10 for u in bots), f"等级应为 20 的一半 = 10：{[u.level for u in bots]}")
    check(all(not u.pop for u in bots), "召唤物不占人口")


# ============================================================ 补充（复核时加的边界）
@test("妙音", desc="觉醒进度凑满 3 = 觉醒；已觉醒的不能再加")
def t_miaoyin_full():
    eng = new_eng()
    b = put(eng, "扁鹊", awaken=2)
    r = eng.use_card("妙音", b)
    check(r is not False and b.awakened, f"进度 2 → 3 应觉醒：返回 {r} 进度 {b.awaken}")
    r = eng.use_card("妙音", b)
    check(r is False and b.awaken == 3, f"已觉醒的不能再用：返回 {r} 进度 {b.awaken}")


@test("弦音", desc="已觉醒的 1-3 阶英雄也不能再用")
def t_xianyin_awakened():
    eng = new_eng()
    m = put(eng, "墨子", awaken=3)
    r = eng.use_card("弦音", m)
    check(r is False and m.awaken == 3, f"已觉醒墨子不能再加进度：返回 {r} 进度 {m.awaken}")


@test("鲁班七号", desc="每 3 次合成各给 1 张（6 次 = 2 张），别人的合成也算")
def t_luban7_six():
    eng = new_eng()
    put(eng, "鲁班七号")
    a = put(eng, "墨子")
    b = put(eng, "扁鹊")
    for u in (a, a, b, b, a, b):
        eng.merge(u)
    t = [x for x in eng.s.hand if x in TACTICS]
    check(len(t) == 2, f"6 次合成应得 2 张战术牌：{eng.s.hand}")


@test("移花接木", desc="下场战斗：随机 1 名英雄获得临时等级（敌方最高一半），永久等级不变，战斗后清零")
def t_yihua():
    for sd in SEEDS:
        eng = new_eng(sd, round=8)
        us = [put(eng, n, level=5) for n in ("墨子", "扁鹊", "沈梦溪")]
        r = eng.use_card("移花接木")
        check(r is not False, f"种子 {sd}：使用失败")
        got = [u for u in us if u.temp > 0]
        check(len(got) == 1, f"种子 {sd}：应恰好 1 名英雄得到临时等级：{[(u.name, u.temp) for u in us]}")
        check(all(u.level == 5 for u in us), f"种子 {sd}：永久等级不该变：{[u.level for u in us]}")
        eng.battle()
        check(all(u.temp == 0 for u in us), f"种子 {sd}：战斗后临时等级应清零：{[u.temp for u in us]}")


# ============================================================ 运行
def main() -> int:
    from agent.sim.engine import REGISTRY
    impl("古币")                                         # 触发注册
    ok = fail = 0
    for cards, desc, f in TESTS:
        tag = "/".join(cards)
        missing = [c for c in cards if c not in REGISTRY]
        if missing:
            fail += 1
            print(f"FAIL [{tag}] {desc} —— 未实现：{'、'.join(missing)}")
            continue
        try:
            f()
            ok += 1
            print(f"OK   [{tag}] {desc}")
        except Fail as e:
            fail += 1
            print(f"FAIL [{tag}] {desc} —— {e}")
        except Exception as e:                           # noqa: BLE001
            fail += 1
            tb = traceback.extract_tb(e.__traceback__)[-1]
            print(f"FAIL [{tag}] {desc} —— 异常 {type(e).__name__}: {e}（{Path(tb.filename).name}:{tb.lineno}）")
    print()
    print("跳过（只有战斗效果 / 无从观测）：")
    for n, why in SKIPPED.items():
        print(f"  - {n}：{why}")
    print()
    print(f"通过 {ok} / 失败 {fail}（共 {ok + fail} 条）")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
