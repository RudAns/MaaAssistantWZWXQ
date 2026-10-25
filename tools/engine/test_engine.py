"""推演器测试：构造局面单元测试（手算期望）+ 真实对局回放误差 + 设计考题 + 性能。

  .venv/Scripts/python.exe -X utf8 tools/engine/test_engine.py              # 全部
  .venv/Scripts/python.exe -X utf8 tools/engine/test_engine.py --unit       # 只跑单元测试
  .venv/Scripts/python.exe -X utf8 tools/engine/test_engine.py --replay 5   # 回放最近 5 局

不连模拟器、不用显卡。
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.engine import board as B  # noqa: E402
from agent.engine import odds  # noqa: E402
from agent.engine.effects import compile_formula, db, eval_formula  # noqa: E402
from agent.engine.options import Action, do, explain, score_options  # noqa: E402
from agent.engine.project import PARAMS, Sim, base_aps, enemy_power, project  # noqa: E402
from agent.engine.state import Position, from_point, make_unit  # noqa: E402

FAILS: list = []
PASSES = [0]


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    if ok:
        PASSES[0] += 1
    else:
        FAILS.append(f"{name}: got {got!r} want {want!r}")
    print(f"  {'OK ' if ok else 'BAD'} {name}: {got!r} (期望 {want!r})")


def pos(units, **kw):
    """units: [(名字, 等级, 格子, {附加})]"""
    us = []
    for t in units:
        n, lv, cell = t[0], t[1], t[2]
        ex = t[3] if len(t) > 3 else {}
        u = make_unit(n, lv, cell, ex.get("awaken", 0), ex.get("equips"), awakened=ex.get("awakened", False))
        us.append(u)
    kw.setdefault("round", 5)
    kw.setdefault("lord_level", 3)
    kw.setdefault("pop_cap", 7)
    return Position(units=us, **kw)


def lv(p, n):
    return p.unit(n).level


# 格子：row3 = (300,569)(433,569)(555,569)(681,571)(786,569)(917,570)(1027,572)；row0 远端 (918,368)
A, Bc, C, D, FAR, FAR2 = (300, 569), (433, 569), (555, 569), (681, 571), (918, 368), (319, 375)


def unit_tests():
    print("== 公式求值")
    ctx = {"ally_units": 7, "enemy_units": 5, "lord_level": 4, "level": 50,
           "faction_count": lambda f: {"河洛": 3}.get(f, 0)}
    for f, want in [("2*max(0,ally_units-enemy_units)", 4), ("faction_count:河洛", 3), ("lord_level*7", 28),
                    ("max(3,round_?(0.08*level))", 4), ("min(level*1%,50%)", None)]:
        from agent.engine.normalize import _pct
        f2 = _pct(f) if "%" in f else f
        code = compile_formula(str(f2))
        got = eval_formula(code, ctx) if code is not None else None
        if want is None:
            check(f"公式 {f}", round(got, 4), 0.5)
        else:
            check(f"公式 {f}", got, want)
    check("未知变量 → None", eval_formula(compile_formula("foo_bar*2"), ctx), None)

    print("== 古币 / 最近同派系（整备）：李元芳 → 相邻的守约 +2")
    p = pos([("李元芳", 5, A), ("百里守约", 10, Bc), ("苏烈", 3, FAR)])
    Sim(p).round_start()
    check("守约 +2", lv(p, "百里守约"), 12)
    check("来源分解", p.unit("百里守约").src.get("李元芳"), 2)

    print("== 古币放大 coin_amp：武则天在场 → 每张 +1")
    p = pos([("李元芳", 5, A), ("百里守约", 10, Bc), ("武则天", 3, (1027, 572))])
    Sim(p).round_start()
    check("守约 +3", lv(p, "百里守约"), 13)

    print("== 程咬金 整备：对自身用 1 张古币")
    p = pos([("程咬金", 4, A)])
    Sim(p).round_start()
    check("程咬金 +2", lv(p, "程咬金"), 6)

    print("== 上官婉儿：随机 3 名河洛各 1 古币 → 总量 6（4 名河洛时每人期望 1.5）")
    p = pos([("上官婉儿", 1, A), ("百里守约", 10, Bc), ("李元芳", 1, FAR), ("苏烈", 1, FAR2)])
    p.units = [u for u in p.units if u.name != "李元芳"] + [make_unit("狄仁杰", 1, (529, 372))]
    before = sum(u.level for u in p.units)
    Sim(p).round_start()
    check("总量 +6", sum(u.level for u in p.units) - before, 6)

    print("== 反应器 react:prepare|open：韩信（每次整备 +2；觉醒 +4）")
    p = pos([("李元芳", 5, A), ("百里守约", 10, Bc), ("韩信", 8, FAR)])
    Sim(p).round_start()
    check("韩信 +2", lv(p, "韩信"), 10)
    p = pos([("李元芳", 5, A), ("百里守约", 10, Bc), ("韩信", 8, FAR, {"awakened": True})])
    Sim(p).round_start()
    check("觉醒韩信 +4", lv(p, "韩信"), 12)

    print("== 触发器 trigger：临阵磨枪 → 李元芳 额外整备 1 次（守约 +2，韩信 +2）")
    p = pos([("李元芳", 5, A), ("百里守约", 10, Bc), ("韩信", 8, FAR)], hand=["临阵磨枪"])
    s = Sim(p)
    do(p, Action("use", "", {"name": "临阵磨枪", "target": "李元芳"}), s)
    check("守约 +2", lv(p, "百里守约"), 12)
    check("韩信 +2", lv(p, "韩信"), 10)
    check("临阵磨枪 不能给没有整备条目的守约",
          do(pos([("百里守约", 10, Bc)], hand=["临阵磨枪"]), Action("use", "", {"name": "临阵磨枪", "target": "百里守约"}),
             Sim(Position())), False)

    print("== 转换器：雅典娜 开团 → 日落海英雄整备（亚连：装备最多者 +2、核心 +1）")
    p = pos([("雅典娜", 3, A), ("亚连", 5, Bc, {"equips": ["?", "?"]})], carry="亚连")
    s = Sim(p)
    for u in list(p.on_board()):
        s.fire(u, "open")
    check("亚连 +2（被雅典娜转成整备）", lv(p, "亚连"), 7)
    check("核心 +1", p.core, 1)

    print("== 牺牲 sacrifice：苏烈 +2 × 阵亡概率")
    p = pos([("苏烈", 4, FAR2), ("百里守约", 10, Bc)])
    s = Sim(p)
    rec = s.run_battle()
    w = rec.win_p
    pd = ((1 - w) * PARAMS["death_lose"] + w * PARAMS["death_win"]) * PARAMS["front_mult"]
    check("苏烈 = 4 + 2·p阵亡", round(lv(p, "苏烈"), 6), round(4 + 2 * min(0.98, pd), 6))

    print("== 临时转永久：明世隐 最近友军 temp 5 → 转 2；登场后额度 +2 → 转 4")
    p = pos([("明世隐", 1, A), ("百里守约", 10, Bc)])
    p.unit("百里守约").temp = 5
    Sim(p).run_battle()
    check("守约 +2", lv(p, "百里守约"), 12)
    p = pos([("百里守约", 10, Bc)])
    s = Sim(p)
    s.deploy("明世隐", cell=A)
    p.unit("百里守约").temp = 5
    s.run_battle()
    check("登场后 守约 +4", lv(p, "百里守约"), 14)

    print("== 普攻计数 attack_n：少司缘 → 因缘图腾，全队每合计 25 次普攻 +1（动态攻速）")
    p = pos([("百里守约", 10, Bc), ("牛魔", 3, FAR2)], round=3)
    s = Sim(p)
    s.deploy("少司缘", cell=C)
    names = [u.name for u in p.units]
    check("图腾随少司缘上场", "因缘图腾" in names, True)
    lv0 = {u.name: u.level for u in p.heroes()}
    s.run_battle()
    tot = sum(s.battle["attacks"].values())
    for n in ("百里守约", "牛魔"):
        check(f"{n} +合计普攻/25", round(lv(p, n) - lv0[n], 6), round(tot / 25, 6))

    print("== 始祖熊灵（天赋送出、可当主C）：每多 1 个单位 +2；开团 +20% 攻速；刘邦摧毁图腾是反协同")
    def bear_board(extra):
        p = pos([("百里守约", 10, Bc), ("牛魔", 3, FAR2), ("鬼谷子", 2, D), ("梦奇", 1, (718, 371)),
                 ("盾山", 1, (818, 374))] + extra, round=1, lord_level=1)
        s = Sim(p)
        do(p, Action("talent", "", {"name": "始祖熊灵", "carry": "始祖熊灵"}), s)
        s.deploy_summons(p.unit("鬼谷子"))
        return p, s
    p, s = bear_board([])
    en = enemy_power(1)[1]
    ally = len(p.on_board())
    s.run_battle()
    check("我方 7 单位（5 英雄 + 熊灵 + 先知图腾）vs 敌方 5", (ally, en), (7, 5))
    check("熊灵永久 +2·(7-5)=+4", lv(p, "始祖熊灵"), 5)
    check("熊灵是主C", p.carry, "始祖熊灵")
    p2, s2 = bear_board([("刘邦", 1, (529, 372)), ("李元芳", 1, (617, 372))])
    p3, s3 = bear_board([("苏烈", 1, (529, 372)), ("李元芳", 1, (617, 372))])   # 同样多 2 人，但不摧毁图腾
    s2.run_battle()
    s3.run_battle()
    check("刘邦摧毁图腾 → 熊灵少 2 级", lv(p3, "始祖熊灵") - lv(p2, "始祖熊灵"), 2)

    print("== 开团叠攻速 → 动态普攻次数")
    p = pos([("百里守约", 10, Bc), ("先知图腾", 1, D)], round=1)
    s = Sim(p)
    do(p, Action("talent", "", {"name": "始祖熊灵", "carry": "始祖熊灵"}), s)
    bear = p.unit("始祖熊灵")
    s.removed_totems = 0
    s.battle = {"enemy_n": 5, "attacks": {}}
    for u in list(p.on_board()):
        s.fire(u, "open")
    check("先知图腾开团 1 次 → 熊灵攻速 +0.2", round(bear.aspd, 6), 0.2)

    print("== 夺取 steal_level：孙悟空上场，相邻唯一友军 15 级 → 夺 10")
    p = pos([("百里守约", 15, Bc)])
    Sim(p).deploy("孙悟空", cell=A)
    check("守约 15→5", lv(p, "百里守约"), 5)
    check("孙悟空 1→11", lv(p, "孙悟空"), 11)

    print("== 装备 round_start：学识宝石 50 级 → +4（8%，至少 3）")
    p = pos([("百里守约", 50, Bc, {"equips": ["学识宝石"]})])
    Sim(p).round_start()
    check("守约 +4", lv(p, "百里守约"), 54)

    print("== 收入 / 经验 / 自动升级 / 弈星（棋手升级时经验 +2）")
    p = pos([("弈星", 1, A)], round=2, lord_level=1, xp=6, energy=0)
    s = Sim(p)
    s.round_end()
    check("第 3 回合收入 6", p.energy, 6)
    check("经验满 7 自动升 2 级", p.lord_level, 2)
    check("弈星 +2 经验（余 0 + 2）", p.xp, 2)
    p = pos([], round=3, lord_level=2, xp=0, energy=0)
    Sim(p).round_end()
    check("第 3 回合战后拍卖：经验 +3", p.xp, 3)
    check("第 4 回合收入 8", p.energy, 8)

    print("== 绑定装备：干将莫邪自带双剑·雄，卖人不退回、不能转移")
    p = pos([("百里守约", 10, Bc)], carry="百里守约")
    s = Sim(p)
    s.deploy("干将莫邪", cell=A)
    gj = p.unit("干将莫邪")
    check("自带 双剑·雄 且绑定", gj.bound, ["双剑·雄"])
    check("绑定装备 move_equip 不合法",
          do(p, Action("move_equip", "", {"from": "干将莫邪", "item": "双剑·雄", "to": "百里守约"}), s), False)
    gj.equips.append("速击之枪")
    check("可转移装备 move_equip 合法",
          do(p, Action("move_equip", "", {"from": "干将莫邪", "item": "速击之枪", "to": "百里守约"}), s), True)
    do(p, Action("sell", "", {"sell": "干将莫邪"}), s)
    check("卖掉后 双剑·雄 没回手", "双剑·雄" in p.hand, False)

    print("== 换主C：红叶染锋按卡面把旧宿主 50% 等级搬给新主C，可转移装备跟着走，枫叶舞换绑卖旧宿主")
    p = pos([("百里守约", 40, Bc, {"equips": ["速击之枪", "无尽巨剑"]}), ("嬴政", 1, A)], carry="百里守约",
            hand=["红叶染锋", "枫叶舞"], lord_cards=["枫叶舞"], lord_bound="百里守约")
    s = Sim(p)
    do(p, Action("carry", "", {"name": "嬴政"}), s)
    check("嬴政 1 + 20", lv(p, "嬴政"), 21)
    check("装备跟过去", sorted(p.unit("嬴政").equips), sorted(["速击之枪", "无尽巨剑"]))
    check("旧宿主被卖", p.unit("百里守约") is None, True)
    check("枫叶舞绑到嬴政", p.lord_bound, "嬴政")

    print("== 商店概率：2 级 1 阶英雄一次看店命中率（手算）")
    q1 = 0.5 * 14 / (9 * 14)
    check("p_view", round(odds.p_view("苏烈", 2), 6), round(1 - (1 - q1) ** 4, 6))
    d = odds.eta_distribution(["百里守约"], (), 3, 3)
    check("ETA 单调", all(a <= b + 1e-12 for a, b in zip(list(d["p_by_round"].values()),
                                                         list(d["p_by_round"].values())[1:])), True)


# ------------------------------------------------------------------ 设计考题（构造局面）
def design_exams():
    print("\n== 设计考题（data/engine/exam_v1.jsonl 的 design 条目，构造局面，看 argmax ΔV）")
    res = []

    def top(p, **kw):
        s = score_options(p, **kw)
        return s

    # design:bear_opening_talent —— 开局三选一里有始祖熊灵
    p = pos([("苏烈", 1, FAR2), ("百里守约", 1, Bc)], round=2, lord_level=2, energy=2, pop_cap=5,
            lord_cards=["枫叶舞"], hand=["枫叶舞"])
    s = top(p, talents=["始祖熊灵", "三十而达", "强者自强"])
    tal = [x for x in s if x.action.kind == "talent"]
    res.append(("design:bear_opening_talent", tal[0].action.args["name"] == "始祖熊灵", explain(tal, 3)))

    # design:bear_totem_loop —— 有熊灵、开团件 ≥2，商店出少司缘
    p = pos([("鬼谷子", 3, D), ("先知图腾", 1, (617, 496)), ("雅典娜", 3, FAR2), ("太乙真人", 2, (529, 372))],
            round=6, lord_level=3, energy=6, pop_cap=6, shop=["少司缘", "苏烈", "李元芳", "海诺"])
    s0 = Sim(p)
    do(p, Action("talent", "", {"name": "始祖熊灵", "carry": "始祖熊灵"}), s0)
    p.talents = ["始祖熊灵"]
    s = top(p)
    buys = [x for x in s if x.action.kind == "buy"]
    res.append(("design:bear_totem_loop", buys[0].action.args["name"] == "少司缘", explain(buys, 4)))

    # design:bear_as_carry —— 熊灵 vs 普通 2 阶输出（等级不高）
    p = pos([("鬼谷子", 3, D), ("先知图腾", 1, (617, 496)), ("少司缘", 3, C), ("因缘图腾", 1, (499, 500)),
             ("雅典娜", 3, FAR2), ("太乙真人", 2, (529, 372)), ("虞姬", 6, A)], round=6, lord_level=4, pop_cap=6,
            carry="虞姬", hand=["枫叶舞"], lord_cards=["枫叶舞"], lord_bound="虞姬")
    s0 = Sim(p)
    do(p, Action("talent", "", {"name": "始祖熊灵"}), s0)
    p.unit("始祖熊灵").level = 12
    s = top(p)
    cs = [x for x in s if x.action.kind == "carry"]
    res.append(("design:bear_as_carry", bool(cs) and cs[0].action.args["name"] == "始祖熊灵" and cs[0].dV > 0,
                explain(cs, 3)))

    # design:final_carry_can_switch —— 第 9 回合 守约 45 级 3 件 + 红叶染锋，场上觉醒嬴政已经拿到一件强力法术装（噬神之书）
    # （2026-10-14 修正：原题嬴政身上什么都没有，换过去只能接守约的 3 件物理装 —— 法师穿物理装不算战力，推演判"不换"是对的；
    #  题目前提是"有强力的提升等级或法术装备"）
    p = pos([("百里守约", 45, Bc, {"equips": ["速击之枪", "无尽巨剑", "破晓"]}), ("苏烈", 20, FAR2),
             ("钟馗", 15, (529, 372)), ("明世隐", 10, C), ("嬴政", 8, A, {"awakened": True, "equips": ["噬神之书"]})], round=9, lord_level=5,
            pop_cap=7, carry="百里守约", hand=["红叶染锋", "枫叶舞"], lord_cards=["枫叶舞"], lord_bound="百里守约")
    s = top(p)
    cs = [x for x in s if x.action.kind == "carry" and x.action.args["name"] == "嬴政"]
    res.append(("design:final_carry_can_switch", bool(cs) and cs[0].dV > 0, explain(cs, 2)))

    # design:temp_carry_keep —— 第 7 回合 干将觉醒 60 级满装；守约刚到 1 级，无搬等级牌
    # （2026-10-14 修正：干将是法师，原题给他穿 3 件物理装 = 0 件有效装备，换给物理的守约反而对口，推演判"换"是对的；改成法术装）
    p = pos([("干将莫邪", 60, A, {"awakened": True, "equips": ["贤者之书", "虚无法杖", "噬神之书"]}),
             ("百里守约", 1, Bc), ("苏烈", 10, FAR2), ("钟馗", 8, (529, 372))], round=7, lord_level=4, pop_cap=6,
            carry="干将莫邪", hand=["枫叶舞"], lord_cards=["枫叶舞"], lord_bound="干将莫邪")
    s = top(p)
    cs = [x for x in s if x.action.kind == "carry" and x.action.args["name"] == "百里守约"]
    res.append(("design:temp_carry_keep", bool(cs) and cs[0].dV < 0, explain(cs, 2)))

    # design:yingzheng_early_switch —— 第 5 回合临时主C 干将 12 级；嬴政 3 级 + 炽热支配者，逐鹿核心 芈月/西施/白起 在场
    p = pos([("干将莫邪", 12, Bc, {"equips": ["贤者之书"]}), ("嬴政", 3, A, {"equips": ["炽热支配者"]}),
             ("芈月", 4, (529, 372)), ("西施", 3, C), ("白起", 3, (617, 372))], round=5, lord_level=4,
            pop_cap=6, carry="干将莫邪", hand=["枫叶舞"], lord_cards=["枫叶舞"], lord_bound="干将莫邪")
    s = top(p)
    cs = [x for x in s if x.action.kind == "carry" and x.action.args["name"] == "嬴政"]
    res.append(("design:yingzheng_early_switch", bool(cs) and cs[0].dV > 0, explain(cs, 2)))

    for rid, ok, txt in res:
        print(f"  {'OK ' if ok else 'BAD'} {rid}\n    " + txt.replace("\n", "\n    "))
    return res


def point_exams():
    """exam_v1.jsonl 里带 point_id 的天赋题：从 laya_points 取局面与三个天赋，看 argmax ΔV 是否在 good 里。"""
    print("\n== 考题（真实局面·天赋题）")
    out = []
    for line in (ROOT / "data" / "engine" / "exam_v1.jsonl").open(encoding="utf-8"):
        ex = json.loads(line)
        if ex.get("kind") != "talent" or not ex.get("point_id"):
            continue
        f = ROOT / "data" / "logs" / ex["run"] / "laya_points.jsonl"
        pt = None
        for l2 in f.open(encoding="utf-8") if f.exists() else []:
            r = json.loads(l2)
            if r.get("id") == ex["point_id"]:
                pt = r
                break
        if not pt:
            print("  缺局面", ex["id"])
            continue
        names = [t[0] if isinstance(t, (list, tuple)) else t for t in (pt.get("raw") or {}).get("talents") or []]
        p = from_point(pt)
        sc = [x for x in score_options(p, talents=names) if x.action.kind == "talent"]
        pick = names.index(sc[0].action.args["name"]) if sc else -1
        ok = f"opt{pick}" in ex["good"]
        out.append(ok)
        print(f"  {'OK ' if ok else 'BAD'} {ex['id']}  选 opt{pick}，正确 {ex['good']}  |  " +
              "  ".join(f"{x.action.args['name']} {x.dV:+.1f}" for x in sc))
    return out



def choice_exams():
    """exam_v1.jsonl 里带 point_id 的 swap / shop / auction 题：题面选项按 raw 映射成引擎动作，看 argmax ΔV 落在 good（且不在 bad）。"""
    from agent.engine.options import Action
    print("\n== 考题（真实局面·换人 / 商店 / 拍卖）")
    out = []
    for line in (ROOT / "data" / "engine" / "exam_v1.jsonl").open(encoding="utf-8"):
        ex = json.loads(line)
        if ex.get("kind") not in ("swap", "shop", "auction") or not ex.get("point_id"):
            continue
        f = ROOT / "data" / "logs" / ex["run"] / "laya_points.jsonl"
        pt = next((r for r in (json.loads(x) for x in f.open(encoding="utf-8")) if r.get("id") == ex["point_id"]), None)             if f.exists() else None
        if not pt:
            print("  缺局面", ex["id"])
            continue
        raw, p = pt.get("raw") or {}, from_point(pt)
        acts = {}
        for k in pt.get("options") or []:
            if ex["kind"] == "swap":
                inc = raw.get("incoming") or ""
                if inc and inc not in p.hand:
                    p.hand.append(inc)
                nm = (raw.get("names") or {}).get(k)
                acts[k] = Action("stop", "不换") if k == "bench" else Action("swap", f"卖 {nm} 换上 {inc}", {"sell": nm, "name": inc})
            elif ex["kind"] == "shop":
                if k.startswith("buy"):
                    it = next((x for x in raw.get("shop") or [] if f"buy{x[0]}" == k), None)
                    if it:
                        if it[1] not in p.shop:
                            p.shop.append(it[1])
                        acts[k] = Action("buy", f"买 {it[1]}", {"name": it[1], "cost": it[2]})
                elif k == "refresh":
                    acts[k] = Action("refresh", "刷新")
                else:
                    acts[k] = Action("stop", k)
            else:
                nm = (raw.get("names") or {}).get(k)
                acts[k] = Action("stop", "不拍") if k == "pass" else Action("auction", f"拍 {nm}", {"name": nm, "cost": 0})
        sc = score_options(p, actions=list(acts.values()))
        if not sc:
            print("  无可行动作", ex["id"])
            continue
        top = sc[0].action
        kof = lambda a0: next(k for k, a in acts.items() if a is a0 or id(a) == a0.args.get("_group"))
        key = kof(top)
        good = ex.get("good") or []
        ok = (key in good or not good) and key not in (ex.get("bad") or [])
        out.append(ok)
        print(f"  {'OK ' if ok else 'BAD'} {ex['id']}  选 {key}，正确 {good or '(不在 bad 里即可)'} 错误 {ex.get('bad')}  |  " +
              "  ".join(f"{kof(x.action)}:{x.action.label} {x.dV:+.1f}" for x in sc[:5]))
    return out

# ------------------------------------------------------------------ 回放
def _talents_by_round(run_dir: Path) -> dict:
    out = {}
    f = run_dir / "laya_points.jsonl"
    if f.exists():
        for line in f.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ctx = r.get("ctx") or {}
            rd = ctx.get("round") or r.get("round")
            if rd:
                out[int(rd)] = list(ctx.get("talents") or [])
    return out


LEVEL_ACTS = {"drag_card", "use_card", "equip", "deploy", "sequence", "move_unit", "sell", "buy_piece"}
SKIP_S = 4.0      # 回合开头跳过几秒
FIRST_N = 12      # 下一回合只看前 12 帧（≈ 准备阶段前 4~6 秒）：之后我们自己的操作（合成 / 用古币）会混进读数


def _worn_by_round(run_dir: Path) -> dict:
    """账本 → 每回合结束时各英雄穿的装备名（用牌·装备牌 加上，卖出 清空）。"""
    out, cur = {}, collections.defaultdict(list)
    f = run_dir / "ledger.jsonl"
    if not f.exists():
        return out
    last_r = 0
    for line in f.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rd = r.get("round") or last_r
        if rd != last_r:
            out[last_r] = {k: list(v) for k, v in cur.items()}
            last_r = rd
        ev = r.get("event")
        if ev == "用牌" and r.get("类别") == "装备牌" and r.get("目标"):
            cur[r["目标"]].append(r.get("牌"))
        elif ev == "卖出" and r.get("英雄"):
            cur.pop(r["英雄"], None)
    out[last_r] = {k: list(v) for k, v in cur.items()}
    return out


def replay(n_runs: int = 5, strict: bool = True, skip_s: float = 4.0):
    mode = ("严格：只用第 R+1 回合第一个改等级动作之前的读数" if strict
            else f"宽松：第 R+1 回合开头跳过 {skip_s:g}s 后前 {FIRST_N} 帧，可能混进自己的操作")
    print(f"\n== 回放（{mode}）：最近 {n_runs} 局，第 R 回合最后准备帧推一步 vs 第 R+1 回合整备后读数")
    runs = sorted([d for d in (ROOT / "data" / "logs").glob("live_*") if d.is_dir() and (d / "states.jsonl").exists()
                   and (d / "laya_points.jsonl").exists()], key=lambda d: d.name)[-n_runs:]
    errs, base_errs = [], []
    by_src = collections.defaultdict(list)
    by_round = collections.defaultdict(list)
    by_name = collections.defaultdict(list)
    worst = []
    for d in runs:
        rows = collections.defaultdict(list)
        for line in (d / "states.jsonl").open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            st = r.get("state") or {}
            if st.get("phase") == "prepare" and st.get("round_no") and r.get("board"):
                rows[int(st["round_no"])].append(r)
        tal = _talents_by_round(d)
        worn_r = _worn_by_round(d)
        for R in sorted(rows):
            if R + 1 not in rows:
                continue
            last = json.loads(json.dumps(rows[R][-1]))
            for name, b in (last.get("board") or {}).items():     # 最后一帧读不到的等级用本回合最近一次读数补
                if not b.get("lv"):
                    for r0 in reversed(rows[R]):
                        b0 = (r0.get("board") or {}).get(name) or {}
                        if b0.get("lv") and b0.get("cell") == b.get("cell"):
                            b["lv"] = b0["lv"]
                            break
            # 下一回合"整备结算后"的读数：跳过准备阶段开头 SKIP_S 秒（回合切换那几帧读到的是上回合的缓存 / 整备动画未结束），
            # 之后每个单位取第一个"连续两帧相同"的读数；只看前 FIRST_N 帧，避免混进我们自己的操作（合成 / 用古币）
            nxt = {"board": {}}
            t0 = rows[R + 1][0].get("ts") or 0
            seq = []
            for r1 in rows[R + 1]:                    # 截到本回合第一个真正执行的动作为止（之后的读数混进了我们自己的操作）
                if strict and r1.get("pre_action") is False:   # 2026-10-14 起日志直接标了"我们已经动过手"
                    break
                if (r1.get("ts") or 0) >= t0 + skip_s:
                    seq.append(r1)
                if strict and r1.get("executed") and (r1.get("ts") or 0) >= t0 + skip_s and any(
                        isinstance(a, dict) and a.get("type") in LEVEL_ACTS for a in (r1.get("actions") or [])):
                    break
            seq = seq[:FIRST_N]
            prev = {}
            for r1 in seq:
                for name, b1 in (r1.get("board") or {}).items():
                    v = b1.get("lv")
                    if not v or name in nxt["board"]:
                        continue
                    if prev.get(name) == (v, b1.get("cell")):
                        nxt["board"][name] = b1
                    prev[name] = (v, b1.get("cell"))
            tk = [v for k, v in tal.items() if k <= R]
            wk = [k for k in worn_r if k <= R]
            p = from_point(last, talents=tk[-1] if tk else [], worn=worn_r[max(wk)] if wk else {})
            pr = project(p, 1, params={"invest_frac": 0.0})   # 回放真值取在"我们动手之前"，日常投入（靠我们的操作兑现）不算
            for name, bd in (nxt.get("board") or {}).items():
                b0 = (last.get("board") or {}).get(name)
                if not b0 or not bd.get("lv") or not b0.get("lv"):
                    continue
                if (bd.get("cell") != b0.get("cell")):
                    continue                      # 换了格子（多半是重新上场 / 记账修正），不比
                pred = pr.end.unit(name).level if pr.end.unit(name) else b0["lv"]
                act = bd["lv"]
                e = pred - act
                errs.append(e)
                base_errs.append(b0["lv"] - act)
                u = pr.end.unit(name)
                srcs = [k for k, v in (u.src if u else {}).items() if abs(v) > 0.01]
                key = "无引擎来源" if not srcs else ("+".join(sorted(set(s.split("(")[0] for s in srcs)))[:20])
                by_src["有引擎来源" if srcs else "无引擎来源"].append(e)
                for s0 in set(x.split("(")[0] for x in srcs):
                    by_name[s0].append(e)
                by_round[min(R, 12)].append(e)
                worst.append((abs(e), d.name, R, name, b0["lv"], round(pred, 1), act, key))
    if not errs:
        print("  没有可比样本")
        return {}

    def stats(v):
        a = [abs(x) for x in v]
        return (f"n={len(v)} MAE={statistics.mean(a):.2f} 中位|e|={statistics.median(a):.2f} "
                f"偏差={statistics.mean(v):+.2f} |e|≤1 {sum(1 for x in a if x <= 1) / len(a):.0%} "
                f"|e|≤3 {sum(1 for x in a if x <= 3) / len(a):.0%}")
    print("  推演器       ", stats(errs))
    print("  基线(不变)   ", stats(base_errs))
    for k, v in by_src.items():
        print(f"  {k:<10}", stats(v))
    print("  按来源（该单位本步推演里有这个来源的样本）：")
    for k, v in sorted(by_name.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"    {k:<8}", stats(v))
    for R in sorted(by_round):
        print(f"  R{R:<3}", stats(by_round[R]))
    hist = collections.Counter(max(-10, min(10, round(x))) for x in errs)
    print("  误差分布（预测-实际，截断 ±10）:", " ".join(f"{k}:{hist[k]}" for k in sorted(hist)))
    worst.sort(reverse=True)
    print("  误差最大的 10 个：")
    for w in worst[:10]:
        print("   ", w)
    return {"engine": stats(errs), "baseline": stats(base_errs), "runs": [d.name for d in runs]}


def perf():
    print("\n== 性能：options 评估（真实局面）")
    d = sorted([x for x in (ROOT / "data" / "logs").glob("live_*") if (x / "states.jsonl").exists()],
               key=lambda x: x.name)[-1]
    rows = [json.loads(x) for x in (d / "states.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if (r.get("state") or {}).get("phase") == "prepare" and len(r.get("board") or {}) >= 5]
    r = rows[len(rows) // 2]
    p = from_point(r)
    p.shop = ["李元芳", "苏烈", "明世隐", "钟馗"]
    p.energy = max(p.energy, 10)
    p.hand = (p.hand + ["古币", "临阵磨枪"])[:10]
    ts = []
    for _ in range(20):
        t = time.perf_counter()
        s = score_options(p, horizon=3)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    n3 = sum(1 for x in s if x.action.horizon == 3)
    print(f"  动作 {len(s)}（3 回合视野 {n3}，换主C 视野到对局结束）  p50 {ts[10]:.1f}ms  p90 {ts[18]:.1f}ms")
    only3 = [a.action for a in s if a.action.horizon == 3 and a.action.kind != "refresh"][:10]
    ts2 = []
    for _ in range(20):
        t = time.perf_counter()
        score_options(p, actions=only3)
        ts2.append((time.perf_counter() - t) * 1000)
    ts2.sort()
    print(f"  10 个动作 × 3 回合：p50 {ts2[10]:.1f}ms p90 {ts2[18]:.1f}ms（目标 <50ms）")
    print(explain(s, 6))
    return ts2[10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", action="store_true")
    ap.add_argument("--replay", type=int, default=5)
    args = ap.parse_args()
    print("效果表来源：", db().source_count())
    unit_tests()
    if not args.unit:
        design_exams()
        point_exams()
        choice_exams()
        replay(args.replay, strict=True)
        replay(args.replay, strict=False, skip_s=8.0)
        perf()
    print(f"\n单元测试：通过 {PASSES[0]}，失败 {len(FAILS)}")
    for f in FAILS:
        print("  失败：", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
