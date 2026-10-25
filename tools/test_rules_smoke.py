"""规则引擎离线冒烟测试：构造典型局面跑 decide_all，不连游戏。

覆盖：准备阶段买牌/上场/升级/装备/技能/刷新/锁定/升级按钮、拍卖、天赋。
任何异常或输出明显不合理都会报错退出。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agent.actions.executor import validate_action
from agent.decision.rules import RuleEngine  # noqa: E402
from agent.decision import interim_rules as _IRT  # noqa: E402
_IRT.RULES['equip_hold_for_real_carry']['on'] = False   # 老场景的主C 多是 1 阶（曜）；这条临时规则在末尾专门测
from agent.perception.state import (  # noqa: E402
    AuctionLot,
    AuctionState,
    GameState,
    ShopSlot,
    TalentOption,
)

from agent.perception.extractors import cell_key  # noqa: E402
rules = RuleEngine(PROJECT_DIR / "data" / "chess_pieces.json",
                   cards_db_path=PROJECT_DIR / "data" / "cards.json")
RuleEngine.ROUTE_ENABLED = False     # 老场景测的是"阵容会跟着牌换"；官方路线锁定在文件末尾的 143~146 单独测

failures = []


def check(name: str, actions: list, expect_types: tuple[str, ...] | None = None):
    print(f"[{name}] {actions}")
    if expect_types and not any(a["type"] in expect_types for a in actions):
        failures.append(f"{name}: 期望 {expect_types}，实际 {actions}")
    # **顺手校验执行器要的字段在不在**。2026-09-23 实战：`sell_unit` 有一处调用
    # 给的是 x/y、执行器读的是 src，每次都 KeyError，一个英雄都没卖出去过；
    # 而异常被主循环吞成一行 ERROR，日志上只像是"这次没卖成"。
    # 决策层的输出是给执行层吃的，**冒烟测试必须连字段一起对**。
    for a in actions:
        problem = validate_action(a)
        if problem:
            failures.append(f"{name}: {problem}")
        for step in a.get("steps", []) or []:
            problem = validate_action(step)
            if problem:
                failures.append(f"{name}(sequence 内): {problem}")


def check_act(name: str, act, expect_type: str | None = None):
    """单个动作（decide_* 直接返回的那种）也要过字段校验。"""
    check(name, [act] if act else [], (expect_type,) if expect_type else None)
    return act


# 场景1：准备阶段，手牌有英雄+装备+主角技能，商店有牌，能量充足
s = GameState(shop_open=True, population_valid=True, phase="prepare", round_no=2, gold=12, population=1, population_cap=4,
              banned_faction="大河")
s.shop = [ShopSlot(0, "曜", 3, "逐鹿"), ShopSlot(1, "牛魔", 3, "河洛"),
          ShopSlot(2, "苏烈", 3, "河洛"), ShopSlot(3, "孙膑", 3, "逐鹿")]
s.hand = ["曜", "速击之枪", "枫叶舞"]
s.hand_slots = [{"name": "曜", "x": 300, "y": 670},
                {"name": "速击之枪", "x": 450, "y": 670},
                {"name": "枫叶舞", "x": 200, "y": 670}]
rules._field = {"牛魔": (660, 430)}   # 场上已有牛魔（河洛，但河洛没禁？禁的是大河）
rules._carry_piece = "牛魔"
rules._carry = (660, 430)
# 新流程：商店开着时先把该买的买完，棋盘操作等关店之后（商店面板挡着棋盘）
check("准备-能量足-先买", rules.decide_all(s), ("buy_piece",))

# 同一局面但商店已关 → 这时才做棋盘操作
s_closed = GameState(shop_open=False, population_valid=True, phase="prepare",
                     round_no=2, gold=12, population=1, population_cap=4,
                     banned_faction="大河")
s_closed.hand = s.hand
s_closed.hand_slots = s.hand_slots
check("准备-关店后上场", rules.decide_all(s_closed), ("drag_card",))

# 场景2：能量不足（0）→ 不该买牌
s2 = GameState(shop_open=True, population_valid=True, phase="prepare", round_no=3, gold=0, population=1, population_cap=4)
s2.shop = [ShopSlot(0, "曜", 3, "逐鹿"), ShopSlot(1, "牛魔", 3, "河洛"),
           ShopSlot(2, "苏烈", 3, "河洛"), ShopSlot(3, "孙膑", 3, "逐鹿")]
acts = rules.decide_all(s2)
if any(a["type"] == "buy_piece" for a in acts):
    failures.append(f"能量0仍买牌: {acts}")
print(f"[准备-能量0] {acts}")

# 场景3：手里有场上同名英雄 → 升级（合成）
s3 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=4, gold=9, population=2, population_cap=5)
s3.hand = ["牛魔"]
s3.hand_slots = [{"name": "牛魔", "x": 320, "y": 670}]
rules._field = {"牛魔": (660, 430), "孙膑": (540, 505)}
rules._carry_piece = "牛魔"
acts = rules.decide_all(s3)
check("准备-升级", acts, ("drag_card",))
if acts and acts[0].get("dst") != (660, 430):
    failures.append(f"升级目标错误: {acts}")

# 场景4：人口满 → 不上新英雄
s4 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=5, gold=20, population=5, population_cap=5)
s4.hand = ["李白"]
s4.hand_slots = [{"name": "李白", "x": 320, "y": 670}]
acts = rules.decide_all(s4)
if any(a["type"] == "drag_card" and a.get("piece") == "李白" for a in acts):
    failures.append(f"人口满仍上场: {acts}")
print(f"[准备-人口满] {acts}")

# 场景5：拍卖
s5 = GameState(phase="auction", round_no=7, gold=0)
s5.auction = AuctionState(lots=[
    AuctionLot(0, "海诺", 2, "对手甲", 1, True),
    AuctionLot(1, "上官婉儿", 0, "", 0, False),
    AuctionLot(2, "花木兰", 1, "对手乙", 1, False),
], countdown=10, my_gold=14)
check("拍卖", rules.decide_all(s5), ("auction_bid",))

# 场景6：天赋
s6 = GameState(phase="talent", round_no=5)
s6.talents = [TalentOption(0, "银算盘", "出售场上英雄时，随机1名英雄等级+2。"),
              TalentOption(1, "加速成长", "指定1名英雄等级+5。"),
              TalentOption(2, "节能商店", "刷新商店后总有1张牌能量-1。")]
check("天赋", rules.decide_all(s6), ("talent_pick",))

# 场景7：禁用主力派系 → 自动换备选阵容；店里有核心牌但钱不够 → 锁店
# 各场景共用一个引擎：前面"准备-能量0"也是第 3 回合、而且现在会为**当前主C 牛魔**的
# 合成材料锁店（这是对的），"本回合已锁过"的标记会漏进这里 —— 清掉，让场景独立。
rules._locked_round = -1
s7 = GameState(shop_open=True, population_valid=True, phase="prepare", round_no=3, gold=2, population=1, population_cap=4,
               banned_faction="大河流域")
s7.shop = [ShopSlot(0, "百里玄策", 3, "河洛"), ShopSlot(1, "李白", 3, "河洛"),
           ShopSlot(2, "白起", 3, "逐鹿"), ShopSlot(3, "曜", 3, "逐鹿")]
acts7 = rules.decide_all(s7)
print(f"[禁用换阵容] active={rules._active_comp['faction'] if rules._active_comp else None} {acts7}")
if not rules._active_comp or rules._active_comp["faction"] == "大河流域":
    failures.append(f"禁用后未换阵容: {rules._active_comp}")
if not any(a["type"] == "lock_shop" for a in acts7):
    failures.append(f"核心牌买不起未锁店: {acts7}")

# ---- 场景8：升级棋手（2026-09-22 实战：攒了一堆能量却整局停在 1 级） ----
# 根因是升级也吃 gold_reserve=10：能量 6、升级费 5，6-5=1 < 10 → 永远不升。
# 现在升级走独立的 level_reserve(=2)。
rules.reset_match()
s8 = GameState(shop_open=False, population_valid=True, phase="prepare",
               round_no=3, gold=8, level_cost=5, population=2, population_cap=5, hp=60,
               level_btn=(95, 550))
_acts8 = rules.decide_all(s8)
check("升级棋手-能量够", _acts8, ("buy_xp",))
rules._round_seen = s8.round_no
for _a8 in _acts8:
    rules.note_executed(_a8)                 # 升级在真正点出去之后才记账（2026-09-30）
# 同一回合不该再升第二次
if any(a["type"] == "buy_xp" for a in rules.decide_all(s8)):
    failures.append("同一回合重复升级")

# 没有 level_btn（画面上没看见"升级"二字）时**绝不能**产出 buy_xp：
# 那个写死坐标在英雄详情面板打开时是「出售」按钮，盲点就是把英雄卖了。
rules.reset_match()
s8b = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=3, gold=20, level_cost=5, population=2, population_cap=5, hp=60)
if any(a["type"] == "buy_xp" for a in rules.decide_all(s8b)):
    failures.append("画面上没有升级按钮却仍然点了（会误点「出售」）")

# ---- 场景9：弈星（棋手经验型英雄）该买、该赶在升级前上场、升完该卖 ----
rules.reset_match()
s9 = GameState(shop_open=True, population_valid=True, phase="prepare",
               round_no=2, gold=12, level_cost=4, population=1, population_cap=5, hp=60,
               level_btn=(95, 550))
s9.shop = [ShopSlot(0, "弈星", 3, "无阵营"), ShopSlot(1, None, None, ""),
           ShopSlot(2, None, None, ""), ShopSlot(3, None, None, "")]
acts9 = rules.decide_all(s9)
check("弈星-该买", acts9, ("buy_piece",))
if not any(a.get("piece") == "弈星" for a in acts9):
    failures.append(f"弈星没被买: {acts9}")

rules.reset_match()
s9b = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=2, gold=12, level_cost=4, population=1, population_cap=5, hp=60,
                level_btn=(95, 550))
s9b.hand = ["弈星"]
s9b.hand_slots = [{"name": "弈星", "x": 600, "y": 663}]
acts9b = rules.decide_all(s9b)
check("弈星-先上场再升级", acts9b, ("drag_card",))
order = [a["type"] for a in acts9b]
if "buy_xp" in order and order.index("buy_xp") < order.index("drag_card"):
    failures.append(f"弈星应在升级之前上场: {acts9b}")

# ---- 场景10：拖拽回执（confirm_drag）——失败后必须换个格子，不能原地重放 ----
rules.reset_match()
s10 = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=2, gold=0, population=0, population_cap=5)
s10.hand = ["牛魔"]
s10.hand_slots = [{"name": "牛魔", "x": 600, "y": 663}]
a10 = rules.decide_all(s10)
first = next((a for a in a10 if a["type"] == "drag_card"), None)
if first is None:
    failures.append(f"没有产生上场动作: {a10}")
else:
    rules.confirm_drag(first, ok=False)          # 画面告诉我们：没拖进去
    a10b = rules.decide_all(s10)
    second = next((a for a in a10b if a["type"] == "drag_card"), None)
    print(f"[拖拽失败换格] {first['dst']} -> {second['dst'] if second else None}")
    if second is None or tuple(second["dst"]) == tuple(first["dst"]):
        failures.append(f"拖拽失败后仍然放同一个格子: {first} / {second}")

# ---- 场景11：白送棋手经验的 0 费效果牌要立刻打出，且**落点必须是英雄** ----
# 2026-09-22 第七局实测：往空格拖连续三次全失败（目标 None 不在那儿），
# 这游戏的效果牌是"拖到英雄身上"打出的，没有"丢到空地"这种用法。
rules.reset_match()
rules._field = {"牛魔": (660, 430)}
rules._carry_piece, rules._carry = "牛魔", (660, 430)
s11 = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=1, gold=0, population=1, population_cap=5)
s11.hand = ["研读会"]
s11.hand_slots = [{"name": "研读会", "x": 600, "y": 663}]
a11 = rules.decide_all(s11)
check("自用经验牌-拖到英雄身上", a11, ("drag_card",))
cast = next((a for a in a11 if a.get("self_cast")), None)
if cast is None:
    failures.append(f"研读会没被当成自用牌打出: {a11}")
elif tuple(cast["dst"]) != (660, 430):
    failures.append(f"自用牌落点不是英雄位置: {cast['dst']}")

# 场上一个英雄都没有时**不该打**（往空格拖必然失败，白费一次拖拽）
rules.reset_match()
s11b = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=1, gold=0, population=0, population_cap=5)
s11b.hand = ["研读会"]
s11b.hand_slots = [{"name": "研读会", "x": 600, "y": 663}]
if any(a.get("self_cast") for a in rules.decide_all(s11b)):
    failures.append("场上没英雄却把效果牌往空格拖")

# ---- 场景12：装备要跟着主力**实际落点**走，不能用"打算放哪"的旧记录 ----
# 2026-09-22 第三局：曜 第一次拖 (660,420) 失败、最终落在 (780,420)，
# 但装备一直往 (660,420) 这个空格拖，四选一弹窗当然也找不到。
rules.reset_match()
s12 = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=2, gold=0, population=0, population_cap=5)
s12.hand = ["曜", "风暴巨剑"]
s12.hand_slots = [{"name": "曜", "x": 600, "y": 663},
                  {"name": "风暴巨剑", "x": 700, "y": 663}]
a12 = rules.decide_all(s12)
first = next(a for a in a12 if a["type"] == "drag_card" and a["piece"] == "曜")
rules.confirm_drag(first, ok=False)              # 第一个落点没进去
a12b = rules.decide_all(s12)
second = next(a for a in a12b if a["type"] == "drag_card" and a["piece"] == "曜")
rules.confirm_drag(second, ok=True)              # 换个格子成功了
s12.hand = ["风暴巨剑"]
s12.hand_slots = [{"name": "风暴巨剑", "x": 700, "y": 663}]
s12.population = 1
a12c = rules.decide_all(s12)
equip = next((a for a in a12c if a.get("kind") == "装备牌"), None)
print(f"[装备跟随主力] 曜 最终在 {second['dst']}，装备投向 {equip['dst'] if equip else None}")
if equip is None or tuple(equip["dst"]) != tuple(second["dst"]):
    failures.append(f"装备没跟着主力实际落点：英雄在 {second['dst']}，装备去了 "
                    f"{equip['dst'] if equip else None}")

# ---- 场景13：效果锦囊 → N选1 这条链路（2026-09-22 确认的机制） ----
# 锦囊是"每合成几次英雄系统白送"的 0 费牌，效果是"选择 1 张对应阶级的效果牌"。
# 它**不需要目标**（空场也该打），而且**打出后会弹 N 选 1**，
# 那个界面占着商店的位置、不选就一直卡死 —— 所以这两步必须都通。
rules.reset_match()
rules._field = {"牛魔": (660, 430)}
rules._carry_piece, rules._carry = "牛魔", (660, 430)
s13 = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=3, gold=0, population=1, population_cap=5)
s13.hand = ["1阶效果锦囊"]
s13.hand_slots = [{"name": "1阶效果锦囊", "x": 600, "y": 663}]
a13 = rules.decide_all(s13)
check("锦囊-拖到英雄身上打出", a13, ("drag_card",))
cast13 = next((a for a in a13 if a.get("self_cast")), None)
if cast13 is None:
    failures.append(f"锦囊没被当成自用牌打出: {a13}")
elif tuple(cast13["dst"]) != (660, 430):
    failures.append(f"锦囊落点不是英雄位置: {cast13['dst']}")

# 打出后弹 N 选 1：必须选，且**优先级最高**（不选会把后续全卡死）
s13b = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=3, gold=20, population=0, population_cap=5,
                 level_cost=5, level_btn=(95, 550))
s13b.card_pick = True
s13b.card_options = [{"name": "研读会", "x": 200, "y": 250},
                     {"name": "存款利息", "x": 500, "y": 250},
                     {"name": "扶摇直上", "x": 800, "y": 250}]
a13b = rules.decide_all(s13b)
check("锦囊-N选1必须选", a13b, ("card_pick",))
if a13b and a13b[0]["type"] != "card_pick":
    failures.append(f"N选1 不是最高优先级（会被卡死）: {a13b}")

# ---- 场景14：棋盘观测**只做加法**（2026-09-22 第八局的教训） ----
# 第一版让观测去删记账，结果单帧霍夫检测抖动（1格→0格→2格→1格），
# 把拖拽回执刚证实过的记录删掉，英雄被反复"删掉再重上"。
# 根因是**让弱证据推翻强证据**。现在观测只能标占用，不能否定任何记录。
rules.reset_match()
rules._field = {"牛魔": (660, 420), "曜": (540, 480)}
rules._cell_owner = {(660, 420): "牛魔", (540, 480): "曜"}
s14 = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=4, gold=0, population=1, population_cap=5)
s14.board_seen = True
s14.board_occupied = [(660, 420), (780, 480)]
rules.apply_board_observation(s14)          # 第一次：还没连续两帧，不该有动作
if rules._busy_cells:
    failures.append(f"单帧观测就标了占用（应该要连续两帧）: {rules._busy_cells}")
rules.apply_board_observation(s14)          # 第二次：同样的观测 → 采信
print(f"[棋盘观测-只做加法] _field={rules._field} busy={sorted(rules._busy_cells)}")
if "曜" in rules._field and "牛魔" in rules._field:
    pass
else:
    failures.append(f"观测删掉了记录（现在不该删任何记录）: {rules._field}")
if (780, 480) not in rules._busy_cells:
    failures.append("连续两帧观测到陌生单位却没标占用")

# 观测到的占用格不能再被选作落点
rules.reset_match()
s14b = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=4, gold=0, population=2, population_cap=5)
s14b.board_seen = True
s14b.board_occupied = [(660, 420)]
s14b.hand = ["牛魔"]
s14b.hand_slots = [{"name": "牛魔", "x": 600, "y": 663}]
a14 = rules.decide_all(s14b)
dep = next((a for a in a14 if a["type"] == "drag_card"), None)
print(f"[棋盘观测-避开占用] 落点 {dep['dst'] if dep else None}（(660,420) 已被观测占用）")
if dep and tuple(dep["dst"]) == (660, 420):
    failures.append("往观测到有人的格子上放英雄")

# **观测抖动不能毁掉记录**：一帧说有、一帧说没有，记录必须岿然不动
rules.reset_match()
rules._field = {"苏烈": (660, 420)}
for occ in ([(660, 420)], [], [(660, 420), (480, 480)], []):
    st = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=2)
    st.board_seen = True; st.board_occupied = occ
    rules.apply_board_observation(st)
if rules._field.get("苏烈") != (660, 420):
    failures.append("观测抖动把拖拽回执证实过的记录毁掉了")

# ---- 场景15：消耗型单位不能当投资目标（2026-09-22） ----
# 弈星是"升完级就卖"的棋手经验件，把枫叶舞/装备砸在它身上等于白扔。
# 旧逻辑是"首个上场的就当主力"，而弈星恰好第一个上场就被选中了。
rules.reset_match()
s15 = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=2, gold=0, population=0, population_cap=5)
s15.hand = ["弈星"]
s15.hand_slots = [{"name": "弈星", "x": 600, "y": 663}]
rules.decide_all(s15)                      # 弈星上场
print(f"[消耗件不当主力] 弈星上场后 _carry_piece={rules._carry_piece!r}（应为空）")
if rules._carry_piece == "弈星":
    failures.append("把消耗型单位（弈星）当成了装备/技能的投资目标")

# 再上一个正经英雄（**输出位**：坦克/辅助不当主C，2026-09-23 改），它才该成为投资目标
# 2026-10-11：孙膑卡面是全队增益（之后是配套），不当临时主C → 换成纯输出位孙尚香
s15.hand = ["孙尚香"]
s15.hand_slots = [{"name": "孙尚香", "x": 600, "y": 663}]
s15.population = 1
rules.decide_all(s15)
print(f"[消耗件不当主力] 孙尚香上场后 _carry_piece={rules._carry_piece!r}")
if rules._carry_piece != "孙尚香":
    failures.append(f"非消耗英雄上场后没接管投资目标: {rules._carry_piece!r}")

# 棋手技能牌的目标也要避开消耗件
rules.reset_match()
rules._field = {"弈星": (300, 569), "牛魔": (555, 569)}
if rules._skill_target() == "弈星":
    failures.append("棋手技能牌指给了消耗型单位")

# ---- 场景16：弈星的卖出时机（2026-09-22 修正） ----
# **"卖"是独立的价值判断**，不是"升完了就卖"：弈星自己也有战斗力（只是不大），
# 所以只有**这个位置有更好的用途**（人口吃紧 / 手牌里压着更该上场的阵容英雄）才卖。
def _mk(hand_names=(), pop=3, cap=6):
    st = GameState(shop_open=False, population_valid=True, phase="prepare",
                   round_no=5, gold=0, population=pop, population_cap=cap)
    st.hand_slots = [{"name": n, "x": 400 + i * 90, "y": 660}
                     for i, n in enumerate(hand_names)]
    return st

rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
rules._field = {"弈星": (300, 569), "牛魔": (555, 569)}

# 还没贡献过升级 -> 不能卖（卖早了等于白买）
rules._engine_levels = {}
if rules.decide_sell_unit(_mk()):
    failures.append("弈星还没贡献升级就被卖了")

# 贡献过升级，但人口宽裕、手牌里没有更该上的英雄 -> 留着当身板
rules._engine_levels = {"弈星": 1}
act = rules.decide_sell_unit(_mk(pop=3, cap=6))
print(f"[弈星-位置没人要] {'留着' if act is None else '卖了'}（期望：留着）")
if act is not None:
    failures.append("位置没人要时就把弈星卖了（它也有战斗力）")

# 升了 2 次但位置依然没人要 -> 还是留着（升级次数本身不构成卖的理由）
rules._engine_levels = {"弈星": 2}
act = rules.decide_sell_unit(_mk(pop=3, cap=6))
print(f"[弈星-升2次但位置没人要] {'留着' if act is None else '卖了'}（期望：留着）")
if act is not None:
    failures.append("把升级次数当成了卖的理由")

# 手牌里压着阵容 carry 却没位置 -> 该卖腾位
rules._engine_levels = {"弈星": 1}
# 2026-10-13：原来写 pop=3/6（有空位）也期望卖 —— 有空位直接上场就行，卖弈星只会少一个人（第 601、603 局）
act = rules.decide_sell_unit(_mk(hand_names=("曜",), pop=6, cap=6))
print(f"[弈星-手牌有carry等位] {'卖了' if act else '留着'}（期望：卖了）")
if rules.decide_sell_unit(_mk(hand_names=("曜",), pop=3, cap=6)):
    failures.append("有空位时为手里的主C 卖弈星（直接上场就行）")
if act is None:
    failures.append("手牌里有更该上场的英雄却不肯腾位")

# 人口吃紧 -> 该卖
rules._field = {"弈星": (300, 569), "牛魔": (555, 569)}
rules._engine_levels = {"弈星": 1}
# 2026-10-04 第 5~6 名复盘：人口紧但手里没英雄补位 → 留着（卖了少一人迎战）；手里有英雄 → 卖
act = rules.decide_sell_unit(_mk(pop=5, cap=6))
print(f"[弈星-人口紧-手里没人] {'卖了' if act else '留着'}（期望：留着）")
if act is not None:
    failures.append("人口吃紧、手里没人补位却卖了弈星")
act = rules.decide_sell_unit(_mk(hand_names=("苏烈",), pop=5, cap=6))
print(f"[弈星-人口紧-手里有人] {'卖了' if act else '留着'}（期望：卖了）")
if act is None:
    failures.append("人口吃紧却没卖掉弈星腾位")

print()

# ---- 场景17：攻略派生的运营判据（2026-09-22，来自 D 牌概率攻略） ----
# 术语：**品阶**=攻略说的"费用"，即 cards.json 的 quality；cost 是商店购买价，别混。
from agent.decision import meta  # noqa: E402

# 17a 升本 vs 搜牌：主力品阶决定"升到几级就停"
rules.reset_match()
# 2026-09-23 运营改为**至少按 3 阶主C**（1 阶只是过渡，推荐阵容里 1 阶当主C 的占 0%）
# -> 搜牌等级从 2 级挪到 4 级。下面几条的机制不变，阈值跟着挪。
rules._active_comp = {"carry": "曜", "core": []}          # 曜 1 阶 -> 按 3 阶算 -> 搜牌等级 4
s17 = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=5, gold=9, level_cost=5, population=2, population_cap=6,
                hp=80, level=4, level_btn=(95, 550))
act = rules.decide_economy(s17)
print(f"[搜牌期-不再无脑升本] {'升了' if act else '不升'}（期望：不升）")
if act is not None:
    failures.append("已到搜牌等级仍在无脑升本（钱该留给 D 牌）")

# 人口卡住时仍然该升（有牌上不去是硬伤）
rules.reset_match(); rules._active_comp = {"carry": "曜", "core": []}
s17b = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=5, gold=9, level_cost=5, population=5, population_cap=5,
                 hp=80, level=2, level_btn=(95, 550))
act = rules.decide_economy(s17b)
print(f"[搜牌期-人口卡住] {'升了' if act else '不升'}（期望：升了）")
if act is None:
    failures.append("人口卡住却因为搜牌期不肯升级")

# 还没到搜牌等级 -> 照常升
rules.reset_match(); rules._active_comp = {"carry": "露娜", "core": []}   # 品阶5 -> 搜牌等级6
s17c = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=5, gold=9, level_cost=5, population=2, population_cap=5,
                 hp=80, level=2, level_btn=(95, 550))
act = rules.decide_economy(s17c)
print(f"[高品阶主力-该升本] {'升了' if act else '不升'}（期望：升了）")
if act is None:
    failures.append("高品阶主力还没开出牌池就停止升本")

# 17b 低等级别强行刷高品阶（攻略点名的前期大坑）
rules.reset_match(); rules._active_comp = {"carry": "露娜", "core": []}
s17d = GameState(shop_open=True, population_valid=True, phase="prepare",
                 round_no=5, gold=12, level_cost=5, population=2, population_cap=5,
                 hp=80, level=2, level_btn=(95, 550))
s17d.shop = [ShopSlot(i, None, 3, "") for i in range(4)]   # 店里没有值得买的
act = rules.decide_refresh(s17d)
print(f"[2级搜5品阶] {'刷了' if act else '不刷'}（期望：不刷）")
if act is not None:
    failures.append("在刷不出主力品阶的等级上浪费能量刷新")

# 等级够了就该刷
rules.reset_match(); rules._active_comp = {"carry": "曜", "core": []}
s17e = GameState(shop_open=True, population_valid=True, phase="prepare",
                 round_no=5, gold=12, level_cost=5, population=2, population_cap=6,
                 hp=80, level=4, level_btn=(95, 550))      # 3 阶在 4 级才进池
s17e.shop = [ShopSlot(i, None, 3, "") for i in range(4)]
act = rules.decide_refresh(s17e)
print(f"[品阶刷得出-正常刷] {'刷了' if act else '不刷'}（期望：刷了）")
if act is None:
    failures.append("主力品阶刷得出来却不刷新")

# 17c 拍卖时序（2026-09-25 全历史日志实测更正）：第 A 回合（4/8/12）的拍卖在第 A-1 回合战斗之后、
#     第 A 回合收入到账之后、第 A 回合准备阶段之前。所以第 4 回合准备阶段拍卖**已经结束**，不该再压钱；
#     第 3 回合准备阶段只需留"出价上限 - 第 4 回合收入"的差额（收入 8 ≥ 上限 5 → 不用留）。
if meta.rounds_to_auction(3) != 1 or meta.rounds_to_auction(4) != 0 or meta.AUCTION_ROUNDS != (4, 8, 12):
    failures.append("拍卖回合表不对（应为第 4/8/12 回合开头）")
if not meta.auction_after_battle(11) or meta.auction_after_battle(10) or meta.lord_xp_gain(11) != 3         or meta.lord_xp_gain(12) != 1:
    failures.append("拍卖在第 11 回合战斗后（经验 +3 算在第 11 回合结束）")
rules.reset_match(); rules._active_comp = {"carry": "露娜", "core": []}
s17f = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=4, gold=9, level_cost=5, population=2, population_cap=5,
                 hp=80, level=2, level_btn=(95, 550))
h4 = rules._auction_hold(s17f)
s17f.round_no = 3
h3 = rules._auction_hold(s17f)
print(f"[拍卖留钱] 第4回合准备={h4}（期望 0，已拍完）第3回合准备={h3}（期望 0，第4回合收入8盖过出价上限）")
if h4 != 0 or h3 != max(0, rules.auction_max_bid - meta.energy_income(4)):
    failures.append("拍卖留钱按旧时序算（第 4 回合准备阶段拍卖早已结束）")

# 17d 扶摇直上（免费升一级）在升本期该优先拿
rules.reset_match(); rules._active_comp = {"carry": "露娜", "core": []}
s17g = GameState(shop_open=False, phase="prepare", round_no=5, level=2)
s17g.card_options = [{"name": "惊喜补贴", "x": 300, "y": 250},
                     {"name": "扶摇直上", "x": 600, "y": 250},
                     {"name": "研读会", "x": 900, "y": 250}]
act = rules.decide_card_pick(s17g)
print(f"[N选1-升本期] 选 {act['piece']}（期望：扶摇直上）")
if act["piece"] != "扶摇直上":
    failures.append("升本期没优先拿免费升级牌")

print()

# ---- 场景18：前期能量就 6 点，别被"预留"卡死不升级 ----
# 取自 2026-09-22 第十一局的**真实读数**：回合2 能量6 升级费6、回合3 能量6 升级费5，
# 两帧都因为 level_reserve=2 被拒（6-5=1 < 2），整局停在 1 级、能量攒到 44。
# 这是第三次犯同一个错（先是 gold_reserve=10，再是 level_reserve=2）：
# **底线必须对应一笔真能花出去的钱**，留 1~2 点零钱什么都买不到。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"弈星": (300, 569)}          # 棋手经验件在场 -> 这次升级额外吃经验
s18 = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=3, gold=6, level_cost=5, population=3, population_cap=5,
                hp=100, level=1, level_btn=(99, 545))
act = rules.decide_economy(s18)
print(f"[回合3-能量6-费5-弈星在场] {'升了' if act else '不升'}（期望：升了）")
if act is None:
    failures.append("能量 6、升级费 5 且弈星在场仍不升级（预留把自己卡死了）")

# 场上没有经验件时，商店里有想买的牌就该给它留买价（升级和买牌抢同一笔钱）
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
s18b = GameState(shop_open=True, population_valid=True, phase="prepare",
                 round_no=3, gold=6, level_cost=5, population=3, population_cap=5,
                 hp=100, level=1, level_btn=(99, 545))
s18b.shop = [ShopSlot(0, "曜", 3, "逐鹿")] + [ShopSlot(i, None, 3, "") for i in (1, 2, 3)]
act = rules.decide_economy(s18b)
print(f"[同样的钱-但店里有主力] {'升了' if act else '不升'}（期望：不升，先买主力）")
if act is not None:
    failures.append("店里有主力英雄却把钱拿去升级了")

print()

# ---- 场景19：升级奖励的两条机制（2026-09-22 补充） ----
# 人口上限 1~6 级是 5/5/6/6/7/7 —— **只有 2→3 和 4→5 会涨人口**。
if [meta.pop_cap_at(i) for i in range(1, 7)] != [5, 5, 6, 6, 7, 7]:
    failures.append("人口上限表不对")

# 3 级人口满：再升一级人口还是 6，所以**不能拿"人口满了"当升级理由**。
# 注意这两个用例**故意把能量压到刚好不溢出**（20 能量会走"有余钱照样升"那条路，
# 那条路是对的——升级的主要价值是开高品阶池，人口只是副产品），
# 不压住就隔离不出人口判据。
rules.reset_match()
# 2026-09-23 运营改为**至少按 3 阶主C**（1 阶只是过渡，推荐阵容里 1 阶当主C 的占 0%）
# -> 搜牌等级从 2 级挪到 4 级。下面几条的机制不变，阈值跟着挪。
# 3->4 虽然不涨人口，但开出 3/4 阶牌池（升级的主要价值是开高品阶池），是该升的；
# "人口满不能单独当升级理由"改在 5 级（7/7，已过搜牌等级 4，5->6 人口还是 7）测。
rules._active_comp = {"carry": "曜", "core": []}
s19 = GameState(shop_open=False, population_valid=True, phase="prepare",
                round_no=9, gold=10, level_cost=8, population=7, population_cap=7,
                hp=100, level=5, level_btn=(99, 545))
act = rules.decide_economy(s19)
print(f"[5级人口满-升了也不涨] {'升了' if act else '不升'}（期望：不升）")
if act is not None:
    failures.append("5 级人口满时白花升级费（升到 6 级人口还是 7）")

# 2 级人口满：升到 3 级人口 5->6，这次该升
rules.reset_match(); rules._active_comp = {"carry": "曜", "core": []}
s19b = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=6, gold=10, level_cost=8, population=5, population_cap=5,
                 hp=100, level=2, level_btn=(99, 545))
act = rules.decide_economy(s19b)
print(f"[2级人口满-升了能涨] {'升了' if act else '不升'}（期望：升了）")
if act is None:
    failures.append("2 级人口满、升级能涨人口却不升")

# 天赋三选一全不对路 -> 用掉这一级唯一的刷新机会；同一级不会再刷第二次
rules.reset_match()
st19 = GameState(phase="talent", round_no=6, level=2)
st19.talents = [TalentOption(0, "无关甲", "毫无关系"), TalentOption(1, "无关乙", "同样无关"),
                TalentOption(2, "无关丙", "还是无关")]
a1 = rules.decide_talent(st19)
a2 = rules.decide_talent(st19)
print(f"[天赋全不对路] 第一次 {a1['type']} / 第二次 {a2['type']}（期望 refresh 然后 pick）")
if a1["type"] != "talent_refresh" or a2["type"] != "talent_pick":
    failures.append(f"天赋刷新不是一次性的: {a1['type']} -> {a2['type']}")

print()

# ---- 场景20：等弈星上场不能无限期等（2026-09-22 第十二局） ----
# 实战：回合4 能量13、升级费只要2，因为弈星还在手上没上场，连等 11 帧不升级，
# 期间 decide_buy 把能量买光，等弈星上场时只剩 1 点。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
def _s20():
    st = GameState(shop_open=False, population_valid=True, phase="prepare",
                   round_no=5, gold=13, level_cost=2, population=3, population_cap=5,
                   hp=100, level=1, level_btn=(101, 545))
    st.hand_slots = [{"name": "弈星", "x": 600, "y": 660}]      # 在手上，没上场
    return st
got = [rules.decide_economy(_s20()) for _ in range(rules.LEVEL_WAIT_FRAMES + 1)]
# 前 N 帧让位给"上场"，第 N+1 帧必须升级。
# （再往后会返回 None，但那是"一回合只升一次"的正确保护，不是没升。）
waited = next((i for i, a in enumerate(got) if a is not None), len(got))
print(f"[等弈星上场] 让位 {waited} 帧后升级（上限 {rules.LEVEL_WAIT_FRAMES}）")
if waited != rules.LEVEL_WAIT_FRAMES:
    failures.append(f"等弈星上场让位了 {waited} 帧，期望 {rules.LEVEL_WAIT_FRAMES}")
if got[-1] is None:
    failures.append("让位帧数用完后仍然不升级")

print()

# ---- 场景21：装备不能往满员英雄身上拖（2026-09-22） ----
# 每个英雄最多 3 件；满了游戏**直接拒绝拖放**，装备牌留在手上、四选一不弹，
# 决策层看不出失败 -> 无限重拖（实测 风暴巨剑 连拖 3 次）。
# 判据必须来自**像素观测** state.equip_counts，不是自己记的账。
CARRY_POS, OTHER_POS = (718, 371), (617, 372)
def _s21(counts):
    st = GameState(shop_open=False, population_valid=True, phase="prepare",
                   round_no=9, gold=20, population=3, population_cap=7, hp=100, level=4)
    st.hand_slots = [{"name": "风暴巨剑", "x": 583, "y": 662}]
    st.equip_counts = {f'{k[0]},{k[1]}': v for k, v in counts.items()}
    return st

rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"曜": CARRY_POS, "牛魔": OTHER_POS}
rules._carry_piece, rules._carry = "曜", CARRY_POS

act = rules.decide_use_cards(_s21({CARRY_POS: 1, OTHER_POS: 0}))
print(f"[主力没满] 目标={act and act.get('target')}（期望：曜）")
if not act or act.get("target") != "曜":
    failures.append(f"主力还装得下却没给他: {act}")

rules._deployed_round = rules._skill_round = -1
act = rules.decide_use_cards(_s21({CARRY_POS: 3, OTHER_POS: 0}))
print(f"[主力已满3件] 目标={act and act.get('target')}（期望：不给曜；主C 满装后输出装给坦克/战士牛魔，设计约定：多余装备给坦克和战士）")
if not act or act.get("target") != "牛魔":
    failures.append(f"主力满 3 件仍往他身上拖 / 多余装备没给坦克: {act}")

act = rules.decide_use_cards(_s21({CARRY_POS: 3, OTHER_POS: 3}))
print(f"[场上都满了] {act}（期望：None，先留着）")
if act is not None:
    failures.append(f"人人装备满了还硬拖: {act}")

print()

# ---- 场景22：GameState 必须能 json 序列化 ----
# 主循环每帧都把 state 写进 states.jsonl。**任何一个字段不可序列化，整帧就崩**，
# 而崩的位置在动作执行之后、拖拽回执之前 —— 表现为"动作做了但记账学不到结果"，
# 对账于是反复移除再重上，死循环。
# 2026-09-22 第十四局就栽在这：equip_counts 用了元组当键，
# 表面现象是"判断太慢"+"马可波罗不上场"，真因是这里每帧抛 TypeError。
import json as _json  # noqa: E402
_st = GameState(shop_open=True, phase="prepare", round_no=5, level=3)
_st.shop = [ShopSlot(0, "曜", 3, "逐鹿")]
_st.hand_slots = [{"name": "风暴巨剑", "x": 1, "y": 2}]
_st.equip_counts = {"718,371": 3}
_st.card_options = [{"name": "扶摇直上", "x": 1, "y": 2}]
try:
    _json.dumps(_st.to_dict(), ensure_ascii=False)
    print("[GameState 可序列化] OK")
except TypeError as exc:
    print(f"[GameState 可序列化] FAIL: {exc}")
    failures.append(f"GameState 无法 json 序列化（会让主循环每帧崩）: {exc}")

print()

# ---- 场景23：一张牌不能吃光整回合的帧（2026-09-22） ----
# 每帧只允许一个变更动作。镜反复上场失败时，装备牌永远轮不到 ——
# 表面现象是反复使用镜 + 手牌里全是装备。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._round_seen = 5
CELL = (617, 372)
for _ in range(rules.CARD_FAILS_PER_ROUND):
    rules.confirm_drag({"type": "drag_card", "piece": "镜", "dst": CELL}, False)
print(f"[失败{rules.CARD_FAILS_PER_ROUND}次后] 镜被搁置={rules._parked('镜')}（期望 True）"
      f" 别的牌不受影响={not rules._parked('曜')}（期望 True）")
if not rules._parked("镜"):
    failures.append("一张牌连续失败后没有被搁置，会吃光整回合的帧")
if rules._parked("曜"):
    failures.append("搁置误伤了别的牌")

# 换一个回合就该解禁（搁置是"本回合"的，不是整局）
rules._round_seen = 6
if rules._parked("镜"):
    failures.append("搁置没有随回合解除")
print(f"[下一回合] 镜解禁={not rules._parked('镜')}（期望 True）")

print()

# ---- 场景24：「体面离场」这类天赋牌要会用（2026-09-22：实战中没用上） ----
# 两个原因叠在一起：① 这张牌**根本不在卡库里**（天赋「好聚好散」赠送，官网没收录），
# ② 就算在，decide_use_cards 也**完全没处理「天赋牌」这个类别**。
# 它本质是"带补偿的卖人"，所以只能指向消耗件，绝不能指向主力。
if "体面离场" not in rules.pieces:
    failures.append("体面离场 不在卡库（决策层会静默跳过）")

rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"曜": (718, 371), "弈星": (300, 569)}       # 弈星 = 消耗件
rules._carry_piece = "曜"
st24 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=9, gold=5, population=2, population_cap=7, hp=40, level=4)
st24.hand_slots = [{"name": "体面离场", "x": 600, "y": 660}]
act = rules.decide_use_cards(st24)
print(f"[体面离场] 目标={act and act.get('target')}（期望：弈星，不能是曜）")
if not act or act.get("target") != "弈星":
    failures.append(f"体面离场 没有指向消耗件: {act}")

# 场上没有消耗件 -> 先留着，绝不拿主力开刀
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"曜": (718, 371)}
rules._carry_piece = "曜"
st24b = GameState(shop_open=False, population_valid=True, phase="prepare",
                  round_no=9, gold=5, population=1, population_cap=7, hp=40, level=4)
st24b.hand_slots = [{"name": "体面离场", "x": 600, "y": 660}]
act = rules.decide_use_cards(st24b)
print(f"[场上只有主力] {act}（期望：None）")
if act is not None:
    failures.append(f"场上没有消耗件时拿主力开刀了: {act}")

print()

# ---- 场景25：开关商店的预算必须按"实际执行"计数（2026-09-22：第一回合没买英雄） ----
# 原来预算在**产生决策**时就加，而执行层有"同一动作 6 秒内不重发"的去重，
# 于是预算被从没真正点出去的动作耗光 —— 整局商店没打开过、一张英雄都没买。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
st25 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=1, gold=6, population=0, population_cap=5, hp=60, level=1)
acts = [a for a in (rules.decide_all(st25) or []) if a.get("type") == "toggle_shop"]
print(f"[第1回合-空场有钱] {acts}（期望：开店）")
if not acts:
    failures.append("第 1 回合有钱、场上空着，却不开商店")
before = rules._shop_opens
rules.decide_all(st25)                      # 决策再来一次：**不该消耗预算**
if rules._shop_opens != before:
    failures.append("只产生决策就消耗了开店预算（执行层去重会让它白白用光）")
rules.note_executed({"type": "toggle_shop", "intent": "open"})
if rules._shop_opens != before + 1:
    failures.append("实际执行后没有计数")
print(f"[预算计数] 决策 2 次只执行 1 次 -> 已开 {rules._shop_opens} 次（期望 1）")

print()

# ---- 场景26：觉醒回手（判据已回退，见下） ----
# 曾经用"账本说在X格 + 观测到X格没徽章 + 牌在手上（连续两帧）"来认觉醒回手。
# **已撤销**：徽章检测远不够完整（全历史帧统计 dy 连续分布在 80~135，
# 我按单个样本收窄后一个单位都检不到；放宽后又会漏检），
# 于是"没观测到"被当成"离场了"，把真实记录误删 ——
# 即"别让弱证据推翻强证据"。
# 现在仍只用人口读数（商店关着时实测 32/32 可靠）。
CELL_A, CELL_B = (718, 371), (617, 372)

# 枫叶舞的绑定对象（2026-09-22 两条机制合起来）：
#   绑定后要重绑必须先卖掉被绑的人 -> 前期主 C 没定时，绑个"凑合用"的，别绑主力也别绑消耗件；
#   主 C 一旦确定 -> "换绑给场上新的主C"。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._carry_piece = "曜"
rules._field = {"曜": CELL_B, "弈星": CELL_A, "牛魔": (300, 569)}
tgt = rules._skill_target()
print(f"[枫叶舞-主C已定] 目标={tgt}（期望：曜）")
if tgt != "曜":
    failures.append(f"主C 已在场却没绑给它: {tgt}")

rules.reset_match()
rules._active_comp = {"carry": "露娜", "core": []}   # 主C 还没到场
rules._carry_piece = ""
rules._field = {"弈星": CELL_A, "牛魔": (300, 569), "孙尚香": (617, 372)}
tgt = rules._skill_target()
print(f"[枫叶舞-主C未定] 目标={tgt}（期望：孙尚香 —— 能杀人的；不绑消耗件弈星、不绑没输出的牛魔，2026-09-30）")
if tgt != "孙尚香":
    failures.append(f"主C 未定时绑错人: {tgt}（消耗件不该绑）")

print()

# ---- 场景27：人口满了要卖弱的换强的，而不是继续乱买（2026-09-22） ----
# 实战死锁：人口 7/7、手牌堆了 9 张买来的英雄、血只剩 1，还在继续买。
# "现在感觉有啥买啥 疯狂买" / "场上的英雄你也不卖掉 就只买"。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
WEAK, STRONG_CELL = (300, 569), (718, 371)
rules._field = {"孙尚香": WEAK, "牛魔": STRONG_CELL}      # 孙尚香=品阶1 非阵容，最弱
def _s27(hand):
    st = GameState(shop_open=True, population_valid=True, phase="prepare",
                   round_no=10, gold=20, population=2, population_cap=2, hp=30, level=5)
    st.hand_slots = [{"name": n, "x": 500 + 90 * i, "y": 660} for i, n in enumerate(hand)]
    st.shop = [ShopSlot(0, "姜子牙", 3, "")] + [ShopSlot(i, None, 3, "") for i in (1, 2, 3)]
    return st

_st27 = _s27(["曜"])
_open27 = rules.decide_sell_unit(_st27)        # 商店开着：棋盘被挡，不许卖（2026-09-26 抓起来的是商店的牌）
if _open27 and _open27.get("type") == "sell_unit":
    failures.append(f"商店开着还去卖场上的人: {_open27}")
_st27.shop_open = False
act = rules.decide_sell_unit(_st27)     # 手上有主力，场上却是杂牌
print(f"[人口满+手上有主力] {act and (act['type'], act['piece'])}（期望：卖掉孙尚香）")
if not act or act.get("piece") != "孙尚香":
    failures.append(f"人口满、手上压着主力，却不卖弱的换位: {act}")

# 场上只剩阵容内的人 -> 不能乱卖
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
rules._field = {"曜": WEAK, "牛魔": STRONG_CELL}
_st27b = _s27(["孙尚香"]); _st27b.shop_open = False
act = rules.decide_sell_unit(_st27b)
print(f"[场上都是阵容内] {act}（期望：None，绝不卖主力/核心）")
if act is not None:
    failures.append(f"把阵容内的英雄卖了: {act}")

# 人口满且换不掉任何人时，不该继续买
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
rules._field = {"曜": WEAK, "牛魔": STRONG_CELL}
st = _s27([])
worth = rules._worth_buying(st.shop[0], st)
print(f"[人口满-买不买] 值得买={worth}（期望 False：买回来也上不去）")
if worth:
    failures.append("人口满、换不掉场上任何人，还在买牌")

print()

# ---- 场景28（已撤销）----
# 曾经做过"开局认出棋手站的格子并永久拉黑"。**前提是错的**：
# 规则事实：棋手不可能站在棋盘上（2026-09-22 确认）。
# 真正的问题是**徽章→格子的映射太松**（最近格子 + 曼哈顿容差 70），
# 把场外的棋手硬吸到了棋盘格上。已改成按实测纵向偏移卡死
# （真实单位 dy=+118~131／棋手 dy=+85／手牌区的圆 dy=-18），场外的自然排除。
# 相应的回归在 test_perception_offline 的锚点用例里。

# ---- 场景29：函数内 import 不能遮蔽模块级同名导入 ----
# Python 里函数内出现 `from x import f` 会让 f 成为**整个函数**的局部名，
# 于是该函数中**更早**引用同名的模块级导入会抛 UnboundLocalError。
# 这种错只在特定分支跑到时才炸，实战很难复现 ——
# 2026-09-22 就因此一局刷了 50 次"循环异常"（结算分支用 _ocr_boxes 时崩）。
import ast as _ast  # noqa: E402

def _shadowed_imports(path):
    tree = _ast.parse(Path(path).read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            for a in node.names:
                top.add(a.asname or a.name.split(".")[0])
    bad = []
    for fn in _ast.walk(tree):
        if not isinstance(fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        for node in _ast.walk(fn):
            if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                for a in node.names:
                    nm = a.asname or a.name.split(".")[0]
                    if nm in top:
                        bad.append(f"{Path(path).name}:{node.lineno} {fn.name}() 内 import {nm}"
                                   f" 遮蔽了模块级同名导入")
    return bad

_shadow = []
for _f in ("tools/live_match_agent.py", "agent/decision/rules.py",
           "agent/perception/extractors.py", "agent/actions/executor.py"):
    _shadow += _shadowed_imports(PROJECT_DIR / _f if "PROJECT_DIR" in dir() else _f)
print(f"[函数内import遮蔽检查] {len(_shadow)} 处（期望 0）")
for _m in _shadow:
    print("   ", _m)
failures.extend(_shadow)

print()

# ---- 场景30：商店开着时不能无限买（2026-09-22：买了一堆没上场） ----
# 这游戏**买完一张立刻补新卡**，于是"还有值得买的牌"永远成立，
# decide_buy 每帧都 return -> 永远走不到"关店 -> 上场"。
# 实战：19/20 帧全是买/刷新，场上一直 0 个人。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
def _s30(field, buys):
    rules._field = dict(field)
    rules._round_seen = 5        # 对齐回合号，否则 decide_all 会把本回合计数清零
    rules._buys_this_round = buys
    st = GameState(shop_open=True, population_valid=True, phase="prepare",
                   round_no=5, gold=30, population=len(field), population_cap=6,
                   hp=60, level=3)
    st.hand_slots = [{"name": "牛魔", "x": 500, "y": 660}]
    st.shop = [ShopSlot(0, "曜", 3, "逐鹿")] + [ShopSlot(i, None, 3, "") for i in (1, 2, 3)]
    return st

acts = rules.decide_all(_s30({}, 0))
print(f"[场上空+手里有人] {acts}（期望：先关店去上人）")
if not any(a.get("type") == "toggle_shop" for a in acts):
    failures.append(f"场上空着还在买牌: {acts}")

rules.reset_match(); rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
acts = rules.decide_all(_s30({"苏烈": (718, 371)}, rules.BUYS_PER_ROUND))
print(f"[买够了] {acts}（期望：先关店）")
if not any(a.get("type") == "toggle_shop" for a in acts):
    failures.append(f"本回合买够了还在继续买: {acts}")

# 预算按实际执行计
rules.reset_match()
before = rules._buys_this_round
rules.note_executed({"type": "buy_piece", "piece": "曜"})
if rules._buys_this_round != before + 1:
    failures.append("买入没有按实际执行计数")
print(f"[买入计数] {before} -> {rules._buys_this_round}")

# 阵容扣分要按"品阶几级才开池"，不是搜牌等级。2026-09-24 按截图「当前概率」行实测更正：品阶 N 在 N 级进池
_open = {q: meta.min_level_for(q) for q in range(1, 6)}
if _open != {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}:
    failures.append(f"品阶开池等级应为 {{1:1,2:2,3:3,4:4,5:5}}（实测），实际 {_open}")
print(f"[品阶开池等级] {_open}（期望 1..5 一一对应）")

print()

# ---- 场景31：开关商店不能互相打架（2026-09-22 实战无限开关） ----
# 关店条件说"买够了"，开店条件说"没有棋盘活要干" —— 两边互不知情就 close→open→close…
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._round_seen = 5          # 先对齐回合号，否则 decide_all 会把本回合计数清零
rules._buys_this_round = rules.BUYS_PER_ROUND
st31 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=5, gold=30, population=6, population_cap=6, hp=60, level=3)
acts = rules.decide_all(st31)
opens = [a for a in acts if a.get("type") == "toggle_shop" and a.get("intent") == "open"]
print(f"[买够了+没棋盘活] 开店动作 {opens}（期望：空，别再开店）")
if opens:
    failures.append("本回合买够了还去开店，会和关店条件形成无限开关")

print()

# ---- 场景32：场面没填满就该买人（2026-09-22 实战第6回合场上仍只有2人） ----
# 原判据是"前3回合且场上<3"，第4回合起只买品阶>=4；
# 可那时才 2~3 级、商店只出品阶1，于是几乎什么都不买，场面永远起不来。
rules.reset_match()
rules._active_comp = {"carry": "露娜", "core": []}      # 主C 是高品阶，商店里买不到
rules._field = {"苏烈": (718, 371), "牛魔": (617, 372)}
st32 = GameState(shop_open=True, population_valid=True, phase="prepare",
                 round_no=6, gold=20, population=2, population_cap=6, hp=60, level=3)
st32.shop = [ShopSlot(0, "孙尚香", 3, "")] + [ShopSlot(i, None, 3, "") for i in (1, 2, 3)]
worth = rules._worth_buying(st32.shop[0], st32)
print(f"[场上2/6-低品阶杂牌] 值得买={worth}（期望 True：空着场面必输）")
if not worth:
    failures.append("场面没填满却不买人补位")

# 场面满了就该挑了。**用真英雄填场**：原来用假名字 x0~x5，估值是 -1，
# "场上最弱的"就成了 -1 —— 这不是真实局面（真英雄最低也有十几分）。
rules._field = {n: (300 + i * 100, 569)
                for i, n in enumerate(["苏烈", "牛魔", "曜", "孙膑", "狂铁", "李元芳"])}
st32b = GameState(shop_open=True, population_valid=True, phase="prepare",
                  round_no=6, gold=20, population=6, population_cap=6, hp=60, level=3)
st32b.shop = st32.shop
worth = rules._worth_buying(st32b.shop[0], st32b)
print(f"[场上6/6-低品阶杂牌] 值得买={worth}（期望 False）")
if worth:
    failures.append("场面已满还买低品阶杂牌")

print()

# ---- 场景33：场面没填满时，买"新人头"优先于买"合成材料"（2026-09-22 实战） ----
# 打分里"重复卡+5、核心+3"让同名卡永远胜出：9 次买入全是 牛魔×5/曜×3，
# 场上却一直只有 1~2 个人。空着的位置比"让已有的那个更肥"更值钱。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
rules._field = {"牛魔": (617, 372)}
def _s33(pop, cap):
    st = GameState(shop_open=True, population_valid=True, phase="prepare",
                   round_no=4, gold=20, population=pop, population_cap=cap, hp=60, level=2)
    st.shop = [ShopSlot(0, "牛魔", 3, "逐鹿"), ShopSlot(1, "孙尚香", 3, "")]
    st.shop += [ShopSlot(i, None, 3, "") for i in (2, 3)]
    return st
act = rules.decide_buy(_s33(1, 5))
print(f"[场面空着] 买 {act and act['piece']}（期望：孙尚香，新人头优先于合成材料）")
if not act or act["piece"] != "孙尚香":
    failures.append(f"场面没填满却优先买合成材料: {act}")

rules._field = {f"x{i}": (300 + i * 100, 569) for i in range(5)}
rules._field["牛魔"] = (617, 372)
act = rules.decide_buy(_s33(6, 6))
print(f"[场面已满] 买 {act and act['piece']}（期望：牛魔，回到合成优先）")
if not act or act["piece"] != "牛魔":
    failures.append(f"场面已满却不优先合成: {act}")

print()

# ---- 场景34：用**卡面效果文本**参与买牌决策（2026-09-22） ----
# 之前买牌只看品阶和阵容归属，828 张卡的效果文本一个字都没读。
rules.reset_match()
rules._active_comp = {"carry": "露娜", "core": []}
rules._field = {"苏烈": (718, 371), "牛魔": (617, 372)}

# 召唤类英雄：召唤物不占人口，等于白嫖人头
b_sum, why_sum = rules._effect_bonus("鬼谷子", GameState(phase="prepare"))
b_plain, _ = rules._effect_bonus("孙尚香", GameState(phase="prepare"))
print(f"[效果加分] 鬼谷子(召唤)={b_sum} {why_sum} / 孙尚香={b_plain}")
if b_sum <= 0 or "召唤" not in "".join(why_sum):
    failures.append(f"召唤类英雄没有拿到效果加分: {b_sum} {why_sum}")

# 派系呼应：卡面点名的派系，场上得真有人才算数
rules._field = {"苏烈": (718, 371), "百里守约": (617, 372)}   # 两个河洛
st34 = GameState(phase="prepare")
b_hit, why_hit = rules._effect_bonus("李元芳", st34)     # 李元芳整备点名【河洛】
rules._field = {"曜": (718, 371), "牛魔": (617, 372)}        # 没有河洛
b_miss, _ = rules._effect_bonus("李元芳", st34)
print(f"[派系呼应] 场上有河洛={b_hit} {why_hit} / 没有河洛={b_miss}")
if b_hit <= b_miss:
    failures.append("卡面点名的派系在场上有人时没有加分")

# ---- 场景35：回执判不出成败时，也必须停手（2026-09-23 实战 回合6 全废） ----
# 白起 往 (319,375) 合成，拖拽回执每次都返回"判不了"（拖完那一瞬手牌读成空），
# 于是失败计数永远是 0、CARD_FAILS_PER_ROUND 永远不触发 ——
# 整个回合十几帧全在重发同一个动作，装备没贴、10 能量没花、牌一张没买。
# 止损必须有一条"试够次数就停"，不能只有"失败够次数就停"。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"白起": (319, 375)}
rules._round_seen = 6
st35 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=6, gold=10, population=4, population_cap=5, hp=60, level=2)
st35.hand_slots = [{"name": "白起", "x": 798, "y": 663}]
seen = []
for i in range(rules.DRAG_ATTEMPTS_PER_ROUND + 1):
    a = rules.decide_upgrade(st35)
    seen.append(a)
    if a:
        rules.note_executed(a)            # 真的点出去了，但**不给回执**
n_acts = sum(1 for a in seen if a)
print(f"[无回执重发] 连续决策 {len(seen)} 次，产出动作 {n_acts} 个"
      f"（期望 {rules.DRAG_ATTEMPTS_PER_ROUND} 个，之后必须停手）")
if n_acts > rules.DRAG_ATTEMPTS_PER_ROUND:
    failures.append(f"回执判不出成败时无限重发同一个动作（{n_acts} 次）")
if seen[-1] is not None:
    failures.append("试满次数之后还在发同一个动作")
# **试满次数也要给那个格子记一笔**：按牌的搁置是按回合的，下回合从头再来，
# 而回执一直"判不了"时 `_cell_fails` 永远不涨 -> 同一个落点会被反复试整局。
if rules._cell_fails.get((319, 375), 0) < 1:
    failures.append("试满次数被搁置，却没给那个格子记失败（跨回合会重复）")
# 两条止损都要在：牌被搁置（管住装备/效果牌那条链），场上记录被撤销（管住合成那条链）
if not rules._parked("白起"):
    failures.append("试满次数之后这张牌没有被搁置")
if "白起" in rules._field:
    failures.append("合成反复不成，却还认为它在场上")

print()

# ---- 场景36：枫叶舞常驻手牌，不能永远算作"还有棋盘活要干"（2026-09-23 实战） ----
# 它用完仍然留在手上（机制），原来无条件算一件待办 →
# `_board_work_pending` 永远 True → 关了商店再也开不回来 →
# 整局不刷新不买牌，能量从回合6 的 10 点一路攒到回合8 的 23 点。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"曜": (617, 372)}
rules._maple_bound = "曜"                 # 已经绑好了，再打出去是无效动作
st36 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=6, gold=10, population=1, population_cap=5, hp=60, level=2)
st36.hand_slots = [{"name": "枫叶舞", "x": 700, "y": 663}]
pending = rules._board_work_pending(st36)
print(f"[枫叶舞已绑定] 还有棋盘活要干={pending}（期望 False，否则商店永远开不回来）")
if pending:
    failures.append("绑定完成的枫叶舞仍被当成待办，商店会永远开不回来")

# 但它**还亮着**（可以重绑）时，确实是待办
st36.hand_slots = [{"name": "枫叶舞", "x": 700, "y": 663, "ready": True}]
rules._maple_bound = ""
if not rules._board_work_pending(st36):
    failures.append("枫叶舞还能绑却没被当成待办")

print()

# ---- 场景37：拍卖不能自己抬自己（2026-09-23 实战，09-22 已出现过） ----
# 防自抬原来按**位置**记（lot.index），而拍品卖掉一个或滑窗多读一个，整排就重新编号：
# 我们对 0 号位的司空震出价 1，下一帧它变成 1 号位 -> 判据不成立 -> 又加一次价。
from agent.perception.state import AuctionState, AuctionLot
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}

def _s37(lots):
    st = GameState(phase="auction", round_no=4, gold=20, hp=60, level=2)
    st.auction = AuctionState(preview=False, my_gold=20)
    st.auction.lots = lots
    return st

lot_a = AuctionLot(index=0, piece="司空震", top_bid=0, x=327, y=257)
lot_b = AuctionLot(index=1, piece="韩信", top_bid=5, x=533, y=232)
act = rules.decide_auction(_s37([lot_a, lot_b]))
print(f"[首次出价] {act and (act['piece'], act['bid'])}（期望 司空震 1）")
if not act or act["piece"] != "司空震":
    failures.append(f"第一次出价没选中最划算的拍品: {act}")

# 同一张牌**换到别的位置**，价格还是我们出的那个 -> 绝不能再出价
lot_a2 = AuctionLot(index=1, piece="司空震", top_bid=1, x=326, y=232)
act2 = rules.decide_auction(_s37([lot_b, lot_a2]))
same = act2 and act2["piece"] == "司空震"
print(f"[拍品换位后] {act2 and (act2['piece'], act2['bid'])}（期望：不再对司空震出价）")
if same:
    failures.append("拍品换了位置就对同一张牌重复出价（自己抬自己）")

# 别人加价超过我们了 -> 应该允许追价
lot_a3 = AuctionLot(index=1, piece="司空震", top_bid=4, x=326, y=232)
act3 = rules.decide_auction(_s37([lot_a3]))
print(f"[被别人反超] {act3 and (act3['piece'], act3['bid'])}（期望：追价到 5）")
if not act3 or act3["bid"] != 5:
    failures.append(f"被反超之后不敢追价: {act3}")

print()

# ---- 场景38：枫叶舞不绑消耗件（2026-09-22：不挂给奕星） ----
# 原来最后一档兜底是 next(iter(_field))，场上只有弈星时正好绑上去。
# 不绑没有代价：它常驻手牌、随时能再打；绑错了反而要把人卖掉才能解绑。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"弈星": (300, 569)}
tgt = rules._skill_target()
print(f"[场上只有消耗件] 枫叶舞绑给 {tgt}（期望 None）")
if tgt is not None:
    failures.append(f"枫叶舞绑给了消耗件 {tgt}")

rules._field = {"弈星": (300, 569), "牛魔": (617, 372)}
tgt = rules._skill_target()
print(f"[场上只有弈星/牛魔] 枫叶舞绑给 {tgt}（期望 None：牛魔没输出、杀不了人，绑了拿不到阿离专属卡）")
if tgt is not None:
    failures.append(f"枫叶舞没绑给留得住的英雄: {tgt}")

print()

# ---- 场景39：非拍卖回合不该被"静态底线"卡住买不动（2026-09-23 实战回合4） ----
# gold_reserve=10 的本意是给拍卖留钱，但后来加的 _auction_hold 已经在动态算了，
# 两条叠加 = 场上一有 3 个人就要 13 能量才肯买一张 3 费牌，而中期收入才 8~12。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
rules._field = {"牛魔": (617, 372), "蒙恬": (529, 372), "芈月": (300, 569)}

def _s39(round_no, gold):
    st = GameState(shop_open=True, population_valid=False, phase="prepare",
                   round_no=round_no, gold=gold, population=0, population_cap=0,
                   hp=60, level=2)
    st.shop = [ShopSlot(0, "苏烈", 3, "河洛")] + [ShopSlot(i, None, 3, "") for i in (1, 2, 3)]
    return st

st = _s39(5, 8)          # 非拍卖回合
ok = rules._worth_buying(st.shop[0], st)
print(f"[非拍卖回合 8 能量] 苏烈值不值得买={ok}（期望 True，场面没填满就该补位）")
if not ok:
    failures.append("非拍卖回合、场面没填满，却因为静态底线不买牌")

# 第 4 回合准备阶段拍卖已经结束（2026-09-25 实测时序），7 能量照样该补位。
st = _s39(4, 7)
ok7 = rules._worth_buying(st.shop[0], st)
print(f"[拍卖回合准备阶段] 7 能量买={ok7}（期望 True，拍卖在准备阶段之前就拍完了）")
if not ok7:
    failures.append("第 4 回合准备阶段还在为已经拍完的拍卖压钱")

print()

# ---- 场景40：觉醒回手要能重新上场（2026-09-22 / 09-23 两次出现） ----
# 觉醒后人从棋盘消失、卡回到手牌，而 _field 还记着它在场上 ->
# 同名牌被当成合成材料，往空格子上拖，永远失败，牌就烂在手里。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["李元芳"]}
rules._field = {"李元芳": (617, 372)}
act = {"type": "drag_card", "piece": "李元芳", "merge": True, "dst": (617, 372)}
rules.confirm_drag(act, False)
rules.confirm_drag(act, False)          # 连续两次合成失败
print(f"[合成连failed两次] 还认为在场上={'李元芳' in rules._field}（期望 False）"
      f" 标成觉醒回手={'李元芳' in rules._awakened}（期望 True）")
if "李元芳" in rules._field:
    failures.append("合成连续失败后仍然认为它在场上")
if "李元芳" not in rules._awakened:
    failures.append("没把它标成觉醒回手，重新上场拿不到优先级")

st40 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=6, gold=0, population=0, population_cap=6, hp=60, level=3)
st40.hand_slots = [{"name": "孙尚香", "x": 400, "y": 663},
                   {"name": "李元芳", "x": 500, "y": 663}]
act = rules.decide_deploy(st40)
print(f"[重新上场] 先上 {act and act['piece']}（期望 李元芳，觉醒回手优先）")
if not act or act["piece"] != "李元芳":
    failures.append(f"觉醒回手的牌没被优先重新打出: {act}")

print()

# ---- 场景41：卖人动作必须带执行器认的字段（2026-09-23 实战 KeyError） ----
# `sell_unit` 有一处调用给的是 x/y、执行器读的是 src -> 每次 KeyError，
# 一个英雄都没卖出去过。2026-09-22 就出现过"场上的英雄不卖、只买"，
# 当时查的是决策逻辑，没想到是**执行层字段对不上**。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
rules._field = {n: (300 + i * 110, 569) for i, n in
                enumerate(["狂铁", "孙膑", "牛魔", "蒙恬", "白起", "猪八戒"])}
st41 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=6, gold=0, population=6, population_cap=6, hp=60, level=3)
st41.hand_slots = [{"name": "曜", "x": 500, "y": 663}]
act41 = rules.decide_sell_unit(st41)
print(f"[人口满+手里有主C] 卖人动作 {act41}")
if not act41:
    failures.append("人口满、手里压着主C，却不肯卖掉最弱的换位")
else:
    problem = validate_action(act41)
    if problem:
        failures.append(f"卖人动作执行器吃不下: {problem}")

print()

# ---- 场景42：商店开着人口读不到时，买牌要按"上次读到的人口"判满（2026-09-23 实战） ----
# 记账会掉人（回执丢失/对账修正/卖人执行失败），实战出现过 账本2 / 实际5。
# 那一帧商店开着、人口读不到，于是按"场上才 2 个"继续买，手牌堆到 8/10。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"孙膑": (300, 569), "牛魔": (617, 372)}   # 账本只剩 2 个
rules._last_pop_seen = (5, 5)                             # 但上次读到的是满的
st42 = GameState(shop_open=True, population_valid=False, phase="prepare",
                 round_no=6, gold=12, population=0, population_cap=0, hp=60, level=2)
st42.shop = [ShopSlot(0, "蒙恬", 3, "逐鹿")] + [ShopSlot(i, None, 3, "") for i in (1, 2, 3)]
ok = rules._worth_buying(st42.shop[0], st42)
print(f"[人口读不到+上次满员] 还买杂牌吗={ok}（期望 False）")
if ok:
    failures.append("人口读不到时按残缺的账本判断场面，继续买牌堆手牌")

rules._last_pop_seen = (2, 5)          # 上次读到确实没满 -> 该买
if not rules._worth_buying(st42.shop[0], st42):
    failures.append("上次读到场面没满，却不肯补位")

print()

# ---- 场景43：过了前期就别再为"搜牌"压着不升级（2026-09-23 实战 8 回合还 2 级） ----
# 攻略"1/2 费核心暂缓升级"说的是开局那几回合（那时低品阶 100% 能刷到）。
# 套成整局的后果：第 8 回合还停在 2 级、攒 17 能量，人口锁死 5，血从 60 掉到 30。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}      # 曜 是品阶1 -> 搜牌等级 2
rules._field = {"曜": (617, 372)}

def _s43(round_no, gold):
    return GameState(shop_open=False, population_valid=False, phase="prepare",
                     round_no=round_no, gold=gold, hp=40, level=2,
                     level_cost=6, level_btn=(95, 550))

# 前期 11 能量：升完剩 5，不够再刷三次 -> 钱该留给 D 牌，不升
act_early = rules.decide_economy(_s43(3, 11))
rules._leveled_round = -1
# 第 9 回合同样 11 能量（**不是拍卖回合**，不用留举牌钱）：升完剩 5，
# 够买一张牌 -> 该升（人口和高品阶池比牌池浓度重要）
act_late = rules.decide_economy(_s43(9, 11))
print(f"[前期第3回合 2级] {'升级' if act_early else '压着不升'}（期望：升级 —— 1 阶只是过渡，2 级不是搜牌期）")
print(f"[第9回合] {'升级' if act_late else '压着不升'}（期望：升级，人口比牌池浓度重要）")
if not act_early:
    failures.append("第 3 回合 2 级还压着不升（那局第 10 回合才 4 级，第 5 名）")
# 搜牌期（4 级）的前期：钱留给 D 牌
rules._leveled_round = -1
_s43b = GameState(shop_open=False, population_valid=False, phase="prepare",
                  round_no=5, gold=11, hp=40, level=4, level_cost=6, level_btn=(95, 550))
act_s = rules.decide_economy(_s43b)
print(f"[第5回合 4级] {'升级' if act_s else '压着不升'}（期望：压着不升，已到搜牌等级，钱留给D牌）")
if act_s:
    failures.append("到了搜牌等级（4 级）还在把钱花在升本上")
if not act_late:
    failures.append("打到第 9 回合还为搜牌压着不升级")

print()

# ---- 场景44：主力格这一帧没看见徽章 -> **还是给主力**；拖失败两次才换人 ----
# 两次真实事故互相拉扯，取舍记在这里：
#   ① 2026-09-23 早：「守护者之铠」连拖两次到记账里的主力位，那格是空的 ——
#      当时的修法是"没看见徽章就转投别人"。
#   ② 2026-09-23 午（实战截图：装备穿到了弈星身上）：主力格只是**漏检**（徽章检测 ~1/3 漏检），
#      转投到一个记账已过期的格子，装备穿到了弈星身上，名字还记成主力。
# ② 更糟：送错人收不回来；① 的代价只是两次失败拖拽，而 `_cell_dead` 两次后就会换人。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
rules._carry_piece = "曜"
rules._field = {"曜": (529, 372), "牛魔": (617, 372)}
st44 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=6, gold=0, population=2, population_cap=6, hp=60, level=3)
st44.unit_anchors = {"617,372": 1}        # 画面上只看到牛魔那一格有人（主力格漏检）
tgt = rules._equip_target(st44)
print(f"[主力格没看见徽章] 装备贴给 {tgt}（期望：还是主力 曜 @529,372）")
if not tgt or tgt[0] != "曜" or tuple(tgt[1]) != (529, 372):
    failures.append(f"主力格只是没看见就把装备转给了别人: {tgt}")
rules._cell_fails[(529, 372)] = rules.CELL_FAIL_LIMIT       # 真拖失败了两次
tgt = rules._equip_target(st44)
print(f"[主力格拖失败两次] 装备贴给 {tgt}（期望：牛魔 @617,372，名字报牛魔）")
if not tgt or tgt[0] != "牛魔" or tuple(tgt[1]) != (617, 372):
    failures.append(f"主力格已废却没换人，或换了人名字没改: {tgt}")
rules._cell_fails.clear()
# 主力在手里（觉醒回手/刚买）时，装备先留着，别塞给路人
rules._field = {"牛魔": (617, 372), "弈星": (672, 430)}
st44.hand_slots = [{"name": "曜", "x": 700, "y": 663, "awakened": True}]
tgt = rules._equip_target(st44)
print(f"[主力还在手里] 装备 {tgt}（期望 None：等主力上场）")
if tgt is not None:
    failures.append(f"主力还在手里就把装备给了别人: {tgt}")
st44.hand_slots = []
rules._field = {"曜": (529, 372), "牛魔": (617, 372)}   # 还原给下面那条原有检查

st44.unit_anchors = {"529,372": 1, "617,372": 1}    # 两格都有人 -> 回到主力
tgt = rules._equip_target(st44)
if not tgt or tuple(tgt[1]) != (529, 372):
    failures.append(f"主力位画面上有人，装备却没贴给主力: {tgt}")

print()

# ---- 场景45：记账空了但画面上有人，装备照样要贴（2026-09-23 实战） ----
# 账本会被对账/觉醒判定/卖人失败清掉，而画面上可能满编 5 个人。
# 原来 `if not self._field: return None` 这条门把装备牌挡了整整两个回合。
rules.reset_match()
rules._active_comp = {"carry": "沈梦溪", "core": []}
rules._carry_piece = "沈梦溪"
rules._field = {}                                  # 账本空
st45 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=5, gold=1, population=5, population_cap=5, hp=54, level=2)
st45.hand_slots = [{"name": "圣者法典", "x": 500, "y": 663}]
st45.unit_anchors = {"433,569": 1, "300,569": 1}    # 但画面上看到两格有人
act45 = rules.decide_use_cards(st45)
print(f"[账本空+画面有人] 装备动作 {act45}")
if not act45 or act45.get("piece") != "圣者法典":
    failures.append(f"记账空了就不肯贴装备，可画面上明明有人: {act45}")

rules._field = {}
st45.unit_anchors = {}                             # 画面上也没人 -> 确实不该贴
if rules.decide_use_cards(st45):
    failures.append("场上真没人却还要贴装备")

print()

# ---- 场景46：按观测修正落点（2026-09-23 实战） ----
# 拖拽回执只能说明"牌离手了"，落到哪一格要靠看。孙膑 记在 (300,569)、
# 画面上人却在 (411,372)，于是给它贴装备全拖到空地，装备件数一直是 0。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"孙膑": (300, 569)}
rules._cell_owner = {(300, 569): "孙膑"}
rules.relocate_unit("孙膑", (411, 372))
print(f"[落点修正] 孙膑 -> {rules._field.get('孙膑')}（期望 (411, 372)）")
if rules._field.get("孙膑") != (411, 372):
    failures.append("relocate_unit 没把落点改过来")
if rules._cell_owner.get((300, 569)) == "孙膑":
    failures.append("旧格子的归属没清掉，以后别人放不进去")
if rules._cell_owner.get((411, 372)) != "孙膑":
    failures.append("新格子没登记归属")

print()

# ---- 场景47：删记账要连续两帧确认（2026-09-23 自评报告暴露） ----
# 人口偶尔读成 1/5（真值 6/6），单帧就删的话一次抹掉 5 条记录。
# 自评报告里"结束时场上 1 人 / 人口 6/6"、"对账异常 9 次"就是这么来的。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {n: (300 + i * 110, 569) for i, n in
                enumerate(["曜", "牛魔", "蒙恬", "白起", "孙膑", "芈月"])}

def _s47(pop):
    st = GameState(shop_open=False, population_valid=True, phase="prepare",
                   round_no=7, gold=5, population=pop, population_cap=6, hp=30, level=3)
    return st

rules.reconcile(_s47(1))          # 第一次读到 1：先别信
print(f"[人口突然读成1·第一帧] 账本还剩 {len(rules._field)} 条（期望 6，先不删）")
if len(rules._field) != 6:
    failures.append("单帧人口误读就把账本删了")

rules.reconcile(_s47(6))          # 下一帧读回 6：说明刚才是误读，计数要清掉
print(f"[下一帧读回6] 账本 {len(rules._field)} 条（期望 6）")
if len(rules._field) != 6:
    failures.append("人口读回正常值后账本仍被删")

rules.reconcile(_s47(2))          # 连续两帧都读到 2：这次才该信
rules.reconcile(_s47(2))
print(f"[连续两帧读到2] 账本 {len(rules._field)} 条（期望 2）")
if len(rules._field) != 2:
    failures.append(f"连续两帧一致却没修正账本: {len(rules._field)}")

print()

# ---- 场景48：场面没填满时，升级不能把买牌钱花光（2026-09-23 实战） ----
# 原来有一条"场上有弈星就连买牌钱也不留"的例外，第 2 回合把 6 能量全花在 1->2 级上，
# 而 1->2 既不涨人口（5->5）也不开新品阶（1->1），结果第 3 回合场上还只有 1 个人。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"弈星": (300, 569)}        # 场上就一个棋手经验件

def _s48(gold):
    st = GameState(shop_open=True, population_valid=True, phase="prepare",
                   round_no=2, gold=gold, population=1, population_cap=5, hp=60,
                   level=1, level_cost=6, level_btn=(95, 550))
    st.shop = [ShopSlot(0, "牛魔", 3, "河洛")] + [ShopSlot(i, None, 3, "") for i in (1, 2, 3)]
    return st

act = rules.decide_economy(_s48(8))      # 8-6=2，不够再买一张 -> 先别升
print(f"[8能量·场面空] {'升级' if act else '不升，留着买牌'}（期望：不升）")
if act:
    failures.append("场面没填满还把买牌钱花在升级上")

rules._leveled_round = -1
act = rules.decide_economy(_s48(9))      # 9-6=3，升完还买得起一张 -> 该升
print(f"[9能量·场面空] {'升级' if act else '不升'}（期望：升级，升完还剩 3 够买一张）")
if not act:
    failures.append("升完还买得起牌却不肯升级")

print()

# ---- 场景49：商店开着时不能往被面板盖住的格子拖（2026-09-23 证据帧） ----
# "关店点了 2 次还开着就带着商店硬干"这个逃生口，造成的正是它想避免的伤害：
# 曜 上场和 破碎圣杯 贴装备都拖到了 (617,372)，而那一格在商店面板底下
# （面板盖到 y≈430），棋盘上一个单位都没有，拖拽回执却还报"成功"。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._shop_closes = 2            # 已经点过两次关店，逃生口打开
st49 = GameState(shop_open=True, population_valid=True, phase="prepare",
                 round_no=1, gold=0, population=0, population_cap=5, hp=60, level=1)
st49.hand_slots = [{"name": "曜", "x": 797, "y": 662}]
act = rules.decide_deploy(st49)
print(f"[商店开着·硬干] 落点 {act and act['dst']}（期望 y>=440，别拖到面板底下）")
if not act:
    failures.append("商店关不上时完全不上场（逃生口失效）")
elif act["dst"][1] < rules.SHOP_PANEL_BOTTOM_Y:
    failures.append(f"往商店面板盖住的格子上场了: {act['dst']}")

# 商店关上之后，前排照常可用
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
st49.shop_open = False
st49.hand_slots = [{"name": "牛魔", "x": 797, "y": 662}]   # 坦克 -> 前排
act = rules.decide_deploy(st49)
print(f"[商店关着] 落点 {act and act['dst']}（期望前排，y<440 也允许）")
if not act:
    failures.append("商店关着却不上场")

print()

# ---- 场景50：拖失败不该吃掉本回合的上场配额（2026-09-23 实战回合6） ----
# 人口读不到时 decide_deploy 退化成"每回合最多上 1 个"，而配额在**产生决策**时就加了。
# 于是一次拖失败就把整回合的上场机会吃光：手里压着 5 张牌、场面 3/5，
# 剩下十几帧全是空决策。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
st50 = GameState(shop_open=False, population_valid=False, phase="prepare",
                 round_no=6, gold=0, population=0, population_cap=0, hp=48, level=2)
st50.hand_slots = [{"name": "百里玄策", "x": 844, "y": 663},
                   {"name": "牛魔", "x": 944, "y": 663}]
a1 = rules.decide_deploy(st50)
print(f"[第一次上场] {a1 and a1['piece']}")
rules.confirm_drag(a1, False)             # 拖失败
a2 = rules.decide_deploy(st50)
print(f"[失败后再试] {a2 and a2['piece']}（期望：还能再上一个，别空转整回合）")
if not a2:
    failures.append("一次拖失败就把整回合的上场配额吃光了")

# 成功之后才真的用掉配额
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
a3 = rules.decide_deploy(st50)
rules.confirm_drag(a3, True)
a4 = rules.decide_deploy(st50)
print(f"[成功后再试] {a4 and a4['piece']}（期望 None，人口读不到时每回合只上 1 个）")
if a4:
    failures.append("人口读不到时一回合上了不止 1 个")

print()

# ---- 场景51：人口读不到时也要判"场面满没满"（2026-09-23 实战回合5~6） ----
# 场面其实已经 5/5，而那几帧人口读不到 -> "满了就不上"这条检查被跳过 ->
# 干将莫邪 连着往三个不同格子拖，每一格都失败，因为根本没有空位。
# 看日志像"落点选得不对"，其实是"压根不该上场"。
rules.reset_match()
rules._active_comp = {"carry": "海月", "core": []}
rules._field = {n: (300 + i * 110, 569) for i, n in enumerate(["a", "b", "c", "d", "e"])}
rules._last_pop_seen = (5, 5)              # 上次读到：满了
st51 = GameState(shop_open=False, population_valid=False, phase="prepare",
                 round_no=6, gold=0, population=0, population_cap=0, hp=20, level=2)
st51.hand_slots = [{"name": "干将莫邪", "x": 844, "y": 663}]
act = rules.decide_deploy(st51)
print(f"[人口读不到+上次满员] 还上场吗={bool(act)}（期望 False）")
if act:
    failures.append("人口读不到时往满编的场上硬塞，白白拖失败")

rules._last_pop_seen = (3, 5)              # 上次读到：没满 -> 该上
if not rules.decide_deploy(st51):
    failures.append("上次读到场面没满，却不肯上场")

print()

# ---- 场景52：N选1 点不掉时要换落点、再不行就放弃（2026-09-23 实战连点 25 次） ----
# 同一组候选（小小的助力/翩跹舞步·新生）被连点 25 次都没关掉，账本刷出 28 条「N选1」。
# 一个固定偏移点不中就永远点不中，所以要换落点；换完还不行就别再点。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
st52 = GameState(phase="card_pick", round_no=9, gold=5, hp=30, level=3)
st52.card_pick = True
st52.card_options = [{"name": "翩跹舞步·新生", "x": 500, "y": 215},
                     {"name": "研读会", "x": 780, "y": 215}]
tries = []
for _ in range(rules.CARD_PICK_MAX_TRIES + 2):
    a = rules.decide_card_pick(st52)
    tries.append(a["try"] if a else None)
print(f"[N选1 连点] 各次的落点档位 {tries}（期望前 {rules.CARD_PICK_MAX_TRIES} 次是 0/1/2，之后 None）")
if tries[:rules.CARD_PICK_MAX_TRIES] != list(range(rules.CARD_PICK_MAX_TRIES)):
    failures.append(f"N选1 重试没有换落点档位: {tries}")
if any(t is not None for t in tries[rules.CARD_PICK_MAX_TRIES:]):
    failures.append(f"N选1 点不掉却一直点下去: {tries}")

print()

# ---- 场景53：落点要铺开两行，不能全堆在一行（2026-09-23 统计） ----
# 按 248 个有锚点的 prepare 帧统计，实际占用是 row0 396 次 / row3 367 次，
# 而 row1 只有 5 次、row2 只有 10 次 —— 我们自己把人堆在两行，然后反复撞车。
from agent.perception.extractors import (OWN_BOARD_DROPS_BACK as _BACK,
                                         OWN_BOARD_DROPS_FRONT as _FRONT)
front_rows = {y for _x, y in _FRONT[:4]}
back_rows = {y for _x, y in _BACK[:4]}
print(f"[落点铺开] 前排前4个落点覆盖 {len(front_rows)} 种 y / 后排 {len(back_rows)} 种")
if len(front_rows) < 2:
    failures.append(f"前排前 4 个落点全挤在一行: {_FRONT[:4]}")
if len(back_rows) < 2:
    failures.append(f"后排前 4 个落点全挤在一行: {_BACK[:4]}")

print()

# ---- 场景54：优先选"这一局从没见过有人"的格子（2026-09-23 统计） ----
# 观测会漏检，漏检的格子和真空格长得一样。实测失败最多的落点永远是
# **占用观测次数最高的那几格** —— 说明我们总往"曾经有人"的格子上放。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._seen_occupied = {(617, 372), (672, 430)}      # 这两格本局见过有人
st54 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=5, gold=0, population=1, population_cap=6, hp=50, level=2)
st54.hand_slots = [{"name": "牛魔", "x": 800, "y": 663}]    # 坦克 -> 前排
act = rules.decide_deploy(st54)
print(f"[避开见过有人的格] 落点 {act and act['dst']}（期望不是 617,372 / 672,430）")
if not act:
    failures.append("有空位却不上场")
elif tuple(act["dst"]) in {(617, 372), (672, 430)}:
    failures.append(f"优先选了本局见过有人的格子: {act['dst']}")

# **不是硬排除**：虚格排完了还得回来试（人会被卖/阵亡，曾经有人≠现在有人）
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
from agent.perception.extractors import OWN_BOARD_DROPS_FRONT as _F2
rules._seen_occupied = {tuple(c) for c in _F2}       # 前排每一格都见过有人
st54.hand_slots = [{"name": "牛魔", "x": 800, "y": 663}]
if not rules.decide_deploy(st54):
    failures.append("所有格子都见过有人时就完全不上场了（应该照样试）")

print()

# ---- 场景55：合成前先看那一格有没有人（2026-09-23 自评报告剩下的最后一项） ----
# 合成落点是记账位，而记账会过期（阵亡/觉醒回手/落点本来就记错）。
# 拖拽失败已清零之后，「合成失败」还剩 4 次，全是往空格子上拖。
# **不能因为观测不到就删记录**（观测会漏检）：先等两帧，之后照拖，由回执判。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"曜": (617, 372)}
st55 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=6, gold=0, population=1, population_cap=6, hp=40, level=3)
st55.hand_slots = [{"name": "曜", "x": 700, "y": 663}]
st55.unit_anchors = {"529,372": 1}          # 画面上只有别的格子有人

a1 = rules.decide_upgrade(st55)
print(f"[第一帧观测不到] {'照样合成' if a1 else '先不合成'}（期望：先不合成）")
if a1:
    failures.append("记账位画面上没人，却照样往空格子上合成")
# **2026-09-23 改了规则**：没看到只"等两帧"，**绝不因此删记录**（徽章漏检 ~1/3，
# 那局账本因此从 7 个人掉到 2 个）。等满两帧照样去合成，由拖拽回执判真假。
a2 = rules.decide_upgrade(st55)
print(f"[第二帧仍观测不到] {'合成' if a2 else '再等'}，场上还记着曜={('曜' in rules._field)}（期望：再等，记录保留）")
if "曜" not in rules._field:
    failures.append("只因为观测不到就删了记录（徽章会漏检，不能当成'不在'）")
a3 = rules.decide_upgrade(st55)
print(f"[第三帧] {a3 and (a3['type'], a3.get('dst'))}（期望：照样往 617,372 合成，让拖拽回执去判）")
if not (a3 and tuple(a3.get("dst", ())) == (617, 372)):
    failures.append("等满两帧后没有去合成（应该交给拖拽回执判断）")

# 观测到那一格有人时，照常合成
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"曜": (617, 372)}
st55.unit_anchors = {"617,372": 1}
if not rules.decide_upgrade(st55):
    failures.append("画面上那格明明有人，却不肯合成")

print()

# ---- 场景56：装备/效果牌也要避开"本局已拖失败多次"的格子（2026-09-23 实战） ----
# `_cell_fails` 本来只用在上场的落点筛选里，而卡牌目标走的是"主力记账位"，
# 完全绕过了它 -> (672,430) 一格连续失败 **13 次**。
# 按牌记的止损只管得住"同一张牌"，管不住"同一个坑"。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
rules._carry_piece = "曜"
rules._field = {"曜": (672, 430), "牛魔": (529, 372)}
rules._cell_fails = {(672, 430): rules.CELL_FAIL_LIMIT}     # 这一格已经废了
st56 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=7, gold=0, population=2, population_cap=6, hp=40, level=3)
st56.unit_anchors = {"672,430": 1, "529,372": 1}
st56.hand_slots = [{"name": "圣者法典", "x": 500, "y": 663}]
act = rules.decide_use_cards(st56)
print(f"[避开已废的格] 装备落点 {act and act['dst']}（期望不是 672,430）")
# **没有替代格时要干脆不打**：_carry_pos 在找不到替代格时仍会把废格返回来。
rules._field = {"曜": (672, 430)}                # 场上只剩主力，且那格已废
st56.unit_anchors = {"672,430": 1}
st56.hand_slots = [{"name": "2阶效果锦囊", "x": 500, "y": 663}]
if rules.decide_use_cards(st56):
    failures.append("落点已废、又没有替代格，却还是把牌打出去了")
if not act:
    failures.append("因为主力那格废了就完全不贴装备")
elif tuple(act["dst"]) == (672, 430):
    failures.append(f"装备还是往本局已失败多次的格子送: {act['dst']}")

print()

# ---- 场景57：自用牌也要"避开废格 + 优先观测到有人的格子"（2026-09-23） ----
# 同一条判据今晚在三条路上各漏过一次（上场落点 / 装备效果牌目标 / 自用牌目标），
# 每次复发的样子都一样：一张牌反复撞同一个坑十几次。现在三条路共用 `_hero_cell`。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._carry_piece = "曜"
rules._field = {"曜": (617, 372), "牛魔": (529, 372)}
rules._cell_fails = {(617, 372): rules.CELL_FAIL_LIMIT}      # 主力那格已废
st57 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=8, gold=0, population=2, population_cap=6, hp=30, level=3)
st57.unit_anchors = {"529,372": 1}        # 画面上只看到牛魔那格有人
pos = rules._any_hero_pos(st57)
print(f"[自用牌落点] {pos}（期望 529,372：避开废格、且画面上确实有人）")
if pos != (529, 372):
    failures.append(f"自用牌落点没避开废格/没选观测到有人的格: {pos}")

# 场上格子全废 -> 这一帧别打自用牌
rules._cell_fails = {(617, 372): rules.CELL_FAIL_LIMIT, (529, 372): rules.CELL_FAIL_LIMIT}
if rules._any_hero_pos(st57) is not None:
    failures.append("场上格子全废了却还给自用牌一个落点")

print()

# ---- 场景58：满编 + 手里压着一堆上不去的英雄 -> 别再买英雄（2026-09-23 实战） ----
# 满编 6/6 之后继续买，买了 12 张，结束时手里压着 6 个上不了场的英雄。
# 那些能量本该去升级（开人口上限和高品阶池）。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": ["牛魔"]}
rules._field = {n: (300 + i * 110, 569) for i, n in
                enumerate(["牛魔", "b", "c", "d", "e", "f"])}
rules._last_pop_seen = (6, 6)
st58 = GameState(shop_open=True, population_valid=True, phase="prepare",
                 round_no=8, gold=24, population=6, population_cap=6, hp=20, level=3)
st58.hand_slots = [{"name": n, "x": 400 + i * 90, "y": 663}
                   for i, n in enumerate(["张良", "明世隐", "公孙离"])]
st58.shop = [ShopSlot(0, "廉颇", 3, "逐鹿"), ShopSlot(1, "牛魔", 3, "河洛"),
             ShopSlot(2, "曜", 3, "逐鹿")] + [ShopSlot(3, None, 3, "")]
print(f"[手里压着3个上不去的] 杂牌廉颇={rules._worth_buying(st58.shop[0], st58)}"
      f" / 合成材料牛魔={rules._worth_buying(st58.shop[1], st58)}"
      f" / 主C曜={rules._worth_buying(st58.shop[2], st58)}")
if rules._worth_buying(st58.shop[0], st58):
    failures.append("满编且手牌堆积时还买杂牌英雄")
if not rules._worth_buying(st58.shop[1], st58):
    failures.append("同名合成材料被误拦（它不占人口）")
if not rules._worth_buying(st58.shop[2], st58):
    failures.append("主C被误拦")

print()

# ---- 场景59：泄压阀要衰减，不能整份清空（2026-09-23 实战） ----
# 满编时 free() 经常找不到候选 -> 泄压阀反复触发 -> 废格记忆一次次清零 ->
# (1027,572) 被反复选中、失败 4 次，而阈值明明是 2。
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
from agent.perception.extractors import OWN_BOARD_DROPS_BACK as _B2
rules._field = {}
rules._cell_fails = {tuple(c): rules.CELL_FAIL_LIMIT for c in _B2}
rules._cell_fails.update({tuple(c): rules.CELL_FAIL_LIMIT
                          for c in __import__("agent.perception.extractors",
                                              fromlist=["x"]).OWN_BOARD_DROPS_FRONT})
before = dict(rules._cell_fails)
st59 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=7, gold=0, population=0, population_cap=6, hp=30, level=3)
st59.hand_slots = [{"name": "狄仁杰", "x": 800, "y": 663}]   # 3 阶：第 7 回合的 1 阶按新规则不上场，测不到泄压阀
rules.decide_deploy(st59)          # 所有格子都废了 -> 触发泄压阀
after = rules._cell_fails
print(f"[泄压阀] 衰减前 {len(before)} 格，衰减后 {len(after)} 格，"
      f"某格次数 {before.get((617,372))} -> {after.get((617,372))}")
if not after:
    failures.append("泄压阀把废格记忆整份清空了（应该只衰减一档）")
elif after.get((617, 372)) != rules.CELL_FAIL_LIMIT - 1:
    failures.append(f"衰减幅度不对: {after.get((617,372))}")

print()

# ---- 场景60：合成也要避开废格（_cell_dead 漏掉的第四条路径，2026-09-23） ----
# 合成落点必须是那个英雄的记账位（不能换格），所以格子废了只能撤销记录重新上场。
# 实测：干将莫邪 往 (300,569) 和 (839,502) 各合成失败 4 次，而阈值是 2。
rules.reset_match()
rules._active_comp = {"carry": "干将莫邪", "core": []}
rules._field = {"干将莫邪": (300, 569)}
rules._cell_fails = {(300, 569): rules.CELL_FAIL_LIMIT}
st60 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=7, gold=0, population=1, population_cap=6, hp=30, level=3)
st60.hand_slots = [{"name": "干将莫邪", "x": 700, "y": 663}]
st60.unit_anchors = {"300,569": 1}        # 画面上那格"有人"，但格子已废
st60.vision_units = {"529,372": "hero"}   # 视觉这一帧跑了、D-FINE 在那格没看到人（2026-09-30：视觉没跑时先等，不撤销）
act = rules.decide_upgrade(st60)
print(f"[合成落点已废] 动作={act}（期望 None：撤销记录、走上场）")
if act:
    failures.append("合成还是往本局已失败多次的格子拖")
if "干将莫邪" in rules._field:
    failures.append("格子废了却没撤销记录")

print()

# ---- 场景61：觉醒卡在手 -> 撤销旧记录、解除搁置、第一个上场（2026-09-23 实战截图） ----
# 实战：干将莫邪觉醒回手，我们没认出来，当合成材料往旧格子拖 3 次失败 ->
# 被"试够 3 次就搁置"挡住，重新上场也一起被挡，名额让给了公孙离/弈星。
rules.reset_match()
rules._active_comp = {"carry": "干将莫邪", "core": []}
rules._field = {"干将莫邪": (839, 502), "蔡文姬": (276, 501)}
rules._cell_owner = {(839, 502): "干将莫邪", (276, 501): "蔡文姬"}
st61 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=8, gold=16, population=4, population_cap=6, hp=30, level=3)
rules._round_seen = 8
rules._parked_cards.add((8, "干将莫邪"))             # 合成失败攒出来的搁置
st61.hand_slots = [{"name": "弈星", "x": 540, "y": 663},
                   {"name": "公孙离", "x": 640, "y": 663},
                   {"name": "干将莫邪", "x": 740, "y": 663},
                   {"name": "干将莫邪", "x": 840, "y": 663, "awakened": True}]
rules.sync_awakened(st61)
if "干将莫邪" in rules._field:
    failures.append("手里有觉醒卡，场上的旧记录却没撤销")
if rules._parked("干将莫邪"):
    failures.append("觉醒卡还被合成失败的搁置挡着")
act = rules.decide_deploy(st61)
check_act("觉醒卡优先上场", act, "drag_card")
if not act or act.get("piece") != "干将莫邪" or act.get("merge"):
    failures.append(f"觉醒卡没有第一个上场（或被当成合成）: {act}")
elif act.get("src") != (840, 663) and act.get("src") != [840, 663]:
    failures.append(f"拖的不是那张觉醒卡: src={act.get('src')}")

print()

# ---- 场景62：人口满 + 手里有觉醒卡 -> 卖最弱的给它腾位 ----
rules.reset_match()
rules._active_comp = {"carry": "干将莫邪", "core": []}
rules._field = {"蔡文姬": (276, 501), "项羽": (617, 372), "少司缘": (1027, 572),
                "公孙离": (555, 569), "弈星": (499, 500), "东皇太一": (672, 430)}
st62 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=8, gold=16, population=6, population_cap=6, hp=30, level=3)
st62.hand_slots = [{"name": "李元芳", "x": 840, "y": 663, "awakened": True}]
rules.sync_awakened(st62)
acts = rules.decide_all(st62)
print(f"[人口满+觉醒卡] {acts}")
if not any(a["type"] in ("sell_unit", "sell") for a in acts):
    failures.append(f"人口满时没有给觉醒卡腾位: {acts}")
if rules.unit_value("李元芳") <= max(rules.unit_value(n) for n in ("蔡文姬", "项羽")):
    failures.append("觉醒卡的估值不高于场上的普通杂牌")

print()

# ---- 场景63：阵容不能跟着"每帧手牌抖动"来回切（2026-09-23：阵容来回切） ----
# 实战：「干将无敌流」↔「是我帮助了我」一局切 3 次，同一回合里就翻了一次。
rules.reset_match()
saved_comps = rules.comps
A = {"name": "干将流", "carry": "干将莫邪", "core": ["干将莫邪", "弈星", "张良"],
     "factions": {}, "faction": "", "use_num": 7519}
B = {"name": "公孙离流", "carry": "公孙离", "core": ["公孙离", "蔡文姬", "项羽", "李元芳"],
     "factions": {}, "faction": "", "use_num": 1002}
rules.comps = [A, B]
def frame(rnd, hand):
    st = GameState(shop_open=False, population_valid=True, phase="prepare",
                   round_no=rnd, gold=0, population=3, population_cap=6, hp=40, level=3)
    st.hand_slots = [{"name": n, "x": 300 + 100 * i, "y": 663} for i, n in enumerate(hand)]
    rules.resolve_comp(st)
    return rules._active_comp["name"] if rules._active_comp else None
first = frame(5, ["干将莫邪", "弈星"])
rules._field = {"干将莫邪": (529, 372)}                  # 主C 上场了
flips = []
# 同一回合里手牌读数来回抖（B 的核心牌时有时无）
for hand in (["公孙离", "蔡文姬", "项羽", "李元芳"], ["弈星"], ["公孙离", "蔡文姬", "项羽", "李元芳"]):
    flips.append(frame(6, hand))
print(f"[阵容抖动] 初选 {first}，同回合抖动后 {flips}（期望一直是 干将流）")
if first != "干将流" or any(f != "干将流" for f in flips):
    failures.append(f"阵容跟着单帧手牌抖动换了: {first} -> {flips}")
# 主C 不在场、挑战者**连续两个回合**都明显更好 -> 才换
rules._field = {}
strong = ["公孙离", "蔡文姬", "项羽", "李元芳"]
r7 = frame(7, strong)
r8 = frame(8, strong)
print(f"[阵容确实该换] 第7回合 {r7} -> 第8回合 {r8}（期望 干将流 -> 公孙离流）")
if r7 != "干将流" or r8 != "公孙离流":
    failures.append(f"挑战者连续两回合都更好却没换（或换早了）: {r7} -> {r8}")
# 换完 2 个回合内不再换回去
r9 = frame(9, ["干将莫邪", "弈星", "张良", "干将莫邪"])
print(f"[冷却期] 第9回合 {r9}（期望仍是 公孙离流）")
if r9 != "公孙离流":
    failures.append(f"刚换完阵容又换回去了: {r9}")
rules.comps = saved_comps
rules.reset_match()

print()

# ---- 场景64：阿离的专属牌不能烂在手里（2026-09-23：阿离专属天赋牌一直没放到身上） ----
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._carry_piece = "曜"
rules._field = {"曜": (617, 372), "牛魔": (672, 430)}
st64 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=6, gold=0, population=2, population_cap=6, hp=40, level=3)
st64.hand_slots = [{"name": "加速成长", "x": 500, "y": 663}]
act = rules.decide_use_cards(st64)
check_act("加速成长(天赋牌+5级)给主C", act, "drag_card")
if not act or act.get("piece") != "加速成长" or tuple(act["dst"])[0] != 617:
    failures.append(f"加速成长没有拖给主C: {act}")
if not rules._board_work_pending(st64):
    failures.append("手里有加速成长，却说棋盘上没活（会先去开商店）")
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._carry_piece = "曜"
rules._field = {"曜": (617, 372)}
st64.hand_slots = [{"name": "翩跹舞步", "x": 420, "y": 663}]     # 后缀没读出来
act = rules.decide_use_cards(st64)
check_act("后缀读不全的翩跹舞步", act, "drag_card")
if not act or act.get("piece") != "翩跹舞步":
    failures.append(f"翩跹舞步（后缀没读出来）没被打出去: {act}")
# 反例：移除英雄的天赋牌不能被当成加成牌拖给主C
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._carry_piece = "曜"
rules._field = {"曜": (617, 372)}
st64.hand_slots = [{"name": "体面离场", "x": 420, "y": 663}]
act = rules.decide_use_cards(st64)
print(f"[体面离场+只有主C] {act}（期望 None：没有消耗件可移除）")
if act:
    failures.append(f"移除类天赋牌被拖给了主C: {act}")

print()

# ---- 场景65：动态选主C（2026-09-23：没有切换主C） ----
rules.reset_match()
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json")
def carry_after(field, levels, rnd, hand=()):
    st = GameState(shop_open=False, population_valid=True, phase="prepare",
                   round_no=rnd, gold=0, population=len(field), population_cap=6, hp=40, level=3)
    st.unit_levels = {f"{p[0]},{p[1]}": lv for p, lv in levels.items()}
    st.hand_slots = [{"name": n, "x": 400 + 90 * i, "y": 663} for i, n in enumerate(hand)]
    rules._field = dict(field)
    rules.choose_carry(st)
    return rules._carry_piece
rules._carry_piece = "曜"
got = carry_after({"曜": (617, 372), "百里守约": (300, 569)}, {(617, 372): 3, (300, 569): 2}, 4)
print(f"[换主C] 1阶曜(3级) vs 2阶百里守约(2级) 第4回合 -> {got}（期望 百里守约）")
if got != "百里守约":
    failures.append(f"2 阶输出没能取代开局的 1 阶主C: {got}")
rules.reset_match(); rules._carry_piece = "干将莫邪"
got = carry_after({"干将莫邪": (617, 372), "百里守约": (300, 569)}, {(617, 372): 60, (300, 569): 1}, 5)
print(f"[换主C] 60级干将 vs 1级守约 第5回合 -> {got}（期望 还是干将莫邪：等级攒着以后用红叶染锋转）")
if got != "干将莫邪":
    failures.append(f"已经练到 60 级的前期主C 被一张 1 级新卡顶掉了: {got}")
rules.reset_match(); rules._carry_piece = "干将莫邪"
got = carry_after({"干将莫邪": (617, 372), "狄仁杰": (300, 569)}, {(617, 372): 12, (300, 569): 6}, 8)
print(f"[换主C] 第8回合 12级干将(1阶) vs 6级狄仁杰(3阶) -> {got}（期望 狄仁杰：1阶过了第6回合要让位）")
if got != "狄仁杰":
    failures.append(f"第 8 回合还让 1 阶当主C: {got}")
rules.reset_match(); rules._carry_piece = "百里守约"
got = carry_after({"百里守约": (300, 569), "公孙离": (617, 372)}, {(300, 569): 5, (617, 372): 6}, 6)
print(f"[换主C] 守约5级 vs 公孙离6级（差不多）-> {got}（期望 不换）")
if got != "百里守约":
    failures.append(f"分差不大也换主C（会来回横跳）: {got}")
# 2026-09-23 实战：场上唯一的 3 阶是姬小满（前排、技能是经济效果），光凭品阶就被选成主C。
# 它在 455 套推荐阵容里一次都没当过主C —— 不该换过去。
rules.reset_match(); rules._carry_piece = "曜"
got = carry_after({"曜": (529, 372), "姬小满": (556, 430), "牛魔": (617, 372)},
                  {(529, 372): 10, (556, 430): 1, (617, 372): 3}, 4)
print(f"[换主C] 10级曜 vs 1级姬小满(3阶功能牌) 第4回合 -> {got}（期望 还是曜）")
if got != "曜":
    failures.append(f"把从没当过主C 的功能牌选成了主C: {got}")
# N选1：红叶染锋必拿
rules.reset_match()
st = GameState(phase="card_pick", round_no=5, card_pick=True)
st.card_options = [{"name": "翩跹舞步·流动", "x": 400, "y": 300}, {"name": "红叶染锋", "x": 640, "y": 300},
                   {"name": "翩跹舞步·余韵", "x": 880, "y": 300}]
act = rules.decide_card_pick(st)
print(f"[N选1] {[o['name'] for o in st.card_options]} -> {act and act.get('name', act)}")
if not act or "红叶染锋" not in str(act):
    failures.append(f"N 选 1 没拿红叶染锋: {act}")
rules._field = {}; rules._carry_piece = ""
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json")

print()

# ---- 场景66：第 1 回合 6 能量别只买 1 张就关店（2026-09-23 实战：第 3 回合场上还是 1/5） ----
rules.reset_match(); rules._active_comp = {"carry": "干将莫邪", "core": []}
rules._round_seen = 1
st66 = GameState(shop_open=True, population_valid=True, phase="prepare",
                 round_no=1, gold=3, population=0, population_cap=5, hp=60, level=1)
st66.hand_slots = [{"name": "干将莫邪", "x": 700, "y": 663}]            # 已经买了 1 张
st66.shop = [ShopSlot(0, "牛魔", 3, "河洛"), ShopSlot(1, "李元芳", 3, "河洛")]
acts = rules.decide_all(st66)
print(f"[第1回合 空场+手里1个+还剩3能量] {acts}（期望：再买一张）")
if not any(a.get("type") == "buy_piece" for a in acts):
    failures.append(f"第 1 回合只买 1 张就关店了: {acts}")

print()

# ---- 场景67：点开看场确认过的格子，不能再被"失败次数"撤销（2026-09-23 实战：弈星来回三轮） ----
rules.reset_match(); rules._active_comp = {"carry": "米莱狄", "core": []}
rules._field = {"弈星": (390, 502)}
rules._cell_fails[(390, 502)] = rules.CELL_FAIL_LIMIT        # 那一格被贴牌失败"毒死"了
st67 = GameState(shop_open=False, population_valid=True, phase="prepare",
                 round_no=5, gold=0, population=3, population_cap=5, hp=50, level=2)
st67.hand_slots = [{"name": "弈星", "x": 700, "y": 663}]
rules.confirm_unit("弈星", (390, 502))                       # 点开确认：就是它
act = rules.decide_upgrade(st67)
print(f"[点开确认后合成] {act}（期望：往 390,502 合成，不撤销记录）")
if not act or tuple(act.get("dst", ())) != (390, 502) or "弈星" not in rules._field:
    failures.append(f"点开确认过的格子还是被当成废格: {act} field={rules._field}")

print()

# ---- 场景68~71：2026-09-23 TensorRT 那一局暴露的四个决策层问题 ----
# 68. 天赋三选一被收起（左侧「天赋未领取」）：必须去点开，不能挂一整回合
rules.reset_match()
st68 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=15,
                 gold=36, population=7, population_cap=7, hp=5, level=6)
st68.talent_pending = [86, 548]
act = (rules.decide_all(st68) or [None])[0]
print(f"[天赋未领取] {act}（期望 open_talent @86,548）")
if not act or act["type"] != "open_talent" or (act["x"], act["y"]) != (86, 548):
    failures.append(f"「天赋未领取」没去点开: {act}")
# 69. 满级：不再想升级（原来升级费粘着 18，每帧"想升级找不到按钮"，36 能量一分没花）
rules.reset_match()
st69 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=15,
                 gold=36, population=7, population_cap=7, hp=20, level=6,
                 level_cost=18, level_btn=(60, 546))
st69.lord_maxed = True
act = rules.decide_economy(st69)
print(f"[满级] 升级决策={act}（期望 None）")
if act:
    failures.append(f"满级了还要升级: {act}")
# 70. 拖牌前抓牌校验连续找不到这张牌：3 次后本回合搁置（原来整回合卡死在这一个动作上）
rules.reset_match()
rules._round_seen = 12
for _ in range(3):
    rules.note_src_missing("刘邦")
print(f"[抓牌校验失败 3 次] 刘邦被搁置={rules._parked('刘邦')}（期望 True）")
if not rules._parked("刘邦"):
    failures.append("抓牌校验连续失败 3 次仍没搁置，会整回合卡在同一个动作上")
# 71. 拍卖：已经拍到一件（顶部「已获得拍品」）就不再出价
rules.reset_match()
st71 = GameState(phase="auction", round_no=4, gold=10)
st71.auction = AuctionState(lots=[AuctionLot(index=0, piece="亚连", top_bid=1, sold=False, x=225, y=230)],
                            my_gold=10, won=True)
act = rules.decide_auction(st71)
print(f"[拍卖已获得] {act}（期望 None）")
if act:
    failures.append(f"已经拍到一件还在出价: {act}")

# 72. 一格只能站一个人：海诺真的上场到弈星记账的那一格后，弈星的旧记录要撤掉，
#     不能再把那格当成"弈星的同名卡"去合成（2026-09-23）
rules.reset_match()
rules._field = {"弈星": (433, 569)}
rules.note_drag_result({"type": "drag_card", "piece": "海诺", "dst": (433, 569)}, True)     if hasattr(rules, "note_drag_result") else rules.relocate_unit("海诺", (433, 569))
print(f"[同格冲突] 账本={rules._field}（期望只剩 海诺@433,569）")
if "弈星" in rules._field:
    failures.append(f"同一格记了两个人: {rules._field}")
# 73. 「天赋未领取」一回合最多试 3 次，之后不再挡住别的动作
rules.reset_match()
tries = 0
for _ in range(5):
    acts = rules.decide_all(st68)
    tries += bool(acts and acts[0]["type"] == "open_talent")
print(f"[天赋未领取上限] 5 帧里打开了 {tries} 次（期望 3）")
if tries != 3:
    failures.append(f"「天赋未领取」每回合应只试 3 次，实际 {tries}")

# 74. 装备满 3 件的人不再作为基础装备目标：单帧没读到件数也要记得（2026-09-23 马可波罗）
rules.reset_match()
rules._active_comp = {"carry": "马可波罗", "core": []}
rules._carry_piece = "马可波罗"
rules._field = {"马可波罗": (276, 501), "李白": (555, 569)}
st74 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=12,
                 gold=10, population=2, population_cap=7, hp=20, level=4)
st74.equip_counts = {"276,501": 3}
rules._note_equip_seen(st74)
st74.equip_counts = {}                       # 下一帧没读到件数
tgt = rules._equip_target(st74) if hasattr(rules, "_equip_target") else None
print(f"[满 3 件不再贴] 目标={tgt}（期望不是 马可波罗）")
if tgt and tgt[0] == "马可波罗":
    failures.append(f"马可波罗已满 3 件，单帧读不到件数就又选了他: {tgt}")
rules.note_equip_rejected((555, 569))
tgt = rules._equip_target(st74)
# 一次没弹四选一**不算满**（2026-09-23 第八局：特效挡住、牌根本没拖出去，就被记成满的）
print(f"[没弹四选一 1 次 -> 还不算满] 目标={tgt}（期望 李白）")
if not tgt or tgt[0] != "李白":
    failures.append(f"一次没弹四选一就当成满了: {tgt}")
rules.note_equip_rejected((555, 569))
tgt = rules._equip_target(st74)
print(f"[没弹四选一 2 次 -> 当满] 目标={tgt}（期望 None：两个人都满了）")
if tgt:
    failures.append(f"同一格两次没弹四选一的人还在被选: {tgt}")

# 75. 合成"回执成功"但牌弹回手里：同一格一回合最多试 3 次，之后搁置（不能被"成功"清零）
rules.reset_match()
rules._round_seen = 7
rules._field = {"弈星": (1027, 572)}
act75 = {"type": "drag_card", "piece": "弈星", "merge": True, "src": (631, 662), "dst": (1027, 572)}
for _ in range(3):
    rules.note_executed(dict(act75))
    rules.confirm_drag(dict(act75), True)
print(f"[假成功的合成] 弈星被搁置={rules._parked('弈星')}（期望 True）")
if not rules._parked("弈星"):
    failures.append("合成'成功'3 次牌却没少，还在无限重试（成功把计数清零了）")

# 76. 拍卖：读不到出价（这种界面拍品下面不显示出价，全是 0）时，出过一次价就不再出
#     —— 不能在几张拍品之间来回出价、同一张自己抬自己（2026-09-23）
rules.reset_match()
st76 = GameState(phase="auction", round_no=3, gold=9)
lots76 = [AuctionLot(index=0, piece="姬小满", top_bid=0, x=224, y=255),
          AuctionLot(index=1, piece="猪八戒", top_bid=0, x=430, y=233)]
st76.auction = AuctionState(lots=lots76, my_gold=9)
bids76 = []
for _ in range(5):
    a = rules.decide_auction(st76)
    if a and not a.get("select_only"):
        bids76.append(a["piece"])
print(f"[拍卖不自抬] 5 帧里出价 {bids76}（期望只出 1 次）")
if len(bids76) != 1:
    failures.append(f"读不到出价时在反复出价/自己抬价: {bids76}")
# 看到被别人超过了，才追价
lots76[0].top_bid = 3 if bids76 and bids76[0] == "姬小满" else 0
lots76[1].top_bid = 3 if bids76 and bids76[0] == "猪八戒" else 0
lots76[0].top_bidder = lots76[1].top_bidder = "别人"
a = rules.decide_auction(st76)
print(f"[被超过后重新出价] {a and (a['piece'], a['bid'])}（期望：追那张出 4，或改拍更便宜的一张）")
if not a or (a["piece"] == bids76[0] and a["bid"] != 4):
    failures.append(f"被别人超过后没有重新出价: {a}")

# 77. 装备的穿戴限制：日渊「限定远程英雄」不能给近战的曜（2026-09-23 拖了 5 次都不生效）
rules.reset_match()
rules._carry_piece = "曜"
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"曜": (718, 371), "公孙离": (839, 502)}
st77 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=5,
                 gold=5, population=2, population_cap=5, hp=50, level=3)
tgt = rules._equip_target(st77, "日渊")
print(f"[限定远程] 日渊给 {tgt and tgt[0]}（期望 公孙离，不能给近战的曜）")
if not tgt or tgt[0] != "公孙离":
    failures.append(f"日渊（限定远程）给了 {tgt}")
rules._field = {"曜": (718, 371)}
tgt = rules._equip_target(st77, "日渊")
print(f"[限定远程·场上只有近战] {tgt}（期望 None，先留着）")
if tgt:
    failures.append(f"场上只有近战英雄还要穿日渊: {tgt}")

# 78. 升级后多出 1 个空位，不能一口气买一堆英雄（2026-09-23：每次到 3 级就大量购买英雄）
rules.reset_match()
rules._field = {"孙膑": (300, 569), "曜": (556, 430), "蒙恬": (529, 372), "墨子": (783, 438), "虞姬": (955, 502)}
st78 = GameState(shop_open=True, population_valid=True, phase="prepare", round_no=8,
                 gold=15, population=5, population_cap=6, hp=30, level=3)
st78.shop = [ShopSlot(index=i, piece=n, cost=3) for i, n in enumerate(["阿轲", "公孙离", "云中君", "干将莫邪"])]
bought = []
for _ in range(4):
    a = rules.decide_buy(st78)
    if not a:
        break
    rules.note_executed(a)
    bought.append(a["piece"])
    st78.hand_slots = st78.hand_slots + [{"name": a["piece"], "x": 600 + 80 * len(bought), "y": 663}]
    st78.hand = [s["name"] for s in st78.hand_slots]
    st78.gold -= 3
print(f"[升级后买人] 空 1 格，买了 {bought}（期望最多 1 张新英雄）")
if len(bought) > 1:
    failures.append(f"只空 1 格却买了 {len(bought)} 张英雄: {bought}")

# 79. 免费刷新：商店里没想要的就用掉，不受"给拍卖留钱"限制（2026-09-23）
rules.reset_match()
rules._field = {"孙膑": (300, 569), "曜": (556, 430), "蒙恬": (529, 372)}
st79 = GameState(shop_open=True, population_valid=False, phase="prepare", round_no=7,
                 gold=4, population=6, population_cap=6, hp=30, level=4)
st79.free_refresh = 1
st79.shop = [ShopSlot(index=i, piece=n, cost=3, sold=True) for i, n in enumerate(["盾山", "牛魔", "大乔", "蔡文姬"])]
a = rules.decide_refresh(st79)
print(f"[免费刷新] {a}（期望 refresh_shop，不带 cost）")
if not a or a["type"] != "refresh_shop" or a.get("cost"):
    failures.append(f"有免费刷新却没用: {a}")

# 80. 古币「指定1名【河洛】英雄」：只能给河洛的人（2026-09-23 拖给诸葛亮失败两次）
rules.reset_match()
rules._carry_piece = "诸葛亮"
rules._active_comp = {"carry": "诸葛亮", "core": []}
rules._field = {"诸葛亮": (276, 501), "苏烈": (617, 372)}
st80 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=9,
                 gold=2, population=2, population_cap=7, hp=40, level=5)
st80.hand_slots = [{"name": "古币", "x": 721, "y": 663}]
st80.hand = ["古币"]
a = rules.decide_use_cards(st80)
print(f"[古币限河洛] {a and (a.get('piece'), a.get('target'))}（期望 古币 -> 苏烈）")
if not a or a.get("target") != "苏烈":
    failures.append(f"古币没给河洛英雄: {a}")

# 81. 觉醒回手（人口少了 + 牌在手上）→ 人口满时要为它腾位，哪怕要换下阵容核心
#     （2026-09-23 第七局：有英雄觉醒了，却先上了其他英雄，导致最强的觉醒英雄上不了场）
rules.reset_match()
rules._carry_piece = "孙悟空"
rules._active_comp = {"carry": "孙悟空", "core": ["刘邦", "韩信", "吕布"]}
rules._field = {"孙悟空": (818, 374), "诸葛亮": (276, 501), "吕布": (556, 430), "敖隐": (955, 502),
                "刘邦": (672, 430), "韩信": (718, 371), "米莱狄": (1027, 572)}
st81 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=13,
                 gold=18, population=6, population_cap=7, hp=16, level=6)
# 觉醒要有直接证据（颜色候选 + 点开确认后打上 awakened 标记）；只凭"人口少了 + 同名在手"不算（可能是合成材料）
st81.hand_slots = [{"name": "枫叶舞", "x": 480, "y": 663}, {"name": "米莱狄", "x": 800, "y": 663, "awakened": True}]
st81.hand = ["枫叶舞", "米莱狄"]
rules.sync_field(st81)
ok_awk = "米莱狄" in rules._ever_awakened and "米莱狄" not in rules._field
# 另一个英雄随后补满了人口（第七局就是这样被抢走位置的）
rules._field["大司命"] = (617, 372)
st81.population = 7
a = rules.decide_sell_unit(st81)
print(f"[觉醒腾位] 认成觉醒={ok_awk}，决策 {a and (a.get('type'), a.get('piece'))}"
      f"（期望卖掉一个非主C 给米莱狄腾位）")
if not ok_awk:
    failures.append("人口少了 + 牌在手上的觉醒回手没记进 _ever_awakened（估值没 +40）")
if not a or a.get("type") != "sell_unit" or a.get("piece") in ("孙悟空", "米莱狄"):
    failures.append(f"觉醒卡在手、人口满，却没腾位: {a}")
else:
    rules.note_executed(a)
    sold_pos = tuple(a["src"])
    st81.population = 6
    d = rules.decide_deploy(st81)
    print(f"[觉醒上场] {d and (d.get('piece'), d.get('dst'))}（期望 米莱狄 -> 刚腾出的 {sold_pos}）")
    if not d or d.get("piece") != "米莱狄" or tuple(d.get("dst", ())) != sold_pos:
        failures.append(f"腾位后没把觉醒卡放进刚空出来的格子: {d}")

# 82. 战斗画面的人口读数不能删人（2026-09-23 第八局：战斗里读到 1/7，下回合把主C 蒙犽删了）
rules.reset_match()
rules._field = {"蒙犽": (672, 430), "西施": (1027, 572), "猪八戒": (529, 372)}
st82 = GameState(shop_open=False, population_valid=False, phase="battle", round_no=12,
                 gold=20, population=1, population_cap=7, hp=5, level=5)
st82.hand_slots = [{"name": "蒙犽", "x": 618, "y": 663}]
rules.sync_field(st82)
st82.phase = "prepare"
rules.sync_field(st82)                      # 人口不是本帧真读到的 -> 也不能删
print(f"[战斗人口不删人] 场上 {sorted(rules._field)}（期望 蒙犽 还在）")
if "蒙犽" not in rules._field:
    failures.append("人口读数无效时 sync_field 仍把主C 删了")

# 83. 主C 不在场上：效果牌改给场上真在的人，名字按实际收件人（第八局 推演战术 死循环）
rules.reset_match()
rules._carry_piece = "蒙犽"
rules._active_comp = {"carry": "蒙犽", "core": []}
rules._field = {"西施": (1027, 572), "猪八戒": (529, 372)}
st83 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=12,
                 gold=5, population=2, population_cap=7, hp=5, level=5)
st83.hand_slots = [{"name": "特训战术", "x": 900, "y": 663}]
st83.hand = ["特训战术"]
a83 = rules.decide_use_cards(st83)
print(f"[主C 不在场] {a83 and (a83.get('piece'), a83.get('target'), a83.get('dst'))}（期望目标是场上的人）")
if a83 and a83.get("target") not in rules._field:
    failures.append(f"主C 不在场上，效果牌还写着它的名字: {a83}")

# 84. 手牌账：买进的牌叠放后只露出一个字，也能补回来
rules.reset_match()
rules._round_seen = 11
rules.note_executed({"type": "buy_piece", "piece": "蒙犽", "slot": 0, "cost": 3, "round": 11})
st84 = GameState(shop_open=True, population_valid=False, phase="prepare", round_no=11,
                 gold=6, population=7, population_cap=7, hp=5, level=5)
st84.hand_slots = [{"name": "小乔", "x": 700, "y": 663}]
st84.hand_unread = [[520, 663, "蒙"]]
rules.sync_hand_book(st84)
rules.fill_hand_from_book(st84)
got = [s["name"] for s in st84.hand_slots]
print(f"[手牌账补位] {got}（期望含 蒙犽）")
if "蒙犽" not in got:
    failures.append(f"手牌账没把只露出「蒙」的牌补回来: {got}")

# 84b. 枫叶舞（棋手技能牌）不进手牌账、也不会被补到别的位置上（第九局实测）
rules.reset_match()
st84b = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=2,
                  gold=6, population=1, population_cap=5, hp=60, level=2)
st84b.hand_by_layout = True
st84b.hand_slots = [{"name": "枫叶舞", "x": 520, "y": 663}, {"name": "曜", "x": 700, "y": 663}]
rules.sync_hand_book(st84b)
st84b.hand_slots = [{"name": "曜", "x": 700, "y": 663}]
st84b.hand_unread = [[792, 663, ""]]
rules.fill_hand_from_book(st84b)
got = [s["name"] for s in st84b.hand_slots]
print(f"[枫叶舞不进账] {got}（期望没有 枫叶舞）")
if "枫叶舞" in got:
    failures.append(f"手牌账把枫叶舞补到了别的位置: {got}")

# 85. 人口满时商店里的"加等级"效果牌也要买（第八局：精密协作摆着 38 秒没买）
rules.reset_match()
rules._field = {n: (0, i) for i, n in enumerate(["西施", "猪八戒", "孙膑", "鲁班大师", "蒙犽", "李白", "庄周"])}
st85 = GameState(shop_open=True, population_valid=False, phase="prepare", round_no=11,
                 gold=6, population=7, population_cap=7, hp=5, level=5)
st85.shop = [ShopSlot(index=2, piece="精密协作", cost=2)]
b85 = rules.decide_buy(st85)
print(f"[买加等级效果牌] {b85 and b85.get('piece')}（期望 精密协作）")
if not b85 or b85.get("piece") != "精密协作":
    failures.append(f"人口满时没买加等级的效果牌: {b85}")

# 86. 辅助不当主C、上场贴着主C（2026-09-23：辅助基本没有输出能力，尽量放到主C 边上）
rules.reset_match()
rules._field = {"明世隐": (955, 502), "曜": (617, 372)}
st86 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=7,
                 gold=5, population=2, population_cap=6, hp=40, level=3)
rules.choose_carry(st86)
print(f"[辅助不当主C] 主C={rules._carry_piece}（期望 曜）")
if rules._carry_piece == "明世隐":
    failures.append("辅助 明世隐 被选成了主C")
rules._carry_piece = "曜"
st86.hand_slots = [{"name": "蔡文姬", "x": 700, "y": 663}]
st86.hand = ["蔡文姬"]
d86 = rules.decide_deploy(st86)
dist = lambda c: (c[0] - 617) ** 2 + (c[1] - 372) ** 2   # noqa: E731
print(f"[辅助贴主C] {d86 and (d86.get('piece'), d86.get('dst'))}")
from agent.perception.extractors import OWN_BOARD_DROPS_BACK, OWN_BOARD_DROPS_FRONT
free_cells = [c for c in OWN_BOARD_DROPS_BACK + OWN_BOARD_DROPS_FRONT if tuple(c) not in {(955, 502), (617, 372)}]
if not d86 or dist(d86["dst"]) > min(dist(c) for c in free_cells):
    failures.append(f"辅助没放到离主C 最近的空格: {d86}")

# 86b. 主C 在手里时不换主C；枫叶舞不绑辅助（第九局：韩信在手里，主C 换成 1 级辅助明世隐、枫叶舞绑过去）
rules.reset_match()
rules._carry_piece = "韩信"
rules._field = {"明世隐": (955, 502), "苏烈": (617, 372)}
st86b = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=9,
                  gold=5, population=2, population_cap=6, hp=40, level=3)
st86b.hand_slots = [{"name": "韩信", "x": 700, "y": 663}]
rules.choose_carry(st86b)
tgt86 = rules._skill_target()
print(f"[主C在手里不换] 主C={rules._carry_piece}，枫叶舞绑 {tgt86}（期望 韩信 / 不是明世隐）")
if rules._carry_piece != "韩信":
    failures.append(f"主C 只是在手里，却被换成了 {rules._carry_piece}")
if tgt86 == "明世隐":
    failures.append("枫叶舞绑给了辅助")

# 86c. 换主C 收尾：枫叶舞绑在别人身上 -> 卖掉他，然后枫叶舞改绑新主C（2026-09-23）
rules.reset_match()
rules._carry_piece = "狄仁杰"
rules._field = {"狄仁杰": (617, 372), "干将莫邪": (300, 569), "苏烈": (672, 430)}
rules._maple_bound = "干将莫邪"
st86c = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=8,
                  gold=5, population=3, population_cap=6, hp=40, level=4)
a86 = rules.decide_sell_unit(st86c)
print(f"[换主C收尾] {a86 and (a86.get('type'), a86.get('piece'), a86.get('why'))}（期望卖 干将莫邪）")
if not a86 or a86.get("piece") != "干将莫邪":
    failures.append(f"换主C 后没卖掉枫叶舞绑着的人: {a86}")
else:
    rules.note_executed(a86)
    print(f"  卖完后枫叶舞绑定对象 = {rules._maple_bound!r}，枫叶舞目标 = {rules._skill_target()}")
    if rules._maple_bound or rules._skill_target() != "狄仁杰":
        failures.append("卖掉被绑的人后，枫叶舞没转给新主C")

# 86d. 同一回合两个格子上场都失败 -> 推断人口已满，人口读不到时不再上场（第十局 庄周 两回合各白拖两次）
rules.reset_match()
rules._round_seen = 10
rules._field = {n: (0, i) for i, n in enumerate(["蒙犽", "明世隐", "雅典娜", "少司缘", "蒙恬", "老夫子"])}
rules._last_pop_seen = (6, 7)
for dst in ((839, 502), (720, 500)):
    rules.confirm_drag({"type": "drag_card", "piece": "庄周", "src": (650, 662), "dst": dst}, False)
st86d = GameState(shop_open=True, population_valid=False, phase="prepare", round_no=10,
                  gold=5, population=6, population_cap=7, hp=20, level=5)
st86d.hand_slots = [{"name": "扁鹊", "x": 700, "y": 663}]
d86d = rules.decide_deploy(st86d)
print(f"[两格都失败→当满] {d86d}（期望 None）")
if d86d:
    failures.append(f"两个格子都上场失败后还在上场: {d86d}")

# 86f. 同名牌在手里（合成材料）+ 人口读数少于账本：不能把场上那个删掉（第十一局 裴擒虎）
rules.reset_match()
rules._carry_piece = "裴擒虎"
rules._field = {"裴擒虎": (839, 502), "曜": (617, 372), "牛魔": (451, 433), "白起": (529, 372),
                "蒙恬": (556, 430), "貂蝉": (672, 430), "孙膑": (300, 569)}
st86f = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=6,
                  gold=5, population=5, population_cap=6, hp=60, level=3)
st86f.hand_slots = [{"name": "裴擒虎", "x": 650, "y": 663}]
rules.sync_field(st86f)
print(f"[合成材料不算觉醒] 场上有裴擒虎={'裴擒虎' in rules._field} 主C={rules._carry_piece}")
if "裴擒虎" not in rules._field or rules._carry_piece != "裴擒虎":
    failures.append("手里的合成材料让场上的主C 被当成觉醒回手删掉了")

# 86e. 作为"吃整备效果"上场的英雄后来成了主C，不能按杂牌卖掉（第十一局 裴擒虎）
rules.reset_match()
rules._field = {"裴擒虎": (839, 502), "曜": (617, 372), "孙膑": (433, 569)}
rules._entry_fodder = {"裴擒虎": 4}
rules._carry_piece = "裴擒虎"
st86e = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=5,
                  gold=5, population=3, population_cap=5, hp=60, level=2)
a86e = rules.decide_sell_unit(st86e)
print(f"[主C 不按杂牌卖] {a86e}（期望 不卖 裴擒虎）")
if a86e and a86e.get("piece") == "裴擒虎":
    failures.append("主C 被当成吃完效果的杂牌卖掉了")

# 86g. 商店关着看不到货、场面没铺满、没有弈星：升级要给买牌留 3 能量（第十三局 R3：6 能量升级剩 1，人口停在 3/5）
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {"曜": (617, 372), "苏烈": (300, 569), "牛魔": (529, 372)}
st86g = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=3, gold=6,
                  level_cost=5, population=3, population_cap=5, hp=60, level=1, level_btn=(99, 545))
e86g = rules.decide_economy(st86g)
print(f"[场面没满先铺场] {'升了' if e86g else '不升'}（期望：不升，留钱买人）")
if e86g:
    failures.append("场面没铺满、商店看不到货时把钱全花在升级上")

# 86h. 场面铺满、等级落后进度表：拍卖回合只留举牌钱，其余预留让位给升级（第十四局 R4：1 级、升级费 2、8 能量没升）
rules.reset_match()
rules._active_comp = {"carry": "曜", "core": []}
rules._field = {n: (0, i) for i, n in enumerate(["曜", "苏烈", "牛魔", "亚连", "孙膑"])}
st86h = GameState(shop_open=True, population_valid=True, phase="prepare", round_no=4, gold=8,
                  level_cost=2, population=5, population_cap=5, hp=60, level=1, level_btn=(99, 545))
st86h.shop = [ShopSlot(index=0, piece="苏烈", cost=3)]
e86h = rules.decide_economy(st86h)
print(f"[落后进度先升级] {'升了' if e86h else '不升'}（期望：升了）")
if not e86h:
    failures.append("场面铺满、等级落后进度表，升级费 2 却没升")

# 86i. 觉醒过的 1 阶不按"过时过渡件"估值、也不为换绑枫叶舞卖掉（第十四局 觉醒曜被卖）
rules.reset_match()
rules._round_seen = 7
rules._carry_piece = "虞姬"
rules._carry_switch_round = 5
rules._field = {"曜": (499, 500), "虞姬": (617, 372), "苏烈": (300, 569)}
rules._maple_bound = "曜"
rules._ever_awakened = {"曜"}
st86i = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=7,
                  gold=5, population=3, population_cap=6, hp=60, level=3)
a86i = rules.decide_sell_unit(st86i)
print(f"[觉醒不卖] 曜估值={rules.unit_value('曜')}，决策 {a86i}（期望 估值>3、不卖曜）")
if rules.unit_value("曜") <= 3 or (a86i and a86i.get("piece") == "曜"):
    failures.append(f"觉醒过的曜被当成过渡件/被换绑卖掉: {a86i}")

# 87. 等弈星上场再升级的那几帧里，买牌不能把升级费花掉（第九局 2 级卡到第 8 回合）
rules.reset_match()
rules._round_seen = 6
rules._field = {n: (0, i) for i, n in enumerate(["曜", "明世隐", "孙膑", "西施", "鬼谷子"])}
st87 = GameState(shop_open=True, population_valid=False, phase="prepare", round_no=6,
                 gold=4, population=5, population_cap=5, hp=40, level=2, level_cost=2)
st87.level_btn = (61, 545)
st87.hand_slots = [{"name": "弈星", "x": 700, "y": 663}]
st87.hand = ["弈星"]
st87.shop = [ShopSlot(index=0, piece="孙膑", cost=3)]
rules._last_pop_seen = ()
e87 = rules.decide_economy(st87)          # 人口读不到、没有旧读数 -> 会等弈星
b87 = rules.decide_buy(st87)
print(f"[等弈星时不花升级费] economy={e87} buy={b87 and b87.get('piece')}（期望不买：4-3<2）")
if e87 is None and b87:
    failures.append(f"等弈星上场期间把升级费花在买牌上: {b87}")

# ---- 场景 88（2026-09-24 第一轮第 3 局）：转瞬牌立刻打、碰碰运气给主C、人口变红不再上场 ----
rules.reset_match()
for nm in ("援护战术", "招募战术"):
    ok88 = rules._is_self_cast(nm, rules.pieces.get(nm, {}))
    print(f"[转瞬不点名=自用] {nm}: {ok88}（期望 True）")
    if not ok88:
        failures.append(f"{nm}（转瞬、不需要指定目标）没被当成拿到就打的牌")
gu = rules.pieces.get("古币", {})
if gu and rules._is_self_cast("古币", gu):
    failures.append("古币（指定【河洛】英雄）被当成自用牌 —— 它要指定目标")
cat88 = rules._use_category("碰碰运气", rules.pieces.get("碰碰运气", {}))
print(f"[碰碰运气按效果牌用] {cat88}（期望 效果牌）")
if cat88 != "效果牌":
    failures.append(f"碰碰运气（指定英雄提升5-20级）没被当成加成牌：{cat88}")
rules.note_pop_over(9)
s88 = GameState(phase="prepare", round_no=9, gold=10, population=5, population_cap=6, population_valid=False,
                shop_open=False)
s88.hand_slots = [{"name": "扁鹊", "x": 700, "y": 663}]
rules._round_seen = 9
d88 = rules.decide_deploy(s88)
print(f"[人口变红后不再上场] {d88}（期望 None）")
if d88 is not None:
    failures.append(f"本回合拖新英雄时人口已变红，仍然决定上场：{d88}")

# ---- 场景 89（第一轮第 4 局）：拍卖预览期"只选中"不能当成已出价 ----
from agent.perception.state import AuctionLot, AuctionState
rules.reset_match()
lots89 = [AuctionLot(index=i, piece=p, x=200 + 180 * i, y=320) for i, p in enumerate(["鲁班大师", "东皇太一", "雅典娜"])]
s89 = GameState(phase="auction", round_no=8, gold=13)
s89.auction = AuctionState(lots=lots89, preview=True, my_gold=13)
a89 = rules.decide_auction(s89)
s89b = GameState(phase="auction", round_no=8, gold=13)
s89b.auction = AuctionState(lots=[AuctionLot(index=i, piece=l.piece, x=l.x, y=l.y) for i, l in enumerate(lots89)],
                            preview=False, my_gold=13)
b89 = rules.decide_auction(s89b)
print(f"[拍卖预览选中后正式出价] 预览={a89 and a89.get('select_only')} 正式={b89 and (b89.get('piece'), b89.get('select_only'))}（期望 正式能出价）")
if not (b89 and not b89.get("select_only")):
    failures.append(f"拍卖预览期选中后，正式开始没有出价：{b89}")

# ---- 场景 90：同一回合同一张牌不给同一个英雄用第二次 ----
rules.reset_match()
rules._round_seen = 5
rules.note_used_on(5, "特训战术", "曜")
print(f"[同回合同牌同人] 已记录={('5','特训战术','曜')}  集合={sorted(rules._used_on)}")
if (5, "特训战术", "曜") not in rules._used_on:
    failures.append("note_used_on 没记下")

# ---- 场景 91（第一轮第 4 局：武装分解要给有装备的角色用）：拆装备的牌给有装备的非主C ----
from agent.perception.extractors import cell_key as _ck91
rules.reset_match()
rules._round_seen = 9
rules._carry_piece = "蒙恬"
rules._field = {"蒙恬": (672, 430), "弈星": (433, 569), "苏烈": (529, 372)}
s91 = GameState(phase="prepare", round_no=9, gold=10, population=3, population_cap=6, population_valid=True,
                shop_open=False)
s91.hand_slots = [{"name": "武装分解", "x": 700, "y": 663}]
s91.equip_counts = {_ck91((672, 430)): 2, _ck91((433, 569)): 1}
rules._worn = {"弈星": ["力量腰带"]}               # 2026-10-15：摧毁装备只认穿戴账本（第 637 局件数模型把廉颇读成 1 件）
s91.unit_names = {}
rules._equip_seen = {_ck91((529, 372)): 2}       # 2026-10-11：格子历史件数不算（苏烈从没穿过装备）
d91x = rules.decide_use_cards(s91)               # 认人模型没认出弈星 → 不拆
print(f"[武装分解·没认出人] {d91x and (d91x.get('type'), d91x.get('target'))}（期望 None：先留着，不拆也不卖）")
if d91x and d91x.get("piece") == "武装分解":
    failures.append(f"武装分解没认人就拆：{d91x}")
rules._equip_seen = {}
s91.unit_names = {_ck91((433, 569)): ["弈星", 0.99]}
d91 = rules.decide_use_cards(s91)
print(f"[武装分解给有装备的非主C] {d91 and (d91.get('piece'), d91.get('target'))}（期望 弈星）")
if not (d91 and d91.get("target") == "弈星"):
    failures.append(f"武装分解没给身上有装备的非主C：{d91}")
s91.equip_counts = {_ck91((672, 430)): 2}
rules._worn = {"蒙恬": ["力量腰带", "风暴巨剑"]}
rules._equip_seen.clear()
d91b = rules.decide_use_cards(s91)
print(f"[只有主C 有装备时不拆主C] {d91b}（期望 卖掉或留着，绝不拖给主C；不适合用就卖掉）")
if d91b and d91b.get("piece") == "武装分解" and d91b.get("type") != "sell_card":
    failures.append(f"只有主C 身上有装备，武装分解却打出去了：{d91b}")
# 2026-09-25 第四轮第 3 局：未来主C（主C 表 ≥7 分）不能被当成"可拆"，即使它不是当前主C
rules._equip_seen.clear()
rules._carry_piece = "苏烈"
rules._field = {"苏烈": (529, 372), "武则天": (672, 430)}
s91.equip_counts = {_ck91((672, 430)): 2}
d91c = rules.decide_use_cards(s91)
print(f"[未来主C 武则天不拆] {d91c}（期望 不拖给武则天）")
if d91c and d91c.get("target") == "武则天":
    failures.append(f"武装分解拆了未来主C 武则天的装备：{d91c}")

# ---- 场景 92（第一轮第 5 局：坦克在前、其他尽量后排缩角落）：历史失败率不能盖过前后排 ----
from agent.perception.extractors import OWN_BOARD_ROWS as _ROWS92
_back92 = {tuple(c) for c in _ROWS92[2] + _ROWS92[3]}
_front92 = {tuple(c) for c in _ROWS92[0] + _ROWS92[1]}
for hero92, want92 in (("干将莫邪", _back92), ("牛魔", _front92)):
    rules.reset_match()
    rules._round_seen = 2
    rules._field = {"苏烈": (681, 571)}
    rules._cell_fails[(300, 569)] = 9      # 实战：后排角落那格刚失败过（被拉黑），要改投另一个后排格，不能跑到前排
    s92 = GameState(phase="prepare", round_no=2, gold=0, population=1, population_cap=5, population_valid=True,
                    shop_open=False)
    s92.hand_slots = [{"name": hero92, "x": 700, "y": 663}]
    d92 = rules.decide_deploy(s92)
    dst92 = tuple(d92["dst"]) if d92 else None
    print(f"[站位 {hero92}] {dst92}（期望 {'后排' if want92 is _back92 else '前排'}）")
    if dst92 not in want92:
        failures.append(f"{hero92} 被放到 {dst92}，不在{'后排' if want92 is _back92 else '前排'}")

# ---- 场景 93（第一轮第 5 局：装备没给主C）：主C 那格的"看到过满 3 件"是旧记录，装备仍要给主C ----
rules.reset_match()
rules._round_seen = 7
rules._carry_piece = "米莱狄"
rules._field = {"狂铁": (556, 430), "牛魔": (499, 500)}
rules._equip_seen = {"300,569": 3}          # 之前抓错牌、往这个空格拖了两次装备没弹四选一 -> 记成"满 3 件"
rules.confirm_drag({"type": "drag_card", "piece": "米莱狄", "src": (700, 663), "dst": (300, 569)}, True)
s93 = GameState(phase="prepare", round_no=7, gold=9, population=3, population_cap=6, population_valid=True,
                shop_open=False)
s93.hand_slots = [{"name": "影牙", "x": 650, "y": 663}]
s93.equip_counts = {}
d93 = rules.decide_use_cards(s93)
print(f"[装备给主C（旧的满员记录不算）] {d93 and d93.get('target')}（期望 米莱狄）")
if not (d93 and d93.get("target") == "米莱狄"):
    failures.append(f"主C 那格只是记过满员、实际 0 件，装备却没给主C：{d93}")

# ---- 场景 95（第二轮第 1 局：推演战术没有使用）：同回合同牌同人**默认允许**，被拒一次才记账 ----
import agent.decision.rules as _R95
_R95.RuleEngine.LEARNED_PATH = Path(__file__).resolve().parent / "_tmp_learned_rules.json"
if _R95.RuleEngine.LEARNED_PATH.exists():
    _R95.RuleEngine.LEARNED_PATH.unlink()
rules.reset_match()
rules._no_repeat = set()
rules._round_seen = 6
rules._carry_piece = "曜"
rules._field = {"曜": (529, 372)}
a95 = {"type": "drag_card", "piece": "推演战术", "src": (721, 663), "dst": (529, 372), "kind": "效果牌", "target": "曜"}
rules.confirm_drag(dict(a95), False)          # 第一次就失败：不能学成"不能重复用"
if "推演战术" in rules._no_repeat:
    failures.append("第一次使用就失败，被误学成了'同回合不能对同一人再用'")
rules._drag_fails.clear(); rules._cell_fails.clear()
rules.confirm_drag(dict(a95), True)           # 成功一次
rules.confirm_drag(dict(a95), False)          # 同回合再对曜用被拒 -> 只记一次观察，还不学（2026-09-30 古币误学）
if "推演战术" in rules._no_repeat:
    failures.append("只在一个回合被拒一次就学成了'不能重复用'（要两个不同回合）")
rules._round_seen = 7
rules._drag_fails.clear(); rules._cell_fails.clear()
rules.confirm_drag(dict(a95), True)           # 下一回合：成功一次
rules.confirm_drag(dict(a95), False)          # 同回合再被拒 -> 两个回合都观察到，也不学（10-18：绝大多数牌可重复用，名单只人工维护）
print(f"[同回合重复用：被拒不自动学] 学到={sorted(rules._no_repeat)}（期望 []）")
if rules._no_repeat:
    failures.append("被拒后自动学成了'同回合不能对同一人再用'（设计约定：不要自动学）")
if _R95.RuleEngine.LEARNED_PATH.exists():
    _R95.RuleEngine.LEARNED_PATH.unlink()
rules._no_repeat = set()

# ---- 场景 96（2026-09-24：换主C 要有快速提高等级的方法）：手里有夺取之术时可以换一个等级更低的新主C，
#      打夺取之术前先把老主C 挪到新主C 旁边 ----
rules.reset_match()
rules._round_seen = 8
rules._field = {"蒙恬": (529, 372), "亚连": (1027, 572), "牛魔": (955, 502)}
rules._prev_carry = "蒙恬"
rules._carry_piece = "亚连"
s96 = GameState(phase="prepare", round_no=8, gold=10, population=3, population_cap=6, population_valid=True,
                shop_open=False)
s96.hand_slots = [{"name": "夺取之术", "x": 650, "y": 663}]
print(f"[搬等级手段] ready={rules._level_move_ready(s96, '亚连', '蒙恬')}（期望 True）")
if not rules._level_move_ready(s96, "亚连", "蒙恬"):
    failures.append("手里有夺取之术，却不算'能搬等级'")
d96 = rules.decide_use_cards(s96)
steps96 = [st["type"] for st in (d96 or {}).get("steps", [])]
print(f"[夺取之术先挪老主C] {d96 and d96.get('type')} {steps96}（期望 sequence [move_unit, drag_card]）")
if not (d96 and d96.get("type") == "sequence" and steps96 == ["move_unit", "drag_card"]
        and d96["steps"][1]["dst"] == (1027, 572) or d96 and d96.get("type") == "sequence"
        and steps96 == ["move_unit", "drag_card"]):
    failures.append(f"夺取之术没有先把老主C 挪到新主C 旁边：{d96}")

# ---- 场景 97（2026-09-24：确认点击要记账，防止无限点击确认）：没铸成的装备每件只补选一次 ----
rules.reset_match()
rules.note_unforged((300, 569), "速击之枪")
j1 = rules.next_reforge(); j2 = rules.next_reforge()
rules.note_unforged((300, 569), "速击之枪")        # 同一件再记一次也不会再点
j3 = rules.next_reforge()
print(f"[补选装备只点一次] 第一次={j1} 第二次={j2} 再记一次后={j3}（期望 ('300,569','速击之枪') / None / None）")
if j1 != ("300,569", "速击之枪") or j2 is not None or j3 is not None:
    failures.append(f"没铸成装备的补选没有做到'每件只点一次'：{j1} {j2} {j3}")

# ---- 场景 98（2026-09-24：只有卖掉老主C 枫叶舞才能重新使用）：红叶染锋（离场后 50% 等级给下一个枫叶舞英雄）
#      是换主C 搬等级的手段，不能被"离场"否决；以阵亡为前提的仍否决 ----
from agent.perception.state import TalentOption as _TO98
rules.reset_match()
_saved_pick = rules._laya_pick
rules._laya_pick = None
s98 = GameState(phase="lord_boost", round_no=6)
s98.talents = [_TO98(index=0, name="红叶染锋", desc="枫叶舞英雄离场后，其真实等级的50%会保留给下一个枫叶舞英雄，限一次。"),
               _TO98(index=1, name="某阵亡特性", desc="枫叶舞英雄阵亡时，其等级给随机英雄。"),
               _TO98(index=2, name="枫影无拘", desc="转移后5秒内免疫受到的控制。")]
d98 = rules.decide_lord_boost(s98)
rules._laya_pick = _saved_pick
print(f"[红叶染锋不被否决] 选 {d98 and d98.get('name')}（期望 红叶染锋 或 枫影无拘，不能是阵亡那张）")
if not d98 or d98.get("name") == "某阵亡特性":
    failures.append(f"棋手特性选了以阵亡为前提的那张：{d98}")
_viable98 = [o.name for o in s98.talents if not any(k in o.name + o.desc for k in rules.LORD_BOOST_VETO)]
if "红叶染锋" not in _viable98:
    failures.append("红叶染锋 仍被棋手特性否决（它是换主C 搬等级的手段）")

# ---- 场景 99（第二轮第 3 局：枫叶舞没用上）：账本记着"绑在牛魔"、卡面却是亮的 = 没绑上，要重新绑 ----
rules.reset_match()
rules._round_seen = 3
rules._carry_piece = "孙膑"
rules._field = {"孙膑": (529, 372), "曜": (433, 569)}
rules._maple_bound = "牛魔"
rules._maple_try_ts = 0.0
s99 = GameState(phase="prepare", round_no=3, gold=5, population=2, population_cap=5, population_valid=True,
                shop_open=False)
s99.hand_slots = [{"name": "枫叶舞", "x": 407, "y": 663, "ready": True}]
d99a = rules.decide_use_cards(s99)          # 10-21：第一帧亮只开始计时（第 851 局一帧误读就清了记录）
if d99a and d99a.get("piece") == "枫叶舞":
    failures.append(f"枫叶舞只亮了一帧就重绑了：{d99a}")
import time as _t99
rules.__dict__["_maple_bright_since"] = _t99.time() - 3   # 连续亮了 3 秒
d99 = rules.decide_use_cards(s99)
print(f"[枫叶舞亮着=没绑上] {d99 and (d99.get('piece'), d99.get('target'))}（期望 重新绑，目标不是牛魔）")
if not (d99 and d99.get("piece") == "枫叶舞" and d99.get("target") != "牛魔"):
    failures.append(f"枫叶舞卡面亮着（没绑上），却没有重新绑：{d99}")

# ---- 场景 100（第二轮第 3 局：坦克牛魔站到了后排）：确认过位置的坦克在后排、前排有空格 → 挪到前排 ----
rules.reset_match()
rules._round_seen = 6
rules._carry_piece = "孙膑"
rules._field = {"牛魔": (300, 569), "孙膑": (433, 569)}
rules._confirmed_at = {"牛魔": (6, (300, 569)), "孙膑": (6, (433, 569))}
s100 = GameState(phase="prepare", round_no=6, gold=5, population=2, population_cap=6, population_valid=True,
                 shop_open=False)
d100x = rules.decide_reposition(s100)
print(f"[画面没看到徽章不挪] {d100x}（期望 None：以画面为准）")
if d100x is not None:
    failures.append(f"画面上没看到那一格有人，却按账本去挪：{d100x}")
s100.unit_anchors = {"300,569": 1, "433,569": 1}
d100 = rules.decide_reposition(s100)
from agent.perception.extractors import OWN_BOARD_ROWS as _R100
_front100 = {tuple(c) for c in _R100[0] + _R100[1]}
print(f"[坦克挪前排] {d100}（期望 move_unit 牛魔 到前排）")
if not (d100 and d100.get("type") == "move_unit" and d100.get("piece") == "牛魔" and tuple(d100["dst"]) in _front100):
    failures.append(f"坦克站在后排没有挪到前排：{d100}")
if rules.decide_reposition(s100) is not None:
    failures.append("站位纠正同一回合做了两次")

# ---- 场景 101（2026-09-24 实战截图：大明星、人才搜寻×3 攒在手里不用）：用牌表标"能打出、不需要目标"的天赋牌也要打 ----
rules.reset_match()
rules._round_seen = 9
rules._carry_piece = "孙膑"
rules._field = {"孙膑": (529, 372), "曜": (433, 569)}
s101 = GameState(phase="prepare", round_no=9, gold=3, population=2, population_cap=6, population_valid=True,
                 shop_open=False)
s101.hand_slots = [{"name": "大明星", "x": 500, "y": 663}, {"name": "人才搜寻", "x": 640, "y": 663}]
d101 = rules.decide_use_cards(s101)
print(f"[大明星/人才搜寻要打] {d101 and (d101.get('piece'), d101.get('self_cast'))}（期望 其中一张，自用）")
if not (d101 and d101.get("piece") in ("大明星", "人才搜寻")):
    failures.append(f"大明星/人才搜寻（用牌表：能打出、不需要目标）没打出去：{d101}")
s101b = GameState(phase="prepare", round_no=9, gold=3, population=6, population_cap=6, population_valid=True,
                  shop_open=False)
s101b.hand_slots = [{"name": "大明星", "x": 500, "y": 663}]
d101b = rules.decide_use_cards(s101b)
print(f"[大明星人口满先留着] {d101b}（期望 None：它要空人口）")
if d101b and d101b.get("piece") == "大明星":
    failures.append("人口已满还打大明星（它会直接上阵一名英雄，需要空人口）")

# ---- 场景 102（2026-09-24：逐张读卡牌、自己写使用逻辑）：审定用牌表 ----
def _st102(gold=10, pop=3, cap=6, hand=()):
    st = GameState(phase="prepare", round_no=8, gold=gold, population=pop, population_cap=cap, population_valid=True,
                   shop_open=False, hp=40)
    st.hand_slots = [{"name": n, "x": 500 + 70 * i, "y": 663} for i, n in enumerate(hand)]
    st.hand = list(hand)
    return st
rules.reset_match(); rules._round_seen = 8
rules._carry_piece = "孙膑"; rules._field = {"孙膑": (529, 372), "牛魔": (617, 372), "弈星": (300, 569)}
d = rules.decide_use_cards(_st102(hand=["遗志"]))
print(f"[遗志没换主C 时不用] {d}（期望 None —— 原来会拖给当前主C、把主C 移除）")
if d and d.get("piece") == "遗志":
    failures.append(f"没换主C 时打了遗志（会移除英雄）：{d}")
# 遗志在明确更换主C 时可以拖给主C —— 打给**老主C**
rules._prev_carry = "牛魔"; rules._carry_switch_round = 6
d = rules.decide_use_cards(_st102(hand=["遗志"]))
print(f"[遗志场上 3 人不用] {d}（期望 None：等级会随机给到弈星）")
if d and d.get("piece") == "遗志":
    failures.append(f"场上 3 人时打了遗志（很难确定等级会分配给谁）：{d}")
rules._field = {"孙膑": (529, 372), "牛魔": (617, 372)}
d = rules.decide_use_cards(_st102(hand=["遗志"]))
print(f"[遗志只剩新老主C 给老主C] {d and d.get('target')}（期望 牛魔）")
if not (d and d.get("target") == "牛魔"):
    failures.append(f"换主C 之后遗志没打给老主C：{d}")
rules.confirm_drag(dict(d or {}, type="drag_card"), True)
if "牛魔" in rules._field:
    failures.append("遗志移除老主C 成功后，账本里还留着他")
rules._prev_carry = ""; rules._carry_switch_round = -1
rules._field = {"孙膑": (529, 372), "牛魔": (617, 372), "弈星": (300, 569)}
d = rules.decide_use_cards(_st102(hand=["听天由命"]))
if d and d.get("piece") == "听天由命":
    failures.append(f"听天由命被自动打出（消除 4 名最低等级英雄）：{d}")
d = rules.decide_use_cards(_st102(gold=6, hand=["等价交换"]))
print(f"[等价交换能量多时不用] {d}（期望 None）")
if d and d.get("piece") == "等价交换":
    failures.append(f"能量 6 时打了等价交换（会扣光能量）：{d}")
d = rules.decide_use_cards(_st102(gold=3, hand=["等价交换"]))
print(f"[等价交换能量 2~4 用] {d and d.get('piece')}（期望 等价交换；2~4 换基础装备）")
if not (d and d.get("piece") == "等价交换"):
    failures.append(f"能量 3 时没打等价交换：{d}")
d = rules.decide_use_cards(_st102(gold=1, hand=["等价交换"]))
if d and d.get("piece") == "等价交换":
    failures.append(f"能量 1 时打了等价交换（只换 2 张随机英雄，时机改成 2~4）：{d}")
# 神鹰锻匠：选了它，云中君按派系贴大河主C
_tt = set(rules._talents_taken); _cp = rules._carry_piece
rules._carry_piece = "虞姬"; rules._talents_taken = set()
if rules._carry_buddy("云中君"):
    failures.append("没选神鹰锻匠时云中君也要贴主C")
rules._talents_taken = {"神鹰锻匠"}
if not rules._carry_buddy("云中君"):
    failures.append(f"选了神鹰锻匠、主C 虞姬，云中君没按贴主C 处理（派系={rules._buddy_faction('云中君')}）")
rules._talents_taken = _tt; rules._carry_piece = _cp
d = rules.decide_use_cards(_st102(hand=["化铁为金"]))
print(f"[化铁为金不点主C] {d and d.get('target')}（期望 不是孙膑：变身牌绝不点主C，第 42 局）")
if d and d.get("target") == "孙膑":
    failures.append(f"化铁为金点了主C（会把主C 变成随机 5 阶）：{d}")
ok_t, why_t = rules.card_target_ok("化铁为金", "孙膑")
if ok_t:
    failures.append("card_target_ok 放行了对主C 用变身牌（Laya 用牌题也会被放行）")
d = rules.decide_use_cards(_st102(hand=["霸者重铠"]))
print(f"[防御装给坦克] {d and d.get('target')}（期望 牛魔）")
if not (d and d.get("target") == "牛魔"):
    failures.append(f"防御装（霸者重铠）没给坦克：{d}")

# ---- 场景 103（遗志代替卖人 + 移除主C 总闸） ----
rules.reset_match(); rules._round_seen = 8
rules._carry_piece = "孙膑"
rules._field = {"孙膑": (529, 372), "苏烈": (617, 372), "曜": (433, 569), "西施": (300, 569)}
rules._unit_lv = {"苏烈": 18, "孙膑": 30, "曜": 20, "西施": 15}
s103 = _st102(hand=["遗志"])
y103 = rules._yizhi_instead_of_sell(s103, "苏烈", (617, 372))
print(f"[遗志代替卖人] {y103 and (y103.get('piece'), y103.get('target'))}（期望 遗志 -> 苏烈）")
if not (y103 and y103.get("target") == "苏烈"):
    failures.append(f"18 级的过渡件要卖、其余都是要留的人，却没用遗志代替：{y103}")
if rules._yizhi_instead_of_sell(s103, "孙膑", (529, 372)) is not None:
    failures.append("遗志代替卖人 打给了当前主C")
_orig103 = rules.decide_upgrade
rules.decide_upgrade = lambda st: {"type": "drag_card", "piece": "遗志", "src": (500, 663), "dst": (529, 372),
                                   "kind": "效果牌", "target": "孙膑"}
acts103 = rules.decide_all(_st102(hand=["遗志"]))
rules.decide_upgrade = _orig103
print(f"[移除主C 总闸] {[(a.get('piece'), a.get('target')) for a in acts103]}（期望 没有 遗志->孙膑）")
if any(a.get("piece") == "遗志" and a.get("target") == "孙膑" for a in acts103):
    failures.append("总闸没拦住：遗志打给了当前主C")

# ---- 场景 104（2026-09-24：有群体技能的射手很适合当主C，李信这类技能描述的战士也是）：读技能的评分进选主C ----
print(f"[主C 资格] 蒙犽={rules._carry_eligible('蒙犽')} 李信={rules._carry_eligible('李信')} 牛魔={rules._carry_eligible('牛魔')} 明世隐={rules._carry_eligible('明世隐')}（期望 True True False False）")
if not (rules._carry_eligible("蒙犽") and rules._carry_eligible("李信")):
    failures.append("群体射手蒙犽 / 李信 不能当主C")
if rules._carry_eligible("牛魔") or rules._carry_eligible("明世隐"):
    failures.append("坦克/辅助 被允许当主C")
rules.reset_match(); rules._round_seen = 8
rules._field = {"蒙犽": (300, 569), "曹操": (529, 372)}
_s104 = GameState(phase="prepare", round_no=8, gold=5, population=2, population_cap=6, population_valid=True, shop_open=False)
_s104.unit_levels = {"300,569": 10, "529,372": 10}
sc_a, sc_b = rules._carry_score("蒙犽", _s104), rules._carry_score("曹操", _s104)
print(f"[技能评分进主C 打分] 蒙犽 {sc_a:.0f} vs 曹操 {sc_b:.0f}（期望 蒙犽 更高）")
if not (sc_a is not None and sc_b is not None and sc_a > sc_b):
    failures.append(f"同等级下群体射手蒙犽的主C 分不高于曹操：{sc_a} vs {sc_b}")

# ---- 场景 105（2026-09-24：能做单核的射手角色还有好几个，如安琪拉）：伞单核候选优先买 ----
rules.reset_match(); rules._round_seen = 5; rules._carry_piece = ""
rules._field = {"牛魔": (617, 372), "苏烈": (529, 372)}
_s105 = GameState(phase="prepare", round_no=5, gold=10, level=3, population=2, population_cap=6, population_valid=True,
                  shop_open=True, hp=50)
_s105.shop = [ShopSlot(index=0, piece="狂铁", cost=3), ShopSlot(index=1, piece="安琪拉", cost=3),
              ShopSlot(index=2, piece="白起", cost=3), ShopSlot(index=3, piece="梦奇", cost=3)]
d105 = rules.decide_buy(_s105)
print(f"[伞单核候选优先] 买 {d105 and d105.get('piece')}（期望 安琪拉）")
if not (d105 and d105.get("piece") == "安琪拉"):
    failures.append(f"没有主C 时商店里有安琪拉（群体法师 8 分），却买了别的：{d105}")

# ---- 场景 106（第二轮第 5 局第 11 回合，实战截图）：主C 账本上已穿 3 件，装备不再往他身上拖，改给前排 ----
rules.reset_match(); rules._round_seen = 11; rules._carry_piece = "嬴政"
rules._field = {"嬴政": (276, 501), "蒙恬": (617, 372), "甄姬": (529, 569)}
rules._worn = {"嬴政": ["聚灵法杖", "生命魔典", "纯净苍穹"]}
_s106 = GameState(phase="prepare", round_no=11, gold=20, level=5, population=3, population_cap=7, population_valid=True,
                  shop_open=False)
n106 = rules._equips_at(_s106, (276, 501))
t106 = rules._equip_target(_s106, "反伤重甲")
print(f"[主C 已满 3 件] 件数={n106} 反伤重甲 -> {t106}（期望 3 件、不给嬴政）")
if n106 < 3:
    failures.append(f"穿戴账本里嬴政 3 件，判满时只算出 {n106} 件")
if t106 and t106[0] == "嬴政":
    failures.append(f"主C 已穿 3 件还把反伤重甲给主C：{t106}")
r106 = rules._reviewed_target(_s106, "反伤重甲", rules.usage_reviewed.get("反伤重甲", {}))
print(f"[防御装给前排] 审定表目标 -> {r106}（期望 蒙恬）")
if not (isinstance(r106, tuple) and r106[0] == "蒙恬"):
    failures.append(f"反伤重甲 没给前排蒙恬：{r106}")

# ---- 场景 107（第二轮第 5 局第 13 回合）：手里刚买的同名嬴政是合成材料，对账不能把场上的主C 嬴政删掉 ----
rules.reset_match(); rules._round_seen = 13; rules._carry_piece = "嬴政"
rules._field = {"嬴政": (276, 501), "少司缘": (390, 502), "甄姬": (839, 502), "明世隐": (319, 375),
                "百里玄策": (617, 496), "敖隐": (617, 372), "蒙恬": (672, 430), "庄周": (529, 372)}
_s107 = GameState(phase="prepare", round_no=13, gold=20, level=6, population=7, population_cap=7, population_valid=True,
                  shop_open=False)
_s107.hand_slots = [{"name": "嬴政", "x": 669, "y": 670}, {"name": "东皇太一", "x": 1045, "y": 670}]
rules.reconcile(_s107); rules.reconcile(_s107)
print(f"[对账不删主C] 场上 {sorted(rules._field)}（期望 仍有嬴政、共 7 个）")
if "嬴政" not in rules._field or len(rules._field) != 7:
    failures.append(f"对账把手里有同名材料的主C 嬴政删了 / 人数不对：{sorted(rules._field)}")

# ---- 场景 108（同一局第 12 回合）：化铁为金打在蒙恬身上 -> 蒙恬离场；变出来的东皇太一按觉醒卡优先上场 ----
rules.reset_match(); rules._round_seen = 12
rules._field = {"嬴政": (276, 501), "蒙恬": (672, 430)}
rules.confirm_drag({"type": "drag_card", "piece": "化铁为金", "src": (650, 663), "dst": (672, 430),
                    "kind": "效果牌", "target": "蒙恬"}, True)
_s108a = GameState(phase="prepare", round_no=12); _s108a.hand_slots = [{"name": "枫叶舞", "x": 300, "y": 670}]
_s108b = GameState(phase="prepare", round_no=12)
_s108b.hand_slots = [{"name": "枫叶舞", "x": 300, "y": 670}, {"name": "东皇太一", "x": 934, "y": 670}]
rules.track_hand_gains(_s108a); rules.track_hand_gains(_s108b)
print(f"[化铁为金] 场上 {sorted(rules._field)} 觉醒待上场 {sorted(rules._awakened)}（期望 无蒙恬、东皇太一 觉醒）")
if "蒙恬" in rules._field:
    failures.append("化铁为金打成功后蒙恬还记在场上")
if "东皇太一" not in rules._awakened:
    failures.append("化铁为金变出来的东皇太一没被当成觉醒卡")

# ---- 场景 109（2026-09-24）：A6/A7/B6/B7 不放英雄（徽章会挡住人口数字）----
from agent.perception.extractors import (OWN_BOARD_DROPS_BACK as _DB, OWN_BOARD_DROPS_FRONT as _DF,
                                         POP_OCCLUDING_CELLS as _POC)
_bad109 = [c for c in list(_DF) + list(_DB) if c in _POC]
rules.reset_match(); rules._round_seen = 5
_s109 = GameState(phase="prepare", round_no=5, gold=10, level=5, population=0, population_cap=7, population_valid=True,
                  shop_open=False)
_near109 = rules._nearest_free_cell(_s109, (900, 400))
print(f"[避开挡人口的格子] 落点列表里的禁区格 {_bad109}；离 (900,400) 最近的空格 {_near109}（期望不在 {sorted(_POC)}）")
if _bad109 or _near109 in _POC:
    failures.append(f"A6/A7/B6/B7 还会被选作落点：{_bad109} {_near109}")

# ---- 场景 94（第一轮第 5 局第 1 回合）：抓错牌 —— 想拖干将莫邪、上场的是牛魔：按实际记账，不拉黑格子、不算失败 ----
rules.reset_match()
rules._round_seen = 1
rules.note_wrong_grab({"type": "drag_card", "piece": "干将莫邪", "src": (863, 663), "dst": (499, 500)}, "牛魔", True)
print(f"[抓错牌按实际记账] 场上={rules._field} 格子失败={rules._cell_fails}")
if rules._field.get("牛魔") != (499, 500) or "干将莫邪" in rules._field:
    failures.append(f"抓错牌后账本不对：{rules._field}")
if rules._cell_fails or rules._drag_fails:
    failures.append(f"抓错牌被算成了格子/牌的失败：{rules._cell_fails} {dict(rules._drag_fails)}")

# ---- 场景 110（2026-09-24 数据局）：天赋「战术模组」可上阵人数+1 → 人口上限可以是 8，要记账、跨重启、换局清零 ----
rules.reset_match()
rules.note_executed({"type": "talent_pick", "index": 2, "name": "战术模组"})
rules.note_executed({"type": "talent_pick", "index": 0, "name": "扩编名额"})     # 花能量可买的，不是立即生效
_pb = rules.pop_bonus
_saved = rules.dump_state()
rules.reset_match()
_pb_reset = rules.pop_bonus
rules.load_state(_saved)
print(f"[天赋加人口] 选战术模组+扩编名额后 +{_pb}，换局 {_pb_reset}，读回存档 {rules.pop_bonus}（期望 1/0/1）")
if (_pb, _pb_reset, rules.pop_bonus) != (1, 0, 1):
    failures.append(f"人口加成记账不对：{_pb} {_pb_reset} {rules.pop_bonus}")
rules.reset_match()

# ---- 场景 111（2026-09-24 数据局）：40 级的 1 阶主C 孙膑，不能被等级没读到的 4 阶甄姬顶掉（手里没有搬等级手段）----
rules.reset_match()
rules._round_seen = 8
rules._field = {"孙膑": (300, 569), "甄姬": (1027, 572), "狂铁": (672, 430)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "孙膑", (300, 569)
_s111 = GameState(phase="prepare", round_no=8, gold=10, level=5, population=3, population_cap=7, population_valid=True,
                  unit_levels={"300,569": 40, "672,430": 6})
rules.choose_carry(_s111)
_c1 = rules._carry_piece
_s111.unit_levels["1027,572"] = 3
rules._carry_switch_round = -1
rules.choose_carry(_s111)
_c2 = rules._carry_piece
print(f"[换主C 看等级] 甄姬等级读不到时主C={_c1}，读到 3 级时主C={_c2}（期望都是 孙膑）")
if (_c1, _c2) != ("孙膑", "孙膑"):
    failures.append(f"40 级孙膑被低等级/未读到等级的甄姬顶掉：{_c1} {_c2}")
rules.reset_match()

# ---- 场景 112（2026-09-24）：满编后要给"留到最后"的场上英雄凑觉醒，别只买主C；付费刷新连刷 4 次停手 ----
from agent.perception.state import ShopSlot as _SS112
rules.reset_match()
rules._round_seen = 13
rules._field = {"孙膑": (300, 569), "狂铁": (672, 430), "猪八戒": (529, 372), "廉颇": (556, 430),
                "曜": (617, 372), "甄姬": (1027, 572), "阿轲": (718, 371)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "甄姬", (1027, 572)
_s112 = GameState(phase="prepare", round_no=13, gold=20, level=5, population=7, population_cap=7, population_valid=True,
                  shop_open=True, awaken_progress={"529,372": 1})
_s112.shop = [_SS112(index=0, piece="猪八戒", cost=3)]
_keep = rules._is_keeper("猪八戒")
_worth = rules._worth_buying(_s112.shop[0], _s112)
_left = rules.awaken_left("猪八戒", _s112)
rules._refresh_streak = (13, 4)
_s112.shop = [_SS112(index=0, piece="苏烈", cost=3)]
_ref = rules.decide_refresh(_s112)
print(f"[凑觉醒] 猪八戒 keeper={_keep} 值得买={_worth} 还差={_left}；连刷 4 次后 decide_refresh={_ref}（期望 True/True/2/None）")
if not (_keep and _worth and _left == 2) or _ref is not None:
    failures.append(f"满编凑觉醒/连刷上限不对：{_keep} {_worth} {_left} {_ref}")
rules.reset_match()

# ---- 场景 113（2026-09-24 数据局）：拍卖被超价要跟 —— 别家价格读成 0 也算被超；领先者是我就不动 ----
from agent.perception.state import AuctionState as _AS113, AuctionLot as _AL113
rules.reset_match()
rules._round_seen = 11
_s113 = GameState(phase="auction", round_no=11, gold=27)
_s113.auction = _AS113(lots=[_AL113(index=0, piece="众星拱月", top_bid=2, top_bidder="我", x=224, y=233),
                             _AL113(index=1, piece="沧海拾遗", top_bid=0, top_bidder="ontcy", x=635, y=233)],
                       my_gold=27)
rules._bids_round, rules._bids = 11, {"众星拱月": 2}
rules._bid_piece, rules._bid_price = "众星拱月", 2
_lead = rules.decide_auction(_s113)
_s113.auction.lots[0] = _AL113(index=0, piece="众星拱月", top_bid=0, top_bidder="99号技師", x=224, y=233)
_over = rules.decide_auction(_s113)
print(f"[拍卖跟价] 我领先时={_lead}；被 99号技師 超过（价格读成 0）时={_over and (_over['type'], _over['piece'])}（期望 None / 出价）")
if _lead is not None or not (_over and _over.get("type") == "auction_bid"):
    failures.append(f"拍卖跟价不对：领先时 {_lead}，被超时 {_over}")
rules.reset_match()

# ---- 场景 114（2026-09-24 match8）：觉醒过的主C 不让位给没觉醒的（重新上场那一帧等级还没读到）----
rules.reset_match()
rules._round_seen = 4
rules._field = {"曜": (617, 372), "西施": (300, 569), "牛魔": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "曜", (617, 372)
rules._ever_awakened.add("曜")
_s114 = GameState(phase="prepare", round_no=4, gold=10, level=2, population=3, population_cap=5,
                  population_valid=True, unit_levels={"300,569": 6, "529,372": 3})
rules.choose_carry(_s114)
print(f"[觉醒主C 不让位] 主C={rules._carry_piece}（期望 曜）")
if rules._carry_piece != "曜":
    failures.append(f"觉醒过的主C 被没觉醒的顶掉：{rules._carry_piece}")
rules.reset_match()

# ---- 场景 116（2026-09-24 match9）：过渡主C（曜，评分 4）过了前期就换，觉醒 + 40 级也一样 ----
rules.reset_match()
rules._round_seen = 9
rules._field = {"曜": (617, 372), "露娜": (300, 569), "牛魔": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "曜", (617, 372)
rules._ever_awakened.add("曜")
_s116 = GameState(phase="prepare", round_no=9, gold=10, level=5, population=3, population_cap=7,
                  population_valid=True, unit_levels={"617,372": 40, "300,569": 3, "529,372": 5})
rules.choose_carry(_s116)
_c116 = rules._carry_piece
rules._carry_switch_round = -1
rules._field.pop("露娜")
rules._field["牛魔2"] = (300, 569)
rules._carry_piece, rules._carry = "曜", (617, 372)
rules.choose_carry(_s116)
_c116b = rules._carry_piece
print(f"[过渡主C 换掉] 场上有露娜时主C={_c116}（期望 露娜）；没有像样输出位时主C={_c116b}（期望 还是曜，不清空）")
if _c116 != "露娜" or _c116b != "曜":
    failures.append(f"过渡主C 处理不对：{_c116} / {_c116b}")
rules.reset_match()

# ---- 场景 117（2026-09-24）：瑶 / 太乙真人 作用于"最近友军"→ 贴主C；太乙不给装备；瑶只给一件、首选破茧之衣 ----
rules.reset_match()
rules._round_seen = 8
_carry117 = (300, 569)
rules._field = {"百里守约": _carry117, "太乙真人": (672, 430), "牛魔": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "百里守约", _carry117
_s117 = GameState(phase="prepare", round_no=8, gold=10, level=5, population=3, population_cap=7, population_valid=True)
_buddy = (rules._carry_buddy("瑶"), rules._carry_buddy("太乙真人"), rules._carry_buddy("牛魔"))
_t1 = rules._equip_target(_s117, "破茧之衣")            # 瑶不在场：给主C
rules._field["瑶"] = (433, 569); rules._cell_owner[(433, 569)] = "瑶"
_t2 = rules._equip_target(_s117, "破茧之衣")            # 瑶在场、0 件：给瑶
# 2026-09-29：瑶要有一件普通装备 —— 瑶 0 件时任何普通装备都先给瑶（开团分享给 3 名友军）
_t3 = rules._equip_target(_s117, "近卫荣耀")            # 普通装备（成品）：给瑶
_t3b = rules._equip_target(_s117, "风暴巨剑")           # 普通装备（基础）：给瑶
_t3c = rules._equip_target(_s117, "末世")               # 特殊装备（3 阶）：不给瑶
_s117.hand_slots = [{"name": "破碎圣杯", "x": 500, "y": 680}]
_t3d = rules._equip_target(_s117, "风暴巨剑")           # 手里有破碎圣杯（能铸破茧之衣）：风暴巨剑不给瑶
_s117.hand_slots = []
_role = rules.equip_role_for("瑶")
_s117.equip_counts = {"300,569": 3, "433,569": 1}       # 主C 满了、瑶已有一件
_t4 = rules._equip_target(_s117, "近卫荣耀")            # 不能给太乙、不能再给瑶 -> 给牛魔或留着
print(f"[瑶/太乙] 识别={_buddy}（期望 T,T,F）；破茧之衣 瑶不在场→{_t1 and _t1[0]}，在场→{_t2 and _t2[0]}；"
      f"近卫荣耀→{_t3 and _t3[0]}，风暴巨剑→{_t3b and _t3b[0]}，末世→{_t3c and _t3c[0]}，有圣杯时风暴巨剑→{_t3d and _t3d[0]}；"
      f"瑶的四选一={_role}；主C满+瑶已一件→{_t4 and _t4[0]}")
if (_buddy != (True, True, False) or (_t1 and _t1[0]) != "百里守约" or (_t2 and _t2[0]) != "瑶"
        or (_t3 and _t3[0]) != "瑶" or (_t3b and _t3b[0]) != "瑶" or (_t3c and _t3c[0]) == "瑶"
        or (_t3d and _t3d[0]) == "瑶" or _role != "share:瑶" or (_t4 and _t4[0]) in ("太乙真人", "瑶")):
    failures.append(f"瑶/太乙 规则不对：{_buddy} {_t1} {_t2} {_t3} {_t3b} {_t3c} {_t3d} {_role} {_t4}")
from agent.perception.extractors import pick_equip_index as _pei
_pk = _pei(Path(ROOT) / "data" if "ROOT" in globals() else Path("data"), "破碎圣杯", "share:瑶")
print(f"[瑶的四选一] 破碎圣杯 → {_pk}（期望 破茧之衣）")
if not _pk or _pk[1] != "破茧之衣":
    failures.append(f"瑶的四选一没选破茧之衣：{_pk}")
rules.reset_match()

# ---- 场景 118（2026-09-24）：公孙离带夺取 —— 把不重要的 ≥10 级英雄挪到它旁边再合成；主C/输出位挨着时不合成 ----
rules.reset_match()
rules._round_seen = 10
rules._field = {"公孙离": (617, 372), "刘邦": (300, 569), "甄姬": (955, 502), "猪八戒": (1027, 572)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "公孙离", (617, 372)
_s118 = GameState(phase="prepare", round_no=10, gold=10, level=5, population=4, population_cap=7, population_valid=True,
                  shop_open=False, unit_levels={"617,372": 5, "300,569": 43, "955,502": 4, "1027,572": 23})
_s118.hand_slots = [{"name": "公孙离", "x": 600, "y": 663}]
_s118.hand = ["公孙离"]
_a1 = rules.decide_upgrade(_s118)
_ok1 = bool(_a1 and _a1.get("type") == "sequence" and _a1["steps"][0].get("piece") == "刘邦"
            and tuple(_a1["steps"][0]["dst"]) in rules._cell_neighbors((617, 372)))
rules.reset_match()
rules._round_seen = 10
rules._field = {"公孙离": (617, 372), "刘邦": (300, 569), "甄姬": (529, 372)}          # 甄姬（评分 7）挨着
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "公孙离", (617, 372)
_a2 = rules.decide_upgrade(_s118)
print(f"[夺取合成] 喂刘邦(43 级)={_ok1}（{_a1 and _a1.get('steps', [{}])[0]}）；甄姬挨着时={_a2}（期望 True / None）")
if not _ok1 or _a2 is not None:
    failures.append(f"夺取合成不对：{_a1} / {_a2}")
# 场景 118c（2026-09-26 第四轮第 4 局）：主C 孙悟空(评分 8) 挨着阿轲(5) —— 抢来的等级喂主C，照样合成
rules.reset_match()
rules._round_seen = 15
rules._field = {"孙悟空": (783, 438), "阿轲": (718, 371), "亚连": (319, 375)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "孙悟空", (783, 438)
_s118c = GameState(phase="prepare", round_no=15, gold=4, level=6, population=3, population_cap=8, population_valid=True,
                   shop_open=False)
_s118c.hand_slots = [{"name": "孙悟空", "x": 700, "y": 663}]
_s118c.hand = ["孙悟空"]
_a3 = rules.decide_upgrade(_s118c)
print(f"[夺取合成·主C 自己] 孙悟空挨着阿轲 -> {_a3 and (_a3.get('type'), _a3.get('piece'))}（期望 合成孙悟空）")
if not (_a3 and _a3.get("piece") == "孙悟空"):
    failures.append(f"主C 带夺取挨着弱输出位时不合成：{_a3}")
rules.reset_match()
rules.reset_match()

# ---- 场景 119（2026-09-24 match13）：主C 觉醒回手 → 人口少一个 + 手里有场上的同名 → 先确认，别让别人占位 ----
rules.reset_match()
rules._round_seen = 13
_f119 = {"西施": (276, 501), "公孙离": (617, 372), "猪八戒": (529, 372), "刘邦": (556, 430),
         "太乙真人": (672, 430), "小乔": (955, 502), "阿轲": (1027, 572)}
rules._field = dict(_f119)
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "西施", (276, 501)
_s119 = GameState(phase="prepare", round_no=13, gold=4, level=5, population=6, population_cap=7, population_valid=True,
                  shop_open=False)
_s119.hand_slots = [{"name": "西施", "x": 1045, "y": 663}, {"name": "明世隐", "x": 669, "y": 663}]
_s119.hand = ["西施", "明世隐"]
_sus = rules.awaken_suspects(_s119)
_d1 = rules.decide_deploy(_s119)
rules._awk_cleared.add((13, "西施"))                    # 点开看过：只是合成材料
_d2 = rules.decide_deploy(_s119)
print(f"[觉醒回手嫌疑] 嫌疑={_sus}；确认前上场={_d1}；确认不是觉醒后上场={_d2 and _d2.get('piece')}（期望 ['西施'] / None / 明世隐或西施）")
if _sus != ["西施"] or _d1 is not None or not _d2:
    failures.append(f"觉醒回手嫌疑处理不对：{_sus} {_d1} {_d2}")
rules.reset_match()

# ---- 场景 120（2026-09-24）：限制性进化 —— 1~2 阶英雄连同等级换进高 2 阶身体（+10 级）；核心可以换，真主C/觉醒的不换 ----
rules.reset_match()
_r120 = {"target": "low_tier_noncarry", "category": "天赋牌"}
rules._field = {"孙膑": (300, 569), "曜": (617, 372), "牛魔": (529, 372), "墨子": (556, 430)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "孙膑", (300, 569)
rules._unit_lv.update({"孙膑": 12, "曜": 8, "牛魔": 6, "墨子": 3})
rules._round_seen = 4
_e1 = rules._reviewed_target(GameState(phase="prepare", round_no=4), "限制性进化", _r120)
rules._carry_piece, rules._carry = "曜", (617, 372)
rules._unit_lv["曜"] = 40
rules._round_seen = 9
_e2 = rules._reviewed_target(GameState(phase="prepare", round_no=9), "限制性进化", _r120)
_n1 = _e1[0] if isinstance(_e1, tuple) else _e1
_n2 = _e2[0] if isinstance(_e2, tuple) else _e2
print(f"[限制性进化] 第 4 回合（主C 孙膑）→ {_n1}（期望 非孙膑的 1~2 阶）；第 9 回合（过渡主C 曜 40 级）→ {_n2}（期望 曜）")
if _n1 in (None, "skip", "孙膑") or _n2 != "曜":
    failures.append(f"限制性进化目标不对：{_e1} / {_e2}")
rules.reset_match()

# ---- 场景 121（2026-09-24 match14）：卖掉前排的人腾出格子，后排法师嬴政不该被塞进前排（有视觉结果时按定位放）----
rules.reset_match()
rules._round_seen = 10
rules._field = {"西施": (1027, 572), "瑶": (839, 502), "鲁班大师": (672, 430), "李信": (300, 569), "芈月": (276, 501)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "西施", (1027, 572)
rules._freed_cell = ((529, 372), 10)
_s121 = GameState(phase="prepare", round_no=10, gold=5, level=5, population=5, population_cap=7, population_valid=True,
                  shop_open=False, vision_units={k: "hero" for k in ("1027,572", "839,502", "672,430", "300,569", "276,501")})
_s121.hand_slots = [{"name": "嬴政", "x": 650, "y": 663}]
_s121.hand = ["嬴政"]
_d121 = rules.decide_deploy(_s121)
from agent.perception.extractors import OWN_BOARD_ROWS as _R121
_front = {tuple(c) for c in _R121[0] + _R121[1]}
print(f"[嬴政站位] {_d121 and (_d121.get('piece'), _d121.get('dst'))}（期望 后排，不是腾出的前排 (529,372)）")
if not _d121 or tuple(_d121.get("dst")) in _front:
    failures.append(f"后排输出被放进前排：{_d121}")
rules.reset_match()

# ---- 场景 122（2026-09-24 match14：多用视觉检测模型）：视觉连续多帧没看到 → 回手（牌在手）或离场 ----
rules.reset_match()
rules._round_seen = 9
rules._field = {"芈月": (276, 501), "西施": (1027, 572), "老夫子": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
_s122 = GameState(phase="prepare", round_no=9, shop_open=False, vision_units={"1027,572": "hero"})
_s122.hand_slots = [{"name": "芈月", "x": 700, "y": 663}]
for _ in range(3):
    rules.vision_absence(_s122)
_mid = ("芈月" not in rules._field and "芈月" in rules._awakened, "老夫子" in rules._field)
for _ in range(3):
    rules.vision_absence(_s122)
_end = ("老夫子" not in rules._field, "西施" in rules._field)
print(f"[视觉判离场] 3 帧后 芈月回手={_mid[0]} 老夫子还在={_mid[1]}；6 帧后 老夫子删除={_end[0]} 西施保留={_end[1]}（期望 T/T/T/T）")
if not (_mid[0] and _mid[1] and _end[0] and _end[1]):
    failures.append(f"视觉判离场不对：{_mid} {_end}")
rules.reset_match()

# ---- 场景 115：Laya 输入的 train/serve 一致 —— f4 学生看到的文本不能因为实战多了场上事实（board_facts）而变 ----
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))
import features as _F115  # noqa: E402
_act = _F115.ACTIVE
_F115.ACTIVE = "f4"
_c1 = {"round": 5, "gold": 9, "level": 3, "carry": "曜",
       "field": {"曜": {"lv": 12, "awk_left": 1, "keep": True, "late": True}, "牛魔": {"lv": 3, "awk_left": 3}}}
_c0 = dict(_c1, field={"曜": {}, "牛魔": {}})
_same = (_F115.state_text(_c1) == _F115.state_text(_c0)
         and _F115.q_shop([(0, "曜", 3)], _c1, True, True) == _F115.q_shop([(0, "曜", 3)], _c0, True, True))
_F115.ACTIVE = "f5"
_f5_has = "awaken 2/3" in _F115.state_text(_c1)
_c2 = dict(_c1, round=8)
_f5_has = _f5_has and "behind schedule" not in _F115.state_text(_c2)      # f6 才有的行不能漏进 f5（线上 R4B 是 f5）
_F115.ACTIVE = "f6"
_f5_has = _f5_has and "BEHIND" in _F115.state_text(_c2)
_F115.ACTIVE = _act
print(f"[Laya 输入一致] f4 带/不带场上事实文本相同={_same}；f5 含觉醒进度={_f5_has}（期望 True/True）")
if not (_same and _f5_has):
    failures.append(f"Laya 特征版本隔离不对：f4 一致={_same} f5 有觉醒={_f5_has}")

# ---- 场景 123：学生不退回规则（2026-09-25）—— 首选做不了（能量不够）就执行学生自己的下一个选择 ----
from agent.decision.laya_policy import LayaPolicy  # noqa: E402
from agent.perception.state import ShopSlot  # noqa: E402
_lp = object.__new__(LayaPolicy)          # 不加载模型，只测选择逻辑
_lp.rules, _lp._paid_refresh = rules, {}
rules.reset_match()
_s123 = GameState(phase="prepare", round_no=5, gold=4, level=3, level_cost=8, level_btn=(150, 600), shop_open=True,
                  shop=[ShopSlot(index=0, piece="孙膑", cost=3), ShopSlot(index=1, piece="东皇太一", cost=9)])
_r_ok = _lp._shop_action(_s123, "buy0")[0]            # 3 ≤ 4：能买（不再问规则"值不值/有没有位置"）
_r_no = _lp._shop_action(_s123, "buy1")[0]            # 9 > 4：违规
_r_lv = _lp._shop_action(_s123, "levelup")[0]         # 升级费 8 > 4：违规
_r_rf = _lp._shop_action(_s123, "refresh")[0]         # 4 能量：2026-10-15 起"刷完 <3 不刷"关掉（A12，交给学生）→ 照刷
_r_rf = None if _r_rf and _r_rf.get("type") == "refresh_shop" else _r_rf
_s123b = GameState(phase="prepare", round_no=5, gold=5, level=3, level_cost=8, level_btn=(150, 600), shop_open=True,
                   shop=[ShopSlot(index=0, piece="孙膑", cost=3)])
_r_rf5 = _lp._shop_action(_s123b, "refresh")[0]       # 5 能量：刷完剩 3 还买得起 → 刷（付费刷新没有次数上限）
print(f"[学生不退回规则] 买孙膑={bool(_r_ok)} 买东皇(钱不够)={_r_no} 升级(钱不够)={_r_lv} 4能量刷新={_r_rf} 5能量刷新={bool(_r_rf5)}"
      f"（期望 True/None/None/None/True）")
if not (_r_ok and _r_no is None and _r_lv is None and _r_rf is None and _r_rf5):
    failures.append(f"学生动作合法性判断不对：{_r_ok} {_r_no} {_r_lv} {_r_rf}")
rules.reset_match()

# ---- 场景 126：弦音（1-3 阶觉醒进度 +1）不给已觉醒的人（第四轮第 1 局拖给已觉醒的亚连两次被拒） ----
rules.reset_match()
rules._field = {"亚连": (783, 438), "孙膑": (300, 569), "云中君": (451, 433)}
rules._carry_piece = "亚连"
rules._ever_awakened = {"亚连", "孙膑"}
_t126 = rules._reviewed_target(GameState(phase="prepare", round_no=15), "弦音", rules.usage_reviewed.get("弦音") or {})
print(f"[弦音不给已觉醒] 目标={_t126}（期望 云中君：亚连/孙膑都已觉醒）")
if not (isinstance(_t126, tuple) and _t126[0] == "云中君"):
    failures.append(f"弦音目标不对：{_t126}")
rules._ever_awakened = {"亚连", "孙膑", "云中君"}
_t126b = rules._reviewed_target(GameState(phase="prepare", round_no=15), "弦音", rules.usage_reviewed.get("弦音") or {})
print(f"[弦音没对象就留着] 目标={_t126b}（期望 skip）")
if _t126b != "skip":
    failures.append(f"弦音全觉醒时应留着：{_t126b}")
rules.reset_match()

# ---- 场景 127：变身牌（霜寒进化等）不会被通用分支拖给主C ----
_ok127 = bool(rules.TRANSFORM_HERO_RE.search("解冻后，英雄变为同阵营高1阶的随机英雄。")) and     rules.usage_reviewed.get("霜寒进化", {}).get("target") == "hold"
print(f"[变身牌先留着] 认得出+审定为 hold={_ok127}（期望 True）")
if not _ok127:
    failures.append("变身牌保护不对")

# ---- 场景 124：盛华（无法拆卸 + 永久加血）只给留到最后的人，不给消耗件 ----
rules.reset_match()
rules._field = {"弈星": (433, 569), "嬴政": (617, 372)}
rules._round_seen = 6
_g1 = rules._equip_allowed("盛华", "弈星")
_g2 = rules._equip_allowed("红莲炎甲", "弈星") if "红莲炎甲" in rules.pieces else True
print(f"[盛华只给留到最后的] 弈星可穿盛华={_g1} 普通装备={_g2}（期望 False/True）")
if _g1 or not _g2:
    failures.append(f"盛华穿戴对象不对：{_g1} {_g2}")
rules.reset_match()

# ---- 场景 125：对账删人按视觉 —— 人口 2、账本 3，视觉这帧没看到的是最早加入的那个，就删它 ----
rules.reset_match()
rules._field = {"老夫子": (433, 569), "西施": (617, 372), "墨子": (1027, 572)}
_s125 = GameState(phase="prepare", round_no=7, shop_open=False, population=2, population_valid=True,
                  vision_units={cell_key((617, 372)): "hero", cell_key((1027, 572)): "hero"})
rules.reconcile(_s125); rules.reconcile(_s125)
_ok125 = set(rules._field) == {"西施", "墨子"}
print(f"[对账按视觉删] 删完账本={sorted(rules._field)}（期望 删掉视觉没看到的老夫子）")
if not _ok125:
    failures.append(f"对账没按视觉删：{rules._field}")
rules.reset_match()

# ---- 场景 128：装备按伤害类型 —— 主C 孙悟空（物理近战、只有「前排」标签）不能被当成坦克拿防御装 ----
rules.reset_match()
rules._field = {"孙悟空": (319, 375), "牛魔": (617, 372), "诸葛亮": (1027, 572)}
rules._carry_piece = "孙悟空"
_r128 = (rules.equip_role_for("孙悟空"), rules.equip_role_for("牛魔"), rules.equip_role_for("诸葛亮"))
_t128 = rules._reviewed_target(GameState(phase="prepare", round_no=12), "守护者之铠", rules.usage_reviewed.get("守护者之铠") or {})
from agent.perception.extractors import pick_equip_index as _pei  # noqa: E402
_p128 = (_pei(PROJECT_DIR / "data", "圣者法典", "carry:孙悟空")[1], _pei(PROJECT_DIR / "data", "大棒", "carry:诸葛亮")[1])
print(f"[装备按伤害类型] 角色={_r128} 防御装给={_t128[0] if isinstance(_t128, tuple) else _t128} 铸造={_p128}"
      "（期望 carry:孙悟空/tank/carry:诸葛亮、牛魔、炽热支配者/自然法杖）")
if not (_r128 == ("carry:孙悟空", "tank", "carry:诸葛亮") and isinstance(_t128, tuple) and _t128[0] == "牛魔"
        and _p128 == ("炽热支配者", "自然法杖")):
    failures.append(f"装备角色/目标不对：{_r128} {_t128} {_p128}")
rules.reset_match()

print()


# ---- 场景 129（2026-09-25 第四轮第 5 局：能量不足仍尝试购买）：价格读不到也要查钱 ----
rules.reset_match()
rules._round_seen = 3
s129 = GameState(phase="prepare", round_no=3, gold=0, population=4, population_cap=5, population_valid=True, shop_open=True)
s129.shop = [ShopSlot(0, "简易协作", None, "")]
d129 = rules.decide_buy(s129)
print(f"[能量 0、价格读不到] {d129}（期望 None：简易协作卡库标价 1）")
if d129 is not None:
    failures.append(f"能量 0 时仍想买价格读不到的牌：{d129}")
if rules.est_price("简易协作", None) != (1, True) or rules.est_price("孙膑", None) != (3, True):
    failures.append(f"估价不对：{rules.est_price('简易协作', None)} / {rules.est_price('孙膑', None)}")

# ---- 场景 130（2026-09-25：逐步去掉规则引擎）：主C 交给 Laya —— 只查合法性；规则不再自己换人 ----
rules.reset_match()
rules._round_seen = 9
rules._field = {"曜": (617, 372), "孙悟空": (529, 372), "牛魔": (433, 569)}
rules._carry_piece, rules._carry = "曜", (617, 372)
s130 = GameState(phase="prepare", round_no=9, population=3, population_cap=6, population_valid=True)
rules.carry_by_laya = True
rules.choose_carry(s130)
_keep130 = rules._carry_piece
_bad130 = rules.apply_carry("牛魔", s130)            # 坦克不能当主C
_ok130 = rules.apply_carry("孙悟空", s130)
_again130 = rules.apply_carry("曜", s130)            # 本回合已换过
print(f"[Laya 定主C] 规则不动={_keep130} 坦克={_bad130} 孙悟空={_ok130} 同回合再换={_again130} 老主C={rules._prev_carry}"
      "（期望 曜/False/True/False/曜）")
if not (_keep130 == "曜" and not _bad130 and _ok130 and not _again130 and rules._carry_piece == "孙悟空"
        and rules._prev_carry == "曜"):
    failures.append(f"apply_carry 不对：{_keep130} {_bad130} {_ok130} {_again130} {rules._carry_piece}")
rules.carry_by_laya = False
rules.reset_match()

# ---- 场景 131（2026-09-25 第 2 局）：swap 为谁腾的位，本回合上场先上谁（别让"吃登场效果"的杂牌抢走）----
rules.reset_match()
rules._round_seen = 10
rules._field = {"鲁班七号": (617, 372), "苏烈": (529, 372)}
s131 = GameState(phase="prepare", round_no=10, population=2, population_cap=3, population_valid=True, shop_open=False)
s131.hand_slots = [{"name": "甄姬", "x": 500, "y": 680}, {"name": "百里守约", "x": 600, "y": 680}]
rules.deploy_first = (10, "百里守约")
d131 = rules.decide_deploy(s131)
print(f"[腾位先上它] {d131 and d131.get('piece')}（期望 百里守约）")
if not d131 or d131.get("piece") != "百里守约":
    failures.append(f"腾位后没有先上 swap 的目标：{d131}")
rules.reset_match()

# ---- 场景 132（2026-09-26：觉醒英雄回到手牌、却先上其他英雄，问题仍存在且严重）：腾位目标 vs 觉醒回手 → 觉醒先上 ----
rules.reset_match()
rules._round_seen = 9
rules._field = {"鲁班七号": (617, 372), "苏烈": (529, 372)}
s132 = GameState(phase="prepare", round_no=9, population=2, population_cap=4, population_valid=True, shop_open=False)
s132.hand_slots = [{"name": "大司命", "x": 546, "y": 663}, {"name": "百里守约", "x": 1045, "y": 663}]
rules.deploy_first = (9, "大司命")
rules._awakened.add("百里守约")
d132 = rules.decide_deploy(s132)
print(f"[觉醒先于腾位目标] {d132 and d132.get('piece')}（期望 百里守约）")
if not d132 or d132.get("piece") != "百里守约":
    failures.append(f"觉醒回手没有最先上场：{d132}")
rules.reset_match()

# ---- 场景 133（2026-09-26 第四轮第 5 局：换主C 判定出来了，但没有实际执行）----
# Laya 第 6 回合把主C 从曜换成干将莫邪：① 当回合就收尾（不等"当满一回合"）② 商店开着先关店，不许隔着商店卖人
rules.reset_match()
rules._round_seen = 6
rules._field = {"曜": (617, 372), "干将莫邪": (529, 372), "牛魔": (718, 371)}
rules._carry_piece, rules._carry = "曜", (617, 372)
rules._maple_bound = "曜"
s133 = GameState(phase="prepare", round_no=6, population=3, population_cap=5, population_valid=True, shop_open=True)
s133.hand_slots = [{"name": "枫叶舞", "x": 264, "y": 663, "ready": False}]
ok133 = rules.apply_carry("干将莫邪", s133, why="Laya 0.53")   # 和线上 LayaPolicy 传的一样（带概率）
a133 = rules.decide_sell_unit(s133)
s133.shop_open = False
b133 = rules.decide_sell_unit(s133)
print(f"[Laya 换主C 收尾] 换={ok133} 商店开着→{a133 and a133.get('type')} 关着→{b133 and (b133.get('type'), b133.get('piece'))}"
      "（期望 True / toggle_shop / 卖曜）")
if not ok133 or not a133 or a133.get("type") != "toggle_shop" or not b133 or b133.get("piece") != "曜":
    failures.append(f"Laya 换主C 后没有当回合关店卖人换绑：{ok133} {a133} {b133}")
rules.reset_match()

# ---- 场景 134（2026-09-26：装备箱直接使用即可，本质是挑选一个装备）----
rules.reset_match()
rules._round_seen = 10
rules._field = {"艾琳": (617, 372), "牛魔": (529, 372)}
rules._carry_piece, rules._carry = "艾琳", (617, 372)
s134 = GameState(phase="prepare", round_no=10, gold=5, population=2, population_cap=7, population_valid=True, shop_open=False)
s134.hand_slots = [{"name": "装备箱", "x": 579, "y": 663}]
s134.hand = ["装备箱"]
a134 = rules.decide_use_cards(s134)
print(f"[装备箱] {a134 and (a134.get('type'), a134.get('piece'))}（期望 打出装备箱）")
if not a134 or a134.get("piece") != "装备箱":
    failures.append(f"装备箱没有直接使用：{a134}")
rules.reset_match()

# ---- 场景 135（2026-09-26 第四轮第 6 局）：人口 5/5 = 账本 5 人时，视觉连续漏看孙尚香 ≠ 觉醒回手 ----
rules.reset_match()
rules._round_seen = 4
rules._field = {"孙尚香": (300, 569), "苏烈": (672, 430), "牛魔": (529, 372), "曜": (617, 372), "弈星": (718, 371)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
for _ in range(5):
    s135 = GameState(phase="prepare", round_no=4, population=5, population_cap=5, population_valid=True, shop_open=False)
    s135.vision_units = {"672,430": 1, "529,372": 1, "617,372": 1, "718,371": 1}
    s135.hand_slots = [{"name": "孙尚香", "x": 650, "y": 663}]
    rules.vision_absence(s135)
print(f"[漏检≠觉醒回手] 孙尚香在账={'孙尚香' in rules._field} 觉醒={'孙尚香' in rules._awakened}（期望 True / False）")
if "孙尚香" not in rules._field or "孙尚香" in rules._awakened:
    failures.append("人口一个不少时，视觉漏检被判成了觉醒回手")
# 人口真的少了一个 → 照常判觉醒回手；之后人口比账本多、原格有人 → 撤销
for _ in range(3):
    s135 = GameState(phase="prepare", round_no=4, population=4, population_cap=5, population_valid=True, shop_open=False)
    s135.vision_units = {"672,430": 1, "529,372": 1, "617,372": 1, "718,371": 1}
    s135.hand_slots = [{"name": "孙尚香", "x": 650, "y": 663}]
    rules.vision_absence(s135)
_ret = "孙尚香" in rules._awakened and "孙尚香" not in rules._field
s135 = GameState(phase="prepare", round_no=4, population=5, population_cap=5, population_valid=True, shop_open=False)
s135.vision_units = {"300,569": 1, "672,430": 1, "529,372": 1, "617,372": 1, "718,371": 1}
s135.hand_slots = [{"name": "孙尚香", "x": 650, "y": 663}]
rules.vision_absence(s135)
print(f"[觉醒回手→撤销] 判回手={_ret} 撤销后在账={'孙尚香' in rules._field}（期望 True / True）")
if not _ret or "孙尚香" not in rules._field or "孙尚香" in rules._awakened:
    failures.append("人口少一个时没判觉醒回手，或人口/视觉证伪后没撤销")
rules.reset_match()

# ---- 场景 136（2026-09-26 第四轮第 6 局：主C 没有换上）：Laya 换主C 时，老主C 觉醒过也照样卖掉换绑 ----
rules.reset_match()
rules._round_seen = 9
rules._field = {"孙尚香": (300, 569), "司空震": (529, 372), "牛魔": (718, 371)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "孙尚香", (300, 569)
rules._maple_bound = "孙尚香"
rules._ever_awakened.add("孙尚香")
s136 = GameState(phase="prepare", round_no=9, population=3, population_cap=6, population_valid=True, shop_open=False)
s136.hand_slots = [{"name": "枫叶舞", "x": 264, "y": 663, "ready": False}]
rules.apply_carry("司空震", s136, why="Laya 0.53")   # 和线上 LayaPolicy 传的一样（带概率）
a136 = rules.decide_sell_unit(s136)
print(f"[Laya 换主C·老主C 觉醒过] {a136 and (a136.get('type'), a136.get('piece'))}（期望 卖孙尚香）")
if not a136 or a136.get("piece") != "孙尚香":
    failures.append(f"Laya 换主C 后没有卖掉觉醒过的老主C 换绑：{a136}")
rules.reset_match()

# ---- 场景 137（2026-09-26：手牌里没有未来规划要用的英雄卡可以卖掉，不要全堆在手牌里影响判断）----
import time as _time137
rules.reset_match()
rules._round_seen = 7
rules._field = {"孙尚香": (300, 569), "牛魔": (529, 372), "苏烈": (672, 430)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "孙尚香", (300, 569)
s137 = GameState(phase="prepare", round_no=7, population=3, population_cap=3, population_valid=True, shop_open=False)
s137.hand_slots = [{"name": "芈月", "x": 600, "y": 663}, {"name": "艾琳", "x": 700, "y": 663}]
rules._hand_seen_ts = {"芈月": _time137.time() - 10, "艾琳": _time137.time() - 10}
a137 = rules._idle_hand_hero(s137)
print(f"[卖不在规划里的英雄手牌] {a137 and (a137.get('type'), a137.get('piece'))}；艾琳在规划={rules._planned_hero('艾琳')}（期望 卖芈月 / True）")
if not a137 or a137.get("piece") != "芈月" or not rules._planned_hero("艾琳"):
    failures.append(f"闲置英雄手牌处理不对：{a137}")
rules.reset_match()

# ---- 场景 138（第四轮第 9 局）：推荐阵容写着老主C，Laya 已换人 —— 收尾照样卖老主C 换绑 ----
rules.reset_match()
rules.ENGINE_ENABLED = True
rules._round_seen = 8
rules._field = {"干将莫邪": (300, 569), "诸葛亮": (529, 372), "张良": (718, 371)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "干将莫邪", (300, 569)
rules._active_comp = {"carry": "干将莫邪", "core": ["张良"]}
rules._maple_bound = "干将莫邪"
s138 = GameState(phase="prepare", round_no=8, population=3, population_cap=6, population_valid=True, shop_open=False)
s138.hand_slots = [{"name": "枫叶舞", "x": 264, "y": 663, "ready": False}]
rules.apply_carry("诸葛亮", s138, why="Laya 0.57")
a138 = rules.decide_sell_unit(s138)
# 2026-09-26 晚"主C 跟引擎"：干将+张良 = 大河开团 2 个放大器、第 8 回合 → 引擎锁定，干将是零件 → **不为换绑卖它**
print(f"[阵容主C≠Laya 主C 的收尾·引擎锁定] 锁={rules.engine_lock()!r} {a138 and (a138.get('type'), a138.get('piece'))}"
      f"（期望 river_opening / 不卖）")
if rules.engine_lock() != "river_opening" or (a138 and a138.get("type") == "sell_unit"):
    failures.append(f"引擎锁定后收尾还是卖了零件：{a138}")
print(f"[引擎锁定 → 主C 候选] {sorted(rules.carry_allowed() or [])}（期望 大河落点 + 现任诸葛亮）")
if not rules.carry_allowed() or "诸葛亮" not in rules.carry_allowed() or "韩信" not in rules.carry_allowed():
    failures.append(f"主C 候选限引擎落点不对：{rules.carry_allowed()}")
# 引擎没转起来（只有干将一个零件）→ 照旧卖老主C 换绑
rules._field = {"干将莫邪": (300, 569), "诸葛亮": (529, 372), "安琪拉": (718, 371)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._active_comp = {"carry": "干将莫邪", "core": ["安琪拉"]}
a138b = rules.decide_sell_unit(s138)
print(f"[阵容主C≠Laya 主C 的收尾·干将不是零件] 锁={rules.engine_lock()!r} {a138b and (a138b.get('type'), a138b.get('piece'))}（期望 sunset_core / 卖干将莫邪）")
if rules._is_engine_part("干将莫邪") or not a138b or a138b.get("piece") != "干将莫邪":
    failures.append(f"没引擎时 Laya 换主C 收尾被挡：{a138b}")
rules.reset_match()

# ---- 场景 139（第四轮第 11 局）：主C 在账本里"走丢"（记错格被撤掉，不是我们卖的）—— 规则不能自己换主C ----
rules.reset_match()
rules._round_seen = 8
rules.carry_by_laya = True
rules._field = {"亚连": (529, 372), "少司缘": (276, 501), "牛魔": (718, 371)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "虞姬", None      # 2026-10-20：安琪拉不能当主C（route_avoid_weak）→ 场景改用虞姬
s139 = GameState(phase="prepare", round_no=8, population=4, population_cap=6, population_valid=True, shop_open=False)
rules.choose_carry(s139)
kept = rules._carry_piece
rules._sold_names.add("虞姬"); rules.choose_carry(s139)
print(f"[主C 走丢] 没卖→保留={kept}；卖掉后→{rules._carry_piece}（期望 虞姬 / 换人）")
if kept != "虞姬" or rules._carry_piece == "虞姬":
    failures.append(f"主C 走丢时规则换人 / 卖掉后不补：{kept} {rules._carry_piece}")
rules.carry_by_laya = False
rules.reset_match()

# ---- 140~142：等级引擎（2026-09-26）----
rules.reset_match()
rules.ENGINE_ENABLED = True        # 引擎规则默认关着（第 15 局锁死在干将上），这几个场景显式打开
rules._active_comp = None
rules._round_seen = 8
rules._field = {"狂铁": (529, 372), "安琪拉": (433, 569), "海诺": (555, 569), "苏烈": (617, 372)}
_ek = rules.engine_key()
_w = rules._weakest_on_field()
print(f"[引擎零件] 引擎={_ek} 狂铁零件={rules._is_engine_part('狂铁')} 苏烈零件={rules._is_engine_part('苏烈')} "
      f"最弱={_w and _w[0]} 狂铁留场={rules._is_keeper('狂铁')}（期望 sunset_core / True / False / 苏烈 / True）")
if _ek != "sunset_core" or not rules._is_engine_part("狂铁") or rules._is_engine_part("苏烈")         or not _w or _w[0] != "苏烈" or not rules._is_keeper("狂铁"):
    failures.append(f"引擎零件保护：引擎={_ek} 最弱={_w}")

rules._field = {"孙尚香": (529, 372), "牛魔": (433, 569)}
print(f"[单个零件不算引擎] 引擎={rules.engine_key()!r} 孙尚香零件={rules._is_engine_part('孙尚香')}（期望 '' / False）")
if rules.engine_key() or rules._is_engine_part("孙尚香"):
    failures.append("一个孤零零的产出方被当成引擎零件保护")

rules._field = {"狂铁": (529, 372), "安琪拉": (433, 569)}
_st140 = GameState(shop_open=True, population_valid=True, phase="prepare", round_no=4, gold=10, level=2, level_cost=4,
                   population=2, population_cap=5, hp=50)
_st140.level_btn = (120, 600)
_st140.hand_slots = []
_st140.shop = [ShopSlot(0, "海诺", 3, ""), ShopSlot(1, "曜", 3, ""), ShopSlot(2, None, 3, ""), ShopSlot(3, None, 3, "")]
_a = rules.preempt_shop(_st140)
print(f"[进度表升级] {_a and _a['type']}（期望 buy_xp：第 4 回合该 3 级、才 2 级、费 4 ≤ 10）")
if not _a or _a.get("type") != "buy_xp":
    failures.append(f"进度表升级没触发: {_a}")
_st140.level = 3
_a = rules.preempt_shop(_st140)
print(f"[引擎零件先买] {_a and (_a['type'], _a.get('piece'))}（期望 buy_piece 海诺：日落海放大器）")
if not _a or _a.get("piece") != "海诺":
    failures.append(f"引擎零件没先买: {_a}")
_st140.shop = [ShopSlot(0, "曜", 3, ""), ShopSlot(1, None, 3, ""), ShopSlot(2, None, 3, ""), ShopSlot(3, None, 3, "")]
_a = rules.preempt_shop(_st140)
print(f"[不是零件不抢买] {_a}（期望 None）")
if _a:
    failures.append(f"非引擎零件被抢买: {_a}")
_st140.shop = [ShopSlot(0, "海诺", 3, ""), ShopSlot(1, None, 3, ""), ShopSlot(2, None, 3, ""), ShopSlot(3, None, 3, "")]
_st140.population, _st140.population_cap = 5, 5
_st140.hand_slots = [{"name": "亚瑟", "x": 500, "y": 660}]
_a = rules.preempt_shop(_st140)
print(f"[满员且手里有人在等 → 不买新零件] {_a}（期望 None）")
if _a:
    failures.append(f"满员还抢买零件: {_a}")
_st140.hand_slots = [{"name": n, "x": 500 + 60 * i, "y": 660} for i, n in enumerate(["古币"] * 9)]
_st140.population, _st140.population_cap = 2, 5
_a = rules.preempt_shop(_st140)
print(f"[手牌 9 张 → 不买] {_a}（期望 None）")
if _a:
    failures.append(f"手牌快满还抢买: {_a}")

# ---- 143~146：官方路线（2026-09-28：照官方推荐阵容的分回合运营路线打）----
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json") if hasattr(rules, "load_lineups") else None
_st143 = GameState(phase="prepare", round_no=3, level=3, population=4, population_cap=5, population_valid=True)
rules._field = {"李元芳": (529, 372), "盾山": (617, 372), "弈星": (433, 569)}
rules.resolve_comp(_st143)
print(f"[路线·第 3 回合只有零件] 锁={rules.route_locked()}（期望 False：只凭零件要等第 4 回合）")
if rules.route_locked():
    failures.append("第 3 回合只凭零件就锁了路线")
rules.reset_match()
# 2026-09-29 养核心：拿到路线主C 就从第 2 回合起锁（原来要等第 4 回合）
_st143.hand_slots = [{"name": "百里守约", "x": 500, "y": 660}]
rules._field = {"李元芳": (529, 372), "盾山": (617, 372), "弈星": (433, 569)}
rules.resolve_comp(_st143)
_c143 = (rules._active_comp or {}).get("name")
_rsm = rules.route_stage_members(6)
print(f"[路线·第 3 回合有主C] 阵容={_c143} 锁={rules.route_locked()} 李元芳过渡件={rules._obsolete('李元芳', 7)}"
      f"（期望 守约路线 / True / 路线要它就 False：{'李元芳' in rules.route_stage_members(7)}）")
if (not rules.route_locked() or "守约" not in (_c143 or "")
        or rules._obsolete("李元芳", 7) == ("李元芳" in rules.route_stage_members(7))):
    failures.append(f"有主C 没提前锁 / 路线要的 1 阶被当过渡件：{_c143} {rules.route_locked()} {_rsm}")
_st143.round_no = 4
rules.resolve_comp(_st143)
print(f"[路线·第 4 回合锁定] 锁={rules.route_locked()} 主C={rules.route_carry()}（期望 True / 百里守约）")
if not rules.route_locked() or rules.route_carry() != "百里守约":
    failures.append(f"第 4 回合没锁定路线：{rules.route_locked()} {rules.route_carry()}")
# 锁定后手里换成另一套的核心也不换阵容
_st143.round_no = 6
_st143.hand_slots = [{"name": n, "x": 500 + 60 * i, "y": 660} for i, n in enumerate(["干将莫邪", "张良", "云中君", "少司缘"])]
rules.resolve_comp(_st143)
print(f"[路线·锁定后不换] 阵容={(rules._active_comp or {}).get('name')}（期望 仍是 {_c143}）")
if (rules._active_comp or {}).get("name") != _c143:
    failures.append("路线锁定后阵容还是换了")
# 路线外英雄不能买，同名/路线成员能买
print(f"[路线·能不能买] 明世隐={rules.route_allows_hero('明世隐')} 貂蝉={rules.route_allows_hero('貂蝉')} "
      f"李元芳(同名)={rules.route_allows_hero('李元芳')} 古币(非英雄)={rules.route_allows_hero('古币')}（期望 True/False/True/True）")
if not rules.route_allows_hero("明世隐") or rules.route_allows_hero("貂蝉") or not rules.route_allows_hero("古币"):
    failures.append("路线买牌过滤不对")
# 遗志路线（伞海月）不参选
print(f"[路线·排除遗志] 伞海月被排除={all(rules._route_excluded(c) for c in rules.comps if '海月' in (c.get('name') or '') and '遗志' in (c.get('effect_cards') or []))}")
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False

# 场景 147：卖人没生效（视觉复查那格还站着人）→ 账本和枫叶舞绑定恢复；同一人一回合失败 2 次就不再卖
# （2026-09-28 第 26 局：卖孙膑没生效，枫叶舞整局灰着、主C 没绑上）
rules.reset_match()
rules._round_seen = 7
rules._field = {"孙膑": (300, 569), "百里守约": (1027, 572)}
rules._cell_owner = {(300, 569): "孙膑", (1027, 572): "百里守约"}
rules._maple_bound = "孙膑"
_sell = {"type": "sell_unit", "piece": "孙膑", "src": (300, 569)}
rules.note_executed(_sell)
rules.undo_sell()
_ok1 = rules._field.get("孙膑") == (300, 569) and rules._maple_bound == "孙膑" and not rules.sell_blocked("孙膑")
rules.note_executed(_sell)
rules.undo_sell()
_ok2 = rules.sell_blocked("孙膑")
print(f"[卖出撤销] 恢复={_ok1} 两次后不再卖={_ok2}")
if not (_ok1 and _ok2):
    failures.append("卖出没生效时账本没恢复 / 没止损")
rules.reset_match()


# 场景 148（2026-09-29：2-3 阶主C 定调）：只走 2~3 阶主C 的路线；主C 评分 2~3 阶加分，没搬等级手段时不换到 4~5 阶
from agent.decision import routes as _R148
_R148.CARRY_TIERS = (2, 3)
rules.reset_match()
# 2026-10-20 晚起主C 名单由棋手配置决定（data/lords/阿离/carry_pool.json；开了 4 阶的韩信 / 刘邦 / 蒙犽）：
# 关掉的主C 的路线全排除，开着的至少留下守约
from agent.decision import lord_config as _LC148
_c148 = _LC148.load()
_ex_ok = all(rules._route_excluded(lu) for lu in _R148.ali_lineups()
             if not _c148.enabled(_c148.carry_of_lineup(lu.get("key") or "") or lu.get("carry") or ""))
_keep = [lu["name"] for lu in _R148.ali_lineups() if not rules._route_excluded(lu)]
rules._round_seen = 8
rules._field = {"百里守约": (617, 372), "李信": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "百里守约", (617, 372)
_s148 = GameState(phase="prepare", round_no=8, gold=10, level=5, population=2, population_cap=7, population_valid=True)
_a = rules._carry_score("百里守约", _s148); _b = rules._carry_score("李信", _s148)
_R148.CARRY_TIERS = None
print(f"[主C 名单定调] 关掉的主C 路线全排除={_ex_ok}，剩 {len(_keep)} 套（守约在内={any('守约' in n for n in _keep)}）；"
      f"评分 守约={_a:.0f} 李信={_b:.0f}（期望 守约更高）")
if not _ex_ok or not any("守约" in n for n in _keep) or not (_a is not None and _b is not None and _a > _b):
    failures.append(f"2~3 阶主C 定调不对：{_ex_ok} {len(_keep)} {_a} {_b}")
rules.reset_match()


# 场景 149（2026-09-29：要有不同主C 阵容的备选，首选被禁时不能无路可走）：首选被禁 → 换备选；第 5 回合没拿到首选主C、备选主C 在手 → 换
from agent.decision import routes as _R149
_R149.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "50720714303640586", ("user_angela", "1744067911380998")   # 守约 / 安琪拉 / 亚连
_s149 = GameState(phase="prepare", round_no=2, level=2, population=1, population_cap=5, population_valid=True, banned_faction="河洛")
rules.resolve_comp(_s149)
_a149 = rules.route_carry()
rules.reset_match()
_s149 = GameState(phase="prepare", round_no=5, level=3, population=3, population_cap=6, population_valid=True, banned_faction="逐鹿")
_s149.hand_slots = [{"name": "亚连", "x": 500, "y": 660}]
rules.resolve_comp(_s149)
_b149 = rules.route_carry()
rules.reset_match()
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json", lord="阿离")
rules.forced_route, rules.forced_backups = "50720714303640586", ("81682962224371742", "user_angela", "1744067911380998")   # 同实战：沈梦溪在备选清单里
rules.__dict__.pop("_forced_orig", None)
_s149b = GameState(phase="prepare", round_no=5, level=3, population=3, population_cap=6, population_valid=True, banned_faction="逐鹿")
_s149b.hand_slots = [{"name": "沈梦溪", "x": 500, "y": 660}]
rules.resolve_comp(_s149b)
_d149 = rules.route_carry()
rules.reset_match()
_c149 = rules.forced_route
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R149.CARRY_TIERS = None
# 2026-10-10 改：只在**同体系**里换（蒙特卡洛 tools/sim_route_vs_flex.py：体系内谁先到用谁最好）——手里只有体系外的亚连不换，有沈梦溪就换
# 2026-10-13 再改（不定死一个路线，根据获取的其他卡牌灵活切换主C）：体系外的指定路线也能换，同体系优先
# 2026-10-14 改为算账（rules.route_eta：用数值计算性价比，不写死规则）：
# 只在候选路线预计凑齐回合明显更少（少 1 回合以上）时才换 —— 只有沈梦溪本人、配套一张没有时并不比守约快 → 不换
_eta_s = rules.route_eta("81682962224371742", _s149b, {"沈梦溪"}); _eta_y = rules.route_eta("50720714303640586", _s149b, {"沈梦溪"})
print(f"[备选路线] 禁河洛→主C {_a149}（期望 安琪拉：只剩她）；第 5 回合没守约、手里有亚连→{_b149}（期望 亚连）；"
      f"手里只有沈梦溪→{_d149}（算账：沈梦溪路线 {_eta_s} 回合 vs 守约 {_eta_y} 回合）；新局回到首选 {_c149}")
_want_d = "沈梦溪" if _eta_s + 0.5 < _eta_y else "百里守约"
# 2026-10-20：安琪拉只在没有别的人选时当主C（禁河洛、指定路线里只剩她）→ 这里仍落到安琪拉路线
# 2026-10-21：关掉的主C（安琪拉）不再锁路线（第 854 局马可波罗）→ 禁河洛、指定路线只剩安琪拉时不锁（""）
if _a149 not in ("", None) or _b149 != "亚连" or _d149 != _want_d or _c149 != "50720714303640586":
    failures.append(f"备选路线不对：{_a149} {_b149} {_d149} {_c149}")
rules.reset_match()


# 场景 150（2026-09-30 轮换第 3 局：主C 仍是李元芳）：狄仁杰路线 9 回合没刷到主C，枫叶舞一直挂李元芳
#   → 同体系备选的主C（守约）出现也买；商店没有要的牌、钱够 → 刷新找主C；主C 到手后不再刷
from agent.decision import routes as _R150
_R150.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ("50720714303640586", "81682962224371742", "user_angela")
rules.set_route("32706315349545018", 3, why="测试")
rules._field = {"李元芳": (300, 569), "盾山": (556, 430), "程咬金": (783, 438), "明世隐": (390, 502), "苏烈": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._round_seen = 7
_s150 = GameState(phase="prepare", round_no=7, gold=27, level=4, level_cost=30, population=5, population_cap=6,
                  population_valid=True, shop_open=True, banned_faction="三分之地")
_s150.shop = [ShopSlot(0, "百里守约", 3, ""), ShopSlot(1, "貂蝉", 3, ""), ShopSlot(2, None, 3, ""), ShopSlot(3, None, 3, "")]
_a150 = rules.preempt_shop(_s150)
_s150.shop = [ShopSlot(0, "貂蝉", 3, ""), ShopSlot(1, "甄姬", 3, ""), ShopSlot(2, None, 3, ""), ShopSlot(3, None, 3, "")]
_s150.refresh_cost = 2
_b150 = rules.preempt_shop(_s150)
_s150.hand_slots = [{"name": "狄仁杰", "x": 500, "y": 660}]
_c150 = rules.preempt_shop(_s150)
rules.forced_backups = ()
_R150.CARRY_TIERS = None
print(f"[找主C] 商店有守约→{_a150 and (_a150['type'], _a150.get('piece'))}（期望 买守约）；"
      f"没有要的牌→{_b150 and _b150['type']}（期望 refresh_shop）；狄仁杰到手→{_c150 and _c150.get('why')}（期望 不再为主C 刷；搜牌回合里可以为缺的钟馗刷）")
if not (_a150 and _a150.get("piece") == "百里守约") or not (_b150 and _b150["type"] == "refresh_shop")         or (_c150 and "狄仁杰" in (_c150.get("why") or "")):
    failures.append(f"找主C 不对：{_a150} {_b150} {_c150}")
rules.reset_match()


# 场景 151（2026-09-30：看哪个棋手等级的商店刷出得多，专门留 1-2 个回合升级凑配套阵容）
from agent.decision import routes as _R151
_R151.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("32706315349545018", 3, why="测试")          # 狄仁杰路线（3 阶主C）
rules._field = {"李元芳": (300, 569), "盾山": (556, 430)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
_mk = lambda rnd, lv, gold: GameState(phase="prepare", round_no=rnd, gold=gold, level=lv, level_cost=8, level_btn=(150, 640),
                                       population=2, population_cap=5, population_valid=True, shop_open=True, refresh_cost=2,
                                       shop=[ShopSlot(0, "貂蝉", 3, ""), ShopSlot(1, None, 3, ""), ShopSlot(2, None, 3, ""), ShopSlot(3, None, 3, "")])
_rl = rules.roll_level(_mk(5, 2, 20))
_a = rules.preempt_shop(_mk(5, 2, 20))          # 2 级 < 搜牌等级 → 升级
_b = rules.preempt_shop(_mk(5, _rl, 20))        # 到了 → 搜牌回合 1：刷新
rules._round_seen = 5
if _b: rules.note_executed(_b)                   # 真刷了才占名额
_c = rules.preempt_shop(_mk(6, _rl, 20))        # 搜牌回合 2
rules._round_seen = 6
if _c: rules.note_executed(_c)
_d = rules.preempt_shop(_mk(10, _rl, 20))       # 两个搜牌回合用完：第 10 回合进度表该 5 级 → 升级
_R151.CARRY_TIERS = None
print(f"[搜牌等级] 狄仁杰路线搜牌等级={_rl}（期望 4）；2 级→{_a and _a['type']}（期望 buy_xp）；到了→{_b and _b['type']}/{_c and _c['type']}"
      f"（期望 refresh_shop 两回合）；搜牌回合用完→{_d and _d['type']}（期望 buy_xp）")
if _rl != 4 or not (_a and _a["type"] == "buy_xp") or not (_b and _b["type"] == "refresh_shop")         or not (_c and _c["type"] == "refresh_shop") or not (_d and _d["type"] == "buy_xp"):
    failures.append(f"搜牌等级/搜牌回合不对：{_rl} {_a} {_b} {_c} {_d}")
rules.reset_match()


# 场景 152（2026-09-30：读临阵磨枪的技能确定使用者；限定使用者的牌要有规矩；实战：Laya 把临阵磨枪给了狄仁杰）
rules.reset_match()
_cases = [("临阵磨枪", "狄仁杰", False), ("临阵磨枪", "李元芳", True), ("临阵磨枪", "孙悟空", True),
          ("古币", "诸葛亮", False), ("古币", "百里守约", True),
          ("弦音", "海月", False), ("弦音", "孙膑", True),
          ("日渊", "曜", False),
          ("月影之力", "亚连", False), ("月影之力", "露娜", True)]
_bad = []
for _card, _hero, _want in _cases:
    if _card not in rules.pieces and not rules._card_text(_card):
        continue
    _ok, _why = rules.card_target_ok(_card, _hero)
    if _ok != _want:
        _bad.append((_card, _hero, _ok, _why))
print(f"[限定使用者] {len(_cases)} 例，不对的 {_bad}（例：临阵磨枪→狄仁杰 {rules.card_target_ok('临阵磨枪', '狄仁杰')}）")
if _bad:
    failures.append(f"限定使用者判断不对：{_bad}")
rules.reset_match()


# 场景 153（2026-09-30 轮换第 5 局：该换主C 的局面没换）：
#   ① 2~3 阶定调时 1 阶的弈星不当主C；② 第 4 回合守约在商店、9 能量、进度表落后 → 先买守约，不是先升级
from agent.decision import routes as _R153
_R153.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("50720714303640586", 1, why="测试")
_e1 = rules._carry_eligible("弈星")
_s153 = GameState(phase="prepare", round_no=4, gold=9, level=2, level_cost=7, level_btn=(65, 555), population=4,
                  population_cap=5, population_valid=True, shop_open=True,
                  shop=[ShopSlot(0, "亚连", 3, ""), ShopSlot(1, "百里守约", 3, ""), ShopSlot(2, "李元芳", 3, ""), ShopSlot(3, None, 3, "")])
rules._field = {"李元芳": (300, 569), "孙膑": (433, 569), "弈星": (390, 502)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
_a153 = rules.preempt_shop(_s153)
_R153.CARRY_TIERS = None
print(f"[主C 先买] 弈星能当主C={_e1}（期望 False）；第 4 回合守约在店、落后进度 → {_a153 and (_a153['type'], _a153.get('piece'))}（期望 买守约）")
if _e1 or not (_a153 and _a153.get("piece") == "百里守约"):
    failures.append(f"主C 先买 / 1 阶不当主C 不对：{_e1} {_a153}")
rules.reset_match()


# 场景 154（2026-09-30：援护战术在场上没有对应英雄时不用，不能因此卡死）
rules.reset_match()
rules._round_seen = 10
rules._field = {"狄仁杰": (839, 502), "李元芳": (300, 569), "盾山": (617, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
_s154 = GameState(phase="prepare", round_no=10, gold=15, level=5, population=3, population_cap=7, population_valid=True)
_s154.hand_slots = [{"name": "援护战术", "x": 863, "y": 661}]
_a154 = rules._pick_self_cast(_s154)
rules._field["孙膑"] = (433, 569); rules._cell_owner[(433, 569)] = "孙膑"
_b154 = rules._pick_self_cast(_s154)
# 自用牌被拒：不撤销落点那格的记录（原来把主C 狄仁杰撤掉了）
rules.confirm_drag({"type": "drag_card", "piece": "援护战术", "src": (863, 661), "dst": (839, 502), "kind": "效果牌", "self_cast": True}, False)
_c154 = rules._field.get("狄仁杰") == (839, 502) and (839, 502) not in rules._busy_cells
print(f"[群体牌] 没有逐鹿→{_a154 and _a154['piece']}（期望 None）；有孙膑(逐鹿)→{_b154 and _b154['piece']}（期望 援护战术）；"
      f"被拒后狄仁杰记录还在={_c154}（期望 True）")
if _a154 or not (_b154 and _b154["piece"] == "援护战术") or not _c154:
    failures.append(f"援护战术类不对：{_a154} {_b154} {_c154}")
rules.reset_match()


# 场景 155（2026-09-30：后期盲目刷新、不升到 6 级）：第 12 回合起 5 级、钱够 → 先升 6 级，不开搜牌回合
from agent.decision import routes as _R155
_R155.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("50720714303640586", 3, why="测试")
rules._field = {"百里守约": (617, 372), "李元芳": (300, 569), "盾山": (556, 430), "钟馗": (529, 372), "铠": (718, 371), "明世隐": (672, 430)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
_s155 = GameState(phase="prepare", round_no=13, gold=26, level=5, level_cost=22, level_btn=(66, 555), population=6,
                  population_cap=7, population_valid=True, shop_open=True, refresh_cost=1,
                  shop=[ShopSlot(0, "貂蝉", 3, ""), ShopSlot(1, None, 3, ""), ShopSlot(2, None, 3, ""), ShopSlot(3, None, 3, "")])
_a155 = rules.preempt_shop(_s155)
_R155.CARRY_TIERS = None
print(f"[升 6 级] 第 13 回合 5 级、26 能量、升级费 22 → {_a155 and _a155['type']}（期望 buy_xp）")
if not (_a155 and _a155["type"] == "buy_xp"):
    failures.append(f"后期不升 6 级：{_a155}")
rules.reset_match()


# 场景 156（2026-09-30 轮换第 9 局，第 5 名）：① 觉醒主C 百里玄策站在 (300,569)，这格合成失败够次数 —— 视觉看得见就不撤销记录；
#   ② 按"同体系主C"买进的沈梦溪不能被当成闲置手牌卖掉
from agent.decision import routes as _R156
_R156.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_backups = ("81682962224371742", "32706315349545018")
rules.set_route("50720714303640586", 3, why="测试")          # 守约路线，沈梦溪/狄仁杰是同体系
_planned = rules._planned_hero("沈梦溪")
rules.set_route("122215359354172436", 6, why="测试")          # 玄策路线
rules._round_seen = 9
rules._field = {"百里玄策": (300, 569), "苏烈": (529, 372), "盾山": (617, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece = "百里玄策"
rules._cell_fails[(300, 569)] = 9
_s156 = GameState(phase="prepare", round_no=9, gold=17, level=4, population=3, population_cap=6, population_valid=True)
_s156.hand_slots = [{"name": "百里玄策", "x": 500, "y": 660}]
_s156.vision_units = {"300,569": "hero", "529,372": "hero", "617,372": "hero"}
rules.decide_upgrade(_s156)
_kept = rules._field.get("百里玄策") == (300, 569)
rules.forced_backups = ()
_R156.CARRY_TIERS = None
print(f"[第9局] 同体系主C 沈梦溪算阵容规划={_planned}（期望 True）；视觉看得见的玄策记录保留={_kept}（期望 True）")
if not _planned or not _kept:
    failures.append(f"第 9 局两处 bug：{_planned} {_kept}")
rules.reset_match()


# 场景 157（2026-09-30 轮换第 11 局，第 2 名）：① 临时主C（不是路线主C）不为换绑枫叶舞卖人；② 差升级费 ≤8 且落后进度 → 攒钱
from agent.decision import routes as _R157
_R157.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("1744067911380998", 3, why="测试")      # 10-21：安琪拉在棋手配置里关着、锁不上 → 改用开着的伞亚连（路线主C 亚连不在场，米莱狄仍是临时主C）
rules._field = {"狂铁": (617, 372), "米莱狄": (276, 501), "孙膑": (433, 569)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._maple_bound, rules._carry_piece, rules._carry_switch_round, rules._carry_switch_by = "狂铁", "米莱狄", 3, "rule"
_s157 = GameState(phase="prepare", round_no=5, gold=12, level=3, population=3, population_cap=6, population_valid=True)
_v157, _ = rules._transfer_victim(_s157)
_sv = rules.saving_for_level(GameState(phase="prepare", round_no=12, gold=24, level=5, level_cost=25))
_sv2 = rules.saving_for_level(GameState(phase="prepare", round_no=12, gold=10, level=5, level_cost=25))
_R157.CARRY_TIERS = None
print(f"[第11局] 临时主C 米莱狄换绑要卖→{_v157}（期望 None）；24/25 攒钱={_sv}（期望 True）；10/25 攒钱={_sv2}（期望 False）")
if _v157 or not _sv or _sv2:
    failures.append(f"第 11 局修复不对：{_v157} {_sv} {_sv2}")
rules.reset_match()


# 场景 158（2026-09-30 轮换第 12 局）：对账按人数删掉苏烈时，"枫叶舞绑在苏烈"不能跟着清（卡面亮/灰才是直接证据）
rules.reset_match()
rules._field = {"苏烈": (672, 430)}
rules._cell_owner = {(672, 430): "苏烈"}
rules._maple_bound = "苏烈"
rules._forget_gone("苏烈", (672, 430), "测试")
_m158 = rules._maple_bound
print(f"[第12局] 离场收尾后枫叶舞绑定={_m158!r}（期望 '苏烈'）")
if _m158 != "苏烈":
    failures.append(f"离场收尾清掉了枫叶舞绑定：{_m158!r}")
rules.reset_match()


# 场景 159（2026-09-30 第 15 局）：援军"随机获得4张同阵营的英雄牌"不给无阵营的公孙离；视觉没跑时不因格子失败撤销主C 记录
rules.reset_match()
_a159 = rules.card_target_ok("援军", "公孙离")[0]
_b159 = rules.card_target_ok("援军", "百里守约")[0]
rules._round_seen = 6
rules._field = {"公孙离": (433, 569), "张良": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece = "公孙离"
rules._cell_fails[(433, 569)] = 9
_s159 = GameState(phase="prepare", round_no=6, gold=10, level=3, population=2, population_cap=6, population_valid=True, shop_open=True)
_s159.hand_slots = [{"name": "公孙离", "x": 500, "y": 660}]
rules.decide_upgrade(_s159)
_c159 = rules._field.get("公孙离") == (433, 569)
print(f"[第15局] 援军→公孙离 {_a159}（期望 False）、→百里守约 {_b159}（期望 True）；商店开着视觉没跑时主C 记录保留={_c159}（期望 True）")
if _a159 or not _b159 or not _c159:
    failures.append(f"第 15 局修复不对：{_a159} {_b159} {_c159}")
rules.reset_match()


# 场景 160（2026-09-30 第 16 局）：第 3 回合 1 级、6 能量、升级费 5、场上 3 人 → 先升级（不再等铺满）
rules.reset_match()
rules._field = {"曜": (300, 569), "干将莫邪": (433, 569), "牛魔": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
_s160 = GameState(phase="prepare", round_no=3, gold=6, level=1, level_cost=5, level_btn=(60, 545), population=3,
                  population_cap=5, population_valid=True, shop_open=True,
                  shop=[ShopSlot(0, "苏烈", 3, ""), ShopSlot(1, "干将莫邪", 3, ""), ShopSlot(2, None, 3, ""), ShopSlot(3, None, 3, "")])
_a160 = rules.preempt_shop(_s160)
print(f"[第16局] 第 3 回合 1 级、场上 3 人 → {_a160 and _a160['type']}（期望 buy_xp）")
if not (_a160 and _a160["type"] == "buy_xp"):
    failures.append(f"第 3 回合落后没先升级：{_a160}")
rules.reset_match()


# 场景 161（2026-09-30 第 17 局）：新一局首选守约，不能读回上一局存档里的"已换成玄策路线"；同一局中途重启才恢复
rules.reset_match()
rules.forced_route = "50720714303640586"
rules.__dict__.pop("_forced_orig", None)
rules.load_state({"_forced": {"route": "122215359354172436", "orig": "122215359354172436", "released": False}})
_a161 = rules.forced_route
rules.load_state({"_forced": {"route": "32706315349545018", "orig": "50720714303640586", "released": False}})
_b161 = rules.forced_route
rules.forced_route = ""
rules.__dict__.pop("_forced_orig", None)
print(f"[第17局] 上一局存档（首选玄策）→ 本局路线 {_a161}（期望 守约）；同一局存档（首选守约、换成狄仁杰）→ {_b161}（期望 狄仁杰）")
if _a161 != "50720714303640586" or _b161 != "32706315349545018":
    failures.append(f"存档恢复指定路线不对：{_a161} {_b161}")
rules.reset_match()


# 场景 162（2026-09-30：研究哪些牌能顺带给 T 提供强度）：路线的顺带养 T 补件进"路线要的牌"和阶段成员
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("1744067911380998", 5, why="测试")            # 伞亚连：补件 云中君、助力核心
_w162, _m162 = rules.route_wanted_cards(), rules.route_stage_members(6)
print(f"[顺带养T] 伞亚连 补件={rules.route_addons()}；路线要的牌含 云中君/助力核心={'云中君' in _w162 and '助力核心' in _w162}；"
      f"阶段成员含 云中君={'云中君' in _m162}（期望 True/True）")
if not ("云中君" in _w162 and "助力核心" in _w162 and "云中君" in _m162):
    failures.append(f"顺带养 T 补件没接上：{rules.route_addons()} {_w162}")
rules.reset_match()


# 场景 163（2026-09-30：玄策当主C 还需要另一个闪现英雄才能滚雪球，最好还有其他功能性）
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("122215359354172436", 5, why="测试")
_a163 = [n for n in rules.route_addons() if n in ("孙悟空", "云中君", "阿轲", "裴擒虎")]
rules._field = {"百里玄策": (300, 569), "云中君": (529, 372)}
_b163 = [n for n in rules.route_addons() if n in ("孙悟空", "云中君", "阿轲", "裴擒虎")]
# 2026-10-13 改：玄策整备"随机 1 名闪现英雄 +1"，别的闪现英雄会分走等级，搭档只留孙悟空（整备：闪现英雄 +2，直接喂玄策）
print(f"[玄策搭档] 要 {_a163}（期望 只要孙悟空）；场上有云中君 → 要 {_b163}（期望 只要孙悟空）")
if _a163 != ["孙悟空"] or "云中君" in _b163 or "阿轲" in _b163 or "裴擒虎" in _b163:
    failures.append(f"玄策闪现搭档不对：{_a163} {_b163}")
rules.reset_match()


# 场景 164（2026-09-30 第 23 局：卖孙尚香退回的红魔、金色圣剑给了坦克刘禅、钟馗）：成品装备也要对口
rules.reset_match()
_c164 = (rules._equip_fits_role("红魔", "刘禅"), rules._equip_fits_role("金色圣剑", "钟馗"),
         rules._equip_fits_role("反伤重甲", "刘禅"), rules._equip_fits_role("红魔", "百里守约"))
print(f"[成品对口] 红魔→刘禅 {_c164[0]}、金色圣剑→钟馗 {_c164[1]}、反伤重甲→刘禅 {_c164[2]}、红魔→百里守约 {_c164[3]}（期望 F/F/T/F：红魔只给刺客，守约不要 —— 装备画像 heroes_avoid）")
if _c164 != (False, False, True, False):
    failures.append(f"成品装备对口不对：{_c164}")
rules.reset_match()


# 场景 165（2026-09-30：单极限到后期要加一个复活技能的辅助）：第 9 回合起要太乙真人 / 战前叫阵，河洛主C 再要度命之护
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("50720714303640586", 3, why="测试")
rules._carry_piece = "百里守约"
rules._round_seen = 6
_a165 = rules.route_wanted_cards()
rules._round_seen = 10
_b165, _m165 = rules.route_wanted_cards(), rules.route_stage_members(10)
print(f"[后期复活] 第 6 回合要太乙={'太乙真人' in _a165}（期望 False）；第 10 回合要 太乙/战前叫阵/度命之护="
      f"{'太乙真人' in _b165}/{'战前叫阵' in _b165}/{'度命之护' in _b165}，太乙算阶段成员={'太乙真人' in _m165}（期望全 True）")
if "太乙真人" in _a165 or not all(x in _b165 for x in ("太乙真人", "战前叫阵", "度命之护")) or "太乙真人" not in _m165:
    failures.append(f"后期复活辅助不对：{_a165} {_b165}")
rules.reset_match()


# 场景 166（2026-09-30 第 27 局：账本把李元芳记在弈星的格子上，弈星"丢了"，枫叶舞一直挂在它身上）→ 认人模型校正账本
rules.reset_match()
rules._field = {"李元芳": (300, 569), "沈梦溪": (433, 569)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._maple_bound = "弈星"
_s166 = GameState(phase="prepare", round_no=7, shop_open=False,
                  unit_names={"300,569": ["弈星", 0.97], "276,501": ["李元芳", 0.95], "433,569": ["沈梦溪", 0.96]},
                  vision_units={"300,569": "hero", "276,501": "hero", "433,569": "hero"})
rules.vision_identity_sync(_s166)
_f166 = dict(rules._field)
print(f"[认人校正] 账本={_f166}（期望 弈星@300,569 李元芳@276,501 沈梦溪@433,569）")
if _f166 != {"弈星": (300, 569), "李元芳": (276, 501), "沈梦溪": (433, 569)}:
    failures.append(f"认人校正不对：{_f166}")
rules.reset_match()


# 场景 167（2026-09-30 第 30 局：怒莲之刺 + 银莲庇佑 两件都在手里、整局没用）
rules.reset_match()
rules._field = {"狄仁杰": (433, 569), "钟馗": (529, 372), "李白": (617, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece = "狄仁杰"
_s167 = GameState(phase="prepare", round_no=11, gold=5, level=5, population=3, population_cap=7, population_valid=True,
                  unit_levels={"433,569": 39, "529,372": 6, "617,372": 20})
_s167.hand_slots = [{"name": "怒莲之刺", "x": 500, "y": 660}, {"name": "银莲庇佑", "x": 580, "y": 660}]
_a167 = rules._pair_equip(_s167, _s167.hand_slots[0])
_b167 = rules._pair_equip(_s167, _s167.hand_slots[1])
_s167.hand_slots = [{"name": "怒莲之刺", "x": 500, "y": 660}]
_c167 = rules._pair_equip(_s167, _s167.hand_slots[0])
print(f"[成对装备] 怒莲之刺→{_a167 and _a167['target']}（期望 狄仁杰）；银莲庇佑→{_b167 and _b167['target']}（期望 钟馗）；只有一件→{_c167}（期望 None）")
if not (_a167 and _a167["target"] == "狄仁杰") or not (_b167 and _b167["target"] == "钟馗") or _c167:
    failures.append(f"成对装备不对：{_a167} {_b167} {_c167}")
rules.reset_match()


# 场景 168（2026-09-30 第 31 局：没进类表的猪八戒被认成狄仁杰）：认人结果只收本局拿到过的英雄
rules.reset_match()
rules._field = {"猪八戒": (529, 372)}
rules._cell_owner = {(529, 372): "猪八戒"}
_s168 = GameState(phase="prepare", round_no=5, shop_open=False,
                  unit_names={"672,430": ["狄仁杰", 0.62], "529,372": ["猪八戒", 0.9]},
                  vision_units={"672,430": "hero", "529,372": "hero"})
rules.vision_identity_sync(_s168)
print(f"[认人只收拿到过的] 账本={dict(rules._field)}（期望 没有狄仁杰）")
if "狄仁杰" in rules._field:
    failures.append(f"认人把没拿过的英雄记进了账本：{rules._field}")
rules.reset_match()


# 场景 169（2026-09-30：红魔不适合辅助在旁边的射手或法师主C，适合刺客）：实战开关打开时，红魔只给刺客
from agent.decision import hero_traits as _ht169
_ht169.USE_HERO_ITEM_PREF = True
rules.reset_match()
_a169 = (rules._equip_allowed("红魔", "狄仁杰"), rules._equip_allowed("红魔", "阿轲"))
_rec169 = ["冥王", "名刀·司命", "红魔", "魔律之刃"]
_o169 = _ht169.carry_order(_rec169, "狄仁杰")
_ht169.USE_HERO_ITEM_PREF = False
print(f"[红魔只给刺客] 狄仁杰 {_a169[0]}、阿轲 {_a169[1]}（期望 False/True）；雷鸣刃四选一给狄仁杰的顺序 {_o169}（期望 红魔不在前两位）")
if _a169 != (False, True) or "红魔" in _o169[:2]:
    failures.append(f"红魔限制不对：{_a169} {_o169}")
rules.reset_match()


# 场景 170（2026-09-30：判断是多元的，要看主C 的套路阵容、其他配套是否养起来、装备能否穿上）
rules.reset_match()
rules._field = {"沈梦溪": (433, 569), "明世隐": (390, 502), "铠": (529, 372), "盾山": (617, 372), "李信": (672, 430)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._worn = {"沈梦溪": {"痛苦面具"}}
_s170 = GameState(phase="prepare", round_no=11, unit_levels={"433,569": 15, "672,430": 16})
_a170 = rules.carry_support_score("沈梦溪", _s170)
_b170 = rules.carry_support_score("李信", _s170)
print(f"[主C 综合分] 沈梦溪（配套明世隐/铠/盾山、穿痛苦面具，15 级）={_a170:.0f}；李信（16 级、没配套）={_b170:.0f}（期望 李信不高出 25）")
if _b170 >= _a170 + rules.CARRY_YIELD_MARGIN:
    failures.append(f"主C 综合分不对：{_a170} {_b170}")
rules.reset_match()


# 场景 171（2026-09-30 实战截图：太乙没贴到主C 身上）：按格子步数，太乙要站在"主C 是唯一最近的人"的格子
rules.reset_match()
from agent.perception.extractors import OWN_BOARD_ROWS as _R171
_c171 = [tuple(c) for r in _R171 for c in r]
_carry171 = (783, 438) if (783, 438) in _c171 else _c171[len(_c171) // 2]
_nb = [tuple(x) for x in rules._cell_neighbors(_carry171)]
_other = _nb[0]
_s171 = GameState(phase="prepare", round_no=9)
_best = rules._best_buddy_cell(_s171, _carry171, [_other])
_sc = rules._buddy_cell_score(_best, _carry171, [_other]) if _best else None
_bad = next((c for c in _nb[1:] if _other in [tuple(x) for x in rules._cell_neighbors(c)]), None)
print(f"[太乙贴主C] 主C {_carry171}、另一队友 {_other} → 选 {_best} 分 {_sc}（期望 (0,0)）；"
      f"同时挨着两人的格子 {_bad} 分 {_bad and rules._buddy_cell_score(_bad, _carry171, [_other])}")
if _sc != (0, 0):
    failures.append(f"太乙选格不对：{_best} {_sc}")
rules.reset_match()


# 场景 172（2026-09-30 统计：输的局刷新多、合成少）：刷新前先买场上已有英雄（主C / 路线要的人）的同名来合成
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("81682962224371742", 3, why="测试")
rules._field = {"沈梦溪": (433, 569), "李元芳": (300, 569), "苏烈": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece = "沈梦溪"
_s172 = GameState(phase="prepare", round_no=6, gold=10, level=3, level_cost=20, population=3, population_cap=6,
                  population_valid=True, shop_open=True, refresh_cost=2,
                  shop=[ShopSlot(0, "貂蝉", 3, ""), ShopSlot(1, "沈梦溪", 3, ""), ShopSlot(2, None, 3, ""), ShopSlot(3, None, 3, "")])
_a172 = rules.preempt_shop(_s172)
print(f"[先买合成材料] 商店有沈梦溪 → {_a172 and (_a172['type'], _a172.get('piece'))}（期望 买沈梦溪）")
if not (_a172 and _a172.get("piece") == "沈梦溪"):
    failures.append(f"没先买合成材料：{_a172}")
rules.reset_match()


# 场景 173（2026-09-30：仍有练多个主C 的问题；主C 确定不换之后，可以多上功能性英雄）
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("81682962224371742", 3, why="测试")
rules._field = {"沈梦溪": (433, 569), "李元芳": (300, 569), "钟馗": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece = "沈梦溪"
rules._round_seen = 8
_r173 = {n: rules.second_carry(n) for n in ("李信", "阿轲", "花木兰", "太乙真人", "钟馗", "明世隐")}
print(f"[不养第二主C] 主C 已定={rules.carry_settled()}；第二主C 判定={_r173}（期望 李信/阿轲/花木兰 True，其余 False）")
if not rules.carry_settled() or not (_r173["李信"] and _r173["阿轲"] and _r173["花木兰"]) or any(_r173[n] for n in ("太乙真人", "钟馗", "明世隐")):
    failures.append(f"第二主C 判定不对：{_r173}")
rules.reset_match()


# 场景 174（2026-09-30：百里玄策杀人时给其他闪现英雄提供等级，这时可练第二个刺客，但百里需是主C）
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.set_route("122215359354172436", 3, why="测试")
rules._field = {"百里玄策": (433, 569), "钟馗": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece = "百里玄策"
rules._round_seen = 8
_r174 = {n: rules.second_carry(n) for n in ("孙悟空", "阿轲", "李信")}
print(f"[主C 引擎受益者] 玄策主C：第二主C 判定={_r174}（期望 孙悟空/阿轲 False，李信 True）")
if _r174 != {"孙悟空": False, "阿轲": False, "李信": True}:
    failures.append(f"主C 引擎受益者判断不对：{_r174}")
rules.reset_match()

if failures:
    print("FAIL:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("冒烟测试全部通过")

# 场景 175（2026-09-30 第 43 局：有古币没用、被手动给了古币机）：古币机在场，古币打给古币机
rules.reset_match()
rules._field = {"沈梦溪": (390, 502), "李元芳": (346, 430)}
rules._carry_piece, rules._carry = "沈梦溪", (390, 502)
rules.coin_machine_cell = (555, 569)
rules._round_seen = 11
rules.note_used_on(11, "推演战术", "沈梦溪")      # 2026-10-12：沈梦溪本回合已吃过效果牌 → 古币给古币机（没吃过的话第一张给她，见场景 248）
_s175 = GameState(phase="prepare", round_no=11, gold=1, population=2, population_cap=7, population_valid=True, shop_open=False)
_s175.hand_slots = [{"name": "古币", "x": 721, "y": 663}]
_d175 = rules.decide_use_cards(_s175)
print(f"[古币给古币机] {_d175 and (_d175.get('target'), _d175.get('dst'))}（期望 古币机 @ 555,569）")
if not (_d175 and _d175.get("target") == "古币机"):
    failures.append(f"古币机在场，古币没打给它：{_d175}")
_ok175 = rules.card_target_ok("古币", "古币机")[0] and not rules.card_target_ok("精密协作", "古币机")[0]
if not _ok175:
    failures.append("card_target_ok：古币机应只收古币")
rules.coin_machine_cell = None
rules.reset_match()
# 场景 176（2026-09-30：重剑随花木兰一起出现，会占位；古币机同理，不能在它上面放英雄）
from agent.decision import rules as _R176
rules.reset_match()
rules.note_object((555, 569), "古币机")
rules.note_object((718, 371), "重剑")
_cells176 = rules.object_cells_all()
print(f"[占格物件] {sorted(_cells176)}（期望 含 (555,569)、(718,371)），古币机格 {rules.coin_machine_cell}")
if not ({(555, 569), (718, 371)} <= _cells176 and rules.coin_machine_cell == (555, 569)):
    failures.append(f"占格物件没记上：{_cells176}")
_blocked = []
for _c in list(_cells176):
    rules._field = {}
    _st = GameState(phase="prepare", round_no=5, gold=10, population=0, population_cap=6, population_valid=True, shop_open=False)
    _st.hand_slots = [{"name": "孙膑", "x": 650, "y": 663}]
    for _i in range(30):
        _d = rules.decide_deploy(_st) if hasattr(rules, "decide_deploy") else None
        if not _d:
            break
        if tuple(_d.get("dst") or ()) in _cells176:
            _blocked.append(_d.get("dst"))
        break
if _blocked:
    failures.append(f"往占格物件上放英雄：{_blocked}")
rules.reset_match()
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
print("冒烟测试全部通过（含场景 175、176）")

# 场景 177（2026-09-30：刚卖掉一个英雄，打开商店又买回来）：刚卖的英雄两回合内不买回来
rules.reset_match()
rules._round_seen = 8
rules.note_executed({"type": "sell_unit", "piece": "姬小满", "src": (529, 372)}) if False else None
rules._sold_round["姬小满"] = 8
if not rules.recently_sold("姬小满"):
    failures.append("刚卖的姬小满没被认成'刚卖过'")
rules._round_seen = 10
if rules.recently_sold("姬小满"):
    failures.append("两回合后仍不许买回姬小满（应该放开）")
from agent.perception.state import ShopSlot as _SS177
rules._round_seen = 8
_ok177 = rules._worth_buying(_SS177(index=0, piece="姬小满", cost=3), GameState(phase="prepare", round_no=8, gold=20, shop_open=True))
if _ok177:
    failures.append("刚卖的姬小满商店里又判成值得买")
# 场景 178（2026-09-30：在 3 级多呆几个回合，找到主C 之后再开始升级）
rules.reset_match()
rules._field = {"孙膑": (529, 372), "牛魔": (617, 372)}
rules._carry_piece = ""
_s178 = GameState(phase="prepare", round_no=5, gold=20, level=3, level_cost=8, population=2, population_cap=6,
                  population_valid=True, shop_open=True)
_s178.level_btn = (95, 550)
_h178 = rules.hold_level_for_carry(_s178)
print(f"[3 级没主C 先不升] hold={_h178}（期望 True）")
if not _h178:
    failures.append("3 级、第 5 回合、没有主C，却没停级找主C")
_s178b = GameState(phase="prepare", round_no=10, gold=20, level=3, level_cost=8, shop_open=True)
if rules.hold_level_for_carry(_s178b):
    failures.append("第 10 回合还在停级（应只停到第 8 回合）")
rules._field["百里守约"] = (390, 502); rules._carry_piece = "百里守约"
if rules.hold_level_for_carry(_s178):
    failures.append("主C 已到手还在停级")
rules.reset_match()
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 179（第 58 局：枫叶舞绑着 1 阶干将莫邪，路线主C 亚连在场 —— 干将估值再高也要卖掉换绑）
rules.reset_match()
rules._field = {"干将莫邪": (300, 569), "亚连": (783, 438), "牛魔": (617, 372)}
rules._carry_piece, rules._carry = "亚连", (783, 438)
rules._carry_switch_round = 3
rules._maple_bound = "干将莫邪"
rules._unit_lv = {"干将莫邪": 60, "亚连": 5}
rules._route_locked = 3
rules._active_comp = {"carry": "亚连", "name": "伞亚连"}
# 2026-10-06 第 279/283 局后：亚连还不到干将一半 → 第 6 回合先不卖（等追），第 9 回合（CATCHUP_UNTIL_ROUND）照常卖
_s179a = GameState(phase="prepare", round_no=6, gold=10)
_v179a, _ = rules._transfer_victim(_s179a)
rules._transfer_round = -1
_s179 = GameState(phase="prepare", round_no=9, gold=10)
_v179, _w179 = rules._transfer_victim(_s179)
print(f"[枫叶舞换绑给路线主C] 第6回合卖={_v179a}（期望 None）第9回合卖={_v179}（期望 干将莫邪） 原因={_w179}")
if _v179a is not None or _v179 != "干将莫邪":
    failures.append(f"路线主C 亚连在场、枫叶舞绑着 1 阶干将：第6回合 {_v179a} / 第9回合 {_v179}")
rules.reset_match()
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 180（2026-10-01：马超不适合单C阵容；不写死，要理解阿离为什么不适合）：
# 成长靠队友触发（马超：己方施加控制）的输出位，场上提供该触发的单位不够时不当主C；够了才放行。装备关系表同时接入。
from agent.decision import hero_traits as _ht180
_ht180.USE_HERO_ITEM_PREF = True
rules.reset_match()
rules._field = {"马超": (529, 372), "苏烈": (617, 372)}
_a180 = rules._carry_eligible("马超")
_H180 = _ht180.relations().get("heroes") or {}
_ctl180 = [n for n, h in _H180.items() if n != "马超" and any(p.get("tag") == "control" for p in h.get("provides") or [])][:3]
for _i, _n in enumerate(_ctl180):
    rules._field[_n] = (300 + _i * 90, 569)
_b180 = rules.team_feeds_carry("马超")
_c180 = rules.team_feeds_carry("百里守约")
_d180 = _ht180.profile_says("自然法杖", "马超")
print(f"[单C 判断] 马超 无控制={_a180}（期望 False） 加控制 {_ctl180}={_b180}（期望 True） 守约={_c180}（期望 True） 马超×自然法杖={_d180}（期望 avoid）")
if _a180 or not _b180 or not _c180 or _d180 != "avoid":
    failures.append("单C 判断 / 装备关系表接入不符合预期")
_ht180.USE_HERO_ITEM_PREF = False
rules.reset_match()
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 181（2026-10-01 第 78 局：镜输出太弱，不能当主C）：
# 首选伞亚连、第 4 回合亚连没到、手里只有镜 → 不换到镜的路线（主C 资格不过）；手里有守约 → 换守约路线
from agent.decision import routes as _R181
_R181.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "1744067911380998", ("5121767631866960", "50720714303640586")   # 伞亚连 / 镜 / 守约
_s181 = GameState(phase="prepare", round_no=4, level=3, population=3, population_cap=6, population_valid=True, banned_faction="大河流域")
_s181.hand_slots = [{"name": "镜", "x": 500, "y": 660}]
rules.resolve_comp(_s181)
_a181 = rules.route_carry()
rules.reset_match()
rules.forced_route, rules.forced_backups = "1744067911380998", ("5121767631866960", "50720714303640586")
_s181.hand_slots = [{"name": "镜", "x": 500, "y": 660}, {"name": "百里守约", "x": 640, "y": 660}]
rules.resolve_comp(_s181)
_b181 = rules.route_carry()
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R181.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
# 2026-10-10 改：守约（河洛体系）和亚连（日落海）不同体系 → 不换（tools/sim_route_vs_flex.py：跨体系 = 前面凑的配套作废）
# 2026-10-13 再改（灵活切换主C）：亚连没来、守约在手 → 切守约（跨体系可以）；镜（弱主C）仍不换
# 2026-10-14 改为算账（route_eta）：守约路线要明显更快才换
_e_sy = rules.route_eta("50720714303640586", _s181, {"镜", "百里守约"}); _e_yl = rules.route_eta("1744067911380998", _s181, {"镜", "百里守约"})
_want181 = "百里守约" if _e_sy + 0.5 < _e_yl else "亚连"
print(f"[备选路线不换弱主C] 只有镜在手→主C {_a181}（期望 亚连） 有守约→{_b181}（算账 守约 {_e_sy} vs 亚连 {_e_yl} → 期望 {_want181}）")
if _a181 != "亚连" or _b181 != _want181:
    failures.append(f"备选路线换到了弱主C：{_a181} / {_b181}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 182（2026-10-01 第 80 局：主C 没穿上装备；主C 的等级和装备都重要，2~3 阶的装备都很强力）：
# 云中君在场、主C 亚连 0 件，手里雷鸣刃（对亚连适配 7.5）→ 先给主C，不铺给坦克钟馗；换格后旧格的件数不算到新来的人头上
from agent.decision import hero_traits as _ht182
_ht182.USE_HERO_ITEM_PREF = True
rules.reset_match()
rules._field = {"亚连": (672, 430), "云中君": (556, 430), "钟馗": (718, 371), "米莱狄": (1027, 572)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "亚连", (672, 430)
rules._equip_seen["617,372"] = 2
rules.relocate_unit("亚连", (617, 372))
_s182 = GameState(phase="prepare", round_no=12, gold=10)
_t182 = rules._equip_target(_s182, "雷鸣刃")
_n182 = _t182[0] if _t182 else None
_e182 = rules._equips_at(_s182, (617, 372))
print(f"[主C 先满装] 雷鸣刃 → {_n182}（期望 亚连）；亚连换到旧格后件数 {_e182}（期望 0）")
if _n182 != "亚连" or _e182 != 0:
    failures.append(f"主C 先满装不对：{_n182} / 件数 {_e182}")
_ht182.USE_HERO_ITEM_PREF = False
rules.reset_match()
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 183（2026-10-01 第 84 局）：路线锁定守约，人口 5/5、2 级、能量 4，商店有守约 → preempt_shop 必须给出"买守约"
# （laya_policy 在规则空决策时也会调用它；原来只挂在"有商店动作"的分支上，守约摆着被关店）
from agent.decision import routes as _R183
from agent.perception.state import ShopSlot as _SS183
_R183.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "50720714303640586", ()
_s183 = GameState(phase="prepare", round_no=4, level=2, level_cost=10, gold=4, population=5, population_cap=5,
                  population_valid=True, shop_open=True, banned_faction="日落海")
rules._field = {"牛魔": (617, 372), "苏烈": (672, 430), "孙尚香": (300, 569), "孙膑": (276, 501), "西施": (1027, 572)}
rules.resolve_comp(_s183)
_s183.shop = [_SS183(index=0, piece="蔡文姬", cost=3), _SS183(index=1, piece="干将莫邪", cost=3),
              _SS183(index=2, piece="蔡文姬", cost=3), _SS183(index=3, piece="百里守约", cost=3)]
_s183.hand_slots = [{"name": "枫叶舞", "x": 500, "y": 660}, {"name": "加速成长", "x": 600, "y": 660}]
_p183 = rules.preempt_shop(_s183)
print(f"[路线主C 在商店] 路线锁定={rules.route_locked()} 决策={_p183 and (_p183.get('type'), _p183.get('piece'), _p183.get('why'))}（期望 买 百里守约）")
if not (_p183 and _p183.get("piece") == "百里守约"):
    failures.append(f"路线主C 守约在商店却没买：{_p183}")
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R183.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 184（2026-10-02 前三 vs 5~6 名分析）：首选守约，第 3 回合只有安琪拉 → 换安琪拉；第 4 回合守约在商店 → 先买；
# 第 5 回合守约到手、安琪拉 5 级 → 换回守约；若安琪拉已 12 级 → 不换回
from agent.decision import routes as _R184
from agent.perception.state import ShopSlot as _SS184
_R184.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
def _run184(lv):
    rules.reset_match()
    rules.forced_route, rules.forced_backups = "50720714303640586", ("user_angela", "1744067911380998")
    rules.__dict__.pop("_forced_orig", None)
    s = GameState(phase="prepare", round_no=1, level=1, population=1, population_cap=5, population_valid=True, banned_faction="逐鹿")
    rules.resolve_comp(s)
    s = GameState(phase="prepare", round_no=3, level=2, population=2, population_cap=5, population_valid=True, banned_faction="逐鹿")
    s.hand_slots = [{"name": "安琪拉", "x": 500, "y": 660}]
    rules.resolve_comp(s)
    a = rules.route_carry()
    rules._field["安琪拉"] = (529, 372); rules._carry_piece = "安琪拉"; rules._unit_lv["安琪拉"] = lv
    s4 = GameState(phase="prepare", round_no=4, level=3, gold=6, population=3, population_cap=6, population_valid=True,
                   shop_open=True, banned_faction="逐鹿")
    s4.shop = [_SS184(index=0, piece="牛魔", cost=3), _SS184(index=1, piece="百里守约", cost=3)]
    pre = rules.preempt_shop(s4)
    s5 = GameState(phase="prepare", round_no=5, level=3, population=3, population_cap=6, population_valid=True, banned_faction="逐鹿")
    s5.hand_slots = [{"name": "百里守约", "x": 500, "y": 660}]
    rules.resolve_comp(s5)
    return a, (pre or {}).get("piece"), rules.route_carry()
_save184 = (RuleEngine.RETURN_ORIG_ROUND, RuleEngine.PREFERRED_ROUTE)
RuleEngine.RETURN_ORIG_ROUND, RuleEngine.PREFERRED_ROUTE = 6, ""   # 机制测试：开关临时打开（实战 2026-10-02 起关闭）
_a184 = _run184(5)
_b184 = _run184(12)
RuleEngine.RETURN_ORIG_ROUND, RuleEngine.PREFERRED_ROUTE = _save184
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R184.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
# 2026-10-10 起不跨体系换路线（安琪拉不在河洛体系）→ 第 3 回合手里有安琪拉也不换，"换回首选"无从发生，三项都是守约
# 2026-10-13 再改（守约不来就别一直等，先拿到哪个配套主C 就定哪个）：可以跨体系 →
# 第 3 回合安琪拉在手就换；第 5 回合守约到手：安琪拉 5 级（没养起来）换回守约，12 级就不换回（机制测试，实战 RETURN_ORIG_ROUND=0）
print(f"[换回首选守约] 安琪拉 5 级：{_a184}；安琪拉 12 级：{_b184}（期望 先换安琪拉；5 级换回守约，12 级不换回）")
# 机制测试（测试自己指定的备选里有安琪拉路线）；线上 one_match.sh 的备选清单 10-20 已去掉 user_angela_sunset
# 2026-10-21：安琪拉在棋手配置里关着、路线锁不上（第 854 局后 set_route 也查自定义阵容）→ 不会先换到安琪拉，两种都是守约
if _a184 != ("", "百里守约", "百里守约") or _b184 != ("", "百里守约", "百里守约"):
    failures.append(f"换回首选不对：{_a184} / {_b184}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 185（2026-10-02 第二轮 12 局后）：首选沈梦溪（轮换），守约当截胡 —— 第 4 回合守约到手、沈梦溪 3 级 → 切守约；沈梦溪 15 级 → 不切
from agent.decision import routes as _R185
_R185.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
def _run185(lv):
    rules.reset_match()
    rules.forced_route, rules.forced_backups = "81682962224371742", ("50720714303640586", "user_angela", "1744067911380998")
    rules.__dict__.pop("_forced_orig", None)
    s = GameState(phase="prepare", round_no=1, level=1, population=1, population_cap=5, population_valid=True, banned_faction="逐鹿")
    rules.resolve_comp(s)
    rules._field["沈梦溪"] = (529, 372); rules._carry_piece = "沈梦溪"; rules._unit_lv["沈梦溪"] = lv
    s4 = GameState(phase="prepare", round_no=4, level=3, population=3, population_cap=6, population_valid=True, banned_faction="逐鹿")
    s4.hand_slots = [{"name": "百里守约", "x": 500, "y": 660}]
    rules.resolve_comp(s4)
    return rules.route_carry()
_save185 = (RuleEngine.RETURN_ORIG_ROUND, RuleEngine.PREFERRED_ROUTE)
RuleEngine.RETURN_ORIG_ROUND, RuleEngine.PREFERRED_ROUTE = 6, "50720714303640586"
_a185, _b185 = _run185(3), _run185(15)
RuleEngine.RETURN_ORIG_ROUND, RuleEngine.PREFERRED_ROUTE = _save185
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R185.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
print(f"[守约截胡] 沈梦溪 3 级→{_a185}（期望 百里守约）；15 级→{_b185}（期望 沈梦溪）")
if _a185 != "百里守约" or _b185 != "沈梦溪":
    failures.append(f"守约截胡不对：{_a185} / {_b185}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 186（2026-10-02 第 109 局：主C 太多、枫叶舞没切过去）：路线主C 亚连让位给鲁班七号后，
# 只认鲁班七号一个主C，换绑收尾卖掉绑着枫叶舞的亚连
from agent.decision import routes as _R186
_R186.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "1744067911380998", ()
_s186 = GameState(phase="prepare", round_no=9, level=5, gold=10, population=7, population_cap=7, population_valid=True, banned_faction="河洛")
rules._field = {"亚连": (783, 438), "鲁班七号": (617, 372), "太乙真人": (346, 430), "安琪拉": (529, 372)}
rules.resolve_comp(_s186)
rules._carry_piece, rules._carry = "鲁班七号", (617, 372)
rules._forced_released = True
rules._carry_switch_round = 8
rules._maple_bound = "亚连"
_c186 = (rules._is_carry("亚连"), rules._is_carry("鲁班七号"))
_v186, _w186 = rules._transfer_victim(_s186)
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R186.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
print(f"[让位后只认一个主C] 是主C(亚连, 鲁班)={_c186}（期望 False, True）；收尾卖={_v186}（期望 亚连）{_w186}")
if _c186 != (False, True) or _v186 != "亚连":
    failures.append(f"让位后收尾不对：{_c186} / {_v186}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 187（2026-10-02：主C 第 3 回合到手前三 82%，第 5 回合 38%）：首选沈梦溪没到，第 2 回合商店有守约 → 先买；守约在手 → 第 2 回合就切
from agent.decision import routes as _R187
from agent.perception.state import ShopSlot as _SS187
_R187.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "81682962224371742", ("1744067911380998", "50720714303640586")
rules.__dict__.pop("_forced_orig", None)
_s187 = GameState(phase="prepare", round_no=1, level=1, gold=6, population=1, population_cap=5, population_valid=True, banned_faction="逐鹿")
rules.resolve_comp(_s187)
_s187b = GameState(phase="prepare", round_no=2, level=1, gold=6, population=2, population_cap=5, population_valid=True,
                   shop_open=True, banned_faction="逐鹿")
_s187b.shop = [_SS187(index=0, piece="牛魔", cost=3), _SS187(index=1, piece="百里守约", cost=3)]
_p187 = rules.preempt_shop(_s187b)
_s187c = GameState(phase="prepare", round_no=2, level=1, population=2, population_cap=5, population_valid=True, banned_faction="逐鹿")
_s187c.hand_slots = [{"name": "百里守约", "x": 500, "y": 660}]
rules.resolve_comp(_s187c)
_c187 = rules.route_carry()
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R187.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
print(f"[主C 早到手] 第 2 回合商店有守约 → {(_p187 or {}).get('piece')}（期望 百里守约）；守约在手 → 路线主C {_c187}（期望 百里守约）")
if (_p187 or {}).get("piece") != "百里守约" or _c187 != "百里守约":
    failures.append(f"主C 早到手不对：{_p187} / {_c187}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 188（2026-10-02：不定死一个主C，棋手 3 级左右拿到什么用什么，但不能用 1 阶主C 玩到最后）
from agent.decision import routes as _R188
_R188.CARRY_TIERS = (2, 3)
rules.reset_match()
rules._round_seen = 3
_a188 = (rules._carry_eligible("孙尚香"), rules._carry_eligible("弈星"))
rules._round_seen = 7
_b188 = rules._carry_eligible("孙尚香")
rules.reset_match()
_R188.CARRY_TIERS = None
print(f"[前期临时主C] 第 3 回合 孙尚香/弈星 = {_a188}（期望 True, False）；第 7 回合 孙尚香 = {_b188}（期望 False）")
if _a188 != (True, False) or _b188:
    failures.append(f"前期临时主C 不对：{_a188} / {_b188}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 189（2026-10-03：阿轲在阵容里基本没有作用；登场效果只在上场时生效一次，除非用强制触发登场效果的效果牌
#   "不能写死，读效果卡和英雄卡的描述"）：只按卡面判断 —— 效果条目全是「登场：」的英雄，路线锁定、主C 已定、不在路线里 ->
#   人口满且路线成员在等时卖它；手里有「触发其登场效果」的牌时留着；效果里有非登场条目的不算。
from agent.decision import routes as _R189
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
_lu189 = _R189.by_key("1744067911380998")
rules._active_comp, rules._route_locked, rules._route_key, rules._round_seen = _lu189, True, "1744067911380998", 7
rules._field = {"亚连": (556, 430), "阿轲": (529, 372), "牛魔": (617, 372), "苏烈": (672, 430), "猪八戒": (718, 371), "米莱狄": (300, 569)}
rules._carry_piece = "亚连"
_t189 = (RuleEngine.entry_only_text(rules.pieces["阿轲"].get("skill", "")),
         RuleEngine.entry_only_text(rules.pieces["明世隐"].get("skill", "")))
_s189 = GameState(phase="prepare", round_no=7, level=4, gold=5, population=6, population_cap=6, population_valid=True)
_s189.hand_slots = [{"name": "朵莉亚", "x": 600, "y": 660}]
_a189 = rules.decide_sell_unit(_s189)
_b189 = rules.spent_entry_unit("阿轲", ["朵莉亚", "唤醒"])
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[登场已用完] 卡面判定 阿轲/明世隐 = {_t189}（期望 True, False）；人口满、朵莉亚在等 -> {(_a189 or {}).get('piece')}（期望 阿轲）；手里有唤醒 -> 已用完={_b189}（期望 False）")
if _t189 != (True, False) or (_a189 or {}).get("piece") != "阿轲" or _b189:
    failures.append(f"登场已用完的身板不对：{_t189} / {_a189} / {_b189}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 190（2026-10-03：高攻速英雄和少司缘可以形成滚等级的配合）：只读数据 —— 召唤物卡面「每合计进行 N 次普攻…等级」
#   的召唤者（少司缘），在主C 高攻速（强度模型 40 级攻速 ≥1.4 或卡面给自身加攻速）时列为路线补件；主C 不是高攻速时不列。
rules.reset_match()
rules._route_key = "1744067911380998"
rules._carry_piece = "狄仁杰"
_a190 = ("少司缘" in rules.route_addons(), rules.high_attack_speed("狄仁杰"))
rules._carry_piece = "亚连"
_b190 = "少司缘" in rules.route_addons()
rules.reset_match()
print(f"[攻速滚等级] 主C 狄仁杰 → 少司缘是补件/高攻速 = {_a190}（期望 True, True）；主C 亚连 → {_b190}（期望 False）")
if _a190 != (True, True) or _b190:
    failures.append(f"攻速滚等级补件不对：{_a190} / {_b190}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 191（2026-10-03 第 129 局：枫叶舞没切到守约）：枫叶舞绑在觉醒过、但不在当前路线里的干将莫邪，
#   路线主C 守约已在场当主C → 换主C 收尾要卖干将换绑（原来"1 阶且没觉醒"才换）。路线成员身上的照旧不动。
from agent.decision import routes as _R191
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules._active_comp, rules._route_locked, rules._route_key = _R191.by_key("50720714303640586"), True, "50720714303640586"
rules._round_seen = 6
rules._field = {"百里守约": (917, 570), "干将莫邪": (300, 569), "李元芳": (276, 501), "牛魔": (617, 372)}
rules._carry_piece, rules._maple_bound = "百里守约", "干将莫邪"
rules._carry_switch_round, rules._carry_switch_by = 4, "rule"
rules._ever_awakened = {"干将莫邪"}
_s191 = GameState(phase="prepare", round_no=6, level=3, gold=5, population=4, population_cap=6, population_valid=True)
_s191.hand_slots = [{"name": "枫叶舞", "x": 500, "y": 660, "ready": False}]
_a191 = rules._transfer_victim(_s191)[0]
rules._maple_bound = "李元芳"
rules._ever_awakened = {"李元芳"}
_b191 = rules._transfer_victim(_s191)[0]
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[枫叶舞换绑路线主C] 绑在觉醒干将（路线外）→ 卖 {_a191}（期望 干将莫邪）；绑在觉醒李元芳（路线成员）→ {_b191}（期望 None）")
if _a191 != "干将莫邪" or _b191 is not None:
    failures.append(f"枫叶舞换绑路线主C 不对：{_a191} / {_b191}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 192（2026-10-03 第 151 局：第 2 回合场上 2/5，"先买合成材料"把 6 能量全花在两张干将上）：
#   第 4 回合以内、场上 ≤3 人还有空位，买完同名剩不到 3 能量就不买合成材料（路线主C 同名除外）；场面铺开后照买。
from agent.perception.state import ShopSlot as _SS192
rules.reset_match()
rules._field = {"百里守约": (300, 569), "牛魔": (617, 372)}
rules._carry_piece = "百里守约"   # 2026-10-13：临时主C（1 阶）不再买同名，这里改用 2 阶主C 测"先铺人头"
_s192 = GameState(phase="prepare", round_no=2, level=1, gold=4, population=2, population_cap=5, population_valid=True, shop_open=True)
_s192.shop = [_SS192(index=0, piece="百里守约", cost=3)]
_a192 = rules._merge_buy(_s192, 4)
_s192b = GameState(phase="prepare", round_no=2, level=1, gold=7, population=2, population_cap=5, population_valid=True, shop_open=True)
_s192b.shop = [_SS192(index=0, piece="百里守约", cost=3)]
_b192 = rules._merge_buy(_s192b, 7)
_s192c = GameState(phase="prepare", round_no=5, level=3, gold=4, population=5, population_cap=6, population_valid=True, shop_open=True)
_s192c.shop = [_SS192(index=0, piece="百里守约", cost=3)]
_c192 = rules._merge_buy(_s192c, 4)
# 临时主C（1 阶干将、不是路线主C）：场面铺开、钱也够，也不买同名凑觉醒（2026-10-13）
rules._field = {"干将莫邪": (300, 569), "牛魔": (617, 372)}
rules._carry_piece = "干将莫邪"
_d192 = rules._merge_buy(GameState(phase="prepare", round_no=5, level=3, gold=9, population=5, population_cap=6,
                                    population_valid=True, shop_open=True, shop=[_SS192(index=0, piece="干将莫邪", cost=3)]), 9)
from agent.decision import interim_rules as _IR192
if _d192 and _IR192.on("temp_carry_no_copies"):      # 临时规则关掉后（R27 已学会），交给模型
    failures.append(f"临时主C 不该买同名：{_d192}")
rules.reset_match()
print(f"[前期先铺人头] 2/5、4 能量 → {(_a192 or {}).get('piece')}（期望 None）；7 能量 → {(_b192 or {}).get('piece')}（期望 百里守约）；第 5 回合 5/6 → {(_c192 or {}).get('piece')}（期望 百里守约）")
if _a192 or (_b192 or {}).get("piece") != "百里守约" or (_c192 or {}).get("piece") != "百里守约":
    failures.append(f"前期先铺人头不对：{_a192} / {_b192} / {_c192}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 193（2026-10-04：韩信阵容加入推荐阵容）：4 阶主C 的路线写 carry_from_round=8 ——
#   第 4 回合没韩信、备选主C 亚连在手也不换备选；3 级不卡等级找主C（韩信 4 级才刷得出）；第 8 回合起照常可换。
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "user_hanxin_open", ("1744067911380998",)
rules.__dict__.pop("_forced_orig", None)
_s193 = GameState(phase="prepare", round_no=4, level=3, gold=6, population=4, population_cap=6, population_valid=True, banned_faction="逐鹿")
_s193.hand_slots = [{"name": "亚连", "x": 500, "y": 660}]
rules.resolve_comp(_s193)
rules._forced_fallback(_s193)
_a193 = (rules.forced_route, rules.hold_level_for_carry(_s193))
_s193.round_no = 8
rules._forced_fallback(_s193)
_b193 = rules.forced_route
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
RuleEngine.ROUTE_ENABLED = False
# 2026-10-10 起只在同体系里换：亚连（日落海）不是大河开团韩信的同体系 → 第 8 回合也不换
# 2026-10-13 再改（灵活切换主C）：可跨体系 → 第 8 回合韩信还没来、亚连在手 → 换亚连（第 4 回合仍受 carry_from_round 保护）
print(f"[韩信路线] 第 4 回合 路线/卡等级 = {_a193}（期望 user_hanxin_open, False）；第 8 回合 → {_b193}（期望 亚连路线 1744067911380998）")
if _a193 != ("user_hanxin_open", False) or _b193 != "1744067911380998":
    failures.append(f"韩信路线不对：{_a193} / {_b193}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 194（2026-10-04 第 5~6 名复盘）：
#   a) 弈星升过级、人口吃紧，但手里没有能补位的英雄 → 不卖（第 183 局卖了少一人迎战，一回合掉 30 血）；手里有英雄 → 卖
#   b) 不重要的夺取英雄（猪八戒）挨着主C 时不合成（第 179 局一合成亚连 17→5）
rules.reset_match()
_c194 = (556, 430)
_nb194 = rules._cell_neighbors(_c194)[0]
rules._field = {"亚连": _c194, "弈星": (276, 501), "猪八戒": tuple(_nb194), "牛魔": (617, 372)}
rules._carry_piece = "亚连"
rules._engine_levels["弈星"] = 1
_s194 = GameState(phase="prepare", round_no=7, level=4, gold=3, population=4, population_cap=4, population_valid=True)
_s194.hand_slots = [{"name": "古币", "x": 500, "y": 660}]
_a194 = rules.decide_sell_unit(_s194)
_s194.hand_slots = [{"name": "古币", "x": 500, "y": 660}, {"name": "米莱狄", "x": 600, "y": 660}]
_b194 = rules.decide_sell_unit(_s194)
_s194b = GameState(phase="prepare", round_no=7, level=4, gold=3, population=4, population_cap=6, population_valid=True)
_s194b.hand_slots = [{"name": "猪八戒", "x": 600, "y": 660}]
_s194b.hand = ["猪八戒"]
_u194 = rules.decide_upgrade(_s194b)
rules.reset_match()
_sold_a = (_a194 or {}).get("piece"); _sold_b = (_b194 or {}).get("piece"); _merge = (_u194 or {}).get("piece")
print(f"[第5~6名复盘] 没人补位卖弈星={_sold_a}（期望 None）；有米莱狄补位 → {_sold_b}（期望 弈星）；猪八戒挨着主C 合成 → {_merge}（期望 None）")
if _sold_a == "弈星" or _sold_b != "弈星" or _merge == "猪八戒":
    failures.append(f"第5~6名复盘修复不对：{_a194} / {_b194} / {_u194}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 195（2026-10-04 第 189 局：狄仁杰当主C 养到 18 级，第 7 回合因首选沈梦溪没到手换到守约路线，卖掉狄仁杰，主C 18→1）：
#   现任主C 合格且 ≥KEEP_CARRY_LV 级 → 不换备选路线；只有 5 级时照换。
from agent.decision import routes as _R195
_R195.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
def _run195(lv):
    rules.reset_match()
    rules.forced_route, rules.forced_backups = "81682962224371742", ("50720714303640586",)
    rules.__dict__.pop("_forced_orig", None)
    s = GameState(phase="prepare", round_no=7, level=3, gold=6, population=5, population_cap=6, population_valid=True, banned_faction="日落海")
    rules.resolve_comp(s)
    rules._field = {"狄仁杰": (839, 502), "盾山": (617, 372), "明世隐": (956, 502)}
    rules._carry_piece = "狄仁杰"
    rules._unit_lv["狄仁杰"] = lv
    rules._round_seen = 7
    s.hand_slots = [{"name": "百里守约", "x": 500, "y": 660}]
    rules._forced_fallback(s)
    return rules.forced_route
_a195, _b195 = _run195(18), _run195(5)
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R195.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
# 2026-10-10：同体系只在指定路线清单里找（狄仁杰实战成绩差，不在清单）→ 5 级时照旧换到守约路线（同体系）
print(f"[养起来的主C 不换路线] 狄仁杰 18 级 → 路线 {_a195}（期望 81682962224371742 不换）；5 级 → {_b195}（期望 50720714303640586）")
if _a195 != "81682962224371742" or _b195 != "50720714303640586":
    failures.append(f"养起来的主C 不换路线不对：{_a195} / {_b195}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 196（2026-10-04 第 214 局：又用了输出很弱的马超当主C）：还没主C 时枫叶舞"凑合绑"
#   跳过审定评分 < HERO_CARRY_MIN 的（马超 3）；有合格的（虞姬 5）就绑它，只有马超时先不绑。
rules.reset_match()
rules._field = {"马超": (556, 430), "虞姬": (839, 502), "牛魔": (617, 372)}
_a196 = rules._skill_target()
rules._field = {"马超": (556, 430), "牛魔": (617, 372)}
_b196 = rules._skill_target()
rules.reset_match()
# （10-20 晚曾改成"第 1 回合必须挂"，10-21 回到原节奏、maple_temp_host 关闭 → 恢复原期望）
print(f"[枫叶舞不绑弱英雄] 马超+虞姬 → {_a196}（期望 虞姬）；只有马超 → {_b196}（期望 不是马超）")
if _a196 != "虞姬" or _b196 == "马超":
    failures.append(f"枫叶舞凑合绑选人不对：{_a196} / {_b196}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 197（2026-10-04 第 214 局：枫叶舞宿主马超被"同格冲突"挤出账本、认人模型认不出，枫叶舞挂到局末）：
#   宿主被挤掉时记下格子；枫叶舞卡面灰且宿主不在账本 → maple_search_cells 返回那一格 + 认不出的格子；点开确认后宿主回到账本。
rules.reset_match()
rules._field = {"马超": (556, 430), "亚连": (839, 502)}
rules._maple_bound = "马超"
rules._evict_cell((556, 430), "狂铁")
rules._field["狂铁"] = (556, 430)
_s197 = GameState(phase="prepare", round_no=6, level=3, gold=4, population=3, population_cap=6, population_valid=True)
_s197.hand_slots = [{"name": "枫叶舞", "x": 500, "y": 660, "ready": False}]
_s197.unit_anchors = {"556,430": (556, 310), "839,502": (839, 380)}
_a197 = rules.maple_search_cells(_s197)
rules.confirm_unit("马超", (556, 430))
_b197 = ("马超" in rules._field, rules.maple_search_cells(_s197))
rules.reset_match()
print(f"[找回枫叶舞宿主] 要找的格子 {_a197}（期望含 556,430）；点开确认后 在账本/还要找 = {_b197}（期望 True, []）")
if "556,430" not in _a197 or _b197 != (True, []):
    failures.append(f"找回枫叶舞宿主不对：{_a197} / {_b197}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 198（2026-10-05 第 222 局：场上多个主C，等级都没起来）：首选沈梦溪没到手，守约已在场是现任主C、亚连刚到手 —
#   换备选要选守约路线（现任主C 的路线优先），不是按名单顺序先选亚连。
from agent.decision import routes as _R198
_R198.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "81682962224371742", ("1744067911380998", "50720714303640586")
rules.__dict__.pop("_forced_orig", None)
_s198 = GameState(phase="prepare", round_no=6, level=3, gold=6, population=6, population_cap=6, population_valid=True, banned_faction="逐鹿")
rules.resolve_comp(_s198)
rules._field = {"百里守约": (839, 502), "张良": (617, 372), "亚连": (956, 502)}
rules._carry_piece = "百里守约"
rules._unit_lv.update({"百里守约": 6, "亚连": 1})
rules._round_seen = 6
rules._forced_fallback(_s198)
_a198 = rules.forced_route
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R198.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
print(f"[换备选选现任主C 的路线] 守约在场是现任主C、亚连刚到手 → {_a198}（期望 50720714303640586 守约路线）")
if _a198 != "50720714303640586":
    failures.append(f"换备选没选现任主C 的路线：{_a198}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 199（2026-10-05：定一个主C 之后，找能辅助他的配套阵容，再配一些高强度辅助英雄）：
#   主C 在场且合格 → carry_supports 来自官方阵容同台频率（亚连 → 太乙真人/雅典娜/米莱狄…）+ 强力辅助；并入路线成员和这回合要买的人。没定主C 时不给。
from agent.decision import routes as _R199
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules._active_comp, rules._route_locked, rules._route_key = _R199.by_key("1744067911380998"), True, "1744067911380998"
rules._field = {"亚连": (839, 502), "牛魔": (617, 372)}
rules._round_seen = 6
_a199 = rules.route_members()
rules._carry_piece = "亚连"
_b199 = (rules.carry_supports("亚连")[:3], "太乙真人" in rules.route_members(), "明世隐" in rules.route_members())
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[主C 配套阵容] 亚连配套前 3 / 太乙算路线成员 / 强力辅助明世隐算成员 = {_b199}（期望 太乙真人/雅典娜/米莱狄, True, True）")
if not {"太乙真人", "雅典娜"} <= set(_b199[0]) or not _b199[1] or not _b199[2]:     # 10-19 重爬阵容后同台频率顺序变了（雅典娜 / 太乙 / 朵莉亚），只查两个核心在前 3
    failures.append(f"主C 配套阵容不对：{_b199}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 200（2026-10-05 第 254 局：主C 经常漏穿装备，要穿满）：
# 云中君在场、主C 虞姬已有 1 件，手里大棒（对虞姬 5.6）→ 仍先给主C，不铺给云中君/张良；防御装生命魔典（3.9）不给主C
from agent.decision import hero_traits as _ht200
_ht200.USE_HERO_ITEM_PREF = True
rules.reset_match()
rules._field = {"虞姬": (1027, 572), "云中君": (672, 430), "张良": (390, 502), "鬼谷子": (839, 502)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "虞姬", (1027, 572)
rules._equip_seen["1027,572"] = 1
_s200 = GameState(phase="prepare", round_no=8, gold=10)
_t200 = rules._equip_target(_s200, "大棒")
_u200 = rules._equip_target(_s200, "生命魔典")
_r200 = (_t200[0] if _t200 else None, _u200[0] if _u200 else None)
print(f"[主C 满装优先] 大棒 / 生命魔典 → {_r200}（期望 虞姬, 不是虞姬）")
if _r200[0] != "虞姬" or _r200[1] == "虞姬":
    failures.append(f"主C 满装优先不对：{_r200}")
_ht200.USE_HERO_ITEM_PREF = False
rules.reset_match()
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 201（2026-10-05 第 266 局）：守约 0 件，手里魂祭绝刃（3.9，没人更对口）→ 给守约；金色圣剑（法术，avoid）→ 仍留着
_ht200.USE_HERO_ITEM_PREF = True
rules.reset_match()
rules._field = {"百里守约": (1027, 572), "程咬金": (672, 430), "大司命": (839, 502)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "百里守约", (1027, 572)
_s201 = GameState(phase="prepare", round_no=9, gold=10)
_a201 = rules._equip_target(_s201, "魂祭绝刃")
_b201 = rules._equip_target(_s201, "金色圣剑")
_r201 = (_a201[0] if _a201 else None, _b201[0] if _b201 else None)
print(f"[主C 没满装兜底] 魂祭绝刃 / 金色圣剑 → {_r201}（期望 不给守约, None；设计约定：守约不穿攻速装，魂祭绝刃加攻速）")
if _r201[0] == "百里守约" or _r201[1] is not None:
    failures.append(f"主C 没满装兜底不对：{_r201}")
_ht200.USE_HERO_ITEM_PREF = False
rules.reset_match()
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 202（2026-10-06 第 300 局）：大河草案第 10 回合，枫叶舞绑在觉醒干将（86 级，前期名单里有它），主C 虞姬 32 级 → 卖干将换绑
from agent.decision import routes as _R202
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("user_compound_river_totem"), True, "user_compound_river_totem"
rules._round_seen = 10
rules._field = {"虞姬": (917, 570), "干将莫邪": (300, 569), "张良": (672, 430), "云中君": (617, 372)}
rules._carry_piece, rules._maple_bound = "虞姬", "干将莫邪"
rules._carry_switch_round, rules._carry_switch_by = 8, "rule"
rules._ever_awakened = {"干将莫邪"}
rules._unit_lv = {"干将莫邪": 86, "虞姬": 32}
rules._transfer_round = -1
rules.forced_route, rules._forced_released = "", False
_s202 = GameState(phase="prepare", round_no=10, level=5, gold=5, population=4, population_cap=7, population_valid=True)
_s202.hand_slots = [{"name": "枫叶舞", "x": 500, "y": 660, "ready": False}]
_a202 = rules._transfer_victim(_s202)[0]
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[觉醒 1 阶过渡件换绑] 第 10 回合卖 {_a202}（期望 干将莫邪）")
if _a202 != "干将莫邪":
    failures.append(f"觉醒 1 阶过渡件没换绑：{_a202}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 203（2026-10-06 第 322 局）：虞姬实际 2 件（账本），画面有几帧多读成 3 → 基础装备速击之枪仍给主C；被拒两次才当满
from agent.decision import hero_traits as _ht203
_ht203.USE_HERO_ITEM_PREF = True
rules.reset_match()
rules._field = {"虞姬": (499, 500), "大司命": (529, 372), "云中君": (672, 430)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "虞姬", (499, 500)
rules._equip_seen = {"499,500": 3, "529,372": 2}
rules._worn = {"虞姬": ["金色圣剑", "铸梦·逐风"], "大司命": ["聚灵水晶", "极寒风暴"]}
_s203 = GameState(phase="prepare", round_no=12, gold=4)
_s203.equip_counts = {"499,500": 2}
_a203 = (rules._equip_target(_s203, "速击之枪") or [None])[0]
rules._equip_rejects = {"499,500": 2}
_b203 = (rules._equip_target(_s203, "速击之枪") or [None])[0]
print(f"[主C 件数多读] 速击之枪 → {_a203}（期望 虞姬）；被拒两次后 → {_b203}（期望 不是虞姬）")
if _a203 != "虞姬" or _b203 == "虞姬":
    failures.append(f"主C 件数多读：{_a203} / {_b203}")
_ht203.USE_HERO_ITEM_PREF = False
rules.reset_match()
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 204（2026-10-08 第 401 / 407 局）：守约路线已锁、守约还在手里，第 7 回合拍卖 → 花木兰算第二主C；第 4 回合（前期临时主C）不算；守约本人不算
from agent.decision import routes as _R204
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
_k204 = "50720714303640586"
rules._active_comp, rules._route_locked, rules._route_key = _R204.by_key(_k204), True, _k204
rules._field = {"苏烈": (617, 372), "李元芳": (433, 569)}
rules._carry_piece = "李元芳"
rules._round_seen = 7
_a204 = rules.unwanted_second_carry("花木兰")
_c204 = rules.unwanted_second_carry("百里守约")
rules._round_seen = 4
_b204 = rules.unwanted_second_carry("花木兰")
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[路线已锁、主C 在手] 第 7 回合 花木兰 第二主C={_a204}（期望 True）；第 4 回合={_b204}（期望 False）；守约={_c204}（期望 False）")
if not _a204 or _b204 or _c204:
    failures.append(f"路线已锁第二主C：{_a204}/{_b204}/{_c204}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 205（2026-10-08 第 417 局）：学识宝石（每回合等级 +8%）只给主C；主C 不在场先留着；穿在孙尚香身上 → 卖孙尚香转给主C
rules.reset_match()
rules._field = {"马可波罗": (839, 502), "孙尚香": (300, 569), "曜": (617, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece = "马可波罗"
_s205 = GameState(phase="prepare", round_no=7, gold=4)
_a205 = (rules._equip_target(_s205, "学识宝石") or [None])[0]
rules._carry_piece = ""
_b205 = rules._equip_target(_s205, "学识宝石")
rules._carry_piece, rules._carry_switch_round, rules._transfer_round, rules._round_seen = "马可波罗", 5, -1, 7
rules._worn = {"孙尚香": ["学识宝石"]}
_c205 = rules._transfer_victim(_s205)
rules.reset_match()
print(f"[复利成长装备] 学识宝石 → {_a205}（期望 马可波罗）；无主C → {_b205}（期望 None）；收尾卖 {_c205[0]}（期望 孙尚香）")
if _a205 != "马可波罗" or _b205 is not None or _c205[0] != "孙尚香":
    failures.append(f"复利成长装备：{_a205}/{_b205}/{_c205}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 206（2026-10-08：迟迟拿不到想要的主C 时，在商店买个高阶主C 代替，并切换成配套阵容）：
#   首选沈梦溪、备选大河虞姬都没到，第 6 回合商店出现艾琳 → 先买；艾琳在手 → 换到艾琳路线（日落海瞬移艾琳）；第 4 回合不找替补
from agent.perception.state import ShopSlot as _SS206
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "81682962224371742", ("user_compound_river_totem",)
rules.__dict__.pop("_forced_orig", None)
_s206 = GameState(phase="prepare", round_no=6, level=4, gold=12, population=5, population_cap=6, population_valid=True,
                  banned_faction="逐鹿", shop_open=True)
_s206.shop = [_SS206(index=0, piece="艾琳", cost=3), _SS206(index=1, piece="牛魔", cost=3)]
rules.resolve_comp(_s206)
rules._field = {"牛魔": (617, 372), "孙尚香": (300, 569)}
_a206 = (rules.preempt_shop(_s206) or {}).get("piece")
_s206.round_no = 4
_d206 = (rules.preempt_shop(_s206) or {}).get("piece")
_s206.round_no = 6
_s206.hand_slots = [{"name": "艾琳", "x": 500, "y": 660}]
rules._forced_fallback(_s206)
_b206 = rules.forced_route
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
RuleEngine.ROUTE_ENABLED = False
print(f"[替补主C] 第 6 回合商店艾琳 → 买 {_a206}（期望 艾琳）；第 4 回合 → {_d206}（期望 不是艾琳）；艾琳在手 → 路线 {_b206}（期望 74364612829797406）")
if _a206 != "艾琳" or _d206 == "艾琳" or _b206 != "74364612829797406":
    failures.append(f"替补主C：{_a206}/{_d206}/{_b206}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 207（2026-10-08 第 418 局：推演战术这种临时卡，主C 用不了也要用掉）：
#   推演战术（转瞬，同一人同回合不能用两次）本回合已给西施 → 第二张给场上另一个逐鹿英雄白起，不留着
rules.reset_match()
rules._field = {"百里守约": (1027, 572), "西施": (433, 569), "白起": (718, 371), "牛魔": (617, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._round_seen = "百里守约", 9
_s207 = GameState(phase="prepare", round_no=9, gold=3, population=4, population_cap=7, population_valid=True)
_s207.hand_slots = [{"name": "推演战术", "x": 721, "y": 664}]
rules._used_on.add((9, "推演战术", "西施")); rules._no_repeat.add("推演战术")
_a207 = (rules.decide_use_cards(_s207) or {}).get("target")
rules.reset_match()
print(f"[转瞬牌换人用掉] 第二张推演战术 → {_a207}（期望 白起）")
if _a207 != "白起":
    failures.append(f"转瞬牌没换人用掉：{_a207}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 208（2026-10-08 第 422 局）：大河虞姬路线第 6 回合满员 6/6，商店有阶段替补干将莫邪 → 不先买；空位时照买
from agent.perception.state import ShopSlot as _SS208
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("user_compound_river_totem"), True, "user_compound_river_totem"
rules._round_seen = 6
rules._field = {"虞姬": (917, 570), "张良": (672, 430), "云中君": (617, 372), "鬼谷子": (529, 372), "牛魔": (433, 569), "瑶": (839, 502)}
rules._carry_piece = "虞姬"
_eng208, rules.ENGINE_ENABLED = rules.ENGINE_ENABLED, False   # 只测路线替补，不让引擎零件逻辑掺进来
rules._roll_rounds = {1, 2, 3}
_s208 = GameState(phase="prepare", round_no=6, level=3, gold=8, population=6, population_cap=6, population_valid=True, shop_open=True)
_s208.shop = [_SS208(index=0, piece="干将莫邪", cost=3)]
_a208 = (rules.preempt_shop(_s208) or {}).get("piece")
_s208.population, _s208.population_cap = 5, 6
rules._field.pop("瑶")
_b208 = (rules.preempt_shop(_s208) or {}).get("piece")
rules.ENGINE_ENABLED = _eng208
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[满员不买阶段替补] 6/6 → {_a208}（期望 None）；5/6 → {_b208}（期望 干将莫邪）")
if _a208 == "干将莫邪" or _b208 != "干将莫邪":
    failures.append(f"阶段替补：{_a208}/{_b208}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 209（2026-10-08 第 422 局：尽量凑虞姬阵容）：大河虞姬路线第 9 回合满员 7/7、手里有刘邦在等，
#   商店有路线核心张良 → 照买（场上有路线外的程咬金可换）；搜牌回合已用 2 个、核心还缺 ≥2 个 → 仍开搜牌窗口
from agent.perception.state import ShopSlot as _SS209
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
_eng209, rules.ENGINE_ENABLED = rules.ENGINE_ENABLED, False
rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("user_compound_river_totem"), True, "user_compound_river_totem"
rules._round_seen = 9
rules._field = {"虞姬": (917, 570), "瑶": (672, 430), "云中君": (617, 372), "干将莫邪": (529, 372), "程咬金": (433, 569),
                "苏烈": (839, 502), "大司命": (300, 569)}
rules._carry_piece = "虞姬"
_s209 = GameState(phase="prepare", round_no=9, level=5, gold=13, population=7, population_cap=7, population_valid=True, shop_open=True)
_s209.hand_slots = [{"name": "刘邦", "x": 500, "y": 660}]
_s209.shop = [_SS209(index=0, piece="米莱狄", cost=3), _SS209(index=1, piece="张良", cost=3)]
_a209 = (rules.preempt_shop(_s209) or {}).get("piece")
rules._roll_rounds = {5, 6}
_b209 = rules.route_core_missing(_s209)
rules.ENGINE_ENABLED = _eng209
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[凑路线核心] 满员 + 手里有人等 → 买 {_a209}（期望 张良）；核心还缺 {sorted(_b209)}（期望含 少司缘、张良）")
if _a209 != "张良" or not {"少司缘", "张良"} <= _b209:
    failures.append(f"凑路线核心：{_a209}/{_b209}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 210（2026-10-08：定了主C 后切换阵容要果断，场上低等级、不能给新主C 提供等级或效果的可以卖掉，换新主C 阵容里的牌）：
#   虞姬已定、满员 7/7，场上有 3 级苏烈（可换掉）→ 商店的虞姬配套刘禅先买；苏烈 40 级（主坦）→ 不买
from agent.perception.state import ShopSlot as _SS210
def _f210(su_lv):
    RuleEngine.ROUTE_ENABLED = True
    rules.reset_match()
    eng, rules.ENGINE_ENABLED = rules.ENGINE_ENABLED, False
    rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("user_compound_river_totem"), True, "user_compound_river_totem"
    rules._round_seen = 9
    rules._field = {"虞姬": (917, 570), "瑶": (672, 430), "云中君": (617, 372), "鬼谷子": (529, 372), "张良": (433, 569),
                    "苏烈": (839, 502), "大司命": (300, 569)}
    rules._carry_piece = "虞姬"
    rules._unit_lv = {"虞姬": 40, "苏烈": su_lv, "瑶": 8, "云中君": 9, "鬼谷子": 12, "张良": 10, "大司命": 6}
    rules._roll_rounds = {5, 6, 7, 8, 9}
    st = GameState(phase="prepare", round_no=9, level=5, gold=13, population=7, population_cap=7, population_valid=True, shop_open=True)
    st.hand_slots = [{"name": "刘邦", "x": 500, "y": 660}]
    st.shop = [_SS210(index=0, piece="小乔", cost=3)]
    out = (rules.preempt_shop(st) or {}).get("piece")
    rules.ENGINE_ENABLED = eng
    rules.reset_match()
    RuleEngine.ROUTE_ENABLED = False
    return out
_a210, _b210 = _f210(3), _f210(40)
print(f"[主C 定了换配套] 苏烈 3 级 → 买 {_a210}（期望 小乔）；苏烈 40 级 → {_b210}（期望 None）")
if _a210 != "小乔" or _b210 == "小乔":
    failures.append(f"主C 定了换配套：{_a210}/{_b210}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 211（2026-10-08：这种情况要先把李元芳移到其他位置再移回来，否则连接关系是错的）：
#   李元芳（对距离最近的【河洛】英雄用古币）已经挨着主C 沈梦溪，但它是在主C 换人前放下的 → 先挪开、下一帧挪回原格
rules.reset_match()
rules._field = {"沈梦溪": (681, 571), "李元芳": (555, 569), "苏烈": (276, 501)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._round_seen = "沈梦溪", 6
rules._confirmed_at = {"李元芳": (6, (555, 569))}
rules._buddy_link = {"李元芳": ("干将莫邪", (1, 1))}
_s211 = GameState(phase="prepare", round_no=6, gold=3)
_s211.unit_anchors = {f"{p[0]},{p[1]}": 1 for p in rules._field.values()}
_a211 = rules.decide_reposition(_s211) or {}
_b211 = rules.decide_reposition(_s211) or {}
rules.reset_match()
print(f"[刷新连接] 第一步 {_a211.get('src')}→{_a211.get('dst')}；第二步 {_b211.get('src')}→{_b211.get('dst')}（期望挪开再回到 (555, 569)）")
if not _a211 or _b211.get("dst") != (555, 569) or _a211.get("dst") == (555, 569):
    failures.append(f"刷新连接：{_a211}/{_b211}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 212（2026-10-08 第 425 局：这一回合就换主C，卖掉李元芳、再上手牌里的另一个李元芳；先升级再卖就浪费了）：
#   沈梦溪路线，第 6 回合沈梦溪刚接任（本回合换的），枫叶舞绑在 20 级李元芳 → 当回合就卖李元芳换绑；手里那张李元芳不先合成上去
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
_eng212, rules.ENGINE_ENABLED = rules.ENGINE_ENABLED, False
rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("81682962224371742"), True, "81682962224371742"
rules._round_seen = 6
rules._field = {"沈梦溪": (1027, 572), "李元芳": (300, 569), "盾山": (672, 430), "明世隐": (276, 501)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._maple_bound = "沈梦溪", "李元芳"
rules._carry_switch_round, rules._carry_switch_by, rules._transfer_round = 6, "rule", -1
rules._unit_lv = {"李元芳": 20, "沈梦溪": 1}
_s212 = GameState(phase="prepare", round_no=6, level=3, gold=2, population=4, population_cap=6, population_valid=True)
_s212.hand_slots = [{"name": "枫叶舞", "x": 400, "y": 660, "ready": False}, {"name": "李元芳", "x": 500, "y": 660}]
_a212 = rules._transfer_victim(_s212)[0]
_b212 = (rules.decide_upgrade(_s212) or {}).get("piece")
rules.ENGINE_ENABLED = _eng212
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[路线主C 当回合换绑] 卖 {_a212}（期望 李元芳）；合成 {_b212}（期望 不是李元芳）")
if _a212 != "李元芳" or _b212 == "李元芳":
    failures.append(f"路线主C 当回合换绑：{_a212}/{_b212}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 213（2026-10-08 第 429 局：主C 没换）：首选沈梦溪没到，第 5 回合临时主C 1 阶李元芳 23 级，
#   场上有 20 级守约（备选路线主C）→ 换到守约路线（1 阶临时主C 不算"已养起来"）
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "81682962224371742", ("50720714303640586", "user_compound_river_totem")
rules.__dict__.pop("_forced_orig", None)
_s213 = GameState(phase="prepare", round_no=5, level=2, gold=10, population=4, population_cap=5, population_valid=True, banned_faction="逐鹿")
rules.resolve_comp(_s213)
rules._field = {"李元芳": (300, 569), "百里守约": (917, 570), "干将莫邪": (529, 372), "孙尚香": (617, 372)}
rules._carry_piece = "李元芳"
rules._unit_lv = {"李元芳": 23, "百里守约": 20}
rules._forced_fallback(_s213)
_a213 = rules.forced_route
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
RuleEngine.ROUTE_ENABLED = False
print(f"[1 阶临时主C 不挡换备选] → {_a213}（期望 50720714303640586）")
if _a213 != "50720714303640586":
    failures.append(f"1 阶临时主C 挡换备选：{_a213}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 214（2026-10-08 第 429 局）：沈梦溪路线已让位给 30 级守约 → 路线跟着换到守约那条（守约是备选路线主C）
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "81682962224371742", ("50720714303640586", "user_compound_river_totem")
rules.__dict__.pop("_forced_orig", None)
_s214 = GameState(phase="prepare", round_no=7, level=3, gold=5, population=5, population_cap=6, population_valid=True, banned_faction="逐鹿")
rules.resolve_comp(_s214)
rules._field = {"百里守约": (917, 570), "沈梦溪": (276, 501), "盾山": (672, 430)}
rules._carry_piece, rules._forced_released = "百里守约", True
rules._forced_fallback(_s214)
_a214 = rules.forced_route
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
rules._forced_released = False
RuleEngine.ROUTE_ENABLED = False
print(f"[让位后路线跟主C] → {_a214}（期望 50720714303640586）")
if _a214 != "50720714303640586":
    failures.append(f"让位后路线没跟主C：{_a214}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 215（2026-10-08：辅助或坦克里有点输出能力的定位为副C，把过剩的输出装备交给他）：
#   守约路线、守约已满 3 件，场上苏烈（坦克，40 级输出 106）/ 牛魔（49）/ 明世隐 → 输出装速击之枪给副C 苏烈
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
_eng215, rules.ENGINE_ENABLED = rules.ENGINE_ENABLED, False
rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("50720714303640586"), True, "50720714303640586"
rules._round_seen = 9
rules._field = {"百里守约": (917, 570), "苏烈": (617, 372), "牛魔": (529, 372), "明世隐": (839, 502)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece = "百里守约"
rules._worn = {"百里守约": ["魂祭绝刃", "破晓", "末世"]}
rules._equip_seen = {"917,570": 3}
_s215 = GameState(phase="prepare", round_no=9, gold=4)
_s215.equip_counts = {"917,570": 3}
_sc215 = rules.sub_carry()
_a215 = (rules._equip_target(_s215, "速击之枪") or [None])[0]
rules.ENGINE_ENABLED = _eng215
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[副C 接过剩输出装] 副C={_sc215}（期望 苏烈）；速击之枪 → {_a215}（期望 苏烈）")
if _sc215 != "苏烈" or _a215 != "苏烈":
    failures.append(f"副C：{_sc215}/{_a215}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 216（2026-10-08 第 434 局：辅助连接关系又错了）：主C 沈梦溪在角落 (1027,572) 只有 2 个邻格，李元芳占一个，
#   明世隐（连最近的任何人）放哪都和李元芳并列 → 三步：李元芳挪远 → 明世隐挪到 (917,570) 连主C → 李元芳挪回（只比河洛，仍连主C）；下一回合不再重复
rules.reset_match()
rules._field = {"李元芳": (955, 502), "沈梦溪": (1027, 572), "盾山": (672, 430), "明世隐": (839, 502), "程咬金": (529, 372), "钟馗": (556, 430)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._round_seen = "沈梦溪", 7
rules._confirmed_at = {n: (7, p) for n, p in rules._field.items()}
_s216 = GameState(phase="prepare", round_no=7, gold=3)
_s216.unit_anchors = {f"{p[0]},{p[1]}": 1 for p in rules._field.values()}
_m216 = []
for _ in range(4):
    _x = rules.decide_reposition(_s216)
    if _x:
        rules.note_executed(_x)
    _m216.append(_x and (_x["piece"], _x["dst"]))
rules._round_seen = 8
_s216.round_no = 8
rules._confirmed_at = {n: (8, p) for n, p in rules._field.items()}
_n216 = rules.decide_reposition(_s216)
rules.reset_match()
print(f"[两个连接类英雄] {_m216}；下一回合 {_n216 and _n216.get('piece')}（期望 李元芳挪远 / 明世隐→(917,570) / 李元芳→(955,502) / None；下一回合 None）")
if (not _m216[0] or _m216[0][0] != "李元芳" or _m216[1] != ("明世隐", (917, 570))
        or _m216[2] != ("李元芳", (955, 502)) or _m216[3] or _n216):
    failures.append(f"两个连接类英雄：{_m216} / {_n216}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# ===== 场景 217~223（2026-10-09 晚卡牌审查修复）=====
from agent.perception.state import TalentOption as _TO217  # noqa: E402
from agent.decision import hero_traits as _HT217  # noqa: E402
def _st217(rnd=7, hand=(), pop=3, cap=6):
    _s = GameState(phase="prepare", round_no=rnd, gold=6)
    _s.hand_slots = [{"name": n, "x": 400 + 90 * i, "y": 664} for i, n in enumerate(hand)]
    _s.population, _s.population_cap, _s.population_valid = pop, cap, True
    _s.unit_anchors = {f"{p[0]},{p[1]}": 1 for p in rules._field.values()}
    return _s
# 217 预言：上一场没掉血 → 押胜利；掉血 → 押失败；一回合只押一张
rules.reset_match()
rules._field = {"百里守约": (917, 570), "牛魔": (529, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._round_seen = "百里守约", 7
rules._hp_by_round = {6: 50}
_s217 = _st217(7, ("胜利预言", "失败预言"))
_s217.hp = 50
_a217 = rules.decide_use_cards(_s217)
_s217.hp = 41
_b217 = rules.decide_use_cards(_s217)
rules.note_executed(_b217 or {})
_c217 = rules.decide_use_cards(_st217(7, ("胜利预言",)))
print(f"[预言] 没掉血 → {_a217 and _a217['piece']}（期望 胜利预言）；掉血 → {_b217 and _b217['piece']}（期望 失败预言）；同回合再押 → {_c217 and _c217.get('piece')}（期望 None）")
if not _a217 or _a217["piece"] != "胜利预言" or not _b217 or _b217["piece"] != "失败预言" or (_c217 and _c217.get("predict")):
    failures.append(f"预言：{_a217} / {_b217} / {_c217}")
# 218 换帅 / 挂帅：主C 曹操 5 级，最近的三分英雄赵云 20 级（唯一）→ 对主C 用；赵云只有 6 级 → 不用
rules.reset_match()
rules._field = {"曹操": (1027, 572), "赵云": (955, 502), "孙尚香": (300, 569)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._round_seen = "曹操", 8
rules._unit_lv = {"曹操": 5, "赵云": 20, "孙尚香": 30}
_a218 = rules.decide_use_cards(_st217(8, ("换帅",)))
_g218 = rules.decide_use_cards(_st217(8, ("挂帅",)))
rules._unit_lv = {"曹操": 5, "赵云": 6, "孙尚香": 30}
_b218 = rules._level_trade_target(_st217(8), "换帅", "level_swap")
print(f"[换帅/挂帅] 换帅 → {_a218 and _a218.get('target')}（期望 曹操）；挂帅 → {_g218 and _g218.get('target')}（期望 曹操）；差 1 级 → {_b218}（期望 None）")
if (not _a218 or _a218.get("target") != "曹操" or not _a218.get("rule_only")
        or not _g218 or _g218.get("target") != "曹操" or _b218 is not None):
    failures.append(f"换帅/挂帅：{_a218} / {_g218} / {_b218}")
# 219 多彩古币：古币改为只给非河洛、且不给主C（会变身）
rules.reset_match()
rules._field = {"百里守约": (917, 570), "牛魔": (529, 372), "李元芳": (955, 502)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._round_seen = "百里守约", 8
_pre219 = (rules.card_target_ok("古币", "李元芳")[0], rules.card_target_ok("古币", "牛魔")[0])
rules.note_executed({"type": "talent_pick", "index": 0, "name": "多彩古币"})
_post219 = (rules.card_target_ok("古币", "李元芳")[0], rules.card_target_ok("古币", "牛魔")[0],
            rules.card_target_ok("古币", "百里守约")[0])
_t219 = rules._reviewed_target(_st217(8, ("古币",)), "古币", rules.usage_reviewed.get("古币") or {})
print(f"[多彩古币] 之前 李元芳/牛魔 {_pre219}（期望 True/False）；之后 李元芳/牛魔/守约 {_post219}（期望 False/True/False）；给 {_t219}")
if _pre219 != (True, False) or _post219 != (False, True, False) or not _t219 or _t219[0] != "牛魔":
    failures.append(f"多彩古币：{_pre219} / {_post219} / {_t219}")
# 220 装备：小鹿的庇护在阿离局无效（谁都不穿、放满一回合卖掉）；苍穹王权 / 驱雷不再被当法术装
rules.reset_match()
rules._field = {"百里守约": (917, 570), "牛魔": (529, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._round_seen = "百里守约", 8
_v220 = (rules._equip_void("小鹿的庇护"), rules._equip_allowed("小鹿的庇护", "百里守约"))
rules._hand_since = {"小鹿的庇护": 7}
_sell220 = rules._dead_hand_card(_st217(8, ("小鹿的庇护",)))
_d220 = (_HT217.item_dmg("苍穹王权"), _HT217.item_dmg("驱雷"), _HT217.item_dmg("噬神之书"))
_ok220 = rules._equip_allowed("苍穹王权", "百里守约")
print(f"[装备审查] 小鹿 {_v220}（期望 小鹿/False）；放满一回合 → {_sell220 and _sell220['type']}（期望 sell_card）；"
      f"伤害类型 {_d220}（期望 ''/''/mag）；苍穹王权给守约 {_ok220}（期望 True）")
if _v220 != ("小鹿", False) or not _sell220 or _sell220["type"] != "sell_card" or _d220 != ("", "", "mag") or not _ok220:
    failures.append(f"装备审查：{_v220} / {_sell220} / {_d220} / {_ok220}")
# 221 hold 牌：听天由命（规则不用）放满一回合卖；锦囊 / 预言（有 rule_use）不卖
rules.reset_match()
rules._field = {"百里守约": (917, 570)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._round_seen = 9
rules._hand_since = {"听天由命": 8, "锦囊": 6, "胜利预言": 6}
_a221 = rules._dead_hand_card(_st217(9, ("锦囊", "胜利预言", "听天由命")))
rules._hand_since = {"听天由命": 9}
_b221 = rules._dead_hand_card(_st217(9, ("听天由命",)))
print(f"[hold 牌卖掉] 放满一回合 → {_a221 and _a221['piece']}（期望 听天由命）；刚拿到 → {_b221}（期望 None）")
if not _a221 or _a221["piece"] != "听天由命" or _b221 is not None:
    failures.append(f"hold 牌卖掉：{_a221} / {_b221}")
# 222 武装重构：牛魔（非核心）身上 1 件也能拆（原来要 2 件，10 局卖 9 局）；主C 身上装备对口不拆
rules.reset_match()
_ac222, rules._active_comp = rules._active_comp, None
rules._field = {"百里守约": (917, 570), "苏烈": (617, 372), "牛魔": (529, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._round_seen = "百里守约", 8
rules._worn = {"百里守约": ["无尽巨剑"], "牛魔": ["力量腰带"]}
_s222 = _st217(8, ("武装重构",))
_s222.equip_counts = {"917,570": 1, "529,372": 1}
_s222.unit_names = {"529,372": ["牛魔", 0.98]}     # 2026-10-11：拆装备要认人确认
_t222 = rules._rebuild_targets(_s222, "武装重构")
_a222 = rules.decide_use_cards(_s222)
rules._active_comp = _ac222
print(f"[武装重构] 候选 {[x[3] for x in _t222]}（期望 牛魔、不含守约）；动作 {_a222 and (_a222['type'], _a222.get('target'))}")
if not _t222 or _t222[0][3] != "牛魔" or any(x[3] == "百里守约" for x in _t222) or not _a222 or _a222.get("target") != "牛魔":
    failures.append(f"武装重构：{[x[3] for x in _t222]} / {_a222}")
# 223 锦囊先于登场英雄；扩编名额（要另行购买）不选
rules.reset_match()
rules._field = {"百里守约": (917, 570)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._round_seen = 5
_a223 = rules.decide_entry_boost(_st217(5, ("锦囊", "孙尚香"), pop=1, cap=5))
_b223 = rules.decide_entry_boost(_st217(5, ("锦囊", "牛魔"), pop=1, cap=5))
_s223 = GameState(phase="talent", round_no=8)
_s223.talents = [_TO217(0, "扩编名额", "获得1个可购买天赋：花费8能量，可上阵人数+1。可购买2次。"), _TO217(1, "人才市场", "")]
_s223.talent_can_refresh = False
_t223 = rules.decide_talent(_s223)
rules.reset_match()
print(f"[锦囊 / 扩编] 手里有登场英雄 → {_a223 and _a223['piece']}（期望 锦囊）；没有 → {_b223}（期望 None）；天赋 → {_t223 and _t223.get('name')}（期望 人才市场）")
if not _a223 or _a223["piece"] != "锦囊" or _b223 is not None or not _t223 or _t223.get("name") != "人才市场":
    failures.append(f"锦囊/扩编：{_a223} / {_b223} / {_t223}")
# 224 经济节奏接管（ECO_TAKEOVER）：levelup → 留升级费、不卡"3 级找主C"；roll → 当搜牌回合；关着时 eco 答案不起作用
rules.reset_match()
_st224 = GameState(phase="prepare", round_no=6, level=3, level_cost=8, gold=12, population=6, population_cap=6)
_st224.shop_open = True
_orig_lock, _orig_miss = rules.route_locked, rules._route_missing
rules.route_locked = lambda: True
rules._route_missing = lambda st: {"钟馗": 1.0}     # 3 阶，3 级刷得到
rules.set_eco(6, "levelup")
_off = (rules.eco_says(_st224), rules.level_budget(_st224))
RuleEngine.ECO_TAKEOVER = True
_lv = (rules.eco_says(_st224), rules.level_budget(_st224), rules.hold_level_for_carry(_st224))
rules.set_eco(6, "roll")
_ro = (rules._in_roll_window(_st224), rules.level_budget(_st224))
rules._route_missing = lambda st: {"嬴政": 1.0}     # 4 阶，3 级刷不出 → 不开窗口
_ro2 = rules._in_roll_window(_st224)
RuleEngine.ECO_TAKEOVER = False
rules.route_locked, rules._route_missing = _orig_lock, _orig_miss
rules.reset_match()
print(f"[eco 接管] 关着 {_off}；levelup {_lv}（期望 levelup/8/False）；roll 窗口 {_ro}（期望 True/0）；缺的人刷不出 {_ro2}（期望 False）")
if _off[0] != "" or _lv != ("levelup", 8, False) or _ro != (True, 0) or _ro2 is not False:
    failures.append(f"eco 接管：{_off} / {_lv} / {_ro} / {_ro2}")
# 225 主C 满装后剩余的基础装备（大棒）不压在手里（2026-10-10 第 510 局）；2026-10-19 改：辅助（瑶除外）不给，给坦克/战士
rules.reset_match()
_C225, _T225, _M225 = (718, 371), (617, 372), (529, 372)
rules._active_comp = {"carry": "百里守约", "core": []}
rules._field = {"百里守约": _C225, "牛魔": _T225, "明世隐": _M225}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "百里守约", _C225
_s225 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=10, gold=10,
                  population=3, population_cap=7, hp=60, level=5)
_s225.hand_slots = [{"name": "大棒", "x": 583, "y": 662}]
_s225.equip_counts = {"718,371": 3, "617,372": 0, "529,372": 0}
_a225 = rules.decide_use_cards(_s225)
rules.reset_match()
print(f"[剩余基础装备] 大棒 → {_a225 and _a225.get('target')}（期望 牛魔：辅助（瑶除外）不给装备、多余的给坦克/战士；不给满装的守约）")
if not _a225 or _a225.get("target") != "牛魔":
    failures.append(f"剩余基础装备：{_a225}")
# 226 少司缘的因缘图腾挪到后排角落（2026-10-10：否则会被打死，吃不到升级效果）；已在角落就不动；没有召唤者不动
rules.reset_match()
rules._field = {"少司缘": (617, 496), "虞姬": (555, 569)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
_s226 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=6, gold=5,
                  population=2, population_cap=6, hp=60, level=4)
_s226.vision_units = {"617,496": "hero", "555,569": "hero", "672,430": "summon"}
# 2026-10-13：图腾站位只认点开确认过身份的召唤物（第 605 局按"离召唤者最近"猜错了）
_u226 = rules.decide_totem_corner(_s226)
if _u226:
    failures.append(f"图腾身份没确认就挪了：{_u226}")
rules._obj_confirmed = {(672, 430): "因缘图腾", (300, 569): "因缘图腾"}
_a226 = rules.decide_totem_corner(_s226)
_s226b = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=6, gold=5,
                   population=2, population_cap=6, hp=60, level=4)
_s226b.vision_units = {"617,496": "hero", "555,569": "hero", "300,569": "summon"}
_b226 = rules.decide_totem_corner(_s226b)
rules._field = {"虞姬": (555, 569)}
_c226 = rules.decide_totem_corner(_s226)
rules.reset_match()
print(f"[图腾站位] {_a226 and (_a226['piece'], _a226['src'], _a226['dst'])}（期望 因缘图腾 (672,430)→角落）；已在角落 {_b226}；没有少司缘 {_c226}")
if (not _a226 or _a226.get("piece") != "因缘图腾" or tuple(_a226["src"]) != (672, 430)
        or tuple(_a226["dst"]) not in RuleEngine.TOTEM_CORNERS or _b226 is not None or _c226 is not None):
    failures.append(f"图腾站位：{_a226} / {_b226} / {_c226}")
# 238 选了「神明庇佑」（图腾周围 1 格的己方单位 +10% 生命 +20% 攻速，可叠加；2026-10-11：选了这个天赋就要按它摆）
#     → 因缘图腾不去后排角落，挪到主C 虞姬相邻格；没这个天赋照旧去角落
rules.reset_match()
rules._field = {"少司缘": (617, 496), "虞姬": (555, 569)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "虞姬", (555, 569)
rules._talents_taken = {"神明庇佑"}
_s238 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=6, gold=5,
                  population=2, population_cap=6, hp=60, level=4)
_s238.vision_units = {"617,496": "hero", "555,569": "hero", "1027,572": "summon"}
rules._obj_confirmed = {(1027, 572): "因缘图腾"}
_a238 = rules.decide_totem_corner(_s238)
_d238 = _a238 and rules._hex_dist((555, 569), _a238["dst"])
rules.reset_match()
print(f"[图腾光环天赋] {_a238 and (_a238['piece'], _a238['src'], _a238['dst'])}，离主C {_d238} 格（期望 1）")
if not _a238 or _a238.get("piece") != "因缘图腾" or _d238 != 1:
    failures.append(f"图腾光环天赋：{_a238} / {_d238}")
# 239 经济网络接管时商店里规则不抢先（2026-10-11 第 537 局：80 次买 / 刷由规则抢先，Laya 商店题只问到 17 次）：
#     eco=roll、商店有主C 同名 → preempt_shop 不买（交给 Laya）；eco=levelup、钱够 → 只执行升级
rules.reset_match()
RuleEngine.ECO_TAKEOVER = True
rules._field = {"百里玄策": (917, 570)}
rules._carry_piece = "百里玄策"
_s239 = GameState(phase="prepare", round_no=9, level=4, level_cost=10, gold=15, population=6, population_cap=6,
                  population_valid=True, shop_open=True)
_s239.shop = [_SS210(index=0, piece="百里玄策", cost=3)]
_s239.level_btn = (200, 650)
rules.set_eco(9, "roll")
_a239 = rules.preempt_shop(_s239)
# 2026-10-13：定好最终主C 之后大量买配套英雄 → eco=roll 时规则先买主C 同名（之前期望 None）
from agent.decision import interim_rules as _IR239
if _IR239.on("flex_buy") and _IR239.on("flex_buy_settled"):
    _a239 = None if (_a239 or {}).get("piece") == "百里玄策" and (_a239 or {}).get("why") == "最终主C 已定，买配套" else (_a239 or "没买主C 同名")
# flex_buy 关掉（R27 已学会）后回到 10-11 的约定：eco=roll 时规则不抢先，_a239 应为 None
rules.set_eco(9, "levelup")
_b239 = rules.preempt_shop(_s239)
RuleEngine.ECO_TAKEOVER = False
rules.reset_match()
print(f"[模型决定商店] eco roll → 买主C 同名 {'对' if _a239 is None else _a239}（期望 对）；eco levelup → {_b239 and _b239['type']}（期望 buy_xp）")
if _a239 is not None or not _b239 or _b239.get("type") != "buy_xp":
    failures.append(f"模型决定商店：{_a239} / {_b239}")
# 240~242 站位表（2026-10-11）
# 240 李元芳只找【河洛】：主C 虞姬（大河）时不算贴主C；主C 沈梦溪（河洛）时算
rules.reset_match()
rules._carry_piece = "虞姬"
_a240 = rules._carry_buddy("李元芳")
rules._carry_piece = "沈梦溪"
_b240 = rules._carry_buddy("李元芳")
_c240 = (rules._link_sticky("明世隐"), rules._link_sticky("瑶"), rules._link_sticky("太乙真人"))
rules.reset_match()
print(f"[李元芳派系] 主C 虞姬 → {_a240}（期望 False）；沈梦溪 → {_b240}（期望 True）；连接保持 明世隐/瑶/太乙 {_c240}（期望 T,T,F）")
if _a240 or not _b240 or _c240 != (True, True, False):
    failures.append(f"李元芳派系 / 连接保持：{_a240} / {_b240} / {_c240}")
# 241 蔡文姬（周围 1 格回血）离主C 3 格 → 挪进 1 格以内
rules.reset_match()
rules._field = {"虞姬": (390, 502), "蔡文姬": (917, 570), "牛魔": (617, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._round_seen = "虞姬", 7
_s241 = GameState(phase="prepare", round_no=7, gold=3, shop_open=False)
_s241.vision_units = {f"{v[0]},{v[1]}": "hero" for v in rules._field.values()}
_s241.unit_anchors = {f"{v[0]},{v[1]}": 1 for v in rules._field.values()}
_a241 = rules._aura_fix(_s241, {tuple(v) for v in rules._field.values()}, _s241.vision_units)
_d241 = _a241 and rules._hex_dist(_a241["dst"], (390, 502))
rules.reset_match()
print(f"[光环站位] {_a241 and (_a241['piece'], _a241['dst'])}，离主C {_d241}（期望 蔡文姬，1）")
if not _a241 or _a241.get("piece") != "蔡文姬" or _d241 != 1:
    failures.append(f"光环站位：{_a241}")
# 242 主C 落点：要贴主C 的人 ≥2（明世隐 + 太乙）→ (390,502)；没有 → 左下角
rules.reset_match()
rules._carry_piece = "虞姬"
_s242 = GameState(phase="prepare", round_no=6)
_s242.hand_slots = [{"name": "明世隐"}, {"name": "太乙真人"}]
_a242 = rules.carry_home(_s242)
_s242.hand_slots = [{"name": "牛魔"}]
_b242 = rules.carry_home(_s242)
rules.reset_match()
print(f"[主C 落点] 两个贴主C → {_a242}（期望 (390,502)）；没有 → {_b242}（期望 (300,569)）")
if _a242 != (390, 502) or _b242 != (300, 569):
    failures.append(f"主C 落点：{_a242} / {_b242}")
# 243 前期临时主C / 枫叶舞凑合绑不给配套（2026-10-11：开局主C 不宜给李元芳，给了还要卖掉再买，是亏的）
rules.reset_match()
rules.forced_route, rules.forced_backups = "50720714303640586", ("81682962224371742",)
rules._round_seen = 2
rules._field = {"李元芳": (276, 501), "孙尚香": (300, 569), "牛魔": (617, 372)}
_a243 = rules._skill_target()
_b243 = (rules._carry_eligible("李元芳"), rules._carry_eligible("孙尚香"))
rules.forced_route, rules.forced_backups = "", ()
rules.reset_match()
print(f"[临时主C 不给配套] 枫叶舞凑合绑 → {_a243}（期望 孙尚香）；主C 资格 李元芳/孙尚香 {_b243}（期望 False/True）")
if _a243 != "孙尚香" or _b243 != (False, True):
    failures.append(f"临时主C 不给配套：{_a243} / {_b243}")
# 244 场上只有配套（李元芳 / 牛魔）时照样有临时主C、枫叶舞照样绑（2026-10-11 回退：第 567 / 570 局前 4 回合无主C）
rules.reset_match()
rules.forced_route, rules.forced_backups = "50720714303640586", ("81682962224371742",)
rules._round_seen = 2
rules._field = {"李元芳": (276, 501), "牛魔": (617, 372)}
_a244 = rules._carry_eligible("李元芳")
_b244 = rules._skill_target()
rules.forced_route, rules.forced_backups = "", ()
rules.reset_match()
# 2026-10-14 修正：先绑，看能不能凑杀人；之后的李元芳留手里，真主C 到了卖场上的、上手里的 → 只有配套时照绑李元芳
print(f"[只有配套时] 李元芳 可当临时主C={_a244}（期望 True）；枫叶舞 → {_b244}（期望 李元芳）")
if not _a244 or _b244 != "李元芳":
    failures.append(f"只有配套时：{_a244} / {_b244}")
# 227 第 515 局：主C 虞姬在角落 (276,501)，邻格被 少司缘/鬼谷子/张良 占满，瑶在远处 → 先把不需要贴主C 的少司缘挪到后排中间，下一帧瑶挪进腾出的格子；
#     少司缘上场不贴主C（2026-10-10：只有个别英雄必须站在主C 旁边，其余放后排中间左右）；先知图腾去前排左角
rules.reset_match()
rules._field = {"虞姬": (276, 501), "少司缘": (346, 430), "鬼谷子": (390, 502), "张良": (300, 569), "瑶": (411, 372), "牛魔": (617, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "虞姬", (276, 501)
_s227 = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=7, gold=3,
                  population=6, population_cap=7, hp=50, level=4)
_s227.vision_units = {f"{v[0]},{v[1]}": "hero" for v in rules._field.values()}
_a227 = rules.decide_reposition(_s227)
_b227 = rules.decide_reposition(_s227)
print(f"[腾位贴主C] 第一步 {_a227 and (_a227['piece'], _a227['src'], _a227['dst'])}；第二步 {_b227 and (_b227['piece'], _b227['dst'])}")
if (not _a227 or _a227["piece"] != "少司缘" or tuple(_a227["dst"]) not in RuleEngine.BACK_MID_CELLS
        or not _b227 or _b227["piece"] != "瑶" or tuple(_b227["dst"]) != (346, 430)):
    failures.append(f"腾位贴主C：{_a227} / {_b227}")
rules.reset_match()
rules._field = {"虞姬": (276, 501), "牛魔": (617, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "虞姬", (276, 501)
_s227b = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=7, gold=3,
                   population=2, population_cap=7, hp=50, level=4)
_s227b.hand_slots = [{"name": "少司缘", "x": 700, "y": 663}]
_s227b.hand = ["少司缘"]
_d227 = rules.decide_deploy(_s227b)
print(f"[不贴主C 的辅助] 少司缘 → {_d227 and _d227.get('dst')}（期望后排中间 {RuleEngine.BACK_MID_CELLS}）")
if not _d227 or tuple(_d227["dst"]) not in RuleEngine.BACK_MID_CELLS:
    failures.append(f"少司缘落点：{_d227}")
rules.reset_match()
rules._field = {"鬼谷子": (499, 500), "虞姬": (276, 501)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
_s227c = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=7, gold=3,
                   population=2, population_cap=7, hp=50, level=4)
_s227c.vision_units = {"499,500": "hero", "276,501": "hero", "555,569": "summon"}
rules._obj_confirmed = {(555, 569): "先知图腾"}
_t227 = rules.decide_totem_corner(_s227c)
rules.reset_match()
print(f"[先知图腾] {_t227 and (_t227['piece'], _t227['dst'])}（期望 先知图腾 → (319,375)）")
if not _t227 or _t227["piece"] != "先知图腾" or tuple(_t227["dst"]) != (319, 375):
    failures.append(f"先知图腾：{_t227}")
# 228 第 515 局：主C 虞姬 1 件，极寒风暴 / 护命灵珠 给了 1 级大司命（事后手动挪给主C）；护命灵珠"护盾…移除"被当成移除英雄、把穿的人删出账本
def _s228(card):
    st = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=9, gold=5,
                   population=3, population_cap=7, hp=40, level=5)
    st.hand_slots = [{"name": card, "x": 721, "y": 663}]
    st.hand = [card]
    st.equip_counts = {"276,501": 1, "556,430": 0, "617,372": 0}
    return st
def _setup228(tank_lv):
    rules.reset_match()
    rules._field = {"虞姬": (276, 501), "大司命": (556, 430)}
    rules._cell_owner = {v: k for k, v in rules._field.items()}
    rules._carry_piece, rules._carry = "虞姬", (276, 501)
    rules._unit_lv = {"虞姬": 40, "大司命": tank_lv}
_setup228(1)
_a228 = rules.decide_use_cards(_s228("护命灵珠"))
_setup228(1)
_b228 = rules.decide_use_cards(_s228("极寒风暴"))
_setup228(20)
_c228 = rules.decide_use_cards(_s228("极寒风暴"))
_setup228(20)
rules.confirm_drag({"type": "drag_card", "piece": "护命灵珠", "src": (721, 663), "dst": (556, 430), "kind": "装备牌", "target": "大司命"}, True)
_kept228 = "大司命" in rules._field
rules.reset_match()
print(f"[防御装给谁] 护命灵珠 → {_a228 and _a228.get('target')}（期望 虞姬）；极寒风暴 坦克 1 级 → {_b228 and _b228.get('target')}（期望 虞姬）；"
      f"坦克 20 级 → {_c228 and _c228.get('target')}（期望 大司命）；穿护命灵珠后大司命仍在账本 {_kept228}")
if (not _a228 or _a228.get("target") != "虞姬" or not _b228 or _b228.get("target") != "虞姬"
        or not _c228 or _c228.get("target") != "大司命" or not _kept228):
    failures.append(f"防御装给谁：{_a228} / {_b228} / {_c228} / {_kept228}")
# 229 主C 定了（路线已锁、主C 在场）后不买新的 C 位英雄（第 515 局 R11 买了敖隐，事后被卖掉：低等级 C 位英雄后上没有用）
rules.reset_match()
rules._field = {"虞姬": (276, 501), "大司命": (556, 430)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "虞姬", (276, 501)
_orig229 = rules.carry_settled
rules.carry_settled = lambda: True
_lc229 = (rules.late_c_position("敖隐"), rules.late_c_position("虞姬"), rules.late_c_position("大司命"))
rules.carry_settled = lambda: False
_lc229b = rules.late_c_position("敖隐")
rules.carry_settled = _orig229
rules.reset_match()
print(f"[主C 定后不买 C 位] 敖隐/虞姬/大司命 = {_lc229}（期望 True/False/False）；主C 没定时敖隐 {_lc229b}（期望 False）")
if _lc229 != (True, False, False) or _lc229b:
    failures.append(f"主C 定后不买 C 位：{_lc229} / {_lc229b}")
# 230 第 515 局 R14：满员 7/7、场上都是路线成员 / 养起来的人，"引擎缺零件先买"照样买了云中君（阵容已完全成型，买也是白买）
def _f230(extra_lv):
    RuleEngine.ROUTE_ENABLED = True
    rules.reset_match()
    rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("user_compound_river_totem"), True, "user_compound_river_totem"
    rules._round_seen = 14
    mem = [n for n in rules.route_members() if n not in ("云中君",)]
    names = (["虞姬"] + [n for n in mem if n != "虞姬"])[:7]
    cells = [(917, 570), (672, 430), (617, 372), (529, 372), (433, 569), (839, 502), (300, 569)]
    rules._field = dict(zip(names, cells))
    rules._cell_owner = {v: k for k, v in rules._field.items()}
    rules._carry_piece = "虞姬"
    rules._unit_lv = {n: (60 if n == "虞姬" else extra_lv) for n in names}
    st = GameState(phase="prepare", round_no=14, level=6, gold=15, population=7, population_cap=7, population_valid=True, shop_open=True)
    st.shop = [_SS210(index=2, piece="云中君", cost=3)]
    out = (rules.preempt_shop(st) or {}).get("piece")
    rules.reset_match()
    RuleEngine.ROUTE_ENABLED = False
    return out, len(names)
_a230 = _f230(20)
print(f"[满员不买上不了场的零件] 场上 {_a230[1]} 人都是路线成员 → 买 {_a230[0]}（期望 不是 云中君）")
if _a230[0] == "云中君":
    failures.append(f"满员仍买零件：{_a230}")
# 231 第 518 局：禁河洛走大河虞姬、虞姬整局没到；1 阶临时主C 孙尚香第 6 回合起"不合格"被清掉，之后整局没有主C。
#     没有合格接替者时续用临时主C；有合格的人（2 阶以上输出位）时照常清掉重选。
def _f231(extra):
    RuleEngine.ROUTE_ENABLED = True
    rules.reset_match()
    rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("user_compound_river_totem"), True, "user_compound_river_totem"
    rules._round_seen = 8
    rules._field = {"孙尚香": (300, 569), "云中君": (617, 372), **extra}
    rules._cell_owner = {v: k for k, v in rules._field.items()}
    rules._carry_piece, rules._carry = "孙尚香", (300, 569)
    rules._unit_lv = {"孙尚香": 33, "云中君": 9, **{k: 20 for k in extra}}
    st = GameState(phase="prepare", round_no=8, level=4, gold=5, population=2 + len(extra), population_cap=6, population_valid=True)
    rules.choose_carry(st)
    out = rules._carry_piece
    rules.reset_match()
    RuleEngine.ROUTE_ENABLED = False
    return out
_a231, _b231 = _f231({}), _f231({"百里守约": (1027, 572)})
print(f"[临时主C 续用] 没人接替 → {_a231}（期望 孙尚香）；场上有守约 → {_b231}（期望 不是 孙尚香）")
if _a231 != "孙尚香" or _b231 == "孙尚香":
    failures.append(f"临时主C 续用：{_a231} / {_b231}")
# 232 2026-10-10：阿离主要练主C，主T 等级连带着上去就行，碰到卡买一下，不急；多余装备选辅助性质的，比如开局晕对手
rules.reset_match()
rules._field = {"虞姬": (276, 501), "牛魔": (617, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "虞姬", (276, 501)
_r232a = rules.equip_role_for("牛魔")
rules._equip_seen["276,501"] = 3
_r232b = rules.equip_role_for("牛魔")
from agent.perception.extractors import pick_equip_index as _pei232
_p232 = _pei232(PROJECT_DIR / "data", "破碎圣杯", _r232b)
RuleEngine.ROUTE_ENABLED = True
rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("50720714303640586"), True, "50720714303640586"
_st232 = GameState(phase="prepare", round_no=6, level=3, gold=8, population=2, population_cap=6, population_valid=True)
_m232 = rules._route_missing(_st232)
_tanks232 = [n for n in _m232 if rules._is_tank(n) and not rules._essential_support(n)]   # 非必要配套的坦克不该出现
_ess232 = [n for n in _m232 if rules._is_tank(n) and rules._essential_support(n)]
RuleEngine.ROUTE_ENABLED = False
rules.reset_match()
print(f"[主T 不急 / 剩余装备挑控制] 主C 没满装 {_r232a}；满装后 {_r232b} → 破碎圣杯选 {_p232}；守约路线缺的人 {sorted(_m232)}（必要配套的坦克照找 {_ess232}；非必要坦克不该有：{_tanks232}）")
if _r232a != "tank" or not _r232b.startswith("util:") or not _p232 or _p232[1] != "破茧之衣" or _tanks232:
    failures.append(f"主T 不急 / 剩余装备：{_r232a} / {_r232b} / {_p232} / {_tanks232}")
# 233 2026-10-10：① 主C 放左下角、其他远程放右下角（别都挤一个角）；② 觉醒后更能辅助主C 的最终阵容成员，商店碰到同名就买、合到觉醒
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("user_compound_river_totem"), True, "user_compound_river_totem"
rules._round_seen = 6
rules._field = {"牛魔": (617, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece = "虞姬"
def _dep233(name):
    st = GameState(phase="prepare", round_no=6, level=4, gold=0, population=len(rules._field), population_cap=6,
                   population_valid=True, shop_open=False)
    st.hand_slots = [{"name": name, "x": 700, "y": 663}]
    st.hand = [name]
    d = rules.decide_deploy(st)
    return tuple(d["dst"]) if d else None
_c233 = _dep233("虞姬")
rules._field["虞姬"] = _c233 or (300, 569)
rules._cell_owner = {v: k for k, v in rules._field.items()}
_o233 = _dep233("孙尚香")
rules._field = {"虞姬": (300, 569), "张良": (433, 569), "牛魔": (617, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
_st233 = GameState(phase="prepare", round_no=6, level=4, gold=6, population=3, population_cap=6, population_valid=True, shop_open=True)
_st233.shop = [_SS210(index=0, piece="张良", cost=3)]
_b233 = rules._merge_buy(_st233, 6)
RuleEngine.ROUTE_ENABLED = False
rules.reset_match()
print(f"[站位分散 / 觉醒辅助] 主C 虞姬 → {_c233}（期望 左下 (300,569)）；孙尚香 → {_o233}（期望靠右下）；张良同名 → {_b233 and _b233.get('piece')}（期望 张良）")
if _c233 != (300, 569) or not _o233 or _o233[0] < 700 or not _b233 or _b233.get("piece") != "张良":
    failures.append(f"站位分散 / 觉醒辅助：{_c233} / {_o233} / {_b233}")
# 234 第 522 局：主C 守约 (1027,572)，明世隐 (955,502)，太乙真人上场放到了 2 步外 (839,502)（邻格 (917,570) 被老的历史失败率 0.95 排到最后）
rules.reset_match()
rules._field = {"百里守约": (1027, 572), "明世隐": (955, 502), "程咬金": (529, 372), "苏烈": (617, 372)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "百里守约", (1027, 572)
_orig_bad234 = rules._drop_fail_rate
rules._drop_fail_rate = lambda c: 0.95 if tuple(c) == (917, 570) else 0.1
_st234 = GameState(phase="prepare", round_no=9, level=5, gold=0, population=4, population_cap=7, population_valid=True, shop_open=False)
_st234.hand_slots = [{"name": "太乙真人", "x": 700, "y": 663}]
_st234.hand = ["太乙真人"]
_d234 = rules.decide_deploy(_st234)
rules._drop_fail_rate = _orig_bad234
_dst234 = tuple(_d234["dst"]) if _d234 else None
# 2026-10-11 口径改为"主C 是唯一最近的友军（按格子步数）"（明世隐连接最近的英雄，不一定要贴旁边，按格子距离算）：
#   (917,570) 挨着守约但也挨着明世隐（并列）→ 不选；选守约唯一最近的格子
_oth234 = [(955, 502), (529, 372), (617, 372)]
# 守约在角落、明世隐挨着守约 → 空格里没有"守约唯一最近"的格子，退而挨着守约（917,570）（历史失败率 0.95 不能把它排到后面）
_adj234 = _dst234 == (917, 570)
rules.reset_match()
print(f"[贴主C 一把到位] 太乙真人 → {_dst234}（期望守约 (1027,572) 是唯一最近：{_adj234}）")
if not _adj234:
    failures.append(f"太乙没贴主C：{_d234}")
# 235 第 524 局：守约路线，守约第 6 回合没到，亚连当了主C；路线却还挂在守约的河洛体系上 → 第 10 回合守约到场成了第二主C，古币喂不到主C。
#     先预想一个阵容去凑英雄，主C 迟迟不到场就切换另一个已经获得的，按场上的去配；选定后就要坚持；
#     古币喂不到主C 就给 T 或 40 级技能效果好的，李元芳没用
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json", lord="阿离")
rules.forced_route, rules.forced_backups = "50720714303640586", ()
rules._round_seen = 6
rules._field = {"亚连": (672, 430), "程咬金": (556, 430), "钟馗": (718, 371), "盾山": (818, 374), "李元芳": (720, 500)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece = "亚连"
rules._unit_lv = {"亚连": 20, "程咬金": 8, "钟馗": 6, "盾山": 5, "李元芳": 9}
_st235 = GameState(phase="prepare", round_no=6, level=4, gold=5, population=5, population_cap=6, population_valid=True,
                   banned_faction="逐鹿")
rules.resolve_comp(_st235)
rules.resolve_comp(_st235)
_rc235 = (rules._active_comp or {}).get("carry")
_miss235 = set(rules._route_missing(_st235))
rules._field["百里守约"] = (300, 569)
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._unit_lv["百里守约"] = 1
_st235b = GameState(phase="prepare", round_no=10, level=5, gold=5, population=6, population_cap=6, population_valid=True,
                    banned_faction="逐鹿")
rules._round_seen = 10
rules.resolve_comp(_st235b)
_c235 = rules._carry_piece
_st235b.hand_slots = [{"name": "古币", "x": 722, "y": 665}]
_st235b.hand = ["古币"]
_st235b.equip_counts = {}
_a235 = rules.decide_use_cards(_st235b)
rules.forced_route, rules.forced_backups = "", ()
RuleEngine.ROUTE_ENABLED = False
rules.reset_match()
# 2026-10-10 晚（蒙特卡洛 tools/sim_route_vs_flex.py）：**先定体系、主C 在体系内灵活**最好。亚连（日落海）不在守约的河洛体系 →
#   不转正、路线不跟，只当临时主C；守约第 10 回合到场就接任；古币给守约。（同体系的沈梦溪 / 狄仁杰在手才换路线，见场景 149 / 195）
print(f"[体系内灵活] R6 路线主C → {_rc235}（期望 百里守约，亚连不转正）；R10 守约到场后主C={_c235}（期望 百里守约）；"
      f"古币给 {_a235 and _a235.get('target')}（期望 百里守约）")
if _rc235 != "百里守约" or _c235 != "百里守约" or not _a235 or _a235.get("target") != "百里守约":
    failures.append(f"体系内灵活：{_rc235} / {sorted(_miss235)} / {_c235} / {_a235}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 236 第 530 局：配套缺 4 个、第 8 回合 13 能量，eco 选 save → 原来直接关掉搜牌窗口，一次没刷。
#     现在 save 只在"下回合就拍卖"或"钱 < 10"时真攒；否则照规则开搜牌窗口
rules.reset_match()
_orig_lock236, _orig_miss236 = rules.route_locked, rules._route_missing
rules.route_locked = lambda: True
rules._route_missing = lambda st: {"明世隐": 1.0, "太乙真人": 1.0, "程咬金": 1.0, "铠": 1.0}
rules.roll_level = lambda st: 4
RuleEngine.ECO_TAKEOVER = True
_s236 = GameState(phase="prepare", round_no=9, level=4, level_cost=19, gold=13, population=6, population_cap=6)
_s236.shop_open = True
rules.set_eco(9, "save")
_w236 = rules._in_roll_window(_s236)
_s236b = GameState(phase="prepare", round_no=11, level=4, level_cost=19, gold=13, population=6, population_cap=6)
_s236b.shop_open = True
rules.set_eco(11, "save")
_w236b = rules._in_roll_window(_s236b)                 # 第 11 回合后就是第 12 回合拍卖 → 真攒
RuleEngine.ECO_TAKEOVER = False
rules.route_locked, rules._route_missing = _orig_lock236, _orig_miss236
del rules.roll_level
rules.reset_match()
# 2026-10-10：应由经济网络决定买牌合算还是升级合算，不写死规则（训练好后会互相冲突） →
#   eco 选 save 就不开搜牌窗口（规则不替它刷），两种情况都是 False
print(f"[eco save 听网络] 第 9 回合 13 能量 → 搜牌窗口 {_w236}（期望 False）；拍卖前一回合 → {_w236b}（期望 False）")
if _w236 or _w236b:
    failures.append(f"eco save 听网络：{_w236} / {_w236b}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 237 按模拟买牌（之前已做过数学模拟，先按模拟的来）：第 530 局局面 —— 沈梦溪路线，配套缺 明世隐 / 太乙 / 程咬金 / 铠，
#     商店有程咬金 → 买；商店没有要的人、13 能量 → 刷新（不关店）
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json", lord="阿离")
rules.forced_route, rules.forced_backups = "81682962224371742", ("50720714303640586",)
rules.__dict__.pop("_forced_orig", None)
_s237 = GameState(phase="prepare", round_no=8, level=4, level_cost=19, gold=13, population=5, population_cap=6,
                  population_valid=True, shop_open=True, banned_faction="三分之地")
rules.resolve_comp(_s237)
rules._field = {"沈梦溪": (300, 569), "钟馗": (529, 372), "盾山": (411, 372), "孙膑": (955, 502), "牛魔": (617, 372), "墨子": (672, 430)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece = "沈梦溪"
rules._unit_lv = {"沈梦溪": 25, "钟馗": 5, "盾山": 4}
rules._ever_awakened.add("沈梦溪")
_s237.shop = [_SS210(index=0, piece="项羽", cost=3), _SS210(index=1, piece="程咬金", cost=3), _SS210(index=2, piece="李白", cost=3)]
_a237 = rules._sim_policy(_s237, 13)
_s237.population = 6                      # 2026-10-11：满员时新配套（要卖人腾位）不由规则抢先买，交给 Laya / 换人网络
_d237 = rules._sim_policy(_s237, 13)
_s237.population = 5
_s237.shop = [_SS210(index=0, piece="项羽", cost=3), _SS210(index=1, piece="李白", cost=3)]
_s237.refresh_cost = 2
_b237 = rules._sim_policy(_s237, 13)
_c237 = rules._sim_policy(_s237, 4)
rules.forced_route, rules.forced_backups = "", ()
RuleEngine.ROUTE_ENABLED = False
rules.reset_match()
print(f"[按模拟买牌] 程咬金在店 → {_a237 and (_a237['type'], _a237.get('piece'))}（期望 买程咬金）；没要的人 13 能量 → "
      f"{_b237 and _b237['type']}（期望 refresh_shop）；只剩 4 → {_c237}（期望 None，刷完买不起）")
print(f"[按模拟买牌·满员] → {_d237 and (_d237['type'], _d237.get('piece'))}（期望 不买程咬金）")
if _d237 and _d237.get("piece") == "程咬金":
    failures.append(f"满员抢买新配套：{_d237}")
if (not _a237 or _a237.get("piece") != "程咬金" or not _b237 or _b237.get("type") != "refresh_shop" or _c237 is not None):
    failures.append(f"按模拟买牌：{_a237} / {_b237} / {_c237}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 245 最低配套先上（2026-10-11：先凑最低配套再凑其他英雄；太乙等前期不急，先上场核心引擎；
#     第 573 局第 8 回合太乙真人先上、明世隐在手里等了两回合）：守约路线，手里 太乙真人 + 明世隐、只剩 1 个空位 → 先上明世隐；
#     手里只有太乙 → 照上（空位不空着）
rules.reset_match()
rules.forced_route = "50720714303640586"
rules._route_key = "50720714303640586"
rules._field = {"百里守约": (300, 569), "盾山": (529, 372), "牛魔": (617, 372), "瑶": (276, 501), "程咬金": (556, 430)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece = "百里守约"
_s245 = GameState(phase="prepare", round_no=8, level=4, gold=10, population=5, population_cap=6, population_valid=True)
_s245.hand_slots = [{"name": "太乙真人", "x": 520, "y": 663, "name_y": 688}, {"name": "明世隐", "x": 660, "y": 663, "name_y": 688}]
_s245.hand = ["太乙真人", "明世隐"]
_a245 = rules._decide_deploy(_s245)
rules._deployed_count = 0
_s245.hand_slots = [{"name": "太乙真人", "x": 520, "y": 663, "name_y": 688}]
_s245.hand = ["太乙真人"]
_b245 = rules._decide_deploy(_s245)
rules.forced_route, rules._route_key = "", ""
rules.reset_match()
print(f"[最低配套先上] 太乙+明世隐 → {_a245 and _a245.get('piece')}（期望 明世隐）；只有太乙 → {_b245 and _b245.get('piece')}（期望 太乙真人）")
if not _a245 or _a245.get("piece") != "明世隐" or not _b245 or _b245.get("piece") != "太乙真人":
    failures.append(f"最低配套先上：{_a245} / {_b245}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 246（2026-10-12 第 575 局：低级错误，马可波罗等级已很高，不如直接走马可波罗的套路）：
#   a) 守约路线、守约没到手，第 7 回合局面网络把主C 换成马可波罗（体系外）→ 换主C 收尾照常卖枫叶舞宿主干将莫邪
#   b) 路线跟着主C 走：指定路线里有以马可波罗为主C 的 user_marco_sunset → 换过去
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json", lord="阿离")
rules.forced_route, rules.forced_backups = "50720714303640586", ("81682962224371742", "user_marco_sunset")
rules.__dict__.pop("_forced_orig", None)
_s246 = GameState(phase="prepare", round_no=7, level=4, gold=10, population=6, population_cap=6, population_valid=True)
rules.resolve_comp(_s246)
rules._field = {"干将莫邪": (1027, 572), "马可波罗": (300, 569), "李元芳": (917, 570), "程咬金": (672, 430), "盾山": (617, 372), "明世隐": (276, 501)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._unit_lv = {"马可波罗": 46, "干将莫邪": 7, "李元芳": 10}
rules._maple_bound = "干将莫邪"
rules._carry_piece = "马可波罗"
rules._carry_switch_by, rules._carry_switch_round = "Laya 0.57", 6
rules._round_seen = 7
_v246, _w246 = rules._transfer_victim(_s246)
rules._forced_released = True
rules._follow_carry_route(_s246)
_r246 = rules.forced_route
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
RuleEngine.ROUTE_ENABLED = False
print(f"[模型定的主C 收尾 + 路线跟主C] 卖 {_v246}（期望 干将莫邪）；路线 → {_r246}（期望 user_marco_sunset）")
if _v246 != "干将莫邪" or _r246 != "user_marco_sunset":
    failures.append(f"模型定的主C 收尾 / 路线跟主C：{_v246} / {_r246}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 247（2026-10-12 清查"规则挡住模型决定"）：模型接管时
#   a) 经济网络说 roll → 规则自己不升级（第 574 局第 9 回合规则先升了级）
#   b) 换人网络本回合判"先留在手里"的英雄，本回合不被当闲置卖掉（第 575 局朵莉亚 / 扁鹊 5~14 秒后被卖）
#   c) 局面网络接管换主C 时，路线主C 到场不被规则直接 apply_carry
RuleEngine.ECO_TAKEOVER = True
rules.reset_match()
_s247 = GameState(phase="prepare", round_no=9, level=4, gold=20, level_cost=12, population=6, population_cap=6, population_valid=True)
_s247.level_btn = (60, 546)
rules.set_eco(9, "roll")
_a247 = rules.decide_economy(_s247)
rules._field = {"马可波罗": (300, 569), "牛魔": (617, 372), "盾山": (529, 372), "程咬金": (672, 430), "明世隐": (276, 501), "李元芳": (917, 570)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece = "马可波罗"
_s247.hand_slots = [{"name": "朵莉亚", "x": 600, "y": 663, "name_y": 688}]
_s247.hand = ["朵莉亚"]
rules.__dict__.setdefault("_swap_benched", {})["朵莉亚"] = 9
import time as _t247  # noqa: E402
rules._hand_seen_ts["朵莉亚"] = _t247.time() - 60
_b247 = rules._idle_hand_hero(_s247)
RuleEngine.ROUTE_ENABLED = True
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json", lord="阿离")
rules.forced_route, rules.forced_backups = "50720714303640586", ()
rules.__dict__.pop("_forced_orig", None)
rules.carry_by_laya = True
_s247c = GameState(phase="prepare", round_no=7, level=4, gold=10, population=6, population_cap=6, population_valid=True)
rules.resolve_comp(_s247c)
rules._field["百里守约"] = (390, 502)
rules._carry_piece, rules._carry_switch_by = "马可波罗", "Laya 0.6"
rules.resolve_comp(_s247c)
_c247 = rules._carry_piece
rules.carry_by_laya = False
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
RuleEngine.ROUTE_ENABLED = False
RuleEngine.ECO_TAKEOVER = False
rules.reset_match()
print(f"[模型接管不被规则挡] eco=roll 时规则升级 → {_a247}（期望 None）；判过不换的朵莉亚 → {_b247 and _b247.get('piece')}（期望 None）；"
      f"守约到场后主C → {_c247}（期望 马可波罗，由局面网络定）")
if _a247 is not None or _b247 is not None or _c247 != "马可波罗":
    failures.append(f"模型接管不被规则挡：{_a247} / {_b247} / {_c247}")

# 248 古币机（2026-10-12：有古币机且沈梦溪用过效果牌时，古币优先给古币机；古币机可挪到不碍事的地方）
rules.reset_match()
_s248 = GameState(phase="prepare", round_no=6, level=4, gold=8, population=4, population_cap=6, population_valid=True)
rules._round_seen = 6
rules._field.update({"沈梦溪": (390, 502), "李元芳": (276, 501)})
rules.note_object((555, 569), "古币机")
rules.coin_machine_wanted = True
_a248 = rules._reviewed_target(_s248, "古币", {"target": "carry"})
rules.note_used_on(6, "古币", "沈梦溪")
_b248 = rules._reviewed_target(_s248, "古币", {"target": "carry"})
_m248 = rules.decide_coin_machine_move(_s248)
_m248b = rules.decide_coin_machine_move(_s248)
rules.note_object((980, 430), "古币机")
_o248 = (555, 569) in rules.object_cells
rules.reset_match()
print(f"[古币机] 沈梦溪没吃过 → {_a248}；吃过 → {_b248}；挪位 → {_m248 and (_m248['src'], _m248['dst'])}；同回合再挪 → {_m248b}；旧格放出 → {not _o248}")
if (not _a248 or _a248[0] != "沈梦溪" or _b248 != ("古币机", (555, 569)) or not _m248 or _m248["dst"] != (980, 430)
        or _m248b is not None or _o248):
    failures.append(f"古币机：{_a248} / {_b248} / {_m248} / {_m248b} / {_o248}")

# 250 召唤物跟着召唤者（2026-10-12：机关傀儡是鲁班大师召唤的，卖掉鲁班大师就一起消失，不占人口）
rules.reset_match()
_s250 = GameState(phase="prepare", round_no=8, level=4, gold=6, population=6, population_cap=6, population_valid=True)
rules._field = {"鲁班大师": (718, 371), "苏烈": (529, 372)}
rules.note_object((617, 496), "机关傀儡")
_who250 = rules.summoner_of("机关傀儡")
rules.note_executed({"type": "sell_unit", "piece": "鲁班大师", "src": (718, 371)})
_gone250 = (617, 496) not in rules.object_cells
rules.reset_match()
print(f"[召唤物跟召唤者] 机关傀儡的召唤者={_who250}（期望 鲁班大师）；卖掉后格子放出={_gone250}（期望 True）")
if _who250 != "鲁班大师" or not _gone250:
    failures.append(f"召唤物跟召唤者：{_who250} / {_gone250}")

# 251 第 583 局（主C 和明世隐的站位乱了，主要因为有个干将；跟主C 无关的英雄站另一边的角落）
rules.reset_match()
_s251 = GameState(phase="prepare", round_no=10, level=5, gold=6, population=8, population_cap=8, population_valid=True)
_f251 = {"百里守约": (955, 502), "明世隐": (839, 502), "干将莫邪": (917, 570), "李信": (720, 500), "盾山": (783, 438),
         "苏烈": (617, 372), "铠": (529, 372), "李元芳": (1027, 572)}
rules._field = dict(_f251)
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece, rules._carry = "百里守约", (955, 502)
_s251.vision_units = {f"{x},{y}": "hero" for x, y in _f251.values()}
_mv251 = []
for _i in range(4):
    _occ = {tuple(v) for v in rules._field.values()}
    _a = rules._clear_buddy_ties(_s251, _occ, _s251.vision_units)
    if not _a:
        break
    _mv251.append((_a["piece"], _a["dst"]))
    _s251.vision_units = {f"{x},{y}": "hero" for x, y in rules._field.values()}
_mp = rules._field["明世隐"]
_tied = [n for n, q in rules._field.items() if n not in ("明世隐", "百里守约", "李元芳")
         and rules._hex_dist(tuple(q), _mp) <= rules._hex_dist(_mp, (955, 502))]
rules.reset_match()
print(f"[无关英雄让开] 挪了 {_mv251}；还和主C 一样近的={_tied}（期望 [] 或只剩 1 个且已用完次数）")
if not _mv251 or not any(n == "干将莫邪" for n, _ in _mv251) or len(_tied) > 1:
    failures.append(f"无关英雄让开：{_mv251} / {_tied}")

# 252 第 584 局（1 阶占位主C 必须让位，否则枫叶舞滚不起来）：现任 1 阶李元芳（觉醒、绑枫叶舞），沈梦溪（2 阶）上场 → 立即让位并换绑
from agent.decision import routes as _R252
_R252.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules.forced_route, rules.forced_backups = "50720714303640586", ()
rules.__dict__.pop("_forced_orig", None)
rules.carry_by_laya = True
_md252 = (RuleEngine.ECO_TAKEOVER, RuleEngine.MODEL_DECIDES)
RuleEngine.ECO_TAKEOVER = RuleEngine.MODEL_DECIDES = True      # 线上：模型接管
_s252 = GameState(phase="prepare", round_no=4, level=3, gold=4, population=4, population_cap=6, population_valid=True)
rules._field = {"李元芳": (300, 569), "沈梦溪": (917, 570), "牛魔": (529, 372), "苏烈": (617, 372)}
rules.resolve_comp(_s252)
rules._carry_piece, rules._carry = "李元芳", (300, 569)
rules._carry_switch_by, rules._carry_switch_round = "规则", 2
rules._ever_awakened.add("李元芳")
rules._maple_bound = "李元芳"
_y252 = rules.placeholder_yield(_s252)
_ok252 = rules.apply_carry(_y252, _s252, why="laya-rule 1阶占位让位") if _y252 else False
_v252, _w252 = rules._transfer_victim(_s252)
RuleEngine.ECO_TAKEOVER, RuleEngine.MODEL_DECIDES = _md252
rules.carry_by_laya = False
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R252.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
print(f"[1阶占位让位] 接班={_y252}（期望 沈梦溪）换主C={_ok252}；收尾卖={_v252}（期望 李元芳）{_w252}")
if _y252 != "沈梦溪" or not _ok252 or _v252 != "李元芳":
    failures.append(f"1阶占位让位：{_y252} / {_ok252} / {_v252}")

# 253 胜之不武（2026-10-12：可以每回合用一张，性价比高）：血量满也用，同一回合只用一张
rules.reset_match()
rules._round_seen = 5
rules._field = {"百里守约": (390, 502), "苏烈": (529, 372)}
rules._carry_piece, rules._carry = "百里守约", (390, 502)
_s253 = GameState(phase="prepare", round_no=5, level=3, gold=4, hp=60, population=2, population_cap=6, population_valid=True)
_s253.hand_slots = [{"name": "胜之不武", "x": 500 + 90 * i, "y": 663} for i in range(3)]
_a253 = rules.decide_use_cards(_s253)
if _a253:
    rules.note_executed(_a253)
_b253 = rules.decide_use_cards(_s253)
rules.reset_match()
print(f"[胜之不武] 第一张 → {_a253 and _a253.get('piece')}（期望 胜之不武）；同回合第二张 → {_b253 and _b253.get('piece')}（期望 不是胜之不武）")
if not _a253 or _a253.get("piece") != "胜之不武" or (_b253 and _b253.get("piece") == "胜之不武"):
    failures.append(f"胜之不武：{_a253} / {_b253}")

# 254 共战天赋（2026-10-12：崛起 / 双生突袭点名马可波罗、百里守约 —— 其中一个必须是主C，且必须同时上场，不能卖）
rules.reset_match()
rules._talents_taken = {"崛起"}
_s254 = GameState(phase="prepare", round_no=6, level=4, gold=4, population=5, population_cap=6, population_valid=True)
rules._field = {"李元芳": (300, 569), "马可波罗": (839, 502), "百里守约": (955, 502), "苏烈": (529, 372)}
rules._carry_piece, rules._carry = "李元芳", (300, 569)
_d254 = sorted(rules.duo_members())
_y254 = rules.placeholder_yield(_s254)
_acts254 = rules.decide_all.__func__  # noqa
_f254 = [a for a in [{"type": "sell_unit", "piece": "马可波罗", "src": (839, 502)}]
         if not (a["piece"] in rules.duo_members())]
rules.reset_match()
print(f"[共战天赋] 点名={_d254}（期望 马可波罗、百里守约）；接任主C={_y254}（期望 百里守约）；卖马可波罗被拦={not _f254}")
if _d254 != ["百里守约", "马可波罗"] or _y254 != "百里守约" or _f254:
    failures.append(f"共战天赋：{_d254} / {_y254} / {_f254}")

# 255 第 587 局（审查 10-12 A3）：指定守约路线，局面网络选了安琪拉当主C → 路线跟到以安琪拉为主C 的阵容（不在指定名单里也找）
from agent.decision import routes as _R255
_R255.CARRY_TIERS = (2, 3)
RuleEngine.ROUTE_ENABLED = True
_md255 = (RuleEngine.ECO_TAKEOVER, RuleEngine.MODEL_DECIDES)
RuleEngine.ECO_TAKEOVER = RuleEngine.MODEL_DECIDES = True
rules.reset_match()
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json", lord="阿离")
rules.forced_route, rules.forced_backups = "50720714303640586", ("81682962224371742",)
rules.__dict__.pop("_forced_orig", None)
rules.carry_by_laya = True
_s255 = GameState(phase="prepare", round_no=6, level=3, gold=4, population=4, population_cap=6, population_valid=True)
rules._field = {"沈梦溪": (955, 502), "李元芳": (839, 502), "苏烈": (783, 438), "亚连": (617, 496)}   # 10-20：安琪拉不能当主C → 改用沈梦溪
rules.resolve_comp(_s255)
rules.apply_carry("沈梦溪", _s255, why="Laya 0.7")
rules.resolve_comp(_s255)
_r255 = ((rules._active_comp or {}).get("carry"), rules._is_carry("百里守约"))
RuleEngine.ECO_TAKEOVER, RuleEngine.MODEL_DECIDES = _md255
rules.carry_by_laya = False
rules.reset_match()
rules.forced_route, rules.forced_backups = "", ()
rules.__dict__.pop("_forced_orig", None)
_R255.CARRY_TIERS = None
RuleEngine.ROUTE_ENABLED = False
print(f"[路线跟主C] 路线主C={_r255[0]}（期望 沈梦溪）；守约还算主C={_r255[1]}（期望 False）")
if _r255 != ("沈梦溪", False):
    failures.append(f"路线跟主C：{_r255}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 256（2026-10-13 理想打法：多个路线的配套核心卡，见到就买，只买一张）：
#   首选守约 + 备选沈梦溪 / 虞姬：没主C 时三条路线的核心卡（含各自主C）见到就买；已有一张的不再买；禁大河时虞姬路线的少司缘不买
from agent.perception.state import ShopSlot as _SS256
from agent.decision import interim_rules as _IR256
_flex_was = _IR256.RULES["flex_buy"]["on"]
_IR256.RULES["flex_buy"]["on"] = True          # 测的是机制本身（开关关着时线上不走这条，交给 R27）
rules.reset_match()
rules.forced_route, rules.forced_backups = "50720714303640586", ("81682962224371742", "user_compound_river_totem")
rules.__dict__.pop("_forced_orig", None)
_s256 = GameState(phase="prepare", round_no=3, level=2, gold=6, population=2, population_cap=5, population_valid=True, shop_open=True)
_s256.shop = [_SS256(index=0, piece="牛魔", cost=3), _SS256(index=1, piece="少司缘", cost=3)]
_a256 = rules._flex_buy(_s256, 6)
_s256.hand_slots = [{"name": "少司缘", "x": 500, "y": 660}]
_b256 = rules._flex_buy(_s256, 6)
_s256.hand_slots, _s256.banned_faction = [], "大河流域"
_c256 = rules._flex_buy(_s256, 6)
_s256.banned_faction = ""
_s256.shop = [_SS256(index=0, piece="明世隐", cost=3)]
_d256 = rules._flex_buy(_s256, 6)
# 主C 定了（2 阶守约）：配套明世隐已有一张也照买（定好最终主C 之后大量买配套英雄，不限一张）
rules._field = {"百里守约": (917, 570), "明世隐": (300, 569)}
rules._carry_piece = "百里守约"
_s256.hand_slots = []
_e256 = rules._flex_buy(_s256, 6)
from agent.decision import interim_rules as _IR256
if _IR256.on("flex_buy_settled") and (_e256 or {}).get("piece") != "明世隐":
    failures.append(f"主C 定了应不限张数买配套：{_e256}")
if not _IR256.on("flex_buy_settled") and _e256 is not None:      # 2026-10-15 第 632 局后主C 定了的买配套交给 Laya
    failures.append(f"flex_buy_settled 关了，规则不该再买配套：{_e256}")
rules.reset_match(); rules.forced_route, rules.forced_backups = "", ()
print(f"[多路线核心卡] 少司缘 → {(_a256 or {}).get('piece')}（期望 少司缘）；已有一张 → {_b256}（期望 None）；"
      f"禁大河 → {_c256}（期望 None）；明世隐 → {(_d256 or {}).get('piece')}（期望 明世隐）")
_IR256.RULES["flex_buy"]["on"] = _flex_was
if (_a256 or {}).get("piece") != "少司缘" or _b256 or _c256 or (_d256 or {}).get("piece") != "明世隐":
    failures.append(f"多路线核心卡见到就买不对：{_a256} / {_b256} / {_c256} / {_d256}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 257（2026-10-13 第 614 局：主C 被给了反伤甲）：主C 马可波罗有空位，反伤重甲（对马可波罗 2.8 / 对项羽 8.0）
# → 给主坦项羽，不给主C；场上没有坦克时留着（不走"主C 有空位就给"的老兜底）
_ht200.USE_HERO_ITEM_PREF = True
rules.reset_match()
rules._field = {"马可波罗": (839, 502), "项羽": (672, 430), "干将莫邪": (917, 570)}
rules._cell_owner = {v: k for k, v in rules._field.items()}
rules._carry_piece, rules._carry = "马可波罗", (839, 502)
rules._unit_lv = {"马可波罗": 27, "项羽": 17, "干将莫邪": 12}
_s257 = GameState(phase="prepare", round_no=9, gold=6)
_a257 = rules._equip_target(_s257, "反伤重甲")
del rules._field["项羽"]
rules._cell_owner = {v: k for k, v in rules._field.items()}
_b257 = rules._equip_target(_s257, "反伤重甲")
_r257 = (_a257[0] if _a257 else None, _b257[0] if _b257 else None)
print(f"[防御装不给主C] 有项羽 / 没坦克 → {_r257}（期望 项羽, 不是马可波罗）")
if _r257[0] != "项羽" or _r257[1] == "马可波罗":
    failures.append(f"防御装不给主C 不对：{_r257}")
_ht200.USE_HERO_ITEM_PREF = False
rules.reset_match()
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 258（2026-10-13 审计，608/610/613 局）：临时主C 干将（法术）做四选一时按路线主C 虞姬（物理）选；路线主C 自己照常
from agent.decision import routes as _R258
RuleEngine.ROUTE_ENABLED = True
rules.reset_match()
rules._active_comp, rules._route_locked, rules._route_key = _R258.by_key("user_compound_river_totem"), True, "user_compound_river_totem"
rules._field = {"干将莫邪": (917, 570)}
rules._carry_piece = "干将莫邪"
_a258 = rules.equip_role_for("干将莫邪")
rules._field["虞姬"] = (300, 569); rules._carry_piece = "虞姬"
_b258 = rules.equip_role_for("虞姬")
rules.reset_match()
RuleEngine.ROUTE_ENABLED = False
print(f"[临时主C 四选一跟路线主C] 干将 / 虞姬 → {_a258} / {_b258}（期望 carry:虞姬 / carry:虞姬）")
if (_a258, _b258) != ("carry:虞姬", "carry:虞姬"):
    failures.append(f"临时主C 四选一不对：{_a258} / {_b258}")
# 场景 259（2026-10-14 伞海月节奏）：阶段 raise / farm / rush / convert（清场 → 遗志打干将）
from agent.decision import routes as _R259


def _mk259(hand_names=(), pop=3, cap=6):
    st = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=5, gold=0, population=pop, population_cap=cap)
    st.hand_slots = [{"name": n, "x": 400 + i * 90, "y": 660} for i, n in enumerate(hand_names)]
    return st


rules.reset_match()
rules._active_comp, rules._route_locked, rules._route_key = _R259.by_key("user_haiyue_umbrella") or {"key": "user_haiyue_umbrella", "carry": "海月"}, True, "user_haiyue_umbrella"
rules._field = {"干将莫邪": (917, 570), "牛魔": (555, 569)}
_s259 = _mk259(("干将莫邪",), pop=2, cap=5); _s259.level = 2
_p1 = rules.haiyue_phase(_s259)
rules._ever_awakened.add("干将莫邪")
_up1 = rules.decide_upgrade(_s259)
_s259.level = 3; _s259.level_cost = 8; _s259.gold = 20; _s259.level_btn = (64, 640)
_p2 = rules.haiyue_phase(_s259); _hold = rules.haiyue_hold_level(_s259)
_s259b = _mk259(("遗志",), pop=2, cap=6); _s259b.level = 3; _s259b.level_cost = 8; _s259b.gold = 20; _s259b.level_btn = (64, 640)
_p3 = rules.haiyue_phase(_s259b); _a3 = rules.decide_haiyue(_s259b)
rules._field["海月"] = (433, 569)
_s259c = _mk259(("遗志",), pop=3, cap=6); _s259c.level = 5
_a4 = rules.decide_haiyue(_s259c)
del rules._field["牛魔"]
_a5 = rules.decide_haiyue(_s259c)
rules.reset_match()
print(f"[伞海月] 阶段 {_p1}/{_p2}(hold={_hold})/{_p3}，觉醒干将 2 级合成={_up1 and _up1.get('type')}，有遗志 3 级={_a3 and _a3.get('type')}，"
      f"海月到手 → {_a4 and (_a4.get('type'), _a4.get('piece'))} → {_a5 and (_a5.get('type'), _a5.get('piece'), _a5.get('target'))}")
if (_p1, _p2, _hold, _p3) != ("raise", "farm", True, "rush") or _up1 is not None or (_a3 or {}).get("type") != "buy_xp"         or (_a4 or {}).get("piece") != "牛魔" or (_a5 or {}).get("target") != "干将莫邪":
    failures.append("伞海月节奏不对")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 260（2026-10-15 第 631 局 R6）：主C 已定、满员、场上有闲人，但手里等着的蒙恬已被换人网络判过"不换" → 不再买要腾位的新配套
def _f260(benched):
    RuleEngine.ROUTE_ENABLED = True
    rules.reset_match()
    eng, rules.ENGINE_ENABLED = rules.ENGINE_ENABLED, False
    rules._active_comp, rules._route_locked, rules._route_key = _R202.by_key("user_compound_river_totem"), True, "user_compound_river_totem"
    rules._round_seen = 9
    rules._field = {"虞姬": (917, 570), "瑶": (672, 430), "云中君": (617, 372), "鬼谷子": (529, 372), "张良": (433, 569),
                    "苏烈": (839, 502), "大司命": (300, 569)}
    rules._carry_piece = "虞姬"
    rules._unit_lv = {"虞姬": 40, "苏烈": 3, "瑶": 8, "云中君": 9, "鬼谷子": 12, "张良": 10, "大司命": 6}
    rules._roll_rounds = {5, 6, 7, 8, 9}
    if benched:
        rules._swap_benched = {"刘邦": 8}
    st = GameState(phase="prepare", round_no=9, level=5, gold=13, population=7, population_cap=7, population_valid=True, shop_open=True)
    st.hand_slots = [{"name": "刘邦", "x": 500, "y": 660}]
    st.shop = [_SS210(index=0, piece="小乔", cost=3)]
    out = (rules.preempt_shop(st) or {}).get("piece")
    rules.ENGINE_ENABLED = eng
    rules.reset_match()
    RuleEngine.ROUTE_ENABLED = False
    return out
_a260, _b260 = _f260(False), _f260(True)
print(f"[等着的人被判不换] 没判过 → 买 {_a260}（期望 小乔）；判过不换 → {_b260}（期望 None）")
if _a260 != "小乔" or _b260 == "小乔":
    failures.append(f"等着的人被判不换仍买新配套：{_a260}/{_b260}")
if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
# 场景 261（2026-10-15 第 634 局 R13）：人口读数比账本少 2，按人数对账不能删主C（马可波罗 106 级一直在场，被删后换主C 题里没有他，
#   局面网络换成 30 级安琪拉，换绑枫叶舞时把马可波罗卖了）
rules.reset_match()
rules._field = {n: (300 + i * 110, 569) for i, n in enumerate(["马可波罗", "米莱狄", "安琪拉", "狂铁", "朵莉亚", "雅典娜", "亚连"])}
rules._carry_piece = "马可波罗"
def _s261():
    st = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=13, gold=5, population=5, population_cap=7, level=6)
    st.vision_units = {f"{p[0]},{p[1]}": "hero" for n, p in rules._field.items() if n not in ("马可波罗", "朵莉亚")}
    return st
rules.reconcile(_s261()); rules.reconcile(_s261())
_k261 = "马可波罗" in rules._field
print(f"[对账不删主C] 人口 5 < 账本 7 连读两帧 → 主C 还在账本：{_k261}（期望 True），账本 {len(rules._field)} 条")
if not _k261:
    failures.append("按人数对账删掉了主C")
rules.reset_match()

# ---- 场景 262（第 636 局第 12 回合）：商店开着那帧能量 24 读成 23，升级费 24 → 用上一帧读数照升；
#      真升不起时升级预算按"下回合升级"留钱，不把买牌全扣住 ----
rules.reset_match(); rules._round_seen = 12
_s262a = GameState(phase="prepare", round_no=12, gold=24, level=5, level_cost=24, shop_open=False)
rules._note_gold(_s262a)
rules.note_executed({"type": "toggle_shop", "intent": "open"})
_s262b = GameState(phase="prepare", round_no=12, gold=23, level=5, level_cost=24, shop_open=True)
rules._note_gold(_s262b)
_e262 = rules.eff_gold(_s262b)
rules.note_executed({"type": "buy_piece", "piece": "铠"})
_s262c = GameState(phase="prepare", round_no=12, gold=23, level=5, level_cost=24, shop_open=True)
rules._note_gold(_s262c)
_e262c = rules.eff_gold(_s262c)
print(f"[能量单帧误读] 开店前 24、开店后读 23 → 升级用 {_e262}（期望 24）；买过牌后 → {_e262c}（期望 23）")
if _e262 != 24 or _e262c != 23:
    failures.append(f"升级能量没按上一帧纠正 / 花过钱还沿用：{_e262} {_e262c}")
_RE = RuleEngine
_old = getattr(rules, "eco_says")
rules.eco_says = lambda st: "levelup"
_s262d = GameState(phase="prepare", round_no=12, gold=20, level=5, level_cost=24, shop_open=True)
rules._note_gold(_s262d)
_b262 = rules.level_budget(_s262d)
rules.eco_says = _old
print(f"[升不起不全留] 能量 20 < 升级费 24 → 留 {_b262}（期望 < 20）")
if _b262 >= 20:
    failures.append(f"经济网络说升级但升不起，升级预算把能量全留住：{_b262}")

# ---- 场景 263（第 636 局第 5 回合）：无法拆卸的盛华只给阵容里的坦克；没到就留着，不给 1 阶过渡件苏烈 ----
rules.reset_match(); rules._round_seen = 5; rules._carry_piece = "沈梦溪"
rules._field = {"沈梦溪": (300, 569), "苏烈": (617, 372), "牛魔": (529, 372)}
rules._unit_lv.update({"沈梦溪": 6, "苏烈": 3, "牛魔": 3})
_s263 = GameState(phase="prepare", round_no=5, gold=2, level=3, population=3, population_cap=6, population_valid=True)
_rt = rules._route_team
rules._route_team = lambda: {"沈梦溪", "盾山", "铠", "明世隐"}
r263 = rules._reviewed_target(_s263, "盛华", rules.usage_reviewed.get("盛华", {}))
rules._field["盾山"] = (556, 430); rules._unit_lv["盾山"] = 3
r263b = rules._reviewed_target(_s263, "盛华", rules.usage_reviewed.get("盛华", {}))
rules._route_team = _rt
print(f"[无法拆卸只给阵容坦克] 盾山没到 → {r263}（期望 skip）；盾山到了 → {r263b}（期望 盾山）")
if r263 != "skip" or not (isinstance(r263b, tuple) and r263b[0] == "盾山"):
    failures.append(f"盛华 给了阵容外的人 / 阵容坦克到了不给：{r263} {r263b}")
rules.reset_match()

# ---- 场景 264（2026-10-19，第 751 局 R9）：当前场上人数统一口径 rules.pop_now ----
# a) 觉醒回手 + 商店开着人口读不到（感知层沿用旧读数 6/6）、账本 5 → 判不满员（原来按旧读数 6/6 判满，换人题卖了太乙）
rules.reset_match(); rules._round_seen = 9
_f264 = {"牛魔": (672, 430), "百里守约": (300, 569), "苏烈": (529, 372), "程咬金": (718, 371), "铠": (783, 438), "李元芳": (276, 501)}
rules._field = dict(_f264)
_s264v = GameState(phase="prepare", round_no=9, shop_open=False, population=6, population_cap=6, population_valid=True)
rules._remember_pop(_s264v); rules._pop_mark(_s264v)                     # 那一帧真读到 6/6、账本 6
_s264 = GameState(phase="prepare", round_no=9, shop_open=True, population=6, population_cap=6, population_valid=False)
_s264.hand_slots = [{"name": "百里守约", "x": 791, "y": 663, "awakened": True}]
rules.sync_awakened(_s264)                                                 # 手牌认出觉醒守约 → 账本 5
_p264a = rules.pop_now(_s264)
print(f"[觉醒回手+旧读数 6/6] pop_now={_p264a}（期望 5/6、不满员）")
if _p264a[:2] != (5, 6) or rules.pop_full(_s264):
    failures.append(f"觉醒回手后读不到人口，仍按旧读数判满员：{_p264a}")
# b) 读数为主：本帧真读到 4/6（账本 5）→ 信读数；读数超上限要钳住
_s264r = GameState(phase="prepare", round_no=9, shop_open=False, population=4, population_cap=6, population_valid=True)
_s264x = GameState(phase="prepare", round_no=9, shop_open=False, population=8, population_cap=6, population_valid=True)
_p264b, _p264x = rules.pop_now(_s264r), rules.pop_now(_s264x)
print(f"[读数为主] 读到 4/6 → {_p264b}（期望 4/6）；读到 8/6 → {_p264x}（期望钳到 6）")
if _p264b[:2] != (4, 6) or _p264x[0] != 6:
    failures.append(f"人口读数没压过账本 / 没钳在上限内：{_p264b} {_p264x}")
# c) 读不到时卖人立刻生效：锚 6/6，之后卖掉 1 人 → 5/6
rules.reset_match(); rules._round_seen = 9; rules._field = dict(_f264)
rules._pop_mark(_s264v)
rules._field.pop("李元芳")
_p264c = rules.pop_now(_s264)
print(f"[卖人后读不到人口] pop_now={_p264c}（期望 5/6）")
if _p264c[:2] != (5, 6):
    failures.append(f"卖人后读不到人口，场上人数没按账本变化修正：{_p264c}")
# d) 觉醒进度牌打成功、按徽章这一张凑满 → 预期回手：读不到时按已离场算，别的英雄暂不上场、换人题不出
rules.reset_match(); rules._round_seen = 9; rules._field = dict(_f264); rules._field.pop("李元芳")
rules._awaken_seen["百里守约"] = 2
_s264d = GameState(phase="prepare", round_no=9, shop_open=False, population=5, population_cap=6, population_valid=True, gold=5)
rules._pop_mark(_s264d)
rules.confirm_drag({"type": "drag_card", "piece": "弦音", "kind": "效果牌", "target": "百里守约", "dst": (300, 569)}, True)
_s264e = GameState(phase="prepare", round_no=9, shop_open=True, population=5, population_cap=6, population_valid=False)
_p264d = rules.pop_now(_s264e)
_s264d.hand_slots = [{"name": "太乙真人", "x": 721, "y": 663}]
_a264 = rules.decide_deploy(_s264d)
print(f"[弦音凑满觉醒] 预期回手={rules.awaken_expected(_s264d)} 读不到时 pop_now={_p264d}（期望 4/6）；太乙先上场吗={_a264}（期望 None）")
if rules.awaken_expected(_s264d) != ["百里守约"] or _p264d[:2] != (4, 6) or _a264:
    failures.append(f"觉醒进度牌凑满后没按'马上回手'处理：{rules.awaken_expected(_s264d)} {_p264d} {_a264}")
rules.reset_match()
# ---- 场景 265（2026-10-19：临时主C 不先穿装备）：临时主C 期间装备留在手里 ----
_IRT.RULES['equip_hold_for_real_carry']['on'] = True
rules.reset_match()
rules._carry_piece = "曜"
rules._field = {"曜": (529, 372), "苏烈": (617, 372)}
_st265 = GameState(phase="prepare", round_no=3, gold=5, population=2, population_cap=5, population_valid=True,
                   hand_slots=[{"name": "大棒", "x": 721, "y": 663}])
_t265 = rules._equip_target(_st265, "大棒")
print(f"[临时主C 不穿装备] 曜(1 阶) 当主C 时装备目标={_t265}（期望 None）")
if _t265 is not None:
    failures.append(f"临时主C 期间装备没留着: {_t265}")
# 第 850 局 R1：主C 还空着、场上只有 1 阶的孙尚香 / 李元芳 → 圣者法典先留着（原来"主力不在场上，装备改给 孙尚香"）
rules.reset_match()
rules._carry_piece = None
rules._field = {"孙尚香": (917, 570), "李元芳": (1027, 572)}
_st265b = GameState(phase="prepare", round_no=1, gold=0, population=2, population_cap=5, population_valid=True,
                    hand_slots=[{"name": "圣者法典", "x": 720, "y": 663}])
_t265b = rules._equip_target(_st265b, "圣者法典")
print(f"[主C 未定不穿装备] 只有 1 阶孙尚香 / 李元芳 时装备目标={_t265b}（期望 None）")
if _t265b is not None:
    failures.append(f"主C 未定、只有 1 阶单位时装备没留着: {_t265b}")
rules.reset_match()
_IRT.RULES['equip_hold_for_real_carry']['on'] = False

# ---- 场景 266（第 853 局，10-21）：天赋按最终可摆阵容计数 / 非主C 装备辅助性 / 韩信攻速型 ----
from agent.decision import talent_conditions as _TC266, hero_traits as _HT266
_b266 = {"final": {"百里守约", "铠", "盾山", "李元芳", "苏烈", "钟馗", "太乙真人", "明世隐"},
         "must": {"百里守约", "铠", "明世隐", "盾山", "李元芳", "太乙真人"},
         "pool": {"百里守约", "铠", "盾山", "李元芳", "苏烈", "钟馗", "太乙真人", "明世隐", "程咬金"}, "cap": 7}
_g266 = _TC266.check("同心协力", _b266["pool"], "百里守约", "百里守约", True, board=_b266)[0]
_h266 = {"final": {"韩信", "张良", "云中君", "虞姬", "瑶", "鬼谷子", "太乙真人"}, "must": {"韩信", "太乙真人"}, "pool": set(), "cap": 7}
_gh266 = _TC266.check("小小的助力", _h266["final"], "韩信", "韩信", True, board=_h266)[0]
_gf266 = _TC266.frozen("f9z").check("同心协力", _b266["pool"], "百里守约", "百里守约")[0]
print(f"[最终阵容计数] 守约线同心协力={_g266}（期望 C）韩信线小小的助力={_gh266}（期望 B）f9z 冻结表={_gf266}（期望 B，旧题面不变）")
if _g266 != "C" or _gh266 != "B" or _gf266 != "B":
    failures.append(f"天赋最终阵容计数不对：{_g266} {_gh266} {_gf266}")
_u266 = [(_n, _HT266.utility_rank(_n)) for _n in ("御灵之纱", "生命魔典", "炽热支配者", "贤者之书")]
print(f"[非主C 辅助性] {_u266}（期望 0 / 1 / 2 / 3）")
if [x[1] for x in _u266] != [0, 1, 2, 3]:
    failures.append(f"辅助性排序不对：{_u266}")
if not _HT266.attack_speed_carry_live("韩信") or _HT266.attack_speed_carry("韩信"):
    failures.append("韩信攻速型：线上应算、题面旧口径不算")

if failures:
    print("FAIL:"); [print(" -", f) for f in failures]; raise SystemExit(1)
print("冒烟测试全部通过（含场景 175~264）")

# ---- 准备阶段末尾不触发觉醒（2026-10-20 第 843 局 R12：妙音凑满守约觉醒，守约回手、整场战斗不在场） ----
rules.reset_match()
rules._field = {"百里守约": (917, 570), "苏烈": (529, 372)}
rules._awaken_seen = {"百里守约": 2, "苏烈": 0}
_wa = rules.would_awaken({"type": "drag_card", "piece": "妙音", "target": "百里守约", "kind": "效果牌", "src": (425, 663), "dst": (917, 570)})
_wb = rules.would_awaken({"type": "drag_card", "piece": "妙音", "target": "苏烈", "kind": "效果牌", "src": (425, 663), "dst": (529, 372)})
print(f"[觉醒预判] 妙音给进度 2 的守约 → {_wa!r}（期望 百里守约）；给进度 0 的苏烈 → {_wb!r}（期望 空）")
if _wa != "百里守约" or _wb:
    print("FAIL: would_awaken 预判不对"); sys.exit(1)

# ---- 棋手配置架构（2026-10-20 晚） ----
from agent.decision import lord_config as _LCS
rules.load_lineups(PROJECT_DIR / "data" / "lineups.json", lord="阿离")     # 前面场景把 comps 换成了几套测试阵容
_routeS = RuleEngine.ROUTE_ENABLED
# 开局打法三条临时规则 10-21 关了（回到原节奏）：代码留着，这里临时打开测它们还能用
from agent.decision import interim_rules as _IRS
_onS = {k: _IRS.RULES[k]["on"] for k in ("route_lock_opening", "maple_temp_host", "taiyi_shield_early")}
for _k in _onS:
    _IRS.RULES[_k]["on"] = True
RuleEngine.ROUTE_ENABLED = True                   # 前面场景关掉了路线
_cfgS = _LCS.load()
_fS = []
# S1 开局定路线：没禁 → 河洛三C 组；禁河洛 → 亚连
_o1, _o2 = _cfgS.opening_choice(""), _cfgS.opening_choice("河洛")
if not (_o1 and _o1["kind"] == "group" and "百里守约" in _o1["members"]):
    _fS.append(f"开局（没禁）应锁河洛三C 组：{_o1}")
if not (_o2 and _o2["name"] == "亚连"):
    _fS.append(f"开局（禁河洛）应锁亚连：{_o2}")
# S2 组内谁先出用谁：锁组代表阵容，沈梦溪先在场 → 路线换到沈梦溪的阵容
rules.reset_match()
rules.set_route(_o1["key"], 1, why="冒烟")
rules._route_group = _o1["members"]
rules._field = {"沈梦溪": (1027, 572), "孙尚香": (917, 570)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules.group_sync(GameState(phase="prepare", round_no=3, gold=5))
if rules.route_carry() != "沈梦溪":
    _fS.append(f"组员沈梦溪先到，路线主C 应是沈梦溪：{rules.route_carry()}")
# S3 主C 没觉醒 → 经济还要刷他；太乙早到贴盾山
if rules.carry_awaken_need() != ["沈梦溪"]:
    _fS.append(f"主C 没觉醒应算缺：{rules.carry_awaken_need()}")
rules._field.update({"盾山": (529, 372), "太乙真人": (617, 372), "明世隐": (700, 500)})
# 10-21 晚：明世隐只登场 1 次 → 太乙贴盾山
if rules.buddy_anchor("太乙真人") != "盾山":
    _fS.append(f"河洛线主C 没觉醒、盾山在场 → 太乙贴盾山：{rules.buddy_anchor('太乙真人')}")
rules._ever_awakened.add("沈梦溪")
if rules.buddy_anchor("太乙真人") != "盾山":       # 10-21：按回合分，前期贴铠 / 盾山（复生再触发牺牲），不看主C 觉醒没
    _fS.append(f"第 12 回合前主C 觉醒后太乙仍贴盾山：{rules.buddy_anchor('太乙真人')}")
rules._field["铠"] = (705, 372)
if rules.buddy_anchor("太乙真人") != "盾山":     # 10-21 晚：明世隐只登场 1 次 → 贴盾山（不贴铠）
    _fS.append(f"明世隐登场 1 次 → 太乙贴盾山：{rules.buddy_anchor('太乙真人')}")
rules._field.pop("铠")
rules._awaken_seen["明世隐"] = 1                  # 合成过 1 张 = 登场 2 次 → 贴主C
if rules.buddy_anchor("太乙真人") != rules._carry_piece:
    _fS.append(f"明世隐登场 2 次 → 太乙贴主C：{rules.buddy_anchor('太乙真人')}")
rules._awaken_seen.pop("明世隐", None)
_rs0 = rules._round_seen
rules._round_seen = 12
if rules.buddy_anchor("太乙真人") != rules._carry_piece:
    _fS.append("第 12 回合起太乙应回到贴主C")
rules._round_seen = _rs0
# S4 枫叶舞临时宿主：路线已锁、没主C → 绑非核心的输出，不绑程咬金（核心件）
rules.reset_match()
rules.set_route(_cfgS.lineup_key("百里守约"), 1, why="冒烟")
rules._field = {"程咬金": (617, 372), "孙尚香": (917, 570)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
rules._carry_piece = ""
_h = rules._skill_target()
_hn = _h[0] if isinstance(_h, tuple) else _h
if _hn == "程咬金":
    _fS.append(f"枫叶舞临时宿主不该是核心件程咬金：{_h}")
# S5 4 阶路线（韩信）临时主C：干将在场 → 枫叶舞挂干将
rules.reset_match()
if rules.set_route(_cfgS.lineup_key("韩信"), 1, why="冒烟"):
    rules._field = {"干将莫邪": (1027, 572), "孙尚香": (917, 570)}
    rules._cell_owner = {p: n for n, p in rules._field.items()}
    _h5 = rules._skill_target()
    _h5n = _h5[0] if isinstance(_h5, tuple) else _h5
    if _h5n != "干将莫邪":
        _fS.append(f"4 阶路线临时主C 应挂干将：{_h5}")
else:
    _fS.append("韩信路线锁不上（配置 / 合法性）")
# S6 第 1 回合还没锁路线、场上只有开局两张 → 枫叶舞照样挂（不挂核心件程咬金，挂孙尚香）
rules.reset_match()
rules._field = {"程咬金": (617, 372), "孙尚香": (917, 570)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
_h6 = rules._skill_target()
if _h6 != "孙尚香":
    _fS.append(f"第 1 回合没锁路线时枫叶舞应挂非核心的孙尚香：{_h6}")
# S7 场上只有核心件 → 能杀人的也要挂（铠）；杀不了人的坦克（程咬金）不挂（枫叶舞击杀才解锁专属卡）
rules.reset_match()
rules._field = {"铠": (617, 372), "程咬金": (529, 372)}
rules._cell_owner = {p: n for n, p in rules._field.items()}
if rules._skill_target() != "铠":
    _fS.append(f"只剩核心件时能杀人的也要挂：{rules._skill_target()}")
rules.reset_match()
for _k, _v in _onS.items():
    _IRS.RULES[_k]["on"] = _v
RuleEngine.ROUTE_ENABLED = _routeS
print(f"[棋手配置架构] 开局={_o1 and _o1['name']} / 禁河洛={_o2 and _o2['name']}；问题 {_fS or '无'}")
if _fS:
    print("FAIL:"); [print(" -", f) for f in _fS]; sys.exit(1)

# ---- 棋手专属规则（data/lords/阿离/lord_rules.json，2026-10-22）----
# L1 河洛路线主C 满装 → 基础装备 / 辅助成品给铠（spare_equip_target）；限定远程的（日渊）、自然法杖不给近战铠；输出成品照旧
# L2 青莲剑谱：仍 never_pick，但不再"谁都不穿"（拿到了可以给其他近战穿，比如凯）
_fL = []
rules.reset_match()
rules.route_carry = lambda: "百里守约"
rules._carry_piece = "百里守约"
rules._field = {"百里守约": (529, 372), "铠": (617, 372), "狄仁杰": (672, 430)}
rules._worn = {"百里守约": ["凝渊", "破晓", "影牙"]}
rules._equip_seen = {"529,372": 3}
stL = GameState(shop_open=False, population_valid=True, phase="prepare", round_no=10, gold=0,
                population=3, population_cap=7, hp=60, level=5)
stL.unit_anchors = {"529,372": 3, "617,372": 1, "672,430": 1}
for _c, _want in (("圣者法典", "铠"), ("守护者之铠", "铠"), ("近卫荣耀", "铠"), ("青莲剑谱", "铠")):
    _t = rules._equip_target(stL, _c)
    if not _t or _t[0] != _want:
        _fL.append(f"河洛主C 满装：{_c} 应给 {_want}，实际 {_t}")
for _c in ("日渊", "自然法杖"):
    _t = rules._equip_target(stL, _c)
    if _t and _t[0] == "铠":
        _fL.append(f"{_c} 限定远程，不该给近战铠")
if not rules.never_pick("青莲剑谱") or "青莲剑谱" in rules.LORD_USELESS_ITEMS:
    _fL.append("青莲剑谱应仍在 never_pick、不在 LORD_USELESS_ITEMS")
rules.route_carry = lambda: "亚连"
rules._carry_piece = "亚连"
if rules.spare_equip_targets():
    _fL.append(f"日落海路线不该有 spare_equip_target：{rules.spare_equip_targets()}")
del rules.route_carry
rules.reset_match()
print(f"[棋手专属规则] 问题 {_fL or '无'}")
if _fL:
    print("FAIL:"); [print(" -", f) for f in _fL]; sys.exit(1)
