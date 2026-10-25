"""动作执行器：把决策指令翻译成控制器操作。

坐标基准 1280x720（MuMu ADB），与 extractors.py 同源校准。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..perception.extractors import (
    AUCTION_LOT_W,
    AUCTION_LOT_X,
    BTN_AUCTION_BID,
    BTN_EXIT_MATCH,
    BTN_HERO_CONFIRM,
    BTN_LEVEL_UP,
    BTN_LOCK,
    BTN_MATCH_CONFIRM,
    BTN_PLAY_AGAIN,
    BTN_READY,
    BTN_SHOP_TOGGLE,
    BTN_START_GAME,
    BTN_START_MATCH,
    BTN_REFRESH,
    BTN_SETTINGS_CLOSE,
    BTN_SETTLE_NOW,
    BTN_TALENT_REFRESH,
    HERO_CANDIDATES,
    SHOP_CARD_W,
    SHOP_CARD_X,
    TALENT_CARD_W,
    TALENT_CARD_X,
    TAP_SIDE_GROUND, BTN_LORD_PANEL_CLOSE,
)
from . import tap_gate

logger = logging.getLogger(__name__)

# 每种动作**执行时必须有哪些字段**。任一组齐全即可（外层 or，内层 and）。
# 为什么要有这张表：2026-09-23 实战发现 `sell_unit` 有一处调用给的是 x/y、
# 而执行器读的是 src —— 每次都 KeyError，一个英雄都没卖出去过，
# 而异常被主循环吞成一行 ERROR，看日志只当是"这次没卖成"。
# 光靠"执行器认不认这个 type"是查不出来的，必须连字段一起对。
ACTION_REQUIRED_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "drag_card": (("src", "dst"),),
    "move_unit": (("src", "dst"),),
    "sell_unit": (("src",), ("x", "y")),
    "sell_card": (("src",),),
    "buy_piece": (("slot",),),
    "card_pick": (("x", "y"),),
    "talent_pick": (("index",), ("x", "y")),
    "lord_boost_pick": (("index",), ("x", "y")),
    "auction_bid": (("lot",), ("x", "y")),
    "sequence": (("steps",),),
}


def validate_action(action: dict) -> str:
    """动作字段自检。返回空字符串 = 没问题，否则返回该报的错。"""
    atype = action.get("type")
    if not atype:
        return f"动作没有 type: {action}"
    groups = ACTION_REQUIRED_FIELDS.get(atype)
    if groups is None:
        return ""                    # 这种动作不需要额外字段
    for grp in groups:
        if all(k in action and action[k] is not None for k in grp):
            return ""
    want = " 或 ".join("+".join(g) for g in groups)
    return f"{atype} 缺字段（需要 {want}）：{action}"


SHOP_CARD_CLICK_Y = 210           # 商店卡面点击点（卡面中段）
AUCTION_LOT_CLICK_Y = 180         # 拍品卡面点击点（仅在感知层没给出实测坐标时兜底）
AUCTION_LOT_GRIP_DY = -60         # 从卡名框中心上移到卡面主体，避免点在文字/边框上
TALENT_CARD_CLICK_Y = 300         # 天赋卡点击点
# ===== 关模态：三条路都试过，全是坑，结论是「没把握就别点」 =====
#  1. 点固定"空白处" (850,400)：那是棋盘，点中棋子会弹它的详情面板 →
#     关模态变成开模态，死循环（2026-09-22 一局点了 39 次）。
#     而且那个详情面板左下角是「出售」，再盲点就是把英雄卖了。
#  2. 点"己方半场的空格"：`_field` 只是记账，实测比画面少（"账本2/画面3"），
#     所谓的空格上常常站着人，等于回到第 1 条。
#  3. 按 Android 返回键：**这个游戏的返回键是"打开设置面板"**，不是关模态
#     （2026-09-22 实测复现：弹出画面/声音/功能/网络设置页，
#      而且右下角就是「退出对局」）。比前两条更糟。
# 所以现在的做法：**只点 OCR positively 认出来的按钮**；认不出就只存证据帧、
# 什么都不做，等画面自己变。设置面板万一被弹出来，用 close_settings 点它的 ✕。
KEY_BACK = 4                      # 保留常量仅作记录：这个游戏里它是"开设置"，别用
# 2026-10-09 删掉了 TAP_DISMISS=(850,400)（点棋盘，一局 39 次）和 EQUIP_SLOTS（旧 equip 动作的估值装备栏），
# 以及旧 deploy / sell(bench_col) / equip / use_card 四个动作分支 —— 决策层早就不产出这些动作了（grep 确认）。



# 拖起手牌时屏幕左侧出现的「出售」区（右侧也有一块，但紧挨能量圆盘，容易拖偏）。
# 截图实测：左侧出售区 x 0~74、y 464~638，取正中。
SELL_ZONE_LEFT = (37, 551)

# 最下面一行格子（y≈569）的中心**会被手牌区盖住**：手牌多了整排铺到最左边，
# 手牌区上沿就在 y≈568。落点压在手牌上 = 取消拖拽 —— 2026-09-23 孙膑往 (300,569) 合成
# 两次都"没生效"（合成前核对已确认那格就是孙膑），红点正好落在枫叶舞技能卡的上沿。
# 往上挪到 y=545：仍在同一格里（离 569 近、离上一行 501 远），但在手牌区上方。
BOARD_DROP_MAX_Y = 545
HAND_DRAG_MIN_Y = 600     # 起点在这条线以下 = 从手牌拖出
HAND_LIFT_DY = 140        # 先竖直往上提这么多（提出手牌区），再平移到目标
HAND_LIFT_STEPS = 8       # 竖直段占 24 步里的几步


def board_drop(dst):
    x, y = dst
    return (x, min(y, BOARD_DROP_MAX_Y))

class ActionExecutor:
    def __init__(self, controller, tap_delay: float = 0.3):
        """
        controller: maafw Controller 对象（AdbController 等）
        tap_delay: 每次操作后的等待秒数，给游戏 UI 响应时间
        """
        self.controller = controller
        self.tap_delay = tap_delay

    def _dismiss_enlarged(self, phase: str = "", allow_skip: bool = False) -> None:
        """关掉"选完卡后弹出的放大详情"。**必须多点几次，不能一次就走**。

        2026-09-22 第十一局：天赋三选一选完（选对了）之后这个面板挂了 3 分钟，
        直到被下一个拍卖界面盖掉才脱身 —— 它**不会自己关**。
        原实现是 0.6s 后点一次空白，多半是**点早了**：放大动画还没出来，
        那一下落在空处，面板随后才弹出来，于是再也没人去关它。
        现在按 0.6 / 1.2 / 2.0s 各点一次。点的是 TAP_SIDE_GROUND(1120,500)（经界面 gate），
        实测是空景，不会碰到棋子（**绝不能改成点棋盘中间**，关模态的已知坑）。
        """
        if phase == "auction":
            # **拍卖界面上绝不能点空地**：TAP_SIDE_GROUND 是按**棋盘画面**定的点，
            # 拍卖时那儿是舞台背景 —— 点下去会**取消已选中的拍品**，
            # 而"出价"按钮只在拍品被选中时才显示，于是整场拍卖一次都出不了价。
            # 2026-09-22 发现一直在点这个位置，就是这三下。
            logger.info("拍卖界面不点空地（会取消选中的拍品），跳过关闭放大详情")
            return
        # **先看一眼是不是已经回到正常画面**：N 选 1 选完多半没有放大详情，
        # 原来不管有没有都固定点三下、等 2 秒（2026-09-23 第八局：选一张牌花 4 秒）。
        # `ui_settled` 由主循环注入（它有 OCR）；没注入时照旧点三下。
        check = getattr(self, "ui_settled", None) if allow_skip else None
        if check is not None:
            deadline = time.time() + 0.8
            while time.time() < deadline:
                time.sleep(0.2)
                try:
                    if check():
                        logger.info("选完已经回到正常画面，不用点空地")
                        return
                except Exception:
                    break
        for wait in (0.6, 0.6, 0.8):
            time.sleep(wait)
            # 点之前问 screen_cls 现在是什么界面（2026-10-09）：选完多半已经回到准备画面（没东西可关），
            # 战斗 / N 选多还没选完时点这里有害 —— 只在放大详情 / 英雄面板上点
            if not self._tap_fixed("TAP_SIDE_GROUND", *TAP_SIDE_GROUND, why="选完卡关放大详情"):
                # 右侧空地在这个界面不让点（选完天赋的放大详情 screen_cls 判成 talent）→ 换场馆背景 (400,60)，
                # 它在天赋 / 放大详情 / 英雄面板上都被允许、历史上都关得掉（2026-10-10 第 524 局：三下全被拦，卡了 35 秒）
                self._tap_fixed("TAP_CLOSE_BIG_CARD", 400, 60, why="选完卡关放大详情（换场馆背景）")

    _act_src = None            # execute() 正在执行的动作类型，作为点击日志的默认来源

    def _src(self, src):
        return src or (f"act:{self._act_src}" if self._act_src else None)

    def _tap(self, x: int, y: int, src: str | None = None) -> None:
        self.controller.post_click(x, y).wait()
        tap_gate.log_touch("tap", [(x, y)], self._src(src))
        time.sleep(self.tap_delay)

    def _tap_fixed(self, name: str, x: int, y: int, image=None, verified: bool = False,
                   why: str = "", hold: float = 0.0) -> bool:
        """**写死坐标的兜底点击**一律走这里：先过界面 gate（tap_gate.SAFE_TAPS），不在允许的界面上就不点。
        返回是否真的点了。OCR 认出按钮文字再点的那类不要走这里。"""
        if not tap_gate.gate(name, (x, y), image=image, verified=verified, why=why):
            return False
        if hold:
            self._tap_hold(x, y, hold=hold, src=f"const:{name}")
        else:
            self._tap(x, y, src=f"const:{name}")
        return True

    def _tap_hold(self, x: int, y: int, hold: float = 0.12, src: str | None = None) -> None:
        """按下—停一会儿—抬起。**给那些不吃瞬时点击的控件用。**

        和 `_drag` 同一个道理：这个游戏对输入时长敏感（`_drag` 的注释里早就写了
        "按下后要稍作停顿，否则卡牌不进入拖拽态"）。2026-09-23 实测，
        右下角能量圆盘（开关商店）用 `post_click` 点下去**没有任何反应**——
        日志里"点过 2 次关店但商店仍开着"出现过很多次，证据帧证明坐标正落在圆盘正中，
        所以问题不在坐标，在**这一下太快了**。
        """
        self.controller.post_touch_down(x, y).wait()
        time.sleep(hold)
        self.controller.post_touch_up().wait()
        tap_gate.log_touch("tap_hold", [(x, y)], self._src(src))
        time.sleep(self.tap_delay)

    def dismiss_modal(self, occupied=()) -> None:
        """**已废弃，不要调用。** 保留只为挡住"再写一次盲点"的冲动。

        它做的是"点己方半场的一个空格来关模态"。三个问题：
          - 哪格是空的我们并不真知道（`_field` 只是记账，实测比画面少）
          - 点中棋子会弹出它的详情面板，而那个面板左下角是「出售」
          - 于是"关模态"变成"开模态"，甚至变成"卖英雄"
        现在的原则：只点 OCR 认出来的按钮；认不出就不点（见 live agent 的 unknown 分支）。
        """
        raise NotImplementedError(
            "dismiss_modal 已废弃：盲点棋盘会弹英雄详情面板（左下角是「出售」）。"
            "只点 OCR 认出来的按钮，认不出就等。")

    def execute(self, action: dict[str, Any]) -> bool:
        """执行单个决策动作，返回是否执行成功（尽力而为，不保证游戏侧生效）。"""
        problem = validate_action(action)
        if problem:
            logger.error("动作字段自检不过，拒绝执行：%s", problem)
            return

        atype = action.get("type")
        logger.info("执行动作: %s", action)
        outer_src = self._act_src
        if not outer_src:                 # sequence 里的子步骤沿用外层（"act:sequence"）
            self._act_src = atype
        try:
            if atype == "buy_piece":
                self._tap(SHOP_CARD_X[action["slot"]] + SHOP_CARD_W // 2,
                          SHOP_CARD_CLICK_Y)
            elif atype == "refresh_shop":
                self._tap(*BTN_REFRESH)
            elif atype == "lock_shop":
                self._tap(*BTN_LOCK)
            elif atype == "buy_xp":
                # 坐标由决策层给（感知层 OCR 定位到的"升级"二字中心）。
                # BTN_LEVEL_UP 只是兜底常量，**正常路径不该走到它**——
                # 详情面板打开时那个位置是「出售」，点下去会卖掉英雄。
                self._tap(action.get("x", BTN_LEVEL_UP[0]),
                          action.get("y", BTN_LEVEL_UP[1]))
            elif atype == "toggle_shop":
                # 右下角能量圆盘：点一次开商店，再点一次关。
                # 关掉才能看见人口数、也才能把英雄拖到棋盘上（面板挡着上半部分）。
                # 面板有开合动画，点完要多等一会儿——等不够的话下一帧还看得见商店，
                # 决策层会以为没关上又点一次，结果把它重新打开（2026-09-22 实战踩到）。
                # 用**按住再抬**，不要瞬时点击：实测圆盘不吃 post_click
                self._tap_hold(*BTN_SHOP_TOGGLE)
                time.sleep(0.9)
            elif atype == "ready":
                self._tap(*BTN_READY)
            elif atype == "hero_pick":
                # 优先用显式坐标（OCR 定位"阿离"），否则点候选位再确认
                # 选棋手界面时间充裕：点头像前、点确定前各等 1 秒（2026-10-08：确定点得太快；这里时间充裕，等 1s 再点）
                time.sleep(1.0)
                if "x" in action:
                    self._tap(action["x"], action.get("y", 655))
                else:
                    self._tap(*HERO_CANDIDATES[action.get("index", 1)])
                time.sleep(1.0)
                self._tap(*BTN_HERO_CONFIRM)
            elif atype == "talent_pick":
                self._tap(TALENT_CARD_X[action["index"]] + TALENT_CARD_W // 2,
                          TALENT_CARD_CLICK_Y)
                self._dismiss_enlarged(action.get("_phase", ""))
            elif atype == "lord_boost_pick":
                # 棋手特性三选一：用感知层实测的卡名坐标，别写死三列位置
                x, y = action.get("x", -1), action.get("y", -1)
                if x < 0:
                    x = TALENT_CARD_X[action["index"]] + TALENT_CARD_W // 2
                    y = TALENT_CARD_CLICK_Y
                self._tap(x, y)
                self._dismiss_enlarged(action.get("_phase", ""))
            elif atype == "card_pick":
                # 卡牌N选1：点实测坐标（卡名框中心上移到卡面），再关掉可能的放大详情。
                # **同一个弹窗点不掉时要换落点**：2026-09-23 实战第 9 回合，
                # 同一组候选（小小的助力/翩跹舞步·新生）被连点了 25 次都没关掉，
                # 账本里刷出 28 条「N选1」。一个固定偏移点不中就永远点不中，
                # 所以按重试次数在卡面上/卡名上/卡面下三个位置轮换。
                dy = (-60, 0, 60)[min(action.get("try", 0), 2)]
                self._tap_hold(action["x"], action["y"] + dy, hold=0.12)   # 这游戏对输入时长敏感
                if action.get("multi_more"):
                    time.sleep(0.5)          # N 选多还没选完：不点空地（会把多选界面收起来），等下一帧选下一张
                else:
                    self._dismiss_enlarged(action.get("_phase", ""), allow_skip=True)
            elif atype == "dismiss_detail":
                # 选完天赋后弹出的放大详情（左：选中的天赋，右：它送的牌）。点 (400,60) 场馆背景关掉 ——
                # TAP_BLANK(170,100) 在天赋界面上是左侧等级进度条，点了没用（2026-09-29 第 12 局卡 60 秒、同一张天赋点 7 次）。
                for _ in range(2):
                    self._tap_fixed("TAP_CLOSE_BIG_CARD", 400, 60, why="关天赋放大详情")
                    time.sleep(0.5)
            elif atype == "talent_refresh":
                self._tap(*BTN_TALENT_REFRESH)
            elif atype == "auction_bid":
                # 拍卖出价两步，且**必须先真的选中拍品**：不选中，"出价"按钮根本不显示
                # （2026-09-22 实战教训：之前直接点写死的 BTN_AUCTION_BID，那个位置
                #  当时压根没有按钮，等于每次出价都是空点，一局都没拍到）。
                #
                # 第一步用感知层实测的卡名坐标（lot.x/lot.y），不用写死的槽位；
                # 第二步由调用方（live agent）OCR 定位"出价"按钮后回调 bid_button，
                # 找不到按钮就说明第一步没选中——宁可不出价也不要空点。
                x = action.get("x", -1)
                y = action.get("y", -1)
                if x < 0 or y < 0:
                    x = AUCTION_LOT_X[action["lot"]] + AUCTION_LOT_W // 2
                    y = AUCTION_LOT_CLICK_Y          # 兜底：旧版坐标
                self._tap(x, y + AUCTION_LOT_GRIP_DY)
                time.sleep(0.6)
                btn = action.get("bid_button")
                if btn:
                    self._tap(*btn)
                else:
                    logger.warning("拍卖：未定位到'出价'按钮，只做了选中（本次不出价）")
                    return False
            elif atype == "open_talent":
                # 「天赋未领取」= 这次升级的天赋三选一还没选。**入口就是开商店那个能量圆盘**
                # （2026-09-23 确认），点开弹出的是天赋三选一而不是商店。
                # **不能点那个气泡本身**：它贴着站在棋盘上的阿离，点下去打开的是棋手详情面板，
                # 关掉又点、又开，一整回合什么都没做（实战第 9 回合）。
                self._tap_hold(*BTN_SHOP_TOGGLE)
                time.sleep(0.9)
            elif atype == "close_lord_panel":
                # 点面板外侧关掉棋手详情面板。**不能用 TAP_SIDE_GROUND**，
                # 那个点在面板内部（见 BTN_LORD_PANEL_CLOSE 注释）。
                self._tap_fixed("BTN_LORD_PANEL_CLOSE", *BTN_LORD_PANEL_CLOSE, why="关棋手详情面板")
            elif atype == "close_settings":
                # 设置面板左上角的 ✕。**不要点右下角**——那儿是「退出对局」。
                # 设置面板 screen_cls 只有 3 张训练样本，不可信 → 只影子记录（阶段是 OCR 认的「画面/声音」页签）
                self._tap_fixed("BTN_SETTINGS_CLOSE", *BTN_SETTINGS_CLOSE, verified=True, why="关设置面板")
            elif atype == "settle_now":
                # 坐标优先用调用方 OCR 实测的按钮中心，常量只作兜底
                self._tap(action.get("x", BTN_SETTLE_NOW[0]),
                          action.get("y", BTN_SETTLE_NOW[1]))
            elif atype == "exit_match":
                self._tap(action.get("x", BTN_EXIT_MATCH[0]),
                          action.get("y", BTN_EXIT_MATCH[1]))
            elif atype == "play_again":
                self._tap(*BTN_PLAY_AGAIN)
            elif atype == "start_match":
                self._tap(*BTN_START_MATCH)
            elif atype == "confirm_match":
                self._tap(*BTN_MATCH_CONFIRM)
            elif atype == "start_game":
                self._tap(*BTN_START_GAME)
            elif atype == "drag_card":
                # 手牌拖到目标点（英雄牌上场/升级、装备/效果牌给英雄）：src/dst 显式坐标
                # 往场上放**新英雄**（不是合成、不是装备/效果牌）才检查人口会不会超
                guards = []
                # ① **抓对了没有**（所有从手牌拖出的牌）：牌举在半空时读一眼手牌，想拖的那张还在、少的是别的 → 抓错了，拖回去
                #    （2026-09-24 第二轮第 2 局：拖前核对都通过了，按下去的仍是隔壁那张 —— 只能在半空确认）
                if self.grab_guard is not None and action["src"][1] >= 600:
                    guards.append(lambda: self.grab_guard(action))
                # ② 往场上放**新英雄**（不是合成、不是装备/效果牌）才检查人口会不会超
                if self.drop_guard is not None and not action.get("kind") and not action.get("merge"):
                    guards.append(self.drop_guard)
                g = (lambda: any(f() for f in guards)) if guards else None
                self.last_drag_aborted = False
                self._drag(tuple(action["src"]), board_drop(tuple(action["dst"])), guard=g)
            elif atype == "move_unit":
                # 挪动场上英雄（喂等级/换位）：src/dst 显式坐标
                self._drag(board_drop(tuple(action["src"])), board_drop(tuple(action["dst"])))
            elif atype == "sequence":
                # 多步序列（如：挪来喂等级 → 升级 → 挪回），逐步执行，一步失败即停
                for step in action.get("steps", []):
                    if not self.execute(step):
                        return False
                return True
            elif atype == "sell_unit":
                # 场上英雄拖到右下角能量圆盘 = 卖出（吃完登场效果后回收人口/能量）
                # **src 和 x/y 两种写法都认**：决策层有两处调用，写法曾经不一致，
                # 结果其中一条路每次都 KeyError（2026-09-23 实战才发现，
                # 因为异常被主循环吞成一条 ERROR，看日志只当是"这次没卖成"）。
                src = action.get("src")
                if src is None and "x" in action and "y" in action:
                    src = (action["x"], action["y"])
                if src is None:
                    logger.error("sell_unit 缺少落点（src 或 x/y）：%s", action)
                    return
                # **起点也要避开手牌区**：最下面一行格子 (y≈569) 被手牌区盖着时，
                # 从那儿按下去抓起的是**枫叶舞技能卡**，拖到圆盘上 = 卖枫叶舞
                # （2026-09-23 误报"要卖掉枫叶舞" —— 卖的其实是 (300,569) 的孙尚香）
                self._drag(board_drop(tuple(src)), BTN_SHOP_TOGGLE)
            elif atype == "sell_card":
                # **手牌拖到屏幕左侧的「出售」区**。拖起一张牌时左右两边会出现「+N 出售」区域
                # （实测截图：左 x≈40、右 x≈1240，y≈600）。原来拖到右下角能量圆盘 (1195,652)，
                # 历史 118 次一次都没卖掉 —— 右边出售区紧挨圆盘，差一点就偏出去了。
                # 用**左边**那块：离圆盘远，不会被圆盘吃掉。
                self._drag(tuple(action["src"]), SELL_ZONE_LEFT,
                           lift_check=(lambda: self.sell_grab_ok(action.get("piece") or "")) if self.sell_grab_ok else None)
                if self.last_drag_aborted:
                    return False
            else:
                logger.warning("未知动作类型: %s", atype)
                return False
            return True
        except Exception:
            logger.exception("动作执行失败: %s", action)
            return False
        finally:
            self._act_src = outer_src

    drop_guard = None          # 由主循环设置：无参调用，返回 True = 这次不能松手（例：人口变红）
    grab_guard = None          # 由主循环设置：grab_guard(action) 返回 True = 抓错牌了，拖回去
    sell_grab_ok = None        # 由主循环设置：无参调用，返回 True = 两侧出现了「+N 出售」区（手里的牌真被拿起来了）
    last_drag_aborted = False

    def _drag(self, src: tuple[int, int], dst: tuple[int, int], guard=None, tag: str | None = None,
              lift_check=None) -> None:
        """拟人拖拽（卡牌/英雄拖动）：touch_down → 慢速带中间路径点 → touch_up。

        不用 post_swipe：过快的一次性滑动很多游戏不识别为"拖拽"；
        本游戏实测需要按下后稍作停顿再慢速移动，否则卡牌不进入拖拽态。
        """
        steps = 24
        self.last_drag_aborted = False
        self.controller.post_touch_down(src[0], src[1]).wait()
        time.sleep(0.35)  # 长按停顿，让游戏进入拖拽态
        lifted = None
        if lift_check is not None and src[1] >= HAND_DRAG_MIN_Y:
            # **先确认拿起来了再横移**（2026-10-20 第 839 局：卖手牌司空震，按下去没被认成拖拽，手指上提后横扫棋盘，
            # 游戏把路过的李元芳拿起来拖进了出售区 —— 卖掉了场上的古币件）。
            # 先只走竖直段（手指还在这张牌正上方、没扫过别的格子），截一帧：两侧出现「出售」区、且放大详情的名字对得上才横移；
            # 否则原路放回。只提 40 像素时牌跟着手指抬起、但游戏还没进拖拽态（840 局莱西奥连续 4 次被误判）→ 用完整的竖直段。
            lift_y = min(src[1] - HAND_LIFT_DY, dst[1])
            for i in range(1, HAND_LIFT_STEPS + 1):
                self.controller.post_touch_move(src[0], int(src[1] + (lift_y - src[1]) * i / HAND_LIFT_STEPS)).wait()
                time.sleep(0.045)
            time.sleep(0.15)
            try:
                ok = bool(lift_check())
                if not ok:
                    time.sleep(0.25)                            # 动画慢一拍：再看一次
                    ok = bool(lift_check())
            except Exception:                                   # noqa: BLE001
                ok = True
            if not ok:
                for i in range(HAND_LIFT_STEPS - 1, -1, -1):
                    self.controller.post_touch_move(src[0], int(src[1] + (lift_y - src[1]) * i / HAND_LIFT_STEPS)).wait()
                    time.sleep(0.03)
                time.sleep(0.1)
                self.controller.post_touch_up().wait()
                self.last_drag_aborted = True
                logger.warning("拖拽 %s：竖直上提后没确认拿起（无出售区 / 详情名字不对）→ 放回，不横移", src)
                tap_gate.log_touch("drag", [src, dst], self._src(tag), {"aborted": True, "why": "no_sell_zone"})
                time.sleep(self.tap_delay)
                return
            lifted = (src[0], lift_y)
        # **从手牌拖出时先竖直往上提、再平移到目标**（2026-09-24 第二轮第 3 局，影子模式的"按下前/半空"两帧定案）：
        # 原来从手牌斜着直线拖向目标，目标在左边时手指先在手牌条上**横着滑过**几张牌，游戏要等移动够一段距离才认"开始拖"，
        # 那一刻手指下面已经是**拖拽方向上的另一张牌** —— 按在孙膑上、抬起来的是左边隔两张的牛魔。
        # 历史抓错案例全部吻合：往左拖抓到左边那张（干将→牛魔、孙尚香→速击之枪），往右拖抓到右边那张（西施→孙膑）。
        path = []
        if lifted is not None:
            start, rest = lifted, steps - HAND_LIFT_STEPS
        elif src[1] >= HAND_DRAG_MIN_Y:
            lift_y = min(src[1] - HAND_LIFT_DY, dst[1])
            for i in range(1, HAND_LIFT_STEPS + 1):
                path.append((src[0], int(src[1] + (lift_y - src[1]) * i / HAND_LIFT_STEPS)))
            start = (src[0], lift_y)
            rest = steps - HAND_LIFT_STEPS
        else:
            start, rest = src, steps
        for i in range(1, rest + 1):
            path.append((int(start[0] + (dst[0] - start[0]) * i / rest),
                         int(start[1] + (dst[1] - start[1]) * i / rest)))
        for x, y in path:
            self.controller.post_touch_move(x, y).wait()
            time.sleep(0.045)
        time.sleep(0.25)  # 到目标后停顿再松手
        if guard is not None:
            try:
                abort = bool(guard())
            except Exception:                                   # noqa: BLE001
                abort = False
            if abort:
                # 按原路拖回手牌再松手 = 取消这次拖拽
                for i in range(steps, -1, -2):
                    x = int(src[0] + (dst[0] - src[0]) * i / steps)
                    y = int(src[1] + (dst[1] - src[1]) * i / steps)
                    self.controller.post_touch_move(x, y).wait()
                    time.sleep(0.03)
                time.sleep(0.2)
                self.controller.post_touch_up().wait()
                self.last_drag_aborted = True
                tap_gate.log_touch("drag", [src, dst], self._src(tag), {"aborted": True})
                time.sleep(self.tap_delay)
                return
        self.controller.post_touch_up().wait()
        tap_gate.log_touch("drag", [src, dst], self._src(tag))
        time.sleep(self.tap_delay)

    def execute_all(self, actions: list[dict[str, Any]]) -> int:
        """顺序执行，返回成功数。"""
        return sum(1 for a in actions if self.execute(a))
