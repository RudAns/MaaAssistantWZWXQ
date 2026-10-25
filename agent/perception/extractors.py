"""感知层：把游戏截图提取为 GameState。

坐标基准：1280x720（MuMu 模拟器 ADB 截图），已按对局录像帧校准：
  data/logs/match_170447/frame_0014  战斗（左侧血量列表/战斗统计）
  data/logs/match_170447/frame_0039  准备（商店 4 卡/手牌/金币/升级）
  data/logs/match_170447/frame_0026  拍卖预览（6 拍品/出价列表/倒计时）
  data/logs/match_170447/frame_0097  天赋三选一（模态）
  data/logs/match_170447/frame_0224  出局结算（尚未结束/立即结算/继续观战）
"""

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import onnx_ocr as _onnx_ocr

from .state import (
    AuctionLot,
    AuctionState,
    BoardCell,
    GameState,
    PlayerRow,
    ShopSlot,
    TalentOption,
)

logger = logging.getLogger(__name__)

# ==================== 通用 ROI（1280x720 已校准） ====================
ROI_ROUND = (500, 0, 300, 50)           # 顶部 "第N回合|Ns"
ROI_BAN = (900, 0, 200, 45)             # 右上 "禁：大河"
ROI_GOLD = (1175, 630, 60, 55)          # 右下能量（货币）数字本体——这是本游戏唯一货币
ROI_GOLD_TIGHT = (1190, 640, 45, 40)    # 能量重读用紧 ROI（只罩数字）
ROI_GOLD_WHITE = (1170, 625, 75, 65)    # 白字读法的搜索区（含左边的闪电图标，靠颜色把它排掉）
ROI_INCOME = (1160, 575, 80, 45)        # 能量上方 "+N"（下回合增加）
ROI_ENERGY = (160, 685, 90, 35)         # 左下头像旁能量 "0/3"
ROI_LEVEL_BTN = (40, 520, 110, 55)      # 左下 "升级" 按钮（文字部分）
ROI_LEVEL_AREA = (20, 500, 220, 90)     # 升级按钮整片区域（徽章数字 + "升级" 二字）
                                        # 费用在按钮**左侧的圆形徽章**里，是独立文本块。
                                        # 2026-09-22 两次踩坑：
                                        #  1. 原用正则 r"(\d+)\s*升级" 匹配拼接文本 —— 文本块顺序不定，
                                        #     31 帧里 26 帧读不到，升级判据恒为 None，整局几乎不升级
                                        #  2. 改成只罩徽章的紧 ROI (50,530,34,32) —— 裁得太小 OCR 检测直接失败
                                        # 最终解法：在整片区域 OCR，按**相对位置**取"升级"左边的数字块
ROI_FREE_REFRESH = (1150, 320, 130, 105) # 右侧 "免费刷新/临时刷新/刷新" 按钮（阶段判据用）
ROI_REFRESH_BADGE = (1176, 350, 56, 28)  # 刷新按钮**下方小圆里的「×N」**。必须是细 ROI（<40px）整块识别：
                                        # 45px 高时要先检测文字框，小字常框不出来 -> 读成空，
                                        # 免费刷新次数一直是 0，整局一次都没用过（2026-09-23 发现）
                                        # 数字含义随按钮文字变（2026-09-22 确认）：
                                        #   "免费刷新" + N  → 剩余免费次数（**用效果卡获得的**，不是随回合白给）
                                        #   "刷新"     + N  → 本次刷新要花 N 能量
                                        # 三次踩坑同一个模式：× N 正则匹配不到（独立文本块）；
                                        # 放进大 ROI 一起读 → 小字号在检测阶段被丢掉（只剩"免费刷新"）；
                                        # 只有紧 ROI 才稳定读出数字。
BTN_REFRESH = (1210, 374)               # 刷新按钮点击点
BTN_LOCK = (1191, 196)                  # 锁定按钮（刷新按钮上方的小锁）。2026-10-11：原来 (1195,275) 点在锁和刷新之间的空白处，最近 240 次锁店 0 次生效（下回合商店全变）；证据 fixtures/shop_open_x_button.png
BTN_LEVEL_UP = (95, 550)                # 升级点击点
BTN_READY = (1195, 545)                 # "准备"按钮点击点
ROI_READY_BTN = (1140, 515, 140, 60)    # 右下角"准备/开战"按钮（商店关闭时的 prepare 判据）
                                        # 按钮文字会随状态变：未准备时"准备"，可开战时"开战"。
                                        # 2026-09-22 实战：只认"准备"导致关店后阶段判成 unknown，
                                        # 棋盘操作全停摆，手牌堆到 12 张溢出被系统自动卖掉。

# 商店 4 卡槽：卡面整体 y 38-397，宽 ~250
# 白字抠图阈值（`_ocr_white_text`）：白字低饱和高亮度，亮底立绘（金剑/雪地）高饱和。
WHITE_TEXT_SAT_MAX = 70
WHITE_TEXT_VAL_MIN = 185
SHOP_CARD_X = [38, 300, 560, 819]
SHOP_CARD_W = 250
ROI_SHOP_NAME_Y = (222, 48)             # 卡名（相对卡面 x+30）
ROI_SHOP_COST_Y = (45, 48)              # 费用徽章（相对卡面右上角 x+205）
ROI_SHOP_FACTION_Y = (360, 38)          # 派系（相对卡面 x+60）
ROI_SHOP_PROB = (620, 405, 460, 55)     # "当前概率" 行（阶段判据之二，UI 版本间有偏移，取大区域）

# 手牌：底部横排，卡名在每张卡底部 y 686-715；多张时重叠压缩
ROI_HAND_STRIP = (208, 638, 1072, 88)
# **x 不能从 100 开始**：那会把左下角的人口数字（"7/15"）当成一张手牌读进来。
# **y 必须放宽**：卡名那一行的高度**不是固定的** —— 实测同一局里出现过
# y≈653 / 665 / 701 三种位置（手牌数量不同、卡面会上下浮动）。
# 我一度按某一帧收成 (230,683,1050,38)，结果换个时刻**整条 ROI 一个字都读不到**，
# 手牌报 0 张、于是 decide_deploy 遍历空列表，买回来的英雄一个也上不去。
# 现在覆盖 y638~726。
HAND_GRIP_DY = -25                      # 拖拽抓手点 = 卡名框中心上移（点到卡面而非文字）
HAND_SCAN_WIN = 170      # 手牌条滑窗宽度（一张卡名的宽度量级）
HAND_SCAN_STEP = 60      # 步长：窗口之间大幅重叠，保证每张卡都被某个窗口整体罩住
HAND_SCAN_DEDUP = 45     # 同一张卡被多个窗口读到时按 x 去重
HAND_SCAN_EDGE = 4       # 识别框离窗口边缘小于它就判为骑跨读数，丢弃
HAND_NAME_DEDUP = 70     # 同一个**卡名**在这个距离内重复出现，只留一个
# **原来是 120，比真实牌距还大** —— 于是任何两张**挨着的同名牌**都被吞掉一张
# （2026-09-23 截图：手上 3 张弈星只认出 2 张、觉醒干将莫邪挨着普通干将莫邪，
#  整整一个回合都"看不见"）。这个游戏靠 3 张同名合成，等于专门吃合成材料。
# 全历史 5894 帧相邻读数间距：真实牌距最密一档 80~89（满手牌），
# 同一张卡被相邻窗口读两次的间距是 50~69（注释里的实例：797 与 855，差 58）。
HAND_FRAG_GAP = 18       # 间距小于它的两段文字判为**同一张卡**被 OCR 断开
HAND_GAP_MIN_RATIO = 0.6 # 相邻卡间距小于中位间距的这个比例 -> 判为伪卡
HAND_PSEUDO_MAX_GAP = 50 # 2026-09-26：65 → 50（满手牌叠放时真实牌距 ≈61，圣者法典旁边的装备箱被当伪卡删掉）。 ……但阈值**不能超过这个绝对值**。中位间距是拿**读到的**牌算的，
# 漏读几张之后中位数被撑大（实测 157 -> 阈值 94），相距 98 的两张真卡就被当成伪卡删掉了
# （2026-09-23 那一帧删掉的正是觉醒干将莫邪）。伪卡是长卡名断成两段，
# 两段挨得很近（几十 px），65 足够把它们和真实牌距（>=75）分开。
# 觉醒卡**不在这里认**：卡面上的 AWAKENED 在 1280x720 下只有 8~9px 高，OCR 只读出 'TTA'/'WA'；
# 卡面色调会在金色/冰蓝之间变、底栏亮度和普通卡也分不开（2026-09-23 实测）。
# 改由主循环在"分不清"时**点开这张牌看放大详情**，见 live_match_agent.inspect_hand。
HAND_MAX = 10                           # 手牌上限，超出部分开战后删除

# ---- 手牌布局（按每张牌左上角的圆形图标定位，2026-09-23）----
# 手牌多了会**压缩间距叠在一起**，每张只露出左边一截（卡片会缩小间距堆叠）。
# 整条 OCR 会把相邻两张的名字连成一个框（实测「孙悟空」+「刘邦」读成「孙作刘邦」），
# 这一张就"凭空消失"，拖牌前的校验找不到它，整回合卡死。
# 每张牌左上角都有一个**圆形图标**（派系/类别），叠得再紧也露在外面。布局规律（多局实测）：
#   - N ≤ 5：整排**居中**，间距 142，所有图标的中心在 x≈670；
#   - N ≥ 6：从 x≈385 排到 x≈994（最后一张完整露出），间距 = 609/(N-1)。
# 于是布局只剩一个未知数 N：对每个 N 算出网格，和检测到的圆比对打分（误检/漏检都不怕）。
# 枫叶舞（棋手技能牌）槽固定在第一张牌左边，它右上角的红色「0」角标在第一张图标左边 ~67px，
# 作为"第一张在哪"的佐证，也防止把它算成一张牌。
HAND_ICON_Y = 588
HAND_ICON_BAND = (290, 555, 890, 70)    # 找图标的区域（x 290~1180：左边要罩住叠放时 x≈319 的角标，右边再往外是能量圆盘）
HAND_ICON_TOL = 8                       # 图标离网格点多远以内算命中
HAND_FIRST_ICON_X = 385
HAND_SPAN = 609                         # 叠放时第一张到最后一张图标的距离
HAND_LAST_ICON_MAX_X = 1010             # 卡牌图标最右不会超过这里（最后一张在 ~994）
HAND_PITCH_MAX = 142                    # 不叠放时的牌距 = 一张牌的宽度
HAND_CENTER_ICON_X = 670                # 不叠放时整排图标的中心
HAND_ICON_TO_LEFT = 20                  # 图标中心 -> 卡牌左边缘
HAND_NAME_DY = 100                      # 图标中心 -> 卡名中心（纵向）
SKILL_BADGE_DX = 67                     # 第一张牌图标 -> 左边枫叶舞槽「0」角标
SKILL_NAME_DX = 121                     # 第一张牌图标 -> 枫叶舞槽卡名中心
LEVEL_BTN_TAP_DX = 38                   # 升级按钮：从「升级」二字中心往左挪到费用圆（避开站在上方的棋手）

# 战斗：左侧玩家血量列表 6 行
PLAYER_ROWS_Y = [92, 153, 215, 277, 339, 400]   # 六人血条行首 y（实测框标定）
ROI_PLAYER_HP = (112, 0, 70, 32)        # 血量数字（相对行 y）。
# **x 原来是 80/宽65，整体偏左 ~50px**，罩住的是头像不是数字 —— 实测读出
# 6/6/5/5/5/15 这种鬼值（真值 52/35/10），又一次"ROI 切字"。
# 实测框：52@[118,94,39,24] / 35@[124,155,28,22] / 10@[126,218,25,17]。
# 已淘汰的玩家那几行是灰的、没有数字，读空即可（hp=-1）。
ROI_PLAYER_NAME = (118, -24, 97, 24)    # 昵称（相对行 y）。**我方昵称是绿色的**，
# 靠这个认出六行里哪一行是自己：实测我方行绿像素 306，其余 ≤4，毫无歧义。
PLAYER_NAME_GREEN_MIN = 60              # 判我方行的绿像素下限（留了 5 倍余量）
# 第一名的绿像素要比第二名高出这个倍数才算认清了（见 own_hp_row）。
OWN_ROW_MARGIN = 2.0
# **两种排版**（2026-10-22 查血量乱跳时确认）：
#   带昵称（准备阶段多见）：昵称在上、血量数字左对齐从 x≈125 起；
#   紧凑（战斗中多见）：**没有昵称**，数字右对齐、整体左移到 x≈100~128。
# 两种排版里**我方的血量数字本身也是绿的**（别人白 / 蓝 / 紫），所以认行看"绿色数字"，
# 不只看昵称 —— 紧凑排版没昵称，旧判据返回 None、沿用旧行号，列表一重排就读成别人的血。
# 旧 ROI 从 x=112 起，紧凑排版的首位数字被切掉（37→7、29→9、57→7），也是乱跳的来源。
HP_DIGIT_BAND = (96, 4, 70, 26)     # 找绿色数字的带（相对行 y），覆盖两种排版
HP_DIGIT_START_MAX = 132            # 数字串最左列不会超过这里（数字后面的 ○ / 三叶图标从 ≥135 开始）
HP_DIGIT_MIN_TALL = 12              # 数字的绿像素至少纵跨这么多行（彩虹拖尾、横条特效只有 4~8 行）
HP_DIGIT_GAP = 6                    # 同一串数字的列间空隙上限（实测「37」两字隔 5 列；
#   数字和后面的 ○ 隔 ≥8 列，万一并进来 OCR 读成「370」，_player_hp 截两位仍是 37）
HP_DIGIT_MAX_W = 44                 # 两位数 + 万一并进来的 ○ 不超过这么宽（实测两位数 24~30）
HP_DIGIT_MAX_FILL = 0.62            # 数字串外接框里绿像素的占比（实测 0.25~0.54；卡牌立绘色块 0.76）
HP_NAMED_W = 44                     # 只靠昵称认出时我方血量框的宽（从 ROI_PLAYER_HP 的 x 起）
HP_COMPACT_X = 92                  # 紧凑排版读别人血量的 ROI 起点（宽 HP_COMPACT_W）
HP_COMPACT_W = 70
ROI_HERO_PANEL = (40, 500, 240, 110)    # 英雄详情面板（点棋盘上的英雄弹出）
# 判据用「出售」「剩余数量」—— 这块**平时是升级按钮所在的位置**。
# 面板开着时升级按钮被盖住，`_extract_economy` 找不到「升级」两个字，
# 于是安全规则拒绝点击（那个位置现在是「出售」，盲点下去就是把英雄卖了）。
# 后果：**有钱也升不了级**，实战攒到 36 能量、升级费只要 8，连着几十帧都在
# "想升级但画面上没找到「升级」按钮"（2026-09-22）。
# 而这时阶段仍然判成 prepare，unknown 兜底也不会触发 —— 所以必须单独认出来并关掉。
ROI_LORD_PANEL = (930, 30, 250, 40)     # 棋手详情面板顶部的「天赋 | 技能」两个页签
# **点棋盘上的棋手本体就会弹出这个面板**（点左下角头像不会）。它几乎盖满屏幕
# （实测 x280~1160 / y30~560），而原来的逃生点 TAP_SIDE_GROUND(1120,500)
# **正好落在面板内部** —— 点了等于没点，于是一路 unknown 到死
# （2026-09-22：点到英雄后死循环）。
# 关闭方式：点面板**外面**。这里取右侧边缘外的背景，避开手牌区（y>620 会抓起卡牌）。
BTN_LORD_PANEL_CLOSE = (1240, 300)
ROI_DESIGNATE = (500, 0, 300, 36)       # 顶部横幅「请指定英雄使用」（实测框 564,2,148,29）
# **这是个两段式交互**：某些牌（装备、指定类效果牌）拖出去之后并不会直接生效，
# 游戏会挂出这条横幅等你**再点一下目标英雄**。我们原来只做了第一段，
# 于是牌一直卡在"举着等指定"的状态：装备四选一根本没弹出来过，
# 而日志里"圆 10 个、识别出 1~3 个选项块"检出的其实是**棋盘单位的等级徽章**
# （2026-09-22 实战 5/5 帧全是这条横幅）。
# 棋手技能牌（枫叶舞）**可不可以换绑，卡面自己会说**：能换绑时整张卡是亮的，
# 已经绑住时是灰的（2026-09-22 确认：枫叶舞重新亮起来就说明可以换人绑）。
# 实测同一位置取样：亮起 饱和度 125/亮度 218，灰色 42/100 —— 两个维度都差 2 倍以上。
# **这比记账可靠**：绑定要靠卖人才能解除，而人可能被系统卖掉/觉醒回手，账记不全。
MAPLE_ART_DY, MAPLE_ART_H, MAPLE_ART_HALF_W = -70, 50, 45
MAPLE_READY_SAT, MAPLE_READY_VAL = 80, 150
MAX_POPULATION = 7                      # 人口上限的上限（官方：初始5，3级6，5级7，最多7）。

# **人口上限只可能是 5/6/7**（官方：初始 5，3 级变 6，5 级变 7，常规对局最多 7）。
# 读到别的值一律是误读 —— 2026-09-23 实战读出过 "1/3"，而 3 根本不是合法上限。
# 这条很重要：上限一旦被污染，买牌/上场的"场面满没满"就全错。
VALID_POP_CAPS = (5, 6, 7)
# ⚠ 上面是**按等级**的上限。天赋/效果牌还能再加（「战术模组」可上阵人数+1，2026-09-24 数据局画面 7/8），
# 所以合法值是 VALID_POP_CAPS + pop_bonus：用 `self._valid_caps()`，别直接查这个元组。
# pop_bonus 由规则引擎按知识库记账（rules.pop_bonus），主循环每帧同步给感知层。
# **必须硬钳制**：原来只拒绝"比上次小"的读数，于是一次误读出 9 就永久粘住
# （2026-09-22 实战日志里一路 "人口 3/9"，而真实上限是 7）——
# "只增不减"的守卫防得住抖动，防不住一次偏大的误读。
MAX_PLAYER_HP = 60                      # 每位棋手初始 60 生命值（官方新手指引）
# 一局最多掉多少血才算"讲得通"。**没有官方伤害公式**（查不到，见 meta.MAX_PLAYER_HP 注释），
# 但 60 血打十几个回合，单回合掉 25 已经是极端值；实测正常是 3~10。
# 这条替掉了原来的"大跌幅等下一帧复核"——**那个复核是循环论证**：
# 复核走的是像素缓存分支，而"像素没变"恰恰说明是**同一批像素**，
# 同一批像素读错一次就会读错第二次，等于自己给自己背书。
# 2026-09-23 实战后果：回合4 血 57，回合5 直接被"复核通过"成 **0** —— 人还活着。
HP_MAX_DROP = 25
# 连续这么多次**独立读数**都给同一个超限值，就认了（防止真掉了大血却永远卡在旧值）。
HP_REJECT_GIVE_IN = 3
HP_CONFIRM_DROP = 12             # 一次掉超过这么多要连读 2 次才认（948 局 23 → 1 误读）
# ……而且这个值要**持续够久**（秒）。"连读 N 次"原本隐含"持续了十几秒"（一帧 1~3 秒），
# 换成 TensorRT 后一帧 0.36 秒，9 次连读只要 3 秒 —— 列表重排 / 动画的几秒钟误读
# 就能凑够次数（2026-09-23：51 被读成 3 并采信，实际还有 45）。次数和时长两个条件都要满足。
HP_HOLD_S = 6.0
# 本局**首次**血量读数的合理性门槛：第 N 回合最少该剩 60 - N*这个数。
HP_FIRST_READ_PER_ROUND = 12
# 2026-09-24 游戏更新：右上角图标下加了文字（统计/天赋/聊天）和一条提示栏（"等待战斗结束""给XX穿戴装备"），
# 「战斗统计」从 y≈68~98 挤到 y≈137~164。区域放宽到 55~175，新旧界面都认；准备阶段这片区域没有这四个字（实测）。
ROI_BATTLE_STAT = (1100, 55, 180, 170)  # "战斗统计"（阶段判据）。去对手棋盘打时右侧多两行「打开商店」，它被挤到 y≈202（match14 复盘，unknown_1790262532.png）
ROI_POPULATION = (820, 235, 170, 70)    # 人口数 "5/5"（己方场地右侧，召唤物/图腾不占）
# 棋手等级徽章：头像右下角的**点数图标**（2026-09-22 确认）。
# 1 点=1 级 … 5 点=5 级，和商店「当前概率」行里的品阶图标是同一套点阵。
# **不是数字**，所以不能 OCR；这是 UI 贴图、不随卡牌等级变 → 数亮点即可。
ROI_LORD_LEVEL = (104, 682, 34, 32)
ROI_HAND_COUNT = (140, 682, 65, 32)     # 左下角剪贴板图标的手牌数 "N/M"（2026-10-15 第 632 局截图 13/15）
ROI_LORD_XP = (55, 686, 70, 24)          # 头像左边的棋手经验 "N/M"（满级时写「满级」）
LORD_XP_NEEDED = {7: 1, 13: 2, 17: 3, 23: 4, 27: 5}   # 升级所需经验 -> 当前等级（实测 0/23=4 级、7/17=3 级、3/13=2 级）
LEVEL_DOT_THRESHOLD = 190        # 亮点阈值（白点 vs 深色底盘）
LEVEL_DOT_AREA = (8, 90)         # 单点面积范围。**下限必须 >4**：4 级徽章中心有一个
                                 # 面积 4 的反光，算进去会把 4 级读成 5 级
                                 # （189 帧实测：0 帧读到 4、6 帧读到 5，全是这个原因）
# **1~6 级各有不同点数**（2026-09-22 确认；最初以为 6 级和 5 级一样只是彩色）。
# 已在真实帧上核实：卡死帧 stuck_1790072565.png 左侧的**等级进度条**完整露出六个图标，
# 自下而上依次是 1点 / 2点 / 3点(三角) / 4点 / 5点(四瓣+心) / 6点(六瓣)。
# 所以只要数点，不需要看颜色。
# **注意：5 点和 6 点只在进度条那种大图标上核实过**；头像徽章只有 34x32，
# 现有日志最高打到 4 级，5/6 在小徽章上会不会粘连成一团未知 —— 第一次读到要人工核对。
ROI_POPULATION_TIGHT = (840, 245, 130, 50)  # 只罩数字：OCR 认不出中间的斜杠，
                                        # 宽 ROI 会把右边的人口图标也当成数字（"1/5" → "1151"），
                                        # 这个范围稳定给出 "115"/"05" → 取**末两位**解析成 1/5、0/5。
                                        # 2026-09-22 人工标注的实际位置换算后是 (815,232,164,71)，
                                        # 与 ROI_POPULATION 基本重合——ROI 一直是对的，错的是解析。
BTN_SHOP_TOGGLE = (1195, 652)           # 右下角能量圆盘：点一次开商店/天赋，再点一次关
                                        # （2026-09-22 确认）。关掉商店才能看见人口数。

# 拍卖：6 个拍品位，卡面 y 70-380，出价列表 y 380-440
AUCTION_LOT_X = [15 + i * 211 for i in range(6)]
AUCTION_LOT_W = 200
ROI_AUCTION_TITLE = (520, 0, 320, 50)   # "预览|拍卖即将开始"
ROI_AUCTION_NAME_Y = (218, 48)          # 拍品卡名（相对卡面 x+20）
ROI_AUCTION_BID_Y = (355, 110)          # 出价区（相对卡面 x+10，含出价人名+价格两行）
OUR_LORD = "阿离"      # 我们只玩阿离；别家棋手的 lord 专属牌不进认名候选
ROI_AUCTION_COUNTDOWN = (1150, 605, 110, 100)  # 旧版右下圆形倒计时（预览期）；出价期此位是能量
ROI_AUCTION_COUNTDOWN_TOP = (600, 0, 240, 50)   # 新版顶部 "预览 | N" 倒计时
BTN_AUCTION_BID = (1065, 655)           # "出价"按钮（选中拍品后点击出价，价格=当前价+1）
ROI_AUCTION_DIAL = (1140, 600, 140, 110) # 右下角能量圆盘所在区域。
                                        # **判断"能不能出价"就看它在不在**（2026-09-22 确认）：
                                        # 顶部"拍卖开启"只闪一瞬，之后一直显示"预览"，文字判据不可用。
                                        # 圆盘出现 = 可以选卡出价；预览期那里是舞台背景。
                                        # 16 张实帧验证：有圆盘的 14 张饱和度 88-91、霍夫圆=1；
                                        # 没圆盘的 2 张饱和度 43-47、圆=0，且正是当时报"找不到出价按钮"的两次。

# 装备四选一弹窗：基础装备拖到英雄身上后，在英雄旁弹出 4 个选项卡
# （选项是成品装备牌，用 OCR 在棋盘区域内匹配装备牌名定位，见 live agent 处理）
ROI_EQUIP_POPUP = (250, 80, 800, 480)   # 弹窗大致区域（棋盘上半部）

# 天赋三选一（模态）
TALENT_CARD_X = [269, 538, 806]
TALENT_CARD_W = 243
# 2026-09-24 实测更正（tools/ocr，150 帧 450 张天赋卡）：卡名中心 y≈360（金底卡 325~340）、底边 ≤374；
# 描述第一行从 y≈392~403 开始。原来的 (360,50) 正好从卡名中心开始切 —— 上半截丢了，金底卡整个在框外，
# 卡名只认出 31%，一直靠"描述反查"兜底；描述区 (415,70) 又切掉了第一行描述。
# 改后卡名直接认出 84%，卡名与描述反查指向同一张卡 373/379（旧 134/141）。
ROI_TALENT_NAME_Y = (310, 68)           # 天赋名（相对卡面 x+20）
ROI_TALENT_DESC_Y = (385, 115)          # 描述（相对卡面 x+15）
ROI_TALENT_MARKER = (580, 610, 240, 45) # "全局1次"（阶段判据）
# 每张天赋卡右下角的类别小字（「天赋」/「装备」+ 英文 TALENT/EQUIPMENT）。
# **满级时底部那行「全局N次」没有了**，只靠 ROI_TALENT_MARKER 就认不出天赋界面，
# 三选一一直挂着没人选（2026-09-23 第 15 回合）。小字 OCR 常读偏（天摄/天腻/天限…），全收。
# 全历史帧校验：天赋 102/120 命中，准备阶段 0/120、其余各阶段 0/40 误中。
# 不收 "HONOR"：准备阶段有 1 张误中（卡面底部的 HONOR OF KINGS）。
ROI_TALENT_TAGS = (250, 455, 840, 50)
# 满级那种**橙金色**天赋卡上，「天赋」会读成「天」「天膜」「天娱」、TALENT 读成 LENT/LEMT/DAENT
# （2026-09-23 第 13 回合实测，整回合没认出来）。放宽后全历史帧：天赋 111/120、准备阶段 0/120，
# 棋手面板有 1 张单处误中（一整句话里含"天赋"）—— 被"至少 2 处"挡掉。
TALENT_TAG_RE = r"^天.?$|天赋|TALENT|LENT|LEMT|AENT|EQU"
# 天赋三选一被收起时，左侧挂着「天赋未领取」（点它重新打开）
ROI_TALENT_PENDING = (25, 525, 140, 50)
# 棋手满级：头像下方写「满级」，点数徽章和升级按钮都没了
ROI_LORD_MAXED = (40, 680, 100, 36)
BTN_TALENT_REFRESH = (690, 668)

# 棋手特性三选一（阿离叫"枫舞强化"）：跟常规天赋**不是同一个界面**，没有"全局1次"标记。
# 触发时机：棋手任务达成一档（阿离"惊鸿游龙"：枫叶舞指定的英雄击败 3/7/11/15 名敌人，
# 每档给 1 张秘技牌 + 解锁 1 项特性）。漏掉就等于白打——2026-09-22 实战整局没识别。
# 判据用底部那排"<X>强化"小字（每张卡下面都有一条），标题栏是艺术字 OCR 不准。
ROI_LORD_BOOST_MARKER = (200, 365, 900, 45)   # 三张卡下方的 "枫舞强化" 标签行
ROI_LORD_BOOST_NAMES = (150, 150, 930, 60)    # 三个特性名（金色描边字，靠卡牌库反查）
# ↑ 右边界收到 1080：再往右是**已获得天赋列**（荆棘牢笼/翩跹舞步·蓄势…，x≈1085+），
#   扫进来会被当成候选（2026-09-23 点了天赋列里的「荆」）。
RIGHT_TALENT_COL_X = 1080   # 右侧已获得天赋列的左边界：所有"扫一排卡名"的检测都不能越过它
ROI_LORD_BOOST_DESCS = (150, 215, 1000, 80)   # 三段效果描述

# 选棋手：底部候选头像排（图标 y≈642-672，名字 y≈675-720），确认按钮右下
HERO_CANDIDATES = [(289, 665), (395, 665), (500, 665)]   # 兜底坐标（OCR 失败时用）
ROI_HERO_NAMES = (150, 675, 1050, 45)     # 候选棋手名字条
ROI_HERO_PICK_MARK = (540, 0, 320, 50)  # 顶部居中的阶段条："选择棋手|Ns" / "挑选搭配|Ns"
                                        # 2026-09-22 第二局实测：原来的 x 从 600 起，而"挑选搭配"
                                        # 那行字实际从 **x≈597** 开始，第一个字又被切掉
                                        # （和 ROI_START_GAME / ROI_START_MATCH 同一个坑），
                                        # 于是换装页落进 unknown、6 秒后兜底开始盲点棋盘。
                                        # 这行字是居中排版的，ROI 要围着中心 640 留足两边余量。
BTN_HERO_CONFIRM = (1136, 648)          # "确定"按钮

# 关"卡牌放大详情"用的**安全空白点**（2026-09-22 确认：点卡牌以外任意空白处即可关）。
# 这个点是在真实卡死帧 `stuck_*.png` 上逐个候选标注验证出来的，不是估的：
#   (170,100)  ← 选它。左上角场景装饰：在放大卡牌(x265-512)之外、棋盘(x>300,y>150)之外、
#                 天赋图标列(x≈40)之外、所有按钮之外。
#   (170,620)  ✗ 那里正好是「升级」按钮
#   (640,80) / (1240,110)  可用但贴着棋盘边缘，余量不如上面那个
# **不要改成棋盘上的点**——点中棋子会弹英雄详情面板，那个面板左下角是「出售」。
# 关"选完天赋后展开的天赋槽位栏"用的落点：**棋盘右侧的空地**（2026-09-22 定）。
# 为什么不是 TAP_BLANK(170,100)：实测点它三次完全无效，面板一直挂到被下一个
# 拍卖界面盖掉，连废两局（第十一、十二局）。
# 选点依据：x=1120 在最右一列棋格(x≈1027)之外，y=500 避开右侧那一列按钮
# （锁定 y≈195 / 免费刷新 y≈320 / 能量圆盘 y≈652）。两帧实测都是景物，
# **不在棋格内 → 不会点中棋子弹英雄详情**（这正是禁止盲点棋盘的原因）。
TAP_SIDE_GROUND = (1120, 500)
TAP_BLANK = (170, 100)

# 游戏设置面板（画面/声音/功能/网络 四个页签 + 左上角 ✕ + 右下角「退出对局」）。
# 它是**按 Android 返回键弹出来的**——这个游戏的返回键是"打开设置"，不是"关模态"
# （2026-09-22 发现）。所以返回键已从兜底里删掉；但面板万一被弹出来，
# 得有条路关掉它，否则整个画面被盖住、后面全是 unknown。
# 注意：面板右下角那个「退出对局」离得很近，兜底的按钮正则**绝不能**匹配"退出"。
ROI_SETTINGS_TABS = (150, 8, 640, 48)   # 顶部页签行，判据用"声音"（其他界面不会出现）
BTN_SETTINGS_CLOSE = (76, 30)           # 左上角 ✕（实测 1920 空间 (114,45) → 1280 空间 (76,30)）

# 读条界面
ROI_LOADING_BAN = (30, 615, 420, 70)    # 读条界面左下角 "本局禁用：河洛"
                                        # （2026-09-22 实测：原来圈的 (340,100,600,200) 在屏幕中上部，
                                        #  位置完全不对，读条阶段一直被判成 unknown，
                                        #  连本局禁用派系都读不到——而那是选阵容的关键输入）

# 结算
ROI_SETTLE = (440, 240, 420, 140)       # "尚未结束" / "第N名"（RANKING 大字区）
ROI_SETTLE_BTNS = (240, 580, 820, 90)   # 结算按钮行："立即结算/继续观战/现在退出"
BTN_SETTLE_NOW = (634, 614)             # 立即结算（出局时的"尚未结束"弹窗）
BTN_EXIT_MATCH = (500, 627)             # 最终名次页"现在退出"
BTN_SPECTATE = (1005, 614)              # 继续观战
BTN_PLAY_AGAIN = (1120, 672)            # 最终战绩页"再来一局"
ROI_PLAY_AGAIN = (1000, 640, 240, 60)   # "再来一局"按钮区域
ROI_START_MATCH = (980, 595, 300, 100)  # 组队房间"开始匹配"按钮区域。
                                        # 与 ROI_START_GAME 同一个坑（2026-09-22 实战）：原 ROI 从 x1040 起，
                                        # 把"开始匹配"（实测 x1000-1235 / y610-678）的第一个字切掉，
                                        # OCR 只读出"开始"，lobby 判据 "开始匹配" 不命中 → 组队房间判成 unknown。
                                        # 当时是 unknown 兜底正则碰巧匹配到"开始"才点进去的，属于蒙对。
BTN_START_MATCH = (1117, 644)           # 大字中心（与"开始游戏"同一位置的艺术字按钮）
ROI_START_GAME = (980, 595, 300, 100)   # 模式选择页"开始游戏"按钮区域。
                                        # 2026-09-22 实测：原来的 (1040,640,230,70) **把大字切掉了**——
                                        # "开始游戏" 实际占 x1002-1245 / y610-678，ROI 从 (1040,640) 起只
                                        # 罩住右下一角，OCR 读成 "干如游戏"，判据 "开始游戏" 永远不命中，
                                        # 整个模式选择页被判成 unknown（unknown 兜底会开始乱点）。
BTN_START_GAME = (1122, 644)            # 大字中心（实测 box 中点）。原 (1175,680) 压在下面那行小字
                                        # "训练-万象匹配赛" 上，已经贴着按钮边缘
ROI_MATCH_FOUND = (500, 80, 300, 120)   # "匹配成功"标题区域
BTN_MATCH_CONFIRM = (636, 517)          # 匹配成功"确定"按钮

# ==================== 己方半场落子点（人工标注实测，2026-09-22） ====================
# **这 28 个坐标是人工在截图上逐格标注、程序提取出来的，不是估的。改之前先确认有新的标注。**
#
# 之前的版本是目测的"前排5个 + 后排5个 = 10 个点"，两处硬伤：
#   1. 真实棋盘是 **4 行 × 7 列 = 28 格**，10 个点连一半都覆盖不到
#   2. 老坐标的 y=420/480 **正好夹在真实的第1行(y≈431)和第2行(y≈500)之间** —— 落在格线上，
#      所以经常"看着放上去了、其实归到隔壁格"，进而造成同格堆叠
#      （2026-09-22 发现：英雄总被摆在棋盘格线上）
# 棋盘是透视梯形，所以每行的 x 间距和起点都不同，**不能用等差数列生成**。
OWN_BOARD_ROWS = [                                   # 由近敌(row0)到近己(row3)
    [(319, 375), (411, 372), (529, 372), (617, 372), (718, 371), (818, 374), (918, 368)],
    [(346, 430), (451, 433), (556, 430), (672, 430), (783, 438), (885, 437), (980, 430)],
    [(276, 501), (390, 502), (499, 500), (617, 496), (720, 500), (839, 502), (955, 502)],
    [(300, 569), (433, 569), (555, 569), (681, 571), (786, 569), (917, 570), (1027, 572)],
]




def _gpu_session(path, name):
    from .gpu_session import session
    return session(path, name)

def _cpu_so():
    """CPU 上的小模型一律限 2 线程（2026-10-12：没设时 onnxruntime 默认开满 24 核，手牌 / 场上一多就把整机冲到 70~90%）。"""
    import onnxruntime as _ort
    so = _ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return so

def _center_out(cells):
    """从中间往两边排（坦克要先占中路，挡住对面主攻方向）。"""
    mid = len(cells) // 2
    order = sorted(range(len(cells)), key=lambda i: abs(i - mid))
    return [cells[i] for i in order]


def _corners_in(cells):
    """从角落往中间排（输出堆角落不容易被近身，也不会被 AOE 一锅端）。"""
    mid = len(cells) // 2
    order = sorted(range(len(cells)), key=lambda i: -abs(i - mid))
    return [cells[i] for i in order]


# 摆放原则（2026-09-22 定）：**坦克放前排，其他职业尽量后排往角落堆**。
def _interleave(a, b):
    """两行交错：a[0], b[0], a[1], b[1] …

    为什么不是"先把第一行排满再用第二行"（原来的写法）：
    2026-09-23 按 248 个有锚点的 prepare 帧统计，实际占用是
    **row0 396 次、row3 367 次，而 row1 只有 5 次、row2 只有 10 次** ——
    等于我们自己把人全堆在两行上，然后在那两行反复撞车
    （失败最多的落点 529,372 / 300,569 / 1027,572 / 617,372 正是占用最高的四格）。
    交错之后同样是"坦克靠前、输出靠后"，但会自然铺开两行，撞车概率大降。
    """
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


OWN_BOARD_DROPS_FRONT = _interleave(_center_out(OWN_BOARD_ROWS[0]),
                                    _center_out(OWN_BOARD_ROWS[1]))
OWN_BOARD_DROPS_BACK = _interleave(_corners_in(OWN_BOARD_ROWS[3]),
                                   _corners_in(OWN_BOARD_ROWS[2]))
# **A6、A7、B6、B7 不放英雄**：站在这四格的单位，头顶等级徽章会压在人口数字上。
# 2026-09-24 第一次发现（「25」「27」都来自 A6）时只是"排到最后"；视觉标注又查到两帧满员 6/6 被读成 5
# （第一位的 6 被 A6 单位的徽章挡掉左半边）。满员时有"数字变白"兜底，**没满员时被挡就没有任何兜底**。
# 2026-09-24：放英雄避开 A6、A7、B6、B7 四个位置。人口最多 7，剩下 24 格够用。
POP_OCCLUDING_CELLS = {(818, 374), (918, 368), (885, 437), (980, 430)}
OWN_BOARD_DROPS_FRONT = [c for c in OWN_BOARD_DROPS_FRONT if c not in POP_OCCLUDING_CELLS]
OWN_BOARD_DROPS_BACK = [c for c in OWN_BOARD_DROPS_BACK if c not in POP_OCCLUDING_CELLS]

# （2026-10-09 删掉了估值的 BENCH_CELLS / BOARD_ORIGIN / BOARD_CELL 与没人调用的 extract_board：棋盘格以 OWN_BOARD_ROWS 为准）


# ==================== 棋盘占用（观测，不是记账） ====================
# 到 2026-09-22 为止，"哪一格站着人"一直是**猜**的：`rules._field` 只记录"我拖过什么上去"，
# 拖失败、阵亡、被卖、觉醒回手都不会同步，于是长期出现"账本说有人、画面其实是空地"，
# 导致上场白拖、装备拖到空格、对账对不上。这里改成**从画面直接看**。
#
# 用什么信号：**单位头顶的等级徽章**（深色小圆 + 1~2 位数字）。
# 为什么不用立绘/卡图：**立绘会随卡牌等级变化**（2026-09-22 确认，实测 4 张里 3 张
# 匹配度低于随机负样本），美术资源一律不能当模板；而徽章是固定 UI 贴图，不随等级变。
# 为什么用滑动窄窗口而不是一条宽 ROI：宽 ROI 会让小字号在 OCR 检测阶段被丢掉
# （本文件多处踩过，整条带一次读实测漏一半）。
# **别收窄这个 ROI**：试过收成 (400,335,520,110) 只留 6 个滑窗（150ms），
# 结果徽章全丢/认错（board_now 从 (840,480) 变成 (780,480)，两张 equip 帧直接归零）。
# OCR 的检测阶段对窗口内容很敏感，换个裁剪结果就变——和"合并 ROI 丢小字"同一类现象。
# 所以保持这个实测有效的范围（48 个滑窗、约 700-840ms），用**像素缓存**把成本摊掉：
# 占用只在上场/卖出/阵亡时变，稳态下直接复用上一次结果（见 _extract_board_units）。
ROI_ARENA = (280, 120, 740, 430)        # 竞技场（含双方），徽章都在这里面
# WARNING **这几个参数是对着「老的错误网格」标定的，2026-09-22 换成人工实测的 4x7 网格后全部作废。**
# 老网格只有两排、间距 60px；真实网格是 4 排、行距仅 55~70px，而徽章离单位约 60px ——
# 用旧偏移量会把徽章吸附到**相邻行**，造成大面积误标占用。
# 所以先置 False，等拿新网格重新量过 DY/SNAP 再打开。
# （观测本来就是「只做加法」的辅助信号，关掉只是少一层保险，不影响主流程。）
BOARD_OBS_ENABLED = False
BADGE_DROP_DY = 60                      # 待重新标定（旧值对应已废弃的网格）
BADGE_SNAP_MAX = 45                     # 映射到落点的最大距离；超过就不是己方落子点上的单位
BADGE_WIN = (160, 110, 80, 60)          # 滑窗 宽/高/x步长/y步长


# 徽章的像素签名（2026-09-22 在真实帧上量出来的，阈值不是猜的）：
#   真徽章    p10= 5.9 / 9.0 / 3.0   sat= 77~108   ← 内芯是纯黑圆
#   误检圆    p10=31~38              sat=115~169
#   装备图标  p10=17                 sat=203       ← 彩色，靠饱和度排除
BADGE_MAX_P10 = 20        # 暗部分位数：徽章内芯必须够黑
BADGE_MAX_SAT = 150       # 饱和度：徽章是灰黑的，彩色圆（装备图标）排除


def _badge_like(image: np.ndarray, cx: int, cy: int, r: int) -> bool:
    """这个圆看起来像不像等级徽章（深色内芯 + 不饱和）。"""
    import cv2
    rr = max(6, int(r * 0.75))                 # 只看内芯，避开亮色描边
    patch = image[max(0, cy - rr):cy + rr, max(0, cx - rr):cx + rr]
    if patch.size == 0:
        return False
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)[:, :, 1].mean()
    return float(np.percentile(gray, 10)) < BADGE_MAX_P10 and sat < BADGE_MAX_SAT


# ==================== 英雄身上的装备件数 ====================
# **每个英雄最多带 3 件装备**（网查 + 2026-09-22 确认）。满了之后游戏会
# **直接拒绝拖放**：装备牌留在手上、四选一根本不弹，而决策层看不出区别，
# 于是对着满员的英雄反复重拖（实战：风暴巨剑连拖 3 次都被拒）。
#
# 设计要求：**不能只靠记账，要实际看英雄身上的图标**。
# （记账也确实不够：卖掉带装备的英雄，装备会退回手牌，计数就对不上了；
#   后期转移主力时更要知道装备当前在谁身上。）
#
# 几何（在真实 prepare 帧上标定）：等级徽章是个**深色大圆**(r≈11~13、饱和度<75)，
# 装备图标是**一排彩色小圆**(r≈6~12、饱和度≥70)，紧贴在徽章正下方 10~42px，
# 横向不超过 ±45px。徽章中心大约在格子中心**上方 125px**（单位画得高）。
# 两段式检测：先找徽章 → 只在该徽章下面那一小块里数彩色圆。
# **不能按格子直接扫**：相邻行的带会重叠，会把别人的装备算到自己头上。
MAX_EQUIPS = 3
EQUIP_BOARD_ROI = (200, 180, 900, 340)   # 己方棋盘上**单位徽章**出现的范围。
# 2026-09-30 第 26 局：左边界 250 → 200。最左一格 (276,501) 上的单位徽章在 x≈242，被切在框外，66 级的沈梦溪整局读不到等级
# 高度只到 y=480：实测真实徽章落在 y240~460，再往下是手牌区和界面元素 ——
# 原来罩到 y=600，把手牌顶部的圆也当成了单位徽章。
EQUIP_BADGE_DY = 116                     # 徽章中心 -> 格子中心的纵向距离
BADGE_DY_PEAK = 122                      # 全历史帧 dy 峰值 120~124
EQUIP_BADGE_DY_TOL = 61                  # 纵向偏移容差 -> 实际接受 55~177
# ↑ 2026-09-25 下限 75 → 55：**矮个子英雄**（鲁班七号…）的徽章离格心只有 ~60px，(433,569) 上的鲁班七号整局 0 帧读到等级
#   （第四轮第 1 局，老师标注反复报"主C 等级读不到"）。bench_badges 627 帧：准确 361→366、漏检 194→187、多检 72→74；
#   最新一局 40 帧：(433,569) 徽章检出 0→18。又一次"阈值只按见过的样本定"。
# **这个容差不能收窄**：徽章在英雄头顶，而**不同英雄模型身高差很多**，
# 全历史帧统计 dy 连续分布在 80~135（峰值 120~124）。
# 我一度按单个样本收成 126±15，结果**一个单位都检测不到**，
# 进而让"没观测到 = 离场了"的判据误删真实记录。
# 收窄只能排掉手牌区的圆（dy≈-18），排不掉别的 —— 别指望靠它区分棋手。
EQUIP_BADGE_DX_TOL = 45                  # 横向偏移容差

# ---- 徽章"是不是徽章"的判据（2026-09-23 用 249 帧真值重测后改写）----
# **原来用"饱和度 < 75"排除彩色物，这条判据是错的，而且错得很彻底。**
# 拿 population 当真值（人口数 = 场上单位数，商店关着时读数可靠）量了 249 个准备帧：
#   旧判据 sat<75 + 半径 10~14：**准确率只有 37%**，139 帧漏检，平均误差 1.10 个。
# 逐点量了真假样本才看出来：
#   真徽章 val(r*0.5) = 35~44，**而 sat 在 79~95** —— 早就越过 75 了，
#   所以**真徽章一直在被自己的过滤器扔掉**；被留下来的反倒是卡面上的深色圆。
#   假圆（卡面图标/血条/棋盘贴图）val = 114~178，sat 反而散在 40~143，完全分不开。
# 结论：**徽章的不变量是"深色"，不是"灰色"**。换成 val<70 并把半径放宽到 14~26
# （真徽章实测 r=20~22，旧的 rmax=14 直接把它排除在外了）：
#   准确率 37% -> 59%，漏检 139 -> 91，平均误差 1.10 -> 0.66。
# 再把"平均亮度"换成**中位亮度**（高等级徽章里是两位数，白色笔画把均值拉过了 70，
# 实测等级 58 的徽章中位 28、均值 >70 被扔掉）：
#   准确率 59% -> **66%**，漏检 91 -> **59**，多检 12 -> 25，平均误差 0.66 -> 0.55。
#   多检变多是可以接受的方向：锚点只用来"标占用"，多认只是少用一个格子，漏认才会撞车。
#   中位 <60/<80/<90 也量过（65%/54%/48%），70 最好。
# 复现：`tools/bench_badges.py`（改这三个常数前后都要跑）。
BADGE_MAX_V = 70                         # 平均亮度上限（超过就不是徽章）
BADGE_R = (14, 26)                       # 霍夫半径区间
BADGE_PARAM2 = 20                        # 霍夫累加阈值
# **不能用"最近格子 + 曼哈顿容差"这种松判据**（2026-09-22 发现坐标定位有问题）：
# 它会把**场外的东西硬吸到棋盘格上**。实测三类 dy 截然不同：
#   真实单位 dy=+118~+131 ／ **棋手** dy=+85（棋手根本不站在棋盘上）
#   ／ 手牌区的圆 dy=-18
# 卡死 dy 区间之后，后两类自然被排除 —— 这比"认出棋手站哪格再避开"更正确，
# 因为棋手压根就没有"所在格"。


def _hough(gray, param2: int, rmin: int, rmax: int, min_dist: int):
    import cv2
    found = cv2.HoughCircles(cv2.medianBlur(gray, 3), cv2.HOUGH_GRADIENT, dp=1,
                             minDist=min_dist, param1=90, param2=param2,
                             minRadius=rmin, maxRadius=rmax)
    return [] if found is None else np.round(found[0]).astype(int).tolist()


def _mean_sat(hsv, x: int, y: int, r: int) -> float:
    import cv2
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.circle(mask, (x, y), max(2, r - 2), 255, -1)
    return float(hsv[..., 1][mask > 0].mean())


def _median_val(hsv, x: int, y: int, r: int, frac: float = 0.5) -> float:
    """圆心附近亮度的**中位数**。等级徽章用它，不用均值。

    徽章是"深色底 + 白色等级数字"，数字位数不同、白色笔画多少就不同：
    实测等级 58 的徽章，中心中位亮度 28（明明是深色），均值却超过 70 被当成亮圆扔掉。
    高等级单位恰恰是最值钱的那几个，漏认它们代价最大。
    """
    import cv2
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.circle(mask, (x, y), max(2, int(r * frac)), 255, -1)
    return float(np.median(hsv[..., 2][mask > 0]))


def _mean_val(hsv, x: int, y: int, r: int, frac: float = 0.5) -> float:
    """圆心附近的平均亮度。**取半径的一半**，别取整圆。

    整圆会把圆外的底色（棋盘是亮黄色）吃进来：实测同一个等级徽章，
    r*0.85 量到 40、r*0.5 量到 35，而旁边的亮色图标是 114~178 ——
    半径越小，两类分得越开。
    """
    import cv2
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.circle(mask, (x, y), max(2, int(r * frac)), 255, -1)
    return float(hsv[..., 2][mask > 0].mean())


def cell_key(cell: tuple[int, int]) -> str:
    """格子坐标 -> 字符串键。GameState 每帧要 json 序列化，不能用元组当键。"""
    return f"{cell[0]},{cell[1]}"


UNIT_ANCHOR_DY = 40     # 徽章(头顶) -> 可拖放的"身体"位置。徽章在格子中心上方 125px，
                        # 往下 40 大致是胸口，既在单位身上、又不会低到踩进空地。


def detect_unit_anchors(image: np.ndarray) -> dict[str, list[int]]:
    """{"x,y"(格子): [锚点x, 锚点y]} —— **往单位身上拖东西时要用这个，不是格子中心**。

    2026-09-22：装备总是往空地送。日志佐证：风暴巨剑连拖 3 次到 (917,570)
    全失败，而马可波罗**上场到同一个格子**是成功的 —— 因为上场认的是格子，
    而装备必须命中**单位本体**，单位是画在格子上方的（徽章高出 125px）。
    这里不拍脑袋定偏移，直接用**观测到的徽章位置**当锚点。
    """
    return {k: [x, y + UNIT_ANCHOR_DY] for k, (x, y) in _detect_badges(image).items()}


VISION_ANCHOR_UP = 100   # D-FINE 框底 ≈ 格心下 15px；徽章锚点 ≈ 格心上 85px（徽章高 125 + 往下 40）→ 框底往上 100


def _vision_fill_anchors(image: np.ndarray, state) -> None:
    """D-FINE 视觉检测：旧徽章检测漏掉的己方单位格补进 unit_anchors。

    400 帧人工标注上旧徽章检测格子召回 89.5%、C1 漏 111 次；D-FINE 测试集召回 100%、C1 9/9（部署链路同样验证过）。
    **只加占用、不删记录**（铁律）；同一格徽章也认出时用徽章锚点（拖东西更准），setdefault 不覆盖。
    """
    from . import vision_units
    if not vision_units.enabled():
        return
    cells = [tuple(c) for row in OWN_BOARD_ROWS for c in row]
    try:
        found = vision_units.cell_units(image, cells)
    except Exception:  # noqa: BLE001
        logger.exception("视觉单位检测失败（忽略，本帧只用徽章）")
        return
    added = []
    try:
        from . import unit_id
        state.unit_names = unit_id.identify(image, {cell_key(c): v[2] for c, v in found.items() if v[0] == "hero"})
    except Exception:  # noqa: BLE001
        logger.exception("认人模型失败（忽略，本帧只靠账本）")
    for c, (cls, _sc, (x0, _y0, x1, y1)) in found.items():
        k = cell_key(c)
        state.vision_units[k] = cls
        if k not in state.unit_anchors:
            state.unit_anchors.setdefault(k, [int((x0 + x1) / 2), int(y1 - VISION_ANCHOR_UP)])
            added.append(k)
    if added:
        logger.info("视觉检测补上徽章漏认的单位格：%s", added)


def read_unit_levels(context, image: np.ndarray) -> dict[str, int]:
    """{"x,y"(格子): 单位等级} —— 读每个等级徽章中间的数字。

    徽章里的数字就是单位等级（实测一帧里我方 52/17/6/1…，对面 52/26/23…），
    合成、觉醒、「加速成长」+5、孙膑的"每 3 次技能 +1"都体现在这个数上 ——
    **换主C 看的就是它**（2026-09-23：要控制在 3 阶左右、能升到高等级的英雄）。
    读不出来的格子不写（别拿 0 冒充"1 级"）。
    """
    badges = _detect_badges(image)
    if not badges:
        return {}
    got = _levels_by_model(image, badges)        # 2026-09-25：小模型优先（OCR 在徽章上只读对 24.7%）

    def one(item):
        key, (bx, by) = item
        txt = _ocr(context, image, (bx - 18, by - 13, 36, 26), only_digits=True)
        if not (txt and txt.isdigit() and 0 < int(txt) < 100):
            # 过百的徽章更宽（橙色箭头样式，「205」约 45px）：窄框读不全，放宽重读一次
            wide = _ocr(context, image, (bx - 28, by - 15, 56, 30), only_digits=True)
            if wide and wide.isdigit() and 100 <= int(wide) < 1000:
                return key, int(wide)
        return key, (int(txt) if txt and txt.isdigit() and 0 < int(txt) < 100 else None)

    rest = [(k, v) for k, v in badges.items() if k not in got]
    got.update({k: v for k, v in _map_parallel(one, rest) if v is not None})
    return got


# 2026-10-13 换成 level_cls_v2（tools/vision/train_level_v2.py）：
# 输入输出与 v1 相同；三位数复核测试集 216 个整数全对 10.2% → 98.1%，门槛 0.9 覆盖 82.9%、采纳全对；v1 测试集不退步。
# 回滚：文件名改回 level_cls_v1.onnx，并把 _levels_by_model 里的 LEVEL_MAX 改回 100。
LEVEL_MODEL = Path(__file__).resolve().parents[2] / "assets" / "model" / "vision" / "level_cls_v2.onnx"
LEVEL_MAX = 1000              # 模型结果只收 1 <= v < LEVEL_MAX（v1 时是 100：v1 没见过三位数，一路报 100）
LEVEL_MIN_CONF = 0.9          # 验证集定的门槛是 0.5（被采纳的准确率 ≥99.5%），线上再保守一档：宁可读不到，不要读错
_LEVEL_SESS = None


def _level_session():
    """单位等级识别小模型（tools/vision/train_level.py；数据 data/vision/level_v1，测试集人工复核）。
    复核测试集 356 个徽章：模型 100%；原 OCR 读对 24.7% / 读不到 62.9% / 读错 12.4%（1 级的细"1"几乎读不出）。
    局限：验证/测试集里没有 ≥100 级的样本（训练集只有约 10 个），三位数的准确率未验证。模型文件不在就返回 None。"""
    global _LEVEL_SESS
    if _LEVEL_SESS is None:
        try:
            import onnxruntime as ort
            _LEVEL_SESS = _gpu_session(LEVEL_MODEL, "等级") \
                if LEVEL_MODEL.exists() else False
        except Exception:                                           # noqa: BLE001
            logger.exception("等级识别模型加载失败，退回 OCR")
            _LEVEL_SESS = False
    return _LEVEL_SESS or None


def _levels_by_model(image: np.ndarray, badges: dict) -> dict[str, int]:
    """{"x,y": 等级}，只收置信度 ≥ LEVEL_MIN_CONF 的。输入与训练一致：徽章中心 48x48，RGB，/255。"""
    sess = _level_session()
    if sess is None:
        return {}
    keys, crops = [], []
    h, w = image.shape[:2]
    for key, (bx, by) in badges.items():
        if 24 <= bx < w - 24 and 24 <= by < h - 24:
            keys.append(key)
            crops.append(image[by - 24:by + 24, bx - 24:bx + 24])
    if not crops:
        return {}
    x = np.ascontiguousarray(np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
    nd, dg = sess.run(None, {"x": x})

    def softmax(a, axis):
        e = np.exp(a - a.max(axis, keepdims=True))
        return e / e.sum(axis, keepdims=True)
    pn, pd = softmax(nd, 1), softmax(dg, 2)
    out = {}
    for i, key in enumerate(keys):
        k = int(pn[i].argmax()) + 1
        ds = [int(pd[i, j].argmax()) for j in range(k)]
        conf = float(pn[i, k - 1]) * float(np.prod([pd[i, j, ds[j]] for j in range(k)]))
        v = int("".join(map(str, ds)))
        # 三位数：v1 不收（2026-09-30 第 39 局守约实际 205 级，v1 一路报 100）；v2 用三位数样本重训过，
        # 按同一个置信度门槛采纳（复核测试集 0.9 下三位数采纳全对，没把握的 17% 照旧交给 OCR 宽框）。
        # 过百后读到低值的保护在 StateExtractor._guard_levels，不受影响。
        if conf >= LEVEL_MIN_CONF and 1 <= v < LEVEL_MAX and not (k > 1 and ds[0] == 0):
            out[key] = v
    return out


EQUIP_MODEL = Path(__file__).resolve().parents[2] / "assets" / "model" / "vision" / "equip_cnt_v2.onnx"
EQUIP_MIN_CONF = 0.9          # 复核测试集 356 个：0.9 下覆盖 83%、采纳全对（label smoothing 0.05，最高置信约 0.96）
_EQUIP_SESS = None


def _equip_session():
    """装备件数小模型（tools/vision/train_equip_cnt.py）。
    复核测试集 356 个：模型 100%，霍夫圆规则 96.4%（规则常把英雄身上的蓝/金花纹当成装备图标）。
    回滚：环境变量 WZWXQ_EQUIP_MODEL=0，或删掉模型文件 → 全部走规则。"""
    global _EQUIP_SESS
    if _EQUIP_SESS is None:
        try:
            ok = EQUIP_MODEL.exists() and os.environ.get("WZWXQ_EQUIP_MODEL", "1") != "0"
            _EQUIP_SESS = _gpu_session(EQUIP_MODEL, "装备件数") if ok else False
        except Exception:                                           # noqa: BLE001
            logger.exception("装备件数模型加载失败，退回规则")
            _EQUIP_SESS = False
    return _EQUIP_SESS or None


def equip_counts_by_model(image: np.ndarray, badges: dict) -> dict[str, int]:
    """{"x,y": 件数}，只收置信度 ≥ EQUIP_MIN_CONF 的。
    裁图与训练一致：徽章中心 (bx,by) 下方 image[by+4:by+56, bx-48:bx+48]（96x52），越界补黑；RGB /255。"""
    sess = _equip_session()
    if sess is None or not badges:
        return {}
    import cv2
    pad = cv2.copyMakeBorder(image, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    keys = list(badges)
    crops = [pad[by + 4 + 64:by + 56 + 64, bx - 48 + 64:bx + 48 + 64] for bx, by in (badges[k] for k in keys)]
    x = np.ascontiguousarray(np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    out = {}
    for i, key in enumerate(keys):
        j = int(p[i].argmax())
        if float(p[i, j]) >= EQUIP_MIN_CONF:
            out[key] = min(j, MAX_EQUIPS)
    return out


def detect_equip_counts(image: np.ndarray) -> dict[str, int]:
    """{"x,y": 该格英雄身上的装备件数}。只在**准备阶段、商店关着**时才准
    （战斗阶段单位会移动，徽章和格子对不上）。

    2026-10-13：先用装备件数小模型（置信 ≥0.9），没把握的格子退回霍夫圆规则（_equip_counts_rule）。
    输出格式不变；调用方（主循环逐帧、live_match_agent 的四选一复查）都不用改。"""
    badges = _detect_badges(image)
    if not badges:
        return {}
    got = {}
    try:
        got = equip_counts_by_model(image, badges)
    except Exception:                                               # noqa: BLE001
        logger.exception("装备件数模型推理失败，本帧全走规则")
    if len(got) < len(badges):
        for k, n in _equip_counts_rule(image, badges).items():
            got.setdefault(k, n)
    return got


def _equip_counts_rule(image: np.ndarray, badges: Optional[dict] = None) -> dict[str, int]:
    """霍夫圆规则读法（2026-10-13 前的 detect_equip_counts 原样保留，作模型没把握时的兜底）。"""
    import cv2

    x0, y0, w, h = EQUIP_BOARD_ROI
    roi = image[y0:y0 + h, x0:x0 + w]
    if roi.size == 0:
        return {}
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    counts: dict[str, int] = {}
    for key, (abs_bx, abs_by) in (badges if badges is not None else _detect_badges(image)).items():
        bx, by = abs_bx - x0, abs_by - y0
        sy0, sy1 = by + 10, by + 42
        sx0, sx1 = max(0, bx - 45), min(w, bx + 45)
        if sy1 >= h or sx1 <= sx0:
            continue
        sub = roi[sy0:sy1, sx0:sx1]
        sub_gray = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
        sub_hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)
        n = sum(1 for x, y, r in _hough(sub_gray, 13, 6, 12, 14)
                if _mean_sat(sub_hsv, x, y, r) >= 70)
        counts[key] = max(counts.get(key, 0), min(n, MAX_EQUIPS))
    return counts


def _detect_badges(image: np.ndarray) -> dict[str, tuple[int, int]]:
    """{"x,y"(格子): (徽章绝对坐标)}。等级徽章 = 深色大圆，是定位单位本体的锚。"""
    import cv2

    x0, y0, w, h = EQUIP_BOARD_ROI
    roi = image[y0:y0 + h, x0:x0 + w]
    if roi.size == 0:
        return {}
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    cells = [c for row in OWN_BOARD_ROWS for c in row]
    found: dict[str, tuple[int, int]] = {}
    for bx, by, br in _hough(gray, BADGE_PARAM2, BADGE_R[0], BADGE_R[1], 30):
        if _median_val(hsv, bx, by, br) >= BADGE_MAX_V:
            continue                     # 亮的不是等级徽章（徽章是深色底 + 白色等级数字）
        abs_bx, abs_by = x0 + bx, y0 + by
        # 只认"正上方 EQUIP_BADGE_DY 处确实有一个格子"的徽章
        hit = [c for c in cells
               if abs(c[0] - abs_bx) <= EQUIP_BADGE_DX_TOL
               and abs((c[1] - abs_by) - EQUIP_BADGE_DY) <= EQUIP_BADGE_DY_TOL]
        if hit:
            # 横向、纵向偏差一起算（原来只按横向最近选）。2026-09-30 第 40 局：守约徽章 (362,372)，
            # (346,430) 横向近但纵向只差 58（不像头顶徽章），真格子 (390,502) 纵向差 130 —— 32 级记到了邻格，
            # 守约那格一直是别处的旧读数 8 级（守约实际应有几十级）。
            cell = min(hit, key=lambda c: abs(c[0] - abs_bx) / EQUIP_BADGE_DX_TOL
                       + abs((c[1] - abs_by) - BADGE_DY_PEAK) / EQUIP_BADGE_DY_TOL)
            found[cell_key(cell)] = (abs_bx, abs_by)
    return found


# ---- 觉醒进度（2026-09-24 确认：数字外面的一圈分成三段，展示的就是 3 段觉醒进度）----
# 等级徽章外圈分 左 / 上 / 右 三段，每合成一次同名卡亮一段（亮蓝），顺序 左 → 上 → 右。
# 证据：40 级孙膑 左+上 = 2/3（live_1790246871/phase_prepare_1790247589.png）。
# ⚠ 这是**过渡用的像素读法**（也是分类器训练数据的预标注器，tools/vision/build_awaken_pool.py 同一份逻辑）：
# 对背景透色、圆心偏移敏感，读不准时返回 None（不写进结果）。分类器训好后替换掉这里。
AWAKEN_SEG = {"L": (160, 215), "T": (65, 115), "R": (-35, 20)}   # 角度，0=右、90=上
AWAKEN_RING_R = (13, 14, 15, 16, 17)


def _ring_blue_frac(hsv, cx, cy, a0, a1) -> float:
    b = n = 0
    for a in range(a0, a1 + 1, 5):
        for r in AWAKEN_RING_R:
            x = int(round(cx + r * np.cos(np.radians(a))))
            y = int(round(cy - r * np.sin(np.radians(a))))
            if not (0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]):
                continue
            h, s, v = (int(t) for t in hsv[y, x])
            n += 1
            b += (92 <= h <= 118 and s >= 80 and v >= 130)
    return b / max(n, 1)


def awaken_progress_at(image: np.ndarray, cx: int, cy: int) -> Optional[int]:
    """徽章 (cx,cy) 外圈亮了几段（0~3）；读不准返回 None。每段在 ±2px 抖动内取最大（霍夫圆心常偏 1~2 像素）。"""
    import cv2
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    fr = {k: max(_ring_blue_frac(hsv, cx + dx, cy + dy, *v) for dx in (-2, 0, 2) for dy in (-2, 0, 2))
          for k, v in AWAKEN_SEG.items()}
    if any(0.15 < fr[k] < 0.4 for k in "LTR"):
        return None
    on = [fr[k] >= 0.4 for k in "LTR"]
    if on not in ([False] * 3, [True, False, False], [True, True, False], [True, True, True]):
        return None                      # 不符合 左→上→右 的填充顺序 = 背景里的蓝色，不是进度
    return sum(on)


AWAKEN_MODEL = Path(__file__).resolve().parents[2] / "assets" / "model" / "vision" / "awaken_cls_v2.onnx"
AWAKEN_CLASSES = ("0", "1", "2", "3", "na")     # 3 = 已觉醒（整圈连成一条亮环）；na = 召唤物（没有觉醒圈）
AWAKEN_MIN_CONF = 0.9
_AWAKEN_SESS = None


def _awaken_session():
    """觉醒进度分类器（tools/vision/train_awaken.py；数据 data/vision/awaken_v1，验证/测试集人工复核）。
    小 CNN，CPU 上一帧十来个徽章不到几毫秒，不占显卡。模型文件不在就返回 None（退回像素读法）。"""
    global _AWAKEN_SESS
    if _AWAKEN_SESS is None:
        try:
            import onnxruntime as ort
            _AWAKEN_SESS = _gpu_session(AWAKEN_MODEL, "觉醒进度") \
                if AWAKEN_MODEL.exists() else False
        except Exception:                                           # noqa: BLE001
            logger.exception("觉醒进度分类器加载失败，退回像素读法")
            _AWAKEN_SESS = False
    return _AWAKEN_SESS or None


def read_awaken_progress(image: np.ndarray) -> dict[str, int]:
    """{"x,y"(己方格子): 觉醒进度 0~3（3 = 已觉醒）}，只收读得准的。
    优先用分类器（置信度 ≥ AWAKEN_MIN_CONF 才收，召唤物 na 不收）；没有模型才用像素读法。"""
    badges = _detect_badges(image)
    sess = _awaken_session()
    if sess is None:
        out = {}
        for key, (bx, by) in badges.items():
            p = awaken_progress_at(image, bx, by)
            if p is not None:
                out[key] = p
        return out
    keys, crops = [], []
    h, w = image.shape[:2]
    for key, (bx, by) in badges.items():
        if 24 <= bx < w - 24 and 24 <= by < h - 24:
            keys.append(key)
            crops.append(image[by - 24:by + 24, bx - 24:bx + 24])    # 和训练一致：徽章中心 48x48，RGB
    if not crops:
        return {}
    x = np.ascontiguousarray(np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
    logits = sess.run(None, {"x": x})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    prob = e / e.sum(1, keepdims=True)
    out = {}
    for key, pr in zip(keys, prob):
        c = int(pr.argmax())
        if pr[c] >= AWAKEN_MIN_CONF:
            if AWAKEN_CLASSES[c] != "na":
                out[key] = int(AWAKEN_CLASSES[c])
            continue
        # 分类器没把握 → 退回像素读法（它在圈看得清时很准，读不准会自己返回 None）。
        # 两路互补：实测证据帧上"4 级、亮 1 段"的曜分类器只有 0.76，像素读法读对了。
        bx, by = badges[key]
        p = awaken_progress_at(image, bx, by)
        if p is not None:
            out[key] = p
    return out


AUCTION_BIDDER_DY = (100, 230)     # 拍品名 -> 出价人列表的纵向范围
AUCTION_BIDDER_HALF_W = 110
AUCTION_PITCH = 205                     # 拍品卡中心间距（1280 宽实测：125/327/532/737/942/1148）


def auction_slots(xs: list) -> list:
    """由读到的拍品名字 x 推出全部拍品位：等距 AUCTION_PITCH、整排以 640 为中心。读到 <2 个时推不出（返回 []）。"""
    xs = sorted(xs)
    if len(xs) < 2:
        return []
    p = AUCTION_PITCH
    n = int(round((xs[-1] - xs[0]) / p)) + 1
    lo = xs[0]
    for _ in range(6):                        # 不居中 → 缺的在一侧，往偏离中心的反方向补
        c = lo + (n - 1) * p / 2
        if abs(c - 640) <= p * 0.3 or n >= 6:
            break
        if c > 640:
            lo -= p
        n += 1
    return [int(lo + i * p) for i in range(n)]


AUCTION_FACTION_WORDS = {"逐鹿", "河洛", "三分之地", "三分", "日落海", "大河流域", "大河", "无阵营"}
AUCTION_MINE_GREEN = 60            # 判"这行是我"的绿像素下限（实测我方 178，别人 0）


def mark_own_bids(context, image, lots) -> None:
    """标出哪些拍品**当前由我们领先** —— 每个拍品下方最上面那一行就是最高价者，
    而**我方昵称是绿色的**（实测该行绿像素 178，其余 0）。

    为什么必须靠观测：原来防自抬用的是我们自己记的 `_bid_piece/_bid_price`，
    而**拍品索引会随拍品售出、玩家淘汰而错位**，一错位就对着自己加价、白白抬高价格
    （2026-09-22 发现）。画面上本来就写着谁领先，没理由去猜。
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for lot in lots:
        if lot.x < 0 or lot.y < 0:
            continue
        roi = (max(0, lot.x - AUCTION_BIDDER_HALF_W), lot.y + AUCTION_BIDDER_DY[0],
               AUCTION_BIDDER_HALF_W * 2, AUCTION_BIDDER_DY[1] - AUCTION_BIDDER_DY[0])
        rows = [(b[1], t, b) for t, b in _ocr_boxes(context, image, roi)
                if len(t.strip()) >= 2]
        if not rows:
            continue
        _, text, box = min(rows)                 # 最上面那一行 = 当前最高价者
        x, y, w, h = box
        patch = hsv[y:y + h, x:x + w]
        if patch.size == 0:
            continue
        green = int(((patch[..., 0] > 35) & (patch[..., 0] < 85)
                     & (patch[..., 1] > 90) & (patch[..., 2] > 90)).sum())
        lot.top_bidder = "我" if green >= AUCTION_MINE_GREEN else text.strip()


def _maple_ready(image: np.ndarray, cx: int, cy: int) -> bool:
    """棋手技能牌**现在能不能（重新）绑人** —— 看卡面是亮的还是灰的。

    绑定只有把被绑的英雄卖掉才能解除，而那个英雄可能被系统卖掉、或觉醒回手，
    光靠自己记账会漏。卡面自己会说：能绑时整张卡亮着，绑住时是灰的。
    """
    import cv2

    y0 = max(0, cy + MAPLE_ART_DY)
    patch = image[y0:y0 + MAPLE_ART_H,
                  max(0, cx - MAPLE_ART_HALF_W):cx + MAPLE_ART_HALF_W]
    if patch.size == 0:
        return True                      # 取不到样就别拦着，交给记账判断
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return (float(hsv[..., 1].mean()) >= MAPLE_READY_SAT
            and float(hsv[..., 2].mean()) >= MAPLE_READY_VAL)


def _player_hp(text: str) -> int:
    """把血条那一格的 OCR 文本解析成血量。读不出返回 -1。

    **上限 60 ⇒ 最多两位数**，读到三位数必是尾部多粘了一位
    （实测 '608'=60、'548'=54 —— 旁边的图标被当成数字）。
    这个约束比调 OCR 参数好使：它来自游戏规则，不会因为版本换皮失效。
    """
    digits = re.sub(r"\D", "", text or "")
    if not digits:
        return -1
    if len(digits) >= 3:
        digits = digits[:2]
    value = int(digits)
    return value if 0 <= value <= MAX_PLAYER_HP else -1


def _green_mask(hsv: np.ndarray) -> np.ndarray:
    """我方昵称 / 血量数字的绿色（HSV，OpenCV 色相 0~180）。"""
    return ((hsv[..., 0] > 35) & (hsv[..., 0] < 85)
            & (hsv[..., 1] > 90) & (hsv[..., 2] > 90))


def _green_runs(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """按列切绿色串：每列 ≥2 个绿像素算有字，列间空隙 > gap 就断开。返回 [(起列, 止列)]。"""
    xs = np.flatnonzero(mask.sum(axis=0) >= 2)
    if not xs.size:
        return []
    cuts = np.flatnonzero(np.diff(xs) > gap)
    starts = [xs[0]] + [xs[c + 1] for c in cuts]
    ends = [xs[c] for c in cuts] + [xs[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def own_hp_row(image_rgb: np.ndarray) -> tuple[Optional[int], Optional[tuple[int, int, int, int]], list[int], bool]:
    """六人列表里哪一行是自己 + 读自己血量的 OCR 框。**纯函数**，离线可测。

    返回 (行号或 None, OCR 框或 None, 每行得分, 是否靠绿色数字认出)。
    得分 = 该行"绿色数字串"的绿像素数 + "像一行字"的绿昵称像素数。绿色数字串的要求：
      - 在 HP_DIGIT_BAND 里、最左列 ≤ HP_DIGIT_START_MAX（排除数字后面的绿色 ○ 图标）；
      - 绿像素纵跨 ≥ HP_DIGIT_MIN_TALL 行（排除彩虹拖尾这种横条特效）；
      - 宽 ≤ HP_DIGIT_MAX_W、填充率 ≤ HP_DIGIT_MAX_FILL（排除卡牌大图的绿色立绘色块）。
    第一名 < PLAYER_NAME_GREEN_MIN 或不到第二名的 OWN_ROW_MARGIN 倍 → None（**调用方本帧不更新血量**）。
    最后一个返回值 False = 只靠昵称认出（实测 1 血时数字变白）：调用方必须再用 OCR 核对昵称，
    因为卡牌大图 / 英雄详情面板盖住列表时，昵称带里也会出现"像字"的绿色。
    """
    import cv2

    # 只转左侧列表这一块（从原点裁，坐标不变）
    hsv = cv2.cvtColor(image_rgb[:PLAYER_ROWS_Y[-1] + 40, :240], cv2.COLOR_RGB2HSV)
    h_img, w_img = hsv.shape[:2]
    scores: list[int] = []
    spans: list[Optional[tuple[int, int]]] = []
    cut_x: list[Optional[int]] = []
    bx, bdy, bw, bh = HP_DIGIT_BAND
    nx, ndy, nw, nh = ROI_PLAYER_NAME
    for y in PLAYER_ROWS_Y:
        score, span, cut = 0, None, None
        y0, y1 = max(0, y + bdy), min(h_img, y + bdy + bh)
        band = _green_mask(hsv[y0:y1, bx:min(w_img, bx + bw)])
        runs = _green_runs(band, HP_DIGIT_GAP) if band.size else []
        for a, b in runs:
            if bx + a > HP_DIGIT_START_MAX:
                cut = bx + a          # 数字后面的绿色 ○：读血量的框到它前面为止（不然 "1○" 读成 10）
                break
            sub = band[:, a:b + 1]
            tall = int((sub.sum(axis=1) > 0).sum())
            if (tall >= HP_DIGIT_MIN_TALL and b - a + 1 <= HP_DIGIT_MAX_W
                    and float(sub.mean()) <= HP_DIGIT_MAX_FILL and span is None):
                score += int(sub.sum())
                span = (bx + a, bx + b)
        name = hsv[max(0, y + ndy):max(0, y + ndy + nh), nx:nx + nw]
        if name.size:
            nm = _green_mask(name)
            ncols = np.flatnonzero(nm.sum(axis=0) > 0)
            nrows = int((nm.sum(axis=1) > 0).sum())
            # 昵称要"像一行字"：左对齐从 x≈125 起、字高 6~20 行。
            # 挡掉的东西（实测）：右侧"等级提升"弹窗里绿头发的头像（从 x≥150 起）、立绘铺满整条带。
            if ncols.size and nx + int(ncols[0]) <= HP_DIGIT_START_MAX and 6 <= nrows <= 20:
                score += int(nm.sum())
        scores.append(score)
        spans.append(span)
        cut_x.append(cut)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    best, top = order[0], scores[order[0]]
    second = scores[order[1]] if len(order) > 1 else 0
    if top < PLAYER_NAME_GREEN_MIN or second * OWN_ROW_MARGIN > top:
        return None, None, scores, False
    y = PLAYER_ROWS_Y[best]
    span = spans[best]
    if span is not None:
        # 框只包住绿数字，左右各放 4 像素：避开左边的头像边、右边的图标
        x0 = max(0, span[0] - 4)
        x1 = span[1] + 5
    else:
        # 只靠昵称认出 = 带昵称排版（数字从 x≈125 起、两位数到 ≈152）：框到 x=156 为止，
        # 不罩后面的彩虹拖尾（实测 1 血 + 彩虹，旧 70 宽的框读成 21，44 宽读 1）；再截到绿色 ○ 前
        x0 = ROI_PLAYER_HP[0]
        x1 = ROI_PLAYER_HP[0] + HP_NAMED_W
    if cut_x[best] is not None:
        x1 = min(x1, cut_x[best] - 2)
    return best, (x0, y, max(8, x1 - x0), ROI_PLAYER_HP[3]), scores, span is not None


def nickname_matches(text: str, nickname: str) -> bool:
    """OCR 读到的昵称是否是自己（忽略大小写和空白；允许 OCR 错 1 个字符）。"""
    a = re.sub(r"\s", "", text or "").lower()
    b = re.sub(r"\s", "", nickname or "").lower()
    if not a or not b:
        return False
    if b in a:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    # 编辑距离 ≤ 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 1


_PLAYER_PROFILE = Path(__file__).resolve().parents[2] / "data" / "player_profile.json"


def own_nickname() -> str:
    """账号昵称（data/player_profile.json 的 nickname；换账号改这里）。读不到返回空串 = 不启用昵称核对。"""
    try:
        import json
        return str(json.loads(_PLAYER_PROFILE.read_text(encoding="utf-8")).get("nickname", "")).strip()
    except Exception:
        return ""


def read_lord_level(image: np.ndarray) -> int:
    """棋手等级 = 头像徽章上的亮点个数。读不到返回 -1。

    自检方式（无需人工标注）：**同一局内等级只增不减**。
    34 次运行 189 个 prepare 帧实测零违例（两个"下降"的目录经查是一次运行打了两局，
    回合号 8→1，等级归 1 是真的）。
    """
    import cv2

    x, y, w, h = ROI_LORD_LEVEL
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return -1
    # **通道顺序在这里无所谓**：实时截图是 RGB（见 _badge_like），离线 _imread 是 BGR。
    # 亮点是白的，灰度两种顺序都接近 255；饱和度 S=(max-min)/max 本身与通道序无关。
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, LEVEL_DOT_THRESHOLD, 255, cv2.THRESH_BINARY)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lo, hi = LEVEL_DOT_AREA
    keep = [i for i in range(1, count) if lo <= stats[i, cv2.CC_STAT_AREA] <= hi]
    dots = len(keep)
    return dots if 1 <= dots <= 6 else -1


def detect_board_units(context, image) -> set:
    """返回**己方半场被占用**的落点集合。看画面得出，不依赖任何记账。

    做法：霍夫圆找候选 → 距离门槛（必须落在己方落子点正上方）→ 像素验证。
    **不跑 OCR**：占用判断只需要知道"那儿有个徽章"，徽章上的数字是几无关紧要。
    第一版用 48 个滑窗 OCR 扫数字，约 800ms；换成这条路后约 7ms，**快 100 倍**，
    而且不受 OCR 对小字号的检测抖动影响。

    `context` 保留在签名里只为兼容调用方，这条路径用不到它。

    **装备四选一弹窗开着时不要信这里的结果**：那个弹窗本身就是 4 个深色圆环绕英雄，
    几何上和徽章分不开。试过用"圆的聚集程度"去猜弹窗，判据不可靠（实测没打中），
    所以改由**调用方**抑制——主循环本来就确定地知道弹窗在不在（equip_popup_until），
    用确定的事实比在这里猜可靠。
    """
    import cv2
    if not BOARD_OBS_ENABLED:
        return set()
    ax, ay, aw, ah = ROI_ARENA
    if image is None or image.shape[0] < ay + ah or image.shape[1] < ax + aw:
        return set()
    roi = image[ay:ay + ah, ax:ax + aw]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    found = cv2.HoughCircles(cv2.medianBlur(gray, 3), cv2.HOUGH_GRADIENT,
                             dp=1, minDist=30, param1=100, param2=22,
                             minRadius=10, maxRadius=22)
    if found is None:
        return set()
    drops = OWN_BOARD_DROPS_FRONT + OWN_BOARD_DROPS_BACK
    occupied: set = set()
    for c in found[0]:
        cx, cy, r = ax + int(c[0]), ay + int(c[1]), int(c[2])
        py = cy + BADGE_DROP_DY
        near = min(drops, key=lambda p: (p[0] - cx) ** 2 + (p[1] - py) ** 2)
        if (near[0] - cx) ** 2 + (near[1] - py) ** 2 > BADGE_SNAP_MAX ** 2:
            continue
        if not _badge_like(image, cx, cy, r):
            continue
        occupied.add(near)
    return occupied


def _imread(path) -> Optional[np.ndarray]:
    """cv2.imread 在 Windows 上读不了中文路径，用 imdecode+fromfile 兜底。"""
    import cv2

    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


# 像素比对格子：必须与实际 OCR 的 ROI 一一对应，**不能用包围盒**
# （包围盒会跨过卡面立绘/玩家头像这些会动的东西，命中率直接归零，实战验证过）
SHOP_NAME_CELLS = [(x0 + 30, ROI_SHOP_NAME_Y[0], 190, ROI_SHOP_NAME_Y[1])
                   for x0 in SHOP_CARD_X]
# 血量格子：从 x=96 起（紧凑排版的首位数字在 100 附近，原来从 124 起看不到首位变化），
# 并包住昵称带（绿昵称跟着重排移动，重排了就必须重认行）。
PLAYER_HP_CELLS = [(HP_DIGIT_BAND[0], y + ROI_PLAYER_NAME[1],
                    ROI_PLAYER_HP[0] + ROI_PLAYER_HP[2] - 8 - HP_DIGIT_BAND[0],
                    ROI_PLAYER_HP[3] - 4 - ROI_PLAYER_NAME[1])
                   for y in PLAYER_ROWS_Y]


# ==================== 阶段判据表 ====================
# phase -> [(roi, 正则), ...]，任一命中即判定为该阶段。
# PHASE_ORDER 是优先级顺序，与重构前 detect_phase 的 if 链完全一致，不要随意调换。

# 拍卖界面顶部横幅的**所有已知说法**。只认"预览|拍卖|开启"是不够的：
# 出价阶段会变成「已为最高出价者，不可切换」，一个关键词都不含 ——
# 2026-09-23 实测，那一帧阶段被误判成 prepare，N选1 检测器随即把 5 个拍品
# 当成候选、连点 3 次（等于对着拍品乱点）。证据帧 phase_card_pick_1790129957.png。
# 拍到一件之后顶部变成「已获得拍品」（2026-09-23 证据帧 fixtures/auction_won.png）
AUCTION_BANNER_RE = "预览|拍卖|开启|出价|切换|竞价|拍品"
# 拍卖界面右下角的圆形「出价」按钮（选中拍品后出现）。顶部横幅有时被遮得很暗、一个字都读不出
# （2026-09-23 证据帧 fixtures/auction_dim_banner.png），这时靠它认拍卖，也靠它把拍卖排除出 N 选 1。
ROI_AUCTION_BID_BTN = (1025, 600, 105, 75)

PHASE_PREDICATES: dict[str, list[tuple[tuple[int, int, int, int], str]]] = {
    # 设置面板盖在一切之上，必须第一个查
    "settings": [(ROI_SETTINGS_TABS, "声音")],
    # 棋手详情面板：盖满屏幕的模态，必须优先于底层阶段识别
    "lord_panel": [(ROI_LORD_PANEL, "技能")],
    "settle": [(ROI_SETTLE, "尚未结束|第.{1,3}名|RANKING"),
               (ROI_SETTLE_BTNS, "立即结算|现在退出")],
    "match_found": [(ROI_MATCH_FOUND, "匹配成功")],
    "settle_final": [(ROI_PLAY_AGAIN, "再来一局")],
    "lobby": [(ROI_START_MATCH, "开始匹配")],
    "mode_lobby": [(ROI_START_GAME, "开始游戏")],
    # 底部那行：刷新还剩几次写「全局N次」，用光了写「刷新次数已用完」。
    # **两种都要认**：只认"全局"的话，我们自己刷新一次之后就再也认不出这个界面，
    # 刷出来的三选一挂在那儿没人选，一路 unknown 到对局结束（第十二局回合8 实测）。
    # 用「全局|次数|已用」而不是「刷新」：这行字号小，**OCR 很不稳**，
    # 同一块区域两次分别读成「刷新次数已用宗」和「剧新次数已用宗」——
    # "刷"和"完"都会认错，只有"次数""已用"两次都对。
    # 标签判据要**至少 2 个**：选完天赋弹出的**放大详情**（左边一张选中的卡 + 右边它送的牌）
    # 也带一个「天赋」标签，只要 1 个就会被认成三选一、一直去点（2026-09-23 第 6 回合实测）。
    # 真三选一有 3 张卡：历史帧 ≥2 个的占 63%，其余靠底部「全局N次」那条兜。
    "talent": [(ROI_TALENT_MARKER, "全局|次数|已用"), (ROI_TALENT_TAGS, (TALENT_TAG_RE, 2))],
    # 「强化」后面**不能跟"牌"**：战斗中弹的「任务达成」动画卡，描述写着"枫叶舞挑选1张强化牌"，
    # 正好落在这条带上 —— 历史 52 次"棋手特性"里 12 次是它（2026-09-23 核对），
    # 候选再从右侧天赋列里凑出一个「荆」，于是去点天赋列。真三选一的标签是「枫舞强化」。
    "lord_boost": [(ROI_LORD_BOOST_MARKER, r"强化(?!牌)")],
    "auction": [(ROI_AUCTION_TITLE, AUCTION_BANNER_RE), (ROI_AUCTION_BID_BTN, "出价")],
    "hero_pick": [(ROI_HERO_PICK_MARK, "选择棋手")],
    # 第三条判据是给"商店关闭时"用的：关店后"刷新"和"当前概率"都不可见，
    # 但右侧"准备"按钮一直在（2026-09-22 实帧验证 '准备' @ (1186,535)）。
    # 没有它，一关商店阶段就掉进 unknown，兜底逻辑会开始乱点。
    "prepare": [(ROI_FREE_REFRESH, "刷新"), (ROI_SHOP_PROB, "当前概率"),
                (ROI_READY_BTN, "准备|开战")],
    "battle": [(ROI_BATTLE_STAT, "战斗统计")],
    # "挑选搭配" 是选完棋手之后、正式读条之前的**换装展示页**（约 5 秒，顶部居中计时）。
    # 它没有任何按钮，也没有"本局禁用"，以前落进 unknown，6 秒兜底就开始盲点棋盘
    # （2026-09-22 实战复现）。归到 loading = 什么都不做，等它自己过去。
    "loading": [(ROI_LOADING_BAN, "本局禁用"), (ROI_HERO_PICK_MARK, "挑选搭配")],
}
PHASE_ORDER = list(PHASE_PREDICATES)

# 模态阶段：能叠加在任何底层阶段之上（天赋弹窗不选就一直在；出局结算随时可能弹），
# 所以每帧都必须先查，不能被"上一帧阶段"的短路跳过。
MODAL_PHASES = ("settings", "lord_panel", "settle", "talent", "lord_boost")


# ==================== 性能埋点 ====================
# 一帧的成本 ≈ 截图 + 阶段识别 + 各项提取，而后两者几乎全是 OCR 往返。
# 默认关闭（if 判断的开销可忽略），live_match_agent 加 --perf 打开。


# **固定的字形误读**（日志里反复出现、模糊匹配兜不住的）：生僻字被拆成两个常见字，
# difflib 相似度只剩 0.4。「芈」→「不半」：第 14 局（2026-09-24）觉醒芈月回手后，
# 抓牌校验连续 9 次读成「不半月」→"手里找不到芈月"→ 不拖，觉醒主力一回合没上场。
# 只收**每条都在日志里核对过**的；拿不准的（如「鱼班」可能是鲁班七号/鲁班大师）不收。
OCR_NAME_ALIASES = {
    "不半月": "芈月",
    "金军列陈": "全军列阵",
    "业热": "亚瑟",          # 2026-09-25 第四轮第 4 局手牌条（N 选 1 刚拿到的亚瑟）
    "奔星": "弈星",          # 2026-10-11 识别审查：商店里「弈」读成「奔」，全库匹配落到天赋「逐星」
}

class _Perf:
    """按阶段累计耗时 + 统计 OCR 调用次数。"""

    def __init__(self) -> None:
        self.enabled = False
        self.stages: dict[str, float] = {}
        self.ocr_calls = 0
        self.ocr_ms = 0.0

    def reset(self) -> None:
        self.stages = {}
        self.ocr_calls = 0
        self.ocr_ms = 0.0

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        before = self.ocr_calls
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000
            self.stages[name] = self.stages.get(name, 0.0) + ms
            key = name + "#ocr"
            self.stages[key] = self.stages.get(key, 0.0) + (self.ocr_calls - before)

    def note_ocr(self, ms: float) -> None:
        self.ocr_calls += 1
        self.ocr_ms += ms

    def summary(self) -> str:
        """按耗时降序列出各阶段：名称 耗时ms(OCR次数)。"""
        items = [(k, v) for k, v in self.stages.items() if not k.endswith("#ocr")]
        items.sort(key=lambda kv: -kv[1])
        parts = [f"{k} {v:.0f}ms({int(self.stages.get(k + '#ocr', 0))}次)"
                 for k, v in items]
        return (f"OCR {self.ocr_calls}次/{self.ocr_ms:.0f}ms | " + " ".join(parts))


PERF = _Perf()


# ==================== OCR 结果缓存（按像素） ====================
# 2026-09-23 实测（tools/bench_prepare_frame.py，7 张真实准备帧）：整帧感知 2.6s，
# **同一帧原样再跑一遍还要 1.9s** —— 画面一个像素都没变，大部分 OCR 照样重做。
# 准备阶段的文字区域（手牌卡名、能量、回合、按钮）绝大多数帧都不变，
# 而每帧最多只做一个动作 —— 这 2~3 秒就是"60 秒准备时间里做不完事、人没上场"的根因
# （决策速度很慢，经常超过 60s 的准备时间）。
# 键 = (ROI, 参数, 这块 ROI 的像素哈希)。**同样的像素必然得到同样的结果**，没有过期问题；
# 以前各处零散做的"区域像素缓存"（商店/血量）是同一个思路，这里收到最底层统一做。
_OCR_CACHE: dict = {}
_OCR_CACHE_MAX = 4096


def _ocr_key(kind: str, image, roi, extra) -> Optional[tuple]:
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        return None
    patch = image[max(0, y):y + h, max(0, x):x + w]
    if patch.size == 0:
        return None
    return (kind, x, y, w, h, extra, patch.shape, hash(patch.tobytes()))


def _ocr_cache_put(key, value) -> None:
    if len(_OCR_CACHE) >= _OCR_CACHE_MAX:
        # 简单粗暴：满了就清掉最早插入的一半（dict 保持插入顺序）
        for k in list(_OCR_CACHE)[:_OCR_CACHE_MAX // 2]:
            _OCR_CACHE.pop(k, None)
    _OCR_CACHE[key] = value


def _ocr(context, image, roi: tuple[int, int, int, int],
         expected: str | None = None, only_digits: bool = False) -> str:
    """在指定 ROI 上 OCR，返回文本；expected 为正则过滤（命中其中一段）。"""
    key = _ocr_key("ocr", image, roi, (expected, only_digits))
    if key is not None and key in _OCR_CACHE:
        return _OCR_CACHE[key]
    text = _ocr_uncached(context, image, roi, expected, only_digits)
    if key is not None:
        _ocr_cache_put(key, text)
    return text


def _ocr_uncached(context, image, roi, expected=None, only_digits=False) -> str:
    if _onnx_ocr.enabled():
        # 绕过 MaaFW 直接跑同一套模型（见 onnx_ocr.py 顶部的实测数字）
        text = _onnx_ocr.OnnxOcr.get().text(image, roi, expected=expected)
        if only_digits:
            text = re.sub(r"\D", "", text)
        return text.strip()
    override: dict[str, Any] = {
        "OCR": {"recognition": "OCR", "roi": list(roi)}
    }
    if expected:
        override["OCR"]["expected"] = expected
    if PERF.enabled:
        _t0 = time.perf_counter()
        detail = context.run_recognition("OCR", image, pipeline_override=override)
        PERF.note_ocr((time.perf_counter() - _t0) * 1000)
    else:
        detail = context.run_recognition("OCR", image, pipeline_override=override)
    if detail is None or not detail.hit:
        return ""
    text = ""
    if getattr(detail, "best_detail", None) is not None:
        text = detail.best_detail.text or ""
    elif getattr(detail, "all_results", None):
        text = "".join(r.text for r in detail.all_results)
    if only_digits:
        text = re.sub(r"\D", "", text)
    return text.strip()


def _ocr_boxes(context, image, roi: tuple[int, int, int, int]) -> list[tuple[str, list]]:
    """OCR 并返回 ROI 内全部 (文本, 绝对坐标box) 对。（按像素缓存，见 _OCR_CACHE）"""
    key = _ocr_key("boxes", image, roi, None)
    if key is not None and key in _OCR_CACHE:
        return [(t, list(b)) for t, b in _OCR_CACHE[key]]
    out = _ocr_boxes_uncached(context, image, roi)
    if key is not None:
        _ocr_cache_put(key, [(t, list(b)) for t, b in out])
    return out


def _ocr_boxes_uncached(context, image, roi) -> list[tuple[str, list]]:
    if _onnx_ocr.enabled():
        return _onnx_ocr.OnnxOcr.get().boxes(image, roi)
    _t0 = time.perf_counter() if PERF.enabled else 0.0
    detail = context.run_recognition(
        "OCR",
        image,
        pipeline_override={"OCR": {"recognition": "OCR", "roi": list(roi)}},
    )
    if PERF.enabled:
        PERF.note_ocr((time.perf_counter() - _t0) * 1000)
    if detail is None or not detail.hit:
        return []
    results = getattr(detail, "all_results", None) or []
    return [(r.text, list(r.box)) for r in results
            if getattr(r, "text", "") and getattr(r, "box", None) is not None]


def filter_hp(last: int, hp: int, reject: tuple[int, int],
              first: bool = False, round_no: int = 0,
              rounds_elapsed: int = 99, before: int = MAX_PLAYER_HP,
              held_s: float = 99.0) -> tuple[int, tuple[int, int]]:
    """血量读数的合法性判据，**纯函数**，方便离线测（不用开模拟器）。

    输入上一次认定的血量 `last`、本帧读数 `hp`、被拒计数 `reject=(值, 连读次数)`，
    返回 (采信后的血量, 新的被拒计数)。四条不变量，每条都是被实战打出来的：

    1. **0 血不可能还在打** —— 0 = 已淘汰 = 在结算页了。
    2. **只减不增** —— 这个游戏没有回血。
    3. **首位数字被吃掉**（57→7、48→8）是最常见的 OCR 失效，它同时满足第 2 条。
    4. **单回合掉血有上限**（HP_MAX_DROP）—— 正常 3~10。
       超上限先不信；但同一个值连读 HP_REJECT_GIVE_IN 次就认，
       否则真掉了大血会被永久冻在旧值。
       **"连读就认"也要讲时间**（2026-09-23 实战）：52 被持续读成 2，连读 3 次后认了 ——
       可一回合掉 48 血是不可能的。现在只放行 `HP_MAX_DROP × 距上次认定隔了几个回合`
       以内的跌幅（`rounds_elapsed`；回合未知时给 99，保持原行为）。
    5. **认错了要能改回来**：第 2 条"只减不增"一旦认下一个偏低的误读，
       后面所有正确读数都会被它拒掉（那一局从第 6 回合到结束一直以为自己只剩 2 血，
       真实是 52，决策层全程按"血量危险"在打）。往上的读数**同一个值连读
       HP_REJECT_GIVE_IN 次、且不超过掉血之前的值（`before`）**就纠正回来 ——
       偶发的往上误读仍然挡得住。
    """
    if not (0 < hp <= MAX_PLAYER_HP):
        return last, reject
    if first:
        # **本局第一次读到血量**：基准还是初始值 60，而我们可能是中途接手
        # （离线回归帧就是第 5 回合、血 10）。这时任何"跌幅"判据都不成立，
        # 只要数字本身合法就采信，否则会一直卡在 60 —— 决策层以为自己满血。
        # **但首读也要讲得通**：每回合最多掉 HP_MAX_DROP，所以第 N 回合最少还剩
        # 60 - N*上限。2026-09-23 实战：第 1 回合没读到血，第 2 回合首读到 30 就直接采信，
        # 而开局两回合不可能掉 30 血 —— 首读的豁免不该连"物理上限"一起豁免。
        # 回合号未知（0）时不设限，保持中途接手时的原行为。
        # 用 HP_FIRST_READ_PER_ROUND 而不是 HP_MAX_DROP：后者(25)是"单回合物理上限"，
        # 拿它当首读的门槛太松（第 2 回合的下限会算成 10，30 照样放行）。
        # 实测掉血节奏是前期 3~8、后期 10~25，取 12 作平均上限。
        # 首读被拒没有代价：后面"同一超限值连读 3 次就认"会在几帧内收敛到真值。
        floor = MAX_PLAYER_HP - max(round_no, 0) * HP_FIRST_READ_PER_ROUND
        if round_no > 0 and hp < floor:
            logger.info("本局首次血量读数 %d 在第 %d 回合讲不通（最少也该有 %d），不采信",
                        hp, round_no, floor)
            return last, reject
        # **首读也要连续两帧一致**（2026-09-23 第十三局：战斗中重启后首读 15 —— 我们那一行被
        # "LIVE FEED"图标和伞挡住，真值 57 —— 第 5 回合下限算出来是 0，照单全收）
        if reject != (hp, 1):
            return last, (hp, 1)
        return hp, (0, 0)
    if hp > last:
        val, n = reject
        n = n + 1 if val == hp else 1
        if hp <= before and n >= HP_REJECT_GIVE_IN + 1 and held_s >= HP_HOLD_S:
            logger.warning("血量 %d 连读 %d 次、且不超过掉血前的 %d —— 之前认下的 %d 是误读，纠正回来",
                           hp, n, before, last)
            return hp, (0, 0)
        return last, (hp, n)
    if len(str(last)) > len(str(hp)) and str(last).endswith(str(hp)):
        # 首位被吃（旧裁切框常见）。10-22 第 937 局：真实 19→9 也长这样，被永远拒掉（26 次），血量冻在 19 →
        # 同一个值连读够次数、停够时间、跌幅在物理上限内就认（新识别按整串绿色数字框，首位很少再被切）
        val, n = reject
        n = n + 1 if val == hp else 1
        if n >= HP_REJECT_GIVE_IN and held_s >= HP_HOLD_S and last - hp <= HP_MAX_DROP * max(1, rounds_elapsed):
            logger.warning("血量 %d -> %d 像首位被吃，但连读 %d 次都是它、跌幅讲得通，认了", last, hp, n)
            return hp, (0, 0)
        return last, (hp, n)
    if last - hp > HP_MAX_DROP:
        val, n = reject
        n = n + 1 if val == hp else 1
        if n < HP_REJECT_GIVE_IN or held_s < HP_HOLD_S:
            return last, (hp, n)
        if last - hp > HP_MAX_DROP * max(1, rounds_elapsed):
            # 连读再多次也不认：这段时间里物理上掉不了这么多血（多半是首位数字被吃掉，52->2）
            return last, (hp, n)
        logger.warning("血量 %d -> %d 跌幅超上限，但连读 %d 次都是它，认了", last, hp, n)
        return hp, (0, 0)
    if last - hp > HP_CONFIRM_DROP:
        # 10-22 第 948 局 R11：23 → 1 只读到一帧就认了（≤ 上限 25 不用确认），之后真值 17 被当成"往上跳"拒掉；
        # 大跌幅要同一个值连读 2 次才认（真掉血会持续好几帧，一两帧的误读挡掉）
        val, n = reject
        n = n + 1 if val == hp else 1
        if n < 2:
            return last, (hp, n)
    return hp, (0, 0)


def _ocr_white_text(context, image, roi: tuple[int, int, int, int]) -> str:
    """把 ROI 里的**白色文字**抠出来再 OCR —— 卡名压在高亮立绘上时的补救手段。

    为什么需要：卡名是白字，OCR 在**亮底**上会直接在检测阶段丢掉它。
    2026-09-23 实测：商店第 2 格的「墨子」，名字正好压在一把金色的剑上，
    连着好几帧读成空 → `piece=None` → 这张牌在决策层里根本不存在。
    同一帧另外三张（苏烈/蒙恬/廉颇）都正常，所以这**不是 ROI 问题，是对比度问题**。

    判据用 HSV 而不是灰度亮度：金色的剑同样很亮（V 高），但**饱和度高**；
    白字饱和度低。所以 `S < 70 且 V > 185` 能把字留下、把亮底去掉（实测四张全干净）。
    输出成黑字白底交给 OCR。

    只在第一次读失败时才调用，稳态不额外花钱。
    """
    import cv2
    x, y, w, h = roi
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return ""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < WHITE_TEXT_SAT_MAX) & (hsv[:, :, 2] > WHITE_TEXT_VAL_MIN))
    patched = image.copy()
    patched[y:y + h, x:x + w] = cv2.cvtColor(
        (255 - mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    return _ocr(context, image=patched, roi=roi)


def _read_white_digits(context, image, roi: tuple[int, int, int, int],
                       min_h: int = 14, min_area: int = 25) -> int:
    """只认 ROI 里**白色的数字**：先按颜色抠出白色笔画，裁到笔画的外框，黑字白底再识别。

    为什么（2026-09-23，换成直接跑模型之后暴露）：右下角能量数字左边紧挨着一个红色闪电图标，
    小区域里检测不出文字框时整块拿去认，图标就被认成数字 —— 7 读成 37、0 读成 30、1 读成 51；
    紧 ROI 又会切到图标边缘，3 读成 33、9 读成 19。历史 330 帧里两种读法不一致的有 50 帧。
    图标是**红的**、数字是**白的**，按颜色分开最稳：随机 60 帧 + 那 50 帧难例，肉眼核对 110/110 全对。
    `min_h` 排掉图标里那道矮的闪电和零星亮点。读不出返回 -1。
    """
    import cv2
    x, y, w, h = roi
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return -1
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] < WHITE_TEXT_SAT_MAX) & (hsv[:, :, 2] > WHITE_TEXT_VAL_MIN)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    keep = [k for k in range(1, n) if st[k][3] >= min_h and st[k][4] >= min_area]
    if not keep:
        return -1
    x0 = min(st[k][0] for k in keep); y0 = min(st[k][1] for k in keep)
    x1 = max(st[k][0] + st[k][2] for k in keep); y1 = max(st[k][1] + st[k][3] for k in keep)
    ink = np.isin(lab, keep)
    patched = image.copy()
    patched[y:y + h, x:x + w] = np.where(ink[:, :, None], 0, 255).astype(np.uint8)
    pad = 6
    tight = (max(0, x + x0 - pad), max(0, y + y0 - pad),
             (x1 - x0) + pad * 2, (y1 - y0) + pad * 2)
    digits = _ocr(context, patched, tight, only_digits=True)
    return int(digits) if digits and len(digits) <= 2 else -1


PRICE_BADGE_MIN_RATIO = 0.22


ROI_PICK_QUOTA = (470, 370, 340, 50)          # 「请选择1/2张卡牌」横幅（多选 N 选 k）
PICK_NAME_ROW = (205, 225)                  # N 选 1 卡名那一行的 y 范围
PRICE_BADGE_DX, PRICE_BADGE_DY = 95, -170   # 商店价格角标圆心相对卡名中心的偏移（4/4 张实测）


def _price_badge_ratio(image, name_x: int, name_y: int) -> float:
    """卡名中心往右上那块（商店牌的价格角标位置）里红褐色像素的占比。"""
    import cv2
    x0, y0 = name_x + 70, name_y - 200
    crop = image[max(0, y0):max(0, y0) + 55, max(0, x0):max(0, x0) + 55]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2HSV)
    red = (((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 70))
    return float(red.mean())


def _hand_circles(image) -> tuple[list[float], Optional[float], int]:
    """手牌那一行的圆：(卡牌图标 x 列表, 枫叶舞槽红色角标 x 或 None, 图标行 y)。纯像素，~2ms。

    **红色角标**（枫叶舞槽右上角的「0」）和卡牌图标按颜色分：角标红色像素占比稳定 0.66~0.67，
    卡牌图标是 0；立绘里偶尔的红圈最多 ~0.6（多局实测）。
    """
    import cv2
    x, y, w, h = HAND_ICON_BAND
    g = cv2.cvtColor(np.ascontiguousarray(image[y:y + h, x:x + w]), cv2.COLOR_RGB2GRAY)
    c = cv2.HoughCircles(cv2.medianBlur(g, 3), cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                         param1=100, param2=16, minRadius=9, maxRadius=17)
    if c is None:
        return [], None, HAND_ICON_Y
    hsv = cv2.cvtColor(np.ascontiguousarray(image[y:y + h, x:x + w]), cv2.COLOR_RGB2HSV)
    icons, badges, ys = [], [], []
    for cx, cy, r in c[0]:
        if abs(cy + y - HAND_ICON_Y) > 13:
            continue
        m = np.zeros(g.shape, np.uint8)
        cv2.circle(m, (int(cx), int(cy)), max(2, int(r * 0.9)), 1, -1)
        px = hsv[m > 0]
        red = float((((px[:, 0] < 10) | (px[:, 0] > 170)) & (px[:, 1] > 90) & (px[:, 2] > 80)).mean())
        if red >= 0.62 and abs(cy + y - 590) <= 4 and cx + x < 720:
            badges.append(float(cx + x))
        elif red < 0.3 and cx + x <= HAND_LAST_ICON_MAX_X:
            # 最后一张牌的图标最右也就 ~994；再往右是能量圆盘那一带的圆，
            # 算进来会把"图标行 y"带偏，卡名读取带跟着偏、字被切掉（实测 x=1088 那个圆）
            icons.append(float(cx + x))
            ys.append(cy + y)
    ys.sort()
    return (sorted(icons), (min(badges) if badges else None),
            int(ys[len(ys) // 2]) if ys else HAND_ICON_Y)


AWAKEN_ICE_MIN = 0.25     # 露出部分里"冰蓝色（浅、不太饱和）"像素的最低占比
AWAKEN_DARK_MAX = 0.09    # 露出部分里深色像素的最高占比（普通卡立绘大片深色，觉醒卡几乎没有）
# **金色觉醒卡**（2026-09-23 第九局，蓝色地图上的觉醒孙膑：金 0.67 / 深 0.02；金色牌很好辨认）。
# 立绘本身偏黄的普通卡会混进来：大司命 金 0.64~0.71 / 深 0.05~0.07、朵莉亚 0.52/0.04、扁鹊 —— 靠深色区分，门槛更严。
AWAKEN_GOLD_MIN = 0.50
AWAKEN_GOLD_DARK_MAX = 0.03


def awaken_tint_spans(image) -> list[tuple[int, int, float, float]]:
    """手牌里**颜色像觉醒卡**的那几张：[(露出部分左 x, 右 x, 冰蓝占比, 深色占比)]。纯像素，几毫秒。

    2026-09-23：金色牌很好辨认，就算叠在一起颜色也很明显 —— 先确认颜色，再打开牌确认是哪个英雄。
    全历史 4234 张露出部分实测：觉醒卡（曜/干将莫邪/弈星/孙膑/芈月/廉颇/大司命/苏烈…）整张冰蓝、
    深色占比 0.00~0.02（李元芳头发深，0.05~0.07）；普通卡就算立绘偏蓝，深色也有 0.12 以上。
    **阿离的「翩跹舞步」卡面也是冰蓝的**（深色 0.05~0.09）—— 调用方只认英雄牌，靠类别排除。
    冰蓝和金色两种都认（同一张觉醒孙膑在不同地图/时刻两种颜色都出现过）。
    这只是"候选"，**是不是觉醒、是哪个英雄，一律点开看了才算**。
    """
    import cv2
    lay = hand_layout(image)
    if not lay:
        return []
    xs = lay["xs"]
    by_model = _hand_awaken_by_model(image, xs)
    if by_model is not None:
        return by_model
    out = []
    for i, x in enumerate(xs):
        x0 = int(x) - 10
        x1 = min(int(xs[i + 1]) - 20 if i + 1 < len(xs) else int(x) + 110, int(x) + 110, 1279)
        if x1 - x0 < 20:
            continue
        hsv = cv2.cvtColor(np.ascontiguousarray(image[590:710, x0:x1]), cv2.COLOR_RGB2HSV).astype(int)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        ice = float(((h >= 80) & (h <= 110) & (s >= 20) & (s <= 130) & (v >= 140)).mean())
        gold = float(((h >= 12) & (h <= 35) & (s >= 40) & (v >= 150)).mean())
        dark = float((v < 70).mean())
        if ice >= AWAKEN_ICE_MIN and dark <= AWAKEN_DARK_MAX:
            out.append((x0, x1, round(ice, 2), round(dark, 2)))
        elif gold >= AWAKEN_GOLD_MIN and dark <= AWAKEN_GOLD_DARK_MAX:
            out.append((x0, x1, round(gold, 2), round(dark, 2)))
    return out


HAND_AWAKEN_MODEL = Path(__file__).resolve().parents[2] / "assets" / "model" / "vision" / "hand_awaken_v1.onnx"
HAND_AWAKEN_THRESH = 0.5       # 验证/测试集上召回 100%、精度 100%（正样本 28 / 11 张，全是冰蓝觉醒卡；金色只在训练集里见过）
_HAND_AWAKEN_SESS = None


def _hand_awaken_by_model(image, xs) -> Optional[list]:
    """手牌觉醒卡识别小模型（tools/vision/train_hand_awaken.py；数据 data/vision/hand_awaken_v1）。
    2026-09-25：是否觉醒英雄卡改用图像识别，不再纯采样颜色。
    颜色判据精度只有 0.24~0.38（立绘偏金/蓝的卡、翩跹舞步、效果牌都会触发；武则天一局被点开 9 次），模型测试集 11/11、0 误报。
    返回格式与 awaken_tint_spans 相同（第三项放模型概率、第四项 0）；模型不在返回 None（退回颜色判据）。"""
    global _HAND_AWAKEN_SESS
    if _HAND_AWAKEN_SESS is None:
        try:
            import onnxruntime as ort
            _HAND_AWAKEN_SESS = _gpu_session(HAND_AWAKEN_MODEL, "手牌觉醒") \
                if HAND_AWAKEN_MODEL.exists() else False
        except Exception:                                           # noqa: BLE001
            logger.exception("手牌觉醒识别模型加载失败，退回颜色判据")
            _HAND_AWAKEN_SESS = False
    if not _HAND_AWAKEN_SESS:
        return None
    crops, spans = [], []
    for i, x in enumerate(xs):
        x0 = int(x) - 12
        if x0 < 0 or x0 + 64 > image.shape[1]:
            continue
        c = image[595:715, x0:x0 + 64]                   # 与训练一致：64x120 → 中心 56x112，RGB
        crops.append(c[4:116, 4:60])
        x1 = min(int(xs[i + 1]) - 20 if i + 1 < len(xs) else int(x) + 110, int(x) + 110, 1279)
        spans.append((int(x) - 10, x1))
    if not crops:
        return []
    xin = np.ascontiguousarray(np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
    lg = _HAND_AWAKEN_SESS.run(None, {"x": xin})[0]
    e = np.exp(lg - lg.max(1, keepdims=True))
    p = (e / e.sum(1, keepdims=True))[:, 1]
    return [(a, b, round(float(q), 3), 0.0) for (a, b), q in zip(spans, p) if q >= HAND_AWAKEN_THRESH]


def hand_grid(n: int) -> list[float]:
    """N 张手牌时每张牌图标的 x（见 HAND_* 常量上面的布局说明）。"""
    if n <= 0:
        return []
    if n <= 5:
        x0 = HAND_CENTER_ICON_X - (n - 1) * HAND_PITCH_MAX / 2
        return [x0 + k * HAND_PITCH_MAX for k in range(n)]
    p = HAND_SPAN / (n - 1)
    return [HAND_FIRST_ICON_X + k * p for k in range(n)]


# ---------------------------------------------------------------- 手牌条识别模型（2026-10-09）
# 2026-10-09：改用图像识别，不再用颜色区分 —— 替掉"霍夫圆找图标 + 红角标 + 网格打分 + 空牌位像素补丁"。
# 测试集：张数 99.6%、拖起错位帧 99.3%、已知空牌位误判"有牌" 0%（旧方法 94%），抓取点误差 p90 0.2px，CPU p90 2.6ms。
HAND_STRIP_MODEL = Path(__file__).resolve().parents[2] / "assets" / "model" / "vision" / "hand_strip_v1.onnx"
_HS_X0, _HS_X1, _HS_Y0, _HS_Y1 = 160, 1184, 528, 720
_HS_IN_H, _HS_IN_W, _HS_COLS, _HS_CARD_W = 48, 256, 128, 142
_HS_COL_PX = (_HS_X1 - _HS_X0) / _HS_COLS
_HAND_STRIP_SESS = None
HAND_STRIP_MIN_PEAK = 0.7       # 每张牌左边缘的热图峰值至少这么高才算"有把握"
_HS_DIFF = {"n": 0, "last": 0.0}


def hand_layout_model(image) -> Optional[dict]:
    """手牌条模型 → 与 hand_layout 同结构的 dict（xs = 左边缘 + HAND_ICON_TO_LEFT，即"图标 x"口径），外加 spans / conf。
    没把握（张数头与解码不一致、某张左边缘峰 < HAND_STRIP_MIN_PEAK）或模型不在 → None（调用方退回旧 hand_layout）。
    解码照抄 tools/vision/train_hand_strip.decode_one（agent 里不 import torch）。图像是 RGB（线上截图）。"""
    global _HAND_STRIP_SESS
    if _HAND_STRIP_SESS is None:
        try:
            import onnxruntime as ort
            if HAND_STRIP_MODEL.exists():
                so = ort.SessionOptions()
                so.intra_op_num_threads = 2
                _HAND_STRIP_SESS = _gpu_session(HAND_STRIP_MODEL, "手牌条")
            else:
                _HAND_STRIP_SESS = False
        except Exception:                                           # noqa: BLE001
            logger.exception("手牌条模型加载失败，退回 hand_layout")
            _HAND_STRIP_SESS = False
    if not _HAND_STRIP_SESS or image is None or image.shape[0] < _HS_Y1 or image.shape[1] < _HS_X1:
        return None
    try:
        import cv2
        crop = np.ascontiguousarray(image[_HS_Y0:_HS_Y1, _HS_X0:_HS_X1, :3])
        x = cv2.resize(crop, (_HS_IN_W, _HS_IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        seg_l, edge_l, off_p, awk_l, cnt_l = [o[0] for o in _HAND_STRIP_SESS.run(None, {"x": x})]
    except Exception:                                               # noqa: BLE001
        logger.exception("手牌条模型推理出错，退回 hand_layout")
        return None
    seg = seg_l.argmax(0)
    e = 1 / (1 + np.exp(-edge_l))
    cand = sorted((j for j in range(_HS_COLS) if e[j] >= 0.5 and (seg[j] == 2 or (j + 1 < _HS_COLS and seg[j + 1] == 2))),
                  key=lambda j: (-e[j], j))
    peaks: list = []
    for j in cand:
        if all(abs(j - q) > 4 for q in peaks):
            peaks.append(j)
    peaks.sort()
    lefts = [_HS_X0 + (j + float(np.clip(off_p[j], 0, 1))) * _HS_COL_PX for j in peaks]
    spans = []
    for k, l in enumerate(lefts):
        if k + 1 < len(lefts):
            r = lefts[k + 1]
        else:
            j = peaks[k]
            while j + 1 < _HS_COLS and seg[j + 1] == 2:
                j += 1
            r = min(_HS_X0 + (j + 1) * _HS_COL_PX, l + _HS_CARD_W)
        spans.append((round(l, 1), round(r, 1)))
    n = len(spans)
    count_head = int(np.argmax(cnt_l))
    conf = min((float(e[j]) for j in peaks), default=1.0)
    if count_head != n or (n and conf < HAND_STRIP_MIN_PEAK):
        return None
    xs = [l + HAND_ICON_TO_LEFT for l, _r in spans]
    try:
        _icons, _badge, icon_y = _hand_circles(image)
    except Exception:                                               # noqa: BLE001
        icon_y = None
    return {"n": n, "xs": xs, "icon_y": icon_y or 588, "by": "模型",
            "pitch": (xs[1] - xs[0]) if n > 1 else HAND_PITCH_MAX, "hits": n,
            "spans": spans, "conf": round(conf, 3), "skill": bool((seg == 1).sum() >= 4)}


def _slot_is_ground(image, icon_x: float) -> bool:
    """手牌第 k 格（图标 x）下面是不是场地背景而不是牌：卡牌下半截是深色底（均值 30~100），场地亮（>115 且起伏小）。"""
    try:
        x = int(icon_x)
        r = image[650:700, max(0, x - 10):x + 60]
        if r.size == 0:
            return False
        return float(r.mean()) > 115 and float(r.std()) < 45
    except Exception:                                               # noqa: BLE001
        return False


def hand_layout(image) -> Optional[dict]:
    """手牌有几张、每张在哪 —— **优先用手牌条模型**（hand_layout_model），没把握才走下面的图标推算（旧方法，兜底）。
    两者张数不一致时记日志（【手牌模型分歧】，每 30 秒最多一条）。"""
    lay = hand_layout_model(image)
    old = _hand_layout_icons(image)
    if lay is not None:
        if old is not None and old.get("n") != lay["n"]:
            _HS_DIFF["n"] += 1
            if time.time() - _HS_DIFF["last"] > 30:
                _HS_DIFF["last"] = time.time()
                logger.info("【手牌模型分歧】模型 %d 张 %s / 图标推算 %d 张（累计 %d 次，用模型）",
                            lay["n"], [round(a) for a, _b in lay["spans"]], old.get("n"), _HS_DIFF["n"])
        return lay
    return old


def _hand_layout_icons(image) -> Optional[dict]:
    """（旧方法，兜底）手牌有几张、每张在哪。返回 {"n", "xs"(图标x), "icon_y", "pitch", "hits", "by"}；什么都没找到返回 None。

    **优先用枫叶舞槽的红色角标定位**：它总在第一张牌图标左边 SKILL_BADGE_DX。
    不叠放时（N ≤ 5）第一张牌的位置直接决定张数：x0 = 670 - (N-1)*71 —— 不依赖卡牌图标，
    因为**有的牌没有左上角图标**（「翩跹舞步」这一族棋手专属牌，实测整张卡没有那个圆）。
    叠放时（角标在最左 ~319）第一张固定在 385，张数再靠图标打分决定。
    """
    icons, badge, icon_y = _hand_circles(image)
    near = lambda a, b: abs(a - b) <= HAND_ICON_TOL   # noqa: E731

    def score(n: int) -> float:
        grid = hand_grid(n)
        if not grid:
            return 0.0
        hit = sum(any(near(g, x) for x in icons) for g in grid)
        extra = sum(1 for x in icons if x < grid[-1] + HAND_ICON_TOL
                    and not any(near(g, x) for g in grid))
        return hit - 0.6 * (n - hit) - 0.3 * extra

    how = "图标"
    if badge is not None:
        x0 = badge + SKILL_BADGE_DX
        if x0 > HAND_FIRST_ICON_X + 30:          # 没叠放：角标位置直接给出张数
            n = max(0, min(5, round((HAND_CENTER_ICON_X - x0) / (HAND_PITCH_MAX / 2)) + 1))
            how = "角标"
        else:                                     # 叠放（或正好 5 张）：张数靠图标
            n = max(range(5, HAND_MAX + 8), key=score)
            how = "角标+图标"
    else:
        if not icons:
            return None
        n = max(range(0, HAND_MAX + 8), key=score)
    grid = hand_grid(n)
    if how != "角标":
        # **最右边那几格没图标、而且那里是亮的场地背景 → 不是牌**（2026-10-09：经常去拖动空牌位，导致出现表情发送轮盘；
        # 证据帧 live_1791538545/inspect_no_view_1791538790：只有 1 张大棒，按图标打分推成 2 张，第 2 格 x=741 是场地）。
        # 卡牌底部那条是深色（均值 ~35），场地背景亮（~130）
        while n > 0 and grid and not any(near(grid[-1], x) for x in icons) and _slot_is_ground(image, grid[-1]):
            n -= 1
            grid = hand_grid(n)
    return {"n": n, "xs": grid, "icon_y": icon_y, "by": how,
            "pitch": (grid[1] - grid[0]) if n > 1 else HAND_PITCH_MAX,
            "hits": sum(any(near(g, x) for x in icons) for g in grid)}


def _ocr_multi(context, image, roi: tuple[int, int, int, int]) -> list[str]:
    """OCR 并返回 ROI 内全部文本块（用于手牌等多目标区域）。"""
    return [t for t, _ in _ocr_boxes(context, image, roi)]


# 人口数字的颜色（图像是 RGB）。2026-09-24 确认：没满是绿色、满员是白色，
# **拖着新英雄、放下就会超上限时变成偏红色**（而且字变大、往左扩约 50px，所以红色用更宽的区域数）。
# 实测：满员白色像素 635~842（亮度>150），没满时 0。红色阈值按人工截图（橙红 R≈230 G≈110 B≈95）定，
# 排除等级徽章的金色圈（G≈180）；第一次实战触发时要人工核对证据帧。
POP_WHITE_ROI = (840, 240, 140, 60)
POP_RED_ROI = (780, 230, 200, 80)


def pop_color(image: np.ndarray):
    """'red' / 'white' / None。"""
    if image is None or image.shape[0] < 320 or image.shape[1] < 1000:
        return None
    x, y, w, h = POP_RED_ROI
    c = image[y:y + h, x:x + w].astype(np.int16)
    R, G, B = c[..., 0], c[..., 1], c[..., 2]
    red = int(((R > 170) & (G < 140) & (B < 140) & (R - G > 70)).sum())
    x, y, w, h = POP_WHITE_ROI
    c = image[y:y + h, x:x + w].astype(np.int16)
    mn, mx = c.min(-1), c.max(-1)
    white = int(((mn > 150) & (mx - mn < 30)).sum())
    if red > 300 and red > white:
        return "red"
    # 实测（商店关着的准备帧）：满员 635~842，没满 0 —— 商店开着时卡面会干扰，调用方已跳过
    if white > 500:
        return "white"
    return None


def _has_energy_dial(image: np.ndarray) -> bool:
    """右下角的能量圆盘在不在——拍卖里用它判断"能不能出价"。

    纯像素判据，不用 OCR：顶部"拍卖开启"只闪一瞬就变回"预览"，文字靠不住。
    圆盘是高饱和的蓝紫色圆形；预览期那个位置是舞台背景（低饱和、无圆）。
    """
    import cv2

    x, y, w, h = ROI_AUCTION_DIAL
    if image is None or image.shape[0] < y + h or image.shape[1] < x + w:
        return False
    roi = image[y:y + h, x:x + w]
    sat = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)[:, :, 1].mean()
    if sat < 65:                      # 实测：有圆盘 88-91，没有 43-47
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(cv2.medianBlur(gray, 3), cv2.HOUGH_GRADIENT,
                               dp=1, minDist=40, param1=100, param2=30,
                               minRadius=25, maxRadius=60)
    return circles is not None


def _hit(context, image, roi: tuple[int, int, int, int], pattern) -> bool:
    """ROI 里有没有匹配 pattern 的文字。pattern 也可以是 (正则, 至少几处)：要求多个文字框各自命中。"""
    if isinstance(pattern, tuple):
        pat, need = pattern
        return sum(1 for t, _b in _ocr_boxes(context, image, roi) if re.search(pat, t)) >= need
    return bool(_ocr(context, image, roi, expected=pattern))


# ==================== 并发 OCR ====================
# 一帧的成本几乎全是 OCR 往返（单次约 62-72ms），而同一帧里的各个 ROI 之间**互不依赖**，
# 纯属可以同时跑。2026-09-22 实测（13 个 ROI，同一张帧）：
#     串行            937ms（72.1ms/次）
#     线程池 4 worker 335ms（25.8ms/次）  → 2.80x
#     先 post 全部再 wait 334ms           → 同样 2.80x（MaaFW 内部本来就是流水线）
# 两种方式等价，这里用线程池：它对 `FrameContext`（离线回放桥）和 MaaFW 原生 Context
# 都适用，不依赖某个 context 实现暴露 post/wait 两段式 API。
# 4 个 worker 就到顶了（8 个反而略慢），再多只是排队。
_OCR_WORKERS = 4
_POOL = None
_POOL_LOCK = __import__("threading").Lock()
_IN_POOL = __import__("threading").local()


def _get_pool():
    """全局共享线程池：每次现开现关的话，池子启停开销会吃掉一部分收益。"""
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                from concurrent.futures import ThreadPoolExecutor
                _POOL = ThreadPoolExecutor(max_workers=_OCR_WORKERS,
                                           thread_name_prefix="ocr")
    return _POOL


def _map_parallel(fn, items: list):
    """并发执行 fn(item)，**保持输入顺序**返回结果列表。

    两条短路，都不是可选的：
    - 任务少于 3 个：直接串行，调度开销比省下来的多。
    - **已经在池子里的线程上**：必须串行。池子是有界的，在 worker 里再往同一个
      池子提交任务并 wait，会出现"worker 等着排在自己后面的任务"→ 死锁。
      提取层是两层嵌套的（extract 并发跑各项，其中 _extract_economy 自己又要并发），
      所以这条一定会走到。外层已经把 4 个 worker 占满了，内层串行也不亏。
    """
    if len(items) < 3 or getattr(_IN_POOL, "flag", False):
        return [fn(it) for it in items]

    def _wrapped(it):
        _IN_POOL.flag = True
        try:
            return fn(it)
        finally:
            _IN_POOL.flag = False

    return list(_get_pool().map(_wrapped, items))


def _to_int(text: str, default: int = -1) -> int:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else default


class StateExtractor:
    """从截图提取 GameState。

    cards_db_path 提供全量卡名表，OCR 文本模糊匹配回标准卡名（修正 stylized 字体误识）。
    模板图在首次使用时从 assets/image/ 加载。
    """

    def __init__(self, image_dir: Path | str, cards_db_path: Path | str | None = None):
        self.image_dir = Path(image_dir)
        self._templates: Optional[dict[str, np.ndarray]] = None
        self._equip_names: set = set()
        self._equip_recipes: Optional[dict] = None
        self._phase_cache: Optional[str] = None   # 上一帧识别出的阶段，用于短路
        self._region_hash: dict[str, int] = {}    # 区域像素指纹，用于跳过没变化的 OCR
        self._shop_cache: list[ShopSlot] = []
        self._hand_seen: dict[str, float] = {}   # 最近在手牌里认出过的卡名 -> 时间（叠放时按前缀认牌用）
        self.hand_hint: set = set()              # 决策层的**手牌账**里有的牌（叠放时按前缀认牌优先在这里找）
        self._players_cache: list[PlayerRow] = []
        self._board_cache: Optional[set] = None   # 棋盘占用观测缓存（徽章带像素没变就复用）
        self._last_round: int = 0
        self._hp_reject_since: float = 0.0
        self._last_population: int = 0
        self._last_lord_level: int = 1
        self._last_hp: int = MAX_PLAYER_HP
        self._hp_reject: tuple[int, int] = (0, 0)  # (被拒的血量读数, 连读次数)
        self._hp_round = 0                         # 上次认定血量变化的回合
        self._hp_before = MAX_PLAYER_HP            # 最近一次掉血之前的血量
        self._hp_seen: bool = False             # 本局有没有真正采信过一次血量读数
        self._auction_miss: int = 0             # 连续几帧认不出拍卖界面（防 N选1 抢走）
        self._own_row: Optional[int] = None    # 我方在六人血条里最近一次认出的行号（只用于记"重排"日志，不再沿用读血）
        self._own_row_miss: bool = False       # 上一帧是否没认出我方行（日志只在转为认不出时打一次）
        self._last_pop_cap: int = 0
        self._last_ban: str = ""                  # 本局禁用派系（一局不变，读到就记住）
        self._last_level_cost: int = -1           # 升级费用（读不到就沿用上次，见 _extract_economy）
        self._last_level_btn: Optional[tuple] = None   # 升级按钮实测坐标（同上，但见到「出售」立刻作废）
        self._card_names: list[str] = []
        self._card_name_set: set[str] = set()   # 精确匹配用（单字卡名靠它放行，见 _extract_hand）
        self._card_descs: dict[str, str] = {}
        self._card_cats: dict[str, str] = {}    # 卡名 -> 类别（手牌里认棋手技能牌用）
        db_path = Path(cards_db_path) if cards_db_path else (
            self.image_dir.parent.parent / "data" / "cards.json"
        )
        if db_path.exists():
            import json

            db = json.loads(db_path.read_text(encoding="utf-8"))
            for n, c in db.get("cards", {}).items():
                # 棋手牌只在**选棋手阶段**出现一次，选完整局就不存在了
                # （2026-09-22 确认：之后只有该棋手的专属效果牌）。
                # 留在模糊匹配表里只会帮倒忙——商店/手牌的卡名被错配成棋手牌，
                # 决策层拿到一张 cost=0 / 没有 tags 的空壳记录。
                # "名字(类别)" 那种消歧 key 同理（同名冲突的败者留档，见
                #  fetch_card_db.merge_cards），它不该参与卡名匹配。
                if c.get("category") == "棋手牌" or n.endswith(")"):
                    continue
                if c.get("lord") and c.get("lord") != OUR_LORD:
                    continue        # 别家棋手的专属牌 / 专属天赋（安琪拉飞行器模块、香香战术、昭君冰封…）：阿离局压根拿不到（2026-10-20），留着只会被错配
                desc = c.get("skill") or c.get("desc") or ""
                self._card_cats[n] = c.get("category", "")
                if desc:
                    self._card_descs[n] = desc
            self._card_names = sorted(self._card_descs.keys() or db.get("cards", {}).keys())
            self._card_name_set = set(self._card_names)
            # 商店只卖英雄 / 效果 / 装备（2026-10-11 识别审查：全库匹配把「奔星」→ 天赋「逐星」，300 帧 6 格认成天赋牌）
            self._shop_names = [n for n in self._card_names if self._card_cats.get(n) in ("英雄牌", "效果牌", "装备牌")]                 or self._card_names
            logger.info("卡名表加载 %d 个（模糊匹配用，已排除棋手牌）",
                        len(self._card_names))
            # **手牌区的候选卡名：去掉永远不会进手牌的被动天赋**（data/card_usage.json 里 usable_from_hand=false 的天赋牌）。
            # 2026-09-24 查出：OCR 把「弈星」的"弈"读成"奕/亦/变"时，全库模糊匹配会落到天赋「逐星」上
            # （两者都只差 1 个字、并列），历史帧里「逐星」在手牌里出现过 1100 多帧 —— 弈星（升棋手等级的关键英雄）
            # 于是被当成一张"不用打的天赋牌"。被动天赋选中即生效、从不进手牌，排掉它们不会误伤。
            self._hand_names = self._card_names
            try:
                import json as _json
                usage = _json.loads((Path(__file__).resolve().parents[2] / "data" / "card_usage.json")
                                    .read_text(encoding="utf-8"))
                passive = {n for n, u in usage.items()
                           if u.get("category") == "天赋牌" and not u.get("usable_from_hand", True)}
                self._hand_names = [n for n in self._card_names if n not in passive]
                logger.info("手牌候选卡名 %d 个（排除被动天赋 %d 个）", len(self._hand_names),
                            len(self._card_names) - len(self._hand_names))
            except (OSError, ValueError):
                logger.warning("data/card_usage.json 读不到，手牌卡名按全库匹配")
            self._hand_name_set = set(self._hand_names)

    def _single_char_hero(self, text: str) -> bool:
        """去掉非汉字后正好是一个字、且是卡库里的英雄名（镜 / 瑶 / 曜）。"""
        h = "".join(re.findall(r"[一-鿿]", text or ""))
        return len(h) == 1 and h in (getattr(self, "_card_name_set", None) or set())

    def resolve_name(self, text: str, cutoff: float = 0.75,
                     keep_raw: bool = True, pool: Optional[list] = None) -> Optional[str]:
        """把 OCR 文本解析为标准卡名：精确命中 → 模糊匹配；都不中时看 keep_raw。

        **商店/拍卖一律用 keep_raw=False**：那些位置只会出现卡牌库里有的牌
        （库里已有 716 张，含全部棋手关联卡）。保留原文的后果是把任何 OCR 噪声
        都当成棋子——2026-09-22 实战把别的玩家昵称"画个圈圈嗨"和数字"35"
        当成了商店里的牌，进而让"商店是否开着"的判断永远为真。

        手牌保留 keep_raw=True：万一真有库里没收录的牌，要让决策层看见并告警补录。
        """
        import difflib

        text = text.strip().strip("·- ")
        if not text:
            return None
        names = pool if pool is not None else self._card_names
        if text in names:
            return text
        if OCR_NAME_ALIASES.get(text) in names:
            return OCR_NAME_ALIASES[text]
        # 短名字要放宽门槛：2 个字差 1 个字，difflib 相似度只有 0.5，
        # 用默认 0.75 永远匹配不上（实测"兼颇"/"乘颇"都是"廉颇"的误识）。
        if len(text) <= 3:
            cutoff = min(cutoff, 0.45)
        matches = difflib.get_close_matches(text, names, n=1, cutoff=cutoff)
        if matches:
            logger.debug("卡名模糊匹配: %r -> %r", text, matches[0])
            return matches[0]
        # **difflib 兜不住"前缀被认花、后缀认对"的情况**：
        # 「翩跹舞步·新生」的"翩跹"实测被读成 解冠/曾跃/翻冠氨，
        # difflib 的编辑距离直接判不像，于是整张卡一局都用不上（又没用上）。
        # 但这一族有 5 张同前缀的牌，**区分本来就靠后缀，而后缀 OCR 是对的**。
        # 所以再用一道**字符覆盖率**兜底，并要求"第一名明显好过第二名"，
        # 避免在同前缀族里瞎选（实测 解冠舞步·新生 -> 新生 0.71 / 余韵 0.43）。
        if len(text) >= 4:
            scored = sorted(((_text_coverage(text, n), n) for n in names),
                            reverse=True)
            if scored and scored[0][0] >= 0.66 and (
                    len(scored) < 2 or scored[0][0] - scored[1][0] >= 0.15):
                logger.info("卡名按字符覆盖率兜底: %r -> %r (%.2f)",
                            text, scored[0][1], scored[0][0])
                return scored[0][1]
        # 最后一道：**公共前缀足够长、而且只有一张卡对得上**。
        # 冲着"尾巴被认花"这一类失效去的：2026-09-23 实战把「破碎圣杯」读成
        # 「破碎圣书A」—— difflib 0.67、字符覆盖率 0.60，两道都差一点点，
        # 于是这张装备整局对决策层不存在。
        # 安全性是**数过的**：748 个卡名里，3 字前缀重复的只有 18 组，
        # 所以"前缀 3 字且全库唯一"极少误判；不唯一就老实返回 None
        # （「翩跹舞步」后面跟着 5 张同族卡，本来就分不出来，硬猜只会猜错）。
        # 两道闸，缺一不可（2026-09-23 实战当场打脸：没有第二道时，
        # 一整句 '姜子牙），鲁班七号' 被认成了「姜子牙」——它只是恰好以卡名开头）：
        #   ① 卡名里**只可能出现汉字/数字/中点**，出现逗号括号就不是卡名，是一句话；
        #   ② OCR 的错字/多字是**少量**的，认出来的卡名长度必须和原文接近。
        if len(text) >= 3 and not re.search(r"[，。、（）()：:；;]", text):
            pref = text[:3]
            hits = [n for n in names if n.startswith(pref)]
            if len(hits) == 1 and len(text) <= len(hits[0]) + 2:
                logger.info("卡名按唯一前缀兜底: %r -> %r", text, hits[0])
                return hits[0]
        if keep_raw:
            logger.info("卡名未收录于 DB: %r", text)
            return text
        return None

    def resolve_by_desc(self, desc: str, cutoff: float = 0.5) -> Optional[str]:
        """按效果描述文本反查卡名（天赋等 stylized 字体名称 OCR 不准时兜底）。"""
        import difflib

        desc = desc.strip()
        if len(desc) < 4:
            return None
        matches = difflib.get_close_matches(
            desc, self._card_descs.values(), n=1, cutoff=cutoff)
        if not matches:
            return None
        for name, d in self._card_descs.items():
            if d == matches[0]:
                logger.debug("描述反查: %r -> %r", desc[:20], name)
                return name
        return None

    @property
    def templates(self) -> dict[str, np.ndarray]:
        if self._templates is None:
            self._templates = {}
            self._equip_names = {p.stem for p in self.image_dir.glob("pieces/装备牌/*.png")}
            for p in self.image_dir.glob("pieces/**/*.png"):
                img = _imread(p)
                if img is not None:
                    self._templates[p.stem] = img
            logger.info("加载棋子模板 %d 个（其中装备 %d 个）",
                        len(self._templates), len(self._equip_names))
        return self._templates

    @property
    def equip_recipes(self) -> dict:
        """基础装备 -> 4 个可铸成品（四选一用）。"""
        if self._equip_recipes is None:
            self._equip_recipes = load_equip_recipes(
                self.image_dir.parent.parent / "data")
        return self._equip_recipes

    @property
    def equip_names(self) -> set:
        """装备牌模板名集合（四选一弹窗匹配用）。"""
        self.templates          # 触发懒加载
        return self._equip_names

    # ---------- 阶段识别 ----------

    # ---- 「卡牌N选1」检测要不要跑 ----
    # 2026-09-23 `--perf` 实测：这一项**每帧 505~609ms、11~13 次 OCR**（滑窗扫棋盘上方，
    # 那里有动画，OCR 缓存几乎不命中），占了一帧的三分之一，而 N 选 1 一局就出现几次。
    # 它不会凭空出现：用了效果牌/天赋牌、买了牌、回合刚开始 —— 主循环在这些动作之后
    # 设 `card_pick_hint_until`；其余时候每 CARD_PICK_EVERY 帧抽查一次兜底
    # （N 选 1 不选就一直挂着，晚几帧发现只是慢一点，不会错过）。
    CARD_PICK_EVERY = 4
    card_pick_hint_until = 0.0
    _cp_frames = 0
    _cp_round = -1

    # 单位等级：每回合第一帧、主循环提示过（合成/用了加等级的牌）、平时每 3 帧读一次。
    # 每个徽章一次 OCR，单位有动画、像素缓存不太命中，所以别每帧都读。
    _lv_frames = 0
    _lv_round = -1

    def _levels_due(self, state: GameState) -> bool:
        if not hasattr(self, "_last_levels"):
            self._last_levels = {}
            self._last_awaken = {}
        self._lv_frames += 1
        if state.round_no != self._lv_round:
            self._lv_round = state.round_no
            self._last_levels = {}                     # 新回合：阵亡/卖掉的都得重新看
            self._last_awaken = {}
            return True
        return time.time() < self.card_pick_hint_until or self._lv_frames % 3 == 0

    # 过百等级的徽章是橙色箭头样式，常被读成个位/两位数（第 388 局 136→12/14、第 414 局 73+→1）：
    # 一个格子见过 ≥100 级，之后读到不足它 30% 的值先不信，同一个值连读 3 次才认（换了人就会连续读到）。
    LV_PEAK_MIN = 100
    LV_DROP_RATIO = 0.3

    def _guard_levels(self, new: dict) -> dict:
        peak = self.__dict__.setdefault("_lv_peak", {})
        pend = self.__dict__.setdefault("_lv_pend", {})
        out = {}
        for cell, v in (new or {}).items():
            pk = peak.get(cell, 0)
            if isinstance(v, int) and pk >= self.LV_PEAK_MIN and v < pk * self.LV_DROP_RATIO:
                n = pend.get(cell, (None, 0))
                n = (v, n[1] + 1) if n[0] == v else (v, 1)
                pend[cell] = n
                if n[1] < 3:
                    continue                    # 先不信，沿用 _last_levels 里的旧值
                peak[cell] = v                  # 连读 3 次一致：认了（多半是换了人）
            pend.pop(cell, None)
            if isinstance(v, int) and v > pk:
                peak[cell] = v
            out[cell] = v
        return out

    def reset_level_peaks(self) -> None:
        self._lv_peak, self._lv_pend = {}, {}

    def _card_pick_due(self, state: GameState) -> bool:
        self._cp_frames += 1
        if state.round_no != self._cp_round:
            self._cp_round = state.round_no
            return True
        return (time.time() < self.card_pick_hint_until
                or self._cp_frames % self.CARD_PICK_EVERY == 0)

    def _region_changed(self, key: str, image: np.ndarray,
                        cells: list[tuple[int, int, int, int]]) -> bool:
        """这些格子的像素相对上一帧是否变化（用于跳过没必要重读的 OCR）。

        **只比对实际要 OCR 的那几个小格子**，不要用一个大包围盒——包围盒会跨过
        卡面立绘、玩家头像这些会动的东西，导致永远判定为"变了"。
        （2026-09-21 实战教训：第一版用 (68,218,975,190) 包住商店卡名带+派系带，
        中间夹着会动的卡图，缓存命中率 0。）

        保守策略：**任何**像素差异都算变了，宁可多读一次也不用过期数据。
        首帧、尺寸异常一律返回 True。
        """
        if image is None:
            return True
        h_img, w_img = image.shape[:2]
        chunks = []
        for x, y, w, h in cells:
            if y + h > h_img or x + w > w_img:
                return True
            chunks.append(image[y:y + h, x:x + w].tobytes())
        digest = hash(b"".join(chunks))
        changed = self._region_hash.get(key) != digest
        self._region_hash[key] = digest
        return changed

    def _phase_hits(self, context, image, phase: str) -> bool:
        """单个阶段的判据：任一 ROI 命中即算。

        单个阶段最多 3 条判据，串行短路（第一条通常就命中）比起线程更划算。
        真正费时的是"全链扫描"，那条路走 `_phases_hit_parallel`。
        """
        return any(_hit(context, image, roi, pattern)
                   for roi, pattern in PHASE_PREDICATES[phase])

    def _first_phase_parallel(self, context, image, phases: list[str]) -> Optional[str]:
        """并发跑多个阶段的**全部**判据，返回 PHASE_ORDER 里最靠前的命中者。

        串行版是"逐条试探、命中即停"，最坏 13 次 OCR ≈ 840ms；unknown 帧每次都走最坏情况。
        并发版一次把所有判据发出去（4 worker），实测约 3x。代价是命中后仍会跑完其余判据
        —— 但这条路本来就是"前面都没命中"才会走到，短路省不下什么。
        """
        probes = [(ph, roi, pat) for ph in phases
                  for roi, pat in PHASE_PREDICATES[ph]]
        hits = _map_parallel(lambda p: _hit(context, image, p[1], p[2]), probes)
        got = {ph for (ph, _, _), ok in zip(probes, hits) if ok}
        return next((ph for ph in phases if ph in got), None)

    def detect_phase(self, context, image: np.ndarray) -> str:
        """识别当前阶段。天赋/结算是模态，优先于底层阶段。

        性能（2026-09-21 实测）：单次 OCR 约 62ms，原本逐个试探最多 13 次 =
        831ms，占一帧 951ms 的 87%。而阶段在帧与帧之间极少变化，所以：
          1. 先查模态（settle/talent）—— 它们能叠加在任何阶段之上，不能跳过
          2. 再验证上一帧的阶段是否仍成立 —— 命中就直接返回
          3. 都不成立才回退到完整顺序试探
        稳态下 13 次 OCR 降到 4 次。`unknown` 不缓存，下一帧走完整链。
        """
        modal = self._first_phase_parallel(context, image, list(MODAL_PHASES))
        if modal:
            self._phase_cache = modal
            return modal

        last = self._phase_cache
        if last == "card_pick":
            # **card_pick 没有文字判据，进不了 PHASE_PREDICATES**，所以复验必须走它
            # 自己的检测器。走 _phase_hits 会 KeyError —— 2026-09-22 实战踩到：
            # 一局里只要出现过一次 N 选 1，**之后每一帧都抛异常**，整局报废。
            # 教训：缓存里存的值域必须和查表的键域一致，多一个特例就要多一条分支。
            probe = GameState()
            self._detect_card_pick(context, image, probe)
            if probe.card_pick:
                return "card_pick"
        elif (last and last not in MODAL_PHASES and last in PHASE_PREDICATES
                and self._phase_hits(context, image, last)):
            return last

        rest = [p for p in PHASE_ORDER if p not in MODAL_PHASES]
        phase = self._first_phase_parallel(context, image, rest)
        if phase:
            self._phase_cache = phase
            return phase

        # 掉到 unknown 之前，再看一眼是不是「卡牌 N 选 1」。
        # 它**没有自己的文字判据**（判据是"商店位置上摆着几张查得到的卡、且没有商店面板标志"），
        # 所以进不了 PHASE_PREDICATES；而它又会**把「准备/开战」按钮盖住**，
        # 于是 prepare 判据全不命中 → 落进 unknown → 原来只在 prepare 分支里跑的
        # `_detect_card_pick` 根本没机会执行 → 界面一直挂着，后续全卡死
        # （2026-09-22 第六局实测：锦囊打出后弹出三选一，agent 干等到超时）。
        # **拍卖界面不能被 N选1 抢走**（2026-09-23 实战：auction → card_pick → auction
        # 来回跳）。出价阶段顶部那行字会变，`_detect_card_pick` 里按
        # ROI_AUCTION_TITLE 做的排除就失效了，而拍品也是"一排查得到的卡名"，
        # 于是被 N选1 认领 —— 后果是对着拍品乱点（等于乱出价）。
        # 判据：上一帧还是拍卖，就要**连续两帧**都认不出拍卖，才允许让位。
        if last == "auction":
            self._auction_miss += 1
            if self._auction_miss < 2:
                logger.info("这一帧认不出拍卖界面，但上一帧是拍卖 —— 先按拍卖处理，"
                            "不让 N选1 抢走")
                return "auction"
        else:
            self._auction_miss = 0
        probe = GameState()
        self._detect_card_pick(context, image, probe)
        if probe.card_pick:
            self._phase_cache = "card_pick"
            return "card_pick"

        self._phase_cache = None              # 未识别不缓存，避免把错误状态粘住
        return "unknown"

    # ---------- 主入口 ----------

    def extract(self, context, image: np.ndarray, phase: str | None = None) -> GameState:
        state = GameState()
        PERF.reset()
        with PERF.stage("detect_phase"):
            state.phase = phase or self.detect_phase(context, image)
        # 界面分类模型（2026-09-30，**影子模式**：只记录、不改阶段；主循环比对分歧并存帧，跑几局再按类放权）
        try:
            from agent.perception import screen_cls as _SC
            r = _SC.classify(image)
            if r:
                state.screen_cls = [r[0], round(float(r[1]), 3)]
        except Exception:                                           # noqa: BLE001
            pass

        # 新一局开始（大厅/选棋手）：清空跨局缓存的回合/人口读数
        if state.phase in ("mode_lobby", "lobby", "hero_pick"):
            self._last_round = 0
            self._last_population = 0
            self._last_pop_cap = 0
            self.pop_bonus_seen, self._pop_extra_votes = 0, (0, 0)
            self._last_lord_level = 1
            self._last_hp = MAX_PLAYER_HP
            self.reset_level_peaks()
            self._own_row = None
            self._hp_reject = (0, 0)
            self._hp_reject_since = 0.0
            self._hp_seen = False
            self._hp_round = 0
            self._hp_before = MAX_PLAYER_HP
            self._last_ban = ""            # 新一局，禁用派系重新读
            self._last_level_cost = -1
            self._last_level_btn = None
            self._board_cache = None

        # round / ban 和后面各项互不依赖，攒进同一个并发批次（见下面 tasks）
        after_batch: list = []           # 依赖批次结果（OCR 缓存）的后处理
        tasks = [lambda: self._extract_round(context, image, state),
                 lambda: self._extract_ban(context, image, state)]

        if state.phase in ("prepare", "battle"):
            # 商店开没开，必须用**面板专有元素**判定（"当前概率"行 / "刷新"按钮）。
            # 不能用"卡名 ROI 读到了东西"——商店关闭后那几个位置会读到背景里的
            # 无关文字（实测读出过别的玩家昵称"画个圈圈嗨"和数字"35"），
            # 于是永远判成"开着"，人口永远读不到，上场一直被锁在保守模式。
            with PERF.stage("shop_open"):
                state.shop_open = bool(
                    _hit(context, image, ROI_SHOP_PROB, "当前概率")
                    or _hit(context, image, ROI_FREE_REFRESH, "刷新"))
                if not state.shop_open and self._card_pick_due(state):
                    # 商店面板不在，但那个位置可能是**效果牌触发的"卡牌N选1"界面**
                    # （吸纳人才/3阶英雄兑换券/惊喜补贴…），它**占着商店的位置**，
                    # 不选就一直挂着，后续流程全卡死（2026-09-22 实战 109 帧卡在 unknown）。
                    self._detect_card_pick(context, image, state)
            # **手牌的 16 个滑窗拆成独立任务**进并发批次（结果落进 _OCR_CACHE），
            # 批次结束后再跑 _extract_hand —— 它的每次读都会命中缓存。
            # 原来 _extract_hand 整个作为一个任务跑在池子里，池内不能再并发，
            # 16 次 OCR 只能串行（实测 777ms），是一帧里最长的那条路径。
            hx0, hy0, hw, hh = ROI_HAND_STRIP
            if _onnx_ocr.enabled():
                tasks.append(lambda: _ocr_boxes(context, image, (hx0, hy0, hw, hh)))
            else:
                tasks += [(lambda wx=wx: _ocr_boxes(context, image, (wx, hy0, HAND_SCAN_WIN, hh)))
                          for wx in range(hx0, hx0 + hw - HAND_SCAN_WIN + 1, HAND_SCAN_STEP)]
            after_batch.append(lambda: self._extract_hand(context, image, state))
            tasks += [lambda: self._extract_economy(context, image, state),
                      lambda: self._extract_players(context, image, state),
                      lambda: self._extract_population(context, image, state),
                      lambda: self._extract_lord_level(image, state, context),
                      lambda: setattr(state, "awaiting_target",
                                      _hit(context, image, ROI_DESIGNATE, "指定")),
                      lambda: setattr(state, "hero_panel_open",
                                      _hit(context, image, ROI_HERO_PANEL,
                                           "出售|剩余数量")),
                      lambda: self._extract_talent_pending(context, image, state),
                      lambda: setattr(state, "lord_maxed",
                                      _hit(context, image, ROI_LORD_MAXED, "满级"))]
            if state.shop_open:          # 关着就别读了，省 12 次 OCR
                tasks.append(lambda: self._extract_shop(context, image, state))
            else:
                state.shop = []
                # 商店关着才看得见棋盘 —— 这时候才有意义去观测占用
                if state.phase == "prepare":
                    tasks.append(lambda: self._extract_board_units(context, image, state))
                    # 装备件数：纯像素运算，不占 OCR
                    tasks.append(lambda: state.equip_counts.update(
                        detect_equip_counts(image)))
                    tasks.append(lambda: state.unit_anchors.update(
                        detect_unit_anchors(image)))
                    tasks.append(lambda: _vision_fill_anchors(image, state))
                    if self._levels_due(state):
                        tasks.append(lambda: self._last_levels.update(
                            self._guard_levels(read_unit_levels(context, image))))
                        tasks.append(lambda: self._last_awaken.update(read_awaken_progress(image)))
                    state.unit_levels = self._last_levels
                    state.awaken_progress = self._last_awaken
        elif state.phase == "card_pick":
            tasks.append(lambda: self._detect_card_pick(context, image, state))
        elif state.phase == "auction":
            tasks.append(lambda: self._extract_auction(context, image, state))
        elif state.phase == "talent":
            tasks.append(lambda: self._extract_talents(context, image, state))
        elif state.phase == "lord_boost":
            tasks.append(lambda: self._extract_lord_boost(context, image, state))

        # 各提取项写的是 state 上**互不相同**的字段，缓存也是各写各的 key，
        # 所以可以直接并发跑。一帧的墙钟时间 ≈ 最慢的那一项，而不是全部之和。
        # （PERF 埋点在并发下没法按项拆分，统一记成一个 "extract" 阶段。）
        with PERF.stage("extract"):
            _map_parallel(lambda f: f(), tasks)
            self._finalize_level_cost(state)
            for f in after_batch:
                f()
            self._apply_pop_white(state)
            # **等级下限必须在并发批次之后算**：人口和等级在同一批里跑，
            # 读等级时 population_cap 可能还没算出来（先后顺序不定）。
            if state.population_valid and state.population_cap > 0:
                from agent.decision.meta import level_floor_from_pop_cap
                floor = level_floor_from_pop_cap(state.population_cap - self.pop_bonus)
                if floor > state.level:
                    logger.info("徽章只读到 %d 级，但人口上限 %d 说明至少 %d 级，按 %d 算",
                                state.level, state.population_cap, floor, floor)
                    state.level = floor
                    self._last_lord_level = floor
            if state.lord_maxed:
                # 满级后点数徽章换成了「满级」字样，数点数会读成 1；升级费还粘着旧值（实战 18）。
                # 两个都在这里纠正 —— 否则决策层会一直"想升级、找不到按钮"，钱一分都花不出去
                from agent.decision.meta import MAX_LORD_LEVEL
                state.level = self._last_lord_level = MAX_LORD_LEVEL
                state.level_cost, state.level_btn = -1, None
                self._last_level_cost, self._last_level_btn = -1, None

        logger.debug("状态提取: phase=%s round=%s gold=%s",
                     state.phase, state.round_no, state.gold)
        return state

    # ---------- 各项提取 ----------

    def _extract_talent_pending(self, context, image, state: GameState) -> None:
        """左侧「天赋未领取」：天赋三选一被收起来了，要点它才会重新弹出。"""
        for t, box in _ocr_boxes(context, image, ROI_TALENT_PENDING):
            if "领取" in t:
                state.talent_pending = [box[0] + box[2] // 2, box[1] + box[3] // 2]
                return

    def _extract_round(self, context, image, state: GameState) -> None:
        text = _ocr(context, image, ROI_ROUND)
        m = re.search(r"第\s*(\d+)\s*回合", text)
        r_new = int(m.group(1)) if m else None
        last = self._last_round or 0
        ok = r_new is not None and 1 <= r_new <= 40 and (last == 0 or last <= r_new <= last + 2 or (r_new <= 2 and last <= 2))
        if r_new is not None and not ok and 1 <= r_new <= 40:
            # 2026-10-19 第 787 局（太乙被换下、铠被换上，不合理）：第 16 回合一帧错读成 1 被"新局 1~2"放行，
            # 之后 16 / 19 都按"跳太多"拒收、卡在 1 共 404 帧 → 太乙被当成第 12 回合前的后期件让位。
            # 改：跳变（往回到 1~2 / 往前跳 >2）要连续 3 帧读到同一个数才接受（真新局 / 卡住后能自愈）
            v = getattr(self, "_round_votes", (None, 0))
            v = (r_new, v[1] + 1) if v[0] == r_new else (r_new, 1)
            self._round_votes = v
            if v[1] >= 3:
                logger.info("回合读数 %d 连续 %d 帧一致（上次 %d）→ 接受", r_new, v[1], last)
                ok = True
        else:
            self._round_votes = (None, 0)
        if r_new is not None and not ok:
            # **回合数钳制**（2026-10-18：live_1791345992 读成 114 进了题面"round 114"；近 3 局各有 1 帧 >25）：
            # 只接受 = 上次 / +1~2；跳变要连续 3 帧一致；单帧 OCR 错读沿用上一帧
            logger.info("回合读数 %d 不可信（上次 %d，原文 %r），沿用上一帧", r_new, last, text)
            m = None
        if m:
            state.round_no = r_new
            self._last_round = state.round_no
        else:
            # 回合切换动画（+8 收益横幅）会盖住回合数，沿用上一帧的值
            state.round_no = self._last_round
        # 倒计时只在"第N回合"**后面**找（2026-10-20：原来整串找 `数字+[s5]`，第 15 / 25 回合的"15"被拆成 1+"5" → 倒计时读成 1，
        # 整回合以为只剩 1 秒；"5" 是 OCR 把 s 认成 5 的兜底，"8s"→"85" 仍按 8 读）
        _mr = re.search(r"第\s*\d+\s*回合", text)
        rest = text[_mr.end():] if _mr else (text.split("|", 1)[1] if "|" in text else "")
        m = re.search(r"(\d+)\s*[s5]", rest)
        if m and 0 <= int(m.group(1)) <= 120:
            state.countdown = int(m.group(1))

    def _extract_ban(self, context, image, state: GameState) -> None:
        # "禁：" 和派系名常被 OCR 分成多个文本块，拼接后再解析。
        # 读条阶段禁用信息在左下角（对局内则在右上角）——读条时就拿到，
        # 比等进对局再读更早，阵容选择第一回合就能用上。
        roi = ROI_LOADING_BAN if state.phase == "loading" else ROI_BAN
        text = "".join(_ocr_multi(context, image, roi))
        # 两种写法都要吃：对局内右上角 "禁：河洛"、读条界面左下角 "本局禁用：河洛"
        # （原正则 r"禁[:：]?" 在后者上会把"用"当成派系名）
        m = re.search(r"禁(?:用)?[:：]?\s*([^\s:：，。]+)", text)
        if m:
            faction = re.sub(r"[^一-鿿]", "", m.group(1))  # 滤掉 OCR 噪声字母
            # 规范成卡牌库派系名：OCR 读出「大河口」「三分」，库里是「大河流域」「三分之地」——
            # 原来直接比字符串，这两个派系的禁用在阵容过滤 / 商店过滤里一直没生效（2026-09-29 查出）
            from agent.decision.routes import norm_banned
            faction = norm_banned(faction)
            if faction:
                state.banned_faction = faction
                self._last_ban = faction
        if not state.banned_faction:
            # 禁用派系**一局之内不会变**，读到一次就记住。
            # 商店关闭时右上角那行读不到，之前会变成空 → 阵容判定反复横跳。
            state.banned_faction = self._last_ban

    def _extract_economy(self, context, image, state: GameState) -> None:
        # 5 个互不依赖的 ROI 一起发（能量/己方能量条/升级区/刷新按钮/下回合收入）
        gold_t, energy, label, income = _map_parallel(
            lambda job: job(),
            [lambda: _ocr(context, image, ROI_GOLD, expected=r"\d{1,4}"),
             lambda: _ocr(context, image, ROI_ENERGY),
             lambda: _ocr(context, image, ROI_FREE_REFRESH),
             lambda: _ocr(context, image, ROI_INCOME)])
        boxes = _ocr_boxes(context, image, ROI_LEVEL_AREA)

        # 能量先用白字读法（见 _read_white_digits：图标是红的、数字是白的）；
        # 读不出再退回原来的整块读 + 紧 ROI 重读
        white = _read_white_digits(context, image, ROI_GOLD_WHITE)
        state.gold = white if white >= 0 else max(0, _to_int(gold_t, 0))
        if white < 0 and state.gold > 99:
            # 能量是 1-2 位数，读出 3 位说明把图标/装饰认成数字了，换更紧的 ROI 重读
            retry = _to_int(_ocr(context, image, ROI_GOLD_TIGHT, only_digits=True))
            state.gold = retry if 0 <= retry <= 99 else -1  # -1 = 读数不可信，决策层禁止花钱
        m = re.match(r"(\d+)\s*/\s*(\d+)", energy)
        if m:
            state.energy, state.energy_max = int(m.group(1)), int(m.group(2))
        # 升级费用：整片区域一次 OCR，找到"升级"文字块后取它左边、纵向重叠的数字块
        btn = next((b for txt, b in boxes if "升级" in txt), None)
        if btn is not None:
            bx, by, _, bh = btn
            best, best_dx = -1, 999
            for txt, b in boxes:
                digits = re.sub(r"\D", "", txt)
                if not digits or b[0] >= bx:
                    continue                      # 只看按钮左侧
                if b[3] < 10:
                    continue                      # 高度过滤：徽章数字 h≈14-21，
                                                  # 更矮的是 "+1人口" 图标碎片等噪声
                                                  # （实测噪声 '1' h=6 曾把真实费用 '5' 顶掉）
                if b[1] + b[3] < by or b[1] > by + bh:
                    continue                      # 纵向要有重叠
                dx = bx - b[0]
                if dx < best_dx and 0 < int(digits) <= 99:
                    best, best_dx = int(digits), dx
            if best <= 0:
                # 认出了"升级"却没认出左边那个数字 —— 费用徽章是**白字压在暗红底**上，
                # 和商店「墨子」压在金剑上是同一类失效（见 _ocr_white_text）。
                # 2026-09-23 实战：画面上明明写着「13 升级」，我们一路读成 -1，
                # 于是攒了 26 能量一次级都没升。抠白字重读那一小块。
                pad = 6
                roi = (max(0, ROI_LEVEL_AREA[0]), max(0, by - pad),
                       max(1, bx - ROI_LEVEL_AREA[0]), bh + pad * 2)
                digits = re.sub(r"\D", "",
                                _ocr_white_text(context, image, roi) or "")
                if digits and 0 < int(digits[:2]) <= 99:
                    best = int(digits[:2])
                    logger.info("升级费抠白字才读出来：%d", best)
            if best > 0:
                state.level_cost = best
            # 按钮实测坐标：**写死的 (95,550) 不能用**。左下角那一片内容是会变的——
            # 2026-09-22 第二局实测，英雄详情面板一开，那个位置就变成「+1 出售」按钮，
            # 盲点"升级"等于**把英雄卖了**。按项目既有规矩：按钮一律运行时 OCR 定位。
            # **点按钮左侧的费用圆，不点"升级"二字**：阿离就站在按钮正上方，腿压在字上，
            # 点字会点中棋手、弹出棋手详情面板（2026-09-23 第 9 回合：升级没升上，
            # 这回合的升级机会被这一下吃掉；前几局日志里也多次"升级后紧接着弹棋手面板"）。
            # **按"升级"框的左边缘定位**，不按框中心：OCR 有时把费用和文字连成一个框
            # （「13升级」，左边缘就是数字），有时只框住"升级"二字（左边缘紧挨费用圆）。
            # 原来用"框中心 - 38"，连成一个框时就偏到了按钮左边缘外（2026-09-23 点到 (50,546)，
            # 弹出了棋手面板）。两种情况下"左边缘附近"都是费用圆。
            btn_txt = next((t for t, b in boxes if b is btn), "")
            tap_x = bx + 12 if re.search(r"\d", btn_txt) else bx - LEVEL_BTN_TAP_DX // 3
            # 限在实测范围里（成功点击 x=59~66；2026-09-30 第 21 局 OCR 框切歪算出 x=36，没点中，这回合没升上级）
            tap_x = min(max(tap_x, 55), 70)
            state.level_btn = (tap_x, by + bh // 2)
            self._last_level_btn = state.level_btn
        else:
            # 这一帧没读到"升级"二字。两种情况要分开，**不能一律当成"按钮不在"**：
            #   a) 那块区域显示的是别的东西（英雄详情面板的「出售」/「剩余数量」）
            #      → 真·危险，盲点会把英雄卖掉，必须拒绝。
            #   b) 什么都没读出来（小字号 OCR 抖动）→ 按钮多半还在原地。
            # 原来不分情况一律置 None，结果 23 帧里只有 2 帧能升级，
            # 能量一路攒到 31 也花不出去（2026-09-22 发现）。
            area_text = "".join(t for t, _ in boxes)
            if re.search(r"出售|剩余数量|出局", area_text):
                state.level_btn = None
                self._last_level_btn = None
            elif self._last_level_btn and state.shop_open:
                # **只在商店开着时沿用**：升级按钮基本是商店面板的一部分
                # （实测 商店开 15/25 帧看得到、商店关 5/41）。商店关着时沿用旧位置，
                # 点下去是站在那儿的阿离 —— 弹棋手面板（2026-09-23 第 6 回合）。
                state.level_btn = self._last_level_btn
        # 升级费用**粘住**：这个徽章数字很小，商店面板开着时那一片会被压暗，
        # 实战 23 帧里有 15 帧读不出来（level_cost=-1）。而 decide_economy 见到 -1
        # 就直接不升级——于是"攒了一堆能量却整局 1 级"（2026-09-22 发现）。
        # 它在同一等级内是常数，读到过就沿用，直到下次读到新值或新一局清掉。
        if state.level_cost is not None and state.level_cost > 0:
            self._last_level_cost = state.level_cost
        elif self._last_level_cost > 0:
            state.level_cost = self._last_level_cost
        # 刷新按钮：文字（免费/付费）用宽 ROI 上面已读，数字角标要紧 ROI 单独读
        if "刷新" in label:
            # 角标读出来是「×2」，前面常粘一个被认成 "(" / "1" 的刷新箭头（'1×5' 其实是 5）：
            # 取 × 后面的数字；没有 × 才退回"全部数字"
            badge = _ocr(context, image, ROI_REFRESH_BADGE) or ""
            m_x = re.search(r"[×xX]\s*(\d{1,2})", badge)
            n = int(m_x.group(1)) if m_x else _to_int(re.sub(r"\D", "", badge))
            if 0 <= n <= 99:
                if "免费" in label or "临时" in label:
                    # 「临时刷新 ×3」（天赋无影手送的）也是不花能量的刷新，原来只认「免费」→ 被读成"刷新要 3 能量"
                    # （2026-10-14 天赋复盘：第 616 / 620 局无影手几乎没兑现，620 局 R4~R10 有 5 个回合一次没刷；证据帧
                    #  live_1791899709/phase_prepare_1791900289.png）
                    # 第 1060 局 R3：单帧把「×3」读成 31 → 经济题写"31 free"。两位数的免费次数要连续两帧一样才信
                    _prev_fr = getattr(self, "_fr_raw_prev", None)
                    self._fr_raw_prev = n
                    if n < 10 or _prev_fr == n:
                        state.free_refresh = n
                else:
                    state.refresh_cost = n
        m = re.search(r"\d+", income)
        if m:
            state.income_next = int(m.group())

    def _detect_card_pick(self, context, image, state: GameState) -> None:
        """识别"卡牌N选1"界面：商店位置上摆着 2~4 张**没有价格**的效果牌。

        判据：卡名带能在卡牌库里查到（效果牌居多），且商店面板标志（刷新/当前概率）不在。
        点击点用卡名 OCR 框的实测坐标，不写死槽位。
        """
        state.card_pick = False
        state.card_options = []
        # **先排除商店**。文档里一直写着判据含"且没有商店面板标志"，但代码从来没实现过。
        # 后果（2026-09-22 第 N 局 ledger 实锤）：N选1 的卡名带 y≈215 和商店卡名带
        # y=222~270 **在 y 上重叠**，光凭"卡名查得到"，整排商店卡都会被当成候选。
        # 而 detect_phase 对 card_pick 的复验**只走这个检测器**（它没有文字判据），
        # 于是一局里出现过一次真的 N选1 之后就**永久卡在 card_pick**：
        #   回合3「N选1 选 明世隐」连打 4 次，候选从 2 张涨到 5 张（= 商店那一排）；
        #   回合7 选的是韩信/太乙真人，到手的却是候选里的**花木兰**
        #   —— 因为点的根本是商店卡面，等于在盲买。
        #   现象是经常买一些莫名其妙的英雄、有啥买啥疯狂买。
        # 判据：商店面板标志（刷新 / 当前概率）可见 = 这是商店。真的 N选1 会把它们盖住
        #      （证据：真 N选1 帧读到的候选是 2~3 张，从来没读到商店那 4 张）。
        # **拍卖界面同样要排除**：它那一排拍品的卡名也落在这条带里
        # （2026-09-23 实战：拍卖帧被认成「N选1」，候选就是那 6 个拍品）。
        # 这条和上面商店那条是同一类漏洞 —— 这个检测器只说了"画面上有什么"
        # （几个查得到的卡名），没说"不是什么"，而它又是**兜底分支**，
        # 于是任何"摆着一排卡"的界面都会被它认领。每多一个这样的界面就要多一条排除。
        for roi, pat in ((ROI_FREE_REFRESH, "刷新"), (ROI_SHOP_PROB, "当前概率"),
                         (ROI_AUCTION_TITLE, AUCTION_BANNER_RE), (ROI_AUCTION_BID_BTN, "出价")):
            if re.search(pat, _ocr(context, image, roi) or ""):
                return
        # **不能用商店的固定槽位**（x=38/300/560/819、卡名 y=222）：
        # N 选 1 界面的卡是**按实际张数居中排布**的，3 张时实测在 x≈372/640/906、
        # 卡名 y≈215，和商店槽位完全对不上，按槽位扫一个都扫不到
        # （2026-09-22 第六局：锦囊弹出三选一，界面挂着没人接，整局卡死）。
        # 改成滑动窄窗口扫卡名带：窗口在 x 上仍然窄（小字号才检测得住），
        # 步长重叠，不假设张数和位置；y 用一条能同时覆盖两种布局的带。
        band_y, band_h = 195, 80          # 覆盖 N选1(215) 与商店卡名(222-270)
        win_w, step = 200, 100
        opts: list[dict] = []
        # x 只扫到 RIGHT_TALENT_COL_X：右侧那一列是**已获得的天赋**，名字正好也在这条带上，
        # 原来扫到 1280 —— 2026-09-23 把「翩跹舞步·蓄势」「嬴政」当成 N 选 1 候选、反复点天赋列
        # N 选 1 有 3/4/5 张（2026-09-23 确认），五张时最右那张会越过 1080，所以扫满整条；
        # 越界的候选在下面按"和左边几张等距"再验一次，天赋列里零星读到的名字过不了这一关。
        windows = [(x, band_y, win_w, band_h)
                   for x in range(0, 1280 - win_w + step, step)]
        for boxes in _map_parallel(lambda r: _ocr_boxes(context, image, r), windows):
            for txt, box in boxes:
                if len(re.findall(r"[一-鿿]", txt)) < 2:
                    # **单字英雄名（瑶/曜/镜…）要放行**（2026-09-30 第 37 局四选二：「瑶」被这条过滤丢掉，
                    # 加上墨子没读出来，只剩狂铁一张 → 整个界面没认成 N 选 1、卡住）。
                    # 只收"整框就是一个完整卡名、且落在卡名那一行"的，免得单字碎片乱认。
                    t1 = txt.strip()
                    cy = box[1] + box[3] // 2
                    if not (len(t1) == 1 and t1 in self._card_name_set and PICK_NAME_ROW[0] <= cy <= PICK_NAME_ROW[1]):
                        continue
                name = self.resolve_name(txt, keep_raw=False)
                if not name:
                    continue
                cx = box[0] + box[2] // 2
                if any(abs(o["x"] - cx) < 70 for o in opts):
                    continue              # 重叠窗口读到同一张，去重
                opts.append({"name": name, "x": cx,
                             "y": box[1] + box[3] // 2})
        opts.sort(key=lambda o: o["x"])
        inner = [o for o in opts if o["x"] <= RIGHT_TALENT_COL_X]
        outer = [o for o in opts if o["x"] > RIGHT_TALENT_COL_X]
        if outer:
            keep = []
            if len(inner) >= 2:      # 原来要 3 张（第 37 局四选二只认出 2 张，最右的赵云被丢）
                gaps = [b["x"] - a["x"] for a, b in zip(inner, inner[1:])]
                pitch = sorted(gaps)[len(gaps) // 2]
                last = inner[-1]["x"]
                for o in outer:
                    if abs((o["x"] - last) - pitch) <= pitch * 0.25:
                        keep.append(o)
                        last = o["x"]
            if len(keep) != len(outer):
                logger.debug("N选1：右侧 %s 不在等距排列里（多半是已获得天赋列），不算候选",
                             [o["name"] for o in outer if o not in keep])
            opts = inner + keep
        # **商店的牌右上角有价格角标，免费选的没有**（2026-09-23 确认）。
        # 实测（牌名中心往右上 (+70..+125, -200..-145) 那一块的红褐色占比）：
        #   商店 0.24~0.79（每张都有角标）；N 选 1 0.00~0.19。取中位数、阈值 0.22。
        # 这比"商店面板标志（刷新/当前概率）"那条排除直接：那两行字被遮住时照样分得清。
        if len(opts) >= 2:
            ratios = sorted(_price_badge_ratio(image, o["x"], o["y"]) for o in opts)
            if ratios[len(ratios) // 2] >= PRICE_BADGE_MIN_RATIO:
                # **光看"那块是红的"不够，角标里还得有数字**（2026-09-23 第七局：这里又卡死
                # 不选择了）：英雄三选一的卡更大、卡名更靠上，"右上角"那块落在**橙色场馆背景 /
                # 赵云的红色立绘**上，红色占比 0.33/0.27 过了阈值 → 被当成商店、一直不选。
                # 商店的价格角标是"红底白字的数字"，每张都读得出（实测 4/4 读成 '3'）；
                # 立绘和背景读不出数字（实测 0/3）。
                digits = sum(bool(re.fullmatch(r"\d{1,2}", _ocr(
                    context, image, (o["x"] + PRICE_BADGE_DX - 22, o["y"] + PRICE_BADGE_DY - 22,
                                     44, 44)) or "")) for o in opts)
                if digits * 2 >= len(opts):
                    return
                # 没有数字时还要确认**真是一排卡**：N 选 1 的卡名都在 y≈215 那一行
                # （历史 210 帧里 3/4 张的真 N 选 1 全是 214~216）。战斗中飘着的天赋说明字
                # （刘邦/控制感应/膨化生命，y=213/244/254）也会偏红、也没数字，只有一个落在这行。
                if sum(PICK_NAME_ROW[0] <= o["y"] <= PICK_NAME_ROW[1] for o in opts) < 2:
                    return
                logger.info("N选1：卡面右上角偏红（中位 %.2f），但只有 %d/%d 张读得出价格数字 —— "
                            "不是商店", ratios[len(ratios) // 2], digits, len(opts))
        in_row = [o for o in opts if PICK_NAME_ROW[0] <= o["y"] <= PICK_NAME_ROW[1]]
        if len(in_row) == 1 and len(opts) >= 2:
            # 2026-10-15 第 649 局二选一（孙悟空 / 太乙真人）卡了 3 分钟：孙悟空的卡名那一行没读出来，
            # 反而把描述里的「闪现」读成「闪现突击者」(y=265) → "卡名不在同一行"、整个界面认不出。
            # 不在行里的候选：用它的 x 在卡名那一行用矮窗口重读，读得出真卡名就挪进来
            row_y = in_row[0]["y"]
            for o in opts:
                if o in in_row:
                    continue
                # 只救"读到的是下面描述区（比卡名行低 30px 以上）、重读后换成了另一个名字、间距像正常卡距"的
                # （第 649 局：闪现突击者 y=265 → 孙悟空，间距 279）；战斗飘字（开团号令 y=227，间距 379）重读还是它自己，不收
                if o["y"] - row_y < 30 or not (230 <= min(abs(r["x"] - o["x"]) for r in in_row) <= 330):
                    continue
                n = self._pick_reread(context, image, o["x"], row_y)
                if n and n != o["name"] and not any(abs(r["x"] - o["x"]) < 70 for r in in_row):
                    logger.info("N选1：%s(y=%d) 在卡名那一行重读为 %s", o["name"], o["y"], n)
                    in_row.append({"name": n, "x": o["x"], "y": row_y})
            in_row.sort(key=lambda o: o["x"])
            if len(in_row) >= 2:
                opts = list(in_row)
        if len(in_row) >= 2 and len(in_row) < len(opts):
            # 卡面说明文字里也会出现卡名（「开团号令」「分红」，y≈266），它们不是选项 ——
            # 点下去落在说明栏上。真卡名都在同一行（2026-09-23 第八局）。
            logger.info("N选1：%s 不在卡名那一行（是卡面说明里的词），不算候选",
                        [o["name"] for o in opts if o not in in_row])
            opts = in_row
        if len(opts) >= 2 and len(in_row) < 2:
            # **真 N 选 1 的卡名都在 y≈215 那一行**（历史 210 帧里真的 N 选 1 全是 214~216）。
            # 2026-09-23 第八局：战斗开始时的单位详情「机关傀儡」(y=213) + 飘字「开团」(y=227)
            # 被认成二选一，点到了战斗中的棋盘上。
            logger.debug("N选1：%s 的卡名不在同一行（y=%s），不算",
                         [o["name"] for o in opts], [o["y"] for o in opts])
            return
        if len(opts) >= 2:
            opts = self._refine_pick(context, image, opts)
        if len(opts) >= 2:
            state.card_pick = True
            state.card_options = opts
            # 「请选择1/2张卡牌」= 要选两张（2026-09-30 第 37 局：只选了一张就停，界面一直挂着）
            m = re.search(r"选择\s*(\d)\s*/\s*(\d)\s*张", _ocr(context, image, ROI_PICK_QUOTA) or "")
            if m:
                state.card_pick_quota = [int(m.group(1)), int(m.group(2))]
            logger.info("检测到卡牌N选1界面：%s%s", [o["name"] for o in opts],
                        f"（要选 {state.card_pick_quota[1]} 张）" if state.card_pick_quota else "")

    def _pick_reread(self, context, image, x: int, y: int) -> Optional[str]:
        """N 选 1 的一个卡位：矮窗口（<40px，整块识别、不走检测）重读卡名；读不出 / 不是卡名返回 None。"""
        for roi in ((max(0, x - 95), y - 17, 190, 36), (max(0, x - 70), y - 15, 140, 32)):
            t = _ocr(context, image, roi) or ""
            h = re.findall(r"[一-鿿]", t)
            if len(h) >= 2 or self._single_char_hero(t):
                # 保留数字（「4阶英雄」「3阶英雄兑换券」）：只留汉字会被模糊匹配成「5阶英雄」
                n = self.resolve_name(re.sub(r"[^0-9一-鿿]", "", t) if len(h) >= 2 else t, keep_raw=False)
                if n:
                    return n
        return None

    def _refine_pick(self, context, image, opts: list) -> list:
        """N 选 1 候选的二次确认（2026-10-11 识别审查：约 15% 的帧读错 —— 宽窗口把卡名切开，碎片被匹配成别的卡
        「三分征召令」→「三分领袖」、「兑换券」→「换帅」多出一个候选；「墨子」整张漏读）：
        ① 每个候选按卡名那一行用矮窗口整块重读，以它为准；② 相距 <150px 的合并（留重读成功 / 名字更长的）；
        ③ 按读到的间距补中间空出两倍间距的卡位、以及两端再往外一格，矮窗口重读，读得出真卡名才收。"""
        y_row = sorted(o["y"] for o in opts)[len(opts) // 2]
        for o in opts:
            n = self._pick_reread(context, image, o["x"], y_row)
            o["_ok"] = bool(n)
            if n and n != o["name"]:
                logger.info("N选1：%s 重读为 %s（x=%d）", o["name"], n, o["x"])
                o["name"] = n
        opts.sort(key=lambda o: o["x"])
        _g = sorted(b["x"] - a["x"] for a, b in zip(opts, opts[1:]))
        _typ = _g[len(_g) // 2] if _g else 0          # 一般间距（紧凑排布的界面只有 ~120，不能按固定 150 合并）
        merged: list = []
        for o in opts:
            if merged and o["x"] - merged[-1]["x"] < _typ * 0.5:
                a = merged[-1]
                better = o if (o["_ok"], len(o["name"])) > (a["_ok"], len(a["name"])) else a
                logger.info("N选1：%s(x=%d) 和 %s(x=%d) 挨得太近，只留 %s", a["name"], a["x"], o["name"], o["x"], better["name"])
                merged[-1] = better
            else:
                merged.append(o)
        opts = merged
        if len(opts) >= 2:
            gaps = [b["x"] - a["x"] for a, b in zip(opts, opts[1:])]
            pitch = sorted(gaps)[0]
            tries = []
            for a, b in zip(opts, opts[1:]):
                if b["x"] - a["x"] > pitch * 1.6:
                    tries.append((a["x"] + b["x"]) // 2)
            tries += [opts[0]["x"] - pitch, opts[-1]["x"] + pitch]
            for x in tries:
                if not (90 <= x <= RIGHT_TALENT_COL_X + 60) or any(abs(o["x"] - x) < 120 for o in opts):
                    continue
                n = self._pick_reread(context, image, int(x), y_row)
                if n and all(o["name"] != n or abs(o["x"] - x) > pitch * 0.8 for o in opts):
                    logger.info("N选1：补读到漏掉的卡 %s（x=%d）", n, x)
                    opts.append({"name": n, "x": int(x), "y": y_row, "_ok": True})
            opts.sort(key=lambda o: o["x"])
        for o in opts:
            o.pop("_ok", None)
        return opts

    def _extract_shop(self, context, image, state: GameState) -> None:
        # 商店内容只在买牌/刷新/新回合后变化。文字区像素没变 => 直接复用上一帧结果，
        # 省掉 12 次 OCR（约 740ms）。像素比对是 numpy 逐字节比，亚毫秒级。
        # 注意：必须逐格窄 ROI 读，合并成宽带会让 OCR 检测丢掉小字号（实测费用只剩 2/4）。
        if not self._region_changed("shop", image, SHOP_NAME_CELLS) and self._shop_cache:
            state.shop = [ShopSlot(**vars(s)) for s in self._shop_cache]
            return

        # 12 个窄 ROI（4 卡 × 名/费用/派系）互不依赖 → 并发读，实测省约 2/3 时间
        jobs: list[tuple[str, tuple[int, int, int, int]]] = []
        for x0 in SHOP_CARD_X:
            jobs.append(("name", (x0 + 30, ROI_SHOP_NAME_Y[0], 190, ROI_SHOP_NAME_Y[1])))
            jobs.append(("cost", (x0 + 195, ROI_SHOP_COST_Y[0], 58, ROI_SHOP_COST_Y[1])))
            jobs.append(("faction",
                         (x0 + 60, ROI_SHOP_FACTION_Y[0], 130, ROI_SHOP_FACTION_Y[1])))

        def _read(job):
            kind, roi = job
            if kind == "cost":
                return _ocr(context, image, roi, expected=r"\d{1,2}", only_digits=True)
            return _ocr(context, image, roi)

        texts = _map_parallel(_read, jobs)

        state.shop = []
        for i in range(len(SHOP_CARD_X)):
            name = texts[i * 3]
            cost = _to_int(texts[i * 3 + 1])
            faction = texts[i * 3 + 2].strip("·- ")
            piece = self.resolve_name(name, keep_raw=False, pool=getattr(self, "_shop_names", None)) or None
            if not (0 <= cost <= 99) and name:
                # 价格读空约 30%（48px 高走检测，单个数字检测不出来；面板上下偏 ~20px 还会切掉角标）→ 矮窗口整块再读
                c2 = _to_int(_ocr(context, image, (SHOP_CARD_X[i] + 205, 58, 50, 36), expected=r"\d{1,2}", only_digits=True))
                if 0 <= c2 <= 99:
                    cost = c2
            if piece is None:
                # 认不出来**先别当它不存在**：卡名压在亮底上时 OCR 会整段丢掉
                # （2026-09-23 实测「墨子」压在金剑上，连读三帧全空）。
                # 抠白字重读一次；派系/费用读得到更说明这格**有牌**，不是空位。
                retry = _ocr_white_text(context, image,
                                        (SHOP_CARD_X[i] + 30, ROI_SHOP_NAME_Y[0],
                                         190, ROI_SHOP_NAME_Y[1]))
                piece = self.resolve_name(retry, keep_raw=False, pool=getattr(self, "_shop_names", None)) or None
                if piece:
                    logger.info("商店第%d格卡名抠白字才读出来：%s（原读 %r）",
                                i + 1, piece, name)
                    name = retry
            state.shop.append(ShopSlot(
                index=i,
                piece=piece,
                cost=cost if 0 <= cost <= 99 else None,
                faction=faction,
                sold=not name,
            ))
        if state.shop and all(s.sold for s in state.shop):
            # 4 格同时读空 = 这一帧读不出（翻牌 / 淡入动画、面板偏移），不是全部售空（识别审查 2026-10-11）
            for s in state.shop:
                s.sold = False
            state.shop_unreadable = True
            return
        self._shop_cache = [ShopSlot(**vars(s)) for s in state.shop]

    # 徽章带的像素格：只罩己方落点正上方那几小块，不要用大包围盒
    # （包围盒会跨过会动的 3D 模型/特效，缓存命中率归零——同 _region_changed 的注释）。
    BADGE_CELLS = [(x - 22, y - BADGE_DROP_DY - 14, 44, 28)
                   for x, y in OWN_BOARD_DROPS_FRONT + OWN_BOARD_DROPS_BACK]

    def _extract_board_units(self, context, image, state: GameState) -> None:
        """看一眼棋盘：哪些己方落点上真的站着人。

        整套扫描约 700-840ms（48 个滑窗），但**占用只在上场/卖出/阵亡时变**，
        所以用像素缓存：徽章带那几小块没变就复用上一次结果，稳态下几乎不花钱。
        """
        if not BOARD_OBS_ENABLED:
            return                  # 没看 -> board_seen 保持 False，决策层不会误用空观测
        state.board_seen = True
        if not self._region_changed("badges", image, self.BADGE_CELLS)                 and self._board_cache is not None:
            state.board_occupied = list(self._board_cache)
            return
        occ = detect_board_units(context, image)
        self._board_cache = set(occ)
        state.board_occupied = list(occ)
        logger.info("棋盘观测：%d 格有人 %s", len(occ), sorted(occ))

    def _resolve_hand_fragment(self, text: str) -> Optional[str]:
        """叠放时每张牌只露出卡名的**前一截**（「孙悟」「干将」「刘」）。按前缀认：

        1. 完整卡名 / 模糊匹配（最后一张牌总是完整露出的）；
        2. 在**小范围候选**里找唯一前缀：最近在手里认出过的牌 + 商店里刚见过的牌
           （单字「刘」在全库里有 刘邦/刘禅，但这局手里/商店里多半只有「刘邦」）；
        3. 全库里唯一前缀（「末」全库只有「末世」、「干将」只有「干将莫邪」）。
        都不行返回 None —— 交给点开/长按去看，**不猜**。
        """
        t = re.sub(r"[^一-鿿·A-Za-z0-9]", "", text or "")
        if not t:
            return None
        if t in self._hand_name_set:
            return t
        if len(t) >= 2:
            full = self.resolve_name(t, keep_raw=False, pool=self._hand_names)
            if full and len(full) <= len(t) + 1:
                return full
        # 先在**手牌账**里找（刚买的蒙犽只露出「蒙」，全库有蒙犽/蒙恬两张，账里只有蒙犽）
        for p in dict.fromkeys([t, t[:-1]] if len(t) >= 2 else [t]):
            cands = [n for n in self.hand_hint if p and n.startswith(p)]
            if len(cands) == 1:
                return cands[0]
        recent = set(self._hand_seen) | {s.piece for s in self._shop_cache if s.piece}
        # **露出来的是卡名的后半截**（2026-10-11 第 573 局：芈月只露出「月」，点开看又失败，整回合留在手里卖不掉）：
        # 按结尾认，只在手牌账、再在最近见过的牌里找唯一（全库按结尾猜太宽）
        for pool in (list(self.hand_hint), list(recent)):
            cands = sorted({n for n in pool if n and n.endswith(t) and n != t})
            if len(cands) == 1:
                return cands[0]
        allnames = [n for n in self._hand_name_set if self._card_cats.get(n) != "召唤物"]
        # 露出来的**最后一个字常被下一张牌挡掉一半**（实测「干将」读成「干卷」、「孙悟」只剩「孙」），
        # 所以同时试"去掉最后一个字"的前缀。
        for p in dict.fromkeys([t, t[:-1]] if len(t) >= 2 else [t]):
            if not p:
                continue
            cands = [n for n in recent if n.startswith(p)]
            if len(cands) == 1:
                return cands[0]
            cands = [n for n in allnames if n.startswith(p)]
            if len(cands) == 1:
                return cands[0]
        # **露出来的是描述不是卡名**（叠放 / 卡名被挡：第 43 局古币露出「指定1名【河洁】英雄等级+2。」，
        # 认不出 → 抓牌校验连败 3 次被搁置，转瞬古币白扔）：按卡面描述反查，**只在手牌账 + 最近见过的牌里找**
        # （全库里「红笺锦鲤符」0.80 和古币 0.83 挨得太近，不能全库猜）。
        tc = re.sub(r"[^一-鿿0-9]", "", text or "")
        if len(tc) >= 6:
            import difflib
            pool = (set(self.hand_hint) | recent) & set(self._card_descs)
            sc = sorted(((difflib.SequenceMatcher(None, tc, re.sub(r"[^一-鿿0-9]", "", self._card_descs[n])).ratio(), n)
                         for n in pool), reverse=True)
            if sc and sc[0][0] >= 0.75 and (len(sc) == 1 or sc[0][0] - sc[1][0] >= 0.05):
                return sc[0][1]
        return None

    def _hand_name_closed(self, image, roi, ocr_text: str, icon_x: Optional[float] = None) -> Optional[str]:
        """手牌一张牌的卡名：**服务器版认字 + 闭集解码**（agent/perception/hand_names.py）优先。
        它只会输出真卡名，读不准就说读不出（测试集难例：读错 0，原自由解码读错 7/60 —— 雷鸣刃→驱雷、力量腰带→新生力量、亚瑟→业热）。
        只认出多张牌共用的前缀 → 交给原来的"最近见过/商店里的牌"唯一化；读取器不可用时退回原读法。
        读取器读不出、而自由解码"认出"了一张牌时，**只在它和读取器给的前缀一致时**才采信（宁缺勿错，读不出的交给点开看）。"""
        from .hand_names import HandNameReader
        reader = HandNameReader.get() if _onnx_ocr.enabled() else None
        if reader is None:
            return self._resolve_hand_fragment(ocr_text)
        import cv2
        x0, y0, w, h = roi
        crop = cv2.cvtColor(np.ascontiguousarray(image[y0:y0 + h, x0:x0 + w]), cv2.COLOR_RGB2BGR)
        try:
            name, prefix, _m = reader.read(crop, w)
        except Exception:                                   # noqa: BLE001
            logger.exception("手牌卡名读取器出错，这一张退回原读法")
            return self._resolve_hand_fragment(ocr_text)
        if name:
            return name
        # 读字定不了 → **看立绘**（叠放时卡名被盖住，露出来的是左边一截立绘；hand_names.HandArtReader）。
        # 名字条露出了前缀时，立绘结果必须和前缀一致才采信（叠放牌上实测一致 753 / 矛盾 9）。
        art = ""
        if icon_x is not None:
            from .hand_names import HandArtReader
            ar = HandArtReader.get()
            if ar is not None:
                try:
                    art, _p = ar.read(image, icon_x)
                except Exception:                           # noqa: BLE001
                    logger.exception("手牌立绘识别出错（忽略）")
                    art = ""
        if not art and icon_x is not None and self.hand_book_hint:
            # 2026-10-16 晚第 678 局：捕能手套第 4~5 回合一直"认不出"（立绘把握没过 0.7、卡名条读到说明文字），规则没法拖它。
            # 账本里有、画面还没认出来的那几张（规则每帧同步）是很小的闭集：立绘第一选择落在里面且 ≥0.4 就采信
            from .hand_names import HandArtReader as _HAR
            ar = _HAR.get()
            if ar is not None:
                try:
                    top, pt = ar.read_top(image, icon_x)
                except Exception:                           # noqa: BLE001
                    top, pt = "", 0.0
                if top and top in self.hand_book_hint and pt >= 0.4 and (not prefix or top.startswith(prefix)):
                    logger.info("手牌 x=%d 卡名读不出，立绘 %s(%.2f) 在账本待认名单里 → 采信", int(icon_x), top, pt)
                    return top
                if top:
                    logger.info("手牌 x=%d 认不出：立绘第一选择 %s(%.2f)，账本待认 %s", int(icon_x), top, pt, sorted(self.hand_book_hint)[:4])
        if prefix:
            if art and art.startswith(prefix):
                return art
            n = self._resolve_hand_fragment(prefix)
            if n:
                return n
            old = self._resolve_hand_fragment(ocr_text)
            return old if old and old.startswith(prefix) else None
        return art or None

    def _extract_hand_by_layout(self, context, image, state: GameState, layout: dict) -> None:
        """按图标布局逐张读手牌（见 HAND_* 常量上面的说明）。"""
        xs, icon_y, n = layout["xs"], layout["icon_y"], layout["n"]
        name_y = icon_y + HAND_NAME_DY
        grip_y = name_y + HAND_GRIP_DY
        half = 15
        # 左边的枫叶舞槽（棋手技能牌）：不在网格里，单独读
        skill_cx = int((xs[0] if xs else HAND_CENTER_ICON_X + HAND_PITCH_MAX / 2) - SKILL_NAME_DX)
        jobs = [("skill", (skill_cx - 62, name_y - half, 124, half * 2))]
        for k, ix in enumerate(xs):
            left = ix - HAND_ICON_TO_LEFT
            right = (xs[k + 1] - HAND_ICON_TO_LEFT) if k + 1 < n else left + HAND_PITCH_MAX
            jobs.append((k, (int(left) + 3, name_y - half, max(8, int(right - left) - 6), half * 2)))
        texts = _map_parallel(lambda j: _ocr(context, image, j[1]), jobs)
        now = time.time()
        for (k, roi), t in zip(jobs, texts):
            if k == "skill":
                name = self.resolve_name(t, keep_raw=False) if t else None
                if name and self._card_cats.get(name) == "棋手技能牌":
                    slot = {"name": name, "x": skill_cx, "y": grip_y, "name_y": name_y,
                            "ready": _maple_ready(image, skill_cx, grip_y)}
                    state.hand.append(name)
                    state.hand_slots.append(slot)
                continue
            cx = roi[0] + roi[2] // 2              # 抓手点 = 露出来那一截的中心
            name = self._hand_name_closed(image, roi, t, icon_x=xs[k])
            if name:
                self._hand_seen[name] = now
                state.hand.append(name)
                state.hand_slots.append({"name": name, "x": cx, "y": grip_y, "name_y": name_y})
            else:
                state.hand_unread.append([cx, grip_y, t or "", int(xs[k])])
                logger.info("手牌第 %d 张（x=%d）露出来的字「%s」认不出是哪张，待点开看",
                            k + 1, cx, t)
        # **觉醒卡：模型很有把握就直接标上**（不再等点开确认）。手牌觉醒识别测试集 11/0/0（hand_awaken_v1），
        # 卡名也已读出 —— 点开只是多花 1 秒；2026-09-25 第四轮第 5 局觉醒孙悟空因为"先点开确认"又晚了 3 秒，期间刘邦占了人口。
        try:
            for x0, x1, prob, _d in _hand_awaken_by_model(image, xs) or []:
                if prob < 0.99:
                    continue
                for sl in state.hand_slots:
                    if x0 <= sl["x"] <= x1 and self._card_cats.get(sl["name"]) == "英雄牌":
                        sl["awakened"] = True
        except Exception:                                   # noqa: BLE001
            logger.exception("手牌觉醒识别出错（忽略，退回点开确认）")
        # 最近见过的名字只留 2 分钟（换回合后商店/手牌都变了，老候选只会造成误认）
        for nm in [nm for nm, ts in self._hand_seen.items() if now - ts > 120]:
            self._hand_seen.pop(nm, None)

    def _extract_hand(self, context, image, state: GameState) -> None:
        state.hand = []
        state.hand_slots = []
        state.hand_unread = []
        try:
            m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})", _ocr(context, image, ROI_HAND_COUNT) or "")
            if m and 0 <= int(m.group(1)) <= int(m.group(2)) <= 20:
                state.hand_count = [int(m.group(1)), int(m.group(2))]
        except Exception:                                           # noqa: BLE001
            pass
        if _onnx_ocr.enabled():
            layout = hand_layout(image)
            # 布局可信才用：有角标定位就信；只靠图标时命中要过半（旧版 UI 帧/被面板挡住时退回整条 OCR）
            if layout and layout["n"] >= 1 and (layout["by"] != "图标"
                                                or layout["hits"] * 2 >= layout["n"]):
                state.hand_by_layout = True
                self._extract_hand_by_layout(context, image, state, layout)
                return
        # **滑窗窄读**：一条 1050px 宽的 ROI 在满手牌时会漏检 ——
        # OCR 的检测阶段对小字号敏感，窗口越宽越容易丢掉边缘/偏暗的卡名。
        # 实测代价很实在：2026-09-22 一局里"手牌 4 张"而画面是 11 张，
        # 另一局刚买到的主C「曜」被漏读，于是既没上场也没记账、等于凭空消失。
        # 这和拍卖拍品、卡牌N选1 用的是同一招（见 _extract_auction / _detect_card_pick）。
        x0, y0, w, h = ROI_HAND_STRIP
        seen_x: list[int] = []
        boxes = []
        # 用 onnxruntime 后端时：**整条手牌只检测一次**（实测 60ms，7 张全对），不切滑窗 ——
        # 滑窗是给 MaaFW 用的（它在宽 ROI 上会漏检小字），而新后端在窄窗口里文字框容易贴边、
        # 被下面"贴边丢弃"误杀（hand_high_row 2 张只认出 1 张）。
        onnx = _onnx_ocr.enabled()
        windows = ([(x0, w)] if onnx else
                   [(wx, HAND_SCAN_WIN) for wx in range(x0, x0 + w - HAND_SCAN_WIN + 1, HAND_SCAN_STEP)])
        for wx, ww in windows:
            for t, box in _ocr_boxes(context, image, (wx, y0, ww, h)):
                # **贴着窗口边缘的一律丢掉**：那是窗口骑跨在两张卡上读出来的拼接文本
                # （实测把"枫叶舞"的尾巴和"魂祭绝刃"的头拼成了"舞·魂察"）。
                # 真正整张罩住的卡名两侧都会留白，因为步长远小于窗口宽度，
                # 每张卡都必然被某个窗口完整罩住。
                if not onnx and (box[0] <= wx + HAND_SCAN_EDGE
                                 or box[0] + box[2] >= wx + HAND_SCAN_WIN - HAND_SCAN_EDGE):
                    continue
                cx = box[0] + box[2] // 2
                if not onnx and any(abs(cx - px) < HAND_SCAN_DEDUP for px in seen_x):
                    # 去重只对滑窗有意义（同一张卡被相邻窗口各读一次）。整条只检测一次时没有重复读，
                    # 反而会误杀：枫叶舞上方那行「Ali SKILL」和卡名 x 几乎一样，先到的它把卡名顶掉了
                    continue
                seen_x.append(cx)
                boxes.append((t, box))
        boxes.sort(key=lambda tb: tb[1][0])  # 按 x 从左到右
        # **把同一张卡被拆开的碎片先拼回去**：长卡名（「翩跹舞步·新生」7 字）会被
        # OCR 断成两段，后半段"步·新生"又刚好能匹配到另一张真卡「有叶新生」——
        # 于是一张卡变成两张，还是错的两张。
        # 判据是间距：同一张卡的碎片紧挨着（<18px），不同卡之间至少隔着卡面边框。
        merged: list = []
        for t, box in boxes:
            if merged:
                pt, pb = merged[-1]
                # **只拼同一行上的碎片**：枫叶舞卡面上方还有一行英文「Ali SKILL」（读成
                # 'MSKIL'/'e'），和卡名 x 重叠 -> 间距是负的，原来会被当成同一张卡的碎片拼进去，
                # 拼出来的名字查不到卡，整张被丢（onnx 后端整条检测时暴露；滑窗时它常被切掉）。
                same_line = abs((box[1] + box[3] / 2) - (pb[1] + pb[3] / 2)) < 0.5 * max(box[3], pb[3])
                if same_line and box[0] - (pb[0] + pb[2]) < HAND_FRAG_GAP:
                    merged[-1] = (pt + t, [pb[0], min(pb[1], box[1]),
                                           box[0] + box[2] - pb[0], max(pb[3], box[3])])
                    continue
            merged.append((t, list(box)))
        boxes = merged
        for t, box in boxes:
            # 单字过滤有个大坑：卡库里**真有 5 张单字名的卡**（曜/苍/铠/瑶/镜）。
            # 一刀切掉单字，这些英雄就**永远打不出去**——能在商店买到（商店提取没这条过滤），
            # 买回来烂在手里。2026-09-22 手牌里的镜一直留着，根因就在这儿。
            # 改成：单字必须**精确命中卡名表**才放行，其余仍按噪声丢掉
            #（实测噪声里出现过单字"大"，那个不在卡名表里）。
            if len(re.findall(r"[一-鿿]", t)) < 2 and t.strip() not in self._card_name_set:
                continue  # 滤掉 "0/3"、图标、单字残片
            # 滤掉卡面**描述文本**：卡名都很短且不含标点/关键词标记，
            # 而描述长且带"："【】等（实测读到过"整备：对距离最近的【河洛】"）
            if len(t) > 7 or re.search(r"[：:【】，。]", t):
                continue
            name = self.resolve_name(t, keep_raw=False, pool=self._hand_names)   # 兜底整条 OCR 不收乱码名（识别审查 2026-10-11：「空默给馨」）
            if name:
                cx = box[0] + box[2] // 2
                # 同一张卡可能被相邻窗口各读到一次（实测"天籁琴音"在 797 和 855 各一次）。
                # 按 x 去重挡不住（相距 58 > 45），所以再按**卡名**去一次重。
                if any(sl["name"] == name and abs(sl["x"] - cx) < HAND_NAME_DEDUP
                       for sl in state.hand_slots):
                    continue
                cy = box[1] + box[3] // 2 + HAND_GRIP_DY
                state.hand.append(name)
                slot = {"name": name, "x": cx, "y": cy,
                        "name_y": box[1] + box[3] // 2}
                if self._card_cats.get(name) == "棋手技能牌":
                    slot["ready"] = _maple_ready(image, cx, cy)
                state.hand_slots.append(slot)

        # **手牌是等间距排列的**，用这个几何约束剔除"挤在一起的伪卡"。
        # 长卡名会被 OCR 断成两段，后半段有时**刚好misread 成另一张真卡**
        # （实测「翩跹舞步·新生」的后半被读成「有叶新生」），
        # 这种伪卡一定紧挨着真卡，间距远小于正常卡距。
        # 注意：**宁可多认也不能少认** —— 少认一张的代价是刚买的主C凭空消失，
        # 多认一张只是白试一次拖拽（会被本回合搁置机制兜住）。所以判据放得保守。
        if len(state.hand_slots) >= 3:
            xs = [sl["x"] for sl in state.hand_slots]
            gaps = sorted(b - a for a, b in zip(xs, xs[1:]))
            median = gaps[len(gaps) // 2]
            min_gap = min(median * HAND_GAP_MIN_RATIO, HAND_PSEUDO_MAX_GAP)
            kept = [state.hand_slots[0]]
            for sl in state.hand_slots[1:]:
                if sl["x"] - kept[-1]["x"] < min_gap:
                    # 挤在一起：留名字更长的那个（更可能是完整卡名）
                    if len(sl["name"]) > len(kept[-1]["name"]):
                        kept[-1] = sl
                    continue
                kept.append(sl)
            if len(kept) != len(state.hand_slots):
                logger.info("手牌间距过滤：%s -> %s",
                            [sl["name"] for sl in state.hand_slots],
                            [sl["name"] for sl in kept])
                state.hand_slots = kept
                state.hand = [sl["name"] for sl in kept]
        now = time.time()
        for sl in state.hand_slots:
            self._hand_seen[sl["name"]] = now

    def _extract_players(self, context, image, state: GameState) -> None:
        # 血量只在一轮战斗结算后变化，同样用像素比对跳过（省 6 次 OCR 约 370ms）
        if not self._region_changed("players", image, PLAYER_HP_CELLS) and self._players_cache:
            state.players = [PlayerRow(**vars(p)) for p in self._players_cache]
            # **走缓存也必须把 state.hp 填上**，否则它会回落到 GameState 的默认值，
            # 看起来像"满血"——决策层据此以为很安全。
            # **不要在这里"复核"挂起的读数**：像素没变 = 还是那批像素，
            # 再确认一次只是把同一个误读抄了一遍（2026-09-23 血 57→0 就是这么来的）。
            state.hp = self._last_hp
            return

        # **每帧重新认自己那一行**（六人列表会按排名重排）。认法见 own_hp_row：
        # 我方的**血量数字和昵称都是绿的**；战斗中的紧凑排版没有昵称，只能看绿数字。
        # **认不出来就本帧不更新血量**（2026-10-22）：原来退回"上一次认出的行号"，
        # 而战斗中列表没昵称、旧判据几乎每帧都认不出，列表一重排就一直读别人那一行
        # （927 局 R15 读成 9、实际 26，把"血量 ≤8 才用"的绝处逢生打了出去）。
        row, own_roi, scores, by_digits = own_hp_row(image)
        if row is not None and not by_digits:
            # 只靠绿昵称认出：卡牌大图 / 英雄详情面板盖住列表时昵称带里也会有"像字"的绿色，
            # 必须 OCR 核对昵称（data/player_profile.json；没配就不认）
            nx, ndy, nw, nh = ROI_PLAYER_NAME
            nick = own_nickname()
            name_txt = _ocr(context, image, (nx, PLAYER_ROWS_Y[row] + ndy, nw, nh)) if nick else ""
            if not nickname_matches(name_txt, nick):
                logger.debug("第 %d 行绿昵称核对不上（读到 %r），不认", row, name_txt)
                row, own_roi = None, None
        # 紧凑排版数字整体左移到 x≈100：别人的血量也换成从 92 起的框（旧框会切掉首位数字）
        compact = by_digits and own_roi is not None and own_roi[0] + 4 < 120
        if compact:
            rois = [(HP_COMPACT_X, y, HP_COMPACT_W, ROI_PLAYER_HP[3]) for y in PLAYER_ROWS_Y]
        else:
            rois = [(ROI_PLAYER_HP[0], y, ROI_PLAYER_HP[2], ROI_PLAYER_HP[3])
                    for y in PLAYER_ROWS_Y]
        if row is not None:
            rois[row] = own_roi   # 自己那行只框绿数字，避开头像边和后面的图标
        texts = _map_parallel(                      # 6 行血量并发读
            lambda r: _ocr(context, image, r, expected=r"\d{1,3}", only_digits=True), rois)
        state.players = [PlayerRow(index=i, hp=_player_hp(t))
                         for i, t in enumerate(texts)]
        if row is None:
            if not self._own_row_miss:
                logger.info("我方血量行本帧认不出（各行绿像素 %s），血量不更新、保持 %d",
                            scores, self._last_hp)
            self._own_row_miss = True
        else:
            self._own_row_miss = False
            if self._own_row is not None and row != self._own_row:
                logger.info("六人列表重排了：我方从第 %d 行挪到第 %d 行",
                            self._own_row, row)
            self._own_row = row
        mine = row
        if mine is not None and 0 <= mine < len(state.players):
            state.players[mine].name = "我"
            hp = state.players[mine].hp
            # 回合号和血量是**同一批并发读的**，读血量时 state.round_no 可能还没填上（=0），
            # 原来那样就算成"隔了 99 回合"、掉血上限整个失效（2026-09-23：51 -> 3 被采信）。
            # 这时用上一帧读到的回合号。
            rn = state.round_no or self._last_round
            elapsed = (rn - self._hp_round if rn > 0 and self._hp_round > 0 else 99)
            # 被拒的那个值**持续了多久**（不只是连读了几次，见 HP_HOLD_S）
            now = time.time()
            if hp != self._hp_reject[0] or not self._hp_reject_since:
                self._hp_reject_since = now
            kept, self._hp_reject = filter_hp(self._last_hp, hp, self._hp_reject,
                                              first=not self._hp_seen,
                                              round_no=rn,
                                              rounds_elapsed=elapsed,
                                              before=self._hp_before,
                                              held_s=now - self._hp_reject_since)
            if self._hp_reject == (0, 0):
                self._hp_reject_since = 0.0
            if kept == hp:
                self._hp_seen = True
            if kept != hp:
                logger.info("血量读数 %d 不可信（上次 %d），沿用旧值", hp, self._last_hp)
            if kept == hp and kept > 0:
                # 往回纠正的上限 = **本局认定过的最高血量**。原来用"最近一次掉血前的值"，
                # 可那个值本身也可能是误读偏低的（实战：误认 26 -> 又误认 1，真实 33 永远纠正不回来）
                self._hp_before = max(self._hp_before if self._hp_seen else 0, kept)
            if kept == hp and rn > 0:
                # **读数和认定值一致 = 又确认了一次**，重新计时。原来只在"血量变了"时记回合，
                # 血量稳定几回合后，"隔了几个回合"就被算大、掉血上限跟着放宽 ——
                # 2026-09-23 实战：40 从第 6 回合稳到第 8 回合，第 8 回合误读成 3，
                # 算成"隔了 2 回合、上限 50"，连读 3 次就认了（实际血量 24）。
                self._hp_round = rn
            self._last_hp = kept
            state.hp = kept
        else:
            state.hp = self._last_hp
        self._players_cache = [PlayerRow(**vars(p)) for p in state.players]

    def _finalize_level_cost(self, state: GameState) -> None:
        """**升级费 = 所需经验 − 当前经验**（经验条 "N/M" 读到时以它为准）。

        2026-09-24 视觉验证局第 6 回合：刚从 2 级升到 3 级，经验条 1/17（还差 16），
        升级按钮上的费用读不出来，于是沿用了 2 级时粘住的旧值 7 —— 以为 10 能量够升级，连点两次都没反应，
        开店预算用完，整回合空转。升级费本来就是"还差多少经验"（1 能量 = 1 经验），经验条每帧都读得到，比按钮上的小数字可靠。
        放在所有并行任务之后，不受先后顺序影响。
        """
        if 0 < state.xp_needed and 0 <= state.xp < state.xp_needed:
            cost = state.xp_needed - state.xp
            if state.level_cost != cost:
                if state.level_cost and state.level_cost > 0:
                    logger.info("升级费按经验条改为 %d（%d/%d），按钮读数/旧值是 %s", cost, state.xp, state.xp_needed, state.level_cost)
                state.level_cost = cost
            self._last_level_cost = cost

    def _extract_lord_level(self, image, state: GameState, context=None) -> None:
        """棋手等级（点数徽章）。读不到沿用上一帧；**只增不减**，读到更小的值当误读。

        等级决定商店能刷出哪些品阶（见 agent/decision/meta.py 的 MAX_TIER_BY_LEVEL），
        是"该升级还是该 D 牌"的唯一判据，所以宁可沿用旧值也不要给错值。
        """
        # **先读经验条 "N/M"**：M 是这一级升级所需经验，官方表 7/13/17/23/27 一一对应 1~5 级，
        # 满级显示「满级」。比点数徽章可靠得多（2026-09-23 第十二局：画面 0/23 = 4 级，徽章读成 3 级）。
        # 读到就以它为准（也能纠正徽章读偏大的旧值）；读不到才退回徽章。
        txt = _ocr(context, image, ROI_LORD_XP)
        m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})", txt or "")
        xp_level = 6 if "满级" in (txt or "") else (LORD_XP_NEEDED.get(int(m.group(2))) if m else None)
        if m and xp_level and xp_level < 6 and int(m.group(1)) < int(m.group(2)):
            state.xp, state.xp_needed = int(m.group(1)), int(m.group(2))   # 收尾时据此算升级费（见 _finalize_level_cost）
        if xp_level:
            state.level = xp_level
            self._last_lord_level = xp_level
            return
        level = read_lord_level(image)
        # **人口上限能反推等级下限**（官方表 5/5/6/6/7/7）。徽章在高等级读不准
        # （6 点会连成一片、有时还是彩色的），所以两边取大：
        # 徽章读得到就用徽章，读不到/偏低就用人口上限兜底。
        best = max(level if level > 0 else 0, self._last_lord_level)
        state.level = best
        self._last_lord_level = best

    def pieces_quality(self, name: str):
        """卡名 -> 品阶（给"剩余数量"读数的分母校验用）。"""
        if not hasattr(self, "_quality_map"):
            try:
                import json as _json
                cards = _json.loads((Path(__file__).resolve().parents[2] / "data" / "cards.json")
                                    .read_text(encoding="utf-8"))["cards"]
                self._quality_map = {k: v.get("quality") for k, v in cards.items()}
            except Exception:                               # noqa: BLE001
                self._quality_map = {}
        return self._quality_map.get(name)

    pop_bonus = 0     # 天赋/效果牌额外给的人口上限，主循环每帧从 rules.pop_bonus 同步
    hand_book_hint: set = set()   # 手牌账里有、上一帧画面没认出来的牌（主循环每帧从 rules 同步；_hand_name_closed 闭集核对用）
    pop_bonus_seen = 0            # 从画面读数认出来的额外人口（记账漏了时的兜底），主循环同步回规则引擎
    pop_bonus_revoked = False     # 白色满员证明画面认的额外人口是误读 → 主循环把规则引擎的人口加成退回记账值
    _pop_extra_votes = (0, 0)     # (读到的超限上限值, 连续帧数)

    def _valid_caps(self) -> tuple:
        return tuple(c + self.pop_bonus for c in VALID_POP_CAPS)

    def _extract_population(self, context, image, state: GameState) -> None:
        self._extract_population_ocr(context, image, state)
        # **颜色比认字稳**（2026-09-24 第一轮第 3 局）：满员时数字变白。那一局旁边站着的单位头顶的
        # 等级徽章「25」贴着数字左边，窄 ROI 把「6」的左半边切掉、读成 5 —— 一直以为还有空位，
        # 反复往场上拖英雄。白色就等于满员，不管读出来是几。
        if state.shop_open or state.phase == "battle":
            return
        col = pop_color(image)
        valid = self._valid_caps()
        cap = state.population_cap if state.population_cap in valid else self._last_pop_cap
        # 2026-10-13 复盘：白色判据会被旁边的白色徽章 / 特效触发，
        # 4/6、5/6 被一次改成 6/6（截图里其实是绿色）→ 换人网络在"满员"题面下卖人。现在只在读数恰好差 1 时修正
        # 2026-10-15 第 644 局：上限被误读成 8（7 级上限 7）、连读 3 帧被当成"额外人口 +1"，之后数字已经变白（满员），
        # 却按 7/8 一直往场上拖最低配套件少司缘、不腾位。白色 + 读数恰好等于按等级的上限 + 额外人口只来自画面 → 撤销画面认的额外人口
        if col == "white" and state.population_valid and self.pop_bonus_seen > 0 and                 state.population in VALID_POP_CAPS and state.population == (state.population_cap or 0) - self.pop_bonus_seen:
            self._pop_white_votes = getattr(self, "_pop_white_votes", 0) + 1
            if self._pop_white_votes >= 3:
                logger.info("人口数字连续 3 帧是白色（满员）、读数 %d/%d：画面认的额外人口 +%d 是误读，撤销",
                            state.population, state.population_cap, self.pop_bonus_seen)
                self.pop_bonus_seen, self.pop_bonus_revoked, self._pop_white_votes = 0, True, 0
                self._pop_extra_votes = (0, 0)
                state.population_cap = state.population
                self._last_population = self._last_pop_cap = state.population
                return
        else:
            self._pop_white_votes = 0
        if (col == "white" and cap in valid and state.population_valid and (state.population or 0) < cap) or (
                col == "white" and cap in valid and not state.population_valid and (self._last_population or 0) >= cap - 1):
            # 2026-10-19（第 738 局 R7~R16：蒙恬站在挡人口格，徽章盖住第一个数字，8/8 读成 3/8、7/7 读成 1/7 共 77 题）：
            # 原来只修"恰好差 1"；差 >1 时必须有视觉英雄数佐证（_apply_pop_white 里 strict：没视觉结果不按颜色改）
            state._pop_white_strict = state.population_valid and (state.population or 0) < cap - 1
            # 2026-10-16 第 667 局：5/6 被白色判据改成 6/6 共 7 次（召唤物 / 白色特效站在地上的人口数字旁）→ 换人卖掉云中君、开战缺人。
            # 颜色只记"疑似满员"，等同一帧的视觉单位检测跑完再核对（_apply_pop_white）：认出的己方英雄数达到上限才改。
            # 第 655~667 局 3831 帧：视觉英雄数 = 人口读数 93.2%，多认 0.9% → 视觉数等于读数时信读数。
            state._pop_white_cap = cap

    def _apply_pop_white(self, state: GameState) -> None:
        """白色"疑似满员"的核对（_extract_population 只记 state._pop_white_cap）：视觉检测认出的己方英雄（不含召唤物）达到上限才改成满员；
        这一帧没有视觉检测结果时照旧按颜色改（9-24 第 3 局：徽章挡住数字、6 读成 5 的情况）。"""
        cap = getattr(state, "_pop_white_cap", None)
        if not cap:
            return
        try:
            del state._pop_white_cap
        except AttributeError:
            pass
        vu = state.vision_units or {}
        heroes = sum(1 for v in vu.values() if v == "hero")
        strict = getattr(state, "_pop_white_strict", False)
        if strict and not vu:
            logger.info("人口数字发白、读数 %s/%s 差上限 >1，这一帧没有视觉结果：不按颜色改", state.population, state.population_cap)
            return
        if vu and heroes < cap:
            logger.info("人口数字发白，但视觉只认出 %d 个己方英雄（< 上限 %d）：不按颜色改，保留读数 %s/%s",
                        heroes, cap, state.population, state.population_cap)
            return
        logger.info("人口数字是白色（满员）：读数 %s/%s 改为 %d/%d（视觉英雄 %s）", state.population, state.population_cap, cap, cap,
                    heroes if vu else "无")
        state.population, state.population_cap, state.population_valid = cap, cap, True
        self._last_population, self._last_pop_cap = cap, cap

    def _extract_population_ocr(self, context, image, state: GameState) -> None:
        """人口数 "5/5"：纯英雄上场数/上限（召唤物、图腾不占）。
        读数失败时沿用上一帧（该 UI 元素较小，OCR 偶发失败）。"""
        # **商店开着时这块区域被商店卡面盖住**，读出来的是卡面上的数字（实测把
        # "古币多多"卡读成人口 0/6），假数据会让对账系统反过来删掉正确的账本记录，
        # 导致英雄被反复重上。宁可读不到，也不要读错。
        # **战斗画面的人口也不能用**：阵亡的单位不算人口，打到后半场读出来是 1/7
        # （2026-09-23 第八局）。这个数被当成"上次读到的人口"，下回合 sync_field 看到
        # "人口 1 < 账本 7"，把主C 蒙犽判成"觉醒回手"删掉 —— 之后想多上一个人、
        # 效果牌全往错的格子打，整回合死循环（人口判断又错了）。
        # 注意别写成 `phase != "prepare"`：拖拽回执用新建的 GameState（phase=unknown）当探针读人口，
        # 那样会让"人口 +1 = 上场成功"整条判据失效（2026-09-23 第十一局实测：上场成功被判失败）
        if state.shop_open or state.phase == "battle":
            state.population = self._last_population
            state.population_cap = self._last_pop_cap
            state.population_valid = False
            return

        # 先试紧 ROI（只有数字，最稳），再退回宽 ROI
        text = "".join(_ocr_multi(context, image, ROI_POPULATION_TIGHT))
        digits = re.sub(r"\D", "", text)
        # 显示是 "N|M人"：OCR 把分隔符 "|" 读成 1、人形图标读成 L 或 1。
        # 实测样本 → 真实值：
        #   "15"→1/5   "115"→1/5   "1151"→1/5   "015L"→0/5   "0151"→0/5
        # 所以 **N=第1位数字，M=第3位**（第2位是被误读的分隔符）；
        # 只有两位时说明分隔符被丢掉，取第1、2位。
        # 曾用"末两位"规则，把 0/5 读成 1/5、1151 读成 5/1（2026-09-22 实战踩坑）。
        if len(digits) >= 2:
            n = int(digits[0])
            cap = int(digits[2]) if len(digits) >= 3 else int(digits[1])
            # 人口上限只会因升级而增加，**不会变小**。读到更小的值必是误读
            # （实测出现过 1/1，而当时上限是 5）。
            valid = self._valid_caps()
            # **画面比记账可信**（2026-09-24：用了特殊牌/天赋之后重新看一次，关掉商店就能读到）：
            # 读到比已知合法上限多 1~2 的值（天赋/效果牌加的人口，记账漏了 —— 比如重启前选的天赋），
            # 同一个值连续 3 帧 → 认下来，记进 pop_bonus_seen，主循环同步给规则引擎。
            # 单帧误读（徽章压字、'9'）过不了"连续 3 帧同值 + 最多高 2"这两道。
            if max(valid) < cap <= max(valid) + 2 and n <= cap:
                v, k = self._pop_extra_votes
                self._pop_extra_votes = (cap, k + 1 if v == cap else 1)
                if self._pop_extra_votes[1] >= 3:
                    self.pop_bonus_seen = cap - max(VALID_POP_CAPS)
                    self.pop_bonus = max(self.pop_bonus, self.pop_bonus_seen)
                    logger.info("人口上限读到 %d（超出按等级的上限），连续 3 帧一致 → 认定额外人口 +%d",
                                cap, self.pop_bonus_seen)
                    valid = self._valid_caps()
            else:
                self._pop_extra_votes = (0, 0)
            if cap < self._last_pop_cap or cap not in valid:
                cap = -1
            if cap in valid and n <= cap:
                state.population, state.population_cap = n, cap
                state.population_valid = True
                self._last_population, self._last_pop_cap = n, cap
                return
        text = "".join(_ocr_multi(context, image, ROI_POPULATION))
        m = re.search(r"(\d+)\s*[/|]\s*(\d+)", text)
        n2, cap2 = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        if m and cap2 in self._valid_caps() and n2 <= cap2:
            state.population, state.population_cap = n2, cap2
            state.population_valid = True
            self._last_population, self._last_pop_cap = n2, cap2
        else:
            # **读不到就沿用旧值，但绝不声称有效**：population_valid=False 时
            # 决策层会退回保守模式（每回合最多上 1 个），比拿着陈旧上限乱放人强。
            state.population, state.population_cap = self._last_population, self._last_pop_cap
            state.population_valid = False

    def _extract_auction(self, context, image, state: GameState) -> None:
        auc = AuctionState()
        # 预览期顶部写"预览"，此时拍品还在飞入、出价按钮没出来。
        # 2026-09-22 确认：**先点中卡牌，等拍卖正式开始出价按钮才出现**。
        # 预览还是可出价：**看右下角能量圆盘在不在**（见 ROI_AUCTION_DIAL 注释）
        auc.preview = not _has_energy_dial(image)
        title = _ocr(context, image, ROI_AUCTION_TITLE) or ""
        auc.won = "已获得" in title
        # 「等待拍卖中」：拍卖还没开，画面是自家棋盘 + 右侧天赋列（2026-09-26 第四轮第 4 局
        # 证据帧 live_1790350830/auction_1790352022_after_select.png）。真拍卖的第 6 件拍品在 x≈1147，
        # 不能靠"x 不过天赋列"过滤 —— 这里直接不读拍品，否则天赋列（战略中枢、翩跹舞步）会被当成拍品去点。
        if "等待" in title:
            state.auction = auc
            return
        # 新版顶部倒计时优先，旧版右下兜底
        auc.countdown = _to_int(_ocr(context, image, ROI_AUCTION_COUNTDOWN_TOP,
                                     only_digits=True))
        if auc.countdown < 0:
            auc.countdown = _to_int(_ocr(context, image, ROI_AUCTION_COUNTDOWN,
                                         only_digits=True))
        # 出价阶段右下角就是能量圆盘，直接读（3 位数是误读，重读紧 ROI）
        # 同准备阶段：白字读法优先（图标会被认成数字，见 _read_white_digits）
        gold = _read_white_digits(context, image, ROI_GOLD_WHITE)
        if gold < 0:
            gold = _to_int(_ocr(context, image, ROI_GOLD, expected=r"\d{1,4}"), -1)
            if gold > 99:
                gold = _to_int(_ocr(context, image, ROI_GOLD_TIGHT, only_digits=True))
        auc.my_gold = gold if 0 <= gold <= 99 else -1
        # 拍卖阶段 `_extract_economy` 不跑（那些按钮都不在），但决策层和日志都看 state.gold。
        # 不同步过去的话整场拍卖都显示"能量0"，看日志时会误以为没钱所以不出价
        # （2026-09-22 实战：圆盘明明写着 14，日志一路 能量0）。
        if auc.my_gold >= 0:
            state.gold = auc.my_gold
        # 6 个拍品 × 3 次 OCR（卡名/出价区/已拍出）= 18 次往返，是拍卖帧的全部开销。
        # 拍卖是**抢时间**的场景（倒计时几秒，慢一拍就被别人拍走），必须并发。
        # **拍品位置按实测定，不要用写死的 6 等分网格**（2026-09-22 发现）。
        # 拍品被拍走后剩下的会重新排布，固定网格的第 i 格可能正好跨在两张卡**之间的缝**里：
        # 那里既读不到完整卡名，点下去也是空点——整场拍卖都在点缝隙。
        # 做法同手牌：先把整条卡名带一次 OCR 出来，**检测到几个名字就是几个拍品**，
        # 各自的真实 x 由它自己的名字框给出；出价区/已拍出再按这个 x 去读。
        # 用**滑动窄窗口**扫这条卡名带，而不是一条宽 ROI：
        # 宽 ROI 会让小字号在 OCR 检测阶段被丢掉（这个坑以前记过，
        # 实测整条带一次读只认出 6 个拍品里的 4 个）。窗口保持和原来一样窄(200px)，
        # 但**步长只有 100px、互相重叠**，所以不依赖"6 等分"这个假设——
        # 拍品少了、位置移了都能扫到。重叠导致同一个名字被读到多次，按 x 去重。
        win_w, step = 200, 100
        windows = [(x, ROI_AUCTION_NAME_Y[0], win_w, ROI_AUCTION_NAME_Y[1])
                   for x in range(0, 1280 - win_w + step, step)]
        seen: list[tuple[str, list]] = []
        for boxes in _map_parallel(
                lambda r: _ocr_boxes(context, image, r), windows):
            for t, b in boxes:
                if len(re.findall(r"[一-鿿]", t)) < 2 and not self._single_char_hero(t):
                    continue      # 单字英雄（镜 / 瑶 / 曜）原来被这条整个丢掉（2026-10-11 第 566 局：6 件只认出 4 件）
                cx = b[0] + b[2] // 2
                dup = next((i for i, (_t2, b2) in enumerate(seen)
                            if abs((b2[0] + b2[2] // 2) - cx) < 70), None)
                if dup is None:
                    seen.append((t, b))
                elif len(t) > len(seen[dup][0]):
                    seen[dup] = (t, b)      # 同一个名字被窗口切过，留最完整的那次
        lots_found = seen
        lots_found.sort(key=lambda tb: tb[1][0])
        # **先推算这场有几张拍品**（2026-10-11：先看有多少张牌再识别，拍卖出现的牌数取决于剩余玩家数量）：
        # 拍品卡等距（≈AUCTION_PITCH）、整排居中 → 由读到的名字位置推出总张数和缺的那几张的位置，缺的再用窄框单独重读一次
        xs = [b[0] + b[2] // 2 for _t, b in lots_found]
        slots = auction_slots(xs)
        auc.expected = len(slots) if slots else -1
        if slots and len(slots) > len(xs):
            for sx in slots:
                if any(abs(sx - x) < 70 for x in xs):
                    continue
                got = None
                # 单字名在文字检测这一步不稳（同一块区域挪 3px 就检不出「瑶」）→ 多试几个窗口；矮窗口（<40px）整块识别、不走检测
                for roi in ((max(0, sx - 90), ROI_AUCTION_NAME_Y[0], 180, ROI_AUCTION_NAME_Y[1]),
                            (max(0, sx - 50), ROI_AUCTION_NAME_Y[0] + 14, 100, 36),
                            (max(0, sx - 45), ROI_AUCTION_NAME_Y[0] + 22, 90, 30)):
                    for t, b in _ocr_boxes(context, image, roi):
                        if re.findall(r"[一-鿿]", t) and (len(re.findall(r"[一-鿿]", t)) >= 2 or self._single_char_hero(t)):
                            got = (t, b)
                            break
                    if got:
                        break
                if got:
                    lots_found.append(got)
            lots_found.sort(key=lambda tb: tb[1][0])
        if not lots_found:                       # 一个都没读到（预览期卡还在飞入）
            state.auction = auc
            return

        probes: list[tuple[str, tuple[int, int, int, int]]] = []
        for _t, b in lots_found:
            cx = b[0] + b[2] // 2
            ny = b[1] + b[3] // 2
            # 2026-10-11 识别审查：出价区 / 「已拍出」都跟着卡名的 y 走（没人出价的卡不抬起，绝对 y 的出价区会切到卡面底栏
            # "HONOR OF KINGS / NO.45" 被当成领先者；「已拍出」实际在 y≈211，原 ROI 120~180 从来没认出过）
            probes.append(("bids", (max(0, cx - 95), max(ROI_AUCTION_BID_Y[0], ny + 100),
                                    190, ROI_AUCTION_BID_Y[1])))
            probes.append(("sold", (max(0, cx - 80), max(100, ny - 110), 160, 120)))

        def _probe(p):
            kind, roi = p
            if kind == "sold":
                return _hit(context, image, roi, "已拍出")
            return _ocr_boxes(context, image, roi)

        res = _map_parallel(_probe, probes)
        lot_reads = [([lots_found[i]], res[i * 2], res[i * 2 + 1])
                     for i in range(len(lots_found))]
        for i, (name_boxes, bid_boxes, sold_flag) in enumerate(lot_reads):
            # 用 _ocr_boxes 而不是 _ocr：除了文本还要拿到框的真实位置，
            # 点拍品时用实测坐标，UI 版本偏移也不会点空（写死的 AUCTION_LOT_X
            # 是旧版 UI 标的，2026-09-22 实战出价失败就是点空了）。
            name, name_xy = "", (-1, -1)
            for text, box in name_boxes:
                if re.search(r"[一-鿿]", text):
                    name = text
                    name_xy = (box[0] + box[2] // 2, box[1] + box[3] // 2)
                    break
            # 出价区结构：每个出价人 = 名字文本块 + 其正下方 ~18px 的价格数字块
            # **最上面一行是领先者**。2026-09-24 数据局修正（逐秒截图 live_1790246871/auction_r11_seq/f15）：
            #   ① 原来要求名字里有汉字 —— 我们的昵称 RudAns 是英文，**自己那一行整行被丢掉**，
            #      于是永远看不到"是我领先 / 被别人超过"，出价 1 后就不再跟（被别人叫价赶超之后没有反应）；
            #   ② 领先者按"价格最大"取，而下面几行的价格常读不到 → 取到的是随便一行；
            #   ③ 卡面底部的「效果」会落进这块区域，被当成出价人（'效果'、'党效樂'）；
            #   ④ "这一行是我"看名字是不是绿色（mark_own_bids 原来写好了却没人调用）。
            import cv2
            boxes = bid_boxes
            bids: list[tuple[str, int, tuple]] = []
            for text, box in boxes:
                t = text.strip()
                if len(t) < 2 or re.fullmatch(r"\d+", t) or "效" in t:
                    continue
                if re.search(r"HONOR|ONOR|OROF|HOOR|KINGS|INGS|HERO|ROFIN|NO\.?\s?\d{2}", t) and not re.search(r"[一-鿿]", t):
                    continue          # 卡面底栏水印（HERO NO.45 / HONOR OF KINGS ACE）；只认水印的大写词，英文昵称（RudAns / ontcy）照收
                # 卡面底部的**派系名**也会落进来（"日落海·""河洛·""三分之地。"，2026-09-24 match9 实测）：
                # 还没人出价时它会被当成领先者；若某帧没读到我方那行而读到它，会误判"被超"去自抬价格
                if re.sub(r"[·。.、\s]", "", t) in AUCTION_FACTION_WORDS:
                    continue
                bx, by = box[0], box[1]
                price = 0
                for t2, b2 in boxes:
                    if re.fullmatch(r"\d+", t2.strip()) and 0 <= b2[1] - by <= 35 \
                            and abs(b2[0] - bx) <= 20:
                        price = max(price, int(t2.strip()))
                bids.append((t, price, tuple(box)))
            sold = sold_flag
            top = min(bids, key=lambda b: b[2][1]) if bids else ("", 0, ())
            if top[2]:
                x, y, w, h = top[2]
                patch = image[y:y + h, x:x + w]
                if patch.size:
                    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
                    green = int(((hsv[..., 0] > 35) & (hsv[..., 0] < 85)
                                 & (hsv[..., 1] > 90) & (hsv[..., 2] > 90)).sum())
                    if green >= AUCTION_MINE_GREEN:
                        top = ("我", top[1], top[2])
            auc.lots.append(AuctionLot(
                index=i,
                piece=self.resolve_name(name, keep_raw=False) or None,
                top_bid=top[1],
                top_bidder=top[0],
                bidders=len(bids),
                sold=sold,
                x=name_xy[0],
                y=name_xy[1],
            ))
        # 「出价」按钮下的数字 = 对当前选中拍品再出一次的价（2026-09-24 逐秒截图：我们出 2 之前显示 2、之后显示 3）。
        # 别家的出价数字常读不到，这个数是唯一可靠的"现价"。按钮位置用 OCR 找（和 executor 点它的方式一致）。
        if not auc.preview:
            btn = next(((t, b) for t, b in _ocr_boxes(context, image, (1000, 560, 220, 150)) if "出价" in t), None)
            if btn:
                bx, by, bw, bh = btn[1]
                digits = [t.strip() for t, _ in _ocr_boxes(context, image, (bx - 5, by + bh - 4, bw + 10, 34))
                          if re.fullmatch(r"\d{1,2}", t.strip())]
                if digits:
                    auc.next_price = int(digits[0])
        state.auction = auc

    def _extract_lord_boost(self, context, image, state: GameState) -> None:
        """棋手特性三选一。名字是金色描边艺术字，OCR 容易错，靠卡牌库反查修正；
        点击点用**名字框的实测坐标**，不写死三列位置（同拍卖的教训）。"""
        state.talents = []
        names = _ocr_boxes(context, image, ROI_LORD_BOOST_NAMES)
        descs = [txt for txt, _ in _ocr_boxes(context, image, ROI_LORD_BOOST_DESCS)]
        names.sort(key=lambda tb: tb[1][0])          # 从左到右
        for i, (txt, box) in enumerate(names):
            if not re.search(r"[一-鿿]", txt):
                continue
            resolved = self.resolve_name(txt, keep_raw=False)
            if resolved is None and i < len(descs):
                resolved = self.resolve_by_desc(descs[i]) or txt
            state.talents.append(TalentOption(
                index=len(state.talents),
                name=resolved or txt,
                desc=descs[i] if i < len(descs) else "",
                x=box[0] + box[2] // 2,
                y=box[1] + box[3] // 2,
            ))

    def _extract_talents(self, context, image, state: GameState) -> None:
        state.talents = []
        # 底部那行：刷新还剩几次写「全局N次」，用光了写「刷新次数已用完」。
        # **必须从画面读，不能只靠自己记账**——记账会因为等级读数漂移而失效，
        # 而"刷新用光了还去点刷新"的后果是：点了没反应，界面挂着没人选，整局卡死
        # （2026-09-22 第十二局回合8 实测）。OCR 会把"已用完"认成"已用宗"，两种都认。
        marker = "".join(_ocr_multi(context, image, ROI_TALENT_MARKER))
        # 满级时这一行整个没有 —— **读不到就当不能刷**（原来"没写用完 = 能刷"，
        # 满级界面会去点一个不存在的刷新按钮）
        state.talent_can_refresh = (bool(re.search(r"全局|次", marker))
                                    and not ("用完" in marker or "用宗" in marker))
        for i, x0 in enumerate(TALENT_CARD_X):
            name = _ocr(context, image,
                        (x0 + 20, ROI_TALENT_NAME_Y[0], 200, ROI_TALENT_NAME_Y[1]))
            desc = "".join(_ocr_multi(context, image,
                                      (x0 + 15, ROI_TALENT_DESC_Y[0], 215, ROI_TALENT_DESC_Y[1])))
            resolved = self.resolve_name(name, keep_raw=False) or self.resolve_name(
                re.sub(r"[A-Za-z0-9\s]", "", name or ""), keep_raw=False)   # 去掉卡面水印前缀（'TA末世'）
            if resolved is None:
                # stylized 金色字体名称常误识，用描述反查 DB
                resolved = self.resolve_by_desc(desc) or name
            elif len((desc or "").strip()) >= 8 and self._card_descs.get(resolved):
                # **名字和描述对不上 → 以描述为准**（2026-10-21 第 850 局 R5：祝你好运送的"碰碰运气"小卡飞过第 3 张卡的名字，
                # 阿科米亚之子被读成碰碰运气 → 三件特殊装备变成"2 装备 + 1 天赋"，该刷新没刷、选了它穿给守约）
                import difflib as _dl
                # 天赋版（"获得魂祭绝刃：…"）描述里就带着名字 → 名字是对的，只是卡库那条是装备本体的写法，不改（审计 10-21：16 例都是这种）
                if resolved not in desc and _dl.SequenceMatcher(None, desc.strip(), self._card_descs[resolved]).ratio() < 0.35:
                    by_desc = self.resolve_by_desc(desc)
                    if by_desc and by_desc != resolved:
                        logger.warning("三选一第 %d 张：名字读成 %s，但描述是 %s 的 → 按描述改成 %s", i + 1, resolved, by_desc, by_desc)
                        resolved = by_desc
            state.talents.append(TalentOption(index=i, name=resolved, desc=desc))


# ==================== 装备四选一弹窗（纯图标，必须模板匹配） ====================
# 基础装备拖到英雄身上后，会在英雄附近弹出 4 个**纯图标、无文字**的选项
# （2026-09-22 确认）。OCR 在这里永远找不到东西，只能靠图标模板匹配。
# 模板来自百科爬虫：assets/image/pieces/装备牌/<名称>.png，86x90。

def load_equip_recipes(data_dir: Path) -> dict[str, list[str]]:
    """基础装备 -> 四选一的 4 个成品。没有这张表时返回空 dict（退化为全库匹配）。"""
    path = Path(data_dir) / "equip_recipes.json"
    if not path.exists():
        return {}
    import json
    return json.loads(path.read_text(encoding="utf-8")).get("recipes", {})


UTIL_CTRL_RE = re.compile(r"眩晕|晕眩|击飞|冰冻|冻结|沉默|嘲讽|禁锢|定身")


def pick_equip_index(data_dir: Path, base_equip: str,
                     role: str) -> Optional[tuple[int, str]]:
    """装备四选一**选第几个**：(recipes 下标, 成品名)。查不到这件基础装备返回 None。

    偏好来自 equip_recipes.json 的 `prefer`（人工按描述排的，理由写在文件里）。
    没有这件的偏好时退回 recipes[0] 并在日志里说明 —— 不假装自己比较过。
    """
    import json
    path = Path(data_dir) / "equip_recipes.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    recipes = data.get("recipes", {}).get(base_equip)
    if not recipes:
        return None
    if role.startswith("share:"):
        # 瑶的首件普通装备会分享给 3 名友军：破碎圣杯 → 破茧之衣（开战眩晕敌方高等级英雄，4 个人各放一次）
        if "破茧之衣" in recipes:
            return recipes.index("破茧之衣"), "破茧之衣"
        role = "dps"
    if role.startswith("util:"):
        # **主C 满装后的剩余装备：挑辅助性质的**（2026-10-10：多余装备选辅助性质的，比如开局晕对手的那种）。
        # 读卡面：开战 / 开团就生效的控制（破茧之衣"战斗开始…眩晕敌方等级最高"）> 其他控制（御灵之纱、极寒风暴）；都没有就按穿的人挑。
        import json as _j
        try:
            _cards = _j.loads((Path(data_dir) / "cards.json").read_text(encoding="utf-8")).get("cards", {})
        except (OSError, ValueError):
            _cards = {}
        def _u(n):
            sk = (_cards.get(n) or {}).get("skill", "") or ""
            if not UTIL_CTRL_RE.search(sk):
                return 0
            return 2 if re.search(r"战斗开始|开团|开战", sk) else 1
        best = max(recipes, key=lambda n: (_u(n), -recipes.index(n)))
        if _u(best):
            return recipes.index(best), best
        _h, _, _base = role[5:].partition("|")          # util:<英雄>|<carry 或 tank>：没有控制装就按他本来的需要挑
        role = ("carry:" + _h) if _base == "carry" else "tank"
    if role.startswith("carry:"):
        # **给输出位：按这个英雄的伤害类型挑**（2026-09-25：要结合这个英雄是物理伤害还是法术伤害来定；
        # 原来只有 tank/dps 两张表，主C 孙悟空被当成坦克铸出了御灵之纱/反伤重甲）
        from agent.decision.hero_traits import carry_order
        name = carry_order(recipes, role[6:])[0]
        return recipes.index(name), name
    order = (data.get("prefer", {}).get(base_equip, {}).get(role)
             or data.get("prefer", {}).get(base_equip, {}).get("dps"))
    for name in order or []:
        if name in recipes:
            return recipes.index(name), name
    logger.warning("装备四选一：%s 没有 %s 的偏好表，退回第 0 个 %s",
                   base_equip, role, recipes[0])
    return 0, recipes[0]


def _text_overlap(a: str, b: str) -> float:
    """两段中文的 2-gram 重合度（0~1）。用来把弹窗里的效果描述对上装备技能文本。"""
    ga = {a[i:i + 2] for i in range(len(a) - 1)}
    gb = {b[i:i + 2] for i in range(len(b) - 1)}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


_COV_STOP = set("的与和了个时每点获得状态持续唯一")


def _text_coverage(short: str, long: str) -> float:
    """短文本被长文本覆盖的比例。**用它而不是对称重合度**来认装备四选一。

    弹窗上写的是**效果概括**（"根据承伤提升伤害"），卡库里是**完整描述**
    （"每承受3%最大生命值的伤害，获得1%伤害加成，上限100%"）。
    两边长度差一个数量级，对称重合度会被长文本稀释：实测 4 个选项里
    只有 2 个能排到第一，另外两个 0.0~0.14 混在一起分不出来。
    换成"短文本的字有多少出现在长文本里"，4/4 全部正确且差距明显：
    名刀·司命 0.75(次 0.12) / 红魔 0.43(0.29) / 冥王 0.36(0.09) / 魔律之刃 0.88(0.0)。
    """
    chars = set(short) - _COV_STOP
    if not chars:
        return 0.0
    return len(chars & set(long)) / len(chars)


# ==================== 装备四选一：按**位置**选，不读文字 ====================
# 2026-09-23 定的做法：
#   不读文本去识别，而是出现四个图标的小弹窗后，直接点击其中一个。
#    因为每次顺序都是一样的，都是那四个、都是固定的位置，
#    可以记住每个基础装备出现的 4 个可选项的位置（应该跟爬虫爬出来的顺序一样）。
#
# 实测印证（fixture `data/logs/fixtures/equip_popup_4options.png`，目标格 (499,500)，
# 基础装备 **力量腰带**）：4 个图标排在单位头顶的一段圆弧上，
# **沿弧从左到右恰好就是 recipes 的次序**：
#   (385,458) 下左「高生命值获得伤害减免」       = 霸者重铠   recipes[0]
#   (433,370) 上左「增加生命值与提高生命恢复效果」 = 不死鸟之眼 recipes[1]
#   (539,370) 上右「提供法防与法术护盾」         = 魔女斗篷   recipes[2]
#   (589,458) 下右「造成法术伤害并减疗」         = 红莲炎甲   recipes[3]
# 对应关系是拿弹窗上的效果概括去比 cards.json 的 skill 文本核出来的（每条都一对一）。
# 于是规则很简单：**按 x 从小到大排，第 i 个就是 recipes[i]**。
# **而 recipes 的次序 = 卡牌 id 从小到大**（即『跟爬虫顺序一样』）。4 件实测全部成立；
#   原来那张手录表有 3 件次序不对（风暴巨剑错开两位、大棒末两位反了、速击之枪），
#   实战里点错过两次。现已全部按 id 重排，test_perception_offline 守着这条不变量。
# ⚠ 我一开始按"左列上下、右列上下"去猜，把 0 和 1 弄反了 —— 核对描述文本才发现。
#   目前只在**力量腰带这一个样本**上核对过。每次选完都会把"第几个/哪一件/点在哪"写进日志，
#   以后拿到别的基础装备的弹窗帧，要再按同样的办法对一次。
EQUIP_SLOT_NAMES = ("下左", "上左", "上右", "下右")   # 仅用于日志，次序 = recipes 次序

EQUIP_OPT_MIN_V = 70        # 选项图标是**彩色亮圆**；等级徽章是深色（实测 val≈35~44）
EQUIP_OPT_RADIUS = (17, 28)
EQUIP_OPT_DIST = (60, 230)  # 离目标格的距离：太近是英雄身上的标记，太远是别的 UI
EQUIP_OPT_MIN_COUNT = 3     # 认出几个才算"弹窗真的在"（真弹窗 4 个，留 1 个漏检余量）
# 判据的反例（都在 test_perception_offline 里守着）：
#   英雄被拿起时的"四向箭头 + 白色圆弧"（equip_popup_1790132996.png）-> 0 个
#   拖动中的单位（equip_popup_1790133178.png）-> 1 个
#   普通准备帧（phase_prepare_1790133042.png）-> 0 个
# 这三张以前都被旧判据当成了弹窗 —— 旧判据数的是"随便什么圆"，
# 检出的其实是等级徽章（以前记过：「圆 10 个、识别出 1~3 个选项块」）。


def find_equip_slots(image: np.ndarray,
                     center: tuple[int, int]) -> list[Optional[tuple[int, int]]]:
    """装备四选一：返回 **4 个选项的点击点**，下标 = recipes 里的次序。弹窗不在时全是 None。

    **一个字都不读** —— 只找"环绕目标英雄的彩色亮圆"，按 x 从左到右排。
    弹窗上写的是效果概括、不是装备名，旧的 OCR 匹配一直在 3/4 和 4/4 之间摇摆；
    而**选项次序是固定的**（2026-09-23 确认），按位置点就不存在"认错"这回事。
    """
    import cv2

    none: list[Optional[tuple[int, int]]] = [None, None, None, None]
    cx, cy = center
    h_img, w_img = image.shape[:2]
    x0, y0 = max(0, cx - 210), max(0, cy - 200)
    x1, y1 = min(w_img, cx + 210), min(h_img, cy + 60)
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        return none
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    rmin, rmax = EQUIP_OPT_RADIUS
    dmin, dmax = EQUIP_OPT_DIST
    found: list[tuple[int, int]] = []
    for x, y, r in _hough(gray, 24, rmin, rmax, 40):
        if _mean_val(hsv, x, y, r) < EQUIP_OPT_MIN_V:
            continue                      # 深色的是等级徽章，不是选项
        ax, ay = x0 + x, y0 + y
        if not dmin ** 2 <= (ax - cx) ** 2 + (ay - cy) ** 2 <= dmax ** 2:
            continue
        found.append((ax, ay))
    if len(found) < EQUIP_OPT_MIN_COUNT:
        return none
    slots = _fit_equip_ring(found, cx)
    if slots is None:
        logger.info("装备四选一：目标 %s 周围 %d 个亮圆 %s 拼不出弹窗的形状，判定不在",
                    center, len(found), found)
        return none
    return slots


# 弹窗的形状（两个样本、两件不同的基础装备实测，几乎一模一样）：
#                    力量腰带        雷鸣刃
#   上排半宽           53              54
#   下排半宽          102             103
#   上下行距           88              88
#   两排对称轴之差      1               0
# 所以不去"数有几个圆"——那会被旁边随便一个亮色 UI 搅乱
# （雷鸣刃那帧上方 110px 就多了一个，旧写法"多于 4 个就放弃"整个弹窗都不点了）——
# 而是**找符合这个形状的那 4 个**，多出来的杂圆自然落选。
EQUIP_RING_TOP_HW = 53
EQUIP_RING_BOTTOM_HW = 102
EQUIP_RING_ROW_DY = 88
EQUIP_RING_TOL = 14          # 以上三个量的容差（px）
EQUIP_RING_AXIS_TOL = 60     # 对称轴离目标格中心 x 的上限（单位站在格子里会偏一点）


def _fit_equip_ring(pts: list[tuple[int, int]],
                    cx: int) -> Optional[list[tuple[int, int]]]:
    """从一堆亮圆里挑出"上窄下宽、左右对称"的 4 个，按 x 从左到右返回；拼不出返回 None。

    只凑出 3 个时（漏认一个）按对称补第 4 个：对称轴取完整那一排的中点，
    漏掉的 = 同排另一个的镜像。试过"均值当弧心 + 固定半径/角度"，偏 17~36px 会点空；
    镜像法偏 ≤2px。
    """
    tol = EQUIP_RING_TOL
    pairs = []                                    # (左, 右, 对称轴, 半宽, 行 y)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a, b = sorted((pts[i], pts[j]))
            if abs(a[1] - b[1]) > 10:
                continue
            axis, hw = (a[0] + b[0]) / 2, (b[0] - a[0]) / 2
            if abs(axis - cx) > EQUIP_RING_AXIS_TOL:
                continue
            pairs.append((a, b, axis, hw, (a[1] + b[1]) / 2))
    tops = [p for p in pairs if abs(p[3] - EQUIP_RING_TOP_HW) <= tol]
    bottoms = [p for p in pairs if abs(p[3] - EQUIP_RING_BOTTOM_HW) <= tol]

    best, best_err = None, None
    for t in tops:
        for bt in bottoms:
            err = abs(t[2] - bt[2]) + abs((bt[4] - t[4]) - EQUIP_RING_ROW_DY)
            if err > 2 * tol:
                continue
            if best_err is None or err < best_err:
                best, best_err = [bt[0], t[0], t[1], bt[1]], err
    if best is not None:
        return best

    # 只凑出一排 + 另一排的一个：按对称补
    for row, other_hw, sign in ([(t, EQUIP_RING_BOTTOM_HW, +1) for t in tops]
                                + [(bt, EQUIP_RING_TOP_HW, -1) for bt in bottoms]):
        left, right, axis, _hw, ry = row
        for q in pts:
            if q in (left, right):
                continue
            if abs((q[1] - ry) - sign * EQUIP_RING_ROW_DY) > tol:
                continue
            if abs(abs(q[0] - axis) - other_hw) > tol:
                continue
            mirrored = (int(round(2 * axis - q[0])), q[1])
            logger.info("装备四选一：只认出 3 个图标，按左右对称补出第 4 个 %s", mirrored)
            return sorted([left, right, q, mirrored], key=lambda p: p[0])
    return None


# 游戏自己的**推荐标记 👍**：某个选项图标右上角 (+24,-26) 的小圆，里面一个大拇指。
# 它知道英雄是物理还是法术（我们的偏好表只分 tank/dps）：实测给法师干将莫邪推荐的是
# 纯净苍穹（法强）、虚无法杖（法穿），而偏好表选的是暴击/攻速装。所以 👍 在就听游戏的。
# **用模板匹配，不用颜色**：同一个 👍 一次是鲜艳浅蓝（饱和度 ≥90）、一次几乎是灰白（22），
# 颜色阈值漏掉了第二次。它是固定的界面贴图（不随等级变），正适合模板
# （UI 贴图用模板，会随等级变的立绘不能用）。模板 assets/image/ui/equip_thumb.png。
# 5 张弹窗 20 个位置实测：有 👍 的 1.00 / 0.81 / 0.75，没有的全部 ≤0.44。
# （标注时我漏看了大棒那张的 👍，是模板先把它找出来的。）
EQUIP_THUMB_DXY = (24, -26)
EQUIP_THUMB_MIN = 0.6          # 最高分下限
EQUIP_THUMB_MARGIN = 0.15      # 要比第二名高出这么多，免得两个都像时瞎选
_THUMB_TEMPLATE = None


def find_equip_thumb(image: np.ndarray,
                     slots: list[Optional[tuple[int, int]]]) -> Optional[int]:
    """哪个选项挂着游戏的 👍 推荐标记（返回下标），认不准返回 None。"""
    import cv2

    global _THUMB_TEMPLATE
    if _THUMB_TEMPLATE is None:
        path = Path(__file__).resolve().parents[2] / "assets/image/ui/equip_thumb.png"
        _THUMB_TEMPLATE = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                                       cv2.IMREAD_GRAYSCALE) if path.exists() else False
    if _THUMB_TEMPLATE is False:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    scores = []
    for pt in slots:
        if pt is None:
            scores.append(-1.0)
            continue
        cx, cy = pt[0] + EQUIP_THUMB_DXY[0], pt[1] + EQUIP_THUMB_DXY[1]
        win = gray[max(0, cy - 20):cy + 20, max(0, cx - 20):cx + 20]
        if win.shape[0] < _THUMB_TEMPLATE.shape[0] or win.shape[1] < _THUMB_TEMPLATE.shape[1]:
            scores.append(-1.0)
            continue
        scores.append(float(cv2.matchTemplate(win, _THUMB_TEMPLATE, cv2.TM_CCOEFF_NORMED).max()))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    best = order[0]
    second = scores[order[1]] if len(order) > 1 else -1.0
    if scores[best] >= EQUIP_THUMB_MIN and scores[best] - second >= EQUIP_THUMB_MARGIN:
        return best
    return None


def equip_popup_present(image: np.ndarray, center: tuple[int, int]) -> bool:
    """弹窗在不在。和 `find_equip_slots` 是同一条判据，给"点完之后复查"用。"""
    return find_equip_slots(image, center)[0] is not None


# `find_equip_options`（OCR 读弹窗上的效果概括、再去比卡库描述）已于 2026-09-23 删除。
# 定下的做法是**按位置选**（见 find_equip_slots）：选项次序固定、和 recipes 一致，
# 不需要认字。旧方案一直在 3/4 和 4/4 之间摇摆，别再接回来。




# ---------------------------------------------------------------- 古币机（天赋召唤的物件）
# 2026-10-12（古币机是天赋点出来的、放在场上，可以挪到不碍事的地方）：原来"逐格点开找"，
# 古币机点开不弹英雄详情面板，拿过古币机的 15 局一次都没找到 → 古币全打给了英雄。
# 古币机是固定外观的物件（不随等级变）→ 灰度多模板匹配。assets/image/objects/coin_machine_*.png 取自 4 局真实截图。
# 实测：没拿古币机的 60 局最高 0.57；拿了的局（不含做模板的局）0.69~0.99 → 门槛 COIN_MACHINE_MIN=0.62。
# 调用方只在"已拿古币机天赋、还没找到"时调用，并要求连续两帧同一格。模板中心往下 ~52px = 格子中心。
COIN_MACHINE_MIN = 0.62
_COIN_TPLS = None


def find_coin_machine(image_rgb) -> Optional[tuple]:
    """-> ((格子 x, y), 分数) 或 None。image 是 RGB（线上截图）。"""
    global _COIN_TPLS
    import cv2
    if _COIN_TPLS is None:
        _COIN_TPLS = []
        for f in sorted((Path(__file__).resolve().parents[2] / "assets" / "image" / "objects").glob("coin_machine_*.png")):
            t = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_GRAYSCALE)
            if t is not None:
                _COIN_TPLS.append(t)
    if not _COIN_TPLS or image_rgb is None:
        return None
    x0, y0 = 200, 150
    g = cv2.cvtColor(image_rgb[y0:640, x0:1100], cv2.COLOR_RGB2GRAY)
    best = (-1.0, None)
    for tg in _COIN_TPLS:
        for s in (0.82, 0.88, 0.94, 1.0, 1.06, 1.12):
            t = cv2.resize(tg, None, fx=s, fy=s)
            r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (mx, (loc[0] + x0 + t.shape[1] // 2, loc[1] + y0 + t.shape[0] // 2))
    score, c = best
    if c is None or score < COIN_MACHINE_MIN:
        return None
    foot = (c[0], c[1] + 52)
    cell = min((cc for row in OWN_BOARD_ROWS for cc in row), key=lambda cc: (cc[0] - foot[0]) ** 2 + (cc[1] - foot[1]) ** 2)
    if (cell[0] - foot[0]) ** 2 + (cell[1] - foot[1]) ** 2 > 60 ** 2:
        return None
    return tuple(cell), round(float(score), 3)
