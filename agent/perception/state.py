"""对局状态数据模型。

GameState 是感知层的输出，同时是决策层（规则引擎 / Laya）的输入。
Laya 直接消费 to_laya_state() 产出的 JSON。

阶段（phase）取值：
  hero_pick   选棋手
  loading     读条（本局禁用公示）
  prepare     准备阶段（商店打开，可买牌/用牌/布阵）
  battle      战斗阶段（可看商店但不能用牌）
  auction     拍卖（预览+出价同屏，倒计时结束统一落槌）
  talent      天赋三选一（模态弹窗，不选就一直在，可叠加在其他阶段上）
  settle      出局/结算（"尚未结束" 或最终排名）
  unknown     未识别
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ShopSlot:
    """商店一个槽位（共 4 个）。"""

    index: int                      # 0-3
    piece: Optional[str] = None     # 卡名（对应 data/cards.json 键），识别失败为 None
    cost: Optional[int] = None
    faction: str = ""               # 派系（逐鹿/河洛/三分之地/日落海/大河…）
    sold: bool = False              # 已购买后槽位为空


@dataclass
class BoardCell:
    """棋盘/备战席一个格子上的棋子。"""

    piece: str
    row: int = -1                   # 棋盘行；备战席为 -1
    col: int = -1
    star: int = 1                   # 星级
    items: list[str] = field(default_factory=list)


@dataclass
class AuctionLot:
    """拍卖一个拍品位。"""

    index: int                      # 0-5
    piece: Optional[str] = None     # 拍品卡名
    top_bid: int = 0                # 当前最高出价
    top_bidder: str = ""            # 最高出价者昵称（自己=我方昵称时即为领先）
    bidders: int = 0                # 出价人数
    sold: bool = False              # 已拍出
    x: int = -1                     # 卡名 OCR 框中心（实测坐标，不是算出来的槽位）
    y: int = -1                     # 点拍品用它，UI 偏移也不会点空


@dataclass
class TalentOption:
    """天赋三选一 / 棋手特性三选一的一个选项。"""

    index: int                      # 0-2
    name: str = ""
    desc: str = ""
    x: int = -1                     # 卡名 OCR 框中心（棋手特性用实测坐标点击，不写死三列）
    y: int = -1


@dataclass
class AuctionState:
    lots: list[AuctionLot] = field(default_factory=list)
    countdown: int = -1             # 右下角倒计时秒数，-1 未知
    my_gold: int = -1               # 拍卖时我的金币（-1 未读到）
    preview: bool = False           # 预览期：拍品还在飞入，**出价按钮尚未出现**
    next_price: int = -1            # 「出价」按钮下面的数字 = 对**当前选中的拍品**再出一次要付的价（-1 读不到）
    expected: int = -1              # 按拍品卡等距居中排布推算的总张数（= 剩余玩家数）；-1 = 推不出（2026-10-11）
    won: bool = False               # 顶部写「已获得拍品」：本场已经拍到一件（每人只能拍一件），
                                    # 出价按钮不会再出现 —— 再去点拍品纯属浪费（2026-09-23）
                                    # 2026-09-22 确认：先点中卡牌，等正式开始按钮才出来


@dataclass
class PlayerRow:
    """左侧玩家列表一行。"""

    index: int
    hp: int = -1
    name: str = ""                  # 通常 OCR 不到名字，仅血量可靠


@dataclass
class GameState:
    """一帧完整的对局状态。"""

    phase: str = "unknown"
    round_no: int = 0               # 第 N 回合
    countdown: int = -1             # 阶段剩余秒数
    gold: int = 0
    energy: int = 0                 # 手牌能量 "0/3" 的分子
    energy_max: int = 0
    hp: int = 60
    #  我方生命值。**初始 60，不是 100**（官方新手指引）；读不到时沿用上一帧。
    #  2026-09-22 之前这个字段**从来没被提取过**，1989 个 prepare 帧全是默认值，
    #  等于规则层所有"血量危险"分支一次都没生效过。
    level: int = 1
    level_cost: int = -1            # "升级"按钮上的花费
    level_btn: tuple | None = None  # "升级"按钮的实测中心点（OCR 定位；None=这帧没看见按钮，禁止点）
    board_seen: bool = False        # 这一帧真的看过棋盘（商店关着才看得见）；False 时 board_occupied 无意义
    board_occupied: list = field(default_factory=list)  # **观测到**的己方占用落点（不是记账！见 detect_board_units）
    refresh_cost: int = -1          # "刷新"按钮上的花费（-1=读不到；有免费次数时按钮显示"免费刷新"）
    xp: int = 0
    xp_needed: int = 0
    banned_faction: str = ""        # 本局禁用派系（读条/商店右上角"禁：大河"）
    population: int = 0             # 已上场纯英雄数
    population_cap: int = 0
    shop_open: bool = False         # 商店面板是否展开（面板会盖住棋盘上半部分和人口显示）
    shop_unreadable: bool = False   # 商店 4 格同时读空（翻牌 / 淡入动画）：这一帧读不出，不是售空（2026-10-11）
    card_pick: bool = False         # 效果牌触发的"卡牌N选1"界面（**占用商店位置**，不选就一直卡着）
    card_options: list = field(default_factory=list)  # [{name,x,y}] N选1的候选
    screen_cls: list = field(default_factory=list)  # 界面分类模型（影子）：[细类, 置信度]
    card_pick_quota: list = field(default_factory=list)  # 「请选择k/n张卡牌」横幅读到的 [k, n]（多选时 n>=2）
    population_valid: bool = False  # 本帧是否**真的**读到了人口（商店开着时读不到，
                                    # 那时 population 是上一帧的缓存，不能用来做对账）         # 人口上限（0=未读到；召唤物/图腾不占人口）
    shop: list[ShopSlot] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)           # 手牌卡名
    hand_slots: list[dict] = field(default_factory=list)    # 手牌位置 [{name,x,y}]（拖拽抓手点）
    hand_by_layout: bool = False      # 手牌是按图标布局读的（位置/张数可靠，不需要再推断"漏读"）
    hand_unread: list = field(default_factory=list)
    hand_count: list = field(default_factory=list)   # 左下角剪贴板图标的 "N/M"（手牌数 / 上限，2026-10-15）；读不到为空
    #  [[x, y, 露出来的字], ...]：按图标布局**确定有一张牌**、但露出来的字认不出是谁的位置
    #  （叠放时只露一两个字）。执行层会点开 / 长按去看。
    income_next: int = 0              # 下回合能量增加（"+N"）
    bench: list[BoardCell] = field(default_factory=list)
    board: list[BoardCell] = field(default_factory=list)
    players: list[PlayerRow] = field(default_factory=list)  # 左侧血量列表
    auction: Optional[AuctionState] = None                  # 拍卖阶段才有
    talents: list[TalentOption] = field(default_factory=list)  # 天赋弹窗才有
    hero_panel_open: bool = False
    talent_pending: Optional[list] = None
    #  [x, y]：左侧挂着「天赋未领取」—— 天赋三选一被收起来了，点它才会重新弹出。
    #  不处理的话这次升级的天赋就白丢了（2026-09-23 第 15 回合实战）。
    lord_maxed: bool = False
    #  棋手已满级：左下角写「满级」，原来的点数徽章和升级按钮都没了。
    #  读不出来时等级会被读成 1、升级费沿用旧值 → 决策层一直"想升级找不到按钮"，钱花不出去。
    #  英雄详情面板开着（左下角出现「出售」「剩余数量」）。
    #  它盖住升级按钮，会导致"有钱也升不了级"，必须先关掉。
    awaiting_target: bool = False
    #  顶部挂着「请指定英雄使用」：牌已经举在手上，**还差"再点一下目标"这一步**。
    #  不补这一步的话，牌就一直卡在这个状态，后续动作全部无效。
    unit_anchors: dict[str, list] = field(default_factory=dict)
    vision_units: dict[str, str] = field(default_factory=dict)
    unit_names: dict[str, list] = field(default_factory=dict)   # 认人模型：格子 -> [英雄名, 置信度]（agent/perception/unit_id.py）
    #  {"x,y"(格子): "hero"/"summon"} —— D-FINE 视觉检测认出的己方单位（agent/perception/vision_units.py）。
    #  旧徽章检测漏掉的格子（尤其 C1）会据此补进 unit_anchors；**只加占用、不删记录**。
    #  {"x,y"(格子): [锚点x, 锚点y]} —— **往单位身上拖东西用这个，不是格子中心**。
    #  单位是画在格子上方的（等级徽章高出约 125px），往格子中心丢 = 丢在脚前空地
    #  （2026-09-22 发现装备总是往空地送）。锚点来自**观测到的徽章位置**。
    equip_counts: dict[str, int] = field(default_factory=dict)
    #  {"x,y": 装备件数}。**每个英雄最多 3 件**，满了游戏会直接拒绝拖放
    #  （装备牌留在手上、四选一不弹），所以给装备前必须先看这个。
    # {"x,y"(格子): 单位等级}。**等级徽章里的数字**（1 / 6 / 17 / 52 …），是"谁练得最高"的
    # 直接证据 —— 换主C 就看它（2026-09-23：主C 不能一直停留在开场时的 1 阶英雄，
    # 要换成 3 阶左右、能升到高等级的英雄）。不是每帧都读，见 extractors.read_unit_levels。
    unit_levels: dict[str, int] = field(default_factory=dict)
    # {"x,y"(格子): 觉醒进度 0~3}：等级徽章外圈亮了几段（每合成一次同名亮一段，左→上→右）。只收读得准的。
    awaken_progress: dict[str, int] = field(default_factory=dict)
    #  **键必须是字符串不能是元组**：GameState 每帧都要 json.dumps 进 states.jsonl，
    #  元组键会抛 TypeError，而那一抛会打断整帧后续流程（包括拖拽回执），
    #  表现为"动作执行了但记账学不到结果"→ 对账反复移除再重上，死循环
    #  （2026-09-22 第十四局实测：马可波罗一直上不去，其实是这里崩的）。
    talent_can_refresh: bool = True   # 天赋界面底部写「刷新次数已用完」时为 False
                                     # （别只靠自己记账：记账会因为等级读数漂移而失效）
    free_refresh: int = 0           # 免费刷新次数
    timestamp: float = field(default_factory=time.time)

    @property
    def board_size(self) -> int:
        return len(self.board)

    @property
    def bench_free(self) -> int:
        return max(0, 9 - len(self.bench))

    def to_laya_state(self) -> dict[str, Any]:
        """序列化为 Laya predict 的 state。全中文描述。"""
        shop_desc = [
            f"槽位{s.index}: {s.piece or '未识别'}({s.cost or '?'}费/{s.faction or '?'})"
            for s in self.shop
        ]
        board_desc = [
            f"{c.piece}({c.star}星)位于{c.row}行{c.col}列" for c in self.board
        ]
        bench_desc = [f"{c.piece}({c.star}星)" for c in self.bench]
        data: dict[str, Any] = {
            "阶段": self.phase,
            "回合": self.round_no,
            "剩余秒数": self.countdown,
            "金币": self.gold,
            "能量": f"{self.energy}/{self.energy_max}",
            "血量": self.hp,
            "等级": self.level,
            "本局禁用派系": self.banned_faction,
            "商店": shop_desc,
            "手牌": self.hand,
            "场上棋子": board_desc,
            "备战席": bench_desc,
        }
        if self.auction:
            data["拍卖"] = [
                f"{lot.piece or '?'} 当前价{lot.top_bid} 出价人{lot.top_bidder or '无'}"
                + ("（已拍出）" if lot.sold else "")
                for lot in self.auction.lots
            ]
            data["拍卖倒计时"] = self.auction.countdown
        if self.talents:
            data["天赋选项"] = [f"{t.name}: {t.desc}" for t in self.talents]
        return data

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameState":
        data = dict(data)
        data["shop"] = [ShopSlot(**s) for s in data.get("shop", [])]
        data["bench"] = [BoardCell(**c) for c in data.get("bench", [])]
        data["board"] = [BoardCell(**c) for c in data.get("board", [])]
        data["players"] = [PlayerRow(**p) for p in data.get("players", [])]
        data["talents"] = [TalentOption(**t) for t in data.get("talents", [])]
        if data.get("auction"):
            auc = dict(data["auction"])
            auc["lots"] = [AuctionLot(**lot) for lot in auc.get("lots", [])]
            data["auction"] = AuctionState(**auc)
        return cls(**data)
