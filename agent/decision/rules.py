"""规则决策引擎 v0：纯启发式规则，不依赖 Laya。

每个 decide_* 方法返回一个 Action（dict），由 executor 执行；
返回 None 表示该决策点无操作。
"""

from __future__ import annotations

import json
import math
import time
import logging
import collections
import re
from pathlib import Path
from typing import Any, Optional

from ..perception.extractors import BTN_SHOP_TOGGLE, HAND_MAX, cell_key
from ..perception.state import GameState, ShopSlot

from agent.decision import meta
from agent.decision import engines as ENG
from agent.decision.lord_config import LordRule as _LordRule      # 棋手专属名单 / 阈值（data/lords/<棋手>/lord_rules.json；2026-10-22）

logger = logging.getLogger(__name__)

# 会改变手牌/棋盘的动作：执行层一帧只做一个，所以决策层一次也只产出一个。
# **截断有两处**（decide_all 的 `_verified` 早退分支 + 循环末尾），改的时候两处都要改：
# 2026-09-22 第六局就是只改了后一处，前一处绕过去了，枫叶舞又被吞了一整局。
# 装备的阵营排除（合法性：游戏拒收；卡面 / 官网文字没写，已人工确认）。
# 2026-10-16 晚第 676 局：阿科米亚之子往日落海安琪拉身上拖，"没生效"——阿科米亚之子只能给非日落海英雄携带
EQUIP_FACTION_EXCLUDE = {"阿科米亚之子": "日落海"}

MUTATING_TYPES = {"drag_card", "move_unit", "deploy", "sell_unit", "sell_card",
                  "sell", "use_card", "equip", "sequence"}


def _R_norm(f: str) -> str:
    try:
        from agent.decision import routes as _Rn
        return _Rn.norm_banned(f)
    except Exception:                                       # noqa: BLE001
        return f


def _R_members(comp) -> list:
    if not comp:
        return []
    try:
        from agent.decision import routes as _Rm
        return list(_Rm.members(comp))
    except Exception:                                       # noqa: BLE001
        return [h.get("name") for h in comp.get("heroes") or [] if isinstance(h, dict)]


def _IR_on(name: str) -> bool:
    from agent.decision import interim_rules as _IRm
    return _IRm.on(name)



def _pop_occluding() -> set:
    """挡人口数字的格子（extractors.POP_OCCLUDING_CELLS）：站人后头顶徽章 / 血条盖住数字。"""
    from ..perception.extractors import POP_OCCLUDING_CELLS
    return {tuple(c) for c in POP_OCCLUDING_CELLS}

class RuleEngine:
    """规则引擎。

    数据源（按优先级合并）：
    - data/cards.json：爬虫从游戏百科自动抓取的全量卡牌库（名称/派系/标签/技能）
    - data/chess_pieces.json：人工维护的策略覆盖层（费用/优先级/定位），可选
    """

    def __init__(self, pieces_db_path: Path | str, cards_db_path: Path | str | None = None):
        self.pieces: dict[str, dict[str, Any]] = {}

        # 爬虫库（基础）
        cards_path = Path(cards_db_path) if cards_db_path else None
        if cards_path and cards_path.exists():
            db = json.loads(cards_path.read_text(encoding="utf-8"))
            for name, card in db.get("cards", {}).items():
                tags = card.get("tags", [])
                self.pieces[name] = {
                    "cost": card.get("cost", 0),
                    "category": card.get("category", ""),
                    "quality": card.get("quality", 0),
                    "synergies": [card["faction"]] if card.get("faction") else [],
                    "role": "tank" if any("前排" in t or "坦克" in t for t in tags) else "dps",
                    "priority": 5,          # 默认优先级，由覆盖层调整
                    "skill": card.get("skill", ""),
                    "tags": tags,
                    # 攻击距离：近战 1、远程 4~5（有的装备"限定远程/近战英雄"，见 _equip_allowed）
                    "range": (card.get("properties") or {}).get("attackDistance", 0),
                }

            # **天赋送出的同名牌**（爬虫存成「X(赠送)」，键「X」是天赋本身）：手牌里读到的「X」其实是送出的那张 ——
            # 原来按天赋牌处理、从不使用（2026-10-14 第 627 局始祖熊灵选了却一直没上场；卡面审计第 1 条）。
            # 手牌里不会出现天赋本身，所以手牌用途按赠送牌的类别；召唤物类（不占人口的单位）标 no_pop，上场不受人口限制
            for gname, gcard in db.get("cards", {}).items():
                if not gname.endswith("(赠送)"):
                    continue
                base = gname[: -len("(赠送)")]
                if (self.pieces.get(base) or {}).get("category") != "天赋牌":
                    continue
                gcat = gcard.get("category", "")
                self.pieces[base] = dict(self.pieces[base], category="英雄牌" if gcat in ("召唤物", "英雄牌") else gcat,
                                         quality=gcard.get("quality", 0), skill=gcard.get("skill", ""),
                                         no_pop=gcat == "召唤物" or "不占人口" in (db["cards"][base].get("skill") or ""),
                                         talent_gift=True)

        # 人工覆盖层（费用/优先级/羁绊阈值/运营策略）
        db = json.loads(Path(pieces_db_path).read_text(encoding="utf-8"))
        for name, override in db.get("pieces", {}).items():
            self.pieces.setdefault(name, {}).update(override)
        self.synergy_threshold: dict[str, list[int]] = db.get("synergy_bonus_threshold", {})
        strategy = db.get("strategy", {})
        self.main_factions: list[str] = strategy.get("main_factions", [])
        self.auction_max_bid: int = strategy.get("auction_max_bid", 5)
        self.auction_snipe_sec: int = strategy.get("auction_snipe_sec", 8)
        self.gold_reserve: int = strategy.get("gold_reserve", 10)
        self.refresh_cost: int = strategy.get("refresh_cost", 1)
        # 升级棋手的单独底线：与 gold_reserve（买牌预留）分开，否则前期永远升不了级
        self.level_reserve: int = strategy.get("level_reserve", 2)
        # 拍卖的预留同样要和 gold_reserve 分开：拍卖几回合才一次，且拍到的往往是
        # 商店刷不出来的高阶牌，为了"留 10 点买 3 费牌"而整场不出价是亏的
        # （2026-09-22 修正：整场不出价）。
        self.auction_reserve: int = strategy.get("auction_reserve", 2)
        # 多套备选阵容（开发期 Laya 设计）：运行时按本局禁用派系自动选择
        self.comps: list[dict] = strategy.get("comps", [])
        self._active_comp: dict | None = None
        self._resolved_ban: str = ""
        self.last_gold: int = 0         # 最近一次准备阶段读到的金币（拍卖阶段金币位被倒计时占用时兜底）
        self._deployed_round: int = -1  # 已计数的回合
        self._deployed_count: int = 0   # 本回合已拖上场的英雄数
        self._deployed_positions: list[tuple[int, int]] = []  # 本回合上场位
        self._carry: tuple[int, int] = (660, 505)               # 装备/效果卡目标（默认己方后排中）
        self._carry_piece: str = ""                             # 主力英雄名（首个上场），装备评分按其定位
        self._locked_round: int = -1                            # 本回合已锁过商店
        self._leveled_round: int = -1                           # 本回合已升过一次级
        self._talent_refreshed_level: int = -1
        self._level_wait_round, self._level_wait_frames = -1, 0                  # 这一级的天赋刷新机会用掉没（**必须在 __init__ 里也建**：
                                                        # __init__ 不调 reset_match，agent 从对局中途启动时
                                                        # reset_match 可能一次都没跑过 -> AttributeError）
        self._field: dict[str, tuple[int, int]] = {}            # 我方场上英雄 -> 位置（跨回合记忆）
        self._bid_piece: str = ""                               # 拍卖：自己出过价的拍品**卡名**
        self._bid_price: int = 0                                # 拍卖：自己的出价
        self._bids: dict = {}                                   # 本场拍卖：卡名 -> 我对它出过的价
        self._bids_round: int = -1
        self._hand_last: list[str] = []                         # 上一帧手牌（拍卖升级材料判定用）
        self._skill_round: int = -1
        self._explained_round: int = -1
        self._gone_prev: set = set()                            # 上一帧疑似离场的英雄（要连续两帧才采信）
        self._maple_bound: str = ""                             # 枫叶舞当前绑在谁身上（**卖掉他才能重绑**）                             # 本回合已用过棋手技能牌（一次即生效）
        self._awakened: set[str] = set()                        # 觉醒后回到手牌、待重新上场的英雄
        self._ever_awakened: set[str] = set()                   # 本局觉醒过的英雄（估值用，上场后也保留）
        self._hand_seen_ts: dict[str, float] = {}
        self._sold_round: dict = {}                             # 英雄 -> 最近一次卖掉（场上/手牌）的回合：刚卖的别马上买回来
        self._sold_names: set[str] = set()                      # 本局我们亲手卖掉过的英雄（主C "走丢"时据此区分卖掉 / 账本丢了位置）               # 英雄手牌第一次看到的时间（卖闲置手牌前给换人题 5 秒）
        self._vision_returned: dict[str, tuple] = {}            # 靠"视觉没看到"推断的觉醒回手 -> 原格（人口证伪时撤销）
        self._confirmed_awakened: set[str] = set()              # 有直接证据（点开看 / 觉醒色）的觉醒
        self._hand_since: dict[str, int] = {}                   # 手牌里的英雄从第几回合起就在
        self._sell_tries: dict = {}                             # (回合, 牌) -> 卖手牌点出去的次数
        self._sell_decided: dict = {}                           # (回合, 牌) -> 决定卖它的次数
        self._comp_challenger = None                            # (阵容名, 首次胜出的回合)
        self._comp_switch_round = -99                           # 上一次换阵容的回合
        self._route_locked = False                              # 官方路线已锁定（第 ROUTE_LOCK_ROUND 回合起不再换阵容）
        self._route_key = ""                                    # 锁定的路线（lineups.json 的 key；重名阵容靠它区分）
        self._route_since = 0                                   # 第几回合锁定的（Laya 题面 f9k 写 "committed rN"）
        self._carry_switch_round = -99                          # 上一次换主C 的回合
        self._carry_switch_by = ""                              # 上一次换主C 的来源（"laya" = Laya carry 题）
        self._lord_carry_freq: dict[str, float] = {}
        self._any_carry_freq: dict[str, float] = {}
        self._unknown_logged: set[str] = set()                  # 已提醒过的未知手牌（避免刷屏）
        self._shop_closed_round: int = -1                       # 本回合已关店读过人口
        from agent.decision import lord_config as _LCi
        self._lord: str = _LCi.current()                        # 本局棋手（lord_rules.json 等按它读）
        self._entry_fodder: dict[str, int] = {}                 # 只为吃登场/整备效果上场的英雄 -> 上场回合
        self._prep_left: float = 99.0                            # 本回合准备阶段估计还剩几秒（主循环每帧写；2026-10-14 提速）
        self._drag_tries: dict[tuple, int] = {}                 # (回合,牌名,落点) -> 已尝试次数
        self._bad_cells: set = set()                            # 实测拖不进去的落点，本局不再尝试
        self._busy_cells: set = set()                           # 疑似已被别的单位占着的格子（拖失败推断出来的）
        self._cell_owner: dict = {}                             # 格子 -> 曾放过谁（位置规划：一人一格，不抢位）
        self._card_fails_round: dict[tuple, int] = {}          # (回合,牌名) -> 本回合失败次数
        self._parked_cards: set = set()                        # 本回合搁置的牌（失败太多，让位给别的）
        self._drag_fails: dict[tuple, int] = {}                 # (牌名,落点) -> 实测失败次数（由 confirm_drag 写）
        self._verified = False                                  # 调用方是否在给拖拽回执（接了就关掉开环猜测）
        self._reconcile_sig: tuple | None = None                # 对账告警去重（同一回合同一差值只报一次）
        self._board_prev: set = set()                           # 上一帧的棋盘观测（要连续两帧一致才采信）
        self._last_sold: tuple = ("", -1)                       # (本回合卖掉的英雄, 回合) —— 装备会跟着回到手里
        self._sell_undo: dict = {}                              # 最近一次卖人前的账本快照（卖出没生效时 undo_sell 恢复）
        self._sell_fails: dict = {}                             # (回合, 英雄) -> 卖出没生效次数；≥2 本回合不再卖他
        self._freed_cell: tuple = ((), -1)                      # (本回合卖人腾出的格子, 回合) —— 上场优先用
        self._worn: dict[str, set] = {}                         # 英雄 -> 身上穿着的装备名（同名只能穿一件）
        self._equip_rejects: dict[str, int] = {}                # 格 -> 拖装备没弹四选一的次数
        self._shop_idle: list = [-1, 0]                         # [回合, 商店开着却无事可做的连续帧数]
        self._closed_idle: list = [-1, 0]                       # [回合, 商店关着且无事可做的连续帧数]
        self._seq_tries: dict = {}                              # (回合, 牌) -> 组合动作执行次数
        self._idle_opened_round: int = -1                       # 本回合是因为"关着空转"才开的店
        self._level_hold: tuple = (-1, 0)                       # (回合, 升级费)：等弈星上场期间给升级留钱
        self._prev_carry: str = ""                              # 被换下来的老主C
        self._lord_boosts: set = set()                          # 本局选过的棋手特性（红叶染锋等）
        self._talents_taken: set = set()                        # 本局选过的天赋（多彩古币这类会改写出牌规则，card_target_ok 要读）
        self._core_seen: bool = False                           # 阿科米亚核心本局有过等级来源（核心只增不减；跳级"core_level"条件）
        self._hp_by_round: dict = {}                            # 回合 -> 准备阶段读到的棋手血量（预言牌按上一场输赢押）
        self._predict_round: int = -1                           # 本回合已经押过预言
        self._boost_round: int = -1                             # 本回合已经打过锦囊（登场多触发一次）
        # 天赋等额外给的人口上限（「战术模组」：可上阵人数+1）。感知层据此放宽合法上限，见 extractors.VALID_POP_CAPS
        self.pop_bonus = 0
        self._refresh_streak = (-1, 0)       # (回合, 本回合连续付费刷新而没买到牌的次数)
        self._awaken_seen: dict = {}          # 英雄 -> 上次读到的觉醒进度（0~3）
        self._awk_suspect_since: dict = {}    # (回合, 英雄) -> 第一次出现觉醒回手嫌疑的时间
        self._awk_cleared: set = set()        # (回合, 英雄)：点开看过、只是合成材料，不再当嫌疑
        self._vision_miss: dict = {}          # 英雄 -> 视觉连续多少帧没在它的格子看到人
        self.pool_left: dict = {}             # 英雄 -> 公共牌池剩余张数（点开详情「剩余数量」，主循环同步）
        self._maple_tries: dict = {}                            # 回合 -> 本回合试着绑枫叶舞的次数
        self._reposition_round: int = -1                        # 本回合已做过一次站位纠正
        self._last_hero_played: tuple = ()                      # (回合, 英雄)：本回合最近一次**成功**打出的英雄牌
        self._maple_try_ts: float = 0.0                         # 最近一次拖枫叶舞的时间（等卡面变灰）
        self._steal_moved: set = set()                          # (回合, 牌名)：本回合已为夺取之术挪过人
        self._unforged: dict = {}                               # 格 -> 没铸成的基础装备名（四选一漏选，身上带白光环）
        self._reforge_tried: set = set()                        # (格, 装备名)：已经点过一次补选（成没成都不再点）
        self._deploy_fail_cells: dict = {}                      # 回合 -> 本回合上场失败过的格子
        self._confirmed_at: dict = {}                           # 英雄 -> (回合, 格)：本回合点开确认过它在这一格
        self._transform_pending = None                          # (回合, 牌)：化铁为金刚打出，下一张新到手的 5 阶英雄就是它变的（觉醒）
        self._unit_lv: dict = {}                                # 英雄 -> 最近读到的徽章等级（估值用）
        self._swapped_for: set = set()                          # (回合, 手牌英雄)：本回合已经为它卖过人
        self._pop_full_guess: int = -1                          # 推断"人口已满"的回合
        self._pop_over_round: int = -1                          # 拖新英雄时人口变红（硬证据）的回合
        self._used_on: set = set()                              # (回合, 牌名, 英雄)：本回合已经对这个英雄用过这张牌
        self._repeat_reject_rounds: dict = {}                   # 牌名 -> {回合}：观察到"同回合对同一人再用被拒"的回合
        self.coin_machine_cell: Optional[tuple] = None          # 古币机（天赋召唤物）站的格子，点开确认过才填
        self.object_cells: dict = {}                            # 格子 -> 物件名（重剑 / 古币机 / 图腾…）：占格、不是英雄，不能往上放人
        self.coin_machine_wanted: bool = False                  # 拿了「古币机」天赋、还没找到它在哪
        self._no_repeat: set = self._load_learned().get("no_repeat_same_hero", set())  # 实测"同回合不能对同一人再用"的牌
        self._stop_shop_round: int = -1                         # 本回合已决定"不买了"（Laya 选 stop），别再开店
        self.decisions: list = []                               # 决策点记录（主循环落盘到 decisions.jsonl）
        self._decision_keys: set = set()
        self._transfer_round: int = -1                          # 本回合是否已为"转移给新主C"卖过人
        self._hand_book: collections.Counter = collections.Counter()   # **手牌账**：卡名 -> 张数
        self._vision_cleared: dict = {}                                    # 格子 -> 认人模型清掉失败记录的回合（每格每回合一次）
        self._book_pending: collections.Counter = collections.Counter()  # 上一帧"读全了"的手牌（连续两帧一致才校正）
        self._engine_levels: dict = {}                          # 棋手经验型英雄 -> 它在场期间贡献过几次升级
        # 开店/关店**分开计数**。合用一个计数器的后果：一回合里只要关过一次店，
        # 配额就用光了，之后再也打不开 —— 2026-09-22 第二局回合7 因此拿着 12 能量、
        # 商店里躺着主C 干将莫邪，整回合一下都没买（表现为"不动了"）。
        # 正常节奏是 开→买→关→上场→再开→再买，所以各给 2 次。
        self._shop_opens: int = 0                               # 本回合已开店次数
        self._shop_closes: int = 0                              # 本回合已关店次数
        self._round_seen: int = -1                              # 已初始化过的回合（每回合计数器复位用）
        self._buys_this_round: int = 0                          # 本回合已买入几张（按实际执行计）
        self.ledger: list[dict] = []                            # 事件账本：我做过什么、因此得到了什么
        self._hand_prev: set = set()                            # 上一帧手牌（用来发现"凭空多出来的牌"）
        self._last_card_use: tuple = ("", -1)                   # 最近用掉的效果牌 (牌名, 回合)
        self._drag_attempts: dict = {}          # (回合,牌,落点) -> 真正点出去过几次（不看回执）
        self._src_missing: dict = {}            # (回合,牌) -> 拖前抓牌校验没找到几次
        self._equip_seen: dict = {}             # 格 -> 看到过的最多装备件数（只增，卖人/挪走时清）
        self._entered: dict = {}                # 卡名 -> 这次上场是第几回合（board_facts 首次看到它在账本里；离场即删）
        self._talent_open_tries: dict = {}      # 回合 -> 试着打开「天赋未领取」几次
        self._last_pop_seen: tuple = ()         # 最近一次**真读到**的 (人口, 上限)，只用于日志
        self._cell_fails: dict = {}             # 落点 -> 这一局在这格上拖失败过几次
        self._pop_low_pending = None            # 待确认的"人口比账本少"读数（删记录要两帧）
        self._card_pick_tries: dict = {}         # 同一组 N选1 候选点过几次
        self._seen_occupied: set = set()         # 本局**曾经**观测到有人的格子（只加不减）
        self._merge_missing: dict = {}           # (牌,记账位) -> 连续几帧观测不到（合成前的复核）
        self._skip_logged: dict = {}             # 牌名 -> 上次为它打过"跳过"日志的回合
        # 特性"选哪个"交给 Laya 临场判断（分层：规则管能不能，Laya 管值不值）。
        # 由 HybridEngine 注入，签名 (state, viable) -> TalentOption | None
        self._laya_pick = None

    # ---------- 对局生命周期 ----------

    def reset_match(self) -> None:
        """新对局开始：清空所有跨帧记忆。

        agent 进程会连续打多局，而这些字段都是**单局状态**。不清的话，
        上一局的场上英雄记录会被当成本局的——2026-09-22 实战里第 1 回合
        场上 0 个英雄，决策层却拿着上一局的 `_field` 去给"英雄"穿装备。
        """
        self.__dict__.pop("_unconfirmed_deploy", None)      # 第 708 局：未确认上场表也是单局状态
        self.__dict__.pop("_maple_cleared", None); self.__dict__.pop("_maple_bright_since", None)   # 10-21 枫叶舞亮灰防抖
        # **这个 guard 只能管日志，不能提前 return**：它判"干净"只看三个字段，
        # 而下面还清着十几个回合级计数器（_leveled_round / _locked_round /
        # _shop_opens …）。上一局如果没买没上人（例如开局就中断），那三个字段是空的，
        # 提前返回就会把计数器漏进下一局 —— 表现为新局里某个回合莫名其妙不升级/不锁店。
        if self._field or self.ledger or self._entry_fodder:
            logger.info("新对局：清空上一局的场上记录（%d 个英雄）、账本（%d 条）",
                        len(self._field), len(self.ledger))
        self._field.clear()
        for _k in ("_unclear_cells", "_move_hist", "_talent_route_carry", "_swap_benched", "_swap_asked", "_wait_lvbtn",      # 2026-10-09 新加的单局状态（审查）
                   "_pending_use", "_pending_transfer", "_pending_pop",             # 2026-10-13 执行后才记账的暂存
                   "_card_hero_fail", "_pending_pick",                               # 2026-10-15 牌对人无效 / N 选 1 候选
                   "_pop_anchor", "_awk_expect"):                                    # 2026-10-19 pop_now 锚点 / 预期觉醒回手
            self.__dict__.pop(_k, None)
        self._entry_fodder.clear()
        self._awakened.clear()
        self._ever_awakened.clear()
        self._vision_returned.clear()
        self._confirmed_awakened.clear()
        self._hand_since.clear()
        self._hand_seen_ts.clear()
        self._sold_names.clear()
        self._sold_round = {}
        self._sell_tries.clear()
        self._sell_decided.clear()
        self._comp_challenger = None
        self._comp_switch_round = -99
        self._route_locked = False
        self._route_key = ""
        self._route_since = 0
        self._forced_released = False
        self.__dict__.pop("_followed_carry", None)
        self._forced_blocked = False
        self._forced_switched = 0
        self._obtained = set()               # 本局拿到过的英雄（手牌里出现过 / 在场上）：认人模型只认这些
        self._roll_rounds = set()
        if self.__dict__.get("_forced_orig"):
            self.forced_route = self._forced_orig          # 新局回到首选路线
        self._forced_lv = {}
        self._carry_switch_round = -99
        self._carry_switch_by = ""
        self.deploy_first = None
        self._bad_cells.clear()
        self._busy_cells.clear()
        self._cell_owner.clear()
        self._drag_tries.clear()
        self._drag_fails.clear()
        self._card_fails_round.clear()
        self._parked_cards.clear()
        self._drag_attempts.clear()
        self._src_missing.clear()
        self._equip_seen.clear()
        self.__dict__.pop("_equip_hist", None)
        self._entered.clear()
        self._talent_open_tries.clear()
        self._last_pop_seen = ()
        self._cell_fails.clear()
        self._pop_low_pending = None
        self._card_pick_tries.clear()
        self._seen_occupied.clear()
        self._merge_missing.clear()
        self._skip_logged.clear()
        self._reconcile_sig = None
        self._board_prev = set()
        self._last_sold = ("", -1)
        self._sell_undo = {}
        self._sell_fails = {}
        self._freed_cell = ((), -1)
        self._worn = {}
        self._equip_rejects = {}
        self._hand_book = collections.Counter()
        self._vision_cleared = {}
        self._book_pending = collections.Counter()
        self._level_hold = (-1, 0)
        self._prev_carry = ""
        self._lord_boosts = set()
        self._talents_taken = set()
        self._core_seen = False
        self._hp_by_round = {}
        self._contract_since = {}
        self._timing_held = {}
        self._predict_round = -1
        self._boost_round = -1
        self.pop_bonus = 0
        self.pop_bonus_ledger = 0          # 记账（卡牌知识库）给的人口加成；画面认的那部分被证伪时退回到它
        self._refresh_streak = (-1, 0)
        self._awaken_seen = {}
        self._awk_suspect_since = {}
        self._awk_cleared = set()
        self._vision_miss = {}
        self._maple_tries = {}
        self._reposition_round = -1
        self.__dict__.pop("_totem_tries", None)
        self.__dict__.pop("_destroy_wait", None)
        self._buddy_link, self._buddy_return = {}, None
        self._last_hero_played = ()
        self._maple_try_ts = 0.0
        self._steal_moved = set()
        self._unforged = {}
        self._reforge_tried = set()
        self._transfer_round = -1
        self._deploy_fail_cells = {}
        self._pop_full_guess = -1
        self._used_on = set()
        self.coin_machine_cell, self.coin_machine_wanted = None, False
        self._coin_move_tries = {}
        self.object_cells = {}
        self._obj_confirmed = {}        # 点开确认过身份的物件格（图腾站位只认它）
        self._confirmed_at = {}
        self._transform_pending = None
        self._unit_lv = {}
        self._swapped_for = set()
        self._decision_keys = set()
        self._engine_levels = {}
        self._hand_prev = set()
        self.ledger = []
        self._carry, self._carry_piece = (660, 505), ""
        self._deployed_round = self._locked_round = self._skill_round = -1
        self._maple_bound = ""
        self._route_group, self._group_settled = (), False      # 主C 组（lord_config）：新局清空
        self._group_followed = False                            # 主C 组谁先出用谁：路线跟现任走（只换一次）
        from agent.decision import hero_traits as _htr0
        _htr0.GLOBAL_RESONANCE = False
        self._maple_lost_cell = None
        self._gone_prev = set()
        self._explained_round = -1
        self._leveled_round = -1
        self.__dict__.pop("_eco_choice", None)
        self._talent_refreshed_level = -1
        self._level_wait_round, self._level_wait_frames = -1, 0
        self._shop_closed_round = self._round_seen = -1
        self._buys_this_round = 0
        self._deployed_count = 0
        self._shop_opens = self._shop_closes = 0
        self._deployed_positions = []
        self._bid_piece, self._bid_price = "", 0
        self._bids, self._bids_round = {}, -1
        self._active_comp, self._resolved_ban = None, None

    # ---------- 拖拽结果回执（闭环反馈） ----------

    # ---------- 手牌账（手牌堆叠后开店就看不清，所以要自己记账，
    #            再用实际查看去修正） ----------
    # 手里有什么全是我们自己的动作带来的：买进、N 选 1 选中、拖出去、卖掉（都按**执行后的回执**记）。
    # 画面读全的那一帧（商店关着、没有认不出的位置）以画面为准整个校正；读不全时账只用来**补认不出的位置**，
    # 读到了的一律以画面为准 —— 和"弱证据不推翻强证据"一致。

    def hand_book_names(self) -> set:
        return {n for n, c in self._hand_book.items() if c > 0}

    def sync_hand_book(self, state: GameState) -> None:
        """每帧对账：读全了就以画面为准；读不全就只把"看到的张数"补进账里（只加不减）。"""
        if state.phase != "prepare":
            return
        # 棋手技能牌（枫叶舞）常驻最左边的技能槽，不在手牌网格里、也不会被用掉，**不进账**
        # （2026-09-23 第九局：技能槽那一帧没读到，账里"多出来"的枫叶舞被补到了 x=792 上）
        seen = collections.Counter(sl["name"] for sl in state.hand_slots or []
                                   if sl["name"] in self.pieces
                                   and self.pieces[sl["name"]].get("category") != "棋手技能牌")
        if not state.shop_open and not state.hand_unread and state.hand_by_layout:
            # **连续两帧读到一样的才整体校正**（2026-09-23 第九局：12 张叠放时一帧读乱，一次减了 7 张）
            if seen != self._book_pending:
                self._book_pending = seen
                return
            if seen != self._hand_book:
                diff = {n: seen[n] - self._hand_book[n] for n in set(seen) | set(self._hand_book)
                        if seen[n] != self._hand_book[n]}
                if self._hand_book:
                    logger.info("手牌账按画面校正：%s", diff)
            self._hand_book = seen
            return
        for n, c in seen.items():
            if self._hand_book[n] < c:
                self._hand_book[n] = c

    def fill_hand_from_book(self, state: GameState) -> None:
        """只剩**一个**位置认不出、账里也正好多出**一张**：就是它（露出来的字对得上才补）。"""
        unread = state.hand_unread or []
        if state.phase != "prepare" or len(unread) != 1:
            return
        seen = collections.Counter(sl["name"] for sl in state.hand_slots or [])
        left = +(self._hand_book - seen)
        left = collections.Counter({n: c for n, c in left.items()
                                    if self.pieces.get(n, {}).get("category") != "棋手技能牌"})
        if sum(left.values()) != 1:
            return
        name = next(iter(left))
        u = unread[0]
        frag = re.sub(r"[^一-鿿]", "", u[2] if len(u) > 2 else "")
        if frag and len(frag) >= 2 and not any(ch in name for ch in frag[:2]):
            return                                   # 露出两个字以上还对不上，不补
        # 2026-10-16 第 667 局：牛魔只露出「摩」（OCR 把"魔"认成形近字）→ 一直不补、没上场。只露一个字是弱证据，
        # 账（买入回执）是强证据：只剩一个位置认不出、账里正好多一张时，单字对不上也按账补
        state.hand_slots.append({"name": name, "x": u[0], "y": u[1], "name_y": u[1] + 25,
                                 "from_book": True})
        state.hand_slots.sort(key=lambda s: s["x"])
        state.hand = [s["name"] for s in state.hand_slots]
        state.hand_unread = []
        logger.info("手牌 x=%d 认不出（露出「%s」），按手牌账补上：%s", u[0], frag, name)

    def note_worn(self, hero: str, item: str) -> None:
        """记下"某英雄身上穿上了某件装备"（拖拽回执成功 / 四选一铸造成功时调用）。"""
        if hero and item:
            self._worn.setdefault(hero, set()).add(item)
            logger.info("装备记录：%s 身上有 %s", hero, sorted(self._worn[hero]))

    def note_wrong_grab(self, act: dict[str, Any], other: str, deployed: bool) -> None:
        """**抓错牌**：想拖 act["piece"]，拖完它还在手里，少的是另一张 `other`（拖前核对时明明对得上）。

        2026-09-24 第一轮第 5 局第 1~2 回合：想拖干将莫邪，实际上场的是牛魔 / 举起来的是风暴巨剑。
        原来按"没生效"处理：拉黑落点格、搁置干将莫邪、推断人口已满；抓起来的装备落在空格上没弹四选一，
        又把那一格记成"装备已满 3 件" —— 后来新主C 站进那一格，所有装备都绕开了主C。
        **这不是格子的问题，也不是这张牌的问题**：不计失败、不拉黑格子。
        抓错的是英雄且人口 +1（deployed）就按实际记账：`other` 站在落点上。
        """
        piece = act.get("piece")
        dst = tuple(act.get("dst") or ())
        self.note("抓错牌", 想拖=piece, 实际=other, 落点=f"{dst[0]},{dst[1]}" if dst else "", 上场=deployed)
        if deployed and dst and self.pieces.get(other, {}).get("category") == "英雄牌":
            self._evict_cell(dst, other)
            self._field[other] = dst
            self._cell_owner[dst] = other
            self._equip_seen.pop(cell_key(dst), None)
            (self.__dict__.get("_equip_hist") or {}).pop(cell_key(dst), None)
            self._equip_rejects.pop(cell_key(dst), None)
            if self._hand_book[other] > 0:
                self._hand_book[other] -= 1
            logger.warning("抓错牌：想拖 %s，上场的其实是 %s —— 按实际记账：%s 在 %s", piece, other, other, dst)
        else:
            logger.warning("抓错牌：想拖 %s，动的其实是 %s —— 不算这张牌/这个格子的失败", piece, other)

    LEARNED_PATH = Path(__file__).resolve().parents[2] / "data" / "learned_card_rules.json"

    @classmethod
    def _load_learned(cls) -> dict:
        """实战里**学到**的卡牌规则（不是卡面写的，是游戏实际行为）。格式：{"no_repeat_same_hero": [...], ...}"""
        try:
            d = json.loads(cls.LEARNED_PATH.read_text(encoding="utf-8"))
            return {k: set(v) for k, v in d.items() if isinstance(v, list)}
        except (OSError, ValueError):
            return {}

    def _save_learned(self, key: str, values: set) -> None:
        d = {k: sorted(v) for k, v in self._load_learned().items()}
        d[key] = sorted(values)
        try:
            self.LEARNED_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            logger.exception("学到的卡牌规则写不进 %s", self.LEARNED_PATH)

    def note_repeat_rejected(self, card: str) -> None:
        # **要在两个不同回合各被拒一次才学**（2026-09-30 第 38 局：古币 被一次偶发的拒收学成"不能重复用"，
        # 而河洛牺牲古币体系本来就是把一堆古币叠在同一个主C 身上 —— 两张转瞬古币烂在手里）。
        # 一次拒收常见的别的原因：抓错牌、目标格记错、拖的那一刻在放特效。
        seen = self._repeat_reject_rounds.setdefault(card, set())
        seen.add(self._round_seen)
        if len(seen) < 2:
            logger.info("记账：%s 同回合对同一英雄第二次被拒（第 %d 次观察，要两个不同回合都出现才学）", card, len(seen))
            return
        # 绝大多数牌（包括古币）可以对同一英雄重复使用 —— 只有推演战术一回合对同一英雄限一次（名单里只留它）。
        # 自动学过 9 张（加速成长 / 简易协作 / 妙音…），都是拒收误判（抓错牌、目标格记错、特效中）→ 不再自动学，只记日志供复盘查执行问题
        self.note("重复用被拒", 牌=card)
        logger.warning("记账：%s 同一回合对同一个英雄用第二次被拒（不学成规则：绝大多数牌可重复用）—— 查执行 / 回执", card)

    def note_unforged(self, cell, base: str) -> None:
        """这一格的英雄身上多了一件**没铸成**的基础装备（四选一漏选/过期），身上会带白光环。"""
        k = cell_key(tuple(cell))
        if (k, base) in self._reforge_tried:
            return
        self._unforged[k] = base
        self.note("未铸成装备", 格=k, 装备=base)
        logger.info("记账：%s 那一格身上有一件没铸成的 %s（之后点它一次试试能不能补选四选一）", k, base)

    def next_reforge(self) -> Optional[tuple]:
        """还没试过补选的 (格, 装备名)；取出即记为"试过"—— **每件只点一次**（记账，防止无限点击）。"""
        for k, base in list(self._unforged.items()):
            self._unforged.pop(k, None)
            if (k, base) in self._reforge_tried:
                continue
            self._reforge_tried.add((k, base))
            return k, base
        return None

    def note_empty_cell(self, cell) -> None:
        """点开那一格没弹出英雄详情 = 多半没人：记一次格子失败（两次后 _cell_dead，装备/效果牌改找别人）。"""
        c = tuple(cell)
        self._cell_fails[c] = self._cell_fails.get(c, 0) + 1
        owner = next((n for n, p in self._field.items() if tuple(p) == c), "")
        self.note("点开是空格", 格=f"{c[0]},{c[1]}", 账本记的=owner, 失败次数=self._cell_fails[c])

    def note_used_on(self, rnd: int, card: str, hero: str) -> None:
        if card and hero:
            self._used_on.add((rnd, card, hero))

    def note_pop_over(self, rnd: int) -> None:
        """拖新英雄时人口数字变红：这回合人口是满的（比读数、比推断都硬）。"""
        self._pop_over_round = rnd
        self._pop_full_guess = rnd
        self.note("人口已满", round=rnd, 证据="拖动时人口数字变红")

    def confirm_drag(self, act: dict[str, Any], ok: bool,
                     blame_cell: bool = True) -> None:
        """拖拽执行完之后，由调用方**重新看一眼画面**给出的回执。

        为什么必须有这个：在此之前，决策层是**开环**的——拖了就记进 `_field`，
        当成一定成功。2026-09-22 实战因此整局卡死：
          第 1 回合把李元芳记成 (540,480)，实际那格站的是牛魔；
          之后每一帧都认为"李元芳在场上"→ 手里那张同名牌被当成合成 →
          往同一个被占的格子拖 → 永远失败 → 触发"合成失败=觉醒回手"的自愈 →
          又"重新上场"到**同一个格子** → 死循环，整局李元芳都没上去。
        回执把"我以为"换成"我看到了"：成功就坐实格子归属，失败就把落点拉黑换一个。

        ok 的判据由调用方给（最省的一种：拖完重新读手牌，那张牌还在 = 没拖进去）。
        """
        self._verified = True
        atype = act.get("type")
        if atype not in ("drag_card", "move_unit"):
            return
        piece = act.get("piece")
        if piece:
            (self.__dict__.get("_unconfirmed_deploy") or {}).pop(piece, None)     # 有回执了（成功坐实 / 失败另行撤销）
        _pp = (self.__dict__.get("_pending_pop") or {}).pop(piece, None) if piece else None
        if _pp and ok:
            self.pop_bonus += _pp
            self.pop_bonus_ledger = self.__dict__.get("pop_bonus_ledger", 0) + _pp
            logger.info("%s：拖拽成功，人口上限 +%d（累计 +%d）", piece, _pp, self.pop_bonus)
        dst = tuple(act.get("dst") or ())
        key = (piece, dst)
        if ok and act.get("kind") == "装备牌":
            self.note_worn(act.get("target", ""), piece)
        tgt = act.get("target") or ""
        if ok and atype == "drag_card" and tgt and self.TRANSFORM_TO_HAND_RE.search(self.pieces.get(piece, {}).get("skill") or ""):
            # 「化铁为金」：目标变成**不在场的 5 阶觉醒英雄**回到手牌（等级保留）。
            # 2026-09-24 第二轮第 5 局：蒙恬变成觉醒东皇太一，账本没删蒙恬 → 空出的人口被庄周占了、
            # 对账又删错了人（主C 嬴政），东皇太一一直上不去。
            self._transform_pending = (self._round_seen, piece)
            if self._last_pop_seen:
                here, cap = self._last_pop_seen
                self._last_pop_seen = (max(0, here - 1), cap)
        if (ok and atype == "drag_card" and tgt and tgt in self._field
                and self.pieces.get(piece, {}).get("category") != "装备牌"    # 护命灵珠"护盾被击破则移除"不是移除英雄（10-10 第 515 局把牛魔 / 大司命删出账本）
                and ("移除" in (self.pieces.get(piece, {}).get("skill") or "")
                     or self.TRANSFORM_TO_HAND_RE.search(self.pieces.get(piece, {}).get("skill") or ""))):
            # **"移除英雄"类的牌（体面离场/遗志/限制性进化）打成功 = 那个人离场了**，账本立刻删掉他
            # （原来不删，要等下次点开看场才纠正；期间装备/枫叶舞/合成都还往那格送）
            pos = self._field.pop(tgt)
            self._cell_owner.pop(pos, None)
            self._equip_seen.pop(cell_key(pos), None)
            (self.__dict__.get("_equip_hist") or {}).pop(cell_key(pos), None)
            self._worn.pop(tgt, None)
            if self._maple_bound == tgt:
                self._maple_bound = ""               # 被绑的人离场了，枫叶舞可以重绑
            if self._prev_carry == tgt:
                self._prev_carry = ""
            if self._carry_piece == tgt:
                # 主C 被移除 / 变身：清掉重选（原来一直"保留主C 等看场找回"，第 42 局枫叶舞回手后没人重绑）
                logger.warning("主C %s 被 %s 移除/变身 —— 清掉主C，重新选", tgt, piece)
                self._carry_piece, self._carry = "", (660, 505)
            logger.info("%s 用在 %s 身上：%s 离场，从账本删除", piece, tgt, tgt)
            self.note("移除英雄", 牌=piece, 英雄=tgt)
        if (ok and atype == "drag_card" and not act.get("kind") and act.get("merge") and piece in self._field
                and piece not in self._ever_awakened and self._awaken_seen.get(piece) is not None
                and self._awaken_seen[piece] + 1 >= 3):
            # 同名合成凑满第 4 张 = 觉醒回手，同一个确定事件（合成回执偶尔误报成功，所以也只记"预期"）
            self.__dict__.setdefault("_awk_expect", {})[piece] = (self._round_seen, time.time())
            logger.info("%s 合成凑满（觉醒进度 %d → 满）：预期觉醒回手", piece, self._awaken_seen[piece])
        if ok and atype == "drag_card" and tgt and tgt in self._field and tgt not in self._ever_awakened:
            # **觉醒进度牌打成功、按徽章读数这一张就凑满 → 他马上觉醒回手**（2026-10-19，第 751 局 R9：弦音打给守约，
            # 守约回手要 5 秒才在手牌里认出来，期间人口读到 5/6，太乙真人先占了他空出来的格子，随后换人题又卖太乙给他腾位）。
            # 读数是"打牌前"的进度；读不到进度就不推断。只记"预期"：pop_now 读不到人口时按已离场算、decide_deploy 暂停别人上场，
            # 手牌认出他（sync_awakened / sync_field 删账本）或超时（AWK_EXPECT_S）就失效
            _m = re.search(r"觉醒进度(?:额外)?\s*\+\s*(\d)", self.pieces.get(piece, {}).get("skill") or "")
            _p = self._awaken_seen.get(tgt)
            if _m and _p is not None and _p + int(_m.group(1)) >= 3:
                self.__dict__.setdefault("_awk_expect", {})[tgt] = (self._round_seen, time.time())
                logger.info("%s 打给 %s（觉醒进度 %d → 满）：预期他觉醒回手，确认前按已离场算、别的英雄先别占他的格子",
                            piece, tgt, _p)
        if ok and atype == "drag_card" and not act.get("kind") and act.get("merge"):
            self._last_hero_played = (self._round_seen, piece)
        if ok and act.get("kind") == "效果牌" and act.get("target"):
            # "本回合对谁用过"只在**回执确认成功**后记：执行时就记的话，第一次就失败的也会被当成"用过"，
            # 再失败一次就被误学成"不能重复用"。
            self.note_used_on(self._round_seen, piece, act.get("target"))
        if ok and self.DESTROY_EQUIP_RE.search(self.pieces.get(piece, {}).get("skill") or ""):
            self._worn.pop(act.get("target", ""), None)          # 武装分解/重构：那人身上的装备没了
            self._equip_seen.pop(cell_key(tuple(act.get("dst") or (0, 0))), None)
        if (ok and atype == "drag_card" and act.get("kind") != "棋手技能牌"
                and self._hand_book[piece] > 0):
            self._hand_book[piece] -= 1          # 手牌账：牌拖出去了
        if ok:
            self._drag_fails.pop(key, None)
            # **合成的"成功"不清零尝试计数**：合成被游戏拒收时牌会弹回手里，
            # 而回执是在牌还在半空时查的 —— 看到手里少了一张就判了"成功"。
            # 2026-09-23 第 7 回合：弈星往 (1027,572) "合成成功" 15 次，手里 3 张一张没少，
            # 每次成功都把计数清零，"试够 3 次就停"永远到不了，效果牌/天赋牌整回合轮不到。
            # 真合成一回合往同一格也就一两次，3 次上限不会误伤。
            if not act.get("merge"):
                self._drag_attempts.pop((self._round_seen, piece, dst), None)
            if atype == "drag_card" and not act.get("kind") and not act.get("merge"):
                self._evict_cell(dst, piece)
                self._field[piece] = dst          # 坐实
                self._cell_owner[dst] = piece
                self._last_hero_played = (self._round_seen, piece)
                # **新上场的英雄身上没有装备**：这一格"看到过的最多件数"是上一个主人的（或误读），清掉。
                # 2026-09-24 第一轮第 5 局：(300,569) 记着 3 件，新主C 米莱狄 站进来后
                # 每件装备都被判"米莱狄装不下（0 件 / 上限 3）"，全给了坦克狂铁。
                self._equip_seen.pop(cell_key(tuple(dst)), None)
                (self.__dict__.get("_equip_hist") or {}).pop(cell_key(tuple(dst)), None)
                self._equip_rejects.pop(cell_key(tuple(dst)), None)
            return

        # ---- 失败 ----
        if (atype == "drag_card" and act.get("kind") == "效果牌" and piece
                and (self._round_seen, piece, act.get("target")) in self._used_on
                and piece not in self._no_repeat):
            # 同一回合、同一张牌、同一个英雄：上一次成功了，这一次被拒 → 这张牌"同回合不能对同一人再用"。记账（存文件，下局也有效）。
            self.note_repeat_rejected(piece)
        # **失败的上场不该占掉"这回合还能上几个"的配额**。
        # 人口读不到时 decide_deploy 退化成"每回合最多上 1 个"（保守模式），
        # 而 `_deployed_count` 在**产生决策**时就加了 1 —— 于是一次拖失败
        # 就把整个回合的上场机会吃光：2026-09-23 实战回合 6，百里玄策 拖到
        # 棋手那一格失败后，剩下十几帧全是空决策，手里压着 5 张牌、场面 3/5。
        # 同一条老教训的第 N 次：**有限次数的预算要按"事实"计数，不按"意图"**。
        if (atype == "drag_card" and not act.get("kind") and not act.get("merge")
                and self._deployed_count > 0):
            self._deployed_count -= 1
        if atype == "drag_card" and not act.get("kind") and not act.get("merge") and dst:
            # **同一回合往两个不同的格子上场都失败 → 人口多半已经满了**（2026-09-23 第十局：
            # 招募战术之类的效果把人送上了场，我们以为 6/7、其实 7/7，庄周两个回合各白拖两次）
            cells = self._deploy_fail_cells.setdefault(self._round_seen, set())
            cells.add(tuple(dst))
            if len(cells) >= 2 and self._pop_full_guess != self._round_seen:
                self._pop_full_guess = self._round_seen
                logger.warning("本回合往 %d 个不同格子上场都失败，推断人口已满，读到人口之前不再上场",
                               len(cells))
        self._drag_fails[key] = self._drag_fails.get(key, 0) + 1
        if not dst:
            return
        if not blame_cell:
            # 失败是真的，但**原因不在这个格子**——最典型的是拖的时候商店面板还开着，
            # 面板盖住棋盘上半部分，落点根本不在棋盘上。
            # 这种情况下拉黑格子是错误归因：2026-09-22 第二局把干将莫邪自己站的
            # (780,480) 拉黑了，而那是它的合成目标位。只记次数，不动格子。
            logger.warning("%s 拖到 %s 没生效，但当时商店面板开着（落点被挡），"
                           "不拉黑该格，先把商店关掉再说", piece, dst)
            return
        if act.get("merge"):
            # 合成失败**不能拉黑落点**：那个格子上站的就是要合成的目标英雄本人，
            # 拉黑等于以后再也不往自己的主力身上合成。
            # 真正的可能是：它其实已经不在那儿了（阵亡/觉醒回手/被我记错）。
            # 连续失败到阈值才撤销"它在场上"这个记录，让它重新走上场流程。
            if self._drag_fails[key] >= 2:
                logger.warning("%s 往 %s 连续合成失败 %d 次，判定它已不在该格，撤销记录重新上场",
                               piece, dst, self._drag_fails[key])
                if self._field.get(piece) == dst:
                    self._field.pop(piece, None)
                self._cell_owner.pop(dst, None)
                # **最可能的原因就是它觉醒回手了**（凑够 4 张同名 / 效果卡给觉醒阶段）：
                # 人从棋盘上消失、卡回到手牌，而我们还当它在场上、拿同名牌去合成。
                # 标记成"觉醒回手"，decide_deploy 会给 +3 优先级尽快把它重新打出去
                # ——重新打出还能**再吃一次登场效果**。
                # 2026-09-22、2026-09-23 两次出现"觉醒回手后没重新上场"。
                self._awakened.add(piece)
                self.note("判定觉醒回手", 牌=piece, 原格=f"{dst[0]},{dst[1]}",
                          依据=f"合成连续失败{self._drag_fails[key]}次")
            self.note("合成失败", 牌=piece, 落点=f"{dst[0]},{dst[1]}",
                      次数=self._drag_fails[key])
            return
        if act.get("cost_card"):
            # **卡面要付能量的牌（碰碰运气"消耗 N 能量"，N 随机 1~5）被拒，最可能是钱不够**，不是目标不在：
            # 2026-09-24 第三轮第 4 局，拖前刚点开核对过孙尚香就在 (300,569)，被拒后却撤销了她的记录。
            logger.warning("%s 拖到 %s 没生效 —— 卡面要付能量，多半是能量不够，不怪格子、不撤销记录", piece, dst)
            return
        if act.get("self_cast"):
            # **自用牌被拒 ≠ 格子有问题**（2026-09-30 轮换第 5 局：援护战术"全体【逐鹿】英雄护盾"，场上没有逐鹿英雄，
            # 游戏不让打；原来按"目标不在那儿"撤销了狄仁杰（主C）的格子记录并拉黑那一格）。
            # 多半是使用条件不满足：不动格子记录，这张牌本回合搁置。
            if piece and ("锦囊" in piece or "兑换券" in piece
                          or re.match(r"卡牌\d选1", (self.pieces.get(piece) or {}).get("skill") or "")):
                # 2026-10-19 第 779 局 R12（拿到 4 张五阶英雄兑换券只用了两张就结束回合）：兑换券 / 卡牌 N 选 1 同样没有使用条件，
                # 第二张在三选一刚关（选牌动画）时拖、被吞，原来一次失败就整回合搁置、同名的另外两张一起扣住
                # 第 712 局（2026-10-17，五阶效果锦囊一回合只用出一个）：锦囊没有使用条件，打不出去多半是手牌位置读偏 / 抓错牌；
                # 原来第一次没生效就整回合搁置，手里第二张也一起扣住。改成本回合失败 2 次才搁置（下一帧重读手牌再试）
                fk = (self._round_seen, piece)
                nf = self.__dict__.setdefault("_selfcast_fails", {})
                nf[fk] = nf.get(fk, 0) + 1
                if nf[fk] < 3:
                    logger.warning("%s 自用打出没生效（第 %d 次）—— 没有使用条件（锦囊 / 兑换券），多半是抓错牌或动画没结束，下一帧重读手牌再试", piece, nf[fk])
                    return
            logger.warning("%s 自用打出没生效 —— 多半是使用条件不满足，不怪格子、不撤销记录，本回合先留着", piece)
            self.park_card(piece)
            return
        if act.get("kind"):
            # 装备/技能牌没贴上去 = 目标英雄不在那个格子（我记错了）
            owner = self._cell_owner.get(dst) or act.get("target")
            if owner and act.get("kind") in ("装备牌", "效果牌", "天赋牌"):
                # 同一张牌在同一个人身上**跨 2 个回合**都失败 → 这局不再给他（2026-10-15 第 634 局：阿科米亚之子往马可波罗身上
                # 每回合拖两次、连续 10 回合没穿上 —— 本回合搁置后下回合又从头试）。卡面没写的穿戴限制，用失败记录学
                _cf = self.__dict__.setdefault("_card_hero_fail", {})
                # **卡面限制明明满足的不学**（2026-10-21 第 850 局：古币「指定 1 名【河洛】英雄」打守约（河洛），左下角 (300,569) 挨着手牌区偶尔失手，
                # 跨两回合各失手一次 → 守约被拉黑，之后的古币全给了铠。卡面条件成立 = 失败只能是拖拽失手，不是隐藏限制）
                _sk = (self.pieces.get(piece) or {}).get("skill") or self._card_text(piece) or ""
                _mf = re.search(r"指定\d*名【([^】]+)】英雄", _sk)
                _legal = bool(_mf) and self._faction(owner) == _R_norm(_mf.group(1))
                if _legal:
                    logger.warning("%s 打 %s 没生效，但卡面条件（【%s】）成立 → 当拖拽失手，不拉黑", piece, owner, _mf.group(1))
                else:
                    _cf.setdefault((piece, owner), set()).add(self._round_seen)
                if not _legal and len(_cf[(piece, owner)]) == 2:
                    logger.warning("%s 在 %s 身上跨两个回合都没生效 —— 这局不再给他", piece, owner)
                    self.note("牌对人无效", 牌=piece, 目标=owner)
            logger.warning("%s 没能贴到 %s 上（目标 %s 可能不在那儿），拉黑该格、记录先保留",
                           piece, dst, owner)
        else:
            logger.warning("%s 拖到 %s 没生效 —— 判定该格已被别的单位占着，拉黑换格",
                           piece, dst)
            # 我以为放上去的那个（可能就是它自己）其实没在，撤销
            if self._field.get(piece) == dst:
                self._field.pop(piece, None)
        # 无论哪种失败，这个格子都不该再用：它上面站着一个我不认识的单位。
        # （只有"连自己记的主人都对不上"才说明是格子本身无效，那种由 _bad_cells 兜底）
        self._busy_cells.add(dst)
        # **按格子记账，不分是谁拖的**。原来失败计数的键是 (牌, 落点)，
        # 于是同一个格子被三张不同的牌各撞一次，谁都到不了阈值，
        # 而下面"候选全排除就清空重来"又会把 _busy_cells 抹掉 ——
        # 结果 (433,569) 在 2026-09-23 那一局被反复选中、反复失败，
        # 公孙离/孙膑 就这么烂在手里（手牌越堆越多）。
        # 拖不进去是**直接证据**：那格有东西。这条记忆整局不清。
        self._cell_fails[dst] = self._cell_fails.get(dst, 0) + 1
        owner = self._cell_owner.get(dst)
        if owner and owner != piece:
            pass                                   # 格子归属保持，别人的地盘
        else:
            self._cell_owner.pop(dst, None)
        if self._drag_fails[key] >= 3:
            self._bad_cells.add(dst)
            logger.error("%s -> %s 连续失败 %d 次，本局彻底拉黑该落点",
                         piece, dst, self._drag_fails[key])
        # **同一张牌一个回合里不能无限重试**：每帧只允许一个变更动作，
        # 一张牌一直失败就会把整个回合的帧全吃掉，后面的牌（装备/效果牌）永远轮不到
        # —— 2026-09-22：反复使用镜、手牌里堆满装备。
        rk = (self._round_seen, piece)
        self._card_fails_round[rk] = self._card_fails_round.get(rk, 0) + 1
        if self._card_fails_round[rk] >= self.CARD_FAILS_PER_ROUND:
            self._parked_cards.add(rk)
            logger.warning("%s 本回合已失败 %d 次，先搁置它，把机会让给别的牌",
                           piece, self._card_fails_round[rk])
        self.note("拖拽失败", 牌=piece, 落点=f"{dst[0]},{dst[1]}",
                  次数=self._drag_fails[key], 处理="拉黑该格改投其他格")

    SRC_MISSING_PER_ROUND = 3   # 拖之前"抓牌校验"找不到这张牌几次，就本回合搁置它

    def note_src_missing(self, piece: str) -> None:
        """执行层拖牌前**当场核对**抓取点，没找到这张牌就不拖 —— 这也要计数。

        这是"失败够次数才停"漏掉的又一个入口（"试够次数就停"原则）：
        校验不通过 = 没拖 = 不算失败、也不算尝试，于是下一帧决策层给出**一模一样的动作**，
        整回合卡死在这一张牌上。2026-09-23 第 12 回合：手牌叠放读成「孙作刘邦」，
        刘邦 找不到，连着几十帧只有这一个动作，别的什么都没做。
        """
        k = (self._round_seen, piece)
        n = self._src_missing.get(k, 0) + 1
        self._src_missing[k] = n
        if n >= self.SRC_MISSING_PER_ROUND and k not in self._parked_cards:
            self._parked_cards.add(k)
            logger.warning("%s 连续 %d 次在手牌里找不到（抓牌校验不过），本回合搁置，先做别的",
                           piece, n)
            self.note("搁置", 牌=piece, 原因="抓牌校验连续失败", 次数=n)

    def _evict_cell(self, cell: tuple, keep: str) -> None:
        """**一格只能站一个人**：把账本里记在 `cell` 上的其他英雄清掉。

        2026-09-23 实战：弈星 和 海诺 同时记在 (433,569)（海诺是后来真的上场到那一格的），
        于是弈星把那格当成"自己的同名卡"去合成 —— 人口已满、那格也不是弈星。
        新写入的记录（拖拽回执坐实 / 点开看确认 / 观测修正）证据更直接，旧的让位。
        """
        cell = tuple(cell)
        for other, pos in list(self._field.items()):
            if other != keep and pos is not None and tuple(pos) == cell:
                self._field.pop(other, None)
                if other == self._maple_bound:
                    self._maple_lost_cell = cell      # 枫叶舞宿主被挤出账本：记下最后的格子，看场时回来找（第 214 局马超）
                logger.warning("账本里 %s 和 %s 记在同一格 %s —— 一格只能站一个，撤掉 %s 的旧记录",
                               other, keep, cell, other)
                self.note("同格冲突", 撤销=other, 保留=keep, 格=f"{cell[0]},{cell[1]}")

    def _note_equip_seen(self, state: GameState) -> None:
        """每格**看到过的最多装备件数**。装备只增不减（除非人被卖/挪走），
        而件数识别偶尔读不到 —— 单帧读数不能当成"还装得下"。"""
        hist = self.__dict__.setdefault("_equip_hist", {})
        for k, n in (state.equip_counts or {}).items():
            if n is not None and n > self._equip_seen.get(k, 0):
                self._equip_seen[k] = n
            if n is not None:
                h = hist.setdefault(k, collections.deque(maxlen=12))
                h.append(n)

    def equip_now(self, state: GameState, k: str):
        """这一格现在穿了几件：最近 12 次读到的数里**过半的那个**（读不到的帧不算）；没有过半的就用这一帧的读数；都没有返回 None。
        2026-10-13 第 617 局：马可波罗那格 68 帧里 60 帧读 3、8 帧读不到 —— 旧逻辑恰好在读不到的帧把它当 0 件，拖了雷鸣刃被拒收；
        而第 6 回合真实 2 件时有 35% 的帧误读成 3（"见过的最多件数"会一直当满）。过半数两边都对。"""
        h = (self.__dict__.get("_equip_hist") or {}).get(k)
        if h:
            n, c = collections.Counter(h).most_common(1)[0]
            if c * 2 > len(h):
                return n
        return (state.equip_counts or {}).get(k)

    def _forget_gone(self, name: str, pos, why: str) -> None:
        """单位**确实离场了**（视觉连续多帧没看到且手里没有 / 对账删掉）：和它有关的记录一起清，别留着卡住后面的动作。
        2026-09-30 审查：原来只删 `_field`，主C / 枫叶舞绑定 / 装备记录 / 该格的"已满"标记都留着 ——
        手动卖掉主C 弈星后，规则层一直"保留主C 等看场找回"，枫叶舞"绑定不明"，新主C 绑不上。"""
        if pos is not None:
            k = cell_key(tuple(pos))
            self._cell_owner.pop(tuple(pos), None)
            self._equip_seen.pop(k, None)
            (self.__dict__.get("_equip_hist") or {}).pop(k, None)
            self._equip_rejects.pop(k, None)
        self._worn.pop(name, None)
        # 枫叶舞绑定**不在这里清**：卡面亮 / 灰才是直接证据（decide_use_cards 里"卡面亮着 = 没绑上"会清）。
        # 2026-09-30 轮换第 12 局：对账按人数删了苏烈，这里顺手清掉"枫叶舞绑在苏烈"，整局"绑定不明"、换绑做不了。
        if self._prev_carry == name:
            self._prev_carry = ""
        if self._carry_piece == name and name not in self.hand_book_names():
            logger.warning("主C %s 已离场（%s）—— 清掉主C，重新选", name, why)
            self.note("主C离场", 英雄=name, 原因=why)
            self._carry_piece, self._carry = "", (660, 505)

    def note_equip_rejected(self, cell) -> None:
        """基础装备拖过去**没弹四选一**：**同一格两次**才当成那人已满 3 件。

        一次不够（2026-09-23 第八局）：拖装备那一刻棋盘上在放能量爆开的特效，
        牌根本没拖出去，蒙犽、西施就被一次各记成"满 3 件"，之后装备再也不给他们。
        没弹窗的原因不止"满了"一种，一次是弱证据。
        """
        k = cell_key(tuple(cell))
        n = self._equip_rejects.get(k, 0) + 1
        self._equip_rejects[k] = n
        if n < 2:
            logger.info("%s 那一格拖装备没弹四选一（第 1 次，可能是特效挡住/没拖出去），先不当成满的", k)
            return
        self._equip_seen[k] = max(self._equip_seen.get(k, 0), meta.MAX_EQUIPS)
        logger.warning("%s 那一格拖装备没弹四选一，按「已带满 %d 件」处理，装备改给别人",
                       k, meta.MAX_EQUIPS)

    def _equips_at(self, state: GameState, pos) -> int:
        """这一格的人身上几件装备：画面读数、看到过的最多件数、**穿戴账本**三者取大。

        2026-09-24 第二轮第 5 局（主C 已有 3 件装备，仍往主C 身上穿装备、失败重试多次）：
        账本里嬴政记着 聚灵法杖/生命魔典/纯净苍穹 三件，可判满只看画面读数和"被拒两次" ——
        画面没数出来，反伤重甲就一直往他身上拖。拖拽回执坐实的穿戴记录是直接证据，要算进去。
        """
        k = cell_key(tuple(pos))
        worn = max((len(self._worn.get(h, ())) for h, p in self._field.items()
                    if p is not None and cell_key(tuple(p)) == k), default=0)
        return max((state.equip_counts or {}).get(k, 0) or 0, self._equip_seen.get(k, 0), worn)

    def _equips_now(self, state: GameState, name: str, pos) -> int:
        """**不可逆动作（摧毁装备）用的件数**：只认这一帧的画面读数和这个英雄的穿戴账本，不用"这一格见过的最多件数"
        （2026-10-11 第 538 局：太乙真人从没穿过装备，格子的历史最多件数 2 → 武装分解拖给他，那格站的还不是他）。"""
        # 2026-10-15 第 637 局：装备件数模型把廉颇的蓝色护臂认成装备图标（第 7~11 回合几乎每帧读 1、偶尔 2），实际 0 件 →
        # 武装分解两次拖给他都被游戏拒收。摧毁装备只认穿戴账本（我们自己穿上去 / 四选一铸出的）
        return len(self._worn.get(name, ()) or ())

    def _identity_ok(self, state: GameState, name: str, pos) -> bool:
        """这一格就是 name 的直接证据：认人模型这一帧认出来，或本回合点开确认过。"""
        k = cell_key(tuple(pos))
        v = (state.unit_names or {}).get(k)
        if v and v[0] == name and (len(v) < 2 or (v[1] or 0) >= 0.5):
            return True
        return self._confirmed_at.get(name) == (self._round_seen, tuple(pos))

    def park_card(self, piece: str, round_no: int | None = None) -> None:
        """本回合不再用这张牌（Laya use_card 题选了"先留着"）。"""
        self._parked_cards.add((self._round_seen if round_no is None else round_no, piece))

    def _parked(self, piece: str) -> bool:
        """这张牌本回合是不是已经失败太多次、被搁置了。"""
        return (self._round_seen, piece) in self._parked_cards

    # ---------- 阵容解析 ----------

    def load_lineups(self, path: Path | str, lord: str = "阿离") -> None:
        """载入官网推荐阵容库（tools/fetch_lineups.py 产出），转成内部 comps。

        官网数据（data/lineups.json）给的是"英雄列表 + 推荐装备 + 分回合运营路线"，
        没有派系字段——派系要从 cards.json 反查每个英雄得到。
        只保留能用指定棋手的阵容（当前固定阿离）。
        """
        path = Path(path)
        if not path.exists():
            logger.info("没有阵容库 %s，沿用 chess_pieces.json 里的 comps", path)
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        kept = []
        for lu in data.get("lineups", []):
            if lord not in lu.get("lord_names", []):
                continue
            names = [h["name"] for h in lu.get("heroes", []) if h.get("name")]
            if not names:
                continue
            # 派系分布：禁用判定要靠它（官网数据里没有）
            factions: dict[str, int] = {}
            for n in names:
                for f in self.pieces.get(n, {}).get("synergies", []):
                    factions[f] = factions.get(f, 0) + 1
            # core 要把**分回合运营路线**里的英雄也算上：官网的 heroes 是"最终阵容"，
            # 而 stages 里的前期主力（例：回合1-3 的李元芳、弈星）常常不在最终阵容中。
            # 只用最终阵容的话，前期核心会被当成废牌上场吃效果然后卖掉
            # （2026-09-22 李元芳被卖掉，根因就在这里）。
            stage_heroes = [h for st in lu.get("stages", []) for h in st.get("main", [])]
            core = list(dict.fromkeys(names + stage_heroes))
            kept.append({
                "name": lu.get("name"),
                "key": str(lu.get("key") or ""),
                "carry": lu.get("carry"),
                "core": core,
                "final_core": names,
                # **后期阵容角色**（场上多数英雄、特别是推荐阵容里的后期阵容角色，普遍会觉醒）：
                # 最终阵容 + 最后一段运营路线的主力/替补
                "late": list(dict.fromkeys(names + [h for st in (lu.get("stages") or [])[-1:]
                                                    for h in st.get("main", []) + st.get("sub", [])])),
                "factions": factions,
                "faction": max(factions, key=factions.get) if factions else "",
                "use_num": lu.get("use_num", 0),
                "equips": {h["name"]: h.get("equips", []) for h in lu.get("heroes", [])
                           if h.get("name")},
                "stages": lu.get("stages", []),
                "equip_desc": lu.get("equip_desc", ""),
                "brief": lu.get("brief", ""),
                "effect_cards": lu.get("effect_cards") or [],
            })
        kept.sort(key=lambda c: -(c["use_num"] or 0))
        # 手动指定路线（data/custom_lineups.json）排在最后：只在 --force-route 时按 key 锁定，平时不参与选阵容
        cpath = path.parent / "custom_lineups.json"
        self.custom_comps = []
        if cpath.exists():
            for lu in json.loads(cpath.read_text(encoding="utf-8")).get("lineups", []):
                names = [h["name"] for h in lu.get("heroes", []) if h.get("name")]
                stage_heroes = [h for st in lu.get("stages", []) for h in st.get("main", [])]
                fac: dict[str, int] = {}
                for n in names:
                    for f in self.pieces.get(n, {}).get("synergies", []):
                        fac[f] = fac.get(f, 0) + 1
                self.custom_comps.append({
                    "name": lu.get("name"), "key": str(lu.get("key") or ""), "carry": lu.get("carry"),
                    "core": list(dict.fromkeys(names + stage_heroes)), "final_core": names, "late": names,
                    "factions": fac, "faction": max(fac, key=fac.get) if fac else "", "use_num": 0,
                    "equips": {n: [] for n in names}, "stages": lu.get("stages", []),
                    "equip_desc": "", "brief": lu.get("brief", ""), "effect_cards": lu.get("effect_cards") or [],
                    "custom": True})
        self.comps = kept
        # 全部阵容（不分棋手）里当主C 的热度（按使用人数，0~1）：它能不能当主C 的先验。
        # 2026-09-23 实战：场上唯一的 3 阶是姬小满（前排、技能是"卖 4 名英雄给 1 次刷新"），
        # 光凭品阶就把它选成了主C —— 它在 455 套里一次都没当过主C。
        allw = collections.Counter()
        for lu in data.get("lineups", []):
            if lu.get("carry"):
                allw[lu["carry"]] += lu.get("use_num") or 0
        mx = max(allw.values()) if allw else 1
        self._any_carry_freq = {n: (v / mx) ** 0.5 for n, v in allw.items()}
        # 本棋手阵容里，各英雄当主C 的频率（0~1）—— 动态选主C 时当一项加分
        freq = collections.Counter(c.get("carry") for c in kept if c.get("carry"))
        top = max(freq.values()) if freq else 1
        self._lord_carry_freq = {n: v / top for n, v in freq.items()}
        logger.info("载入官网阵容 %d 套（棋手=%s，按热度排序）", len(kept), lord)

    def _comp_viable(self, comp: dict, banned: str) -> bool:
        """禁用某派系后这套阵容还能不能成立。

        判据（保守）：主C 属于被禁派系 → 直接废；
        被禁派系占阵容英雄数 1/3 以上 → 也判废（核心件缺太多）。
        """
        from agent.decision.routes import norm_banned
        banned = norm_banned(banned)       # 「大河口」→「大河流域」（原来直接比字符串，大河/三分的禁用一直没生效）
        if not banned:
            return True
        carry_factions = self.pieces.get(comp.get("carry") or "", {}).get("synergies", [])
        if banned in carry_factions:
            return False
        n_banned = comp.get("factions", {}).get(banned, 0)
        return n_banned * 3 < max(1, len(comp.get("core") or []))

    COMP_SWITCH_COOLDOWN = 2   # 换完阵容几个回合内不再换

    # ---------- 官方路线（照官方推荐阵容的分回合运营路线打） ----------
    ROUTE_ENABLED = True
    ROUTE_LOCK_ROUND = 4
    ROUTE_OPENERS = frozenset({"孙膑", "干将莫邪", "曜", "牛魔", "孙尚香", "苏烈", "弈星"})
    ROUTE_EXCLUDE_CARDS = ("遗志",)          # 伞海月一族：要凑遗志、卖光全场搬等级（执行层难操作）

    def _committed_carry(self, name: str) -> bool:
        """名单外但已经当上主C（2 阶以上、在场）= 认了他（例如给到了孙悟空，就组一套孙悟空的阵容）。"""
        return bool(name) and name == self._carry_piece and name in self._field             and int((self.pieces.get(name) or {}).get("quality") or 0) >= 2

    def _committed_carry_offlist(self, name: str) -> bool:
        """名单外（carry_pool 没开着）且已经认了的主C（10-21 审查 1：路线主C / 名单内主C 不能再把他顶掉）。"""
        from agent.decision import lord_config as _LCo
        _co = _LCo.load()
        return _co.present and not _co.enabled(name) and self._committed_carry(name)

    SUPER_CORE_EXTRA = _LordRule("super_core_extra", ("明世隐",))      # 阿离：data/lords/阿离/lord_rules.json
    ENGINE_ANCHORS_LOW4 = _LordRule("engine_anchors_low4", ("明世隐", "铠", "雅典娜", "盾山"))   # 一个响指判定：路线没锁时这些引擎件在最低 4 人里也算会被删（10-22 第 929 局）

    def super_core(self) -> set:
        """超核心件（有就买，但只凑到觉醒，例如明世隐、雅典娜）：
        路线引擎表的核心件（core_set，不含主C）+ 明世隐（在路线 / 场上时）。"""
        me = self.min_engine() or {}
        out = set(me.get("early_priority") or []) - {self.route_carry(), self._carry_piece}
        out |= set(self.SUPER_CORE_EXTRA) & (self._route_team() | set(self._field))
        return out - {"", None}

    def is_awakened(self, name: str) -> bool:
        return name in self._ever_awakened or name in self._awakened

    def _route_excluded(self, comp: dict) -> bool:
        """不参选的路线：① 伞海月一族（效果牌要遗志 / 名字带遗志 / 主C 是海月 —— 要凑遗志、卖光全场搬等级，执行层难操作）；
        ② 主C 是过渡型（曜 / 赵云 / 姬小满，评分 ≤4 —— 等级高也太弱，资源堆给他性价比太低）。"""
        carry = comp.get("carry") or ""
        from agent.decision import routes as _R
        from agent.decision import lord_config as _LC
        _cfg = _LC.load()
        if _cfg.present:
            # 棋手配置（data/lords/阿离/carry_pool.json）是唯一的主C 名单：主C 关掉 / 阵容屏蔽 → 排除。
            # 官网阵容的 carry 字段是英雄列表第一个人、常标错 → 先按配置里登记的阵容编号认真主C
            real = _cfg.carry_of_lineup(comp.get("key") or "") or carry
            if _cfg.blocked_lineup(comp.get("key") or ""):
                return True
            if not _cfg.enabled(real) and not self._committed_carry(real):
                return True                  # 10-21 "给了就认"：名单外的人已经当上主C（2 阶以上）→ 他的路线可以锁
            return (any(c in (comp.get("effect_cards") or []) for c in self.ROUTE_EXCLUDE_CARDS)
                    or "遗志" in (comp.get("name") or ""))       # 遗志流（卖光全场搬等级）执行层做不了：按阵容本身排除
        if _R.CARRY_TIERS and not comp.get("custom") and (self.pieces.get(carry, {}).get("quality") or 0) not in _R.CARRY_TIERS:
            return True                     # 只走 2~3 阶主C 的路线（2026-09-29）
        if carry in _R.EXCLUDE_CARRIES:
            return True                     # 实战不走的主C（公孙离：夺取站位操作不了，2026-09-30）
        return (any(c in (comp.get("effect_cards") or []) for c in self.ROUTE_EXCLUDE_CARDS)
                or "遗志" in (comp.get("name") or "") or carry == "海月"
                or self.hero_reviewed.get(carry, {}).get("carry_score", 9) <= 4
                or (self.pieces.get(carry, {}).get("quality") or 0) == 1)   # 1 阶主C（干将无敌流：第 15 局锁死在 1 阶干将上）

    def _maybe_lock_route(self, state: GameState) -> None:
        rnd = state.round_no or 0
        owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []}
        c = self._active_comp or {}
        # 只数**非开局件**的路线成员：孙膑/干将/曜/牛魔/孙尚香/苏烈/弈星 几乎每局开局都有，拿它们当"命中"会锁到够不着的路线
        # （第 18 局：靠孙膑 + 干将锁定养马流，主C 马超 3 阶、其余 3~5 阶，2 级时一张都刷不到，整局 11 张牌、第 9 回合出局）
        real = (owned & self.route_members()) - self.ROUTE_OPENERS
        carry_in = bool(c.get("carry")) and c["carry"] in owned
        # **拿到路线主C 就从第 2 回合锁**（2026-09-29 养核心：守约路线的主C 多在第 4~6 回合才上场，
        # 官方是"第 2 回合升 2 级找守约、枫叶舞一挂直接启动"）；只凭零件的仍等到第 4 回合
        if (self.ROUTE_ENABLED and not self.route_by_laya and not self._route_locked and c and not self._route_excluded(c)
                and ((carry_in and rnd >= 2) or (rnd >= self.ROUTE_LOCK_ROUND and len(real) >= 2))):
            # 锁定条件：已拿到路线主C，或已有 ≥2 个非开局件的路线成员；不再"到回合就强制锁"
            self._route_locked = True
            self._route_key, self._route_since = self._active_comp.get("key") or "", rnd
            logger.info("锁定官方路线：%s（主C %s，成员 %s）", self._active_comp.get("name"),
                        self._active_comp.get("carry"), self.route_members())
            self.note("锁定路线", round=state.round_no, 阵容=self._active_comp.get("name"), 主C=self._active_comp.get("carry"))

    forced_backups: tuple = ()  # --force-route 逗号后面的备选路线（备选要覆盖不同主C 的阵容）

    def _switch_forced(self, key: str, state: GameState, why: str) -> None:
        from agent.decision import routes as _R
        old = _R.by_key(self.forced_route) or {}
        new = _R.by_key(key) or {}
        if "_forced_orig" not in self.__dict__:
            self._forced_orig = self.forced_route
        self.forced_route = key
        self._route_locked, self._route_key = False, ""
        self._forced_released = False
        logger.info("指定路线换备选：%s（主C %s）→ %s（主C %s）：%s", old.get("name"), old.get("carry"),
                    new.get("name"), new.get("carry"), why)
        self.note("换备选路线", round=state.round_no, 原=old.get("name"), 新=new.get("name"), 原因=why)

    def _forced_fallback(self, state: GameState) -> None:
        """首选路线打不成时换备选（首选被禁用时不能卡死）：
          ① 禁用派系把它废了（主C 被禁 / 阵容 1/3 以上被禁）→ 立刻换第一条没被废的备选；
          ② 第 5 回合起首选主C 还没拿到（场上手里都没有）、而某条备选的主C 已经在手 → 换过去（每局一次）。
        备选全废才放开，交回正常选路线。"""
        from agent.decision import routes as _R
        ban = state.banned_faction or ""
        # 主C 当不了单C（成长靠队友触发、场上喂不饱，team_feeds_carry）的路线不当备选（2026-10-01：养马流的马超）
        _subs = tuple(self.substitute_keys(ban)) if self.substitute_window(state) else ()
        cands = [k for k in (self.__dict__.get("_forced_orig") or self.forced_route,) + tuple(self.forced_backups) + _subs
                 if _R.by_key(k) and _R.viable(_R.by_key(k), ban)
                 and self.team_feeds_carry((_R.by_key(k) or {}).get("carry") or "")
                 # 主C 必须过主C 资格（评分 ≥HERO_CARRY_MIN 等）—— 2026-10-01 第 78 局：第 3 回合亚连没到、镜碰巧在手，
                 # 就换到镜的路线（镜输出太弱，打不过任何人）。原来只看"主C 在不在手"，不看强不强。
                 and (k == (self.__dict__.get("_forced_orig") or self.forced_route)
                      or self._carry_eligible((_R.by_key(k) or {}).get("carry") or ""))]
        cur = _R.by_key(self.forced_route)
        # **路线主C 已让位给现任，而现任正好是某条备选路线的主C → 整条路线跟着换过去**（2026-10-08 第 429 局，
        # 后来拿到沈梦溪，变成双主C：主C 已是 30 级守约，路线还是沈梦溪那条，1 级沈梦溪被当路线核心留在场上，
        # 配套也按沈梦溪选）。换过去之后配套 / 换人 / 卖人都按守约的阵容走，沈梦溪变成普通人
        _cc = self._carry_piece
        if self._forced_released and _cc and _cc in self._field and cur and cur.get("carry") != _cc:
            nxt = next((k for k in tuple(self.forced_backups) + ((self.__dict__.get("_forced_orig") or ""),)
                        if k and k != self.forced_route and (_R.by_key(k) or {}).get("carry") == _cc
                        and _R.viable(_R.by_key(k), ban)), None)
            if nxt:
                self._switch_forced(nxt, state, f"主C 已是 {_cc}，路线跟着换到它的阵容")
                return
        if ban and cur and not _R.viable(cur, ban):
            nxt = next((k for k in cands if k != self.forced_route), None)
            if nxt:
                self._switch_forced(nxt, state, f"禁用 {ban}")
            else:
                self._forced_blocked = True
                if self._route_locked and self._route_key == self.forced_route:
                    self._route_locked, self._route_key = False, ""
                logger.info("指定路线和备选全被禁用派系 %s 废了，本局交回正常选路线", ban)
            return
        rnd = state.round_no or 0
        # 第 3 回合起就看（原来第 5 回合）：第 45 局首选沈梦溪一直没到手，第 6 回合才切到狄仁杰，前 6 回合没有真主C、
        # 狄仁杰 1 级起步，第 5 回合全场总等级才 26 —— 备选主C 一到手就该切，主C 越早定越能吃到养成。
        # 路线可以写 carry_from_round（主C 预计到手回合，如 4 阶韩信要棋手 4 级 ≈ 第 7 回合）：之前不因"还没拿到主C"换备选
        if (rnd >= max(self.FORCED_FALLBACK_ROUND, int((cur or {}).get("carry_from_round") or 0))
                and state.phase == "prepare" and int(self.__dict__.get("_forced_switched") or 0) < 2 and cur):
            owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []}
            # 首选主C "有没有"也看手牌账：叠放时这一帧可能漏读，漏读就不可逆地换路线（2026-09-30 审查）
            # 现任主C 已经养起来（合格、≥KEEP_CARRY_LV 级）就不为"首选主C 没到手"换备选路线（第 189 局：狄仁杰 18 级时换到守约路线）
            _cc = self._carry_piece
            _cl = (self.__dict__.get("_forced_lv") or {}).get(_cc or "") or self._unit_lv.get(_cc or "") or 0
            # 1 阶、没觉醒的临时主C 不算"养起来了"（2026-10-08 第 429 局，主C 一直没换：
            # 第 5 回合李元芳 23 级挡着，场上 20 级的守约（备选路线主C）一直没换上去）
            _cq1 = (self.pieces.get(_cc or "", {}).get("quality") or 0) == 1 and _cc not in self._ever_awakened
            settled_cur = bool(_cc and _cc in self._field and self._carry_eligible(_cc) and _cl >= self.KEEP_CARRY_LV
                               and not _cq1)
            if (not settled_cur and cur.get("carry") not in owned | self.hand_book_names() and self._carry_piece != cur.get("carry")
                    and cur.get("carry") not in self._awakened):
                # **备选按"已经养着的人"优先，不按名单顺序**（2026-10-05 第 222 局，场上多个主C、等级都没起来：
                # 第 6 回合守约已在场被定为主C，同一帧换备选按名单先选了刚到手的 1 级亚连，主C 守约→亚连，两个输出位分资源）。
                # 顺序：主C 就是现任主C 的路线 > 主C 在场上（等级高的先）> 主C 只在手里。
                # **只在同体系里换**（2026-10-10 模拟：体系内谁先到用谁最好；跨体系 = 前面凑的配套作废）。
                # 同体系 = 和首选路线共用 ≥3 个配套的路线（含不在备选名单里的 狄仁杰 / 玄策）
                _fam = [k for k, _c in self.family_routes(ban, base_key=self.__dict__.get("_forced_orig") or self.forced_route)]
                # 替补主C（迟迟拿不到想要的主C 时，在商店买个高阶主C 代替并切换成配套阵容）是唯一的跨体系例外：
                # 只在替补窗口（第 5~9 回合、同体系主C 都没到、现任没养起来）里才进候选
                # 2026-10-13：路线不定死、按拿到的牌灵活切换主C → 体系外的指定路线也进候选
                # （第 622 局守约整局没到、主C 只能用米莱狄凑合；第 620 局第 8 回合才有主C）。同体系优先，
                # 再按"已经拿到的这条路线的核心件有几张"排（前面凑的不作废的那条先）
                cands = ([k for k in cands if k in _fam or k in _subs] + [k for k in _fam if k not in cands]
                         + [k for k in cands if k not in _fam and k not in _subs])
                _lvs = {**self._unit_lv, **(self.__dict__.get("_forced_lv") or {})}
                _own = set(owned) | set(self.hand_book_names())

                def _core_own(k):
                    lu = _R.by_key(k) or {}
                    return len(_own & (set(_R.final_team(lu)) - {lu.get("carry")}))

                def _rank(k):
                    c = (_R.by_key(k) or {}).get("carry")
                    return (c != self._carry_piece, c not in self._field, k not in _fam, -_core_own(k),
                            -(_lvs.get(c) or 0), cands.index(k))
                # 不是碰到就转 —— 规则兜底只在候选路线的配套已经拿到 ≥2 张时才换
                # （主C 就是现任主C 的除外）；更细的比较（首选主C 刷到的概率、两条路线还差什么、换的代价）写进题面交给模型
                # 不是碰到就转，按数值算性价比、不写死规则：
                # 按商店概率算每条路线"预计几回合凑齐核心"（route_eta），候选路线要比首选（含首选主C 还没到）明显更快才换
                _cur_eta = self.route_eta(self.forced_route, state, owned)
                nxt = next(iter(sorted((k for k in cands if k != self.forced_route
                                        and (_R.by_key(k) or {}).get("carry") in owned
                                        and ((_R.by_key(k) or {}).get("carry") == self._carry_piece
                                             or self.route_eta(k, state, owned) + 0.5 < _cur_eta)),
                                       key=_rank)), None)
                if nxt:
                    # 每局最多换 2 次（2026-09-30 第 21 局：唯一一次机会被换到公孙离路线用掉，之后亚连在场也换不过去）
                    self._forced_switched = int(self.__dict__.get("_forced_switched") or 0) + 1
                    self._switch_forced(nxt, state, f"第 {rnd} 回合还没拿到 {cur.get('carry')}，备选主C 已在手")
            # **换回首选**（2026-10-02 前三 vs 5~6 名对局分析：最终主C 是守约的 10 局全部前三；第 53/59/68/74/78/84 局守约在第 3~6 回合
            # 出现过却没拿，全是 5~6 名）：已经换到备选、首选主C 后来到手了、还在窗口期（≤ RETURN_ORIG_ROUND）、
            # 现任主C 还弱（<RETURN_ORIG_MAX_LV 级、没觉醒）→ 换回首选路线。
            elif self.forced_return_wanted(state):
                orig = self.PREFERRED_ROUTE or self.__dict__.get("_forced_orig") or ""
                self._forced_switched = int(self.__dict__.get("_forced_switched") or 0) + 1
                self._switch_forced(orig, state, f"首选主C {(_R.by_key(orig) or {}).get('carry')} 在第 {rnd} 回合到手，现任主C 还弱 → 换回首选")

    # **替补主C**（迟迟拿不到想要的主C 时，自己判断在商店买个高阶主C 代替，并切换成配套阵容）：
    # 第 SUBSTITUTE_FROM_ROUND~SUBSTITUTE_UNTIL 回合，路线主C（含同体系）没到手、现任主C 没养起来 → 商店出现替补表里
    # 推荐的高阶主C 就买，到手后 _forced_fallback 切到它的配套路线。替补表 data/carry_substitutes.json。
    # 先只上 recommend=strong 且路线查得到的（目前只有艾琳·日落海瞬移），按"主C 到手回合"分组看效果再加。
    SUBSTITUTE_FROM_ROUND = 5
    SUBSTITUTE_UNTIL = 9
    SUBSTITUTE_LEVELS = ("strong",)
    _SUBS_CACHE: Optional[list] = None

    def substitute_keys(self, ban: str = "") -> list:
        from agent.decision import routes as _R
        if RuleEngine._SUBS_CACHE is None:
            try:
                d = json.loads((Path(__file__).resolve().parents[2] / "data" / "carry_substitutes.json").read_text(encoding="utf-8"))
                d = d.get("heroes", d) if isinstance(d, dict) else {}
                RuleEngine._SUBS_CACHE = [(n, str(v.get("route_key"))) for n, v in d.items()
                                          if isinstance(v, dict) and v.get("recommend") in self.SUBSTITUTE_LEVELS
                                          and v.get("route_key")]
            except Exception:
                RuleEngine._SUBS_CACHE = []
        out = []
        for n, k in RuleEngine._SUBS_CACHE:
            lu = _R.by_key(k)
            if lu and lu.get("carry") == n and _R.viable(lu, ban) and self._carry_eligible(n):
                out.append(k)
        return out

    def substitute_window(self, state: GameState) -> bool:
        """现在该不该找替补主C：回合在窗口内、指定路线的主C（含同体系）不在场 / 手里、现任主C 没养起来。"""
        rnd = state.round_no or 0
        if not (self.forced_route and self.SUBSTITUTE_FROM_ROUND <= rnd <= self.SUBSTITUTE_UNTIL):
            return False
        if self.carry_in_hand(state):
            return False
        cc = self._carry_piece
        lv = (self.__dict__.get("_forced_lv") or {}).get(cc or "") or self._unit_lv.get(cc or "") or 0
        return not (cc and cc in self._field and self._carry_eligible(cc) and lv >= self.KEEP_CARRY_LV)

    YIELD_INVESTED_RATIO = 1.5   # 路线主C 已投入（≥2 件装备 / 绑枫叶舞）时，没觉醒的新人等级要 ≥ 这个倍数才让位（第 343 局）
    KEEP_CARRY_LV = 10          # 现任主C 合格且已养到这么多级：不为没到手 / 迟到的路线主C 换掉它（第 189 局）
    FORCED_FALLBACK_ROUND = 2   # 首选主C 没到手、备选主C 在手时，从第几回合起换备选路线（2026-10-02：3→2，主C 第 3 回合到手前三 82%、第 4 回合 53%、第 5 回合 38%）
    EARLY_CARRY_ROUND = 5       # 主C 还没到手时，这回合以内商店出现任何可走备选路线的主C 也先买
    RETURN_ORIG_ROUND = 0       # 换回首选：关（2026-10-02 第二轮复盘，见 PREFERRED_ROUTE）；原值 6
    RETURN_ORIG_MAX_LV = 10     # 现任主C 已经这么多级（或觉醒）就不换回了（养成代价）

    # **截胡路线**（2026-10-02 第二轮 12 局：首选一律守约 → 守约 12 局只拿到 1 次，前几回合按守约路线买的人白买、主C 定得晚，
    # 前三率 50%（第一轮轮换亚连 / 沈梦溪首选 64%）。改成首选照旧轮换，守约当"截胡"：窗口内到手 / 在商店就切过去 / 先买。
    # 数据依据：第 49~85 局最终主C 是守约的 10 局全部前三）。空 = 退回"换回首选"的老语义。
    PREFERRED_ROUTE = ""            # 2026-10-02 第二轮复盘：守约截胡 / 首选守约让"河洛没禁"的局从 14/21 前三掉到 4/12，关掉（守约 10/10 前三是选择偏差）

    def forced_orig_carry(self, state: GameState) -> str:
        """截胡路线（PREFERRED_ROUTE，没有就是首选路线）的主C：当前不在它上、它仍可走、还在窗口内 → 返回主C 名；否则 ""。
        preempt_shop 用它在窗口内先买。"""
        from agent.decision import routes as _R
        orig = self.PREFERRED_ROUTE or self.__dict__.get("_forced_orig") or ""
        if not orig or orig == self.forced_route or not (state.round_no or 0) <= self.RETURN_ORIG_ROUND:
            return ""
        lu = _R.by_key(orig) or {}
        if not lu or not _R.viable(lu, state.banned_faction or ""):
            return ""
        return lu.get("carry") or ""

    def forced_return_wanted(self, state: GameState) -> bool:
        oc = self.forced_orig_carry(state)
        if not oc or int(self.__dict__.get("_forced_switched") or 0) >= 2:
            return False
        owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []} | self.hand_book_names()
        if oc not in owned:
            return False
        cur = self._carry_piece
        lv = self._unit_lv.get(cur) if cur else None
        if cur and (cur in self._ever_awakened or (lv is not None and lv >= self.RETURN_ORIG_MAX_LV)):
            return False
        return True
    route_by_laya = False       # LayaPolicy 带 route 题时置 True：路线由 Laya 选（set_route），规则不再自动锁
    forced_route = ""           # live_match_agent --force-route <key>：开局就锁这条路线（手动指定，如 user_ganjiang）
    _forced_released = False    # 手动指定路线的主C 已让位给明显更强的人（本局不再强拉回）
    _forced_lv: dict = {}
    custom_comps: list = []

    def _follow_carry_route(self, state: GameState) -> None:
        """**主C 定了，路线跟着主C 走**（2026-10-10 第 524 局：先定主C、再给主C 配套英雄；路线选定后持之以恒；
        路线主C 到场即接任）：守约第 6 回合没到，亚连转正当了主C，但路线还挂在守约的河洛古币体系上 ——
        买牌 / 刷牌 / 留人都按河洛配，守约第 10 回合到了又成了第二个主C。
        现在：路线主C 让位给现任（_forced_released）后，把路线换成**以现任为主C**的阵容（官方阵容里覆盖度最高的一条），
        之后配套跟着现任走，原路线主C 不再是"要找的人"。每个主C 只换一次；找不到以他为主C 的阵容就不动。"""
        cur = self._carry_piece
        _model_pick = (self.carry_by_laya and self.model_decides()
                       and str(self._carry_switch_by).lower().startswith("laya")   # 主C 是模型选的（清查 10-12）
                       and "唯一" not in str(self._carry_switch_by))
        # 第 707 局（2026-10-17）："laya-only 唯一接班人 / 唯一合格"是合法性硬塞的（1 阶占位必须让位、场上只有一个合格的），
        # 不是模型的选择 —— 原来也算 _model_pick，路线从守约（李元芳 / 苏烈 / 铠已在场）换成了伞亚连，三个核心件一下全成了路线外
        # 2026-10-20（review-carry-route 第 2 条，第 823 局整局没锁路线）："唯一合格 / 唯一接班人"补上的主C 永远不算模型选的 →
        # 路线题只问到第 8 回合，之后再没有锁路线的入口。路线没锁时，第 8 回合起也让这类主C 带路线
        _uniq_ok = "唯一" in str(self._carry_switch_by) and (state.round_no or 0) >= self.__dict__.get("_uniq_follow_round", 8)
        if (not self.route_locked() and (_model_pick or _uniq_ok) and cur and cur in self._field
                and (self.pieces.get(cur, {}).get("quality") or 0) >= 2 and (state.round_no or 0) >= 6
                and self.__dict__.get("_followed_carry") != cur):
            # 第 693 局（2026-10-17）：路线题只给"伞亚连"（亚连没到手），Laya 每回合"先不定"；R6 模型把主C 定成马可波罗，
            # "日落海核心马可波罗"（自定义阵容）不在路线题选项里 → 整局没路线，换人网络卖了少司缘（快攻马可波罗的图腾），亚连到了又成第二个主C。
            # 主C 是模型定的（2 阶以上、第 6 回合起）→ 路线跟着主C：锁以他为主C、覆盖度最高的阵容（执行模型的主C 决定，同下面已锁路线的情形）
            from agent.decision import routes as _R
            self._followed_carry = cur
            owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []}
            from agent.decision import interim_rules as _IRw
            # 临时规则 route_avoid_weak 也管这条路（第 800 局：主C 定成安琪拉 → 自动锁了"阿离日落海安琪拉（改自日落海挂机流）"，没经过路线题的屏蔽）
            _weak = ("阿离日落海安琪拉", "双刃亚连") if _IRw.on("route_avoid_weak") else ()
            from agent.decision import lord_config as _LCw
            _cfgw = _LCw.load()
            comps = sorted((c for c in list(self.comps) + list(self.custom_comps)
                            if c.get("carry") == cur and not (self._route_excluded(c) and not c.get("custom"))
                            and not str(c.get("name") or "").startswith(_weak or ("__never__",))
                            and not (_weak and c.get("carry") in ("安琪拉", "马可波罗"))
                            and not (_cfgw.present and ((not _cfgw.enabled(_cfgw.carry_of_lineup(c.get("key") or "") or c.get("carry") or "")
                                                          and not self._committed_carry(cur))
                                                         or _cfgw.blocked_lineup(c.get("key") or "")))),     # 第 804 局：主C 安琪拉的路线不论名字一律不锁
                           key=lambda c: (-_R.fit_score(c, owned), -math.log10((c.get("use_num") or 0) + 1)))
            # 2026-10-21 撤掉 54.28-12 的兜底（"屏蔽的主C 已经是主C 就照常锁他的路线"）：棋手配置原意"关掉 = 路线不选他、主C 位不留空"，
            # 临时顶上的关掉主C 不锁路线（第 854 局 R6：马可波罗临时当主C → 锁了"日落海核心马可波罗"，整局按关掉的主C 凑阵容，第 5）
            if comps and cur in self.no_lock_outside_group(state):
                logger.info("主C %s 在开局组外、没禁河洛 → 路线先不跟他锁，等开局组主C（临时规则 no_yalian_unless_heluo_ban）", cur)
                comps = []
                self._followed_carry = ""          # 第 8 回合后还要再跟一次（不能当成已经处理过）
            if comps:
                best = comps[0]
                self._switch_forced(str(best.get("key")), state, f"主C 已定为 {cur}（路线未锁）：锁以他为主C 的路线 {best.get('name')}")
                self.set_route(self.forced_route, state.round_no or 0, why=f"主C {cur} 的阵容（路线未锁）")
            return
        _tlc = self.__dict__.get("_talent_route_carry") or ""
        if self.route_locked() and cur and _tlc and _tlc != cur and self.route_carry() == _tlc and self._group_carry_beats_talent(cur, _tlc, state):
            # 10-24 第 1010 局：幸运硬币锁狄仁杰，狄仁杰 R11 才到；沈梦溪（河洛三C）R4 起就是主C，路线却按狄仁杰凑、R11 还卖了沈梦溪（第 6）。
            # 其他河洛主C 凑合也能用——天赋主C 还没到手、开局组的河洛主C 已当上主C → 路线改跟他（古币照样吃）
            logger.info("天赋定死的路线主C %s 还没到手，开局组的 %s 已是主C → 路线改跟 %s", _tlc, cur, cur)
            self.__dict__["_talent_route_carry"] = ""
            from agent.decision import lord_config as _LCt2
            _k2 = (_LCt2.rule("preferred_route_by_carry", {}) or {}).get(cur) or _LCt2.load().lineup_key(cur)
            if _k2 and self.set_route(str(_k2), state.round_no or 0, why=f"天赋主C {_tlc} 未到，改锁开局组主C {cur} 的路线"):
                self._followed_carry = cur
                return
        if not (self.route_locked() and (self._forced_released or _model_pick) and cur and cur in self._field):
            return
        if _model_pick and cur == self.route_carry():
            return
        if _tlc and _tlc != cur and self.route_carry() == _tlc:
            # 10-23 第 973 局：幸运硬币定死狄仁杰路线，R4 主C 换成玄策 → 路线跟着换成玄策（设计：选了就定死狄仁杰，到了就换主C）
            # → 天赋定死的路线不跟现任主C 走；现任只是顶着，路线主C 到了再换
            return
        _rc0 = self.route_carry()
        if (_rc0 and _rc0 != cur and not self._forced_released
                and (_rc0 in self._field or _rc0 in {sl.get("name") for sl in state.hand_slots or []})):
            # 2026-10-20（review-interim-conflicts 第 3 条 / review-carry-route 第 3 条）：路线主C 已到手（场上或手里）时，
            # 现任主C 换成别人不改路线（路线主C 还在手里没上场时，模型换主C 会把刚锁的路线改成现任的、还绕过屏蔽）
            return
        # **1 阶临时主C 不带走路线**（2026-10-13 第 594 局：禁河洛，路线大河虞姬，临时主C 1 阶干将 → 路线跟成"干将养成"，
        # 之后买牌 / 配套全按干将走，没人再找真正的主C，1 阶主C 打满全场第 6 名）。1 阶主C 不能玩到最后
        if (self.pieces.get(cur, {}).get("quality") or 0) == 1:
            # 10-20（review-carry-route 第 4 条）：觉醒过的 1 阶也不带走路线（原来放行 → 路线成"阿离干将养成"，1 阶打到底）
            return
        if (self._active_comp or {}).get("carry") == cur or self.__dict__.get("_followed_carry") == cur:
            return
        self._followed_carry = cur
        from agent.decision import routes as _R
        banned = state.banned_faction or ""
        fam = {k for k, c in self.family_routes(banned) if c == cur}
        owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []}
        if not fam:
            # **体系外的现任主C：到所有指定路线（首选 + 备选）里找以他为主C 的那条**（2026-10-12 第 575 局：
            # 马可波罗等级已经很高，直接改走马可波罗的套路、养核心等级）。原来体系外一律"路线不跟"，
            # 主C 换成马可波罗以后还按守约凑最低配套、买守约的人。现任已经是模型定的主C，路线就该跟他走。
            keys = [self.forced_route, *list(self.forced_backups or ())]
            comps = [c for c in list(self.comps) + list(self.custom_comps)
                     if str(c.get("key")) in {str(k) for k in keys} and c.get("carry") == cur]
            if not comps:
                # 审查 10-12（A3，第 587 局：主C 安琪拉，指定路线里没有以她为主C 的 → 路线一直停在守约，按守约体系买人）：
                # 再到全部阵容（官方阿离阵容 + 自定义）里找以现任为主C、覆盖度最高的那条
                comps = sorted((c for c in list(self.comps) + list(self.custom_comps)
                                if c.get("carry") == cur and not (self._route_excluded(c) and not c.get("custom"))),
                               key=lambda c: (-_R.fit_score(c, owned), -math.log10((c.get("use_num") or 0) + 1)))
            if not comps:
                # 一条都没有：至少让原路线主C 不再被当成主C（A4：_is_carry 还认路线写死的主C → 双主C）
                self._forced_released = True
                logger.info("主C %s 没有以他为主C 的阵容 —— 原路线主C 让位，路线配套不再生效", cur)
                return
            best = comps[0]
            self._switch_forced(str(best.get("key")), state, f"主C 已定为 {cur}：换成以他为主C 的路线 {best.get('name')}")
            self._forced_switched = int(self.__dict__.get("_forced_switched") or 0) + 1
            self.set_route(self.forced_route, state.round_no or 0, why=f"主C {cur} 的阵容")
            return
        pool = [lu for lu in _R.ali_lineups() if str(lu.get("key")) in fam]
        if not pool:
            logger.info("主C %s 没有以他为主C 的可用阵容，路线不变", cur)
            return
        best = max(pool, key=lambda lu: (_R.fit_score(lu, owned), math.log10((lu.get("use_num") or 0) + 1)))
        self._switch_forced(str(best.get("key")), state, f"主C 已定为 {cur}：阵容跟着主C 走，换成 {best.get('name')}")
        self._forced_switched = int(self.__dict__.get("_forced_switched") or 0) + 1
        self.set_route(self.forced_route, state.round_no or 0, why=f"主C {cur} 的阵容")

    def set_route(self, key: str, rnd: int, why: str = "") -> bool:
        """锁定路线 `key`（Laya route 题的答案）。只查合法性：阵容存在、不在排除名单里。"""
        comp = next((c for c in list(self.comps) + list(self.custom_comps) if c.get("key") == str(key)), None)
        if not comp or (self._route_excluded(comp) and not comp.get("custom")):
            return False
        from agent.decision import lord_config as _LCs
        _cs = _LCs.load()
        if _cs.present and comp.get("custom"):
            # 自定义阵容也按棋手配置查（第 854 局：user_marco_sunset 绕过 _route_excluded，锁了关掉的马可波罗）
            real = _cs.carry_of_lineup(comp.get("key") or "") or comp.get("carry") or ""
            if _cs.blocked_lineup(comp.get("key") or "") or (not _cs.enabled(real) and not self._committed_carry(real)):
                logger.info("不锁路线 %s：主C %s 在棋手配置里关着 / 阵容屏蔽（%s）", comp.get("name"), real, why)
                return False
        self._active_comp = comp
        self._route_locked, self._route_key, self._route_since = True, comp["key"], rnd
        logger.info("锁定官方路线（%s）：%s（主C %s）", why or "Laya", comp.get("name"), comp.get("carry"))
        self.note("锁定路线", round=rnd, 阵容=comp.get("name"), 主C=comp.get("carry"), 来源=why or "Laya")
        return True

    def route_ctx(self) -> dict:
        """给 Laya 题面的路线状态（f9k）：锁定的路线 key 和回合；没锁定为空。"""
        return {"route": self._route_key, "route_since": self._route_since} if self.route_locked() else {}

    def route_members(self) -> set:
        """当前路线的全部成员：最终阵容 + 各阶段主力/替补 + **现任主C 的配套**（carry_supports）。"""
        c = self._active_comp or {}
        out = set(c.get("core") or []) | set(c.get("late") or [])
        for st in c.get("stages") or []:
            out |= set(st.get("main") or []) | set(st.get("sub") or [])
        if c.get("carry"):
            out.add(c["carry"])
        return out | set(self.carry_supports_now()[:self.CARRY_SUPPORT_KEEP])

    CARRY_SUPPORT_KEEP = 8           # 配套里前几个算路线成员（换人不卖、Laya 商店可买）
    CARRY_SUPPORT_MIN = 0.3          # 官方阵容里和这个主C 同台的加权频率 ≥ 这个才算"配套"
    STRONG_SUPPORT_FV = 8            # 卡面分析的功能价值 ≥ 这个的辅助 / 坦克 / 控制位 = 高强度辅助
    _SUPPORT_CACHE: dict = {}

    def carry_supports(self, carry: str) -> list:
        """**围绕主C 配阵容**（定一个主C 之后，找能辅助他的配套阵容，再配一些高强度的辅助英雄）。
        数据来源，不写名字：① 455 套官方阵容里主C 是他的那些，按 log 使用人数加权的同台频率 ≥ CARRY_SUPPORT_MIN（亚连 → 太乙/雅典娜/米莱狄/朵莉亚/海诺；
        守约 → 钟馗/铠/太乙/程咬金/明世隐/盾山）；② 卡牌关系图里直接给他涨级 / 回蓝 / 放大他的英雄；
        ③ 卡面分析功能价值 ≥ STRONG_SUPPORT_FV 的辅助 / 坦克 / 控制位（明世隐、太乙、张良、瑶、朵莉亚、刘禅…）。顺序 ①②③。"""
        if not carry:
            return []
        if carry in RuleEngine._SUPPORT_CACHE:
            return RuleEngine._SUPPORT_CACHE[carry]
        root = Path(__file__).resolve().parents[2] / "data"
        out: list = []
        try:
            L = json.loads((root / "lineups.json").read_text(encoding="utf-8"))
            L = L.get("lineups", L) if isinstance(L, dict) else L
            w, tot = collections.Counter(), 0.0
            for x in L:
                if x.get("carry") != carry:
                    continue
                u = math.log10(1 + (x.get("use_num") or 0))
                tot += u
                for h in x.get("heroes") or []:
                    n = h.get("name") if isinstance(h, dict) else h
                    if n and n != carry:
                        w[n] += u
            out += [n for n, v in w.most_common() if tot and v / tot >= self.CARRY_SUPPORT_MIN]
        except (OSError, ValueError):
            pass
        try:
            g = json.loads((root / "card_graph.json").read_text(encoding="utf-8"))
            for e in g.get("edges") or []:
                if (e.get("to") == carry and e.get("type") in ("gives_level", "gives_mana", "amplifies", "feeds")
                        and self.pieces.get(e.get("from"), {}).get("category") == "英雄牌" and e["from"] not in out):
                    out.append(e["from"])
        except (OSError, ValueError):
            pass
        strong = [n for n, r in (RuleEngine._HERO_ROLES or self.hero_role(carry) and RuleEngine._HERO_ROLES or {}).items()
                  if (r.get("functional_value") or 0) >= self.STRONG_SUPPORT_FV
                  and set(r.get("combat_role") or []) & {"support", "tank", "controller"} and n not in out and n != carry]
        out += sorted(strong, key=lambda n: -((RuleEngine._HERO_ROLES or {}).get(n, {}).get("functional_value") or 0))
        # **第二主C 型的人不算配套**（2026-10-08 第 429 局：加价拍花木兰，拍到就成 3 主C）：
        # 官方阵容里和守约同台的花木兰、海月、项羽是输出位 / 第二核心，算进配套后拍卖、商店、换人都把它当成"适合主C"
        def _second(n):
            hr = self.hero_role(n) or {}
            return bool(hr.get("is_second_carry_risk")) and (hr.get("carry_value") or 0) >= 5
        out = [n for n in out if not _second(n)]
        RuleEngine._SUPPORT_CACHE[carry] = out
        return out

    def carry_supports_now(self) -> list:
        """主C 定了（在场、合格）才给配套；没定主C 时返回空（别提前锁人）。"""
        c = self._carry_piece
        if not c or c not in self._field or not self._carry_eligible(c):
            return []
        return self.carry_supports(c)

    def route_locked(self) -> bool:
        return bool(self.ROUTE_ENABLED and self._route_locked and self._active_comp)

    def route_carry(self) -> str:
        if not self.route_locked():
            return ""
        from agent.decision import lord_config as _LC
        _c = _LC.load()
        key = (self._active_comp or {}).get("key") or ""
        # 官网阵容 carry 字段常标错（英雄列表第一个人）→ 有棋手配置时按配置认真主C
        return (_c.carry_of_lineup(key) if _c.present else "") or (self._active_comp or {}).get("carry") or ""

    def carry_awaken_need(self) -> list:
        """**主C 还要刷**（在这个棋手等级停留多刷几次，把主C 刷到觉醒）：路线已锁、路线主C 还没觉醒 → [路线主C]；
        路线是组、组员一个都没到 → 组员全列（谁先出用谁）。经济（engine_tier_eco / eco_roll_follow）把它当"缺的件"。"""
        if not self.route_locked():
            return []
        grp = tuple(self.__dict__.get("_route_group") or ())
        if grp and not self.__dict__.get("_group_settled") and not any(m in self._field for m in grp):
            return [m for m in grp if m not in self._ever_awakened]
        rc = self.route_carry()
        return [rc] if rc and rc not in self._ever_awakened and rc not in self._awakened else []

    def group_sync(self, state: GameState) -> None:
        """**主C 组"谁先出用谁"**（例：沈梦溪 / 守约 / 玄策一组）：路线锁的是组时，组员谁先在场（场上等级最高的那个），
        路线就换到他自己的阵容（只换这一次），之后路线主C 就是他、不再换（"开局锁定就是最终路线"）。"""
        grp = tuple(self.__dict__.get("_route_group") or ())
        if not grp or not self.route_locked() or self.__dict__.get("_group_settled"):
            return
        from agent.decision import lord_config as _LC
        _c = _LC.load()
        if self._carry_piece in grp:
            tgt = self._carry_piece
        else:
            lv = {n: ((state.unit_levels or {}).get(cell_key(tuple(p))) or 0) for n, p in self._field.items()}
            here = [m for m in grp if m in self._field]
            tgt = max(here, key=lambda m: lv.get(m, 0)) if here else ""
        if not tgt:
            return
        self._group_settled = True
        key = _c.lineup_key(tgt)
        if key and key != self._route_key:
            self.set_route(key, state.round_no or 0, why=f"主C 组里 {tgt} 先到 → 路线定为他的阵容")
        else:
            logger.info("主C 组里 %s 先到，路线不变（就是他的阵容）", tgt)

    def route_allows_hero(self, name: str, owned=()) -> bool:
        """路线锁定后能不能买 / 留这个英雄：路线成员，或已经拥有的同名（合成材料）。没锁定时一律可以。"""
        if not self.route_locked() or self.pieces.get(name, {}).get("category") != "英雄牌":
            return True
        return name in self.route_members() or name in self._field or name in set(owned)

    def resolve_comp(self, state: GameState) -> None:
        """**每回合**按手上/场上实际有的牌重新评估该打哪套阵容。

        核心问题：不能开局定死一套阵容再拿它当过滤器，
        那是"牌跟着阵容走"。自走棋是反的——**阵容要跟着牌走**：
        商店给什么就往哪套靠，中途发现另一套更顺手就该换。

        打分 = 已拥有的核心牌数 × 10 + 热度（同分时用官网使用人数兜底）。
        带迟滞：新阵容要**明显更好**（高 1 张核心牌以上）才换，否则会来回横跳。
        """
        if not self.comps:
            return
        if self.forced_route and not self.__dict__.get("_forced_blocked"):
            self._forced_fallback(state)
        if self.forced_route and not self.__dict__.get("_forced_blocked") and not self._route_locked and self.ROUTE_ENABLED:
            self.set_route(self.forced_route, state.round_no or 0, why="用户指定路线")
        if (self.forced_route and not self.__dict__.get("_forced_blocked") and self.route_locked()
                and state.phase == "prepare" and not self._forced_released
                # 2026-10-12 清查：局面网络接管换主C 时，谁当主C 只由它定（这里原来会把路线主C 直接 apply_carry、覆盖网络的选择）
                and not (self.carry_by_laya and self.model_decides())):
            # 手动指定路线：路线主C 一上场就定为主C、默认不换（一直升路线主C）；
            # **但场上出现明显更强的主C 时让位**（2026-09-29 第 1 局：本该把主C 让给司空震——
            # 第 14 回合司空震 5 阶 fit9、已觉醒、65 级，干将 38 级）。让位后不再强拉回路线主C。
            rc = self.route_carry()
            try:
                bf = self.board_facts(state)
            except Exception:                                    # noqa: BLE001
                bf = {}
            if "_forced_lv" not in self.__dict__:
                self._forced_lv = {}                      # 实例自己的一份（类属性的 {} 会被多个实例共用）
            for n, v in bf.items():
                if isinstance((v or {}).get("lv"), int):
                    # 只增不减（英雄等级不会降；2026-09-30 第 22 局：守约 91 级被读成 9，主C 让给了 15 级的武则天）
                    self._forced_lv[n] = max(v["lv"], self._forced_lv.get(n) or 0)
            lv = self._forced_lv
            best = None
            for n in self._field:
                if n in (rc, self._carry_piece) or not self._carry_eligible(n):
                    continue
                q = self.pieces.get(n, {}).get("quality", 0) or 0
                fit = (self.hero_reviewed.get(n) or {}).get("carry_score", 0) or 0
                awk = (bf.get(n) or {}).get("awk_left") == 0 or n in self._ever_awakened
                # 等级都要**读到**才比（2026-09-30 第 13 局：安琪拉等级没读到按 1 算，刚上场的 1 级海诺"明显更强"被放行、主C 让了位）；
                # 新人至少 10 级：觉醒过的要有路线主C 一半等级，没觉醒的要不低于它
                ln, lr = lv.get(n), lv.get(rc)
                # 等级都要**读到**、新人至少 10 级、等级不能更低（第 13 / 32 局）；
                # 然后看**综合分**（判断是多元的：看这个主C 的套路阵容、其他配套是否养起来、
                # 装备有没有能给他穿上的）：候选比路线主C 高出 CARRY_YIELD_MARGIN 才让位
                # **路线主C 已经投入了（≥2 件装备 / 绑着枫叶舞）→ 新人要觉醒过或等级 ≥1.5 倍才让位**（2026-10-07 第 343 局：
                # N 选 1 拿的司空震 35 级 vs 满装沈梦溪 29 级就让位，卖掉沈梦溪、3 件装备和等级白扔，第 5 名；最终不应该出现双主C）
                _rcp = self._field.get(rc) if rc else None
                invested = bool(rc and (self._maple_bound == rc
                                        or (_rcp is not None and self._equips_at(state, _rcp) >= 2)))
                if invested and not awk and ln is not None and lr is not None and ln < lr * self.YIELD_INVESTED_RATIO:
                    continue
                # 这一帧读得到两人等级时也要满足"新人不低于路线主C"（历史最高值会留下误读：第 435 局武则天被记成 15 级，
                # 实际 7 级，沈梦溪 24 级 —— 让位后又被收尾撤回，主C 来回换）
                _ln_now = (bf.get(n) or {}).get("lv")
                _lr_now = (bf.get(rc) or {}).get("lv")
                if isinstance(_ln_now, int) and isinstance(_lr_now, int) and _ln_now < _lr_now:
                    continue
                if (q >= 4 and fit >= 8 and ln is not None and lr is not None and ln >= 10 and ln >= lr):
                    sn, sr = self.carry_support_score(n, state, lv), self.carry_support_score(rc, state, lv)
                    if sn >= sr + self.CARRY_YIELD_MARGIN and (best is None or sn > best[1]):
                        best = (n, sn, sr)
            if best:
                best, sn, sr = best
                self._forced_released = True
                self._carry_switch_round = -99
                self.apply_carry(best, state, why=f"用户指定路线：{best} 综合分 {sn:.0f} 明显高于 {rc} 的 {sr:.0f}"
                                                  f"（lv{lv.get(best)} vs lv{lv.get(rc)}），让位")
            elif rc and rc in self._field and self._carry_piece != rc:
                cur, lc, lr = self._carry_piece, lv.get(self._carry_piece or ""), lv.get(rc)
                # 1 阶、没觉醒的临时主C 不在此列（2026-10-07 第 346 局：李元芳 19 级让沈梦溪"让位"，随后又卖李元芳换绑，
                # 两头都没养起来，第 6 名）。1 阶打不到后期，早卖的风险由 _transfer_victim 的 CATCHUP 推迟兜住。
                cur_q1 = (self.pieces.get(cur or "", {}).get("quality") or 0) == 1 and cur not in self._ever_awakened
                if lr is None and cur and cur in self._field and lc is not None and lc >= self.KEEP_CARRY_LV and not cur_q1:
                    # 路线主C 刚上场、等级还没读到：先别定（2026-10-08 第 384 局：虞姬"None 级"被当 0 级，
                    # 10 级的安琪拉永久占住主C，虞姬当了第二主C，第 6 名）。下一帧读到再判
                    pass
                elif (cur and cur in self._field and self._carry_eligible(cur) and lc is not None and not cur_q1
                        and lc >= max(self.KEEP_CARRY_LV, (lr or 0) + self.KEEP_CARRY_LV)):
                    # **现任主C 已经养起来了，迟到的路线主C 不顶掉它**（2026-10-04 第 189 局：狄仁杰 18 级、枫叶舞在身上，
                    # 1 级守约一到手就换主C、卖掉狄仁杰，主C 等级 18→1，第 6 名）。改成路线主C 让位给现任
                    self._forced_released = True
                    logger.info("用户指定路线：现任主C %s 已 %s 级，路线主C %s 才 %s 级 —— 不换，路线主C 让位", cur, lc, rc, lr)
                else:
                    self._carry_switch_round = -99          # 不受"本回合已换过"限制
                    self.apply_carry(rc, state, why="用户指定路线")
            elif (rc and not self._forced_released and (state.round_no or 0) > self.EARLY_FLEX_ROUND
                  and not self.carry_in_hand(state)):
                cur = self._carry_piece
                if (cur and cur != rc and cur in self._field and self._carry_eligible(cur)
                        and (self.pieces.get(cur, {}).get("quality") or 0) >= 2
                        and cur in self._sibling_carries(state.banned_faction or "")):
                    # **只有同体系的现任能转正**（2026-10-10 模拟 + 第 524 局：体系外的亚连转正 → 河洛配套白凑、双主C）。
                    # 体系外的只当临时主C，同体系主C 一到就接任；配套一直按体系买
                    # **过了前期路线主C 还没到手，就认现任**（2026-10-08 第 417 局，该直接用马可波罗当主C 却没切换：
                    # 第 6 回合马可波罗当了主C，可"现任不是路线主C → 枫叶舞先留在干将"挡着换绑收尾）。
                    # 记成路线主C 让位：换绑 / 转装备照常做，以后路线主C 来了也不再把主C 换回去
                    self._forced_released = True
                    logger.info("用户指定路线：第 %s 回合 %s 还没到手，现任主C %s 转正（路线主C 让位）",
                                state.round_no, rc, cur)
        self._follow_carry_route(state)
        if self._route_locked and self._route_key and (self._active_comp or {}).get("key") != self._route_key:
            self._active_comp = next((c for c in list(self.comps) + list(self.custom_comps)
                                      if c.get("key") == self._route_key), self._active_comp)
        self._maybe_lock_route(state)
        if self.ROUTE_ENABLED and self._route_locked and self._active_comp:
            return          # 官方路线已锁定：整局照这套打（照官方推荐阵容的分回合路线打）
        banned = state.banned_faction or ""
        owned = set(self._field) | {s["name"] for s in state.hand_slots}

        def score(comp: dict) -> float:
            core = set(comp.get("core") or [])
            hit = len(owned & core)
            trans = self._transitional_carry(comp.get("carry") or "", state.round_no)
            if comp.get("carry") in owned and not trans:
                hit += 1                      # 主C 已经在手，价值更高
            s = hit * 10 + (comp.get("use_num", 0) or 0) / 1e7
            if trans:
                # **过了前期，以过渡主C（曜这类）为核心的阵容降分**（曜前期一过就换掉）：
                # 否则阵容一直是"逐鹿(主C 曜)"，后期角色、留场判断都围着曜转
                s -= 25
            # **够不够得着**：主C 的品阶要现在（或快了）刷得出来才算数。
            # 2026-09-22 实战：开局一张核心都没命中时所有阵容都是 0 分，
            # 于是纯按热度挑，挑中了品阶5 主C 的热门阵容 —— 而那时才 1 级，
            # 商店只出品阶1，日志自己打了"⚠1级刷不出品阶5"，
            # 等于所有 carry/core 加分都指向这个阶段根本摸不到的牌。
            # 扣分按"还差几级才开这个品阶池"算，正好对应攻略的搜牌等级表；
            # 等级上去、或手上真攒到核心牌（hit*10）之后，高阶阵容自然重新胜出，
            # 也就是"前期低阶过渡 → 中期转高阶"。
            cq = self.pieces.get(comp.get("carry"), {}).get("quality", 0)
            if cq and state.level > 0 and not meta.can_appear(cq, state.level):
                s -= (meta.min_level_for(cq) - state.level) * 3
            # **阵容 = 等级引擎**（2026-09-26）：这套阵容的引擎零件我们已经有几个
            # （产出方 1 / 放大器 1.5 / 落点 1），比"命中核心牌张数"更能说明它滚不滚得起来；不沾任何引擎的阵容不加分
            ek, _ = ENG.lineup_engine(tuple(core), comp.get("carry") or "") if self.ENGINE_ENABLED else ("", 0)
            if ek:
                s += ENG.status(ek, owned)["score"] * 6
            if self.ROUTE_ENABLED:
                s += 3 * math.log10((comp.get("use_num") or 0) + 1)    # 热度：108 万人用 ≈ +18，1 万 ≈ +12（差不到 1 张核心牌）
            return s

        viable = [c for c in self.comps
                  if (self._comp_viable(c, banned) if "factions" in c
                      else c.get("faction") != banned)
                  and not (self.ROUTE_ENABLED and self._route_excluded(c))]
        if not viable:
            if self._active_comp is not None:
                logger.warning("没有阵容能避开禁用派系 %s，本局按通用规则打", banned)
            self._active_comp = None
            return

        best = max(viable, key=score)
        cur = self._active_comp
        if cur is not None and cur in viable:
            if best is cur:
                self._comp_challenger = None
                return
            # **迟滞要比"单帧波动"更硬**（2026-09-23：阵容来回切换、乱套）：
            # 这个函数每帧都跑，而"手上有哪些牌"每帧都在抖（同名卡/觉醒卡时有时无），
            # 两套阵容本来只差 1~2 张 —— 实战一局里「干将无敌流」↔「是我帮助了我」
            # 切了 3 次（同一回合里就翻了一次），主C 在干将莫邪和公孙离之间来回换，
            # 装备、上场优先级、该卖谁全跟着变，干将莫邪白买了 6 张。三道闸：
            #   ① 新的要多命中 1 张核心牌（原有）；**主C 已经在场上时要多 3 张** ——
            #      换阵容等于把主C 身上的装备、合成全扔掉
            #   ② 挑战者要**撑过一个回合**（下一回合还是它最好）才换，单帧抖动换不动
            #   ③ 换完 2 个回合内不再换
            margin = (30 if cur.get("carry") in self._field
                      and not self._transitional_carry(cur.get("carry") or "", state.round_no) else 10)
            # 引擎已经转起来（≥2 个产出方/放大器在手）且过了铺场期 → 锁定：攻略"中期后避免更换阵容"
            ek_cur, _ = ENG.lineup_engine(tuple(cur.get("core") or []), cur.get("carry") or "") if self.ENGINE_ENABLED else ("", 0)
            if state.round_no >= 5 and ek_cur and ENG.status(ek_cur, owned)["started"]:
                margin += 20
            if score(best) < score(cur) + margin:
                self._comp_challenger = None
                return
            if state.round_no - self._comp_switch_round < self.COMP_SWITCH_COOLDOWN:
                return
            name = best.get("name")
            # 观察期计的是"**当前阵容**从哪个回合起一直被明显超过"，不是"同一个挑战者持续领先"：
            # 2026-09-24 match11 两个挑战者（韩信 / 亚连）在同一回合里每帧轮流领先，原来挑战者一换名就重新计时，
            # 观察期永远走不完、永远换不掉以过渡主C 曜为核心的阵容。换人不重置起点，只更新名字。
            if not self._comp_challenger:
                self._comp_challenger = (name, state.round_no)
                logger.info("阵容「%s」比当前「%s」更顺手，观察一个回合再决定换不换",
                            name, cur.get("name"))
                return
            if self._comp_challenger[0] != name:
                self._comp_challenger = (name, self._comp_challenger[1])
            if state.round_no <= self._comp_challenger[1]:
                return                      # 还在同一个回合里，等下一回合再确认
        elif cur is best:
            return
        self._comp_challenger = None
        if cur is not None:
            self._comp_switch_round = state.round_no

        self._active_comp = best
        self.main_factions = [best["faction"]] if best.get("faction") else []
        hit = len(owned & set(best.get("core") or []))
        logger.info("选定阵容: %s（主派系 %s，禁:%s）主C=%s 已有核心牌 %d 张 热度=%s",
                    best.get("name") or best.get("faction"),
                    best.get("faction") or "-", banned or "-",
                    best.get("carry"), hit, best.get("use_num", "-"))
        if best.get("brief"):
            logger.info("  思路: %s", " ".join(best["brief"].split())[:80])
        self._maybe_lock_route(state)

    # 「指定移除一名英雄换能量」类的天赋牌（体面离场）。用卡面文本认，
    # 不写死卡名 —— 以后同类牌自动吃到。
    REMOVE_CARD_RE = re.compile(r"移除一名英雄|离场")
    TRANSFORM_TO_HAND_RE = re.compile(r"变为不在场的.*觉醒")
    TRANSFORM_HERO_RE = re.compile(r"英雄变为|变为(?:同阵营)?(?:高\d阶|\d阶)?的?(?:随机)?英雄")   # 霜寒进化/破冰裂变/寒冰王座/升华      # 化铁为金：目标离场、变成觉醒卡回到手牌
    DESTROY_EQUIP_RE = re.compile(r"摧毁其所有装备")      # 武装分解 / 武装重构：目标必须身上有装备
    REBUILD_EQUIP_RE = re.compile(r"摧毁其所有装备.{0,12}对应数量\+1的基础装备")   # 武装重构：n 件换 n+1 件基础装备（重铸）
    BUYS_PER_ROUND = 4         # 每回合最多买几张（钱要留给升级/拍卖，手牌也有上限）
    TALENT_OPEN_TRIES = 3      # 「天赋未领取」每回合最多试着打开几次
    BUYS_PER_ROUND_MAXED = 8   # 满级后：钱已经不用留给升级了（2026-09-23 满级攒到 36 能量花不出去）

    def _buy_cap(self, state: GameState) -> int:
        if getattr(self, "shop_by_laya", False):
            # 商店交给 Laya 时不按张数卡（2026-10-09 去掉硬性限制）：买不买、买几张由 Laya 每次局面变了时判断，
            # 钱留给升级由 level_budget 管。这里只留防 bug 的上限
            return 30
        maxed = state.lord_maxed or state.level >= meta.MAX_LORD_LEVEL
        if maxed or self._in_roll_window(state):
            return self.BUYS_PER_ROUND_MAXED     # 满级 / 搜牌回合：这回合本来就是来凑人的，4 张不够
        return self.BUYS_PER_ROUND
    CARD_FAILS_PER_ROUND = 2   # 一张牌一回合最多失败几次，超了本回合搁置
    LEVEL_WAIT_FRAMES = 3      # 每回合为等弈星上场最多让位几帧，超了就照常升级

    def _carry_quality(self) -> int:
        """运营按哪个品阶的主C 来走（决定"升到几级停下来 D 牌"）。0 = 未知，各判据据此放行。

        **至少按 3 阶算**（2026-09-23 学完推荐阵容后改）：
        推荐阵容里 1 阶当主C 的占 0%，1 阶只是过渡。原来直接拿阵容写死的主C 的品阶 ——
        开局阵容是 1 阶的曜，于是"2 级就进搜牌期"，整局压着不升：第 4 回合还是 1 级、
        第 10 回合才 4 级（那局第 5 名）。阿离守约线的原文也是"二回合升二级，上三级找全阵容"。
        """
        names = [(self._active_comp or {}).get("carry"), self._carry_piece]
        qs = [self.pieces.get(n, {}).get("quality", 0) for n in names if n]
        q = max(qs) if qs else 0
        return max(q, 3) if q else 0

    def _in_search_phase(self, state: GameState) -> bool:
        """是不是已经进入"停止升本、集中 D 牌"的阶段（攻略：升本与搜牌的时机选择）。

        判据是**主力品阶对应的搜牌等级**：低品阶核心 2 级就 100% 能刷到，
        再升级只会让高品阶稀释牌池；高品阶核心必须先把池子开出来才值得搜。
        """
        q = self._carry_quality()
        if not q or state.level <= 0:
            return False
        return state.level >= meta.search_level(q)

    def _auction_hold(self, state: GameState) -> int:
        """拍卖前要压着不花的能量。实测：第 4/8/12 回合开头拍卖（上一回合战斗后），此前须留钱。

        关键是**别过度省**：第 4/8/12 回合的拍卖在上一回合战斗之后、本回合准备之前，而每回合开始就发能量
        （攻略给的阶梯 1-3 回合 6 点、4-7 回合 8 点、8-11 回合 12 点、12+ 15 点）。
        所以提前一回合压钱基本没必要——第 4 回合开拍前先进账 8 点，本身就盖过
        出价上限 5 了。只有当**下回合收入仍不够出价**时才需要现在就省。

        恒定底线是错的（那会让每个回合都白白压着钱），这也是 auction_reserve
        不能拿来当通用 reserve 的原因。
        """
        left = meta.rounds_to_auction(state.round_no)
        if left == 0:
            # 2026-09-25 实测：拍卖在本回合准备阶段**之前**就拍完了（原来这里留 5 能量是白压着）
            return 0
        if left == 1:                            # 本回合战斗后就拍；第 A 回合收入在拍卖前到账
            gap = self.auction_max_bid - meta.energy_income(state.round_no + 1)
            return max(0, gap)                   # 下回合收入补不齐的部分才提前省
        return 0

    def _is_carry(self, piece: str) -> bool:
        """是不是主C。**跟"当前主C"走**（choose_carry 每回合动态选），还没选出来时才用阵容里写死的。

        原来只认阵容写死的 carry —— 开局选中的阵容主C 是 1 阶的曜，就整局把装备、枫叶舞、
        加等级牌全砸在曜身上（没有切换主C）。"""
        if self._carry_piece and piece == self._carry_piece:
            return True
        if self._forced_released and self._carry_piece:
            return False        # 路线主C 已让位：只认现任主C 一个（2026-10-02 第 109 局：亚连 / 鲁班七号 都被当主C）
        if self._carry_piece and self.carry_by_laya and self.model_decides():
            return False        # 审查 10-12（A4）：主C 由局面网络定时只认现任一个（第 587 局安琪拉 / 守约同时"分100/主C"）
        # 阵容里写死的主C：照样是想买的目标、也受"不卖"的保护 ——
        # **除非它是过了第 6 回合的 1 阶**（不然开局阵容定了 1 阶曜，就整局护着曜、卖不掉）。
        comp_carry = self._active_comp.get("carry") if self._active_comp else None
        # 阵容写死的主C 也要是**合格的输出位**（和 choose_carry 同一套判据：不是辅助/坦克、阵容里真当过主C）
        return (bool(comp_carry) and piece == comp_carry and not self._obsolete(piece, self._round_seen)
                and not self._transitional_carry(piece, self._round_seen) and self._carry_eligible(piece))

    TRANSITION_CARRY_MAX = 4         # 主C 评分 ≤ 这个的只配当前期过渡主C（曜、赵云、姬小满）

    def _transitional_carry(self, name: str, round_no: int) -> bool:
        """**过渡主C 过了前期就不当主C**，觉醒了、等级高也一样（2026-09-24 match9：
        曜不值得作为主C，前期一过就换掉；等级高也太弱，资源堆给他性价比太低）。
        评分表里曜 4 分（"前期过渡主C"）—— 之前觉醒卡不算过时，于是觉醒的 40 级曜整局当主C、吃掉所有资源。"""
        hr = self.hero_reviewed.get(name)
        return (round_no > self.LOW_TIER_CUTOFF_ROUND and hr is not None
                and hr.get("carry_score", 5) <= self.TRANSITION_CARRY_MAX)

    EARLY_FLEX_ROUND = 5             # 这回合以内主C 是"临时"的：拿到什么用什么（不卡 1 阶、不卡强度分）

    def _flex_eligible(self, name: str) -> bool:
        """按"前期临时主C"的标准（不卡品阶 / 强度分）看能不能当主C —— 用来判断过了前期还能不能续用。"""
        saved = self._round_seen
        try:
            self._round_seen = self.EARLY_FLEX_ROUND
            return self._carry_eligible(name)
        finally:
            self._round_seen = saved
    HERO_CARRY_MIN = 4               # 英雄主C 评分（data/hero_carry_reviewed.json）至少这么多才能当主C
    _HERO_REVIEWED: Optional[dict] = None

    @property
    def hero_reviewed(self) -> dict:
        """主 agent 逐个读英雄技能（含 10/40/100 级强化）后的"适不适合当主C"评分（tools/cards/reviewed_heroes.py）。"""
        if RuleEngine._HERO_REVIEWED is None:
            try:
                RuleEngine._HERO_REVIEWED = json.loads(
                    (Path(__file__).resolve().parents[2] / "data" / "hero_carry_reviewed.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                RuleEngine._HERO_REVIEWED = {}
        return RuleEngine._HERO_REVIEWED

    _STRENGTH = None
    _LORD_OVERRIDES = None
    LORD = "阿离"                 # 我们打的棋手（强度表的按棋手修正用）
    STRENGTH_CARRY_MIN = 5.5        # 输出模型分 ≥ 这个才能当主C
    STRENGTH_ROUTE_MIN = 4.0        # 4~5.5：只当路线指定的主C
    STRENGTH_HIST_TOP3 = 0.30       # 历史前三率 ≥30%（≥5 局）的不受模型分限制（亚连：模型 4.2，实战 27 局前三 37%）

    def carry_strength_ok(self, name: str) -> bool:
        """**主C 强度审查**（data/hero_strength.json，tools/check_hero_strength.py：按技能原文估各等级有效伤害 × 存活 × 等级可兑现度，
        147 局历史名次校准）。例如镜输出太弱、不能当主C；强度本身必须审查。
        模型和实战名次的相关只有 0.40，所以：实战数据够（≥5 局、前三率 ≥30%）的以实战为准；否则 <4 不当主C、4~5.5 只当路线指定的主C。
        只在实战开关打开时生效（离线题面不变）。"""
        from agent.decision import hero_traits as _ht
        if not _ht.USE_HERO_ITEM_PREF or not name:
            return True
        if RuleEngine._STRENGTH is None:
            try:
                import json as _json
                RuleEngine._STRENGTH = (_json.loads((Path(__file__).resolve().parents[2] / "data" / "hero_strength.json")
                                                   .read_text(encoding="utf-8")).get("heroes") or {})
            except (OSError, ValueError):
                RuleEngine._STRENGTH = {}
        if RuleEngine._LORD_OVERRIDES is None:
            try:
                import json as _json
                _d = _json.loads((Path(__file__).resolve().parents[2] / "data" / "hero_strength.json").read_text(encoding="utf-8"))
                RuleEngine._LORD_OVERRIDES = {k: v for k, v in ((_d.get("lord_overrides") or {}).get(self.LORD) or {}).items()
                                              if isinstance(v, dict)}
            except (OSError, ValueError):
                RuleEngine._LORD_OVERRIDES = {}
        h = RuleEngine._STRENGTH.get(name)
        if not h or h.get("score") is None:
            return True
        sc = float(h["score"])
        ov = (RuleEngine._LORD_OVERRIDES or {}).get(name)
        if ov and ov.get("score") is not None:
            sc = float(ov["score"])           # 按棋手的审定修正（hero_strength.json lord_overrides：模型算不到棋手机制的叠加，如阿离 × 海月）
        hist = h.get("history") or {}
        if (hist.get("games") or 0) >= 5 and (hist.get("top3") or 0) / hist["games"] >= self.STRENGTH_HIST_TOP3:
            return True
        if sc >= self.STRENGTH_CARRY_MIN:
            return True
        is_route = name == (self._active_comp or {}).get("carry")
        if is_route and (self._active_comp or {}).get("custom"):
            return True       # 手动指定的路线主C（伞海月：强度模型只算技能 dps，按生命值造成的伤害它算不到，2026-10-14）
        return is_route and sc >= self.STRENGTH_ROUTE_MIN

    def team_feeds_carry(self, name: str) -> bool:
        """能不能当**单C**（阿离的打法：枫叶舞 / 搬等级 / 加等级牌 / 装备都喂给一个人）。
        依据 data/equip_relations.json 的 solo_carry_ok / team_dependency / provides（逐个读英雄原文推导，不写英雄名）：
        solo_carry_ok=false 的输出位（成长主要靠"己方/队友做某事"触发，如马超"己方施加控制 → 法力与暴击"）只有在
        **场上已经有足够多的单位提供这种触发**时才当主C（按 needs 的 need_count 数 provides）。
        例如马超不适合单C 阵容：不写死名字，按机制判断为什么阿离（单C）用不了。
        只在实战开关（hero_traits.USE_HERO_ITEM_PREF）打开时生效，离线题面不变。坦克/辅助不在这里判（另有规则）。"""
        from agent.decision import hero_traits as _ht
        if not _ht.USE_HERO_ITEM_PREF:
            return True
        H = (_ht.relations().get("heroes") or {})
        h = H.get(name) or {}
        if h.get("solo_carry_ok") is not False or h.get("role") in ("tank", "support"):
            return True
        deps = [d for d in (h.get("team_dependency") or []) if not d.get("universal") and d.get("need_count")]
        if not deps:
            return False
        for d in deps:
            have = sum(1 for n in self._field if n != name
                       and any(p.get("tag") == d.get("kind") for p in (H.get(n) or {}).get("provides") or []))
            if have < int(d["need_count"]):
                return False
        return True

    def _carry_eligible(self, name: str, hard_only: bool = False) -> bool:
        """能不能当主C（**不调用任何别的主C 判断**，免得递归）：英雄牌、输出位。

        - 辅助/坦克不当主C（第九局 1 级辅助明世隐被选成了主C）；
        - 推荐阵容里几乎从不当主C、又没有输出职业标签的不当（第十一局：鲁班大师 5 阶 57 级被选成主C ——
          它在 455 套阵容里当主C 的频率 0.9%，卡库里只有「前排」没有职业；蒙犽 2.2% 同理）。
          卡库有 25 个英雄缺职业标签，光看标签判不了输出位，阵容数据更可靠。
        """
        # （2026-10-20 曾在这里全局挡安琪拉 / 马可波罗当主C → 只剩她 / 他能当时主C 一直空着（第 827 局）；已撤，改为只在换主C 题有别的人选时去掉）
        info = self.pieces.get(name, {})
        tags = info.get("tags") or []
        if info.get("category") != "英雄牌":
            return False
        from agent.decision import lord_config as _LCe
        if _LCe.load().present and _LCe.load().enabled(name):
            return True                     # 棋手配置开着的主C 一律可当（刘邦卡库标签"前排 / 坦克"，旧判据会挡掉；以 10-20 晚的配置名单为准）
        from agent.decision import routes as _R
        if name in _R.EXCLUDE_CARRIES:
            return False                    # 实战不当主C（公孙离：夺取站位操作不了，2026-09-30）
        if hard_only and not _IR_on("carry_value_filter"):
            # A1（2026-10-15：冲突的规则都关掉）：模型选主C 时只保留**明确定下的硬约束**，其余（审定评分 / 队友喂不饱 / 阵容频率）写在题面。
            # 第 642 局教训：连"弈星不当主C"一起关了 → 开局场上只有弈星，被定成主C、枫叶舞绑上去（枫叶舞不该挂给弈星）
            early_h = 0 < (self._round_seen or 0) <= self.EARLY_FLEX_ROUND
            if self._is_level_engine(name):
                return False                # 弈星这类棋手经验件（09-22）
            if self._is_support(name) or "坦克" in tags:
                return False                # 辅助 / 坦克（不能把枫叶舞给辅助，09-23）
            if (info.get("quality") or 0) == 1 and not early_h and name != (self._active_comp or {}).get("carry"):
                return False                # 过了前期 1 阶（1 阶主C 不能玩到最后，10-02）
            return True
        # **前期拿到什么用什么**（2026-10-02：主C 不定死，棋手 3 级左右时拿到什么用什么，但 1 阶主C 不能玩到最后）：
        # 第 EARLY_FLEX_ROUND 回合以内是"临时主C"阶段 —— 不卡 1 阶、不卡强度模型分（只排坦克/辅助/经验件/靠队友才长的）；过了前期 1 阶自动不合格、会被换掉。
        early = 0 < (self._round_seen or 0) <= self.EARLY_FLEX_ROUND
        if early and self._is_level_engine(name):
            return False                    # 弈星这类棋手经验件升完级就卖，不当临时主C
        if (not hard_only and early and name != self._carry_piece and name not in self._ever_awakened
                and self.keep_as_support(name) and self._nonsupport_temp_exists(name)):
            return False                    # 之后是配套（路线成员 / 喂主C 等级）的人不当临时主C（2026-10-11：李元芳当了主C 还要卖掉再买）
        if (_R.CARRY_TIERS and (info.get("quality") or 0) == 1 and not early
                and name != (self._active_comp or {}).get("carry")):
            # 2~3 阶定调时 1 阶不当主C（2026-09-30 轮换第 5 局：开局只有弈星 → 弈星当了主C、枫叶舞绑上去，
            # 弈星升完级就卖，枫叶舞从此"绑定不明"；路线自己指定的 1 阶主C（干将路线）除外）
            return False
        if not hard_only and not early and not self.carry_strength_ok(name):
            # hard_only（审查 10-12 B3：模型选主C 时）：强度审查、"之后是配套"是价值判断，写在题面里（f9lk carry_why_not），不当合法性删候选
            return False                    # 输出强度审查没过（data/hero_strength.json）
        if not self.team_feeds_carry(name):
            return False                    # 成长靠队友触发、而场上喂不饱（马超：己方施加控制 → 法力/暴击；阿离是单C，资源只能喂等级和装备）
        hr = self.hero_reviewed.get(name)
        if hr is not None:
            # **主 agent 读技能后的评分优先**（读英雄牌技能说明后的评分）：≥4 分才当主C。
            # 例：蒙犽在推荐阵容里当主C 的频率只有 2.2%、原来被排除，但它是 8→11 枚导弹的群体射手（评 8 分）。
            return hr.get("carry_score", 0) >= self.HERO_CARRY_MIN
        if self._is_support(name) or "坦克" in tags:
            return False
        if (self._any_carry_freq and self._any_carry_freq.get(name, 0.0) < 0.05
                and not any(t in tags for t in ("射手", "法师", "刺客", "战士"))):
            return False
        return True

    # **作用于"最近友军"的英雄要贴着主C**（太乙真人和瑶机制特殊：要考虑摆放位置、要不要给装备）：
    #   太乙真人  开团：使距离最近的 1 名英雄获得复生（官方阵容："太乙复活给守约"）
    #   瑶        备战阶段选最近的友方英雄作为保护目标，开战附身（护盾 + 双攻双防，100 级分享 30% 属性）
    #   明世隐    备战阶段以法器连接 1 名最近的友方英雄（本来就按辅助贴主C）
    # 按卡面文字识别，不写死名字。
    CARRY_BUDDY_RE = re.compile(r"最近的友方英雄|距离最近的1名英雄获得|连接1名最近的友方英雄|距离最近的【[^】]+】英雄|距离最近的英雄")
    # 装备：太乙真人官方 180 套阵容里 141 套（78%）不给 —— 它开团就冲进敌方后排，价值在给主C 的复生；
    # 瑶"开团：自身的首件普通装备分享给随机 3 名友军"—— 官方最常见只给一件破茧之衣（开战眩晕敌方高等级英雄，分享后多人各放一次）。
    NO_EQUIP_UNITS = ("太乙真人",)
    # 临时规则 carry_late_defense（2026-10-21 第 891 局：主C 前两件都是输出、到后期还缺一件时，可以选一件防御装，
    # 但只限特殊装备里不降低攻速的防御装，或血量高于阈值时减伤的那件（霸者重铠）；否则优先等输出或免控装）
    # 名单绑定棋手（data/lords/阿离/lord_rules.json carry_late_def / carry_any_def / carry_late_def_base；下面是兜底值）
    CARRY_LATE_DEF = _LordRule("carry_late_def", ("护命灵珠", "翼苍穹盾", "苍穹王权", "霸者重铠", "磐石之心"))
    CARRY_ANY_DEF = _LordRule("carry_any_def", ("磐石之心",))      # 磐石之心任何主C 都能带，防秒杀很强——不看远近、不看回合（10-21 第 892 局起）
    CARRY_LATE_DEF_BASE = _LordRule("carry_late_def_base", {"力量腰带": "霸者重铠"})

    def carry_late_def_ok(self, state) -> bool:
        """第 12 回合起、主C 在场、身上正好 2 件且都是输出装（不是辅助 / 防御）→ 第三格可以给一件不降攻速的防御装。"""
        from agent.decision import interim_rules as _IRd, hero_traits as _htd
        c = self._carry_piece
        rnd = (getattr(state, "round_no", None) if state is not None else None) or self._round_seen or 0
        if not (_IRd.on("carry_late_defense") and c and c in self._field):
            return False
        worn = list(self._worn.get(c, ()))
        if _htd.is_melee(c):
            # 近战主C 可以带一件防御装 —— 近战主C 不等后期：已有 ≥1 件输出、身上还没有防御装、没满
            return (len(worn) < 3 and any(_htd.utility_rank(w) == 3 for w in worn)
                    and not any(w in self.CARRY_LATE_DEF for w in worn))
        return rnd >= 12 and len(worn) == 2 and all(_htd.utility_rank(w) == 3 for w in worn)
    SPEED_LOCK_CARRIES = ("花木兰", "嬴政")   # 机器头只给这两位输出（攻速锁定机制特殊，一般只有花木兰和嬴政作主C 才选）
    SHARE_EQUIP = {"瑶": ("破茧之衣", "破碎圣杯")}      # 破碎圣杯 四选一能铸出破茧之衣

    _NORMAL_EQUIPS: Optional[set] = None

    @classmethod
    def _is_normal_equip(cls, card: str) -> bool:
        """「普通装备」= 8 件基础装备（穿上铸造成普通装备）+ 32 件四选一成品（data/equip_recipes.json）。
        末世 / 盛华 这类 3~5 阶特殊装备不算 —— 瑶只分享"首件**普通**装备"（所以要给瑶一件普通装备）。"""
        if cls._NORMAL_EQUIPS is None:
            try:
                rec = json.loads((Path(__file__).resolve().parents[2] / "data" / "equip_recipes.json")
                                 .read_text(encoding="utf-8")).get("recipes") or {}
                cls._NORMAL_EQUIPS = set(rec) | {x for v in rec.values() for x in v}
            except Exception:                              # noqa: BLE001
                cls._NORMAL_EQUIPS = set()
        return card in cls._NORMAL_EQUIPS

    _SKILL_TEXT: Optional[dict] = None

    AURA_RE = re.compile(r"(?:自身)?(?:周围|相邻)\s*\d*\s*格?的?(?:友军|友方)")

    def _aura_near(self, name: str) -> bool:
        """卡面 / 技能写着"周围 N 格的友军""相邻友军"的光环（蔡文姬：琴音给周围 1 格友军回血）—— 要站在主C 旁边才照顾得到主C。
        和 _carry_buddy（开战选最近友军）不同：不存在"连错人"的问题，挨着就行。"""
        return bool(self.AURA_RE.search(self._card_text(name)))

    def _wants_carry_side(self, name: str) -> bool:
        """**需要站主C 旁边**：只看卡面（只有个别英雄必须站在主C 旁边，读卡牌文本判断）。"""
        return self._carry_buddy(name) or self._aura_near(name)

    def _carry_buddy(self, name: str) -> bool:
        """卡面 + **英雄技能**文本里有"最近的友方英雄 / 距离最近的1名英雄获得"。
        技能（skills）不在 self.pieces 里（瑶的"选最近友军附身"只写在技能里），直接读卡牌库。
        只找某派系的（李元芳"距离最近的【河洛】英雄"）：主C 不是这个派系时不算（2026-10-11 站位表：主C 非河洛时它仍占着主C 邻格）。"""
        if not self.CARRY_BUDDY_RE.search(self._card_text(name)):
            return False
        fac = self._buddy_faction(name)
        if fac and self._carry_piece:
            cf = self.pieces.get(self._carry_piece, {})
            if fac not in (cf.get("synergies") or []) and fac != (cf.get("faction") or ""):
                return False
        return True

    _BATTLE_START_NEAREST = re.compile(r"开团[:：][^。]*最近")

    def _link_sticky(self, name: str) -> bool:
        """放下时定连接、之后挪动只有主C 严格最近才改连的（备战阶段选人：明世隐"连接"、瑶"选择最近的友方英雄作为保护目标"；
        李元芳 2026-10-08 实测要挪开再挪回才刷新）。**开团时现算**的（太乙"开团：使距离最近的1名英雄获得复生"）不保持，
        中转格对它无效、最终格上主C 必须唯一最近（2026-10-11 站位表）。"""
        return not self._BATTLE_START_NEAREST.search(self._card_text(name) or "")

    AURA_N_RE = re.compile(r"(?:周围|相邻)\s*(\d*)\s*格?的?(?:友军|友方)")

    def aura_radius(self, name: str) -> int:
        m = self.AURA_N_RE.search(self._card_text(name) or "")
        return int(m.group(1)) if m and m.group(1) else 1

    SACRIFICE_RE = re.compile(r"(?:^|\s)牺牲[:：]")

    def _is_sacrifice(self, name: str) -> bool:
        """自带「牺牲：」效果的英雄（铠/苏烈/盾山…）—— 站在钟馗左右才吃到钟馗给的牺牲，阵亡时产古币 / 临时等级。"""
        return name != "钟馗" and bool(self.SACRIFICE_RE.search(self.pieces.get(name, {}).get("skill") or ""))

    @staticmethod
    def _row_sides(cell) -> list:
        """同一行左右相邻的格子（钟馗"自身及左右 1 格"只算同一行）。"""
        from ..perception.extractors import OWN_BOARD_ROWS
        for row in OWN_BOARD_ROWS:
            row = [tuple(c) for c in row]
            if tuple(cell) in row:
                i = row.index(tuple(cell))
                return [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
        return []

    LEVEL_FOCUS_CARDS = ("众星拱月", "红叶染锋")

    def route_wanted_cards(self) -> set:
        """路线锁定后"一出现就要"的牌：路线推荐效果牌（阵容库 effect_cards，如破境之证）+ 这套阵容所属等级引擎的用牌
        （engines.json cards，如 自我奉献 / 四海来援 / 古币多多）。2026-09-29：等级引擎没转起来的补件之一。"""
        # **集中等级的牌不管走哪条路线都要**（2026-09-29 网上攻略：阿离单核养成靠 众星拱月 把全队等级集中给主C、
        # 红叶染锋 换主C 时带走一半等级 —— 冲 999 的关键牌）
        out = set(self.LEVEL_FOCUS_CARDS)
        if not self.route_locked():
            return out
        from agent.decision import routes as _R
        lu = _R.by_key(self._route_key) or {}
        out |= set(lu.get("effect_cards") or [])
        out |= set(self.route_addons())                   # 顺带养 T 的补件（效果牌 / 天赋 / 英雄）
        try:
            ek, _sc = ENG.lineup_engine(tuple(_R.final_team(lu)), lu.get("carry") or "")
            if ek:
                out |= set(ENG.engines().get(ek, {}).get("cards") or [])
        except Exception:                                     # noqa: BLE001
            pass
        return {c for c in out if c}

    _CARD_Q = None

    def _card_quality(self, name: str) -> int:
        """卡牌库里这张卡的品阶（装备：基础 1 / 四选一成品 2 / 星源等特殊 3 / 天赋专属 5）。"""
        if RuleEngine._CARD_Q is None:
            try:
                cards = json.loads((Path(__file__).resolve().parents[2] / "data" / "cards.json")
                                   .read_text(encoding="utf-8"))["cards"]
                RuleEngine._CARD_Q = {k: int(str(v.get("quality") or 0) or 0) for k, v in cards.items()}
            except (OSError, ValueError, KeyError):
                RuleEngine._CARD_Q = {}
        return RuleEngine._CARD_Q.get(name, 0)

    def _card_text(self, name: str) -> str:
        """卡牌库里这张卡的卡面 + 技能原文（self.pieces 不含装备/技能全文）。"""
        if RuleEngine._SKILL_TEXT is None:
            try:
                cards = json.loads((Path(__file__).resolve().parents[2] / "data" / "cards.json")
                                   .read_text(encoding="utf-8"))["cards"]
                RuleEngine._SKILL_TEXT = {k: (v.get("skill") or "") + " " + " ".join(
                    sk.get("desc", "") for sk in (v.get("skills") or []))
                    + (f" 【送出的同名牌】{v['given_skill']}" if v.get("given_skill") else "")      # 2026-10-13 爬虫补的同名赠送牌
                    for k, v in cards.items()}
            except (OSError, ValueError, KeyError):
                RuleEngine._SKILL_TEXT = {}
        t = RuleEngine._SKILL_TEXT.get(name, "")
        if name == "云中君" and "神鹰锻匠" in (self.__dict__.get("_talents_taken") or ()):
            # 天赋神鹰锻匠：云中君开团给「距离最近的1个【大河】英雄」加神兵 → 按派系贴主C 处理（2026-10-20 摆位已支持）
            t += " 开团：为距离最近的【大河流域】英雄添加神兵装备"
        return t

    def _no_gear(self, piece: str) -> bool:
        """不给装备的辅助（除瑶之外辅助大都没必要给装备，装备优先给坦克和战士）。
        瑶（SHARE_EQUIP：首件普通装备开团分享给 3 名友军）例外。"""
        return self._is_support(piece) and piece not in self.SHARE_EQUIP

    def _is_support(self, piece: str) -> bool:
        """辅助英雄（卡库标签「辅助」：明世隐/蔡文姬/大乔…）。
        辅助基本没有输出能力，尽量放在主C 边上。"""
        hr = self.hero_role(piece)
        if hr:
            return "support" in (hr.get("combat_role") or [])[:1]   # 按卡面文本分析的主定位（data/hero_roles.json）
        return "辅助" in (self.pieces.get(piece, {}).get("tags") or [])     # 临时兜底：hero_roles 缺项

    # ---------- 等级引擎（agent/decision/engines.py） ----------

    ENGINE_ENABLED = False     # 2026-09-26 晚：引擎规则第 15 局锁死在 1 阶干将上（第 5 名），先关掉，等题面/规则理顺再开

    def engine_key(self) -> str:
        """当前目标引擎：阵容定了就用阵容的引擎；没定阵容按场上+手里最成型的那套。开关关着时返回空串（所有引擎逻辑失效）。"""
        if not self.ENGINE_ENABLED:
            return ""
        comp = self._active_comp or {}
        if comp:
            ek, _ = ENG.lineup_engine(tuple(comp.get("core") or []), comp.get("carry") or "")
            if ek:
                return ek
        b = ENG.best(set(self._field) | set(self._hand_book))
        return b["key"] if b and b["started"] else ""       # 只有 1 个零件不算"有引擎"（孙尚香一个人不是三分登场流）

    def _is_engine_part(self, name: str) -> bool:
        """这个英雄是不是目标引擎的产出方/放大器（落点由主C 逻辑管，不算）——卖人、闲置手牌、过渡件判定都要绕开它。
        前 3 回合的开局件（干将/孙膑/曜/牛魔…）也算。"""
        if not name or not self.ENGINE_ENABLED:
            return False
        if name in ENG.opener_heroes(self._round_seen):
            return True
        ek = self.engine_key()
        if not ek or ENG.role(ek, name) not in ("producer", "amplifier"):
            return False
        # 引擎得真的转起来（≥2 个产出方/放大器在手）才保护零件；否则一个孤零零的产出方还是普通过渡件
        return ENG.status(ek, set(self._field) | set(self._hand_book))["started"]

    ENGINE_LOCK_ROUND = 5      # 第 5 回合起、引擎有 ≥2 个零件在手 → 主C 跟引擎（2026-09-26）

    def engine_lock(self) -> str:
        """引擎"转起来了"就返回它的 key（此后主C 候选限引擎落点、换人/收尾不卖零件），否则空串。"""
        ek = self.engine_key()
        if not ek or self._round_seen < self.ENGINE_LOCK_ROUND:
            return ""
        return ek if ENG.status(ek, set(self._field) | set(self._hand_book))["started"] else ""

    def carry_allowed(self, levels: Optional[dict] = None) -> Optional[set[str]]:
        """Laya 换主C 题允许的候选：引擎锁定后 = 引擎落点 ∪ 现任 ∪ **品阶和等级都不低于现任的场上英雄**；没锁定返回 None。
        第三类的来由（2026-09-26）：百里守约等级已比干将高，却不换主C —— 干将无敌流的落点就是 1 阶干将自己，
        引擎锁死后 2 阶 40 级的百里守约被删出候选。品阶、等级双双不低于现任的，摆给学生选不算破坏引擎。"""
        ek = self.engine_lock()
        if not ek:
            return None
        allowed = set((ENG.engines().get(ek) or {}).get("payoff") or [])
        cur = self._carry_piece
        if cur:
            allowed.add(cur)
            qc = self.pieces.get(cur, {}).get("quality", 0) or 0
            lc = (levels or {}).get(cur)
            for n in self._field:
                if n == cur:
                    continue
                qn = self.pieces.get(n, {}).get("quality", 0) or 0
                ln = (levels or {}).get(n)
                if qn >= qc and (lc is None or ln is None or ln >= lc):
                    allowed.add(n)
        return allowed

    # **按模拟买牌**（2026-10-10 晚：已经数学模拟过，先按模拟结果来；tools/sim_route_vs_flex.py 的 S2）：
    #   ① 商店出现同体系主C（主C 还没到手时）/ 缺的体系配套 / 主C 同名（没觉醒）→ 钱够就买；
    #   ② 还缺东西、且刷完还剩 ≥3（买得起一张英雄）→ 刷新（免费刷新照用）。
    # 排在 Laya 商店题之前（preempt_shop），eco 的 save / Laya 的 stop 不再让"有钱不刷"（第 530 局第 7~8 回合 14 能量 0 刷新、配套缺 4 个）。
    # 升级仍由进度表 / eco levelup 先做（spend 已扣掉升级预算）。
    SIM_POLICY = True
    SIM_REFRESH_PER_ROUND = 8          # 每回合最多付费刷几次（安全阀；模拟里一般 2~6 次就花完了）

    def _sim_missing(self, state: GameState) -> list:
        """路线最终阵容里还没拥有的人（场上 + 手牌 + 手牌账），给 level_budget 判断"配套还缺几个"。"""
        if not self.route_locked():
            return []
        from agent.decision import routes as _R
        lu = _R.by_key(self._route_key) or {}
        owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []} | set(self.hand_book_names())
        return [n for n in _R.final_team(lu) if n not in owned and self.pieces.get(n, {}).get("category") == "英雄牌"]

    def _sim_policy(self, state: GameState, spend: int) -> Optional[dict[str, Any]]:
        if not (self.SIM_POLICY and self.route_locked() and state.shop_open) or state.lord_maxed:
            return None
        if len(state.hand_slots or []) >= self.HAND_LIMIT - 1:
            return None
        from agent.decision import routes as _R
        ban = state.banned_faction or ""
        rc = self.route_carry() or ""
        lu = _R.by_key(self._route_key) or {}
        sup = [n for n in _R.final_team(lu) if n != rc and self.pieces.get(n, {}).get("category") == "英雄牌"]
        owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []} | set(self.hand_book_names())
        fam = ({rc} if rc else set()) | set(self._sibling_carries(ban))
        carry_owned = bool(self._carry_piece and self._carry_piece in owned) or bool(fam & owned)
        want = set() if carry_owned else set(fam)
        want |= {n for n in sup if n not in owned}
        cp = self._carry_piece if self._carry_piece in fam else ""
        if cp and cp not in self._ever_awakened:
            want.add(cp)                                  # 主C 同名：合成升星
        want = {n for n in want if not self.recently_sold(n) and not self.awakened_support(n)}
        eco = self.eco_says(state)
        if eco == "save":
            want &= fam                                   # 经济网络说攒钱：只放行主C（主C 越早到越好），配套不花
            spend = max(spend, state.gold or 0) if want else spend
        _cap = state.population_cap or 0
        _here = self.pop_now(state)[0]
        _full = bool(_cap) and _here >= _cap
        for sl in state.shop:
            if not sl.piece or sl.sold or sl.piece not in want:
                continue
            if _full and sl.piece not in owned and sl.piece not in fam:
                continue      # 2026-10-11：满员时新配套要卖人腾位，交给 Laya 商店题 / 换人网络
            price, _e = self.est_price(sl.piece, sl.cost)
            if spend >= price:
                kind = "主C" if sl.piece in fam else "体系配套"
                logger.info("按模拟买牌：%s（%s）", sl.piece, kind)
                return {"type": "buy_piece", "slot": sl.index, "piece": sl.piece, "cost": price, "cost_est": _e,
                        "round": state.round_no, "why": f"按模拟：{kind}"}
        missing = (not carry_owned) or any(n not in owned for n in sup)
        if not missing:
            return None
        if eco and eco != "roll":
            return None                                   # 刷不刷由经济网络决定：只有 roll 才刷（免费刷新也一样，免费的也是它说了算）
        if state.free_refresh > 0:
            return {"type": "refresh_shop", "why": "按模拟：还缺 " + "/".join(sorted(n for n in want if n not in owned)[:4])}
        cost = state.refresh_cost if (state.refresh_cost or 0) > 0 else 1   # 实际 1 能量（2026-10-19 核实）
        key = ("sim_refresh", state.round_no)
        k = self._skip_logged.get(key, 0)
        if k >= self.SIM_REFRESH_PER_ROUND or spend - cost < 3:
            return None
        self._skip_logged[key] = k + 1
        logger.info("按模拟刷新：还缺 %s（本回合第 %d 次，可用 %d）", "/".join(sorted(n for n in want if n not in owned)[:4]), k + 1, spend)
        return {"type": "refresh_shop", "cost": cost, "why": "按模拟：找缺的体系配套 / 主C"}

    def preempt_shop(self, state: GameState) -> Optional[dict[str, Any]]:
        """商店开着时，**先于 Laya** 做两件攻略里最确定的事（2026-09-26）：
          ① 升级按进度表（第 2/4/7/10 回合 2/3/4/5 级）：落后了、钱够、按钮在 → 先升级；
          ② 目标引擎还缺的零件出现在商店 → 买（钱够、手牌没满、买回来放得下或是同名）。
        其余照旧交给 Laya。"""
        if state.phase != "prepare" or not state.shop_open:
            return None
        gold = state.gold if isinstance(state.gold, int) else 0
        if gold < 0:
            return None
        if self.model_decides() and self._unreachable_engine_rush(state):
            logger.info("缺的引擎件 %s 当前 %s 级刷不出 → 钱先升级（临时规则 engine_tier_rush）", self.engine_missing_unreachable(state), state.level)
            return {"type": "buy_xp", "x": state.level_btn[0], "y": state.level_btn[1], "why": "engine_tier_rush"}
        if self.model_decides():
            # 2026-10-11 复盘（查残留规则对决策的影响）：第 537 局商店里 80 次买 / 刷由这里抢先做，
            # Laya 商店题只被问到 17 次。经济网络接管时，这里只执行经济网络的"升级"；买什么、刷不刷交给 Laya 商店题。
            if (self.eco_says(state) == "levelup" and self._leveled_round_n(state) == 0 and not state.lord_maxed
                    and state.level_cost and 0 < state.level_cost <= max(gold, self.eff_gold(state)) and state.level_btn):
                from agent.decision import interim_rules as _IRb
                if _IRb.on("buy_before_levelup") and self.route_locked():
                    # 临时规则 buy_before_levelup（规范 54.29-3："钱够两样时先买牌，升级不会改变商店，下一题再升"）：
                    # 店里有主C 同名 / 现在缺的核心件、买完还够升级 → 先买这张（下一帧再升级）
                    _cq4 = int((self.pieces.get(self._carry_piece or "") or {}).get("quality") or 0)
                    _want = ({self.route_carry()} | ({self._carry_piece} if _cq4 >= 2 else set()) | self.route_core_missing_now(state)) - {"", None}
                    _tried = self.__dict__.setdefault("_bbl_tried", set())      # 10-21 审查 4：每回合每格只试 1 次，别把升级饿死
                    for sl in state.shop:
                        if (sl.piece in _want and not sl.sold and len(state.hand_slots or []) < self.HAND_LIMIT
                                and (state.round_no, sl.index, sl.piece) not in _tried):
                            _tried.add((state.round_no, sl.index, sl.piece))
                            price, _e = self.est_price(sl.piece, sl.cost)
                            if gold >= price + state.level_cost:
                                logger.info("经济网络说升级，但 %s 在店里、钱够两样 → 先买再升级（临时规则 buy_before_levelup）", sl.piece)
                                return {"type": "buy_piece", "slot": sl.index, "piece": sl.piece, "cost": price,
                                        "cost_est": _e, "round": state.round_no, "why": "buy_before_levelup"}
                            _eng = set((self.min_engine() or {}).get("early_priority") or []) - {self.route_carry(), self._carry_piece}
                            if sl.piece in _eng and gold >= price and (state.round_no or 0) < 12:
                                # 2026-10-22 第 916 局 R9：铠（最低配套缺了 R4~R10）在店里，经济网络说升级、钱只够一样 → 升了级，铠拖到 R11。
                                # "先启动等级引擎"（engine-before-carry）：缺的引擎件先买，这回合不升
                                logger.info("经济网络说升级，但缺的引擎件 %s 在店里、钱只够一样 → 先买引擎件、这回合不升（临时规则 buy_before_levelup）", sl.piece)
                                self.__dict__["_skip_levelup_round"] = state.round_no
                                return {"type": "buy_piece", "slot": sl.index, "piece": sl.piece, "cost": price,
                                        "cost_est": _e, "round": state.round_no, "why": "engine_before_levelup"}
                if self.__dict__.get("_skip_levelup_round") == state.round_no:
                    return None
                logger.info("经济网络说本回合升级：升级费 %d ≤ 能量 %d → 升级", state.level_cost, gold)
                if self.haiyue_hold_level(state):
                    logger.info("【伞海月】3 级刷遗志：经济网络说升级，先不升")
                    return None
                return {"type": "buy_xp", "x": state.level_btn[0], "y": state.level_btn[1], "why": "局面网络 eco：升级"}
            # 2026-10-13 定的理想打法（多路线核心卡见到就买 / 主C 定了疯狂买配套）先于 Laya，但尊重经济网络：
            # save 不买；levelup 先留出升级费
            eco = self.eco_says(state)
            if eco == "save":
                return None
            spare = gold - (state.level_cost or 0) if eco == "levelup" and not state.lord_maxed else gold
            return self._flex_buy(state, spare)
        fb = self._flex_buy(state, gold)
        if fb:
            return fb
        target = self.live_level_target(state.round_no)
        cap0 = state.population_cap or 0
        here0 = self.pop_now(state)[0]
        # 场面没铺满、升完又一分不剩时不抢先升级（第 24 局：第 2 回合 6 能量全拿去升级，场上 2 个人一直到第 3 回合，血 60→28）
        board_ok = ((cap0 > 0 and here0 >= cap0 - 1) or gold - (state.level_cost or 0) >= 3
                    # 第 3 回合起、场上已有 3 人就不再等"铺满"（2026-09-30 第 16 局：第 2~3 回合 6 能量全买了人，
                    # 第 4 回合还是 1 级、3 阶主C 刷不出来，第 6 回合才拿到，血 60→8）
                    or ((state.round_no or 0) >= 3 and here0 >= 3))
        # **路线主C（或同体系主C）在商店、买得起 → 先买再升级**（2026-09-30 轮换第 5 局：第 4 回合守约在商店、9 能量，
        # 进度表先花 7 能量升级，剩 2 买不起，下回合守约没了，只好换路线）
        if self.route_locked():
            _rc = (self._active_comp or {}).get("carry") or ""
            _own = set(self._field) | {sl["name"] for sl in state.hand_slots or []}
            _want = {_rc} if _rc else set()
            if _rc and _rc not in _own and self._carry_piece != _rc:
                _want |= self._sibling_carries(state.banned_faction or "")
                # **主C 越早到手越好**（2026-10-02 第 61~111 局：第 3 回合到手 11 局 9 次前三，第 5 回合 13 局 5 次）：
                # 还没主C、在 EARLY_CARRY_ROUND 以内 → 任何可走备选路线的主C 出现在商店也先买（谁先来用谁，_forced_fallback 会切过去）
                if (state.round_no or 0) <= self.EARLY_CARRY_ROUND and self.forced_route:
                    # 只买**同体系**的主C（2026-10-10 蒙特卡洛：体系内谁先到用谁最好；体系外的主C 买来也不会换过去）
                    for _k, _c in self.family_routes(state.banned_faction or ""):
                        if _c and self._carry_eligible(_c) and self.team_feeds_carry(_c):
                            _want.add(_c)
            if self.substitute_window(state):
                from agent.decision import routes as _R3
                for _k in self.substitute_keys(state.banned_faction or ""):
                    _c = (_R3.by_key(_k) or {}).get("carry") or ""
                    if _c and _c not in _own:
                        _want.add(_c)
            _oc = self.forced_orig_carry(state)
            if _oc and _oc not in _own:
                _want.add(_oc)          # 换回窗口内，首选主C（守约）出现也先买
            for sl in state.shop:
                if sl.piece in _want and not sl.sold and len(state.hand_slots or []) < self.HAND_LIMIT:
                    price, _e = self.est_price(sl.piece, sl.cost)
                    if gold >= price:
                        logger.info("路线主C %s 在商店 → 先买（再考虑升级）", sl.piece)
                        return {"type": "buy_piece", "slot": sl.index, "piece": sl.piece, "cost": price,
                                "cost_est": _e, "round": state.round_no, "why": "路线主C"}
        if self.eco_says(state) == "levelup" and self._leveled_round_n(state) == 0:
            target = max(target, (state.level or 0) + 1)    # 局面网络说本回合升级：先升一级（压过搜牌窗口），剩下的钱再按规则搜牌
        elif self._in_roll_window(state) or self.hold_level_for_carry(state):
            target = 0                  # 搜牌窗口 / 3 级找主C：停下来刷牌，不按进度表升级
        if (not state.lord_maxed and 0 < (state.level or 0) < target and state.level_cost and board_ok
                and state.level_cost > 0 and gold >= state.level_cost and state.level_btn):
            if self._skip_logged.get(("preempt_level", state.round_no)) != state.level:
                self._skip_logged[("preempt_level", state.round_no)] = state.level
                logger.info("进度表：第 %d 回合该 %d 级、现在 %d 级、升级费 %d ≤ 能量 %d → 先升级",
                            state.round_no, target, state.level, state.level_cost, gold)
            return {"type": "buy_xp", "x": state.level_btn[0], "y": state.level_btn[1]}
        # **以下（路线要的牌 / 引擎零件 / 配套 / 合成材料 / 搜牌刷新）都扣掉升级预算再花**（2026-10-09 第 474 局：R9 15 能量、要留 9，
        # 这几条先于 Laya 连买 4 张 + 刷新一次，R10 差 3 没升成）。路线主C 和进度表升级在上面，不受影响
        spend = gold - self.level_budget(state)
        wantc = self.route_wanted_cards()
        if wantc and len(state.hand_slots or []) < self.HAND_LIMIT - 1:
            for sl in state.shop:
                if sl.piece in wantc and not sl.sold and not self.recently_sold(sl.piece) and not self.swap_declined_full(state, sl.piece):
                    price, _e = self.est_price(sl.piece, sl.cost)
                    if spend >= price:
                        logger.info("路线要的牌 %s 在商店 → 先买", sl.piece)
                        return {"type": "buy_piece", "slot": sl.index, "piece": sl.piece, "cost": price,
                                "cost_est": _e, "round": state.round_no, "why": "路线要的牌"}
        ek = self.engine_key()
        hand = [sl["name"] for sl in state.hand_slots or []]
        if len(hand) >= self.HAND_LIMIT - 1:
            return None
        owned = set(self._field) | set(hand)
        want = set(ENG.missing_parts(ek, owned)) if ek else set()
        # 养核心：路线这个阶段点名要的人还没有 → 先买；**路线主C 的同名一律买**（合成 = 堆等级）
        rc = (self._active_comp or {}).get("carry") if self.route_locked() else ""
        _sub = self.route_stage_subs(state.round_no)
        _cap0 = state.population_cap or 0
        _here0 = self.pop_now(state)[0]
        # 阶段**替补**（sub）只在场上还有空位时先买：满员时买来要卖人腾位（2026-10-08 第 422 局：第 6 回合满员，
        # 按大河路线第 4~7 回合的替补买了 1 阶干将莫邪，局面网络再卖狄仁杰给它腾位，买来没有任何作用）
        want |= {n for n in self.route_stage_members(state.round_no) if n not in owned
                 and not (n in _sub and _cap0 and _here0 >= _cap0)}
        want = {n for n in want if not self.awakened_support(n)}     # 已觉醒的辅助不再买同名
        if rc:
            want.add(rc)
        # **主C 配套也先买**（2026-10-08 第 422 局：商店刷出很多适合虞姬的阵容英雄牌都没买 ——
        # 鬼谷子 / 张良 / 敖隐 / 大司命 都在商店出现过）：路线核心 + 现任主C 的配套前 6 个（官方阵容同台 / 关系图 / 高强度辅助），
        # 第二主C 不算。满员时只在场上有路线外的人可换时买（见下面的满员判断）
        if self.route_locked():
            want |= {n for n in (self.route_core() | set(self.carry_supports_now()[:6]))
                     if n not in owned and not self.unwanted_second_carry(n)}
        _fit = self.carry_fit_heroes()
        if _fit or self.carry_settled():
            # 主C 定了：适合主C 的人 + 给主C 涨等级的人都先买（满员时由下面判断场上有没有可换掉的人）
            want |= {sl.piece for sl in state.shop if sl.piece and sl.piece not in owned
                     and self.pieces.get(sl.piece, {}).get("category") == "英雄牌"
                     and sl.piece in _fit and not self.unwanted_second_carry(sl.piece)}
        rc_missing = bool(rc) and rc not in owned and self._carry_piece != rc
        if rc_missing and (state.round_no or 0) >= 3:
            # 路线主C 还没到手：**同体系**备选路线的主C 出现也买（2026-09-30 轮换第 3 局：狄仁杰 9 个回合没刷到，
            # 同一套李元芳/盾山/程咬金的守约、沈梦溪路线主C 也没去拿，枫叶舞一直挂在李元芳身上）
            want |= self._sibling_carries(state.banned_faction or "")
            want |= {n for n in ENG.opener_heroes(state.round_no) if n not in self._field}
        if not want:
            return (self._merge_buy(state, spend) or self._carry_search(state, spend, rc_missing, rc)
                    or self._sim_policy(state, spend))
        cap = state.population_cap or 0
        here = self.pop_now(state)[0]
        waiting = [n for n in hand if n not in self._field and self.pieces.get(n, {}).get("category") == "英雄牌"]
        for sl in state.shop:
            if not sl.piece or sl.sold or sl.piece not in want or self.recently_sold(sl.piece) or self.swap_declined_full(state, sl.piece):
                continue
            price, _e = self.est_price(sl.piece, sl.cost)
            if spend < price:
                continue
            if cap and here >= cap and sl.piece not in owned and self.ECO_TAKEOVER:
                continue    # 2026-10-11：满员时新人头买不买（要卖人腾位）交给 Laya 商店题 / 换人网络，规则不抢先（第 537 局买铠卖明世隐）
            if cap and here >= cap and sl.piece not in owned:
                # **满员时新零件要有人能换下才买**（2026-10-10 第 515 局：第 14 回合 7/7、场上全是路线成员，照样买了云中君，
                # 阵容已完全成型、上不了场，买也白买）。原来只在"手里已有人在等"时才拦。
                # 能换下的：路线核心 / 主C 配套 → 场上路线外的人；喂主C / 配套 → 场上的"死重"（低等级、不喂主C）。
                # 手里已经在等上场的英雄先占掉这些名额。
                if sl.piece in (self.route_core() | set(self.carry_supports_now()[:6])):
                    room_n = sum(1 for n in self._field if n not in self.route_members() and n != self._carry_piece)
                elif sl.piece in _fit or self.feeds_carry(sl.piece):
                    room_n = sum(1 for n in self._field if self.dead_weight(n))
                else:
                    room_n = 0
                _wait_plan = [n for n in waiting if n == self._carry_piece or n in self.route_members()
                              or n in _fit or self.feeds_carry(n)]          # 手里不打算上场的（会被当闲置卖）不占名额
                if room_n <= len(_wait_plan):
                    continue    # 同名（合成）在 owned 里，不受影响
            ename = ENG.engines().get(ek, {}).get("name", ek) if ek else ("路线" if self.route_locked() else "开局")
            logger.info("引擎 %s 缺 %s（%s），商店有 → 先买", ename, sl.piece, ENG.role(ek, sl.piece) or "opener")
            return {"type": "buy_piece", "slot": sl.index, "piece": sl.piece,
                    "cost": sl.cost if sl.cost is not None else 3, "cost_est": sl.cost is None, "round": state.round_no}
        return (self._merge_buy(state, spend) or self._carry_search(state, spend, rc_missing, rc)
                or self._sim_policy(state, spend))

    CARRY_SEARCH_PER_ROUND = 3

    FLEX_BUY = True     # 2026-10-13 理想打法第 3 条（见 flex_core_cards）

    def flex_core_cards(self, state: GameState) -> set:
        """**多条路线的核心卡，见到就买、只买一张**（2026-10-13 理想打法：1. 开局定一个初始路线，可以改；2. 前期定一个临时 C 位，
        把枫叶舞装上；3. 多个路线的配套核心卡，你见到就买，只买一张；4. 碰到高分 C、且有配套，直接换成最终主C。打法要有灵活性）。
        候选 = 指定路线清单（首选 + 备选）里没被禁派系废掉的路线的核心引擎（route_min_engine.json 的 core_set，含主C）。
        主C 已定（不是 1 阶临时主C）后，只留当前路线的核心。"""
        if RuleEngine._MIN_ENGINE is None:
            self.min_engine()
        from agent.decision import routes as _R
        settled = self.final_carry_settled()
        keys = ([str(self._route_key)] if settled and self._route_key else
                [k for k in ((self.__dict__.get("_forced_orig") or self.forced_route or ""),) + tuple(self.forced_backups or ()) if k])
        out = set()
        for k in keys:
            lu = _R.by_key(k)
            if not lu or not _R.viable(lu, state.banned_faction or ""):
                continue
            for u in ((RuleEngine._MIN_ENGINE or {}).get(str(k)) or {}).get("core_set") or []:
                t = str(u.get("unit") or "")
                if t.startswith("（牌）") or "搭档" in t or "或" in t:
                    continue
                out |= {n for n, c in self.pieces.items() if c.get("category") == "英雄牌" and n == t}
        return out

    def is_temp_carry(self, name: str) -> bool:
        """临时主C = 1 阶（**觉醒过的也算**）且不是锁定路线的主C。和 Laya 题面 f9lm 的 TEMPORARY / hold 行 placeholder 同口径
        （2026-10-13 老师反馈：原来觉醒的 1 阶孙尚香 / 李元芳被当成"最终主C 已定"，和 NO settled carry yet 自相矛盾）。"""
        if not name or (self.pieces.get(name, {}).get("quality") or 0) != 1:
            return False
        return name != (self.route_carry() if self.route_locked() else "")

    def final_carry_settled(self) -> bool:
        """最终主C 已定 = 有主C、不是 1 阶（觉醒的 1 阶也算临时）、能当主C（_carry_eligible）。
        和题面 f9lm 的 FINAL carry settled 同口径（题面另要求"没有不同的 future carry"，规则层没有这个概念）。"""
        cur = self._carry_piece
        return bool(cur) and (self.pieces.get(cur, {}).get("quality") or 0) >= 2 and self._carry_eligible(cur)

    def swap_declined_full(self, state: GameState, name: str) -> bool:
        """买 name 会多一个要腾位的新人，而换人网络已经对手里等着上场的人判过"不换" → True（规则不再先买）。
        第 631 局 R5/R6：蒙恬在手里等、网络连续判"不换"（不卖场上 1 级的守约 / 沈梦溪），规则仍按"场上有闲人"买了铠，
        两张英雄躺了 6 个回合。卖不卖由换人网络定，规则只是别再买它不会让上场的人。"""
        if self.pieces.get(name, {}).get("category") != "英雄牌" or name in self._field:
            return False
        cap = state.population_cap or 0
        if not cap:
            return False
        here = self.pop_now(state)[0]
        benched = self.__dict__.get("_swap_benched") or {}
        waiting = {sl["name"] for sl in state.hand_slots or []
                   if self.pieces.get(sl["name"], {}).get("category") == "英雄牌" and sl["name"] not in self._field}
        return here + len(waiting) >= cap and any(w in benched for w in waiting)

    def _flex_buy(self, state: GameState, gold: int) -> Optional[dict[str, Any]]:
        """主C 没定：flex_core_cards 里、场上 / 手里一张都没有的，商店出现就买一张（钱够、手牌留 2 格）。
        **最终主C 定了：疯狂买配套，不限一张**（定好最终主C 之后疯狂买配套英雄，不限一张）——
        当前路线核心（含主C）+ 主C 的配套（carry_fit_heroes），同名也买（合成升级），已觉醒的辅助不再买。"""
        from agent.decision import interim_rules as _IR
        if not (self.FLEX_BUY and _IR.on("flex_buy")) or len(state.hand_slots or []) >= self.HAND_LIMIT - 2:
            return None
        settled = self.final_carry_settled()
        if settled and not _IR.on("flex_buy_settled"):
            return None     # 2026-10-15 第 632 局：主C 定了之后"疯狂买配套"由 Laya 商店题判断（规则在满员时还连买，手牌堆到 13 张）
        want = set(self.flex_core_cards(state))
        if settled:
            want |= set(self.carry_fit_heroes() or ()) | {self._carry_piece}
        if not want:
            return None
        owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []}
        for sl in state.shop or []:
            n = sl.piece
            if not n or sl.sold or n not in want or self.recently_sold(n):
                continue
            if n in owned and (not settled or self.awakened_support(n) or n in self._ever_awakened and n != self._carry_piece):
                continue
            if settled and n not in owned:
                # 新配套：满员又没有可换掉的人（dead_weight）时先不买（10-08 场景 210 的约定）；同名合成不受限
                cap = state.population_cap or 0
                here = self.pop_now(state)[0]
                if self.swap_declined_full(state, n):       # 第 631 局 R6（见 swap_declined_full）
                    continue
                _team = self._route_team()
                _spare = any(x not in _team and x != self._carry_piece for x in self._field)
                _me_missing = n in (self.min_engine().get("early_priority") or [])
                # 路线最低配套缺件（第 627 局：李元芳为换绑枫叶舞卖掉后 R3 / R8 两次在店都没买回，玄策古币引擎停摆，R10 主C 47 级 vs 对面 163）
                # 满员但场上有路线外的备用卡时照买 —— 由 spare_swap 卖备用卡腾位
                if cap and here >= cap and not any(self.dead_weight(x) for x in self._field) and not (_me_missing and _spare):
                    continue
            price, _e = self.est_price(n, sl.cost)
            if gold >= price:
                why = "最终主C 已定，买配套" if settled else "多路线核心卡（只买一张）"
                logger.info("%s：%s", why, n)
                return {"type": "buy_piece", "slot": sl.index, "piece": n, "cost": price, "cost_est": _e,
                        "round": state.round_no, "why": why}
        return None

    def route_eta(self, key: str, state: GameState, owned=None) -> float:
        """**这条路线预计还要几个回合凑齐核心**（主C + 最低配套 core_set；2026-10-14 起按数值算性价比）。
        每缺一个人：按当前棋手等级（第 4 回合后按进度表的下一级）用 meta.shop_hit_prob 算每看一次商店见到他的概率，
        每回合按 VIEWS 次看店，取"所有缺件都至少见到一次"的概率过半的回合数。不算拍卖 / N 选 1（会更快）。"""
        from agent.decision import routes as _R
        lu = _R.by_key(key) or {}
        if not lu:
            return 99.0
        own = set(owned or ()) | set(self._field) | set(self.hand_book_names())
        self.min_engine()                                   # 确保最低配套表已加载
        me = (RuleEngine._MIN_ENGINE or {}).get(str(key)) or {}
        need = [lu.get("carry")] + [u.get("unit") for u in (me.get("core_set") or [])]
        need = [n for n in dict.fromkeys(need) if n and n not in own and self.pieces.get(n, {}).get("category") == "英雄牌"]
        if not need:
            return 0.0
        VIEWS = 3.0
        rnd0 = state.round_no or 1
        miss_p = {n: 1.0 for n in need}           # 到第 t 回合为止还没见到过的概率
        for t in range(1, 25):
            lv = min(max(state.level or 1, meta.level_target(rnd0 + t)), meta.MAX_LORD_LEVEL)   # 等级按进度表往上走
            for n in need:
                q = self.pieces.get(n, {}).get("quality") or 3
                pv = meta.shop_hit_prob(q, lv) or 0.0
                miss_p[n] *= (1 - pv) ** VIEWS
            prob = 1.0
            for n in need:
                prob *= 1 - miss_p[n]
            if prob >= 0.5:
                return float(t)
        return 25.0

    def family_routes(self, banned: str = "", base_key: str = "") -> list:
        """**同体系路线**：和基准路线（默认当前路线）最终阵容共用 ≥3 个配套、没被禁用派系废掉、主C 过主C 资格的
        官方阵容 + 指定的备选路线。2026-10-10 蒙特卡洛（tools/sim_route_vs_flex.py）：**先定体系、主C 在体系内谁先到用谁**
        比"认死一个主C"（第 4 回合前拿到主C 57% → 94%）和"完全随缘"（第 8 回合主C+配套 4.80 → 5.06/7）都好 ——
        守约 / 沈梦溪（指定路线清单里的同体系主C；狄仁杰 / 玄策 实战成绩差不进清单）共用 明世隐 / 太乙 / 钟馗 / 程咬金 / 铠 / 盾山。"""
        from agent.decision import routes as _R
        base = _R.by_key(base_key or self._route_key) or {}
        team = set(_R.final_team(base)) - {base.get("carry")}
        if not team:
            return []
        out, seen = [], set()
        # **只在指定路线清单里找**（首选 + 备选，one_match.sh 维护）：全部官方阵容会把实战成绩差、早就被去掉的主C 放回来
        # （2026-10-10 第 529 局：换到狄仁杰路线，第 6 名；第 330 局起最终主C 狄仁杰 4 局平均 5.0、0 前三）。
        # 蒙特卡洛：守约 / 沈梦溪 两个主C 时 ≤第 4 回合拿到主C 83%（四个时 94%），第 8 回合主C+配套 5.08 vs 5.06
        _keys = ((self.__dict__.get("_forced_orig") or self.forced_route or ""),) + tuple(self.forced_backups or ())
        pool = [_R.by_key(k) for k in _keys if k]
        if not self.forced_route:
            pool += list(_R.ali_lineups())                 # 没有指定路线清单（离线 / 老场景）才放开到全部官方阵容
        for lu in pool:
            if not lu or not lu.get("carry") or _R.excluded(lu) and not lu.get("custom"):
                continue
            c = lu["carry"]
            if c in seen or not _R.viable(lu, banned):
                continue
            if len(team & (set(_R.final_team(lu)) - {c})) >= 3 and (c == base.get("carry") or self._carry_eligible(c)):
                seen.add(c)
                out.append((str(lu.get("key")), c))
        return out

    def _sibling_carries(self, banned: str = "") -> set:
        """当前路线的**同体系**主C（family_routes：共用 ≥3 个配套的官方阵容 + 备选路线，不只看备选名单 —— 狄仁杰 / 玄策 不在备选里）。"""
        return {c for _, c in self.family_routes(banned)}

    ROLL_ROUNDS_EXTRA = 3           # 路线核心还缺 ≥2 个时额外的搜牌回合
    # **经济节奏交给局面网络**（2026-10-10）：开着时每回合第一次开店问 eco 题
    # （升级 / 停级刷牌 / 攒钱），答案写进 _eco_choice[回合]：roll = 本回合当搜牌回合（不按进度表升级、刷新找缺的人，不占规则名额），
    # levelup = 本回合升一级（压过"3 级找主C"的停级），save = 照常。规则只查合法性（钱 / 按钮 / 手牌满 / 缺的人这一级刷不出）。
    ECO_TAKEOVER = False
    MODEL_DECIDES = True        # 经济网络接管时，商店里规则不抢先（只执行 eco 的升级），买 / 刷交给 Laya 商店题（2026-10-11）

    def model_decides(self) -> bool:
        return bool(self.ECO_TAKEOVER and self.MODEL_DECIDES)

    NO_LOCK_OUTSIDE_GROUP = _LordRule("no_lock_outside_group", ("亚连", "狄仁杰"))

    def _group_carry_beats_talent(self, cur: str, tlc: str, state: GameState) -> bool:
        """天赋定死的主C（tlc）还没到手（场上 / 手里都没有），现任主C cur 是开局组员、2 阶以上、和 tlc 同阵营 → True。"""
        from agent.decision import lord_config as _LCg
        own = set(self._field) | {sl.get("name") for sl in state.hand_slots or []}
        if tlc in own or cur not in self._field:
            return False
        _oc = _LCg.load().opening_choice(state.banned_faction or "")
        mem = set((_oc or {}).get("members") or ()) if (_oc or {}).get("kind") == "group" else set()
        return (cur in mem and int((self.pieces.get(cur) or {}).get("quality") or 0) >= 2
                and self._faction(cur) == self._faction(tlc))

    def no_lock_outside_group(self, state: GameState) -> set:
        """临时规则 no_yalian_unless_heluo_ban（10-23 试行；10-23 第 1005 局扩到狄仁杰）：开局配置是组（河洛没禁）时，
        这些组外主C 的路线不锁（狄仁杰拿了幸运硬币例外）。879~1005 局：亚连非禁河洛均 4.3、狄仁杰无硬币 6/6/3/6。"""
        from agent.decision import interim_rules as _IRn, lord_config as _LCn
        if not _IRn.on("no_yalian_unless_heluo_ban"):
            return set()
        if (state.round_no or 0) > 8:
            # 10-24 第 1021 / 1023 局：组外主C 整局不锁 → 没路线、核心件不刷，6 / 5 名（比"第 8 回合后退到亚连"均 4.3 还差）→ 只挡前 8 回合
            return set()
        _oc = _LCn.load().opening_choice(state.banned_faction or "")
        if not (_oc and _oc.get("kind") == "group"):
            return set()
        _cfg = _LCn.load()
        # 10-23 第 1006 局：只挡亚连 / 狄仁杰 → 换主C 换成蒙犽（4 阶逐鹿）又锁了他的路线（第 6）。开局组能走时组外主C 一律不锁
        out = (set(self.NO_LOCK_OUTSIDE_GROUP) | {n for n in (self.pieces or {}) if _cfg.enabled(n)}) - set(_oc.get("members") or ())
        if self.__dict__.get("_talent_route_carry"):
            out.discard(self.__dict__.get("_talent_route_carry"))
        return out

    def group_wait_missing(self, state: GameState) -> list:
        """河洛没禁、路线没锁、开局组（河洛三C）一个都不在场上 / 手里时，当前等级刷得出的组员（10-24：1002~1030 局没禁河洛
        又等不到组员的 9 局均 5.4 —— 前 8 回合只是干等，没去刷）。"""
        from agent.decision import lord_config as _LCw, meta as _mtw, interim_rules as _IRw2
        if not _IRw2.on("core_roll_first") or self.route_locked() or not (3 <= (state.round_no or 0) <= 8):
            return []
        _oc = _LCw.load().opening_choice(state.banned_faction or "")
        if not (_oc and _oc.get("kind") == "group"):
            return []
        mem = list(_oc.get("members") or ())
        own = set(self._field) | {sl.get("name") for sl in state.hand_slots or []}
        if any(m in own for m in mem):
            return []
        lv = state.level or 1
        return [m for m in mem if _mtw.can_appear(int((self.pieces.get(m) or {}).get("quality") or 0), lv)]

    def core_roll_active(self, state: GameState) -> bool:
        """临时规则 core_roll_first 的生效条件：路线已锁、第 4~11 回合、路线核心件缺且当前等级刷得出、
        **棋手等级已到本回合进度线**（10-22 第 948 局：3 级就停级刷明世隐，R6~R9 一直 3 级；目标是"在合适的棋手等级刷出"）。"""
        from agent.decision import interim_rules as _IRa
        if (self.group_wait_missing(state) and (state.level or 0) >= meta.level_target(state.round_no or 0)
                and (lambda sp: sp is None or sp >= 2)(self.level_spare(state))
                and (lambda rn: rn[0] != state.round_no or rn[1] < self.ROLL_REFRESH_MAX)(self.__dict__.get("_refresh_round_n") or (0, 0))):
            return True             # 路线没锁、等开局组主C：刷他们（同样服从进度线 / 升级额度 / 每回合上限）
        _rcm = self.route_carry() if self.route_locked() else ""
        _carry_missing = bool(_rcm and _rcm in self.route_core_missing_now(state) and 3 <= (state.round_no or 0) <= 8
                              and (state.level or 0) >= meta.level_target(state.round_no or 0))   # 10-24 第 1010 局：刷狄仁杰拖得 R5~R9 一直 3 级 → 落后进度线先升级
        # 10-23 第 1003 局：禁河洛锁了亚连，亚连 R7~R8 才到（1 阶孙尚香顶到 R7，R5~R10 掉 46 血）——
        # 缺的是路线主C 本人（且刷得出）时不等进度线，第 3~8 回合就刷他（额度 / 每回合上限照旧）
        return bool(_IRa.on("core_roll_first") and self.route_locked() and (4 <= (state.round_no or 0) <= 11 or _carry_missing)
                    and ((state.level or 0) >= meta.level_target((state.round_no or 0) + 2) or _carry_missing)
                    # 10-22 第 965 局：R5 刷 4 次，R7 升不起 4 级 → 两回合后的进度线已经到了才刷（先升到合适的等级、再停级刷）
                    and (lambda sp: sp is None or sp >= 2)(self.level_spare(state))   # 10-22 第 953 局：R8~R9 刷光了，R10 升不起 5 级 → 服从"下回合还升得起"额度
                    and (_carry_missing or not self.core_missing_unreachable(state))   # 缺主C 本人时不因别的引擎件刷不出而停
                    and (lambda rn: rn[0] != state.round_no or rn[1] < self.ROLL_REFRESH_MAX)(self.__dict__.get("_refresh_round_n") or (0, 0))
                    # 10-22 第 957 局：R4 27 能量连刷 10 次，R5~R7 停 3 级 → 每回合最多 ROLL_REFRESH_MAX 次    # 10-22 第 955 局：雅典娜（4 阶）3 级刷不出，R5~R6 却刷光了钱，R7 升不起 4 级
                    and self.route_core_missing_now(state))

    def core_missing_unreachable(self, state: GameState) -> list:
        """最低配套引擎件（雅典娜等）/ 路线主C 里还缺、当前棋手等级刷不出的（后期件除外）。有就先攒钱升级，不为别的缺件刷光。
        只看引擎件和主C：守约线最终阵容里的李信 / 大司命 / 花木兰前期本来就刷不出，算进来会让 core_roll_first 前期永不生效。"""
        from agent.decision import meta as _mt, lord_config as _LCm
        lv = state.level or 1
        me = self.min_engine() or {}
        own = set(self._field) | {sl.get("name") for sl in state.hand_slots or []} | set(self.hand_book_names())
        sub = _LCm.load().substitutes
        want = {n for n in (set(me.get("early_priority") or []) | {self.route_carry()})
                if n and n not in own and not any(x in own for x in sub(n))}
        want -= self.late_pieces() | {"", None}
        return sorted(n for n in want if not _mt.can_appear(int((self.pieces.get(n) or {}).get("quality") or 0), lv))

    def eco_says(self, state: GameState) -> str:
        if not self.ECO_TAKEOVER:
            return ""
        eco = (self.__dict__.get("_eco_choice") or {}).get(state.round_no or 0, "")
        # 临时规则 core_roll_first（10-22：核心阵容没有在合适的棋手等级刷出）：路线核心件缺、当前等级刷得出 → 钱先刷
        try:
            from agent.decision import interim_rules as _IRc
            if eco != "roll" and self.core_roll_active(state) and not (
                    eco == "levelup" and state.level_cost and (state.gold or 0) >= state.level_cost + 8):
                if self._skip_logged.get("core_roll_first") != state.round_no:
                    self._skip_logged["core_roll_first"] = state.round_no
                    logger.info("路线核心件 %s 缺、%s 级刷得出 → 经济改成刷新（原 %s，临时规则 core_roll_first）",
                                sorted(self.route_core_missing_now(state)), state.level, eco or "无")
                return "roll"
        except Exception:                                   # noqa: BLE001
            logger.exception("core_roll_first 判定失败")
        return eco

    def set_eco(self, round_no: int, choice: str) -> None:
        self.__dict__.setdefault("_eco_choice", {})[round_no or 0] = choice

    def _leveled_round_n(self, state: GameState) -> int:
        return 1 if state.round_no and state.round_no == self._leveled_round else 0
    ROLL_ROUNDS = 2                 # 到了搜牌等级，专门留几个回合刷牌凑阵容（特意留 1-2 个回合，2026-09-30）
    ROLL_REFRESH_MAX = 4            # 搜牌窗口里每回合最多付费刷几次（8→4：2026-09-30 统计，输的局刷新多、合成少）
    CARRY_SEARCH_PER_ROUND = 3      # 窗口外只为主C 刷，每回合最多几次（前 6 回合只刷 1 次，见 _carry_search）

    def _essential_support(self, name: str) -> bool:
        """对最终主C 是**必要配套**：路线核心、主C 配套前 6（官方阵容高频队友 + 关系图里给主C 涨级 / 回蓝 / 放大的）、
        给主C 涨等级的（feeds_carry）、要贴主C 的（瑶 / 太乙 / 明世隐…）。"""
        try:
            sup = set(self.carry_supports_now()[:6])
        except Exception:                                # noqa: BLE001
            sup = set()
        return (name in self.route_core() or name in sup or self.feeds_carry(name) or self._wants_carry_side(name))

    AWAKEN_HELP_RE = re.compile(r"己方|所有英雄|全体|最近|古币|等级|友方|左右\d*格")

    def awaken_helps_carry(self, name: str) -> bool:
        """**觉醒后能更好辅助主C 的最终阵容成员**（其他英雄觉醒后效果能更好地辅助主C、
        且这个英雄是最终阵容的，也可以买牌升到觉醒）。只读卡面：觉醒文本和普通文本不同（效果变强），效果作用在别人身上
        （给主C / 全队涨级、贴主C 的、己方 / 全体 / 最近、产古币），只作用于自身的不算（程咬金"对自身使用古币"）；
        并且在当前路线的最终阵容里、不是主C、还没觉醒。"""
        if not name or name == self._carry_piece or name in self._ever_awakened or not self.route_locked():
            return False
        v = self.pieces.get(name, {})
        if v.get("category") != "英雄牌":
            return False
        from agent.decision import routes as _R
        if name not in set(_R.final_team(_R.by_key(self._route_key) or {})):
            return False
        sk = v.get("skill", "") or ""
        if RuleEngine.__dict__.get("_AWAKEN_TEXT") is None:
            try:
                _c = json.loads((Path(__file__).resolve().parents[2] / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
                RuleEngine._AWAKEN_TEXT = {k: (v.get("awaken") or "") for k, v in _c.items()}
            except (OSError, ValueError, KeyError):
                RuleEngine._AWAKEN_TEXT = {}
        aw = RuleEngine._AWAKEN_TEXT.get(name, "")
        if not aw or aw.strip() == sk.strip():
            return False                      # 觉醒不改效果（或卡库没写）：不为觉醒专门买
        if "自身" in sk and not re.search(r"最近|己方|所有|全体|友方|左右", sk):
            return False
        summ = [t for t in re.findall(r"召唤(\S+?)[。，,]", sk + "。")]
        if any(self.AWAKEN_HELP_RE.search(self.pieces.get(t, {}).get("skill", "") or "") for t in summ):
            return True                       # 召唤物给别人加东西（因缘图腾全队涨级 / 先知图腾给最近的人装备），觉醒"召唤并强化"
        return bool(self.feeds_carry(name) or self._wants_carry_side(name) or self.AWAKEN_HELP_RE.search(sk))

    def _merge_buy(self, state: GameState, gold: int) -> Optional[dict[str, Any]]:
        """**刷新之前先买合成材料**（2026-09-30 统计 19 局：前 10 回合前三名的局合成平均 ~7 次、刷新 6~10 次；
        第 29~34 局合成 1~4 次、刷新 6~19 次 —— 钱花在刷商店找人上，场上已有的人没合成、养不起来）。
        商店里有**场上已有英雄**的同名：主C / 路线这阶段要的人 / 手里已经有 1 张同名的（凑合成）→ 买。"""
        if len(state.hand_slots or []) >= self.HAND_LIMIT - 1:
            return None
        hand = collections.Counter(sl["name"] for sl in state.hand_slots or [])
        keep = self.route_stage_members(state.round_no) | ({self._carry_piece} if self._carry_piece else set())
        # **前期场面薄时先铺人头，合成材料让位**（2026-10-03 第 151 局：第 2 回合 6 能量买两张干将合成，场上一直 2/5；
        # 统计第 61~150 局：第 3 回合场上人数 前三 4.25 / 3.91 vs 第 4~6 名 3.80 / 3.62，铂金、钻石两段都是前三多）。
        # 第 4 回合以内、场上 ≤3 人且还有空位：买完同名后剩的钱不够再买一个人头（3）就不买，路线主C 的同名除外。
        here = self.pop_now(state)[0]
        cap = state.population_cap or 5
        thin = (state.round_no or 0) <= 4 and here <= 3 and here < cap
        rc = (self._active_comp or {}).get("carry") if self.route_locked() else ""
        for sl in state.shop:
            n = sl.piece
            if not n or sl.sold or n not in self._field or self.pieces.get(n, {}).get("category") != "英雄牌":
                continue
            if not (n in keep or hand.get(n, 0) >= 1 or self.awaken_helps_carry(n)):
                continue                    # 觉醒后更能辅助主C 的最终阵容成员也买同名、合到觉醒（2026-10-10）
            if self.awakened_support(n) or self.outgoing_unit(n):
                continue                    # 已觉醒的辅助 / 马上要被换掉的老主C 不再买同名
            from agent.decision import interim_rules as _IR
            if (_IR.on("temp_carry_no_copies") and n == self._carry_piece and n != rc and self.is_temp_carry(n)):   # 觉醒的 1 阶也算临时（10-13）
                continue                    # 临时主C（1 阶）不买同名凑觉醒（临时的不值得，不如多买最终 C 位的配套，2026-10-13）
            if self._obsolete(n, state.round_no) and n not in keep:
                continue
            price, _e = self.est_price(n, sl.cost)
            if thin and n != rc and gold - price < 3:
                continue                    # 前期先铺人头（见上）：交给 Laya / 买新人
            if gold >= price:
                logger.info("先买合成材料：%s（场上已有%s）", n, "，手里还有 1 张" if hand.get(n) else "")
                return {"type": "buy_piece", "slot": sl.index, "piece": n, "cost": price, "cost_est": _e,
                        "round": state.round_no, "why": "合成材料"}
        return None

    def _route_missing(self, state: GameState) -> dict:
        """路线锁定后，这回合点名要、但场上手里都没有的**英雄**：{名字: 权重}（主C 权重 2）。"""
        if not self.route_locked():
            return {}
        owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []}
        rc = (self._active_comp or {}).get("carry") or ""
        out = {}
        from agent.decision import routes as _R
        lu = _R.by_key(self._route_key) or {}
        st = _R.stage_now(lu, state.round_no or 0) if any((x.get("to_round") or 0) > 0 for x in lu.get("stages") or []) else None
        # 阶段写了回合：只算**这个阶段**点名的人（后期才要的高阶终局英雄不提前拉高搜牌等级）；没写回合才用最终阵容
        names = (set(st.get("main") or []) | set(st.get("sub") or [])) if st else set(_R.final_team(lu))
        _cap, _here = state.population_cap or 0, (self.pop_now(state)[0])
        if _cap and _here >= _cap:
            names -= self.route_stage_subs(state.round_no)      # 满员时不为阶段替补刷牌（第 422 局干将莫邪）
        for n in names:
            if n in owned or self.pieces.get(n, {}).get("category") != "英雄牌":
                continue
            if n != rc and self._is_tank(n) and not self._essential_support(n):
                # **阿离炼主C，主T 不急**（主T 等级连带着上去就行，碰到卡可以买，不急）：
                # 只为"普通前排"不停级刷牌；**对最终主C 是必要配套、辅佐能力很强的坦克照样要等着买到**
                # （同日补充：如果是已定最终主C 的必要英雄、辅佐能力很强，仍要等着买到）
                continue
            out[n] = 2.0 if n == rc else 1.0
        if rc and rc not in owned and self._carry_piece != rc:
            out[rc] = 2.0
        return out

    def roll_level(self, state: GameState) -> int:
        """**缺的人在几级商店里刷得最多**（按各级商店刷出概率判断）：
        对 2~6 级算 Σ 权重 × P(刷一次见到这个人)（meta.shop_hit_prob），取最大的；并列取低的（省升级费）。
        例：守约路线缺 2 阶守约 + 3 阶钟馗/铠 → 3 级；狄仁杰路线缺 3 阶的人 → 5 级（4 级只差一点，见 _in_roll_window）。"""
        miss = self._route_missing(state)
        if not miss:
            return 0
        best, best_v = 0, 0.0
        for lv in range(2, meta.MAX_LORD_LEVEL + 1):
            v = sum(w * meta.shop_hit_prob(self.pieces.get(n, {}).get("quality", 0) or 0, lv) for n, w in miss.items())
            if v > best_v * 1.15:           # 高一级要好 15% 以上才算（升级有成本；3 阶：3→4 级 +16% 升，4→5 级 +14% 不升）
                best, best_v = lv, v
        rc = (self._active_comp or {}).get("carry") or ""
        if best and rc in miss:
            best = max(best, meta.min_level_for(self.pieces.get(rc, {}).get("quality", 0) or 0))   # 主C 没到手：至少刷得出主C
        return best

    CARRY_YIELD_MARGIN = 25

    def carry_support_score(self, name: str, state: GameState, lv: dict | None = None) -> float:
        """**一个主C 现在"养得怎么样"的综合分**（换主C / 让位时比较用；多因素综合判断）：
          等级 0.5/级（上限 100 级）＋ 主C 强度 (评分−5)×20 ＋ 觉醒 15
          ＋ 套路配套：它当主C 的官方阵容里，场上 / 手里每有 1 个成员 +6（最多 5 个）
          ＋ 装备：身上每件对口 +5，手里对口、还穿得上的每件 +3。"""
        from agent.decision import routes as _R
        from . import hero_traits as _ht
        lvl = (lv or {}).get(name)
        if lvl is None:
            pos = self._field.get(name)
            lvl = (state.unit_levels or {}).get(cell_key(tuple(pos))) if pos else None
        sc = 0.5 * min(lvl or 0, 100)
        sc += 20 * ((self.hero_reviewed.get(name) or {}).get("carry_score", 5) - 5)
        if name in self._ever_awakened:
            sc += 15
        owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []} | self.hand_book_names()
        best_team = 0
        for lu in _R.ali_lineups():
            if lu.get("carry") == name:
                best_team = max(best_team, len((set(_R.final_team(lu)) - {name}) & owned))
        sc += 6 * min(best_team, 5)
        worn = [e for e in (self._worn.get(name) or ()) if _ht.fits(e, name)[0]]
        sc += 5 * len(worn)
        pos = self._field.get(name)
        room = (meta.MAX_EQUIPS - self._equips_at(state, pos)) if pos else 0
        hand_fit = [sl["name"] for sl in state.hand_slots or []
                    if self.pieces.get(sl["name"], {}).get("category") == "装备牌"
                    and (_ht.fits(sl["name"], name)[0] or _ht.base_kind(sl["name"]))
                    and self._equip_allowed(sl["name"], name)]
        sc += 3 * min(len(hand_fit), max(room, 0))
        return sc

    LEVEL6_ROUND = 12

    HOLD_LEVEL_FOR_CARRY = 3        # 到这一级还没有主C 就停下来找（在 3 级多停几个回合，找到主C 之后再开始升级）
    HOLD_CARRY_MAX_ROUND = 8        # 最多停到第几回合（再晚等级落后太多）
    HOLD_CARRY_REFRESH = 4          # 停级找主C 时每回合最多付费刷几次
    # 2026-10-06 第 255~314 局：主C 第 4 回合内到手 35 局 13 次前三、第 5 回合起 22 局 3 次；前三的局前 6 回合刷新中位 6 次、其余 4 次
    # → 找主C 提前到第 3 回合、前 6 回合每回合最多 2 次（原第 4 回合起、1 次）
    CARRY_SEARCH_FROM_ROUND = 3
    CARRY_SEARCH_EARLY = 2

    def carry_in_hand(self, state: GameState) -> bool:
        """路线主C（或同体系候选主C）已经到手 —— 在场 / 手牌 / 手牌账 / 觉醒回手。"""
        owned = set(self._field) | {sl.get("name") for sl in state.hand_slots or []} | self.hand_book_names()
        rc = self.route_carry() if self.route_locked() else ""
        cands = ({rc} if rc else set()) | (self._sibling_carries() if rc else set())
        if cands:
            return bool(cands & owned) or rc in self._awakened
        c = self._carry_piece
        return bool(c and c in owned and self._carry_eligible(c))

    def hold_level_for_carry(self, state: GameState) -> bool:
        """3 级、还没有主C、第 8 回合以内 → 不升级，钱用来刷新找主C（第 45/46 局：主C 第 5~6 回合才到手、1 级起步）。"""
        if self.eco_says(state) == "levelup":
            return False                 # 局面网络说本回合升级（ECO_TAKEOVER）
        if self.model_decides():
            return False                 # 模型接管时升不升级由经济网络定（审查 2026-10-11：卡住升级、却没有规则去刷主C）
        lv = state.level or 0
        rc = (self._active_comp or {}).get("carry") if self.route_locked() else ""
        if rc and meta.min_level_for(self.pieces.get(rc, {}).get("quality") or 0) > lv:
            return False        # 路线主C 在这一级根本刷不出来（4 阶韩信要 4 级）：卡在 3 级找它没用，照常升级
        if rc:
            # 升一级反而更容易刷到要找的主C（3 阶主C：3 级 30% → 4 级 35%）就别卡（2026-10-08 第 412 局：
            # 为等 3 阶沈梦溪卡在 3 级到第 8 回合、21 能量没花，第 6 名）。只有 2 阶主C（3 级 50% → 4 级 30%）卡级才划算
            sought = {rc} | set(self._sibling_carries() or [])
            qs = {self.pieces.get(n, {}).get("quality") or 0 for n in sought} - {0}
            if qs and all(meta.tier_odds(q, lv + 1) > meta.tier_odds(q, lv) for q in qs):
                return False
        cur = self._carry_piece
        if cur and cur in self._field and (self.pieces.get(cur, {}).get("quality") or 0) >= 2 and self._carry_eligible(cur):
            return False        # 已经有一个合格的 2 阶以上主C 在场（第 384 局：安琪拉当主C，仍为等虞姬卡在 3 级到第 8 回合）
        return bool(lv >= self.HOLD_LEVEL_FOR_CARRY and not state.lord_maxed
                    and 0 < (state.round_no or 0) <= self.HOLD_CARRY_MAX_ROUND and not self.carry_in_hand(state))

    def saving_for_level(self, state: GameState) -> bool:
        """等级落后进度表、钱差升级费不到 8、没有免费刷新 → 这回合攒钱升级，别付费刷新。
        2026-09-30 轮换第 11 局：第 12 回合 24 能量、升级费 25，Laya 连刷 5 次，拖到第 13 回合才升 6 级（盲目刷新）。"""
        lv, cost, gold = state.level or 0, state.level_cost or 0, state.gold if isinstance(state.gold, int) else 0
        if self.hold_level_for_carry(state):
            return False
        return bool(lv and cost > 0 and not state.lord_maxed and lv < self.live_level_target(state.round_no)
                    and 0 < cost - gold <= 8 and not state.free_refresh)

    _SPEND_TYPES = ("buy_piece", "refresh_shop", "buy_xp", "auction_bid", "drag_card", "use_card", "sell_unit", "sell_card",
                    "card_pick", "talent_pick", "lord_boost_pick", "sequence")

    def _note_gold(self, state: GameState) -> None:
        """记下上一帧的能量读数（第 636 局第 12 回合：商店开着那一帧把 24 读成 23，升级费 24 → 经济网络的升级没点、
        升级预算又把买牌全扣住，24 能量整回合没花）。"""
        if state.phase != "prepare" or not isinstance(state.gold, int) or state.gold < 0:
            return
        self._gold_prev = self.__dict__.get("_gold_obs")
        self._gold_obs = (state.round_no, state.gold, self.__dict__.get("_spend_seq", 0))

    def eff_gold(self, state: GameState) -> int:
        """升级判断用的能量：同一回合、上一帧之后没有任何会改能量的动作执行过 → 能量不可能变，取两帧里大的那个
        （小 1~2 的单帧误读点下去最多是升级按钮不响应，没有代价）。"""
        g = state.gold if isinstance(state.gold, int) else -1
        prev = self.__dict__.get("_gold_prev")
        if (g >= 0 and prev and prev[0] == state.round_no and prev[2] == self.__dict__.get("_spend_seq", 0)
                and 0 < prev[1] - g <= 2):
            return prev[1]
        return g

    def level_budget(self, state: GameState) -> int:
        """为按进度表升级要**留住**的能量（0 = 不用留）。不是次数限制，是按局面算的预算：
        - 现在就落后进度表 → 留住升级费（能升就由 preempt_shop 先升）；
        - 下回合进度表要涨一级 → 留住"升级费 − 1（回合结束 +1 经验）− 下回合收入"，下回合开局就能升。
        2026-10-09 第 469 局：第 9 回合 20 能量、升级费 22，钱花在买牌上，第 10/11 回合 15/18 能量都不够升，整局停在 4 级
        （放宽后不能又乱花钱导致无法升级棋手）。"""
        lv, cost = state.level or 0, state.level_cost or 0
        eco = self.eco_says(state)
        if eco:
            # 经济网络开着：留不留钱完全按它（levelup = 留住升级费；roll = 不留；save = 全留，只放行同体系主C，见 _sim_policy）
            if eco == "levelup":
                if not (lv and cost > 0 and not state.lord_maxed and self._leveled_round_n(state) == 0):
                    return 0
                if self.eff_gold(state) >= cost:
                    return cost
                eco = "save"            # 这回合升不起（第 636 局：差 1 点留住全部 → 整回合一张不买）：按"下回合升级"留钱，多的放给 Laya
            if eco == "save":
                # 2026-10-13 第 610 局：原来"save = 全留"，第 9 回合 17 能量一张不买，刷出的李元芳 / 程咬金（守约配套）也买不了。
                # 攒钱是为了下回合升级（+ 下回合拍卖举牌）：只留"下回合升级费 − 1 经验 − 下回合收入"（+ 拍卖 5），多的放给 Laya
                r = state.round_no or 0
                if not lv or cost <= 0 or state.lord_maxed:
                    need = 0
                else:
                    need = max(0, cost - 1 - meta.energy_income(r + 1))
                if r + 1 in meta.AUCTION_ROUNDS:
                    need += 5
                return min(state.gold or 0, need)
            return 0
        if not lv or cost <= 0 or state.lord_maxed or self.hold_level_for_carry(state) or self._in_roll_window(state):
            return 0                     # 搜牌回合本来就是停级刷牌（审查 10-09：原来照样按进度表留钱、压掉搜牌刷新）
        r = state.round_no or 0
        here = self.pop_now(state)[0]
        if here < (state.population_cap or 5) - 1:
            return 0                     # 场面还空着两格以上：先铺人头（空场必输），不为升级攒钱
        if lv < self.live_level_target(r):
            return cost
        if lv < self.live_level_target(r + 1):
            # **配套还缺 ≥2 个时不为"下回合的升级"提前留钱**（2026-10-10 第 530 局第 10 回合：5 级、12 能量、6 级要 26，
            # 预留 26-1-12=13 > 12 → 一分钱都不花，河洛配套缺 3 个、血只剩 10，有钱却一直不花）。
            # 模拟（tools/sim_route_vs_flex.py S2）：配套没齐时钱先花在买 / 刷配套上
            if len(self._sim_missing(state)) >= 2:
                return 0
            return max(0, cost - 1 - meta.energy_income(r + 1))
        return 0

    def live_level_target(self, round_no: int) -> int:
        """实战用的等级进度表 = meta.level_target（2/4/7/10 回合 2/3/4/5 级）+ **第 12 回合 6 级**。
        2026-09-30：后期盲目刷新、始终不升到 6 级——第 11~15 回合 5 级、19~28 能量、升级费 21~26，
        一直在刷新。6 级解锁阿离专属天赋（枫之洪流 / 翩跹馈赠…，守约路线原文"后期棋手上 6 级…直接开始起飞"）。
        6 级这一档只加在实战规则层，不动 meta（Laya 题面用 meta.level_target，改了会和训练数据不一致）。"""
        t = meta.level_target(round_no)
        if (round_no or 0) >= self.LEVEL6_ROUND:
            t = max(t, 6)
        return t

    def _in_roll_window(self, state: GameState) -> bool:
        """到了搜牌等级、还缺人、搜牌回合本局还没用满 ROLL_ROUNDS 个 → 这回合停下来刷牌。
        **名额按真正执行的搜牌刷新记**（note_executed，2026-09-30 审查：原来一判定就占名额，没钱没刷也算用掉）。
        手牌快满 / 缺的人这一级刷不出来时不开窗口（否则会一直压着不升级）。"""
        eco = self.eco_says(state)
        # **买牌划算还是升级划算由经济网络决定**：
        # 只有 eco 选 roll 才开搜牌窗口；levelup / save 时规则不替它刷牌（规则只查合法性：刷不刷得到、手牌满没满）
        if eco:
            if eco != "roll" or not self.route_locked() or state.lord_maxed:
                return False
            if len(state.hand_slots or []) >= self.HAND_LIMIT - 1:
                return False
            lv0 = state.level or 0
            return any(meta.can_appear(self.pieces.get(n, {}).get("quality", 0) or 0, lv0) for n in self._route_missing(state))
        if not self.route_locked() or state.lord_maxed or (state.round_no or 0) < 3:
            return False
        lv = state.level or 0
        rl = self.roll_level(state)
        if not rl or lv < rl:
            return False
        if lv < self.live_level_target(state.round_no) and state.round_no not in self.__dict__.get("_roll_rounds", set()):
            return False                 # 等级落后于进度表：先补等级，不开新的搜牌回合（第 15 回合 5 级还在"搜牌"）
        rolled = self.__dict__.setdefault("_roll_rounds", set())
        if state.round_no in rolled:
            return True
        if len(rolled) >= self.ROLL_ROUNDS + (self.ROLL_ROUNDS_EXTRA if len(self.route_core_missing(state)) >= 2 else 0):
            # 路线核心还缺 ≥2 个就多给搜牌回合（2026-10-08 第 422 局，虞姬阵容没凑起来：
            # 2 个搜牌回合在第 5、6 回合用完，第 7、8 回合 11~23 能量开店就关、一次没刷，少司缘 / 张良一个没凑上）
            return False
        if len(state.hand_slots or []) >= self.HAND_LIMIT - 1:
            return False
        miss = self._route_missing(state)
        if not any(meta.can_appear(self.pieces.get(n, {}).get("quality", 0) or 0, lv) for n in miss):
            return False
        if self._skip_logged.get(("roll_window", state.round_no)) is None:
            self._skip_logged[("roll_window", state.round_no)] = 1
            logger.info("搜牌窗口：%d 级（缺的人在这一级刷得最多），第 %d 回合停下来刷牌凑阵容（已用 %d/%d 个搜牌回合）；缺 %s",
                        lv, state.round_no, len(rolled), self.ROLL_ROUNDS, sorted(miss))
        return True

    def _carry_search(self, state: GameState, gold: int, rc_missing: bool, rc: str) -> Optional[dict[str, Any]]:
        """**等级和要买的牌自己平衡**：
          · 等级还没到"缺的人刷得最多"的那一级，钱够 → 先升级；
          · 到了 → 搜牌窗口（ROLL_ROUNDS 个回合）：不升级，刷新找路线缺的人，每回合最多 ROLL_REFRESH_MAX 次；
          · 窗口外只为还没到手的主C 刷（每回合最多 CARRY_SEARCH_PER_ROUND 次）。
        刷完都要留得下"买一张 3 能量 + 拍卖预留"。"""
        if not self.route_locked() or (state.round_no or 0) < 3:
            return None
        miss = self._route_missing(state)
        if not miss:
            return None
        if len(state.hand_slots or []) >= self.HAND_LIMIT - 1:
            return None
        lv = state.level or 0
        rl = self.roll_level(state)
        cost = state.refresh_cost if state.refresh_cost >= 0 else self.refresh_cost
        hold = self.hold_level_for_carry(state)
        full = state.gold if isinstance(state.gold, int) else gold   # 升级本身不扣升级预算（审查 10-09：原来被双重扣钱）
        if (not hold and rl and lv and lv < rl and not state.lord_maxed and state.level_btn and state.level_cost
                and state.level_cost > 0 and full >= state.level_cost + 3 + self._auction_hold(state)):
            logger.info("缺 %s，%d 级刷得最多、现在 %d 级 → 先升级（升级费 %d，能量 %d）",
                        sorted(miss), rl, lv, state.level_cost, gold)
            return {"type": "buy_xp", "x": state.level_btn[0], "y": state.level_btn[1], "why": f"升到搜牌等级 {rl}"}
        window = self._in_roll_window(state)
        if not window and not hold and not (rc_missing and (state.round_no or 0) >= self.CARRY_SEARCH_FROM_ROUND):
            return None
        for n in miss:
            q = self.pieces.get(n, {}).get("quality", 0) or 0
            if q and lv and meta.can_appear(q, lv):
                break
        else:
            return None                                   # 缺的人这一级都刷不出来
        key = ("carry_search", state.round_no)
        k = self._skip_logged.get(key, 0)
        why = ("搜牌回合：找 " if window else "找主C ") + "/".join(sorted(miss))
        if state.free_refresh > 0:
            logger.info("%s → 免费刷新", why)
            return {"type": "refresh_shop", "why": why}
        cap = (self.HOLD_CARRY_REFRESH if hold else
               self.ROLL_REFRESH_MAX if window else (self.CARRY_SEARCH_EARLY if (state.round_no or 0) <= 6 else self.CARRY_SEARCH_PER_ROUND))
        if hold:
            why = "3 级找主C：" + "/".join(sorted(miss))
        if k >= cap or gold - cost < 3 + self._auction_hold(state):
            return None
        self._skip_logged[key] = k + 1
        logger.info("%s → 付费刷新（本回合第 %d 次，能量 %d）", why, k + 1, gold)
        return {"type": "refresh_shop", "cost": cost, "why": why}

    def _is_core(self, piece: str) -> bool:
        # 过了第 6 回合的 1 阶**不再是核心**（和 _is_carry 同理）：2026-09-23 实战第 9 回合，
        # 场上最弱的干将莫邪（3 分）因为是阵容"核心"不能卖，手里的敖隐(5 阶)、韩信(4 阶)
        # 就一直上不去，整回合一个动作都没有（英雄卡在手上、也不升级）。
        return (bool(self._active_comp) and piece in (self._active_comp.get("core") or [])
                and not self._obsolete(piece, self._round_seen))

    # ---------- 经济 ----------

    # ------------------------------------------------------------------ 伞海月（指定的打法节奏）
    HAIYUE_KEY = "user_haiyue_umbrella"
    HAIYUE_TEMP = "干将莫邪"
    HAIYUE_FARM_LEVEL = 3          # 3 级合成送的是 3 阶效果锦囊 —— 遗志只从 3 阶锦囊里开（17/44 实测）
    HAIYUE_FARM_JN = 4             # 3 阶锦囊开了这么多个还没出遗志就不再停（4 个累计 ~86%，推演器 jinnang_tick）
    HAIYUE_FARM_UNTIL = 9          # 最晚第 9 回合结束停留
    HAIYUE_RUSH_LEVEL = 5          # 海月 5 阶：5 级商店才出

    def haiyue_phase(self, state: GameState) -> str:
        """伞海月路线的阶段（不在这条路线上返回 ""）：
        raise 1~2 级养干将（觉醒后同名先捏在手里）→ farm 3 级集中合成刷 3 阶锦囊开遗志 → rush 冲 5 级 →
        hunt 5 级以上刷海月 → convert 海月到手：清场后对干将打遗志（100%）或卖干将换绑（红叶染锋 50%）。"""
        if (self._active_comp or {}).get("key") != self.HAIYUE_KEY and self._route_key != self.HAIYUE_KEY:
            return ""
        if "海月" in self._field:
            return "convert" if self.HAIYUE_TEMP in self._field else "done"
        lv = state.level or 1
        if lv < self.HAIYUE_FARM_LEVEL:
            return "raise"
        hand = {sl.get("name") for sl in state.hand_slots or []} | self._hand_names()
        jn3 = sum(1 for e in self.ledger if e.get("event") == "获得卡牌" and e.get("牌") == "3阶效果锦囊")
        if (lv == self.HAIYUE_FARM_LEVEL and "遗志" not in hand and jn3 < self.HAIYUE_FARM_JN
                and (state.round_no or 0) <= self.HAIYUE_FARM_UNTIL):
            return "farm"
        return "rush" if lv < self.HAIYUE_RUSH_LEVEL else "hunt"

    def _hand_names(self) -> set:
        return {n for n, c in self._hand_book.items() if c > 0}

    def decide_haiyue(self, state: GameState) -> Optional[dict[str, Any]]:
        """伞海月的结构性动作（指定打法；登记在 interim_rules：haiyue_plan）。"""
        from agent.decision import interim_rules as _IR
        if not _IR.on("haiyue_plan") or self._shop_blocks_board(state):
            return None
        ph = self.haiyue_phase(state)
        if ph in ("raise", "farm", "rush", "hunt") and self.HAIYUE_TEMP in self._field and self._carry_piece != self.HAIYUE_TEMP:
            # 海月到手前干将是临时主C（前期集中养干将）；枫叶舞换绑由换主C 收尾照常处理
            if self.apply_carry(self.HAIYUE_TEMP, state, why="haiyue_plan"):
                logger.info("【伞海月】%s：干将莫邪当临时主C", ph)
        if ph == "rush" and state.level_cost and 0 < state.level_cost <= state.gold and state.level_btn \
                and state.round_no != self._leveled_round:
            logger.info("【伞海月】%s：冲 %d 级找海月（升级费 %s / 能量 %s）", ph, self.HAIYUE_RUSH_LEVEL, state.level_cost, state.gold)
            return {"type": "buy_xp", "x": state.level_btn[0], "y": state.level_btn[1], "why": "伞海月：冲 5 级找海月"}
        if ph != "convert":
            return None
        if self._carry_piece != "海月":
            self.apply_carry("海月", state, why="haiyue_plan")
        hand = {sl.get("name"): sl for sl in state.hand_slots or []}
        if "遗志" in hand:
            # 遗志：移除指定英雄、等级给 1 名**随机**英雄 → 先清到只剩 海月 + 干将（召唤物 / 图腾不算英雄）
            others = [n for n in self._field if n not in ("海月", self.HAIYUE_TEMP)
                      and self.pieces.get(n, {}).get("category") == "英雄牌"]
            if others:
                v = min(others, key=lambda n: self.unit_value(n) if hasattr(self, "unit_value") else 0)
                logger.info("【伞海月】清场准备打遗志：卖 %s（场上还剩 %d 个别的英雄）", v, len(others) - 1)
                return {"type": "sell_unit", "piece": v, "src": self._field[v], "why": "伞海月：清场后对干将打遗志"}
            sl = hand["遗志"]
            logger.info("【伞海月】场上只剩海月和干将：对干将打遗志（干将等级 100%% 给海月）")
            return {"type": "drag_card", "piece": "遗志", "src": (sl["x"], sl["y"]), "dst": self._field[self.HAIYUE_TEMP],
                    "kind": "效果牌", "target": self.HAIYUE_TEMP, "why": "伞海月：遗志移除干将、等级给海月"}
        if "红叶染锋" in self._lord_boosts and self._maple_bound == self.HAIYUE_TEMP:
            logger.info("【伞海月】没有遗志、有红叶染锋：卖干将，枫叶舞换绑海月（保留 50%%）")
            return {"type": "sell_unit", "piece": self.HAIYUE_TEMP, "src": self._field[self.HAIYUE_TEMP],
                    "why": "伞海月：红叶染锋，卖干将换绑海月"}
        return None

    def haiyue_hold_level(self, state: GameState) -> bool:
        """3 级刷遗志期间不升级（Laya 商店题去掉 levelup、经济网络的 levelup 不执行）。"""
        from agent.decision import interim_rules as _IR
        return _IR.on("haiyue_plan") and self.haiyue_phase(state) == "farm"

    def haiyue_save_equip(self, state: GameState, card: str) -> bool:
        """海月到手之前，干将有自带的双剑，别的装备都存着给海月（拆不下来 / 出生自带的不在此列）。"""
        from agent.decision import interim_rules as _IR
        if not _IR.on("haiyue_plan") or self.haiyue_phase(state) in ("", "done", "convert"):
            return False
        c = self.cards.get(card, {}) if hasattr(self, "cards") else {}
        return self.pieces.get(card, {}).get("category", c.get("category")) == "装备牌" or "装备" in str(c.get("category") or "")

    def decide_economy(self, state: GameState) -> Optional[dict[str, Any]]:
        """是否点升级（花费为按钮上的 level_cost）。

        规则：留 level_reserve 点零钱即可升（血量危险时连这点都不留，全力提战力）。
        """
        if self.__dict__.get("_skip_levelup_round") == state.round_no:
            return None          # 本回合先买了缺的引擎件、钱不够升（10-22 第 916 局）
        if state.lord_maxed or state.level >= meta.MAX_LORD_LEVEL:
            return None                 # 满级：升级按钮已经没了（原位置写着「满级」）
        if self.haiyue_hold_level(state):
            return None                 # 伞海月：3 级刷遗志，先不升
        if self.model_decides() and self.eco_says(state) != "levelup":
            # 2026-10-12 清查：经济网络接管时升不升级只听它的（第 574 局第 9 回合规则先升了级，经济网络随后答的是 roll）
            return None
        cost = state.level_cost
        if cost is None or cost < 0:
            return None
        if state.gold < 0:
            return None                 # 能量读数不可信，禁止一切花费
        if state.round_no == self._leveled_round:
            return None                 # 一回合最多升一次，避免连点把能量掏空
        # 棋手经验型英雄（弈星）的收益**只在升级那一刻结算**，所以升级前要先把它就位：
        #   - 已在手里还没上场 → 先上场
        #   - 还在商店、且这笔钱同时够买它和升级 → 先买
        # 否则这一次升级就白升了（2026-09-22 定的打法）。
        # 人口满 / 钱不够两者兼顾时不等，照常升级，免得互相卡死。
        # **这个等待必须有上限**（2026-09-22 第十二局血的教训）：
        # 原实现只要弈星没上场就无限期不升级，实战连等了两个回合、
        # 期间 decide_buy 把能量全买了牌 —— 回合4 有 13 能量、升级只要 2，硬是没升，
        # 等弈星终于上场时只剩 1 能量。等于先上场再升级变成了一边等一边把钱花光。
        # 现在每回合最多让位 LEVEL_WAIT_FRAMES 帧，之后照升不误。
        # 商店开着读不到人口时，按 pop_now（上次真读到的 + 之后账本的确定变化）判（2026-09-23 第九局：5/5 满了，
        # 却因为"读不到 = 没满"去等弈星上场，弈星根本上不去，这回合的级就没升）
        pop_full = self.pop_full(state)
        if not pop_full:
            waiting = any(self._is_level_engine(sl["name"]) and sl["name"] not in self._field
                          for sl in state.hand_slots)
            why = "手里有棋手经验型英雄没上场，先上场再升级"
            if not waiting:
                for slot in state.shop:
                    if (slot.piece and not slot.sold and self._is_level_engine(slot.piece)
                            and slot.piece not in self._field
                            and state.gold >= (slot.cost or 3) + cost):
                        waiting = True
                        why = f"商店里有棋手经验型英雄 {slot.piece} 且买得起，先买再升级"
                        break
            if waiting:
                if self._level_wait_round != state.round_no:
                    self._level_wait_round, self._level_wait_frames = state.round_no, 0
                self._level_wait_frames += 1
                if self._level_wait_frames <= self.LEVEL_WAIT_FRAMES:
                    logger.info("%s（第 %d/%d 帧）", why,
                                self._level_wait_frames, self.LEVEL_WAIT_FRAMES)
                    # **等的这几帧里，买牌不许把升级费花掉**（2026-09-23 第九局：
                    # 第 5~7 回合 2 级、升级费 2，等弈星的几帧里 decide_buy 把钱全买了牌，
                    # 等满了再想升级已经没钱 —— "一边等一边把钱花光"的老问题又出现了一次）
                    self._level_hold = (state.round_no, cost)
                    return None
                logger.info("等棋手经验型英雄上场已等满 %d 帧，不再等，先把级升了",
                            self.LEVEL_WAIT_FRAMES)
        # 升级**不吃 gold_reserve**。那条底线是给"买牌/刷新"留买卡钱用的，
        # 套在升级上等于永远不升：实测 gold_reserve=10，前期能量 6、升级费 5，
        # 6-5=1 < 10 → 一整局停在 1 级（2026-09-22：钱很多却不升级棋手）。
        # 棋手等级决定人口上限和商店高阶概率，是前期最大的收益杠杆，
        # 留 2 点零钱（够补一张 3 费牌的一半）就够了。
        # **底线必须对应"一笔真能花出去的钱"，不能是个拍脑袋的数**
        # （2026-09-22 第三次出现不升级；前两次只是把数字从 10 改成 2，没改错误本身）。
        # 攻略给的能量阶梯：1-3 回合每回合 6 点，而升级费 5~6 —— 留 2 点零钱就等于
        # 整局升不了级，且那 2 点买不到任何东西（英雄牌 3 点；刷新 2 点但刷完没钱买）。
        # 所以只留两样：
        #   1. 拍卖当回合的举牌钱
        #   2. 商店里确有想买的牌时，留出它的买价（升级和买牌抢同一笔能量，前期战力不能断）
        reserve = self._auction_hold(state)
        # **场面没填满时，买牌钱一定要留** —— 哪怕场上有棋手经验件（弈星）。
        # 原来那条"有弈星就连买牌钱也不留"的例外太激进：2026-09-23 实战第 2 回合
        # 把 6 能量全花在 1→2 级上，而按攻略自己的表，**1→2 既不涨人口（5→5）
        # 也不开新品阶（1→1）**，于是第 3 回合场上还只有 1 个人、能量 0。
        # 原则：空着场面必输，上个杂牌至少能挡刀。
        # 弈星的经验加成是**效率优化**，铺场是**生存问题**，前者不该压过后者。
        board_full = (state.population_valid
                      and state.population >= state.population_cap)
        _here_n, _cap_n, _ = self.pop_now(state)          # 读不到时的口径（2026-10-19：原来直接用上次读数）
        if state.hp > meta.HP_DANGER and (not self._engine_on_field() or not board_full):
            wanted = [sl.cost or 3 for sl in state.shop if self._worth_buying(sl, state)]
            if wanted:
                reserve += min(wanted)
            elif not state.shop and not board_full and not self._engine_on_field() and (
                    (state.population_valid and state.population < state.population_cap)
                    or (not state.population_valid and _cap_n > 0 and _here_n < _cap_n)):
                # **商店关着看不到货，场面又没铺满：也要留一张英雄牌的钱**（2026-09-23 第十三局：
                # 第 3 回合 6 能量拿 5 点从 1 级升 2 级，剩 1 点，人口停在 3/5 —— 留钱只按"商店里
                # 想买的牌"算，而升级那一刻商店关着，列表是空的）。弈星在场时例外：这次升级多吃经验，
                # 且场景18（9-22 第十一局"预留把自己卡死在 1 级"）要求那时照升。
                reserve += 3
        # **等级落后于进度表就先升级**（2026-09-23 第十四局：第 5 回合还是 1 级 —— 第 3 回合被"场面没满先留钱"
        # 拦下、第 4 回合升级费只要 2 却被"拍卖留 5 + 买牌留 3"拦下；攻略原话"二回合升二级，上三级找全阵容"）。
        # 落后时只保留拍卖当回合的举牌钱，其余预留都让位。
        # 只在**场面已经铺满**时生效 —— 场面没满还是先买人（空着场面必输；场景 30/43/86g 守着）
        target = self.live_level_target(state.round_no)
        full_known = board_full or (not state.population_valid and _cap_n > 0 and _here_n >= _cap_n)
        if state.level < target and full_known:
            reserve = self._auction_hold(state)
            if state.gold - cost >= reserve:
                logger.info("等级 %d 落后于第 %d 回合的进度（%d 级），先升级", state.level,
                            state.round_no, target)
        if state.gold - cost < reserve:
            return None                 # 先看买不买得起，再谈按钮在不在（否则每帧刷日志）
        # **到了搜牌等级就"暂缓"升本，不是停掉**（攻略：升本与搜牌本质是资源分配）。
        # **升级的主要价值是让商店刷出更高品阶的角色**，
        # 人口上限只是副产品 —— 所以别把"人口满了"当成唯一的升级理由。
        # 这里的逻辑是：搜牌期优先把钱用在 D 牌上，但下面两种情况仍然该升：
        #   - 人口卡住**且这一级真的涨人口**（5/5/6/6/7/7，只有 2→3、4→5 涨）
        #   - 能量溢出（升完还够刷好几次）—— 这条就是"有余钱就该去开高品阶池"
        # **"搜牌期暂缓升级"只适用于前期**。攻略那句话的语境是"1/2 费核心在 2 级就
        # 100% 能刷到，这时候升本只会稀释牌池"——它说的是开局那几回合。
        # 我把它套成了整局：2026-09-23 实战打到**第 8 回合还停在 2 级**、攒了 17 能量，
        # 人口上限锁死在 5，而对手早就 6~7 人口了，血从 60 掉到 30。
        # 过了 SEARCH_HOLD_UNTIL_ROUND 就不再拦：那时人口和高品阶池的价值远大于牌池浓度。
        if self._in_search_phase(state) and self.eco_says(state) != "levelup":
            # （经济网络说本回合升级时不再按"搜牌期暂缓"拦 —— 第 636 局第 12 回合这里拦住了模型的升级）
            # "人口满了就升级"这条捷径**必须先确认这一级真的涨人口**：
            # 人口上限是 5/5/6/6/7/7（1~6 级），只有 2→3 和 4→5 会涨。
            # 在 3 级人口满时升到 4 级，人口还是 6 —— 白花一次升级费。
            pop_tight = (state.population_valid
                         and state.population >= state.population_cap
                         and meta.level_raises_pop(state.level))
            # "能量溢出就该升本"的门槛**分前后期**：
            #   前期（<= SEARCH_HOLD_UNTIL_ROUND）要求升完还够刷三次（refresh*3），
            #     因为那时钱主要该花在 D 牌上；
            #   过了前期只要求**升完还买得起一张牌**。原来整局都用前期那条高门槛，
            #     结果 2026-09-23 实战打到**第 8 回合还停在 2 级**、攒着 17 能量，
            #     人口锁死 5、商店只出品阶1，血从 60 掉到 30。
            #     升级之后，商店会刷出更高阶的角色——
            #     池子的价值到中期远大于牌池浓度。
            spare = (self.refresh_cost * 3
                     if state.round_no <= self.SEARCH_HOLD_UNTIL_ROUND else 3)
            flush = state.gold - cost >= reserve + spare
            if not (pop_tight or flush):
                logger.info("已到搜牌等级（等级 %d，主力品阶 %d），能量留给 D 牌不升本",
                            state.level, self._carry_quality())
                return None
        if not state.level_btn:
            # 这一帧没在画面上找到"升级"两个字 = 按钮不在（被面板挡住/布局变了）。
            # **绝不能退回写死坐标**：那个位置在英雄详情面板打开时是「出售」按钮，
            # 盲点下去就是把英雄卖了（2026-09-22 第二局实测面板长这样）。
            logger.info("想升级但画面上没找到「升级」按钮，本帧跳过（能量 %d，费用 %d）",
                        state.gold, cost)
            return None
        # _leveled_round / _engine_levels 在 note_executed 里记（2026-09-30 审查：原来在决定时就写，而这个升级常被 Laya /
        # preempt 换掉 —— 弈星被当成"已贡献过升级"估值掉到 5 被卖；别的路径的升级又都不记，一回合可能升两次）
        if self.hold_level_for_carry(state):
            logger.info("3 级还没有主C（第 %d 回合）→ 先不升级，钱留着刷新找主C", state.round_no)
            return None
        self.record_decision("level_up", state, [{"cost": cost}], True)
        return {"type": "buy_xp", "x": state.level_btn[0], "y": state.level_btn[1]}

    # 过了第几回合就不再为"搜牌"暂缓升级（见 decide_economy 里的说明）。
    SEARCH_HOLD_UNTIL_ROUND = 5

    def decide_refresh(self, state: GameState) -> Optional[dict[str, Any]]:
        """刷新商店。前提是**刷完买得起**——否则刷出什么都拿不了，纯属浪费。

        免费次数也是有限资源（靠效果卡获得），同样不能在买不起的时候乱花。
        2026-09-22：只剩 1 点能量还去刷新，刷出神装也买不了。
        """
        if not state.shop_open:
            return None     # 商店关着，刷新/锁定按钮都不在屏幕上，点了就是乱点
        if any(self._worth_buying(s, state) for s in state.shop):
            return None
        # 刷完至少要买得起一张牌（英雄牌 3 能量），否则刷了也白刷
        min_buy = min([s.cost for s in state.shop if s.cost] or [3])
        if state.gold >= 0 and state.gold < min_buy:
            return None
        # **免费刷新不花能量**：不受"给拍卖留钱""等级还刷不出主力品阶"这些限制，
        # 只要刷完买得起一张就用（2026-09-23：免费刷新一直没被利用 ——
        # 几乎每回合都送免费刷新，原来读不出次数，整局一次没用）。
        if state.free_refresh > 0:
            return {"type": "refresh_shop"}
        # **连刷上限**（2026-09-24：后期回合会为了找主C 无限刷新）：
        # 付费刷新连续 REFRESH_STREAK_MAX 次一张都没买，这回合就停手 —— 钱留给升级/下回合。
        if (self._refresh_streak[0] == state.round_no
                and self._refresh_streak[1] >= self.REFRESH_STREAK_MAX):
            return None
        if state.gold >= 0 and state.gold - min_buy < self._auction_hold(state):
            return None                 # 拍卖就在眼前，这点钱留着举牌
        # **低等级不要强行刷高品阶**（攻略明确点名的前期大坑）：主力品阶在当前
        # 等级根本不会出现时，刷新就是在抽必然抽不到的东西，钱该先拿去升本。
        q = self._carry_quality()
        if q and state.level > 0 and not meta.can_appear(q, state.level):
            if state.level_cost is not None and state.level_cost > 0:
                logger.info("主力品阶 %d 在 %d 级刷不出来，先升本再搜牌", q, state.level)
                return None
        if state.free_refresh > 0:
            return {"type": "refresh_shop"}
        # 费用以界面读数为准（按钮上就写着），读不到才用配置里的估值
        cost = state.refresh_cost if state.refresh_cost >= 0 else self.refresh_cost
        # 底线同样只认"拍卖预留 + 刷完还买得起一张牌"，不再叠静态的 gold_reserve
        # （静态底线 10 意味着 12 能量以下永远不刷新，而搜牌期的钱本来就该花在刷新上）。
        if state.gold >= 0 and state.gold - cost - min_buy >= self._auction_hold(state):
            return {"type": "refresh_shop", "cost": cost}
        return None

    REFRESH_STREAK_MAX = 4

    def _late_roles(self) -> list:
        return list((self._active_comp or {}).get("late") or [])

    def est_price(self, piece: str, read_cost) -> tuple[int, bool]:
        """商店价：读到就用读到的；读不到用卡牌库标价（英雄牌统一 3），库里也没有按 1 估。→ (价格, 是否估的)"""
        if read_cost is not None:
            return int(read_cost), False
        info = self.pieces.get(piece, {})
        if info.get("category") == "英雄牌":
            return 3, True
        v = info.get("cost")
        return (int(v) if isinstance(v, (int, float)) and v > 0 else 1), True

    def _is_keeper(self, name: str) -> bool:
        """**要留到最后、值得凑觉醒的场上英雄**：推荐阵容的后期角色，或者场上估值靠前的非消耗件。
        消耗件（弈星/吃登场效果的杂牌）、过了期的 1 阶不算。"""
        if name not in self._field or self._is_disposable(name):
            return False
        if self._is_engine_part(name):
            return True          # 引擎零件留到最后（狂铁/孙膑这种 1 阶产出方过了前期也不算过期）
        if self._obsolete(name, self._round_seen):
            return False
        if self._is_carry(name) or name in self._late_roles():
            return True
        vals = sorted((self.unit_value(n) for n in self._field), reverse=True)
        return len(vals) > 0 and self.unit_value(name) >= vals[min(len(vals) - 1, 3)]   # 场上估值前 4

    def awaken_left(self, name: str, state: GameState) -> Optional[int]:
        """离觉醒还差几张同名（0 = 已觉醒 / 圈满；None = 读不到）。看徽章外圈（视觉），读不到用上次读到的。"""
        if name in self._ever_awakened:
            return 0
        pos = self._field.get(name)
        if pos is None:
            return None
        p = (state.awaken_progress or {}).get(cell_key(tuple(pos)))
        if p is None:
            p = self._awaken_seen.get(name)
        else:
            self._awaken_seen[name] = p
        return None if p is None else max(0, 3 - p)

    def board_facts(self, state: GameState) -> dict:
        """场上每人的**事实**（给 Laya 的输入 + 老师标注用，实战与离线同源：主循环每帧写进 states.jsonl 的 "board"）。
        {卡名: {"cell", "lv"(徽章等级), "awk_left"(觉醒还差几张，None=读不到), "keep"(留到最后), "late"(后期阵容角色)}}"""
        lv = state.unit_levels or {}
        eq = getattr(state, "equip_counts", None) or {}          # f6：装备件数（感知读数，按格）
        # f9i：这次上场是第几回合（第 12 局 R14：刚上场的雅典娜 7 秒后就被 swap 卖掉，题面看不出"刚上场"）
        for n in [k for k in self._entered if k not in self._field]:
            self._entered.pop(n)
        if state.round_no:
            for n in self._field:
                self._entered.setdefault(n, state.round_no)
        return {n: {"cell": cell_key(tuple(p)), "lv": lv.get(cell_key(tuple(p))), "in_rd": self._entered.get(n),
                    "eq": eq.get(cell_key(tuple(p))) if isinstance(eq, dict) else None,
                    "awk_left": self.awaken_left(n, state), "keep": self._is_keeper(n),
                    "late": n in self._late_roles(), "pool_left": self.pool_left.get(n)}
                for n, p in self._field.items()}

    def decide_lock(self, state: GameState) -> Optional[dict[str, Any]]:
        """锁定商店：钱不够但店里有主力/核心英雄 → 锁到下一回合（每回合最多锁一次）。"""
        if not state.shop_open or self.lock_by_laya:
            return None     # 商店关着，锁定按钮不在屏幕上；学生题面带锁店选项时由学生决定
        if state.round_no == self._locked_round:
            return None
        wanted = [
            s for s in state.shop
            if s.piece and not s.sold
            and (self._is_carry(s.piece) or self._is_core(s.piece))
            and (s.cost is None or s.cost > state.gold)
        ]
        if wanted:
            self._locked_round = state.round_no
            return {"type": "lock_shop", "pieces": [s.piece for s in wanted]}
        return None

    def _worth_buying_ignore_gold(self, slot: ShopSlot, state: GameState) -> bool:
        """想买但买不起时的价值判断（不看能量、不看手牌上限，只看卡本身值不值）。"""
        if self.recently_sold(slot.piece or ""):
            return False             # 刚卖掉的英雄别马上买回来（2026-09-30）
        if (slot.piece and slot.piece not in self._field and slot.piece != self.route_carry()
                and self.pieces.get(slot.piece, {}).get("category") == "英雄牌"
                and self.pop_full(state)
                and self.lineup_complete()):     # 商店开着时人口读数无效 → 用账本人数（审查 10-09）
            return False             # 阵容已成型：新英雄上不了场（2026-10-09）
        if slot.piece is None or slot.sold:
            return False
        info = self.pieces.get(slot.piece)
        if info is None:
            return False
        if slot.faction and slot.faction == state.banned_faction:
            return False
        if self.main_factions and slot.faction in self.main_factions:
            return True
        return info.get("priority", 0) >= 5

    # ---------- 购买 ----------

    def _owned_copies(self, piece: str, state: GameState) -> int:
        """已拥有的同名英雄数（场上 + 手牌）——升级材料价值。"""
        n = sum(1 for h in state.hand if h == piece)
        if piece in self._field:
            n += 1
        return n

    def explain(self, state: GameState) -> None:
        """每回合打一次**局面总览 + 决策依据**。

        2026-09-22：日志需要更多分析信息。
        原来的日志只有动作流水，看不出"为什么这么决定"，每次排查都要翻代码倒推。
        这里把决策真正依赖的量一次性摊开，让日志自己能回答"当时它是怎么想的"。
        """
        if state.round_no == self._explained_round or state.round_no <= 0:
            return
        self._explained_round = state.round_no

        comp = self._active_comp or {}
        carry = comp.get("carry") or "?"
        # 回合头是最常看的一行诊断。商店开着时人口那块被卡面盖住 → population_valid=False，
        # 原来直接打"读不到"，于是**每一回合的头都在说读不到**（回合头正好在开店时打），
        # 场面填没填满这件最关键的事在日志里完全看不见。
        # 决策仍然只认本帧真读到的值（见 decide_deploy），这里只是把上一次读到的显示出来。
        _ph, _pc, _ps = self.pop_now(state)
        if state.population_valid:
            pop = f"{state.population}/{state.population_cap}"
        elif _pc > 0:
            pop = f"{_ph}/{_pc}({_ps})"          # 2026-10-19：原来显示"上次读到"的旧值，卖人 / 觉醒回手后还是旧数
        else:
            pop = "读不到"
        logger.info("┌─ 回合 %d ─ 血 %d ─ 能量 %d ─ 等级 %d(升级费 %s) ─ 人口 %s ─ 阵容 %s(主C %s)",
                    state.round_no, state.hp, state.gold, state.level,
                    state.level_cost, pop, comp.get("faction", "?"), carry)

        field = sorted(self._field, key=self.unit_value, reverse=True)
        eq = state.equip_counts or {}
        logger.info("│  场上 %d: %s", len(field), " ".join(
            f"{n}(分{self.unit_value(n)}"
            f"{'/装' + str(eq[cell_key(self._field[n])]) if cell_key(self._field[n]) in eq else ''}"
            f"{'/主C' if self._is_carry(n) else '/核心' if self._is_core(n) else ''}"
            f"{'/消耗' if self._is_disposable(n) else ''})"
            for n in field) or "空")

        hand = []
        for slot in state.hand_slots:
            n = slot["name"]
            cat = self.pieces.get(n, {}).get("category", "未知")
            if cat == "英雄牌":
                hand.append(f"{n}[{cat}分{self.unit_value(n)}"
                            f"{'·已在场' if n in self._field else ''}]")
            else:
                hand.append(f"{n}[{cat}]")
        logger.info("│  手牌 %d/%d: %s", len(state.hand_slots), HAND_MAX,
                    " ".join(hand) or "空")

        if state.shop_open and state.shop:
            logger.info("│  商店: %s", " ".join(
                f"{sl.piece or '?'}"
                f"{'✓买' if sl.piece and self._worth_buying(sl, state) else '✗'}"
                for sl in state.shop))

        # 为什么升级 / 为什么不升级
        q = self._carry_quality()
        notes = []
        if q:
            notes.append(f"主C品阶{q}→搜牌等级{meta.search_level(q)}"
                         f"{'(已进入搜牌期)' if self._in_search_phase(state) else ''}")
            if state.level > 0 and not meta.can_appear(q, state.level):
                notes.append(f"⚠{state.level}级刷不出品阶{q}")
        left = meta.rounds_to_auction(state.round_no)
        if left >= 0:
            notes.append(f"距拍卖{left}回合(留{self._auction_hold(state)}能量)")
        if self._maple_bound:
            notes.append(f"枫叶舞绑在{self._maple_bound}")
        if notes:
            logger.info("└  %s", " │ ".join(notes))

    # 6 回合之后，弈星这种 1 星英雄太弱，不再上场。
    LOW_TIER_CUTOFF_ROUND = 6

    def _obsolete(self, name: str, round_no: int) -> bool:
        """过了前期还是 1 阶的英雄：不再上新的、手里的卖掉、场上的优先换下。
        **路线这个阶段点名要的不算**（2026-09-29：守约路线 18 局里李元芳只上过 1 次 —— 它是古币的产出方，
        官方路线第 4~7 回合要它、有的路线终局也要它，却被这条规则当成过渡件）。"""
        return (round_no > self.LOW_TIER_CUTOFF_ROUND
                and self.pieces.get(name, {}).get("category") == "英雄牌"
                and self.pieces.get(name, {}).get("quality", 0) == 1
                and name not in self._ever_awakened
                and name not in self.route_stage_members(round_no)
                # 主C 用得上的不算（1 阶英雄如果主C 能用上，仍可上场：李元芳给河洛主C 产古币等）
                and not self.feeds_carry(name))

    ROUTE_ADDONS_N = 2
    _TEAM_GROWTH: Optional[dict] = None

    LATE_REVIVE_ROUND = 9

    def revive_support(self) -> list:
        """**单核打到后期，配一个复活**（单核打到后期，加一个带复活技能的辅助性价比很高）：
        第 9 回合起（4 级后才刷得出 4 阶的太乙）要 太乙真人（开团：距离最近的 1 名英雄获得复生；已有"贴主C 站位 / 不给装备"）
        + 天赋「战前叫阵」（双方等级最高者比等级，赢了获得完美复生 —— 单核等级最高）；
        主C 是【河洛】的再要 度命之护（贤者的庇护：河洛英雄首次被击败后复生）。"""
        if (self._round_seen or 0) < self.LATE_REVIVE_ROUND:
            return []
        out = ["战前叫阵"]
        if "太乙真人" not in self._field:
            out.insert(0, "太乙真人")
        carry = self._carry_piece or (self._active_comp or {}).get("carry") or ""
        if "河洛" in (self.pieces.get(carry, {}).get("synergies") or []):
            out.append("度命之护")
        return out

    HIGH_APS = 1.4                # 40 级每秒普攻次数 ≥ 这个算"高攻速"（hero_strength：狄仁杰 2.0 / 莱西奥 1.6 / 艾琳 1.5 / 虞姬、公孙离 1.46 / 吕布 1.43）
    SELF_APS_RE = re.compile(r"(自身|其|获得自身等级×\d+%的)攻击速度|攻击速度\+\d+%")

    def high_attack_speed(self, name: str) -> bool:
        """按数据判断是不是高攻速（普攻次数多）：强度模型的 40 级攻速，或卡面写着给自己加攻速。"""
        if RuleEngine._STRENGTH is None:
            try:
                RuleEngine._STRENGTH = (json.loads((Path(__file__).resolve().parents[2] / "data" / "hero_strength.json")
                                                   .read_text(encoding="utf-8")).get("heroes") or {})
            except (OSError, ValueError):
                RuleEngine._STRENGTH = {}
        aps = ((RuleEngine._STRENGTH or {}).get(name) or {}).get("lv40", {}).get("aps") or 0
        return aps >= self.HIGH_APS or bool(self.SELF_APS_RE.search(self.pieces.get(name, {}).get("skill", "") or ""))

    _ATTACK_ROLL: Optional[list] = None

    def attack_roll_summoners(self) -> list:
        """**高攻速主C + "按普攻次数给全队涨级"的图腾 = 滚等级**（高攻速英雄和少司缘可以形成滚等级的配合）。
        只读卡面：召唤物的效果是「你的单位每合计进行 N 次普攻，所有己方英雄等级永久+1」→ 找卡面写着「召唤<它>」的英雄（现在是少司缘 → 因缘图腾；
        她的技能还给沿途友军 +30% 攻速）。主C（路线主C 或现任）是高攻速时，把召唤者列为路线补件（买 / 留）。"""
        carry = self._carry_piece or ((self._active_comp or {}).get("carry") if self.route_locked() else "")
        if not carry or not self.high_attack_speed(carry):
            return []
        if RuleEngine._ATTACK_ROLL is None:
            totems = [n for n, v in self.pieces.items() if v.get("category") == "召唤物"
                      and re.search(r"每合计进行\d+次普攻.{0,12}等级", v.get("skill", "") or "")]
            RuleEngine._ATTACK_ROLL = [n for n, v in self.pieces.items() if v.get("category") == "英雄牌"
                                       and any(f"召唤{t}" in (v.get("skill", "") or "") for t in totems)]
        return [n for n in RuleEngine._ATTACK_ROLL if n != carry]   # 被禁派系的本来就刷不出来

    _MIN_ENGINE: Optional[dict] = None
    MIN_ENGINE_UNTIL = 11                # 第 11 回合（含）以前：最低配套先上、后期件（太乙这类）不抢格子（10-18：第 12 回合前太乙降低优先级；原 8）

    def min_engine(self) -> dict:
        """当前路线的**最低配套**（data/route_min_engine.json；先凑最低配套，再去凑其他英雄；
        太乙这类前期不急，后期再补，先上场核心引擎）。没锁路线 / 没收录返回 {}。"""
        if RuleEngine._MIN_ENGINE is None:
            try:
                RuleEngine._MIN_ENGINE = json.loads((Path(__file__).resolve().parents[2] / "data" / "route_min_engine.json")
                                                    .read_text(encoding="utf-8"))
            except Exception:                                   # noqa: BLE001
                RuleEngine._MIN_ENGINE = {}
            try:        # 2026-10-19：补 5 条路线（伞亚连 / 守约三条 / 瞬步狄仁杰）
                for k, v in json.loads((Path(__file__).resolve().parents[2] / "data" / "route_min_engine_f9v.json")
                                       .read_text(encoding="utf-8")).items():
                    if not k.startswith("_"):
                        RuleEngine._MIN_ENGINE[k] = dict(dict(v, **(v.get("f9w_override") or {})), **(v.get("f9z_override") or {}))   # 10-20 f9z_override   # 覆盖（10-19：除沈梦溪 / 狄仁杰外明世隐 > 李元芳）
            except Exception:                                   # noqa: BLE001
                pass
        key = str(self._route_key or getattr(self, "forced_route", "") or "")
        me = (RuleEngine._MIN_ENGINE or {}).get(key) or {}
        if not me and key:
            # 2026-10-21 审查：锁上的阵容没收录（第 851 局"阿离伞亚连" 75491133230282772、主C 组跟随换到沈梦溪 618167530505238）
            # → 引擎类规则（engine_tier_eco / lock_for_carry_copy / final_team_first）整局失效。借同一主C 收录过的阵容：配置里的核心阵容优先。
            from agent.decision import lord_config as _LCm, routes as _Rm
            carry = (_Rm.by_key(key) or {}).get("carry") or _LCm.load().carry_of_lineup(key)
            if carry:
                pref = _LCm.load().lineup_key(carry)
                tabs = RuleEngine._MIN_ENGINE or {}
                alt = pref if (tabs.get(pref) or {}).get("carry") == carry else next(
                    (k for k, v in tabs.items() if isinstance(v, dict) and v.get("carry") == carry), "")
                me = dict(tabs.get(alt) or {}, borrowed_from=alt) if alt else {}
        if me.get("core_set"):
            # 2026-10-12（核心引擎一般只有 3 个（包括主C），如虞姬配个少司缘就算初步成型）：规则层用 ≤3 人的核心引擎
            names = [n for u in me["core_set"] for n in self.pieces if self.pieces[n].get("category") == "英雄牌"
                     and n in str(u.get("unit") or "")]
            me = dict(me, min_set=me["core_set"], early_priority=names or me.get("early_priority") or [],
                      # 10-20：完整名单（含雅典娜 / 李元芳等），价值类临时规则用这份，与题面一致；
                      # 10-21 第 855 局：再并上核心引擎（core_set）——沈梦溪炸弹猫线 core_set 有铠、early_priority 没有，
                      # final_team_first 把铠当外人，R14 卖铠给武则天、卖李元芳给铠、再卖铠给上官婉儿
                      early_priority_full=list(dict.fromkeys(list(me.get("early_priority") or []) + names)))
        return me

    def _flex_carries(self) -> set:
        """指定路线清单（首选 + 备选）里各条路线的主C。"""
        from agent.decision import routes as _Rf
        keys = ((self.__dict__.get("_forced_orig") or self.forced_route),) + tuple(self.forced_backups or ())
        return {(_Rf.by_key(k) or {}).get("carry") for k in keys if k} - {None, ""}

    def spare_swap(self, state: GameState) -> Optional[dict[str, Any]]:
        """**备用卡给路线配件让位**（临时规则 swap_spare_for_route：备用卡可以上场，总比场上空着好；拿到路线配件后要卖掉备用卡）。
        条件：准备阶段、商店关着、人口满；手里有**当前路线阵容里**的英雄在等（不在场上）；场上有**备用卡**
        （别的路线的主C / 不在当前路线阵容里的人，不是主C、不是枫叶舞宿主、没觉醒）→ 卖掉最弱的备用卡。每回合最多一次。"""
        if state.phase != "prepare" or state.shop_open or not self.route_locked():
            return None
        if not (state.population_valid and state.population >= (state.population_cap or 99)):
            return None
        if self.__dict__.get("_spare_swap_round") == state.round_no:
            return None
        team = self._route_team()
        waiting = [sl["name"] for sl in state.hand_slots or []
                   if sl.get("name") in team and sl.get("name") not in self._field
                   and self.pieces.get(sl.get("name"), {}).get("category") == "英雄牌"]
        if not waiting:
            return None
        spares = [n for n in self._field if n not in team and n != self._carry_piece and n != self._maple_bound
                  and n not in self._ever_awakened and self.pieces.get(n, {}).get("category") == "英雄牌"
                  and not (self.pieces.get(n) or {}).get("no_pop")]
        if not spares:
            return None
        victim = min(spares, key=lambda n: (self.unit_value(n), self._unit_lv.get(n) or 0))
        self._spare_swap_round = state.round_no
        logger.info("备用卡给路线配件让位：卖 %s，给手里的 %s 腾位（临时规则 swap_spare_for_route）", victim, waiting[0])
        return {"type": "sell_unit", "piece": victim, "src": self._field[victim],
                "why": f"备用卡给路线配件 {waiting[0]} 让位"}

    def duo_active(self) -> set:
        """共战两人的保护（必须同时上场、不卖）只在**主C 本人就是其中之一**时生效（2026-10-14 第 626 局：主C 守约时貂蝉 / 上官婉儿占 2 格）。"""
        duo = self.duo_members()
        return duo if (duo and self._carry_piece in duo) else set()

    def late_pieces(self) -> set:
        """后期补位件：路线 min_engine 的 late_fill + 卡面"给别人复生"的（太乙真人）。
        太乙的复活作用到后期效果才明显——不管哪条路线都算（第 737 局沈梦溪路线的 late_fill 里没有太乙，R10 卖杨玉环上太乙）。"""
        rev = self.__dict__.get("_revive_givers")
        if rev is None:
            rev = self._revive_givers = {n for n, c in self.pieces.items()
                                         if c.get("category") == "英雄牌" and re.search(r"使[^。]{0,12}获得复生", c.get("skill") or "")}
        # 2026-10-18 r9zw 老师：route_min_engine.json 的 late_fill 不可靠（云中君 / 少司缘 / 钟馗 都是成长 / 牺牲件）→ 只认卡面"给别人复生"的（即太乙）
        return set(rev)

    def engine_piece_rank(self, name: str, round_no: int) -> int:
        """上场排序用：最低配套 +1；后期件在第 MIN_ENGINE_UNTIL 回合以前 -1；其他 0。"""
        if name in self.duo_active():
            return 2                     # 共战天赋的两人必须同时上场（2026-10-12；2026-10-14 起只在主C 是其中之一时）
        me = self.min_engine()
        if not me:
            return 0
        if name in (me.get("early_priority") or []):
            return 1
        if name in self.late_pieces() and 0 < round_no <= self.MIN_ENGINE_UNTIL:
            return -1
        return 0

    def route_addons(self) -> list:
        """路线的**顺带养 T 补件**前 N 个（data/team_growth.json，2026-09-30 研究：养主C / 核心的同时效果覆盖全队或前排，
        坦克顺带变强，不用单独给 T 预留资源）。"""
        if not self._route_key:
            return []
        if RuleEngine._TEAM_GROWTH is None:
            try:
                RuleEngine._TEAM_GROWTH = json.loads((Path(__file__).resolve().parents[2] / "data" / "team_growth.json")
                                                     .read_text(encoding="utf-8"))
            except Exception:                              # noqa: BLE001
                RuleEngine._TEAM_GROWTH = {}
        r = (RuleEngine._TEAM_GROWTH.get("routes") or {}).get(str(self._route_key)) or {}
        out = [a["name"] for a in (r.get("add") or [])[:self.ROUTE_ADDONS_N] if a.get("name")]
        out += [n for n in self.revive_support() if n not in out]
        out += [n for n in self.attack_roll_summoners() if n not in out]
        partners = list(r.get("partners") or [])
        if partners:
            # **搭档**（玄策要另一个闪现英雄）：还没有就全列上、谁先出现要谁；有了一个就只留它，不再追别的
            have = [n for n in partners if n in self._field or self._hand_book.get(n, 0) > 0]
            out += [n for n in (have[:1] or partners) if n not in out]
        return out

    def route_stage_members(self, round_no: int) -> set:
        """路线锁定后，**这个回合**官方路线点名要的英雄：当前阶段的主力/替补 + 最终阵容 + 主C。
        阶段没写回合（0-0）时只算最终阵容。没锁路线返回空集。"""
        if not self.route_locked() or not self._route_key:
            return set()
        from agent.decision import routes as _R
        lu = _R.by_key(self._route_key)
        if not lu:
            return set()
        out = set(_R.final_team(lu)) | ({lu["carry"]} if lu.get("carry") else set())
        st = _R.stage_now(lu, round_no or 0) if any((x.get("to_round") or 0) > 0 for x in lu.get("stages") or []) else None
        if st:
            out |= set(st.get("main") or []) | set(st.get("sub") or [])
        out |= {n for n in self.route_addons() if self.pieces.get(n, {}).get("category") == "英雄牌"}   # 顺带养 T 的补件英雄
        return out

    FIT_SUPPORTS = 8             # 现任主C 的配套前几个算"适合主C"

    def carry_fit_heroes(self) -> set:
        """**适合现任主C 的英雄**（定了主C 之后切换阵容要果断：换成新主C 用得上的牌，不必拘泥于某一个官方阵容）：
        路线核心 + 主C 配套前 FIT_SUPPORTS 个（官方阵容同台 / 关系图涨级回蓝 / 高强度辅助）。"给主C 涨等级的"由 feeds_carry 单独判。"""
        if not self.carry_settled():
            return set()
        return self.route_core() | set(self.carry_supports_now()[:self.FIT_SUPPORTS])

    def dead_weight(self, name: str) -> bool:
        """场上**可以换掉的人**：不是主C / 主坦 / 路线核心 / 主C 配套，不给主C 涨等级，没觉醒，等级低（≤10 或不到主C 的 30%）。"""
        if not self.carry_settled() or name not in self._field or name == self._carry_piece:
            return False
        if name in self.carry_fit_heroes() or self.feeds_carry(name) or name in self._ever_awakened:
            return False
        lv = {**self._unit_lv, **(self.__dict__.get("_forced_lv") or {})}
        mine = lv.get(name) or 0
        if self._is_tank(name):
            tanks = sorted((lv.get(n) or 0 for n in self._field if self._is_tank(n)), reverse=True)
            if tanks and mine >= tanks[0]:
                return False             # 主坦（要养主坦）
        cl = lv.get(self._carry_piece or "") or 0
        return mine <= max(10, cl * 0.3)

    def route_core(self) -> set:
        """路线核心：主C + 最终阵容（不含阶段替补）。"""
        if not self.route_locked() or not self._route_key:
            return set()
        from agent.decision import routes as _R
        lu = _R.by_key(self._route_key) or {}
        return set(_R.final_team(lu)) | ({lu["carry"]} if lu.get("carry") else set())

    # 分摊升级牌：等级分给多个人、主C 只拿到一部分（如自己升 2 级、其他升 1 级的分摊升级牌，不包含古币）。
    # 集中给 1 人的（古币 / 精致古币 / 特训战术 / 高阶特训战术 / 八方来援）不算
    SPREAD_LEVEL_CARDS = ("简易协作", "精密协作", "有福同享", "阵营祝福", "演练战术", "高阶演练战术", "图腾祝福", "水穷云起")

    def is_level_card(self, name: str) -> bool:
        """core_before_level_cards 要让位的"分摊升级牌"。"""
        return name in self.SPREAD_LEVEL_CARDS

    def route_core_missing(self, state: GameState) -> set:
        from agent.decision import routes as _R
        owned = set(self._field) | {sl["name"] for sl in state.hand_slots or []} | self.hand_book_names()
        ban = _R.norm_banned(state.banned_faction or "") if state.banned_faction else ""
        from agent.decision import lord_config as _LCs2
        _sub = _LCs2.load().substitutes            # 互相替代的核心件（朵莉亚 ↔ 海诺，10-21）：有一个就不算缺
        return {n for n in self.route_core() if n not in owned
                and self.pieces.get(n, {}).get("category") == "英雄牌"
                and not (ban and ban in (self.pieces.get(n, {}).get("synergies") or []))
                and not any(x in owned for x in _sub(n))}

    def level_spare(self, state: GameState) -> Optional[int]:
        """'花完下回合还能升级'的额度（题面 f9ll "can spend Xe now and still level up next round" 同一算法）：
        下回合就是进度点（当前等级 < 下回合进度线）、攒着下回合升得起 → 返回现在最多能花多少；否则 None（不设额度）。
        2026-10-21 eco876 分析：进度点前一回合（R3/R6/R9）eco 判 roll 后实际花超额度 58/89 题，下回合升不起又晚一回合。"""
        rnd, lv, cost, g = state.round_no or 0, state.level or 0, state.level_cost or 0, state.gold
        if not rnd or not lv or cost <= 0 or not isinstance(g, int) or state.lord_maxed or lv >= meta.MAX_LORD_LEVEL:
            return None
        if rnd < 2:
            return None              # 第 1 回合先买人补空格（54.20-5 / 54.33-4），不按升级额度卡
        if lv >= meta.level_target(rnd + 1):
            return None
        nc = cost      # 10-23 第 987 局：按"打完自动给经验、下回合升级费会降"算，实测升级费常常不降（R6/R7 都是 16）→ 差 1 升不起；不再扣自动经验
        if nc <= 0:
            return None
        spare = g + meta.energy_income(rnd + 1) - nc
        return max(0, min(g, spare)) if spare >= 0 else None

    def route_core_missing_now(self, state: GameState) -> set:
        """现在真缺、该先凑的核心件：route_core_missing 去掉第 12 回合前的后期件（太乙，late_fill_before_r12 / 规范 54.29-6）
        和当前棋手等级刷不出的品阶（和 eco_roll_follow 同口径）。2026-10-21 审查：core_first_no_copies 用全量缺件，
        太乙第 12 回合前不急买 → "核心缺人"一直成立，铠 / 盾山 / 钟馗同名整段不买（852a 局 3 次）。"""
        from agent.decision import meta as _mt
        lf = self.late_pieces()            # 10-21 审查 8：后期件（太乙）任何回合都不算"真缺"
        return {x for x in self.route_core_missing(state) if x not in lf
                and _mt.can_appear(int((self.pieces.get(x) or {}).get("quality") or 0), state.level or 1)}

    def engine_missing_unreachable(self, state: GameState) -> list:
        """最低配套（引擎核心件）还缺、而且当前棋手等级根本刷不出的件（2026-10-21 第 883 局：凑核心阵容要结合剩余英雄的品阶，
        3 级商店一直刷新，但缺的都是 4 阶英雄）。缺的引擎件里只要有一个现在刷得出 → 返回 []（刷新还有用）。"""
        if not self.route_locked():
            return []
        from agent.decision import meta as _mt, lord_config as _LCu
        me = self.min_engine() or {}
        eng = set(me.get("early_priority") or []) - {self.route_carry(), self._carry_piece, "", None} - self.late_pieces()
        eng |= set(self.route_core_missing(state)) - {self.route_carry(), self._carry_piece, "", None} - self.late_pieces()   # 10-22：路线核心件也算（铠 / 明世隐刷不出时先升级）
        own = set(self._field) | {sl.get("name") for sl in state.hand_slots or []} | set(self.hand_book_names())
        sub = _LCu.load().substitutes
        miss = [n for n in eng if n not in own and not any(x in own for x in sub(n))]
        if not miss:
            return []
        lv = state.level or 1
        if any(_mt.can_appear(int((self.pieces.get(n) or {}).get("quality") or 0), lv) for n in miss):
            return []
        return sorted(miss)

    def _unreachable_engine_rush(self, state: GameState) -> bool:
        """临时规则 engine_tier_rush（亚连路线要先养亚连，还要早点拿到雅典娜，按棋手等级算刷出概率）：
        缺的引擎件当前等级刷不出（3 级缺 4 阶雅典娜）→ 第 5~10 回合钱够就升级（4 级每次刷新约 4.4% 见到雅典娜、5 级约 6.5%；
        亚连局实测第 8~9 回合才到 4 级，晚于进度第 7 回合）。每回合升一次，升完照常交给经济网络 / Laya。"""
        from agent.decision import interim_rules as _IRr
        if not (_IRr.on("engine_tier_rush") and state.phase == "prepare" and state.level_btn and not state.lord_maxed
                and 4 <= (state.round_no or 0) <= 10 and state.level_cost and 0 < state.level_cost <= (state.gold or 0)
                and self._leveled_round_n(state) == 0):
            return False
        return bool(self.engine_missing_unreachable(state))

    def final_board(self, state: GameState) -> Optional[dict]:
        """最终可摆阵容（天赋硬条件 *_final 类用；2026-10-21 第 853 局同心协力：人口 7 − 明世隐 − 太乙 = 河洛 5）。
        路线没锁 → None（不判）。must = 主C + 核心引擎件（min_engine early_priority_full）+ 路线里的后期件（太乙）；cap = 满级人口 + 人口天赋。"""
        if not self.route_locked():
            return None
        carry = self.route_carry() or self._carry_piece or ""
        team = set(self._route_team()) | ({carry} if carry else set())
        me = self.min_engine() or {}
        eng = set(me.get("early_priority_full") or me.get("early_priority") or [])
        must = ({carry} | eng | (self.late_pieces() & team)) - {""}
        final = team | must
        pool = final | set(self._field) | {sl["name"] for sl in state.hand_slots or []}
        return {"final": final, "must": must, "pool": pool,
                "cap": max(meta.POP_CAP_BY_LEVEL.values()) + int(self.pop_bonus or 0)}

    def route_stage_subs(self, round_no: int) -> set:
        """当前阶段只点名为"替补"（sub）、不在主力 / 最终阵容 / 主C 里的人。"""
        if not self.route_locked() or not self._route_key:
            return set()
        from agent.decision import routes as _R
        lu = _R.by_key(self._route_key)
        if not lu or not any((x.get("to_round") or 0) > 0 for x in lu.get("stages") or []):
            return set()
        st = _R.stage_now(lu, round_no or 0) or {}
        core = set(_R.final_team(lu)) | set(st.get("main") or []) | ({lu["carry"]} if lu.get("carry") else set())
        return set(st.get("sub") or []) - core

    def unit_value(self, name: str) -> int:
        """一个英雄现在值多少（只用来做**相对比较**：该卖谁、该不该买）。

        排序意图：阵容主力 > 阵容核心 > 高品阶 > 低品阶 > 用完就卖的消耗件。
        品阶权重故意压得比"阵容内"低 ——
        **低品阶不等于弱**，成型快、觉醒容易，成型牌组的核心常常不是 5 阶。
        """
        info = self.pieces.get(name, {})
        if info.get("category") != "英雄牌":
            return -1
        # **棋手经验件（弈星）贡献过升级之后，不再享受"阵容核心"的保护**：
        # 它在某些官网阵容里确实被列为 core，于是拿到 60 分、被换人逻辑当成不能卖的人。
        # 但它的打法是升一次棋手基本就可以卖掉 —— 收益兑现后它就是个待换的位置。
        if self._is_level_engine(name) and self._engine_levels.get(name, 0) >= 1:
            return 5
        # **觉醒过的英雄**：3 张合成才换来的，比任何普通卡都值钱（金色的卡是觉醒卡，
        # 要优先上场）。不加这条的话，人口满时腾位逻辑
        # 会觉得它不比场上的人强，于是它一直烂在手里。
        bonus = 40 if name in self._ever_awakened else 0
        if self._obsolete(name, self._round_seen) and not self._is_carry(name) and not bonus:
            return 3            # 觉醒过的不算过渡件（第十四局：觉醒曜按"过时 1 阶"估值 3，被换绑收尾卖掉）
        if self._is_carry(name):
            return 100 + bonus
        if self._is_core(name):
            return 60 + bonus
        if self._is_disposable(name) and not bonus:
            return 1
        # 品阶 ×6 + **能不能当主C** 的先验（455 套推荐阵容里当主C 的热度，0~20）。
        # 原来只有品阶 ×4：4 阶的艾琳（热门主C，4.1M）= 26 分，场上 1 阶杂牌 14 分，
        # 差值没过换人门槛 15 —— 于是"换不下任何人"，放了一回合就被卖掉（2026-09-23 实战）。
        prior = round(20 * self._any_carry_freq.get(name, 0.0))
        # **实际等级也算钱**：合成/成长攒下来的等级，卖掉就没了（原来估值完全不看等级）
        lv = min(self._unit_lv.get(name, 0) or 0, 60)
        return 10 + info.get("quality", 0) * 6 + prior + bonus + int(lv * 0.5)

    MIN_FRONT_TANKS = 2          # 卖人换后排卡时，场上至少留这么多坦克

    def _weakest_on_field(self, allow_core: bool = False, incoming: str = ""):
        """场上**能换下去的**单位里最弱的那个（名字, 位置, 分值）。没有返回 None。

        `allow_core`：给**觉醒卡**腾位时，阵容核心也可以换下（主C 永远不卖）。
        觉醒卡是 3 张合成换来的、词条翻倍，比任何没觉醒的核心都值钱，
        觉醒英雄回到手牌应立刻上场，优先级高。

        只在"可以卖"的单位里找（主C / 核心本来就受保护）。原来在全部单位里找最弱的，
        最弱的恰好受保护时调用方就直接放弃 —— **不会去看第二弱的**，于是手里的强卡永远上不去。
        """
        cands = [n for n in self._field
                 if not (self._is_carry(n) or (self._is_core(n) and not allow_core))
                 and n not in self._ever_awakened        # 觉醒过的不拿去换觉醒卡
                 and not (self._is_engine_part(n) and not allow_core)]   # 引擎零件不当"最弱"（第 13 局把产出方当闲置卖了 7 个）
        # **别把前排坦克卖光**（2026-09-24 match14 复盘：为上后排的嬴政/蒙犽/瑶陆续卖掉牛魔、老夫子、鲁班大师，
        # 最后一场前排一个坦克都没有，第 10 回合出局）。换上来的不是坦克、场上坦克只剩 MIN_FRONT_TANKS 个时，坦克不卖。
        if incoming and not self._is_tank(incoming):
            tanks = [n for n in self._field if self._is_tank(n)]
            if len(tanks) <= self.MIN_FRONT_TANKS:
                cands = [n for n in cands if not self._is_tank(n)]
                if not cands:
                    return None
        if not cands:
            return None
        name = min(cands, key=self.unit_value)
        return name, self._field[name], self.unit_value(name)

    def _best_in_hand(self, state: GameState):
        """手牌里最值得上场的英雄（名字, 分值）。没有返回 None。"""
        best = None
        for slot in state.hand_slots:
            n = slot["name"]
            if self.pieces.get(n, {}).get("category") != "英雄牌":
                continue
            if n in self._field and not slot.get("awakened"):
                continue                 # 同名走合成，不占新位置
            v = self.unit_value(n)
            if best is None or v > best[1]:
                best = (n, v)
        return best

    TRANSFER_FROM_ROUND = 5  # 前期场上人少、每个单位都在挡刀，不为换绑枫叶舞卖人
    LEVEL_TARGET = meta.LEVEL_TARGET   # (从第几回合起, 至少几级)：落后就先升级（表在 meta，Laya 输入也用它）

    def _replacement_waiting(self, state: GameState) -> bool:
        """手里有没有能上场顶位的英雄（不在场上、不是过时的 1 阶）。"""
        return any(self.pieces.get(sl["name"], {}).get("category") == "英雄牌"
                   and sl["name"] not in self._field and not self._obsolete(sl["name"], state.round_no)
                   for sl in state.hand_slots or [])

    _DROP_STATS: Optional[dict] = None
    _FEATURES: Optional[dict] = None

    def _features(self, name: str) -> dict:
        """卡牌语义特征（data/card_features.json）。没有就给空 dict。"""
        if RuleEngine._FEATURES is None:
            try:
                p = Path(__file__).resolve().parents[2] / "data" / "card_features.json"
                RuleEngine._FEATURES = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                RuleEngine._FEATURES = {}
        return RuleEngine._FEATURES.get(name) or {}

    def _drop_fail_rate(self, cell) -> float:
        """这个落点的历史失败率（data/drop_cell_stats.json，没有数据按 0.5）。"""
        if RuleEngine._DROP_STATS is None:
            try:
                p = Path(__file__).resolve().parents[2] / "data" / "drop_cell_stats.json"
                RuleEngine._DROP_STATS = json.loads(p.read_text(encoding="utf-8")).get("cells", {})
            except (OSError, ValueError):
                RuleEngine._DROP_STATS = {}
        v = RuleEngine._DROP_STATS.get(f"{cell[0]},{cell[1]}")
        return v["fail_rate"] if v else 0.5
    SWAP_MARGIN = 15        # 手牌里的人要明显强过场上最弱的，才值得卖人换位

    def _worth_buying(self, slot: ShopSlot, state: GameState) -> bool:
        """值不值得买。判据只有：升级材料 / 主派系 / 前期补强度 / 高优先级。

        **刻意不把"有登场效果"算进来**：花 3 能量买一张只为吃一次登场效果的牌，
        性价比极低。白嫖登场效果只用手上已有的牌。
        """
        if self.recently_sold(slot.piece or ""):
            return False             # 刚卖掉的英雄别马上买回来（2026-09-30）
        _u = self._usage_self_cast(slot.piece or "") or {}
        _fw = [w for w in (_u.get("when") or []) if str(w).startswith("faction_on_field:")]
        if _fw and not self._usage_when_ok(dict(_u, when=_fw), state, slot.piece or "")[0]:
            # 2026-10-19 第 785 局 R11（逍遥自在的卡面效果理解错了）：场上 0 个无阵营英雄还买了逍遥自在
            # （"所有无阵营英雄获得 2×场上无阵营数量的等级"）→ 卡面要的阵营人数现在不够就别买（买回来用不了）
            return False
        if self.late_c_position(slot.piece or ""):
            return False             # 主C 定了：新的 C 位英雄买来也是低等级、没用（2026-10-10）
        if slot.piece is None or slot.sold:
            return False
        info = self.pieces.get(slot.piece)
        if info is None:
            return False
        if slot.faction and slot.faction == state.banned_faction:
            return False
        if slot.cost is not None and slot.cost > state.gold:
            return False
        if (self._level_hold[0] == state.round_no
                and state.gold - (slot.cost or 3) < self._level_hold[1]):
            return False              # 正在等弈星上场后升级：升级费留着
        # 手牌接近上限（上限 10，超出开战后被系统自动卖掉 = 白花能量）：
        # 只放行主力/核心，升级材料也要让路——2026-09-22 实战买到 11 张溢出。
        if len(state.hand) >= 8 and not (
                self._is_carry(slot.piece) or self._is_core(slot.piece)):
            return False
        # **商店里的"给英雄加等级"效果牌**（精密协作：指定1名英雄等级+4、随机1名+3；特训战术 +4）。
        # 原来 _worth_buying 只为英雄牌写了放行条件，效果牌一张都不会买 —— 2026-09-23 第八局
        # 人口 7/7、6 能量、商店里摆着 2 费的精密协作，38 秒什么都没做（商店见到效果牌卡死）。
        # 人口满的时候它正是最值的：不占人口，直接给场上的人涨等级。
        # 放在"给拍卖留钱"之前：2 能量换 +4 级，比攒着去拍卖值。
        info0 = self.pieces.get(slot.piece, {})
        if (info0.get("category") == "效果牌" and self._field
                and re.search(r"等级\+\d", info0.get("skill", ""))
                and not any(k in info0.get("skill", "") for k in self.HOLD_KEYWORDS)):
            return True
        # 留够拍卖的钱：拍卖是拿高价值牌的主要渠道，买杂牌把能量花光就彻底缺席
        # （实测一局买 34 张，拍卖时能量 0，一件都没拍）。
        # 但**前期铺场优先于攒钱**——场面还没起来就攒钱，只会被打死得更快。
        # 所以只在"场面已经有 3 个以上英雄"时才守这条底线，且主力/核心不受限。
        # **预留只按拍卖算，不再叠一个静态底线**。
        # `gold_reserve=10` 这条底线的本意就是"给拍卖留钱"，但后来加的
        # `_auction_hold()` 已经在按"距下次拍卖几回合 + 这期间的收入"动态算了 ——
        # 两条叠在一起是重复扣钱：场上一有 3 个人，就要 13 能量才肯买一张 3 费牌，
        # 而中期每回合收入才 8~12 → 大部分回合一张都买不了
        # （2026-09-23 实战回合4：8 能量、场上 3 人、商店四张全被判 ✗）。
        if (len(self._field) >= 3
                and state.gold - (slot.cost or 3) < self._auction_hold(state)
                and not (self._is_carry(slot.piece) or self._is_core(slot.piece))):
            return False
        # **给"要留到最后"的场上英雄凑觉醒**（大多数英雄只吃到了一张卡是不对的；
        # 这个英雄是否要保留到最后、完全觉醒还差几个，也要成为商店购买的依据）。
        # 合成不占人口，满编时这正是钱该去的地方；原来只有主C/阵容核心的同名卡才买，于是满编后只剩主C 可买 → 一直刷新。
        if (info.get("category") == "英雄牌" and self._is_keeper(slot.piece)
                and self.awaken_left(slot.piece, state) != 0):
            return True
        # 升级材料（同名已拥有）：**只对阵容内的牌**算数。
        # 非阵容杂牌越买越多，既占手牌又没战力，还把拍卖的钱花光。
        if self._owned_copies(slot.piece, state) >= 1 and (
                self._is_carry(slot.piece) or self._is_core(slot.piece)
                or self._owned_copies(slot.piece, state) >= 2):
            return True
        # **主C（当前主C / 阵容定的主C）出现在商店里就买**（2026-09-23 第十局：人口 6/7、手里等着一个庄周，
        # 按"没位置"判不买；最后那条又只放行 4 阶以上，3 阶的阵容主C 亚连 19 能量摆着没买）。
        # 买回来放不下时由"卖弱换强"去腾位 —— 主C 的估值最高，一定换得上去。
        if info.get("category") == "英雄牌" and self._is_carry(slot.piece):
            return True
        # 棋手经验型英雄（弈星："棋手升级时，棋手经验+2"）：前期买它等于买"升级折扣"。
        # 它不是阵容牌，靠上面那几条判据一辈子买不到（无阵营、priority 低、
        # 不是升级材料）——所以单独开一条，且只在**前期、手上还没有**时放行。
        # 打法：买来上场 → 升一次级白拿经验 → 卖掉腾人口。
        if (self._is_level_engine(slot.piece)
                and state.round_no <= 8
                and self._owned_copies(slot.piece, state) == 0
                and slot.piece not in self._field):
            return True
        # **空位要减掉"手里已经等着上场的英雄"**（2026-09-23：每次到 3 级就疯狂购买英雄）：
        # 升 3 级人口上限 5 -> 6，多出 1 个空位，原来每张英雄都判"有空位"，一口气买了 4 张。
        waiting = sum(1 for sl in state.hand_slots
                      if self.pieces.get(sl["name"], {}).get("category") == "英雄牌"
                      and sl["name"] not in self._field
                      and not self._obsolete(sl["name"], state.round_no))
        here0, cap0, _src0 = self.pop_now(state)       # 2026-10-19：读不到时 = 上次读数 + 之后账本的确定变化
        cap0 = cap0 or 5
        room_left = cap0 - here0 - waiting
        # 主派系：也得有位置放（或者能合成）才买
        if self.main_factions and slot.faction in self.main_factions and room_left > 0:
            return True
        # **场面没填满就该买人补位** —— 判据是"场上人数 vs 人口上限"，**不是回合数**。
        # 原来写的是"前 3 回合且场上<3"，于是第 4 回合起就只买品阶≥4 的牌；
        # 可那时才 2~3 级、商店只出品阶1，等于几乎什么都不买 ——
        # 2026-09-22 实战打到第 6 回合场上仍然只有 2 个人（人口上限 6）。
        # 原则：空着场面必输，上个杂牌至少能挡刀。
        # 场面填没填满：**人口读数优先于记账**，读不到就用"上一次真读到的"。
        # 记账会掉人（拖拽回执丢失、对账修正、卖人执行失败…），实战出现过
        # 账本 2 / 实际 5 的差距 —— 那一帧商店开着、人口读不到，于是按"场上才 2 个"
        # 继续买，手牌一路堆到 8/10（2026-09-23：手上堆积了手牌）。
        # 买多了的代价（占手牌上限、花光拍卖和升级的钱）比买少了大，所以这里取**保守**的一边。
        here, cap = here0, cap0
        if here < cap and room_left > 0:
            return True
        # **场面已满、手里又压着一堆上不去的英雄 —— 别再买英雄了**。
        # 2026-09-23 实战：满编 6/6 之后继续买，买了 12 张，结束时手里压着
        # 6 个上不了场的英雄（9-23 "手上堆积了手牌"就是这个形态）。
        # 那些能量本该去升级（开人口上限和高品阶池），而不是变成手牌上限的压力。
        # 例外：**同名合成材料**（不占人口，直接变强）和**主C**（换人也要换它）。
        idle = [sl["name"] for sl in state.hand_slots
                if self.pieces.get(sl["name"], {}).get("category") == "英雄牌"
                and sl["name"] not in self._field]
        if (len(idle) >= self.IDLE_HAND_HEROES
                and slot.piece not in self._field
                and not self._is_carry(slot.piece)):
            return False
        # 中后期只买高品质。**但人口满了还得看买回来放得下放不下**：
        # 2026-09-22 实战，人口 7/7、手牌堆了 9 张英雄、血只剩 1，还在继续买
        # （有啥买啥、场上的英雄也不卖，只顾买）。
        # 买回来上不了场的牌既占手牌上限、又把拍卖和升级的钱花光。
        if state.population_valid and state.population >= state.population_cap:
            weakest = self._weakest_on_field()
            if weakest is None and slot.piece not in self._field:
                return False      # 场上全是主C/核心，没人可换 -> 买新人头没意义（同名合成材料除外）
            if weakest and self.unit_value(slot.piece) <= weakest[2] + self.SWAP_MARGIN:
                return False      # 换不掉场上任何人 -> 买它没意义
            if weakest:
                # **换得掉就该买**。这一段原来只会否决、从不放行，于是落到最后那条
                # "品阶≥4 才买"上 —— 而**品阶1 的主C 过不去**。
                # 2026-09-23 实战：满编之后连主C 都不买了，而主C 正是最该买来
                # 合成/换人的那张（打法就是"低阶重复合成"）。
                logger.info("买 %s：场面满了，但它(%d分)明显强过场上最弱的 %s(%d分)，"
                            "买来换人", slot.piece, self.unit_value(slot.piece),
                            weakest[0], weakest[2])
                return True
        return info.get("priority", 0) >= 7 or info.get("quality", 0) >= 4

    # 效果牌 N 选 1 的取舍关键词：能直接变成阵容英雄 > 直接战力 > 资源
    CARD_PICK_KEYS = (
        ("英雄牌", 8), ("英雄", 6), ("等级", 5), ("觉醒", 5),
        ("装备", 3), ("刷新", 2), ("能量", 2), ("金币", 2),
    )
    # 阿离换主C 靠"搬等级"。红叶染锋 = 枫叶舞英雄离场后 50% 等级留给下一个，
    # 阵容原文"伞强化必须要这个红叶染锋"；新生 = 枫叶舞英雄每回合 +2 级；蓄势 = 击杀 +1 级。
    CARD_PICK_PREFER = {
        "红叶染锋": 40, "翩跹舞步·新生": 22, "翩跹舞步·蓄势": 16,
        "破境之证": 30, "加速成长": 20, "精密协作": 14, "简易协作": 10,
        "遗志": 12, "化铁为金": 12,
    }

    def item_wrong_type(self, item: str) -> bool:
        """装备和主C、未来主C（阵容主C）的伤害类型都相反（如噬神之书是法术道具，都是物理系时不能选）。"""
        from . import hero_traits as _ht
        fc = (self._active_comp or {}).get("carry") if self._active_comp else None
        return _ht.item_mismatch(item, [self._carry_piece, fc])

    def decide_card_pick(self, state: GameState) -> Optional[dict[str, Any]]:
        """卡牌 N 选 1：挑一张对当前阵容最有用的。

        这个界面**占着商店的位置**，不选会一直挂着把后续流程全卡死，
        所以**必须选**——哪怕全都打 0 分也要选第一张。
        """
        best, best_score = state.card_options[0], -1.0
        scored_opts = []
        for opt in state.card_options:
            info = self.pieces.get(opt["name"], {})
            text = opt["name"] + info.get("skill", "")
            score = sum(w for k, w in self.CARD_PICK_KEYS if k in text)
            # 攻略点名：想加速升级就优先拿「扶摇直上」（直接 +1 级，满级转 10 能量）。
            # 但**已经进入搜牌期就不需要它了**——那时候升本反而是负收益。
            if opt["name"] == meta.FREE_LEVEL_CARD:
                score += 2 if self._in_search_phase(state) else 12
            # 阵容核心相关的额外加分
            if any(h in text for h in (self._active_comp or {}).get("core", [])):
                score += 6
            score += self.CARD_PICK_PREFER.get(opt["name"], 0)
            if info.get("category") == "英雄牌":
                # **挑英雄时要看它能不能当主C**（2026-09-23 实战：5 选 1 里有孙悟空、露娜这种
                # 热门主C，却选了甄姬 —— 原来只数关键词，所有英雄牌得分几乎一样）
                score += 16 * self._any_carry_freq.get(opt["name"], 0.0)
            # 当前主C 的同名卡（合成材料）
            if self._carry_piece and opt["name"] == self._carry_piece:
                score += 10
            # 主C 定了：能给主C 涨等级的英雄加分，分资源的第二主C 减分（第 332 局 花木兰）
            if info.get("category") == "英雄牌" and self.carry_settled():
                if self.feeds_carry(opt["name"]):
                    score += 12
                elif self.unwanted_second_carry(opt["name"]) or self.other_c(opt["name"]):
                    score -= 30         # 第 343 局：-10 不够，司空震 17 分仍被选中、随后让位卖掉满装主C
            if self.item_wrong_type(opt["name"]):
                score -= 100        # 2026-09-26：主C 与未来主C 都是物理系时不选法术装（噬神之书）

            scored_opts.append({"name": opt["name"], "score": round(score, 1)})
            if score > best_score:
                best, best_score = opt, score
        self.record_decision("card_pick", state, scored_opts, best["name"])
        logger.info("卡牌N选1：选 %s（%.0f 分，候选 %s）",
                    best["name"], best_score, [o["name"] for o in state.card_options])
        # 账本只记执行了的那一张（2026-10-15 第 632 局 R7：这里记了"亚瑟"，Laya 改选苏烈并执行，账本却是亚瑟）→ 挪到 note_executed
        self.__dict__["_pending_pick"] = [o["name"] for o in state.card_options]
        # **同一组候选点不掉就换落点、再点不掉就放弃**（2026-09-23 实战连点 25 次）。
        key = tuple(sorted(o["name"] for o in state.card_options))
        tries = self._card_pick_tries.get(key, 0)
        if tries >= self.CARD_PICK_MAX_TRIES:
            if tries == self.CARD_PICK_MAX_TRIES:
                self._card_pick_tries[key] = tries + 1      # 只警告一次
                logger.error("卡牌N选1 点了 %d 次都没关掉（候选 %s）—— 不再点，"
                             "等它自己消失或人工处理", tries, list(key))
            return None
        self._card_pick_tries[key] = tries + 1
        return {"type": "card_pick", "piece": best["name"],
                "x": best["x"], "y": best["y"], "try": tries}

    # 卡面效果 -> 加分。**这是"用效果数据库参与决策"的入口**（2026-09-22 加入）。
    # 之前买牌只看品阶和阵容归属，828 张卡的效果文本一个字都没读。
    # 权重的依据写在每条后面，改之前先想清楚为什么。
    EFFECT_BONUS = (
        # 召唤物**不占人口**（规则事实），等于白嫖人头 —— 场面越满越值钱。
        (r"召唤", 6, "召唤(不占人口)"),
        # 给英雄加等级/经验的，收益随回合累积。
        (r"等级.{0,4}\+|经验", 3, "涨等级/经验"),
        # 自带装备 = 省一张装备牌，而装备每人上限 3 件。
        (r"获得.{0,6}装备|穿戴", 3, "自带装备"),
        (r"免费刷新", 2, "免费刷新"),
    )

    def _effect_bonus(self, piece: str, state: GameState) -> tuple[float, list[str]]:
        """按**卡面效果文本**给一张牌加分，返回 (加分, 理由)。

        除了固定权重的关键词，还有一条**要看场面**的：
        卡面点名某个派系（如"对距离最近的【河洛】英雄…"）时，
        只有**场上真有那个派系的人**才有意义 —— 这也是这游戏派系的唯一作用方式
        （规则事实：派系没有传统羁绊，全靠卡牌效果生效）。
        """
        info = self.pieces.get(piece) or {}
        text = (info.get("skill") or "") + (info.get("awaken") or "")
        if not text:
            return 0.0, []
        score, why = 0.0, []
        for pat, pts, label in self.EFFECT_BONUS:
            if re.search(pat, text):
                score += pts
                why.append(label)
        # 派系呼应：卡面点名的派系，场上得真有人
        for fac in re.findall(r"【([^】]+)】", text):
            same = sum(1 for n in self._field
                       if fac in (self.pieces.get(n, {}).get("synergies") or []))
            if same >= 2:
                score += 4
                why.append(f"呼应场上{same}个{fac}")
                break
        return score, why

    def decide_buy(self, state: GameState) -> Optional[dict[str, Any]]:
        """买最值得的一张：升级材料 > 主派系 > 高优先级。"""
        if not state.shop_open:
            return None     # 商店关着，卡槽坐标上是棋盘，点了会选中棋子
        candidates = []
        for slot in state.shop:
            if not self._worth_buying(slot, state):
                continue
            info = self.pieces.get(slot.piece, {})
            score = info.get("priority", 0)
            if self._is_carry(slot.piece):
                score += 6
            elif self._is_core(slot.piece):
                score += 3
            dup = self._owned_copies(slot.piece, state) >= 1
            if dup:
                score += 5                       # 升级材料（合成涨等级）
            cap0 = state.population_cap if state.population_valid else 5
            if self._is_keeper(slot.piece) and len(self._field) >= cap0:
                # 场面满了才加（没满时"新人头 +10"优先 —— 空位比合成更值，见下面那条）
                left = self.awaken_left(slot.piece, state)
                # 越接近觉醒越值（差 1 张 +8，差 3 张 +4；读不到按中间算）；后期阵容角色再 +3
                score += 4 + 2 * (3 - (left if left is not None else 2))
                if slot.piece in self._late_roles():
                    score += 3
            # **场面没填满时，人头优先于合成**（2026-09-22 实战）：
            # 打分里"重复卡 +5、核心 +3"让同名卡永远胜出，
            # 结果 9 次买入全是 牛魔×5 / 曜×3，场上却一直只有 1~2 个人。
            # 合成是对的（打法就是低阶重复合成），但**空着的位置更值钱** ——
            # 一个 1 级杂牌也能挡刀、能吃伤害，而合成只是让已有的那个更肥。
            # 场面填满之后这条自动失效，回到"合成优先"。
            cap = state.population_cap if state.population_valid else 5
            if len(self._field) < cap and not dup and slot.piece not in self._field:
                score += 10
            if self.main_factions and slot.faction in self.main_factions:
                score += 2
            bonus, why = self._effect_bonus(slot.piece, state)
            score += bonus
            # **伞单核候选**（不必只走守约单核，能做单核的射手角色还有好几个）：
            # 按主 agent 读技能的评分（data/hero_carry_reviewed.json），评分 ≥7 的输出位在还没定主C、
            # 或比当前主C 更好、或本身就是当前主C（同名=合成材料）时加分 —— 2~3 阶前期就能刷到的加得多，
            # 4~5 阶大C 等 4 级起、通常要靠搬等级转过去，加得少一点。
            hr = self.hero_reviewed.get(slot.piece)
            if hr and info.get("category") == "英雄牌" and hr.get("carry_score", 0) >= 7:
                cur = self._carry_piece
                cur_cs = self.hero_reviewed.get(cur, {}).get("carry_score", 0) if cur else 0
                if (not cur or slot.piece == cur or hr["carry_score"] > cur_cs) and not self.late_c_position(slot.piece):
                    q = info.get("quality", 0) or 0
                    add = 7 if q <= 3 else (4 if (state.level or 0) >= 4 else 0)
                    if add:
                        score += add
                        why = list(why) + [f"单核候选{hr['carry_score']}分"]
            candidates.append((score, slot, why))
        if not candidates:
            self._hand_last = list(state.hand)
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        self.record_decision("buy", state,
                             [{"name": sl.piece, "score": sc, "why": why, "cost": sl.cost}
                              for sc, sl, why in candidates]
                             + [{"name": sl.piece, "rejected": True} for sl in state.shop
                                if sl.piece and all(sl is not c[1] for c in candidates)],
                             best.piece)
        if candidates[0][2]:
            logger.info("买 %s：效果加分 %s", best.piece, "+".join(candidates[0][2]))
        self._hand_last = list(state.hand)
        # **价格读不到也要查钱够不够**（2026-09-25 第四轮第 5 局：能量 0 时想买「简易协作」，价格没读到 → 原来这条检查整个跳过，
        # 每 0.4 秒重复决策一次）。读不到就用卡牌库标价估（英雄统一 3），库里也没有按最低 1 估。
        price, _est = self.est_price(best.piece, best.cost)
        if state.gold >= 0 and state.gold < price:
            return None
        # **账本在 note_executed 里写，不在这里写**：这里只是"决定要买"，
        # 而执行层可能因为去重/阶段切换把动作丢掉，或者点到一个已售空的槽位。
        # 2026-09-22 实战：账本里「买牌 项羽」记了 4 次，按能量推算那回合只真的买成 1 次
        # —— 记账记的是意图而不是事实，事后所有分析都被带偏。
        # 费用徽章是全屏字号最小的一处，实测识别率 89%，
        # 读不到时账本就记成 `花费=None`，事后完全算不出这一局的能量都花哪了。
        # **英雄牌的商店价统一 3 能量**，所以读不到就按 3 估，并在账本里标明是估的
        # —— 标明很重要，别让估值以后被当成实测数据。
        cost, est = best.cost, False
        if cost is None and self.pieces.get(best.piece, {}).get("category") == "英雄牌":
            cost, est = 3, True
        return {"type": "buy_piece", "slot": best.index, "piece": best.piece,
                "cost": cost, "cost_est": est, "round": state.round_no}

    # ---------- 升级 / 布阵 ----------

    def _adjacent_free(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        """找 pos 周围一个未被我方占用的格子（喂等级/挪位用）。"""
        occupied = set(self._field.values())
        for cand in ((pos[0] - 80, pos[1]), (pos[0] + 80, pos[1]),
                     (pos[0], pos[1] - 70), (pos[0], pos[1] + 70)):
            if cand not in occupied:
                return cand
        return None

    @staticmethod
    def _cell_neighbors(cell) -> list:
        """棋盘格的相邻格（蜂窝排布：同行左右各一个；偶数行连上下行的 i-1、i，奇数行连 i、i+1）。"""
        from ..perception.extractors import OWN_BOARD_ROWS
        rows = [[tuple(c) for c in r] for r in OWN_BOARD_ROWS]
        cell = tuple(cell)
        for r, row in enumerate(rows):
            if cell in row:
                i = row.index(cell)
                out = [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
                off = (i - 1, i) if r % 2 == 0 else (i, i + 1)
                for rr in (r - 1, r + 1):
                    if 0 <= rr < len(rows):
                        out += [rows[rr][j] for j in off if 0 <= j < len(rows[rr])]
                return out
        return []

    def _steal_fodder(self, state: GameState, thief: str) -> Optional[tuple[str, int]]:
        """**可以被夺取的"不重要"英雄**：主C 评分 ≤4（坦克/辅助/过渡件）、不是主C、等级 ≥10（夺满 10 级），等级最高的优先。
        打法：公孙离有夺取，把一个不重要的、大于 10 级的英雄放到他旁边，然后再合成升级。"""
        lv = state.unit_levels or {}
        best = None
        for n, p in self._field.items():
            if n in (thief, self._carry_piece):
                continue
            if self.hero_reviewed.get(n, {}).get("carry_score", 5) > self.TRANSITION_CARRY_MAX:
                continue                                   # 输出位的等级别抢
            l = lv.get(cell_key(tuple(p))) or self._unit_lv.get(n)
            if l and l >= 10 and (best is None or l > best[1]):
                best = (n, l)
        return best

    def decide_upgrade(self, state: GameState) -> Optional[dict[str, Any]]:
        """手里有场上同名英雄 → 拖上去升级（合成）；主力/核心的同名牌最优先。

        主力英雄带"夺取"（吸收旁边己方英雄等级）时：先挪一个非主力英雄到他旁边喂等级，
        升级完再挪回原位（sequence 三步，executor 顺序执行）。
        """
        if self._shop_blocks_board(state):
            return None      # 商店面板开着，先关店再合成升级（2026-09-23）
        # 主力同名牌最优先升级
        if self._active_comp:
            carry = self._active_comp.get("carry")

            def _prio(s):
                n = s["name"]
                return 0 if n == carry else (1 if self._is_core(n) else 2)

            ordered = sorted(state.hand_slots, key=_prio)
        else:
            ordered = state.hand_slots
        _victim = self._transfer_victim(state)[0] if self._carry_piece else None
        for slot in ordered:
            name = slot["name"]
            if name not in self._field:
                continue
            if (name == self._maple_bound and self.keep_as_support(name)
                    and not self.final_carry_settled()):
                # **枫叶舞先绑在配套件（李元芳）身上时，之后拿到的同名留在手里**（2026-10-14：先绑，看能不能凑击杀；
                # 拿到的其他李元芳先不用来升级，等真主C 到手后，卖掉场上的，再上手牌里的）
                continue
            if name == _victim:
                # **要为换绑卖掉的人，别先往他身上合成**（2026-10-08 第 425 局：这一回合就换主C，然后卖掉李元芳，
                # 再把手牌上的另一个李元芳用上 —— 先合成再卖就浪费了）。卖掉后手里这张会作为新人重新上场
                continue
            if self._parked(name):
                continue  # 本回合这张牌已经试到上限，别再吃帧（2026-09-23 白起吃掉整个回合6）
            if name == self.HAIYUE_TEMP and name in self._ever_awakened and self.haiyue_phase(state) == "raise":
                continue  # 伞海月：干将觉醒后的同名先捏在手里，到 3 级再一起合成刷 3 阶锦囊（2026-10-14）
            info = self.pieces.get(name, {})
            if info.get("category") != "英雄牌":
                continue
            pos = self._field[name]
            skill = info.get("skill", "")
            if "夺取" in skill and not (name == self._carry_piece or self._is_keeper(name)):
                # **不重要的夺取英雄合成也会随机抢相邻一人 10 级**（2026-10-04 第 5~6 名复盘，第 179 局：猪八戒站到亚连旁边，
                # 一合成亚连 17→5 级）。原来这条护栏只对主C / 核心生效。相邻有主C 或评分高的输出位 → 不合成
                nbrs0 = set(self._cell_neighbors(pos))
                guard = [n for n, p in self._field.items() if n != name and tuple(p) in nbrs0
                         and (n == self._carry_piece or n in self._ever_awakened
                              or self.hero_reviewed.get(n, {}).get("carry_score", 5) > self.TRANSITION_CARRY_MAX)]
                if guard:
                    if self._skip_logged.get(("steal_guard", name)) != self._round_seen:
                        self._skip_logged[("steal_guard", name)] = self._round_seen
                        logger.info("%s 带夺取，旁边站着 %s（合成会随机抢 10 级），先不合成", name, guard)
                    continue
            if "夺取" in skill and (name == self._carry_piece or self._is_keeper(name)):
                # **夺取 = 使用此卡牌（上场 / 合成）时夺取周围 1 格随机 1 名己方英雄的等级（最多 10 级）**（官方关键词）。
                # 打法：把一个不重要、≥10 级的英雄放到它旁边再合成。随机抢相邻一人 → 相邻格里不能有主C / 输出位。
                nbrs = set(self._cell_neighbors(pos))
                adj_units = [n for n, p in self._field.items() if n != name and tuple(p) in nbrs]
                if name == self._carry_piece:
                    # 它自己就是主C：抢来的等级正好喂主C，只护着比它还值钱的人（2026-09-26 第四轮第 4 局：
                    # 主C 孙悟空 旁边站着阿轲就"先不合成"，手里 3 张孙悟空整回合不用，还接着买）
                    # 只护着"主C 评分不低于它"的输出位（甄姬挨着公孙离）和比它值钱的觉醒卡；阿轲(5) 挨着孙悟空(8) 就抢
                    mine = self.unit_value(name)
                    my_cs = self.hero_reviewed.get(name, {}).get("carry_score", 5)
                    protected = [n for n in adj_units
                                 if self.hero_reviewed.get(n, {}).get("carry_score", 5) >= my_cs
                                 or (n in self._ever_awakened and self.unit_value(n) > mine)]
                else:
                    protected = [n for n in adj_units if n == self._carry_piece
                                 or self.hero_reviewed.get(n, {}).get("carry_score", 5) > self.TRANSITION_CARRY_MAX]
                if protected:
                    if self._skip_logged.get(("steal_guard", name)) != self._round_seen:
                        self._skip_logged[("steal_guard", name)] = self._round_seen
                        logger.info("%s 带夺取，旁边站着 %s（会被随机抢 10 级），先不合成", name, protected)
                    continue
                fod = self._steal_fodder(state, name)
                if fod and fod[0] not in adj_units:
                    occ = {tuple(v) for v in self._field.values()} | set(self._anchor_cells(state))
                    free_nb = [c for c in nbrs if c not in occ and not self._cell_dead(c)
                               and not self._covered_by_shop(state, c)]
                    if free_nb:
                        src = tuple(self._field[fod[0]])
                        dst = free_nb[0]
                        self._cell_owner.pop(src, None)
                        self._field[fod[0]] = dst
                        self._cell_owner[dst] = fod[0]
                        logger.info("夺取合成：先把 %s（%s 级，不重要）挪到 %s 旁边 %s，再合成 %s 抢它 10 级",
                                    fod[0], fod[1], name, dst, name)
                        self.note("夺取合成", 英雄=name, 喂等级=fod[0], 等级=fod[1], 挪到=f"{dst[0]},{dst[1]}")
                        return {"type": "sequence", "piece": name, "steps": [
                            {"type": "move_unit", "piece": fod[0], "src": src, "dst": dst},
                            {"type": "drag_card", "piece": name, "src": (slot["x"], slot["y"]), "dst": pos,
                             "merge": True},
                        ]}
            # **那一格本局已经拖失败够多次 -> 记录肯定错了，撤销重来**。
            # 合成的落点**必须**是那个英雄的记账位（不能换格），所以格子废了
            # 就只能撤销记录、让它重新上场。
            # 这是 `_cell_dead` 漏掉的**第四条路径**（前三条：上场落点、
            # 装备/效果牌目标、自用牌目标）—— 2026-09-23 实测：干将莫邪
            # 往 (300,569) 和 (839,502) 各合成失败 **4** 次，而阈值是 2。
            if self._cell_fails.get(tuple(pos), 0) >= self.CELL_FAIL_LIMIT and (
                    not state.vision_units or state.shop_open           # 视觉这一帧没跑（商店开着）→ 先等，别撤销
                    or (state.vision_units or {}).get(cell_key(tuple(pos))) == "hero"
                    or self._confirmed_at.get(name) == (self._round_seen, tuple(pos))):
                # **视觉看得见那格有人 / 本回合点开确认过是它 → 记录没错，是这格拖不进去**（最下面一行常被手牌挡住）。
                # 2026-09-30 轮换第 9 局：觉醒的主C 百里玄策站在 (300,569)，合成失败够次数就被撤销记录、当成"觉醒回手"，
                # 之后两个回合"手里有觉醒的百里玄策等上场 —— 先关店"，商店开不了、升不了级，第 5 名。
                if self._skip_logged.get(("merge_dead", name)) != self._round_seen:
                    self._skip_logged[("merge_dead", name)] = self._round_seen
                    logger.info("%s 在 %s（视觉看得见），但这格本局拖失败多次 —— 本回合不往这格合成，记录保留", name, pos)
                continue
            if self._cell_fails.get(tuple(pos), 0) >= self.CELL_FAIL_LIMIT:
                # 只认"真的拖失败够次数"，不认本回合"拖前核对点歪"的临时标记（审查 10-09：点歪一次就撤销记录、误标觉醒回手）
                logger.warning("%s 记在 %s，而那一格本局已拖失败多次 —— 撤销记录重新上场",
                               name, pos)
                self._field.pop(name, None)
                self._cell_owner.pop(tuple(pos), None)
                self._awakened.add(name)
                self.note("判定觉醒回手", 牌=name, 原格=f"{pos[0]},{pos[1]}",
                          依据="该格已拖失败多次")
                continue
            # **合成之前先看一眼那一格画面上有没有人**：没看到就先等一两帧（可能在播动画）。
            # **但绝不因为"没看到"删记录**（铁律，这里第二次违反过）：
            # 徽章检测漏掉约 1/3 的单位，"没看到"和"不在"在画面上长得一样。
            # 原来连续两帧没看到就撤销记录 —— 2026-09-23 那局账本从 7 个人掉到 2 个，
            # 于是"卖弱换强"找不到最弱的、手里的云中君/刘邦一直上不去。
            # 现在等满两帧照样去拖，由**拖拽回执**判：拖失败够次数 `_cell_dead` 才撤销（强证据）。
            seen = set(self._anchor_cells(state))
            if seen and tuple(pos) not in seen:
                k = (name, tuple(pos))
                n = self._merge_missing.get(k, 0) + 1
                self._merge_missing[k] = n
                if n <= 2:
                    logger.info("%s 记在 %s，但画面上那格没看到人（第 %d 帧）—— 等一下再合成",
                                name, pos, n)
                    continue
            self._merge_missing.pop((name, tuple(pos)), None)
            return {"type": "drag_card", "piece": name, "merge": True,
                    "src": (slot["x"], slot["y"]), "dst": pos}
        return None

    def sync_field(self, state: GameState) -> None:
        """把 `_field`（我方场上英雄的动作记忆）与实际人口对齐。

        `_field` 只是"我拖了什么上去"的记忆，没有真正的棋盘检测（见 HANDOVER 九.6）。
        英雄离场时它不会自动清理，最典型的就是**觉醒**：

          本体 + 再打出 3 张同名（共 4 张）→ 英雄觉醒并**回到手牌**（金色卡）。
          此时它已不在棋盘上，但 `_field` 还记着旧位置 →
          `decide_deploy` 的"同名走升级"会永远跳过它，
          升级路径又会把卡拖到一个已经空掉的格子上，白费动作。
          （2026-09-22 实战：干将莫邪被反复拖了一整局。）

        判据：人口读数 < `_field` 记录数，且该英雄同时出现在手牌里 → 判定它回手了。
        人口读不到（商店挡住等）时一律不动，宁可不同步也不要误删。
        """
        # 注：曾经加过"账本说在X格 + 观测到X格没徽章 + 牌在手上"的强判据来认觉醒回手，
        # **已撤销**。徽章检测远不够完整（很多帧一个都检不到，而场上确实有人），
        # 于是"没观测到"被当成"离场了"，把真实记录误删 ——
        # 正是"让弱证据推翻强证据"的错误。
        # 现在仍然只用人口读数这一条：它在商店关着时实测 32/32 可靠。
        # **只认"确认过是觉醒卡"的**（2026-09-23 第十一局）：手里有同名牌最常见的原因是**合成材料**，
        # 不是觉醒回手。原来"人口 < 账本 + 同名在手"就删场上记录，于是刚买来合成用的裴擒虎
        # 把场上的主C 裴擒虎删了、还记成"觉醒过"、主C 清空。现在觉醒有直接证据（颜色候选 + 点开确认）。
        hand_names = {s["name"] for s in state.hand_slots if s.get("awakened")}
        # **只认本帧真读到的人口**（战斗/商店开着时是沿用的旧值，可能是战斗里读到的 1/7）
        if (not state.population_valid or state.phase != "prepare"
                or state.population <= 0 or len(self._field) <= state.population):
            return
        for name in [n for n in self._field if n in hand_names]:
            if len(self._field) <= state.population:
                break
            logger.info("检测到 %s 已离场并回到手牌（觉醒？），清除场上记忆并等待重新上场", name)
            pos = self._field.pop(name, None)
            if pos:
                self._cell_owner.pop(pos, None)
            self._awakened.add(name)
            # **估值也要跟上**（2026-09-23 第七局：有英雄觉醒了，却先上了其他英雄，
            # 导致最强的觉醒英雄上不了场）：原来只有"手牌里看到 AWAKENED"那条路径才给 +40，
            # 这条"人口少了 + 牌在手上"的路径只加了上场优先级。人口满时腾位比的是 unit_value，
            # 于是米莱狄(觉醒)比不过手上的大司命，腾出来的位置给了别人，觉醒卡一直烂在手上。
            # 人口读数（商店关着时 32/32 可靠）+ 牌在手上 = 它回手了；英雄回手只有觉醒这一条路。
            self._ever_awakened.add(name)
            # 主C 回手不清空（choose_carry 看到主C 在手里就不换，等它重新上场）

    # ---------- 账本与对账 ----------

    def match_summary(self, state: GameState | None = None) -> str:
        """一局打完的**自评报告**：这一局到底做成了几件事、卡在哪儿。

        目的：多记录日志，自己找出不足去修正问题，
        每局都比上一局更好。
        散在几百行日志里的单条事件很难看出趋势，所以在对局结束时把账本聚一次：
        **只报可以被画面证伪的量**（拖成/拖失败、买了几张、剩几张烂在手里），
        不报"我打算做什么"。
        """
        from collections import Counter
        ev = Counter(r.get("event", "") for r in self.ledger)
        buys = [r for r in self.ledger if r.get("event") == "买牌"]
        spent = sum(r.get("花费") or 0 for r in buys)
        est = sum(1 for r in buys if r.get("花费来源", "").startswith("估"))
        fails = [r for r in self.ledger if r.get("event") in
                 ("拖拽失败", "合成失败", "重复无效动作")]
        bad_cells = Counter(r.get("落点") for r in fails if r.get("落点"))
        lines = [
            "╔═══ 本局自评 ═══",
            f"║ 上场 {ev['上场']} 次 / 拖拽失败 {ev['拖拽失败']} 次 / "
            f"合成失败 {ev['合成失败']} 次 / 重复无效动作 {ev['重复无效动作']} 次",
            f"║ 买牌 {len(buys)} 张，共 {spent} 能量（其中 {est} 张费用是估的）",
            f"║ 用牌 {ev['用牌']} 次 / N选1 {ev['N选1']} 次 / 判定觉醒回手 {ev['判定觉醒回手']} 次",
            # **这项不要叫"异常"**：它全是"画面比账本多"这个方向，而那是设计上
            # 修不了的（召唤物、开局自动上场、漏记的单位——观测只能加不能减）。
            # 叫"异常"会让这个数字虚高，看报告的人分不清哪些值得查。
            f"║ 画面比账本多 {ev['对账异常']} 次（召唤物/漏记，已知上限）"
            f" / 换阵容卖人 {ev['换阵容卖人']} 次",
        ]
        if bad_cells:
            lines.append("║ 最常拖失败的落点：" + " ".join(
                f"{c}×{n}" for c, n in bad_cells.most_common(3)))
        if state is not None:
            # **卡库里查不到的名字要标出来**：手牌允许保留 OCR 原文（万一有没收录的牌），
            # 于是噪声（"风县"/"鲁理"/"兼颜"）也会进这张表。它们永远打不出去，
            # 混在"还剩没打出去"里会让这项指标虚高 —— 指标不可信就没法用来判断改动效果。
            left, junk = [], []
            for sl in (state.hand_slots or []):
                (left if sl["name"] in self.pieces else junk).append(sl["name"])
            lines.append(f"║ 结束时：场上 {len(self._field)} 人，等级 {state.level}，"
                         f"血 {state.hp}")
            if left:
                # **满编时剩牌是正常的**：牌从拍卖/效果牌/N选1 白来的速度
                # 比场上 6~7 个位置消化得快。实测一局剩 7 张里只有 2 张是买的。
                # 不加这句话，看报告的人会把"机制决定的剩牌"当成 bug 去查。
                full = (state.population_valid
                        and state.population >= state.population_cap)
                why = "（场面已满，没位置了）" if full else ""
                lines.append(f"║ **手里还剩 {len(left)} 张没打出去**{why}：{' '.join(left)}")
            if junk:
                lines.append(f"║ （另有 {len(junk)} 个卡库里查不到的名字，多半是 OCR 噪声："
                             f"{' '.join(junk)}）")
        lines.append("╚" + "═" * 18)
        return chr(10).join(lines)

    # ---------- 记忆存盘（进程重启后同一局接着用） ----------
    # 2026-09-23 晚：修 bug 要中途重启 agent，每次重启规则引擎都"失忆" —— 主C/枫叶舞绑定/场上账/
    # 手牌账/装备记录全清空：第十局重启后选了 -16 分的 1 阶当主C、场上 6 人只认得 3 个。
    _KB = None

    @classmethod
    def _kb_pop_gain(cls, name: str) -> int:
        """卡牌知识库里这张牌打出/选中后**立即**加的人口上限（没有就是 0）。"""
        if cls._KB is None:
            try:
                p = Path(__file__).resolve().parents[2] / "data" / "card_kb.json"
                cls._KB = json.loads(p.read_text(encoding="utf-8"))["entries"]
            except Exception:                                       # noqa: BLE001
                cls._KB = {}
        e = cls._KB.get(name or "") or {}
        return sum(g["n"] for g in e.get("gives", [])
                   if g.get("res") == "人口" and isinstance(g.get("n"), int))

    _PERSIST = ("_carry_piece", "_maple_bound", "_prev_carry", "_round_seen", "_carry_switch_round", "_carry_switch_by", "pop_bonus", "pop_bonus_ledger",
                "_route_locked", "_route_key", "_route_since")
    _PERSIST_SETS = ("_ever_awakened", "_awakened", "_roll_rounds", "_sold_names")

    def dump_state(self) -> dict:
        d = {k: getattr(self, k) for k in self._PERSIST}
        d.update({k: sorted(getattr(self, k)) for k in self._PERSIST_SETS})
        d["_field"] = {n: list(p) for n, p in self._field.items()}
        d["coin_machine"] = [self.coin_machine_wanted, list(self.coin_machine_cell) if self.coin_machine_cell else None]
        d["object_cells"] = [[list(c), n] for c, n in self.object_cells.items()]
        d["_hand_book"] = dict(self._hand_book)
        d["_worn"] = {h: sorted(v) for h, v in self._worn.items()}
        d["_lord_boosts"] = sorted(self._lord_boosts)
        d["_talents_taken"] = sorted(self._talents_taken)
        d["_route_group"], d["_group_settled"] = list(self.__dict__.get("_route_group") or ()), bool(self.__dict__.get("_group_settled"))
        d["_group_followed"] = bool(self.__dict__.get("_group_followed"))
        d["_core_seen"] = bool(self.__dict__.get("_core_seen"))
        d["_equip_seen"] = dict(self._equip_seen)
        d["_entered"] = dict(self._entered)
        # **阵容也要跨重启**（match14 复盘：重启后阵容从「阿离耀天帝」换成主C 整局都没出现过的「镜神无限分身流」）
        d["_active_comp_name"] = (self._active_comp or {}).get("name", "")
        d["_comp_switch_round"] = self._comp_switch_round
        # 指定路线的运行时状态（2026-09-30 审查：重启后丢了"已换备选 / 已让位"，会把让出去的主C 拉回来）
        d["_forced"] = {"route": self.forced_route, "orig": self.__dict__.get("_forced_orig", ""),
                        "released": self._forced_released, "switched": self.__dict__.get("_forced_switched", False),
                        "blocked": self.__dict__.get("_forced_blocked", False)}
        return d

    def load_state(self, d: dict) -> None:
        for k in self._PERSIST:
            if k in d:
                setattr(self, k, d[k])
        for k in self._PERSIST_SETS:
            if k in d:
                setattr(self, k, set(d[k]))
        fz = d.get("_forced") or {}
        # 只在**同一个首选**时恢复（= 同一局中途重启）。2026-09-30 第 17 局：新一局首选守约，却读回上一局存档里
        # "已换成玄策路线"，整局按玄策打（reset_match 又把首选恢复成存档里的 orig）
        if fz.get("route") and self.forced_route and (fz.get("orig") or fz.get("route")) == self.forced_route:
            self.forced_route = fz["route"]
            if fz.get("orig"):
                self._forced_orig = fz["orig"]
            self._forced_released = bool(fz.get("released"))
            self._forced_switched = int(fz.get("switched") or 0)
            self._forced_blocked = bool(fz.get("blocked"))
        self._field = {n: tuple(p) for n, p in (d.get("_field") or {}).items()}
        cm = d.get("coin_machine") or [False, None]
        self.coin_machine_wanted, self.coin_machine_cell = bool(cm[0]), (tuple(cm[1]) if cm[1] else None)
        self.object_cells = {tuple(c): n for c, n in (d.get("object_cells") or [])}
        self._cell_owner = {tuple(p): n for n, p in self._field.items()}
        self._hand_book = collections.Counter(d.get("_hand_book") or {})
        self._worn = {h: set(v) for h, v in (d.get("_worn") or {}).items()}
        self._lord_boosts = set(d.get("_lord_boosts") or [])
        self._talents_taken = set(d.get("_talents_taken") or [])
        self._route_group, self._group_settled = tuple(d.get("_route_group") or ()), bool(d.get("_group_settled"))
        self._group_followed = bool(d.get("_group_followed"))
        self._core_seen = bool(d.get("_core_seen"))
        self._equip_seen = dict(d.get("_equip_seen") or {})
        self._entered = dict(d.get("_entered") or {})
        cname = d.get("_active_comp_name")
        if cname:
            # 路线锁定时按 key 找（lineups.json 有 39 个重名阵容），含自定义路线（安琪拉 / 干将）
            pool = list(self.comps) + list(self.custom_comps)
            comp = (next((c for c in pool if self._route_key and c.get("key") == self._route_key), None)
                    or next((c for c in pool if c.get("name") == cname), None))
            if comp is not None:
                self._active_comp = comp
                self.main_factions = [comp["faction"]] if comp.get("faction") else []
                self._comp_switch_round = d.get("_comp_switch_round", self._comp_switch_round)
        if self._carry_piece in self._field:
            self._carry = self._field[self._carry_piece]
        logger.info("接着上一个进程的记忆：回合 %s，主C %s，枫叶舞绑 %s，场上 %s",
                    self._round_seen, self._carry_piece or "无", self._maple_bound or "无",
                    list(self._field))

    # ---------- 决策点记录（Laya 微调的训练数据） ----------
    # 每个决策点记：局面快照 + 所有候选（及规则打分/理由）+ 规则的选择。**只记录，不影响任何动作**。
    # 卡名原样记，英文语义特征赛后按 data/card_features.json 离线换算（特征格式还会迭代，别在线上固化）。
    # 同一回合同一决策点、同样的候选和选择只记一次（决策每帧都会重算）。

    def _snapshot(self, state: GameState) -> dict:
        lv = state.unit_levels or {}
        return {"round": state.round_no, "hp": state.hp, "gold": state.gold, "level": state.level,
                "pop": [state.population, state.population_cap, bool(state.population_valid)],
                "carry": self._carry_piece, "maple": self._maple_bound,
                "field": {n: {"q": self.pieces.get(n, {}).get("quality", 0),
                              "lv": lv.get(cell_key(tuple(p))), "value": self.unit_value(n),
                              # 觉醒还差几张 / 是否要留到最后（这两项要成为买牌依据、被 Laya 学习）
                              "awk_left": self.awaken_left(n, state), "keep": self._is_keeper(n),
                              "late": n in self._late_roles()}
                          for n, p in self._field.items()},
                "hand": [sl["name"] for sl in state.hand_slots or []],
                "banned": state.banned_faction}

    def record_decision(self, kind: str, state: GameState, options: list, chosen,
                        **extra) -> None:
        key = (kind, state.round_no, json.dumps(options, ensure_ascii=False, sort_keys=True, default=str),
               json.dumps(chosen, ensure_ascii=False, default=str))
        if key in self._decision_keys:
            return
        self._decision_keys.add(key)
        self.decisions.append({"kind": kind, "state": self._snapshot(state),
                               "options": options, "chosen": chosen, **extra})

    def note(self, event: str, **kw) -> None:
        """记一笔。账本是决策层的"我以为"，要定期跟画面对账（见 reconcile）。"""
        if event == "用牌" and kw.get("牌") in self.CORE_CARDS and kw.get("_executed"):
            self._core_seen = True
        if event == "用牌" and kw.get("牌") and not kw.pop("_executed", False):
            # **用牌等执行之后再记**（2026-10-13 第 606 局：规则决定"有福同享给干将"就记了账，实际 Laya 选了先留着 / 改给别人，
            # 复盘把它当成"资源给了被卖的人"）。先暂存，note_executed 里按实际执行的牌 / 目标落账
            self.__dict__.setdefault("_pending_use", {})[kw["牌"]] = kw
            return
        if event == "换主C收尾" and kw.get("卖") and not kw.pop("_executed", False):
            # 同上（2026-10-13 审计第 12 条，第 610 局：决定"卖干将换绑"就记了账，同一帧被 Laya swap 盖掉、实际卖的是牛魔）
            self.__dict__.setdefault("_pending_transfer", {})[kw["卖"]] = kw
            return
        rec = {"round": kw.pop("round", None), "event": event, **kw}
        self.ledger.append(rec)
        logger.info("账本: %s %s", event,
                    " ".join(f"{k}={v}" for k, v in kw.items() if k != "round"))

    def track_hand_gains(self, state: GameState) -> None:
        """对比上一帧手牌，记录**凭空多出来的牌**是怎么来的。

        效果牌、拍卖、登场效果都会往手里塞牌，但没有任何事件通知。
        只能靠"这一帧手上多了张上一帧没有的牌"来推断，并把当时刚用过的
        效果牌记为来源（通过什么效果牌拿到了什么英雄，要记下来）。
        """
        now = {s["name"] for s in state.hand_slots}
        gained = now - self._hand_prev
        if gained and self._hand_prev:          # 首帧不算
            # 只把**同回合刚用掉的效果牌**算作来源，别把无关事件硬扯成因果
            card, rnd = self._last_card_use
            used_src = card if (card and rnd == state.round_no) else ""
            for name in gained:
                if len(name) < 2 or name not in self.pieces:
                    continue          # OCR 残片（实测出现过单字"大"），不记进账本
                # 来源猜测按牌分别判：「N阶效果锦囊」的主要来源是
                # **合成英雄攒够次数系统白送**（规则事实），
                # 其余白来的牌多为拍卖/登场效果/回合奖励。
                # **卖掉带装备的英雄，装备会退回手牌**（规则事实）。
                # 这也意味着"卖人"的代价比看上去小：装备不会跟着没掉，能再穿给别人。
                sold_name, sold_round = self._last_sold
                cat_now = self.pieces.get(name, {}).get("category", "")
                if (not used_src and cat_now == "装备牌"
                        and sold_name and sold_round == state.round_no):
                    src = f"出售{sold_name}退回"
                else:
                    src = used_src or ("未知(合成赠送?)"
                                       if self.SELF_CAST_NAME_RE.match(name)
                                       else "未知(拍卖/登场效果/回合奖励?)")
                cat = self.pieces.get(name, {}).get("category", "?")
                tp = self._transform_pending
                if (tp and tp[0] == self._round_seen and cat == "英雄牌"
                        and (self.pieces.get(name, {}).get("quality") or 0) >= 5):
                    # 化铁为金变出来的**必定是觉醒卡**（卡面原文"觉醒并保留其当前等级"）——
                    # 金色卡面点开读 AWAKENED 不一定认得出，直接按卡面规则标上
                    self._transform_pending = None
                    self._awakened.add(name)
                    self._ever_awakened.add(name)
                    self._parked_cards.discard((self._round_seen, name))
                    src = tp[1]
                    logger.info("%s 是「%s」变出来的觉醒卡（等级保留），优先上场", name, tp[1])
                self.note("获得卡牌", round=state.round_no, 牌=name, 类别=cat, 来源=src)
        self._hand_prev = now

    def apply_board_observation(self, state: GameState) -> None:
        """拿**画面观测**补充记账。注意：**只做加法，不做减法。**

        2026-09-22 第八局的教训（这是我设计上的错误，不是实现 bug）：
        第一版让观测去**删除**记账里的位置，结果连续几帧 `1格→0格→2格→1格` 乱跳，
        苏烈刚被**拖拽回执直接证实**落在 (660,420)，下一帧一次抖动的观测就把记录删了 ——
        于是英雄被反复"删掉再重上"，整个第一回合没能把人留在场上。

        根因：**我让弱证据推翻了强证据**。
          - 拖拽回执 = 直接证据（那张牌真的离开手牌了）
          - 单帧霍夫圆 = 推断，会漏检（3D 单位有动画、徽章会被遮挡）
        而"漏检"和"那格真没人"在画面上长得一模一样，无法区分 —— 所以观测**不能**用来否定。

        现在只保留安全的那一半：
          - 观测到某格有人、而且不是我记的任何人 → 标进 `_busy_cells`，别往那儿放
          - 记录和观测对不上 → **什么都不做**，交给拖拽回执去证伪（它拖失败自然会撤销）
        漏检的代价只是"可能往占着的格子放一次"，回执会兜住；
        误检的代价只是"少用一个空格"。两边都有界，不会再出现自毁记录。
        """
        if not state.board_seen:
            return
        occ = {tuple(c) for c in state.board_occupied}
        # 要求**连续两次**观测到同一格才作数：单帧会抖，抖一下不能当事实
        stable = occ & self._board_prev
        self._board_prev = occ
        mine = {tuple(p) for p in self._field.values()}
        for cell in stable - mine:
            if cell not in self._busy_cells:
                self._busy_cells.add(cell)
                logger.info("观测（连续两帧确认）：%s 上有我不认识的单位，标为占用", cell)
                self.note("观测-标记占用", round=state.round_no,
                          位置=f"{cell[0]},{cell[1]}")

    def reconcile(self, state: GameState) -> None:
        """拿画面读数核对账本：人口数 vs 我记录的场上英雄数。

        `_field` 只是"我拖过什么上去"的记忆，拖失败、被卖、觉醒回手、战斗阵亡
        都不会自动同步。人口是画面上的**客观读数**，对不上就说明账本错了。
        对不上时宁可保守修正并报警，也不要继续拿错数据做决策。
        """
        if not state.population_valid:
            return      # 本帧没真读到人口（商店挡着 / OCR 失败）——
                        # 拿缓存或假数据对账会把正确的账本记录删掉，反而更糟
        n_field = n_before = len(self._field)
        n_real = state.population
        if n_field == n_real:
            self._pop_low_pending = None
            return
        if n_field > n_real:
            # **删记录是破坏性的，必须连续两帧读到同一个更小的人口才动手。**
            # 单帧就删的后果（2026-09-23 实测）：人口偶尔读成 1/5（真值 6/6），
            # 一帧就把 5 条记录抹掉 —— 自评报告里"结束时场上 1 人 / 人口 6/6"、
            # "对账异常 9 次"就是这么来的。账本一残，卖人换人全都做不了
            # （连自己有谁都不知道），手牌自然越堆越多。
            # 慢一帧的代价只是晚一点修正；删错的代价是整局决策失去依据。
            if self._pop_low_pending != n_real:
                self._pop_low_pending = n_real
                logger.info("人口读到 %d < 账本 %d，等下一帧确认再修正账本",
                            n_real, n_field)
                return
            self._pop_low_pending = None
            # 我记多了：优先删掉手牌里出现的（觉醒回手），其次删最近加进来的
            # **只有确认过的觉醒卡才算"回到手牌"**（2026-09-24 第二轮第 5 局：场上嬴政是主C，同名卡却没用来合成升级）：
            # 刚买的嬴政是合成材料，对账却按"手里有同名 = 回手"把场上的主C 嬴政删了 → 卖蒙恬给它腾位、主C 换成敖隐。
            # 手里有同名牌最常见的原因是合成材料（sync_field 那条 09-23 已经收紧过，这里是同一个坑的另一个入口）。
            hand = {s["name"] for s in state.hand_slots if s.get("awakened") or s["name"] in self._awakened}
            protect = lambda n: (n == self._carry_piece  # noqa: E731
                                 or self._confirmed_at.get(n, (None,))[0] == self._round_seen)
            for name in list(self._field):
                if n_field <= n_real:
                    break
                if name in hand and not protect(name):
                    self._field.pop(name); n_field -= 1
                    self.note("对账修正-移除", round=state.round_no, 英雄=name, 原因="觉醒回到手牌")
            # 删不认识的：**先删视觉这一帧在它格子上没看到人的**（2026-09-24 第 14 局复盘：原来按"最近加进来的先删"，
            # 画面上明明站着的人被删、真正空掉的那格记录还留着）。D-FINE 格子召回 99.3%，比"加入顺序"强得多。
            # 其次：本回合没点开确认过、不是主C 的，最近加进来的先删。
            vis = (set(state.vision_units or {}) | set(state.unit_anchors or {})) if (
                state.vision_units and not state.shop_open) else set()
            seen_now = lambda n: cell_key(tuple(self._field[n])) in vis  # noqa: E731
            rest = list(reversed(list(self._field)))
            # 主C / 本回合点开确认过的人**不靠人数删**（2026-10-15 第 634 局 R13：人口读成 5 < 账本 7，差 2 个，删到了主C 马可波罗（106 级、
            # 已觉醒，人一直在场）→ 换主C 题里没有他、局面网络选了 30 级安琪拉 → 换绑枫叶舞时把马可波罗卖了）。
            # 他们只由视觉连续多帧没看到（vision_absence）或我们自己卖 / 变身才离场；人数对不上就先留着差额
            order = ([n for n in rest if vis and not seen_now(n) and not protect(n)]
                     + [n for n in rest if not (vis and not seen_now(n)) and not protect(n)])
            for name in order:
                if n_field <= n_real:
                    break
                self._field.pop(name); n_field -= 1          # 按人数对账是弱证据：只删场上记录，别的记忆不动
                self.note("对账修正-移除", round=state.round_no, 英雄=name, 原因="画面上不存在")
            logger.warning("账本与画面不一致：原记录 %d 个，画面 %d 个 —— 已修正为 %d 个",
                           n_before, n_real, len(self._field))
            return
        # 画面比账本多：**这一支修不了**——只知道多了几个，不知道多的是谁、在哪格。
        #（来源：召唤物、开局自动上场、之前拖成功但回执丢了…）
        # 原来这里每帧刷一条 "已修正为 N 个" 的 WARNING，既没修也不准，
        # 2026-09-22 第二局刷了几十条把真正的告警淹了。改成如实说 + 每回合只报一次。
        sig = (state.round_no, n_field, n_real)
        if sig == self._reconcile_sig:
            return
        self._reconcile_sig = sig
        # **先看多出来的是不是召唤物**：召唤物占格子但不占人口，账本里也没有它
        # （账本只记"我拖上去的牌"）。场上有召唤系英雄时，多出来几个是**正常的**，
        # 当成异常报会把真正的告警淹掉 —— 2026-09-22 第二局就是几十条 WARNING 刷屏。
        # **刚卖掉人的那几帧不算异常**：卖出会立刻把记录扣掉，而画面观测要过几帧
        # 才跟上（棋子有消失动画）。2026-09-23 实战：同一回合卖了 3 个人，
        # 账本 4→3→2 而画面一直是 5，刷出 10 条"对账异常" —— 全是误报，
        # 把自评报告里这项指标污染掉了。
        sold_name, sold_round = self._last_sold
        if sold_name and sold_round == state.round_no:
            logger.info("场上比账本多 %d 个，但本回合刚卖掉 %s —— 画面还没更新，不算异常",
                        n_real - n_field, sold_name)
            return
        summoners = [n for n in self._field
                     if "召唤" in self.pieces.get(n, {}).get("skill", "")]
        if summoners and n_real - n_field <= len(summoners):
            logger.info("场上比账本多 %d 个，场上有召唤系英雄 %s —— 多出来的多半是召唤物，不算异常",
                        n_real - n_field, "/".join(summoners))
            return
        logger.warning("场上比账本多：账本 %d 个，画面 %d 个 —— 无法定位多出来的是谁，"
                       "落点只能靠拖拽回执试出来", n_field, n_real)
        self.note("对账异常", round=state.round_no, 账本=n_field, 画面=n_real,
                  召唤系=summoners,
                  说明="场上比记录的多，且召唤物解释不了")

    def _remember_pop(self, state: GameState) -> None:
        """记下最近一次真读到的人口，只给日志用（决策一律只认本帧读数）。"""
        if state.population_valid and state.population_cap > 0:
            self._last_pop_seen = (state.population, state.population_cap)

    # ---- 当前场上人数（2026-10-19，第 751 局 R9 复盘）：所有"满没满 / 还剩几格"都走 pop_now ----
    AWK_EXPECT_S = 12.0          # 觉醒进度牌打成功、按读数应当觉醒 → 这么多秒内当"他马上回手"

    def _pop_ledger_n(self) -> int:
        """账本里占人口的人数（不含不占人口的单位，如天赋送的始祖熊灵）。"""
        return sum(1 for n in self._field if not (self.pieces.get(n) or {}).get("no_pop"))

    def _awk_expected_gone(self) -> list:
        """打了觉醒进度牌、应当已经觉醒回手、但账本里还记着在场的人（确定事件，等手牌认出来之前先按"已离场"算）。"""
        exp = self.__dict__.get("_awk_expect") or {}
        now = time.time()
        return [n for n, (rn, t) in exp.items()
                if n in self._field and rn == self._round_seen and now - t < self.AWK_EXPECT_S]

    def pop_now(self, state: GameState) -> tuple:
        """**当前场上人数 / 人口上限 / 来源**（唯一口径，决策点都用它判"满没满"）。

        - 本帧真读到（population_valid：商店 / 拍卖关着、不在战斗）→ 信读数（视觉为主，一致的读数推翻账本），钳在 [0, 上限]。
        - 读不到 → 以**最近一次真读到的人数**为锚，加上之后账本里的确定变化（卖人、上场回执、觉醒回手、移除类牌）：
          锚 + (账本人数 − 锚定时的账本人数)；本局还没读到过就用账本人数。再扣掉"觉醒进度牌已打成功、马上回手"还在账上的人。
          state.population 在读不到时是感知层沿用的旧值（复盘表里的"(上次读到)"），**不直接用**。
        - 上限：本帧 / 感知层沿用的上限 → 锚里的上限 → 0（未知；调用方自己给默认）。人数一律钳在上限以内（物理上界）。
        返回 (here, cap, src)，src ∈ {"读数", "读数+账本", "账本"}。"""
        cap = int(state.population_cap or 0)
        a = self.__dict__.get("_pop_anchor")
        if cap <= 0 and a:
            cap = a[1]
        if state.population_valid and (state.population_cap or 0) > 0:
            here = max(0, int(state.population or 0))
            return (min(here, cap) if cap > 0 else here), cap, "读数"
        n_led = self._pop_ledger_n()
        gone = len(self._awk_expected_gone())
        if a:
            here, src = a[0] + (n_led - a[2]) - gone, "读数+账本"
        elif self._last_pop_seen:
            # 没有锚（只在离线场景 / 刚重启时出现）：退回上一次真读到的（觉醒回手 / 变身时已就地 −1）
            here, src = self._last_pop_seen[0] - gone, "上次读数"
            cap = cap if cap > 0 else self._last_pop_seen[1]
        else:
            here, src = n_led - gone, "账本"
        here = max(0, here)
        return (min(here, cap) if cap > 0 else here), cap, src

    def pop_full(self, state: GameState) -> bool:
        """满员（上限未知时算没满 —— 和原来 `cap and here >= cap` 同口径）。"""
        here, cap, _ = self.pop_now(state)
        return cap > 0 and here >= cap

    def _pop_mark(self, state: GameState) -> None:
        """每帧对账结束后记锚：本帧真读到人口时，记 (人数, 上限, 当时账本人数)。必须在 reconcile / sync_field 之后调，
        否则对账删掉的记录会被当成"之后又少了一个"。"""
        if state.population_valid and (state.population_cap or 0) > 0:
            here = max(0, min(int(state.population or 0), int(state.population_cap)))
            self._pop_anchor = (here, int(state.population_cap), self._pop_ledger_n(), state.round_no)

    def awaken_expected(self, state: GameState) -> list:
        """觉醒进度牌刚打成功、按徽章读数这一张就该觉醒的人（他会回到手牌）：确认前别让别的英雄占掉他空出来的格子。"""
        if state.phase != "prepare":
            return []
        return [n for n in self._awk_expected_gone() if (self._round_seen, n) not in self._awk_cleared]

    def note_executed(self, act: dict[str, Any]) -> None:
        """**动作真的点出去之后**才记账。

        2026-09-22 实战教训：开关商店的"每回合最多 2 次"预算原来在**产生决策**时就加，
        而执行层有一条"同一动作 6 秒内不重发"的去重 —— 于是预算被
        **从没真正发出去的点击**耗光，整局商店没打开过、一张英雄都没买。
        **凡是"有限次数"的预算，都必须按实际执行计数，不能按决策计数。**
        """
        if act.get("type") in self._SPEND_TYPES:
            self._spend_seq = self.__dict__.get("_spend_seq", 0) + 1
        _pc = act.get("piece") or ""
        _pu = (self.__dict__.get("_pending_use") or {}).pop(_pc, None) if _pc else None
        if _pu is not None and act.get("type") in ("drag_card", "use_card", "equip"):
            _kw = dict(_pu)
            if act.get("target"):
                _kw["目标"] = act["target"]          # 以实际执行的目标为准（Laya 可能改给了别人）
            self.note("用牌", _executed=True, **_kw)
        if act.get("type") == "card_pick" and _pc:
            self.note("N选1", round=self._round_seen, 选择=_pc, 候选=self.__dict__.get("_pending_pick") or [])
        _pt = (self.__dict__.get("_pending_transfer") or {}).pop(_pc, None) if _pc else None
        if _pt is not None and act.get("type") == "sell_unit":
            self.note("换主C收尾", _executed=True, **_pt)
        _pc = act.get("piece") or ""
        if (act.get("type") in ("move_unit", "drag_card") and _pc and self._carry_buddy(_pc)
                and self.pieces.get(_pc, {}).get("category") == "英雄牌" and not act.get("merge")):
            # 贴主C 的英雄（李元芳"对距离最近的【河洛】英雄…"）**放下那一刻**定连接对象：记下当时的主C 和它的格子
            _cp = tuple(self._field.get(self._carry_piece) or ())
            _dst = tuple(act.get("dst") or ())
            _oth = self._buddy_others(_pc, {})        # 派系限定的（李元芳只连河洛）只比同派系
            # 放下的格子主C 不是**唯一**最近的人 → 连接可能落在别人身上，记成"未连主C"，之后站位纠正会挪到对的格子再刷新
            _ok = bool(_cp and _dst and self._buddy_cell_score(_dst, _cp, _oth)[0] == 0)   # 主C 是唯一最近的人（不论几步）
            _old = (self.__dict__.get("_buddy_link") or {}).get(_pc)
            if act.get("keep_link") and self._link_sticky(_pc) and _old and _old[0] == self._carry_piece and _cp and _dst                     and self._hex_dist(_dst, _cp) <= min((self._hex_dist(_dst, q) for q in _oth), default=9):
                _ok = True       # 从"已连主C"的中转格挪过来、主C 仍是最近的人之一（并列）→ 连接不变，仍连主C
            self.__dict__.setdefault("_buddy_link", {})[_pc] = ((self._carry_piece if _ok else None), _cp)
        if act.get("type") == "lord_boost_pick" and act.get("name"):
            self._lord_boosts.add(act["name"])
        if act.get("type") == "talent_pick" and act.get("name"):
            # 记下选过的天赋（多彩古币会改写古币的合法目标，见 card_rule_mod）
            self._talents_taken.add(act["name"])
            try:
                # 2026-10-22 第 906 局：幸运硬币这类天赋可以作为特殊路线的定死天赋，选上就直接锁定狄仁杰
                from agent.decision import talent_conditions as _tlk, lord_config as _LCk
                _lc = (_tlk.table().get(act["name"]) or {}).get("lock_route_carry")
                _sc = self._carry_piece or ""
                _settled = (self.final_carry_settled() and _sc in self._field
                            and ((self._unit_lv.get(_sc) or 0) >= 10 or self._maple_bound == _sc))
                if _lc and not self.route_locked() and not _settled:
                    _key = _LCk.load().lineup_key(_lc)
                    if _key and self.set_route(_key, self._round_seen or 0, why=f"天赋 {act['name']} 定死 {_lc} 路线"):
                        logger.info("选了 %s → 直接锁定 %s 路线（%s）", act["name"], _lc, _key)
                        self.__dict__["_talent_route_carry"] = _lc
                        self.__dict__["_group_settled"] = True       # 主C 组"谁先出用谁"不再改这条路线        # 10-22：到了就换主C——他一上场就接任（不受"名单外主C 给了就认"的例外）
                        self.note("天赋定路线", 天赋=act["name"], 主C=_lc, 路线=_key)
            except Exception:                               # noqa: BLE001
                logger.exception("天赋定路线失败")
            if act["name"] == "全域共鸣":
                from agent.decision import hero_traits as _htg
                _htg.GLOBAL_RESONANCE = True        # 星源装备的日落海专属效果对任意英雄生效（复审 v2 发现线上没接）
            if self.card_rule_mods():
                logger.info("天赋 %s 改写了出牌规则：%s", act["name"], self.card_rule_mods())
        if act.get("type") == "drag_card" and act.get("predict"):
            self._predict_round = self._round_seen          # 一回合只押一次预言（点出去才算）
        if act.get("type") == "drag_card" and act.get("entry_boost"):
            self._boost_round = self._round_seen
        if act.get("type") == "drag_card" and act.get("piece"):
            _cu = self.__dict__.setdefault("_card_uses", {})
            _cu[(self._round_seen, act["piece"])] = _cu.get((self._round_seen, act["piece"]), 0) + 1
        if act.get("type") == "talent_pick" and "古币机" in (act.get("name") or ""):
            self.coin_machine_wanted = True          # 召唤了古币机：之后去找它在哪一格（find_coin_machine）
        if act.get("type") == "buy_xp":
            # **所有路径的升级都在真正点出去之后记**（规则 / preempt 进度表 / 搜牌等级 / Laya / swap）
            self._leveled_round = self._round_seen
            for n in self._field:                    # 场上的棋手经验件各记一次贡献
                if self._is_level_engine(n):
                    self._engine_levels[n] = self._engine_levels.get(n, 0) + 1
        if act.get("type") == "refresh_shop" and str(act.get("why") or "").startswith("搜牌回合"):
            self.__dict__.setdefault("_roll_rounds", set()).add(self._round_seen)   # 真刷了才算用掉一个搜牌回合
        if act.get("type") in ("talent_pick", "drag_card", "use_card") and act.get("name", act.get("piece")):
            # **天赋/效果牌会加人口上限**（2026-09-24 数据局：选了「战术模组」可上阵人数+1，画面 7/8，
            # 而感知层把上限硬限定在 5/6/7，8 被当成误读压成 7/7 → 以为满员，第 8 个位置一直空着）。
            # 读的是卡牌知识库的结构化收益（gives: 人口 n），不是正则。"花 8 能量可买"那种不是立即生效，不算。
            n = self._kb_pop_gain(act.get("name") or act.get("piece"))
            if n and act.get("type") != "talent_pick":
                # 拖牌要等回执成功才加（2026-10-13 卡面审计第 5 条：拖失败重试一次就成了 +2，满员还往场上拖人）。
                # 回执"判不了"时不加 —— 画面读到更大的上限由 extractors.pop_bonus_seen 兜底
                self.__dict__.setdefault("_pending_pop", {})[act.get("piece")] = n
                n = 0
            if n:
                self.pop_bonus += n
                self.pop_bonus_ledger = self.__dict__.get("pop_bonus_ledger", 0) + n
                logger.info("%s：人口上限 +%d（累计 +%d）", act.get("name") or act.get("piece"), n, self.pop_bonus)
        if act.get("type") == "sequence" and act.get("piece"):
            # **组合动作（夺取主C 的"挪人-合成-挪回"）也要"试够次数就停"**（第十四局：没有回执、也不计数，
            # 每 12 秒重做一遍，整回合只干这一件事）
            k = (self._round_seen, act["piece"])
            self._seq_tries[k] = self._seq_tries.get(k, 0) + 1
            if self._seq_tries[k] >= 2 and k not in self._parked_cards:
                self._parked_cards.add(k)
                logger.warning("%s 的组合动作本回合已做 %d 次，先搁置", act["piece"], self._seq_tries[k])
        # ---- 手牌账：买进 / N 选 1 选中 = 进手牌；卖手牌 = 出手牌（拖出去的在 confirm_drag 里减）
        if act.get("type") in ("buy_piece", "card_pick") and act.get("piece") in self.pieces:
            self._hand_book[act["piece"]] += 1
        if act.get("type") == "sell_card" and act.get("piece") and self._hand_book[act["piece"]] > 0:
            self._hand_book[act["piece"]] -= 1
        if act.get("type") == "sell_card" and act.get("piece"):
            if self.pieces.get(act["piece"], {}).get("category") == "英雄牌":
                self._sold_round[act["piece"]] = self._round_seen      # 卖掉的英雄手牌也别马上买回来
            k = (self._round_seen, act["piece"])
            self._sell_tries[k] = self._sell_tries.get(k, 0) + 1
        if act.get("type") == "refresh_shop":
            r = self._round_seen
            self._refresh_streak = (r, (self._refresh_streak[1] + 1) if self._refresh_streak[0] == r else 1)
            _rn = self.__dict__.get("_refresh_round_n") or (r, 0)
            self.__dict__["_refresh_round_n"] = (r, _rn[1] + 1 if _rn[0] == r else 1)      # 本回合刷新次数（core_roll_first 上限用）
        if act.get("type") == "buy_piece":
            self._buys_this_round += 1
            self._refresh_streak = (self._round_seen, 0)
            self.note("买牌", round=act.get("round", -1), 牌=act.get("piece"),
                      花费=act.get("cost"),
                      花费来源="估(英雄牌统一3)" if act.get("cost_est") else "读到")
            return
        if act.get("type") == "sell_unit":
            # **卖人的账在这里落**：决定卖的时候不动账本（执行可能失败/被去重）。
            name = act.get("piece")
            self._sell_undo = {"name": name, "pos": self._field.get(name), "maple": self._maple_bound,
                               "awakened": name in self._awakened, "fodder": self._entry_fodder.get(name),
                               "worn": self._worn.get(name), "freed": self._freed_cell, "last_sold": self._last_sold,
                               "prev_carry": self._prev_carry,
                               "equip_seen": self._equip_seen.get(cell_key(tuple(self._field[name])))
                               if self._field.get(name) else None}
            if name and name == self._maple_bound:
                self._maple_bound = ""             # 被绑的人卖掉了，枫叶舞可以重绑（给新主C）
            if name and name == self._prev_carry:
                self._prev_carry = ""              # 老主C 真卖掉了，装备退回手牌
            self._awakened.discard(name)           # 卖掉了就不再是"等着重新上场的觉醒卡"
            # 召唤者卖掉，他的召唤物一起消失（规则事实：卖掉鲁班大师，召唤物一起消失）→ 那些格子放出来
            for _oc, _on in list(self.object_cells.items()):
                if name and self.summoner_of(_on) == name:
                    self.object_cells.pop(_oc, None)
                    logger.info("%s 卖掉，召唤物「%s」一起消失，%s 放出来", name, _on, _oc)
            pos = self._field.pop(name, None)
            if pos:
                self._cell_owner.pop(tuple(pos), None)   # 卖掉了，格子真的空出来
                self._equip_seen.pop(cell_key(tuple(pos)), None)   # 装备退回手牌了
                self._equip_rejects.pop(cell_key(tuple(pos)), None)
            self._entry_fodder.pop(name, None)
            self._last_sold = (name, self._round_seen)
            if pos:
                # **刚卖掉的人站的那一格是"确定空着"的**（我们亲手把他拿走的），
                # 比任何"没观测到人"的格子都可靠。下一次上场优先用它 —— 第七局就是
                # 卖掉上官婉儿腾出 (617,372)，觉醒米莱狄却被派去账本以为空、
                # 实际站着大司命的 (1027,572)，连撞两次被搁置。
                self._freed_cell = (tuple(pos), self._round_seen)
            self._worn.pop(name, None)            # 卖人时装备退回手牌
            self._sold_names.add(name)
            self._sold_round[name] = self._round_seen
            self.note("卖出", round=self._round_seen, 英雄=name,
                      原因=act.get("why", ""))
            return
        if act.get("type") in ("drag_card", "move_unit"):
            self._note_drag_attempt(act)
            return
        if act.get("type") != "toggle_shop":
            return
        if act.get("intent") == "close":
            self._shop_closes += 1
        else:
            self._shop_opens += 1

    # 同一张牌、同一个落点，一个回合里最多**真正点出去**几次。
    # 这条和 CARD_FAILS_PER_ROUND 的区别是**它不需要回执**：
    # 2026-09-23 实战，白起往 (319,375) 合成，拖拽回执每次都返回"判不了"
    # （拖完那一瞬手牌读成空），于是失败计数永远是 0、永远不搁置 ——
    # 整个回合 6 的十几帧全在重发同一个动作，装备没贴、10 能量没花、牌没买。
    # 教训：**任何"失败才停"的止损都要配一条"试够次数就停"**，
    # 否则一旦判不出成败，止损逻辑整条失效。
    DRAG_ATTEMPTS_PER_ROUND = 3

    def _note_drag_attempt(self, act: dict[str, Any]) -> None:
        piece = act.get("piece")
        dst = tuple(act.get("dst") or ())
        if not piece:
            return
        key = (self._round_seen, piece, dst)
        n = self._drag_attempts.get(key, 0) + 1
        self._drag_attempts[key] = n
        if n < self.DRAG_ATTEMPTS_PER_ROUND:
            return
        logger.warning("%s -> %s 本回合已经点出去 %d 次还在重发（回执判不出成败），"
                       "先搁置这张牌", piece, dst, n)
        self._parked_cards.add((self._round_seen, piece))
        if (act.get("merge") and self._field.get(piece) == dst
                and self._confirmed_at.get(piece) == (self._round_seen, dst)):
            # **本回合点开确认过它就在这一格**：试够次数只搁置，不撤销记录（2026-09-23 第十四局：
            # 手里有多张曜，第 2 次合成成功、第 3 次触发止损 —— 把刚点开确认过的主C 撤销了，主C 被换成孙膑）
            logger.info("%s 本回合点开确认过就在 %s，只搁置、不撤销记录", piece, dst)
        elif act.get("merge") and self._field.get(piece) == dst:
            # 合成反复不成 = 它多半不在那一格（阵亡/觉醒回手/我记错了）。
            # 撤掉记录让它重新走上场流程，和 confirm_drag 里合成失败的处理一致。
            logger.warning("撤销 %s 在 %s 的记录，让它重新走上场（多半是觉醒回手）", piece, dst)
            self._field.pop(piece, None)
            self._cell_owner.pop(dst, None)
            self._awakened.add(piece)     # 和 confirm_drag 的合成失败分支同一套判断
        # **也给这个格子记一笔失败**。两套止损原来是断开的：
        # 按牌的搁置是**按回合**的，下回合从头再来；而回执一直"判不了"时
        # `_cell_fails` 永远不涨 —— 于是「八方来援」「翩跹舞步·进化」每回合
        # 往 (672,430) 试 3 次、被搁置、下回合再试，整局重复 5 轮（2026-09-23 实测）。
        # 拖出去了却什么都没发生，本身就是"这个落点有问题"的证据。
        if dst:
            self._cell_fails[dst] = self._cell_fails.get(dst, 0) + 1
        self.note("重复无效动作", 牌=piece, 落点=f"{dst[0]},{dst[1]}" if dst else "",
                  次数=n, 处理="搁置本回合，并给该格记一次失败")

    # 商店面板盖住棋盘的**下边界**（实测：卡面底边 y≈388，"当前概率"行 y≈418）。
    # y 小于这个值的格子在商店开着时是点不到的 —— 往那儿拖等于拖到面板上。
    SHOP_PANEL_BOTTOM_Y = 440

    def _hero_cell(self, state: GameState, prefer: str = "",
                   allow_disposable: bool = True):
        """**"往某个英雄身上送东西"时统一用这个挑格子。**

        今晚（2026-09-23）同一个 bug 以三种面貌复发了三次：上场落点、装备/效果牌目标、
        自用牌目标 —— 因为"挑哪一格"在三条路上各写了一份，给其中一条加了判据、
        另外两条就漏。每次复发的样子都一样：**一张牌反复撞同一个坑十几次**。
        所以把判据收拢到这里，三条路共用：

          1. 跳过**本局已拖失败多次**的格子（`_cell_dead`）——直接证据，那儿放不进去
          2. 跳过**商店面板底下**的格子——这一帧根本点不到
          3. **优先选观测到确实有人的格子**（等级徽章锚点）——记账会过期，画面不会
          4. 都不满足时退回记账位（观测会漏检，不能因为看不见就瘫掉）

        `prefer` 给出希望优先的英雄名（通常是主力）。
        """
        seen = set(self._anchor_cells(state))

        def usable(cell):
            return (cell is not None and not self._cell_dead(cell)
                    and not self._covered_by_shop(state, cell))

        names = ([prefer] if prefer in self._field else []) +                 [n for n in self._field if n != prefer]
        if not allow_disposable:
            names = [n for n in names if not self._is_disposable(n)] or names
        # 先要"画面上确实有人"的，没有再退回记账
        for want_seen in (True, False):
            for n in names:
                pos = self._field.get(n)
                if not usable(pos):
                    continue
                if want_seen and seen and tuple(pos) not in seen:
                    continue
                return pos
        return None

    def _cell_dead(self, cell) -> bool:
        """这一格本局已经拖失败够多次，别再往它上面送东西。

        `_cell_fails` 本来只用在**上场**的落点筛选里，而装备/效果牌的目标走的是
        "主力记账位"，完全绕过了它 —— 2026-09-23 实战：(672,430) 一格连续失败
        **13 次**（用牌 28 次里大半都撞在它身上），而按牌记的止损
        （CARD_FAILS_PER_ROUND）只管得住"同一张牌"，管不住"同一个坑"。
        """
        if cell is None:
            return False
        _uc = (self.__dict__.get("_unclear_cells") or {}).get((self._round_seen, tuple(cell)))
        if _uc and (_uc[0] >= self.UNCLEAR_LIMIT or time.time() - _uc[1] < self.UNCLEAR_RETRY_S):
            return True     # 拖前核对看不清是谁（点歪）：几秒内先别往这格送东西；同一回合累计 3 次才整回合放弃
        return self._cell_fails.get(tuple(cell), 0) >= self.CELL_FAIL_LIMIT

    UNCLEAR_RETRY_S = 3.0
    UNCLEAR_LIMIT = 3

    def note_unclear_cell(self, cell) -> None:
        """拖前核对：那格看到的人账本记在隔壁（多半点歪）→ 本回合不再把它当目标（2026-10-08 第 435 局：古币每帧瞄 (390,502)
        被拦，三次后整张牌搁置，只能手动打出）。"""
        # 2026-10-21 第 887 局 R9：认人模型一帧把守约那格看成隔壁李元芳 → 整回合不再瞄主C，古币压在手里到开战。
        # 改：3 秒后重试（下一次拖前核对会再看一次），同一回合同一格累计 3 次才整回合放弃
        d = self.__dict__.get("_unclear_cells")
        if not isinstance(d, dict):
            d = self.__dict__["_unclear_cells"] = {}
        k = (self._round_seen, tuple(cell))
        d[k] = [(d.get(k) or [0])[0] + 1, time.time()]

    def _covered_by_shop(self, state: GameState, cell) -> bool:
        """这一格现在是不是被商店面板盖着。

        为什么需要：`_shop_blocks_board` 有个逃生口——"关店点了 2 次还开着就带着商店硬干"，
        本意是别让整回合空转。但 2026-09-23 实战证明**它造成的正是它想避免的伤害**：
        第 1 回合商店没关上，于是把 曜 上场和 破碎圣杯 贴装备都拖到了 (617,372)，
        而那一格**正在商店面板底下**（证据帧：面板盖到 y≈430，棋盘上一个单位都没有）。
        拖拽回执还报"成功"（牌确实离手了），于是账本记下一个根本不存在的落点。
        现在：硬干可以，但只用**面板下方**的格子。
        """
        return bool(state.shop_open and cell and cell[1] < self.SHOP_PANEL_BOTTOM_Y)

    def _shop_blocks_board(self, state: GameState) -> bool:
        """商店面板开着 = 现在不该动手牌/棋盘。

        使用手牌前先关掉商店，
        避免误判界面。理由：
          - 面板盖住棋盘上半部分，落点要么在面板上、要么在被挡住的区域；
          - 面板占着商店那块屏幕，手牌/人口/棋盘的读数在那几帧都不可信。
        原来只有"有待办就关店"这一条路，**兜底路径**（待办判定为空、或者买不动了
        走到决策函数列表）仍然可能开着商店就拖牌。

        例外：**已经点了 2 次都没关上**。那种情况下继续等于整回合什么都不做，
        宁可带着商店硬干（这条兜底本来就在 decide_all 里，这里保持一致）。
        """
        return state.shop_open and self._shop_closes < 2

    def _board_work_pending(self, state: GameState) -> bool:
        """手上还有没有**现在真能做**的棋盘操作。

        关键是"能不能做"，不是"有没有牌"：场上一个英雄都没有时，
        装备牌/效果牌/棋手技能牌全都无处可用——它们不该算作"待办"。
        否则会死锁：待办永远清不掉 → 商店永远不开 → 买不到英雄 →
        场上永远是空的（2026-09-22 实战第 1 回合就卡在这里，整局没买过牌）。
        """
        if self._entry_fodder:
            return True
        if self._transfer_victim(state)[0]:
            return True          # 换主C 收尾（卖人换绑枫叶舞 / 转装备）要关店做
        has_field = bool(self._field)
        room = (not state.population_valid
                or state.population < state.population_cap)
        counts = dict(state.equip_counts or {})
        for k, n in self._equip_seen.items():       # 单帧读不到的，用看到过的最大值
            counts[k] = max(counts.get(k, 0) or 0, n)
        equip_room = (not counts) or any(
            n < meta.MAX_EQUIPS for n in counts.values())
        for slot in state.hand_slots:
            name = slot["name"]
            # **本回合已经搁置的牌不是"待办"**：它不会再被打出去，
            # 却会一直把"还有棋盘活要干"顶成 True，于是商店永远不开。
            if self._parked(name):
                continue
            info = self.pieces.get(name, {})
            cat = self._use_category(name, info)
            if cat == "英雄牌":
                # 同名可合成（不占人口），或有空位可上场
                if name in self._field or room:
                    return True
                # **觉醒卡**：没空位也算待办 —— 前提是**真能腾出位来**
                # （场上有可卖的、且它明显更强，和 decide_sell_unit 同一套判据）。
                # 不加前提的话，场上全是主力/核心时它会永远算待办，商店再也开不了
                # （「待办永远清不掉 → 商店永远不开」那类死锁）。
                if slot.get("awakened"):
                    weakest = self._weakest_on_field()
                    if (weakest and not (self._is_carry(weakest[0]) or self._is_core(weakest[0]))
                            and self.unit_value(name) > weakest[2] + self.SWAP_MARGIN):
                        return True
            elif cat == "棋手技能牌":
                # **枫叶舞用完仍然留在手牌里**（机制，不是 bug）。
                # 原来它无条件算一件"待办"，于是一旦绑定完成，
                # `_board_work_pending` 就**永远是 True** —— 关了商店再也开不回来，
                # 整个回合不刷新、不买牌，能量一路攒到二十几点
                # （2026-09-23 实战回合 6~8 实测）。
                # 判据和 decide_use_cards 保持一致：卡面灰了、或已经绑在场上某人身上，
                # 就不是待办。
                if slot.get("ready") is False:
                    continue
                if self._maple_bound and self._maple_bound in self._field:
                    continue
                if has_field:
                    return True
            elif cat in ("装备牌", "效果牌"):
                if not has_field:
                    continue           # 场上有人才谈得上给谁用
                text = name + info.get("skill", "")
                if any(k in text for k in self.HOLD_KEYWORDS):
                    continue           # 整备类是留牌，本回合不出
                if cat == "装备牌" and not equip_room:
                    continue           # 场上人人 3 件满了，这张贴不上去
                return True
        return False

    # 同一个格子拖失败几次就当它整局有人。1 次就够狠了吗？——不够：
    # 拖拽本身偶尔会因为动画/时机失败。2 次是"两次独立证据"，代价只是少用一个格子。
    CELL_FAIL_LIMIT = 2
    # 手里压着这么多"上不了场的英雄"时，场面满了就别再买新英雄（见 _worth_buying）
    IDLE_HAND_HEROES = 3
    # 同一组 N选1 候选最多点几次（执行器会按次数换三个落点）
    CARD_PICK_MAX_TRIES = 3

    # ---- 动态选主C----
    # 推荐阵容按使用人数加权：主C 4 阶 51% / 5 阶 26% / 2 阶 12% / 3 阶 10% / 1 阶 0%。
    # 主C 不能一直停留在开场时的 1 阶英雄，要控制在 3 阶左右、能升到高等级的英雄。
    # 5 阶不给最高分：刷得出来要 5 级、每位只有 6 张，现实里很难成型（所以取"3 阶左右"）。
    CARRY_TIER_SCORE = {1: 0, 2: 18, 3: 30, 4: 32, 5: 30}
    CARRY_SWITCH_MARGIN = 20

    def _carry_score(self, name: str, state: GameState) -> Optional[float]:
        info = self.pieces.get(name, {})
        if not self._carry_eligible(name) or self._is_disposable(name):
            return None
        s = float(self.CARRY_TIER_SCORE.get(info.get("quality", 0), 0))
        if info.get("role") == "dps":
            s += 10                                   # 主C 是输出位
        pos = self._field.get(name)
        lv = (state.unit_levels or {}).get(f"{pos[0]},{pos[1]}") if pos else None
        if lv:
            s += min(lv, 80) * 0.6                    # **已经练起来的等级**（徽章读数），10/40/100 级技能质变
        s += 12 * self._lord_carry_freq.get(name, 0.0)   # 阿离阵容里常当主C（百里守约最高）
        if self._any_carry_freq:
            f = self._any_carry_freq.get(name)
            s += 12 * f if f else -10                 # 455 套里从没当过主C 的（姬小满这种功能牌）扣分
        if name in self._ever_awakened:
            s += 15                                   # 觉醒：词条翻倍
        s += 4 * sum(1 for sl in state.hand_slots or [] if sl["name"] == name)   # 手里的合成材料
        # **方便升级**（新主C 必须方便升级，要着重培养）：
        # 卡面自带涨等级的（韩信"等级永久+1"、镜"每使用2张效果牌，等级+1"、苏烈"等级永久+2"），
        # 持续成长（永久 / 每…）比一次性的更值
        # 成长分来自 data/card_features.json（tools/card_features.py 从卡面文本逐条正则得出，0~5）
        s += 2.4 * (self._features(name).get("grow_score") or 0)
        hr = self.hero_reviewed.get(name)
        if hr is not None:
            # 技能评分（主C 强度表，逐个读技能审定）：**每分 20，是选主C 的主导项**。
            # 原来每分 6，被"输出位 +10 / 成长分 / 阵容频率"等加项盖过 —— 2026-09-25 第四轮第 3 局，
            # 6 分的米莱狄（靠召唤仆从）当了主C，8 分的"阿离第一主C"百里守约在场上却没被选（米莱狄不适合当主C，
            # 主C 推荐表也没喂给 Laya）。已经练起来的主C 仍受"新人等级不能更低"保护，不会被 1 级新人顶掉。
            s += 20 * (hr.get("carry_score", 5) - 5)
        # **1~2 阶当主C 再降一档**（2026-09-25：1-2 阶英雄技能太弱，作为主C 的评分再降低）：
        # 1 阶 −30、2 阶 −15（相当于强度表上 1.5 / 0.75 分）。第四轮第 5 局 1 阶孙尚香（6 分）差点把主C 从 4 阶孙悟空手里抢回去。
        q0 = info.get("quality", 0) or 0
        from agent.decision import routes as _R
        if q0 == 1:
            s -= 30
        elif _R.CARRY_TIERS:
            # **2~3 阶主C 定调**（2026-09-29，推翻 09-25 的"2 阶再降 15"）：2~3 阶前期抽得到、能尽早堆等级；
            # 4~5 阶要遗愿/众星拱月这类特殊牌、还要清场换人才接得住等级 —— 手里没有搬等级手段时不换过去
            if q0 in _R.CARRY_TIERS:
                s += 10
            elif q0 >= 4 and name != self._carry_piece and self._carry_piece                     and not self._level_move_ready(state, name, self._carry_piece):
                s -= 25
        elif q0 == 2:
            s -= 15
        # **后期抽不到同名来合成**（后期抽不到同名英雄用于合成升级）：按当前棋手等级这个品阶在商店的出现概率降分
        # （meta.TIER_ODDS：5 级 1 阶 1%、2 阶 20%；6 级 1~2 阶直接 0）。概率 <5% −25、<15% −10；4 级起才算（前期大家都是低阶）。
        lv0 = state.level or 0
        if lv0 >= 4 and q0 and not (_R.CARRY_TIERS and name == self._carry_piece):
            # 2~3 阶定调时，**现任主C** 不因"后期抽不到同名"降分（它的等级靠枫叶舞 / 古币 / 牺牲件堆，不靠合成）
            p_tier = meta.TIER_ODDS.get(min(lv0, max(meta.TIER_ODDS)), {}).get(q0, 0.0)
            if p_tier < 0.05:
                s -= 25
            elif p_tier < 0.15:
                s -= 10
        if self._obsolete(name, state.round_no):
            s -= 40                                   # 第 6 回合后的 1 阶
        if ("steal_level" in (self._features(name).get("tags") or []) and name in self._field
                and self._steal_fodder(state, name)):
            s += 15                                   # 带夺取、场上有 ≥10 级的不重要英雄可抢：每次合成 +10 级，方便养大
        if self._transitional_carry(name, state.round_no):
            s -= 60                                   # 过渡主C（曜这类）过了前期：觉醒/高等级也不留作主C
        return s

    # **换主C 的搬等级手段**（换主C 要有快速提高等级的方法）：
    # 有这些手段时，"新主C 等级不能比老主C 低"那条门槛可以放宽 —— 等级能搬过去。
    STEAL_CARDS = ("夺取之术", "众星拱月")        # 遗志不算：等级给随机一人（很难确定等级会分配给谁）
    STEAL_NEAREST_RE = re.compile(r"夺取最近1名己方英雄")

    def _level_move_ready(self, state: GameState, new: str, old: str) -> bool:
        hand = {sl["name"] for sl in state.hand_slots or []} | {n for n, k in self._hand_book.items() if k > 0}
        if any(c in hand for c in self.STEAL_CARDS):
            return True                                  # 夺取之术（最近 1 人最多 10 级）/ 众星拱月（每人 10 级）
        if "红叶染锋" in self._lord_boosts and self._maple_bound == old:
            return True                                  # 卖掉枫叶舞绑着的老主C，50% 等级留给下一个枫叶舞英雄
        if "steal_level" in (self._features(new).get("tags") or []) and old in self._field:
            return True                                  # 新主C 自带「夺取」：挨着老主C 就能夺等级
        return False

    def _hex_dist(self, a, b) -> int:
        """棋盘格之间的步数（按 _cell_neighbors 的蜂窝相邻关系 BFS）。游戏里"距离最近的友军"按格子算，不按屏幕像素。"""
        a, b = tuple(a), tuple(b)
        if a == b:
            return 0
        seen, frontier, d = {a}, [a], 0
        while frontier and d < 12:
            d += 1
            nxt = []
            for c in frontier:
                for n in self._cell_neighbors(c):
                    n = tuple(n)
                    if n == b:
                        return d
                    if n not in seen:
                        seen.add(n)
                        nxt.append(n)
            frontier = nxt
        return 99

    BUDDY_FAC_RE = re.compile(r"距离最近的【([^】]+)】英雄")

    def _buddy_faction(self, name: str) -> str:
        """连接对象限定派系的（李元芳"对距离最近的【河洛】英雄…"）返回派系名；明世隐 / 太乙这类连任何人的返回 ""。"""
        m = self.BUDDY_FAC_RE.search(self._card_text(name) or "")
        if not m:
            return ""
        from agent.decision import routes as _R
        return _R.norm_banned(m.group(1))

    # 名单 / 阈值绑定棋手（data/lords/阿离/lord_rules.json taiyi_*；下面是兜底值）
    TAIYI_EARLY_ANCHOR = _LordRule("taiyi_early_anchor", ("太乙真人", "盾山"))
    TAIYI_MSY_MAX_ENTRIES = _LordRule("taiyi_msy_max_entries", 1)       # 明世隐登场 ≤ 这么多次才贴盾山（10-21 第 900 局）
    TAIYI_SAC_FACTIONS = _LordRule("taiyi_sac_factions", ("河洛",))     # 牺牲 / 古币线的路线主C 阵营
    TAIYI_EARLY_UNTIL = _LordRule("taiyi_early_until", 12)       # 第 12 回合起算后期：太乙回到贴主C（10-21：太乙先贴盾山，后期贴主C）

    def buddy_anchor(self, name: str) -> str:
        """"选最近友军"的英雄该贴谁：默认主C。**太乙早到先贴盾山**（临时规则 taiyi_shield_early：
        前期碰到太乙可以先挂到盾山上，让盾山多复活一次，多触发一次牺牲拿一个古币）：
        路线主C 是河洛（古币 / 牺牲线）、盾山在场、路线主C 还没觉醒 → 太乙贴盾山。"""
        t, anchor = self.TAIYI_EARLY_ANCHOR
        if name == t and _IR_on("taiyi_shield_early") and self.route_locked() and (self._round_seen or 0) < self.TAIYI_EARLY_UNTIL:
            # 2026-10-21 第 881 局：太乙前期挂到铠或盾山旁边、后期再挂到主C 旁边（太乙可以多次触发牺牲英雄的牺牲效果）：
            # 牺牲线（路线主C 卡面吃"牺牲效果的英雄阵亡"，或河洛古币线）第 12 回合前贴铠 / 盾山（铠优先：复生再死再给临时等级），不再要求主C 没觉醒
            rc = self.route_carry() or ""
            sac_line = ("牺牲效果的英雄阵亡" in (self._card_text(rc) or "")) or self._faction(rc) in self.TAIYI_SAC_FACTIONS
            # 2026-10-21 第 900 局：要按明世隐的觉醒等级算；他才登场一次时，多给的临时等级转化不了，不如放到主C 旁边：
            # 铠多死一次 = 多 +5 临时等级，但明世隐没觉醒时每回合只转 +2（觉醒 +4），多出来的临时等级战后清空 → 只有明世隐已觉醒才值得贴铠 / 盾山
            # 2026-10-21 晚（更正）：明世隐只登场过一次（上场后没有合成过）贴盾山；上场过 2 次以上一律贴主C
            # （盾山牺牲给古币 = 永久等级，不靠明世隐转化）
            if sac_line and self.msy_entries() <= self.TAIYI_MSY_MAX_ENTRIES and anchor in self._field:
                return anchor
        return self._carry_piece

    def _buddy_others(self, name: str, vis: dict) -> list:
        """算"谁离它更近"时要比的人：派系限定的只比同派系（账本里查得到派系的），否则比所有人（含画面上看到、账本没记的）。"""
        fac = self._buddy_faction(name)
        if fac:
            return [tuple(q) for n2, q in self._field.items() if n2 not in (name, self.buddy_anchor(name))
                    and fac in (self.pieces.get(n2, {}).get("synergies") or [])]
        out = [tuple(q) for n2, q in self._field.items() if n2 not in (name, self.buddy_anchor(name))]
        out += [tuple(int(v) for v in kk.split(",")) for kk, t in (vis or {}).items()
                if t == "hero" and tuple(int(v) for v in kk.split(",")) not in {tuple(q) for q in self._field.values()}]
        return out

    def _buddy_cell_score(self, cell, carry_pos, others) -> tuple:
        """贴主C 的辅助（太乙「距离最近的 1 名英雄获得复生」/ 瑶「选最近的友方英雄附身」）站在 cell 的好坏，越小越好：
        (和主C 的步数离 1 差多少, 距离不比主C 远的其他队友个数)。(0, 0) = 主C 是唯一最近的人。
        2026-09-30 第 33 局：太乙和亚连、雅典娜都相邻，复生不一定落在亚连身上（原来按像素距离算，以为亚连最近）。"""
        dc = self._hex_dist(cell, carry_pos)
        # 2026-10-19 第 784 局 R12（太乙的位置经常摆错，需要手动调整）：(555,569) 离明世隐 1 步、离主C 沈梦溪 2 步，
        # 复生一定给明世隐，却因为"≤主C 距离的人只有 1 个"排在了贴着沈梦溪（并列 3 人）的格子前面 → 先数**比主C 更近**的人（有就最差）
        closer = sum(1 for o in others if self._hex_dist(cell, o) < dc)
        if closer:
            return (99 + closer, dc)
        tie = sum(1 for o in others if self._hex_dist(cell, o) <= dc)
        # 2026-10-11（明世隐连接最近的英雄，不一定要贴到旁边，要按格子距离算）：**主C 唯一最近**优先，其次才是离主C 几步。
        # 原来是 (离 1 步差多少, 并列数)，"挨着但并列"胜过"隔一格但主C 唯一最近"，正好反了。(0, 0) 仍 = 唯一最近且挨着
        return (tie, abs(dc - 1))

    def _best_buddy_cell(self, state: GameState, carry_pos, others, current=None):
        from ..perception.extractors import OWN_BOARD_ROWS, POP_OCCLUDING_CELLS
        occupied = {tuple(v) for v in self._field.values()} | set(POP_OCCLUDING_CELLS)
        occupied |= self.object_cells_all() | self._summon_cells(state)    # 图腾 / 物件占的格（审查 2026-10-11）
        for key in (state.unit_anchors or {}):
            try:
                cx, cy = key.split(",")
                occupied.add((int(cx), int(cy)))
            except ValueError:
                continue
        cands = [tuple(c) for row in OWN_BOARD_ROWS for c in row
                 if tuple(c) not in occupied and not self._cell_dead(c) and not self._covered_by_shop(state, c)]
        if not cands:
            return None
        return min(cands, key=lambda c: (self._buddy_cell_score(c, carry_pos, others), self._hex_dist(c, carry_pos)))

    def _nearest_free_cell(self, state: GameState, pos) -> Optional[tuple[int, int]]:
        """真实棋盘格里离 pos 最近、确实空着的一格（记账没人、画面上也没看到徽章、没被判废）。"""
        from ..perception.extractors import OWN_BOARD_ROWS, POP_OCCLUDING_CELLS
        occupied = {tuple(v) for v in self._field.values()} | set(POP_OCCLUDING_CELLS)   # 挡人口数字的四格不放人
        for key in (state.unit_anchors or {}):
            try:
                cx, cy = key.split(",")
                occupied.add((int(cx), int(cy)))
            except ValueError:
                continue
        cands = [tuple(c) for row in OWN_BOARD_ROWS for c in row
                 if tuple(c) not in occupied and not self._cell_dead(c) and not self._covered_by_shop(state, c)]
        if not cands:
            return None
        return min(cands, key=lambda c: (c[0] - pos[0]) ** 2 + ((c[1] - pos[1]) * 1.6) ** 2)

    def choose_carry(self, state: GameState) -> None:
        """每回合从**场上**挑主C；明显更强（+CARRY_SWITCH_MARGIN）才换，每回合最多换一次。

        换了之后：装备、枫叶舞、加等级牌都跟着新主C 走（它们都看 `_carry_piece`）；
        旧主C 不再受"主C 不卖"的保护，腾位时会被换下，装备退回手牌再穿给新主C。
        """
        if state.phase != "prepare" or not self._field:
            return
        cur0 = self._carry_piece
        _hard0 = self.model_decides() and str(self._carry_switch_by).lower().startswith("laya")   # 模型选的主C 只按硬约束复查
        if cur0 and not self._carry_eligible(cur0, hard_only=_hard0):
            if cur0 in self._field and self._flex_eligible(cur0) and not any(
                    n != cur0 and self._carry_eligible(n) for n in list(self._field)
                    if self.pieces.get(n, {}).get("category") == "英雄牌"):
                # 2026-10-11（第 565 局）：接替者只在手里时不清空 —— 原来手里有安琪拉就清掉孙膑（31 级），安琪拉第 8 回合才上场，空了两回合
                # **临时主C 过了前期、但没人能接替 → 先续用**（2026-10-10 第 518 局：禁河洛走大河虞姬，虞姬整局没刷到，
                # 1 阶孙尚香第 6 回合起"不合格"被清掉，之后整局没有主C —— 枫叶舞 / 装备 / 加等级牌都没人接）。
                # 原则是不能用 1 阶主C 玩到最后：有合格的人就换，没有时空着比续用更糟。
                if self._skip_logged.get(("keep_temp_carry", cur0)) != self._round_seen:
                    self._skip_logged[("keep_temp_carry", cur0)] = self._round_seen
                    logger.info("临时主C %s 过了前期，但场上 / 手里没有合格的接替者 —— 先续用", cur0)
            else:
                # 现任主C 已经不合格（存档读回的 / 规则变了，如公孙离被排除）→ 清掉重选
                # （2026-09-30 第 21 局：重启后读回"主C 公孙离"，整局没换，枫叶舞一直挂在狂铁身上）
                logger.info("现任主C %s 不能当主C 了 —— 清掉重选", cur0)
                self._carry_piece, cur0 = "", ""
        if self.carry_by_laya and cur0 and cur0 in self._field:
            # **谁当主C 交给 Laya**（逐步把规则引擎去掉）：现任还在场上时规则不换人，
            # 只在现任没了（卖掉/阵亡回手之外的情况）或本来没有时，由下面的规则补一个，免得装备/枫叶舞无主。
            return
        if self.carry_by_laya and cur0 and cur0 not in self._field and cur0 not in self._sold_names:
            # **主C 只是账本里"走丢"了，不是我们卖的**（第四轮第 11 局：账本把安琪拉记错格，拖前核对发现那格是少司缘，
            # 撤掉了她的记录 → 规则当场把主C 换成 2 阶 1 级的亚连）。人多半还在场上，"点开看场"会把她找回来；
            # 换主C 交给 Laya 的 carry 题，这里不补人。
            if self._skip_logged.get(("carry_lost", cur0)) != self._round_seen:
                self._skip_logged[("carry_lost", cur0)] = self._round_seen
                logger.info("主C %s 不在账本里（不是我们卖的）—— 多半是记错格，保留主C 等看场找回", cur0)
            return
        if cur0 and cur0 not in self._field and not self._transitional_carry(cur0, state.round_no) and (
                any(sl["name"] == cur0 for sl in state.hand_slots or [])
                or self._hand_book[cur0] > 0 or cur0 in self._awakened):
            # 手牌读漏（叠放）时也看**手牌账**和"觉醒回手"记录（第十四局：觉醒曜正在重新上场，
            # 那一帧手牌没读到它，主C 就被换成了 2 阶虞姬）
            # **主C 只是暂时不在场上（在手里等上场 / 觉醒回手），不换**（2026-09-23：
            # 主C 转换没有依据 —— 第九局韩信在手里，主C 被换成了场上 1 级的辅助明世隐，
            # 枫叶舞也跟着绑了过去）。
            return
        if self.carry_by_laya and self.model_decides() and not cur0:
            # 审查 10-12（B1，第 587 局"换主C：无(不在场) -> 安琪拉(124 分)"）：主C 空缺时规则不再按 _carry_score 补位，
            # 交给局面网络的 carry 题（laya_policy._shadow_carry 在主C 变化时当帧出题；只有一个合格的人时直接用他）
            if self._skip_logged.get(("carry_vacant", self._round_seen)) is None:
                self._skip_logged[("carry_vacant", self._round_seen)] = 1
                logger.info("主C 空缺 —— 交给局面网络选")
            return
        scored = {n: self._carry_score(n, state) for n in self._field}
        scored = {n: v for n, v in scored.items() if v is not None}
        if not scored:
            return
        best = max(scored, key=scored.get)
        cur = self._carry_piece
        if best == cur:
            return
        cur_s = scored.get(cur) if cur else None
        # 这一帧读不到徽章就用最近一次读到的（原来读不到就放行"等级不能更低"这条 ——
        # 第十二局 13 级的马可波罗被刚上场的孙悟空顶掉、当场卖掉换绑枫叶舞）
        lv = lambda n: (((state.unit_levels or {}).get(cell_key(self._field[n])) if n in self._field else None)  # noqa: E731
                        or self._unit_lv.get(n))
        q = lambda n: self.pieces.get(n, {}).get("quality", 0)  # noqa: E731
        trans = bool(cur) and self._transitional_carry(cur, state.round_no)
        if trans:
            # 过渡主C 过了前期：只要场上有**真正的输出位**就换，不看等级/觉醒/品阶
            # （评分 ≥6、不是过渡件、不是过时的 1 阶；每回合最多换一次）
            if (self.hero_reviewed.get(best, {}).get("carry_score", 5) < 6
                    or self._transitional_carry(best, state.round_no)
                    or self._obsolete(best, state.round_no)
                    or (q(best) < 2 and best not in self._ever_awakened)
                    or self._carry_switch_round == state.round_no):
                return
            logger.info("过渡主C %s 过了前期，换成真正的输出位 %s（设计约定：曜这类前期一过就换）", cur, best)
        elif cur_s is not None:
            if scored[best] < cur_s + self.CARRY_SWITCH_MARGIN:
                return
            # **觉醒过的主C 只让位给同样觉醒过的**（2026-09-24 数据局 match8：曜觉醒回手、重新上场，
            # 那一帧等级还没读到 → 等级比较被跳过，按品阶换成了 3 阶西施）
            if cur in self._ever_awakened and best not in self._ever_awakened:
                return
            if self._carry_switch_round == state.round_no:
                return
            # **新主C 必须真的比旧的强**：
            # 等级不能更低，且品阶更高或觉醒过 —— 光靠"推荐阵容里常当主C"这种先验分，
            # 不许一个刚上场的新人换掉已经练起来的主C。
            # "阿离能把等级搬过去"**只有手里真有搬等级的手段才成立**（`_level_move_ready`）。
            # 原来旧主C 是 1 阶就直接不看等级、新人等级读不到也放行 —— 2026-09-24 数据局：40 级的孙膑
            # 被刚拍来、等级没读到的甄姬顶掉（那局没选红叶染锋、手里也没有夺取牌）：甄姬等级低、
            # 也没合成过，场上明显力不从心。现在：新人等级读不到 → 先不换；比旧主C 低 → 没有搬等级手段就不换。
            # 旧主C 是 1 阶过渡件时放宽到"新人至少有它一半的等级"（12 级干将 → 6 级狄仁杰可以，40 级孙膑 → 3 级甄姬不行）。
            lb, lc = lv(best), lv(cur)
            if not self._level_move_ready(state, best, cur):
                if lb is None and lc is not None:
                    return
                # 旧主C 是 1 阶过渡件、或只是**凑合型**（评分 ≤6，例：西施）而新人是真正的主C（评分 ≥8）时，
                # 新人有旧主C 一半的等级就换（2026-09-24 match14：西施等级太低、伤害也很低，不适合当主C；
                # 场上已有嬴政/李信（8 分）却因"等级不能更低"一直换不过去）。
                hr = self.hero_reviewed
                weak_cur = hr.get(cur, {}).get("carry_score", 5) <= 6 and hr.get(best, {}).get("carry_score", 5) >= 8
                need = lc * 0.5 if (lc is not None and (q(cur) < 2 or weak_cur)) else lc
                if lb is not None and need is not None and lb < need:
                    return
            if not (q(best) > q(cur) or best in self._ever_awakened):
                return
        else:
            # 旧主C 真的没了（不在场也不在手）或者**本来就没有**（开局 / 进程重启后记忆清空）：
            # 新主C 至少要像样 —— 过了前期（第 6 回合起）要 2 阶以上或觉醒过，
            # 没有就宁可暂时没有主C，装备/枫叶舞先留着（不乱给 1 阶/1 级的过渡件）。
            # 第十局重启后主C 为空，直接选了 -16 分的 1 阶孙膑当主C。
            if (self._obsolete(best, state.round_no)
                    or (state.round_no >= 6 and q(best) < 2 and best not in self._ever_awakened)):
                logger.info("主C %s 不在了，场上没有像样的输出位（最好的是 %s），暂时不定主C", cur or "（无）", best)
                self._carry_piece = ""
                return
        logger.info("换主C：%s(%s) -> %s(%.0f 分，%d 阶%s)", cur or "无",
                    f"{cur_s:.0f} 分" if cur_s is not None else "不在场",
                    best, scored[best], self.pieces.get(best, {}).get("quality", 0),
                    f"，{state.unit_levels.get(cell_key(self._field[best]))} 级"
                    if state.unit_levels and cell_key(self._field[best]) in state.unit_levels else "")
        self.note("换主C", round=state.round_no, 旧=cur or "", 新=best,
                  分数=f"{cur_s if cur_s is not None else '-'}->{scored[best]:.0f}")
        self.record_decision("carry", state, [{"name": n, "score": round(v, 1)} for n, v in scored.items()],
                             best, old=cur or "")
        if cur and cur in self._field:
            self._prev_carry = cur        # 老主C：等新主C 站稳后卖掉，枫叶舞/装备转给新主C
        self._carry_piece = best
        self._carry = self._field[best]
        self._carry_switch_round = state.round_no
        self._carry_switch_by = "rule"       # 审查 10-12（A3b）：原来不改，留着上一次"Laya keep"，规则补位被当成模型选的

    carry_by_laya = False       # LayaPolicy 带 carry 题时置 True
    deploy_first = None         # (回合, 英雄)：swap 为它腾了位，本回合上场先上它
    lock_by_laya = False        # 学生题面 ≥ f9d 时，锁店由商店题的 lock 选项决定
    auction_target = None       # (回合, 拍品名)：Laya 拍卖题选的目标；拍品名 "" = 不拍
    auction_by_laya = False     # Laya 接管拍卖题时为 True：规则不自己挑拍品
    swap_by_laya = False        # LayaPolicy 带 swap 题时置 True（decide_sell_unit 的"换阵容卖人"让出来）

    def duo_members(self) -> set:
        """**共战天赋点名的英雄**（2026-10-12：崛起「获得1张马可波罗和1张百里守约。共战时…」、双生突袭「马可波罗、百里守约共战…」——
        有这种效果时，两人中一个必须是主C，且两人必须同时上场、不能卖）。只读已选天赋的卡面，不写死名字。"""
        out = set()
        heroes = [n for n, v in self.pieces.items() if v.get("category") == "英雄牌"]
        for t in self._talents_taken:
            txt = (self.pieces.get(t, {}).get("skill") or "") + self._card_text(t)
            if "共战" not in txt:
                continue
            names = [h for h in heroes if h in txt]
            if len(names) >= 2:
                out |= set(names)
        return out

    PLACEHOLDER_YIELD_FIT = 6

    def placeholder_yield(self, state: GameState) -> str:
        """**1 阶占位主C 必须让位**（2026-10-12：1 阶占位主C 不让位，枫叶舞就滚不起来）：现任主C 是 1 阶
        （觉醒过、是路线成员也一样；手动指定的 1 阶路线主C 如干将路线除外），场上已有合格（_carry_eligible）的 2 阶以上、
        审定评分 ≥PLACEHOLDER_YIELD_FIT 的输出位 → 返回接班人：路线主C > 同体系路线主C > 评分 > 等级。没有返回 ""。"""
        cur = self._carry_piece
        if cur == self.HAIYUE_TEMP and self.haiyue_phase(state) in ("raise", "farm", "rush", "hunt"):
            return ""        # 伞海月：干将是计划中的过渡主C，养到海月来（第 630 局第 6 回合被这条让位给了沈梦溪）
        # 2026-10-14 起：共战牌要权衡换主C 的代价、能否为那个英雄凑出配套阵容 → 不再强制让位给共战之一，
        # 交给换主C 题（题面写共战两人的主C 评分、可凑的配套阵容、现任投入）
        duo = set()
        _cur_invested = bool(cur and cur in self._field and ((self._unit_lv.get(cur) or 0) >= self.KEEP_CARRY_LV
                                                              or cur in self._ever_awakened or self._maple_bound == cur))
        if duo and cur not in duo and not _cur_invested:
            # 现任已经养起来（≥10 级 / 觉醒 / 绑枫叶舞）时不为共战天赋换主C（r9za 老师：兄妹「铠 + 露娜」时主C 是 155 级沈梦溪）
            # 共战天赋：主C 必须是两人之一（2026-10-12）→ 场上合格的那个立即接任（评分高的先）
            cand = sorted((n for n in duo if n in self._field and self._carry_eligible(n)),
                          key=lambda n: (-(self.hero_reviewed.get(n, {}).get("carry_score", 5) or 0), -(self._unit_lv.get(n) or 0), n))
            if cand:
                return cand[0]
        if not cur or cur not in self._field or (self.pieces.get(cur, {}).get("quality") or 0) != 1:
            return ""
        if cur == self.route_carry():
            return ""
        fam = set()
        try:
            from agent.decision import routes as _R
            fam = {(_R.by_key(k) or {}).get("carry") for k, _c in self.family_routes(
                state.banned_faction or "", base_key=self.__dict__.get("_forced_orig") or self.forced_route)}
        except Exception:                                   # noqa: BLE001
            pass
        rc = self.route_carry()
        # 路线主C / 同体系路线主C 不受评分门槛限制（2026-10-13 第 614 局：路线主C 虞姬评分 5 < 6，第 4 回合上场后
        # 1 阶孙尚香没让位，局面网络答"维持"，枫叶舞留在孙尚香身上；虞姬买了也上场了，主C 却没切过去）
        cands = [n for n in self._field if n != cur and (self.pieces.get(n, {}).get("quality") or 0) >= 2
                 and self.pieces.get(n, {}).get("category") == "英雄牌" and self._carry_eligible(n)
                 and (n == rc or n in fam
                      or (self.hero_reviewed.get(n, {}).get("carry_score", 5) or 0) >= self.PLACEHOLDER_YIELD_FIT)]
        from agent.decision import lord_config as _LCq
        _cq = _LCq.load()
        if _cq.present:
            cands = [n for n in cands if _cq.enabled(n)]      # 接班人只从主C 名单开着的人里挑（第 851 局：安琪拉评分高被挑中、亚连没轮上）
        if not cands:
            return ""
        cands.sort(key=lambda n: (n != rc, n not in fam, -(self.hero_reviewed.get(n, {}).get("carry_score", 5) or 0),
                                  -(self._unit_lv.get(n) or 0), n))
        return cands[0]

    def apply_carry(self, name: str, state: GameState, why: str = "laya") -> bool:
        """外部（Laya carry 题）指定主C。只查**合法性**，不做价值判断：
        必须在场上、是能当主C 的输出位（`_carry_eligible`）、本回合还没换过。"""
        cur = self._carry_piece
        _hard = self.model_decides() and str(why).lower().startswith("laya")
        if not name or name == cur or name not in self._field or not self._carry_eligible(name, hard_only=_hard):
            return False
        if self._carry_switch_round == state.round_no:
            return False
        _uc = (self.__dict__.get("_unconfirmed_deploy") or {}).get(name)
        if _uc and time.time() - _uc < 30:
            logger.info("换主C 暂缓：%s 刚决定上场、拖拽回执还没确认（第 708 局：拖之前就换主C，结果没拖上去）", name)
            return False
        logger.info("换主C（%s）：%s -> %s", why, cur or "无", name)
        self.note("换主C", round=state.round_no, 旧=cur or "", 新=name, 来源=why)
        if cur and cur in self._field:
            self._prev_carry = cur
        self._carry_piece = name
        self._carry = self._field[name]
        self._carry_switch_round = state.round_no
        self._carry_switch_by = why
        return True

    def sync_awakened(self, state: GameState) -> None:
        """手牌里**看得见**觉醒卡 -> 立刻按"觉醒回手"处理，不再靠"合成失败两次"去猜。

        2026-09-23 实战：干将莫邪觉醒回手，我们没认出来，
        把它当成合成材料往旧格子 (839,502) 拖了 3 次，全失败后才猜"多半是觉醒回手"，
        可那时"试够 3 次就搁置本回合"已经把这张牌搁置了 —— **重新上场也被一起挡掉**，
        名额让给了公孙离和弈星，主C 带着 3 件装备的位置整回合空着。
        觉醒卡在手里是**直接证据**（一个单位不可能既在场上又在手牌里），
        和"观测不能删记录"那条铁律不冲突：那条说的是"没看见"不能当证据，这里是"看见了"。
        """
        for slot in state.hand_slots or []:
            if not slot.get("awakened"):
                continue
            name = slot["name"]
            if self.pieces.get(name, {}).get("category") != "英雄牌":
                continue
            first = name not in self._awakened
            self._awakened.add(name)
            self._ever_awakened.add(name)
            self._confirmed_awakened.add(name)       # 手牌上直接认出（颜色 + 点开看）
            if name in self._field:
                pos = self._field.pop(name)
                self._cell_owner.pop(pos, None)
                logger.warning("手牌里有觉醒的 %s —— 它已经不在 %s 了，撤销记录、优先重新上场",
                               name, pos)
                # **它离场了，人口就少一个 —— 马上记上**（2026-09-23 第八局：觉醒卡等了 9 秒才上场，不够优先）：
                # 认出觉醒的那一帧人口读不到，于是沿用"上次读到的 6/6"—— 那个 6 里还算着孙膑本人，
                # decide_deploy 判"满了"，这一帧去打了效果牌，又弹 N 选 1，觉醒卡等了 9 秒才上场。
                if self._last_pop_seen:
                    here, cap = self._last_pop_seen
                    self._last_pop_seen = (max(0, here - 1), cap)
                if not state.population_valid and state.population > 0:
                    state.population -= 1
                self.note("判定觉醒回手", 牌=name, 原格=f"{pos[0]},{pos[1]}", 依据="手牌里看到觉醒卡")
            elif first:
                logger.info("手牌里有觉醒的 %s，优先上场", name)
            # 搁置是给"往旧格子合成"记的，不该挡住重新上场
            self._parked_cards.discard((self._round_seen, name))

    VISION_MISS_RETURN = 3      # 视觉连续这么多帧没看到、牌在手里 → 回手（觉醒）
    VISION_MISS_GONE = 6        # 连续这么多帧没看到、手里也没有 → 已离场（被移除 / 转化），删记录

    # 成对装备（2026-09-30 第 30 局：两件都在手里、整局没用）：主C 穿「怒莲之刺」（攻速 +100%，受到的伤害全转给穿银莲的人），
    # 坦克穿「银莲庇佑」（+10 双防，替主C 扛伤害）—— 正是"顺带给 T 强度"的那类牌
    PAIR_EQUIPS = {"怒莲之刺": "carry", "银莲庇佑": "tank"}

    def _pair_equip(self, state: GameState, slot: dict) -> Optional[dict[str, Any]]:
        name = slot["name"]
        partner = next(n for n in self.PAIR_EQUIPS if n != name)
        hand = {sl["name"] for sl in state.hand_slots or []}
        partner_worn = any(partner in (v or ()) for v in self._worn.values())
        if partner not in hand and not partner_worn:
            return None                        # 只有一件：先留着（单穿没意义）
        role = self.PAIR_EQUIPS[name]
        if role == "carry":
            who = self._carry_piece if self._carry_piece in self._field else None
        else:
            lv = {n: ((state.unit_levels or {}).get(cell_key(tuple(p))) or 0) for n, p in self._field.items()}
            _tags = lambda n: self.pieces.get(n, {}).get("tags") or []   # noqa: E731
            true_tank = lambda n: ("tank" in (self.hero_role(n).get("combat_role") or [])) if self.hero_role(n)                 else not any(t in _tags(n) for t in ("刺客", "射手", "法师", "战士"))   # noqa: E731   卡面分析优先
            tanks = sorted((n for n in self._field if self._is_tank(n) and n != self._carry_piece),
                           key=lambda n: (true_tank(n), lv.get(n, 0)), reverse=True)   # 不带输出职业的前排优先（李白是「前排、刺客」）
            who = tanks[0] if tanks else None
        if not who:
            return None
        pos = self._field[who]
        if self._equips_at(state, pos) >= meta.MAX_EQUIPS or not self._equip_allowed(name, who) or self._cell_dead(pos):
            return None
        logger.info("成对装备：%s 给 %s（%s）", name, who, "主C" if role == "carry" else "坦克")
        self.note("用牌", round=state.round_no, 牌=name, 类别="装备牌", 目标=who, 依据="成对装备")
        return {"type": "drag_card", "piece": name, "src": (slot["x"], slot["y"]),
                "dst": self._on_unit(state, pos), "kind": "装备牌", "target": who}

    def trusted_unit_names(self, state: GameState) -> dict:
        """认人模型的结果里，**只收本局拿到过的英雄**（2026-09-30 第 31 局：没进类表的猪八戒被认成狄仁杰，
        账本凭空多了个狄仁杰）。拿到过 = 手牌里出现过 / 账本在场上。"""
        ob = self.__dict__.setdefault("_obtained", set())
        ob.update(sl["name"] for sl in state.hand_slots or [] if sl.get("name"))
        ob.update(self._field)
        ob.update(n for n in (self._maple_bound, self._carry_piece, self._prev_carry) if n)   # 账本提到过的人（枫叶舞绑定对象找回）
        return {k: v for k, v in (state.unit_names or {}).items() if v and v[0] in ob}

    def vision_identity_sync(self, state: GameState) -> None:
        """**用认人模型校正账本**（2026-09-30，agent/perception/unit_id.py；测试集门槛 0.5 采纳准确率 99.94%）。
        这之前"谁在哪"只靠账本 + 点开看，点歪一次账本就被改反（卖错人、枫叶舞绑定丢失、主C 被删）。
        只在准备阶段、商店关着时用；同一个名字同帧出现在两格（还没合成的同名单位）跳过。
        被挤掉的原主：这一帧在别处认出来 → 下面的循环会放过去；没认出 → 放到唯一一个"视觉有人、没认出是谁"的格子，
        放不了就先移出账本（交给看场 / 视觉离场判断）。"""
        names = self.trusted_unit_names(state)
        if not names or state.shop_open or state.phase != "prepare":
            return
        cnt = collections.Counter(v[0] for v in names.values())
        moved = []
        for k, v in names.items():
            name = v[0]
            if cnt[name] > 1 or self.pieces.get(name, {}).get("category") != "英雄牌":
                continue
            cell = tuple(int(x) for x in k.split(","))
            self._confirmed_at[name] = (self._round_seen, cell)
            # 认人模型认出"就是他"= 直接证据，和点开看（confirm_unit）一样清掉这一格的失败记录
            # （2026-10-01 第 64 局：亚连刚上场 1 秒就往他身上打加速成长，两次"没生效"→ 那格判死，
            # 之后模型每帧都认出亚连在那儿，加速成长 / 装备整回合送不出去，日志也不说为什么，直到点开看才解）。
            # 每格每回合最多清一次，防止"失败 → 清掉 → 再失败"来回打转。
            if self._cell_fails.get(cell) and self._vision_cleared.get(cell) != self._round_seen:
                self._vision_cleared[cell] = self._round_seen
                self._cell_fails.pop(cell, None)
                logger.info("认人模型确认 %s 就在 %s，清掉这一格的失败记录", name, cell)
            if tuple(self._field.get(name) or ()) == cell:
                continue
            prev = self._cell_owner.get(cell)
            self.relocate_unit(name, cell)
            moved.append(f"{name}→{k}")
            if prev and prev != name and prev not in self._field:
                seen_elsewhere = any(w[0] == prev for w in names.values())
                if not seen_elsewhere:
                    spare = [kk for kk, t in (state.vision_units or {}).items()
                             if t == "hero" and kk not in names
                             and tuple(int(x) for x in kk.split(",")) not in self._cell_owner]
                    if len(spare) == 1:
                        self.relocate_unit(prev, tuple(int(x) for x in spare[0].split(",")))
                        moved.append(f"{prev}→{spare[0]}(推断)")
        if moved:
            logger.info("认人模型校正账本：%s", moved)
            self.note("认人校正", round=state.round_no, 改=moved)

    def vision_absence(self, state: GameState) -> None:
        """**用视觉判断"人不在了"**（2026-09-24 match14：视觉检测准确率已经很高，算法上多用视觉检测模型；
        每次升级时维护一下列表，而不是逐个点开看）。
        以前"没看到不能删记录"是因为徽章检测只有 66% 准；D-FINE 格子召回 99.3%（ch2 §2.7b），
        连续 3 帧都漏的概率极低 —— 现在"连续多帧没看到"是够强的证据。
        只在视觉这一帧确实跑了（`vision_units` 非空，商店关着）时计数，商店开着/战斗中不算。"""
        if state.phase != "prepare" or state.shop_open or not state.vision_units:
            return
        seen = set(state.vision_units) | set(state.unit_anchors or {})
        hand = {sl["name"] for sl in state.hand_slots or []}
        # **人口读数是硬证据**（2026-09-26 第四轮第 6 局）：人口 5/5 = 账本 5 人时，"连续 3 帧没看到孙尚香"只是漏检 ——
        # 却被判成觉醒回手（手里那张是合成材料），删了记录、当成觉醒卡，此后两个回合"先关店等它上场"来回开关，一张牌没买。
        pop_ok = state.population_valid and state.population > 0
        pop_full = pop_ok and state.population >= len(self._field)
        # 自愈：之前判成"觉醒回手"的，人口读数比账本多、原格又看到了人 → 其实还在场上，恢复记录
        for name, pos in list(self._vision_returned.items()):
            if name in self._field or name not in self._awakened:
                self._vision_returned.pop(name, None)
                continue
            if pop_ok and state.population > len(self._field) and cell_key(tuple(pos)) in seen                     and tuple(pos) not in self._cell_owner:
                self._field[name] = tuple(pos)
                self._cell_owner[tuple(pos)] = name
                self._awakened.discard(name)
                if name not in self._confirmed_awakened:
                    self._ever_awakened.discard(name)
                self._vision_returned.pop(name, None)
                logger.warning("人口 %d > 账本 %d 且 %s 原格有人 —— 之前的\"觉醒回手\"是漏检，恢复记录",
                               state.population, len(self._field) - 1, name)
                self.note("撤销觉醒回手", 牌=name, 格=cell_key(tuple(pos)), 依据="人口读数比账本多 + 原格视觉有人")
        for name, pos in list(self._field.items()):
            k = cell_key(tuple(pos))
            if k in seen:
                self._vision_miss.pop(name, None)
                continue
            if pop_full:
                self._vision_miss.pop(name, None)     # 人口一个不少：没看到 = 漏检
                continue
            n = self._vision_miss.get(name, 0) + 1
            self._vision_miss[name] = n
            if n >= self.VISION_MISS_RETURN and name in hand:
                self._field.pop(name, None)
                self._cell_owner.pop(tuple(pos), None)
                self._awakened.add(name)
                self._ever_awakened.add(name)
                self._vision_returned[name] = tuple(pos)
                self._vision_miss.pop(name, None)
                logger.warning("视觉连续 %d 帧没在 %s 看到 %s，而它的牌在手里 —— 判定觉醒回手，优先重新上场", n, k, name)
                self.note("判定觉醒回手", 牌=name, 原格=k, 依据=f"视觉连续 {n} 帧没看到 + 牌在手里")
            elif (n >= self.VISION_MISS_GONE and name == self._maple_bound
                  and any(sl.get("name") == "枫叶舞" and sl.get("ready") is False for sl in state.hand_slots or [])):
                # 2026-10-21 第 889 局 R5：认人模型连续 6 帧没看到守约 → 删记录、清主C、改立狄仁杰（双主C，R11 出局）；
                # 但枫叶舞一直灰着绑在他身上、他也没被卖 —— 灰卡 = 宿主还活着在场，比"没看到"强得多：不删（设计原则：没看到是弱证据）
                if self._skip_logged.get(("maple_host_kept", name)) != self._round_seen:
                    self._skip_logged[("maple_host_kept", name)] = self._round_seen
                    logger.warning("视觉连续 %d 帧没在 %s 看到 %s，但枫叶舞灰着绑在他身上 → 他还在场，不删记录", n, k, name)
                    self.note("离场推断否决", 英雄=name, 原位置=k, 依据="枫叶舞灰卡绑定在他身上")
            elif n >= self.VISION_MISS_GONE:
                self._field.pop(name, None)
                self._vision_miss.pop(name, None)
                self._forget_gone(name, pos, f"视觉连续 {n} 帧没看到、手里也没有")
                logger.warning("视觉连续 %d 帧没在 %s 看到 %s、手里也没有 —— 判定已离场，删记录", n, k, name)
                self.note("推断离场", 英雄=name, 原位置=k, 依据=f"视觉连续 {n} 帧没看到")

    AWAKEN_SUSPECT_HOLD_S = 6.0

    def awaken_suspects(self, state: GameState) -> list:
        """**觉醒回手嫌疑**：本帧人口读数比账本少，而手里正好有场上某人的同名牌（2026-09-24 match13）。
        单看"同名在手"是合成材料的常态（不能据此删记录），但**加上人口少了一个**就是很强的信号 ——
        先点开确认，确认前别让别的英雄占掉它空出来的格子。"""
        if not (state.population_valid and state.phase == "prepare" and 0 < state.population < len(self._field)):
            return []
        hand = {sl["name"] for sl in state.hand_slots or []}
        return [n for n in self._field if n in hand and n not in self._awakened
                and (self._round_seen, n) not in self._awk_cleared]

    def decide_deploy(self, state: GameState) -> Optional[dict[str, Any]]:
        # 不占人口的单位（天赋送的始祖熊灵）：有空格就上，不看人口
        for sl in state.hand_slots or []:
            n = sl.get("name") or ""
            if n and (self.pieces.get(n) or {}).get("no_pop") and n not in self._field and not self._parked(n):
                from ..perception.extractors import OWN_BOARD_ROWS, POP_OCCLUDING_CELLS
                occ = ({tuple(v) for v in self._field.values()} | self.object_cells_all() | self._summon_cells(state)
                       | {tuple(int(x) for x in k.split(",")) for k in (state.unit_anchors or {}) if "," in k})
                cells = [tuple(c) for row in OWN_BOARD_ROWS for c in row
                         if tuple(c) not in occ and tuple(c) not in set(POP_OCCLUDING_CELLS) and not self._cell_dead(c)]
                if cells:
                    logger.info("%s 不占人口：直接上场 %s", n, cells[0])
                    return {"type": "drag_card", "piece": n, "src": (sl["x"], sl["y"]), "dst": cells[0]}
        exp = self.awaken_expected(state)        # 觉醒进度牌 / 合成凑满刚成功：自带 AWK_EXPECT_S 超时
        if exp:
            if self._skip_logged.get(("awk_expect", exp[0])) != self._round_seen:
                self._skip_logged[("awk_expect", exp[0])] = self._round_seen
                logger.info("%s 应当刚觉醒回手（还没在手牌里认出来）—— 暂停别的英雄上场，等他重新上场", exp)
            return None
        sus = self.awaken_suspects(state)
        if sus:
            import time as _t
            first = self._awk_suspect_since.setdefault((self._round_seen, sus[0]), _t.time())
            if _t.time() - first < self.AWAKEN_SUSPECT_HOLD_S:
                if self._skip_logged.get(("awk_suspect", sus[0])) != self._round_seen:
                    self._skip_logged[("awk_suspect", sus[0])] = self._round_seen
                    logger.info("人口 %d < 账本 %d 且手里有场上的 %s —— 疑似觉醒回手，先确认，暂停别的英雄上场",
                                state.population, len(self._field), sus)
                return None
        return self._decide_deploy(state)

    def _decide_deploy(self, state: GameState) -> Optional[dict[str, Any]]:
        """把手牌里的新英雄拖上己方场地（已在场上的同名英雄走升级，不走这里）。

        人口上限以界面读数为准；读不到时（商店挡着等）每回合最多上 1 个，防超上限。
        """
        if self._shop_blocks_board(state):
            return None      # 商店面板开着，先关店再上场
        if state.round_no != self._deployed_round:
            self._deployed_round = state.round_no
            self._deployed_count = 0
            self._deployed_positions = []
        # 人口上限只在**本帧真读到**时才作数（商店开着时那块区域被卡面盖住，
        # 缓存值可能是几回合前的，拿它挡上场会让场面一直填不满）
        if state.population_valid and state.population >= state.population_cap:
            return None
        if not state.population_valid and self._pop_full_guess == state.round_no:
            return None
        # 本回合拖新英雄时人口变红过（硬证据），除非之后真读到有空位（卖了人），否则不再上场
        if self._pop_over_round == state.round_no and not (
                state.population_valid and state.population < state.population_cap):
            return None      # 本回合已经在两个格子上场失败过，人口多半满了（等真读到人口再说）
        if not state.population_valid and (self.__dict__.get("_pop_anchor") or self._last_pop_seen) and self.pop_full(state):
            # **人口读不到时，用"上一次真读到的 + 之后账本的确定变化"判满**（pop_now，和商店同一条口径）。
            # 不判的后果（2026-09-23 实战回合 5~6）：场面其实已经 5/5 满了，
            # 而那几帧人口读不到 → 这条检查被跳过 → 干将莫邪 连着往
            # (300,569)/(917,570)/(555,569) 三个格子拖，**每一格都失败**，
            # 因为根本没有空位。看日志像"落点选得不对"，其实是"压根不该上场"。
            # 2026-10-19：原来直接用上次读数，卖人 / 觉醒回手之后还当满员
            return None
        conservative = not state.population_valid   # 读不到 → 保守，每回合只上 1 个
        if self._deployed_count >= (1 if conservative else 8):
            return None
        from ..perception.extractors import OWN_BOARD_DROPS_BACK, OWN_BOARD_DROPS_FRONT
        occupied = set(self._field.values())
        deployable: list = []
        for slot in state.hand_slots:
            name = slot["name"]
            if slot.get("awakened") and self.pieces.get(name, {}).get("category") == "英雄牌" \
                    and name not in self._field:
                # **觉醒卡永远排第一**。不看搁置：搁置是合成失败记的。
                deployable.append((10, slot))
                continue
            if self._obsolete(name, state.round_no) and not self._is_carry(name):
                continue  # 第 6 回合后 1 阶英雄不再上新的（2026-09-23），手里的由 decide_sell 卖掉
            if self._parked(name) and name not in self._awakened:
                # 本回合已经失败太多次，让位给别的牌（否则它会吃光整回合的帧）。
                # **判定觉醒回手的除外**：那次搁置是"往旧格子合成"失败攒出来的，
                # 重新上场是另一件事（2026-09-23：主C 因此三个回合没上场）。
                continue
            if name in self._field:
                continue  # 同名走升级
            info = self.pieces.get(name)
            if not info or info.get("category") != "英雄牌":
                continue
            # 觉醒回手的卡重新打出还能**再触发一次登场效果**，优先级高于普通新卡
            prio = 2 if self._is_carry(name) else (1 if self._is_core(name) else 0)
            # 觉醒回手的卡重新打出能**再触发一次登场效果**，优先级高于普通新卡
            if name in self._awakened:
                prio += 3
            # 阵容外的废牌：带登场/整备效果且人口有富余时才值得上，吃完效果就卖。
            # **只用手上已有的牌**（拍卖赢的、效果牌送的、买阵容牌时顺带来的）——
            # 绝不为了吃一次登场效果去花 3 能量买牌，那个性价比极低
            # 买不买是 decide_buy 的事，那里不看登场效果。
            # 棋手经验型英雄要**赶在升级之前**上场，否则这回合的升级白升了
            if self._is_level_engine(name):
                prio = max(prio, 1 if self._level_engine_ready(state) else 0)
            if self.engine_piece_rank(name, state.round_no) < 0 and any(
                    self.engine_piece_rank(s2["name"], state.round_no) > 0 and s2["name"] not in self._field
                    for s2 in state.hand_slots):
                # **后期件不抢最低配套的格子**（第 573 局：第 8 回合太乙真人先上、明世隐在手里等了两回合）
                if self._skip_logged.get(("late_fill", name)) != state.round_no:
                    self._skip_logged[("late_fill", name)] = state.round_no
                    logger.info("%s 是后期件，手里有最低配套的人在等，先不上", name)
                continue
            if prio == 0:
                # 非阵容英雄：**只要场面没填满就该上**——空着场面必输，
                # 上个杂牌至少能挡刀、能触发羁绊。带登场/整备效果的再优先一点。
                # （2026-09-22 苏烈一直不上场：原来非阵容英雄
                #  只有带登场效果才上，导致买了一堆却全烂在手里、手牌溢出。）
                if state.population_valid and state.population >= state.population_cap:
                    continue
                prio = -1 if self._entry_effect_kind(name) else -2
            deployable.append((prio, slot))
        if not deployable:
            return None
        # 主力/核心优先上场。同名的几张里，**判定觉醒的取最右边那张**：
        # 新进手牌的排在最右，觉醒回手的那张就在那儿（两次实战截图都是）。
        # **为谁腾的位就先上谁**（2026-09-25 第 2 局：Laya 卖孙尚香给百里守约腾位，同一回合规则先上了"吃登场效果"的甄姬，
        # 腾出的格子被抢走，百里守约烂在手里）。
        # **觉醒卡永远第一，腾位目标第二**（2026-09-26：觉醒英雄回手后先上了别的英雄，问题仍在且很严重：
        # 第 4 局 Laya 先为大司命/艾琳腾了位，随后百里守约/大司命觉醒回手，"腾位优先"排在了觉醒前面 → 先上了别人）。
        df = self.deploy_first if self.deploy_first and self.deploy_first[0] == state.round_no else None
        awk = lambda sl: bool(sl.get("awakened")) or sl["name"] in self._awakened  # noqa: E731
        deployable.sort(key=lambda x: (-(1 if awk(x[1]) else 0),
                                       -(1 if df and x[1]["name"] == df[1] else 0),
                                       -self.engine_piece_rank(x[1]["name"], state.round_no),
                                       -x[0], -(x[1]["x"] if x[1]["name"] in self._awakened else 0)))
        slot = deployable[0][1]
        name = slot["name"]
        # 按职业选落点：坦克前排，其他后排（后排列表已按角落优先排序）
        role = self.pieces.get(name, {}).get("role", "dps")
        tags = self.pieces.get(name, {}).get("tags", [])
        is_tank = role == "tank" or any("前排" in x or "坦克" in x for x in tags)
        from agent.decision import lord_config as _LCp
        _pos = _LCp.load().position(name) if name in (self._carry_piece, self.route_carry()) else ""
        if _pos:
            is_tank = _pos == "front"      # 棋手配置指定主C 站位（韩信站后排闪现切后排，调研 10-20 晚）
        lanes = ([OWN_BOARD_DROPS_FRONT, OWN_BOARD_DROPS_BACK] if is_tank
                 else [OWN_BOARD_DROPS_BACK, OWN_BOARD_DROPS_FRONT])
        # 跳过已占用的、以及实测拖不进去的格子（棋盘有透视，边角格可能无效）
        # 三重避让：我记得放了人的、实测拖不进去的、以及**疑似已被别人占着**的。
        # 最后一类是关键：`_field` 只是我自己的动作记忆，棋盘上还有我不知道的单位
        # （开局自动放的、对账时被误删记录的），往那儿放必然失败
        # （2026-09-22：一直往已有英雄的位置上放）。
        # 四重避让。第四条（_cell_owner）是位置规划的核心：
        # 棋盘我们**看不见**，`_field` 只是自己的动作记忆。一个格子只要分配给过别人，
        # 就不能再分给新英雄——哪怕原主已从 `_field` 里被移除（觉醒/对账修正）。
        # 否则新英雄会一次次撞到还站在那儿的人身上
        # （2026-09-22：后续所有英雄都想放到安琪拉的位置上）。
        observed = {tuple(c) for c in state.board_occupied} if state.board_seen else set()
        # **等级徽章是"这格有人"的直接证据**，比旧的棋盘观测稳（它是为装备件数做的，
        # 用的是标定过的 28 格坐标，实测 5/5 命中）。
        # 2026-09-22 又出现英雄往已经有英雄的位置上放 —— 那几次失败的
        # 落点画面上都站着人，只是我们的账本不知道。
        # 注意**只用来加占用、不删记录**（设计原则：让弱证据推翻强证据是设计错误）。
        for key in (state.unit_anchors or {}):
            try:
                cx, cy = key.split(",")
                observed.add((int(cx), int(cy)))
            except ValueError:
                continue

        def free(c):
            return (c not in occupied and c not in self._bad_cells
                    and tuple(c) not in self.object_cells_all()          # 古币机 / 重剑 这类物件占着的格（清空重规划也不清）
                    and not self._covered_by_shop(state, c)
                    and c not in self._busy_cells
                    and self._cell_fails.get(c, 0) < self.CELL_FAIL_LIMIT
                    and c not in observed          # 画面上看到有人，最硬的证据
                    and self._cell_owner.get(c) in (None, name))
        for c in observed:
            self._seen_occupied.add(tuple(c))
        # **优先选"这一局从没见过有人"的格子**。`free()` 已经排掉了当下看得见的占用，
        # 但观测会漏检（3D 单位有动画、徽章会被遮挡），漏检的格子和真空格长得一样。
        # 2026-09-23 实测：失败最多的落点永远是**占用观测次数最高的那几格**
        # （改前是 529,372/300,569/1027,572，交错之后变成 672,430/556,430）——
        # 说明我们总在往"曾经有人"的格子上放。
        # 这里只当**排序偏好**，不当硬排除 —— 符合"观测只能加占用、不能否定"那条铁律：
        # 曾经有人不代表现在有人（人会被卖/阵亡），所以虚格排完了还是要回来试它们。
        cands = [c for lane in lanes for c in lane if free(c)]
        freed, freed_round = self._freed_cell
        _anc = self.buddy_anchor(name)                  # 太乙早到贴盾山（taiyi_shield_early）；其余 = 主C
        carry_at = self._field.get(_anc) if _anc else None
        bad = self._drop_fail_rate                     # 历史失败率（tools/drop_cell_stats.py）
        if self._is_support(name) and not self._wants_carry_side(name) and carry_at:
            # **只有卡面写着"选最近的友军"的英雄才贴主C**（2026-10-10：只有个别英雄必须站在主C 旁边，按卡牌文本判断；
            # 不需要的让位，放到后排中间左右）。其他辅助（少司缘：治疗沿途友军）放后排中间，主C 的邻格留给瑶 / 太乙 / 明世隐。
            cands.sort(key=lambda c: (bad(c) > 0.8, tuple(c) not in self.BACK_MID_CELLS,
                                      self._hex_dist(c, carry_at) <= 1, tuple(c) in self._seen_occupied))
            if cands:
                logger.info("%s 是辅助但不需要贴主C，放后排中间 %s", name, cands[0])
        elif self._wants_carry_side(name) and carry_at:
            # **辅助 / 作用于最近友军的（瑶、太乙真人）贴着主C 放**（2026-09-23 / 09-24）：按离主C 的距离排，其次才是"没见过有人"。
            # 历史上几乎拖不进去的格子（失败率 >0.8）排到最后 —— 第十局贴主C 连选两个坏格子
            cx, cy = carry_at
            if self._carry_buddy(name):
                # 太乙 / 瑶：按**格子步数**选"主C 是唯一最近的人"的格子（2026-09-30 第 33 局：太乙和亚连、雅典娜都相邻）
                others = [tuple(q) for n2, q in self._field.items() if n2 != _anc]
                # 画面上看到有人、账本没记的格子也算"别的队友"（2026-10-08 第 423 局：明世隐上场时放到了李元芳附近，
                # 再挪位置离李元芳还是很近，不会重新连接到主C —— 明世隐"连接 1 名最近的友方英雄"）
                for kk, t in (state.vision_units or {}).items():
                    try:
                        q = tuple(int(v) for v in kk.split(","))
                    except ValueError:
                        continue
                    if t == "hero" and q not in others and q != tuple(carry_at):
                        others.append(q)
                # **"主C 是唯一最近的人"排第一**：原来先按"这格以前见过有人"排，把挨着李元芳的格子排到了前面
                # **挨着主C 排第一**（2026-10-10 第 522 局：太乙真人这种必须在主C 旁边的英雄被放到了明世隐旁边，
                # 主C 旁边明明有空位）：主C 守约在 (1027,572)，邻格 (917,570) 的历史失败率 0.95（老数据），被 bad(c) 排到了最后。
                # 历史失败率只在"同样挨着主C"的格子之间排先后
                cands.sort(key=lambda c: (self._buddy_cell_score(c, carry_at, others),
                                          bad(c) > 0.8, tuple(c) in self._seen_occupied, self._hex_dist(c, carry_at)))
                if (cands and self._buddy_cell_score(cands[0], carry_at, others)[0] != 0
                        and self._link_sticky(name)):
                    # 没有"挨着主C 且主C 是唯一最近的人"的空格：**先放在主C 严格最近（不论几步）的格子连上主C**，
                    # 站位纠正再挪到主C 旁（并列也保持连主C）。挨着别人放下会连到别人，之后挪过去也不重连（游戏小 bug，2026-10-08）
                    stage = [c for c in cands if bad(c) <= 0.8 and self._buddy_cell_score(c, carry_at, others)[0] == 0]
                    if stage:
                        stage.sort(key=lambda c: self._hex_dist(c, carry_at))
                        cands = stage + [c for c in cands if c not in stage]
            else:
                _r = self.aura_radius(name)        # 光环（蔡文姬"周围 1 格"）：按格子步数，进 N 格以内（2026-10-11，原来按像素距离）
                cands.sort(key=lambda c: (self._hex_dist(c, carry_at) > _r, bad(c) > 0.8, tuple(c) in self._seen_occupied,
                                          self._hex_dist(c, carry_at), (c[0] - cx) ** 2 + (c[1] - cy) ** 2))
            if cands:
                logger.info("%s 是辅助，放到主C %s 旁边 %s", name, self._carry_piece, cands[0])
        elif self._is_sacrifice(name) and "钟馗" in self._field:
            # **牺牲件站到钟馗左右**（2026-09-29：守约路线 4/5 局零件都在，回合间自动涨的等级仍≈0 ——
            # 钟馗只给"自身及左右 1 格"挂牺牲，牺牲件站别处时阵亡不产古币 / 临时等级）
            sides = set(self._row_sides(self._field["钟馗"]))
            cands.sort(key=lambda c: (tuple(c) not in sides, bad(c) > 0.8, tuple(c) in self._seen_occupied))
            if cands and tuple(cands[0]) in sides:
                logger.info("%s 带牺牲，放到钟馗旁边 %s", name, cands[0])
        elif name == "钟馗":
            # 钟馗：自身 + 左右 1 格挂牺牲 → 前排正中，让两边的前排都吃到
            mid = {(529, 372), (617, 372), (556, 430), (672, 430)}
            cands.sort(key=lambda c: (tuple(c) not in mid, bad(c) > 0.8, tuple(c) in self._seen_occupied))
        elif name == self._carry_piece and any(
                n2 != name and self._carry_buddy(n2) for n2 in self._field):
            # **主C（重新）上场要落在辅助旁边**（2026-09-26：主C 重新上场换了位置，辅助效果失效）：
            # 觉醒回手 / 卖了重上时，辅助（瑶 / 太乙 / 明世隐…）还站在原处，按离最近辅助的距离排
            bud = [tuple(q) for n2, q in self._field.items()
                   if n2 != name and self._carry_buddy(n2)]
            dd = lambda a, b: (a[0] - b[0]) ** 2 + ((a[1] - b[1]) * 1.6) ** 2      # noqa: E731
            cands.sort(key=lambda c: (bad(c) > 0.8, min(dd(tuple(c), b) for b in bud)))
            if cands:
                logger.info("主C %s 上场，放到辅助旁边 %s", name, cands[0])
        elif "steal_level" in (self._features(name).get("tags") or []) and carry_at:
            # **带「夺取」的英雄离主C 越远越好**（公孙离/猪八戒/孙悟空：上场时从相邻友军身上
            # 夺走最多 10 级，放在主C 旁边等于抢主C 的等级）
            cx, cy = carry_at
            cands.sort(key=lambda c: (bad(c) > 0.8, tuple(c) in self._seen_occupied,
                                      -((c[0] - cx) ** 2 + (c[1] - cy) ** 2)))
            if cands:
                logger.info("%s 带「夺取」，放得离主C %s 远一点 %s", name, self._carry_piece, cands[0])
        else:
            # **先分前排/后排，再看别的**（2026-09-24：坦克在前面，其他尽量后排缩角落）。
            # 原来的排序键里最后一项是"历史失败率"，它对**所有候选一起排**，把 lanes 的前后排顺序整个盖掉了：
            # 后排角落历史上撞车多、失败率偏高，于是后排法师干将莫邪被放到前排正中 (529,372)。
            # 历史失败率只在同一排里当先验（>0.8 的排到本排最后），同排内保持"角落优先 / 中路优先"的原顺序。
            # 唯一排在前后排之前的是"本回合亲手卖人腾出的那一格"：卖人腾位说明场上满了，别的"空格"多半有人。
            lane_rank = {}
            for li, lane in enumerate(lanes):
                for ci, c in enumerate(lane):
                    lane_rank.setdefault(tuple(c), (li, ci))
            # 腾出的格子**只有在它该站的那一排**才优先（2026-09-24 match14：嬴政被放到了前排 ——
            # 卖掉前排的老夫子腾出 (529,372)，后排法师嬴政就被塞了进去）。
            # 例外：这一帧**没有视觉检测结果**时，别的"空格"多半其实有人（老的占用观测不可靠）→ 仍优先用亲手腾出的确定空格。
            freed_ok = (freed_round == self._round_seen
                        and (lane_rank.get(tuple(freed), (9, 0))[0] == 0 or not state.vision_units))
            # **后排不要都挤一个角**（2026-10-10：阵容不要过度堆在一个角落，否则后面来了辅助英雄要频繁挪动；
            # 主C 放左下角，其他远程放右下角）：主C（含路线主C）按离左下角 (300,569) 排，其他后排按离右下角 (1027,572) 排。
            # 主C 左边留给要贴主C 的辅助（瑶 / 太乙 / 明世隐…），后来的辅助直接贴上去，不用再挪人。
            _rc = (self._active_comp or {}).get("carry") if self.route_locked() else ""
            _anchor = None
            if not is_tank:
                _anchor = self.carry_home(state) if name in (self._carry_piece, _rc) else self.RANGED_HOME
            _ad = (lambda c: (c[0] - _anchor[0]) ** 2 + ((c[1] - _anchor[1]) * 1.6) ** 2) if _anchor else (lambda c: 0)  # noqa: E731
            cands.sort(key=lambda c: (not (freed_ok and tuple(c) == freed),
                                      lane_rank.get(tuple(c), (9, 0))[0],
                                      bad(c) > 0.8, tuple(c) in self._seen_occupied,
                                      _ad(c), lane_rank.get(tuple(c), (9, 0))[1]))
        dst = cands[0] if cands else None
        if dst is None:
            # 所有候选都被排除了：棋子会被卖/阵亡腾出位置，清掉推断出来的占用重试一轮，
            # 别让场面卡死。`_cell_owner` 也一起清——重新摸一遍棋盘。
            if self._busy_cells or self._cell_owner:
                logger.info("候选落点都被排除（占用 %d / 归属 %d），清空重新规划",
                            len(self._busy_cells), len(self._cell_owner))
                self._busy_cells.clear()
                self._cell_owner.clear()
                # **`_cell_fails` 不清**：它是"拖不进去"这个直接证据攒出来的，
                # 清掉就等于回到"反复往同一个坑里拖"。
                dst = next((c for lane in lanes for c in lane
                            if c not in occupied and c not in self._bad_cells
                            and self._cell_fails.get(c, 0) < self.CELL_FAIL_LIMIT), None)
        if dst is None and self._cell_fails:
            # **泄压阀**：棋子会被卖/阵亡腾出位置，所以"废格"不能是永久判决。
            # 两个坑，都是 2026-09-23 实测出来的：
            #   ① **不能整份清空**：满编时 `free()` 经常找不到候选，这条泄压阀
            #      就会反复触发、把记忆一次次清零 —— (1027,572) 因此被反复选中、
            #      失败 **4** 次，而阈值明明是 2。改成**衰减**：每格减 1。
            #   ② **不能嵌在"有推断占用才清空"那个分支里**：`_busy_cells` 为空时
            #      根本走不到，等于泄压阀失灵。所以提到外层。
            logger.warning("连兜底落点都没有了，把格子失败记忆衰减一档（%d 格）",
                           len(self._cell_fails))
            self._cell_fails = {c: n - 1 for c, n in self._cell_fails.items()
                                if n > 1}
            dst = next((c for lane in lanes for c in lane
                        if c not in occupied and c not in self._bad_cells
                        and self._cell_fails.get(c, 0) < self.CELL_FAIL_LIMIT), None)
        if dst is None:
            return None
        logger.debug("上场 %s（%s）-> %s", name, "前排" if is_tank else "后排", dst)
        self._deployed_count += 1
        self._deployed_positions.append(dst)
        self._evict_cell(dst, name)
        self._field[name] = dst
        self._cell_owner[dst] = name      # 这个格子归它，别人不许再往这儿放
        self._awakened.discard(name)
        # 第 708 局（2026-10-17）：决定上场时就写进了 _field，换主C 题在拖之前出题，选了"刚上场、lv?"的西施；
        # 随后抓牌校验发现那张是 3 阶效果锦囊、没拖 —— 主C 换成了不在场的人，李元芳被卖去换绑。拖拽回执确认前不能当换主C 对象
        self.__dict__.setdefault("_unconfirmed_deploy", {})[name] = time.time()
        self.note("上场", round=state.round_no, 英雄=name, 位置=f"{dst[0]},{dst[1]}",
                  定位="前排" if is_tank else "后排")
        # 只有"人口快满了、需要腾位置"时才把非阵容英雄登记为耗材。
        # 场面还空着就留着他打——卖掉只会让场面更空。
        slot_pressure = (state.population_valid
                         and state.population_cap - state.population <= 1)
        if deployable[0][0] == -1 and slot_pressure and self.pieces.get(name, {}).get("quality", 0) < 3                 and self._prep_left >= 20:      # 时间不够就不登记"吃完效果再卖"（上场 + 卖掉约 5 秒，第 628 局 R12 白耗两次）
            # 3 阶以上不登记（卖出前 _is_disposable 也会挡，这里不登记免得日志误导："孙悟空上场吃整备效果、稍后卖掉"）
            self._entry_fodder[name] = state.round_no
            logger.info("%s 上场吃%s效果，人口吃紧稍后卖掉腾位",
                        name, self._entry_effect_kind(name))
        # **消耗型单位不能当投资目标**（2026-09-22：枫叶舞不应该给弈星上）。
        # 弈星这类棋手经验件、以及只为吃一次登场效果才上场的杂牌，本来就打算卖掉，
        # 把棋手技能牌和装备砸在它们身上等于白扔。
        # 之前的条件是"首个上场的就当主力"，而弈星恰好第一个上场，就被选中了。
        # **主C 只由 choose_carry 定**（第十四局：这里原来"上场的是阵容主C 就直接设成主C"，绕过了全部换主C 规则 ——
        # 亚连上场没成、被判不在场，主C 在同一帧里 亚连→米莱狄→虞姬 连跳）。只有完全没有主C 时才先占个位。
        # 占位也要是**合格输出位**（2026-10-01 第 50 局：坦克海月被占位成主C，题面一路写 current carry: 海月，换主C 日志「海月 -> 百里守约」）
        # 锁了路线后只让**路线主C / 同体系主C** 占位（第 51 局：路线是伞亚连，海月（只在遗志流当主C，评分 7）仍被占位成主C 到 R5）
        _route_ok = (not self.route_locked() or 0 < (self._round_seen or 0) <= self.EARLY_FLEX_ROUND
                     or name in ({self.route_carry()} | set(self._sibling_carries())))
        if (not self._carry_piece and not self._is_disposable(name) and self._carry_eligible(name) and _route_ok
                and self._carry_score(name, state) is not None and (_IR_on("placeholder_carry") or not self.model_decides())):
            self._carry_piece = name
        if name == self._carry_piece:
            self._carry = dst         # 主力换格子了，投放点跟着走
        act = {"type": "drag_card", "piece": name, "src": (slot["x"], slot["y"]), "dst": dst}
        if slot.get("awakened") or name in self._awakened:
            act["awk"] = True        # 觉醒卡：执行层按颜色核对抓取点（手牌重排时按旧坐标会抓错牌）
        return act

    # ---------- 用牌（装备/效果） ----------

    # 保留类关键词：这类牌在手牌里留着下回合触发更划算（整备=回合开始时触发）
    HOLD_KEYWORDS = ("整备",)

    # 给英雄加成的天赋牌（"指定1名英雄等级+5"）—— 用法同效果牌
    # 2026-09-24：碰碰运气原来一直不用，有钱时应给主C 使用。它写的是"提升5-20级"，没有"+"，原来认不出。
    BUFF_TALENT_RE = re.compile(r"指定\d*名?(己方)?英雄(.{0,6}(等级|攻击|生命|护盾|属性|法术|物理)[^，。]{0,6}\+|.{0,6}提升\d+(-\d+)?(等级|级))")
    COST_CARD_RE = re.compile(r"消耗\d*(?:-\d+)?能量")
    COST_CARD_MIN_GOLD = 5        # 卡面写"消耗…能量"的牌（碰碰运气随机消耗 1~5）：能量够付最坏情况才用
    # 天赋送的"指定1名英雄"牌（终极复印机/二象坍缩/援军/前浪的荣光/全军列阵/盗火）：原来一律不用。
    # 判断（写进标注规范 g3，下一轮教给 Laya）：都给**主C**（复印=主C 的合成材料；同阵营/高阶英雄=跟主C 体系；
    # 盗火=加等级）。主C 不在场就先留着，不给别人。另外两个时机条件：
    #   塞英雄牌进手牌的（二象坍缩/援军）手牌 ≤ HAND_ROOM_FOR_GIFTS 才用；召唤上场的（全军列阵）场上空位 ≥2 才用。
    TARGET_TALENT_RE = re.compile(r"^(?:转瞬\s*)?指定1名英雄[，,]")
    HAND_ROOM_FOR_GIFTS = 6
    MAPLE_FAMILY_PREFIX = "翩跹舞步"     # 阿离的枫叶舞强化牌一族，后缀常读不全

    def _use_category(self, name: str, info: dict) -> Optional[str]:
        """这张牌**实际按什么用**。决策层各处（用牌、"棋盘上还有没有活"）都要走这里，
        否则一处认、一处不认 —— 用牌那边想打，待办那边却说"没活了"先去开商店。"""
        category = info.get("category")
        if category == "天赋牌" and (self.BUFF_TALENT_RE.search(info.get("skill", ""))
                                    or self.TARGET_TALENT_RE.search(info.get("skill", ""))):
            return "效果牌"
        if not info and name.startswith(self.MAPLE_FAMILY_PREFIX):
            return "效果牌"
        return category

    FREE_LEVEL_RE = re.compile(r"棋手(?:等级)?\s*(?:[+＋]|提升)\s*1(?!\d)|免费升级")

    def free_level_card(self, state: GameState) -> Optional[dict[str, Any]]:
        """手里"棋手 +1 级"的牌（扶摇直上：棋手 +1 级、满级给 10 能量）→ 自用拖拽动作；没有返回 None。只读卡面。"""
        if state.level and state.level >= meta.MAX_LORD_LEVEL:
            return None
        for sl in state.hand_slots or []:
            n = sl.get("name") or ""
            txt = (self.pieces.get(n, {}).get("skill") or "") + self._card_text(n)
            if n and (n == "扶摇直上" or self.FREE_LEVEL_RE.search(txt)) and self.pieces.get(n, {}).get("category") != "英雄牌":
                dst = self._any_hero_pos(state) if hasattr(self, "_any_hero_pos") else None
                if dst is None:
                    continue
                return {"type": "drag_card", "piece": n, "src": (sl["x"], sl["y"]), "dst": dst,
                        "kind": "效果牌", "self_cast": True, "why": "先用免费升级牌"}
        return None

    def decide_use_cards(self, state: GameState) -> Optional[dict[str, Any]]:
        """手牌里的装备/效果牌：拖到己方主力英雄（本回合首个上场位）身上。

        - 整备类留牌，本回合不出
        - 装备牌：拖到英雄身上（基础装备会弹四选一，由上层处理弹窗）
        - 效果牌（强化/能量/等级类自用词）：拖到己方英雄身上
        """
        self._note_hp(state)                 # 预言牌按上一场输赢押：每回合记一次血量
        if self._shop_blocks_board(state):
            return None      # 商店面板开着，先关店再用手牌
        # 场上真的有英雄才谈得上穿装备/用效果牌。
        # 判据用 _field（我方实际上场记录），不要只看 population——
        # 人口读不到时是 0，而 _deployed_positions 只记"本回合拖过"，
        # 拖失败也会留记录 → 2026-09-22 实战出现"场上没英雄却想穿装备"。
        # ---- 先处理"不需要目标的自用牌"：它们不依赖场上有没有英雄 ----
        # 典型是给棋手经验/等级的那几张（研读会+3、经典研读会+4、管理进修+3、
        # 扶摇直上 等级+1），全都 0 费、即时生效、收益随回合复利。
        # 2026-09-22 发现这条链路没做：手里攥着白送的经验却整局停在 1 级。
        # 它们**不要求场上有英雄**，所以必须放在 `if not self._field` 那道门前面，
        # 否则开局第一回合（场面还空着）正是最该用它们的时候，却被挡掉。
        selfcast = self._pick_self_cast(state)
        if selfcast:
            return selfcast
        if not self._field:
            # **记账空了不代表场上没人**。账本会被对账/觉醒判定/卖人失败清掉，
            # 而画面上可能站着满编 5 个人 —— 2026-09-23 实战：人口读数 5/5、
            # `_field` 却是空的，于是"场上没英雄"这条门把装备牌挡了整整两个回合
            # （结果手牌越堆越多）。
            # 只要**观测到**有人（等级徽章锚点），就照常往那儿贴。
            if not self._anchor_cells(state):
                return None
        for slot in state.hand_slots:
            name = slot["name"]
            info = self.pieces.get(name, {})
            category = info.get("category")
            if (self._prep_left < self.LATE_REMOVE_S and category in ("效果牌", "天赋牌")
                    and re.search(r"移除(?:1名|一名)|英雄变为|变为(?:同阵营)?(?:高\d阶|\d阶)?的?(?:随机)?英雄", info.get("skill") or "")
                    and not re.search(r"敌方", info.get("skill") or "")):
                # **准备阶段快结束时不用"移除 / 变身己方英雄"的牌**（第 728 局 R3：限制性进化在回合最后几秒用掉，换回的高阶牌来不及上场，
                # 2/5 开战）。移除 + 换回 + 再拖上场约 8~10 秒；时间不够 = 这回合必然少一个人 → 合法性（做不完），留到下回合开头
                if self._skip_logged.get(("late_remove", name)) != state.round_no:
                    self._skip_logged[("late_remove", name)] = state.round_no
                    logger.info("%s 会移除 / 变身己方英雄，准备阶段只剩约 %.0f 秒，换回的牌来不及上场 → 留到下回合", name, self._prep_left)
                continue
            if not info and name not in self._unknown_logged:
                # 卡牌库里没有 = 决策层完全看不见这张牌，只能干瞪眼。
                # 棋手专属卡（阿离的"翩跹舞步·新生"等）官网爬虫抓不到，就是这么漏的。
                # 记下来，之后补进 fetch_card_db.MANUAL_CARDS。
                self._unknown_logged.add(name)
                logger.warning("手牌里有卡牌库中不存在的牌: %r —— 决策层会忽略它，"
                               "请补录到 MANUAL_CARDS", name)
            if name in self.PAIR_EQUIPS and not self._parked(name):
                act = self._pair_equip(state, slot)
                if act:
                    return act
            rv = self.usage_reviewed.get(name)
            if rv and category != "棋手技能牌":
                # **审定用牌表先行**（逐张读卡面写的，2026-09-24 起）：
                # hold = 不自动用（遗志/传承/听天由命/预言/机关…）—— 遗志原来会掉进"拖给主C"的通用分支、把主C 移除；
                # when 条件不满足先留着；特殊目标（阵营/1-3 阶/坦克/辅助/指定英雄…）在这里选人。
                if rv.get("target") == "hold":
                    # **rule_use：审定 hold 的牌里，读卡面能写出合法、明确、低风险用法的几张**（2026-10-09 晚卡牌审查第 5 条）
                    ru = rv.get("rule_use")
                    if ru == "predict" and not self._parked(name):
                        act = self._predict_act(state, slot, name)
                        if act:
                            return act
                        continue
                    if ru in ("level_swap", "level_copy") and not self._parked(name):
                        sp = self._level_trade_target(state, name, ru)
                        if sp:
                            tname, tpos, gain = sp
                            tdst = self._on_unit(state, tpos)
                            if not (self._cell_dead(tdst) or self._covered_by_shop(state, tdst)):
                                self._last_card_use = (name, state.round_no)
                                self.note("用牌", round=state.round_no, 牌=name, 类别=category, 目标=tname,
                                          依据=f"rule_use:{ru}", 等级收益=gain)
                                logger.info("用「%s」给 %s（%s：最近的同派系英雄高 %d 级）", name, tname, ru, gain)
                                # rule_only：方向是算出来的（只在目标会升级时用），Laya 用牌题换人可能把等级换低 → 不交给它改目标
                                return {"type": "drag_card", "piece": name, "src": (slot["x"], slot["y"]),
                                        "dst": tdst, "kind": "效果牌", "target": tname, "rule_only": True}
                        continue
                    if ru:
                        continue          # entry_boost 在 decide_entry_boost 里（要先于上场）；keep = 不用
                    if self._skip_logged.get(("hold", name)) != self._round_seen:
                        self._skip_logged[("hold", name)] = self._round_seen
                        logger.info("%s 不自动用（审定：%s）", name, rv.get("reason", "")[:40])
                    continue
                ok_w, why_w = self._usage_when_ok(rv, state, name)
                if not ok_w:
                    self.__dict__.setdefault("_timing_held", {})[(self._round_seen, name)] = why_w   # 交给 Laya 影子题 use_timing
                    if self._skip_logged.get(("when", name)) != self._round_seen:
                        self._skip_logged[("when", name)] = self._round_seen
                        logger.info("%s 先留着（%s）", name, why_w)
                    continue
                if self._parked(name):
                    continue
                sp = self._reviewed_target(state, name, rv)
                if (info.get("category") == "装备牌" and name in self.CARRY_LATE_DEF
                        and (name in self.CARRY_ANY_DEF or self.carry_late_def_ok(state))
                        and self._carry_piece in self._field and self._equip_allowed(name, self._carry_piece)
                        and self._equips_at(state, self._field[self._carry_piece]) < meta.MAX_EQUIPS):
                    # 2026-10-22 第 901 局：磐石之心审定目标是"前排坦克"，给了狂铁；主C 亚连是近战，按 carry_late_defense 这件该给主C
                    sp = (self._carry_piece, self._field[self._carry_piece])
                if sp == "skip" and "转瞬" in (self._card_text(name) or info.get("skill") or ""):
                    # 转瞬牌回合结束就删：审定目标不在也找个合规的人用掉（2026-10-08），全场没人合规才留着
                    _alt = [n for n in self._field
                            if (state.round_no, name, n) not in self._used_on and self.card_target_ok(name, n)[0]
                            and not self._cell_dead(self._on_unit(state, self._field[n]))]
                    if _alt:
                        _alt.sort(key=lambda n: -self.unit_value(n))
                        sp = (_alt[0], self._field[_alt[0]])
                if sp == "skip":
                    if self._skip_logged.get(("rtgt", name)) != self._round_seen:
                        self._skip_logged[("rtgt", name)] = self._round_seen
                        logger.info("%s 场上没有合适的对象（审定目标 %s），先留着", name, rv.get("target"))
                    continue
                if sp is not None:
                    tname, tpos = sp
                    kind = "装备牌" if category == "装备牌" else "效果牌"
                    if (state.round_no, name, tname) in self._used_on and name in self._no_repeat:
                        # 同一张牌这回合不能再给同一人（推演战术）→ **换个合规的人用掉**，别留着：
                        # 转瞬牌回合结束就删（2026-10-08 第 418 局：推演战术这种临时卡，主C 用不了也要给别人用掉，
                        # 否则就浪费，除非全场都没有对应种族的角色 —— 第二张推演战术审定目标还是西施，跳过后整回合没用，
                        # 场上明明还有逐鹿的白起）
                        alt = [n for n in self._field
                               if n != tname and (state.round_no, name, n) not in self._used_on
                               and self.card_target_ok(name, n)[0] and not self._cell_dead(self._on_unit(state, self._field[n]))]
                        if not alt:
                            continue
                        alt.sort(key=lambda n: -self.unit_value(n))
                        tname, tpos = alt[0], self._field[alt[0]]
                    tdst = self._on_unit(state, tpos)
                    if self._cell_dead(tdst) or self._covered_by_shop(state, tdst):
                        continue
                    self._last_card_use = (name, state.round_no)
                    self.note("用牌", round=state.round_no, 牌=name, 类别=category, 目标=tname,
                              依据=f"审定:{rv.get('target')}")
                    logger.info("用「%s」给 %s（审定：%s）", name, tname, rv.get("reason", "")[:40])
                    act = {"type": "drag_card", "piece": name, "src": (slot["x"], slot["y"]),
                           "dst": tdst, "kind": kind, "target": tname}
                    if self.COST_CARD_RE.search(self._card_text(name) or info.get("skill") or ""):
                        act["cost_card"] = True     # 卡面要付能量：没生效多半是钱不够，别怪格子
                    return act
            if category == "棋手技能牌":
                # 棋手技能牌（枫叶舞）：给一名英雄的永久 buff。用完**仍留在手牌**，
                # 所以"牌还在手里"不代表没生效（拖拽回执对它永远失效）。
                #
                # **绑定之后必须先把那个英雄卖掉才能重绑**（2026-09-22 更正；
                # 此前文档里写的"每回合都能换人重新指定"是错的）。
                # 于是：已经绑在场上某人身上时，再打出去就是**无效动作**，直接跳过。
                # **卡面自己会说能不能绑**：能绑时整张卡是亮的，绑住时是灰的
                # （2026-09-22 实测：枫叶舞重新亮起来，说明可以换人绑）。
                # 观测优先于记账 —— 被绑的英雄可能被系统卖掉/觉醒回手，账记不全。
                ready = slot.get("ready")
                if ready is not True:
                    self.__dict__["_maple_bright_since"] = 0.0
                if ready is False:
                    _prevb = self.__dict__.get("_maple_cleared", "")
                    if not self._maple_bound and _prevb and _prevb in self._field:
                        # 10-21 审查（第 851 局）：刚清掉的记录又看到卡面灰 = 那次"亮"是误读，人还在场 → 恢复记录
                        logger.info("枫叶舞卡面又是灰的，%s 还在场 → 恢复绑定记录（之前那次'亮'是误读）", _prevb)
                        self._maple_bound = _prevb
                        self.__dict__["_maple_cleared"] = ""
                    if (not self._maple_bound or self._maple_bound not in self._field)                             and self._skip_logged.get("maple_grey_unknown") != state.round_no:
                        # 卡面灰 = 绑在某人身上，账上却不知道是谁（2026-09-28 第 26 局：卖孙膑没生效，整局灰着）
                        self._skip_logged["maple_grey_unknown"] = state.round_no
                        logger.warning("枫叶舞卡面是灰的（已绑定），但账上记的绑定对象 %r 不在场 —— 账本和画面对不上",
                                       self._maple_bound)
                        self.note("枫叶舞绑定不明", round=state.round_no, 记录=self._maple_bound)
                    # 2026-10-22 第 928 局：宿主孙尚香 R5 起不在场、卡面一直灰，枫叶舞捏了 8 个回合没用 →
                    # 记录的宿主连续 2 个回合不在场，就每回合试 1 次打给主C（整局最多 3 回合；游戏不让绑会原样留在手里）
                    _gu = self.__dict__.setdefault("_maple_grey_absent", {})
                    if self._maple_bound and self._maple_bound not in self._field:
                        _gu.setdefault(self._maple_bound, state.round_no)
                    _r0 = _gu.get(self._maple_bound or "")
                    _tried_g = self.__dict__.setdefault("_maple_grey_tries", set())
                    if not (_r0 is not None and (state.round_no or 0) - _r0 >= 2 and len(_tried_g) < 3
                            and state.round_no not in _tried_g and self._carry_piece in self._field):
                        continue
                    _tried_g.add(state.round_no)
                    logger.warning("枫叶舞宿主 %s 已 %d 回合不在场、卡面仍灰 → 试着打给主C %s",
                                   self._maple_bound, (state.round_no or 0) - _r0, self._carry_piece)
                    self.note("枫叶舞灰卡重试", 记录=self._maple_bound, 主C=self._carry_piece)
                    self._maple_bound = ""
                if ready is True and self._maple_bound:
                    # **卡面亮着 = 没绑上**（或被绑的人已经离场）。"已绑"原来是在**做决定时**就记的 ——
                    # 2026-09-24 第二轮第 3 局：第 1 回合拖向牛魔的格子，而牛魔根本不在场上（上场被误拦），
                    # 账本却记成"绑在牛魔"，此后几回合都不再打，第 5 回合还为"解绑"把牛魔卖了（枫叶舞实际没用上）。
                    # 刚拖完卡面还没来得及变灰，给 3 秒。
                    if time.time() - self._maple_try_ts < 3:
                        continue
                    # 要**连续亮 2.5 秒**才算（第 851 局 R6：刚选完枫叶舞强化读到一帧"亮"就清了记录，之后卡面一直是灰的、整局不知道绑在谁）
                    _bs = self.__dict__.get("_maple_bright_since") or 0.0
                    if not _bs:
                        self.__dict__["_maple_bright_since"] = time.time()
                        continue
                    if time.time() - _bs < 2.5:
                        continue
                    logger.warning("枫叶舞卡面是亮的 —— %s 身上其实没绑上（或已离场），清掉记录重新绑", self._maple_bound)
                    self.note("枫叶舞未绑上", 记录=self._maple_bound)
                    self.__dict__["_maple_cleared"] = self._maple_bound
                    self._maple_bound = ""
                if ready is None and self._maple_bound and self._maple_bound in self._field:
                    continue           # 没有观测时退回记账判断
                if self._maple_tries.get(state.round_no, 0) >= 2:
                    continue           # 同回合最多试 2 次（手牌会挤动，src 会漂）
                self._banned_seen = state.banned_faction or self.__dict__.get("_banned_seen", "")   # 枫叶舞临时宿主按开局路线避开核心件要用
                target_name = self._skill_target()
                if target_name is None:
                    continue  # 场上没英雄，留着
                self._skill_round = state.round_no
                self._maple_tries[state.round_no] = self._maple_tries.get(state.round_no, 0) + 1
                self._maple_try_ts = time.time()
                self._maple_bound = target_name      # 暂记；卡面没变灰会在上面被清掉
                return {"type": "drag_card", "piece": name,
                        "src": (slot["x"], slot["y"]),
                        "dst": self._on_unit(state, self._field[target_name]),
                        "kind": category, "target": target_name}
            # **不看类别，只看卡面文本**：这类牌的效果是"移除指定英雄"，
            # 一旦归错类掉进下面"拖给主力"的通用分支，就是自己把主力删了。
            # 而归错类是真会发生的：爬虫的 type->类别 映射我就填错过一次
            # （URL 后缀是 _8.js 但主卡 type 是 6），体面离场 被归成过效果牌。
            if self.REMOVE_CARD_RE.search(name + info.get("skill", "")):
                # **"指定移除一名英雄换能量"类的天赋牌**（实测：体面离场，
                # 天赋「好聚好散」赠送）。它本质是"带补偿的卖人"，
                # 所以目标只挑**用完就卖的消耗件**，绝不指向主力/核心。
                # 场上没有消耗件就先留着 —— 这张牌不会过期。
                if self._parked(name):
                    continue
                victim = next((n for n in self._field if self._is_disposable(n)), None)
                if victim is None:
                    continue
                self._last_card_use = (name, state.round_no)
                self.note("用牌", round=state.round_no, 牌=name, 类别=category,
                          目标=victim)
                logger.info("用「%s」移除消耗件 %s 换能量", name, victim)
                return {"type": "drag_card", "piece": name,
                        "src": (slot["x"], slot["y"]),
                        "dst": self._on_unit(state, self._field[victim]),
                        "kind": category, "target": victim}
            if self.TRANSFORM_HERO_RE.search(info.get("skill") or self._card_text(name)):
                # **"指定英雄变为另一个英雄"的牌没审定过就先留着**（2026-09-25 第四轮第 1 局第 17 回合：
                # 昭君的「霜寒进化」（随机 N 选 1 拿到的）掉进下面"拖给主力"的通用分支，把已觉醒的主C 亚连
                # 变成了随机 3 阶英雄 —— 觉醒、枫叶舞绑定全丢）。这类牌用在谁身上是价值判断，将来交给 Laya（use_card）。
                if self._skip_logged.get(("transform", name)) != self._round_seen:
                    self._skip_logged[("transform", name)] = self._round_seen
                    logger.info("%s 会把目标英雄变成别的英雄，没审定用法 —— 先留着，不自动用", name)
                continue
            if self.DESTROY_EQUIP_RE.search(info.get("skill") or ""):
                # **「摧毁其所有装备」的牌（武装分解 / 武装重构）要给身上有装备的人**
                # （2026-09-24）。原来掉进下面的通用分支，
                # 拖给主C —— 主C 没装备时白打，有装备时把主C 的成品拆了，两头都错。
                # 目标：不是主C、也不是阵容核心（消耗件除外），身上至少 1 件；
                # 先挑消耗件（它的装备本来就要跟着被卖），再挑件数多的（3 件 = 珍品宝箱 / 重构多给 1 件）。
                if self._parked(name):
                    continue
                # 2026-09-25 第四轮第 3 局（武装分解仍用错）：规则两次想拆**武则天**（5 阶、主C 表 9 分的未来主C）的装备 ——
                # 原过滤只排除"当前主C / 阵容核心"，当时主C 被错定成米莱狄，武则天就成了"可拆"。
                # 现在：主C 表 ≥7 分的输出位、要留到最后的人（keeper）一律不拆；非消耗件至少 2 件才拆
                # （1 件只换 4 能量，拆掉一件成品不划算；卖人的话装备还会退回手牌）。
                hr = self.hero_reviewed
                cands = sorted(((self._is_disposable(n), self._equips_now(state, n, p), -self.unit_value(n), n, p)
                                for n, p in self._field.items()
                                if n != self._carry_piece and (self._is_disposable(n) or not self._is_core(n))
                                and hr.get(n, {}).get("carry_score", 5) < 7 and not self._is_keeper(n)
                                and self._equips_now(state, n, p) >= (1 if self._is_disposable(n) else 2)
                                and not self._cell_dead(p)
                                and (state.round_no, name, n) not in self._used_on),
                               reverse=True)
                if not cands and self.REBUILD_EQUIP_RE.search(info.get("skill") or ""):
                    cands = self._rebuild_targets(state, name)
                _w = self.__dict__.setdefault("_destroy_wait", {})
                if cands and not any(self._identity_ok(state, c[3], c[4]) for c in cands):
                    # 有可拆的人，但这一帧没有直接证据确认那格就是他（认人模型 / 点开）→ 先留着，不卖（2026-10-11）；
                    # 等了 2 个回合还确认不了（认人模型认不出的英雄）→ 当作没有合适目标，走下面"第 6 回合起卖掉"
                    _w.setdefault(name, state.round_no or 0)
                    if (state.round_no or 0) - _w[name] < 2:
                        if self._skip_logged.get(("destroy_id", name)) != self._round_seen:
                            self._skip_logged[("destroy_id", name)] = self._round_seen
                            logger.info("%s 想拆 %s 的装备，但还没确认那格就是他 → 先留着", name, cands[0][3])
                        continue
                else:
                    _w.pop(name, None)
                cands = [c for c in cands if self._identity_ok(state, c[3], c[4])]
                if not cands:
                    # **用不上就卖掉**（2026-09-25：武装分解不适合使用时卖掉）：
                    # 第 6 回合起还没有合适目标，就卖掉换能量，别一直占手牌。
                    if (state.round_no or 0) >= 6 and self.HAND_SELL_ENABLED and not self.model_decides():
                        # 模型接管用牌时不卖：留在手里，由 Laya 用牌题判断拆谁 / 先留着（2026-10-15 第 632 局：Laya N 选 1 刚拿的武装分解，规则转手就卖了）
                        logger.info("%s 没有合适目标（只拆消耗件/多余装备，不拆未来主C 与留场英雄）→ 卖掉", name)
                        self.note("卖手牌", round=state.round_no, 牌=name, 原因="摧毁装备的牌没有合适目标")
                        return {"type": "sell_card", "piece": name, "src": (slot["x"], slot["y"])}
                    if self._skip_logged.get(("destroy_equip", name)) != self._round_seen:
                        self._skip_logged[("destroy_equip", name)] = self._round_seen
                        logger.info("%s 要给身上有装备的非主C 英雄，场上没有合适的，先留着", name)
                    continue
                _d, n_eq, _v, victim, p = cands[0]
                dst = self._on_unit(state, p)
                if self._covered_by_shop(state, dst):
                    continue
                self._last_card_use = (name, state.round_no)
                self.note("用牌", round=state.round_no, 牌=name, 类别="效果牌", 目标=victim, 装备件数=n_eq)
                logger.info("用「%s」拆 %s 身上的 %d 件装备", name, victim, n_eq)
                return {"type": "drag_card", "piece": name, "src": (slot["x"], slot["y"]),
                        "dst": dst, "kind": "效果牌", "target": victim}
            # ---- 两类"用法和效果牌一样、却一直被跳过"的牌（2026-09-23：阿离的专属天赋牌
            # 一直没放到英雄身上）：
            #  ① **给英雄加成的天赋牌**（加速成长："指定1名英雄等级+5"）。这里原来只认
            #     装备牌/效果牌，天赋牌除了上面"移除换能量"那种之外一律跳过 —— 一局攥着两张 +5 级。
            #  ② **后缀没读出来的「翩跹舞步」**：OCR 常只读出前四个字，卡库查不到 -> 类别"未知"。
            #     这一族 5 张（余韵/进化/新生/流动/蓄势）都是"为枫叶舞挑选1张强化牌"的效果牌，
            #     **用法完全一样**，不必知道是哪一张就能打出去（卡名解析那边不猜是对的，
            #     猜错了名字会写错账；但"怎么用"不依赖名字）。
            category = self._use_category(name, info)
            _sk = info.get("skill") or ""
            if "消耗" in _sk and "能量" in _sk and state.gold < self.COST_CARD_MIN_GOLD:
                continue          # 要花能量的牌，钱不够最坏情况就先留着
            if category not in ("装备牌", "效果牌"):
                continue
            carry_only = (info.get("category") == "天赋牌" and self.TARGET_TALENT_RE.search(_sk) is not None)
            if carry_only:
                if self._carry_piece not in self._field:
                    continue          # 主C 不在场：这类牌只给主C，先留着
                if "获得" in _sk and "英雄牌" in _sk and len(state.hand_slots) > self.HAND_ROOM_FOR_GIFTS:
                    continue          # 会往手牌里塞好几张英雄牌，手牌太满会溢出
                if "召唤" in _sk and "上场" in _sk and not (
                        state.population_valid and state.population_cap - state.population >= 2):
                    continue          # 召唤补满场面：空位不到 2 个就是浪费
            if self._parked(name):
                continue  # 本回合失败太多次，先搁置
            text = name + info.get("skill", "")
            if any(k in text for k in self.HOLD_KEYWORDS):
                continue
            target = self._carry_piece
            dst = self._on_unit(state, self._carry_pos(state))
            # **主C 不在场上时不能写着主C 的名字往别人格子上打**（2026-09-23 第八局：
            # 蒙犽被误删出账本，`_carry_pos` 退回到西施的格子，目标却还写蒙犽 ->
            # 拖前核对"那儿是西施" -> 不拖 -> 下一帧同一个动作，整回合死循环）。
            # 和"卡面限定了使用对象"同一个处理：从场上挑一个真在那儿的、按实际收件人记名。
            if category != "装备牌" and (target not in self._field
                                         or not self._equip_allowed(name, target)):
                # 卡面限定了使用对象（某派系 / 远程 / 近战），主C 不符合 -> 挑一个符合的
                alt = sorted(((not self._is_support(n), self.unit_value(n), n, p)
                              for n, p in self._field.items()
                              if self._equip_allowed(name, n) and not self._cell_dead(p)),
                             reverse=True)
                if not alt:
                    if self._skip_logged.get(("target_limit", name)) != self._round_seen:
                        self._skip_logged[("target_limit", name)] = self._round_seen
                        logger.info("%s 卡面限定了使用对象（%s），场上没有符合的英雄，先留着",
                                    name, self.pieces.get(name, {}).get("skill", "")[:24])
                    continue
                _ns, _v, target, p = alt[0]
                dst = self._on_unit(state, p)
                logger.info("%s 限定了使用对象，改给符合条件的 %s", name, target)
            # **同一回合同一张牌不给同一个英雄用第二次**（卡面没写，是游戏里的规则；2026-09-24 发现）。
            # 转瞬牌（回合结束作废）改给另一个合适的人；其余留到下回合再给同一个人（加等级仍集中在主C）。
            # 只对**实测过**不能重复用的牌生效（`_no_repeat`，见 note_repeat_rejected）。原来对所有牌一刀切，
            # 推演战术（每次使用效果递增、天赋「复现」会把它返还）打在主C 身上一次后就再也不给主C，
            # 改投给一个账本过期的西施，失败两次被搁置（第二轮第 1 局：推演战术又没有使用）。
            if (category == "效果牌" and name in self._no_repeat
                    and (state.round_no, name, target) in self._used_on):
                if "转瞬" not in _sk or carry_only:
                    continue
                alt2 = sorted(((not self._is_support(n), self.unit_value(n), n, p)
                               for n, p in self._field.items()
                               if n != target and (state.round_no, name, n) not in self._used_on
                               and self._equip_allowed(name, n) and not self._cell_dead(p)),
                              reverse=True)
                if not alt2:
                    continue
                _ns, _v, target, p = alt2[0]
                dst = self._on_unit(state, p)
                logger.info("%s 本回合已经给 %s 用过，这张转瞬牌改给 %s", name, self._carry_piece, target)
            if (self.STEAL_NEAREST_RE.search(_sk) and target == self._carry_piece and target in self._field
                    and self._prev_carry and self._prev_carry in self._field and self._prev_carry != target
                    and (state.round_no, name) not in self._steal_moved):
                # **夺取之术"夺取最近 1 名己方英雄"**：换主C 之后要夺的是老主C 的等级 ——
                # 老主C 不是离新主C 最近的那个，就先把他挪到新主C 旁边的空格，再打这张牌（一次组合动作）。
                cpos = self._field[target]
                ppos = self._field[self._prev_carry]
                d = lambda a, b: (a[0] - b[0]) ** 2 + ((a[1] - b[1]) * 1.6) ** 2  # noqa: E731
                nearest = min((n for n in self._field if n != target), key=lambda n: d(self._field[n], cpos))
                if nearest != self._prev_carry:
                    adj = self._nearest_free_cell(state, cpos)
                    if adj and d(adj, cpos) < d(self._field[nearest], cpos):
                        self._steal_moved.add((state.round_no, name))
                        self._field[self._prev_carry] = adj
                        self._cell_owner.pop(ppos, None)
                        self._cell_owner[adj] = self._prev_carry
                        logger.info("夺取之术：先把老主C %s 从 %s 挪到新主C %s 旁边 %s，再夺他的等级",
                                    self._prev_carry, ppos, target, adj)
                        self.note("搬等级", 牌=name, 新主C=target, 老主C=self._prev_carry, 挪到=f"{adj[0]},{adj[1]}")
                        return {"type": "sequence", "piece": name, "steps": [
                            {"type": "move_unit", "piece": self._prev_carry, "src": ppos, "dst": adj},
                            {"type": "drag_card", "piece": name, "src": (slot["x"], slot["y"]), "dst": dst,
                             "kind": "效果牌", "target": target},
                        ]}
            if category == "装备牌":
                picked = self._equip_target(state, name)
                if picked is None:
                    continue          # 场上人人装备满了，这张先留着
                target, dst = picked
            if self._cell_dead(dst):
                # `_carry_pos` 在"找不到替代格"时仍会把那个废格返回来 ——
                # 2026-09-23 实战：「2阶效果锦囊」往 (617,372) 连拖 3 次。
                # 这里是最后一道闸：落点已经废了就别打这张牌，等场面变了再说。
                # **每回合只说一次**：牌留在手上、每帧都会重新评估，
                # 不去重的话这条日志会把整局日志刷爆（实测一局刷了几十条）。
                if self._skip_logged.get(name) != self._round_seen:
                    self._skip_logged[name] = self._round_seen
                    logger.info("%s 的落点 %s 本局已拖失败多次，先不打出去", name, dst)
                continue
            if self._covered_by_shop(state, dst):
                # 落点在商店面板底下，这一帧拖过去只会拖到面板上。
                # （`_shop_blocks_board` 的逃生口允许"带着商店硬干"，但只限面板下方的格子。）
                logger.info("%s 的落点 %s 在商店面板底下，这一帧先不打出去", name, dst)
                continue
            self._last_card_use = (name, state.round_no)
            self.note("用牌", round=state.round_no, 牌=name, 类别=category,
                      目标=target or "主力位")
            return {"type": "drag_card", "piece": name,
                    "src": (slot["x"], slot["y"]), "dst": dst,
                    "kind": category, "target": target}
        return None

    # 不需要指定目标、打出即生效的自用牌。判据用**卡面文本**而不是卡名白名单，
    # 这样新版本加的同类牌自动吃到（卡库里现有：研读会+3经验、经典研读会+4、
    # 管理进修+3且升级时再来一次、扶摇直上 棋手等级+1、扫拂 给场上最低两名加等级）。
    SELF_CAST_KEYS = ("棋手经验", "棋手等级提升", "等级提升1", "为枫叶舞挑选")
    # 「N阶效果锦囊」按卡名认（2026-09-22 确认的机制）：
    # 它是**每合成几次英雄系统白送**的牌，效果是"选择 1 张对应阶级的效果牌"。
    # 白来的、0 费、不需要目标，留在手里只会占手牌上限 —— 拿到就该打。
    # 打出后会弹「效果牌 N 选 1」界面，**那个界面占着商店的位置、不选就一直卡着**，
    # 由 decide_all 最前面的 card_pick 分支接手（两件事是一条链路，不能只做一半）。
    SELF_CAST_NAME_RE = re.compile(r"^(?:\d阶效果锦囊|装备箱)$")   # 打出 → 弹 N 选 1；装备箱：直接使用即可，效果就是挑选一个装备（2026-09-26）

    HAND_LIMIT = 10                    # 手牌上限（界面 "手牌 N/10"）
    _USAGE_CACHE: Optional[dict] = None

    @property
    def usage(self) -> dict:
        """逐张整理的**用牌表**（data/card_usage.json：能不能从手牌打出、给谁、什么时候、理由）。"""
        if RuleEngine._USAGE_CACHE is None:
            try:
                RuleEngine._USAGE_CACHE = json.loads(
                    (Path(__file__).resolve().parents[2] / "data" / "card_usage.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                RuleEngine._USAGE_CACHE = {}
        return RuleEngine._USAGE_CACHE

    def _usage_self_cast(self, name: str) -> Optional[dict]:
        """**我逐张审过的**"能从手牌打出、不需要目标"的牌（data/card_usage_reviewed.json，reviewed=true）。

        子 agent 整理的 data/card_usage.json 只当草稿：里面标成"不需要目标"的 120 张里有
        听天由命（消除 4 名最低等级英雄）、等价交换（扣光能量）、失败预言、胜之不武这类**绝不能拿到就打**的牌
        （2026-09-24 起改为逐张读卡牌数据库里的卡面效果、自己写使用逻辑）。
        """
        u = self.usage_reviewed.get(name)
        if (u and u.get("reviewed") and u.get("usable_from_hand") and u.get("target") == "none"
                and u.get("category") in ("效果牌", "天赋牌")):
            return u
        return None

    _REVIEWED_CACHE: Optional[dict] = None

    @property
    def usage_reviewed(self) -> dict:
        if RuleEngine._REVIEWED_CACHE is None:
            try:
                raw = json.loads(
                    (Path(__file__).resolve().parents[2] / "data" / "card_usage_reviewed.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raw = {}
            # **rule_target 只改规则层的视图**（2026-10-09 卡牌审查：炽热支配者 / 阿科米亚之子 / 小鹿的庇护 的目标审错了）。
            # target 字段题面（tools/laya/facts.py）也在读，直接改会让旧版本题面重渲染不一致 → 规则层用 rule_target 覆盖
            view = {}
            for k, v in raw.items():
                if isinstance(v, dict) and v.get("rule_target"):
                    v = dict(v, target=v["rule_target"], target_reviewed=v.get("target"))
                view[k] = v
            RuleEngine._REVIEWED_CACHE = view
        return RuleEngine._REVIEWED_CACHE

    def _usage_when_ok(self, u: dict, state: GameState, name: str = "") -> tuple[bool, str]:
        """审定用牌表的 when 条件（全部满足才用）。词汇见 tools/cards/reviewed_usage.py 文件头。"""
        hand = state.hand_slots or []
        for w in u.get("when") or []:
            if w == "now":
                continue
            if w == "laya_timing":
                # f9lw ㊵（2026-10-16：点兵点将不该拿到就用，要看场上核心阵容牌多少）：时机交给 Laya 的 use_timing 题；
                # 它答 now 才用（laya_policy._shadow_timing 写 _timing_go）。规则不判值不值
                if (state.round_no, name) not in (self.__dict__.get("_timing_go") or set()):
                    return False, "时机交给 Laya（use_timing）"
                continue
            if w.startswith("carry_nearest_lv>="):
                # 夺取之术（2026-10-21 第 876 局：只有主C 附近的英雄等级大于 10 级才有意义）：夺最近 1 名己方英雄最多 10 级
                need = int(w.split(">=")[1])
                c = self._carry_piece
                cp = self._field.get(c) if c else None
                if not cp:
                    return False, "主C 不在场"
                others = [(n, pos) for n, pos in self._field.items() if n != c]
                if not others:
                    return False, "主C 旁边没人"
                pc = self._prev_carry if self._prev_carry and self._prev_carry in self._field and self._prev_carry != c else ""
                if pc:
                    # 换主C 搬等级：规则会先把老主C 挪到新主C 旁边再夺（场景 96）→ 按老主C 的等级判（读不到时照用）
                    lvp = (state.unit_levels or {}).get(cell_key(tuple(self._field[pc])))
                    if not isinstance(lvp, int) or lvp >= need:
                        continue
                    return False, f"会把老主C {pc}（{lvp} 级 < {need}）挪过来夺，不划算"      # 10-21 审查 6：执行时夺的是老主C
                nn, npos = min(others, key=lambda x: (x[1][0] - cp[0]) ** 2 + (x[1][1] - cp[1]) ** 2)
                lvn = (state.unit_levels or {}).get(cell_key(tuple(npos)))
                if not isinstance(lvn, int) or lvn < need:
                    return False, f"主C 最近的 {nn} 等级 {lvn if lvn is not None else '?'} < {need}，夺取不划算"
                continue
            if w.startswith("gold>="):
                if state.gold < int(w[6:]):
                    return False, f"要 {w[6:]} 能量"
                continue
            if w.startswith("gold<"):
                if state.gold >= int(w[5:]):
                    return False, f"能量要少于 {w[5:]}"
                continue
            if w.startswith("field>="):
                if len(self._field) < int(w[7:]):
                    return False, f"场上要有 {w[7:]} 人"
                continue
            if w.startswith("field<="):
                if len(self._field) > int(w[7:]):
                    return False, f"场上至多 {w[7:]} 人（多了等级会随机给错人）"
                continue
            if w.startswith("round>="):
                if state.round_no < int(w[7:]):
                    return False, f"第 {w[7:]} 回合起才用"
                continue
            if w == "contract":
                ok, why = self.contract_ready(state, name)
                if not ok:
                    return False, why
                continue
            if w.startswith("hp<="):
                if not (0 < (state.hp or 0) <= int(w[4:])):
                    return False, f"血量 ≤{w[4:]} 才用"
                continue
            if w.startswith("hand_hero>="):
                n = sum(1 for sl in hand if self.pieces.get(sl.get("name"), {}).get("category") == "英雄牌")
                if n < int(w[11:]):
                    return False, "手里要有英雄牌"
                continue
            if w == "hand_room_field":
                if len(hand) > self.HAND_LIMIT - max(1, len(self._field)):
                    return False, f"手牌要空出 {len(self._field)} 张（场上人数）"
                continue
            if w == "once_per_round":
                # 每回合只用一张（胜之不武：下场战斗 +20 临时等级，3 张分 3 回合用；2026-10-12）
                if (self.__dict__.get("_card_uses") or {}).get((state.round_no, name), 0) >= 1:
                    return False, "本回合已经用过一张"
                continue
            if w == "after_carry_played":
                if self._last_hero_played != (state.round_no, self._carry_piece):
                    return False, "要在刚打出主C 的英雄牌之后用"
                continue
            k, _, v = w.partition(":")
            if k == "board_room":
                if not (state.population_valid and state.population_cap - state.population >= max(1, int(v or 1))):
                    return False, f"要 {v or 1} 个空人口"
            elif k == "hand_room":
                if len(hand) > self.HAND_LIMIT - max(1, int(v or 1)):
                    return False, f"手牌要空出 {v or 1} 张"
            elif k == "needs_energy":
                if state.gold < int(v):
                    return False, f"要 {v} 能量"
            elif k == "hold_next_round":
                return False, "留到下回合"
            elif k == "tag_on_field":
                tag, _, n = v.partition(":")
                # 自己有「整备：/登场：」效果条目才算（2026-10-13 卡面审计：朵莉亚 / 韩信 / 小乔这类反应器只是"提到"它）
                cnt = sum(1 for nm in self._field if re.search(rf"{tag}\s*[：:]", self.pieces.get(nm, {}).get("skill") or ""))
                if (cnt < int(n or 1) and tag == "登场" and cnt >= 1 and "明世隐" in self._field and self.route_locked()
                        and ((state.round_no or 0) >= 12 or not self.route_core_missing(state))):
                    # 2026-10-21 第 886 局：已经是最终阵容时，集体唤醒即使只有明世隐生效也可以用
                    # —— 阵容已经定型（核心不缺 / 第 12 回合起），没有更好的时机了，只有明世隐吃也照用
                    cnt = int(n or 1)
                if cnt < int(n or 1):
                    return False, f"场上要有 {n or 1} 个带「{tag}」的英雄"
            elif k == "core_level":
                if not self._core_has_level():
                    return False, "阿科米亚核心还没有等级（本局没出现过给核心加等级的来源）"
            elif k == "faction_on_field":
                fac, _, n = v.partition(":")
                cnt = sum(1 for nm in self._field if self._faction(nm) == fac)
                if cnt < int(n or 1):
                    return False, f"场上要有 {n or 1} 个【{fac}】英雄"
            elif k == "equips_on_field":
                tot = sum(self._equips_at(state, pos) for pos in self._field.values())
                if tot < int(v):
                    return False, f"场上装备要攒到 {v} 件"
        return True, ""

    CORE_TALENTS = ("场外供能", "动能灌注", "积少成多", "能量汲取")
    CORE_CARDS = ("助力核心",)

    def _core_has_level(self) -> bool:
        """阿科米亚核心有没有等级（2026-10-17：核心等级存在时，跳级直接给主C 用）。
        核心只增不减 → 本局出现过任何给核心加等级的来源就算有：场上有卡面写"阿科米亚核心…等级+"的英雄（狂铁 / 亚连 / 马可波罗 / 亚瑟 / 安琪拉 / 露娜…）、
        选过加核心的天赋、用过助力核心。账本推断（核心水珠的数字读数做出来后以视觉为准）。"""
        if self.__dict__.get("_core_seen"):
            return True
        for nm in self._field:
            if re.search(r"阿科米亚核心[^。]{0,8}等级(额外)?\+|核心的等级\+", self.pieces.get(nm, {}).get("skill") or ""):
                self._core_seen = True
                return True
        if any(t in self._talents_taken for t in self.CORE_TALENTS):
            self._core_seen = True
        return bool(self.__dict__.get("_core_seen"))

    def _faction(self, name: str) -> str:
        """英雄阵营（规则里的卡牌记录把阵营放在 synergies 里，第一个就是）。"""
        p = self.pieces.get(name, {})
        f = p.get("faction") or ((p.get("synergies") or [""])[0])
        return {"三分": "三分之地", "大河": "大河流域"}.get(f, f)

    # ================= 2026-10-09 晚：卡牌审查补的用法 =================
    # 原则：只读卡面文字推用法，不写死卡名；只做"合法、方向明确、猜错代价小"的，高风险（移除 / 变身）的仍 hold，
    # 但规则不用的牌不该一直占手牌 → _dead_hand_card 放满一回合就卖。

    ENTRY_BOOST_RE = re.compile(r"本回合打出的登场英雄牌.{0,8}额外触发")
    ENTRY_LINE_RE = re.compile(r"(?:^|\n)\s*登场\s*[:：]")
    NEAREST_FAC_RE = re.compile(r"指定\d*名【([^】]+)】英雄.{0,16}最近的【([^】]+)】英雄")
    PREDICT_RE = re.compile(r"预测下场战斗(胜利|失败)")
    VOID_UNIT_RE = re.compile(r"转移(?:到|给)([一-龥]{1,4}?)身上")
    RULE_MOD_RE = re.compile(r"([一-龥]{1,4})只能对非【([^】]+)】英雄主动使用(.*)")
    LEVEL_TRADE_MIN = 3          # 换帅 / 挂帅：至少多拿 3 级才用（等级读数有误差，差一两级不值得赌）

    def card_rule_mods(self) -> list[tuple[str, str, bool, str]]:
        """本局选过的天赋里**改写出牌规则**的条目：[(牌名关键字, 排除的派系, 是否变身, 天赋名)]。
        只读天赋卡面：「多彩古币：古币只能对非【河洛】英雄主动使用：随机变成其他同阶非【河洛】英雄，等级额外+7」
        → ("古币", "河洛", True)。（卡牌审查第 14 条：选了它以后古币的合法目标正好反过来，而且是变身效果、不能给主C）"""
        out = []
        for t in sorted(self._talents_taken):
            m = self.RULE_MOD_RE.search(self._card_text(t) or self.pieces.get(t, {}).get("skill") or "")
            if m:
                fac = {"三分": "三分之地", "大河": "大河流域"}.get(m.group(2), m.group(2))
                out.append((m.group(1), fac, "变成" in m.group(3) or "变为" in m.group(3), t))
        return out

    def card_rule_mod(self, card: str):
        return next((x for x in self.card_rule_mods() if x[0] and x[0] in (card or "")), None)

    def contract_estimate(self, state: GameState, name: str = "契约瓶") -> tuple[int, float, int]:
        """契约瓶（"获胜时累积 3 张随机英雄牌；失败时损失一半"）：按拿到之后每场战斗的输赢推算现在攒了几张。
        输赢 = 下一回合准备阶段血量有没有掉（与预言牌同一来源 _hp_by_round；缺读数的那场不算）。
        返回 (张数, 下一场赢的估计概率 p, 算进去的场数)。p = (近 4 场赢 + 1) / (场数 + 2)。"""
        since = self.__dict__.setdefault("_contract_since", {})
        r0 = since.setdefault(name, state.round_no or 0)
        c, res = 0, []
        for r in range(r0, (state.round_no or 0)):
            a, b = self._hp_by_round.get(r), self._hp_by_round.get(r + 1)
            if not a or not b:
                continue
            won = b >= a
            res.append(won)
            c = c + 3 if won else c // 2
        last = res[-4:]
        p = (sum(last) + 1) / (len(last) + 2)
        return c, p, len(res)

    def contract_ready(self, state: GameState, name: str = "契约瓶") -> tuple[bool, str]:
        """留着划算 ⇔ 下一场后的期望张数 p(c+3)+(1-p)c/2 > c ⇔ c < 6p/(1-p)。到了这个数就兑现；
        第 14 回合起（随时可能出局）或血量 ≤15 有张就兑现。2026-10-15：契约瓶一直不用（原来写死"第 9 回合起"）。"""
        c, p, n = self.contract_estimate(state, name)
        th = 6 * p / max(1e-6, 1 - p)
        if c > 0 and (c >= th or (state.round_no or 0) >= 14 or 0 < (state.hp or 0) <= 15):
            return True, ""
        return False, f"契约瓶约 {c} 张（{n} 场，胜率估计 {p:.0%}，攒到 {th:.1f} 张再用）"

    def _note_hp(self, state: GameState) -> None:
        if state.phase == "prepare" and (state.round_no or 0) > 0 and 0 < (state.hp or 0) <= 100:
            self._hp_by_round[state.round_no] = state.hp

    def _predict_act(self, state: GameState, slot: dict, name: str) -> Optional[dict[str, Any]]:
        """**胜利预言 / 失败预言**（天赋「预言家」送的一对，卡面"预测下场战斗胜利/失败，预测成功获得 2 张 3 阶兑换券 / 4 能量"）。
        不用 = 收益 0，用错 = 白用一张 → 按"下一场和上一场一样"押：上一场掉血 → 押失败，没掉 → 押胜利（血量只减不增）。
        一回合押一张；读不到两回合血量就先不押；和判断相反的那张在手里放过 2 个回合也照押（总比烂在手里强）。"""
        m = self.PREDICT_RE.search(self._card_text(name) or self.pieces.get(name, {}).get("skill") or "")
        r = state.round_no or 0
        if not m or self._predict_round == r or r <= 1:
            return None
        h1, h0 = self._hp_by_round.get(r), self._hp_by_round.get(r - 1)
        if h1 is None or h0 is None:
            return None
        expect = "失败" if h1 < h0 else "胜利"
        held = r - self._hand_since.get(name, r)
        if m.group(1) != expect and held < 2:
            return None
        dst = self._any_hero_pos(state)
        if dst is None:
            return None
        self._last_card_use = (name, r)
        self.note("用牌", round=r, 牌=name, 类别="天赋牌", 目标="自用", 依据=f"上一场血量 {h0}->{h1}，押{m.group(1)}")
        logger.info("打出「%s」（上一场血量 %s -> %s，预计下一场%s；在手里 %d 回合）", name, h0, h1, expect, held)
        return {"type": "drag_card", "piece": name, "src": (slot["x"], slot["y"]), "dst": dst,
                "kind": "效果牌", "self_cast": True, "predict": True}

    def decide_entry_boost(self, state: GameState) -> Optional[dict[str, Any]]:
        """**锦囊**（卡面"使用后，本回合打出的登场英雄牌，会额外触发1次登场效果"）：必须**先于**登场英雄打出，
        所以排在合成 / 上场前面。条件：手里有带「登场：」效果的英雄牌、本回合打得出去（有空人口，或场上有同名可合成）。
        一回合一张（同回合再打不叠加多少，留给下回合）。"""
        if self._shop_blocks_board(state) or self._boost_round == state.round_no:
            return None
        hand = state.hand_slots or []
        slot = next((sl for sl in hand if self.ENTRY_BOOST_RE.search(
            self._card_text(sl["name"]) or self.pieces.get(sl["name"], {}).get("skill") or "")
                     and not self._parked(sl["name"])), None)
        if slot is None:
            return None
        room = state.population_valid and state.population_cap - state.population >= 1
        entry = [sl["name"] for sl in hand
                 if self.pieces.get(sl["name"], {}).get("category") == "英雄牌"
                 and self.ENTRY_LINE_RE.search(self.pieces.get(sl["name"], {}).get("skill") or "")
                 and (sl["name"] in self._field or room)]
        if not entry:
            return None
        dst = self._any_hero_pos(state)
        if dst is None:
            return None
        self._last_card_use = (slot["name"], state.round_no)
        self.note("用牌", round=state.round_no, 牌=slot["name"], 类别="天赋牌", 目标="自用", 依据=f"先于登场英雄 {entry}")
        logger.info("先打「%s」：本回合要打出的登场英雄 %s 会多触发一次登场", slot["name"], entry)
        return {"type": "drag_card", "piece": slot["name"], "src": (slot["x"], slot["y"]), "dst": dst,
                "kind": "效果牌", "self_cast": True, "entry_boost": True}

    def _nearest_same_fac(self, name: str, fac: str) -> tuple[list[str], int]:
        """场上离 name 最近的 fac 派系英雄（按格子步数）：(并列最近的那些人, 步数)。"""
        p = self._field.get(name)
        if p is None:
            return [], 99
        others = [(self._hex_dist(p, q), n) for n, q in self._field.items()
                  if n != name and self._faction(n) == fac and self.pieces.get(n, {}).get("category") == "英雄牌"]
        if not others:
            return [], 99
        d = min(x[0] for x in others)
        return [n for dd, n in others if dd == d], d

    def _level_trade_target(self, state: GameState, name: str, kind: str):
        """**换帅**（"指定1名【三分】英雄与其最近的【三分】英雄交换等级"，level_swap）/
        **挂帅**（"指定1名【三分】英雄，使其等级等同于1名最近的【三分】英雄"，level_copy）给谁 → (英雄, 格子, 多拿几级) 或 None。
        换帅只对主C 用、且主C 的最近同派系英雄要**唯一**（不知道跟谁换就不换）；挂帅对任何非消耗件都行，并列最近时按最差的算。
        两人等级都要读得到（读不到不赌），至少多拿 LEVEL_TRADE_MIN 级。"""
        m = self.NEAREST_FAC_RE.search(self._card_text(name) or self.pieces.get(name, {}).get("skill") or "")
        if not m:
            return None
        fac = {"三分": "三分之地", "大河": "大河流域"}.get(m.group(1), m.group(1))
        carry = self._carry_piece if self._carry_piece in self._field else ""
        best = None
        for t, p in self._field.items():
            if self._faction(t) != fac or self._cell_dead(p) or self.pieces.get(t, {}).get("category") != "英雄牌":
                continue
            if kind == "level_swap" and t != carry:
                continue
            if t != carry and self._is_disposable(t):
                continue
            near, d = self._nearest_same_fac(t, fac)
            if not near or d >= 99 or (kind == "level_swap" and len(near) != 1):
                continue
            lt, ln = self._unit_lv.get(t), [self._unit_lv.get(n) for n in near]
            if lt is None or any(x is None for x in ln):
                continue
            gain = min(ln) - lt
            if gain < self.LEVEL_TRADE_MIN or not self.card_target_ok(name, t)[0]:
                continue
            key = (t == carry, self._is_keeper(t), gain)
            if best is None or key > best[0]:
                best = (key, t, p, gain)
        return (best[1], best[2], best[3]) if best else None

    def _equip_nobody(self, card: str) -> bool:
        """临时规则 dead_equip_route（2026-10-16 晚第 676 局：阿科米亚之子 / 月影之力在手里，场上全是日落海、没有露娜）：
        场上的人、锁定路线最终阵容里的人，没有一个能穿（_equip_allowed）→ 这局基本用不上。路线没锁时不判。"""
        if not self.route_locked():
            return False
        from agent.decision import routes as _R
        lu = _R.by_key(self._route_key) if self._route_key else None
        team = set(_R.final_team(lu)) if lu else set()
        for h in set(self._field) | team:
            if self.pieces.get(h, {}).get("category") == "英雄牌" and self._equip_allowed(card, h):
                return False
        return True

    def _equip_void(self, card: str) -> str:
        """装备效果要靠**场上某个特定单位**、而这局不可能有它 → 返回那个单位名（装备无效），否则 ""。
        只读卡面：「穿戴者所受伤害的30%将转移到**小鹿**身上」—— 小鹿是哪位棋手的召唤物，看哪张棋手技能牌召唤它
        （cards.json 的 lord 字段：可爱小鹿 → 瑶妹）。不是本局棋手的、又不在场上 → 这件装备白穿，还占主C 一个装备位（3 件上限）。"""
        txt = self._card_text(card) or self.pieces.get(card, {}).get("skill") or ""
        if not txt:
            return ""
        from agent.decision import hero_traits as _ht
        db = _ht._cards()
        for m in self.VOID_UNIT_RE.finditer(txt):
            x = m.group(1)
            if (db.get(x) or {}).get("category") == "英雄牌" or x in self._field:
                continue
            owners = {c.get("lord") for c in db.values()
                      if c.get("lord") and c.get("category") == "棋手技能牌" and x in (c.get("skill") or "")}
            if owners and self._lord not in owners:
                return x
        return ""

    def _dead_hand_card(self, state: GameState) -> Optional[dict[str, Any]]:
        """**规则不会用的牌别一直占手牌**（卡牌审查第 5 / 8 条：传承第 470/474 局、换帅第 449 局、锦囊第 458/459 局 0 次使用；
        手牌越挤读得越不准）。卖：审定 hold 且没有 rule_use 的牌（听天由命 / 机关 / 变身 / 驱雷 / 星源星徽 / 望梅止渴…）、
        这局无效的装备（_equip_void）。在手里放过一整回合才卖（给条件变化 / Laya 影子题留一回合）；手牌快满（≥8）时不等。
        不卖：有 rule_use 的（锦囊 / 预言 / 换帅 / 挂帅 / 神秘英雄牌）、成对装备（等另一件）、转瞬牌（卖了不给能量）、
        拿血量换东西的牌（绝处逢生：残血时很值，留给 Laya 判）。"""
        hand = state.hand_slots or []
        r = state.round_no or 0
        for slot in hand:
            name = slot["name"]
            info = self.pieces.get(name, {})
            rv = self.usage_reviewed.get(name) or {}
            if rv.get("rule_use") or name in self.PAIR_EQUIPS or not info:
                continue
            txt = (info.get("skill") or "") + (self._card_text(name) or "")
            if "转瞬" in txt or self.HP_COST_RE.search(txt) or info.get("category") in ("英雄牌", "棋手技能牌"):
                continue
            if rv.get("target") == "hold":
                why = f"审定 hold、规则不会用（{(rv.get('reason') or '')[:24]}）"
            elif info.get("category") == "装备牌":
                continue        # 装备不能卖（2026-10-19）—— 装备牌不进卖手牌（原来 dead_equip_route / 无效装备会去卖，白点）
            else:
                continue
            if self._hand_since.get(name, r) >= r and len(hand) < self.HAND_LIMIT - 2:
                continue
            if (self._sell_tries.get((r, name), 0) >= 2 or self._sell_decided.get((r, name), 0) >= 3):
                continue
            k = (r, name)
            self._sell_decided[k] = self._sell_decided.get(k, 0) + 1
            logger.info("卖掉用不上的手牌 %s（%s）", name, why)
            self.note("卖手牌", round=r, 牌=name, 原因=why)
            return {"type": "sell_card", "piece": name, "src": (slot["x"], slot["y"])}
        return None

    def _rebuild_targets(self, state: GameState, card: str) -> list:
        """**武装重构**（n 件装备 → n+1 件随机**基础**装备，基础装备会再弹四选一、按穿的人重铸）放宽后的目标
        （卡牌审查第 10 条：老条件"非核心且至少 2 件"太严，10 局卖了 9 局）。读卡面：净多 1 件装备、代价是成品被打回基础装备，
        所以拆"装错了 / 装备对他不重要"的人最赚：
          ① 主C：身上的成品**全部**对主C 适配分 <4（错配；穿戴账本要记全）→ 拆了按主C 重铸；
          ② 非主C、不是留场核心、不是主C 人选（评分 ≥7）：身上 ≥1 件就行（1 件换 2 件）；
          ③ 留场核心（主坦）：已满 3 件、且身上的成品对他都 <4。
        一件都没有的不拆（游戏要"摧毁其所有装备"）。返回与原分支同格式、最好的排第一。"""
        from agent.decision import hero_traits as _ht
        hr = self.hero_reviewed
        out = []
        for n, p in self._field.items():
            if self._cell_dead(p) or (state.round_no, card, n) in self._used_on:
                continue
            worn = list(self._worn.get(n, ()) or ())
            k = self._equips_now(state, n, p)          # 不可逆：不用格子的历史最多件数（2026-10-11）
            if k < 1:
                continue
            fits = [f for f in (_ht.item_fit_score(it, n) for it in worn) if f is not None]
            misfit = len(worn) >= k and len(fits) == len(worn) and all(f < 4 for f in fits)
            if n == self._carry_piece:
                if misfit:
                    out.append((2, k, n, p))
                continue
            # 留场核心：阵容核心 / 后期角色 / 引擎零件 / 觉醒过 / 副C。**不用 _is_keeper**：它把"场上估值前 4"都算进去，
            # 场上不到 5 人时人人都是 keeper —— 老条件就是这么把所有目标排光的
            if (self._is_core(n) or n in self._late_roles() or self._is_engine_part(n)
                    or n in self._ever_awakened or n == self.sub_carry()):
                if k >= meta.MAX_EQUIPS and misfit:
                    out.append((1, k, n, p))
                continue
            if hr.get(n, {}).get("carry_score", 5) >= 7:
                continue
            out.append((1 if misfit else 0, k, n, p))
        out.sort(key=lambda x: (x[0], x[1], -self.unit_value(x[2])), reverse=True)
        return [(False, k, -self.unit_value(n), n, p) for _, k, n, p in out]

    def _reviewed_target(self, state: GameState, name: str, r: dict):
        """审定表里的目标类型 -> (收件人, 格子)；"skip" = 现在没有合适的人（先留着）；None = 走原有流程。"""
        mod = self.card_rule_mod(name)
        if mod and not ("古币" in name and self.coin_machine_cell):
            # **天赋改写了这张牌的合法目标**（多彩古币：古币只能给非河洛、还会把人变成同阶随机英雄 +7 级）→
            # 挑 card_target_ok 放行的人里**最不值钱**的（变身丢身份：主C / 路线成员 / 觉醒 / 枫叶舞宿主已被 card_target_ok 挡掉）
            c = [n for n in self._field if not self._cell_dead(self._field[n]) and self.card_target_ok(name, n)[0]
                 and self.pieces.get(n, {}).get("category") == "英雄牌"]
            if not c:
                return "skip"
            pick = min(c, key=lambda n: (self.unit_value(n), self._unit_lv.get(n) or 0))
            logger.info("「%s」按天赋「%s」改写的规则用：给 %s（非【%s】%s）", name, mod[3], pick, mod[1],
                        "，会变身" if mod[2] else "")
            return pick, self._field[pick]
        t = r.get("target") or ""
        f = self._field
        lv = lambda n: self._unit_lv.get(n) or 0  # noqa: E731
        q = lambda n: self.pieces.get(n, {}).get("quality", 0) or 0  # noqa: E731
        carry = self._carry_piece if self._carry_piece in f else ""
        room = lambda n: self._equips_at(state, f[n]) < meta.MAX_EQUIPS  # noqa: E731
        live = [n for n in f if not self._cell_dead(f[n])]
        best = lambda cands: max(cands, key=lambda n: (self.unit_value(n), lv(n))) if cands else None  # noqa: E731
        pick = None
        is_equip = r.get("category") == "装备牌"
        sk_txt = self.pieces.get(name, {}).get("skill") or self._card_text(name)
        if "古币" in name and self.coin_machine_cell:
            # **古币机在场：古币都打给它**（2026-09-30 第 43 局：有古币没用，应给古币机）。
            # 古币机：每对其使用 4 张古币（精致古币算 2 次）依次得 5 张河洛英雄牌、贤者的庇护，之后随机超速成长/普通装备/8 能量
            # —— 比 +2 级值钱得多。古币机不在账本（它不是英雄），格子由 live_match_agent.find_coin_machine 点开确认。
            # 2026-10-12（有古币机且沈梦溪本回合已吃过效果牌时，古币优先给古币机）：卡面「每对X使用1张效果牌，
            # 本回合法力值上限-N」的英雄在场、本回合还没吃过效果牌 → 这一张先给她（法力上限 -15/-30），其余全给古币机。
            # 古币机没有投入上限（4 张一轮，奖励一直给），所以不设封顶。
            mg = self._effect_magnet(live)
            if (mg and self.card_target_ok(name, mg)[0]
                    and not any(r0 == self._round_seen and h == mg for r0, _c, h in self._used_on)):
                logger.info("「%s」本回合先给 %s（每吃一张效果牌法力上限下降），之后的给古币机", name, mg)
                return mg, f[mg]
            return "古币机", self.coin_machine_cell
        if re.search(r"觉醒进度\s*\+", sk_txt):
            # **觉醒进度 +1 的牌（弦音/妙音…）不能给已觉醒的英雄**（2026-09-25 第四轮第 1 局：弦音两次拖给已觉醒的亚连，
            # 游戏拒收，还把亚连从账本里撤了）。只在未觉醒的里挑最有用的：主C > 留到最后的 > 离觉醒最近 > 估值 > 等级；
            # 卡面有品阶范围（弦音"1-3阶"）照范围过滤；没有就先留着。
            m = re.search(r"(\d)\s*-\s*(\d)\s*阶", sk_txt)
            lo_t, hi_t = (int(m.group(1)), int(m.group(2))) if m else (0, 9)

            _exp = set(self.awaken_expected(state) or [])     # 第 793 局 R12：刚打满觉醒、还没回手的盾山又被选成妙音目标

            def not_awk(n):
                return n not in self._ever_awakened and n not in _exp and self.awaken_left(n, state) != 0

            c = [n for n in live if not_awk(n) and lo_t <= q(n) <= hi_t and not self._is_disposable(n)
                 and not (n in self.SINGLE_COPY and _IR_on("single_copy"))]      # 太乙不觉醒（10-20 晚）
            if not c:
                return "skip"
            pick = max(c, key=lambda n: (n == carry, self._is_keeper(n),
                                         -(self.awaken_left(n, state) if self.awaken_left(n, state) is not None else 3),
                                         self.unit_value(n), lv(n)))
            return pick, f[pick]
        if t in ("none", "equipped_noncarry"):
            return None
        if t == "level_dump":
            # **传承：等级摊给 3 名队友**（卡面"指定场上1名10级以上的英雄，将其等级随机分配给3名其他英雄"）。
            # 源 = 等级对它没用的人：马上要卖的老主C（outgoing_unit）、或主C 价值 ≤3 的辅助 / 坦克（等级是被连带效果抬上来的）。
            # 主C 本人、枫叶舞宿主（要卖也由换绑收尾管，先传承再卖更赚 → 宿主是 outgoing 时允许）不做源。
            # 场上要有 ≥3 个别的英雄（不然分不满）；挑等级最高的源（摊出去的最多）。
            def _useless(n):
                if n == carry or lv(n) < 10:
                    return False
                if self.outgoing_unit(n):
                    return True
                if n == self._maple_bound:
                    return False
                hr = self.hero_role(n) or {}
                roles = set(hr.get("combat_role") or [])
                return (hr.get("carry_value", 5) <= 3 and not (roles & {"carry_mag", "carry_phys", "assassin"}))
            heroes = [n for n in live if self.pieces.get(n, {}).get("category") == "英雄牌"]
            if len(heroes) < 4:
                return "skip"
            c = [n for n in heroes if _useless(n)]
            if not c:
                return "skip"
            pick = max(c, key=lambda n: lv(n))
            logger.info("传承：%s（%s 级，等级对它没用）的等级摊给 3 名队友", pick, lv(pick))
            return pick, f[pick]
        if t == "carry":
            if is_equip or self.STEAL_NEAREST_RE.search(self.pieces.get(name, {}).get("skill") or ""):
                return None                        # 装备走 _equip_target；夺取之术走"先挪老主C 再夺"的组合动作
            pick = carry or None
        elif t.startswith("carry_faction"):
            fac = t.partition(":")[2] or next((w.split(":")[1] for w in r.get("when", [])
                                                if w.startswith("faction_on_field:")), "")
            if carry and self._faction(carry) == fac:
                pick = carry
            else:
                # **古币喂不到主C 时**（2026-10-10：古币喂不到主C 时优先给坦克或 40 级技能效果更好的角色；
                # 李元芳的效果只是给其他英雄提供等级，喂他没用）：路线主C 在场且同派系 → 给他；
                # 否则同派系里 坦克 / 40 级输出高的优先，只会给别人涨等级的（李元芳这类，不是坦克）排最后
                rc = (self._active_comp or {}).get("carry") if self.route_locked() else ""
                fl = [n for n in live if self._faction(n) == fac]
                if rc in fl:
                    pick = rc
                elif fl:
                    # 坦克 > 非 C 位里 40 级技能效果好的 > 其他；别喂出第二个主C（C 位英雄排后），只给别人加等级的（李元芳）最后
                    pick = max(fl, key=lambda n: (not (self.feeds_carry(n) and not self._is_tank(n)),
                                                  self._is_tank(n), not self.c_position_hero(n),
                                                  self._dps40(n), lv(n) or 0))
                else:
                    pick = None
        elif t == "lowest_tier_high_level":
            if re.search(r"变为", sk_txt or ""):
                # **变身牌（化铁为金：1~3 阶随机变成不在场的 5 阶觉醒英雄、保留等级）绝不点主C / 路线成员**
                # （2026-09-30 第 42 局：换英雄的效果牌打在了玄策身上）：主C 是整条路线的落点
                # （玄策的闪现引擎要它当主C），变成随机 5 阶等于把路线拆了。只给"等级高、但不在路线里、不是主C"的
                # 1~3 阶功能件；没有就先留着。
                want = set(self.route_stage_members(self._round_seen) or ())
                c = [n for n in live if q(n) <= 3 and n != carry and n not in want
                     and n not in self._ever_awakened and not self._is_engine_part(n)
                     and not self._is_level_engine(n) and (lv(n) or 0) >= 10]
                pick = max(c, key=lambda n: (lv(n) or 0)) if c else None
            elif carry and q(carry) <= 3:
                pick = carry
            else:
                c = [n for n in live if q(n) <= 3 and not self._is_disposable(n)]
                want = self.route_stage_members(self._round_seen)
                # 路线这阶段要的人优先（2026-09-30 第 36 局：两张复印机都给了牛魔，牛魔觉醒后卖不掉、枫叶舞卡在它身上）
                pick = max(c, key=lambda n: (n in want, self.can_kill(n), lv(n), self.unit_value(n))) if c else None
        elif t == "low_tier_noncarry":
            # **限制性进化**（2026-09-24 研究的使用时机）：
            # 卡面"指定移除 1 名 1~2 阶英雄，获得高 2 阶的随机英雄牌，且上阵时为原等级 +10" —— 把低阶英雄**连同等级**
            # 换进高 2 阶的身体、再白送 10 级。原来排除了阵容核心，而场上 1~2 阶的全被标成核心/主C → 两张牌一直烂在手里。
            #   可选：1~2 阶、没觉醒（觉醒的词条翻倍，换掉亏）、不是真正的输出主C（评分 ≥6 且非过渡件）；阵容核心也可以；
            #         前期的主C 不动；**过了前期的过渡主C（曜这类）最该换**（等级 +10 搬进 3~4 阶身体，比卖掉强得多）。
            #   排序：等级越高越好；过渡主C +15；卡面带全队被动（"你的英雄…"、"每…永久"）的 −8（换掉就丢了效果）。
            rnd = state.round_no

            def evo_ok(n):
                if q(n) > 2 or n in self._ever_awakened:
                    return False
                trans = self._transitional_carry(n, rnd)
                if n == self._carry_piece and not trans:
                    return False
                return trans or self.hero_reviewed.get(n, {}).get("carry_score", 5) < 6

            def evo_score(n):
                sk = self.pieces.get(n, {}).get("skill") or ""
                team_engine = "你的英雄" in sk or ("每" in sk and "永久" in sk)
                return (lv(n) or 1) + (15 if self._transitional_carry(n, rnd) else 0) - (8 if team_engine else 0)
            c = [n for n in live if evo_ok(n)]
            pick = max(c, key=evo_score) if c else None
        elif t in ("entry_hero", "prep_hero"):
            tag = "登场" if t == "entry_hero" else "整备"
            c = [n for n in live if re.search(rf"{tag}\s*[：:]", self.pieces.get(n, {}).get("skill") or "")
                 and self.card_target_ok(name, n)[0]]     # 同 card_target_ok：反应器（朵莉亚「每当…触发整备效果时」）游戏会拒收
            rich = [n for n in c if re.search(r"获得|等级|能量", self.pieces.get(n, {}).get("skill") or "")]
            pick = best(rich or c)
        elif t == "tier_bulk":
            by = collections.Counter(q(n) for n in live)
            if by:
                top = max(by.values())
                tiers = [k for k, v in by.items() if v == top]
                tier = q(carry) if carry and q(carry) in tiers else max(tiers)
                pick = carry if carry and q(carry) == tier else best([n for n in live if q(n) == tier])
        elif t == "prev_carry":
            # 换主C 之后的老主C（遗志）：新主C 已在场且站满一回合、老主C 还在场上，才打给老主C
            pc = self._prev_carry
            if (pc and pc in f and carry and pc != carry and self._carry_switch_round < state.round_no
                    and not self._is_carry(pc)):
                if len(f) <= 2 or ("红叶染锋" in self._lord_boosts and self._maple_bound == pc):
                    # 只剩新老主C 两人（等级只能给新主C）；或老主C 绑着枫叶舞且有红叶染锋：
                    # 离场后 50% 真实等级**确定**留给改绑的新主C，遗志再把 100% 给随机一人
                    pick = pc
        elif t.startswith("awakened_faction:"):
            fac = t.split(":")[1]
            pick = best([n for n in live if n in self._ever_awakened and self._faction(n) == fac])
        elif t == "disposable":
            if self.REMOVE_CARD_RE.search(name + (self.pieces.get(name, {}).get("skill") or "")):
                return None                        # 体面离场 走原有"移除换能量"分支
            c = [n for n in live if self._is_disposable(n)]
            pick = max(c, key=lambda n: lv(n)) if c else None
        elif t == "tank":
            # 先给**真坦克**（职业/标签是坦克），没有才给其他前排（战士如姬小满）—— 2026-09-24 第二轮第 4 局反伤重甲给了姬小满
            # role 字段太粗（姬小满、曜这类战士也标 tank），只有标签里的「坦克」能区分牛魔、白起这样的真坦克
            true_tank = lambda n: any("坦克" in x for x in self.pieces.get(n, {}).get("tags", []))  # noqa: E731
            # 防御装**不给主C 和输出位**（2026-09-25 第四轮第 1 局：主C 孙悟空只有「前排」标签，被当成坦克穿了两件防御装）
            # **坦克得是留着用的人**（2026-10-10 第 515 局：主C 虞姬只有 1 件，极寒风暴 / 护命灵珠 却给了刚上场的 1 级大司命，后来手动挪回主C）：
            # 消耗件不给；等级低于主C 的 1/4（至少 3 级）也不给 —— 防御装压在马上会被换掉的人身上等于白扔，退回给主C
            _clv = lv(carry) if carry else 0
            _min_lv = max(3, (_clv or 0) // 4)
            ok_t = [n for n in live if self._is_tank(n) and room(n) and self._equip_allowed(name, n)
                    and self.equip_role_for(n).split("|")[-1] in ("tank",) and not self._is_disposable(n)
                    and (self._unit_lv.get(n) is None or self._unit_lv.get(n) >= _min_lv)]
            _team = self._route_team()             # 主坦从核心阵容里选（2026-10-13），阵容里没有坦克才给别的前排
            if "无法拆卸" in self._card_text(name):
                # 无法拆卸 = 穿上就跟那个人走（第 636 局第 5 回合：盛华给了 1 阶过渡件苏烈，路线主坦盾山第 6 回合才到）。
                # 只给阵容里的坦克；还没到就先留在手里
                pick = best([n for n in ok_t if n in _team and true_tank(n)]) or best([n for n in ok_t if n in _team])
                return (pick, f[pick]) if pick else "skip"
            pick = (best([n for n in ok_t if n in _team and true_tank(n)]) or best([n for n in ok_t if n in _team])
                    or best([n for n in ok_t if true_tank(n)]) or best(ok_t))
            if pick is None:
                return None                        # 没有坦克：退回给主C
        elif t == "support":
            # 10-20（review-data-consistency 第 4 条）：装备牌不给辅助（瑶除外，10-19）→ 灵泉项链 / 玄微之种 / 不屈意志给坦克
            _eq = (self.pieces.get(name) or {}).get("category") == "装备牌"
            pick = (best([n for n in live if self._is_support(n) and room(n) and not (_eq and self._no_gear(n))])
                    or best([n for n in live if self._is_tank(n) and room(n)]))
            if pick is None:
                return None
        elif t in ("ranged", "melee", "low_tier"):
            ok = lambda n: self._equip_allowed(name, n) and room(n) and (t != "low_tier" or q(n) <= 3)  # noqa: E731
            pick = carry if carry and ok(carry) else best([n for n in live if ok(n) and not self._is_support(n)
                                                           and not self._is_tank(n)])
        elif t.startswith("hero:"):
            h = t.split(":", 1)[1]
            if h in f and room(h):
                pick = h
            elif name == "双剑·雄":
                return None                        # 没有干将莫邪：当攻速装给主C
        elif t.startswith("faction_tank:"):
            fac = t.split(":", 1)[1]
            c = [n for n in live if self._faction(n) == fac and room(n) and n != carry and self._is_tank(n)]
            pick = best(c)
            if pick is None:
                return "skip"
        elif t.startswith("faction_near_carry:"):
            fac = t.split(":", 1)[1]
            cpos = f.get(carry) if carry else None
            c = [n for n in live if self._faction(n) == fac and room(n) and n != carry]
            if not c or cpos is None:
                return "skip"
            pick = min(c, key=lambda n: (self._hex_dist(f[n], cpos), -self.unit_value(n)))
        elif t == "support_carry_faction":
            cf = self._faction(carry) if carry else ""
            c = [n for n in live if room(n) and n != carry and cf and self._faction(n) == cf
                 and (self._is_support(n) or self._is_tank(n))]
            pick = best(c)
            if pick is None:
                return "skip"
        elif t == "carry_not_fast":
            ok = lambda n: room(n) and not self.high_attack_speed(n) and not self._is_support(n) and not self._is_tank(n)  # noqa: E731
            pick = carry if carry and ok(carry) else best([n for n in live if ok(n) and n != carry])
            if pick is None:
                return "skip"
        elif t.startswith("faction_weakest:"):
            # 星源星徽（【日落海】专属，穿戴者不被阿科米亚核心分配等级）：给场上最没用的低等级同派系英雄，
            # 核心的临时等级就集中到主C 等其他日落海英雄身上（2026-10-13）。不给主C / 路线核心件
            fac = t.split(":", 1)[1]
            _core = self.route_core() | set(self.min_engine().get("early_priority") or [])
            c = [n for n in live if self._faction(n) == fac and room(n) and n != carry and n not in _core]
            pick = min(c, key=lambda n: (self.unit_value(n), lv(n))) if c else None
            if pick is None:
                return "skip"
        elif t.startswith("faction:"):
            fac = t.split(":", 1)[1]
            pick = best([n for n in live if self._faction(n) == fac and room(n)])
        else:
            return None
        if not pick or pick not in f:
            return "skip"
        # **"对他用效果牌有额外收益"的英雄优先接牌**（2026-09-30：按卡面效果利用英雄特性：
        # 沈梦溪「每对沈梦溪使用1张效果牌，本回合法力值上限-15/-30」→ 大招更勤）。只在**不改变收益**时改点他：
        # 他是主C 本来就会点他；同品阶的群体牌（有福同享）/ 同派系牌（古币）他也吃得到时改点他。
        if not is_equip and pick != (mag := self._effect_magnet(live)) and mag and self.card_target_ok(name, mag)[0]:
            # carry_faction（古币：指定 1 人 +2 级）是**单体**加等级 —— 改给他 = 等级从主C 身上挪走，收益变了
            # （2026-10-07 第 337 局：主C 狄仁杰，5 张古币 + 精致古币改给了不是主C 的沈梦溪，双主C）。只有原目标不是主C 时才改。
            same = ((t == "tier_bulk" and q(mag) == q(pick))
                    or (t.startswith("carry_faction") and self._faction(mag) == self._faction(pick)
                        and pick != self._carry_piece)
                    or (t.startswith("faction:") and self._faction(mag) == t.split(":", 1)[1] and room(mag)))
            if same:
                logger.info("「%s」改给 %s（对他用效果牌有额外收益），原目标 %s", name, mag, pick)
                pick = mag
        ok_t, why_t = self.card_target_ok(name, pick)
        if ok_t and is_equip:
            from agent.decision import hero_traits as _htm
            if _htm.item_dmg(name) and _htm.item_mismatch(name, [pick]):
                # 审定路径原来不查伤害类型（2026-10-13 特殊装备逐件核对：梦魇之牙 / 法潮灵杖 / 聚灵法杖 / 静谧之铃 审定"给主C"，物理主C 也会拿到）
                ok_t, why_t = False, f"伤害类型不对（{_htm.item_dmg(name)} 装给 {_htm.dmg_type(pick, legacy=False)}）"
        if not ok_t:
            # **审定目标也要过卡面限制**（2026-10-13 卡面审计第 3 条：援军「随机获得 4 张同阵营英雄牌」给无阵营主C 公孙离 → 游戏拒收）。
            # 合法性检查，不是价值判断：不合规就改投合规的人（同样按估值），没人合规先留着
            from agent.decision import hero_traits as _htm2
            _locked = "无法拆卸" in (self._card_text(name) or "")
            alt = best([n for n in live if n != pick and self.card_target_ok(name, n)[0] and (not is_equip or room(n))
                        and not (is_equip and _htm2.item_dmg(name) and _htm2.item_mismatch(name, [n]))
                        # 装上拆不下来的神装（聚灵法杖 / 百锻刀 / 盛华）只改投能当主C 的输出位，不给辅助 / 坦克 / 补位件
                        # （2026-10-14 第 626 局：聚灵法杖对物理主C 守约不对口 → 改投给了辅助明世隐；它是法师神装，装上就拿不下来）
                        and not (_locked and (self._is_support(n) or self._is_tank(n) or not self._carry_eligible(n)))])
            logger.info("「%s」审定目标 %s 不合卡面限制（%s）→ %s", name, pick, why_t, alt or "先留着")
            if alt is None:
                return "skip"
            pick = alt
        return pick, f[pick]

    HP_COST_RE = re.compile(r"棋手生命值变为|棋手生命值-|扣除.{0,4}棋手生命|失去.{0,6}生命值")

    def _is_self_cast(self, name: str, info: dict) -> bool:
        # **拿棋手血量换东西的牌不自动用**（2026-09-30：绝处逢生不能随便用，「棋手生命值变为1，随机获得10张效果牌」
        # 第 44 局被当自用牌直接打了）。血量读数又不可靠，没法判断"已经是残血、代价很小"—— 一律先留着，交给人。
        txt = (info.get("skill") or "") + (info.get("desc") or "") + (self._card_text(name) or "")
        if self.HP_COST_RE.search(txt):
            # 过渡期：先留着，但**记下决策点**攒 Laya 训练数据（不是一律不打，要看性价比，后面交给 Laya 判断）。
            # 以后做成 Laya 的"现在用 / 先留着"题（题面写清扣多少血、换来什么、当前血量 / 回合 / 能量 / 场面）。
            st = getattr(self, "_last_state_for_record", None)
            if st is not None:
                self.record_decision("hp_trade", st, [{"card": name, "text": txt[:80]}, "hold"], "hold",
                                     hp=getattr(st, "hp", None), round=getattr(st, "round_no", None),
                                     gold=getattr(st, "gold", None), policy="rule_hold_pending_laya")
            return False
        if self._usage_self_cast(name):
            # **用牌表说"能打出、不需要目标"就是自用牌，不分效果牌/天赋牌**（2026-09-24：大明星、人才搜寻×3
            # 攒在手里不用 —— 这两张是天赋牌，原来这里只认效果牌；用牌表早就标好了，只是一直没接进规则）。
            return True
        if info.get("category") != "效果牌":
            return False
        if self.SELF_CAST_NAME_RE.match(name):
            return True
        text = (info.get("skill") or "") + (info.get("desc") or "")
        if any(k in text for k in self.HOLD_KEYWORDS):
            return False                                   # 整备类留到下回合
        # **「转瞬」且不需要指定目标的效果牌 = 拿到就打**（2026-09-24：援护战术、招募战术要立即用掉）。
        # 转瞬：回合结束自动销毁、卖了也不给能量；「全体…」「随机获得…」这类不点名，拖到任一英雄身上即可。
        # 原来援护战术被当成"给主C 的牌"，拖向账本里过期的主C 格子，失败。
        if "转瞬" in text and "指定" not in text:
            return True
        return any(k in text for k in self.SELF_CAST_KEYS)

    def _pick_self_cast(self, state: GameState) -> Optional[dict[str, Any]]:
        """手里有"白送棋手经验/等级"或「N阶效果锦囊」这类牌就立刻打出去。

        这类牌的收益**随回合复利**（早一回合升级 = 早一回合多人口 + 高阶商店概率），
        而且是白来的 0 费牌，留在手里只会挤爆手牌上限。

        **落点必须是场上的一个英雄，不能是空格。**
        2026-09-22 第七局实测：往空格拖连续三次全部失败
        （`目标 None 不在那儿`，(540,480)/(660,480)/(480,480) 依次白拖），
        说明这游戏的效果牌是"拖到英雄身上"打出的，不存在"丢到空地"这种用法。
        所以场上一个英雄都没有时**宁可不打**，也别再去空格上试。
        """
        self._last_state_for_record = state
        dst = self._any_hero_pos(state)
        # 转瞬牌排最前（回合结束就没了）
        slots = sorted(state.hand_slots, key=lambda sl: (
            "转瞬" not in (self.pieces.get(sl["name"], {}).get("skill") or ""),
            -(self.usage_reviewed.get(sl["name"], {}).get("priority") or 0)))
        for slot in slots:
            name = slot["name"]
            info = self.pieces.get(name, {})
            if not self._is_self_cast(name, info):
                continue
            if self._parked(name):
                continue          # 本回合已搁置（这条路径原来漏了检查，搁置了还每帧出同一个动作）
            ok_g, why_g = self.group_card_ok(name)
            if not ok_g:
                if self._skip_logged.get(("group_card", name)) != self._round_seen:
                    self._skip_logged[("group_card", name)] = self._round_seen
                    logger.info("%s 先留着：%s", name, why_g)
                continue
            u = self._usage_self_cast(name)
            if u:
                ok, why = self._usage_when_ok(u, state, name)
                if not ok:
                    if self._skip_logged.get(("usage_when", name)) != self._round_seen:
                        self._skip_logged[("usage_when", name)] = self._round_seen
                        logger.info("%s 先留着（用牌表：%s）", name, why)
                    continue
            if dst is None:
                logger.info("%s 想打出但场上没有英雄可作落点，先留着（效果牌必须拖到英雄身上）",
                            name)
                return None
            self.note("用牌", round=state.round_no, 牌=name, 类别="效果牌",
                      目标="自用（拖到场上英雄身上打出）")
            return {"type": "drag_card", "piece": name,
                    "src": (slot["x"], slot["y"]), "dst": dst,
                    "kind": "效果牌", "self_cast": True}
        return None

    def _any_hero_pos(self, state: GameState) -> Optional[tuple[int, int]]:
        """场上任意一个英雄的位置，优先主力（效果牌的落点）。"""
        # 自用牌（N阶效果锦囊/研读会…）也要落到某个英雄身上，判据和装备/效果牌完全一样，
        # 所以共用 `_hero_cell`（见那里的说明：这条判据今晚在三条路上各漏过一次）。
        return self._hero_cell(state, prefer=self._carry_piece)

    # ---------- 棋手经验型英雄（弈星等） ----------
    #
    # 一条运营链（2026-09-22）：**弈星**（英雄牌，3 费，无阵营，
    # "棋手升级时，棋手经验+2"，觉醒 +4）在前期大家都低等级时特别值——
    # 买来上场 → 升一次棋手（白拿 2 点经验）→ 它不是阵容牌，吃完收益就卖掉腾人口。
    # 判据同样用卡面文本，不写死"弈星"：以后出同类英雄自动吃到。
    #
    # 注：这张牌长期被漏掉还有个更隐蔽的原因——cards.json 按卡名做 key，而
    # 「弈星」同时是棋手牌和英雄牌，英雄版被棋手版整张覆盖，决策层查到的是一张
    # skill 为空的棋手牌，自然永远不会买。已在 fetch_card_db.merge_cards 修掉。
    LEVEL_SYNERGY_KEYS = ("棋手升级时", "棋手经验")

    def _is_level_engine(self, name: str) -> bool:
        info = self.pieces.get(name, {})
        if info.get("category") != "英雄牌":
            return False
        text = (info.get("skill") or "") + (info.get("awaken") or "")
        return any(k in text for k in self.LEVEL_SYNERGY_KEYS)

    def _engine_on_field(self) -> bool:
        """场上有没有棋手经验件（弈星这类，升级时额外吃经验）。"""
        return any(self._is_level_engine(n) for n in self._field)

    def _level_engine_ready(self, state: GameState) -> bool:
        """这回合还打算升级吗——决定弈星该不该先上场、以及上完还该不该留。"""
        return (state.level_cost is not None and state.level_cost > 0
                and state.gold - state.level_cost >= self.level_reserve
                and state.round_no != self._leveled_round)

    @staticmethod
    def entry_only_text(skill: str) -> bool:
        """卡面的效果条目**全部**是「登场：…」（关键词行如「闪现」不算效果）。
        这类英雄的价值只在上场那一下兑现（登场效果只在上场时生效一次，
        除非用强制触发登场效果的效果牌），之后只剩身板。全库 85 个英雄里约 10 个：
        孙尚香 / 马可波罗 / 赵云 / 阿轲 / 貂蝉 / 甄姬 / 庄周 / 蒙犽 / 露娜 / 周瑜。"""
        lines = [l.strip() for l in (skill or "").splitlines() if l.strip()]
        eff = [l for l in lines if not (len(l) <= 4 and not re.search(r"[，。：:、（）()]", l))]
        return bool(eff) and all(l.startswith("登场") for l in eff)

    def spent_entry_unit(self, name: str, hand_names=()) -> bool:
        """场上这个英雄的登场效果已经用掉、又不在当前路线里 —— 只剩一个占人口的身板。
        第 126 局：拍卖来的阿轲（3 阶）登场给主C 亚连 +1 级，之后在 6/6 的场上一直占着位置，
        而 3 阶以上不算"登场杂牌"（`_is_disposable` 的老规矩，为第十局的蒙犽加的），没人卖它。
        条件：路线已锁、主C 定了、它不是路线成员 / 主C / 核心 / 引擎零件 / 觉醒过；
        手里有能强制触发登场效果的牌时不算（还能再吃一次）：按卡面原文认「触发其登场效果」「触发所有英雄的登场效果」
        （唤醒 / 集体唤醒）；「你的英雄触发登场效果时…」是反应器（小乔 / 乐善好施），不算。
        注：关键词词典里「合成」= 使用场上已存在的英雄牌，「登场」= 使用此卡牌后触发 —— 合成同名可能再触发一次登场，
        所以这里不拦"买同名合成"，只管"占着人口的身板该不该让位"。"""
        if name not in self._field or not self.route_locked() or not self.carry_settled():
            return False
        info = self.pieces.get(name, {})
        if info.get("category") != "英雄牌" or not self.entry_only_text(info.get("skill", "")):
            return False
        if (name in self.route_members() or name in self._ever_awakened or self._is_carry(name)
                or self._is_core(name) or self._is_engine_part(name)):
            return False
        for h in hand_names or ():
            if re.search(r"触发(其|所有英雄的|.{0,4}名英雄的)登场",
                         self._card_text(h) or self.pieces.get(h, {}).get("skill", "") or ""):
                return False
        return True

    def _is_disposable(self, name: str) -> bool:
        """这个单位是不是"用完就卖"的消耗件 —— 别把技能牌/装备投资在它身上。

        两类：棋手经验件（弈星，升完级就卖掉腾人口）、
        只为吃一次登场/整备效果才上场的阵容外杂牌（`_entry_fodder`）。
        阵容内的主力/核心永远不算消耗件，哪怕它恰好也带这些标签。
        """
        if self._is_carry(name) or self._is_core(name) or self._is_engine_part(name):
            return False
        # **3 阶以上不当"吃登场效果的杂牌"**（2026-09-23 第十局：4 阶蒙犽作为登场杂牌上场，
        # 估值变成 1，第 8 回合被卖掉换了个 28 分的老夫子 —— 而它后来是主C）
        fodder = name in self._entry_fodder and self.pieces.get(name, {}).get("quality", 0) <= 2
        # 路线已锁、主C 已定后，阵容外"只有登场效果"的英雄登场完就是身板（不分品阶，见 spent_entry_unit）
        return self._is_level_engine(name) or fodder or self.spent_entry_unit(name)

    def _nonsupport_temp_exists(self, exclude: str) -> bool:
        """场上有别的、不是配套、能当临时主C 的人（2026-10-11 回退：一律不让配套当临时主C → 开局场上只有配套时没有主C、
        枫叶舞也没人绑；第 565~570 局每局平均 1 个回合无主C，第 8 回合主C 中位 3.5 级）。只在有替代时才让配套让开。"""
        if self.__dict__.get("_ns_guard"):
            return False
        self._ns_guard = True
        try:
            return any(n != exclude and self.pieces.get(n, {}).get("category") == "英雄牌"
                       and not self.keep_as_support(n) and self._carry_eligible(n) for n in self._field)
        finally:
            self._ns_guard = False

    def keep_as_support(self, name: str) -> bool:
        """**之后在阵容里有用的配套**：指定路线（首选 + 备选）最终阵容里的非主C 成员，或卡面功能是给主C / 全队涨等级的人
        （hero_roles functions：carry_level / team_level）。前期不让他们当临时主C、不绑枫叶舞 —— 否则真主C 到了要卖掉他们才能换绑
        （2026-10-11：开局不宜让李元芳当主C —— 很多阵容里他有用，当了主C 之后还要卖掉再买一个，是亏的）。"""
        if not name:
            return False
        from agent.decision import routes as _R0
        _keys = (self.__dict__.get("_forced_orig") or self.forced_route,) + tuple(self.forced_backups or ())
        _carries = {(_R0.by_key(k) or {}).get("carry") for k in _keys if k} | {self.route_carry()}
        if name in _carries:
            return False       # 某条指定路线的主C（虞姬 / 亚连 …）不算配套（审查 2026-10-11：原来被另一条路线的"成员"身份误伤）
        fs = self.hero_role(name).get("functions") or {}
        if fs.get("carry_level"):
            return True        # 给主C 涨等级的（李元芳 / 明世隐 / 武则天）；team_level 太宽（虞姬 / 孙膑 也有），只靠路线成员判断
        from agent.decision import routes as _R
        for k in (self.__dict__.get("_forced_orig") or self.forced_route,) + tuple(self.forced_backups or ()):
            lu = _R.by_key(k) if k else None
            if lu and name != lu.get("carry") and name in _R.final_team(lu):
                return True
        return False

    def _skill_target(self) -> Optional[str]:
        """棋手技能牌该绑给谁。**前期刻意不绑主力**（2026-09-22 定的打法）。

        机制约束：绑定后要重绑，必须先把被绑的英雄**卖掉**。所以打法是
        "前期先绑一个凑合用的英雄拿收益 → 主 C 确定后把它卖掉 → 再绑给主 C"。
        因此这里的选择顺序是：
          1. **跳过主力**：现在绑上去，等阵容成型想换人就得卖主力，代价太大；
             而且前期的"主力"未必是最终主 C。
          2. **跳过消耗件**（弈星这类升完级就卖的）：不挂给弈星 ——
             它战斗力本来就小、又很快被卖，buff 等于白扔。
          3. 剩下的场上英雄里挑一个"凑合用"的。
        实在只剩主力/消耗件时才退而求其次（空着更亏）。
        """
        from agent.decision import lord_config as _LCm
        _cm = _LCm.load()
        if _cm.present:
            # 第 878 局 R6：亚连刚买进手里、孙膑卖掉空出枫叶舞 → 绑给了名单外的临时主C 西施，下回合又卖西施换绑（3 张西施白买）。
            # 名单内主C（2 阶以上）已在手里 / 正在上场 → 先不绑，等他上场
            _new_this = {e.get("牌") for e in (self.ledger or []) if e.get("event") in ("买牌", "获得卡牌")
                         and e.get("round") == self._round_seen}              # 只等本回合刚拿到的（压在手里好几回合就不等了）
            _soon = (set(self.hand_book_names()) & _new_this) | set((self.__dict__.get("_unconfirmed_deploy") or {}).keys())
            if any(_cm.enabled(n) and _cm.tier(n) >= 2 and n not in self._field for n in _soon)                     and not (self._carry_piece and _cm.enabled(self._carry_piece)):
                return None

        # **主 C 一旦确定，就换绑给主 C**（2026-09-22：换绑给场上新的主C）。
        # 前期还没确定主 C 时才绑"凑合用"的 —— 那时绑上去的人，日后要换绑得先卖掉。
        carry = self._carry_piece
        if carry and carry in self._field and self._is_carry(carry):
            # 2026-10-14 更正：配套件（李元芳）当临时主C 时**照绑**（先绑，看能不能凑出击杀）；
            # 之后的同名留在手里（decide_upgrade），真主C 到了卖掉场上这个、换绑、再把手里的打上去
            return carry
        if carry and carry not in self._field and (
                self._hand_book.get(carry, 0) > 0 or carry in self._awakened or carry in self._ever_awakened):
            # **主C 已定、只是暂时在手里 / 觉醒回手**：等它上场，别绑给别人（2026-09-30 审查：
            # 原来直接退到"凑合"候选，绑上去的人日后要卖掉才能换绑）
            return None
        if self.model_decides() and not (_IR_on("maple_fallback") or _IR_on("maple_temp_host")):
            return None                      # A2（2026-10-15）：没有主C 时不由规则"凑合绑"；主C 由换主C 网络定了再绑
        if _IR_on("maple_temp_host"):
            # 临时规则 maple_temp_host（2026-10-20 晚：开局买两个英雄，把枫叶舞挂在非该主C 核心配件的英雄身上，
            # 第一回合就挂枫叶舞，作为临时主C）：主C 没到时**第 1 回合就**绑临时宿主；
            # 路线主C 是 4 阶以上（韩信 / 刘邦 / 蒙犽）→ 优先干将莫邪 / 孙膑（刻意养的临时主C）。
            # 路线还没锁（同一帧锁路线可能在后面）→ 按开局会锁的那条路线避开核心件
            from agent.decision import lord_config as _LCm
            _cm = _LCm.load()
            if self.route_locked():
                _rc, _core = self.route_carry(), set(self.route_members()) | set(self.__dict__.get("_route_group") or ())
            else:
                _ch = _cm.opening_choice(self.__dict__.get("_banned_seen") or "") if _cm.present else None
                _comp = next((c for c in self.comps if str(c.get("key")) == str((_ch or {}).get("key"))), None)
                _rc = (_ch or {}).get("name") if (_ch or {}).get("kind") == "carry" else ""
                _core = set(_R_members(_comp)) | set((_ch or {}).get("members") or ())
            _tc = _cm.temp_carry_for(_cm.tier(_rc) if _rc else 0) if _cm.present else {}
            for h in (_tc.get("heroes") or []):
                if h in self._field:
                    return h
            _all = sorted(self._field, key=self.unit_value, reverse=True)
            _ok = lambda n: not self._is_disposable(n) and not self._is_level_engine(n)   # noqa: E731  弈星这类升完级就卖的不挂（09-22）
            _strong = lambda n: (self.hero_reviewed.get(n, {}).get("carry_score", 5) or 0) >= self.HERO_CARRY_MIN   # noqa: E731  10-04 马超这类弱的排后（第 214 局）
            _host = ([n for n in _all if n not in _core and _ok(n) and not self._is_support(n) and self.can_kill(n) and _strong(n)]
                     or [n for n in _all if n not in _core and _ok(n) and not self._is_support(n) and self.can_kill(n)]
                     or [n for n in _all if _ok(n) and not self._is_support(n) and self.can_kill(n)])    # 只剩核心件：照样挂（第 1 回合必须挂）
            # 辅助 / 杀不了人的（牛魔）不挂：枫叶舞不给辅助（09-23）、枫叶舞英雄击杀才解锁阿离专属卡（09-30）
            if _host:
                return _host[0]
        # **辅助不绑**（2026-09-23：枫叶舞曾被给了一个 1 级的辅助）：
        # 枫叶舞的收益是给被绑的人涨等级/回血，辅助没有输出，等于白给。
        # 凑合用的人里挑**最值钱**的（原来取 `candidates[0]`，就是账本里第一个，毫无依据）。
        candidates = sorted((n for n in self._field
                             if n != self._carry_piece and not self._is_disposable(n)
                             and not self._is_level_engine(n)      # 弈星：路线阶段成员时 _is_disposable 放行了它（第 40 局 R1 绑上去）
                             and not self._is_support(n) and self.can_kill(n)
                             # **审定评分 ≤ HERO_CARRY_MIN 以下的不绑**（2026-10-04 第 214 局：枫叶舞第 4 回合凑合绑了拍来的马超，
                             # 马超输出非常弱，不该当主C）。这类英雄注定要卖掉才能换绑，绑上去的成长白给
                             and (self.hero_reviewed.get(n, {}).get("carry_score", 5) or 0) >= self.HERO_CARRY_MIN),
                            key=self.unit_value, reverse=True)
        # 配套（李元芳这类，2026-10-11）先让开；只剩配套时照样绑（空着更亏，第 567 / 570 局枫叶舞拖到第 4 回合）
        # 2026-10-13 审计（607/609/612 局开局都绑李元芳，换绑时把这个最低配套件卖了）：给主C 涨等级的配套（carry_level；team_level 太宽，虞姬 / 孙膑 也有）
        # 和当前路线的最低配套件**一律不绑**（10-11 定），宁可晚绑；其他配套仍是"只剩他们时照样绑"
        _me = self.min_engine()
        _me_set = set(_me.get("early_priority") or []) if _me else set()
        _hard = lambda n: n in _me_set or bool((self.hero_role(n).get("functions") or {}).get("carry_level"))  # noqa: E731
        # 2026-10-14 更正：先绑，看能不能凑出击杀 —— 配套件不再一律不绑，只排在非配套的输出位之后
        candidates = ([n for n in candidates if not _hard(n) and not self.keep_as_support(n)]
                      or [n for n in candidates if not _hard(n)] or candidates)
        if candidates:
            return candidates[0]
        if any(True for _ in self._field):
            # **只绑能杀人的**（枫叶舞英雄击杀才解锁阿离专属卡；2026-09-30 第 36 局绑给了牛魔）：没有就先不绑
            if self._skip_logged.get("maple_no_killer") != self._round_seen:
                self._skip_logged["maple_no_killer"] = self._round_seen
                logger.info("场上没有能杀人的英雄，枫叶舞先不绑（绑给坦克/辅助拿不到专属卡）")
            return None
        # 场上只剩主力/消耗件了：绑总比不绑强（buff 是永久的），但记一笔，
        # 因为这意味着以后要换绑就得卖人。
        fallback = next((n for n in self._field
                         if not self._is_disposable(n) and not self._is_level_engine(n)
                         and not self._is_support(n) and not self.keep_as_support(n)), None) or next(
            (n for n in sorted(self._field, key=self.unit_value, reverse=True)
             if not self._is_disposable(n) and not self._is_level_engine(n) and not self._is_support(n)
             and self.can_kill(n)), None)     # 只剩配套时也绑（空着更亏；2026-10-11 回退）
        if fallback:
            logger.info("场上没有凑合用的英雄，枫叶舞只能绑给 %s（以后换绑要卖掉它）",
                        fallback)
            return fallback
        # 场上**只剩消耗件**（弈星这类升完级就卖的）：**不绑**。
        # 枫叶舞不挂给弈星（2026-09-22），而原来这行兜底
        # `next(iter(self._field))` 正好会绑上去 —— 2026-09-23 实战回合1 就绑给了弈星。
        # 不绑没有代价：枫叶舞是常驻手牌、永远可以再打；绑错了反而要卖人才能解绑。
        logger.info("场上只有消耗件，枫叶舞先不绑，等一个留得住的英雄")
        return None

    @staticmethod
    def _on_unit(state: GameState, cell):
        """往单位身上拖东西的落点。**就用格子坐标**（2026-09-22 确认）。

        不能拖到英雄上方的等级图标上，要拖到英雄身上或格子所在的位置。

        这里我走过一段弯路，值得留个记号：
        当时看到"风暴巨剑连拖 3 次到 (917,570) 全失败，而马可波罗上场到同一个格子成功"，
        就推断"上场认格子、贴东西认单位本体"，于是把落点按观测到的徽章往上挪了 40px。
        **真正的原因其实是那个英雄已经满 3 件装备，游戏直接拒绝了拖放** ——
        两件事被我合成了一个因果，结果"修"了一个本来没坏的东西，还把落点挪到了等级图标上。
        教训：**同一批症状里若有一个已经解释得通的原因，先别急着再造一个。**
        """
        return cell

    def _effect_magnet(self, live) -> str:
        """场上"每对他使用效果牌就有额外收益"的英雄（卡面「每对X使用1张效果牌…」，如沈梦溪法力上限 −15/−30）。只读卡面，不写死名字。"""
        for n in live:
            info = self.pieces.get(n, {})
            txt = (info.get("skill") or "") + (info.get("awaken") or "") or self._card_text(n) or ""
            if re.search(rf"每对{re.escape(n)}使用\d*张?效果牌", txt):
                return n
        return ""

    def outgoing_unit(self, name: str) -> bool:
        """**马上要被换掉的老主C**：路线主C 已经在场 / 在手（拍卖、N 选 1、商店拿到的），而 name 是现任或绑着枫叶舞的
        1 阶过渡件、这阶段路线也不要它 → 接任收尾会把它卖掉，再买它的同名（合成 / 觉醒材料）是白花钱。
        2026-10-09 第 473 局（手牌已拿到新主C，开店仍买了一个需要卖掉的老主C）：
        拍卖拿到守约，拍完开店先花 3 能量买了干将莫邪。"""
        if not name or not self.route_locked():
            return False
        rc = self.route_carry()
        if not rc or name == rc:
            return False
        if not (rc in self._field or rc in self.hand_book_names()):
            return False
        if name not in (self._carry_piece, self._maple_bound):
            return False
        if (self.pieces.get(name, {}).get("quality") or 0) != 1:
            return False
        return name not in self.route_stage_members(self._round_seen or 0)

    def recently_sold(self, name: str, rounds: int = 2) -> bool:
        """这回合 / 上回合刚亲手卖掉（场上或手牌）的英雄：别马上从商店买回来（2026-09-30：先卖掉一个英雄，
        开店又买回来，再上场或卖掉，纯属浪费）。路线主C 除外（卖错了要能买回来）。"""
        if self.outgoing_unit(name):
            return True                  # 马上要被换掉卖的老主C：同样别买（见 outgoing_unit）
        r = self._sold_round.get(name)
        if r is None or not name:
            return False
        if name == self._carry_piece or name == (self._active_comp or {}).get("carry"):
            return False
        if name in (self.min_engine().get("early_priority") or []):
            # 当前路线的核心件（2026-10-12 第 585 局：换主C 玄策时为换绑枫叶舞卖了李元芳 —— 他正是玄策路线的核心件，
            # 下一回合商店里出现李元芳，被"刚卖掉"挡掉，整局没补回来）
            return False
        return self._round_seen - r < rounds

    @staticmethod
    def _summon_cells(state: GameState) -> set:
        """画面上 D-FINE 认成 summon（图腾 / 召唤物）的格子。"""
        out = set()
        for k, cls in (state.vision_units or {}).items():
            if cls == "summon":
                try:
                    x, y = k.split(",")
                    out.add((int(x), int(y)))
                except ValueError:
                    continue
        return out

    def object_cells_all(self) -> set:
        """被**非英雄物件**占住的格子（2026-09-30：重剑随花木兰一起出来，会占格子；古币机同理，
        不能在它上面放英雄）。点开看到的是物件就记下，整局有效（物件被移走前一直占着）。"""
        cells = set(self.object_cells)
        if self.coin_machine_cell:
            cells.add(tuple(self.coin_machine_cell))
        return cells

    def summoner_of(self, summon: str) -> str:
        """召唤物是谁召唤的：卡面（技能 / 觉醒）里写着这个召唤物名字的英雄（不写死，读卡面）。"""
        if not summon or self.pieces.get(summon, {}).get("category") == "英雄牌":
            return ""
        cache = self.__dict__.setdefault("_summoner_cache", {})
        if summon not in cache:
            cache[summon] = next((n for n, p in self.pieces.items() if p.get("category") == "英雄牌"
                                  and summon in ((p.get("skill") or "") + (p.get("awaken") or "") + self._card_text(n))), "")
        return cache[summon]

    def note_object(self, cell: tuple, name: str) -> None:
        cell = tuple(cell)
        self.__dict__.setdefault("_obj_confirmed", {})[cell] = name     # 点开读到的名字 = 直接证据（图腾站位只认它）
        if self.object_cells.get(cell) != name:
            self.object_cells[cell] = name
            logger.info("【占格物件】%s 在 %s —— 不是英雄，不往上放人", name, cell)
            self.note("占格物件", 物件=name, 格=f"{cell[0]},{cell[1]}")
        if name == "古币机":
            old = self.coin_machine_cell
            if old and tuple(old) != cell and self.object_cells.get(tuple(old)) == "古币机":
                self.object_cells.pop(tuple(old), None)          # 挪走了（或原来认错格）：旧格子放出来
            self.coin_machine_cell = cell
        # 这一格若在账本里记着某个英雄，那是记错了
        wrong = self._cell_owner.get(cell)
        if wrong and self._field.get(wrong) == cell:
            logger.info("账本把 %s 记在 %s，实际那格是 %s —— 撤掉这条记录", wrong, cell, name)
            self._field.pop(wrong, None)
            self._cell_owner.pop(cell, None)

    _maple_lost_cell: Optional[tuple] = None

    def maple_search_cells(self, state: GameState) -> list:
        """**枫叶舞宿主不在账本里**（卡面灰 = 绑着某人，账上的宿主却不在场）→ 需要点开看的格子（"x,y"）。
        2026-10-04 第 214 局：枫叶舞绑马超，马超因"同格冲突"被撤出账本、认人模型又认不出他（不在类表里），
        换绑收尾要求宿主在账本里 → 枫叶舞挂到局末。
        找法：宿主被挤掉时的那一格 + 画面上有人但认人模型没认出是谁的格子。看场点开读到宿主名字 → confirm_unit 记回账本。"""
        mb = self._maple_bound
        if not mb or mb in self._field or state.phase != "prepare" or state.shop_open:
            return []
        grey = any(sl.get("ready") is False for sl in state.hand_slots or []
                   if self.pieces.get(sl.get("name"), {}).get("category") == "棋手技能牌")
        if not grey:
            return []
        names = self.trusted_unit_names(state) or {}
        out = []
        if self._maple_lost_cell:
            out.append(cell_key(tuple(self._maple_lost_cell)))
        for k in list(state.unit_anchors or {}) + [k for k, t in (state.vision_units or {}).items() if t == "hero"]:
            if k not in names and k not in out:
                out.append(k)
        return out

    def confirm_unit(self, piece: str, cell: tuple) -> None:
        """**点开看过**：这一格站的就是 `piece`。改记账，并清掉这一格攒下的失败记录。

        失败计数（`_cell_fails`）是推断："往这格贴牌/合成失败过几次 -> 那格记错了"。
        可贴牌失败的原因很多（两段式牌、抓错牌、装备满了），不一定是人不在 ——
        2026-09-23 实战：弈星 在 (390,502)，那一格被失败次数"毒死"后，每回合都撤销它的记录，
        点开看场确认是它、记回来，下一帧又被撤销，来回三轮。点开看是**直接证据**，比推断硬。
        """
        cell = tuple(cell)
        self.relocate_unit(piece, cell)
        if piece == self._maple_bound and self._maple_lost_cell is not None:
            logger.info("找回枫叶舞宿主 %s：在 %s", piece, cell)
            self._maple_lost_cell = None
        self._confirmed_at[piece] = (self._round_seen, cell)   # 直接证据：本回合看到它在这一格
        if self._cell_fails.pop(cell, None):
            logger.info("点开确认 %s 就在 %s，清掉这一格的失败记录", piece, cell)
        self._busy_cells.discard(cell)
        self._bad_cells.discard(cell)

    def relocate_unit(self, piece: str, cell: tuple) -> None:
        """把 `piece` 的落点改成**观测到的那一格**。

        为什么需要：拖上场之后，棋子**未必停在我们拖的那一格**
        （游戏会自动归位到最近的合法格，或者那一格其实有人）。
        2026-09-23 实战：孙膑 记在 (300,569)，画面上人却站在 (411,372) 附近，
        于是后面给它贴装备全都拖到了空地上，`装备件数还是 0`。
        拖拽回执只能告诉我们"牌离手了"，落到哪一格要靠**看**。
        """
        cell = tuple(cell)
        old = self._field.get(piece)
        if old == cell:
            return
        if old:
            self._cell_owner.pop(old, None)
            self._equip_seen.pop(cell_key(tuple(old)), None)
            (self.__dict__.get("_equip_hist") or {}).pop(cell_key(tuple(old)), None)
        # 换格后这一格"看到过的最多件数"是上一个站这儿的人的（2026-10-01 第 80 局：亚连 0 件，被旧件数算成 1 件，
        # 云中君铺装备分支就把唯一的输出装雷鸣刃给了坦克钟馗）。件数靠穿戴账本 + 新画面重新算。
        self._equip_seen.pop(cell_key(cell), None)
        (self.__dict__.get("_equip_hist") or {}).pop(cell_key(cell), None)
        self._evict_cell(cell, piece)
        self._field[piece] = cell
        self._cell_owner[cell] = piece
        logger.info("按观测修正落点：%s 从 %s 改到 %s", piece, old, cell)
        self.note("落点修正", 英雄=piece, 原记=f"{old[0]},{old[1]}" if old else "",
                  实际=f"{cell[0]},{cell[1]}", 依据="等级徽章观测")

    @staticmethod
    def _anchor_cells(state: GameState) -> list:
        """画面上**观测到**站着人的格子（等级徽章锚点）。记账为空时的救命稻草。"""
        out = []
        for key in (state.unit_anchors or {}):
            try:
                cx, cy = key.split(",")
                out.append((int(cx), int(cy)))
            except ValueError:
                continue
        return out

    EQUIP_FIT_MIN = 5        # 装备画像适配分 ≥5 才算对口（0~10；英雄前 6 件 = 10）

    def _equip_fits_role(self, card: str, hero: str) -> bool:
        """基础装备和英雄定位对不对口：输出型（风暴巨剑/雷鸣刃/速击之枪/大棒）不给坦克/辅助，
        防御型（守护者之铠/力量腰带）不给输出位；混合型和成品装备不管（成品另有 item_mismatch）。"""
        from . import hero_traits as _ht
        if hero:
            # **按装备画像 / 关系图谱判断谁适合这件**（2026-09-30 起按装备关系图看谁适合）
            sc = _ht.item_fit_score(card, hero, self.hero_role(hero).get("combat_role") or [])
            if sc is not None:
                return sc >= self.EQUIP_FIT_MIN
        kind = _ht.base_kind(card)
        if not kind and card in _ht.EQUIP_CLASS:
            # **成品也要对口**（2026-09-30 第 23 局：卖孙尚香退回的红魔、金色圣剑给了刘禅、钟馗两个坦克）：
            # 输出 / 施法装只给输出位，坦克装只给坦克，保命装（名刀·司命等）谁都行
            cls = _ht.EQUIP_CLASS[card][0]
            kind = "output" if cls.startswith("dps") or cls == "caster" else ("defense" if cls == "tank" else "mixed")
        if not kind or kind == "mixed" or not hero:
            return True
        # 画像里没有这件：退回按卡面定位的粗分（输出装 / 防御装）
        roles = (self.hero_role(hero).get("combat_role") or [])
        if roles:
            r0 = roles[0]
            if kind == "output":
                return r0 in ("carry_phys", "carry_mag", "assassin", "bruiser")
            return r0 in ("tank", "support", "bruiser") or (r0 == "controller" and "tank" in roles)
        tags = self.pieces.get(hero, {}).get("tags") or []
        out = any(t in tags for t in ("射手", "法师", "刺客", "战士")) and not self._is_tank(hero) and not self._is_support(hero)
        if kind == "output":
            return out
        return not out            # defense → 坦克 / 辅助 / 没标签的前排

    # ---------- 限定使用者的牌（2026-09-30：临阵磨枪这类限定使用者的牌，统一按卡面判定能给谁用） ----------
    _TRIGGER_RE = re.compile(r"触发其(整备|登场|牺牲|退场|开团)效果")
    # 注意「【日落海】专属装备」**不是**穿戴限制：天赋「全域共鸣」原文"火鹰装备的【日落海】专属效果对穿戴它的任意英雄生效"
    # 说明谁都能穿、只是专属效果只对日落海生效（2026-10-09 一度当成限制加进来，卡牌审查指出后撤回）。
    # 给不给非日落海英雄穿是价值判断，由 equip_relations（restrict=faction 日落海 → 适配分低）管
    _FACTION_LIMIT_RE = re.compile(r"(?:指定\d*名|给|对|仅|可穿戴给)(?:已觉醒的)?【([^】]+)】英雄|仅(三分|河洛|逐鹿|日落海|大河)英雄"
                                   r"|(?!)(x)")
    _ONLY_HERO_RE = re.compile(r"仅([^\s，。、（）()]{1,5}?)穿戴时生效")

    _GROUP_RE = re.compile(r"(?:全体|你的|所有|场上的?)【([^】]+)】英雄")

    def group_card_ok(self, card: str) -> tuple[bool, str]:
        """作用于"全体/你的/所有【X】英雄"的牌（援护战术、膨化生命…）：场上至少 1 个 X 派系的人才打。
        2026-09-30：援护战术场上没有对应英雄就不用，不要因此卡死。"""
        text = self.pieces.get(card, {}).get("skill") or self._card_text(card) or ""
        m = self._GROUP_RE.search(text)
        if not m:
            return True, ""
        from agent.decision import routes as _R
        fac = _R.norm_banned(m.group(1))
        n = sum(1 for h in self._field if fac in (self.pieces.get(h, {}).get("synergies") or []))
        return (n > 0, "" if n else f"场上没有【{fac}】英雄")

    def card_target_ok(self, card: str, hero: str) -> tuple[bool, str]:
        """这张牌能不能（值不值得）给这个英雄用：(能否, 原因)。**只读卡面文字，不写死卡名**：
          · 派系：「指定1名【河洛】英雄」「仅三分英雄穿戴」「可穿戴给【河洛】英雄」→ 只给该派系；
          · 品阶：「1-3阶英雄」→ 只给 1~3 阶；
          · 觉醒：「已觉醒的…英雄」→ 只给觉醒过的；「觉醒进度+1」→ 只给没觉醒的；
          · 远近：「限定远程/近战英雄」；专属：「仅露娜穿戴时生效」→ 只给露娜；
          · 触发类：「触发其整备/登场/牺牲…效果」（临阵磨枪 / 唤醒）→ 只给技能里有这个关键词的英雄，给别人等于白扔。"""
        if not hero:
            return True, ""
        if hero == "古币机":
            return ("古币" in card), ("" if "古币" in card else "古币机只吃古币")
        if len((self.__dict__.get("_card_hero_fail") or {}).get((card, hero), ())) >= 2:
            return False, f"{card} 在 {hero} 身上跨两个回合都没生效"
        text = self.pieces.get(card, {}).get("skill") or self._card_text(card) or ""
        info = self.pieces.get(hero, {})
        mod = self.card_rule_mod(card)
        if mod:
            # **天赋改写了这张牌的规则**（多彩古币："古币只能对非【河洛】英雄主动使用：随机变成其他同阶非【河洛】英雄，等级额外+7"）：
            # 原卡面的派系限制作废、反过来只给非 X 派系；会变身的不给主C / 路线成员 / 觉醒过的 / 枫叶舞宿主（身份全丢）
            if mod[1] in (info.get("synergies") or []) or self._faction(hero) == mod[1]:
                return False, f"选了天赋「{mod[3]}」：{card} 只能给非【{mod[1]}】英雄"
            if mod[2] and (hero == self._carry_piece or hero in self._ever_awakened or hero == self._maple_bound
                           or hero in set(self.route_stage_members(self._round_seen) or ())):
                return False, f"选了天赋「{mod[3]}」：{card} 会把 {hero} 变成别的英雄 —— 主C / 路线成员 / 觉醒 / 枫叶舞宿主不能变"
            return True, ""
        m = re.search(r"指定场上\d*名(\d+)级以上", text)
        if m and (self._unit_lv.get(hero) is not None) and self._unit_lv[hero] < int(m.group(1)):
            return False, f"{card} 只能给 {m.group(1)} 级以上的英雄（{hero} {self._unit_lv[hero]} 级）"
        m = self.NEAREST_FAC_RE.search(text)
        if m and hero in self._field:
            fac2 = {"三分": "三分之地", "大河": "大河流域"}.get(m.group(2), m.group(2))
            if not self._nearest_same_fac(hero, fac2)[0]:
                return False, f"{card} 要和最近的【{m.group(2)}】英雄换 / 看齐，场上没有第二个【{m.group(2)}】英雄"
        if self.DESTROY_EQUIP_RE.search(text) and hero in self._field:
            # 「摧毁其所有装备」：目标身上一件都没有就白打（卡牌审查第 17 条）。件数看穿戴账本 + 见过的最多件数
            k = max(len(self._worn.get(hero, ()) or ()),
                    self._equip_seen.get(cell_key(tuple(self._field[hero])), 0) or 0)
            if k < 1:
                return False, f"{card} 要摧毁装备，{hero} 身上没有装备"
        if re.search(r"变为", text) and (hero == self._carry_piece
                                          or hero in set(self.route_stage_members(self._round_seen) or ())):
            # 变身牌（化铁为金/霜寒进化/升华…）绝不点主C / 路线成员（2026-09-30 第 42 局：玄策被变成海月，主C 直接没了）
            return False, f"{card} 会把 {hero} 变成别的英雄 —— 主C / 路线成员不能变"
        m = self._TRIGGER_RE.search(text)
        if m:
            hs = " | ".join([info.get("skill") or "", info.get("awaken") or "", self._card_text(hero) or ""])
            # 目标**自己要有**「整备：…」这条效果（关键词后跟冒号）。只是提到这个词的不算 ——
            # 朵莉亚"每当你的英雄触发整备效果时…"（2026-09-30 第 44 局 Laya 把临阵磨枪给了她，游戏拒收）
            if not re.search(rf"{m.group(1)}\s*[：:]", hs):
                return False, f"{card} 触发{m.group(1)}效果，{hero} 自己没有{m.group(1)}效果"
        m = self._FACTION_LIMIT_RE.search(text)
        if m:
            from agent.decision import routes as _R
            fac = _R.norm_banned(m.group(1) or m.group(2) or m.group(3))
            if fac not in (info.get("synergies") or []):
                return False, f"{card} 只能给【{fac}】英雄"
        # 只管"指定1名英雄 → 拿它的同阵营牌"这类（援军/求贤/全军列阵…）；「或」二选一的（二象坍缩）和装备光环（鼓舞之戒）不拦
        # （2026-09-30 卡牌关系图谱审计发现原来过严）
        if (re.search(r"指定\d*名英雄[^。]*同阵营", text) and "或" not in text
                and not [f for f in (info.get("synergies") or []) if f not in ("无阵营", "通用", "")]):
            # 「随机获得4张同阵营的英雄牌」（援军）给无阵营的人游戏不收（2026-09-30 第 15 局：给公孙离拖了 8 次）
            return False, f"{card} 要同阵营，{hero} 没有阵营"
        # 品阶区间只认"限定使用对象"的写法：指定1名1-3阶 / 指定移除1名1~2阶 / 仅1-3阶英雄可穿戴 / 1-3阶英雄限定
        # （原来只认"1-3阶"字面，限制性进化的"1~2阶"被当成谁都能用 —— 2026-09-30 图谱审计）
        m = (re.search(r"(?:指定(?:移除)?\d*名|仅)(\d)[-~～](\d)阶", text)
             or re.search(r"(\d)[-~～](\d)阶英雄限定", text))
        if m and not (int(m.group(1)) <= (info.get("quality") or 0) <= int(m.group(2))):
            return False, f"{card} 只能给 {m.group(1)}~{m.group(2)} 阶英雄"
        if "已觉醒" in text and re.search(r"指定\d*名已觉醒", text) and hero not in self._ever_awakened:
            return False, f"{card} 只能给已觉醒的英雄"
        # 只有"加觉醒进度"的牌（弦音/妙音/寒冰觉醒…）对已觉醒的没用；体面离场"0/1/2/3觉醒进度时获得3/5/7/9能量"
        # 恰恰是已觉醒的换得最多，不能拦（2026-09-30 图谱审计）
        if re.search(r"觉醒进度(?:额外)?\+", text) and hero in self._ever_awakened:
            return False, f"{card} 给已觉醒的 {hero} 没用"
        m = self._ONLY_HERO_RE.search(text)
        if m and m.group(1) != hero:
            return False, f"{card} 只对 {m.group(1)} 生效"
        if "限定远程" in text or "限定近战" in text:
            rng = info.get("range", 0) or 0
            if not rng or (rng < 2 if "限定远程" in text else rng > 1):
                return False, f"{card} {'限定远程' if '限定远程' in text else '限定近战'}"
        return True, ""

    def _equip_allowed(self, card: str, hero: str) -> bool:
        """装备卡面上的**穿戴限制**：「限定远程英雄」「限定近战英雄」。

        2026-09-23：日渊（"唯一，限定远程英雄"）往近战的曜身上拖了 5 次，每次都举起来、补点目标、
        不生效 —— 游戏根本不让穿。决策层从来没读过这一句。英雄的远近看卡牌库里的攻击距离（近战=1）。
        """
        skill = self.pieces.get(card, {}).get("skill", "")
        if self.pieces.get(card, {}).get("category") == "装备牌" and self._equip_void(card):
            return False            # 效果要靠这局不会出现的单位（小鹿的庇护 → 瑶妹的小鹿）：谁穿都白穿，还占装备位
        if hero:
            from . import hero_traits as _ht
            if _ht.item_mismatch(card, [hero]):
                return False        # 法术装不穿给物理英雄、物理装不穿给法术英雄（2026-09-26：噬神之书）
        # **同名装备一人只能穿一件**（卡面"唯一"）。2026-09-23 第八局：四选一刚给猪八戒铸出
        # 「不死鸟之眼」，手里另一张「不死鸟之眼」又往他身上拖，游戏拒收。
        if hero and card in self._worn.get(hero, ()):
            return False
        if (hero and hero in _ht._lr("no_onhit_carries") and card not in self.PAIR_EQUIPS     # 怒莲之刺例外（10-20 晚确认可以带：伤害转给穿银莲的坦克）
                and re.search(r"(提升|[+＋])\s*\d+%?\s*(攻速|攻击速度)|(攻速|攻击速度)\s*[+＋]\s*\d", self._card_text(card) or skill or "")):
            return False        # 守约 / 沈梦溪 / 诸葛亮不穿加攻速的装备（10-20 确认；第 847 局成对装备把怒莲之刺 +100% 攻速给了守约）
        if hero and card in _ht.RANGED_ONLY and _ht.is_melee(hero):
            return False        # 自然法杖只给远程（2026-10-21 第 884 局；第 886 局 R17 用牌路径穿给了近战钟馗）
        if hero and hero == self._carry_piece and _ht.onhit_mismatch(card, hero, _ht.live_onhit(hero)):
            return False        # 普攻叠层装（寒霜侵袭 / 金色圣剑）不给技能型主C（第 886 局 R4：曹操卖掉退回的寒霜侵袭直接拖给了守约）
        if hero and card in _ht._lr("ranged_fast_only") and (_ht.is_melee(hero) or not _ht.attack_speed_carry_live(hero)):
            return False        # 分身镜：只适合远程高攻速英雄（2026-10-20 晚第 846 局：分身镜穿给了近战亚连）
        if hero and hero == self._carry_piece and card in _ht.SUNSET_ONLY_ITEMS and not _ht._is_sunset(hero):
            return False        # 星源装甲 / 狩枪给非日落海主C：唯一被动无效、剩下的属性太差（2026-10-20 确认）
        if card in self.LORD_USELESS_ITEMS:
            return False        # 阿离用不上的特殊套路装备：谁都不穿（2026-10-21：阿科米亚之子是给其他棋手用的）
        if hero and hero == self._carry_piece and card in self.NOT_FOR_CARRY_LIVE:
            return False        # 不加主C 输出属性的装备不占主C 位（10-20 晚装备复审：静谧之铃对沈梦溪 / 捕能手套只攒经济 / 唤鹰骨笛只召唤）
        if hero and hero == self._carry_piece:
            # 10-20 第 836 局（大棒给守约铸出自然法杖）：基础装备给主C 时，四个成品里最好的那件要对主C 合适，否则不给主C（给别人）
            _rec = self.__dict__.get("_recipes_cache")
            if _rec is None:
                try:
                    import json as _j
                    _rec = _j.loads((Path(__file__).resolve().parents[2] / "data" / "equip_recipes.json").read_text(encoding="utf-8")).get("recipes", {})
                except Exception:                                   # noqa: BLE001
                    _rec = {}
                self._recipes_cache = _rec
            if card in _rec and not (card in self.CARRY_LATE_DEF_BASE and self.carry_late_def_ok(None)):
                from . import hero_traits as _ht2
                # 同名成品一人只能穿一件 → 已穿的那件不算（第 838 局：守约已有凝渊，第二把速击之枪只能铸出破晓 / 影牙这类攻速 / 普攻装）
                _left = [x for x in _rec[card] if x not in set(self._worn.get(hero, ()))] or _rec[card]
                # 线上覆盖的成品伤害类型（hero_traits 表把制裁之剑记成"通用输出"，改表会动旧题面 → 线上先在这里覆盖，题面 f9za 再改）：
                # 2026-10-20 第 844 局：制裁之剑给了法术主C 沈梦溪 —— 第二把风暴巨剑，纯净苍穹已穿，只剩无尽 / 制裁 / 逐风，全是物理
                _dt = _ht2.dmg_type(hero)
                _left = [x for x in _left if not (self.ITEM_DMG_LIVE.get(x) and _dt and _dt != "mixed"
                                                  and self.ITEM_DMG_LIVE[x] != _dt)]
                if not _left:
                    return False
                _best = _ht2.carry_order(_left, hero, onhit=_ht2.live_onhit(hero))[0]
                if not _ht2.live_fits(_best, hero)[0]:
                    return False
        if (hero and "固定0.4的攻击速度" in skill and hero not in self.SPEED_LOCK_CARRIES
                and not self._is_tank(hero)):
            return False                    # 机器头（攻速锁 0.4）：法力靠普攻回，只有花木兰 / 嬴政机制特殊能用；坦克 / 战士照穿（2026-10-19）
        if hero and not self.card_target_ok(card, hero)[0]:
            return False                    # 卡面限定的使用者（派系 / 品阶 / 觉醒 / 远近 / 专属 / 触发类）
        _ex = EQUIP_FACTION_EXCLUDE.get(card)
        if hero and _ex and _ex in (self.pieces.get(hero, {}).get("synergies") or []):
            return False                    # 卡面没写、实测确认的阵营排除（阿科米亚之子：日落海英雄穿不了，游戏拒收）
        if hero and _ht.profile_says(card, hero) == "avoid":
            return False                    # 装备分类：这件在该英雄的"不要给"名单里（data/equip_profiles.json）
        if hero and _ht.USE_HERO_ITEM_PREF and _ht.LIVE_ITEM_LIMITS.get(card) == "assassin" and not _ht.is_assassin(hero):
            return False                    # 红魔只给刺客（独处 / 挨打叠增伤；后排主C 旁边有辅助、吃不到）
        # **派系限制**：「指定1名【河洛】英雄等级+2」（古币）—— 只能给那个派系的人。
        # 2026-09-23：古币一直没用上 —— 拖给了不是河洛的诸葛亮，失败两次被搁置，
        # 而它是「转瞬」牌，回合结束就没了。
        m = None if self.card_rule_mod(card) else re.search(r"指定\d*名【([^】]+)】英雄", skill)   # 天赋改写过的（多彩古币）由 card_target_ok 管
        if m:
            if not hero or m.group(1) not in self.pieces.get(hero, {}).get("synergies", []):
                return False
        # **无法拆卸 + 永久成长的装备只给留到最后的人**（盛华："唯一，无法拆卸。每次战斗结束后永久增加200点生命值"）。
        # 2026-09-24 第 14 局复盘：盛华穿在后来被卖掉的过渡件身上，攒了几回合的血量跟着人一起没了。
        # 卡面若没进 pieces 就读卡牌库原文（装备的 skill 字段常为空）。
        if not skill:
            skill = self._card_text(card)
        if hero and "无法拆卸" in skill and "永久" in skill:
            if not self._is_keeper(hero) or self._transitional_carry(hero, self._round_seen):
                return False
        if "限定远程" not in skill and "限定近战" not in skill:
            return True
        # 远近程直接按卡牌库 properties.attackDistance（hero_traits.is_melee）：self.pieces 的 range 只在传了 cards_db_path 时才有（线上有），离线工具常不传
        from . import hero_traits as _htr
        if not hero or not (_htr._cards().get(hero) or {}).get("properties"):
            return False            # 不知道是谁/查不到距离：有限制的装备宁可先留着
        return (not _htr.is_melee(hero)) if "限定远程" in skill else _htr.is_melee(hero)

    # 名单绑定棋手（data/lords/阿离/lord_rules.json lord_useless_items / not_for_carry_live；下面是兜底值）
    # 10-22：青莲剑谱移出（拿到可以给其他近战穿，比如凯；限定近战按卡面走）
    LORD_USELESS_ITEMS = _LordRule("lord_useless_items", ("阿科米亚之子", "驱雷", "月影之力"))     # 驱雷敌我不分会劈自己主C；月影之力只露娜生效（特殊装备复审 v2，10-21）
    NOT_FOR_CARRY_LIVE = _LordRule("not_for_carry_live", {"静谧之铃", "捕能手套", "唤鹰骨笛", "阿科米亚之子"})     # 阿科米亚之子：只是参与核心分配 + 小回蓝，不加主C 输出（第 850 局穿给守约占位）     # 线上：主C 不穿（data/review/equips_carry.json）
    ITEM_DMG_LIVE = {"制裁之剑": "phys"}     # 线上覆盖：hero_traits 记成通用输出，实为物理装（第 844 局沈梦溪）
    SPARE_EQUIP_TARGET = _LordRule("spare_equip_target", {})     # 路线主C 阵营 → 主C 满装后多余装备优先给谁（data/lords/阿离/lord_rules.json；2026-10-22）

    def spare_equip_targets(self) -> tuple:
        """当前路线（没锁看当前主C）阵营对应的"多余装备优先给谁"名单（棋手配置 spare_equip_target）。"""
        rc = self.route_carry() or self._carry_piece or ""
        return tuple((self.SPARE_EQUIP_TARGET or {}).get(self._faction(rc)) or ()) if rc else ()

    EQUIP_SPARE_FIT = 1.0        # 主C 满装后，剩余装备给"相对最合适"的人（2026-10-09：冗余装备也要找个合适的人，很多有额外效果，压在手上就是浪费；原 4.5 时铸梦·逐风 / 魔律之刃一直压在手里）。avoid / 穿戴限制 / 消耗件仍排除
    SHOP_OPEN_SAFETY = 10    # 每回合开店次数的**防 bug 保险**（不是决策条件；正常一回合开 1~3 次）
    CATCHUP_MIN_LV = 30      # 临时主C 至少这么多级才等路线主C 追（第 279/283 局）
    CATCHUP_UNTIL_ROUND = 9  # 最晚到这回合照常换绑
    CARRY_GEAR_FIT = 5      # 主C 没满装时，适配分 ≥ 这个就先给主C（2026-10-05 第 254 局）
    CARRY_GEAR_FALLBACK = 3  # 场上没人对口时，主C 没满装、适配分 ≥ 这个也给主C（不压在手里；第 266 局）

    def _equip_target(self, state: GameState, card: str = ""):
        """装备该给谁：返回 (**实际收件人**, 落点)，或 None = 先留着。

        2026-09-23：装备穿给了弈星，随后四选一又失败。日志写的是
        `目标=公孙离`，落点却是 (1027,572) —— 记账里那是少司缘的格子，画面上站的是弈星。
        两个毛病叠在一起：
          1. 公孙离那格**这一帧没看见徽章**（徽章检测还有 ~1/3 漏检），就被当成"没人"，
             转投到别的格子。"没看见"是弱证据（观测不能单独否定记录），
             真拖不进去时 `_cell_dead` 在失败 2 次后本来就会接手。
          2. 转投之后**名字还写着主力**，于是日志、打分（按公孙离的定位选四选一）全是错的。
        现在：
          - 主力在场上、还装得下 -> 就给主力，**看没看见都一样**
          - 主力还在手里（刚买 / 觉醒回手）-> **先留着**，等它上场，别塞给路人
          - 否则按分值挑一个装得下的非消耗件，**名字报它自己的**
        """
        if card and self.haiyue_save_equip(state, card):
            return None          # 伞海月：装备存着给海月（干将有自带的双剑）
        from agent.decision import interim_rules as _IRq
        if (_IRq.on("equip_hold_for_real_carry") and self.is_temp_carry(self._carry_piece)
                and len(state.hand_slots or []) < 8):
            # 2026-10-19：临时主C 不先穿装备（真主C 到场时可能缺装备）——临时主C 期间装备留在手里（手牌快满时照常穿）
            if self._skip_logged.get(("equip_hold", card)) != self._round_seen:
                self._skip_logged[("equip_hold", card)] = self._round_seen
                logger.info("现任主C %s 是临时主C → %s 先留着给真主C（临时规则 equip_hold_for_real_carry）", self._carry_piece, card)
            return None
        counts = state.equip_counts or {}

        def room(pos):
            return (pos is not None
                    and not self._cell_dead(pos)
                    and not self._covered_by_shop(state, pos)
                    and self._equips_at(state, pos) < meta.MAX_EQUIPS)

        carry = self._carry_piece
        pos = self._field.get(carry) if carry else None
        hand_names = {sl["name"] for sl in state.hand_slots or []}
        from agent.decision import hero_traits as _ht0

        def carry_room(p):
            """主C 那格还能不能穿：**基础装备**不信"看到过的最多件数"（画面会多读一个图标，2026-10-06 第 322 局：
            虞姬实际 2 件、有几帧读成 3 → 一直当满，对口的速击之枪给了坦克）。看本帧读数 + 穿戴账本 + 被拒次数；
            真满了游戏拒收、不弹四选一，note_equip_rejected 两次后照旧当满。成品装备没有弹窗回执，照旧用 room()。"""
            if not _ht0.base_kind(card) and (p is None or self.equip_now(state, cell_key(tuple(p))) is None):
                return room(p)
            # 成品装备这一帧读得到件数时同样不信"看到过的最多件数"（2026-10-13 第 617 局：马可波罗实际 2 件，
            # 第 6 回合 84 帧里 29 帧读成 3 → 一直当满，天赋送的捕能手套压在手里，只能手动穿上）。
            # 真满了游戏拒收、牌留在手里 → 拖拽回执判失败，"试够次数就停"兜底
            if p is None or self._cell_dead(p) or self._covered_by_shop(state, p):
                return False
            k = cell_key(tuple(p))
            if self._equip_rejects.get(k, 0) >= 2:
                return False
            now = self.equip_now(state, k)
            if now is None:
                return room(p)                 # 读不到就别当 0 件，退回"见过的最多件数"
            return max(now, len(self._worn.get(carry, ()))) < meta.MAX_EQUIPS
        if card and "无法拆卸" in (self._card_text(card) or ""):
            # **装上拆不下来的神装**（聚灵法杖 / 百锻刀 / 盛华…）只给主C；主C 不对口（伤害类型 / 限制）就给能当主C 的对口输出位，
            # 都没有就留着（2026-10-14 第 626 局：聚灵法杖给了辅助明世隐——这是法师神装，装上就拆不下来）
            from agent.decision import hero_traits as _htl
            def _ok(n):
                p0 = self._field.get(n)
                return (p0 is not None and room(p0) and self._equip_allowed(card, n)
                        and not (_htl.item_dmg(card) and _htl.item_mismatch(card, [n])))
            if carry and _ok(carry):
                return carry, self._on_unit(state, self._field[carry])
            alt = [n for n in self._field if n != carry and _ok(n) and self._carry_eligible(n)
                   and not self._is_support(n) and not self._is_tank(n)]
            if alt:
                n = max(alt, key=lambda x: (self.unit_value(x), self._unit_lv.get(x) or 0))
                return n, self._on_unit(state, self._field[n])
            if self._skip_logged.get(("locked_equip", card)) != self._round_seen:
                self._skip_logged[("locked_equip", card)] = self._round_seen
                logger.info("%s 装上拆不下来，主C %s 不对口、场上也没有对口的主C 人选 → 先留着", card, carry or "未定")
            return None
        if self.growth_equip(card):
            # **复利成长装备只给主C**（学识宝石：每回合开始等级 +8%；2026-10-08 第 417 局：这类特殊装备要尽快分给主C，
            # 当时第 5 回合主C 还没定，随手穿给了孙尚香）。主C 不在场 / 装不下就先留着
            if carry and pos is not None and carry_room(pos):
                return carry, pos
            logger.info("%s 是复利成长装备，只给主C（主C %s 现在不在场或装不下），先留着", card, carry or "未定")
            return None
        for buddy, wanted in self.SHARE_EQUIP.items():
            bpos = self._field.get(buddy)
            # 瑶"开团：自身的**首件普通装备**分享给随机 3 名友方英雄"—— 一件装备 4 个人用，优先级高于主C。
            # 原来只认成品「破茧之衣」，而手里多是基础装备 → 实战几乎从不触发（2026-09-29 补上）。
            # 现在任何普通装备都行（不看定位：分享给的是别人），手里有 破茧之衣 / 破碎圣杯 时先等它。
            if (bpos is not None and room(bpos) and self._equips_at(state, bpos) == 0
                    and self._is_normal_equip(card) and self._equip_allowed(card, buddy)
                    and (card in wanted or not (hand_names & set(wanted)))):
                logger.info("%s 给 %s 当首件普通装备（开团分享给随机 3 名友军）", card, buddy)
                return (buddy, self._on_unit(state, bpos))
        # **主C 先满装**（2026-10-01：主C 的等级和装备都重要，2~3 阶装备很强力，优先给主C）：
        # 对主C 对口（适配分 ≥6）、主C 还没满 3 件 → 先给主C；2~3 阶装备（四选一成品 / 星源等特殊装备）对口就绝不分给别人。
        # 云中君"每件装备开团 +1 永久"的全队铺装只用主C 不要的装备，或主C 已满。
        from agent.decision import hero_traits as _ht
        _fit_c = _ht.item_fit_score(card, carry) if (carry and card) else None
        if (pos is not None and carry_room(pos) and self._equip_allowed(card, carry)
                and _fit_c is not None and _fit_c >= self.CARRY_GEAR_FIT):
            # 2026-10-05：主C 经常漏穿装备（上一局要手动补穿）。
            # 原门槛 ≥6（或 2~3 阶 ≥5）：基础装备对虞姬 5.6 → 大棒/圣者法典被云中君"全队铺装"分走，
            # 虞姬整局 1 件（第 254 局）。现在主C 没满 3 件、对口（≥5，基础装备按四个成品里最合适的算）就先给主C。
            return (carry, self._on_unit(state, pos))
        if ((card in self.CARRY_LATE_DEF or card in self.CARRY_LATE_DEF_BASE) and pos is not None and carry_room(pos)
                and self._equip_allowed(card, carry) and (card in self.CARRY_ANY_DEF or self.carry_late_def_ok(state))):
            logger.info("主C %s 两件输出装、后期第三格还空 → %s 给主C（临时规则 carry_late_defense）", carry, card)
            return (carry, self._on_unit(state, pos))
        if "云中君" in self._field and (pos is None or self._equips_at(state, pos) >= 1):
            # 云中君开团"每件装备 +1 永久"对**全队**生效 → 主C 有 1 件后，先给还没装备的人各穿 1 件。
            # **但装备要对口**（2026-09-26：出现过把输出装备给坦克）：输出型基础装备只给输出位，防御型只给坦克/辅助
            bare = sorted(((self.unit_value(n), n, p) for n, p in self._field.items()
                           if n != carry and room(p) and self._equips_at(state, p) == 0
                           and not self._is_disposable(n) and self._equip_allowed(card, n)
                           and self._equip_fits_role(card, n) and not self._no_gear(n)
                           and n not in self.NO_EQUIP_UNITS and n not in self.SHARE_EQUIP), reverse=True)
            if bare:
                _v, name, p = bare[0]
                logger.info("云中君在场：%s 先给还没装备的 %s（每件装备开团 +1 永久）", card, name)
                return (name, self._on_unit(state, p))
        _defense = ((_ht.EQUIP_CLASS.get(card) or ("",))[0] == "tank") or _ht.base_kind(card) == "defense"
        if room(pos) and self._equip_allowed(card, carry) and (
                _fit_c is None or (_fit_c >= self.CARRY_GEAR_FALLBACK and not _defense
                                   and _ht.profile_says(card, carry) != "avoid")):
            # 主C 有空位的老兜底，现在要求：查不到适配分，或适配 ≥CARRY_GEAR_FALLBACK 且不是防御装、画像不是 avoid。原来不看适配分：只要主C 有空位，任何装备都先给主C ——
            # 下面"按适配分找对口的人 / 不对口就留着 / 主C 适配 ≥3 才给"全被跳过（2026-10-13 第 614 局：
            # 卖曜退回的反伤重甲对马可波罗 2.8 分、对主坦项羽 8.0 分，照样把反伤甲穿给了主C 马可波罗）
            return (carry, self._on_unit(state, pos))
        if (pos is None and carry and self._equip_allowed(card, carry)
                and any(sl["name"] == carry for sl in state.hand_slots or [])):
            if self._skip_logged.get(("equip_hold", carry)) != self._round_seen:
                self._skip_logged[("equip_hold", carry)] = self._round_seen
                logger.info("主力 %s 还在手里，装备先留着等它上场（不塞给别人）", carry)
            return None
        seen = set(self._anchor_cells(state))
        if not self._field and not self._equip_allowed(card, ""):
            return None                     # 有穿戴限制的装备，不知道谁站哪儿就先留着
        if not self._field:
            # **账本空了但画面上有人**（对账/觉醒判定/卖人失败都会清账）：
            # 原来 `if not self._field: return None` 把装备挡了整整两个回合（场景45）。
            # 不知道那格站的是谁 -> 名字给空串，打分退回主力的定位。
            cell = next((c for c in seen if room(c)), None)
            return ("", self._on_unit(state, cell)) if cell else None
        if pos is not None and not carry_room(pos) and card:
            # **主C 已满装 + 棋手配置 spare_equip_target**（阿离：河洛路线 → 铠；2026-10-22：河洛阵容里主C 满装后
            # 多余的选辅助类型、给铠，注意铠是近战）：基础装备（四选一走非主C 辅助性分支 util:）或辅助类成品（utility_rank < 3）
            # 先给配置里的人；输出类成品照旧走下面的定位对口。穿戴合法性照卡面（_equip_allowed：限定远程 / 自然法杖不给近战、同名唯一、物法错配）
            from . import hero_traits as _hts
            if _hts.base_kind(card) or _hts.utility_rank(card) < 3:
                for _sn in self.spare_equip_targets():
                    _sp = self._field.get(_sn)
                    if (_sn != carry and _sp is not None and room(_sp) and self._equip_allowed(card, _sn)
                            and not self._no_gear(_sn) and _hts.profile_says(card, _sn) != "avoid"):
                        logger.info("主C %s 已满装，%s 给 %s（棋手配置 spare_equip_target：%s 路线多余装备给他）",
                                    carry, card, _sn, self._faction(self.route_carry() or carry))
                        return (_sn, self._on_unit(state, _sp))
        # 先找定位对口的（输出装给输出位、防御装给坦克）；一个都没有再退回"谁装得下给谁"（总比压在手里强）
        fits_any = any(self._equip_fits_role(card, n) and n != carry and room(p) and not self._is_disposable(n)
                       and self._equip_allowed(card, n) for n, p in self._field.items())
        from . import hero_traits as _ht
        cands = sorted(((_ht.item_fit_score(card, n, self.hero_role(n).get("combat_role") or []) or 0,
                         not self._is_support(n), tuple(p) in seen, self.unit_value(n), n, p)
                        for n, p in self._field.items()
                        if n != carry and not self._is_disposable(n) and room(p)
                        and self._equip_allowed(card, n) and self._equip_fits_role(card, n)
                        and not self._no_gear(n)                    # 2026-10-19：辅助（瑶除外）不给装备
                        and n not in self.NO_EQUIP_UNITS            # 太乙真人不给装备（官方 78% 不给）
                        and not (n in self.SHARE_EQUIP and (self._equips_at(state, p) >= 1
                                                            or not self._is_normal_equip(card)))),   # 瑶只要一件普通装备
                       reverse=True)
        if cands:
            _fit, _ns, _seen, _v, name, p = cands[0]
            if pos is not None and not room(pos):
                logger.info("%s 身上装不下了（%d 件 / 上限 %d），改给 %s",
                            carry or "主力", self._equips_at(state, pos), meta.MAX_EQUIPS, name)
            elif pos is not None:
                logger.info("%s 不适合主C %s（伤害类型 / 穿戴限制 / 适配分 %s），改给 %s",
                            card, carry, _fit_c, name)
                # 第 1035 局 R6：守护者之铠给了张良（8 分）而不是 0 件的牛魔（10 分），离线重放选牛魔 → 记下每人被排除的原因
                _why = {}
                for _n, _p in self._field.items():
                    if _n == carry or any(c[4] == _n for c in cands):
                        continue
                    _why[_n] = [t for t, bad in (
                        ("消耗件", self._is_disposable(_n)), ("装不下", not room(_p)),
                        ("件数", self._equips_at(state, _p) if _p is not None else None),
                        ("限制", not self._equip_allowed(card, _n)), ("不对口", not self._equip_fits_role(card, _n)),
                        ("辅助", self._no_gear(_n)), ("不给装", _n in self.NO_EQUIP_UNITS)) if bad]
                logger.info("  装备候选 %s；排除 %s", [(c[4], c[0]) for c in cands[:4]], _why)
            else:
                if (not carry and _IRq.on("equip_hold_for_real_carry") and len(state.hand_slots or []) < 8
                        and (self.pieces.get(name, {}).get("quality") or 0) == 1):
                    # 第 850 局 R1：主C 还空着时圣者法典给了 1 阶孙尚香（下一步才被定成临时主C，上面的暂留没拦到）
                    if self._skip_logged.get(("equip_hold", card)) != self._round_seen:
                        self._skip_logged[("equip_hold", card)] = self._round_seen
                        logger.info("主C 未定、只剩 1 阶的 %s 可穿 → %s 先留着给真主C（临时规则 equip_hold_for_real_carry）", name, card)
                    return None
                logger.info("主力 %s 不在场上，装备改给 %s", carry or "?", name)
            return (name, self._on_unit(state, p))
        # 原来"一个对口的都没有就谁装得下给谁"——第 43 局把防御装守护者之铠给了输出孙膑（输出拿到了坦克装）。
        # 现在不对口就留着（等对口的人上场，或后期卖掉）。
        # **但主C 没满装时，不冲突的装备先给主C**（2026-10-05 第 266 局：守约 0 件，手里的魂祭绝刃 对他 3.9 分 ——
        # 攻速 +50% 吸血 +30%，只因"开战扣半血"扣了分 —— 整整 3 回合压在手里，主C 漏穿装备）。
        # 门槛：适配分 ≥ CARRY_GEAR_FALLBACK、画像不是 avoid（伤害类型错配 / 限定条件不符的已经被 _equip_allowed 和低分挡住）。
        if (pos is not None and carry_room(pos) and self._equip_allowed(card, carry) and _fit_c is not None
                and _fit_c >= self.CARRY_GEAR_FALLBACK and _ht.profile_says(card, carry) != "avoid"):
            logger.info("场上没人更适合穿 %s，主C %s 还没满装（适配 %.1f）→ 给主C", card, carry, _fit_c)
            return (carry, self._on_unit(state, pos))
        if pos is not None and not carry_room(pos):
            # **主C 已满装：手里的装备别浪费**（2026-10-08：主C 满装后手上的装备给其他符合定位的英雄穿上，
            # 多加点辅助效果）。门槛放宽到适配分 ≥ EQUIP_SPARE_FIT（仍排除画像 avoid、伤害类型 / 穿戴限制不符、消耗件、太乙）
            sc = self.sub_carry()
            _out = _ht.base_kind(card) == "output" or bool(_ht.item_dmg(card))
            if sc and _out:
                sp = self._field.get(sc)
                if (sp is not None and room(sp) and self._equip_allowed(card, sc)
                        and not _ht.item_mismatch(card, [sc]) and _ht.profile_says(card, sc) != "avoid"):
                    logger.info("主C %s 已满装，输出装 %s 给副C %s（40 级输出 %.0f）", carry, card, sc, self._dps40(sc))
                    return (sc, self._on_unit(state, sp))
            spare = sorted(((_ht.item_fit_score(card, n, self.hero_role(n).get("combat_role") or []) or 0,
                             self.unit_value(n), n, p)
                            for n, p in self._field.items()
                            if n != carry and not self._is_disposable(n) and room(p)
                            and self._equip_allowed(card, n) and _ht.profile_says(card, n) != "avoid"
                            and not self._no_gear(n)
                            and n not in self.NO_EQUIP_UNITS
                            and not (n in self.SHARE_EQUIP and (self._equips_at(state, p) >= 1
                                                                or not self._is_normal_equip(card)))),
                           reverse=True)
            # 基础装备原来要求 ≥4.5（"留给以后的人"），结果主C / 主T 满装后大棒一路压在手里（2026-10-10 第 510 局：
            # 主C 和主T 都满装时，剩余装备也要找个合适的人装上）。现在和成品一样给相对最合适的人；
            # 四选一时按穿的人选对口成品（item_fit_score 对基础装备取四个成品里最合适的那件），所以不会"输出装给坦克"。
            _thr = self.EQUIP_SPARE_FIT
            _bk = _ht.base_kind(card)
            _cls = ((_ht.EQUIP_CLASS.get(card) or ("",))[0] or "")
            if _bk == "output" or _cls.startswith("dps") or _ht.item_dmg(card):
                # 输出装（基础 + 成品）不给坦克 / 辅助（09-26 出现过输出装给坦克；2026-10-13 第 623 局：成品无尽巨剑给了苏烈，适配 2.2）
                # 2026-10-19 调整：主C 满装后输出装可以给坦克 / 战士（_is_tank 含 bruiser），只不给辅助
                spare = [x for x in spare if not self._is_support(x[2])]
                _team = self._route_team()
                if _team and any(x[2] in _team for x in spare):
                    spare = [x for x in spare if x[2] in _team]                 # 核心阵容里的人优先（定位都按核心阵容算）
            elif _bk == "defense":
                spare = [x for x in spare if self._is_tank(x[2])]          # 防御装给坦克 / 战士（2026-10-19：辅助不给）
            if spare and spare[0][0] >= _thr:
                _f, _v, name, p = spare[0]
                logger.info("主C %s 已满装，%s 给 %s（适配 %.1f），不压在手里", carry, card, name, _f)
                return (name, self._on_unit(state, p))
            # 2026-10-19：装备不能卖，主C 满了就给其他人 —— 对口的人里适配分不到门槛也给（输出装仍不给坦克 / 辅助，见上面的 spare 过滤）
            if spare:
                _f, _v, name, p = spare[0]
                logger.info("主C %s 已满装，%s 给 %s（适配 %.1f，低于门槛也给：装备不能卖，别压手里）", carry, card, name, _f)
                return (name, self._on_unit(state, p))
        if self._skip_logged.get(("equip_nofit", card)) != self._round_seen:     # 每回合只报一次（第 48 局一局刷了 456 次）
            self._skip_logged[("equip_nofit", card)] = self._round_seen
            logger.info("场上没有定位对口、装得下的人穿 %s，先留着", card)
        return None

    def _carry_pos(self, state: GameState | None = None) -> tuple[int, int]:
        """装备/效果牌该拖到哪儿：**以 `_field` 里的实际位置为准**。

        `self._carry` 是"打算把主力放哪"的记录，会过期：
        2026-09-22 第三局，曜 第一次拖 (660,420) 失败、最终落在 (780,420)，
        但 `_carry` 还停在 (660,420) —— 于是「风暴巨剑」被拖到一个**空格子**上。
        `_field` 是拖拽回执坐实过的位置，比意图可靠。

        **主力在场上就用主力的格子，这一帧看没看见徽章都一样**（2026-09-23 改）。
        原来"画面上那格没看见人"就转投别人，而徽章检测有 ~1/3 漏检 ——
        实战里把装备送到了记账已过期的格子上（站着的是弈星）。只有拖失败够多次
        （`_cell_dead`）才换格子，那是直接证据。
        """
        seen = set(self._anchor_cells(state)) if state is not None else set()
        pos = self._field.get(self._carry_piece)
        if pos and not self._cell_dead(pos):
            return pos
        # 主力格已废 / 主力不在场上：挑一个活着的**非消耗件**，看得见的优先、分值高的优先
        cands = sorted(((tuple(p) in seen, self.unit_value(n), p)
                        for n, p in self._field.items()
                        if not self._cell_dead(p) and not self._is_disposable(n)),
                       reverse=True)
        if cands:
            alt = cands[0][2]
            if pos:
                logger.info("主力记账位 %s 本局已拖失败多次，改投 %s", pos, alt)
            return alt
        if self._field:
            return next(iter(self._field.values()))
        anchors = self._anchor_cells(state) if state is not None else []
        return anchors[0] if anchors else self._carry

    # ---------- 拍卖 ----------

    def _entry_effect_kind(self, name: str) -> Optional[str]:
        """这张英雄牌吃不吃"上场白嫖"：返回 "登场" / "整备" / None。

        注：**觉醒的触发条件不止"凑够 4 张同名"**——有些效果卡直接给"觉醒阶段+1"，
        所以不能靠数手上有几张来预测觉醒（2026-09-22）。
        觉醒只能靠"合成拖拽连续失败"这类**信号**来推断，见 decide_all 里的推断逻辑。

        时机以官方关键词词典（data/keywords.json）为准，直接决定什么时候能卖：
          登场：**使用此卡牌后**立即触发 → 当回合就能卖
          整备：**回合开始时**触发 → 得留到下一回合开始后才卖
        （我一度以为整备是"开战之后"，实为"回合开始时"，卖出时机正好一致。）

        另外两个相关关键词：
          退场：场上英雄**被出售时**触发效果 → 这类英雄卖掉本身就有收益
          转瞬：回合结束自动销毁，**出售不给能量** → 别浪费操作去卖
        """
        info = self.pieces.get(name, {})
        text = info.get("skill", "") + info.get("awaken", "")
        if "登场" in text:
            return "登场"
        if "整备" in text:
            return "整备"
        return None

    YIZHI_MIN_LEVEL = 10        # 被移除的人至少这么多级，遗志才比卖掉换能量划算

    def _yizhi_instead_of_sell(self, state: GameState, victim: str, pos) -> Optional[dict[str, Any]]:
        """**遗志代替卖人**：本来就要换下去的人等级高（≥10），卖掉只换一点能量，
        用遗志移除则把他的真实等级**全部**转给场上随机一人（按卡面分析；2026-09-24：遗志的等级落点很难确定）。
        等级落点随机 —— 只在**其余在场的人大多是要留下的**（非消耗件 ≥ 2/3）时用，落到谁身上都不算浪费。
        **绝不打给当前主C**（不换主C 时把遗志用给主C 等于直接输掉）。
        """
        if victim == self._carry_piece or self._is_carry(victim):
            return None
        slot = next((sl for sl in state.hand_slots or [] if sl.get("name") == "遗志"), None)
        if slot is None or self._parked("遗志"):
            return None
        if (self._unit_lv.get(victim) or 0) < self.YIZHI_MIN_LEVEL:
            return None
        rest = [n for n in self._field if n != victim]
        keep = [n for n in rest if not self._is_disposable(n)]
        if not rest or len(keep) * 3 < len(rest) * 2:
            return None
        logger.info("用遗志代替卖掉 %s（%s 级）：等级转给场上随机一人（其余 %d 人里 %d 个要留）",
                    victim, self._unit_lv.get(victim), len(rest), len(keep))
        self.note("遗志代替卖人", 移除=victim, 等级=self._unit_lv.get(victim), 留下的人=len(keep), 其余=len(rest))
        return {"type": "drag_card", "piece": "遗志", "src": (slot["x"], slot["y"]),
                "dst": self._on_unit(state, pos), "kind": "效果牌", "target": victim}

    def _transfer_victim(self, state: GameState) -> tuple[str | None, str]:
        """换主C 收尾这一回合该卖谁（纯判断，无副作用；_board_work_pending 也用它）。"""
        carry = self._carry_piece
        _laya_pick = str(self._carry_switch_by).lower().startswith("laya")
        if not (carry and carry in self._field and state.phase == "prepare"
                # 2026-10-12 清查：模型（局面网络 / Laya）选定的主C 前期也照常收尾，不受 TRANSFER_FROM_ROUND 限制
                and (state.round_no >= self.TRANSFER_FROM_ROUND or (_laya_pick and self.model_decides()))
                # 新主C 至少当满一回合才动老人（站稳）—— **Laya 换的例外**：carry 题本身有门槛和冷却，
                # 选定了就当回合收尾（2026-09-26 第四轮第 5 局：第 6 回合换成干将莫邪，枫叶舞整回合留在曜身上）
                and (self._carry_switch_round < state.round_no or str(self._carry_switch_by).lower().startswith("laya")
                     # **路线主C 接任当回合就收尾**（2026-10-08 第 425 局：换主C 总拖到下个回合才收尾）
                     or (carry == ((self._active_comp or {}).get("carry") if self.route_locked() else "")))
                and self._transfer_round != state.round_no):
            return None, ""
        mb = self._maple_bound
        maple_lit = any(sl.get("ready") is True for sl in state.hand_slots or []
                        if self.pieces.get(sl.get("name"), {}).get("category") == "棋手技能牌")
        # **Laya 定的换主C 不再用"觉醒过 / 估值更高"拦收尾**（2026-09-26 第四轮第 6 局：主C 没换上）：
        # carry 题面里写着 `cost: sell 孙尚香 lv20`，学生看着代价选了换，执行层就该照做；
        # 原来被"觉醒过的不为换绑卖掉"挡住，枫叶舞和装备一直留在老主C 身上，只有效果牌给了新主C。
        by_laya = str(self._carry_switch_by).lower().startswith("laya")
        # **路线主C 已让位**（_forced_released）= 换了主C，收尾照做（2026-10-02 第 109 局：主C 太多、枫叶舞没切过去——
        # 亚连让位给鲁班七号后，下面"现任不是路线主C → 不卖人"把收尾挡了，枫叶舞留在亚连、牌给鲁班，两个人都被当主C）
        by_laya = by_laya or bool(self._forced_released)
        # _is_carry 还认"推荐阵容的主C"（第 9 局：阵容写着干将莫邪，Laya 已换成米莱狄，收尾被当成"主C 不能卖"挡住）
        # 路线主C 在场、绑着的是这阶段已经不用的 1 阶开局件（干将这类）→ 不算"不能卖的引擎零件"，照常换绑
        # （2026-10-06 第 300 局：干将是大河·开团的放大器，枫叶舞因此留在它身上到最后，主C 虞姬 32 级 / 干将 86 级）
        _rc0 = (self._active_comp or {}).get("carry") if self.route_locked() else ""
        _route_transitional = bool(_rc0 and carry == _rc0 and mb
                                   and (self.pieces.get(mb, {}).get("quality") or 0) == 1
                                   and mb not in self.route_stage_members(state.round_no))
        if (mb and mb != carry and mb in self._field and self.engine_lock() and self._is_engine_part(mb)
                and not _route_transitional and (_IR_on("transfer_guards") or not self.model_decides())):
            # 引擎转起来后不为换绑枫叶舞卖引擎零件（第 14 局：卖掉干将莫邪 = 云中君的装备源没了）；枫叶舞先留在原处
            if self._skip_logged.get(("transfer_engine", mb)) != self._round_seen:
                self._skip_logged[("transfer_engine", mb)] = self._round_seen
                logger.info("换主C 收尾：%s 是引擎零件，不为换绑枫叶舞卖它，枫叶舞先留着", mb)
            return None, ""
        _rc = (self._active_comp or {}).get("carry") if self.route_locked() else ""
        _model_set = by_laya and (self.model_decides() or (state.round_no or 0) > self.EARLY_FLEX_ROUND)
        if mb and mb != carry and _rc and carry != _rc and not self._forced_released and not _model_set:
            # **现任主C 只是临时的（路线主C 还没到手）→ 不为换绑枫叶舞卖人**，等路线主C 到手一次换好
            # （2026-09-30 轮换第 11 局：第 5 回合狂铁→米莱狄→亚连，一分钟内卖掉两个日落海成员，血 52→28）
            # **但过了前期、主C 是模型（局面网络 / Laya）选定的 → 照常收尾**（2026-10-12 第 575 局：
            # 第 6 回合局面网络把主C 换成马可波罗，这里一直"枫叶舞先留在干将莫邪"，马可波罗吃不到枫叶舞。
            # 同一个坑第 417 局修过一次，10-10 加的"只有同体系的现任能转正"又把它挡回来了 —— 主C 由模型定，收尾必须跟着走）
            if self._skip_logged.get(("transfer_tmp", carry)) != self._round_seen:
                self._skip_logged[("transfer_tmp", carry)] = self._round_seen
                logger.info("换主C 收尾：现任主C %s 不是路线主C %s，枫叶舞先留在 %s，不卖人", carry, _rc, mb)
            return None, ""
        # **路线主C 已在场、枫叶舞绑在 1 阶过渡件上 → 不比估值，直接换绑**（2026-10-01 第 58 局：枫叶舞第 1 回合绑干将莫邪，
        # 主C 是锁死的路线主C 亚连；干将靠枫叶舞每回合白涨 20+ 级、估值越来越高，"为换绑卖掉他"永远不成立 ——
        # R11 干将 133 级、亚连 14 级，资源全堆在不会当主C 的人身上，第 4 名）。越早换越便宜。
        # **绑着的人不在当前路线里 → 觉醒过也换**（2026-10-03 第 129 局：枫叶舞没切到守约——
        # 第 1 回合枫叶舞绑干将莫邪（开局临时主C），第 4 回合干将合成觉醒，第 5 回合路线主C 守约到场 ——
        # 这里原来要求"1 阶且没觉醒"，觉醒的干将挡住了换绑，枫叶舞在干将身上留到第 12 回合，第 6 名）
        _off_route = mb not in self.route_members()
        route_fix = bool(_rc and carry == _rc and mb and mb != carry and mb in self._field and not maple_lit
                         # 1 阶**觉醒过也换**（2026-10-06 第 300 局：大河草案前期名单里有干将 → 不算路线外，觉醒过 → 不算过渡件，
                         # 枫叶舞从第 8 回合留在干将身上到最后，干将 86 级 / 主C 虞姬 32 级）。第 129 局起定为"觉醒过也换"；
                         # 强过渡件早卖的风险由下面的 CATCHUP 推迟兜住（最晚第 9 回合）。
                         # 只换"当前阶段已经不用"的 1 阶：守约路线这阶段还要的李元芳（古币产出方）照旧不动（冒烟 191）
                         and (((self.pieces.get(mb, {}).get("quality") or 0) == 1
                               and mb not in self.route_stage_members(state.round_no))
                              or _off_route))
        if route_fix:
            # **新主C 还太弱时先别卖强的临时主C**（2026-10-06 第 279 / 283 局：觉醒干将 47 / 58 级、满装，
            # 为换绑卖掉后场上最高只剩 10 级的人，两局 6 / 5 名）。临时主C ≥ CATCHUP_MIN_LV 级、路线主C 不到它一半、
            # 还没到 CATCHUP_UNTIL_ROUND → 先留着（装备 / 加等级牌照常给路线主C），追上一半或到回合就照常换绑。
            # 不会回到第 129 局"枫叶舞留到第 12 回合"：有截止回合。
            _lv = {**self._unit_lv, **(self.__dict__.get("_forced_lv") or {})}
            lm, lc = _lv.get(mb), _lv.get(carry)
            if (lm and lm >= self.CATCHUP_MIN_LV and (state.round_no or 0) < self.CATCHUP_UNTIL_ROUND
                    and (lc or 0) * 2 < lm and not (_laya_pick and self.model_decides())):   # 模型选定的主C 不推迟（清查 10-12）
                if self._skip_logged.get(("transfer_catchup", mb)) != self._round_seen:
                    self._skip_logged[("transfer_catchup", mb)] = self._round_seen
                    logger.info("换主C 收尾：临时主C %s %s 级、路线主C %s 才 %s 级 —— 先不卖，等追到一半或第 %d 回合",
                                mb, lm, carry, lc, self.CATCHUP_UNTIL_ROUND)
                return None, ""
            return mb, "枫叶舞换绑给路线主C（1 阶过渡件不比估值）"
        if (mb and mb != carry and mb in self._field and (by_laya or not self._is_carry(mb))
                and not maple_lit                          # 卡面亮着 = 根本没绑上，不用卖人解绑（第二轮第 3 局卖错了牛魔）
                and (by_laya or not self.can_kill(mb)                   # 绑着枫叶舞的人没输出（坦克/辅助）→ 觉醒过也卖（第 36 局牛魔）
                     or (mb not in self._ever_awakened          # 觉醒过的不为换绑卖掉
                                 and self.unit_value(mb) < self.unit_value(carry)))):
            return mb, "枫叶舞换绑给新主C"
        _mp = _laya_pick and self.model_decides()      # 模型选定的主C：转装备只查"手里有人补位"（清查 10-12）
        if (self._prev_carry and self._prev_carry != carry and self._prev_carry in self._field
                and (by_laya or not self._is_carry(self._prev_carry))
                and (_mp or self._prev_carry not in self._ever_awakened)   # 觉醒过的不为转 1 件装备卖掉（2026-09-25 验证局卖了觉醒 23 级曜）
                and self._equips_at(state, self._field[self._prev_carry]) > 0
                # 路线这阶段还要的人不为转装备卖（2026-10-09 第 438 局：守约路线的 10 级李元芳被卖，只为把装备给 1 级干将）；
                # 新主C 是 1 阶临时主C 时也不卖 —— 临时主C 之间转装备没有收益
                and (_mp or self._prev_carry not in self.route_stage_members(state.round_no or 0))
                # 最低配套件 / 涨级配套不为转装备卖（2026-10-13 第 621 局：李元芳当临时主C 吃了 1 件极寒风暴，守约接班后为转这 1 件把他卖了 ——
                # 卖人转装备是规则自己的动作，不是模型的决定，模型接管时也照查）
                and (_mp and not _IR_on("transfer_guards")
                     or (self._prev_carry not in (self.min_engine().get("early_priority") or [])
                         and not self.keep_as_support(self._prev_carry)))
                and (_mp or not ((self.pieces.get(carry, {}).get("quality") or 0) == 1 and carry not in self._ever_awakened))
                and (_mp or self.unit_value(self._prev_carry) < self.unit_value(carry))
                and self._replacement_waiting(state)):
            # 只为转移装备卖老主C 时，**手里要有人能顶上这个位置**（第十一局：人口 5/6 时卖掉蒙恬，
            # 场上剩 4 个）。装备其实能不卖人转移（点带装备的英雄会扇形展开），那个界面还没做识别。
            return self._prev_carry, "老主C 的装备转给新主C"
        cpos = self._field.get(carry)
        if cpos is not None and self._equips_at(state, cpos) < meta.MAX_EQUIPS:
            for n, worn in self._worn.items():
                if (n != carry and n in self._field and any(self.growth_equip(w) for w in worn or ())
                        and (not self._is_engine_part(n) or (self.pieces.get(n, {}).get("quality") or 0) <= 2)
                        and n != _rc
                        and self.unit_value(n) < self.unit_value(carry)):
                    # 复利成长装备穿在别人身上 → 卖掉他，装备退回手牌再给主C（第 417 局：学识宝石在孙尚香身上）
                    return n, "复利成长装备（学识宝石）转给主C"
        return None, ""

    GROWTH_EQUIP_RE = re.compile(r"每回合开始时?等级\s*[+＋]")

    def growth_equip(self, card: str) -> bool:
        """卡面"每回合开始等级 +N"的装备 = 复利成长装备（目前只有学识宝石），只读卡面文字。"""
        if not card or self.pieces.get(card, {}).get("category") != "装备牌":
            return False
        return bool(self.GROWTH_EQUIP_RE.search(self._card_text(card) or ""))

    def undo_sell(self) -> None:
        """**卖出没生效**（卖完视觉复查那一格还站着人）→ 把账本恢复成卖之前。

        2026-09-28 第 26 局：第 7 回合"卖孙膑"没生效（拖前核对"看不清"照卖、卖完没有回执），
        账本却删了他、枫叶舞记成"已解绑"；此后孙膑带着枫叶舞一直站在 (300,569)，
        卡面整局是灰的，主C 百里守约到结束都没绑上 —— 而"场上比账本多 1 个"又被解释成"画面还没更新"。
        """
        u = self._sell_undo
        name, pos = u.get("name"), u.get("pos")
        if not name or not pos:
            return
        self._field[name] = tuple(pos)
        self._cell_owner[tuple(pos)] = name
        if u.get("maple") == name:
            self._maple_bound = name
        if u.get("awakened"):
            self._awakened.add(name)
        if u.get("fodder") is not None:
            self._entry_fodder[name] = u["fodder"]
        if u.get("worn") is not None:
            self._worn[name] = u["worn"]
        if u.get("equip_seen") is not None:
            self._equip_seen[cell_key(tuple(pos))] = u["equip_seen"]
        self._freed_cell = u.get("freed")
        if u.get("prev_carry"):
            self._prev_carry = u["prev_carry"]
        self._last_sold = u.get("last_sold") or ("", -1)
        self._sold_names.discard(name)
        k = (self._round_seen, name)
        self._sell_fails[k] = self._sell_fails.get(k, 0) + 1
        self._sell_undo = {}
        logger.warning("卖 %s 没生效（%s 那格还站着人）—— 账本恢复%s", name, pos,
                       "，本回合不再卖他" if self._sell_fails[k] >= 2 else "")
        self.note("卖出没生效", round=self._round_seen, 英雄=name, 格=f"{pos[0]},{pos[1]}", 次数=self._sell_fails[k])

    def sell_blocked(self, name: str) -> bool:
        return self._sell_fails.get((self._round_seen, name), 0) >= 2

    def decide_sell_unit(self, state: GameState) -> Optional[dict[str, Any]]:
        """把"只为吃效果而上场"的英雄卖掉，腾出人口。

        技巧：用不上的英雄牌，人口有空位时先放上场吃掉效果
        （例：孙尚香"登场：获得1次免费刷新"），效果结算后拖到右下角卖掉，
        等于白嫖一次效果 + 回收能量。
        """
        # ---- 换主C 之后的收尾：卖掉枫叶舞绑着的人 / 老主C，把枫叶舞和装备都转给新主C ----
        # 2026-09-23：要找时机卖掉老主C，把枫叶舞挂给新的输出角色，
        # 并把装备都给他。枫叶舞绑住后**只有卖掉被绑的人才能重绑**，装备也是卖人才退回手牌。
        # 时机：新主C 已经在场上（站稳了才转，不然转过去没人接），每回合最多卖一个。
        carry = self._carry_piece
        victim, why = self._transfer_victim(state)
        if victim and self._shop_blocks_board(state):
            # **商店开着不能卖人**（2026-09-26 第四轮第 5 局：要卖的曜在 (617,372)，正好压在商店卡面下，
            # 拖拽前核对写着"看不清"照样拖，抓起来的是商店里的牌 —— 买了一张、账本却记成"卖出曜"，
            # 枫叶舞也就没换绑——换主C 判定出来了却没实际执行）
            logger.info("换主C 收尾要卖 %s（%s）—— 商店开着挡住棋盘，先关店", victim, why)
            return {"type": "toggle_shop", "intent": "close"}
        if victim:
            mb = self._maple_bound
            # **按格子上的等级读数再核一次**（不信账本里的身份）：要卖的那格明显比新主C 那格等级高 → 不卖、换主C 撤回。
            # 2026-09-25 第四轮第 2 局：账本把两格的人记反了，18 级绑枫叶舞的干将莫邪被当成 3 级，
            # "换主C 等级不能更低"被放过，接着这里把它卖掉换绑给 3 级亚连 —— 之后一路掉血、第 6 名。
            lvl = state.unit_levels or {}
            lv_v = lvl.get(cell_key(self._field[victim])) or self._unit_lv.get(victim)
            lv_c = lvl.get(cell_key(self._field[carry])) or self._unit_lv.get(carry)
            # 只拦**离谱**的差距（身份记反那种：第 1 局 18 级 vs 3 级）。两个都是真实高等级单位时（66 vs 41）不拦 ——
            # 2026-09-25 第四轮第 5 局：主C 已换成孙悟空（评分 8、觉醒、41 级），原门槛 1.5 倍会把它撤回给孙尚香（主C 换不过去）
            # Laya 定的换主C 不走这条（2026-09-26：不给 Laya 加保护；第 9 局把 Laya 选的海诺撤回成嬴政）
            _rc_now = (self._active_comp or {}).get("carry") if self.route_locked() else ""
            if (not str(self._carry_switch_by).lower().startswith("laya") and "路线主C" not in why
                    and carry != _rc_now     # 新主C 就是路线主C：不撤回（第 425 局：20 级李元芳 vs 刚到的沈梦溪，撤回后两人来回换）
                    and lv_v is not None and lv_c is not None and lv_v >= max(lv_c * 3, lv_c + 15)):
                logger.info("换主C 收尾取消：要卖的 %s 那格 %s 级，远高于新主C %s 那格 %s 级 —— 撤回换主C",
                            victim, lv_v, carry, lv_c)
                self.note("换主C收尾取消", round=state.round_no, 卖=victim, 等级=lv_v, 新主C=carry, 新主C等级=lv_c)
                if victim == self._prev_carry or victim == mb:
                    self._carry_piece, self._carry = victim, self._field[victim]
                    self._prev_carry = ""
                self._transfer_round = state.round_no
                return None
            self._transfer_round = state.round_no
            logger.info("换主C 收尾：卖掉 %s（%s，新主C %s）", victim, why, carry)
            self.note("换主C收尾", round=state.round_no, 卖=victim, 新主C=carry, 原因=why)
            # _prev_carry 在卖人真正执行后（note_executed）才清，卖没成 / 撤销时还能再转装备（2026-09-30 审查）
            self.record_decision("carry_transfer", state, [{"sell": victim}], victim, why=why)
            return {"type": "sell_unit", "piece": victim, "src": self._field[victim], "why": why}
        # ---- 人口满了 + 手上有明显更强的 -> 卖掉场上最弱的换位 ----
        # **这是"前期低阶过渡 → 中期换高阶阵容"那条打法主线的执行环节**
        # （2026-09-22）。没有它就会死锁：人口 7/7、手牌堆满买来的英雄、
        # 钱还在往外花，但一个也上不去（实战血量掉到 1 还在买）。
        # 门槛 SWAP_MARGIN：必须**明显**更强才动刀 —— 势均力敌时换人只是白扔一次买入，
        # 而且**误卖核心**是这个项目已经踩过的坑。
        if self._shop_blocks_board(state):
            return None          # 商店开着挡住棋盘：下面几种卖人都等关店后再说（抓起来的会是商店的牌）
        waiting_awk = [n for n in self._awakened if n not in self._field
                       and any(sl["name"] == n for sl in state.hand_slots or [])]
        cap0 = state.population_cap if state.population_cap and state.population_cap > 0 else (
            self._last_pop_seen[1] if self._last_pop_seen else 0)
        if waiting_awk and cap0 and len(self._field) < cap0:
            # 只在**账本里场上确实有空位**时才不卖（觉醒回手那个人已经从账本移除 → 它的格子空着）；
            # 觉醒卡来自别处（化铁为金 / N 选 1）、场上真的满了，照常卖最弱的腾位。
            # **觉醒回手的英雄自己空出了一格，不用为它卖人**（2026-09-24 match13：主C 觉醒回手后没上场，
            # 直接导致输局）：推演战术让主C 西施在准备阶段最后几秒觉醒回手，同一帧这里按"人口满"先卖了阿轲，
            # 再去拖西施时准备阶段结束 —— 场上少了两个人。觉醒卡上场是最高优先级，交给 decide_deploy 直接上。
            if self._skip_logged.get(("awk_no_swap", waiting_awk[0])) != self._round_seen:
                self._skip_logged[("awk_no_swap", waiting_awk[0])] = self._round_seen
                logger.info("%s 觉醒回手，自己的格子空着 —— 不卖人，直接上场", waiting_awk)
            return None
        # **给"直接上阵高阶英雄"的牌腾位**（2026-09-26：大明星满员时一直留在手里用不出去）。
        # 大明星 = 上阵 1 名 棋手等级×7 级的 5 阶英雄（带觉醒进度）—— 比场上最弱的非核心强得多。
        # 只认审定过的"要空人口、优先级 ≥8"的自用牌；每回合每张牌只为它卖一次（没打出去也不再卖第二个）。
        if state.population_valid and state.population >= state.population_cap and self._field:
            for sl in state.hand_slots or []:
                n = sl["name"]
                u = self._usage_self_cast(n)
                if not (u and u.get("priority", 0) >= 8
                        and any(str(w).startswith("board_room") for w in u.get("when") or [])):
                    continue
                if (state.round_no, n) in self._swapped_for:
                    continue
                weakest = self._weakest_on_field()
                if not weakest or self._is_support(weakest[0]) or self._carry_buddy(weakest[0]):
                    continue
                name, pos, val = weakest
                self._swapped_for.add((state.round_no, n))
                logger.info("人口满，为 %s 卖掉场上最弱的 %s(%d) 腾位", n, name, val)
                self.note("换阵容卖人", round=state.round_no, 卖=name, 为了上=n, 分值=f"{val}->高阶牌")
                return {"type": "sell_unit", "piece": name, "src": pos}
        if (state.population_valid and state.population >= state.population_cap
                and self._field):
            best = self._best_in_hand(state)
            for_awakened = bool(best) and best[0] in self._ever_awakened
            weakest = self._weakest_on_field(allow_core=for_awakened, incoming=best[0] if best else "")
            if best and (state.round_no, best[0]) in self._swapped_for:
                best = None      # 本回合已经为它卖过一个人了，它没上去就别再卖第二个（第十局为露娜连卖两人）
            if best and self.swap_by_laya and (not for_awakened or not _IR_on("awakened_sell_pick")):
                best = None      # 卖谁换人交给 swap 题；A15（2026-10-15）觉醒卡腾位也交给换人网络（觉醒卡在等时题里去掉 bench）
            if weakest and best and best[1] > weakest[2] + self.SWAP_MARGIN:
                name, pos, val = weakest
                if not (self._is_carry(name) or (self._is_core(name) and not for_awakened)):
                    logger.info("人口满且手上有更强的 %s(%d)，卖掉场上最弱的 %s(%d) 换位",
                                best[0], best[1], name, val)
                    self.note("换阵容卖人", round=state.round_no, 卖=name,
                              为了上=best[0], 分值=f"{val}->{best[1]}")
                    # **账本在 note_executed 里改，不在这里改**（同"买牌"那条教训）：
                    # 这一步只是"决定卖"，执行层可能因为去重/异常根本没点出去。
                    # 2026-09-23 实战：sell_unit 因为字段对不上每次都 KeyError，
                    # 而账本已经先把人从 _field 里删了 —— 画面上人还在，账上没了。
                    # **字段名必须是 src**：执行器读的是 action["src"]。
                    # 这里原来给的是 x/y，于是"换阵容卖人"这条路**每次都抛 KeyError**，
                    # 一个英雄都没卖出去过 —— 2026-09-22 就出现过
                    # "场上的英雄只买不卖"，当时查的是决策逻辑，
                    # 没想到是执行层字段对不上（异常被主循环吞成一条 ERROR 就过去了）。
                    self._swapped_for.add((state.round_no, best[0]))
                    self.record_decision("swap", state, [{"sell": name, "value": val},
                                                         {"deploy": best[0], "value": best[1]}], name)
                    yz = self._yizhi_instead_of_sell(state, name, pos)
                    if yz:
                        return yz
                    return {"type": "sell_unit", "piece": name, "src": pos}

        # 棋手经验型英雄（弈星）：它的价值**只在升级那一瞬间**兑现。
        # 这回合已经升过级、而且它不是阵容牌 → 收益吃完了，卖掉腾人口换能量。
        # 还没升级的话必须留着，卖早了等于白买（2026-09-22）。
        for name in list(self._field):
            if self.model_decides() and not _IR_on("sell_level_engine"):
                break                        # A4（2026-10-15）：卖弈星交给换人网络
            if not self._is_level_engine(name) or self._is_core(name) or self._is_carry(name):
                continue
            levels = self._engine_levels.get(name, 0)
            if levels < 1:
                continue                     # 收益还没兑现，卖了等于白买
            if len(self._field) <= 1:
                continue
            # **"卖"是独立的价值判断，不能和"升完了"绑死**（2026-09-22 更正）。
            # 弈星本身**是有战斗力的，只是不大**——所以"升完这一次"并不等于它没用了。
            # 真正该卖的条件是：**这个位置有更好的用途**。否则留着当个身板，
            # 顺便攒下一次升级的经验（没有更好的牌时，可以升 2 次再卖）。
            pop_tight = (state.population_valid
                         and state.population_cap - state.population <= 1)
            better_waiting = any(
                sl["name"] not in self._field
                and (self._is_carry(sl["name"]) or self._is_core(sl["name"]))
                and self.pieces.get(sl["name"], {}).get("category") == "英雄牌"
                for sl in state.hand_slots)
            # **手里要有英雄能补这个位置才卖**（2026-10-04 第 5~6 名复盘：第 183 局第 7 回合"人口吃紧"卖掉弈星、手里没人补，
            # 少一个人迎战一回合掉 30 血；第 86~183 局这样卖了 20 次，12 次下一回合场上缺人）
            any_waiting = any(sl["name"] not in self._field and sl["name"] != name
                              and self.pieces.get(sl["name"], {}).get("category") == "英雄牌"
                              for sl in state.hand_slots)
            # 2026-10-13 复盘 a（第 601、603 局第 4 回合 3/5 就卖）："手里有更该上场的人"只在满员时才需要腾位 —— 有空位直接上就行
            if better_waiting and not pop_tight and not (state.population_valid and state.population >= state.population_cap):
                better_waiting = False
            if not (pop_tight or better_waiting) or not any_waiting:
                logger.info("%s 已贡献 %d 次升级，但%s（它也有身板），留着继续攒经验", name, levels,
                            "手里没有能补位的英雄" if not any_waiting else "场上位置暂时没人要")
                continue
            why = "人口吃紧" if pop_tight else "手牌里有更该上场的阵容英雄"
            pos = self._field[name]          # 账本等执行成功再改（见 note_executed）
            logger.info("%s（棋手经验型）已贡献 %d 次升级，%s → 卖掉腾位", name, levels, why)
            return {"type": "sell_unit", "piece": name, "src": pos,
                    "why": f"棋手经验型，已贡献{levels}次升级；{why}"}

        # **登场效果已用完的阵容外英雄**（2026-10-03 第 126 局：阿轲留在阵容里基本没有作用，
        # 只给主C 提供了 1 级）：人口满、手里有路线成员等着上场 → 卖它腾位。
        hand_names = [sl["name"] for sl in state.hand_slots]
        pop_full = (state.population_valid and state.population_cap > 0
                    and state.population >= state.population_cap)
        members = self.route_members()
        waiting = [n for n in hand_names if n in members and n not in self._field
                   and self.pieces.get(n, {}).get("category") == "英雄牌"]
        if pop_full and waiting and len(self._field) > 1 and self.model_decides() and self.swap_by_laya:
            # 2026-10-12：卖谁给路线成员腾位是换人题（有人等着上场时换人网络会被问到），规则不抢先
            _spent = [n for n in self._field if self.spent_entry_unit(n, hand_names)]
            if _spent and self._skip_logged.get(("spent_entry_md", waiting[0])) != state.round_no:
                self._skip_logged[("spent_entry_md", waiting[0])] = state.round_no
                logger.info("（仅记录）旧规则会卖登场效果已用完的 %s 给 %s 腾位，交给换人网络", _spent, waiting[0])
        elif pop_full and waiting and len(self._field) > 1:
            for name in list(self._field):
                if not self.spent_entry_unit(name, hand_names):
                    continue
                if (self._is_tank(name) and not self._is_tank(waiting[0])
                        and sum(1 for n in self._field if self._is_tank(n)) <= self.MIN_FRONT_TANKS):
                    continue
                if (state.round_no, waiting[0]) in self._swapped_for:
                    continue
                self._swapped_for.add((state.round_no, waiting[0]))
                pos = self._field[name]
                logger.info("%s 的登场效果已用完、不在路线里，人口满 → 卖掉给路线成员 %s 腾位", name, waiting[0])
                self.note("换阵容卖人", round=state.round_no, 卖=name, 为了上=waiting[0], 分值="登场已用完")
                return {"type": "sell_unit", "piece": name, "src": pos,
                        "why": f"{name} 登场效果已用完、不在路线里；给 {waiting[0]} 腾位"}

        for name, deployed_round in list(self._entry_fodder.items()):
            if self.model_decides():
                # 2026-10-12 第 578 局：规则把"吃完整备效果的杂牌"米莱狄卖了 —— 她是日落海核心体系的人（整备给核心 +3）。
                # 卖不卖人交给换人网络（人口满、有人等着上场时会出题）；没人等着时卖掉只是少一个人头
                if name in self._field and self._skip_logged.get(("fodder_md", name)) != state.round_no:
                    self._skip_logged[("fodder_md", name)] = state.round_no
                    logger.info("（仅记录）旧规则会卖掉吃完效果的 %s，交给换人网络", name)
                continue
            if (self._entry_effect_kind(name) == "整备"
                    and state.round_no <= deployed_round):
                continue                     # 整备要开战后才结算，下回合再卖
            if name not in self._field:
                self._entry_fodder.pop(name, None)
                continue
            if not self._is_disposable(name):
                # **后来成了主C / 核心 / 3 阶以上的，不再是"吃完效果就卖"的杂牌**
                # （2026-09-23 第十一局：裴擒虎作为整备杂牌上场、随后被选成主C，下回合按杂牌卖掉了）
                self._entry_fodder.pop(name, None)
                continue
            if len(self._field) <= 1:
                # 场上就这一个，卖了就空场——白嫖效果不值这个代价
                # （2026-09-22 实战：刚上场的狂铁下回合被卖掉，场面又归零）
                logger.info("%s 是场上唯一英雄，暂不卖出", name)
                continue
            pos = self._field[name]          # 账本等执行成功再改（见 note_executed）
            kind = self._entry_effect_kind(name)
            if "退场" in self.pieces.get(name, {}).get("skill", ""):
                logger.info("%s 带退场效果，卖掉还能再触发一次", name)
            logger.info("卖掉吃完%s效果的 %s（第%d回合上场，现在第%d回合）",
                        kind, name, deployed_round, state.round_no)
            return {"type": "sell_unit", "piece": name, "src": pos}
        return None

    # **卖手牌暂时关掉**（2026-09-23）：翻遍历史日志，"把手牌拖到右下角能量圆盘"执行了 118 次，
    # **一次都没卖掉过**（卖场上的英雄同样的拖法是成功的）。每次失败还会反复占掉决策帧。
    # 正确的卖法要实测确认后再打开。
    HAND_SELL_ENABLED = True       # 2026-09-23 改成拖到左侧「出售」区后重新打开（待实战确认）

    def carry_settled(self) -> bool:
        """主C 定了、不换了：路线已锁，路线主C 在场且就是现任主C（或已让位给更强的人）。"""
        if not self.route_locked() or not self._carry_piece or self._carry_piece not in self._field:
            return False
        return self._carry_piece == (self._active_comp or {}).get("carry") or bool(self._forced_released)

    _HERO_ROLES: Optional[dict] = None

    def hero_role(self, name: str) -> dict:
        """读卡面文本 + 技能文本得出的英雄定位（data/hero_roles.json，2026-09-30：不通过现有标签定位角色，
        而是实际分析英雄卡的基础文本及技能文本）：combat_role / functions / functional_value / carry_value /
        is_second_carry_risk / engine_grants（主C 引擎给谁涨级）/ engine_needs_carry。"""
        if RuleEngine._HERO_ROLES is None:
            try:
                RuleEngine._HERO_ROLES = json.loads((Path(__file__).resolve().parents[2] / "data" / "hero_roles.json")
                                                    .read_text(encoding="utf-8")).get("heroes") or {}
            except Exception:                              # noqa: BLE001
                RuleEngine._HERO_ROLES = {}
        return RuleEngine._HERO_ROLES.get(name) or {}

    def can_kill(self, name: str) -> bool:
        """有没有输出、能不能杀人（按卡面文本分析的主定位；hero_roles 缺项才看主C 评分）。
        阿离的专属卡要靠**枫叶舞绑定的英雄击杀**才解锁 —— 绑给牛魔这种没输出的，一张都拿不到（2026-09-30）。"""
        hr = self.hero_role(name)
        if hr:
            roles = hr.get("combat_role") or []
            return bool(roles) and roles[0] in ("carry_phys", "carry_mag", "assassin", "bruiser")
        return ((self.hero_reviewed.get(name) or {}).get("carry_score", 0) or 0) >= 5

    def functional_hero(self, name: str) -> bool:
        """功能性英雄：路线这阶段要的人、贴主C 的；其余按**卡面文本分析**的功能价值（≥6 且不是第二主C 风险）。
        hero_roles 里查不到才临时退回标签。"""
        if self._carry_buddy(name) or name in self.route_stage_members(self._round_seen):
            return True
        hr = self.hero_role(name)
        if hr:
            return (hr.get("functional_value") or 0) >= 6 and not hr.get("is_second_carry_risk")
        if self._is_support(name):
            return True
        tags = self.pieces.get(name, {}).get("tags") or []   # 临时兜底：hero_roles 缺项
        cs = (self.hero_reviewed.get(name) or {}).get("carry_score", 5)
        return "坦克" in tags or ("前排" in tags and cs <= 4 and not any(t in tags for t in ("刺客", "射手", "法师", "战士")))

    _ENGINE_GROUP_RE = re.compile(r"你的(?:【([^】]+)】|([一-龥]{2}))英雄")

    def carry_engine_beneficiary(self, name: str) -> bool:
        """**吃主C 引擎红利的英雄**（2026-09-30：刺客百里杀人后下一回合会按杀人数给其他闪现英雄等级，这时可以练第二个刺客；
        但百里需要是主C，因为只有主C 杀人数才够）：主C 卡面写着「你的【X】英雄 / 你的X英雄…等级」，
        name 属于 X（派系对得上，或自己技能带 X 关键词，如「闪现」）→ 它是受益者，不算分资源的第二主C。"""
        carry = self._carry_piece
        if not carry or carry == name:
            return False
        for g in self.hero_role(carry).get("engine_grants") or []:
            if name in (g.get("feeds") or []):
                return True                 # 分类里明确写了"主C 引擎喂给谁"
        ct = (self.pieces.get(carry, {}).get("skill") or "") + (self._card_text(carry) or "")
        if "等级" not in ct:
            return False
        me = (self.pieces.get(name, {}).get("skill") or "") + (self._card_text(name) or "")
        from agent.decision import routes as _R
        for fac, kw in self._ENGINE_GROUP_RE.findall(ct):
            if fac and _R.norm_banned(fac) in (self.pieces.get(name, {}).get("synergies") or []):
                return True
            if kw and kw not in ("其他", "所有", "全部") and kw in me:
                return True
        return False

    def carry_locked_in(self, state) -> bool:
        """**主C 真正定下来了**：carry_settled + 过了前期（> EARLY_FLEX_ROUND）+ 现任主C 不是弱/过渡主C（审定评分 ≥5）。
        2026-10-05：不能影响前期还没定下主C、要替换的时候 —— 只有这时才拦"第二主C"的买入和加分。"""
        if not self.carry_settled() or (state.round_no or self._round_seen or 0) <= self.EARLY_FLEX_ROUND:
            return False
        return (self.hero_reviewed.get(self._carry_piece, {}).get("carry_score", 5) or 0) >= 5

    FEED_FUNCS = ("carry_level", "team_level")

    def feeds_carry(self, name: str) -> bool:
        """**能给主C 涨等级的英雄**（读卡面分析 hero_roles 的 functions：carry_level / team_level）。
        2026-10-06 第 332 局：多了个完全没作用的花木兰，不如能给主C 提供等级的角色
        —— 5 阶兑换券三选一 武则天（古币等级 +1，守约路线正好吃古币）/ 杨玉环 / 花木兰，选了花木兰。"""
        if not name or name == self._carry_piece:
            return False
        fs = self.hero_role(name).get("functions") or {}
        carry_fac = set(self.pieces.get(self._carry_piece or "", {}).get("synergies") or [])
        for k in self.FEED_FUNCS:
            txt = fs.get(k)
            if not txt:
                continue
            if k == "team_level" and re.search(r"开团|登场|合成|阵亡|牺牲|随机", str(txt)):
                # 有触发条件 / 随机给的"全队涨级"不算喂主C（2026-10-08 第 434 局：按这条买了张良 / 阿轲 / 猪八戒，
                # 河洛阵容里没开团、登场只一次、合成同阶才涨 —— 买来毫无意义）
                continue
            # 卡面点名派系（「…1名【日落海】英雄等级+2」）→ 主C 得是这个派系才吃得到（2026-10-07 第 354 局：
            # 亚连的涨级只给日落海，主C 守约是河洛，却被当成"喂主C"放行）
            facs = re.findall(r"【([^】]+)】", str(txt))
            from agent.decision import routes as _Rn
            _norm = lambda xs: {(_Rn.norm_banned(x) or x)[:2] for x in xs}   # noqa: E731  「三分」vs「三分之地」（2026-10-13 卡面审计第 7 条）
            if facs and self._carry_piece and not (_norm(facs) & _norm(carry_fac)):
                continue
            return True
        return False

    LATE_REMOVE_S = 15           # 准备阶段剩余少于这么多秒不用移除 / 变身己方英雄的牌（第 728 局）
    SUB_CARRY_MIN_DPS = 75       # 副C 至少要有这点输出（hero_strength 40 级 dps）
    _HS_CACHE: Optional[dict] = None

    def _dps40(self, name: str) -> float:
        if RuleEngine._HS_CACHE is None:
            try:
                d = json.loads((Path(__file__).resolve().parents[2] / "data" / "hero_strength.json").read_text(encoding="utf-8"))
                RuleEngine._HS_CACHE = d.get("heroes", d)
            except (OSError, ValueError):
                RuleEngine._HS_CACHE = {}
        return float(((RuleEngine._HS_CACHE.get(name) or {}).get("lv40") or {}).get("dps") or 0)

    def _route_team(self) -> set:
        """锁定路线的最终阵容（主C / 副C / 主辅助 / 副辅助 / 主坦 / 副坦都从这里选；没锁路线返回空集）。"""
        if not self.route_locked():
            return set()
        try:
            from agent.decision import routes as _Rt
            # 2026-10-19（该卖钟馗、留李元芳、上明世隐）：引擎表 not_wanted 的人不算最终阵容件（狄仁杰路线的钟馗）
            _me = self.min_engine()
            # extra_wanted（10-19：沈梦溪路线的刘禅 —— 放技能 +2 临时等级、明世隐转永久）算进最终阵容
            # 2026-10-20（review-interim-conflicts 第 1 条）：规则层的 _active_comp 没有 heroes 字段 → final_team 几乎总是空集，
            # final_team_first 等把最终阵容件当成"外人"（第 817 / 822 局卖李元芳）。改按路线 key 取官方 / 自定义阵容，取不到再退回 _active_comp
            _lu = _Rt.by_key(self._route_key) or (self._active_comp or {})
            _fin = set(_Rt.final_team(_lu)) or set(_Rt.final_team(self._active_comp or {}))
            _fin = (_fin | set(_me.get("extra_wanted") or [])) - set(_me.get("not_wanted") or [])
            from agent.decision import lord_config as _LCt
            _subf = _LCt.load().substitutes
            return _fin | {x for n in _fin for x in _subf(n) if x in self._field}     # 10-21 审查 9：场上的替代件也算最终阵容件
        except Exception:                                   # noqa: BLE001
            return set()

    def sub_carry(self) -> str:
        """**副C**（2026-10-08：辅助主C 的英雄或坦克里有些技能带一点输出能力，定位为副C，过剩的输出装备交给他）：
        主C 定了以后，场上不是主C、不是别的 C 位、不是消耗件的人里，40 级输出（hero_strength dps）最高、且 ≥ SUB_CARRY_MIN_DPS 的那个。
        它不吃主C 的成长资源，只接主C 穿不下的输出装。"""
        if not self.carry_settled():
            return ""
        cands = [n for n in self._field if n != self._carry_piece and not self.other_c(n)
                 and not self._is_disposable(n) and n not in self.NO_EQUIP_UNITS]
        # 2026-10-13：主C、副C、主辅助、副辅助、主坦、副坦都指核心阵容里的人 → 路线锁定时先在路线最终阵容里选
        # 阵容里没有够格的（40 级输出不到门槛）才在别的人里找
        _team = self._route_team()
        lv = self._unit_lv
        _tc = [n for n in cands if n in _team]
        if _tc:
            b = max(_tc, key=lambda n: (self._dps40(n), lv.get(n) or 0))
            if self._dps40(b) >= self.SUB_CARRY_MIN_DPS:
                return b
        if not cands:
            return ""
        best = max(cands, key=lambda n: (self._dps40(n), lv.get(n) or 0))
        return best if self._dps40(best) >= self.SUB_CARRY_MIN_DPS else ""

    SUPPORT_ROLES = {"support", "summoner", "utility"}

    def awakened_support(self, name: str) -> bool:
        """**已觉醒的辅助不再买同名**（2026-10-08：辅助定位的英雄觉醒后不需要再买同名卡升级，
        没什么作用，而且会拖慢棋手升级速度）：场上这名英雄已觉醒、定位是辅助（卡面分析 support / summoner / utility），且不是主C。"""
        if not name or name == self._carry_piece or name not in self._field:
            return False
        if name not in self._ever_awakened and name not in self._awakened:
            return False
        if re.search(r"登场：[^。]*可转化的等级\s*\+", self._card_text(name) or ""):
            # 例外（2026-10-19：明世隐要多来几张，每次登场临时等级转换数值都会变高）：登场效果可叠加的辅助，
            # 同名牌合成 / 再打出都重复触发登场（卡面答复 C23）→ 觉醒后照样要
            return False
        return bool(set((self.hero_role(name) or {}).get("combat_role") or []) & self.SUPPORT_ROLES)

    LOWEST_LEVEL_RE = re.compile(r"(等级最低|最低等级)[^。]*(提升|获得|等级\s*[+＋]|分配)")

    NEVER_PICK = _LordRule("never_pick", ("星源星徽", "阿科米亚之子", "驱雷", "月影之力", "阵营祝福", "青莲剑谱"))     # 绑定棋手：data/lords/阿离/lord_rules.json never_pick（兜底值同）  # 青莲剑谱：坦克主C 用的，阿离没有坦克主C（2026-10-22）     # 阵营祝福：各阵营随机 1 人 +4，随到主C 概率太低（10-21）     # 阿科米亚之子：别的棋手特殊套路的装备，阿离用不上（2026-10-21）     # 毫无意义的装备，N 选 1 / 商店 / 拍卖都不拿（2026-10-20 晚第 848 局）；拿到了只给最没用的日落海英雄（faction_weakest）

    def never_pick(self, name: str) -> bool:
        return name in self.NEVER_PICK or self.feeds_lowest(name)

    def feeds_lowest(self, name: str) -> bool:
        """效果 / 天赋牌把等级给"场上等级最低"的人（扶倾济弱 +9、扫拂、传承分配）——单主C 阵容主C 永远不是最低的，这类牌给不到主C
        （2026-10-20 第 844 局：扶倾济弱给最低等级英雄加等级，对养主C 的阵容无效）。英雄牌不算。"""
        c = self.pieces.get(name, {}) or {}
        if c.get("category") in ("英雄牌", "召唤物") or "(" in name:
            return False
        t = c.get("skill") or self._card_text(name) or ""
        return bool(self.LOWEST_LEVEL_RE.search(t.replace("敌方等级最低", "")))      # 鲸吞"吞噬敌方等级最低的"不算

    def would_awaken(self, act: dict) -> str:
        """这个动作执行成功后会不会让某人**马上觉醒回手**（和 note_action 里"预期觉醒回手"同一套判断，执行前算）；会 → 返回那人名字。
        用途：准备阶段剩不到几秒时先不做（2026-10-20 第 843 局 R12：最后不到 1 秒妙音凑满守约觉醒，守约回手、整场战斗主C 不在场）。"""
        if not act or act.get("type") != "drag_card":
            return ""
        piece, tgt = act.get("piece") or "", act.get("target") or ""
        if (not act.get("kind") and act.get("merge") and piece in self._field and piece not in self._ever_awakened
                and self._awaken_seen.get(piece) is not None and self._awaken_seen[piece] + 1 >= 3):
            return piece
        if tgt and tgt in self._field and tgt not in self._ever_awakened:
            _m = re.search(r"觉醒进度(?:额外)?\s*\+\s*(\d)", self.pieces.get(piece, {}).get("skill") or "")
            _p = self._awaken_seen.get(tgt)
            if _m and _p is not None and _p + int(_m.group(1)) >= 3:
                return tgt
        return ""

    SINGLE_COPY = ("太乙真人",)     # 只要 1 张、不觉醒（2026-10-20 晚第 846 局：太乙真人只需要 1 个，不觉醒）

    def single_copy_owned(self, name: str, state=None) -> bool:
        """临时规则 single_copy：SINGLE_COPY 里的人已经有了（场上 / 手里）→ 不再拿同名。"""
        if name not in self.SINGLE_COPY:
            return False
        hand = {sl.get("name") for sl in (getattr(state, "hand_slots", None) or [])} if state is not None else set()
        return name in self._field or name in hand or self._hand_book.get(name, 0) > 0

    def awakened_no_copy(self, name: str) -> bool:
        """**已觉醒的非主C 不再买同名（棋手满级前）**（临时规则 awakened_no_copy；2026-10-20 晚：
        辅助英雄觉醒后先不买同名卡，等棋手满级再补，优先升主C 和棋手等级；当时盾山、李元芳已觉醒仍在买同名卡）。
        比 awakened_support 宽：不看定位（盾山是坦克、李元芳是古币件也算）；主C / 路线主C、登场效果可叠加的（明世隐）除外。"""
        if not name or name in (self._carry_piece, self.route_carry()) or name not in self._field:
            return False
        if name not in self._ever_awakened and name not in self._awakened and name not in self._confirmed_awakened:
            return False
        return True             # 10-21：超核心件只凑到觉醒（明世隐 / 雅典娜）——原来对登场可叠加的明世隐开的例外撤掉

    def other_c(self, name: str) -> bool:
        """**主C 定了以后的"别的 C 位"**：拍卖 / N 选 1 / 商店统一不拿（阿离单主C，2026-10-08 归纳）。
        主C 同名（合成）和给主C 涨等级的人除外。"""
        rc = self.route_carry() if self.route_locked() else ""
        # 主C 已定，或路线主C 已经到手（在手里 / 手牌账里，刚买还没上场）都算"定了"（2026-10-08 第 436 局：
        # 刚买了沈梦溪，同一帧 Laya 又买了守约，两个 C 一起上场）
        settled = self.carry_settled() or bool(rc and rc in self.hand_book_names())
        return bool(name and settled and name != self._carry_piece and name != rc
                    and self.pieces.get(name, {}).get("category") == "英雄牌"
                    and self.c_position_hero(name) and not self.feeds_carry(name))

    C_ROLES = {"carry_mag", "carry_phys", "assassin"}

    def c_position_hero(self, name: str) -> bool:
        """C 位英雄（输出位）：卡面分析的战斗定位含法系 / 物理输出 / 刺客，或主C 价值 ≥7。不看标签。"""
        hr = self.hero_role(name) or {}
        cv = hr.get("carry_value") or 0
        return (bool(set(hr.get("combat_role") or []) & self.C_ROLES) or cv >= 7
                or (bool(hr.get("is_second_carry_risk")) and cv >= 5))      # 花木兰（斗士，但有第二核心风险）也算

    def late_c_position(self, name: str) -> bool:
        """**主C 定了以后，别再买新的 C 位英雄**（2026-10-10 第 515 局：敖隐在这套阵容里也是 C 位英雄，等级低、后期上场没用，被手动卖掉）：
        第 11 回合规则按"单核候选"给敖隐加分买了（路线成员也不放过 —— 第 367 局把它当成大河路线收尾件放行过）。
        场上已有同名（合成）、给主C 涨等级的、主C 本人除外。"""
        if not name or name == self._carry_piece or not self.carry_settled():
            return False
        if self.pieces.get(name, {}).get("category") != "英雄牌" or name in self._field:
            return False
        return self.c_position_hero(name) and not self.feeds_carry(name)

    def unwanted_second_carry(self, name: str) -> bool:
        """主C 定了以后不该再拿的英雄：第二主C、不给主C 涨等级、也不在路线任何阶段的名单里
        （2026-10-07 第 367 局：大河路线第 7 回合拍卖的敖隐是路线后期收尾件，被当成第二主C 拦了）。"""
        return (bool(name) and self.pieces.get(name, {}).get("category") == "英雄牌"
                and self.second_carry(name) and not self.feeds_carry(name)
                and name not in self.route_members())

    def second_carry(self, name: str) -> bool:
        """主C 定了以后，路线外、强度评分 ≥7 的输出型英雄 = "第二个主C"（2026-09-30 第 35 局：沈梦溪定了之后还上阿轲 / 李信 / 花木兰，
        仍在练多个主C；主C 确定不换之后应多上功能性英雄）。"""
        if not self.carry_settled():
            # 路线已锁、主C 还在手里：路线成员（含路线主C）以外的同类也算（2026-10-08 第 401 / 407 局：
            # 拍卖时守约还没上场 → 这里直接返回 False，拍卖护栏形同虚设，两局都拍来花木兰）
            # 前期（≤EARLY_FLEX_ROUND）照旧放行：那时"拿到什么用什么"，临时主C 可以是路线外的人
            if (not self.route_locked() or (self._round_seen or 0) <= self.EARLY_FLEX_ROUND
                    or name in self.route_members()):
                return False
        if name == self._carry_piece or self.functional_hero(name):
            return False
        if self.carry_engine_beneficiary(name):
            return False                    # 吃主C 引擎红利（玄策主C → 其他闪现英雄），可以练
        hr = self.hero_role(name)
        if hr:
            return bool(hr.get("is_second_carry_risk")) and (hr.get("carry_value") or 0) >= 5
        cs = (self.hero_reviewed.get(name) or {}).get("carry_score", 0)       # 临时兜底：hero_roles 缺项
        tags = self.pieces.get(name, {}).get("tags") or []
        return cs >= 6 or (cs >= 5 and any(t in tags for t in ("刺客", "射手", "法师", "战士")))

    def _planned_hero(self, name: str) -> bool:
        """这张英雄牌在不在阵容规划里：当前阵容的主C / 核心，或能顶替现任的主C 人选。"""
        comp = self._active_comp or {}
        if name == comp.get("carry") or name in (comp.get("core") or []) or self._is_engine_part(name):
            return True
        if name in self.carry_fit_heroes() or (self.carry_settled() and self.feeds_carry(name)):
            return True                     # 适合现任主C / 给主C 涨等级（2026-10-08 主C 定了换配套）
        if self.route_locked() and (name in self.route_stage_members(self._round_seen)
                                    or name in self._sibling_carries()):
            # 路线这个阶段点名要的人、同体系备选路线的主C（2026-09-30 轮换第 9 局：第 3 回合按"同体系主C"
            # 买进沈梦溪，几秒后被当成"不在阵容规划里"卖掉）
            return True
        if self.second_carry(name):
            return False                    # 主C 定了：路线外的输出型不算规划里的人（手里的会被当闲置卖掉换能量）
        cs = self.hero_reviewed.get(name, {}).get("carry_score", 0)
        q = self.pieces.get(name, {}).get("quality", 0)
        cur = self._carry_piece
        cur_cs = self.hero_reviewed.get(cur, {}).get("carry_score", 0) if cur else 0
        return q >= 4 and cs >= 8 and cs > cur_cs and self._carry_eligible(name)

    def lineup_complete(self) -> bool:
        """阵容已成型：主C 定了，场上每个人都是主C / 规划里的人（路线成员、核心、配套、喂主C），没有可换掉的闲人。"""
        if not self.carry_settled() or not self._field:
            return False
        for n in self._field:
            if n == self._carry_piece:
                continue
            if self.dead_weight(n) or not self._planned_hero(n):
                return False
        return True

    def _idle_hand_hero(self, state: GameState) -> Optional[dict[str, Any]]:
        """**用不上的英雄手牌卖掉**（2026-09-23）。

        以前只在手牌快满（9/10）时才卖，平时场上满员、手里压着几张永远上不去的英雄，
        占着手牌位（手牌越挤越读不准）、压着能量。两类：
          ① 过了第 6 回合的 1 阶英雄（弈星这种）：直接卖（太弱，不上场）
          ② 其他英雄：场上满员、换人逻辑也用不上它（不比场上最弱的明显更强）、
             已经在手里放了一整个回合 —— 给换人逻辑留出机会再卖
        都不卖的：**觉醒卡**（在手里卖不掉，只能先上场，2026-09-23）、
        场上同名单位的合成材料、主C、「转瞬」（回合结束自己销毁，卖了也不给能量）。
        """
        seen_now = set()
        for slot in state.hand_slots or []:
            seen_now.add(slot["name"])
            self._hand_since.setdefault(slot["name"], state.round_no)
        if getattr(self, "hand_by_laya", False):
            return None       # f9lx：卖不卖手里的英雄交给 Laya 的 sell_hand 题（手牌爆满另由 decide_sell 兜底，那是合法性）
        for n in list(self._hand_since):
            if n not in seen_now:
                self._hand_since.pop(n, None)
        for n in list(self._hand_seen_ts):
            if n not in seen_now:
                self._hand_seen_ts.pop(n, None)
        full = state.population_valid and state.population >= state.population_cap > 0
        weakest = self._weakest_on_field()
        for slot in reversed(state.hand_slots or []):
            name = slot["name"]
            info = self.pieces.get(name, {})
            if info.get("category") != "英雄牌" or slot.get("awakened"):
                continue
            if self._sell_tries.get((state.round_no, name), 0) >= 2:
                continue      # 本回合卖过两次还在手里：别再试了（实战连卖 6 次）
            if self._sell_decided.get((state.round_no, name), 0) >= 3:
                continue      # **按决策次数也止损**：重复的决策会被执行层 6 秒去重挡掉、根本没点出去，
                              # 只按"点出去的次数"算的话计数永远不涨（实战项羽连着决策了十几次）
            if name in self._field or self._is_carry(name) or name == self._carry_piece:
                continue
            if "转瞬" in info.get("skill", ""):
                continue
            # **不在规划里的英雄手牌就卖**（2026-09-26：未来规划用不上的英雄卡卖掉，不要全堆在手牌里影响判断）。
            # 原来要"在手里放满一整个回合"，前中期 4 阶 / 热门主C 候选一律留着 —— 手里常年堆着几张上不了场的英雄。
            # 规划 = 当前阵容的主C / 核心 + 能顶替现任的主C 人选（≥4 阶、评分 ≥8、比现任主C 高）；场上同名（合成材料）和觉醒卡上面已跳过。
            seen_ts = self._hand_seen_ts.setdefault(name, time.time())
            # 2026-10-12 清查（第 574 / 575 局：换人网络刚判"先留在手里"，5~14 秒后就被下面两条卖掉 —— 赵云 / 朵莉亚 / 扁鹊）：
            # 模型接管时，本回合局面网络判过"不换"的英雄，只走最后那条"上回合判过不换、这回合还上不去"
            _benched_now = (self.model_decides()
                            and (self.__dict__.get("_swap_benched") or {}).get(name) == state.round_no)
            if (self._obsolete(name, state.round_no) and not self._is_engine_part(name)
                    and not self.keep_as_support(name) and name not in (self.min_engine().get("early_priority") or [])):
                # 配套件 / 最低配套件不按"1 阶太弱"卖（2026-10-13 核心阵容审计：第 596、610 局卖了守约路线的李元芳）
                why = f"第 {state.round_no} 回合了，1 阶英雄太弱，不上场"
            elif (_benched_now and full and self.route_locked() and self.carry_settled()
                  and name != self.route_carry() and name not in self.route_core()
                  and name not in (self.min_engine().get("early_priority") or [])
                  and self._hand_since.get(name, state.round_no) < state.round_no
                  and time.time() - seen_ts >= 5):
                # 本回合刚买进来的不卖（2026-10-13 第 607 局：第 11 回合买守约、同回合判"先留着"就卖，−2 能量；下回合升级多 1 人口本可上场）
                # 换人网络本回合判"不上场"，又不是当前路线的主C / 核心件 → 手里留着也上不去，卖掉换能量
                # （2026-10-12 第 585 局：该卖上官婉儿和铠——路线玄策，铠是守约 / 沈梦溪体系的件、上官婉儿只是古币放大器）
                why = "换人网络判它不上场，也不是当前路线的主C / 核心件"
            elif _benched_now:
                continue
            elif full and not self.model_decides() and not self._planned_hero(name) and time.time() - seen_ts >= 5:
                # 审查 10-12（C1）：模型接管时不再按"阵容规划"卖（第 587 局按守约路线把安琪拉阵容的亚连卖了）——
                # 卖不卖交给换人网络（判"不换"之后才走下面那两条）
                why = "场上满员，它不在阵容规划里（不是主C 人选 / 核心 / 合成材料）"
            elif full and name != self.route_carry() and name not in self.route_core() and self.lineup_complete():
                # **阵容已经是规划里的人了，手里别的英雄基本上不了场**（2026-10-09：已经是最优阵容时，
                # 没必要再留其他英雄牌，基本上不了场）。场上同名的合成材料、觉醒卡、主C 在上面已经跳过
                why = "场上满员且都是规划里的人（阵容已成型），这张英雄上不了场"
            elif (full and name != self.route_carry()
                  # 和上面第一条同口径：路线核心件 / 最低配套件不在这里卖（2026-10-13 第 615 局：程咬金是沈梦溪路线的最低配套件，
                  # R13~R14 被卖了 3 张；该不该给他腾位是换人网络的事 —— 题面 f9ln 写出等位置的人在路线里的角色）
                  and name not in self.route_core() and name not in (self.min_engine().get("early_priority") or [])
                  and (self.__dict__.get("_swap_benched") or {}).get(name, 99) < state.round_no
                  and (self.__dict__.get("_swap_asked") or {}).get(name) == state.round_no
                  and self._hand_since.get(name, state.round_no) < state.round_no):
                # 2026-10-13 复盘 ⑦（第 590 局）：原来只看"上回合判了不换"，本回合换人网络还没问就把海诺卖了，
                # 30ms 后本回合的答案是"卖李元芳给海诺腾位"。现在要等本回合也问过（还是"不换"）才卖
                # **局面网络上一回合就判"不换、先留在手里"，这回合还上不去** → 卖（2026-10-09 第 477 局：
                # 堆积大量没必要的卡牌导致反应很慢——满员时手里压着吕布 / 杨玉环 / 鲁班…都算"规划内"，一直不卖）。
                # 上不上场交给局面网络判断，这里只是把它判了"不换"的牌清掉
                why = "场上满员，局面网络上回合判它不换上场，还压在手里"

            else:
                continue
            k = (state.round_no, name)
            self._sell_decided[k] = self._sell_decided.get(k, 0) + 1
            logger.info("卖掉用不上的英雄手牌 %s（%s）", name, why)
            self.note("卖手牌", round=state.round_no, 牌=name, 原因=why)
            return {"type": "sell_card", "piece": name, "src": (slot["x"], slot["y"])}
        return None

    def decide_sell(self, state: GameState) -> Optional[dict[str, Any]]:
        """手牌满了就卖掉最没用的一张（拖到右下角能量圆盘）。

        只在**逼近手牌上限**时才卖：手牌上限 10，超出的开战后会被删除，
        与其被系统删不如换成能量。留下的判据跟买牌一致——
        主C/核心/升级材料/主派系一律不卖。
        """
        if not self.HAND_SELL_ENABLED:
            return None
        if state.shop_open:
            # **商店开着时卖不掉手牌**（2026-09-23 实战：拖到能量圆盘没反应，
            # 后面每帧都产出同一个卖牌动作、被执行层的去重拦下，整帧白过）。关了店再卖。
            return None
        idle = self._idle_hand_hero(state)
        if idle is not None:
            return idle
        dead = self._dead_hand_card(state)          # 规则不会用的 hold 牌 / 这局无效的装备（卡牌审查第 5、8 条）
        if dead is not None:
            return dead
        if len(state.hand_slots) < HAND_MAX - 1:
            return None
        junk = []
        for slot in state.hand_slots:
            name = slot["name"]
            if self._is_carry(name) or self._is_core(name) or name in self._field:
                continue
            if slot.get("awakened"):
                continue                      # 觉醒卡在手里卖不掉，只能先上场（2026-09-23）
            info = self.pieces.get(name, {})
            if self.main_factions and any(f in self.main_factions
                                          for f in info.get("synergies", [])):
                continue
            if info.get("category") in ("棋手技能牌", "效果牌"):
                continue                      # 效果牌/技能牌留着用，不卖
            if "转瞬" in info.get("skill", ""):
                continue                      # 转瞬：回合结束自动销毁且出售不给能量，卖了白费操作
            junk.append((info.get("priority", 0), slot))
        if not junk:
            return None
        junk.sort(key=lambda x: x[0])         # 优先级最低的先卖
        slot = junk[0][1]
        logger.info("手牌 %d/%d 逼近上限，卖掉 %s", len(state.hand_slots), HAND_MAX, slot["name"])
        return {"type": "sell_card", "piece": slot["name"],
                "src": (slot["x"], slot["y"])}

    def decide_auction(self, state: GameState) -> Optional[dict[str, Any]]:
        """拍卖出价（策略：同价先到先得 → 抢早出价，不等抄底）：

        - 目标 = 未拍出拍品中价值最高的（主派系/手里已有=升级材料 加分）
        - 出价 = 当前最高价 + 1，硬上限 auction_max_bid，保留 auction_reserve 能量底线
        - 超上限 → 转拍次优拍品；全超 → 放弃
        - 防自抬：自己出过价且仍是最高价时跳过（对手加价后再追）
        """
        auc = state.auction
        if auc is None:
            return None
        if auc.won:
            return None                 # 已经拍到一件了（顶部「已获得拍品」），每人限一件
        gold = auc.my_gold if auc.my_gold >= 0 else (state.gold or self.last_gold)

        # **本场拍卖出过价的全部记下来**（原来只记最后一张 —— 对猪八戒出价后姬小满的记录被覆盖，
        # 下一帧又对姬小满出价：在几张之间来回，同一张自己给自己抬价；2026-09-23 发现）。
        if state.round_no != self._bids_round:
            self._bids_round, self._bids = state.round_no, {}
        tgt = self.auction_target if self.auction_target and self.auction_target[0] == state.round_no else None
        if tgt is not None and tgt[1] == "":
            return None                 # Laya：这场不拍
        if tgt is None and self.auction_by_laya:
            return None                 # **拍哪件由 Laya 定**（2026-10-11：规则不再影响拍卖目标）：它还没给目标时规则不出价，只做目标的出价机械
        by_piece = {lot.piece: lot for lot in auc.lots if lot.piece}

        def outbid(piece: str) -> bool:
            # **领先者不是我 = 被超了**（2026-09-24：别家的出价数字常读不到、是 0，原来还要求"对方价 > 我的价"，
            # 于是永远判不出被超，出价 1 后就不再跟）。领先者读不出来（空）时不算被超。
            lot = by_piece.get(piece)
            return bool(lot and not lot.sold and lot.top_bidder != "我"
                        and (lot.top_bidder or lot.top_bid > self._bids.get(piece, 0)))

        # **手上有一个出价、又没看到被别人超过，就不再出价**：每人只能拍到一件，
        # 同时挂着几处出价只会白白占住能量。有些拍卖界面**拍品下面根本不显示出价**
        # （出价写在舞台上每个玩家头顶），读到的全是 0 —— 这时"没看到被超过"就当作仍然领先。
        active = [p for p in self._bids
                  if p in by_piece and not by_piece[p].sold and not outbid(p)]
        if active:
            return None
        candidates = []
        for lot in auc.lots:
            if lot.sold or lot.piece is None:
                continue
            if lot.piece in self._bids and not outbid(lot.piece):
                continue      # 出过价、没被超过：绝不自己抬自己
            if lot.top_bidder == "我":
                continue  # **画面上写着我领先**，不抬自己（观测优先于记账）
            if tgt is not None and lot.piece != tgt[1]:
                continue  # Laya 选了目标：只争这一件（出价机械仍由规则做）
            if tgt is None and (self.carry_settled() or self.route_locked()) and (
                    self.unwanted_second_carry(lot.piece) or self.other_c(lot.piece)):
                continue  # 主C 定了：规则自己挑拍品时也不拍第二主C / 别的 C 位（与 Laya 拍卖护栏一致；第 536 局花木兰）
            # 记账兜底（观测不到出价人时还用得上）。
            # **按卡名记，不能按位置记**：拍品卖掉一个、或者滑窗多读出一个，
            # 整排就会重新编号 —— 2026-09-23 实战，我们对 0 号位的司空震出价 1，
            # 下一帧它变成了 1 号位，`lot.index == self._bid_lot` 不成立，
            # 于是**对着同一张牌又加了一次价**，没人跟我们抢，纯粹自己抬自己
            # （2026-09-22 就出现过：没人压价，自己又出了一次价）。
            if lot.piece == self._bid_piece and lot.top_bid <= self._bid_price and not outbid(lot.piece):
                continue
            info = self.pieces.get(lot.piece, {})
            score = info.get("priority", 0)
            if self.main_factions and any(f in self.main_factions
                                          for f in info.get("synergies", [])):
                score += 3
            if lot.piece in self._field or any(lot.piece == h for h in self._hand_last):
                score += 3  # 升级材料
            if state.banned_faction and state.banned_faction in info.get("synergies", []):
                continue
            # 价格惩罚：换目标时必须考虑对方的现价——高价拍品要明显更值才值得跳
            score -= (lot.top_bid + 1) * 0.6
            candidates.append((score, lot))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)

        for score, lot in candidates:
            next_bid = lot.top_bid + 1
            if lot.piece == self._bid_piece and auc.next_price > 0:
                next_bid = auc.next_price        # 按钮上写着的现价（只对当前选中的拍品有效）
            if next_bid > self.auction_max_bid:
                continue                    # 被抬过上限 → 转拍下一个
            if gold - next_bid < self.auction_reserve:
                break                       # 更贵的都拍不起，后面的也不试
            if not auc.preview:
                # 只有**真出价**才记账；预览期只是选中拍品（按钮还没出来），记了就会以为"手上有出价"而不再出
                self._bid_piece = lot.piece
                self._bid_price = next_bid
                self._bids[lot.piece] = next_bid
            # 预览期出价按钮还没出来，这时只把拍品选中（executor 第一步），
            # 等正式开始再出价——否则每帧都白点一次并报"找不到出价按钮"
            return {"type": "auction_bid", "lot": lot.index,
                    "piece": lot.piece, "bid": next_bid,
                    "x": lot.x, "y": lot.y,
                    "select_only": bool(auc.preview)}
        return None

    # ---------- 棋手特性（枫舞强化） ----------

    # 特性取舍是纯游戏理解，做成**显式优先级表**放在 data/lord_boost_priority.json，
    # 改数据即可调整，不用动代码。表里没有的牌退化为关键词打分（下面这组）。
    #
    # 关键教训（2026-09-22 更正）：不能只看"等级/永久"这类字面收益。
    # 「红叶染锋」写着"离场后保留 50% 真实等级给下一个枫叶舞英雄"，字面很强，
    # 但**单核阵容里主C 离场就意味着他死了**，把一半等级给场上另一个凑数英雄毫无意义。
    # 所以"以主C 阵亡/自残为前提"的效果一律压到最低档。
    LORD_BOOST_KEYS = (
        ("护盾", 5), ("伤害加成", 5), ("免疫", 4), ("吸血", 3),
        ("恢复", 3), ("眩晕", 2), ("法力", 2), ("移动速度", 1),
    )
    # 不维护"优先级表"：同一张特性在不同阵容/局势下价值完全不同，必须临场判断。
    # 离线数据（卡牌库里的效果文本）只用来**构造候选描述**喂给 Laya。
    # **不含"离场"**（2026-09-24）：枫叶舞只有**卖掉被绑的英雄**才能重绑，而"卖掉"就是离场 ——
    # 红叶染锋（枫叶舞英雄离场后 50% 真实等级留给下一个枫叶舞英雄）正是换主C 时的搬等级手段：
    # 卖老主C（离场）→ 枫叶舞改绑新主C → 新主C 拿到一半等级。原来连它一起否决，Laya 连选的机会都没有。
    # 09-22 那次纠正针对的是"主C 战死时把等级给凑数英雄"，那种以阵亡为前提的仍然否决。
    LORD_BOOST_VETO = ("阵亡", "死亡")     # 以主C 战死为前提 → 结构性废牌

    def decide_lord_boost(self, state: GameState) -> Optional[dict[str, Any]]:
        """棋手特性三选一（阿离=枫舞强化）。按关键词打分，选最强的一项。

        这个模态不选就一直挂着，会挡住后续所有操作，所以**必须选**——
        就算打分全为 0 也要选第一个，不能返回 None。
        """
        if not state.talents:
            return None
        # 1) 规则层只否决**结构性废牌**（能不能用），不做价值排序
        viable = []
        for opt in state.talents:
            text = (opt.name or "") + (opt.desc or "")
            if self.model_decides() and not _IR_on("lord_boost_veto"):
                viable.append(opt)           # A11（2026-10-15）：否决交给 Laya（题面写着以阵亡为前提 / 主C 已定）
            elif any(k in text for k in self.LORD_BOOST_VETO):
                # 单核阵容下主C 离场就意味着他死了，把等级传给凑数英雄毫无意义
                # （2026-09-22 更正的"红叶染锋"例子）
                logger.info("特性 %r 被否决：以主C 出事为前提", opt.name)
            elif ("离场" in text and "下一个枫叶舞英雄" in text and self.carry_settled()
                  and self._maple_bound == self._carry_piece):
                # 红叶染锋 = 枫叶舞英雄离场后 50% 等级留给下一个 —— 只在"还要换主C"时有用。主C 已定、枫叶舞已绑在主C 身上、
                # 过渡件都卖了，这张就没机会生效（2026-10-08 第 436 局：主C 已定、之前的都卖掉了，这个天赋毫无意义）
                logger.info("特性 %r 被否决：主C %s 已定、枫叶舞已绑在主C 身上，不会再换主C", opt.name, self._carry_piece)
            else:
                viable.append(opt)
        if not viable:
            viable = list(state.talents)     # 全被否决也得选，模态不选会一直挂着挡操作

        # 2) "选哪个"交给 Laya 临场判断，低置信度回退启发式
        best = self._laya_pick(state, viable) if self._laya_pick else None
        if best is None:
            best = max(viable, key=lambda o: sum(
                w for k, w in self.LORD_BOOST_KEYS if k in (o.name or "") + (o.desc or "")))
            logger.info("Laya 未给出高置信选择，按启发式选 %s", best.name)
        logger.info("棋手特性三选一: 选 %s（可选 %s，否决 %d 个）",
                    best.name, [o.name for o in viable],
                    len(state.talents) - len(viable))
        self.record_decision("lord_boost", state, [{"name": o.name} for o in viable], best.name)
        return {"type": "lord_boost_pick", "index": best.index,
                "name": best.name, "x": best.x, "y": best.y}

    # ---------- 天赋 ----------

    # 经济类关键词（前期优先），战力类关键词（后期优先）
    TALENT_ECO_KEYS = ("金币", "能量", "经验", "刷新", "商店", "出售", "升级")
    TALENT_COMBAT_KEYS = ("伤害", "生命", "攻击", "技能", "等级+")

    # **可购买天赋**（扩编名额 / 扩编高峰："获得1个可购买天赋：花费8能量，可上阵人数+1"）：选完要在某个界面再花能量"购买"，
    # 而购买入口 / 按钮还没有实机证据（443/476/479 局只截到右侧天赋列多了一行「扩编高峰」图标，没人点过它）——
    # 没有购买动作之前，选它等于白选（全部日志选过 17 次、买过 0 次）。拿到证据、做出 buy_talent 动作后改成 True。
    CAN_BUY_TALENT = False

    def talent_unsupported(self, name: str, desc: str = "") -> str:
        """这个天赋 agent 现在执行不了（返回原因），或 ""。只读卡面。"""
        txt = (name or "") + (desc or "") + (self._card_text(name) or "")
        if not self.CAN_BUY_TALENT and "可购买天赋" in txt:
            return "要花能量另行购买，agent 还没有购买动作"
        return ""

    def decide_talent(self, state: GameState) -> Optional[dict[str, Any]]:
        """天赋三选一：前期(回合≤8)经济优先，后期战力优先。"""
        if not state.talents:
            return None
        keys = self.TALENT_ECO_KEYS if state.round_no <= 8 else self.TALENT_COMBAT_KEYS
        best_idx, best_score = 0, -1
        for t in state.talents:
            score = sum(1 for k in keys if k in t.desc or k in t.name)
            if self.talent_unsupported(t.name, t.desc):
                score = -0.5           # 做不出来的（要"购买"的天赋）排最后，但三个都是时仍得选一个
            if score > best_score:
                best_idx, best_score = t.index, score
        # **三个都不对路时，刷新一次**（每次升级给一次刷新机会，2026-09-22 确认）。
        # 只在"一个关键词都没命中"时才刷 —— 刷新也可能刷出更差的，不能贪。
        # 刷新次数按**等级**记，不是按回合：天赋是升级时弹的，一级只弹一次。
        if (best_score <= 0 and state.talent_can_refresh
                and state.level != self._talent_refreshed_level):
            self._talent_refreshed_level = state.level
            logger.info("天赋三选一 %s 没有一个对路，用掉这一级的刷新机会",
                        [t.name for t in state.talents])
            self.record_decision("talent", state, [{"name": t.name, "desc": t.desc} for t in state.talents], "refresh")
            return {"type": "talent_refresh"}
        self.record_decision("talent", state, [{"name": t.name, "desc": t.desc} for t in state.talents],
                             state.talents[best_idx].name)
        return {"type": "talent_pick", "index": best_idx,
                "name": state.talents[best_idx].name}

    # ---------- 装备评分（四选一用） ----------

    # (正则键, dps 权重/tank 权重)——按每点数值给分，做性价比代理
    EQUIP_STATS = [
        (r"法术攻击力\s*\+?\s*(\d+)", 1.0, 0.1),
        (r"攻击力\s*\+?\s*(\d+)", 1.0, 0.25),
        (r"攻击速度\s*\+?\s*(\d+)", 2.0, 0.3),       # 百分比数值
        (r"暴击率\s*\+?\s*(\d+)", 2.0, 0.2),
        (r"暴击效果\s*\+?\s*(\d+)", 1.0, 0.1),
        (r"生命值\s*\+?\s*(\d+)", 0.3, 0.8),
        (r"生命\s*\+?\s*(\d+)", 0.3, 0.8),
        (r"护盾", 0.5, 1.2),
        (r"法术防御力\s*\+?\s*(\d+)", 0.4, 1.0),
        (r"物理防御力\s*\+?\s*(\d+)", 0.4, 1.0),
        (r"等级\s*[x×]\s*(\d+)", None, None),        # 按 10 级估：值×10×0.5
    ]

    def score_equipment_option(self, name: str, desc: str, role: str) -> float:
        """按效果文本数值给成品装备打分（性价比代理）。

        role 为装备目标英雄的定位（tank/dps），词条权重不同；
        "唯一"效果打折（不可叠加），对敌减益不计分。
        """
        score = 0.0
        for pattern, w_dps, w_tank in self.EQUIP_STATS:
            w = w_dps if role == "dps" else (w_tank if role == "tank" else 0.5)
            for m in re.finditer(pattern, desc):
                v = float(m.group(1)) if m.groups() else 30.0  # 无数值词条（护盾等）按固定值
                if w_dps is None:
                    score += v * 10 * 0.5
                else:
                    score += v * w
        if "唯一" in desc:
            score *= 0.75
        return score

    # ---------- 主入口 ----------

    def _simulate_cost(self, act: dict[str, Any], state: GameState) -> None:
        """同一帧里先扣掉这笔花销，避免把同一份能量花两次。"""
        if act["type"] == "buy_piece" and act.get("piece"):
            state.gold -= self.pieces.get(act["piece"], {}).get("cost", 0) or 0
        elif act["type"] == "refresh_shop":
            state.gold -= act.get("cost", 0)
        elif act["type"] == "buy_xp":
            state.gold -= max(0, state.level_cost)

    def equip_count_hint(self, cell) -> int:
        """这一格看到过的装备件数（_equip_seen，视觉读数的最大值）和穿戴账本里记的件数，取大的。"""
        k = cell_key(tuple(cell))
        n = (self._equip_seen or {}).get(k, 0) or 0
        who = self._cell_owner.get(tuple(cell))
        return max(n, len((self._worn or {}).get(who, ()) or ()))

    def equip_role_for(self, name: str) -> str:
        """装备四选一按谁的需要挑：输出位返回 "carry:<名字>"（按伤害类型选输出装，agent/decision/hero_traits），
        坦克/辅助返回 "tank"。输出位 = 主C，或主C 评分 ≥5 且不是真坦克（标签「坦克」）的英雄。"""
        if not name:
            return "tank"
        if name in self.SHARE_EQUIP:
            return "share:" + name          # 瑶：首件普通装备开团分享给 3 名友军 → 四选一按 SHARE_EQUIP 挑（破茧之衣）
        true_tank = any("坦克" in x for x in self.pieces.get(name, {}).get("tags", []))
        score = self.hero_reviewed.get(name, {}).get("carry_score", 0) or 0
        if name == self._carry_piece:
            # 2026-10-13 审计第 5 条（608/610/613 局）：临时主C 干将（法术）铸出虚无法杖 / 贤者之书，物理路线主C 接班后
            # 这些装备退回手牌压 5~6 回合。临时主C 的四选一按路线主C 的伤害类型选（装备迟早要转给最终主C）
            rc = self.route_carry()
            if rc and rc != name and self.is_temp_carry(name):
                from agent.decision import hero_traits as _ht0
                if _ht0.dmg_type(rc, legacy=False) in ("phys", "mag") and _ht0.dmg_type(rc, legacy=False) != _ht0.dmg_type(name, legacy=False):
                    return "carry:" + rc
            return "carry:" + name
        base = "carry" if (score >= 5 and not true_tank) else "tank"
        cp = self._field.get(self._carry_piece) if self._carry_piece else None
        if cp is not None and self.equip_count_hint(cp) >= 3:
            # **主C 满装后给别人的装备：四选一挑辅助性质的**（2026-10-10：多余装备选辅助性质的，比如开局眩晕对手的那种）
            return f"util:{name}|{base}"
        return ("carry:" + name) if base == "carry" else "tank"

    def _is_tank(self, name: str) -> bool:
        """站前排、扛伤的：按卡面文本分析的定位（tank / bruiser / controller 站前排）；hero_roles 缺项才看标签。
        （2026-09-30：不通过现有标签定位角色——周瑜/大司命标签是坦克，实际是范围法师/物理战士）"""
        hr = self.hero_role(name)
        if hr:
            roles = hr.get("combat_role") or []
            return bool(roles) and roles[0] in ("tank", "bruiser") or (roles[:1] == ["controller"] and "tank" in roles)
        info = self.pieces.get(name, {})
        return info.get("role") == "tank" or any("前排" in x or "坦克" in x for x in info.get("tags", []))

    def _buddy_joint_plan(self, state: GameState, occupied: set, vis: dict) -> Optional[dict[str, Any]]:
        """**两个连接类英雄抢主C 旁边的格子**（2026-10-08 第 434 局：辅助连接关系错了——主C 沈梦溪在角落只有 2 个邻格，
        李元芳占着一个，明世隐放哪都和李元芳并列 → 连到李元芳）。李元芳只连【河洛】，可以退到"同派系里主C 仍最近"的远一点的格子，
        把主C 旁边让给明世隐这类连任何人的。做法：先挪李元芳到 S（在 S 仍连主C），下一帧把明世隐挪到 T（主C 唯一最近）。"""
        carry = self._carry_piece
        cpos = tuple(self._field.get(carry) or ()) if carry else ()
        if not cpos:
            return None
        from ..perception.extractors import OWN_BOARD_ROWS, POP_OCCLUDING_CELLS
        free = [tuple(c) for row in OWN_BOARD_ROWS for c in row
                if tuple(c) not in occupied and tuple(c) not in set(POP_OCCLUDING_CELLS)
                and not self._cell_dead(c) and not self._covered_by_shop(state, c)]
        links = self.__dict__.get("_buddy_link") or {}
        for u, upos in self._field.items():
            if u == carry or not self._carry_buddy(u) or self._buddy_faction(u) or not self._link_sticky(u):
                continue                     # 开战现算最近友军的（太乙）"挪开再挪回"无效（审查 2026-10-11）
            upos = tuple(upos)
            uo = self._buddy_others(u, vis)
            lk = links.get(u)
            if lk == (carry, cpos) or (lk is None and self._buddy_cell_score(upos, cpos, uo) == (0, 0)):
                continue                     # 已经连着主C（并列也保持）
            for lname, lpos in self._field.items():
                lpos = tuple(lpos)
                if lname in (u, carry) or not self._buddy_faction(lname) or self._hex_dist(lpos, cpos) != 1:
                    continue
                uo_wo = [q for q in uo if q != lpos]
                lo = self._buddy_others(lname, vis)
                # 李元芳只比同派系：它回到原格时主C 仍是同派系里唯一最近的 → 连接不受明世隐影响
                if self._buddy_cell_score(lpos, cpos, lo)[0] != 0:
                    continue
                for t in [c for c in free if c != upos and self._hex_dist(c, cpos) == 1]:
                    if self._buddy_cell_score(t, cpos, uo_wo) != (0, 0):
                        continue
                    # 三步：李元芳先挪到远处 → 明世隐挪到 t（此时主C 唯一最近，连主C）→ 李元芳挪回原格（同派系里主C 最近，仍连主C）
                    far = [c for c in free if c != t and self._hex_dist(c, t) >= 2]
                    if not far:
                        continue
                    sc = max(far, key=lambda c: self._hex_dist(c, t))
                    self._reposition_round = state.round_no
                    self._buddy_return = [(state.round_no, u, t), (state.round_no, lname, lpos)]
                    self._evict_cell(sc, lname)
                    self._cell_owner.pop(lpos, None)
                    self._field[lname] = sc
                    self._cell_owner[sc] = lname
                    logger.info("刷新连接：%s 先挪开到 %s，%s 挪到 %s 连主C，%s 再挪回 %s", lname, sc, u, t, lname, lpos)
                    self.note("刷新连接", 英雄=lname, 让位给=u, 到=f"{t[0]},{t[1]}")
                    return {"type": "move_unit", "piece": lname, "src": lpos, "dst": sc}
        return None

    def _aura_fix(self, state: GameState, occupied: set, vis: dict) -> Optional[dict[str, Any]]:
        """**光环类（"周围 N 格 / 相邻的友军"，如蔡文姬）离主C 超过 N 格 → 挪进 N 格以内**（2026-10-11 站位表：蔡文姬在主C 1 格内 0%）。
        只挪画面上这一帧确实看到站着英雄的格子；不挪主C、不挪"最近友军"类（另有逻辑）。"""
        carry = self._carry_piece
        cpos = tuple(self._field.get(carry) or ()) if carry else ()
        if not cpos:
            return None
        from ..perception.extractors import OWN_BOARD_ROWS, POP_OCCLUDING_CELLS
        for name, pos in self._field.items():
            pos = tuple(pos)
            if name == carry or self._carry_buddy(name) or not self._aura_near(name):
                continue
            r = self.aura_radius(name)
            if self._hex_dist(pos, cpos) <= r or vis.get(cell_key(pos)) != "hero" or self._cell_dead(pos):
                continue
            free = [tuple(c) for row in OWN_BOARD_ROWS for c in row
                    if tuple(c) not in occupied and tuple(c) not in set(POP_OCCLUDING_CELLS)
                    and tuple(c) not in self.object_cells and not self._cell_dead(c)
                    and not self._covered_by_shop(state, c) and self._hex_dist(tuple(c), cpos) <= r]
            if not free:
                continue
            dst = min(free, key=lambda c: (self._hex_dist(c, cpos), -c[1]))
            self._reposition_round = state.round_no
            self._evict_cell(dst, name)
            self._cell_owner.pop(pos, None)
            self._field[name] = dst
            self._cell_owner[dst] = name
            logger.info("光环站位：%s（周围 %d 格）离主C %s 太远，从 %s 挪到 %s", name, r, carry, pos, dst)
            self.note("光环站位", 英雄=name, 从=f"{pos[0]},{pos[1]}", 到=f"{dst[0]},{dst[1]}")
            return {"type": "move_unit", "piece": name, "src": pos, "dst": dst}
        return None

    def _clear_for_buddy(self, state: GameState, occupied: set, vis: dict) -> Optional[dict[str, Any]]:
        """**主C 旁边没空格时，给"开战选最近友军"的英雄腾位**（2026-10-10 第 515 局：瑶必须在主C 旁边，少司缘不需要）：
        虞姬在角落 (276,501)，三个邻格被 少司缘 / 鬼谷子 / 张良 占满，瑶只能落在远处 (411,372)。
        做法：把主C 邻格里一个不需要贴主C 的人（不是贴主C 英雄、不是主C）挪到远处空格，下一帧原有的贴主C 挪位把瑶挪进来。
        优先挪：辅助 > 其他非坦克 > 坦克（坦克挡在主C 前面有用）。只挪视觉这一帧确实看到站着英雄的格子。"""
        carry = self._carry_piece
        cpos = tuple(self._field.get(carry) or ()) if carry else ()
        if not cpos:
            return None
        from ..perception.extractors import OWN_BOARD_ROWS, POP_OCCLUDING_CELLS, OWN_BOARD_DROPS_FRONT, OWN_BOARD_DROPS_BACK
        free = [tuple(c) for row in OWN_BOARD_ROWS for c in row
                if tuple(c) not in occupied and tuple(c) not in set(POP_OCCLUDING_CELLS)
                and tuple(c) not in self.object_cells and not self._cell_dead(c) and not self._covered_by_shop(state, c)]
        adj_free = [c for c in free if self._hex_dist(c, cpos) == 1]
        if adj_free:
            return None                                   # 有空邻格：原有逻辑会把贴主C 英雄挪进来
        for u, upos in self._field.items():
            if u == carry or not self._carry_buddy(u) or self._hex_dist(tuple(upos), cpos) == 1:
                continue
            cands = [(n, tuple(q)) for n, q in self._field.items()
                     if n not in (u, carry) and not self._wants_carry_side(n) and self._hex_dist(tuple(q), cpos) == 1
                     and vis.get(cell_key(tuple(q))) == "hero" and not self._cell_dead(q)]
            if not cands:
                continue
            cands.sort(key=lambda x: (0 if self._is_support(x[0]) else (2 if self._is_tank(x[0]) else 1), x[0]))
            name, src = cands[0]
            lane = OWN_BOARD_DROPS_FRONT if self._is_tank(name) else list(self.BACK_MID_CELLS) + list(OWN_BOARD_DROPS_BACK)
            dst = next((tuple(c) for c in list(lane) + free if tuple(c) in free and self._hex_dist(tuple(c), cpos) >= 2
                        and tuple(c) != tuple(upos)), None)
            if dst is None:
                return None
            self._reposition_round = state.round_no
            self._buddy_return = [(state.round_no, u, src)]     # 下一帧把贴主C 英雄挪进腾出的格子（这条在"每回合一次"检查之前执行）
            self._evict_cell(dst, name)
            self._cell_owner.pop(src, None)
            self._field[name] = dst
            self._cell_owner[dst] = name
            logger.info("给 %s 腾主C 旁边的格子：%s 从 %s 挪到 %s（主C %s 的邻格都满了）", u, name, src, dst, carry)
            self.note("腾位贴主C", 让位=name, 给=u, 从=f"{src[0]},{src[1]}", 到=f"{dst[0]},{dst[1]}")
            return {"type": "move_unit", "piece": name, "src": src, "dst": dst}
        return None

    BUDDY_TIE_MOVES_PER_ROUND = 3

    def _clear_buddy_ties(self, state: GameState, occupied: set, vis: dict) -> Optional[dict[str, Any]]:
        """**跟主C 无关的英雄站到另一边的角落**（2026-10-12 第 583 局：主C 和明世隐的站位乱了，主要因为有个干将；
        跟主C 无关的英雄应站另一边的角落）：守约 (955,502)、明世隐 (839,502) 挨着，但干将 (917,570)、李信、盾山也都和明世隐一样近 →
        主C 不是明世隐唯一最近的人，连接随机。做法：作用于最近友军的英雄在场时，离它不比主C 远的无关英雄（不需要贴主C 的）
        挪到离主C 最远的空格；挪完原有的"贴主C"逻辑会把明世隐重新连上主C。每回合最多挪 BUDDY_TIE_MOVES_PER_ROUND 次。"""
        carry = self._carry_piece
        cpos = tuple(self._field.get(carry) or ()) if carry else ()
        if not cpos:
            return None
        moves = self.__dict__.setdefault("_tie_moves", {})
        if moves.get(state.round_no, 0) >= self.BUDDY_TIE_MOVES_PER_ROUND:
            return None
        from ..perception.extractors import OWN_BOARD_ROWS, POP_OCCLUDING_CELLS, OWN_BOARD_DROPS_FRONT
        free = [tuple(c) for row in OWN_BOARD_ROWS for c in row
                if tuple(c) not in occupied and tuple(c) not in set(POP_OCCLUDING_CELLS)
                and tuple(c) not in self.object_cells and not self._cell_dead(c) and not self._covered_by_shop(state, c)]
        if not free:
            return None
        for u, upos in self._field.items():
            upos = tuple(upos)
            if u == carry or not self._carry_buddy(u) or vis.get(cell_key(upos)) != "hero":
                continue
            d_c = self._hex_dist(upos, cpos)
            if d_c > 1:
                # 2026-10-21 第 899 局：主C 从 (917,570) 挪到 (681,571) 后，李元芳还留在原处 (955,502)，
                # 这条先把上官婉儿 / 盾山 / 程咬金 / 铠一个个挪开（铠离开钟馗左右），R13 李元芳才自己贴过去。
                # 辅助离主C 不止 1 格 → 该挪的是辅助本人（站位纠正"贴主C"），别挪一圈无关的人
                continue
            fac = self._buddy_faction(u)
            _zks = set(self._row_sides(self._field["钟馗"])) if "钟馗" in self._field else set()
            blockers = [(n, tuple(q)) for n, q in self._field.items()
                        if n not in (u, carry) and not self._wants_carry_side(n)
                        # 2026-10-21 第 899 局（牺牲件要站钟馗左右两侧）：牺牲件站在钟馗左右是有用的站位，不为让开辅助挪走
                        and not (self._is_sacrifice(n) and tuple(q) in _zks)
                        and (not fac or fac in (self.pieces.get(n, {}).get("synergies") or []))
                        and self._hex_dist(tuple(q), upos) <= d_c
                        and vis.get(cell_key(tuple(q))) == "hero" and not self._cell_dead(q)]
            if not blockers:
                continue
            # 先挪非坦克（坦克站前排挡刀有用，挪也只挪到前排远端），离明世隐近的先挪
            blockers.sort(key=lambda x: (self._is_tank(x[0]), self._hex_dist(x[1], upos), x[0]))
            name, src = blockers[0]
            lane = [tuple(c) for c in OWN_BOARD_DROPS_FRONT] if self._is_tank(name) else free
            cand = [c for c in free if c in set(lane) and self._hex_dist(c, upos) > d_c] or                    [c for c in free if self._hex_dist(c, upos) > d_c]
            if not cand:
                continue
            cand = [c for c in cand if self._move_ok(state, name, src, c)]
            if not cand:
                continue
            # 2026-10-21 第 894 局（主C 旁的辅助和钟馗旁的人反复挪位置）：
            # 让开时优先挑同时满足"站位纠正"的格子——带牺牲的贴钟馗（否则下一帧又被挪回来，铠在两格之间跳了 6 次）
            _zk = self._field.get("钟馗") if self._is_sacrifice(name) else None
            _nz = set(self._row_sides(tuple(_zk))) if _zk else set()      # 钟馗只给"自身及左右 1 格"，不含斜对角
            dst = max(cand, key=lambda c: (c in _nz, self._hex_dist(c, cpos), self._hex_dist(c, upos), -c[0]))
            self._note_move(state, name, src, dst)
            moves[state.round_no] = moves.get(state.round_no, 0) + 1
            self._evict_cell(dst, name)
            self._cell_owner.pop(src, None)
            self._field[name] = dst
            self._cell_owner[dst] = name
            logger.info("站位：%s 和主C %s 一样靠近 %s（作用于最近友军），挪到另一边 %s（从 %s）", name, carry, u, dst, src)
            self.note("无关英雄让开", 英雄=name, 给=u, 从=f"{src[0]},{src[1]}", 到=f"{dst[0]},{dst[1]}")
            return {"type": "move_unit", "piece": name, "src": src, "dst": dst}
        return None

    MOVES_PER_UNIT_ROUND = 2
    def msy_entries(self) -> int:
        """明世隐登场过几次：上场 1 次 + 合成的同名（觉醒进度）+ 唤醒 / 集体唤醒 / 锦囊用在他身上 + 觉醒回手重上 1 次；不在场 = 0。"""
        if "明世隐" not in self._field:
            return 0
        aw = self.is_awakened("明世隐")
        merged = 3 if aw else int(self._awaken_seen.get("明世隐") or 0)
        trig = sum(1 for e in self.ledger if e.get("event") == "用牌" and e.get("目标") == "明世隐"
                   and e.get("牌") in ("唤醒", "集体唤醒", "锦囊"))
        return 1 + merged + trig + (1 if aw else 0)

    def msy_conversion(self) -> int:
        """明世隐每回合能把多少临时等级转成永久（2026-10-21 更正：每次登场这个值累加 2，觉醒后累加 4）：
        卡面「临时等级中的 2 级可转化；登场：可转化的等级 +2」（觉醒：4 级、登场 +4）——每次登场都累加。
        登场次数 ≈ 上场 1 次 + 合成的同名（觉醒进度，徽章外圈）+ 唤醒 / 集体唤醒 / 锦囊用在他身上 + 觉醒回手后重新上场 1 次。"""
        if "明世隐" not in self._field:
            return 0
        aw = self.is_awakened("明世隐")
        merged = 3 if aw else int(self._awaken_seen.get("明世隐") or 0)
        trig = sum(1 for e in self.ledger if e.get("event") == "用牌" and e.get("目标") == "明世隐"
                   and e.get("牌") in ("唤醒", "集体唤醒", "锦囊"))
        entries = 1 + merged + trig + (1 if aw else 0)
        per = 4 if aw else 2
        return per + per * entries

    def _blocks_carry_buddy(self, name: str, cell) -> bool:
        """站在 cell 会不会和主C 一样靠近"作用于最近友军"的英雄（明世隐等）→ 抢走它的连接（同"无关英雄让开"的判据）。"""
        carry = self._carry_piece
        cpos = tuple(self._field.get(carry) or ()) if carry else ()
        if not cpos or name == carry:
            return False
        for u, upos in self._field.items():
            if u in (carry, name) or not self._carry_buddy(u):
                continue
            if self._hex_dist(tuple(cell), tuple(upos)) <= self._hex_dist(cpos, tuple(upos)):
                return True
        return False

    def _move_ok(self, state, name: str, src, dst) -> bool:
        """防来回挪（2026-10-21 第 894 局：铠在 (783,438) / (346,430) 之间被两条站位规则各挪 3 次）：
        同一回合一个单位不挪回它这回合待过的格子，最多挪 MOVES_PER_UNIT_ROUND 次。"""
        h = (self.__dict__.setdefault("_move_hist", {})).get((state.round_no, name)) or []
        if tuple(dst) in {tuple(x) for x in h}:
            return False
        return len(h) < self.MOVES_PER_UNIT_ROUND + 1     # h 里含起点，挪 2 次 = 3 个格子

    def _note_move(self, state, name: str, src, dst) -> None:
        d = self.__dict__.setdefault("_move_hist", {})
        h = d.setdefault((state.round_no, name), [])
        if not h:
            h.append(tuple(src))
        h.append(tuple(dst))

    BACK_MID_CELLS = ((499, 500), (720, 500), (617, 496), (786, 569), (555, 569), (681, 571))   # 后排中间偏左右（不贴主C 的辅助）；(681,571)/(555,569) 留给因缘图腾，排最后
    CARRY_HOME = (300, 569)                          # 主C 放左下角（2026-10-10）
    CARRY_HOME_CROWDED = (390, 502)                  # 要贴主C 的人 ≥2 个：倒数第二行偏左，6 个邻格（左下角只有 3 个）

    def carry_home(self, state: GameState) -> tuple:
        """主C 的落点锚：场上 + 手里要贴主C 的人（最近友军类 / 周围 N 格光环 / 图腾光环天赋）≥2 个 → 倒数第二行偏左 (390,502)，
        否则左下角。（2026-10-11 站位表：主C 在左下角时两个连接类都成立的摆法只有 4 种，在 (390,502) 有几十种）"""
        names = set(self._field) | {sl.get("name") for sl in state.hand_slots or [] if sl.get("name")}
        n = sum(1 for x in names if x and x != self._carry_piece
                and self.pieces.get(x, {}).get("category") == "英雄牌" and self._wants_carry_side(x))
        if self.totem_aura_radius() and any(h in names for h, _t, _c in self._totem_plan()):
            n += 1                         # 有图腾光环天赋、且场上 / 手里有召唤图腾的人
        return self.CARRY_HOME_CROWDED if n >= 2 else self.CARRY_HOME
    RANGED_HOME = (1027, 572)                        # 其他后排远程放右下角一侧
    TOTEM_CORNERS = ((681, 571), (555, 569))        # 姻缘（全队涨级）图腾：后排中间（2026-10-10 改：原来是后排两角，和主C 抢左下角）
    OPENER_TOTEM_CORNERS = ((319, 375), (918, 368))  # 其他图腾：前排两侧（2026-10-10）

    def _totem_plan(self) -> list:
        """[(召唤者英雄, 图腾名, 目标角落)]：只读卡面，不写名字。
        - 给全队涨级的图腾（召唤物卡面"…等级永久+N"）→ 后排角（TOTEM_CORNERS），站前排会被打死、吃不到涨级（2026-10-10）；
        - 只有「开团：」效果的图腾（开场给一次就没用了，如先知图腾）→ 前排左角（OPENER_TOTEM_CORNERS），把后排让给主C / 涨级图腾（同日）。"""
        if RuleEngine.__dict__.get("_TOTEM_PLAN") is None:
            plan = []
            for t, v in self.pieces.items():
                if v.get("category") != "召唤物":
                    continue
                sk = (v.get("skill", "") or "").strip()
                if re.search(r"所有己方英雄等级永久\+", sk):
                    corners = self.TOTEM_CORNERS            # 给全队涨级（因缘图腾）：后排中间，护住它
                elif "图腾" in t:
                    corners = self.OPENER_TOTEM_CORNERS     # 其他图腾（先知 / 往生…）：前排两侧
                else:
                    continue
                for h, hv in self.pieces.items():
                    if hv.get("category") == "英雄牌" and f"召唤{t}" in (hv.get("skill", "") or ""):
                        plan.append((h, t, tuple(corners)))
            plan.sort(key=lambda x: 0 if x[2] == tuple(self.TOTEM_CORNERS) else 1)    # 涨级图腾先挪
            RuleEngine._TOTEM_PLAN = plan
        return RuleEngine._TOTEM_PLAN

    TOTEM_AURA_RE = re.compile(r"图腾周围(\d+)格的己方单位")

    def totem_aura_radius(self) -> int:
        """已选天赋里"图腾周围 N 格的己方单位 …"的 N（只读天赋卡面，不写天赋名；如「神明庇佑」：开战时图腾周围 1 格己方单位
        +10% 生命上限、+20% 攻速，可叠加）。没有 → 0。"""
        n = 0
        for t in self._talents_taken:
            m = self.TOTEM_AURA_RE.search(self._card_text(t) or self.pieces.get(t, {}).get("skill") or "")
            if m:
                n = max(n, int(m.group(1)))
        return n

    def _totem_aura_cells(self) -> tuple:
        """有图腾光环天赋、主C 在场 → 主C 周围 N 格内的格子（近的先、后排先）；否则空。"""
        r, C = self.totem_aura_radius(), self._carry_piece
        if not r or not C or C not in self._field:
            return ()
        from ..perception.extractors import OWN_BOARD_ROWS
        cc = tuple(self._field[C])
        cells = [tuple(c) for row in OWN_BOARD_ROWS for c in row]
        ring = [c for c in cells if c != cc and self._hex_dist(cc, c) <= r]
        return tuple(sorted(ring, key=lambda c: (self._hex_dist(cc, c), -c[1], c[0])))

    def decide_totem_corner(self, state: GameState) -> Optional[dict[str, Any]]:
        """**图腾站位**（2026-10-10：少司缘的因缘图腾放角落，否则会被打死；先知图腾开场后没有其他技能，放前排角落即可）。
        图腾点开不弹英雄详情、认人模型也没有它，位置靠 D-FINE 的 summon 类：召唤者在场时，离召唤者最近、还没认领的 summon 格就是它。
        挪完把目标格记成物件格（object_cells），上场不会往那儿放人。每回合最多试 2 次。"""
        plan = [(h, t, cs) for h, t, cs in self._totem_plan() if h in self._field]
        if not plan:
            return None
        aura = self._totem_aura_cells()
        if aura:
            # 选了"图腾周围 N 格的己方单位得到加成"的天赋（2026-10-11：「神明庇佑」—— 选了这个天赋后的站位）
            # → 所有图腾都围着主C 放，加成叠加在主C 身上；不再按"涨级图腾后排 / 开团图腾前排"分角落
            plan = [(h, t, aura) for h, t, _ in plan]
        sc = []
        for k, cls in (state.vision_units or {}).items():
            if cls != "summon":
                continue
            try:
                x, y = k.split(",")
                sc.append((int(x), int(y)))
            except ValueError:
                continue
        if not sc:
            return None
        # **只挪点开确认过身份的召唤物**（2026-10-13 第 605 局：按"离召唤者最近"猜身份，场上同时有因缘图腾 / 先知图腾 / 机关傀儡，
        # 把因缘图腾当成先知图腾挪到了前排角落，因缘图腾本该放后排）。没确认的由看场点开（totem_inspect_cells）
        conf = self.__dict__.setdefault("_obj_confirmed", {})
        claimed: dict = {}                       # 格子 -> 图腾名（已在自己的角落里的先认领）
        todo = []
        for h, t, cs in plan:
            here = next((c for c in sc if c in cs and c not in claimed and conf.get(c) == t), None)
            if here:
                claimed[here] = t
                self.object_cells[here] = t
            else:
                todo.append((h, t, cs))
        tries = self.__dict__.setdefault("_totem_tries", {})
        if not todo or tries.get(state.round_no, 0) >= 2:
            return None
        occupied = {tuple(v) for v in self._field.values()} | set(sc) | _pop_occluding()   # 挡人口数字的四格不放人（2026-10-19 第 738 局）
        occupied |= {tuple(c) for c, n in self.object_cells.items()}
        for k in (state.unit_anchors or {}):
            try:
                x, y = k.split(",")
                occupied.add((int(x), int(y)))
            except ValueError:
                continue
        for h, t, cs in todo:
            hp = tuple(self._field[h])
            cand = [c for c in sc if conf.get(c) == t and c not in claimed]
            if not cand:
                continue                            # 还没认出是哪个召唤物：等看场点开确认
            src = min(cand, key=lambda c: (c[0] - hp[0]) ** 2 + (c[1] - hp[1]) ** 2)
            dst = next((c for c in cs if c not in occupied and not self._cell_dead(c)
                        and not self._covered_by_shop(state, c)), None)
            if dst is None:
                if self._skip_logged.get(("totem_corner", t, state.round_no)) is None:
                    self._skip_logged[("totem_corner", t, state.round_no)] = 1
                    logger.info("%s 在 %s，目标格 %s 都有人，暂时挪不了", t, src, list(cs))
                continue
            tries[state.round_no] = tries.get(state.round_no, 0) + 1
            self.object_cells.pop(src, None)
            self.object_cells[dst] = t
            conf.pop(src, None)
            conf[dst] = t
            logger.info("图腾站位：%s 从 %s 挪到 %s（召唤者 %s%s）", t, src, dst, h,
                        f"；图腾光环天赋 → 贴主C {self._carry_piece}" if aura else "，角落")
            self.note("图腾站位", 图腾=t, 从=f"{src[0]},{src[1]}", 到=f"{dst[0]},{dst[1]}")
            return {"type": "move_unit", "piece": t, "src": src, "dst": dst}
        return None

    def totem_inspect_cells(self, state: GameState) -> list:
        """要挪的图腾还没确认在哪格时，视觉标成 summon、身份没确认的格子（交给看场点开读名字）。"""
        plan = [t for h, t, cs in self._totem_plan() if h in self._field]
        if not plan:
            return []
        conf = self.__dict__.setdefault("_obj_confirmed", {})
        out = []
        for k, cls in (state.vision_units or {}).items():
            if cls != "summon":
                continue
            try:
                x, y = (int(v) for v in k.split(","))
            except ValueError:
                continue
            if (x, y) not in conf:
                out.append((x, y))
        return out if any(t not in conf.values() for t in plan) else []

    # 古币机挪去的格子：A7、B7 这两格英雄本来就不站（站人时头顶徽章压住人口数字，POP_OCCLUDING_CELLS），
    # 古币机没有徽章、又在人口数字（y 235~305）下方 —— 放在这里等于白腾出一个英雄格。A6/B6 太靠近人口数字（y≈316 的机身会压上去），不用。
    COIN_MACHINE_HOME = ((980, 430), (885, 437))

    def decide_coin_machine_move(self, state: GameState) -> Optional[dict[str, Any]]:
        """**古币机挪到不碍事的地方**（2026-10-12）。
        它常被召唤在中间的英雄格上；挪到 COIN_MACHINE_HOME（英雄本来不用的格子）。每回合最多 1 次、每局最多 3 次；
        挪完先按新格子记，live_match_agent 的模板复查会在几帧内纠正（没挪成就认回旧格）。"""
        c = tuple(self.coin_machine_cell or ())
        if not c or c in self.COIN_MACHINE_HOME or self._round_seen <= 0:
            return None
        tries = self.__dict__.setdefault("_coin_move_tries", {})
        if tries.get(self._round_seen) or sum(tries.values()) >= 3:
            return None
        occupied = {tuple(v) for v in self._field.values()} | set(self.object_cells_all()) | self._summon_cells(state)   # 古币机本来就挪进挡人口格（英雄不用的格子）
        for k in (state.unit_anchors or {}):
            try:
                x, y = k.split(",")
                occupied.add((int(x), int(y)))
            except ValueError:
                continue
        dst = next((d for d in self.COIN_MACHINE_HOME if d not in occupied and not self._cell_dead(d)
                    and not self._covered_by_shop(state, d)), None)
        if dst is None:
            return None
        tries[self._round_seen] = 1
        self.object_cells.pop(c, None)
        self.object_cells[dst] = "古币机"
        self.coin_machine_cell = dst
        logger.info("古币机从 %s 挪到 %s（英雄不用的格子，腾出原来的英雄格）", c, dst)
        self.note("古币机挪位", 从=f"{c[0]},{c[1]}", 到=f"{dst[0]},{dst[1]}")
        return {"type": "move_unit", "piece": "古币机", "src": c, "dst": dst}

    def decide_reposition(self, state: GameState) -> Optional[dict[str, Any]]:
        """**站位纠正**：坦克在后排、前排有空格 → 挪到前排；主C 在前排、后排有空格 → 挪到后排角落。每回合最多挪一个。

        2026-09-24：坦克牛魔站到了后排。上场时的落点排序是对的（坦克前排、其他后排角落），
        但抓错牌会把人放进别人的格子（想放孙膑到后排 (300,569)，上场的是牛魔），之后就一直站在那儿。
        只挪**本回合点开确认过位置**的人（`_confirmed_at`）—— 账本过期时按旧坐标去抓，抓到的是别人。
        """
        if state.phase != "prepare" or self._shop_blocks_board(state):
            return None
        ret = self.__dict__.get("_buddy_return")
        if isinstance(ret, list):
            ret = [x for x in ret if x[0] == state.round_no]
            self._buddy_return = ret[1:] or None
            ret = ret[0] if ret else None
        else:
            self._buddy_return = None
        if ret and ret[0] == state.round_no:
            # 刷新连接的后续步骤：挪开之后再挪到目标格（2026-10-08：李元芳先挪到别处再挪回来，否则连接关系不刷新）
            _, name, home = ret
            pos = self._field.get(name)
            if pos and tuple(pos) != tuple(home) and tuple(home) not in {tuple(v) for v in self._field.values()}:
                self._cell_owner.pop(tuple(pos), None)
                self._field[name] = tuple(home)
                self._cell_owner[tuple(home)] = name
                logger.info("刷新连接：%s 挪到 %s", name, home)
                return {"type": "move_unit", "piece": name, "src": tuple(pos), "dst": tuple(home), "keep_link": True}
        tot = self.decide_totem_corner(state)
        if tot:
            return tot
        cm = self.decide_coin_machine_move(state)
        if cm:
            return cm
        # 每回合原来只挪 1 次：第 625 局 R8 先挪了李元芳，明世隐（也贴主C）整回合留在远处，只能手动挪。
        # 贴主C 的英雄还没贴好时，本回合最多挪 3 次（每次挪完仍是一帧一个动作）
        if self._reposition_round == state.round_no:
            _pending_buddy = any(self._carry_buddy(n) and n != self._carry_piece for n in self._field)
            _moved = sum(1 for e in self.ledger if e.get("event") == "站位纠正" and e.get("round") == state.round_no)
            if not _pending_buddy or _moved >= 3:
                return None
        from ..perception.extractors import OWN_BOARD_ROWS, OWN_BOARD_DROPS_FRONT, OWN_BOARD_DROPS_BACK
        front = {tuple(c) for c in OWN_BOARD_ROWS[0] + OWN_BOARD_ROWS[1]}
        back = {tuple(c) for c in OWN_BOARD_ROWS[2] + OWN_BOARD_ROWS[3]}
        occupied = ({tuple(v) for v in self._field.values()} | self.object_cells_all() | self._summon_cells(state)   # 图腾 / 物件格也算占用（审查 2026-10-11）
                    | _pop_occluding())   # 挡人口数字的四格（2026-10-19：第 738 局「牺牲贴钟馗」把蒙恬挪到 (818,374)，8/8 读成 3/8）
        for key in (state.unit_anchors or {}):
            try:
                cx, cy = key.split(",")
                occupied.add((int(cx), int(cy)))
            except ValueError:
                continue
        seen_now = set()
        for key in (state.unit_anchors or {}):
            try:
                cx, cy = key.split(",")
                seen_now.add((int(cx), int(cy)))
            except ValueError:
                continue
        vis = state.vision_units or {}
        plan = self._buddy_joint_plan(state, occupied, vis)
        if plan:
            return plan
        plan = self._clear_for_buddy(state, occupied, vis)
        if plan:
            return plan
        plan = self._clear_buddy_ties(state, occupied, vis)
        if plan:
            return plan
        plan = self._aura_fix(state, occupied, vis)
        if plan:
            return plan
        for name, pos in self._field.items():
            pos = tuple(pos)
            confirmed = self._confirmed_at.get(name) == (self._round_seen, pos)
            # **贴主C 的英雄（明世隐/瑶/太乙）视觉看到那格有人也算**（2026-09-29 第 9 局：明世隐在场 633 帧、一次都没贴着守约 ——
            # 例行点开复查取消后几乎没人被"确认过位置"，这条纠正从来不触发）。坦克/主C 前后排的纠正仍只挪确认过的。
            sac = self._is_sacrifice(name) and "钟馗" in self._field and name != self._carry_piece
            if not confirmed and not ((self._carry_buddy(name) or sac) and vis.get(cell_key(pos)) == "hero"):
                continue
            if self._cell_dead(pos):
                continue
            if pos not in seen_now and vis.get(cell_key(pos)) != "hero":
                continue       # **以画面为准**：这一帧那一格没看到徽章就不挪（哪些位置有英雄以画面为准）
            _anc = self.buddy_anchor(name) if self._carry_buddy(name) else self._carry_piece    # 太乙早到贴盾山
            carry_pos = self._field.get(_anc) if _anc else None
            if self._carry_buddy(name) and carry_pos and name != _anc:
                # 作用于"最近友军"的（明世隐 / 李元芳 / 瑶 / 太乙）。**连接在放下那一刻定，挪动时只有"主C 比别人都近"才改连**
                # （游戏小 bug，2026-10-08：先放到李元芳旁边距离 1，再挪到主C 旁边，离李元芳和主C 都是 1，仍会连李元芳）。
                # 做法：最终格 F = 离主C 1 步、和别人并列最少的格；F 上主C 不是唯一最近 → 先经过一个"主C 严格最近"的中转格 S
                # （在 S 连上主C），下一帧再挪到 F（并列时连接保持不变）。
                others = self._buddy_others(name, vis)
                cur = self._buddy_cell_score(pos, carry_pos, others)
                link = (self.__dict__.get("_buddy_link") or {}).get(name)
                sticky = self._link_sticky(name)       # 只有"连接"类（明世隐）放下后连接不变；太乙 / 瑶开战时现算最近的人
                if sticky and link is not None:
                    linked = link == (self._carry_piece, tuple(carry_pos))   # 备战阶段选人：以放下那一刻记的连接为准
                else:
                    linked = cur[0] == 0                                       # 开团现算 / 没记录：看现在主C 是不是唯一最近
                best = self._best_buddy_cell(state, carry_pos, others)
                target = best if (best and self._buddy_cell_score(best, carry_pos, others) < cur) else pos
                if target == pos and linked:
                    continue
                if self._buddy_cell_score(target, carry_pos, others)[0] == 0 and target != pos:
                    lane, why = [target], f"作用于最近友军，挪到主C {self._carry_piece} 是唯一最近友军的格子"
                elif not sticky:
                    continue                           # 现算最近的（太乙 / 瑶）：没有"主C 唯一最近"的空格就先不动，中转对它无效
                elif linked and target != pos:
                    # 已经连着主C（在中转格放下的）→ 直接挪到最终格，并列也保持连主C
                    self._reposition_round = state.round_no
                    self._evict_cell(target, name)
                    self._cell_owner.pop(pos, None)
                    self._field[name] = tuple(target)
                    self._cell_owner[tuple(target)] = name
                    logger.info("刷新连接：%s 已连主C，从 %s 挪到主C 旁 %s", name, pos, target)
                    return {"type": "move_unit", "piece": name, "src": pos, "dst": tuple(target), "keep_link": True}
                else:
                    # 需要中转：S = 空格里主C 严格最近（并列 0）、离主C 最近的那格
                    from ..perception.extractors import POP_OCCLUDING_CELLS as _POC
                    stage = [tuple(c) for row in OWN_BOARD_ROWS for c in row
                             if tuple(c) not in occupied and tuple(c) != tuple(target) and not self._cell_dead(c)
                             and tuple(c) not in set(map(tuple, _POC))
                             and not self._covered_by_shop(state, c)
                             and self._buddy_cell_score(c, carry_pos, others)[0] == 0]
                    if not stage:
                        continue
                    S = min(stage, key=lambda c: self._hex_dist(c, carry_pos))
                    self._reposition_round = state.round_no
                    self._buddy_return = (state.round_no, name, tuple(target))
                    self._evict_cell(S, name)
                    self._cell_owner.pop(pos, None)
                    self._field[name] = S
                    self._cell_owner[S] = name
                    logger.info("刷新连接：%s 先挪到 %s（主C %s 是唯一最近的人，在这里连上主C），下一帧再挪到 %s",
                                name, S, self._carry_piece, target)
                    self.note("刷新连接", 英雄=name, 中转=f"{S[0]},{S[1]}", 主C=self._carry_piece)
                    return {"type": "move_unit", "piece": name, "src": pos, "dst": S}
            elif sac:
                sides = [c for c in self._row_sides(self._field["钟馗"]) if tuple(c) != pos]
                if pos in {tuple(c) for c in self._row_sides(self._field["钟馗"])}:
                    continue                              # 已经在钟馗旁边
                free = [c for c in sides if tuple(c) not in occupied and not self._cell_dead(c)
                        and not self._covered_by_shop(state, c)]
                if not free:
                    continue
                lane, why = free, "带牺牲，挪到钟馗旁边（吃钟馗的牺牲）"
            elif not confirmed:
                continue                                  # 只凭视觉的，只做"贴主C / 牺牲件贴钟馗"这两种挪动
            elif self._is_tank(name) and pos in back:
                lane, why = OWN_BOARD_DROPS_FRONT, "坦克在后排，挪到前排"
            elif name == self._carry_piece and not self._is_tank(name) and pos in front:
                lane, why = OWN_BOARD_DROPS_BACK, "主C 在前排，挪到后排角落"
            else:
                continue
            # 带「夺取」的英雄不挪到主C 旁边（之后合成会随机抢主C 10 级；第 179 局猪八戒被挪到亚连旁边，亚连 17→5）
            near_carry = (set(self._cell_neighbors(tuple(carry_pos)))
                          if carry_pos and name != self._carry_piece
                          and "夺取" in self.pieces.get(name, {}).get("skill", "") else set())
            dst = next((tuple(c) for c in lane if tuple(c) not in occupied and not self._cell_dead(c)
                        and not self._covered_by_shop(state, c) and tuple(c) not in near_carry), None)
            if dst is not None and self._blocks_carry_buddy(name, dst) and not (
                    self._is_sacrifice(name) and "钟馗" in self._field
                    and tuple(dst) in {tuple(c) for c in self._row_sides(self._field["钟馗"])}):
                # （10-21 第 899 局：要在钟馗的左右两侧——牺牲件贴钟馗左右优先，不受这条限制；"让开"那边也不再挪它们）
                # 2026-10-21 第 895 局（铠每回合"站位纠正贴钟馗 → 让开明世隐"一来一回）：钟馗旁的空格和主C 离明世隐一样近 →
                # 明世隐转等级给主C 的收益 > 吃钟馗牺牲，纠正不挪进会抢明世隐的格子
                dst = next((tuple(c) for c in lane if tuple(c) not in occupied and not self._cell_dead(c)
                            and not self._covered_by_shop(state, c) and tuple(c) not in near_carry
                            and not self._blocks_carry_buddy(name, tuple(c))), None)
            if dst is None or not self._move_ok(state, name, pos, dst):
                continue
            self._note_move(state, name, pos, dst)
            self._reposition_round = state.round_no
            self._evict_cell(dst, name)
            self._cell_owner.pop(pos, None)
            self._field[name] = dst
            self._cell_owner[dst] = name
            logger.info("站位纠正：%s 从 %s 挪到 %s（%s）", name, pos, dst, why)
            self.note("站位纠正", round=state.round_no, 英雄=name, 从=f"{pos[0]},{pos[1]}", 到=f"{dst[0]},{dst[1]}", 原因=why)
            return {"type": "move_unit", "piece": name, "src": pos, "dst": dst}
        return None

    def decide_all(self, state: GameState) -> list[dict[str, Any]]:
        """按阶段分发决策，返回有序操作列表（最外层过一道"绝不移除当前主C"的总闸）。"""
        carry0 = self._carry_piece                      # 这一帧里可能先换了主C：两个都算"当前主C"
        self._note_gold(state)
        acts = self._decide_all_raw(state)
        carries = {c for c in (carry0, self._carry_piece) if c}
        if _IR_on("swap_spare_for_route") and not any(a.get("type") in ("sell_unit", "sell_card") for a in acts or []):
            sp = self.spare_swap(state)
            if sp:
                acts = [sp]
        duo = self.duo_active() if (_IR_on("duo_no_sell") or not self.model_decides()) else set()
        if duo:
            _before = list(acts or [])
            acts = [a for a in _before if not (a.get("type") in ("sell_unit", "sell_card") and a.get("piece") in duo)]
            if len(acts) != len(_before):
                logger.info("共战天赋的英雄不卖：%s", sorted(duo))
        kept = []
        for act in acts or []:
            if (act.get("type") == "drag_card" and act.get("target")
                    and act.get("target") in carries
                    # 限制性进化 = 把 1~2 阶英雄换进高 2 阶身体 +10 级，打给**临时主C**正合适（硬规则清查：前提已过时）
                    and not (act.get("piece") == "限制性进化" and self.is_temp_carry(act.get("target")))
                    and self.pieces.get(act.get("piece"), {}).get("category") != "装备牌"   # 护命灵珠"护盾…移除"不算
                    and "移除" in (self.pieces.get(act.get("piece"), {}).get("skill") or "")):
                # **总闸**：遗志/体面离场/限制性进化 这类"移除英雄"的牌绝不打给当前主C（2026-09-24：用了就直接输了）。
                # 放在最外层：decide_all 里有好几条提前返回的路径，只在其中一处拦会漏。
                logger.error("拦下：%s 会移除当前主C %s —— 拒绝执行（这是 bug，查是哪条路径给的目标）",
                             act.get("piece"), act.get("target"))
                self.note("拦下移除主C", 牌=act.get("piece"), 主C=act.get("target"))
                continue
            kept.append(act)
        return kept

    def _decide_all_raw(self, state: GameState) -> list[dict[str, Any]]:
        """按阶段分发决策，返回有序操作列表。"""
        self._remember_pop(state)
        # 局外阶段 = 新的一局要开始了，把上一局的记忆清干净
        if state.phase in ("lobby", "mode_lobby", "match_found", "hero_pick"):
            self.reset_match()
            return []
        if state.phase == "card_pick":
            # 独立阶段的 N 选 1（锦囊/兑换券打出后弹的）。**必须选**：
            # 这个界面占着商店位置、不选就一直挂着，后面什么都干不了。
            return [a] if (a := self.decide_card_pick(state)) else []
        if state.phase == "auction":
            act = self.decide_auction(state)
            return [act] if act else []
        if state.phase == "talent":
            act = self.decide_talent(state)
            return [act] if act else []
        if state.phase == "lord_boost":
            act = self.decide_lord_boost(state)
            return [act] if act else []
        if state.phase != "prepare":
            return []
        self.resolve_comp(state)
        if state.phase == "prepare":
            self.explain(state)           # 每回合一次：把局面和决策依据摊开写进日志
        self._note_equip_seen(state)
        self.track_hand_gains(state)      # 先记账：手上多了什么牌
        self.apply_board_observation(state)  # 再用**画面观测**校正位置（比人口对账精确）
        self.vision_identity_sync(state)  # 认人模型：格子上认出的是谁，比账本和点开看都硬
        self.reconcile(state)             # 最后按人口数兜底对账
        self.sync_field(state)         # 最后处理觉醒回手
        self.vision_absence(state)     # D-FINE 连续多帧看不到的人：回手（觉醒）或已离场
        self._pop_mark(state)          # 对账完再记人口锚（pop_now 读不到时 = 锚 + 之后账本的确定变化）
        self.last_gold = state.gold
        # 天赋三选一被收起来了（左侧「天赋未领取」）—— 点开它，下一帧走 talent 分支去选。
        # 不点就等于白丢一个天赋（2026-09-23 第 15 回合：挂了整整一回合没人管）
        # 入口是右下角能量圆盘（和开商店同一个）。**每回合最多试 3 次**，
        # 打不开就照常做别的 —— 不能让它排在最前面把整回合卡死。
        if state.talent_pending and not state.shop_open:
            k = state.round_no
            n = self._talent_open_tries.get(k, 0)
            if n < self.TALENT_OPEN_TRIES:
                self._talent_open_tries[k] = n + 1
                logger.info("左侧挂着「天赋未领取」（天赋三选一还没选），点能量圆盘打开（第 %d 次）", n + 1)
                x, y = state.talent_pending
                return [{"type": "open_talent", "x": x, "y": y}]
        # 卡牌N选1 界面占着商店的位置，不选就一直卡着 —— 最优先处理
        if state.card_pick and state.card_options:
            act = self.decide_card_pick(state)
            if act:
                return [act]

        actions = []
        # 每回合复位用**独立字段**：不能蹭 _deployed_round（那个在 decide_deploy 里才更新，
        # 而关商店分支会提前 return，导致计数每帧都被当成新回合清零 →
        # 2026-09-22 实战出现无限开关商店）
        if state.round_no != self._round_seen:
            self._round_seen = state.round_no
            self._drag_tries.clear()
            # **占用是每回合都会变的**：上回合阵亡的、被卖的、觉醒回手的，格子都空了。
            # 把"疑似有人"的记忆带到下一回合，只会让可用落点越来越少——
            # 2026-09-22 第三局前排 (660,420)/(540,420)/(780,420) 全被拉黑，
            # 曜 觉醒回到手牌后无处可放，再也上不去。
            # 注意只清 `_busy_cells`（占用，回合性）；`_bad_cells` 是"这个点根本拖不进去"
            # 的几何问题，整局都成立，不清。
            self._busy_cells.clear()
            self._drag_fails.clear()
            self._shop_opens = self._shop_closes = 0
            self._buys_this_round = 0

        # ---- 升级棋手：**不受商店开关影响**，也不该被"开店买牌"挤到后面 ----
        # 升级按钮在左下角 (95,550)，商店面板（y38-397）压根盖不到它，任何时候都能点。
        # 而原来它排在决策链最末尾，前面"关着商店就先开店"那条会提前 return，
        # 于是"能量够了却不升级"（2026-09-22：能量很多却不升级棋手）。
        # 棋手等级决定人口上限和高阶商店概率，是前期最大的复利，值得插在最前面。
        # 手里看得见觉醒卡 -> 先把记账改对，后面的合成/上场/腾位才不会拿旧账做决定
        self.sync_awakened(state)
        lv_by_cell = state.unit_levels or {}
        self._unit_lv = {n: lv_by_cell.get(cell_key(tuple(p))) for n, p in self._field.items()
                         if lv_by_cell.get(cell_key(tuple(p)))} or self._unit_lv
        self.choose_carry(state)
        # **觉醒卡在手里等上场 = 最高优先级**（2026-09-25 第四轮第 5 局：觉醒牌回到手牌后又先上了其他牌，导致觉醒牌无法上场，
        # 而这张牌是未来的主C）：觉醒孙悟空回手后，先上了刘邦（主C 表 1 分）占满人口，又去买牌，23 秒后才卖人腾位，
        # 回合结束时孙悟空还在手里 —— 整回合缺阵。现在：有觉醒英雄在手里没上场，先关店、再腾位/上场，别的都等它。
        awk_wait = [sl for sl in state.hand_slots or []
                    if sl.get("name") in self._awakened and sl.get("name") not in self._field
                    and self.pieces.get(sl.get("name"), {}).get("category") == "英雄牌"]
        if awk_wait:
            if state.shop_open and self._shop_closes < 4:
                logger.info("手里有觉醒的 %s 等上场 —— 先关店", awk_wait[0]["name"])
                return [{"type": "toggle_shop", "intent": "close"}]
            if not state.shop_open:
                for fn in (self.decide_sell_unit, self.decide_deploy):
                    act = fn(state)
                    if act:
                        return [act]
        hy = self.decide_haiyue(state)
        if hy:
            return [hy]
        level_act = self.decide_economy(state)
        if level_act:
            logger.info("升级棋手（能量 %d，费用 %s）", state.gold, state.level_cost)
            return [level_act]

        # ---- 商店面板会盖住棋盘上半部分，挡着就没法把英雄拖上场 ----
        # 2026-09-22 确定的正确顺序：**先买完 → 关商店 → 再操作棋盘**。
        # 之前一直是开着商店硬拖，落点要么在面板上要么在被挡住的区域，
        # 所以"看不到英雄被拖上场"——它根本没落到棋盘上。
        # 关店是**当回合**的操作，新回合要重新打开，否则买不了牌
        # （2026-09-22 实战：第 1 回合关店后一直没开，第 2 回合商店空着干瞪眼）
        # **开店条件必须和关店条件对称**，否则两边互相不知情就会无限开关：
        # 关店说"买够了"，开店说"没有棋盘活要干"，于是 close→open→close…
        # （2026-09-22 实战复现：连着刷了好几轮 toggle_shop）。
        # 开不开店看局面（钱、棋盘活干完没有、Laya 是否选了"停手"→ _stop_shop_round），不按次数卡
        # （2026-10-09：不设硬性次数限制，按局面判断）。SHOP_OPEN_SAFETY 只防程序 bug 造成的开关死循环
        if (not state.shop_open and self._shop_opens < self.SHOP_OPEN_SAFETY and self._shop_closes < 4 and state.gold >= 3
                and self._stop_shop_round != state.round_no
                and self._buys_this_round < self._buy_cap(state)
                and not self._board_work_pending(state)):
            # 注意顺序：**棋盘上的活先干完**（上场/穿装备/卖单位），再开店买牌。
            # 反过来的话，商店面板一开又把棋盘挡住，这一回合就白过了。
            logger.info("棋盘操作已完成，打开商店准备买牌（能量 %d，本回合已开 %d 次）",
                        state.gold, self._shop_opens)
            return [{"type": "toggle_shop", "intent": "open"}]

        if state.shop_open and state.shop and any(sl.piece for sl in state.shop):
            self.__dict__["_shop_seen_round"] = state.round_no          # 本回合看到过商店内容（没钱时开店锁店用）
        if (not state.shop_open and state.gold < 3 and self.__dict__.get("_shop_seen_round") != state.round_no
                and self.__dict__.get("_lock_peek_round") != state.round_no and self._locked_round != state.round_no
                and self._stop_shop_round != state.round_no and self._shop_closes < 4
                and not self._board_work_pending(state) and (state.round_no or 0) >= 2):
            # 2026-10-21 第 887 局 R10（刷出了明世隐却没重新开店锁定，手动开店后才锁上）：
            # 升级把钱花光、本回合一眼都没看过商店 → 开一次店看看（锁店不要钱，锁不锁交给 Laya 商店题）
            self.__dict__["_lock_peek_round"] = state.round_no
            logger.info("本回合还没看过商店、钱不够买（%de）→ 开店看一眼，有要的牌好锁店", state.gold)
            return [{"type": "toggle_shop", "intent": "open"}]
        if state.shop_open and state.shop and all(not sl.piece for sl in state.shop):
            # 2026-10-21 第 887 局 R10（商店刷出明世隐却没重新开店锁定）：回合开头商店开着、4 格一张都没读出来
            # （刚翻页 / 动画）→ 代码当"买完了"关店、升级花光，整回合没看过商店。读不出来就等几帧再判（每回合最多等 6 帧）
            _k = ("shop_unread_wait", state.round_no)
            n0 = self._skip_logged.get(_k, 0)
            if n0 < 6:
                self._skip_logged[_k] = n0 + 1
                if n0 == 0:
                    logger.info("商店开着但一张都没读出来 → 先等几帧再决定（不当买完了关店）")
                return []
        if state.shop_open:
            # **商店开着时不能无限买下去**（2026-09-22 实战：19/20 帧全是买/刷新，
            # 一次上场都没有，场上一直 0 个人）。根因是这游戏**买完一张立刻补新卡**，
            # 于是"还有值得买的牌"永远成立，`decide_buy` 每帧都 return，
            # 永远走不到"关店 -> 上场"。
            # 两条硬闸：
            #   ① 场上空着且手里有能上的人 —— 空场必输，先把人放上去再说
            #   ② 每回合买入上限 —— 钱要留给升级和拍卖，手牌也有上限
            must_deploy = (not self._field) and self._board_work_pending(state)
            hand_heroes = sum(1 for sl in state.hand_slots or []
                              if self.pieces.get(sl["name"], {}).get("category") == "英雄牌")
            if must_deploy and state.round_no <= 2 and hand_heroes < 2 and state.gold >= 3 and any(
                    self.pieces.get(sl.piece, {}).get("category") == "英雄牌"
                    and self._worth_buying(sl, state) for sl in state.shop if sl.piece):
                # **第 1~2 回合、手里还不到 2 个英雄：钱够就再买一张再关店**（2026-09-23 第 1 回合：
                # 6 能量只买了 1 张就关店去上场，第 1 回合只有 30 秒，来不及再开店，
                # 剩 3 能量、场上 1 个人，第 3 回合还是 1/5）。只放宽这么窄：场景30 那条
                # "场上空着别无限买"（实战 19/20 帧都在买、场上 0 人）在其余时候照旧。
                must_deploy = False
            over_budget = self._buys_this_round >= self._buy_cap(state)
            if over_budget and not must_deploy:
                # 买够了也先看**路线主C** 在不在商店（2026-09-30 审查：关店分支在 Laya / preempt 之前就返回，主C 摆着被跳过）
                pre = self.preempt_shop(state)
                if pre and pre.get("why") == "路线主C":
                    return [pre]
            if (must_deploy or over_budget) and self._shop_closes < 2:
                logger.info("先关商店去操作棋盘（%s；本回合已买 %d 张）",
                            "场上还是空的" if must_deploy else "买够了",
                            self._buys_this_round)
                return [{"type": "toggle_shop", "intent": "close"}]
            buy = self.decide_buy(state)
            if buy:
                self.last_gold = state.gold
                logger.info("规则引擎决策: %s", [buy])
                return [buy]                        # 先把该买的买完
            if self._board_work_pending(state) and self._idle_opened_round != state.round_no:
                # （本回合是因为"关着空转"才开的店：棋盘的"待办"其实做不了，别又关掉 —— 第十四局开了马上关、空转再开）
                if self._shop_closes < 2:
                    logger.info("买完了，关闭商店腾出棋盘（手牌 %d 张待处理，本回合已关 %d 次）",
                                len(state.hand_slots), self._shop_closes)
                    return [{"type": "toggle_shop", "intent": "close"}]
                # 点了两次还看得见商店：要么没点中，要么这版 UI 不是这样关的。
                # 继续点只会反复开关，不如带着商店硬干（拖拽可能落空但至少不空转）。
                if self._shop_closes == 2:
                    self._shop_closes += 1      # 只报警一次
                    logger.warning("点过 2 次关店但商店仍开着（坐标 %s 可能不对），"
                                   "本回合不再尝试，继续后续决策", BTN_SHOP_TOGGLE)
        # 人口读不到（商店面板挡住了）→ 关店一次把它读出来。
        # 没有人口上限，decide_deploy 会退化成"每回合最多上 1 个"，整局场面残缺。
        # 商店关了也不影响阶段识别：prepare 判据里有"准备"按钮兜底。
        if (state.population_cap <= 0 and state.round_no != self._shop_closed_round
                and not any(self._worth_buying(s, state) for s in state.shop)):
            self._shop_closed_round = state.round_no
            logger.info("人口读不到，关闭商店以读取人口上限")
            return [{"type": "toggle_shop", "intent": "close"}]

        for fn in (self.decide_entry_boost, self.decide_upgrade, self.decide_sell_unit, self.decide_deploy,
                   self.decide_use_cards, self.decide_reposition, self.decide_sell, self.decide_buy,
                   self.decide_refresh, self.decide_lock):
            act = fn(state)
            if not act:
                continue
            # 拖拽类动作加失败保护：同一回合、同一张牌、同一落点连拖 3 次仍在出，
            # 说明它根本没生效（坐标不对 / 目标格被占 / 手牌位置漂移）。
            # 再拖也是白拖，而且会一直挤掉后面的决策——停手并报警，等人看日志。
            # 合成（牌名已在 _field 里）单独处理：它本来就会把同名副本反复拖到
            # 场上那个英雄身上，每张副本一次，不能按"上场失败"拉黑格子。
            # 但**连续多次都没消耗掉手牌**说明那个位置上的英雄已经不在了
            # ——最常见的原因是**觉醒回到了手牌**（2026-09-22 李元芳觉醒后没重上）。
            # 用**动作自带的标记**判断是不是合成，不能看 `piece in self._field`：
            # decide_deploy 在返回前就把英雄写进了 _field，导致"上场"被误判成"合成"，
            # 一次调用里计数加两次，动作还被丢掉
            # （2026-09-22：李元芳觉醒后一直重上不去，根因就在这儿）。
            piece = act.get("piece")
            if act.get("type") == "sell_unit" and self.sell_blocked(piece):
                continue           # 本回合已经卖了 2 次都没生效（undo_sell），别再吃帧
            # 下面这两段是**开环猜测**（"拖了 N 次还在出 → 大概失败了"）。
            # 一旦调用方接了 `confirm_drag`（拖完真去看画面），就有实测回执了，
            # 猜测反而会误伤：它曾把"落点被占"猜成"英雄觉醒回手"，于是原地重放
            # 到同一个被占的格子，死循环（2026-09-22 整局李元芳没上去）。
            # 所以有回执时直接跳过，只在干跑/没接回执的场景里保留。
            if self._verified:
                actions.append(act)
                self._simulate_cost(act, state)
                if act["type"] in MUTATING_TYPES:
                    break        # 同下面那条：一次只出一个改变画面的动作
                continue
            if act.get("merge"):
                mkey = (state.round_no, "merge", piece)
                self._drag_tries[mkey] = self._drag_tries.get(mkey, 0) + 1
                if self._drag_tries[mkey] >= 3:
                    logger.warning("%s 往 %s 合成了 %d 次都没生效——判定它已不在场上"
                                   "（多半是觉醒回到手牌），清除记录并重新上场",
                                   piece, self._field.get(piece), self._drag_tries[mkey])
                    pos = self._field.pop(piece, None)
                    if pos:
                        self._cell_owner.pop(pos, None)   # 它真的走了，格子腾出来
                    self._awakened.add(piece)
                    self.note("推断离场", round=state.round_no, 英雄=piece, 原位置=str(pos),
                              依据="合成连续失败", 后续="重新上场（场上计数已减）")
                    continue

            if (act.get("type") in ("drag_card", "move_unit", "sell_unit")
                    and piece not in self._field):
                key = (state.round_no, act.get("piece"), tuple(act.get("dst") or ()))
                self._drag_tries[key] = self._drag_tries.get(key, 0) + 1
                if self._drag_tries[key] >= 2:
                    dst = tuple(act.get("dst") or ())
                    # 连拖 2 次不生效 = 那个格子多半已经有人（我不知道的单位）。
                    # 先当"被占"换下一个格子；真到 4 次还不行才认定格子本身无效。
                    if dst and self._drag_tries[key] < 4:
                        if dst not in self._busy_cells:
                            self._busy_cells.add(dst)
                            logger.info("落点 %s 拖了 %d 次没进去，判为已被占用，换个格子",
                                        dst, self._drag_tries[key])
                    elif dst and dst not in self._bad_cells:
                        self._bad_cells.add(dst)
                        logger.error("落点 %s 连拖 %d 次都没生效，本局拉黑（可能不在有效区内）",
                                     dst, self._drag_tries[key])
                    continue
            actions.append(act)
            self._simulate_cost(act, state)
            # **最多只产出一个"会改变手牌/棋盘"的动作**，和执行层对齐。
            # 执行层是"一帧一个改变画面的动作，做完立刻重新感知"（手牌会重新居中，
            # 后面动作的坐标全过期）。而 `decide_*` 在**生成**动作时就写下了
            # "这回合已经做过"的标记（_skill_round / _locked_round / 账本…），
            # 于是排在后面、根本没被执行的动作也被记成做过了 ——
            # 2026-09-22 第五局：枫叶舞每回合都排在上场动作后面，一次都没真打出去，
            # 却每回合都被标记成"技能已用"（枫叶舞实际没用出去）。
            # 截断在这里，标记就只会在真正要执行的那个动作上打。
            if act["type"] in MUTATING_TYPES:
                break
        if self._field:
            logger.info("场上布局: %s", {n: f"{x},{y}" for n, (x, y) in self._field.items()})
        logger.info("规则引擎决策: %s", actions)
        if state.shop_open and not actions:
            # **商店开着、却一个动作都没有：关店**（2026-09-23 第八局：人口满、商店里没想买的，
            # 38 秒每帧都是"决策: []"，一直等到回合结束）。关了店才看得清手牌、才能点开看牌/合成/上场。
            if self._shop_idle[0] != state.round_no:
                self._shop_idle = [state.round_no, 0]
            self._shop_idle[1] += 1
            # 上限放宽到 4：原来 2 次用光后商店开着就再也关不掉（空转到回合结束）
            if self._shop_idle[1] >= 3 and self._shop_closes < 4:
                logger.info("商店开着但连续 %d 帧无事可做，关店去处理手牌/棋盘", self._shop_idle[1])
                self._shop_idle[1] = 0
                return [{"type": "toggle_shop", "intent": "close"}]
        elif actions:
            self._shop_idle = [state.round_no, 0]
        # 2026-10-09 第 471 局（后期能量太多花不完）：R15 已开 2 次店被次数上限挡住，带 25 能量空转 50 秒。
        # 现在只看局面：关着、连续 6 帧无事可做、有钱、Laya 没说停手 → 开店
        if (not state.shop_open and not actions and state.gold >= 3 and self._shop_opens < self.SHOP_OPEN_SAFETY and self._shop_closes < 4
                and self._stop_shop_round != state.round_no):
            # **商店关着、也没有任何动作：开店**（和上面"开着无事可做就关店"对称）。
            # 2026-09-23 第十三局 R8：手里有装备和两个英雄（人口满），`_board_work_pending` 说"棋盘还有活"
            # 所以不开店，可后面的决策链一个动作也给不出来 —— 13 能量、3 级，48 秒空转到回合结束，
            # 商店没开就连升级费都读不到。
            if self._closed_idle[0] != state.round_no:
                self._closed_idle = [state.round_no, 0]
            self._closed_idle[1] += 1
            # **场上有空位、手里没有能补的英雄 → 不用等 6 帧，马上开店补人**（2026-10-13 复盘 b：第 603 局第 5 回合换绑卖人后 2/6，
            # 几十帧没动作；第 602 局 4/5、剩 5 能量没开店。第 5~8 回合空位是第 5~6 名的头号输因）
            _gap = (state.phase == "prepare" and state.population_valid and 0 < (state.population_cap or 0)
                    and (state.population or 0) < state.population_cap)
            _waiting = any(sl.get("name") not in self._field and self.pieces.get(sl.get("name"), {}).get("category") == "英雄牌"
                           for sl in state.hand_slots or [])
            if self._closed_idle[1] >= (1 if _gap and not _waiting else 6):
                logger.info("商店关着且连续 %d 帧无事可做，开店（能量 %d）", self._closed_idle[1], state.gold)
                self._closed_idle[1] = 0
                self._idle_opened_round = state.round_no
                return [{"type": "toggle_shop", "intent": "open"}]
        elif actions:
            self._closed_idle = [state.round_no, 0]
        return actions
