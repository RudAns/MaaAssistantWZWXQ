"""正向推演的局面。只放养成相关的量，全部可以深拷贝。

2026-10-14 统一接口：各批次原来各用各的 flags 键名记的量，改成下面的正式字段（写卡规范见 cards/README.md）：
  core            阿科米亚核心等级（原 flags["core"] / flags["阿科米亚核心"]）
  hp / hp_guard   棋手生命值（原 flags["lord_hp"] / ["hp"] / ["lord_hp_delta"]）、"降到 0 保留 1 点"次数（一线生机）
  rank            对局排名（原 flags["rank"]；None = 不知道）
  eliminated      已被淘汰的其他棋手数（engine.eliminate 发 eliminated 事件）
  energy_floor    能量下限（原 flags["energy_floor"]：透支 / 贷款协议）
  shop_extra_cost 商店购买额外花费（原 flags["shop_extra_cost"]：大买卖）
  free_refresh_all 本回合刷新商店不耗能量（原 flags["round:free_refresh_all"]：展翅腾飞），回合开始清
  hand_meta       手牌逐张状态：{卡名: [ {got: 获得回合, ...}, ... ]}，同名按获得先后排（先拿到的先用）
  round_cards     本回合打出过、整回合生效的牌（锦囊…；原 flags["round:挂靠"]），回合开始清
  battle_dead / battle_deaths  本场战斗结束时仍处于阵亡的单位 uid、各单位本场阵亡次数（复生后又活下来的不在 battle_dead 里）
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Unit:
    name: str
    level: float = 1.0                 # 永久等级（真实等级）
    temp: float = 0.0                  # 临时等级（本回合，战斗结束清零）
    awaken: int = 0                    # 觉醒进度：合成过几张同名（本体 + 3 张 = 觉醒）
    equips: list = field(default_factory=list)     # 装备名
    bound: list = field(default_factory=list)      # 其中拿不下来的（出生自带 / 无法拆卸）
    cell: Optional[tuple] = None
    pop: bool = True                   # 占不占人口（召唤物 / 视为不占人口的单位为 False）
    kind: str = "hero"                 # hero / summon / object
    counters: dict = field(default_factory=dict)   # 卡面计数（"已击败 N"、"每使用 4 张古币"…）；battle_only = 只在本场战斗存在
    keywords: set = field(default_factory=set)     # 额外获得的关键词（钟馗给的"牺牲"、灵泉法杖给的"闪现"…；每回合重算）
    uid: int = 0

    @property
    def awakened(self) -> bool:
        return self.awaken >= 3

    def lv(self) -> float:
        return self.level + self.temp


@dataclass
class State:
    round: int = 1
    lord: str = "阿离"
    lord_level: int = 1
    xp: int = 0
    energy: int = 6
    pop_bonus: int = 0                 # 天赋 / 效果给的额外人口
    units: list = field(default_factory=list)      # 场上（含召唤物）
    hand: list = field(default_factory=list)       # 手牌（卡名）
    hand_meta: dict = field(default_factory=dict)  # 手牌逐张状态（见文件头）；engine.gain_card 写、engine.take_meta 取
    shop: list = field(default_factory=list)       # 当前商店 4 格（卡名或 None）
    shop_locked: bool = False
    free_refresh: int = 0              # 免费刷新次数
    free_refresh_all: bool = False     # 本回合刷新不耗能量（展翅腾飞），回合开始清
    shop_extra_cost: int = 0           # 商店购买额外花费（大买卖）
    energy_floor: int = 0              # 能量下限（透支 / 贷款协议：-15 / -20）
    talents: list = field(default_factory=list)    # 已选天赋（卡名）
    lord_cards: list = field(default_factory=list) # 棋手技能牌 + 强化牌（枫叶舞、翩跹舞步·新生…）
    round_cards: list = field(default_factory=list)  # 本回合打出过、整回合生效的牌（锦囊…），回合开始清
    maple: str = ""                    # 枫叶舞绑定的单位名（"" = 没绑）
    maple_keep: float = 0.0            # 红叶染锋：留给下一个枫叶舞英雄的等级
    carry: str = ""                    # 当前主C（推演打法定的）
    core: float = 0.0                  # 阿科米亚核心等级（开战时以临时等级分给日落海英雄，engine.core_distribute）
    hp: int = 60                       # 棋手生命值（上限 engine.LORD_HP_MAX = 60）
    hp_guard: int = 0                  # "生命值降到 0 时保留 1 点"的剩余次数（一线生机）
    rank: Optional[int] = None         # 对局排名（1 = 第一；None = 不知道）
    eliminated: int = 0                # 已被淘汰的其他棋手数
    battle_dead: list = field(default_factory=list)    # 本场战斗结束时仍阵亡的单位 uid
    battle_deaths: dict = field(default_factory=dict)  # 本场各单位阵亡次数 {uid: n}
    pool_taken: dict = field(default_factory=dict) # 牌库里我方拿走的张数（对手持有按 data/engine/shop_stats.json 的 opp_take 另算）
    banned: str = ""                   # 本局被禁派系：这个派系的英雄不进牌库（实测：76396 个商店英雄格里 5 次，基本是读错）
    merges: int = 0                    # 本局合成次数（每 3 次送 1 个效果锦囊）
    flags: dict = field(default_factory=dict)      # 卡面状态（"限一次"已用、本回合计数…；"round:" 前缀回合开始清）
    log: list = field(default_factory=list)        # (回合, 事件, 详情)
    next_uid: int = 1

    def clone(self, keep_log: bool = True) -> "State":
        """深拷贝。keep_log=False：不复制日志（往后看的推演用不到，日志越到后期越长，复制它是白花时间）。"""
        if keep_log:
            return copy.deepcopy(self)
        log, self.log = self.log, []
        try:
            return copy.deepcopy(self)
        finally:
            self.log = log

    # -------- 查询
    def on_board(self) -> list:
        return [u for u in self.units if u.kind != "object"]

    def heroes(self) -> list:
        return [u for u in self.units if u.kind == "hero"]

    def unit(self, name: str) -> Optional[Unit]:
        return next((u for u in self.units if u.name == name), None)

    def pop_used(self) -> int:
        return sum(1 for u in self.units if u.pop)
