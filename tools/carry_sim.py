"""主C 选择 / 换主C 的**机制模拟器**（2026-10-12：选主C / 换主C 靠模拟找最优解；
历史数据不准，只用实际游戏规则和卡面信息模拟）。

模型里**没有任何参数来自历史对局**。只用：
  - 游戏规则：agent/decision/meta.py（收入、棋手经验、拍卖回合、商店品阶概率 TIER_ODDS、牌池 POOL_COPIES/POOL_HEROES、
    人口表、等级进度表）、人工确认的规则（合成 / 觉醒 = 本体 + 3 张同名、词条翻倍 / 临时等级战后清除 /
    枫叶舞绑定后要卖掉被绑者才能重绑）、data/keywords.json（整备 = 回合开始、开团 = 战斗开始、牺牲 = 阵亡时…）；
  - 卡面原文：data/cards.json 的 skill / awaken（下方 ENGINE 里逐条写了对应原文）、阿离的枫叶舞强化牌；
  - 战力：data/hero_strength.json 的 levels[lv].dmg（那是从技能原文按公式估的有效伤害，不含历史名次）；
  - 路线核心：data/route_min_engine.json 的 core_set。
历史对局只用来**取题**（laya_points.jsonl 里 kind=carry 的局面作为起点）和报告线上模型选了什么。

卡面 / 规则没写、只能假设的量（全部列在 ASSUME，报告里做敏感性分析）：
  合成一次涨几级、牺牲件每场阵亡概率、主C 每场击杀数、免费刷新次数、对手每回合拿走几张牌、装备多久来一件…

用法：
  .venv/Scripts/python.exe -X utf8 tools/carry_sim.py --questions            # 跑最近 60 局全部 carry 题，写 data/analysis/carry_sim_*.json(l)
  .venv/Scripts/python.exe -X utf8 tools/carry_sim.py --id 5ce89122e1e3       # 单题详细
  .venv/Scripts/python.exe -X utf8 tools/carry_sim.py --questions --sens      # 加敏感性分析
单进程（对局在跑时别占 CPU）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.decision import meta  # noqa: E402

OUT_DIR = ROOT / "data" / "analysis"

# ---------------------------------------------------------------------------
# 假设（卡面 / 规则没写的量）。默认值是"中性猜测"，不是拟合出来的。
# ---------------------------------------------------------------------------
ASSUME = {
    "merge_lv": 1,          # 合成一次（同名英雄牌打到场上同名身上）涨几级。规则只说"升等级"；「有舍有得：合成提升的等级-1」「曜：+1」说明基数很小
    "p_die": 0.5,           # 有牺牲效果的单位每场战斗阵亡的概率
    "kills": 1.0,           # 枫叶舞主C 每场击杀敌方英雄数（枫舞扶摇 / 蓄势 / 玄策用）
    "free_refresh": 1,      # 每回合免费刷新次数
    "paid_refresh_max": 4,  # 每回合最多付费刷新几次（刷完至少留一张英雄的钱）
    "opp_take": 14,         # 7 个对手每回合共从牌池拿走几张
    "item_every": 2,        # 主C 每几回合多一件装备（上限 3）
    "skill_share": 0.6,     # 主C 有效伤害里技能部分占比（法力上限降低只加速技能）
    "maple_blade_uptime": 0.75,   # 枫华之刃"转移后 +25% 伤害"在一场里生效的时间比例
    "effect_cards": 1.0,     # 每回合打给主C 的效果牌张数（古币等引擎产出之外；只影响沈梦溪 / 镜 这类"按张数"生效的词条，不加等级）
    "level_cards": True,     # 商店里的指定型加等级牌：每看一次商店按 meta.LEVEL_CARD_HIT[棋手等级] 出现一张（卡在 LEVEL_CARDS，按卡面加级），买得起就买给主C
    "kit_pick": 3,           # 效果锦囊（每合成 3 次送 1 个，品阶 = min(棋手等级,5)，人工确认的规则）从该品阶效果牌里 N 选 1 的 N（卡面没写，假设 3）
    "level_card_rate": 0.0,  # 通用加等级牌（不限派系）每回合给主C 的期望等级。对所有选项一样，默认 0
}
HORIZON = (10, 11, 12)       # 结果 = 这几回合战斗时"队伍战力"的平均：最强单位 + 0.5 × 第二强（每个单位都按卡面强度曲线 × 自身词条算）
LAST_ROUND = 12
TAU = 0.05                   # 软标签温度：期望战力每低 5%（相对最优）概率降到 1/e
N_SIM = 300

XP_NEED = {1: 7, 2: 13, 3: 17, 4: 23, 5: 27}     # 经验条 N/M 的 M（游戏画面读数，非统计）
HERO_PRICE, REFRESH_PRICE = 3, 2
BAN_NORM = {"sanfen": "三分之地", "heluo": "河洛", "dahe": "大河流域", "riluohai": "日落海", "rilu": "日落海", "zhulu": "逐鹿"}

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
_C = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
CARDS: dict = {}
for _k, _v in _C.items():
    n = _v.get("name") or _k
    if n not in CARDS or _v.get("category") == "英雄牌":
        CARDS[n] = _v
HEROES = {n: v["quality"] for n, v in CARDS.items() if v.get("category") == "英雄牌" and v.get("quality")}
FACTION = {n: (CARDS[n].get("faction") or "") for n in HEROES}
TEXT = {n: (CARDS[n].get("skill") or "") for n in HEROES}
HS = json.loads((ROOT / "data" / "hero_strength.json").read_text(encoding="utf-8"))["heroes"]
RM = json.loads((ROOT / "data" / "route_min_engine.json").read_text(encoding="utf-8"))


def has_kw(n: str, kw: str) -> bool:
    return f"{kw}：" in TEXT.get(n, "") or (kw == "牺牲" and n in ("钟馗",))


# 指定型加等级效果牌（卡面原文"指定1名英雄等级+N"）：(品阶, 名字, 价格, 只限河洛, 指定加几级, 随机再给1人几级)
LEVEL_CARDS = []
for _n, _v in _C.items():
    if _v.get("category") != "效果牌" or "(" in _n:
        continue
    _m = re.search(r"指定1名(【.+?】)?英雄等级\+(\d+)", _v.get("skill") or "")
    if _m and (_v.get("quality") or 9) <= 5:
        _r = re.search(r"随机1名英雄等级\+(\d+)", _v.get("skill") or "")
        LEVEL_CARDS.append((_v["quality"], _n, int(_v.get("cost") or 0), bool(_m.group(1)), int(_m.group(2)), int(_r.group(1)) if _r else 0))
EFFECT_BY_Q = {}
for _n, _v in _C.items():
    if _v.get("category") == "效果牌" and "(" not in _n and _v.get("quality"):
        EFFECT_BY_Q.setdefault(_v["quality"], []).append(_n)
# 2026-10-13 卡库重爬后的新写法（平衡调整）：
#   有福同享「指定1名英雄，场上和其品阶相同的英雄等级+4」→ 主C 和场上同品阶的人各 +N
#   封神一瞬「指定1名英雄获得其品阶×5的等级，且每次等级永久提升时，都额外+1级」→ 主C +品阶×N（"额外+1" 没建模，偏保守）
SAME_TIER_CARDS, TIERX_CARDS = {}, {}
for _n, _v in _C.items():
    if _v.get("category") != "效果牌" or "(" in _n or not _v.get("quality") or _v["quality"] > 5:
        continue
    _sk = _v.get("skill") or ""
    _m1 = re.search(r"场上和其品阶相同的英雄等级\+(\d+)", _sk)
    _m2 = re.search(r"获得其品阶×(\d+)的等级", _sk)
    if _m1 and _n not in {c[1] for c in LEVEL_CARDS}:
        SAME_TIER_CARDS[_n] = int(_m1.group(1))
        LEVEL_CARDS.append((_v["quality"], _n, int(_v.get("cost") or 0), False, int(_m1.group(1)), 0))
    elif _m2 and _n not in {c[1] for c in LEVEL_CARDS}:
        TIERX_CARDS[_n] = int(_m2.group(1))
        LEVEL_CARDS.append((_v["quality"], _n, int(_v.get("cost") or 0), False, 0, 0))
_LC = {c[1]: c for c in LEVEL_CARDS}

FLASH = {n for n in HEROES if TEXT[n].startswith("闪现") or "\n闪现" in TEXT[n] or TEXT[n].split("\n")[0].strip().startswith("闪现")}
ZHENGBEI = {n for n in HEROES if "整备：" in TEXT[n]}
KAITUAN = {n for n in HEROES if "开团：" in TEXT[n]}
SACRIFICE = {n for n in HEROES if "牺牲：" in TEXT[n]}


def _unit_names(t: str) -> list[str]:
    hits = sorted({(t.find(h), h) for h in HEROES if h in t})
    return [h for p, h in hits if not any(h != o and h in o and t.find(o) <= p < t.find(o) + len(o) for _q, o in hits)]


def _route_keys():
    return [k for k, v in RM.items() if isinstance(v, dict) and v.get("carry")]


def route_of_carry(name: str, prefer: str = "") -> str:
    """以 name 为主C 的路线 key（优先 prefer）。"""
    if prefer and (RM.get(prefer) or {}).get("carry") == name:
        return prefer
    return next((k for k in _route_keys() if RM[k].get("carry") == name), "")


def is_temp_hero(name: str, awk: bool = False) -> bool:
    """临时主C = 1 阶且没觉醒（与 rules.final_carry_settled 同口径）。"""
    return HEROES.get(name) == 1 and not awk


def route_core(key: str) -> tuple[str, list[list[str]], list[list[str]]]:
    """route_min_engine → (路线主C, core_set 组, min_set 里 core_set 之外的组)。每组是"任选其一"的英雄名单
    （「李元芳 或 程咬金」「闪现搭档 1 名（孙悟空最好，其次 …）」）；（牌）开头的条目不是英雄，跳过。"""
    v = RM.get(key) or {}
    carry = v.get("carry") or ""

    def groups(lst):
        out = []
        for u in lst or []:
            t = str(u.get("unit") if isinstance(u, dict) else u)
            if t.startswith("（牌）"):
                continue
            g = _unit_names(t)
            if g:
                out.append(g)
        return out
    core = groups(v.get("core_set"))
    if carry and not any(carry in g for g in core):
        core.insert(0, [carry])
    extra = [g for g in groups(v.get("min_set")) if not any(set(g) & set(c) for c in core)]
    return carry, core, extra


# ---------------------------------------------------------------------------
# 战力：hero_strength 的有效伤害曲线（按技能原文估的，含存活），乘上卡面里"只影响自己战斗"的词条
# ---------------------------------------------------------------------------
_LV_ANCH = (15, 40, 70, 100, 150)


TIER_SCALE = 0.0   # 同等级下品阶带来的额外倍率（每高 1 阶 +x）。卡面数值（hero_strength）已经体现了各英雄的技能/属性差异，默认 0；
                   # 经验上同样的主C 阵容，4 阶主C 基本比 2-3 阶主C 强 —— 若卡面以外还有隐藏的品阶成长，用 --tier-scale 打开做敏感性


def base_dmg(hero: str, lv: float) -> float:
    return _base_dmg(hero, lv) * (1 + TIER_SCALE * (HEROES.get(hero, 2) - 2))


def _base_dmg(hero: str, lv: float) -> float:
    h = HS.get(hero)
    if not h:
        return 0.0
    L = h["levels"]
    pts = [(int(k), float(v["dmg"])) for k, v in L.items()]
    pts.sort()
    lv = max(1.0, float(lv))
    if lv <= pts[0][0]:                       # 15 级以下：只有属性成长 (1+0.02·lv)（hero_strength 的 g=0.02）
        return pts[0][1] * (1 + 0.02 * lv) / (1 + 0.02 * pts[0][0])
    for (a, da), (b, db) in zip(pts, pts[1:]):
        if lv <= b:
            return da + (db - da) * (lv - a) / (b - a)
    (a, da), (b, db) = pts[-2], pts[-1]
    return db + (db - da) * (lv - b) / (b - a)


def assumed_items(lv: float) -> int:          # hero_strength 公式里的装备件数假设：40 级 1 件、70 级 2 件、≥100 级 3 件
    return 0 if lv < 40 else 1 if lv < 70 else 2 if lv < 100 else 3


def mana_max(hero: str) -> float:
    m = str((HS.get(hero) or {}).get("factors", {}).get("mana") or "")
    try:
        return float(m.split("/")[-1])
    except ValueError:
        return 0.0


def aps(hero: str, lv: float) -> tuple[float, float]:
    h = HS.get(hero)
    if not h:
        return 0.7, 15.0
    k = min(_LV_ANCH, key=lambda a: abs(a - lv))
    x = h["levels"][str(k)]
    return float(x.get("aps") or 0.7), float(x.get("alive") or 15.0)


# ---------------------------------------------------------------------------
# 一次模拟
# ---------------------------------------------------------------------------
class Unit:
    __slots__ = ("name", "lv", "merges", "awk", "items")

    def __init__(self, name, lv=1, merges=0, awk=False, items=0):
        self.name, self.lv, self.merges, self.awk, self.items = name, float(lv), merges, awk, items

    @property
    def k(self):          # 觉醒 = 词条效果翻倍
        return 2 if self.awk else 1


class Sim:
    core_lv = 0.0
    _mingshi_bonus = 0
    _pending_coin = 0
    plan = None            # 经济规划：{"lv": {4: 回合, 5: 回合}, "reserve": 刷新后至少留几能量, "max_roll": 每回合最多付费刷新}
    pursue = frozenset()   # 转型目标：这些 4 阶主C 里先到手的那个
    pursue_deadline = 99   # 超过这回合才到手就不转了
    transfer = "none"      # 转型时搬等级的手段
    carry = ""
    maple = ""
    arrival = None         # (回合, 到手时能量, 老主C 等级, 搬了多少级, 4 阶主C 名字)

    def __init__(self, q: dict, rng: random.Random, A: dict):
        self.A = A
        self.rng = rng
        c = q["ctx"]
        self.r = int(c.get("round") or 1)
        self.L = int(c.get("level") or 1)
        self.xp = int(c.get("xp") or 0)
        self.gold = int(c.get("gold") or 0)
        ban = str(c.get("banned") or "")
        self.ban = BAN_NORM.get(ban, ban)
        self.boosts = set(c.get("lord_boosts") or [])
        hand = list(c.get("hand") or [])
        self.red_leaf = "红叶染锋" in self.boosts or "红叶染锋" in hand
        if "翩跹舞步·新生" in hand:          # 自用牌，拿着就会用（为枫叶舞挑强化 + 每回合 +2）
            self.boosts.add("翩跹舞步·新生")
        self.route_carry, core, extra = route_core(str(c.get("route") or ""))
        self.groups = [([h for h in g if FACTION.get(h) != self.ban], pr) for gs, pr in ((core, 3), (extra, 2.5)) for g in gs]
        self.groups = [(g, pr) for g, pr in self.groups if g]
        self.core = {h for g, _pr in self.groups for h in g}
        self.route_key = str(c.get("route") or "")
        # 合格接班人 = 各候选路线（route_min_engine 里有主C 的路线，派系没被禁）的主C，且不是 1 阶；
        # 多路线核心卡（2026-10-13，同 rules.flex_core_cards）= 这些路线 core_set 里写明的单个英雄（含主C）
        self.successors, self.flex_core = set(), set()
        for k in _route_keys():
            rc = RM[k]["carry"]
            if rc not in HEROES or FACTION.get(rc) == self.ban:
                continue
            if HEROES.get(rc, 1) >= 2:
                self.successors.add(rc)
            for u in RM[k].get("core_set") or []:
                t = str(u.get("unit") if isinstance(u, dict) else u)
                if t in HEROES and FACTION.get(t) != self.ban:
                    self.flex_core.add(t)
        self.succeeded = None
        self.temp_swaps = 0
        self.units: dict[str, Unit] = {}
        for n, b in (q.get("board") or {}).items():
            if n not in HEROES:
                continue
            awk_left = b.get("awk_left")
            merges = 3 - awk_left if isinstance(awk_left, int) else 0
            lv = b.get("lv") if isinstance(b.get("lv"), (int, float)) else 1
            self.units[n] = Unit(n, lv, merges, merges >= 3, int(b.get("eq") or 0))
        self.hand_heroes = [h for h in hand if h in HEROES]
        # 手里有场上同名 → 合成（规则：手牌英雄拖到场上同名 = 合成）
        rest = []
        for h in self.hand_heroes:
            if h in self.units:
                self.merge(h)
            else:
                rest.append(h)
        self.hand_heroes = rest
        self.carry = c.get("carry") or ""
        self._hand_to_deploy = list(rest)
        self.maple = c.get("maple") or ""
        self.pool = {n: meta.POOL_COPIES[q_] for n, q_ in HEROES.items()}
        for n in self.pool:
            if self.ban and FACTION.get(n) == self.ban:
                self.pool[n] = 0
        for n in list(self.units) + self.hand_heroes:
            if n in self.pool:
                self.pool[n] = max(0, self.pool[n] - 1 - (self.units[n].merges if n in self.units else 0))
        self.core_lv = 0.0               # 阿科米亚核心等级
        self.flash_kills = 0.0
        self.coins_on_carry = 0          # 本回合对主C 用的效果牌张数（沈梦溪 / 镜）
        self.carry_aspd = 0.0            # 狄仁杰：古币给的永久攻速
        self.used_red_leaf = False
        for h in sorted(self._hand_to_deploy, key=lambda x: x not in self.core):   # 手里的英雄有位置就上（路线件先上）
            if h in self.hand_heroes and (len(self.units) < self.cap() or h in self.core):
                self.hand_heroes.remove(h)
                self.deploy(h)
        self.wait_target = ""            # 等待策略：这个主C 一到手就换
        self.trace = []

    # ---- 临时主C（2026-10-13：前期 1~5 回合临时主C 只需要 1 个；不为临时主C 多买卡觉醒，多买中期最终 C 位的配套卡）----
    def is_temp(self, n=None):
        n = self.carry if n is None else n
        u = self.units.get(n)
        return bool(n) and is_temp_hero(n, bool(u and u.awk))

    def set_route_for(self, carry):
        """换成某路线主C 后，买牌目标跟着换成他那条路线的核心。"""
        k = route_of_carry(carry, self.route_key)
        if not k:
            return
        self.route_key = k
        self.route_carry, core, extra = route_core(k)
        self.groups = [([h for h in g if FACTION.get(h) != self.ban], pr) for gs, pr in ((core, 3), (extra, 2.5)) for g in gs]
        self.groups = [(g, pr) for g, pr in self.groups if g]
        self.core = {h for g, _pr in self.groups for h in g}

    def succession(self):
        """临时主C 在合格接班人到手（场上 / 手里）时被顶掉：换主C（卖掉枫叶舞宿主换绑、装备退回给新主C），
        临时主C 身上的等级 / 枫叶舞投入随之丢掉（红叶染锋保留 50%）。不是固定惩罚 —— 接班人什么时候到手由商店抽牌决定。"""
        if not self.is_temp():
            return False
        owned = [n for n in self.successors if n in self.units or n in self.hand_heroes]
        if not owned:
            return False
        new = max(owned, key=lambda n: (n == self.route_carry, self.units[n].lv if n in self.units else 0))
        self.switch_carry(new)
        self.set_route_for(new)
        self.succeeded = (self.r, new)
        return True

    # ---- 基本操作 ----
    def cap(self):
        return meta.pop_cap_at(min(self.L, 6))

    merge_count = 0

    def level_card(self, card):
        """按卡面加等级：指定的给主C（只限河洛的走 give_coin），"随机1名"那份按人数分摊期望给主C。"""
        q, name, cost, heluo, amt, rnd = _LC[card]
        if name in SAME_TIER_CARDS:                 # 有福同享：场上和主C 同品阶的英雄各 +N
            cq = HEROES.get(self.carry)
            for n_, u_ in self.units.items():
                if HEROES.get(n_) == cq:
                    u_.lv += SAME_TIER_CARDS[name]
            if self.carry_unit() is not None:       # 初始化阶段（手里同名合成送锦囊）还没有主C / 计数器：同 give_carry 的处理
                self.coins_on_carry = getattr(self, "coins_on_carry", 0) + 1
            return
        if name in TIERX_CARDS:                     # 封神一瞬：主C 获得 品阶×N 级
            self.give_carry(TIERX_CARDS[name] * HEROES.get(self.carry, 1), card=True)
            return
        if name == "八方来援":                       # 场上每有1种阵营，额外触发1次
            amt *= 1 + len({FACTION.get(x) for x in self.units if FACTION.get(x) not in ("", "无阵营")})
        if heluo:
            self.give_coin(amt)
        else:
            self.give_carry(amt, card=True)
        if rnd and self.units:
            self.give_carry(rnd / len(self.units))

    def merge(self, n):
        self.merge_count += 1
        if self.merge_count % 3 == 0 and self.A.get("kit_pick"):     # 每合成 3 次送 1 个效果锦囊（品阶 = 棋手等级，封顶 5）
            pool = EFFECT_BY_Q.get(min(self.L, 5)) or []
            if pool:
                opts = [c for c in self.rng.sample(pool, min(self.A["kit_pick"], len(pool))) if c in _LC]
                if opts:
                    self.level_card(max(opts, key=lambda c: _LC[c][4]))
        u = self.units[n]
        u.lv += self.A["merge_lv"] + (1 if "曜" in self.units and n != "曜" else 0)
        u.merges += 1
        if u.merges >= 3 and not u.awk:
            u.awk = True
            if n == "马可波罗":
                self.on_deploy(n)       # 觉醒卡回手重新打出 = 再触发一次登场

    def on_deploy(self, n):
        u = self.units[n]
        if n == "马可波罗":              # 登场：自身及阿科米亚核心等级+2（觉醒 +4）
            u.lv += 2 * u.k
            self.core_lv += 2 * u.k
        if n == "明世隐":                # 登场：可转化的等级+2（觉醒 +4）—— 一次性多转 2
            self._mingshi_bonus = getattr(self, "_mingshi_bonus", 0) + 2 * u.k

    def value_of(self, n):            # 满员时换下谁：非主C、非核心、等级最低的
        if n == self.carry:
            return 1e9
        return (1e6 if n in self.core else 0) + self.units[n].lv

    def deploy(self, n):
        if n in self.units:
            self.merge(n)
            return
        if len(self.units) >= self.cap():
            victim = min(self.units, key=self.value_of)
            if self.value_of(victim) >= 1e6 and n not in self.core and n != self.carry:
                return                   # 满员且全是核心：不上
            self.sell(victim)
        self.units[n] = Unit(n)
        self.on_deploy(n)

    def sell(self, n):
        u = self.units.pop(n, None)
        if u is None:
            return None
        self.pool[n] = self.pool.get(n, 0) + 1 + u.merges
        if self.maple == n:
            self.maple = ""
        return u

    def switch_carry(self, new):
        """换主C：新主C 上场（在手里就打出）；枫叶舞绑在别人身上 → 卖掉被绑者重绑（规则：绑定后要卖掉才能重绑）；
        老主C 被卖时装备退回给新主C；红叶染锋：被绑者离场后 50% 真实等级给下一个枫叶舞英雄（限一次）。
        阿离单C（2026-10-12）：枫叶舞绑在谁身上谁就是主C —— 主C 与枫叶舞英雄永远是同一个人，没有"换主C 不换绑"。"""
        old = self.carry
        old_temp = self.is_temp(old) and old in self.units
        if new not in self.units:
            if new in self.hand_heroes:
                self.hand_heroes.remove(new)
            self.deploy(new)
            if new not in self.units:
                return
        self.carry = new
        # 临时主C 之间互换（1 阶 → 1 阶，2026-10-13）：单C 打法下旧临时主C 没有留场价值，卖掉（他身上的合成 / 等级丢掉），
        # 枫叶舞若绑着他走下面的"卖宿主换绑"。装备随卖出退回手牌、给新主C。
        if old_temp and self.is_temp(new) and old != new and self.maple != old:
            u = self.sell(old)
            if u is not None:
                self.units[new].items = min(3, self.units[new].items + u.items)
        if old_temp and self.is_temp(new) and old != new:
            self.temp_swaps += 1
        if self.maple and self.maple != new:
            binder = self.sell(self.maple)
            if binder is not None:
                if self.red_leaf and not self.used_red_leaf:
                    self.units[new].lv += 0.5 * binder.lv
                    self.used_red_leaf = True
                if binder.name == old:
                    self.units[new].items = min(3, self.units[new].items + binder.items)
        self.maple = new
        self.carry_aspd = 0.0

    # ---- 商店 ----
    def draw(self, level):
        odds = meta.TIER_ODDS.get(min(level, 6)) or {1: 1.0}
        tiers, w = zip(*odds.items())
        t = self.rng.choices(tiers, w)[0]
        names = [n for n, q in HEROES.items() if q == t and self.pool[n] > 0]
        if not names:
            return None
        return self.rng.choices(names, [self.pool[n] for n in names])[0]

    def wants(self, n):
        """买牌优先级（所有选项同一套）：主C 同名 > 等待中的主C > 缺的核心 > 核心件同名（觉醒引擎）。0 = 不买。"""
        if n is None:
            return 0
        if self.pursue and self.arrival is None and n in self.pursue and self.r <= self.pursue_deadline:
            return 5
        if self.is_temp():
            owned = n in self.units or n in self.hand_heroes
            if n in self.successors and not owned:
                return 5                      # 最终主C（合格接班人）见到就买
            if n == self.carry:
                return 0                      # 不为临时主C 买同名凑觉醒
            if n in self.flex_core and not owned:
                return 3                      # 多条路线的核心卡，各买一张
        elif n == self.carry:
            return 4
        if self.wait_target and n == self.wait_target:
            return 3.5
        for g, pr in self.groups:
            if n in g and not any(x in self.units or x in self.hand_heroes for x in g):
                return pr
        if n in self.core and n in self.units and not self.units[n].awk:
            return 1
        return 0

    def shop_view(self):
        if self.A.get("level_cards") and self.rng.random() < meta.LEVEL_CARD_HIT.get(min(self.L, 6), 0.0):
            cands = [c for c in LEVEL_CARDS if c[0] <= meta.MAX_TIER_BY_LEVEL.get(min(self.L, 6), 1)]
            if cands:
                c = self.rng.choice(cands)
                if self.gold >= c[2]:
                    self.gold -= c[2]
                    self.level_card(c[1])
        shop = [self.draw(self.L) for _ in range(meta.SHOP_SLOTS)]
        for n in sorted(shop, key=lambda x: -self.wants(x)):
            if self.wants(n) and self.gold >= HERO_PRICE and self.pool.get(n, 0) > 0:
                self.gold -= HERO_PRICE
                self.pool[n] -= 1
                if self.is_temp() and n in self.successors and n != self.carry:
                    self.hand_heroes.append(n)
                    self.succession()
                elif self.wait_target and n == self.wait_target and n != self.carry:
                    self.hand_heroes.append(n)
                    self.switch_carry(n)
                elif self.pursue and self.arrival is None and n in self.pursue and n != self.carry:
                    self.transition(n)
                else:
                    self.deploy(n)

    def align_carry(self, name):
        """起点对齐成单C 口径：name 当主C；枫叶舞没绑 → 直接绑他；绑在别人身上 → 卖掉宿主换绑（同 switch_carry）。"""
        if self.maple and self.maple not in self.units:
            self.maple = ""                      # 宿主不在场上（记录过期）：当作没绑
        if name != self.carry or (self.maple and self.maple != name):
            self.switch_carry(name)
        elif not self.maple and name in self.units:
            self.maple = name

    def transition(self, new):
        """转型：4 阶主C 到手 → 上场、换主C、按 self.transfer 搬等级（只按卡面原文）：
          red_leaf  红叶染锋（枫叶舞强化）：枫叶舞英雄离场后，其真实等级的 50% 保留给下一个枫叶舞英雄 —— 卖掉老主C、枫叶舞换绑，可指定
          steal     夺取之术：指定1名英雄，夺取最近1名己方英雄的等级（最多 10 级）—— 可指定，上限 10
          stars     众星拱月：指定1名英雄，夺取其他英雄各 10 级 —— 可指定
          will      遗志：指定场上1名英雄将其移除，并转移其等级给 1 名**随机**英雄 —— 落到新主C 的概率 = 1/(场上其他人数)
          legacy    传承：指定场上1名 10 级以上英雄，将其等级**随机**分配给 3 名其他英雄 —— 新主C 被抽中才有，份额随机
          none      没有搬等级手段：老主C 卖掉换绑枫叶舞，新主C 从当前等级起步"""
        old = self.carry
        oldu = self.units.get(old)
        old_lv = oldu.lv if oldu else 0.0
        if new not in self.units:
            self.deploy(new)
            if new not in self.units:
                return
        nu = self.units[new]
        lv0 = nu.lv
        m = self.transfer
        self.red_leaf = m == "red_leaf"            # 研究口径：只有选了红叶染锋这一档才有它
        if m == "steal" and oldu:
            g = min(10.0, oldu.lv)
            oldu.lv -= g
            nu.lv += g
        elif m == "stars":
            for n, u in self.units.items():
                if n != new:
                    g = min(10.0, u.lv)
                    u.lv -= g
                    nu.lv += g
        elif m == "will" and oldu:
            if self.maple == old:
                self.maple = ""
            self.units.pop(old, None)
            self.pool[old] = self.pool.get(old, 0) + 1 + oldu.merges
            others = list(self.units)
            if others:
                self.units[self.rng.choice(others)].lv += oldu.lv
            nu.items = min(3, nu.items + oldu.items)
        elif m == "legacy" and oldu and oldu.lv >= 10:
            others = sorted((n for n in self.units if n != old), key=lambda n: self.units[n].lv)
            picks = others[:3]                         # 2026-10-13 卡面：随机分配给"其他英雄中等级最低的3名"
            cuts = sorted(self.rng.random() for _ in range(len(picks) - 1))
            shares = [b - a for a, b in zip([0.0] + cuts, cuts + [1.0])]
            for n, sh in zip(picks, shares):
                self.units[n].lv += oldu.lv * sh
            oldu.lv = 0.0
        self.switch_carry(new)
        got = (self.units[new].lv - lv0) if new in self.units else 0.0
        self.arrival = (self.r, self.gold, round(old_lv, 1), round(got, 1), new)

    def missing_any(self):
        if self.pursue and self.arrival is None and self.L >= 4 and self.r <= self.pursue_deadline:
            return True
        if self.is_temp():
            return any(n not in self.units and n not in self.hand_heroes for n in self.successors | self.flex_core)
        if self.carry in self.units and not self.units[self.carry].awk:
            return True
        return any(not any(x in self.units for x in g) for g, _pr in self.groups) or bool(self.wait_target and self.wait_target != self.carry)

    def plan_level(self):
        """本回合按经济规划该到几级：≤3 级照进度表，4/5 级按规划的回合。"""
        t = min(meta.level_target(self.r), 3)
        for lv, rd in sorted(self.plan["lv"].items()):
            if self.r >= rd:
                t = max(t, lv)
        return t

    def shopping(self):
        if self.plan is not None:
            return self.shopping_plan()
        # 升级：落后于进度表且付得起（meta.LEVEL_TARGET）
        while self.L < 6 and self.L < meta.level_target(self.r):
            cost = max(1, XP_NEED.get(self.L, 27) - self.xp)
            if self.gold < cost:
                break
            self.gold -= cost
            self.L += 1
            self.xp = 0
        self.shop_view()
        for _ in range(self.A["free_refresh"]):
            self.shop_view()
        for _ in range(self.A["paid_refresh_max"]):
            if not self.missing_any() or self.gold < REFRESH_PRICE + HERO_PRICE:
                break
            self.gold -= REFRESH_PRICE
            self.shop_view()

    def shopping_plan(self):
        P = self.plan
        saving = False
        while self.L < 6 and self.L < self.plan_level():
            cost = max(1, XP_NEED.get(self.L, 27) - self.xp)
            if self.gold < cost:
                saving = True            # 该升没钱：本回合攒钱，不刷
                break
            self.gold -= cost
            self.L += 1
            self.xp = 0
        self.shop_view()
        for _ in range(self.A["free_refresh"]):
            self.shop_view()
        if saving:
            return
        # 下回合就要升级：给升级费留钱（下回合收入 + 自然经验补不齐的部分）
        nxt = self.r + 1
        keep = 0
        for lv, rd in P["lv"].items():
            if rd == nxt and self.L == lv - 1:
                keep = max(keep, XP_NEED.get(self.L, 27) - self.xp - meta.lord_xp_gain(self.r) - meta.energy_income(nxt))
        for _ in range(P.get("max_roll", 10)):
            if not self.missing_any() or self.gold - REFRESH_PRICE < max(P["reserve"], keep) or self.gold < REFRESH_PRICE + HERO_PRICE:
                break
            self.gold -= REFRESH_PRICE
            self.shop_view()

    def opponents(self):
        lv = meta.level_target(self.r)
        for _ in range(self.A["opp_take"]):
            n = self.draw(lv)
            if n:
                self.pool[n] -= 1

    # ---- 回合开始：整备 ----
    def carry_unit(self):
        return self.units.get(self.carry)

    def give_carry(self, lv, card=False, coin=False):
        cu = self.carry_unit()
        if cu is None:
            return
        cu.lv += lv
        if card:
            self.coins_on_carry += 1
        if coin and self.carry == "狄仁杰":    # 对狄仁杰使用古币时，其攻击速度永久+5%（觉醒 10%）
            self.carry_aspd += 0.05 * cu.k

    def heluo_carry(self):
        return FACTION.get(self.carry) == "河洛" and self.carry in self.units

    def give_coin(self, lv, src=""):
        """古币（只能给【河洛】英雄）：主C 是河洛且不是来源本人 → 给主C；否则给场上等级最高的其他河洛英雄
        （"距离最近的河洛英雄"——站位上会把将来的主C 放在旁边，所以按"最可能接班的人"给）。"""
        if self.heluo_carry() and self.carry != src:
            self.give_carry(lv, card=True, coin=True)
            return
        others = [u for n, u in self.units.items() if FACTION.get(n) == "河洛" and n != src]
        if others:
            max(others, key=lambda u: u.lv).lv += lv

    def zhengbei(self, n):
        """一个单位触发一次整备。返回是否触发（给韩信 / 朵莉亚计数）。"""
        u = self.units.get(n)
        if u is None:
            return False
        k = u.k
        if n == "李元芳":                 # 整备：对距离最近的【河洛】英雄使用1张古币（觉醒 精致古币 +4）
            self.give_coin(2 * k, src="李元芳")
        elif n == "程咬金":               # 整备：获得1张古币，对自身使用1张古币 —— 获得的那张给河洛主C
            u.lv += 2 * k
            self.give_coin(2 * k, src="程咬金")
        elif n == "米莱狄":               # 整备：阿科米亚核心等级额外+3（觉醒触发2次）
            self.core_lv += 3 * k
        elif n == "亚连":                 # 整备：穿戴装备最多的1名英雄等级+2，且核心+1（觉醒触发2次）；主C 装备最多
            self.give_carry(2 * k)
            self.core_lv += 1 * k
        elif n == "孙悟空":               # 整备：拥有闪现效果的英雄等级+2（觉醒 +4）
            if self.carry in FLASH:
                self.give_carry(2 * k)
        elif n == "百里玄策":             # 整备：闪现英雄上回合每击败1名敌方英雄，随机1名闪现英雄等级+1（觉醒 +2）
            fl = [x for x in self.units if x in FLASH]
            if fl and self.carry in fl:
                self.give_carry(k * self.flash_kills / len(fl))
        elif n == "苏烈":
            pass
        return n in ZHENGBEI

    def triggered(self, kind, times=1):
        """每当你的英雄触发整备 / 开团：韩信 +2（觉醒 +4）；张良（开团）随机3名 +1（觉醒 全体）；朵莉亚（整备）随机3名 +1、核心 +1。"""
        n_units = max(1, len(self.units))
        for _ in range(times):
            if "韩信" in self.units:
                u = self.units["韩信"]
                u.lv += 2 * u.k
            if kind == "开团" and "张良" in self.units:
                z = self.units["张良"]
                p = 1.0 if z.awk else min(1.0, 3 / n_units)
                self.give_carry(p)
            if kind == "整备" and "朵莉亚" in self.units:
                d = self.units["朵莉亚"]
                p = 1.0 if d.awk else min(1.0, 3 / n_units)
                self.give_carry(p)
                self.core_lv += 1

    def round_start(self):
        self.coins_on_carry = 0
        n_zb = 0
        for n in list(self.units):
            if n in ZHENGBEI and self.zhengbei(n):
                n_zb += 1
        self.triggered("整备", n_zb)
        # 枫叶舞强化：翩跹舞步·新生 —— 枫叶舞英雄每回合等级+2
        if "翩跹舞步·新生" in self.boosts and self.carry in self.units:    # 枫叶舞英雄（= 主C）每回合等级+2
            self.units[self.carry].lv += 2
        if self.A["level_card_rate"]:
            self.give_carry(self.A["level_card_rate"])
        self.coins_on_carry += self.A["effect_cards"]
        # 装备
        cu = self.carry_unit()
        if cu and self.A["item_every"] and self.r % self.A["item_every"] == 0:
            cu.items = min(3, cu.items + 1)

    # ---- 战斗 ----
    def battle(self) -> float:
        cu = self.carry_unit()
        A = self.A
        temp = 0.0
        n_kt = 0
        totems = 1 if "少司缘" in self.units else 0
        # 开团
        for n in list(self.units):
            u = self.units[n]
            if n not in KAITUAN:
                continue
            n_kt += 1
            if n == "虞姬":                   # 开团：场上每有1个图腾，自身及随机2名【大河流域】英雄等级+1（觉醒 4 名）
                if totems:
                    u.lv += totems
                    if self.carry != "虞姬" and FACTION.get(self.carry) == "大河流域":
                        dh = max(1, sum(1 for x in self.units if FACTION.get(x) == "大河流域") - 1)
                        self.give_carry(totems * min(1.0, 2 * u.k / dh))
            elif n == "云中君" and cu:         # 开团：你的英雄每穿戴1件装备，该英雄等级永久+1（觉醒 +2）
                self.give_carry(cu.items * u.k)
            elif n == "雅典娜":                # 开团：随机触发2名【日落海】英雄的整备效果（觉醒 4 名）
                sun = [x for x in self.units if FACTION.get(x) == "日落海" and x in ZHENGBEI]
                fired = 0
                for _ in range(2 * u.k):
                    if sun:
                        if self.zhengbei(self.rng.choice(sun)):
                            fired += 1
                self.triggered("整备", fired)
        self.triggered("开团", n_kt)
        cu = self.carry_unit()
        temp = {n: 0.0 for n in self.units}
        # 阿科米亚核心：开战时以临时等级平均分给己方日落海英雄（单人上限 50%）
        sun = [x for x in self.units if FACTION.get(x) == "日落海"]
        if self.core_lv and sun:
            for x in sun:
                temp[x] += min(self.core_lv / len(sun), 0.5 * self.core_lv)
        # 牺牲件阵亡
        sac = [n for n in self.units if n in SACRIFICE and n != self.carry]
        if "钟馗" in self.units:             # 钟馗：自身及左右1格的英雄获得牺牲 —— 记 2 个额外牺牲单位
            sac += ["_钟馗邻"] * 2
        deaths = 0
        for n in sac:
            if self.rng.random() < A["p_die"]:
                deaths += 1
                u = self.units.get(n)
                if n == "铠":                                           # 牺牲：你的【河洛】英雄获得5临时等级（觉醒 10）
                    for x in self.units:
                        if FACTION.get(x) == "河洛":
                            temp[x] += 5 * u.k
                if n == "盾山":                                         # 牺牲：获得1张古币（觉醒 精致古币）→ 下回合打出
                    self._pending_coin = getattr(self, "_pending_coin", 0) + 2 * u.k
                if n == "苏烈" and u is not None:                       # 牺牲：苏烈的等级永久+2
                    u.lv += 2
        if "李信" in self.units:                                       # 每当你拥有牺牲效果的英雄阵亡时，李信临时等级+7（觉醒 14）
            temp["李信"] += 7 * self.units["李信"].k * deaths
        # 枫叶舞：强化效果跟着枫叶舞英雄（不一定是主C）
        mh = self.units.get(self.carry)                                 # 枫叶舞英雄 = 主C
        if mh is not None and "枫舞扶摇" in self.boosts:              # 每回合首次转移后获得5临时等级，本局每击败1名英雄 +1
            temp[self.carry] += 5 + A["kills"] * max(0, self.r - 1) * 0.5
        # 明世隐：距离最近的英雄（站位让主C 最近），临时等级中的 2 级转永久（觉醒 4；登场多 2）
        perm_from_temp = 0.0
        if "明世隐" in self.units and cu is not None and temp.get(self.carry, 0) > 0:
            m = self.units["明世隐"]
            conv = min(temp[self.carry], 2 * m.k + self._mingshi_bonus)
            self._mingshi_bonus = 0
            cu.lv += conv
            temp[self.carry] -= conv
            perm_from_temp = conv
        # 战力：每个单位按卡面强度曲线算，再乘只作用于自己的词条
        powers = {}
        for n, u in self.units.items():
            lvb = u.lv + temp.get(n, 0.0)
            p = base_dmg(n, lvb) * (1 + 0.2 * u.items) / (1 + 0.2 * assumed_items(lvb))   # hero_strength 的装备假设换成实际件数
            mult = 1.0
            E = mana_max(n)
            red = 0.0
            if n == "百里守约":                                   # 牺牲英雄阵亡 → 本回合法力上限 -20（觉醒 -40）
                red = 20 * u.k * deaths
            elif n == "沈梦溪" and n == self.carry:              # 每对沈梦溪使用1张效果牌，本回合法力上限 -15（觉醒 -30）
                red = 15 * u.k * self.coins_on_carry
            if E and red:
                mult *= 1 + A["skill_share"] * (E / max(E - red, 0.25 * E) - 1)
            if n == "狄仁杰" and n == self.carry and self.carry_aspd:
                mult *= 1 + self.carry_aspd
            if n == "艾琳":                                       # 开团：获得自身等级×1% 攻速（觉醒 2%）
                mult *= 1 + 0.01 * u.k * lvb * (1 - A["skill_share"])
            if n == self.carry and "枫华之刃" in self.boosts:     # 转移后获得25%伤害加成（枫叶舞英雄 = 主C）
                mult *= 1 + 0.25 * A["maple_blade_uptime"]
            powers[n] = p * mult
        if cu is not None and self.carry == "镜":                 # 每使用2张效果牌，等级+1
            cu.lv += 0.5 * cu.k * self.coins_on_carry
        if mh is not None and "翩跹舞步·蓄势" in self.boosts:     # 击败敌方英雄时永久等级+1
            mh.lv += A["kills"]
        # 战后：因缘图腾（每合计 25 次普攻，所有己方英雄永久 +1）
        if totems:
            hits = sum(a_ * min(t, 20.0) for a_, t in (aps(n, self.units[n].lv) for n in self.units))
            if "少司缘" in self.units and self.units["少司缘"].awk:
                hits *= 1.5                                             # "召唤并强化"——强化内容卡面没写，按 ×1.5 估（局限）
            for u in self.units.values():
                u.lv += hits / 25
        self.flash_kills = A["kills"] if self.carry in FLASH else 0.0
        pc = self._pending_coin
        if pc:
            self.give_coin(pc)
            self._pending_coin = 0
        top = sorted(powers.values(), reverse=True) + [0.0, 0.0]
        team = top[0] + 0.5 * top[1]
        carry_p = powers.get(self.carry, 0.0)
        self.trace.append((self.r, self.carry, round(cu.lv, 1) if cu else 0, round(perm_from_temp, 1), round(carry_p), round(team)))
        return team, carry_p

    def next_round(self):
        self.xp += meta.lord_xp_gain(self.r)
        self.r += 1
        self.gold += meta.energy_income(self.r)
        while self.L < 6 and self.xp >= XP_NEED.get(self.L, 27):
            self.xp -= XP_NEED.get(self.L, 27)
            self.L += 1
        self.opponents()


def run_option(q, option_name, policy, seed, A, horizon=HORIZON):
    """option_name：现在当主C 的人；policy：'stick' / 'wait'（还没到手的路线主C 一到手就换）/ ('later', 人, 回合)（到那回合再换给他）。
    返回 (结果, 第10/12回合战力, 最终主C, 最终等级, trace)"""
    sim = Sim(q, random.Random(seed), A)
    later = policy if isinstance(policy, tuple) else None
    sim.align_carry(option_name)
    if policy == "wait" and sim.route_carry and sim.route_carry != sim.carry and FACTION.get(sim.route_carry) != sim.ban:
        sim.wait_target = sim.route_carry
        if sim.route_carry in sim.units or sim.route_carry in sim.hand_heroes:
            sim.switch_carry(sim.route_carry)
    # 本回合（题目所在的准备阶段）：整备已经发生，只剩买牌和战斗
    sim.shopping()
    powers = {}
    while True:
        powers[sim.r] = sim.battle()
        if sim.r >= LAST_ROUND:
            break
        sim.next_round()
        sim.round_start()
        sim.succession()
        if later and sim.r == later[2] and later[1] != sim.carry and (later[1] in sim.units or later[1] in sim.hand_heroes) \
                and not (is_temp_hero(later[1]) and not sim.is_temp()):     # 已由最终主C 接班：不再换回临时主C
            sim.switch_carry(later[1])
        sim.shopping()
    rs = [r for r in horizon if r in powers] or [max(powers)]
    cu = sim.carry_unit()
    team = st.mean(powers[r][0] for r in rs)
    carry_only = st.mean(powers[r][1] for r in rs)
    return (team, (powers.get(10) or (0, 0))[0], (powers.get(12) or (0, 0))[0], sim.carry, (cu.lv if cu else 0), sim.trace, carry_only)


def run_plan(q, A, seed, plan=None, pursue=(), deadline=99, transfer="none", horizon=(12, 13, 14), last_round=14):
    """转型研究用：按经济规划 plan 打，pursue 里的 4 阶主C 先到手的那个在 deadline 前到手就转过去（transfer 搬等级）。
    返回 (队伍战力均值, arrival, 每回合 (棋手等级, 能量, 动作) 轨迹)。"""
    sim = Sim(q, random.Random(seed), A)
    sim.align_carry(sim.carry)
    sim.plan = plan
    sim.pursue = frozenset(h for h in pursue if FACTION.get(h) != sim.ban)
    sim.pursue_deadline = deadline
    sim.transfer = transfer
    sim.shopping()
    pw = {}
    while True:
        pw[sim.r] = sim.battle()[0]
        if sim.r >= last_round:
            break
        sim.next_round()
        sim.round_start()
        sim.shopping()
    rs = [r for r in horizon if r in pw] or [max(pw)]
    return st.mean(pw[r] for r in rs), sim.arrival, pw


# ---------------------------------------------------------------------------
# 转型研究：2~3 阶主C 滚等级 → 4 阶主C 到手后换过去（2026-10-12）
# ---------------------------------------------------------------------------
FOUR_CARRIES = ("蒙犽", "诸葛亮", "嬴政", "艾琳", "孙悟空", "李信")   # 4 阶、主C 审定评分 ≥8（hero_carry_reviewed.json，按卡面审的）
TRANSFERS = ("none", "red_leaf", "steal", "stars", "will", "legacy")
TRANSFER_CN = {"none": "无搬等级手段", "red_leaf": "红叶染锋(50%,可指定)", "steal": "夺取之术(≤10级,可指定)",
               "stars": "众星拱月(每人10级,可指定)", "will": "遗志(随机1人)", "legacy": "传承(随机3人)"}


def plan_grid(r0):
    out = []
    for a4 in (5, 6, 7, 8, 9):
        if a4 < r0:
            continue
        for a5 in (a4 + 1, a4 + 2, a4 + 3, 99):
            for rsv in (0, 4, 10):
                out.append({"lv": {4: a4, 5: a5}, "reserve": rsv, "max_roll": 10})
    return out


def plan_name(p):
    if p is None:
        return "默认进度表(R7 4级/R10 5级)+刷到剩5"
    a5 = p["lv"][5]
    return f"R{p['lv'][4]}升4级/" + (f"R{a5}升5级" if a5 < 99 else "不主动升5级") + f"/刷新后留{p['reserve']}"


def transition_states(qs, k=8):
    """起点：真实 carry 题里现任是 2~3 阶路线主C、第 4~7 回合、等级读到的局面（每局取一个）。"""
    out, seen = [], set()
    for q in qs:
        c = q["ctx"]
        cur = c.get("carry") or ""
        rc = route_core(str(c.get("route") or ""))[0]
        b = (q.get("board") or {}).get(cur) or {}
        if (HEROES.get(cur) in (2, 3) and cur == rc and 4 <= int(c.get("round") or 0) <= 7
                and isinstance(b.get("lv"), (int, float)) and q["_match"] not in seen):
            out.append(q)
            seen.add(q["_match"])
    return out[-k:]


def transition_study(qs, A, n1=80, n2=150, k=8):
    states = transition_states(qs, k)
    report = {"states": [], "four": FOUR_CARRIES, "tier_scale": TIER_SCALE}
    buckets = {}
    for si, q in enumerate(states):
        r0 = int(q["ctx"]["round"])
        grid = plan_grid(r0)
        rowA, rowB = [], []
        for pi, pl in enumerate([None] + grid):
            va, vb, arr = [], [], []
            for i in range(n1):
                seed = 9100003 * (si + 1) + i
                va.append(run_plan(q, A, seed, plan=pl)[0])
                if pl is not None:
                    v, a, _pw = run_plan(q, A, seed, plan=pl, pursue=FOUR_CARRIES, deadline=12)
                    vb.append(v)
                    arr.append(a)
            rowA.append((st.mean(va), pi))
            if pl is not None:
                got = [a for a in arr if a]
                rowB.append((st.mean(vb), pi, {
                    "p_by8": sum(1 for a in got if a[0] <= 8) / n1, "p_by10": sum(1 for a in got if a[0] <= 10) / n1,
                    "p_by12": len(got) / n1, "arr_round": st.mean(a[0] for a in got) if got else None,
                    "gold_at": st.mean(a[1] for a in got) if got else None}))
        plans = [None] + grid
        bestA = max(rowA)
        defA = rowA[0]
        rowB.sort(key=lambda x: -x[0])
        st_rep = {"id": q["id"], "brief": brief(q), "round": r0,
                  "A_default": {"plan": plan_name(None), "ev": round(defA[0])},
                  "A_best": {"plan": plan_name(plans[bestA[1]]), "ev": round(bestA[0])},
                  "B_top": [{"plan": plan_name(plans[pi]), "ev": round(v), **{k_: (round(x, 2) if isinstance(x, float) else x) for k_, x in info.items()}}
                            for v, pi, info in rowB[:5]],
                  "B_fastest": max(({"plan": plan_name(plans[pi]), "ev": round(v), **info} for v, pi, info in rowB), key=lambda d: (d["p_by8"], d["p_by10"]))}
        # 第二阶段：最好的 2 个规划 × 搬等级手段 × 截止回合
        st_rep["transfer"] = []
        for v, pi, _info in rowB[:2]:
            pl = plans[pi]
            for m in TRANSFERS:
                for dl in (8, 10, 12):
                    vals = []
                    for i in range(n2):
                        seed = 7700001 * (si + 1) + i
                        val, a, pw = run_plan(q, A, seed, plan=pl, pursue=FOUR_CARRIES, deadline=dl, transfer=m)
                        vals.append(val)
                        if a and dl == 12 and pi == rowB[0][1]:
                            key = (m, a[0])
                            buckets.setdefault(key, []).append((val / bestA[0], a[2], a[3]))
                    st_rep["transfer"].append({"plan": plan_name(pl), "transfer": m, "deadline": dl, "ev": round(st.mean(vals)),
                                               "vs_A_best": round(st.mean(vals) / bestA[0] - 1, 3)})
        report["states"].append(st_rep)
        print(si, q["id"], "A默认", round(defA[0]), "A最优", round(bestA[0]), plan_name(plans[bestA[1]]), "| B最优", round(rowB[0][0]), plan_name(plans[rowB[0][1]]), rowB[0][2], flush=True)
    # 到手回合 × 手段：转型后结果 / A 最优（同一局面）
    report["by_arrival"] = [{"transfer": m, "arrival_round": r, "n": len(v), "ratio_vs_A_best": round(st.mean(x[0] for x in v), 3),
                             "p_better": round(sum(1 for x in v if x[0] > 1) / len(v), 2),
                             "old_lv": round(st.mean(x[1] for x in v), 1), "moved_lv": round(st.mean(x[2] for x in v), 1)}
                            for (m, r), v in sorted(buckets.items()) if len(v) >= 5]
    return report


def evaluate(q, A=None, n=N_SIM, seed0=7):
    A = A or ASSUME
    names = (q.get("raw") or {}).get("names") or {}
    opts = list(q.get("options") or names)
    res = {}
    route_carry = route_core(str(q["ctx"].get("route") or ""))[0]
    for o in opts:
        nm = names.get(o) or (q["ctx"].get("carry") if o == "keep" else o)
        if nm not in HEROES:
            continue
        out = {}
        pols = ["stick"]
        owned = set(q.get("board") or {}) | set(q["ctx"].get("hand") or [])
        if route_carry and route_carry != nm and route_carry not in owned:
            pols.append("wait")
        if o == "keep":          # 现在不换 ≠ 永远不换：把"晚 1~3 回合再换给某个候选"也算进 keep 的价值
            r0 = int(q["ctx"].get("round") or 1)
            for o2 in opts:
                n2 = names.get(o2)
                if o2 != "keep" and n2 in HEROES:
                    for t in (r0 + 1, r0 + 2, r0 + 3):
                        if t <= LAST_ROUND:
                            pols.append(("later", n2, t))
        for pol in pols:
            vals, v10, v12, final_c, lv12, vc = [], [], [], {}, [], []
            for i in range(n):
                s, p10, p12, fc, l12, _tr, cpow = run_option(q, nm, pol, seed0 * 100003 + i, A)
                vc.append(cpow)
                vals.append(s)
                v10.append(p10 or 0)
                v12.append(p12 or 0)
                lv12.append(l12)
                final_c[fc] = final_c.get(fc, 0) + 1
            out[pol] = {"ev": st.mean(vals), "se": st.pstdev(vals) / math.sqrt(len(vals)), "p10": st.mean(v10),
                        "p12": st.mean(v12), "lv12": st.mean(lv12), "final_carry": final_c, "ev_carry": st.mean(vc)}
        best_pol = max(out, key=lambda k: out[k]["ev"])
        pk = lambda k: k if isinstance(k, str) else f"{k[1]}@R{k[2]}"  # noqa: E731
        res[o] = {"name": nm, "best_policy": pk(best_pol), **out[best_pol], "policies": {pk(k): round(v["ev"]) for k, v in out.items()}}
    return res


def label(res):
    if not res:
        return None
    best = max(res, key=lambda o: res[o]["ev"])
    top = res[best]["ev"] or 1.0
    z = {o: (res[o]["ev"] / top - 1) / TAU for o in res}
    m = max(z.values())
    e = {o: math.exp(v - m) for o, v in z.items()}
    s = sum(e.values())
    return best, {o: round(v / s, 4) for o, v in e.items()}


# ---------------------------------------------------------------------------
# 取题
# ---------------------------------------------------------------------------
def load_questions(last_n=60):
    ms = [json.loads(line) for line in (ROOT / "data/laya/iterations/batch2_matches.jsonl").open(encoding="utf-8")]
    ms = sorted(ms, key=lambda m: m["match"])
    runs = [(m["match"], m["run"], m.get("rank")) for m in ms[-last_n:]]
    # 正在跑 / 还没进 batch2 的最新一局（取题用，名次未知）
    known = {r for _m, r, _k in runs}
    newer = sorted(p.name for p in (ROOT / "data/logs").glob("live_*") if p.is_dir() and p.name > max(known))
    for i, r in enumerate(newer):
        runs.append((ms[-1]["match"] + 1 + i, r, None))
    qs = []
    for mt, run, rank in runs:
        p = ROOT / "data/logs" / run / "laya_points.jsonl"
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            if x.get("kind") == "carry":
                x["_match"], x["_rank"] = mt, rank
                qs.append(x)
    return qs


def brief(q):
    c = q["ctx"]
    b = q.get("board") or {}
    return (f"局{q['_match']} R{c.get('round')} 棋手{c.get('level')}级 能量{c.get('gold')} 现任{c.get('carry')} 枫叶舞→{c.get('maple') or '无'}；"
            + "场上 " + "、".join(f"{n}{(v.get('lv') if v.get('lv') is not None else '?')}级" + ("觉醒" if v.get("awk_left") == 0 else "")
                               for n, v in b.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", action="store_true")
    ap.add_argument("--id", default="")
    ap.add_argument("--n", type=int, default=N_SIM)
    ap.add_argument("--sens", action="store_true")
    ap.add_argument("--last", type=int, default=60)
    ap.add_argument("--transition", action="store_true")
    ap.add_argument("--tier-scale", type=float, default=0.0)
    ap.add_argument("--assume", default="", help="覆盖假设，如 level_card_rate=3,p_die=0.8")
    a = ap.parse_args()
    for kv in filter(None, a.assume.split(",")):
        k_, v_ = kv.split("=")
        ASSUME[k_] = type(ASSUME[k_])(float(v_)) if not isinstance(ASSUME[k_], bool) else v_ not in ("0", "False", "false")
    global TIER_SCALE
    TIER_SCALE = a.tier_scale
    qs = load_questions(a.last)
    if a.transition:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        rep_ = transition_study(qs, ASSUME)
        tag = ("" if not TIER_SCALE else f"_ts{TIER_SCALE:g}") + ("_" + a.assume.replace("=", "").replace(",", "_") if a.assume else "")
        (OUT_DIR / f"carry_sim_transition{tag}.json").write_text(json.dumps(rep_, ensure_ascii=False, indent=1), encoding="utf-8")
        print("done transition")
        return 0
    if a.id:
        q = next(x for x in qs if x.get("id") == a.id)
        print(brief(q))
        res = evaluate(q, n=a.n)
        for o, v in res.items():
            print(o, v["name"], round(v["ev"]), "±", round(v["se"]), "R10", round(v["p10"]), "R12", round(v["p12"]), "lv12", round(v["lv12"], 1), v["policies"], v["final_carry"])
        print(label(res))
        for pol in ("stick", "wait"):
            for o in res:
                tr = run_option(q, res[o]["name"], pol, 1, ASSUME)[5]
                print(pol, o, tr)
        return 0
    if not a.questions:
        ap.print_help()
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with (OUT_DIR / "carry_sim_labels.jsonl").open("w", encoding="utf-8") as f:
        for i, q in enumerate(qs):
            res = evaluate(q, n=a.n)
            lb = label(res)
            if not lb:
                continue
            best, dist = lb
            ev = {o: round(v["ev"], 1) for o, v in res.items()}
            f.write(json.dumps({"id": q["id"], "dist": dist, "best": best, "ev": ev}, ensure_ascii=False) + "\n")
            pick = q.get("laya_choice")
            rows.append({"id": q["id"], "match": q["_match"], "rank": q["_rank"], "round": q["ctx"].get("round"),
                         "brief": brief(q), "best": best, "pick": pick, "model": q.get("model"), "used": q.get("used"),
                         "online": q.get("laya"), "dist": dist,
                         "res": {o: {k: (round(v[k], 1) if isinstance(v[k], float) else v[k]) for k in
                                     ("name", "ev", "ev_carry", "se", "p10", "p12", "lv12", "best_policy", "policies", "final_carry")} for o, v in res.items()},
                         "regret": (res[best]["ev"] - res[pick]["ev"]) / res[best]["ev"] if pick in res and res[best]["ev"] else None})
            if i % 5 == 0:
                print(f"{i}/{len(qs)} {time.time() - t0:.0f}s", flush=True)
    (OUT_DIR / "carry_sim_results.json").write_text(json.dumps({"assume": ASSUME, "tau": TAU, "n_sim": a.n, "horizon": HORIZON,
                                                                 "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.sens:
        variants = {"merge_lv=3": {"merge_lv": 3}, "p_die=0.25": {"p_die": 0.25}, "p_die=0.8": {"p_die": 0.8},
                    "kills=0": {"kills": 0.0}, "kills=2": {"kills": 2.0}, "opp_take=21": {"opp_take": 21},
                    "level_card_rate=3": {"level_card_rate": 3.0}, "level_cards=off": {"level_cards": False, "kit_pick": 0},
                    "paid_refresh_max=1": {"paid_refresh_max": 1}, "effect_cards=3": {"effect_cards": 3.0}}
        sens = {}
        base = {r["id"]: r["best"] for r in rows}
        for name, ch in variants.items():
            A = dict(ASSUME, **ch)
            same = tot = 0
            for q in qs:
                if q["id"] not in base:
                    continue
                res = evaluate(q, A=A, n=max(60, a.n // 4))
                lb = label(res)
                if lb:
                    tot += 1
                    same += lb[0] == base[q["id"]]
            sens[name] = {"same_best": same, "total": tot}
            print(name, same, tot, flush=True)
        (OUT_DIR / "carry_sim_sensitivity.json").write_text(json.dumps(sens, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", len(rows), f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
