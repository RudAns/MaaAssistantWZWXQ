"""正向推演引擎：事件分发 + 基本操作。卡牌实现见 cards/（写卡规范 cards/README.md）。

时机（卡面关键词 → 钩子名）：
  登场 on_deploy   整备 on_prepare   开团 on_open   战斗开始 on_battle_start   牺牲 on_sacrifice（阵亡且拥有牺牲）
  阵亡 on_death    击杀 on_kill(n)   凯旋 on_win    败阵 on_lose    退场 on_sell   合成 on_merge   战斗结束 on_battle_end
  回合开始 on_round_start / 回合结束 on_round_end（天赋 / 装备 / 棋手牌 / 手牌里 in_hand 的牌也有）
  整场战斗开始一次 on_battle_begin（天赋 / 棋手牌这类不挂在单位上的，开团之前）
  使用（效果牌 / 天赋牌）on_use(target)   穿戴 on_equip(unit)
"每当 …时"的反应器：react_<事件>(eng, holder, **kw)，holder = 挂着这张卡的单位（天赋 / 棋手牌 / 手牌为 None）。
觉醒 = 卡面效果翻倍：自身时机钩子执行两次（卡面另有写法的卡在实现里自己处理）。

钩子只会收到它签名里声明过的关键字参数（_call 按签名过滤），所以引擎给事件加新参数不会弄坏旧卡。
"""
from __future__ import annotations

import inspect
import json
import math
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from agent.engine import board as B
from .state import State, Unit

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- 实测常量（日志）
MERGE_GAIN = 3            # 合成一张同名 +3 级（第 480~630 局 1452 次合成中位数）
MERGES_PER_JINNANG = 3    # 每合成 3 次送 1 个效果锦囊，品阶 = min(棋手等级, 6)
COIN_LEVELS = 2           # 古币：指定 1 名【河洛】英雄等级 +2（卡面）
BASE_WIN = 0.49           # 每回合赢的比例（实测输 0.51）—— 只用来决定"胜利 / 击杀 / 阵亡"触发，不估强弱
KILL_BASE = 0.3           # 输了也能击杀的敌方比例
DEATH_LOSE, DEATH_WIN = 0.92, 0.78     # 最前排（row0）阵亡率：负 / 胜（2026-10-14 用苏烈"牺牲 +2"量的：250 局 824 个回合，胜负按棋手血量变没变）
# 各行相对最前排的倍数（row0 = 最靠敌方）。row0/1 是实测（row1 胜 0.64 / 负 0.89），row2/3 没有探针，按趋势外推。
ROW_MULT_WIN, ROW_MULT_LOSE = (1.0, 0.82, 0.64, 0.45), (1.0, 0.97, 0.92, 0.87)
FRONT_MULT, BACK_MULT = 1.0, 1.0       # 旧参数（测试里还在 patch），不再参与计算
CAST_SECONDS = 14.0                    # 估放技能次数用的回蓝时间：孙膑"每 3 次技能 +1"回放校准，20 秒偏高 +0.38、12 秒 -0.16（2026-10-14，508 样本）
ATTACK_SECONDS = 4.0                   # 一场战斗里单位"有效普攻时间"秒数（攻速 × 它 = 普攻次数）：走位、放技能、阵亡都少打。
#   回放校准：因缘图腾（每 25 次普攻全队 +1）实测每回合只给 ~0.7 级，取 3~5 秒偏差最小（2026-10-14，120 局 206 个样本）
ENEMY_LV_AHEAD, ENEMY_EXTRA = 1, 1     # 对手棋手等级领先进度表几级、对手场上除英雄外还有几个可被击败的单位
# 枫叶舞英雄每回合击杀数的均值（外生，不随推演里的等级变）。来源：惊鸿游龙逐档 3/10/21/36 杀给翩跹舞步，
# 200 局实测给牌回合中位 R3/R7/R11/R14 → 每回合 R1-3 约 1、R4-7 约 1.75、R8-11 约 2.75、R12 起约 5（与翩跹舞步·蓄势的回放残差一致）。
HOST_KILLS = ((3, 1.0), (7, 1.75), (11, 2.75), (99, 5.0))
HOST_KILL_WIN, HOST_KILL_LOSE = 1.5, 0.5   # 赢的回合多杀、输的少杀（均值按 BASE_WIN 归一后≈1）
HOST_KILL_SCALE, DEATH_SCALE = 1.0, 1.0    # 情景倍数（见 BATTLE_SCENARIOS）

# ---------------------------------------------------------------- 战斗结果情景
# 胜率 / 击杀 / 阵亡 / 普攻次数 / 放技能次数都是**战斗结果**：取决于阵容强弱和对手，推演器不估。
# 上面那些"实测"值来自我们自己模型打的局（2026-10-14 复核：现场数据也不准），只能当"中"档。
# 凡是比较方案（阵容、路线），都要在低 / 中 / 高三档下各推一遍，只信三档都成立的结论（tools/sim/comp_search.py --scen）。
BATTLE_SCENARIOS = {
    "低": {"BASE_WIN": 0.30, "HOST_KILL_SCALE": 0.5, "DEATH_SCALE": 1.08, "ATTACK_SECONDS": 2.0, "CAST_SECONDS": 8.0},
    "中": {"BASE_WIN": 0.49, "HOST_KILL_SCALE": 1.0, "DEATH_SCALE": 1.0, "ATTACK_SECONDS": 4.0, "CAST_SECONDS": 14.0},
    "高": {"BASE_WIN": 0.70, "HOST_KILL_SCALE": 2.0, "DEATH_SCALE": 0.75, "ATTACK_SECONDS": 12.0, "CAST_SECONDS": 20.0},
}


def set_battle_scenario(name: str) -> None:
    """把战斗结果类常量切到某一档（改的是本模块的全局量，对之后新建的 Engine 都生效）。"""
    for k, v in BATTLE_SCENARIOS[name].items():
        globals()[k] = v

BATTLE_SECONDS = 20.0     # 一场战斗约 20 秒（实测口径；普攻次数 / 放技能次数 / 每 N 秒类效果都用它）
MANA_PER_ATTACK = 10      # 普攻一次回 10 蓝
ATTACK_SPEED_DEFAULT = 7000   # 卡库没有攻速面板时按 0.7
LORD_HP_MAX = 60          # 棋手生命值上限（开局 60）
CORE_SHARE_CAP = 0.5      # 阿科米亚核心：单个英雄最多分到核心等级的 50%（data/keywords.json）
CORE_KW = "参与阿科米亚核心"   # 带这个关键词的单位（阿科米亚之子）不是日落海也参与核心分配
def host_kill_mean(round_no: int) -> float:
    for r, m in HOST_KILLS:
        if round_no <= r:
            return m
    return HOST_KILLS[-1][1]
FLOOD_CASTS = {"百里守约": 2, "百里玄策": 2, "沈梦溪": 3}   # 枫之洪流 3 秒内能多放几次（2026-10-14 确认：守约长前摇 2 次、玄策约 2 次、沈梦溪 3 次）
FLOOD_CASTS_DEFAULT = 3                                       # 其他大部分 3 次
EQUIP_DROP_ROUNDS = (1, 4, 8, 12, 16)   # 这几个回合开头随机给 1 件基础装备（2026-10-14 确认：第 1 回合开局手里就有一件；账本实测第 3/4、7/8、11/12、16 回合交界各约 1 件）
SELL_CARD_GAIN = 1        # 卖手牌 +1 能量（实测：弈星手牌卖出 +1）


@lru_cache(maxsize=1)
def shop_stats() -> dict:
    """data/engine/shop_stats.json（tools/engine/build_shop_stats.py）：各棋手等级的商店品阶概率、非英雄格占比与出现频率、对手持有量级。"""
    try:
        return json.loads((ROOT / "data/engine/shop_stats.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def base_equips() -> tuple:
    """8 件基础装备（data/equip_recipes.json 的配方键：风暴巨剑 / 圣者法典 / 雷鸣刃 / 守护者之铠 / 力量腰带 / 破碎圣杯 / 速击之枪 / 大棒）。"""
    try:
        return tuple(json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8")).get("recipes") or {})
    except Exception:  # noqa: BLE001
        return ()


@lru_cache(maxsize=1)
def auction_stats() -> dict:
    """data/engine/auction_stats.json（tools/engine/build_auction_stats.py）：拍品被争抢的比例、成交价分布。"""
    try:
        return json.loads((ROOT / "data/engine/auction_stats.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def opp_take(tier: int, round_no: int) -> float:
    """对手平均持有某个英雄几张（只作量级：前期 1 阶约 5 张、中期 1~2；2~4 阶约 1；5 阶几乎 0）。"""
    d = (shop_stats().get("opp_take") or {}).get(str(tier)) or {}
    if not d:
        return 0.0
    rs = sorted(int(r) for r in d)
    r = min(rs, key=lambda x: abs(x - round_no))
    return float(d[str(r)])


@lru_cache(maxsize=None)
def _heroes_of(tier: int, fac: str = "") -> tuple:
    """某品阶（可限派系）的英雄牌名单（缓存：商店抽牌原来每次扫一遍全卡库，占推演 1/4 的时间）。"""
    return tuple(n for n, c in card_db().items() if c.get("category") == "英雄牌" and quality(n) == tier
                 and (not fac or faction(n) == fac))


@lru_cache(maxsize=1)
def card_db() -> dict:
    cs = json.loads((ROOT / "data/cards.json").read_text(encoding="utf-8"))
    cs = cs.get("cards", cs) if isinstance(cs, dict) else cs
    cs = cs if isinstance(cs, list) else list(cs.values())
    out = {}
    for x in cs:
        out.setdefault(x["name"], x)
    return out


FACTION_ALIAS = {"大河": "大河流域", "三分": "三分之地"}
FACTIONS = ("河洛", "逐鹿", "大河流域", "日落海", "三分之地")


def faction(name: str) -> str:
    f = str((card_db().get(name) or {}).get("faction") or "")
    return FACTION_ALIAS.get(f, f)


def quality(name: str) -> int:
    try:
        return int((card_db().get(name) or {}).get("quality") or 0)
    except (TypeError, ValueError):
        return 0


def category(name: str) -> str:
    return str((card_db().get(name) or {}).get("category") or "")


def card_text(name: str) -> str:
    c = card_db().get(name) or {}
    return (c.get("skill") or "") + (c.get("desc") or "")


_TOK = re.compile(r"[，,、\s]+")


def has_keyword(name: str, kw: str) -> bool:
    """这张卡**拥有**关键词 kw：卡面有"kw："这一条效果，或卡面第一行单独列出 kw（"闪现""复生""夺取，…"）。
    cards.json 的 keywords 列表里还有卡面只是"提到"的关键词（李信 / 大司命"拥有牺牲效果的英雄阵亡时"、
    张良 / 朵莉亚 / 大乔 / 小乔"触发 X 效果时"），那些不算拥有 —— 要按"提到"判断用 mentions_keyword。"""
    t = card_text(name)
    if (kw + "：") in t or (kw + ":") in t:
        return True
    first = ((card_db().get(name) or {}).get("skill") or "").split("\n")[0].strip()
    return kw in _TOK.split(first)


has_effect = has_keyword


def mentions_keyword(name: str, kw: str) -> bool:
    """卡面提到了关键词 kw（cards.json keywords 列表 / 卡面文字），含只是"提到"的反应器。"""
    c = card_db().get(name) or {}
    return any(k.get("name") == kw for k in c.get("keywords") or []) or kw in card_text(name)


@lru_cache(maxsize=1)
def equip_recipes() -> dict:
    """基础装备 → 四选一的 4 个成品（data/equip_recipes.json，按卡牌 id 升序）。"""
    try:
        return json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8")).get("recipes") or {}
    except (OSError, ValueError):
        return {}


def base_equips() -> tuple:
    return tuple(equip_recipes())


def is_unique(name: str) -> bool:
    return str((card_db().get(name) or {}).get("skill") or "").startswith("唯一")


def is_unremovable(name: str) -> bool:
    return "无法拆卸" in card_text(name)


def is_melee(name: str) -> bool:
    """近战 = 卡库 properties.attackDistance <= 1（读不到按远程）。"""
    p = (card_db().get(name) or {}).get("properties") or {}
    try:
        return int(p.get("attackDistance") or 99) <= 1
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def lord_boost_pool() -> tuple:
    """阿离枫叶舞强化牌池 = data/lord_cards.json 阿离 related 里去掉 翩跹舞步·* 秘技牌和"即刻起舞"本身。"""
    try:
        d = json.loads((ROOT / "data/lord_cards.json").read_text(encoding="utf-8"))
        rel = list((d.get("阿离") or {}).get("related") or [])
    except (OSError, ValueError):
        rel = []
    return tuple(x for x in rel if not x.startswith("翩跹舞步") and x != "即刻起舞")


# ---------------------------------------------------------------- 卡牌注册
REGISTRY: dict = {}


def card(*names: str):
    """@card("古币") 注册一张卡的实现（一个类，方法见本文件开头）。同一实现可挂多个卡名。"""
    def deco(cls):
        for n in names:
            REGISTRY[n] = cls()
        cls.names = names
        return cls
    return deco


def impl(name: str):
    if not REGISTRY:
        from . import cards  # noqa: F401  导入时注册
    return REGISTRY.get(name)


_SIG: dict = {}


def _call(f, *args, **kw):
    """调用钩子，只传它签名里有的关键字参数（有 **kw 的照单全收）。"""
    fn = getattr(f, "__func__", f)
    acc = _SIG.get(fn)
    if acc is None:
        try:
            ps = list(inspect.signature(fn).parameters.values())
            acc = True if any(p.kind == p.VAR_KEYWORD for p in ps) else frozenset(p.name for p in ps)
        except (TypeError, ValueError):
            acc = True
        _SIG[fn] = acc
    if acc is not True:
        kw = {k: v for k, v in kw.items() if k in acc}
    return f(*args, **kw)


class Engine:
    def __init__(self, st: State, rng: Optional[random.Random] = None):
        self.s = st
        self.rng = rng or random.Random(0)
        # 共同随机数（2026-10-15 推演参考核对：拿不拿某个天赋会改变之后从同一条随机流里抽的次数，
        # 商店 / 战斗全部错位，比的其实是运气 —— 4 次推演两两方向翻转 39%）：商店每一眼、每场战斗
        # 各用一条按 (种子, 类别, 回合, 第几次) 确定的随机流，不同选项在同一时刻看到同一个商店、同一场战斗
        self._seed = (rng or random.Random(0)).getrandbits(32) if rng is not None else 0
        self._streams: dict = {}
        self.depth = 0
        self.triggered = 0           # >0：当前的登场 / 整备是"被别的效果触发"的（不是使用英雄牌）
        self.chooser = None          # 推演打法：chooser(eng, kind, options, **ctx) -> 选中的一项；None = 第一项

    # ------------------------------------------------------------ 选择（N 选 1 / 兑换券 / 指定目标…交给推演打法）
    def choose(self, kind: str, options: list, **ctx):
        if not options:
            return None
        if self.chooser:
            return self.chooser(self, kind, options, **ctx)
        return options[0]

    # ------------------------------------------------------------ 抽牌（商店 / 发英雄牌）
    def pool_left(self, name: str) -> float:
        """牌库里这张英雄还剩几张 = 总张数 − 我方拿走 − 对手持有（按回合的实测量级，shop_stats.opp_take）；被禁派系 = 0。"""
        if self.s.banned and faction(name) == self.s.banned:
            return 0
        q = quality(name)
        return max(0.0, B.meta.POOL_COPIES.get(q, 6) - self.s.pool_taken.get(name, 0) - opp_take(q, self.s.round))

    def _stream(self, kind: str) -> random.Random:
        k = (kind, self.s.round)
        n = self._streams.get(k, 0)
        self._streams[k] = n + 1
        return random.Random(f"{self._seed}:{kind}:{self.s.round}:{n}")

    def shop_view(self) -> list:
        first = self._streams.get(("shop", self.s.round), 0) == 0          # 本回合第一眼 = 回合开始的自动刷新
        saved, self.rng = self.rng, self._stream("shop")
        try:
            out = self._shop_view()
            # 破格（天赋）：回合开始商店自动刷新时出现 1 个高 1 阶英雄牌（2026-10-15：原来只记了 flag、商店从不读，体检里和"不拿"完全一样）
            if first and self.s.flags.get("破格") and not self.s.shop_locked:
                tier = min(5, max(1, min(5, self.s.lord_level)) + 1)
                h = self.draw_hero(tier)
                if h:
                    i = self.rng.randrange(len(out))
                    out[i] = h
                    self.note("破格", hero=h, tier=tier)
            return out
        finally:
            self.rng = saved

    def _shop_view(self) -> list:
        """看一次商店（4 格）：每格先按实测比例决定是不是非英雄牌（效果 / 天赋 / 装备，按棋手等级的出现频率抽），
        否则按实测品阶概率 + 牌库剩余抽英雄。商店内容是游戏的随机，买不买由推演打法定。"""
        s = self.s
        if s.flags.get("shop_fresh") and s.shop:          # 准备阶段第一眼 = 真实商店（从真实局面开推时读进来的）
            s.flags.pop("shop_fresh", None)
            out = list(s.shop)[:B.meta.SHOP_SLOTS]
            return out + [None] * (B.meta.SHOP_SLOTS - len(out))
        st = shop_stats()
        lv = str(self.s.lord_level)
        rate = float((st.get("nonhero") or {}).get(lv) or 0.0)
        cards = (st.get("cards") or {}).get(lv) or {}
        out = []
        for _ in range(B.meta.SHOP_SLOTS):
            if cards and self.rng.random() < rate:
                out.append(self.rng.choices(list(cards), list(cards.values()))[0])
            else:
                out.append(self.draw_hero())
        return out

    def game_start(self) -> None:
        """开局（第 1 回合准备阶段之前）：手牌里随机 1 件基础装备（2026-10-14）。只做一次。"""
        s = self.s
        if s.flags.get("开局"):
            return
        s.flags["开局"] = True
        if s.round == 1 and 1 in EQUIP_DROP_ROUNDS and base_equips():
            self.gain_card(self.rng.choice(base_equips()))
            self.note("回合投放装备", round=1)

    def auction(self, pick: str, lots: list) -> Optional[str]:
        """拍卖结算（官方规则 + 实测）：起拍 1、每次 +1、每人只拍 1 张，拍不到不扣钱；拍品数 = 剩余人数，每人都会分到一张。
        想要的拍品按实测比例被人争抢（auction_stats.contested），争抢时成交价按实测分布抽，否则 1 能量拿下；
        钱不够 → 拍不到，随机分到一张剩下的拍品（不花钱）。有效出价进分红池，平均分给在场的人
        （官方：血越少分得越多 —— 推演不估血量，按平均）。2026-10-14 确认：拍卖等待期间没有掉落（就是等别人打完）。
        返回拿到的拍品。"""
        s = self.s
        st = auction_stats()
        q = float(st.get("contested") or 0.38)
        prices = {int(k): v for k, v in (st.get("price") or {"3": 1}).items()}
        contested_mean = sum(k * v for k, v in prices.items()) / max(1, sum(prices.values()))
        price = self.rng.choices(list(prices), list(prices.values()))[0] if self.rng.random() < q else 1
        got = None
        if pick and s.energy - price >= s.energy_floor and self.spend(price, src="拍卖"):
            got = pick
        else:
            # 没拍到想要的 → 强制以 1 能量分到一张没人出价的拍品（2026-10-14 确认：有几个人就几个拍品，每人都会分到一个）
            rest = [x for x in lots if x and x != pick]
            got = self.rng.choice(rest) if rest else None
            price = 1 if (got and self.spend(1, src="拍卖")) else 0
        n = max(1, len(lots))
        pool = price + (n - 1) * q * contested_mean            # 自己的成交价 + 别人的期望成交价
        s.energy += int(round(pool / n))
        self.note("拍卖", pick=pick, got=got, price=price, dividend=int(round(pool / n)))
        return got

    def lock_shop(self, view: list) -> None:
        """锁定商店：这一眼的商店（买剩下的）留到下回合（实测：可以把想要但买不起的牌留到下一回合）。"""
        self.s.shop = [x for x in view if x]
        self.s.shop_locked = True
        self.s.flags["shop_fresh"] = True
        self.note("锁定商店", shop=list(self.s.shop))

    def shop_cost(self, name: str) -> int:
        if category(name) == "英雄牌":
            return B.HERO_COST + self.s.shop_extra_cost
        c = (shop_stats().get("cost") or {}).get(name)
        if c is None:
            c = (card_db().get(name) or {}).get("cost") or 0
        return int(c) + self.s.shop_extra_cost

    def draw_hero(self, tier: int | None = None, fac: str | None = None) -> Optional[str]:
        """按牌库剩余张数抽一张英雄（tier=None：按棋手等级的商店品阶概率先抽品阶）。"""
        if tier is None:
            odds = {int(k): v for k, v in ((shop_stats().get("tier_odds") or {}).get(str(self.s.lord_level)) or {}).items()}                 or B.meta.TIER_ODDS.get(self.s.lord_level) or {1: 1.0}
            ts, ws = list(odds), list(odds.values())
            tier = self.rng.choices(ts, ws)[0]
        names = _heroes_of(tier, FACTION_ALIAS.get(fac, fac) if fac else "")
        ws = [self.pool_left(n) for n in names]
        if not names or sum(ws) <= 0:
            return None
        return self.rng.choices(names, ws)[0]

    def take(self, name: str) -> None:
        self.s.pool_taken[name] = self.s.pool_taken.get(name, 0) + 1

    def give_back(self, name: str, n: int = 1) -> None:
        """英雄牌回公共牌库（2026-10-14 确认的规则：卖掉场上或手牌的英雄，牌回公共牌库；
        没觉醒时场上用了几张同名就占几张；觉醒后固定占 4 张，再合成的同名直接回牌库）。"""
        if category(name) != "英雄牌" or n <= 0:
            return
        self.s.pool_taken[name] = max(0, self.s.pool_taken.get(name, 0) - n)

    # ------------------------------------------------------------ 日志
    def note(self, ev: str, **kw) -> None:
        self.s.log.append((self.s.round, ev, kw))

    # ------------------------------------------------------------ 查询
    def heroes(self) -> list:
        return self.s.heroes()

    def board_units(self) -> list:
        return self.s.on_board()

    def faction_of(self, u: Unit) -> str:
        return faction(u.name)

    def is_faction(self, u: Unit, fac: str) -> bool:
        fac = FACTION_ALIAS.get(fac, fac)
        return faction(u.name) == fac or fac in u.keywords

    def has_kw(self, u: Unit, kw: str) -> bool:
        """单位拥有关键词：卡面拥有（has_keyword，只"提到"的不算）或被别的效果赋予（u.keywords）。"""
        return kw in u.keywords or has_keyword(u.name, kw)

    def unit_has_effect(self, u: Unit, kw: str) -> bool:
        """单位本体或身上装备有"kw："效果。"""
        return has_keyword(u.name, kw) or any(has_keyword(e, kw) for e in u.equips)

    def dist(self, a: Unit, b: Unit) -> int:
        if not a.cell or not b.cell:
            return 99
        return B.hex_dist(a.cell, b.cell)

    def nearest(self, u: Unit, pred: Callable[[Unit], bool] = lambda x: True, k: int = 1, include_self=False) -> list:
        """距离最近的 k 名（格子步数；并列随机）。"""
        cands = [x for x in self.heroes() if (include_self or x is not u) and pred(x)]
        self.rng.shuffle(cands)
        cands.sort(key=lambda x: self.dist(u, x))
        return cands[:k]

    def pick(self, items: list, k: int = 1) -> list:
        items = list(items)
        self.rng.shuffle(items)
        return items[:k]

    def lowest(self, items: list, k: int = 1) -> list:
        items = list(items)
        self.rng.shuffle(items)
        return sorted(items, key=lambda x: x.level)[:k]

    def highest(self, items: list, k: int = 1) -> list:
        items = list(items)
        self.rng.shuffle(items)
        return sorted(items, key=lambda x: -x.level)[:k]

    def unit_by_uid(self, uid) -> Optional[Unit]:
        return next((x for x in self.s.units if x.uid == uid), None)

    # ------------------------------------------------------------ 战斗估计（引擎不模拟战斗过程）
    def _poisson(self, lam: float) -> int:
        import math as _m
        L, k, p = _m.exp(-lam), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= L:
                return k
            k += 1

    def attacks(self, u: Unit) -> float:
        """一场战斗里该单位的普攻次数估计 = 攻速 × BATTLE_SECONDS（攻速取卡库面板，没有按 0.7）。"""
        aps = ((card_db().get(u.name) or {}).get("properties") or {}).get("attackSpeed") or ATTACK_SPEED_DEFAULT
        return aps / 10000.0 * ATTACK_SECONDS

    def casts(self, u: Unit) -> int:
        """一场战斗里该单位放几次技能。优先读 u.counters["casts"]（推演打法可以填）；
        否则按卡库面板估：(初始法力 + 普攻次数 × 10) // 法力上限。待核对：没算被打回蓝、法力上限降低、阵亡时间。"""
        if "casts" in u.counters:
            return int(u.counters["casts"])
        p = (card_db().get(u.name) or {}).get("properties") or {}
        mana = p.get("energy") or 0
        if mana <= 0:
            return 0
        attacks = (p.get("attackSpeed") or 0) / 10000.0 * CAST_SECONDS
        n = int(((p.get("initEnergy") or 0) + MANA_PER_ATTACK * attacks) // mana)
        if u.name == self.s.maple and "枫之洪流" in self.s.lord_cards:
            n += FLOOD_CASTS.get(u.name, FLOOD_CASTS_DEFAULT)   # 枫之洪流：首次转移后 3 秒无限施法，受前后摇限制只能多放 2~3 次
        return n

    def survivors(self) -> list:
        """最近一场战斗结束时还活着的英雄（复生后活下来的算活着）。"""
        dead = set(self.s.battle_dead)
        return [h for h in self.heroes() if h.uid not in dead]

    # ------------------------------------------------------------ 持有者（谁身上挂着哪张卡）
    def holders(self):
        """(卡名, 实现, 挂靠单位或 None)：场上单位本体、单位身上装备、天赋、棋手牌、
        手牌里 in_hand=True 的牌（每个名字一次）、本回合打出过的整回合生效牌（round_cards，重复 = 叠加）。"""
        s = self.s
        for u in list(s.units):
            c = impl(u.name)
            if c:
                yield u.name, c, u
            for e in list(u.equips):
                ce = impl(e)
                if ce:
                    yield e, ce, u
        for t in list(s.talents):
            c = impl(t)
            if c:
                yield t, c, None
        for t in list(s.lord_cards):
            c = impl(t)
            if c:
                yield t, c, None
        seen = set()
        for n in list(s.hand):
            if n in seen:
                continue
            seen.add(n)
            c = impl(n)
            if c is not None and getattr(c, "in_hand", False):
                yield n, c, None
        for n in list(s.round_cards):
            c = impl(n)
            if c is not None:
                yield n, c, None

    def emit(self, ev: str, **kw) -> None:
        """反应器："每当 …时"。"""
        if self.depth > 6:
            return
        self.depth += 1
        try:
            for name, c, holder in list(self.holders()):
                f = getattr(c, "react_" + ev, None)
                if f:
                    _call(f, self, holder, **kw)
        finally:
            self.depth -= 1

    def fire(self, u: Unit, timing: str, triggered: bool = False, **kw) -> None:
        """单位自身的时机（整备 / 登场 / 开团 …）：执行它卡面的钩子（觉醒翻倍 = 执行两次），再发反应器事件。
        triggered=True：这是被别的效果"触发其登场 / 整备效果"（曹操、唤醒、临阵磨枪…），不是使用英雄牌 ——
        "每当你使用 / 打出英雄牌时"的卡用 eng.is_card_use() 区分。"""
        if self.depth > 6 or u not in self.s.units:
            return
        if triggered:
            self.triggered += 1
        try:
            self.depth += 1
            c = impl(u.name)
            execs = self.exec_times(u)
            try:
                f = getattr(c, "on_" + timing, None) if c else None
                if f:
                    for _ in range(2 if u.awakened and not getattr(c, "awaken_custom", False) else 1):
                        _call(f, self, u, **kw)
                for e in list(u.equips):
                    ce = impl(e)
                    g = getattr(ce, "on_" + timing, None) if ce else None
                    if g:
                        _call(g, self, u, **kw)
            finally:
                self.depth -= 1
            # execs：这次触发算几次（10-19 规则确认：觉醒英雄执行两次 = 计数 2 次；小乔 / 张良 / 朵莉亚 / 韩信 这类"每触发 1 次"的反应器按它计）
            self.emit(timing, unit=u, triggered=self.triggered > 0, execs=execs, **kw)
        finally:
            if triggered:
                self.triggered -= 1

    @staticmethod
    def exec_times(u: Unit) -> int:
        """单位自身时机这一次"算触发几次"：觉醒单位 → 2，否则 1。
        依据：2026-10-19 规则确认"觉醒英雄执行两次 = 计数 2 次"（不管卡类是引擎执行两次还是 awaken_custom 自己按卡面数值处理）。"""
        return 2 if u.awakened else 1

    def trigger(self, u: Unit, timing: str, **kw) -> None:
        """"触发其登场 / 整备效果"：fire(u, timing, triggered=True)。"""
        self.fire(u, timing, triggered=True, **kw)

    def is_card_use(self) -> bool:
        """当前这次登场是"使用了英雄牌"（上场 / 合成 / 觉醒回手再打出），不是被别的效果触发的。"""
        return self.triggered == 0

    # ------------------------------------------------------------ 养成操作
    def level(self, u: Unit, n: float, src: str = "") -> None:
        if u is None or n == 0:
            return
        u.level = max(1.0, u.level + n)
        self.note("等级", unit=u.name, n=n, src=src)

    def temp(self, u: Unit, n: float, src: str = "") -> None:
        """临时等级（战斗结束清零）。发 temp 事件（司空震"其他英雄获得临时等级时"）。"""
        if u is None or n == 0:
            return
        u.temp += n
        self.note("临时等级", unit=u.name, n=n, src=src)
        self.emit("temp", unit=u, n=n, src=src)

    def coin(self, target: Unit, n: int = 1, src: str = "", fine: bool = False, active: bool = False) -> None:
        """对 target 使用 n 张古币（卡面：指定【河洛】英雄等级 +2；武则天等的加成走 coin_bonus）。
        active=True：从手牌直接打出的古币（传家宝 / 幸运硬币只认这种）；李元芳 等被动使用为 False。"""
        for _ in range(n):
            if target is None or target not in self.s.units:
                return
            bonus = 0
            for name, c, holder in list(self.holders()):
                f = getattr(c, "coin_bonus", None)
                if f:
                    bonus += _call(f, self, holder, target, fine, active=active) or 0
            self.level(target, COIN_LEVELS * (2 if fine else 1) + bonus, src=src or "古币")
            self.emit("coin", target=target, fine=fine, active=active)

    def use_card(self, name: str, target: Optional[Unit] = None) -> bool:
        """打出一张效果牌 / 天赋牌（拿到即用的），结算它的 on_use，再发"使用效果牌"事件。"""
        c = impl(name)
        if not c or not hasattr(c, "on_use"):
            self.note("未实现", card=name)
            return False
        ok = c.on_use(self, target)
        if ok is False:
            return False
        if category(name) == "效果牌":
            self.emit("effect_card", card=name, target=target)
        return True

    # ------------------------------------------------------------ 手牌
    def _raw_gain(self, name: str, meta: dict) -> bool:
        s = self.s
        # 手牌可以超过上限（2026-10-14 确认：能买牌，超过的部分下个回合自动销毁）—— 超出的在 round_start 销毁
        self._meta_sync(name)
        s.hand.append(name)
        s.hand_meta.setdefault(name, []).append(dict({"got": s.round}, **meta))
        self.note("获得", card=name)
        return True

    def gain_card(self, name: str, **meta) -> None:
        """获得一张牌（进手牌；可以超过 10 张，超出的下回合开始销毁）。meta 记进这张牌的手牌状态（transient=True 等）。
        持有牌的 gain_redirect(eng, holder, name) 返回列表 = 改成获得这些牌（零钱兑换、演武）。发 gain 事件。"""
        if not name:
            return
        for hn, c, holder in list(self.holders()):
            f = getattr(c, "gain_redirect", None)
            if f:
                rep = f(self, holder, name)
                if rep is not None:
                    self.note("兑换", card=name, to=list(rep), src=hn)
                    for x in rep:
                        if self._raw_gain(x, meta):
                            self.emit("gain", card=x)
                    return
        if self._raw_gain(name, meta):
            self.emit("gain", card=name)

    def _meta_sync(self, name: str) -> list:
        """手牌状态和手牌对齐：多出来的（牌被直接从 hand 拿掉了）丢最早的，少的（直接塞进 hand 的）补空状态。"""
        s = self.s
        n = s.hand.count(name)
        lst = s.hand_meta.setdefault(name, [])
        if len(lst) > n:
            del lst[:len(lst) - n]
        while len(lst) < n:
            lst.insert(0, {})
        if not lst:
            s.hand_meta.pop(name, None)
        return lst

    def hand_meta(self, name: str) -> list:
        """手里这张牌（同名多张按获得先后）的状态列表。"""
        return list(self._meta_sync(name))

    def take_meta(self, name: str) -> dict:
        """取走手里最早获得的那张同名牌的状态（打出这张牌时调；牌本身由打法从 hand 拿掉）。"""
        lst = self._meta_sync(name)
        return lst.pop(0) if lst else {}

    def hand_remove(self, name: str) -> Optional[dict]:
        """从手牌拿掉一张（最早获得的那张），返回它的状态；手里没有返回 None。"""
        if name not in self.s.hand:
            return None
        m = self.take_meta(name)
        self.s.hand.remove(name)
        self._meta_sync(name)
        return m

    def sell_card(self, name: str) -> bool:
        """出售一张手牌：+1 能量（SELL_CARD_GAIN），发 sell_card 事件（丝绸之路"出售古币"、大买卖"出售英雄牌"）。"""
        if self.hand_remove(name) is None:
            return False
        self.give_back(name)
        self.s.energy += SELL_CARD_GAIN
        self.note("出售手牌", card=name)
        self.emit("sell_card", card=name)
        return True

    def attach_round(self, name: str) -> None:
        """打出后本回合整回合生效的牌（锦囊…）：进 round_cards，回合开始清。"""
        self.s.round_cards.append(name)

    # ------------------------------------------------------------ 能量 / 商店
    def spend(self, n: int, src: str = "") -> bool:
        """花能量（不能低于能量下限 energy_floor）。发 spend 事件（消费返利）。"""
        if n <= 0:
            return True
        if self.s.energy - n < self.s.energy_floor:
            return False
        self.s.energy -= n
        self.emit("spend", n=n, src=src)
        return True

    def can_play_hero(self) -> bool:
        """透支：回合开始能量为负 → 本回合不能打出英雄牌（flags["round:no_hero_card"]）。"""
        return not self.s.flags.get("round:no_hero_card")

    def buy(self, name: str, cost: Optional[int] = None) -> bool:
        """商店购买：扣费（英雄牌 HERO_COST，+ shop_extra_cost）→ 牌库记拿走 → 进手牌，发 buy 事件。"""
        s = self.s
        if not name:
            return False
        if cost is None:
            cost = self.shop_cost(name)                 # 英雄 3；非英雄按商店实测价格（含 shop_extra_cost）
        else:
            cost += s.shop_extra_cost
        if not self.spend(cost, src="购买"):
            return False
        if category(name) == "英雄牌":
            self.take(name)
        self.gain_card(name)
        self.note("购买", card=name, cost=cost)
        self.emit("buy", card=name, cost=cost)
        return True

    def refresh_shop(self) -> bool:
        """手动刷新商店：本回合免刷新费（展翅腾飞）> 免费刷新次数 > 花 REFRESH_COST。发 refresh 事件
        （节能商店；手牌里的神秘英雄牌在这时变身）。商店内容本身由推演打法抽。"""
        s = self.s
        if s.free_refresh_all:
            pass
        elif s.free_refresh > 0:
            s.free_refresh -= 1
        elif not self.spend(B.REFRESH_COST, src="刷新"):
            return False
        self.note("刷新")
        self.emit("refresh")
        return True

    # ------------------------------------------------------------ 棋手生命 / 淘汰
    def add_hp(self, delta: int = 0, set_to: Optional[int] = None, src: str = "") -> int:
        """改棋手生命值（上限 LORD_HP_MAX）。降到 0 时若还有 hp_guard（一线生机）→ 保留 1 点。"""
        s = self.s
        new = int(set_to) if set_to is not None else int(s.hp) + int(delta)
        new = min(new, LORD_HP_MAX)
        if new <= 0 and s.hp_guard > 0:
            new = 1
            s.hp_guard -= 1
        s.hp = max(0, new)
        self.note("棋手生命", hp=s.hp, src=src)
        return s.hp

    def eliminate(self, n: int = 1) -> None:
        """其他棋手被淘汰 n 位：发 eliminated 事件（拾荒者、资源再分配）。"""
        if n <= 0:
            return
        self.s.eliminated += n
        self.note("棋手淘汰", n=n)
        self.emit("eliminated", n=n)

    # ------------------------------------------------------------ 阿科米亚核心
    def core_add(self, n: float, src: str = "", event: bool = True) -> None:
        """核心等级 +n（n 可为负：盗火 / 跳级 / 能源置换；最低 0）。提升（n>0）时发 core_up 事件（海诺）；
        event=False 用于"额外 +1"这类不算新一次提升的加成。"""
        if not n:
            return
        self.s.core = max(0.0, self.s.core + n)
        self.note("核心等级", n=n, src=src)
        if n > 0 and event:
            self.emit("core_up", n=n, src=src)

    def core_eligible(self, u: Unit) -> bool:
        """参与核心分配：【日落海】英雄，或带"参与阿科米亚核心"关键词的（阿科米亚之子）；
        身上有 blocks_core 装备（星源星徽）的不分。"""
        if u.kind != "hero" or u not in self.s.units:
            return False
        if not (self.is_faction(u, "日落海") or CORE_KW in u.keywords):
            return False
        return not any(getattr(impl(e), "blocks_core", False) for e in u.equips)

    def core_distribute(self) -> None:
        """关键词「阿科米亚核心」：战斗开始时，核心等级以临时等级平均分给参与分配的英雄，单人上限为核心等级的 50%。
        持有牌带 core_no_share（能量改造 / 光棱塔）→ 不分配；带 core_direct_cap（定向传输：200）→
        先给穿装备最多的那名（不受 50% 上限，至多 cap 级），剩下的再平均分给其他人。每份向下取整。"""
        s = self.s
        core = s.core
        if core <= 0:
            return
        direct = None
        for name, c, holder in list(self.holders()):
            if getattr(c, "core_no_share", False):
                return
            if getattr(c, "core_direct_cap", None):
                direct = max(direct or 0, c.core_direct_cap)
        sun = [u for u in s.on_board() if self.core_eligible(u)]
        if not sun:
            return
        rest = core
        if direct:
            m = max(len(u.equips) for u in sun)
            t = self.pick([u for u in sun if len(u.equips) == m], 1)[0]
            a = math.floor(min(core, direct))
            self.temp(t, a, src="阿科米亚核心(定向传输)")
            rest -= a
            sun = [u for u in sun if u is not t]
        if not sun or rest <= 0:
            return
        each = math.floor(min(rest / len(sun), core * CORE_SHARE_CAP))
        for u in sun:
            self.temp(u, each, src="阿科米亚核心")

    # ------------------------------------------------------------ 觉醒 / 夺取 / 棋手强化
    def awaken_step(self, u: Unit, n: int = 1, src: str = "") -> bool:
        """非合成途径的觉醒进度 +n（弦音 / 妙音 / 冰封强化 / 空降人才…；封顶 3）。凑满 3 = 觉醒：
        和合成觉醒一样按"回手再打出 = 再一次登场（算使用英雄牌）"处理。已觉醒 / 不在场返回 False。"""
        if u is None or u not in self.s.units or u.awakened or n <= 0:
            return False
        u.awaken = min(3, u.awaken + n)
        self.note("觉醒进度", unit=u.name, n=n, src=src)
        if u.awakened:
            self.note("觉醒", unit=u.name, src=src)
            self.fire(u, "deploy")
        return True

    def steal(self, u: Unit, src: str = "夺取") -> float:
        """关键词「夺取」：使用此卡牌（上场 / 合成）时，夺取周围 1 格随机 1 名己方英雄的等级（最多 10 级，不含临时等级）。"""
        nb = [x for x in self.heroes() if x is not u and self.dist(u, x) <= 1]
        if not nb:
            return 0.0
        v = self.pick(nb, 1)[0]
        a = min(10.0, max(0.0, v.level - 1))
        if a <= 0:
            return 0.0
        v.level -= a
        self.level(u, a, src=src)
        return a

    def add_lord_card(self, name: str) -> None:
        if name not in self.s.lord_cards:
            self.s.lord_cards.append(name)

    def pick_lord_boost(self, k: int = 3, kind: str = "lord_boost") -> Optional[str]:
        """为枫叶舞挑选 1 张强化牌：从没拿过的强化里随机 k 张，交给推演打法选，放进 lord_cards。"""
        pool = [x for x in lord_boost_pool() if x not in self.s.lord_cards]
        sel = self.choose(kind, self.pick(pool, k))
        if sel and sel not in self.s.lord_cards:
            self.s.lord_cards.append(sel)
            self.note("枫叶舞强化", card=sel)
        return sel

    # ------------------------------------------------------------ 装备
    def can_wear(self, u: Unit, name: str) -> bool:
        """穿戴限制（读卡面）：限定远程 / 近战英雄、仅三分英雄、仅 1-3 阶、可穿戴给【X】英雄；
        装备类自己的 can_equip(eng, u) / wearable(eng, u) / wear_faction 也认。
        【日落海】专属装备不限制穿戴（全域共鸣"专属效果对穿戴它的任意英雄生效" → 非日落海也能穿，只是专属效果不生效）。"""
        if u is None or u.kind not in ("hero", "summon"):
            return False
        if u.kind == "summon" and "图腾" in u.name:      # 10-19 规则确认：召唤物能穿装备，但图腾不能穿装备
            return False
        t = card_text(name)
        if "限定远程英雄" in t and is_melee(u.name):
            return False
        if "限定近战英雄" in t and not is_melee(u.name):
            return False
        if ("仅三分英雄" in t or "仅【三分】英雄" in t) and not self.is_faction(u, "三分之地"):
            return False
        if "仅1-3阶英雄" in t and not 1 <= quality(u.name) <= 3:
            return False
        m = re.search(r"可穿戴给【([^】]+)】英雄", t)
        if m and not self.is_faction(u, m.group(1)):
            return False
        c = impl(name)
        if c is not None:
            wf = getattr(c, "wear_faction", None)
            if wf and not self.is_faction(u, wf):
                return False
            for meth in ("can_equip", "wearable"):
                f = getattr(c, meth, None)
                if f and not f(self, u):
                    return False
        return True

    def exclusive_active(self, u: Unit, name: str) -> bool:
        """"【X】专属装备"的专属效果对这名穿戴者生效吗：是 X 派系，或有全域共鸣（日落海）。"""
        m = re.match(r"【([^】]+)】专属装备", card_text(name))
        if not m:
            return True
        fac = FACTION_ALIAS.get(m.group(1), m.group(1))
        return self.is_faction(u, fac) or (fac == "日落海" and ("全域共鸣" in self.s.talents or self.s.flags.get("全域共鸣")))

    def equip(self, u: Unit, name: str, bound: bool = False, forge: bool = True, force: bool = False,
              src: str = "") -> bool:
        """穿装备：3 件上限、"唯一"同名一人一件、穿戴限制（can_wear）；基础装备（equip_recipes.json）
        当场四选一铸造成品（eng.choose("forge", 成品…)，选项排除已穿的唯一成品、穿不了的成品）；
        bound=True 或卡面"无法拆卸" → 进 u.bound（离场不退回手牌）；最后调 on_equip(eng, u)，发 equip 事件。
        force=True：出生自带（干将的双剑），不查件数和限制。成功返回 True。"""
        if u is None or u not in self.s.units or not name:
            return False
        if not force:
            if len(u.equips) >= B.MAX_EQUIPS or not self.can_wear(u, name):
                return False
            if is_unique(name) and name in u.equips:
                return False
        rec = equip_recipes()
        if forge and name in rec:
            opts = [x for x in rec[name] if not (is_unique(x) and x in u.equips) and self.can_wear(u, x)]
            pick = self.choose("forge", opts, unit=u, base=name)
            if not pick:
                return False
            self.note("铸造", unit=u.name, base=name, equip=pick)
            name = pick
        u.equips.append(name)
        if bound or is_unremovable(name):
            u.bound.append(name)
        self.note("穿戴", unit=u.name, equip=name, src=src)
        c = impl(name)
        f = getattr(c, "on_equip", None) if c else None
        if f:
            f(self, u)
        self.emit("equip", unit=u, card=name)
        return True

    # ------------------------------------------------------------ 上场 / 合成 / 离场
    def free_cells(self) -> list:
        used = {u.cell for u in self.s.units if u.cell}
        return [c for c in B.CELLS if c not in used]

    def deploy(self, name: str, cell=None, level: float = 1.0, kind: str = "hero", pop: bool = True) -> Optional[Unit]:
        free = self.free_cells()
        if not free:
            return None
        if cell not in free:
            cell = free[0]
        u = Unit(name=name, level=level, cell=cell, kind=kind, pop=pop, uid=self.s.next_uid)
        self.s.next_uid += 1
        self.s.units.append(u)
        self.note("上场", unit=name, cell=cell)
        if kind == "hero":
            self.fire(u, "deploy")
        return u

    def summon(self, name: str, near: Optional[Unit] = None, level: float = 1.0, battle_only: bool = False,
               cell=None) -> Optional[Unit]:
        """召唤（不占人口、不触发登场）。battle_only=True：只在本场战斗存在，战斗结束由引擎移除。
        发 summon 事件（往生图腾吸收、反应器）—— 返回的单位可能已被吸收（不在场上）。"""
        if cell is None and near is not None and near.cell:
            free = self.free_cells()
            cell = min(free, key=lambda c: B.hex_dist(near.cell, c)) if free else None
        u = self.deploy(name, cell=cell, level=level, kind="summon", pop=False)
        if u is not None:
            if battle_only:
                u.counters["battle_only"] = True
            self.emit("summon", unit=u)
        return u

    def merge(self, u: Unit) -> None:
        """合成同名：+3 级、觉醒进度 +1；使用英雄牌 = 一次登场。凑满 3 张同名 → 觉醒（卡面效果翻倍，回手再打出 = 再一次登场）。"""
        was = u.awakened
        if was:
            self.give_back(u.name)                          # 觉醒后固定占 4 张，再合成的这张直接回牌库
        self.level(u, MERGE_GAIN, src="合成")
        u.awaken += 1
        self.fire(u, "merge")
        self.fire(u, "deploy")
        if u.awakened and not was:
            self.note("觉醒", unit=u.name)
            self.fire(u, "deploy")
        self.s.merges += 1
        if self.s.merges % MERGES_PER_JINNANG == 0:
            tier = min(self.s.lord_level, 6)               # 棋手 6 级时送 6 阶锦囊（众星拱月等 6 阶效果牌，2026-10-14）
            self.gain_card(f"{tier}阶效果锦囊")

    def remove(self, u: Unit, src: str = "") -> None:
        """离场（被卖 / 被移除）。枫叶舞宿主离场：红叶染锋记下 50% 真实等级（限一次）；可转移装备回手牌。"""
        if u not in self.s.units:
            return
        if self.s.maple == u.name:
            if "红叶染锋" in self.s.lord_cards and not self.s.flags.get("红叶染锋_used"):
                self.s.maple_keep = math.floor(u.level * 0.5)
                self.s.flags["红叶染锋_used"] = True
            self.s.maple = ""
        self.s.units.remove(u)
        if u.kind == "hero" and u.pop:                     # 离场：占着的牌回牌库（觉醒 = 4 张；没觉醒 = 本体 + 合成过的张数）
            self.give_back(u.name, 4 if u.awakened else 1 + u.awaken)
        for e in u.equips:
            if e not in u.bound:
                self.gain_card(e)
        self.note("离场", unit=u.name, src=src)

    def sell(self, u: Unit) -> None:
        self.fire(u, "sell")
        self.remove(u, src="出售")
        self.s.energy += 1

    def bind_maple(self, u: Unit) -> None:
        """枫叶舞绑定（换绑要先让旧宿主离场，由推演打法处理）。红叶染锋留下的等级给新宿主。"""
        self.s.maple = u.name
        if self.s.maple_keep:
            self.level(u, self.s.maple_keep, src="红叶染锋")
            self.s.maple_keep = 0.0

    def rename(self, u: Unit, new: str, src: str = "") -> None:
        """单位变身（化铁为金 / 多彩古币 / 冰之心解冻…）：改名，主C 和枫叶舞记的名字一起改。"""
        old = u.name
        u.name = new
        if self.s.carry == old:
            self.s.carry = new
        if self.s.maple == old:
            self.s.maple = new
        self.note("变身", unit=old, to=new, src=src)

    # ------------------------------------------------------------ 回合
    def round_start(self) -> None:
        """回合开始：清本回合状态 → 赋予关键词 → 天赋 / 装备 / 棋手牌 / 手牌的回合开始效果 → 整备（场上单位）。"""
        s = self.s
        s.flags = {k: v for k, v in s.flags.items() if not k.startswith("round:")}
        if s.round > 1 and s.round in EQUIP_DROP_ROUNDS and base_equips():   # 第 1 回合的那件在 game_start 里给
            self.gain_card(self.rng.choice(base_equips()))
            self.note("回合投放装备", round=s.round)
        if len(s.hand) > B.HAND_LIMIT:                    # 手牌超上限的部分在新回合开始销毁（后拿到的先销毁）
            for name in s.hand[B.HAND_LIMIT:]:
                self.note("手牌超上限销毁", card=name)
            del s.hand[B.HAND_LIMIT:]
            for name in list(s.hand_meta):
                self._meta_sync(name)
        if not s.shop_locked:                             # 新回合商店自动刷新（锁了就留着，锁只管一个回合）：真实商店只在读进来的那个回合有效
            s.flags.pop("shop_fresh", None)
        s.shop_locked = False
        s.free_refresh_all = False
        s.round_cards = []
        for u in list(s.units):
            u.keywords = set()
        for name, c, holder in list(self.holders()):
            f = getattr(c, "grant_keywords", None)        # 钟馗"自身及左右 1 格获得牺牲"这类
            if f:
                f(self, holder)
        for name, c, holder in list(self.holders()):
            f = getattr(c, "on_round_start", None)
            if f:
                f(self, holder)
        for u in list(s.units):
            self.fire(u, "prepare")

    def battle(self) -> bool:
        saved, self.rng = self.rng, self._stream("battle")
        try:
            return self._battle()
        finally:
            self.rng = saved

    def _battle(self) -> bool:
        """战斗：只产生养成相关的事件，胜负与击杀按实测平均抽。顺序：
        整场开始 on_battle_begin → 阿科米亚核心分配 → 开团 → 战斗开始 → 击杀 → 阵亡 / 牺牲（复生）→ 凯旋 / 败阵 →
        战斗结束 on_battle_end → 移除只在本场存在的召唤物（counters["battle_only"]）→ 临时等级清零。
        阵亡记录在 s.battle_deaths（每人次数）/ s.battle_dead（结束时仍阵亡，复生后活下来的不算）。"""
        s = self.s
        s.battle_dead, s.battle_deaths = [], {}
        for name, c, holder in list(self.holders()):
            f = getattr(c, "on_battle_begin", None)
            if f:
                f(self, holder)
        self.core_distribute()
        for u in list(s.on_board()):
            self.fire(u, "open")
        for u in list(s.on_board()):
            self.fire(u, "battle_start")
        win = self.rng.random() < BASE_WIN
        enemy_lv = max(s.lord_level, B.meta.level_target(s.round) + ENEMY_LV_AHEAD)
        enemy_n = B.pop_cap(max(1, min(6, enemy_lv))) + ENEMY_EXTRA
        kills = sum(1 for _ in range(enemy_n) if self.rng.random() < KILL_BASE + (1 - KILL_BASE) * (1.0 if win else 0.0))
        team = [u for u in s.on_board() if not getattr(impl(u.name), "no_attack", False)]
        # 击杀是战斗结果，推演不估强弱 → 击杀数是"外生"的：只按回合和胜负从实测分布抽，**不看推演里的等级 / 阵容**，
        # 否则"等级高 → 击杀多 → 蓄势 / 惊鸿游龙给更多等级"会形成正反馈、把估不准的量越推越大（2026-10-14）。
        host = s.unit(s.maple) if s.maple else None
        host_k = 0
        if host is not None and host in team:
            host_k = self._poisson(host_kill_mean(s.round) * HOST_KILL_SCALE * (HOST_KILL_WIN if win else HOST_KILL_LOSE))
            host.counters["kills_round"] = host_k
        others = [u for u in team if u is not host]
        for _ in range(max(0, kills - host_k)):          # 其余击杀均分给别的单位（无法攻击的召唤物不拿）
            if not others:
                break
            u = self.rng.choice(others)
            u.counters["kills_round"] = u.counters.get("kills_round", 0) + 1
        for u in team:
            k = u.counters.pop("kills_round", 0)
            if k:
                self.fire(u, "kill", n=k)
        for u in list(s.on_board()):
            if u not in s.units:
                continue
            row = B.row_of(u.cell) if u.cell else 2
            pd = DEATH_SCALE * (DEATH_WIN if win else DEATH_LOSE) * (ROW_MULT_WIN if win else ROW_MULT_LOSE)[min(3, max(0, row))]
            if self.rng.random() < min(0.98, pd):
                s.battle_deaths[u.uid] = s.battle_deaths.get(u.uid, 0) + 1
                revived = False
                if u.counters.get("revive"):            # 复生：多一条命（太乙真人 / 自带复生）
                    u.counters["revive"] -= 1
                    revived = True
                self.fire(u, "death")
                if self.has_kw(u, "牺牲"):
                    self.fire(u, "sacrifice")
                if not revived:
                    s.battle_dead.append(u.uid)
        for u in list(s.on_board()):
            self.fire(u, "win" if win else "lose")
        for name, c, holder in list(self.holders()):
            f = getattr(c, "on_battle_end", None)
            if f:
                _call(f, self, holder, win=win)
        for u in list(s.units):
            if u.counters.get("battle_only"):
                self.remove(u, src="战斗结束")
        for u in s.units:
            u.temp = 0.0
            u.counters.pop("revive", None)
        self.note("战斗", win=win, kills=kills)
        return win

    def round_end(self) -> None:
        """回合结束：on_round_end / react_round_end（收入到账之前：存款利息、资本杠杆）→ 经验 → 下一回合 → 收入 → 升级。"""
        s = self.s
        for name, c, holder in list(self.holders()):
            f = getattr(c, "on_round_end", None)
            if f:
                f(self, holder)
        self.emit("round_end")
        s.xp += B.xp_gain(s.round)
        s.round += 1
        s.energy += B.income(s.round)
        self.level_ups()

    def xp_need(self, level: Optional[int] = None) -> int:
        """升到下一级所需经验：XP_NEED + 持有牌的 xp_need_delta（人才济济 -2），至少 1。"""
        lv = self.s.lord_level if level is None else level
        d = 0
        for name, c, holder in list(self.holders()):
            f = getattr(c, "xp_need_delta", None)
            if f:
                d += int(f(self, holder) or 0)
        return max(1, B.XP_NEED.get(lv, 999) + d)

    def level_ups(self) -> int:
        s = self.s
        n = 0
        while s.lord_level < B.MAX_LORD_LEVEL:
            need = self.xp_need()
            if s.xp < need:
                break
            s.xp -= need
            s.lord_level += 1
            n += 1
            self.emit("lord_levelup", level=s.lord_level)
        return n

    def add_xp(self, n: int, src: str = "") -> int:
        """棋手经验 +n，够了立即升级；返回升了几级。"""
        self.s.xp += n
        self.note("棋手经验", n=n, src=src)
        return self.level_ups()

    def pop_cap(self) -> int:
        return B.pop_cap(self.s.lord_level) + self.s.pop_bonus
