"""f6 事实引擎：把"老师花 10 分钟能推出来的结论"算好写进题面。

设计目标（2026-09-24）：老师用 10 分钟能推出的结论，Laya 要在 60ms 内给出 —— 前提是题面信息给够；
不参考规则引擎，直接从卡牌效果与玩法推；每个选择都要求准确。

**信息源只用原始材料**（不 import 规则引擎，不用它的阵容选择/估值/主C 判断）：
  cards.json（卡面原文）、card_kb.json（逐张理解的机制库）、card_usage_reviewed.json（这张牌作用于谁）、
  hero_carry_reviewed.json（主C 强度表 0~9）、card_features.json（定位/前后排/成长方式）、lineups.json（官方推荐阵容：
  阿离 ×1.0、其他棋手 ×0.8）、meta.py（实测的品阶概率/牌池/人口/收入表）、感知读数（ctx）。

输出只写**事实**（会发生什么），不写"该不该"；读不到写 "?"。
学生（f6）与老师（g5）看的是同一份文本（逐字同源）。

ctx 字段（都可缺）：round, gold, level, level_cost, xp, xp_needed, pop, pop_cap, free_refresh, banned,
  auction_in, auction_done, hand[list], hand_awk[list], maple(枫叶舞绑定对象),
  field{name: {cell, lv, awk_left, pool_left, eq}}  —— cell/lv/觉醒/牌池/装备件数都是感知读数（或账本记的格子）。
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
LORD = "阿离"
OTHER_LORD_W = 0.8            # 2026-09-24：阿离权重更高，其他棋手阵容按 8 折计
FV = "f6"                     # 当前渲染的特征版本（features 按 checkpoint 设置）；f7 起的新事实都挂在 FV >= "f7" 下，
                              # 保证线上 f6 学生（R5F6b）看到的文本和训练时逐字一致
SKILL_THRESHOLDS = (10, 40, 100)    # 英雄技能质变等级（卡库 skills[].enhance）
HAND_LIMIT = 10
FACTION_EN = {"逐鹿": "zhulu", "河洛": "heluo", "三分之地": "sanfen", "三分": "sanfen", "大河流域": "dahe", "大河": "dahe",
              "日落海": "riluohai", "无阵营": "neutral", "通用": "neutral", "": "none"}
CAT_EN = {"英雄牌": "hero", "效果牌": "effect", "天赋牌": "talent", "装备牌": "equipment", "棋手技能牌": "lord skill",
          "棋手牌": "lord card", "召唤物": "summon"}
OUTPUT_ROLES = {"mage", "marksman", "assassin", "fighter", "backline"}


# ---------------------------------------------------------------- 静态数据
def _load(rel: str):
    return json.loads((PROJECT_DIR / rel).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def cards() -> dict:
    return _load("data/cards.json")["cards"]


@lru_cache(maxsize=1)
def feats() -> dict:
    return _load("data/card_features.json")


@lru_cache(maxsize=1)
def kb() -> dict:
    return _load("data/card_kb.json")["entries"]


@lru_cache(maxsize=1)
def carry_table() -> dict:
    return _load("data/hero_carry_reviewed.json")


@lru_cache(maxsize=1)
def keywords() -> dict:
    try:
        return _load("data/keywords.json").get("keywords", {})
    except Exception:                                   # noqa: BLE001
        return {}


# 官方关键词（data/keywords.json）的精简英文解释，逐条对照官方原文写（原文太长，15 条全列要 ~300 token）
KW_EN = {
    "牺牲": "on death",
    "合成": "play a copy onto the same hero on board to strengthen it",
    "整备": "at round start",
    "阿科米亚核心": "battle start: the core's level is split as temp levels among your 日落海 heroes (max 50% each)",
    "登场": "when this card is played",
    "凯旋": "if the battle is won while on board",
    "临时等级": "cleared when the battle ends",
    "退场": "when this hero is SOLD",
    "败阵": "if the battle is lost while on board",
    "开团": "at battle start",
    "图腾": "大河流域 unit that gives effects while alive",
    "夺取": "on play: steal levels (max 10, not temp) from 1 random adjacent ally",
    "闪现": "battle start: jumps to the mirrored enemy cell",
    "复生": "revives at 50% HP after dying (max 10 per round)",
    "转瞬": "destroyed at round end; selling gives no energy",
}


@lru_cache(maxsize=1)
def usage() -> dict:
    return _load("data/card_usage_reviewed.json")


@lru_cache(maxsize=1)
def base_equips() -> set:
    try:
        return set(_load("data/equip_recipes.json").get("recipes", {}).keys())
    except Exception:                                   # noqa: BLE001
        return set()


@lru_cache(maxsize=1)
def lineups() -> list:
    """[(阵容, 权重)]：权重 = 使用人数 × (阿离 1.0 / 其他 0.8)。"""
    out = []
    for lu in _load("data/lineups.json").get("lineups", []):
        w = (lu.get("use_num") or 0) * (1.0 if LORD in (lu.get("lord_names") or []) else OTHER_LORD_W)
        if w > 0 and lu.get("heroes"):
            out.append((lu, w))
    return out


def _ht():
    import sys
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from agent.decision import hero_traits
    return hero_traits


@lru_cache(maxsize=1)
def _load_recipes() -> dict:
    try:
        return _load("data/equip_recipes.json").get("recipes", {})
    except Exception:                                   # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def _routes():
    """官方路线（agent/decision/routes.py，规则层同源）：排除名单、候选排名。f9k 起写进题面。"""
    import sys
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from agent.decision import routes

    class _NotWanted:
        """f9x（规范 54.26-2：狄仁杰线的钟馗 `not wanted` 仍进 ROUTE member / missing / CORE 计数 / teammates）：
        路线引擎表 not_wanted 的人从这条路线的 final_team / stage_units / members 里扣掉。f9x 以前原样透传（旧版本逐字不变）。
        "not wanted on this route" 标签由 Board.route_not_wanted 判，不受影响。"""
        def __getattr__(self, name):
            return getattr(routes, name)

        @staticmethod
        def final_team(lu):
            out = routes.final_team(lu)
            nw = route_nw(lu)
            return [h for h in out if h not in nw] if nw else out

        @staticmethod
        def stage_units(lu):
            out = routes.stage_units(lu)
            nw = route_nw(lu)
            return [h for h in out if h not in nw] if nw else out

        @staticmethod
        def members(lu):
            out = routes.members(lu)
            nw = route_nw(lu)
            return out - nw if nw else out
    return _NotWanted()


def route_nw(lu) -> set:
    """f9x：这条路线引擎表里的 not_wanted（f9x 以前恒为空）。"""
    if FV < F9X or not lu:
        return set()
    k = (FV, str(lu.get("key") or ""))
    if k not in _NW_CACHE:
        _NW_CACHE[k] = frozenset(min_engine_of(lu.get("key")).get("not_wanted") or [])
    return _NW_CACHE[k]


_NW_CACHE: dict = {}


@lru_cache(maxsize=1)
def _eng():
    """等级引擎目录（data/engines.json）：产出方 + 放大器 + 落点。f9j 起写进题面。"""
    import sys
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from agent.decision import engines

    class _Versioned:
        """f9lj 起题面用修正后的引擎目录（太乙不是放大器…）；旧版本关掉修正，保持逐字一致。
        规则层同进程默认开着修正，所以每次调用前后切换、调用完恢复。"""
        def __getattr__(self, name):
            fn = getattr(engines, name)
            if not callable(fn):
                return fn

            def call(*a, **k):
                old, oldz = engines.CORRECTED, engines.F9ZB
                engines.CORRECTED = FV >= F9LJ
                engines.F9ZB = FV >= F9ZB        # f9zb（第 884 局）：按主C 选引擎 + 河洛·古币加铠 / 明世隐
                try:
                    return fn(*a, **k)
                finally:
                    engines.CORRECTED, engines.F9ZB = old, oldz
            return call
    return _Versioned()


@lru_cache(maxsize=1)
def hero_roles() -> dict:
    """f9l：英雄定位（data/hero_roles.json：combat_role、engine_grants 主C 引擎喂给谁）。只读数据，不用规则引擎。"""
    try:
        return _load("data/hero_roles.json").get("heroes", {})
    except Exception:                                   # noqa: BLE001
        return {}


def engine_fed_by(n: str, carry: str) -> bool:
    """f9l：n 吃 carry 的引擎红利（hero_roles.engine_grants.feeds 里点名，如玄策主C → 孙悟空/云中君/阿轲/裴擒虎）。"""
    if not carry or not n or n == carry:
        return False
    return any(n in (g.get("feeds") or []) for g in (hero_roles().get(carry) or {}).get("engine_grants") or [])


def route_siblings(route: dict, banned: str = "") -> list:
    """f9l：锁定路线的同体系路线主C（阿离阵容里最终阵容和它共用 ≥3 人、没被禁用派系废掉的；与 rules._sibling_carries 同一判据，
    只是不依赖运行时的备选名单）。有序，保证逐字确定。"""
    R = _routes()
    team = set(R.final_team(route))
    out = []
    for lu in R.ali_lineups():
        k = lu.get("carry") or ""
        if k and k != route.get("carry") and k not in out and len(team & set(R.final_team(lu))) >= 3 and R.viable(lu, banned):
            out.append(k)
    return out


def meta():
    import sys
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from agent.decision import meta as m
    if FV >= F9LT:
        return _MetaLT(m)
    return m


# f9lt（2026-10-15）：6 级商店品阶概率实测（第 3xx~640 局，3841 个满级商店 / 13435 个英雄格），原表只有攻略给的 t3~t5，t1/t2 题面写"unmeasured"。
# 只在题面 f9lt 起使用；规则层 / 局面网络特征仍用 meta.TIER_ODDS（不改已部署模型的输入）。
TIER_ODDS_LV6_MEASURED = {1: 0.01, 2: 0.09, 3: 0.40, 4: 0.35, 5: 0.15}


class _MetaLT:
    """f9lt：meta 的代理，只把 6 级品阶概率换成实测值。"""
    def __init__(self, m):
        self._m = m
        self.TIER_ODDS = dict(m.TIER_ODDS)
        self.TIER_ODDS[6] = dict(TIER_ODDS_LV6_MEASURED)

    def __getattr__(self, k):
        return getattr(self._m, k)

    def tier_odds(self, quality: int, level: int) -> float:
        return self.TIER_ODDS.get(level, {}).get(quality, 0.0)

    def shop_hit_prob(self, quality: int, level: int, remaining=None) -> float:
        m = self._m
        if not quality or quality not in m.POOL_COPIES:
            return 0.0
        rem = m.POOL_COPIES[quality] if remaining is None else max(0, remaining)
        total = m.POOL_HEROES[quality] * m.POOL_COPIES[quality]
        p_slot = self.tier_odds(quality, level) * rem / total
        return 1.0 - (1.0 - p_slot) ** m.SHOP_SLOTS


@lru_cache(maxsize=1)
def board_rows() -> list:
    import sys
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from agent.perception.extractors import OWN_BOARD_ROWS      # 棋盘几何（感知层常量，不是规则判断）
    return [[tuple(c) for c in r] for r in OWN_BOARD_ROWS]


# ---------------------------------------------------------------- 单卡属性
def c(n: str) -> dict:
    return cards().get(n) or {}


def cat(n: str) -> str:
    return c(n).get("category", "")


def is_hero(n: str) -> bool:
    return cat(n) == "英雄牌"


def tier(n: str) -> int:
    return int(c(n).get("quality") or 0)


def fac(n: str) -> str:
    return c(n).get("faction") or ""


FAC_ALIAS = {"三分": "三分之地", "大河": "大河流域"}      # 卡面写简称，卡库写全称（核查 E1：约 100 题结论反了）


def norm_fac(s: str) -> str:
    return FAC_ALIAS.get(s, s)


def faction_count(f: str, B, self_name: str = "", candidate: bool = False) -> int:
    """场上【f】英雄数。卡面写"其他/另外"才排除自己（核查 E2/M6：亚连/貂蝉/吕布的效果会落到自己身上）；
    candidate=True 表示这张牌还没上场，买下后它自己也在场上。"""
    f = norm_fac(f)
    txt = skill(self_name) if self_name else ""
    excl_self = bool(re.search(r"其他|另外|其余", txt))
    cnt = sum(1 for h in B.heroes if fac(h) == f and not (excl_self and h == self_name))
    if candidate and self_name and self_name not in B.heroes and fac(self_name) == f and not excl_self:
        cnt += 1
    return cnt


def fac_en(n: str) -> str:
    return FACTION_EN.get(fac(n), "other")


def role(n: str) -> str:
    return (feats().get(n) or {}).get("role") or ""


def row(n: str) -> str:
    r = (feats().get(n) or {}).get("row")
    if r:
        return r
    t = c(n).get("tags") or []
    return "front" if "前排" in t else "back" if "后排" in t else ""


def melee(n: str) -> bool:
    return ((c(n).get("properties") or {}).get("attackDistance") or 0) <= 1


def skill_base(n: str) -> str:
    """卡面原文（不含赠送牌效果）。resolve() 有 lru_cache、又是拿屏幕上的卡面比对，只能用这个（与特征版本无关）。"""
    return re.sub(r"\s+", " ", c(n).get("skill") or "").strip()


def skill(n: str) -> str:
    s = skill_base(n)
    if FV >= F9LM and c(n).get("given_skill"):
        # f9lm（2026-10-13 卡库重爬）：天赋送出的同名牌的效果（望梅止渴 / 神秘英雄 / 装备天赋…）—— 不用【】，免得被当成派系
        s += " 赠送牌：" + re.sub(r"\s+", " ", c(n)["given_skill"]).strip()
    return s


def skill_all(n: str) -> str:
    """卡面 + 英雄技能描述（瑶/太乙的"最近友军"只写在技能里）。"""
    return skill(n) + " " + " ".join(sk.get("desc", "") for sk in (c(n).get("skills") or []))


def fit(n: str):
    v = carry_table().get(n)
    return v.get("carry_score") if v else None


def fit_why(n: str, limit: int = 22) -> str:
    v = carry_table().get(n) or {}
    w = re.split(r"[；;。（(]", v.get("why", ""))[0]
    return w[:limit]


def is_tanky(n: str) -> bool:
    """坦克：卡库标「坦克」或定位 tank；只有「前排」标签的看主C 评分（≤2 才算坦克 —— 曜只有前排标签，评分 4，是过渡输出）。"""
    if role(n) == "tank" or "坦克" in (c(n).get("tags") or []):
        return True
    if (fit(n) or 0) <= 3 and "偏坦" in (carry_table().get(n) or {}).get("why", ""):
        return True                    # 核查 M4：墨子"前排屏障法伤，偏坦"、铠"战士减伤护盾，偏坦"
    return role(n) == "frontline" and (fit(n) or 0) <= 2


def is_front_tank(n: str) -> bool:
    """站前排的坦克（大司命是后排坦克型输出，攻击距离 2，不算；核查 M10）。"""
    return is_tanky(n) and row(n) == "front"


def is_support(n: str) -> bool:
    if role(n) == "support" or "辅助" in (c(n).get("tags") or []):
        return True
    return (fit(n) or 0) <= 3 and "偏辅助" in (carry_table().get(n) or {}).get("why", "")   # 芈月


def is_output(n: str) -> bool:
    return is_hero(n) and not is_tanky(n) and not is_support(n)


def is_transitional(n: str) -> bool:
    """评分 3~4 的输出位：前期能打，后期性价比低（曜/赵云/姬小满；曜过了前期就该换）。"""
    s = fit(n)
    return is_output(n) and s is not None and 3 <= s <= 4


def lord_xp_unit(n: str) -> bool:
    """f9lk：棋手经验件（弈星："棋手升级时…"/"棋手经验"）—— 升完级就卖，不当主C、不算输出位。"""
    t = skill_all(n)
    return "棋手升级时" in t or "棋手经验" in t


FAC_GATE = {"古币": "河洛"}          # 资源只对某阵营有效（游戏规则：古币只有河洛英雄能用）
LV_GAIN_RE = re.compile(r"等级\s*\+|临时等级|核心等级|古币")
PER_ROUND_RE = re.compile(r"整备|回合开始|每回合|开团|每场")


def carry_feed_gate_ok(n: str, C: str) -> bool:
    """f9p：n 的喂养有阵营限制（卡面【X】英雄 / 古币 = 河洛）时，主C C 是不是那个阵营。没有限制返回 True。"""
    if not C:
        return True
    txt = skill_all(n) + " " + str(((hero_roles().get(n) or {}).get("functions") or {}).get("carry_level") or "")
    facs = {norm_fac(x) for x in re.findall(r"【([^】]+)】", txt)}
    facs |= {v for k, v in FAC_GATE.items() if k in txt}
    facs.discard("")
    return not facs or norm_fac(fac(C)) in facs


def producer_kind(n: str) -> str:
    """f9p：引擎产出方的卡面强度：'' = 卡面不给等级（吕布 / 嬴政只加攻）、'per-round' / 'conditional'（每当…）/ 'one-time'（只有登场）。"""
    sk = skill_base(n)
    if not LV_GAIN_RE.search(sk):
        return ""
    if PER_ROUND_RE.search(sk):
        return "per-round"
    if "每当" in sk:
        return "conditional"
    if "登场" in sk:
        return "one-time"
    return "per-round"


def feeds_carry_role(n: str) -> bool:
    """f9lk：卡面功能是给主C 涨等级的配套（hero_roles functions.carry_level：李元芳 / 明世隐 / 武则天）。"""
    return bool(((hero_roles().get(n) or {}).get("functions") or {}).get("carry_level"))


_STRENGTH = None


def strength_ok(n: str, route_carry: bool) -> bool:
    """f9lk：输出强度审查（data/hero_strength.json，与 rules.carry_strength_ok 同一组门槛：实战 ≥5 局前三率 ≥30% 的放行；
    否则模型分 ≥5.5 可当主C、4~5.5 只当路线指定主C）。只读数据。"""
    global _STRENGTH
    if _STRENGTH is None:
        try:
            _STRENGTH = _load("data/hero_strength.json").get("heroes") or {}
        except Exception:                                        # noqa: BLE001
            _STRENGTH = {}
    h = _STRENGTH.get(n)
    if not h or h.get("score") is None:
        return True
    hist = h.get("history") or {}
    if (hist.get("games") or 0) >= 5 and (hist.get("top3") or 0) / hist["games"] >= 0.30:
        return True
    sc = float(h["score"])
    return sc >= 5.5 or (route_carry and sc >= 4.0)


def nearest_buddy(n: str) -> bool:
    """作用于"距离最近的（友方）英雄"的：必须贴着主C 站（瑶/太乙/明世隐…）。"""
    return bool(re.search(r"距离最近的(?:1名)?(?:友方)?英雄|最近的友方英雄|附身", skill_all(n)))


def fleeting(n: str) -> bool:
    """这张牌自己带「转瞬」：卡面以"转瞬"开头（古币/战术牌…）。「复现」这类"返还的牌带转瞬"的天赋本身不算（核查 E5）。"""
    return skill(n).startswith("转瞬") or any(k.get("name") == "转瞬" for k in c(n).get("keywords") or [])


def fleeting_now(n: str, B) -> bool:
    """f9ll：已选天赋改掉了转瞬（「传家宝」：古币不再附带转瞬，r9za 老师）。只读已选天赋的卡面原文。"""
    if not fleeting(n):
        return False
    if FV < F9LL:
        return True
    for t in (getattr(B, "ctx", {}) or {}).get("talents") or []:
        m = re.search(r"([一-龥]{1,6})不再附带转瞬", skill(t) or "")
        if m and m.group(1) in n:
            return False
    return True


def kw(n: str) -> set:
    return {k.get("name") for k in c(n).get("keywords") or []}


# ---------------------------------------------------------------- 棋盘几何
def cell_rc(cell):
    if not cell:
        return None
    if isinstance(cell, str):
        try:
            cell = tuple(int(x) for x in cell.split(","))
        except ValueError:
            return None
    cell = tuple(cell)
    for r, rw in enumerate(board_rows()):
        if cell in rw:
            return r, rw.index(cell)
    return None


def neighbors(cell) -> set:
    rc = cell_rc(cell)
    if rc is None:
        return set()
    r, i = rc
    rows = board_rows()
    out = {(r, j) for j in (i - 1, i + 1) if 0 <= j < len(rows[r])}
    off = (i - 1, i) if r % 2 == 0 else (i, i + 1)
    for rr in (r - 1, r + 1):
        if 0 <= rr < len(rows):
            out |= {(rr, j) for j in off if 0 <= j < len(rows[rr])}
    return out


def adjacent(a_cell, b_cell) -> bool | None:
    ra, rb = cell_rc(a_cell), cell_rc(b_cell)
    if ra is None or rb is None:
        return None
    return rb in neighbors(a_cell)


# ---------------------------------------------------------------- 阵容统计（阿离 ×1.0 / 其他 ×0.8）
def _final_list(lu: dict) -> list:
    """最终阵容（**有序 list**，保证文本逐字确定；核查 1.3）。
    阵容记录的 heroes 有时是前期过渡阵（核查 M1：阿离遗志海月流 那条 heroes=干将/孙膑/曜/牛魔，最后阶段才是海月一套）：
    最后一段运营计划从第 8 回合以后开始、且和 heroes 重合不到一半时，以最后一段的主力为准。"""
    heroes = [h["name"] for h in lu.get("heroes") or [] if h.get("name")]
    sts = [st for st in lu.get("stages") or [] if st.get("main")]
    if sts:
        last = max(sts, key=lambda st: (st.get("to_round") or 0, st.get("from_round") or 0))
        main = list(dict.fromkeys(last.get("main") or []))
        if (last.get("from_round") or 0) >= 8 and len(set(main) & set(heroes)) * 2 < len(set(main)):
            return main
    return list(dict.fromkeys(heroes))


def lineup_members(lu: dict) -> tuple[list, list]:
    final = _final_list(lu)
    stage = list(dict.fromkeys(h for st in lu.get("stages") or [] for h in (st.get("main") or []) + (st.get("sub") or [])
                               if h not in final))
    return final, stage


def lineup_carry(lu: dict) -> str:
    """阵容主C：记录的 carry 若不在（修正后的）最终阵容里，就不采信（前期阵记录的 carry 是过渡主C）。"""
    final = _final_list(lu)
    k = lu.get("carry") or ""
    return k if (k in final or not final) else ""


@lru_cache(maxsize=256)
def anchored(carry: str) -> dict:
    """以 carry 为主C 的所有阵容：队友出现率、主C 装备出现率、阵容数（阿离几套）。"""
    ls = [(lu, w) for lu, w in lineups() if lineup_carry(lu) == carry]
    tot = sum(w for _, w in ls)
    if not ls or tot <= 0:
        return {"n": 0, "ali": 0, "mates": {}, "equips": {}, "awaken": {}}
    mates, eqs, awk = {}, {}, {}
    for lu, w in ls:
        final = _final_list(lu)
        byname = {h.get("name"): h for h in lu.get("heroes") or []}
        for nm in final:
            h = byname.get(nm) or {}
            if nm != carry:
                mates[nm] = mates.get(nm, 0) + w
            else:
                for e in dict.fromkeys(h.get("equips") or []):        # 同一套里重复挂的同名装备只算一次（核查 E4）
                    eqs[e] = eqs.get(e, 0) + w
            if h.get("is_awakened"):
                awk[nm] = awk.get(nm, 0) + w
    norm = lambda d: {k: v / tot for k, v in sorted(d.items(), key=lambda x: -x[1])}  # noqa: E731
    return {"n": len(ls), "ali": sum(1 for lu, _ in ls if LORD in (lu.get("lord_names") or [])),
            "mates": norm(mates), "equips": norm(eqs), "awaken": norm(awk)}


@lru_cache(maxsize=1)
def carry_popularity() -> dict:
    """每个英雄当主C 的加权阵容数（相对最热门者 0~1）和套数。"""
    w, n = {}, {}
    for lu, ww in lineups():
        k = lineup_carry(lu)
        if k:
            w[k] = w.get(k, 0) + ww
            n[k] = n.get(k, 0) + 1
    mx = max(w.values()) if w else 1
    return {k: (v / mx, n[k]) for k, v in w.items()}


@lru_cache(maxsize=1)
def card_recommend() -> dict:
    """天赋/效果牌被多少阵容列为推荐（加权占比，0~1）+ 阿离几套。"""
    tot = sum(w for _, w in lineups()) or 1
    out = {}
    for lu, w in lineups():
        for k in (lu.get("talent_cards") or []) + (lu.get("effect_cards") or []):
            s, a = out.get(k, (0.0, 0))
            out[k] = (s + w / tot, a + (1 if LORD in (lu.get("lord_names") or []) else 0))
    return out


def coverage(owned: set, top: int = 2, route: dict | None = None) -> list:
    """按我方拥有的英雄（场上+手里）对阵容的覆盖排序：最终阵容 2 分、运营阶段英雄 1 分，× 权重因子。
    f9k：不参选的路线（遗志/海月/过渡主C/1 阶主C，与规则层同一名单）不列；已锁定的路线（route）固定排第 1。"""
    res = []
    for lu, w in lineups():
        if FV >= "f9k" and (_routes().excluded(lu) or (route and str(lu.get("key")) == str(route.get("key")))):
            continue
        final, stage = lineup_members(lu)
        if len(owned & set(final)) < 2:
            continue                   # 只覆盖 1 个英雄的阵容说明不了"往哪走"，是噪声
        pts = 2 * len(owned & set(final)) + len(owned & set(stage))
        f = 1.0 if LORD in (lu.get("lord_names") or []) else OTHER_LORD_W
        res.append((pts * f, math.log10(w + 1), lu))
    res.sort(key=lambda x: (-x[0], -x[1]))
    out, seen = [], set()
    for _, _, lu in res:                       # 同一套最终阵容（不同作者/名字）只留热度最高的一个
        sig = (lineup_carry(lu), frozenset(lineup_members(lu)[0]))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(lu)
        if len(out) >= top:
            break
    if FV >= "f9k" and route:
        out = [route] + out[:top - 1]
    return out


# ---------------------------------------------------------------- 局面分析（每题一次）
class Board:
    def __init__(self, ctx: dict):
        self.ctx = ctx
        self.field = {n: (v if isinstance(v, dict) else {}) for n, v in (ctx.get("field") or {}).items()}
        self.hand = list(ctx.get("hand") or [])
        self.rnd = ctx.get("round") or 0
        lv = ctx.get("level")
        self.level = lv if isinstance(lv, int) and 1 <= lv <= 6 else 0
        self.heroes = [n for n in self.field if is_hero(n)]
        self.hand_heroes = [n for n in self.hand if is_hero(n) and n not in self.field]
        self.owned = set(self.heroes) | {n for n in self.hand if is_hero(n)}
        # ---- 未来主C：场上评分最高的输出位（评分 ≥6；同分按等级）；手里有更高分的另记
        # f9lk（第 580 局审查：第 6 回合 1 阶孙尚香被写成 FUTURE CARRY，整题以它为锚）：候选要过主C 资格（carry_ok）
        self._rc0 = ((_routes().by_key(ctx.get("route")) or {}).get("carry") or "") if FV >= F9LK else ""
        cands = sorted((n for n in self.heroes if is_output(n) and (fit(n) or 0) >= 5 and self.carry_ok(n)),
                       key=lambda n: (fit(n) or 0, self.lv(n) or 0, carry_popularity().get(n, (0, 0))[0]),
                       reverse=True)                  # 同分按"当主C 的阵容热度"（核查 M2：守约 8 分和敖隐同分时顺序随意）
        if FV >= F9ZA:
            # f9za-fix（54.32-12）：carry_pool 开着的主C 排在名单外的人前面（名单外的 fit 再高也只能临时顶位）；稳定排序，名单内外各自原顺序
            cands = sorted(cands, key=lambda n: not f9za_carry_listed(n))
        _blk = set(BLOCKED_CARRIES_F9Y) if FV >= F9Y and not _routes().by_key(ctx.get("route")) else set()
        if FV >= F9Z:
            # f9z（待办：路线失效时亚连仍挂 FUTURE CARRY；54.28-12：屏蔽只管"不新选他当主C"）：
            # 路线失效（路线主C 及同体系主C 都不在我方、第 5 回合起）也按没锁算；现任主C（≥2 阶）本人不屏蔽
            _r0 = _routes().by_key(ctx.get("route"))
            _own0 = {n for n in self.field if is_hero(n)} | {n for n in self.hand if is_hero(n)}
            _stale0 = bool(_r0) and bool(self._rc0) and self.rnd >= 5 and not (
                {self._rc0, *route_siblings(_r0, ctx.get("banned") or "")} & _own0)
            _cc0 = ctx.get("carry") or ""
            _blk = (set(BLOCKED_CARRIES_F9Y) if (not _r0 or _stale0) else set()) - ({_cc0} if (tier(_cc0) or 0) >= 2 else set())
        if _blk:
            # f9y（54.27-17，第 806 / 809 局）：路线没锁时屏蔽路线的主C（安琪拉 / 亚连）不当未来主C / 主C 人选
            cands = [n for n in cands if n not in _blk]
        self.carry_cands = cands[:2]
        self.fc = cands[0] if cands and (fit(cands[0]) or 0) >= 6 else None
        hand_best = sorted((n for n in dict.fromkeys(self.hand_heroes) if is_output(n) and (fit(n) or 0) >= 6 and self.carry_ok(n)
                            and n not in _blk),
                           key=lambda n: (fit(n) or 0, carry_popularity().get(n, (0, 0))[0], n), reverse=True)
        # ↑ 同分时按"当主C 的阵容热度"再按名字 —— 原来遍历 set，字符串哈希每个进程不同，同一题每次渲染可能选不同的人
        self.fc_in_hand = hand_best[0] if hand_best and (not self.fc or (fit(hand_best[0]) or 0) > (fit(self.fc) or 0)) else None
        if FV >= "f9" and not self.fc_in_hand:
            # f9：手里 ≥3 张同名的输出位（上场即可合成到觉醒）也算"更强的候选"（老师 r8f8：艾琳 x5 与守约同分没标出来）
            stack = [n for n in hand_best if self.hand.count(n) >= 3 and (fit(n) or 0) >= ((fit(self.fc) or 0) if self.fc else 0)]
            self.fc_in_hand = stack[0] if stack else None
        # ---- 主T：前排坦克里 觉醒 > 等级 > 品阶
        tanks = [n for n in self.heroes if is_front_tank(n) or (row(n) == "front" and not is_output(n) and not is_support(n))]
        tanks.sort(key=lambda n: (self.awk(n) == 0, self.lv(n) or 0, tier(n)), reverse=True)
        self.main_tank = tanks[0] if tanks else None
        self.front = [n for n in self.heroes if row(n) == "front"]
        self.front_tanks = [n for n in self.heroes if is_front_tank(n)]
        # ---- 阵容：以未来主C 为锚 + 覆盖度。锚点必须**有阵容数据**（核查 M2：锚点 0 套阵容时写"0% of X"毫无信息）
        # f9k：锁定的官方路线（规则层 / Laya route 题定的，是局面状态，同"现任主C"）
        self.route = _routes().by_key(ctx.get("route")) if FV >= "f9k" else None
        self.route_since = ctx.get("route_since") or 0
        rc = (self.route or {}).get("carry") or ""
        if rc and rc in self.heroes and not (FV >= F9LL and not self.carry_ok(rc)):
            # f9ll：路线主C 本身不合格（镜 fit3 这类，r9za 老师）不再被写成 FUTURE CARRY
            # 锁定路线的主C 在场：它就是未来主C（第 34 节"主C = 路线主C"），评分高的非路线输出位退为候选
            self.fc = rc
            self.carry_cands = [rc] + [n for n in self.carry_cands if n != rc][:1]
            if self.fc_in_hand and self.fc_in_hand != rc:
                self.fc_in_hand = None
        elif rc and rc in self.hand_heroes:
            self.fc_in_hand = rc
        # f9l（r9n 老师 14 批反馈：路线主C 一直不在我方，题面仍打 ROUTE member / not in ROUTE，真正在养的主C 被标 not in ROUTE）：
        # 路线主C（含同体系路线的主C）不在场也不在手 → 第 5 回合起判"路线失效"，题面写 stale、不再打任何路线标签；
        # 第 5 回合以前写 "carry X not owned yet"（仍按路线打标签）
        self.route_carry_missing = self.route_stale = False
        if FV >= "f9l" and self.route and rc:
            if not ({rc, *route_siblings(self.route, ctx.get("banned") or "")} & self.owned):
                self.route_carry_missing = True
                self.route_stale = self.rnd >= 5
        self.min_eng = (min_engine_of(ctx.get("route")) if FV >= F9LJ and self.route and not self.route_stale else {})
        if FV >= F9LN and self.route and not self.route_stale:
            # f9ln（2026-10-13 第 617 局：主坦也要在核心阵容里选——主C、副C、主辅助、副辅助、主坦、副坦都指核心阵容里的人）：
            # 墨子不在马可波罗日落海阵容里，只因站前排 + 觉醒被标成 MAIN TANK，换人网络因此一直不换他。路线有效时主T 只从路线最终阵容里选
            _team = set(_routes().final_team(self.route))
            _core_t = [n for n in tanks if n in _team]
            self.main_tank = _core_t[0] if _core_t else None
        # f9lk：路线主C 没到手（含失效）→ 同体系备选路线的主C（route_siblings）是接班人选，场上 / 手里有就优先当未来主C
        self.backup_carries = (route_siblings(self.route, ctx.get("banned") or "")
                               if FV >= F9LK and self.route and self.route_carry_missing else [])
        if self.backup_carries:
            sib_on = [n for n in self.backup_carries if n in self.heroes and self.carry_ok(n)]
            if sib_on and self.fc not in self.backup_carries:
                self.fc = sib_on[0]
                self.carry_cands = [self.fc] + [n for n in self.carry_cands if n != self.fc][:1]
            sib_h = [n for n in self.backup_carries if n in self.hand_heroes and self.carry_ok(n)]
            if sib_h and not sib_on:
                self.fc_in_hand = sib_h[0]
        self.cover = coverage(self.owned, route=self.route)
        chain = [self.fc, *self.carry_cands, self.fc_in_hand, *[lineup_carry(lu) for lu in self.cover]]
        if FV >= F9LK:
            # f9lk：路线有效时锚点 = 路线主C；其余锚点至少 3 套官方阵容（1~2 套的"0% of X lineups"是噪声）
            chain = ([rc] if rc and self.route and not self.route_stale else []) + [a for a in chain if a and anchored(a)["n"] >= 3]
        self.anchor = next((a for a in chain if a and anchored(a)["n"] > 0), None)
        self.anch = anchored(self.anchor) if self.anchor else {"n": 0, "mates": {}, "equips": {}, "awaken": {}, "ali": 0}
        self.nw = route_nw(self.route) if self.route and not self.route_stale else set()
        if self.nw and self.anchor == rc:
            # f9x（54.26-2）：锁定路线不要的人（狄仁杰线钟馗）不进 "teammates X%" / "in X% of 主C lineups"（官方阵容统计里他 98%）
            self.anch = dict(self.anch, mates={k: v for k, v in self.anch["mates"].items() if k not in self.nw})
        # 商店里每张牌几张（同名汇总用；由 features.q_shop 放进 ctx["_shop_all"]）
        self.shop_cnt: dict = {}
        for x in ctx.get("_shop_all") or []:
            self.shop_cnt[x] = self.shop_cnt.get(x, 0) + 1
        # ---- 计数
        self.factions: dict = {}
        for n in self.heroes:
            self.factions[fac(n)] = self.factions.get(fac(n), 0) + 1
        self.awakened = [n for n in self.heroes if self.awk(n) == 0]
        self.temp_src = [n for n in self.heroes if any(g.get("res") == "临时等级" for g in (kb().get(n) or {}).get("gives", []))]
        self.coin_src = [n for n in self.heroes if any(g.get("res") in ("古币", "精致古币") for g in (kb().get(n) or {}).get("gives", []))]
        # f9v（2026-10-19：缺少提供临时等级的件）：明世隐能转的是**英雄**身上的临时等级 —— 米莱狄只给召唤单位、不算；
        # 只给自己的（李信 / 诸葛亮 / 敖隐 / 东皇 / 花木兰 / 司空震）只有本人是主C 时才算；日落海核心按场上日落海英雄算
        _c0 = ctx.get("carry") or ""
        self.temp_src_hero = [n for n in self.heroes if re.search(TEMP_TO_HEROES_RE, skill(n) or "")
                              or (n == _c0 and re.search(TEMP_TO_SELF_RE, skill(n) or ""))]
        if any(fac(n) == norm_fac("日落海") for n in self.heroes):
            self.temp_src_hero.append("阿科米亚核心")
        self.blink = [n for n in self.heroes if "闪现" in kw(n) or skill(n).startswith("闪现")]
        self.deploy_src = [n for n in self.heroes if "登场" in kw(n)]
        self.prepare_src = [n for n in self.heroes if "整备" in kw(n)]
        self.sacrifice_src = [n for n in self.heroes if "牺牲" in kw(n)]
        self.effect_in_hand = [n for n in self.hand if cat(n) == "效果牌"]
        pop, cap = ctx.get("pop"), ctx.get("pop_cap")
        self.cap_raised = False
        if FV >= F9LK and self.level and isinstance(cap, int) and cap < meta().pop_cap_at(self.level):
            cap, self.cap_raised = meta().pop_cap_at(self.level), True   # 升级后人口上限没重读（商店开着读不到）：按等级表
        if FV >= F9LQ and self.level and not (isinstance(pop, int) and pop >= 0 and isinstance(cap, int) and cap > 0):
            # f9lq（第 631 局 R1：人口没读到 → "free slots ?"，学生不知道场上 0 人、5 个空位，留着 3 能量只买了 1 个人，当回合掉 10 血）：
            # 读不到就按棋盘名单数人头、按等级表取上限（两者都是原始材料：账本里的单位 + 游戏规则表）
            if not (isinstance(cap, int) and cap > 0):
                cap = meta().pop_cap_at(self.level)
            if not (isinstance(pop, int) and pop >= 0):
                pop = len(self.heroes)
        self.cap = cap
        self.pop = pop
        self.pop_counted = FV >= F9LQ and not (isinstance(ctx.get("pop"), int) and ctx.get("pop") >= 0) and isinstance(pop, int)
        self.free = (cap - pop) if isinstance(pop, int) and isinstance(cap, int) and cap >= pop else None
        # 感知矛盾：账本名单和人口读数对不上（核查缺 10：全量 345 题）—— 标出来，以人口为准
        self.mismatch = (f"board list has {len(self.heroes)} heroes but population reads {pop} (list may be stale; trust population)"
                         if isinstance(pop, int) and pop >= 0 and len(self.heroes) != pop else "")
        # f9e：阵容骨架 = 现任主C + 未来主C（+ 手里更强的候选）；其余输出位是"多余输出"
        # （2026-09-26：不宜有过多输出英雄，应以主C 和未来主C 为核心，配上主T 和未来主T，其它以化学反应或辅助为主；
        #  孙尚香、百里在的情况下又买艾琳和阿轲，输出英雄在等级低的情况下毫无作用"）
        cc = ctx.get("carry") or None
        self.cur_carry = cc if cc in self.field else None
        self.core_out = [h for h in dict.fromkeys([self.cur_carry, self.fc, self.fc_in_hand]) if h]
        self.outputs = [n for n in self.heroes if is_output(n) and not (FV >= F9LK and lord_xp_unit(n))]
        self.extra_out = [n for n in self.outputs if self.is_extra_output(n)]
        # f9j：等级引擎（2026-09-26：开团、整备、核心等级这些 Tag 的效果可以叠加，没利用上时等级比其他玩家少约 100 级）
        # 场上 + 手里最成型的一套；started = 产出方/放大器 ≥2；题面写状态 + 还缺谁，候选/单位行标零件
        self.engine = None
        if FV >= "f9j":
            own = set(self.heroes) | set(self.hand)
            if FV >= F9LI:
                # f9li（2026-10-11 第 564 局）：同一人不重复计角色；有锁定路线时引擎跟路线走（路线最终阵容属于哪套引擎），
                # 场上碰巧凑出来的别的体系引擎不认（原来把沈梦溪路线上的占位张良 / 鬼谷子标成"大河·开团 ENGINE PART"）
                b = None
                if self.route and not self.route_stale:
                    ek, _ = (_eng().lineup_engine(tuple(_routes().final_team(self.route)), self.route.get("carry") or "")
                             if FV >= F9ZB else _eng().lineup_engine(tuple(_routes().final_team(self.route))))
                    b = _eng().status(ek, own) if ek else None
                else:
                    b = _eng().best(own)
            else:
                b = _eng().best(own, distinct=False)
            self.engine = b if b and b["score"] >= 1.5 else None      # 只有一个落点/产出方不算"有引擎"
        self.engine_started = bool(self.engine and self.engine["started"]
                                   and (FV < "f9k" or self.engine.get("producers")))
        if FV >= F9LL and self.route and not self.route_stale:
            # f9ll（第 587 局：主C 已是安琪拉 fit8，锁定的守约路线却把"守约本人"写成 MIN ENGINE piece missing，R25b 0.84 买了守约、
            # 换人网络再卖人给他腾位 → 双主C）：已有合格主C、又不是路线主C、路线主C 也不在场 → 这条路线不再适用，按 stale 处理
            _rc = self.route.get("carry") or ""
            _cs = self.carry_set()
            if _cs and _cs != _rc and _rc not in self.field:
                self.route_stale = True
                self.min_eng = {}
                self.route_superseded = _cs
        # f9lm（2026-10-13 理想打法）：临时主C / 最终主C 已定 / 候选路线核心卡（与 rules.flex_core_cards 同一份原始数据，自己算）
        self.temp_carry = self.final_carry = None
        self.flex_keys: list = []
        self.flex_core: dict = {}
        if FV >= F9LM:
            cc = self.cur_carry
            if cc:
                # 2026-10-13 老师反馈：1 阶觉醒（孙尚香 / 李元芳）、不合格（西施 will be replaced）、或另有不同的 future carry 时
                # 原来也写 settled，和 hold 行 "NO settled carry yet (X t1 is a placeholder)" 自相矛盾 → 收紧，与 rules.final_carry_settled 同口径
                t1 = (tier(cc) or 0) == 1                       # 觉醒的 1 阶也算临时
                if t1 and cc != self._route_carry_raw():
                    self.temp_carry = cc
                if not t1 and self.carry_ok(cc) and not (self.fc and self.fc != cc):
                    # f9s（r9zv 老师 6 题）：规范 54.15-7"已定 = 枫叶舞在他身上或 ≥10 级"，题面原来不看这两条 → 老师每题自己重判
                    if FV < F9S or ctx.get("maple") == cc or (self.lv(cc) or 0) >= 10:
                        self.final_carry = cc
            self.flex_keys = flex_route_keys(ctx, settled=bool(self.final_carry), carry=self.final_carry or "")
            for k in self.flex_keys:
                lab = flex_route_label(k)
                for h in flex_core_of(k):
                    self.flex_core.setdefault(h, [])
                    if lab not in self.flex_core[h]:
                        self.flex_core[h].append(lab)

    def _route_carry_raw(self) -> str:
        """f9lm：锁定路线的主C（不论路线是否失效）。"""
        return ((_routes().by_key(self.ctx.get("route")) or {}).get("carry") or "")

    def own_copies(self, n: str) -> int:
        """f9lm：已有张数 = 场上 1 个单位 + 手里的同名张数。"""
        return (1 if n in self.field else 0) + self.hand.count(n)

    def carry_ok(self, n: str) -> bool:
        """f9lk：能不能当主C（与 rules._carry_eligible 同口径的事实部分）：输出位、不是棋手经验件；路线主C 一律可以；
        第 5 回合后 1 阶不行（觉醒也不行）、强度审查不过的不行；给主C 涨等级的配套（李元芳…）不当主C（现任除外）。"""
        if FV < F9LK:
            return True
        if not is_output(n) or lord_xp_unit(n):
            return False
        if (fit(n) or 0) < 4:
            return False               # 审定主C 评分 <4（镜 / 马超）路线主C 也不行（与 rules HERO_CARRY_MIN 一致；r9z 老师：镜 fit3 成了未来主C）
        if FV >= F9ZA and f9za_carry_off(n):
            return False               # f9za-fix（r9zzg batch_009）：carry_pool 关掉的 2 阶以上主C 不当未来主C / 主C 人选（不标 FUTURE CARRY / CARRY CANDIDATE）
        if n == self._rc0:
            return True
        if self.rnd > 5 and ((tier(n) or 0) <= 1 or not strength_ok(n, n in self._route_carries())):
            return False
        if feeds_carry_role(n) and n != (self.ctx.get("carry") or ""):
            # 给主C 涨等级的配套（李元芳 / 武则天）不当主C；但本身是阿离某条路线主C 的（亚连：日落海涨级）不算配套
            return any((lu.get("carry") or "") == n for lu in _routes().ali_lineups())
        return True

    def _route_carries(self) -> set:
        """f9lk：算"路线指定主C"的人（强度 4~5.5 只能当路线主C）：锁定路线的主C + 同体系路线的主C；没锁路线时 = 阿离路线的主C。"""
        if "_rcs" not in self.__dict__:
            R = _routes()
            rt = R.by_key(self.ctx.get("route"))
            if rt:
                self._rcs = {rt.get("carry") or ""} | set(route_siblings(rt, self.ctx.get("banned") or ""))
            else:
                self._rcs = {lu.get("carry") or "" for lu in R.ali_lineups() if R.viable(lu, self.ctx.get("banned") or "")}
        return self._rcs

    def carry_why_not(self, n: str) -> str:
        """f9lk：carry_ok 不通过的具体原因（题面写出来，r9z 老师：4~5 阶也被套上"t1 / lord-xp …"一串固定理由）。"""
        if not is_output(n):
            return "not an output hero"
        if lord_xp_unit(n):
            return "lord-xp unit"
        if (fit(n) or 0) < 4:
            return f"carry fit {fit(n)}/9 too low"
        if FV >= F9ZA and f9za_carry_off(n):
            return "switched off in Ali's carry list"          # f9za-fix（同 carry_ok）
        if n == self._rc0:
            return ""
        if self.rnd > 5 and (tier(n) or 0) <= 1:
            return "t1 past round 5"
        if self.rnd > 5 and not strength_ok(n, n in self._route_carries()):
            return "damage too low for a carry (strength review; not a route carry)"
        if feeds_carry_role(n) and n != (self.ctx.get("carry") or ""):
            return "carry feeder (levels the carry)"
        return ""

    def hand_carry_label(self) -> str:
        """手里那个候选怎么称呼：f9k 起，它是路线主C 且评分不比场上未来主C 高时写 ROUTE carry（r9m6 老师：6/9 被写成 stronger）。"""
        n = self.fc_in_hand
        if FV >= "f9k" and self.route and n == self.route.get("carry") and self.fc and (fit(n) or 0) <= (fit(self.fc) or 0):
            return "ROUTE carry"
        return "stronger carry"

    def route_past_stage(self, n: str) -> bool:
        """f9lv：路线成员里的"开局件"已经过了它所在的阶段（不在最终阵容、当前及之后的阶段里也没有它）。"""
        R = _routes()
        if not self.route or n in R.final_team(self.route) or n == self.route.get("carry") or not self.rnd:
            return False
        sts = self.route.get("stages") or []

        def _to(i, st):
            t = st.get("to_round") or 0
            if t:
                return t
            # f9t（第 747 局：伞亚连 3 个阶段都没写回合，开局阶段"干将 + 牛魔"被当成一直有效 → R10 牛魔仍是 ROUTE member、卖不掉；
            # 阵容库 171 个阶段里 102 个没写回合）：没写回合的多阶段路线，第一个阶段算开局（到第 6 回合）
            if FV >= F9T and i == 0 and len(sts) >= 2 and not (st.get("from_round") or 0):
                return 6
            return 99
        later = {h for i, st in enumerate(sts) if _to(i, st) >= self.rnd
                 for h in (st.get("main") or []) + (st.get("sub") or [])}
        return n not in later

    def route_member(self, n: str) -> bool:
        return bool(self.route) and not self.route_stale and n in _routes().members(self.route)

    def route_not_wanted(self, n: str) -> bool:
        """f9w（2026-10-19：该卖钟馗、保留李元芳、上明世隐）：路线引擎表 not_wanted 的人（狄仁杰路线的钟馗）不算路线成员 / 最终阵容。"""
        return FV >= F9W and n in set((self.min_eng or {}).get("not_wanted") or [])

    def route_sub_only(self, n: str) -> bool:
        """f9x（54.26-12：亚瑟只在马可波罗路线最后阶段的 sub 里，题面写 ROUTE stage unit，老师当成路线外）：
        路线成员里只出现在阶段 sub（不在最终阵容、任何阶段 main、early_priority、extra_wanted）的人 = 计划外的可选替补。"""
        if FV < F9X or not self.route or self.route_stale or n == self.route.get("carry"):
            return False
        R = _routes()
        if n not in R.members(self.route) or n in R.final_team(self.route) or self.route_extra_wanted(n):
            return False
        if (n in ((self.min_eng or {}).get("early_priority") or []) or n in ((self.min_eng or {}).get("late_fill") or [])
                or (self.anch["mates"].get(n) or 0) >= 0.8):     # late_fill（狄仁杰线上官婉儿、太乙）也算计划内
            return False                # 主C 官方阵容几乎必带的（守约线程咬金 98%）算在计划里（同 f9o）
        return not any(n in (st.get("main") or []) for st in self.route.get("stages") or [])

    def route_core_engine(self, n: str) -> bool:
        """f9y（54.27-22 / 54.23-5 / 54.21-8：日落海路线的雅典娜标成 SECONDARY / not in ROUTE）：路线引擎表 early_priority / 最低配套里的人
        不在官方阵容名单里也是路线核心引擎件。"""
        if FV < F9Y or not self.route or self.route_stale or not is_hero(n) or self.route_not_wanted(n):
            return False
        me = self.min_eng or {}
        if (FV >= F9Z and n in SUNSET_ENGINE_F9Z and n != self.route.get("carry")
                and fac(self.route.get("carry") or "") == "日落海"):
            return True         # f9z（54.25-2 / 54.28-3：日落海路线上明世隐写 not in ROUTE）：日落海线的明世隐 / 雅典娜按引擎件
        return n in (me.get("early_priority") or []) or any(n in g for g in min_engine_groups(me))

    def route_extra_wanted(self, n: str) -> bool:
        """f9w（2026-10-19：沈梦溪路线该上刘禅）：路线引擎表 extra_wanted 的人按最终阵容件算。"""
        return FV >= F9W and n in set((self.min_eng or {}).get("extra_wanted") or [])

    def secondary_engine_piece(self, n: str) -> bool:
        """f9lz（第 682 局亚瑟；2026-10-17：日落海推荐阵容里没有他，可不选，但可作次选，总比牛魔一直在场上好）：
        路线锁定后，不在推荐阵容、但是路线主C 等级引擎的产出方 / 放大器、且与路线主C 同阵营 → 次选（核心之后、路线外之前）。牺牲类引擎不算。"""
        if FV < F9LZ or not (self.route and not self.route_stale) or n in _routes().members(self.route):
            return False
        if self.route_core_engine(n):
            return False                # f9y：引擎表 early_priority（日落海雅典娜）是核心件，不是次选
        if not self.engine or "sacrifice" in str(self.engine.get("key") or ""):
            return False
        rc = self.route.get("carry") or ""
        if FV >= F9M and self.destroys_own_totems(n):
            return False
        return bool(rc) and self.engine_role(n) in ("producer", "amplifier") and bool(fac(n)) and fac(n) == fac(rc)

    def destroys_own_totems(self, n: str) -> bool:
        """f9m（r9zq 老师 001/014/017 报刘邦）：卡面"摧毁你所有的图腾"，而场上或路线里有召唤图腾的单位 → 它拆的是本路线的图腾。"""
        if "摧毁你所有的图腾" not in (skill(n) or ""):
            return False
        pool = set(self.heroes) | (set(_routes().members(self.route)) if self.route and not self.route_stale else set())
        return any(re.search(r"召唤\S{0,4}图腾", skill(x) or "") for x in pool if x != n)

    # ---- f9lw（第 652~655 局复盘裁决）----
    def maple_on_temp(self, n: str) -> bool:
        """㊱：枫叶舞宿主不是未来主C、而未来主C 已在场 → 宿主是临时主C，枫叶舞的等级在浪费。"""
        if FV >= F9ZB and f9zb_committed_offlist(self, n):
            return False               # f9zb（54.37-1）：已认的名单外主C 不是临时宿主，不写换绑
        if not (self.fc and n != self.fc and self.fc in self.heroes):
            return False
        # r9zn_c/000：宿主本身是合格、已投入的主C（t3 狄仁杰 13 级）时不算"临时主C"——裁决针对的是临时主C 上的枫叶舞
        if not (n == self.temp_carry or (tier(n) or 9) <= 1 or not self.carry_ok(n)):
            return False
        # 老师 r9zn/004、011：接班的"未来主C"本身是 1 阶临时人选（孙尚香）时，不算"该换绑"（临时不换临时，50.2）
        rc = (self.route or {}).get("carry") if self.route and not self.route_stale else ""
        return (tier(self.fc) or 0) >= 2 or self.fc == rc

    def one_carry_locked(self) -> bool:
        """㊴：路线已锁、路线主C 已拥有（在场或在手）→ 只养这一个主C（不要求他已经是现任主C；第 40 节旧条件太窄）。"""
        if not self.route or self.route_stale:
            return False
        rc = self.route.get("carry") or ""
        return bool(rc) and (rc in self.heroes or rc in self.hand)

    def other_lineup_carry(self, n: str) -> str:
        """㊴：别的阵容的主C、不在本路线核心阵容里 → 在这局没有位置（阿离只养一个主C）。返回标签，不是返回 ""。"""
        if not is_hero(n):
            return ""
        locked = self.one_carry_locked()
        # f9ly（2026-10-16 晚第 673 局：主C 马可波罗已定、路线还没锁时买了守约，没有道理）：
        # "定下" = 路线已锁，或主C 已定（现任、非临时、合格主C）—— 任一即可
        set_carry = (FV >= F9LY and not locked and self.cur_carry and self.cur_carry != self.temp_carry
                     and (tier(self.cur_carry) or 0) >= 2 and self.carry_ok(self.cur_carry))
        if not (locked or set_carry):
            return ""
        rc = (self.route.get("carry") or "") if locked else self.cur_carry
        if n == rc or n == self.cur_carry or (locked and (n in _routes().final_team(self.route) or (self.route_member(n) and FV < F9LY))):
            return ""          # 老师 r9zn_b/027：现任主C（lv73 守约）被标"别家主C"——现任主C 有位置，换不换是 carry 题的事
        # f9ly（2026-10-16 晚，第 670 局沈梦溪：官方路线里只是替补 sub）：定下守约路线之后，它占掉一张可以用来养守约的牌 →
        # 路线锁定后，路线阶段的替补也算别家主C（只有最终阵容、喂主C 的人除外，下面 feeds_or_part 判）
        lus = _routes().ali_lineups()
        if FV >= F9LX:
            lus = lus + _routes().custom_lineups(include_disabled=FV < F9Q)   # f9lx（第 665 局：马可波罗只在自定义路线里当主C，锁守约后照样被买上场）
        if not any((lu.get("carry") or "") == n for lu in lus):
            return ""
        if is_support(n) or nearest_buddy(n) or any(n in g for g in min_engine_groups(self.min_eng or {})):
            return ""          # 老师 r9zn/008：太乙真人 / 明世隐这类功能辅助、最低配套件不是"没位置的别家主C"
        if FV >= F9R and is_front_tank(n):
            return ""          # f9r（r9zt 老师 test batch_003 e32d34dd644f）：坦克程咬金在某套官方阵容里当主C，但在阿离单主C 打法里是坦克，不是"没位置的别家主C"
        if self.feeds_or_part(n):
            return ""          # 老师 r9zn/009：孙膑（放技能给主C 加等级）、海月（牺牲触发守约）不是"没位置"，是喂主C 的人
        if FV >= F9LY:
            # f9ly（2026-10-16 晚：可以多买主C，但路线定下之后，其他主C 优先级低于核心牌）：不是"没位置"，是排在核心牌后面、不喂资源
            return f"OTHER LINEUP carry: ranks BELOW core cards, never fed (Ali raises 1 carry: {rc})"
        return f"OTHER LINEUP carry: no role (Ali raises 1 carry: {rc})"

    def feeds_or_part(self, n: str) -> bool:
        """f9lw：n 会给现任 / 未来主C 涨等级、产资源，或是当前等级引擎的产出方 / 放大器 → 不是"只放装备的第二输出 / 别家主C"。"""
        if feeds_carry_role(n) and not (FV >= F9P and not carry_feed_gate_ok(n, self.cur_carry or self.fc)):
            return True
        if self.engine and self.engine_role(n) in ("producer", "amplifier")                 and not (FV >= F9P and self.engine_role(n) == "producer" and not producer_kind(n)):
            # f9lz（2026-10-16 晚：花木兰和项羽的牺牲效果只提升自己、对战局帮助不大，守约为核心的阵容里不会有他们）：
            # 路线锁定后，路线外的人只靠"带牺牲 / 带整备"算进引擎的不算喂主C —— 阵容按核心阵容来
            if not (FV >= F9LZ and self.route and not self.route_stale and "sacrifice" in str(self.engine.get("key") or "")
                    and n not in set(_routes().final_team(self.route)) | set(_routes().members(self.route))):
                return True
        C = self.cur_carry or self.fc
        if C and n != C:
            try:
                syn = carry_synergy(n, self) or ""
            except Exception:  # noqa: BLE001
                syn = ""
            if re.search(r"\+\d+ (?:temp )?lv|FUEL|levels? the carry", syn):
                return True
        return False

    def filler_paid(self, n: str):
        """㊳：弈星上场后棋手升过几级（ctx lordxp_since = {名字: 第一次在场时的棋手等级}）；不知道返回 None。"""
        since = (self.ctx.get("lordxp_since") or {}).get(n)
        if not isinstance(since, int) or not self.level:
            return None
        return max(0, self.level - since)

    def filler_text(self, n: str) -> str:
        k = self.filler_paid(n) if n in self.heroes else None
        base = "TRANSITIONAL filler (helps no hero, lord xp negligible)"
        if k is None:
            return base          # 不知道上场后棋手升过几级（老日志没记）：不猜
        if k >= 1:
            return f"{base}; PAID OFF (lord +{k} lv since deployed): sell now"
        return base + "; pays off after 1 lord level-up on board"

    def carry_set(self) -> str:
        """f9l：已定的主C（现任主C，没有就是未来主C）。1 阶 / 过渡型（评分 ≤4）的不算"定了"—— 第 35 节：1 阶主C 过了前期要换。"""
        if FV >= F9LK:
            # f9lk（r9z 老师反馈：现任 1 阶 will be replaced、而觉醒守约已是未来主C 时，商店输出位仍标 CARRY CANDIDATE / C-POSITION 指向旧主C）：
            # 现任不合格就看未来主C
            # 现任 / 未来主C 里第一个"合格且已定型"的（路线指定主C 不论品阶；其余 2 阶以上、评分 >4）
            return next((x for x in (self.cur_carry, self.fc) if x and self.carry_ok(x)
                         and (x == self._rc0 or ((tier(x) or 0) >= 2 and (fit(x) or 0) > 4))), "")
        else:
            C = self.cur_carry or self.fc or ""
        return C if C and tier(C) >= 2 and (fit(C) or 0) > 4 else ""

    def second_carry_engine_tag(self, n: str) -> str:
        """f9l：同一个英雄既是 2nd carry candidate 又是引擎零件时，写清楚它在引擎里是哪一种（老师：第 33 / 35 节冲突）：
        产出方/放大器 = 零件，照留照养，不是第二主C；落点 = 主C 已定时就是第二主C，不养（吃主C 引擎红利的除外）。"""
        r = self.engine_role(n)
        if not r or not self.engine:
            return ""
        name = self.engine["name"]
        if r in ("producer", "amplifier"):
            return (f"ENGINE PART, not a 2nd carry ({r} of {name}" + ("" if self.engine_started else ", engine not started")
                    + f"; fit {fit(n)}/9: keep & feed)")
        feeder = next((C for C in dict.fromkeys([self.cur_carry, self.fc]) if C and engine_fed_by(n, C)), "")
        if feeder:
            return f"ENGINE PAYOFF fed by {feeder}'s engine ({name}; fit {fit(n)}/9: can raise)"
        if not self.carry_set():
            return ""                  # 主C 还没定（或是 1 阶 / 过渡型）：照旧写 2nd carry candidate + 引擎标签
        if self.route_member(n):
            # 路线成员照留（第 34 节 ROUTE 优先于引擎 / 第二主C），只是资源先给主C；"ROUTE member" 标签另起一条
            return f"ENGINE PAYOFF, keep ({name}; carry {self.carry_set()} set: carry gets resources first; fit {fit(n)}/9)"
        return f"2nd carry = extra ENGINE PAYOFF ({name}; carry {self.carry_set()} set: don't raise; fit {fit(n)}/9)"

    def engine_role(self, n: str) -> str:
        """n 在当前引擎里的角色：producer / amplifier / payoff / ''。"""
        if not self.engine:
            return ""
        if FV >= "f9k" and n in (self.engine.get("payoff") or []):
            return "payoff"      # f9k ⑤：百里守约/李信/云中君既是放大器又是落点，按落点报（和引擎行一致）
        return _eng().role(self.engine["key"], n)

    def engine_critical(self, n: str) -> bool:
        """f9lv：卖掉 n 引擎会不会断（产出方 + 放大器去掉 n 后不足 2 个），或 n 是唯一的放大器。
        r9zl 老师 13 批里 10 批报"可卖的全标 ENGINE PART"：牺牲产出方常有 4~5 个，卖一个不断引擎。"""
        if not self.engine:
            return False
        prod = [x for x in self.engine.get("producers") or [] if x != n]
        amp = [x for x in self.engine.get("amplifiers") or [] if x != n]
        if n in (self.engine.get("amplifiers") or []) and not amp:
            return True
        return len(set(prod) | set(amp)) < 2

    def engine_part_text(self, n: str, r: str) -> str:
        """f9lv：零件标签——卖了会断才写 ENGINE PART，否则写 redundant（同角色还有几个）。"""
        name = self.engine["name"]
        if self.engine_critical(n):
            return f"ENGINE PART ({r} of {name}): selling breaks the level engine"
        same = [x for x in self.engine.get(r + "s" if r != "payoff" else "payoff") or [] if x != n]
        return f"redundant {r} of {name} ({len(same)} others remain; selling does not stop the engine)"

    def engine_tag(self, n: str) -> str:
        """单位行 / 候选行的引擎标签（f9j）。零件只在引擎转起来后才标（一个孤零零的产出方不算）。"""
        r = self.engine_role(n)
        if not r or not self.engine:
            return ""
        if FV >= F9N and self.destroys_own_totems(n):
            return "destroys your own totems at battle start (works AGAINST the totem engine)"
        name = self.engine["name"]
        if r == "payoff":
            return f"ENGINE PAYOFF ({name})"
        if FV >= F9LV and self.engine_started:
            return self.engine_part_text(n, r).split(":")[0]
        return f"ENGINE PART ({r} of {name})" if self.engine_started else f"engine part ({r} of {name}, engine not started)"

    def engine_missing(self) -> list[str]:
        if not self.engine:
            return []
        return _eng().missing_parts(self.engine["key"], set(self.heroes) | set(self.hand))

    def is_extra_output(self, n: str) -> bool:
        """f9e：多余输出 = 不在核心里、且评分**不高于**核心里最好的那个的输出位。
        评分更高的是"主C 人选"不是多余（老师 r9f：现任是过渡件曜 / 场上还没有未来主C 时，fit 8 的百里守约也被标成多余）；
        评分 ≤2 的（瑶 fit 1：附身友军）本来就不是输出，也不算。"""
        if not is_output(n) or n in self.core_out or (fit(n) or 0) <= 2 or nearest_buddy(n):
            return False
        if FV >= F9LK and lord_xp_unit(n):
            return False
        core = self.core_out
        if FV >= "f9h" and self.rnd > 6:
            # f9h（r9j 老师 4 批一致报告：百里守约 t2 当主C 时 4 阶 fit8 的艾琳仍标 EXTRA，和第 30 节"该接班"矛盾）：
            # 过了前期，1~2 阶的核心不作为"评分上限"—— 比它品阶高的输出位是接班人选，不是多余
            core = [h for h in core if (tier(h) or 9) > 2 or (tier(n) or 0) <= 2]
        best = max([fit(h) or 0 for h in core] or [0])
        return best > 0 and (fit(n) or 0) <= best

    # 读数
    def v(self, n):
        return self.field.get(n) or {}

    def lv(self, n):
        x = self.v(n).get("lv")
        return x if isinstance(x, int) and x > 0 else None

    def awk(self, n):
        x = self.v(n).get("awk_left")
        return x if isinstance(x, int) else None

    def pool(self, n):
        x = self.v(n).get("pool_left")
        return x if isinstance(x, int) else None

    def eq(self, n):
        x = self.v(n).get("eq")
        return x if isinstance(x, int) else None

    def cell(self, n):
        return self.v(n).get("cell")

    def fodder(self) -> list:
        """可被夺取/可移除的非核心：不是未来主C、不是主T、评分 ≤4。
        f7：再加上"后期低价值"（过了前期、不在主C 阵容里的 1~2 阶）—— 老师 r6 标注指出：场上有"孙膑 lv 2 | low value late"
        （评分 5），题面却写"NO free slot (only core heroes)"。"""
        excl = (self.fc, self.main_tank) + ((self.ctx.get("carry"),) if FV >= "f9f" else ())
        if FV >= "f9j" and self.engine_started:
            # f9lv：只排除"卖了引擎就断"的零件；冗余的产出方可以当换下的人
            excl = excl + tuple(n for n in self.heroes if self.engine_role(n) in ("producer", "amplifier")
                                and (FV < F9LV or self.engine_critical(n)))
        if FV >= "f9k":
            excl = excl + tuple(n for n in self.heroes if (self.engine_started and self.engine_role(n) == "payoff")
                                or self.route_member(n))
        base = [n for n in self.heroes if n not in excl and (fit(n) or 0) <= 4]
        if FV >= "f7":
            base += [n for n in self.heroes if n not in base and n not in excl
                     and any(t.startswith("low value late") for t in self.identity(n))]
        return base

    def carry_lineup_mate(self, n: str) -> str:
        """f9m（第 686 局 R5：主C 马可波罗已定、路线没锁，「限制性进化」把米莱狄 / 狂铁变掉了 —— 米莱狄是"日落海核心马可波罗"的核心件，
        题面只写 `transitional carry: early game only`；马可波罗官方阵容只有 1 套，"in N% of X lineups"锚点要 ≥3 套，没写出来）：
        路线没锁、主C 已定时，n 在主C 的官方 / 自定义阵容里 → 返回阵容名。"""
        if FV < F9M or (self.route and not self.route_stale) or not self.fc or n == self.fc:
            return ""
        R = _routes()
        ban = self.ctx.get("banned") or ""
        for lu in tuple(R.ali_lineups()) + tuple(R.custom_lineups(include_disabled=FV < F9Q)):
            if lu.get("carry") == self.fc and n in R.members(lu) and R.viable(lu, ban):
                return lu.get("name") or "?"
        return ""

    def identity(self, n: str) -> list[str]:
        out = []
        _clm = self.carry_lineup_mate(n)
        if _clm:
            out.append(f"in carry {self.fc}'s lineup {_clm}")
        if FV >= F9LI and n == self.cur_carry and n != self.fc:
            out.append("CURRENT CARRY")          # f9li：现任主C 原来没有任何标签，还会被写成 2nd carry / transitional
        if FV >= F9LI and n and n == self.ctx.get("maple"):
            if FV >= F9LW and self.maple_on_temp(n):
                # f9lw ㊱（2026-10-16）：枫叶舞的等级只在真正的主C 身上有价值；接班主C 在场就该卖宿主换绑，不看备份 / 引擎件
                out.append(f"MAPLE HOST on temp carry: re-bind to {self.fc} (sell this)" if FV >= F9ZB
                           else f"MAPLE HOST on TEMPORARY carry: maple levels wasted until re-bound to {self.fc} (sell this unit)")
            else:
                out.append("MAPLE HOST (re-binding the maple needs selling this unit)")
        olc = FV >= F9LW and self.other_lineup_carry(n)
        if n == self.fc:
            pop, cnt = carry_popularity().get(n, (0, 0))
            out.append(f"FUTURE CARRY (fit {fit(n)}/9: {fit_why(n)}; carry in {cnt} lineups)")
        elif olc:
            out.append(olc)
        elif n in self.carry_cands and self.fc and n != self.fc and not (FV >= F9LI and n == self.cur_carry):
            combo = self.second_carry_engine_tag(n) if FV >= "f9l" else ""
            out.append(combo or (SECOND_OUTPUT if FV >= F9LW and self.one_carry_locked() and not self.feeds_or_part(n)
                                 else f"2nd carry candidate (fit {fit(n)}/9)"))
        if FV >= "f9e" and n in self.extra_out and not (FV >= "f9j" and self.engine_role(n) in ("producer", "amplifier")) \
                and not (FV >= "f9k" and (self.engine_role(n) == "payoff" or self.route_member(n))) and not olc:
            out.append("EXTRA OUTPUT: " + SECOND_OUTPUT_WHY if FV >= F9LW and self.one_carry_locked() and not self.feeds_or_part(n)
                       else "EXTRA OUTPUT (not carry / future carry)")
        if FV >= "f9k" and self.route and is_hero(n) and not self.route_stale:
            if n == self.route.get("carry"):
                out.append("ROUTE carry")
            elif FV >= F9ZA and f9za_listed_takeover(n, self):
                out.append(LISTED_TAKEOVER_F9ZA)        # f9za-fix（协调方 10）：开着的主C、现任是名单外的人 → 不写 ROUTE optional sub
            elif FV >= F9W and self.route_not_wanted(n):
                out.append("not wanted on this route")
            elif self.route_member(n) or (FV >= F9W and self.route_extra_wanted(n)):
                if FV >= F9X and self.route_sub_only(n):
                    out.append(SUB_ONLY_F9X)          # f9x（54.26-12）
                elif FV >= F9LV and self.route_past_stage(n) and not (FV >= F9LW and self.feeds_or_part(n)):
                    # f9lv（第 647 局 R9：开局件弈星被标 ROUTE member，换人网络 4 道题都不卖、雅典娜压在手里）
                    out.append("ROUTE early-only (past its stage, not in final team)")
                else:
                    out.append(f9zb_off_route_tag(n, self) or "ROUTE member")      # f9zb（54.37-5）：<10% 且不在 lineup#1 → off-route
            elif self.route_core_engine(n):
                out.append("ROUTE core engine piece")     # f9y：引擎表 early_priority（日落海雅典娜）
            elif FV >= F9O and (self.anch["mates"].get(n) or 0) >= 0.8 and n != self.anchor:
                pass            # f9o（r9zr 老师多批：明世隐 `not in ROUTE` 同时 `in 99% of 百里守约 lineups`）：主C 阵容几乎必带的人不标路线外
            else:
                out.append("not in ROUTE")
        if FV >= "f9j":
            et = self.engine_tag(n)
            if et and not (FV >= "f9l" and n in self.carry_cands and self.fc and n != self.fc and self.second_carry_engine_tag(n)):
                out.append(et)          # f9l：2nd carry candidate 的引擎角色已合进上面那一条
        if n == self.main_tank:
            out.append("MAIN TANK")
        elif n in self.front_tanks:
            out.append("OFF-TANK")
        if is_transitional(n) and not _clm:     # f9m：主C 阵容里的人不再写"过渡主C"
            out.append("transitional carry: early game only" + (" (past it)" if self.rnd > 6 else ""))
        if nearest_buddy(n) and is_hero(n):
            adj = adjacent(self.cell(n), self.cell(self.fc)) if self.fc and n != self.fc else None
            out.append("buffs NEAREST ally → must stand next to carry"
                       + ("" if adj is None else (" (adjacent now)" if adj else " (NOT adjacent now)")))
        if "棋手经验" in skill(n):
            if FV >= F9LW:
                # f9lw ㊳（2026-10-16）：弈星是纯过渡件，经验贡献吃满也不值钱；在场期间棋手升 1 次级就回本，之后直接卖
                out.append(self.filler_text(n))
            elif FV >= F9LV and self.level:
                # f9lv：写剩余价值（第 647 局：5 级时弈星只剩"5→6 最后一次 +2"）
                left = meta().MAX_LORD_LEVEL - self.level
                out.append("lord-xp unit: lord MAX, no value left" if left <= 0
                           else f"lord-xp unit: {left} lord level-up(s) left")
            else:
                out.append("lord-xp unit")
        if (self.lv(n) or 0) >= 10 and n not in (self.fc,) and (fit(n) or 0) <= 4:
            out.append("stealable (≥10 lv, not carry)")
        if is_hero(n) and tier(n) <= 2 and self.rnd > 6 and n not in (self.fc, self.main_tank, *self.carry_cands) \
                and not is_transitional(n) and self.anch["n"] and self.anch["mates"].get(n, 0) < 0.1 \
                and not (FV >= "f8" and self.awk(n) == 0) \
                and not (FV >= F9LK and (n == self.cur_carry or self.route_member(n) or (self.engine and self.engine_role(n))
                                         or any(n in g for g in min_engine_groups(self.min_eng or {})))):
            # f8：觉醒过的不算后期低价值（老师 r8b：孙膑两个标签打架）；f9lk：现任主C / 路线成员 / 引擎零件 / 最低配套件也不算
            out.append("low value late (t%d, not in carry's lineups)" % tier(n))
        return out

    def awaken_str(self, n: str) -> str:
        k = self.awk(n)
        if k is None:
            return "awaken ?"
        if k == 0:
            return "AWAKENED"
        p = self.pool(n)
        s = f"awaken {3 - k}/3 (needs {k}" + (f", pool {p}" if p is not None else "") + ")"
        if p is not None and p < k:
            s += " IMPOSSIBLE (pool too small)"
        return s

    def unit_line(self, n: str) -> str:
        head = f"{n} t{tier(n)} {role(n) or '?'} {row(n) or '?'} {'melee' if melee(n) else 'ranged'} " \
               f"{fac_en(n)} lv {self.lv(n) or '?'}"
        if FV >= "f7":
            head += f" {_dmg(n) or '?'}"      # 物理 / 法术 / 混合：装备配置的依据（2026-09-25）
        parts = [head, self.awaken_str(n)]
        e = self.eq(n)
        parts.append(f"equip {e if e is not None else '?'}/3")
        if n != self.fc and fit(n) is not None and is_output(n):
            parts.append(f"fit {fit(n)}/9")
        parts += self.identity(n)
        if FV >= F9X:
            tg = f9x_trigger_fact(n, self)       # f9x（54.26-18）：条件触发型数来源、全队效果写 team-wide
            if tg:
                parts.append(tg)
        if self.anch["mates"].get(n) and n != self.anchor and not (FV >= F9ZB and any(x.startswith("off-route (") for x in parts)):
            parts.append(f"in {round(self.anch['mates'][n] * 100)}% of {self.anchor} lineups")     # f9zb：off-route 标签里已写占比
        if FV >= "f9d":
            syn = carry_synergy(n, self)     # f9d：场上每个人对主C 的作用
            if syn:
                parts.append(syn)
        return " | ".join(parts)


# ---------------------------------------------------------------- 候选事实
def _retrigger(n: str, B: "Board") -> tuple[int, str]:
    """f9lq：卡面"场上每有1种阵营，额外触发1次此效果"（八方来援）→ 效果触发 1 + 场上阵营数 次（不含无阵营；无阵营算不算待确认）。"""
    if FV < F9LQ or "场上每有1种阵营，额外触发1次" not in (skill(n) or ""):
        return 1, ""
    facs = sorted({fac(h) for h in B.heroes if fac(h) and fac(h) not in ("无阵营", "通用")})
    if FV >= F9ZB:
        return 1 + len(facs), f"x{1 + len(facs)}: {len(facs)} factions"
    return 1 + len(facs), f"x{1 + len(facs)}: 1 + {len(facs)} factions on board"


def _lvl_result(B: Board, who: str, n_add) -> str:
    """给 who 加 n 级：写成 L → L+n，并标出是否跨过技能质变等级。"""
    L = B.lv(who)
    if not isinstance(n_add, (int, float)):
        return f"{who} +{n_add} lv"
    if L is None:
        return f"{who} +{int(n_add)} lv (now lv ?)"
    s = f"{who} lv {L}→{L + int(n_add)}"
    for t in SKILL_THRESHOLDS:
        if L < t <= L + n_add:
            s += f" (crosses lv{t})" if FV >= F9ZB else f" (crosses lv{t} skill upgrade)"     # f9zb：写短（八方来援 47 token）
    return s


def _num(x):
    """数量：数字，或字符串开头的第一个整数（"+2（胜利 +1）" → 2）；说明性文字返回 None。"""
    if isinstance(x, (int, float)):
        return x
    m = re.match(r"\s*\+?(\d+)", str(x or ""))
    return int(m.group(1)) if m else None


def _faction_refs(n: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"【([^】]+)】", skill(n))))


TRIG_EN = {"登场": "on deploy", "整备": "each round start", "开团": "battle start", "牺牲": "on death", "凯旋": "on win",
           "败阵": "on loss", "退场": "when sold", "合成": "on merge", "常驻": "passive", "战斗中": "in battle",
           "召唤": "while on board", "使用": "on use", "回合开始": "each round start"}


FAST_ATTACK_CARRIES = ("马可波罗", "狄仁杰", "虞姬")   # 高攻速主C：马可波罗 = 2026-10-16 晚确认（高攻速主C 和少司缘非常配）；狄仁杰 / 虞姬 = 规范 42.6


def summon_fact_ctx(n: str, B: "Board") -> str:
    """summon_fact + 局面：按普攻次数触发的召唤物，现任主C 是高攻速时写出"触发得快"（马可波罗这种高攻速主C 和少司缘非常配）。"""
    t = summon_fact(n)
    if t and "attacks" in t:
        C = B.cur_carry or B.fc
        if C in FAST_ATTACK_CARRIES:
            t += f"; fast-attack {C} → often"
    return t


def summon_fact(n: str) -> str:
    """f9ly（第 673 局 R11：少司缘召唤因缘图腾 = 全队每 25 次普攻永久 +1 级，题面只写"support / ROUTE member"，被卖掉换朵莉亚；
    少司缘能给这个阵容提供等级，苏烈则完全没有意义）：召唤类英雄写出召出来的东西做什么；只给自己涨级的写 levels only itself。"""
    e = kb().get(n) or {}
    for g in e.get("gives") or []:
        if g.get("res") in ("图腾", "召唤物", "召唤") and isinstance(g.get("n"), str) and kb().get(g["n"]):
            sg = (kb().get(g["n"]).get("gives") or [{}])[0]
            to, per = str(sg.get("to") or ""), str(sg.get("per") or "")
            if sg.get("res") in ("等级", "临时等级") and re.search(r"所有|全体|全队", to):     # 短写：选项只有 ~44 token
                m = re.search(r"(\d+)\s*次普攻", per)
                return (f"summons {g['n']}: ALL heroes +{sg.get('n')} lv" + (" permanent" if "永久" in to else "")
                        + (f" per {m.group(1)} team attacks" if m else ""))
            eff = kb_effect(g["n"], 1).split(": ", 1)[-1]
            if eff:
                return f"summons {g['n']}: {eff[:40]}"
    gs = [g for g in e.get("gives") or [] if g.get("res") in ("等级", "临时等级")]
    if gs and all(str(g.get("to") or "") in ("自身", "") for g in gs):
        return "levels only itself (feeds nobody)"
    return ""


# f9t（2026-10-19 卡面 25 问确认 + 卡面复读）：机制库（card_kb.json）里过期的数值 / 对象 / 备注，
# 只在 f9t 起替换（card_kb.json 本身不动 → 旧版本题面逐字不变）
KB_F9T_GIVES = {
    # 2026-10-19：以卡面为准（kb 写错的数字）
    "紧急行动": [{"res": "棋手生命", "n": 10}, {"res": "装备", "n": "装备箱"}],
    "体面离场": [{"res": "能量", "n": "3/6/9/12（按被移除者觉醒进度 0/1/2/3）"}],
    "兵行诡招": [{"res": "临时等级", "n": "2（胜利 +1）", "to": "全体", "per": "每场"}, {"res": "棋手生命", "n": -10}],
    "有福同享": [{"res": "等级", "n": 4, "to": "场上与指定英雄同品阶的英雄"}],                       # 新卡面 +4（旧 +3）
    "韩信": [{"res": "等级", "n": "2（觉醒 4）", "to": "自身", "per": "每有英雄触发整备或开团"}],     # 新卡面 +2 / 觉醒 +4（旧 +1）
    "先知图腾": [{"res": "装备", "n": "玄微之种", "to": "最近 2 名己方英雄（不限阵营）", "per": "每场"}],  # 新卡面不限大河
    "传承": [{"res": "搬等级", "n": "等级平分给其余等级最低的 3 名，原英雄留 1 级"}],                  # 10-19 卡面确认第 19 问
    "夺取之术": [{"res": "搬等级", "n": "夺周围 1 格己方英雄至多 10 级（对方至少留 1 级）", "to": "指定 1 名英雄"}],  # 第 22 问
    "众星拱月": [{"res": "搬等级", "n": "夺其他每名英雄各至多 10 级（各至少留 1 级）", "to": "指定 1 名英雄"}],
}
KB_F9T_NOTE = {
    "传承": "原英雄留 1 级、不离场；等级给的是等级最低的 3 人，喂不到主C",
    "化铁为金": "变出的 5 阶觉醒英雄直接上场、触发登场（用户 2026-10-19）；临时等级战后清空",
    "鬼谷子": "先知图腾：开团时最近 2 名己方英雄（不限阵营）获得玄微之种（牺牲时给血最少的友军回血）",
}


def kb_entry(n: str) -> dict:
    """机制库条目；f9t 起按 KB_F9T_* 更正（返回副本，不改缓存）。"""
    e = kb().get(n) or {}
    if FV >= F9T and (n in KB_F9T_GIVES or n in KB_F9T_NOTE):
        e = dict(e)
        if n in KB_F9T_GIVES:
            e["gives"] = KB_F9T_GIVES[n]
        if n in KB_F9T_NOTE:
            e["note"] = KB_F9T_NOTE[n]
    return e


CAST_GROW_RE = re.compile(r"你的英雄(?:每)?释放\d*次?技能[^。]{0,12}(?:等级|临时等级)")


def cast_growers(B: "Board") -> list:
    """f9t：场上"你的英雄放技能就涨级"的人（孙膑每 3 次永久 +1、刘禅每次 +2 临时）。"""
    return [h for h in B.heroes if CAST_GROW_RE.search(skill(h) or "")]


def f9t_card_fact(n: str, B: "Board") -> str:
    """f9t（2026-10-19 卡面确认）：效果牌的真实机制，一句短英文，放选项最前。"""
    if n == "开局狂飙":
        g = cast_growers(B)
        return (f"every cast counts: {'/'.join(g[:2])} on board → target gains lv per cast" if g
                else "each cast counts as a skill cast (no 孙膑/刘禅 on board)")
    if n == "众星闪耀":
        return "all +10 if used this round, +15 next round (+5 per round held)"
    if n == "破境之证":
        return "then +10 PERMANENT/round, stays with hero (kept on awaken, lost if sold)"
    if n == "遗志":
        return "removed hero's items return to hand; no energy refund"
    if n == "化铁为金":
        return "becomes AWAKENED t5, deploys at once (deploy fires), keeps lv"
    if n == "夺取之术":
        return "steals ≤10 lv from a 1-cell ally (it keeps ≥1)"
    if n == "众星拱月":
        return "≤10 lv from EACH other hero (each keeps ≥1)"
    if n == "传承":
        return "levels go evenly to the 3 LOWEST-lv others (never the carry); source keeps lv 1"
    return ""


def kb_effect(n: str, limit: int = 2) -> str:
    """机制库里这张卡"什么时候、产出什么、给谁"（逐张人工理解的，比正则准）。"""
    e = kb_entry(n) if FV >= F9T else (kb().get(n) or {})
    gs = e.get("gives") or []
    if not gs or e.get("trig") in (None, "无"):
        return ""
    items = []
    for g in gs[:limit]:
        amt = g.get("n")
        items.append(f"{g.get('res')}{' ' + str(amt) if amt not in (None, '') else ''}"
                     + (f"→{g['to']}" if g.get("to") else "") + (f"/{g['per']}" if g.get("per") else ""))
    return f"{TRIG_EN.get(e.get('trig'), e.get('trig'))}: " + "; ".join(items)


def hero_facts(n: str, B: Board, kind: str = "") -> tuple[list[str], list[str]]:
    """英雄牌候选 → (选项里的关键结论, 局面详情)。"""
    key, det, rel = [], [], []          # rel：关系结论（效果空转/转永久/可夺取…），选项里排第 2
    # 合成 / 新单位
    inh = B.hand.count(n)                       # 手里的同名（合成材料）
    if FV >= "f9" and kind == "swap":
        inh = max(0, inh - 1)                   # f9：swap 题等着上场的那张就在手里，不算"另一张"（老师 r8f8："+1 copy" 实为它自己）
    extra_shop = max(0, B.shop_cnt.get(n, 1) - 1)   # 商店里另外几张同名（核查缺 3：dedupe 吞掉了"商店有 2 张"）
    if n in B.field:
        k = B.awk(n)
        if k is None:
            key.append(f"merges into board {n} (lv {B.lv(n) or '?'}, awaken ?)")
        elif k == 0:
            key.append(f"merges into board {n} (already AWAKENED)")
        elif k == 1:
            key.append(f"COMPLETES AWAKENING of board {n}: effects x2, returns to hand to re-deploy")
        else:
            key.append(f"merge: {n} awaken {3 - k}→{4 - k}/3 (then needs {k - 1})")
        more = inh + extra_shop
        if k and k > 1 and more:                # 核查缺 2：手里/商店的同名一起算
            reach = min(3, 4 - k + more)
            key.append(f"with {inh} in hand + {extra_shop} more in shop → awaken {reach}/3"
                       + (" (COMPLETE)" if reach == 3 else ""))
        p = B.pool(n)
        if k and p is not None and p < k:
            det.append(f"pool {p} < {k} needed: awakening impossible")
        idt = B.identity(n)
        if idt:
            key.append(idt[0].split(" (")[0])
    else:
        if inh:
            key.append(f"+{inh} copy already in hand (not deployed)")
        if extra_shop:
            det.append(f"{extra_shop + 1} copies in shop (buy both = new unit + 1 merge)")
        wait = sorted({h for h in B.hand_heroes if h != n})      # 手里等着上场的**其他**英雄（同名是合成材料，不占位）
        if FV >= F9LY and B.route and not B.route_stale:
            # f9ly：手里路线外的杂牌英雄该卖、不会上场，不算"等位"（同 slot_tag）
            _tm = set(_routes().final_team(B.route)) | set(_routes().members(B.route))
            wait = [h for h in wait if h in _tm or B.feeds_or_part(h)]
        no_room = (FV >= "f9i" and (kind == "shop" or B.ctx.get("_shop_q")) and B.free is not None
                   and len(wait) >= max(B.free, 0) and wait)
        _fill = []
        if no_room and FV >= F9LN and B.route and not B.route_stale:
            # f9ln（2026-10-13 第 617 局：早点上亚连就能赢——亚连是马可波罗路线配套，R6/R7/R8/R11 都在店里，
            # 题面一直写 NO ROOM → unused → sold，可场上站着牛魔 / 孙膑 / 张良这些路线外的补位件）：
            # 路线成员 + 场上有路线外的补位件 → 写"可以换下谁"，不写"白亏"
            _team = set(_routes().final_team(B.route)) | set(_routes().members(B.route))   # 含各阶段 main / sub（亚连是马可波罗路线的 sub）
            if n in _team:
                _fill = sorted(h for h in B.heroes if (h not in _team or (FV >= F9LV and B.route_past_stage(h)))
                               and h != B.fc and h != (B.route or {}).get("carry"))
                if FV >= F9LU:
                    _fill = [h for h in _fill if is_filler(h, B)]
                    _queue = [h for h in wait if h in _team]          # 手里已经在等这些杂牌位的路线成员
                    if len(_queue) >= len(_fill):
                        _fill = []
        if FV >= F9LW and lord_xp_unit(n):
            _fill = []          # 老师 r9zn_b/029：弈星（纯过渡件）不该被写成"能换掉杂牌的路线件"
        if FV >= F9ZA and _fill and not f9za_stage_ok(n, B):
            _fill = []          # f9za-fix（协调方 10；54.29-4，dup_002 571af6384705 武则天 2%）：计划外的阶段件不算"路线件"
        if no_room and _fill:
            key.insert(0, f"ROUTE unit: can replace filler {'/'.join(_fill[:2])}")     # 短：选项只有 ~44 token
        elif no_room:
            # f9i（第 13 局：满员时低置信买了 7 个英雄，全被当闲置手牌卖掉，3e 买 1e 卖）
            key.insert(0, f"NO ROOM ({len(wait)} waiting, {max(B.free, 0)} free): unused → sold 1e, −2e")   # 短：选项只有 44 token
        if no_room and B.free > 0:
            pass                                                # 空位已被手里的人占了，不再写 "new unit (N free slots)"
        elif B.free is None:
            key.append("new unit (free slots ?)")
        elif B.free > 0:
            if not (FV >= F9LM and kind not in ("swap", "carry")):      # f9lm：由 slot_tag（fills an empty slot …）代替
                key.append(f"new unit ({B.free} free slot{'s' if B.free > 1 else ''})")
        else:
            sell = [h for h in B.fodder()][:3]
            if FV >= "f9" and len(B.front_tanks) <= 2:   # f9：前排坦克 ≤2 时不推荐卖坦克（老师：和"前排至少留 2 个"冲突）
                sell = [h for h in sell if h not in B.front_tanks]
            if FV >= F9ZA:
                # f9za-fix（r9zzg_c 9367c47cf741："NO free slot (sell e.g. 明世隐)"）：与满员买人行同一可卖口径（f9za_sellable）
                _ok = set(f9za_sellable(B))
                sell = [h for h in sell if h in _ok]
            if FV >= F9LV and B.route and not B.route_stale and not _fill:
                # f9lv（r9zl 老师 ⑪：满员、手里没人在等时，路线件只写 "NO free slot (sell e.g. X)"，没有 can replace filler）
                _team = set(_routes().final_team(B.route)) | set(_routes().members(B.route))
                if n in _team and not (FV >= F9LW and lord_xp_unit(n)) and not (FV >= F9ZA and not f9za_stage_ok(n, B)):      # f9lw：弈星不是"能换掉杂牌的路线件"（r9zn_c/000）
                    _f2 = [h for h in sorted(B.heroes) if is_filler(h, B) and (h not in _team or B.route_past_stage(h))]
                    if _f2:
                        key.insert(0, f"ROUTE unit: can replace filler {'/'.join(_f2[:2])}")
            if FV >= "f9c" and not sell and not (FV >= F9ZA and B.route and not B.route_stale):      # f9za-fix：路线锁定、没有可卖的 → only core heroes
                # f9c：场上全是"核心"时也写出最弱的那个（2026-09-25 第 3 局：R11 攒到 33e，新英雄全标 "only core heroes"，
                # 学生读成"买不了"；场上其实有 1 级敖隐、2 级大司命可换 —— 输入缺信息把升级阵容的路堵死了）
                # 排除未来主C / 现任主C / 主T / 觉醒过的（老师 r9c：名单不全时"最弱"指到了 FUTURE CARRY、MAIN TANK、觉醒孙膑）
                pool = [h for h in B.heroes if h not in (B.fc, B.ctx.get("carry"), B.main_tank) and B.awk(h) != 0
                        and not (FV >= "f9k" and ((B.engine_started and B.engine_role(h)) or B.route_member(h)))
                        and not (FV >= F9ZA and f9za_engine_piece(h, B))]
                w = min(pool, key=lambda h: (tier(h), B.lv(h) if B.lv(h) is not None else 99, fit(h) or 0, h)) if pool else None   # 等级读不到的不当"最弱"
                key.append("NO free slot" + (f" (weakest on board: {w} t{tier(w)} lv{B.lv(w) or '?'})" if w
                                             else " (only carry / main tank / awakened units known on board)"))
            else:
                key.append("NO free slot" + (f" (sell e.g. {sell[0]})" if sell else " (only core heroes)"))
            if FV >= F9X and kind not in ("swap", "carry"):
                nd = f9x_no_deploy(n, B)            # f9x（54.26-25）：满员、不是路线最终阵容 / 核心引擎件
                if nd:
                    key.insert(0, nd)
            if len(sell) > 1:
                det.append("expendable units to sell: " + ", ".join(sell))
    if FV >= "f9j" and n not in B.field and B.engine and n in B.engine_missing():
        # f9j：目标引擎还缺的零件 —— 买它引擎就多一段（等级来自引擎，不来自单张加等级牌）
        r = B.engine_role(n)
        C = B.carry_set() if FV >= "f9l" else ""
        if FV >= "f9l" and r == "payoff" and C and n != C and is_output(n) and not B.route_member(n):
            # f9l：主C 已定时引擎落点 = 第二主C（第 35 节"不养"），吃主C 引擎红利的例外
            feeder = next((x for x in dict.fromkeys([B.cur_carry, B.fc]) if x and engine_fed_by(n, x)), "")
            txt = (f"ENGINE PAYOFF wanted (fed by {feeder}'s engine)" if feeder
                   else f"ENGINE PAYOFF = 2nd carry (carry {C} set: not wanted)")
        elif (FV >= F9LL and C and n != C and is_output(n) and not B.route_member(n) and not feeds_carry_role(n)
              and not engine_fed_by(n, C) and pure_second_c(n, B)):      # 和主C 有化学反应（小乔：每次登场 +1 级）的不算（r9za 第三轮老师）
            # f9ll（r9za 老师：李信 / 韩信这类只给自己涨级的纯输出被写成 ENGINE PART wanted，和"主C 定了不养第二主C"打架）
            txt = f"engine {r} of {B.engine['name']}, but a 2nd carry (carry {C} set)"
        elif FV >= F9LL:
            # f9ll（第 587 局：马可波罗只在登场时给一次核心等级，官方 25 套安琪拉阵容里只出现 1 次，却被写成 ENGINE PART wanted、
            # R25b 连买两张）：不写"wanted"这种结论，写事实 —— 引擎角色 + 在主C 官方阵容里的出现率
            Cx = C or B.cur_carry or B.fc
            pct = anchored(Cx)["mates"].get(n) if Cx and anchored(Cx)["n"] >= 3 and n not in B.nw else None   # f9x：路线不要的人不写占比
            txt = (f"engine {r} of {B.engine['name']}" + ("" if B.engine_started else " (engine not started)")
                   + (f"; in {round(pct * 100)}% of {Cx} lineups" if pct is not None else ""))
            if FV >= F9LZ and B.secondary_engine_piece(n):
                # f9lz（2026-10-17：亚瑟可作次选，总比牛魔一直在场上好）
                txt = f"SECONDARY: engine {r}, not core (below core, above off-route)"
        else:
            txt = (f"ENGINE PART wanted ({r} of {B.engine['name']}" + ("" if B.engine_started else ", would start the engine") + ")")
        key.insert(1 if (FV >= "f9k" and key and key[0].startswith("NO ROOM")) else 0, txt)
    if FV >= F9LJ and n not in B.field and is_hero(n):
        t = min_engine_tag(n, B, on_board=False)
        if t:
            key.insert(0, t)
    if FV >= F9LL and n not in B.field and is_hero(n) and re.search(r"退场：获得\d*张(弦音|妙音)", skill(n) or ""):
        # f9ll（r9za 老师多次：杨玉环"退场：获得1张弦音"，买来卖掉就能推主C 觉醒，题面没写这个结论）
        Cx = B.carry_set() or B.cur_carry or B.fc
        k = B.awk(Cx) if Cx else None
        if Cx and k and FV >= F9LQ and tier(Cx) > 3:
            key.insert(0, f"its 弦音 only works on t1-3 heroes: cannot awaken {Cx} (t{tier(Cx)})")
        elif Cx and k and FV >= F9O and (B.free is None or B.free <= 0):
            # f9o（第 716 局：满员时买了 4 次杨玉环、从手牌卖掉，一张弦音没拿到。日志统计：手牌卖 61 次 → 弦音 0；场上卖 25 次 → 21）
            key.insert(0, f"弦音 only if DEPLOYED then sold (selling from hand gives nothing); board full → needs a swap first")
        elif Cx and k:
            key.insert(0, ("deploy it then sell it" if FV >= F9O else "sell it at once")
                       + f" → 弦音 = {Cx} awaken {3 - k}→{4 - k}/3" + (" COMPLETES" if k == 1 else ""))
    if FV >= F9LL and n not in B.field and is_hero(n) and any(n in names for _t, names in duo_talents(B)):
        key.insert(0, "DUO talent partner: must be on board with its partner (one of them is the carry)")
    if FV >= F9LL and n not in B.field and is_hero(n) and is_output(n) and (fit(n) or 0) >= 5:
        gg = next((g for g, _w in held_god_gear(B) if god_gear_fits(g, n)), None)
        if gg:
            key.insert(0, f"CAN WEAR held t5 {gg} (carry can't use it; fit {fit(n)}/9, t{tier(n)})")
    if FV >= F9LK and n not in B.field and is_hero(n) and B.carry_ok(n) and ((fit(n) or 0) >= 6 or n in B.backup_carries):
        # f9lk（第 580 局 R6：主C 没定，商店里的百里玄策只被写成"引擎零件 / 对孙尚香没用"）：
        # 主C 没定（或现任不合格）时，合格的输出位写成主C 人选；路线失效时同体系备选路线的主C 单独标出
        if n == (B.route or {}).get("carry") and B.route_carry_missing:
            key.insert(0, f"ROUTE carry (route carry not owned yet; fit {fit(n)}/9, t{tier(n)})")
        elif n in B.backup_carries:
            key.insert(0, f"BACKUP ROUTE carry (same family as route; route carry not owned; fit {fit(n)}/9, t{tier(n)})")
        elif (not B.carry_set() and B.rnd >= 3 and (tier(n) or 0) >= 2
              and not (FV >= F9Y and n in BLOCKED_CARRIES_F9Y)          # f9y（54.27-17）：屏蔽路线的主C 不标主C 人选
              and not (B.fc_in_hand and B.carry_ok(B.fc_in_hand) and B.fc_in_hand != n)
              and not (B.route and not B.route_stale)):       # 路线还有效：目标是路线主C / 同体系备选，体系外的人不标（r9z 老师）      # 第 1~2 回合 / 1 阶：人人都是候选，标了没有区分度（r9z 老师）
            ph = B.cur_carry or B.fc
            key.insert(0, f"CARRY CANDIDATE (no settled carry" + (f": {ph} is a t{tier(ph)} placeholder" if ph and ph != n else "")
                       + f"; fit {fit(n)}/9, t{tier(n)})")
    if (FV >= F9LI and n not in B.field and is_hero(n) and is_output(n) and B.carry_set() and n != B.carry_set() and pure_second_c(n, B)
            and not (FV >= F9LK and B.engine and B.engine_role(n))
            and not (FV >= F9LL and any(god_gear_fits(g, n) for g, _w in held_god_gear(B)))   # 能穿神装的不算 C 位（r9za 老师）
            and not (FV >= F9LL and any(n in names for _t, names in duo_talents(B)))           # 共战天赋的搭档不算 C 位（r9za 第三轮老师）
            and not (FV >= F9LL and (feeds_carry_role(n) or any(g.get("res") in ("古币", "精致古币", "弦音", "妙音", "觉醒进度")
                                                                 for g in (kb().get(n) or {}).get("gives", []))
                                     or re.search(r"弦音|妙音|古币", skill_all(n) or "")))):
        # f9ll（r9za 第二轮老师：上官婉儿发古币 / 杨玉环给弦音也被标 C 位）：会给主C 资源的人不算 C 位
        # f9li：主C 已定时又一个输出位 —— 会和主C 抢装备 / 等级牌 / 枫叶舞（第 538 局拍卖敖隐：路线成员里的 C 位，题面只写了 ENGINE PART）
        key.insert(1 if key else 0, f"C-POSITION hero (carry {B.carry_set()} set: competes for gear/levels)")
    if FV >= "f9e" and n not in B.field and B.is_extra_output(n) and not (FV >= "f9j" and B.engine_role(n) in ("producer", "amplifier")) \
            and not (FV >= F9LY and B.route and not B.route_stale and n in _routes().final_team(B.route)):
        # f9ly（666 局 R12：李元芳是"枪有点饿了"最终阵容，却被标 EXTRA OUTPUT #2）：路线最终阵容成员不打第二输出标签
        # f9e：不会成为未来主C 的输出位 = 第 N 个输出、从 1 级起步（低等级的输出英雄毫无作用）
        key.insert(1 if key else 0, f"EXTRA OUTPUT #{len(B.outputs) + 1} (core: {', '.join(B.core_out)}), starts lv 1")
    olc = B.other_lineup_carry(n) if FV >= F9LW and n not in B.field else ""
    if olc:
        # f9lw ㊴（第 655 局 R6 买百里守约：题面写"fit 8/9 > carry 沈梦溪"）：路线已锁、路线主C 已拥有 → 别家主C 不在核心阵容，没有位置
        key.insert(0, olc)
    elif FV >= F9LW and n not in B.field and B.one_carry_locked() and not B.feeds_or_part(n)             and any(k.startswith(("C-POSITION", "EXTRA OUTPUT")) for k in key):
        key.insert(0, SECOND_OUTPUT)
    # 主C 强度
    s = fit(n)
    if s is not None and is_output(n):
        t = f"fit {s}/9"
        if B.fc and n != B.fc and not olc:
            t += " > carry " + B.fc if s > (fit(B.fc) or 0) else f" (carry {B.fc} {fit(B.fc)}/9)"
        if is_transitional(n):
            t += ", transitional"
        key.append(t)
        det.append(f"carry review: {fit_why(n, 40)}")
        pop, cnt = carry_popularity().get(n, (0, 0))
        if cnt and not olc:
            det.append(f"is the carry in {cnt} lineups")
    elif is_tanky(n):
        tk = len(B.front_tanks)
        key.append(("front tank" if is_front_tank(n) else "back-row tanky") + f" (front tanks now {tk})")
    elif is_support(n):
        key.append("support (no damage)")
    # 阵容
    if B.anchor and B.anchor != n and B.anch["n"]:
        p = B.anch["mates"].get(n, 0)
        if n not in B.nw:                   # f9x（54.26-2）：路线不要的人不写阵容占比（旁边已有 not wanted on this route）
            key.append(f"{round(p * 100)}% of {B.anchor} lineups")
    elif B.anchor == n:
        key.append(f"carry of {B.anch['n']} lineups (anchor)")
    for i, lu in enumerate(B.cover, 1):
        final, stage = lineup_members(lu)
        if FV >= "f9k" and B.route and lu is B.route:
            if B.route_stale:
                if lu.get("carry") == n:
                    det.append("carry of the stale ROUTE")      # f9l：失效路线不打标签，只在详情里提一句
                continue
            # f9k：锁定路线的成员进短文本（选项只有 44 token，det 常被截掉）
            if lu.get("carry") == n:
                key.append("ROUTE carry")
            elif FV >= F9ZA and f9za_listed_takeover(n, B):
                key.append(LISTED_TAKEOVER_F9ZA)                  # f9za-fix（协调方 10；dup_002 dd45f758a413）
            elif FV >= F9W and B.route_not_wanted(n):
                key.append("not wanted on this route")       # f9w：钟馗在狄仁杰路线里
            elif n in final or (FV >= F9W and B.route_extra_wanted(n)):
                key.append("off-route" if f9zb_off_route_tag(n, B) else "ROUTE final team")   # f9zb（54.37-5）：占比已在前面写了
            elif n in stage and FV >= F9X and B.route_sub_only(n) and not lord_xp_unit(n):
                key.append(SUB_ONLY_F9X)                           # f9x（54.26-12）：计划外的可选替补不写 ROUTE stage unit
            elif n in stage and FV >= F9ZA and not lord_xp_unit(n) and not f9za_stage_ok(n, B):
                key.append(SUB_ONLY_F9X)          # f9za-fix（协调方 10；54.29-4，dup_002 571af6384705 武则天 2%）：ROUTE stage unit 只给计划内 / 占比 ≥80% 的人
            elif n in stage:
                key.append("TRANSITIONAL filler" if FV >= F9LW and lord_xp_unit(n) else "ROUTE stage unit")
            elif B.route_core_engine(n):
                key.append("ROUTE core engine piece")       # f9y：引擎表 early_priority（日落海雅典娜）
            elif is_hero(n) and not (FV >= F9O and (B.anch["mates"].get(n) or 0) >= 0.8 and n != B.anchor):
                det.append("not in ROUTE")
            continue
        if lineup_carry(lu) == n:
            det.append(f"carry of lineup#{i}")
        elif n in final:
            aw = any(h.get("name") == n and h.get("is_awakened") for h in lu.get("heroes") or [])
            det.append(f"in lineup#{i} final team" + (" (awakened)" if aw else ""))
        elif n in stage:
            det.append(f"early/mid-stage unit of lineup#{i}")
    # 卡面效果 × 我方局面
    ke = kb_effect(n)
    if ke:
        det.insert(0, ke)
    sk = skill_all(n)
    for f in _faction_refs(n):
        cnt = faction_count(f, B, n, candidate=n not in B.field)
        (rel if cnt == 0 else det).append(f"refs 【{f}】: {cnt} on board" + (" (incl. itself)" if fac(n) == norm_fac(f) and cnt else "")
                                          + (" → effect idle" if cnt == 0 else ""))
    mt = re.search(r"触发(?:\d+名)?【([^】]+)】英雄的(登场|整备)效果", skill(n))
    if mt:                                    # 核查缺 4：曹操 × 孙尚香 = 每回合 +1 免费刷新
        f, k = norm_fac(mt.group(1)), mt.group(2)
        hs = [h for h in B.heroes if fac(h) == f and k in kw(h) and h != n]
        if FV >= F9ZB:
            # f9zb（check_dropped_lines：效果明细 42~56 token，整条放不进）：只写名字
            rel.append(f"re-triggers {'deploy' if k == '登场' else 'prepare'} of: " + (", ".join(hs[:3]) if hs else "none on board"))
        else:
            rel.append(f"re-triggers {'deploy' if k == '登场' else 'prepare'} of: "
                       + (", ".join(f"{h} ({kb_effect(h, 1).split(': ', 1)[-1]})" for h in hs[:2]) if hs else "none on board"))
    if "闪现英雄" in sk:
        bp = len([h for h in B.blink if h != n])
        (rel if bp == 0 else det).append(f"blink partners on board: {bp}")
    if FV >= F9V and "临时等级" in sk and "转化" in sk:
        # f9v（2026-10-19：明世隐也需要多来几张，每次登场临时等级转换数值越高；卡面答复 C23：合成 / 再打出都触发登场）
        if n in B.field:
            rel.append(f"copy merges onto board {n} (no slot): 登场 again → convertible +2")
        rel.append("temp lv→permanent; hero temp src: " + (", ".join(B.temp_src_hero[:3]) or "none"))
    elif "临时等级" in sk and "转化" in sk:
        rel.append("nearest ally's temp lv→permanent (temp src: " + (", ".join(B.temp_src) or "none") + ")")
        det.append("converts TEMP levels of the NEAREST linked hero to permanent (must stand next to the carry); "
                   "temp levels go to the hero who casts, so the carry itself must cast")
    elif nearest_buddy(n):
        det.append("acts on the NEAREST ally → must stand next to the carry")
    if "夺取" in kw(n):
        fod = [(h, B.lv(h)) for h in B.fodder() if (B.lv(h) or 0) >= 10]
        if FV >= F9ZB:
            (rel if fod else det).append("steals ≤10 lv from an adjacent ally; fodder ≥10: "      # f9zb：写短
                                         + (", ".join(f"{h} lv{lv}" for h, lv in fod[:2]) if fod else "none"))
        else:
            (rel if fod else det).append("steals ≤10 lv from a random adjacent ally when played; ≥10-lv fodder: "
                                         + (", ".join(f"{h} lv{lv}" for h, lv in fod) if fod else "none"))
    if "已觉醒" in sk or "觉醒英雄" in sk:
        det.append(f"awakened heroes on board: {len(B.awakened)}")
    if "古币" in sk and n not in B.coin_src:
        det.append(f"ancient-coin sources on board: {', '.join(B.coin_src) if B.coin_src else 'none'}")
    if any(g.get("res") == "召唤单位" for g in (kb().get(n) or {}).get("gives", [])) and (kb().get(n) or {}).get("trig") != "战斗中":
        det.append("summons units (do not use population)")
    if is_front_tank(n) and len(B.front_tanks) < 2:
        det.append(f"front tanks on board {len(B.front_tanks)} (<2)")
    if FV >= "f9d":
        syn = carry_synergy(n, B)      # f9d：和当前主C 的化学反应（2026-09-25），放在关系结论最前面
        if syn:
            rel = [syn] + rel
    if FV >= F9LM and kind not in ("swap", "carry"):
        # f9lm（2026-10-13）：候选路线核心卡（带已有张数）、临时主C 的同名 —— 事实，结论前置
        stg = slot_tag(n, B)              # 能不能直接上场（第 5~6 名复盘：带空位打是头号输因）
        if stg:
            key.insert(0, stg)
        ct = flex_core_tag(n, B)
        if ct:
            key.insert(1 if (FV >= F9LR and stg.startswith("FULL +")) else 0, ct)    # f9lr：满员还有人在等时"只能等"放最前
        if B.temp_carry and n == B.temp_carry:
            key.insert(0, "copy of TEMPORARY carry")
    if FV >= F9LX and olc:
        # f9lx（第 662 局 sell_hand 题：海月的 OTHER LINEUP 标签排第 3、被 44 token 挤掉）：放最前、写短
        key = [k for k in key if not k.startswith("OTHER LINEUP")]
        return ["OTHER LINEUP carry: no role (1 carry only)"] + key[:1] + rel + key[1:], det
    if FV >= F9O and kind not in ("swap", "carry"):
        # f9o（第 714 局 R7 N 选 1：雅典娜的 "engine amplifier of 日落海·核心; in 100% of 亚连 lineups" 排在 "FULL + would wait too" 和
        # "same build" 后面、被 44 token 挤掉，Laya 海诺 0.44 > 雅典娜 0.40；日落海里雅典娜的等级提升效果非常高，太乙不买也要先买雅典娜）
        # → 主C 引擎的产出方 / 放大器、路线核心这类身份放最前，能不能上场排第二
        pri = [k for k in key if k.startswith(("engine amplifier", "engine producer", "CORE"))]
        if pri:
            rest = [k for k in key if k not in pri]
            return pri + rest[:1] + rel + rest[1:], det
    return key[:1] + rel + key[1:], det


def rift_candidates(B: "Board") -> list:
    """f9ly：混沌裂隙的目标候选 —— 非主C、非贴身辅助 / 最近搭档、非共战搭档，按等级从高到低。"""
    duo = {h for _t, names in duo_talents(B) for h in names}
    hs = [h for h in B.heroes if h not in (B.fc, B.cur_carry) and not is_support(h) and not nearest_buddy(h) and h not in duo]
    return sorted(hs, key=lambda h: -(B.lv(h) or 0))


def _resolve_target(n: str, B: Board) -> tuple[str | None, str]:
    """这张牌作用于谁：审定用法 + 卡面限制 → (场上对象, 说明)。"""
    u = usage().get(n) or {}
    t = (u.get("target") or "").split(":")
    kind, arg = t[0], (t[1] if len(t) > 1 else "")
    if FV >= F9LY and n == "混沌裂隙":
        # f9ly（2026-10-16 午：放等级高的那个）：裂隙按被吞者等级 ×2 换人 → 放等级最高的非主C（非贴身辅助、非共战搭档）
        cand = rift_candidates(B)
        if cand:
            return cand[0], f"highest-level non-carry: rift lv {2 * (B.lv(cand[0]) or 0) or '?'}"
        return None, "no non-carry hero on board"
    if FV >= F9LX and n == "混沌裂隙":
        kind = "tank"          # 2026-10-16：一般放置主T（用牌表 target 保持 disposable，旧版本题面逐字不变）
    sk = skill(n)
    m = re.search(r"指定\d*名【([^】]+)】英雄", sk)
    if m:
        kind, arg = "faction", m.group(1)
    arg = norm_fac(arg)
    tl = re.search(r"(\d)阶及以下|品阶(?:不超过|≤)(\d)", sk)
    if FV >= "f7" and re.search(r"觉醒进度\s*\+", sk):
        # 觉醒进度 +1（弦音/妙音）：只能给**未觉醒**的英雄（已觉醒的游戏直接拒收）；卡面品阶范围照写
        mt = re.search(r"(\d)\s*-\s*(\d)\s*阶", sk)
        lo_t, hi_t = (int(mt.group(1)), int(mt.group(2))) if mt else (0, 9)
        hs = [h for h in B.heroes if B.awk(h) != 0 and lo_t <= tier(h) <= hi_t]
        if re.search(r"随机\d*名(?:\d-\d阶)?英雄觉醒进度", sk):
            # 有舍有得"回合开始时，随机1名英雄觉醒进度+1"：不能指定（g6 老师指出原来写成了 target X）
            return None, (f"RANDOM: 1 of {len(hs)} not-awakened heroes (" + ", ".join(sorted(hs)[:4]) + ")"
                          if hs else "RANDOM: no not-awakened hero on board")
        if not hs:
            return None, (f"needs a NOT-awakened t{lo_t}-{hi_t} hero: none" if mt else "needs a NOT-awakened hero: none")
        order = sorted(hs, key=lambda h: (h == B.fc, h == B.main_tank, -(B.awk(h) if B.awk(h) is not None else 3),
                                          fit(h) or 0, B.lv(h) or 0), reverse=True)
        if FV >= F9ZB:
            # f9zb（天籁琴音候选 48~50 token）：写短
            return order[0], "not awake: " + ", ".join(
                f"{h} {3 - B.awk(h) if B.awk(h) is not None else '?'}/3" for h in order[:3])
        return order[0], "not-awakened candidates: " + ", ".join(
            f"{h} (awaken {3 - B.awk(h) if B.awk(h) is not None else '?'}/3)" for h in order[:3])
    if FV >= "f7" and re.search(r"英雄变为|变为(?:同阵营)?(?:高\d阶|\d阶)?的?(?:随机)?英雄", sk)             and not re.search(r"敌方[^。；]{0,10}变为", sk):
        # 变身牌（昭君：霜寒进化/破冰裂变/寒冰王座/升华）：目标变成另一个英雄 —— 觉醒、枫叶舞绑定、身份都丢。
        # 2026-09-25 第四轮第 1 局：规则把霜寒进化用在已觉醒的主C 亚连身上。合适对象 = 没觉醒的过渡件/消耗件。
        ok = [h for h in B.fodder() if B.awk(h) != 0]
        never = list(dict.fromkeys([h for h in (B.fc, B.main_tank) if h] + [h for h in B.heroes if B.awk(h) == 0]))
        if not ok:
            return None, ("TRANSFORMS the target into another hero (loses awakening/maple/identity); "
                          "no expendable non-awakened unit on board")
        return ok[0], ("TRANSFORMS the target into another hero; fits expendable units: " + ", ".join(ok[:3])
                       + ("; NEVER on: " + ", ".join(never[:4]) if never else ""))
    if kind == "hold":
        return None, "situational (needs a judgment call)"
    if kind in ("none", ""):
        return None, "no target needed"
    if kind == "prev_carry" and FV >= F9LZ:
        # f9lz（第 678 局 R7：效果牌 3 选 1 拿了遗志，题面写 "target 沈梦溪 (carry)" —— 遗志会移除目标，用在现任主C 上 = 直接输；
        # 这次不是换主C 的套路）：只对换下来的老主C 用
        prev = (B.ctx.get("carry_prev") or "") if isinstance(B.ctx.get("carry_prev"), str) else ""
        if prev and prev in B.field and prev not in (B.cur_carry, B.fc):
            return prev, f"OLD carry {prev} (switched away): removes it, its levels go to a RANDOM hero"
        return None, "NO use now: only after a carry switch (on the current carry it removes the carry = lose)"
    if kind in ("carry", "carry_faction", "prev_carry"):
        if not B.fc and FV >= F9LI and B.cur_carry and B.cur_carry in B.field:
            # f9li：没有未来主C 时退回现任主C（第 539 局：孙膑是现任主C、已绑枫叶舞，题面却写 no carry-grade hero → no use now）
            if kind == "carry_faction" and arg and fac(B.cur_carry) != norm_fac(arg):
                return None, f"carry {B.cur_carry} is not 【{arg}】"
            return B.cur_carry, f"carry {B.cur_carry} (current, t{tier(B.cur_carry)} fit {fit(B.cur_carry)})"
        if not B.fc:
            return None, "no carry-grade hero on board"
        if kind == "carry_faction" and arg and fac(B.fc) != norm_fac(arg):
            return None, f"carry {B.fc} is not 【{arg}】"
        return B.fc, "carry"
    if kind == "tank":
        return (B.main_tank, "main tank") if B.main_tank else (B.fc, "no tank → carry") if B.fc else (None, "no tank on board")
    if kind == "faction":
        hs = [h for h in B.heroes if fac(h) == arg]
        if not hs:
            return None, f"needs a 【{arg}】 hero: 0 on board"
        best = B.fc if B.fc in hs else max(hs, key=lambda h: (fit(h) or 0, B.lv(h) or 0))
        return best, f"【{arg}】 heroes on board: {len(hs)}"
    if kind in ("low_tier", "low_tier_noncarry", "lowest_tier_high_level"):
        lim = int(arg) if arg.isdigit() else 3
        hs = [h for h in B.heroes if tier(h) <= lim and (kind == "low_tier" or h != B.fc)]
        if not hs:
            return None, f"needs a t≤{lim} hero: none"
        return max(hs, key=lambda h: B.lv(h) or 0), f"t≤{lim} heroes on board: {len(hs)}"
    if kind in ("ranged", "melee"):
        hs = [h for h in B.heroes if melee(h) == (kind == "melee")]
        best = B.fc if B.fc in hs else (hs[0] if hs else None)
        return best, f"{kind} heroes on board: {len(hs)}"
    if kind in ("entry_hero", "prep_hero"):
        k = "登场" if kind == "entry_hero" else "整备"
        hs = [h for h in B.heroes if k in kw(h)]
        if not hs:
            return None, f"needs a hero with a {'deploy' if k == '登场' else 'prepare'} effect: none on board"
        if FV >= F9ZB:
            # f9zb（唤醒 / 临阵磨枪 target 行 47~93 token）：效果明细见局面行，选项只写名字
            return hs[0], f"{'deploy' if k == '登场' else 'prepare'} heroes: " + ", ".join(hs[:3])
        return hs[0], (f"{'deploy' if k == '登场' else 'prepare'}-effect heroes: "
                       + ", ".join(f"{h} ({kb_effect(h, 1) or skill(h)[:20]})" for h in hs[:3]))
    if kind in ("disposable",):
        fod = B.fodder()
        return (fod[0], "expendable unit") if fod else (None, "no expendable unit (would remove a core hero)")
    if kind == "support":
        hs = [h for h in B.heroes if is_support(h)]
        return (hs[0], "support") if hs else (None, "no support on board")
    if kind == "hero":
        return (arg, "named hero") if arg in B.field else (None, f"needs {arg} on board")
    if kind == "equipped_noncarry":
        hs = [h for h in B.heroes if h != B.fc and (B.eq(h) or 0) > 0
              and not (FV >= "f9k" and h == B.ctx.get("carry"))]
        return (hs[0], "equipped non-carry") if hs else (None, "no equipped non-carry")
    if FV >= F9LI and kind == "level_dump":
        # f9li：传承"指定≥10级英雄，等级随机分给另外 3 名" —— 用在主C 身上等于把主C 拆了（原来落到兜底"给主C"）
        C = B.carry_set()
        hs = [h for h in B.heroes if h != C and (B.lv(h) or 0) >= 10]
        if not hs:
            return None, "needs a ≥10-level NON-carry hero to dump (using it on the carry scatters the carry's levels)"
        h = max(hs, key=lambda x: B.lv(x) or 0)
        if FV >= F9T:
            # f9t（2026-10-19 卡面确认第 19 问 + 新卡面）：平均分给其余等级最低的 3 名，原英雄留 1 级 → 主C 一般分不到
            low = sorted((x for x in B.heroes if x != h), key=lambda x: (B.lv(x) or 0, x))[:3]
            return h, (f"spreads {h}'s lv {B.lv(h)} over the 3 lowest: {'/'.join(low)}" + (" (incl. carry)" if C in low else " (not carry)"))
        return h, (f"dumps {h}'s lv {B.lv(h)} onto 3 random others"
                   + (f" (on carry {C} it would scatter lv {B.lv(C) or '?'})" if C else ""))
    if tl:
        pass
    if FV >= F9LI and kind == "awakened_faction":
        hs = [h for h in B.heroes if fac(h) == arg and B.awk(h) == 0]
        if not hs:
            return None, f"needs an AWAKENED 【{arg}】 hero: none"
        return max(hs, key=lambda h: B.lv(h) or 0), f"awakened 【{arg}】 heroes: {len(hs)}"
    if FV >= F9LI and kind not in ("carry", "", "tier_bulk"):
        return None, f"target ? (usage {kind})"      # f9li：没处理的用法不再默认给主C
    return (B.fc, "carry") if B.fc else (None, "target ?")


def can_wear(item: str, h: str) -> bool:
    """f9ll：卡面穿戴限制（限定远程 / 近战、可穿戴给【X】/ 指定【X】、仅X穿戴）。"""
    sk = skill(item)
    if ("限定远程" in sk and melee(h)) or ("限定近战" in sk and not melee(h)):
        return False
    m = re.search(r"(?:可穿戴给|指定\d*名)【([^】]+)】", sk)
    if m and fac(h) != norm_fac(m.group(1)):
        return False
    m = re.search(r"仅(\S{1,4}?)穿戴", sk)
    if m and m.group(1) not in ("近战", "远程") and m.group(1) != h and is_hero(m.group(1)):
        return False
    return True


def gear_restriction(item: str) -> str:
    sk = skill(item)
    if "限定近战" in sk:
        return "melee only"
    if "限定远程" in sk:
        return "ranged only"
    m = re.search(r"(?:可穿戴给|指定\d*名)【([^】]+)】", sk)
    return f"【{m.group(1)}】 only" if m else ""


def held_god_gear(B: "Board") -> list:
    """f9ll：手里的 5 阶装备，主C（已定的，没有就是现任 / 未来主C）穿不了或不对口（适配分 <5）→ [(装备, 原因)]。"""
    C = B.carry_set() or B.cur_carry or B.fc
    if not C:
        return []
    H = _ht()
    out = []
    for g in dict.fromkeys(B.hand):
        if cat(g) != "装备牌" or (c(g).get("quality") or 0) < 5:
            continue
        if not can_wear(g, C):
            out.append((g, f"{gear_restriction(g) or 'restricted'}; carry {C} can't wear"))
            continue
        # 只算"穿戴限制挡住主C"的（典型例子是青莲剑谱「限定近战」）；"不对口"按适配分判会把银莲庇佑这类
        # 防御 / 辅助型 5 阶装也算进来，给所有输出位打 CAN WEAR（r9za 抽查），所以不算
    return out


def god_gear_wearers(g: str, B: "Board", any_owned: bool = True, probe: bool = False) -> list:
    """能穿这件 5 阶装备、又很对口（适配分 ≥7）的输出位：默认在己方（场上 + 手里）里找；probe=True 时在全部英雄里找（判断这件是不是输出神装）。"""
    H = _ht()
    pool = list(dict.fromkeys(list(B.heroes) + list(B.hand_heroes))) if not probe else [h for h in carry_table() if is_hero(h)]
    out = []
    for h in pool:
        if not is_output(h) or not can_wear(g, h):
            continue
        sc = H.item_fit_score(g, h, (hero_roles().get(h) or {}).get("combat_role"))
        if sc is not None and sc >= 7:
            out.append(h)
            if probe:
                break
    return out


DUO_F9T_RE = re.compile(r"共战|二人在场|二人同时在场|同时在场")     # f9t（天赋审查 10-18：巨像写"二人在场"，没有"共战"二字）
DUO_ALIAS = {"婉儿": "上官婉儿"}                                       # f9t：舞力全开卡面写"婉儿"


def duo_pair(txt: str) -> list:
    """f9t：双人天赋点名的英雄（含简称）。"""
    names = [h for h in carry_table() if is_hero(h) and h in txt]
    for a, full in DUO_ALIAS.items():
        if a in txt and full not in names and is_hero(full):
            names.append(full)
    return names


def duo_talents(B) -> list:
    """f9ll：已选的"共战"天赋（崛起 / 双生突袭：点名两个英雄，两人同时在场才生效）→ [(天赋, [英雄…])]。只读天赋卡面。"""
    out = []
    for t in (getattr(B, "ctx", {}) or {}).get("talents") or []:
        txt = skill(t) or ""
        if "共战" not in txt and not (FV >= F9T and DUO_F9T_RE.search(txt)):
            continue
        names = duo_pair(txt) if FV >= F9T else [h for h in carry_table() if is_hero(h) and h in txt]
        if len(names) >= 2:
            out.append((t, names))
    return out


def f9z_tactic_zhulu_only(g: dict) -> str:
    """f9z：一条 gives 给的是卡面限【逐鹿】的战术牌（援护 / 推演 / 招募）→ 返回牌名；随机战术牌 / 特训 / 演练 → ""。"""
    if g.get("res") != "战术牌":
        return ""
    t = str(g.get("n") or "")
    nm = next((x for x in ("援护战术", "推演战术", "招募战术", "特训战术", "演练战术") if x in t), "")
    return nm if nm and "【逐鹿】" in (skill(nm) or "") else ""


def god_gear_fits(g: str, n: str) -> bool:
    H = _ht()
    if not can_wear(g, n):
        return False
    if FV >= F9Z and "固定0.4的攻击速度" in (skill(g) or "") and n not in SPEED_LOCK_CARRIES_F9Z:
        return False            # f9z（待办：机器头对沈梦溪写 FITS him）：机器头只给花木兰 / 嬴政（54.28-3，同 rules.SPEED_LOCK_CARRIES）
    sc = H.item_fit_score(g, n, (hero_roles().get(n) or {}).get("combat_role"))
    return sc is None or sc >= 7


def next_equip_target(item: str, B: Board, exclude: str) -> tuple[str | None, float | None]:
    """f9l：主C 装备满 3/3（游戏拒收）时这件装备的下一个对口的人：场上装备没满、hero_traits.item_fit_score ≥5，
    分高者优先，同分按 现任/未来主C → 等级 → 名字（逐字确定）。卡面"限定远程/近战"照查。"""
    H = _ht()
    sk = skill(item)
    best = []
    for h in B.heroes:
        if h == exclude or (B.eq(h) is not None and B.eq(h) >= 3):
            continue
        if ("限定远程" in sk and melee(h)) or ("限定近战" in sk and not melee(h)):
            continue
        if FV >= F9X and is_support(h) and h not in NO_GEAR_EXEMPT_F9X:
            continue            # f9x（54.26-3 / 54.24-2）：辅助（瑶除外）不给装备，主C 满装后给主坦 / 战士
        sc = H.item_fit_score(item, h, (hero_roles().get(h) or {}).get("combat_role"))
        if sc is None or sc < 5:
            continue
        if FV >= F9X:
            # f9x：主坦 / 前排坦克 / 战士优先（同类里再按适配分）
            tf = h == B.main_tank or is_front_tank(h) or role(h) in ("fighter", "tank", "frontline")
            best.append((not tf, -sc, h not in (B.cur_carry, B.fc), -(B.lv(h) or 0), h, sc))
            continue
        best.append((-sc, h not in (B.cur_carry, B.fc), -(B.lv(h) or 0), h, sc))
    if not best:
        return None, None
    best.sort()
    return best[0][-2], best[0][-1]


GIVE_N_RE = re.compile(r"获得(\d+)张([^\d张，。、（(「」]{2,8})")


def multi_give(n: str) -> tuple[int, str] | None:
    """f9l："获得 N 张 X"（N ≥2，X 在卡库里）的牌 —— 天籁琴音（3 张妙音）、古币多多（2 张古币）、战略中枢（2 张有福同享）…"""
    m = GIVE_N_RE.search(skill(n))
    if not m or int(m.group(1)) < 2:
        return None
    y = m.group(2)
    return (int(m.group(1)), y) if (c(y) and y != n) else None


def _awk_total(B: Board, t: str | None, k_copies: int) -> str:
    k = B.awk(t) if t else None
    if k is None:
        return f"awaken +{k_copies} (now ?)"
    if k == 0:
        return "already AWAKENED"
    a = 3 - k
    a2 = min(3, a + k_copies)
    s = f"awaken {a}→{a2}/3" + (" = COMPLETES (x2, back to hand)" if a2 == 3 else "")
    if k_copies > k:
        s += f", {k_copies - k} spare for another hero"
    return s


def multi_total(n: str, B: Board, t: str | None = None) -> str:
    """f9l：N 张 X 合计的效果（老师：天籁琴音选项只写了一张妙音的效果 awaken 0→1/3）。t 给定 = 全部用在 t 身上。"""
    mg = multi_give(n)
    if not mg:
        return ""
    k, y = mg
    if FV >= F9O:
        fs = _faction_cards().get(y)
        if fs and not any(fac(h) in fs for h in B.heroes):
            # f9o（r9zr 老师 test_014：古币多多写 `2× 古币: +4 lv total`，场上没有河洛英雄）
            return f"{k}× {y}: UNUSABLE now (only 【{'/'.join(sorted(fs))}】 heroes; none on board)"
    if FV >= F9Y and y in ("古币", "精致古币") and "古币机" in (B.ctx.get("talents") or []):
        # f9y（54.23-7）：选了古币机后古币都打进古币机（rules：古币机在场全给它），不直接给主C 涨级
        return f"{k}× {y}: go into 古币机 (it takes every 古币), not onto the carry"
    tgt = t or _resolve_target(y, B)[0]
    e = kb_entry(y) if FV >= F9T else (kb().get(y) or {})
    for g in e.get("gives") or []:
        if g.get("res") == "觉醒进度":
            return f"{k}× {y} all on {tgt or '?'}: {_awk_total(B, tgt, k * (_num(g.get('n')) or 1))}"
        if g.get("res") in ("等级", "临时等级") and _num(g.get("n")) is not None:
            if not tgt:
                return f"{k}× {y}: +{k * _num(g.get('n'))} {'TEMP ' if g.get('res') == '临时等级' else ''}lv total"
            return f"{k}× {y}: " + _lvl_result(B, tgt, k * _num(g.get("n"))) + (" TEMP" if g.get("res") == "临时等级" else "")
    eff = kb_effect(y, 1) or skill(y)[:30]
    return f"{k}× {y} ({eff})"


def card_facts(n: str, B: Board, desc: str = "") -> tuple[list[str], list[str]]:
    """效果牌 / 天赋牌 / 装备牌 / 棋手牌候选 → (关键结论, 详情)。"""
    key, det = [], []
    entry = kb_entry(n) if FV >= F9T else (kb().get(n) or {})
    if not c(n) and not entry:
        return ["unknown card"], [f"text: {desc[:60]}"] if desc else []
    if fleeting_now(n, B):
        key.append("FLEETING (gone at round end)")
    if FV >= F9Q and cat(n) == "装备牌":
        # f9q（2026-10-17 晚：特殊装备三选一要围绕主C 选，包括存在手牌里还没上场的主C）
        fh = getattr(B, "fc_in_hand", None)
        if fh and fh not in B.field:
            key.insert(0, f"carry {fh} (in hand, not deployed yet): " + ("FITS him" if god_gear_fits(n, fh) else "does NOT fit him"))
    if FV >= F9LI:
        nu = usable_now_fact(n, B)
        if FV >= F9W and not nu and B.ctx.get("_shop_q") and not any(str(w).startswith("faction_on_field") for w in (usage().get(n) or {}).get("when") or []):
            nu = f9w_card_faction_need(n, B)        # f9w：审定用法没写阵营条件的，按卡面"指定 / 所有【X】英雄"
        if nu and FV >= F9W:
            key.insert(0, nu)       # f9w（54.25-6：逍遥自在场上 0 个无阵营还买）：放最前，不被预算挤掉
        elif nu:
            key.append(nu)          # f9li：审定用法的"能用的前提"（第 568 局：跳级要场上有日落海，学生买了 4 张一张没用上）
    who, why = _resolve_target(n, B)
    full_from = None
    if FV >= "f9l" and cat(n) == "装备牌" and who and (B.eq(who) or 0) >= 3:
        # f9l（老师：主C 装备 3/3 满了，装备题仍写 target <carry>）：游戏拒收 → 改给下一个对口的人
        full_from = who
        who, _sc = next_equip_target(n, B, full_from)
        why = f"item fit {_sc:g}/10" if who else ""
    gives = entry.get("gives") or []
    results = []
    for g in gives:
        res, amt, to = g.get("res"), g.get("n"), g.get("to", "")
        num = _num(amt)
        if FV >= F9T and B.level:
            # f9t（天赋审查 10-18：四重奏"5×棋手等级"读成 5，4 级时实际 +20 写成 +5）
            _ml = re.search(r"(\d+)\s*[×x*]\s*棋手等级", str(amt) + ("" if num is not None else skill(n)))
            if _ml:
                num = int(_ml.group(1)) * B.level
        if res == "等级":
            tgt = who if (who and ("指定" in to or "主C" in to or not to)) else None
            if tgt:
                mul, why_m = _retrigger(n, B)
                if mul > 1 and num is not None:
                    results.append(_lvl_result(B, tgt, num * mul) + f" ({why_m})")
                else:
                    results.append(_lvl_result(B, tgt, num if num is not None else amt))
            else:
                results.append(f"+{amt} lv to {to or '?'}")
        elif res in ("临时等级", "等级") and "品阶" in str(amt) and who and re.search(r"品阶\s*[×x*]\s*(\d+)", str(amt)):
            k3 = int(re.search(r"品阶\s*[×x*]\s*(\d+)", str(amt)).group(1))
            results.append(f"{who} (t{tier(who)}) +{tier(who) * k3} {'TEMP ' if res == '临时等级' else ''}lv"
                           + (" (this battle only)" if res == "临时等级" else ""))
        elif res == "临时等级" or (res == "等级" and re.search(r"全体|每名|所有|己方英雄", to or "")):
            per = f"+{amt} {'TEMP ' if res == '临时等级' else ''}lv"
            if FV >= F9T and re.search(r"随机分配|随机分给|随机分到", skill(n)) and num is not None:
                # f9t（天赋审查 10-18：临时增幅"20 临时等级随机分给全队"写成每人 20、7 人 +140）
                k = max(1, len(B.heroes))
                results.append(per + f" in TOTAL, split randomly over {k} heroes (carry gets ~{num // k})"
                               + (" (this battle only)" if res == "临时等级" else ""))
                continue
            if re.search(r"全体|每名|所有|己方英雄", to or "") and num is not None:
                per += f" each × {len(B.heroes)} heroes = +{num * len(B.heroes)}"
            else:
                per += f" to {to or '?'}"
            results.append(per + (" (this battle only)" if res == "临时等级" else ""))
        elif res == "觉醒进度" and FV >= "f7" and why.startswith("RANDOM"):
            results.append("a random not-awakened hero awaken +1")
        elif res == "觉醒进度":
            tgt = who or B.fc
            k = B.awk(tgt) if tgt else None
            results.append(f"{tgt or '?'} awaken " + (f"{3 - k}→{4 - k}/3" if k else ("already done" if k == 0 else "+1 (now ?)")))
        elif res == "能量":
            mi = re.search(r"每(?:有)?\s*(\d+)\s*能量", str(amt) + skill(n))
            g0 = B.ctx.get("gold")
            if mi and isinstance(g0, int):
                results.append(f"interest: +1 per {mi.group(1)}e held → at {g0}e now +{g0 // int(mi.group(1))}/round"
                               + (f"; plus {num} now" if num else ""))
            elif str(amt)[:1] != "-":
                results.append(f"+{amt} energy")
        elif res == "人口":
            full = B.free == 0
            results.append(f"+{amt or 1} population" + (" (pop FULL now" + (f", {len(B.hand_heroes)} hero waiting)" if B.hand_heroes else ")") if full else ""))
        elif res == "免费刷新":
            results.append(f"+{amt} free refresh")
        elif res == "棋手经验":
            xp, need = B.ctx.get("xp"), B.ctx.get("xp_needed")
            s = f"+{amt} lord xp" + (f" (only when: {g['per']})" if g.get("per") else "")
            if isinstance(xp, int) and isinstance(need, int) and need and num:
                s += " → levels up" if xp + num >= need else f" (xp {xp + num}/{need})"
            results.append(s)
        elif res in ("英雄牌", "同名英雄牌", "效果牌", "天赋牌", "装备", "锦囊", "兑换券", "古币", "精致古币", "战术牌"):
            t_hero = None
            if res == "英雄牌":
                if "棋手等级+1" in skill(n) and B.level:
                    t_hero = min(5, B.level + 1)
                elif FV >= F9LX and re.search(r"高(\d)-(\d)阶的同阵营", skill(n)):
                    # f9lx（老师 r9zn_d/000、2026-10-16：前浪的荣光 = 对指定英雄用，随机得 1 张高 1~2 阶的同阵营英雄牌，定向搜阵营 + 品阶）
                    m2 = re.search(r"高(\d)-(\d)阶", skill(n))
                    results.append(f"use on a hero: gives 1 SAME-FACTION hero card {m2.group(1)}-{m2.group(2)} tiers above it (targeted search)")
                    continue
                elif FV >= "f7" and re.search(r"高(\d)阶", skill(n)):
                    # 限制性进化"移除 1~2 阶英雄，获得**高 2 阶**的随机英雄牌"：原来把"1~2阶"里的"2阶"当成结果（老师 r6 标注指出）
                    up = int(re.search(r"高(\d)阶", skill(n)).group(1))
                    results.append(f"gives a random hero card {up} tiers above the removed one")
                    continue
                elif FV >= F9T and re.search(r"(\d)\s*阶?\s*(?:-|~|或|至)\s*(\d)\s*阶", skill(n)):
                    # f9t（天赋审查 10-18：逐鹿精兵"4或5阶 3选1"写成 gives 4 random t4；双人游"2-3阶"写成 t3）
                    mr = re.search(r"(\d)\s*阶?\s*(?:-|~|或|至)\s*(\d)\s*阶", skill(n))
                    pk = re.search(r"(\d)\s*选\s*(\d)", skill(n))
                    fm2 = re.search(r"【([^】]+)】", skill(n))
                    cnt = re.search(r"(\d+)\s*张", skill(n))
                    results.append((f"pick {pk.group(2)} of {pk.group(1)}: " if pk else f"gives {cnt.group(1) if cnt else 1} ")
                                   + f"t{mr.group(1)}-t{mr.group(2)} " + (f"【{fm2.group(1)}】 " if fm2 else "") + "hero card(s)")
                    continue
                else:
                    mt = re.search(r"(\d)阶", n + skill(n))
                    t_hero = int(mt.group(1)) if mt else None
            if t_hero:
                own = sorted(h for h in B.owned if tier(h) == t_hero)     # 排序：输出逐字确定
                want = [h for h, p in B.anch["mates"].items() if tier(h) == t_hero and p >= 0.3]
                fm = re.search(r"【([^】]+)】英雄牌", skill(n))
                if FV >= "f7" and fm:
                    # 日落海新援"随机1张2-3阶【日落海】英雄牌"：只列该阵营的（g6 老师指出列了别的阵营）
                    ff = norm_fac(fm.group(1))
                    own = [h for h in own if fac(h) == ff]
                    want = [h for h in want if fac(h) == ff]
                if FV >= F9ZB:
                    # f9zb（兑换券 owned / wanted 48~50 token）：想要的分成"缺的"和"已有（同名）"，已有的只写个数
                    wm = [h for h in want if h not in B.owned]
                    results.append(f"gives {num or 1} random t{t_hero}"
                                   + (f"; missing: {' '.join(wm[:3])}" if wm else "; missing: none")
                                   + (f"; owned {len(own)}" + (f" ({len(want) - len(wm)} wanted)" if len(want) > len(wm) else "")
                                      if own else ""))
                else:
                    results.append(f"gives {num or 1} random t{t_hero} hero card(s)"
                                   + (f"; owned t{t_hero}: {', '.join(own)}" if own else "")
                                   + (f"; wanted t{t_hero}: {', '.join(want[:3])}" if want else ""))
            else:
                results.append(f"gives {amt or ''} {res}{'→' + to if to else ''}".replace("  ", " "))
        elif res == "搬等级":
            results.append(f"moves levels: {amt}"[:70])
        elif res:
            results.append(f"{res} {amt or ''}{' → ' + to if to else ''}".strip()[:50])
        if g.get("per") and res not in ("棋手经验",) and results:
            results[-1] += f" (per {g['per']})"
            if FV >= F9LQ and re.search(r"没有?打出英雄牌", str(g["per"]) + skill(n)):
                # f9lq（第 631 局 R4 整装待发 0.39 / 0.31 / 0.30 几乎在猜）：条件天赋写触发率。条件是"一回合没打出英雄牌"，
                # 我们最近 120 局实测：第 1~7 回合 12.6%、第 8~11 回合 25.6%、第 12 回合起 17.5% 的回合一张英雄牌都没打
                r0 = B.ctx.get("round") or 1
                exp = sum(NO_HERO_ROUND_RATE(r) for r in range(int(r0), 14))
                results[-1] += f"; condition met ~{NO_HERO_ROUND_RATE(r0):.0%}/round in our games → ~{exp:.1f} triggers by r13"
    if full_from:
        lab = "carry" if full_from in (B.fc, B.cur_carry) else full_from
        key.append(f"{lab} full (3/3) → target {who} ({why})" if who
                   else f"{lab} full (3/3) → no other hero fits (item fit ≥5)")
    elif who:
        key.append(f"target {who} ({why})")
    elif why not in ("no target needed",):
        blocked = why.startswith(("needs", "no ")) and "target needed" not in why
        if blocked and fleeting(n):
            key.append(why + " → UNUSABLE (fleeting: lost at round end)")
        elif blocked and cat(n) == "天赋牌" and (entry.get("trig") in ("使用", None)):
            key.append(why + " → card can be kept for later")     # 核查 M7：天赋是拿一张牌留着用，不是立即生效
        elif blocked:
            key.append(why + " → no use now (can be kept)")
        else:
            key.append(why)
    key += results[:2]
    det += results[2:]
    if FV >= "f9l" and multi_give(n):
        # f9l：N 张 X 的合计效果放最前（它自带目标名；原来排第一的 "target X (候选列表…)" 太长，auction 选项整条被 44 token 预算丢掉）
        key.insert(1 if key and key[0].startswith("FLEETING") else 0, multi_total(n, B))
    if FV >= F9LF:
        gf = grant_fact(n, B)               # f9lf：发英雄牌 → 卡池里有几个阵容要的人、期望拿到几个（结论前置）
        if gf:
            key.insert(1 if key and key[0].startswith("FLEETING") else 0, gf[0])
        if n in ALI_EXCLUSIVE_TALENTS:      # f9lf：阿离 6 级专属天赋写卡面效果
            key.insert(0, exclusive_talent_fact(n))
    if FV >= F9LY and is_hero(n) and summon_fact(n) and n not in (B.fc, B.cur_carry, (B.route or {}).get("carry")):
        key.insert(0, summon_fact_ctx(n, B))   # f9ly：召唤物做什么 / 只给自己涨级（第 673 局少司缘 vs 苏烈）
    if FV >= F9LZ and re.search(r"阿科米亚核心失去\s*\d+\s*级", skill(n) or ""):
        # f9lz（第 682 局：伞亚连日落海路线选了「能源置换」—— 每回合核心 −20 级换 10 能量；日落海路线不能选这个天赋，根本没法玩）
        rc = ((B.route or {}).get("carry") if B.route and not B.route_stale else "") or B.cur_carry or B.fc or ""
        if (B.engine and B.engine.get("key") == "sunset_core") or (rc and fac(rc) == "日落海"):
            key.insert(0, "AVOID on a 日落海 route: drains 阿科米亚核心 levels every round = the carry's level source")
    if FV >= F9LY and n == "珍品宝箱":
        # f9ly（2026-10-16 晚更正）：随机开 3~4 个选 1，出的是特殊装备（午间"大多基础装备"作废）
        key.insert(0, "real effect (user-confirmed): pick 1 of 3-4 SPECIAL equipment" if FV >= F9LZ
                   else "real effect (user-confirmed): pick 1 of 3-4 random items, MOSTLY BASIC equipment")   # f9ly 文本保持 R36 训练时原样
    if FV >= F9O and cat(n) != "装备牌" and re.search(r"获得[^。]{0,16}装备|装备[^。]{0,8}获得|装备(牌)?[^。]{0,4}选|宝箱", skill(n) or "")             and not (FV >= F9U and str(skill(n) or "").startswith("召唤")):
        # f9u（天赋复查 10-18 系统问题 5）：召唤类天赋（古币机：奖励里顺带"普通装备牌"）不是装备天赋，别只剩 "carry X n/3 items"
        # f9o（第 706 局 R13：远洋宝藏 = 5 件日落海专属装备随机 1 件，主C 亚连 2/3 件、日落海，Laya 买 0.31 < 刷新 0.33；
        # 选项没写主C 还空几格、专属装备主C 能不能穿）
        C = ((B.route or {}).get("carry") if B.route and not B.route_stale else "") or B.cur_carry or B.fc or ""
        if C:
            e = B.eq(C)
            txt = f"carry {C} {e if e is not None else '?'}/3 items" + (" FULL" if e == 3 else "")
            m = re.search(r"【([^】]+)】专属装备", skill(n) or "")
            if m:
                txt += ", can wear" if norm_fac(m.group(1)) == fac(C) else f", CANNOT wear ({m.group(1)} only)"
            key.insert(0, txt)
    # 费用（碰碰运气这类"随机消耗 1~5"以知识库为准）
    for g in entry.get("gives") or []:
        if g.get("res") == "能量" and str(g.get("n", "")).startswith("-"):
            cst = str(g['n']).replace('-', '') if FV >= F9LY else str(g['n']).lstrip('-')
            det.append(f"costs {cst} energy" + (" (random)" if FV >= F9LY and "~" in cst else "") + f" (have {B.ctx.get('gold', '?')})")
    m = re.search(r"消耗(\d+)(?:-(\d+))?能量", skill(n))
    if m and m.group(1) != "0":
        det.append(f"costs {m.group(0)[2:]} (have {B.ctx.get('gold', '?')})")
    # 触发 / 持续
    trig = entry.get("trig")
    if trig in ("回合开始", "整备"):
        left = max(0, 14 - B.rnd) if B.rnd else None
        det.append("repeats every round" + (f" (~{left} rounds left)" if left else ""))
    elif trig == "常驻":
        det.append("passive (always on)")
    elif trig == "开团":
        det.append("at every battle start")
    # 收益随局面放大
    sk = skill(n)
    if "觉醒英雄" in sk or "已觉醒" in sk:
        det.append(f"awakened heroes on board: {len(B.awakened)}")
    for f in _faction_refs(n):
        det.append(f"【{f}】 heroes on board: {faction_count(f, B)}")
    if FV >= "f7" and re.search(r"触发.{0,8}(整备|登场)效果", sk) and not re.search(r"【[^】]+】", sk):
        k = re.search(r"触发.{0,8}(整备|登场)效果", sk).group(1)
        hs = B.prepare_src if k == "整备" else B.deploy_src
        det.insert(0, f"{'prepare' if k == '整备' else 'deploy'}-effect heroes on board: "
                   + (", ".join(f"{h} ({kb_effect(h, 1).split(': ', 1)[-1]})" for h in hs[:4]) if hs else "none"))
    if "效果牌" in sk and ("每使用" in sk or "每打出" in sk):
        det.append(f"effect cards in hand: {len(B.effect_in_hand)}")
    # 装备
    wear = (who if full_from else B.fc) if FV >= "f9l" else B.fc     # f9l：主C 满了按下一个对口的人写
    if FV >= "f7" and cat(n) == "装备牌" and wear:
        # 2026-09-25：主C 只穿了 2 件；装备要结合这个英雄是物理还是法术伤害来定
        H = _ht()
        rec_b = _load_recipes().get(n)
        if rec_b:                                   # 结论前置：选项里放不下的会被截掉
            key.insert(1 if full_from else 0, f"forges for {wear} ({_dmg(wear)}): {H.carry_order(rec_b, wear, _onhit_lv())[0]}")
        elif n in H.EQUIP_CLASS:
            ok, _why = H.fits(n, wear, _onhit_lv())
            oh = f9x_onhit_text(n, wear, B) if FV >= F9X else ""
            # f9x（54.26-17）：普攻叠层装给技能型主C → 写原因（原来只有 does NOT fit）
            key.insert(1 if full_from else 0, oh or (("fits" if ok else "does NOT fit") + f" {wear} ({_dmg(wear)}; item {H.EQUIP_CLASS[n][0]})"))
        elif FV >= F9LN and H.item_dmg(n):
            # f9ln（2026-10-13 第 614 局 + 题面审计第 8 条）：基础装备 / EQUIP_CLASS 以外的成品（噬神之书、聚灵法杖、梦魇之牙…，
            # 含天赋送的装备）从没写伤害类型 → 学生看不出"法术装给物理主C"。只写事实：装备类型、目标类型、路线主C 类型
            it = H.item_dmg(n)
            who_t = _dmg(wear)
            s = f"{'magic' if it == 'mag' else 'physical'} item; {wear} {who_t}" + (" → MISMATCH" if who_t and who_t != it and who_t in ("mag", "phys") else "")
            rc = (B.route or {}).get("carry") if (B.route and not B.route_stale) else ""
            rc = rc if rc and rc != wear else ""
            if rc:
                rt = _dmg(rc)
                s += f"; route carry {rc} {rt}" + (" → MISMATCH" if rt and rt != it and rt in ("mag", "phys") else "")
            key.insert(1 if full_from else 0, s)
        if (FV >= F9X and "阿科米亚核心" in (skill(n) or "") and "参与" in (skill(n) or "") and fac(wear) != norm_fac("日落海")
                and not (B.ctx.get("_equip_pick") and wear == f9w_carry(B))):     # N 选 1 装备题已由 f9w_equip_pick_fact 写
            key.insert(1 if full_from else 0, f"sunset-only: {wear} not 【日落海】 -> no core levels")     # f9x（54.26-21）
    if FV >= F9U and cat(n) == "装备牌" and wear and "固定0.4的攻击速度" in (skill(n) or ""):
        # f9u（第 760 局 R5）：机器头把攻速锁在 0.4 —— 靠攻速成长 / 普攻输出的主C 被废掉；只读卡面 + 伤害来源
        _src = ((hero_roles().get(wear) or {}).get("damage_shape") or {}).get("source") or ""
        if FV >= F9V:
            # f9v（2026-10-19：机器头机制特殊，法力靠普攻回，不适合技能型主C，一般只有花木兰和嬴政作为主C 才会选）
            if wear in ("花木兰", "嬴政"):
                key.insert(0, f"speed lock 0.4: fits {wear} (花木兰/嬴政 only)")
            else:
                key.insert(0, f"speed lock 0.4: only 花木兰/嬴政; {wear} MISMATCH")
        elif re.search(r"攻击速度|攻速", skill(wear) or ""):
            key.insert(0, f"locks attack speed at 0.4: {wear} grows attack speed → wasted, MISMATCH")
        elif _src in ("auto", "attack"):
            key.insert(0, f"locks attack speed at 0.4: {wear} is a basic-attack carry → MISMATCH")
    if FV >= F9LN and cat(n) == "装备牌" and B.fc:
        # f9ln（2026-10-13：题面要写主C 是否满装，否则选了也白选；满了可以选辅助装备或 T 的装备）：
        # 只写事实 —— 主C 几件、满没满；满了再写主坦几件（谁还穿得下）
        ce = B.eq(B.fc) if B.fc in B.field else None
        s = f"carry {B.fc} equips {'?' if ce is None else ce}/3" + (" FULL" if ce is not None and ce >= 3 else "")
        if ce is not None and ce >= 3 and B.main_tank and B.main_tank in B.field:
            te = B.eq(B.main_tank)
            s += f"; main tank {B.main_tank} {'?' if te is None else te}/3"
        key.insert(0, s)
    if cat(n) == "装备牌":
        rec = B.anch["equips"].get(n) if B.anchor else None
        if rec:
            key.append(f"on {B.anchor} in {round(rec * 100)}% of lineups")
        if n in base_equips():
            det.append("base item: opens 4-choice forge on the wearer")
        if who:
            e = B.eq(who)
            det.append(f"{who} equips {e if e is not None else '?'}/3")
        if "限定远程" in sk or "限定近战" in sk:
            det.append("ranged heroes only" if "限定远程" in sk else "melee heroes only")
        if "无法拆卸" in sk:
            det.append("cannot be removed (lost if the wearer is sold)")
    if entry.get("cond"):
        key.append("only if: " + entry["cond"][:40])
    if FV >= F9LM and B.temp_carry and (who == B.temp_carry or (cat(n) == "装备牌" and wear == B.temp_carry)):
        # f9lm（2026-10-13：加等级牌和装备留给最终主C）：作用对象就是 1 阶临时主C
        key.insert(1 if key and key[0].startswith("FLEETING") else 0, "target is TEMPORARY carry")
    if FV >= F9LM and cat(n) == "天赋牌":
        # f9lm（2026-10-13 第 605 局「擂台猛虎」：天赋点名的派系我们没有就用不上）
        for f in _faction_refs(n)[:2]:
            if FV >= F9LU and ("装备" in f or "专属" in f):
                continue        # f9lu（r9zl 老师报）：「【日落海专属装备】」不是派系，原来写成 "our 【日落海专属装备】 heroes: board 0" → 误判为不生效
            key.insert(0, faction_own_line(f, B))
    # 阵容推荐
    rec = card_recommend().get(n)
    if rec and rec[0] >= 0.005:
        det.append(f"recommended by {round(rec[0] * 100, 1)}% of lineups (Ali lineups: {rec[1]})")
    note = (entry.get("note") or "").replace("\n", " ")
    me = re.search(r"给装备\s*(\S+?)（效果见装备条目）", note) if FV >= "f7" else None
    if me and c(me.group(1)):
        # "给装备 末世（效果见装备条目）"：题面里没有装备条目，展开成卡面原文（g6 老师报看不到效果）
        det.append(f"gives item {me.group(1)}: {skill(me.group(1))[:70]}")
    elif note:
        det.append("note: " + note[:70])
    if FV >= F9T:
        ft = f9t_card_fact(n, B)
        if ft:
            if n == "众星拱月" and who:
                ft = f"+{sum(min(10, max(0, (B.lv(h) or 0) - 1)) for h in B.heroes if h != who)} to {who}: " + ft
                key = [k for k in key if not k.startswith(f"target {who}")]       # 目标名已在这句里（省预算）
            key.insert(1 if key and key[0].startswith(("FLEETING", "target ")) else 0, ft)
        mg = multi_give(n)
        if mg and f9t_card_fact(mg[1], B):
            # 狂飙三倍速 = 3 张开局狂飙：合计行原来是机制库中文原句，换成更正后的短句
            mt = multi_total(n, B)
            key = [f"{mg[0]}× {mg[1]}: {f9t_card_fact(mg[1], B)}" if k == mt else k for k in key]
    if FV >= F9LZ:
        # f9lz：次选标签放最前（选项只有 ~44 token，排第 3 时被"同一套打法"长句挤掉）
        sec = [k for k in key if k.startswith("SECONDARY")]
        key = sec + [k for k in key if not k.startswith("SECONDARY")]
    return key, det


def _unit_brief(n: str) -> str:
    return f"{n} ({'melee' if melee(n) else 'ranged'} {role(n) or '?'} {row(n) or '?'})"


def lord_boost_facts(n: str, B: Board, desc: str = "") -> tuple[list[str], list[str]]:
    """枫叶舞强化：价值取决于绑定对象（核查缺 9：原来几乎零局面信息）。"""
    key, det = card_facts(n, B, desc)
    bound = B.ctx.get("maple")
    if FV >= F9LI:
        # f9li（第 564 局）：宿主刚卖掉 / 没绑时原来只写 "maple bound to ?"，红叶染锋（宿主以后离场才兑现）被 0.55 选中
        C = B.carry_set() or B.cur_carry or B.fc
        if bound and is_hero(bound) and bound in B.field:
            head = f"maple host {bound if FV >= F9ZB else _unit_brief(bound)}" + (" = CARRY" if bound == C else f"; carry {C}" if C else "")
        else:
            head = "maple UNBOUND now" + (f"; next host on board = carry {C}" if C and C in B.field else "")
        if re.search(r"离场后.*下一个枫叶舞英雄", skill(n) or ""):
            stays = (bound == C) or not (bound and is_hero(bound) and bound in B.field)
            if FV >= F9LW and bound and bound in B.field and (bound == B.temp_carry or B.maple_on_temp(bound)
                                                             or (tier(bound) or 9) <= 1):
                stays = False          # 老师 r9zn_b/014、017，r9zn_test/013：临时宿主之后一定换绑 → 红叶染锋会兑现
            nxt = f9x_maple_successor(bound, B) if FV >= F9X else None
            if nxt is not None:
                # f9x（54.26-1，10-19 晚：临时主C 等级不会高，换 C 之后吃不了多少等级）：宿主是临时主C → 只带过去一半的低等级
                lv = B.lv(bound)
                if FV >= F9ZB:
                    head += (f"; temp host lv {lv if lv is not None else '?'}: carries ~{lv // 2 if isinstance(lv, int) else '?'} lv"
                             " -> little value" + (f" (next {nxt})" if nxt else ""))
                else:
                    head += (f"; host is a temp carry (lv {lv if lv is not None else '?'}): carries over only "
                             f"~{lv // 2 if isinstance(lv, int) else '?'} lv -> little value" + (f" (next carry {nxt})" if nxt else ""))
            else:
                head += ("; pays ONLY if the host leaves later — host is/will be the carry (stays)" if stays and C
                         else "; pays when the current host leaves (host is not the carry)")
        key.insert(0, head)
        if FV >= F9W:
            cc = f9w_cc_fact(n, B)          # f9w（54.25-5）：主C 没免控装 → 枫影无拘补
            if cc:
                key.insert(0, cc)
        return _lord_boost_mech(key, det)
    head = f"maple bound to {_unit_brief(bound) if bound and is_hero(bound) else (bound or '?')}"
    if FV >= F9Y and bound and bound not in B.field and bound not in B.hand:
        head += f" (stale: {bound} not on board)"      # f9y：宿主已离场，记账陈旧
    if bound and bound == B.fc:
        head += " = FUTURE CARRY"
    elif B.fc:
        head += f"; future carry {_unit_brief(B.fc)}"
    key.insert(0, head)
    if FV >= "f7":
        # 老师 r6 标注：枫叶舞强化题"没说转移本身怎么触发、护盾多大"。机制写进详情（机制库原文，已确认过的理解）
        e = kb().get("枫叶舞") or {}
        mech = "; ".join(f"{g.get('res')} {g.get('n') or ''}".strip() for g in e.get("gives") or [])
        det.insert(0, "maple dance: bound hero gets a shield; " + (mech or "")
                   + ("; " + (e.get("note") or "")[:120] if e.get("note") else ""))
    return key, det


def f9x_maple_successor(bound: str, B: "Board"):
    """f9x（54.26-1）：枫叶舞宿主是不是临时主C（1 阶占位 / 接班主C 已在场或在手 / 不是已锁路线的主C 而路线主C 已到手）。
    不是 → None；是 → 接班主C 名（不知道是谁 → ""）。"""
    if not (bound and is_hero(bound) and bound in B.field):
        return None
    rc = (B.route or {}).get("carry") if B.route and not B.route_stale else ""
    if rc and bound == rc:
        return None
    if FV >= F9ZB and f9zb_committed_offlist(B, bound):
        return None                     # f9zb（54.37-1）：已认的名单外主C 不算临时宿主
    succ = [x for x in (rc if rc in B.owned else "", B.fc, B.fc_in_hand) if x and x != bound]
    temp = (bound == B.temp_carry or (tier(bound) or 9) <= 1 or B.maple_on_temp(bound)
            or bool(succ and (succ[0] in B.heroes or succ[0] in B.hand) and (B.lv(bound) or 0) < 20))   # 已养到 20 级的不算"低等级临时"
    return (succ[0] if succ else "") if temp else None


def _lord_boost_mech(key: list, det: list) -> tuple[list[str], list[str]]:
    e = kb().get("枫叶舞") or {}
    mech = "; ".join(f"{g.get('res')} {g.get('n') or ''}".strip() for g in e.get("gives") or [])
    det.insert(0, "maple dance: bound hero gets a shield; " + (mech or "")
               + ("; " + (e.get("note") or "")[:120] if e.get("note") else ""))
    return key, det


def _norm_txt(t: str) -> str:
    return re.sub(r"[\s，。、：:；;（）()【】\d+\-×%]", "", t or "")


@lru_cache(maxsize=4096)
def resolve(n: str, desc: str = "") -> tuple[str, str]:
    """卡名 vs 屏幕描述交叉校验（核查 E6/E7：OCR 把「升格」读成「破格」，题面照着另一张卡描述）。
    返回 (用来查事实的卡名, 说明)。描述和卡面重合太低就按描述在卡库里找；找不到就返回 ("", 说明) —— 只用屏幕描述。"""
    import difflib
    d = _norm_txt(desc)
    if not d or len(d) < 6:
        return n, ""
    if c(n):
        r = difflib.SequenceMatcher(None, _norm_txt(skill_base(n)), d).ratio()
        if r >= 0.5:
            return n, ""
    best, br = "", 0.0
    for k, v in cards().items():
        if v.get("category") not in ("天赋牌", "效果牌", "棋手牌", "棋手技能牌", "装备牌"):
            continue
        r = difflib.SequenceMatcher(None, _norm_txt(v.get("skill") or ""), d).ratio()
        if r > br:
            best, br = k, r
    if br >= 0.6:
        return best, f"name read as {n}, text matches {best}"
    if FV >= "f7" and c(n):
        # 多行卡面描述常被 OCR 打乱行序（"速+60%，值上限-20%。暴击率+40%，生命"），按顺序比会误判。
        # 没有别的卡按顺序明显更像（上面 br<0.6）、卡名本身在库里、屏幕上的字 ≥75% 出现在它的卡面原文里 → 认。
        # （2026-09-25 老师 g5 报"小身体大能量/大明星 认不出"）
        sk = _norm_txt(skill_base(n))
        if sk and sum(ch in sk for ch in d) / len(d) >= 0.75:
            return n, ""
    return "", f"name {n} does not match the on-screen text; card unknown"


GIVE_EQUIPS_RE = re.compile(r"获得\d张([^\d张与，。]{2,6})与\d张([^\d张与，。]{2,6})")


def _unknown_card_facts(n: str, desc: str, B: Board) -> tuple[list[str], list[str]]:
    """f7：卡库里查不到（或名字/描述都对不上）的卡。能从屏幕描述里读出的事实尽量读出来。
    - 2026-09-24 游戏更新后的「进攻型/均衡型/防御型武装」天赋：卡库没有，描述是"获得1张X与1张Y"（两件基础装备）；
    - 阿离满级专属天赋（落英汲取/枫之洪流/翩跹馈赠…）卡库也没有。"""
    known = bool(c(n))
    key = ["card not in database" if not known else "card unknown (name/text mismatch)"]
    det = [f"on-screen text: {desc[:80]}"]
    m = GIVE_EQUIPS_RE.search(desc)
    base = base_equips()
    if m:
        items = [x for x in m.groups() if x in base]
        if items:
            key = [f"gives base items {' + '.join(items)}"]
            wear, pre = B.fc, ""
            if FV >= "f9l" and B.fc and (B.eq(B.fc) or 0) >= 3:
                wear = next_equip_target(items[0], B, B.fc)[0]       # f9l：主C 满了 → 下一个对口的人
                pre = "carry full (3/3) → " if wear else "carry full (3/3) → no other hero fits"
                if not wear:
                    key.insert(0, pre)
            if wear:
                H = _ht()
                rec = _load_recipes()
                forged = []
                for x in items:
                    if rec.get(x):
                        best = H.carry_order(rec[x], wear, _onhit_lv())[0]
                        forged.append(best if H.fits(best, wear, _onhit_lv())[0] else f"{best} (tank item)")
                if forged:
                    key.insert(0, f"{pre}forges for {wear} ({_dmg(wear)}): {', '.join(forged)}")
                e = B.eq(wear)
                det.append(f"{wear} equips {e if e is not None else '?'}/3")
    elif "枫叶舞" in desc or "转移" in desc:
        key.append("lord Ali talent (maple dance)")
    return key, det


def facts_for(n: str, B: Board, desc: str = "", kind: str = "") -> tuple[list[str], list[str]]:
    if FV >= F9LF and not is_hero(n):
        ex = exclusive_talent_name(n, desc)        # f9lf：OCR 带杂字的「拉落英汲取」也认成专属天赋
        if ex:
            return card_facts(ex, B, "")
    if not is_hero(n) and desc:
        rn, note = resolve(n, desc)
        if not rn and FV >= "f7":
            return _unknown_card_facts(n, desc, B)
        if not rn:
            return ["card unknown (name/text mismatch)"], [f"on-screen text: {desc[:80]}"]
        if rn != n:
            key, det = facts_for(rn, B, "", kind)
            return key, [note] + det
    if kind == "lord_boost":
        return lord_boost_facts(n, B, desc)
    if is_hero(n):
        return hero_facts(n, B, kind) if FV >= "f9" else hero_facts(n, B)
    return card_facts(n, B, desc)


# ---------------------------------------------------------------- 商店非购买选项
def wanted(B: Board) -> list[tuple[str, int | None]]:
    """"想要的牌"（核查 M3：原来漏了锚点本人、手里的更强主C、覆盖阵容主C，543 题写成"0%"）：
    未来主C / 主T / 锚点 / 手里更强的主C / 主C 候选 / 覆盖阵容的主C（没觉醒的）→ 以主C 为锚的阵容里没拥有的高频队友。"""
    out: list = []
    seen: set = set()

    def add(n, rem=None):
        if n and n not in seen and is_hero(n) and B.awk(n) != 0:
            seen.add(n)
            out.append((n, rem if rem is not None else B.pool(n)))
    if FV >= F9LM and B.temp_carry:
        seen.add(B.temp_carry)          # f9lm：临时主C 的同名不算"想要的牌"（不为它凑觉醒）
    seen.update(B.nw)                   # f9x（54.26-2）：路线不要的人不算想要的牌（f9x 以前 B.nw 恒空）
    if FV >= F9LW and B.one_carry_locked():
        # f9lw ㊴（第 655 局 R6：刷新一行把百里守约列成"想要的牌"）：路线锁定且路线主C 已拥有 → 别家主C 不算想要的牌
        seen.update(x for x in (*B.carry_cands, *[lineup_carry(lu) for lu in B.cover]) if x and B.other_lineup_carry(x))
    for n in (B.fc, B.main_tank, B.anchor, B.fc_in_hand, *B.carry_cands, *[lineup_carry(lu) for lu in B.cover]):
        add(n)
    if FV >= F9LN:
        # f9ln（2026-10-13 第 614 局：买的英雄都配不上马可波罗）：自定义路线没有官方阵容 → anch["mates"] 为空，
        # 路线最低配套（米莱狄 / 朵莉亚）从没进"想要的牌"，刷新一行只写马可波罗 / 项羽，学生看不出刷新是为了凑配套
        for g in min_engine_groups(getattr(B, "min_eng", None) or {}):
            if not any(x in B.field or x in B.hand_heroes for x in g):
                for x in g:
                    add(x)
    for n in B.heroes:
        if B.anch["mates"].get(n, 0) >= 0.3:
            add(n)
    for n, p in B.anch["mates"].items():
        if len(out) >= 7:
            break
        if n not in B.owned and p >= 0.3:
            add(n)
    return out[:7]


def _rcost(B: "Board") -> int:
    """商店刷新费：f9v 起按读到的（读不到按 1；2026-10-19 核实实际 1 能量，239 次扣 1 —— 原来写死 2）；旧版本照旧 2（逐字不变）。"""
    if FV < F9V:
        return 2
    rc = B.ctx.get("refresh_cost")
    return rc if isinstance(rc, int) and rc > 0 else 1


def refresh_parts(B: Board) -> tuple[str, list[str]]:
    """(选项开头, 按优先级排的结论)。每张"想要的牌"刷一次至少看到 1 张的概率（品阶概率 × 牌池剩余 / 该品阶总张数，4 个槽位）。"""
    m = meta()
    free = B.ctx.get("free_refresh") or 0
    cost = f"free, {free} left" if free else f"{_rcost(B)}e"
    if not B.level:
        return f"refresh ({cost})", ["odds unknown (lord level ?)"]
    ps = []
    for n, rem in wanted(B):
        q = tier(n)
        if not q:
            continue
        if not m.can_appear(q, B.level):
            ps.append((n, 0.0, f" (t{q} not in shop at lv {B.level})"))
        elif m.tier_odds(q, B.level) <= 0:
            ps.append((n, None, f" (t{q} odds unmeasured at lv {B.level})"))      # 核查 M9：6 级表只实测了 t3~t5
        else:
            ps.append((n, m.shop_hit_prob(q, B.level, rem), ""))
    known = [p for _, p, _ in ps if p is not None]
    left = f", {B.ctx['gold'] - _rcost(B)}e left" if (not free and isinstance(B.ctx.get("gold"), int)) else ""
    if FV >= F9LF:
        left += refresh_afford_note(B)      # f9lf：刷完买不买得起英雄，直接写结论
    lvl_note = []
    g, lc = B.ctx.get("gold"), B.ctx.get("level_cost")
    if (FV >= "f9l" and B.free == 0 and B.level < m.MAX_LORD_LEVEL and isinstance(g, int) and isinstance(lc, int) and lc > 0
            and B.rnd):
        # f9l（老师：满员商店题要知道"刷完还攒不攒得够下回合升级"）：刷新后剩余 + 下回合收入 vs 现在的升级费
        after = g - (0 if free else _rcost(B))
        nxt = after + m.energy_income(B.rnd + 1)
        lc_next = max(0, lc - 1) if FV >= F9LU else lc        # f9lu：回合结束 +1 经验，下回合升级费少 1（和局面行同一口径）
        if FV >= F9LY:
            # f9ly（r9zo 11 道商店题前后矛盾：局面行用 lord_xp_gain，拍卖前一回合 +3 经验；这里写死 −1）→ 同一口径
            lc_next = max(0, lc - m.lord_xp_gain(B.rnd))
        lvl_note = [f"then {after}e +{m.energy_income(B.rnd + 1)} next = {nxt}e vs level-up {lc_next}e: "
                    + ("still levels next round" if nxt >= lc_next else "NOT enough to level next round")]
        if FV >= F9ZB:
            # f9zb（第 881/882 局 check_dropped_lines：这行 42~57 token，单条就放不进选项）
            lvl_note = [f"then {nxt}e {'≥' if nxt >= lc_next else '<'} lvl-up {lc_next}e next rd"]
    if FV >= F9LR and B.free is not None and B.free <= 0:
        # f9lr（第 632 局 R12、R14~R18：满员、手里压着 3~5 个英雄，还付费刷了约 30 次找新人，刷到的也只能在手里等）
        w = len({h for h in B.hand_heroes if h not in B.field})
        if w:
            new_only = bool(ps) and all(n not in B.field for n, _, _ in ps)
            core_miss, filler = [], []
            if FV >= F9S and getattr(B, "route", None):
                # f9s（第 735 局 R9：eco 判"停 4 级多刷"、缺明世隐，商店题写 "only merges help" → 免费刷完就停、留 10e；明世隐拖到 R11。
                # 10-18：路线定好了不能换，要在对应棋手等级多刷几下）：缺的核心 / 最低配套件刷到了能顶掉场上路线外的人，不是"只能等"
                team = set(_routes().final_team(B.route))
                for g in min_engine_groups(getattr(B, "min_eng", None) or {}):
                    if not any(x in B.field or x in B.hand_heroes for x in g):
                        core_miss += [x for x in g if any(x == n0 for n0, p0, _ in ps if p0 and (FV < F9T or round(p0 * 100) >= 1))]
                filler = [h for h in B.field if h not in team and h != (B.cur_carry or "")
                          and not (FV >= F9T and B.route_member(h))]      # f9t（r9zw_test 老师：Y 写成路线成员李元芳）：只顶路线外的人
                if FV >= F9Z:
                    # f9z（df31d67d84f9：would replace 明世隐 —— 路线核心引擎件被当成被顶的人，场上明明有张良 / 项羽）：
                    # 路线件（early_priority / 最低配套 / extra_wanted / 日落海明世隐·雅典娜）不顶，只挑路线外的人
                    _pc = f9z_route_pieces(B)
                    filler = [h for h in filler if is_hero(h) and h not in _pc]
            if core_miss and filler:
                lvl_note.insert(0, f"FULL: missing core {core_miss[0]} would replace {filler[0]}")
            else:
                lvl_note.insert(0, f"FULL + {w} waiting: " + ("finds would only wait" if new_only else "only merges help"))
    _temp_free = free and any("仅本回合可用的免费商店刷新" in (skill(t) or "") for t in (B.ctx.get("talents") or []))
    if FV >= F9LT and _temp_free:
        # f9lt（r9zj 老师：无影手的免费刷新是本回合限定，题面没说"不用就作废"，43.1 的"刷完买不起 ≤0.25"误伤）
        lvl_note.insert(0, "free refreshes are THIS ROUND ONLY: unused ones expire")
    elif FV >= F9LU and free:
        lvl_note.insert(0, "free refreshes are KEPT to next round if unused")    # 2026-10-15：普通免费刷新会留到下回合
    if FV >= F9ZB:
        return f9zb_refresh(B, ps, cost, lvl_note)
    if not ps:
        return f"refresh ({cost}{left})", lvl_note + ["no specific wanted card"]
    anyp = 1 - math.prod(1 - p for p in known) if known else 0
    return f"refresh ({cost}{left}): {round(anyp * 100)}% to see a wanted card", lvl_note + [f"{n} {'?' if p is None else round(p * 100)}%{note}" for n, p, note in
         sorted(ps, key=lambda x: -(x[1] if x[1] is not None else -1))]


def levelup_parts(B: Board) -> tuple[str, list[str]]:
    m = meta()
    cost, gold = B.ctx.get("level_cost"), B.ctx.get("gold")
    lv = B.level
    if not lv:
        return "level up (cost ?)", []
    if lv >= m.MAX_LORD_LEVEL:
        return f"level up: already MAX level {lv} (no effect)", []                     # 核查 E3：满级还写 level up to 7
    head = f"level up to {lv + 1} ({cost if (cost or -1) > 0 else '?'}e"            + (", CAN'T AFFORD" if isinstance(cost, int) and isinstance(gold, int) and cost > gold else
              (f", {gold - cost}e left" if isinstance(cost, int) and cost > 0 and isinstance(gold, int) else "")) + ")"
    items = []
    if FV >= F9LM:
        af = after_level_fill(B, cost, gold)
        if af:
            items.append(af)
    tgt = m.level_target(B.rnd) if B.rnd else 0
    if tgt and lv < tgt:
        items.append(f"BEHIND schedule (lv {tgt} by now)")
    new_t = [t for t in (m.TIER_ODDS.get(lv + 1) or {}) if t not in (m.TIER_ODDS.get(lv) or {})]
    if new_t:
        items.append(f"opens t{new_t[0]} ({round(m.TIER_ODDS[lv + 1][new_t[0]] * 100)}%)")
    if m.level_raises_pop(lv):
        items.append("+1 population")
    if isinstance(cost, int) and cost > 0:
        rounds, x, r = 0, 0, B.rnd
        while x < cost and rounds < 20:          # 本回合结束时就有经验（拍卖回合 +3）
            rounds += 1
            if FV >= "f7":                       # 拍卖经验在第 A-1 回合结束时（2026-09-25 实测时序）
                x += 3 if r + 1 in m.AUCTION_ROUNDS else 1
            else:
                x += 3 if r in m.AUCTION_ROUNDS_LEGACY else 1
            r += 1
        items.append(f"else auto level-up in ~{rounds} rounds")
    items.append("talent pick")
    return head, items


def stop_facts(B: Board) -> str:
    m = meta()
    g = B.ctx.get("gold")
    s = f"stop: keep {g if g is not None else '?'}e (+{m.energy_income(B.rnd + 1) if B.rnd else '?'} next round)"
    if B.free and not B.hand_heroes:
        s += f"; {B.free} slot{'s' if B.free > 1 else ''} stay EMPTY this battle"
    if FV >= F9Y:
        miss = f9y_missing_in_shop(B)
        if miss:
            # f9y（54.27-15，r9zzd d2e80b16c226 / 3fbd0a1b588e：经济计划 SAVE、明世隐在店却没有 buy 选项）
            s += f"; missing core/engine {'/'.join(miss[:2])} in shop now: lock keeps it"
    return s


def f9y_missing_in_shop(B: Board) -> list:
    """f9y：眼前商店里有、我方还没有的路线核心件 / 引擎产出方·放大器。"""
    shop = list(dict.fromkeys((B.ctx.get("_shop_full") or []) + (B.ctx.get("_shop_all") or [])))
    if not shop:
        return []
    if FV >= F9Z:
        return f9z_missing_in_shop(B, shop)
    team = f9x_route_team(B)
    eng = set(B.engine_missing()) if B.engine else set()
    out = []
    for h in shop:
        if not is_hero(h) or h in B.owned or h in B.nw:
            continue
        if h in team or (h in eng and B.engine_role(h) in ("producer", "amplifier")):
            out.append(h)
    return out


def f9z_route_pieces(B: Board) -> set:
    """f9z：锁定路线的"路线件" = 路线主C + 最终阵容（扣 not_wanted）+ 引擎表 early_priority（= 规则层 early_priority_full）+ 最低配套 + extra_wanted
    + 日落海线的明世隐 / 雅典娜。不含现任 / 未来主C（别家主C）。没锁或失效 → 空。"""
    if not B.route or B.route_stale:
        return set()
    me = B.min_eng or {}
    team = set(_routes().final_team(B.route)) | set(me.get("early_priority") or []) | set(me.get("extra_wanted") or [])
    for g in min_engine_groups(me):
        team |= set(g)
    team |= {x for x in SUNSET_ENGINE_F9Z if B.route_core_engine(x)}
    if B.route.get("carry"):
        team.add(B.route["carry"])
    return team - B.nw


def f9z_missing_in_shop(B: Board, shop: list) -> list:
    """f9z（54.28-13 / 54.28-6 / 54.28-15；df31d67d84f9 蒙恬、李信）：锁店留件只认路线核心 / 引擎（early_priority_full）/ 最终阵容件；
    路线没锁时只认候选路线核心卡（CORE of route）。别家主C / SECONDARY / 引擎表外的产出方不写。
    日落海主C 没满装时，店里的远洋宝藏也算（54.28-15）。"""
    locked = bool(B.route) and not B.route_stale
    if locked:
        team = f9z_route_pieces(B)
    else:
        # 路线没锁：只在主C 未定时认路线题候选路线（≥2 阶）的主C（54.28-5"主C 未定且候选是路线主C → 锁店"）
        team = set()
        if not B.carry_set():
            try:
                team = {lu.get("carry") for lu in route_candidates(B, banned=B.ctx.get("banned") or "")
                        if (tier(lu.get("carry") or "") or 0) >= 2} - {None, ""}
            except Exception:                                   # noqa: BLE001
                team = set()
    out = []
    for h in shop:
        if not is_hero(h) or h in B.owned or h in B.nw:
            continue
        if h in team:
            out.append(h)
    C = (B.route.get("carry") if locked else "") or B.carry_set()
    if C and C in B.field and fac(C) == "日落海" and "远洋宝藏" in shop and isinstance(B.eq(C), int) and B.eq(C) < 3:
        out.append("远洋宝藏")
    return out


# ---------------------------------------------------------------- 局面文本
def state_block(ctx: dict, cands: list, kind: str = "") -> str:
    B = Board(ctx)
    m = meta()
    L = []
    rnd = B.rnd
    phase = "early" if rnd <= 3 else "mid" if rnd <= 9 else "late"
    auc = ctx.get("auction_in")
    auc_s = ("AUCTION this round" + (" (done)" if ctx.get("auction_done") else "")) if auc == 0 else \
        (f"auction in {auc} rounds" if isinstance(auc, int) and auc > 0 else "no more auctions" if auc == -1 else "")
    if FV >= "f7" and rnd:
        # 老师 g5 标注反复问"拍卖开始了没有"。2026-09-25 全历史日志实测：第 A 回合（4/8/12）的拍卖在
        # 第 A-1 回合战斗之后、第 A 回合收入到账之后、第 A 回合准备阶段之前 —— 准备阶段里它要么已拍完、
        # 要么在本回合战斗之后。不用 ctx 里按旧口径 (4,8,11) 算好的 auction_in。
        nxt = [a for a in m.AUCTION_ROUNDS if a > rnd]
        if rnd in m.AUCTION_ROUNDS:
            auc_s = "auction already held at the start of this round"
        elif nxt and nxt[0] == rnd + 1:
            auc_s = (f"AUCTION right after this battle (bid with energy kept now"
                     f" + next round's +{m.energy_income(rnd + 1)})")
        elif nxt:
            auc_s = f"next auction after round {nxt[0] - 1}'s battle ({nxt[0] - 1 - rnd} round{'s' if nxt[0] - 1 - rnd > 1 else ''} away)"
        else:
            auc_s = "no more auctions"
    g = ctx.get("gold")
    L.append(f"round {rnd or '?'} ({phase} game{'; ' + auc_s if auc_s else ''}); energy {g if g is not None else '?'}"
             f" (+{m.energy_income(rnd + 1) if rnd else '?'} next round)")
    lv = B.level
    lvl = f"lord lv {lv or '?'}" + (" (MAX)" if lv and lv >= m.MAX_LORD_LEVEL else "")
    if isinstance(ctx.get("xp"), int) and ctx.get("xp_needed"):
        lvl += f", xp {ctx['xp']}/{ctx['xp_needed']}"
    if (ctx.get("level_cost") or -1) > 0:
        lvl += f", level-up {ctx['level_cost']}e"
        if FV >= F9LL and lv and lv < m.MAX_LORD_LEVEL and rnd:
            # f9ll（r9za 老师反复要："买完下回合还升不升得起"要自己推）：回合结束自动加经验 → 下回合升级费
            nc = ctx["level_cost"] - m.lord_xp_gain(rnd)
            lvl += (f" (next round {nc}e after +{m.lord_xp_gain(rnd)} xp)" if nc > 0 else " (auto level-up at round end)")
            g = ctx.get("gold")
            if nc > 0 and isinstance(g, int):
                # r9za 老师反复要："出价 / 买牌后下回合还升不升得起级"要自己算 → 直接写出不耽误下回合升级最多能花多少
                spare = g + m.energy_income(rnd + 1) - nc
                if FV >= F9LM and spare < 0:
                    # f9lm（老师：4e + 下回合 8 = 12 < 14 却写 "can spend 0e now and still level up"）
                    lvl += f"; can NOT level up next round even saving all ({g}e + {m.energy_income(rnd + 1)} = {g + m.energy_income(rnd + 1)} < {nc}e)"
                else:
                    lvl += f"; can spend {max(0, min(g, spare))}e now and still level up next round"
    elif FV >= F9LL and lv and lv < m.MAX_LORD_LEVEL:
        need = {1: 7, 2: 13, 3: 17, 4: 23, 5: 27}.get(lv)
        if need and isinstance(ctx.get("xp"), int):
            lvl += f", level-up {need - ctx['xp']}e (from xp table; cost unread)"
    tgt = m.level_target(rnd) if rnd else 0
    if lv and tgt:
        lvl += f"; schedule lv {tgt} by now" + (" → BEHIND" if lv < tgt else "")
    L.append(lvl)
    pop, cap = ctx.get("pop"), ctx.get("pop_cap")
    if FV >= F9LK:
        cap = B.cap
    if FV >= F9LQ:
        pop = B.pop
    L.append(f"population {pop if pop is not None else '?'}/{cap if cap is not None else '?'}"
             + (" (unread: counted from board list)" if FV >= F9LQ and B.pop_counted else "")
             + (f" ({B.free} free slot{'s' if B.free != 1 else ''})" if B.free is not None else "")
             + (" (cap from level table: leveled up, population not re-read)" if FV >= F9LK and B.cap_raised else "")
             + f"; heroes waiting in hand {len(set(B.hand_heroes)) if FV >= 'f9k' else len(B.hand_heroes)}"   # f9k：同名几张只占一个位置
             + (f" ({', '.join(h + (' tank' if is_front_tank(h) else '') for h in sorted(dict.fromkeys(B.hand_heroes)))})"
                if FV >= "f9k" and B.hand_heroes else "")
             + f"; hand {len(B.hand)}/{HAND_LIMIT}"
             + ("; at/over the limit: buying still allowed, cards beyond 10 are destroyed at next round start"
                if FV >= F9LQ and len(B.hand) >= HAND_LIMIT else "")
             + (f"; free refreshes {ctx['free_refresh']}" if ctx.get("free_refresh") else "")
             + (f"; banned faction {FACTION_EN.get(ctx.get('banned') or '', ctx.get('banned'))}" if ctx.get("banned") else ""))
    if FV >= F9LM and kind in ("shop", "eco", "swap"):
        L.append(empty_slot_line(B))
    if FV >= F9LN and kind == "shop" and ctx.get("eco"):
        # f9ln（2026-10-13 审计第 2 条）：经济网络本回合的决定写成事实（原来执行层按它删掉 stop/lock，逼学生满员买杂牌）
        L.append(f"economy plan this round (economy network): {ctx['eco']}"
                 + {"roll": " (spend on refreshes to find missing pieces)", "save": " (keep energy for next round)",
                    "levelup": " (level-up cost reserved)"}.get(ctx["eco"], ""))
    if B.mismatch:
        L.append("note: " + B.mismatch)
    if FV >= F9LI:
        L.extend(talent_lines(B))
        ln = lineup_line(B)
        if ln:
            L.append(ln)
    if FV >= F9LJ:
        ln = min_engine_line(B)
        if ln:
            L.append(ln)
    if FV >= F9LR:
        ln = carry_fuel_line(B, kind, cands)
        if ln:
            L.append(ln)
    if FV >= F9LU:
        for c0 in cands or []:
            ap = auto_pick_note(c0, B)
            if ap:
                L.append(f"AUTO-PICK CARD {c0}: {ap.replace('target choice ignored: ', '')}")
        ep = ctx.get("eco_plan")
        if FV >= F9LW and ep == "levelup" and B.level and B.level >= meta().MAX_LORD_LEVEL:
            ep = None          # 老师 r9zn_test/006：满级还写 LEVEL UP（经济网络输出无意义）
        if kind == "shop" and ep:
            # f9lu（第 642 局：经济网络说 roll，商店题却 stop、手里剩 4~13 能量 —— 商店题看不到经济网络的决定）
            L.append({"roll": "ECONOMY PLAN this round (economy network): ROLL — stay at this level and refresh for missing pieces / copies",
                      "levelup": "ECONOMY PLAN this round (economy network): LEVEL UP — keep the level-up cost",
                      "save": "ECONOMY PLAN this round (economy network): SAVE — keep energy for next round"}.get(ep, f"ECONOMY PLAN: {ep}"))
    if FV >= F9LP:
        ln = flow_options_line(B)
        if ln:
            L.append(ln)
    if lv:
        od = lambda x: " ".join(f"t{t} {round(p * 100)}%" for t, p in sorted((m.TIER_ODDS.get(x) or {}).items()) if p > 0)  # noqa: E731
        L.append(f"shop odds: {od(lv)}" + (" (lv6 table only measured for t3-t5)" if lv >= 6 else "")
                 + (f" | after level-up: {od(lv + 1)}" if lv < 6 else ""))
    # 主C / 主T 概览
    if B.fc:
        L.append(f"future carry: {B.fc} (fit {fit(B.fc)}/9)"
                 + ((f"; {B.hand_carry_label()} in hand: {B.fc_in_hand} ({fit(B.fc_in_hand)}/9)") if B.fc_in_hand else ""))
    else:
        best = B.carry_cands[0] if B.carry_cands else None
        L.append(("future carry: none on board (no eligible output hero: fit ≥6, t2+ after round 5, not lord-xp / carry-feeder)"
                  if FV >= F9LK else "future carry: none on board (no output hero with fit ≥6)")
                 + (f"; best candidate {best} ({fit(best)}/9)" if best else "")
                 + (f"; in hand: {B.fc_in_hand} ({fit(B.fc_in_hand)}/9)" if B.fc_in_hand else ""))
    if FV >= F9LL:
        for g, why in held_god_gear(B)[:2]:
            ws = god_gear_wearers(g, B)
            C = B.carry_set() or B.cur_carry or B.fc
            L.append(f"held t5 gear {g} ({why}; carry lv{B.lv(C) or '?'})"
                     + ("; owned heroes who can wear it: " + ", ".join(f"{h} lv{B.lv(h) or '?'}" + ("" if h in B.field else " (hand)") for h in ws[:3])
                        if ws else "; no owned hero can wear it"))
    if FV >= F9LL:
        for t, names in duo_talents(B):
            own = lambda x: "on board" if x in B.field else ("in hand" if x in B.hand_heroes else "not owned")  # noqa: E731
            L.append(f"DUO talent {t}: " + " + ".join(f"{x} ({own(x)})" for x in names)
                     + " — both must be on board together (never sell either); if the carry is not invested yet, it should be one of them")
    if FV >= F9LK and B.backup_carries:
        own = lambda x: (" (on board)" if x in B.field else (" (in hand)" if x in B.hand_heroes else ""))  # noqa: E731
        L.append(("route stale" if B.route_stale else "route carry not owned yet") + "; same-family backup route carries: " + ", ".join(x + own(x) for x in B.backup_carries[:4]))
    if FV >= F9LM:
        L.extend(flex_core_lines(B))
    if FV >= "f9e":
        if B.cur_carry and B.cur_carry != B.fc:
            L.append(f"current carry: {B.cur_carry} (t{tier(B.cur_carry)} fit {fit(B.cur_carry)}/9"
                     + (", NOT an output hero" if FV >= "f9k" and not is_output(B.cur_carry) else "")
                     + (f", {COMMITTED_OFFLIST_F9ZB}" if f9zb_committed_offlist(B, B.cur_carry) else
                        f", NOT carry-eligible now ({B.carry_why_not(B.cur_carry)}): will be replaced"
                        if FV >= F9LK and is_output(B.cur_carry) and not B.carry_ok(B.cur_carry) else "")
                     + (") — differs from future carry; 'with current carry' lines refer to it"
                        if not (FV >= F9ZA and f9za_syn_carry(B, B.cur_carry) != B.cur_carry)
                        else f") — switched off in Ali's carry list; 'with carry' lines refer to listed carry {f9za_syn_carry(B, B.cur_carry)}"))
        L.append(f"outputs on board {len(B.outputs)}: core " + (", ".join(B.core_out) or "none")
                 + ("; extra " + ", ".join(f"{h} lv{B.lv(h) or '?'}" for h in B.extra_out) if B.extra_out else "; extra none"))
    if FV >= "f9j":
        if B.engine:
            e = B.engine
            miss = B.engine_missing()
            L.append(f"level engine: {e['name']} — producers {', '.join(e['producers']) or 'none'}; amplifiers {', '.join(e['amplifiers']) or 'none'};"
                     f" payoff {', '.join(e['payoff']) or 'none'}; " + (_engine_state(e, B) if FV >= F9LK else
                                                                         ("COMPLETE" if e["complete"] else ("STARTED" if B.engine_started else "NOT started")))
                     + (f"; missing: {', '.join(miss[:5])}" if miss else "")
                     + f"; per round: {_eng().engines().get(e['key'], {}).get('per_round', '')}")
        else:
            L.append("level engine: none (no 2 parts of any engine; heroes alone do not stack levels each round)")
    L.append(f"main tank: {B.main_tank or 'none'}; front tanks {len(B.front_tanks)}")
    # 场上
    if B.field:
        L.append(f"board ({len(B.field)}):")
        order = sorted(B.field, key=lambda n: (n != B.fc, n != B.main_tank, -(fit(n) or 0)))
        for n in order:
            L.append("  " + B.unit_line(n) if is_hero(n) else f"  {n} ({CAT_EN.get(cat(n), cat(n))})")
    else:
        L.append("board: empty")
    if B.hand:
        cnt: dict = {}
        for n in B.hand:
            cnt[n] = cnt.get(n, 0) + 1
        awk = set(ctx.get("hand_awk") or [])
        items = []
        for n, k in cnt.items():
            s = f"{n} ({CAT_EN.get(cat(n), cat(n) or '?')}{' t%d' % tier(n) if tier(n) else ''})" + (f" x{k}" if k > 1 else "")
            if n in awk:
                s += " [AWAKENED, waiting to deploy]"
            elif is_hero(n) and n in B.field:
                s += " [merge copy]"
            items.append(s)
        L.append("hand: " + "; ".join(items))
        if FV >= "f7":
            # 后期闲置的英雄牌（2026-09-25：不如卖掉，去刷新商店，升级主C、主T 和副C）
            idle = [n for n in dict.fromkeys(B.hand) if is_hero(n) and n not in B.field and n not in awk]
            if idle and B.free == 0:
                L.append("idle hero cards in hand (no free slot, not merge copies): " + ", ".join(idle[:6])
                         + " — selling them gives energy for refreshes")
    # 阵容
    if B.anchor and B.anch["n"]:
        a = B.anch
        mates = ", ".join(f"{k} {round(v * 100)}%{'*' if k in B.owned else ''}" for k, v in list(a["mates"].items())[:7])
        eqs = ", ".join(f"{k} {round(v * 100)}%" for k, v in list(a["equips"].items())[:4])
        L.append(f"lineups with {B.anchor} as carry: {a['n']} (Ali {a['ali']}); teammates {mates} (*owned)")
        if eqs:
            L.append(f"  {B.anchor} equips: {eqs}")
    if FV >= "f9k" and not B.route:
        L.append("route: none committed yet (lineups below are just the closest matches)")
    for i, lu in enumerate(B.cover, 1):
        final, stage = lineup_members(lu)
        if B.nw and lu is B.route:
            final = [h for h in final if h not in B.nw]      # f9x（54.26-2）：路线不要的人不算 owns / missing
        have = [h for h in final if h in B.owned]
        miss = [h for h in final if h not in B.owned][:5]
        ali = LORD in (lu.get("lord_names") or [])
        lc = lineup_carry(lu) or "?"
        tag = (f"ROUTE (committed r{B.route_since}) " if FV >= "f9k" and B.route and lu is B.route else "")
        if tag and B.route_stale:
            tag = f"ROUTE stale (carry {B.route.get('carry')} not owned, committed r{B.route_since}) — ignore route tags; "
        elif tag and B.route_carry_missing:
            tag = f"ROUTE (carry {B.route.get('carry')} not owned yet, committed r{B.route_since}) "
        s = (f"{tag}lineup#{i} {lu.get('name')} ({'Ali' if ali else 'other lord'}): carry {lc}"
             f"{'*' if lc in B.owned else ''}; owns {len(have)}/{len(final)}; missing {', '.join(miss) or '-'}")
        if FV >= F9ZA and i == 1:
            # f9za-fix（协调方 12；test batch_008 c80d83a5f2b3：owns 7/7 但场上 + 手里有 8 人）：写出不在 lineup#1 里的人
            _out = [h for h in dict.fromkeys(list(B.heroes) + list(B.hand_heroes)) if h not in final]
            if _out:
                s += "; owned but not in it: " + ", ".join(f"{h}{' (hand)' if h not in B.field else ''}" for h in _out[:4])
        L.append(s)
        if ali and not (B.route_stale and lu is B.route):          # f9l：失效路线不再写当前阶段计划
            st = next((x for x in lu.get("stages") or [] if x.get("from_round", 0) <= rnd <= x.get("to_round", 99)), None)
            if st:
                sdesc = re.sub(r"\s+", " ", st.get("desc") or "")[:60]
                _pm = [h for h in st.get('main') or [] if not (lu is B.route and h in B.nw)]     # f9x：路线不要的人不进计划行
                L.append(f"  plan r{st.get('from_round')}-{st.get('to_round')}: {' '.join(_pm)}; {sdesc}")
    # 计数
    fs = ", ".join(f"{FACTION_EN.get(k, k)} {v}" for k, v in sorted(B.factions.items(), key=lambda x: -x[1]))
    L.append(f"factions on board: {fs or '-'} | awakened {len(B.awakened)}; deploy-effect {len(B.deploy_src)}; "
             f"prepare-effect {len(B.prepare_src)}; sacrifice {len(B.sacrifice_src)}; blink {len(B.blink)}; "
             f"temp-level sources {len(B.temp_src)}; coin sources {len(B.coin_src)}"
             + (f"; battle-start {sum(1 for h in B.heroes if '开团' in kw(h))}" if FV >= "f9" else ""))
    if ctx.get("maple") is not None:
        _mp = ctx.get('maple') or ''
        _st = (FV >= F9Y and _mp and _mp not in B.field and _mp not in B.hand)
        # f9y（r9zzd：宿主已卖掉 / 离场，枫叶舞记账没清）：读数陈旧时写明
        L.append(f"maple dance bound to: {_mp or 'nobody'}" + (f" (stale: {_mp} not on board)" if _st else ""))
    if kind == "shop":
        h, it = refresh_parts(B)
        L.append("shop actions: " + h + (" — " + ", ".join(it) if it else ""))
        h, it = levelup_parts(B)
        L.append("  " + h + (" — " + "; ".join(it) if it else ""))
        L.append("  " + stop_facts(B))
    if FV >= "f7" and cands:
        # 候选卡面里出现的官方关键词，附上官方解释（老师 g5 标注：没说"阿科米亚核心""夺取"是什么）
        txt = " ".join(skill(x if isinstance(x, str) else x[0]) + " " + (x[1] if isinstance(x, tuple) else "") for x in cands)
        kws = [(k, KW_EN.get(k, v)) for k, v in keywords().items() if k in txt]
        if kws:
            L.append("keywords: " + "; ".join(f"{k}={v}" for k, v in kws[:6]))
    if FV >= F9X and kind == "card_pick" and cands:
        nu = f9x_special_none_usable([x if isinstance(x, str) else x[0] for x in cands], B)
        if nu:
            L.append(nu)        # f9x（54.26-21）：特殊装备三选一三件都不对口
    # 候选详情
    if cands:
        L.append("candidate details:")
        for i, item in enumerate(cands):
            n, desc = (item if isinstance(item, tuple) else (item, ""))
            if FV >= "f8" and kind == "carry" and n in B.field:
                # f8：carry 候选都站在场上，不是要买的牌 —— 只写场上事实（老师 r8：沿用商店"merge…"措辞读起来像商店）
                lv_ = B.lv(n)
                nxt_ = next((t for t in SKILL_THRESHOLDS if lv_ is not None and lv_ < t), None)
                key = [f"on board lv {lv_ or '?'}" + (f" (next skill jump lv{nxt_}, +{nxt_ - lv_})" if nxt_ else
                                                      (" (all skill jumps done)" if lv_ else "")),
                       B.awaken_str(n)] + [t.split(" (")[0] for t in B.identity(n)[:2]]
                det = []
            else:
                key, det = facts_for(n, B, desc, kind)
                if FV >= F9ZA and cat(n) == "装备牌":
                    key = f9za_detail_fit(n, B, key)      # f9za-fix（batch_009 da438202de9f）：详情行和选项行同一适配口径
            rn = resolve(n, desc)[0] if (desc and not is_hero(n)) else n
            txt = re.sub(r"\s+", " ", ((c(rn).get("skill") if rn else "") or desc or "")).strip()
            if is_hero(n):
                txt = txt.replace(n, "自身")
            if FV >= F9LM and rn and c(rn).get("given_skill"):
                txt = txt[:90] + " 赠送牌：" + re.sub(r"\s+", " ", c(rn)["given_skill"]).strip()[:80]   # f9lm：天赋送出的同名牌效果
            head = (f"[{i}] {n} t{tier(n) or '?'} {CAT_EN.get(cat(n), cat(n) or '?')}"
                    + (f" {fac_en(n)} {role(n)} {row(n)} {'melee' if melee(n) else 'ranged'}" if is_hero(n) else ""))
            L.append("  " + head + " || " + (txt[:180] if FV >= F9LM and " 赠送牌：" in txt else txt[:90]))
            L.append("     " + " | ".join(key + det))
    return "\n".join(L)


# ---------------------------------------------------------------- 选项文本（≤48 token，结论前置）
@lru_cache(maxsize=1)
def _tok():
    """Laya 的 tokenizer（mmBERT，所有微调 checkpoint 共用同一个）。从本地 HF 缓存加载，不联网。"""
    import glob
    import os
    roots = [os.environ.get("HF_HOME") or "", str(PROJECT_DIR / ".hf_cache"), str(Path.home() / ".cache" / "huggingface")]
    for r in roots:
        for d in glob.glob(os.path.join(r, "hub", "models--convaiinnovations--laya", "snapshots", "*", "multilingual", "tokenizer")):
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(d)
            except Exception:                           # noqa: BLE001
                pass
    return None


def est_tokens(s: str) -> int:
    """选项的 token 数（和 Laya build_sequence 一样前面加空格、不加特殊符号）。拿不到 tokenizer 时按字符估 ×1.35 保底。"""
    t = _tok()
    if t is not None:
        return len(t(" " + s, add_special_tokens=False)["input_ids"])
    cjk = sum(1 for ch in s if "一" <= ch <= "鿿")
    return math.ceil((cjk + (len(s) - cjk) / 3.6) * 1.35)


OPT_TOKENS = 44          # Laya 单个选项硬截断 48 token（含开头的 [MASK]），留点余量


def pack(head: str, items: list[str], budget: int = OPT_TOKENS) -> str:
    """结论按优先级往选项里放，放不下就停（宁可少一条，也不让最关键的被截掉一半）。"""
    if FV >= F9ZB:
        return pack_f9zb(head, items, budget)
    out = head
    for it in items:
        sep = "; " if out != head else ("" if head.endswith((" ", ":")) else " — ")
        cand = out + sep + it
        if est_tokens(cand) > budget:
            _keep = ((FV >= F9T and re.match(r"AVOID|DEAD|.*no value|RACE", it))
                     or (FV >= F9Y and re.match(r"LOW VALUE|for carry .*MISMATCH|AWAKENS core|commits the route", it)))
            if _keep and out == head:
                # f9t（天赋审查 10-18：乘胜追击的 DEAD 句 47 token，被整句丢掉，模型看不到）：关键否定句截断保留
                words = it.split(" ")
                while words and est_tokens(out + sep + " ".join(words)) > budget:
                    words.pop()
                if words:
                    out = out + sep + " ".join(words)
                continue
            if FV >= F9LM:
                continue          # f9lm：前置了 CORE / TEMPORARY 标签后，放不下的那条跳过、后面短的照放（原来整条后面全丢）
            break
        out = cand
    return out


def duo_talent_fact(n: str, B: "Board") -> str:
    """f9o（第 717 局 R10：兄妹 0.35 被选，给铠 + 露娜、两人同时在场才有效；主C 百里守约不是其中之一、满员 6/6 还有 1 人等位；
    阵容不需要露娜，上不了场的话这个天赋纯属白费）：共战天赋写清生效条件、主C 是否其中之一、路线外、还缺几个位置。"""
    if cat(n) != "天赋牌" or ("共战" not in (skill(n) or "") and not (FV >= F9T and DUO_F9T_RE.search(skill(n) or ""))):
        return ""
    pair = duo_pair(skill(n) or "") if FV >= F9T else [h for h in carry_table() if is_hero(h) and h in (skill(n) or "")]
    if len(pair) < 2:
        return ""
    C = B.carry_set() or B.cur_carry or B.fc
    need = sum(1 for h in pair if h not in B.field)
    route = (set(_routes().final_team(B.route)) | set(_routes().members(B.route))) if B.route else set()
    off = [h for h in pair if route and h not in route]
    if FV >= F9X:
        dx = f9x_duo_fact(pair, B)      # f9x（54.26-23）：两人都要在路线里、且其一是主C
        if dx:
            return dx
    if FV >= F9T and C and C not in pair:
        # f9t（2026-10-18 第 748 局：兄妹只有在其中一个是主C 时才有意义）
        return f"DUO {'+'.join(pair)}: carry {C} is not one of them → no value for us"
    return (f"DUO {'+'.join(pair)} both on board"
            + (f", carry not one" if C and C not in pair else "")
            + (f", off-route {','.join(off)}" if off else "")
            + (f", needs {need} slots (free {B.free if B.free is not None else '?'})" if need else ""))


def option_text(n: str, ctx: dict, prefix: str = "", desc: str = "", kind: str = "", B: Board | None = None) -> str:
    B = B or Board(ctx)
    key, _ = facts_for(n, B, desc, kind)
    if FV >= F9LU and kind in ("", "talent") and (cat(n) == "天赋牌" or ctx.get("_talent_q")):
        key = talent_value_facts(n, B) + key
    if FV >= F9T and cat(n) == "效果牌" and re.search(r"阿科米亚核心等级\s*\+\s*(\d+)", skill(n) or ""):
        # f9t（2026-10-19 第 756 局 R6：已经是日落海阵容却没买核心助力那张卡——题面只写公式，Laya 0.25 选刷新）
        base = int(re.search(r"阿科米亚核心等级\s*\+\s*(\d+)", skill(n)).group(1))
        sun = sum(1 for h in B.heroes if fac(h) == norm_fac("日落海"))
        tot = base * (1 + sun) if re.search(r"每有\s*1\s*名【日落海】英雄重复", skill(n)) else base
        C = B.carry_set() or B.cur_carry or B.fc
        feeds = bool(C) and fac(C) == norm_fac("日落海")
        key = [f"core +{tot} lv now ({sun} 【日落海】 on board)" + ("; carry is 【日落海】: core levels feed the carry" if feeds else "; carry not 【日落海】: little use")] + key
    if FV >= F9O and kind in ("", "talent"):
        dt = duo_talent_fact(n, B)
        if dt:
            key = [dt] + key
    if FV >= F9LZ and any(k.startswith("SECONDARY") for k in key):
        key = [k for k in key if k.startswith("SECONDARY")] + [k for k in key if not k.startswith("SECONDARY")]   # f9lz：次选标签最先（预算有限）
    if FV >= F9O:
        # f9o（第 711 局 R3 拍卖：马可波罗只写 "CORE of route 马可波罗"，主C 候选标签被预算挤掉；公孙离写着 CARRY CANDIDATE fit 6 → 拍了公孙离）：
        # 路线核心本身是候选主C 时合成一条，放最前
        cc = next((k for k in key if k.startswith("CARRY CANDIDATE")), None)
        co = next((k for k in key if k.startswith("CORE of route")), None)
        if cc and co:
            fm = re.search(r"fit \d/9, t\d", cc)
            key = [co + (f" = its carry ({fm.group(0)})" if fm else "")] + [k for k in key if k not in (cc, co)]
    if FV >= F9Q:
        # f9q（第 723 局 R8：刘禅 "with carry: 沈梦溪 +2 temp lv 每次放技能; same build as 沈梦溪: 施法流" 太长被 44 token 预算整条丢掉，
        # 题面只剩"满员 / 最弱是牛魔"）：with carry 只留分号前的效果
        key = [k.split("; ")[0] if k.startswith("with carry:") else k for k in key]
    if FV >= F9U:
        key = list(dict.fromkeys(key))     # f9u：天赋事实和 facts_for 都写 "carry X equips n/3"（鼓舞之戒两遍），重复的占预算
    if FV >= F9W:
        # f9w（54.25-5）：N 选 1 全是装备 → 主C 件数 + 输出 / 防御；高攻速主C 没吸血装 → 吸血装 / 吸血天赋写出来。放最前
        pre = [x for x in ((f9w_equip_pick_fact(n, B) if ctx.get("_equip_pick") else ""), f9w_lifesteal_fact(n, B)) if x]
        if FV >= F9Y and kind in ("", "talent") and ctx.get("_talent_q"):
            sf = f9y_special_fit(n, B)          # f9y：天赋界面的特殊装备三选一，对主C 的伤害类型适配
            if sf:
                pre = [sf] + pre
                # 旧的伤害类型 / 适配行（"magic item"、"X phys → MISMATCH"、普攻叠层 MISMATCH）合进这一行，不重复
                key = [k for k in key if not (re.fullmatch(r"(magic|physical) item", k) or "MISMATCH" in k
                                              or re.fullmatch(r"\S+ (phys|mag|hybrid)", k))]
        if pre:
            _dup = f"carry {f9w_carry(B)} equips" if ctx.get("_equip_pick") and cat(n) == "装备牌" else None
            key = pre + [k for k in key if k not in pre and not (_dup and k.startswith(_dup))]     # 件数已在前面写了
    if FV >= F9X and is_hero(n) and kind != "talent":
        # f9x：触发来源 / 全队效果（54.26-18）放第 2 条；攻速主C 的少司缘补位（54.26-24）放最前
        tg = f9x_trigger_fact(n, B)
        if tg and tg not in key:
            key = key[:1] + [tg] + key[1:]
        ss = f9x_shaosiyuan_fact(n, B)
        if ss:
            key = [ss] + [k for k in key if k != ss]
    if FV >= F9Y and "ROUTE core engine piece" in key:
        key = ["ROUTE core engine piece"] + [k for k in key if k != "ROUTE core engine piece"]     # f9y：路线身份放最前（预算有限）
    if FV >= F9Y and ctx.get("_shop_q") and kind == "":
        aw6 = f9y_awaken_vs_lv6(n, B)      # f9y（54.27-14）：5 级时"买下觉醒核心件"与"下回合升 6"并写
        if aw6:
            key = [aw6] + key
    if FV >= F9ZA and kind != "lord_boost":
        key = f9za_option_lines(n, B, ctx, kind, prefix, key)     # f9za：主C 等级收益估算 / 装备适配（同执行层）/ 满员买人要卖谁 / 守约燃料
    if FV >= F9LF and key:
        key = fit_first_item(prefix + n + ": ", key)     # f9lf：结论太长时缩短而不是整条丢掉
    return pack(prefix + n + ": ", key)


# ---------------------------------------------------------------- use_card 题（特殊效果牌：用在谁 / 先留着）
# 2026-09-25：这种特殊效果牌最终交给 Laya 决定。先影子模式：只出题记录，不执行。
SPECIAL_TARGETS = {"hold", "low_tier_noncarry", "lowest_tier_high_level", "disposable", "entry_hero", "prep_hero"}
TRANSFORM_RE = re.compile(r"英雄变为|变为(?:同阵营)?(?:高\d阶|\d阶)?的?(?:随机)?英雄")
REMOVE_RE = re.compile(r"移除(?:1名|一名)|消除")


def _is_transform(sk: str) -> bool:
    """己方英雄变身牌。排除"敌方…变为"（天赋猪猪：敌方 2 名随机英雄变为小猪，g6 老师指出被当成了变身牌）。"""
    return bool(TRANSFORM_RE.search(sk)) and not re.search(r"敌方[^。；]{0,10}变为", sk)


def special_card(n: str) -> bool:
    """需要"用在谁"判断的效果牌：审定为 hold / 特殊目标，或卡面是变身、移除、觉醒进度 +1。"""
    if is_hero(n) or cat(n) not in ("效果牌", "天赋牌"):
        return False
    t = (usage().get(n) or {}).get("target", "").split(":")[0]
    sk = skill(n)
    return (t in SPECIAL_TARGETS or _is_transform(sk) or bool(REMOVE_RE.search(sk))
            or bool(re.search(r"觉醒进度\s*\+", sk))
            or (FV >= F9LR and bool(DESTROY_EQ_RE.search(sk))))     # f9lr：武装分解 / 武装重构（第 632 局规则判"没目标"直接卖了）


TARGET_RES = {"等级", "临时等级", "攻击力", "攻速", "暴击", "护盾", "生命上限", "法力上限减", "复生", "装备"}


def targeted_card(n: str) -> bool:
    """f9d：指定 1 名英雄的效果牌 / 天赋牌（加等级、临时等级、攻击…）—— 给谁由学生判（2026-09-26：
    "主C 是垃圾英雄，获得的效果牌全用给主C，导致其他英雄跟不上"）。"""
    if is_hero(n) or cat(n) not in ("效果牌", "天赋牌"):
        return False
    sk = skill(n)
    if not re.search(r"指定\d*名", sk):
        return False
    return any(g.get("res") in TARGET_RES for g in (kb().get(n) or {}).get("gives") or []) or "等级" in sk


def use_card_question(n: str) -> bool:
    return special_card(n) or (FV >= "f9d" and targeted_card(n))


def use_card_targets(n: str, B: "Board", limit: int = 6) -> list[str]:
    """按卡面限制列出场上的**合法目标**：品阶范围（1~2 阶/1-3 阶）、【阵营】、限远程/近战、觉醒进度牌不给已觉醒。"""
    sk = skill(n)
    hs = list(B.heroes)
    m = re.search(r"(\d)\s*[~-]\s*(\d)\s*阶", sk)
    if m and FV >= F9O and sk[max(0, m.start() - 1):m.start()] == "高":
        m = None        # f9o（第 715 局：前浪的荣光"获得1张**高**1-2阶的同阵营英雄牌"被当成"只能指定 1~2 阶"，主C 狄仁杰不在选项里，Laya 0.65 一直留着）
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        hs = [h for h in hs if lo <= tier(h) <= hi]
    mf = re.search(r"指定\d*名【([^】]+)】英雄", sk)
    if mf:
        hs = [h for h in hs if fac(h) == norm_fac(mf.group(1))]
    if "限定远程" in sk:
        hs = [h for h in hs if not melee(h)]
    if "限定近战" in sk:
        hs = [h for h in hs if melee(h)]
    if re.search(r"觉醒进度\s*\+", sk):
        hs = [h for h in hs if B.awk(h) != 0]
    if FV >= F9LR and DESTROY_EQ_RE.search(sk):
        hs = [h for h in hs if (B.eq(h) or 0) > 0]          # 摧毁装备：只有身上有装备的人是合法目标
    if FV >= F9LT:
        mt = TRIGGER_RE.search(sk)
        if mt:
            # f9lt（r9zj 老师：临阵磨枪题面写着整备英雄是程咬金，选项里却没有他）：触发类牌只列自带「X：」效果条目的人（同 rules.card_target_ok）
            hs = [h for h in hs if own_kw(h, mt.group(1))]
    if FV >= F9Y:
        # f9y（r9zzd：有福同享选项漏了主C 沈梦溪）：现任主C / 路线主C 排在未来主C 之后、其他人之前，不会被 limit 截掉
        _rc = (B.route.get("carry") or "") if B.route and not B.route_stale else ""
        _cs = [x for x in (B.fc, B.cur_carry, B.ctx.get("carry"), _rc) if x]
        hs.sort(key=lambda h: (_cs.index(h) if h in _cs else len(_cs), -(fit(h) or 0), -(B.lv(h) or 0), h))
        if TIER_GROUP_RE_F9Y.search(sk):
            # 有福同享（"场上和其品阶相同的英雄等级+4"）：同品阶的目标效果完全相同 → 每个品阶只留一个（排在前面的：主C 优先）
            seen, keep = set(), []
            for h in hs:
                if tier(h) not in seen:
                    seen.add(tier(h))
                    keep.append(h)
            hs = keep
        elif auto_pick_note(n, B):
            hs = hs[:1]             # 卡面自己挑对象（指定谁都一样）→ 只留一个目标
        return hs[:limit]
    hs.sort(key=lambda h: (h != B.fc, -(fit(h) or 0), -(B.lv(h) or 0), h))
    return hs[:limit]


TIER_GROUP_RE_F9Y = re.compile(r"和其品阶相同的英雄")


def _part_note(n: str, B: "Board") -> str:
    """f9k：被卖的人是引擎零件 / 路线成员时的短注（换主C 的代价）。"""
    if FV < "f9k":
        return ""
    if B.engine_started and B.engine_role(n) in ("producer", "amplifier"):
        return f" (ENGINE {B.engine_role(n)})"
    if FV >= F9W and B.route_not_wanted(n):
        return ""
    if B.route and (B.route_member(n) if FV >= "f9l" else n in _routes().members(B.route)):
        return "" if f9zb_off_route_tag(n, B) else " (ROUTE member)"         # f9zb（54.37-5）
    return ""


def auto_pick_note(n: str, B: "Board") -> str:
    """f9lt：卡面按规则自己挑对象（"消除场上4名等级最低的英雄"）—— 指定的目标不起作用，写出实际会作用到谁。"""
    sk = skill(n) or ""
    m = AUTO_LOWEST_RE.search(sk)
    if not m:
        return ""
    k = int(m.group(2))
    hs = sorted(B.heroes, key=lambda h: (B.lv(h) or 0, h))
    hit = hs[:k]
    left = len(B.heroes) - len(hit)
    verb = "removes" if m.group(1) in ("消除", "移除") else "affects"
    return (f"target choice ignored: {verb} the {k} lowest-lv heroes = {'/'.join(f'{h} lv{B.lv(h) or 0}' for h in hit) or 'none'}"
            + (f"; leaves {left} on board" if verb == "removes" else ""))


def use_on_facts(n: str, t: str, B: "Board") -> list[str]:
    """把牌 n 用在 t 身上会发生什么 —— **分条、结果在前且短**（选项 ≤44 token，放不下的后几条会被丢掉）。"""
    sk = skill(n)
    if FV >= F9LT:
        ap = auto_pick_note(n, B)
        if ap:
            return ["same effect whichever target (see AUTO-PICK line)"] if FV >= F9LU else [ap]
    idt = [x.split(" (")[0] for x in B.identity(t)]
    role = ", ".join(idt[:2])
    state_s = f"t{tier(t)} lv {B.lv(t) or '?'} {B.awaken_str(t).split(' (')[0]}"
    if FV >= "f9d" and is_output(t) and fit(t) is not None:
        state_s += f" fit {fit(t)}/9"            # f9d：给谁 = 谁的成长最值（强度分是关键）
    res = []
    mup = re.search(r"获得\d*张高(\d)\s*[~-]\s*(\d)\s*阶的同阵营英雄牌", sk) if FV >= F9O else None
    if mup and tier(t):
        lo, hi = min(5, tier(t) + int(mup.group(1))), min(5, tier(t) + int(mup.group(2)))
        res.append(f"→ gets 1 random 【{fac(t) or '?'}】 hero card t{lo}" + (f"-t{hi}" if hi > lo else "") + " into hand")
    mtr = re.search(r"触发其(登场|整备)效果", sk) if FV >= F9LY else None
    if mtr:
        # f9ly（第 673 局 R6：唤醒 → 马可波罗，选项只写"MAPLE HOST, FUTURE CARRY"，没写再触发一次登场 = 自身 +2、核心 +2；Laya 0.5/0.55 判留着）
        k = mtr.group(1)
        e = kb().get(t) or {}
        if e.get("trig") == k:
            eff = kb_effect(t, 2).split(": ", 1)[-1]
            res.append(f"re-triggers its {'deploy' if k == '登场' else 'prepare'}: {eff}" + (" (awakened: stronger)" if B.awk(t) == 0 else ""))
        elif FV >= F9T and own_kw(t, k):
            # f9t（2026-10-19：明世隐在最终阵容时，触发登场效果的牌可以用到明世隐身上；明世隐 kb 的 trig 不是登场，原来误写 nothing happens）
            mk = re.search(rf"{k}[：:]\s*([^。]+)", skill(t) or "")
            res.append(f"re-triggers its {'deploy' if k == '登场' else 'prepare'}: {(mk.group(1) if mk else '?')[:30]}"
                       + (" (final team piece)" if B.route and t in _routes().final_team(B.route) else ""))
        else:
            res.append(f"has no {'deploy' if k == '登场' else 'prepare'} effect → nothing happens")
    if FV >= F9LW and n == "混沌裂隙":
        # f9lw ㊵（已确认的游戏规则）：每场战前把格上的人换成随机英雄（2 倍等级、满装备）只打这一场，战后原英雄回来
        lv = B.lv(t)
        others = sorted((h for h in B.heroes if h != t and h not in (B.fc, B.cur_carry) and not is_support(h) and not nearest_buddy(h)),
                        key=lambda h: -(B.lv(h) or 0))
        rank = 1 + sum(1 for h in others if (B.lv(h) or 0) > (lv or 0))
        kind = ("CARRY" if t in (B.fc, B.cur_carry) else "SUPPORT" if (is_support(t) or nearest_buddy(t))
                else f"#{rank} non-carry")          # 老师 r9zn_test/007：贴身辅助（太乙真人）算 SUPPORT
        if FV >= F9LY:
            # f9ly（10-16 午：放等级高的那个）：等级最高的非主C 是首选；搭档照旧标出代价
            if any(t in names for _t, names in duo_talents(B)):
                kind = "DUO partner: its duo effect is lost in that battle"
            elif t in (rc := rift_candidates(B)) and (B.lv(t) or 0) == (B.lv(rc[0]) or 0):
                kind = "HIGHEST-level non-carry: the rift spot"
        elif FV >= F9LX:
            # 2026-10-16：一般放置主T；搭档英雄放这里，搭档效果战斗时就没了
            if t == B.main_tank:
                kind = "MAIN TANK: the usual rift spot"
            elif any(t in names for _t, names in duo_talents(B)):
                kind = "DUO partner: its duo effect is lost in that battle"
        # 老师 r9zn 多批：排名那条被 pack 挤掉（选项 44 token）→ 只留一条短的放最前
        return [f"rift lv {2 * lv if lv else '?'} ({kind})", state_s]
    if _is_transform(sk):
        loses = [x for x, c in (("AWAKENING", B.awk(t) == 0), ("maple", B.ctx.get("maple") == t),
                                ("CARRY role", t == B.fc or (FV >= "f9k" and t == B.cur_carry)),
                                ("ROUTE carry", FV >= "f9k" and B.route is not None and t == (B.route.get("carry") or "")),
                                ("MAIN TANK role", t == B.main_tank)) if c]
        res.append("becomes a random hero" + ("; LOSES " + ", ".join(loses) if loses else ""))
    elif FV >= "f9l" and multi_give(n):
        res.append(multi_total(n, B, t).replace(f" all on {t}:", " all here:"))    # f9l：N 张一起用在他身上的合计
    elif re.search(r"觉醒进度\s*\+", sk):
        k = B.awk(t)
        res.append((f"awaken {3 - k}→{4 - k}/3" + (" COMPLETES (x2, back to hand)" if k == 1 else "")) if k else "awaken +1 (now ?)")
    elif REMOVE_RE.search(sk) or "移除" in sk:
        up = re.search(r"高(\d)阶", sk)
        if up:
            res.append(f"removed → random t+{up.group(1)} hero card" + (f" at lv {(B.lv(t) or 0) + 10}" if "+10" in sk and B.lv(t) else ""))
        else:
            res.append("REMOVED from board")
        loses = [x for x, c in (("AWAKENING", B.awk(t) == 0),
                                ("CARRY role", t == B.fc or (FV >= "f9k" and t == B.cur_carry)),
                                ("ROUTE carry", FV >= "f9k" and B.route is not None and t == (B.route.get("carry") or "")),
                                ("MAIN TANK role", t == B.main_tank)) if c]
        if FV >= F9M:
            # f9m（第 686 局 R5：限制性进化变掉了主C 马可波罗阵容里的米莱狄 / 狂铁）：变掉的是主C 阵容件 / 路线成员时写出来
            if B.carry_lineup_mate(t):
                loses.append(f"a piece of carry {B.fc}'s lineup")
            elif B.route and not B.route_stale and t in _routes().members(B.route) and t != (B.route.get("carry") or ""):
                loses.append("a ROUTE member")
        if loses:
            res[-1] += "; LOSES " + ", ".join(loses)
        if "能量" in sk and B.awk(t) in (0, 1, 2, 3):
            res.append(f"+{[9, 7, 5, 3][B.awk(t)]} energy")
    elif FV >= F9LR and DESTROY_EQ_RE.search(sk):
        k = B.eq(t) or 0
        if FV >= F9Y and (B.ctx.get("worn") or {}).get(t):
            k = len((B.ctx.get("worn") or {}).get(t))       # f9y（r9zzd：destroys its 2 却列 3 件）：件数只认穿戴账本（同下面列出的装备）
        loses = [x for x, c in (("CARRY gear", t == B.fc or t == B.cur_carry), ("MAIN TANK gear", t == B.main_tank)) if c]
        res.append(f"destroys its {k} item(s)" + ("; LOSES " + ", ".join(loses) if loses else ""))
        worn = (B.ctx.get("worn") or {}).get(t) or []
        if FV >= F9LY and worn:
            # f9ly（老师 r9zo 多批：拆装备题不写被毁的是哪几件 → 54.10-8"有特殊高阶装备不用"判不了）：写出装备名和品阶
            res.append("items: " + ", ".join(f"{w} t{tier(w) or '?'}" for w in worn[:4]))
    else:
        e = kb_entry(n) if FV >= F9T else (kb().get(n) or {})
        g = next((g for g in e.get("gives") or [] if g.get("res") in ("等级", "临时等级")), None)
        if g and _num(g.get("n")) is not None:
            mul, why_m = _retrigger(n, B)
            res.append(_lvl_result(B, t, _num(g.get("n")) * mul).replace(t + " ", "") + (" TEMP" if g.get("res") == "临时等级" else "")
                       + (f" ({why_m})" if mul > 1 else ""))
    if FV >= F9T:
        # f9t（2026-10-19 卡面确认）：开局狂飙每次放技能都算 / 化铁为金直接上场 / 传承给等级最低 3 人 / 夺取之术只够周围 1 格
        if n == "开局狂飙":
            g = cast_growers(B)
            if g:           # 没有孙膑 / 刘禅时每个目标都一样，不写（省预算）
                res.insert(0, f"casts nonstop 15s: {'/'.join(g[:2])} on board → {t} gains lv per cast")
        elif n == "化铁为金":
            lz = [x for x, c in (("maple", B.ctx.get("maple") == t), ("CARRY role", t in (B.fc, B.cur_carry)),
                                 ("MAIN TANK role", t == B.main_tank)) if c]
            res.insert(0, f"becomes a random AWAKENED t5, deploys at once, keeps lv {B.lv(t) or '?'}" + ("; LOSES " + ", ".join(lz) if lz else ""))
        elif n == "传承":
            low = sorted((h for h in B.heroes if h != t), key=lambda h: (B.lv(h) or 0, h))[:3]
            C = B.carry_set() or B.cur_carry or B.fc
            res.insert(0, f"{max(0, (B.lv(t) or 1) - 1)} lv → {'/'.join(low) or 'nobody'}" + (" (incl. carry)" if C in low else "") + "; keeps lv 1")
        elif n == "众星拱月":
            res.insert(0, f"+{sum(min(10, max(0, (B.lv(h) or 0) - 1)) for h in B.heroes if h != t)} lv (≤10 from each other, each keeps 1)")
        elif n == "夺取之术":
            nb = [h for h in B.heroes if h != t and adjacent(B.cell(h), B.cell(t))]
            if B.cell(t) is not None:
                gain = max((min(10, max(0, (B.lv(h) or 0) - 1)) for h in nb), default=0)
                res.insert(0, f"steals from 1-cell ally ({'/'.join(nb[:3]) or 'none in reach'}): up to +{gain}")
    if FV >= "f9h" and "品阶相同" in sk and res:
        # f9h（r9j 老师：有福同享 = 场上同品阶的人一起 +N，选项只写了目标本人，几个选项看起来不同其实一样）
        co = [h for h in B.heroes if h != t and tier(h) == tier(t)]
        res.append(("also: " + ", ".join(co[:4]) + (f" +{len(co) - 4}" if len(co) > 4 else "")) if co else f"only t{tier(t)} on board")
    if FV >= F9LM and B.temp_carry and t == B.temp_carry:
        res.insert(0, "target is TEMPORARY carry")      # f9lm：加等级 / 装备留给最终主C（2026-10-13）
    return res + ([role] if role else []) + [state_s]


ACCUM_RE = re.compile(r"获胜时，累积(\d+)张[^；;。]*；失败时，损失一半")
HOLD_GROW_RE = re.compile(r"每持有1回合[^。；]{0,12}?\+(\d+)")


def timing_parts(n: str, B: "Board") -> tuple[list[str], list[str]]:
    """f9lt use_timing 题（现在用 / 先留着）：两边各自的事实。只写卡面和算好的数，不写规则怎么判。"""
    sk = skill(n) or ""
    now = [sk[:60]]
    hold = []
    m = ACCUM_RE.search(sk)
    ct = B.ctx.get("contract") or {}
    if m and ct:
        c, p, k = ct.get("count", 0), ct.get("p", 0.5), ct.get("battles", 0)
        th = 6 * p / max(1e-6, 1 - p)
        now.insert(0, f"cash in ~{c} hero cards now")
        hold.append(f"~{c} cards after {k} battles; win rate est {round(p * 100)}%: win +{m.group(1)}, loss halves; "
                    f"holding gains on average only while cards < {th:.1f}")
    g = HOLD_GROW_RE.search(sk)
    if g:
        hold.append(f"grows +{g.group(1)} per round held")
    if FV >= F9LW and re.search(r"获得场上(?:所有)?英雄牌各1张", sk):
        # f9lw ㊵（2026-10-16：点兵点将看场上是否都是核心阵容牌，越多性价比越高）：写核心占比、谁差 1 张觉醒
        core = set(_routes().final_team(B.route)) | {B.route.get("carry") or ""} if B.route and not B.route_stale else set()
        core |= {x for x in (B.fc, B.cur_carry) if x}
        cs = [h for h in B.heroes if h in core]
        near = [h for h in cs if B.awk(h) == 1]
        junk = [h for h in B.heroes if h not in core]
        # 分条、短（pack 按条放，放不下整条丢）：核心占比 → 立刻觉醒的 → 杂牌复制品
        now[:] = [f"copy of each board hero: CORE {len(cs)}/{len(B.heroes)}"]
        if near:
            now.append("awakens NOW: " + ", ".join(near))
        if cs:
            now.append("core: " + ", ".join(cs[:4]))
        if junk:
            now.append("non-core copies (sell only): " + ", ".join(junk[:3]))
        hold.append("later: board may hold more CORE pieces (copies go to them)")
    if FV >= F9LT and B.rnd:
        hold.append(f"round {B.rnd}; hand {len(B.hand)}/{HAND_LIMIT}")
    if FV >= F9ZB and f9zb_entry_all_now(n, B):
        now.insert(0, USE_NOW_F9ZB)             # f9zb（54.36-6）
    return now, hold


def use_card_hold(n: str, B: "Board") -> str:
    s = "hold: keep it in hand"
    if fleeting_now(n, B):
        s += " (FLEETING: lost at round end)"
    s += f"; hand {len(B.hand)}/{HAND_LIMIT}"
    if FV >= F9LL and LEVEL_GAIN_RE.search(skill(n)) and not B.carry_set():
        # f9ll（2026-10-12：主C 未确定之前，涨等级的效果牌先留着，不要乱用）
        ph = B.cur_carry or B.fc
        s += "; NO settled carry yet" + (f" ({ph} t{tier(ph)} is a placeholder)" if ph else "") + ": levels now go to someone who will be replaced"
    if FV >= F9LY and re.search(r"觉醒进度\s*\+", skill(n)):
        C = B.cur_carry or B.fc
        if C and B.awk(C) == 0:
            # f9ly（2026-10-16 晚：主C 觉醒后妙音如有剩余，也可以给其他核心英雄使用；673 局 R12~R13 马可波罗觉醒后妙音留着 0.34/0.35）
            s += f"; carry {C} already AWAKENED: no later use for it — give it to another core hero"
    return s


LEVEL_GAIN_RE = re.compile(r"等级\s*[+＋]\s*\d|提升\s*\d+(?:-\d+)?\s*级|等级提升")


# ---------------------------------------------------------------- carry 题：谁当主C（2026-09-25，影子题）
# 设计：出足够多的题，让 Laya 决策是否需要换主C；要结合刷出升级卡和升级效果卡的概率去判断；
# "1-2 阶英雄作为主C 技能都太弱"；"后期你是抽不到同名英雄用于升级的"。
# 事实只写"是什么"：品阶、强度分、等级、觉醒进度、**每次刷新看到同名的概率（现在 / 升一级后）**、装备、枫叶舞绑在谁、换过去要付出什么。
LEVEL_MOVE_CARDS = ("夺取之术", "众星拱月")
LEVEL_MOVE_CARDS_F8 = ("遗志", "传承")          # f8：天赋送的搬等级牌（卡面：移除并转移等级 / 10 级以上的等级分给 3 人）
MAPLE_BOOST_EN = {                             # f8：影响换主C 代价/收益的枫叶舞强化（卡面原意）
    "红叶染锋": "maple hero leaving keeps 50% of its real level for the next maple hero, once",
    "翩跹舞步·新生": "maple hero +2 lv per round",
    "翩跹舞步·蓄势": "maple hero +1 permanent lv per kill",
    "翩跹舞步·进化": "maple hero +4 atk per kill this match",
}


def carry_candidates(B: "Board", cur: str) -> list:
    """场上能当主C 的：输出位、强度分 ≥5；当前主C 不论分数都列上。按强度分、等级排，最多 5 个。"""
    c = [n for n in B.heroes if is_output(n) and (fit(n) or 0) >= 5]
    if FV >= F9LW:
        c = [n for n in c if not lord_xp_unit(n)]          # f9lw ㊳：弈星是纯过渡件，不当主C 人选（第 655 局题面一直列着他）
    if cur and cur in B.heroes and cur not in c:
        c.append(cur)
    if FV >= F9ZA:
        # f9za（10-21 待办：换主C 题候选读 carry_pool）：关掉的 2 阶以上主C 不当候选（1 阶临时主C 照常）；
        # 同线上 route_avoid_weak：删完只剩"维持空缺"时不删（主C 位永不留空，第 849 局）
        _keep = [n for n in c if not f9za_carry_off(n)]
        if any(n != cur for n in _keep) or (cur and cur in _keep):
            c = _keep
    c.sort(key=lambda n: (n == cur, fit(n) or 0, B.lv(n) or 0, n), reverse=True)
    return c[:5]


def carry_option(n: str, B: "Board", cur: str, maple: str) -> list[str]:
    """结论前置、每条短（选项 ≤44 token，放不下的整条被丢）：身份 → 成长空间 → 换人代价 → 装备/枫叶舞 → 技能概要。
    2026-09-25：要考虑主C 位置一直占着是否还有升级空间、是否应该早早让给其他人。"""
    m = meta()
    lv = B.lv(n)
    items = [f"t{tier(n)} fit{fit(n) if fit(n) is not None else '?'} lv{lv or '?'}"]
    if FV >= "f8":
        return items + _carry_option_f8(n, B, cur, maple)
    nxt = next((t for t in SKILL_THRESHOLDS if lv is not None and lv < t), None)
    if lv is not None:
        items.append(f"skill jump lv{nxt} (+{nxt - lv})" if nxt else "all skill jumps done")
    q, L = tier(n), B.level
    if B.awk(n) == 0:
        items.append("AWAKENED")
    elif q and L:
        k = B.awk(n)
        a = f"awk {3 - k}/3" if k is not None else "awk ?"
        if (m.TIER_ODDS.get(L) or {}).get(q, 0.0) <= 0:
            items.append((f"{a}, t{q} no longer in shop at lv {L}" if L >= m.min_level_for(q) else f"{a}, t{q} not in shop until lv {m.min_level_for(q)}") if FV >= "f9" else f"{a}, copies gone")
        else:
            items.append(f"{a}, copy {m.shop_hit_prob(q, L, B.pool(n)):.0%}/refresh")
    if n != cur and maple and maple != n and maple in B.heroes:
        items.append(f"cost: sell {maple} lv{B.lv(maple) or '?'}" + _part_note(maple, B))
    if n != cur and cur and cur in B.heroes and (B.eq(cur) or 0) > 0:
        items.append(f"{cur} keeps {B.eq(cur)} items")
    e = B.eq(n)
    items.append(f"items {e if e is not None else '?'}/3" + (", maple here" if maple and maple == n else ""))
    items.append(fit_why(n))
    return items


def _carry_option_f8(n: str, B: "Board", cur: str, maple: str) -> list[str]:
    """f8 选项顺序：代价 → 成长（觉醒 / 同名命中）→ 技能质变点 → 装备 / 枫叶舞 → 技能概要。
    老师 r8b：名字长时 f7 排第 4 的 cost 被 44 token 预算整条丢掉（856ad0bb630f）；代价与"旧主C 的装备"合成一条。"""
    m = meta()
    lv = B.lv(n)
    out = []
    if FV >= F9LW and maple and maple in B.heroes and B.maple_on_temp(maple):
        # f9lw ㊱（2026-10-16：就算没有备份也该卖掉临时主C 换绑，主C 才吃得到等级）
        if n == B.fc:
            out.append("maple moves to the future carry (stops wasting maple levels)")
        elif n == maple:
            out.append("maple stays on a TEMPORARY carry: its levels are wasted")
    if FV >= F9LW and n != cur:
        olc = B.other_lineup_carry(n)
        if olc:
            out.append("OTHER LINEUP's carry (not in core team)")
    if FV >= F9LX and n != B.fc and ((tier(n) or 9) <= 1 or n == B.temp_carry) and B.feeds_or_part(n):
        # 2026-10-16：第一回合可以给李元芳绑上；一般给明确可以卖掉的临时主C 绑定，最终主C 到了直接卖
        out.insert(0, "MULTI-USE unit (feeds the carry / engine part): as maple host it must be SOLD when the final carry comes")
    if FV >= F9LL and n != cur:
        gg = next((g for g, _w in held_god_gear(B) if god_gear_fits(g, n)), None)
        if gg:
            out.append(f"can wear held t5 {gg} (current carry can't)")
    if FV >= F9LK:
        if n == cur and (tier(n) or 9) <= 1 and B.rnd > 5 and not (B.route and B.route.get("carry") == n):
            out.append("t1 carry past early game (low skill ceiling; awakened or not)")
    elif (FV >= "f9f" and n == cur and (tier(n) or 9) <= 2 and B.rnd > 6
            and not (FV >= "f9k" and ((fit(n) or 0) >= 8 or (B.route and B.route.get("carry") == n)
                                      or B.awk(n) == 0 or ((fit(n) or 0) >= 7 and B.engine_role(n) == "payoff")))):
        # f9f（2026-09-26：孙尚香这个主C 太弱，很靠后的回合还在用 1 阶主C）：品阶上限写在最前（不能被截掉）
        # f9k：评分 ≥8 的低阶主C（百里守约 2 阶，阿离最热门的前 4 套里 3 套是它，108 万人用）和锁定路线的主C 不写 —— 第 30 节原来说错了
        out.append(f"t{tier(n)} carry past early game (low skill ceiling)")
    if FV >= "f9j" and B.engine_started:
        r = B.engine_role(n)
        out.append(f"engine payoff of {B.engine['name']}" if r == "payoff"
                   else (f"engine {r} (not its payoff)" if r else f"not in engine {B.engine['name']}"))
    if FV >= "f9g" and n == cur and B.ctx.get("carry_since") is not None:
        # f9g（第四轮第 9 局：一局换了 5 次主C，干将莫邪→米莱狄→干将莫邪→嬴政→海诺）：写出现任的来历，让学生看见"刚换过"
        since, prev, k = B.ctx.get("carry_since") or 0, B.ctx.get("carry_prev") or "", B.ctx.get("carry_switches") or 0
        if B.rnd and since > B.rnd:
            since = 0                           # 记录串局了（不可能晚于当前回合）
        if prev == n:
            prev = ""                           # 记账里的"现任"和题面的现任对不上时，别写"替掉了自己"
        h = f"carry since r{since}" if since else "carry"
        if since and B.rnd and B.rnd - since <= 1:
            h += " (JUST switched)"
        if prev:
            h += f", replaced {prev}"
            if FV >= "f9h" and prev in B.heroes:
                # f9h（r9j 老师：露娜 lv166 觉醒被刚上场的 4 级海诺替掉，第 31 节却要 keep）：写出被替掉的人现在多强
                h += f" (still on board: t{tier(prev)} lv{B.lv(prev) or '?'}{' AWAKENED' if B.awk(prev) == 0 else ''})"
        if k:
            h += f", {k} switch{'es' if k > 1 else ''} this game"
        out.append(h)
    if n != cur:
        # 装备的去向对每个选项都一样，写在题面里一次（carry_state_extra），选项里只写要卖谁
        out.append(f"cost: sell {maple} lv{B.lv(maple) or '?'}" + _part_note(maple, B) if (maple and maple != n and maple in B.heroes)
                   else "cost: none")
    elif FV >= "f9" and maple and maple != n and maple in B.heroes:
        # f9：维持现任也要卖掉绑定对象才能把枫叶舞挪过来（老师 r8f8：keep 看起来像零代价）
        out.append(f"to move maple here: sell {maple} lv{B.lv(maple) or '?'}")
    q, L = tier(n), B.level
    if B.awk(n) == 0:
        out.append("AWAKENED")
    elif q and L:
        k = B.awk(n)
        a = f"awk {3 - k}/3" if k is not None else "awk ?"
        if (m.TIER_ODDS.get(L) or {}).get(q, 0.0) <= 0:
            out.append((f"{a}, t{q} no longer in shop at lv {L}" if L >= m.min_level_for(q) else f"{a}, t{q} not in shop until lv {m.min_level_for(q)}") if FV >= "f9" else f"{a}, copies gone")
        else:
            out.append(f"{a}, copy {m.shop_hit_prob(q, L, B.pool(n)):.0%}/refresh")
    nxt = next((t for t in SKILL_THRESHOLDS if lv is not None and lv < t), None)
    if lv is not None:
        out.append(f"skill jump lv{nxt} (+{nxt - lv})" if nxt else "all skill jumps done")
    e = B.eq(n)
    out.append(f"items {e if e is not None else '?'}/3" + (", maple here" if maple and maple == n else ""))
    out.append(fit_why(n))
    return out


def carry_state_extra(B: "Board") -> str:
    m = meta()
    L = B.level
    hit = getattr(m, "LEVEL_CARD_HIT", {}).get(L) if L else None
    moves = LEVEL_MOVE_CARDS + (LEVEL_MOVE_CARDS_F8 if FV >= "f8" else ())
    tools = [c for c in (dict.fromkeys(B.hand) if FV >= "f8" else B.hand) if c in moves]
    s = f"level-up cards: {hit:.0%} per refresh at lv {L} (any carry can use them)" if hit is not None else "level-up cards: ?"
    s += ("; level-transfer in hand: " + ", ".join(tools) if tools else "; no level-transfer card in hand")
    if FV >= "f8":
        lb = [b for b in (B.ctx.get("lord_boosts") or []) if b in MAPLE_BOOST_EN]
        if lb:
            s += "; maple boosts: " + ", ".join(f"{b} ({MAPLE_BOOST_EN[b]})" for b in lb)
        if B.fc_in_hand:
            s += (f"; {B.hand_carry_label()} still in HAND: {B.fc_in_hand} (t{tier(B.fc_in_hand)} fit {fit(B.fc_in_hand)}/9, "
                  "not an option until deployed)")
        cur, maple = B.ctx.get("carry") or "", B.ctx.get("maple") or ""
        if cur and maple and maple != cur and maple in B.heroes:
            s += f"; note: maple is on {maple}, not on the current carry {cur}"
        if maple and maple in B.heroes and (B.eq(maple) or 0) > 0:
            s += f"; selling {maple} returns its {B.eq(maple)} items to hand"
        if cur and cur != maple and cur in B.heroes and (B.eq(cur) or 0) > 0:
            s += f"; {cur}'s {B.eq(cur)} items move only if {cur} is sold"
    return s


# ---------------------------------------------------------------- swap 题（人口满：卖场上谁给手里的英雄腾位 / 先不换）
# 2026-09-25：逐步去掉规则引擎，用更充分的信息与选项或者 YesOrNo 去训练 Laya。
# 规则引擎原来按 unit_value 卖"最弱的"（门槛 SWAP_MARGIN），历史上 137 次 —— 现在交给学生判。
# 当前主C 不作为选项（换主C 是 carry 题的事；卖主C 不可逆）。
def swap_incoming(B: "Board") -> str | None:
    """手里等着上场的英雄里最该上的那个：觉醒卡 > 强度分 > 品阶（名字兜底，保证逐字确定）。"""
    awk = set(B.ctx.get("hand_awk") or [])
    skip = set(B.ctx.get("_swap_skip") or ())
    # **只有线上会带 _swap_skip**（本回合已经判过"不换"的等位英雄；离线数据集没有这个键，题面渲染不变）：
    # 2026-10-16 第 668 局：R5~R9 每回合只问刘禅、判不换后就不再问，同在手里的路线核心盾山 / 程咬金一次都没轮到
    hh = [n for n in dict.fromkeys(B.hand_heroes) if n not in skip]
    pref = [n for n in hh if n in set(B.ctx.get("_swap_pref") or ())]
    if pref:
        # **只有线上会带 _swap_pref**（路线当前阶段要的人；离线数据集没有这个键，题面渲染不变）：
        # 2026-09-30 第 14 局：钟馗/明世隐/太乙/盾山压在手里，换人题一直问强度分更高的司空震/蒙犽
        return sorted(pref, key=lambda n: (n in awk, B.hand.count(n) >= 2, fit(n) or 0, tier(n), n), reverse=True)[0]
    if FV >= "f8":
        # f8：手里同名 ≥2 张 = 上场后马上能合成（老师 r8b：艾琳 x5 不进比较）
        xs = sorted(hh, key=lambda n: (n in awk, B.hand.count(n) >= 2, fit(n) or 0, tier(n), n), reverse=True)
        return xs[0] if xs else None
    xs = sorted(hh, key=lambda n: (n in awk, fit(n) or 0, tier(n), n), reverse=True)
    return xs[0] if xs else None


def swap_candidates(B: "Board", cur: str) -> list:
    return sorted((n for n in B.heroes if n != cur), key=lambda n: (tier(n), B.lv(n) or 0, n))[:7]


def sell_hand_candidates(B: "Board", cur: str) -> list:
    """f9lx sell_hand 题（2026-10-16：最终阵容已定，海月这个别家主C 不该留；及时卖掉路线不需要的英雄）：
    手里的英雄牌（不在场、不是场上单位的合成材料、不是主C / 未来主C）。只列事实上的候选，卖不卖由 Laya 判。"""
    out = []
    for n in dict.fromkeys(B.hand_heroes):
        if n in (cur, B.fc) or n in B.field:
            continue
        if n in set(B.ctx.get("hand_awk") or []):
            continue                       # 觉醒卡在手里卖不掉（只能先上场）
        if "转瞬" in (skill(n) or ""):
            continue
        out.append(n)
    return out[:4]


def sell_filler_candidates(B: "Board", cur: str) -> list:
    """f9lw sell_filler 题（㊳㊴）：场上没人等位时也该卖的"占位件"——回本的弈星、过了阶段的开局件、别家主C。只列事实上属于这三类的人。"""
    if not isinstance(B.free, int) or B.free > 0:
        return []          # 老师 r9zn 多批：已有空位时卖了只多空一格；本意是卖掉腾出位子 → 只在满员时问
    out = []
    for n in B.heroes:
        if n in (cur, B.fc) or not is_hero(n) or B.feeds_or_part(n):
            continue
        ids = B.identity(n)
        if any(("PAID OFF" in t) or t.startswith(("ROUTE early-only", "OTHER LINEUP")) for t in ids):
            out.append(n)
    return sorted(out, key=lambda n: (B.lv(n) or 0, n))[:4]


def swap_option(n: str, B: "Board", maple: str) -> list[str]:
    """结论前置：身份 → 等级/品阶 → 觉醒进度 → 卖掉的附带后果（装备回手 / 枫叶舞解绑）。"""
    if FV >= "f8":      # f8：保留 "(past it)"（老师：过渡主C 过没过前期是卖不卖的依据），去掉括号里的长解释
        ident = [t.replace(": early game only", "").split(" (fit")[0].split(" (t")[0].split(" (≥")[0] for t in B.identity(n)]
    else:
        ident = [t.split(" (")[0] for t in B.identity(n)]
    items = [f"t{tier(n)} lv{B.lv(n) or '?'} {role(n) or '?'} {row(n) or '?'}"]
    if FV >= F9LY and summon_fact(n) and n not in (B.fc, B.cur_carry):
        items.insert(0, summon_fact_ctx(n, B))
    if FV >= "f9i" and B.rnd and B.v(n).get("in_rd") == B.rnd:
        # f9i（第 12 局 R14：刘禅→雅典娜，7 秒后又卖刚上场的雅典娜给太乙真人）：卖刚上场的人 = 这回合的上场白做
        items.insert(0, "JUST deployed this round")
    if FV >= "f9j" and B.engine_started and B.engine_role(n) in ("producer", "amplifier"):
        if FV >= F9LV:
            items.insert(0, B.engine_part_text(n, B.engine_role(n)))
        else:
            items.insert(0, f"ENGINE PART ({B.engine_role(n)} of {B.engine['name']}): selling breaks the level engine")
    if FV >= F9LJ:
        t = min_engine_tag(n, B, on_board=True)
        if t:
            items.insert(0, t)
    if (FV >= F9T and n != "雅典娜" and "整备" in (skill(n) or "") and fac(n) == norm_fac("日落海")
            and ("雅典娜" in B.field or "雅典娜" in B.hand_heroes)):
        # f9t（2026-10-19 第 755 局 R12：卖米莱狄上雅典娜，应卖云中君）：整备型日落海英雄会被雅典娜开团再触发
        items.insert(0, "整备 re-triggered by 雅典娜 (core engine)")
    if FV >= "f9d":
        syn = carry_synergy(n, B)      # f9d：卖掉它会不会断了主C 的养料
        if syn:
            items.append(syn)
    if FV >= F9X:
        tg = f9x_trigger_fact(n, B)    # f9x（54.26-18）：条件触发型数来源、全队效果写 team-wide（第 791 局卖云中君留朵莉亚）
        if tg:
            items.insert(0, tg)
        ss = f9x_shaosiyuan_fact(n, B)     # f9x（54.26-24）：放最前（预算有限时召唤图腾那条可能被挤掉，结论优先）
        if ss:
            items.insert(0, ss)
    if FV >= "f9":
        # f9：卖掉的附带后果排在身份前（老师 r8f8aug：名字长时"maple dance unbinds"被 44 token 预算截掉）
        if maple and maple == n:
            items.append("maple unbinds (can rebind)")
        if B.eq(n):
            items.append(f"{B.eq(n)} items back to hand")
        _offroute = (FV >= F9LY and B.route and not B.route_stale
                     and n not in set(_routes().final_team(B.route)) | set(_routes().members(B.route)))
        if n in B.front_tanks and len(B.front_tanks) <= 2 and not _offroute:
            # f9ly（2026-10-16 晚：这种保护没有意义，重要的是凑套路阵容；主T、唯一前排也应该在套路阵容里找）：路线外单位不写前排保护
            _in = B.ctx.get("_swap_in") if FV >= F9LV else None
            if _in and is_front_tank(_in):
                # f9lv（r9zl 老师 ⑨：等上场的人自己是前排时，"最后 2 个前排"的保护不成立）
                items.append(f"front tank, but incoming {_in} is front too (front tanks stay {len(B.front_tanks)})")
            else:
                items.append(f"one of only {len(B.front_tanks)} front tanks")
    if FV >= F9LK and items and (items[0].startswith("ENGINE PART") or (FV >= F9LV and items[0].startswith("redundant"))):
        ident = [t for t in ident if not t.lower().startswith(("engine part", "redundant"))]     # f9lk：已在最前写过，不重复占预算
    if FV >= F9LW and any(t.startswith("TRANSITIONAL filler") for t in ident):
        # f9lw ㊳：弈星这条放最前、写短（原文太长被 pack 挤掉）
        paid = (B.filler_paid(n) or 0) >= 1
        items.insert(0, "PAID-OFF filler (helps no hero): sell now" if paid else "TRANSITIONAL filler (helps no hero)")
        ident = [t for t in ident if not t.startswith("TRANSITIONAL filler")]
    items += ident[:2]
    if is_output(n) and fit(n) is not None:
        items.append(f"fit {fit(n)}/9")
    items.append(B.awaken_str(n).replace("awaken ", "awk "))
    e = B.eq(n)
    if e and FV < "f9":
        items.append(f"{e} items go back to hand")
    if maple and maple == n and FV < "f9":
        items.append("maple dance unbinds (can rebind)")
    if B.anch["mates"].get(n) and n != B.anchor:
        items.append(f"in {round(B.anch['mates'][n] * 100)}% of {B.anchor} lineups")
    if FV >= F9ZA:
        fu = f9za_sacrifice_fuel(n, B)          # f9za（54.31-3）：守约线带牺牲的英雄 = 主C 法力燃料（卖了断燃料）
        if fu:
            items.insert(0, fu)
    if FV >= F9ZB:
        zb = f9zb_swap_lines(n, B)              # f9zb（54.36-2/3/5/7）：明世隐叠层、李元芳 vs 太乙、卖外人上太乙、同路线前排互换
        items = zb + [x for x in items if x not in zb]
    return items


def swap_head(x: str, B: "Board") -> str:
    awk = x in set(B.ctx.get("hand_awk") or [])
    s = f"{x} t{tier(x)} {role(x) or '?'} {row(x) or '?'}"
    if is_output(x) and fit(x) is not None:
        s += f" fit {fit(x)}/9"
    s += " (AWAKENED card, returns at its old level)" if awk else " (new card, enters at lv 1)"
    if FV >= "f8" and B.hand.count(x) >= 2:
        s += f", {B.hand.count(x)} copies in hand (merge right after deploying)"
    if B.anch["mates"].get(x):
        s += f", in {round(B.anch['mates'][x] * 100)}% of {B.anchor} lineups"
    if FV >= F9LJ:
        t = min_engine_tag(x, B, on_board=False)
        if t:
            s += "; " + t
    if FV >= F9X:
        ss = f9x_shaosiyuan_fact(x, B)      # f9x（54.26-24）：攻速主C 时等位的少司缘是补位件
        if ss:
            s += "; " + ss
    if FV >= F9ZA:
        for _t in (f9za_bought_note(x, B), f9za_sacrifice_fuel(x, B)):     # f9za（54.31-2 / 54.31-3）：自己刚买的 / 守约燃料
            if _t:
                s += "; " + _t
    if FV >= F9ZB:
        s += f9zb_swap_head(x, B)               # f9zb（54.36-3）：第 12 回合起新到的明世隐只转 +2
    return s


# ---------------------------------------------------------------- f9d：英雄效果对当前主C 的"化学反应"
# 2026-09-25：很多英雄牌自带效果，可以和主C 产生化学反应，推荐阵容里的组合大多也是通过这种化学反应养起主C 的，
# 这一点你没有考虑进去"。原来题面只有零散的效果和"在主C 阵容里的比例"，没把"这张牌能不能喂到主C"算出来。
# 只用知识库（tools/cards/kb → data/card_kb.json）的结构化字段：trig / gives[res,n,to,per] / needs / routes，不写建议。
SYN_RES = {"等级": "lv", "临时等级": "temp lv", "攻速": "atk speed", "法力上限减": "mana cap -", "装备": "items",
           "复生": "revive", "暴击": "crit", "护盾": "shield", "法力": "mana", "伤害": "dmg", "生命上限": "max hp",
           "免控": "CC immune", "攻击力": "atk"}
SYN_FAC = {"河洛": "河洛", "三分": "三分之地", "大河": "大河流域", "日落海": "日落海", "逐鹿": "逐鹿"}
SYN_ROUTE_SKIP = {"防守", "单核", "无阵营", "输出装", "经济", "找牌", "刷新", "卖人"}


def _syn_target(to: str, n: str, C: str, B: "Board") -> tuple[bool | None, str]:
    """这条效果的对象会不会落到主C 身上 -> (True/False/None=看情况, 条件说明)。"""
    t = to or ""
    if not t or any(k in t for k in ("核心", "召唤", "图腾", "手牌", "棋手", "敌人", "古币的目标", "护盾被击破")):
        return False, ""
    if "等级最低" in t:
        return False, "goes to lowest-level allies"
    if t.startswith("自身") and "及" not in t:
        return (True, "") if n == C else (False, "")
    for k, full in SYN_FAC.items():
        if k in t:
            ok = fac(C) == full or norm_fac(fac(C)) == norm_fac(full)
            return ok, "" if ok else f"needs {FACTION_EN.get(full, full)} (carry is {fac_en(C)})"
    if "闪现" in t:
        ok = C in B.blink
        return ok, "" if ok else "needs blink hero"
    if any(k in t for k in ("最近", "左右 1 格")):
        if n in B.field and C in B.field:
            adj = adjacent(B.cell(n), B.cell(C))
            if adj is True:
                return True, "adjacent now"
            if adj is False:
                return None, "only if placed next to carry"
        return None, "place next to carry"
    if any(k in t for k in ("放技能", "合成", "等级最高", "所有", "全体", "每名", "其他英雄", "随机", "指定")):
        return True, ""
    return None, ""


def carry_synergy(n: str, B: "Board") -> str | None:
    """n 的效果对当前主C 的作用（一句话，结论前置）；和主C 无关就返回 None（题面不加字）。"""
    C = B.ctx.get("carry") or B.fc
    _c0 = C
    if FV >= F9ZA:
        C = f9za_syn_carry(B, C)        # f9za-fix（54.32-16）：现任主C 在 carry_pool 关着、开着的主C 在场 → 按开着的主C 写
    if not C or n == C or not is_hero(n) or not is_hero(C):
        return None
    k = kb().get(n) or {}
    parts, relevant = [], False
    for g in k.get("gives") or []:
        res = SYN_RES.get(g.get("res"))
        if not res:
            continue
        hit, why = _syn_target(g.get("to"), n, C, B)
        if hit is False and not why:
            continue
        amt = g.get("n")
        amt = amt if isinstance(amt, int) else (str(amt)[:14] if amt else "")
        per = (g.get("per") or k.get("trig") or "")[:10]
        s = f"+{amt} {res}" + (f" {per}" if per else "")
        if hit is False:
            parts.append(f"no effect on {C}: {why}")
        else:
            relevant = True
            parts.append(f"{C} {s}" + (f" ({why})" if why else ""))
    for nd in (k.get("needs") or []) if relevant else []:     # 前提只在效果本身和主C 有关时才写
        r = nd.get("res") if isinstance(nd, dict) else nd
        if r == "临时等级" and not B.temp_src:
            parts.append("idle: no temp-lv source on board")
        elif r == "图腾" and not any(any(gg.get("res") == "图腾" for gg in (kb().get(h) or {}).get("gives", [])) for h in B.heroes):
            parts.append("idle: no totem on board")
    shared = [r for r in (k.get("routes") or []) if r in set((kb().get(C) or {}).get("routes") or []) and r not in SYN_ROUTE_SKIP]
    if shared and not (FV >= F9N and B.destroys_own_totems(n)):
        parts.append(f"same build as {C}: {shared[0]}")
    if FV >= F9LR:
        rn = reactor_note(n, B)
        if rn.startswith("idle") or rn.startswith("its "):
            parts.insert(0, rn)
        if EFFECT_FED_RE.search(skill(C) or ""):
            for g in k.get("gives") or []:
                if g.get("res") in FUEL_RES:
                    _zl = f9z_tactic_zhulu_only(g) if FV >= F9Z else ""
                    if _zl and fac(C) != "逐鹿":
                        # f9z（待办：白起被标成沈梦溪的 steady carry FUEL）：卡面限【逐鹿】的战术牌打不到非逐鹿主C 身上
                        parts.insert(0, f"gives {_zl}: 【逐鹿】-only tactic card -> no fuel for {C}")
                        break
                    parts.insert(0, "steady carry FUEL (effect card every round)" if "每回合" in str(g.get("per") or "")
                                 else f"carry fuel only {g.get('per') or 'sometimes'}")
                    break
    lab = "with current carry: " if FV >= "f9e" and B.fc and C != B.fc else "with carry: "
    if FV >= F9ZA and C != _c0:
        lab = "with carry: "            # f9za-fix（54.32-16）：读数已改按开着的主C 写，不再叫 current carry
    return (lab + "; ".join(parts[:2])) if parts else None


# ---------------------------------------------------------------- f9d：锁定商店（2026-09-25：钱不够时判断是否锁定商店）
def lock_parts(B: "Board", full: list) -> tuple[str, list[str]]:
    """锁店 = 停手并把现在这 4 张留到下回合。写出其中值得留的牌（合成 / 主C 阵容 / 强度分高 / 喂主C）和下回合的钱。"""
    m = meta()
    g = B.ctx.get("gold")
    inc = m.energy_income(B.rnd + 1) if B.rnd else None
    keep = []
    for n in dict.fromkeys(x for x in full if x):
        tag = None
        if is_hero(n):
            if n in B.field or n in B.hand:
                tag = "merge"
            elif B.anch["mates"].get(n, 0) >= 0.3 or n == B.anchor:
                tag = f"carry lineup {round(B.anch['mates'].get(n, 1) * 100)}%"
            elif (fit(n) or 0) >= 7 and is_output(n):
                tag = f"fit {fit(n)}/9"
            elif FV >= "f9e":
                syn = carry_synergy(n, B) or ""
                if syn and "idle" not in syn and "no effect" not in syn.split(";")[0]:
                    tag = "feeds carry"
            elif FV >= "f9d" and (carry_synergy(n, B) or "").startswith("with carry: " + (B.ctx.get("carry") or B.fc or "?")):
                tag = "feeds carry"
        if tag:
            if FV >= "f9e" and is_hero(n):       # f9e：老师 r9d："keeps 狄仁杰 (fit 7/9)" 在候选详情里查不到这张牌
                tag = f"t{tier(n)} {role(n) or '?'}, {tag}" + (", EXTRA OUTPUT" if n not in B.field
                                                            and B.is_extra_output(n) else "")
            keep.append(f"{n} ({tag})")
        elif FV >= F9LS and not is_hero(n) and cat(n) in ("效果牌", "装备牌"):
            # f9ls：锁店只列英雄 → 老师看不到"锁住的是跳级 / 装备"（第 636 局第 9 回合锁店留下的跳级没写出来）
            keep.append(f"{n} ({'effect card' if cat(n) == '效果牌' else 'item'})")
    head = "lock the shop and stop: "
    nxt = f"next round {g + inc if isinstance(g, int) and inc else '?'}e"
    if FV >= "f9f":
        # f9f：逐张放（老师 r9g：一整串 keeps 超出预算时整条被丢，只剩空的 "lock the shop and stop: "）
        if not keep:
            return head, ["nothing notable in this shop", nxt]
        return head, [f"keeps {keep[0]}", nxt] + keep[1:3]
    items = [("keeps " + ", ".join(keep[:3])) if keep else "nothing notable in this shop", nxt]
    return head, items


# ---------------------------------------------------------------- f9k：选路线（route 题）
def route_candidates(B: "Board", banned: str = "") -> list:
    """可选的官方路线（与规则层同一套排除 / 排名，agent/decision/routes.py），最多 3 条。"""
    R = _routes()
    _weak = (lambda lu: False)
    if FV >= F9Z:
        # f9z（review-carry-route 第 5 条，827a / 823 R4~R8：先截 3 条再删屏蔽路线 → 只剩 flex，Laya 只能答"先不定"）：
        # 屏蔽路线（WEAK_ROUTES 前缀 / WEAK_CARRIES 主C）先排除再取前 3（同主C 的下一条自然补位）；现任主C（≥2 阶）就是该路线主C 时不排除（54.28-12）
        _cur = B.ctx.get("carry") or ""
        _cur = _cur if (tier(_cur) or 0) >= 2 else ""
        _weak = (lambda lu: (str(lu.get("name") or "").startswith(WEAK_ROUTES_F9Z) or lu.get("carry") in WEAK_CARRIES_F9Z)
                 and not (_cur and lu.get("carry") == _cur))
        if FV >= F9ZA:
            _weak = f9za_route_blocked      # f9za：读棋手配置（关掉的主C / blocked 阵容），先排除再取前 3；不给现任主C 开例外（10-21 已决定）
        cands, seen_c = [], set()
        for lu in R.candidates(B.owned, banned=banned, top=10 ** 6, one_per_carry=False):    # 同排序；one_per_carry 在排除后自己做
            if _weak(lu) or lu.get("carry") in seen_c:
                continue
            seen_c.add(lu.get("carry")); cands.append(lu)
            if len(cands) >= 3:
                break
    else:
        cands = R.candidates(B.owned, banned=banned, top=3, one_per_carry=True)   # 三套都是同一主C 的变体时学生分不出
    if FV >= F9W and not cands:
        # f9w（第 785 局：禁河洛 + 只走 2~3 阶主C → 官方路线全被筛掉，R3~R5 路线题"候选 0 条"，1 阶干将占主C 到 R5；
        # 虞姬和马可波罗差不多，看核心阵容谁先凑齐）：官方为空时补自定义路线（主C 品阶合规、阵营没被禁），按已拥有的件排
        _tiers = getattr(R, "CARRY_TIERS", None)
        own = [lu for lu in R.custom_lineups() if R.viable(lu, banned)
               and (not _tiers or (tier(lu.get("carry") or "") or 0) in _tiers) and not _weak(lu)]
        own.sort(key=lambda lu: (-R.fit_score(lu, B.owned), lu.get("name") or ""))
        seen_c, cands = set(), []
        for lu in own:
            if lu.get("carry") in seen_c:
                continue
            seen_c.add(lu.get("carry")); cands.append(lu)
            if len(cands) >= 3:
                break
    _c = next((x for x in (B.cur_carry, B.fc) if x and x != getattr(B, "temp_carry", None) and (tier(x) or 0) >= 2), None)
    if FV >= F9N and _c and not any(lu.get("carry") == _c for lu in cands):
        # f9n（第 693 局：主C 已定为马可波罗，"日落海核心马可波罗"是自定义阵容、不在候选里 → Laya 一直 flex，整局没路线）
        own = [lu for lu in tuple(R.ali_lineups()) + tuple(R.custom_lineups(include_disabled=FV < F9Q))
               if lu.get("carry") == _c and R.viable(lu, banned) and (lu.get("custom") or not R.excluded(lu)) and not _weak(lu)]
        if own:
            best = max(own, key=lambda lu: R.fit_score(lu, B.owned))
            cands = (list(cands[:2]) + [best]) if len(cands) >= 3 else (list(cands) + [best])
    return cands


def route_option(lu: dict, B: "Board") -> list[str]:
    """一条路线对我方意味着什么 —— 结论在前、写短（选项 ≤44 token，中文阵容名本身就占 10+）：主C 有没有、凑了几个、缺谁。"""
    R = _routes()
    carry = lu.get("carry") or "?"
    fin, stg = R.final_team(lu), R.stage_units(lu)
    have = [h for h in fin if h in B.owned]
    miss = [h for h in fin if h not in B.owned and h != carry]      # 主C 在第一条里已经说了
    st = f"lv{B.lv(carry) or '?'} on board" if carry in B.field else ("in hand" if carry in B.hand else "not owned")
    q = tier(carry) if carry != "?" else 0
    m = meta()
    odds = (m.TIER_ODDS.get(B.level) or {}).get(q, 0.0) if B.level else None
    items = [f"carry {carry} {st}", f"team {len(have)}/{len(fin)}", f"t{q} fit {fit(carry)}/9",
             f"{lu.get('use_num') or 0} players",          # 老师 r9m3：放最后会被 44 token 截掉，规范又要参考它
             "missing " + (", ".join(miss[:3]) or "-")]
    if odds is not None:
        items.append(f"t{q} {odds:.0%} of shop")
    now = [h for h in ((R.stage_now(lu, B.rnd or 0) or {}).get("main") or []) if h]
    if now:
        items.insert(3, f"plan now {sum(1 for h in now if h in B.owned)}/{len(now)}")
    if FV >= F9V:
        # f9v（2026-10-19，740~776 局统计：守约系均名次 ~2.1、前三 8/9；伞亚连 5.4、0/5）：本系统打这条路线的历史名次（冻结快照，逐字确定）
        rs = (_route_stats().get(lu.get("name") or "") or {})
        items.insert(2, f"ours avg#{rs['avg']:.1f} n{rs['n']}" if rs else "ours n0")
    return items


_ROUTE_STATS = None


def _route_stats() -> dict:
    global _ROUTE_STATS
    if _ROUTE_STATS is None:
        f = Path(__file__).resolve().parents[2] / "data" / "laya" / "route_stats_f9v.json"
        _ROUTE_STATS = json.loads(f.read_text(encoding="utf-8")).get("routes", {}) if f.exists() else {}
    return _ROUTE_STATS


def route_state_extra(B: "Board") -> str:
    return (f"Round {B.rnd}, lord level {B.level or '?'}. Committing locks the plan for the rest of the game: "
            "shop buys, carry choice and selling all follow the route's members. "
            "A route whose carry and core you cannot get wastes the game; staying flexible keeps options but drifts.")


# ---------------------------------------------------------------- eco 题（经济节奏：升级 / 停级刷牌 / 攒钱），f9le 起
# 2026-10-09。
# 定了主C 要看配套凑齐没有、都分布在哪个棋手等级；都在 2~3 阶就不该盲目升棋手等级，升到 5 级很难刷出低阶配套。
# 版本号 f9le：字符串比较 "f9l" < "f9le" < "f9m"，所以 f9le = f9l 全部 + eco 题；f9m（待办里的题面修改）将来自然包含它。
# **只用原始材料**：路线阶段成员（agent/decision/routes.py，规则层与题面共用的官方路线数据）、以主C 为锚的官方阵容队友
# （anchored）、meta 的实测品阶概率 / 人口表 / 进度表 / 收入表、卡面原文。不引用规则引擎的任何判断。
# 这些函数只被 features.q_eco 调用（eco 是新题型），不影响任何已有题型的渲染。
ALI_EXCLUSIVE_TALENTS = ("落英汲取", "枫之洪流", "翩跹馈赠")     # 阿离 6 级专属（卡库 lord_related，效果读卡面原文）
ECO_MISSING_MAX = 10


def eco_carry(B: "Board") -> str:
    """eco 题里的"主C"：现任主C（在场或在手）> 场上未来主C > 手里更强的候选。"""
    cc = B.ctx.get("carry") or ""
    if cc and (cc in B.field or cc in B.hand):
        return cc
    return B.fc or B.fc_in_hand or ""


def _eco_hit(q: int, L: int):
    """刷一次商店至少见到某张 q 阶指定英雄的概率（满池近似）；等级 > 6 / 该等级没实测 → None。"""
    m = meta()
    if not q or not L or L > m.MAX_LORD_LEVEL:
        return None
    if not m.can_appear(q, L):
        return 0.0
    if m.tier_odds(q, L) <= 0:
        return None
    return m.shop_hit_prob(q, L)


def eco_missing(B: "Board") -> list:
    """缺的配套 [(英雄, 品阶, 来源)]：路线当前阶段成员（main+sub）+ 以主C 为锚的官方阵容高频队友（出现率 ≥30%），
    去掉已拥有的、主C 本人、被禁派系的（商店刷不出）。有序（先路线、再按队友出现率），保证逐字确定。"""
    R = _routes()
    owned = B.owned
    banned = R.norm_banned(B.ctx.get("banned") or "")
    C = eco_carry(B)
    out: list = []
    seen: set = set(B.nw)               # f9x（54.26-2）：路线不要的人不算缺件（f9x 以前 B.nw 恒空）

    def add(n, src):
        if not n or n in seen or not is_hero(n) or n in owned:     # 主C 本人没到手也算缺（已拥有的由 owned 排除）
            return
        if banned and norm_fac(fac(n)) == banned:
            return
        seen.add(n)
        out.append((n, tier(n), src))
    if B.route and not B.route_stale:
        # 路线主C 还没到手：它本身就是最要找的人，排第一（老师反馈 2026-10-09：缺件表只列 1 阶，"升级更难刷"是假象，
        # 2 阶路线主C 在 1 级根本刷不出来）
        add(B.route.get("carry") or "", "ROUTE CARRY not owned")
        if FV >= F9LN:
            # f9ln（2026-10-13：核心英雄需要多少棋手等级才好刷，应写进题面）：最低配套（等级引擎）缺件排在最前，
            # 单独标出来 —— 原来和官方阵容队友、开局填充件混在一起求和，"哪一级最好刷"被稀释（第 614 局 R6：10 人求和 lv3 最高）
            for g in min_engine_groups(getattr(B, "min_eng", None) or {}):
                if not any(x in owned for x in g):
                    for x in g:
                        add(x, "MIN ENGINE core")
        st = R.stage_now(B.route, B.rnd) if B.rnd else None
        if st:
            span = f"route r{st.get('from_round') or '?'}-{st.get('to_round') or '?'}"
            for h in list(st.get("main") or []) + list(st.get("sub") or []):
                add(h, span)
    if C and not (FV >= F9LN and B.route and not B.route_stale and str(B.route.get("key") or "").startswith("user_")):
        # f9ln：自定义路线（user_*）不用官方阵容队友 —— 官方以马可波罗为主C 的只有 1 套、还不是阿离的（甄姬 / 曹操 / 露娜，审计第 1 条）
        for n, p in anchored(C)["mates"].items():
            if p < 0.3:
                break
            add(n, f"with {C} {round(p * 100)}%")
    return out[:ECO_MISSING_MAX]


def eco_hit_sums(miss: list, L: int) -> dict:
    """{等级: (Σ 命中概率 = 每刷一次期望见到的缺件数, 至少见到 1 个的概率, 有没有未实测的)}，L、L+1、L+2（≤6）。"""
    out = {}
    for lv in range(L, min(L + 2, meta().MAX_LORD_LEVEL) + 1):
        ps = [_eco_hit(q, lv) for _, q, _ in miss]
        known = [p for p in ps if p is not None]
        out[lv] = (sum(known), 1 - math.prod(1 - p for p in known) if known else 0.0, any(p is None for p in ps))
    return out


def eco_level_cost(ctx: dict):
    """升级费：读数 > 0 用读数；读不到用 所需经验 − 当前经验（升级费 = 所需经验 − 当前经验）。"""
    lc = ctx.get("level_cost")
    if isinstance(lc, int) and lc > 0:
        return lc
    x, xn = ctx.get("xp"), ctx.get("xp_needed")
    if isinstance(x, int) and isinstance(xn, int) and xn > x >= 0:
        return xn - x
    return None


def eco_talent_note(L: int) -> str:
    """下一级的天赋：5 级必出金色；6 级解锁阿离专属天赋（写卡面效果 —— 星耀局反馈：学生原来只看到 card not in database）。"""
    if L + 1 == 6:
        effs = []
        for n in ALI_EXCLUSIVE_TALENTS:
            sk = (c(n).get("skill") or "").strip()
            effs.append(f"{n}（{sk}）" if sk else n)
        return "lv 6 unlocks Ali EXCLUSIVE talents (pick 1 of 3): " + " / ".join(effs)
    if L + 1 == 5:
        return "lv 5 talent pick is guaranteed gold"
    return "level-up gives a talent pick (1 of 3)" if L + 1 <= 6 else ""


F9LH = "f9lh"   # f9lh（2026-10-11：阵容凑齐、只差主C 觉醒，高等级商店 2 阶主C 很难刷到 → 不如攒钱升级）：
                # eco 写出 主C 觉醒还差几张 + 各等级每次刷新见到主C 同名的概率、场上已满、攒一回合够不够升级。旧版本一字不变。


def eco_carry_copies(B: "Board"):
    """主C 觉醒还差几张（读数；觉醒了 / 读不到 → None）。"""
    C = eco_carry(B)
    if not C or C not in B.field:
        return None
    k = B.awk(C)
    return k if isinstance(k, int) and k > 0 else None


def eco_state_extra(B: "Board") -> str:
    """eco 题的说明（放 instructions）：血量、主C、进度表、人口、缺的配套逐个列品阶与 L/L+1/L+2 的命中率、合计、天赋。"""
    m = meta()
    ctx = B.ctx
    L = B.level
    hp = ctx.get("hp")
    parts = [f"lord hp {hp if isinstance(hp, int) else '?'}/60 (hp reading can be off)"]
    C = eco_carry(B)
    if C:
        lvC = B.lv(C)
        where = "on board" if C in B.field else "in hand"
        parts.append(f"carry {C} t{tier(C)} lv {lvC if lvC else '?'} ({where})")
        k = eco_carry_copies(B) if FV >= F9LH else None
        if k and L:
            def pct2(p):
                return "?" if p is None else f"{round(p * 100)}%"
            lv3 = list(range(L, min(L + 2, m.MAX_LORD_LEVEL) + 1))
            pl = B.pool(C)
            parts.append(f"carry NOT awakened: needs {k} more cop{'y' if k == 1 else 'ies'}"
                         + (f" ({pl} left in pool)" if isinstance(pl, int) else "")
                         + f"; per-refresh chance to see a {C} copy at lv {'/'.join(map(str, lv3))}: "
                         + "/".join(pct2(_eco_hit(tier(C), x)) for x in lv3))
    else:
        parts.append("carry: none yet")
    if not L:
        parts.append("lord level unread: shop odds unknown")
        return "; ".join(parts) + "."
    tgt = m.level_target(B.rnd) if B.rnd else 0
    if tgt:
        parts.append(f"lord lv {L}, schedule lv {tgt} by round {B.rnd}"
                     + (f" → BEHIND by {tgt - L}" if L < tgt else " → on schedule" if L == tgt else f" → AHEAD by {L - tgt}"))
    cap = ctx.get("pop_cap")
    if L < m.MAX_LORD_LEVEL:
        bonus = (cap - m.pop_cap_at(L)) if isinstance(cap, int) and cap >= m.pop_cap_at(L) else 0
        _after = m.pop_cap_at(L + 1) + bonus
        _chg = ((" (+1)" if isinstance(cap, int) and _after > cap else " (no change)") if FV >= F9LI
                else (" (+1)" if m.level_raises_pop(L) else " (no change)"))     # f9li：按前后两个数字写，不再自相矛盾
        parts.append(f"population cap {cap if isinstance(cap, int) else '?'} now, {_after} after level-up"
                     + _chg
                     + f"; heroes waiting in hand {len(set(B.hand_heroes))}")
    if FV >= F9LH and B.free == 0:
        parts.append(f"board FULL ({ctx.get('pop')}/{cap}): a missing support hero can only join by selling someone on board")
    miss = eco_missing(B)
    lvls = list(range(L, min(L + 2, m.MAX_LORD_LEVEL) + 1))
    if miss:
        def pct(p):
            return "?" if p is None else f"{round(p * 100)}%"
        rows = [f"{n} t{q} [{src}] " + "/".join(pct(_eco_hit(q, lv)) for lv in lvls) for n, q, src in miss]
        parts.append(f"MISSING support ({len(miss)}; per-refresh chance to see each at lv {'/'.join(map(str, lvls))}): "
                     + ", ".join(rows))
        sums = eco_hit_sums(miss, L)
        parts.append("missing seen per refresh (sum): "
                     + ", ".join(f"lv{lv} {sums[lv][0]:.2f}" + ("+?" if sums[lv][2] else "") for lv in lvls)
                     + f"; P(>=1 per refresh) lv{L} {round(sums[L][1] * 100)}%")
        core = [(n, q, src) for n, q, src in miss if FV >= F9LN and src in ("MIN ENGINE core", "ROUTE CARRY not owned")]
        if core:
            cs = eco_hit_sums(core, L)
            best = max(lvls, key=lambda lv: (round(cs[lv][0], 3), -lv))
            parts.append("CORE pieces (" + ", ".join(f"{n} t{q}" for n, q, _ in core) + ") seen per refresh: "
                         + ", ".join(f"lv{lv} {cs[lv][0]:.2f}" for lv in lvls) + f" → best at lv{best}")
        ts = sorted({q for _, q, _ in miss})
        parts.append("missing tiers: " + ", ".join(f"t{q}x{sum(1 for _, x, _ in miss if x == q)}" for q in ts))
    else:
        parts.append("MISSING support: none known (route stage and carry lineup mates all owned, or no carry yet)")
    tn = eco_talent_note(L) if L < m.MAX_LORD_LEVEL else ""
    if tn:
        parts.append(tn)
    return "; ".join(parts) + "."


def eco_levelup_parts(B: "Board"):
    """(选项开头, 结论)；满级 / 等级读不到返回 None（没有这个选项）。"""
    m = meta()
    L = B.level
    if not L or L >= m.MAX_LORD_LEVEL:
        return None
    ctx = B.ctx
    cost, g = eco_level_cost(ctx), ctx.get("gold")
    if isinstance(cost, int) and isinstance(g, int) and cost > g:
        return f"level up to {L + 1} ({cost}e): ", [f"CANNOT AFFORD (have {g}e) → does nothing, same as save"]
    head = (f"level up to {L + 1} ({cost if cost else '?'}e"
            + (f", {g - cost}e left" if isinstance(cost, int) and isinstance(g, int) else "") + "): ")
    items = []
    if FV >= F9LM:
        af = after_level_fill(B, cost, g)
        if af:
            items.append(af)
    miss = eco_missing(B)
    if miss:
        s = eco_hit_sums(miss, L)
        a, b = s[L][0], s[L + 1][0]
        items.append(f"missing seen/refresh {a:.2f}→{b:.2f}" + (" (WORSE)" if b < a - 0.005 else " (better)" if b > a + 0.005 else ""))
    k = eco_carry_copies(B) if FV >= F9LH else None
    if k:
        a, b = _eco_hit(tier(eco_carry(B)), L), _eco_hit(tier(eco_carry(B)), L + 1)
        if a is not None and b is not None:
            items.append(f"carry copy/refresh {round(a * 100)}%→{round(b * 100)}%"
                         + (" (WORSE)" if b < a - 0.005 else " (better)" if b > a + 0.005 else ""))
    if m.level_raises_pop(L):
        items.append("+1 slot" + (f" ({len(set(B.hand_heroes))} waiting in hand)" if B.hand_heroes else ""))
    new_t = [t for t in (m.TIER_ODDS.get(L + 1) or {}) if t not in (m.TIER_ODDS.get(L) or {})]
    if new_t:
        items.append(f"opens t{new_t[0]}")
    tgt = m.level_target(B.rnd) if B.rnd else 0
    if tgt and L < tgt:
        items.append(f"catches up schedule ({tgt - L} behind)")
    if L + 1 == 6:
        items.append("Ali exclusive talent")
    return head, items


def eco_roll_parts(B: "Board"):
    """(选项开头, 结论)；刷不起（没免费、能量 < 2）返回 None。"""
    ctx = B.ctx
    g, free, L = ctx.get("gold"), ctx.get("free_refresh") or 0, B.level
    n = (g // 2 if isinstance(g, int) and g > 0 else 0) + free
    if n <= 0:
        return None
    head = f"stay lv {L or '?'} and refresh ({free} free + 2e each, up to {n}): "
    items = []
    if FV >= F9LI and free:
        items.append(f"{free} free refresh{'es' if free > 1 else ''} cost no energy (compatible with saving)")
    miss = eco_missing(B)
    k = eco_carry_copies(B) if FV >= F9LH else None
    carry_item = None
    if k and L:
        h = _eco_hit(tier(eco_carry(B)), L)
        if h is not None:
            carry_item = f"≈{n * h:.1f} carry copies seen (needs {k})"
    if carry_item and FV >= F9LQ:
        # f9lq（第 631 局 R9 / R10：这一项排在免费刷新那句长话后面，被选项 44 token 预算挤掉，学生只看到"缺件见到 1.9 次"）：
        # 主C 觉醒放到最前
        items.insert(0, carry_item)
    if miss and L:
        s = eco_hit_sums(miss, L)
        items.append(f"≈{n * s[L][0]:.1f} missing sightings if all spent")
        items.append(f"{round(s[L][1] * 100)}% per refresh to see one")
    if carry_item and FV < F9LQ:
        items.append(carry_item)
    if not miss:
        items.append("no missing support known")
    cost = eco_level_cost(ctx)
    if cost:
        items.append(f"level-up still {cost}e (minus xp gained)")
    return head, items


def eco_save_parts(B: "Board"):
    m = meta()
    g = B.ctx.get("gold")
    inc = m.energy_income(B.rnd + 1) if B.rnd else None
    head = f"save: keep {g if g is not None else '?'}e" + (f" (+{inc} next round = {g + inc}e)" if isinstance(g, int) and inc else "")
    items = []
    nxt = [a for a in m.AUCTION_ROUNDS if a > (B.rnd or 0)]
    if B.rnd and nxt and nxt[0] == B.rnd + 1:
        items.append("AUCTION right after this battle")
    if B.free and not B.hand_heroes:
        items.append(f"{B.free} slot{'s' if B.free > 1 else ''} stay EMPTY")
    if FV >= F9LH and isinstance(g, int) and inc and B.level and B.level < m.MAX_LORD_LEVEL:
        cost = eco_level_cost(B.ctx)
        if isinstance(cost, int) and cost > g:
            c2 = cost - 1 if FV >= F9LI else cost          # f9li：回合结束经验 +1，下回合升级费少 1（老师 r9v_g9t 反馈）
            items.append(f"next round {g + inc}e vs level-up {c2}e" + (" (after +1 xp)" if FV >= F9LI else "") + " → "
                         + ("CAN level up next round" if g + inc >= c2 else f"still short {c2 - g - inc}e"))
    return head, items


# ---------------- f9lf（2026-10-10）----------------
# f9lf = f9le 全部 + 三类事实（字符串比较 "f9le" < "f9lf" < "f9m"，f9m 已预留给待办里的题面修改）：
#   a. 商店刷新选项：在 "Ne left" 后面写结论 —— 刷完剩的钱买不买得起一张英雄（3 能量）；
#   b. 发英雄牌（商店 / N 选 1 / 天赋 / 用牌题）：卡池（data/card_graph_v3.json 的 grants_pool 边）有多大、
#      其中几个是阵容要的人（路线当前阶段成员 + 主C 配套，口径同 eco_missing，但不去掉已拥有的 —— 拿到同名能合成）、期望拿到几个；
#   c. 阿离 6 级专属天赋（落英汲取 / 枫之洪流 / 翩跹馈赠）写卡面原文。
# 全部只用原始材料（卡面、图谱卡池、官方路线 / 阵容、感知读数），不引用规则引擎。旧版本（≤ f9le）一字不变。
F9LF = "f9lf"
def NO_HERO_ROUND_RATE(r) -> float:
    """我们自己的对局里"一回合一张英雄牌都没打出（上场 / 合成）"的比例（最近 120 局 states.jsonl 实测，2026-10-15）。"""
    r = int(r or 1)
    return 0.126 if r <= 7 else 0.256 if r <= 11 else 0.175


F9LR = "f9lr"   # f9lr（2026-10-15 第 632 局：场上这个套路里朵莉亚没什么用；古币靠牺牲队友获得，上一场没牺牲就没有古币，下一场就没法用效果卡降法力上限）：
                # ① 反应器（"每当…触发X效果时"）写场上能触发 X 的英雄有几个，0 个 = idle；
                # ② 吃效果牌的主C（"每对X使用1张效果牌…"）写燃料来源：每回合稳定的（整备 古币）/ 有条件的（牺牲 = 阵亡才给）。只用卡面 + 知识库 gives。
REACT_RE = re.compile(r"每当[^，。]{0,12}?触发(整备|登场|牺牲|开团|闪现|退场)效果")
DESTROY_EQ_RE = re.compile(r"摧毁其所有装备")
EFFECT_FED_RE = re.compile(r"每对\S{0,8}?使用1张效果牌")
FUEL_RES = ("古币", "精致古币", "效果牌", "战术牌")


def own_kw(h: str, k: str) -> bool:
    """h 自己有 k 效果条目（卡面"整备："这种关键词 + 冒号），只是提到 k 的反应器不算。"""
    s = skill(h) or ""
    return f"{k}：" in s or f"{k}:" in s


def reactor_note(n: str, B: "Board") -> str:
    """f9lr：反应器现在能不能触发。只说"反应"这部分（大司命"在场召唤图腾"那部分照样有用，不能整个人说成 idle）；
    手里等着上场的英雄自带 X 效果时写出来（张良 idle，但等着的云中君 / 瑶自带开团，一上场就激活）。"""
    s = skill(n) or ""
    m = REACT_RE.search(s)
    if not m:
        return ""
    k = m.group(1)
    src = sorted(h for h in B.heroes if h != n and own_kw(h, k))
    if src:
        return f"triggers from {len(src)} {k} hero(es): {'/'.join(src[:3])}"
    other = [seg for seg in re.split(r"[。;]", s) if seg.strip() and "每当" not in seg]
    wait = sorted(h for h in set(B.hand_heroes) if h != n and h not in B.field and own_kw(h, k))
    head = f"idle: 0 heroes with own {k} effect on board" if not other else f"its {k}-reaction idle (0 {k} heroes; other effects still work)"
    return head + (f"; waiting {'/'.join(wait[:2])} would trigger it" if wait else "")


def is_filler(h: str, B: "Board") -> bool:
    """f9lu：规范 54.6 的"杂牌"= 1~3 级、不在路线、不是坦克；另外排除现任 / 未来主C、枫叶舞宿主、共战搭档、已觉醒、稳定燃料件。"""
    if h in (B.cur_carry, B.fc) or h == B.ctx.get("maple"):
        return False
    if any(h in names for _t, names in duo_talents(B)):
        return False
    if B.awk(h) == 0 or is_tanky(h):
        return False
    if (B.lv(h) or 1) > 3:
        return False
    if FV >= F9LY and B.feeds_or_part(h):
        return False        # f9ly（r9zo_test_c/005、008：明世隐贴主C 转永久等级，却被写成 "can replace filler 明世隐"）
    for g in (kb().get(h) or {}).get("gives") or []:
        if g.get("res") in FUEL_RES and "每回合" in str(g.get("per") or ""):
            return False
    return True


TALENT_FAC_RE = re.compile(r"至少有?(\d+)名【([^】]+)】英雄")
KW_HERO_RE = re.compile(r"(?:你的|场上的?|每名|所有|己方)(整备|登场|牺牲|开团|闪现|退场)(?:效果)?英雄")   # "获得 3 张开团英雄"是送牌，不算受益条件


_GIVES_FUEL_RE = re.compile(r"(?<!出售)(?<!出售\d)获得\d+张(?:古币|[^，。]{0,6}(?:效果牌|战术牌|锦囊))")
_BOOST_FUEL_RE = re.compile(r"古币牌会额外|手牌中古币等级|打出古币时|古币不再附带转瞬")


def _feeds_effect_cards(sk: str) -> bool:
    """f9lu（r9zl 老师报：丝绸之路 / 钱生战力 / 千金换酒 被标成喂主C）：只认"直接给古币 / 效果牌"或"让手里的古币更强"，
    不认"卖人才给古币""战斗中按古币数发临时等级""送一个英雄"。"""
    if "出售" in sk and "古币" in sk.split("出售", 1)[1][:12]:
        return False
    return bool(_GIVES_FUEL_RE.search(sk) or _BOOST_FUEL_RE.search(sk))


_FAC_CARDS = None


def _faction_cards() -> dict:
    """效果 / 战术牌里"指定1名【X】英雄"这类只能对某阵营用的牌 → {牌名: {阵营}}（只读卡面）。"""
    global _FAC_CARDS
    if _FAC_CARDS is None:
        _FAC_CARDS = {}
        for nm, x in cards().items():
            if not isinstance(x, dict) or x.get("category") not in ("效果牌", "战术牌") or len(nm) < 2:
                continue
            gs = x.get("skill") or ""
            fs = {norm_fac(f) for f in re.findall(r"指定\d*名【([^】]+)】英雄", gs)}
            if fs:
                _FAC_CARDS[nm] = fs
    return _FAC_CARDS


def talent_faction_fit(n: str, B: "Board") -> str:
    """f9lz：天赋给的牌（"获得2张古币"）或天赋自己只作用于"【X】英雄"时，X 和路线主C 阵营对不上 → 写 DEAD；对得上不写。只读卡面。"""
    sk = skill(n) or ""
    rc = ((B.route or {}).get("carry") if B.route and not B.route_stale else "") or B.cur_carry or B.fc or ""
    if FV >= F9M:
        rc = ((B.route or {}).get("carry") if B.route and not B.route_stale else "") or ""
        if not rc and B.fc and B.fc != getattr(B, "temp_carry", None) and (tier(B.fc) or 0) >= 2:
            rc = B.fc
    if not rc or not fac(rc) or re.search(r"非【|其他每个阵营|召唤", sk):
        return ""                                   # 多彩古币（只能对非河洛用）/ 一起发财（加到别的阵营）/ 古币机（对召唤物用）不算
    targets, via = set(), ""
    for g, fs in _faction_cards().items():          # 天赋文字里提到的、只能对某阵营英雄用的牌（古币 / 高阶推演战术…）
        if g in sk and g != n:
            targets |= fs
            via = g
    if not targets:
        return ""
    cf = fac(rc)
    if cf in targets:
        return ""
    have = sum(1 for h in B.heroes if fac(h) in targets)
    tl = "/".join(sorted(targets))
    return (f"DEAD for this route: {via} only works on 【{tl}】 heroes; route carry {rc} is 【{cf}】"
            f" (board has {have} 【{tl}】 heroes)")


def talent_value_facts(n: str, B: "Board") -> list[str]:
    """f9lu：天赋"在这局现在值多少"（第 642 局：命运图腾条件不满足、强者自强受益 0 人、给满装主C 选装备，老师平均把握 0.51）。只写事实。"""
    sk = skill(n) or ""
    out = []
    m = TALENT_FAC_RE.search(sk)
    if m:
        k, f = int(m.group(1)), norm_fac(m.group(2))
        have = sum(1 for h in B.heroes if fac(h) == f)
        out.append(f"needs {k} 【{m.group(2)}】: have {have} → " + ("ACTIVE" if have >= k else "INACTIVE now"))
    if "已觉醒英雄" in sk or "觉醒英雄" in sk:
        out.append(f"awakened heroes now: {sum(1 for h in B.heroes if B.awk(h) == 0)}")
    mk = KW_HERO_RE.search(sk)
    if mk:
        out.append(f"{mk.group(1)} heroes now: {sum(1 for h in B.heroes if own_kw(h, mk.group(1)))}")
    C = B.cur_carry or B.fc
    if C and EFFECT_FED_RE.search(skill(C) or "") and _feeds_effect_cards(sk):
        if FV >= F9LV and re.search(r"至少|若|当你|场上有", sk):
            out.append(f"FEEDS carry {C} only while its condition holds (see above)")     # f9lv ⑯：王牌部队"≥3 名 4-5 阶"才给
        else:
            out.append(f"FEEDS carry {C} (uses effect cards)")
    if C and cat(n) == "装备牌":
        e = B.eq(C)
        out.append(f"carry {C} equips {e if e is not None else '?'}/3" + (" FULL" if e == 3 else ""))
    if "枫叶舞英雄" in sk:
        # 2026-10-15（r9zl 老师反复报）：原来只写 "carry X is marksman"，守约被当成普攻型 —— 而卡面分析（hero_roles damage_shape.source）
        # 守约是技能型。现在写伤害来源；专属天赋作用在枫叶舞宿主身上，宿主 ≠ 主C 时两个都写。
        def _src(h):
            s = ((hero_roles().get(h) or {}).get("damage_shape") or {}).get("source") or "?"
            return {"skill": "SKILL damage", "auto": "BASIC-ATTACK damage", "attack": "BASIC-ATTACK damage", "mixed": "mixed skill+attack damage"}.get(s, "damage source ?")
        tag = "rewards skill casting" if "释放技能" in sk else "rewards basic attacks" if "普攻" in sk else ""
        host = B.ctx.get("maple")
        who = []
        if host and is_hero(host):
            who.append(f"maple host {host} deals {_src(host)}")
        if C and C != host:
            who.append(f"carry {C} deals {_src(C)}")
        if tag and who:
            out.append(f"{tag}; " + "; ".join(who))
        if FV >= F9N and "1格内" in sk and host and is_hero(host):
            # f9n（第 690 局 R12：宿主安琪拉是后排法师，Laya 0.43 选了落英汲取 —— 42.6：只给近战前排宿主）
            out.insert(0, f"maple host {host} is {row(host) or '?'} {'melee' if melee(host) else 'ranged'}"
                          + ("" if melee(host) else ": few enemies within 1 cell → little effect"))
    if FV >= F9U and n == "古币机":
        # f9u（2026-10-19：古币机被选的概率过高，古币机更偏团队性；第 736 / 758 局）
        RC = ((B.route or {}).get("carry") if B.route and not B.route_stale else "") or C or ""
        src = len(B.coin_src)
        if RC and fac(RC) == "河洛":
            out.insert(0, f"TEAM talent, COSTS carry: all 古币 go to it, {RC} −2 lv per 古币 ({src} sources)")
        elif RC:
            out.insert(0, f"TEAM talent: pays random 河洛 heroes; carry {RC} not 河洛 → nothing for carry")
    if FV >= F9LZ:
        # f9lz（第 683 局：日落海马可波罗路线选了「升值」—— 得 2 张古币，古币卡面"指定1名【河洛】英雄等级+2"；
        # 日落海吃不了古币，不该选这个天赋；天赋跟套路的适配问题仍然严重）：天赋给的牌只能用在某阵营英雄身上时，对上路线主C 阵营
        fl = talent_faction_fit(n, B)
        if fl:
            out.insert(0, fl)
    if FV >= F9S and re.search(r"第[1一]个到", sk):
        # f9s（2026-10-18 第 735 局 R5 选「学霸」：不是快速升级打法却选了它）：抢先到 5/6 级的竞速天赋。
        # 我们是慢刷打法，实测（第 724~734 局）棋手 5 级在第 10~12 回合才到、第 8 回合多在 3 级 → 抢不到第一；失败奖励留级证明只在 1 级有用
        fail = ""
        m = re.search(r"失败(?:则获得|给)([一-鿿]+?)(?:兑换券|[。，,；\s]|$)", sk)
        if m:
            fx = skill(m.group(1).strip()) or ""
            fail = f"; fail {m.group(1).strip()}" + (" = lv1 only, worthless" if "棋手等级1" in fx else "")
        now = re.match(r"获得(\d+)能量", sk)
        out.insert(0, (f"+{now.group(1)}e now; " if now else "") + f"RACE: 1st to lv5 unlikely (we reach lv5 ~r11){fail}")
    if FV >= F9T and re.search(r"核心失去\s*\d+\s*级", sk) and not any(fac(h) == norm_fac("日落海") for h in B.heroes):
        # f9t（10-19：按卡面应该是不给的）：没有日落海英雄 → 核心 0 级，能源置换拿不到能量
        out.insert(0, "no 【日落海】 hero: core stays 0 lv → gives nothing")
    if FV >= F9T and "品阶数字" in sk:
        # f9t（2026-10-18 第 751 局 R2 选「战争头牌」，用法不对）：每回合的品阶数字感知层读不到，打出英雄时碰运气
        out.insert(0, "AVOID: the per-round tier number is not readable by us, so we can't play to it — only random hits (~nothing)")
    if FV >= F9T:
        av = f9t_talent_avoid(n, sk, B)
        if av:
            out.insert(0, av)
    if FV >= F9LW and "混沌裂隙" in sk:
        # f9lw ㊶（2026-10-16：这张卡执行麻烦，可以避开）：裂隙要放到格上（放下不能挪）、还要把人站上去，执行上做不好 → 实际收益接近 0
        out.insert(0, "AVOID: the rift must be placed on a fixed cell and a hero kept standing on it — our play can't run it reliably, "
                      "so in practice it gives ~nothing")
    if FV >= F9W:
        ly = f9y_talent_lines(n, B) if FV >= F9Y else None
        if ly is not None:
            # f9y：硬条件表一行结论替代 f9x 的低价值名单（不重复写两遍）
            out = [x for x in out if not x.startswith("LOW VALUE")]
            for x in reversed(ly):
                out.insert(0, x)
        else:
            lv = (f9x_low_value_talent if FV >= F9X else f9w_low_value_talent)(n, B)      # f9w（54.25-4）：复盘点名的低价值天赋；f9x 扩表
            if lv:
                out.insert(0, lv)
        tf = f9w_talent_hero_tie(n, B)       # f9w（10-19：送英雄 / 点名英雄的天赋都打标记，data/talent_hero_ties.json）
        if tf:
            out.insert(0, tf)
    return out


def f9t_talent_avoid(n: str, sk: str, B: "Board") -> str:
    """f9t（10-18 天赋审查 + 2026-10-19 卡面确认第 14 问）：执行层做不到 / 卡面在本局必然无效的天赋，一句结论。"""
    if "可购买天赋" in sk:
        return "AVOID: it's a buy-later talent and we have no buy action; the pick gets swapped"     # rules.CAN_BUY_TALENT=False
    if re.search(r"能量可用至\s*-", sk):
        return "AVOID: we can't read negative energy, so the credit is never used"
    if "保持在棋手等级1" in sk:
        return "AVOID: pays only at lord lv1; talents come after level-ups, so never"
    if "电磁弹簧" in sk:
        return "AVOID: the spring must be placed on a cell; we can't, it just gets sold"
    if "古币不再附带转瞬" in sk:
        return "AVOID: we play 古币 on arrival, so hoarding never happens"
    if "擂台" in sk and "【三分】" in sk and not any(fac(h) == "三分之地" for h in B.owned):
        return "AVOID: needs 【三分】 heroes and placing the arena; we have neither"
    if "阿科米亚核心" in sk and "失去" not in sk:
        rc = ((B.route or {}).get("carry") if B.route and not B.route_stale else "") or B.cur_carry or B.fc or ""
        if FV >= F9Z:
            rc = f9y_talent_team(B)[1]      # f9z：同硬条件行的主C 口径（1 阶临时主C / 没到手的路线主C 不算），免得两行说法打架
        sun =(B.engine and B.engine.get("key") == "sunset_core") or (rc and fac(rc) == "日落海")
        if "不再分配等级" in sk:
            return ("AVOID on a 日落海 route: core stops feeding the carry's levels" if sun
                    else "AVOID: on 日落海 it stops core levels; off 日落海 the core is lv 0")
        if rc and fac(rc) != "日落海":
            return f"no value: core only grows with / feeds 【日落海】; carry {rc} isn't"
        if not rc and not any(fac(h) == "日落海" for h in B.owned):
            return "no value: no 【日落海】 hero → core stays lv 0"
    return ""


def fuel_card_ok(x: str, C: str) -> bool:
    """f9ls：手里这张效果牌能不能打在主C 身上（= 算不算燃料）。变身 / 移除 / 摧毁装备不能用在主C 上；卡面限阶 / 阵营 / 远近战按主C 查。"""
    if cat(x) != "效果牌":
        return False
    sk = skill(x) or ""
    if _is_transform(sk) or REMOVE_RE.search(sk) or DESTROY_EQ_RE.search(sk):
        return False
    m = re.search(r"(\d)\s*[~-]\s*(\d)\s*阶", sk)
    if m and not (int(m.group(1)) <= tier(C) <= int(m.group(2))):
        return False
    mf = re.search(r"指定\d*名【([^】]+)】英雄", sk)
    if mf and fac(C) != norm_fac(mf.group(1)):
        return False
    if ("限定远程" in sk and melee(C)) or ("限定近战" in sk and not melee(C)):
        return False
    return True


def carry_fuel_line(B: "Board", kind: str = "", cands: list = ()) -> str:
    """f9lr：吃效果牌的主C 的燃料来源（每回合稳定 / 有条件）。
    f9ls（R30 / R30b 用牌题退步）：主C 没到手不写；变身 / 移除 / 摧毁装备的用牌题不写（那张牌本来就不能给主C）；
    手里能打在主C 身上的效果牌才算；燃料件在手里没上场不算稳定来源。"""
    C = B.cur_carry or B.fc
    if not C or not EFFECT_FED_RE.search(skill(C) or ""):
        return ""
    if FV >= F9LS:
        if C not in B.field and C not in B.hand_heroes:
            return ""
        if kind == "use_card" and any((_is_transform(skill(x) or "") or REMOVE_RE.search(skill(x) or "")
                                       or DESTROY_EQ_RE.search(skill(x) or "")) for x in cands if x):
            return ""
    steady, cond = [], []
    for h in sorted(set(B.heroes) | set(B.hand_heroes)):
        for g in (kb().get(h) or {}).get("gives") or []:
            if g.get("res") in FUEL_RES:
                per = str(g.get("per") or "")
                if FV >= F9LS and h not in B.field:
                    cond.append(f"{h} (in hand, not deployed: gives nothing until deployed)")
                    continue
                tag = f"{h} ({'board' if h in B.field else 'hand'}, {per})"
                (steady if "每回合" in per else cond).append(tag)
    eff = [x for x in B.hand if (fuel_card_ok(x, C) if FV >= F9LS else cat(x) == "效果牌")]
    return (f"CARRY FUEL: {C} needs effect cards used on it EVERY round (each lowers its mana cap this round; 古币 is fleeting); "
            f"steady sources: {', '.join(steady) or 'none'}; conditional: {', '.join(cond) or 'none'}; effect cards in hand now: {len(eff)}")


F9LW = "f9lw"   # f9lw（2026-10-16，第 652~655 局复盘裁决 ㊱~㊶）：枫叶舞在临时主C 上写"等级浪费、该换绑"；
#               弈星 = 纯过渡件 + 回本（在场期间棋手升过级）就卖；路线已锁时别家主C 写"不在核心阵容、阿离只养一个主C"；第二输出位只放装备；
#               点兵点将写核心阵容占比；混沌裂隙写每个目标召唤 2 倍等级、标辅助/主C；天赋混沌次元写"执行难、避开"；sell_filler 新题型
F9LX = "f9lx"   # f9lx（2026-10-16 上午，R34 上线后 656~664 局 + 8+7 条规则答复，待办 ㊸~51）：sell_hand 新题型（手里用不上的英雄卖不卖）；
F9LY = "f9ly"   # f9ly（2026-10-16 午，665 局后）：碰碰运气费用写成 "costs 1~5 energy (random)"（旧文本 "1~-5"；已确认随机消耗 1~5）
def _dmg(n: str) -> str:
    """伤害类型：f9o 起用人工确认表（亚连 = 物理），旧版本保持原推断（逐字不变）。"""
    return _ht().dmg_type(n, legacy=FV < F9O)


F9R = "f9r"     # f9r（2026-10-18，r9zt 老师 22）：坦克不标 OTHER LINEUP carry（程咬金被当成别家主C）
F9S = "f9s"     # f9s（2026-10-18，r9zv 老师 6 题）：FINAL carry settled 要求枫叶舞在他身上或 ≥10 级（54.15-7）；LATE-FILL 按"其余最终阵容件都在场"而不是第 8 回合（第 734 局太乙）；竞速天赋（学霸）写慢刷抢不到第一（第 735 局）；满员刷新：缺最低配套件能顶掉路线外的人时写 worth paid refreshes（第 735 局 R9）
F9T = "f9t"     # f9t（2026-10-18 晚，r9zw 老师）：云中君等"给永久等级"的英雄不标 LATE-FILL；没写回合的路线开局阶段算到第 6 回合（第 747 局牛魔）
F9U = "f9u"     # f9u（2026-10-19，第 760 局 R5：机器头锁攻速 0.4，主C 狄仁杰靠古币叠攻速 → 题面只写"攻速 固定 0.4"，Laya 0.45 选了它）
F9V = "f9v"     # f9v（2026-10-19 上午，卡面确认答复 + r9zz 老师反馈）：机器头只给花木兰 / 嬴政（法力靠普攻回）
F9X = "f9x"     # f9x（2026-10-19 晚，r9zzc 老师冲突 + 晚间复盘反馈，规范 54.26）：马可波罗算攻速型；路线 not_wanted 不进最终阵容 / 缺件 / CORE 计数 / teammates；
                # 装备满后不给辅助（瑶除外）；计划外 sub 不标 ROUTE stage unit；觉醒前提吸血；红叶染锋临时宿主；特殊装备都不对口；
                # 条件触发 / 全队效果；满员不能上场；低价值天赋扩表 + 共战收紧；攻速主C 少司缘补位；普攻叠层装 MISMATCH
SUB_ONLY_F9X = "ROUTE optional sub (not in plan)"
F9Y = "f9y"     # f9y（2026-10-20，r9zzd 老师题面错 + 规范 54.27）：天赋硬条件表（data/talent_conditions.json）每个选项一行；
                # 屏蔽路线的主C（安琪拉 / 亚连）路线没锁时不当未来主C；阵容占比 ≥80% 不写 not in ROUTE；early_priority（日落海雅典娜）算路线核心；
                # 有福同享补主C、同效果目标合并；拆装备件数按账本；马可波罗线少司缘 late_fill 也写补位；枫叶舞宿主不在场写 stale；
                # 古币机吃古币；特殊装备伤害类型适配；5 级时"觉醒核心件"与"下回合升 6"并写；商店 stop 写缺的引擎件在店
BLOCKED_CARRIES_F9Y = ("安琪拉", "亚连")     # 同 laya_policy.WEAK_CARRIES + WEAK_ROUTES 的主C（临时规则 route_avoid_weak；54.27-17）
F9ZB = "f9zb"   # f9zb（2026-10-21 起；R52）：选项预算超长行改短、按主C 选引擎（第 884 局玄策 = 河洛·古币）等
F9ZA = "f9za"   # f9za（2026-10-20 晚起；R51）：天赋 / 装备 / 效果牌按主C 复审（data/review/*_carry.json）、r9zzf 老师题面 bug
F9Z = "f9z"     # f9z（2026-10-20 晚，r9zze 老师 + 规范 54.28 + 三份 10-20 审查）：引擎表 f9z_override；
                # 普攻装扩到 BASIC_ATTACK_ITEMS_2；路线题屏蔽路线先排除再取前 3；天赋主C 口径同线上（1 阶临时主C / 没到手的路线主C 不算）；
                # 锁店留件只认路线件；候选路线件不写 cannot be deployed；逐鹿专用战术牌不算非逐鹿主C 的燃料；机器头只给花木兰 / 嬴政；
                # 日落海线明世隐 / 雅典娜算引擎件；满员刷新被顶的人先挑路线外；特殊操作天赋写清做不到；屏蔽主C 已是现任主C / 路线失效
WEAK_ROUTES_F9Z = ("阿离日落海安琪拉", "伞亚连", "双剑出鞘伞亚连", "双刃亚连")    # 同 laya_policy.WEAK_ROUTES（route_avoid_weak；冻结副本，保证渲染确定）
WEAK_CARRIES_F9Z = ("安琪拉",)                                                  # 同 laya_policy.WEAK_CARRIES（马可波罗 10-20 第 827 局后撤回）
SUNSET_ENGINE_F9Z = ("明世隐", "雅典娜")     # 54.25-2 / 54.27-22 / 54.28-3：日落海路线上明世隐、雅典娜按引擎件
SPEED_LOCK_CARRIES_F9Z = ("花木兰", "嬴政")  # 同 rules.SPEED_LOCK_CARRIES：机器头只给这两位
NO_GEAR_EXEMPT_F9X = ("瑶",)       # 同 rules.SHARE_EQUIP：首件普通装备开团分享给 3 名友军，辅助里唯一要装备的
F9W = "f9w"     # f9w（2026-10-19 午后）：route_min_engine_f9v.json 里 since=f9w 的路线（代表法律制裁你 / 双剑出鞘伞亚连 / 日落海剑圣）
TEMP_TO_HEROES_RE = r"(英雄|【[^】]+】英雄)获得\s*\d+\s*临时等级|该英雄获得\s*\d+\s*临时等级|额外提供\s*\d+\s*临时等级|其他英雄等级数一半的临时等级"
TEMP_TO_SELF_RE = r"临时等级\s*[+＋]|获得[^。]{0,12}临时等级"
F9Q = "f9q"     # f9q（2026-10-17 晚）：已停用的自定义阵容（user_ganjiang）不进题面
F9P = "f9p"     # f9p（2026-10-17 晚，r9zs 老师 66 条）：喂主C 看卡面阵营限制（古币只给河洛 / 芈月要逐鹿）；引擎产出方只认卡面真给等级的、分每回合 / 条件 / 一次性；满员时写路线外的人先让给谁
F9O = "f9o"     # f9o（2026-10-17，r9zr 老师）：主C 阵容占比 ≥80% 的人不写 not in ROUTE；只能给某阵营用的牌（古币）场上没该阵营英雄时写 UNUSABLE
F9N = "f9n"     # f9n（2026-10-17，第 690 局 + r9zq_c 老师）：刘邦这类拆本方图腾的单位不再标引擎件 / same build；落英汲取写宿主前后排
F9M = "f9m"     # f9m（2026-10-17，r9zq 老师反馈）：天赋阵营 DEAD 只认锁定路线主C / 已定的非临时主C（283dc238c841 把 1 阶临时主C 曜当路线主C）；
                # 路线外牺牲件不标引擎件；会摧毁本方图腾的单位（刘邦）在有图腾的路线里不算次选 / 引擎件
F9LZ = "f9lz"   # f9lz（2026-10-16 晚，R36 开训之后）：路线外"带牺牲 / 整备"不算喂主C（花木兰 / 项羽不在守约核心阵容）；珍品宝箱出特殊装备（已更正）
#               路线锁定且路线主C 已拥有后 CORE of route 只给已锁路线；枫叶舞宿主多用途写代价；裂隙主T 首选 / 共战搭档；
#               武装重构写被摧毁的装备；前浪的荣光 卡面
SECOND_OUTPUT_WHY = "only parks gear the carry can't use; never feed it"
SECOND_OUTPUT = "2nd output (Ali raises ONE carry): " + SECOND_OUTPUT_WHY
F9LV = "f9lv"   # f9lv（2026-10-15 夜，第 643~647 局 + r9zl 老师反馈）：引擎零件只标"卖了就断"的那个、开局件过了阶段不再算路线成员、弈星写剩余价值
F9LU = "f9lu"   # f9lu（2026-10-15，第 642 局复盘 + r9zj / r9zk 老师反馈）：杂牌判定收紧（规范定义 + 扣掉已在等位的）；天赋题写"现在值多少"
                # （条件满不满足、受益人数、配不配主C、装备栏、6 级专属对主C 类型）；商店题写本回合经济计划；
                # 听天由命类挪到局面行；升级口径统一按下回合；普通免费刷新写"留到下回合"
F9LT = "f9lt"   # f9lt（2026-10-15，r9zj 老师反馈）：6 级商店实测概率；触发类牌只列合规目标；"消除等级最低 N 名"类写实际对象；
                # 本回合限定的免费刷新（无影手）写"不用就作废"
TRIGGER_RE = re.compile(r"触发其(整备|登场|牺牲|开团|闪现|退场)效果")
AUTO_LOWEST_RE = re.compile(r"(消除|移除|使)场上(\d+)名等级最低")
F9LS = "f9ls"   # f9ls（2026-10-15，R30 / R30b 用牌题退步）：燃料行只算能用在主C 身上的效果牌；主C 没到手不写；变身 / 移除 / 摧毁装备的用牌题不写；
                # 燃料件只在手里没上场不算稳定来源；锁店选项列出非英雄牌
F9LQ = "f9lq"   # f9lq（2026-10-14 晚，r9zd 老师反馈）：弦音只对 1~3 阶（主C 4 阶以上不写"卖掉拿弦音完成觉醒"）；
                # "场上每有1种阵营，额外触发1次此效果"（八方来援）等级加成乘上 1 + 场上阵营数（不含无阵营）；
                # 手牌到 / 超上限时写明"照样能买，超过 10 张的部分下回合开始销毁"（2026-10-14）
F9LP = "f9lp"   # f9lp（2026-10-14：核心配件场上有一个、商店里有两个，买了就全部凑齐——这个事实要能判定出来）：
                # 最低配套行补"缺的人在不在眼前商店里、一次买齐多少能量、钱和空位够就写 COMPLETES"；新增 FLOW OPTIONS 行（agent/sim/flows 的流派库）
F9LN = "f9ln"   # f9ln（2026-10-13 第 614 局）：refresh 的"想要的牌"补上路线最低配套缺件（自定义路线没有官方阵容 mates）
F9LM = "f9lm"   # f9lm（2026-10-13，理想打法）：临时主C（TEMPORARY）、候选路线核心卡（CORE of route）、最终主C 已定（FINAL carry settled）、
                #   天赋点名派系时我方有几个该派系英雄、卡面带上天赋送出的同名牌效果（given_skill）
# f9lm：实战的指定路线清单（data/laya/iterations/one_match.sh 的 --force-route：首选 + 备选，2026-10-12 起这 9 条）。
# 这是对局配置（"我们会走哪些路线"），不是规则引擎的判断；规则层 rules.flex_core_cards 用的是同一份清单。改 one_match.sh 的 ALL 时同步这里
# （新版本号下改，旧版本不受影响——只有 FV >= F9LM 才读它）。
FLEX_ROUTES = ("50720714303640586", "81682962224371742", "32706315349545018", "122215359354172436", "user_hanxin_sunset",
               "user_compound_river_totem", "user_hanxin_open", "user_marco_sunset", "user_angela_sunset")


def flex_route_keys(ctx: dict, settled: bool, carry: str = "") -> list:
    """f9lm：候选路线 = 指定路线清单里没被禁派系废掉的（routes.viable，规则层同一判据）；最终主C 已定后只看当前锁定路线。有序、去重。
    锁定路线没有核心卡数据（旧路线 key，如 user_angela）时，按主C 找清单里以他为主C 的路线；都没有就退回整份清单。"""
    R = _routes()
    rk = str(ctx.get("route") or "")
    allk = ([rk] if rk in FLEX_ROUTES else []) + [k for k in FLEX_ROUTES if k != rk]
    keys = allk
    if settled:
        rc = ((R.by_key(rk) or {}).get("carry") or "") if rk else ""
        # 主C 已定：只看他的路线；清单里没有以他为主C 的路线（如亚连）→ 不标任何候选路线核心卡
        # （2026-10-13 第 606 / 611 局：原来退回整份清单，别的路线的马可波罗被标成 CORE，买了当回合就卖）
        keys = ([rk] if rk and flex_core_of(rk) else []) or [k for k in FLEX_ROUTES if ((R.by_key(k) or {}).get("carry") or "") in {rc, carry} - {""}]
    out = []
    _en = None
    if FV >= F9ZB:
        # f9zb（第 902 局 R3：马可波罗已从棋手主C 名单关掉，题面仍写 "CORE of route 马可波罗"，Laya 0.46 买了）：候选路线按主C 名单过滤
        try:
            from agent.decision import lord_config as _LCf
            _c = _LCf.load()
            _en = set(_c.enabled_list()) if _c.present else None
        except Exception:                                   # noqa: BLE001
            _en = None
    for k in keys:
        lu = R.by_key(k)
        if lu and R.viable(lu, ctx.get("banned") or "") and k not in out:
            if _en is not None and (lu.get("carry") or "") not in _en and k != rk:
                continue
            out.append(k)
    return out


def flex_route_label(k: str) -> str:
    """路线的短名 = 主C 名；清单里两条路线主C 同名（韩信：日落海整备 / 大河开团）时加阵营。"""
    R = _routes()
    lu = R.by_key(k) or {}
    cr = lu.get("carry") or str(k)
    dup = [x for x in FLEX_ROUTES if x != k and ((R.by_key(x) or {}).get("carry") or "") == cr]
    if dup:
        nm = lu.get("name") or ""
        f = next((FACTION_EN[w] for w in ("日落海", "大河", "河洛", "逐鹿", "三分") if w in nm), "")
        if f:
            cr += "-" + f
    return cr


def flex_core_of(k: str) -> list:
    """路线核心卡（route_min_engine.json 的 core_set，含主C）里的英雄 —— 与 rules.flex_core_cards 同口径：
    跳过牌的条件（"（牌）…"）、"搭档"槽、"A 或 B"槽，只认卡库里的英雄名。"""
    min_engine_of(None)                                  # 确保已加载
    out = []
    for u in ((_MIN_ENG or {}).get(str(k)) or {}).get("core_set") or []:
        t = str(u.get("unit") or "")
        if t.startswith("（牌）") or "搭档" in t or "或" in t:
            continue
        if is_hero(t) and t not in out:
            out.append(t)
    return out


def flex_core_lines(B: "Board") -> list:
    """f9lm 状态行：临时主C / 最终主C 已定 + 候选路线核心卡与已有张数。"""
    L = []
    if B.temp_carry:
        L.append(f"current carry {B.temp_carry} is TEMPORARY (t1, will be replaced by the final carry)")
    elif B.final_carry:
        L.append(f"FINAL carry settled: {B.final_carry}")
    if B.flex_keys:
        cores = ", ".join(f"{h} {B.own_copies(h)}" for h in B.flex_core)
        L.append(("current route core cards" if B.final_carry else f"candidate route cores ({len(B.flex_keys)} routes")
                 + ("" if B.final_carry else ")") + " owned copies: " + (cores or "-"))   # 路线名在选项的 CORE of route 标签里
    return L


def flex_core_tag(n: str, B: "Board") -> str:
    """f9lm：商店 / 拍卖 / N 选 1 的英雄选项 —— 候选路线核心卡（带已有张数）。"""
    labs = B.flex_core.get(n) if FV >= F9LM else None
    if labs and FV >= F9LX and B.one_carry_locked():
        # f9lx ㊸（第 656 局 R6~R7：路线 r4 锁狄仁杰后，商店还把马可波罗写成 CORE of route 马可波罗 → 连买 4 张养到 22 级）
        rk = str((B.route or {}).get("key") or "")
        rname = (B.route or {}).get("name") or ""
        labs = [x for x in labs if x in (rk, rname) or x == (B.route or {}).get("carry")]
    return f"CORE of route {'/'.join(labs[:3])} (own {B.own_copies(n)})" if labs else ""


def empty_slot_line(B: "Board") -> str:
    """f9lm（第 5~6 名复盘：输局第 5~8 回合平均 2.9 个空位·回合，第 1 名 0.2）：现在的空位 + 手里能补的英雄。"""
    pop = B.pop if FV >= F9LQ else B.ctx.get("pop")
    fill = sorted(dict.fromkeys(B.hand_heroes))
    return (f"empty slots now: {B.free if B.free is not None else '?'} (population {pop if pop is not None else '?'}/{B.cap if B.cap is not None else '?'})"
            + "; hand heroes that can fill them: " + (", ".join(fill) if fill else "none"))


def after_level_fill(B: "Board", cost, gold) -> str:
    """f9lm：升完级还剩多少能量、能补几个空位（手里等着的英雄先上，剩下的空位要买；英雄 3 能量）。"""
    m = meta()
    if not (isinstance(cost, int) and cost > 0 and isinstance(gold, int) and gold >= cost and B.free is not None and B.level):
        return ""
    left = gold - cost
    slots = max(B.free, 0) + (1 if m.level_raises_pop(B.level) else 0)
    k = max(0, slots - len(set(B.hand_heroes)))
    n = min(left // 3, k)
    return f"after level-up: {left}e left → can buy {n} hero(es) to fill {k} empty slot(s)"


def slot_tag(n: str, B: "Board") -> str:
    """f9lm：这张英雄拿到后能不能直接上场（新单位；同名合成不占位）。"""
    if B.free is None or n in B.field:
        return ""
    wait = len({h for h in B.hand_heroes if h != n})
    if FV >= F9LY and B.route and not B.route_stale:
        # f9ly（第 643~666 局：75 次路线核心在店被写成 "would wait"，其中 38 次手里等位的全是路线外的杂牌英雄 —— 它们该卖，不会上场）
        _team = set(_routes().final_team(B.route)) | set(_routes().members(B.route))
        if FV >= F9W:
            _team = (_team | set(B.min_eng.get("extra_wanted") or [])) - set(B.min_eng.get("not_wanted") or [])   # f9w
        wait = len({h for h in B.hand_heroes if h != n and (h in _team or B.feeds_or_part(h))})
    if B.free <= 0:
        if FV >= F9O and wait and B.route and not B.route_stale and n in _team:
            # f9o（第 711 局 R12：雅典娜 ROUTE final team 写 "would wait too"，场上有别家主C 公孙离 / 路线外蒙恬可换，Laya 停手 0.29 > 买 0.28；
            # 商店还是没买雅典娜，拍卖也同理）：路线件且场上路线外单位比手里等位的路线件多 → 写能换掉谁
            offr = [h for h in sorted(B.heroes, key=lambda h: (B.lv(h) or 0))
                    if h not in _team and h not in (B.fc, B.cur_carry) and not B.feeds_or_part(h)
                    and not (FV >= F9P and (B.anch["mates"].get(h) or 0) >= 0.8)]
            if len(offr) > wait:
                return f"FULL: can replace off-route {offr[0]}"
            if FV >= F9Q and offr:
                return f"FULL: {offr[0]}'s slot goes to {wait} waiting first"     # f9q：缩短（原句挤掉了 with carry）
            if FV >= F9P and offr:
                # f9p（r9zs 老师 51/61：场上有路线外的人却只写 "would wait too"）：路线外的位置先让给手里等着的路线件
                return f"FULL: off-route {', '.join(offr[:2])} go to the {wait} waiting route piece(s) first"
        if FV >= F9LR and wait:
            # f9lr（第 632 局：满员、手里已压着 3~5 个英雄时，商店 / N 选 1 / 拍卖还在拿新英雄，最后手牌 13 张）
            return f"FULL + {wait} waiting in hand: would wait too"
        return "board full: needs a swap to deploy"
    if B.free > wait:
        return "fills an empty slot"
    return f"empty slots taken by {wait} waiting hand hero(es): needs a swap to deploy"


def faction_own_line(f: str, B: "Board") -> str:
    """f9lm 天赋题：卡面点名【X】时，我方场上 / 手里 / 候选路线核心卡里各有几个【X】英雄。"""
    ff = norm_fac(f)
    a = sum(1 for h in B.heroes if fac(h) == ff)
    b = sum(1 for h in dict.fromkeys(B.hand_heroes) if fac(h) == ff)
    cc = sum(1 for h in B.flex_core if fac(h) == ff)
    return f"our 【{f}】 heroes: board {a}, hand {b}, candidate-route cores {cc}"


F9LL = "f9ll"   # f9ll（2026-10-12：碰到 5 级神装、现有主C 等级较低时，要想办法买一个能穿这件装备的主C）：手里的 5 阶装备主C 用不了 → 写出来，能穿的主C 人选打标
F9LK = "f9lk"   # f9lk（2026-10-12 第 573~580 局题面审查）：主C 资格（1 阶 / 经验件 / 配套 / 强度）、路线失效时同体系备选、锚点跟路线、人口按等级表、标签打架
F9LJ = "f9lj"   # f9lj（2026-10-11：先凑最低配套，再凑其他英雄；太乙这类前期不用急）：路线最低配套（等级引擎）进度 + 候选/场上单位的 MIN ENGINE / LATE-FILL 标签
_MIN_ENG = None


_MIN_ENG_F9V = None


def _min_eng_f9v() -> dict:
    global _MIN_ENG_F9V
    if _MIN_ENG_F9V is None:
        f = PROJECT_DIR / "data" / "route_min_engine_f9v.json"
        _MIN_ENG_F9V = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return _MIN_ENG_F9V


def min_engine_of(route_key) -> dict:
    """data/route_min_engine.json：路线 → 最低配套（min_set 单位可写"A 或 B"）、early_priority、late_fill。"""
    global _MIN_ENG
    if _MIN_ENG is None:
        try:
            _MIN_ENG = json.loads((PROJECT_DIR / "data" / "route_min_engine.json").read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            _MIN_ENG = {}
    me = (_MIN_ENG or {}).get(str(route_key or "")) or {}
    _e9 = _min_eng_f9v().get(str(route_key or "")) or {}
    if _e9 and FV >= F9W and _e9.get("f9w_override"):
        _e9 = dict(_e9, **_e9["f9w_override"])     # f9w：狄仁杰路线核心改李元芳 + 程咬金（10-19）
    if _e9 and FV >= F9Z and _e9.get("f9z_override"):
        # f9z（10-20 审查 fixes #8）：合并顺序同 rules.min_engine（base → f9w_override → f9z_override），early_priority 即规则层 early_priority_full
        _e9 = dict(_e9, **_e9["f9z_override"])
    if FV >= F9V and _e9 and FV >= (_e9.get("since") or F9V):
        # f9v：补 5 条路线（伞亚连 / 守约三条 / 瞬步狄仁杰）+ 覆盖 4 条（10-19：除沈梦溪外明世隐 > 李元芳）
        me = _e9 if FV >= F9W else _min_eng_f9v()[str(route_key or "")]     # f9w：带上 f9w_override（原来合并后又被原表覆盖）
    if FV >= F9ZA and not me and route_key and not _f9za_borrowing:
        # f9za（同 rules.min_engine 10-21）：阵容没收录（沈梦溪 618167530505238 / 狄仁杰 / 玄策…）→ 借同一主C 收录过的阵容
        try:
            globals()["_f9za_borrowing"] = True
            return f9za_borrow_engine(str(route_key))
        finally:
            globals()["_f9za_borrowing"] = False
    if FV >= F9LL and me.get("core_set"):
        me = dict(me, min_set=me["core_set"])        # f9ll：≤3 人的核心引擎（2026-10-12）
        if FV >= F9ZA:
            # f9za（10-21 待办，第 855 局沈梦溪炸弹猫线铠在 core_set 不在 early_priority）：路线件 = 原口径 ∪ core_set（= 规则层 early_priority_full）
            me["early_priority"] = list(dict.fromkeys(list(me.get("early_priority") or []) + f9za_core_names(me)))
    return me


_f9za_borrowing = False


def min_engine_groups(me: dict) -> list:
    """最低配套按"槽"列出：每槽是可互相替代的英雄（"李元芳 或 程咬金"、"闪现搭档 1 名（孙悟空…/阿轲…）"）。
    只认文本里出现的真实英雄名（按出现顺序）；一个英雄都没有的槽（"每回合 ≥3 张效果牌"）不算英雄槽。"""
    out = []
    for u in me.get("min_set") or []:
        t = str(u.get("unit") or "")
        if t.startswith("（牌）"):
            continue                                  # 牌的条件（"每回合 ≥3 张效果牌"），不是英雄槽
        names = sorted((n for n in _hero_names() if n in t), key=lambda n: (t.index(n), -len(n)))
        names = [n for i, n in enumerate(names) if not any(n != m and n in m for m in names)]   # 去掉被更长名字包含的
        if names and names not in out:
            out.append(names)
    return out


_HERO_NAMES = None


def _hero_names() -> list:
    global _HERO_NAMES
    if _HERO_NAMES is None:
        _HERO_NAMES = sorted(n for n in cards() if is_hero(n))
    return _HERO_NAMES


F9LI = "f9li"   # f9li（2026-10-11）：已选天赋 + 能从文字算出的进度、阵容是否成型、主C 已定时的 C 位标签、eco 三处缺口（升级费 −1 经验 / 免费刷新 / 人口矛盾）
F9LG = "f9lg"                     # f9lg：发英雄牌 plan 并入 B.anch 阵容（与 wanted 同口径）+ 写出合成材料期望
HERO_PRICE = 3                    # 英雄牌商店价（cards.json 英雄牌 cost 统一 3）
COUPON_CHOICE = 3                 # 兑换券 = 卡牌 3 选 1（卡面「X阶英雄兑换券」原文）
PLAN_NAMES_MAX = 4


def refresh_afford_note(B: "Board") -> str:
    """付费刷新：刷完剩 g-2，够不够买一张英雄；免费刷新：钱不动，只在不够 3 能量时写（刷到也买不起）。"""
    g, free = B.ctx.get("gold"), B.ctx.get("free_refresh") or 0
    if not isinstance(g, int):
        return ""
    if free:
        return f"; {g}e → can't afford a hero (3e) after" if g < HERO_PRICE else ""
    after = g - _rcost(B)
    if after < 0:
        return " → CAN'T AFFORD the refresh"
    if after < HERO_PRICE:
        return " → can't afford a hero (3e) after"
    k = after // HERO_PRICE
    return f" → can still buy {k} hero{'es' if k > 1 else ''} after"


@lru_cache(maxsize=1)
def grant_pools() -> dict:
    """{牌名: {"n": 张数, "pick": random / same / choose:s选n / coupon, "tier_rule": ..., "pool": [英雄…]}}。
    来源：data/card_graph_v3.json 的 grants_pool 边（按卡面关键词 / 阵营 / 品阶自动展开，不写死名字）。"""
    f = PROJECT_DIR / "data" / "card_graph_v3.json"
    if not f.exists():
        return {}
    g = json.loads(f.read_text(encoding="utf-8"))
    out: dict = {}
    for e in g.get("edges") or []:
        if e.get("type") != "grants_pool":
            continue
        a = e.get("attrs") or {}
        d = out.setdefault(e["from"], {"n": int(a.get("n") or 1), "pick": str(a.get("pick") or "random"),
                                       "tier_rule": a.get("tier_rule") or "", "same": bool(a.get("same")), "pool": []})
        if e["to"] not in d["pool"]:
            d["pool"].append(e["to"])
    for d in out.values():
        d["pool"].sort()                                    # 逐字确定
    return out


def plan_heroes(B: "Board") -> list:
    """阵容要的人（有序、去重）：路线主C、路线当前阶段成员（main+sub）、主C 本人、以主C 为锚的官方阵容高频队友（≥30%）。
    口径同 eco_missing，区别是**不去掉已拥有的**（发到同名能合成 / 觉醒）；被禁派系去掉（这一局刷不出）。"""
    R = _routes()
    banned = R.norm_banned(B.ctx.get("banned") or "")
    C = eco_carry(B)
    out: list = []

    def add(n):
        if n and n not in out and is_hero(n) and not (banned and norm_fac(fac(n)) == banned) and n not in B.nw:   # f9x：路线不要的人
            out.append(n)
    if B.route and not B.route_stale:
        add(B.route.get("carry") or "")
        st = R.stage_now(B.route, B.rnd) if B.rnd else None
        if st:
            for h in list(st.get("main") or []) + list(st.get("sub") or []):
                add(h)
    add(C)
    if C:
        for h, p in anchored(C)["mates"].items():
            if p < 0.3:
                break
            add(h)
    if FV >= F9LG and B.anchor and B.anchor != C:
        # f9lg：wanted 行（"wanted tN"）按 B.anch（未来主C / 手里候选的阵容）算，plan 原来只按 eco_carry，
        # 两处口径不一 → 同一题出现 "wanted t5: 鲁班七号" 又 "your plan 0, expect 0.00"（r9u batch_027 老师）。
        add(B.anchor)
        for h, p in B.anch["mates"].items():
            if p < 0.3:
                break
            add(h)
    return out


def _expect_useful(pick: str, n: int, k: int, N: int) -> float:
    """期望拿到几个阵容要的人（全池均匀）。random/same：n×k/N；s选n：E[min(n, Bin(s,k/N))]；兑换券：n×(1-(1-k/N)^3)。"""
    if N <= 0 or k <= 0:
        return 0.0
    p = k / N
    m = re.match(r"choose:(\d+)选(\d+)", pick)
    if m:
        s, take = int(m.group(1)), int(m.group(2))
        return sum(min(take, x) * math.comb(s, x) * p ** x * (1 - p) ** (s - x) for x in range(s + 1))
    if pick == "coupon":
        return n * (1 - (1 - p) ** COUPON_CHOICE)
    return n * p


def grant_fact(n: str, B: "Board") -> tuple[str, str] | None:
    """(完整写法, 放不下时的短写法)；不是发英雄牌的牌返回 None。
    完整：`pool 8, your plan 3 (盾山 铠 钟馗), expect 0.75 useful of 2`。"""
    d = grant_pools().get(n)
    if not d:
        return None
    R = _routes()
    banned = R.norm_banned(B.ctx.get("banned") or "")
    pool = list(d["pool"])
    note = ""
    if d["tier_rule"] == "lord_level+1":
        if not B.level:
            return (f"pool ? (tier = lord lv+1, lv unread), gives {d['n']}", f"pool ?, gives {d['n']}")
        t = min(5, B.level + 1)
        pool = [h for h in pool if tier(h) == t]
        note = f" t{t}"
    if banned:
        pool = [h for h in pool if norm_fac(fac(h)) != banned]
    if not pool:
        return None
    mp = re.match(r"choose:(\d+)选(\d+)", d["pick"])
    if mp:
        note += f", pick {mp.group(2)} of {mp.group(1)}"
    elif d["pick"] == "coupon":
        note += f", {d['n']} coupons pick 1 of {COUPON_CHOICE}"
    elif d["same"]:
        note += ", 2 identical"
    if "每回合" in skill(n) or (kb().get(n) or {}).get("trig") in ("回合开始", "整备"):
        note += ", every round"
    plan = plan_heroes(B)
    hit = [h for h in plan if h in pool]
    N, k = len(pool), len(hit)
    got = d["n"] if not mp else int(mp.group(2))
    if not plan:
        return (f"pool {N}{note}, your plan: none yet (no route/carry)", f"pool {N}, plan none yet")
    ex = _expect_useful(d["pick"], d["n"], k, N)
    names = " ".join(hit[:PLAN_NAMES_MAX]) + (" …" if k > PLAN_NAMES_MAX else "")
    full = f"pool {N}{note}, your plan {k}" + (f" ({names})" if k else "") + f", expect {ex:.2f} useful of {got}"
    if FV >= F9LG and k:
        # f9lg：拿到已拥有的 = 合成材料（不占人口），拿到没有的 = 新人头（满员要卖人腾位）—— 满员时两者价值差很多（r9u batch_028/030 老师）
        km = sum(1 for h in hit if h in B.owned)
        if km:
            full += f" ({_expect_useful(d['pick'], d['n'], km, N):.2f} merge, no slot needed)"
    short = f"pool {N}, plan {k}, expect {ex:.2f} useful of {got}"
    if FV >= F9LL:
        # f9ll（2026-10-12：商店里有些效果牌可以抽英雄，有时不如选这些牌碰碰运气，看能不能抽到主C）：
        # 主C 还没定 / 路线主C 没到手时，写出抽到主C 人选（路线主C + 同体系备选；没有路线时 = 合格的 2 阶以上输出位）的概率
        ch = carry_hit(d, pool, B)
        if FV >= F9ZB:
            return f9zb_grant_text(N, note, k, names, ex, got, full, ch, B, d, hit)
        if ch:
            full, short = ch + "; " + full, ch + "; " + short
    return full, short


def carry_hit(d: dict, pool: list, B: "Board") -> str:
    """f9ll：发英雄牌抽到"主C 人选"的概率（至少见到 1 个）。主C 已定且在场 → 不写。"""
    if B.carry_set():
        return ""                  # 已有合格的主C（哪怕不是路线主C）：不再写抽主C 的概率（r9za 抽查：安琪拉 fit8 在场还写 CARRY hit）
    rc = (B.route or {}).get("carry") or ""
    if B.route and not B.route_stale:
        targets = [rc] + list(getattr(B, "backup_carries", []) or [])
    else:
        targets = [h for h in pool if is_output(h) and (fit(h) or 0) >= 6 and tier(h) >= 2 and B.carry_ok(h)
                   and not (FV >= F9Z and h in BLOCKED_CARRIES_F9Y)]       # f9z：屏蔽路线的主C 不算抽中主C 人选
    hit = [h for h in dict.fromkeys(targets) if h and h in pool and h not in B.owned]
    if FV >= F9ZA:
        hit = [h for h in hit if f9za_carry_listed(h)]      # f9za-fix（协调方 7；batch_020 e18f1895b0fa）：命中率只算 carry_pool 开着的主C
    if not hit:
        return ""
    q = len(hit) / len(pool)
    mp = re.match(r"choose:(\d+)选(\d+)", d["pick"])
    seen = int(mp.group(1)) if mp else (d["n"] * COUPON_CHOICE if d["pick"] == "coupon" else d["n"])
    pr = 1 - (1 - q) ** max(1, seen)
    return f"CARRY hit {pr:.0%} ({' '.join(hit[:3])})"


def exclusive_talent_name(n: str, desc: str = "") -> str:
    """OCR 读到的卡名 / 屏幕描述 → 阿离 6 级专属天赋的标准名（认不出返回 ""）。"""
    for x in ALI_EXCLUSIVE_TALENTS:
        if x in (n or ""):
            return x
    d = _norm_txt(desc)
    if len(d) >= 8:
        for x in ALI_EXCLUSIVE_TALENTS:
            sk = _norm_txt(skill(x))
            if sk and sum(ch in sk for ch in d) / len(d) >= 0.75 and d[:6] in sk:
                return x
    return ""


def exclusive_talent_fact(n: str) -> str:
    """写标准名（OCR 常把卡名读成别的卡，例：「开局狂飙」配着枫之洪流的描述）+ 卡面原文。"""
    if FV >= F9ZB:
        return f9zb_exclusive_text(n)
    return f"= Ali lv6 EXCLUSIVE talent {n}: " + re.sub(r"\s+", "", skill(n))


def fit_first_item(head: str, key: list) -> list:
    """第一条结论单独放不下时（pack 会整条丢掉）：发英雄牌换短写法，专属天赋原文逐字截短。其余照旧。"""
    if est_tokens(head + key[0]) <= OPT_TOKENS:
        return key
    first = key[0]
    if first.startswith("pool "):
        for alt in (re.sub(r" \([^)]*\)", "", first, count=1), re.sub(r"pool (\S+?)[,;].*?your plan (\d+).*?(expect [\d.]+ useful of \d+)",
                                                                 (r"pool \1, plan \2, \3" if FV >= F9LL else "pool , plan , "), first)):
            if est_tokens(head + alt) <= OPT_TOKENS:
                return [alt] + key[1:]
    if first.startswith("= Ali lv6 EXCLUSIVE"):
        t = first
        while len(t) > 30 and est_tokens(head + t + "…") > OPT_TOKENS:
            t = t[:-1]
        return [t + "…"] + key[1:]
    return key


# ---------------- f9li（2026-10-11）：已选天赋 / 阵容成型 ----------------
TALENT_NOTE_MAX = 6


def talent_lines(B: "Board") -> list:
    """已选天赋：名字 + 卡面前 50 字；能从文字和场上读数算出进度的写一句结论（事实，不写建议）。只用卡面原文 + 感知读数。"""
    names = [t for t in (B.ctx.get("talents") or []) if t]
    if not names:
        return []
    out = []
    for t in names[:TALENT_NOTE_MAX]:
        sk = re.sub(r"\s+", "", skill(t) or "")
        note = ""
        if re.search(r"品阶最高的\d?名?英雄", sk):
            on = [h for h in B.field if is_hero(h)]
            if on:
                top = max(on, key=lambda h: (tier(h) or 0))
                C = B.carry_set()
                note = f" → highest-tier on board now: {top} t{tier(top)}" + ("" if not C else (" (= carry)" if top == C else f" (NOT carry {C})"))
        m = re.search(r"每(?:留|保留|剩余)?(\d+)(?:点)?能量", sk)
        if m and isinstance(B.ctx.get("gold"), int):
            k = int(m.group(1))
            g = B.ctx["gold"]
            note = f" → energy {g}: {g // k} step(s) now, {k - g % k}e to the next"
        if re.search(r"图腾周围\d+格", sk):
            note = " → units next to totems get it (totems around the carry)"
        out.append(f"{t}（{sk[:50]}）{note}")
    return ["talents taken: " + "; ".join(out)]


def min_engine_tag(n: str, B: "Board", on_board: bool) -> str:
    """f9lj：这个英雄在路线最低配套里的角色。on_board=True 时写"卖掉会怎样"，否则写"拿来补哪一槽"。"""
    me = getattr(B, "min_eng", None) or {}
    if not me:
        return ""
    for g in min_engine_groups(me):
        if n in g:
            if on_board:
                mates = [x for x in g if x in B.field]
                if len(mates) > 1:
                    # 同一槽两人都在场（李元芳 + 程咬金）：等级高的那个算占槽，另一个是多余的（老师 r9y 反馈）
                    keep = max(mates, key=lambda x: (B.lv(x) or 0, -g.index(x)))
                    if n != keep:
                        return f"same MIN ENGINE slot as {keep} (one is enough)"
                return "MIN ENGINE piece on board: selling stalls the carry's level growth"
            have = [x for x in g if x in B.field]
            if have:
                return f"MIN ENGINE slot already filled by {have[0]}"
            return "MIN ENGINE piece missing (" + " or ".join(g) + "): fills the carry's level engine"
    _late_in = (bool(re.search(r"使[^。]{0,12}获得复生", skill(n) or "")) if FV >= F9T else n in (me.get("late_fill") or []))
    # f9t（r9zw / r9zx 老师多批：钟馗 / 少司缘 / 云中君被路线数据标成 LATE-FILL；后期补位件指的是太乙这类"给复生"的）：只认卡面复生件
    if (FV >= F9S and _late_in and not (FV >= F9Q and producer_kind(n)) and getattr(B, "route", None)
            and not (B.engine and B.engine_role(n) in ("producer", "amplifier"))
            and not (FV >= F9T and re.search(r"等级永久\+|永久等级\+", skill(n) or ""))):     # f9t：云中君（每件装备全队永久 +1 级）不是后期防守件（r9zw 老师）
        # f9s（2026-10-18 第 734 局：太乙这种可以最后补位，前面优先核心件；太早上太乙导致没有位置）：
        # 原来按回合（第 8 回合后就上）→ R9 卖狂铁上太乙，明世隐到手没位置、卖刘禅。改成按最终阵容：其余最终阵容件都在场之后才轮到它
        late = set(me.get("late_fill") or [])
        rest = [x for x in _routes().final_team(B.route) if x not in late and x != n]
        miss = [x for x in rest if x not in B.field]
        if FV >= F9ZB:
            zb = f9zb_late_fill(n, B, on_board, miss)
            if zb is not None:
                return zb
        if B.rnd and B.rnd < 12:
            # 2026-10-18：太乙的复活作用到后期才明显，前期优先攒等级；第 12 回合之前都降低优先级，没位置可以不上
            if on_board:
                return "LATE-FILL piece (revive/defense pays off late): before round 12 its slot yields to growth / final-team pieces"
            return "LATE-FILL piece (revive/defense pays off late): low priority before round 12 — early slots go to growth; no free slot → do not field"
        if miss:
            if on_board:
                return f"LATE-FILL piece on board early ({len(miss)} final-team piece(s) still missing: {', '.join(miss[:3])}): its slot goes to them first"
            return f"LATE-FILL piece: add LAST, after the rest of the final team ({len(miss)} missing: {', '.join(miss[:3])})"
        return ""
    if FV >= F9LK and (on_board or (B.engine and B.engine_role(n) in ("producer", "amplifier"))):
        return ""            # f9lk：已在场的不写"之后再上"；引擎零件不写 LATE-FILL（和 ENGINE PART wanted 打架）
    if n in (me.get("late_fill") or []) and B.rnd and B.rnd <= 8:
        if FV >= F9Q and producer_kind(n):
            # f9q（第 723 局 R8：刘禅"你的英雄释放技能时获得 2 临时等级"被路线数据放进 late_fill（防御/功能），主C 沈梦溪 + 明世隐在场，
            # 题面写"第 8 回合后再上"、喂主C 那句被挤掉；用技能就给临时等级的件非常适合沈梦溪）：卡面给等级的不是防御件
            return ""
        return "LATE-FILL piece (defense/utility): add after round 8, min engine first"
    return ""


def _engine_state(e: dict, B: "Board") -> str:
    """f9lk：引擎状态按**场上**算（审查：虞姬还在手里、最低配套 0/4，题面却写 COMPLETE）。"""
    rs = ("producers", "amplifiers", "payoff")
    on = {x for r in rs for x in e.get(r) or [] if x in B.field}
    roles_on = all(any(x in B.field for x in e.get(r) or []) for r in rs)
    if e["complete"] and roles_on and len(on) >= 2:
        return "COMPLETE on board"
    off = [x for r in rs for x in e.get(r) or [] if x not in B.field]
    head = "ASSEMBLED (not all on board)" if e["complete"] else ("STARTED" if B.engine_started else "NOT started")
    return head + f"; on board {len(on)}" + (f"; in hand: {', '.join(dict.fromkeys(off))}" if off else "")


def min_engine_line(B: "Board") -> str:
    """f9lj：路线最低配套进度（问题在于等级没滚起来 —— 每条路线让主C 涨等级的最小英雄集合）。"""
    me = getattr(B, "min_eng", None) or {}
    if not me:
        return ""
    gs = min_engine_groups(me)
    on = [g for g in gs if any(x in B.field for x in g)]
    inh = [g for g in gs if g not in on and any(x in B.hand_heroes for x in g)]
    miss = [g for g in gs if g not in on and g not in inh]
    nm = lambda g: "/".join(g)     # noqa: E731
    s = f"MIN ENGINE (level engine of route, {len(gs)} pieces): on board {len(on)}/{len(gs)}"
    if inh:
        s += "; in hand: " + ", ".join(nm(g) for g in inh)
    s += ("; missing: " + ", ".join(nm(g) for g in miss)) if miss else ("; all owned, not all on board" if FV >= F9LK and inh else "; COMPLETE")
    if FV >= F9LP and miss:
        s += shop_complete_text([next((x for x in g if x in B.shop_cnt), None) for g in miss], B, "the min engine")
    late = [x for x in me.get("late_fill") or [] if x in B.field and not any(x in g for g in gs)]
    if late and B.rnd and B.rnd <= 8 and (miss or inh):
        s += "; late-fill pieces taking slots: " + ", ".join(late)
    return s


def shop_complete_text(found: list, B: "Board", what: str) -> str:
    """f9lp：缺的人里哪些就在眼前商店里、一次买齐多少能量；全在商店里且钱和空位都够 → 写 COMPLETES。found 里 None = 那一件商店里没有。"""
    got = [x for x in found if x]
    if not got:
        return ""
    cost = 3 * len(got)
    gold = B.ctx.get("gold") if isinstance(B.ctx.get("gold"), int) else None
    s = f"; in shop now: {', '.join(got)} ({cost}e"
    s += (f", have {gold}e)" if gold is not None else ")")
    if len(got) == len(found):
        room = B.free if isinstance(B.free, int) else None
        if gold is not None and gold >= cost and (room is None or room >= len(got)):
            s += f": buying them now COMPLETES {what}"
        elif gold is not None and gold < cost:
            s += f": would complete {what} but short {cost - gold}e"
        elif room is not None and room < len(got):
            s += f": would complete {what} but only {room} empty slot(s)"
    return s


def flow_options_line(B: "Board", k: int = 3) -> str:
    """f9lp：流派 = 终点阵容 + 核心件（agent/sim/flows.FLOWS：官网阿离阵容按主C 合并 + 路线最低配套 + 遗志海月）。
    只列事实：每个流派已有几件、缺谁、缺的在不在商店里（agent/sim/flows.flow_rank，和局面网络特征同一份）。"""
    try:
        from agent.sim.flows import flow_rank
    except Exception:  # noqa: BLE001
        return ""
    shop = [x for x, c in sorted(B.shop_cnt.items()) for _ in range(c)]
    out = []
    for r in flow_rank(B.owned, shop)[:k]:
        if B.nw and r["final"] == (B.route or {}).get("carry"):
            # f9x（54.26-2）：路线主C 的流派里扣掉路线不要的人（狄仁杰流的钟馗）
            r = dict(r, have=[x for x in r["have"] if x not in B.nw], pieces=[x for x in r["pieces"] if x not in B.nw],
                     missing=[x for x in r["missing"] if x not in B.nw])
        txt = f"{r['final']} flow: have {len(r['have'])}/{len(r['pieces'])} ({', '.join(r['have'])})"
        if r["missing"]:
            txt += "; missing " + ", ".join(r["missing"][:5]) + (" …" if len(r["missing"]) > 5 else "")
            txt += shop_complete_text([x if x in r["in_shop"] else None for x in r["missing"]], B, "this flow's core")
        else:
            txt += "; core COMPLETE"
        out.append(txt)
    return ("FLOW OPTIONS (end comps from official lineups / route engines; facts only): " + " | ".join(out)) if out else ""


def lineup_line(B: "Board") -> str:
    """满员时：场上有几人属于主C 阵容 / 路线（主C + 配套），手里几人在等位置 —— 区分"换人"和"白买"（第 538 局满员还买东皇太一 / 刘禅）。"""
    if B.free != 0:
        return ""
    C = B.carry_set()
    team = set()
    if B.route and not B.route_stale:
        team |= set(_routes().final_team(B.route))
    if C:
        team |= {C} | {n for n, p in anchored(C)["mates"].items() if p >= 0.3}
    team -= B.nw                        # f9x（54.26-2）：路线不要的人算在 outside
    if not team:
        return ""
    on = [h for h in B.field if is_hero(h)]
    inn = [h for h in on if h in team]
    outn = [h for h in on if h not in team]
    return (f"lineup: {len(inn)}/{len(on)} on board are carry/route lineup"
            + (f" (outside: {', '.join(outn)})" if outn else " (FINAL lineup on board)")
            + f"; {len(set(B.hand_heroes))} hero(es) already waiting in hand")


def pure_second_c(n: str, B: "Board") -> bool:
    """f9li C-POSITION 只给"纯第二个 C"：不吃主C 引擎红利、不是路线成员、不在主C 阵容高频队友里（≥30%）、
    和主C 没有化学反应（喂等级 / 回蓝 / 放大）、不是贴主C 的辅助。（r9w 第一批 23 批老师一致反映：原定义把武则天 / 上官婉儿 /
    云中君等喂主C 的、敖隐 / 海诺等路线成员、瑶也标成了 C 位）"""
    C = B.carry_set()
    if not C or engine_fed_by(n, C) or B.route_member(n) or nearest_buddy(n) or is_support(n):
        return False
    if (B.anch.get("mates") or {}).get(n, 0) >= 0.3 or (anchored(C)["mates"].get(n, 0) >= 0.3):
        return False
    syn = carry_synergy(n, B) or ""
    if syn and "idle" not in syn and "no effect" not in syn.split(";")[0]:
        return False
    return True


# 审定条件改过的牌：旧版本题面用旧条件（2026-10-17：跳级"核心有等级就直接给主C"，不再要求场上有日落海；
# 新条件 core_level 是本局累计账本，单帧题面看不到 → f9o 起不写这条前提）
_LEGACY_WHEN = {"跳级": ["faction_on_field:日落海:1"]}


def usable_now_fact(n: str, B: "Board") -> str:
    """f9li：审定用法 when 里的场面前提（派系 / 关键词 / 场上人数 / 装备件数）当前不满足 → "UNUSABLE now: needs …"；满足返回 ""。
    只读审定数据和场上读数，是事实不是建议。"""
    u = usage().get(n) or {}
    whens = u.get("when") or []
    if FV < F9O and n in _LEGACY_WHEN:
        whens = _LEGACY_WHEN[n]       # f9o 前的题面按旧审定条件渲染（逐字不变）
    for w in whens:
        parts = w.split(":")
        k = parts[0]
        if k == "faction_on_field" and len(parts) >= 3:
            f, need = norm_fac(parts[1]), int(parts[2] or 1)
            have = sum(1 for h in B.heroes if h in B.field and fac(h) == f)
            if have < need:
                return f"UNUSABLE now: needs {need} 【{parts[1]}】 hero(es) on board (have {have})"
        elif k == "tag_on_field" and len(parts) >= 3:
            need = int(parts[2] or 1)
            have = sum(1 for h in B.heroes if h in B.field and parts[1] in (kw(h) or ""))
            if have < need:
                return f"UNUSABLE now: needs {need} hero(es) with {parts[1]} on board (have {have})"
        elif k == "equips_on_field" and len(parts) >= 2:
            need = int(parts[1] or 1)
            have = sum((B.eq(h) or 0) for h in B.heroes if h in B.field)
            if have < need:
                return f"UNUSABLE now: needs {need} items worn on board (have {have})"
        elif re.fullmatch(r"field>=\d+", k):
            need = int(k.split(">=")[1])
            have = sum(1 for h in B.heroes if h in B.field)
            if have < need:
                return f"UNUSABLE now: needs {need} heroes on board (have {have})"
    return ""


# ---------------------------------------------------------------- f9w（2026-10-19 下午，规范 54.25）
LOW_VALUE_TALENTS_F9W = {          # 54.25-4：复盘点名的低价值天赋（一句卡面后果）
    "节能商店": "discount lands on a random card",
    "限制性进化": "removes an unneeded low-level hero",
    "成年礼": "gives other-faction heroes",
}
TOP_TIER_TALENTS_F9W = ("随机成长", "随机强化", "随机进化")      # "你品阶最高的1名英雄等级永久+N"
LIFESTEAL_TALENTS_F9W = ("魂祭绝刃", "微弱优势", "强者恒强")      # 54.25-5：吸血类天赋
CARD_FAC_NEED_RE = re.compile(r"(所有|指定(\d*)名|每名|随机\d*名)【?(逐鹿|河洛|三分之地|三分|大河流域|大河|日落海|无阵营)】?英雄(?!牌)")


def f9w_carry(B: "Board") -> str:
    return B.carry_set() or B.cur_carry or B.fc or ""


def f9w_worn(B: "Board", h: str):
    """主C 身上穿的装备（穿戴账本 ctx.worn）。账本没有这人：视觉件数 0 → []；件数 >0 或读不到 → None（不知道）。"""
    w = (B.ctx.get("worn") or {}).get(h)
    if w:
        return list(w)
    return [] if B.eq(h) == 0 or (w is not None and not B.eq(h)) else None


def f9w_low_value_talent(n: str, B: "Board") -> str:
    if n in LOW_VALUE_TALENTS_F9W:
        return "LOW VALUE: " + LOW_VALUE_TALENTS_F9W[n]
    if n in TOP_TIER_TALENTS_F9W:
        C = f9w_carry(B)
        top = max((tier(h) or 0 for h in B.heroes), default=0)
        tops = [h for h in B.heroes if (tier(h) or 0) == top]
        if C and tops == [C]:
            return f"goes to the highest-TIER hero = carry {C} now"
        return "LOW VALUE: goes to the highest-TIER hero, not the carry"
    return ""


def f9w_lifesteal_fact(n: str, B: "Board") -> str:
    """54.25-5：高攻速主C 身上没有吸血装 → 吸血装 / 吸血天赋 / 送吸血装的牌写出来（被反伤甲弹死）。"""
    ht = _ht()
    ls = ht.lifesteal_items()
    hit = ((cat(n) == "装备牌" and n in ls) or n in LIFESTEAL_TALENTS_F9W
           or (cat(n) in ("天赋牌", "效果牌") and any(x in (skill(n) or "") for x in ls)))
    if not hit:
        return ""
    C = f9w_carry(B)
    if not C or not ht.attack_speed_carry(C) or (FV < F9X and C in ht.ATTACK_SPEED_ADDED_F9X):
        return ""
    w = f9w_worn(B, C)
    if w is None or set(w) & ls:
        return ""
    if FV >= F9X and n == "强者恒强":
        # f9x（54.26-6）：20% 吸血要"所有己方英雄已觉醒"，场上觉醒 0 人时不是补吸血
        aw, tot = len(B.awakened), len(B.heroes)
        if aw < tot or not tot:
            return f"carry {C} lacks lifesteal, but this needs ALL heroes awakened ({aw}/{tot} now) -> no lifesteal now"
        return f"lifesteal: carry {C} has none; all {tot} heroes awakened -> active"
    if FV >= F9X and n == "微弱优势":
        k = B.awk(C)
        if k == 0 and C in B.field:
            return f"carry {C} lacks lifesteal, but it is already awakened on board -> never triggers for it"
        return (f"carry {C} lacks lifesteal; only when its AWAKENED card is played"
                + (f" (awaken {3 - k}/3 now)" if isinstance(k, int) else ""))
    return f"lifesteal: carry {C} has none (anti-thorns)"


def f9w_cc_fact(n: str, B: "Board") -> str:
    """54.25-5：免控靠枫叶舞强化「枫影无拘」补 —— 主C 没有免控装、也还没选过时写出来。"""
    ht = _ht()
    if n != ht.CC_IMMUNE_MAPLE or n in (B.ctx.get("lord_boosts") or []):
        return ""
    C = f9w_carry(B)
    if not C:
        return ""
    w = f9w_worn(B, C)
    if w is None or set(w) & (set(ht.CC_IMMUNE_ITEMS) | {"机器头"}):
        return ""
    return "CC immunity: carry has no CC-immune item"


def f9w_equip_pick_fact(n: str, B: "Board") -> str:
    """54.25-5：N 选 1 全是装备时，每件写主C 件数 + 输出 / 防御（主C 没满装时输出优先）。"""
    if cat(n) != "装备牌":
        return ""
    C = f9w_carry(B)
    if not C:
        return ""
    ht = _ht()
    e = B.eq(C)
    if e is None:
        w = f9w_worn(B, C)
        e = len(w) if w is not None else None
    s = f"carry {C} {e if e is not None else '?'}/3; {f9w_item_kind(n)} item"
    if FV >= F9X and n not in base_equips():
        # f9x（54.26-21 / 54.26-17）：特殊装备对主C 能不能用（日落海专属 / 法术物理 / 普攻叠层）
        ok, why = f9x_item_usable(n, C)
        if why == "sunset-only":
            s += f"; sunset-only: carry {C} not 【日落海】"       # 普攻叠层装的 MISMATCH 由 card_facts 的适配行写
    return s


_EQ_REL = None


def f9w_item_kind(n: str) -> str:
    """装备输出 / 防御（equip_relations 的 category，按卡面分析；基础件按 hero_traits.base_kind）。"""
    global _EQ_REL
    if _EQ_REL is None:
        try:
            d = _load("data/equip_relations.json")
            _EQ_REL = d.get("items", d)
        except Exception:                                   # noqa: BLE001
            _EQ_REL = {}
    ht = _ht()
    v = _EQ_REL.get(n) if isinstance(_EQ_REL.get(n), dict) else {}
    cat_ = v.get("category") or ""
    if cat_ == "基础" or not cat_:
        bk = ht.base_kind(n)
        if bk in ("output", "defense"):
            return "output" if bk == "output" else "defensive"
        if not cat_:
            cls = ((ht.EQUIP_CLASS.get(n) or ("",))[0] or "")
            if cls.startswith("dps") or cls == "caster" or ht.item_dmg(n):
                return "output"
            if cls in ("tank", "survive"):
                return "defensive"
        return "mixed"
    if cat_.startswith("输出") or cat_ in ("暴击", "攻速", "施法"):
        return "output"
    if cat_ in ("坦克", "保命", "防御"):
        return "defensive"
    b = v.get("base") or {}
    if (b.get("carry") or 0) >= (b.get("tank") or 0) + 2:
        return "output"
    if (b.get("tank") or 0) > (b.get("carry") or 0):
        return "defensive"
    return "utility"


def f9w_card_faction_need(n: str, B: "Board") -> str:
    """54.25-6：效果牌卡面要场上有某阵营英雄（审定用法没写条件的，按卡面）而现在没有 → UNUSABLE now。"""
    if cat(n) != "效果牌":
        return ""
    m = CARD_FAC_NEED_RE.search(skill(n) or "")
    if not m:
        return ""
    need = 2 if "最近的【" in (skill(n) or "") else 1         # 换帅：与"最近的同阵营英雄"交换 → 要 2 名
    f = norm_fac(m.group(3))
    have = sum(1 for h in B.heroes if h in B.field and fac(h) == f)
    if have >= need:
        return ""
    return f"UNUSABLE now: needs {need} 【{m.group(3)}】 hero(es) on board (have {have})"


_TIES = None


def _talent_ties() -> dict:
    global _TIES
    if _TIES is None:
        f = PROJECT_DIR / "data" / "talent_hero_ties.json"
        _TIES = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return _TIES


def f9w_talent_hero_tie(n: str, B: "Board") -> str:
    """f9w（2026-10-19 第 787 局：献祭之赐送李信、主C 沈梦溪，选了又卖；送英雄或跟特定英雄有关的天赋都要打上标记）：
    天赋点名的英雄（data/talent_hero_ties.json）在不在路线最终阵容 / 核心引擎 / 主C 里。没锁路线不写。"""
    t = _talent_ties().get(n) or {}
    hs = t.get("heroes") or []
    if not hs or not B.route or B.route_stale:
        return ""
    if FV >= F9X and t.get("kind") == "duo" and duo_talent_fact(n, B):
        return ""           # f9x（54.26-23）：共战天赋由 duo 行判（两人都要在路线里），"送的英雄在路线里"不单独加分
    R = _routes()
    team = set(R.final_team(B.route)) | set((getattr(B, "min_eng", None) or {}).get("early_priority") or [])
    team |= {x for x in (B.cur_carry, B.fc, (B.route or {}).get("carry")) if x}
    hit = [h for h in hs if h in team]
    names = "/".join(hs)
    if hit:
        return f"names {names}: in route ({'/'.join(hit)})"
    if t.get("kind") == "gives_plus":
        return f"names {names}: NOT in route (only the extra gain counts)"
    return f"names {names}: NOT in route -> no value"



# ---------------------------------------------------------------- f9x（2026-10-19 晚，规范 54.26）
@lru_cache(maxsize=256)
def _as_carry(h: str) -> bool:
    """hero_traits.attack_speed_carry 每次都读 cards.json，这里缓存。马可波罗 f9x 起才算攻速型（ATTACK_SPEED_ADDED_F9X，只在 f9x 调用）。"""
    return bool(h) and bool(_ht().attack_speed_carry(h))


def f9x_attack_speed_carry(B: "Board") -> str:
    C = f9w_carry(B)
    return C if C and _as_carry(C) else ""


LOW_VALUE_TALENTS_F9X = {          # 54.26-16/19/20/23 + 10-19 阿离天赋适配审查"建议新增 C"（一句卡面后果）
    "群星璀璨": "needs 5+ t4-5 heroes on board (one-carry comps never get there); only the 2 vouchers count",
    "兵行诡招": "costs your own HP",
    "极限操作": "costs your own HP",
    "搏命": "costs your own HP",
    "拾荒者": "t5 vouchers only after eliminations; our carry / core are t1-3",
    "随机进化": "+6 lv to the highest-TIER hero, usually not the carry",
    "零钱兑换": "turns 古币 into 破损古币 that give no levels",
    "大买卖": "buys cost +1e, sells give +1e: we buy more than we sell",
    "整装待发": "pays only on rounds with no hero card played (we play heroes every round)",
    "小小的助力": "buffs t4-5 heroes only; carry is t1-3",
    "四重奏": "needs 3+ factions (neutral not counted); our routes run 1 faction",
    "临时增幅": "levels land on random heroes (~1/7 on the carry)",
    "银算盘": "levels land on a random hero (~1/7 on the carry)",
    "金算盘": "levels land on a random hero (~1/7 on the carry)",
    "有舍有得": "random awakening, and every merge gives -1 lv",
    "多彩古币": "its 古币 can't go on 【河洛】 heroes and transforms the target",
    "一起发财": "pays heroes of OTHER factions, not the carry",
    "丝绸之路": "needs 4 board sells per 古币; too slow",
    "乘胜追击": "tactic cards only target 【逐鹿】 heroes",
    "演武": "upgrades tactic cards; we have no tactic source",
    "厉兵秣马": "2 more items only if the lord level stays 5 rounds; we level up",
    "虚构装备": "temp items go to 3 random heroes",
    "虚构武装": "temp items go to 3 random heroes",
    "丢盔弃甲": "removes a hero's equip slots",
    "五福临门": "random hero cards: rarely the route's",
    "双人游": "random hero cards: rarely the route's",
    "神秘英雄": "random hero cards: rarely the route's",
    "瓶中人": "random hero cards: rarely the route's",
    "二象性": "random hero-card effect: rarely the route's",
    "永不独行": "random hero cards: rarely the route's",
    "慕名而来": "random hero cards: rarely the route's",
    "全军列阵": "fills the board with random same-faction heroes",
    "小咖秀": "needs a t1-3-only board; random board copy",
    "镜像迷离": "steals only from same-name enemy heroes",
    "防御领域": "only reduces damage to the lord; no board effect",
    "绝对领域": "only reduces damage to the lord; no board effect",
    "前线堡垒": "small defense for the front row only",
    "猪猪": "one 3s crowd control per battle",
    "始祖熊灵": "a 大河 bear that doesn't level the carry",
}
LOW_VALUE_T45_F9X = ("三人游", "天兵", "大明星", "河洛精兵", "大河精兵")      # 只给 4-5 / 5 阶英雄（54.26-20 同理；路线缺 4-5 阶件时例外）
LOW_VALUE_ACE_F9X = ("王牌认证", "王牌部队")                                    # 要 ≥3 名 4-5 阶
LOW_VALUE_FAC_F9X = {"三分新援": "三分之地", "三分强援": "三分之地", "逐鹿新援": "逐鹿", "逐鹿增援": "逐鹿", "逐鹿强援": "逐鹿"}


def f9x_route_team(B: "Board") -> set:
    """路线最终阵容（扣掉 not_wanted）+ 核心引擎 + early_priority + extra_wanted + 主C。没锁路线 → 空。"""
    if not B.route or B.route_stale:
        return set()
    me = B.min_eng or {}
    team = set(_routes().final_team(B.route)) | set(me.get("early_priority") or []) | set(me.get("extra_wanted") or [])
    for g in min_engine_groups(me):
        team |= set(g)
    team |= {x for x in (B.route.get("carry"), B.cur_carry, B.fc) if x}
    return team - B.nw


def f9x_low_value_talent(n: str, B: "Board") -> str:
    C = f9w_carry(B)
    if n == "因缘际会":
        # 10-19 晚裁决：图腾同少司缘（每 25 次普攻全队 +1）→ 主C 攻速型可选，否则 C
        if C and not _as_carry(C):
            return f"LOW VALUE: totem pays on basic attacks; carry {C} is not an attack-speed carry"
        return f"totem pays on basic attacks: attack-speed carry {C}" if C else ""
    if n == "协同一致" and C and fac(C) == "无阵营":
        return ""                                   # 审查表：公孙离（无阵营）主C 例外
    if n in LOW_VALUE_TALENTS_F9X:
        return "LOW VALUE: " + LOW_VALUE_TALENTS_F9X[n]
    if n in LOW_VALUE_T45_F9X:
        miss = [h for h in f9x_route_team(B) if (tier(h) or 0) >= 4 and h not in B.owned]
        if miss:
            return f"t4-5 heroes only; route still misses t4-5 {'/'.join(sorted(miss)[:2])}"
        return "LOW VALUE: t4-5 heroes only; our carry / core are t1-3"
    if n in LOW_VALUE_ACE_F9X:
        have = sum(1 for h in B.heroes if (tier(h) or 0) >= 4)
        return "" if have >= 3 else f"LOW VALUE: needs 3+ t4-5 heroes (have {have})"
    if n in LOW_VALUE_FAC_F9X:
        f = LOW_VALUE_FAC_F9X[n]
        team = f9x_route_team(B)
        if team and not any(fac(h) == f for h in team):
            return f"LOW VALUE: gives 【{FACTION_EN.get(f, f)}】 heroes, none in route; only the energy counts"
        return ""
    return f9w_low_value_talent(n, B)


def f9x_duo_fact(pair: list, B: "Board") -> str:
    """54.26-23（10-19 晚：只有两人都有用时才选）：两人都在路线最终阵容 / 核心引擎里、且其一是主C 才有用。"""
    team = f9x_route_team(B)
    C = f9w_carry(B)
    if not team or (all(h in team for h in pair) and C in pair):
        return ""                       # 没锁路线：按旧口径（主C 在不在其中）
    return f"duo {'+'.join(pair)}: needs both in route" + (f" (carry {C})" if C else "") + " -> no value"


def f9x_item_usable(item: str, h: str) -> tuple[bool, str]:
    """特殊装备对这个人能不能用（卡面硬条件：阵营专属 / 法术物理 / 普攻叠层 / 穿戴限制 / 适配分）。"""
    if not can_wear(item, h):
        return False, "can't wear"
    sk = skill(item) or ""
    if "阿科米亚核心" in sk and "参与" in sk:
        ok = fac(h) == norm_fac("日落海")
        return ok, ("" if ok else "sunset-only")
    H = _ht()
    if H.onhit_mismatch(item, h, 2 if FV >= F9Z else 1):
        return False, "basic-attack item, SKILL damage"
    it = H.item_dmg(item)
    d = _dmg(h)
    if it in ("mag", "phys") and d in ("mag", "phys") and d != it:
        return False, f"{'magic' if it == 'mag' else 'physical'} item, {h} {d}"
    sc = H.item_fit_score(item, h, (hero_roles().get(h) or {}).get("combat_role"))
    if sc is not None and sc < 5:
        return False, f"fit {sc:g}/10"
    return True, ""


def f9x_special_none_usable(cands: list, B: "Board") -> str:
    """54.26-21（第 794 局 R4：守约主C，静谧之铃 / 阿科米亚之子 / 聚灵法杖，这三件其实都不能用）。"""
    if len(cands) < 2 or not all(cat(n) == "装备牌" and n not in base_equips() for n in cands):
        return ""
    C = f9w_carry(B)
    if not C or any(f9x_item_usable(n, C)[0] for n in cands):
        return ""
    pool = [h for h in dict.fromkeys(list(B.heroes) + sorted(f9x_route_team(B)))
            if is_hero(h) and h != C and not (is_support(h) and h not in NO_GEAR_EXEMPT_F9X)]
    who = [h for h in pool if any(f9x_item_usable(n, h)[0] for n in cands)]
    return f"special items: none usable by carry {C}; usable by: " + ("/".join(who[:4]) if who else "none")


TRIG_HERO_RE_F9X = re.compile(r"每当你的(?:英雄|单位)触发(整备|登场|开团)(?:或(整备|登场|开团))?效果")
TEAM_WIDE_RE_F9X = re.compile(r"你的英雄每(穿戴|释放)")


def f9x_trigger_fact(n: str, B: "Board") -> str:
    """54.26-18（第 791 局 R12：卖了云中君、留了朵莉亚）：条件触发型写场上触发来源；全队按件数 / 次数吃的写 team-wide。"""
    if not is_hero(n):
        return ""
    sk = skill(n) or ""
    m = TRIG_HERO_RE_F9X.search(sk)
    if m:
        ks = [k for k in m.groups() if k]
        src = [h for h in B.heroes if h != n and any(f"{k}：" in (skill(h) or "") or f"{k}:" in (skill(h) or "") for k in ks)]
        return f"{'/'.join(ks)} sources on board: {len(src)}" + (f" ({', '.join(src[:4])})" if src else "")
    if TEAM_WIDE_RE_F9X.search(sk):
        if "穿戴" in sk:
            worn = sum((B.eq(h) or 0) for h in B.heroes)
            return f"team-wide: every hero +lv per worn item ({worn} items worn on board)"
        return "team-wide: every hero gains from its own casts"
    return ""


def f9x_shaosiyuan_fact(n: str, B: "Board") -> str:
    """54.26-24（10-19 晚：攻速型主C 的核心英雄买不到时，可用少司缘补位）。"""
    if n != "少司缘":
        return ""
    C = f9x_attack_speed_carry(B)
    if not C:
        return ""
    team = f9x_route_team(B)
    late = FV >= F9Y and "少司缘" in ((B.min_eng or {}).get("late_fill") or [])
    if "少司缘" in team and not late:
        return ""                       # 路线本身要她（马可波罗日落海 / 大河开团）：已有 ROUTE 标签
    if late:
        # f9y（54.27-16）：路线把她标成 late_fill（马可波罗日落海）——核心件没到时照样当补位
        late_all = set((B.min_eng or {}).get("late_fill") or [])
        if team and all(h in B.owned for h in team - late_all - {"少司缘"}):
            return ""
    elif team and all(h in B.owned for h in team):
        return ""
    if B.free is not None and B.free <= 0 and n not in B.field and not f9x_offroute_on_board(B):
        return ""                       # 满员、场上没有路线外杂牌可换 → 上不了场
    return "attack-speed carry: 少司缘 totem = filler while core missing"


def f9x_offroute_on_board(B: "Board") -> list:
    team = f9x_route_team(B)
    return [h for h in B.heroes if team and h not in team and h not in (B.fc, B.cur_carry)]


def f9x_no_deploy(n: str, B: "Board") -> str:
    """54.26-25（第 795 局 R14：7/7 满员买杨玉环）：上不了场、又不是路线最终阵容 / 核心引擎件的英雄。"""
    if n in B.field or n in B.hand or n in (B.fc, B.fc_in_hand, B.cur_carry):
        return ""
    if n in f9x_route_team(B) or n in B.flex_core or (B.route and n == B.route.get("carry")):
        return ""
    if f9x_shaosiyuan_fact(n, B):
        return ""                       # 攻速主C 的少司缘补位：有路线外杂牌可换（54.26-24）
    if FV >= F9Z and (n in f9z_route_pieces(B) or n in f9z_candidate_route_pieces(B)):
        return ""                       # f9z（54.28-14，3efc05504a51 盾山）：候选路线主C / 核心 / 最终阵容件的第一张照买，不写"上不了场"
    return "cannot be deployed: no value unless merging"


def f9z_candidate_route_pieces(B: "Board") -> set:
    """f9z（54.28-14）：路线没锁时，路线题候选路线（route_candidates，已排除屏蔽路线）里**主C 已拥有、≥2 阶**的那几条的主C + 最终阵容件，
    加上同一主C 流派（FLOW OPTIONS）的核心件。"""
    if B.route and not B.route_stale:
        return set()
    if getattr(B, "_f9z_cand", None) is None:
        out, crs = set(), set()
        try:
            for lu in route_candidates(B, banned=B.ctx.get("banned") or ""):
                cr = lu.get("carry") or ""
                if cr and cr in B.owned and (tier(cr) or 0) >= 2:
                    crs.add(cr)
                    out |= ({cr} | set(_routes().final_team(lu))) - route_nw(lu)      # 狄仁杰线的钟馗（not_wanted）不算
        except Exception:                                   # noqa: BLE001
            pass
        try:
            # 题面 FLOW OPTIONS 同一份（flow_rank 前 3）：主C 已拥有的流派的核心件（3efc05504a51：守约流缺盾山、盾山在店）
            from agent.sim.flows import flow_rank
            shop = [x for x, k in sorted(B.shop_cnt.items()) for _ in range(k)]
            for r in flow_rank(B.owned, shop)[:3]:
                if r.get("final") in crs:
                    _lu = next((lu for lu in route_candidates(B, banned=B.ctx.get("banned") or "") if lu.get("carry") == r.get("final")), None)
                    out |= set(r.get("pieces") or []) - (route_nw(_lu) if _lu else set())
        except Exception:                                   # noqa: BLE001
            pass
        B._f9z_cand = out
    return B._f9z_cand


def _onhit_lv():
    """hero_traits 普攻装档位：f9z 起 2（加 BASIC_ATTACK_ITEMS_2：影牙 / 破晓 / 日渊 / 铸梦·逐风）；f9x/f9y 1（True）；更早 False。"""
    return 2 if FV >= F9Z else (FV >= F9X)


def f9x_onhit_text(item: str, h: str, B: "Board") -> str:
    """规范 54.26-17：普攻叠层装（hero_traits.BASIC_ATTACK_ITEMS）给技能型主C。"""
    if not h or not _ht().onhit_mismatch(item, h, 2 if FV >= F9Z else 1):
        return ""
    who = "carry" if h in (B.fc, B.cur_carry, f9w_carry(B)) else h
    return f"basic-attack item, {who} deals SKILL damage -> MISMATCH"


# ---------------------------------------------------------------- f9y（2026-10-20，规范 54.27）
def _tcd():
    from agent.decision import talent_conditions as _t
    if FV < F9Z:
        # 10-20：表在 f9y 之后又改过（买橘子去 / 传家宝 / 执行能力复审）→ f9y 题面读冻结版 data/talent_conditions_f9y.json，渲染逐字不变
        return _t.frozen("f9y")
    if FV < F9ZA:
        # 10-20 晚：按主C 逐个复审（data/review/talents_carry.json）又改了表 → f9z（R50 训练用的文本）读冻结版 data/talent_conditions_f9z.json
        return _t.frozen("f9z")
    # f9za：冻结表 data/talent_conditions_f9za.json + 线上求值口径（最终可摆阵容 / grade_if / 韩信攻速型）
    return _t.frozen("f9za", live=True)


TACTIC_TALENTS_F9Y = ("令行如流", "乘胜追击", "演武")       # 54.27-23：战术牌只给【逐鹿】英雄


def f9y_talent_team(B: "Board") -> tuple[set, str, str]:
    """与线上 talent_low_value 同口径：team = 路线最终阵容（+extra_wanted −not_wanted）+ 场上 + 手里；carry = 现任主C（没有就路线主C / 未来主C）；
    route_carry = 已锁且有效路线的主C。"""
    locked = bool(B.route) and not B.route_stale
    team = set(B.heroes) | {h for h in B.hand if is_hero(h)}
    if locked:
        me = B.min_eng or {}
        team |= (set(_routes().final_team(B.route)) | set(me.get("extra_wanted") or [])) - set(me.get("not_wanted") or [])
    rc = (B.route.get("carry") or "") if locked else ""
    if FV >= F9Z:
        # f9z（54.28-2 / 54.23-2，与线上 laya_policy talent_low_value 同口径）：路线主C 没到手（carry X not owned yet）= 路线没定，不按他判；
        # 现任是 1 阶临时主C 时不判主C 类条件（传空，同线上）；没有现任主C 时按未来主C（≥2 阶，54.23-2）
        rc = rc if rc and rc in B.owned else ""
        cc = B.ctx.get("carry") or ""
        if rc:
            C = rc
        elif cc:
            C = cc if (tier(cc) or 0) >= 2 else ""
        else:
            fut = f9w_carry(B)
            C = fut if fut and (tier(fut) or 0) >= 2 else ""
        return team, C, rc
    C = (B.ctx.get("carry") or "") or rc or f9w_carry(B)
    return team, C, rc


def _f9y_cond(c: dict, n: str, team: set, C: str, rc: str) -> tuple[str, bool | None]:
    """一条硬条件 → (英文描述, 满足? / None = 判不了或只是说明)。判定与 talent_conditions.check 一致。"""
    T = _tcd()
    heroes = {h for h in team if is_hero(h)}
    t = c.get("type")
    if t == "never":
        return "!never useful for one-carry Ali", False
    if t == "note":
        return str(c.get("text") or ""), None
    if t == "faction_on_team":
        f, k = c.get("faction") or "", int(c.get("min") or 1)
        have = sum(1 for h in heroes if T._fac(h) == f)
        if n in TACTIC_TALENTS_F9Y and f == "逐鹿":
            if FV >= F9Z:
                # f9z（review-data-consistency 第 12 条 / 54.27-24②）：特训 / 演练战术给谁都行，推演 / 援护 / 招募只给逐鹿
                return (f"!tactic cards mostly 【逐鹿】-only (特训/演练 work on anyone): {have} 逐鹿 heroes in route/board/hand"
                        + (" -> no value" if have < k else "")), have >= k
            return f"!tactic cards only for 【逐鹿】: {have} 逐鹿 heroes in route/board/hand" + (" -> no value" if have < k else ""), have >= k
        return f"{k}+ 【{f}】 heroes in route/board/hand (have {have})", have >= k
    if t == "carry_faction":
        f = c.get("faction") or ""
        if not C:
            return f"carry 【{f}】 (no carry yet)", None
        return f"carry 【{f}】 ({C} is 【{T._fac(C) or '?'}】)", T._fac(C) == f
    if t in ("attack_speed_carry", "skill_carry"):
        w = "attack-speed" if t == "attack_speed_carry" else "skill-damage"
        if not C:
            return f"{w} carry (no carry yet)", None
        fast = bool(_ht().attack_speed_carry_live(C)) if FV >= F9ZA else bool(_as_carry(C))     # f9za：韩信算攻速型（LIVE_FAST_ADDED）
        return f"{w} carry ({C})", fast if t == "attack_speed_carry" else not fast
    if t == "heroes_in_team":
        hs = c.get("heroes") or []
        inn = [h for h in hs if h in heroes]
        allm = c.get("mode") == "all"
        ok = len(inn) == len(hs) if allm else bool(inn)
        if FV >= F9ZB:
            return f"{'all' if allm else 'any'} of {'/'.join(hs)} (have {'/'.join(inn) or 'none'})", ok
        return f"{'all' if allm else 'any'} of {'/'.join(hs)} in route/board/hand (have {'/'.join(inn) or 'none'})", ok
    if t == "carry_in":
        hs = c.get("heroes") or []
        if not C:
            if FV >= F9ZA:
                return f"carry {'/'.join(hs)} (no carry yet → counts as not met)", False      # f9za：同线上 unknown_carry_fails
            return f"carry {'/'.join(hs)} (no carry yet)", None
        return f"carry {'/'.join(hs)} (carry now {C})", C in hs
    if t == "route_carry_is":
        hs = c.get("heroes") or []
        if not rc:
            return f"route carry {'/'.join(hs)} (route not locked)", None
        return f"route carry {'/'.join(hs)} (route carry now {rc})", rc in hs
    if t == "high_tier_count":
        ts = set(c.get("tiers") or [4, 5])
        k = int(c.get("min") or 1)
        have = sum(1 for h in heroes if T._tier(h) in ts)
        return f"{k}+ t{'/'.join(map(str, sorted(ts)))} heroes (have {have})", have >= k
    if t == "lord_level_reach":
        return f"lord reaches lv {c.get('level')}", None
    return str(t), None


def f9y_talent_lines(n: str, B: "Board") -> list[str] | None:
    """f9y（2026-10-20：天赋数量不多，按 data/talent_conditions.json 逐张复审）：每个天赋一行结论。
    C 或硬条件不满足 → LOW VALUE: 原因；B 且条件满足 → conditional: 条件 (met)；绑定主C 的（幸运硬币）→ commits the route to X。
    表里没有这张 → None（调用方按 f9x 写）。"""
    T = _tcd()
    e = T.table().get(n)
    if not isinstance(e, dict) or n.startswith("_"):
        return None
    team, C, rc = f9y_talent_team(B)
    if FV >= F9ZA:
        # f9za（同线上 laya_policy talent_low_value）：主C 未定按不满足（unknown_carry_fails）+ 按最终可摆阵容计数（rules.final_board 口径）+ grade_if 上调
        _bd = f9za_talent_board(B, C)
        g, _why = T.check(n, team, C, rc, unknown_carry_fails=True, board=_bd)
    else:
        g, _why = T.check(n, team, C, rc)
    out = []
    if e.get("commit_route_carry"):
        out.append(f"commits the route to {e['commit_route_carry']}")
    if FV >= F9ZA:
        conds = [(f9za_final_cond(c, C, _bd) if c.get("type") in T.FINAL_TYPES else _f9y_cond(c, n, team, C, rc))
                 for c in e.get("conditions") or []]
        if g == "C" and _why and not any(ok is False for _d, ok in conds):
            conds.append((_why, False))       # 线上判 C 的原因（carry_in 主C 未定等）题面也写出来
        if g != "C" and (e.get("grade") or "?") == "C":
            # grade_if 上调（如 4 阶主C 的随机成长、蒙犽的逐鹿系）：写出成立的条件
            for gi in e.get("grade_if") or []:
                ws = [(f9za_final_cond(c, C, _bd) if c.get("type") in T.FINAL_TYPES else _f9y_cond(c, n, team, C, rc)) for c in gi.get("when") or []]
                if ws and all(ok is not False for _d, ok in ws):
                    _sure = all(ok is True for _d, ok in ws)
                    out.append(("worth it here: " if _sure else "upgraded only if: ") + "; ".join(d.lstrip("!") for d, _ok in ws))
                    break
    else:
        conds = [_f9y_cond(c, n, team, C, rc) for c in e.get("conditions") or []]
    if g == "C" and FV >= F9Z and not any(ok is False for _d, ok in conds):
        ch = f9z_carry_lottery(n, B)
        if ch:
            # f9z（54.28-1）：随机给英雄的天赋（五福临门等）、路线主C / 主C 还没到手、抽到主C 人选 ≥40% → 按"买一张主C 候选"，不写 LOW VALUE
            out.append(ch)
            return out
    if g == "C":
        bad = next(((d[1:] if d.startswith("!") else "needs " + d) for d, ok in conds if ok is False), "")
        if FV >= F9Z and str(e.get("why") or "").startswith("需要特殊操作"):
            # f9z（54.28-20，10-20：带特殊操作、执行层做不到的就不要选）：买橘子去 / 站这别动 / 班叔的遗留物
            bad = "needs special placement (a hero must keep standing on a chosen cell) - our executor can't do it -> no value"
        if not bad:
            bad = LOW_VALUE_TALENTS_F9X.get(n) or LOW_VALUE_TALENTS_F9W.get(n) or ""
        if not bad:
            w = re.split(r"[。；]", str(e.get("why") or ""))[0]
            bad = w[:60]
        out.append("LOW VALUE: " + (bad or "low value for one-carry Ali"))
    elif g == "B":
        hard = [d.lstrip("!") for d, ok in conds if ok is True]
        soft = [d for d, ok in conds if ok is None and d]
        if hard:
            out.append("conditional: " + "; ".join(hard) + " (met)")
        if soft:
            out.append("conditional: " + "; ".join(soft))
    return out


def f9z_carry_lottery(n: str, B: "Board") -> str:
    """f9z（54.28-1）：发英雄牌的天赋写着 CARRY hit ≥40%（carry_hit：主C 未定 / 路线主C 没到手时才写）→ 一句"抽主C"的结论；否则 ""。"""
    try:
        gf = grant_fact(n, B)
    except Exception:                                   # noqa: BLE001
        return ""
    m = re.match(r"CARRY hit (\d+)% \(([^)]*)\)", (gf or ("", ""))[0] or "")
    if not m or int(m.group(1)) < 40:
        return ""
    return f"carry lottery: CARRY hit {m.group(1)}% ({m.group(2)}), carry not owned yet -> worth about one carry-candidate buy"


def f9y_special_fit(n: str, B: "Board") -> str:
    """f9y（第 811 局 R8：天赋界面三选一特殊装备选了法潮灵杖（法系）给物理马可波罗；review-805-814）：
    每件写对主C 的伤害类型适配（物理 / 法术、普攻 / 技能），不对口写 MISMATCH。"""
    if FV >= F9ZA:
        # f9za（10-21 待办：天赋三选一里直接给装备的选项写对主C 的适用性，第 855 局沈梦溪选了末世）：同线上 hero_traits.live_fits，
        # 分类表没有的读卡面（靠普攻的只给攻速型主C）；基础装备也写
        C = f9y_talent_team(B)[1]
        return f9za_item_line(n, B, carry=C) if (C and is_hero(C)) else ""
    if cat(n) != "装备牌" or n in base_equips():
        return ""
    C = f9y_talent_team(B)[1]
    if not C or not is_hero(C):
        return ""
    H = _ht()
    d = _dmg(C)
    src = H.damage_source(C)
    who = f"carry {C} ({d or '?'}" + (", attack-speed" if _as_carry(C) else (f", {src}" if src else "")) + ")"
    sk = skill(n) or ""
    if "阿科米亚核心" in sk and "参与" in sk and fac(C) != norm_fac("日落海"):
        return f"for {who}: sunset-only item -> MISMATCH"
    if FV >= F9Z and "固定0.4的攻击速度" in sk and C not in SPEED_LOCK_CARRIES_F9Z:
        # f9z（待办 / 54.28-3：天赋界面机器头对沈梦溪 / 亚连写 any-damage item fits）：机器头只给花木兰 / 嬴政
        return f"for {who}: speed lock 0.4, only 花木兰/嬴政 -> MISMATCH"
    if H.onhit_mismatch(n, C, 2 if FV >= F9Z else 1):
        return f"for {who}: basic-attack item, carry deals SKILL damage -> MISMATCH"
    it = H.item_dmg(n)
    if it in ("mag", "phys") and d in ("mag", "phys") and d != it:
        return f"for {who}: {'magic' if it == 'mag' else 'physical'} item -> MISMATCH"
    ok, why = f9x_item_usable(n, C)
    if not ok:
        return f"for {who}: {why or 'not usable'} -> MISMATCH"
    return f"for {who}: {('magic' if it == 'mag' else 'physical' if it == 'phys' else 'any-damage')} item fits"


def f9y_awaken_vs_lv6(n: str, B: "Board") -> str:
    """f9y（54.27-14 / 21：买下就能让核心件觉醒 vs 一回合内攒够升 6 级）：5 级商店里两件事同时成立时并写，不下结论。"""
    if B.level != 5 or not is_hero(n):
        return ""
    team = f9x_route_team(B) | {x for x in (f9w_carry(B), B.fc, B.cur_carry) if x}
    if not (B.route and not B.route_stale):
        team |= set(B.flex_core)          # 路线没锁：候选路线核心卡也算核心件
    if n not in team:
        return ""
    k = B.awk(n) if n in B.field else None
    if not (k == 1 or (n not in B.field and B.hand.count(n) == 2)):
        return ""
    m = meta()
    ctx = B.ctx
    g, lc, rnd = ctx.get("gold"), ctx.get("level_cost") or -1, B.rnd
    c = int((cards().get(n) or {}).get("cost") or 3)
    if not (isinstance(g, int) and lc > 0 and rnd):
        return ""
    nc = lc - m.lord_xp_gain(rnd)
    inc = m.energy_income(rnd + 1)
    if g - c >= lc:
        lv6 = "lv6 now still OK after buy"
    elif nc <= 0:
        lv6 = "lv6 at round end anyway"
    elif g - c + inc >= nc:
        lv6 = f"lv6 next round OK after buy ({g - c}+{inc}≥{nc}e)"
    elif g + inc >= nc:
        lv6 = f"buy delays lv6 ({g - c}+{inc}<{nc}e; saving reaches it)"
    else:
        lv6 = f"no lv6 next round even saving ({g}+{inc}<{nc}e)"
    return f"AWAKENS core {n}; {lv6}"



# ---------------------------------------------------------------- f9za（2026-10-21；R51 用；规范 54.29 / 54.30 / 54.31）
# 只在 FV >= F9ZA 时被调用；FV < F9ZA 的渲染一律不经过这里（旧版本题面必须逐字不变）。
LORD_USELESS_F9ZA = ("阿科米亚之子", "驱雷", "月影之力")                             # 同 rules.LORD_USELESS_ITEMS（冻结副本，渲染确定）
NOT_FOR_CARRY_F9ZA = ("静谧之铃", "捕能手套", "唤鹰骨笛", "阿科米亚之子")           # 同 rules.NOT_FOR_CARRY_LIVE
ITEM_DMG_F9ZA = {"制裁之剑": "phys"}                                                 # 同 rules.ITEM_DMG_LIVE（hero_traits 旧表记成通用输出）
PAIR_EQUIPS_F9ZA = ("怒莲之刺", "银莲庇佑")                                          # 同 rules.PAIR_EQUIPS（守约等也可带怒莲之刺）
MINGSHIYIN_F9ZA = "明世隐"
GAME_END_ROUND_F9ZA = 15            # 估算"剩余回合"按第 15 回合结束（54.31-1）
SACRIFICE_FUEL_RE_F9ZA = re.compile(r"拥有牺牲效果的英雄阵亡")
AS_TEXT_RE_F9ZA = re.compile(r"(提升|[+＋])\s*\d+%?\s*(攻速|攻击速度)|(攻速|攻击速度)\s*[+＋]\s*\d")   # 同 rules：守约 / 沈梦溪 / 诸葛亮不穿加攻速的


def _lc_f9za():
    from agent.decision import lord_config as _L
    return _L.load()


def f9za_route_blocked(lu: dict) -> bool:
    """f9za（10-21 审查：路线题先截 3 条再删）：路线按棋手配置排除 —— 真主C（carry_of_lineup，官网首位不一定是主C）关着，
    或阵容本身 blocked_now。与 rules 锁路线的合法性（rules.py 1075 / 1412）同口径。没有配置 → 退回冻结名单。
    10-21 已决定：关掉的主C 临时顶上时也不锁他的路线（54.28-12 / 54.29-24 的"放行"作废）→ 不再给现任主C 开例外。"""
    c = _lc_f9za()
    key = str(lu.get("key") or "")
    if not c.present:
        return str(lu.get("name") or "").startswith(WEAK_ROUTES_F9Z) or lu.get("carry") in WEAK_CARRIES_F9Z
    real = c.carry_of_lineup(key) or lu.get("carry") or ""
    return (not c.enabled(real)) or c.blocked_lineup(key)


def f9za_carry_off(n: str) -> bool:
    """f9za 换主C 题：名单里关掉的 2 阶以上主C（1 阶的是临时主C，照常）。与 laya_policy.WEAK_CARRIES 同口径。"""
    c = _lc_f9za()
    if not c.present:
        return n in ("安琪拉", "马可波罗")
    return n in c.disabled() and c.tier(n) != 1


LISTED_TAKEOVER_F9ZA = "listed carry (should take over)"


def f9za_stage_ok(n: str, B: "Board") -> bool:
    """路线阶段件算"计划内"：路线件（最终阵容 / early_priority / 最低配套 / extra_wanted）、主C 阵容占比 ≥80%，或当前阶段的 main。"""
    if n in f9z_route_pieces(B) or (B.anch["mates"].get(n) or 0) >= 0.8:
        return True
    for i, st in enumerate((B.route or {}).get("stages") or []):
        fr, to = st.get("from_round") or 0, st.get("to_round") or 99
        if fr <= (B.rnd or 0) <= to and n in (st.get("main") or []):
            return True
    return False


def f9za_carry_listed(n: str) -> bool:
    """carry_pool 里开着的主C（名单外 / 关掉的都不算）。没有配置 → True。"""
    c = _lc_f9za()
    return (not c.present) or c.enabled(n)


def f9za_listed_takeover(n: str, B: "Board") -> bool:
    """f9za-fix（协调方 10；dup_002 dd45f758a413）：n 是开着的 ≥2 阶主C、在场 / 等位，而现任 / 路线主C 是名单外（或关掉）的人 → 应接班。"""
    c = _lc_f9za()
    if not c.present or not is_hero(n) or n not in B.owned or not c.enabled(n) or (c.tier(n) or tier(n) or 0) < 2:
        return False
    cur = B.cur_carry or B.ctx.get("carry") or ((B.route or {}).get("carry") if (B.route and not B.route_stale) else "") or ""
    if FV >= F9ZB and f9zb_committed_offlist(B, cur):
        return False                    # f9zb（54.37-1）：现任是已认的名单外主C → 名单内的人不写接班；"with carry" 按现任
    return bool(cur) and cur != n and not c.enabled(cur)


def f9za_syn_carry(B: "Board", C: str) -> str:
    """f9za-fix（54.32-16；r9zzg cc3588628c99 / 40e36cf418a0 / 88338723c755）："with carry" 读数按谁写：现任主C C 在 carry_pool 关着、
    而开着的主C 接班人（f9za_listed_takeover）在场 / 等位 → 改按接班人（未来主C 优先）；否则原样返回 C。"""
    try:
        c = _lc_f9za()
        if not C or not c.present or c.enabled(C):
            return C
        cand = [B.fc] if B.fc else []
        cand += sorted(h for h in B.field if h != B.fc)
        return next((h for h in cand if h and h != C and f9za_listed_takeover(h, B)), C)
    except AttributeError:          # Board 还没初始化完（__init__ 里早调）→ 原样
        return C


def f9za_borrow_engine(route_key: str) -> dict:
    """f9za（同 rules.min_engine 10-21）：锁上的阵容在引擎表里没有条目 → 借同一主C 收录过的阵容（配置里的核心阵容优先）。"""
    key = str(route_key or "")
    try:
        carry = (_routes().by_key(key) or {}).get("carry") or ""
    except Exception:                                   # noqa: BLE001
        carry = ""
    carry = carry or _lc_f9za().carry_of_lineup(key)
    if not carry:
        return {}
    base, f9v = (_MIN_ENG or {}), _min_eng_f9v()
    keys = [k for k in base if not str(k).startswith("_")] + [k for k in f9v if k not in base and not str(k).startswith("_")]

    def _cv(k):
        v = f9v.get(k) if isinstance(f9v.get(k), dict) else (base.get(k) if isinstance(base.get(k), dict) else {})
        return (v or {}).get("carry")
    pref = _lc_f9za().lineup_key(carry)
    alt = pref if pref and pref != key and _cv(pref) == carry else next((k for k in keys if k != key and _cv(k) == carry), "")
    return dict(min_engine_of(alt), borrowed_from=alt) if alt else {}


def f9za_core_names(me: dict) -> list:
    """引擎表 core_set 里点名的英雄（同 rules.min_engine：英雄名出现在 unit 文本里就算）。"""
    out = []
    for u in me.get("core_set") or []:
        t = str(u.get("unit") or "")
        out += [h for h in _hero_names() if h in t]
    return list(dict.fromkeys(out))


def f9za_carry(B: "Board") -> str:
    """f9za 估算 / 适配用的主C：已锁路线的主C（到手了的）> 已定主C / 现任 / 未来主C。"""
    rc = (B.route.get("carry") or "") if (B.route and not B.route_stale) else ""
    if rc and rc in B.owned:
        return rc
    return f9w_carry(B) or rc


# f9za-fix（协调方 13；r9zzg_test dup_000 524768b39f4e / 5ad2a7db05fc：第 15 回合写成 ×~0 rounds）：
# 第 845~863 局我方实际结束回合（states.jsonl 最大 round_no；冻结副本，渲染确定）。剩余回合 = 活到本回合的那些局的平均结束回合 − 本回合，至少 2；
# 没有比本回合更晚结束的局 → max(2, 18 − 本回合)
END_ROUNDS_F9ZA = (12, 12, 17, 12, 17, 5, 11, 16, 11, 14, 10, 15, 13, 15, 11, 18, 16, 17)


def f9za_rounds_left(B: "Board") -> int:
    r = B.rnd or 0
    later = [e for e in END_ROUNDS_F9ZA if e > r]
    if not later:
        return max(2, 18 - r)
    return max(2, int(round(sum(later) / len(later) - r)))


def f9za_temp_sources(B: "Board") -> tuple:
    """临时等级来源（不含明世隐本人）：(场上的, 路线里还没到手的)。"""
    on = [h for h in B.temp_src if h != MINGSHIYIN_F9ZA]
    rt = sorted(h for h in (f9x_route_team(B) | f9z_route_pieces(B)) if h not in B.field and h != MINGSHIYIN_F9ZA
                and any(g.get("res") == "临时等级" for g in (kb().get(h) or {}).get("gives", [])))
    return on, rt


def f9za_msy_gain(B: "Board") -> str:
    """明世隐多触发一次登场：之后每回合多转 +2（觉醒 +4）临时等级 → 主C 永久等级（54.31-1）。"""
    if MINGSHIYIN_F9ZA not in B.field:
        return ""
    C = f9za_carry(B)
    left = f9za_rounds_left(B)
    aw = B.awk(MINGSHIYIN_F9ZA) == 0
    per = 4 if aw else 2
    who = f"carry {C}" if C else "nearest hero (no carry yet)"
    st = "awakened" if aw else "not awakened"
    on, rt = f9za_temp_sources(B)
    if FV >= F9ZB:
        # f9zb（check_dropped_lines：原行 ≈50 token，单条放不进选项）
        w = f"carry {C}" if C else "nearest hero"
        if not on and not rt:
            return f"{w} ≈+{per} lv once (明世隐 +{per}, no temp src)"
        return f"{w} ≈+{per * left} lv (明世隐 +{per}/rd ×{left}, src {on[0] if on else rt[0] + ' later'})"
    if not on and not rt:
        # f9za-fix（54.32-5）：场上 / 路线都没有临时等级来源 → "×剩余回合"兑现不了，按一次性收益写（选项只有 ~44 token，写短）
        return f"{who} ≈ +{per} lv once: 明世隐 ({st}) converts +{per} more temp lv; no temp-lv source (no ×rounds)"
    src = "/".join(on[:2]) if on else ("/".join(rt[:1]) + " not on board yet")
    return f"{who} ≈ +{per * left} lv: 明世隐 ({st}) converts +{per} more temp lv/round × ~{left} rounds (src {src})"


def f9za_level_gain(n: str, B: "Board") -> str:
    """f9za（54.31-1，第 853 局 R15 选有福同享不选唤醒）：涨等级的效果 / 天赋牌，估算给主C 的等级收益（一行；读不出来 → ""）。"""
    if cat(n) not in ("效果牌", "天赋牌"):
        return ""
    sk = re.sub(r"^转瞬\s*", "", skill(n) or "")
    C = f9za_carry(B)
    cw = f"carry {C}" if C else "carry (none yet)"
    if "触发其登场效果" in sk or "触发所有英雄的登场效果" in sk:
        if f9za_msy_gain(B) and (usable_now_fact(n, B) or f9w_card_faction_need(n, B)):
            return f"{cw}: UNUSABLE now (hold for later)"      # f9za-fix（54.32-5）：同选项写着 UNUSABLE now 时不写 ≈ +N lv
        return f9za_msy_gain(B)
    if "额外触发1次登场效果" in sk:
        # 锦囊：本回合打出的登场英雄再触发一次 → 本回合上场的明世隐多转一次
        if MINGSHIYIN_F9ZA in B.hand and MINGSHIYIN_F9ZA not in B.field:
            left = f9za_rounds_left(B)
            if FV >= F9ZB:
                return f"{cw} ≈+{2 * left} lv if 明世隐 deployed now (+2/rd ×{left})"
            return f"{cw} ≈ +{2 * left} lv if 明世隐 is deployed this round (+2 more conversion/round × ~{left} rounds)"
        return ""
    if re.search(r"等级最低[^。]{0,16}(提升\d+级|获得[^。]{0,8}等级)|等级随机分配给[^。]{0,8}等级最低", sk):
        return f"{cw} +0: goes to the LOWEST-level unit, never the carry"     # 扶倾济弱 / 扫拂 / 传承（一个响指是消除，不算）
    m = re.search(r"场上和其品阶相同的英雄等级\+(\d+)", sk)
    if m:
        k = int(m.group(1))
        if not C:
            return f"+{k} to every unit of the chosen tier"
        mates = [h for h in B.heroes if h != C and tier(h) == tier(C)]
        return f"{cw} +{k} once" + (f" (same-tier t{tier(C)} also +{k}: {'/'.join(mates[:3])})" if mates else f" (no other t{tier(C)} unit)")
    m = re.search(r"指定1名【([^】]+)】英雄等级\+(\d+)", sk)
    if m:
        k = int(m.group(2))
        if not C:
            return f"+{k} to one 【{m.group(1)}】 hero"
        ok = fac(C) == norm_fac(m.group(1))
        return f"{cw} +{k} once" if ok else f"{cw} +0: only 【{m.group(1)}】 heroes, carry is 【{fac(C) or '?'}】"
    m = re.search(r"获得(\d+)张天赋牌[^：]*：指定1名英雄等级\+(\d+)(?:。(\d+)回合后，再\+(\d+))?", sk)
    if m:
        k, v = int(m.group(1)), int(m.group(2))
        tot = k * v
        if m.group(3) and f9za_rounds_left(B) >= int(m.group(3)):
            tot += k * int(m.group(4))
        return f"{cw} +{tot} total ({k} card{'s' if k > 1 else ''})"
    m = re.search(r"每消耗(\d+)能量，获得1张[^：]*：指定1名英雄等级\+(\d+)", sk)
    if m:
        return f"{cw} +{m.group(2)} per {m.group(1)}e spent"
    m = re.search(r"^指定1名英雄等级\+(\d+)(?:，随机1名英雄等级\+(\d+))?", sk)
    if m:
        k = int(m.group(1))
        if "场上每有1种阵营" in sk:
            nf = len({fac(h) for h in B.heroes if fac(h) and fac(h) != "无阵营"})
            return f"{cw} +{k * (1 + nf)} once ({nf} factions on board)"
        return f"{cw} +{k} once" + (f" (+{m.group(2)} to a random unit)" if m.group(2) else "")
    m = re.search(r"^(?:所有|全体(?:己方)?)英雄(?:的)?等级\+(\d+)", sk)
    if m:
        return f"{cw} +{m.group(1)} once (every unit +{m.group(1)})" + ("; grows +5 per round held" if "每持有一回合" in sk else "")
    m = re.search(r"各阵营中随机1名英雄等级\+(\d+)", sk)
    if m and C:
        k = sum(1 for h in B.heroes if fac(h) == fac(C)) or 1
        return f"{cw} +{m.group(1)} with ~1/{k} chance (random 【{fac(C) or '?'}】 unit)"
    return ""


def f9za_item_fit(item: str, C: str) -> tuple:
    """f9za（10-21 待办：题面 FITS 和执行层对齐）：这件装备对主C 能不能用 → (True / False / None 不知道, 英文原因)。
    同 rules 穿装合法性 + hero_traits.live_fits；分类表没收录的读卡面：靠普攻生效的只给攻速型主C（同 laya_policy 天赋三选一）。"""
    H = _ht()
    sk = skill(item) or ""
    if item in LORD_USELESS_F9ZA:
        return False, "useless for lord Ali (other lords' set item)"
    if item == "星源星徽":
        return False, "never worth taking"
    if item in NOT_FOR_CARRY_F9ZA:
        return False, "adds no carry output: not for the carry"
    if C in H.NO_ONHIT_CARRIES and item not in PAIR_EQUIPS_F9ZA and AS_TEXT_RE_F9ZA.search(sk):
        return False, f"attack-speed item, {C} is a skill carry"
    d = H.dmg_type(C)
    if ITEM_DMG_F9ZA.get(item) and d in ("mag", "phys") and ITEM_DMG_F9ZA[item] != d:
        return False, f"{'physical' if ITEM_DMG_F9ZA[item] == 'phys' else 'magic'} item, {C} {d}"
    if "固定0.4的攻击速度" in sk and C not in SPEED_LOCK_CARRIES_F9Z:
        return False, "speed lock 0.4, only 花木兰/嬴政"
    if "限定近战" in sk and not H.is_melee(C):
        return False, f"melee heroes only, {C} ranged"        # f9za-fix（batch_024 328c10b46ec0：青莲剑谱对虞姬写 FITS）
    if "限定远程" in sk and H.is_melee(C):
        return False, f"ranged heroes only, {C} melee"
    if FV >= F9ZB and item in RANGED_ONLY_F9ZB and H.is_melee(C):
        return False, f"ranged heroes only, {C} melee"        # f9zb（10-21 第 884 局：自然法杖只能远程用；同 hero_traits.RANGED_ONLY）
    ok, why = H.live_fits(item, C)
    if why == "unclassified":
        it0 = H.item_dmg(item)
        if it0 in ("mag", "phys") and d in ("mag", "phys") and it0 != d:
            # f9za-fix（r9zzg batch_009 da438202de9f：聚灵法杖对亚连写 FITS，详情行写 magic item → MISMATCH）：伤害类型不对先判
            return False, f"{'magic' if it0 == 'mag' else 'physical'} item, {C} {d}"
        if "普通攻击" in sk or "普攻" in sk:
            # 规范 54.29-18：普攻装只禁技能型的守约 / 沈梦溪 / 诸葛亮（NO_ONHIT_CARRIES）；亚连 / 韩信这类近战普攻照穿
            bad = C in H.NO_ONHIT_CARRIES
            return (not bad), (f"basic-attack effect, {C} is a skill carry (no on-hit)" if bad else "basic-attack effect, carry attacks")
        it = H.item_dmg(item)
        if it in ("mag", "phys") and d in ("mag", "phys"):
            return it == d, f"{'magic' if it == 'mag' else 'physical'} item, {C} {d}"
        return None, ""
    why = re.sub(r"^live: ", "", str(why or ""))
    if why.startswith("equip_relations"):
        why = "item relations table"
    return bool(ok), why


UTIL_RANK_EN_F9ZA = ("control (stun / pig / freeze)", "heal / team buff", "self-defense", "damage only")


def f9za_item_line(item: str, B: "Board", wear: str = "", carry: str | None = None) -> str:
    """一行：对主C 的适配（主C 穿得下时）；穿戴对象不是主C（主C 满装）→ 写辅助性等级（54.31-5：控制 > 回血 / 团队 > 自保 > 输出）。"""
    if cat(item) != "装备牌":
        return ""
    C = f9za_carry(B) if carry is None else carry
    if C and (not wear or wear == C):
        ok, why = f9za_item_fit(item, C)
        if ok is None:
            return ""
        return f"for carry {C}: " + ("FITS" if ok else "NO USE") + (f" ({why})" if why else "")
    if not wear:
        return ""
    r = _ht().utility_rank(item)
    return f"for non-carry {wear}: {UTIL_RANK_EN_F9ZA[r]}" + (" → prefer control/heal items" if r == 3 else "")


UTIL_SHORT_F9ZA = ("control", "heal/team", "self-defense")


def f9za_full_others(n: str, C: str) -> str:
    """f9za-fix（54.32-11）：主C 满装 —— 基础装备写出可合成的辅助性成品（控制 > 回血 / 团队 > 自保）；成品照旧写自己的辅助等级。"""
    H = _ht()
    rec = _load_recipes().get(n)
    if rec:
        prods = sorted(((H.utility_rank(x), x) for x in rec if H.utility_rank(x) < 3), key=lambda t: (t[0], t[1]))
        if prods:
            return f"carry {C} full: for others = forges into " + " / ".join(f"{UTIL_SHORT_F9ZA[r]} {x}" for r, x in prods[:3])
        return f"carry {C} full: for others = damage only (no utility forge)"
    return f"carry {C} full: for others = {UTIL_RANK_EN_F9ZA[H.utility_rank(n)]}"


def f9za_special_all_bad(names: list, B: "Board") -> str:
    """f9za-fix（协调方 11；54.29-15 / 54.32-33；r9zzg batch_024 328c10b46ec0）：天赋三选一全是"特殊高级装备"（静谧之铃这一档：
    不是 8 件基础装备、也不是四选一成品 —— 同 rules._is_normal_equip）且三件都配不上主C（f9za_item_fit 不是 True）→ 返回一句；否则 ""。
    与线上 laya_policy._talent_decide 的 _spec_bad 同口径（不分回合可刷）。"""
    if len(names) < 2:
        return ""
    rec = _load_recipes() or {}
    normal = set(rec) | {x for v in rec.values() for x in (v or [])}
    if not all(cat(n) == "装备牌" and n not in normal for n in names):
        return ""
    C = f9za_carry(B)
    if not C or any(f9za_item_fit(n, C)[0] is True for n in names):
        return ""
    return f"special items: none fit carry {C}"


def f9za_detail_fit(n: str, B: "Board", key: list) -> list:
    """f9za-fix：候选详情里的装备适配 —— 主C 没满装、f9za_item_fit 有结论时，旧的适配片段（fits / does NOT fit / magic item; X phys → MISMATCH /
    speed lock … MISMATCH）不再写，换成和选项行同一句（以 f9za_item_fit 为准，不出现相反结论）。"""
    C = f9za_carry(B)
    if not C or (C in B.field and isinstance(B.eq(C), int) and B.eq(C) >= 3):
        return key
    ln = f9za_item_line(n, B)
    if not ln:
        return key
    return [ln] + [k for k in key if not (F9ZA_OLD_FIT_RE.search(k) or "MISMATCH" in k or k.startswith("speed lock 0.4"))]


def f9za_sacrifice_fuel(n: str, B: "Board") -> str:
    """f9za（54.31-3）：守约线 —— 卡面"每当你拥有牺牲效果的英雄阵亡"→ 带牺牲的英雄（读技能原文）是主C 的法力燃料。"""
    if not is_hero(n):
        return ""
    C = f9za_carry(B)
    if not C or C == n or not SACRIFICE_FUEL_RE_F9ZA.search(skill(C) or ""):
        return ""
    if "牺牲" not in (skill_all(n) or ""):
        return ""
    # f9za-fix（54.32-3）：燃料只是路线外的人之间的加分项；别家主C 不挂；路线内的照写 ROUTE fuel
    if n in (f9x_route_team(B) | f9z_route_pieces(B)) or B.route_member(n):
        return f"ROUTE fuel: has 牺牲, its death cuts {C} mana cap -20"
    if B.other_lineup_carry(n) or any((lu.get("carry") or "") == n for lu in _routes().ali_lineups()):
        return ""
    return f"off-route sacrifice fuel (bonus vs other off-route units only): death cuts {C} mana cap -20"


def f9za_bought_note(x: str, B: "Board") -> str:
    """f9za（54.31-2，第 853 局 R6 买蒙恬又不上场）：等位的人是本回合 / 上回合 Laya 主动买的新英雄（线上 laya_policy._laya_bought → ctx.laya_bought）。"""
    r = (B.ctx.get("laya_bought") or {}).get(x)
    if not isinstance(r, int) or not B.rnd:
        return ""
    if r == B.rnd:
        return "bought this round (our own buy)"
    if B.rnd - r == 1:
        return "bought last round (our own buy)"
    return ""


def f9za_full_buy_note(n: str, B: "Board") -> str:
    """f9za（54.31-2）：商店题满员时买新英雄（不是合成同名）= 要卖掉一个路线外的人；没人可卖 → 不该买。"""
    if not is_hero(n) or n in B.field or n in B.hand or B.free is None or B.free > 0:
        return ""
    if B.mismatch:
        return ""           # f9za-fix（r9zzg batch_007 5e222c09b5a9）：场上名单读不全 → 不知道能卖谁，不写 none（留 "needs a swap"）
    sell = f9za_sellable(B)
    if sell:
        return "board full: buying means selling one of: " + ", ".join(sell[:3])
    if f9za_core_buy(n, B):
        return "board full, core piece: buy and hold (deploy when a slot frees) or lock"      # 54.32-1
    return "board full: buying means selling one of: none → don't buy"


def f9za_core_buy(n: str, B: "Board") -> bool:
    """54.32-1：路线核心 / 最终阵容 / 引擎件 / 主C 本人（含同名）—— 满员没人可卖也不写 don't buy。
    f9za-fix（r9zzg_c 77fd288c69c5 花木兰、batch_006 李信 / 莱西奥 / 蒙恬；54.29-4）：路线锁定后只认路线件（f9z_route_pieces：最终阵容 =
    lineup#1、early_priority_full ∪ core_set、最低配套、extra_wanted）、路线主C 阵容占比 ≥80% 的人、主C 同名；路线外的引擎产出方
    （带牺牲 / 整备）、别家主C、SECONDARY 都不算。路线没锁才按引擎角色认，且别家主C / SECONDARY 除外。"""
    locked = bool(B.route) and not B.route_stale
    if n in {x for x in (B.fc, B.cur_carry, B.ctx.get("carry"), f9za_carry(B), (B.route or {}).get("carry")) if x}:
        return True
    if locked:
        if n in f9z_route_pieces(B) or B.route_core_engine(n):
            return True
        rc = B.route.get("carry") or f9za_carry(B)
        return bool(rc) and n not in B.nw and not B.other_lineup_carry(n) and not B.secondary_engine_piece(n) \
            and (anchored(rc)["mates"].get(n) or 0) >= 0.8
    if B.other_lineup_carry(n):
        return False
    if B.engine and B.engine_role(n) in ("producer", "amplifier", "payoff"):
        return True
    C = B.carry_set() or B.fc
    if C and (anchored(C)["mates"].get(n) or 0) >= 0.8:
        return True          # 54.32-9：主C 阵容占比 ≥80% = 最终阵容件
    return False


def f9za_engine_piece(h: str, B: "Board") -> bool:
    """f9za-fix（r9zzg_c 9367c47cf741 / caa0e23917c5：满员买人把明世隐列成可卖）：核心引擎件 = 路线引擎表 early_priority_full ∪ core_set、
    路线核心引擎、当前引擎的放大器、卖了引擎就断的产出方、临时等级来源（明世隐这类）。冗余的路线外产出方不算。"""
    me = B.min_eng or {}
    if h in set(me.get("early_priority") or []) | set(f9za_core_names(me)) or B.route_core_engine(h):
        return True
    if h in (B.temp_src or []):
        return True
    if B.engine and (B.engine_role(h) == "amplifier"
                     or (B.engine_role(h) == "producer" and B.engine_critical(h))):
        return True
    return False


def f9za_lineup_outside(B: "Board") -> list:
    """同 lineup_line 的 outside 口径（路线最终阵容 ∪ 主C 阵容占比 ≥30% − not_wanted 之外的场上英雄）。"""
    C = B.carry_set()
    team = set()
    if B.route and not B.route_stale:
        team |= set(_routes().final_team(B.route))
    if C:
        team |= {C} | {x for x, p in anchored(C)["mates"].items() if p >= 0.3}
    team -= B.nw
    if not team:
        return []
    return [h for h in B.field if is_hero(h) and h not in team]


def f9za_sellable(B: "Board") -> list:
    """f9za-fix（54.32-1；r9zzg batch_008 db3b12bf99a4、batch_015 04ca5c19599a / 19ab2c6119e1）：满员买人时可卖的人，
    与同题 "outside: X" / "would replace X" / "NO free slot (sell e.g. X)" / "can replace filler X" 同一口径：
    路线件（f9x_route_team ∪ f9z_route_pieces）、主C / 未来主C / 枫叶舞宿主、卖了引擎就断的零件不卖；其余路线外的人都算（喂主C 的路线外件也算）。"""
    keep = {x for x in (B.fc, B.cur_carry, B.ctx.get("carry"), B.ctx.get("maple"), f9za_carry(B)) if x}
    keep |= {h for h in B.heroes if f9za_listed_takeover(h, B)}           # 开着的主C 接班人不卖（协调方 10）
    pieces = f9x_route_team(B) | f9z_route_pieces(B)

    def _crit(h):
        return bool(B.engine) and B.engine_role(h) in ("producer", "amplifier") and B.engine_critical(h)
    order = sorted(B.heroes, key=lambda h: (tier(h), B.lv(h) or 0, h))
    if pieces:
        main = [h for h in order if h not in pieces and h not in keep and not _crit(h)]
    else:
        main = [h for h in B.fodder() if h not in keep]
    extra = list(f9za_lineup_outside(B)) + list(B.fodder())
    extra += [h for h in sorted(B.heroes) if is_filler(h, B) and (h not in pieces or B.route_past_stage(h))]
    extra = [h for h in extra if h in B.field and is_hero(h) and h not in keep and not _crit(h)]
    # f9za-fix（r9zzg_c 9367c47cf741 / caa0e23917c5：明世隐经 fodder / lineup_outside 漏进可卖名单）：可卖只能是路线外、非引擎、
    # 非主C、非开着的主C 接班人 —— 路线件、核心引擎件、路线主C 阵容占比 ≥80% 的人一律不卖
    rc = (B.route.get("carry") or "") if (B.route and not B.route_stale) else ""
    C80 = rc or B.carry_set() or B.fc
    m80 = {x for x, p in anchored(C80)["mates"].items() if p >= 0.8} if C80 else set()
    return [h for h in dict.fromkeys(main + extra)
            if h not in pieces and h not in m80 and h != rc and not f9za_engine_piece(h, B)]


def f9za_talent_board(B: "Board", C: str) -> dict | None:
    """同 rules.final_board：路线没锁 → None。must = 主C + 核心引擎件（early_priority ∪ core_set）+ 路线里的后期件（太乙：卡面"使…获得复生"）；
    cap = 满级人口 + 人口天赋（ctx.pop_cap 超出等级表的部分）。"""
    if not B.route or B.route_stale:
        # f9za-fix（协调方 9；r9zzg batch_021 ce9d7bbf0019：王牌部队场上已有 3 名 4-5 阶，路线没锁写成静态 LOW VALUE）：
        # 路线没锁 → 用"场上 + 手里"的实际阵容当最终阵容，人口上限按当前上限
        now = {h for h in B.heroes} | {h for h in B.hand if is_hero(h)}
        if not now:
            return None
        cap0 = B.cap if isinstance(B.cap, int) and B.cap > 0 else (B.ctx.get("pop_cap") if isinstance(B.ctx.get("pop_cap"), int) else len(now))
        must0 = {C} & now if C else set()
        return {"final": set(now), "must": must0, "pool": set(now), "cap": int(cap0), "_now": True}
    m = meta()
    carry = (B.route.get("carry") or "") or C or ""
    me = B.min_eng or {}
    team = ((set(_routes().final_team(B.route)) | set(me.get("extra_wanted") or [])) - set(me.get("not_wanted") or [])) | ({carry} if carry else set())
    eng = set(me.get("early_priority") or []) | set(f9za_core_names(me))
    late = {h for h in team if re.search(r"使[^。]{0,12}获得复生", skill(h) or "")}
    must = ({carry} | eng | late) - {""}
    final = team | must
    pool = final | set(B.heroes) | {h for h in B.hand if is_hero(h)}
    cap = B.ctx.get("pop_cap")
    bonus = (cap - m.pop_cap_at(B.level)) if (B.level and isinstance(cap, int) and cap >= m.pop_cap_at(B.level)) else 0
    return {"final": final, "must": must, "pool": pool, "cap": max(m.POP_CAP_BY_LEVEL.values()) + int(bonus or 0)}


def f9za_final_cond(c: dict, C: str, board: dict | None) -> tuple:
    """*_final 类条件（最终可摆阵容计数）→ (英文描述, 满足? / None)。判定复用 talent_conditions._final_check（不另写一份）。"""
    T = _tcd()
    ok, why = T._final_check(c, C, board)
    t = c.get("type")
    d = {"carry_tier_in": f"carry tier {'/'.join(map(str, c.get('tiers') or []))}",
         "carry_has_keyword": f"carry skill has 「{c.get('kw')}」",
         "faction_count_final": f"{c.get('min')}+ same-faction heroes placeable on the final board"
                                + (f" (【{c['faction']}】)" if c.get("faction") else ""),
         "faction_kinds_final": f"{c.get('min')}+ factions in the final lineup",
         "tier_count_final": f"{c.get('min')}+ t{'/'.join(map(str, c.get('tiers') or []))} in the final lineup",
         "carry_is_top_tier_final": "carry is the top-tier unit of the final lineup",
         "keyword_count_final": f"{c.get('min')}+ final-lineup heroes with 「{c.get('kw')}」"}.get(t, str(t))
    if ok is None:
        return d + (" (route not locked)" if t not in ("carry_tier_in", "carry_has_keyword") else ""), None
    if FV >= F9ZA and board and board.get("_now") and t not in ("carry_tier_in", "carry_has_keyword"):
        d = d.replace("the final lineup", "board+hand now").replace("the final board", "board+hand now") + " (route not locked)"
    return (d + (": " + why if (ok is False and why) else "")), ok


F9ZA_OLD_FIT_RE = re.compile(r"^(fits|does NOT fit) |MISMATCH|^basic-attack item|^(magic|physical) item(;|$)|^for carry .*(fits|MISMATCH)")


def f9za_option_lines(n: str, B: "Board", ctx: dict, kind: str, prefix: str, key: list) -> list:
    """f9za：option_text 的新行（商店 / N 选 1 / 拍卖 / 天赋 / 卖手牌共用），放在最前（选项预算有限）。"""
    pre = []
    if cat(n) == "装备牌" and not (kind in ("", "talent") and ctx.get("_talent_q")):      # 天赋题由 f9y_special_fit 写
        C = f9za_carry(B)
        full = bool(C) and C in B.field and isinstance(B.eq(C), int) and B.eq(C) >= 3
        ln = (f9za_full_others(n, C) if full else f9za_item_line(n, B))
        if ln:
            pre.append(ln)
            key = [k for k in key if not F9ZA_OLD_FIT_RE.search(k)]      # 旧的适配行（口径和执行层不一致）不再写
    lg = f9za_level_gain(n, B)
    if FV >= F9ZB and f9zb_entry_all_now(n, B):
        pre.append(USE_NOW_F9ZB)                # f9zb（54.36-6）：阵容定型、场上有明世隐 → 集体唤醒照用
    if lg:
        pre.append(lg)
    if is_hero(n):
        if ctx.get("_shop_q") and prefix.startswith("buy"):
            fb = f9za_full_buy_note(n, B)
            if fb:
                pre.append(fb)
                key = [k for k in key if k != "board full: needs a swap to deploy"]      # 同一件事，新行更具体
                if "selling one of: none" not in fb:
                    # f9za-fix（54.32-1）：已写出可卖的人 → 旧的 "NO free slot (only core heroes / only carry …)" 是相反结论，不再写
                    key = [k for k in key if not re.match(r"NO free slot \(only", k)]
        fu = f9za_sacrifice_fuel(n, B)
        if fu:
            pre.append(fu)
    return pre + [k for k in key if k not in pre]


def f9za_use_gain(n: str, t: str, B: "Board") -> str:
    """f9za（54.31-1）：用牌题 —— 把涨等级的牌用在 t 身上，主C 估算拿到多少等级。"""
    sk = re.sub(r"^转瞬\s*", "", skill(n) or "")
    C = f9za_carry(B)
    if "触发其登场效果" in sk:
        return f9za_msy_gain(B) if t == MINGSHIYIN_F9ZA else ""
    m = re.search(r"场上和其品阶相同的英雄等级\+(\d+)", sk)
    if m:
        k = int(m.group(1))
        same = [h for h in B.heroes if tier(h) == tier(t)]
        return (f"+{k} to all t{tier(t)} ({len(same)} units)" + (f"; carry {C} +{k}" if C and tier(C) == tier(t)
                                                                 else (f"; carry {C} +0" if C else "")))
    return ""          # 固定加等级给主C：原行 "lv a→b" 已写，不重复


_use_on_facts_f9z = use_on_facts


def use_on_facts(n: str, t: str, B: "Board") -> list[str]:          # noqa: F811  f9za：在原函数前面加一行主C 等级收益（FV < F9ZA 原样返回）
    out = _use_on_facts_f9z(n, t, B)
    if FV >= F9ZA:
        g = f9za_use_gain(n, t, B)
        if g:
            out = [g] + [x for x in out if x != g]
    return out



# ---------------------------------------------------------------- f9zb（2026-10-21 晚；R52 用；规范 54.34~54.36）
# 只在 FV >= F9ZB 时被调用；FV < F9ZB 的渲染一律不经过这里（旧版本题面必须逐字不变）。
REVIVE_RE_F9ZB = re.compile(r"使[^。]{0,12}获得复生")
RANGED_ONLY_F9ZB = ("自然法杖",)          # 同 hero_traits.RANGED_ONLY（冻结副本，渲染确定）
LIYUANFANG_F9ZB = "李元芳"
USE_NOW_F9ZB = "use now: no better timing left"
ENTRY_ALL_RE_F9ZB = re.compile(r"触发所有英雄的登场效果")


def f9zb_subs(n: str) -> tuple:
    """carry_pool.json substitutes（朵莉亚 ↔ 海诺）：有其中一个就算这个位置凑齐（同 rules.route_core_missing）。"""
    import sys
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    try:
        from agent.decision import lord_config as _LC
        return tuple(_LC.load().substitutes(n))
    except Exception:
        return ()


def f9zb_locked(B: "Board") -> bool:
    return bool(B.route) and not B.route_stale


COMMITTED_OFFLIST_F9ZB = "off-list carry, committed (keep; build his lineup)"


def f9zb_committed_offlist(B: "Board", n: str = "") -> bool:
    """54.37-1（"给了就认"，10-21）：现任主C 不在 carry_pool 开着的名单里、但已认定 → 不换给名单内主C、不写换绑 / 接班。
    条件 = rules._committed_carry_offlist（配置存在、名单没开他、他是现任主C 且在场、≥2 阶）
    + 54.37-1 的投入条件（≥10 级 或 挂着枫叶舞 或 ≥1 件装备）。只在 FV >= F9ZB 时生效。"""
    if FV < F9ZB:
        return False
    try:
        cc = B.ctx.get("carry") or ""
        n = n or cc
        if not n or n != cc or n not in B.field:
            return False
        c = _lc_f9za()
        if not c.present or c.enabled(n) or (tier(n) or 0) < 2:
            return False
        return ((B.lv(n) or 0) >= 10 or B.ctx.get("maple") == n
                or len((B.ctx.get("worn") or {}).get(n) or []) >= 1 or (B.eq(n) or 0) >= 1)
    except AttributeError:          # Board 还没初始化完
        return False


def f9zb_off_route_tag(n: str, B: "Board") -> str:
    """54.37-5：路线成员口径以 lineup#1 + ≥80% 占比为准 —— 不在 lineup#1 最终阵容、主C 阵容占比 <10% 的人不写 ROUTE member /
    ROUTE final team，改写 "off-route (N% of X lineups)"；引擎表 extra_wanted（路线计划指定）的照旧。不适用 → ""。"""
    if FV < F9ZB or not is_hero(n) or not B.cover or not B.anchor or not B.anch["n"] or n == B.anchor:
        return ""
    lu = B.cover[0]
    final, _ = lineup_members(lu)
    if n in final or n == (lu.get("carry") or "") or B.route_extra_wanted(n):
        return ""
    p = B.anch["mates"].get(n) or 0
    return f"off-route ({round(p * 100)}% of {B.anchor} lineups)" if p < 0.1 else ""


def f9zb_missing(names, B: "Board") -> list:
    return sorted(x for x in names if x and is_hero(x) and x not in B.owned and not any(y in B.owned for y in f9zb_subs(x)))


def f9zb_core_missing(B: "Board") -> list:
    """同 rules.route_core_missing：路线主C + 最终阵容里还没拥有的英雄（替代件算有）。"""
    if not f9zb_locked(B):
        return []
    core = set(_routes().final_team(B.route)) | ({B.route["carry"]} if B.route.get("carry") else set())
    return f9zb_missing(core - B.nw, B)


def f9zb_engine_unreachable(B: "Board") -> list:
    """同 rules.engine_missing_unreachable（54.36-4，第 883 局）：路线引擎核心件（early_priority，去掉主C / 复生件）还缺，
    且当前棋手等级一个都刷不出 → 缺件名单；有一个刷得出 / 不缺 → []。"""
    if not f9zb_locked(B) or not B.level:
        return []
    rc = B.route.get("carry") or ""
    eng = [x for x in (B.min_eng or {}).get("early_priority") or [] if x != rc and not REVIVE_RE_F9ZB.search(skill(x) or "")]
    miss = f9zb_missing(eng, B)
    if not miss or any(meta().can_appear(tier(x), B.level) for x in miss):
        return []
    return miss


def f9zb_afford(B: "Board") -> str:
    g, free = B.ctx.get("gold"), B.ctx.get("free_refresh") or 0
    if not isinstance(g, int):
        return ""
    if free:
        return f", {g}e: no hero after" if g < HERO_PRICE else ""
    after = g - _rcost(B)
    if after < 0:
        return " → CAN'T AFFORD"
    return f" → {after}e" + (", no hero after" if after < HERO_PRICE else "")


def f9zb_refresh(B: "Board", ps: list, cost: str, lvl_note: list) -> tuple[str, list[str]]:
    """f9zb 刷新选项（54.36-4，第 883 局）：缺的件（没拥有）和已有件的同名（凑觉醒）分开算命中率；
    路线缺的引擎核心件当前等级全刷不出 → 写 "level up first"（同 rules.engine_missing_unreachable）。"""
    aff = f9zb_afford(B)
    free = B.ctx.get("free_refresh") or 0
    cost = f"free ×{free}" if free else cost
    items = []
    un = f9zb_engine_unreachable(B)
    if un:
        # 54.37-3：这句只指路线引擎缺件（点名），主C 同名另起一条（见下）
        items.append(f"missing engine {', '.join(un[:2])} unreachable at lv {B.level}: level up first")
    items += lvl_note
    if not ps:
        items = ["free refreshes kept if unused" if x == "free refreshes are KEPT to next round if unused" else x for x in items]
        return f"refresh ({cost}{aff})", items + ["no specific wanted card"]

    def _any(xs):
        known = [p for _, p, _ in xs if p is not None]
        return round((1 - math.prod(1 - p for p in known)) * 100) if known else 0

    def _ln(n, p, note, dup):
        nt = re.sub(r" \(t(\d) not in shop at lv (\d)\)", r" (t\1 > lv\2)", note)
        nt = re.sub(r" \(t\d odds unmeasured at lv \d\)", " (odds ?)", nt)
        return f"{n} {'?' if p is None else round(p * 100)}%{nt}" + (" dup" if dup else "")

    items = ["free refreshes kept if unused" if x == "free refreshes are KEPT to next round if unused" else x for x in items]
    ps = [x for x in ps if x[0] in B.owned or not any(y in B.owned for y in f9zb_subs(x[0]))]   # 54.34：有替代件（海诺 ↔ 朵莉亚）不算缺
    miss = [x for x in ps if x[0] not in B.owned]
    dup = [x for x in ps if x[0] in B.owned]
    carries = {x for x in (B.cur_carry, B.fc, (B.route or {}).get("carry") if f9zb_locked(B) else "") if x}
    if B.level >= meta().MAX_LORD_LEVEL or (B.rnd or 0) >= 12:
        # 54.37-4：满级 / 第 12 回合后 dup 只算主C 同名 + 未觉醒超核心件（同 rules.super_core / is_awakened）；只要 1 张的太乙、已觉醒件不算
        sc = f9zb_super_core(B)
        dup = [x for x in dup if x[0] in carries or (x[0] in sc and not f9zb_awakened(x[0], B))]
    cdup = [x for x in dup if x[0] in carries] if un else []
    if not _any(cdup):
        cdup = []                       # 刷不到（0%）就不单写
    order = (lambda x: -(x[1] if x[1] is not None else -1))
    head = f"refresh ({cost}{aff}): " + (f"missing {_any(miss)}%" if miss else "no missing card")
    if dup:
        head += f", dup {_any(dup)}%"
    if cdup:
        items.insert(1, f"carry dup {_any(cdup)}%: buy if seen, don't pay to chase")      # 54.37-3：主C 同名不和引擎缺件混写
    items += [_ln(*x, False) for x in sorted(miss, key=order)] + [_ln(*x, True) for x in sorted(dup, key=order) if x not in cdup]
    return head, items


def f9zb_super_core(B: "Board") -> set:
    """同 rules.super_core：路线引擎表 early_priority（去掉路线主C / 现任主C）+ 明世隐（在路线 / 场上时）。"""
    rc = (B.route or {}).get("carry") or "" if f9zb_locked(B) else ""
    out = set((B.min_eng or {}).get("early_priority") or []) - {rc, B.cur_carry or "", B.ctx.get("carry") or ""}
    if MINGSHIYIN_F9ZA in (f9x_route_team(B) | set(B.field)):
        out.add(MINGSHIYIN_F9ZA)
    return out - {""}


def f9zb_awakened(n: str, B: "Board") -> bool:
    """同 rules.is_awakened：场上觉醒，或手里那张是觉醒卡。"""
    return (n in B.field and B.awk(n) == 0) or n in set(B.ctx.get("hand_awk") or [])


def f9zb_grant_text(N, note, k, names, ex, got, full, ch, B, d, hit) -> tuple[str, str]:
    """f9zb 发英雄牌的期望行（原写法 49~78 token）：主C 命中、计划数、期望（合成另算）写短。"""
    km = sum(1 for h in hit if h in B.owned)
    mg = f" ({_expect_useful(d['pick'], d['n'], km, N):.2f} merge)" if km else ""
    nm = " ".join(hit[:3]) + (" …" if k > 3 else "")
    a = f"pool {N}{note}, plan {k}" + (f" ({nm})" if k else "") + f", exp {ex:.2f}/{got}{mg}"
    b = f"pool {N}, plan {k}, exp {ex:.2f}/{got}{mg}"
    chs = re.sub(r"CARRY hit (\d+%) \((\S+)[^)]*\)", r"CARRY \2 \1", ch) if ch else ""
    for cand in ([f"{chs}; {a}", f"{chs}; {b}", chs] if chs else [a, b]):
        if est_tokens(cand) <= 30:
            return cand, b
    return (chs or b), b


def f9zb_exclusive_text(n: str) -> str:
    """f9zb：阿离专属天赋卡面原文写短（原 56 token）：去标点空白，截到选项放得下（全文在局面行）。"""
    t = re.sub(r"\s+", "", skill(n))
    base = f"= Ali lv6 talent {n}: "
    while len(t) > 12 and est_tokens(base + t) > 36:
        t = t[:-1]
    return base + t + ("" if t == re.sub(r"\s+", "", skill(n)) else "…")


def f9zb_offroute(h: str, B: "Board") -> bool:
    """路线外、没用的人：路线锁定、不在路线件 / 路线成员里、不是主C、不是引擎件。"""
    if not f9zb_locked(B) or not is_hero(h) or h in (B.cur_carry, B.fc, B.ctx.get("carry")):
        return False
    if h in f9z_route_pieces(B) or B.route_member(h) or B.route_core_engine(h):
        return False
    return not (B.engine and B.engine_role(h) in ("producer", "amplifier"))


def f9zb_late_fill(n: str, B: "Board", on_board: bool, miss: list):
    """f9zb：LATE-FILL 行写短（原 42~48 token）；第 12 回合前、场上有路线外的人 → 写"卖外人上它"（54.36-2，第 881 局）。"""
    if B.rnd and B.rnd < 12:
        if on_board:
            return "LATE-FILL (revive): before r12 its slot yields to growth pieces"
        out = [h for h in B.field if f9zb_offroute(h, B)]
        if out:
            return f"LATE-FILL (revive): before r12 only replaces off-route {out[0]}"
        return "LATE-FILL (revive): low priority before r12; no free slot → don't field"
    if miss:
        if on_board:
            return f"LATE-FILL on board early: slot goes to missing {', '.join(miss[:2])}"
        return f"LATE-FILL: add LAST ({len(miss)} missing: {', '.join(miss[:2])})"
    return None


def f9zb_entry_all_now(n: str, B: "Board") -> bool:
    """54.36-6（第 886 局；同 rules tag_on_field 放宽）：集体唤醒 —— 路线锁定、阵容定型（核心不缺 / 第 12 回合起）、场上有明世隐 → 照用。"""
    if not ENTRY_ALL_RE_F9ZB.search(skill(n) or "") or MINGSHIYIN_F9ZA not in B.field or not f9zb_locked(B):
        return False
    return (B.rnd or 0) >= 12 or not f9zb_core_missing(B)


def f9zb_same_route_front(n: str, x: str, B: "Board") -> bool:
    """54.36-5（第 884 局钟馗 ↔ 蒙恬）：双方都是路线成员、都是前排、都不是引擎件 / 主C / 复生件。"""
    if not (f9zb_locked(B) and x and n != x and is_hero(n) and is_hero(x)):
        return False
    rc = B.route.get("carry") or ""
    pieces = f9z_route_pieces(B)
    for h in (n, x):
        if h in (rc, B.cur_carry, B.fc, B.ctx.get("carry")) or not (h in pieces or B.route_member(h)):
            return False
        if B.route_core_engine(h) or (B.engine and B.engine_role(h) in ("producer", "amplifier")):
            return False
        if REVIVE_RE_F9ZB.search(skill(h) or "") or h == MINGSHIYIN_F9ZA or not is_front_tank(h):
            return False
    awk = set(B.ctx.get("hand_awk") or [])
    return x not in awk and B.hand.count(x) < 2


def f9zb_swap_head(x: str, B: "Board") -> str:
    if x == MINGSHIYIN_F9ZA and x not in B.field and (B.rnd or 0) >= 12:
        if not B.temp_src_hero:
            return "; new 明世隐 ≈ 0 (no temp-lv source)"          # 54.37-7：场上没有临时等级来源（同 "hero temp src: none"）
        return f"; new 明世隐 ≈ +2 × {f9za_rounds_left(B)} rounds left (only +2/rd)"
    return ""


def f9zb_swap_lines(n: str, B: "Board") -> list:
    x = B.ctx.get("_swap_in") or ""
    out = []
    if n == MINGSHIYIN_F9ZA and B.awk(n) == 0:
        out.append("明世隐 awakened: entry stacks lost if sold")
    if x and REVIVE_RE_F9ZB.search(skill(x) or ""):
        if (n == LIYUANFANG_F9ZB and (B.rnd or 0) >= 12 and B.awk(n) != 0
                and MINGSHIYIN_F9ZA in B.field and B.awk(MINGSHIYIN_F9ZA) == 0):
            out.append(f"李元芳 +2 lv/round (1 coin) < {x} revive")
        elif (B.rnd or 0) < 12 and f9zb_offroute(n, B):
            out.append(f"off-route: OK to sell for {x}")
    if f9zb_same_route_front(n, x, B):
        out.append("same-route swap, no clear gain")
    return out


_usable_now_fact_f9za = usable_now_fact


def usable_now_fact(n: str, B: "Board") -> str:          # noqa: F811  f9zb：集体唤醒阵容定型后不写 UNUSABLE（FV < F9ZB 原样返回）
    s = _usable_now_fact_f9za(n, B)
    if s and FV >= F9ZB and f9zb_entry_all_now(n, B):
        return ""
    return s


_use_on_facts_f9za = use_on_facts


def use_on_facts(n: str, t: str, B: "Board") -> list[str]:          # noqa: F811  f9zb：集体唤醒定型后写 use now（FV < F9ZB 原样返回）
    out = _use_on_facts_f9za(n, t, B)
    if FV >= F9ZB and f9zb_entry_all_now(n, B):
        out = [USE_NOW_F9ZB] + [x for x in out if x != USE_NOW_F9ZB]
    return out



# f9zb 选项打包（"f9zb 选项预算"；2026-10-21 第 881 局）：关键结论（收益 / UNUSABLE / AVOID…）排最前；
# 单条就超预算的行先缩写（ZB_ABBR）→ 从后往前去括号 → 最后才截断，不再整条丢掉。正常的模板都已在源头写短，
# 这里只兜没见过的长行（tools/laya/check_dropped_lines.py --no-fallback 能看到还有哪些靠兜底）。
ZB_FALLBACK = True
KEY_RE_F9ZB = re.compile(r"^(AVOID|DEAD|UNUSABLE|LOW VALUE|NO use now|use now: no better)|UNUSABLE|≈ ?\+\d+ lv|no value"
                         r"|unreachable at lv|MISMATCH|does NOT fit|entry stacks lost|no clear gain|< \S+ revive")
ZB_ABBR = ((" rounds", " rd"), (" round", " rd"), ("/round", "/rd"), (" level-up", " lvl-up"), (" levels", " lv"),
           (" level", " lv"), ("heroes", "hrs"), (" on board", ""), (" (no slot needed)", ""), (", no slot needed", ""),
           ("awakened", "awk"), ("candidates", "cands"))


def zb_short(head: str, it: str, budget: int = OPT_TOKENS) -> str:
    sep = "" if head.endswith((" ", ":")) else " — "

    def ok(t):
        return est_tokens(head + sep + t) <= budget
    if not ZB_FALLBACK or ok(it):
        return it
    t = it
    for a, b in ZB_ABBR:
        t = t.replace(a, b)
    if ok(t):
        return t
    while True:
        ms = list(re.finditer(r"\s*[（(][^()（）]*[)）]", t))
        if not ms:
            break
        t = t[:ms[-1].start()] + t[ms[-1].end():]
        if ok(t):
            return t
    while len(t) > 8 and not ok(t + "…"):
        t = t[:-1].rstrip()
    return t + "…"


def pack_f9zb(head: str, items: list[str], budget: int = OPT_TOKENS) -> str:
    items = list(dict.fromkeys(it for it in items if it))
    key = [it for it in items if KEY_RE_F9ZB.search(it)]
    out = head
    for it in key + [it for it in items if it not in key]:
        it = zb_short(head, it, budget)
        sep = "; " if out != head else ("" if head.endswith((" ", ":")) else " — ")
        cand = out + sep + it
        if est_tokens(cand) > budget:
            continue
        out = cand
    return out
