"""天赋硬条件表（data/lords/<棋手>/talent_conditions.json，旧路径 data/talent_conditions.json 兜底；2026-10-20 逐张复审）的求值器。

2026-10-20 起：天赋数量不多，逐张分析硬条件；"战术牌天赋没有逐鹿阵容"这类问题复盘时要能查出来。
复盘（tools/review_focus.py）和线上临时规则 talent_low_value 共用这一份判断，不再各写一份名单。

check(name, team, carry) -> (等级, 不满足的原因 or "")
  team：我方英雄名集合（场上 + 手里 + 已锁路线的最终阵容）；carry：主C（没有就传 ""）。
  note 类条件只是说明，不参与判定。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_P = Path(__file__).resolve().parents[2]
_T = None
_C = None
EXPECTED_ROUTE_CARRY = ""
BANNED_FACTION = ""
SETTLED_CARRY = ""           # 线上天赋题：已定死的主C（2 阶以上、≥10 级或挂枫叶舞；laya_policy 每题设）          # 线上天赋题：本局禁的阵营（laya_policy 每题设）    # 线上天赋题：路线没锁时，开局配置预计的路线主C（laya_policy 每题设）
LOWEST_OFFROUTE = None       # 线上天赋题：场上等级最低的 N 人是否全是路线外占位（None = 判不了；laya_policy 每题设，10-22 第 929 局一个响指）
_IN_FROZEN = False   # 题面渲染（冻结表）求值中：只给线上过滤用的新判定在这里关掉，旧版本题面逐字不变
_LIVE = True      # 冻结表（题面旧版本）求值时置 False：线上新增的判定（韩信攻速型等）不影响旧题面渲染


_T_LORD = ""      # _T 是哪个棋手的表


def table_path(lord: str = "") -> Path:
    """当前棋手的天赋条件表：data/lords/<棋手>/talent_conditions.json（2026-10-22 起，阿离单主C 的价值判断绑定棋手）；
    棋手目录没有 → 旧路径 data/talent_conditions.json 兜底。"""
    from agent.decision import lord_config as _LCt
    f = _LCt.lord_dir(lord) / "talent_conditions.json"
    return f if f.exists() else _P / "data" / "talent_conditions.json"


def table() -> dict:
    global _T, _T_LORD
    from agent.decision import lord_config as _LCt
    lord = _LCt.current()
    if _T is None or (_T_LORD and _T_LORD != lord):
        f = table_path(lord)
        _T = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        _T_LORD = lord
    return _T


def _cards() -> dict:
    global _C
    if _C is None:
        _C = json.loads((_P / "data" / "cards.json").read_text(encoding="utf-8")).get("cards", {})
    return _C


def _fac(h: str) -> str:
    return (_cards().get(h) or {}).get("faction") or ""


def _tier(h: str) -> int:
    return int((_cards().get(h) or {}).get("quality") or 0)


def _skill(h: str) -> str:
    v = _cards().get(h) or {}
    return str(v.get("skill") or v.get("desc") or "")


NO_FACTION = ("", "无阵营")


def _final_check(c: dict, carry: str, board: dict) -> tuple:
    """按棋盘实际摆法求值（2026-10-21 第 853 局同心协力：人口 7 − 明世隐 − 太乙 = 河洛 5）。
    board = {final: 最终可摆阵容（路线最终阵容 ∪ 必带件 ∪ 主C）, must: 必带件, pool: final ∪ 场上 ∪ 手里, cap: 人口上限}。
    返回 (满足?, 原因)；不是这几类 / 没 board → (None, "")。"""
    t = c.get("type")
    if t not in FINAL_TYPES:
        return None, ""
    if t == "carry_tier_in":
        ok = _tier(carry) in set(c.get("tiers") or [])
        return ok, ("" if ok else f"主C {carry or '未定'} 是 {_tier(carry)} 阶，要 {'/'.join(map(str, c.get('tiers') or []))} 阶")
    if t == "carry_has_keyword":
        kw = c.get("kw") or ""
        ok = bool(carry) and kw in _skill(carry)          # 10-21：读技能原文，不用卡库 keywords（会误标）
        return ok, ("" if ok else f"主C {carry or '未定'} 技能原文没有「{kw}」")
    if not board:
        return None, ""
    final, must, pool = set(board.get("final") or ()), set(board.get("must") or ()), set(board.get("pool") or ())
    cap = int(board.get("cap") or 7)
    hero = lambda h: (_cards().get(h) or {}).get("category") == "英雄牌"          # noqa: E731
    final, must, pool = {h for h in final if hero(h)}, {h for h in must if hero(h)}, {h for h in pool if hero(h)}
    if t == "faction_count_final":
        facs = [c["faction"]] if c.get("faction") else sorted({_fac(h) for h in pool if _fac(h) not in NO_FACTION})
        best, bf = 0, ""
        for F in facs:
            n = min(cap - len([h for h in must if _fac(h) != F]), len([h for h in pool if _fac(h) == F]))
            if n > best:
                best, bf = n, F
        ok = best >= int(c.get("min") or 1)
        return ok, ("" if ok else f"最终可摆同阵营最多 {best}（{bf or '-'}；人口 {cap} − 必带 {'/'.join(sorted(h for h in must if _fac(h) != bf)) or '无'}），要 ≥{c.get('min')}")
    if t == "faction_kinds_final":
        k = {_fac(h) for h in final if _fac(h) not in NO_FACTION}
        ok = len(k) >= int(c.get("min") or 1)
        return ok, ("" if ok else f"最终阵容只有 {len(k)} 个阵营（{'/'.join(sorted(k)) or '-'}），要 ≥{c.get('min')}")
    if t == "tier_count_final":
        tiers = set(c.get("tiers") or [])
        n = len([h for h in final if _tier(h) in tiers])
        ok = n >= int(c.get("min") or 1)
        return ok, ("" if ok else f"最终阵容 {'/'.join(map(str, sorted(tiers)))} 阶 {n} 名，要 ≥{c.get('min')}")
    if t == "carry_is_top_tier_final":
        top = max([_tier(h) for h in final] or [0])
        tied = [h for h in final if _tier(h) == top]
        # 10-21 确认：并列最高品阶时随机给一个 → 并列 ≤2 人（一半概率）算满足
        ok = bool(carry) and _tier(carry) == top and len(tied) <= 2
        return ok, ("" if ok else f"最终阵容最高品阶 {top} 阶是 {'/'.join(sorted(tied))}，主C {carry or '未定'} {_tier(carry)} 阶")
    if t == "keyword_count_final":
        kw = c.get("kw") or ""
        n = len([h for h in final if kw in _skill(h)])
        ok = n >= int(c.get("min") or 1)
        return ok, ("" if ok else f"最终阵容技能原文带「{kw}」的 {n} 名，要 ≥{c.get('min')}")
    return None, ""


FINAL_TYPES = ("carry_tier_in", "carry_has_keyword", "faction_count_final", "faction_kinds_final", "tier_count_final",
               "carry_is_top_tier_final", "keyword_count_final")


def check(name: str, team: set, carry: str, route_carry: str = "", unknown_carry_fails: bool = False,
          board: dict | None = None) -> tuple[str, str]:
    """board（线上传，题面旧版本不传）：见 _final_check。表字段 grade_if = [{when: [条件…], grade}]：硬条件都过了、when 全满足 → 用该等级（可上调）。"""
    e = table().get(name)
    if not e and not _IN_FROZEN:
        # 10-22 天赋审查：表里 14 个装备天赋登记成"X(天赋牌)"，线上读到的是"X" → 一直查不到、按"?"放行（943 局 R6 选了法潮灵杖）；
        # OCR 偶尔带前缀（"#套娃枪""A一莲托生"）→ 去掉非汉字前缀再查
        _n2 = re.sub(r"^[^一-鿿]+", "", name or "")
        e = table().get(name + "(天赋牌)") or table().get(_n2) or table().get(_n2 + "(天赋牌)")
    if not e:
        return "?", ""
    g, why = _run(e, team, carry, route_carry, unknown_carry_fails, board)
    if g == "C" and why:
        return g, why
    if _LIVE:
        for gi in e.get("grade_if") or []:
            whens = gi.get("when") or []
            # 上调要**明确满足**：按最终阵容计数的条件判不了（路线没锁、board=None）不算满足（f9za 抽样：路线没锁时随机成长被上调）
            if whens and all((_final_check(c, carry, board)[0] is True) if c.get("type") in FINAL_TYPES
                             else _run({"grade": "A", "conditions": [c]}, team, carry, route_carry, unknown_carry_fails, board)[0] == "A"
                             for c in whens):
                return gi.get("grade") or g, ""
    return g, why


def _run(e: dict, team: set, carry: str, route_carry: str, unknown_carry_fails: bool, board) -> tuple[str, str]:
    g = e.get("grade") or "?"
    heroes = {h for h in team if (_cards().get(h) or {}).get("category") == "英雄牌"}
    for c in e.get("conditions") or []:
        ok, why = _final_check(c, carry, board) if _LIVE else (None, "")
        if ok is False:
            return "C", why
        if ok is not None:
            continue
        t = c.get("type")
        if t == "never":
            return "C", "对阿离单核永远没用"
        if t == "faction_on_team":
            n = sum(1 for h in heroes if _fac(h) == c.get("faction"))
            if n < int(c.get("min") or 1):
                return "C", f"要【{c.get('faction')}】英雄 ≥{c.get('min') or 1}，阵容里 {n} 个"
        elif t == "carry_not" and _LIVE:
            # 2026-10-22 第 904 局（神鹰锻匠，主C 云中君）：这个天赋给云中君之外的英雄加装备，云中君自己吃不到；
            # 阿离阵容除了主C 都很弱，装备提升不了战力 → 主C 是这几个人时 C
            _who = {x for x in (carry, route_carry) if x}
            if _who & set(c.get("heroes") or []):
                return "C", f"主C {'/'.join(sorted(_who))} 自己吃不到（效果给别人）"
        elif t == "route_faction" and _LIVE and not route_carry and EXPECTED_ROUTE_CARRY:
            # 路线还没锁：按开局配置预计要走的路线主C 判（河洛没禁 → 河洛组；禁了 → 亚连…），不看眼前的临时主C
            if _fac(EXPECTED_ROUTE_CARRY) != c.get("faction"):
                return "C", f"预计路线主C {EXPECTED_ROUTE_CARRY} 不是【{c.get('faction')}】"
        elif t == "route_faction" and _LIVE:
            # 2026-10-21 第 893 局（望梅止渴 / 灵祭升华 vs 三人游）：另两个是定死路线的天赋，而那条路线不在备选里：
            # 绑阵营的天赋只在**锁定路线的主C**是这个阵营时才有用；路线没锁 / 不是这个阵营 → C（不看眼前的临时主C）
            if not route_carry or _fac(route_carry) != c.get("faction"):
                return "C", f"锁定路线的主C 要是【{c.get('faction')}】（现在：{route_carry or '路线未锁'}）"
        elif t == "route_carry_in_or_unlocked" and _LIVE:
            # 定路线天赋（幸运硬币 → 狄仁杰）：路线没锁 → 过（选了就锁）；锁了别人 → C；该主C 阵营被禁（BANNED_FACTION）→ C
            _set = route_carry or SETTLED_CARRY
            if _set and _set not in set(c.get("heroes") or []) and _fac(_set) != c.get("ok_faction"):
                # 10-22：主C 已定死就不选这个天赋；其他河洛主C 凑合能用，但不如狄仁杰
                return "C", f"主C 已定 {_set}，不是 {'/'.join(c.get('heroes') or [])}（也不是【{c.get('ok_faction')}】）"
            if BANNED_FACTION and any(_fac(h) == BANNED_FACTION for h in c.get("heroes") or []):
                return "C", f"{'/'.join(c.get('heroes') or [])} 的阵营【{BANNED_FACTION}】被禁"
        elif t in ("carry_in", "route_carry_in") and _LIVE:
            # carry_in：现任 / 路线主C 是名单里的人；route_carry_in：路线主C（没锁按开局预计）是名单里的人
            _who = (route_carry or EXPECTED_ROUTE_CARRY) if t == "route_carry_in" else (route_carry or carry or ("" if _IN_FROZEN else EXPECTED_ROUTE_CARRY))
            # 10-22 第 934 局 R4：现任是 1 阶临时主C（传空）、路线没锁 → carry_in 也按开局预计的路线主C 判（中坚力量在亚连线被选）
            if _who and _who not in set(c.get("heroes") or []):
                return "C", f"主C 要是 {'/'.join(c.get('heroes') or [])}（现在 {_who}）"
        elif t == "carry_faction" and carry and _LIVE and any(x.get("type") == "route_faction" for x in (e.get("conditions") or [])):
            pass        # 10-22 天赋审查：有 route_faction 的条目不再拿眼前的临时 / 非路线主C 判阵营（903 R5 古币翻新被亚连判成 C）
        elif t == "carry_faction" and carry:
            if _fac(carry) != c.get("faction"):
                return "C", f"主C 要是【{c.get('faction')}】，{carry} 是【{_fac(carry) or '?'}】"
        elif t in ("attack_speed_carry", "skill_carry") and carry:
            from agent.decision import hero_traits as _ht
            fast = _ht.attack_speed_carry_live(carry) if _LIVE else _ht.attack_speed_carry(carry)
            if t == "attack_speed_carry" and not fast:
                return "C", f"主C {carry} 不是攻速型"
            if t == "skill_carry" and fast:
                return "C", f"主C {carry} 是攻速型，不是技能型"
        elif t == "heroes_in_team":
            hs = c.get("heroes") or []
            inn = [h for h in hs if h in heroes]
            if (c.get("mode") == "all" and len(inn) < len(hs)) or (c.get("mode") != "all" and not inn):
                return "C", f"要 {'全部' if c.get('mode') == 'all' else '任一'}：{'/'.join(hs)}，阵容里只有 {'/'.join(inn) or '0 个'}"
        elif t == "carry_in" and (carry or unknown_carry_fails):
            # unknown_carry_fails（线上过滤传 True；题面 f9y/f9z 不传，渲染不变）：主C 未定时也按不满足（10-20 中坚力量）
            if carry not in (c.get("heroes") or []):
                return "C", f"主C 要是 {'/'.join(c.get('heroes') or [])} 之一，现在是 {carry or '未定'}"
        elif t == "route_carry_is":
            # 只看已锁路线的主C（没锁路线 route_carry 为空 → 不判；临时主C 不算）
            if route_carry and route_carry not in (c.get("heroes") or []):
                return "C", f"天赋绑定主C {'/'.join(c.get('heroes') or [])}，现在路线主C 是 {route_carry}"
        elif t == "lowest_offroute" and _LIVE and LOWEST_OFFROUTE is not None:
            # 10-22 第 929 局 R7：一个响指（听天由命"消除场上 4 名等级最低的英雄"）在最低 4 人里有明世隐 / 铠时选了，整局没用出去
            ok_l, who_l = LOWEST_OFFROUTE
            if not ok_l:
                return "C", f"场上等级最低的 {c.get('n') or 4} 人里有路线件 {'/'.join(who_l)}，用了会删掉他们"
        elif t == "high_tier_count":
            tiers = set(c.get("tiers") or [4, 5])
            n = sum(1 for h in heroes if _tier(h) in tiers)
            if n < int(c.get("min") or 1):
                return "C", f"要 {'/'.join(map(str, sorted(tiers)))} 阶 ≥{c.get('min')} 名，阵容里 {n} 名"
    return g, ""


class _Frozen:
    """题面旧版本用的冻结表（data/talent_conditions_<ver>.json）：表后来改了，旧版本题面渲染要逐字不变（题面版本化约定）。"""

    def __init__(self, ver: str, live: bool = False):
        f = _P / "data" / f"talent_conditions_{ver}.json"
        self._t = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        self._live = live      # f9za 起：冻结的是表，求值仍按线上口径（最终可摆阵容 / grade_if / 韩信攻速型）

    def table(self) -> dict:
        return self._t if self._t is not None else table()

    def __getattr__(self, k):          # _fac / _tier 等辅助函数照用模块里的
        return globals()[k]

    def check(self, *a, **k):
        global _T, _LIVE, _IN_FROZEN
        if self._t is None:
            return check(*a, **k)
        keep, _T = _T, self._t
        live0, _LIVE = _LIVE, self._live
        fz0, _IN_FROZEN = _IN_FROZEN, True
        try:
            return check(*a, **k)
        finally:
            _T, _LIVE, _IN_FROZEN = keep, live0, fz0


_FROZEN: dict = {}


def frozen(ver: str, live: bool = False) -> "_Frozen":
    """live=False（f9y / f9z）：冻结表 + 旧求值口径；live=True（f9za 起）：冻结表 + 线上求值口径。"""
    k = (ver, bool(live))
    if k not in _FROZEN:
        _FROZEN[k] = _Frozen(ver, bool(live))
    return _FROZEN[k]
