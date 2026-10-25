"""等级引擎：产出方 + 放大器 + 落点。

2026-09-26 发现：阵容到最后没形成套路、没参考推荐阵容；没利用开团、整备、核心等级这些 Tag 效果的叠加，
只靠用牌提升等级，等级比其他玩家少 100 级 —— 十连打第 8 局对面 200 级、我方 20 级：海诺要"核心涨等级"却一个核心产出方都没有。

只做三件事（规则层，不碰 Laya 题面）：
  1. 给推荐阵容打引擎标签（`lineup_engine`）→ 选阵容时按"引擎完成度"加分并锁定；
  2. 判断某个英雄是不是当前目标引擎的零件（`is_part`）→ 卖人 / 闲置手牌 / 过渡件判定都不动它；
  3. 商店里出现目标引擎还缺的零件 → 规则先买（`missing_parts`）。
排除操作难度过高的套路（伞海月：凑遗志、卖光全场搬等级），见 engines.json 的 excluded。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data" / "engines.json"

W_PRODUCER, W_AMPLIFIER, W_PAYOFF = 1.0, 1.5, 1.0


@lru_cache(maxsize=1)
def book() -> dict:
    try:
        return json.loads(_DATA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"engines": {}, "opener": {"rounds": 0, "heroes": []}}


CORRECTED = True          # 用 corrections_20261011 修正后的目录（Laya 旧题面 < f9lj 渲染时由 facts 临时关掉）
F9ZB = True               # corrections_20261021 + 按主C 选引擎（Laya 旧题面 < f9zb 渲染时由 facts 临时关掉）


@lru_cache(maxsize=1)
def _corrected() -> dict:
    import copy
    eng = copy.deepcopy(book().get("engines") or {})
    for key, roles in ((book().get("corrections_20261011") or {}).get("remove") or {}).items():
        for r, names in roles.items():
            if key in eng and isinstance(eng[key].get(r), list):
                eng[key][r] = [n for n in eng[key][r] if n not in names]
    return eng


@lru_cache(maxsize=1)
def _corrected_zb() -> dict:
    import copy
    eng = copy.deepcopy(_corrected())
    for key, roles in ((book().get("corrections_20261021") or {}).get("add") or {}).items():
        for r, names in roles.items():
            if key in eng:
                eng[key][r] = list(eng[key].get(r) or []) + [n for n in names if n not in (eng[key].get(r) or [])]
    return eng


def engines() -> dict:
    if CORRECTED and F9ZB:
        return _corrected_zb()
    return _corrected() if CORRECTED else (book().get("engines") or {})


def parts(key: str) -> set[str]:
    e = engines().get(key) or {}
    return set(e.get("producers") or []) | set(e.get("amplifiers") or []) | set(e.get("payoff") or [])


def role(key: str, name: str) -> str:
    """'producer' / 'amplifier' / 'payoff' / ''（一个英雄可能同时是几种，按这个优先级报）。"""
    e = engines().get(key) or {}
    for r, k in (("amplifier", "amplifiers"), ("producer", "producers"), ("payoff", "payoff")):
        if name in (e.get(k) or []):
            return r
    return ""


def status(key: str, owned, distinct: bool = True) -> dict:
    """某套引擎在 owned（场上 + 手里的名字集合）里的完成度。
    distinct（2026-10-11 第 564 局）：同一个英雄身兼产出方 / 放大器 / 落点时只算一个人 —— 原来一个云中君就让"大河·开团"显示 COMPLETE。
    旧题面版本（facts FV < f9li）传 distinct=False 保持原文。"""
    e = engines().get(key) or {}
    owned = set(owned)
    prod = [n for n in e.get("producers") or [] if n in owned]
    amp = [n for n in e.get("amplifiers") or [] if n in owned]
    pay = [n for n in e.get("payoff") or [] if n in owned]
    score = len(prod) * W_PRODUCER + len(amp) * W_AMPLIFIER + len(pay) * W_PAYOFF
    if e.get("difficulty") == "weak":
        score *= 0.6
    if distinct:
        heads = set(prod) | set(amp) | set(pay)
        score = sum(max(W_PRODUCER if n in prod else 0, W_AMPLIFIER if n in amp else 0, W_PAYOFF if n in pay else 0)
                    for n in heads) * (0.6 if e.get("difficulty") == "weak" else 1.0)
        return {"key": key, "name": e.get("name", key), "producers": prod, "amplifiers": amp, "payoff": pay,
                "score": score, "complete": bool(prod and amp and pay) and len(heads) >= 2,
                "started": len(set(prod) | set(amp)) >= 2}
    return {"key": key, "name": e.get("name", key), "producers": prod, "amplifiers": amp, "payoff": pay,
            "score": score, "complete": bool(prod and amp and pay), "started": len(prod) + len(amp) >= 2}


def best(owned, distinct: bool = True) -> dict | None:
    """owned 里最成型的引擎（score 最高；没有任何零件返回 None）。"""
    cands = [status(k, owned, distinct) for k in engines()]
    cands = [c for c in cands if c["score"] > 0]
    return max(cands, key=lambda c: (c["score"], c["complete"], c["started"])) if cands else None


def lineup_engine(members: tuple, carry: str = "") -> tuple[str, float]:
    """carry（F9ZB 起）：主C 是某套引擎的落点 → 优先那套（2026-10-21 第 884 局：玄策阵容里牺牲件多，
    按零件数被判成河洛·牺牲，实际玄策吃的是古币 + 明世隐转临时等级）。"""
    if carry and F9ZB:
        k = _carry_engine(tuple(members), carry, CORRECTED)
        if k[0]:
            return k
    return _lineup_engine(members, CORRECTED, F9ZB)


@lru_cache(maxsize=1024)
def _carry_engine(members: tuple, carry: str, _corr: bool) -> tuple[str, float]:
    best_ = ("", 0.0)
    for k, e in engines().items():
        if carry in (e.get("payoff") or []):
            s = status(k, tuple(members) + (carry,))["score"]
            if s > best_[1]:
                best_ = (k, s)
    return best_


@lru_cache(maxsize=4096)
def _lineup_engine(members: tuple, _corr: bool, _zb: bool = False) -> tuple[str, float]:
    """一套阵容属于哪套引擎：成员命中零件最多的那套（返回 (key, 命中分)）；一个都不沾返回 ('', 0)。"""
    bests = ("", 0.0)
    for k in engines():
        s = status(k, members)["score"]
        if s > bests[1]:
            bests = (k, s)
    return bests


def missing_parts(key: str, owned) -> list[str]:
    """目标引擎还缺的零件，按重要性排：放大器 > 产出方 > 落点（落点由主C 逻辑另管，排最后）。"""
    e = engines().get(key) or {}
    owned = set(owned)
    out = [n for n in e.get("amplifiers") or [] if n not in owned]
    out += [n for n in e.get("producers") or [] if n not in owned and n not in out]
    out += [n for n in e.get("payoff") or [] if n not in owned and n not in out]
    return out


def opener_heroes(round_no: int) -> set[str]:
    op = book().get("opener") or {}
    return set(op.get("heroes") or []) if 1 <= (round_no or 0) <= (op.get("rounds") or 0) else set()   # 0 = 回合未知


def describe(owned) -> str:
    """日志用的一行摘要。"""
    b = best(owned)
    if not b:
        return "引擎：无"
    return (f"引擎 {b['name']}：产出方 {b['producers'] or '-'} 放大器 {b['amplifiers'] or '-'} 落点 {b['payoff'] or '-'}"
            + ("（成套）" if b["complete"] else ""))
