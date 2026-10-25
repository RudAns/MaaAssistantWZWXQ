"""官方推荐阵容的"分回合运营路线"（2026-09-28 起按官方推荐阵容的分回合运营路线打）。

规则层（rules.py 锁定路线 / 候选过滤）和 Laya 题面（tools/laya/facts.py 的路线行、route 题）**共用这里**，
保证两边的"哪些路线不参选""候选怎么排"逐字一致。只读 data/ 下的原始数据，不依赖规则引擎的状态。

路线用阵容记录的 `key` 做唯一标识（lineups.json 里有 39 个重名阵容）。
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LORD = "阿离"
# 开局过渡件：前 3 回合谁都会拿，不能拿来说明"在走哪条路线"
OPENERS = frozenset({"孙膑", "干将莫邪", "曜", "牛魔", "孙尚香", "苏烈", "弈星"})
EXCLUDE_CARDS = ("遗志",)          # 伞海月一族：要凑遗志、卖光全场搬等级（操作难度太高，排除）
# **只走 2~3 阶主C 的路线**（2026-09-29：2-3 阶主C 最强，前期能抽到、能尽早堆等级；
# 4-5 阶也强，但要遗愿之类的特殊效果牌、要清场来换，很难 —— 以 2-3 阶主C 定调）。
# 默认 None = 不限（离线重渲染 / 数据集与 R15 训练时一致）；实战由 live_match_agent --carry-tiers 设成 (2, 3)。
CARRY_TIERS: tuple | None = None
# 实战不走的主C（默认空 = 离线工具不变）。公孙离：要"夺取"合成站位才转得起来，我们操作不了（2026-09-30）
EXCLUDE_CARRIES: frozenset = frozenset()


def _load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def pieces() -> dict:
    """{卡名: {"quality": 品阶, "synergies": [派系]}}（与 rules.pieces 同源：data/cards.json）。"""
    return {n: {"quality": c.get("quality", 0), "synergies": [c["faction"]] if c.get("faction") else []}
            for n, c in (_load("data/cards.json").get("cards") or {}).items()}


@lru_cache(maxsize=1)
def reviewed() -> dict:
    return _load("data/hero_carry_reviewed.json")


@lru_cache(maxsize=1)
def ali_lineups() -> tuple:
    """阿离能用的全部阵容（按使用人数降序，顺序确定）。"""
    out = [lu for lu in _load("data/lineups.json").get("lineups", [])
           if LORD in (lu.get("lord_names") or []) and any(h.get("name") for h in lu.get("heroes") or [])]
    out.sort(key=lambda lu: (-(lu.get("use_num") or 0), str(lu.get("key"))))
    return tuple(out)


@lru_cache(maxsize=1)
def custom_lineups(include_disabled: bool = True) -> tuple:
    """用户指定的路线（data/custom_lineups.json）：只供 --force-route 按 key 锁定，**不进 candidates**（route 题与 R15 训练时一致）。
    include_disabled=False：去掉标了 disabled 的（手动停用的，如 2026-10-17 晚的 user_ganjiang；题面 f9q 起这样取）。"""
    p = ROOT / "data" / "custom_lineups.json"
    lus = tuple(json.loads(p.read_text(encoding="utf-8")).get("lineups", [])) if p.exists() else ()
    return lus if include_disabled else tuple(lu for lu in lus if not lu.get("disabled"))


def by_key(key) -> dict | None:
    if not key:
        return None
    return next((lu for lu in ali_lineups() + custom_lineups() if str(lu.get("key")) == str(key)), None)


def final_team(lu: dict) -> list:
    return list(dict.fromkeys(h["name"] for h in lu.get("heroes") or [] if h.get("name")))


def stage_units(lu: dict) -> list:
    """运营阶段里出现、但不在最终阵容里的英雄（有序）。"""
    fin = set(final_team(lu))
    return list(dict.fromkeys(h for st in lu.get("stages") or [] for h in (st.get("main") or []) + (st.get("sub") or [])
                              if h not in fin))


def members(lu: dict) -> set:
    return set(final_team(lu)) | set(stage_units(lu)) | ({lu["carry"]} if lu.get("carry") else set())


def stage_now(lu: dict, rnd: int) -> dict | None:
    return next((st for st in lu.get("stages") or [] if (st.get("from_round") or 0) <= rnd <= (st.get("to_round") or 99)), None)


def excluded(lu: dict) -> bool:
    """不参选的路线：① 伞海月一族（效果牌要遗志 / 名字带遗志 / 主C 是海月）；② 主C 是过渡型（评分 ≤4：曜/赵云/姬小满）；
    ③ 主C 1 阶（干将无敌流：第 15 局锁死在 1 阶干将上）。"""
    if lu.get("custom"):
        return False          # 用户指定的路线不受排除名单限制（如 1 阶主C 干将）
    carry = lu.get("carry") or ""
    if CARRY_TIERS and (pieces().get(carry) or {}).get("quality", 0) not in CARRY_TIERS:
        return True
    if carry in EXCLUDE_CARRIES:
        return True
    return (any(c in (lu.get("effect_cards") or []) for c in EXCLUDE_CARDS)
            or "遗志" in (lu.get("name") or "") or carry == "海月"
            or (reviewed().get(carry) or {}).get("carry_score", 9) <= 4
            or (pieces().get(carry) or {}).get("quality", 0) == 1)


FACTIONS = ("河洛", "逐鹿", "日落海", "三分之地", "大河流域")


def norm_banned(b: str) -> str:
    """禁用派系读数 → 卡牌库派系名。OCR 读出来是「大河」「大河口」「三分」「日落海口」，
    库里是「大河流域」「三分之地」—— 原来直接比字符串，这两个派系的禁用过滤一直没生效（2026-09-29 查出）。"""
    b = (b or "").strip()
    if not b:
        return ""
    return next((f for f in FACTIONS if f[:2] == b[:2]), b)


def viable(lu: dict, banned: str = "") -> bool:
    """禁用派系后还能不能成立：主C 属于被禁派系 → 废；被禁派系占阵容 1/3 以上 → 废（与 rules._comp_viable 同一判据）。"""
    if excluded(lu):
        return False
    banned = norm_banned(banned)
    if not banned:
        return True
    if banned in ((pieces().get(lu.get("carry") or "") or {}).get("synergies") or []):
        return False
    core = list(dict.fromkeys(final_team(lu) + [h for st in lu.get("stages") or [] for h in st.get("main") or []]))
    nb = sum(1 for n in final_team(lu) if banned in ((pieces().get(n) or {}).get("synergies") or []))
    return nb * 3 < max(1, len(core))


def fit_score(lu: dict, owned) -> float:
    """我方已有英雄对这条路线的覆盖：最终阵容 2 分、运营阶段 1 分、开局过渡件只算一半，已有主C 另 +2；热度作平手裁决。"""
    owned = set(owned)
    fin, stg = set(final_team(lu)), set(stage_units(lu))
    s = 0.0
    for n in owned & (fin | stg):
        s += (2.0 if n in fin else 1.0) * (0.5 if n in OPENERS else 1.0)
    if lu.get("carry") in owned:
        s += 2.0
    return s


def candidates(owned, banned: str = "", top: int = 3, min_fit: float = 2.0, one_per_carry: bool = False) -> list:
    """按覆盖度排的可选路线（同一主C + 同一最终阵容只留最热门的一条；one_per_carry 时每个主C 只留一条）。"""
    res = []
    for lu in ali_lineups():
        if not viable(lu, banned):
            continue
        f = fit_score(lu, owned)
        if f < min_fit:
            continue
        res.append((f, math.log10((lu.get("use_num") or 0) + 1), lu))
    res.sort(key=lambda x: (-x[0], -x[1], str(x[2].get("key"))))
    out, seen = [], set()
    for _, _, lu in res:
        sig = lu.get("carry") if one_per_carry else (lu.get("carry"), frozenset(final_team(lu)))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(lu)
        if len(out) >= top:
            break
    return out
