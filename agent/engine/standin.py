"""effects_v1.json / effects_out 都没有某张卡时的替身：从 cards.json 卡面按句正则抽取主干效果。

只覆盖推演最常用的几类（等级 / 临时等级 / 古币 / 触发整备 / 能量 / 棋手经验），结构化不了的整句记 effect=other。
输出和第 1 层同一格式（record dict），再统一走 normalize.normalize_records。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_KW = ("登场", "整备", "开团", "牺牲", "凯旋", "败阵", "退场")
_WHEN = {"登场": "deploy", "整备": "prepare", "开团": "open", "牺牲": "sacrifice", "凯旋": "other", "败阵": "other",
         "退场": "sell"}


@lru_cache(maxsize=1)
def load_cards() -> dict:
    c = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))
    return c.get("cards") or c


def _target(s: str, own: str) -> dict:
    m = re.search(r"距离最近的(\d+名)?【([^】]+)】英雄", s)
    if m:
        return {"who": "nearest_faction", "faction": m.group(2)}
    if "距离最近的" in s:
        return {"who": "nearest_ally"}
    m = re.search(r"(所有|全体)【([^】]+)】英雄", s)
    if m:
        return {"who": "faction", "faction": m.group(2)}
    if re.search(r"所有英雄|全体英雄|所有己方英雄", s):
        return {"who": "all_allies"}
    m = re.search(r"随机(\d+)名", s)
    if m:
        return {"who": "random_ally", "count": int(m.group(1))}
    m = re.search(r"指定(\d+)名(【([^】]+)】)?", s)
    if m:
        return {"who": "chosen", "count": int(m.group(1)), **({"faction": m.group(3)} if m.group(3) else {})}
    if "等级最低" in s:
        return {"who": "lowest_level"}
    if own and own in s or "自身" in s:
        return {"who": "self"}
    return {"who": "other"}


def _sentence(s: str, own: str, default_when: str) -> list:
    s = s.strip().strip("。")
    if not s or s in ("转瞬", "唯一", "无法拆卸"):
        return []
    when = default_when
    for k in _KW:
        if s.startswith(k):
            when = _WHEN[k]
            break
    out = []
    tgt = _target(s, own)
    for m in re.finditer(r"(临时等级|等级)(永久)?\+(\d+)", s):
        out.append({"when": when, "target": tgt, "effect": "temp_level" if m.group(1) == "临时等级" else "level",
                    "amount": int(m.group(3)), "permanent": m.group(1) != "临时等级", "raw": s})
    m = re.search(r"使用(\d+)张(精致)?古币", s)
    if m:
        out.append({"when": when, "target": tgt, "effect": "coin", "amount": int(m.group(1)), "raw": s,
                    **({"coin_type": "精致古币"} if m.group(2) else {})})
    m = re.search(r"触发其(整备|开团|登场|牺牲)效果", s)
    if m:
        out.append({"when": when, "target": tgt, "effect": "trigger", "amount": 1,
                    "trigger_kind": {"整备": "prepare", "开团": "open", "登场": "deploy", "牺牲": "sacrifice"}[m.group(1)],
                    "raw": s})
    m = re.search(r"获得(\d+)(点)?能量", s)
    if m:
        out.append({"when": when, "target": {"who": "lord"}, "effect": "energy", "amount": int(m.group(1)), "raw": s})
    m = re.search(r"棋手经验\+(\d+)", s)
    if m:
        out.append({"when": when, "target": {"who": "lord"}, "effect": "lord_xp", "amount": int(m.group(1)), "raw": s})
    if not out:
        out.append({"when": "other", "target": {"who": "other"}, "effect": "other", "raw": s})
    return out


def extract(key: str, c: dict) -> dict:
    cat = c.get("category") or ""
    own = c.get("name") or key
    default = {"效果牌": "use", "天赋牌": "use", "装备牌": "equip", "棋手技能牌": "use"}.get(cat, "passive")

    def parse(text):
        out = []
        for line in re.split(r"[\n。]", text or ""):
            out += _sentence(line, own, default)
        return out
    return {"key": key, "name": own, "category": cat, "quality": c.get("quality") or 0,
            "faction": c.get("faction") or "", "keywords": [k.get("name") if isinstance(k, dict) else k
                                                          for k in (c.get("keywords") or [])],
            "effects": parse(c.get("skill")), "awaken_effects": parse(c.get("awaken")), "skill_effects": [],
            "notes": "standin: cards.json 正则替身"}
