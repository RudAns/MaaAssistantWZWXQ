"""第 1 层抽取结果的规范化（merge_effects.py 写盘前、EffectDB 加载时都走这里，幂等）。

抽取代理各写各的，同一件事有多种写法。这里只按**卡面原文（raw）**和通用规则统一，不写死卡名：
  1. 派系名统一成库名（大河→大河流域、三分→三分之地），target.faction / 卡的 faction / 公式里的 faction_count:X
  2. summon_name → summon
  3. "每当…时" 的反应器：event = react:<prepare|open|sacrifice|effect_card|coin|merge|level>；
     同一句 raw 被拆成多条（韩信"整备或开团"拆成 prepare + open 两条）→ 合并成一条 react:prepare|open
  4. target.who == other 时按 raw 推断：穿戴装备最多 → most_equipped；阿科米亚核心 → core；
     棋手资源 → lord；raw 点名的单位 → named(names=[…])
  5. 百分数 amount（"50%"、"min(level*1%,50%)"）→ 小数
  6. 放大器 "古币(牌)提升的等级+N" → effect=coin_amp（引擎扩展类型）
  7. "可转化的等级+N"（登场提高临时转永久上限）→ effect=conv_cap
  8. 天赋送出的召唤单位（summon_units_v1.json 里的单位），其效果挂在天赋条目上 → 挪到单位条目（who=self）
  9. 化为熊灵这类 summon 的 target=self → who=summoned
  10. 召唤物类别里的"英雄复制"（名字也是英雄牌）标 unit_kind=hero_copy，真召唤单位标 summon
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMON_UNITS_PATH = ROOT / "data" / "engine" / "summon_units_v1.json"

FACTIONS = ("河洛", "逐鹿", "三分之地", "日落海", "大河流域", "无阵营")
FACTION_ALIAS = {"大河": "大河流域", "大河口": "大河流域", "三分": "三分之地", "通用": "无阵营", "无": "无阵营"}
_FAC_TOKEN = re.compile(r"(?<=[:【])(大河|三分)(?!流域|之地)")
LISTS = ("effects", "awaken_effects", "skill_effects")

_REACT = [
    (re.compile(r"(每当|每次).{0,16}(触发|拥有).{0,6}整备"), "react:prepare"),
    (re.compile(r"(每当|每次).{0,16}触发.{0,6}开团"), "react:open"),
    (re.compile(r"(每当|每次).{0,20}(牺牲|阵亡)"), "react:sacrifice"),
    (re.compile(r"(每当|每次|每)(你)?使用.{0,6}古币|每获得\d*次古币"), "react:coin"),
    (re.compile(r"(每当|每次|每)(你)?(使用|打出).{0,8}效果牌"), "react:effect_card"),
    (re.compile(r"(每当|每次).{0,10}合成"), "react:merge"),
]
_PREFIX = [("凯旋", "win"), ("败阵", "lose"), ("整备", "prepare"), ("开团", "open"), ("登场", "deploy"),
           ("牺牲", "sacrifice"), ("退场", "sell")]
COIN_AMP = re.compile(r"古币(牌)?(提升|增加)的等级\+(\d+)")
CONV_CAP = re.compile(r"可转化的等级\+(\d+)")


def norm_faction(f):
    if not f or not isinstance(f, str):
        return f
    return FACTION_ALIAS.get(f, f)


def _pct(s: str):
    s2 = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: f"({m.group(1)}/100)", s)
    s2 = re.sub(r"(/100\))\s*(?=[A-Za-z_(])", r"*", s2)          # "(25/100)base_hp" → "(25/100)*base_hp"
    if re.fullmatch(r"[+\-]?\(\d+(?:\.\d+)?/100\)", s2.strip()):
        return round(eval(s2.strip().lstrip("+")), 6)  # 纯百分数，安全（正则保证只有数字）
    return s2


def summon_units() -> dict:
    try:
        return json.loads(SUMMON_UNITS_PATH.read_text(encoding="utf-8")).get("units") or {}
    except Exception:
        return {}


def react_event(raw: str) -> str | None:
    if not raw:
        return None
    for rx, ev in _REACT:
        if rx.search(raw):
            if ev == "react:prepare" and "开团" in raw:
                return "react:prepare|open"
            return ev
    return None


def event_of(e: dict) -> str:
    raw = (e.get("raw") or "").strip()
    w = str(e.get("when") or "other")
    if e.get("effect") in ("coin_amp", "conv_cap") or (w == "passive" and e.get("effect") == "temp_to_perm"):
        return w if e.get("effect") != "conv_cap" else "deploy"
    r = react_event(raw)
    if r and w not in ("use", "deploy", "equip"):
        return r
    if w in ("other", "passive"):
        for p, ev in _PREFIX:
            if raw.startswith(p):
                return ev
        if "回合开始时" in raw or "每回合开始" in raw:
            return "round_start"
        if "战斗开始时" in raw:
            return "battle_start"
        if raw.startswith("战斗胜利") or "若战斗胜利" in raw:
            return "win"
    return w


def _infer_who(e: dict, rec: dict, names: set, gifted: list) -> None:
    t = e.setdefault("target", {})
    if not isinstance(t, dict):
        e["target"] = t = {}
    if t.get("faction"):
        t["faction"] = norm_faction(t["faction"])
    who = t.get("who") or "other"
    raw = e.get("raw") or ""
    if who != "other":
        if who == "self" and e.get("effect") == "summon" and (e.get("summon") or e.get("summon_name")):
            t["who"] = "summoned"
        return
    eff = e.get("effect")
    if "穿戴装备最多" in raw:
        t["who"] = "most_equipped"
    elif eff == "core_level" or ("阿科米亚核心" in raw and eff in ("level", "core_level")):
        t["who"] = "core"
        e["effect"] = "core_level"
    elif eff in ("energy", "lord_xp", "refresh", "pop") or "棋手" in raw[:6]:
        t["who"] = "lord"
    else:
        hit = [n for n in sorted(names, key=len, reverse=True) if n and n in raw]
        hit = [n for n in hit if not any(n != o and n in o for o in hit)]
        if gifted and (not hit or re.search(r"二人|两人|他们|均", raw)):
            hit = list(dict.fromkeys(hit + [g for g in gifted if g not in hit])) if re.search(r"二人|两人|他们|均", raw) else hit
        own = rec.get("name")
        if hit == [own] or (not hit and own and own in raw):
            t["who"] = "self"
        elif hit:
            t["who"] = "named"
            t["names"] = hit


def normalize_effect(e: dict, rec: dict, names: set, gifted: list) -> dict:
    e = copy.deepcopy(e)
    if "summon_name" in e and not e.get("summon"):
        e["summon"] = e.pop("summon_name")
    elif "summon_name" in e:
        e.pop("summon_name")
    raw = e.get("raw") or ""
    m = COIN_AMP.search(raw)
    if m:
        e["effect"], e["amount"] = "coin_amp", int(m.group(3))
        e["target"] = {"who": "lord"}
    m = CONV_CAP.search(raw)
    if m and e.get("when") == "deploy":
        e["effect"], e["amount"] = "conv_cap", int(m.group(1))
        e["target"] = {"who": "self"}
    a = e.get("amount")
    if isinstance(a, str):
        a = _FAC_TOKEN.sub(lambda mm: FACTION_ALIAS[mm.group(1)], a)
        a = _pct(a) if "%" in a else a
        e["amount"] = a
    if "穿戴装备最多" in raw and isinstance(e.get("target"), dict):
        e["target"]["who"] = "most_equipped"
    _infer_who(e, rec, names, gifted)
    e["event"] = event_of(e)
    if rec.get("category") in ("英雄牌", "召唤物") and re.search(r"使用效果牌时", raw):
        e["event"] = "react:effect_card"         # 芈月 / 白起：英雄卡面上的"每当你使用效果牌时"是反应器，不是"使用这张牌"
        if "每回合首次" in raw:
            e["per_round_cap"] = 1
    return e


def normalize_records(recs: dict, cards: dict | None = None) -> tuple[dict, list]:
    """recs: {key: record}（抽取原样）→ (规范化后的新 dict, 问题清单)。幂等。"""
    issues: list = []
    out = {k: copy.deepcopy(r) for k, r in recs.items()}
    su = summon_units()
    hero_names = {r.get("name") for r in out.values() if r.get("category") == "英雄牌"}
    unit_names = set(hero_names) | set(su) | {r.get("name") for r in out.values() if r.get("category") == "召唤物"}
    unit_names.discard(None)
    # 被送出 / 召唤出的单位条目：名字 → key（优先 (赠送)，其次 (召唤物)，再其次同名召唤物）
    unit_key = {}
    for k, r in out.items():
        if r.get("category") == "召唤物":
            n = r.get("name")
            if n not in unit_key or k.endswith("(赠送)"):
                unit_key[n] = k
    for k, r in out.items():
        r["faction"] = norm_faction(r.get("faction"))
        if r.get("category") == "召唤物":
            r["unit_kind"] = "hero_copy" if r.get("name") in hero_names else "summon"
        elif r.get("category") == "英雄牌":
            r["unit_kind"] = "hero"
        gifted = []
        for e in r.get("effects") or []:
            g = e.get("card") or e.get("summon") or e.get("summon_name")
            if e.get("effect") in ("hero_card", "summon") and isinstance(g, str) and g in unit_names:
                gifted.append(g)
        r["_gifted"] = gifted
        for L in LISTS:
            new, seen = [], {}
            for e in r.get(L) or []:
                if not isinstance(e, dict):
                    issues.append((k, L, "非 dict 效果"))
                    continue
                ne = normalize_effect(e, r, unit_names, gifted)
                sig = (ne.get("raw"), ne.get("effect"), str(ne.get("amount")))
                if ne["event"].startswith("react:") and sig in seen:       # 同一句反应器被拆成两条
                    continue
                seen[sig] = ne
                new.append(ne)
            r[L] = new
    # 8. 天赋送出的召唤单位：效果挪到单位条目
    for k, r in out.items():
        if r.get("category") != "天赋牌":
            continue
        for g in r.get("_gifted") or []:
            if g not in su or g not in unit_key or unit_key[g] == k:
                continue
            dst = out[unit_key[g]]
            have = {(x.get("raw") or "").rstrip("。") for L in LISTS for x in dst.get(L) or []}
            for L in LISTS:
                keep = []
                for e in r.get(L) or []:
                    t = e.get("target") or {}
                    names = t.get("names") or []
                    if e.get("effect") not in ("hero_card", "summon", "card") and (names == [g] or (t.get("who") == "self" and g in (e.get("raw") or ""))):
                        if (e.get("raw") or "").rstrip("。") not in have:
                            ne = copy.deepcopy(e)
                            ne["target"] = {k2: v for k2, v in t.items() if k2 != "names"} | {"who": "self"}
                            ne["moved_from"] = k
                            dst.setdefault("effects", []).append(ne)
                            issues.append((k, L, f"效果挪到 {unit_key[g]}: {(e.get('raw') or '')[:24]}"))
                        continue
                    keep.append(e)
                r[L] = keep
    for r in out.values():
        r.pop("_gifted", None)
    return out, issues
