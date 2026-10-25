"""棋手配置：主C 名单（开关）、主C 组、开局路线顺序、高品阶主C 的临时主C。

数据在 data/lords/<棋手>/carry_pool.json，每个棋手一份（2026-10-20 晚：名单是阿离这个棋手独有的，后续其他棋手也会有各自的配置）。
这里**只读数据**，不做价值判断；没有配置文件的棋手 → 全部"不限制"，调用方退回旧行为。

2026-10-22 起（写死的打法规则都绑定到具体棋手，便于后续扩展其他棋手的打法）：
  - data/lords/<棋手>/lord_rules.json：该棋手打法的名单 / 阈值（rules / hero_traits 用 `LordRule` 描述符或 `rule()` 读，取不到用代码里的兜底值）；
  - data/lords/<棋手>/talent_conditions.json：天赋硬条件表（agent/decision/talent_conditions.table() 按当前棋手读）；
  - data/lords/<棋手>/interim_overrides.json（可选）：{规则名: true/false}，覆盖 interim_rules.RULES 的默认开关。
当前棋手 = `current()`（默认阿离；主循环识别出别的棋手后 `set_current()`）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LORD = "阿离"

_CACHE: dict = {}
CURRENT = DEFAULT_LORD          # 本局棋手（set_current 改；没识别 = 阿离）


def current() -> str:
    return CURRENT or DEFAULT_LORD


def set_current(lord: str) -> None:
    global CURRENT
    CURRENT = lord or DEFAULT_LORD


def lord_dir(lord: str = "") -> Path:
    return ROOT / "data" / "lords" / (lord or current())


def _read_json(lord: str, fname: str) -> Optional[dict]:
    f = lord_dir(lord) / fname
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _like(v, default):
    """配置里的 JSON 值转成和兜底值同类型（list → tuple / set / frozenset；数字 → int / float）。"""
    if isinstance(default, frozenset) and isinstance(v, list):
        return frozenset(v)
    if isinstance(default, set) and isinstance(v, list):
        return set(v)
    if isinstance(default, tuple) and isinstance(v, list):
        return tuple(v)
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int) and isinstance(v, (int, float)):
        return int(v)
    if isinstance(default, float) and isinstance(v, (int, float)):
        return float(v)
    return v


def _norm_faction(f: str) -> str:
    f = (f or "").strip()
    return {"大河": "大河流域", "三分": "三分之地"}.get(f, f)


class LordConfig:
    def __init__(self, lord: str, data: Optional[dict]):
        self.lord = lord
        self.data = data or {}
        self.carries: dict = self.data.get("carries") or {}
        self.groups: dict = self.data.get("groups") or {}
        self.rules: dict = _read_json(lord, "lord_rules.json") or {}
        self.interim: dict = {k: v for k, v in (_read_json(lord, "interim_overrides.json") or {}).items()
                              if not k.startswith("_") and isinstance(v, bool)}
        self._rule_cache: dict = {}

    # ---- 棋手打法名单 / 阈值（lord_rules.json；每项 {"_doc": 来源, "value": 值}）
    def rule(self, key: str, default=None):
        """取 lord_rules.json 的一项；没有这一项 / 没有文件 → default（代码里的兜底值）。返回值与 default 同类型。
        注意：返回的是缓存对象，调用方不要原地修改。"""
        if key in self._rule_cache:
            return self._rule_cache[key]
        e = self.rules.get(key)
        v = e.get("value") if isinstance(e, dict) and "value" in e else e
        out = default if v is None else _like(v, default)
        self._rule_cache[key] = out
        return out

    def interim_override(self, name: str) -> Optional[bool]:
        """interim_overrides.json 里这条临时规则的开关；没写 → None（用 interim_rules 默认）。"""
        return self.interim.get(name)

    @property
    def present(self) -> bool:
        return bool(self.carries)

    # ---- 主C 开关
    def enabled(self, name: str) -> bool:
        """名单里开着的主C。没有配置 → True（不限制）；名单里没有这个人 → False。"""
        if not self.present:
            return True
        return bool((self.carries.get(name) or {}).get("enabled"))

    def disabled(self) -> tuple:
        return tuple(n for n, v in self.carries.items() if not v.get("enabled"))

    def enabled_list(self) -> tuple:
        return tuple(n for n, v in self.carries.items() if v.get("enabled"))

    def tier(self, name: str) -> int:
        return int((self.carries.get(name) or {}).get("tier") or 0)

    def position(self, name: str) -> str:
        """配置里指定的站位（"back" / "front"）；没写 → ""（按卡库标签）。"""
        return str((self.carries.get(name) or {}).get("position") or "")

    def faction(self, name: str) -> str:
        return _norm_faction((self.carries.get(name) or {}).get("faction") or "")

    def lineup_key(self, name: str) -> str:
        v = self.carries.get(name) or {}
        k = v.get("core_lineup_key")
        if k:
            return str(k)
        ls = [lu for lu in v.get("lineups") or [] if not lu.get("blocked_now")]
        return str(ls[0].get("key")) if ls else ""

    def blocked_lineup(self, key: str) -> bool:
        """阵容本身被屏蔽（双刃亚连 / 日落海安琪拉…）：主C 开着也不走这套。"""
        key = str(key)
        return any(str(lu.get("key")) == key and lu.get("blocked_now")
                   for v in self.carries.values() for lu in v.get("lineups") or [])

    def carry_of_lineup(self, key: str) -> str:
        key = str(key)
        for n, v in self.carries.items():
            if any(str(lu.get("key")) == key for lu in v.get("lineups") or []) or str(v.get("core_lineup_key") or "") == key:
                return n
        return ""

    def substitutes(self, name: str) -> tuple:
        """可互相替代的核心件（carry_pool.json substitutes）：有其中一个就算这个位置凑齐。"""
        return tuple((self.data.get("substitutes") or {}).get(name) or ())

    # ---- 主C 组（共用配件、谁先出用谁）
    def group_of(self, name: str) -> tuple:
        """返回 (组名, 成员)；不在组里 → ("", ())。"""
        for g, v in self.groups.items():
            if name in (v.get("members") or []):
                return g, tuple(v.get("members") or [])
        return "", ()

    # ---- 开局定路线
    def opening_choice(self, banned: str = "") -> Optional[dict]:
        """按 opening_order 取第一个"开着、阵营没被禁"的组 / 主C。返回 {kind, name, members, key, tier}；没有 → None。"""
        ban = _norm_faction(banned)
        for e in self.data.get("opening_order") or []:
            if e.get("group"):
                g = self.groups.get(e["group"]) or {}
                mem = [m for m in g.get("members") or [] if self.enabled(m) and self.faction(m) != ban]
                if not mem:
                    continue
                key = str(g.get("key") or self.lineup_key(mem[0]))
                return {"kind": "group", "name": e["group"], "members": tuple(mem), "key": key,
                        "tier": min(self.tier(m) for m in mem)}
            c = e.get("carry") or ""
            if c and self.enabled(c) and self.faction(c) != ban and self.lineup_key(c):
                return {"kind": "carry", "name": c, "members": (c,), "key": self.lineup_key(c), "tier": self.tier(c)}
        return None

    # ---- 高品阶主C 的临时主C
    def temp_carry_for(self, tier: int) -> dict:
        t = self.data.get("temp_carry") or {}
        if t and tier >= int(t.get("min_tier") or 99):
            return t
        return {}


def load(lord: str = "") -> LordConfig:
    lord = lord or current()
    if lord not in _CACHE:
        f = ROOT / "data" / "lords" / lord / "carry_pool.json"
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        _CACHE[lord] = LordConfig(lord, data)
    return _CACHE[lord]


def reload(lord: str = "") -> LordConfig:
    lord = lord or current()
    _CACHE.pop(lord, None)
    return load(lord)


def rule(key: str, default=None, lord: str = ""):
    """模块级取值：当前棋手（或指定棋手）的 lord_rules.json 一项，取不到用 default。"""
    return load(lord).rule(key, default)


class LordRule:
    """类属性描述符：`NEVER_PICK = LordRule("never_pick", (...兜底...))`。
    实例访问按实例的 `_lord`（RuleEngine 本局棋手），类访问（RuleEngine.CARRY_LATE_DEF_BASE）按 current()。"""

    def __init__(self, key: str, default):
        self.key, self.default = key, default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, owner=None):
        lord = (obj.__dict__.get("_lord") if obj is not None else None) or current()
        return load(lord).rule(self.key, self.default)
