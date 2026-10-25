"""第 1 层效果表：加载 / Effect 数据类 / amount 公式求值。

来源优先级（每张卡单独取）：
  1. data/engine/effects_v1.json（tools/engine/merge_effects.py 合并 + 校验后的正式表）
  2. data/engine/effects_out/chunk_XX.jsonl（第 1 层代理的原始输出，正式表没生成时直接读）
  3. agent/engine/standin.py（从 cards.json 卡面正则抽取的替身，只覆盖等级 / 古币 / 触发 / 能量等主干）
`EffectDB.source_count()` 报每种来源各多少张。

公式：amount 可以是数字或字符串公式（"lord_level*7"、"faction_count:河洛"、"max(3,round_?(0.08*level))"）。
只允许白名单 AST（数字、变量、+ - * / // % **、比较、min/max/floor/ceil/round/abs）；
出现求不了值的变量 → 返回 None，调用方把这条效果计为"未建模"（不会瞎猜成 0）。
"""
from __future__ import annotations

import ast
import glob
import json
import math
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .normalize import event_of, normalize_records

ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "data" / "engine" / "effects_v1.json"
OUT_GLOB = str(ROOT / "data" / "engine" / "effects_out" / "chunk_*.jsonl")

WHENS = {"deploy", "prepare", "open", "sacrifice", "death", "kill", "battle_start", "round_start", "use", "equip",
         "passive", "merge", "sell", "attack_n", "skill_cast", "other"}
EFFECTS = {"level", "temp_level", "temp_to_perm", "coin", "energy", "card", "hero_card", "equip", "trigger", "revive",
           "shield", "atk", "aspd", "hp", "mana", "pop", "refresh", "lord_xp", "transform", "remove", "steal_level",
           "summon", "core_level", "other"}

# 按名字查卡时同名多条的优先序（李元芳 英雄牌 / 召唤物 同名）
_CAT_PRI = {"英雄牌": 0, "效果牌": 1, "装备牌": 2, "天赋牌": 3, "棋手技能牌": 4, "召唤物": 5, "棋手牌": 6}


@dataclass
class Effect:
    when: str = "other"
    effect: str = "other"
    amount: Any = None
    target: dict = field(default_factory=dict)
    when_n: Optional[int] = None
    permanent: Optional[bool] = None
    trigger_kind: Optional[str] = None
    condition: Any = None
    per: Optional[str] = None
    raw: str = ""
    extra: dict = field(default_factory=dict)   # summon / card / names 等附加字段（原样）
    # 派生（加载时算好，推演热路径不再解析字符串）
    event: str = "other"          # 规范化事件：when 之外补 win / lose / react:<x>
    _code: Any = None             # 编译好的公式（None = 常数或编译失败）
    _const: Optional[float] = None
    _bad: bool = False            # 公式编译失败

    @property
    def who(self) -> str:
        return str((self.target or {}).get("who") or "other")

    @property
    def tfaction(self) -> str:
        return str((self.target or {}).get("faction") or "")

    @property
    def tcount(self) -> int:
        try:
            return max(1, int((self.target or {}).get("count") or 1))
        except Exception:
            return 1

    @classmethod
    def from_dict(cls, d: dict) -> "Effect":
        e = cls(when=str(d.get("when") or "other"), effect=str(d.get("effect") or "other"), amount=d.get("amount"),
                target=dict(d.get("target") or {}), when_n=_int(d.get("when_n")), permanent=d.get("permanent"),
                trigger_kind=d.get("trigger_kind"), condition=d.get("condition"), per=d.get("per"),
                raw=str(d.get("raw") or ""),
                extra={k: v for k, v in d.items() if k not in _CORE})
        e.event = str(d.get("event") or "") or event_of(d)
        a = e.amount
        if isinstance(a, bool):
            e._const = float(a)
        elif isinstance(a, (int, float)):
            e._const = float(a)
        elif a is None:
            e._const = 1.0 if e.effect in ("trigger", "coin", "revive", "summon", "hero_card", "card", "equip") else None
            if e._const is None:
                e._bad = True
        else:
            e._code = compile_formula(str(a))
            if e._code is None:
                e._bad = True
        return e

    def value(self, ctx: dict) -> Optional[float]:
        """在局面 ctx 下求 amount。None = 求不了（变量缺失 / 公式不可解析）。"""
        if self._const is not None:
            return self._const
        if self._code is None:
            return None
        return eval_formula(self._code, ctx)


_CORE = {"when", "effect", "amount", "target", "when_n", "permanent", "trigger_kind", "condition", "per", "raw", "event"}


def _int(v):
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------- 公式
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMP = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Eq: operator.eq,
        ast.NotEq: operator.ne}
_FUNCS = {"min": min, "max": max, "floor": math.floor, "ceil": math.ceil, "round": round, "abs": abs, "int": int}
_COLON_VAR = re.compile(r"\b(faction_count|level_of|count_of|keyword_count):([^\s,()*/+\-]+)")


def compile_formula(s: str):
    s = s.strip().replace("round_?", "round").replace("×", "*").replace("−", "-")
    s = re.sub(r"/s\b", "", s)
    s = _COLON_VAR.sub(lambda m: f'{m.group(1)}("{m.group(2)}")', s)
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call,
                              ast.Compare, ast.USub, ast.UAdd, *_BIN, *_CMP)):
            return None
        if isinstance(n, ast.Call) and not isinstance(n.func, ast.Name):
            return None
    return tree.body


class _Missing(Exception):
    pass


def eval_formula(node, ctx: dict) -> Optional[float]:
    try:
        v = _ev(node, ctx)
        return float(v)
    except (_Missing, ZeroDivisionError, TypeError, ValueError):
        return None


def _ev(n, ctx):
    if isinstance(n, ast.Constant):
        if isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n.value, str):
            return n.value
        raise _Missing()
    if isinstance(n, ast.Name):
        if n.id in ctx and not callable(ctx[n.id]):
            return ctx[n.id]
        raise _Missing(n.id)
    if isinstance(n, ast.BinOp):
        return _BIN[type(n.op)](_ev(n.left, ctx), _ev(n.right, ctx))
    if isinstance(n, ast.UnaryOp):
        v = _ev(n.operand, ctx)
        return -v if isinstance(n.op, ast.USub) else v
    if isinstance(n, ast.Compare):
        left = _ev(n.left, ctx)
        for op, c in zip(n.ops, n.comparators):
            r = _ev(c, ctx)
            if not _CMP[type(op)](left, r):
                return 0
            left = r
        return 1
    if isinstance(n, ast.Call):
        name = n.func.id
        args = [_ev(a, ctx) for a in n.args]
        if name in _FUNCS:
            return _FUNCS[name](*args)
        f = ctx.get(name)
        if callable(f):
            v = f(*args)
            if v is None:
                raise _Missing(name)
            return v
        raise _Missing(name)
    raise _Missing()


# ---------------------------------------------------------------- 卡
@dataclass
class CardFx:
    key: str
    name: str
    category: str = ""
    quality: int = 0
    faction: str = ""
    keywords: list = field(default_factory=list)
    effects: list = field(default_factory=list)          # list[Effect]，卡面
    awaken_effects: list = field(default_factory=list)   # 觉醒版卡面（空 = 按"词条翻倍"处理）
    skill_effects: list = field(default_factory=list)    # 技能（含 10/40/100 级强化）
    equip: dict = field(default_factory=dict)
    notes: str = ""
    source: str = "standin"
    unit_kind: str = ""          # hero / summon / hero_copy / ""（非单位）

    @classmethod
    def from_dict(cls, d: dict, source: str) -> "CardFx":
        ef = lambda L: [Effect.from_dict(x) for x in (L or []) if isinstance(x, dict)]  # noqa: E731
        try:
            q = int(d.get("quality") or 0)
        except Exception:
            q = 0
        return cls(key=str(d.get("key") or d.get("name")), name=str(d.get("name") or d.get("key")),
                   category=str(d.get("category") or ""), quality=q, faction=str(d.get("faction") or ""),
                   keywords=list(d.get("keywords") or []), effects=ef(d.get("effects")),
                   awaken_effects=ef(d.get("awaken_effects")), skill_effects=ef(d.get("skill_effects")),
                   equip=dict(d.get("equip") or {}), notes=str(d.get("notes") or ""), source=source,
                   unit_kind=str(d.get("unit_kind") or ""))

    def fx(self, awakened: bool = False) -> list:
        return self.awaken_effects if (awakened and self.awaken_effects) else self.effects

    def has_keyword(self, kw: str) -> bool:
        return kw in self.keywords or any(kw in (e.raw or "") for e in self.effects)


def read_out_chunks() -> dict:
    """effects_out/chunk_*.jsonl → {key: dict}（坏行跳过）。"""
    out = {}
    for f in sorted(glob.glob(OUT_GLOB)):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict) and (r.get("key") or r.get("name")):
                out[str(r.get("key") or r.get("name"))] = r
    return out


class EffectDB:
    def __init__(self, use_v1: bool = True, use_out: bool = True, use_standin: bool = True):
        from . import standin
        self.cards_raw = standin.load_cards()                       # cards.json 原样（key → dict）
        recs: dict = {}
        self.sources: dict = {}
        if use_v1 and V1_PATH.exists():
            d = json.loads(V1_PATH.read_text(encoding="utf-8"))
            for k, r in (d.get("cards") or d).items():
                if isinstance(r, dict) and not k.startswith("_"):
                    recs[k] = {"key": k, **r}
                    self.sources[k] = "v1"
        if use_out:
            for k, r in read_out_chunks().items():
                if k not in recs:
                    recs[k] = r
                    self.sources[k] = "out"
        if use_standin:
            for k, c in self.cards_raw.items():
                if k not in recs:
                    recs[k] = standin.extract(k, c)
                    self.sources[k] = "standin"
        recs, self.issues = normalize_records(recs, self.cards_raw)
        self.records = recs
        self.by_key: dict[str, CardFx] = {k: CardFx.from_dict(r, self.sources.get(k, "?")) for k, r in recs.items()}
        self.by_name: dict[str, CardFx] = {}
        for c in sorted(self.by_key.values(), key=lambda c: _CAT_PRI.get(c.category, 9)):
            self.by_name.setdefault(c.name, c)

    def get(self, name: str, category: str | None = None) -> Optional[CardFx]:
        if category:
            for c in self.by_key.values():
                if c.name == name and c.category == category:
                    return c
        return self.by_name.get(name) or self.by_key.get(name)

    def unit(self, name: str) -> Optional[CardFx]:
        """场上单位用的条目：英雄牌 > 送出的召唤单位 (赠送) > (召唤物) > 同名召唤物。"""
        c = self.by_name.get(name)
        if c and c.category == "英雄牌":
            return c
        for k in (f"{name}(赠送)", f"{name}(召唤物)", name):
            c2 = self.by_key.get(k)
            if c2 and c2.category in ("召唤物", "英雄牌"):
                return c2
        return c

    def raw(self, name: str) -> dict:
        c = self.get(name)
        return self.cards_raw.get(c.key if c else name) or {}

    # ------------------------------------------------ 公式变量反查：谁会让这个量变大
    VAR_PRODUCERS = {
        # 变量 → 判定函数（effect, card）→ bool。只看效果类型 / 事件，不看卡名
        "ally_units": lambda e, c: e.effect in ("summon",) and e.event in ("passive", "deploy", "prepare", "round_start", "use")
                                   or c.unit_kind == "hero",
        "totems_on_board": lambda e, c: e.effect == "summon" and "图腾" in (e.raw + str(e.extra.get("summon") or "")),
        "totems_count": lambda e, c: e.effect == "summon" and "图腾" in (e.raw + str(e.extra.get("summon") or "")),
        "equips_worn": lambda e, c: e.effect == "equip" or c.category == "装备牌",
        "open_triggers": lambda e, c: e.event == "open",
        "prepare_triggers": lambda e, c: e.event == "prepare" or (e.effect == "trigger" and e.trigger_kind == "prepare"),
        "kills_last_round": lambda e, c: e.effect in ("atk", "aspd", "level", "temp_level"),
    }

    def producers(self, var: str) -> list:
        """公式变量 var 的"产出方"卡名（如 ally_units → 所有在场召唤 / 图腾产出方 + 英雄；open_triggers → 所有开团件）。"""
        base = var.split(":")[0]
        if base == "faction_count":
            fac = var.split(":", 1)[1] if ":" in var else ""
            return sorted({c.name for c in self.by_key.values() if c.unit_kind == "hero" and c.faction == fac})
        f = self.VAR_PRODUCERS.get(base)
        if not f:
            return []
        out = set()
        for c in self.by_key.values():
            for e in c.effects + c.awaken_effects:
                if f(e, c):
                    out.add(c.name)
                    break
        return sorted(out)

    def formula_vars(self, name: str) -> set:
        """一张卡的 amount 公式里引用了哪些变量（'谁增加它'反查的起点）。"""
        import re as _re
        c = self.unit(name) or self.get(name)
        out = set()
        for e in (c.effects + c.awaken_effects + c.skill_effects) if c else []:
            if isinstance(e.amount, str):
                for m in _re.finditer(r"[A-Za-z_][A-Za-z_0-9]*(?::[^\s,()*/+\-]+)?", e.amount):
                    if m.group(0) not in ("min", "max", "floor", "ceil", "round", "abs", "int"):
                        out.add(m.group(0))
            if e.event in ("react:open", "react:prepare|open"):
                out.add("open_triggers")
            if e.event in ("react:prepare", "react:prepare|open"):
                out.add("prepare_triggers")
        return out

    def source_count(self) -> dict:
        out: dict = {}
        for c in self.by_key.values():
            out[c.source] = out.get(c.source, 0) + 1
        return out


_DB: Optional[EffectDB] = None


def db() -> EffectDB:
    global _DB
    if _DB is None:
        _DB = EffectDB()
    return _DB


def reload(**kw) -> EffectDB:
    global _DB
    _DB = EffectDB(**kw)
    return _DB
