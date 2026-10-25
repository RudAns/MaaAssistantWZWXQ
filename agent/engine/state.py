"""通用局面：Unit / Position。只读地从线上 GameState + RuleEngine 账本、或离线日志行构造。

不含任何棋手 / 路线专用字段：棋手技能（枫叶舞等）和棋手特性、天赋都只是 `lord_cards` / `talents` 里的卡名，
它们的作用全部来自效果表。主C = `carry`（外部设定）或价值最高的输出位 / can_carry 单位（project.pick_carry）。
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import board as B
from .effects import db

ROOT = Path(__file__).resolve().parents[2]


def _summon_attrs() -> dict:
    try:
        return json.loads((ROOT / "data" / "engine" / "summon_units_v1.json").read_text(encoding="utf-8")).get("units") or {}
    except Exception:
        return {}


SUMMON_ATTRS = _summon_attrs()


@dataclass
class Unit:
    name: str
    level: float = 1.0               # 永久等级
    temp: float = 0.0                # 临时等级（战斗结束清除）
    awaken: int = 0                  # 觉醒进度 0~3；>=3 视为已觉醒
    awakened: bool = False
    equips: list = field(default_factory=list)   # 装备名（未知用 "?" 占位）
    bound: list = field(default_factory=list)    # 其中绑定的（出生自带 / 无法拆卸）：不能转移，卖人即失去
    cell: Optional[tuple] = None
    pop: bool = True                 # 占不占人口（召唤物 / 图腾 False）
    kind: str = "hero"               # hero / summon / object
    presence: str = "prep"           # prep = 备战期在场；battle = 只在战斗中出现
    can_carry: bool = False
    can_equip: bool = True
    aspd: float = 0.0                # 本场战斗额外攻速（开团叠加等，战斗结束清）
    atk: float = 0.0                 # 装备 / 永久成长带来的攻击力加成（按伤害类型折算进战力）
    conv_cap: float = 0.0            # "临时转永久"额外上限（登场累加的部分）
    q_fill: int = 0                  # 推演里补位的"替补"单位：品阶（真实单位为 0，按卡库）
    mana_red: float = 0.0
    mana_cut: float = 0.0            # 本回合法力值上限降低（推演战术类，回合开始清零）            # 上一场战斗里"法力值上限 -N"累计的期望值（守约：每个牺牲单位阵亡 -20），用来估施法次数
    src: dict = field(default_factory=dict)      # 推演期间永久等级来源分解 {来源: +级}

    @property
    def is_awakened(self) -> bool:
        return self.awakened or self.awaken >= 3

    @property
    def fx(self):
        return db().unit(self.name)

    @property
    def quality(self) -> int:
        if self.q_fill:
            return self.q_fill
        c = self.fx
        return c.quality if c else 0

    @property
    def faction(self) -> str:
        c = self.fx
        f = c.faction if c else ""
        if self.kind == "summon":
            ca = str((SUMMON_ATTRS.get(self.name) or {}).get("counts_as") or "")
            for fac in ("河洛", "逐鹿", "三分之地", "日落海", "大河流域"):
                if fac in ca:
                    return fac
        return f or ""

    @property
    def counts_as_hero(self) -> bool:
        """卡面"视为【X】英雄"的召唤单位按英雄算（派系计数 / 指定英雄的牌能选中）。"""
        if self.kind == "hero":
            return True
        return "英雄" in str((SUMMON_ATTRS.get(self.name) or {}).get("counts_as") or "")


def _bound_list() -> dict:
    try:
        return json.loads((ROOT / "data" / "engine" / "equip_bound_v1.json").read_text(encoding="utf-8")).get("bound") or {}
    except Exception:
        return {}


BOUND_EQUIPS = _bound_list()


def equip_bound(item: str) -> bool:
    """装备是否绑定（不可转移）：以 data/engine/equip_bound_v1.json 为准（locked 无法拆卸 / innate 英雄自带），
    清单外再看卡面"无法拆卸"兜底。"""
    if item in BOUND_EQUIPS:
        return True
    c = db().get(item)
    if not c:
        return False
    eq = c.equip or {}
    return bool(eq.get("undetachable") or eq.get("unremovable") or "无法拆卸" in c.keywords
                or any("无法拆卸" in (e.raw or "") for e in c.effects))


def make_unit(name: str, level=1, cell=None, awaken=0, equips=None, awakened=False, temp=0) -> Unit:
    c = db().unit(name)
    sa = SUMMON_ATTRS.get(name)
    if sa or (c and c.category == "召唤物" and c.unit_kind != "hero_copy"):
        sa = sa or {}
        ce = sa.get("can_equip")
        return Unit(name=name, level=float(level or 1), temp=float(temp or 0), cell=cell, pop=bool(sa.get("pop") or 0),
                    kind="summon", presence=str(sa.get("presence") or "prep"), can_carry=bool(sa.get("can_carry")),
                    can_equip=ce is True, awaken=int(awaken or 0), awakened=awakened, equips=list(equips or []))
    eq = list(equips or [])
    return Unit(name=name, level=float(level or 1), temp=float(temp or 0), cell=cell, awaken=int(awaken or 0),
                awakened=awakened, equips=eq, bound=[x for x in eq if x != "?" and equip_bound(x)], kind="hero",
                can_carry=True)


@dataclass
class Position:
    round: int = 1
    lord_level: int = 1
    xp: int = 0
    energy: int = 0
    pop_cap: int = 5
    units: list = field(default_factory=list)       # list[Unit]
    hand: list = field(default_factory=list)        # 卡名
    shop: list = field(default_factory=list)        # 卡名（已售空的不列）
    talents: list = field(default_factory=list)     # 已选天赋卡名
    lord_cards: list = field(default_factory=list)  # 棋手技能牌 + 棋手特性（如 枫叶舞 / 已选强化）
    lord_bound: str = ""                            # 指定型棋手技能当前绑定的单位名（枫叶舞宿主等）
    carry: str = ""                                 # 外部设定的主C（空 = 引擎自己挑）
    banned: str = ""
    free_refresh: int = 0
    core: float = 0.0                               # 阿科米亚核心等级（关键词规则）
    hp: int = 60
    meta: dict = field(default_factory=dict)        # 推演中的计数器（击杀 / 触发次数等）

    # ---------------------------------------------------------- 查询
    def unit(self, name: str) -> Optional[Unit]:
        for u in self.units:
            if u.name == name:
                return u
        return None

    def heroes(self) -> list:
        return [u for u in self.units if u.counts_as_hero and u.presence == "prep"]

    def on_board(self) -> list:
        """备战期在场的单位（英雄 + 召唤物 + 图腾）。"""
        return [u for u in self.units if u.presence == "prep" and u.kind != "object"]

    def pop_used(self) -> int:
        return sum(1 for u in self.units if u.pop)

    def free_cells(self) -> list:
        used = {u.cell for u in self.units if u.cell}
        return [c for c in B.CELLS if c not in used]

    def clone(self) -> "Position":
        p = copy.copy(self)
        p.units = [copy.copy(u) for u in self.units]
        for u, o in zip(p.units, self.units):
            u.equips = list(o.equips)
            u.bound = list(o.bound)
            u.src = dict(o.src)
        p.hand, p.shop, p.talents, p.lord_cards = list(self.hand), list(self.shop), list(self.talents), list(self.lord_cards)
        p.meta = dict(self.meta)
        return p

    def summary(self) -> str:
        us = ", ".join(f"{u.name}{'*' if u.is_awakened else ''} lv{u.level:g}{'+%g' % u.temp if u.temp else ''}"
                       f"{' e%d' % len(u.equips) if u.equips else ''}" for u in self.units)
        return (f"R{self.round} lv{self.lord_level} xp{self.xp} E{self.energy} pop{self.pop_used()}/{self.pop_cap} "
                f"carry={self.carry or '-'} | {us} | hand {self.hand}")


# ------------------------------------------------------------------ 构造
def _cellkey(c) -> str:
    return f"{c[0]},{c[1]}" if c else ""


def from_live(state, rules=None) -> Position:
    """线上：GameState + RuleEngine 账本（只读）。"""
    g = lambda o, k, d=None: getattr(o, k, d) if o is not None else d  # noqa: E731
    field_ = dict(g(rules, "_field", {}) or {})
    lvls = dict(g(state, "unit_levels", {}) or {})
    awk = dict(g(state, "awaken_progress", {}) or {})
    eqc = dict(g(state, "equip_counts", {}) or {})
    worn = dict(g(rules, "_worn", {}) or {})
    unit_lv = dict(g(rules, "_unit_lv", {}) or {})
    ever = set(g(rules, "_ever_awakened", set()) or set()) | set(g(rules, "_awakened", set()) or set())
    units = []
    for name, cell in field_.items():
        c = B.cell_of(cell)
        k = _cellkey(c)
        lv = lvls.get(k) or unit_lv.get(name) or 1
        eq = list(worn.get(name) or [])
        eq += ["?"] * max(0, int(eqc.get(k) or 0) - len(eq))
        units.append(make_unit(name, lv, c, awk.get(k) or 0, eq[:B.MAX_EQUIPS], awakened=name in ever))
    _add_summons(units, g(state, "unit_names", {}) or {}, g(state, "vision_units", {}) or {}, lvls,
                 dict(g(rules, "object_cells", {}) or {}))
    hand = list(g(state, "hand", []) or [])
    if not hand:
        for n, k in (g(rules, "_hand_book", {}) or {}).items():
            hand += [n] * int(k or 0)
    shop = [s.piece for s in (g(state, "shop", []) or []) if getattr(s, "piece", None) and not getattr(s, "sold", False)]
    lv = int(g(state, "level", 1) or 1)
    lord_cards = [h for h in hand if (db().get(h) or None) and db().get(h).category == "棋手技能牌"]
    lord_cards += list(g(rules, "_lord_boosts", []) or [])
    mb = g(rules, "_maple_bound", "")
    return Position(round=int(g(state, "round_no", 1) or 1), lord_level=lv, xp=int(g(state, "xp", 0) or 0),
                    energy=int(g(state, "gold", 0) or 0), pop_cap=int(g(state, "population_cap", 0) or 0) or B.pop_cap(lv),
                    units=units, hand=hand, shop=shop, talents=list(g(rules, "_talents_taken", []) or []),
                    lord_cards=list(dict.fromkeys(lord_cards)), lord_bound=mb if isinstance(mb, str) else "",
                    carry=str(g(rules, "_carry_piece", "") or ""), banned=str(g(state, "banned_faction", "") or ""),
                    free_refresh=int(g(state, "free_refresh", 0) or 0), hp=int(g(state, "hp", 60) or 60))


def _add_summons(units, unit_names: dict, vision: dict, lvls: dict, objects: dict):
    have = {u.cell for u in units}
    for k, v in (unit_names or {}).items():
        c = B.cell_of(k)
        n = v[0] if isinstance(v, (list, tuple)) else v
        if not c or c in have or not n:
            continue
        cf = db().unit(n)
        if n in SUMMON_ATTRS or (cf and cf.category == "召唤物" and cf.unit_kind != "hero_copy"):
            units.append(make_unit(n, lvls.get(k) or 1, c))
            have.add(c)
    for k, n in (objects or {}).items():
        c = B.cell_of(k)
        if c and c not in have:
            u = make_unit(n, 1, c)
            u.kind = "object" if not (SUMMON_ATTRS.get(n) or {}).get("can_equip") else u.kind
            units.append(u)
            have.add(c)
    for k, t in (vision or {}).items():
        c = B.cell_of(k)
        if t == "summon" and c and c not in have:
            u = make_unit("召唤物?", lvls.get(k) or 1, c)
            u.kind, u.pop, u.can_carry, u.can_equip = "summon", False, False, False
            units.append(u)
            have.add(c)


def _i(v, d=0) -> int:
    """读数转整数：'?' / None / 坏值 → 默认值（题面里棋手等级、回合等偶尔是 '?'）。"""
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def from_point(row: dict, talents: list | None = None, worn: dict | None = None) -> Position:
    """离线：states.jsonl 的一行（有 state+board）或 laya_points 的一题（有 ctx+board）。"""
    b = row.get("board") or {}
    if "ctx" in row:
        ctx = row.get("ctx") or {}
        st = {"round_no": ctx.get("round") or row.get("round"), "level": ctx.get("level"), "xp": ctx.get("xp"),
              "gold": ctx.get("gold"), "population_cap": ctx.get("pop_cap"), "hand": ctx.get("hand") or [],
              "banned_faction": ctx.get("banned"), "free_refresh": ctx.get("free_refresh"), "hp": ctx.get("hp")}
        carry, maple, boosts = ctx.get("carry") or "", ctx.get("maple") or "", ctx.get("lord_boosts") or []
        talents = talents if talents is not None else list(ctx.get("talents") or [])
    else:
        st = row.get("state") or {}
        carry, maple, boosts = row.get("carry") or "", row.get("maple") or "", row.get("lord_boosts") or []
    lvls = st.get("unit_levels") or {}
    units = []
    for name, d in b.items():
        c = B.cell_of(d.get("cell"))
        awk_left = d.get("awk_left")
        prog = 3 - int(awk_left) if isinstance(awk_left, (int, float)) else 0
        eq = [x for x in (worn or {}).get(name, []) if x][:B.MAX_EQUIPS]   # 装备名：调用方从账本恢复（离线行里只有件数）
        eq += ["?"] * max(0, _i(d.get("eq"), 0) - len(eq))
        units.append(make_unit(name, d.get("lv") or 1, c, max(0, prog), eq))
    _add_summons(units, st.get("unit_names") or {}, st.get("vision_units") or {}, lvls, {})
    hand = [h for h in (st.get("hand") or []) if h]
    lv = _i(st.get("level"), 1)
    lord_cards = [h for h in hand if db().get(h) and db().get(h).category == "棋手技能牌"] + list(boosts or [])
    shop = [(s.get("piece") if isinstance(s, dict) else s) for s in (st.get("shop") or [])]
    shop = [s for s in shop if s]
    return Position(round=_i(st.get("round_no"), 1), lord_level=lv, xp=_i(st.get("xp"), 0),
                    energy=_i(st.get("gold"), 0), pop_cap=_i(st.get("population_cap"), 0) or B.pop_cap(lv),
                    units=units, hand=hand, shop=shop, talents=list(talents or []),
                    lord_cards=list(dict.fromkeys(lord_cards)), lord_bound=maple or "", carry=carry,
                    banned=str(st.get("banned_faction") or ""), free_refresh=_i(st.get("free_refresh"), 0),
                    hp=_i(st.get("hp"), 60))
