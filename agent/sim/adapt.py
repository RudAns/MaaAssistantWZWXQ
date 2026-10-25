"""真实局面 → 正向推演局面（agent.sim.State）。离线用 states.jsonl 的行，线上用 GameState + 规则账本（复用 agent.engine.state 的解析）。"""
from __future__ import annotations

import re

from agent.engine import state as ES
from .engine import card_db, card_text, category
from .state import State, Unit


def _norm_faction(s: str) -> str:
    """禁用派系读数规范成库名（「逐」「大河口」这类读残的按前缀认）。"""
    from .engine import FACTIONS, FACTION_ALIAS
    s = FACTION_ALIAS.get(s, s)
    if s in FACTIONS:
        return s
    for f in FACTIONS:
        if s and (f.startswith(s[:2]) or s[:2] in f):
            return f
    return ""


def _standing_summons(name: str) -> list:
    """卡面"在场时，召唤X"的常驻召唤物（因缘图腾 / 先知图腾 / 机关傀儡…）。棋盘记录只有英雄，没有它们，要按卡面补回来。"""
    t = card_text(name)
    m = re.search(r"在场时，召唤([^。]*)", t)
    if not m:
        return []
    seg = m.group(1)
    return [n for n in card_db() if category(n) == "召唤物" and n in seg]


def from_row(row: dict, talents: list | None = None, worn: dict | None = None) -> State:
    """states.jsonl 的一行（准备阶段、有 board）→ State。装备名离线只有件数时记 "?"。"""
    p = ES.from_point(row, talents=talents, worn=worn)
    s = State(round=p.round, lord_level=p.lord_level, xp=p.xp, energy=p.energy,
              hand=list(p.hand), talents=list(p.talents), lord_cards=list(dict.fromkeys((list(p.lord_cards) or ["枫叶舞"]) + ["惊鸿游龙"])),
              maple=p.lord_bound or "", carry=p.carry or "", banned=_norm_faction(getattr(p, "banned", "") or ""))
    for i, u in enumerate(p.units):
        s.units.append(Unit(name=u.name, level=float(u.level or 1), awaken=int(u.awaken or 0),
                            equips=[e for e in u.equips], bound=list(u.bound), cell=u.cell,
                            pop=bool(u.pop), kind=u.kind if u.kind in ("hero", "summon", "object") else "hero",
                            uid=i + 1))
    st = row.get("state") if isinstance(row.get("state"), dict) else {}
    if not st and isinstance(row.get("ctx"), dict):          # laya_points 的题：state 是题面文字，商店在 ctx 里
        st = {"shop": [x if isinstance(x, dict) else {"piece": x} for x in (row["ctx"].get("shop") or [])]}
    s.shop = [x.get("piece") for x in (st.get("shop") or []) if isinstance(x, dict) and x.get("piece") and not x.get("sold")]
    if s.shop:
        s.flags["shop_fresh"] = True                    # 准备阶段第一眼看到的就是这个商店（之后刷新才随机）
    have = {u.name for u in s.units}
    for u in list(s.units):
        for sm in _standing_summons(u.name):
            if sm in have:
                continue
            lv = u.level if "等级与自身相同" in card_text(u.name) else 1.0
            s.units.append(Unit(name=sm, level=lv, pop=False, kind="summon", cell=None, uid=len(s.units) + 1,
                                counters={"att": 12.5}))   # 累计计数（因缘图腾的普攻数）不知道，按期望取一半
            have.add(sm)
    s.next_uid = len(s.units) + 1
    return s
