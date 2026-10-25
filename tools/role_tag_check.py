"""题面"定位 / 角色"标签审计（2026-10-13：排查定位误标导致的严重 bug）。

原则：主C、副C、主辅助、副辅助、主坦、副坦这些定位，说的都是**核心阵容**（当前路线的最终阵容 / 各阶段成员）里的人。
本脚本逐题读某局的 laya_points.jsonl，把题面（局面描述 + 选项）里贴在每个英雄身上的定位标签拆出来，对照
"当时的路线阵容 + 卡面定位（data/hero_roles.json 的 combat_role）"找四类问题：

  R1 误标·路线外：MAIN TANK / OFF-TANK / ENGINE PART / ENGINE PAYOFF / 引擎缺件 / MIN ENGINE / CORE（主C 已定后）/ ROUTE * / eco 缺件
     贴在不在核心阵容里的人身上
  R2 矛盾：同一道题同一个英雄两个互相打架的标签（CORE vs EXTRA OUTPUT、ENGINE PART vs not in ROUTE、FUTURE CARRY vs TEMPORARY …）
  R3 定位错配：输出位标签（FUTURE CARRY / C-POSITION / EXTRA OUTPUT / 2nd carry / transitional / CARRY CANDIDATE）给了
     没有输出定位的人（combat_role 里没有 carry_phys / carry_mag / assassin），或坦克/辅助标签给了纯输出位
  R4 漏标：路线有效时，路线阵容里的人被标 not in ROUTE / EXTRA OUTPUT / C-POSITION
  R5 目标错：遗志（审定用法 prev_carry，"绝不打给当前主C"）的目标写成主C

核心阵容（scope）：路线有效（facts.Board 判定未失效）→ routes.members(路线) ∪ 路线最低配套（route_min_engine.json）；
路线失效 / 没路线但主C 已定 → 主C + 主C 官方阵容高频队友（≥30%）；都没有 → 不判 R1 / R4。

每道题看两版：题目原文（当时线上的版本，如 f9lm）和用 --ver（默认 f9ln）重渲染的版本。
影响分级：标签出现在**被选中的选项**里 = 3；出现在其他选项里 = 2；只在局面描述里 = 1。排序 = Σ 影响分。

  .venv/Scripts/python.exe -X utf8 tools/role_tag_check.py 623
  .venv/Scripts/python.exe -X utf8 tools/role_tag_check.py 606-623 --show 3
  .venv/Scripts/python.exe -X utf8 tools/role_tag_check.py live_1791904698 --no-rerender
  .venv/Scripts/python.exe -X utf8 tools/role_tag_check.py 606-623 --json out.json
只读，不改任何文件（--json 只写指定的输出文件）。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "laya"))

import features as F  # noqa: E402
import facts  # noqa: E402
import rerender as RR  # noqa: E402
from agent.decision import routes as R  # noqa: E402

# ---------------------------------------------------------------- 标签分类（正则 → 类别）
TAGS = [
    ("MAIN_TANK", re.compile(r"\bMAIN TANK\b")),
    ("OFF_TANK", re.compile(r"\bOFF-TANK\b")),
    ("ENGINE_PART", re.compile(r"ENGINE PART|\bengine part\b|^engine (?:producer|amplifier) of|\(ENGINE (?:producer|amplifier)\)")),
    ("ENGINE_PAYOFF", re.compile(r"ENGINE PAYOFF|^engine payoff of")),
    ("MIN_ENGINE", re.compile(r"MIN ENGINE piece|same MIN ENGINE slot|MIN ENGINE slot already|MIN ENGINE core")),
    ("LATE_FILL", re.compile(r"LATE-FILL")),
    ("CORE", re.compile(r"^CORE of route")),
    ("ROUTE_IN", re.compile(r"^ROUTE (?:member|carry|final team|stage unit|unit)\b|\(ROUTE member\)|^route r\d")),
    ("NOT_ROUTE", re.compile(r"^not in ROUTE")),
    ("FUTURE_CARRY", re.compile(r"FUTURE CARRY")),
    ("CURRENT_CARRY", re.compile(r"^CURRENT CARRY")),
    ("TEMPORARY", re.compile(r"TEMPORARY")),
    ("C_POS", re.compile(r"C-POSITION")),
    ("EXTRA_OUTPUT", re.compile(r"EXTRA OUTPUT")),
    ("SECOND_CARRY", re.compile(r"(?<!not a )2nd carry")),
    ("TRANSITIONAL", re.compile(r"transitional")),
    ("CARRY_CAND", re.compile(r"CARRY CANDIDATE|BACKUP ROUTE carry")),
    ("SUPPORT", re.compile(r"support \(no damage\)")),
    ("FRONT_TANK", re.compile(r"^front tank\b")),
    ("MAPLE_HOST", re.compile(r"MAPLE HOST")),
    ("LOW_VALUE", re.compile(r"low value late")),
]
ROUTE_SCOPED = {"MAIN_TANK", "OFF_TANK", "ENGINE_PART", "ENGINE_PAYOFF", "ENGINE_MISSING", "MIN_ENGINE", "ROUTE_IN", "ECO_WANTED"}
OUTPUT_TAGS = {"FUTURE_CARRY", "C_POS", "EXTRA_OUTPUT", "SECOND_CARRY", "TRANSITIONAL", "CARRY_CAND"}
SUPTANK_TAGS = {"SUPPORT", "MAIN_TANK", "OFF_TANK", "FRONT_TANK"}
# 互相矛盾的标签对（同一题同一英雄）
CONFLICTS = [
    ("CORE", "EXTRA_OUTPUT"), ("CORE", "NOT_ROUTE"), ("CORE", "C_POS"),
    ("ROUTE_IN", "NOT_ROUTE"), ("ROUTE_IN", "EXTRA_OUTPUT"), ("ROUTE_IN", "C_POS"),
    ("MIN_ENGINE", "EXTRA_OUTPUT"), ("MIN_ENGINE", "NOT_ROUTE"), ("MIN_ENGINE", "LATE_FILL"),
    ("ENGINE_PART", "NOT_ROUTE"), ("ENGINE_PAYOFF", "NOT_ROUTE"), ("MAIN_TANK", "NOT_ROUTE"), ("OFF_TANK", "NOT_ROUTE"),
    ("FUTURE_CARRY", "TEMPORARY"), ("FUTURE_CARRY", "EXTRA_OUTPUT"), ("FUTURE_CARRY", "C_POS"), ("FUTURE_CARRY", "SECOND_CARRY"),
    ("FUTURE_CARRY", "MAIN_TANK"), ("CURRENT_CARRY", "EXTRA_OUTPUT"), ("CURRENT_CARRY", "C_POS"),
    ("SUPPORT", "EXTRA_OUTPUT"), ("SUPPORT", "C_POS"), ("SUPPORT", "FUTURE_CARRY"), ("SUPPORT", "TRANSITIONAL"),
    ("MAIN_TANK", "C_POS"), ("MAIN_TANK", "EXTRA_OUTPUT"),
]
OUT_ROLES = {"carry_phys", "carry_mag", "assassin"}
NONOUT_ROLES = {"tank", "support", "controller", "summoner", "utility", "bruiser"}

_HR = None


def hero_roles() -> dict:
    global _HR
    if _HR is None:
        _HR = json.loads((ROOT / "data" / "hero_roles.json").read_text(encoding="utf-8")).get("heroes") or {}
    return _HR


def roles_of(n: str) -> list:
    return list((hero_roles().get(n) or {}).get("combat_role") or [])


def has_output_role(n: str) -> bool | None:
    """有输出定位：combat_role 含 carry_phys / carry_mag / assassin，或是 bruiser 且主C 价值 ≥4（花木兰 6、曜 / 赵云 / 姬小满 4）。
    没有数据返回 None（不判）。"""
    v = hero_roles().get(n) or {}
    r = set(v.get("combat_role") or [])
    if not r:
        return None
    return bool(r & OUT_ROLES) or ("bruiser" in r and float(v.get("carry_value") or 0) >= 4)


def pure_output(n: str) -> bool:
    r = roles_of(n)
    return bool(r) and set(r) <= (OUT_ROLES | {"utility", "summoner"}) and r[0] in OUT_ROLES


_NAMES = None


def hero_names() -> list:
    global _NAMES
    if _NAMES is None:
        cs = facts.cards()
        _NAMES = sorted((n for n in cs if facts.is_hero(n)), key=len, reverse=True)
    return _NAMES


_NAME_RE = None


def name_re():
    global _NAME_RE
    if _NAME_RE is None:
        _NAME_RE = re.compile("|".join(re.escape(n) for n in hero_names()))
    return _NAME_RE


# ---------------------------------------------------------------- 对局定位
def jl(f: Path) -> list:
    out = []
    if f.exists():
        for line in f.open(encoding="utf-8", errors="replace"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def run_dir(arg: str) -> Path:
    if arg.startswith("live_"):
        return ROOT / "data" / "logs" / arg
    log = ROOT / "data" / "logs" / f"agent_one_{arg}.log"
    m = re.search(rb"live_\d+", log.read_bytes()[:400000]) if log.exists() else None
    if not m:
        raise SystemExit(f"找不到第 {arg} 局的 live 目录")
    return ROOT / "data" / "logs" / m.group(0).decode()


def expand(args: list) -> list:
    out = []
    for a in args:
        m = re.fullmatch(r"(\d+)-(\d+)", a)
        out += [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)] if m else [a]
    return out


# ---------------------------------------------------------------- 题面拆标签
def classify(chunk: str) -> list:
    out = []
    for cat, rx in TAGS:
        m = rx.search(chunk)
        if m:
            out.append((cat, m.start()))
    return out


def attribute(chunk: str, subject: str | None) -> list:
    """一段文字里的标签 → [(英雄, 类别, 原文)]：每个标签归给它前面最近出现的英雄名，没有就归给选项主体。"""
    res = []
    names = [(m.start(), m.group(0)) for m in name_re().finditer(chunk)]
    for cat, pos in classify(chunk):
        prev = [n for p, n in names if p < pos]
        who = prev[-1] if prev else subject
        if who:
            res.append((who, cat, chunk.strip()[:160]))
    return res


def split_chunks(s: str) -> list:
    return [c for c in re.split(r" \| |; ", s) if c.strip()]


ENGINE_LINE = re.compile(r"^level engine: (.+?) — producers (.*?); amplifiers (.*?); payoff (.*?); (.*)$")


def parse_state(state: str) -> list:
    """局面描述 → [(英雄, 类别, 原文)]。"""
    out = []
    lines = state.splitlines()
    sec = ""
    pending = None                       # candidate details: "[i] NAME ..." 的下一行属于 NAME
    for ln in lines:
        s = ln.strip()
        if s.startswith("board ("):
            sec = "board"
            continue
        if s.startswith("candidate details"):
            sec = "cand"
            continue
        if not ln.startswith("  "):
            sec = "" if not s.startswith("candidate details") else sec
        if sec == "board" and ln.startswith("  ") and not ln.startswith("     "):
            m = name_re().match(s)
            if m:
                nm = m.group(0)
                for c in s.split(" | ")[1:]:
                    out += attribute(c, nm)
            continue
        if sec == "cand":
            m = re.match(r"\[\d+\] (\S+)", s)
            if m:
                nm = name_re().match(m.group(1))
                pending = nm.group(0) if nm and nm.group(0) == m.group(1) else None
                continue
            if pending and ln.startswith("     "):
                for c in s.split(" | "):
                    out += attribute(c, pending)
                pending = None
            continue
        m = ENGINE_LINE.match(s)
        if m:
            for cat, grp in (("ENGINE_PART", m.group(2)), ("ENGINE_PART", m.group(3)), ("ENGINE_PAYOFF", m.group(4))):
                for n in [x.strip() for x in grp.split(",")]:
                    if n in hero_names():
                        out.append((n, cat, s[:160]))
            mm = re.search(r"missing: ([^;]+)", m.group(5))
            if mm:
                for n in [x.strip() for x in mm.group(1).split(",")]:
                    if n in hero_names():
                        out.append((n, "ENGINE_MISSING", s[:160]))
            continue
        m = re.match(r"^main tank: (\S+);", s)
        if m and m.group(1) != "none":
            out.append((m.group(1), "MAIN_TANK", s[:160]))
            continue
        m = re.match(r"^future carry: (\S+) \(", s)
        if m:
            out.append((m.group(1), "FUTURE_CARRY", s[:160]))
            continue
        m = re.match(r"^current carry (\S+) is TEMPORARY", s)
        if m:
            out.append((m.group(1), "TEMPORARY", s[:160]))
            continue
        m = re.match(r"^outputs on board \d+: core (.*?); extra (.*)$", s)
        if m:
            for n in [x.strip() for x in m.group(2).split(",")]:
                if n in hero_names():
                    out.append((n, "EXTRA_OUTPUT", s[:160]))
            continue
    return out


SUBJ_RE = {
    "shop": re.compile(r"^buy \([^)]*\) (\S+?):"),
    "card_pick": re.compile(r"^(\S+?):"),
    "auction": re.compile(r"^bid for (\S+?):"),
    "swap": re.compile(r"^sell (\S+?):"),
    "carry": re.compile(r"^(?:keep|switch to) (\S+?):"),
    "use_card": re.compile(r"^use on (\S+?):"),
}


def option_subjects(p: dict, q: dict) -> dict:
    """选项 key → 主体英雄：先按选项文字开头解析（重渲染后选项顺序可能和原题 raw 不同），解析不出再用 raw。"""
    raw = p.get("raw") or {}
    kind = p["kind"]
    fb = {}
    if kind == "shop":
        fb = {f"buy{i}": n for i, n, _c in raw.get("shop") or []}
    elif kind == "card_pick":
        fb = {f"opt{i}": n for i, n in enumerate(raw.get("cards") or [])}
    elif kind in ("auction", "swap", "carry"):
        fb = {k: v for k, v in (raw.get("names") or {}).items() if v}
    elif kind == "use_card":
        fb = {f"use{i}": n for i, n in enumerate(raw.get("targets") or [])}
    out = {}
    rx = SUBJ_RE.get(kind)
    for k, v in q["criteria"].items():
        m = rx.match(v) if rx else None
        nm = m.group(1) if m else None
        out[k] = nm if nm and nm in set(hero_names()) else fb.get(k)
    return out


def parse_question(p: dict, q: dict, state: str) -> list:
    """一道题 → [(英雄, 类别, 原文, 位置)]；位置 = "state" 或选项 key。"""
    res = [(h, c, t, "state") for h, c, t in parse_state(state)]
    subj = option_subjects(p, q)
    for k, v in q["criteria"].items():
        for c in split_chunks(v):
            res += [(h, cat, t, k) for h, cat, t in attribute(c, subj.get(k))]
    raw = p.get("raw") or {}
    if p["kind"] == "swap" and raw.get("incoming"):
        ins = q.get("instructions") or ""
        mm = re.search(r"Hero waiting in hand: (.*?)\. Sell one", ins)
        if mm:
            for c in split_chunks(mm.group(1)):
                res += [(raw["incoming"], cat, t, "instructions") for _h, cat, t in attribute(c, raw["incoming"])
                        if _h == raw["incoming"]]
    if p["kind"] == "eco":
        for x in raw.get("missing") or []:
            if isinstance(x, (list, tuple)) and len(x) >= 3:
                res.append((x[0], "ECO_WANTED", f"eco missing: {x[0]} [{x[2]}]", "state"))
    return res


YIZHI_TARGET = re.compile(r"遗志[^\n]*?\n?[^\n]*?target (\S+?) \((?:carry|current)")


def yizhi_flags(p: dict, q: dict, state: str, carry_names: set) -> list:
    """R5：遗志（审定用法 prev_carry："绝不打给当前主C"）的目标写成了主C。只看遗志自己的那段：
    选项文字以"遗志:"开头的、局面描述里 "[i] 遗志 …" 的下一行。"""
    out = []
    segs = [(k, v) for k, v in q["criteria"].items() if v.startswith("遗志:") or v.startswith("遗志 ")]
    lines = state.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"\s*\[\d+\] 遗志 ", ln) and i + 1 < len(lines):
            segs.append(("state", lines[i + 1]))
    for where, t in segs:
        for m in re.finditer(r"target (\S+?) \((carry|current)", t):
            if m.group(1) in carry_names:
                out.append((m.group(1), "YIZHI_CARRY", t.strip()[:160], where))
    return out


# ---------------------------------------------------------------- 核心阵容
def build_ctx(p: dict) -> dict:
    """和 rerender.rerender 同一套还原（路线 / 主C 来历 / 已选天赋 / 场上读数）。"""
    ctx = dict(p.get("ctx") or {})
    if "route" not in ctx and p.get("kind") != "route":
        ctx.update(RR.route_at(p.get("run"), p.get("ts"), ctx.get("round")))
    board = p.get("board") or {}
    ctx["field"] = {n: dict(board.get(n) or {}) for n in (p.get("field") or [])}
    return ctx


def scope_of(p: dict, ver: str) -> dict:
    F.ACTIVE = ver
    F._facts()
    ctx = build_ctx(p)
    try:
        B = facts.Board(ctx)
    except Exception as e:                       # noqa: BLE001
        return {"src": f"err {e}", "team": None}
    route = R.by_key(ctx.get("route"))
    C = B.carry_set() or B.cur_carry or B.fc or ""
    if route and not B.route_stale:
        team = set(R.members(route))
        for g in facts.min_engine_groups(facts.min_engine_of(ctx.get("route")) or {}):
            team |= set(g)
        return {"src": f"路线 {route.get('name') or route.get('key')}（主C {route.get('carry')}）", "team": team,
                "route_carry": route.get("carry") or "", "carry": C, "route_valid": True}
    if C:
        team = {C} | {n for n, w in facts.anchored(C)["mates"].items() if w >= 0.3}
        for lu in R.ali_lineups() + R.custom_lineups():
            if (lu.get("carry") or "") == C:
                team |= set(R.final_team(lu))
        return {"src": f"主C {C} 阵容（路线{'失效' if route else '无'}）", "team": team, "route_carry": "", "carry": C,
                "route_valid": False}
    return {"src": "无路线无主C", "team": None, "route_carry": "", "carry": "", "route_valid": False}


# ---------------------------------------------------------------- 检查
def check(tags: list, sc: dict, final_settled: bool) -> list:
    """tags: [(英雄, 类别, 原文, 位置)] → [(规则, 子类, 英雄, 原文, 位置)]"""
    out = []
    team = sc.get("team")
    by = collections.defaultdict(set)
    first = {}
    for h, c, t, w in tags:
        by[h].add(c)
        first.setdefault((h, c), (t, w))
    for h, c, t, w in tags:
        # R1 路线外
        if team is not None and h not in team:
            if c in ROUTE_SCOPED or (c == "CORE" and final_settled):
                out.append(("R1", c + ("" if sc.get("route_valid") else "（路线失效，按主C 阵容）"), h, t, w))
        # R3 定位错配
        if c in OUTPUT_TAGS and has_output_role(h) is False:
            out.append(("R3", c + "→非输出位", h, t, w))
        if c in SUPTANK_TAGS and pure_output(h):
            out.append(("R3", c + "→纯输出位", h, t, w))
        # R4 漏标
        if team is not None and sc.get("route_valid") and h in team and c in ("NOT_ROUTE", "EXTRA_OUTPUT", "C_POS"):
            out.append(("R4", c + "→路线成员", h, t, w))
        if c == "YIZHI_CARRY":
            out.append(("R5", "遗志目标=主C", h, t, w))
    # R2 矛盾（每个英雄每对只记一次，原文取后一个标签的出处）
    for h, cs in by.items():
        for a, b in CONFLICTS:
            if a in cs and b in cs:
                ta, wa = first[(h, a)]
                tb, wb = first[(h, b)]
                out.append(("R2", f"{a}×{b}", h, f"{ta}  ‖  {tb}", wb if wb != "state" else wa))
    return out


def impact(where: str, chosen: str | None) -> int:
    if where in ("state",):
        return 1
    if chosen and where == chosen:
        return 3
    return 2


def audit_point(p: dict, ver: str | None) -> dict:
    out = {"orig": [], "new": []}
    chosen = p.get("laya_choice")
    q0 = p["question"]
    st0 = p.get("state") or ""
    over = p.get("feature_version") or "?"
    sc0 = scope_of(p, over)
    fs0 = "FINAL carry settled" in st0
    carry_names = {x for x in (sc0.get("carry"), (p.get("ctx") or {}).get("carry")) if x}
    tg0 = parse_question(p, q0, st0) + yizhi_flags(p, q0, st0, carry_names)
    out["orig"] = [(r, sub, h, t, w, impact(w, chosen)) for r, sub, h, t, w in check(tg0, sc0, fs0)]
    out["scope"] = sc0["src"]
    if ver:
        F.ACTIVE = ver
        F._facts()
        RR.ALLOW_NEW_KEYS = True
        try:
            r = RR.rerender(p)
        except Exception as e:                    # noqa: BLE001
            r = None
            out["rerender_err"] = str(e)[:120]
        if r is not None:
            sc1 = scope_of(p, ver)
            st1 = r.get("state") or ""
            fs1 = "FINAL carry settled" in st1
            tg1 = parse_question(r, r["question"], st1) + yizhi_flags(r, r["question"], st1, carry_names)
            out["new"] = [(rr, sub, h, t, w, impact(w, chosen if chosen in r["question"]["criteria"] else None))
                          for rr, sub, h, t, w in check(tg1, sc1, fs1)]
            out["scope_new"] = sc1["src"]
    return out


RULE_CN = {"R1": "误标·路线外", "R2": "矛盾", "R3": "定位错配", "R4": "漏标", "R5": "目标错"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="+", help="局号 / 区间 606-623 / live_xxx")
    ap.add_argument("--ver", default="f9ln", help="重渲染用的题面版本（默认 f9ln）")
    ap.add_argument("--no-rerender", action="store_true")
    ap.add_argument("--show", type=int, default=2, help="每类打印几条例句（优先选项里的）")
    ap.add_argument("--json", help="把逐条结果写到这个文件")
    a = ap.parse_args()
    ver = None if a.no_rerender else a.ver
    sides = ("orig", "new") if ver else ("orig",)
    st = {sd: {"n": collections.Counter(), "q": collections.defaultdict(set), "opt": collections.Counter(),
               "chosen": collections.Counter(), "score": collections.Counter(),
               "heroes": collections.defaultdict(collections.Counter), "ex": collections.defaultdict(list)} for sd in sides}
    rows, npts, vers = [], 0, collections.Counter()
    for m in expand(a.matches):
        try:
            d = run_dir(m)
        except SystemExit as e:
            print(e)
            continue
        for p in jl(d / "laya_points.jsonl"):
            if not p.get("question") or not p.get("kind"):
                continue
            npts += 1
            vers[p.get("feature_version") or "?"] += 1
            res = audit_point(p, ver)
            qid = f"{m}:{p.get('id') or npts}"
            for sd in sides:
                S = st[sd]
                seen = set()
                for r, sub, h, t, w, imp in sorted(res[sd], key=lambda x: -x[5]):
                    key = (r, sub)
                    if (key, h, w) in seen:
                        continue
                    seen.add((key, h, w))
                    S["n"][key] += 1
                    S["q"][key].add(qid)
                    S["score"][key] += imp
                    S["heroes"][key][h] += 1
                    if imp >= 2:
                        S["opt"][key] += 1
                    if imp == 3:
                        S["chosen"][key] += 1
                    exl = S["ex"][key]
                    line = (imp, h, f"第{m}局 R{p.get('round')} {p['kind']} [{w}{'=被选' if imp == 3 else ''}] {t}")
                    if not any(x[1] == h for x in exl):
                        exl.append(line)
                        exl.sort(key=lambda x: -x[0])
                        del exl[a.show:]
                    rows.append({"match": m, "round": p.get("round"), "kind": p["kind"], "id": p.get("id"), "side": sd,
                                 "rule": r, "sub": sub, "hero": h, "where": w, "impact": imp, "text": t,
                                 "chosen": p.get("laya_choice"), "scope": res.get("scope_new" if sd == "new" else "scope")})
    print(f"共 {npts} 道题；原文版本 {dict(vers)}；重渲染 {ver or '（未做）'}")
    for sd in sides:
        S = st[sd]
        print(f"\n===== {'题目原文' if sd == 'orig' else '重渲染 ' + ver}：按影响分排序（被选选项 3 / 其他选项 2 / 只在局面描述 1）=====")
        for key, sc in S["score"].most_common():
            r, sub = key
            top = ", ".join(f"{h}×{c}" for h, c in S["heroes"][key].most_common(6))
            print(f"[{r} {RULE_CN[r]}] {sub}: {len(S['q'][key])} 题 / {S['n'][key]} 处（选项里 {S['opt'][key]}，被选 {S['chosen'][key]}），"
                  f"影响分 {sc}；{top}")
            for _i, _h, s in S["ex"][key]:
                print("    " + s)
    if ver:
        print("\n===== 原文 → 重渲染（题数 / 选项处数）=====")
        keys = set(st["orig"]["n"]) | set(st["new"]["n"])
        for key in sorted(keys, key=lambda k: -(st["new"]["score"][k] or st["orig"]["score"][k])):
            o, n = st["orig"], st["new"]
            print(f"  {key[0]} {key[1]}: {len(o['q'][key])}题/{o['opt'][key]} → {len(n['q'][key])}题/{n['opt'][key]}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"\n逐条结果 {len(rows)} 条 → {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
