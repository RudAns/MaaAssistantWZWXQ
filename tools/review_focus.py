"""每局重点复盘（2026-10-15：重点看 1. 天赋选的对不对 2. 是否又是多个主C 3. 主C 的核心阵容是否凑齐）。

  .venv/Scripts/python.exe -X utf8 tools/review_focus.py <局号>

数据：data/logs/<run>/laya_points.jsonl（每道题的题面 = 当时的局面事实）、ledger.jsonl（换主C 事件）。
输出：一屏的三段结论 + 每段的证据行；同时写 data/logs/<run>/review_focus.md。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]


def jl(f: Path) -> list:
    out = []
    if f.exists():
        for line in f.open(encoding="utf-8", errors="ignore"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def run_of(no: int) -> str:
    for m in jl(P / "data/laya/iterations/batch2_matches.jsonl"):
        if m.get("match") == no:
            return m["run"]
    sys.exit(f"batch2_matches.jsonl 里没有第 {no} 局")


def grab(state: str, pat: str) -> str:
    m = re.search(pat, state or "", re.M)
    return m.group(1).strip() if m else ""


BAD_TALENT = re.compile(r"INACTIVE|UNUSABLE|board 0, hand 0|needs \d+ .*have 0|0 heroes with own|already taken")


def _talent_grades() -> dict:
    """天赋审查表的 名 -> (等级, 理由)。"""
    f = P / "docs" / "talent-ali-fit-review-2026-10-19.md"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 5 and c[2] in ("A", "B", "C"):
                out[c[0]] = (c[2], c[4])
    # 2026-10-19 晚的裁决覆盖审查表
    for n in ("零钱兑换", "随机强化", "随机进化", "随机成长", "群星璀璨", "拾荒者", "兵行诡招", "极限操作", "搏命"):
        out[n] = ("C", "裁决低价值")
    return out


_CARDS = None
_RK = {"A": 0, "B": 1, "C": 2}
_FAC = {"三分": "三分之地", "大河": "大河流域", "三分之地": "三分之地", "大河流域": "大河流域", "逐鹿": "逐鹿", "河洛": "河洛", "日落海": "日落海"}
_TACTIC = ("令行如流", "乘胜追击", "演武")


def _team_of(p: dict, led: list) -> tuple[set, str]:
    """这道天赋题当时的"我方英雄"：场上 + 手里 + 已锁路线的官方阵容；以及主C。"""
    global _CARDS
    if _CARDS is None:
        _CARDS = json.loads((P / "data/cards.json").read_text(encoding="utf-8"))["cards"]
    st = p.get("state") if isinstance(p.get("state"), str) else ""
    heroes = {n for n in re.findall(r"^\s{2}(\S+) t\d", st, re.M) if n in _CARDS}
    heroes |= {n for n in re.findall(r"[（(]?(\S+?)[)）]?(?=[,，;；]|$)", grab(st, r"^heroes waiting in hand[^:]*:? ?(.*)$") or "") if n in _CARDS}
    rnd = p.get("round") or 0
    route = ""
    for e in led:
        if e.get("event") == "锁定路线" and (e.get("round") or 0) <= rnd:
            route = e.get("阵容") or route
    if route:
        for f in ("data/lineups.json", "data/custom_lineups.json"):
            try:
                L = json.loads((P / f).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            L = L.get("lineups", L) if isinstance(L, dict) else L
            for v in (L.values() if isinstance(L, dict) else L):
                if isinstance(v, dict) and v.get("name") == route:
                    heroes |= {h.get("name") if isinstance(h, dict) else h for h in v.get("heroes") or []}
    carry = re.split(r"[ (]", grab(st, r"^FINAL carry settled: (.*)$") or grab(st, r"^future carry: (.*)$") or "")[0]
    return {h for h in heroes if h}, carry


def _talent_cond(name: str, p: dict, led: list) -> tuple[str, str]:
    """天赋硬条件（data/talent_conditions.json，agent/decision/talent_conditions.py 求值）在这局满不满足：返回 (等级, 不满足原因)。"""
    sys.path.insert(0, str(P))
    from agent.decision import talent_conditions as _tc
    team, carry = _team_of(p, led)
    if carry and int((_CARDS.get(carry) or {}).get("quality") or 0) <= 1:
        carry = ""          # 1 阶临时主C 不按他判主C 类条件（同线上 54.23-2）
    rc = ""
    for e in led:
        if e.get("event") == "锁定路线" and (e.get("round") or 0) <= (p.get("round") or 0):
            rc = e.get("主C") or rc
    if rc and rc not in team - set(re.findall(r"\S+", "")):     # 路线主C 没到手 = 路线没定（54.28-2）
        st = p.get("state") if isinstance(p.get("state"), str) else ""
        if f"carry {rc} not owned yet" in st:
            rc = ""
    # 同线上（laya_policy 天赋题）：路线没锁时按开局配置预计的路线主C 判、带上本局禁的阵营（10-22：937 / 941 局误报"路线未锁"）
    _ban = ((p.get("ctx") or {}).get("banned") or "") if isinstance(p.get("ctx"), dict) else ""
    if not _ban:
        _m = re.search(r'"banned": "([^"]*)"', json.dumps(p, ensure_ascii=False))
        _ban = _m.group(1) if _m else ""
    try:
        from agent.decision import lord_config as _LCf
        _oc = _LCf.load().opening_choice(_ban) if not rc else None
        _tc.EXPECTED_ROUTE_CARRY = ((_oc or {}).get("members") or ("",))[0]
        _tc.BANNED_FACTION = _LCf._norm_faction(_ban)
    except Exception:                                   # noqa: BLE001
        pass
    g, why = _tc.check(name, team, carry, rc)
    if g == "?":
        g, why = _talent_extra(name, rc or carry)
    return g, why


def _talent_extra(name: str, carry: str) -> tuple[str, str]:
    """天赋表外的两类（10-21 补充）：阿离 6 级专属天赋（规范 42.6）、三选一里直接给的装备（线上同口径 live_fits）。"""
    from agent.decision import hero_traits as _ht
    if not carry:
        return "?", ""
    fast = _ht.attack_speed_carry_live(carry)
    if name == "枫之洪流":
        return ("A", "") if not fast else ("B", f"技能型主C 才是质变，{carry} 是攻速型")
    if name == "翩跹馈赠":
        return ("A", "") if fast else ("C", f"普攻快的主C 才有用，{carry} 是技能型")
    if name == "落英汲取":
        return "B", "转移后范围法伤 + 回血：前排 / 近战主C 才吃得到"
    if (_CARDS.get(name) or {}).get("category") == "装备牌":
        from agent.decision.rules import RuleEngine as _RE
        if name in set(_RE.NEVER_PICK) | set(_RE.LORD_USELESS_ITEMS) | set(_RE.NOT_FOR_CARRY_LIVE):
            return "C", f"{name} 阿离用不上 / 主C 不穿（线上 NEVER_PICK / LORD_USELESS / NOT_FOR_CARRY_LIVE）"
        ok, why = _ht.live_fits(name, carry)
        txt = str((_CARDS.get(name) or {}).get("skill") or "")
        if "限定近战" in txt and not _ht.is_melee(carry):
            ok, why = False, f"限定近战，{carry} 是远程"
        elif "限定远程" in txt and _ht.is_melee(carry):
            ok, why = False, f"限定远程，{carry} 是近战"
        elif why == "unclassified":
            ok = not ("普通攻击" in txt and carry in _ht.NO_ONHIT_CARRIES)
            why = "" if ok else "靠普攻、主C 是技能型"
        return ("A", "") if ok else ("C", f"对主C {carry} 不适用（{why}）")
    return "?", ""


def main() -> int:
    no = int(sys.argv[1])
    run = sys.argv[2] if len(sys.argv) > 2 else run_of(no)      # 对局中途：直接给 live_ 目录名
    d = P / "data/logs" / run
    pts = jl(d / "laya_points.jsonl")
    led = jl(d / "ledger.jsonl")
    out = [f"# 第 {no} 局重点复盘（{run}）", ""]

    # ---- 1. 天赋
    out.append("## 1. 天赋")
    bad = 0
    for p in pts:
        if p.get("kind") != "talent" or p.get("used") == "shadow":
            continue
        crit = (p.get("question") or {}).get("criteria") or {}
        ch = p.get("laya_choice") or ""
        txt = crit.get(ch, ch)
        flag = bool(BAD_TALENT.search(txt or ""))
        bad += flag
        out.append(f"- R{p.get('round')} 选 **{txt.split(':')[0]}**（Laya {p.get('laya_conf', 0):.2f}，规则答 {p.get('rule_choice')}）"
                   + ("  ⚠ 题面写着不生效" if flag else ""))
        for k, v in crit.items():
            mark = "→" if k == ch else " "
            out.append(f"    {mark} {k}: {str(v)[:150]}")
    out.append(f"- 结论：{'有 %d 次选了题面写着不生效的天赋' % bad if bad else '没有选到题面写着不生效的天赋'}（其余是否最优要看下面的选项）")
    out.append("")

    # ---- 2. 主C
    out.append("## 2. 主C（是否多个）")
    sw = [e for e in led if e.get("event") == "换主C"]
    for e in sw:
        out.append(f"- 换主C：{e.get('旧') or '(无)'} → {e.get('新')}（{e.get('来源', '')}）")
    by_round = {}
    for p in pts:
        st = p.get("state")
        if isinstance(st, str) and p.get("round") is not None:
            by_round[p["round"]] = st                        # 每回合取最后一道题的局面
    multi = 0
    for r in sorted(by_round):
        st = by_round[r]
        outs = grab(st, r"^outputs on board (.*)$")
        carry = grab(st, r"^FINAL carry settled: (.*)$") or grab(st, r"^future carry: (.*)$") or "?"
        extra = grab(outs, r"extra (.*)$")
        # 只有"等级达到主C 六成以上"的额外输出位才算可能的第二主C（引擎产出件 / 低等级过渡件不算，第 646 局误报）
        cname = re.split(r"[ (]", carry)[0]
        clv = grab(st, rf"^\s*{re.escape(cname)} .*? lv (\d+)") if cname else ""
        rivals = []
        for name, lv in re.findall(r"([^\s,]+) lv(\d+)", extra or ""):
            if clv and int(lv) >= 0.6 * int(clv) and int(lv) >= 10:
                rivals.append(f"{name} lv{lv}")
        is_multi = bool(rivals)
        multi += is_multi
        out.append(f"- R{r}: 主C {carry}{(' lv' + clv) if clv else ''} | 场上输出 {outs or '?'}"
                   + (f"  ⚠ 可能的第二主C：{', '.join(rivals)}" if is_multi else ""))
    out.append(f"- 结论：换主C {len(sw)} 次；{'%d 个回合有等级接近主C 的额外输出位（可能是第二主C）' % multi if multi else '没有出现等级接近主C 的第二输出位'}")
    out.append("")

    # ---- 3. 核心阵容
    out.append("## 3. 主C 核心阵容是否凑齐")
    last_full = None
    for r in sorted(by_round):
        st = by_round[r]
        me = grab(st, r"^MIN ENGINE[^:]*: (.*)$")
        core = grab(st, r"^current route core cards owned copies: (.*)$")
        eng = grab(st, r"^level engine: (.*)$")
        full = "missing" not in me if me else None
        if full and last_full is None:
            last_full = r
        out.append(f"- R{r}: 最低配套 {me or '?'} | 核心卡 {core or '?'}"
                   + (f" | 引擎 {eng[:90]}" if eng else ""))
    out.append(f"- 结论：最低配套{'第 %s 回合凑齐' % last_full if last_full else '整局没凑齐'}")
    out.append("")

    # ---- 4. 天赋可用性（2026-10-20 起重点）：选中的天赋按审查表等级（A 直接有用 / B 有条件 / C 低价值）
    out.append("## 4. 天赋可用性（data/talent_conditions.json 硬条件按本局阵容求值）")
    grade = _talent_grades()
    got = [e for e in led if e.get("event") == "获得卡牌" and e.get("类别") == "天赋牌"]
    cnt = {"A": 0, "B": 0, "C": 0, "?": 0}
    for p in pts:
        if p.get("kind") != "talent" or p.get("used") == "shadow":
            continue
        crit = (p.get("question") or {}).get("criteria") or {}
        ch = p.get("laya_choice") or ""
        name = str(crit.get(ch, ch)).split(":")[0].strip()
        if ch == "refresh" or not name:
            continue
        g, why = grade.get(name, ("?", ""))
        g2, bad_cond = _talent_cond(name, p, led)    # 战术牌天赋没逐鹿阵容这类问题，复盘天赋时要能查出来
        if g2 != "?":
            g = g2
        cnt[g] = cnt.get(g, 0) + 1
        alts = [f"{str(v).split(':')[0].strip()}({_talent_cond(str(v).split(':')[0].strip(), p, led)[0]})"
                for k, v in crit.items() if k.startswith("opt") and k != ch]
        out.append(f"- R{p.get('round')} 选 {name}：**{g}**" + (f"（⚠ 条件不满足：{bad_cond}）" if bad_cond else (f"（{why[:60]}）" if g in ("B", "C") and why else ""))
                   + (f" | 另两项 {', '.join(alts)}" if alts else "") + ("  ⚠ 有更高等级的选项" if any(_RK.get(a[-2], 9) < _RK.get(g, 9) for a in alts) else ""))
    if got and not any(p.get("kind") == "talent" for p in pts):
        out.append(f"- 账本记了 {len(got)} 个天赋但没有天赋题记录")
    out.append(f"- 结论：选中 A {cnt['A']} / B {cnt['B']} / C {cnt['C']} / 未评 {cnt['?']}（B 要看条件是否满足）")
    out.append("")

    # ---- 5. 核心英雄的最佳刷出等级（按棋手等级）
    out.append("## 5. 核心英雄 vs 棋手等级（每次刷新见到该品阶的概率：当时等级 / 最佳等级）")
    sys.path.insert(0, str(P))
    from agent.decision import meta as _M
    cards = json.loads((P / "data/cards.json").read_text(encoding="utf-8"))["cards"]
    best_lv = {q: max(range(1, 7), key=lambda L: (_M.tier_odds(q, L), -L)) for q in range(1, 6)}
    seen_missing = {}
    for r in sorted(by_round):
        st = by_round[r]
        lv = grab(st, r"^lord lv (\d+)")
        me = grab(st, r"^MIN ENGINE[^:]*: (.*)$") or grab(st, r"^(ROUTE \(committed.*)$") or grab(st, r"^level engine: (.*)$")
        miss = re.findall(r"missing:? ([^;|]+)", me or "")
        names = [n.strip() for part in miss for n in re.split(r"[,，、/]", part) if n.strip() in cards]
        if not lv or not names:
            continue
        L = int(lv)
        segs = []
        for n in names:
            q = int(cards[n].get("quality") or 0)
            if not q:
                continue
            b = best_lv[q]
            now, best = _M.tier_odds(q, L), _M.tier_odds(q, b)
            seen_missing.setdefault(n, []).append((r, L))
            flag = "  ✗ 当前等级刷不出" if now == 0 else ("  ↑ 偏低" if now < best * 0.6 else "")
            segs.append(f"{n} t{q} {now:.0%}/{best:.0%}@{b}级{flag}")
        if segs:
            out.append(f"- R{r} 棋手 {L} 级：缺 " + "；".join(segs))
    if seen_missing:
        worst = [n for n, v in seen_missing.items()
                 if any(_M.tier_odds(int(cards[n]["quality"]), L) == 0 for _, L in v)]
        out.append(f"- 结论：{'等级不够、根本刷不出的核心件：' + '、'.join(worst) if worst else '缺的核心件在当时等级都刷得出'}")
    else:
        out.append("- 结论：题面没有缺件记录（最低配套已齐或没锁路线）")
    text = "\n".join(out)
    (d / "review_focus.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
