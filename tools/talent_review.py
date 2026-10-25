"""天赋选择复盘（只读）：抽出每局每次天赋三选一 + 选后兑现情况的原始事实。

用法：
  .venv/Scripts/python.exe -X utf8 tools/talent_review.py 600 624 [--out data/analysis/talent_review_600_624.json]

输出每次选择：局号 / 名次 / 回合 / 三个选项（名字+卡面）/ Laya 概率 / 选了哪个 / 题面 criteria，
以及选后事实：剩余回合、之后卖了哪些场上英雄（含最近读到的等级）、天赋送的牌后来怎么用、
各回合场上阵营人数最大值、4-5 阶人数、主C 变化。价值判断不在这里做（写在报告里）。
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "logs"


def load_cards():
    cards = json.load(open(ROOT / "data" / "cards.json", encoding="utf-8"))["cards"]
    heroes, talents, anyc = {}, {}, {}
    for k, c in cards.items():
        nm = c.get("name") or k
        anyc.setdefault(nm, c)
        if c.get("category") == "英雄牌":
            heroes[nm] = c
        if c.get("category") == "天赋牌":
            talents[nm] = c
    return heroes, talents, anyc


def ranks():
    out = {}
    p = ROOT / "data" / "laya" / "iterations" / "batch2_matches.jsonl"
    for l in open(p, encoding="utf-8"):
        try:
            d = json.loads(l)
        except Exception:
            continue
        out[d["match"]] = d.get("rank")
    return out


TALENT_RE = re.compile(r"\[talent\] 回合(\d+).*?天赋:(\[[^\]]*\]) => (\[.*\])")
LAYA_RE = re.compile(r"Laya\[talent\] 选 (\w+)（([0-9.]+)")
LEDGER_RE = re.compile(r"账本: (\S+) (.*)")
ROUND_RE = re.compile(r"\[(?:prepare|battle|auction)\] 回合(\d+)")
RANK_RE = re.compile(r"本局名次：第 (\d) 名")


def parse_kv(s):
    d = {}
    for m in re.finditer(r"(\S+?)=(\S+)", s):
        d[m.group(1)] = m.group(2)
    return d


def run_of(n):
    t = (LOGS / f"agent_one_{n}.log").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"live_17\d+", t)
    return t, (m.group(0) if m else None)


def analyse(n, heroes, talents, anyc, rk):
    text, run = run_of(n)
    rank = rk.get(n)
    if rank is None:
        m = RANK_RE.findall(text)
        rank = int(m[-1]) if m else None
    rd = LOGS / run if run else None
    # laya points: talent questions + per-round board
    tq, boards = [], {}
    if rd and (rd / "laya_points.jsonl").exists():
        for l in open(rd / "laya_points.jsonl", encoding="utf-8"):
            try:
                p = json.loads(l)
            except Exception:
                continue
            r = p.get("round") or p.get("ctx", {}).get("round")
            if p.get("kind") == "talent":
                tq.append(p)
            if r and r <= 30 and p.get("board"):
                boards[r] = (p.get("board"), p.get("ctx", {}))
    # sequential log scan
    events, cur = [], 0
    picks = []
    pending_laya = None
    for line in text.splitlines():
        m = ROUND_RE.search(line)
        if m and int(m.group(1)) <= 30:
            cur = int(m.group(1))
        m = LAYA_RE.search(line)
        if m:
            pending_laya = (m.group(1), float(m.group(2)))
        m = TALENT_RE.search(line)
        if m:
            r = int(m.group(1))
            opts = eval(m.group(2))
            act = m.group(3)
            nm = re.search(r"'name': '([^']*)'", act)
            picks.append({"round": r, "opts": opts, "chosen": nm.group(1) if nm else "refresh",
                          "laya": pending_laya, "seq": len(events)})
            pending_laya = None
            cur = max(cur, r)
            continue
        m = LEDGER_RE.search(line)
        if m:
            events.append((cur, m.group(1), parse_kv(m.group(2))))
    final_round = max([e[0] for e in events] + [b for b in boards] + [0])
    # dedupe duplicate pick lines (同一回合同一结果打两次)
    ded = []
    for p in picks:
        if ded and ded[-1]["round"] == p["round"] and ded[-1]["opts"] == p["opts"] and ded[-1]["chosen"] == p["chosen"]:
            continue
        ded.append(p)
    # attach laya question
    for p in ded:
        q = None
        for cand in tq:
            raw = [t[0] for t in (cand.get("raw") or {}).get("talents", [])]
            if (cand.get("round") == p["round"]) and raw == p["opts"]:
                q = cand
        p["q"] = q

    def lv_of(hero, upto_round):
        best = None
        for r in sorted(boards):
            if r > upto_round:
                break
            b = boards[r][0].get(hero)
            if b and b.get("lv") is not None:
                best = b["lv"]
        return best

    out = []
    for p in ded:
        r0, seq = p["round"], p["seq"]
        after = events[seq:]
        sells = [(r, e.get("英雄"), lv_of(e.get("英雄"), r), e.get("原因", "")[:30]) for r, ev, e in after if ev == "卖出"]
        hand_sells = [(r, e.get("牌")) for r, ev, e in after if ev == "卖手牌"]
        # 天赋送的牌：同回合 / 下一回合内 获得的天赋牌 / 装备牌 / 效果牌
        given = [(r, e.get("牌"), e.get("类别")) for r, ev, e in after[:80]
                 if ev == "获得卡牌" and e.get("类别") in ("天赋牌", "装备牌", "效果牌") and r <= r0 + 1]
        gnames = {g[1] for g in given}
        uses = [(r, e.get("牌"), e.get("目标")) for r, ev, e in after if ev == "用牌" and e.get("牌") in gnames]
        # 阵营 / 品阶统计（选后每回合场上）
        fac_max, hi_max, fac_by_r = Counter(), 0, {}
        for r in sorted(boards):
            if r < r0:
                continue
            b, ctx = boards[r]
            fc = Counter(heroes[h]["faction"] for h in b if h in heroes)
            fac_by_r[r] = dict(fc)
            for f, c in fc.items():
                fac_max[f] = max(fac_max[f], c)
            hi = sum(1 for h in b if h in heroes and heroes[h].get("quality", 0) >= 4)
            hi_max = max(hi_max, hi)
        # 每回合场上概况（判断条件类天赋兑现了没有）
        per_round = []
        for r in sorted(boards):
            if r < r0:
                continue
            b, ctx = boards[r]
            hs = [h for h in b if h in heroes]
            q = lambda h: heroes[h].get("quality", 0)
            kws = lambda h: " ".join(k["name"] for k in heroes[h].get("keywords", [])) + heroes[h].get("skill", "")
            top = max([q(h) for h in hs] or [0])
            per_round.append({
                "r": r, "carry": ctx.get("carry"),
                "top_tier": top, "top_heroes": [h for h in hs if q(h) == top],
                "factions": sorted({heroes[h]["faction"] for h in hs}),
                "awakened": [h for h in hs if b[h].get("awk_left") == 0],
                "prep_heroes": [h for h in hs if "整备" in kws(h)],
                "t45": [h for h in hs if q(h) >= 4],
                "no_faction": [h for h in hs if heroes[h]["faction"] == "无阵营"],
                "lv": {h: b[h].get("lv") for h in hs},
            })
        carries = []
        for r in sorted(boards):
            if r >= r0:
                c = boards[r][1].get("carry")
                if not carries or carries[-1][1] != c:
                    carries.append((r, c))
        q = p["q"] or {}
        crit = (q.get("question") or {}).get("criteria") or {}
        rec = {
            "match": n, "rank": rank, "round": r0, "final_round": final_round,
            "opts": [{"name": o, "text": (talents.get(o) or anyc.get(o) or {}).get("skill", "?"),
                      "given": (talents.get(o) or {}).get("given_skill", "")} for o in p["opts"]],
            "chosen": p["chosen"], "laya": q.get("laya"), "laya_choice": q.get("laya_choice"),
            "rule_choice": q.get("rule_choice"), "criteria": crit,
            "sells_after": sells, "hand_sells_after": hand_sells,
            "given": given, "given_uses": uses,
            "faction_max_after": dict(fac_max), "faction_by_round": fac_by_r,
            "t45_max_after": hi_max, "carries_after": carries,
            "per_round": per_round,
            "board_at_pick": (boards.get(r0) or boards.get(r0 - 1) or ({}, {}))[0],
        }
        out.append(rec)
    return out


def main():
    a, b = int(sys.argv[1]), int(sys.argv[2])
    outp = ROOT / "data" / "analysis" / f"talent_review_{a}_{b}.json"
    if "--out" in sys.argv:
        outp = Path(sys.argv[sys.argv.index("--out") + 1])
    heroes, talents, anyc = load_cards()
    rk = ranks()
    allr = []
    for n in range(a, b + 1):
        if not (LOGS / f"agent_one_{n}.log").exists():
            continue
        allr += analyse(n, heroes, talents, anyc, rk)
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(allr, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(allr)} 次天赋选择 -> {outp}")
    pick = Counter(r["chosen"] for r in allr)
    offer = Counter(o["name"] for r in allr for o in r["opts"])
    print("选/出现:", ", ".join(f"{k} {pick[k]}/{offer[k]}" for k, _ in pick.most_common()))


if __name__ == "__main__":
    main()
