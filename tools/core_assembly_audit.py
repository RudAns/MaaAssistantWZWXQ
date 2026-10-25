"""核心阵容凑齐时间线审计（2026-10-13，只读）。

对第 A~B 局逐局抽取：路线主C / 核心件(core_set) / 最低配套(early_priority) 第一次在商店出现、第一次拿到、第一次上场的回合；
每回合棋手等级、能量（准备阶段开始 / 结束）、付费 / 免费刷新次数、经济网络答案、买入非路线英雄数、卖出数；
看到核心件没买（当时钱够不够、Laya 商店题怎么选的）、核心件买了又卖、锁店、拍卖 / N 选 1 是否拿核心件、路线切换。

  .venv/Scripts/python.exe -X utf8 tools/core_assembly_audit.py --from 560 --to 617
输出 data/analysis/core_assembly_<from>_<to>.json（逐局明细），终端打印分组汇总。
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.decision import meta  # noqa: E402

LOGS = ROOT / "data" / "logs"
MAX_ROUND = 20


def load_cards():
    c = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))
    c = c.get("cards", c)
    out = {}
    for n, x in (c.items() if isinstance(c, dict) else ((y.get("name"), y) for y in c)):
        if n and (n not in out or x.get("category") == "英雄牌"):
            out[n] = x
    return out


CARDS = load_cards()
HEROES = {n for n, x in CARDS.items() if x.get("category") == "英雄牌"}


def q(n):
    return CARDS.get(n, {}).get("quality") or 0


def parse_units(text: str):
    hits = sorted({(text.find(h), h) for h in HEROES if h in text})
    names = []
    for pos, h in hits:
        if not any(h != o and h in o and text.find(o) <= pos < text.find(o) + len(o) for _p, o in hits):
            names.append(h)
    return names


def load_routes():
    d = json.loads((ROOT / "data" / "route_min_engine.json").read_text(encoding="utf-8"))
    out = {}
    for k, v in d.items():
        if not isinstance(v, dict) or not v.get("carry"):
            continue
        groups = []
        for u in v.get("core_set") or v.get("min_set") or []:
            t = str(u.get("unit") or "")
            if t.startswith("（牌）"):
                continue
            names = parse_units(t)
            if names:
                groups.append(names)
        if not any(v["carry"] in g for g in groups):
            groups.insert(0, [v["carry"]])
        out[k] = {"name": v.get("name"), "carry": v["carry"], "core": groups,
                  "early": [x for x in v.get("early_priority") or [] if x in HEROES],
                  "late": [x for x in v.get("late_fill") or [] if x in HEROES], "std": True}
    # 非 route_min_engine 的路线（官网阵容 / 手动指定）：只取主C
    for rel in ("data/lineups.json", "data/custom_lineups.json"):
        for lu in json.loads((ROOT / rel).read_text(encoding="utf-8")).get("lineups", []):
            k = str(lu.get("key"))
            if k in out or not lu.get("carry"):
                continue
            carry = lu["carry"] if isinstance(lu["carry"], str) else (lu["carry"] or {}).get("name", "")
            mem = [h.get("name") for h in lu.get("heroes") or [] if isinstance(h, dict) and h.get("name") in HEROES]
            out[k] = {"name": lu.get("name"), "carry": carry, "core": [[carry]] if carry else [],
                      "early": [carry], "late": mem, "std": False}
    return out


ROUTES = load_routes()


def matches(a, b):
    rows = {}
    for p in sorted((ROOT / "data" / "laya" / "iterations").glob("*_matches.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            m = d.get("match")
            if isinstance(m, int) and a <= m <= b and d.get("run"):
                rows[m] = d
    return [rows[m] for m in sorted(rows)]


def jl(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def analyse(row):
    run = LOGS / row["run"]
    states = jl(run / "states.jsonl")
    ledger = jl(run / "ledger.jsonl")
    points = jl(run / "laya_points.jsonl")
    R = collections.defaultdict(lambda: {"lv": [], "gold": [], "shop": [], "paid_ref": 0, "free_ref": 0, "levelups": 0,
                                         "locks": [], "route": None, "carry": None, "board": set(), "hand": set(),
                                         "pop": None, "free_ref_avail": 0})
    route_seq = []
    shop_seen_gold = collections.defaultdict(dict)   # rnd -> name -> max gold while visible
    for d in states:
        s = d.get("state") or {}
        rnd = s.get("round_no") or 0
        if rnd <= 0 or rnd > MAX_ROUND:     # 回合号误读（1414 / 111）
            continue
        rk = (d.get("route") or {}).get("route")
        if rk and (not route_seq or route_seq[-1][1] != rk):
            route_seq.append((rnd, rk))
        r = R[rnd]
        if s.get("phase") == "prepare":
            if isinstance(s.get("level"), int) and s["level"] > 0:
                r["lv"].append(s["level"])
            if isinstance(s.get("gold"), int) and 0 <= s["gold"] < 100:
                r["gold"].append(s["gold"])
            if rk:
                r["route"] = rk
            if d.get("carry") is not None:
                r["carry"] = d.get("carry")
            r["board"] |= set((d.get("board") or {}).keys())
            r["hand"] |= {x for x in s.get("hand") or [] if x in HEROES}
            r["free_ref_avail"] = max(r["free_ref_avail"], s.get("free_refresh") or 0)
            if s.get("population_valid") and s.get("population_cap"):
                r["pop"] = (s.get("population"), s.get("population_cap"))
            if s.get("shop_open") and s.get("shop"):
                names = [x.get("piece") for x in s["shop"] if x.get("piece") and not x.get("sold")]
                r["shop"].append(names)
                g = s.get("gold") if isinstance(s.get("gold"), int) else -1
                for n in names:
                    shop_seen_gold[rnd][n] = max(shop_seen_gold[rnd].get(n, -1), g)
        if d.get("executed"):
            for a in d.get("actions") or []:
                t = a.get("type")
                if t == "refresh_shop":
                    if (s.get("free_refresh") or 0) > 0:
                        r["free_ref"] += 1
                    else:
                        r["paid_ref"] += 1
                elif t == "buy_xp":
                    r["levelups"] += 1
                elif t == "lock_shop":
                    r["locks"].append(a.get("pieces") or [])
    led = collections.defaultdict(lambda: {"buy": [], "sell": [], "gain": [], "pick": []})
    switches = []
    cur = 0
    for e in ledger:
        rr = e.get("round")
        if isinstance(rr, int) and 0 < rr <= MAX_ROUND:
            cur = rr
        rnd = cur
        ev = e.get("event")
        if ev == "买牌":
            led[rnd]["buy"].append((e.get("牌"), e.get("花费")))
        elif ev in ("卖出", "卖手牌"):
            led[rnd]["sell"].append((e.get("英雄") or e.get("牌"), e.get("原因"), ev))
        elif ev == "获得卡牌" and e.get("类别") == "英雄牌":
            led[rnd]["gain"].append((e.get("牌"), e.get("来源")))
        elif ev == "N选1":
            led[rnd]["pick"].append((e.get("选择"), e.get("候选")))
        elif ev in ("换备选路线", "锁定路线", "换主C"):
            switches.append({"round": rnd, **{k: v for k, v in e.items() if k != "round"}})
    eco = {}
    qs = collections.defaultdict(list)
    for p in points:
        rnd = p.get("round") or (p.get("ctx") or {}).get("round") or 0
        if not isinstance(rnd, int) or rnd > MAX_ROUND:
            continue
        if p.get("kind") == "eco" and rnd not in eco:
            pr = p.get("laya") or {}
            eco[rnd] = max(pr, key=pr.get) if pr else None
        qs[(p.get("kind"), rnd)].append(p)

    final_rk = route_seq[-1][1] if route_seq else None
    rk8 = None
    for rnd, k in route_seq:
        if rnd <= 8:
            rk8 = k
    main = rk8 or final_rk
    route = ROUTES.get(main or "", {})
    last_round = max(R) if R else 0

    def first(pred):
        for rnd in sorted(R):
            if pred(rnd):
                return rnd
        return None

    gained_by_round = collections.defaultdict(set)
    for rnd, v in led.items():
        for n, _src in v["gain"]:
            gained_by_round[rnd].add(n)

    def owned_round(names):
        return first(lambda r: any(n in R[r]["board"] or n in gained_by_round[r] for n in names))

    def board_round(names):
        return first(lambda r: any(n in R[r]["board"] for n in names))

    def seen_round(names):
        return first(lambda r: any(n in sh for sh in R[r]["shop"] for n in names))

    core = []
    for g in route.get("core", []):
        core.append({"units": g, "tier": min(q(x) for x in g), "seen": seen_round(g), "owned": owned_round(g),
                     "board": board_round(g)})
    early = []
    for u in route.get("early", []):
        early.append({"unit": u, "tier": q(u), "seen": seen_round([u]), "owned": owned_round([u]), "board": board_round([u])})
    assembled = None
    if core and all(c["board"] for c in core):
        assembled = max(c["board"] for c in core)
    owned_all = None
    if core and all(c["owned"] for c in core):
        owned_all = max(c["owned"] for c in core)

    route_units = set(x for g in route.get("core", []) for x in g) | set(route.get("early", [])) | set(route.get("late", []))
    rounds = []
    misses = []
    for rnd in sorted(R):
        r = R[rnd]
        rk = r["route"] or main
        rr = ROUTES.get(rk or "", route)
        r_units = set(x for g in rr.get("core", []) for x in g) | set(rr.get("early", [])) | set(rr.get("late", []))
        buys = led[rnd]["buy"]
        hero_buys = [n for n, _c in buys if n in HEROES]
        prior_board = set().union(*[R[x]["board"] for x in R if x < rnd]) if rnd > 1 else set()
        nonroute = [n for n in hero_buys if n not in r_units and n not in prior_board and n not in r["board"]]
        merge = [n for n in hero_buys if n not in r_units and (n in prior_board or n in r["board"])]
        rounds.append({
            "round": rnd, "lv0": r["lv"][0] if r["lv"] else None, "lv1": r["lv"][-1] if r["lv"] else None,
            "g0": r["gold"][0] if r["gold"] else None, "g1": r["gold"][-1] if r["gold"] else None,
            "gmax": max(r["gold"]) if r["gold"] else None,
            "paid_ref": r["paid_ref"], "free_ref": r["free_ref"], "free_avail": r["free_ref_avail"],
            "levelups": r["levelups"], "eco": eco.get(rnd), "buys": len(buys), "hero_buys": len(hero_buys),
            "route_buys": len([n for n in hero_buys if n in r_units]), "nonroute_buys": len(nonroute), "nonroute": nonroute,
            "merge_buys": len(merge), "sells": len(led[rnd]["sell"]), "locks": r["locks"], "route": rk,
            "shop_views": len({tuple(x) for x in r["shop"]}), "pop": r["pop"],
        })
        # 看到核心件 / 最低配套件没买
        owned_before = set().union(*[R[x]["board"] | gained_by_round[x] for x in R if x < rnd]) if rnd > 1 else set()
        bought_now = {n for n, _c in buys}
        targets = [x for g in rr.get("core", []) for x in g] + [x for x in rr.get("early", [])]
        for n in dict.fromkeys(targets):
            if n in shop_seen_gold[rnd] and n not in owned_before and n not in bought_now:
                gq = [p for p in qs[("shop", rnd)] if any(n in str(v) for v in (p.get("question") or {}).get("criteria", {}).values())]
                ex = None
                if gq:
                    p = gq[0]
                    crit = p["question"]["criteria"]
                    key = next((k for k, v in crit.items() if k.startswith("buy") and n in v.split(":")[0]), None)
                    ex = {"id": p.get("id"), "choice": p.get("laya_choice"), "used": p.get("used"),
                          "p_buy": (p.get("laya") or {}).get(key) if key else None, "opt": crit.get(key) if key else None,
                          "chosen_text": crit.get(p.get("laya_choice") or "", "")[:160]}
                is_core = any(n in g for g in rr.get("core", []))
                misses.append({"round": rnd, "unit": n, "tier": q(n), "core": is_core, "gold": shop_seen_gold[rnd][n],
                               "eco": eco.get(rnd), "lv": r["lv"][-1] if r["lv"] else None,
                               "spent_nonroute": len(nonroute), "paid_ref": r["paid_ref"], "laya": ex,
                               "locked": any(n in lk for lk in r["locks"])})
    # 核心件买了又卖
    sold_core = []
    core_names = {x for g in route.get("core", []) for x in g} | set(route.get("early", []))
    for rnd, v in led.items():
        for n, why, ev in v["sell"]:
            if n in core_names:
                sold_core.append({"round": rnd, "unit": n, "why": why, "ev": ev})
    # 拍卖 / N 选 1 有核心件
    picks = []
    for (kind, rnd), ps in qs.items():
        if kind not in ("auction", "card_pick"):
            continue
        for p in ps[:1]:
            crit = (p.get("question") or {}).get("criteria", {})
            names = ((p.get("raw") or {}).get("names") or {})
            for k, v in crit.items():
                head = v.split(":")[0]
                hit = [x for x in core_names if x in head]
                if hit:
                    picks.append({"kind": kind, "round": rnd, "opt": k, "unit": hit[0], "chosen": p.get("laya_choice"),
                                  "p": (p.get("laya") or {}).get(k), "p_chosen": p.get("laya_conf"),
                                  "chosen_text": crit.get(p.get("laya_choice") or "", "")[:140], "id": p.get("id"),
                                  "opt_text": v[:160]})
    return {"match": row["match"], "rank": row.get("rank"), "run": row["run"], "ban": row.get("ban"),
            "route_seq": route_seq, "main": main, "main_name": route.get("name"), "carry": route.get("carry"),
            "std": route.get("std"), "core": core, "early": early, "assembled": assembled, "owned_all": owned_all,
            "last_round": last_round, "rounds": rounds, "misses": misses, "sold_core": sold_core, "picks": picks,
            "switches": switches}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="a", type=int, default=560)
    ap.add_argument("--to", dest="b", type=int, default=617)
    a = ap.parse_args()
    out = [analyse(r) for r in matches(a.a, a.b)]
    p = ROOT / "data" / "analysis" / f"core_assembly_{a.a}_{a.b}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    print("写出", p)
    for m in out:
        cs = " ".join(f"{'/'.join(c['units'])}[见{c['seen']} 得{c['owned']} 上{c['board']}]" for c in m["core"])
        print(f"#{m['match']} 名次{m['rank']} {m['main_name']} 凑齐R{m['assembled']} 结束R{m['last_round']} :: {cs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
