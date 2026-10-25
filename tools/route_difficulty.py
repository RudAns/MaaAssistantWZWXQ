"""各路线"核心凑齐难度"（2026-10-12：给每个阵容的核心英雄凑齐难度一个可计算的数）。

蒙特卡洛：按实测商店概率（meta.TIER_ODDS / 牌池张数）、等级进度表（meta.LEVEL_TARGET）、每回合看商店次数，
模拟"见到就买"，算每条路线的 主C / 最低配套 第几回合能凑齐。可从某个局面出发（已拥有的英雄、当前回合与等级）。

  .venv/Scripts/python.exe -X utf8 tools/route_difficulty.py                       # 开局出发
  .venv/Scripts/python.exe -X utf8 tools/route_difficulty.py --run live_xxx --round 4   # 从某局第 4 回合的局面出发

近似（写在结论旁边）：满池（别人拿走的看不到）、每回合看商店 VIEWS 次（1 次自然 + 免费 / 付费刷新）、
不算拍卖 / N 选 1 / 抽英雄牌（这些只会让结果更快）、买得起。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.decision import meta  # noqa: E402

VIEWS = 3          # 每回合看几次商店（1 自然 + 2 刷新；免费刷新常有 4~8 次，付费刷新 2 能量）
LAST_ROUND = 12


def cards() -> dict:
    c = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))
    c = c if isinstance(c, list) else list((c.get("cards") or c).values()) if isinstance(c, dict) else c
    out = {}
    for x in c:   # 同名多条（召唤物版的铠 / 李元芳）：英雄牌优先
        if isinstance(x, dict) and x.get("name") and (x["name"] not in out or x.get("category") == "英雄牌"):
            out[x["name"]] = x
    return out


def routes() -> list:
    d = json.loads((ROOT / "data" / "route_min_engine.json").read_text(encoding="utf-8"))
    C = cards()
    out = []
    for k, v in d.items():
        if not isinstance(v, dict) or not v.get("carry"):
            continue
        groups = []
        for u in v.get("core_set") or v.get("min_set") or []:
            t = str(u.get("unit") or "")
            if t.startswith("（牌）"):
                continue
            heroes = [h for h, c in C.items() if c.get("category") == "英雄牌"]
            hits = sorted({(t.find(h), h) for h in heroes if h in t})
            names = []
            for pos, h in hits:            # 长名优先：「百里守约」里不再算出「守约」之类的子串
                if not any(h != o and h in o and t.find(o) <= pos < t.find(o) + len(o) for _p, o in hits):
                    names.append(h)
            if names:
                groups.append(names)
        if v.get("carry") and not any(v["carry"] in g for g in groups):
            groups.insert(0, [v["carry"]])
        out.append({"key": k, "name": v.get("name"), "carry": v.get("carry"), "groups": groups})
    return out


def sim(groups: list, owned: set, start_round: int, start_level: int, n: int = 4000, seed: int = 1) -> dict:
    """返回 {"carry": [回合...], "all": [回合...]}（凑不齐记 99）。groups[0] 是主C。"""
    C = cards()
    rng = random.Random(seed)
    res = {"carry": [], "all": []}
    for _ in range(n):
        have = {i for i, g in enumerate(groups) if any(x in owned for x in g)}
        rc = start_round if 0 in have else 99
        ra = start_round if len(have) == len(groups) else 99
        for rd in range(start_round + 1, LAST_ROUND + 1):
            lv = max(start_level, meta.level_target(rd))
            for _v in range(VIEWS):
                for i, g in enumerate(groups):
                    if i in have:
                        continue
                    p = 1 - 1
                    miss = 1.0
                    for x in g:
                        q = C.get(x, {}).get("quality") or 0
                        miss *= 1 - meta.shop_hit_prob(q, lv)
                    if rng.random() < 1 - miss:
                        have.add(i)
            if rc == 99 and 0 in have:
                rc = rd
            if ra == 99 and len(have) == len(groups):
                ra = rd
                break
        res["carry"].append(rc)
        res["all"].append(ra)
    return res


def summary(v: list) -> str:
    ok = [x for x in v if x < 99]
    p = lambda r: sum(1 for x in v if x <= r) / len(v)  # noqa: E731
    med = statistics.median(v) if len(ok) > len(v) / 2 else None
    return (f"中位 R{med:.0f}" if med else "中位 >R12") + f"  ≤R6 {p(6):.0%}  ≤R8 {p(8):.0%}  ≤R10 {p(10):.0%}"


def owned_at(run: str, rnd: int):
    last = None
    for line in (ROOT / "data" / "logs" / run / "states.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        s = r.get("state") or {}
        if s.get("phase") == "prepare" and s.get("round_no") == rnd:
            last = r
    if not last:
        return set(), 1
    s = last["state"]
    return set(last.get("board") or {}) | set(s.get("hand") or []), s.get("level") or meta.level_target(rnd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--round", type=int, default=0)
    a = ap.parse_args()
    owned, lv, rnd = set(), 1, 0
    if a.run:
        owned, lv = owned_at(a.run, a.round)
        rnd = a.round
        print(f"从 {a.run} 第 {rnd} 回合出发：棋手 {lv} 级，已有 {sorted(owned)}")
    C = cards()
    print(f"假设：每回合看商店 {VIEWS} 次、满池、见到就买、等级按进度表（R2 2 级 / R4 3 级 / R7 4 级 / R10 5 级），不算拍卖 / N 选 1 / 抽英雄牌\n")
    for r in routes():
        g = r["groups"]
        tiers = ["/".join(f"{x}{C.get(x, {}).get('quality')}阶" for x in grp) for grp in g]
        s = sim(g, owned, rnd, lv)
        have = [grp for grp in g if any(x in owned for x in grp)]
        print(f"■ {r['name']}（主C {r['carry']}）核心：{'，'.join(tiers)}" + (f"；已有 {len(have)}/{len(g)}" if owned else ""))
        print(f"    主C 到手：{summary(s['carry'])}")
        print(f"    最低配套全齐：{summary(s['all'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
