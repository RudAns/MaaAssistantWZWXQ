"""换人题（swap）用推演器评估：每个选项先把"这一回合卖谁腾位"做掉，之后交给推演打法往后推 N 回合，
按低 / 中 / 高三档战斗情景比 dev_score（养成好坏），三档都同向才算有结论。

  .venv/Scripts/python.exe -X utf8 tools/sim/swap_eval.py --points data/logs/live_1791992321/laya_points.jsonl --round 6
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as S
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from agent.sim import engine as E  # noqa: E402
from agent.sim.adapt import from_row  # noqa: E402
from agent.sim.play import Player, dev_score  # noqa: E402

AHEAD = 4


def waiting_hero(p: dict) -> str:
    ins = p["question"]["instructions"]
    import re
    m = re.search(r"Hero waiting in hand: (\S+)", ins)
    return m.group(1) if m else ""


def option_target(p: dict, k: str) -> str:
    """s2 → 被卖的人；bench → ''；levelup → 'LEVELUP'"""
    if k == "bench":
        return ""
    if k == "levelup":
        return "LEVELUP"
    import re
    m = re.match(r"sell (\S+?):", p["question"]["criteria"][k])
    return m.group(1) if m else "?"


def run_option(st0, opt: str, hero: str, seed: int, ahead: int):
    s = st0.clone(keep_log=False)
    e = E.Engine(s, random.Random(seed))
    pl = Player(e, lookahead=1, samples=1)
    pl.no_swap_round = s.round
    if opt == "LEVELUP":
        cost = E.B.level_cost(s.lord_level, s.xp)
        if cost and e.spend(cost, src="升级"):
            s.xp += cost
            e.level_ups()
    elif opt:
        u = s.unit(opt)
        if u is None:
            return None
        e.sell(u)
    if opt and hero in s.hand and s.pop_used() < e.pop_cap():
        s.hand.remove(hero)
        u2 = s.unit(hero)
        if u2 is not None:
            e.merge(u2)
        else:
            e.deploy(hero, cell=pl.cell_for(hero))
    r_end = s.round + ahead
    while s.round < r_end:
        pl.round()
    return dev_score(s), (s.unit(s.carry).level if s.carry and s.unit(s.carry) else 0.0)


def eval_point(p: dict, n: int = 24, ahead: int = AHEAD) -> dict:
    st0 = from_row(p)
    hero = waiting_hero(p)
    opts = list(p["question"]["criteria"])
    out = {}
    for sc in ("低", "中", "高"):
        E.set_battle_scenario(sc)
        for k in opts:
            tgt = option_target(p, k)
            vals = [run_option(st0, tgt, hero, 7919 * i + 13, ahead) for i in range(n)]
            vals = [v for v in vals if v]
            if vals:
                out.setdefault(k, {})[sc] = (S.mean(v[0] for v in vals), S.median(v[1] for v in vals))
    E.set_battle_scenario("中")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    for line in open(a.points, encoding="utf-8"):
        p = json.loads(line)
        if p.get("kind") != "swap" or (a.round and p.get("round") != a.round):
            continue
        res = eval_point(p, a.n)
        print(f"R{p['round']} 等上场 {waiting_hero(p)} 线上选 {p.get('laya_choice')}")
        for k, d in res.items():
            print(f"  {k:8s} {option_target(p, k) or '不换':6s} " + "  ".join(f"{sc} dev {v[0]:.1f} 主C {v[1]:.0f}" for sc, v in d.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
