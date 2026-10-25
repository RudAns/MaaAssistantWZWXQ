"""整局校验：默认打法从第 1 回合推到第 14 回合，和实战对比——主C 等级曲线、每局拿到各类牌的张数。
  .venv/Scripts/python.exe -X utf8 tools/sim/whole_game_check.py --n 60
"""
from __future__ import annotations

import argparse
import collections
import random
import statistics as S
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.sim import engine as E  # noqa: E402
from agent.sim.play import Player  # noqa: E402
from agent.sim.state import State  # noqa: E402

REAL_CARRY = {5: 9, 8: 25.5, 10: 41, 12: 84}           # 实战主C 等级中位数（第 480~630 局）
REAL_GAIN = {"英雄牌": 26.2, "效果牌": 22.3, "装备牌": 7.5, "天赋牌": 1.8}   # 实战每局获得（账本，含买的）


def one(i):
    s = State(round=1, energy=6, lord_cards=["枫叶舞", "惊鸿游龙"])
    e = E.Engine(s, random.Random(i))
    p = Player(e, lookahead=1, samples=1)
    lv = {}
    while s.round < 14:
        p.round()
        c = s.unit(s.carry) if s.carry else None
        lv[s.round] = c.level if c else 0
    gain = collections.Counter(E.category(kw.get("card") or "") for (R, k, kw) in s.log if k == "获得" and not kw.get("lost"))
    return lv, dict(gain), s.lord_level, s.merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=20) as ex:
        res = list(ex.map(one, range(a.n)))
    print("主C 等级中位数（推演 / 实战）：" + "  ".join(f"R{R} {S.median(r[0].get(R, 0) for r in res):.0f}/{v}" for R, v in REAL_CARRY.items()))
    tot = collections.Counter()
    for r in res:
        tot.update(r[1])
    print("每局获得（推演 / 实战）：" + "  ".join(f"{k} {tot[k] / a.n:.1f}/{v}" for k, v in REAL_GAIN.items()))
    print(f"第 14 回合棋手等级中位 {S.median(r[2] for r in res)}，合成次数中位 {S.median(r[3] for r in res)}")


if __name__ == "__main__":
    main()
