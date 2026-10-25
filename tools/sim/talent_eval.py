"""天赋三选一用推演器评估（2026-10-15：天赋选择问题较多，需确认推演器已在线上起作用）。
每个选项：从这道题的真实局面出发，拿这个天赋（不拿 = refresh 的下限参考），之后交给推演打法往后推 N 回合，
按低 / 中 / 高三档战斗情景比 dev_score（养成）和主C 等级；三档都同向才算有结论。

  .venv/Scripts/python.exe -X utf8 tools/sim/talent_eval.py --points data/logs/live_1792047112/laya_points.jsonl
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

AHEAD = 6


def option_name(p: dict, k: str) -> str:
    if k == "refresh":
        return ""
    v = p["question"]["criteria"][k]
    return v.split(":", 1)[0].strip()


def run_option(st0, name: str, seed: int, ahead: int):
    s = st0.clone(keep_log=False)
    e = E.Engine(s, random.Random(seed))
    pl = Player(e, lookahead=1, samples=1)
    if name:
        pl._take("talent", name)
    else:
        pl.play_hand()                       # 对照组走同样的流程（_take 末尾会 play_hand），只差拿不拿
    r_end = s.round + ahead
    while s.round < r_end:
        pl.round()
    cu = s.unit(s.carry) if s.carry else None
    return dev_score(s), (cu.level if cu else 0.0)


def eval_point(p: dict, n: int = 16, ahead: int = AHEAD, seed_off: int = 13) -> dict:
    st0 = from_row(p)
    out = {}
    for sc in ("低", "中", "高"):
        E.set_battle_scenario(sc)
        for k in p["question"]["criteria"]:
            nm = option_name(p, k)
            vals = [run_option(st0, nm, 7919 * i + seed_off, ahead) for i in range(n)]
            if all(vals):
                # 第 3 项 = 每个种子的 dev（同一个 i 在所有选项上是同一个种子 → 可做配对检验，见 verdict）
                out.setdefault(k, {})[sc] = (round(S.mean(v[0] for v in vals), 1), round(S.median(v[1] for v in vals), 1),
                                             [round(v[0], 2) for v in vals])
    E.set_battle_scenario("中")
    return out


def verdict(res: dict, z: float = 2.0) -> str:
    """配对检验（2026-10-15：单次推演分叉太大，两组种子两两比较方向翻转 38%）：
    同一个种子下"第一名 − 每个别的选项"的 dev 差，三档都要 均值 − z·标准误 > 0 才算有结论；否则 ""（推演分不出）。"""
    best = set()
    for sc in ("低", "中", "高"):
        rows = {k: d[sc] for k, d in res.items() if sc in d and len(d[sc]) >= 3}
        if len(rows) < 2:
            return ""
        top = max(rows, key=lambda k: rows[k][0])
        for k, r in rows.items():
            if k == top:
                continue
            ds = [a - b for a, b in zip(rows[top][2], r[2])]
            if len(ds) < 4:
                return ""
            se = S.stdev(ds) / len(ds) ** 0.5 if len(set(ds)) > 1 else 0.0
            if not (S.mean(ds) - z * se > 0 and S.mean(ds) > 0):
                return ""
        best.add(top)
    return next(iter(best)) if len(best) == 1 else ""


def no_effect(res: dict) -> list:
    """推演里和"不拿"每个种子都一样的选项（推演没模拟到 / 只影响战斗 / 条件不满足）。没有 refresh 选项时返回 []。"""
    base = res.get("refresh")
    if not base:
        return []
    return [k for k, d in res.items() if k != "refresh" and all(sc in d and sc in base and d[sc][2] == base[sc][2] for sc in ("低", "中", "高"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ahead", type=int, default=AHEAD)
    a = ap.parse_args()
    for line in open(a.points, encoding="utf-8"):
        p = json.loads(line)
        if p.get("kind") != "talent":
            continue
        try:
            res = eval_point(p, a.n, a.ahead)
        except Exception as ex:                                         # noqa: BLE001
            print(f"R{p['round']} 推演失败：{ex!r}")
            continue
        v = verdict(res)
        print(f"R{p['round']} 线上选 {p.get('laya_choice')}（{p.get('laya_conf')}） 推演三档一致最优：{v and option_name(p, v) or '无一致结论'}")
        for k, d in res.items():
            print(f"  {k:8s} {option_name(p, k) or '不拿（刷新下限）':10s} " + "  ".join(f"{sc} dev {x[0]} 主C {x[1]}" for sc, x in d.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
