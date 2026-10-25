"""推演器不变量检查（2026-10-15）：只加资源、不减任何东西的天赋（白送能量 / 人口 / 英雄牌 / 装备）拿了不应该变差。
  .venv/Scripts/python.exe -X utf8 tools/sim/free_resource_probe.py [--states 12 --seeds 4 --ahead 3]
输出每张天赋：平均 dev 差、变差的比例；以及变差样本里拿 / 不拿时的主C 是谁、多少级（诊断用）。"""
from __future__ import annotations
import argparse, json, random, statistics as S, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P))
from agent.sim import engine as E                     # noqa: E402
from agent.sim.adapt import from_row                  # noqa: E402
from agent.sim.play import Player, dev_score          # noqa: E402

FREE = ["小电池包", "电池包", "新生力量", "战术模组", "5阶英雄", "如法炮制", "三人游", "河洛征召令"]


def one(st0, name, seed, ahead):
    s = st0.clone(keep_log=False)
    e = E.Engine(s, random.Random(seed))
    pl = Player(e, lookahead=0, samples=1, events=False)
    pl._take("talent", name) if name else pl.play_hand()
    end = s.round + ahead
    while s.round < end:
        pl.round()
    cu = s.unit(s.carry) if s.carry else None
    return dev_score(s), (s.carry and cu.name if cu else None), (cu.level if cu else 0), s.energy, len(s.heroes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=12); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--ahead", type=int, default=3); ap.add_argument("--names", default=",".join(FREE))
    a = ap.parse_args()
    E.set_battle_scenario("中")
    pts = [json.loads(l) for l in (P / "data/laya/datasets/r9zk/points.jsonl").open(encoding="utf-8")]
    tal = [p for p in pts if p["kind"] == "talent"]
    random.Random(3).shuffle(tal)
    rows = []
    for p in tal:
        try:
            rows.append(from_row(p))
        except Exception:                                  # noqa: BLE001
            continue
        if len(rows) >= a.states:
            break
    for name in a.names.split(","):
        ds, bad = [], []
        for i, st0 in enumerate(rows):
            for k in range(a.seeds):
                try:
                    x = one(st0, "", 1000 + 31 * i + k, a.ahead)
                    y = one(st0, name, 1000 + 31 * i + k, a.ahead)
                except Exception as ex:                    # noqa: BLE001
                    continue
                ds.append(y[0] - x[0])
                if y[0] < x[0] - 1e-6:
                    bad.append((round(y[0] - x[0], 1), x[1:], y[1:]))
        if ds:
            print(f"{name:8s} n={len(ds):3d} 平均 dev 差 {S.mean(ds):+6.2f}  变差 {len(bad)}/{len(ds)}")
            for b in sorted(bad)[:3]:
                print(f"      {b[0]:+.1f}  不拿(主C,级,能量,英雄数)={b[1]}  拿={b[2]}")


if __name__ == "__main__":
    main()
