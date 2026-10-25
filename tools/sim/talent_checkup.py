"""推演器天赋体检（2026-10-15，确认推演器是否真的起作用）：每张天赋在若干真实局面里"拿 / 不拿"各推一遍（同一随机种子、往后 3 回合），
比 dev_score 和主C 等级。所有局面里都完全相同 = 推演器里这张天赋不起作用（没实现 / 打法没用上它 / 条件在样本局面里都不满足）。
输出 data/sim/talent_checkup.json：{天赋: {"diff": 有差别的局面数, "n": 局面数, "mean_dev": 平均 dev 差, "impl": 有没有实现}}。

  .venv/Scripts/python.exe -X utf8 tools/sim/talent_checkup.py --states 6 --procs 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))


def _work(args):
    name, rows, ahead = args
    from agent.sim import engine as E
    from agent.sim.adapt import from_row
    from agent.sim.play import Player, dev_score
    from agent.sim.engine import impl
    E.set_battle_scenario("中")
    diff, n, ds = 0, 0, []
    for i, row in enumerate(rows):
        try:
            st0 = from_row(row)
        except Exception:                                           # noqa: BLE001
            continue
        res = []
        for take in (False, True):
            s = st0.clone(keep_log=False)
            e = E.Engine(s, random.Random(1000 + i))
            pl = Player(e, lookahead=0, samples=1, events=False)
            try:
                if take:
                    pl._take("talent", name)
                else:
                    pl.play_hand()                   # 对照组走同样的流程（_take 末尾会 play_hand），只差"拿不拿这张天赋"
            except Exception:                                       # noqa: BLE001
                break
            end = s.round + ahead
            try:
                while s.round < end:
                    pl.round()
            except Exception:                                       # noqa: BLE001
                break
            cu = s.unit(s.carry) if s.carry else None
            res.append((round(dev_score(s), 3), round(cu.level if cu else 0.0, 1)))
        if len(res) == 2:
            n += 1
            if res[0] != res[1]:
                diff += 1
            ds.append(res[1][0] - res[0][0])
    return name, {"diff": diff, "n": n, "mean_dev": round(sum(ds) / len(ds), 2) if ds else None,
                  "impl": impl(name) is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=6)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    pts = [json.loads(l) for l in (PROJECT_DIR / "data" / "laya" / "datasets" / "r9zk" / "points.jsonl").open(encoding="utf-8")]
    tal = [p for p in pts if p["kind"] == "talent"]
    from agent.sim.engine import card_db
    db = card_db()
    names = sorted({v.split(":", 1)[0].strip() for p in tal for k, v in p["question"]["criteria"].items() if k != "refresh"})
    names = [n for n in names if n in db]                     # OCR 读坏的名字不算
    rng = random.Random(7)
    # 局面样本：按回合分层（前期 / 中期 / 后期各取几个）
    by = {0: [], 1: [], 2: []}
    for p in tal:
        r = p.get("round") or 0
        by[0 if r <= 4 else 1 if r <= 9 else 2].append(p)
    rows = []
    for b in by.values():
        rng.shuffle(b)
        rows += b[: max(1, a.states // 3)]
    out = {}
    with Pool(a.procs) as pool:
        for name, r in pool.imap_unordered(_work, [(nm, rows, a.ahead) for nm in names]):
            out[name] = r
    dst = PROJECT_DIR / "data" / "sim" / "talent_checkup.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    noop = sorted(k for k, v in out.items() if v["n"] and v["diff"] == 0)
    noimpl = sorted(k for k, v in out.items() if not v["impl"])
    print(f"天赋 {len(out)} 张；推演里不起作用 {len(noop)} 张（其中没实现 {len([k for k in noop if k in noimpl])}）；有作用 {sum(1 for v in out.values() if v['diff'])} 张")
    print("不起作用（有实现）:", [k for k in noop if k not in noimpl][:60])
    print("没实现:", noimpl[:40])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
