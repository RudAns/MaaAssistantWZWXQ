"""列出 cgnet 答错（种子平均概率的 argmax ≠ 老师 best）的测试题，附老师分布 / 模型分布 / R15 的选择 / 老师理由 / 题面选项文本，
供人工做失败分析。

  python tools/cgnet/failures.py --run v1b --n 40 > 输出
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT = PROJECT_DIR / "data" / "cgnet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1b")
    ap.add_argument("--ds", default="r9m")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--kind", default="")
    a = ap.parse_args()
    S = {s["id"]: s for s in pickle.loads((ROOT / a.ds / "samples.pkl").read_bytes()) if s["split"] == "test"}
    P = {}
    for l in (PROJECT_DIR / "data/laya/datasets" / a.ds / "points.jsonl").open(encoding="utf-8"):
        d = json.loads(l)
        if d["id"] in S:
            P[d["id"]] = d
    labs = {}
    for l in (PROJECT_DIR / "data/laya/labels/r6" / f"labels_all_{a.ds}.jsonl").open(encoding="utf-8"):
        d = json.loads(l)
        if d["id"] in S:
            labs[d["id"]] = d
    L = {json.loads(l)["id"]: json.loads(l) for l in (PROJECT_DIR / "data/laya/runs/R15_s1_top8/preds_test.jsonl").open(encoding="utf-8")}
    seeds = []
    for f in sorted(glob.glob(str(ROOT / "runs" / a.run / "seed*" / "preds_test.jsonl"))):
        seeds.append({json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")})
    rows = []
    for i, s in S.items():
        if a.kind and s["kind"] != a.kind:
            continue
        p = np.mean([np.array([dict(zip(sd[i]["keys"], sd[i]["p"]))[k] for k in s["keys"]]) for sd in seeds], axis=0)
        a_ = s["keys"][int(np.argmax(p))]
        if a_ == s["t_best"]:
            continue
        t = dict(zip(s["keys"], s["target"]))
        lp = dict(zip(L[i]["keys"], L[i]["model"])) if i in L else {}
        rows.append((t[s["t_best"]] - t[a_], i, s, p, a_, lp))
    rows.sort(key=lambda r: -r[0])
    print(f"{len(rows)} wrong of {len(S)}")
    for gap, i, s, p, a_, lp in rows[:a.n]:
        pt = P[i]
        crit = pt["question"]["criteria"]
        lb = max(lp, key=lp.get) if lp else "-"
        print("=" * 100)
        print(f"id={i} kind={s['kind']} round={s['round']} run={pt.get('run')}  teacher_best={s['t_best']}  cgnet={a_}  R15={lb}  "
              f"teacher-mass gap={gap:.2f}")
        ctx = pt.get("ctx") or {}
        print("ctx:", {k: ctx.get(k) for k in ("round", "gold", "level", "level_cost", "pop", "pop_cap", "carry", "maple",
                                                "hand", "banned", "free_refresh", "auction_in")})
        print("board:", pt.get("board") or pt.get("field"))
        for k in s["keys"]:
            print(f"  {k:8s} T={dict(zip(s['keys'], s['target']))[k]:.2f} C={p[s['keys'].index(k)]:.2f} L={lp.get(k, 0):.2f} | {crit.get(k, '')[:150]}")
        print("teacher reason:", labs[i].get("reason", "")[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
