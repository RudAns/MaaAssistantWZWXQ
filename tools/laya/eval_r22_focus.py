"""R22 专项评估（2026-10-10）：只看受 f9lf 新事实影响的测试题（datasets/r9u/relabel.json 的 A/B/C 三类），R17 vs R22。

  .venv/Scripts/python.exe -X utf8 tools/laya/eval_r22_focus.py [--r22 data/laya/runs/R22_s1_top8/checkpoint]

- A 刷新后买不起英雄：学生给 refresh 的平均概率（越低越对）、老师给的平均概率；
- B 发英雄牌 / C 专属天赋：按类 top1 / tmass。
R17 读 r9u_f9l（f9l 文本），R22 读 r9u（f9lf 文本），标签 labels/r9u_all（g9r）。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "laya"))
import torch  # noqa: E402
import train as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r17", default="data/laya/runs/R17_s1_top8/checkpoint")
    ap.add_argument("--r22", default="data/laya/runs/R22_s1_top8/checkpoint")
    ap.add_argument("--labels", default="data/laya/labels/r9u_all/labels_all.jsonl")
    ap.add_argument("--splits", default="test_r9u_new,test_r9s_new,test_r9r_new,test_r9q_new")
    args = ap.parse_args()
    relabel = json.loads((ROOT / "data/laya/datasets/r9u/relabel.json").read_text(encoding="utf-8"))
    splits = set(args.splits.split(","))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    for name, ck, ver in (("R17", args.r17, "r9u_f9l"), ("R22", args.r22, "r9u")):
        rows = [r for r in T.load_data(ver, ROOT / args.labels) if r["split"] in splits and r["id"] in relabel]
        model, tok, cfg = T.load_model(Path(ck), dev)
        temp = (cfg.get("temperature") or [1.0])[0]
        preds = T.predict(model, tok, cfg, rows, dev, temp=temp)
        by = {r["id"]: r for r in rows}
        out = {}
        for p in preds:
            r = by[p["id"]]
            keys, q = p["keys"], np.array(p["p"])
            t = np.array([r["teacher"].get(k, 0.0) for k in keys])
            t = t / t.sum()
            a = int(q.argmax())
            out[p["id"]] = {"top1": float(keys[a] == r["t_best"]), "tmass": float(t[a]),
                            "p_refresh": float(q[keys.index("refresh")]) if "refresh" in keys else None,
                            "t_refresh": float(t[keys.index("refresh")]) if "refresh" in keys else None}
        res[name] = out
        del model
        torch.cuda.empty_cache()
    ids = sorted(set(res["R17"]) & set(res["R22"]))
    groups = collections.defaultdict(list)
    for i in ids:
        for t in relabel[i]:
            groups[t].append(i)
    rng = np.random.default_rng(0)
    print(f"受新事实影响的测试题：{len(ids)}（{', '.join(sorted(splits))}）")
    for t, g in sorted(groups.items()):
        line = f"  {t:24s} n={len(g):4d}"
        for nm in ("R17", "R22"):
            line += f" | {nm} top1 {np.mean([res[nm][i]['top1'] for i in g]):.3f} tmass {np.mean([res[nm][i]['tmass'] for i in g]):.3f}"
        d = np.array([res["R22"][i]["tmass"] - res["R17"][i]["tmass"] for i in g])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        line += f" | Δtmass {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(line)
        if t == "A_refresh_cant_afford":
            pr = [i for i in g if res["R22"][i]["p_refresh"] is not None]
            if pr:
                print(f"      P(refresh)：老师 {np.mean([res['R22'][i]['t_refresh'] for i in pr]):.3f}  "
                      f"R17 {np.mean([res['R17'][i]['p_refresh'] for i in pr]):.3f}  R22 {np.mean([res['R22'][i]['p_refresh'] for i in pr]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
