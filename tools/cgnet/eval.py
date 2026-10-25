"""cgnet 与 Laya 学生在同一测试集（r9m test，299 题）、同一口径（top1 / tmass）上对比，附 bootstrap 置信区间。

  python tools/cgnet/eval.py --runs v1b,abl_no_rel,abl_no_text,abl_no_tech --out data/cgnet/eval_v1.json

- 每个 cgnet run：多个种子逐题取平均（top1 为 0~1 之间的"种子平均命中率"），再报均值；
  另报种子间的均值 ± 标准差。
- Laya：R15（r9m 同一份标签，preds_test.jsonl 直接重算）；R14（它在 r9l 上评测，这里用 r9m 标签重算共同题目）。
- CI：对题目做 bootstrap（10000 次）；"差值"为配对 bootstrap（同一批题目）。
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
LAYA = {"R15": PROJECT_DIR / "data/laya/runs/R15_s1_top8/preds_test.jsonl",
        "R14": PROJECT_DIR / "data/laya/runs/R14_s1_top8/preds_test.jsonl"}


def item_scores(samples: dict, preds: list[dict], pkey: str = "p") -> dict:
    out = {}
    for pr in preds:
        s = samples.get(pr["id"])
        if s is None:
            continue
        pm = dict(zip(pr["keys"], pr[pkey]))
        keys = s["keys"]
        q = np.array([pm.get(k, 0.0) for k in keys])
        t = np.asarray(s["target"], np.float64)
        a = int(q.argmax())
        out[pr["id"]] = (float(keys[a] == s["t_best"]), float(t[a]))
    return out


def boot(x: np.ndarray, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="r9m")
    ap.add_argument("--runs", default="v1b")
    ap.add_argument("--ref", default="R15")
    ap.add_argument("--out", default=str(ROOT / "eval_v1.json"))
    ap.add_argument("--kinds", default="", help="只评这些题型（如 carry,swap；另报它们合并的 all）")
    a = ap.parse_args()
    samples = {s["id"]: s for s in pickle.loads((ROOT / a.ds / "samples.pkl").read_bytes()) if s["split"] == "test"}
    if a.kinds:
        samples = {i: s for i, s in samples.items() if s["kind"] in set(a.kinds.split(","))}
    ids = sorted(samples)
    kinds = sorted({s["kind"] for s in samples.values()})
    systems: dict[str, dict] = {}
    seed_stats: dict[str, dict] = {}
    for name, path in LAYA.items():
        preds = [json.loads(l) for l in path.open(encoding="utf-8")]
        systems[name] = item_scores(samples, preds, "model")
    runs = a.runs.split(",") + ([a.ref] if a.ref not in LAYA and a.ref not in a.runs.split(",") else [])
    for run in runs:
        per_seed = []
        for f in sorted(glob.glob(str(ROOT / "runs" / run / "seed*" / "preds_test.jsonl"))):
            preds = [json.loads(l) for l in open(f, encoding="utf-8")]
            per_seed.append(item_scores(samples, preds))
        if not per_seed:
            continue
        cov = [i for i in ids if all(i in ps for ps in per_seed)]          # 专项模型只覆盖部分题型
        systems[run] = {i: tuple(float(np.mean([ps[i][k] for ps in per_seed])) for k in (0, 1)) for i in cov}
        seed_stats[run] = {"n_seeds": len(per_seed), "n_items": len(cov),
                           "top1": [float(np.mean([ps[i][0] for i in cov])) for ps in per_seed],
                           "tmass": [float(np.mean([ps[i][1] for i in cov])) for ps in per_seed]}
    res = {"n_test": len(ids), "systems": {}, "diff_vs_ref": {}, "seed_stats": seed_stats}
    ref = systems[a.ref]
    for name, sc in systems.items():
        row = {}
        for g in ["all"] + kinds:
            gi = [i for i in ids if i in sc and (g == "all" or samples[i]["kind"] == g)]
            if not gi:
                continue
            t1 = np.array([sc[i][0] for i in gi])
            tm = np.array([sc[i][1] for i in gi])
            row[g] = {"n": len(gi), "top1": float(t1.mean()), "top1_ci": boot(t1), "tmass": float(tm.mean()),
                      "tmass_ci": boot(tm)}
        res["systems"][name] = row
        if name != a.ref:
            d = {}
            for g in ["all"] + kinds:
                gi = [i for i in ids if i in sc and i in ref and (g == "all" or samples[i]["kind"] == g)]
                if not gi:
                    continue
                dt = np.array([sc[i][0] - ref[i][0] for i in gi])
                dm = np.array([sc[i][1] - ref[i][1] for i in gi])
                d[g] = {"n": len(gi), "d_top1": float(dt.mean()), "d_top1_ci": boot(dt), "d_tmass": float(dm.mean()),
                        "d_tmass_ci": boot(dm)}
            res["diff_vs_ref"][name] = d
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # 打印
    print(f"test n={len(ids)}  (ref={a.ref})")
    hdr = "system".ljust(16) + "".join(f"{g}".rjust(16) for g in ["all"] + kinds)
    print(hdr)
    for name, row in res["systems"].items():
        print(name.ljust(16) + "".join((f"{row[g]['top1']:.3f}/{row[g]['tmass']:.3f}" if g in row else "-").rjust(16)
                                        for g in ["all"] + kinds))
    print("diff vs", a.ref, "(top1, 95% CI | tmass, 95% CI) on all:")
    for name, d in res["diff_vs_ref"].items():
        x = d["all"]
        print(f"  {name:16s} n={x['n']} dtop1={x['d_top1']:+.3f} [{x['d_top1_ci'][0]:+.3f},{x['d_top1_ci'][1]:+.3f}]"
              f"  dtmass={x['d_tmass']:+.3f} [{x['d_tmass_ci'][0]:+.3f},{x['d_tmass_ci'][1]:+.3f}]")
    for run, st in seed_stats.items():
        print(f"  seeds {run}: top1 {np.mean(st['top1']):.3f}±{np.std(st['top1']):.3f}  tmass {np.mean(st['tmass']):.3f}±{np.std(st['tmass']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
