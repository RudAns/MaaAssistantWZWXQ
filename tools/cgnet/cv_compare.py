"""比较 cv_cs.py 的各配置（同一批留出题，配对 bootstrap 10000 次）。

  python tools/cgnet/cv_compare.py --ref cv_v1 --runs cv_v2,cv_v2_notechdyn,cv_v2_notech,cv_v2_nocond,cv_v2_nofacts,cv_v1_multi,cv_v2_multi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data" / "cgnet" / "runs"


def boot(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="cv_v1")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=str(ROOT.parent / "cv_compare.json"))
    a = ap.parse_args()
    load = lambda r: json.loads((ROOT / r / "cv_items.json").read_text(encoding="utf-8"))  # noqa: E731
    ref = load(a.ref)
    res = {"ref": a.ref, "rows": {}}
    names = [a.ref] + a.runs.split(",")
    for r in names:
        it = load(r)
        row = {}
        for sub in ("all", "real"):
            for kind in ("both", "carry", "swap"):
                ids = [i for i, v in it.items() if i in ref and (sub == "all" or not v["aug"])
                       and (kind == "both" or v["kind"] == kind)]
                t1 = np.array([it[i]["top1"] for i in ids])
                tm = np.array([it[i]["tmass"] for i in ids])
                d1 = np.array([it[i]["top1"] - ref[i]["top1"] for i in ids])
                dm = np.array([it[i]["tmass"] - ref[i]["tmass"] for i in ids])
                row[f"{sub}/{kind}"] = {"n": len(ids), "top1": float(t1.mean()), "tmass": float(tm.mean()),
                                        "top1_ci": boot(t1), "d_top1": float(d1.mean()), "d_top1_ci": boot(d1),
                                        "d_tmass": float(dm.mean()), "d_tmass_ci": boot(dm)}
        res["rows"][r] = row
        x = row["all/both"]
        y = row["real/both"]
        print(f"{r:18s} all n={x['n']} top1={x['top1']:.3f} tmass={x['tmass']:.3f} d={x['d_top1']:+.3f} "
              f"[{x['d_top1_ci'][0]:+.3f},{x['d_top1_ci'][1]:+.3f}] dtm={x['d_tmass']:+.3f} [{x['d_tmass_ci'][0]:+.3f},{x['d_tmass_ci'][1]:+.3f}]"
              f" | real n={y['n']} top1={y['top1']:.3f} d={y['d_top1']:+.3f} [{y['d_top1_ci'][0]:+.3f},{y['d_top1_ci'][1]:+.3f}]"
              f" | carry {row['all/carry']['top1']:.3f} swap {row['all/swap']['top1']:.3f}")
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
