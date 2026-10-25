"""线上 cgnet_policy 端到端延迟：CGNetPolicy.probs(题) = 特征构造 + collate + ONNX（多种子）。r9m 测试集 carry/swap 题 × 5 遍。

  python tools/cgnet/bench_policy.py [--deploy data/cgnet/deploy] [--seeds 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from agent.decision.cgnet_policy import CGNetModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default=str(PROJECT_DIR / "data/cgnet/deploy"))
    ap.add_argument("--seeds", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    pol = CGNetModel(Path(a.deploy), None, a.seeds or None)
    load_s = time.time() - t0
    pts = [json.loads(l) for l in (PROJECT_DIR / "data/laya/datasets/r9m/points.jsonl").open(encoding="utf-8")]
    pts = [p for p in pts if p["kind"] in pol.kinds and p["split"] == "test"]
    for p in pts[:5]:
        pol.probs(p, [])
    lat = []
    for _ in range(a.reps):
        for p in pts:
            t = time.perf_counter()
            pol.probs(p, [])
            lat.append((time.perf_counter() - t) * 1000)
    res = {"deploy": pol.manifest["run"], "version": pol.manifest["version"], "seeds_used": len(pol.sessions),
           "load_s": round(load_s, 2), "n": len(lat), "p50_ms": float(np.percentile(lat, 50)),
           "p90_ms": float(np.percentile(lat, 90)), "max_ms": float(np.max(lat))}
    print(json.dumps(res))
    out = PROJECT_DIR / "data/cgnet" / f"policy_latency_{pol.manifest['run']}_s{len(pol.sessions)}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
