"""cgnet 特征构造耗时：从一道题的结构化字段（ctx/board/raw）构造对象表 + 关系（Builder.convert），单题 p50/p90/max。

  python tools/cgnet/bench_convert.py --ds r9m
（线上真正部署时，这一步 + ONNX 推理 = 端到端；日志回放那部分线上不需要，已有实时状态。）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
import build_data as BD  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="r9m")
    ap.add_argument("--v2", action="store_true", help="再加 build_data_v2.V2.augment（汇总事实 / 手法绑定 / 条件边）")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--cold", action="store_true", help="每题清空 V2 的谓词缓存（线上最坏口径）")
    a = ap.parse_args()
    G = BD.Graph()
    G.path = str(PROJECT_DIR / "data" / "card_graph.json")
    B = BD.Builder(G, None)
    V = None
    if a.v2:
        import build_data_v2 as BD2  # noqa: E402
        V = BD2.V2(G)
        meta1 = {"tok_types": BD.TOK_TYPES, "n_channels": 2 * len(BD.EDGE_TYPES) + len(BD.EXTRA_CH)}
    pts = [json.loads(l) for l in (PROJECT_DIR / "data/laya/datasets" / a.ds / "points.jsonl").open(encoding="utf-8")]
    pts = [p for p in pts if p["split"] == "test" and (not a.kinds or p["kind"] in a.kinds.split(","))]
    lab = {"dist": {}, "best": ""}
    for p in pts[:20]:
        B.convert(p, lab, [], None)
    lat = []
    for _ in range(3):
        for p in pts:
            t0 = time.perf_counter()
            sm = B.convert(p, lab, [], None)
            if V is not None:
                if a.cold:
                    V.__dict__.pop("_pm", None)          # 冷启动口径：每题都清空谓词缓存
                V.augment(sm, p, meta1)
            lat.append((time.perf_counter() - t0) * 1000)
    res = {"n": len(pts), "p50_ms": float(np.percentile(lat, 50)), "p90_ms": float(np.percentile(lat, 90)),
           "max_ms": float(np.max(lat))}
    res.update({"v2": a.v2, "kinds": a.kinds, "cold": a.cold})
    print(json.dumps(res))
    (PROJECT_DIR / "data/cgnet" / f"convert_latency{'_v2' if a.v2 else ''}{'_cold' if a.cold else ''}{'_' + a.kinds.replace(',', '_') if a.kinds else ''}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
