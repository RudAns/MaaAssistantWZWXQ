"""cgnet 导出 ONNX + onnxruntime CPU 单题延迟（p50/p90/max）+ 与 PyTorch 输出一致性检查。

  python tools/cgnet/export_onnx.py --run v1b --seed 1

产物：data/cgnet/runs/<run>/seed<s>/cgnet.onnx、latency.json
注：导出用 TorchScript 路径（dynamo=False），本环境没装 onnx / onnxscript 包（不装新库）。
延迟口径：单题（batch=1），输入为该题真实的对象数 N / 选项数 K（动态轴），CPU 单线程与默认线程各测一次；
计时只含 session.run（特征张量已准备好）；另测"从样本 dict 组张量 + run"的端到端时间。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
from model import CGNet  # noqa: E402
from train import collate  # noqa: E402

ROOT = PROJECT_DIR / "data" / "cgnet"
NAMES = ["tt", "ot", "node", "text", "num", "glob", "mask", "rel", "opt", "optm"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1b")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=299, help="测多少道题")
    a = ap.parse_args()
    d = ROOT / "runs" / a.run / f"seed{a.seed}"
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    meta = json.loads((ROOT / cfg["ds"] / "meta.json").read_text(encoding="utf-8"))
    tt = np.load(ROOT / cfg["ds"] / f"text_table_{cfg['enc']}.npy")
    nf = np.load(ROOT / cfg["ds"] / "node_feat.npy")
    model = CGNet(torch.from_numpy(tt), torch.from_numpy(nf), len(meta["tok_types"]), len(meta["opt_types"]),
                  meta["num_dim"], meta["glob_dim"], meta["n_channels"], d=cfg["d"], heads=cfg["heads"],
                  layers=cfg["layers"], ff=cfg["ff"], drop=0.0, use_text=cfg["ablate"] != "no_text",
                  use_rel=cfg["ablate"] != "no_rel", use_node_id=cfg["ablate"] != "no_nodeid")
    model.load_state_dict(ck["state"])
    model.eval()
    samples = [s for s in pickle.loads((ROOT / cfg["ds"] / "samples.pkl").read_bytes()) if s["split"] == "test"][:a.n]
    x0, _ = collate([samples[0]], meta["n_channels"], torch.device("cpu"))
    onnx_path = d / "cgnet.onnx"
    dyn = {"tt": {1: "N"}, "ot": {1: "N"}, "node": {1: "N"}, "text": {1: "N"}, "num": {1: "N"}, "mask": {1: "N"},
           "rel": {1: "N", 2: "N2"}, "opt": {1: "K"}, "optm": {1: "K"}, "logits": {1: "K"}}
    torch.onnx.export(model, tuple(x0[k] for k in NAMES), str(onnx_path), input_names=NAMES, output_names=["logits"],
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    import onnxruntime as ort
    res = {"run": a.run, "seed": a.seed, "onnx_bytes": onnx_path.stat().st_size, "n_items": len(samples)}
    maxdiff, agree = 0.0, 0
    for threads in (1, 0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        feeds = []
        for s in samples:
            x, _ = collate([s], meta["n_channels"], torch.device("cpu"))
            feeds.append({k: x[k].numpy() for k in NAMES})
        for f in feeds[:20]:
            sess.run(None, f)                      # 预热
        lat = []
        for rep in range(3):
            for f in feeds:
                t0 = time.perf_counter()
                out = sess.run(None, f)[0]
                lat.append((time.perf_counter() - t0) * 1000)
        e2e = []
        for s in samples:
            t0 = time.perf_counter()
            x, _ = collate([s], meta["n_channels"], torch.device("cpu"))
            sess.run(None, {k: x[k].numpy() for k in NAMES})
            e2e.append((time.perf_counter() - t0) * 1000)
        if threads == 1:
            with torch.no_grad():
                for s, f in zip(samples, feeds):
                    x, _ = collate([s], meta["n_channels"], torch.device("cpu"))
                    pt = model(**x).numpy()[0]
                    ox = sess.run(None, f)[0][0]
                    maxdiff = max(maxdiff, float(np.abs(pt - ox).max()))
                    agree += int(pt.argmax() == ox.argmax())
        key = "threads1" if threads else "threads_default"
        res[key] = {"p50_ms": float(np.percentile(lat, 50)), "p90_ms": float(np.percentile(lat, 90)),
                    "max_ms": float(np.max(lat)), "e2e_p50_ms": float(np.percentile(e2e, 50)),
                    "e2e_p90_ms": float(np.percentile(e2e, 90)), "e2e_max_ms": float(np.max(e2e))}
    res["torch_vs_onnx_max_abs_logit_diff"] = maxdiff
    res["argmax_agree"] = f"{agree}/{len(samples)}"
    res["objects_N"] = {"p50": float(np.median([len(s['tt']) for s in samples])), "max": int(max(len(s['tt']) for s in samples))}
    (d / "latency.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
