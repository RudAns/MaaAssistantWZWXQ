"""评测**线上模块本身**（agent/perception/vision_units.py，onnxruntime + TensorRT/CPU），确认部署后指标没掉；并测延迟。

  .venv/Scripts/python.exe tools/vision/eval_deployed.py --split test

与 tools/vision/eval_cells.py（训练环境 PyTorch）不同之处：预处理用 cv2 线性插值、FP16 TensorRT。两者结果应接近；
差异大说明部署链路有问题。测试集此前已按同一模型/阈值评过一次（非新增调参），本脚本不改任何超参。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.perception import vision_units  # noqa: E402
from agent.perception.extractors import OWN_BOARD_ROWS  # noqa: E402
from tools.vision.eval_cells import bootstrap, score  # noqa: E402

ROWS = "ABCD"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--frames", default="frames.jsonl", help="帧清单；v2 = frames_v2split.jsonl")
    ap.add_argument("--coco", default="coco_v1")
    args = ap.parse_args()
    ep = vision_units.warmup()
    name = {tuple(c): f"{ROWS[r]}{i + 1}" for r, row in enumerate(OWN_BOARD_ROWS) for i, c in enumerate(row)}
    cells = list(name)
    pool = ROOT / "data/vision/pool_v1"
    frames = {json.loads(l)["id"]: json.loads(l) for l in (pool / args.frames).open(encoding="utf-8")}
    labels = {}
    for fn in ("labels_train.jsonl", "labels_evalset.jsonl"):
        for l in (pool / fn).open(encoding="utf-8"):
            r = json.loads(l); labels[r["id"]] = r
    pairs, runs, ms = [], [], []
    for fid, f in sorted(frames.items()):
        if f["split"] != args.split or fid not in labels:
            continue
        im = cv2.cvtColor(cv2.imdecode(np.fromfile(str(ROOT / f["frame"]), np.uint8), 1), cv2.COLOR_BGR2RGB)
        t = time.perf_counter()
        occ = vision_units.occupied_cells(im, cells)
        ms.append((time.perf_counter() - t) * 1000)
        pairs.append((set(labels[fid]["occupied"]), {name[c] for c in occ}))
        runs.append(f.get("match", f["run"]))
    res = {"split": args.split, "engine": ep, "deployed": score(pairs), "ci95_by_run": bootstrap(pairs, runs),
           "latency_ms_p50": round(float(np.median(ms)), 1), "latency_ms_p90": round(float(np.percentile(ms, 90)), 1)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    with (ROOT / "data/vision/runs/eval_cells.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(res, ckpt="deployed:" + vision_units.MODEL.name), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
