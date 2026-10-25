"""实测：TensorRT OCR + Laya 同时在一张卡上跑，各占多少显存 / 内存。

分阶段记录：进程内存（RSS）、PyTorch 显存（精确，只含 Laya）、整卡显存（nvidia-smi，含别的进程，取多次中位数）。
  python tools/laya/bench_memory.py [--ckpt <laya checkpoint 目录>] [--no-ocr]
结果打印成 JSON，并写 data/laya/runs/bench_memory.json（报告"部署"一节用）。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))


def gpu_used_mib(n=5) -> float:
    vals = []
    for _ in range(n):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()[0]
        vals.append(float(out))
        time.sleep(0.2)
    return statistics.median(vals)


def rss_mib() -> float:
    """进程工作集（Windows，GetProcessMemoryInfo；不额外装 psutil）。"""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2**20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()
    rec = {"stages": []}

    def mark(name, **kw):
        import torch
        d = {"stage": name, "rss_mib": round(rss_mib()), "gpu_total_used_mib": gpu_used_mib()}
        if torch.cuda.is_available():
            d["torch_alloc_mib"] = round(torch.cuda.memory_allocated() / 2**20)
            d["torch_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            d["torch_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20)
        d.update(kw)
        rec["stages"].append(d)
        print(json.dumps(d, ensure_ascii=False), flush=True)

    import torch      # 先 import torch 再加载 onnxruntime 的 GPU 库（顺序反了会崩）
    mark("start")

    if not args.no_ocr:
        import cv2
        import numpy as np
        from agent.perception.onnx_ocr import OnnxOcr
        ocr = OnnxOcr.get()
        frame = next((PROJECT_DIR / "data" / "logs").glob("live_*/phase_prepare_*.png"))
        img = cv2.cvtColor(cv2.imdecode(np.fromfile(str(frame), np.uint8), 1), cv2.COLOR_BGR2RGB)
        t0 = time.time()
        for _ in range(3):
            ocr.boxes(img, (0, 0, 1280, 720))
            ocr.boxes(img, (200, 630, 1080, 90))
        mark("ocr_loaded_and_warm", ocr_backend=ocr.describe(), warm_s=round(time.time() - t0, 1))

    from train import BASE_CKPT, load_model, make_item
    from laya.common import collate_items
    device = torch.device("cuda")
    ckpt = Path(args.ckpt) if args.ckpt else BASE_CKPT
    model, tok, cfg = load_model(ckpt, device)
    model.eval()
    mark("laya_loaded_fp32", ckpt=str(ckpt))

    pts = [json.loads(l) for l in open(PROJECT_DIR / "data/laya/datasets/v3/points.jsonl", encoding="utf-8")]
    pts.sort(key=lambda p: -len(p["state"]))          # 最长的局面 = 最坏情况
    worst = [{**p, "teacher": {}} for p in pts[:4]]

    def run(batch_pts, reps=20):
        items = [make_item(tok, p, cfg) for p in batch_pts]
        b = collate_items([items], tok.pad_token_id)
        lat = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(reps):
                torch.cuda.synchronize(); t = time.perf_counter()
                model(*(b[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
                torch.cuda.synchronize(); lat.append((time.perf_counter() - t) * 1000)
        return round(statistics.median(lat[2:]), 1), int(b["input_ids"].shape[1])

    torch.cuda.reset_peak_memory_stats()
    l1, n1 = run(worst[:1])
    l4, n4 = run(worst)
    mark("laya_fp32_infer", latency_1q_ms=l1, latency_4q_ms=l4, seq_len=n4)

    model.half()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    l1h, _ = run(worst[:1])
    l4h, _ = run(worst)
    mark("laya_fp16_infer", latency_1q_ms=l1h, latency_4q_ms=l4h)

    out = PROJECT_DIR / "data" / "laya" / "runs" / "bench_memory.json"
    rec["note"] = "gpu_total_used_mib 是整卡读数，含同时在跑的其他进程（例如训练），只看相邻阶段的差值"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
