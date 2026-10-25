"""D-FINE ONNX 在 onnxruntime + TensorRT FP16 上的延迟（主环境 .venv，和线上 OCR 同一套运行时）。
  .venv/Scripts/python.exe tools/vision_bench_trt.py assets/model/vision/dfine_m_coco_640x640.onnx ...
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: F401  先 import torch 再加载 onnxruntime 的 GPU 库
import numpy as np
import onnxruntime as ort
from agent.perception.onnx_ocr import _load_gpu_dlls

_load_gpu_dlls(ort)
cache = ROOT / ".trt_cache" / "vision"
cache.mkdir(parents=True, exist_ok=True)
for f in sys.argv[1:]:
    t0 = time.time()
    s = ort.InferenceSession(f, providers=[("TensorrtExecutionProvider", {
        "trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
        "trt_timing_cache_enable": True}), "CUDAExecutionProvider"])
    w, h = (int(v) for v in Path(f).stem.split("_")[-1].split("x"))
    x = np.random.rand(1, 3, h, w).astype(np.float32)
    sz = np.array([[w, h]], dtype=np.int64)
    s.run(None, {"images": x, "orig_target_sizes": sz})
    print(f"{Path(f).name}: 建引擎/加载 {time.time()-t0:.0f}s, provider={s.get_providers()[0]}")
    for _ in range(20):
        s.run(None, {"images": x, "orig_target_sizes": sz})
    t = time.perf_counter(); n = 100
    for _ in range(n):
        s.run(None, {"images": x, "orig_target_sizes": sz})
    print(f"  {w}x{h} TensorRT FP16: {1000*(time.perf_counter()-t)/n:.2f} ms/帧（含 CPU<->GPU 拷贝）")
