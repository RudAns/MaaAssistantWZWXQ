"""感知层小模型的推理会话：优先 GPU（CUDA），失败才退回 CPU（限 2 线程）。

2026-10-12（RTX3060 余量较大，而占用 CPU 会影响这台电脑的其他工作）：
等级 / 觉醒进度 / 手牌觉醒 / 手牌条 / 手牌立绘 / 认人 / 界面分类原来都跑 CPU、且没限线程，单帧会把 24 核冲到 70~90%。
实测（batch 4）CUDA 比 CPU 2 线程快 5~12 倍：立绘 5.0→0.6ms、认人 12.0→1.0ms、界面分类 8.4→1.2ms、等级 2.6→0.5ms、觉醒 2.7→0.4ms。

用 CUDA 而不是 TensorRT：这些模型输入批量随场面变，TensorRT 每种尺寸要建引擎；CUDA 已经够快。
**加载后必须确认实际跑在 CUDA 上**：onnxruntime 加载失败会悄悄退回 CPU（2026-10-12 第一次测速就是这样被骗的）。
环境变量 WZWXQ_SMALL_EP=cpu 可强制回 CPU。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
_GPU_OK: bool | None = None
_DLLS = False


def session(path, name: str = ""):
    """返回 onnxruntime.InferenceSession；GPU 不可用时退回 CPU（2 线程）。"""
    global _GPU_OK
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    if os.environ.get("WZWXQ_SMALL_EP", "cuda").lower() != "cpu" and _GPU_OK is not False:
        try:
            global _DLLS
            if not _DLLS:
                from .onnx_ocr import _load_gpu_dlls
                _load_gpu_dlls(ort)                  # 只加载一次（每次加载 onnxruntime 都打一遍 CUDA 版本警告）
                _DLLS = True
            s = ort.InferenceSession(str(path), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            if s.get_providers()[0] == "CUDAExecutionProvider":
                _GPU_OK = True
                logger.info("小模型 %s 跑在 GPU（CUDA）", name or os.path.basename(str(path)))
                return s
            logger.warning("小模型 %s 的 CUDA 没加载上（实际 %s），退回 CPU", name or path, s.get_providers()[0])
            _GPU_OK = False
        except Exception as ex:                              # noqa: BLE001
            logger.warning("小模型 %s 上 GPU 失败（%s），退回 CPU", name or path, ex)
            _GPU_OK = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
