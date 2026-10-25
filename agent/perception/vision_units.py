"""己方场上单位检测（D-FINE-M 微调）：画面 → 哪些格子站着我方单位。

用途：补充"格子占用"观测。旧方法（等级徽章霍夫圆）400 帧人工标注上格子召回 89.5%、C1 几乎全漏；
D-FINE 测试集格子召回 100%、整帧正确 92.6%（旧 61.1%）。
**只用来"加占用"**（铁律：观测只能加、不能删账本记录），等级数字仍由徽章 + OCR 读。

模型：assets/model/vision/board_m_v1_640.onnx（tools/vision/export_board.py 导出，固定 640x640、含后处理）。
推理引擎：`WZWXQ_VISION_EP`（默认 auto：先 TensorRT，失败退回 onnxruntime CPU）；`WZWXQ_VISION=0` 关闭整个模块。
格子映射：框底中心上移 15px ≈ 格心（训练框的定义，tools/vision/build_coco.py），取最近格、距离 <60px 才算。
阈值 0.3：只在验证集上选的（整帧正确率最高）。
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
_VDIR = Path(__file__).resolve().parents[2] / "assets" / "model" / "vision"
# 默认模型：**grid_sample 换成 gather 双线性**、不带后处理（后处理在下面用 numpy 做）。
# 原版在 TensorRT 下"编译成功但结果错"（部署对比）：测试集格子召回 FP16 85.3% / FP32 72.8%；
# gather 版 FP16/FP32 均与 PyTorch 一致（100%、整帧 92.6%），FP16 p50 11.4ms。
MODEL = _VDIR / os.environ.get("WZWXQ_VISION_MODEL", "board_m_v2_640_gather_raw.onnx")
SIZE = 640
# 阈值只在验证集上选：v1 0.3（按目录切分）；v2 0.4（按对局切分，best_stg2，验证集整帧 0.889，0.4/0.5 并列取低）
THRESH = float(os.environ.get("WZWXQ_VISION_THRESH", "0.4"))
BOX_DOWN = 15                  # 与 tools/vision/build_coco.py 一致
MAX_CELL_DIST = 60
CLASSES = ("hero", "summon")

_lock = threading.Lock()
_sess = None
_failed = False


def enabled() -> bool:
    return os.environ.get("WZWXQ_VISION", "1") != "0" and MODEL.exists() and not _failed


def _session():
    """懒加载；TensorRT 引擎首次构建要几分钟（缓存在 .trt_cache/vision）。"""
    global _sess, _failed
    with _lock:
        if _sess is not None or _failed:
            return _sess
        try:
            import onnxruntime as ort
            from .onnx_ocr import TRT_CACHE, _load_gpu_dlls
            ep = os.environ.get("WZWXQ_VISION_EP", "auto").lower()
            so = ort.SessionOptions()
            so.log_severity_level = 3
            order = ["tensorrt", "cpu"] if ep == "auto" else [ep]
            for e in order:
                try:
                    if e == "tensorrt":
                        _load_gpu_dlls(ort)
                        # FP16（默认）：gather 版在 FP16 下与 PyTorch 一致；WZWXQ_VISION_FP16=0 改 FP32（17.7ms）。
                        fp16 = os.environ.get("WZWXQ_VISION_FP16", "1") == "1"
                        cache = TRT_CACHE / f"vision_{MODEL.stem}_{'fp16' if fp16 else 'fp32'}"
                        cache.mkdir(parents=True, exist_ok=True)
                        prov = [("TensorrtExecutionProvider", {"trt_fp16_enable": fp16,
                                                               "trt_engine_cache_enable": True,
                                                               "trt_engine_cache_path": str(cache),
                                                               "trt_timing_cache_enable": True}),
                                "CUDAExecutionProvider", "CPUExecutionProvider"]
                        want = "TensorrtExecutionProvider"
                    elif e == "cuda":
                        _load_gpu_dlls(ort)
                        prov, want = ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"
                    else:
                        prov, want = ["CPUExecutionProvider"], "CPUExecutionProvider"
                    s = ort.InferenceSession(str(MODEL), so, providers=prov)
                    if s.get_providers()[0] != want:      # TensorRT 失败会悄悄退回 CUDA（onnx_ocr 里踩过）
                        raise RuntimeError(f"{e} 没加载成功，退回了 {s.get_providers()[0]}")
                    _sess = s
                    logger.info("视觉单位检测已加载：%s（%s）", MODEL.name, want)
                    break
                except Exception as ex:  # noqa: BLE001
                    logger.warning("视觉单位检测 %s 加载失败：%s", e, ex)
            if _sess is None:
                _failed = True
        except Exception as ex:  # noqa: BLE001
            logger.warning("视觉单位检测不可用：%s", ex)
            _failed = True
        return _sess


def detect(image_rgb: np.ndarray, thresh: float = THRESH) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    """-> [(类别, 分数, (x0,y0,x1,y1) 原图坐标)]。image 是 RGB（与 extractors 一致）。"""
    import cv2
    s = _session() if enabled() else None
    if s is None or image_rgb is None:
        return []
    h, w = image_rgb.shape[:2]
    x = cv2.resize(image_rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    if len(s.get_inputs()) == 1:
        # 不带后处理的导出：pred_logits(1,Q,C) 过 sigmoid（focal loss）、pred_boxes(1,Q,4) 是归一化 cxcywh
        logits, cxcywh = s.run(None, {"images": x})
        prob = 1.0 / (1.0 + np.exp(-logits[0]))
        out = []
        for q, c in zip(*np.nonzero(prob >= thresh)):
            cx, cy, bw, bh = cxcywh[0, q]
            out.append((int(c), float(prob[q, c]), ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
        return out
    labels, boxes, scores = s.run(None, {"images": x, "orig_target_sizes": np.array([[w, h]], np.int64)})
    return [(int(l), float(sc), tuple(map(float, b))) for l, b, sc in zip(labels[0], boxes[0], scores[0]) if sc >= thresh]


def cell_units(image_rgb: np.ndarray, cells, thresh: float = THRESH) -> dict[tuple[int, int], tuple[str, float, tuple]]:
    """-> {格心坐标: (类别 'hero'/'summon', 分数, 框 x0y0x1y1)}；同一格多个框取分数最高的。"""
    best: dict[tuple[int, int], tuple[str, float, tuple]] = {}
    for lab, sc, box in detect(image_rgb, thresh):
        x0, y0, x1, y1 = box
        fx, fy = (x0 + x1) / 2, y1 - BOX_DOWN
        c = min(cells, key=lambda c: (c[0] - fx) ** 2 + (c[1] - fy) ** 2)
        if ((c[0] - fx) ** 2 + (c[1] - fy) ** 2) ** .5 < MAX_CELL_DIST and sc > best.get(tuple(c), ("", 0, ()))[1]:
            best[tuple(c)] = (CLASSES[lab] if 0 <= lab < len(CLASSES) else "hero", sc, box)
    return best


def occupied_cells(image_rgb: np.ndarray, cells, thresh: float = THRESH) -> dict[tuple[int, int], str]:
    """-> {格心坐标: 'hero'/'summon'}。cells = 己方格心坐标列表（extractors.OWN_BOARD_ROWS 展平）。"""
    return {c: v[0] for c, v in cell_units(image_rgb, cells, thresh).items()}


def warmup() -> Optional[str]:
    """进程启动时预热（建 TensorRT 引擎），返回实际用的引擎名。"""
    s = _session() if enabled() else None
    if s is None:
        return None
    detect(np.zeros((720, 1280, 3), np.uint8))
    return s.get_providers()[0]
