"""场上单位认人（2026-09-30：规则判据一直认错人，改用图像识别）。

D-FINE（vision_units.cell_units）只回答"这一格有没有英雄"，这里回答"是谁"。
模型：assets/model/vision/unit_id_v1.onnx（tools/vision/train_unit_id.py）：
  97 局 15097 样本、61 类 + 「其他」；棋盘 3D 模型不随等级变；测试集门槛 0.5 下已知英雄覆盖 98.3%、采纳准确率 99.94%。
2026-10-13 换成 unit_id_v2（tools/vision/train_unit_id_v2.py）：
  656 局、83 个英雄 + 「其他」（新增 23 类：瑶/狂铁/敖隐/甄姬/马超/露娜/韩信/鲁班大师…）；测试 17,805 个，
  门槛 0.5 覆盖 91.3% → 99.2%、采纳准确 99.84% → 99.86%。输入裁法与 v1 完全相同。
  回滚：把下面 MODEL / META 改回 unit_id_v1.onnx / data/vision/unit_id_v1/classes.json（文件都保留着）。
输入：按 D-FINE 框裁（左右各扩 6px、去掉框顶 15% 的等级徽章、底部 +4px）→ 缩放 80x128 → 中心裁 72x120 → RGB /255 → (N,3,120,72)。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "assets" / "model" / "vision" / "unit_id_v2.onnx"
META = ROOT / "data" / "vision" / "unit_id_v2" / "classes.json"
CROP_W, CROP_H = 80, 128
IN_W, IN_H = 72, 120
BADGE_CUT = 0.15

_sess = None
_classes: list = []
_thresh = 0.5
_other = "其他"




def _gpu_session(path, name):
    from .gpu_session import session
    return session(path, name)

def _cpu_so():
    """CPU 上的小模型一律限 2 线程（2026-10-12：没设时 onnxruntime 默认开满 24 核，手牌 / 场上一多就把整机冲到 70~90%）。"""
    import onnxruntime as _ort
    so = _ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return so

def _load() -> bool:
    global _sess, _classes, _thresh, _other
    if _sess is not None:
        return bool(_sess)
    try:
        import onnxruntime as ort
        meta = json.loads(META.read_text(encoding="utf-8"))
        _classes = list(meta["classes"])
        _thresh = float(meta.get("thresh", 0.5))
        _other = meta.get("other", "其他")
        _sess = _gpu_session(MODEL, "认人")
        n_out = _sess.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(_classes):      # 模型和类表不配套 = 名字全错位，宁可不用
            raise RuntimeError(f"认人模型输出 {n_out} 类，类表 {len(_classes)} 类，不配套")
    except Exception:                                           # noqa: BLE001
        logger.exception("认人模型加载失败（忽略，只靠账本 + 点开看）")
        _sess = False
    return bool(_sess)


def _crop(image_rgb: np.ndarray, box) -> np.ndarray:
    import cv2
    x0, y0, x1, y1 = box
    h = y1 - y0
    X0, X1 = int(round(x0 - 6)), int(round(x1 + 6))
    Y0, Y1 = int(round(y0 + BADGE_CUT * h)), int(round(y1 + 4))
    pad = cv2.copyMakeBorder(image_rgb, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    c = pad[Y0 + 64:Y1 + 64, X0 + 64:X1 + 64]
    c = cv2.resize(c, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    oy, ox = (CROP_H - IN_H) // 2, (CROP_W - IN_W) // 2
    return c[oy:oy + IN_H, ox:ox + IN_W]


def identify(image_rgb: np.ndarray, boxes: dict) -> dict:
    """boxes: {格子key: (x0,y0,x1,y1)} → {格子key: [名字, 置信度]}，只收 ≥ 门槛、且不是「其他」的。"""
    if not boxes or not _load():
        return {}
    keys = list(boxes)
    x = np.stack([_crop(image_rgb, boxes[k]) for k in keys]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    logits = _sess.run(None, {_sess.get_inputs()[0].name: np.ascontiguousarray(x)})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    out = {}
    for i, k in enumerate(keys):
        j = int(p[i].argmax())
        name, conf = _classes[j], float(p[i, j])
        if conf >= _thresh and name != _other:
            out[k] = [name, round(conf, 3)]
    return out


def warmup() -> Optional[str]:
    if not _load():
        return None
    return "gpu" if _sess.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
