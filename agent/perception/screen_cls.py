"""界面分类（2026-09-30：OCR 准确度太低，改为训练专用模型）。

整屏截图 -> 当前是哪种界面，用来替代 / 核对 extractors.detect_phase 的 OCR 关键字判据。
模型：assets/model/vision/screen_cls_v1.onnx（tools/vision/train_screen_cls.py）。
类别表与门槛：data/vision/screen_cls_v1/classes.json。
输入：整屏 RGB（MaaFW 截图 1280x720）-> INTER_AREA 缩到 256x144 -> /255 -> "x" (1,3,144,256) float32，不做均值方差归一化；
  模型若有第二个输入 "c"：三块固定区域的高分辨率裁图拼成 64x224（CROP_REGIONS，与 tools/vision/build_screen_cls_crops.py 一致）。

与 detect_phase 取值的对应（classes.json 的 "phase_map"，to_phase()）：
  prepare / prepare_shop / hero_panel / equip_popup -> "prepare"（后两种是盖在准备阶段上的面板，接线时按细类处理关闭动作）；
  reward_popup（拍到拍品后的展示）-> "auction"；
  board_idle / transition / post_match / enlarged_detail / solo_popup / notice / other -> None（旧体系没有这个阶段：
  等下一帧，或退回旧判据）。注意 board_idle 与 battle 单帧分不清（报告第 4 节），两者线上都应"什么都不做"。
  notice（公告面板）训练数据里一张都没有，模型不会输出它；settings 只有 3 张训练样本，不可信。

用法：
    from agent.perception import screen_cls
    r = screen_cls.classify(image_rgb)        # -> ("card_pick", 0.97) ；加载失败返回 None
    if r and r[1] >= screen_cls.threshold(): ...   # 低于门槛 -> 退回旧 OCR 判据
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "assets" / "model" / "vision" / "screen_cls_v1.onnx"
META = ROOT / "data" / "vision" / "screen_cls_v1" / "classes.json"
IN_W, IN_H = 256, 144
# 与 tools/vision/build_screen_cls_crops.py 的 REGIONS 一致：右上角（战斗统计）/ 右下角（准备·开战·出价）/ 顶部横幅
CROP_REGIONS = [((1070, 25, 1280, 250), (64, 64)), ((1120, 490, 1280, 600), (96, 64)), ((430, 0, 850, 45), (64, 64))]

_sess = None
_classes: list = []
_thresh = 0.9
_phase_map: dict = {}



def _gpu_session(path, name):
    from .gpu_session import session
    return session(path, name)

def _load() -> bool:
    global _sess, _classes, _thresh, _phase_map
    if _sess is not None:
        return bool(_sess)
    try:
        import onnxruntime as ort
        meta = json.loads(META.read_text(encoding="utf-8"))
        _classes = list(meta["classes"])
        _thresh = float(meta.get("thresh", 0.9))
        _phase_map = dict(meta.get("phase_map", {}))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.log_severity_level = 3
        # CPU：一帧 ~几 ms，不和 TensorRT OCR 抢显卡
        _sess = _gpu_session(MODEL, "界面分类")
    except Exception:                                           # noqa: BLE001
        logger.exception("界面分类模型加载失败（忽略，只用 OCR 判阶段）")
        _sess = False
    return bool(_sess)


def make_inputs(image_rgb: np.ndarray, names=("x",)) -> dict:
    """整屏 RGB（1280x720）-> 模型输入字典。训练脚本测延迟也用它，保证线上线下预处理一致。"""
    import cv2
    if image_rgb.shape[:2] != (720, 1280):
        image_rgb = cv2.resize(image_rgb, (1280, 720), interpolation=cv2.INTER_AREA)
    x = cv2.resize(image_rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    feed = {"x": np.ascontiguousarray(x.transpose(2, 0, 1)[None])}
    if "c" in names:
        c = np.hstack([cv2.resize(image_rgb[y0:y1, x0:x1], wh, interpolation=cv2.INTER_AREA)
                       for (x0, y0, x1, y1), wh in CROP_REGIONS]).astype(np.float32) / 255.0
        feed["c"] = np.ascontiguousarray(c.transpose(2, 0, 1)[None])
    return feed


def probs(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """整屏 RGB -> 各类概率（顺序同 classes()）；加载失败返回 None。"""
    if not _load():
        return None
    logits = _sess.run(None, make_inputs(image_rgb, [i.name for i in _sess.get_inputs()]))[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def classify(image_rgb: np.ndarray) -> Optional[tuple[str, float]]:
    """-> (细类名, 置信度)；加载失败返回 None。门槛判断交给调用方（threshold()）。"""
    p = probs(image_rgb)
    if p is None:
        return None
    j = int(p.argmax())
    return _classes[j], float(p[j])


def to_phase(label: str) -> Optional[str]:
    """细类 -> detect_phase 的取值；没有对应阶段（board_idle / transition / other）返回 None。"""
    _load()
    return _phase_map.get(label)


def classes() -> list:
    _load()
    return list(_classes)


def threshold() -> float:
    _load()
    return _thresh


def warmup() -> Optional[str]:
    if not _load():
        return None
    probs(np.zeros((720, 1280, 3), np.uint8))
    return "cpu"
