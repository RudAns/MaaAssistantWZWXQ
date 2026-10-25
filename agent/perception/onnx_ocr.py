"""直接用 onnxruntime 跑 PP-OCR 模型（assets/model/ocr/det.onnx + rec.onnx），绕过 MaaFW 的 OCR。

为什么（2026-09-23，tools/bench_rec_batch.py 实测）：
  - MaaFW 一次 OCR 往返（检测+识别）50~70ms，准备阶段一帧 13~60 次 -> 0.6~2.6 秒/帧，
    是"识别和操作慢"的主因。
  - 同一套模型直接跑：识别 **3.8ms/张**（批量）；整条手牌"检测一次 + 批量识别" **60ms**，
    MaaFW 滑窗读同一帧要 777ms，**结果一样**（7 张全对）。
截图、点击、拖拽继续用 MaaFW（它们不慢）。

**默认就走这里**（2026-09-23 起）。`WZWXQ_OCR=maa` 才退回 MaaFW 自带的 OCR；
没装 onnxruntime 时也自动退回 MaaFW。推理引擎见下面的 `WZWXQ_OCR_EP`（默认 auto）。

模型输入约定（PP-OCR）：
  det: (N,3,H,W) H/W 为 32 的倍数，BGR->RGB 后按 ImageNet 均值方差归一化；输出文字概率图。
  rec: (N,3,48,W) BGR，(x/255-0.5)/0.5；输出 (N,T,18385) —— 0 是 CTC 空白，末位是空格。
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2] / "assets" / "model" / "ocr"
# 实验用：分别换找字 / 认字模型（例如 PP-OCRv5 服务器版 assets/model/ocr_server、微调后的认字模型）。
# 不设 = 线上默认的轻量版。认字模型的字典（keys.txt）跟着认字模型的目录走。
SERVER_DIR = Path(__file__).resolve().parents[2] / "assets" / "model" / "ocr_server"
DET_DIR = Path(os.environ.get("WZWXQ_OCR_DET_DIR") or MODEL_DIR)
# **认字默认用服务器版**（2026-09-25 起全部改用 Server 模型，轻量版只作对照）。
# 手牌卡名难例上 服务器版+闭集 读错 1 vs 轻量自由解码 12。
# 回退轻量：WZWXQ_OCR_REC_DIR=assets/model/ocr。找字（det）暂留轻量：09-24 实测服务器版找字在我们的窄 ROI 里更差（把图标、残行也框进来），待复测。
REC_DIR = Path(os.environ.get("WZWXQ_OCR_REC_DIR") or SERVER_DIR)
REC_H = 48
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
DET_THRESH = 0.3          # 概率图二值化阈值（PP-OCR 默认）
DET_BOX_THRESH = 0.5      # 框内平均概率下限
DET_UNCLIP = 0.75         # DB 输出的是收缩过的文字核，往外扩 0.75 倍字高（≈ PP-OCR unclip 1.5）
THIN_ROI_H = 40           # 比这还矮的 ROI 不做检测，整块当一行识别（MaaFW 对细 ROI 也是这么返回的）


logger = logging.getLogger(__name__)
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


def enabled() -> bool:
    """默认 True；`WZWXQ_OCR=maa` 或没装 onnxruntime 时 False（走 MaaFW 的 OCR）。"""
    return _HAS_ORT and os.environ.get("WZWXQ_OCR", "onnx").lower() != "maa"


# ---------------- 推理后端（同一对 .onnx 文件，换引擎跑）----------------
# `WZWXQ_OCR_EP`：
#   auto（默认）先试 TensorRT，不行就降级到 onnxruntime CPU —— A 卡 / 核显 / 没装 GPU 包的都能跑；
#   tensorrt / cpu / cuda / openvino / openvino-gpu —— 指定哪个就只用哪个，加载失败直接报错。
# tensorrt / cuda 要装 onnxruntime-gpu（requirements-nvidia.txt），它和 CPU 版 onnxruntime 互斥。
# 实测（tools/bench_ocr_backends.py）：整帧 TensorRT 91ms / CPU 377ms，判定结果一致。
AUTO_ORDER = ("tensorrt", "cpu")
EP_ENV = "WZWXQ_OCR_EP"
TRT_CACHE = Path(__file__).resolve().parents[2] / ".trt_cache"
# TensorRT 要事先声明输入尺寸范围（优化 profile），超出范围会重建引擎（几十秒）。
# det：ROI 最大就是整帧 1280x720 -> 补到 32 的倍数；rec：批量 ≤32，宽 ≤960（见 recognize）。
TRT_PROFILES = {
    "det": ("x:1x3x32x32", "x:1x3x64x320", "x:1x3x736x1280"),
    "rec": ("x:1x3x48x64", "x:8x3x48x320", "x:32x3x48x960"),
}


def backend() -> str:
    return os.environ.get(EP_ENV, "auto").lower()


def _trt_cache_ready() -> bool:
    return TRT_CACHE.is_dir() and any(TRT_CACHE.glob("*.engine"))


class _Runner:
    """一个模型 + 一个推理引擎。调用：runner(x) -> 第一个输出。线程安全。"""

    def __init__(self, name: str, ep: str, threads: int = 2, model_dir: Optional[Path] = None) -> None:
        # model_dir：单独加载别的模型（手牌卡名用服务器版认字模型，见 hand_names.py），不影响全局 OCR
        mdir = Path(model_dir or (DET_DIR if name == "det" else REC_DIR))
        path = str(mdir / f"{name}.onnx")
        # TensorRT 引擎缓存按模型目录分开（默认轻量版在根目录，其余在子目录）—— 换模型不会误用别的引擎
        trt_dir = TRT_CACHE if mdir.resolve() == MODEL_DIR.resolve() else TRT_CACHE / mdir.name
        self.ep = ep
        if ep.startswith("openvino"):
            import openvino as ov
            dev = "GPU" if ep.endswith("gpu") else "CPU"
            cfg = {"PERFORMANCE_HINT": "LATENCY"}
            if dev == "CPU":
                cfg["INFERENCE_NUM_THREADS"] = threads
            self._ov = ov.Core().compile_model(path, dev, cfg)
            self._local = threading.local()      # 每个线程一个 infer request（它不是线程安全的）
            return
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads        # 提取层有 4 个并发 worker，别互相抢核
        so.log_severity_level = 3                # TensorRT 每次加载刷几十行无害 WARNING，只留错误
        if ep in ("tensorrt", "cuda"):
            _load_gpu_dlls(ort)
        if ep == "tensorrt":
            trt_dir.mkdir(parents=True, exist_ok=True)
            lo, opt, hi = TRT_PROFILES[name]
            prov = [("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_dir),
                "trt_timing_cache_enable": True,
                "trt_profile_min_shapes": lo,
                "trt_profile_opt_shapes": opt,
                "trt_profile_max_shapes": hi,
            }), "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif ep == "cuda":
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            prov = ["CPUExecutionProvider"]
        self._ort = ort.InferenceSession(path, so, providers=prov)
        # 加载失败时 onnxruntime **不报错、悄悄退到下一个**（TensorRT -> CUDA），
        # 2026-09-23 实测：装成 TensorRT 11 时它一直在用 CUDA，测出来的"TensorRT"数字全是 CUDA 的。
        want = {"tensorrt": "TensorrtExecutionProvider", "cuda": "CUDAExecutionProvider"}.get(ep)
        got = self._ort.get_providers()[0]
        if want and got != want:
            raise RuntimeError(f"{ep} 没加载成功，onnxruntime 退回了 {got}（缺 DLL / 版本不对？）")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.ep.startswith("openvino"):
            req = getattr(self._local, "req", None)
            if req is None:
                req = self._local.req = self._ov.create_infer_request()
            return req.infer({0: x})[0].copy()
        out = self._ort.run(None, {"x": x})[0]
        if self.ep == "tensorrt" and not getattr(self, "_demoted", False) \
                and self._ort.get_providers()[0] != "TensorrtExecutionProvider":
            # 运行时出错（如输入超出引擎尺寸范围）onnxruntime 会**永久**把会话降到 CUDA，
            # 只打一行日志、不抛异常 —— 结果照样对，只是慢 10 倍。至少要让日志里看得见。
            self._demoted = True
            logger.error("TensorRT 在运行中被 onnxruntime 撤掉了，已退回 %s（输入 %s 超出引擎范围？）",
                         self._ort.get_providers()[0], tuple(x.shape))
        return out


def _load_gpu_dlls(ort) -> None:
    """CUDA/cuDNN/TensorRT 的 DLL 都是 pip 包带的，不在系统 PATH 里，要先登记目录。

    **装了 torch 就必须先 import torch**（2026-09-23 实测）：CUDA 版 torch 自带一份 cuDNN，
    若 onnxruntime 先把 pip 包里那份 cuDNN 装进进程，之后 import torch 直接崩
    （`WinError 127 ... cudnn_cnn64_9.dll`）—— 而 Laya 就要 import torch。
    反过来 torch 先加载，两边都正常（OCR 仍是 TensorRT、torch 卷积也能跑）。
    """
    if importlib.util.find_spec("torch") is not None:
        import torch  # noqa: F401  —— 只为抢先加载它那份 CUDA/cuDNN
    try:
        import tensorrt_libs          # tensorrt-cu12 带的 nvinfer_10.dll 等
        d = os.path.dirname(tensorrt_libs.__file__)
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()            # nvidia-cuda-runtime / cudnn / cublas 的 pip 包


REC_BATCH = 32                     # = TRT_PROFILES["rec"] 的批量上限

# 服务器版字表里有带圈数字 / 全角数字：拍卖出价「2」挨着圆形金币图标被读成「②」，下游只认 ASCII 数字 → 丢了（2026-09-25 换服务器版后）。
# 统一归一成普通数字（带圈 11~20 会变成两位数字）。
_DIGIT_NORM = str.maketrans({**{chr(0x2460 + k): str(k + 1) for k in range(20)},        # ①..⑳
                             **{chr(0x2776 + k): str(k + 1) for k in range(10)},        # ❶..❿
                             **{chr(0x2780 + k): str(k + 1) for k in range(10)},        # ➀..➉
                             **{chr(0xFF10 + k): str(k) for k in range(10)},            # ０..９
                             "⓪": "0", "⓿": "0"})


def _aspect(c) -> float:
    return c.shape[1] / max(1, c.shape[0]) if c is not None and c.size else 0.0


class OnnxOcr:
    _inst: Optional["OnnxOcr"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "OnnxOcr":
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def __init__(self, ep: Optional[str] = None, threads: int = 2) -> None:
        want = ep or backend()
        self.fallback: list[str] = []          # auto 模式下试过但没成的后端及原因
        for cand in (AUTO_ORDER if want == "auto" else (want,)):
            try:
                if cand == "tensorrt":
                    self._precheck_trt()
                self.det = _Runner("det", cand, threads)
                self.rec = _Runner("rec", cand, threads)
                self.ep = cand
                break
            except Exception as e:                      # noqa: BLE001
                if want != "auto":
                    raise
                self.fallback.append(f"{cand}: {type(e).__name__}: {e}")
                logger.warning("OCR 后端 %s 不可用，降级：%s", cand, e)
        else:
            raise RuntimeError("没有可用的 OCR 推理后端：" + "；".join(self.fallback))
        keys = (REC_DIR / "keys.txt").read_text(encoding="utf-8").split("\n")
        if keys and keys[-1] == "":
            keys = keys[:-1]
        self.chars = ["<blank>"] + keys + [" "]

    @staticmethod
    def _precheck_trt() -> None:
        """在真去加载之前先快速排除：CPU 版 onnxruntime 根本没有 TensorRT 这个选项。"""
        import onnxruntime as ort
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("装的是 CPU 版 onnxruntime（N 卡加速见 requirements-nvidia.txt）")
        if not _trt_cache_ready():
            logger.warning("第一次用 TensorRT：要为本机显卡建引擎，约 8 分钟，只需一次"
                           "（缓存在 %s）", TRT_CACHE)

    def describe(self) -> str:
        name = {"tensorrt": "TensorRT（N 卡）", "cpu": "onnxruntime CPU",
                "cuda": "CUDA", "openvino": "OpenVINO CPU", "openvino-gpu": "OpenVINO GPU"}
        s = name.get(self.ep, self.ep)
        return s + (f"（降级自：{'；'.join(self.fallback)}）" if self.fallback else "")

    # ---------------- 识别 ----------------
    def recognize(self, crops_bgr: list[np.ndarray]) -> list[str]:
        """批量识别。超过 REC_BATCH 张就按宽窄排序后分批（TensorRT 引擎只收一批 ≤32 张，
        超了 onnxruntime 会**悄悄永久退回 CUDA**——2026-09-23 整屏预热 48 个框时撞上）。"""
        if len(crops_bgr) > REC_BATCH:
            order = sorted(range(len(crops_bgr)), key=lambda i: _aspect(crops_bgr[i]))
            out = [""] * len(crops_bgr)
            for k in range(0, len(order), REC_BATCH):
                idx = order[k:k + REC_BATCH]
                for i, t in zip(idx, self._recognize([crops_bgr[i] for i in idx])):
                    out[i] = t
            return out
        return self._recognize(crops_bgr)

    def _recognize(self, crops_bgr: list[np.ndarray]) -> list[str]:
        import cv2
        crops = [c for c in crops_bgr if c is not None and c.size and c.shape[0] > 1 and c.shape[1] > 1]
        if len(crops) != len(crops_bgr):
            # 保持下标对齐：空图给空串
            out, it = [], iter(self._recognize(crops)) if crops else iter(())
            for c in crops_bgr:
                ok = c is not None and c.size and c.shape[0] > 1 and c.shape[1] > 1
                out.append(next(it) if ok else "")
            return out
        widths = [max(1, int(round(c.shape[1] * REC_H / c.shape[0]))) for c in crops]
        W = (max(64, min(960, max(widths))) + 7) // 8 * 8
        batch = np.zeros((len(crops), 3, REC_H, W), np.float32)
        for i, (c, w) in enumerate(zip(crops, widths)):
            w = min(w, W)
            im = cv2.resize(c, (w, REC_H)).astype(np.float32) / 255.0
            batch[i, :, :, :w] = ((im - 0.5) / 0.5).transpose(2, 0, 1)
        probs = self.rec(batch)
        return [self._ctc(p) for p in probs]

    def _ctc(self, probs: np.ndarray) -> str:
        idx = probs.argmax(axis=-1)
        out, prev = [], -1
        for i in idx:
            if i != prev and i != 0 and i < len(self.chars):
                out.append(self.chars[i])
            prev = i
        return "".join(out).strip().translate(_DIGIT_NORM)

    # ---------------- 检测 ----------------
    def detect(self, img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """文字框 (x, y, w, h)，坐标相对输入图。"""
        import cv2
        h, w = img_bgr.shape[:2]
        H, W = max(32, (h + 31) // 32 * 32), max(32, (w + 31) // 32 * 32)
        im = cv2.resize(img_bgr, (W, H)).astype(np.float32) / 255.0
        im = ((im[:, :, ::-1] - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        prob = self.det(im)[0, 0]
        mask = (prob > DET_THRESH).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        # 每块的平均概率：一次 bincount 算完。原来逐块 `prob[lab == k].mean()`
        # 每块都扫一遍整张图，整帧上百块时这一步比推理本身还慢（2026-09-23）。
        sums = np.bincount(lab.ravel(), weights=prob.ravel(), minlength=n)
        boxes = []
        sx, sy = w / W, h / H
        for k in range(1, n):
            x, y, bw, bh, area = st[k]
            if area < 12 or bh < 2:
                continue
            if sums[k] / area < DET_BOX_THRESH:
                continue
            e = int(bh * DET_UNCLIP) + 2
            x0, y0 = max(0, x - e), max(0, y - e)
            x1, y1 = min(W, x + bw + e), min(H, y + bh + e)
            boxes.append((int(x0 * sx), int(y0 * sy),
                          max(1, int((x1 - x0) * sx)), max(1, int((y1 - y0) * sy))))
        boxes.sort(key=lambda b: (b[1] // 12, b[0]))      # 大致按阅读顺序
        return boxes

    # ---------------- 对外：和 extractors._ocr_boxes 同一个口径 ----------------
    def boxes(self, image_rgb: np.ndarray, roi) -> list[tuple[str, list]]:
        """ROI 内全部 (文本, 绝对坐标 [x,y,w,h])。"""
        import cv2
        x, y, w, h = (int(v) for v in roi)
        H, W = image_rgb.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return []
        patch = cv2.cvtColor(np.ascontiguousarray(image_rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2BGR)
        if patch.shape[0] < THIN_ROI_H:
            # 细 ROI 本来就只有一行字：不检测，整块识别（MaaFW 对细 ROI 也是这么返回的）
            t = self.recognize([patch])[0]
            if t:
                return [(t, [x0, y0, x1 - x0, y1 - y0])]
            # **整块读空 → 再检测一次、读紧框**（2026-09-25 认字换服务器版后：拍卖出价按钮下 74x34 框里的单个小「3」，
            # 服务器版整块读成空串，闭集解码也几乎没信号；检测出 17x16 的紧框后服务器版能读对。轻量版整块能读，所以原来没这一步）
            # 检测要四周留白：在上下各扩 20、左右各扩 10 的区域上检测，只收中心落在原框内的文字框
            ex0, ey0 = max(0, x0 - 10), max(0, y0 - 20)
            ex1, ey1 = min(W, x1 + 10), min(H, y1 + 20)
            big = cv2.cvtColor(np.ascontiguousarray(image_rgb[ey0:ey1, ex0:ex1]), cv2.COLOR_RGB2BGR)
            found = [b for b in self.detect(big)
                     if x0 <= ex0 + b[0] + b[2] // 2 < x1 and y0 <= ey0 + b[1] + b[3] // 2 < y1]
            if not found:
                return []
            crops = [big[b[1]:b[1] + b[3], b[0]:b[0] + b[2]] for b in found]
            texts = self.recognize(crops)
            return [(t, [ex0 + b[0], ey0 + b[1], b[2], b[3]]) for t, b in zip(texts, found) if t]
        found = self.detect(patch)
        if not found:
            # **高 ROI 检测不到字 = 没有字**，和 MaaFW 一致，不要整块硬认。
            # 原来这里会把整块拿去识别 —— 立绘被认成乱码、再被模糊匹配成卡名
            # （2026-09-23：敌方单位认成「马超」「小乔」，凑出一个假的 N 选 1；
            # 能量区连闪电图标一起认，7 读成 37）。
            return []
        crops = [patch[b[1]:b[1] + b[3], b[0]:b[0] + b[2]] for b in found]
        texts = self.recognize(crops)
        return [(t, [x0 + b[0], y0 + b[1], b[2], b[3]]) for t, b in zip(texts, found) if t]

    def text(self, image_rgb: np.ndarray, roi, expected: Optional[str] = None) -> str:
        """和 extractors._ocr 同一个口径：有 expected 时返回第一个命中的文本，否则按阅读顺序拼起来。"""
        items = self.boxes(image_rgb, roi)
        if expected:
            pat = re.compile(expected)
            hit = next((t for t, _b in items if pat.search(t)), "")
            return hit
        return "".join(t for t, _b in items)
