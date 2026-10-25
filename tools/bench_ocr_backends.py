"""阶段识别（detect_phase 的文字判据）换推理引擎 / 换识别方式的对比基准。**不要模拟器。**

为什么（2026-09-23）：onnx 后端下一帧 406ms，其中 detect_phase 约 300ms —— 17 条判据
每条都是"ROI 内检测 + 识别"。把 OpenVINO / TensorRT 都试一遍，特别是阶段识别。

样本：data/logs/live_*/phase_<阶段>_<ts>.png —— 文件名就是当时判出的阶段（MaaFW OCR 判的），
每个阶段均匀抽 K 张。

三种识别方式（都用同一对 det.onnx / rec.onnx）：
  roi     现在的做法：每条判据单独"ROI 内检测 + 识别"（17 次检测）
  roi-b   每个 ROI 检测一次，所有文字框攒成**一批**识别（15 次检测 + 1 次识别）
  frame   整帧只检测**一次**，所有文字框一批识别，再按"框中心落在哪个 ROI"分给判据

用法：
  .venv/Scripts/python.exe -X utf8 tools/bench_ocr_backends.py --ep cpu openvino openvino-gpu
  .venv-trt/Scripts/python.exe -X utf8 tools/bench_ocr_backends.py --ep tensorrt cuda
  结果追加到 data/logs/bench_ocr_backends.jsonl（两个环境的结果最后合在一起看）。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agent.perception import onnx_ocr as O  # noqa: E402
from agent.perception.extractors import MODAL_PHASES, PHASE_ORDER, PHASE_PREDICATES  # noqa: E402

OUT = PROJECT_DIR / "data" / "logs" / "bench_ocr_backends.jsonl"
ORDER = list(MODAL_PHASES) + [p for p in PHASE_ORDER if p not in MODAL_PHASES]
PROBES = [(ph, tuple(roi), pat[0] if isinstance(pat, tuple) else pat)
          for ph in ORDER for roi, pat in PHASE_PREDICATES[ph]]  # 计数型判据这里近似成"有一处命中"
UNIQUE_ROIS = list(dict.fromkeys(r for _, r, _ in PROBES))
NO_TEXT_PHASES = ("card_pick", "unknown")      # 这两类没有文字判据，期望"一条都不中"


def frames(per_phase: int) -> list[tuple[str, str]]:
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(str(PROJECT_DIR / "data/logs/live_*/phase_*.png"))):
        m = re.match(r"phase_([a-z_]+)_\d+\.png$", os.path.basename(f))
        if m:
            by[m.group(1)].append(f)
    out = []
    for ph, fs in sorted(by.items()):
        step = max(1, len(fs) // per_phase)
        out += [(ph, f) for f in fs[::step][:per_phase]]
    return out


def first_hit(texts_by_probe: list[str]) -> str | None:
    hits = {ph for (ph, _r, pat), t in zip(PROBES, texts_by_probe) if t and re.search(pat, t)}
    return next((ph for ph in ORDER if ph in hits), None)


# ---------------- 三种识别方式：输入 RGB 帧，返回每条判据读到的文字 ----------------

def mode_roi(ocr: O.OnnxOcr, img, pool=None) -> list[str]:
    fn = (lambda p: ocr.text(img, p[1], expected=p[2]))
    return list(pool.map(fn, PROBES)) if pool else [fn(p) for p in PROBES]


def _crop_bgr(img, roi):
    x, y, w, h = roi
    return cv2.cvtColor(np.ascontiguousarray(img[y:y + h, x:x + w]), cv2.COLOR_RGB2BGR)


def mode_roi_batched(ocr: O.OnnxOcr, img, pool=None) -> list[str]:
    patches = [_crop_bgr(img, r) for r in UNIQUE_ROIS]

    def boxes_of(p):
        return [] if p.shape[0] < O.THIN_ROI_H else ocr.detect(p)
    found = list(pool.map(boxes_of, patches)) if pool else [boxes_of(p) for p in patches]
    crops, owner = [], []
    for i, (p, bs) in enumerate(zip(patches, found)):
        for b in (bs or [(0, 0, p.shape[1], p.shape[0])]):
            crops.append(p[b[1]:b[1] + b[3], b[0]:b[0] + b[2]])
            owner.append(i)
    texts = _rec_chunks(ocr, crops)
    per_roi = collections.defaultdict(list)
    for i, t in zip(owner, texts):
        per_roi[UNIQUE_ROIS[i]].append(t)
    return ["".join(per_roi[r]) for _, r, _ in PROBES]


def mode_frame(ocr: O.OnnxOcr, img, pool=None) -> list[str]:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    boxes = ocr.detect(bgr)
    texts = _rec_chunks(ocr, [bgr[b[1]:b[1] + b[3], b[0]:b[0] + b[2]] for b in boxes])
    out = []
    for _ph, (x, y, w, h), _pat in PROBES:
        out.append("".join(t for t, b in zip(texts, boxes)
                           if x <= b[0] + b[2] / 2 <= x + w and y <= b[1] + b[3] / 2 <= y + h))
    return out


def _rec_chunks(ocr, crops, n=32):
    """按宽度排序后分批（一批的宽度按最宽那张补齐，混着排会白算很多空白）。"""
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(1, crops[i].shape[0]))
    res = [""] * len(crops)
    for k in range(0, len(order), n):
        idx = order[k:k + n]
        for i, t in zip(idx, ocr.recognize([crops[i] for i in idx])):
            res[i] = t
    return res


MODES = {"roi": mode_roi, "roi-b": mode_roi_batched, "frame": mode_frame}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", nargs="+", default=["cpu"])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--per-phase", type=int, default=15)
    ap.add_argument("--threads", type=int, default=2, help="CPU 推理线程（线上是 2 线程 x 4 worker）")
    ap.add_argument("--workers", type=int, default=4, help="并发 worker（线上 4）；0=串行")
    args = ap.parse_args()

    data = frames(args.per_phase)
    imgs = [(ph, cv2.cvtColor(cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR),
                              cv2.COLOR_BGR2RGB)) for ph, f in data]
    print(f"{len(imgs)} 帧，{len(PROBES)} 条判据 / {len(UNIQUE_ROIS)} 个不同 ROI")

    ref_path = PROJECT_DIR / "data" / "logs" / "bench_ocr_backends_ref.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
    for ep in args.ep:
        t0 = time.perf_counter()
        try:
            ocr = O.OnnxOcr(ep=ep, threads=args.threads)
            for _ in range(3):                      # 预热（TensorRT 首次要建引擎，很慢）
                for m in args.modes:
                    MODES[m](ocr, imgs[0][1])
        except Exception as e:                      # noqa: BLE001 —— 某个后端装不上就跳过
            print(f"[{ep}] 加载失败：{type(e).__name__}: {e}")
            continue
        print(f"[{ep}] 加载+预热 {time.perf_counter() - t0:.1f}s")
        pool = ThreadPoolExecutor(args.workers) if args.workers else None
        for m in args.modes:
            ms, got = [], []
            for _ph, img in imgs:
                s = time.perf_counter()
                texts = MODES[m](ocr, img, pool)
                ms.append((time.perf_counter() - s) * 1000)
                got.append(first_hit(texts))
            known = [(ph, g) for (ph, _), g in zip(imgs, got) if ph not in NO_TEXT_PHASES]
            acc = sum(ph == g for ph, g in known)
            none_ok = sum(g is None for (ph, _), g in zip(imgs, got) if ph in NO_TEXT_PHASES)
            n_none = sum(ph in NO_TEXT_PHASES for ph, _ in imgs)
            if ep == "cpu" and m == "roi" and ref is None:
                ref = got
                ref_path.write_text(json.dumps(got), encoding="utf-8")
            same = sum(a == b for a, b in zip(ref or [], got)) if ref and len(ref) == len(got) else -1
            wrong = collections.Counter((ph, g) for ph, g in known if ph != g).most_common(4)
            row = {"ep": ep, "mode": m, "frames": len(imgs), "workers": args.workers,
                   "threads": args.threads, "ms_median": round(statistics.median(ms), 1),
                   "ms_p90": round(sorted(ms)[int(len(ms) * 0.9)], 1),
                   "label_acc": f"{acc}/{len(known)}", "no_text_ok": f"{none_ok}/{n_none}",
                   "same_as_ref": f"{same}/{len(got)}", "top_wrong": wrong}
            print(json.dumps(row, ensure_ascii=False))
            with OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if pool:
            pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
