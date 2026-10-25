"""离线基准：绕过 MaaFW，直接用 onnxruntime 跑 OCR 识别模型（rec.onnx），比较
**逐张识别** vs **批量识别** 的速度，以及和 MaaFW 当时读数的一致率。**不要模拟器。**

为什么：2026-09-23 `--perf` 实测，每帧 13~60 次 OCR、每次 50~70ms（先检测文字在哪、再识别），
一帧 0.6~2.6 秒，是"识别和操作慢"的主因。
游戏界面布局固定，卡名/数字的位置都已知 —— 不需要"检测"，切好小图后**一次推理批量识别**即可。

样本：最近几局 states.jsonl 里记录的手牌（卡名 + x + name_y）配上同一时刻存下的准备帧，
按位置切出卡名小图；"标准答案"是 MaaFW 当时读出并解析成功的卡名。

用法：python -X utf8 tools/bench_rec_batch.py [--runs 6]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

MODEL_DIR = PROJECT_DIR / "assets" / "model" / "ocr"
REC_H = 48


class BatchRec:
    """PP-OCR 识别模型的最小封装：切好的小图 -> 文本。可一次喂多张。"""

    def __init__(self, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(MODEL_DIR / "rec.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        keys = (MODEL_DIR / "keys.txt").read_text(encoding="utf-8").split("\n")
        if keys and keys[-1] == "":
            keys = keys[:-1]
        self.chars = ["<blank>"] + keys + [" "]

    @staticmethod
    def _prep(img_bgr: np.ndarray, width: int) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        nw = min(width, max(1, int(round(w * REC_H / max(h, 1)))))
        im = cv2.resize(img_bgr, (nw, REC_H)).astype(np.float32) / 255.0
        im = (im - 0.5) / 0.5
        out = np.zeros((3, REC_H, width), np.float32)
        out[:, :, :nw] = im.transpose(2, 0, 1)
        return out

    def _decode(self, probs: np.ndarray) -> str:
        idx = probs.argmax(axis=-1)
        out, prev = [], -1
        for i in idx:
            if i != prev and i != 0:
                out.append(self.chars[i] if i < len(self.chars) else "")
            prev = i
        return "".join(out).strip()

    def __call__(self, crops: list[np.ndarray]) -> list[str]:
        if not crops:
            return []
        width = max(64, min(640, max(int(round(c.shape[1] * REC_H / max(c.shape[0], 1)))
                                     for c in crops)))
        width = (width + 7) // 8 * 8
        batch = np.stack([self._prep(c, width) for c in crops])
        probs = self.sess.run(None, {"x": batch})[0]
        return [self._decode(p) for p in probs]


def samples(runs: int):
    """(小图BGR, 标准答案卡名) —— 手牌卡名。"""
    dirs = sorted(glob.glob(str(PROJECT_DIR / "data/logs/live_*")), key=os.path.getmtime)[-runs:]
    out = []
    for d in dirs:
        p = os.path.join(d, "states.jsonl")
        if not os.path.exists(p):
            continue
        states = {}
        for line in open(p, encoding="utf-8"):
            try:
                st = json.loads(line).get("state") or {}
            except json.JSONDecodeError:
                continue
            states[int(st.get("timestamp", 0))] = st
        for f in glob.glob(os.path.join(d, "phase_prepare_*.png")):
            t = int(Path(f).stem.split("_")[-1])
            st = states.get(t) or states.get(t - 1) or states.get(t + 1)
            if not st or st.get("shop_open"):
                continue
            img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            for sl in st.get("hand_slots") or []:
                ny = sl.get("name_y") or (sl["y"] + 25)
                x0, y0 = max(0, sl["x"] - 62), max(0, ny - 17)
                crop = img[y0:ny + 17, x0:sl["x"] + 62]
                if crop.size:
                    out.append((crop, sl["name"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()
    from agent.perception.extractors import StateExtractor
    ex = StateExtractor(PROJECT_DIR / "assets" / "image")
    data = samples(args.runs)
    print(f"样本 {len(data)} 张手牌卡名小图（最近 {args.runs} 局）")
    if not data:
        return 0
    rec = BatchRec()
    rec([data[0][0]])                                   # 预热
    crops = [c for c, _ in data]

    t0 = time.perf_counter()
    one = [rec([c])[0] for c in crops]
    t_one = time.perf_counter() - t0

    t0 = time.perf_counter()
    batched = []
    for i in range(0, len(crops), 16):                  # 一帧手牌最多 10 张，这里按 16 一批
        batched += rec(crops[i:i + 16])
    t_batch = time.perf_counter() - t0

    def acc(texts):
        ok = sum(1 for t, (_c, name) in zip(texts, data)
                 if t and (ex.resolve_name(t) == name or t == name))
        return ok, len(data)

    print(f"逐张识别：{t_one * 1000 / len(crops):6.1f} ms/张   一致率 {acc(one)[0]}/{len(data)}")
    print(f"批量识别：{t_batch * 1000 / len(crops):6.1f} ms/张   一致率 {acc(batched)[0]}/{len(data)}")
    miss = [(t, name) for t, (_c, name) in zip(batched, data)
            if not (t and (ex.resolve_name(t) == name or t == name))][:12]
    print("不一致的例子（读到 -> MaaFW 当时的答案）:", miss)
    print("参考：MaaFW 一次 OCR 往返（检测+识别）实测 50~70ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
