"""界面分类第二路输入：几个固定区域的高分辨率裁图（整屏缩到 256x144 后小字只剩 5px 高）。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_screen_cls_crops.py   -> data/vision/screen_cls_v1/crops.npy (N x 64 x 224 x 3, BGR)

拼成一张 64x224 的小图（与 samples.jsonl 同序）：
  [0:64)    右上角 (1070,25)-(1280,250)：「战斗统计」/ 天赋、聊天图标 / 等待拍卖结束提示栏
  [64:160)  右下角 (1120,490)-(1280,600)：「准备 / 开战」按钮、拍卖的圆形「出价」
  [160:224) 顶部横幅 (430,0)-(850,45)：拍卖横幅、「请指定英雄使用」、回合倒计时（压扁成 64x64，只要纹理）
线上用同一个函数 agent/perception/screen_cls.py::_crops（保持一致）。
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "vision" / "screen_cls_v1"
REGIONS = [((1070, 25, 1280, 250), (64, 64)), ((1120, 490, 1280, 600), (96, 64)), ((430, 0, 850, 45), (64, 64))]


def crops(img: np.ndarray) -> np.ndarray:
    return np.hstack([cv2.resize(img[y0:y1, x0:x1], wh, interpolation=cv2.INTER_AREA)
                      for (x0, y0, x1, y1), wh in REGIONS])


def _job(path):
    im = cv2.imdecode(np.fromfile(str(ROOT / path), np.uint8), cv2.IMREAD_COLOR)
    return crops(im) if im is not None and im.shape[:2] == (720, 1280) else np.zeros((64, 224, 3), np.uint8)


def main() -> int:
    ss = [json.loads(l) for l in (DS / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(_job, [s["path"] for s in ss], chunksize=64))
    np.save(DS / "crops.npy", np.stack(out))
    print(len(out), "->", DS / "crops.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
