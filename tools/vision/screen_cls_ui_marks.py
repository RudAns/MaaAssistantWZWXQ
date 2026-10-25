"""界面分类训练集清洗用：两个固定 UI 标记的模板匹配分数（不走 OCR）。

  .venv/Scripts/python.exe -X utf8 tools/vision/screen_cls_ui_marks.py          # 算全部样本 -> ui_marks.jsonl
  .venv/Scripts/python.exe -X utf8 tools/vision/screen_cls_ui_marks.py --check  # 在人工复核过的 val/test 上校准门槛

为什么：棋盘画面的三个子类在缩略图上几乎一样 ——
  prepare（右下角有「准备 / 开战」按钮）、battle（右上角有「战斗统计」）、board_idle（两者都没有：回合间过场、等拍卖、等战斗结束）。
旧 OCR 弱标签在这三类之间错得最多（阶段切换那一帧常常是过场），训练集又没人工复核，
所以用这两个**不随版本/等级变化的 UI 贴图**做模板匹配，给训练集的棋盘帧重新定子类。
匹配在白色笔画二值图上做（HSV：V>190 且 S<70），模板是 templates/ 下三张真实帧裁图：
  stat.png（「战斗统计」）、ready_zb.png（「准备」）、ready_kz.png（「开战」）。
**只用于训练集清洗**；门槛在复核过的 val/test 上校准（--check 输出一致率），模型本身不用这些特征。
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "vision" / "screen_cls_v1"
WIN = {"stat": (1070, 25, 1280, 250), "ready": (1120, 490, 1280, 600)}


def white(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return (((hsv[..., 2] > 190) & (hsv[..., 1] < 70)) * 255).astype(np.uint8)


_T = None


def templates():
    global _T
    if _T is None:
        rd = lambda n: white(cv2.imdecode(np.fromfile(str(DS / "templates" / f"{n}.png"), np.uint8), 1))
        _T = {"stat": [rd("stat")], "ready": [rd("ready_zb"), rd("ready_kz")]}
    return _T


def marks(path: str) -> dict:
    im = cv2.imdecode(np.fromfile(str(ROOT / path), np.uint8), 1)
    if im is None or im.shape[:2] != (720, 1280):
        return {}
    out = {}
    for k, (x0, y0, x1, y1) in WIN.items():
        w = white(im[y0:y1, x0:x1])
        best = 0.0
        for t in templates()[k]:
            r = cv2.matchTemplate(w, t, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(np.nan_to_num(r).max()))
        out[k] = round(best, 3)
    return out


def _job(s):
    return {"id": s["id"], **marks(s["path"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = DS / "ui_marks.jsonl"
    if not a.check:
        ss = [json.loads(l) for l in (DS / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
        with ProcessPoolExecutor(4) as ex, open(out, "w", encoding="utf-8") as f:
            for i, r in enumerate(ex.map(_job, ss, chunksize=64)):
                f.write(json.dumps(r) + "\n")
                if i % 2000 == 0:
                    print(i, len(ss), flush=True)
        print("->", out)
        return 0
    M = {r["id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    R = {r["id"]: r["label"] for r in map(json.loads, (DS / "labels_review.jsonl").read_text(encoding="utf-8").splitlines())}
    board = ("prepare", "battle", "board_idle")
    for ts in (0.5, 0.6, 0.7):
        for tr in (0.5, 0.6, 0.7):
            ok = n = 0
            conf = {}
            for i, lab in R.items():
                if lab not in board or not M.get(i):
                    continue
                m = M[i]
                g = "battle" if m["stat"] >= ts else "prepare" if m["ready"] >= tr else "board_idle"
                n += 1; ok += g == lab
                if g != lab:
                    conf[(lab, g)] = conf.get((lab, g), 0) + 1
            print(f"stat≥{ts} ready≥{tr}: 复核棋盘帧 {n} 张，一致 {ok / n:.4f}  不一致 {sorted(conf.items(), key=lambda x: -x[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
