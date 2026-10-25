"""界面分类数据集的人工复核工具：把若干张整屏截图拼成带编号的缩略图拼板（contact sheet）。

  .venv/Scripts/python.exe -X utf8 tools/vision/screen_cls_sheet.py --ids ID1,ID2,... --out sheet.png
  .venv/Scripts/python.exe -X utf8 tools/vision/screen_cls_sheet.py --paths a.png,b.png --out sheet.png

每格 400x225（原图 1280x720 缩 3.2 倍，按钮文字、价格角标仍看得清），左上角白底黑字编号 = 列表里的序号。
复核时一次看 12 张（3 列 x 4 行），比逐张打开省得多。ID 查 data/vision/screen_cls_v1/samples.jsonl 的 path。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "vision" / "screen_cls_v1"
TW, TH, COLS = 400, 225, 3


def imread(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def sheet(paths: list[Path], labels: list[str] | None = None, cols: int = COLS, tw: int = TW) -> np.ndarray:
    TW, COLS = tw, cols
    TH = tw * 9 // 16
    rows = (len(paths) + COLS - 1) // COLS
    out = np.full((rows * (TH + 4), COLS * (TW + 4), 3), 40, np.uint8)
    for i, p in enumerate(paths):
        im = imread(p)
        im = np.zeros((TH, TW, 3), np.uint8) if im is None else cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, COLS)
        y, x = r * (TH + 4), c * (TW + 4)
        out[y:y + TH, x:x + TW] = im
        tag = str(i) + (f" {labels[i]}" if labels else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, y), (x + tw + 6, y + th + 8), (255, 255, 255), -1)
        cv2.putText(out, tag, (x + 3, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--paths", default="")
    ap.add_argument("--labels", default="", help="逗号分隔，与 ids/paths 对应（只允许 ASCII）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--tw", type=int, default=TW)
    a = ap.parse_args()
    if a.ids:
        idx = {}
        for line in (DS / "samples.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            idx[r["id"]] = r["path"]
        paths = [ROOT / idx[i] for i in a.ids.split(",")]
    else:
        paths = [Path(p) for p in a.paths.split(",")]
    labels = a.labels.split(",") if a.labels else None
    ok, buf = cv2.imencode(".png", sheet(paths, labels, a.cols, a.tw))
    buf.tofile(a.out)
    print(a.out, len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
