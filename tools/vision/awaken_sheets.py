"""把觉醒徽章样本按"划分 × 预标注类别"拼成带编号的复核图（人工复核预标注用）。

  .venv/Scripts/python.exe -X utf8 tools/vision/awaken_sheets.py --split test
输出 data/vision/awaken_v1/review/<split>_<label>_<k>.jpg，每张 80 格（10x8），格子左上角是样本序号，
序号对应 review/<split>_index.json 里的样本 id。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "vision" / "awaken_v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--per", type=int, default=80)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(POOL / "samples.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] == args.split]
    out = POOL / "review"
    out.mkdir(exist_ok=True)
    index = {}
    n = 0
    for lab in (0, 1, 2, 3, None):
        grp = [r for r in rows if r["prelabel"] == lab]
        for k in range(0, len(grp), args.per):
            tiles = []
            for r in grp[k:k + args.per]:
                im = cv2.imdecode(np.fromfile(str(POOL / "crops" / f"{r['id']}.png"), np.uint8), 1)
                im = cv2.resize(im, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
                cv2.putText(im, str(n), (2, 14), 0, 0.45, (0, 255, 255), 1)
                index[n] = r["id"]
                n += 1
                tiles.append(im)
            while len(tiles) % 10:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)])
            cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
                str(out / f"{args.split}_{lab}_{k // args.per}.jpg"))
    (out / f"{args.split}_index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"{args.split}: {n} 个样本")
    return 0


if __name__ == "__main__":
    sys.exit(main())
