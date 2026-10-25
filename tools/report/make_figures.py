"""重新生成报告用的示例图。

  python tools/report/make_figures.py

复核叠加图（fig_review_overlay_*、fig_offboard_summon_*）直接取自 data/vision/pool_v1/review/，由 tools/vision/build_pool.py 生成。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "laya-report" / "figures"


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), 1)


def imwrite(p, im):
    cv2.imencode(Path(p).suffix, im)[1].tofile(str(p))


def pop_occlusion(frames):
    """人口数字区域放大：f0102 画面 6/7 未遮挡；f0106/f0132 满员 6/6，第一位 6 被 A6 单位徽章挡住左半边。"""
    tiles = []
    for fid in ("f0102", "f0106", "f0132"):
        crop = imread(ROOT / frames[fid]["frame"])[220:320, 780:1000].copy()
        cv2.putText(crop, f"{fid} rec={frames[fid]['population']}", (5, 15), 0, .45, (0, 0, 255), 1)
        tiles.append(cv2.resize(crop, None, fx=2, fy=2))
    imwrite(OUT / "fig_pop_occlusion.png", np.vstack(tiles))


def dfine_cell_boxes():
    """D-FINE 训练框（以格子为锚的固定框，绿=英雄、橙=召唤物），示例 f0061。"""
    d = json.loads((ROOT / "data/vision/coco_v1/val/val.json").read_text(encoding="utf-8"))
    img = next(i for i in d["images"] if i["file_name"] == "f0061.png")
    im = imread(ROOT / "data/vision/coco_v1/val/images" / img["file_name"])
    for a in d["annotations"]:
        if a["image_id"] != img["id"]:
            continue
        x, y, w, h = map(int, a["bbox"])
        col = (0, 255, 0) if a["category_id"] == 0 else (0, 128, 255)
        cv2.rectangle(im, (x, y), (x + w, y + h), col, 2)
        cv2.putText(im, a["cell"], (x, y + h - 3), 0, .6, col, 2)
    imwrite(OUT / "fig_dfine_cell_boxes.jpg", im)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/vision/pool_v1/frames.jsonl").open(encoding="utf-8")}
    pop_occlusion(frames)
    dfine_cell_boxes()
    for fid, name in (("f0003", "fig_review_overlay_f0003.jpg"), ("f0061", "fig_review_overlay_f0061.jpg"),
                      ("f0372", "fig_offboard_summon_f0372.jpg")):
        shutil.copyfile(ROOT / "data/vision/pool_v1/review" / f"{fid}.jpg", OUT / name)
    print("ok ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
