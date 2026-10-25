"""把"己方格子占用"的人工标注转成 D-FINE 训练用的 COCO 数据集。

  python tools/vision/build_coco.py            # -> data/vision/coco_v1/{train,val,test}/{images,*.json}

- 标注来源：data/vision/pool_v1/labels_{train,evalset}.jsonl（老师逐帧标的己方 28 格占用 + 召唤物）。
- 图片用**原始截图**（frames.jsonl 的 frame），不用带编号/绿点叠加层的复核图（否则模型学会认绿点）。
- 框：以格子为锚的固定框 —— 底边在格心下方 15px，向上 150px 覆盖身体和头顶徽章，宽 90px（格距约 90~100px）。
  推理时取框底部中心 → 最近的格子 = 占用格。单位体型不同，框只是"这一格有单位"的载体，不追求贴合轮廓。
- 类别：0 = hero（英雄），1 = summon（召唤物/图腾，不占人口）。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.perception.extractors import OWN_BOARD_ROWS  # noqa: E402

ROWS = "ABCD"
W, H = 1280, 720
BOX_W, BOX_UP, BOX_DOWN = 90, 150, 15


def cell_xy() -> dict[str, tuple[int, int]]:
    return {f"{ROWS[r]}{c + 1}": tuple(cell) for r, row in enumerate(OWN_BOARD_ROWS) for c, cell in enumerate(row)}


def box(cx: int, cy: int) -> list[float]:
    x0, y0 = max(0, cx - BOX_W / 2), max(0, cy - BOX_UP)
    x1, y1 = min(W, cx + BOX_W / 2), min(H, cy + BOX_DOWN)
    return [x0, y0, x1 - x0, y1 - y0]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames.jsonl", help="帧清单（含 split）；v2 = frames_v2split.jsonl（按对局切分，见 ch2 2.3.3）")
    ap.add_argument("--out", default="coco_v1")
    args = ap.parse_args()
    pool = ROOT / "data" / "vision" / "pool_v1"
    out = ROOT / "data" / "vision" / args.out
    frames = {json.loads(l)["id"]: json.loads(l) for l in (pool / args.frames).open(encoding="utf-8")}
    labels = {}
    for fn in ("labels_train.jsonl", "labels_evalset.jsonl"):
        for l in (pool / fn).open(encoding="utf-8"):
            r = json.loads(l)
            labels[r["id"]] = r
    cells = cell_xy()
    cats = [{"id": 0, "name": "hero"}, {"id": 1, "name": "summon"}]
    stats = {}
    for split in ("train", "val", "test"):
        d = out / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        for i, (fid, f) in enumerate(sorted((k, v) for k, v in frames.items() if v["split"] == split)):
            r = labels.get(fid)
            if r is None:
                continue
            src = ROOT / f["frame"]
            name = f"{fid}.png"
            shutil.copyfile(src, d / "images" / name)
            images.append({"id": i, "file_name": name, "width": W, "height": H})
            summ = set(r.get("summons") or [])
            for c in r["occupied"]:
                if c not in cells:
                    continue
                b = box(*cells[c])
                anns.append({"id": len(anns), "image_id": i, "category_id": 1 if c in summ else 0, "bbox": b,
                             "area": b[2] * b[3], "iscrowd": 0, "cell": c})
        (d / f"{split}.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats},
                                                    ensure_ascii=False), encoding="utf-8")
        stats[split] = {"images": len(images), "boxes": len(anns), "summon": sum(a["category_id"] for a in anns)}
    (out / "info.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
