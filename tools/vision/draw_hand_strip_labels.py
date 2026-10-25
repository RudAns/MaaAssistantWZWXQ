"""把 hand_strip_v1 的标签画回原帧，拼成核对图（人工看图核标签用）。

  .venv/Scripts/python.exe -X utf8 tools/vision/draw_hand_strip_labels.py --n 40 --seed 0 [--tier A] [--split test]
      [--out <输出目录>]

每行一帧（原帧 y 500~720、x 150~1200，原分辨率）。画法：
  绿竖线 = 卡牌左边缘 l，黄竖线 = 露出部分右边缘 r（下一张的 l 或 l+142），红点 = 抓取中心 (l+r)/2，
  上方写序号:卡名（awk=1 加「觉」），蓝框 = 枫叶舞槽，紫竖线 = 空位样本点过的 x，held 的牌标「拖」。
每 10 帧一张图：<out>_<k>.png，并写 <out>_index.json（图里第几行 = 哪个样本 id），核对结论按 id 记。
磁盘图是 BGR，按 BGR 画、按 BGR 存（cv2 一致），看图时颜色就是正常的。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "vision" / "hand_strip_v1"
VX0, VX1, VY0, VY1 = 150, 1200, 500, 720


def _font(sz=16):
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def draw(r: dict, idx: int) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(str(ROOT / r["frame"]), np.uint8), cv2.IMREAD_COLOR)
    if bgr.shape[:2] != (720, 1280):
        bgr = cv2.resize(bgr, (1280, 720))
    c = bgr[VY0:VY1, VX0:VX1].copy()
    X = lambda x: int(round(x - VX0))   # noqa: E731
    for k, cd in enumerate(r.get("cards") or []):
        cv2.line(c, (X(cd["l"]), 60), (X(cd["l"]), 220), (0, 255, 0), 2)
        cv2.line(c, (X(cd["r"]) - 1, 90), (X(cd["r"]) - 1, 220), (0, 220, 255), 1)
        cv2.circle(c, (X((cd["l"] + cd["r"]) / 2), 165), 5, (0, 0, 255), -1)
    if r.get("skill"):
        a, b = r["skill"]
        cv2.rectangle(c, (X(a), 66), (X(b), 218), (255, 128, 0), 2)
    if r.get("empty_x") is not None:
        cv2.line(c, (X(r["empty_x"]), 40), (X(r["empty_x"]), 220), (255, 0, 255), 3)
    im = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    f = _font(15)
    head = f"#{idx} {r['id']} tier={r['tier']} src={r.get('src')} n={r.get('n')}"
    dr.rectangle((0, 0, 620, 20), fill=(0, 0, 0))
    dr.text((4, 1), head, fill=(255, 255, 0), font=f)
    for k, cd in enumerate(r.get("cards") or []):
        tag = f"{k}:{cd.get('name', '')}" + ("觉" if cd.get("awk") == 1 else "") + ("拖" if cd.get("held") else "")
        x = X(cd["l"]) + 2
        y = 24 + (k % 2) * 18
        dr.rectangle((x, y, x + 9 * len(tag) + 14, y + 17), fill=(0, 0, 0))
        dr.text((x + 1, y), tag, fill=(0, 255, 0), font=f)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", default="")
    ap.add_argument("--split", default="")
    ap.add_argument("--ids", default="", help="逗号分隔的样本 id（指定就不抽样）")
    ap.add_argument("--out", default=str(ROOT / "docs" / "laya-report" / "figures" / "hand_model_labels"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in (D / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        want = args.ids.split(",")
        rows = [r for r in rows if r["id"] in want]
    else:
        if args.tier:
            rows = [r for r in rows if r["tier"] in args.tier.split(",")]
        if args.split:
            rows = [r for r in rows if r["split"] in args.split.split(",")]
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    index = []
    for p in range(0, len(rows), 10):
        tiles = [draw(r, p + i) for i, r in enumerate(rows[p:p + 10])]
        fp = Path(f"{out}_{p // 10}.png")
        cv2.imencode(".png", np.vstack(tiles))[1].tofile(str(fp))
        index += [{"row": p + i, "img": fp.name, "id": r["id"], "tier": r["tier"], "n": r.get("n"),
                   "names": [c.get("name") for c in r.get("cards") or []]} for i, r in enumerate(rows[p:p + 10])]
        print(fp)
    Path(f"{out}_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
