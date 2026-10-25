"""觉醒进度数据池：从真实帧里裁出己方单位头顶的等级徽章，并用像素规则预标注（2026-09-24）。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_awaken_pool.py            # 默认用 pool_v1 的 400 帧（沿用它按对局的划分）
  .venv/Scripts/python.exe -X utf8 tools/vision/build_awaken_pool.py --extra 2000   # 另从其它对局补帧（按 run 当作对局分组）

背景：徽章数字外面那圈分成**左 / 上 / 右三段**，每合成一次同名卡亮一段（亮蓝色），
顺序是 左 → 上 → 右（证据：40 级孙膑 左+上 = 2/3，live_1790246871/phase_prepare_1790247589.png）。
三段填满 = 再合成一次就觉醒。

**这里的像素规则只是预标注器**，不是最终读法：它对背景透色、圆心偏移敏感。最终读法是用这批数据
（人工复核后）训练的分类器。

一个样本要同时满足两路证据才收：
  - 霍夫圆找到深色徽章（extractors._detect_badges，已按"正上方有格子"映射到己方格），
  - D-FINE 在同一格检测到单位（agent/perception/vision_units.cell_units）。
输出 data/vision/awaken_v1/：crops/<id>.png（64x64，BGR 原像素）+ samples.jsonl。
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.perception import extractors as ex          # noqa: E402
from agent.perception import vision_units as vu        # noqa: E402

OUT = ROOT / "data" / "vision" / "awaken_v1"
CROP = 32                         # 半边长 → 64x64
SEG = {"L": (160, 215), "T": (65, 115), "R": (-35, 20)}   # 角度（0=右、90=上）
RING_R = (13, 14, 15, 16, 17)


def _blue_frac(hsv, cx, cy, a0, a1) -> float:
    b = n = 0
    for a in range(a0, a1 + 1, 5):
        for r in RING_R:
            x = int(round(cx + r * np.cos(np.radians(a))))
            y = int(round(cy - r * np.sin(np.radians(a))))
            if not (0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]):
                continue
            h, s, v = (int(t) for t in hsv[y, x])
            n += 1
            b += (92 <= h <= 118 and s >= 80 and v >= 130)
    return b / max(n, 1)


def prelabel(img_rgb, cx, cy):
    """-> (0~3 或 None, 三段蓝色覆盖率)。每段在 ±2px 抖动内取最大（霍夫圆心常偏 1~2 像素）。
    None = 读不了：某段覆盖率落在 0.15~0.4 的模糊区，或填充不符合 左→上→右 的顺序。"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    fr = {k: max(_blue_frac(hsv, cx + dx, cy + dy, *v) for dx in (-2, 0, 2) for dy in (-2, 0, 2))
          for k, v in SEG.items()}
    if any(0.15 < fr[k] < 0.4 for k in "LTR"):
        return None, fr
    on = [fr[k] >= 0.4 for k in "LTR"]
    if on not in ([False] * 3, [True, False, False], [True, True, False], [True, True, True]):
        return None, fr
    return sum(on), fr


def _load(p):
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=int, default=0, help="另从 pool_v1 以外的对局补多少帧")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    frames = [json.loads(l) for l in open(ROOT / "data/vision/pool_v1/frames_v2split.jsonl", encoding="utf-8")]
    if args.extra:
        used_runs = {f["run"] for f in frames}
        used = {Path(f["frame"]).name for f in frames}
        cand = [p for p in glob.glob(str(ROOT / "data/logs/live_*/phase_prepare_*.png")) if Path(p).name not in used]
        random.Random(args.seed).shuffle(cand)
        for p in cand[:args.extra]:
            run = Path(p).parent.name
            # 新对局一律进 train（测试集只用 pool_v1 按对局划好的那部分，避免泄漏）；
            # 和 pool_v1 同一局的帧沿用那一局的划分
            split = next((f["split"] for f in frames if f["run"] == run), "train") if run in used_runs else "train"
            match = next((f["match"] for f in frames if f["run"] == run), run)
            frames.append({"id": "x" + Path(p).stem.split("_")[-1], "run": run, "frame": p, "split": split, "match": match})

    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    cells = [tuple(c) for row in ex.OWN_BOARD_ROWS for c in row]
    stats = {}
    n = 0
    with open(OUT / "samples.jsonl", "w", encoding="utf-8") as fo:
        for f in frames:
            bgr = _load(ROOT / f["frame"] if not Path(f["frame"]).is_absolute() else f["frame"])
            if bgr is None:
                continue
            if bgr.shape[1] != 1280:
                bgr = cv2.resize(bgr, (1280, 720))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            badges = ex._detect_badges(rgb)
            units = vu.cell_units(rgb, cells)
            for key, (bx, by) in badges.items():
                cx_, cy_ = (int(t) for t in key.split(","))
                if (cx_, cy_) not in units:
                    continue                                   # D-FINE 没在这一格看到单位：不收
                if not (CROP <= bx < 1280 - CROP and CROP <= by < 720 - CROP):
                    continue
                lab, fr = prelabel(rgb, bx, by)
                sid = f"{f['id']}_{cx_}_{cy_}"
                cv2.imencode(".png", bgr[by - CROP:by + CROP, bx - CROP:bx + CROP])[1].tofile(str(OUT / "crops" / f"{sid}.png"))
                fo.write(json.dumps({"id": sid, "frame": str(f["frame"]), "run": f["run"], "match": f["match"],
                                     "split": f["split"], "cell": key, "center": [bx, by], "unit": units[(cx_, cy_)][0],
                                     "prelabel": lab, "blue": {k: round(v, 2) for k, v in fr.items()}},
                                    ensure_ascii=False) + "\n")
                stats[(f["split"], lab)] = stats.get((f["split"], lab), 0) + 1
                n += 1
    print(f"{len(frames)} 帧 → {n} 个徽章")
    for k in sorted(stats, key=str):
        print(" ", k, stats[k])
    return 0


if __name__ == "__main__":
    sys.exit(main())
