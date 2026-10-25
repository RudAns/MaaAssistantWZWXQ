"""视觉模型（D-FINE）第一类"己方场上单位在哪一格"的**帧池 + 初稿**。

2026-09-24：画面上应该能看出哪些位置有英雄 —— 现在的徽章检测只有 66%，先训这一类。

  python tools/vision/build_pool.py --max 400

做法：
  1. 取 data/logs/live_*/phase_prepare_*.png，按时间戳找 states.jsonl 里最近的一帧状态（同一个运行目录、|Δt|<2s）。
  2. 只留**商店关着、人口读数有效**的帧：人口读数 = 场上单位数，是每帧现成的校验（标出来的格子数对不上人口就是标错了）。
  3. 初稿 = 现有等级徽章检测（extractors._detect_badges）认出的格子。
  4. 画一张复核图：己方 28 格标上编号（A1~D7，A 行最靠近敌人），初稿格子画绿框，给老师看图确认/补漏。
输出 data/vision/pool_v1/{frames.jsonl, review/*.jpg}；按对局切 train/val/test（同一局不跨集合）。
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.perception.extractors import OWN_BOARD_ROWS, _detect_badges, cell_key  # noqa: E402

ROWS = "ABCD"


def cell_ids() -> dict[str, tuple[int, int]]:
    return {f"{ROWS[r]}{c + 1}": tuple(cell) for r, row in enumerate(OWN_BOARD_ROWS) for c, cell in enumerate(row)}


def load_states(run_dir: Path):
    ts, rows = [], []
    f = run_dir / "states.jsonl"
    if not f.exists():
        return ts, rows
    for line in f.open(encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        ts.append(r.get("ts", 0))
        rows.append(r.get("state", {}))
    return ts, rows


def render(img_bgr, ids, pre_cells, pop, out):
    im = img_bgr.copy()
    for cid, (x, y) in ids.items():
        on = cell_key((x, y)) in pre_cells
        col = (0, 220, 0) if on else (200, 200, 200)
        cv2.circle(im, (x, y), 7, col, -1 if on else 2)
        cv2.putText(im, cid, (x - 14, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(im, cid, (x - 14, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col if on else (255, 255, 255), 1)
    cv2.putText(im, f"population={pop}  prelabel={len(pre_cells)}", (20, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(im, f"population={pop}  prelabel={len(pre_cells)}", (20, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--out", default="data/vision/pool_v1")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "review").mkdir(parents=True, exist_ok=True)
    ids = cell_ids()
    rng = random.Random(args.seed)
    cands = []
    for rd in sorted(glob.glob(str(ROOT / "data" / "logs" / "live_*"))):
        rd = Path(rd)
        ts, rows = load_states(rd)
        if not ts:
            continue
        for png in rd.glob("phase_prepare_*.png"):
            try:
                t = int(png.stem.split("_")[-1])
            except ValueError:
                continue
            i = bisect.bisect_left(ts, t)
            best = min((j for j in (i - 1, i, i + 1) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t), default=None)
            if best is None or abs(ts[best] - t) > 2:
                continue
            st = rows[best]
            if st.get("shop_open") or not st.get("population_valid") or st.get("phase") != "prepare":
                continue
            pop = st.get("population") or 0
            if not 1 <= pop <= 7:
                continue
            cands.append((rd.name, png, pop))
    rng.shuffle(cands)
    cands = cands[:args.max]
    runs = sorted({c[0] for c in cands})
    rng.shuffle(runs)
    n = len(runs)
    split = {r: ("test" if k < max(1, n // 7) else "val" if k < max(2, 2 * n // 7) else "train") for k, r in enumerate(runs)}
    agree = 0
    with (out / "frames.jsonl").open("w", encoding="utf-8") as fh:
        for k, (run, png, pop) in enumerate(cands):
            im = cv2.imdecode(np.fromfile(str(png), np.uint8), 1)
            if im is None:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            pre = set(_detect_badges(rgb).keys())
            rid = f"f{k:04d}"
            render(im, ids, pre, pop, out / "review" / f"{rid}.jpg")
            agree += len(pre) == pop
            fh.write(json.dumps({"id": rid, "run": run, "frame": str(png.relative_to(ROOT)), "population": pop,
                                 "prelabel": sorted(cid for cid, xy in ids.items() if cell_key(xy) in pre),
                                 "split": split[run]}, ensure_ascii=False) + "\n")
    info = {"frames": len(cands), "runs": n, "prelabel_count_matches_population": agree,
            "split": {s: sum(1 for c in cands if split[c[0]] == s) for s in ("train", "val", "test")}}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
