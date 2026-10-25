"""人工复核用的拼图：把一批裁图按网格拼起来、每格标序号和自动标签（2026-10-12，v2 加训测试集复核）。

  .venv/Scripts/python.exe -X utf8 tools/vision/sheet_v2.py level  [--start 0 --n 60]
  .venv/Scripts/python.exe -X utf8 tools/vision/sheet_v2.py equip  [--start 0 --n 60]
输出 data/vision_v2/review/<kind>_<start>.png；序号 = 候选列表里的位置（候选列表写在 <kind>_candidates.json）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data" / "vision_v2"
REV = V2 / "review"


def candidates(kind):
    f = REV / f"{kind}_candidates.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    rng = random.Random(0)
    if kind == "level":
        rows = [json.loads(l) for l in (V2 / "level_v2" / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        c = [r for r in rows if r["split"] == "test" and r["level"] >= 100]
        rng.shuffle(c)
        out = [{"id": r["id"], "label": r["level"]} for r in c]
    else:
        rows = [json.loads(l) for l in (V2 / "equip_cnt" / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        out = []
        for k in range(4):
            c = [r for r in rows if r["split"] == "test" and r["strong"] and not r["mono_bad"] and r["label"] == k]
            rng.shuffle(c)
            out += [{"id": r["id"], "label": r["label"]} for r in c[:90]]
        rng.shuffle(out)
    REV.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["level", "equip"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    cand = candidates(a.kind)[a.start:a.start + a.n]
    if a.kind == "level":
        z = np.load(V2 / "level_v2" / "crops.npz")["badge"]
        scale, cols = 2, 10
    else:
        z = np.load(V2 / "equip_cnt" / "crops.npz")["equip"]
        scale, cols = 2, 6
    tiles = []
    for i, c in enumerate(cand):
        im = cv2.resize(z[c["id"]], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        bar = np.zeros((18, im.shape[1], 3), np.uint8)
        cv2.putText(bar, f"#{a.start + i} L={c['label']}", (2, 13), 0, 0.45, (0, 255, 255), 1)
        tiles.append(np.vstack([bar, im]))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = REV / f"{a.kind}_{a.start}.png"
    cv2.imencode(".png", sheet)[1].tofile(str(out))
    print(out, len(cand), "/", len(candidates(a.kind)))


if __name__ == "__main__":
    main()
