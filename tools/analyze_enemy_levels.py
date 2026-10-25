"""对手等级曲线（2026-09-29，提升方案第 2 步：看对手怎么打）。

  python tools/analyze_enemy_levels.py                 # 全部 live_* 目录里存下的战斗截图
  python tools/analyze_enemy_levels.py live_xxx ...    # 指定对局

战斗画面里每个单位头顶有「等级数字 + 血条」：**蓝色血条 = 对手，绿色 = 我方**（2026-09-29 截图核对）。
先按颜色找血条（细长矩形），再在血条正上方用小 ROI 认字（大 ROI 会把小字号丢掉）。
每张 phase_battle_*.png 按文件名时间戳对到 states.jsonl 的回合号，每回合取对手 / 我方的最高等级与等级总和。
只用 CPU OCR（WZWXQ_OCR_EP=cpu），不和实战抢显卡。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from bisect import bisect_right
from collections import defaultdict

os.environ.setdefault("WZWXQ_OCR_EP", "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SIDES = (("enemy", (105, 80, 60), (140, 255, 255)), ("ours", (35, 80, 80), (85, 255, 255)))


def read_levels(img_bgr, ocr) -> dict:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    out = {"enemy": [], "ours": []}
    for side, lo, hi in SIDES:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            x, y, w, h = cv2.boundingRect(c)
            # 血条：细长；避开左侧玩家列表（x<300）/ 右侧面板（x>1080）/ 底部手牌区
            if not (40 <= w <= 100 and 3 <= h <= 12 and 35 < y < 600 and 300 <= x <= 1080):
                continue
            t = ocr.text(rgb, (x + w // 2 - 24, y - 19, 48, 18)).strip()
            if re.fullmatch(r"\d{1,3}", t):
                out[side].append(int(t))
    return out


def round_index(run_dir: str):
    ts, rd = [], []
    f = os.path.join(run_dir, "states.jsonl")
    if not os.path.exists(f):
        return None
    for line in open(f, encoding="utf-8"):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        r = (x.get("state") or {}).get("round_no")
        if r:
            ts.append(float(x.get("ts") or 0))
            rd.append(r)
    return ts, rd


def analyze(run_dir: str, ocr) -> dict:
    idx = round_index(run_dir)
    if not idx or not idx[0]:
        return {}
    ts, rd = idx
    per = defaultdict(lambda: {"enemy_max": 0, "enemy_sum": 0, "ours_max": 0, "ours_sum": 0, "n": 0})
    for p in sorted(glob.glob(os.path.join(run_dir, "phase_battle_*.png"))):
        m = re.search(r"_(\d{10})", os.path.basename(p))
        if not m:
            continue
        i = bisect_right(ts, float(m.group(1))) - 1
        if i < 0:
            continue
        r = rd[i]
        img = cv2.imdecode(np.fromfile(p, np.uint8), 1)
        if img is None:
            continue
        lv = read_levels(img, ocr)
        a = per[r]
        a["n"] += 1
        for side in ("enemy", "ours"):
            if lv[side]:
                a[f"{side}_max"] = max(a[f"{side}_max"], max(lv[side]))
                a[f"{side}_sum"] = max(a[f"{side}_sum"], sum(lv[side]))
    return dict(per)


def main() -> int:
    from agent.perception.onnx_ocr import OnnxOcr
    ocr = OnnxOcr.get()
    args = sys.argv[1:]
    runs = [os.path.join(ROOT, "data", "logs", a) for a in args] if args else \
        sorted(glob.glob(os.path.join(ROOT, "data", "logs", "live_*")), key=os.path.getmtime)
    agg = defaultdict(lambda: defaultdict(list))
    out = {}
    for d in runs:
        per = analyze(d, ocr)
        if not per:
            continue
        out[os.path.basename(d)] = per
        for r, a in per.items():
            for k in ("enemy_max", "enemy_sum", "ours_max", "ours_sum"):
                if a[k]:
                    agg[r][k].append(a[k])
    json.dump(out, open(os.path.join(ROOT, "data", "enemy_levels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    med = lambda v: sorted(v)[len(v) // 2] if v else 0  # noqa: E731
    print("回合 | 对手最高(中位) 对手总和(中位) | 我方最高(中位) 我方总和(中位) | 样本局数")
    for r in sorted(agg):
        a = agg[r]
        print(f"R{r:>2} | {med(a['enemy_max']):>4} {med(a['enemy_sum']):>5} | {med(a['ours_max']):>4} {med(a['ours_sum']):>5} | {len(a['enemy_max'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
