"""等级徽章检测的准确率基准 —— **不要模拟器**，几十秒跑完。

真值从哪来：准备阶段、商店关着时，人口读数 `N/M` 的 N 就是场上单位数，
而这个读数实测 100% 可靠。所以
"检出的徽章个数 == population" 就是一条不需要人工标注的判据。

为什么要有它：`_detect_badges` 的判据改过好几轮，每次都是凭一两帧拍板，
"用一两个样本定阈值 = 赌博"的教训已经出现过三次。
2026-09-23 第一次用这个脚本量，旧判据（饱和度<75）只有 37% 准确 ——
真徽章的饱和度其实在 79~95，一直被自己的过滤器扔掉。

用法：
    python -X utf8 tools/bench_badges.py
改 BADGE_MAX_V / BADGE_R / BADGE_PARAM2 前后都跑一次，只接受"准确率不降"的改动。
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agent.perception.extractors import _detect_badges  # noqa: E402

LOG_DIR = PROJECT_DIR / "data" / "logs"


def _states(run_dir: str) -> list[tuple[float, dict]]:
    path = os.path.join(run_dir, "states.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        st = rec.get("state") or {}
        out.append((st.get("timestamp") or rec.get("ts") or 0, st))
    return out


def samples():
    """(图像, 人口) —— 只要"准备阶段 + 人口读数有效 + 商店关着"的帧。"""
    for run_dir in sorted(glob.glob(str(LOG_DIR / "live_*"))):
        states = _states(run_dir)
        if not states:
            continue
        for png in glob.glob(os.path.join(run_dir, "phase_prepare_*.png")):
            m = re.search(r"_(\d+)\.png$", png)
            if not m:
                continue
            ts = int(m.group(1))
            best_ts, st = min(states, key=lambda s: abs(s[0] - ts))
            if abs(best_ts - ts) > 3:
                continue
            if not st.get("population_valid") or st.get("shop_open"):
                continue
            pop = st.get("population", 0)
            if not 0 < pop <= 7:
                continue
            img = cv2.imdecode(np.fromfile(png, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                # **实战里 MaaFW 给的是 RGB**，存盘的 png 读回来是 BGR。
                # 灰度换算对通道顺序敏感，不转的话量出来的就不是实战的数。
                yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB), pop


def main() -> int:
    c = collections.Counter()
    err = 0
    n = 0
    for img, pop in samples():
        got = len(_detect_badges(img))
        n += 1
        err += abs(got - pop)
        c["准确" if got == pop else ("漏检" if got < pop else "多检")] += 1
    if not n:
        print("没有可用的样本帧（需要 data/logs/live_*/phase_prepare_*.png + states.jsonl）")
        return 0
    print(f"样本 {n} 帧（准备阶段 / 人口有效 / 商店关着）")
    print(f"  准确 {c['准确']}/{n} ({c['准确'] / n:.0%})  "
          f"漏检 {c['漏检']}  多检 {c['多检']}  平均误差 {err / n:.2f} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
