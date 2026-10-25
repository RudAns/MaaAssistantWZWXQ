"""准备阶段一帧的感知耗时，**逐项拆开**计时 + 数 OCR 次数（要模拟器在线，只为绑定 Tasker）。

为什么要有它：2026-09-23 实测商店关着的准备帧**稳定 3.1 秒一帧**（25%~90% 分位都在 3.08~3.16），
而每帧最多做一个会改棋盘的动作 —— 60 秒准备时间只够做 5~10 件事，实际表现为决策太慢、
经常超过 60 秒准备时间、很多没上场。之前记录的优化后数字是 365ms，这几天加了
手牌滑窗、白字重读、徽章、装备件数、N 选 1 检测……慢了将近 10 倍，但不知道慢在哪一项。
extract() 里各项是并发跑的，一帧的墙钟 ≈ 最慢那项 + 串行部分，所以这里**每项单独串行跑**。

用法：
    python -X utf8 tools/bench_prepare_frame.py [帧目录或文件...]
默认取最近 3 局里商店关着的准备帧各 3 张。
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools"))

from test_perception_offline import FrameContext  # noqa: E402

from agent.adb_client import connect_adb_controller  # noqa: E402
from agent.perception import extractors as E  # noqa: E402
from agent.perception.state import GameState  # noqa: E402
from agent.win32_client import create_tasker  # noqa: E402

CALLS = collections.Counter()


def _count(fn, key):
    def wrapped(*a, **kw):
        CALLS[key] += 1
        return fn(*a, **kw)
    return wrapped


def frames(limit_runs=3, per_run=3):
    runs = sorted(glob.glob(str(PROJECT_DIR / "data/logs/live_*")), key=os.path.getmtime)[-limit_runs:]
    out = []
    for r in runs:
        states = {}
        p = os.path.join(r, "states.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                try:
                    st = json.loads(line).get("state") or {}
                except json.JSONDecodeError:
                    continue
                states[int(st.get("timestamp", 0))] = st
        picked = 0
        for f in sorted(glob.glob(os.path.join(r, "phase_prepare_*.png"))):
            t = int(Path(f).stem.split("_")[-1])
            st = states.get(t) or states.get(t - 1) or states.get(t + 1)
            if st and not st.get("shop_open") and picked < per_run:
                out.append(f)
                picked += 1
    return out


def main() -> int:
    files = sys.argv[1:] or frames()
    ex = E.StateExtractor(PROJECT_DIR / "assets" / "image")
    from agent.perception import onnx_ocr
    if onnx_ocr.enabled():
        ctx = None                                     # onnxruntime 后端：不连模拟器
        E._ocr_uncached = _count(E._ocr_uncached, "ocr")
        E._ocr_boxes_uncached = _count(E._ocr_boxes_uncached, "ocr")
        print(f"[OCR 后端] {onnx_ocr.OnnxOcr.get().describe()}")
    else:
        controller = connect_adb_controller()
        tasker = create_tasker(controller, PROJECT_DIR / "assets")
        ctx = FrameContext(tasker)
        # 数 OCR：在**最底层**数（_ocr_boxes / _ocr / _hit 最终都走 context.run_recognition），
        # 数上层入口会重复计数
        ctx.run_recognition = _count(ctx.run_recognition, "ocr")

    steps = {
        "detect_phase": lambda img, st: ex.detect_phase(ctx, img),
        "shop_open": lambda img, st: (E._hit(ctx, img, E.ROI_SHOP_PROB, "当前概率")
                                      or E._hit(ctx, img, E.ROI_FREE_REFRESH, "刷新")),
        "card_pick检测": lambda img, st: ex._detect_card_pick(ctx, img, st),
        "economy": lambda img, st: ex._extract_economy(ctx, img, st),
        "hand": lambda img, st: ex._extract_hand(ctx, img, st),
        "players": lambda img, st: ex._extract_players(ctx, img, st),
        "population": lambda img, st: ex._extract_population(ctx, img, st),
        "lord_level": lambda img, st: ex._extract_lord_level(img, st),
        "指定横幅": lambda img, st: E._hit(ctx, img, E.ROI_DESIGNATE, "指定"),
        "英雄面板": lambda img, st: E._hit(ctx, img, E.ROI_HERO_PANEL, "出售|剩余数量"),
        "board_units": lambda img, st: ex._extract_board_units(ctx, img, st),
        "equip_counts": lambda img, st: E.detect_equip_counts(img),
        "unit_anchors": lambda img, st: E.detect_unit_anchors(img),
    }
    tot = collections.defaultdict(float)
    ocr = collections.Counter()
    whole = []
    for f in files:
        img = cv2.cvtColor(E._imread(Path(f)), cv2.COLOR_BGR2RGB)
        ex._region_hash.clear()                 # 每张都当"第一次见"，量的是缓存未命中的代价
        E._OCR_CACHE.clear()
        st = GameState(phase="prepare")
        for name, fn in steps.items():
            before = CALLS["ocr"]
            t0 = time.perf_counter()
            fn(img, st)
            tot[name] += time.perf_counter() - t0
            ocr[name] += CALLS["ocr"] - before
        ex._region_hash.clear()
        E._OCR_CACHE.clear()
        t0 = time.perf_counter()
        ex.extract(ctx, img)
        whole.append(time.perf_counter() - t0)
        # 第二次：同一帧再跑一遍 = 缓存全命中的下限
        t0 = time.perf_counter()
        ex.extract(ctx, img)
        whole.append(-(time.perf_counter() - t0))
    n = len(files)
    print(f"{n} 帧（准备阶段，商店关着）——各项**串行**单独跑的平均耗时：")
    for name in sorted(tot, key=lambda k: -tot[k]):
        print(f"  {name:12s} {tot[name] / n * 1000:7.0f} ms   OCR {ocr[name] / n:5.1f} 次")
    cold = [w for w in whole if w > 0]
    warm = [-w for w in whole if w < 0]
    print(f"整帧 extract()（并发，真实路径）：首次 {sum(cold) / len(cold) * 1000:.0f} ms，"
          f"同一帧再跑（缓存命中）{sum(warm) / len(warm) * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
