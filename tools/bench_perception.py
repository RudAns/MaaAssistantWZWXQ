"""感知层性能基准：每帧的 OCR 次数与耗时分解，含阶段短路的前后对比。

对同一帧连跑两次：第一次是冷路径（阶段缓存为空，走完整判据链），
第二次是稳态（上一帧同阶段，走短路）。真实对局绝大多数帧是后者。

用法:
  .venv/Scripts/python.exe -X utf8 -u tools/bench_perception.py
需要模拟器在线（仅为绑定 Tasker，识别吃的是录像帧）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools"))

import cv2  # noqa: E402

from test_perception_offline import FrameContext  # noqa: E402

from agent.adb_client import connect_adb_controller  # noqa: E402
from agent.perception.extractors import PERF, StateExtractor  # noqa: E402
from agent.win32_client import create_tasker  # noqa: E402

FRAMES = ["frame_0039.png", "frame_0062.png", "frame_0014.png",
          "frame_0026.png", "frame_0097.png", "frame_0224.png"]


def main() -> int:
    frames_dir = PROJECT_DIR / "data" / "logs" / "match_170447"
    ctx = FrameContext(create_tasker(connect_adb_controller(), PROJECT_DIR / "assets"))
    extractor = StateExtractor(PROJECT_DIR / "assets" / "image")
    PERF.enabled = True

    print(f"{'帧':<16} {'阶段':<12} {'冷路径':>22} {'稳态(短路)':>22}  收益")
    print("-" * 88)
    totals = [0.0, 0.0, 0, 0]
    for name in FRAMES:
        image = cv2.imread(str(frames_dir / name))
        if image is None:
            print(f"[SKIP] {name}")
            continue

        extractor._phase_cache = None                 # 冷路径：清掉阶段缓存
        t0 = time.perf_counter()
        state = extractor.extract(ctx, image)
        cold_ms, cold_ocr = (time.perf_counter() - t0) * 1000, PERF.ocr_calls
        cold_phase_ms = PERF.stages.get("detect_phase", 0)

        t0 = time.perf_counter()                      # 稳态：缓存已是同一阶段
        extractor.extract(ctx, image)
        warm_ms, warm_ocr = (time.perf_counter() - t0) * 1000, PERF.ocr_calls
        warm_phase_ms = PERF.stages.get("detect_phase", 0)

        gain = (1 - warm_ms / cold_ms) * 100 if cold_ms else 0
        print(f"{name:<16} {state.phase:<12} "
              f"{cold_ms:>8.0f}ms/{cold_ocr:>2}次OCR "
              f"{warm_ms:>8.0f}ms/{warm_ocr:>2}次OCR  省 {gain:.0f}%"
              f"   (detect_phase {cold_phase_ms:.0f}→{warm_phase_ms:.0f}ms)")
        print(f"                 稳态分解: {PERF.summary()}")
        totals[0] += cold_ms
        totals[1] += warm_ms
        totals[2] += cold_ocr
        totals[3] += warm_ocr

    n = len(FRAMES)
    print("-" * 88)
    print(f"{'平均':<16} {'':<12} {totals[0]/n:>8.0f}ms/{totals[2]/n:>4.1f}次OCR "
          f"{totals[1]/n:>8.0f}ms/{totals[3]/n:>4.1f}次OCR  "
          f"省 {(1 - totals[1]/totals[0])*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
