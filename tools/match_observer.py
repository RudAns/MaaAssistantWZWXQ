"""对局观察器：整场对局每 2 秒截一帧并 OCR 顶部回合信息，供离线校准。

保存到 data/logs/match_<时间戳>/frame_NNNN.png + frames.jsonl（每帧的 OCR 摘要）。
Ctrl+C 或进程被杀即停止。不点击任何东西，纯观察。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.adb_client import connect_adb_controller
from agent.win32_client import create_tasker, screencap, ocr_once
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "logs" / f"match_{time.strftime('%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)

controller = connect_adb_controller()
tasker = create_tasker(controller, ROOT / "assets")

log = open(OUT / "frames.jsonl", "w", encoding="utf-8")
i = 0
print(f"观察器启动，输出目录: {OUT}", flush=True)
while True:
    img = screencap(controller)
    cv2.imwrite(str(OUT / f"frame_{i:04d}.png"), img)
    top = ocr_once(tasker, img, (500, 0, 400, 50), multi=True)      # 回合/倒计时
    log.write(json.dumps({"i": i, "t": time.time(), "top": top}, ensure_ascii=False) + "\n")
    log.flush()
    i += 1
    print(f"\r帧 {i}  top={top!r}   ", end="", flush=True)
    time.sleep(2)
