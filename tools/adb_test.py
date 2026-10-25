"""ADB: 点掉 60 帧提示弹窗（确定），验证触屏注入。"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.win32_client import init_toolkit
from maa.controller import AdbController
from maa.toolkit import Toolkit
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

init_toolkit()
dev = Toolkit.find_adb_devices()[0]
controller = AdbController(
    adb_path=dev.adb_path, address=dev.address,
    screencap_methods=dev.screencap_methods,
    input_methods=dev.input_methods, config=dev.config,
)
controller.post_connection().wait()

controller.post_click(784, 475).wait()   # "确定"按钮
time.sleep(1.5)
controller.post_screencap().wait()
arr = np.array(controller.cached_image)
p = ROOT / "data/logs/adb_test2.png"
p.write_bytes(cv2.imencode(".png", arr)[1].tobytes())
print(f"点击后截图: {p}", flush=True)
