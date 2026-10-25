"""调试：用 Seize 输入方式点菜单页的"百科"。"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.win32_client import connect_controller, create_tasker, screencap, ocr_once
from maa.define import MaaWin32InputMethodEnum
import cv2

ROOT = Path(__file__).resolve().parent.parent

def imwrite(name, img):
    p = ROOT / "data/logs" / name
    p.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    print(f"saved {name}")

controller = connect_controller(input_method=MaaWin32InputMethodEnum.Seize)
tasker = create_tasker(controller, ROOT / "assets")

controller.post_click(236, 320).wait()
time.sleep(2.5)
img = screencap(controller)
imwrite("dbg6_seize_baike.png", img)
head = ocr_once(tasker, img, (20, 15, 150, 45))
print(f"页头 OCR: {head!r}")
