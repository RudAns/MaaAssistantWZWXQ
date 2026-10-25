"""测试 WithWindowPos 变体 + 滚轮行为（后台、是否动真实鼠标）。"""
import ctypes
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.win32_client import connect_controller, create_tasker, screencap, ocr_once
from maa.define import MaaWin32InputMethodEnum

ROOT = Path(__file__).resolve().parent.parent
CARD_A = (75, 195)    # 干将莫邪
CARD_B = (173, 195)   # 鬼谷子
ROI_NAME = (985, 240, 180, 60)

user32 = ctypes.windll.user32

def cursor_pos():
    class P(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    p = P()
    user32.GetCursorPos(ctypes.byref(p))
    return (p.x, p.y)

for method_name in ["SendMessageWithWindowPos", "PostMessageWithWindowPos"]:
    method = getattr(MaaWin32InputMethodEnum, method_name)
    try:
        controller = connect_controller(input_method=method)
    except Exception as e:
        print(f"{method_name}: 连接失败 {e}")
        continue
    tasker = create_tasker(controller, ROOT / "assets")
    controller.post_click(*CARD_A).wait()
    time.sleep(1.5)
    name_a = ocr_once(tasker, screencap(controller), ROI_NAME)
    controller.post_click(*CARD_B).wait()
    time.sleep(1.5)
    name_b = ocr_once(tasker, screencap(controller), ROI_NAME)
    ok = name_a != name_b and name_b
    print(f"{method_name}: A={name_a!r} B={name_b!r} -> {'生效' if ok else '无效'}", flush=True)

# 滚轮测试：Seize 下滑动一格，看鼠标是否移动、列表是否滚动
controller = connect_controller(input_method=MaaWin32InputMethodEnum.Seize)
tasker = create_tasker(controller, ROOT / "assets")
before = cursor_pos()
img0 = screencap(controller)
labels0 = ocr_once(tasker, img0, (30, 150, 905, 510), multi=True)
for dy in (-120, -240, -480):       # 尝试不同滚轮幅度
    controller.post_scroll(0, dy).wait()
    time.sleep(1.0)
    img = screencap(controller)
    labels = ocr_once(tasker, img, (30, 150, 905, 510), multi=True)
    print(f"scroll dy={dy}: 鼠标 {before}->{cursor_pos()}  网格首行变化: {labels0.split(chr(10))[:3]} -> {labels.split(chr(10))[:3]}", flush=True)
    labels0 = labels
