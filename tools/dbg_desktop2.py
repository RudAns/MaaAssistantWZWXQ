"""桌面2窗口: 隔5秒两帧 PrintWindow 截图对比，判断游戏是否仍在渲染。"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.win32_client import find_game_window
from maa.define import MaaWin32InputMethodEnum, MaaWin32ScreencapMethodEnum
from maa.controller import Win32Controller
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

win = find_game_window()
ctl = Win32Controller(
    hWnd=win.hwnd,
    screencap_method=MaaWin32ScreencapMethodEnum.PrintWindow,
    mouse_method=MaaWin32InputMethodEnum.Seize,
    keyboard_method=MaaWin32InputMethodEnum.Seize,
)
ctl.post_connection().wait()

frames = []
for i in range(3):
    ctl.post_screencap().wait()
    arr = np.array(ctl.cached_image)
    frames.append(arr)
    (ROOT / f"data/logs/dbg9_frame{i}.png").write_bytes(cv2.imencode(".png", arr)[1].tobytes())
    time.sleep(5)

d01 = np.abs(frames[0].astype(int) - frames[1].astype(int)).mean()
d12 = np.abs(frames[1].astype(int) - frames[2].astype(int)).mean()
print(f"帧间差异: f0-f1={d01:.2f}  f1-f2={d12:.2f}  (>1 表示画面在动/仍在渲染)", flush=True)
