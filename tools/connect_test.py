"""设备连接自检工具（PC 版 Win32）：找到游戏窗口 → 恢复窗口 → 连接 → 截图保存。

用法: python tools/connect_test.py
注意：游戏窗口不能处于最小化状态（FramePool 截图需要窗口可见），
本工具会尝试自动恢复最小化窗口；若游戏以管理员权限运行，本工具也需以管理员运行。
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

from maa.controller import Win32Controller
from maa.toolkit import Toolkit

WINDOW_CLASS = "UnityWndClass"      # 王者万象棋 PC 版窗口类名
WM_SYSCOMMAND = 0x0112
SC_RESTORE = 0xF120


def find_game_window():
    windows = Toolkit.find_desktop_windows()
    for w in windows:
        if w.class_name == WINDOW_CLASS and w.window_name:
            return w
    return None


def restore_if_minimized(hwnd: int) -> None:
    """窗口最小化时通过 WM_SYSCOMMAND/SC_RESTORE 恢复（跨进程可用）。"""
    user32 = ctypes.windll.user32
    if user32.IsIconic(hwnd):
        user32.PostMessageW(hwnd, WM_SYSCOMMAND, SC_RESTORE, 0)
        time.sleep(2)
        if user32.IsIconic(hwnd):
            raise RuntimeError("窗口恢复失败，请手动点开游戏窗口后重试")


def main() -> int:
    Toolkit.init_option(str(Path.cwd()))

    win = find_game_window()
    if win is None:
        print("未找到游戏窗口（UnityWndClass）。请确认王者万象棋 PC 版已启动。")
        return 1
    print(f"找到游戏窗口: title={win.window_name!r} hwnd={win.hwnd}")

    restore_if_minimized(win.hwnd)

    controller = Win32Controller(hWnd=win.hwnd)
    controller.post_connection().wait()
    if not controller.connected:
        print("连接失败。若游戏以管理员身份运行，请用管理员权限的终端重试。")
        return 1
    print("已连接")

    controller.post_screencap().wait()
    image = controller.cached_image
    if image is None:
        print("截图失败")
        return 1

    out = Path("data/logs/connect_test.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np

    cv2.imwrite(str(out), np.array(image))
    print(f"截图已保存: {out}  shape={np.array(image).shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
