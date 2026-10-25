"""Win32 游戏客户端连接助手：find window → restore → connect → (可选) 创建 Tasker。

供 tools/connect_test.py、tools/update_card_db.py 及 agent 复用。
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Optional

from maa.controller import Win32Controller
from maa.define import MaaWin32InputMethodEnum, MaaWin32ScreencapMethodEnum
from maa.resource import Resource
from maa.tasker import Tasker
from maa.toolkit import Toolkit

WINDOW_CLASS = "UnityWndClass"      # 王者万象棋 PC 版窗口类名
WM_SYSCOMMAND = 0x0112
SC_RESTORE = 0xF120

# 默认输入方式：Seize（直接向窗口投递输入消息，后台工作、不抢真实鼠标；需管理员权限）。
# 注意：AnchoredTouch 在本游戏实测无效（点击无反应），LegacyEvent 会抢真实鼠标。
DEFAULT_INPUT_METHOD = MaaWin32InputMethodEnum.Seize

_initialized = False


def init_toolkit(user_path: str | Path = ".") -> None:
    global _initialized
    if not _initialized:
        Toolkit.init_option(str(user_path))
        _initialized = True


def find_game_window():
    """返回游戏窗口对象（含 hwnd/window_name/class_name），未找到返回 None。"""
    init_toolkit()
    for w in Toolkit.find_desktop_windows():
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


def connect_controller(hwnd: Optional[int] = None,
                       input_method: Optional[int] = None) -> Win32Controller:
    """连接游戏窗口，返回已连接的 Win32Controller。

    input_method 默认 AnchoredTouch（后台输入，不抢鼠标），需管理员权限。
    """
    if hwnd is None:
        win = find_game_window()
        if win is None:
            raise RuntimeError(f"未找到游戏窗口（{WINDOW_CLASS}），请确认王者万象棋已启动")
        hwnd = win.hwnd
    restore_if_minimized(hwnd)
    method = input_method if input_method is not None else DEFAULT_INPUT_METHOD
    controller = Win32Controller(
        hWnd=hwnd,
        screencap_method=MaaWin32ScreencapMethodEnum.FramePool,
        mouse_method=method,
        keyboard_method=method,
    )
    controller.post_connection().wait()
    if not controller.connected:
        raise RuntimeError("连接失败。若游戏以管理员身份运行，请用管理员权限的终端重试")
    return controller


def create_tasker(controller, resource_path: str | Path) -> Tasker:
    """创建绑定了资源与控制器的 Tasker（可使用 pipeline/OCR 识别）。"""
    resource = Resource()
    resource.post_bundle(str(resource_path)).wait()
    if not resource.loaded:
        raise RuntimeError(f"资源加载失败: {resource_path}")
    tasker = Tasker()
    tasker.bind(resource, controller)
    if not tasker.inited:
        raise RuntimeError("Tasker 初始化失败")
    return tasker


def screencap(controller) -> "object":
    """截图并返回图像对象（可用 np.array() 转 ndarray）。"""
    import numpy as np

    controller.post_screencap().wait()
    image = controller.cached_image
    if image is None:
        raise RuntimeError("截图失败")
    return np.array(image)


def ocr_once(tasker, image, roi: tuple[int, int, int, int], expected: str = "",
             multi: bool = False) -> str:
    """在指定 ROI 上执行一次 OCR，返回识别文本。

    通过临时 pipeline 节点驱动 MaaFramework 内置 OCR（使用 assets/model/ocr 模型）。
    multi=True 时把所有识别到的文本块按从上到下顺序拼接（多行文本用）。
    """
    node = {
        "recognition": "OCR",
        "roi": list(roi),
        "action": "DoNothing",
    }
    if expected:
        node["expected"] = expected
    task_detail = tasker.post_task("__ocr_probe__", {"__ocr_probe__": node}).wait().get()
    if task_detail is None:
        return ""
    for node_detail in task_detail.nodes:
        reco = node_detail.recognition
        if reco is None or not reco.hit:
            continue
        if not multi:
            if reco.best_result is not None:
                return reco.best_result.text.strip()
            return ""
        results = reco.filtered_results or reco.all_results or []
        results = sorted(results, key=lambda r: (r.box[1], r.box[0]))
        return "\n".join(r.text.strip() for r in results if r.text.strip())
    return ""
