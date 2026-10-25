"""ADB（模拟器）客户端连接助手：找设备 → 连接 → 截图。

ADB 通道完全后台：不占鼠标、不抢前台，模拟器窗口最小化也能工作。
与 win32_client 二选一，由工具脚本的 --backend 参数决定。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from maa.controller import AdbController
from maa.toolkit import Toolkit

from agent.win32_client import init_toolkit  # 复用 Toolkit 初始化


def find_adb_device(address: Optional[str] = None):
    """返回第一个（或指定地址的）ADB 设备对象，未找到返回 None。"""
    init_toolkit()
    devices = Toolkit.find_adb_devices()
    if address:
        for d in devices:
            if d.address == address:
                return d
        return None
    return devices[0] if devices else None


def connect_adb_controller(address: Optional[str] = None) -> AdbController:
    """连接模拟器，返回已连接的 AdbController。"""
    dev = find_adb_device(address)
    if dev is None:
        raise RuntimeError("未发现 ADB 设备（模拟器未启动？MuMu 默认端口 16384/16416）")
    controller = AdbController(
        adb_path=dev.adb_path,
        address=dev.address,
        screencap_methods=dev.screencap_methods,
        input_methods=dev.input_methods,
        config=dev.config,
    )
    controller.post_connection().wait()
    if not controller.connected:
        raise RuntimeError(f"ADB 连接失败: {dev.address}")
    return controller
