"""后台输入方式测试（需管理员权限运行）。

逐个测试不抢占鼠标的输入方式，点击一张卡后校验详情面板是否切换。
结果写入 data/logs/bg_input_test.txt。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "data" / "logs" / "bg_input_test.txt"


def log(msg: str) -> None:
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


try:
    from maa.controller import Win32Controller
    from maa.define import MaaWin32InputMethodEnum, MaaWin32ScreencapMethodEnum

    from agent.win32_client import (
        create_tasker, find_game_window, ocr_once, restore_if_minimized, screencap,
    )

    win = find_game_window()
    restore_if_minimized(win.hwnd)

    # 后台输入候选（Bg=Yes，不抢鼠标）
    methods = {
        "PostMessage": MaaWin32InputMethodEnum.PostMessage,
        "SendMessage": MaaWin32InputMethodEnum.SendMessage,
        "AnchoredTouch": MaaWin32InputMethodEnum.AnchoredTouch,
        "PostMessageWithCursorPos": MaaWin32InputMethodEnum.PostMessageWithCursorPos,
    }

    for name, m in methods.items():
        ctrl = Win32Controller(
            hWnd=win.hwnd,
            screencap_method=MaaWin32ScreencapMethodEnum.FramePool,
            mouse_method=m, keyboard_method=m,
        )
        ctrl.post_connection().wait()
        if not ctrl.connected:
            log(f"{name}: 连接失败")
            continue
        tasker = create_tasker(ctrl, "assets")
        before = ocr_once(tasker, screencap(ctrl), (985, 240, 180, 60))
        ctrl.post_click(73, 344).wait()     # 韩信头像
        time.sleep(2)
        after = ocr_once(tasker, screencap(ctrl), (985, 240, 180, 60))
        ok = bool(after) and before != after
        log(f"{name}: {before!r} -> {after!r}  {'后台点击生效!' if ok else '无效'}")
        if ok:
            log(f"推荐后台输入方式: {name}")
            break
except Exception as e:
    log(f"异常: {type(e).__name__}: {e}")
