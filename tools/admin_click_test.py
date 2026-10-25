"""管理员权限点击测试（由 Start-Process -Verb RunAs 拉起）。

结果写入 data/logs/admin_click_test.txt。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "data" / "logs" / "admin_click_test.txt"


def log(msg: str) -> None:
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


try:
    from agent.win32_client import (
        connect_controller, create_tasker, find_game_window, ocr_once, screencap,
    )

    import ctypes
    is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    log(f"IsUserAnAdmin: {is_admin}")

    win = find_game_window()
    controller = connect_controller(win.hwnd)
    tasker = create_tasker(controller, "assets")

    img0 = screencap(controller)
    before = ocr_once(tasker, img0, (985, 240, 180, 60))
    log(f"点击前: {before!r}")

    controller.post_click(73, 509).wait()   # 苏烈
    time.sleep(2)
    img1 = screencap(controller)
    after = ocr_once(tasker, img1, (985, 240, 180, 60))
    log(f"点击后: {after!r}")
    log(f"结果: {'点击生效!' if before != after and after else '仍无效'}")
except Exception as e:
    log(f"异常: {type(e).__name__}: {e}")
