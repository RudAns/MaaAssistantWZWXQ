"""OCR 区域校准：对百科详情面板的候选 ROI 跑 OCR 验证。

用法: python tools/ocr_calib.py
对真实游戏窗口只做截图 + OCR，不点击、不改动任何状态，可安全反复运行。
（DbgController 的 DLL 未随 pip 包分发，因此直接用 Win32 控制器做只读探测。）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.win32_client import connect_controller, create_tasker, ocr_once, screencap  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# 百科详情面板候选 ROI (x, y, w, h)，基于 1186x720 截图测量
CANDIDATE_ROIS = {
    "card_name": (995, 245, 160, 55),       # 大号卡牌名（项羽）
    "card_tags": (943, 398, 245, 45),       # 前排/坦克/出装 标签行
    "skill_text": (943, 305, 245, 90),      # 技能描述文本
    "faction": (990, 368, 180, 35),         # -大河流域- 派系
    "grid_first_name": (42, 278, 92, 30),   # 网格第1格名称标签（干将莫邪）
    "collect_count": (795, 110, 200, 30),   # 已收集 85/85
}


def main() -> int:
    controller = connect_controller()
    image = screencap(controller)
    print(f"已连接游戏窗口，截图 shape={image.shape}")

    tasker = create_tasker(controller, ASSETS)
    for name, roi in CANDIDATE_ROIS.items():
        text = ocr_once(tasker, image, roi)
        print(f"[{name:>16}] roi={roi} -> {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
