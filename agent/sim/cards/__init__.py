"""卡牌实现：导入即注册（engine.REGISTRY）。每批一个模块（batch_00 … batch_07 由分批任务写入）。"""
import importlib
import logging

from . import pilot  # noqa: F401

for _i in range(8):
    try:
        importlib.import_module(f"{__name__}.batch_{_i:02d}")
    except ModuleNotFoundError:
        pass
    except Exception:                                    # noqa: BLE001
        logging.getLogger(__name__).exception("卡牌实现 batch_%02d 导入失败", _i)
