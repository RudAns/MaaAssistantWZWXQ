"""R52 标注批次导出（2026-10-21 晚，f9zb / g9zzd = 规范 54.29~54.36）：同 R51（r9zzg）的参数 —— --batch 40 --dup 0.1 --seed 7，
训练 = teacher_train_ids.json → labels/r9zzh，测试 = teacher_test_ids.json → labels/r9zzh_test。

label_batches.GUIDE_VERSION 还是旧值（线上对局期间不改那份文件）→ 这里在导出前把它设成 g9zzd（批次标头 / manifest 的规范版本）。
merge 时显式传 --guide g9zzd（retrain_r52_home.sh 已写）。

  .venv/Scripts/python.exe -X utf8 tools/laya/export_r9zzh.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label_batches as LB  # noqa: E402

GUIDE = "g9zzd"


def main() -> int:
    LB.GUIDE_VERSION = GUIDE
    for ids, out in (("teacher_train_ids.json", "r9zzh"), ("teacher_test_ids.json", "r9zzh_test")):
        root = LB.PROJECT_DIR / "data/laya/labels" / out / "batches"
        if root.exists() and any(root.iterdir()):
            print(f"{out} 已有批次，不覆盖（要重导先手动删 data/laya/labels/{out}/）")
            continue
        sys.argv = ["label_batches.py", "export", "--ver", "r9zzh", "--ids-file", f"data/laya/datasets/r9zzh/{ids}",
                    "--out-ver", out, "--batch", "40", "--dup", "0.1", "--seed", "7", "--shop", "100000"]   # shop 默认封顶 800（r9zzh 训练 936 道）
        rc = LB.main()
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
