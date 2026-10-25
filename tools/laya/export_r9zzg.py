"""R51 标注批次导出（2026-10-21，f9za / g9zzc = 规范 54.29~54.31）：同 R50（r9zzf）的参数 —— --batch 40 --dup 0.1 --seed 7，
训练 = teacher_train_ids.json → labels/r9zzg，测试 = teacher_test_ids.json → labels/r9zzg_test。

label_batches.GUIDE_VERSION 还是 g9zza（线上对局期间不改那份文件）→ 这里在导出前把它设成 g9zzc（批次标头 / manifest 的规范版本）。
merge 时显式传 --guide g9zzc（retrain_r51_home.sh 已写）。

  .venv/Scripts/python.exe -X utf8 tools/laya/export_r9zzg.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label_batches as LB  # noqa: E402

GUIDE = "g9zzc"


def main() -> int:
    LB.GUIDE_VERSION = GUIDE
    for ids, out in (("teacher_train_ids.json", "r9zzg"), ("teacher_test_ids.json", "r9zzg_test")):
        root = LB.PROJECT_DIR / "data/laya/labels" / out / "batches"
        if root.exists() and any(root.iterdir()):
            print(f"{out} 已有批次，不覆盖（要重导先手动删 data/laya/labels/{out}/）")
            continue
        sys.argv = ["label_batches.py", "export", "--ver", "r9zzg", "--ids-file", f"data/laya/datasets/r9zzg/{ids}",
                    "--out-ver", out, "--batch", "40", "--dup", "0.1", "--seed", "7"]
        rc = LB.main()
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
