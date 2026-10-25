"""r9zu 数据集（2026-10-18 凌晨，f9q 题面适应，**不重标**）：r9zt 全部题按 f9q 重渲染，题号不变，沿用 r9zt_all 标签。
10-18 凌晨：训练机有一整晚可用。f9q 改的是事实写法（装备 N 选 1 写手里未上场主C 能不能穿、刘禅等给等级的人不挂 LATE-FILL、
with carry 只留分号前、满员句缩短、停用 user_ganjiang），不改标签含义 → R41a = R40 + r9zu（f9q）适应。规范裁决 + 重标留给 R42。
选项键变了的题丢弃（标签对不上）；upsample / targets 沿用 r9zt。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zu.py
"""
from __future__ import annotations

import collections
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_r9s as BR  # noqa: E402,F401
import build_r9za as BA  # noqa: E402
import rerender as RR  # noqa: E402

VER = "f9q"
BASE = "r9zt"
NAME = "r9zu"


def main() -> int:
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data/laya/datasets"
    od = ds / NAME
    od.mkdir(parents=True, exist_ok=True)
    st = collections.Counter()
    t0 = time.time()
    n_changed = 0
    with (od / "points.jsonl").open("w", encoding="utf-8") as fh:
        for line in (ds / BASE / "points.jsonl").open(encoding="utf-8"):
            p = json.loads(line)
            keys = list(p["question"]["criteria"])
            a = BA.render(p, VER, keys)
            if a is None or set(a["question"]["criteria"]) != set(keys):
                st["dropped:" + p["kind"]] += 1
                continue
            if json.dumps([a.get("state"), a["question"]], ensure_ascii=False, sort_keys=True) != \
                    json.dumps([p.get("state"), p["question"]], ensure_ascii=False, sort_keys=True):
                n_changed += 1
            a["feature_version"] = VER
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
            st["kept"] += 1
    for f in ("upsample_ids.json", "tags.json", "new_ids.json", "teacher_train_ids.json", "teacher_test_ids.json"):
        if (ds / BASE / f).exists():
            shutil.copy(ds / BASE / f, od / f)
    if (od / "targets").exists():
        shutil.rmtree(od / "targets")
    if (ds / BASE / "targets").exists():
        shutil.copytree(ds / BASE / "targets", od / "targets")
    info = {"base": BASE, "feature_version": VER, "text_changed_vs_f9p": n_changed, "stats": dict(st),
            "secs": round(time.time() - t0), "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (od / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
