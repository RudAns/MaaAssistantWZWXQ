"""r9zb 第二轮批次（2026-10-13）：f9lm 题面修了 bug（FINAL carry settled / TEMPORARY 收紧、天赋派系计数、升级算术、刷新 wanted 去掉临时主C）之后，
已标完的题里**题面文本变了的**和**还没标的**重新组批，导到 data/laya/labels/r9zb_r2/、r9zb_r2_test/。已标且文本没变的保留原标签。

  .venv/Scripts/python.exe -X utf8 tools/laya/rebatch_r9zb.py          # 写 datasets/r9zb/rebatch_r2_{train,test}_ids.json + 统计
  然后：label_batches.py export --ver r9zb --ids-file data/laya/datasets/r9zb/rebatch_r2_train_ids.json --out-ver r9zb_r2 ...

前提：datasets/r9zb_v1/ 是第一轮批次对应的旧题面（points.jsonl + teacher_*_ids.json），datasets/r9zb/ 已按修好的 facts 重建；
重标清单沿用第一轮的 teacher_*_ids（不按新标签重抽，免得已标的白标），写回 datasets/r9zb/teacher_*_ids.json。
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DS = PROJECT_DIR / "data" / "laya" / "datasets"
LAB = PROJECT_DIR / "data" / "laya" / "labels"


def _pts(d: Path) -> dict:
    out = {}
    for line in (d / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        out[p["id"]] = p
    return out


def _text(p: dict) -> str:
    return json.dumps([p.get("state"), p["question"].get("criteria"), p["question"].get("instructions")], ensure_ascii=False)


def _labeled(ver: str) -> set:
    ids = set()
    for f in glob.glob(str(LAB / ver / "out" / "*.jsonl")):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
    return ids


def main() -> int:
    old, new = _pts(DS / "r9zb_v1"), _pts(DS / "r9zb")
    tr = json.loads((DS / "r9zb_v1" / "teacher_train_ids.json").read_text(encoding="utf-8"))
    te = json.loads((DS / "r9zb_v1" / "teacher_test_ids.json").read_text(encoding="utf-8"))
    (DS / "r9zb" / "teacher_train_ids.json").write_text(json.dumps(tr), encoding="utf-8")
    (DS / "r9zb" / "teacher_test_ids.json").write_text(json.dumps(te), encoding="utf-8")
    done = _labeled("r9zb") | _labeled("r9zb_test")
    changed_all = {i for i in old if i in new and _text(old[i]) != _text(new[i])}
    out, st = {}, {}
    for name, ids in (("train", tr), ("test", te)):
        lab = [i for i in ids if i in done]
        ch = [i for i in lab if i in changed_all or i not in new]
        todo = sorted(set(i for i in ids if i not in done) | set(ch))
        out[name] = todo
        st[name] = {"selected": len(ids), "labeled_v1": len(lab), "labeled_text_changed": len(ch),
                    "labeled_kept": len(lab) - len(ch), "unlabeled": len(ids) - len(lab), "rebatch": len(todo),
                    "changed_ids": sorted(ch)}
        (DS / "r9zb" / f"rebatch_r2_{name}_ids.json").write_text(json.dumps(todo), encoding="utf-8")
    st["text_changed_whole_dataset"] = len(changed_all)
    (DS / "r9zb" / "rebatch_r2_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "changed_ids"} if isinstance(v, dict) else v)
                      for k, v in st.items()}, ensure_ascii=False, indent=1))
    print("题面变了的已标题：", st["train"]["changed_ids"] + st["test"]["changed_ids"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
