"""r9n 标注批次（2026-09-30 晚）：把 build_r9n.py 选出来的新题切成老师批次，格式与 label_batches.py export 完全相同。

  python tools/laya/export_r9n_batches.py --ds r9n

- 老师视图 = label_batches.render_point（规范 g9j；f6 起老师看的就是学生输入原文，逐字同源）。
- 三组批次，文件名前缀区分（label_batches.py merge 照常能合并：非 dup_ 前缀都算主标注）：
    batch_XXX   新题里 split=train 的（new_ids.json）
    test_XXX    新题里 split=test_r9n_new 的（新测试集，单独成批便于先标 / 复核）
    carryx_XXX  离线补出的 carry 题（carryx_ids.json；线上这一回合没出 carry 题的回合），可选标
- 每批 40 题（同 r9m），批内题型打散（seed 固定）。
- 写 data/laya/labels/<ds>/batches/*.md + *.ids.json、out/（空）、manifest.json、batches.json（每批路径 / 题数 / 题型 / 划分分布）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))

import label_batches as LB  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="r9n")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dsd = PROJECT_DIR / "data" / "laya" / "datasets" / args.ds
    pts = {p["id"]: p for p in LB.load_points(args.ds)}
    need = json.loads((dsd / "new_ids.json").read_text(encoding="utf-8"))
    carryx = json.loads((dsd / "carryx_ids.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    groups = {
        "batch": [pts[i] for i in need if pts[i]["split"] == "train"],
        "test": [pts[i] for i in need if pts[i]["split"] != "train"],
        "carryx": [pts[i] for i in carryx],
    }
    root = PROJECT_DIR / "data" / "laya" / "labels" / args.ds
    (root / "batches").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)
    listing = []
    for prefix, rows in groups.items():
        rows = sorted(rows, key=lambda p: p["id"])
        rng.shuffle(rows)
        for b in range(0, len(rows), args.batch):
            part = rows[b:b + args.batch]
            name = f"{prefix}_{b // args.batch:03d}"
            md = [f"# 标注批次 {name}（{len(part)} 题，规范 {LB.GUIDE_VERSION}）\n"] + [LB.render_point(p) + "\n" for p in part]
            (root / "batches" / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
            (root / "batches" / f"{name}.ids.json").write_text(json.dumps(
                {p["id"]: p["options"] for p in part}, ensure_ascii=False), encoding="utf-8")
            listing.append({"name": name, "group": prefix,
                            "md": f"data/laya/labels/{args.ds}/batches/{name}.md",
                            "ids": f"data/laya/labels/{args.ds}/batches/{name}.ids.json",
                            "out": f"data/laya/labels/{args.ds}/out/{name}.jsonl",
                            "n": len(part),
                            "by_kind": dict(sorted(collections.Counter(p["kind"] for p in part).items())),
                            "by_split": dict(collections.Counter(p["split"] for p in part))})
    summary = {g: {"n": len(rows), "batches": sum(1 for x in listing if x["group"] == g),
                   "by_kind": dict(sorted(collections.Counter(p["kind"] for p in rows).items()))}
               for g, rows in groups.items()}
    manifest = {"ver": args.ds, "guide": LB.GUIDE_VERSION, "seed": args.seed, "batch_size": args.batch,
                "n": sum(len(r) for r in groups.values()), "batches": len(listing), "dup_n": 0, "dup_batches": 0,
                "groups": summary,
                "teacher_prompt": "docs/laya-report/teacher-prompt-g9j.txt",
                "by_kind": dict(collections.Counter(p["kind"] for r in groups.values() for p in r))}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "batches.json").write_text(json.dumps({"summary": summary, "batches": listing}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
