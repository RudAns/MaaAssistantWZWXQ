"""r9y 标签合并（2026-10-11，复制自 assemble_r9w.py）：r9w_all 全部标签 + g9v 定向重标（覆盖同 id）。
以下为 r9w 原说明：
r9w 标签合并（2026-10-10，docs/laya-r22-plan-2026-10-10.md）：r9s 全部标签 + g9u 定向重标 / 新标（覆盖同 id 的旧标签）。

  .venv/Scripts/python.exe -X utf8 tools/laya/label_batches.py merge --ver r9w --guide g9u
  .venv/Scripts/python.exe -X utf8 tools/laya/label_batches.py merge --ver r9w_test --guide g9u
  .venv/Scripts/python.exe -X utf8 tools/laya/assemble_r9w.py [--min-frac 0.9]

输出：data/laya/labels/r9w_all/labels_all.jsonl（R23 训练 / 评估）、data/laya/labels/r9w_all/stats.json。
没标完（重标清单覆盖率 < --min-frac，测试集 < 100%）时退出码 1。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
LAB = PROJECT_DIR / "data" / "laya" / "labels"
DS = PROJECT_DIR / "data" / "laya" / "datasets" / "r9y"


def read(f: Path) -> dict:
    out = {}
    if f.exists():
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-frac", type=float, default=0.9, help="训练重标清单至少标完多少比例")
    args = ap.parse_args()
    old = read(LAB / "r9w_all" / "labels_all.jsonl")
    new = {**read(LAB / "r9y" / "labels.jsonl"), **read(LAB / "r9y_test" / "labels.jsonl")}
    # 第二轮（题面修了同槽 / 玄策搭档 / 沈梦溪牌槽后选项变了的 18 题 + 新进清单的 16 题）：第一轮标签作废，用 r9y_r2
    _r2 = set(json.loads((DS / "relabel_round2_ids.json").read_text(encoding="utf-8")))
    new = {k: v for k, v in new.items() if k not in _r2}
    new.update(read(LAB / "r9y_r2" / "labels.jsonl"))
    # 第三轮：等上场的人是 MIN ENGINE 缺件 / LATE-FILL 的换人题（第一轮按选项文字挑题漏掉了，旧标签与 g9v 冲突）
    new.update(read(LAB / "r9y_r3" / "labels.jsonl"))
    # 2026-10-11：C-POSITION 定义收窄后题面变了的 372 题，第一批 g9u 标签作废（训练题退回旧标签，测试题用 r9w_test2 重标）
    pts = {}
    for line in (DS / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    relabel = json.loads((DS / "relabel.json").read_text(encoding="utf-8"))
    tr = set(json.loads((DS / "relabel_train_ids.json").read_text(encoding="utf-8")))
    te = set(json.loads((DS / "relabel_test_ids.json").read_text(encoding="utf-8")))
    bad = [i for i, r in new.items() if i not in pts or set(r["dist"]) != set(pts[i]["options"])]
    if bad:
        print("新标签与 r9y 选项对不上：", bad[:10])
        return 1
    merged = {i: r for i, r in old.items() if i in pts and set(r["dist"]) == set(pts[i]["options"])}
    dropped_old = len(old) - len(merged)
    merged.update(new)
    out_dir = LAB / "r9y_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labels_all.jsonl").open("w", encoding="utf-8") as fh:
        for r in merged.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    st: dict = {"merged": len(merged), "old": len(old), "old_dropped_not_in_r9w": dropped_old, "g9u_labels": len(new),
                "relabel_train": len(tr), "relabel_train_done": len(tr & set(new)),
                "relabel_test": len(te), "relabel_test_done": len(te & set(new))}
    c = collections.Counter()
    mass = collections.defaultdict(list)
    for i, r in new.items():
        p = pts[i]
        sp = "train" if p["split"] == "train" else p["split"]
        for t in relabel.get(i, ["?"]):
            c[(sp, p["kind"], t)] += 1
            o = old.get(i)
            if o:
                c[(sp, p["kind"], t, "best_same" if o["best"] == r["best"] else "best_changed")] += 1
            if t == "A_refresh_cant_afford":
                mass["refresh_new"].append(r["dist"].get("refresh", 0.0))
                if o:
                    mass["refresh_old"].append(o["dist"].get("refresh", 0.0))
    st["counts"] = {"|".join(map(str, k)): v for k, v in sorted(c.items())}
    st["refresh_mass_A"] = {k: round(sum(v) / len(v), 3) for k, v in mass.items() if v}
    st["mean_conf"] = round(sum(float(r.get("confidence") or 0) for r in new.values()) / max(1, len(new)), 3)
    st["missing_info_top"] = collections.Counter(m for r in new.values() for m in (r.get("missing_info") or [])).most_common(20)
    (out_dir / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in st.items() if k != "counts"}, ensure_ascii=False, indent=1))
    ok = st["relabel_train_done"] >= args.min_frac * max(1, len(tr)) and st["relabel_test_done"] == len(te)
    if not ok:
        print(f"标注未完成：训练 {st['relabel_train_done']}/{len(tr)}（要求 ≥{args.min_frac:.0%}），测试 {st['relabel_test_done']}/{len(te)}（要求全部）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
