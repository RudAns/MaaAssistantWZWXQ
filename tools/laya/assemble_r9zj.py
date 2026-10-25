"""r9zj 标签合并（2026-10-15，复制自 assemble_r9zh.py）：r9zh_all 全部标签 + g9zd 定向重标（f9ls 改了文本的题 + 第 633~640 局全部题），覆盖同 id。
**f9ls 改了文本、这轮没重标的题去掉旧标签**（旧标签是按 f9lr 的燃料行标的 —— R30 / R30b 用牌题退步的疑似来源），--keep-stale 保留。
输出：data/laya/labels/r9zj_all/labels_all.jsonl（R31 训练 / 评估）、stats.json。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
LAB = PROJECT_DIR / "data" / "laya" / "labels"
DS = PROJECT_DIR / "data" / "laya" / "datasets" / "r9zj"


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
    ap.add_argument("--keep-stale", action="store_true")
    args = ap.parse_args()
    old = read(LAB / "r9zh_all" / "labels_all.jsonl")
    new = {**read(LAB / "r9zj" / "labels.jsonl"), **read(LAB / "r9zj_test" / "labels.jsonl")}
    pts = {}
    for line in (DS / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    relabel = json.loads((DS / "relabel.json").read_text(encoding="utf-8"))
    tr = set(json.loads((DS / "teacher_train_ids.json").read_text(encoding="utf-8")))
    te = set(json.loads((DS / "teacher_test_ids.json").read_text(encoding="utf-8")))
    bad = [i for i, r in new.items() if i not in pts or set(r["dist"]) != set(pts[i]["options"])]
    if bad:
        print("新标签与 r9zj 选项对不上：", bad[:10])
        return 1
    merged = {i: r for i, r in old.items() if i in pts and set(r["dist"]) == set(pts[i]["options"])}
    dropped_old = len(old) - len(merged)
    fine = json.loads((DS / "relabel_fine.json").read_text(encoding="utf-8"))     # reselect_r9zj.py：只丢状态行变了（燃料行）的旧标签
    stale = [i for i, t in fine.items() if "Z_fuel" in t and i not in new and i in merged
             and pts[i]["kind"] not in ("carry", "swap")]
    if not args.keep_stale:
        for i in stale:
            merged.pop(i, None)
    merged.update(new)
    out_dir = LAB / "r9zj_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labels_all.jsonl").open("w", encoding="utf-8") as fh:
        for r in merged.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st: dict = {"merged": len(merged), "old": len(old), "old_dropped_options_changed": dropped_old, "g9zd_labels": len(new), "stale_f9ls_dropped": 0 if args.keep_stale else len(stale),
                "relabel_train": len(tr), "relabel_train_done": len(tr & set(new)),
                "relabel_test": len(te), "relabel_test_done": len(te & set(new)),
                "unlabeled_points": sum(1 for i in pts if i not in merged)}
    c = collections.Counter()
    for i, r in new.items():
        p = pts[i]
        sp = "train" if p["split"] == "train" else p["split"]
        for t in relabel.get(i, ["?"]):
            c[(sp, p["kind"], t)] += 1
            o = old.get(i)
            if o:
                c[(sp, p["kind"], t, "best_same" if o["best"] == r["best"] else "best_changed")] += 1
    st["counts"] = {"|".join(map(str, k)): v for k, v in sorted(c.items())}
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
