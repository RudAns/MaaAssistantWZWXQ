"""r9zb 标签合并（2026-10-13，复制自 assemble_r9za.py）：r9za_all 全部标签 + g9y 定向重标（f9lm：临时主C / 候选路线核心卡 /
天赋派系 / 空位；新局 588~605 全部题），覆盖同 id。换主C 题（carry）的局面网络用模拟器标签，不在这里。

  .venv/Scripts/python.exe -X utf8 tools/laya/label_batches.py merge --ver r9zb --guide g9y
  .venv/Scripts/python.exe -X utf8 tools/laya/label_batches.py merge --ver r9zb_test --guide g9y
  .venv/Scripts/python.exe -X utf8 tools/laya/label_batches.py merge --ver r9zb_r2 --guide g9y
  .venv/Scripts/python.exe -X utf8 tools/laya/label_batches.py merge --ver r9zb_r2_test --guide g9y
  .venv/Scripts/python.exe -X utf8 tools/laya/assemble_r9zb.py [--min-frac 0.9]

输出：data/laya/labels/r9zb_all/labels_all.jsonl（R27 训练 / 评估）、stats.json。
退出码 1：新标签与选项对不上 / **被选进重标清单的 TEMPORARY、CORE of route 训练题没有全部是 g9y 标签** / 重标清单没标完。
（省 token 版：carry / swap 不重标；旧题按标签抽样封顶，没选中的保留旧标签，见 build_r9zb.select_relabel）
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
LAB = PROJECT_DIR / "data" / "laya" / "labels"
DS = PROJECT_DIR / "data" / "laya" / "datasets" / "r9zb"
NEW_TAGS = ("TEMPORARY", "CORE of route")          # 训练集里选项带这两类标签的题必须全部是新规范（g9y）标签
SOFT_TAGS = ("our 【", "fills an empty slot", "board full: needs a swap")   # 只报告覆盖率


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
    old = read(LAB / "r9za_all" / "labels_all.jsonl")
    new = {**read(LAB / "r9zb" / "labels.jsonl"), **read(LAB / "r9zb_test" / "labels.jsonl")}
    # 第二轮（f9lm 题面修 bug 后）：题面变了的已标题 + 第一轮没标的题，重导到 r9zb_r2 / r9zb_r2_test；
    # 这些 id 的第一轮标签（按旧题面标的，或第一轮批次晚到的）一律作废，只认第二轮标签
    _r2 = set()
    for f in ("rebatch_r2_train_ids.json", "rebatch_r2_test_ids.json"):
        if (DS / f).exists():
            _r2 |= set(json.loads((DS / f).read_text(encoding="utf-8")))
    new = {k: v for k, v in new.items() if k not in _r2}
    new.update({**read(LAB / "r9zb_r2" / "labels.jsonl"), **read(LAB / "r9zb_r2_test" / "labels.jsonl")})
    pts = {}
    for line in (DS / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    relabel = json.loads((DS / "relabel.json").read_text(encoding="utf-8"))
    tr = set(json.loads((DS / "teacher_train_ids.json").read_text(encoding="utf-8")))
    te = set(json.loads((DS / "teacher_test_ids.json").read_text(encoding="utf-8")))
    bad = [i for i, r in new.items() if i not in pts or set(r["dist"]) != set(pts[i]["options"])]
    if bad:
        print("新标签与 r9zb 选项对不上：", bad[:10])
        return 1
    merged = {i: r for i, r in old.items() if i in pts and set(r["dist"]) == set(pts[i]["options"])}
    dropped_old = len(old) - len(merged)
    merged.update(new)
    out_dir = LAB / "r9zb_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labels_all.jsonl").open("w", encoding="utf-8") as fh:
        for r in merged.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    st: dict = {"merged": len(merged), "old": len(old), "old_dropped_options_changed": dropped_old, "g9y_labels": len(new),
                "relabel_train": len(tr), "relabel_train_done": len(tr & set(new)),
                "relabel_test": len(te), "relabel_test_done": len(te & set(new))}
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
    # **新标签覆盖检查**（R25 教训；省 token 版，协调方 2026-10-13 定）：
    #   带 TEMPORARY / CORE of route 的训练题里，**被选进重标清单的**必须全部是 g9y 标签；没被选中的旧题保留旧标签。
    #   carry / swap 不进这轮 Laya 重标（线上由局面网络回答），不检查。同时报告新标签题里新 / 旧标签的比例。
    tag_tot, tag_new, tag_old, tag_sel, tag_sel_new = (collections.Counter() for _ in range(5))
    for i, p in pts.items():
        if p["split"] != "train" or p["kind"] in ("carry", "swap"):
            continue
        for t in NEW_TAGS + SOFT_TAGS:
            if any(t in v for v in p["question"]["criteria"].values()):
                tag_tot[t] += 1
                if i in new:
                    tag_new[t] += 1
                elif i in merged:
                    tag_old[t] += 1
                if i in tr:
                    tag_sel[t] += 1
                    tag_sel_new[t] += i in new
    st["new_tag_coverage"] = {t: {"train_with_tag": tag_tot[t], "g9y_label": tag_new[t], "old_label": tag_old[t],
                                  "unlabeled": tag_tot[t] - tag_new[t] - tag_old[t],
                                  "g9y_share_of_labeled": round(tag_new[t] / max(1, tag_new[t] + tag_old[t]), 3),
                                  "selected_done": f"{tag_sel_new[t]}/{tag_sel[t]}"} for t in NEW_TAGS + SOFT_TAGS}
    (out_dir / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in st.items() if k not in ("counts", "new_tag_coverage")}, ensure_ascii=False, indent=1))
    print("新标签题（训练、非 carry/swap）的新 / 旧标签：")
    for t, v in st["new_tag_coverage"].items():
        print(f"  {t}: {v}")
    if any(tag_sel_new[t] < tag_sel[t] for t in NEW_TAGS):
        print("新标签覆盖不足：被选进重标清单的 TEMPORARY / CORE of route 训练题必须全部是 g9y 标签")
        return 1
    ok = st["relabel_train_done"] >= args.min_frac * max(1, len(tr)) and st["relabel_test_done"] == len(te)
    if not ok:
        print(f"标注未完成：训练 {st['relabel_train_done']}/{len(tr)}（要求 ≥{args.min_frac:.0%}），测试 {st['relabel_test_done']}/{len(te)}（要求全部）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
