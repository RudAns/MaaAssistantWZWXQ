"""r9z 标签合并（2026-10-12，复制自 assemble_r9y.py）：r9y_all 全部标签 + g9w 定向重标（f9lk 主C 资格 / 备选主C / 人口，覆盖同 id）。
以下为 r9y 原说明：
r9y 标签合并（2026-10-11）
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
DS = PROJECT_DIR / "data" / "laya" / "datasets" / "r9z"


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
    old = read(LAB / "r9y_all" / "labels_all.jsonl")
    new = {**read(LAB / "r9z" / "labels.jsonl"), **read(LAB / "r9z_test" / "labels.jsonl")}
    # 第二轮：第一轮批次的题面有 CARRY CANDIDATE 误打 bug（现任 will be replaced 但未来主C 已合格），修好后选项变了的 417 题重标
    _r2 = set(json.loads((DS / "relabel_round2_ids.json").read_text(encoding="utf-8")))
    new = {k: v for k, v in new.items() if k not in _r2}
    new.update(read(LAB / "r9z_r2" / "labels.jsonl"))
    # 第三轮：第二轮期间题面又修了（主C 已定取第一个合格者、手里有人选 / 路线有效时不打 CARRY CANDIDATE、写出不合格原因），选项变了的 70 题
    _r3 = set(json.loads((DS / "relabel_round3_ids.json").read_text(encoding="utf-8")))
    new = {k: v for k, v in new.items() if k not in _r3}
    new.update(read(LAB / "r9z_r3" / "labels.jsonl"))
    # 第四轮：训练集里带 BACKUP ROUTE carry / CARRY CANDIDATE、还挂着旧规范标签的 570 题（R25 没学会备选主C：第 580 局 R6 玄策 0.12）
    new.update(read(LAB / "r9z_r4" / "labels.jsonl"))
    pts = {}
    for line in (DS / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    relabel = json.loads((DS / "relabel.json").read_text(encoding="utf-8"))
    tr = set(json.loads((DS / "relabel_train_ids.json").read_text(encoding="utf-8")))
    te = set(json.loads((DS / "relabel_test_ids.json").read_text(encoding="utf-8")))
    bad = [i for i, r in new.items() if i not in pts or set(r["dist"]) != set(pts[i]["options"])]
    if bad:
        print("新标签与 r9z 选项对不上：", bad[:10])
        return 1
    merged = {i: r for i, r in old.items() if i in pts and set(r["dist"]) == set(pts[i]["options"])}
    dropped_old = len(old) - len(merged)
    merged.update(new)
    out_dir = LAB / "r9z_all"
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
    # **新标签覆盖检查**（2026-10-12 R25 教训：带 CARRY CANDIDATE 的 544 道训练题里 410 道还是旧规范标签，
    # BACKUP ROUTE carry 只有 10 道新标签 → 学生没学会，2.5 小时训练白跑）。训练集里选项带新标签的题必须全部是新规范标签。
    NEW_TAGS = ("BACKUP ROUTE carry", "CARRY CANDIDATE")
    tag_tot, tag_new = collections.Counter(), collections.Counter()
    for i, p in pts.items():
        if p["split"] != "train" or i not in merged:
            continue
        for t in NEW_TAGS:
            if any(t in v for v in p["question"]["criteria"].values()):
                tag_tot[t] += 1
                tag_new[t] += i in new
    st["new_tag_coverage"] = {t: f"{tag_new[t]}/{tag_tot[t]}" for t in NEW_TAGS}
    print("新标签覆盖（新规范标签 / 带标签的训练题）：", st["new_tag_coverage"])
    if any(tag_new[t] < tag_tot[t] or tag_tot[t] < 30 for t in NEW_TAGS):
        print("新标签覆盖不足：训练前先把带新标签的题全部按新规范重标（每种至少 30 题）")
        return 1
    ok = st["relabel_train_done"] >= args.min_frac * max(1, len(tr)) and st["relabel_test_done"] == len(te)
    if not ok:
        print(f"标注未完成：训练 {st['relabel_train_done']}/{len(tr)}（要求 ≥{args.min_frac:.0%}），测试 {st['relabel_test_done']}/{len(te)}（要求全部）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
