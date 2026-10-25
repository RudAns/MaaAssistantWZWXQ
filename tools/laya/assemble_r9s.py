"""r9s 标签合并（2026-10-08 夜）：r9r 全部标签 + g9p 新标 / 重标的 carry / swap 题（覆盖同 id 的旧标签）。

  .venv/Scripts/python.exe -X utf8 tools/laya/label_batches.py merge --ver r9s
  .venv/Scripts/python.exe -X utf8 tools/laya/label_batches.py merge --ver r9s_test
  .venv/Scripts/python.exe -X utf8 tools/laya/assemble_r9s.py

输出：data/laya/labels/r9s_all/labels_all.jsonl 与 data/laya/labels/r6/labels_all_r9s.jsonl（cgnet build_data / eval_deploy 读这个），
      data/laya/labels/r9s_all/stats.json（标注统计：新题 / 重标题各自的答案分布、重标前后 best 是否改变）。
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
LAB = PROJECT_DIR / "data" / "laya" / "labels"
DS = PROJECT_DIR / "data" / "laya" / "datasets" / "r9s"


def read(f: Path) -> dict:
    out = {}
    if f.exists():
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def main() -> int:
    old = read(LAB / "r6" / "labels_all_r9r.jsonl")
    new = {**read(LAB / "r9s" / "labels.jsonl"), **read(LAB / "r9s_test" / "labels.jsonl")}
    pts = {}
    for line in (DS / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    new_ids = set(json.loads((DS / "new_ids.json").read_text(encoding="utf-8")))
    rel_ids = set(json.loads((DS / "relabel_ids.json").read_text(encoding="utf-8")))
    rel_info = json.loads((DS / "relabel_info.json").read_text(encoding="utf-8"))["items"]
    want = new_ids | rel_ids
    missing = sorted(want - set(new))
    for i, r in new.items():
        assert set(r["dist"]) == set(pts[i]["options"]), i
    merged = dict(old)
    merged.update(new)
    for i in merged:
        assert i in pts, i
    for d in (LAB / "r9s_all", LAB / "r6"):
        d.mkdir(parents=True, exist_ok=True)
    for f in (LAB / "r9s_all" / "labels_all.jsonl", LAB / "r6" / "labels_all_r9s.jsonl"):
        with f.open("w", encoding="utf-8") as fh:
            for r in merged.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def opt_kind(p, k):
        if p["kind"] == "carry":
            return "keep" if k == "keep" else "switch"
        return k if k in ("bench", "levelup") else "sell"

    st = {"merged": len(merged), "old": len(old), "g9p_labels": len(new), "missing": missing[:50], "n_missing": len(missing)}
    c = collections.Counter()
    conf = collections.defaultdict(list)
    for i, r in new.items():
        p = pts[i]
        grp = "new" if i in new_ids else "relabel"
        sp = "train" if p["split"] == "train" else "test"
        c[(grp, sp, p["kind"], opt_kind(p, r["best"]))] += 1
        conf[(grp, p["kind"])].append(float(r.get("confidence") or 0))
        if grp == "relabel" and i in old:
            ob, nb = old[i]["best"], r["best"]
            c[("relabel_change", p["kind"], rel_info[i]["rule"], "same" if ob == nb else "changed")] += 1
            # 重标前后，在"换主C / 卖人"这一侧的概率
            side = "keep" if p["kind"] == "carry" else "bench"
            c[("relabel_mass_" + side, p["kind"], "up" if r["dist"].get(side, 0) > old[i]["dist"].get(side, 0) + 0.05
               else "down" if r["dist"].get(side, 0) < old[i]["dist"].get(side, 0) - 0.05 else "flat")] += 1
    st["counts"] = {"|".join(map(str, k)): v for k, v in sorted(c.items())}
    st["mean_conf"] = {"|".join(k): round(sum(v) / len(v), 3) for k, v in conf.items()}
    st["missing_info_top"] = collections.Counter(m for r in new.values() for m in (r.get("missing_info") or [])).most_common(30)
    (LAB / "r9s_all" / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in st.items() if k != "missing_info_top"}, ensure_ascii=False, indent=1))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
