"""把多轮标签按"后标的覆盖先标的"合成一份训练用标签（2026-09-26）。

  python tools/laya/assemble_labels.py --base data/laya/labels/r6/labels_all_r9c.jsonl \
      --over r9d r9e r9f --out data/laya/labels/r6/labels_all_r9f.jsonl --ds r9f

--over 里每一轮是 data/laya/labels/<ver>/labels.jsonl（label_batches merge 的产物）；
--ds 给出的数据集里没有标签的题会列出来（应当为 0）。
"""
import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--over", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds", required=True)
    a = ap.parse_args()
    lab = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / a.base).open(encoding="utf-8")}
    src = collections.Counter({"base": len(lab)})
    for v in a.over:
        for l in (ROOT / "data/laya/labels" / v / "labels.jsonl").open(encoding="utf-8"):
            r = json.loads(l)
            lab[r["id"]] = r
            src[v] += 1
    pts = [json.loads(l) for l in (ROOT / "data/laya/datasets" / a.ds / "points.jsonl").open(encoding="utf-8")]
    ids = {p["id"] for p in pts}
    miss = collections.Counter(p["kind"] for p in pts if p["id"] not in lab)
    bad = collections.Counter(p["kind"] for p in pts if p["id"] in lab and set(lab[p["id"]]["dist"]) != set(p["options"]))
    with (ROOT / a.out).open("w", encoding="utf-8") as fh:
        for i, r in lab.items():
            if i in ids:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"sources": dict(src), "points": len(pts), "missing": dict(miss), "key_mismatch": dict(bad)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
