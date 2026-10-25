"""r9m 的训练标签（2026-09-29）：r9l 旧标签打底 → r9m 各波里仍有效的标签覆盖 → 最后一波（r9m7）全收。

  python tools/laya/assemble_r9m.py --out data/laya/labels/r6/labels_all_r9m.jsonl

r9m 标注期间题面改过几轮（规范补充一~六、f9k 小修），每轮改完只保留"选项文本没变"的已标题
（build 期间记在 data/laya/datasets/r9m/kept_ids_final.json），其余并入下一波重标。
这里按同一份清单取标签，并逐题校验标签的选项键 == 数据集里这道题的选项键（对不上的丢弃并报数）。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_out(pattern: str) -> dict:
    out = {}
    for f in sorted(glob.glob(str(ROOT / pattern))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get("id") and isinstance(x.get("dist"), dict):
                out[x["id"]] = x
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/laya/labels/r6/labels_all_r9l.jsonl")
    ap.add_argument("--ds", default="r9m")
    ap.add_argument("--out", default="data/laya/labels/r6/labels_all_r9m.jsonl")
    args = ap.parse_args()
    pts = {}
    for line in open(ROOT / "data" / "laya" / "datasets" / args.ds / "points.jsonl", encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    labels = {}
    for line in open(ROOT / args.base, encoding="utf-8"):
        x = json.loads(line)
        labels[x["id"]] = dict(x, _src="r9l")
    keep = json.load(open(ROOT / "data" / "laya" / "datasets" / args.ds / "kept_ids_final.json", encoding="utf-8"))
    waves = {"w1": "r9m", "w2": "r9m2", "w3": "r9m3", "w4": "r9m4", "w5": "r9m5", "w6": "r9m6"}
    src = collections.Counter()
    for w, d in waves.items():
        got = load_out(f"data/laya/labels/{d}/out/*.jsonl")
        for i in keep.get(w, []):
            if i in got:
                labels[i] = dict(got[i], _src=d)
                src[d] += 1
    for i, x in load_out("data/laya/labels/r9m7/out/*.jsonl").items():
        labels[i] = dict(x, _src="r9m7")
        src["r9m7"] += 1
    need = set(json.load(open(ROOT / "data" / "laya" / "datasets" / args.ds / "new_ids.json", encoding="utf-8")))
    bad, missing, stale = collections.Counter(), [], 0
    out = []
    for i, p in pts.items():
        x = labels.get(i)
        if x is None:
            missing.append(i)
            continue
        if i in need and x["_src"] == "r9l":
            stale += 1                       # 需要重标却还是 r9l 的旧标签（不该出现）
            continue
        if set(x["dist"]) != set(p["options"]):
            bad[p["kind"]] += 1
            continue
        out.append({k: v for k, v in x.items() if k != "_src"})
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(ROOT / args.out, "w", encoding="utf-8") as fh:
        for x in out:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({"points": len(pts), "labels": len(out), "from": dict(src), "missing": len(missing),
                      "need_but_old": stale, "key_mismatch": dict(bad)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
