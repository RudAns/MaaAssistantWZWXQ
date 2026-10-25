"""两份老师标注的一致性：同一批题，比 top1 一致率、tmass（B 的分布在 A 的 best 上的概率质量）、平均 L1 距离。

  python tools/laya/compare_teachers.py --ver r9l                 # dup_* vs 正式批次（只比两边都有的题）
  python tools/laya/compare_teachers.py --ver r9k --ver r9j       # 多个版本汇总（同模型自一致基线）

用途：换老师模型（如 Sonnet）前，看它和原老师的差距是否落在"同模型标两次"的噪声以内。
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if isinstance(x.get("dist"), dict) and x.get("id"):
            out[x["id"]] = x
    return out


def best(d: dict) -> str:
    return max(d, key=d.get)


def compare(ver: str) -> list[tuple]:
    od = ROOT / "data" / "laya" / "labels" / ver / "out"
    main, dup = {}, {}
    for f in sorted(od.glob("*.jsonl")):
        (dup if f.name.startswith("dup_") else main).update(load(f))
    kinds = {}
    pts = ROOT / "data" / "laya" / "datasets" / ver / "points.jsonl"
    if pts.exists():
        for line in pts.open(encoding="utf-8"):
            p = json.loads(line)
            kinds[p["id"]] = p.get("kind", "?")
    rows = []
    for i, b in dup.items():
        a = main.get(i)
        if not a or set(a["dist"]) != set(b["dist"]):
            continue
        da, db = a["dist"], b["dist"]
        rows.append((kinds.get(i, "?"), best(da) == best(db), db.get(best(da), 0.0),
                     sum(abs(da.get(k, 0) - db.get(k, 0)) for k in da)))
    return rows


def report(name: str, rows: list[tuple]) -> None:
    if not rows:
        print(f"{name}: 没有可比的题")
        return
    n = len(rows)
    print(f"{name}: n={n} top1一致 {sum(r[1] for r in rows) / n:.3f}  tmass {sum(r[2] for r in rows) / n:.3f}  "
          f"L1 {sum(r[3] for r in rows) / n:.3f}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    for k, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"    {k:10s} n={len(rs):3d} top1一致 {sum(r[1] for r in rs) / len(rs):.3f}  L1 {sum(r[3] for r in rs) / len(rs):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", action="append", required=True)
    args = ap.parse_args()
    allr = []
    for v in args.ver:
        r = compare(v)
        report(v, r)
        allr += r
    if len(args.ver) > 1:
        report("合计", allr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
