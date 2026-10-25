"""迭代训练的数据集：上一版数据集 + 本轮实战里**学生实际答过的题**（on-policy，DAgger 式）。

  python tools/laya/build_round.py --round 1 --base v4 --out r1

- 本轮对局清单：data/laya/iterations/round<N>_matches.jsonl（每局一行，run 可以是列表 = 中途重启过）。
- 新题取自各局的 laya_points.jsonl（线上用的就是 build_dataset 的同一套构造函数，特征版本一致）。
  学生当时的答案（laya / laya_choice / used / why）**从题面里剥掉**，另存 student_r<N>.jsonl 供分析；
  老师标注时看不到学生答案，也看不到规则的选择（label_batches 本来就隐藏 rule_choice）。
- 新题一律进 train 划分；**测试集保持 base 的 test 不变**（论文里每轮都在同一批 248 题上报告）。
- 输出 data/laya/datasets/<out>/points.jsonl（base 全部 + 新题）、new_ids.json（只需标注的新题）。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
STUDENT_KEYS = ("laya", "laya_choice", "laya_conf", "laya_ms", "used", "why", "ckpt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--base", default="v4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    base = [json.loads(l) for l in (ds / args.base / "points.jsonl").open(encoding="utf-8")]
    have = {p["id"] for p in base}
    fv = collections.Counter(p.get("feature_version") for p in base)
    matches = [json.loads(l) for l in
               (PROJECT_DIR / "data" / "laya" / "iterations" / f"round{args.round}_matches.jsonl").open(encoding="utf-8")]
    new, student = [], []
    for m in matches:
        runs = m["run"] if isinstance(m["run"], list) else [m["run"]]
        for run in runs:
            f = PROJECT_DIR / "data" / "logs" / run / "laya_points.jsonl"
            if not f.exists():
                continue
            for line in f.open(encoding="utf-8"):
                p = json.loads(line)
                if p["id"] in have:
                    continue
                have.add(p["id"])
                student.append({k: p.get(k) for k in ("id", "kind", "round", "rule_choice") + STUDENT_KEYS})
                q = {k: v for k, v in p.items() if k not in STUDENT_KEYS}
                q.update(split="train", match=f"r{args.round}m{m['match']}", rank=m.get("rank"),
                         source=f"round{args.round}")
                new.append(q)
    bad_fv = [p["id"] for p in new if p.get("feature_version") not in fv]
    if bad_fv:
        print(f"⚠ {len(bad_fv)} 道新题的特征版本不在 base 里：{bad_fv[:5]}")
    out = ds / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "points.jsonl").open("w", encoding="utf-8") as fh:
        for p in base + new:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "new_ids.json").write_text(json.dumps([p["id"] for p in new]), encoding="utf-8")
    with (out / f"student_r{args.round}.jsonl").open("w", encoding="utf-8") as fh:
        for s in student:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    info = {"base": args.base, "base_n": len(base), "new_n": len(new),
            "new_by_kind": dict(collections.Counter(p["kind"] for p in new)),
            "new_by_match": dict(collections.Counter(p["match"] for p in new)),
            "splits": dict(collections.Counter(p["split"] for p in base + new))}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
