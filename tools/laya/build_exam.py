"""**考题**：从实战对局的逐帧状态里挖出学生容易答错的局面类型，批量构造成题目交给老师标注（2026-09-24：
不用规则强制处理，目标是让 Laya 变强 —— 学生错了不靠规则兜，而是把这类局面做成考题）。

  python tools/laya/build_exam.py --rounds 1 2 --base r1 --out r2exam

- 题目来源：iterations/round<N>_matches.jsonl 里各局的 data/logs/<run>/states.jsonl（逐帧），用和线上同一套的
  build_dataset.extract_run 构造（特征版本一致）；已在 base 数据集里的题（按 id / 同局同回合同题面）跳过。
- 题型配额（商店）：
    same_name  商店里有**场上或手里已有的同名英雄**（合成不占人口）      —— g3 第 10 条
    maxed      棋手满级、能量 ≥10（钱只能花在刷新/买牌上）               —— g3 第 8 条
    level_move 手里有搬等级的牌（夺取之术/众星拱月）或选过红叶染锋      —— g3 第 9 条
    other      其余商店题随机抽，保持分布
  天赋 / N选1 / 棋手特性的新题全收（量少、价值高）。
- 划分：按对局切，约 20% 的对局做**考题测试集**（split=exam_test，不参与训练，专看专项能力），其余 train。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))
import build_dataset as B  # noqa: E402
import features as F  # noqa: E402

STEAL_CARDS = ("夺取之术", "众星拱月")


def exam_type(p: dict) -> str:
    if p["kind"] != "shop":
        return p["kind"]
    ctx = p.get("ctx") or {}
    hand = ctx.get("hand") or []
    field = set(p.get("field") or [])
    shop = (p.get("raw") or {}).get("shop") or []
    if any(f"buy{slot}" in p["options"] and n and (n in field or n in hand) for slot, n, _c in shop):
        return "same_name"
    if (ctx.get("level") or 0) >= 6 and (ctx.get("gold") or 0) >= 10:
        return "maxed"
    if any(c in hand for c in STEAL_CARDS):
        return "level_move"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--base", default="r1")
    ap.add_argument("--out", default="r2exam")
    ap.add_argument("--quota", default="same_name=160,maxed=80,level_move=40,other=80")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=24)
    args = ap.parse_args()
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    base = [json.loads(line) for line in (ds / args.base / "points.jsonl").open(encoding="utf-8")]
    have_id = {p["id"] for p in base}
    have_key = {(p.get("run"), p["round"], p["kind"],
                 json.dumps(p["question"]["criteria"], ensure_ascii=False, sort_keys=True)) for p in base}
    runs = []
    for r in args.rounds:
        f = PROJECT_DIR / "data" / "laya" / "iterations" / f"round{r}_matches.jsonl"
        for line in f.open(encoding="utf-8"):
            m = json.loads(line)
            for run in (m["run"] if isinstance(m["run"], list) else [m["run"]]):
                runs.append((f"r{r}m{m['match']}", m.get("rank"), run))
    pts, seen = [], set()
    for match, rank, run in runs:
        rd = PROJECT_DIR / "data" / "logs" / run
        if not (rd / "states.jsonl").exists():
            continue
        for s in B.extract_run(rd):
            key = (s.get("run"), s["round"], s["kind"],
                   json.dumps(s["question"]["criteria"], ensure_ascii=False, sort_keys=True))
            s["id"] = hashlib.md5(f"{s['run']}|{s['ts']}|{s['kind']}".encode()).hexdigest()[:12]
            if s["id"] in have_id or key in have_key or key in seen:
                continue
            seen.add(key)
            s.update(match=match, rank=rank, feature_version=F.FEATURE_VERSION, source=f"exam_{args.out}")
            s["exam_type"] = exam_type(s)
            pts.append(s)
    rng = random.Random(args.seed)
    quota = {k: int(v) for k, v in (kv.split("=") for kv in args.quota.split(","))}
    by = collections.defaultdict(list)
    for s in pts:
        by[s["exam_type"]].append(s)
    chosen = []
    for t, rows in by.items():
        rng.shuffle(rows)
        chosen += rows[:quota.get(t, 10 ** 6)]          # 天赋/N选1/特性不设上限
    matches = sorted({s["match"] for s in chosen})
    rng.shuffle(matches)
    test_m = set(matches[:max(1, int(len(matches) * args.test_frac))])
    for s in chosen:
        s["split"] = "exam_test" if s["match"] in test_m else "train"
    out = ds / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "points.jsonl").open("w", encoding="utf-8") as fh:
        for p in base + chosen:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "new_ids.json").write_text(json.dumps([s["id"] for s in chosen]), encoding="utf-8")
    info = {"base": args.base, "runs": len(runs), "candidates": dict(collections.Counter(s["exam_type"] for s in pts)),
            "chosen": dict(collections.Counter(s["exam_type"] for s in chosen)),
            "exam_test_matches": sorted(test_m),
            "split": dict(collections.Counter(s["split"] for s in chosen))}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
