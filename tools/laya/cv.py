"""5 折交叉验证：逐折调用 train.py（--cv K --fold i），汇总成均值 ± 标准差，追加到 experiments.md。

  python tools/laya/cv.py --exp S1_head -- --stage head --epochs 8 --lr 3e-4
  python tools/laya/cv.py --summarize S1_head         # 只重新汇总
  python tools/laya/cv.py --teacher                   # 老师自身一致性（重复标注）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
RUNS = PROJECT_DIR / "data" / "laya" / "runs"
EXPMD = PROJECT_DIR / "docs" / "laya-report" / "experiments.md"
PY = sys.executable
METRICS = ("top1", "tmass", "top1_eq", "tmass_eq", "ce", "kl", "ece")


def summarize(exp: str, k: int, extra: str = "") -> dict:
    per_fold = []
    for i in range(k):
        f = RUNS / f"{exp}_f{i}" / "metrics.json"
        if f.exists():
            per_fold.append(json.loads(f.read_text(encoding="utf-8")))
    if not per_fold:
        raise SystemExit(f"{exp}: 没有任何折的结果")
    groups = sorted({g for m in per_fold for g in m["test"]})
    out = {"exp": exp, "folds": len(per_fold), "groups": {}}
    for g in groups:
        ms = [m["test"][g] for m in per_fold if g in m["test"]]
        n = sum(m["n"] for m in ms)
        row = {"n": n}
        for key in METRICS + ("rule_vs_teacher_ref", "random_tmass", "rule_top1_ref"):
            # 按样本数加权的均值（各折大小不同）+ 折间标准差
            if any(key not in m for m in ms):
                continue
            vals = np.array([m[key] for m in ms], dtype=float)
            w = np.array([m["n"] for m in ms], dtype=float)
            row[key] = float((vals * w).sum() / w.sum())
            row[key + "_sd"] = float(vals.std()) if len(vals) > 1 else 0.0
        out["groups"][g] = row
    out["temperature"] = [m.get("temperature") for m in per_fold]
    out["best_epoch"] = [m.get("best_epoch") for m in per_fold]
    (RUNS / f"{exp}_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    a = out["groups"]["all"]
    line = (f"| {exp} | {time.strftime('%m-%d %H:%M')} | {a['n']} | {a['top1']:.3f}±{a['top1_sd']:.3f} | "
            f"{a.get('top1_eq', float('nan')):.3f} | {a['tmass']:.3f}±{a['tmass_sd']:.3f} | {a['ce']:.3f} | {a['ece']:.3f} | "
            + " / ".join(f"{g}:{out['groups'][g]['top1']:.2f}" for g in ("shop", "talent", "card_pick", "lord_boost")
                         if g in out["groups"])
            + f" | {a['random_tmass']:.3f} / {a['rule_vs_teacher_ref']:.3f} | {extra} |\n")
    if not EXPMD.exists():
        EXPMD.write_text(
            "# 实验一览\n\n> 自动追加（`tools/laya/cv.py`）。指标均相对**老师（Claude）标签**，5 折（按对局分折）测试集汇总，"
            "均值按样本数加权，± 为折间标准差。\n> 参照列：随机选的 tmass / 规则引擎当时的选择与老师 best 一致的比例。\n\n"
            "| 实验 | 时间 | n | top1 | top1(等价) | tmass | CE | ECE | 分类型 top1 | 随机tmass / 规则top1 | 配置 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with EXPMD.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line.strip())
    return out


def teacher_agreement(ver: str = "v2") -> dict:
    root = PROJECT_DIR / "data" / "laya" / "labels" / ver
    pts = {}
    for line in (PROJECT_DIR / "data" / "laya" / "datasets" / ver / "points.jsonl").open(encoding="utf-8"):
        x = json.loads(line)
        pts[x["id"]] = x
    main = {json.loads(l)["id"]: json.loads(l) for l in (root / "labels.jsonl").open(encoding="utf-8")}
    dup = [json.loads(l) for l in (root / "labels_dup.jsonl").open(encoding="utf-8")]
    rows = []
    for d in dup:
        m = main.get(d["id"])
        if not m:
            continue
        keys = list(m["dist"])
        p = np.array([m["dist"][k] for k in keys])
        q = np.array([d["dist"].get(k, 0) for k in keys])
        tv = 0.5 * np.abs(p - q).sum()
        crit = pts[d["id"]]["question"]["criteria"]
        same = [i for i, k in enumerate(keys) if crit.get(k) == crit.get(d["best"])]
        rows.append({"top1": float(m["best"] == d["best"]), "tv": float(tv),
                     "top1_eq": float(crit.get(m["best"]) == crit.get(d["best"])),
                     "tmass_eq": float(p[same].sum()), "kind": pts[d["id"]]["kind"],
                     # "第二位老师的选择在第一位老师分布里的质量"：与学生的 tmass 同口径
                     "tmass": float(p[keys.index(d["best"])])})
    res = {"n": len(rows)}
    for key in ("top1", "top1_eq", "tv", "tmass", "tmass_eq"):
        res[key] = float(np.mean([r[key] for r in rows])) if rows else None
    res["by_kind"] = {}
    for k in sorted({r["kind"] for r in rows}):
        rr = [r for r in rows if r["kind"] == k]
        res["by_kind"][k] = {"n": len(rr), "top1": float(np.mean([r["top1"] for r in rr])),
                             "top1_eq": float(np.mean([r["top1_eq"] for r in rr]))}
    (RUNS / "teacher_agreement.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return res


def main() -> int:
    argv = sys.argv[1:]
    passthru = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthru = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--summarize")
    ap.add_argument("--teacher", action="store_true")
    args = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.teacher:
        teacher_agreement()
        return 0
    if args.summarize:
        summarize(args.summarize, args.k, "(re-summarized)")
        return 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for i in range(args.k):
        cmd = [PY, "-X", "utf8", str(PROJECT_DIR / "tools" / "laya" / "train.py"), "--run", f"{args.exp}_f{i}",
               "--cv", str(args.k), "--fold", str(i), "--no-save"] + passthru
        print(">>", " ".join(cmd[3:]), flush=True)
        r = subprocess.run(cmd, env=env, cwd=str(PROJECT_DIR))
        if r.returncode != 0:
            print(f"fold {i} 失败 rc={r.returncode}")
            return r.returncode
    summarize(args.exp, args.k, " ".join(passthru))
    return 0


if __name__ == "__main__":
    sys.exit(main())
