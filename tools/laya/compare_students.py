"""两个（或多个）学生在**同一批题、同一版老师标签**上配对比较 —— 学生可以各自用自己的题面版本（f5 / f6…）。

  python tools/laya/compare_students.py --labels data/laya/labels/test_g5/labels.jsonl --split test \
      --student R4B=data/laya/runs/R4B_f5_top8/checkpoint@r5f5 \
      --student R5F6b=data/laya/runs/R5F6b_top8/checkpoint@r5f6

每个 --student 是 名字=checkpoint@数据集版本（数据集决定题面文本；题目 id 与选项键在各版本间一致，rerender 保证）。
输出：每个学生的 top1 / tmass / ce（总体 + 按类型）；两两配对的 McNemar 精确检验（top1）与 tmass 差的 bootstrap 95% CI。
结果追加到 data/laya/runs/compare_students.jsonl，逐题预测存到 data/laya/runs/compare_<name>_<labels 目录>.jsonl。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "laya"))
import torch  # noqa: E402
import train as T  # noqa: E402


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar（二项检验，p=0.5）。b = 只有 A 对，c = 只有 B 对。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--student", action="append", required=True, help="名字=checkpoint@数据集版本")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--ids", nargs="*", help="只比这些清单里的题（json 题号列表；给了就忽略 --split）")
    args = ap.parse_args()
    want = set()
    for f in args.ids or []:
        want |= set(json.loads(Path(f).read_text(encoding="utf-8")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {}
    for s in args.student:
        name, rest = s.split("=", 1)
        ck, ver = rest.rsplit("@", 1)
        rows = [r for r in T.load_data(ver, ROOT / args.labels) if (r["id"] in want if want else r["split"] == args.split)]
        model, tok, cfg = T.load_model(Path(ck), dev)
        temp = (cfg.get("temperature") or [1.0])[0]
        preds = T.predict(model, tok, cfg, rows, dev, temp=temp)
        m = T.metrics(rows, preds)
        by_row = {r["id"]: r for r in rows}
        per = {}
        for p in preds:
            r = by_row[p["id"]]
            keys, q = p["keys"], np.array(p["p"])
            t = np.array([r["teacher"].get(k, 0.0) for k in keys])
            t = t / t.sum()
            a = int(q.argmax())
            # **等价选项**（商店里可能出现同样的牌，选哪个都算正确答案）：
            # 商店同名牌出题时已合并（features.dedupe_shop，2814 题里 738 题有同名、都只留一个选项）；
            # 其余题型按"选项名相同"（冒号前）判等价 —— 天赋有 12 题出现同名选项（测试集里 0 题）。
            crit = (r.get("question") or {}).get("criteria") or r.get("criteria") or {}
            nm = lambda k: re.split(r"[:：]", str(crit.get(k, k)))[0].strip()     # noqa: E731
            same = [i for i, k in enumerate(keys) if nm(k) == nm(r["t_best"])]
            per[p["id"]] = {"top1": float(keys[a] == r["t_best"]), "tmass": float(t[a]),
                            "top1_eq": float(a in same), "tmass_eq": float(t[same].sum()) if same else float(t[a]) * 0,
                            "choice": keys[a], "kind": r["kind"],
                            # 老师的答案不是百分百集中在一个选项 —— 软标签下 top1 太粗：
                            # tmax = 老师最佳选项的概率（tmass 的上限）；regret = tmax - tmass（选了老师眼里差多少的选项）
                            "tmax": float(t.max()), "regret": float(t.max() - t[a])}
        res[name] = {"m": m, "per": per, "ver": ver, "ck": ck}
        a = m["all"]
        eq1 = np.mean([v["top1_eq"] for v in per.values()])
        print(f"{name:10s} ({ver}, max_len {cfg.get('max_len')}): n={a['n']} top1 {a['top1']:.3f} tmass {a['tmass']:.3f} "
              f"ce {a['ce']:.3f} ece {a.get('ece', float('nan')):.3f} | 等价 top1 {eq1:.3f}")
        _tm = np.mean([v["tmax"] for v in per.values()]); _rg = np.mean([v["regret"] for v in per.values()])
        _cf = [v for v in per.values() if v["tmax"] >= 0.6]
        _ok = np.mean([v["regret"] <= 0.1 for v in per.values()])
        print(f"    软标签：tmass/上限 {a['tmass'] / max(_tm, 1e-9):.3f}（上限 {_tm:.3f}）| 平均损失 regret {_rg:.3f} | "
              f"选到老师眼里差 ≤0.1 的选项 {_ok:.3f} | 老师有把握题(最佳≥0.6) n={len(_cf)} top1 "
              f"{(np.mean([v['top1'] for v in _cf]) if _cf else float('nan')):.3f}")
        for k, v in m.items():
            if k != "all":
                print(f"    {k:10s} n={v['n']:3d} top1 {v['top1']:.3f} tmass {v['tmass']:.3f}")
        out = ROOT / "data" / "laya" / "runs" / f"compare_{name}_{Path(args.labels).parent.name}.jsonl"
        out.write_text("\n".join(json.dumps({"id": k, **v}, ensure_ascii=False) for k, v in per.items()), encoding="utf-8")
        del model
        torch.cuda.empty_cache()
    rng = np.random.default_rng(0)
    pairs = []
    for A, B in itertools.combinations(res, 2):
        ids = sorted(set(res[A]["per"]) & set(res[B]["per"]))
        pa, pb = res[A]["per"], res[B]["per"]
        b = sum(1 for i in ids if pa[i]["top1"] and not pb[i]["top1"])
        c = sum(1 for i in ids if pb[i]["top1"] and not pa[i]["top1"])
        d = np.array([pb[i]["tmass"] - pa[i]["tmass"] for i in ids])
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = mcnemar_exact(b, c)
        print(f"{B} vs {A}: n={len(ids)}  top1 {np.mean([pb[i]['top1'] for i in ids]) - np.mean([pa[i]['top1'] for i in ids]):+.3f} "
              f"(只 {A} 对 {b} / 只 {B} 对 {c}, McNemar p={p:.3f})  tmass {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        pairs.append({"A": A, "B": B, "n": len(ids), "b": b, "c": c, "p": p, "dtmass": float(d.mean()), "ci": [float(lo), float(hi)]})
    with (ROOT / "data" / "laya" / "runs" / "compare_students.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "labels": args.labels, "split": args.split,
                            "students": {k: {"ver": v["ver"], "ck": v["ck"], "all": v["m"]["all"]} for k, v in res.items()},
                            "pairs": pairs}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
