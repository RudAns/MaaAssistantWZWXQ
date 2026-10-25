"""carry / swap 两题的按对局交叉验证（选型用）。

r9m 测试集里 carry 24 题、swap 9 题，种子之间差 3~4 题就能把 top1 摆动 0.1 —— 靠它选 v1/v2、选"专项 vs 多题联合"不可靠。
这里把**测试集以外**的 carry/swap 题按对局（run）分成 K 折：每折留出一折当评估集，其余训练；增强题（aug）跟着它的原对局走
（留出折里的增强题也参与评估，另单独报"只算真实题"）。r9m 测试集在这里完全不碰。

  python tools/cgnet/cv_cs.py --name cv_v1 --ds r9m
  python tools/cgnet/cv_cs.py --name cv_v2 --ds r9m_v2
  python tools/cgnet/cv_cs.py --name cv_v2_multi --ds r9m_v2 --multi      # 多题联合训练（其它题型全部进训练）
产物 data/cgnet/runs/<name>/fold<k>/...（run_one 的标准产物）+ data/cgnet/runs/<name>/cv_items.json（逐题得分，供配对比较）
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
import train as T  # noqa: E402
from train_v2 import apply  # noqa: E402

ROOT = PROJECT_DIR / "data" / "cgnet"
CS = {"carry", "swap"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ds", default="r9m_v2")
    ap.add_argument("--v2ablate", default="none")
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    args = argparse.Namespace(run="", enc="bge", ablate="none", epochs=a.epochs, bs=64, lr=1e-3, wd=0.05, d=128,
                              heads=4, layers=3, ff=256, drop=0.1, kind_alpha=0.0, cpu=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((ROOT / a.ds / "meta.json").read_text(encoding="utf-8"))
    allS = pickle.loads((ROOT / a.ds / "samples.pkl").read_bytes())
    if a.v2ablate != "none":
        allS = apply(allS, meta, a.v2ablate)
    text_table = np.load(ROOT / a.ds / "text_table_bge.npy")
    node_feat = np.load(ROOT / a.ds / "node_feat.npy")
    cs_runs = sorted({s["run"] for s in allS if s["kind"] in CS and s["split"] in ("train", "val")})
    rng = np.random.default_rng(0)                  # 折的划分与配置无关（固定种子 0）
    rng.shuffle(cs_runs)
    fold_of = {r: i % a.folds for i, r in enumerate(cs_runs)}
    items = {}
    for k in range(a.folds):
        held = {r for r, f in fold_of.items() if f == k}
        inner = sorted(r for r, f in fold_of.items() if f == (k + 1) % a.folds)   # 下一折当选 epoch 的验证集
        S = []
        for s in allS:
            if s["split"] in ("test", "exam_test"):
                continue                                           # r9m 测试集不碰
            if s["kind"] in CS:
                sp = "test" if s["run"] in held else "val" if s["run"] in inner else "train"
            else:
                if not a.multi:
                    continue
                sp = s["split"] if s["split"] in ("train", "val") else None
                if sp is None:
                    continue
                if sp == "val":
                    sp = "train"                                   # 验证只看 carry/swap（与专项一致）
            S.append(dict(s, split=sp))
        args.run = f"{a.name}/fold{k}"
        res = T.run_one(args, a.seed, meta, S, text_table, node_feat, device)
        preds = [json.loads(l) for l in (ROOT / "runs" / a.name / f"fold{k}" / f"seed{a.seed}" / "preds_test.jsonl").open(encoding="utf-8")]
        byid = {s["id"]: s for s in S}
        for pr in preds:
            s = byid[pr["id"]]
            q = np.array(pr["p"])
            t = np.asarray(s["target"], np.float64)
            i = int(q.argmax())
            items[pr["id"]] = {"kind": s["kind"], "aug": s.get("source") == "aug", "fold": k,
                               "top1": float(s["keys"][i] == s["t_best"]), "tmass": float(t[i])}
    (ROOT / "runs" / a.name / "cv_items.json").write_text(json.dumps(items), encoding="utf-8")
    for sub, f in (("all", lambda v: True), ("real", lambda v: not v["aug"])):
        for kind in ("carry", "swap", None):
            vs = [v for v in items.values() if f(v) and (kind is None or v["kind"] == kind)]
            if vs:
                print(f"{a.name} {sub:4s} {kind or 'both':5s} n={len(vs):4d} top1={np.mean([v['top1'] for v in vs]):.3f} "
                      f"tmass={np.mean([v['tmass'] for v in vs]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
