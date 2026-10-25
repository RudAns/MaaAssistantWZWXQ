"""cgnet v2 训练 + 消融（模型、训练循环、指标与 v1 完全相同：复用 train.run_one；只换数据与消融方式）。

  python tools/cgnet/train_v2.py --run v2 --ds r9m_v2 --seeds 1,2,3
  python tools/cgnet/train_v2.py --run v2_no_facts --ds r9m_v2 --v2ablate no_facts
消融（--v2ablate）：
  no_facts    汇总事实置零（全局 64..111、对象 28..35）
  no_techdyn  去掉手法槽位关系通道 + 手法数值（36..43）置零（手法对象保留）
  no_tech     去掉全部手法对象（及其所有关系）
  no_cond     去掉条件边通道
  no_v2       以上全部去掉（等价 v1 输入，但手法对象选择规则不同 → 仅作参照）
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

ROOT = PROJECT_DIR / "data" / "cgnet"


def apply(samples, meta, mode: str):
    if mode == "none":
        return samples
    cr = meta["channel_ranges"]
    tech = meta["tok_types"].index("technique")
    out = []
    for s in samples:
        s = dict(s)
        s["num"] = s["num"].copy()
        s["glob"] = s["glob"].copy()
        r, rv = s["rel"], s["relv"]
        keepm = np.ones(len(r), bool)
        if mode in ("no_facts", "no_v2"):
            s["glob"][meta["fact_glob"][0]:meta["fact_glob"][1]] = 0
            s["num"][:, meta["fact_num"][0]:meta["fact_num"][1]] = 0
        if mode in ("no_techdyn", "no_v2"):
            keepm &= ~((r[:, 2] >= cr["slot"][0]) & (r[:, 2] < cr["slot"][1]))
            s["num"][:, meta["tech_num"][0]:meta["tech_num"][1]] = 0
        if mode in ("no_cond", "no_v2"):
            keepm &= ~((r[:, 2] >= cr["cond"][0]) & (r[:, 2] < cr["cond"][1]))
        if mode == "no_tech":
            n = int((s["tt"] != tech).sum())
            for k in ("tt", "ot", "node", "text", "num"):
                s[k] = s[k][:n]
            s["names"] = s["names"][:n]
            keepm &= (r[:, 0] < n) & (r[:, 1] < n)
        s["rel"], s["relv"] = r[keepm], rv[keepm]
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ds", default="r9m_v2")
    ap.add_argument("--v2ablate", default="none", choices=["none", "no_facts", "no_techdyn", "no_tech", "no_cond", "no_v2"])
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--kind_alpha", type=float, default=0.0)
    ap.add_argument("--kinds", default="", help="只用这些题型训练/评估，如 carry,swap（专项模型）")
    ap.add_argument("--inner_val", type=float, default=0.0,
                    help="从 train 里按对局（run）切出这个比例做选 epoch 的验证集（专项模型用：原 val 里 carry/swap 只有 2 题）")
    args = ap.parse_args()
    args.enc, args.ablate, args.cpu = "bge", "none", False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((ROOT / args.ds / "meta.json").read_text(encoding="utf-8"))
    samples = apply(pickle.loads((ROOT / args.ds / "samples.pkl").read_bytes()), meta, args.v2ablate)
    if args.kinds:
        ks = set(args.kinds.split(","))
        samples = [s for s in samples if s["kind"] in ks]
    if args.inner_val > 0:
        # 按对局切：增强题（aug）跟着它的原对局走；测试集不动
        runs = sorted({s["run"] for s in samples if s["split"] == "train"})
        rng = np.random.default_rng(0)
        rng.shuffle(runs)
        cnt = {r: sum(1 for s in samples if s["split"] == "train" and s["run"] == r) for r in runs}
        tot, acc, held = sum(cnt.values()), 0, set()
        for r in runs:
            if acc >= args.inner_val * tot:
                break
            held.add(r)
            acc += cnt[r]
        samples = [dict(s, split="val") if (s["split"] == "train" and s["run"] in held) else s for s in samples]
        print(f"inner val: {len(held)} runs, {acc} samples of {tot}")
    text_table = np.load(ROOT / args.ds / "text_table_bge.npy")
    node_feat = np.load(ROOT / args.ds / "node_feat.npy")
    allres = {}
    for sd in [int(x) for x in args.seeds.split(",")]:
        allres[sd] = T.run_one(args, sd, meta, samples, text_table, node_feat, device)
    ks = sorted({k for r in allres.values() for k in r["test"]})
    summ = {k: {m: float(np.mean([r["test"][k][m] for r in allres.values() if k in r["test"]]))
                for m in ("top1", "tmass", "kl")} for k in ks}
    (ROOT / "runs" / args.run / "summary.json").write_text(json.dumps(
        {"seeds": list(allres), "ds": args.ds, "v2ablate": args.v2ablate, "test_mean": summ,
         "n_params": allres[next(iter(allres))]["n_params"],
         "train_secs_mean": float(np.mean([r["train_secs"] for r in allres.values()]))}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("MEAN", json.dumps(summ["all"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
