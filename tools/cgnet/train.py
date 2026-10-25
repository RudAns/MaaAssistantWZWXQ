"""cgnet v1 训练：对老师软标签做 KL 蒸馏，多题型联合训练。

用法：
  python tools/cgnet/train.py --run full --seeds 1,2,3
  python tools/cgnet/train.py --run no_rel --ablate no_rel --seeds 1,2,3
  python tools/cgnet/train.py --run laya_text --enc laya --seeds 1,2,3
消融：no_rel（去掉关系偏置）/ no_text（去掉文本向量）/ no_tech（去掉手法节点）/ no_nodeid（去掉节点 id 嵌入）

产物 data/cgnet/runs/<run>/seed<s>/：config.json、curves.jsonl、metrics.json、preds_test.jsonl、preds_val.jsonl、model.pt
划分沿用 points.jsonl 的 split（train 训练 / val 选 epoch / test 报告；exam_test 不用，与 Laya 相同）。
指标口径与 tools/laya/train.py 的 metrics() 一致：top1 = argmax == 老师 best；tmass = 老师分布在 argmax 项上的质量。
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
from model import CGNet, n_params  # noqa: E402

ROOT = PROJECT_DIR / "data" / "cgnet"


def load(ds: str, ablate: str):
    d = ROOT / ds
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    samples = pickle.loads((d / "samples.pkl").read_bytes())
    if ablate == "no_tech":
        tech = meta["tok_types"].index("technique")
        out = []
        for s in samples:
            keep = int((s["tt"] != tech).sum())          # 手法节点都排在最后
            s = dict(s)
            for k in ("tt", "ot", "node", "text", "num"):
                s[k] = s[k][:keep]
            s["names"] = s["names"][:keep]
            r = s["rel"]
            km = (r[:, 0] < keep) & (r[:, 1] < keep) if len(r) else np.zeros(0, bool)
            s["rel"] = r[km] if len(r) else r
            if "relv" in s:
                s["relv"] = s["relv"][km] if len(r) else s["relv"]
            out.append(s)
        samples = out
    return meta, samples


def collate(batch, n_ch: int, device):
    B = len(batch)
    N = max(len(s["tt"]) for s in batch)
    K = max(len(s["opt_pos"]) for s in batch)
    tt = np.zeros((B, N), np.int64)
    ot = np.zeros((B, N), np.int64)
    node = np.zeros((B, N), np.int64)
    text = np.zeros((B, N), np.int64)
    num = np.zeros((B, N, batch[0]["num"].shape[1]), np.float32)
    mask = np.zeros((B, N), bool)
    rel = np.zeros((B, N, N, n_ch), np.float32)
    opt = np.zeros((B, K), np.int64)
    optm = np.zeros((B, K), bool)
    tgt = np.zeros((B, K), np.float32)
    glob = np.stack([s["glob"] for s in batch])
    for b, s in enumerate(batch):
        n = len(s["tt"])
        tt[b, :n], ot[b, :n], node[b, :n], text[b, :n] = s["tt"], s["ot"], s["node"], s["text"]
        num[b, :n] = s["num"]
        mask[b, :n] = True
        r = s["rel"]
        if len(r):
            rel[b, r[:, 0], r[:, 1], r[:, 2]] = s["relv"].astype(np.float32) if "relv" in s else 1.0
        k = len(s["opt_pos"])
        opt[b, :k] = s["opt_pos"]
        optm[b, :k] = True
        tgt[b, :k] = s["target"]
    T = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return {"tt": T(tt), "ot": T(ot), "node": T(node), "text": T(text), "num": T(num), "glob": T(glob),
            "mask": T(mask), "rel": T(rel), "opt": T(opt), "optm": T(optm)}, T(tgt)


@torch.no_grad()
def predict(model, samples, n_ch, device, bs=128):
    model.eval()
    out = []
    for b in range(0, len(samples), bs):
        batch = samples[b:b + bs]
        x, _ = collate(batch, n_ch, device)
        lg = model(**x).float().cpu().numpy()
        for r, s in enumerate(batch):
            k = len(s["keys"])
            z = lg[r, :k]
            p = np.exp(z - z.max())
            p /= p.sum()
            out.append({"id": s["id"], "kind": s["kind"], "keys": s["keys"], "p": p.tolist()})
    return out


def metrics(samples, preds) -> dict:
    by = {s["id"]: s for s in samples}
    agg: dict = {}
    for pr in preds:
        s = by[pr["id"]]
        q = np.array(pr["p"])
        t = np.asarray(s["target"], np.float64)
        a = int(q.argmax())
        rec = {"top1": float(s["keys"][a] == s["t_best"]), "tmass": float(t[a]),
               "ce": float(-(t * np.log(np.clip(q, 1e-12, 1))).sum()),
               "h_t": float(-(t * np.log(np.clip(t, 1e-12, 1))).sum()), "rand_tmass": float(t.mean()), "n": 1}
        for g in ("all", s["kind"]):
            d = agg.setdefault(g, {})
            for k, v in rec.items():
                d[k] = d.get(k, 0.0) + v
    return {g: {"n": int(d["n"]), "top1": d["top1"] / d["n"], "tmass": d["tmass"] / d["n"],
                "ce": d["ce"] / d["n"], "kl": (d["ce"] - d["h_t"]) / d["n"], "random_tmass": d["rand_tmass"] / d["n"]}
            for g, d in agg.items()}


def run_one(args, seed: int, meta, samples, text_table, node_feat, device) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_ch = meta["n_channels"]
    tr = [s for s in samples if s["split"] == "train"]
    va = [s for s in samples if s["split"] == "val"]
    te = [s for s in samples if s["split"] == "test"]
    model = CGNet(torch.from_numpy(text_table), torch.from_numpy(node_feat), len(meta["tok_types"]),
                  len(meta["opt_types"]), meta["num_dim"], meta["glob_dim"], n_ch, d=args.d, heads=args.heads,
                  layers=args.layers, ff=args.ff, drop=args.drop, use_text=args.ablate != "no_text",
                  use_rel=args.ablate != "no_rel", use_node_id=args.ablate != "no_nodeid").to(device)
    npar = n_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((len(tr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    # 题型权重：w_kind ∝ n_kind^-alpha（alpha=0 = 与 Laya 一样不加权）
    cnt = {}
    for s in tr:
        cnt[s["kind"]] = cnt.get(s["kind"], 0) + 1
    kw = {k: (v ** -args.kind_alpha) for k, v in cnt.items()}
    z = sum(kw[s["kind"]] for s in tr) / len(tr)
    kw = {k: v / z for k, v in kw.items()}
    out_dir = ROOT / "runs" / args.run / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(args), "seed": seed, "n_params": npar, "n": {"train": len(tr), "val": len(va), "test": len(te)},
           "kind_weights": kw, "device": str(device), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    curves = (out_dir / "curves.jsonl").open("w", encoding="utf-8")
    best, best_state = (1e9, -1), None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(tr)
        tot = 0.0
        for b in range(0, len(tr), args.bs):
            batch = tr[b:b + args.bs]
            x, tgt = collate(batch, n_ch, device)
            lg = model(**x)
            logp = F.log_softmax(lg, dim=-1)
            ce = -(tgt * logp.clamp(min=-1e4)).sum(-1)
            w = torch.tensor([kw[s["kind"]] for s in batch], device=device)
            loss = (ce * w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(batch)
        vm = metrics(va, predict(model, va, n_ch, device))["all"]
        rec = {"epoch": ep, "train_loss": tot / len(tr), "val": vm, "secs": round(time.time() - t0, 1)}
        curves.write(json.dumps(rec) + "\n")
        curves.flush()
        if vm["kl"] < best[0]:
            best = (vm["kl"], ep)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    curves.close()
    train_secs = time.time() - t0
    model.load_state_dict(best_state)
    pv, pt = predict(model, va, n_ch, device), predict(model, te, n_ch, device)
    res = {"best_epoch": best[1], "train_secs": round(train_secs, 1), "n_params": npar,
           "val": metrics(va, pv), "test": metrics(te, pt)}
    (out_dir / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, pp in (("preds_val.jsonl", pv), ("preds_test.jsonl", pt)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in pp:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    torch.save({"state": model.state_dict(), "cfg": cfg}, out_dir / "model.pt")
    t = res["test"]["all"]
    print(f"[{args.run} seed{seed}] best_ep={best[1]} params={npar} {train_secs:.0f}s  "
          f"TEST top1={t['top1']:.3f} tmass={t['tmass']:.3f} kl={t['kl']:.3f} | "
          + " ".join(f"{k}:{v['top1']:.2f}/{v['tmass']:.2f}" for k, v in res["test"].items() if k != "all"), flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ds", default="r9m")
    ap.add_argument("--enc", default="bge", choices=["bge", "laya"])
    ap.add_argument("--ablate", default="none", choices=["none", "no_rel", "no_text", "no_tech", "no_nodeid"])
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
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    meta, samples = load(args.ds, args.ablate)
    text_table = np.load(ROOT / args.ds / f"text_table_{args.enc}.npy")
    node_feat = np.load(ROOT / args.ds / "node_feat.npy")
    allres = {}
    for s in [int(x) for x in args.seeds.split(",")]:
        allres[s] = run_one(args, s, meta, samples, text_table, node_feat, device)
    ks = sorted({k for r in allres.values() for k in r["test"]})
    summ = {k: {m: float(np.mean([r["test"][k][m] for r in allres.values() if k in r["test"]]))
                for m in ("top1", "tmass", "kl")} for k in ks}
    (ROOT / "runs" / args.run / "summary.json").write_text(json.dumps(
        {"seeds": list(allres), "test_mean": summ, "n_params": allres[next(iter(allres))]["n_params"],
         "train_secs_mean": float(np.mean([r["train_secs"] for r in allres.values()]))}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("MEAN", json.dumps(summ["all"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
