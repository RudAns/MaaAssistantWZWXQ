"""手牌立绘识别 v2：真实样本（全部历史对局，hand_names_v2 / hand_awaken_v2）+ 官网卡面合成样本（hand_synth_v1）。

  CUDA_VISIBLE_DEVICES= .venv/Scripts/python.exe -X utf8 tools/vision/train_hand_art_v2.py [--epochs 30] [--min-n 5]

与 v1（train_hand_art.py）同一网络、同一输入（64x120 裁图 → 中心 56x112，RGB/255）、同样的训练增强；区别：
- 真实标签来自 build_hand_v2.py（闭集解码 + 账本兜底），满宽（≥110px、不叠放）且非觉醒（hand_awaken_v1 概率 <0.5）；
- 词表 = 真实训练集 ≥min_n 的卡 ∪ 有官网卡面的全部手牌类卡（合成样本只进训练集）；
- 每个 epoch：真实样本按 1/sqrt(类频) 抽 N_REAL 张 + 合成样本均匀抽 N_SYN 张；
- 门槛只在**真实验证集**上定（被采纳准确率 ≥99%）；测试只用真实样本（test = v1 的测试局，test_new = 最近 20 局整局）。
输出 data/vision/runs/hand_art_v2/{best.pt, metrics.json, hand_art_v2.onnx, hand_art_v2.json}。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "vision"))
from train_hand_art import Net, augment, crop, evaluate, probs, to_tensor  # noqa: E402

CROPS = ROOT / "data/vision/hand_awaken_v2/crops"
SYN = ROOT / "data/vision/hand_synth_v1"


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def load_real():
    # 直接读 parts/（构建可能还在续跑；合并后的 pool.jsonl 与 parts 内容相同）
    rows = [json.loads(l) for f in sorted((ROOT / "data/vision/hand_names_v2/parts").glob("*.jsonl"))
            for l in f.read_text(encoding="utf-8").splitlines() if l]
    return [r for r in rows if r["roi"][2] >= 110 and r["label"] and r["p_awk"] < 0.5 and r.get("awk_v1") != 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--n-real", type=int, default=16000)
    ap.add_argument("--n-syn", type=int, default=8000)
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "data/vision/runs/hand_art_v2"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads); cv2.setNumThreads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    real = load_real()
    syn = [] if args.no_synth else [json.loads(l) for l in (SYN / "index.jsonl").read_text(encoding="utf-8").splitlines() if l]
    cnt = Counter(r["label"] for r in real if r["split"] == "train")
    classes = sorted({c for c, v in cnt.items() if v >= args.min_n} | {s["label"] for s in syn})
    cid = {c: k for k, c in enumerate(classes)}
    data = {k: [] for k in ("train", "val", "test", "test_new")}
    for r in real:
        data[r["split"]].append((imread(CROPS / f"{r['id']}.png"), cid.get(r["label"], -1), r["run"], r["id"], r["label"]))
    syn_d = [(imread(SYN / "crops" / f"{s['id']}.png"), cid[s["label"]], "synth", s["id"], s["label"]) for s in syn]
    tr = [d for d in data["train"] if d[1] >= 0]
    va = data["val"]
    print(f"类别 {len(classes)}（真实≥{args.min_n} 的 {sum(1 for v in cnt.values() if v >= args.min_n)}）；真实 train {len(tr)}  合成 {len(syn_d)}  "
          f"val {len(va)}  test {len(data['test'])}  test_new {len(data['test_new'])}", flush=True)
    c2 = Counter(d[1] for d in tr)
    w = torch.tensor([1.0 / c2[d[1]] ** 0.5 for d in tr])
    model = Net(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = random.Random(args.seed)
    best, best_ep = -1.0, -1
    nr = min(args.n_real, len(tr) * 2)
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        batch = [tr[j] for j in torch.multinomial(w, nr, replacement=True).tolist()]
        if syn_d:
            batch += [syn_d[rng.randrange(len(syn_d))] for _ in range(args.n_syn)]
        rng.shuffle(batch)
        for i in range(0, len(batch), 128):
            b = batch[i:i + 128]
            loss = F.cross_entropy(model(to_tensor([augment(d[0], rng) for d in b])), torch.tensor([d[1] for d in b]))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m = evaluate(probs(model, va), va, 0.0)
        print(f"ep {ep:2d} 验证 词表内准确率 {m['acc_in_vocab']:.4f}  {time.time() - t0:.0f}s", flush=True)
        if m["acc_in_vocab"] >= best:
            best, best_ep = m["acc_in_vocab"], ep
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    Pv = probs(model, va)
    thr = 0.99
    for t in np.arange(0.3, 1.0, 0.01):
        if evaluate(Pv, va, t)["acc_at_thr"] >= 0.99:
            thr = float(round(t, 2)); break
    res = {"best_epoch": best_ep, "classes": classes, "thresh": thr, "min_n": args.min_n, "synth": len(syn_d),
           "val": evaluate(Pv, va, thr), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_params": sum(p.numel() for p in model.parameters())}
    for k in ("test", "test_new"):
        res[k] = evaluate(probs(model, data[k]), data[k], thr)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("val", "test", "test_new"):
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in res[k].items()})
    print("门槛", thr)
    # 主 .venv 没装 onnx：用 .venv-dfine 导出（tools/vision/export_hand_art.py，CPU）
    import subprocess
    subprocess.run([str(ROOT / ".venv-dfine/Scripts/python.exe"), "-X", "utf8", str(ROOT / "tools/vision/export_hand_art.py"), str(out)],
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
