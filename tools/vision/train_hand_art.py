"""手牌立绘识别：叠放时卡名被盖住，靠露出来的左边一截（图标 + 立绘，64x120）认是哪张牌。

  .venv/Scripts/python.exe -X utf8 tools/vision/train_hand_art.py [--epochs 40] [--min-n 8]

起因（2026-09-25 手牌卡名复核）：手牌 ≥6 张叠放时卡名在牌面正中、被下一张盖住，任何读字方法都定不了；
露在外面的是每张牌的**左边一截**，而这一截在叠放 / 不叠放时是一样的 → 用"不叠放、卡名读得准"的牌做训练标签，
训出来的分类器直接用在叠放的牌上。
数据：裁图复用 data/vision/hand_awaken_v1/crops（与 hand_names_v1 同 id：同一帧同一张牌）；
标签 = hand_names_v1 里**满宽（≥110px）**名字条的 OCR 读数（非觉醒卡）；划分按对局（两个池相同）。
只训样本 ≥ min_n 的类别；其余不在词表里，靠置信门槛拒识。**不做翻转**。
注意：英雄立绘会随等级变，手牌里多是基础立绘 —— 场上/回手的高等级卡可能认不准，靠门槛兜底。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CROPS = ROOT / "data" / "vision" / "hand_awaken_v1" / "crops"
W, H = 56, 112


def load(min_n):
    N = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/vision/hand_names_v1/pool.jsonl").read_text(encoding="utf-8").splitlines()}
    AL = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/vision/hand_awaken_v1/labels.jsonl").read_text(encoding="utf-8").splitlines()}
    rows = [(i, N[i]["label"], N[i]["split"], N[i]["run"]) for i in N
            if N[i]["roi"][2] >= 110 and N[i]["label"] and AL.get(i, {}).get("label") == 0 and (CROPS / f"{i}.png").exists()]
    cnt = Counter(r[1] for r in rows if r[2] == "train")
    classes = sorted(c for c, v in cnt.items() if v >= min_n)
    cid = {c: k for k, c in enumerate(classes)}
    data = {"train": [], "val": [], "test": []}
    for i, lab, sp, run in rows:
        im = cv2.imdecode(np.fromfile(str(CROPS / f"{i}.png"), np.uint8), cv2.IMREAD_COLOR)
        data[sp].append((im, cid.get(lab, -1), run, i, lab))
    return classes, data


def crop(im, dx=0, dy=0):
    ox, oy = (64 - W) // 2, (120 - H) // 2
    return im[oy + dy:oy + dy + H, ox + dx:ox + dx + W]


def to_tensor(ims):
    x = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ims]).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def augment(im, rng):
    c = crop(im, rng.randint(-4, 4), rng.randint(-4, 4)).astype(np.float32)
    c = c * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
    return np.clip(c, 0, 255).astype(np.uint8)


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 24), blk(24, 48), blk(48, 96), blk(96, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n))

    def forward(self, x):
        return self.head(self.f(x))


def probs(model, data, B=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data), B):
            out.append(F.softmax(model(to_tensor([crop(d[0]) for d in data[i:i + B]])), 1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def evaluate(P, data, thr):
    conf, pred = P.max(1), P.argmax(1)
    y = np.array([d[1] for d in data])
    inv = y >= 0                                  # 类别在词表里的样本
    keep = conf >= thr
    return {"n": int(len(y)), "in_vocab": int(inv.sum()),
            "acc_in_vocab": float((pred[inv] == y[inv]).mean()) if inv.any() else 0.0,
            "coverage": float(keep.mean()),
            "acc_at_thr": float((pred[keep] == y[keep]).mean()) if keep.any() else 0.0,
            "wrong_at_thr": int((keep & (pred != y)).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "vision" / "runs" / "hand_art_v1"))
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    classes, data = load(args.min_n)
    tr = [d for d in data["train"] if d[1] >= 0]
    va, te = data["val"], data["test"]
    print(f"类别 {len(classes)}；train {len(tr)}  val {len(va)}（词表内 {sum(d[1] >= 0 for d in va)}）  test {len(te)}（词表内 {sum(d[1] >= 0 for d in te)}）")
    cnt = Counter(d[1] for d in tr)
    w = torch.tensor([1.0 / cnt[d[1]] ** 0.5 for d in tr])
    model = Net(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = random.Random(args.seed)
    best, best_ep = -1.0, -1
    for ep in range(args.epochs):
        model.train()
        idx = torch.multinomial(w, len(tr), replacement=True).tolist()
        for i in range(0, len(idx), 128):
            b = [tr[j] for j in idx[i:i + 128]]
            loss = F.cross_entropy(model(to_tensor([augment(d[0], rng) for d in b])), torch.tensor([d[1] for d in b]))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m = evaluate(probs(model, va), va, 0.0)
        print(f"ep {ep:2d} 验证 词表内准确率 {m['acc_in_vocab']:.3f}", flush=True)
        if m["acc_in_vocab"] > best:
            best, best_ep = m["acc_in_vocab"], ep
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    Pv = probs(model, va)
    # 门槛只在验证集上定：被采纳的预测（含词表外样本被误收）准确率 ≥ 99%
    thr = 0.99
    for t in np.arange(0.3, 1.0, 0.01):
        m = evaluate(Pv, va, t)
        if m["acc_at_thr"] >= 0.99:
            thr = float(round(t, 2))
            break
    Pt = probs(model, te)                         # 测试集只评这一次
    res = {"best_epoch": best_ep, "classes": classes, "thresh": thr,
           "val": evaluate(Pv, va, thr), "test": evaluate(Pt, te, thr),
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_params": sum(p.numel() for p in model.parameters())}
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("val", "test"):
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[k].items()})
    print("门槛", thr, "参数量", res["n_params"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
