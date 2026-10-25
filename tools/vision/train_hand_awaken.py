"""手牌觉醒卡识别（二分类）—— 训练 + 验证集定门槛 + 测试集只评一次。

  .venv/Scripts/python.exe -X utf8 tools/vision/train_hand_awaken.py [--epochs 30]

数据：data/vision/hand_awaken_v1/（build_hand_awaken.py 生成；labels.jsonl：颜色候选按组人工标注，非候选默认 0、抽样核对）。
基线（颜色判据 awaken_tint_spans）：召回 ≈100%、精度只有 0.24~0.38 —— 每点开 3 次有 2 次是白看（立绘偏金/蓝的卡、翩跹舞步、效果牌）。
目标：**召回不降（漏认觉醒卡代价大：觉醒回手没被认出、空位被别人占）**，精度尽量高 → 门槛在验证集上取"召回 = 100% 的最高门槛"。
模型：与觉醒进度/等级识别同风格的小 CNN，从头训练；输入 64x120 手牌露出部分（训练时 ±3 像素平移、亮度扰动），**不做翻转**。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "vision" / "hand_awaken_v1"
W, H = 56, 112          # 64x120 裁图中心取 56x112，训练时 ±4 平移


def load(split):
    out = []
    for line in (D / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] != split or r["label"] is None:
            continue
        im = cv2.imdecode(np.fromfile(str(D / "crops" / f"{r['id']}.png"), np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (120, 64):
            continue
        out.append((im, int(r["label"]), r["run"], r["id"], bool(r["rule"])))
    return out


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
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.f(x))


def probs(model, data, dev, B=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data), B):
            out.append(F.softmax(model(to_tensor([crop(d[0]) for d in data[i:i + B]]).to(dev)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def pr(y, pred):
    tp = int(sum(1 for a, b in zip(y, pred) if a and b)); fp = int(sum(1 for a, b in zip(y, pred) if not a and b))
    fn = int(sum(1 for a, b in zip(y, pred) if a and not b))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "vision" / "runs" / "hand_awaken_v1"))
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cpu"
    tr, va, te = load("train"), load("val"), load("test")
    print(f"train {len(tr)}（正 {sum(d[1] for d in tr)}） val {len(va)}（正 {sum(d[1] for d in va)}） test {len(te)}（正 {sum(d[1] for d in te)}）")
    n1 = sum(d[1] for d in tr)
    w = torch.tensor([(len(tr) / max(1, n1)) ** 0.5 if d[1] else 1.0 for d in tr])     # 正样本稀少：平方根反比过采样
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = random.Random(args.seed)
    best, best_ep, curves = -1.0, -1, []
    yv = [d[1] for d in va]
    for ep in range(args.epochs):
        model.train()
        idx = torch.multinomial(w, len(tr), replacement=True).tolist()
        tl = 0.0
        for i in range(0, len(idx), 128):
            b = [tr[j] for j in idx[i:i + 128]]
            loss = F.cross_entropy(model(to_tensor([augment(d[0], rng) for d in b]).to(dev)),
                                   torch.tensor([d[1] for d in b]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        pv = probs(model, va, dev)
        # 选模型：召回 100% 前提下的最佳精度（再看 AP 近似）
        pos = sorted(pv[[i for i, y in enumerate(yv) if y]])
        thr = float(pos[0]) if pos else 0.5
        m = pr(yv, [p >= thr for p in pv])
        score = m["precision"] + 0.001 * float(np.mean(pos) if pos else 0)
        curves.append({"epoch": ep, "loss": tl / len(tr), "val_prec_at_full_recall": m["precision"], "thr": thr})
        print(f"ep {ep:2d} loss {tl / len(tr):.4f} 验证 召回100%时精度 {m['precision']:.3f}（门槛 {thr:.3f}）")
        if score > best:
            best, best_ep = score, ep
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    pv = probs(model, va, dev)
    pos = sorted(pv[[i for i, y in enumerate(yv) if y]])
    thr = float(max(0.05, min(0.5, pos[0] * 0.8))) if pos else 0.5      # 召回 100% 的门槛再放宽 20% 留余量
    res = {"best_epoch": best_ep, "thresh": thr,
           "val": {"model": pr(yv, [p >= thr for p in pv]), "color_rule": pr(yv, [d[4] for d in va])}}
    pt = probs(model, te, dev)                                            # 测试集只评这一次
    yt = [d[1] for d in te]
    res["test"] = {"model": pr(yt, [p >= thr for p in pt]), "color_rule": pr(yt, [d[4] for d in te]),
                   "errors": [{"id": d[3], "truth": d[1], "p": round(float(p), 3)} for d, p in zip(te, pt)
                              if (p >= thr) != bool(d[1])][:40]}
    res["curves"] = curves
    res["n_params"] = sum(p.numel() for p in model.parameters())
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("val", "test"):
        print(k, "模型", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[k]["model"].items()},
              "| 颜色判据", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[k]["color_rule"].items()})
    print("门槛", round(thr, 3), "参数量", res["n_params"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
