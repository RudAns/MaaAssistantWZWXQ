"""觉醒进度分类器（等级徽章外圈亮了几段）—— 训练 + 验证集选模型 + 测试集只评一次。

  .venv/Scripts/python.exe -X utf8 tools/vision/train_awaken.py [--epochs 30] [--out data/vision/runs/awaken_v1]

数据：data/vision/awaken_v1/（tools/vision/build_awaken_pool.py 生成）
  - train：像素规则的预标注当弱标签（3 段那部分抽查过：整圈亮蓝 = 已觉醒）；
  - val / test：主 agent 逐张看图复核（labels_*.jsonl 里 reviewer=claude-visual），剔除被遮挡/非徽章的样本；
  - 划分沿用 pool_v1 的"按对局"划分，测试集里的对局训练时没见过。
类别：0/1/2 = 亮了几段（左→上→右），3 = 已觉醒（整圈连成一条亮环），na = 召唤物（没有觉醒圈，D-FINE 判的 summon）。
**不做水平翻转增强**：左、右两段的先后就是信息，翻转会把 "左亮" 变成 "右亮"（不合法的状态）。
输出：best.pt、metrics.json（val/test 准确率、宏 F1、按对局 bootstrap 95% CI、混淆矩阵、与像素规则的对比）、awaken_cls.onnx。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "vision" / "awaken_v1"
CLASSES = ["0", "1", "2", "3", "na"]
SIZE = 48                      # 64x64 裁图中心裁 48（徽章直径约 44）；训练时在 ±3 像素内随机平移


def load_split(split: str) -> list[tuple[np.ndarray, int, str]]:
    samples = {json.loads(l)["id"]: json.loads(l) for l in open(POOL / "samples.jsonl", encoding="utf-8")}
    out = []
    for l in open(POOL / f"labels_{split}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["label"] == "exclude":
            continue
        im = cv2.imdecode(np.fromfile(str(POOL / "crops" / f"{r['id']}.png"), np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (64, 64):
            continue
        out.append((im, CLASSES.index(str(r["label"])), samples[r["id"]]["match"], r["id"],
                    samples[r["id"]]["prelabel"]))
    return out


def crop(im: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    o = (64 - SIZE) // 2
    return im[o + dy:o + dy + SIZE, o + dx:o + dx + SIZE]


def to_tensor(ims: list[np.ndarray]) -> torch.Tensor:
    x = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ims]).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def augment(im: np.ndarray, rng: random.Random) -> np.ndarray:
    c = crop(im, rng.randint(-3, 3), rng.randint(-3, 3)).astype(np.float32)
    c = c * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)          # 亮度/对比度（背景透色、特效）
    return np.clip(c, 0, 255).astype(np.uint8)


class Net(nn.Module):
    """小 CNN：3 个卷积块 + 全局池化。徽章是固定 UI 贴图，不需要大模型。"""
    def __init__(self, n: int = len(CLASSES)):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 24), blk(24, 48), blk(48, 96))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, n))

    def forward(self, x):
        return self.head(self.f(x))


def evaluate(model, data, dev, B=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(data), B):
            x = to_tensor([crop(d[0]) for d in data[i:i + B]]).to(dev)
            probs.append(F.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, len(CLASSES)))


def macro_f1(y, p, labels) -> float:
    fs = []
    for c in labels:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        if tp + fp + fn == 0:
            continue
        fs.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs)) if fs else 0.0


def bootstrap(y, p, groups, iters=2000, seed=0):
    """按对局重采样（同一局的帧高度相关，不能按帧独立重采样）→ 准确率 95% CI。"""
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups))
    by = {g: [i for i, x in enumerate(groups) if x == g] for g in gs}
    accs = []
    for _ in range(iters):
        idx = [i for g in rng.choice(gs, len(gs), replace=True) for i in by[g]]
        accs.append(np.mean([y[i] == p[i] for i in idx]))
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def report(name, data, probs, thresh):
    y = [d[1] for d in data]
    pred = probs.argmax(1).tolist()
    conf = probs.max(1)
    keep = [i for i in range(len(y)) if conf[i] >= thresh]
    acc_all = float(np.mean([y[i] == pred[i] for i in range(len(y))])) if y else 0.0
    acc_keep = float(np.mean([y[i] == pred[i] for i in keep])) if keep else 0.0
    cm = [[sum(1 for i in range(len(y)) if y[i] == a and pred[i] == b) for b in range(len(CLASSES))]
          for a in range(len(CLASSES))]
    # 像素规则基线：prelabel（None = 读不了；召唤物规则上会给 0，这里按 na 类算它错）
    base = [(CLASSES.index(str(d[4])) if d[4] is not None else -1) for d in data]
    base_cov = float(np.mean([b >= 0 for b in base])) if base else 0.0
    base_acc = float(np.mean([b == t for b, t in zip(base, y) if b >= 0])) if base else 0.0
    return {"n": len(y), "acc": acc_all, "acc_ci95_by_match": bootstrap(y, pred, [d[2] for d in data]),
            "macro_f1": macro_f1(y, pred, range(len(CLASSES))),
            "coverage_at_thresh": len(keep) / max(len(y), 1), "acc_at_thresh": acc_keep, "thresh": thresh,
            "confusion(rows=truth,cols=pred," + ",".join(CLASSES) + ")": cm,
            "pixel_rule": {"coverage": base_cov, "acc_on_covered": base_acc},
            "errors": [{"id": data[i][3], "truth": CLASSES[y[i]], "pred": CLASSES[pred[i]],
                        "conf": round(float(conf[i]), 3)} for i in range(len(y)) if y[i] != pred[i]][:40]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "vision" / "runs" / "awaken_v1"))
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  device {dev}")
    # 类别不均衡（0 段占大头）：按类别频率的平方根反比加权
    cnt = np.bincount([d[1] for d in tr], minlength=len(CLASSES)).astype(np.float32)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32).to(dev)
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = random.Random(args.seed)
    best, best_ep, curves = -1.0, -1, []
    for ep in range(args.epochs):
        model.train()
        idx = list(range(len(tr))); rng.shuffle(idx)
        tl = 0.0
        for i in range(0, len(idx), 128):
            b = [tr[j] for j in idx[i:i + 128]]
            x = to_tensor([augment(d[0], rng) for d in b]).to(dev)
            y = torch.tensor([d[1] for d in b]).to(dev)
            loss = F.cross_entropy(model(x), y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        pv = evaluate(model, va, dev)
        f1 = macro_f1([d[1] for d in va], pv.argmax(1).tolist(), range(len(CLASSES)))
        acc = float(np.mean(pv.argmax(1) == np.array([d[1] for d in va])))
        curves.append({"epoch": ep, "train_loss": tl / len(tr), "val_acc": acc, "val_macro_f1": f1})
        print(f"ep {ep:2d} loss {tl / len(tr):.4f} val acc {acc:.4f} f1 {f1:.4f}")
        if f1 > best:
            best, best_ep = f1, ep
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    pv = evaluate(model, va, dev)
    # 置信度门槛只在验证集上定：保证验证集上被采纳的预测准确率 ≥ 99.5% 的最低门槛
    yv = np.array([d[1] for d in va])
    thresh = 0.5
    for t in np.arange(0.5, 1.0, 0.01):
        m = pv.max(1) >= t
        if m.sum() and (pv.argmax(1)[m] == yv[m]).mean() >= 0.995:
            thresh = float(round(t, 2))
            break
    res = {"best_epoch": best_ep, "classes": CLASSES, "val": report("val", va, pv, thresh)}
    pt = evaluate(model, te, dev)                       # 测试集只评这一次
    res["test"] = report("test", te, pt, thresh)
    res["curves"] = curves
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # 导出 ONNX（感知层用 onnxruntime 跑，和 OCR / D-FINE 一致）
    model.eval().cpu()
    try:                                  # 主 .venv 没装 onnx 包：用 .venv-dfine 跑 export_awaken 那两行（见 docs/paper/ch2_vision.md）
        torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), str(out / "awaken_cls.onnx"),
                          input_names=["x"], output_names=["logits"], dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
                          opset_version=17, dynamo=False)
    except Exception as e:                # noqa: BLE001
        print(f"[跳过 ONNX 导出] {e!r}（用 .venv-dfine/Scripts/python.exe 导出）")
    for k in ("val", "test"):
        r = res[k]
        print(f"{k}: n={r['n']} acc={r['acc']:.4f} CI{r['acc_ci95_by_match']} macroF1={r['macro_f1']:.4f} "
              f"| 门槛{thresh} 覆盖{r['coverage_at_thresh']:.3f} 准确{r['acc_at_thresh']:.4f} "
              f"| 像素规则 覆盖{r['pixel_rule']['coverage']:.3f} 准确{r['pixel_rule']['acc_on_covered']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
