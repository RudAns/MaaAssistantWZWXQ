"""单位等级识别（等级徽章中间的数字）—— 训练 + 验证集选模型 + 测试集只评一次。

  .venv/Scripts/python.exe -X utf8 tools/vision/train_level.py [--epochs 40] [--out data/vision/runs/level_v1]

起因（2026-09-25：读数不准的信息都尝试用图像识别提高）：线上用 OCR 读徽章数字，
复核过的测试集 356 个徽章上**只读对 24.7%、读不到 62.9%、读错 12.4%**；1 级的细"1"几乎读不出（174 个里 4 个）。
Laya 老师题面里满屏 `lv ?` 就是它。

数据：data/vision/level_v1/（徽章裁图复用觉醒进度的 data/vision/awaken_v1/crops，64x64，按对局划分同 pool_v1）
  - 标签：白色笔画连通域切字符 → 聚类 → 主 agent 看簇样例定字 → 灰度小网络补难例（5/6/8/9）→ 按 x 拼成等级；
  - train / val：自动标签（测试集上核出的自动标签错误率 ≈1.1%）；test：主 agent 逐张复核（labels_test.jsonl）。
模型：与觉醒分类器同规模的小 CNN（从头训练，不用预训练），48x48 输入；
  输出头 = 位数（1/2/3）+ 3 个位置的数字（左→右，各 10 类，超出位数的位置不计损失）。
**不做水平翻转**（数字翻转后不是合法字形）。
输出：best.pt、metrics.json（整数完全正确率、按对局 bootstrap 95% CI、分等级段准确率、与 OCR 基线同题对比、门槛/覆盖）、level_cls.onnx。
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
CROPS = ROOT / "data" / "vision" / "awaken_v1" / "crops"
LV = ROOT / "data" / "vision" / "level_v1"
SIZE = 48
MAXD = 3


def load_split(split: str) -> list[tuple]:
    """[(img64, level, match, id)]；test 用复核标签，其余用自动标签。"""
    src = LV / ("labels_test.jsonl" if split == "test" else "badge_labels_auto.jsonl")
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] != split or not r.get("level") or r["level"] >= 10 ** MAXD:
            continue
        im = cv2.imdecode(np.fromfile(str(CROPS / f"{r['id']}.png"), np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (64, 64):
            continue
        out.append((im, int(r["level"]), r["match"], r["id"]))
    return out


def crop(im: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    o = (64 - SIZE) // 2
    return im[o + dy:o + dy + SIZE, o + dx:o + dx + SIZE]


def to_tensor(ims: list[np.ndarray]) -> torch.Tensor:
    x = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ims]).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def augment(im: np.ndarray, rng: random.Random) -> np.ndarray:
    c = crop(im, rng.randint(-3, 3), rng.randint(-3, 3)).astype(np.float32)
    c = c * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
    return np.clip(c, 0, 255).astype(np.uint8)


def targets(level: int) -> tuple[int, list[int]]:
    s = str(level)
    return len(s) - 1, [int(ch) for ch in s] + [-100] * (MAXD - len(s))


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 24), blk(24, 48), blk(48, 96))       # 48 → 6x6x96
        self.pool = nn.AdaptiveAvgPool2d((1, 3))                            # 保留横向位置（左/中/右）
        self.drop = nn.Dropout(0.2)
        self.nd = nn.Linear(96 * 3, MAXD)
        self.dig = nn.Linear(96 * 3, MAXD * 10)

    def forward(self, x):
        h = self.drop(self.pool(self.f(x)).flatten(1))
        return self.nd(h), self.dig(h).view(-1, MAXD, 10)


def decode(nd_logits, dig_logits) -> tuple[list[int], np.ndarray]:
    pn = F.softmax(nd_logits, 1)
    pd = F.softmax(dig_logits, 2)
    n = pn.argmax(1)
    vals, confs = [], []
    for i in range(len(n)):
        k = int(n[i]) + 1
        ds = pd[i, :k].argmax(1).tolist()
        vals.append(int("".join(map(str, ds))))
        confs.append(float(pn[i, k - 1]) * float(np.prod([float(pd[i, j, ds[j]]) for j in range(k)])))
    return vals, np.array(confs)


def predict(model, data, dev, B=256):
    model.eval()
    vs, cs = [], []
    with torch.no_grad():
        for i in range(0, len(data), B):
            x = to_tensor([crop(d[0]) for d in data[i:i + B]]).to(dev)
            v, c = decode(*model(x))
            vs += v
            cs.append(c)
    return vs, (np.concatenate(cs) if cs else np.zeros(0))


def bootstrap(ok: list[bool], groups: list, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups))
    by = {g: [i for i, x in enumerate(groups) if x == g] for g in gs}
    accs = [np.mean([ok[i] for g in rng.choice(gs, len(gs), replace=True) for i in by[g]]) for _ in range(iters)]
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def bucket(lv: int) -> str:
    return "1" if lv == 1 else "2-9" if lv < 10 else "10-99" if lv < 100 else "100+"


def report(data, vals, confs, thresh, ocr=None):
    y = [d[1] for d in data]
    ok = [a == b for a, b in zip(y, vals)]
    keep = [i for i in range(len(y)) if confs[i] >= thresh]
    r = {"n": len(y), "acc": float(np.mean(ok)) if ok else 0.0,
         "acc_ci95_by_match": bootstrap(ok, [d[2] for d in data]) if ok else None,
         "by_bucket": {b: {"n": sum(1 for v in y if bucket(v) == b),
                           "acc": float(np.mean([ok[i] for i in range(len(y)) if bucket(y[i]) == b]) or 0)}
                       for b in ("1", "2-9", "10-99", "100+") if any(bucket(v) == b for v in y)},
         "thresh": thresh, "coverage_at_thresh": len(keep) / max(len(y), 1),
         "acc_at_thresh": float(np.mean([ok[i] for i in keep])) if keep else 0.0,
         "wrong_at_thresh": sum(1 for i in keep if not ok[i]),
         "errors": [{"id": data[i][3], "truth": y[i], "pred": vals[i], "conf": round(float(confs[i]), 3)}
                    for i in range(len(y)) if not ok[i]][:40]}
    if ocr is not None:
        o = [ocr.get(d[3]) for d in data]
        r["ocr_baseline"] = {"correct": sum(1 for a, b in zip(o, y) if a == b) / len(y),
                             "unreadable": sum(1 for a in o if a is None) / len(y),
                             "wrong": sum(1 for a, b in zip(o, y) if a is not None and a != b) / len(y)}
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "data" / "vision" / "runs" / "level_v1"))
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  device {dev}")
    # 等级段极不均衡（1 级占一半以上）：按等级段频率的平方根反比采样
    cnt = {}
    for d in tr:
        cnt[bucket(d[1])] = cnt.get(bucket(d[1]), 0) + 1
    w = torch.tensor([(len(tr) / cnt[bucket(d[1])]) ** 0.5 for d in tr])
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = random.Random(args.seed)
    best, best_ep, curves = -1.0, -1, []
    for ep in range(args.epochs):
        model.train()
        idx = torch.multinomial(w, len(tr), replacement=True).tolist()
        tl = 0.0
        for i in range(0, len(idx), 128):
            b = [tr[j] for j in idx[i:i + 128]]
            x = to_tensor([augment(d[0], rng) for d in b]).to(dev)
            tn, td = zip(*[targets(d[1]) for d in b])
            nd, dg = model(x)
            loss = F.cross_entropy(nd, torch.tensor(tn).to(dev)) + \
                F.cross_entropy(dg.reshape(-1, 10), torch.tensor(td).reshape(-1).to(dev), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        vv, _ = predict(model, va, dev)
        acc = float(np.mean([a == d[1] for a, d in zip(vv, va)]))
        curves.append({"epoch": ep, "train_loss": tl / len(tr), "val_acc": acc})
        print(f"ep {ep:2d} loss {tl / len(tr):.4f} val acc {acc:.4f}")
        if acc > best:
            best, best_ep = acc, ep
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    vv, vc = predict(model, va, dev)
    # 置信门槛只在验证集上定：被采纳的预测准确率 ≥ 99.5% 的最低门槛（线上低于门槛 = 当读不到，宁缺勿错）
    yv = np.array([d[1] for d in va]); pv = np.array(vv)
    thresh = 0.5
    for t in np.arange(0.5, 1.0, 0.01):
        m = vc >= t
        if m.sum() and (pv[m] == yv[m]).mean() >= 0.995:
            thresh = float(round(t, 2))
            break
    ocr_file = LV / "ocr_baseline_test.json"
    ocr = {r["id"]: r["ocr"] for r in json.loads(ocr_file.read_text(encoding="utf-8"))} if ocr_file.exists() else None
    res = {"best_epoch": best_ep, "val": report(va, vv, vc, thresh)}
    tv, tc = predict(model, te, dev)                    # 测试集只评这一次
    res["test"] = report(te, tv, tc, thresh, ocr)
    res["curves"] = curves
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["n_params"] = sum(p.numel() for p in model.parameters())
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    model.eval().cpu()
    try:
        torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), str(out / "level_cls.onnx"),
                          input_names=["x"], output_names=["ndigits", "digits"],
                          dynamic_axes={"x": {0: "n"}, "ndigits": {0: "n"}, "digits": {0: "n"}},
                          opset_version=17, dynamo=False)
    except Exception as e:                # noqa: BLE001
        print(f"[跳过 ONNX 导出] {e!r}（用 .venv-dfine/Scripts/python.exe 导出）")
    for k in ("val", "test"):
        r = res[k]
        print(f"{k}: n={r['n']} acc={r['acc']:.4f} CI{r['acc_ci95_by_match']} 分段{ {b: round(v['acc'], 3) for b, v in r['by_bucket'].items()} } "
              f"| 门槛{thresh} 覆盖{r['coverage_at_thresh']:.3f} 准确{r['acc_at_thresh']:.4f} 错{r['wrong_at_thresh']}")
    if "ocr_baseline" in res["test"]:
        print("OCR 基线(同一测试集):", {k: round(v, 3) for k, v in res["test"]["ocr_baseline"].items()})
    print("参数量", res["n_params"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
