"""场上单位认人（格子上站的英雄是谁）—— 训练 + 验证集选模型/定门槛 + 测试集只评一次 + 导出 ONNX + 延迟基准。

  .venv/Scripts/python.exe -X utf8 tools/vision/train_unit_id.py                       # 主模型
  .venv/Scripts/python.exe -X utf8 tools/vision/train_unit_id.py --max-train-level 5 --out data/vision/runs/unit_id_lowlv
        # 等级对外观的影响：只拿 ≤5 级的样本训练，看测试集高等级样本认不认得出
  .venv/Scripts/python.exe -X utf8 tools/vision/train_unit_id.py --bench-only          # 只测延迟

数据：data/vision/unit_id_v1/（tools/vision/build_unit_id.py）。按对局划分；测试集若有 labels_test.jsonl（人工复核）则以它为准。
输入：80x128 裁图（D-FINE 框左右各扩 6px、去掉框顶 15%、底部扩 4px，BGR 转 RGB，/255，**不做均值方差归一化**），
      训练随机裁 72x120，评估/线上取中心 72x120 → 模型输入 (N,3,120,72) float32。
类别：训练集样本数 ≥ MIN_TRAIN 的英雄 + "其他"（其余英雄合成一类，预测成"其他"或低于门槛 = 认不出，交回点开看）。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))  # noqa: E401,E702
DS = ROOT / "data" / "vision" / "unit_id_v1"
ASSET = ROOT / "assets" / "model" / "vision" / "unit_id_v1.onnx"
W, H = 72, 120
MIN_TRAIN = 15
OTHER = "其他"


def load(max_train_level=None):
    rows = [json.loads(l) for l in (DS / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    rev = DS / "labels_test.jsonl"
    if rev.exists():                      # 人工复核：{"id","name"}，name=null 表示看不清/标签错且无法确定 → 剔除
        fix = {r["id"]: r["name"] for r in map(json.loads, rev.read_text(encoding="utf-8").splitlines())}
        rows = [dict(r, name=fix[r["id"]]) if r["id"] in fix else r for r in rows]
        rows = [r for r in rows if r["name"]]
    tr = [r for r in rows if r["split"] == "train"]
    if max_train_level is not None:
        tr = [r for r in tr if r.get("level") and r["level"] <= max_train_level]
    cnt = Counter(r["name"] for r in tr)
    classes = sorted(n for n, c in cnt.items() if c >= MIN_TRAIN) + [OTHER]
    cid = {n: i for i, n in enumerate(classes)}
    ims = {}

    def pack(rs):
        out = []
        for r in rs:
            if r["id"] not in ims:
                ims[r["id"]] = cv2.imdecode(np.fromfile(str(DS / "crops" / f"{r['id']}.png"), np.uint8), cv2.IMREAD_COLOR)
            out.append((ims[r["id"]], cid.get(r["name"], cid[OTHER]), r))
        return out
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]
    return classes, pack(tr), pack(va), pack(te)


def center(im):
    oy, ox = (im.shape[0] - H) // 2, (im.shape[1] - W) // 2
    return im[oy:oy + H, ox:ox + W]


def augment(im, rng):
    oy, ox = rng.randint(0, im.shape[0] - H), rng.randint(0, im.shape[1] - W)
    c = im[oy:oy + H, ox:ox + W].astype(np.float32)
    c = c * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    c = c * np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)     # 轻微偏色（棋盘地图底色不同）
    if rng.random() < 0.3:                                                         # 模拟相邻单位/特效遮挡
        x0, y0 = rng.randint(0, W - 16), rng.randint(0, H - 16)
        c[y0:y0 + rng.randint(8, 30), x0:x0 + rng.randint(8, 24)] = rng.uniform(0, 255)
    return np.clip(c, 0, 255).astype(np.uint8)


def to_tensor(ims):
    x = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ims]).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 192))   # 120x72 → 7x4
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(192, n)

    def forward(self, x):
        return self.fc(self.drop(F.adaptive_avg_pool2d(self.f(x), 1).flatten(1)))


def predict(model, data, dev, B=256):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(data), B):
            ps.append(F.softmax(model(to_tensor([center(d[0]) for d in data[i:i + B]]).to(dev)), 1).cpu().numpy())
    return np.concatenate(ps) if ps else np.zeros((0, 1))


def bootstrap(ok, groups, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for o, g in zip(ok, groups):
        by[g].append(o)
    gs = list(by)
    accs = [np.mean(sum((by[g] for g in rng.choice(gs, len(gs))), [])) for _ in range(iters)]
    return [round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)]


def lv_bucket(lv):
    return "?" if not lv else "1" if lv == 1 else "2-9" if lv < 10 else "10-29" if lv < 30 else "30+"


def report(classes, data, P, thresh):
    oth = classes.index(OTHER)
    y = np.array([d[1] for d in data]); pred = P.argmax(1); conf = P.max(1)
    ok = pred == y
    accept = (conf >= thresh) & (pred != oth)
    r = {"n": int(len(y)), "n_known": int((y != oth).sum()), "top1": float(ok.mean()),
         "top1_ci95_by_match": bootstrap(ok.tolist(), [d[2]["match"] for d in data]),
         "top1_known_only": float(ok[y != oth].mean()) if (y != oth).any() else None,
         "thresh": thresh, "coverage": float(accept.mean()),
         "coverage_known": float(accept[y != oth].mean()) if (y != oth).any() else None,
         "acc_accepted": float(ok[accept].mean()) if accept.any() else None,
         "wrong_accepted": int((accept & ~ok).sum()),
         "curve": [], "by_level": {}, "by_class": {}, "confusions": []}
    for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995):
        a = (conf >= t) & (pred != oth)
        r["curve"].append({"t": t, "coverage": round(float(a.mean()), 4),
                           "acc": round(float(ok[a].mean()), 4) if a.any() else None})
    for b in ("1", "2-9", "10-29", "30+", "?"):
        m = np.array([lv_bucket(d[2].get("level")) == b for d in data]) & (y != oth)
        if m.any():
            a = m & accept
            r["by_level"][b] = {"n": int(m.sum()), "top1": round(float(ok[m].mean()), 4),
                                "coverage": round(float(a.sum() / m.sum()), 4),
                                "acc_accepted": round(float(ok[a].mean()), 4) if a.any() else None}
    for i, c in enumerate(classes):
        m = y == i
        if m.any():
            r["by_class"][c] = {"n": int(m.sum()), "top1": round(float(ok[m].mean()), 4)}
    conf_pairs = Counter((classes[a], classes[b]) for a, b in zip(y, pred) if a != b)
    r["confusions"] = [{"truth": a, "pred": b, "n": n} for (a, b), n in conf_pairs.most_common(15)]
    r["errors"] = [{"id": d[2]["id"], "truth": classes[d[1]], "pred": classes[p], "conf": round(float(c), 3)}
                   for d, p, c in zip(data, pred, conf) if p != d[1]][:80]
    return r


def pick_thresh(classes, va, P, target=0.98):
    oth = classes.index(OTHER)
    y = np.array([d[1] for d in va]); pred = P.argmax(1); conf = P.max(1)
    for t in np.arange(0.5, 1.0, 0.005):              # 下限 0.5：没见过的英雄（其他）常以 0.2~0.35 落到某个像的类上
        a = (conf >= t) & (pred != oth)
        if a.sum() >= 20 and (pred[a] == y[a]).mean() >= target:
            return float(round(t, 3))
    return 0.995


def bench(onnx_path, reps=300):
    import onnxruntime as ort
    res = {}
    so = ort.SessionOptions(); so.log_severity_level = 3
    for ep, prov in (("cpu", ["CPUExecutionProvider"]), ("gpu_cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])):
        try:
            if ep != "cpu":
                from agent.perception.onnx_ocr import _load_gpu_dlls
                _load_gpu_dlls(ort)
            s = ort.InferenceSession(str(onnx_path), so, providers=prov)
            if s.get_providers()[0] != prov[0]:
                res[ep] = f"退回了 {s.get_providers()[0]}"
                continue
            for n in (1, 7):
                x = np.random.rand(n, 3, H, W).astype(np.float32)
                for _ in range(20):
                    s.run(None, {"x": x})
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter(); s.run(None, {"x": x}); ts.append((time.perf_counter() - t0) * 1000)
                res[f"{ep}_batch{n}"] = {"p50_ms": round(float(np.percentile(ts, 50)), 2),
                                         "p90_ms": round(float(np.percentile(ts, 90)), 2),
                                         "max_ms": round(float(np.max(ts)), 2)}
        except Exception as e:  # noqa: BLE001
            res[ep] = f"失败 {e!r}"[:200]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-train-level", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "vision" / "runs" / "unit_id_v1"))
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="不训练，读 out/best.pt 重新评估/导出")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.bench_only:
        print(json.dumps(bench(ASSET), ensure_ascii=False, indent=1)); return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    classes, tr, va, te = load(args.max_train_level)
    print(f"类别 {len(classes)}  train {len(tr)}  val {len(va)}  test {len(te)}  device {args.device}")
    cnt = Counter(d[1] for d in tr)
    w = torch.tensor([(len(tr) / cnt[d[1]]) ** 0.5 for d in tr])       # 频率平方根反比采样
    dev = args.device
    model = Net(len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = random.Random(args.seed)
    best, best_ep, curves = -1.0, -1, []
    for ep in (range(0) if args.eval_only else range(args.epochs)):
        model.train()
        idx = torch.multinomial(w, len(tr), replacement=True).tolist()
        tl = 0.0
        for i in range(0, len(idx), 64):                                 # batch 小一点，别占满实战局在用的显卡
            b = [tr[j] for j in idx[i:i + 64]]
            x = to_tensor([augment(d[0], rng) for d in b]).to(dev)
            loss = F.cross_entropy(model(x), torch.tensor([d[1] for d in b]).to(dev), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        P = predict(model, va, dev)
        acc = float((P.argmax(1) == np.array([d[1] for d in va])).mean())
        curves.append({"epoch": ep, "train_loss": round(tl / len(tr), 4), "val_top1": round(acc, 4)})
        print(f"ep {ep:2d} loss {tl / len(tr):.4f} val top1 {acc:.4f}", flush=True)
        if acc > best:
            best, best_ep = acc, ep
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    Pv = predict(model, va, dev)
    thresh = pick_thresh(classes, va, Pv)                                # 门槛只在验证集上定
    res = {"classes": classes, "train_counts": {classes[k]: v for k, v in sorted(cnt.items())},
           "max_train_level": args.max_train_level, "best_epoch": best_ep,
           "val": report(classes, va, Pv, thresh)}
    Pt = predict(model, te, dev)                                         # 测试集只评这一次
    res["test"] = report(classes, te, Pt, thresh)
    res["curves"] = curves
    res["n_params"] = sum(p.numel() for p in model.parameters())
    res["input"] = {"shape": [None, 3, H, W], "color": "RGB", "scale": "x/255", "crop": "80x128 中心裁 72x120"}
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not args.no_export:
        model.eval().cpu()
        onnx = out / "unit_id.onnx"
        try:
            torch.onnx.export(model, torch.zeros(1, 3, H, W), str(onnx), input_names=["x"], output_names=["logits"],
                              dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
        except Exception as e:  # noqa: BLE001
            print(f"[跳过 ONNX 导出] {e!r}（主 .venv 没装 onnx：用 .venv-dfine/Scripts/python.exe 加 --eval-only 导出）")
        if onnx.exists():
            try:
                res["bench"] = bench(onnx)
            except Exception as e:  # noqa: BLE001
                res["bench"] = repr(e)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("val", "test"):
        r = res[k]
        print(f"{k}: n={r['n']} top1={r['top1']:.4f} CI{r['top1_ci95_by_match']} | 门槛{thresh} 覆盖{r['coverage']:.3f} "
              f"采纳准确{r['acc_accepted']} 采纳错{r['wrong_accepted']} 分等级{r['by_level']}")
    print("bench", res.get("bench"))


if __name__ == "__main__":
    main()
