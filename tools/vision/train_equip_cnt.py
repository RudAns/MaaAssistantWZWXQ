"""equip_cnt_v1：英雄身上装备件数 0~3 的小 CNN（2026-10-12）。

  .venv/Scripts/python.exe -X utf8 tools/vision/train_equip_cnt.py [--device cuda]
  .venv-dfine/Scripts/python.exe -X utf8 tools/vision/train_equip_cnt.py --export-only

数据：data/vision_v2/equip_cnt（tools/vision/build_equip_cnt.py）。
  训练：split=train 的全部自动标签（strong + weak；mono_bad 的丢掉）；验证：split=val 且 strong；
  测试：split=test 且 strong 且 mono 正常；若有 labels_test_reviewed.jsonl（人工复核）则只用复核过的、以复核标签为准。
输入：徽章中心下方 104x60 裁图（BGR→RGB，/255），训练随机裁 96x52、评估取中心 96x52 → (N,3,52,96)。
对照：规则 detect_equip_counts 在**同一张整帧截图**上重跑（RGB），取同一格的读数。
输出 data/vision_v2/runs/equip_cnt_v1/{best.pt, metrics.json, equip_cnt_v1.onnx}。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cv2.setNumThreads(4)
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DS = ROOT / "data" / "vision_v2" / "equip_cnt"
RUN = __import__("os").environ.get("EQ_RUN", "equip_cnt_v1")
OUT = ROOT / "data" / "vision_v2" / "runs" / RUN
W, H = 96, 52
NC = 4


def load():
    z = np.load(DS / "crops.npz")["equip"]
    rows = [json.loads(l) for l in (DS / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    rev = DS / "labels_test_reviewed.jsonl"
    fix = {r["id"]: r["label"] for r in map(json.loads, rev.read_text(encoding="utf-8").splitlines())} if rev.exists() else None
    tr, va, te = [], [], []
    for r in rows:
        if r["mono_bad"]:
            continue
        item = (z[r["id"]], r["label"], r)
        if r["split"] == "train":
            tr.append(item)
        elif r["split"] == "val" and r["strong"]:
            va.append(item)
        elif r["split"] == "test":
            if fix is not None:
                if r["id"] in fix and fix[r["id"]] is not None:
                    te.append((z[r["id"]], int(fix[r["id"]]), r))
            elif r["strong"]:
                te.append(item)
    return tr, va, te


def center(im):
    oy, ox = (im.shape[0] - H) // 2, (im.shape[1] - W) // 2
    return im[oy:oy + H, ox:ox + W]


def augment(im, rng):
    oy, ox = rng.randint(0, im.shape[0] - H), rng.randint(0, im.shape[1] - W)
    c = im[oy:oy + H, ox:ox + W].astype(np.float32)
    c = c * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
    c = c * np.array([rng.uniform(0.92, 1.08) for _ in range(3)], np.float32)
    return np.clip(c, 0, 255).astype(np.uint8)


def to_tensor(ims):
    x = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ims]).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 24), blk(24, 48), blk(48, 96))       # 52x96 → 6x12
        self.pool = nn.AdaptiveAvgPool2d((1, 4))                            # 保留横向位置
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(96 * 4, NC)

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.f(x)).flatten(1)))


def predict(model, data, dev, B=512):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(data), B):
            ps.append(F.softmax(model(to_tensor([center(d[0]) for d in data[i:i + B]]).to(dev)), 1).cpu().numpy())
    return np.concatenate(ps) if ps else np.zeros((0, NC))


def bootstrap(ok, groups, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for o, g in zip(ok, groups):
        by[g].append(o)
    gs = list(by)
    accs = [np.mean(sum((by[g] for g in rng.choice(gs, len(gs))), [])) for _ in range(iters)]
    return [round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)]


def rule_on_frames(data):
    """规则在整帧上重跑：{(frame, cell): 读数}"""
    from agent.perception.extractors import detect_equip_counts
    out = {}
    frames = defaultdict(list)
    for d in data:
        frames[d[2]["frame"]].append(d[2]["cell"])
    for fr, cells in frames.items():
        bgr = cv2.imdecode(np.fromfile(str(ROOT / fr), np.uint8), cv2.IMREAD_COLOR)
        got = detect_equip_counts(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        for c in cells:
            out[(fr, c)] = got.get(c)
    return out


def report(data, P, thresh, rule=None):
    y = np.array([d[1] for d in data]); pred = P.argmax(1); conf = P.max(1)
    ok = pred == y
    acc_m = conf >= thresh
    r = {"n": int(len(y)), "by_label_n": {int(k): int(v) for k, v in sorted(Counter(y.tolist()).items())},
         "model_top1": round(float(ok.mean()), 4), "model_top1_ci95": bootstrap(ok.tolist(), [d[2]["match"] for d in data]),
         "thresh": thresh, "model_coverage": round(float(acc_m.mean()), 4),
         "model_acc_accepted": round(float(ok[acc_m].mean()), 4) if acc_m.any() else None,
         "model_by_label": {int(k): round(float(ok[y == k].mean()), 4) for k in sorted(set(y.tolist()))},
         "confusion_model": [[int(((y == a) & (pred == b)).sum()) for b in range(NC)] for a in range(NC)]}
    if rule is not None:
        rv = np.array([-1 if rule.get((d[2]["frame"], d[2]["cell"])) is None else rule[(d[2]["frame"], d[2]["cell"])] for d in data])
        rok = rv == y
        comb = np.where(acc_m, pred, rv)                 # 线上方案：模型 ≥门槛用模型，否则退回规则
        r.update({"rule_acc": round(float(rok.mean()), 4), "rule_ci95": bootstrap(rok.tolist(), [d[2]["match"] for d in data]),
                  "rule_unread": round(float((rv < 0).mean()), 4),
                  "rule_over": round(float(((rv > y) & (rv >= 0)).mean()), 4),
                  "rule_under": round(float(((rv < y) & (rv >= 0)).mean()), 4),
                  "rule_by_label": {int(k): round(float(rok[y == k].mean()), 4) for k in sorted(set(y.tolist()))},
                  "confusion_rule(-1=读不到)": {int(a): dict(Counter(rv[y == a].tolist())) for a in range(NC)},
                  "combined_acc": round(float((comb == y).mean()), 4),
                  "both_paired": {"model_ok_rule_bad": int((ok & ~rok).sum()), "model_bad_rule_ok": int((~ok & rok).sum())}})
    r["errors"] = [{"id": d[2]["id"], "truth": int(d[1]), "pred": int(p), "conf": round(float(c), 3), "frame": d[2]["frame"], "cell": d[2]["cell"]}
                   for d, p, c in zip(data, pred, conf) if p != d[1]][:40]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        m = Net(); m.load_state_dict(torch.load(OUT / "best.pt", map_location="cpu")); m.eval()
        torch.onnx.export(m, torch.zeros(1, 3, H, W), str(OUT / f"{RUN}.onnx"), input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
        print("导出", OUT / f"{RUN}.onnx"); return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tr, va, te = load()
    cnt = Counter(d[1] for d in tr)
    print(f"train {len(tr)} {dict(cnt)}  val {len(va)}  test {len(te)}  device {args.device}", flush=True)
    w = torch.tensor([(len(tr) / cnt[d[1]]) ** 0.5 * (1.0 if d[2]["strong"] else 0.6) for d in tr])
    dev = args.device
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_ep, curves = -1.0, -1, []
    n_ep = min(len(tr), 60000)
    for ep in (range(0) if args.eval_only else range(args.epochs)):
        model.train()
        idx = torch.multinomial(w, n_ep, replacement=True).tolist()
        tl = 0.0
        for i in range(0, len(idx), 256):
            b = [tr[j] for j in idx[i:i + 256]]
            x = to_tensor([augment(d[0], rng) for d in b]).to(dev)
            loss = F.cross_entropy(model(x), torch.tensor([d[1] for d in b]).to(dev), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        P = predict(model, va, dev)
        y = np.array([d[1] for d in va])
        acc = float(np.mean([(P.argmax(1)[y == k] == k).mean() for k in range(NC) if (y == k).any()]))   # 类别平均
        curves.append({"epoch": ep, "loss": round(tl / n_ep, 4), "val_bal_acc": round(acc, 4), "val_acc": round(float((P.argmax(1) == y).mean()), 4)})
        print(curves[-1], flush=True)
        if acc > best:
            best, best_ep = acc, ep
            torch.save(model.state_dict(), OUT / "best.pt")
    model.load_state_dict(torch.load(OUT / "best.pt", map_location=dev))
    Pv = predict(model, va, dev)
    yv = np.array([d[1] for d in va]); pv = Pv.argmax(1); cv = Pv.max(1)
    thresh = 0.9
    for t in np.arange(0.5, 1.0, 0.01):
        m = cv >= t
        if m.sum() and (pv[m] == yv[m]).mean() >= 0.995:
            thresh = max(0.9, float(round(t, 2))); break
    res = {"best_epoch": best_ep, "curves": curves, "train_counts": dict(cnt), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_params": sum(p.numel() for p in model.parameters())}
    res["val"] = report(va, Pv, thresh)
    Pt = predict(model, te, dev)
    res["test"] = report(te, Pt, thresh, rule_on_frames(te))
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    t = res["test"]
    print("test", {k: t[k] for k in ("n", "by_label_n", "model_top1", "model_top1_ci95", "model_coverage", "model_acc_accepted",
                                     "rule_acc", "rule_ci95", "rule_unread", "rule_over", "rule_under", "combined_acc", "both_paired")})


if __name__ == "__main__":
    main()
