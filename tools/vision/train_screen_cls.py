"""界面分类（整屏截图 -> 哪种界面）—— 训练 + 验证集定门槛 + 测试集只评一次 + 导出 ONNX + 延迟基准。

  .venv/Scripts/python.exe -X utf8 tools/vision/train_screen_cls.py --out data/vision/runs/screen_cls_v1
  .venv/Scripts/python.exe -X utf8 tools/vision/train_screen_cls.py --stage1 --out data/vision/runs/screen_cls_s1   # 只用弱标签（初始模型）
  .venv/Scripts/python.exe -X utf8 tools/vision/train_screen_cls.py --bench-only                                  # 只测 ONNX 延迟

数据：data/vision/screen_cls_v1/（build_screen_cls.py），标签优先级：
  labels_review.jsonl（val/test 逐张人工复核；train 里人工抽查改过的也在这）
  > labels_train_auto.jsonl（train 里 unknown 帧按特征聚类后整簇人工定类）
  > 弱标签（旧 OCR 判据）。弱标签是 unknown 且没有人工/聚类标签的样本不参加训练。
  标签 "drop" = 人工判为无法归类/坏图，丢弃。
  --stage1：完全只用弱标签（排除 unknown），用来做初始模型、给复核排序和聚类提特征。
模型：从头训练的小 CNN（不用预训练），输入 256x144 整屏缩图（RGB /255），无水平翻转（界面左右不对称）。
测试集：只用人工复核过的标签评；另外报"旧 OCR 判据（=弱标签）在同一测试集上的准确率"作对比。
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
DS = ROOT / "data" / "vision" / "screen_cls_v1"
W, H = 256, 144
CLASSES = ["prepare", "prepare_shop", "board_idle", "battle", "auction", "talent", "card_pick", "lord_boost",
           "hero_pick", "loading", "settle", "settle_final", "post_match", "lobby", "mode_lobby", "match_found", "settings",
           "lord_panel", "hero_panel", "enlarged_detail", "equip_popup", "reward_popup", "transition", "notice", "solo_popup", "other"]
CI = {c: i for i, c in enumerate(CLASSES)}


# 弱标签按来源改写：装备四选一证据帧（equip_popup_*）复核下来绝大多数是弹窗本身，不是普通 prepare
WEAK_REMAP = {"equip_popup": "equip_popup"}


def val_half(match: str) -> str:
    """复核过的 val 按对局再分两半：fit 进训练（新类别只在复核集里有），sel 用于选模型 / 定门槛。"""
    import hashlib
    return "fit" if int(hashlib.md5(("half_" + match).encode()).hexdigest(), 16) % 2 == 0 else "sel"


def read_jsonl(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r["label"]
    return out


def load(stage1: bool):
    samples = [json.loads(l) for l in (DS / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    thumbs = np.load(DS / "thumbs.npy")               # 2.2GB 全读进内存（mmap 随机读太慢）
    review = {} if stage1 else read_jsonl(DS / "labels_review.jsonl")
    auto = {} if stage1 else read_jsonl(DS / "labels_train_auto.jsonl")
    for s in samples:
        weak = WEAK_REMAP.get(s["src"], s["weak"])
        lab = review.get(s["id"]) or (auto.get(s["id"]) if s["split"] == "train" else None) or weak
        s["reviewed"] = s["id"] in review
        s["label"] = lab
    return samples, thumbs, review


def usable(s) -> bool:
    return s["label"] in CI


CROPS = None          # --crops 时 = data/vision/screen_cls_v1/crops.npy（N x 64 x 224 x 3），第二路输入


class Net(nn.Module):
    def __init__(self, n: int, use_crops: bool = False):
        super().__init__()
        self.use_crops = use_crops

        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        # 144x256 -> 72x128 -> 36x64 -> 18x32 -> 9x16 -> 5x8
        self.f = nn.Sequential(blk(3, 24), blk(24, 48), blk(48, 96), blk(96, 144), blk(144, 192))
        self.drop = nn.Dropout(0.25)
        # 全局平均 + 保留粗位置（5x4 网格）：界面的区别常在"某个角落有没有某个按钮"，纯全局平均会丢位置。
        # 末层特征图 5x8，池化尺寸必须整除它（ONNX 导出不支持非整除的自适应池化）
        self.pool = nn.AdaptiveAvgPool2d((5, 4))
        self.red = nn.Conv2d(192, 16, 1)
        extra = 0
        if use_crops:
            # 第二路：固定区域高分辨率裁图（64x224）-> 4x14 特征图 -> 保留位置的 1x7 池化
            def cb(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                     nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
            self.cf = nn.Sequential(cb(3, 16), cb(16, 32), cb(32, 64), cb(64, 96))      # 64x224 -> 4x14
            self.cpool = nn.AdaptiveAvgPool2d((1, 7))
            extra = 96 * 7
        self.fc = nn.Linear(192 + 16 * 20 + extra, n)

    def feats(self, x, c=None):
        h = self.f(x)
        g = h.mean((2, 3))
        loc = F.relu(self.red(self.pool(h))).flatten(1)
        parts = [g, loc]
        if self.use_crops:
            parts.append(self.cpool(self.cf(c)).flatten(1))
        return torch.cat(parts, 1)

    def forward(self, x, c=None):
        return self.fc(self.drop(self.feats(x, c)))


def to_tensor(ims: np.ndarray) -> torch.Tensor:
    x = ims[..., ::-1].astype(np.float32) / 255.0            # BGR -> RGB
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2)


def augment(im: np.ndarray, rng: random.Random) -> np.ndarray:
    dx, dy = rng.randint(-6, 6), rng.randint(-4, 4)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    s = rng.uniform(0.95, 1.05)
    if abs(s - 1) > 1e-3:
        M = np.float32([[s, 0, dx + (1 - s) * W / 2], [0, s, dy + (1 - s) * H / 2]])
    im = cv2.warpAffine(im, M, (W, H), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    im = im * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    im = im * np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)
    if rng.random() < 0.3:                                      # 轻微模糊（MaaFW 缩图/动画拖影）
        im = cv2.GaussianBlur(im, (3, 3), 0)
    return np.clip(im, 0, 255).astype(np.uint8)


def augment_crop(im: np.ndarray, rng: random.Random) -> np.ndarray:
    M = np.float32([[1, 0, rng.randint(-3, 3)], [0, 1, rng.randint(-3, 3)]])
    im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    im = im * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    return np.clip(im, 0, 255).astype(np.uint8)


def ctensor(idxs, dev, rng=None):
    if CROPS is None:
        return None
    ims = [CROPS[j] if rng is None else augment_crop(CROPS[j], rng) for j in idxs]
    return to_tensor(np.stack(ims)).to(dev)


def predict(model, thumbs, idxs, dev, B=256, want_feats=False):
    model.eval()
    ps, fs = [], []
    with torch.no_grad():
        for i in range(0, len(idxs), B):
            x = to_tensor(np.stack([thumbs[j] for j in idxs[i:i + B]])).to(dev)
            f = model.feats(x, ctensor(idxs[i:i + B], dev))
            ps.append(F.softmax(model.fc(f), 1).cpu().numpy())
            if want_feats:
                fs.append(f.cpu().numpy())
    P = np.concatenate(ps) if ps else np.zeros((0, len(CLASSES)))
    return (P, np.concatenate(fs)) if want_feats else P


def bootstrap(ok, groups, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups))
    by = {g: [i for i, x in enumerate(groups) if x == g] for g in gs}
    accs = [np.mean([ok[i] for g in rng.choice(gs, len(gs), replace=True) for i in by[g]]) for _ in range(iters)]
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def report(ss, P, thresh):
    y = np.array([CI[s["label"]] for s in ss])
    pred, conf = P.argmax(1), P.max(1)
    ok = (pred == y)
    per = {}
    for c in sorted(set(y) | set(pred)):
        tp = int(((pred == c) & (y == c)).sum())
        per[CLASSES[c]] = {"n": int((y == c).sum()), "pred_n": int((pred == c).sum()),
                           "precision": tp / max(int((pred == c).sum()), 1), "recall": tp / max(int((y == c).sum()), 1)}
    conf_m = Counter((CLASSES[a], CLASSES[b]) for a, b in zip(y, pred) if a != b)
    keep = conf >= thresh
    weak_ok = [s["weak"] == s["label"] for s in ss]
    return {"n": len(ss), "acc": float(ok.mean()), "acc_ci95_by_match": bootstrap(list(ok), [s["match"] for s in ss]),
            "per_class": per, "confusions": [[a, b, n] for (a, b), n in conf_m.most_common(40)],
            "thresh": thresh, "coverage_at_thresh": float(keep.mean()),
            "acc_at_thresh": float(ok[keep].mean()) if keep.any() else 0.0, "wrong_at_thresh": int((~ok & keep).sum()),
            "weak_label_acc": float(np.mean(weak_ok)),
            "weak_label_acc_ci95_by_match": bootstrap(weak_ok, [s["match"] for s in ss]),
            "weak_confusions": [[a, b, n] for (a, b), n in Counter((s["label"], s["weak"]) for s in ss
                                                                     if s["weak"] != s["label"]).most_common(40)],
            "errors": [{"id": s["id"], "truth": s["label"], "pred": CLASSES[int(p)], "conf": round(float(c), 3), "weak": s["weak"]}
                       for s, p, c, o in zip(ss, pred, conf, ok) if not o][:80]}


def bench(onnx_path: Path, n=300):
    import onnxruntime as ort
    out = {}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    img = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
    import sys
    sys.path.insert(0, str(ROOT))
    from agent.perception import screen_cls as sc
    ins = [i.name for i in s.get_inputs()]
    feed_rand = {"x": np.random.rand(1, 3, H, W).astype(np.float32)}
    if "c" in ins:
        feed_rand["c"] = np.random.rand(1, 3, 64, 224).astype(np.float32)
    for name, fn in [("model_only", lambda: s.run(None, feed_rand)),
                     ("preprocess+model", lambda: s.run(None, sc.make_inputs(img, ins)))]:
        for _ in range(20):
            fn()
        ts = []
        for _ in range(n):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
        out[name] = {"p50": round(float(np.percentile(ts, 50)), 2), "p90": round(float(np.percentile(ts, 90)), 2),
                     "max": round(float(np.max(ts)), 2)}
    return out


def export_onnx(out: Path):
    sd = torch.load(out / "best.pt", map_location="cpu")
    uc = any(k.startswith("cf.") for k in sd)
    model = Net(len(CLASSES), uc)
    model.load_state_dict(sd)
    model.eval()
    x, c = torch.rand(4, 3, H, W), torch.rand(4, 3, 64, 224)
    args = (x, c) if uc else (x,)
    names = ["x", "c"] if uc else ["x"]
    torch.onnx.export(model, args, str(out / "screen_cls.onnx"), input_names=names, output_names=["logits"],
                      dynamic_axes={**{k: {0: "n"} for k in names}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    # 与 PyTorch 输出逐项核对
    import onnxruntime as ort
    with torch.no_grad():
        ref = model(*args).numpy()
    got = ort.InferenceSession(str(out / "screen_cls.onnx"), providers=["CPUExecutionProvider"]).run(
        None, {k: v.numpy() for k, v in zip(names, args)})[0]
    print("ONNX 导出完成，与 PyTorch 最大误差", float(np.abs(ref - got).max()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--crops", action="store_true", help="加第二路输入：固定区域高分辨率裁图（build_screen_cls_crops.py）")
    ap.add_argument("--fit-rep", type=int, default=3, help="复核过的 val-fit 样本在训练里重复几次")
    ap.add_argument("--out", default=str(ROOT / "data" / "vision" / "runs" / "screen_cls_v1"))
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--export-only", action="store_true", help="只从 best.pt 导出 ONNX + 测延迟（用 .venv-dfine 跑）")
    ap.add_argument("--eval-only", action="store_true", help="不训练，直接用 best.pt 评估并写 preds/metrics")
    ap.add_argument("--no-test", action="store_true", help="不评测试集（复核/清洗过程中的中间模型）")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        export_onnx(out)
        m = json.loads((out / "metrics.json").read_text(encoding="utf-8")) if (out / "metrics.json").exists() else {}
        m["latency_cpu_ms"] = bench(out / "screen_cls.onnx")
        (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        print("延迟", m["latency_cpu_ms"])
        return 0
    if args.bench_only:
        print(json.dumps(bench(out / "screen_cls.onnx"), indent=1))
        return 0
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = args.device
    global CROPS
    if args.crops:
        CROPS = np.load(DS / "crops.npy")
    samples, thumbs, review = load(args.stage1)
    tr = [s for s in samples if s["split"] == "train" and usable(s)]
    if args.stage1:
        va = [s for s in samples if s["split"] == "val" and usable(s)]
        te = []
    else:                                                   # val/test 只用人工复核过的
        # 复核过的 val 一半（按对局）并进训练：board_idle / reward_popup / transition 等新类别在弱标签里根本不存在
        fit = [s for s in samples if s["split"] == "val" and s["reviewed"] and usable(s) and val_half(s["match"]) == "fit"]
        tr = tr + fit * args.fit_rep
        va = [s for s in samples if s["split"] == "val" and s["reviewed"] and usable(s) and val_half(s["match"]) == "sel"]
        te = [] if args.no_test else [s for s in samples if s["split"] == "test" and s["reviewed"] and usable(s)]
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  device {dev}")
    print("train 类别:", dict(Counter(s["label"] for s in tr).most_common()))
    cnt = Counter(s["label"] for s in tr)
    w = torch.tensor([(len(tr) / cnt[s["label"]]) ** 0.5 for s in tr])      # 频率平方根反比采样
    model = Net(len(CLASSES), CROPS is not None).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    steps = args.epochs * ((len(tr) + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)
    rng = random.Random(args.seed)
    best, best_ep, curves = -1.0, -1, []
    yv = np.array([CI[s["label"]] for s in va])
    for ep in range(0 if args.eval_only else args.epochs):
        model.train()
        idx = torch.multinomial(w, len(tr), replacement=True).tolist()
        tl = 0.0
        for i in range(0, len(idx), 128):
            b = [tr[j] for j in idx[i:i + 128]]
            x = to_tensor(np.stack([augment(np.asarray(thumbs[s["idx"]]), rng) for s in b])).to(dev)
            y = torch.tensor([CI[s["label"]] for s in b]).to(dev)
            loss = F.cross_entropy(model(x, ctensor([s["idx"] for s in b], dev, rng)), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(b)
        P = predict(model, thumbs, [s["idx"] for s in va], dev)
        acc = float((P.argmax(1) == yv).mean()) if len(va) else 0.0
        # 按类平均召回做选模标准：大类（prepare/battle）占多数，只看总准确率会忽视小类
        rec = np.mean([((P.argmax(1) == c) & (yv == c)).sum() / (yv == c).sum() for c in set(yv)]) if len(va) else 0.0
        curves.append({"epoch": ep, "train_loss": tl / len(tr), "val_acc": acc, "val_macro_recall": float(rec)})
        print(f"ep {ep:2d} loss {tl / len(tr):.4f} val acc {acc:.4f} macro-recall {rec:.4f}", flush=True)
        score = acc + rec
        if score > best:
            best, best_ep = score, ep
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    Pv = predict(model, thumbs, [s["idx"] for s in va], dev)
    # 门槛只在验证集定：被采纳预测准确率 ≥ 99.5% 的最低门槛（低于门槛 = 退回旧 OCR 判据）
    thresh = 0.5
    okv = Pv.argmax(1) == yv
    for t in np.arange(0.3, 1.0, 0.01):
        m = Pv.max(1) >= t
        if m.sum() and okv[m].mean() >= 0.995:
            thresh = float(round(t, 2))
            break
    curve_v = []
    for t in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        m = Pv.max(1) >= t
        curve_v.append({"t": t, "coverage": float(m.mean()), "acc": float(okv[m].mean()) if m.any() else 0.0})
    res = {"best_epoch": best_ep, "stage1": args.stage1, "classes": CLASSES, "val": report(va, Pv, thresh),
           "val_curve": curve_v}
    if te:
        Pt = predict(model, thumbs, [s["idx"] for s in te], dev)       # 测试集只评这一次
        res["test"] = report(te, Pt, thresh)
        okt = Pt.argmax(1) == np.array([CI[s["label"]] for s in te])
        res["test_curve"] = [{"t": t, "coverage": float((Pt.max(1) >= t).mean()),
                              "acc": float(okt[Pt.max(1) >= t].mean()) if (Pt.max(1) >= t).any() else 0.0}
                             for t in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)]
    # 全部样本的预测（给复核排序 / 找弱标签不一致 / 聚类用）
    allidx = [s["idx"] for s in samples]
    Pa, Fa = predict(model, thumbs, allidx, dev, want_feats=True)
    np.save(out / "feats.npy", Fa.astype(np.float16))
    with open(out / "preds.jsonl", "w", encoding="utf-8") as f:
        for s, p in zip(samples, Pa):
            o = np.argsort(-p)[:3]
            f.write(json.dumps({"id": s["id"], "split": s["split"], "weak": s["weak"], "label": s["label"],
                                "pred": CLASSES[o[0]], "conf": round(float(p[o[0]]), 4),
                                "top3": [[CLASSES[k], round(float(p[k]), 3)] for k in o]}, ensure_ascii=False) + "\n")
    res["curves"] = curves
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["n_params"] = sum(p.numel() for p in model.parameters())
    res["train_counts"] = dict(cnt.most_common())
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        export_onnx(out)
        res["latency_cpu_ms"] = bench(out / "screen_cls.onnx")
        (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:                # noqa: BLE001
        print(f"[跳过 ONNX 导出] {e!r}（.venv 没装 onnx：用 .venv-dfine/Scripts/python.exe 跑 --export-only）")
    for k in ("val", "test"):
        if k in res:
            r = res[k]
            print(f"{k}: n={r['n']} acc={r['acc']:.4f} CI{r['acc_ci95_by_match']} 弱标签acc={r['weak_label_acc']:.4f} | "
                  f"门槛{thresh} 覆盖{r['coverage_at_thresh']:.3f} 准确{r['acc_at_thresh']:.4f} 错{r['wrong_at_thresh']}")
    print("参数量", res["n_params"], "延迟", res.get("latency_cpu_ms"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
