"""unit_id_v2：全部对局重建的数据（build_unit_id_v2）+ 同一网络结构重训，和线上 v1 在同一测试集上对比（2026-10-12）。

  .venv/Scripts/python.exe -X utf8 tools/vision/train_unit_id_v2.py [--device cuda]
  .venv-dfine/Scripts/python.exe -X utf8 tools/vision/train_unit_id_v2.py --export-only

训练：split=train 的 strong + weak；验证：split=val 的 strong（定门槛、选 epoch）；测试：split=test 的 strong（只用"点开读到的名字"）。
类别：训练集样本数 ≥ MIN_TRAIN 的英雄 + 「其他」。
对比口径（线上用法）：置信 ≥ 门槛（v1 线上 0.5）且不是「其他」才采纳；
  覆盖率 = 采纳数 / 样本数，采纳准确率 = 采纳里名字对的比例；按类各报一遍。v1 不认识的英雄，它只要"采纳"就算错。
输出 data/vision_v2/runs/unit_id_v2/{best.pt, metrics.json, classes.json, unit_id_v2.onnx}。
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
import torch.nn.functional as F

cv2.setNumThreads(4)
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools" / "vision"))  # noqa: E702
import train_unit_id as U  # noqa: E402

DS = ROOT / "data" / "vision_v2" / "unit_id_v2"
OUT = ROOT / "data" / "vision_v2" / "runs" / "unit_id_v2"
V1_ONNX = ROOT / "assets" / "model" / "vision" / "unit_id_v1.onnx"
V1_META = ROOT / "data" / "vision" / "unit_id_v1" / "classes.json"
MIN_TRAIN = 30
OTHER = "其他"


def load():
    z = np.load(DS / "crops.npz")["unit"]
    rows = [json.loads(l) for l in (DS / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val" and r["src"] == "strong"]
    te = [r for r in rows if r["split"] == "test" and r["src"] == "strong"]
    cnt = Counter(r["name"] for r in tr)
    classes = sorted(n for n, c in cnt.items() if c >= MIN_TRAIN) + [OTHER]
    cid = {n: i for i, n in enumerate(classes)}

    def pack(rs):
        return [(z[r["i"]], cid.get(r["name"], cid[OTHER]), r) for r in rs]
    return classes, cnt, pack(tr), pack(va), pack(te)


def onnx_probs(path, data):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.log_severity_level = 3
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    ps = []
    for i in range(0, len(data), 256):
        x = np.stack([cv2.cvtColor(U.center(d[0]), cv2.COLOR_BGR2RGB) for d in data[i:i + 256]]).astype(np.float32) / 255.0
        lg = s.run(None, {"x": np.ascontiguousarray(x.transpose(0, 3, 1, 2))})[0]
        e = np.exp(lg - lg.max(1, keepdims=True)); ps.append(e / e.sum(1, keepdims=True))
    return np.concatenate(ps)


def compare(data, classes, P, thresh):
    """按名字比（不同模型类表不同）。"""
    names = [d[2]["name"] for d in data]
    pred = [classes[i] for i in P.argmax(1)]
    conf = P.max(1)
    acc = [(c >= thresh and p != OTHER) for p, c in zip(pred, conf)]
    ok = [p == n for p, n in zip(pred, names)]
    r = {"n": len(names), "thresh": thresh,
         "coverage": round(float(np.mean(acc)), 4),
         "acc_accepted": round(float(np.mean([o for o, a in zip(ok, acc) if a])), 4) if any(acc) else None,
         "wrong_accepted": int(sum(a and not o for a, o in zip(acc, ok))),
         "correct_accepted_rate": round(float(np.mean([a and o for a, o in zip(acc, ok)])), 4),
         "cov_ci95_by_match": U.bootstrap([a and o for a, o in zip(acc, ok)], [d[2]["match"] for d in data]),
         "by_class": {}}
    by = defaultdict(list)
    for n, a, o in zip(names, acc, ok):
        by[n].append((a, o))
    for n, lst in sorted(by.items(), key=lambda x: -len(x[1])):
        na = sum(a for a, _ in lst)
        r["by_class"][n] = {"n": len(lst), "coverage": round(na / len(lst), 3),
                            "acc_accepted": round(sum(a and o for a, o in lst) / na, 3) if na else None}
    r["confusions"] = [{"truth": a, "pred": b, "n": c} for (a, b), c in
                       Counter((n, p) for n, p, a, o in zip(names, pred, acc, ok) if a and not o).most_common(15)]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="不训练，读 best.pt 评估")
    ap.add_argument("--init", default="", help="从这个权重继续训练（中断后接着训）")
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        meta = json.loads((OUT / "classes.json").read_text(encoding="utf-8"))
        m = U.Net(len(meta["classes"])); m.load_state_dict(torch.load(OUT / "best.pt", map_location="cpu")); m.eval()
        torch.onnx.export(m, torch.zeros(1, 3, U.H, U.W), str(OUT / "unit_id_v2.onnx"), input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
        print("导出", OUT / "unit_id_v2.onnx"); return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    classes, cnt_name, tr, va, te = load()
    cnt = Counter(d[1] for d in tr)
    print(f"类别 {len(classes)}  train {len(tr)}  val {len(va)}  test {len(te)}  device {args.device}", flush=True)
    w = torch.tensor([(len(tr) / cnt[d[1]]) ** 0.5 * (1.0 if d[2]["src"] == "strong" else 0.7) for d in tr])
    dev = args.device
    model = U.Net(len(classes)).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n_ep = min(len(tr), 40000)
    best, best_ep, curves = -1.0, -1, []
    for ep in (range(0) if args.eval_only else range(args.epochs)):
        model.train()
        idx = torch.multinomial(w, n_ep, replacement=True).tolist()
        tl = 0.0
        for i in range(0, len(idx), 128):
            b = [tr[j] for j in idx[i:i + 128]]
            x = U.to_tensor([U.augment(d[0], rng) for d in b]).to(dev)
            loss = F.cross_entropy(model(x), torch.tensor([d[1] for d in b]).to(dev), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        P = U.predict(model, va, dev)
        acc = float((P.argmax(1) == np.array([d[1] for d in va])).mean())
        curves.append({"epoch": ep, "loss": round(tl / n_ep, 4), "val_top1": round(acc, 4)})
        print(curves[-1], flush=True)
        if acc > best:
            best, best_ep = acc, ep
            torch.save(model.state_dict(), OUT / "best.pt")
    model.load_state_dict(torch.load(OUT / "best.pt", map_location=dev))
    Pv = U.predict(model, va, dev)
    thresh = U.pick_thresh(classes, va, Pv)
    (OUT / "classes.json").write_text(json.dumps({"classes": classes, "thresh": max(thresh, 0.5), "other": OTHER,
                                                  "input": "80x128 中心裁 72x120, RGB /255"}, ensure_ascii=False, indent=1), encoding="utf-8")
    v1meta = json.loads(V1_META.read_text(encoding="utf-8"))
    res = {"classes": classes, "train_counts": dict(sorted(cnt_name.items(), key=lambda x: -x[1])), "best_epoch": best_ep,
           "thresh_val": thresh, "curves": curves, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "v2_classes_new": sorted(set(classes) - set(v1meta["classes"]))}
    Pt = U.predict(model, te, dev)
    P1 = onnx_probs(V1_ONNX, te)
    for t in (0.5, 0.9):
        res[f"test_v2@{t}"] = compare(te, classes, Pt, t)
        res[f"test_v1@{t}"] = compare(te, v1meta["classes"], P1, t)
    # 只看 v1 也认识的英雄（"老类不退步"）
    old = [d for d in te if d[2]["name"] in v1meta["classes"]]
    if old:
        idx = [i for i, d in enumerate(te) if d[2]["name"] in v1meta["classes"]]
        res["test_oldclasses_v2@0.5"] = {k: v for k, v in compare(old, classes, Pt[idx], 0.5).items() if k != "by_class"}
        res["test_oldclasses_v1@0.5"] = {k: v for k, v in compare(old, v1meta["classes"], P1[idx], 0.5).items() if k != "by_class"}
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in [k for k in res if k.startswith("test_")]:
        r = res[k]
        print(k, {x: r[x] for x in ("n", "coverage", "acc_accepted", "wrong_accepted", "correct_accepted_rate")})


if __name__ == "__main__":
    main()
