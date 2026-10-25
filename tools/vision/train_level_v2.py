"""level_cls_v2：v1 的数据 + build_level_v2 补的三位数（≥100）样本 + 可选合成三位数，同一网络结构重训（2026-10-12）。

  .venv/Scripts/python.exe -X utf8 tools/vision/train_level_v2.py [--device cuda] [--epochs 40] [--synth 0]
  .venv-dfine/Scripts/python.exe -X utf8 tools/vision/train_level_v2.py --export-only     # 导出 ONNX

输入 / 输出头与 v1 完全相同（48x48 RGB /255；位数 1~3 + 3 个位置的数字），线上替换只需换文件 + 放开"三位数不收"。
测试：
  A. v1 原测试集（data/vision/level_v1/labels_test.jsonl，人工复核 356 个）—— 不退步；
  B. 三位数测试集（data/vision_v2/level_v2 里 split=test、level≥100；若有 labels_hi_test_reviewed.jsonl 以复核为准）；
  两个模型（assets 里的 v1 ONNX、v2）在同一批裁图上评。
输出 data/vision_v2/runs/level_v2/{best.pt, metrics.json, level_cls_v2.onnx}。
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
import torch.nn.functional as F

cv2.setNumThreads(4)
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "vision"))
import train_level as L  # noqa: E402

V2DS = ROOT / "data" / "vision_v2" / "level_v2"
OUT = ROOT / "data" / "vision_v2" / "runs" / "level_v2"
V1_ONNX = ROOT / "assets" / "model" / "vision" / "level_cls_v1.onnx"


def load_v2():
    z = np.load(V2DS / "crops.npz")["badge"]
    rows = [json.loads(l) for l in (V2DS / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    rev = V2DS / "labels_hi_test_reviewed.jsonl"
    fix = {}
    if rev.exists():
        fix = {r["id"]: r["level"] for r in map(json.loads, rev.read_text(encoding="utf-8").splitlines())}
    out = {"train": [], "val": [], "test_hi": [], "test_low": []}
    for r in rows:
        im = z[r["id"]]
        lv = fix.get(r["id"], r["level"])
        if not lv:
            continue                                            # 复核判"看不清/标签错且定不了"
        item = (im, int(lv), r["match"], f"v2_{r['id']}")
        if r["split"] in ("train", "val"):
            # 自动标签里 100 基本都是 9-30 前 v1 模型把所有三位数报成 100 的残留（复核 23 个只对 1 个），≥400 多是乱读：都不进训练/验证
            if lv == 100 or lv >= 400:
                continue
            out[r["split"]].append(item)
        elif r["split"] == "test":
            if lv >= 100 and (not fix or r["id"] in fix):
                out["test_hi"].append(item)
            elif lv < 100:
                out["test_low"].append(item)
    return out


def synth_hi(base: list, n: int, rng: random.Random) -> list:
    """合成三位数：从真实三位数徽章（训练集）里切出数字列，拼成新的三位数。只在真实样本太少时用。"""
    hi = [d for d in base if d[1] >= 100]
    if not hi or n <= 0:
        return []
    out = []
    for _ in range(n):
        a, b = rng.choice(hi), rng.choice(hi)
        im = a[0].copy()
        # 徽章中心 ±~22px 是三位数；把 a 的右半（第三位）换成 b 的右半，标签相应改
        sa, sb = str(a[1]), str(b[1])
        if len(sa) != 3 or len(sb) != 3:
            continue
        im[:, 38:56] = b[0][:, 38:56]
        out.append((im, int(sa[:2] + sb[2]), "synth", f"synth_{len(out)}"))
    return out


def onnx_predict(path, data):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.log_severity_level = 3
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    vals, confs = [], []
    for i in range(0, len(data), 256):
        x = np.stack([cv2.cvtColor(L.crop(d[0]), cv2.COLOR_BGR2RGB) for d in data[i:i + 256]]).astype(np.float32) / 255.0
        nd, dg = s.run(None, {"x": np.ascontiguousarray(x.transpose(0, 3, 1, 2))})
        v, c = L.decode(torch.from_numpy(nd), torch.from_numpy(dg))
        vals += v; confs.append(c)
    return vals, np.concatenate(confs) if confs else np.zeros(0)


def summarize(data, vals, confs, thresh, online_v1=False):
    y = [d[1] for d in data]
    ok = [a == b for a, b in zip(y, vals)]
    acc_mask = [c >= thresh and (not online_v1 or v < 100) for v, c in zip(vals, confs)]
    keep = [i for i in range(len(y)) if acc_mask[i]]
    return {"n": len(y), "acc": round(float(np.mean(ok)), 4) if ok else None,
            "ci95_by_match": L.bootstrap(ok, [d[2] for d in data]) if ok else None,
            "thresh": thresh, "coverage": round(len(keep) / max(len(y), 1), 4),
            "acc_accepted": round(float(np.mean([ok[i] for i in keep])), 4) if keep else None,
            "wrong_accepted": sum(1 for i in keep if not ok[i]),
            "errors": [{"id": data[i][3], "truth": y[i], "pred": vals[i], "conf": round(float(confs[i]), 3)}
                       for i in range(len(y)) if not ok[i]][:30]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--synth", type=int, default=0)
    ap.add_argument("--hi-weight", type=float, default=3.0, help="三位数样本的采样权重倍数")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--clean-with", default="", help="用这个 best.pt 清洗训练/验证里的三位数自动标签（预测≠标签且置信≥0.95 的丢掉）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    globals()["OUT"] = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        m = L.Net(); m.load_state_dict(torch.load(OUT / "best.pt", map_location="cpu")); m.eval()
        torch.onnx.export(m, torch.zeros(1, 3, L.SIZE, L.SIZE), str(OUT / "level_cls_v2.onnx"), input_names=["x"],
                          output_names=["ndigits", "digits"],
                          dynamic_axes={"x": {0: "n"}, "ndigits": {0: "n"}, "digits": {0: "n"}}, opset_version=17, dynamo=False)
        print("导出", OUT / "level_cls_v2.onnx"); return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tr1, va1, te1 = L.load_split("train"), L.load_split("val"), L.load_split("test")
    v2 = load_v2()
    if args.clean_with:
        cm = L.Net().to(args.device); cm.load_state_dict(torch.load(args.clean_with, map_location=args.device))
        for sp in ("train", "val"):
            hi = [d for d in v2[sp] if d[1] >= 100]
            pv, pc = L.predict(cm, hi, args.device)
            bad = {d[3] for d, p, c in zip(hi, pv, pc) if p != d[1] and c >= 0.95}
            print(f"清洗 {sp}: 三位数 {len(hi)} 里丢掉 {len(bad)}", flush=True)
            v2[sp] = [d for d in v2[sp] if d[3] not in bad]
    tr = tr1 + v2["train"]
    tr += synth_hi(v2["train"], args.synth, rng)
    va = va1 + v2["val"]
    print(f"train {len(tr)} (v1 {len(tr1)}, 新 {len(v2['train'])}, 其中≥100 {sum(d[1] >= 100 for d in tr)})  "
          f"val {len(va)} (≥100 {sum(d[1] >= 100 for d in va)})  test_v1 {len(te1)}  test_hi {len(v2['test_hi'])}  device {args.device}", flush=True)
    cnt = {}
    for d in tr:
        cnt[L.bucket(d[1])] = cnt.get(L.bucket(d[1]), 0) + 1
    w = torch.tensor([(len(tr) / cnt[L.bucket(d[1])]) ** 0.5 * (args.hi_weight if d[1] >= 100 else 1.0) for d in tr])
    dev = args.device
    model = L.Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_ep, curves = -1.0, -1, []
    va_hi = [d for d in va if d[1] >= 100]
    for ep in range(args.epochs):
        model.train()
        idx = torch.multinomial(w, len(tr), replacement=True).tolist()
        tl = 0.0
        for i in range(0, len(idx), 128):
            b = [tr[j] for j in idx[i:i + 128]]
            x = L.to_tensor([L.augment(d[0], rng) for d in b]).to(dev)
            tn, td = zip(*[L.targets(d[1]) for d in b])
            nd, dg = model(x)
            loss = F.cross_entropy(nd, torch.tensor(tn).to(dev)) + \
                F.cross_entropy(dg.reshape(-1, 10), torch.tensor(td).reshape(-1).to(dev), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        vv, _ = L.predict(model, va, dev)
        acc = float(np.mean([a == d[1] for a, d in zip(vv, va)]))
        hv = [a == d[1] for a, d in zip(vv, va) if d[1] >= 100]
        acc_hi = float(np.mean(hv)) if hv else 0.0
        score = acc + 0.5 * acc_hi * (len(va_hi) > 0)          # 选模型：整体 + 三位数
        curves.append({"epoch": ep, "train_loss": round(tl / len(tr), 4), "val_acc": round(acc, 4), "val_hi_acc": round(acc_hi, 4)})
        print(f"ep {ep:2d} loss {tl / len(tr):.4f} val {acc:.4f} hi {acc_hi:.4f}", flush=True)
        if score > best:
            best, best_ep = score, ep
            torch.save(model.state_dict(), OUT / "best.pt")
    model.load_state_dict(torch.load(OUT / "best.pt"))
    vv, vc = L.predict(model, va, dev)
    yv, pv = np.array([d[1] for d in va]), np.array(vv)
    thresh = 0.5
    for t in np.arange(0.5, 1.0, 0.01):
        m = vc >= t
        if m.sum() and (pv[m] == yv[m]).mean() >= 0.995:
            thresh = float(round(t, 2)); break
    res = {"best_epoch": best_ep, "thresh_val": thresh, "curves": curves, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "train_n": len(tr), "train_hi": sum(d[1] >= 100 for d in tr), "synth": args.synth}
    sets = {"test_v1": te1, "test_hi": v2["test_hi"], "test_low_new": v2["test_low"], "val_hi": va_hi}
    for name, data in sets.items():
        if not data:
            continue
        v2v, v2c = L.predict(model, data, dev)
        v1v, v1c = onnx_predict(V1_ONNX, data)
        res[name] = {"v2@0.9": summarize(data, v2v, v2c, 0.9),
                     "v1@0.9_raw": summarize(data, v1v, v1c, 0.9),
                     "v1@0.9_online(≥100不收)": summarize(data, v1v, v1c, 0.9, online_v1=True)}
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in sets:
        if name in res:
            r = res[name]
            print(name, {k: (v["n"], v["acc"], v["coverage"], v["acc_accepted"], v["wrong_accepted"]) for k, v in r.items()})


if __name__ == "__main__":
    main()
