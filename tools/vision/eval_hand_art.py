"""手牌立绘模型 v1 / v2 同集对比（只用真实样本，标签 = 不叠放名字条的读数）。

  .venv/Scripts/python.exe -X utf8 tools/vision/eval_hand_art.py [--v2 data/vision/runs/hand_art_v2]

测试集：
  v1test  = v1 原测试集（hand_names_v1 + hand_awaken_v1 的原裁图/原标签，v1 训练时报的就是它）
  old     = v2 池里 v1 测试局（同样的对局，用现在的布局/标签重切）
  new     = v2 池 test_new（最近 20 局整局）
指标：n、词表内占比、门槛下被采纳率、被采纳准确率、错几张、"认对率"(被采纳且对 / 全部)；另按卡名列 芈月/海月/镜/瑶/曜。
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
FOCUS = ["芈月", "海月", "镜", "瑶", "曜"]


def model(onnx, meta):
    m = json.loads(Path(meta).read_text(encoding="utf-8"))
    return ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"]), m["classes"], float(m["thresh"])


def predict(sess, classes, ims):
    out = []
    for i in range(0, len(ims), 256):
        x = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[4:116, 4:60] for im in ims[i:i + 256]]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        lg = sess.run(None, {"x": np.ascontiguousarray(x)})[0]
        e = np.exp(lg - lg.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        out += [(classes[k], float(p[j, k])) for j, k in enumerate(p.argmax(1))]
    return out


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def sets():
    N = [json.loads(l) for l in (ROOT / "data/vision/hand_names_v1/pool.jsonl").read_text(encoding="utf-8").splitlines()]
    AL = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/vision/hand_awaken_v1/labels.jsonl").read_text(encoding="utf-8").splitlines()}
    c1 = ROOT / "data/vision/hand_awaken_v1/crops"
    v1t = [(imread(c1 / f"{r['id']}.png"), r["label"]) for r in N if r["split"] == "test" and r["roi"][2] >= 110 and r["label"]
           and AL.get(r["id"], {}).get("label") == 0 and (c1 / f"{r['id']}.png").exists()]
    # 直接读 parts/（构建可能还在续跑；合并后的 pool.jsonl 与 parts 内容相同）
    P = [json.loads(l) for f in sorted((ROOT / "data/vision/hand_names_v2/parts").glob("*.jsonl"))
            for l in f.read_text(encoding="utf-8").splitlines() if l]
    c2 = ROOT / "data/vision/hand_awaken_v2/crops"
    ok = lambda r: r["roi"][2] >= 110 and r["label"] and r["p_awk"] < 0.5 and r.get("awk_v1") != 1   # noqa: E731
    old = [(imread(c2 / f"{r['id']}.png"), r["label"]) for r in P if r["split"] == "test" and ok(r)]
    new = [(imread(c2 / f"{r['id']}.png"), r["label"]) for r in P if r["split"] == "test_new" and ok(r)]
    return {"v1test": v1t, "old": old, "new": new}


def metrics(pred, labels, classes, thr):
    voc = set(classes)
    n = len(labels)
    acc = [(p == y) for (p, c), y in zip(pred, labels) if c >= thr]
    inv = [y in voc for y in labels]
    return {"n": n, "in_vocab": round(sum(inv) / max(1, n), 3), "accepted": round(len(acc) / max(1, n), 3),
            "acc_at_thr": round(sum(acc) / max(1, len(acc)), 4), "wrong": len(acc) - sum(acc),
            "correct_rate": round(sum(acc) / max(1, n), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=str(ROOT / "data/vision/runs/hand_art_v2"))
    args = ap.parse_args()
    v2 = Path(args.v2)
    M = {"v1": model(ROOT / "assets/model/vision/hand_art_v1.onnx", ROOT / "assets/model/vision/hand_art_v1.json"),
         "v2": model(v2 / f"{v2.name}.onnx", v2 / f"{v2.name}.json")}
    S = sets()
    rep = {}
    for mk, (sess, cls, thr) in M.items():
        print(f"== {mk}: 词表 {len(cls)}，门槛 {thr}")
        for sk, data in S.items():
            pred = predict(sess, cls, [d[0] for d in data])
            labels = [d[1] for d in data]
            m = metrics(pred, labels, cls, thr)
            # 自定门槛对照：被采纳准确率 ≥98% 的最低门槛（只作参考，线上用 json 里的门槛）
            foc = {}
            for f in FOCUS:
                ix = [i for i, y in enumerate(labels) if y == f]
                if ix:
                    ok = sum(1 for i in ix if pred[i][1] >= thr and pred[i][0] == f)
                    bad = sum(1 for i in ix if pred[i][1] >= thr and pred[i][0] != f)
                    foc[f] = f"{ok}/{len(ix)} 对, 错{bad}"
                fp = sum(1 for (p, c), y in zip(pred, labels) if c >= thr and p == f and y != f)
                if fp:
                    foc[f + "(误收)"] = fp
            m["focus"] = foc
            curve = {}
            for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
                acc = [(p == y) for (p, c), y in zip(pred, labels) if c >= t]
                curve[t] = f"采纳{len(acc) / max(1, len(labels)):.3f}/准{sum(acc) / max(1, len(acc)):.4f}"
            m["curve"] = curve
            rep[f"{mk}/{sk}"] = m
            print(f"  {sk:7s} {m}")
    (v2 / "eval_v1_vs_v2.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
