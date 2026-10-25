"""手牌卡名条：几种读法在同一批名字条上对比（测试集，人工复核的标签见 data/vision/hand_names_v1/labels_test.jsonl）。

  WZWXQ_OCR_EP=cpu .venv/Scripts/python.exe -X utf8 tools/ocr/eval_hand_names.py [--split test] [--dump]

方法：
  a_ocr      线上现状：轻量认字模型自由解码 + `_resolve_hand_fragment`（pool.jsonl 里的 label）
  b_light    轻量认字模型 + 闭集 CTC 解码（tools/ocr/ctc_closed.py）
  c_server   服务器版认字模型 + 闭集 CTC 解码（OCR 微调主候选）
闭集判定：没截断（ROI 宽 ≥ FULL_W）只在完整卡名里选；截断时连前缀一起选（前缀被多张牌共用 = 歧义，输出前缀本身）。
第一名比第二名高不到 MARGIN（log 概率）→ 判"读不出"（宁缺勿错）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools" / "ocr")]
from ctc_closed import ClosedDecoder  # noqa: E402

D = ROOT / "data" / "vision" / "hand_names_v1"
FULL_W = 110
MARGIN = 2.0
REC_H = 48


def rec_probs(sess, crops_bgr):
    out = []
    for c in crops_bgr:
        w = max(8, int(round(c.shape[1] * REC_H / c.shape[0])))
        W = (max(64, w) + 7) // 8 * 8
        x = np.zeros((1, 3, REC_H, W), np.float32)
        im = cv2.resize(c, (w, REC_H)).astype(np.float32) / 255.0
        x[0, :, :, :w] = ((im - 0.5) / 0.5).transpose(2, 0, 1)
        out.append(sess.run(None, {sess.get_inputs()[0].name: x})[0][0])
    return out


def load_rec(model_dir: Path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_dir / "rec.onnx"), providers=["CPUExecutionProvider"])
    keys = (model_dir / "keys.txt").read_text(encoding="utf-8").split("\n")
    if keys and keys[-1] == "":
        keys = keys[:-1]
    return sess, ["<blank>"] + keys + [" "]


def decide(dec: ClosedDecoder, probs, width: int, unique_prefix: dict):
    trunc = width < FULL_W
    r = dec.rank(probs, prefix=trunc, top=3)
    if not r:
        return None, 0.0
    best, lp, full = r[0]
    margin = lp - (r[1][1] if len(r) > 1 else -1e9)
    if margin < MARGIN:
        return None, margin
    if not full:
        best = unique_prefix.get(best, "前缀:" + best)      # 前缀只对应一张牌就是它；否则是歧义
    return best, margin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dump", action="store_true", help="写出逐条结果 data/vision/hand_names_v1/eval_<split>.jsonl")
    args = ap.parse_args()
    cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    names = [n for n, c in cards.items() if c.get("category") in ("英雄牌", "效果牌", "装备牌", "天赋牌", "棋手技能牌", "召唤物")]
    pref: dict = {}
    for n in names:
        for k in range(1, len(n)):
            pref.setdefault(n[:k], set()).add(n)
    unique_prefix = {p: next(iter(s)) for p, s in pref.items() if len(s) == 1}
    rows = [json.loads(l) for l in (D / "pool.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == args.split]
    light = load_rec(ROOT / "assets" / "model" / "ocr")
    server = load_rec(ROOT / "assets" / "model" / "ocr_server")
    decs = {k: ClosedDecoder(v[1], names) for k, v in (("light", light), ("server", server))}
    out = []
    for i, r in enumerate(rows):
        im = cv2.imdecode(np.fromfile(str(D / "crops" / f"{r['id']}.png"), np.uint8), cv2.IMREAD_COLOR)
        rec = {"id": r["id"], "w": r["roi"][2], "a_ocr": r["label"], "ocr_raw": r["ocr"]}
        for k, (sess, _chars) in (("light", light), ("server", server)):
            p = rec_probs(sess, [im])[0]
            name, m = decide(decs[k], p, r["roi"][2], unique_prefix)
            rec[f"{'b' if k == 'light' else 'c'}_{k}"] = name
            rec[f"{k}_margin"] = round(float(m), 2)
        out.append(rec)
        if i % 200 == 0:
            print(i, "/", len(rows), flush=True)
    if args.dump:
        (D / f"eval_{args.split}.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n",
                                                   encoding="utf-8")
    lab_file = D / f"labels_{args.split}.jsonl"
    if lab_file.exists():
        truth = {json.loads(l)["id"]: json.loads(l)["truth"] for l in lab_file.read_text(encoding="utf-8").splitlines()}
        for m in ("a_ocr", "b_light", "c_server"):
            ok = bad = none = 0
            for o in out:
                t = truth.get(o["id"])
                if t is None:
                    continue
                v = o[m]
                if v is None or str(v).startswith("前缀:"):
                    none += 1
                elif v == t:
                    ok += 1
                else:
                    bad += 1
            n = ok + bad + none
            print(f"{m:9s} n={n} 对 {ok / n:.3f}  读不出 {none / n:.3f}  **读错 {bad / n:.3f}**（{bad}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
