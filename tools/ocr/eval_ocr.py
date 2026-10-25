"""在 OCR 测试集上评估一组找字/认字模型。

调用方式与线上一致：对每张卡名小图调用 OnnxOcr.boxes(整张小图)，把读到的文字按从左到右拼起来。
模型目录用环境变量切换（WZWXQ_OCR_DET_DIR / WZWXQ_OCR_REC_DIR / WZWXQ_OCR_EP），例：
  WZWXQ_OCR_DET_DIR=assets/model/ocr_server WZWXQ_OCR_REC_DIR=assets/model/ocr_server \
    python tools/ocr/eval_ocr.py --ver v1 --tag server_cpu

指标（按类型分别报告）：
  exact     读数（去掉空白和·-等符号）与正确卡名完全一致
  cer       字符错误率 = 编辑距离 / 正确卡名长度（越低越好）
  resolved  读数经"卡牌库模糊匹配"后得到正确卡名（线上真正用到的结果）
  wrong     模糊匹配到了**另一张**卡（最危险：认错比认不出更糟）
结果追加写 data/ocr_bench/<ver>/results.jsonl，逐图读数写 preds_<tag>.jsonl。
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

_STRIP = re.compile(r"[\s·\-—_.,，。:：|'\"`~]+")


_SUFFIX = re.compile(r"[(（][^()（）]*[)）]$")      # 卡牌库里区分同名卡的后缀，如「护命灵珠(天赋牌)」，画面上没有


def display(name: str) -> str:
    return _SUFFIX.sub("", name or "")


def norm(s: str) -> str:
    return _STRIP.sub("", display(s))


def edit(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    import torch  # noqa: F401  —— 先 torch 再 onnxruntime GPU 库
    from agent.perception.onnx_ocr import DET_DIR, REC_DIR, OnnxOcr
    root = PROJECT_DIR / "data" / "ocr_bench" / args.ver
    rows = [json.loads(l) for l in (root / "labels.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == args.split]
    cards = json.loads((PROJECT_DIR / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    names = [n for n in cards]
    nmap = {norm(n): n for n in names}
    ocr = OnnxOcr()
    # 预热（TensorRT 第一次调用会慢）
    im0 = cv2.imdecode(np.fromfile(str(root / rows[0]["file"]), np.uint8), 1)
    for _ in range(3):
        ocr.boxes(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB), (0, 0, im0.shape[1], im0.shape[0]))
    agg = collections.defaultdict(lambda: collections.Counter())
    lat = []
    preds = []
    for r in rows:
        im = cv2.cvtColor(cv2.imdecode(np.fromfile(str(root / r["file"]), np.uint8), 1), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        boxes = ocr.boxes(im, (0, 0, im.shape[1], im.shape[0]))
        lat.append((time.perf_counter() - t0) * 1000)
        boxes = sorted(boxes, key=lambda b: (b[1][0] if b[1] else 0))
        read = norm("".join(b[0] for b in boxes))
        gt = norm(r["gt"])
        res = nmap.get(read)
        if res is None and read:
            m = difflib.get_close_matches(read, list(nmap), n=1, cutoff=0.6)
            res = nmap[m[0]] if m else None
        for g in ("all", r["kind"]):
            a = agg[g]
            a["n"] += 1
            a["exact"] += read == gt
            a["cer_num"] += edit(read, gt)
            a["cer_den"] += max(1, len(gt))
            a["resolved"] += res is not None and norm(res) == gt
            a["wrong"] += (res is not None and norm(res) != gt)
            a["empty"] += not read
        preds.append({"file": r["file"], "gt": r["gt"], "read": read, "resolved": res})
    out = {"tag": args.tag, "ver": args.ver, "split": args.split, "engine": ocr.describe(),
           "det_dir": str(DET_DIR), "rec_dir": str(REC_DIR),
           "latency_ms_p50": round(float(np.percentile(lat, 50)), 2), "latency_ms_p90": round(float(np.percentile(lat, 90)), 2),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "by_kind": {}}
    for g, a in agg.items():
        out["by_kind"][g] = {"n": a["n"], "exact": round(a["exact"] / a["n"], 4),
                             "cer": round(a["cer_num"] / a["cer_den"], 4),
                             "resolved": round(a["resolved"] / a["n"], 4), "wrong": round(a["wrong"] / a["n"], 4),
                             "empty": round(a["empty"] / a["n"], 4)}
    with (root / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    with (root / f"preds_{args.tag}.jsonl").open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
