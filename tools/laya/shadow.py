"""离线影子评估：拿一个 Laya checkpoint，把某些运行目录里的决策点**按实战同样的输入**重放一遍，
记录 Laya 的答案、置信度、延迟，与规则引擎当时的选择并排存盘。

用途：
  1) 新对局打完后，看训练过的 Laya 在"没见过的局面"上怎么选（S3 影子模式的离线版，输入与训练完全同源）；
  2) 导出 Laya 与规则分歧的题，交给老师标注（主动学习：分歧题信息量最大）；
  3) 测延迟（单题 / 批量）。

  python tools/laya/shadow.py --ckpt data/laya/runs/<run>/checkpoint --runs live_20260924_*  --out data/laya/shadow/<name>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))
import build_dataset as B  # noqa: E402
from train import BASE_CKPT, load_model, make_item  # noqa: E402

from laya.common import collate_items  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(BASE_CKPT))
    ap.add_argument("--runs", nargs="+", required=True, help="data/logs 下的运行目录名（可用通配）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_model(Path(args.ckpt), device)
    model.eval()
    T = float((cfg.get("temperature") or [1.0])[0])
    dirs = []
    for pat in args.runs:
        dirs += [Path(p) for p in glob.glob(str(PROJECT_DIR / "data" / "logs" / pat))]
    pts = []
    for d in sorted(set(dirs)):
        pts += B.extract_run(d)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, lat = [], []
    with torch.no_grad():
        # 预热（第一次 CUDA 调用有编译/分配开销，不计入延迟）
        if pts:
            it = make_item(tok, {**pts[0], "teacher": {}, "id": "warm"}, cfg)
            b = collate_items([[it]], tok.pad_token_id)
            model(*(b[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
        for p in pts:
            p = {**p, "teacher": {}, "id": p.get("id") or f"{p['run']}_{p['ts']}"}
            it = make_item(tok, p, cfg)
            b = collate_items([[it]], tok.pad_token_id)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = model(*(b[k].to(device) for k in
                                    ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
            z = logits[0, :len(it["markers"])].float().cpu().numpy() / T
            q = np.exp(z - z.max())
            q /= q.sum()
            keys = it["keys"]
            a = keys[int(q.argmax())]
            rows.append({"run": p["run"], "ts": p["ts"], "round": p["round"], "kind": p["kind"], "keys": keys,
                         "laya": [round(float(x), 4) for x in q], "laya_choice": a, "conf": round(float(q.max()), 4),
                         "rule_choice": p.get("rule_choice"), "agree_rule": a == p.get("rule_choice"),
                         "point": {k: p[k] for k in ("question", "state", "options", "raw", "ctx", "field", "kind")}})
    with (out / "shadow.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {"ckpt": args.ckpt, "n": len(rows), "runs": [d.name for d in dirs],
            "latency_ms": {"p50": float(np.percentile(lat, 50)) if lat else None,
                           "p95": float(np.percentile(lat, 95)) if lat else None},
            "agree_rule": float(np.mean([r["agree_rule"] for r in rows])) if rows else None,
            "by_kind": {}}
    for k in sorted({r["kind"] for r in rows}):
        rr = [r for r in rows if r["kind"] == k]
        summ["by_kind"][k] = {"n": len(rr), "agree_rule": float(np.mean([r["agree_rule"] for r in rr])),
                              "mean_conf": float(np.mean([r["conf"] for r in rr]))}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
