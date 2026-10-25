"""界面分类：与旧 OCR 判据在同一测试集上比较（阶段层面），并写 classes.json / 部署模型。

  .venv/Scripts/python.exe -X utf8 tools/vision/eval_screen_cls.py --run data/vision/runs/screen_cls_v1 [--deploy]

比较口径：
  - 只用"弱标签确实是旧 OCR 判据输出"的测试帧：phase_*.png（阶段切换那一帧）、unknown_*.png、stuck_*.png；
    其余证据帧（board_inspect / equip_popup …）的弱标签来自文件名，不是 OCR 判的，不参加比较。
  - 真值 = 人工复核的细类，按 ACCEPT 映射成"旧阶段体系里算对的取值"；模型细类按 PHASE_MAP 映射到阶段（没有对应 = unknown）。
  - phase_ 帧是"阶段刚切换"的帧，本来就偏向过场/难例；unknown/stuck 帧是 OCR 已经认不出的帧 → 这个比较对 OCR 不利。
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "vision" / "screen_cls_v1"

# 细类 -> detect_phase 的取值（None = 旧体系里没有这个阶段，线上应当"等 / 退回旧判据"）
PHASE_MAP = {
    "prepare": "prepare", "prepare_shop": "prepare", "hero_panel": "prepare", "equip_popup": "prepare",
    "battle": "battle", "auction": "auction", "reward_popup": "auction", "talent": "talent", "card_pick": "card_pick",
    "lord_boost": "lord_boost", "hero_pick": "hero_pick", "loading": "loading", "settle": "settle",
    "settle_final": "settle_final", "post_match": None, "lobby": "lobby", "mode_lobby": "mode_lobby",
    "match_found": "match_found", "settings": "settings", "lord_panel": "lord_panel", "enlarged_detail": None,
    "notice": None, "solo_popup": None, "board_idle": None, "transition": None, "other": None,
}
# 旧 OCR 判据输出哪些阶段算对（没有对应阶段的细类，判 unknown 才算对 —— 认不出就不动，是安全行为）
ACCEPT = {c: {p} if p else {"unknown"} for c, p in PHASE_MAP.items()}
ACCEPT["reward_popup"] = {"auction", "unknown"}      # 拍卖中弹的奖励展示，按拍卖处理或不动都可以
OCR_SRCS = ("phase_", "unknown", "stuck")
# 宽松口径：棋盘空转（board_idle）判成 battle 也算对 —— 两者线上都是"什么都不做、等"，单帧也分不清
ACCEPT_LOOSE = dict(ACCEPT)
ACCEPT_LOOSE["board_idle"] = {"unknown", "battle"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.9, help="部署门槛（验证集上采纳准确率 ≥98% 的最低门槛，见文档）")
    a = ap.parse_args()
    run = Path(a.run)
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    S = {json.loads(l)["id"]: json.loads(l) for l in (DS / "samples.jsonl").read_text(encoding="utf-8").splitlines()}
    P = {json.loads(l)["id"]: json.loads(l) for l in (run / "preds.jsonl").read_text(encoding="utf-8").splitlines()}
    R = {json.loads(l)["id"]: json.loads(l) for l in (DS / "labels_review.jsonl").read_text(encoding="utf-8").splitlines()}
    thr = a.thresh
    rows = []
    for i, r in R.items():
        s = S[i]
        if r["split"] != "test" or r["label"] not in PHASE_MAP or not s["src"].startswith(OCR_SRCS):
            continue
        ocr = s["src"][6:] if s["src"].startswith("phase_") else "unknown"
        ocr = ocr.replace("_multi", "")
        pr, cf = P[i]["pred"], P[i]["conf"]
        mp = PHASE_MAP.get(pr) or "unknown"
        rows.append({"id": i, "truth": r["label"], "ocr": ocr, "model_phase": mp, "conf": cf, "match": s["match"],
                     "ocr_ok": ocr in ACCEPT[r["label"]], "model_ok": mp in ACCEPT[r["label"]],
                     "hybrid_ok": (mp if cf >= thr else ocr) in ACCEPT[r["label"]],
                     "ocr_ok_loose": ocr in ACCEPT_LOOSE[r["label"]], "model_ok_loose": mp in ACCEPT_LOOSE[r["label"]],
                     "hybrid_ok_loose": (mp if cf >= thr else ocr) in ACCEPT_LOOSE[r["label"]]})
    n = len(rows)
    res = {"n": n, "thresh": thr,
           "ocr_acc": sum(r["ocr_ok"] for r in rows) / n, "model_acc": sum(r["model_ok"] for r in rows) / n,
           "hybrid_acc": sum(r["hybrid_ok"] for r in rows) / n,
           "loose": {k: sum(r[k + "_ok_loose"] for r in rows) / n for k in ("ocr", "model", "hybrid")},
           "by_source": {}, "by_truth": {},
           "ocr_errors": Counter(f"{r['truth']}->{r['ocr']}" for r in rows if not r["ocr_ok"]).most_common(25),
           "model_errors": Counter(f"{r['truth']}->{r['model_phase']}" for r in rows if not r["model_ok"]).most_common(25)}
    for src in ("phase", "unknown"):
        rr = [r for r in rows if (r["ocr"] != "unknown") == (src == "phase")]
        if rr:
            res["by_source"][src] = {"n": len(rr), "ocr_acc": float(np.mean([r["ocr_ok"] for r in rr])),
                                     "model_acc": float(np.mean([r["model_ok"] for r in rr])),
                                     "hybrid_acc": float(np.mean([r["hybrid_ok"] for r in rr])),
                                     "loose": {k: float(np.mean([r[k + "_ok_loose"] for r in rr])) for k in ("ocr", "model", "hybrid")}}
    for c in sorted({r["truth"] for r in rows}):
        rr = [r for r in rows if r["truth"] == c]
        res["by_truth"][c] = {"n": len(rr), "ocr_acc": round(float(np.mean([r["ocr_ok"] for r in rr])), 3),
                              "model_acc": round(float(np.mean([r["model_ok"] for r in rr])), 3)}
    # 细类严格 / 合并 battle+board_idle（单帧分不清"正在打"和"棋盘空转"，人工标注本身在两者间不一致，见文档）
    merge = lambda c: "battle" if c == "board_idle" else c
    fine = {}
    for sp in ("val", "test"):
        ids = [i for i, r in R.items() if r["split"] == sp and r["label"] in PHASE_MAP]
        if sp == "val":
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from train_screen_cls import val_half
            ids = [i for i in ids if val_half(S[i]["match"]) == "sel"]
        y = [R[i]["label"] for i in ids]
        pr = [P[i]["pred"] for i in ids]
        cf = np.array([P[i]["conf"] for i in ids])
        ok = np.array([a_ == b_ for a_, b_ in zip(y, pr)])
        okm = np.array([merge(a_) == merge(b_) for a_, b_ in zip(y, pr)])
        keep = cf >= thr
        fine[sp] = {"n": len(ids), "acc": float(ok.mean()), "acc_merged_board": float(okm.mean()),
                    "coverage_at_thresh": float(keep.mean()), "acc_at_thresh": float(ok[keep].mean()),
                    "acc_merged_at_thresh": float(okm[keep].mean()),
                    "curve": [{"t": t, "coverage": float((cf >= t).mean()), "acc": float(ok[cf >= t].mean()),
                               "acc_merged": float(okm[cf >= t].mean())} for t in (0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95)],
                    "per_class": {c: {"n": sum(1 for v in y if v == c),
                                      "precision": round(sum(1 for v, q in zip(y, pr) if q == c and v == c) / max(1, sum(1 for q in pr if q == c)), 3),
                                      "recall": round(sum(1 for v, q in zip(y, pr) if q == c and v == c) / max(1, sum(1 for v in y if v == c)), 3)}
                                  for c in sorted(set(y))},
                    "confusions": Counter(f"{u}->{v}" for u, v in zip(y, pr) if u != v).most_common(20)}
    res["fine"] = fine
    (run / "vs_ocr.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_truth", "fine")}, ensure_ascii=False, indent=1))
    for sp, f in fine.items():
        print(sp, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items() if k not in ("per_class", "curve", "confusions")})
        print("  曲线", [(c["t"], round(c["coverage"], 3), round(c["acc"], 4), round(c["acc_merged"], 4)) for c in f["curve"]])
        print("  混淆", f["confusions"][:10])
    if a.deploy:
        meta = {"classes": m["classes"], "thresh": thr, "phase_map": PHASE_MAP, "input": {"w": 256, "h": 144,
                "format": "RGB /255, NCHW float32, INTER_AREA 缩放整屏 1280x720"}, "model": "assets/model/vision/screen_cls_v1.onnx",
                "created": m.get("created"), "report": "docs/screen-cls-model.md"}
        (DS / "classes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        dst = ROOT / "assets" / "model" / "vision" / "screen_cls_v1.onnx"
        shutil.copy2(run / "screen_cls.onnx", dst)
        print("部署 ->", dst, "；classes.json 已写")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
