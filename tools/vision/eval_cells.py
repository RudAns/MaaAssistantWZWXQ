"""按"格子占用"评测 D-FINE（和旧徽章检测器在同一批留出帧上比）。

  .venv-dfine/Scripts/python.exe tools/vision/eval_cells.py --ckpt data/vision/runs/board_m_v1/best_stg2.pth --split test

指标：整帧完全正确率；格子召回/精度；**C1 召回单独报**（旧检测器的固定盲区）；英雄/召唤物分别的召回。
预测框 → 框底部中心（格心在框底上方 15px）→ 最近的格子（距离 < 60px 才算）。同一格多个框只算一次。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "D-FINE"))
from tools.vision.build_coco import BOX_DOWN, cell_xy  # noqa: E402


def load_model(cfg_path: str, ckpt: str, device: str):
    from src.core import YAMLConfig
    cfg = YAMLConfig(cfg_path, resume=ckpt)
    state = torch.load(ckpt, map_location="cpu")
    state = state["ema"]["module"] if "ema" in state and state.get("ema") else state["model"]
    cfg.model.load_state_dict(state)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()

        def forward(self, x, size):
            return self.post(self.model(x), size)
    return M().to(device).eval()


def predict_cells(model, img_path: Path, device: str, thr: float, cells: dict) -> dict[str, int]:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    x = torch.from_numpy(np.asarray(im.resize((640, 640)), dtype=np.float32) / 255.).permute(2, 0, 1)[None].to(device)
    size = torch.tensor([[w, h]], device=device)
    with torch.no_grad():
        labels, boxes, scores = model(x, size)
    out = {}
    for lab, b, s in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
        if s < thr:
            continue
        fx, fy = (b[0] + b[2]) / 2, b[3] - BOX_DOWN
        best = min(cells.items(), key=lambda kv: (kv[1][0] - fx) ** 2 + (kv[1][1] - fy) ** 2)
        if ((best[1][0] - fx) ** 2 + (best[1][1] - fy) ** 2) ** .5 < 60 and s > out.get(best[0], (0, 0))[1]:
            out[best[0]] = (lab, s)
    return {c: v[0] for c, v in out.items()}


def score(pairs):
    """pairs: [(gt_set, pred_set)] -> 指标"""
    tp = fp = fn = exact = 0
    c1 = [0, 0]
    for gt, pr in pairs:
        tp += len(gt & pr); fp += len(pr - gt); fn += len(gt - pr); exact += gt == pr
        if "C1" in gt:
            c1[0] += 1; c1[1] += "C1" in pr
    return {"frames": len(pairs), "exact": round(exact / len(pairs), 3), "recall": round(tp / max(1, tp + fn), 3),
            "precision": round(tp / max(1, tp + fp), 3), "C1_recall": round(c1[1] / max(1, c1[0]), 3), "C1_n": c1[0]}


def bootstrap(pairs, groups=None, n=2000, seed=0):
    """95% 置信区间（整帧正确率、格子召回、格子精度）。
    groups=None 按帧重采样；给了 groups（每帧所属对局）就**按对局整组重采样** ——
    同一局的帧高度相关（测试集的 4 个错误全来自同一局同一场面），按帧抽会把区间算窄。"""
    import random
    rng = random.Random(seed)
    keys = ("exact", "recall", "precision")
    samples = {k: [] for k in keys}
    units = [[p] for p in pairs] if groups is None else         [[p for p, g in zip(pairs, groups) if g == gg] for gg in sorted(set(groups))]
    for _ in range(n):
        s = score([p for _ in units for p in units[rng.randrange(len(units))]])
        for k in keys:
            samples[k].append(s[k])
    return {k: [round(sorted(v)[int(.025 * n)], 3), round(sorted(v)[int(.975 * n) - 1], 3)] for k, v in samples.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default=str(ROOT / "third_party/D-FINE/configs/dfine/custom/wxq_board_m.yml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--frames", default="frames.jsonl", help="帧清单；v2 = frames_v2split.jsonl")
    ap.add_argument("--coco", default="coco_v1")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = ROOT / "data/vision/pool_v1"
    frames = {json.loads(l)["id"]: json.loads(l) for l in (pool / args.frames).open(encoding="utf-8")}
    labels = {}
    for fn in ("labels_train.jsonl", "labels_evalset.jsonl"):
        for l in (pool / fn).open(encoding="utf-8"):
            r = json.loads(l); labels[r["id"]] = r
    cells = cell_xy()
    model = load_model(args.cfg, args.ckpt, device)
    old, new, runs, per_class = [], [], [], collections.Counter()
    for fid, f in sorted(frames.items()):
        if f["split"] != args.split or fid not in labels:
            continue
        gt = set(labels[fid]["occupied"])
        summ = set(labels[fid].get("summons") or [])
        pred = predict_cells(model, ROOT / "data/vision" / args.coco / args.split / "images" / f"{fid}.png", device, args.thr, cells)
        new.append((gt, set(pred)))
        runs.append(f.get("match", f["run"]))
        old.append((gt, set(f["prelabel"])))
        for c in gt:
            k = "summon" if c in summ else "hero"
            per_class[k + "_n"] += 1
            per_class[k + "_hit"] += c in pred
    res = {"split": args.split, "thr": args.thr, "dfine": score(new), "old_badge": score(old),
           "dfine_ci95": bootstrap(new), "old_badge_ci95": bootstrap(old),
           "n_runs": len(set(runs)),
           "dfine_ci95_by_run": bootstrap(new, runs), "old_badge_ci95_by_run": bootstrap(old, runs),
           "dfine_recall_by_class": {k: round(per_class[k + "_hit"] / max(1, per_class[k + "_n"]), 3) for k in ("hero", "summon")}}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    out = ROOT / "data/vision/runs/eval_cells.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(res, ckpt=args.ckpt), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
