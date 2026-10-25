"""云端试训数据包（2026-10-15）：从 r9zl 抽 1600 训练 + 200 验证题，本机和 PAI DLC 各训一遍对比 loss / 速度。
输出 data/laya/datasets/cloudtest/points.jsonl、data/laya/labels/cloudtest/labels.jsonl、build/cloud_trial/（上传用的代码包）。"""
from __future__ import annotations
import json, random, shutil
from pathlib import Path
P = Path(__file__).resolve().parents[2]
KINDS = {"shop", "talent", "card_pick", "lord_boost", "carry", "swap", "auction", "use_card", "route"}
labs = [json.loads(l) for l in (P / "data/laya/labels/r9zl_all/labels_all.jsonl").open(encoding="utf-8")]
lab = {r["id"]: r for r in labs}
pts = {}
for l in (P / "data/laya/datasets/r9zl/points.jsonl").open(encoding="utf-8"):
    p = json.loads(l)
    if p["id"] in lab and p["kind"] in KINDS and p.get("split") in ("train", "val"):
        pts[p["id"]] = p
rng = random.Random(7)
tr = sorted(i for i, p in pts.items() if p["split"] == "train"); rng.shuffle(tr)
va = sorted(i for i, p in pts.items() if p["split"] == "val"); rng.shuffle(va)
keep = tr[:1600] + va[:200]
ds = P / "data/laya/datasets/cloudtest"; ds.mkdir(parents=True, exist_ok=True)
with (ds / "points.jsonl").open("w", encoding="utf-8") as fh:
    for i in keep:
        fh.write(json.dumps(pts[i], ensure_ascii=False) + "\n")
ld = P / "data/laya/labels/cloudtest"; ld.mkdir(parents=True, exist_ok=True)
with (ld / "labels.jsonl").open("w", encoding="utf-8") as fh:
    for i in keep:
        fh.write(json.dumps(lab[i], ensure_ascii=False) + "\n")
print("train", len(tr[:1600]), "val", len(va[:200]))
# 代码包：tools/laya、agent/decision、data/*.json、数据子集、laya 库源码
B = P / "build/cloud_trial"
if B.exists():
    shutil.rmtree(B)
for sub in ("tools/laya", "agent/decision"):
    shutil.copytree(P / sub, B / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.bak*"))
(B / "agent/__init__.py").write_text("", encoding="utf-8")
(B / "data").mkdir(parents=True, exist_ok=True)
for f in (P / "data").glob("*.json"):
    shutil.copy2(f, B / "data" / f.name)
shutil.copytree(ds, B / "data/laya/datasets/cloudtest")
shutil.copytree(ld, B / "data/laya/labels/cloudtest")
import laya
shutil.copytree(Path(laya.__file__).parent, B / "pylib/laya", ignore=shutil.ignore_patterns("__pycache__"))
print("bundle", B)
