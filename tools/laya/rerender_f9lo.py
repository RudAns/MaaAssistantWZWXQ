"""R28 数据集 r9zb_f9lo：R27 的训练数据 r9zb（f9lm）**原题面不动**，只在每个选项前加推演器结论（features.engine_prefix，
打分来自 data/engine/annot_r9zb.jsonl）。版本号记 "f9lm+sim"（线上按它在 f9lm 题面上加推演结论）。
不用 rerender：用当前代码重渲染 r9zb 会让 27% 的题选项键变（卡库 10-13 重爬等），标签就对不上了。
  .venv/Scripts/python.exe -X utf8 tools/laya/rerender_f9lo.py
"""
import json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "laya")); sys.path.insert(0, str(ROOT))
import features as F  # noqa: E402
F.ACTIVE = F.FEATURE_VERSION = "f9lm+sim"
print("推演器打分", F.load_engine_annot(ROOT / "data/engine/annot_r9zb.jsonl"))
src = ROOT / "data/laya/datasets/r9zb"
dst = ROOT / "data/laya/datasets/r9zb_f9lo"
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.name != "points.jsonl":
        shutil.copy(f, dst / f.name)
n = hit = 0
with open(dst / "points.jsonl", "w", encoding="utf-8") as fo:
    for line in open(src / "points.jsonl", encoding="utf-8"):
        p = json.loads(line)
        ann = F.ENGINE_ANNOT.get(p.get("id"))
        if ann:
            p["question"]["criteria"] = F.engine_prefix(p["question"]["criteria"], ann)
            hit += 1
        p["feature_version"] = "f9lm+sim"
        fo.write(json.dumps(p, ensure_ascii=False) + "\n")
        n += 1
print(f"{n} 道题，{hit} 道加了推演结论")
