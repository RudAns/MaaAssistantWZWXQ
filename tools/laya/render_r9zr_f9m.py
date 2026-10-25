"""线上 R37 看到的文本：r9zr 的测试题按 f9m 重渲染（对比 R37 vs R38 用；训练题不渲染）。"""
import json, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "tools" / "laya")); sys.path.insert(1, str(P))
import features as F, rerender as RR, build_r9s as BR  # noqa: E402
F.ACTIVE = "f9m"; RR.ALLOW_NEW_KEYS = True
out = P / "data/laya/datasets/r9zr_f9m"; out.mkdir(parents=True, exist_ok=True)
n = bad = 0
with (out / "points.jsonl").open("w", encoding="utf-8") as fh:
    for l in (P / "data/laya/datasets/r9zr/points.jsonl").open(encoding="utf-8"):
        p = json.loads(l)
        if p.get("split") == "train":
            continue
        BR.apply_cfg(p.get("render_cfg"))
        r = RR.rerender(dict(p))
        if r is None:
            bad += 1; continue
        q = r["question"]; q["criteria"] = {k: q["criteria"][k] for k in p["question"]["criteria"] if k in q["criteria"]}
        if set(q["criteria"]) != set(p["question"]["criteria"]):
            bad += 1; continue
        p["state"], p["question"] = r["state"], q
        fh.write(json.dumps(p, ensure_ascii=False) + "\n"); n += 1
print("rendered", n, "dropped", bad)
