"""用当前代码按数据集自己的版本重渲染指定题，列出文字变了的题号（2026-10-15：题面在标注后修正，找出要重标的题）。
  .venv/Scripts/python.exe -X utf8 tools/laya/diff_rerender_ids.py --ds r9zl --ver f9lu --ids <ids.json>... --out <changed.json>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "tools" / "laya")); sys.path.insert(1, str(P))
import features as F, rerender as RR, build_r9s as BR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True); ap.add_argument("--ver", required=True)
ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
F.ACTIVE = a.ver; RR.ALLOW_NEW_KEYS = True
want = set()
for f in a.ids:
    want |= set(json.loads(Path(f).read_text(encoding="utf-8")))
changed, same, err = [], 0, 0
for l in (P / "data/laya/datasets" / a.ds / "points.jsonl").open(encoding="utf-8"):
    p = json.loads(l)
    if p["id"] not in want:
        continue
    BR.apply_cfg(p.get("render_cfg"))
    try:
        r = RR.rerender(dict(p))
    except Exception:
        err += 1; continue
    if r is None:
        err += 1; continue
    qa, qb = p["question"], r["question"]
    ca, cb = qa.get("criteria") or {}, qb.get("criteria") or {}
    # 只比数据集里收了的选项（ALLOW_NEW_KEYS 会补出数据集没收的选项，那不是文字变化）
    if r["state"] != p["state"] or any(cb.get(k) != v for k, v in ca.items()) or             {k: v for k, v in qa.items() if k != "criteria"} != {k: v for k, v in qb.items() if k != "criteria"}:
        changed.append(p["id"])
    else:
        same += 1
Path(a.out).write_text(json.dumps(changed), encoding="utf-8")
print(f"变了 {len(changed)}，没变 {same}，渲染失败 {err}")
