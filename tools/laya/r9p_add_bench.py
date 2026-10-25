"""r9p 新训练题里线上被删掉 bench 的 swap 题：按 f9l 重渲染、选项限回线上键 + bench，写回 points.jsonl，
输出需重标的 id 清单 data/laya/datasets/r9o/bench_ids.json（2026-10-01；与 rerender_r9n_f9l 的"train 永远保留 bench"同口径）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_r9n as BR  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402


def main() -> int:
    F.ACTIVE = "f9l"
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data" / "laya" / "datasets" / "r9p"
    new = set(json.loads((ds / "new_ids.json").read_text(encoding="utf-8")))
    pts = [json.loads(l) for l in (ds / "points.jsonl").open(encoding="utf-8")]
    fixed, failed = [], []
    for i, p in enumerate(pts):
        if not (p["id"] in new and p["kind"] == "swap" and p["split"] == "train"
                and "bench" not in p["question"]["criteria"]):
            continue
        BR.apply_cfg(p.get("render_cfg"))
        r = RR.rerender(dict(p))
        if r is None or "bench" not in r["question"]["criteria"]:
            failed.append(p["id"])
            continue
        keys = [k for k in r["question"]["criteria"] if k in set(p["options"]) | {"bench"}]
        if not BR.restrict_to(r, keys):
            failed.append(p["id"])
            continue
        r["feature_version"] = "f9l"
        r["bench_added"] = True
        pts[i] = r
        fixed.append(p["id"])
    BR.apply_cfg(None)
    with (ds / "points.jsonl").open("w", encoding="utf-8") as fh:
        for p in pts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (ds / "bench_ids.json").write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
    print(f"补 bench：{len(fixed)}，失败 {len(failed)} {failed[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
