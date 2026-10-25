"""r9n 新题的题面自检（2026-09-30 晚）：逐字确定 + 与 r9m 不重复。

  python tools/laya/check_r9n_render.py --ds r9n            # 用不同的 PYTHONHASHSEED 各跑一次

- 对每道新题（new_ids.json + carryx_ids.json）按它记下的 render_cfg（当时的 routes.CARRY_TIERS / EXCLUDE_CARRIES）重渲染，
  线上题再限回存下的选项集合，和数据集里存的题面 / 选项 / 说明逐字比对。
  注意：tools/laya/rerender.py 和 check_rerender_same.py 不认 render_cfg —— 对 r9n 的新题要用本脚本查。
- 查 id 唯一、新题与 r9m 题（按 题型+题面+选项）不重复、新旧 run 不交叉。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import build_r9n as B  # noqa: E402
import rerender as RR  # noqa: E402


def sig(p):
    return hashlib.md5((p["kind"] + "\x00" + p["state"] + "\x00" + json.dumps(p["question"]["criteria"], ensure_ascii=False,
                                                                           sort_keys=True)).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="r9n")
    ap.add_argument("--base", default="r9m")
    args = ap.parse_args()
    RR.ALLOW_NEW_KEYS = True
    ds = HERE.parents[1] / "data" / "laya" / "datasets"
    pts = [json.loads(l) for l in (ds / args.ds / "points.jsonl").open(encoding="utf-8")]
    base = [json.loads(l) for l in (ds / args.base / "points.jsonl").open(encoding="utf-8")]
    want = set(json.loads((ds / args.ds / "new_ids.json").read_text(encoding="utf-8"))) | set(json.loads((ds / args.ds / "carryx_ids.json").read_text(encoding="utf-8")))
    new = [p for p in pts if p["id"] in want]
    assert len(new) == len(want), (len(new), len(want))
    ids = collections.Counter(p["id"] for p in pts)
    dup_ids = [i for i, k in ids.items() if k > 1]
    base_sig = {sig(p) for p in base}
    base_runs = {p["run"] for p in base}
    overlap = sum(1 for p in new if sig(p) in base_sig)
    run_cross = sorted({p["run"] for p in new} & base_runs)
    new_sig = collections.Counter(sig(p) for p in new)
    bad, tot = collections.Counter(), collections.Counter()
    for p in sorted(new, key=lambda p: p["run"]):
        tot[p["kind"]] += 1
        B.apply_cfg(p.get("render_cfg"))
        q = dict(p)
        r = RR.rerender(q)
        if r is None:
            bad[p["kind"] + ":dropped"] += 1
            continue
        if p.get("restricted_to_online") or list(r["question"]["criteria"]) != p["options"]:
            B.restrict_to(r, p["options"])
        for part, a, b in (("state", p["state"], r["state"]), ("criteria", p["question"]["criteria"], r["question"]["criteria"]),
                           ("instr", p["question"].get("instructions"), r["question"].get("instructions")),
                           ("options", p["options"], r["options"])):
            if a != b:
                bad[f"{p['kind']}:{part}"] += 1
    B.apply_cfg(None)
    out = {"points": len(pts), "new": len(new), "dup_ids": len(dup_ids), "new_text_in_base": overlap,
           "new_text_dup_within_new": sum(k - 1 for k in new_sig.values() if k > 1),
           "run_cross": run_cross, "checked": dict(tot), "diff": dict(bad)}
    print(json.dumps(out, ensure_ascii=False))
    return 1 if (bad or dup_ids or overlap or run_cross) else 0


if __name__ == "__main__":
    sys.exit(main())
