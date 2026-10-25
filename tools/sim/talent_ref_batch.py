"""给一批天赋题跑推演参考（老师视图用，2026-10-15）：每题 talent_eval.eval_point（往后 6 回合、低 / 中 / 高三档），
写 data/sim/talent_ref_<ds>.json：{题号: {"opts": {选项: {档: [dev, 主C 等级]}}, "best": 三档一致最优的选项或 ""}}。
  .venv/Scripts/python.exe -X utf8 tools/sim/talent_ref_batch.py --ds r9zl --ids data/laya/datasets/r9zl/teacher_train_ids_talent.json --procs 8
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR)); sys.path.insert(0, str(PROJECT_DIR / "tools" / "sim"))


def _one(args):
    p, n = args
    import talent_eval as T
    try:
        res = T.eval_point(p, n=n)
        return p["id"], {"opts": res, "best": T.verdict(res), "no_effect": T.no_effect(res), "n": n}
    except Exception as ex:                                          # noqa: BLE001
        return p["id"], {"error": repr(ex)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    want = set()
    for f in a.ids:
        want |= set(json.loads(Path(f).read_text(encoding="utf-8")))
    pts = [json.loads(l) for l in (PROJECT_DIR / "data" / "laya" / "datasets" / a.ds / "points.jsonl").open(encoding="utf-8")]
    pts = [p for p in pts if p["id"] in want and p["kind"] == "talent"]
    dst = PROJECT_DIR / "data" / "sim" / f"talent_ref_{a.ds}.json"
    out = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {}
    todo = [(p, a.n) for p in pts if p["id"] not in out]
    print(f"待跑 {len(todo)} / {len(pts)}", flush=True)
    with Pool(a.procs) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(_one, todo), 1):
            out[i] = r
            if k % 20 == 0 or k == len(todo):
                dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
                print(f"{k}/{len(todo)}", flush=True)
    dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    ok = [v for v in out.values() if "opts" in v]
    print(f"完成 {len(ok)}，三档一致有结论 {sum(1 for v in ok if v.get('best'))}，出错 {len(out) - len(ok)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
