"""部署包（线上 ONNX 路径）在数据集测试题上的 top1 / tmass（2026-10-01，v3 选型用）。

  python tools/cgnet/eval_deploy.py --deploys data/cgnet/deploy,data/cgnet/deploy_v3 --ds r9mn_f9l --splits test,test_r9n_new

走 CGNetPolicy.probs()（与线上同一条特征构造），"已得天赋"用 TalentTail 回放 states.jsonl 到出题时刻。
标签：data/laya/labels/r6/labels_all_<ds>.jsonl。
"""
from __future__ import annotations

import argparse, collections, json, sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from agent.decision.cgnet_policy import CGNetPolicy, TalentTail  # noqa: E402


def owned_at(run: str, ts: float, cache: dict) -> list:
    key = run
    if key not in cache:
        rows = []
        f = PROJECT_DIR / "data" / "logs" / run / "states.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        cache[key] = rows
    tail = TalentTail(None)
    for x in cache[key]:
        if float(x.get("ts") or 0) >= ts:
            break
        tail.feed(x)
    return list(tail.owned)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploys", required=True)
    ap.add_argument("--ds", default="r9mn_f9l")
    ap.add_argument("--splits", default="test,test_r9n_new")
    ap.add_argument("--kinds", default="carry,swap")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    labs = {}
    for l in (PROJECT_DIR / "data/laya/labels/r6" / f"labels_all_{a.ds}.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        labs[r["id"]] = r
    splits, kinds = set(a.splits.split(",")), set(a.kinds.split(","))
    pts = [json.loads(l) for l in (PROJECT_DIR / "data/laya/datasets" / a.ds / "points.jsonl").open(encoding="utf-8")]
    pts = [p for p in pts if p["split"] in splits and p["kind"] in kinds and p["id"] in labs]
    cache: dict = {}
    owned = {p["id"]: owned_at(p.get("run") or "", float(p.get("ts") or 0), cache) for p in pts}
    res = {}
    per_item = {}
    for dep in a.deploys.split(","):
        pol = CGNetPolicy.load(deploy_dir=PROJECT_DIR / dep)
        agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
        per_item[dep] = {}
        for p in pts:
            lab = labs[p["id"]]
            pr = pol.probs(dict(p), owned_talents=owned[p["id"]])
            ch = max(pr, key=pr.get)
            hit, tm = float(ch == lab["best"]), float(lab["dist"].get(ch, 0.0))
            per_item[dep][p["id"]] = (hit, tm)
            for g in (f"{p['split']}:all", f"{p['split']}:{p['kind']}"):
                agg[g][0] += 1
                agg[g][1] += hit
                agg[g][2] += tm
        res[dep] = {g: {"n": v[0], "top1": round(v[1] / v[0], 3), "tmass": round(v[2] / v[0], 3)} for g, v in sorted(agg.items())}
    for dep, r in res.items():
        print(dep)
        for g, v in r.items():
            print(f"  {g:24s} n={v['n']:3d} top1={v['top1']:.3f} tmass={v['tmass']:.3f}")
    deps = a.deploys.split(",")
    if len(deps) >= 2:
        b, c = deps[0], deps[1]
        ids = list(per_item[b])
        d = np.array([per_item[c][i][1] - per_item[b][i][1] for i in ids])
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        print(f"tmass diff {c} - {b}: {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}] n={len(d)}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
