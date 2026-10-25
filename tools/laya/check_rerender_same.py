"""旧版本重渲染逐字一致检查（改了 facts/features 之后必须跑）。

  python tools/laya/check_rerender_same.py --ds r9l --ver f9j

把特征版本切到 --ver，用 rerender 重渲染数据集里的每道题，和数据集里存的题面 / 选项逐字比对。
不一致 = 新版本的改动漏进了旧版本（没挂版本号），线上旧学生看到的输入就和训练时不一样。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import features as F  # noqa: E402
import rerender as RR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--ver", required=True)
    ap.add_argument("--show", type=int, default=2, help="每类最多打印几个差异")
    ap.add_argument("--limit", type=int, default=0, help="只抽 N 题（按 --seed 固定抽样；0 = 全量）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="", help="把重渲染结果（id → state/criteria/instr）写进这个 jsonl，便于改动前后逐字对比")
    args = ap.parse_args()
    F.ACTIVE = args.ver
    pts = [json.loads(l) for l in (HERE.parents[1] / "data" / "laya" / "datasets" / args.ds / "points.jsonl").open(encoding="utf-8")]
    if args.limit and args.limit < len(pts):
        import random
        pts = random.Random(args.seed).sample(pts, args.limit)
    dump = open(args.dump, "w", encoding="utf-8") if args.dump else None
    bad, tot, shown = collections.Counter(), collections.Counter(), collections.Counter()
    for p in pts:
        tot[p["kind"]] += 1
        r = RR.rerender(p)
        if dump is not None:
            dump.write(json.dumps({"id": p.get("id"), "kind": p["kind"], "r": None if r is None else
                                   [r["state"], r["question"]["criteria"], r["question"].get("instructions")]}, ensure_ascii=False) + "\n")
        if r is None:
            bad[p["kind"] + ":dropped"] += 1
            continue
        for part, a, b in (("state", p["state"], r["state"]), ("criteria", p["question"]["criteria"], r["question"]["criteria"]),
                           ("instr", p["question"].get("instructions"), r["question"].get("instructions"))):
            if a != b:
                bad[f"{p['kind']}:{part}"] += 1
                if shown[p["kind"] + part] < args.show:
                    shown[p["kind"] + part] += 1
                    print(f"--- {p['kind']} {part} {p.get('id')}")
                    if part == "state":
                        la, lb = a.splitlines(), b.splitlines()
                        for x, y in zip(la, lb):
                            if x != y:
                                print("  old:", x[:200]); print("  new:", y[:200]); break
                        if len(la) != len(lb):
                            print(f"  lines {len(la)} -> {len(lb)}")
                    else:
                        print("  old:", json.dumps(a, ensure_ascii=False)[:300]); print("  new:", json.dumps(b, ensure_ascii=False)[:300])
                break
    print("total", sum(tot.values()), dict(tot))
    print("diff", sum(bad.values()), dict(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
