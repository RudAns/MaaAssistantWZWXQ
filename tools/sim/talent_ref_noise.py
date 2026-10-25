"""推演参考的噪声核对（2026-10-15：推演参考要先确认对不对，再用于标注）：
抽 N 道已算过的天赋题，换一组随机种子重算（同样 n），比较两次的"三档一致最优"是否相同。
  .venv/Scripts/python.exe -X utf8 tools/sim/talent_ref_noise.py --ds r9zl --k 40 --n 4 --offset 100003
"""
from __future__ import annotations
import argparse, json, random, statistics as S, sys
from multiprocessing import Pool
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR)); sys.path.insert(0, str(PROJECT_DIR / "tools" / "sim"))


def _one(args):
    p, n, off = args
    import talent_eval as T
    return p["id"], T.eval_point(p, n=n, seed_off=off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--offset", type=int, default=100003)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--both", action="store_true", help="两组种子都按 --n 重算（不用已有参考）")
    a = ap.parse_args()
    ref = json.loads((PROJECT_DIR / "data/sim" / f"talent_ref_{a.ds}.json").read_text(encoding="utf-8"))
    pts = {}
    for l in (PROJECT_DIR / "data/laya/datasets" / a.ds / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(l)
        if p["id"] in ref and "opts" in ref[p["id"]]:
            pts[p["id"]] = p
    random.seed(5); ids = random.sample(sorted(pts), min(a.k, len(pts)))
    with Pool(a.procs) as pool:
        res = dict(pool.imap_unordered(_one, [(pts[i], a.n, a.offset) for i in ids]))
        base = dict(pool.imap_unordered(_one, [(pts[i], a.n, 13) for i in ids])) if a.both else {i: ref[i]["opts"] for i in ids}
    dst = PROJECT_DIR / "data/sim" / f"talent_ref_{a.ds}_noise_n{a.n}.json"
    dst.write_text(json.dumps({i: {"a": base[i], "b": res[i]} for i in ids}, ensure_ascii=False), encoding="utf-8")
    print("写入", dst)


if __name__ == "__main__":
    main()
