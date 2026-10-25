"""r9m 按 f9l 题面重渲染，选项限回 r9m 原来的键（沿用 g9j 标签，不重标），供 R16 与 r9n_f9l 合并训练（2026-10-01）。

  python tools/laya/rerender_r9m_f9l.py            # 写 datasets/r9m_f9l/
  python tools/laya/rerender_r9m_f9l.py --check    # 只渲染并打印统计与内容哈希
限不回原键的题丢弃（标签对不上）。
"""
from __future__ import annotations

import argparse, collections, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_r9n as BR  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

VER = "f9l"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="r9m")
    ap.add_argument("--out", default="r9m_f9l")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    F.ACTIVE = VER
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    pts = [json.loads(l) for l in (ds / args.src / "points.jsonl").open(encoding="utf-8")]
    st = collections.Counter()
    out = {}
    for p in sorted(pts, key=lambda p: (p.get("run") or "", p["id"])):
        BR.apply_cfg(p.get("render_cfg"))
        r = RR.rerender(dict(p))
        if r is None:
            st["dropped_render"] += 1
            continue
        keys = list(p["options"])
        if list(r["question"]["criteria"]) != keys:
            st["keys_changed_" + p["kind"]] += 1
            if not BR.restrict_to(r, keys) or list(r["question"]["criteria"]) != keys:
                st["dropped_keys_" + p["kind"]] += 1
                continue
        r["feature_version"] = VER
        if r["state"] != p["state"] or r["question"] != p["question"]:
            st["text_changed_" + (p.get("split") or "?")] += 1
        out[p["id"]] = r
    BR.apply_cfg(None)
    h = hashlib.md5()
    order = [p["id"] for p in pts if p["id"] in out]
    for i in order:
        h.update(json.dumps(out[i], ensure_ascii=False, sort_keys=True).encode())
    summary = {"src": args.src, "points": len(out), "src_points": len(pts), "stats": dict(sorted(st.items())),
               "by_split": dict(collections.Counter(out[i].get("split") for i in order)), "md5": h.hexdigest()}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    (ds / args.out).mkdir(parents=True, exist_ok=True)
    with (ds / args.out / "points.jsonl").open("w", encoding="utf-8") as fh:
        for i in order:
            fh.write(json.dumps(out[i], ensure_ascii=False) + "\n")
    (ds / args.out / "info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
