"""r9zn 标签合并（2026-10-16，f9lw / g9zh）：r9zm_all（R33 的全部标签，含 g9zg 修正）+ g9zh 重标 r9zn / r9zn_test，覆盖同 id。

**旧标签按裁决校正**（g9zh 54.9；没被重标、但题面 f9lw 已写出裁决对象的题）：
2026-10-16 起要求下次训练结果里不再出现这些旧问题。这几条裁决是硬约束，旧标签和它冲突就会把旧毛病教回去，所以按选项文本校正：
  C1 选项写 `OTHER LINEUP carry` / `OTHER LINEUP's carry` 的买 / 选 / 拍 / 换主C → 概率封顶 0.05（54.9 第 1 条）；
  C2 换主C 题有 `maple moves to the future carry` 的选项 → 至少 0.8（54.9 第 2 条）；
  C3 第 3 回合以后商店买弈星（选项写 `TRANSITIONAL filler` / `PAID-OFF filler`）→ 封顶 0.05（54.9 第 3 条）。
校正只动旧标签，重标过的不动；被校正的记 g9zh_fix，stats 里分条计数。
输出：data/laya/labels/r9zn_all/labels_all.jsonl、stats.json。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
LAB = PROJECT_DIR / "data" / "laya" / "labels"
DS = PROJECT_DIR / "data" / "laya" / "datasets" / "r9zn"
BUY_KINDS = ("shop", "card_pick", "auction", "carry")


def read(f: Path) -> dict:
    out = {}
    if f.exists():
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def _renorm(dist: dict, fixed: dict) -> dict:
    """把 fixed 里的键定成给定值，其余按原比例分剩下的概率。"""
    rest = {k: v for k, v in dist.items() if k not in fixed}
    left = max(0.0, 1.0 - sum(fixed.values()))
    s = sum(rest.values())
    out = dict(fixed)
    for k, v in rest.items():
        out[k] = left * (v / s if s > 0 else 1 / max(1, len(rest)))
    return {k: round(v, 4) for k, v in out.items()}


def fix(p: dict, r: dict) -> tuple[dict, list]:
    crit = p["question"]["criteria"]
    d = dict(r["dist"])
    why = []
    if p["kind"] in BUY_KINDS:
        caps = {k: min(d.get(k, 0), 0.05) for k, v in crit.items() if "OTHER LINEUP" in v and d.get(k, 0) > 0.05}
        if caps and len(caps) < len(d):
            d = _renorm(d, caps)
            why.append("C1")
    if p["kind"] == "carry":
        mv = [k for k, v in crit.items() if "maple moves to the future carry" in v]
        if len(mv) == 1 and d.get(mv[0], 0) < 0.8:
            d = _renorm(d, {mv[0]: 0.8})
            why.append("C2")
    if p["kind"] == "shop" and (p.get("round") or 0) > 3:
        caps = {k: min(d.get(k, 0), 0.05) for k, v in crit.items() if k.startswith("buy")
                and re.search(r"TRANSITIONAL filler|PAID-OFF filler", v) and d.get(k, 0) > 0.05}
        if caps and len(caps) < len(d):
            d = _renorm(d, caps)
            why.append("C3")
    if not why:
        return r, []
    nr = dict(r, dist=d, best=max(d, key=d.get), g9zh_fix=why)
    return nr, why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-frac", type=float, default=0.9)
    args = ap.parse_args()
    old = read(LAB / "r9zm_all" / "labels_all.jsonl")
    stale_f = DS / "r9zn_stale_ids.json"           # 第一轮标完后题面又修过：文本变了 / 题没了的旧标签作废（r9zn_delta.py）
    stale = json.loads(stale_f.read_text(encoding="utf-8")) if stale_f.exists() else {}
    if isinstance(stale, list):
        stale = {"r9zn": stale}
    new = {}
    for d in ("r9zn", "r9zn_b", "r9zn_test", "r9zn_c", "r9zn_test_c", "r9zn_d", "r9zn_test_d"):     # 后一轮覆盖前一轮；作废的（文本已变）不要
        bad = set(stale.get(d) or [])
        new.update({i: r for i, r in read(LAB / d / "labels.jsonl").items() if i not in bad})
    pts = {}
    for line in (DS / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    tr = set(json.loads((DS / "teacher_train_ids.json").read_text(encoding="utf-8")))
    te = set(json.loads((DS / "teacher_test_ids.json").read_text(encoding="utf-8")))
    bad = [i for i, r in new.items() if i not in pts or set(r["dist"]) != set(pts[i]["options"])]
    if bad:
        print("新标签与 r9zn 选项对不上：", bad[:10])
        return 1
    merged = {i: r for i, r in old.items() if i in pts and set(r["dist"]) == set(pts[i]["options"])}
    dropped_old = len(old) - len(merged)
    fixc = collections.Counter()
    for i in list(merged):
        if i in new:
            continue
        nr, why = fix(pts[i], merged[i])
        if why:
            merged[i] = nr
            for w in why:
                fixc[f"{w}:{pts[i]['kind']}:{'train' if pts[i]['split'] == 'train' else 'test'}"] += 1
            if nr["best"] != old[i]["best"]:
                fixc["best_changed"] += 1
    merged.update(new)
    out_dir = LAB / "r9zn_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labels_all.jsonl").open("w", encoding="utf-8") as fh:
        for r in merged.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st: dict = {"merged": len(merged), "old": len(old), "old_dropped_options_changed": dropped_old, "g9zh_labels": len(new),
                "old_label_fixes": dict(sorted(fixc.items())),
                "relabel_train": len(tr), "relabel_train_done": len(tr & set(new)),
                "relabel_test": len(te), "relabel_test_done": len(te & set(new)),
                "unlabeled_points": sum(1 for i in pts if i not in merged)}
    c = collections.Counter()
    for i, r in new.items():
        p = pts[i]
        o = old.get(i)
        c[(p["kind"], "new" if not o else ("best_same" if o["best"] == r["best"] else "best_changed"))] += 1
    st["counts"] = {"|".join(map(str, k)): v for k, v in sorted(c.items())}
    st["mean_conf"] = round(sum(float(r.get("confidence") or 0) for r in new.values()) / max(1, len(new)), 3)
    st["missing_info_top"] = collections.Counter(m for r in new.values() for m in (r.get("missing_info") or [])).most_common(20)
    (out_dir / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=1))
    ok = st["relabel_train_done"] >= args.min_frac * max(1, len(tr)) and st["relabel_test_done"] == len(te)
    if not ok:
        print(f"标注未完成：训练 {st['relabel_train_done']}/{len(tr)}，测试 {st['relabel_test_done']}/{len(te)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
