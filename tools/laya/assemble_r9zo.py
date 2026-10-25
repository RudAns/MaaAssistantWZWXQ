"""r9zo 标签合并（2026-10-16，f9lx / g9zi）：r9zn_all（R34 的全部标签）+ g9zi 重标 r9zo / r9zo_test（+ 增量重标 _c），覆盖同 id。

旧标签校正沿用 assemble_r9zn 的 C1~C3（规范裁决：别家主C 封顶 0.05 / 换绑枫叶舞 ≥0.8 / 第 3 回合后不买弈星）。
f9lx 起 OTHER LINEUP 也覆盖自定义路线主C（第 665 局马可波罗），C1 自动作用到这些旧标签。
**sell_filler 整类剔除**（54.10 已停用：线上不再问；老师按两套口径标，留着只会教乱）。
输出：data/laya/labels/r9zo_all/labels_all.jsonl、stats.json。
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
DS = PROJECT_DIR / "data" / "laya" / "datasets" / "r9zo"
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
    old = read(LAB / "r9zn_all" / "labels_all.jsonl")
    stale_f = DS / "stale_ids.json"           # 第一轮标完后题面又修过：文本变了 / 题没了的旧标签作废（r9zn_delta.py）
    stale = json.loads(stale_f.read_text(encoding="utf-8")) if stale_f.exists() else {}
    if isinstance(stale, list):
        stale = {"r9zo": stale}
    new = {}
    for d in ("r9zo", "r9zo_test", "r9zo_c", "r9zo_test_c"):     # 后一轮覆盖前一轮；作废的（文本已变）不要
        bad = set(stale.get(d) or [])
        new.update({i: r for i, r in read(LAB / d / "labels.jsonl").items() if i not in bad})
    pts = {}
    for line in (DS / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    n_sf = sum(1 for p in pts.values() if p["kind"] == "sell_filler")
    pts = {i: p for i, p in pts.items() if p["kind"] != "sell_filler"}          # 54.10：sell_filler 停用，整类剔除
    new = {i: r for i, r in new.items() if i in pts}
    tr = set(json.loads((DS / "teacher_train_ids.json").read_text(encoding="utf-8"))) & set(pts)
    te = set(json.loads((DS / "teacher_test_ids.json").read_text(encoding="utf-8"))) & set(pts)
    bad = [i for i, r in new.items() if i not in pts or set(r["dist"]) != set(pts[i]["options"])]
    if bad:
        print("新标签与 r9zo 选项对不上：", bad[:10])
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
    out_dir = LAB / "r9zo_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labels_all.jsonl").open("w", encoding="utf-8") as fh:
        for r in merged.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st: dict = {"merged": len(merged), "old": len(old), "old_dropped_options_changed": dropped_old, "g9zi_labels": len(new),
                "old_label_fixes": dict(sorted(fixc.items())), "sell_filler_dropped_points": n_sf,
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
