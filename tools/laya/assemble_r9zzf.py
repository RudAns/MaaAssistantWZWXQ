"""r9zzf 标签合并（2026-10-20 晚，f9z / g9zza）：r9zze_all（R49 的全部标签）+ g9zza 重标 r9zzf / r9zzf_test。
更早：r9zze 标签合并（2026-10-20，f9y / g9zz）：r9zzd_all（R48 的全部标签）+ g9zz 重标 r9zze / r9zze_test。
更早：r9zzd 标签合并（2026-10-19 夜，f9x / g9zy）：r9zzc_all（R47 的全部标签）+ g9zy 重标 r9zzd / r9zzd_test。
更早：r9zzc 标签合并（2026-10-19 晚，f9w / g9zx）：r9zzb_all（R46 的全部标签）+ g9zx 重标 r9zzc / r9zzc_test。
更早：r9zzb 标签合并（2026-10-19，f9u / g9zw）：r9zy_all（R44 的全部标签）+ g9zw 重标 r9zzb / r9zzb_test（含 738~757 局补进的线上题）。
更早：r9zy 标签合并（2026-10-19，f9t / g9zt）：r9zw_all（R43 的全部标签）+ g9zt 重标 r9zy / r9zy_test。
更早：r9zw 标签合并（2026-10-18 晚，f9s / g9zq）：r9zv_all（R42 的全部标签）+ g9zq 重标 r9zw / r9zw_test。
更早：r9zv 标签合并（2026-10-18，f9r / g9zo）：r9zt_all + g9zo 重标 r9zv / r9zv_test。
更早：r9zt 标签合并（2026-10-17 晚，f9p / g9zn）：r9zs_all（R39 的全部标签）+ g9zn 重标 r9zt / r9zt_test。
更早：r9zs 标签合并（2026-10-17 晚，f9o / g9zm）：r9zr_all（R38 的全部标签）+ g9zm 重标 r9zs / r9zs_test，覆盖同 id。
更早：r9zr 标签合并（2026-10-17，f9n / g9zl）：r9zq_all（R37 的全部标签）+ g9zl 重标 r9zr / r9zr_test，覆盖同 id。
更早：r9zq 标签合并（2026-10-17，f9lz / g9zk）：r9zp_all（R36 的全部标签）+ g9zk 重标 r9zq / r9zq_test，覆盖同 id；旧标签校正 C1~C3 照旧。
更早：r9zp 标签合并（2026-10-16 晚，f9ly / g9zj）：r9zo_all（R35 的全部标签）+ g9zj 重标 r9zp / r9zp_test（+ 增量 _c），覆盖同 id。
C1 按新裁决（路线定下之后，其他主C 优先级低于核心牌）：只在同题有核心牌选项时把别家主C 压到核心之下（≤0.1）。
更早：r9zo 标签合并（2026-10-16，f9lx / g9zi）：r9zn_all（R34 的全部标签）+ g9zi 重标 r9zo / r9zo_test（+ 增量重标 _c），覆盖同 id。

旧标签校正沿用 assemble_r9zn 的 C1~C3（裁决：别家主C 封顶 0.05 / 换绑枫叶舞 ≥0.8 / 第 3 回合后不买弈星）。
f9lx 起 OTHER LINEUP 也覆盖自定义路线主C（第 665 局马可波罗），C1 自动作用到这些旧标签。
**sell_filler 整类剔除**（54.10 已停用：线上不再问；老师按两套口径标，留着只会教乱）。
输出：data/laya/labels/r9zp_all/labels_all.jsonl、stats.json。
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
DS = PROJECT_DIR / "data" / "laya" / "datasets" / "r9zzf"
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
        # g9zj 1：别家主C 排在核心牌之后 —— 同题有核心牌 / 路线最终阵容 / 喂主C 的选项时才压（≤0.1 且低于最好的核心选项）
        core = [k for k, v in crit.items() if "OTHER LINEUP" not in v and re.search(r"CORE of route|ROUTE final team|ROUTE carry|feeds the carry|FEEDS carry", v)]
        if core:
            top = max(d.get(k, 0) for k in core)
            caps = {k: min(d.get(k, 0), 0.1, top * 0.9) for k, v in crit.items() if "OTHER LINEUP" in v and d.get(k, 0) > min(0.1, top * 0.9)}
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
    if p["kind"] == "talent":
        # g9zza 54.28-2（老师在幸运硬币上分成两派：r9zzf batch_003 / 010 / dup_001）：路线已锁但路线主C 没到手 → 按路线没定，幸运硬币 ≥0.6
        st = p.get("state") if isinstance(p.get("state"), str) else ""
        lk = [k for k, v in crit.items() if str(v).startswith("幸运硬币") and "needs route carry 狄仁杰" in str(v)]
        if len(lk) == 1 and "not owned yet" in st and d.get(lk[0], 0) < 0.6:
            d = _renorm(d, {lk[0]: 0.6})
            why.append("C4")
    if not why:
        return r, []
    nr = dict(r, dist=d, best=max(d, key=d.get), g9zh_fix=why)
    return nr, why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-frac", type=float, default=0.9)
    args = ap.parse_args()
    old = read(LAB / "r9zze_all" / "labels_all.jsonl")
    assert len(old) > 10000, f"旧标签只有 {len(old)} 条 —— 读旧标签的路径错了？（10-18 R43 第一次就因此只训了 937 题）"
    stale_f = DS / "stale_ids.json"           # 第一轮标完后题面又修过：文本变了 / 题没了的旧标签作废（r9zn_delta.py）
    stale = json.loads(stale_f.read_text(encoding="utf-8")) if stale_f.exists() else {}
    if isinstance(stale, list):
        stale = {"r9zzf": stale}
    new = {}
    for d in ("r9zzf", "r9zzf_test", "r9zzf_c", "r9zzf_test_c"):     # 后一轮覆盖前一轮；作废的（文本已变）不要
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
        print("新标签与 r9zp 选项对不上：", bad[:10])
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
    for i in list(new):                      # C4 也作用于本轮新标签（幸运硬币分歧）
        nr, why = fix(pts[i], new[i])
        if "C4" in why:
            new[i] = nr
            fixc["C4:new"] += 1
    merged.update(new)
    out_dir = LAB / "r9zzf_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labels_all.jsonl").open("w", encoding="utf-8") as fh:
        for r in merged.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st: dict = {"merged": len(merged), "old": len(old), "old_dropped_options_changed": dropped_old, "g9zz_labels": len(new),
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
