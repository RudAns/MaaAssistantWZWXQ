"""r9n 主标注题（train 78 批 + test 14 批）按 f9l 题面 / g9l 老师视图重渲染（2026-09-30 夜）。

  python tools/laya/rerender_r9n_f9l.py            # 写 datasets/r9n_f9l/ 和 labels/r9n_f9l/
  python tools/laya/rerender_r9n_f9l.py --check    # 只渲染、打印统计和内容哈希，不写文件（逐字确定性检查用）

- 批次划分、批内顺序、id 与 r9n 完全相同（逐批读 labels/r9n/batches/<批>.ids.json）。
- 每题按它记下的 render_cfg（当时的 routes.CARRY_TIERS / EXCLUDE_CARRIES）渲染，同 check_r9n_render.py。
- 选项集合：
    test_*  限回 r9n 原来的选项（老师标签沿用 g9k 的 out/test_*.jsonl，键必须一致，不重标）；
    batch_* 限回线上当时的选项，但 swap 题**永远保留 bench**（老师 g9k：只剩 "levelup: CANNOT AFFORD" 代替不换，含义模糊；
            线上 laya_policy 在路线阶段成员等着上场时会删 bench —— 那是执行策略，不该进训练题面）。
- 题面版本 feature_version = "f9l"（features.FEATURE_VERSION 仍是 f9k，线上 R15 不受影响）。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_r9n as BR  # noqa: E402
import features as F  # noqa: E402
import label_batches as LB  # noqa: E402
import rerender as RR  # noqa: E402

VER = "f9l"
GUIDE = "g9l"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="r9n")
    ap.add_argument("--out", default="r9n_f9l")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    F.ACTIVE = VER
    LB.GUIDE_VERSION = GUIDE
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    lab = PROJECT_DIR / "data" / "laya" / "labels"
    pts = {}
    for line in (ds / args.src / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    listing = json.loads((lab / args.src / "batches.json").read_text(encoding="utf-8"))["batches"]
    listing = [b for b in listing if b["group"] in ("batch", "test")]
    order = []                     # (批名, [id…]) 保持原批次顺序
    for b in listing:
        ids = list(json.loads((PROJECT_DIR / b["ids"]).read_text(encoding="utf-8")).keys())
        order.append((b["name"], b["group"], ids))
    want = [i for _, _, ids in order for i in ids]
    out: dict = {}
    st = collections.Counter()
    kinds_changed = collections.Counter()
    # 按 run 排序渲染（同一 run 的 states.jsonl 缓存复用）；结果按 id 存
    for i in sorted(want, key=lambda i: (pts[i]["run"], i)):
        p = pts[i]
        test = p["split"] != "train"
        BR.apply_cfg(p.get("render_cfg"))
        r = RR.rerender(dict(p))
        if r is None:
            st["dropped"] += 1
            continue
        keys = list(p["options"])
        crit = r["question"]["criteria"]
        if not test and p["kind"] == "swap" and "bench" in crit and "bench" not in keys:
            keys = [k for k in crit if k in set(keys) | {"bench"}]
            st["swap_bench_added"] += 1
        if p.get("restricted_to_online") or list(crit) != keys:
            ok = BR.restrict_to(r, keys)
            if not ok:
                st["restrict_failed_" + ("test" if test else "train")] += 1
        if test and list(r["question"]["criteria"]) != list(p["options"]):
            st["test_keys_mismatch"] += 1
        if list(r["question"]["criteria"]) != list(p["options"]):
            kinds_changed[p["kind"]] += 1
        r["feature_version"] = VER
        out[i] = r
        s = r["state"] + "\n" + json.dumps(r["question"], ensure_ascii=False)
        sp = "test" if test else "train"
        if "ROUTE stale" in r["state"]:
            st[f"route_stale_{sp}"] += 1
        if "not owned yet, committed" in r["state"]:
            st[f"route_not_owned_yet_{sp}"] += 1
        if "(committed r" in p["state"]:
            st[f"route_committed_in_f9k_{sp}"] += 1
        for tag, name in (("full (3/3) →", "equip_full_retarget"), ("ENGINE PART, not a 2nd carry", "engine_part_2nd_cand"),
                          ("2nd carry = extra ENGINE PAYOFF", "payoff_2nd_carry"), ("ENGINE PAYOFF = 2nd carry", "shop_payoff_2nd"),
                          ("fed by", "payoff_fed_by_carry"), ("× ", "multi_give"),
                          ("level next round", "full_shop_level_note"), ("does nothing, same as keep", "swap_levelup_cannot")):
            if tag in s:
                st[f"{name}_{sp}"] += 1
        if r["state"] != p["state"] or r["question"] != p["question"]:
            st[f"text_changed_{sp}"] += 1
    BR.apply_cfg(None)
    h = hashlib.md5()
    for i in want:
        if i in out:
            h.update(json.dumps(out[i], ensure_ascii=False, sort_keys=True).encode())
    summary = {"points": len(out), "want": len(want), "stats": dict(sorted(st.items())),
               "option_sets_changed_by_kind": dict(kinds_changed), "md5": h.hexdigest()}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    # ---- 写数据集
    (ds / args.out).mkdir(parents=True, exist_ok=True)
    with (ds / args.out / "points.jsonl").open("w", encoding="utf-8") as fh:
        for i in want:
            if i in out:
                fh.write(json.dumps(out[i], ensure_ascii=False) + "\n")
    (ds / args.out / "info.json").write_text(json.dumps(dict(summary, src=args.src, feature_version=VER, guide=GUIDE),
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    # ---- 写批次（与 r9n 同名同序）
    root = lab / args.out
    (root / "batches").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)
    new_listing = []
    for name, group, ids in order:
        part = [out[i] for i in ids if i in out]
        md = [f"# 标注批次 {name}（{len(part)} 题，规范 {GUIDE}，题面 {VER}）\n"] + [LB.render_point(p) + "\n" for p in part]
        (root / "batches" / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
        (root / "batches" / f"{name}.ids.json").write_text(json.dumps({p["id"]: p["options"] for p in part}, ensure_ascii=False),
                                                           encoding="utf-8")
        new_listing.append({"name": name, "group": group, "md": f"data/laya/labels/{args.out}/batches/{name}.md",
                            "ids": f"data/laya/labels/{args.out}/batches/{name}.ids.json",
                            "out": (f"data/laya/labels/{args.src}/out/{name}.jsonl" if group == "test"
                                    else f"data/laya/labels/{args.out}/out/{name}.jsonl"),
                            "n": len(part), "by_kind": dict(sorted(collections.Counter(p["kind"] for p in part).items()))})
    (root / "batches.json").write_text(json.dumps({"batches": new_listing}, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest = {"ver": args.out, "src": args.src, "guide": GUIDE, "feature_version": VER, "n": len(out),
                "batches": len(new_listing), "teacher_prompt": "docs/laya-report/teacher-prompt-g9l.txt",
                "note": "test_* 不重标：标签沿用 data/laya/labels/r9n/out/test_*.jsonl（g9k），选项键与 r9n 一致",
                "stats": summary["stats"]}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
