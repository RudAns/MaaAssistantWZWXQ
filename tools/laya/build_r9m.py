"""r9m 数据集（f9k，2026-09-29）：r9l 全部重渲染 + r9l 之后各局的线上题 + 全部历史日志离线抽的 route 题。

  python tools/laya/build_r9m.py --base r9l --out r9m

- r9l 的题：按 f9k 重渲染（老题从 states.jsonl / 账本还原当时锁定的路线，rerender.route_at）。
- r9l 之后的对局（run 不在 r9l 里、有 laya_points.jsonl 的 live_* 目录）：线上题剥掉学生答案，重渲染成 f9k，进 train。
- route 题：对全部 live_* 目录跑 build_dataset.extract_run，只留 kind=route；划分跟着该局在 r9l 里的划分（没出现过的进 train），
  保证测试集那几局的 route 题也在测试集里、不漏进训练。
- new_ids.json = 需要（重）标的题：新题 + 选项 / 题目说明和 r9l 不一致的老题（只有局面文字变了的沿用旧标签）。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_dataset as BD  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

STUDENT_KEYS = ("laya", "laya_choice", "laya_conf", "laya_ms", "used", "why", "ckpt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r9l")
    ap.add_argument("--out", default="r9m")
    args = ap.parse_args()
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    base = [json.loads(l) for l in (ds / args.base / "points.jsonl").open(encoding="utf-8")]
    run_split = {}
    for p in base:
        run_split.setdefault(p.get("run"), p.get("split"))
    out, relabel, dropped = [], set(), collections.Counter()
    for p in base:
        r = RR.rerender(p)
        if r is None:
            dropped[p["kind"]] += 1
            continue
        # 只重标**选项 / 题目说明**变了的旧题（省 token，2026-09-28）；只有局面文字变了（多了 `route: none committed yet`、
        # 阵容列表去掉不参选路线）的沿用旧标签 —— 选项结论没变，老师的判断基本不受影响
        if r["question"] != p["question"]:
            relabel.add(r["id"])
        out.append(r)
    have = {p["id"] for p in out} | {p["id"] for p in base}
    base_runs = set(run_split)
    new = []
    # r9l 之后各局的线上题
    for d in sorted((PROJECT_DIR / "data" / "logs").glob("live_*")):
        f = d / "laya_points.jsonl"
        if d.name in base_runs or not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            try:
                p = json.loads(line)
            except ValueError:
                continue
            if p.get("kind") == "route" or not p.get("question"):
                continue
            for k in STUDENT_KEYS:
                p.pop(k, None)
            p.setdefault("run", d.name)
            p["id"] = hashlib.md5(f"{p['run']}|{p.get('ts')}|{p['kind']}".encode()).hexdigest()[:12]
            if p["id"] in have:
                continue
            r = RR.rerender(p)
            if r is None:
                dropped["new:" + p["kind"]] += 1
                continue
            r["split"] = "train"
            have.add(r["id"])
            new.append(r)
    # 全部历史日志的 route 题
    for d in sorted((PROJECT_DIR / "data" / "logs").glob("live_*")):
        if not (d / "states.jsonl").exists():
            continue
        try:
            pts = BD.extract_run(d)
        except Exception as e:  # noqa: BLE001
            print("extract 失败", d.name, e)
            continue
        for p in pts:
            if p["kind"] != "route":
                continue
            p["id"] = hashlib.md5(f"{p['run']}|{p.get('ts')}|{p['kind']}".encode()).hexdigest()[:12]
            if p["id"] in have:
                continue
            p["split"] = run_split.get(p["run"], "train")
            p["feature_version"] = F.FEATURE_VERSION
            have.add(p["id"])
            new.append(p)
    allp = out + new
    (ds / args.out).mkdir(parents=True, exist_ok=True)
    with (ds / args.out / "points.jsonl").open("w", encoding="utf-8") as fh:
        for r in allp:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    need = sorted(relabel | {p["id"] for p in new})
    (ds / args.out / "new_ids.json").write_text(json.dumps(need), encoding="utf-8")
    info = {"base": args.base, "feature_version": F.FEATURE_VERSION, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "points": len(allp), "new_n": len(new), "relabel_n": len(relabel), "need_label": len(need),
            "dropped": dict(dropped),
            "relabel_by_kind": dict(collections.Counter(p["kind"] for p in out if p["id"] in relabel)),
            "new_by_kind": dict(collections.Counter(p["kind"] for p in new)),
            "new_by_split": dict(collections.Counter((p["kind"], p["split"]) for p in new if p["kind"] == "route")),
            "by_split": dict(collections.Counter(p.get("split") for p in allp))}
    info["new_by_split"] = {f"{k[0]}:{k[1]}": v for k, v in info["new_by_split"].items()}
    (ds / args.out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
