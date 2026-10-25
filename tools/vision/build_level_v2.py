"""level_cls_v2 数据集：在 v1（data/vision/level_v1）之上补 ≥100 级（三位数 / 橙色箭头徽章）样本（2026-10-12）。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_level_v2.py

图：build_frames_v2 缓存里的徽章裁图（64x64 BGR，以 _detect_badges 的徽章中心为中心，和 v1 / 线上一致）。
标签（自动，来自逐帧状态 unit_levels；≥100 时线上是 OCR 宽框读数）：
  对帧时刻 t、格子 k，取同一对局、同一回合、准备阶段且商店关着、|ts - t| ≤ WIN 秒内这一格的所有读数 R；
  ① 三位数：R 里 ≥100 的读数按时间单调不减，且候选值 v 在 R 中出现 ≥3 次、离 t 最近的 ≥100 读数就是 v
     （t 前最后一个 ≥100 读数 ≤ v ≤ t 后第一个 ≥100 读数）；R 中 <0.3v 的读数视为误读（正是要修的 102→1、123→12），不否决；
     跨回合：同一格上一回合的众数 ≤ v ≤ 下一回合的众数（只在两边也 ≥100 时检查）。
  ② 1~99：R 里所有读数相同且 ≥3 个（只取一部分补充多样性，按等级段封顶）。
划分：对局划分同 v2_common.match_map；v1 测试集所在对局（awaken_v1 split=test 的 run）里的帧一律不进训练。
输出 data/vision_v2/level_v2/{crops.npz, labels.jsonl, info.json}。
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "vision"))
import v2_common as V  # noqa: E402

OUT = V.V2 / "level_v2"
WIN = 10.0
CAP_LOW = {"1": 600, "2-9": 1200, "10-99": 2500}     # 1~99 新样本封顶（v1 已有 ~4000 张）


def bucket(lv):
    return "1" if lv == 1 else "2-9" if lv < 10 else "10-99" if lv < 100 else "100+"


def main():
    rng = random.Random(0)
    mm = V.match_map()
    v1_test_runs = {json.loads(l)["run"] for l in (ROOT / "data/vision/awaken_v1/samples.jsonl").read_text(encoding="utf-8").splitlines()
                    if json.loads(l)["split"] == "test"}
    OUT.mkdir(parents=True, exist_ok=True)
    crops, rows, stats = [], [], Counter()
    low_pool = defaultdict(list)
    for run in V.frame_runs():
        meta, z = V.load_frames(run)
        if not meta:
            continue
        seq = [r for r in V.load_seq(run) if r["phase"] == "prepare" and not r["shop"] and not r["panel"]]
        by_cell = defaultdict(list)          # cell -> [(ts, round, lv)]
        for r in seq:
            for k, v in r["lv"].items():
                if isinstance(v, int) and v > 0:
                    by_cell[k].append((r["ts"], r["round"], v))
        # 每格每回合 ≥100 读数的众数
        rmode = {}
        for k, lst in by_cell.items():
            g = defaultdict(list)
            for ts, rd, v in lst:
                if v >= 100:
                    g[rd].append(v)
            for rd, vs in g.items():
                rmode[(k, rd)] = Counter(vs).most_common(1)[0][0]
        match, split = mm.get(run, (run, "train"))
        if run in v1_test_runs:
            split = "v1test_excluded"
        for f in meta:
            t, rd = f["t"], f["round"]
            for k, (bx, by, idx) in f["badges"].items():
                R = [(ts, v) for ts, r2, v in by_cell.get(k, []) if r2 == rd and abs(ts - t) <= WIN]
                if len(R) < 3:
                    continue
                hi = [(ts, v) for ts, v in R if v >= 100]
                label, kind = None, None
                if len(hi) >= 3:
                    vs = [v for _, v in hi]
                    if vs != sorted(vs):
                        stats["hi_not_monotonic"] += 1
                        continue
                    near = min(hi, key=lambda x: abs(x[0] - t))
                    v = near[1]
                    before = [x for ts, x in hi if ts <= t]
                    after = [x for ts, x in hi if ts >= t]
                    if vs.count(v) < 3 or (before and before[-1] > v) or (after and after[0] < v):
                        stats["hi_unstable"] += 1
                        continue
                    lo_mis = [x for _, x in R if x < 100 and x >= 0.3 * v]
                    if lo_mis:
                        stats["hi_mid_conflict"] += 1
                        continue
                    if (rmode.get((k, rd - 1)) or 0) > v or (rmode.get((k, rd + 1)) or 10 ** 6) < v:
                        stats["hi_cross_round"] += 1
                        continue
                    nearest_any = min(R, key=lambda x: abs(x[0] - t))
                    label, kind = v, ("hi_misread_frame" if nearest_any[1] < 100 else "hi")
                elif not hi:
                    vs = {v for _, v in R}
                    if len(vs) == 1:
                        v = vs.pop()
                        low_pool[bucket(v)].append((run, idx, v, match, split, k, f["frame"]))
                    continue
                else:
                    stats["hi_too_few"] += 1
                    continue
                rows.append({"id": len(crops), "run": run, "frame": f["frame"], "cell": k, "badge": [bx, by],
                             "level": label, "kind": kind, "match": match, "split": split})
                crops.append(z["badge"][idx].copy())
                stats[kind] += 1
    # 1~99 新样本：每段封顶，随机抽（按对局打散）
    for b, lst in low_pool.items():
        rng.shuffle(lst)
        for run, idx, v, match, split, k, frame in lst[:CAP_LOW[b]]:
            rows.append({"id": len(crops), "run": run, "frame": frame, "cell": k, "level": v, "kind": "low",
                         "match": match, "split": split, "_idx": idx})
            crops.append(None)
            stats["low_" + b] += 1
    # 补齐低等级样本的裁图（按 run 批量读）
    need = defaultdict(list)
    for r in rows:
        if crops[r["id"]] is None:
            need[r["run"]].append(r)
    for run, lst in need.items():
        _, z = V.load_frames(run)
        for r in lst:
            crops[r["id"]] = z["badge"][r.pop("_idx")].copy()
    np.savez_compressed(OUT / "crops.npz", badge=np.stack(crops))
    with open(OUT / "labels.jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    hi = [r for r in rows if r["level"] >= 100]
    info = {"n": len(rows), "stats": dict(stats),
            "hi_by_split": dict(Counter(r["split"] for r in hi)),
            "hi_matches": len({r["match"] for r in hi}),
            "hi_levels": dict(Counter(r["level"] // 50 * 50 for r in hi)),
            "low_by_split": dict(Counter(r["split"] for r in rows if r["level"] < 100))}
    (OUT / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
