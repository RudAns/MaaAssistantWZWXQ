"""unit_id_v2 数据集：用全部对局（v1 只有 97 局）重建"格子上站的是谁"（2026-10-12）。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_unit_id_v2.py

图与检测框：build_frames_v2 的缓存（与 v1 同一裁法 80x128 BGR、同一 D-FINE、同样只取 prepare + 商店关 + 没开详情的帧）。
标签两路：
  ① strong（与 v1 完全相同的规则，build_unit_id.parse_logs / change_cells）：只用"点开详情读到的名字"，
     确认时刻到帧之间这一格没有换人动作、同格冲突/同名两格丢弃、|Δt| ≤ 900s、确认时刻视觉说那格没人的丢弃、可疑（徽章映射到相邻两格）丢弃；
  ② weak（账本一致性，**只进训练集**）：帧对应状态里 账本人数 = 人口读数（有效）= D-FINE 英雄数，且账本格子集合 = D-FINE 英雄格子集合；
     该名字在前后 20 秒所有带账本的状态里都记在这一格、窗口内这一格没有换人动作；线上认人模型（≥0.9）若给出不同名字则丢弃；
     同一帧同名两格丢弃。和 strong 重复的格子以 strong 为准。
划分：与 v1 相同的对局分组 / md5 划分（v2_common.match_map，已核对 v1 的 112 个 run 划分一致）。
输出 data/vision_v2/unit_id_v2/{crops.npz, samples.jsonl, info.json}。
"""
from __future__ import annotations

import json
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools" / "vision"))  # noqa: E702
import v2_common as V  # noqa: E402
from build_unit_id import (MAX_DT, NBR_DIST, change_cells, ckey, load_cards, nearest,  # noqa: E402
                           parse_logs)

OUT = V.V2 / "unit_id_v2"
WEAK_WIN = 20.0
WEAK_CAP = 40          # 每 (对局, 英雄) 最多收这么多 weak 样本（同一局相邻帧高度相似）


def main():
    cat, transform = load_cards()
    ev = parse_logs(cat)
    print(f"确认事件 {len(ev)}（可疑 {sum(e['bad'] for e in ev)}）", flush=True)
    ev.sort(key=lambda e: e["ts"])
    ets = [e["ts"] for e in ev]
    mm = V.match_map()
    OUT.mkdir(parents=True, exist_ok=True)
    crops, samples, stats = [], [], Counter()
    weak_n = Counter()
    for ri, run in enumerate(V.frame_runs()):
        meta, z = V.load_frames(run)
        if not meta:
            continue
        seq = V.load_seq(run)
        tss = [r["ts"] for r in seq]
        recs = [{"ts": r["ts"], "phase": r["phase"], "shop": r["shop"], "panel": r["panel"], "round": r["round"],
                 "vis": r["vis"], "lv": r["lv"]} for r in seq]
        changes = defaultdict(list)
        for r in seq:
            for a in r["actions"]:
                for c, mn in change_cells(a, cat, transform):
                    changes[c].append((r["ts"], mn))
        lo, hi = bisect_left(ets, tss[0] - 5), bisect_left(ets, tss[-1] + 5)
        rev = [e for e in ev[lo:hi] if not e["bad"]]
        for e in rev:
            r = nearest(recs, tss, e["ts"], before=6, after=0.5)
            vis = (r or {}).get("vis")
            if vis:
                e["conf_vis"] = ckey(e["cell"]) in vis
                e["nbr"] = any(k != ckey(e["cell"]) and ((int(k.split(',')[0]) - e["cell"][0]) ** 2 +
                               (int(k.split(',')[1]) - e["cell"][1]) ** 2) ** .5 <= NBR_DIST for k in vis)
            else:
                e["conf_vis"], e["nbr"] = None, None
        boards = [(r["ts"], r["board"]) for r in seq if r["board"]]
        bts = [b[0] for b in boards]
        match, split = mm.get(run, (run, "train"))
        for f in meta:
            tF = f["t"]
            units = {tuple(map(int, k.split(","))): v for k, v in f["units"].items()}
            # ---- ① strong ----
            cand, emeta = defaultdict(set), {}
            for e in rev:
                if abs(e["ts"] - tF) > MAX_DT or e.get("conf_vis") is False:
                    continue
                c = e["cell"]
                a, b = sorted((e["ts"], tF))
                if any(a - 0.5 <= t <= b and mn != e["name"] for t, mn in changes.get(c, [])):
                    continue
                cand[c].add(e["name"])
                m = emeta.get(c)
                if m is None or abs(e["ts"] - tF) < abs(m["ts"] - tF):
                    emeta[c] = e
            got = {}
            names = Counter(next(iter(v)) for v in cand.values() if len(v) == 1)
            for c, nm in cand.items():
                if len(nm) > 1:
                    stats["conflict"] += 1
                    continue
                name = next(iter(nm))
                if names[name] > 1:
                    stats["dup_name"] += 1
                    continue
                u = units.get(c)
                if not u or u[0] != "hero":
                    stats["no_dfine"] += 1
                    continue
                got[c] = (name, "strong", emeta[c])
            # ---- ② weak（账本一致性）----
            bd = f.get("board")
            heroes = {c for c, u in units.items() if u[0] == "hero"}
            if bd and f.get("pop_ok") and f.get("pop") == len(bd) == len(heroes):
                cells = {}
                for n, info in bd.items():
                    cc = info.get("cell") if isinstance(info, dict) else None
                    if cc:
                        cells[n] = tuple(map(int, cc.split(",")))
                if set(cells.values()) == heroes and len(set(cells.values())) == len(cells):
                    i0, i1 = bisect_left(bts, tF - WEAK_WIN), bisect_left(bts, tF + WEAK_WIN)
                    win = boards[i0:i1]
                    for n, c in cells.items():
                        if c in got:
                            continue
                        if len(win) < 2 or any(not isinstance(b.get(n), dict) or b[n].get("cell") != ckey(c) for _, b in win):
                            stats["weak_unstable"] += 1
                            continue
                        if any(tF - WEAK_WIN - 0.5 <= t <= tF + WEAK_WIN and mn != n for t, mn in changes.get(c, [])):
                            stats["weak_changed"] += 1
                            continue
                        on = f["names"].get(ckey(c))
                        if on and on[1] >= 0.9 and on[0] != n:
                            stats["weak_model_disagree"] += 1
                            continue
                        if cat.get(n) != "英雄牌":
                            continue
                        got[c] = (n, "weak", None)
            for c, (name, src, e) in got.items():
                if src == "weak" and split != "train":
                    stats["weak_not_train"] += 1
                    continue
                if src == "weak":
                    if weak_n[(match, name)] >= WEAK_CAP:
                        stats["weak_capped"] += 1
                        continue
                    weak_n[(match, name)] += 1
                u = units[c]
                sid = f"{run[5:]}_{int(tF * 1000)}_{c[0]}_{c[1]}"
                samples.append({"id": sid, "i": len(crops), "run": run, "frame": f["frame"], "kind": f["kind"], "ts": tF,
                                "round": f["round"], "cell": ckey(c), "name": name, "src": src,
                                "box": u[2], "det_score": u[1], "level": f["lv"].get(ckey(c)),
                                "dt": round(e["ts"] - tF, 1) if e else None, "nbr": e.get("nbr") if e else None,
                                "match": match, "split": split})
                crops.append(z["unit"][u[3]].copy())
                stats[src] += 1
        if ri % 50 == 0:
            print(f"[{ri}] {run} 累计 {len(samples)} {dict(stats)}", flush=True)
    np.savez_compressed(OUT / "crops.npz", unit=np.stack(crops))
    with open(OUT / "samples.jsonl", "w", encoding="utf-8") as fo:
        for s in samples:
            fo.write(json.dumps(s, ensure_ascii=False) + "\n")
    by = defaultdict(Counter)
    for s in samples:
        by[s["name"]][s["split"] + "_" + s["src"]] += 1
    info = {"samples": len(samples), "stats": dict(stats), "matches": len({s["match"] for s in samples}),
            "by_split": dict(Counter(s["split"] for s in samples)),
            "classes": {n: dict(c) for n, c in sorted(by.items(), key=lambda x: -sum(x[1].values()))}}
    (OUT / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "classes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
