"""equip_cnt_v1 数据集：英雄身上的装备件数 0~3（2026-10-12，审查报告 ②）。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_equip_cnt.py

图：build_frames_v2 缓存里的装备条裁图（徽章中心为基准 x∈[bx-52,bx+52)、y∈[by,by+60)，104x60 BGR；
    覆盖规则读法 detect_equip_counts 的 bx±45 / by+10~42 那块，往下多留一截让下排图标完整）。
标签（自动，"区间稳定值"）：
  事件 = 这一格的换人动作（上场/卖/挪/变身移除，build_unit_id.change_cells）+ 往这一格拖装备牌 + 拆装备类效果牌；
  帧时刻 t 所在的"无事件区间"（同一回合、准备阶段且商店关）里，真实件数不变 → 区间内规则读数的众数（≥4 个读数、占比 ≥0.6）当标签。
  交叉核对（"账本"）：这一格单位名字（线上认人 ≥0.9，否则账本 board 反查）在 t 之前被执行过的装备拖拽次数
  （从它最近一次被卖掉之后算，封顶 3）。等于标签 → strong；否则 weak。
  单调：同一名字（没被卖过）区间标签按时间不应下降，下降的两段都记 mono_bad。
划分同 v2_common.match_map；测试集只取 strong 且 mono 正常的（再人工复核一部分）。
输出 data/vision_v2/equip_cnt/{crops.npz, labels.jsonl, info.json}。
"""
from __future__ import annotations

import json
import re
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools" / "vision"))  # noqa: E702
import v2_common as V  # noqa: E402
from build_unit_id import change_cells, ckey, load_cards, near_cell  # noqa: E402

OUT = V.V2 / "equip_cnt"
MIN_READ, MIN_SHARE = 4, 0.6


def strip_cards():
    d = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    return {n for n, v in d.items() if v.get("category") == "效果牌"
            and re.search(r"装备", json.dumps(v, ensure_ascii=False)) and re.search(r"摧毁|分解|拆|卸下|移除", json.dumps(v, ensure_ascii=False))}


def main():
    cat, transform = load_cards()
    strip = strip_cards()
    mm = V.match_map()
    OUT.mkdir(parents=True, exist_ok=True)
    crops, rows, stats = [], [], Counter()
    for ri, run in enumerate(V.frame_runs()):
        meta, z = V.load_frames(run)
        if not meta:
            continue
        seq = V.load_seq(run)
        prep = [r for r in seq if r["phase"] == "prepare" and not r["shop"] and not r["panel"]]
        pts = [r["ts"] for r in prep]
        events = defaultdict(list)            # cell -> [ts]
        eq_drags = defaultdict(list)          # name -> [ts]
        sells = defaultdict(list)             # name -> [ts]
        for r in seq:
            for a in r["actions"]:
                t = a.get("type")
                piece = a.get("piece") or ""
                for c, _mn in change_cells(a, cat, transform):
                    events[c].append(r["ts"])
                if t == "drag_card" and a.get("dst") and (a.get("kind") == "装备牌" or cat.get(piece) == "装备牌"):
                    c = near_cell(*a["dst"][:2], tol=70)
                    if c:
                        events[c].append(r["ts"])
                    if a.get("target"):
                        eq_drags[a["target"]].append(r["ts"])
                if t == "drag_card" and a.get("dst") and piece in strip:
                    c = near_cell(*a["dst"][:2], tol=70)
                    if c:
                        events[c].append(r["ts"])
                if t and t.startswith("sell") and t != "sell_card" and piece:
                    sells[piece].append(r["ts"])
        for v in events.values():
            v.sort()
        match, split = mm.get(run, (run, "train"))
        recs_run = []
        for f in meta:
            t, rd = f["t"], f["round"]
            inv = {}
            if f.get("board"):
                for n, info in f["board"].items():
                    if isinstance(info, dict) and info.get("cell"):
                        inv[info["cell"]] = n
            for k, (bx, by, idx) in f["badges"].items():
                c = tuple(map(int, k.split(",")))
                u = f["units"].get(k)
                if not u or u[0] != "hero":
                    stats["no_hero"] += 1
                    continue
                ev = events.get(c, [])
                j = bisect_right(ev, t)
                lo = ev[j - 1] + 1.0 if j else -1e18
                hi = ev[j] - 0.3 if j < len(ev) else 1e18
                i0, i1 = bisect_left(pts, max(lo, t - 120)), bisect_right(pts, min(hi, t + 120))
                R = [prep[i]["eq"].get(k) for i in range(i0, i1) if prep[i]["round"] == rd and k in prep[i]["eq"]]
                R = [x for x in R if isinstance(x, int)]
                if len(R) < MIN_READ:
                    stats["few_reads"] += 1
                    continue
                mode, n = Counter(R).most_common(1)[0]
                if n / len(R) < MIN_SHARE:
                    stats["unstable"] += 1
                    continue
                on = f["names"].get(k)
                name = on[0] if on and on[1] >= 0.9 else inv.get(k)
                led = None
                if name:
                    s0 = max([x for x in sells.get(name, []) if x < t], default=-1e18)
                    led = min(3, sum(1 for x in eq_drags.get(name, []) if s0 < x < t))
                rows.append({"id": len(crops), "run": run, "frame": f["frame"], "kind": f["kind"], "t": t, "round": rd,
                             "cell": k, "badge": [bx, by], "name": name, "label": int(mode), "share": round(n / len(R), 2),
                             "nread": len(R), "ledger": led, "rule_at_state": f["eq"].get(k),
                             "strong": led is not None and led == mode, "match": match, "split": split})
                recs_run.append(rows[-1])
                crops.append(z["equip"][idx].copy())
        # 单调核对（同名、两次之间没被卖）
        by_name = defaultdict(list)
        for r in recs_run:
            if r["name"]:
                by_name[r["name"]].append(r)
        for n, lst in by_name.items():
            lst.sort(key=lambda r: r["t"])
            for a, b in zip(lst, lst[1:]):
                if b["label"] < a["label"] and not any(a["t"] < x < b["t"] for x in sells.get(n, [])):
                    a["mono_bad"] = b["mono_bad"] = True
        if ri % 50 == 0:
            print(f"[{ri}] {run} 累计 {len(rows)}", flush=True)
    np.savez_compressed(OUT / "crops.npz", equip=np.stack(crops))
    with open(OUT / "labels.jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            r.setdefault("mono_bad", False)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"n": len(rows), "stats": dict(stats),
            "label_x_split": {sp: dict(Counter(r["label"] for r in rows if r["split"] == sp)) for sp in ("train", "val", "test")},
            "strong_x_split": {sp: dict(Counter(r["label"] for r in rows if r["split"] == sp and r["strong"] and not r["mono_bad"]))
                               for sp in ("train", "val", "test")},
            "mono_bad": sum(r["mono_bad"] for r in rows),
            "agree_rule_at_state": sum(r["rule_at_state"] == r["label"] for r in rows) / max(len(rows), 1)}
    (OUT / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
