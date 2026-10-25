"""场上单位认人数据集：真实对局帧里，己方棋盘上某一格站的英雄是谁（2026-09-30）。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_unit_id.py              # 全量构建
  .venv/Scripts/python.exe -X utf8 tools/vision/build_unit_id.py --limit 20   # 只跑最近 20 个对局目录（调试）

为什么：格子上"有没有人"已由 D-FINE 解决（召回 99%+），"是谁"只靠账本 + 点开详情面板读名字，
而点开常点歪到相邻格、账本被改反。卡牌**立绘**随等级变化（官网头像模板匹配已证伪），
棋盘上的 **3D 单位模型**是否随等级变化没验证过 —— 这个数据集就是要试这条路。

标签来源（只用"点开详情读到的名字"这一种直接证据，不用账本）：
  agent_*.log 里的
    【看场】X,Y 站的是 NAME，用时 …                     （点的是等级徽章锚点下方 40px = 单位身上）
    【拖前核对】(X, Y) 那一格站的是 NAME（目标是 …）    （点锚点下方 40px，没锚点时点 (x, y-30)）
    【拖前核对】P 不在 (X, Y)（那儿是 NAME）
  排除：没认出名字的、非英雄牌（召唤物/图腾）、紧跟着"徽章映射到相邻两格，不改记账"的（多半点歪）、
        以及 agent 自己判成"点歪了"的（那格看到的是 …，但 … 记在隔壁）。
标签传播到截图（名字 -> 图）：
  图 = phase_prepare_*.png（进入准备阶段那一帧）+ grab_shadow_pre_*.png（抓手牌前一帧，棋盘无遮挡）；
  要求 ① 这帧对应的逐帧状态是 prepare、商店关着、没开英雄详情；② 我们自己的 D-FINE（CPU 重跑）在这一格看到英雄；
  ③ 确认时刻与这帧之间，**这一格没有任何会换人的动作**（逐帧状态里决策出的：英雄上场/非同名落格、卖人、挪位、
     sequence 里的步骤、会"变身/进化/移除"的效果牌；同名合成、装备牌、普通效果牌不算换人）；
  ④ 同一格在窗口内有两个不同名字的确认 → 冲突，丢弃；⑤ 同一帧里同一个名字落在两格 → 两格都丢；
  ⑥ |确认时刻 - 帧时刻| ≤ 900s。
可信度标记（写进 samples.jsonl，不直接丢）：nbr = 确认时刻相邻格（≤130px）也有人（点歪风险）；
  conf_vis = 确认时刻逐帧状态里 D-FINE 是否在这一格看到人（None = 那时视觉没跑）。
输出 data/vision/unit_id_v1/：crops/<id>.png（80x128 BGR，按 D-FINE 框裁、去掉框顶 15% 的等级徽章）+ samples.jsonl + info.json。
划分：按**对局**（同一局中途重启产生的多个 live_ 目录合并成一局），约 70/15/15。
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import re
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.perception.extractors import OWN_BOARD_ROWS      # noqa: E402

LOGS = ROOT / "data" / "logs"
OUT = ROOT / "data" / "vision" / "unit_id_v1"
CELLS = [tuple(c) for r in OWN_BOARD_ROWS for c in r]
CROP_W, CROP_H = 80, 128
BADGE_CUT = 0.15            # 去掉框顶部这一截（等级徽章）
MAX_DT = 900
NBR_DIST = 130

RE_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})")
RE_LOOK = re.compile(r"【看场】(\d+),(\d+) 站的是 (\S+?)(（[^）]*）)?，用时")
RE_SAME = re.compile(r"【看场】(\d+),(\d+) 和账本里的 .* 是同一个 (\S+)（徽章映射到相邻两格")
RE_CHK = re.compile(r"【拖前核对】\((\d+), (\d+)\) 那一格站的是 (\S+?)（目标是")
RE_NOT = re.compile(r"【拖前核对】\S+ 不在 \((\d+), (\d+)\)（那儿是 (\S+?)）")


def near_cell(x, y, tol=40):
    c = min(CELLS, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)
    return c if (c[0] - x) ** 2 + (c[1] - y) ** 2 <= tol * tol else None


def ckey(c):
    return f"{c[0]},{c[1]}"


def load_cards():
    d = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    cat = {n: v.get("category") for n, v in d.items()}
    transform = set()
    for n, v in d.items():
        if v.get("category") == "效果牌":
            txt = json.dumps(v, ensure_ascii=False)
            if re.search(r"变为|进化|替换|移除|变成", txt):
                transform.add(n)
    return cat, transform


def parse_logs(cat):
    """-> [(ts, cell, name, src)] 以及被标成可疑的 (ts, cell)"""
    ev = []
    for f in sorted(LOGS.glob("agent_*.log")):
        last_look = None
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = RE_TS.match(line)
            if not m:
                continue
            ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000
            for rx, src in ((RE_LOOK, "look"), (RE_CHK, "check"), (RE_NOT, "check_not")):
                mm = rx.search(line)
                if mm:
                    x, y, name = int(mm.group(1)), int(mm.group(2)), mm.group(3)
                    c = near_cell(x, y)
                    if c is None or name.startswith("没认出") or cat.get(name) != "英雄牌":
                        break
                    e = {"ts": ts, "cell": c, "name": name, "src": src, "log": f.name, "bad": False}
                    ev.append(e)
                    if src == "look":
                        last_look = e
                    break
            mm = RE_SAME.search(line)
            if mm and last_look and ts - last_look["ts"] < 3 and last_look["name"] == mm.group(3):
                last_look["bad"] = True          # 同一个单位被映射到相邻两格：这次点开读到的名字归属不明
    return ev


def change_cells(act, cat, transform):
    """一个决策动作会让哪些格子**换人** -> [(cell, 同名合成时的名字 or None)]"""
    out = []
    t = act.get("type")
    if t == "sequence":
        for s in act.get("steps") or []:
            out += change_cells(s, cat, transform)
        return out
    piece = act.get("piece") or ""
    if t == "drag_card" and act.get("dst"):
        c = near_cell(*act["dst"][:2], tol=70)
        k = cat.get(piece)
        if c is None or k == "装备牌" or k == "棋手技能牌":
            return out
        if k == "效果牌" and piece not in transform:
            return out
        if k == "英雄牌" and act.get("merge"):
            out.append((c, piece))              # 同名合成：只有原来站的就是它时才不算换人
        else:
            out.append((c, None))
    elif t in ("sell_unit", "sell", "move_unit") or (t and t.startswith("sell") and t != "sell_card"):
        for key in ("src", "dst"):
            if act.get(key) and isinstance(act[key], (list, tuple)):
                c = near_cell(*act[key][:2], tol=70)
                if c:
                    out.append((c, None))
    return out


def load_run(d: Path, cat, transform):
    recs, changes = [], defaultdict(list)       # changes[cell] = [(ts, merge_name)]
    for line in (d / "states.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        s = r.get("state") or {}
        recs.append({"ts": r["ts"], "phase": s.get("phase"), "shop": s.get("shop_open"),
                     "panel": s.get("hero_panel_open"), "round": s.get("round_no"),
                     "vis": s.get("vision_units"), "lv": s.get("unit_levels") or {},
                     "awk": s.get("awaken_progress") or {}})
        for a in r.get("actions") or []:
            if isinstance(a, dict):
                for c, mn in change_cells(a, cat, transform):
                    changes[c].append((r["ts"], mn))
    recs.sort(key=lambda r: r["ts"])
    return recs, changes


def nearest(recs, tss, t, before=3.0, after=1.5):
    i = bisect_left(tss, t)
    best = None
    for j in (i - 2, i - 1, i, i + 1):
        if 0 <= j < len(recs) and -before <= recs[j]["ts"] - t <= after:
            if best is None or abs(recs[j]["ts"] - t) < abs(best["ts"] - t):
                best = recs[j]
    return best


class Detector:
    """D-FINE（与 agent/perception/vision_units 同一个模型、同一套后处理），CPU、限 4 线程，别抢实战局的资源。"""

    def __init__(self):
        import onnxruntime as ort
        from agent.perception import vision_units as vu
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.log_severity_level = 3
        self.vu = vu
        self.s = ort.InferenceSession(str(vu.MODEL), so, providers=["CPUExecutionProvider"])

    def cell_units(self, rgb):
        vu = self.vu
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (vu.SIZE, vu.SIZE)).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        logits, cxcywh = self.s.run(None, {"images": x})
        prob = 1 / (1 + np.exp(-logits[0]))
        best = {}
        for q, c in zip(*np.nonzero(prob >= vu.THRESH)):
            cx, cy, bw, bh = cxcywh[0, q]
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            fx, fy = (box[0] + box[2]) / 2, box[3] - vu.BOX_DOWN
            cell = min(CELLS, key=lambda k: (k[0] - fx) ** 2 + (k[1] - fy) ** 2)
            dd = ((cell[0] - fx) ** 2 + (cell[1] - fy) ** 2) ** .5
            sc = float(prob[q, c])
            if dd < vu.MAX_CELL_DIST and sc > best.get(cell, ("", 0, ()))[1]:
                best[cell] = (vu.CLASSES[c] if c < len(vu.CLASSES) else "hero", sc, tuple(map(float, box)))
        return best


def crop_unit(bgr, box):
    x0, y0, x1, y1 = box
    h = y1 - y0
    X0, X1 = int(round(x0 - 6)), int(round(x1 + 6))
    Y0, Y1 = int(round(y0 + BADGE_CUT * h)), int(round(y1 + 4))
    H, W = bgr.shape[:2]
    pad = cv2.copyMakeBorder(bgr, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    c = pad[Y0 + 64:Y1 + 64, X0 + 64:X1 + 64]
    return cv2.resize(c, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


def group_matches(runs_meta):
    """同一局中途重启 → 多个 live_ 目录，合并成一局（下一个目录 10 分钟内开始、且第一帧回合号 >1）。"""
    order = sorted(runs_meta, key=lambda r: r["t0"])
    gid, groups, prev = 0, {}, None
    for r in order:
        if prev and r["t0"] - prev["t1"] < 600 and (r["r0"] or 0) > 1:
            groups[r["run"]] = groups[prev["run"]]
        else:
            gid += 1
            groups[r["run"]] = f"m{gid:04d}"
        prev = r
    return groups


def split_of(match):
    h = int(hashlib.md5(match.encode()).hexdigest(), 16) % 100
    return "test" if h < 15 else "val" if h < 30 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cat, transform = load_cards()
    ev = parse_logs(cat)
    print(f"确认事件 {len(ev)} 条（可疑 {sum(e['bad'] for e in ev)}），来源 {Counter(e['src'] for e in ev)}")
    runs = sorted(d for d in LOGS.glob("live_*") if (d / "states.jsonl").exists())
    # 先粗筛：只处理时间范围内有确认事件的目录
    ev.sort(key=lambda e: e["ts"])
    ets = [e["ts"] for e in ev]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crops").mkdir(exist_ok=True)
    det = Detector()
    samples, runs_meta, stats = [], [], Counter()
    todo = runs[-args.limit:] if args.limit else runs
    for ri, d in enumerate(todo):
        recs, changes = load_run(d, cat, transform)
        if not recs:
            continue
        tss = [r["ts"] for r in recs]
        rounds = [r["round"] for r in recs if r["phase"] == "prepare" and r["round"]]
        runs_meta.append({"run": d.name, "t0": tss[0], "t1": tss[-1], "r0": rounds[0] if rounds else None})
        lo, hi = bisect_left(ets, tss[0] - 5), bisect_left(ets, tss[-1] + 5)
        rev = [e for e in ev[lo:hi] if not e["bad"]]
        stats["bad_skipped"] += sum(e["bad"] for e in ev[lo:hi])
        if not rev:
            continue
        for e in rev:                                # 确认时刻的视觉旁证
            r = nearest(recs, tss, e["ts"], before=6, after=0.5)
            vis = (r or {}).get("vis")
            if vis:
                e["conf_vis"] = ckey(e["cell"]) in vis
                e["nbr"] = any(k != ckey(e["cell"]) and ((int(k.split(',')[0]) - e["cell"][0]) ** 2 +
                               (int(k.split(',')[1]) - e["cell"][1]) ** 2) ** .5 <= NBR_DIST for k in vis)
            else:
                e["conf_vis"], e["nbr"] = None, None
        frames = []
        for p in d.glob("phase_prepare_*.png"):
            frames.append((float(p.stem.rsplit("_", 1)[1]), p, "prepare"))
        for p in d.glob("grab_shadow_pre_*.png"):
            frames.append((int(p.stem.rsplit("_", 1)[1]) / 1000.0, p, "grab_pre"))
        for tF, p, kind in sorted(frames, key=lambda t: t[0]):
            r = nearest(recs, tss, tF, before=4, after=1.5)
            if not r or r["phase"] != "prepare" or r["shop"] or r["panel"]:
                stats["frame_bad_state"] += 1
                continue
            # 候选标签
            cand = defaultdict(set)
            meta = {}
            for e in rev:
                if abs(e["ts"] - tF) > MAX_DT or e.get("conf_vis") is False:
                    continue
                c = e["cell"]
                a, b = sorted((e["ts"], tF))
                broken = any(a - 0.5 <= t <= b and mn != e["name"] for t, mn in changes.get(c, []))
                if broken:
                    continue
                cand[c].add(e["name"])
                m = meta.get(c)
                if m is None or abs(e["ts"] - tF) < abs(m["ts"] - tF):
                    meta[c] = e
            if not cand:
                continue
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (720, 1280):
                continue
            units = det.cell_units(img[:, :, ::-1])
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
                e = meta[c]
                sid = f"{d.name[5:]}_{int(tF * 1000)}_{c[0]}_{c[1]}"
                cv2.imencode(".png", crop_unit(img, u[2]))[1].tofile(str(OUT / "crops" / f"{sid}.png"))
                samples.append({"id": sid, "run": d.name, "frame": str(p.relative_to(ROOT)), "kind": kind,
                                "ts": tF, "round": r["round"], "cell": ckey(c), "name": name,
                                "box": [round(v, 1) for v in u[2]], "det_score": round(u[1], 3),
                                "level": r["lv"].get(ckey(c)), "awaken": r["awk"].get(ckey(c)),
                                "src": e["src"], "dt": round(e["ts"] - tF, 1), "nbr": e.get("nbr"),
                                "conf_vis": e.get("conf_vis")})
                stats["kept"] += 1
        print(f"[{ri + 1}/{len(todo)}] {d.name} 累计样本 {len(samples)}", flush=True)
    groups = group_matches(runs_meta)
    for s in samples:
        s["match"] = groups[s["run"]]
        s["split"] = split_of(s["match"])
    with open(OUT / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    info = {"samples": len(samples), "stats": dict(stats),
            "by_split": dict(Counter(s["split"] for s in samples)),
            "matches": len({s["match"] for s in samples}),
            "classes": dict(Counter(s["name"] for s in samples).most_common())}
    (OUT / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "classes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
