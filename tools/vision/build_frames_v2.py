"""视觉小模型 v2 加训的公共第一步：把历史对局里"准备阶段、商店关着"的整屏截图过一遍检测，缓存裁图（2026-10-12）。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_frames_v2.py [--limit N]

输入：data/logs/live_*/phase_prepare_*.png（进入准备阶段那一帧）+ grab_shadow_pre_*.png（抓手牌前一帧）。
对每一帧：
  ① 找时间上最近的逐帧状态（与 build_unit_id.py 同一规则：状态 ts - 帧时刻 ∈ [-4, 1.5] 秒），要求 prepare、商店关、没开英雄详情；
  ② 跑和线上同一套的 D-FINE（vision_units.cell_units）→ 己方每格 (类别, 分数, 框)；
  ③ 跑和线上同一套的等级徽章检测（extractors._detect_badges）→ 每格徽章中心；
  ④ 存裁图（BGR）：徽章 64x64（等级 / 觉醒用）、装备条 104x60（徽章下方 bx±52, by~by+60，覆盖规则读法的 bx±45, by+10~42）、
     单位 80x128（与 unit_id 同一裁法）。
输出 data/vision_v2/frames/<run>.npz + <run>.json（逐帧元数据：状态里的等级 / 装备 / 认人 / 账本 board / 人口）。
只读日志；cv2 限 4 线程；D-FINE 走线上同一个会话（TensorRT/CPU 自动）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(4)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.perception import vision_units as vu                       # noqa: E402
from agent.perception.extractors import OWN_BOARD_ROWS, _detect_badges  # noqa: E402

LOGS = ROOT / "data" / "logs"
OUT = ROOT / "data" / "vision_v2" / "frames"
CELLS = [tuple(c) for r in OWN_BOARD_ROWS for c in r]
EQ_W, EQ_H, EQ_DY0 = 104, 60, 0


def ckey(c):
    return f"{c[0]},{c[1]}"


def crop_pad(img, x0, y0, w, h):
    pad = cv2.copyMakeBorder(img, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return pad[y0 + 64:y0 + 64 + h, x0 + 64:x0 + 64 + w].copy()


def crop_unit(bgr, box):
    x0, y0, x1, y1 = box
    h = y1 - y0
    X0, X1 = int(round(x0 - 6)), int(round(x1 + 6))
    Y0, Y1 = int(round(y0 + 0.15 * h)), int(round(y1 + 4))
    pad = cv2.copyMakeBorder(bgr, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    c = pad[Y0 + 64:Y1 + 64, X0 + 64:X1 + 64]
    return cv2.resize(c, (80, 128), interpolation=cv2.INTER_AREA)


def load_states(d: Path):
    recs = []
    for line in (d / "states.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        s = r.get("state") or {}
        recs.append({"ts": r["ts"], "phase": s.get("phase"), "shop": s.get("shop_open"),
                     "panel": s.get("hero_panel_open"), "round": s.get("round_no"),
                     "lv": s.get("unit_levels") or {}, "eq": s.get("equip_counts") or {},
                     "names": s.get("unit_names") or {}, "vis": s.get("vision_units") or {},
                     "pop": s.get("population"), "pop_ok": s.get("population_valid"),
                     "board": r.get("board"), "hand": s.get("hand") or []})
    recs.sort(key=lambda r: r["ts"])
    return recs


def nearest(recs, tss, t, before=4.0, after=1.5):
    i = bisect_left(tss, t)
    best = None
    for j in (i - 2, i - 1, i, i + 1):
        if 0 <= j < len(recs) and -before <= recs[j]["ts"] - t <= after:
            if best is None or abs(recs[j]["ts"] - t) < abs(best["ts"] - t):
                best = recs[j]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(d for d in LOGS.glob("live_*") if (d / "states.jsonl").exists())
    if args.limit:
        runs = runs[-args.limit:]
    t0 = time.time()
    tot = 0
    for ri, d in enumerate(runs):
        if (OUT / f"{d.name}.json").exists():
            continue
        frames = [(float(p.stem.rsplit("_", 1)[1]), p, "prepare") for p in d.glob("phase_prepare_*.png")]
        frames += [(int(p.stem.rsplit("_", 1)[1]) / 1000.0, p, "grab_pre") for p in d.glob("grab_shadow_pre_*.png")]
        meta, badges_c, eq_c, unit_c = [], [], [], []
        if frames:
            recs = load_states(d)
            tss = [r["ts"] for r in recs]
            for tF, p, kind in sorted(frames, key=lambda t: t[0]):
                r = nearest(recs, tss, tF)
                if not r or r["phase"] != "prepare" or r["shop"] or r["panel"]:
                    continue
                bgr = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
                if bgr is None or bgr.shape[:2] != (720, 1280):
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # 线上截图是 RGB
                units = vu.cell_units(rgb, CELLS)
                badges = _detect_badges(rgb)
                fm = {"run": d.name, "frame": str(p.relative_to(ROOT)), "kind": kind, "t": tF, "sts": r["ts"],
                      "round": r["round"], "lv": r["lv"], "eq": r["eq"], "names": r["names"], "vis": r["vis"],
                      "pop": r["pop"], "pop_ok": r["pop_ok"], "board": r["board"], "hand": r["hand"],
                      "badges": {}, "units": {}}
                for k, (bx, by) in badges.items():
                    fm["badges"][k] = [int(bx), int(by), len(badges_c)]
                    badges_c.append(crop_pad(bgr, bx - 32, by - 32, 64, 64))
                    eq_c.append(crop_pad(bgr, bx - EQ_W // 2, by + EQ_DY0, EQ_W, EQ_H))
                for c, (cls, sc, box) in units.items():
                    fm["units"][ckey(c)] = [cls, round(float(sc), 3), [round(float(v), 1) for v in box], len(unit_c)]
                    unit_c.append(crop_unit(bgr, box))
                meta.append(fm)
        if meta:
            np.savez_compressed(OUT / f"{d.name}.npz",
                                badge=np.stack(badges_c) if badges_c else np.zeros((0, 64, 64, 3), np.uint8),
                                equip=np.stack(eq_c) if eq_c else np.zeros((0, EQ_H, EQ_W, 3), np.uint8),
                                unit=np.stack(unit_c) if unit_c else np.zeros((0, 128, 80, 3), np.uint8))
        (OUT / f"{d.name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tot += len(meta)
        if ri % 20 == 0:
            print(f"[{ri + 1}/{len(runs)}] {d.name} 帧 {len(meta)} 累计 {tot}  {time.time() - t0:.0f}s", flush=True)
    print("完成", tot)


if __name__ == "__main__":
    main()
