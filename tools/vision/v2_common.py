"""视觉小模型 v2 加训的公共工具：读 build_frames_v2 的缓存、读逐帧状态序列、对局分组与划分（2026-10-12）。"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "data" / "logs"
V2 = ROOT / "data" / "vision_v2"
FRAMES = V2 / "frames"


def frame_runs() -> list[str]:
    return sorted(p.stem for p in FRAMES.glob("live_*.json"))


def load_frames(run: str):
    meta = json.loads((FRAMES / f"{run}.json").read_text(encoding="utf-8"))
    z = None
    if meta:
        with np.load(FRAMES / f"{run}.npz") as f:      # 一次读进内存（NpzFile 每次 [key] 都会重新解压整个数组）
            z = {k: f[k] for k in f.files}
    return meta, z


def load_seq(run: str):
    """该对局所有逐帧状态（不限阶段），按时间排序：[{ts, phase, shop, panel, round, lv, eq, names, vis, board, pop, pop_ok, hand, actions}]"""
    recs = []
    p = LOGS / run / "states.jsonl"
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        s = r.get("state") or {}
        recs.append({"ts": r["ts"], "phase": s.get("phase"), "shop": s.get("shop_open"),
                     "panel": s.get("hero_panel_open"), "round": s.get("round_no"),
                     "lv": s.get("unit_levels") or {}, "eq": s.get("equip_counts") or {},
                     "names": s.get("unit_names") or {}, "vis": s.get("vision_units") or {},
                     "anchors": s.get("unit_anchors") or {},
                     "pop": s.get("population"), "pop_ok": s.get("population_valid"),
                     "board": r.get("board"), "hand": s.get("hand") or [],
                     "actions": [a for a in (r.get("actions") or []) if isinstance(a, dict)],
                     "executed": r.get("executed")})
    recs.sort(key=lambda r: r["ts"])
    return recs


def group_matches(runs_meta):
    """与 build_unit_id.group_matches 相同：同一局中途重启（下一个目录 10 分钟内开始、首帧回合号 >1）合并成一局。"""
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


def split_of(match: str) -> str:
    h = int(hashlib.md5(match.encode()).hexdigest(), 16) % 100
    return "test" if h < 15 else "val" if h < 30 else "train"


@lru_cache(maxsize=1)
def match_map() -> dict:
    """run -> (match, split)。缓存到 data/vision_v2/matches.json。"""
    f = V2 / "matches.json"
    if f.exists():
        return {k: tuple(v) for k, v in json.loads(f.read_text(encoding="utf-8")).items()}
    meta = []
    for d in sorted(LOGS.glob("live_*")):
        p = d / "states.jsonl"
        if not p.exists():
            continue
        ts, r0 = [], None
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            ts.append(r["ts"])
            s = r.get("state") or {}
            if r0 is None and s.get("phase") == "prepare" and s.get("round_no"):
                r0 = s.get("round_no")
        if ts:
            meta.append({"run": d.name, "t0": min(ts), "t1": max(ts), "r0": r0})
    g = group_matches(meta)
    out = {run: (m, split_of(m)) for run, m in g.items()}
    f.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    return out
