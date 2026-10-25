"""界面分类数据集：整屏截图 -> 当前是哪种界面（2026-09-30）。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_screen_cls.py              # 全量构建
  .venv/Scripts/python.exe -X utf8 tools/vision/build_screen_cls.py --limit 20   # 只跑最近 20 个对局目录（调试）

为什么：阶段识别（extractors.detect_phase / PHASE_PREDICATES / MODAL_PHASES / _detect_card_pick）靠 OCR 读关键字，
误判记录一长串（拍卖/商店被认成 N 选 1、英雄 N 选 1 被认成商店、棋手面板/英雄详情被当 prepare、unknown 其实是各种面板…）。
OCR 准确度太低，改为训练专用模型。

弱标签来源（旧 OCR 判据，**本身有错**，val/test 另行逐张人工复核，见 labels_review.jsonl）：
  phase_<阶段>_<ts>.png        阶段切换那一帧；prepare 再按同一时刻逐帧状态拆：商店开 -> prepare_shop、英雄详情开 -> hero_panel
  unknown_*.png / phase_unknown / stuck_*.png   -> unknown（待归类）
  board_inspect_* / merge_check_*               -> hero_panel（点开场上单位 = 左侧英雄详情面板）
  inspect_card_* / inspect_awake_* / inspect_no_view_* -> enlarged_detail（屏幕中间放大的卡 + 右侧「说明」）
  solo_button_*                                 -> solo_popup（进局组队弹窗）
  multi_pick_*                                  -> card_pick（N 选多）
  shop_close_fail_*                             -> prepare_shop（关店失败，商店还开着）
  grab_shadow_pre_*（每目录抽样）               -> prepare（抓手牌前一帧：商店关、棋盘无遮挡）
  equip_popup_*（每目录抽样）                   -> prepare（装备四选一弹窗挂在准备阶段棋盘上）
去重：同一目录、同一弱标签内，32x18 灰度缩略图平均差 < 2.5 的视为同一画面，只留第一张（unknown 每 8 秒存一张，大量重复）。
上限：每个 (目录, 弱标签) 最多 CAP 张（train）/ CAP_EVAL 张（val/test，要人工复核，控制工作量）。
输出 data/vision/screen_cls_v1/：thumbs.npy（N x 144 x 256 x 3，BGR uint8）+ samples.jsonl + info.json。
划分：按**对局**（同一局中途重启的多个 live_ 目录合并成一局，规则同 build_unit_id），约 70/15/15。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "data" / "logs"
OUT = ROOT / "data" / "vision" / "screen_cls_v1"
TW, TH = 256, 144
CAP, CAP_EVAL = 40, 12
DEDUP = 2.5

PREFIX_WEAK = [                     # (文件名前缀, 弱标签, 每目录最多几张 or None)
    ("board_inspect_", "hero_panel", None), ("merge_check_", "hero_panel", None),
    ("inspect_card_", "enlarged_detail", None), ("inspect_awake_", "enlarged_detail", None),
    ("inspect_no_view_", "enlarged_detail", None),
    ("solo_button_", "solo_popup", None), ("multi_pick_", "card_pick", None),
    ("shop_close_fail_", "prepare_shop", None),
    ("grab_shadow_pre_", "prepare", 4), ("equip_popup_", "prepare", 2),
    ("unknown_", "unknown", None), ("stuck_", "unknown", None),
]
RE_TS = re.compile(r"_(\d{10}(?:\d{3})?(?:\.\d+)?)\.png$")


def imread(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def file_ts(p: Path):
    m = RE_TS.search(p.name)
    if not m:
        return None
    t = float(m.group(1))
    return t / 1000.0 if t > 1e11 else t


def load_states(d: Path):
    recs = []
    for line in (d / "states.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        s = r.get("state") or {}
        recs.append({"ts": r["ts"], "phase": s.get("phase"), "shop": bool(s.get("shop_open")),
                     "panel": bool(s.get("hero_panel_open")), "round": s.get("round_no"),
                     "quota": s.get("card_pick_quota") or []})
    recs.sort(key=lambda r: r["ts"])
    return recs


def nearest(recs, tss, t, before=4.0, after=2.0):
    i = bisect_left(tss, t)
    best = None
    for j in (i - 2, i - 1, i, i + 1, i + 2):
        if 0 <= j < len(recs) and -before <= recs[j]["ts"] - t <= after:
            if best is None or abs(recs[j]["ts"] - t) < abs(best["ts"] - t):
                best = recs[j]
    return best


def group_matches(runs_meta):
    """同一局中途重启 -> 多个 live_ 目录，合并成一局（下一个目录 10 分钟内开始、且第一帧回合号 >1）。与 build_unit_id 相同。"""
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
    h = int(hashlib.md5(("screen_" + match).encode()).hexdigest(), 16) % 100
    return "test" if h < 15 else "val" if h < 30 else "train"


def weak_of_phase(phase: str, r) -> str:
    if phase == "prepare" and r:
        if r["panel"]:
            return "hero_panel"
        if r["shop"]:
            return "prepare_shop"
    return phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(0)
    runs = sorted(d for d in LOGS.glob("live_*") if d.is_dir() and (d / "states.jsonl").exists())
    todo = runs[-args.limit:] if args.limit else runs
    runs_meta, cands = [], []
    for d in todo:
        recs = load_states(d)
        if not recs:
            continue
        tss = [r["ts"] for r in recs]
        rounds = [r["round"] for r in recs if r["phase"] == "prepare" and r["round"]]
        runs_meta.append({"run": d.name, "t0": tss[0], "t1": tss[-1], "r0": rounds[0] if rounds else None})
        per = defaultdict(list)
        for p in d.glob("*.png"):
            t = file_ts(p)
            if t is None:
                continue
            weak, src = None, None
            if p.name.startswith("phase_"):
                ph = p.name[6:].rsplit("_", 1)[0]
                r = nearest(recs, tss, t)
                weak, src = weak_of_phase(ph, r), "phase_" + ph
                if ph == "card_pick" and r and r["quota"]:
                    src += "_multi"
            else:
                for pre, w, lim in PREFIX_WEAK:
                    if p.name.startswith(pre):
                        weak, src = w, pre.rstrip("_")
                        break
            if weak is None:
                continue
            per[src].append((t, p, weak))
        for src, lst in per.items():
            lim = next((l for pre, w, l in PREFIX_WEAK if pre.rstrip("_") == src), None)
            lst.sort(key=lambda x: x[0])
            if lim and len(lst) > lim:
                lst = sorted(rng.sample(lst, lim), key=lambda x: x[0])
            for t, p, weak in lst:
                cands.append({"run": d.name, "ts": t, "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                              "src": src, "weak": weak})
    groups = group_matches(runs_meta)
    for c in cands:
        c["match"] = groups[c["run"]]
        c["split"] = split_of(c["match"])
    print(f"候选 {len(cands)} 张，来源 {Counter(c['src'] for c in cands).most_common()}", flush=True)
    # 读图 + 去重 + 上限
    OUT.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for c in cands:
        by[(c["run"], c["weak"])].append(c)
    samples, thumbs, stats = [], [], Counter()
    for gi, ((run, weak), lst) in enumerate(sorted(by.items())):
        lst.sort(key=lambda c: c["ts"])
        cap = CAP if lst[0]["split"] == "train" else CAP_EVAL
        if len(lst) > cap * 3:                        # 先抽样再读图，省时间
            lst = sorted(rng.sample(lst, cap * 3), key=lambda c: c["ts"])
        kept_small = []
        kept = []
        for c in lst:
            im = imread(ROOT / c["path"])
            if im is None or im.shape[:2] != (720, 1280):
                stats["bad_img"] += 1
                continue
            small = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
            if any(np.abs(small - k).mean() < DEDUP for k in kept_small):
                stats["dup"] += 1
                continue
            kept_small.append(small)
            kept.append((c, cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA)))
        if len(kept) > cap:
            stats["capped"] += len(kept) - cap
            kept = [kept[i] for i in sorted(rng.sample(range(len(kept)), cap))]
        for c, th in kept:
            c["id"] = f"{run[5:]}_{int(c['ts'] * 1000)}_{len(samples)}"
            c["idx"] = len(samples)
            samples.append(c)
            thumbs.append(th)
        if gi % 200 == 0:
            print(f"[{gi}/{len(by)}] 样本 {len(samples)}", flush=True)
    np.save(OUT / "thumbs.npy", np.stack(thumbs))
    with open(OUT / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    info = {"samples": len(samples), "stats": dict(stats),
            "by_split": dict(Counter(s["split"] for s in samples)),
            "matches": len({s["match"] for s in samples}),
            "by_weak": {sp: dict(Counter(s["weak"] for s in samples if s["split"] == sp).most_common())
                        for sp in ("train", "val", "test")}}
    (OUT / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
