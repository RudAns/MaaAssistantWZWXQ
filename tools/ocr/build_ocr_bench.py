"""从历史对局截图切出"卡名"小图，配上正确答案，做成 OCR 测试集 / 微调数据。

正确答案不靠大模型，靠卡牌库 + 游戏结构推出来：
  talent   天赋三选一的卡名（金底/橙金底，最难认）。答案 = 用**效果描述**在卡牌库里反查的卡名
           （描述是整段文字，比卡名好认得多；且与卡名 OCR 独立）。
  shop     商店四个槽位的卡名。答案 = 这一帧认出的卡名；这一帧没认出时，用**同一回合、相邻几秒内、
           其余槽位都一样**的帧里这个槽位认出的卡名（与本帧 OCR 独立 —— 这正是现在的难例）。
  pick     N 选 1 的卡名。答案 = 当时认出并在卡牌库里的卡名。
来源字段 `src` 标明答案是怎么来的（*_self = 与现有模型的读数同源，有偏；desc / neighbor = 独立来源）。

输出：data/ocr_bench/<ver>/crops/*.png + labels.jsonl（file, gt, kind, src, run, ts, roi, split）
按对局目录切分 test / train（同一局的图不会同时出现在两边）。
  python tools/ocr/build_ocr_bench.py --ver v1
"""
from __future__ import annotations

import argparse
import bisect
import collections
import difflib
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from agent.perception.extractors import (ROI_TALENT_NAME_Y, SHOP_NAME_CELLS,  # noqa: E402
                                         TALENT_CARD_X)

SHOP_CELLS = SHOP_NAME_CELLS                       # [(x, y, w, h)] × 4
TALENT_CELLS = [(x0 + 20, ROI_TALENT_NAME_Y[0], 200, ROI_TALENT_NAME_Y[1]) for x0 in TALENT_CARD_X]


def set_talent_roi(y: int, h: int) -> None:
    """实验用：换天赋卡名的切图区域（线上 ROI_TALENT_NAME_Y 偏低，卡名上半截被切掉）。"""
    global TALENT_CELLS
    TALENT_CELLS = [(x0 + 20, y, 200, h) for x0 in TALENT_CARD_X]


def _imread(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), 1)


def _imwrite(p: Path, img) -> None:
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(p))


def load_db():
    cards = json.loads((PROJECT_DIR / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    descs = {}
    for n, c in cards.items():
        d = re.sub(r"\s+", "", c.get("skill") or "")
        if len(d) >= 6:
            descs[n] = d
    return cards, descs


def desc_to_name(desc: str, descs: dict, cands: list) -> str | None:
    """效果描述 -> 卡名：最像的要足够像（≥0.75），且明显好过第二名（差 ≥0.08）。"""
    d = re.sub(r"\s+", "", desc or "")
    if len(d) < 6:
        return None
    scored = sorted(((difflib.SequenceMatcher(None, d, descs[n]).ratio(), n) for n in cands), reverse=True)[:2]
    if not scored or scored[0][0] < 0.75:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.08:
        return None
    return scored[0][1]


def load_states(run_dir: Path):
    recs = []
    for line in (run_dir / "states.jsonl").open(encoding="utf-8"):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if x.get("ts") and x.get("state"):
            recs.append((x["ts"], x["state"]))
    recs.sort(key=lambda r: r[0])
    return recs


def nearest(recs, ts, phase, tol=1.5):
    i = bisect.bisect_left([r[0] for r in recs], ts)
    best = None
    for j in range(max(0, i - 4), min(len(recs), i + 4)):
        t, s = recs[j]
        if s.get("phase") == phase and abs(t - ts) <= tol and (best is None or abs(t - ts) < abs(best[0] - ts)):
            best = (t, s)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--talent-roi", default=None, help="y,h：天赋卡名切图区域（默认 = 线上的 ROI_TALENT_NAME_Y）")
    args = ap.parse_args()
    if args.talent_roi:
        set_talent_roi(*map(int, args.talent_roi.split(",")))
    out = PROJECT_DIR / "data" / "ocr_bench" / args.ver
    (out / "crops").mkdir(parents=True, exist_ok=True)
    cards, descs = load_db()
    talent_names = [n for n, c in cards.items() if c.get("category") in ("天赋牌", "效果牌") and n in descs]
    rows, stat = [], collections.Counter()
    runs = sorted(p for p in (PROJECT_DIR / "data" / "logs").glob("live_*") if (p / "states.jsonl").exists())
    for rd in runs:
        recs = None
        for png in sorted(rd.glob("phase_*.png")):
            m = re.match(r"phase_([a-z_]+)_(\d+)\.png$", png.name)
            if not m or m.group(1) not in ("talent", "prepare", "card_pick"):
                continue
            if recs is None:
                recs = load_states(rd)
                times = [r[0] for r in recs]
            phase, ts = m.group(1), int(m.group(2))
            hit = nearest(recs, ts, phase)
            if not hit:
                stat["no_state"] += 1
                continue
            t0, s = hit
            img = None

            def add(kind, roi, gt, src):
                nonlocal img
                if img is None:
                    img = _imread(png)
                x, y, w, h = roi
                crop = img[max(0, y):y + h, max(0, x):x + w]
                if crop.size == 0:
                    return
                fid = hashlib.md5(f"{png}|{roi}".encode()).hexdigest()[:12]
                _imwrite(out / "crops" / f"{fid}.png", crop)
                rows.append({"file": f"crops/{fid}.png", "gt": gt, "kind": kind, "src": src, "run": rd.name,
                             "ts": ts, "roi": list(roi)})
                stat[f"{kind}/{src}"] += 1

            if phase == "talent" and s.get("talents"):
                for t in s["talents"]:
                    col = int(np.argmin([abs((t.get("x") or -999) - (cx + 20 + 100)) for cx in TALENT_CARD_X])) \
                        if (t.get("x") or -1) > 0 else t.get("index", -1)
                    if not 0 <= col < 3:
                        continue
                    gt = desc_to_name(t.get("desc", ""), descs, talent_names)
                    if gt:
                        add("talent", TALENT_CELLS[col], gt, "desc")
                    else:
                        stat["talent_no_gt"] += 1
            elif phase == "card_pick" and s.get("card_options"):
                for o in s["card_options"]:
                    if o.get("name") in cards and o.get("x", -1) > 0:
                        add("pick", (int(o["x"]) - 100, int(o["y"]) - 22, 200, 44), o["name"], "self")
            elif phase == "prepare" and s.get("shop_open") and s.get("shop"):
                slots = {sl["index"]: sl for sl in s["shop"]}
                for k in range(4):
                    sl = slots.get(k)
                    if not sl or sl.get("sold"):
                        continue
                    if sl.get("piece") in cards:
                        add("shop", SHOP_CELLS[k], sl["piece"], "self")
                        continue
                    # 本帧没认出：同回合、±8 秒内、其余已认出的槽位一致的帧里，这个槽位认出了什么
                    i = bisect.bisect_left(times, t0)
                    votes = collections.Counter()
                    for j in range(max(0, i - 40), min(len(recs), i + 40)):
                        t2, s2 = recs[j]
                        if abs(t2 - t0) > 8 or s2.get("round_no") != s.get("round_no") or not s2.get("shop"):
                            continue
                        o2 = {x["index"]: x for x in s2["shop"]}
                        same = all(o2.get(q, {}).get("piece") == slots[q].get("piece")
                                   for q in slots if q != k and slots[q].get("piece"))
                        p2 = o2.get(k, {}).get("piece")
                        if same and p2 in cards:
                            votes[p2] += 1
                    if votes and votes.most_common(1)[0][1] >= 2 and len(votes) == 1:
                        add("shop", SHOP_CELLS[k], votes.most_common(1)[0][0], "neighbor")
                    else:
                        stat["shop_no_gt"] += 1
    # 按对局目录切分
    rng = random.Random(0)
    run_names = sorted({r["run"] for r in rows})
    rng.shuffle(run_names)
    test_runs = set(run_names[:int(len(run_names) * args.test_frac)])
    for r in rows:
        r["split"] = "test" if r["run"] in test_runs else "train"
    with (out / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"n": len(rows), "by": dict(stat),
               "split": dict(collections.Counter((r["split"], r["kind"], r["src"]) for r in rows).most_common()),
               "distinct_gt": len({r["gt"] for r in rows})}
    summary["split"] = {"/".join(k): v for k, v in summary["split"].items()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
