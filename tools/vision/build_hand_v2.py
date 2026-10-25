"""手牌数据池 v2：覆盖全部历史对局（v1 只到 live_1790313858，189 局），同时产出两个池（同一套 id、同一张牌）：

  CUDA_VISIBLE_DEVICES= WZWXQ_OCR_EP=cpu .venv/Scripts/python.exe -X utf8 tools/vision/build_hand_v2.py [--workers 6]

data/vision/hand_awaken_v2/crops/<id>.png   左边一截 64x120（BGR，与 v1 / 线上 HandArtReader 同一位置）
data/vision/hand_awaken_v2/pool.jsonl       id/帧/对局/划分/第几张/张数/x/觉醒模型概率(hand_awaken_v1)/v1 人工标签(有就带)
data/vision/hand_names_v2/pool.jsonl        id/…/名字条 ROI/OCR 原文/标签/标签来源

与 v1 的区别：
- 布局用线上现在的 E.hand_layout（手牌条模型优先，图标推算兜底）。
- 只对**满宽（≥110px，不叠放）**名字条打标签（立绘训练只用这些）；叠放的只存裁图。
- 标签：① 服务器版认字 + 闭集解码（HandNameReader，线上同款）认定 → 用；
  ② 否则自由解码 OCR + `_resolve_hand_fragment`，且 `hand_hint` = 这一局账本里拿到过的牌（买牌/获得卡牌/N选1）——
     "芈"认不出、只读到「月」时，账本里只有芈月以「月」结尾 → 芈月（与线上 2026-10-11 的修法同一条规则）；有前缀时结果必须以前缀开头。
- 划分：v1 的 189 局沿用 v1 的哈希划分（v1 test 局仍在 test，可同集对比）；之后的新局：最近 RECENT_TEST 局整局进 test_new，
  其余按哈希 90% train / 10% val。正在写的对局（目录 30 分钟内有改动）跳过。
每局结果先写 parts/<run>.jsonl（可断点续跑），最后合并。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("WZWXQ_OCR_EP", "cpu")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_N = ROOT / "data" / "vision" / "hand_names_v2"
OUT_A = ROOT / "data" / "vision" / "hand_awaken_v2"
V1_LAST = "live_1790313858"
RECENT_TEST = 20
STRIDE = 3
W, H, Y0 = 64, 120, 595


def split_v1(run: str) -> str:
    h = int(hashlib.md5(run.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def split_new(run: str) -> str:
    h = int(hashlib.md5(("v2" + run).encode()).hexdigest(), 16) % 100
    return "train" if h < 90 else "val"


def ledger_names(run_dir: Path) -> set:
    out = set()
    f = run_dir / "ledger.jsonl"
    if not f.exists():
        return out
    for l in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            d = json.loads(l)
        except Exception:                                   # noqa: BLE001
            continue
        if d.get("event") in ("买牌", "获得卡牌", "N选1") and isinstance(d.get("牌"), str):
            out.add(d["牌"])
    return out


_W = {}


def _init():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL
    except Exception:                                       # noqa: BLE001
        pass
    import onnxruntime as ort
    from agent.perception import extractors as E
    from agent.perception.hand_names import HandNameReader
    _W["E"] = E
    _W["ex"] = E.StateExtractor(ROOT / "assets" / "image", cards_db_path=ROOT / "data" / "cards.json")
    _W["reader"] = HandNameReader.get()
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    _W["awk"] = ort.InferenceSession(str(ROOT / "assets/model/vision/hand_awaken_v1.onnx"), so, providers=["CPUExecutionProvider"])


def do_run(run: str, split: str, v1lab: dict) -> tuple[str, int]:
    import cv2
    import numpy as np
    if not _W:
        _init()
    E, ex, reader, awk = _W["E"], _W["ex"], _W["reader"], _W["awk"]
    rd = ROOT / "data" / "logs" / run
    hint = {n for n in ledger_names(rd) if ex._card_cats.get(n) in ("英雄牌", "效果牌", "装备牌", "天赋牌", "棋手技能牌")}
    ex.hand_hint = set(hint)
    ex._hand_seen.clear()
    rows = []
    cache: dict = {}
    frames = sorted(rd.glob("phase_prepare_*.png"))
    if split in ("train", "val") and run > V1_LAST:
        frames = frames[::STRIDE]          # 新局的相邻准备帧高度重复：训练/验证局每 STRIDE 帧取 1（测试局全取）
    for f in frames:
        bgr = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        if bgr.shape[:2] != (720, 1280):
            bgr = cv2.resize(bgr, (1280, 720))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            lay = E.hand_layout(rgb)
        except Exception:                                   # noqa: BLE001
            continue
        if not lay or not lay.get("xs"):
            continue
        xs, n = lay["xs"], lay["n"]
        name_y = lay["icon_y"] + E.HAND_NAME_DY
        arts, keep = [], []
        for k, ix in enumerate(xs):
            x0 = int(ix) - 12
            if x0 < 0 or x0 + W > 1280:
                continue
            arts.append(rgb[Y0:Y0 + H, x0:x0 + W]); keep.append(k)
        if not arts:
            continue
        xin = np.ascontiguousarray(np.stack([a[4:116, 4:60] for a in arts]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
        lg = awk.run(None, {"x": xin})[0]
        e = np.exp(lg - lg.max(1, keepdims=True)); pa = (e / e.sum(1, keepdims=True))[:, 1]
        stem = f.stem.split("_")[-1]
        for a, k, p in zip(arts, keep, pa):
            ix = xs[k]
            left = ix - E.HAND_ICON_TO_LEFT
            right = (xs[k + 1] - E.HAND_ICON_TO_LEFT) if k + 1 < n else left + E.HAND_PITCH_MAX
            roi = (int(left) + 3, name_y - 15, max(8, int(right - left) - 6), 30)
            cid = f"{run[5:]}_{stem}_{k}"
            cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))[1].tofile(str(OUT_A / "crops" / f"{cid}.png"))
            t, lab, src = "", None, ""
            x0r, y0r, wr, hr = roi
            if wr >= 110 and x0r >= 0 and x0r + wr <= 1280 and y0r + hr <= 720:
                crop = np.ascontiguousarray(bgr[y0r:y0r + hr, x0r:x0r + wr])
                key = hashlib.md5(crop.tobytes()).hexdigest()
                if key in cache:
                    t, lab, src = cache[key]
                else:
                    name, prefix = "", ""
                    if reader is not None:
                        name, prefix, _m = reader.read(crop, wr)
                    if name:
                        lab, src = name, "closed"
                    t = E._ocr(None, rgb, roi) or ""
                    if not lab:
                        r = ex._resolve_hand_fragment(prefix or t) if (prefix or t) else None
                        if r and (not prefix or r.startswith(prefix)):
                            lab, src = r, "ocr+ledger"
                    cache[key] = (t, lab, src)
            rows.append({"id": cid, "frame": str(f.relative_to(ROOT)), "run": run, "split": split, "k": k, "n": n,
                         "x": int(ix), "roi": list(roi), "ocr": t, "label": lab, "src": src,
                         "p_awk": round(float(p), 4), "awk_v1": v1lab.get(cid)})
    (OUT_N / "parts" / f"{run}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return run, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    for d in (OUT_N / "parts", OUT_A / "crops"):
        d.mkdir(parents=True, exist_ok=True)
    now = time.time()
    runs = sorted(p.name for p in (ROOT / "data" / "logs").glob("live_17*") if p.is_dir()
                  and any(p.glob("phase_prepare_*.png")) and now - p.stat().st_mtime > 1800)
    new = [r for r in runs if r > V1_LAST]
    test_new = set(new[-RECENT_TEST:])
    v1lab = {}
    for l in (ROOT / "data/vision/hand_awaken_v1/labels.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(l); v1lab[d["id"]] = d["label"]
    plan = []
    for r in runs:
        sp = split_v1(r) if r <= V1_LAST else ("test_new" if r in test_new else split_new(r))
        if not (OUT_N / "parts" / f"{r}.jsonl").exists():
            plan.append((r, sp))
    if args.limit:
        plan = plan[:args.limit]
    print(f"对局 {len(runs)}（v1 范围 {len(runs) - len(new)}，新 {len(new)}，test_new {len(test_new)}），待处理 {len(plan)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        futs = [pool.submit(do_run, r, sp, {k: v for k, v in v1lab.items() if k.startswith(r[5:] + "_")}) for r, sp in plan]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r, c = fu.result()
            except Exception as e:                          # noqa: BLE001
                print("失败", e, flush=True); continue
            if i % 10 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} 局  {time.time() - t0:.0f}s  最近 {r} {c} 条", flush=True)
    rows = []
    for r in runs:
        p = OUT_N / "parts" / f"{r}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]
    (OUT_N / "pool.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (OUT_A / "pool.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("id", "frame", "run", "split", "k", "n", "x", "p_awk", "awk_v1")},
                                                         ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    lab = [r for r in rows if r["label"]]
    print(f"裁图 {len(rows)}，满宽 {sum(r['roi'][2] >= 110 for r in rows)}，有标签 {len(lab)}，不同卡名 {len({r['label'] for r in lab})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
