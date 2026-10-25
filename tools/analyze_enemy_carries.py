"""对手高等级主C 是谁（2026-10-08）。

  python tools/analyze_enemy_carries.py extract      # 全部 live_* 的战斗截图：认血条/等级、裁单位图（约 1 小时，CPU 低优先级）
  python tools/analyze_enemy_carries.py train        # 用"我方单位战斗裁图"微调认人模型（标签 = 等级对上准备帧账本）
  python tools/analyze_enemy_carries.py apply        # 认第 396~415 局、第 10 回合以后每张图里对手最高等级单位，写 jsonl

为什么要微调：unit_id_v1 是用**准备阶段**静止的己方单位训练的；战斗帧里单位在动、放技能、互相遮挡，
原模型直接用几乎认不出（见报告第 2 节的对照数字）。

血条：蓝=对手、绿=我方（同 tools/analyze_enemy_levels.py）；血条上方 64px 宽的小 ROI 认等级（容三位数）。
单位裁图：按血条位置合成框 (cx-44, y-30, cx+44, y+120)，再走 unit_id._crop 的同一套裁法 → 80x128。
标签（只有我方有）：本回合最近一个准备帧 states.jsonl 的 board（名字→等级），OCR 等级唯一对上一个名字才算。
只用 CPU（WZWXQ_OCR_EP=cpu），进程设为低优先级，不和实战抢资源。
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
from bisect import bisect_right
from collections import Counter

os.environ["WZWXQ_OCR_EP"] = "cpu"
# 认字用轻量版：服务器版在 CPU 上一张小图 ~1 秒（实战局同时在跑），轻量版一帧一批 ~60ms；数字够用（报告里人工抽查）
os.environ.setdefault("WZWXQ_OCR_REC_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "model", "ocr"))
os.environ["WZWXQ_VISION_EP"] = "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if os.name == "nt":   # 低优先级（BELOW_NORMAL），别和正在跑的对局抢 CPU
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:  # noqa: BLE001
        pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SIDES = (("enemy", (105, 80, 60), (140, 255, 255)), ("ours", (35, 80, 80), (85, 255, 255)))
OUT = os.path.join(ROOT, "data", "analysis", "enemy_carries")
CROP_W, CROP_H, IN_W, IN_H = 80, 128, 72, 120
MIN_MATCH = 396          # 2026-10-08：只看最近 20 局（第 396~415 局）
TEST_EVERY = 4           # 训练/评估按对局切：match % 4 == 3 的局做测试（5 局）


def bars(img_bgr, rgb, ocr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    cand = []
    for side, lo, hi in SIDES:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            x, y, w, h = cv2.boundingRect(c)
            if not (40 <= w <= 100 and 3 <= h <= 12 and 35 < y < 600 and 300 <= x <= 1080):
                continue
            cand.append((side, x, y, w, h, x + w // 2))
    # 只认字（不找字）、一帧一批：等级数字就在血条正上方一行，窄 ROI 不需要检测
    crops = [img_bgr[max(0, y - 20):y - 1, max(0, cx - 32):cx + 32] for _, x, y, w, h, cx in cand]
    texts = ocr.recognize(crops) if crops else []
    out = []
    for (side, x, y, w, h, cx), t in zip(cand, texts):
        t = t.strip()
        if re.fullmatch(r"\d{1,3}", t):
            out.append({"side": side, "lv": int(t), "raw": t, "bar": [x, y, w, h], "cx": cx})
    return out


def crop(rgb, u):
    """→ 80x128 RGB（与 unit_id._crop 同一裁法，只是不做中心裁，留给增强）。"""
    x, y = u["cx"], u["bar"][1]
    x0, y0, x1, y1 = x - 44, y - 30, x + 44, y + 120
    h = y1 - y0
    X0, X1 = int(round(x0 - 6)), int(round(x1 + 6))
    Y0, Y1 = int(round(y0 + 0.15 * h)), int(round(y1 + 4))
    pad = cv2.copyMakeBorder(rgb, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    c = pad[Y0 + 64:Y1 + 64, X0 + 64:X1 + 64]
    return cv2.resize(c, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


def load_states(run_dir):
    f = os.path.join(run_dir, "states.jsonl")
    if not os.path.exists(f):
        return None
    ts, rd, boards = [], [], []
    for line in open(f, encoding="utf-8"):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        s = x.get("state") or {}
        r = s.get("round_no")
        if not r:
            continue
        ts.append(float(x.get("ts") or 0))
        rd.append(r)
        b = x.get("board") or {}
        boards.append((s.get("phase"), {k: v.get("lv") for k, v in b.items() if isinstance(v, dict)}, x.get("carry")))
    return ts, rd, boards


def match_table():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/laya/iterations/batch2_matches.jsonl"), encoding="utf-8")]
    return {r["run"]: r for r in rows}


def extract():
    from agent.perception.onnx_ocr import OnnxOcr
    ocr = OnnxOcr.get()
    mt = match_table()
    os.makedirs(OUT, exist_ok=True)
    meta = open(os.path.join(OUT, "units.jsonl"), "w", encoding="utf-8")
    crops = []
    nfr = 0
    for d in sorted(glob.glob(os.path.join(ROOT, "data", "logs", "live_*"))):
        run = os.path.basename(d)
        fr = sorted(glob.glob(os.path.join(d, "phase_battle_*.png")))
        if not fr:
            continue
        st = load_states(d)
        if not st or not st[0]:
            continue
        ts, rd, boards = st
        m = mt.get(run) or {}
        if (m.get("match") or 0) < MIN_MATCH:
            continue
        for p in fr:
            mm = re.search(r"_(\d{10})", os.path.basename(p))
            i = bisect_right(ts, float(mm.group(1))) - 1 if mm else -1
            if i < 0:
                continue
            r = rd[i]
            ob, carry = {}, None
            for j in range(i, -1, -1):
                if rd[j] != r:
                    break
                if boards[j][0] == "prepare" and boards[j][1]:
                    ob, carry = boards[j][1], boards[j][2]
                    break
            img = cv2.imdecode(np.fromfile(p, np.uint8), 1)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            us = bars(img, rgb, ocr)
            nfr += 1
            lvcnt = Counter(ob.values())
            enemy = sorted([u for u in us if u["side"] == "enemy"], key=lambda u: -u["lv"])
            ours = [u for u in us if u["side"] == "ours"]
            fmeta = {"run": run, "match": m.get("match"), "rank": m.get("rank"), "ban": m.get("ban"), "img": os.path.basename(p),
                     "round": r, "carry": carry, "carry_lv": ob.get(carry) if carry else None, "our_board": ob,
                     "enemy_lvs": [u["lv"] for u in enemy], "ours_lvs": sorted([u["lv"] for u in ours], reverse=True)}
            keep = []
            for k, u in enumerate(enemy[:3]):          # 对手前三高等级都裁（报告只用第一个，其余备查）
                keep.append(dict(u, rank_in_side=k))
            for u in ours:
                names = [n for n, lv in ob.items() if lv == u["lv"]]
                if len(names) == 1 and lvcnt[u["lv"]] == 1:
                    keep.append(dict(u, label=names[0]))
                elif ours and u["lv"] == max(x["lv"] for x in ours):
                    keep.append(dict(u, label=None, ours_top=True))
            for u in keep:
                u["cid"] = len(crops)
                crops.append(crop(rgb, u))
                meta.write(json.dumps({**fmeta, **u}, ensure_ascii=False) + "\n")
            if nfr % 25 == 0:
                print(nfr, len(crops), flush=True)
    meta.close()
    np.save(os.path.join(OUT, "crops.npy"), np.stack(crops))
    print("frames", nfr, "crops", len(crops))


# ---------------------------------------------------------------- 训练（微调 unit_id_v1，CPU）
def _load_units():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "units.jsonl"), encoding="utf-8")]
    crops = np.load(os.path.join(OUT, "crops.npy"), mmap_mode="r")
    return rows, crops


def _classes():
    meta = json.load(open(os.path.join(ROOT, "data/vision/unit_id_v1/classes.json"), encoding="utf-8"))
    return meta["classes"], meta.get("other", "其他")


def train(epochs=12, seed=0):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(ROOT, "tools", "vision"))
    from train_unit_id import Net
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    classes, other = _classes()
    cid = {n: i for i, n in enumerate(classes)}
    rows, crops = _load_units()
    lab = [r for r in rows if r["side"] == "ours" and r.get("label")]
    # 按对局划分（只有 20 局）：match%4==3 测试、==1 验证、其余训练
    sp = {}
    for r in lab:
        mi = r.get("match") or 0
        sp[r["cid"]] = "test" if mi % TEST_EVERY == 3 else ("val" if mi % TEST_EVERY == 1 else "train")
    data = {k: [(r["cid"], cid.get(r["label"], cid[other])) for r in lab if sp[r["cid"]] == k] for k in ("train", "val", "test")}
    print({k: len(v) for k, v in data.items()}, "未进类表的标签占比",
          round(sum(1 for r in lab if r["label"] not in cid) / max(1, len(lab)), 3), flush=True)
    model = Net(len(classes))
    model.load_state_dict(torch.load(os.path.join(ROOT, "data/vision/runs/unit_id_v1/best.pt"), map_location="cpu"))

    def tens(ids, aug, rng):
        ims = []
        for c in ids:
            im = np.asarray(crops[c])
            if aug:
                oy, ox = rng.randint(0, CROP_H - IN_H), rng.randint(0, CROP_W - IN_W)
                im = im[oy:oy + IN_H, ox:ox + IN_W].astype(np.float32)
                im = im * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
                if rng.random() < 0.5:
                    im = im[:, ::-1]                     # 战斗里单位会转身：水平翻转
                im = np.clip(im, 0, 255)
            else:
                oy, ox = (CROP_H - IN_H) // 2, (CROP_W - IN_W) // 2
                im = im[oy:oy + IN_H, ox:ox + IN_W].astype(np.float32)
            ims.append(im)
        return torch.from_numpy(np.ascontiguousarray(np.stack(ims) / 255.0, dtype=np.float32)).permute(0, 3, 1, 2)

    def predict(items):
        model.eval()
        P = []
        with torch.no_grad():
            for i in range(0, len(items), 256):
                P.append(F.softmax(model(tens([c for c, _ in items[i:i + 256]], False, None)), 1).numpy())
        return np.concatenate(P)

    def acc(items, P):
        y = np.array([t for _, t in items])
        return float((P.argmax(1) == y).mean())

    base = {k: acc(v, predict(v)) for k, v in data.items() if v}
    print("微调前 top1", base, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = random.Random(seed)
    cnt = Counter(t for _, t in data["train"])
    w = torch.tensor([(len(data["train"]) / cnt[t]) ** 0.5 for _, t in data["train"]])
    best, curves = -1, []
    for ep in range(epochs):
        model.train()
        idx = torch.multinomial(w, len(data["train"]), replacement=True).tolist()
        tl = 0.0
        for i in range(0, len(idx), 64):
            b = [data["train"][j] for j in idx[i:i + 64]]
            loss = F.cross_entropy(model(tens([c for c, _ in b], True, rng)), torch.tensor([t for _, t in b]), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        a = acc(data["val"], predict(data["val"]))
        curves.append({"epoch": ep, "loss": round(tl / len(idx), 4), "val_top1": round(a, 4)})
        print(curves[-1], flush=True)
        if a > best:
            best = a
            torch.save(model.state_dict(), os.path.join(OUT, "battle_unit_id.pt"))
    model.load_state_dict(torch.load(os.path.join(OUT, "battle_unit_id.pt")))
    # 门槛：验证集上"被采纳的预测（不是其他、置信≥t）准确率 ≥ 0.9"的最低 t
    res = {"before": base, "curves": curves}
    for k in ("val", "test"):
        P = predict(data[k]); y = np.array([t for _, t in data[k]])
        res[k + "_top1"] = acc(data[k], P)
        res[k + "_P"] = (P, y)
    P, y = res.pop("val_P")
    thr = 0.95
    for t in np.arange(0.3, 0.96, 0.05):
        sel = (P.max(1) >= t) & (P.argmax(1) != cid[other])
        if sel.sum() and (P.argmax(1)[sel] == y[sel]).mean() >= 0.9:
            thr = float(t); break
    res["thresh"] = round(thr, 2)
    P, y = res.pop("test_P")
    sel = (P.max(1) >= thr) & (P.argmax(1) != cid[other])
    res["test_coverage"] = round(float(sel.mean()), 4)
    res["test_precision"] = round(float((P.argmax(1)[sel] == y[sel]).mean()) if sel.sum() else 0, 4)
    res["n"] = {k: len(v) for k, v in data.items()}
    # 每类测试准确率（被采纳的）
    per = {}
    for j in set(y.tolist()):
        m = sel & (y == j)
        per[classes[j]] = [int((y == j).sum()), int(m.sum()), round(float((P.argmax(1)[m] == j).mean()), 3) if m.sum() else None]
    res["test_per_class"] = per
    json.dump(res, open(os.path.join(OUT, "train_metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: v for k, v in res.items() if k not in ("curves", "test_per_class")})


def apply(min_match=MIN_MATCH, min_round=10):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(ROOT, "tools", "vision"))
    from train_unit_id import Net
    classes, other = _classes()
    rows, crops = _load_units()
    met = json.load(open(os.path.join(OUT, "train_metrics.json"), encoding="utf-8"))
    thr = met["thresh"]
    models = {}
    for tag, path in (("v1", os.path.join(ROOT, "data/vision/runs/unit_id_v1/best.pt")), ("battle", os.path.join(OUT, "battle_unit_id.pt"))):
        m = Net(len(classes)); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval(); models[tag] = m
    sel = [r for r in rows if (r.get("match") or 0) >= min_match and r["round"] >= min_round
           and (r["side"] == "enemy" or r.get("ours_top") or r.get("label"))]
    oy, ox = (CROP_H - IN_H) // 2, (CROP_W - IN_W) // 2
    out = open(os.path.join(OUT, "apply.jsonl"), "w", encoding="utf-8")
    for i in range(0, len(sel), 256):
        b = sel[i:i + 256]
        x = np.stack([np.asarray(crops[r["cid"]])[oy:oy + IN_H, ox:ox + IN_W] for r in b]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            P = {k: F.softmax(m(x), 1).numpy() for k, m in models.items()}
        for k, r in enumerate(b):
            rec = {kk: r[kk] for kk in ("run", "match", "rank", "ban", "img", "round", "carry", "carry_lv", "side", "lv", "raw",
                                         "rank_in_side", "label", "ours_top", "bar", "our_board", "enemy_lvs", "ours_lvs") if kk in r}
            for tag, p in P.items():
                j = p[k].argsort()[::-1][:3]
                rec["pred_" + tag] = [[classes[t], round(float(p[k][t]), 3)] for t in j]
            top = rec["pred_battle"][0]
            rec["name"] = top[0] if (top[1] >= thr and top[0] != other) else "未知"
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    print("写入", len(sel), "门槛", thr)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "extract"
    {"extract": extract, "train": train, "apply": apply}[cmd]()
