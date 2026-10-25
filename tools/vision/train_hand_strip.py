"""手牌条识别小模型（hand_strip_v1）：训练 + 验证集选模型 + 测试集只评一次 + 导出 ONNX + CPU 延迟基准。

  # 晚上跑（有 onnx 包、有 CUDA 的环境；数据见 tools/vision/build_hand_strip.py）
  .venv-dfine/Scripts/python.exe -X utf8 tools/vision/train_hand_strip.py --device cuda --epochs 40
  # 只测延迟（不训练；随机权重即可）
  .venv-dfine/Scripts/python.exe -X utf8 tools/vision/train_hand_strip.py --bench-only

起因（2026-10-09：经常拖动空牌位、误触出表情发送轮盘，改用图像识别）：
extractors.hand_layout 靠霍夫圆 + 角标 + 网格打分推张数，牌被点起错位 / 没图标的牌 / 场地上的圆都会推错。

模型：输入 = 手牌条固定裁图 y 528~720、x 160~1184（1280x720 坐标），缩到 48x256（原图 /4），RGB，0~1。
  主干：4 个 conv 块，只在高度方向大幅下采样，宽度只降 2 倍 → 每列 8 原图 px，共 128 列；
  高度求平均后接 3 层空洞 1D 卷积（看左右上下文，叠放时要知道"左边还有几张"）。
  每列输出：
    seg   3 类  背景 / 枫叶舞槽 / 手牌露出部分
    edge  1     "这一列是某张牌的左边缘"热图（高斯 σ=1 列）
    off   1     左边缘在这一列里的亚列偏移（0~1，×8px）
    awk   1     这一列所在的牌是不是觉醒卡（只在有标签的牌上监督）
  全局：count 16 类（0~15 张，与解码出的张数互相印证）。
解码（decode）：edge 局部极大且 >0.5、所在位置 seg=手牌 → 每张牌左边缘；露出右边缘 = 下一张左边缘，
  最后一张 = 手牌 seg 连续段的末端；抓取中心 = (l+r)/2（和 extractors 现在的 hand_slots["x"] 同义）。

标签：data/vision/hand_strip_v1/labels.jsonl（tier A/B = 状态+重算布局+图标+前后帧互相印证；
  Z = 0 张手牌（只有枫叶舞槽）；M = 拖拽半空帧（借 pre 的标签，被拖那张不监督）；N = 点开看牌没弹详情的空位，只监督那几列 = 背景）。
输出：<out>/best.pt、metrics.json（测试集分档：张数准确率、count 头准确率、抓取中心误差 mean/p90/max、
  空位误报率；N 档上与 hand_layout 对照）、hand_strip.onnx（--export-to 拷到 assets/model/vision/hand_strip_v1.onnx）。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "vision" / "hand_strip_v1"
X0, X1, Y0, Y1 = 160, 1184, 528, 720          # 与 build_hand_strip 一致
IN_H, IN_W = 48, 256                           # 原图 /4
COLS = 128                                     # 每列 8 原图 px
COL_PX = (X1 - X0) / COLS
MAXN = 15
CARD_W = 142


# ---------------------------------------------------------------- 数据
def load(split: str) -> list[dict]:
    rows = []
    for line in (D / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] != split:
            continue
        im = cv2.imdecode(np.fromfile(str(D / "crops" / f"{r['id']}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        r["img"] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)        # 磁盘 BGR → 模型吃 RGB（线上截图本来就是 RGB）
        rows.append(r)
    return rows


def col_of(x: float) -> float:
    return (x - X0) / COL_PX


def make_targets(r: dict, dx_px: float = 0.0):
    """每列的监督目标。dx_px = 增强时整幅水平平移的像素（标签跟着平移）。ignore = -100 / 权重 0。"""
    seg = np.full(COLS, -100, np.int64)
    edge = np.zeros(COLS, np.float32)
    edge_w = np.zeros(COLS, np.float32)
    off = np.zeros(COLS, np.float32)
    off_w = np.zeros(COLS, np.float32)
    awk = np.zeros(COLS, np.float32)
    awk_w = np.zeros(COLS, np.float32)
    count = -100
    centers = np.arange(COLS) * COL_PX + COL_PX / 2 + X0       # 每列中心的原图 x
    if r["tier"] == "N":
        x = r["empty_x"] + dx_px
        m = np.abs(centers - x) <= 16
        seg[m] = 0
        edge_w[m] = 1.0                                         # 那几列也不该有"左边缘"
        return seg, edge, edge_w, off, off_w, awk, awk_w, count
    seg[:] = 0
    edge_w[:] = 1.0
    cards = r["cards"] or []
    if r.get("skill"):
        a, b = r["skill"][0] + dx_px, r["skill"][1] + dx_px
        seg[(centers >= a) & (centers < b)] = 1
    held_cols = np.zeros(COLS, bool)
    for c in cards:
        l, rr = c["l"] + dx_px, c["r"] + dx_px
        m = (centers >= l) & (centers < rr)
        seg[m] = 2
        if c.get("held"):
            held_cols |= (centers >= l - 12) & (centers < rr + 12)
            continue
        cl = col_of(l)
        j = int(np.floor(cl))
        g = np.exp(-0.5 * ((np.arange(COLS) - j) ** 2))   # σ=1 列，峰在左边缘所在列（整数列，不会并列），亚列位置交给 off

        edge = np.maximum(edge, g.astype(np.float32))
        if 0 <= j < COLS:
            off[j] = cl - j
            off_w[j] = 1.0
        if c.get("awk") in (0, 1):
            awk[m] = float(c["awk"])
            awk_w[m] = 1.0
    seg[held_cols] = -100
    edge_w[held_cols] = 0.0
    awk_w[held_cols] = 0.0
    count = min(len(cards), MAXN)
    return seg, edge, edge_w, off, off_w, awk, awk_w, count


def augment(r: dict, rng: random.Random):
    """亮度/对比度、水平小平移（标签跟着移）、垂直平移（牌条下沉/抬起）、单张"点起"（模拟被点起的牌错位）。"""
    im = r["img"].astype(np.float32)
    h, w = im.shape[:2]                                # 96x512（原图 /2）
    dx = 0
    if r["tier"] != "N" and rng.random() < 0.5:
        dx = rng.randint(-6, 6)                         # 原图 /2 坐标下的像素
        im = np.roll(im, dx, axis=1)
        if dx > 0:
            im[:, :dx] = im[:, dx:dx + 1]
        elif dx < 0:
            im[:, dx:] = im[:, dx - 1:dx]
    if rng.random() < 0.3:
        dy = rng.randint(-8, 16)                        # 原图 -16~+32px：牌条整体下沉（拖拽中）/ 上移
        im = np.roll(im, dy, axis=0)
        if dy > 0:
            im[:dy] = im[dy:dy + 1]
        elif dy < 0:
            im[dy:] = im[dy - 1:dy]
    if r["tier"] in ("A", "B") and r["cards"] and rng.random() < 0.25:
        c = rng.choice(r["cards"])                      # 把一张牌露出的那一截往上抬 5~12（原图 10~24px）
        a = max(0, int((c["l"] - X0) / 2))
        b = min(w, int((c["r"] - X0) / 2))
        up = rng.randint(5, 12)
        if b - a > 4:
            im[:-up, a:b] = im[up:, a:b].copy()
    im = im * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.3:
        im = im + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 4, im.shape)
    im = np.clip(im, 0, 255).astype(np.uint8)
    return im, dx * 2.0


def to_input(ims: list[np.ndarray]) -> torch.Tensor:
    x = np.stack([cv2.resize(i, (IN_W, IN_H), interpolation=cv2.INTER_AREA) for i in ims]).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def batch_targets(rows, dxs):
    ts = [make_targets(r, d) for r, d in zip(rows, dxs)]
    out = []
    for k in range(8):
        v = [t[k] for t in ts]
        out.append(torch.tensor(np.stack(v)) if k < 7 else torch.tensor(v))
    return out


# ---------------------------------------------------------------- 模型
class Net(nn.Module):
    def __init__(self, w: int = 32):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
        self.f = nn.Sequential(blk(3, w, (2, 1)),            # 24x256
                               blk(w, w * 3 // 2, (2, 2)),   # 12x128
                               blk(w * 3 // 2, w * 2, (2, 1)),   # 6x128
                               blk(w * 2, w * 3, (2, 1)))    # 3x128
        c = w * 3
        self.ctx = nn.Sequential(*[nn.Sequential(nn.Conv1d(c, c, 3, padding=d, dilation=d, bias=False),
                                                 nn.BatchNorm1d(c), nn.ReLU()) for d in (1, 2, 4, 8)])
        self.head = nn.Conv1d(c, 3 + 1 + 1 + 1, 1)
        self.count = nn.Linear(c * 2, MAXN + 1)

    def forward(self, x):
        h = self.f(x).mean(2)                  # N x C x 128
        h = h + self.ctx(h)
        o = self.head(h)                       # N x 6 x 128
        g = torch.cat([h.mean(2), h.amax(2)], 1)
        return o[:, :3], o[:, 3], torch.sigmoid(o[:, 4]), o[:, 5], self.count(g)


def losses(out, tg, dev):
    seg_l, edge_l, off_p, awk_l, cnt_l = out
    seg, edge, edge_w, off, off_w, awk, awk_w, cnt = [t.to(dev) for t in tg]
    l_seg = F.cross_entropy(seg_l, seg, ignore_index=-100) if (seg != -100).any() else seg_l.sum() * 0
    bce = F.binary_cross_entropy_with_logits(edge_l, edge, reduction="none")
    pos = 1.0 + 9.0 * (edge > 0.5).float()                       # 正样本列少，加权
    l_edge = (bce * pos * edge_w).sum() / edge_w.sum().clamp(min=1)
    l_off = (torch.abs(off_p - off) * off_w).sum() / off_w.sum().clamp(min=1)
    l_awk = (F.binary_cross_entropy_with_logits(awk_l, awk, reduction="none") * awk_w).sum() / awk_w.sum().clamp(min=1)
    l_cnt = F.cross_entropy(cnt_l, cnt, ignore_index=-100) if (cnt != -100).any() else cnt_l.sum() * 0
    return l_seg + 2.0 * l_edge + l_off + 0.5 * l_awk + 0.3 * l_cnt


# ---------------------------------------------------------------- 解码 / 评估
def decode_one(seg_l, edge_l, off_p, awk_l, cnt_l, thr: float = 0.5) -> dict:
    """numpy 输入（单帧，形状 3xC / C / C / C / 16）→ {"cards": [{l, r, x, awk}], "skill": bool, "count_head": k}。
    线上接入时照抄这个函数（agent 里不 import torch）。"""
    seg = seg_l.argmax(0)
    e = 1 / (1 + np.exp(-edge_l))
    # 贪心 NMS：从高到低取，4 列（32px）内只留一个（最密的叠放牌距也有 ~43px）；并列时取左边那列
    cand = sorted((j for j in range(COLS) if e[j] >= thr and (seg[j] == 2 or (j + 1 < COLS and seg[j + 1] == 2))),
                  key=lambda j: (-e[j], j))
    peaks = []
    for j in cand:
        if all(abs(j - q) > 4 for q in peaks):
            peaks.append(j)
    peaks.sort()
    lefts = [X0 + (j + float(np.clip(off_p[j], 0, 1))) * COL_PX for j in peaks]
    cards = []
    for k, l in enumerate(lefts):
        if k + 1 < len(lefts):
            r = lefts[k + 1]
        else:                                  # 最后一张：手牌 seg 连续段的末端（上限一张牌宽）
            j = peaks[k]
            while j + 1 < COLS and seg[j + 1] == 2:
                j += 1
            r = min(X0 + (j + 1) * COL_PX, l + CARD_W)
        a, b = int(col_of(l)), max(int(col_of(r)), int(col_of(l)) + 1)
        p_awk = float(np.mean(1 / (1 + np.exp(-awk_l[a:b])))) if b > a else 0.0
        cards.append({"l": round(l, 1), "r": round(r, 1), "x": round((l + r) / 2, 1), "awk": round(p_awk, 3)})
    return {"cards": cards, "skill": bool((seg == 1).sum() >= 4), "count_head": int(np.argmax(cnt_l))}


def predict(model, rows, dev, B=128):
    model.eval()
    res = []
    with torch.no_grad():
        for i in range(0, len(rows), B):
            x = to_input([r["img"] for r in rows[i:i + B]]).to(dev)
            o = [t.float().cpu().numpy() for t in model(x)]
            res += [decode_one(o[0][k], o[1][k], o[2][k], o[3][k], o[4][k]) for k in range(len(x))]
    return res


def evaluate(rows, preds) -> dict:
    """分档报：张数准确率 / count 头准确率 / 抓取中心误差 / 空位误报率 / 觉醒。"""
    out = {}
    for name, sel in (("A", ("A",)), ("A+B", ("A", "B")), ("Z", ("Z",)), ("M", ("M",))):
        idx = [i for i, r in enumerate(rows) if r["tier"] in sel]
        if not idx:
            continue
        n_ok, ch_ok, errs, fp_cards, fp_frames, pred_total = 0, 0, [], 0, 0, 0
        by_n = {}
        awk_tp = awk_fp = awk_fn = 0
        for i in idx:
            r, p = rows[i], preds[i]
            tc = [c for c in r["cards"]]
            t_n, p_n = len(tc), len(p["cards"])
            nh = sum(1 for c in tc if c.get("held"))
            ok = t_n == p_n or (nh > 0 and p_n == t_n - nh)    # M 档：被拖的那张可能已离开牌条，少认它不算错
            n_ok += ok
            ch_ok += p["count_head"] == t_n
            b = by_n.setdefault(t_n, [0, 0])
            b[0] += 1
            b[1] += ok
            # 空位误报：预测的抓取中心不落在任何一张真牌的露出段里（held 的牌也算真牌）
            pred_total += p_n
            bad = sum(1 for pc in p["cards"] if not any(c["l"] - 4 <= pc["x"] <= c["r"] + 4 for c in tc))
            fp_cards += bad
            fp_frames += bad > 0
            if t_n == p_n:
                for c, pc in zip(tc, p["cards"]):
                    if not c.get("held"):
                        errs.append(abs((c["l"] + c["r"]) / 2 - pc["x"]))
                        if c.get("awk") in (0, 1):
                            pa = pc["awk"] >= 0.5
                            awk_tp += pa and c["awk"] == 1
                            awk_fp += pa and c["awk"] == 0
                            awk_fn += (not pa) and c["awk"] == 1
        e = np.array(errs) if errs else np.zeros(1)
        out[name] = {"n": len(idx), "count_acc": n_ok / len(idx), "count_head_acc": ch_ok / len(idx),
                     "count_acc_by_n": {k: round(v[1] / v[0], 4) for k, v in sorted(by_n.items())},
                     "n_by_n": {k: v[0] for k, v in sorted(by_n.items())},
                     "center_err_px": {"mean": float(e.mean()), "p90": float(np.percentile(e, 90)), "max": float(e.max())},
                     "empty_fp_card_rate": fp_cards / max(pred_total, 1), "empty_fp_frame_rate": fp_frames / len(idx),
                     "awaken": {"tp": awk_tp, "fp": awk_fp, "fn": awk_fn}}
    idx = [i for i, r in enumerate(rows) if r["tier"] == "N"]
    if idx:
        hit = sum(1 for i in idx if any(abs(pc["x"] - rows[i]["empty_x"]) <= 30 or pc["l"] <= rows[i]["empty_x"] <= pc["r"]
                                        for pc in preds[i]["cards"]))
        out["N"] = {"n": len(idx), "model_says_card_at_empty_x": hit / len(idx)}
    return out


def layout_baseline_on_N(rows) -> dict:
    """N 档（已知空位）上现有 hand_layout 是否在那里放了一张牌（需要原帧）。"""
    import sys
    sys.path.insert(0, str(ROOT))
    from agent.perception import extractors as E
    idx = [r for r in rows if r["tier"] == "N"]
    hit = 0
    for r in idx:
        bgr = cv2.imdecode(np.fromfile(str(ROOT / r["frame"]), np.uint8), cv2.IMREAD_COLOR)
        lay = E.hand_layout(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) or {"xs": []}
        xs = lay["xs"]
        segs = [(x - 20, (xs[k + 1] - 20) if k + 1 < len(xs) else x - 20 + CARD_W) for k, x in enumerate(xs)]
        hit += any(a <= r["empty_x"] <= b for a, b in segs)
    return {"n": len(idx), "hand_layout_says_card_at_empty_x": hit / max(len(idx), 1)}


# ---------------------------------------------------------------- 延迟
def bench(onnx_path: Path, iters: int = 300) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2                      # 线上 agent 同时还在跑 OCR / D-FINE，别吃满核
    s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    ts = []
    for k in range(iters + 20):
        t0 = time.perf_counter()
        c = cv2.resize(frame[Y0:Y1, X0:X1], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        x = (c.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        o = s.run(None, {"x": np.ascontiguousarray(x)})
        decode_one(o[0][0], o[1][0], o[2][0], o[3][0], o[4][0])
        if k >= 20:
            ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)), "max_ms": float(a.max()),
            "threads": 2, "includes": "裁图+缩放+推理+解码"}


def export(model, path: Path) -> None:
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, IN_H, IN_W), str(path), input_names=["x"],
                      output_names=["seg", "edge", "off", "awk", "count"],
                      dynamic_axes={k: {0: "n"} for k in ("x", "seg", "edge", "off", "awk", "count")},
                      opset_version=17, dynamo=False)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "data" / "vision" / "runs" / "hand_strip_v1"))
    ap.add_argument("--export-to", default=str(ROOT / "assets" / "model" / "vision" / "hand_strip_v1.onnx"))
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.bench_only:
        m = Net(args.width)
        p = out / "bench_random.onnx"
        export(m, p)
        r = bench(p)
        r["n_params"] = sum(q.numel() for q in m.parameters())
        print(json.dumps(r, ensure_ascii=False))
        return 0
    dev = args.device
    tr, va, te = load("train"), load("val"), load("test")
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  device {dev}")
    # 张数分布不均（1~3 张最多）：按张数频率的平方根反比采样；N 档固定占 ~5%
    key = lambda r: "N" if r["tier"] == "N" else min(r["n"], 10)   # noqa: E731
    cnt = {}
    for r in tr:
        cnt[key(r)] = cnt.get(key(r), 0) + 1
    w = torch.tensor([(len(tr) / cnt[key(r)]) ** 0.5 for r in tr], dtype=torch.float)
    model = Net(args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = random.Random(args.seed)
    best, best_ep, curves = -1.0, -1, []
    for ep in range(args.epochs):
        model.train()
        idx = torch.multinomial(w, len(tr), replacement=True).tolist()
        tl = 0.0
        for i in range(0, len(idx), args.bs):
            b = [tr[j] for j in idx[i:i + args.bs]]
            ims, dxs = zip(*[augment(r, rng) for r in b])
            x = to_input(list(ims)).to(dev)
            loss = losses(model(x), batch_targets(b, dxs), dev)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        ev = evaluate(va, predict(model, va, dev))
        score = (ev.get("A+B", {}).get("count_acc", 0) - ev.get("A+B", {}).get("empty_fp_frame_rate", 0)
                 + 0.2 * ev.get("Z", {}).get("count_acc", 0) - 0.2 * ev.get("N", {}).get("model_says_card_at_empty_x", 0))
        curves.append({"epoch": ep, "train_loss": tl / len(tr), "val": ev})
        print(f"ep {ep:2d} loss {tl / len(tr):.4f} val 张数 {ev.get('A+B', {}).get('count_acc', 0):.4f} "
              f"空位误报帧 {ev.get('A+B', {}).get('empty_fp_frame_rate', 0):.4f} "
              f"中心误差p90 {ev.get('A+B', {}).get('center_err_px', {}).get('p90', 0):.1f}px", flush=True)
        if score > best:
            best, best_ep = score, ep
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    res = {"best_epoch": best_ep, "val": evaluate(va, predict(model, va, dev))}
    res["test"] = evaluate(te, predict(model, te, dev))          # 测试集只评这一次
    try:
        res["test_N_hand_layout_baseline"] = layout_baseline_on_N(te)
    except Exception as e:                                        # noqa: BLE001
        res["test_N_hand_layout_baseline"] = repr(e)
    res["n_params"] = sum(p.numel() for p in model.parameters())
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["curves"] = curves
    onnx_p = out / "hand_strip.onnx"
    try:
        export(model, onnx_p)
        res["latency_cpu"] = bench(onnx_p)
        if args.export_to:
            shutil.copy(onnx_p, args.export_to)
            print("已导出", args.export_to)
    except Exception as e:                                        # noqa: BLE001
        print(f"[跳过 ONNX 导出] {e!r}（用 .venv-dfine/Scripts/python.exe 导出）")
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("val", "test"):
        print(k, json.dumps({t: {kk: v[kk] for kk in v if kk in ("n", "count_acc", "count_head_acc", "empty_fp_frame_rate",
                                                                     "center_err_px", "model_says_card_at_empty_x")}
                             for t, v in res[k].items()}, ensure_ascii=False))
    print("hand_layout 基线(N 档)", res["test_N_hand_layout_baseline"])
    print("CPU 延迟", res.get("latency_cpu"), "参数量", res["n_params"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
