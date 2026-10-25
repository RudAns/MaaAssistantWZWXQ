"""用官网卡面合成"手牌露出那一截"（64x120，与 hand_awaken_v2/crops 同一位置）→ data/vision/hand_synth_v1/。

  .venv/Scripts/python.exe -X utf8 tools/vision/synth_web_cards.py [--per 40]

映射来自 tools/vision/calib_web_cards.py（96 个真实样本、匹配分中位 0.945）：
  卡面缩放 s=0.3775，卡左边缘 = 图标x-19，卡顶 y=567（中位绝对偏差 0~1px）。
  线上裁图 帧[595:715, 图标x-12 : 图标x+52] ⇔ 卡面 [(595-567)/s : (715-567)/s, 7/s : 71/s]。
增强：真实光照颜色变换（color_bank.json：118 对真实/合成裁图最小二乘拟合的 3x4 颜色矩阵，60% 概率套用）、±2px 平移、±4% 缩放、亮度/对比/色调/饱和度抖动、轻微模糊、JPEG 压缩、右侧被下一张牌盖住（贴另一张卡的左边一截）。
类别 = cards.json 里的手牌类卡名（英雄/效果/装备/天赋/棋手技能牌，召唤物除外）；觉醒版归同一个卡名。
"""
import argparse, json, random
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "data/vision/web_cards"
OUT = ROOT / "data/vision/hand_synth_v1"
S, DX, TOP = 0.3775, -19.0, 567.0
FX0, FX1, FY0, FY1 = -12, 52, 595, 715          # 线上裁图（相对图标 x / 绝对 y）
HANDCATS = ("英雄牌", "效果牌", "装备牌", "天赋牌", "棋手技能牌")


def load(p, bg):
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 4:
        a = im[..., 3:].astype(np.float32) / 255.0
        im = (im[..., :3] * a + np.array(bg, np.float32) * (1 - a)).astype(np.uint8)
    return im                                            # BGR


def cut(card, rng, jit=True):
    s = S * (rng.uniform(0.96, 1.04) if jit else 1.0)
    ox = rng.uniform(-2, 2) if jit else 0.0
    oy = rng.uniform(-2, 2) if jit else 0.0
    # 先按 s 用 INTER_AREA 缩小（和游戏渲染的缩小一致，避免锯齿），再做亚像素平移裁剪：
    # 帧坐标 fx（相对图标 x）→ 缩小后卡面坐标 fx - DX；fy → fy - TOP
    small = cv2.resize(card, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    M = np.array([[1, 0, FX0 + ox - DX], [0, 1, FY0 + oy - TOP]], np.float32)
    return cv2.warpAffine(small, M, (FX1 - FX0, FY1 - FY0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


BANK = []          # 真实裁图 ≈ A·合成裁图 的颜色变换（不同地图光照把手牌染成冷/暖色调），见 color_bank.json


def photo(c, rng):
    if BANK and rng.random() < 0.6:
        A = np.array(rng.choice(BANK), np.float32)
        u = rng.uniform(0.5, 1.0)
        A = u * A + (1 - u) * np.vstack([np.eye(3, dtype=np.float32), np.zeros((1, 3), np.float32)])
        X = np.c_[c.reshape(-1, 3).astype(np.float32), np.ones(c.shape[0] * c.shape[1], np.float32)]
        c = np.clip(X @ A, 0, 255).reshape(c.shape).astype(np.uint8)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    c = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    m = c.mean()
    c = (c - m) * rng.uniform(0.85, 1.15) + m + rng.uniform(-15, 15)
    c = np.clip(c, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        c = cv2.GaussianBlur(c, (3, 3), rng.uniform(0.3, 0.9))
    if rng.random() < 0.6:
        ok, b = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(50, 92)])
        c = cv2.imdecode(b, cv2.IMREAD_COLOR)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=40)
    args = ap.parse_args()
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    if (WEB / "color_bank.json").exists():
        BANK.extend(json.loads((WEB / "color_bank.json").read_text()))
    cards = json.loads((ROOT / "data/cards.json").read_text(encoding="utf-8"))["cards"]
    ok = {n for n, c in cards.items() if c.get("category") in HANDCATS}
    idx = json.loads((WEB / "index.json").read_text(encoding="utf-8"))
    ents = [e for e in idx if e["name"] in ok and (WEB / e["file"]).exists()]
    rng = random.Random(0)
    bases = {}
    for e in ents:
        bases[(e["name"], e["awk"])] = WEB / e["file"]
    keys = list(bases)
    rows, cache = [], {}
    for (name, awk), p in bases.items():
        n = args.per // 2 if awk else args.per
        for k in range(n):
            card = cache.setdefault((name, awk), load(p, [40, 40, 40]))
            c = cut(card, rng)
            if rng.random() < 0.35:                          # 右侧被下一张牌盖住
                ok2 = rng.choice(keys); other = cut(cache.setdefault(ok2, load(bases[ok2], [40, 40, 40])), rng, jit=False)
                col = rng.randint(40, 62)
                w = 64 - col
                c[:, col:] = other[:, 7:7 + w] if 7 + w <= 64 else other[:, :w]
            c = photo(c, rng)
            cid = f"{len(rows):06d}"
            cv2.imencode(".png", c)[1].tofile(str(OUT / "crops" / f"{cid}.png"))
            rows.append({"id": cid, "label": name, "awk": awk})
    (OUT / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("卡", len(bases), "类别", len({r['label'] for r in rows}), "合成", len(rows))


if __name__ == "__main__":
    main()
