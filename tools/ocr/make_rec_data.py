"""认字模型（rec）的微调数据：真实小图（裁到文字行）+ 用卡牌库卡名合成的图。输出 PaddleOCR 的标注格式。

真实：data/ocr_bench/<ver>/labels.jsonl 的 train 划分；先用找字模型框出文字行（多个框取并集），裁紧后配正确卡名。
      train 划分再按对局切出 15% 做训练时的验证集（**测试划分完全不碰**，留给最后评估）。
合成：卡牌库里每个卡名（天赋/效果牌加倍），用系统中文字体渲染成白色/金色字 + 深色描边，贴在从真实截图
      卡面区域随机取的背景块上，再加缩放、模糊、JPEG 压缩，模仿游戏画面。
输出：data/ocr_bench/<ver>/rec/{real,synth}/*.png，rec/train.txt、rec/val.txt（"相对路径\t标签"）
  python tools/ocr/make_rec_data.py --ver v1 --synth-per-name 24
"""
from __future__ import annotations

import argparse
import io
import re
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

_SUFFIX = re.compile(r"[(（][^()（）]*[)）]$")


def display(name: str) -> str:
    """卡牌库里区分同名卡的后缀（「护命灵珠(天赋牌)」）画面上不显示，标签要去掉。"""
    return _SUFFIX.sub("", name or "")


FONTS = ["Dengb.ttf", "Deng.ttf", "NotoSansSC-VF.ttf", "STXIHEI.TTF", "FZYTK.TTF", "STZHONGS.TTF"]
FILLS = [(255, 255, 255), (255, 244, 214), (255, 226, 150), (250, 214, 120), (235, 235, 245)]
STROKES = [(20, 20, 20), (60, 35, 10), (40, 20, 50), (0, 0, 0)]


def _imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), 1)


def _imwrite(p, img):
    cv2.imencode(".png", img)[1].tofile(str(p))


def real_crops(root: Path, out: Path, rows: list, ocr, rng) -> list:
    res = []
    for r in rows:
        im = _imread(root / r["file"])
        boxes = ocr.boxes(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (0, 0, im.shape[1], im.shape[0]))
        if boxes:
            xs = [b[1][0] for b in boxes] + [b[1][0] + b[1][2] for b in boxes]
            ys = [b[1][1] for b in boxes] + [b[1][1] + b[1][3] for b in boxes]
            x0, x1 = max(0, min(xs) - 3), min(im.shape[1], max(xs) + 3)
            y0, y1 = max(0, min(ys) - 3), min(im.shape[0], max(ys) + 3)
            crop = im[y0:y1, x0:x1] if x1 - x0 > 8 and y1 - y0 > 8 else im
        else:
            crop = im                                   # 找不到字：整块给认字模型（线上窄 ROI 也是这么做的）
        name = Path(r["file"]).stem + ".png"
        _imwrite(out / name, crop)
        res.append((f"real/{name}", display(r["gt"]), r["kind"], r["run"]))
    return res


def backgrounds(runs: set, rng, n: int = 600) -> list:
    """从真实截图的卡面区域（立绘/底纹，没有卡名文字的部分）取背景块。"""
    pngs = []
    for rd in (PROJECT_DIR / "data" / "logs").glob("live_*"):
        if rd.name in runs:
            pngs += list(rd.glob("phase_talent_*.png"))[:6] + list(rd.glob("phase_prepare_*.png"))[:6] \
                + list(rd.glob("phase_card_pick_*.png"))[:4]
    rng.shuffle(pngs)
    bgs = []
    for p in pngs[:200]:
        im = _imread(p)
        if im is None:
            continue
        for _ in range(4):
            w, h = rng.randint(140, 260), rng.randint(40, 60)
            x = rng.randint(0, im.shape[1] - w - 1)
            y = rng.randint(60, 600 - h)
            bgs.append(im[y:y + h, x:x + w].copy())
        if len(bgs) >= n:
            break
    return bgs


def render(name: str, bg, rng, fonts) -> np.ndarray:
    font = ImageFont.truetype(rng.choice(fonts), rng.randint(20, 32))
    stroke = rng.choice([0, 1, 1, 2, 2, 3])
    l, t, r, b = font.getbbox(name, stroke_width=stroke)
    tw, th = r - l, b - t
    pad_x, pad_y = rng.randint(3, 14), rng.randint(3, 10)
    W, H = tw + 2 * pad_x, th + 2 * pad_y
    bgr = cv2.resize(bg, (W, H))
    if rng.random() < 0.5:                                 # 金底卡：背景偏暖、偏亮
        tint = np.array([40, 110, 170], np.float32) * rng.uniform(0.3, 0.8)
        bgr = np.clip(bgr.astype(np.float32) * rng.uniform(0.6, 1.0) + tint, 0, 255).astype(np.uint8)
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.text((pad_x - l, pad_y - t), name, font=font, fill=rng.choice(FILLS),
           stroke_width=stroke, stroke_fill=rng.choice(STROKES))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.5:                                 # 模拟截图缩放（设备 1920 -> 1280）
        s = rng.uniform(0.6, 0.9)
        img = img.resize((max(8, int(W * s)), max(8, int(H * s)))).resize((W, H))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(60, 95))
    return cv2.cvtColor(np.array(Image.open(buf)), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--synth-per-name", type=int, default=24)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    rng = random.Random(0)
    root = PROJECT_DIR / "data" / "ocr_bench" / args.ver
    rec = root / "rec"
    for sub in ("real", "synth"):
        (rec / sub).mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (root / "labels.jsonl").open(encoding="utf-8")]
    train_rows = [r for r in rows if r["split"] == "train"]
    runs = sorted({r["run"] for r in train_rows})
    rng.shuffle(runs)
    val_runs = set(runs[:max(1, int(len(runs) * args.val_frac))])

    import torch  # noqa: F401
    import os
    os.environ.setdefault("WZWXQ_OCR_EP", "cpu")
    from agent.perception.onnx_ocr import OnnxOcr
    ocr = OnnxOcr()                                          # 线上的轻量版找字模型
    real = real_crops(root, rec / "real", train_rows, ocr, rng)

    cards = json.loads((PROJECT_DIR / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    fonts = [str(Path("C:/Windows/Fonts") / f) for f in FONTS if (Path("C:/Windows/Fonts") / f).exists()]
    bgs = backgrounds(set(runs) - val_runs, rng)
    synth = []
    for i, (n, c) in enumerate(cards.items()):
        k = args.synth_per_name * (2 if c.get("category") in ("天赋牌", "效果牌") else 1)
        for j in range(k):
            img = render(display(n), rng.choice(bgs), rng, fonts)
            fn = f"s{i:04d}_{j:02d}.png"
            _imwrite(rec / "synth" / fn, img)
            synth.append((f"synth/{fn}", display(n)))

    train, val = [], []
    for path, gt, kind, run in real:
        reps = 3 if kind == "talent" else 1                  # 真实难例（天赋）过采样
        (val if run in val_runs else train).extend([(path, gt)] * (1 if run in val_runs else reps))
    train += synth
    rng.shuffle(train)
    for fname, lst in (("train.txt", train), ("val.txt", val)):
        (rec / fname).write_text("".join(f"{p}\t{g}\n" for p, g in lst), encoding="utf-8")
    info = {"real_train_imgs": sum(1 for x in real if x[3] not in val_runs),
            "real_val_imgs": len(val), "synth_imgs": len(synth), "train_lines": len(train),
            "fonts": [Path(f).name for f in fonts], "bg_patches": len(bgs)}
    (rec / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
