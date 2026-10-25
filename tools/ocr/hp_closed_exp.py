"""血量读法实验：左侧整列找数字框（DB 检测）→ 按 y 分行 → 闭集 CTC 解码（候选只有 "0".."60"）。
校验：六人列表按血量从高到低排，同一帧读数必须从上到下不增；绿色数字 = 我方。"""
import glob, json, random, sys
import cv2, numpy as np
sys.path[:0] = [".", "tools/ocr"]
from agent.perception import onnx_ocr
from ctc_closed import ctc_logp

ocr = onnx_ocr.OnnxOcr.get()
idx = {c: i for i, c in enumerate(ocr.chars)}
CANDS = [(str(v), [idx[ch] for ch in str(v)]) for v in range(0, 61)]
COL = (70, 60, 140, 400)            # x, y, w, h（左侧玩家列表整列）


def read_frame(bgr):
    x0, y0, w, h = COL
    patch = bgr[y0:y0 + h, x0:x0 + w]
    boxes = ocr.detect(patch)
    out = []
    for bx, by, bw, bh in boxes:
        if not (12 <= bh <= 40 and 10 <= bw <= 60):
            continue
        crop = patch[by:by + bh, bx:bx + bw]
        p = ocr.rec(np.ascontiguousarray(_prep(crop)))[0]
        lp = np.log(np.clip(p, 1e-12, 1))
        sc = sorted(((ctc_logp(lp, ids), s) for s, ids in CANDS), reverse=True)
        margin = sc[0][0] - sc[1][0]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = float(((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] > 80) & (hsv[..., 2] > 120)).mean())
        out.append({"y": y0 + by + bh // 2, "x": x0 + bx, "v": int(sc[0][1]), "m": round(margin, 1), "green": round(green, 2)})
    return out


def _prep(c):
    H = 48
    ww = max(8, int(round(c.shape[1] * H / max(1, c.shape[0]))))
    W = (max(64, ww) + 7) // 8 * 8
    x = np.zeros((1, 3, H, W), np.float32)
    im = cv2.resize(c, (ww, H)).astype(np.float32) / 255.0
    x[0, :, :, :ww] = ((im - 0.5) / 0.5).transpose(2, 0, 1)
    return x


if __name__ == "__main__":
    fs = sorted(glob.glob("data/logs/live_*/phase_prepare_*.png")) + sorted(glob.glob("data/logs/live_*/phase_battle_*.png"))
    random.Random(2).shuffle(fs)
    res = []
    for f in fs[:300]:
        bgr = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        if bgr.shape[1] != 1280:
            bgr = cv2.resize(bgr, (1280, 720))
        r = [b for b in read_frame(bgr) if b["m"] >= 2.0]
        r.sort(key=lambda b: b["y"])
        res.append({"f": f, "rows": r})
    n6 = sum(1 for r in res if len(r["rows"]) == 6)
    mono = [r for r in res if len(r["rows"]) >= 3]
    ok = sum(1 for r in mono if all(a["v"] >= b["v"] for a, b in zip(r["rows"], r["rows"][1:])))
    greens = sum(1 for r in res if sum(b["green"] >= 0.15 for b in r["rows"]) == 1)
    print(f"帧 {len(res)}；读出 6 行的 {n6}；≥3 行的 {len(mono)} 里从上到下不增 {ok}（{ok / max(1, len(mono)):.1%}）；恰有 1 行绿色 {greens}")
    json.dump(res, open("data/vision/hp_exp.json", "w"), ensure_ascii=False)
