"""标定"官网卡面 → 手牌露出那一截"的映射：真实不叠放手牌（标签可靠）× 同名官网卡面，多尺度模板匹配。

  .venv/Scripts/python.exe -X utf8 tools/vision/calib_web_cards.py
输出 data/vision/web_cards/calib.json：每个样本的 (scale, 卡左上角相对图标 x 的偏移 dx, 卡顶 y, 匹配分)，以及中位数。
"""
import json, random, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "data/vision/web_cards"


def web_rgb(name, awk=False, bg=40):
    p = WEB / f"{name}{'_awk' if awk else ''}.png"
    if not p.exists():
        return None
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 4:
        a = im[..., 3:].astype(np.float32) / 255.0
        im = (im[..., :3] * a + bg * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def main():
    rows = []
    for p in sorted((ROOT / "data/vision/hand_names_v2/parts").glob("*.jsonl")):
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]
    rows = [r for r in rows if r["label"] and r["roi"][2] >= 110 and r["p_awk"] < 0.5 and (WEB / f"{r['label']}.png").exists()]
    random.seed(0)
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    samp = [x for v in by.values() for x in random.sample(v, min(3, len(v)))]
    out = []
    for r in samp:
        fr = cv2.imdecode(np.fromfile(str(ROOT / r["frame"]), np.uint8), cv2.IMREAD_COLOR)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        x = r["x"]; X0, Y0 = max(0, x - 50), 540
        reg = cv2.cvtColor(fr[Y0:720, X0:min(1280, x + 180)], cv2.COLOR_RGB2GRAY)
        w = cv2.cvtColor(web_rgb(r["label"]), cv2.COLOR_RGB2GRAY)
        best = (-1, None)
        for s in np.arange(0.30, 0.42, 0.0025):
            t = cv2.resize(w, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)[:120]
            if t.shape[0] > reg.shape[0] or t.shape[1] > reg.shape[1]:
                continue
            m = cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(m)
            if v > best[0]:
                best = (v, (float(s), X0 + loc[0] - x, Y0 + loc[1]))
        if best[1]:
            out.append({"id": r["id"], "label": r["label"], "score": round(best[0], 3), "scale": best[1][0], "dx": best[1][1], "top": best[1][2]})
    good = [o for o in out if o["score"] >= 0.5]
    med = {k: float(np.median([o[k] for o in good])) for k in ("scale", "dx", "top")} if good else {}
    res = {k: {"median": med.get(k), "mad": float(np.median([abs(o[k] - med[k]) for o in good])) if good else None} for k in ("scale", "dx", "top")}
    print("样本", len(out), "分数≥0.5", len(good), "分数中位", np.median([o["score"] for o in out]))
    print(res)
    (WEB / "calib.json").write_text(json.dumps({"summary": res, "n": len(out), "n_good": len(good), "samples": out}, ensure_ascii=False, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
