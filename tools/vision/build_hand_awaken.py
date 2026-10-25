"""手牌觉醒卡识别的数据池：历史准备帧 → 按 hand_layout 切每张手牌露出来的一截（64x120）+ 像素特征。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_hand_awaken.py

起因（2026-09-25：是否觉醒英雄卡改用图像识别，不再纯靠采样颜色）：
颜色判据（extractors.awaken_tint_spans）数冰蓝/金色像素占比，立绘本身偏金偏蓝的卡（5 阶武则天）会误报，
第四轮第 3 局同一张武则天 2 分钟被点开 9 次。觉醒卡的真正特征是**整幅立绘被渲染成单色雕像**（冰蓝或金），
不是"有多少蓝/金色"—— 小 CNN 能学到形态。

输出 data/vision/hand_awaken_v1/：crops/<id>.png（BGR，64x120），pool.jsonl（id、帧、对局、x、特征、颜色判据结论、划分）。
划分按对局（运行目录）哈希：train 70% / val 15% / test 15%。同一张卡在相邻帧重复出现很多次 —— 同一对局只进一个集合，不泄漏。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.perception import extractors as E  # noqa: E402

OUT = ROOT / "data" / "vision" / "hand_awaken_v1"
W, H, Y0 = 64, 120, 595          # 图标 x 往左 12 到右 52；卡面 y 595~715（与 awaken_tint_spans 的 590~710 基本重合）


def split_of(run: str) -> str:
    h = int(hashlib.md5(run.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def feats(crop_bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return {"ice": round(float(((h >= 80) & (h <= 110) & (s >= 20) & (s <= 130) & (v >= 140)).mean()), 3),
            "gold": round(float(((h >= 12) & (h <= 35) & (s >= 40) & (v >= 150)).mean()), 3),
            "dark": round(float((v < 70).mean()), 3)}


def main() -> int:
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    frames = sorted((ROOT / "data" / "logs").glob("live_*/phase_prepare_*.png"))
    rows, n_frames = [], 0
    for f in frames:
        bgr = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        if bgr.shape[:2] != (720, 1280):
            bgr = cv2.resize(bgr, (1280, 720))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lay = E.hand_layout(rgb)
        if not lay or not lay["xs"]:
            continue
        n_frames += 1
        tint = E.awaken_tint_spans(rgb)
        run = f.parent.name
        for k, x in enumerate(lay["xs"]):
            x0 = int(x) - 12
            if x0 < 0 or x0 + W > 1280:
                continue
            c = bgr[Y0:Y0 + H, x0:x0 + W]
            cid = f"{f.parent.name[5:]}_{f.stem.split('_')[-1]}_{k}"
            cv2.imencode(".png", c)[1].tofile(str(OUT / "crops" / f"{cid}.png"))
            rows.append({"id": cid, "frame": str(f.relative_to(ROOT)), "run": run, "split": split_of(run),
                         "k": k, "n": lay["n"], "x": int(x), **feats(c),
                         "rule": any(a <= x <= b for a, b, _i, _d in tint)})
    (OUT / "pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"帧 {n_frames}，手牌裁图 {len(rows)}，颜色判据判为觉醒 {sum(r['rule'] for r in rows)}")
    for sp in ("train", "val", "test"):
        print(sp, sum(1 for r in rows if r["split"] == sp), "对局", len({r["run"] for r in rows if r["split"] == sp}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
