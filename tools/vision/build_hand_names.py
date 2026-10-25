"""手牌卡名识别的数据池：历史准备帧 → 按线上同一套 ROI 切每张手牌的卡名条 + OCR 弱标签。

  WZWXQ_OCR_EP=cpu .venv/Scripts/python.exe -X utf8 tools/vision/build_hand_names.py

起因（2026-09-25 第四轮第 4 局）：N 选 1 拿到的亚瑟，手牌没叠放、卡名清清楚楚，OCR 却读成「业热」→ 认不出、整回合没上场。
卡面没有重叠时，识别底部卡名条就能知道是什么牌，不用打开看；反复出现的识别障碍都优先尝试图像识别。
卡名条是固定字体/位置的 UI 元素（立绘随等级变、卡名条不变）→ 闭集分类（卡库里的名字），不会产生「业热」这种非卡名。

ROI 与线上 `StateExtractor._extract_hand_by_layout` 完全一致：每张牌 (left+3, name_y-15, 露出宽度-6, 30)；叠放时只露出前一两个字。
弱标签 = 线上同一套 OCR + `_resolve_hand_fragment`（认不出的记 None，不进训练）。
输出 data/vision/hand_names_v1/：crops/<id>.png（BGR，原尺寸），pool.jsonl（id、帧、对局、第几张、张数、ROI、OCR 原文、标签、划分）。
划分按对局哈希，与 hand_awaken_v1 相同（同一局只进一个集合）。
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

OUT = ROOT / "data" / "vision" / "hand_names_v1"


def split_of(run: str) -> str:
    h = int(hashlib.md5(run.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def main() -> int:
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    ex = E.StateExtractor(ROOT / "assets" / "image", cards_db_path=ROOT / "data" / "cards.json")
    frames = sorted((ROOT / "data" / "logs").glob("live_*/phase_prepare_*.png"))
    rows = []
    for fi, f in enumerate(frames):
        bgr = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        if bgr.shape[:2] != (720, 1280):
            bgr = cv2.resize(bgr, (1280, 720))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lay = E.hand_layout(rgb)
        if not lay or not lay["xs"] or not (lay["by"] != "图标" or lay["hits"] * 2 >= lay["n"]):
            continue
        xs, n = lay["xs"], lay["n"]
        name_y = lay["icon_y"] + E.HAND_NAME_DY
        for k, ix in enumerate(xs):
            left = ix - E.HAND_ICON_TO_LEFT
            right = (xs[k + 1] - E.HAND_ICON_TO_LEFT) if k + 1 < n else left + E.HAND_PITCH_MAX
            roi = (int(left) + 3, name_y - 15, max(8, int(right - left) - 6), 30)
            x0, y0, w, h = roi
            if x0 < 0 or x0 + w > 1280 or y0 + h > 720:
                continue
            t = E._ocr(None, rgb, roi)
            name = ex._resolve_hand_fragment(t) if t else None
            cid = f"{f.parent.name[5:]}_{f.stem.split('_')[-1]}_{k}"
            cv2.imencode(".png", bgr[y0:y0 + h, x0:x0 + w])[1].tofile(str(OUT / "crops" / f"{cid}.png"))
            rows.append({"id": cid, "frame": str(f.relative_to(ROOT)), "run": f.parent.name,
                         "split": split_of(f.parent.name), "k": k, "n": n, "roi": list(roi),
                         "ocr": t or "", "label": name})
        if fi % 200 == 0:
            print(f"{fi}/{len(frames)} 帧，已切 {len(rows)}", flush=True)
    (OUT / "pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    lab = [r for r in rows if r["label"]]
    print(f"卡名条 {len(rows)}，OCR 认得出 {len(lab)}（{len(lab) / max(1, len(rows)):.1%}），不同卡名 {len({r['label'] for r in lab})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
