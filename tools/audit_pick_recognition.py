"""三选一 / N 选 1 识别审计（2026-10-21）。

对最近 N 局存下来的三选一截图（phase_talent_*.png）重跑识别：每个选项的"名字"和"描述"是否对得上同一张卡。
对不上 = 识别错（名字被遮挡 / 读成别的卡名）；名字解析不出来 = 漏识。修复后（名字 ↔ 描述交叉核对）应该全对上。
注意：会用 GPU 跑 OCR，**对局进行中不要跑**。

用法：python tools/audit_pick_recognition.py [局数=30]
"""
from __future__ import annotations

import difflib
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.perception.extractors import StateExtractor  # noqa: E402
from agent.perception.state import GameState  # noqa: E402


def main() -> int:
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    runs = sorted(glob.glob(str(ROOT / "data" / "logs" / "live_17925*")) + glob.glob(str(ROOT / "data" / "logs" / "live_17924*")))[-n_runs:]
    E = StateExtractor(ROOT / "assets" / "image")
    tot = bad = unresolved = 0
    cases = []
    for d in runs:
        for f in sorted(glob.glob(d + "/phase_talent_*.png")):
            im = cv2.imdecode(np.fromfile(f, np.uint8), 1)
            if im is None:
                continue
            st = GameState(phase="talent")
            try:
                E._extract_talents(None, cv2.resize(im, (1280, 720))[:, :, ::-1].copy(), st)
            except Exception as e:                          # noqa: BLE001
                cases.append({"file": f, "err": str(e)})
                continue
            for t in st.talents:
                tot += 1
                kd = E._card_descs.get(t.name)
                if not kd:
                    unresolved += 1
                    cases.append({"file": Path(f).name, "run": Path(d).name, "name": t.name, "desc": t.desc[:30], "issue": "名字不在卡库"})
                    continue
                r = 1.0 if (t.name in (t.desc or "") or len((t.desc or "").strip()) < 8) else difflib.SequenceMatcher(None, (t.desc or "").strip(), kd).ratio()   # 天赋版描述带名字 = 对
                if r < 0.35:
                    bad += 1
                    cases.append({"file": Path(f).name, "run": Path(d).name, "name": t.name, "desc": t.desc[:30], "ratio": round(r, 2), "issue": "名字和描述对不上"})
    out = {"runs": len(runs), "options": tot, "mismatch": bad, "unresolved": unresolved, "cases": cases[:200]}
    (ROOT / "data" / "review" / "pick_recognition_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(runs)} 局、{tot} 个选项：名字和描述对不上 {bad}，名字不在卡库 {unresolved}")
    for c in cases[:25]:
        print(" ", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
