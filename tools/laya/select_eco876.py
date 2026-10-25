"""eco876 挑题（2026-10-21）：从 data/laya/datasets/eco876/points.jsonl（第 845~875 局线上 eco 题，f9z）挑三类题，写题号清单给 label_batches export --ids-file。

- d = 落后进度线（题面 BEHIND schedule，或棋手等级 < meta.level_target(回合)：R2<2 / R4<3 / R7<4 / R10<5）——同上次 r9zx_eco_d（54.22）
- e = 进度点前一回合（没落后、但下回合的进度线 > 当前等级：第 3 / 6 / 9 回合；第 1 回合不收）——本次复盘发现的主因，**等规范 54.33 定了再发老师**
- c = 随机对照（其余题按种子抽 --n-ctrl 道；看老师在非落后题上和旧标签 / 线上答案是否漂移）

  .venv/Scripts/python.exe -X utf8 tools/laya/select_eco876.py
输出 data/laya/labels/eco876_{d,e,c}_ids.json，并打印每类按划分 / 线上答案的计数。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P))
from agent.decision.meta import level_target  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="eco876")
    ap.add_argument("--n-ctrl", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    pts = [json.loads(l) for l in (P / "data/laya/datasets" / a.ds / "points.jsonl").open(encoding="utf-8")]
    cls = {}
    for p in pts:
        lv = (p.get("ctx") or {}).get("level") or 0
        r = p.get("round") or 0
        txt = p["question"]["instructions"] + "\n" + p["state"]
        if lv < 6 and ("BEHIND" in txt or lv < level_target(r)):
            cls[p["id"]] = "d"
        elif lv < 6 and r >= 2 and level_target(r + 1) > lv:
            cls[p["id"]] = "e"
    rest = sorted(p["id"] for p in pts if p["id"] not in cls)
    for i in random.Random(a.seed).sample(rest, min(a.n_ctrl, len(rest))):
        cls[i] = "c"
    by = {p["id"]: p for p in pts}
    out = {}
    for k in ("d", "e", "c"):
        ids = sorted(i for i, v in cls.items() if v == k)
        (P / "data/laya/labels" / f"{a.ds}_{k}_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        out[k] = {"n": len(ids),
                  "split": dict(collections.Counter(by[i]["split"] for i in ids)),
                  "online": dict(collections.Counter((by[i].get("online_eco") or {}).get("choice") for i in ids)),
                  "rounds": dict(sorted(collections.Counter(by[i]["round"] for i in ids).items()))}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
