"""推演器事件层的实测分布（2026-10-14）：天赋 / N 选 1 / 拍卖 / 枫叶舞强化 每局每回合出现几次、给出哪些选项。
推演时按"同回合（拿不到就邻近回合）真实出现过的选项"抽，不编造。
  .venv/Scripts/python.exe -X utf8 tools/engine/build_event_offers.py  → data/engine/event_offers.json
输出：
  rate[kind][round]   每局这一回合平均出现几次
  offers[kind][round] [{"opts": [选项名…], "lord": 棋手等级}]  （每回合最多存 400 条）
"""
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("talent", "card_pick", "auction", "lord_boost")


def opts_of(p):
    raw = p.get("raw") or {}
    k = p["kind"]
    if k == "talent":
        return [t[0] if isinstance(t, (list, tuple)) else t for t in raw.get("talents") or []]
    if k == "card_pick":
        return [x for x in raw.get("cards") or [] if isinstance(x, str)]
    if k == "auction":
        return [n for _, n in raw.get("lots") or []]
    if k == "lord_boost":
        return [t[0] if isinstance(t, (list, tuple)) else t for t in raw.get("boosts") or []]
    return []


def main() -> int:
    files = sorted(glob.glob(str(ROOT / "data/logs/live_*/laya_points.jsonl")))[-400:]
    count = {k: collections.Counter() for k in KINDS}
    offers = {k: collections.defaultdict(list) for k in KINDS}
    by_level = {k: collections.defaultdict(list) for k in KINDS}   # 天赋按棋手等级（官方：同一局 6 人在相同等级拿到的天赋品阶相同）
    n_runs = 0
    for f in files:
        seen = set()
        got = False
        for l in open(f, encoding="utf-8"):
            try:
                p = json.loads(l)
            except ValueError:
                continue
            k, R = p.get("kind"), p.get("round")
            if k not in KINDS or not isinstance(R, int) or not 0 < R < 25:
                continue
            o = opts_of(p)
            if not o:
                continue
            key = (k, R, tuple(o))
            if key in seen:
                continue
            seen.add(key)
            got = True
            count[k][R] += 1
            lv = (p.get("ctx") or {}).get("level")
            if len(offers[k][R]) < 400:
                offers[k][R].append({"opts": o, "lord": lv})
            if isinstance(lv, int) and len(by_level[k][lv]) < 600:
                by_level[k][lv].append({"opts": o, "round": R})
        n_runs += got
    out = {"n_runs": n_runs,
           "rate": {k: {R: round(v / max(1, n_runs), 3) for R, v in sorted(c.items())} for k, c in count.items()},
           "offers": {k: {R: v for R, v in sorted(d.items())} for k, d in offers.items()},
           "offers_by_level": {k: {lv: v for lv, v in sorted(d.items())} for k, d in by_level.items()}}
    (ROOT / "data/engine/event_offers.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("局数", n_runs)
    for k in KINDS:
        print(k, {R: out["rate"][k][R] for R in list(out["rate"][k])[:16]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
