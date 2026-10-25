"""发牌层的实测分布（2026-10-14，推演器第 2 层）：从历史 N 选 1 题统计各品阶效果锦囊 / 发英雄牌里出现各张牌的频率。

  .venv/Scripts/python.exe -X utf8 tools/engine/build_draw_stats.py  → data/engine/draw_stats.json
输出：
  jinnang[tier] = {"n": 题数, "k": 每题选项数分布, "cards": {卡名: 出现次数}}   —— 纯效果牌、同一品阶的 N 选 1（= 开 N 阶效果锦囊）
  heropick[tier] = 同上，纯英雄牌同一品阶（兑换券 / 送英雄）
"""
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    cards = json.loads((ROOT / "data/cards.json").read_text(encoding="utf-8"))
    cards = cards.get("cards", cards) if isinstance(cards, dict) else cards
    cards = cards if isinstance(cards, list) else list(cards.values())
    Q = {}
    for x in cards:
        Q.setdefault(x["name"], (x.get("category"), x.get("quality")))
    out = {"jinnang": {}, "heropick": {}}
    seen = set()
    for f in glob.glob(str(ROOT / "data/logs/live_*/laya_points.jsonl")):
        for l in open(f, encoding="utf-8"):
            try:
                p = json.loads(l)
            except ValueError:
                continue
            if p.get("kind") != "card_pick":
                continue
            names = [n for n in ((p.get("raw") or {}).get("cards") or []) if isinstance(n, str)]
            key = (p.get("run"), p.get("round"), tuple(names))
            if not names or key in seen:
                continue
            seen.add(key)
            cq = {Q.get(n, (None, None)) for n in names}
            cats = {c for c, _ in cq}
            tiers = {q for _, q in cq}
            if len(tiers) != 1 or None in tiers:
                continue
            t = str(tiers.pop())
            kind = "jinnang" if cats == {"效果牌"} else ("heropick" if cats == {"英雄牌"} else None)
            if not kind:
                continue
            d = out[kind].setdefault(t, {"n": 0, "k": collections.Counter(), "cards": collections.Counter()})
            d["n"] += 1
            d["k"][len(names)] += 1
            for n in names:
                d["cards"][n] += 1
    for kind in out:
        for t, d in out[kind].items():
            d["k"] = dict(d["k"])
            d["cards"] = dict(d["cards"].most_common())
    (ROOT / "data/engine/draw_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for kind in out:
        print(kind, {t: (d["n"], len(d["cards"])) for t, d in sorted(out[kind].items())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
