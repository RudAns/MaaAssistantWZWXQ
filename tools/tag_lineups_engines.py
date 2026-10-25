"""给 455 套推荐阵容打等级引擎标签 → data/lineup_engines.json（只是给人看/核查用；规则引擎在线用 engines.lineup_engine 现算）。

  .venv/Scripts/python.exe -X utf8 tools/tag_lineups_engines.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.decision import engines as E  # noqa: E402


def main() -> int:
    data = json.loads((ROOT / "data" / "lineups.json").read_text(encoding="utf-8"))
    out, cnt = {}, collections.Counter()
    for lu in data.get("lineups", []):
        names = [h["name"] for h in lu.get("heroes", []) if h.get("name")]
        stage = [h for st in lu.get("stages", []) for h in st.get("main", [])]
        members = tuple(dict.fromkeys(names + stage))
        key, score = E.lineup_engine(members)
        st = E.status(key, members) if key else None
        out[lu.get("key") or lu.get("name")] = {
            "name": lu.get("name"), "carry": lu.get("carry"), "lords": lu.get("lord_names"),
            "engine": key, "engine_name": st["name"] if st else "", "score": score,
            "producers": st["producers"] if st else [], "amplifiers": st["amplifiers"] if st else [],
            "payoff": st["payoff"] if st else [], "complete": bool(st and st["complete"]),
        }
        cnt[key or "-"] += 1
        if st and st["complete"]:
            cnt["complete:" + key] += 1
    (ROOT / "data" / "lineup_engines.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(cnt), ensure_ascii=False))
    ali = [v for v in out.values() if v["lords"] and "阿离" in v["lords"]]
    print("阿离阵容:", collections.Counter(v["engine"] or "-" for v in ali), "成套:", sum(v["complete"] for v in ali))
    return 0


if __name__ == "__main__":
    sys.exit(main())
