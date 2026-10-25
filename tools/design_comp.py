"""阵容规划器：用本地 Laya 从全量牌库设计一套阵容，写入 data/chess_pieces.json 的 comp 段。

运行时不吃 token——本脚本只在开发/版本更新时运行一次，产出的配置供规则引擎使用。

用法: python -X utf8 tools/design_comp.py [--faction 河洛]（指定派系则跳过 Laya 选派系）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

FACTIONS = ["河洛", "逐鹿", "三分之地", "日落海", "大河流域"]


def load_heroes() -> dict[str, dict]:
    db = json.loads((PROJECT_DIR / "data" / "cards.json").read_text(encoding="utf-8"))
    return {n: c for n, c in db["cards"].items() if c.get("category") == "英雄牌"}


def faction_summary(heroes: dict[str, dict]) -> dict[str, list[str]]:
    """每个派系的可选英雄，按 品质降序/费用升序 排（主力候选靠前）。"""
    by_faction: dict[str, list] = defaultdict(list)
    for name, c in heroes.items():
        f = c.get("faction") or ""
        if f in FACTIONS:
            by_faction[f].append(name)
    for f in by_faction:
        by_faction[f].sort(key=lambda n: (-heroes[n].get("quality", 0),
                                          heroes[n].get("cost", 9)))
    return by_faction


def _kw_names(c: dict) -> list[str]:
    out = []
    for k in c.get("keywords", []):
        out.append(k.get("name", "") if isinstance(k, dict) else str(k))
    return [k for k in out if k]


def describe(name: str, c: dict) -> str:
    pos = "前排" if any("前排" in t for t in c.get("tags", [])) else "后排"
    kw = "、".join(_kw_names(c)[:3])
    return (f"{name}({c.get('cost', '?')}费{pos} 品质{c.get('quality', '?')} "
            f"{kw} {c.get('skill', '')[:30]})")


def laya_rank_factions(by_faction: dict[str, list[str]]) -> list[str]:
    """Laya 给派系排名；失败回退按英雄数量。"""
    try:
        import laya

        router = laya.Router(preload=True)
        state = {"任务": "为自走棋模式准备多套阵容（每局会随机禁用一个派系，需要备选）",
                 "候选派系": {f: f"{len(ns)}名英雄；代表: " + "、".join(ns[:4])
                            for f, ns in by_faction.items()}}
        q = {"rank": {
            "type": "choice",
            "instructions": "哪个派系作为第一主打阵容最稳定？（英雄多、前后排齐全、有成长型主力）",
            "criteria": {f: f for f in by_faction}}}
        ans = router.predict(state, q)["answers"]["rank"]
        first = ans["choice"]
        print(f"Laya 第一派系: {first} (confidence={ans.get('confidence'):.2f})")
        rest = [f for f in by_faction if f != first]
        rest.sort(key=lambda f: -len(by_faction[f]))
        if first in by_faction:
            return [first] + rest
        return sorted(by_faction, key=lambda f: -len(by_faction[f]))
    except Exception as e:
        print(f"Laya 不可用({e})，按英雄数量回退")
        return sorted(by_faction, key=lambda f: -len(by_faction[f]))


def laya_choose_carry(heroes: dict[str, dict], pool: list[str],
                      faction: str) -> str:
    """Laya 从派系池选主力；失败回退第一个后排高品质英雄。"""
    carry_pool = [n for n in pool
                  if not any("前排" in t for t in heroes[n].get("tags", []))] or pool
    try:
        import laya

        router = laya.Router(preload=True)
        state = {"任务": f"为{faction}阵容选择一名主力英雄（重点投资：装备/主角技能/升级）",
                 "候选": [describe(n, heroes[n]) for n in carry_pool[:8]]}
        q = {"carry": {
            "type": "choice",
            "instructions": "谁最适合当主力？（技能有成长/复生/斩杀等后期价值，且为后排输出）",
            "criteria": {n: describe(n, heroes[n]) for n in carry_pool[:8]}}}
        ans = router.predict(state, q)["answers"]["carry"]
        if ans["choice"] in heroes:
            print(f"Laya 主力[{faction}]: {ans['choice']} "
                  f"(confidence={ans.get('confidence'):.2f})")
            return ans["choice"]
    except Exception as e:
        print(f"Laya 选主力失败({e})，回退启发式")
    return carry_pool[0]


def build_comp(heroes: dict[str, dict], faction: str, pool: list[str]) -> dict:
    """单派系阵容：主力 + 前排坦克 + 高品质同派，含升级材料说明。"""
    carry = laya_choose_carry(heroes, pool, faction)
    tanks = [n for n in pool if any("前排" in t for t in heroes[n].get("tags", []))]
    core = [carry] + tanks[:3]
    for n in pool:
        if n != carry and heroes[n].get("quality", 0) >= 4 and n not in core:
            core.append(n)
    return {"faction": faction, "carry": carry, "core": core[:6]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-comps", type=int, default=5,
                    help="设计几套阵容（默认全派系 5 套，运行时按禁用自动换）")
    args = ap.parse_args()

    heroes = load_heroes()
    by_faction = faction_summary(heroes)
    print("派系概况:")
    for f, ns in by_faction.items():
        print(f"  {f}: {len(ns)} 名")

    order = laya_rank_factions(by_faction)
    comps = [build_comp(heroes, f, by_faction[f]) for f in order[:args.max_comps]]
    for c in comps:
        print(f"阵容[{c['faction']}]: 主力={c['carry']} 核心={c['core']}")

    cfg_path = PROJECT_DIR / "data" / "chess_pieces.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("strategy", {})
    cfg["strategy"]["comps"] = comps
    cfg["strategy"].pop("comp", None)
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {cfg_path}: {len(comps)} 套阵容（运行时按本局禁用自动选择）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
