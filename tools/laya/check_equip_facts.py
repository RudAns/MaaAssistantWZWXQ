"""装备类选项题面覆盖检查（2026-10-14：防止装备事实反复漏标）。

拿一道真实的天赋题（默认第 626 局 R4，主C 守约物理），把三个选项逐个换成卡库里的每一件装备，按指定题面版本渲染，
检查每件装备的选项文字里有没有：
  dmg   伤害类型 / 是否对口（magic item / physical item / fits / does NOT fit / forges for / MISMATCH）
  full  主C 装备件数（carry X equips k/3）
  grow  卡面有"永久 / 每击败 / 每回合"成长时，选项里写出成长（permanent / +N per）
  lock  卡面"无法拆卸"时写出 locked / cannot remove
缺项的装备列出来。**训练前必须跑，全过才训练**。

  .venv/Scripts/python.exe -X utf8 tools/laya/check_equip_facts.py --ver f9ln
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "laya"))
sys.path.insert(0, str(ROOT))
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

CHECKS = {
    "dmg": r"magic item|physical item|\bfits\b|does NOT fit|forges for|MISMATCH|mixed item|no damage type",
    "full": r"carry \S+ equips (\d|\?)/3",
    "grow": r"permanent|\bper (kill|round)\b|永久|每击败|每回合",
    "lock": r"locked|cannot remove|无法拆卸",
}


def cards():
    c = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))
    c = c.get("cards", c) if isinstance(c, dict) else c
    c = c if isinstance(c, list) else list(c.values())
    return [x for x in c if isinstance(x, dict) and x.get("category") == "装备牌"]


def base_point(run: str, rnd: int):
    for line in (ROOT / "data" / "logs" / run / "laya_points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        if p.get("kind") == "talent" and p.get("round") == rnd:
            p.setdefault("run", run)
            return p
    raise SystemExit("找不到基准题")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="f9ln")
    ap.add_argument("--run", default="live_1791909444")
    ap.add_argument("--round", type=int, default=4)
    a = ap.parse_args()
    F.ACTIVE = a.ver
    F._facts()
    base = base_point(a.run, a.round)
    miss = collections.defaultdict(list)
    n = 0
    for c in cards():
        name, txt = c["name"], (c.get("skill") or c.get("desc") or "")
        p = copy.deepcopy(base)
        tl = (p.get("raw") or {}).get("talents") or []
        if not tl:
            continue
        tl[0] = [name, txt]
        p["raw"]["talents"] = tl
        try:
            r = RR.rerender(p)
        except Exception as e:            # noqa: BLE001
            miss["render_error"].append(f"{name}: {e}")
            continue
        if not r:
            miss["render_none"].append(name)
            continue
        opt = r["question"]["criteria"].get("opt0", "")
        n += 1
        need = ["dmg", "full"]
        if re.search(r"永久|每击败|每回合|每场", txt):
            need.append("grow")
        if "无法拆卸" in txt:
            need.append("lock")
        for k in need:
            if not re.search(CHECKS[k], opt):
                miss[k].append(f"{name}: {opt[:120]}")
    print(f"题面版本 {a.ver}：渲染 {n} 件装备")
    bad = 0
    for k, v in miss.items():
        bad += len(v)
        print(f"\n## 缺 {k}：{len(v)} 件")
        for x in v[:12]:
            print("  -", x)
    print(f"\n合计缺项 {bad}" + ("（全过）" if not bad else "（训练前必须补齐）"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
