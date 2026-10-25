"""从 r9r（含更早各轮）已标的 carry / swap 题里，挑出和 g9p 第 41 节（阿离单主C）可能冲突的旧标签，送去重标（2026-10-08）。

  .venv/Scripts/python.exe -X utf8 tools/laya/select_r9s_relabel.py

只看"老师当时把概率放在哪"和题面原文（不读规则引擎），启发式偏宽 —— 选进来的题由老师按 g9p 重新判断，不是直接改标签。

carry（规则 C1）：现任主C "已投入"（2 阶以上、不是马超/镜、不是 `past it` 的过渡件，且 lv ≥10 / 身上 ≥2 件装备 / 枫叶舞绑在他身上 / 已觉醒），
  而旧标签在"换给低等级的人"上放了 ≥0.3 概率（候选没觉醒且等级 < 现任 1.5 倍，或觉醒了但等级 < 现任 0.7 倍；`lv?` 的候选按低等级算）；枫叶舞绑在候选身上（换过去不卖人）的不算。
swap：
  S1 旧标签把 ≥0.3 概率放在卖 `MAIN TANK` 上；
  S2 旧标签的 best 是卖"喂主C 的人"（这类选项合计 ≥0.3）（选项写 `with current carry: <主C> +N lv`），而同题有 `EXTRA OUTPUT` / `2nd carry` 的可卖对象；
  S3 等着上场的是第二主C 型英雄（hero_roles：is_second_carry_risk 或 carry_value ≥7，且不喂主C 等级），主C 已定（题面有 `current carry` / `core X`），
     不是 FUTURE CARRY / 路线主C，而旧标签把 ≥0.6 概率放在"卖人让它上"上（它出现在路线 / 阵容 / 觉醒行里的不算）。
输出 data/laya/datasets/r9s/relabel_ids.json + relabel_info.json（每题命中的规则）。
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DS = PROJECT_DIR / "data" / "laya" / "datasets"

KEEP_RE = re.compile(r"keep (\S+?): t(\d) fit(\d+) lv(\d+|\?)")
SW_RE = re.compile(r"switch to (\S+?): t(\d) fit(\d+) lv(\d+|\?)")
NOT_CARRY = {"马超", "镜"}


def hero_roles() -> dict:
    h = json.loads((PROJECT_DIR / "data" / "hero_roles.json").read_text(encoding="utf-8"))
    return h.get("heroes", h)


def second_carry_type(name: str, roles: dict) -> bool:
    r = roles.get(name) or {}
    fn = " ".join((r.get("functions") or {}).keys())
    feeds = any(k in fn for k in ("carry_level", "team_level"))
    return (bool(r.get("is_second_carry_risk")) or (r.get("carry_value") or 0) >= 7) and not feeds


def board_line(state: str, name: str) -> str:
    for ln in state.splitlines():
        if ln.strip().startswith(name + " t"):
            return ln
    return ""


def carry_conflict(p: dict, lab: dict) -> str | None:
    crit = p["question"]["criteria"]
    m = KEEP_RE.search(crit.get("keep", ""))
    if not m or m.group(4) == "?":
        return None
    cur, tier, lv = m.group(1), int(m.group(2)), int(m.group(4))
    if tier < 2 or cur in NOT_CARRY or "past it" in crit.get("keep", ""):
        return None
    st = p.get("state") or ""
    bl = board_line(st, cur)
    eq = re.search(r"equip (\d)/3", bl)
    invested = (lv >= 10 or (eq and int(eq.group(1)) >= 2) or f"maple dance bound to: {cur}" in st or "AWAKENED" in bl)
    if not invested:
        return None
    mb = re.search(r"maple dance bound to: (\S+)", st)
    cands = {s.group(1) for t in crit.values() for s in [SW_RE.search(t)] if s}
    if mb and mb.group(1) != cur and mb.group(1) in cands:
        return None                     # 枫叶舞绑在候选身上：换过去不用卖人
    bad = 0.0
    for k, txt in crit.items():
        if k == "keep":
            continue
        s = SW_RE.search(txt)
        if not s:
            continue
        clv = 0 if s.group(4) == "?" else int(s.group(4))
        awk = "AWAKENED" in txt
        if (not awk and clv < 1.5 * lv) or (awk and clv < 0.7 * lv):
            bad += float(lab["dist"].get(k, 0))
    return "C1" if bad >= 0.3 else None


def swap_conflict(p: dict, lab: dict, roles: dict) -> str | None:
    crit = p["question"]["criteria"]
    dist = lab["dist"]
    hits = []
    if sum(float(dist.get(k, 0)) for k, t in crit.items() if k.startswith("s") and "MAIN TANK" in t) >= 0.3:
        hits.append("S1")
    has_extra = any(k.startswith("s") and ("EXTRA OUTPUT" in t or "2nd carry" in t) for k, t in crit.items())
    feeder = sum(float(dist.get(k, 0)) for k, t in crit.items()
                 if k.startswith("s") and re.search(r"with (current )?carry: \S+ \+\d+ (temp )?lv", t))
    best_t = crit.get(lab["best"], "")
    if has_extra and feeder >= 0.3 and re.search(r"with (current )?carry: \S+ \+\d+ (temp )?lv", best_t):
        hits.append("S2")
    inc = ((p.get("raw") or {}).get("incoming")) or ""
    if not inc:
        m = re.search(r"keep the board: (\S+) stays in hand", crit.get("bench", ""))
        inc = m.group(1) if m else ""
    st = p.get("state") or ""
    carry_set = bool(re.search(r"outputs on board \d+: core \S+", st) or "current carry" in st)
    route_carry = bool(inc) and any(inc in ln for ln in st.splitlines()
                                    if any(w in ln for w in ("ROUTE", "plan", "teammates", "lineup", "awakened card", "AWAKENED")))
    if (inc and carry_set and second_carry_type(inc, roles) and not route_carry
            and f"FUTURE CARRY" not in (crit.get("bench", "")) and "AWAKENED card" not in st.split("population", 1)[-1][:200]):
        if sum(float(dist.get(k, 0)) for k in crit if k.startswith("s")) >= 0.6:
            hits.append("S3")
    return "+".join(hits) or None


def main() -> int:
    labs = {}
    for line in (PROJECT_DIR / "data" / "laya" / "labels" / "r6" / "labels_all_r9r.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        labs[r["id"]] = r
    roles = hero_roles()
    pts = [json.loads(l) for l in (DS / "r9r" / "points.jsonl").open(encoding="utf-8")]
    out, why = [], {}
    cnt = collections.Counter()
    for p in pts:
        if p["kind"] not in ("carry", "swap") or p["id"] not in labs:
            continue
        lab = labs[p["id"]]
        r = carry_conflict(p, lab) if p["kind"] == "carry" else swap_conflict(p, lab, roles)
        if r:
            out.append(p["id"])
            why[p["id"]] = {"rule": r, "kind": p["kind"], "split": p["split"], "old_best": lab["best"], "old_guide": lab.get("guide")}
            for x in r.split("+"):
                cnt[(p["kind"], x, "test" if p["split"] != "train" else "train")] += 1
    (DS / "r9s" / "relabel_ids.json").write_text(json.dumps(sorted(out)), encoding="utf-8")
    (DS / "r9s" / "relabel_info.json").write_text(json.dumps(
        {"n": len(out), "by_rule": {"|".join(k): v for k, v in sorted(cnt.items())}, "items": why},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("重标候选", len(out))
    for k, v in sorted(cnt.items()):
        print("  ", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
