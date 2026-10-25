"""自检 data/equip_relations.json（装备 × 英雄关系图，2026-10-01）。

检查：
  1. 覆盖率：85 个英雄 × 卡牌库全部装备牌都有分数；items / heroes 与 cards.json 一致
  2. 分数范围：-1 或 0~10
  3. 原文里写"限定远程英雄 / 限定近战英雄"的装备，对不符合射程的英雄必须是 -1
     原文里写"可穿戴给【X】 / 【X】专属 / 仅X英雄穿戴 / 仅1-3阶"的，对不符合的英雄必须是 -1
  4. 物理输出英雄（主C/副C）不会给纯法攻装（dmg_req=mag 且 dmg_share≥0.6）≥5 分；反之法术输出英雄不会给纯物理装 ≥5
  5. 基础装备分数 = 四个成品里的最高分
  6. items 里每条 trigger/condition/effect 的引文（非内部标记）必须真在原文里
  7. 列出 solo_carry_ok=false 的英雄（输出位单列）
  8. 2026-10-01 事故回归：马超 × 自然法杖 < 5，马超 × 大棒 的 best_forge 不是自然法杖
用法：.venv/Scripts/python.exe -X utf8 tools/check_equip_relations.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = json.loads((ROOT / "data/equip_relations.json").read_text(encoding="utf-8"))
CARDS = json.loads((ROOT / "data/cards.json").read_text(encoding="utf-8"))["cards"]
RECIPES = json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8"))["recipes"]

EQUIPS = {}
for c in CARDS.values():
    if c.get("category") == "装备牌":
        EQUIPS.setdefault(c["name"], c)
HEROES = {c["name"]: c for c in CARDS.values() if c.get("category") == "英雄牌"}

bad: list[str] = []


def fail(msg):
    bad.append(msg)


items, heroes, fit = REL["items"], REL["heroes"], REL["fit"]

# 1. 覆盖率
if set(items) != set(EQUIPS):
    fail("items 与卡牌库装备不一致：多 %s 少 %s" % (sorted(set(items) - set(EQUIPS)), sorted(set(EQUIPS) - set(items))))
if set(heroes) != set(HEROES):
    fail("heroes 与卡牌库英雄不一致：多 %s 少 %s" % (sorted(set(heroes) - set(HEROES)), sorted(set(HEROES) - set(heroes))))
pairs = 0
for h in HEROES:
    row = fit.get(h)
    if row is None:
        fail("fit 缺英雄 %s" % h)
        continue
    for it in EQUIPS:
        if it not in row:
            fail("fit 缺 %s × %s" % (h, it))
        else:
            pairs += 1

# 2. 分数范围
for h, row in fit.items():
    for it, v in row.items():
        s = v.get("score")
        if not isinstance(s, (int, float)) or not (s == -1 or 0 <= s <= 10):
            fail("分数越界 %s × %s = %r" % (h, it, s))
        if not v.get("why"):
            fail("缺理由 %s × %s" % (h, it))

# 3. 穿戴限制（直接读原文，不信 json 里的 restrict 字段）
FACTIONS = {c["faction"] for c in HEROES.values()}
restrict_checked = 0
for it, c in EQUIPS.items():
    t = c["skill"]
    need = None
    if "限定远程英雄" in t:
        need = ("range", "ranged")
    elif "限定近战英雄" in t:
        need = ("range", "melee")
    m = re.search(r"可穿戴给【(\S+?)】|【(\S+?)】专属装备|仅(\S+?)英雄穿戴", t)
    fac = None
    if m:
        fac = m.group(1) or m.group(2) or m.group(3)
        fac = next((f for f in FACTIONS if f.startswith(fac)), fac)
    tier = re.search(r"仅1-(\d)阶英雄可穿戴", t)
    for h, hc in HEROES.items():
        s = fit[h][it]["score"]
        rng = "melee" if (hc["properties"].get("attackDistance") or 0) <= 1 else "ranged"
        should = False
        if need and rng != need[1]:
            should = True
        if fac and hc["faction"] != fac:
            should = True
        if tier and hc["quality"] > int(tier.group(1)):
            should = True
        if should:
            restrict_checked += 1
            if s != -1:
                fail("穿戴限制违例：%s × %s = %s（原文：%s）" % (h, it, s, t[:30]))
        elif s == -1:
            fail("误判不可穿：%s × %s（原文无对应限制）" % (h, it))

# 4. 伤害类型硬错配
mismatch_checked = 0
for h, hv in heroes.items():
    if hv["role"] not in ("carry", "sub_carry"):
        continue
    for it, iv in items.items():
        if not iv.get("dmg_req") or iv.get("dmg_share", 0) < 0.6:
            continue
        if hv["dmg"] in ("phys", "mag") and iv["dmg_req"] != hv["dmg"]:
            mismatch_checked += 1
            s = fit[h][it]["score"]
            if s >= 5:
                fail("伤害类型错配却 ≥5：%s(%s) × %s(%s) = %s" % (h, hv["dmg"], it, iv["dmg_req"], s))

# 5. 基础装备
for b, prods in RECIPES.items():
    for h in HEROES:
        want = max(fit[h][p]["score"] for p in prods)
        got = fit[h][b]["score"]
        if abs(want - got) > 1e-6:
            fail("基础装备 %s × %s = %s，成品最高 %s" % (b, h, got, want))

# 6. 引文
INTERNAL = {"trigger_class", "restrict", "range_cond", "control"}
for it, iv in items.items():
    t = EQUIPS[it]["skill"]
    for k in ("triggers", "conditions", "effects"):
        for x in iv.get(k, []):
            if x["tag"] in INTERNAL:
                continue
            if x["quote"] not in t:
                fail("引文不在原文：%s.%s %s「%s」" % (it, k, x["tag"], x["quote"]))
for h, hv in heroes.items():
    full = json.dumps(HEROES[h], ensure_ascii=False)
    for x in hv.get("team_dependency", []) + hv.get("provides", []):
        if x["text"] not in full:
            fail("英雄引文不在原文：%s「%s」" % (h, x["text"]))

# 7. 单C
no_solo = [h for h, v in heroes.items() if v.get("solo_carry_ok") is False]
out_no = [h for h in no_solo if heroes[h]["role"] not in ("tank", "support")]
if any("solo_carry_ok" not in v for v in heroes.values()):
    fail("有英雄缺 solo_carry_ok")

# 8. 事故回归
mc = fit.get("马超", {})
if mc.get("自然法杖", {}).get("score", 10) >= 5:
    fail("马超 × 自然法杖 仍 ≥5")
if mc.get("大棒", {}).get("best_forge") == "自然法杖":
    fail("马超 × 大棒 仍铸自然法杖")

print("覆盖：%d 英雄 × %d 装备 = %d 对（应为 %d）" % (len(HEROES), len(EQUIPS), pairs, len(HEROES) * len(EQUIPS)))
print("穿戴限制检查 %d 对；伤害类型错配检查 %d 对" % (restrict_checked, mismatch_checked))
print("solo_carry_ok=false：%d 人；其中输出位 %d 人：%s" % (len(no_solo), len(out_no), "、".join(out_no)))
for h in out_no:
    print("  - %s：%s" % (h, heroes[h]["solo_carry_why"]))
print("  坦克/辅助（S1）：%s" % "、".join(h for h in no_solo if h not in out_no))
print("马超 × 自然法杖 = %s；马超 × 大棒 → %s %s" % (mc["自然法杖"]["score"], mc["大棒"]["best_forge"], mc["大棒"]["score"]))
if bad:
    print("\n违例 %d 条：" % len(bad))
    for b in bad[:200]:
        print("  ✗ " + b)
    sys.exit(1)
print("\n全部通过")
