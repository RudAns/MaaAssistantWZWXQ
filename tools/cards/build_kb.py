"""构建卡牌机制知识库 data/card_kb.json：收集 tools/cards/kb/*.py 里主 agent 手写的条目，
校验（覆盖率、词表），附上卡面原文和用到的关键词原文。

  python tools/cards/build_kb.py            # 构建 + 覆盖率报告
  python tools/cards/build_kb.py --check    # 另跑一致性检查（资源闭环）

"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "tools" / "cards" / "kb"
sys.path.insert(0, str(KB))
from schema import RESOURCES, ROUTES, TRIGGERS  # noqa: E402

# 不是"某张卡产出的资源"，而是以己方场上单位为代价的动作 —— 闭环检查里不算缺口
ACTION_COSTS = {"移除英雄": "以己方场上英雄为代价（遗志/传承/体面离场/听天由命/限制性进化/混沌裂隙），不需要产出者"}

def lord_scope_by_id() -> dict[str, tuple[str, str]]:
    """官方棋手详情页数据（data/web/棋手牌.raw.json → 每个棋手的 技能/秘技/专属 + 秘技给的牌 + relatedCards）：
    卡牌 id -> (棋手, 来源)。**按 id 对，不按名字**：嬴律的"乘胜追击"(94305) 与天赋"乘胜追击"(636413) 同名不同卡。
    数据页面：roledetail.html?roleId=20 是阿离。"""
    raw = json.loads((ROOT / "data" / "web" / "棋手牌.raw.json").read_text(encoding="utf-8"))
    m = {}
    for lord in raw["lords"]:
        for t in lord.get("talent") or []:
            for c in t.get("cards") or []:
                m[str(c["id"])] = (lord["name"], t["name"])
                for pc in c.get("previewCards") or []:
                    m.setdefault(str(pc["id"]), (lord["name"], t["name"] + "给的牌"))
        for c in lord.get("relatedCards") or []:
            m.setdefault(str(c["id"]), (lord["name"], "关联卡牌"))
    return m


PARTS = ["heroes", "summons", "lords", "effects", "effects5", "talents", "talents3", "equips"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    kw = json.loads((ROOT / "data" / "keywords.json").read_text(encoding="utf-8"))["keywords"]
    entries, errs = {}, []
    lords = lord_scope_by_id()
    scope_fix = []
    for part in PARTS:
        if not (KB / f"{part}.py").exists():
            continue
        for e in importlib.import_module(part).E:
            n = e["name"]
            if n not in cards:
                errs.append(f"[{part}] {n}: 卡牌库里没有这张卡")
                continue
            if n in entries:
                errs.append(f"[{part}] {n}: 重复条目")
            if e["trig"] not in TRIGGERS:
                errs.append(f"[{part}] {n}: 未知触发 {e['trig']}")
            for x in e["gives"] + e["needs"]:
                if x["res"] not in RESOURCES:
                    errs.append(f"[{part}] {n}: 未知资源 {x['res']}")
            for r in e["routes"]:
                if r not in ROUTES:
                    errs.append(f"[{part}] {n}: 未知套路 {r}")
            c = cards[n]
            off = lords.get(str(c.get("id")))
            if off:
                want = "阿离可用" if off[0] == "阿离" else f"棋手:{off[0]}"
                if e["scope"] != want:
                    scope_fix.append((n, e["scope"], want, off[1]))
                    e = dict(e, scope=want, scope_manual=e["scope"],
                             routes=e["routes"] if off[0] == "阿离" or "其他棋手" in e["routes"] else e["routes"] + ["其他棋手"])
                e = dict(e, lord_source=f"{off[0]}·{off[1]}")
                if e.get("unsure") and any(k in e["unsure"] for k in ("专属", "棋手", "能否拿到", "↔", "体系", "是宝藏牌", "嬴律")):
                    e = dict(e, unsure="", unsure_resolved=f"{e['unsure']} → 官方棋手数据：{off[0]}·{off[1]}")
            text = (c.get("skill") or "") + (" ｜觉醒：" + c["awaken"] if c.get("awaken") else "")
            e = dict(e, category=c.get("category"), quality=c.get("quality"), faction=c.get("faction", ""),
                     text=text, keywords={k: v for k, v in kw.items() if k in text}, source=f"tools/cards/kb/{part}.py")
            entries[n] = e
    # 覆盖率：去掉"(效果牌)"重复条目后，每张卡都要有条目
    todo = collections.Counter()
    missing = []
    for n, c in cards.items():
        if n.endswith("(效果牌)") or n in entries:
            continue
        missing.append(n)
        todo[c.get("category")] += 1
    total = sum(1 for n in cards if not n.endswith("(效果牌)"))
    print(f"条目 {len(entries)} / {total}（{len(entries) / total:.0%}）；还没写：{dict(todo)}")
    for x in errs:
        print("  ✗", x)
    print(f"适用范围按官方棋手数据校正 {len(scope_fix)} 条（手写判断与官方不一致）：")
    for n, a, b, src in scope_fix:
        print(f"  - {n}: 手写「{a}」→ 官方「{b}」（{src}）")
    unsure = [(n, e["unsure"]) for n, e in entries.items() if e.get("unsure")]
    print(f"标记为'不确定'的 {len(unsure)} 条（集中待确认）")
    out = {"version": "2026-09-24", "method": "主 agent 逐张读卡面原文手写条目（docs/card-kb.md）",
           "count": len(entries), "entries": entries}
    (ROOT / "data" / "card_kb.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.check:
        check(entries)
    return 1 if errs else 0


def check(entries: dict) -> None:
    """资源闭环：每种被消耗/被依赖的资源都要有产出者；每个套路至少有 2 张卡。"""
    prod = collections.defaultdict(list)
    need = collections.defaultdict(list)
    for n, e in entries.items():
        for x in e["gives"]:
            prod[x["res"]].append(n)
        for x in e["needs"]:
            need[x["res"]].append(n)
    print("\n== 资源闭环 ==")
    for r, users in sorted(need.items()):
        if r in ACTION_COSTS:
            print(f"  {r}: {len(users)} 张 —— {ACTION_COSTS[r]}")
            continue
        print(f"  {r}: 消耗/依赖 {len(users)} 张，产出 {len(prod.get(r, []))} 张" + ("  ✗ 没有产出者" if not prod.get(r) else ""))
    routes = collections.defaultdict(list)
    for n, e in entries.items():
        for r in e["routes"]:
            routes[r].append(n)
    print("\n== 套路规模 ==")
    for r, ns in sorted(routes.items(), key=lambda x: -len(x[1])):
        print(f"  {r}: {len(ns)} 张" + ("  ✗ 太少" if len(ns) < 2 else ""))


if __name__ == "__main__":
    sys.exit(main())
