"""合并第 1 层抽取结果 → data/engine/effects_v1.json（含规范化、校验、覆盖统计）。

  .venv/Scripts/python.exe -X utf8 tools/engine/merge_effects.py            # 合并 + 报告
  .venv/Scripts/python.exe -X utf8 tools/engine/merge_effects.py --check    # 只报告，不写盘

输入：data/engine/chunks/chunk_XX.json（每批要抽的卡，key 对齐 cards.json）
      data/engine/effects_out/chunk_XX.jsonl（抽取结果，一行一张卡）
规范化：agent/engine/normalize.py（派系名、summon 字段、反应器事件、who=other 推断、天赋送出单位的效果挪位…）
校验：
  1. 覆盖：911 张卡每张都有条目（缺的列出来）
  2. 枚举值：when / effect / target.who 是否在设计文档的取值表里（引擎扩展的 coin_amp / conv_cap / most_equipped /
     named / core / summoned 单列）
  3. 公式：字符串 amount 能否编译（agent/engine/effects.compile_formula），编译不了的计为"公式不可解析"
  4. 关键词覆盖：卡面（skill / awaken）里含 等级 / 临时等级 / 古币 / 触发 / 能量 / 人口 的句子，
     必须有一条非 other 的效果其 raw 与该句重叠 —— 否则列为"漏结构化"
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.engine import effects as E  # noqa: E402
from agent.engine.normalize import LISTS, normalize_records  # noqa: E402

OUT = ROOT / "data" / "engine" / "effects_v1.json"
EXT_EFFECTS = {"coin_amp", "conv_cap"}
EXT_WHO = {"most_equipped", "named", "core", "summoned"}
DOC_WHO = {"self", "carry", "nearest_ally", "nearest_faction", "random_ally", "all_allies", "faction", "keyword_holders",
           "lowest_level", "highest_tier", "adjacent", "chosen", "lord", "other", "highest_level"}
KW = re.compile(r"临时等级|等级|古币|触发|能量|人口")


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"[\n。；]", text or "") if s.strip()]


def _overlap(a: str, b: str) -> bool:
    a, b = re.sub(r"\s", "", a), re.sub(r"\s", "", b)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    # 6 字窗口重叠
    return any(a[i:i + 6] in b for i in range(0, max(1, len(a) - 5)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    want = []
    for f in sorted((ROOT / "data" / "engine" / "chunks").glob("chunk_*.json")):
        want += [x["key"] for x in json.loads(f.read_text(encoding="utf-8"))]
    raw = E.read_out_chunks()
    cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))
    cards = cards.get("cards") or cards
    missing = [k for k in want if k not in raw]
    extra = [k for k in raw if k not in set(want)]
    recs, issues = normalize_records(raw, cards)

    st = collections.Counter()
    bad_enum, bad_formula, unstruct = [], [], []
    by_cat = collections.defaultdict(lambda: collections.Counter())
    for k, r in recs.items():
        cat = r.get("category") or "?"
        by_cat[cat]["cards"] += 1
        n_eff = 0
        for L in LISTS:
            for e in r.get(L) or []:
                n_eff += 1
                st["effects"] += 1
                w, ef, who = e.get("when"), e.get("effect"), (e.get("target") or {}).get("who") or "other"
                if w not in E.WHENS or (ef not in E.EFFECTS and ef not in EXT_EFFECTS):
                    bad_enum.append((k, w, ef))
                if ef in EXT_EFFECTS:
                    st["ext_effect"] += 1
                if who in EXT_WHO:
                    st["ext_who"] += 1
                elif not (who in DOC_WHO or str(who).startswith("enemy")):
                    bad_enum.append((k, "who", who))
                if ef == "other":
                    st["effect_other"] += 1
                    by_cat[cat]["effect_other"] += 1
                if who == "other" and ef not in ("energy", "lord_xp", "refresh", "pop", "card", "hero_card"):
                    st["who_other"] += 1
                a = e.get("amount")
                if isinstance(a, str):
                    st["formula"] += 1
                    if E.compile_formula(a) is None:
                        bad_formula.append((k, a))
        if n_eff == 0:
            st["no_effect_cards"] += 1
            by_cat[cat]["no_effect"] += 1
        # 关键词覆盖：只查卡面 skill / awaken
        c = cards.get(k) or {}
        structured = [e.get("raw") or "" for L in LISTS for e in r.get(L) or [] if e.get("effect") != "other"]
        for field_ in ("skill", "awaken"):
            for s in _sentences(c.get(field_) or ""):
                if not KW.search(s):
                    continue
                by_cat[cat]["kw_sent"] += 1
                st["kw_sent"] += 1
                if any(_overlap(s, x) for x in structured):
                    by_cat[cat]["kw_ok"] += 1
                    st["kw_ok"] += 1
                else:
                    unstruct.append((k, field_, s[:60]))

    rep = {
        "cards_expected": len(want), "cards_extracted": len(raw), "missing": missing, "extra": extra,
        "effects": st["effects"], "effect_other": st["effect_other"], "who_other_unresolved": st["who_other"],
        "formula": st["formula"], "formula_unparsable": len(bad_formula),
        "keyword_sentences": st["kw_sent"], "keyword_structured": st["kw_ok"],
        "keyword_coverage": round(st["kw_ok"] / max(1, st["kw_sent"]), 4),
        "bad_enum": len(bad_enum), "normalize_moves": len(issues),
        "by_category": {k: dict(v) for k, v in sorted(by_cat.items())},
    }
    print(json.dumps({k: v for k, v in rep.items() if k not in ("missing", "extra")}, ensure_ascii=False, indent=1))
    print(f"缺 {len(missing)}：{missing[:20]}")
    if extra:
        print(f"多出 {len(extra)}：{extra[:10]}")
    for x in bad_enum[:15]:
        print("  取值表外：", x)
    for x in bad_formula[:20]:
        print("  公式不可解析：", x)
    print(f"漏结构化 {len(unstruct)} 句（前 25）：")
    for x in unstruct[:25]:
        print("  ", x)
    for x in issues[:10]:
        print("  规范化：", x)
    if not args.check:
        OUT.write_text(json.dumps({"version": "v1-2026-10-14", "doc": "第 1 层效果表（tools/engine/merge_effects.py 生成，"
                                   "已规范化；格式见 docs/engine-v1-design.md）", "report": rep,
                                   "unstructured": unstruct, "formula_unparsable": bad_formula,
                                   "cards": recs}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("写入", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
