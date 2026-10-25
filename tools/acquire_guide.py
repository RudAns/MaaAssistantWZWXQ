"""凑核心阵容指引：每个开着的主C 的核心件，商店（按棋手等级）和效果牌 / 天赋哪个来得快（2026-10-21：计算哪些效果卡、
棋手等级刷新商店能更快凑出核心阵容，作为指引）。只算概率、不做决定——结果写进题面 / 规范，决定仍交给 Laya / 局面网络 / 经济网络。

模型（和 agent/decision/meta.py 一致）：商店 4 格，每格先按棋手等级抽品阶（TIER_ODDS），再在该品阶英雄里均匀抽一个；
效果牌 / 天赋按卡面在对应池子（阵营 / 品阶 / 关键词）里均匀抽。没算别的玩家从公共牌库拿走的牌（未知），也没算同名剩余张数。

用法：python tools/acquire_guide.py      → docs/lords/阿离-凑核心指引.md、data/lords/阿离/acquire_guide.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.decision import meta  # noqa: E402

CARDS = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
POOL = json.loads((ROOT / "data" / "lords" / "阿离" / "carry_pool.json").read_text(encoding="utf-8"))
FAC_FULL = {"大河": "大河流域", "三分": "三分之地"}
KW = ("整备", "登场", "牺牲", "开团", "闪现")

HEROES = {n: v for n, v in CARDS.items() if v.get("category") == "英雄牌" and "(" not in n and not v.get("lord")}


def tier(n): return int(HEROES.get(n, {}).get("quality") or 0)
def fac(n): return FAC_FULL.get(HEROES.get(n, {}).get("faction") or "", HEROES.get(n, {}).get("faction") or "")
def kws(n): return {k for k in KW if re.search(k + r"\s*[:：]" if k != "闪现" else "闪现", HEROES.get(n, {}).get("skill") or "")}


def pool(pred): return [n for n in HEROES if pred(n)]


def p_any(n_draw: int, size: int) -> float:
    """从 size 个里均匀抽 n_draw 次（可重复），至少抽到某一个的概率。"""
    return 0.0 if size <= 0 else 1 - (1 - 1 / size) ** n_draw


def shop_refresh(t: int, lv: int) -> float:
    return meta.shop_hit_prob(t, lv)


def card_sources(n: str) -> list[tuple[str, float, str]]:
    """(牌名, 一次拿到 n 的概率, 怎么用)"""
    t, f, k = tier(n), fac(n), kws(n)
    out = []
    F = pool(lambda x: fac(x) == f)
    T = pool(lambda x: tier(x) == t)
    short = {"大河流域": "大河", "三分之地": "三分"}.get(f, f)
    if f and f != "无阵营":
        out += [("求贤", 1 - (1 - 1 / max(1, len(F))) ** 3, f"指定任一【{short}】英雄，发现 3 张选 1"),
                ("援军", p_any(4, len(F)), f"指定任一【{short}】英雄，随机 4 张同阵营"),
                (f"{short}征召令", p_any(2, len(F)), "随机 2 张同阵营"),
                ("呼朋唤友", p_any(1, len(F)), f"指定任一【{short}】英雄，随机 1 张"),
                ("二象坍缩", p_any(3, len(T)), f"指定任一 {t} 阶英雄，3 张同阶（另一档是同阵营最高阶）")]
        if t in (2, 3):
            out.append((f"{short}新援", p_any(1, len(pool(lambda x: fac(x) == f and tier(x) in (2, 3)))), "天赋：随机 1 张 2-3 阶同阵营"))
        if t in (3, 4):
            out.append((f"{short}增援", p_any(3, len(pool(lambda x: fac(x) == f and tier(x) in (3, 4)))), "天赋：随机 3 张 3-4 阶同阵营"))
        if t in (4, 5):
            out.append((f"{short}强援", p_any(3, len(pool(lambda x: fac(x) == f and tier(x) in (4, 5)))), "天赋：随机 3 张 4-5 阶同阵营"))
        lower = pool(lambda x: fac(x) == f and t - 2 <= tier(x) <= t - 1)
        if lower:
            out.append(("前浪的荣光", 1 / max(1, len(pool(lambda x: fac(x) == f and tier(x) in (t - 1 + 1, t)))) * 0.5,
                        f"天赋：指定低 1~2 阶的【{short}】英雄（如 {lower[0]}），拿高 1~2 阶的（粗估）"))
    if t:
        out.append((f"{t}阶英雄兑换券", min(1.0, 3 / max(1, len(T))), f"3 选 1 的 {t} 阶（出现就选）"))
        out.append((f"{t}阶英雄", p_any(2, len(T)), f"2 张 {t} 阶"))
    if t in (2, 3):
        out.append(("双人游", p_any(2, len(pool(lambda x: tier(x) in (2, 3)))), "天赋：2 张 2-3 阶"))
    for kw, card, draws, choose in (("整备", "整备高手", 2, False), ("登场", "登场人物", 2, False), ("登场", "乱世", 3, True),
                                    ("牺牲", "牺牲品", 2, False), ("开团", "开团号令", 2, False), ("闪现", "闪现突击者", 2, False),
                                    ("牺牲", "补给线：牺牲", 1, False), ("整备", "补给线：整备", 1, False), ("登场", "补给线：登场", 1, False),
                                    ("开团", "补给线：开团", 3, False)):
        if kw in k:
            K = pool(lambda x, kw=kw: kw in kws(x))
            p = min(1.0, draws / max(1, len(K))) if choose else p_any(draws, len(K))
            out.append((card, p, f"{'3 选 1' if choose else f'{draws} 张'}【{kw}】英雄" + ("（之后每回合 1 张）" if card.startswith("补给线") and card != "补给线：开团" else "")))
    out.append(("高级人才", p_any(1, len(T)), f"随机 1 张（棋手等级+1）阶——棋手 {t - 1} 级时才是 {t} 阶"))
    if t in (3, 4):
        out.append(("控能达人", p_any(1, len(pool(lambda x: tier(x) in (3, 4)))), "天赋：战斗开始能量 ≤3 时每回合 1 张 3-4 阶"))
    out.append(("复印机 / 终极复印机", 1.0, "已经有 1 张后复制（凑同名 / 觉醒用；复印机只限 1-3 阶）"))
    return sorted(out, key=lambda x: -x[1])


def collect_all(core: list[str], lv: int, trials: int = 4000) -> float:
    """固定在 lv 级刷新，核心件每人至少见到 1 张的期望刷新次数（蒙特卡洛；每次刷新 4 格）。"""
    rng = random.Random(7)
    tiers = {n: tier(n) for n in core}
    sizes = {t: len(pool(lambda x, t=t: tier(x) == t)) for t in set(tiers.values())}
    odds = meta.TIER_ODDS.get(lv, {})
    if any(odds.get(t, 0) <= 0 for t in tiers.values()):
        return float("inf")
    qs, ws = list(odds), list(odds.values())
    tot = 0
    for _ in range(trials):
        need, k = set(core), 0
        while need and k < 2000:
            k += 1
            for _s in range(meta.SHOP_SLOTS):
                q = rng.choices(qs, ws)[0]
                for n in [x for x in need if tiers[x] == q]:
                    if rng.random() < 1 / sizes[q]:
                        need.discard(n)
        tot += k
    return tot / trials


def main() -> int:
    out = {"_meta": {"date": "2026-10-21", "model": "meta.TIER_ODDS + 均匀抽；未计别人拿走的牌", "levels": list(meta.TIER_ODDS)}}
    md = ["# 阿离：凑核心阵容指引（商店 vs 效果牌 / 天赋）", "",
          "用途：计算哪些效果卡、棋手等级下刷新商店能更快凑出核心阵容，作为指引。只是参考，决定交给 Laya / 局面网络 / 经济网络。",
          "模型：商店 4 格，按棋手等级抽品阶、再在该品阶均匀抽；效果牌按卡面在对应池子均匀抽；**没算别人拿走的牌**。由 `tools/acquire_guide.py` 生成。", ""]
    for c, v in POOL["carries"].items():
        if not v.get("enabled"):
            continue
        core = [n for n in dict.fromkeys(v.get("core_lineup") or []) if n in HEROES]
        rows, pieces = [], {}
        for n in core:
            t = tier(n)
            per = {lv: shop_refresh(t, lv) for lv in meta.TIER_ODDS if t}
            best_lv = max(per, key=lambda lv: per[lv]) if per else None
            srcs = card_sources(n)
            pieces[n] = {"tier": t, "faction": fac(n), "keywords": sorted(kws(n)),
                         "shop_per_refresh": {str(k): round(x, 4) for k, x in per.items()},
                         "best_shop_level": best_lv, "cards": [{"card": a, "p": round(b, 3), "how": h} for a, b, h in srcs[:8]]}
            top = "；".join([f"{a} {b:.0%}" for a, b, _h in srcs if not a.startswith("复印机")][:4])
            rows.append(f"| {n} | {t} | {fac(n)} | " + " / ".join(f"{per.get(lv, 0):.1%}" for lv in (2, 3, 4, 5)) +
                        f" | {best_lv} 级（约 {1 / per[best_lv]:.0f} 次）| {top} |" if per else f"| {n} | ? | | | | |")
        allc = {lv: collect_all(core, lv) for lv in (3, 4, 5, 6)}
        out[c] = {"core": core, "pieces": pieces, "collect_all_expected_refreshes": {str(k): (None if x == float("inf") else round(x, 1)) for k, x in allc.items()}}
        md += [f"## {c}（{v.get('tier')} 阶 {v.get('faction')}）", "",
               "| 核心件 | 阶 | 阵营 | 每次刷新见到：2 / 3 / 4 / 5 级 | 商店最好在 | 效果牌 / 天赋一次拿到的概率（前 4） |",
               "|---|---|---|---|---|---|"] + rows + ["",
               "全套核心（每人至少见到 1 张）期望刷新次数：" + "；".join(
                   f"{lv} 级 {'刷不出' if x == float('inf') else f'{x:.0f} 次'}" for lv, x in allc.items()), ""]
    (ROOT / "data" / "lords" / "阿离" / "acquire_guide.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "docs" / "lords" / "阿离-凑核心指引.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
