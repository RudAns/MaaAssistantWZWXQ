"""前三 vs 第 5~6 名 差异表（读 data/analysis/match_features.jsonl）。

  python tools/analysis/compare_groups.py

每个数值特征给 均值 / 中位数 (n)，以及 AUC = P(前三局的值 > 5~6 名局的值)（0.5 = 无区分，
>0.5 前三更大，<0.5 前三更小）。另附置换检验 p 值（双侧，5000 次，样本小只作参考）。
"""
from __future__ import annotations

import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = [json.loads(line) for line in (ROOT / "data" / "analysis" / "match_features.jsonl").open(encoding="utf-8")]
G = {"top3": [f for f in R if f["rank"] <= 3], "4th": [f for f in R if f["rank"] == 4],
     "5-6": [f for f in R if f["rank"] >= 5]}

KEYS = ["end_round", "carry_arrive_round", "carry_arrive_lv", "carry_set_round", "carry_changes",
        "carry_lv_R5", "carry_lv_R8", "carry_lv_R10", "final_carry_lv_R8", "final_carry_lv_R10",
        "board_lv_R5", "board_lv_R8", "board_lv_R10", "board_n_R5", "board_n_R8",
        "cards_gained", "cards_gained_R1_8", "cards_to_carry", "cards_to_carry_R1_8", "cards_to_carry_R5_10",
        "noncoin_to_carry_R5_10", "coin_to_carry", "equips_to_carry", "carry_items_n", "carry_items_q2plus",
        "forged_total", "forged_to_carry", "buys", "buys_R1_8", "refreshes", "refreshes_R1_8", "levelups",
        "levelups_R1_8", "card_picks", "card_picks_R1_8", "merges", "merges_R1_8", "sells",
        "energy_left_mean_R2_10", "lord_lv3_round", "lord_lv4_round", "lord_lv5_round", "lord_lv6_round",
        "pop_cap_max", "auction_won_n"]


def nums(v):
    return [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]


def auc(a, b):
    a, b = nums(a), nums(b)
    if not a or not b:
        return None
    return sum((x > y) + 0.5 * (x == y) for x in a for y in b) / len(a) / len(b)


def perm_p(a, b, n=5000, seed=0):
    a, b = nums(a), nums(b)
    if len(a) < 2 or len(b) < 2:
        return None
    rnd = random.Random(seed)
    obs = abs(auc(a, b) - 0.5)
    pool = a + b
    hit = 0
    for _ in range(n):
        rnd.shuffle(pool)
        if abs(auc(pool[:len(a)], pool[len(a):]) - 0.5) >= obs - 1e-12:
            hit += 1
    return hit / n


def fmt(v):
    v = nums(v)
    return f"{st.mean(v):.1f} / {st.median(v):.0f} (n={len(v)})" if v else "-"


def main():
    print("分组：", {k: [f["match"] for f in v] for k, v in G.items()})
    rows = []
    for k in KEYS:
        a = [f.get(k) for f in G["top3"]]
        b = [f.get(k) for f in G["5-6"]]
        rows.append((k, fmt(a), fmt([f.get(k) for f in G["4th"]]), fmt(b), auc(a, b), perm_p(a, b)))
    rows.sort(key=lambda t: -abs((t[4] if t[4] is not None else 0.5) - 0.5))
    print("| 特征 | 前三 均值/中位 | 第4 | 5~6名 | AUC | p |")
    print("|---|---|---|---|---|---|")
    for k, a, m, b, u, p in rows:
        print(f"| {k} | {a} | {m} | {b} | {u:.2f} | {p:.3f} |" if u is not None and p is not None else f"| {k} | {a} | {m} | {b} | - | - |")
    # 类别特征
    print("\n最终主C × 名次：")
    by = {}
    for f in R:
        by.setdefault(f["final_carry"], []).append(f["rank"])
    for c, rs in sorted(by.items(), key=lambda t: st.mean(t[1])):
        print(f"  {c}: {len(rs)} 局 平均 {st.mean(rs):.2f} 前三 {sum(r <= 3 for r in rs)} 名次 {sorted(rs)}")
    print("\n禁用派系 × 名次：")
    by = {}
    for f in R:
        by.setdefault(f["ban"], []).append(f["rank"])
    for c, rs in sorted(by.items(), key=lambda t: st.mean(t[1])):
        print(f"  {c}: {len(rs)} 局 平均 {st.mean(rs):.2f} 名次 {sorted(rs)}")
    print("\n天赋出现（局数）：前三 / 第4 / 5~6")
    cnt = {g: Counter(t for f in v for t in set(f["talents"])) for g, v in G.items()}
    allt = set().union(*cnt.values())
    for t in sorted(allt, key=lambda t: -(cnt["top3"][t] / len(G["top3"]) - cnt["5-6"][t] / len(G["5-6"]))):
        print(f"  {t}: {cnt['top3'][t]} / {cnt['4th'][t]} / {cnt['5-6'][t]}")
    print("\n枫叶舞强化出现（局数）：前三 / 第4 / 5~6")
    cnt = {g: Counter(t for f in v for t in set(f["lord_boosts"])) for g, v in G.items()}
    for t in sorted(set().union(*cnt.values()), key=lambda t: -(cnt["top3"][t] / len(G["top3"]) - cnt["5-6"][t] / len(G["5-6"]))):
        print(f"  {t}: {cnt['top3'][t]} / {cnt['4th'][t]} / {cnt['5-6'][t]}")


if __name__ == "__main__":
    main()
