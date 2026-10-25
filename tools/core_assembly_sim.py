"""核心阵容"本可以多快"（2026-10-13，只读审计，配合 core_assembly_audit.py 的输出）。

三组蒙特卡洛，都只按 meta 的商店概率 / 牌池 / 收入 / 经验算（满池、不算别家拿牌、不算拍卖 / N 选 1 / 抽英雄牌 —— 这些只会更快）：
  A. 实际节奏重放：每回合用该局**实际**的棋手等级、实际看店次数（1 自然 + 免费刷新 + 付费刷新），只把运气平均掉
     → "按我们的节奏，期望第几回合凑齐"，和实际凑齐回合比 = 运气 + 看到没买。
  B. 实际节奏 + 把每回合准备阶段结束时**剩下没花的能量**全用来付费刷新（拍卖前一回合留 5）
     → A 与 B 的差 = "钱攒着没用来刷"损失的回合。
  C. 经济最优：从第 1 回合起完整模拟收入 / 经验 / 升级 / 刷新，在一组策略里挑凑齐最快的
     （策略 = 先升到 L* 级再把钱全刷；前 3 回合按实际留出买杂牌的钱）。
     → A 与 C 的差 = 节奏（升级时机 + 刷新量）损失的回合。

  .venv/Scripts/python.exe -X utf8 tools/core_assembly_sim.py
"""
from __future__ import annotations

import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from agent.decision import meta  # noqa: E402
import core_assembly_audit as A  # noqa: E402

XP_NEED = {1: 7, 2: 13, 3: 17, 4: 23, 5: 27}
LAST = 12
NEVER = 13
N = 2000
FILLER = {1: 6, 2: 3, 3: 3}        # 前 3 回合买杂牌 / 开局件的钱（实际前三名 R1~R3 每回合买 1.9/1.5/1.1 张英雄）
FREE_PER_ROUND = 1                 # 每回合免费刷新（实际 0.6~1.5 次可用）


def p_view(units, lv):
    miss = 1.0
    for u in units:
        miss *= 1 - meta.shop_hit_prob(A.q(u), lv)
    return 1 - miss


def replay(groups, levels, views, rng, n=N):
    out = []
    for _ in range(n):
        have = set()
        done = NEVER
        for rd in range(1, LAST + 1):
            lv = levels.get(rd) or 1
            for _v in range(views.get(rd, 1)):
                for i, g in enumerate(groups):
                    if i not in have and rng.random() < p_view(g, lv):
                        have.add(i)
            if len(have) == len(groups):
                done = rd
                break
        out.append(done)
    return out


def econ(groups, target_lv, rush_from, rng, n=N, keep_auction=True):
    """完整经济模拟：第 rush_from 回合起只要买得起就升级直到 target_lv；之后（以及之前的余钱）全部付费刷新找缺的核心。"""
    out = []
    for _ in range(n):
        E, L, xp, have, done = 0, 1, 0, set(), NEVER
        for rd in range(1, LAST + 1):
            E += meta.energy_income(rd)
            spend_filler = min(E, FILLER.get(rd, 0))
            E -= spend_filler
            # 升级
            while rd >= rush_from and L < target_lv and L < 6 and E >= XP_NEED[L] - xp:
                E -= XP_NEED[L] - xp
                xp = 0
                L += 1
            if L < meta.level_target(rd) and rd < rush_from:      # 没开冲之前按进度表走（不落后）
                while L < meta.level_target(rd) and E >= XP_NEED[L] - xp:
                    E -= XP_NEED[L] - xp
                    xp = 0
                    L += 1
            keep = 5 if (keep_auction and meta.auction_after_battle(rd)) else 0
            views = 1 + FREE_PER_ROUND
            v = 0
            while True:
                for i, g in enumerate(groups):
                    if i not in have and E >= 3 and rng.random() < p_view(g, L):
                        have.add(i)
                        E -= 3
                v += 1
                if len(have) == len(groups):
                    break
                if v < views:
                    continue
                if E - 2 >= 3 + keep:
                    E -= 2
                    continue
                break
            if len(have) == len(groups):
                done = rd
                break
            xp += meta.lord_xp_gain(rd)
            while L < 6 and xp >= XP_NEED[L]:
                xp -= XP_NEED[L]
                L += 1
        out.append(done)
    return out


def stat(v):
    v = sorted(v)
    mean = sum(min(x, NEVER) for x in v) / len(v)
    return {"mean": round(mean, 2), "med": statistics.median(v), "le6": round(sum(x <= 6 for x in v) / len(v), 2),
            "le8": round(sum(x <= 8 for x in v) / len(v), 2), "le10": round(sum(x <= 10 for x in v) / len(v), 2)}


def best_policy(groups, rng):
    best = None
    for tl in (2, 3, 4, 5):
        for rf in (2, 3, 4, 5, 6):
            s = stat(econ(groups, tl, rf, rng, n=600))
            if best is None or s["mean"] < best[0]["mean"]:
                best = (s, tl, rf)
    s = stat(econ(groups, best[1], best[2], rng))
    return s, best[1], best[2]


def main() -> int:
    rng = random.Random(7)
    D = json.loads((ROOT / "data" / "analysis" / "core_assembly_560_617.json").read_text(encoding="utf-8"))
    D = [m for m in D if m["rank"] and m["std"] and m["core"]]
    # 路线级：经济最优 vs 进度表
    print("== 路线：经济最优策略（先升到 L* 再全刷）vs 只按进度表升级 + 余钱全刷 ==")
    pol = {}
    for k, r in A.ROUTES.items():
        if not r.get("std"):
            continue
        g = r["core"]
        s, tl, rf = best_policy(g, rng)
        sched = stat(econ(g, 1, 99, rng))
        pol[k] = (tl, rf, s)
        tiers = "+".join("/".join(f"{x}{A.q(x)}" for x in grp) for grp in g)
        print(f"{r['name']}（{tiers}）最优 L*={tl} 第{rf}回合起冲：{s}；进度表：{sched}")
    print()
    print("== 逐局：实际凑齐 vs A 实际节奏期望 vs B 余钱全刷 vs C 经济最优 ==")
    rows = []
    for m in D:
        g = [c["units"] for c in m["core"]]
        lv = {r["round"]: (r["lv1"] or r["lv0"] or 1) for r in m["rounds"]}
        vw = {r["round"]: 1 + r["free_ref"] + r["paid_ref"] for r in m["rounds"]}
        vw2 = {}
        for r in m["rounds"]:
            keep = 5 if meta.auction_after_battle(r["round"]) else 0
            extra = max(0, ((r["g1"] or 0) - keep - 3)) // 2 if (r["g1"] or 0) - keep >= 5 else 0
            vw2[r["round"]] = vw[r["round"]] + extra
        sa = stat(replay(g, lv, vw, rng))
        sb = stat(replay(g, lv, vw2, rng))
        sc = pol.get(m["main"], (None, None, None))[2]
        actual = m["owned_all"] or NEVER
        rows.append({"match": m["match"], "rank": m["rank"], "route": m["main_name"], "actual_owned": actual,
                     "actual_board": m["assembled"] or NEVER, "A": sa, "B": sb, "C": sc,
                     "views_r1_8": sum(vw.get(i, 0) for i in range(1, 9)), "unspent_r1_8": sum((r["g1"] or 0) for r in m["rounds"] if r["round"] <= 8),
                     "lv_r5": lv.get(5), "lv_r7": lv.get(7)})
        print(f"#{m['match']} 名次{m['rank']} {m['main_name']}: 实际拿齐 R{actual} 上场齐 R{m['assembled'] or NEVER} | "
              f"A {sa['mean']} | B {sb['mean']} | C {sc['mean'] if sc else '-'} | R1~8 看店 {rows[-1]['views_r1_8']} 次、准备结束剩能量合计 {rows[-1]['unspent_r1_8']}")
    out = ROOT / "data" / "analysis" / "core_assembly_sim_560_617.json"
    out.write_text(json.dumps({"routes": {k: {"target_lv": v[0], "rush_from": v[1], "stat": v[2]} for k, v in pol.items()},
                               "matches": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出", out)
    for grp, f in (("前三", lambda r: r["rank"] <= 3), ("4~6", lambda r: r["rank"] > 3)):
        rs = [r for r in rows if f(r)]
        if not rs:
            continue
        av = lambda key: round(sum(key(r) for r in rs) / len(rs), 2)  # noqa: E731
        print(f"{grp} n={len(rs)}: 实际拿齐 {av(lambda r: min(r['actual_owned'], NEVER))} / 上场齐 {av(lambda r: min(r['actual_board'], NEVER))} | "
              f"A {av(lambda r: r['A']['mean'])} | B {av(lambda r: r['B']['mean'])} | C {av(lambda r: r['C']['mean'] if r['C'] else NEVER)} | "
              f"R1~8 看店 {av(lambda r: r['views_r1_8'])} | 剩能量 {av(lambda r: r['unspent_r1_8'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
