"""**升级节奏 × 主C 品阶 的数学模型（v1）**：该卡在几级搜牌、什么时候升本。

问题（2026-09-24）：主C 是 3 阶的话，商店 3~4 级刷出概率最高；但永远卡在 3~4 级打不过对面 ——
觉醒之后英雄等级更多是靠各种效果牌叠加；升级还能拿到更好的棋手天赋，数值很高。这些都要根据获取概率算出数学模型。

  python tools/strategy/plan_model.py --sims 1500      # -> data/strategy/plan_model.json + 打印对比表

## 模型（每一局蒙特卡洛模拟 14 个回合）

每回合：发收入 → 自然经验（战斗 +1，拍卖回合 +3）→ 按节奏花钱升级（升级弹天赋）→ 剩下的钱开店/刷新：
  - 见到主C 就买（合成 +5 级）；见到场上已有的英雄就买（合成 +5 级）；场面没满就买新人头；
  - **每合成 3 次送 1 个效果锦囊**，拍卖回合（4/8/12 开头）再送 1 个；
  - 锦囊 = 从某品阶的效果牌里 N 选 1（3 张里挑最好的）；
  - 天赋 = 升级时弹同一档的 3 个选项挑最好的，档次按**实测**的"升到几级 → 出几档"概率；
  - 买了 4 张主C（本体 + 3 张）= 觉醒。
对手：7 家，每家每回合按同等级品阶概率从公共牌库拿 2 张（估计值）。

## 参数来源（【人工】= 人工说明，【实测】= 日志/截图，【假设】= 没有数据、可调）

| 参数 | 值 | 来源 |
|---|---|---|
| 合成一张同名 | +5 级 | 【人工】按 5 算 |
| 锦囊 | 每 3 次合成 1 个 + 拍卖回合 1 个 | 【人工】 |
| 锦囊品阶 | 按当前棋手等级的商店品阶概率 | 【假设】（攻略："2、3 阶锦囊都能开出遗志""阿离升 2 后才有概率出遗志"） |
| 天赋档次 | 升到 2 级 37/46/18%、3 级 24/41/34%、4 级 0/52/48%、5 级 3 档 100% | 【实测】270 次弹窗，每次三个选项同档 |
| 天赋/效果牌的价值 | 从卡面文字解析：等级 +N、全体/随机 N 名、获得能量、棋手经验、持续回合 | 【实测】卡牌库文字；**条件一律按满足算（偏乐观）** |
| 商店品阶概率、牌库张数、收入、经验 | meta.py | 【实测】 |
| 英雄强度 vs 等级 | 技能四档（1/10/40/100 级）× (1 + 等级/50) | 技能四档来自 hero_power.py（命中人数【假设】）；属性成长【假设】 |
| 觉醒 | 强度 ×1.5 | 【假设】（卡面"词条效果翻倍"，但不是战斗属性翻倍） |
| 阵容强度 | 主C 强度 + 0.5 × 其他人强度之和 | 【假设】 |
| 枫叶舞 | 基础 0（强化靠锦囊里的翩跹舞步）；主C 到手前的等级按红叶染锋留 50% | 【校准】--calibrate 对齐 24 局实测"场上最高英雄等级"，0 误差最小 |
| 养成引擎 | growth_routes.py：138 张卡的一次性/每回合/临时等级产出，英雄引擎上场后持续生效 | 【主 agent 读卡】 |
| 临时等级 | 只加当回合战斗强度，不累积 | 【人工】战斗中有用，有些套路就是靠叠临时等级 |
| 攒钱 | 下回合要按节奏升级时，先留够升级费再刷新 | 【假设】（正常玩家行为） |

输出：每种主C 品阶 × 每种升级节奏，第 8 / 11 / 14 回合的主C 等级、阵容强度（均值）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "strategy"))
from agent.decision import meta  # noqa: E402
from hero_power import power_table  # noqa: E402
from growth_routes import R as ROUTES, value as route_value  # noqa: E402

XP_NEED = {1: 7, 2: 13, 3: 17, 4: 23, 5: 27}
MERGE_LEVELS = 5
MERGES_PER_KIT = 3
OPPONENTS, OPP_BUYS = 7, 2
ROUNDS = 14
REFRESH, BUY = 2, 3
TALENT_TIER_ODDS = {2: {1: .37, 2: .46, 3: .17}, 3: {1: .24, 2: .41, 3: .35}, 4: {2: .52, 3: .48},
                    5: {3: 1.0}, 6: {3: 1.0}}
AWAKEN_MULT = 1.5
# 枫叶舞（阿离）：绑定的英雄每击杀永久 +1（翩跹舞步·蓄势）、每回合 +2（·新生）。没有逐回合实测，
# 用"每回合给绑定者 MAPLE_PER_ROUND 级"近似，按日志里场上最高英雄等级的增速校准（见 --calibrate）。
MAPLE_PER_ROUND = 0.0         # 校准结果（--calibrate）：翩跹舞步·蓄势/新生已作为 5 阶效果牌进了锦囊池，基础值取 0 与实测最吻合
MAPLE_START = 3             # 第几回合起有稳定的击杀/强化
RED_LEAF_KEEP = 0.5         # 红叶染锋：换绑时保留 50%（限一次）；主C 没到手前的枫叶舞等级按这个比例带过去
OTHERS_W = 0.5

# 主C 代表（每个品阶挑技能评分最高、且是远程输出的一位；可以用 --carry 换）
CARRY_BY_TIER = {1: "孙尚香", 2: "百里守约", 3: "安琪拉", 4: "蒙犽", 5: "鲁班七号"}

# 升级节奏：{回合: 目标等级}
PLANS = {
    "速升(2/3/5/8/11)": {2: 2, 3: 3, 5: 4, 8: 5, 11: 6},
    "标准(2/4/7/10/13)": {2: 2, 4: 3, 7: 4, 10: 5, 13: 6},
    "卡2级到8": {2: 2, 8: 3, 10: 4, 12: 5, 13: 6},
    "卡3级到10": {2: 2, 4: 3, 10: 4, 12: 5, 13: 6},
    "卡4级到11": {2: 2, 4: 3, 6: 4, 11: 5, 13: 6},
    "卡3级到7再速升": {2: 2, 4: 3, 7: 4, 9: 5, 11: 6},
    "永远卡3级": {2: 2, 4: 3},
    "永远卡4级": {2: 2, 4: 3, 6: 4},
}
REF_PLAN = "标准(2/4/7/10/13)"



def scalar(v: dict) -> float:
    """挑牌时用的一维价值（等级当量）：给主C 的等级最值钱；全场等级、能量（≈买同名合成）、经验打折。"""
    t = v.get("temp_turns", 0)
    return (.3 * v.get("core", 0) + v.get("carry", 0) + .15 * v.get("board", 0) + .8 * v.get("energy", 0) + .6 * v.get("xp", 0)
            + .5 * v.get("temp_carry", 0) * t + .05 * v.get("temp_board", 0) * t)


def load_pools():
    """锦囊/天赋候选池：**按卡牌机制知识库的适用范围**（data/card_kb.json 的 scope，官方棋手数据校正过）只留阿离拿得到的牌。
    原来用正则 OTHER_LORD_RE 排除"其他棋手专属"，把"核心|图腾"也写了进去 —— 日落海、大河两个阵营套路整类被删（2026-09-24 发现）。"""
    cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    kb = json.loads((ROOT / "data" / "card_kb.json").read_text(encoding="utf-8"))["entries"]
    effect = collections.defaultdict(list)
    talent = collections.defaultdict(list)
    for n, c in cards.items():
        cat, q = c.get("category"), c.get("quality") or 0
        if n.endswith("(效果牌)") or kb.get(n, {}).get("scope", "通用") not in ("通用", "阿离可用"):
            continue
        if cat == "效果牌" and 1 <= q <= 5:
            effect[q].append(n)
        elif cat == "天赋牌" and 1 <= q <= 3:
            talent[q].append(n)
    heroes = {n: c for n, c in cards.items() if c.get("category") == "英雄牌"}
    return heroes, effect, talent


def draw_tier(rng, odds):
    x, acc = rng.random(), 0.0
    for t, p in sorted(odds.items()):
        acc += p
        if x <= acc:
            return t
    return max(odds)


def best_of(rng, pool, rounds_left, k=3):
    """N 选 1：从池子里抽 k 张，按养成引擎表（growth_routes.py，主 agent 读卡写的）在剩余回合里的产出挑最好的。
    表里没有的牌（装备箱、经济、功能牌）当作 0 等级产出。"""
    if not pool:
        return None, {}
    cands = [(n, route_value(ROUTES.get(n, {}), rounds_left)) for n in rng.sample(pool, min(k, len(pool)))]
    return max(cands, key=lambda x: scalar(x[1]))


def unit_power(ptab, name, lv, awakened=False):
    p = ptab.get(name)
    if not p:
        base = 800.0                                    # 表里没有（坦克/辅助）：给一个固定的"挡刀"值
    elif lv >= 100:
        base = p["L100"]
    elif lv >= 40:
        base = p["L40"]
    elif lv >= 10:
        base = p["L10"]
    else:
        base = p["L1"]
    return base * (1 + lv / 50) * (AWAKEN_MULT if awakened else 1)


def simulate(carry: str, plan: dict, heroes, effect, talent, ptab, sims: int, seed: int = 0):
    rng = random.Random(seed)
    by_tier = collections.defaultdict(list)
    for n, c in heroes.items():
        by_tier[c.get("quality", 1)].append(n)
    factions = sorted({c.get("faction") for c in heroes.values() if c.get("faction") not in (None, "", "无阵营")})
    cq = heroes[carry]["quality"]
    runs = []
    for _ in range(sims):
        traj = []
        banned = rng.choice([f for f in factions if f != heroes[carry].get("faction")])
        pool = {n: meta.POOL_COPIES[c["quality"]] for n, c in heroes.items() if c.get("faction") != banned}
        gold = xp = merges = 0
        level = 1
        c_copies, c_lv, c_awake = 0, 0, False
        board: dict[str, int] = {}                      # 其他英雄 -> 等级
        bank = 0.0                                      # 主C 还没到手时，给主C 的等级先记着（阿离能搬等级：夺取/红叶染锋/遗志）
        ongoing = []                                    # 持续产出：[per_c, per_b, temp_c, temp_b, per_energy, 到第几回合, 核心/回合]
        core = {"lv": 0.0, "mult": 1.0, "focus": False}  # 阿科米亚核心：只增不减；海诺 ×1.5；定向传输
        engines = set()                                 # 已经在场上生效的引擎英雄
        for r in range(1, ROUNDS + 1):
            gold += meta.energy_income(r)
            if r > 1:
                xp += meta.lord_xp_gain(r - 1)
            kits = 1 if r in meta.AUCTION_ROUNDS else 0
            if r >= MAPLE_START:
                if c_copies:
                    c_lv += MAPLE_PER_ROUND
                else:
                    bank += MAPLE_PER_ROUND * RED_LEAF_KEEP
            temp_c = temp_b = 0.0
            for e in ongoing:
                if r <= e[5]:
                    add_perm(e[0], e[1])
                    temp_c += e[2]; temp_b += e[3]
                    gold += e[4]
                    core["lv"] += e[6] * core["mult"]

            def level_up():
                nonlocal level, xp, gold, c_lv, bank
                level += 1
                tier = draw_tier(rng, TALENT_TIER_ODDS.get(level, {3: 1.0}))
                apply(best_of(rng, talent[tier], ROUNDS - r + 1)[1], r)

            def add_perm(pc, pb):
                nonlocal c_lv, bank
                if c_copies:
                    c_lv += pc
                else:
                    bank += pc
                others = pb - pc
                if board and others > 0:
                    for k in board:
                        board[k] += others / len(board)

            def apply(v, r_now, name=None):
                """一次性部分立刻结算；持续部分（per/temp/per_energy）登记到 ongoing，从下回合起逐回合结算。"""
                nonlocal gold, xp
                e = ROUTES.get(name, {}) if name else None
                if e is not None and name:
                    turns = min(ROUNDS - r_now, e.get("turns", ROUNDS))
                    oc, ob = e.get("once", (0, 0))
                    pc, pb = e.get("per", (0, 0))
                    tc, tb = e.get("temp", (0, 0))
                    gold += e.get("energy", 0)
                    xp += e.get("xp", 0)
                    add_perm(oc, ob)
                    core["lv"] += e.get("core_once", 0) * core["mult"]
                    if e.get("core_mult"):
                        core["mult"] = max(core["mult"], e["core_mult"])
                    if e.get("core_focus"):
                        core["focus"] = True
                    if pc or pb or tc or tb or e.get("per_energy") or e.get("core"):
                        ongoing.append([pc, pb, tc, tb, e.get("per_energy", 0), r_now + max(1, turns), e.get("core", 0)])
                    return
                gold += v.get("energy", 0)
                xp += v.get("xp", 0)
                add_perm(v.get("carry", 0), v.get("board", 0))

            while level < 6 and xp >= XP_NEED[level]:
                xp -= XP_NEED[level]
                level_up()
            want = max([lv for rr, lv in plan.items() if rr <= r] or [1])
            while level < min(want, 6) and gold >= XP_NEED[level] - xp:
                gold -= XP_NEED[level] - xp
                xp = 0
                level_up()
            odds = meta.TIER_ODDS.get(min(level, 6), {1: 1.0})
            for _o in range(OPPONENTS * OPP_BUYS):
                t = draw_tier(rng, odds)
                cands = [n for n in by_tier[t] if pool.get(n, 0) > 0]
                if cands:
                    pool[rng.choice(cands)] -= 1
            cap = meta.pop_cap_at(level)
            # 攒钱：落后于节奏 -> 钱全留给升级；下一次升级在 k 回合后 -> 留"升级费 - 期间收入和白给经验"
            reserve = 0
            if level < 6:
                cost = XP_NEED[level] - xp
                if level < min(want, 6):
                    reserve = cost
                else:
                    nxt = [rr for rr, lv in sorted(plan.items()) if rr > r and lv > level]
                    if nxt:
                        between = range(r + 1, nxt[0] + 1)
                        reserve = max(0, cost - sum(meta.energy_income(x) for x in between)
                                      - sum(meta.lord_xp_gain(x - 1) for x in between))
            first = True
            carry_rollable = cq <= meta.MAX_TIER_BY_LEVEL.get(min(level, 6), 1)
            while first or (gold - reserve >= REFRESH + BUY and carry_rollable and not c_awake):
                if not first:
                    gold -= REFRESH
                first = False
                for _s in range(4):
                    t = draw_tier(rng, odds)
                    cands = [n for n in by_tier[t] for _k in range(pool.get(n, 0))]
                    if not cands or gold < BUY:
                        continue
                    h = rng.choice(cands)
                    if h != carry and gold - reserve < BUY:
                        continue                        # 攒着升级的钱只留给主C
                    if h == carry:
                        gold -= BUY; pool[h] -= 1
                        c_copies += 1
                        if c_copies == 1:
                            c_lv += 1 + bank; bank = 0
                        else:
                            c_lv += MERGE_LEVELS; merges += 1
                        c_awake = c_copies >= 4
                    elif h in board:
                        gold -= BUY; pool[h] -= 1
                        board[h] += MERGE_LEVELS; merges += 1
                    elif len(board) + 1 < cap and gold - reserve >= BUY + (REFRESH + BUY if carry_rollable else 0):
                        gold -= BUY; pool[h] -= 1
                        board[h] = 1
                        eng = ROUTES.get(h)
                        if eng and eng.get("hero") and h not in engines:
                            engines.add(h)              # 引擎英雄上场：它的卡牌效果从下回合起持续产出
                            apply({}, r, h)
            # 卖最弱的给更高品阶的新人腾位：简化成"人口满时，低品阶的会被逐步换掉"不单独模拟
            kits += merges // MERGES_PER_KIT
            merges %= MERGES_PER_KIT
            for _k in range(kits):
                t = min(level, 5)                       # 【实测】锦囊品阶 = min(棋手等级, 5)（132 个锦囊）
                nm, v = best_of(rng, effect[t], ROUNDS - r + 1)
                apply(v, r, nm)
            # 战斗强度：永久等级 + **当场临时等级**（临时等级打完清空、不累积，但当场战斗力全额算）
            # 阿科米亚核心：开战时以**临时等级**平均分给己方日落海英雄，单人上限为核心等级的 50%（关键词原文）；
            # 定向传输：优先集中给装备最多的日落海英雄（这里 = 日落海主C），至多 200
            ctemp = {}
            sun = [n for n in board if heroes.get(n, {}).get("faction") == "日落海"]
            carry_sun = bool(c_copies) and heroes[carry].get("faction") == "日落海"
            if core["lv"] > 0:
                left, units = core["lv"], sun + ([carry] if carry_sun else [])
                if core["focus"] and carry_sun:
                    ctemp[carry] = min(left, 200); left -= ctemp[carry]; units = sun
                if units and left > 0:
                    each = min(left / len(units), .5 * core["lv"])
                    for n in units:
                        ctemp[n] = ctemp.get(n, 0) + each
            cp = unit_power(ptab, carry, c_lv + temp_c + ctemp.get(carry, 0), c_awake) if c_copies else 0.0
            nb = max(1, len(board))
            op = sum(unit_power(ptab, n, lv + (temp_b - (temp_c if c_copies else 0)) / nb + ctemp.get(n, 0))
                     for n, lv in sorted(board.items(), key=lambda x: -x[1])[:cap - (1 if c_copies else 0)])
            traj.append((c_copies > 0, c_lv if c_copies else 0, c_awake, cp + OTHERS_W * op, level,
                         min(cap, len(board) + (1 if c_copies else 0))))
        runs.append(traj)
    return runs


def summarize(runs, ref=None):
    """每回合均值；给了对手参考强度 ref[r] 就再算输赢与掉血。"""
    out = {}
    n = len(runs)
    for r in range(1, ROUNDS + 1):
        rows = [t[r - 1] for t in runs]
        got = sum(x[0] for x in rows)
        out[r] = {"p_carry": round(got / n, 3), "carry_lv": round(sum(x[1] for x in rows) / max(1, got), 1),
                  "p_awake": round(sum(x[2] for x in rows) / n, 3), "strength": round(sum(x[3] for x in rows) / n),
                  "lord_lv": round(sum(x[4] for x in rows) / n, 2), "units": round(sum(x[5] for x in rows) / n, 2)}
    if ref:
        hp_end, alive, dead_round = [], 0, []
        for t in runs:
            hp = HP0
            died = None
            for r in range(1, ROUNDS + 1):
                pw = win_prob(t[r - 1][3], ref[r])
                hp -= (1 - pw) * loss_damage(r)          # 期望掉血（平滑，不再抽样）
                if hp <= 0 and died is None:
                    died = r
            hp_end.append(max(0.0, hp))
            alive += died is None
            dead_round.append(died or ROUNDS + 1)
        out["hp_end"] = round(sum(hp_end) / n, 1)
        out["p_alive_r14"] = round(alive / n, 3)
        out["mean_out_round"] = round(sum(dead_round) / n, 2)
        out["win_r8"] = round(sum(win_prob(t[7][3], ref[8]) for t in runs) / n, 3)
    return out


# 输赢：对手按"标准节奏 × 各品阶主C"的平均强度；强度比 1.2 倍约 73% 赢（【假设】）
HP0 = 60


def win_prob(s, s_ref):
    import math
    return 1 / (1 + math.exp(-(s / max(1.0, s_ref) - 1) / 0.2))


def loss_damage(r):
    return 3 + r                                        # 【假设】输一回合掉 3+回合数 点血


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    heroes, effect, talent = load_pools()
    ptab = power_table()
    pv = lambda pool: {t: (round(sum(scalar(route_value(ROUTES.get(n, {}), 6)) for n in v) / len(v), 1), len(v))  # noqa: E731
                       for t, v in sorted(pool.items())}
    print("效果牌池（剩 6 回合时的等级当量均值 / 张数）：", pv(effect))
    print("天赋池  （剩 6 回合时的等级当量均值 / 张数）：", pv(talent))
    if args.calibrate:
        # 按日志实测（measure_growth.py，24 局）的"场上最高英雄等级"校准枫叶舞基础产出
        target = {8: 21, 11: 31, 13: 48}
        global MAPLE_PER_ROUND
        for m in (0.0, 1.0, 2.0, 3.0):
            MAPLE_PER_ROUND = m
            rs = [summarize(simulate(c, PLANS[REF_PLAN], heroes, effect, talent, ptab, args.sims, seed=7))
                  for c in CARRY_BY_TIER.values()]
            got = {r: sum(x[r]["carry_lv"] for x in rs) / len(rs) for r in target}
            err = sum(abs(got[r] - target[r]) for r in target)
            print(f"枫叶舞每回合 {m}：主C 等级 " + " ".join(f"R{r}={got[r]:.1f}(实测{target[r]})" for r in target) + f"  误差 {err:.1f}")
        return 0
    raw = {}
    for tier, carry in CARRY_BY_TIER.items():
        for i, (pn, plan) in enumerate(PLANS.items()):
            raw[(carry, pn)] = simulate(carry, plan, heroes, effect, talent, ptab, args.sims, seed=100 * tier + i)
    std = [k for k in raw if k[1] == REF_PLAN]
    ref = {r: sum(t[r - 1][3] for k in std for t in raw[k]) / sum(len(raw[k]) for k in std) for r in range(1, ROUNDS + 1)}
    res = {}
    for tier, carry in CARRY_BY_TIER.items():
        res[carry] = {}
        print(f"\n== 主C {carry}（{tier} 阶） ==")
        print(f"{'节奏':16s} | R8 到手 主C级 强度比 | R11 到手 主C级 强度比 | R14 强度比 棋手级 | 第8回合胜率 剩血 活到14 平均出局")
        for pn in PLANS:
            o = summarize(raw[(carry, pn)], ref)
            res[carry][pn] = o
            f = lambda r: f"{o[r]['p_carry']:.2f} {o[r]['carry_lv']:5.1f} {o[r]['strength'] / ref[r]:5.2f}"  # noqa: E731
            print(f"{pn:16s} | {f(8)}   | {f(11)}    | {o[14]['strength'] / ref[14]:5.2f}  {o[14]['lord_lv']:4.1f}  | "
                  f"{o['win_r8']:.2f}  {o['hp_end']:5.1f}  {o['p_alive_r14']:.2f}  {o['mean_out_round']:5.2f}")
    out = ROOT / "data" / "strategy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan_model.json").write_text(json.dumps({"sims": args.sims, "plans": PLANS, "carry_by_tier": CARRY_BY_TIER,
                                                     "ref_strength": ref, "results": res}, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
