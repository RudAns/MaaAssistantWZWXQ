"""定路线 vs 抽到谁用谁：商店概率蒙特卡洛模拟（2026-10-10：用不同等级商店各英雄的刷新概率，
算出最优解再定规则：到底是定一个路线再去凑快，还是抽到哪个主C 用哪个快）。

模型（全部来自 agent/decision/meta.py 的实测常量）：
  - 商店 4 格，每格先按棋手等级抽品阶（TIER_ODDS），再在该品阶**剩余**牌里按张数抽一位英雄；
  - 公共牌库：每品阶英雄数 POOL_HEROES、每位张数 POOL_COPIES；对手每回合从牌库里拿走 OPP_TAKE 张（按同样的等级概率）；
    热门主C 额外被抢（CONTEST：每回合对手多拿走这些英雄的概率）；
  - 收入 meta.energy_income；棋手等级按实战中位轨迹 LEVELS（第 480~520 局）；
  - 买英雄 3 能量，刷新 2 能量；每回合有 1 次免费刷新（实战几乎每回合都有）。
策略：
  S1 认死一个主C（守约），配套从第 1 回合起按它的路线买；
  S2 同体系灵活：配套按河洛体系买，主C 是 守约 / 沈梦溪 / 狄仁杰 / 百里玄策 里先到的那个；
  S3 完全随缘：候选主C（各体系热门）里先到的那个，**主C 定了才开始买它的配套**（之前只买主C 候选）。
指标：主C 到手回合（未到记 99）、≤第 4 回合到手概率、第 8 回合"主C + 配套"拥有数。
用法：.venv/Scripts/python.exe -X utf8 tools/sim_route_vs_flex.py [--n 20000] [--contest 0.3]
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.decision import meta  # noqa: E402

CARDS = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
HEROES = {n: v["quality"] for n, v in CARDS.items() if v.get("category") == "英雄牌" and v.get("quality")}
LEVELS = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4, 10: 4}    # 实战中位轨迹（480~520 局）
OPP_TAKE = 14                                                              # 7 个对手 × 每回合约 2 张
HELUO = ["明世隐", "太乙真人", "钟馗", "程咬金", "铠", "盾山"]
FAMILY2 = {"carries": (sys.argv[sys.argv.index("--carries") + 1].split(",") if "--carries" in sys.argv else ["百里守约", "沈梦溪", "狄仁杰", "百里玄策"]), "sup": HELUO}
FAMILY2_BANNED = {"carries": ["虞姬"], "sup": ["少司缘", "鬼谷子", "大司命", "张良", "瑶", "云中君"]}   # 河洛被禁：大河体系
FAMILIES = {
    "百里守约": HELUO, "沈梦溪": HELUO, "狄仁杰": HELUO, "百里玄策": HELUO,
    "亚连": ["米莱狄", "海诺", "朵莉亚", "太乙真人", "瑶", "雅典娜"],
    "虞姬": ["少司缘", "鬼谷子", "大司命", "张良", "瑶", "云中君"],
}


def new_pool():
    return {n: meta.POOL_COPIES[q] for n, q in HEROES.items()}


def draw(pool, level, rng):
    odds = meta.TIER_ODDS.get(level) or {1: 1.0}
    tiers, w = zip(*odds.items())
    t = rng.choices(tiers, w)[0]
    names = [n for n, q in HEROES.items() if q == t and pool[n] > 0]
    if not names:
        return None
    return rng.choices(names, [pool[n] for n in names])[0]


BANNED = ""                                   # 被禁派系（--ban 河洛）：该派系的英雄不进商店


def sim(strategy, rng, contest, rounds=10, carry_fixed="百里守约"):
    pool = new_pool()
    if BANNED:
        for n in pool:
            if (CARDS[n].get("faction") or "") == BANNED:
                pool[n] = 0
    gold, carry, got_round, copies = 0, None, 99, 0
    owned = set()
    fam2 = FAMILY2 if not BANNED else FAMILY2_BANNED
    cand_carries = {"S1": [carry_fixed], "S2": fam2["carries"], "S3": list(FAMILIES)}[strategy]
    cand_carries = [c for c in cand_carries if pool[c] > 0 or not BANNED]
    base_sup = fam2["sup"]
    for r in range(1, rounds + 1):
        lv = LEVELS[r]
        # 对手先拿牌（含对热门主C 的争抢）
        for _ in range(OPP_TAKE):
            h = draw(pool, lv, rng)
            if h:
                pool[h] -= 1
        for c in ("百里守约",):
            if pool[c] > 0 and rng.random() < contest:
                pool[c] -= 1
        gold += meta.energy_income(r)
        if carry:
            wanted = set(FAMILIES[carry]) - owned
        elif strategy in ("S1", "S2"):
            wanted = set(base_sup) - owned                # 体系已定：配套从第 1 回合起买
        else:
            wanted = set()                                # 随缘：主C 没定，不知道买谁的配套
        free = 1
        while True:
            shop = [draw(pool, lv, rng) for _ in range(meta.SHOP_SLOTS)]
            for h in shop:
                if not h:
                    continue
                if carry is None and h in cand_carries and gold >= 3:
                    gold -= 3
                    pool[h] -= 1
                    carry, got_round, copies = h, r, 1
                    owned.add(h)
                    wanted = set(FAMILIES[carry]) - owned
                elif carry and h == carry and gold >= 3:
                    gold -= 3                              # 主C 同名：合成升星
                    pool[h] -= 1
                    copies += 1
                elif h in wanted and gold >= 3:
                    gold -= 3
                    pool[h] -= 1
                    owned.add(h)
                    wanted.discard(h)
            need = carry is None or bool(wanted) or copies < 3
            if not need:
                break
            if free:
                free = 0
                continue
            if gold >= 2 + 3:                             # 刷完至少还买得起一张英雄
                gold -= 2
                continue
            break
        if r == 8:
            fam = FAMILIES.get(carry) or (base_sup if strategy != "S3" else [])
            r8 = (1 if carry else 0) + len(owned & set(fam))
            c8 = copies
    return got_round, r8, c8


def main():
    global BANNED
    if "--ban" in sys.argv:
        BANNED = sys.argv[sys.argv.index("--ban") + 1]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20000
    contests = [float(sys.argv[sys.argv.index("--contest") + 1])] if "--contest" in sys.argv else [0.0, 0.3, 0.6]
    for contest in contests:
        print(f"== 对手额外争抢守约的概率 {contest}（每回合）")
        for s in ("S1", "S2", "S3"):
            rng = random.Random(20261010)
            res = [sim(s, rng, contest) for _ in range(n)]
            gr = [g for g, _, _ in res]
            print(f"  {s}: 主C 到手回合中位 {st.median(gr):>4}  ≤R4 {sum(g <= 4 for g in gr) / n:.0%}  ≤R6 {sum(g <= 6 for g in gr) / n:.0%}"
                  f"  R10 仍没主C {sum(g == 99 for g in gr) / n:.0%}  第 8 回合主C+配套 {st.mean(x for _, x, _ in res):.2f}/7"
                  f"  第 8 回合主C 张数 {st.mean(c for _, _, c in res):.2f}")


if __name__ == "__main__":
    main()
