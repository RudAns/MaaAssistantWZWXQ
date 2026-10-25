"""Laya 基准测试：延迟 + 决策质量。

改 Laya 相关代码前后都应重跑此脚本对比。

用法:
  .venv/Scripts/python.exe -X utf8 -u tools/bench_laya.py            # 全跑
  .venv/Scripts/python.exe -X utf8 -u tools/bench_laya.py --latency  # 只测延迟
  .venv/Scripts/python.exe -X utf8 -u tools/bench_laya.py --quality  # 只测质量

注意:
  - 权重缓存在 .hf_cache（约 1.7G/checkpoint）。本地只下了 multilingual 和
    typed-decisions，english 没下，别改成 preload=True。
  - 冷启动 26~63s 是加载 checkpoint 到 GPU，不是推理慢。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))

from agent.decision.questions import build_questions  # noqa: E402
from agent.perception.state import GameState, ShopSlot  # noqa: E402

CHECKPOINTS = ("multilingual", "typed-decisions")


# ==================== 测试局面 ====================
# 取自离线回归帧 data/logs/match_170447 的真实读数，外加早期/后期两个变体，
# 目的是让三个局面在「该存钱还是该花钱」上有明显差异——如果模型对三者给出
# 相同答案，说明它没在读 state，而是在输出常量。

def _mk(round_no: int, gold: int, shop, hand, ban="大河") -> GameState:
    st = GameState()
    st.phase, st.round_no, st.gold, st.banned_faction = "prepare", round_no, gold, ban
    st.hand = hand
    st.shop = [ShopSlot(i, n, c, faction=f) for i, (n, c, f) in enumerate(shop)]
    return st


STATES = [
    # 早期穷：能量刚够买一张，正常打法应该存钱发育，不该刷新
    ("早期穷", _mk(2, 6,
                 [("弈星", 3, "逐鹿"), ("牛魔", 3, "三分之地"),
                  ("孙膑", 3, "逐鹿"), ("曜", 3, "逐鹿")],
                 ["枫叶舞"])),
    # 中期富（= frame_0039 实际读数）：能量宽裕，商店有两张同名可合成
    ("中期富", _mk(4, 24,
                 [("弈星", 3, "逐鹿"), ("牛魔", 3, "三分之地"),
                  ("孙膑", 3, "逐鹿"), ("孙膑", 3, "逐鹿")],
                 ["枫叶舞", "速击之枪", "力量腰带", "马可波罗", "小菟助力"])),
    # 后期满手：手牌 7 张逼近上限 10，能量 40，该全力提战力
    ("后期满手", _mk(7, 40,
                  [("李白", 4, "河洛"), ("花木兰", 4, "河洛"),
                   ("百里玄策", 5, "河洛"), ("铠", 3, "河洛")],
                  ["枫叶舞", "小菟送福", "速击之枪", "力量腰带",
                   "马可波罗", "小菟助力", "孙膑"])),
]

# 每个局面的主打派系（英文语义版要用；对应 chess_pieces.json 的 comps 切换结果）
MAIN_FACTION = {"早期穷": "逐鹿", "中期富": "逐鹿", "后期满手": "河洛"}

# 定位表（英文语义版要用）。真实实现应从 cards.json 的 tags 推导，
# 这里只为基准测试手填这几张。
ROLE = {
    "牛魔": "frontline tank", "铠": "frontline tank",
    "花木兰": "frontline fighter", "弈星": "backline mage",
    "孙膑": "backline support", "曜": "backline assassin",
    "李白": "backline assassin", "百里玄策": "backline assassin",
}


# ==================== 英文语义特征版 ====================
# 假设：卡名（"弈星"）对模型是纯噪声，它没见过这个游戏。把专有名词换成
# 可推理的属性（几费/是否契合主派系/定位/已有几张/买不买得起）应该能提升质量。

def to_english_state(st: GameState, label: str) -> dict:
    main = MAIN_FACTION[label]
    shop = []
    for s in st.shop:
        dupes = sum(1 for x in st.shop if x.piece == s.piece) - 1 + st.hand.count(s.piece)
        shop.append({
            "slot": s.index,
            "unit": f"unit_{s.index}",          # 匿名化：卡名不进 state
            "cost": s.cost,
            "affordable": bool(s.cost and st.gold >= s.cost),
            "matches_my_main_synergy": s.faction == main,
            "synergy_is_banned_this_game": s.faction == st.banned_faction,
            "role": ROLE.get(s.piece, "unknown"),
            "copies_i_already_have": dupes,
        })
    return {
        "phase": "shopping phase, I may buy units, reroll the shop, or buy XP",
        "round": st.round_no,
        "gold": st.gold,
        "reroll_cost": 2, "unit_cost": 3, "xp_cost": 2,
        "my_main_synergy": main,
        "synergy_banned_this_game": st.banned_faction,
        "cards_in_hand": len(st.hand),
        "hand_limit": 10,
        "shop": shop,
    }


def to_english_questions(st: GameState, label: str) -> dict:
    e = to_english_state(st, label)
    opts = {}
    for s in e["shop"]:
        opts[f"slot{s['slot']}"] = (
            f"{s['cost']} gold, {s['role']}, "
            f"{'matches' if s['matches_my_main_synergy'] else 'does NOT match'} my main synergy, "
            f"I already hold {s['copies_i_already_have']} copies, "
            f"{'affordable' if s['affordable'] else 'too expensive'}")
    opts["buy_nothing"] = "save gold this round, buy nothing"
    return {
        "buy_which": {
            "type": "choice",
            "instructions": "Which shop slot should I buy to build the strongest board?",
            "criteria": opts},
        "should_refresh": {
            "type": "noul",
            "instructions": "Should I spend 2 gold to reroll the shop looking for units "
                            "that match my main synergy?"},
        "should_level_up": {
            "type": "noul",
            "instructions": "Should I spend 2 gold on XP to raise my unit cap?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgently should I spend gold instead of saving it?",
            "criteria": ["safe, save gold", "spend moderately", "critical, spend everything"]},
    }


def load_router():
    import laya
    t0 = time.perf_counter()
    r = laya.Router()                      # device=None → 自动选 cuda
    r.preload(list(CHECKPOINTS))           # 不要 preload=True（会连带下载 english）
    print(f"[冷启动] {time.perf_counter() - t0:.1f}s  已加载 {r.loaded}\n", flush=True)
    return r


def bench_latency(r, reps: int = 10) -> None:
    """延迟：对照 Laya 官方标称 33ms/题、批量摊薄 7.2ms（T4 上测）。"""
    label, st = STATES[1]
    payload, qs = st.to_laya_state(), build_questions(st)
    print(f"=== 延迟（局面「{label}」，{len(qs)} 题，各跑 {reps} 次取中位）===")
    for ckpt in CHECKPOINTS:
        r.predict(payload, qs, model=ckpt)          # warmup
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            r.predict(payload, qs, model=ckpt)
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        med = ts[len(ts) // 2]
        print(f"  {ckpt:<18} 全套{len(qs)}题 中位 {med:6.1f}ms  "
              f"最快 {ts[0]:6.1f}ms  最慢 {ts[-1]:6.1f}ms  => 每题 {med / len(qs):.1f}ms")

    one = {"buy_which": qs["buy_which"]}
    r.predict(payload, one, model=CHECKPOINTS[0])
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r.predict(payload, one, model=CHECKPOINTS[0])
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    print(f"  {'单题 ' + CHECKPOINTS[0]:<18} 中位 {ts[len(ts) // 2]:6.1f}ms  最快 {ts[0]:6.1f}ms\n")


def bench_quality(r) -> None:
    """质量：buy_which 是 5 选 1，随机基线 0.200——低于它就等于掷骰子。

    同时看 should_refresh / should_level_up 在三个差异极大的局面上是否给出
    不同答案：若恒定 ~0.99，那是"无脑说 yes"，不是高置信判断。
    """
    print("=== 质量（buy_which 5 选 1，随机基线 0.200）===")
    print(f"{'局面':<10} {'输入':<10} {'checkpoint':<18} {'buy_which 选择':<22} "
          f"{'conf':<8} {'refresh':<8} {'levelup':<8}")
    print("-" * 95)
    rows = []
    for label, st in STATES:
        variants = [
            ("中文原样", st.to_laya_state(), build_questions(st)),
            ("英文语义", to_english_state(st, label), to_english_questions(st, label)),
        ]
        for vname, payload, qs in variants:
            for ckpt in CHECKPOINTS:
                a = r.predict(payload, qs, model=ckpt)["answers"]
                bw = a.get("buy_which", {})
                print(f"{label:<10} {vname:<10} {ckpt:<18} "
                      f"{str(bw.get('choice'))[:20]:<22} {bw.get('confidence', -1):<8.3f} "
                      f"{a.get('should_refresh', {}).get('confidence', -1):<8.3f} "
                      f"{a.get('should_level_up', {}).get('confidence', -1):<8.3f}", flush=True)
                rows.append((vname, ckpt, bw.get("confidence", 0)))

    print("\n--- buy_which 平均置信度（随机基线 0.200）---")
    for vname in ("中文原样", "英文语义"):
        for ckpt in CHECKPOINTS:
            vals = [c for v, k, c in rows if v == vname and k == ckpt]
            verdict = "低于随机" if statistics.mean(vals) < 0.2 else "勉强过线"
            print(f"  {vname} + {ckpt:<18} {statistics.mean(vals):.3f}  {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", action="store_true", help="只测延迟")
    ap.add_argument("--quality", action="store_true", help="只测质量")
    args = ap.parse_args()
    run_all = not (args.latency or args.quality)

    r = load_router()
    if run_all or args.latency:
        bench_latency(r)
    if run_all or args.quality:
        bench_quality(r)


if __name__ == "__main__":
    main()
