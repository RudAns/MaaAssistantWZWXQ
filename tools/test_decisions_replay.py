"""决策重放：把真实对局录下来的每一帧状态重新喂给规则引擎。

**这是之前一直缺的那个"快速验证手段"**（2026-09-22 定下的下一步）。
`data/logs/live_*/states.jsonl` 存的是完整的 GameState 快照，所以重放
**不需要模拟器、不需要 OCR**，几秒钟跑完几千帧。

它和另外两个测试的分工：
  - `test_perception_offline.py`：像素 -> 状态，要模拟器（OCR 在那边跑）
  - `test_rules_smoke.py`：**构造**的局面 -> 决策，查的是"这种局面该怎么做"
  - 本文件：**真实**局面 -> 决策，查的是**不变量**——
    不崩、不产出执行层吃不下的动作、不在场面满的时候继续买。

为什么不变量比"期望动作"更值钱：真实帧有几千个，一个个写期望不现实；
而"不变量被破坏"恰恰是实战里最贵的那类 bug。2026-09-23 的
`sell_unit` 字段对不上（整局一个英雄都卖不出去）就属于这一类 ——
它在冒烟测试里看不出来，但重放一遍所有真实帧会立刻炸出来。

用法：
    python -X utf8 tools/test_decisions_replay.py            # 跑最近 3 局
    python -X utf8 tools/test_decisions_replay.py --all      # 跑全部
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agent.actions.executor import validate_action  # noqa: E402
from agent.decision.rules import RuleEngine  # noqa: E402
from agent.perception.state import (AuctionLot, AuctionState, BoardCell,  # noqa: E402
                                    GameState, PlayerRow, ShopSlot, TalentOption)

LOG_DIR = PROJECT_DIR / "data" / "logs"


def _rebuild(d: dict) -> GameState:
    """把 states.jsonl 里的一条记录还原成 GameState。

    嵌套的 dataclass（商店格/玩家行/拍卖）要手工还原；
    **字段对不上就跳过那一项**，不要因为录制时的版本差异整帧作废。
    """
    st = GameState()
    shop = d.pop("shop", None) or []
    players = d.pop("players", None) or []
    talents = d.pop("talents", None) or []
    bench = d.pop("bench", None) or []
    board = d.pop("board", None) or []
    auction = d.pop("auction", None)
    for k, v in d.items():
        if hasattr(st, k):
            setattr(st, k, v)
    st.shop = [ShopSlot(**{k: v for k, v in s.items()
                           if k in ShopSlot.__annotations__}) for s in shop]
    st.players = [PlayerRow(**{k: v for k, v in p.items()
                               if k in PlayerRow.__annotations__}) for p in players]
    st.talents = [TalentOption(**{k: v for k, v in t.items()
                                  if k in TalentOption.__annotations__})
                  for t in talents]
    st.bench = [BoardCell(**{k: v for k, v in c.items()
                             if k in BoardCell.__annotations__}) for c in bench]
    st.board = [BoardCell(**{k: v for k, v in c.items()
                             if k in BoardCell.__annotations__}) for c in board]
    if auction:
        lots = auction.pop("lots", None) or []
        auc = AuctionState()
        for k, v in auction.items():
            if hasattr(auc, k):
                setattr(auc, k, v)
        auc.lots = [AuctionLot(**{k: v for k, v in lo.items()
                                  if k in AuctionLot.__annotations__}) for lo in lots]
        st.auction = auc
    return st


def _new_engine() -> RuleEngine:
    return RuleEngine(PROJECT_DIR / "data" / "chess_pieces.json",
                      cards_db_path=PROJECT_DIR / "data" / "cards.json")


def replay(path: Path) -> list[str]:
    """重放一局，返回违反不变量的清单。"""
    bad: list[str] = []
    rules = _new_engine()
    try:
        rules.load_lineups(PROJECT_DIR / "data" / "lineups.json")
    except Exception:                                   # noqa: BLE001
        pass                                            # 阵容库不是必需的
    frames = 0
    acts = 0
    idle_with_work = 0      # 明明有活干却什么都没决定的帧（手牌堆积的直接观测量）
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        state = _rebuild(dict(rec.get("state") or {}))
        frames += 1
        try:
            actions = rules.decide_all(state)
        except Exception as exc:                        # noqa: BLE001
            bad.append(f"{path.name} 第{frames}帧 决策抛异常: {exc!r}\n"
                       + traceback.format_exc(limit=3))
            continue
        if (not actions and state.phase == "prepare" and not state.shop_open
                and state.population_valid
                and state.population < state.population_cap):
            # 场面没满、商店也关着，手里却压着能上场的英雄 —— 这一帧是**白过的**。
            # 不当失败报（搁置/本回合上场上限等都是合理原因），但要统计出来：
            # "手上堆积了手牌"（2026-09-23 发现的问题）在这个数上看得最清楚。
            if any(rules.pieces.get(sl["name"], {}).get("category") == "英雄牌"
                   and sl["name"] not in rules._field
                   for sl in (state.hand_slots or [])):
                idle_with_work += 1
        for act in actions or []:
            acts += 1
            # 不变量一：执行层吃得下
            problem = validate_action(act)
            if problem:
                bad.append(f"{path.name} 第{frames}帧 {problem}")
            for step in act.get("steps", []) or []:
                problem = validate_action(step)
                if problem:
                    bad.append(f"{path.name} 第{frames}帧 sequence 内 {problem}")
            # 不变量二：人口真读到而且满了，就不该再买英雄牌
            if (act.get("type") == "buy_piece" and state.population_valid
                    and state.population_cap > 0
                    and state.population >= state.population_cap):
                piece = act.get("piece")
                info = rules.pieces.get(piece, {})
                # 同名合成材料不占人口，允许；纯新人头不允许
                if (info.get("category") == "英雄牌"
                        and piece not in rules._field):
                    bad.append(f"{path.name} 第{frames}帧 人口已满"
                               f"({state.population}/{state.population_cap})"
                               f"还在买新英雄 {piece}")
            # 不变量三：人口上限只可能是 5/6/7（录制时的脏数据不算决策层的错，
            # 但**决策层不该把它当真**——这里只在决策真用到它时才报）
    print(f"  {path.parent.name}: {frames} 帧 / {acts} 个动作 / "
          f"{len(bad)} 条违反 / 空转{idle_with_work}帧(场面没满还压着英雄)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="跑全部对局，不只最近 3 局")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    files = sorted(LOG_DIR.glob("live_*/states.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("没有找到 data/logs/live_*/states.jsonl，跳过")
        return 0
    if not args.all:
        files = files[:args.limit]
    print(f"重放 {len(files)} 局：")
    bad: list[str] = []
    for f in files:
        bad += replay(f)
    if bad:
        print(f"\n{len(bad)} 条违反不变量：")
        for b in bad[:30]:
            print(" -", b)
        if len(bad) > 30:
            print(f" …… 还有 {len(bad) - 30} 条")
        return 1
    print("\n决策重放全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
