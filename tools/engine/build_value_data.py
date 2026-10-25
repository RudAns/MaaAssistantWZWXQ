"""局面价值数据（2026-10-14：这个游戏追求自己养成的长期最优，不考虑对手）。

每局每回合取最后一个准备帧的局面（场上单位 / 手牌 / 能量 / 棋手等级…），标签全是**自己**的：
  pow_k   k 回合后（k=2/4/6）自己的全队战力（推演器 unit_power + agg，取对数）；没打到那么久的局记缺失
  pow_end 本局最后一个准备帧的全队战力（取对数）
  carry_end 本局最后的主C 战力（取对数）
  rank    最终名次（只作参考 —— 受对手影响）
  last_round 打到第几回合
特征由推演器的 Position 现算（与线上同口径），这里只存原始行的引用 + 标签，训练脚本再做特征。
  .venv/Scripts/python.exe -X utf8 tools/engine/build_value_data.py [--runs 700]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.engine import project as PJ  # noqa: E402
from agent.engine.state import from_point  # noqa: E402


def team_log_power(p) -> float:
    return math.log(max(1e-6, PJ.agg(PJ.unit_power(u) for u in p.on_board())))


def carry_log_power(p) -> float:
    c = PJ.pick_carry(p)
    return math.log(max(1e-6, PJ.unit_power(c))) if c else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=700)
    ap.add_argument("--out", default="data/engine/value_data.jsonl")
    a = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / "data/logs/live_*/states.jsonl")))[-a.runs:]
    n_rows = n_runs = 0
    with open(ROOT / a.out, "w", encoding="utf-8") as fo:
        for f in files:
            d = Path(f).parent
            rank = None
            led = d / "ledger.jsonl"
            if led.exists():
                for l in led.open(encoding="utf-8"):
                    if "对局结束" in l:
                        try:
                            rank = int(json.loads(l).get("名次"))
                        except (TypeError, ValueError):
                            pass
            if rank is None:
                continue
            last = {}
            for l in open(f, encoding="utf-8"):
                if '"board"' not in l:
                    continue
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                st = r.get("state") or {}
                R = st.get("round_no")
                if st.get("phase") == "prepare" and isinstance(R, int) and 0 < R < 30 and r.get("board"):
                    last[R] = r
            if len(last) < 4:
                continue
            rounds = sorted(last)
            pw, cp = {}, {}
            for R in rounds:
                try:
                    p = from_point(last[R])
                    pw[R], cp[R] = team_log_power(p), carry_log_power(p)
                except Exception:  # noqa: BLE001
                    pass
            if not pw:
                continue
            Rend = max(pw)
            for R in rounds:
                if R not in pw:
                    continue
                row = {"run": d.name, "round": R, "ts": last[R].get("ts"), "rank": rank, "last_round": Rend,
                       "pow": round(pw[R], 4), "carry_pow": round(cp[R], 4),
                       "pow_end": round(pw[Rend], 4), "carry_end": round(cp[Rend], 4)}
                for k in (2, 4, 6):
                    if R + k in pw:
                        row[f"pow_{k}"] = round(pw[R + k], 4)
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
            n_runs += 1
    print(f"{n_runs} 局，{n_rows} 个局面")
    return 0


if __name__ == "__main__":
    sys.exit(main())
