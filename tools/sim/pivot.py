"""中途转不转流派：给一个局面，把"继续现在的打法"和每个流派放在一起推，列出事实 + 成型概率 + 主C 等级。
只比养成，不估伤害；"值不值"交给局面网络 / Laya（结果作为它们的输入）。

  .venv/Scripts/python.exe -X utf8 tools/sim/pivot.py --row data/logs/live_xxx/states.jsonl:8 --n 16
  .venv/Scripts/python.exe -X utf8 tools/sim/pivot.py --demo yuji_shop --n 16
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as S
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.sim import engine as E  # noqa: E402
from agent.sim.flows import FLOWS, FlowPlayer, flow_facts, flow_pieces  # noqa: E402
from agent.sim.play import Player  # noqa: E402
from agent.sim.state import State, Unit  # noqa: E402


def demo(name):
    if name == "yuji_shop":      # 示例：核心件场上 1 个、商店里正好 2 个
        s = State(round=6, lord_level=3, energy=10, lord_cards=["枫叶舞", "惊鸿游龙"])
        s.units = [Unit("虞姬", level=12, cell=(617, 496), uid=1), Unit("少司缘", level=5, cell=(529, 372), uid=2),
                   Unit("苏烈", level=6, cell=(411, 372), uid=3), Unit("孙膑", level=4, cell=(718, 371), uid=4)]
        s.next_uid = 5
        s.shop = ["鬼谷子", "孙膑", "张良", "牛魔"]
        s.flags["shop_fresh"] = True
        s.carry = s.maple = "虞姬"
        return s
    raise SystemExit("未知 demo")


def row_state(spec, frame="first"):
    from agent.sim.adapt import from_row
    path, R = spec.rsplit(":", 1)
    pick = None
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        st = r.get("state") or {}
        if st.get("phase") == "prepare" and st.get("round_no") == int(R) and r.get("board"):
            if frame == "first" and st.get("shop_open") and st.get("shop"):
                return from_row(r)            # 本回合第一次看到商店的那一帧：还没动手，商店是完整的
            pick = r
    return from_row(pick)


def _task(args):
    st, key, n, sc, horizon = args
    E.set_battle_scenario(sc)
    f = FLOWS.get(key)
    out = []
    for i in range(n):
        s = st.clone()
        e = E.Engine(s, random.Random(1000 + i))
        p = FlowPlayer(e, f, lookahead=1, samples=1) if f else Player(e, lookahead=1, samples=1)
        r0 = s.round
        got = done = None
        lv = {}
        while s.round < 14:
            p.round()
            if f:
                names = {u.name for u in s.heroes()}
                if got is None and f.final in names:
                    got = s.round
                if done is None and all(x in names for x in flow_pieces(f)):
                    done = s.round
            c = s.unit(s.carry) if s.carry else None
            lv[s.round] = c.level if c else 0
        out.append({"got": got, "done": done, "lv12": lv.get(12, 0), "lv14": lv.get(14, 0),
                    "aborted": getattr(p, "aborted", False), "r0": r0})
    h = st.round + horizon
    return {"got_h": sum(1 for x in out if x["got"] and x["got"] <= h) / n,
            "done_h": sum(1 for x in out if x["done"] and x["done"] <= h) / n,
            "lv12": S.median(x["lv12"] for x in out), "lv14": S.median(x["lv14"] for x in out),
            "abort": sum(x["aborted"] for x in out) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default="")
    ap.add_argument("--frame", default="first", help="first：本回合第一次开商店的帧（推荐）；last：本回合最后一帧")
    ap.add_argument("--demo", default="")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--scen", default="中")
    ap.add_argument("--horizon", type=int, default=3, help="几回合内凑齐 / 拿到主C")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    st = demo(a.demo) if a.demo else row_state(a.row, a.frame)
    print(f"局面：第 {st.round} 回合 棋手 {st.lord_level} 级 能量 {st.energy} 主C {st.carry or '-'} 枫叶舞 {st.maple or '-'} 禁 {st.banned or '-'}")
    print(f"  场上 {[(u.name, int(u.level)) for u in st.heroes()]}  手牌 {st.hand}  商店 {st.shop}")
    keys = [""] + list(FLOWS)
    for sc in a.scen.split(","):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(_task, [(st, k, a.n, sc, a.horizon) for k in keys], chunksize=1))
        base = res[0]
        print(f"\n[{sc}] 继续现在的打法（灵活选主C）：主C 第12回合 {base['lv12']:.0f} / 第14回合 {base['lv14']:.0f}")
        rows = []
        for k, r in zip(keys[1:], res[1:]):
            f = FLOWS[k]
            ff = flow_facts(st, f)
            cur = "（当前路线）" if st.carry == f.final else ""
            rows.append((r["lv14"], f"  {f.name}{cur}：已有 {ff['pieces_have'] or '-'} 缺 {ff['pieces_missing'] or '-'}"
                         + (f" 商店里有 {ff['missing_in_shop']}（{ff['buy_all_cost']} 能量）" if ff["missing_in_shop"] else "")
                         + ("，**现在买就凑齐**" if ff["completes_if_buy_now"] else "")
                         + f"\n      {a.horizon} 回合内拿到主C {r['got_h']:.0%} 凑齐 {r['done_h']:.0%} | 放弃 {r['abort']:.0%} | 主C 第12 / 14回合 {r['lv12']:.0f} / {r['lv14']:.0f}"))
        for _, line in sorted(rows, reverse=True):
            print(line)
    E.set_battle_scenario("中")


if __name__ == "__main__":
    main()
