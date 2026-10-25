"""流派转型评估：从一个局面出发，"投入流派" vs "照普通打法"各推 n 局，三档战斗情景。

只报养成和成型情况（不估伤害）：第几回合前转完等级的概率、转等级那一刻仓库几级、最终主C 在第 12 / 14 回合几级、
放弃率；普通打法那一栏是同一起点不转型的主C 等级。

  .venv/Scripts/python.exe -X utf8 tools/sim/flow_eval.py --flow haiyue --start demo --n 40
  .venv/Scripts/python.exe -X utf8 tools/sim/flow_eval.py --flow haiyue --row data/logs/live_xxx/states.jsonl:R  （真实局面，第 R 回合最后一帧）
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.sim import engine as E  # noqa: E402
from agent.sim.flows import FLOWS, FlowPlayer, flow_facts  # noqa: E402
from agent.sim.play import Player  # noqa: E402
from agent.sim.state import State  # noqa: E402

BACK = (617, 496)


def demo_state(round_no=5, lord=3, res_lv=24, res_awk=2, extra=1, tool=False, items=("护命灵珠",)):
    """海月转型前提大致成立的典型局面：第 5 回合、3 级，干将 24 级（合成两次）绑枫叶舞，手里 1 张干将 + 护盾装备。"""
    s = State(round=round_no, lord_level=lord, energy=8, lord_cards=["枫叶舞", "惊鸿游龙"])
    e = E.Engine(s, random.Random(0))
    u = e.deploy("干将莫邪", cell=BACK, level=res_lv)
    u.awaken = res_awk
    s.carry = "干将莫邪"
    e.bind_maple(u)
    for n, lv in (("苏烈", 6), ("牛魔", 4), ("鬼谷子", 5), ("孙膑", 4)):
        e.deploy(n, level=lv)
    s.hand += ["干将莫邪"] * extra + list(items) + (["遗志"] if tool else [])
    return s


def row_state(spec):
    from agent.sim.adapt import from_row
    path, R = spec.rsplit(":", 1)
    last = None
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        st = r.get("state") or {}
        if st.get("phase") == "prepare" and st.get("round_no") == int(R) and r.get("board"):
            last = r
    return from_row(last)


def run(st, flow, n, end=14, seed=0, use_flow=True):
    out = []
    for i in range(n):
        s = st.clone()
        e = E.Engine(s, random.Random(seed * 100003 + i))
        p = FlowPlayer(e, flow, lookahead=1, samples=1) if use_flow else Player(e, lookahead=1, samples=1)
        rec = {}
        while s.round < end:
            p.round()
            c = s.unit(s.carry) if s.carry else None
            rec[s.round] = (s.carry, c.level if c else 0)
        tr = next(((R, kw) for (R, k, kw) in s.log if k == "流派转等级"), None)
        out.append({"rec": rec, "transfer": tr, "aborted": getattr(p, "aborted", False),
                    "final_lv": {R: (s2[1] if s2[0] == flow.final else 0) for R, s2 in rec.items()},
                    "saw_final": any(k == "购买" and kw.get("card") == flow.final for (R, k, kw) in s.log)})
    return out


def summarize(runs, flow, label):
    n = len(runs)
    tr = [r["transfer"][0] for r in runs if r["transfer"]]
    line = [f"{label}"]
    if any(r["transfer"] for r in runs) or flow:
        for R in (10, 12, 14):
            line.append(f"R{R}前转完 {sum(1 for x in tr if x <= R) / n:.0%}")
        lv = [r["transfer"][1]["level"] for r in runs if r["transfer"]]
        if lv:
            line.append(f"转的等级中位 {S.median(lv):.0f}")
        line.append(f"放弃 {sum(r['aborted'] for r in runs) / n:.0%}")
    for R in (12, 14):
        c = [r["rec"][R][1] for r in runs if R in r["rec"]]
        if c:
            q = S.quantiles(c, n=4) if len(c) >= 4 else [S.median(c)] * 3
            line.append(f"R{R}主C {S.median(c):.0f}[{q[0]:.0f}-{q[2]:.0f}]")
    who = {}
    for r in runs:
        k = r["rec"][max(r["rec"])][0]
        who[k] = who.get(k, 0) + 1
    line.append("最终主C " + "、".join(f"{k}{v}" for k, v in sorted(who.items(), key=lambda x: -x[1])[:3]))
    return " | ".join(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="haiyue")
    ap.add_argument("--start", default="demo", help="demo / demo_tool（手里已有遗志）")
    ap.add_argument("--row", default="", help="真实局面：states.jsonl 路径:回合")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--scen", default="低,中,高")
    ap.add_argument("--rows", default="", help="批量：json 文件 [{run, R}, …]（真实局面），逐个评估并汇总")
    a = ap.parse_args()
    f = FLOWS[a.flow]
    if a.rows:
        return batch(a, f)
    st = row_state(a.row) if a.row else demo_state(tool=(a.start == "demo_tool"))
    print("起点：", json.dumps(flow_facts(st, f), ensure_ascii=False))
    for sc in a.scen.split(","):
        E.set_battle_scenario(sc)
        print(f"[{sc}] " + summarize(run(st, f, a.n), f, "投入流派"))
        print(f"[{sc}] " + summarize(run(st, None if False else f, a.n, use_flow=False), None, "普通打法"))
    E.set_battle_scenario("中")


def batch(a, f):
    items = json.loads(Path(a.rows).read_text(encoding="utf-8"))
    for sc in a.scen.split(","):
        E.set_battle_scenario(sc)
        agg = {"flow12": [], "flow14": [], "base12": [], "base14": [], "done12": [], "done14": []}
        for it in items:
            st = row_state(str(ROOT / "data/logs" / it["run"] / "states.jsonl") + f":{it['R']}")
            fr, br = run(st, f, a.n), run(st, f, a.n, use_flow=False)
            med = lambda rs, R: S.median([r["rec"][R][1] for r in rs if R in r["rec"]] or [0])  # noqa: E731
            d12 = sum(1 for r in fr if r["transfer"] and r["transfer"][0] <= 12) / len(fr)
            d14 = sum(1 for r in fr if r["transfer"]) / len(fr)
            agg["flow12"].append(med(fr, 12)); agg["flow14"].append(med(fr, 14))
            agg["base12"].append(med(br, 12)); agg["base14"].append(med(br, 14))
            agg["done12"].append(d12); agg["done14"].append(d14)
            ff = flow_facts(st, f)
            print(f"[{sc}] {it['run']} R{it['R']} 干将{ff['reservoir_level']:.0f} 棋手{ff['lord_level']} 手段{ff['transfer_tool'] or '-'} 护盾件{len(ff['keep_items'])}"
                  f" | 转完 R12前 {d12:.0%} R14前 {d14:.0%} | 主C R12 流派{med(fr, 12):.0f}/普通{med(br, 12):.0f}  R14 流派{med(fr, 14):.0f}/普通{med(br, 14):.0f}", flush=True)
        m = lambda v: S.mean(v) if v else 0  # noqa: E731
        print(f"[{sc}] 汇总 {len(items)} 个局面：转完 R12前 {m(agg['done12']):.0%} R14前 {m(agg['done14']):.0%} | "
              f"主C中位的平均 R12 流派{m(agg['flow12']):.0f}/普通{m(agg['base12']):.0f}  R14 流派{m(agg['flow14']):.0f}/普通{m(agg['base14']):.0f}")
    E.set_battle_scenario("中")


if __name__ == "__main__":
    main()
