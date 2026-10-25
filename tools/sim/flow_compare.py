"""同一个局面下，各流派（agent/sim/flows.FLOWS）投入后能走到哪一步：拿到最终主C 的概率、主C 第 12 / 14 回合等级、放弃率。
只比养成，不估伤害；"值不值"要另外的强度判断。

  .venv/Scripts/python.exe -X utf8 tools/sim/flow_compare.py --round 4 --states 20 --n 8 --scen 中 --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.sim import engine as E  # noqa: E402
from agent.sim.adapt import from_row  # noqa: E402
from agent.sim.flows import FLOWS, FlowPlayer  # noqa: E402
from agent.sim.play import Player  # noqa: E402


def real_states(round_no: int, k: int, seed: int = 0) -> list:
    out = []
    files = sorted(glob.glob(str(ROOT / "data/logs/live_*/states.jsonl")))
    rng = random.Random(seed)
    rng.shuffle(files)
    for path in files:
        last = None
        for l in open(path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            st = r.get("state") or {}
            if st.get("phase") == "prepare" and st.get("round_no") == round_no and r.get("board"):
                last = r
        if last is not None and len(last["board"]) >= 2:
            out.append((Path(path).parent.name, from_row(last)))
        if len(out) >= k:
            break
    return out


def run(st, flow, n, end=14, seed=0):
    res = []
    for i in range(n):
        s = st.clone()
        e = E.Engine(s, random.Random(seed * 100003 + i))
        p = FlowPlayer(e, flow, lookahead=1, samples=1) if flow else Player(e, lookahead=1, samples=1)
        got, lv = None, {}
        while s.round < end:
            p.round()
            if flow and got is None and s.unit(flow.final) is not None:
                got = s.round
            c = s.unit(s.carry) if s.carry else None
            lv[s.round] = c.level if c else 0
        res.append({"got": got, "lv12": lv.get(12, 0), "lv14": lv.get(end, 0), "aborted": getattr(p, "aborted", False),
                    "carry": s.carry})
    return res


def _task(args):
    """一个"局面 × 打法 × 情景"任务（多进程用：Windows 是 spawn，参数要能 pickle）。"""
    st, key, n, sc = args
    E.set_battle_scenario(sc)
    rs = run(st, FLOWS[key] if key else None, n)
    if not key:
        return {"lv12": S.median(r["lv12"] for r in rs), "lv14": S.median(r["lv14"] for r in rs)}
    return {"got10": sum(1 for r in rs if r["got"] and r["got"] <= 10) / len(rs),
            "lv12": S.median(r["lv12"] for r in rs), "lv14": S.median(r["lv14"] for r in rs),
            "abort": sum(r["aborted"] for r in rs) / len(rs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=4)
    ap.add_argument("--states", type=int, default=20)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--scen", default="中")
    ap.add_argument("--flows", default="", help="只比这些（逗号分隔 key）；默认全部")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=20, help="并行进程数（推演是纯 Python 单线程，各局互不相关）")
    a = ap.parse_args()
    keys = [k for k in (a.flows.split(",") if a.flows else FLOWS) if k]
    states = real_states(a.round, a.states)
    print(f"{len(states)} 个第 {a.round} 回合真实局面 × {len(keys)} 个流派 + 普通打法 × {a.n} 局，情景 {a.scen}，{a.workers} 进程", flush=True)
    rows = []
    from concurrent.futures import ProcessPoolExecutor
    for sc in a.scen.split(","):
        tasks = [(st, k, a.n, sc) for _, st in states for k in [""] + keys]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            it = iter(list(ex.map(_task, tasks, chunksize=1)))
        for run_name, st in states:
            row = {"run": run_name, "scen": sc, "board": [u.name for u in st.heroes()], "lord": st.lord_level, "res": {}}
            row["res"]["普通打法"] = next(it)
            for k in keys:
                row["res"][k] = next(it)
            rows.append(row)
            best = sorted(((v["lv14"], k) for k, v in row["res"].items()), reverse=True)[:3]
            print(f"[{sc}] {run_name} 棋手{st.lord_level} 场上{row['board'][:6]} → 第14回合主C最高: "
                  + "、".join(f"{FLOWS[k].name if k in FLOWS else k}{v:.0f}" for v, k in best), flush=True)
        print(f"\n[{sc}] 汇总（{len(states)} 个局面的平均）：流派 | 第10回合前拿到主C | 放弃 | 主C 第12 / 14回合 | 当最优的次数")
        win = {}
        for row in rows:
            if row["scen"] != sc:
                continue
            k = max(row["res"], key=lambda x: row["res"][x]["lv14"])
            win[k] = win.get(k, 0) + 1
        for k in ["普通打法"] + keys:
            v = [r["res"][k] for r in rows if r["scen"] == sc]
            name = FLOWS[k].name if k in FLOWS else k
            g = f"{S.mean(x['got10'] for x in v):.0%}" if k in FLOWS else "-"
            ab = f"{S.mean(x['abort'] for x in v):.0%}" if k in FLOWS else "-"
            print(f"  {name:22s} {g:>5s} {ab:>5s}  {S.mean(x['lv12'] for x in v):5.0f} / {S.mean(x['lv14'] for x in v):5.0f}  {win.get(k, 0)}")
    E.set_battle_scenario("中")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
