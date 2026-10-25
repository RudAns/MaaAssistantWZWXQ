"""考题的稳健性报告（2026-10-14）：每道真实局面考题在 PARAM_RANGES 里抽 n 组参数，看正确选项是最优的比例。
三档：稳定选对（正确选项合计 p_best ≥ 0.8）/ 不确定 / 稳定选错（≤ 0.2）。
  .venv/Scripts/python.exe -X utf8 tools/engine/robust_exams.py [--n 24]
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.engine.options import Action, robust_scores
from agent.engine.state import from_point


def build(ex, pt):
    raw, p = pt.get("raw") or {}, from_point(pt)
    acts = {}
    if ex["kind"] == "talent":
        names = [t[0] if isinstance(t, (list, tuple)) else t for t in raw.get("talents") or []]
        from agent.engine.options import _gift_carry, PARAMS
        for i, n in enumerate(names):
            g = _gift_carry(n)
            acts[f"opt{i}"] = Action("talent", f"天赋 {n}", {"name": n, **({"carry": g} if g else {})})
        return p, acts
    for k in pt.get("options") or []:
        if ex["kind"] == "swap":
            inc = raw.get("incoming") or ""
            if inc and inc not in p.hand:
                p.hand.append(inc)
            nm = (raw.get("names") or {}).get(k)
            acts[k] = Action("stop", "不换") if k == "bench" else Action("swap", f"卖 {nm}", {"sell": nm, "name": inc})
        elif ex["kind"] == "shop":
            if k.startswith("buy"):
                it = next((x for x in raw.get("shop") or [] if f"buy{x[0]}" == k), None)
                if it:
                    if it[1] not in p.shop:
                        p.shop.append(it[1])
                    acts[k] = Action("buy", f"买 {it[1]}", {"name": it[1], "cost": it[2]})
            elif k == "refresh":
                acts[k] = Action("refresh", "刷新")
            else:
                acts[k] = Action("stop", k)
        elif ex["kind"] == "auction":
            nm = (raw.get("names") or {}).get(k)
            acts[k] = Action("stop", "不拍") if k == "pass" else Action("auction", f"拍 {nm}", {"name": nm, "cost": 0})
    return p, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    tally = {"稳定选对": 0, "不确定": 0, "稳定选错": 0}
    for line in (ROOT / "data/engine/exam_v1.jsonl").open(encoding="utf-8"):
        ex = json.loads(line)
        if not ex.get("point_id"):
            continue
        f = ROOT / "data/logs" / ex["run"] / "laya_points.jsonl"
        pt = next((r for r in (json.loads(x) for x in f.open(encoding="utf-8")) if r.get("id") == ex["point_id"]), None)
        if not pt:
            continue
        p, acts = build(ex, pt)
        lab = {a.label: k for k, a in acts.items()}
        r = robust_scores(p, n=a.n, actions=list(acts.values()))
        good = set(ex.get("good") or []) or (set(acts) - set(ex.get("bad") or []))
        pg = sum(v["p_best"] for l, v in r.items() if lab.get(l) in good)
        cls = "稳定选对" if pg >= 0.8 else ("稳定选错" if pg <= 0.2 else "不确定")
        tally[cls] += 1
        top = "  ".join(f"{lab.get(l, l)}:{l} {v['p_best']:.2f}/{v['dV_mean']:+.1f}" for l, v in list(r.items())[:4])
        print(f"[{cls}] {ex['id']}  正确合计 {pg:.2f}  |  {top}")
    print(tally)


if __name__ == "__main__":
    main()
