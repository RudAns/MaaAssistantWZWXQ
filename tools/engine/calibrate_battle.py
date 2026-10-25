"""用真实对局校准推演器的战斗模型（2026-10-14）。
每局每回合：我方 = 第 R 回合最后一个准备帧的场上单位（等级 / 装备 / 觉醒），对手 = data/enemy_levels.json（那一场遇到的对手单位数与等级和），
结果 = 第 R+1 回合血量比第 R 回合低（输）还是没掉（赢）。拟合 P(赢) = 1/(1+(E/P)^k)，在参数网格上比对数似然。
  .venv/Scripts/python.exe -X utf8 tools/engine/calibrate_battle.py [--runs 300]
"""
import argparse, collections, itertools, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.engine import project as PJ
from agent.engine.state import from_point


def samples(nruns, path="data/enemy_levels.json"):
    el = json.loads((ROOT / path).read_text(encoding="utf-8"))
    out = []
    for run in sorted(el)[-nruns:]:
        f = ROOT / "data/logs" / run / "states.jsonl"
        if not f.exists():
            continue
        last, hp = {}, {}
        for l in f.open(encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            st = r.get("state") or {}
            R = st.get("round_no")
            if not R or st.get("phase") != "prepare":
                continue
            if r.get("board"):
                last[R] = r
            h = st.get("hp")
            if isinstance(h, (int, float)) and 0 < h <= 60:
                hp.setdefault(R, []).append(h)
        for R, r in last.items():
            e = (el[run] or {}).get(str(R)) or {}
            if not e.get("enemy_sum") or not e.get("n") or R not in hp or R + 1 not in hp:
                continue
            h0 = sorted(hp[R])[len(hp[R]) // 2]
            h1 = sorted(hp[R + 1])[len(hp[R + 1]) // 2]
            if h1 > h0:
                continue                       # 读数异常
            out.append((r, e, R, int(h1 < h0)))
    return out


def enemy_pow(e, R):
    n = int(e["n"])
    avg = e["enemy_sum"] / max(1, n)
    lv_t = PJ.B.meta.level_target(min(R, 20))
    q = min(5, max(1, lv_t - 1))
    one = PJ.QUALITY_POWER[q] * (1 + avg / 8.0)
    for th, m in ((10, 1.15), (40, 1.25), (100, 1.3)):
        if avg >= th:
            one *= m
    return PJ.agg([one * 1.15] * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--el", default="data/engine/enemy_levels_all.json")
    a = ap.parse_args()
    S = samples(a.runs, a.el)
    print("样本", len(S), "输的比例", round(sum(x[3] for x in S) / max(1, len(S)), 3))
    pos = [(from_point(r), e, R, y) for r, e, R, y in S]
    res = []
    for g, sa, k in itertools.product((1.0, 1.4, 1.8, 2.5), (0.0, 0.4, 0.7, 1.0), (1.0, 1.5, 2.0, 3.0)):
        PJ.PARAMS["team_gamma"], PJ.PARAMS["skill_alpha"] = g, sa
        ll = acc = 0.0
        for p, e, R, y in pos:
            P = PJ.agg(PJ.unit_power(u) for u in p.on_board() if u.level)
            E = enemy_pow(e, R)
            w = 1.0 / (1.0 + min(1e12, (E / max(1e-9, P)) ** k))
            w = min(max(w, 1e-4), 1 - 1e-4)
            ll += math.log(1 - w) if y else math.log(w)
            acc += (w < 0.5) == bool(y)
        res.append((ll / len(pos), acc / len(pos), g, sa, k))
    res.sort(reverse=True)
    print("对数似然/样本  准确率  team_gamma skill_alpha win_k")
    for r in res[:8]:
        print(f"{r[0]:.4f}  {r[1]:.3f}  {r[2]} {r[3]} {r[4]}")
    print("现参数(1.8,0.7,2.0):", [f"{r[0]:.4f}/{r[1]:.3f}" for r in res if r[2:] == (1.8, 0.7, 2.0)])


if __name__ == "__main__":
    main()
