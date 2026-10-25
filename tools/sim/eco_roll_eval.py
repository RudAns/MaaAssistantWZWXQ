"""本回合刷几次再升级 / 攒钱（2026-10-16：棋手低等级时适当多刷几次、多凑低阶核心阵容；刷多少次合适由推演器计算，
刷得越多棋手升级越慢，两者要权衡）。

基于 tools/eco_sim_label.py 的商店概率模型（实测品阶概率 / 牌库 / 收入 / 经验 / 对手拿牌），从每道 eco 题的真实局面出发比较：
  roll_k（k = 0..K）：本回合先付费刷新 k 次（免费刷新照用），见到缺的人 / 体系同名就买，剩下的钱留着；
  levelup：先升一级再看一眼商店（同 eco_sim_label）；
之后每回合按同一套默认打法（落后进度表且升得起就升，再刷到买不起），推 HORIZON 回合，比较
  score = 拿到的缺的人（主C 2 / 配套 1，上场再记一半）+ 合成升星 + 0.2 × 人口增量 + 0.15 × 等级增量 + 0.03 × 剩余能量（同 eco_sim_label）。
**同一组随机数（CRN）**：每个样本 i 的所有选项用同一个种子，比较的是配对差，噪声小得多（talent_eval 同样做法）。
输出每题：各选项均值、最佳 k、对"刷到买不起"（旧 roll）和 levelup 的配对差 + 95% 区间、棋手到 HORIZON 时的平均等级。

  .venv/Scripts/python.exe -X utf8 tools/sim/eco_roll_eval.py --ds r9zo --matches 656-700 --n 300 --out data/sim/eco_roll_r9zo.jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import eco_sim_label as E  # noqa: E402

K_MAX = 6


class RollSim(E.Sim):
    def roll_k(self, k: int) -> None:
        """本回合：先看当前商店，再付费刷新至多 k 次（免费刷新不计次），见到想要的就买；不强迫刷完。"""
        self.look(E.shop(self.pool, self.lv, self.rng))
        paid = 0
        while self.missing or any(v > 0 for v in self.merge.values()):
            if self.free > 0:
                self.free -= 1
            elif paid < k and self.gold - 2 >= 3:
                self.gold -= 2
                paid += 1
            else:
                break
            self.look(E.shop(self.pool, self.lv, self.rng))

    def run_opt(self, opt: str) -> tuple[float, int]:
        self._miss0 = len(self.missing)                   # 核对用：推演里拿到了几个缺的人（eco_sim_validate.py）
        if opt == "levelup":
            if self.level_up():
                self.look(E.shop(self.pool, self.lv, self.rng))
        elif opt == "roll_all":
            self.roll()
        elif opt.startswith("roll_"):
            self.roll_k(int(opt[5:]))
        end = self.r + E.HORIZON
        self.end_round()
        while self.r <= end:
            self.default_round()
            if self.r == end:
                break
            self.end_round()
        dpop = E.POP_CAP.get(min(self.lv, 6), 7) - E.POP_CAP.get(min(self.lv0, 6), 5)
        self.acq = self._miss0 - len(self.missing)
        return self.got + 0.2 * dpop + 0.15 * (self.lv - self.lv0) + 0.03 * self.gold, self.lv


def _ci(xs):
    m = st.mean(xs)
    if len(xs) < 2:
        return m, m, m
    se = st.stdev(xs) / len(xs) ** 0.5
    return m, m - 1.96 * se, m + 1.96 * se


def eval_one(args):
    p, n, seed = args
    ctx = p.get("ctx") or {}
    raw = p.get("raw") or {}
    missing = raw.get("missing") or []
    carry = raw.get("carry") or ""
    owned = [u for u in (p.get("field") or [])] + [h for h in (ctx.get("hand") or []) if h in E.HEROES]
    carry_missing = next((m[0] for m in missing if "CARRY" in str(m[2]).upper() or "主C" in str(m[2])), "") if not carry else ""
    from agent.decision import routes as _R
    family = set(_R.final_team(_R.by_key(ctx.get("route") or "") or {}))
    field_heroes = [u for u in (p.get("field") or []) if u in E.HEROES]
    fam_c = family | ({carry} if carry else set())
    waiting_n = len([h for h in (ctx.get("hand") or []) if h in E.HEROES and h in fam_c and h not in field_heroes])
    opts = [f"roll_{k}" for k in range(K_MAX + 1)] + ["roll_all", "levelup"]
    sc = {o: [] for o in opts}
    lv = {o: [] for o in opts}
    acq = {o: [] for o in opts}
    for i in range(n):
        for o in opts:
            rng = random.Random(seed * 100003 + i)            # CRN：同一样本所有选项同一种子
            sim = RollSim(ctx, missing, owned, carry_missing, rng, carry, family, len(field_heroes), waiting_n)
            s, L = sim.run_opt(o)
            sc[o].append(s)
            lv[o].append(L)
            acq[o].append(sim.acq)
    means = {o: round(st.mean(v), 3) for o, v in sc.items()}
    best = max(means, key=means.get)
    out = {"id": p["id"], "run": p.get("run"), "round": p.get("round"), "match": p.get("match"),
           "gold": ctx.get("gold"), "level": ctx.get("level"), "level_cost": ctx.get("level_cost"),
           "means": means, "best": best, "lv_at_h": {o: round(st.mean(v), 2) for o, v in lv.items()},
           "acq": {o: round(st.mean(v), 3) for o, v in acq.items()}, "missing": [m[0] for m in missing], "ts": p.get("ts")}
    for ref in ("roll_all", "levelup", "roll_0"):
        d = [a - b for a, b in zip(sc[best], sc[ref])]
        m, lo, hi = _ci(d)
        out[f"best_minus_{ref}"] = [round(m, 3), round(lo, 3), round(hi, 3)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="r9zo")
    ap.add_argument("--matches", default="656-700")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    ap.add_argument("--out", default="data/sim/eco_roll_r9zo.jsonl")
    ap.add_argument("--seed-base", type=int, default=0, help="核对种子稳定性：换一组随机数重跑（0 = 原来那组）")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.matches.split("-"))
    # eco 题不在 Laya 数据集里：按 batch2_matches 的局号直接从各局 states.jsonl 抽（build_dataset.extract_eco，和局面网络同一套）
    sys.path.insert(0, str(ROOT / "tools" / "laya"))
    import features as _F
    _F.ACTIVE = "f9lx"
    _F._facts()
    import build_dataset as BD
    pts = []
    for l in (ROOT / "data/laya/iterations/batch2_matches.jsonl").open(encoding="utf-8"):
        m = json.loads(l)
        if lo <= int(m.get("match") or 0) <= hi and m.get("run") and (ROOT / "data/logs" / m["run"]).exists():
            for q in BD.extract_eco(ROOT / "data/logs" / m["run"]):
                q.update(run=m["run"], match=m["match"], match_no=m["match"], id=q.get("id") or f"{m['run']}|{q.get('ts')}")
                pts.append(q)
    print("eco 题", len(pts), flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(p, a.n, i + 1 + a.seed_base * 1000003) for i, p in enumerate(pts)]
    with mp.Pool(a.workers) as pool, out.open("w", encoding="utf-8") as fh:
        for r in pool.imap(eval_one, jobs, chunksize=2):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    from collections import Counter
    print("最佳选项分布", Counter(r["best"] for r in rows))
    sig = sum(1 for r in rows if r["best_minus_roll_all"][1] > 0)
    print(f"最佳 vs 刷到买不起：配对差下限 > 0 的 {sig}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
