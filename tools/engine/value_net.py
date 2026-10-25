"""局面价值网络 v0（2026-10-14）：只看**自己的养成状态**，预测这局最终名次。不用任何伤害 / 战力估计（伤害估不准）。

输入（全部是推演器的"养成账"，确定量）：
  全局：回合、棋手等级、能量、人口上限、手牌张数、手里英雄张数
  场上每个单位（按品阶×等级排序取前 8 个，不足补 0）：品阶、log(1+等级)、觉醒进度、装备件数、是否主C
  主C：品阶、log(1+等级)、觉醒、装备件数
标签：最终名次（1~6）→ 预测期望名次；评估按局划分（同一局的局面不跨训练 / 测试）。
  .venv/Scripts/python.exe -X utf8 tools/engine/value_net.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.engine import project as PJ  # noqa: E402
from agent.engine.state import from_point  # noqa: E402

K = 8


def feats(p) -> list:
    car = PJ.pick_carry(p) if p.carry == "" else (p.unit(p.carry) or PJ.pick_carry(p))
    units = [u for u in p.on_board() if u.kind == "hero" or u.can_carry]
    units.sort(key=lambda u: -(u.quality or 1) * math.log1p(u.level))
    g = [p.round / 16, p.lord_level / 6, min(p.energy, 60) / 60, p.pop_cap / 8, len(p.hand) / 10,
         sum(1 for h in p.hand if PJ.db().get(h) and PJ.db().get(h).category == "英雄牌") / 10,
         len(units) / 8, sum(math.log1p(u.level) for u in units) / 40]
    per = []
    for u in units[:K]:
        per += [(u.quality or 1) / 5, math.log1p(u.level) / 5, min(3, u.awaken) / 3 if not u.is_awakened else 1.0,
                len(u.equips) / 3, 1.0 if u is car else 0.0]
    per += [0.0] * (5 * K - len(per))
    c = [((car.quality or 1) / 5, math.log1p(car.level) / 5, 1.0 if car.is_awakened else min(3, car.awaken) / 3,
          len(car.equips) / 3) if car else (0, 0, 0, 0)][0]
    return g + per + list(c)


def load():
    rows = [json.loads(l) for l in open(ROOT / "data/engine/value_data.jsonl", encoding="utf-8")]
    want = {}
    for r in rows:
        want.setdefault(r["run"], {})[r["round"]] = r
    X, Y, G, R = [], [], [], []
    for run, by in want.items():
        f = ROOT / "data/logs" / run / "states.jsonl"
        last = {}
        for l in open(f, encoding="utf-8"):
            if '"board"' not in l:
                continue
            try:
                x = json.loads(l)
            except ValueError:
                continue
            st = x.get("state") or {}
            if st.get("phase") == "prepare" and st.get("round_no") in by and x.get("board"):
                last[st["round_no"]] = x
        for Rn, x in last.items():
            try:
                X.append(feats(from_point(x)))
            except Exception:  # noqa: BLE001
                continue
            Y.append(by[Rn]["rank"])
            G.append(run)
            R.append(Rn)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(G), np.array(R)


def main() -> int:
    import torch
    X, Y, G, R = load()
    runs = sorted(set(G))
    random.Random(0).shuffle(runs)
    test = set(runs[: len(runs) // 5])
    te = np.array([g in test for g in G])
    print(f"{len(runs)} 局 {len(X)} 个局面；测试 {len(test)} 局 {te.sum()} 个局面")
    Xt, Yt = torch.tensor(X[~te]), torch.tensor(Y[~te])
    m = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                            torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-3)
    for ep in range(300):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = torch.nn.functional.mse_loss(m(Xt[b]).squeeze(1), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = m(torch.tensor(X[te])).squeeze(1).numpy()
    y = Y[te]
    base = np.abs(y - Y[~te].mean()).mean()
    print(f"测试集 名次平均误差 {np.abs(pr - y).mean():.3f}（常数基线 {base:.3f}）  相关 {np.corrcoef(pr, y)[0, 1]:.3f}")
    for lo, hi in ((1, 4), (5, 8), (9, 12), (13, 30)):
        s = te & (R >= lo) & (R <= hi)
        idx = s[te]
        if idx.sum() > 20:
            print(f"  第 {lo}~{hi} 回合：n={idx.sum()} 误差 {np.abs(pr[idx] - y[idx]).mean():.3f} 相关 {np.corrcoef(pr[idx], y[idx])[0, 1]:.3f}")
    torch.save({"state": m.state_dict(), "dim": X.shape[1]}, ROOT / "data/engine/value_net_v0.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
