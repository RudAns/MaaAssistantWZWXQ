"""eco 题的**模拟后悔值**（2026-10-10）：每道测试题按各决策者选的选项，在模拟里比最优选项少拿多少分
（means 来自 tools/eco_sim_label.py；分数 = 拿到缺的人 + 人口 / 等级增量 + 剩余能量，见该文件）。
越小越好；同时报"选中最优"的比例。决策者：部署包（线上 ONNX 路径）、规则当时的做法（rule_choice）、永远 roll / levelup / save。

  .venv/Scripts/python.exe -X utf8 tools/cgnet/eval_eco_regret.py --deploys data/cgnet/deploy,data/cgnet/cand_cs_r9v \\
      --ds r9v_cg --sim data/laya/labels/r9v_sim/labels.jsonl --split test_r9t_new
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
from agent.decision.cgnet_policy import CGNetPolicy  # noqa: E402
from eval_deploy import owned_at  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploys", required=True)
    ap.add_argument("--ds", default="r9v_cg")
    ap.add_argument("--sim", default="data/laya/labels/r9v_sim/labels.jsonl")
    ap.add_argument("--split", default="test_r9t_new")
    a = ap.parse_args()
    sim = {}
    for l in (PROJECT_DIR / a.sim).open(encoding="utf-8"):
        r = json.loads(l)
        sim[r["id"]] = r
    pts = [json.loads(l) for l in (PROJECT_DIR / "data/laya/datasets" / a.ds / "points.jsonl").open(encoding="utf-8")]
    pts = [p for p in pts if p["kind"] == "eco" and p["split"] == a.split and p["id"] in sim]
    cache: dict = {}
    choices: dict = {}
    for dep in a.deploys.split(","):
        pol = CGNetPolicy.load(deploy_dir=PROJECT_DIR / dep)
        if "eco" not in getattr(pol, "kinds", ["eco"]):
            print(dep, "没有 eco 题型，跳过")
            continue
        out = {}
        for p in pts:
            pr = pol.probs(dict(p), owned_talents=owned_at(p.get("run") or "", float(p.get("ts") or 0), cache))
            out[p["id"]] = max(pr, key=pr.get)
        choices[dep] = out
    choices["规则当时（rule_choice）"] = {p["id"]: (p.get("rule_choice") or "save") for p in pts}
    for o in ("roll", "levelup", "save"):
        choices[f"永远 {o}"] = {p["id"]: o for p in pts}
    print(f"测试题 {len(pts)} 道（{a.split}）")
    base = None
    for name, ch in choices.items():
        reg = np.array([max(sim[i]["means"].values()) - sim[i]["means"].get(c, min(sim[i]["means"].values())) for i, c in ch.items()])
        best = np.mean([c == sim[i]["best"] for i, c in ch.items()])
        rng = np.random.default_rng(0)
        bs = [reg[rng.integers(0, len(reg), len(reg))].mean() for _ in range(3000)]
        print(f"  {name:40s} 平均后悔 {reg.mean():.3f} [{np.percentile(bs, 2.5):.3f},{np.percentile(bs, 97.5):.3f}]  选中最优 {best:.1%}")
        if base is None:
            base = (name, reg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
