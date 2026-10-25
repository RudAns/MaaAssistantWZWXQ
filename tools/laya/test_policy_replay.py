"""学生上场（LayaPolicy）的离线回放测试：用真实对局的每一帧跑"规则 + 学生"，查不变量。

不要模拟器。查的是：
  - 不崩（学生出错时整帧回退规则）；
  - 学生给出的动作字段执行层吃得下（validate_action）；
  - 四类决策都被接管过、日志 laya_points.jsonl 写得出来且格式与 points.jsonl 一致；
  - 商店的回退原因分布（硬约束 / 低置信度）合理。

  python tools/laya/test_policy_replay.py [--ckpt <checkpoint>] [--runs live_xxx ...]
没给 --ckpt 就用原版 multilingual（只验证管线，不看决策质量）。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools"))
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))

import torch  # noqa: E402,F401
from test_decisions_replay import _new_engine, _rebuild  # noqa: E402
from agent.actions.executor import validate_action  # noqa: E402
from agent.decision.laya_engine import HybridEngine  # noqa: E402
from agent.decision.laya_policy import LayaPolicy  # noqa: E402
from train import BASE_CKPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(BASE_CKPT))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--kinds", default="card_pick,talent,lord_boost,shop",
                    help="学生接管的题型（和 live_match_agent --laya-kinds 相同；带 carry,swap 验证接管通道）")
    args = ap.parse_args()
    runs = [PROJECT_DIR / "data" / "logs" / r for r in args.runs] if args.runs else \
        sorted((PROJECT_DIR / "data" / "logs").glob("live_*/states.jsonl"),
               key=lambda f: f.stat().st_size, reverse=True)[:2]
    runs = [r.parent if r.name == "states.jsonl" else r for r in runs]
    tmp = Path(tempfile.mkdtemp(prefix="laya_policy_"))
    rules = _new_engine()
    rules.load_lineups(PROJECT_DIR / "data" / "lineups.json")
    policy = LayaPolicy(args.ckpt, rules, tmp, kinds=tuple(args.kinds.split(",")))
    hybrid = HybridEngine(rules, None, policy=policy)
    bad, frames, changed = [], 0, collections.Counter()
    for rd in runs:
        for line in (rd / "states.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                st = _rebuild(dict(rec.get("state") or {}))
            except Exception:                                   # noqa: BLE001
                continue
            frames += 1
            try:
                acts, _ = hybrid.decide(st)
            except Exception as e:                              # noqa: BLE001
                bad.append(f"{rd.name} 崩溃：{type(e).__name__}: {e}")
                continue
            for a in acts:
                err = validate_action(a) if callable(validate_action) else None
                if err:
                    bad.append(f"{rd.name} 动作字段不合法：{a} -> {err}")
            if acts:
                changed[acts[0].get("type")] += 1
    rows = [json.loads(l) for l in (tmp / "laya_points.jsonl").open(encoding="utf-8")] \
        if (tmp / "laya_points.jsonl").exists() else []
    kinds = collections.Counter(r["kind"] for r in rows)
    used = collections.Counter((r["kind"], r["used"]) for r in rows)
    why = collections.Counter(r["why"] for r in rows if r["why"])
    need = {"kind", "question", "state", "options", "raw", "ctx", "field", "id", "laya", "laya_choice", "used"}
    missing = [k for k in need if rows and k not in rows[0]]
    print(f"回放 {len(runs)} 个运行目录、{frames} 帧；学生记录 {len(rows)} 条")
    print("  按类型：", dict(kinds))
    print("  谁执行：", {f"{k}/{u}": n for (k, u), n in used.items()})
    print("  回退原因：", dict(why))
    print("  学生延迟 ms p50：", sorted(r["laya_ms"] for r in rows)[len(rows) // 2] if rows else None)
    # 商店题里场上单位等级已知的比例（2026-09-25：线上沿用读数只在出题时更新 → 题面全是 lv ?；守住别再退回去）
    import re as _re
    lv_known = lv_all = 0
    for r in rows:
        if r["kind"] != "shop":
            continue
        for line in r["state"].splitlines():
            m = _re.search(r" lv (\?|\d+)(?=[\s|]|$)", line)
            if m and line.startswith("  ") and "|" in line:
                lv_all += 1
                lv_known += m.group(1) != "?"
    print(f"  商店题里场上单位等级已知：{lv_known}/{lv_all}" + (f"（{lv_known / lv_all:.0%}）" if lv_all else ""))
    if missing:
        bad.append(f"laya_points.jsonl 缺字段 {missing}")
    if not rows:
        bad.append("一条学生记录都没有")
    for b in bad[:20]:
        print("  ✗", b)
    print("通过" if not bad else f"失败 {len(bad)} 条")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
