"""决策点记录汇总 + 导出英文特征数据集（Laya 微调计划第 2、4 步的起点）。

读 data/logs/live_*/decisions.jsonl（RuleEngine.record_decision 写的），做两件事：
  1. 汇总：每个进程目录几条、各类决策多少、这一段结束时的名次（match_result）。
     一局常常被中途重启拆成几个目录，按"名次记录"往前归并成一局。
  2. --export：把每条决策的候选项换成**英文语义特征**（data/card_features.json 的 en/tags/tier），
     写 data/laya/decisions_all.jsonl —— 卡名对 Laya 是噪声（Laya 实测结论），特征才有用。

用法：
  .venv/Scripts/python.exe -X utf8 tools/decision_report.py            # 汇总
  .venv/Scripts/python.exe -X utf8 tools/decision_report.py --export   # 顺便导出特征数据集
  .venv/Scripts/python.exe -X utf8 tools/decision_report.py --match 3  # 看倒数第 3 局的逐条决策
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT = PROJECT_DIR / "data" / "laya" / "decisions_all.jsonl"


def load_runs() -> list[tuple[str, list[dict]]]:
    runs = []
    for d in sorted(glob.glob(str(PROJECT_DIR / "data/logs/live_*"))):
        f = Path(d) / "decisions.jsonl"
        if not f.exists():
            continue
        recs = []
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
        if recs:
            runs.append((os.path.basename(d), recs))
    return runs


def group_matches(runs) -> list[dict]:
    """把连续的进程目录按"出现 match_result 为止"归成一局（中途重启会把一局拆成几段）。"""
    matches, cur = [], {"dirs": [], "recs": [], "rank": None}
    last_round = 0
    for name, recs in runs:
        cur["dirs"].append(name)
        for r in recs:
            rnd = (r.get("state") or {}).get("round") or 0
            if rnd and rnd < last_round - 1 and cur["recs"]:
                # 回合号往回跳 = 新的一局（名次记录是后来才加的，老对局没有这个分界）
                matches.append(cur)
                cur = {"dirs": [name], "recs": [], "rank": None}
            if rnd:
                last_round = rnd
            if r.get("kind") == "match_result":
                last_round = 0
                cur["rank"] = r.get("rank")
                matches.append(cur)
                cur = {"dirs": [], "recs": [], "rank": None}
            else:
                cur["recs"].append(r)
    if cur["recs"]:
        matches.append(cur)          # 还没打完 / 没认出名次的
    return matches


def backfill_ranks(matches) -> None:
    """名次记录是后来才加的：没有名次的局，去它的目录里找结算页截图现场认（绿色名字那一行）。"""
    try:
        import cv2
        sys.path.insert(0, str(PROJECT_DIR))
        from agent.perception.extractors import _imread
        from tools.live_match_agent import read_my_rank
    except Exception:                                   # noqa: BLE001
        return
    for m in matches:
        if m["rank"]:
            continue
        for d in m["dirs"]:
            for f in sorted(glob.glob(str(PROJECT_DIR / "data/logs" / d / "phase_settle_final_*.png"))):
                # 目录刚建时存的那张多半是上一局的结算页（进程启动时停在结算页），跳过
                if f.endswith(f"_{d.split('_')[-1]}.png"):
                    continue
                r = read_my_rank(cv2.cvtColor(_imread(Path(f)), cv2.COLOR_BGR2RGB))
                if r:
                    m["rank"] = r


def feat(name: str, feats: dict) -> dict:
    f = feats.get(name) or {}
    return {"en": f.get("en", ""), "tags": f.get("tags", []), "tier": f.get("tier"),
            "grow": f.get("grow_score")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--match", type=int, default=0, help="打印倒数第 N 局的逐条决策")
    args = ap.parse_args()
    runs = load_runs()
    matches = group_matches(runs)
    backfill_ranks(matches)
    print(f"{len(runs)} 个进程目录有决策记录，归并成 {len(matches)} 局")
    for i, m in enumerate(matches):
        kinds = collections.Counter(r["kind"] for r in m["recs"])
        rounds = [r["state"]["round"] for r in m["recs"] if r.get("state")]
        print(f"  局{i + 1}: 名次={m['rank'] or '?'}  回合 {min(rounds, default=0)}~{max(rounds, default=0)}"
              f"  {dict(kinds)}  ({', '.join(m['dirs'])})")
    if args.match:
        m = matches[-args.match]
        for r in m["recs"]:
            st = r.get("state", {})
            opts = ", ".join(f"{o.get('name') or o.get('sell') or o.get('deploy') or o}"
                             f"{'(' + str(o['score']) + ')' if 'score' in o else ''}"
                             for o in r.get("options", []))[:150]
            print(f"R{st.get('round')} 血{st.get('hp')} 能量{st.get('gold')} {r['kind']:14s} -> {r['chosen']}  [{opts}]")
    if args.export:
        feats = json.loads((PROJECT_DIR / "data" / "card_features.json").read_text(encoding="utf-8"))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with OUT.open("w", encoding="utf-8") as fh:
            for mi, m in enumerate(matches):
                for r in m["recs"]:
                    st = r.get("state", {})
                    row = {"match": mi, "rank": m["rank"], "kind": r["kind"], "chosen": r["chosen"],
                           "round": st.get("round"), "hp": st.get("hp"), "gold": st.get("gold"),
                           "level": st.get("level"), "pop": st.get("pop"),
                           "carry": feat(st.get("carry", ""), feats) if st.get("carry") else None,
                           "field": [dict(feat(n_, feats), lv=v.get("lv")) for n_, v in (st.get("field") or {}).items()],
                           "options": [dict(o, **({"feat": feat(o["name"], feats)} if o.get("name") else {}))
                                       for o in r.get("options", [])]}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
        print(f"导出 {n} 条 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
