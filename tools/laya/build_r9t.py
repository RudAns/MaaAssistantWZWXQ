"""r9t 数据集（2026-10-09）：**eco 题**（经济节奏：升级 / 停级刷牌 / 攒钱）的历史题，题面版本 f9le。纯 CPU。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9t.py                 # 默认第 330 局起、最新 12 局做测试集

- 对局来源：data/laya/iterations/batch2_matches.jsonl 里 match >= --min-match、没有 note 的局（note = 重启重连 / 不计统计）。
  一局中途重启会分成几个运行目录：按 build_r9s.group_runs 的口径把后续目录并进同一局（同一回合只取第一道题）。
- 出题：build_dataset.extract_eco —— 每回合准备阶段第一帧商店开着的画面一题（这一回合还没升级 / 刷新过）。
  路线配置（--carry-tiers / 公孙离排除）按 build_r9s.run_render_cfgs 从 agent 日志认，和线上当时一致。
- 每题附 hindsight（**只给老师核对，不进题面**）：本回合实际动作计数、后 1/2 回合的棋手等级、题面列的缺件到 R+2 回合拿到了几个、
  该局最终名次（batch2_matches 的 rank）。rule_choice = 本回合实际做了什么（升过级 / 只刷过 / 都没有）。
- 划分：按局号，最新 --test-matches 局 → test_r9t_new，其余 train（eco 是新题型，没有旧测试集可沿用）。
- 另出一份清单 carry_temp_swap_cases.json：待办问题「1 阶临时主C 之间不互换」的反例局（438/452/462/467/470）里，
  线上 carry 题现任主C 和被选的新主C 都是 1 阶的那些题（给 g9q 规范举例 / 重标用）。
输出：data/laya/datasets/r9t/{points.jsonl, info.json, matches.json, carry_temp_swap_cases.json}
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # 只用 CPU（显卡留给其他任务）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_dataset as BD  # noqa: E402
import build_r9s as R9S  # noqa: E402
import features as F  # noqa: E402

VER = "f9le"
TEST_SPLIT = "test_r9t_new"
TEMP_SWAP_MATCHES = (438, 452, 462, 467, 470)


def pid(run, ts, kind) -> str:
    return hashlib.md5(f"{run}|{ts}|{kind}".encode()).hexdigest()[:12]


def temp_swap_cases(runs_by_match: dict) -> list:
    """线上 carry 题：现任主C 与最终被采纳 / 学生首选的新人都是 1 阶 → 列出来（含双方等级、学生概率、是否执行）。"""
    X = F._facts()
    out = []
    for m in TEMP_SWAP_MATCHES:
        for run in runs_by_match.get(m, []):
            f = PROJECT_DIR / "data" / "logs" / run / "laya_points.jsonl"
            if not f.exists():
                continue
            for line in f.open(encoding="utf-8"):
                try:
                    p = json.loads(line)
                except ValueError:
                    continue
                if p.get("kind") != "carry":
                    continue
                names = (p.get("raw") or {}).get("names") or {}
                cur = names.get("keep") or ""
                ch = p.get("laya_choice") or ""
                new = names.get(ch) or ""
                if not cur or not new or ch == "keep":
                    continue
                if X.tier(cur) != 1 or X.tier(new) != 1:
                    continue
                board = p.get("board") or {}
                if isinstance(board, str):
                    import ast
                    try:
                        board = ast.literal_eval(board)
                    except (ValueError, SyntaxError):
                        board = {}
                lv = lambda n: (board.get(n) or {}).get("lv") if isinstance(board.get(n), dict) else None  # noqa: E731
                out.append({"match": m, "run": run, "id": p.get("id"), "ts": p.get("ts"), "round": p.get("round"),
                            "cur_carry": cur, "cur_lv": lv(cur), "chosen": new, "chosen_lv": lv(new),
                            "choice_key": ch, "probs": p.get("laya"), "model": p.get("model"), "used": p.get("used"),
                            "why": p.get("why")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="r9t")
    ap.add_argument("--min-match", type=int, default=330)
    ap.add_argument("--test-matches", type=int, default=12, help="最新的几局做测试集")
    ap.add_argument("--busy-min", type=float, default=10.0, help="states.jsonl 最近多少分钟内还在写 = 进行中的对局，排除")
    args = ap.parse_args()
    F.ACTIVE = VER
    t0 = time.time()
    rows = {}
    for line in (PROJECT_DIR / "data" / "laya" / "iterations" / "batch2_matches.jsonl").open(encoding="utf-8"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if (m.get("match") or 0) >= args.min_match and not m.get("note") and m.get("run"):
            rows[m["match"]] = m
    logs = PROJECT_DIR / "data" / "logs"
    first_run = min(r["run"] for r in rows.values())
    now = time.time()
    newest = max(p for p in logs.glob("live_*") if p.is_dir())
    spans, skipped = [], {}
    for d in sorted(logs.glob("live_*")):
        if not d.is_dir() or d.name < first_run or not (d / "states.jsonl").exists():
            continue
        if d == newest and now - (d / "states.jsonl").stat().st_mtime < args.busy_min * 60:
            skipped[d.name] = "进行中（最新目录，states.jsonl 仍在写）"
            continue
        spans.append((d.name, R9S.run_span(d)))
    groups = R9S.group_runs(spans)
    run2match = {r["run"]: k for k, r in rows.items()}
    runs_by_match: dict = {}
    for g in groups:
        ms = [run2match[x] for x in g if x in run2match]
        if ms:
            runs_by_match[ms[0]] = g
    missing_runs = sorted(k for k in rows if k not in runs_by_match)
    cfgs = R9S.run_render_cfgs()
    out, stats = [], collections.Counter()
    for m in sorted(runs_by_match):
        seen_rounds = set()
        for run in runs_by_match[m]:
            cfg = cfgs.get(run)
            R9S.apply_cfg(cfg)
            for p in BD.extract_eco(logs / run):
                if p["round"] in seen_rounds:
                    stats["dup_round_after_restart"] += 1
                    continue
                seen_rounds.add(p["round"])
                p["match"] = m
                p["rank"] = rows[m].get("rank")
                p["hindsight"]["final_rank"] = rows[m].get("rank")
                p["hindsight"]["ban"] = rows[m].get("ban")
                p["id"] = pid(p["run"], p["ts"], "eco")
                p["feature_version"] = VER
                p["source"] = "offline_eco"
                p["render_cfg"] = cfg
                out.append(p)
        R9S.apply_cfg(None)
    ms = sorted({p["match"] for p in out})
    test = set(ms[-args.test_matches:]) if args.test_matches else set()
    for p in out:
        p["split"] = TEST_SPLIT if p["match"] in test else "train"
    od = PROJECT_DIR / "data" / "laya" / "datasets" / args.out
    od.mkdir(parents=True, exist_ok=True)
    with (od / "points.jsonl").open("w", encoding="utf-8") as fh:
        for p in out:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (od / "matches.json").write_text(json.dumps({str(k): {"runs": runs_by_match[k], "rank": rows[k].get("rank"),
                                                        "split": TEST_SPLIT if k in test else "train"}
                                                 for k in sorted(runs_by_match)}, ensure_ascii=False, indent=1), encoding="utf-8")
    cases = temp_swap_cases(runs_by_match)
    (od / "carry_temp_swap_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    info = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "feature_version": VER, "kind": "eco",
        "min_match": args.min_match, "matches_listed": len(rows), "matches_with_points": len(ms),
        "matches_without_logs": missing_runs, "skipped_runs": skipped,
        "points": len(out), "by_split": dict(collections.Counter(p["split"] for p in out)),
        "test_matches": sorted(test),
        "n_opt_hist": dict(collections.Counter(p["n_opt"] for p in out)),
        "options_hist": dict(collections.Counter("/".join(p["options"]) for p in out)),
        "rule_choice_hist": dict(collections.Counter(str(p["rule_choice"]) for p in out)),
        "with_missing": sum(1 for p in out if (p["raw"].get("missing"))),
        "with_carry": sum(1 for p in out if p["raw"].get("carry")),
        "level_unread": sum(1 for p in out if not isinstance((p.get("ctx") or {}).get("level"), int)),
        "temp_swap_cases": len(cases),
        "stats": dict(stats), "seconds": round(time.time() - t0, 1),
    }
    (od / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
