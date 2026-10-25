"""eco876 数据集（2026-10-21，经济网络补训用）：第 845~875 局**线上** eco 题，按线上题面版本 f9z 重渲染（= build_r9zx_eco 的流程，只换局范围和局↔目录映射）。

为什么：近 15 局经济网络 cs_r9zy_z 系统性偏慢（R4 平均 2.3 级 / R7 3.3 / R10 4.1，进度线 3/4/5；866 R4~R6 连续 save 攒到 18e 仍 3 级、875 R8 才 4 级）。
和上次（r9zx_eco → r9zx_eco_d 54.22 落后进度重标）同口径：先出全量线上 eco 题，再挑"落后进度线"的题 + 随机对照题导出给老师。

和 build_r9zx_eco 的区别：
- 局 ↔ live 目录：按 tools/check_components.matches() 的口径 —— agent_one_<局号>*.log 里提到的 live_ 目录合并（中途重启的局有多个目录），不按时间间隔分组。
- --min-match / --max-match 限定局范围（默认 845~875；876 进行中不收）。
- 题面版本默认 f9z（第 845~875 局线上全是 f9z；局面网络只读 ctx / raw，raw.missing 按同一版本重算）。
- 输出目录 data/laya/datasets/eco876/，测试划分名 test_eco876_new（最新 --test-matches 局）。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_eco876.py --check
  .venv/Scripts/python.exe -X utf8 tools/laya/build_eco876.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_r9zx_eco as B  # noqa: E402

PROJECT_DIR = B.PROJECT_DIR
B.NAME = "eco876"
B.TEST_SPLIT = "test_eco876_new"
B.EXTRA_RUNS = {}


def _pop(flag: str, default: str) -> str:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


MAX_MATCH = int(_pop("--max-match", "875"))
if "--min-match" not in sys.argv:
    sys.argv += ["--min-match", "845"]
if "--ver" not in sys.argv:
    sys.argv += ["--ver", "f9z"]
if "--test-matches" not in sys.argv:
    sys.argv += ["--test-matches", "6"]


def _match_logs(n: int) -> list[Path]:
    pre = f"agent_one_{n}"
    return [f for f in sorted((PROJECT_DIR / "data/logs").glob(pre + "*.log")) if f.stem[len(pre):].isalpha() or f.stem == pre]


def match_runs(min_match: int, busy_min: float, st: collections.Counter) -> tuple[dict, dict]:
    rows = {}
    for line in (PROJECT_DIR / "data/laya/iterations/batch2_matches.jsonl").open(encoding="utf-8"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if min_match <= (m.get("match") or 0) <= MAX_MATCH and m.get("run"):
            rows[m["match"]] = m
    logs = PROJECT_DIR / "data/logs"
    newest = max(p for p in logs.glob("live_*") if p.is_dir())
    busy = []
    if (newest / "states.jsonl").exists() and time.time() - (newest / "states.jsonl").stat().st_mtime < busy_min * 60:
        busy.append(newest.name)
    by_match, seen = {}, set()
    for k in sorted(rows):
        segs = []
        for f in _match_logs(k):
            for d in re.findall(r"live_\d{6,}", f.read_text(encoding="utf-8", errors="ignore")):
                if d not in segs and (logs / d).is_dir():
                    segs.append(d)
        if rows[k]["run"] not in segs and (logs / rows[k]["run"]).is_dir():
            segs.append(rows[k]["run"])
        segs = [d for d in segs if d not in busy and d not in seen and (logs / d / "states.jsonl").exists()]
        seen.update(segs)
        if segs:
            by_match[k] = sorted(segs)
        st[f"segs_{len(segs)}"] += 1
    st["busy_runs_skipped"] = len(busy)
    return by_match, {"rows": rows, "busy": busy, "unmapped": []}


B.match_runs = match_runs

if __name__ == "__main__":
    sys.exit(B.main())
