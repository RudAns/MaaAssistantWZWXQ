"""规则层删商店题选项时删掉了哪些路线核心件（2026-10-13，只读审计，配合 core_assembly_audit.py）。

逐局读 agent_one_<n>.log 里「→ 商店题去掉 [...]」的行，按时间对上紧接着的 laya_points 商店题（raw.shop 给出 buyN → 卡名），
统计被删的 buy 选项里有多少是当前路线的 core_set / early_priority 英雄。

  .venv/Scripts/python.exe -X utf8 tools/core_assembly_restrictions.py --from 560 --to 617
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import core_assembly_audit as A  # noqa: E402

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) \[INFO\] (.*?)→ 商店题去掉(.*)$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="a", type=int, default=560)
    ap.add_argument("--to", dest="b", type=int, default=617)
    a = ap.parse_args()
    tot = collections.Counter()
    core_hits = collections.Counter()
    examples = collections.defaultdict(list)
    for row in A.matches(a.a, a.b):
        log = ROOT / "data" / "logs" / f"agent_one_{row['match']}.log"
        if not log.exists():
            continue
        pts = [p for p in A.jl(A.LOGS / row["run"] / "laya_points.jsonl") if p.get("kind") == "shop"]
        pts.sort(key=lambda p: p.get("ts") or 0)
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = LINE.match(line)
            if not m:
                continue
            ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000
            kind = re.sub(r"[0-9]+", "N", m.group(3)).split("：")[0].strip()
            if "仅记录" in kind:
                continue
            keys = re.findall(r"'(\w+)'", m.group(4))
            p = next((p for p in pts if 0 <= (p.get("ts") or 0) - ts < 3), None)
            if p is None:
                continue
            shop = {f"buy{i}": n for i, n, *_ in (p.get("raw") or {}).get("shop") or []}
            rk = (p.get("ctx") or {}).get("route")
            r = A.ROUTES.get(rk or "", {})
            core = {x for g in r.get("core", []) for x in g}
            early = set(r.get("early", []))
            for k in keys:
                if not k.startswith("buy"):
                    tot[(kind, k)] += 0
                    tot[(kind, "refresh/stop/lock")] += 1
                    continue
                n = shop.get(k, "?")
                tot[(kind, "buy")] += 1
                tag = "core" if n in core else ("early" if n in early else "")
                if tag:
                    core_hits[(kind, tag)] += 1
                    examples[(kind, tag)].append((row["match"], p.get("round"), n, (p.get("ctx") or {}).get("gold"), p.get("id")))
    print("规则删掉的选项（按规则 × 类型）：")
    for k, v in sorted(tot.items(), key=lambda x: -x[1]):
        if v:
            print(f"  {k}: {v}")
    print("其中删掉的是路线核心 / 最低配套：")
    for k, v in core_hits.most_common():
        print(f"  {k}: {v}  例 {examples[k][:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
