"""逐回合复盘表（2026-10-15 第 632 局起：每一回合都要仔细复盘）。

  .venv/Scripts/python.exe -X utf8 tools/review_rounds.py 632            # 按局号（batch2_matches.jsonl / data/logs/agent_one_<局号>.log）
  .venv/Scripts/python.exe -X utf8 tools/review_rounds.py 632 --out x.md

每回合：开局读数（血 / 能量 / 等级 / 人口）、场上、手牌；这一回合的每个决策（题型、谁答的、选了什么、概率、规则原本会选什么）；
实际执行的动作（含规则抢先的理由 why）；账本事件。最后是**手牌流水**：每张牌在哪些回合开局时还在手里、最后怎么离开的 ——
手牌堆积（一路只进不出）一眼就能看出来。只读日志，不改任何东西。
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RE_HEAD = re.compile(r"┌─ 回合 (\d+) ─ (.*)")
RE_TS = re.compile(r"^\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d)")


def find_run(no: int) -> tuple[Path, Path]:
    log = ROOT / "data" / "logs" / f"agent_one_{no}.log"
    run = None
    f = ROOT / "data" / "laya" / "iterations" / "batch2_matches.jsonl"
    for line in f.open(encoding="utf-8"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("match") == no and m.get("run"):
            run = m["run"]
    if run is None and log.exists():                     # 还没记进 batch2_matches：从日志里找 run 目录
        for line in log.open(encoding="utf-8", errors="ignore"):
            mm = re.search(r"(live_\d{10})", line)
            if mm:
                run = mm.group(1)
                break
    if run is None:
        raise SystemExit(f"第 {no} 局找不到 run 目录")
    return log, ROOT / "data" / "logs" / run


def parse_log(log: Path) -> dict:
    """按回合切日志：开局三行 + 决策行 + 执行动作。"""
    R: dict = collections.defaultdict(lambda: {"head": "", "board": "", "hand": "", "dec": [], "act": []})
    cur = 0
    for line in log.open(encoding="utf-8", errors="ignore"):
        line = line.rstrip("\n")
        ts = RE_TS.match(line)
        t = ts.group(1) if ts else ""
        m = RE_HEAD.search(line)
        if m:
            cur = int(m.group(1))
            if not R[cur]["head"]:
                R[cur]["head"] = f"{t} {m.group(2)}"
            continue
        if "│  场上" in line and not R[cur]["board"]:
            R[cur]["board"] = line.split("│  场上", 1)[1].strip()
        elif "│  手牌" in line and not R[cur]["hand"]:
            R[cur]["hand"] = line.split("│  手牌", 1)[1].strip()
        elif ("局面网络[" in line or "Laya[" in line) and " 选 " in line:
            R[cur]["dec"].append(f"{t} " + line.split("[INFO] ", 1)[-1])
        elif "执行动作:" in line:
            body = line.split("执行动作:", 1)[1].strip()
            try:
                a = ast.literal_eval(body)
            except (ValueError, SyntaxError):
                a = {"type": body[:60]}
            if a.get("type") in ("toggle_shop", "close_settings", "dismiss_detail"):
                continue
            bits = [str(a.get("type"))] + [f"{k}={a[k]}" for k in ("piece", "name", "target", "why") if a.get(k)]
            R[cur]["act"].append(f"{t} " + " ".join(bits))
    return R


def hand_names(s: str) -> list:
    """「9/10: 枫叶舞[棋手技能牌] 海月[英雄牌分48] …」→ [(名, 类)]"""
    body = s.split(":", 1)[1] if ":" in s else s
    return re.findall(r"(\S+?)\[([^\]分]+)", body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("match", type=int)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    log, run = find_run(a.match)
    R = parse_log(log)
    led = collections.defaultdict(list)
    lf = run / "ledger.jsonl"
    if lf.exists():
        seen = set()
        for line in lf.open(encoding="utf-8"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "获得卡牌":
                k = (e.get("round"), e.get("牌"))
                if k in seen:
                    continue
                seen.add(k)
            led[e.get("round") or 0].append(" ".join(f"{k}={v}" for k, v in e.items() if k != "round"))
    pts = collections.defaultdict(list)
    pf = run / "laya_points.jsonl"
    if pf.exists():
        for line in pf.open(encoding="utf-8"):
            try:
                p = json.loads(line)
            except ValueError:
                continue
            ch = p.get("laya_choice")
            crit = (p.get("question") or {}).get("criteria") or {}
            pts[p.get("round") or 0].append(f"{p['kind']} [{p.get('model')}] → {ch} ({p.get('laya_conf')}) 规则:{p.get('rule_choice')} | "
                                             + (crit.get(ch, "")[:150] if ch else ""))
    out = [f"# 第 {a.match} 局逐回合复盘表（{run.name}）\n"]
    rounds = sorted(r for r in R if r > 0)
    held = collections.defaultdict(list)            # 牌名#序号 → 开局在手的回合
    for r in rounds:
        cnt = collections.Counter()
        for n, c in hand_names(R[r]["hand"]):
            cnt[n] += 1
            held[(n, c, cnt[n])].append(r)
    for r in rounds:
        d = R[r]
        out.append(f"## 第 {r} 回合　{d['head']}")
        out.append(f"- 场上：{d['board']}")
        out.append(f"- 手牌：{d['hand']}")
        if pts.get(r):
            out.append("- 模型题：")
            out += [f"  - {x}" for x in pts[r]]
        if d["act"]:
            out.append("- 执行：")
            out += [f"  - {x}" for x in d["act"]]
        if led.get(r):
            out.append("- 账本：")
            out += [f"  - {x}" for x in led[r]]
        out.append("")
    out.append("## 手牌流水（开局时在手里的回合；连续 ≥3 回合标 ⚠）")
    for (n, c, i), rs in sorted(held.items(), key=lambda kv: (kv[1][0], kv[0][0])):
        if c == "棋手技能牌":
            continue
        span = f"R{rs[0]}" + (f"~R{rs[-1]}" if len(rs) > 1 else "")
        out.append(f"- {'⚠ ' if len(rs) >= 3 else ''}{n}（{c}）#{i}：{span}，共 {len(rs)} 回合开局在手")
    text = "\n".join(out) + "\n"
    dst = Path(a.out) if a.out else run / "review_rounds.md"
    dst.write_text(text, encoding="utf-8")
    print(f"写入 {dst}（{len(rounds)} 回合）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
