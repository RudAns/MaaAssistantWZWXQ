"""一局复盘（2026-09-30 起：每打完一局报告名次和问题所在）。

  python tools/match_report.py data/logs/agent_rot_9.log

读 agent 日志 + 对应 live_*/ledger.jsonl + states.jsonl，逐回合列：血量 / 能量 / 等级 / 人口 / 主C 及其等级 / 场上阵容，
再列关键事件（路线、换主C、升级、刷新次数、拖拽失败、搁置、卖人）和异常计数。只读，不改任何东西。
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(log_path: str) -> int:
    log = Path(log_path).read_text(encoding="utf-8", errors="replace")
    runs = re.findall(r"live_\d+", log)
    run = ROOT / "data" / "logs" / runs[0] if runs else None
    rank = re.findall(r"本局名次：第 (\d) 名", log)
    print(f"名次：{rank[-1] if rank else '?'}　日志 {log_path}　对局目录 {run.name if run else '?'}")

    # 每回合首帧概况
    head = {}
    for m in re.finditer(r"┌─ 回合 (\d+) ─ 血 (\S+) ─ 能量 (\S+) ─ 等级 (\S+) ─ 人口 (\S+).*?阵容 (\S+)", log):
        head.setdefault(int(m.group(1)), m.groups()[1:])
    ban = re.findall(r"禁:(\S+)", log)
    ban = next((b for b in ban if b != "-"), "-")
    print(f"禁用派系：{ban}")

    # 场上单位等级（states.jsonl 的 board）
    boards = {}
    if run and (run / "states.jsonl").exists():
        for line in (run / "states.jsonl").open(encoding="utf-8", errors="replace"):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            st = x.get("state") or {}
            r = st.get("round_no")
            b = x.get("board")
            if r and b:
                boards[r] = b          # 取每回合最后一帧
    print("\n回合 | 血 | 能量 | 等级 | 人口 | 主C | 场上（等级）")
    for r in sorted(set(head) | set(boards)):
        h = head.get(r, ("?",) * 5)
        b = boards.get(r) or {}
        units = []
        if isinstance(b, dict):
            for n, v in b.items():
                lv = (v or {}).get("lv") if isinstance(v, dict) else None
                units.append(f"{n}{lv if lv is not None else ''}")
        carry = re.search(r"主C ([^)）]+)", h[4] or "")
        print(f"R{r:>2} | {h[0]:>3} | {h[1]:>3} | {h[2]:>10} | {h[3]:>10} | {carry.group(1) if carry else '?'} | {' '.join(units)[:120]}")

    # 关键事件
    keys = ("锁定官方路线", "换备选", "换主C", "主C离场", "主C 已离场", "搜牌窗口", "先买（再考虑升级）", "升到搜牌等级",
            "进度表：", "枫叶舞", "觉醒回手", "卖掉", "换主C 收尾")
    print("\n关键事件：")
    seen = set()
    for line in log.splitlines():
        if "[INFO]" not in line and "[WARNING]" not in line:
            continue
        if "│" in line or "=>" in line or "场上布局" in line:
            continue
        if any(k in line for k in keys):
            msg = line.split("] ", 1)[-1][:150]
            sig = re.sub(r"\d+", "#", msg)
            if sig in seen:
                continue
            seen.add(sig)
            print("  " + line[11:19] + " " + msg)
    # 计数
    cnt = collections.Counter()
    for line in log.splitlines():
        if "执行动作: {'type': 'refresh_shop'" in line:
            cnt["刷新"] += 1
        elif "执行动作: {'type': 'buy_xp'" in line:
            cnt["升级"] += 1
        elif "执行动作: {'type': 'buy_piece'" in line:
            cnt["买牌"] += 1
        elif "没生效" in line:
            cnt["拖拽没生效"] += 1
        elif "搁置" in line:
            cnt["搁置"] += 1
        elif "Traceback" in line or "[ERROR]" in line:
            cnt["错误"] += 1
    print("\n计数：", dict(cnt))
    if run and (run / "ledger.jsonl").exists():
        ev = collections.Counter()
        fails = collections.Counter()
        for line in (run / "ledger.jsonl").open(encoding="utf-8", errors="replace"):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            ev[x.get("event")] += 1
            if x.get("event") in ("拖拽失败", "搁置"):
                fails[x.get("牌")] += 1
        print("账本事件：", dict(ev.most_common(12)))
        if fails:
            print("失败/搁置最多的牌：", dict(fails.most_common(6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
