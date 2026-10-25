"""独立盯"觉醒回手 → 有没有重新上场"（2026-09-24：另开线程实时盯着会不会有卡牌因觉醒回到手牌）。

  python tools/watch_awaken.py data/logs/laya_round3_match3.out      # 跟着日志尾巴读，每个事件打印一行

不碰游戏、不改任何状态，只读 agent 的日志：
  - 「疑似觉醒回手」「**觉醒卡**」「已离场并回到手牌」→ 打印 AWAKEN <英雄>
  - 之后「账本: 上场 英雄=<它>」→ 打印 OK <英雄> 已重新上场（用时 N 秒）
  - 还没上场就切到战斗阶段 → 打印 ALERT <英雄> 觉醒回手后没上场就开战了
设计成 Monitor 的事件源：每行输出就是一条通知。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

PAT_AWK = [re.compile(r"手里有场上的 \[?'?([^'\]\s，,]+)"),
           re.compile(r"：(\S+?)（\*\*觉醒卡\*\*）"),
           re.compile(r"检测到 (\S+) 已离场并回到手牌")]
PAT_DEPLOY = re.compile(r"账本: 上场 英雄=(\S+)")
PAT_NOT_AWK = re.compile(r"（觉醒嫌疑）：(\S+?)，用时")          # 点开看过、不是觉醒卡（只是合成材料）
PAT_BATTLE = re.compile(r"阶段切换: \S+ -> battle")
PAT_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def follow(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.3)
                continue
            yield line


def main() -> int:
    path = Path(sys.argv[1])
    pending: dict[str, float] = {}
    for line in follow(path):
        for p in PAT_AWK:
            m = p.search(line)
            if m:
                name = m.group(1).strip()
                if name not in pending:
                    pending[name] = time.time()
                    ts = PAT_TS.search(line)
                    print(f"AWAKEN {name} 觉醒回手 {ts.group(1) if ts else ''}", flush=True)
        m = PAT_NOT_AWK.search(line)
        if m and m.group(1) in pending:
            pending.pop(m.group(1))
            print(f"CLEAR {m.group(1)} 点开看过，只是合成材料", flush=True)
        m = PAT_DEPLOY.search(line)
        if m and m.group(1) in pending:
            t0 = pending.pop(m.group(1))
            print(f"OK {m.group(1)} 已重新上场（用时 {time.time() - t0:.0f} 秒）", flush=True)
        if PAT_BATTLE.search(line) and pending:
            for name in list(pending):
                print(f"ALERT {name} 觉醒回手后没上场就开战了", flush=True)
            pending.clear()
        if "对局已结束" in line:
            print("END 对局结束", flush=True)
            return 0


if __name__ == "__main__":
    sys.exit(main())
