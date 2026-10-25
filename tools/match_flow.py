"""对局流水（纯文本）：把 data/logs/agent_one_<局号>.log 按回合整理成一份好扫的流水，写到 data/logs/flow_<局号>.txt。

2026-10-17（第 686 局：限制性进化把主C 马可波罗阵容里的米莱狄 / 狂铁变掉了，复盘时没有及时发现）：
日志要记得详细、每局都要看；多记录纯文本日志，便于观察。

每回合列出：回合头（血 / 能量 / 等级 / 人口 / 阵容）、场上 / 手牌、买 / 卖 / 卖手牌 / 上场 / 获得、用牌（给谁）、天赋 / N 选 1 / 拍卖、
换主C / 锁路线、模型决策（Laya / 局面网络，带把握）、**场上单位没卖却消失**（被变身 / 被吞 / 觉醒回手）。
对局进行中也能跑（读到哪算到哪）。

  .venv/Scripts/python.exe -X utf8 tools/match_flow.py 686 [--print]
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

TS = re.compile(r"^\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d)")
LEDGER_KEEP = ("买牌", "卖出", "卖手牌", "上场", "用牌", "换主C", "换主C收尾", "锁定路线", "换备选路线", "移除英雄", "N选1",
               "获得卡牌", "天赋", "拍卖", "对局结束")
MODEL = re.compile(r"(Laya\[[a-z_]+\]|局面网络\[[a-z]+\]) 选 (\S+)（([\d.]+)[^）]*）规则会选 (\S+) → 执行 (\S+?)(?:（(.*)）)?$")


def flow(n: int) -> list[str]:
    f = PROJECT_DIR / "data" / "logs" / f"agent_one_{n}.log"
    out: list[str] = [f"# 第 {n} 局流水（{f.name}）"]
    layout: dict = {}
    sold_recent: set = set()
    for raw in f.open(encoding="utf-8", errors="replace"):
        m = TS.match(raw)
        t = m.group(1) if m else ""
        line = raw.rstrip("\n")
        msg = line.split("[INFO] ", 1)[-1] if "[INFO] " in line else line.split("[WARNING] ", 1)[-1]
        if "┌─ 回合" in msg:
            out += ["", f"{t} {msg.lstrip('┌─ ').strip()}"]
            continue
        if msg.startswith("│  场上") or msg.startswith("│  手牌") or msg.startswith("└  "):
            out.append(f"         {msg.strip('│└ ').strip()}")
            continue
        if msg.startswith("选定阵容:") or msg.startswith("锁定官方路线"):
            out.append(f"{t}   [路线] {msg}")
            continue
        if msg.startswith("账本: "):
            body = msg[4:]
            kind = body.split(" ", 1)[0]
            if kind in LEDGER_KEEP:
                if kind == "获得卡牌" and "来源=购买" in body:
                    continue
                out.append(f"{t}   {body}")
                if kind in ("卖出", "移除英雄"):
                    mm = re.search(r"英雄=(\S+)", body)
                    if mm:
                        sold_recent.add(mm.group(1))
            continue
        if msg.startswith("用「"):
            continue        # 规则每帧都会重复打印"用「X」给 Y"的决定（没执行也会打），只看账本里真正执行的"用牌"
        mm = MODEL.search(msg)
        if mm:
            who, ch, conf, rule, used, what = mm.groups()
            if used in ("shadow",):
                continue
            out.append(f"{t}   [{who}] {ch} {conf}（规则 {rule}，执行 {used}）{('：' + what) if what else ''}")
            continue
        if "天赋三选一" in msg or "'type': 'talent_pick'" in msg and "执行动作" in msg:
            out.append(f"{t}   [天赋] {msg[:160]}")
            continue
        if msg.startswith("场上布局: "):
            try:
                cur = ast.literal_eval(msg[len("场上布局: "):])
            except (ValueError, SyntaxError):
                continue
            gone = [x for x in layout if x not in cur and x not in sold_recent]
            for x in gone:
                out.append(f"{t}   ⚠ 场上 {x} 消失（账本没有卖出 —— 被变身 / 被吞 / 觉醒回手？）")
            layout = cur
            sold_recent = set()
            continue
        if "本局名次" in msg:
            out += ["", f"{t} {msg}"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("match", type=int)
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    lines = flow(a.match)
    dst = PROJECT_DIR / "data" / "logs" / f"flow_{a.match}.txt"
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if a.print:
        print("\n".join(lines))
    print(f"写入 {dst}（{len(lines)} 行）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
