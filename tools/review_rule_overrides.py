"""每局复盘：规则有没有替模型做决定（2026-10-11 起每局检查残留规则是否影响了决策）。

读 data/logs/agent_one_<局号>.log，分四类统计（附例句）：
  1. 模型做的决定：Laya[题型] / 局面网络[题型] 被问到几次、执行了几次（执行 laya / cgnet）
  2. 规则抢先：执行的动作 why 不是模型给的（搜牌回合 / 合成材料 / 路线要的牌 / 按模拟 / 先买 ……）—— 模型根本没被问到
  3. 规则改写模型答案：日志里"改为 / 改选 / 不再改选"等（旧护栏）
  4. 规则删选项：日志里"去掉 …"（候选 / 选项被规则裁掉）；只查合法性的（不合规目标、买不起）单列为"合法性"
  5. 规则不执行模型决定的后续（2026-10-12 加）：日志里"不卖人 / 先留着 / 先不卖 /
     不为…卖 / 路线不跟 / 先别 / 不出…题"——模型已经定了（换主C / 换人 / 经济），规则拦着不去做后续动作
用法：.venv/Scripts/python.exe -X utf8 tools/review_rule_overrides.py 537 [538 ...]
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_RE = re.compile(r"(Laya|局面网络)\[(\w+)\] 选 (\S+?)（.*?→ 执行 (\w+)")
ACT_RE = re.compile(r"执行动作: \{'type': '(\w+)'")
WHY_RE = re.compile(r"'why': '([^']*)'")
NUM = re.compile(r"\d+(\.\d+)?")
# 只查合法性（不是价值判断）的删选项
LEGAL = ("合法性", "不合规", "CANNOT AFFORD", "买不起", "手牌满", "没有按钮", "已关店")
OVERRIDE = ("改为 这场不拍", "改为 ", "改选 ", "旧拍卖护栏")
BLOCK_RE = re.compile(r"不卖人|先留着|先不卖|不为.{0,8}卖|路线不跟|先别|不出.{0,6}题|先留在|不收尾|不换绑")
NOT_OVERRIDE = ("升级费按经验条改为", "人口数字是白色", "装备四选一", "（仅记录）")


def norm(s: str) -> str:
    return NUM.sub("#", s)


def review(n: str) -> str:
    f = ROOT / "data" / "logs" / f"agent_one_{n}.log"
    if not f.exists():
        return f"第 {n} 局：没有日志 {f}"
    asked, done = collections.Counter(), collections.Counter()
    pre, over, prune, legal = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    block = collections.Counter()
    last_model = -99
    for i, raw in enumerate(f.open(encoding="utf-8", errors="replace")):
        line = raw.split("] ", 1)[-1].strip()
        m = MODEL_RE.search(line)
        if m:
            if m.group(4) in ("laya", "cgnet") and m.group(2) not in ("eco",):
                last_model = i         # 紧跟着的那个动作是模型给的（Laya 商店题的动作不带 why）
            k = f"{m.group(2)}"
            asked[k] += 1
            if m.group(4) in ("laya", "cgnet"):
                done[k] += 1
            continue
        if line.startswith("执行动作:"):
            m = ACT_RE.search(line)
            if not m:
                continue
            w = WHY_RE.search(line)
            typ, why = m.group(1), (w.group(1) if w else "")
            if why.startswith(("Laya", "局面网络", "cgnet")) or (not why and i - last_model <= 3):
                continue
            if typ in ("buy_piece", "refresh_shop", "buy_xp", "sell_unit", "sell_card", "auction_bid", "lock_shop"):
                pre[f"{typ} ← {norm(why) or '(规则，无说明)'}"] += 1
            continue
        if any(x in line for x in NOT_OVERRIDE):
            continue
        if "（仅记录）" not in line and BLOCK_RE.search(line) and not line.startswith(("执行动作", "规则引擎决策", "[")):
            block[norm(line)[:110]] += 1
            continue
        if any(x in line for x in OVERRIDE):
            over[norm(line)[:110]] += 1
        elif "去掉" in line:
            (legal if any(x in line for x in LEGAL) else prune)[norm(line)[:110]] += 1
    out = [f"## 第 {n} 局"]
    out.append("模型被问到 / 执行： " + "，".join(f"{k} {asked[k]}/{done[k]}" for k in sorted(asked)))
    for title, c in (("规则抢先（模型没被问到）", pre), ("规则改写模型答案", over), ("规则删选项（价值判断）", prune),
                     ("规则不执行模型决定的后续 / 拦着不做", block), ("删选项（只查合法性，可保留）", legal)):
        out.append(f"- {title}：{sum(c.values())} 次")
        for k, v in c.most_common(12):
            out.append(f"    {v:>3}  {k}")
    return "\n".join(out)


if __name__ == "__main__":
    print("\n\n".join(review(a) for a in sys.argv[1:] or ["537"]))
    # 训练之后容易忘记撤掉之前写死的冲突规则 → 每次复盘都列出还开着的临时规则
    sys.path.insert(0, str(ROOT))
    from agent.decision import interim_rules as _IR
    _act = _IR.active()
    print(f"\n【还开着的临时规则 {len(_act)} 条】（agent/decision/interim_rules.py；对应模型学会后关掉、实战验证、再删代码）")
    for _line in _act:
        print(" -", _line)
