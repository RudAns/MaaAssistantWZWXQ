"""逐局强度统计（2026-09-30）：每局主C 到手回合、第 10 回合主C / 其他单位等级、买牌与刷新次数。用法：python tools/strength_report.py"""
import re, json, glob, sys
from pathlib import Path
rows = []
for f in sorted(glob.glob("data/logs/agent_one_*.log"), key=lambda p: Path(p).stat().st_mtime):
    t = Path(f).read_text(encoding="utf-8", errors="replace")
    rk = re.findall(r"本局名次：第 (\d) 名", t)
    if not rk:
        continue
    runs = re.findall(r"live_\d+", t)
    carry = re.findall(r"换主C（用户指定路线[^）]*）：\S+ -> (\S+)", t)
    carry = carry[-1] if carry else None
    first_round = None
    lv10 = None; sup = []
    route = re.findall(r"锁定官方路线（用户指定路线）：(\S+)（主C", t)
    if runs:
        sp = Path("data/logs") / runs[0] / "states.jsonl"
        if sp.exists():
            for line in sp.open(encoding="utf-8", errors="replace"):
                try: x = json.loads(line)
                except ValueError: continue
                st = x.get("state") or {}; b = x.get("board") or {}
                r = st.get("round_no") or 0
                if carry and carry in b and first_round is None and r:
                    first_round = r
                if r == 10 and b:
                    lv10 = (b.get(carry) or {}).get("lv") if carry else None
                    sup = [(n, (v or {}).get("lv")) for n, v in b.items() if n != carry]
    buys = t.count("执行动作: {'type': 'buy_piece'"); refr = t.count("执行动作: {'type': 'refresh_shop'")
    sl = [v for _, v in sup if isinstance(v, int)]
    rows.append((Path(f).stem.replace("agent_one_", ""), int(rk[-1]), (route[-1] if route else "?")[:10], carry, first_round, lv10,
                 round(sum(sl) / len(sl), 1) if sl else None, buys, refr))
print("局 | 名次 | 路线 | 主C | 主C到手回合 | R10主C等级 | R10其他平均 | 买牌 | 刷新")
for r in rows: print(" | ".join(str(x) for x in r))
