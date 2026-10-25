"""每局自动复盘：资源浪费 / 错过机会清单（2026-10-12：这些问题应由程序按游戏玩法自动发掘，不依赖人工提醒）。

  .venv/Scripts/python.exe -X utf8 tools/audit_match.py 582            # 一局
  .venv/Scripts/python.exe -X utf8 tools/audit_match.py 575 582        # 一段（最后附汇总）

只读 data/logs/live_*/{states,ledger,laya_points}.jsonl 和卡库，不改任何文件。检查项（按游戏玩法，不看名次也该查的）：
  A 主C 到手回合 + 主C 等级曲线（对照前三名的局在同回合的中位数）
  B 手里压着没用的牌：装备 / 效果牌连续 ≥3 个回合在手（5 阶神装单独标）
  C 加等级牌 / 装备给了之后被卖掉或换掉的人（资源白扔）
  D 路线最低配套每一件第几回合上场；商店里出现过、当回合没买的
  E 主C 没定时，商店出现过的主C 人选 / 抽英雄牌没拿
  F 规则或模型卖掉的人是觉醒过的 / 最低配套件
  G 准备阶段结束时剩的能量（≥10 记一次）
"""
from __future__ import annotations

import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVEL_RE = re.compile(r"等级\s*[+＋]\s*\d|提升\s*\d+(?:-\d+)?\s*级|等级提升")


def jl(f: Path) -> list:
    if not f.exists():
        return []
    out = []
    for line in f.open(encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


_CARDS = None


def cards() -> dict:
    global _CARDS
    if _CARDS is None:
        c = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))
        c = c if isinstance(c, list) else list((c.get("cards") or c).values()) if isinstance(c, dict) else c
        out = {}
        for x in c:   # 同名多条（召唤物版的铠 / 李元芳）：英雄牌优先
            if isinstance(x, dict) and x.get("name") and (x["name"] not in out or x.get("category") == "英雄牌"):
                out[x["name"]] = x
        _CARDS = out
    return _CARDS


def cat(n): return (cards().get(n) or {}).get("category", "")
def tier(n): return (cards().get(n) or {}).get("quality") or 0
def text(n): return (cards().get(n) or {}).get("skill") or ""


def matches() -> dict:
    out = {}
    for r in jl(ROOT / "data" / "laya" / "iterations" / "batch2_matches.jsonl"):
        if r.get("match") and r.get("run"):
            out[r["match"]] = r
    return out


def rounds_of(run: str) -> dict:
    """回合 -> 该回合最后一个准备帧（带 board / carry）。"""
    last = {}
    for r in jl(ROOT / "data" / "logs" / run / "states.jsonl"):
        s = r.get("state") or {}
        if s.get("phase") == "prepare" and s.get("round_no"):
            last[s["round_no"]] = r
    return last


def carry_lv(r: dict):
    c = r.get("carry")
    b = r.get("board") or {}
    v = b.get(c) if isinstance(b, dict) else None
    return c, (v.get("lv") if isinstance(v, dict) else None)


_BENCH = None


def bench() -> dict:
    """前三名的局：每回合主C 等级中位数（最近 60 局里的前三）。"""
    global _BENCH
    if _BENCH is None:
        per = collections.defaultdict(list)
        ms = sorted(matches().items())[-80:]
        for _, m in ms:
            if not m.get("rank") or m["rank"] > 3:
                continue
            for rd, r in rounds_of(m["run"]).items():
                _, lv = carry_lv(r)
                if lv:
                    per[rd].append(lv)
        _BENCH = {rd: statistics.median(v) for rd, v in per.items() if len(v) >= 3}
    return _BENCH


def min_engine(route_name: str) -> list:
    d = json.loads((ROOT / "data" / "route_min_engine.json").read_text(encoding="utf-8"))
    for k, v in d.items():
        if isinstance(v, dict) and v.get("name") == route_name:
            return [str(u.get("unit") or "") for u in v.get("min_set") or [] if not str(u.get("unit") or "").startswith("（牌）")]
    return []


def components(run: str) -> list:
    """这局实际在用的决策组件（2026-10-15 第 631 局教训：复盘第一步先列这个 —— 推演器做好了却从没被线上调用，逐步分析时没发现）。
    各题型谁答（Laya / 局面网络）、各多少题、Laya checkpoint 与题面版本、局面网络各题型的部署包与特征开关、推演器进了哪里。"""
    pts = jl(ROOT / "data" / "logs" / run / "laya_points.jsonl")
    if not pts:
        return ["  （没有 laya_points.jsonl）"]
    by = collections.defaultdict(collections.Counter)
    for p in pts:
        by[p.get("kind")][str(p.get("model") or "?") + ("" if p.get("used", True) else "(影子)")] += 1
    ck = collections.Counter(str(p.get("ckpt") or "?").replace("/checkpoint", "").split("/")[-1] for p in pts)
    fv = collections.Counter(str(p.get("feature_version") or "?") for p in pts)
    out = ["  组件：Laya " + ", ".join(f"{k}×{v}" for k, v in ck.most_common()) + "；题面 " + ", ".join(f"{k}×{v}" for k, v in fv.most_common()),
           "  各题型谁答：" + "；".join(f"{k} " + "/".join(f"{m}{n}" for m, n in c.most_common()) for k, c in sorted(by.items()))]
    try:
        router = json.loads((ROOT / "data" / "cgnet" / "deploy" / "router.json").read_text(encoding="utf-8"))
        bits, sim_feat = [], False
        for k, d in sorted(router.items()):
            man = json.loads((ROOT / "data" / "cgnet" / "deploy" / d / "manifest.json").read_text(encoding="utf-8"))
            fl = [f for f in ("min_engine_feat", "eco_core_feat", "engine_feat", "flow_feat") if man.get(f)]
            bits.append(f"{k}={d}[{','.join(fl) or '无附加特征'}]")
            sim_feat = sim_feat or bool(man.get("engine_feat") or man.get("flow_feat"))
        out.append("  局面网络（当前 router，不一定是开局时的）：" + "；".join(bits))
    except (OSError, ValueError):
        sim_feat = False
    flow = sum(1 for p in pts if "FLOW OPTIONS" in str(p.get("state") or ""))
    out.append(f"  推演器（agent/sim）：流派库进 Laya 题面 {flow}/{len(pts)} 题；推演打分 / 流派特征进局面网络：{'是' if sim_feat else '否'}；"
               "推演标签：否（推演只在离线工具里跑）")
    return out


def audit(no: int) -> dict:
    m = matches().get(no)
    if not m:
        print(f"第 {no} 局不在 batch2_matches.jsonl")
        return {}
    run = m["run"]
    R = rounds_of(run)
    led = jl(ROOT / "data" / "logs" / run / "ledger.jsonl")
    F = []                                   # (类别, 文字)
    # A 主C
    final_c = next((R[rd].get("carry") for rd in sorted(R, reverse=True) if R[rd].get("carry")), "")
    arrive = next((rd for rd in sorted(R) if final_c in (R[rd].get("board") or {})), None)
    curve = []
    for rd in sorted(R):
        c, lv = carry_lv(R[rd])
        b = bench().get(rd)
        curve.append(f"R{rd} {c or '-'} {lv if lv is not None else '?'}" + (f"/{b:.0f}" if b else ""))
        if rd >= 6 and lv is not None and b and lv < 0.6 * b:
            F.append(("A主C等级", f"R{rd} 主C {c} {lv} 级，前三名中位 {b:.0f} 级（<60%）"))
    if arrive and arrive >= 5:
        F.append(("A主C到手晚", f"最终主C {final_c} 第 {arrive} 回合才上场"))
    # B 手里压着的牌
    held = collections.defaultdict(list)
    for rd in sorted(R):
        for n in set(R[rd]["state"].get("hand") or []):
            if cat(n) in ("装备牌", "效果牌"):
                held[n].append(rd)
    for n, rds in held.items():
        if len(rds) >= 3:
            tag = "B神装闲置" if cat(n) == "装备牌" and tier(n) >= 5 else "B牌压手"
            F.append((tag, f"{n}（{cat(n)} t{tier(n)}）在手 R{rds[0]}~R{rds[-1]} 共 {len(rds)} 回合"))
    # C 资源给了被换掉的人
    sold = {e.get("英雄") or e.get("卖"): e.get("round") for e in led if e.get("event") in ("卖出", "换主C收尾") and (e.get("英雄") or e.get("卖"))}
    for e in led:
        if e.get("event") != "用牌":
            continue
        n, t = e.get("牌", ""), e.get("目标", "")
        if t in sold and (cat(n) == "装备牌" or LEVEL_RE.search(text(n))):
            F.append(("C资源白扔", f"R{e.get('round')} {n} 给了 {t}，R{sold[t]} {t} 被卖"))
    # D 最低配套
    route = next((e.get("阵容") for e in reversed(led) if e.get("event") == "锁定路线"), "")
    for unit in min_engine(route):
        names = [x for x in re.split(r"\s*或\s*|/|、", unit) if x in cards()]
        if not names or final_c in names:
            continue
        on = next((rd for rd in sorted(R) if any(x in (R[rd].get("board") or {}) for x in names)), None)
        seen = sorted({rd for rd in R for s in (R[rd]["state"].get("shop") or []) if s.get("piece") in names and (on is None or rd < on)})
        if on is None or on >= 6:
            F.append(("D配套晚", f"{'/'.join(names)}：" + (f"第 {on} 回合上场" if on else "整局没上场")
                      + (f"；之前商店出现过 R{','.join(map(str, seen))}" if seen else "；之前商店没出现")))
    # E 主C 没定时错过
    for rd in sorted(R):
        if arrive and rd >= arrive:
            break
        for s in R[rd]["state"].get("shop") or []:
            n = s.get("piece", "")
            if n and ("抽" in text(n) or "获得" in text(n) and "英雄牌" in text(n)) and cat(n) == "效果牌":
                F.append(("E抽英雄牌", f"R{rd} 主C 未到手，商店有 {n}"))
    # F 卖掉觉醒 / 配套
    me_names = {x for u in min_engine(route) for x in re.split(r"\s*或\s*|/|、", u)}
    for e in led:
        if e.get("event") == "卖出":
            h = e.get("英雄", "")
            if h in me_names:
                F.append(("F卖配套", f"R{e.get('round')} 卖了最低配套件 {h}（{e.get('原因', '')[:40]}）"))
    # H 无关英雄占位：1 阶、不是主C、不在路线最终阵容 / 最低配套里，主C 到手后还在场 ≥3 回合（第 583 局干将）
    final_team = set()
    try:
        sys.path.insert(0, str(ROOT))
        from agent.decision import routes as _R
        rt = next((x for x in _R.ali_lineups() + _R.custom_lineups() if x.get("name") == route), None) if hasattr(_R, "custom_lineups") else None
        if rt is None:
            rt = next((x for x in _R.ali_lineups() if x.get("name") == route), None)
        final_team = set(_R.final_team(rt)) if rt else set()
    except Exception:                                    # noqa: BLE001
        pass
    idle = collections.defaultdict(list)
    for rd in sorted(R):
        if not arrive or rd <= arrive:
            continue
        for n in (R[rd].get("board") or {}):
            if n != final_c and tier(n) == 1 and n not in final_team and n not in me_names and cat(n) == "英雄牌":
                idle[n].append(rd)
    for n, rds in idle.items():
        if len(rds) >= 3:
            F.append(("H无关占位", f"{n}（1 阶、不在路线里）主C 到手后还在场 R{rds[0]}~R{rds[-1]} 共 {len(rds)} 回合"))
    # G 剩钱
    for rd in sorted(R):
        g = R[rd]["state"].get("gold") or 0
        if g >= 10:
            F.append(("G剩钱", f"R{rd} 准备阶段最后一帧还剩 {g} 能量"))
    print(f"## 第 {no} 局 第 {m.get('rank')} 名 {m.get('ban', '')} 路线 {route} 最终主C {final_c}")
    for line in components(run):
        print(line)
    print("  主C 等级（本局/前三中位）：" + " | ".join(curve))
    for k, v in F:
        print(f"  [{k}] {v}")
    return {"rank": m.get("rank"), "tags": collections.Counter(k for k, _ in F)}


def main() -> int:
    a = [int(x) for x in sys.argv[1:]] or [max(matches())]
    lo, hi = (a[0], a[-1])
    tot = collections.Counter()
    n = 0
    for no in range(lo, hi + 1):
        r = audit(no)
        if r:
            tot.update(r["tags"]); n += 1
    if n > 1:
        print(f"\n## 汇总（{n} 局）：" + "，".join(f"{k} {v}" for k, v in tot.most_common()))
    # 组件生效检查（2026-10-15：推演器曾连续多天未生效而无人处理）：❌ = 待办，不是信息
    import subprocess
    print("\n## 组件生效检查（❌ 必须处理：接上，或写明已决定不用）")
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tools" / "check_components.py"), str(hi)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
