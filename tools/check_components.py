"""组件生效检查（2026-10-15）：检查各模型、推演器是否真正在线上起作用；
避免组件长期"没用上"却无人处理。

  .venv/Scripts/python.exe -X utf8 tools/check_components.py            # 最近 5 局
  .venv/Scripts/python.exe -X utf8 tools/check_components.py 641        # 指定局（同时看它之前的 4 局）

登记表 COMPONENTS：每个"应该在线上起作用"的组件 + 怎么数它这局实际影响了几次决策。
**0 次 = ❌，连续 ≥2 局 ❌ = 必须处理**（不是"已知情况"——要么接上，要么在这里写明"已决定不用"并改 expected=False）。
有 ❌ 时退出码 1；audit_match.py 第一屏会调用它。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jl(f: Path) -> list:
    out = []
    if f.exists():
        for line in f.open(encoding="utf-8", errors="ignore"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _pts(run: str) -> list:
    # 一局中途重启会分成几个 live 目录（matches() 用 "+" 连起来，第 851 / 852 局）
    return [r for d in run.split("+") for r in jl(ROOT / "data" / "logs" / d / "laya_points.jsonl")]


def _states(run: str) -> list:
    out = []
    for r in (r for d in run.split("+") for r in jl(ROOT / "data" / "logs" / d / "states.jsonl")):
        s = r.get("state") or {}
        if s.get("phase") == "prepare":
            out.append(s)
    return out


def _model_kind(run, model, kinds):
    return sum(1 for p in _pts(run) if p.get("model") == model and p.get("kind") in kinds and p.get("used") != "shadow")


def _match_logs(match_no) -> list:
    """agent_one_<局号>.log 加中途重启前的 agent_one_<局号>a.log / b.log。"""
    pre = f"agent_one_{match_no}"
    return [f for f in sorted((ROOT / "data" / "logs").glob(pre + "*.log")) if f.stem[len(pre):].isalpha() or f.stem == pre]


def _log_lines(run: str, needle: str, match_no: int | None) -> int:
    if not match_no:
        return 0
    # 中途重启前的半局在 agent_one_<局号>a.log / b.log（第 851 / 852 局只数了后半段，Laya 次数 24 / 14、局面网络误报 ❌）
    fs = _match_logs(match_no)
    return sum(1 for f in fs for line in f.open(encoding="utf-8", errors="ignore") if needle in line)


def _sim_labels() -> int:
    """线上学生（one_match.sh 的 --laya-ckpt）训练用的标签里，有多少条来自推演器（source=sim）或参考了推演（sim_ref）。"""
    import re
    sh = (ROOT / "data" / "laya" / "iterations" / "one_match.sh").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"runs/([^/]+)/checkpoint", sh)
    if not m:
        return 0
    cfg = ROOT / "data" / "laya" / "runs" / m.group(1) / "config.json"
    try:
        lab = json.loads(cfg.read_text(encoding="utf-8")).get("labels") or ""
    except (OSError, ValueError):
        return 0
    f = ROOT / lab if lab and not Path(lab).is_absolute() else Path(lab)
    return sum(1 for r in jl(f) if r.get("source") == "sim" or r.get("sim_ref")) if lab else 0


# name, 说明, expected（应该在线上起作用）, 计数函数(run, match_no) -> 影响了几次决策
COMPONENTS = [
    ("Laya 学生", "商店 / N 选 1 / 天赋 / 拍卖 / 用牌 / 枫叶舞强化", True,
     lambda run, no: _model_kind(run, "laya", {"shop", "card_pick", "talent", "auction", "use_card", "lord_boost", "route"})),
    # 10-21：换主C 只有唯一人选（1 阶占位让位给锁定路线主C / 唯一合格）时根本不出题 → 这一局记"无题"（None），不算 0（第 852 / 856 / 857 局误报）
    ("局面网络 carry", "换主C", True,
     lambda run, no: (_model_kind(run, "cgnet", {"carry"})
                      or (0 if any(p.get("kind") == "carry" for p in _pts(run)) else None))),
    # 2026-10-17 起：换人题交还 Laya（局面网络在"路线件等位、场上有路线外杂牌"时反复不换；Laya R36 同类题全对）
    ("Laya swap", "卖谁腾位（10-17 起由 Laya 答）", True, lambda run, no: _model_kind(run, "laya", {"swap"})),
    ("局面网络 eco", "升级 / 刷新 / 攒钱", True, lambda run, no: _model_kind(run, "cgnet", {"eco"})),
    ("认人模型 unit_id", "场上单位是谁（校正账本）", True,
     lambda run, no: sum(1 for s in _states(run) if s.get("unit_names"))),
    ("界面分类 screen_cls", "认不出界面时接管", True,
     lambda run, no: _log_lines(run, "【界面模型接管】", no)),
    # 2026-10-16 晚起：推演器只用于辅助出题（不当标签、不进线上）。核对：商店刷到缺的人的概率准，
    # 等级预测偏慢 0.4~0.65 级 → 校正后再把"刷到核心的概率 / 升级 vs 刷新分差"写进商店题（待办）。接上之前不按"每局必须起作用"计。
    ("推演器 agent/sim", "定位：辅助出题（2026-10-16 起）；当前线上学生训练标签里来自推演 / 参考推演的题数（接进题面前为 0，已决定暂不计）", False,
     lambda run, no: _sim_labels()),
    ("use_timing 影子题", "规则按审定表扣着的牌，问 Laya 现在用 / 先留着（攒数据）", False,
     lambda run, no: sum(1 for p in _pts(run) if p.get("kind") == "use_timing")),
]


def matches() -> list:
    out = []
    for m in jl(ROOT / "data" / "laya" / "iterations" / "batch2_matches.jsonl"):
        if m.get("match") and m.get("run"):
            out.append(m)
    # 一局中途重启会有几个 live 目录：按 agent_one_<局号>*.log 里提到的 live_ 目录合并（第 851 / 852 局）
    for m in out:
        segs = []
        for f in _match_logs(m["match"]):
            for d in re.findall(r"live_\d{6,}", f.read_text(encoding="utf-8", errors="ignore")):
                if d not in segs and (ROOT / "data" / "logs" / d).is_dir():
                    segs.append(d)
        if m["run"] not in segs:
            segs.append(m["run"])
        m["run"] = "+".join(sorted(segs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("match", type=int, nargs="?")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    ms = matches()
    if a.match:
        ms = [m for m in ms if m["match"] <= a.match]
    ms = ms[-a.n:]
    if not ms:
        print("没有对局记录")
        return 0
    head = "组件".ljust(18) + " ".join(f"{m['match']:>5}" for m in ms) + "   结论"
    print(head)
    bad = []
    for name, desc, expected, fn in COMPONENTS:
        cnt = [fn(m["run"], m["match"]) for m in ms]
        streak = 0
        for c in reversed(cnt):
            if c is None:
                continue                  # 这一局没有可答的题（不算起作用、也不算 0）
            if c > 0:
                break
            streak += 1
        if expected and streak >= 1:
            verdict = f"❌ 连续 {streak} 局没起作用" + ("（必须处理）" if streak >= 2 else "")
            bad.append((name, desc, streak))
        elif expected:
            verdict = "✅"
        else:
            verdict = "（影子 / 不要求）"
        print(f"{name:<18}" + " ".join(f"{('无题' if c is None else c):>5}" for c in cnt) + "   " + verdict)
    if bad:
        print("\n必须处理（接上它，或者在 check_components.COMPONENTS 里写明'已决定不用'并改 expected=False）：")
        for name, desc, st in bad:
            print(f"  - {name}：{desc}（连续 {st} 局为 0）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
