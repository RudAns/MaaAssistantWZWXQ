"""正向推演 vs 真实对局：从第 R 回合最后一个准备帧（我们的操作都做完了）推一回合（战斗 → 回合结束 → 下回合整备），
和第 R+1 回合"整备结算完、我们还没动手"的读数逐个单位比等级。战斗是随机的，每个局面跑 n 次取平均。

  .venv/Scripts/python.exe -X utf8 tools/sim/replay_check.py [--runs 30] [--n 20]
输出：总误差 / "等级不变"基线；按单位名的误差（哪张卡的实现和游戏对不上，一眼看出来）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools" / "engine"))
from agent.sim.adapt import from_row  # noqa: E402
from agent.sim.engine import Engine  # noqa: E402
import test_engine as TE  # noqa: E402   _talents_by_round / _worn_by_round

LEVEL_ACTS = {"drag_card", "use_card", "deploy", "sell", "sell_card", "equip", "sequence", "move_unit"}


def played_boosts(d: Path) -> dict:
    """账本里打出过的、打出后整局生效的牌（枫叶舞强化：翩跹舞步·*、红叶染锋、卡面写"枫叶舞英雄"的强化）→ {回合: [到这回合为止打出过的]}。
    规则层的 lord_boosts 只记三选一选到的强化，拖出去打的翩跹舞步没记。"""
    from agent.sim.engine import card_text
    out, acc = {}, []
    f = d / "ledger.jsonl"
    if not f.exists():
        return out
    for l in f.open(encoding="utf-8"):
        try:
            e = json.loads(l)
        except ValueError:
            continue
        n = e.get("牌") or ""
        if e.get("event") == "用牌" and n and (n.startswith("翩跹舞步") or n == "红叶染锋" or "为枫叶舞挑选" in card_text(n)
                                               or "枫叶舞英雄" in card_text(n)) and n not in acc:
            acc.append(n)
            out[int(e.get("round") or 0)] = list(acc)
    return out


def frames(d: Path):
    rows = collections.defaultdict(list)
    for l in (d / "states.jsonl").open(encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        st = r.get("state") or {}
        if st.get("phase") == "prepare" and st.get("round_no") and r.get("board"):
            rows[int(st["round_no"])].append(r)
    return rows


def truth_next(rows_next: list, skip_s: float = 4.0) -> dict:
    """下回合每个单位的"真值"：整备在准备阶段开始后几秒才结算完（实测程咬金每回合 +2 出现在第 6~10 秒），
    所以按单位取"我们第一次对这个单位动手之前"的最后一个稳定读数（连续两帧一致）。
    "对它动手" = 执行过的拖拽落点在它的格子 / 合成它 / 卖它 / 挪它。拍卖、商店、对别人用牌不影响它。"""
    def touched(r, name, cell):
        for a in r.get("actions") or []:
            if not isinstance(a, dict) or a.get("type") not in LEVEL_ACTS:
                continue
            if a.get("piece") == name or a.get("target") == name:
                return True
            dst = a.get("dst") or a.get("src")
            if dst and cell and f"{int(dst[0])},{int(dst[1])}" == cell:
                return True
            if a.get("type") in ("sequence",) or (a.get("kind") in ("效果牌",) and not a.get("target")):
                return True          # 随机 / 全体类、组合动作：保守地当成可能动了所有人
        return False
    t0 = rows_next[0].get("ts") or 0
    out, prev, stopped = {}, {}, set()
    for r in rows_next:
        if (r.get("ts") or 0) < t0 + skip_s:
            continue
        for name, b in (r.get("board") or {}).items():
            if name in stopped:
                continue
            v = b.get("lv")
            if v and prev.get(name) == (v, b.get("cell")):
                out[name] = b
            if v:
                prev[name] = (v, b.get("cell"))
        if r.get("executed"):
            for name, b in (r.get("board") or {}).items():
                if name not in stopped and touched(r, name, b.get("cell")):
                    stopped.add(name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--detail", default="", help="逐样本打印这些单位（逗号分隔）：前 / 推演 / 实际 / 推演来源")
    ap.add_argument("--dump", default="", help="每个样本写一行 jsonl（单位 / 前 / 推演 / 实际 / 推演来源 / 场上 / 枫叶舞 / 天赋 / 强化），离线分析用")
    a = ap.parse_args()
    runs = sorted([d for d in (ROOT / "data/logs").glob("live_*") if (d / "states.jsonl").exists()],
                  key=lambda d: d.name)[-a.runs:]
    err, base = [], []
    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None
    by_name = collections.defaultdict(list)
    for d in runs:
        rows = frames(d)
        tal, worn = TE._talents_by_round(d), TE._worn_by_round(d)
        boosts = played_boosts(d)
        for R in sorted(rows):
            if R + 1 not in rows:
                continue
            last = rows[R][-1]
            nxt = truth_next(rows[R + 1])
            if not nxt:
                continue
            tk = [v for k, v in tal.items() if k <= R]
            wk = [k for k in worn if k <= R]
            pred = collections.defaultdict(list)
            srcs = collections.defaultdict(collections.Counter)
            want = set(x for x in a.detail.split(",") if x)
            if a.dump:
                want = {"*"}
            for i in range(a.n):
                s = from_row(last, talents=tk[-1] if tk else [], worn=worn[max(wk)] if wk else {})
                bk = [k for k in boosts if k <= R]
                for b in (boosts[max(bk)] if bk else []):
                    if b not in s.lord_cards:
                        s.lord_cards.append(b)
                e = Engine(s, random.Random(i))
                try:
                    e.battle(); e.round_end(); e.round_start()
                except Exception as ex:  # noqa: BLE001
                    print("推演出错", d.name, R, repr(ex)[:120])
                    break
                for u in s.heroes():
                    pred[u.name].append(u.level)
                if want:
                    for (_r, ev, kw) in s.log:
                        if ev == "等级" and (kw.get("unit") in want or "*" in want):
                            srcs[kw["unit"]][kw.get("src")] += kw.get("n", 0) / a.n
            b0 = last.get("board") or {}
            for name, bd in nxt.items():
                if name not in pred or not (b0.get(name) or {}).get("lv") or bd.get("cell") != (b0.get(name) or {}).get("cell"):
                    continue
                p = S.mean(pred[name])
                e1 = p - bd["lv"]
                err.append(e1)
                base.append(b0[name]["lv"] - bd["lv"])
                by_name[name].append(e1)
                if dump is not None:
                    dump.write(json.dumps({"run": d.name, "R": R, "unit": name, "before": b0[name]["lv"], "pred": round(p, 2),
                                           "actual": bd["lv"], "awk_left": b0[name].get("awk_left"), "eq": b0[name].get("eq"), "src": {k: round(v, 2) for k, v in srcs[name].items()},
                                           "board": {k: [v.get("lv"), v.get("cell")] for k, v in b0.items()},
                                           "maple": last.get("maple"), "carry": (last.get("state") or {}).get("carry") or last.get("carry"),
                                           "talents": tk[-1] if tk else [], "boosts": s.lord_cards}, ensure_ascii=False) + "\n")
                if name in want:
                    print(f"  {d.name} R{R} {name}: 前 {b0[name]['lv']} 推演 {p:.1f} 实际 {bd['lv']} 来源 "
                          f"{ {k: round(v, 1) for k, v in srcs[name].items()} } 枫叶舞={last.get('maple')} 天赋={(tk[-1] if tk else [])[:5]}")
    if not err:
        print("没有可比样本")
        return 1
    mae = lambda v: S.mean(abs(x) for x in v)  # noqa: E731
    print(f"样本 {len(err)}：推演 MAE {mae(err):.2f} 偏差 {S.mean(err):+.2f} | 等级不变基线 MAE {mae(base):.2f} 偏差 {S.mean(base):+.2f}")
    print("按单位（误差最大的 20 个，n≥3）：")
    rows = sorted(((mae(v), S.mean(v), len(v), n) for n, v in by_name.items() if len(v) >= 3), reverse=True)[:20]
    for m, b, n, name in rows:
        print(f"  {name:8s} n={n:3d} MAE {m:5.2f} 偏差 {b:+5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
