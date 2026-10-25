"""用当前特征版本（features.FEATURE_VERSION）重新渲染一个已有数据集的题面 —— **id、划分、老师标签都不动**。

  python tools/laya/rerender.py --src r2 --out r2f5

为什么不直接用 build_dataset 重建：build_dataset 按"对局洗牌"划分，多一局测试集就变了；
论文要求每一轮都在同一批测试题上报告，所以换特征只能在原数据集上重新渲染文本。

每道题保存了 ctx（CTX_KEYS）、raw（商店/候选原始信息）、field（场上名单），f5 起还有 board（场上事实）。
老数据没有 board → 场上只有名字（和 f4 一样），但 f5 的概率信息、去掉血量这些照样生效。
渲染用的就是 build_dataset / laya_policy 共用的那几个 features 函数（train/serve 一致）。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))

import features as F  # noqa: E402


_TIMELINES: dict = {}


def board_timeline(run: str) -> list:
    """一局的场上读数时间线 [(ts, 沿用后的 board)]（states.jsonl 的 "board"，用 build_dataset.sticky_board 逐帧合并）。"""
    if run in _TIMELINES:
        return _TIMELINES[run]
    import build_dataset as B
    out, prev = [], None
    f = PROJECT_DIR / "data" / "logs" / run / "states.jsonl"
    if f.exists():
        for line in f.open(encoding="utf-8"):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            b = x.get("board")
            if b is None:
                continue
            if isinstance(b, str):
                import ast
                try:
                    b = ast.literal_eval(b)
                except (ValueError, SyntaxError):
                    continue
            prev = B.sticky_board(prev, b)
            try:
                out.append((float(x.get("ts")), prev))
            except (TypeError, ValueError):
                pass
    _TIMELINES[run] = out
    return out


def backfill(p: dict) -> dict:
    """题目当时读不到（None）的等级/觉醒/装备/格子，用这局在题目时刻之前最后一次读到的值补上（同名同格才补）。"""
    run, ts = p.get("run"), p.get("ts")
    if not run or ts is None:
        return p
    tl = board_timeline(run)
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return p
    last = None
    for t, b in tl:
        if t > ts + 0.5:
            break
        last = b
    if not last:
        return p
    board = {n: dict(v) for n, v in (p.get("board") or {}).items()}
    for n in (p.get("field") or []):
        cur = board.setdefault(n, {})
        pv = last.get(n)
        if not isinstance(pv, dict):
            continue
        if cur.get("cell") is None and pv.get("cell") is not None:
            cur["cell"] = pv["cell"]
        if cur.get("cell") == pv.get("cell"):
            for k in ("lv", "awk_left", "eq", "pool_left"):
                if cur.get(k) is None and pv.get(k) is not None:
                    cur[k] = pv[k]
    out = dict(p)
    out["board"] = board
    return out


_FRAMES: dict = {}
ALLOW_NEW_KEYS = False


def shop_full_at(run: str, ts) -> list | None:
    """f9d：历史商店题没存"整个商店"，按 (对局, 时间戳) 回 states.jsonl 找那一帧的商店（含买不起的）。"""
    if not run or ts is None:
        return None
    if run not in _FRAMES:
        rows = {}
        f = PROJECT_DIR / "data" / "logs" / run / "states.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                if '"shop"' not in line:
                    continue
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                s = x.get("state") or {}
                if s.get("shop"):
                    rows[round(float(x.get("ts") or 0), 3)] = [F.clean_name(sl["piece"]) for sl in s["shop"]
                                                                if sl.get("piece") and not sl.get("sold")]
        _FRAMES[run] = rows
    try:
        return _FRAMES[run].get(round(float(ts), 3))
    except (TypeError, ValueError):
        return None


_CARRY_HIST: dict = {}


def carry_hist_at(run: str, ts) -> dict:
    """f9g：历史题没存主C 的来历 —— 按 states.jsonl 逐帧回推到这一刻（和 build_dataset.CarryHistory 同一套）。"""
    import bisect
    import build_dataset as B
    if run not in _CARRY_HIST:
        snaps, ch = [], B.CarryHistory()
        f = PROJECT_DIR / "data" / "logs" / run / "states.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                if '"carry"' not in line:
                    continue
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                if x.get("board") is None:
                    continue
                s = x.get("state") or {}
                ch.update(x.get("carry") or "", s.get("round_no") or 0)
                b = x.get("board")
                if isinstance(b, dict):
                    ch.see_board(list(b), s.get("level"))      # f9lw ㊳：单位上场时的棋手等级（弈星回本）
                snaps.append((float(x.get("ts") or 0), ch.ctx()))
        _CARRY_HIST[run] = snaps
    snaps = _CARRY_HIST[run]
    try:
        i = bisect.bisect_right([t for t, _ in snaps], float(ts)) - 1
    except (TypeError, ValueError):
        i = -1
    return dict(snaps[i][1]) if i >= 0 else {"carry_since": 0, "carry_prev": "", "carry_switches": 0, "lordxp_since": {}}


_ENTERED: dict = {}


def entered_at(run: str, ts) -> dict:
    """f9i：历史题的场上名单没存"第几回合上场" —— 按 states.jsonl 的 board 逐帧回推（和 rules.board_facts 同一套：首次出现记当回合，离场即删）。"""
    import bisect
    if run not in _ENTERED:
        snaps, ent = [], {}
        f = PROJECT_DIR / "data" / "logs" / run / "states.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                if '"board"' not in line:
                    continue
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                b = x.get("board")
                if isinstance(b, str):
                    import ast
                    try:
                        b = ast.literal_eval(b)
                    except (ValueError, SyntaxError):
                        continue
                if not isinstance(b, dict):
                    continue
                rd = (x.get("state") or {}).get("round_no") or 0
                for n in [k for k in ent if k not in b]:
                    ent.pop(n)
                if rd:
                    for n in b:
                        ent.setdefault(n, rd)
                snaps.append((float(x.get("ts") or 0), dict(ent)))
        _ENTERED[run] = snaps
    snaps = _ENTERED[run]
    try:
        i = bisect.bisect_right([t for t, _ in snaps], float(ts)) - 1
    except (TypeError, ValueError):
        i = -1
    return snaps[i][1] if i >= 0 else {}


_ROUTE: dict = {}


def route_at(run: str, ts, rnd) -> dict:
    """f9k：历史题当时锁定的官方路线。新日志逐帧记了 record["route"]（按 ts 取最近一帧）；
    老日志按账本「锁定路线」的回合（build_dataset.route_from_ledger，与离线出题同一套）。"""
    import bisect
    import build_dataset as BD
    if run not in _ROUTE:
        snaps = []
        f = PROJECT_DIR / "data" / "logs" / run / "states.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                if '"route"' not in line:
                    continue
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                if "route" in x:
                    snaps.append((float(x.get("ts") or 0), x["route"] or {}))
        _ROUTE[run] = (snaps, BD.route_from_ledger(BD.load_ledger(PROJECT_DIR / "data" / "logs" / run)))
    snaps, (key, since) = _ROUTE[run]
    if snaps:
        try:
            i = bisect.bisect_right([t for t, _ in snaps], float(ts)) - 1
        except (TypeError, ValueError):
            i = -1
        r = snaps[i][1] if i >= 0 else {}
        return {"route": r.get("route") or "", "route_since": r.get("route_since") or 0}
    if key and (rnd or 0) >= since:
        return {"route": key, "route_since": since}
    return {"route": "", "route_since": 0}


_TAL_CACHE: dict = {}


def talents_at(run: str, ts: float) -> list:
    """这道题之前本局已选的天赋（与局面网络 eval_deploy.owned_at 同一套还原）。"""
    if not run:
        return []
    sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
    from eval_deploy import owned_at
    return sorted(owned_at(run, ts, _TAL_CACHE))


_WORN_CACHE: dict = {}


def worn_at(run: str, rnd: int) -> dict:
    """f9ly：出题回合之前（不含本回合）各英雄身上穿的装备 —— 从 ledger.jsonl 的"用牌 类别=装备牌 目标=X"还原；卖出英雄清空。"""
    if not run:
        return {}
    if run not in _WORN_CACHE:
        ev = []
        f = PROJECT_DIR / "data" / "logs" / run / "ledger.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass
        _WORN_CACHE[run] = ev
    out: dict = {}
    for e in _WORN_CACHE[run]:
        if (e.get("round") or 0) >= (rnd or 0):
            break
        if e.get("event") == "用牌" and e.get("类别") == "装备牌" and e.get("目标"):
            out.setdefault(e["目标"], []).append(e.get("牌"))
        elif e.get("event") == "卖出" and e.get("英雄"):
            out.pop(e["英雄"], None)
    return {h: sorted(v) for h, v in out.items() if v}


_ECO_CACHE: dict = {}


def eco_at(run: str, rnd: int):
    """f9lu：这一局这一回合经济网络（eco 题）答了什么（laya_points 里 kind=eco 的 laya_choice）。没有就 None。"""
    if not run:
        return None
    if run not in _ECO_CACHE:
        m = {}
        f = Path(__file__).resolve().parents[2] / "data" / "logs" / run / "laya_points.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8", errors="ignore"):
                try:
                    q = json.loads(line)
                except ValueError:
                    continue
                if q.get("kind") == "eco" and q.get("round") and q.get("laya_choice"):
                    m.setdefault(q["round"], q["laya_choice"])
        _ECO_CACHE[run] = m
    return _ECO_CACHE[run].get(rnd)


def rerender(p: dict) -> dict | None:
    ctx = dict(p.get("ctx") or {})
    if F._v("f9k") and "route" not in ctx and p.get("kind") != "route":
        ctx.update(route_at(p.get("run"), p.get("ts"), ctx.get("round")))
    if F._v("f9g") and "carry_since" not in ctx:
        ctx.update(carry_hist_at(p.get("run"), p.get("ts")))
    if F._v("f9lw") and "lordxp_since" not in ctx:
        ctx["lordxp_since"] = carry_hist_at(p.get("run"), p.get("ts")).get("lordxp_since") or {}
    if F._v("f9li") and "talents" not in ctx:
        ctx["talents"] = talents_at(p.get("run") or "", float(p.get("ts") or 0))   # f9li：已选天赋（从 states.jsonl 还原）
    if F._v("f9ly") and "worn" not in ctx:
        ctx["worn"] = worn_at(p.get("run") or "", ctx.get("round") or p.get("round") or 0)   # f9ly：穿戴账本（拆装备写装备名）
    board = p.get("board") or {}
    ctx["field"] = {n: dict(board.get(n) or {}) for n in (p.get("field") or [])}
    if F._v("f9lu") and p.get("kind") == "shop" and "eco_plan" not in ctx:
        ctx["eco_plan"] = eco_at(p.get("run") or "", p.get("round") or ctx.get("round") or 0)   # f9lu：本回合经济网络的决定
    if F._v("f9i") and p.get("kind") == "swap" and not any("in_rd" in v for v in ctx["field"].values()):
        ent = entered_at(p.get("run"), p.get("ts"))
        for n, v in ctx["field"].items():
            if n in ent:
                v["in_rd"] = ent[n]
    raw = p.get("raw") or {}
    crit = p["question"]["criteria"]
    kind = p["kind"]
    if kind == "shop":
        shop = [tuple(x) for x in raw.get("shop", [])]
        if F._v("f9d") and "_shop_full" not in ctx:
            full = raw.get("shop_full")
            ctx["_shop_full"] = full if full is not None else shop_full_at(p.get("run"), p.get("ts"))
        q = F.q_shop(shop, ctx, "refresh" in crit, "levelup" in crit)
        extra = [n for _, n, _ in F.dedupe_shop(shop)]
    elif kind == "card_pick":
        opts = raw.get("cards", [])
        q, extra = F.q_card_pick(opts, ctx), opts
    elif kind == "talent":
        opts = [tuple(x) for x in raw.get("talents", [])]
        q, extra = F.q_talent(opts, ctx, "refresh" in crit), opts
    elif kind == "use_card":
        q = F.q_use_card(raw.get("card", ""), ctx)
        if not q:
            return None
        q.pop("_targets", None)
        extra = [raw.get("card", "")]
    elif kind == "lord_boost":
        opts = [tuple(x) for x in raw.get("boosts", [])]
        q, extra = F.q_lord_boost(opts, ctx), opts
    elif kind == "carry":
        q = F.q_carry(ctx)
        if not q:
            return None
        extra = list(q.pop("_names").values())
    elif kind == "swap":
        q = F.q_swap(ctx)
        if not q:
            return None
        q.pop("_names", None)
        extra = [q.pop("_incoming")]
    elif kind == "sell_filler":             # f9lw
        q = F.q_sell_filler(ctx)
        if not q:
            return None
        q.pop("_names", None)
        extra = []
    elif kind == "sell_hand":               # f9lx
        q = F.q_sell_hand(ctx)
        if not q:
            return None
        q.pop("_names", None)
        extra = []
    elif kind == "use_timing":              # f9lt / f9lw
        q = F.q_use_timing(raw.get("card", ""), ctx)
        if not q:
            return None
        extra = [raw.get("card", "")]
    elif kind == "route":
        q = F.q_route(ctx)
        if not q:
            return None
        q.pop("_keys", None)
        extra = []
    elif kind == "auction":
        lots = [tuple(x) for x in raw.get("lots", [])]
        q = F.q_auction(lots, ctx)
        if not q:
            return None
        extra = [n for _, n in lots]
    elif kind == "eco":                     # 2026-10-09（f9le）：经济节奏题，没有候选卡详情
        q = F.q_eco(ctx)
        if not q:
            return None
        extra = []
    else:
        return None
    if list(q["criteria"].keys()) != list(crit.keys()) and not ALLOW_NEW_KEYS:
        return None                     # 选项集合变了 → 标签对不上，丢弃并统计（--allow-new-keys：保留新题面，由调用方重标）
    if F.engine_text_on():
        q["criteria"] = F.engine_prefix(q["criteria"], p.get("_engine") or F.ENGINE_ANNOT.get(p.get("id")))
    out = dict(p)
    out["question"] = q
    out["options"], out["n_opt"] = list(q["criteria"]), len(q["criteria"])
    out["state"] = F.state_text(ctx, extra_cards=extra, kind=kind)
    out["feature_version"] = F.FEATURE_VERSION
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backfill", action="store_true", help="用 states.jsonl 补题目当时读不到的场上读数（f6）")
    ap.add_argument("--allow-new-keys", action="store_true", help="新版本加了选项（如 f9d 的 lock）时保留新题面（需重标）")
    args = ap.parse_args()
    global ALLOW_NEW_KEYS
    ALLOW_NEW_KEYS = args.allow_new_keys
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    src = [json.loads(l) for l in (ds / args.src / "points.jsonl").open(encoding="utf-8")]
    out, dropped = [], collections.Counter()
    for p in src:
        r = rerender(backfill(p) if args.backfill else p)
        if r is None:
            dropped[p["kind"]] += 1
            continue
        out.append(r)
    (ds / args.out).mkdir(parents=True, exist_ok=True)
    with (ds / args.out / "points.jsonl").open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for f in ("new_ids.json", "info.json"):
        if (ds / args.src / f).exists():
            (ds / args.out / f).write_text((ds / args.src / f).read_text(encoding="utf-8"), encoding="utf-8")
    info = {"src": args.src, "feature_version": F.FEATURE_VERSION, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "points": len(out), "dropped": dict(dropped),
            "with_board": sum(1 for r in out if r.get("board")),
            "by_split": dict(collections.Counter(r.get("split") for r in out))}
    (ds / args.out / "rerender.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
