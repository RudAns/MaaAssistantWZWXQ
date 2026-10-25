"""carry / swap 题的**局面增强**（2026-09-25）。

为什么：R8 在 carry 测试 24 题里 23 次答 keep —— 训练集只有 87 道真实 carry 题（老师 57% 说 keep），
学生只学到了先验；swap 32 道同理（多答第一个选项）。真实对局一回合只出一题，攒够要几十局。

怎么做：拿**训练集**里的真实 carry/swap 局面，随机改动决定性的变量，再交给老师重新标注（蒸馏允许老师标任何合理局面）：
  回合（及对应的棋手等级）、场上各人等级 / 觉醒进度、谁是现任主C、枫叶舞绑在谁身上、已选的枫叶舞强化、
  手里是否有搬等级牌、手里是否有更强的输出位（carry）/ 等着上场的是谁、是不是觉醒卡、几张（swap）。
测试集、验证集的题**不增强**（评测只用真实局面）。每个变体记 source="aug" 和来源题 id，题面用线上同一套 features 渲染。

  python tools/laya/augment_decisions.py --src r8f8 --out r8f8aug --carry 6 --swap 6 --seed 0
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools" / "laya")]
import build_dataset as B  # noqa: E402
import facts as X  # noqa: E402
import features as F  # noqa: E402
from agent.decision import meta as M  # noqa: E402

HEROES = [n for n in X.cards() if X.is_hero(n)]
XP_NEED = {1: 7, 2: 13, 3: 17, 4: 23, 5: 27}          # 棋手经验条 N/M 的 M（7/13/17/23/27 对应 1~5 级，6 级满级）


def _lv_for(tier: int, rnd: int, rng: random.Random) -> int:
    """按回合给一个合理的等级：前期个位数，后期主力几十级；偶尔很高（夺取 / 喂等级）。"""
    base = max(1, int(rng.expovariate(1 / max(1.0, rnd * (2.5 if tier <= 2 else 1.8)))))
    return min(base, 99)


def vary_board(ctx: dict, rng: random.Random, rnd: int) -> dict:
    field = {}
    for n, v in (ctx.get("field") or {}).items():
        v = dict(v)
        r = rng.random()
        if r < 0.35 and isinstance(v.get("lv"), int):
            v["lv"] = max(1, min(99, int(v["lv"] * rng.uniform(0.3, 2.2))))
        elif r < 0.85 or not isinstance(v.get("lv"), int):
            v["lv"] = _lv_for(X.tier(n) or 1, rnd, rng)
        a = rng.random()
        v["awk_left"] = 0 if a < 0.12 else (1 if a < 0.3 else (2 if a < 0.55 else 3))
        if rng.random() < 0.4:
            v["eq"] = rng.choice([0, 0, 1, 2, 3])
        field[n] = v
    return field


def base_ctx(p: dict) -> dict:
    ctx = copy.deepcopy(p.get("ctx") or {})
    board = p.get("board") or {}
    ctx["field"] = {n: dict(board.get(n) or {}) for n in (p.get("field") or [])}
    return ctx


def vary_common(ctx: dict, rng: random.Random) -> int:
    rnd = max(2, min(16, (ctx.get("round") or 6) + rng.randint(-3, 4)))
    ctx["round"] = rnd
    lvl = max(1, min(6, M.level_target(rnd) + rng.choice([-1, 0, 0, 0, 1])))
    ctx["auction_in"] = B.auction_in(rnd)
    ctx["auction_done"] = rnd in M.AUCTION_ROUNDS and rng.random() < 0.5
    ctx["field"] = vary_board(ctx, rng, rnd)
    heroes = [n for n in ctx["field"] if X.is_hero(n)]
    set_level(ctx, lvl, rng)
    ctx["gold"] = rng.randint(0, 12 + 2 * rnd)
    ctx["pop"] = len(heroes)
    ctx["pop_cap"] = max(M.pop_cap_at(lvl), len(heroes))
    ctx["units_seen"] = len(heroes)
    ctx["unit_levels"] = sorted((v.get("lv") for v in ctx["field"].values() if isinstance(v.get("lv"), int)), reverse=True)[:8]
    ctx["maple"] = rng.choice(heroes + [""]) if heroes and rng.random() < 0.85 else ""
    ctx["lord_boosts"] = sorted(b for b in X.MAPLE_BOOST_EN if rng.random() < 0.18)
    hand = [c for c in (ctx.get("hand") or []) if c not in X.LEVEL_MOVE_CARDS + X.LEVEL_MOVE_CARDS_F8]
    if rng.random() < 0.15:
        hand.append(rng.choice(X.LEVEL_MOVE_CARDS + X.LEVEL_MOVE_CARDS_F8))
    ctx["hand"] = hand
    return rnd


def set_level(ctx: dict, lvl: int, rng: random.Random) -> None:
    """等级和它决定的读数一起改（不能留原局面的：老师会看到"4 级、经验 4/13"这种矛盾）。"""
    ctx["level"] = lvl
    need = XP_NEED.get(lvl)
    ctx["xp_needed"] = need or 0
    ctx["xp"] = rng.randint(0, need - 1) if need else 0
    ctx["level_cost"] = (need - ctx["xp"]) if need else -1          # 实测：升级费 = 本级所需经验 - 当前经验


def pick_hero(rng: random.Random, level: int, exclude: set, output_only: bool = False) -> str | None:
    odds = M.TIER_ODDS.get(level) or {1: 1.0}
    for _ in range(50):
        q = rng.choices(list(odds), weights=list(odds.values()))[0]
        cands = [n for n in HEROES if X.tier(n) == q and n not in exclude and (not output_only or X.is_output(n))]
        if cands:
            return rng.choice(sorted(cands))
    return None


def aug_carry(p: dict, rng: random.Random) -> dict | None:
    ctx = base_ctx(p)
    vary_common(ctx, rng)
    outs = [n for n in ctx["field"] if X.is_hero(n) and X.is_output(n)]
    if len(outs) < 2:
        return None
    ctx["carry"] = rng.choice(outs)
    ctx["hand"] = [c for c in ctx["hand"] if c not in HEROES or c in ctx["field"]]
    if rng.random() < 0.2:                         # 手里有一个更强的输出位还没上场
        h = pick_hero(rng, max(ctx["level"], 4), set(ctx["field"]), output_only=True)
        if h:
            ctx["hand"].append(h)
    return B.carry_point(ctx, {"run": p["run"], "ts": p["ts"], "round": ctx["round"]})


def aug_swap(p: dict, rng: random.Random) -> dict | None:
    ctx = base_ctx(p)
    vary_common(ctx, rng)
    heroes = [n for n in ctx["field"] if X.is_hero(n)]
    # 人口"满"必须和等级表一致（老师 aug batch_001："4 级却是 4/4"）：挑一个人口上限正好等于场上人数的等级，凑不出就不出题
    if M.pop_cap_at(ctx["level"]) != len(heroes):
        lvls = [lv for lv in range(1, 7) if M.pop_cap_at(lv) == len(heroes)]
        if not lvls:
            return None
        set_level(ctx, min(lvls, key=lambda lv: (abs(lv - ctx["level"]), lv)), rng)
    ctx["pop"] = ctx["pop_cap"] = len(heroes)
    outs = [n for n in heroes if X.is_output(n)]
    ctx["carry"] = rng.choice(outs) if outs and rng.random() < 0.9 else ""
    x = pick_hero(rng, ctx["level"], set(ctx["field"]))
    if not x:
        return None
    k = rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
    ctx["hand"] = [c for c in ctx["hand"] if c not in HEROES] + [x] * k
    ctx["hand_awk"] = [x] if rng.random() < 0.15 else []
    return B.swap_point(ctx, {"run": p["run"], "ts": p["ts"], "round": ctx["round"]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--carry", type=int, default=6, help="每道训练集 carry 题生成几个变体")
    ap.add_argument("--swap", type=int, default=6, help="每道训练集 swap 题生成几个变体（carry 题的局面也拿来出 swap）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", default="carry,swap", help="只生成这些题型（补题用）")
    args = ap.parse_args()
    assert F.ACTIVE >= "f8", F.ACTIVE
    ds = ROOT / "data" / "laya" / "datasets"
    src = [json.loads(l) for l in (ds / args.src / "points.jsonl").open(encoding="utf-8")]
    rng = random.Random(args.seed)
    seen, new = set(), []
    for p in src:
        if p.get("split") != "train" or p["kind"] not in ("carry", "swap") or p.get("source") == "aug":   # 不做"变体的变体"
            continue
        kinds = set(args.kinds.split(","))
        jobs = (([aug_carry] * args.carry if p["kind"] == "carry" and "carry" in kinds else [])
                + ([aug_swap] * args.swap if "swap" in kinds else []))
        for j, fn in enumerate(jobs):
            q = fn(p, rng)
            if not q:
                continue
            sig = (q["kind"], q["state"], json.dumps(q["question"], ensure_ascii=False, sort_keys=True))
            if sig in seen:
                continue
            seen.add(sig)
            q["id"] = hashlib.md5(f"aug|{p['id']}|{fn.__name__}|{j}|{args.seed}".encode()).hexdigest()[:12]
            q.update(split="train", match=p.get("match") or p["run"], source="aug", aug_of=p["id"],
                     feature_version=F.ACTIVE)
            new.append(q)
    out = ds / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "points.jsonl").open("w", encoding="utf-8") as fh:
        for p in src + new:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "new_ids.json").write_text(json.dumps([p["id"] for p in new]), encoding="utf-8")
    info = {"src": args.src, "seed": args.seed, "n": len(src) + len(new), "aug": len(new),
            "by_kind": dict(collections.Counter(p["kind"] for p in new)),
            "carry_best_rule": dict(collections.Counter(p.get("rule_choice") for p in new if p["kind"] == "carry"))}
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
