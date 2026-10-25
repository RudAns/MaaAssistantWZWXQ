"""从历史对局日志抽取**决策点**（局面 + 候选项），供 Claude 标注、Laya 训练。

路线（2026-09-23 修正）：**不以规则引擎为老师**（规则并不完美，与它的一致率不能当验证依据）；
老师 = Claude（Opus 5.5）按标注规范给出的软分布 + 理由。
规则引擎当时的选择只作为参考字段 `rule_choice` 保留（用于对照分析，不参与训练/验证）。

输入：data/logs/live_*/states.jsonl（逐帧：state + 规则引擎给出的 actions + 是否 executed）
      data/logs/live_*/ledger.jsonl（账本：上场/卖出/换主C…，用来还原场上阵容和主C）
      data/logs/live_*/phase_settle_final_*.png（结算页，认名次）
输出：data/laya/datasets/<ver>/points.jsonl（全部决策点）+ split.json（按对局切分）+ stats.json

四类决策（都是 Laya 的 choice 题）：
  card_pick   卡牌 N 选 1（3~5 个候选）
  talent      天赋三选一（+ 可选"刷新"）
  lord_boost  棋手特性三选一
  shop        商店行动：买某张 / 刷新 / 升级 / 不买了

用法：.venv/Scripts/python.exe -X utf8 tools/laya/build_dataset.py --ver v1
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))

import features as F  # noqa: E402

AUCTION_ROUNDS = (4, 8, 11)   # 旧口径（= meta.AUCTION_ROUNDS_LEGACY），f5/f6 题面按它训练，别改；f7 在 facts 里按实测 (4,8,12) 现算


def auction_in(rnd: int):
    nxt = [r for r in AUCTION_ROUNDS if r >= rnd]
    return (nxt[0] - rnd) if nxt else None


def load_ledger(run_dir: Path) -> list[dict]:
    f = run_dir / "ledger.jsonl"
    out = []
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def board_from_ledger(ledger: list[dict], upto_round: int) -> tuple[dict, str]:
    """用账本近似还原"第 upto_round 回合开始时"的场上阵容和主C（账本事件没有时间戳，同回合内先后还原不了）。"""
    field: dict = {}
    carry = ""
    last_round = 0
    for e in ledger:
        r = e.get("round")
        if r is None:
            r = last_round
        else:
            last_round = r
        ev = e.get("event")
        if ev == "换主C" and r <= upto_round:
            carry = e.get("新") or carry
        if r >= upto_round:
            continue
        name = e.get("英雄") or e.get("牌")
        if ev == "上场" and name:
            field[name] = {}
        elif ev in ("卖出", "判定觉醒回手", "推断离场") and name:
            field.pop(name, None)
        elif ev == "落点修正" and e.get("英雄"):
            field.setdefault(e["英雄"], {})
    return field, carry


def _standing(s: dict, hp) -> tuple:
    """六人血条 → (名次, 存活人数)。自己那行按 state.hp 算（players 里的"我"那行偶有误读）。"""
    others = [p.get("hp") for p in (s.get("players") or []) if p.get("name") != "我"]
    others = [h for h in others if isinstance(h, int) and 0 < h <= 60]
    if not isinstance(hp, int) or not others:
        return None, None
    return 1 + sum(1 for h in others if h > hp), len(others) + 1


def ctx_from_state(s: dict, prev: dict) -> dict:
    ctx = dict(prev)
    hp = s.get("hp")
    hp = hp if isinstance(hp, int) and 0 < hp <= 60 else prev.get("hp")    # f3：>60 必是误读（上限 60）
    rank, alive = _standing(s, hp)
    ctx.update({
        "round": s.get("round_no") or prev.get("round"),
        "hp": hp,
        "rank": rank or prev.get("rank"), "alive": alive or prev.get("alive"),
        "xp": s.get("xp"), "xp_needed": s.get("xp_needed"),
        "units_seen": len(s.get("unit_anchors") or {}) if s.get("board_seen") or s.get("unit_anchors") else None,
        "gold": s.get("gold"),
        "level": s.get("level"),
        "level_cost": s.get("level_cost"),
        "pop": s.get("population") if s.get("population_valid") else prev.get("pop", "?"),
        "pop_cap": s.get("population_cap") if s.get("population_valid") else prev.get("pop_cap", "?"),
        "banned": s.get("banned_faction") or prev.get("banned", ""),
        "hand": [h["name"] for h in (s.get("hand_slots") or [])] or s.get("hand") or [],
        # f5：手里**已确认觉醒**的英雄（点开看到 AWAKENED / 觉醒回手）—— 最高优先级上场（2026-09-24）
        "hand_awk": [h["name"] for h in (s.get("hand_slots") or []) if h.get("awakened")],
        "unit_levels": sorted((s.get("unit_levels") or {}).values(), reverse=True)[:8],
        "free_refresh": s.get("free_refresh") or 0,
    })
    ctx["auction_in"] = auction_in(ctx["round"] or 0)
    return ctx


CTX_KEYS = ("round", "hp", "gold", "level", "level_cost", "pop", "pop_cap", "banned", "hand", "carry",
            "auction_in", "auction_done", "rank", "alive", "xp", "xp_needed", "free_refresh", "units_seen",
            "unit_levels", "level_fixed", "level_raw", "shop_max_tier", "hand_awk", "maple", "lord_boosts", "_shop_full", "_lock_ok",
            "carry_since", "carry_prev", "carry_switches", "route", "route_since", "talents", "lordxp_since", "worn",
            "laya_bought")     # f9za：Laya 主动买的新英雄 → 回合（只有 f9za 起的换人题渲染）


STICKY_KEYS = ("lv", "awk_left", "eq", "pool_left")


def sticky_board(prev: dict | None, cur: dict | None) -> dict:
    """**商店开着时棋盘被盖住**，那一帧读不到等级/觉醒/装备 —— 而商店决策恰恰都在商店开着时（f6 数据核查：
    实战帧约一半有等级读数，商店题几乎全空）。同一个人还站在同一格，就沿用上次读到的值；换格/换人就作废。
    线上（laya_policy）与离线（extract_run）共用这一个函数（train/serve 一致）。"""
    out = {}
    for n, v in (cur or {}).items():
        v = dict(v) if isinstance(v, dict) else {}
        pv = (prev or {}).get(n) if isinstance((prev or {}).get(n), dict) else None
        if pv and pv.get("cell") == v.get("cell"):
            for k in STICKY_KEYS:
                if v.get(k) is None and pv.get(k) is not None:
                    v[k] = pv[k]
        out[n] = v
    return out


def make_point(kind, qdef, rule_key, ctx, meta, extra_cards=None, raw=None):
    keys = list(qdef["criteria"].keys())
    if len(keys) < 2:
        return None                       # 只有一个选项不是决策
    return {"kind": kind, "question": qdef, "state": F.state_text(ctx, extra_cards=extra_cards, kind=kind),
            "options": keys, "n_opt": len(keys),
            "rule_choice": rule_key if rule_key in keys else None,
            "raw": raw or {}, "ctx": {k: ctx.get(k) for k in CTX_KEYS},
            "field": list((ctx.get("field") or {}).keys()),
            # 场上每人的事实（等级/觉醒还差几张/留场/后期角色）——看板显示用，也方便事后分析
            "board": {n: {k: v.get(k) for k in ("lv", "awk_left", "keep", "late", "pool_left", "cell", "eq")}
                      for n, v in (ctx.get("field") or {}).items() if isinstance(v, dict)},
            **meta}


# ---------------- 从"一帧状态 + 上下文"构造一道题 ----------------
# 离线建数据集（extract_run）和实战推理（agent/decision/laya_policy.py）**共用这几个函数**，
# 保证学生在实战里看到的输入和训练时逐字一致（train/serve parity）。

def card_pick_point(s: dict, c: dict, meta: dict, chosen=None):
    opts = [F.clean_name(o["name"]) for o in s.get("card_options") or []]
    if len(opts) < 2 or len(set(opts)) != len(opts):
        return None
    key = f"opt{opts.index(chosen)}" if chosen in opts else None
    return make_point("card_pick", F.q_card_pick(opts, c), key, c, meta, extra_cards=opts, raw={"cards": opts})


def talent_point(s: dict, c: dict, meta: dict, refreshed: bool = False, index=None, can_refresh=None):
    opts = [(F.clean_name(tt["name"]), tt.get("desc", "")) for tt in s.get("talents") or []]
    can_ref = (bool(s.get("talent_can_refresh")) or refreshed) if can_refresh is None else can_refresh
    key = "refresh" if refreshed else (f"opt{index}" if index is not None else None)
    return make_point("talent", F.q_talent(opts, c, can_ref), key, c, meta, extra_cards=opts, raw={"talents": opts})


def lord_boost_point(opts: list, c: dict, meta: dict, index=None):
    key = f"opt{index}" if index is not None else None
    return make_point("lord_boost", F.q_lord_boost(opts, c), key, c, meta, extra_cards=opts, raw={"boosts": opts})


def use_card_point(name: str, c: dict, meta: dict):
    """特殊效果牌用在谁（影子题）。选项 hold / use0.. ；raw 里记下每个 useN 对应的目标。"""
    q = F.q_use_card(name, c)
    if not q:
        return None
    targets = q.pop("_targets")
    return make_point("use_card", q, None, c, meta, extra_cards=[name], raw={"card": name, "targets": targets})


def use_timing_point(name: str, c: dict, meta: dict):
    """f9lt：现在用 / 先留着（影子题）。"""
    q = F.q_use_timing(name, c)
    if not q:
        return None
    return make_point("use_timing", q, "hold", c, meta, extra_cards=[name], raw={"card": name})


def swap_point(c: dict, meta: dict, chosen: str | None = None):
    """人口满时卖谁换人（swap 题）。选项 bench / sN；raw 里记每个 sN 的英雄和等着上场的英雄；rule_choice = 规则当时卖的人。"""
    q = F.q_swap(c)
    if not q:
        return None
    names, x = q.pop("_names"), q.pop("_incoming")
    rule_key = next((k for k, n in names.items() if chosen and n == chosen), None)
    return make_point("swap", q, rule_key, c, meta, extra_cards=[x], raw={"names": names, "incoming": x})


def sell_filler_point(c: dict, meta: dict):
    """f9lw：没人等位时卖不卖占位件（回本弈星 / 过阶段开局件 / 别家主C）。选项 keep / sN；raw 记每个 sN 的英雄。"""
    q = F.q_sell_filler(c)
    if not q:
        return None
    names = q.pop("_names")
    return make_point("sell_filler", q, "keep", c, meta, raw={"names": names})


def sell_hand_point(c: dict, meta: dict):
    """f9lx：手里的英雄牌卖不卖。选项 keep / sN；raw 记每个 sN 的英雄。"""
    q = F.q_sell_hand(c)
    if not q:
        return None
    names = q.pop("_names")
    return make_point("sell_hand", q, "keep", c, meta, raw={"names": names})


def auction_point(s: dict, c: dict, meta: dict, chosen: str | None = None):
    """拍卖：争哪一件 / 不拍。选项 lotN / pass；raw 记每个 lotN 对应的卡名。"""
    lots = [(lot["index"], F.clean_name(lot["piece"])) for lot in (s.get("auction") or {}).get("lots") or []
            if lot.get("piece") and not lot.get("sold")]
    q = F.q_auction(lots, c)
    if not q:
        return None
    names = {f"lot{i}": n for i, n in lots}
    key = next((k for k, n in names.items() if chosen and n == chosen), None)
    return make_point("auction", q, key, c, meta, extra_cards=[n for _, n in lots], raw={"lots": lots, "names": names})


AUCTION_QUESTIONS = True      # extract_run 是否出拍卖题（2026-09-26 起）
USE_CARD_QUESTIONS = True     # extract_run 是否按"规则用了哪张指定英雄的效果牌"出 use_card 题（f9d 起）
CARRY_QUESTIONS = True        # extract_run 是否每回合出一道 carry 题（2026-09-25 起；旧数据集重建时可关）
SELL_FILLER_QUESTIONS = False # f9lw：sell_filler 题 + 点兵点将 use_timing 题（建 f9lw 数据集时打开；旧版本数据集重建不受影响）
LAYA_TIMING_CARDS = ("点兵点将",)
SELL_HAND_QUESTIONS = False   # f9lx：sell_hand 题（建 f9lx 数据集时打开）  # f9lw ㊵：用牌表 when=laya_timing 的牌
ROUTE_QUESTIONS = True        # f9k：还没锁定路线时第 3~8 回合每回合出一道 route 题（2026-09-29 起）
ROUTE_Q_ROUNDS = (3, 8)


def route_point(c: dict, meta: dict):
    """选哪条官方路线（f9k）。选项 r1..r3 / flex；raw 里记每个 rN 对应的阵容 key；rule_choice = 规则锁定的那条（没锁 = flex）。"""
    q = F.q_route(c)
    if not q:
        return None
    keys = q.pop("_keys")
    locked = c.get("_route_locked_key") or ""
    rule_key = next((k for k, v in keys.items() if v == locked), None) if locked else "flex"
    return make_point("route", q, rule_key, c, meta, raw={"keys": keys})


def route_from_ledger(ledger: list[dict]) -> tuple[str, int]:
    """老日志（逐帧没记 route）：账本里第一条「锁定路线」→ (阵容 key, 回合)。重名阵容取阿离里最热门、且参选的那条。"""
    from agent.decision import routes as R
    for e in ledger:
        if e.get("event") == "锁定路线" and e.get("阵容"):
            lu = next((x for x in R.ali_lineups() if x.get("name") == e["阵容"] and not R.excluded(x)), None) \
                or next((x for x in R.ali_lineups() if x.get("name") == e["阵容"]), None)
            if lu:
                return str(lu.get("key")), int(e.get("round") or 0)
    return "", 0


def carry_point(c: dict, meta: dict):
    """谁当主C（影子题）。选项 keep / cN；raw 里记每个选项对应的英雄；rule_choice = 规则当前的主C。"""
    q = F.q_carry(c)
    if not q:
        return None
    names = q.pop("_names")
    rule_key = next((k for k, n in names.items() if n == (c.get("carry") or "")), None)
    return make_point("carry", q, rule_key, c, meta, extra_cards=list(names.values()), raw={"names": names})


def shop_point(s: dict, c: dict, meta: dict, act: dict | None = None):
    gold = s.get("gold") or 0
    shop = [(sl["index"], F.clean_name(sl["piece"]), sl.get("cost") or 3) for sl in s.get("shop") or []
            if sl.get("piece") and not sl.get("sold") and (sl.get("cost") or 3) <= gold]
    # f9d：整个商店（含买不起的）—— 锁店题要看"留下的是什么"
    full = [F.clean_name(sl["piece"]) for sl in s.get("shop") or [] if sl.get("piece") and not sl.get("sold")]
    c = dict(c, _shop_full=full)
    can_ref = gold >= 2 or (s.get("free_refresh") or 0) > 0
    lc = s.get("level_cost") or -1
    # f3：升级费没读出来也给"升级"选项（老师：等级明显落后却没有 levelup 可选）；按钮由执行层 OCR 定位
    can_lv = (not s.get("lord_maxed") and (c.get("level") or 0) < 6
              and ((lc > 0 and gold >= lc) or (lc <= 0 and gold >= 4)))
    # f3：等级读数与商店品阶矛盾时，按"这个品阶最早几级出现"修正（3 级商店出 4 阶牌 → 至少 4 级）
    try:
        from agent.decision.meta import min_level_for
        need = max([min_level_for(F.card_tier(n)) for _, n, _ in shop
                    if F.card_tier(n) and F.card_category(n) == "hero"] or [0])
    except Exception:                                   # noqa: BLE001
        need = 0
    if need and (c.get("level") or 0) < need:
        # 老师视图仍显示修正后的等级（与 v3 标注一致）；学生视图（f4）显示原始读数 + 商店最高品阶
        c = dict(c, level_raw=c.get("level"), level=need, level_fixed=True, shop_max_tier=need)
    key = None
    if act:
        t = act.get("type")
        key = {"buy_piece": f"buy{act.get('slot')}", "refresh_shop": "refresh", "buy_xp": "levelup",
               "toggle_shop": "stop", "lock_shop": "lock"}.get(t)
        if t == "buy_piece":        # 同名牌合并后，规则买的那一格映射到保留的那一格
            name = next((n for sl, n, _ in shop if sl == act.get("slot")), None)
            kept = next((sl for sl, n, _ in F.dedupe_shop(shop) if n == name), None)
            if kept is not None:
                key = f"buy{kept}"
    return make_point("shop", F.q_shop(shop, c, can_ref, can_lv), key, c, meta,
                      extra_cards=[n for _, n, _ in F.dedupe_shop(shop)], raw={"shop": shop, "shop_full": full})


class CarryHistory:
    """f9g：主C 的来历 —— 现任从第几回合开始、上一任是谁、这局换过几次（线上 LayaPolicy 与离线 extract_run 用同一套）。
    2026-09-26 第 9 局：一局换了 5 次主C（干将莫邪→米莱狄→干将莫邪→嬴政→海诺），题面里看不出"刚换过"。"""

    def __init__(self):
        self.cur, self.since, self.prev, self.n = "", 0, "", 0

    def update(self, carry: str, rnd: int) -> None:
        carry = carry or ""
        if rnd and self.since and rnd < self.since - 1:
            self.__init__()                     # 回合号倒退 = 同一个运行目录里开了新的一局（老师 r9i：第 5 回合写着 since r7）
        if carry and carry != self.cur:
            if self.cur:
                self.prev, self.n = self.cur, self.n + 1
            self.cur, self.since = carry, rnd or self.since

    def see_board(self, names, level) -> None:
        """f9lw ㊳：每个在场单位第一次在场时的棋手等级（弈星"在场期间棋手升过 1 级就回本"）；离场再上场重新计。"""
        if not isinstance(level, int) or level <= 0:
            return
        on = self.__dict__.setdefault("lord_at_deploy", {})
        names = set(names or ())
        for n in list(on):
            if n not in names:
                on.pop(n)
        for n in names:
            on.setdefault(n, level)

    def ctx(self) -> dict:
        return {"carry_since": self.since, "carry_prev": self.prev, "carry_switches": self.n,
                "lordxp_since": dict(self.__dict__.get("lord_at_deploy") or {})}


def extract_run(run_dir: Path) -> list[dict]:
    f = run_dir / "states.jsonl"
    if not f.exists():
        return []
    ledger = load_ledger(run_dir)
    out = []
    ctx: dict = {}
    auction_round = -1
    last_board = None
    carry_rounds: set = set()
    auction_asked: set = set()
    use_asked: set = set()
    ch = CarryHistory()     # f9g：主C 的来历
    # f9k：锁定的官方路线。新日志逐帧记 record["route"]；老日志按账本「锁定路线」的回合还原
    led_route, led_round = route_from_ledger(ledger)
    route_live: dict | None = None
    route_rounds: set = set()
    boosts: list = []       # 本局选过的棋手特性（枫叶舞强化）：新日志直接记，旧日志按执行过的 lord_boost_pick 推算
    maple = None            # 枫叶舞绑定对象：新日志直接记（record["maple"]），旧日志按"拖枫叶舞给谁 / 被绑的人卖掉"推算
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if "maple" in x:
            maple = x["maple"] or ""
        elif x.get("executed") and x.get("actions"):
            a0 = x["actions"][0]
            if a0.get("piece") == "枫叶舞" and a0.get("target"):
                maple = a0["target"]
            elif a0.get("type") == "sell_unit" and maple and a0.get("piece") == maple:
                maple = ""
        if "route" in x:
            route_live = x["route"] or {}
        if "lord_boosts" in x:
            boosts = list(x["lord_boosts"] or [])
        elif x.get("executed") and x.get("actions") and x["actions"][0].get("type") == "lord_boost_pick"                 and x["actions"][0].get("name"):
            boosts = sorted(set(boosts) | {x["actions"][0]["name"]})
        s = x.get("state") or {}
        ph = s.get("phase")
        if x.get("board") is not None:
            b = x["board"]
            if isinstance(b, str):                       # 旧日志把 dict 写成了 repr 字符串
                import ast
                try:
                    b = ast.literal_eval(b)
                except (ValueError, SyntaxError):
                    b = {}
            last_board = (sticky_board(last_board[0] if last_board else None, b), x.get("carry") or "")
            ch.update(x.get("carry") or "", (s.get("round_no") if isinstance(s, dict) else 0) or ctx.get("round") or 0)
            ch.see_board(list(b), (s.get("level") if isinstance(s, dict) else None) or ctx.get("level"))
        if ph == "prepare":
            ctx = ctx_from_state(s, ctx)
            _r = (s.get("round_no") or ctx.get("round") or 0)
            if route_live is not None:
                ctx["route"], ctx["route_since"] = route_live.get("route") or "", route_live.get("route_since") or 0
            elif led_route and _r >= led_round:
                ctx["route"], ctx["route_since"] = led_route, led_round
            else:
                ctx["route"], ctx["route_since"] = "", 0
        if ph == "auction":
            auction_round = s.get("round_no") or ctx.get("round") or 0
            # **拍卖题**：每场拍卖第一帧读全拍品的画面出一题（2026-09-26，拍卖交给 Laya）
            if AUCTION_QUESTIONS and auction_round not in auction_asked and last_board is not None:
                lots_ok = [lot for lot in (s.get("auction") or {}).get("lots") or [] if lot.get("piece")]
                if len(lots_ok) >= 3:
                    auction_asked.add(auction_round)
                    ca = dict(ctx, field=last_board[0], carry=last_board[1] or board_from_ledger(ledger, auction_round)[1],
                              round=auction_round, maple=(maple if maple is not None else None), lord_boosts=boosts,
                              **ch.ctx())
                    bid = next((a.get("piece") for xx in [x] for a in (xx.get("actions") or []) if a.get("type") == "auction_bid"), None)
                    pa = auction_point(s, ca, {"run": run_dir.name, "ts": x.get("ts"), "round": auction_round}, chosen=bid)
                    if pa:
                        out.append(pa)
        # **谁当主C（carry 题）**：每回合一题 —— 准备阶段、商店关着、人口读得到、有场上事实的第一帧（不依赖执行了什么动作）
        if (CARRY_QUESTIONS and ph == "prepare" and not s.get("shop_open") and s.get("population_valid")
                and last_board is not None):
            rnd_c = s.get("round_no") or ctx.get("round") or 0
            if rnd_c and rnd_c not in carry_rounds:
                carry_rounds.add(rnd_c)
                cc = dict(ctx, field=last_board[0], carry=last_board[1] or board_from_ledger(ledger, rnd_c)[1],
                          round=rnd_c, auction_done=(auction_round == rnd_c),
                          maple=(maple if maple is not None else ("" if "枫叶舞" in (ctx.get("hand") or []) else None)),
                          lord_boosts=boosts, **ch.ctx())
                pc = carry_point(cc, {"run": run_dir.name, "ts": x.get("ts"), "round": rnd_c})
                if pc:
                    out.append(pc)
                # **卖谁换人（swap 题）**：同一帧，人口满且手里有英雄等着上场才出（q_swap 自己判）
                ps = swap_point(cc, {"run": run_dir.name, "ts": (x.get("ts") or 0) + 0.001, "round": rnd_c})
                if ps:
                    out.append(ps)
                elif SELL_FILLER_QUESTIONS:
                    # f9lw：同一帧没人等位 → 问要不要主动卖占位件
                    pf = sell_filler_point(cc, {"run": run_dir.name, "ts": (x.get("ts") or 0) + 0.003, "round": rnd_c})
                    if pf:
                        out.append(pf)
                if SELL_HAND_QUESTIONS:
                    ph_ = sell_hand_point(cc, {"run": run_dir.name, "ts": (x.get("ts") or 0) + 0.0035, "round": rnd_c})
                    if ph_:
                        out.append(ph_)
                if SELL_FILLER_QUESTIONS:
                    # f9lw ㊵：时机交给 Laya 的牌（点兵点将）在手 → 每回合一道 use_timing 题
                    for k, tn in enumerate(n for n in dict.fromkeys(cc.get("hand") or []) if n in LAYA_TIMING_CARDS):
                        pt = use_timing_point(tn, cc, {"run": run_dir.name, "ts": (x.get("ts") or 0) + 0.004 + k * 1e-4, "round": rnd_c})
                        if pt:
                            out.append(pt)
                # **选路线（route 题，f9k）**：还没锁定时第 3~8 回合每回合一题；锁定那一回合也问（那正是做选择的时刻），
                # 题面里不带路线（问的是"现在锁哪条"）；rule_choice = 这一回合规则锁定的那条，没锁 = flex
                lock_rnd = (route_live or {}).get("route_since") if route_live else led_round
                lock_key = (route_live or {}).get("route") if route_live else led_route
                if (ROUTE_QUESTIONS and ROUTE_Q_ROUNDS[0] <= rnd_c <= ROUTE_Q_ROUNDS[1] and rnd_c not in route_rounds
                        and (not lock_rnd or rnd_c <= lock_rnd)):
                    route_rounds.add(rnd_c)
                    cr = dict(cc, route="", route_since=0,
                              _route_locked_key=(lock_key if lock_rnd and rnd_c == lock_rnd else ""))
                    pr = route_point(cr, {"run": run_dir.name, "ts": (x.get("ts") or 0) + 0.002, "round": rnd_c})
                    if pr:
                        out.append(pr)
        if not x.get("executed") or not x.get("actions"):
            continue
        a = x["actions"][0]
        t = a.get("type")
        rnd = s.get("round_no") or ctx.get("round") or 0
        field, carry = board_from_ledger(ledger, rnd)
        if last_board is not None:                # f5：实战时规则引擎写下的场上事实（觉醒进度/留场/后期角色），和在线推理同源
            field, carry = last_board[0], last_board[1] or carry
        c = dict(ctx, field=field, carry=carry, round=rnd, auction_done=(auction_round == rnd), **ch.ctx())
        meta = {"run": run_dir.name, "ts": x.get("ts"), "round": rnd}
        p = None
        if t == "card_pick" and ph == "card_pick" and s.get("card_options"):
            p = card_pick_point(s, c, meta, chosen=a.get("piece"))
        elif t in ("talent_pick", "talent_refresh") and ph == "talent" and s.get("talents"):
            p = talent_point(s, c, meta, refreshed=(t == "talent_refresh"), index=a.get("index", -1))
        elif t == "lord_boost_pick" and ph == "lord_boost" and s.get("talents"):
            opts = [(F.clean_name(tt["name"]), tt.get("desc", "")) for tt in s["talents"]]
            p = lord_boost_point(opts, c, meta, index=a.get("index", -1))
        elif (t == "drag_card" and ph == "prepare" and a.get("target") and a.get("kind") in ("效果牌", "天赋牌")
              and USE_CARD_QUESTIONS and F._facts().use_card_question(F.clean_name(a.get("piece") or ""))):
            # f9d：规则当时把指定英雄的效果牌给了谁 → 一道"给谁"题（rule_choice = 规则选的人）
            if (rnd, a.get("piece")) not in use_asked:
                use_asked.add((rnd, a.get("piece")))
                p = use_card_point(F.clean_name(a["piece"]), c, meta)
                if p:
                    tg = p["raw"]["targets"]
                    p["rule_choice"] = f"use{tg.index(a['target'])}" if a["target"] in tg else None
        elif ph == "prepare" and s.get("shop_open") and t in ("buy_piece", "refresh_shop", "buy_xp", "toggle_shop"):
            if t == "toggle_shop" and a.get("intent") != "close":
                continue
            p = shop_point(s, c, meta, act=a)
        if p:
            out.append(p)
    return out



# ---------------- eco 题（经济节奏：升级 / 停级刷牌 / 攒钱），2026-10-09，题面版本 f9le ----------------
# extract_run **不出** eco 题（已有数据集重建不变），
# 历史题由 extract_eco 单独抽（tools/laya/build_r9t.py 调用）。线上影子题将来也走 eco_point（train/serve 一致）。

def eco_point(c: dict, meta: dict, chosen: str | None = None):
    """经济节奏题。选项 levelup / roll / save；rule_choice = 这一回合实际做了什么（升过级 = levelup，没升级但刷过 = roll，否则 save）。
    raw 记下缺的配套（老师核对 / 事后统计用；题面里已经写了）。"""
    q = F.q_eco(c)
    if not q:
        return None
    X = F._facts()
    B = X.Board(c)
    miss = X.eco_missing(B)
    raw = {"missing": [[n, q_, src] for n, q_, src in miss], "carry": X.eco_carry(B), "level_cost_used": X.eco_level_cost(c)}
    return make_point("eco", q, chosen, c, meta, raw=raw)


def extract_eco(run_dir: Path) -> list[dict]:
    """一个运行目录 → 每回合一道 eco 题（准备阶段**第一帧商店开着**的画面：能量、升级费、免费刷新都读得到，且这一回合还没花钱升级/刷新）。
    同时记下这一回合实际的动作（rule_choice）和事后信息 hindsight（只给老师核对，不进题面）：
      lv_next1 / lv_next2：后 1、2 回合准备阶段的棋手等级；missing_got_by_r2：题面列的缺件里，到第 R+2 回合已拥有的。"""
    f = run_dir / "states.jsonl"
    if not f.exists():
        return []
    ledger = load_ledger(run_dir)
    ctx: dict = {}
    last_board = None
    ch = CarryHistory()
    led_route, led_round = route_from_ledger(ledger)
    route_live: dict | None = None
    boosts: list = []
    maple = None
    asked: dict = {}                     # 回合 -> 题
    acts = collections.defaultdict(collections.Counter)      # 回合 -> 执行过的动作计数
    lv_at: dict = {}                     # 回合 -> 准备阶段第一次读到的等级
    owned_at: dict = {}                  # 回合 -> 准备阶段最后一帧的 场上+手里 英雄
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if "maple" in x:
            maple = x["maple"] or ""
        elif x.get("executed") and x.get("actions"):
            a0 = x["actions"][0]
            if a0.get("piece") == "枫叶舞" and a0.get("target"):
                maple = a0["target"]
            elif a0.get("type") == "sell_unit" and maple and a0.get("piece") == maple:
                maple = ""
        if "route" in x:
            route_live = x["route"] or {}
        if "lord_boosts" in x:
            boosts = list(x["lord_boosts"] or [])
        elif x.get("executed") and x.get("actions") and x["actions"][0].get("type") == "lord_boost_pick" \
                and x["actions"][0].get("name"):
            boosts = sorted(set(boosts) | {x["actions"][0]["name"]})
        s = x.get("state") or {}
        ph = s.get("phase")
        if x.get("board") is not None:
            b = x["board"]
            if isinstance(b, str):
                import ast
                try:
                    b = ast.literal_eval(b)
                except (ValueError, SyntaxError):
                    b = {}
            last_board = (sticky_board(last_board[0] if last_board else None, b), x.get("carry") or "")
            ch.update(x.get("carry") or "", (s.get("round_no") if isinstance(s, dict) else 0) or ctx.get("round") or 0)
            ch.see_board(list(b), (s.get("level") if isinstance(s, dict) else None) or ctx.get("level"))
        if ph != "prepare":
            continue
        ctx = ctx_from_state(s, ctx)
        rnd = s.get("round_no") or ctx.get("round") or 0
        if not rnd or rnd > 30:            # 回合号误读（实测有 111 / 114）
            continue
        if route_live is not None:
            ctx["route"], ctx["route_since"] = route_live.get("route") or "", route_live.get("route_since") or 0
        elif led_route and rnd >= led_round:
            ctx["route"], ctx["route_since"] = led_route, led_round
        else:
            ctx["route"], ctx["route_since"] = "", 0
        lv = s.get("level")
        if isinstance(lv, int) and 1 <= lv <= 6 and rnd not in lv_at:
            lv_at[rnd] = lv
        if last_board is not None:
            owned_at[rnd] = sorted({*last_board[0].keys(), *[h for h in (ctx.get("hand") or []) if F._facts().is_hero(h)]})
        if (rnd not in asked and s.get("shop_open") and isinstance(s.get("gold"), int) and last_board is not None
                and not acts[rnd]["levelup"] and not acts[rnd]["refresh"]):
            c = dict(ctx, field=last_board[0], carry=last_board[1] or board_from_ledger(ledger, rnd)[1], round=rnd,
                     maple=(maple if maple is not None else ("" if "枫叶舞" in (ctx.get("hand") or []) else None)),
                     lord_boosts=boosts, **ch.ctx())
            p = eco_point(c, {"run": run_dir.name, "ts": (x.get("ts") or 0) + 0.003, "round": rnd})
            if p:
                asked[rnd] = p
        if x.get("executed") and x.get("actions"):
            t = x["actions"][0].get("type")
            if t == "buy_xp":
                acts[rnd]["levelup"] += 1
            elif t == "refresh_shop":
                acts[rnd]["refresh"] += 1
            elif t == "buy_piece":
                acts[rnd]["buy"] += 1
    out = []
    for rnd, p in sorted(asked.items()):
        a = acts.get(rnd) or collections.Counter()
        choice = "levelup" if a["levelup"] else "roll" if a["refresh"] else "save"
        p["rule_choice"] = choice if choice in p["options"] else None
        miss = [m[0] for m in (p.get("raw") or {}).get("missing") or []]
        later = set(owned_at.get(rnd + 2) or owned_at.get(rnd + 1) or [])
        p["hindsight"] = {"round_actions": dict(a), "lv_now": lv_at.get(rnd), "lv_next1": lv_at.get(rnd + 1),
                          "lv_next2": lv_at.get(rnd + 2), "missing_n": len(miss),
                          "missing_got_by_r2": [n for n in miss if n in later],
                          "owned_r2_known": (rnd + 2) in owned_at}
        out.append(p)
    return out


def group_matches(points: list[dict]) -> list[dict]:
    """按时间排序，回合号往回跳（下降超过 1）或间隔超过 30 分钟 = 新的一局。"""
    points.sort(key=lambda s: (s["ts"] or 0))
    mid, last_r, last_ts = 0, 0, 0
    for s in points:
        r = s["round"] or 0
        if (r and r < last_r - 1) or (s["ts"] and last_ts and s["ts"] - last_ts > 1800):
            mid += 1
        s["match"] = mid
        if r:
            last_r = r
        last_ts = s["ts"] or last_ts
    return points


def ranks_for_matches(points: list[dict]) -> dict:
    """每局的名次：这一局用到的目录里、最后一个决策点之后 15 分钟内的结算页截图，认绿色名字那一行。"""
    try:
        import cv2
        from agent.perception.extractors import _imread
        sys.path.insert(0, str(PROJECT_DIR / "tools"))
        from live_match_agent import read_my_rank
    except Exception as e:                                    # noqa: BLE001
        print("认名次的依赖加载失败，名次留空：", e)
        return {}
    by_match = collections.defaultdict(lambda: {"dirs": set(), "t1": 0})
    for s in points:
        m = by_match[s["match"]]
        m["dirs"].add(s["run"])
        m["t1"] = max(m["t1"], s["ts"] or 0)
    out = {}
    for mid, m in by_match.items():
        cands = []
        for d in m["dirs"]:
            for f in glob.glob(str(PROJECT_DIR / "data/logs" / d / "phase_settle_final_*.png")):
                t = int(re.search(r"_(\d+)\.png$", f).group(1))
                if m["t1"] <= t <= m["t1"] + 900:
                    cands.append((t, f))
        for t, f in sorted(cands):
            r = read_my_rank(cv2.cvtColor(_imread(Path(f)), cv2.COLOR_BGR2RGB))
            if r:
                out[mid] = r
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()
    points = []
    runs = sorted(Path(p) for p in glob.glob(str(PROJECT_DIR / "data/logs/live_*")) if os.path.isdir(p))
    for rd in runs:
        points += extract_run(rd)
    n_raw = len(points)
    points = group_matches(points)
    # 去重：同一局同一回合同一道题（选项一样）只留一条 —— 规则引擎每帧都会重算、执行层会重发
    seen, uniq = set(), []
    for s in points:
        key = (s["match"], s["round"], s["kind"], json.dumps(s["question"]["criteria"], ensure_ascii=False,
                                                               sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    points = uniq
    ranks = ranks_for_matches(points)
    for s in points:
        s["rank"] = ranks.get(s["match"])
        s["id"] = hashlib.md5(f"{s['run']}|{s['ts']}|{s['kind']}".encode()).hexdigest()[:12]
        s["feature_version"] = F.FEATURE_VERSION
    matches = sorted({s["match"] for s in points})
    rng = random.Random(args.seed)
    rng.shuffle(matches)
    n = len(matches)
    split = {m: ("test" if i < max(1, n // 10) else "val" if i < max(2, n // 5) else "train")
             for i, m in enumerate(matches)}
    for s in points:
        s["split"] = split[s["match"]]
    out_dir = PROJECT_DIR / "data" / "laya" / "datasets" / args.ver
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "points.jsonl").open("w", encoding="utf-8") as fh:
        for r in points:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "version": args.ver, "feature_version": F.FEATURE_VERSION, "seed": args.seed,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": len(runs), "raw_points": n_raw,
        "points": len(points), "matches": n, "matches_with_rank": len(ranks),
        "rank_hist": dict(collections.Counter(ranks.values())),
        "by_split_kind": {sp: dict(collections.Counter(r["kind"] for r in points if r["split"] == sp))
                          for sp in ("train", "val", "test")},
        "n_opt_hist": dict(collections.Counter(s["n_opt"] for s in points)),
        "random_baseline_acc": round(sum(1 / s["n_opt"] for s in points) / max(1, len(points)), 4),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
