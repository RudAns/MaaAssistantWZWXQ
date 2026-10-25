"""把 Laya 老师标注题转成 cgnet 的"对象集合 + 关系"格式（不解析题面文字，只用结构化字段）。

输入：
  data/laya/datasets/<ds>/points.jsonl   题目（含 raw / ctx / field / board / run / ts / split）
  data/laya/labels/r6/labels_all_<ds>.jsonl  老师软标签（dist / best）
  data/logs/<run>/states.jsonl           逐帧状态：补"已得天赋"（执行过的 talent_pick），题目缺 board 时补场上事实
  data/card_graph.json                   节点 / 边（关系偏置）
  data/cgnet/text_emb_<enc>.npy          节点文本向量（textenc.py 产出）

输出 data/cgnet/<ds>/：
  samples.pkl        list[dict]：每题的对象表 + 关系三元组 + 目标分布
  text_keys.json     额外文本（图谱外的天赋 / 强化等，用 OCR 描述编码）+ 对应向量 extra_emb_<enc>.npy
  node_feat.npy      节点静态特征（品阶 / 定位 / 功能标签 …）
  meta.json          词表、通道定义、各题型样本数、名称命中率

划分：完全沿用 points.jsonl 的 split 字段（Laya 训练时同一份，按对局切分），不重新切。

用法：python tools/cgnet/build_data.py --ds r9m
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
from textenc import Encoder, node_text, load_graph  # noqa: E402

OUT_ROOT = PROJECT_DIR / "data" / "cgnet"

# ---------------------------------------------------------------- 词表

TOK_TYPES = ["global", "board", "hand", "shop", "talent_owned", "boost_owned", "subject", "option", "technique",
             "wanted"]      # "wanted"（2026-10-09，只在 eco 题出现）：缺的配套英雄；追加在末尾，旧题型的下标不变
OPT_TYPES = ["none", "buy", "refresh", "levelup", "stop", "lock", "pick", "talent_refresh", "use", "hold",
             "keep", "switch", "sell", "bench", "lot", "pass", "route", "flex"]
KINDS = ["shop", "talent", "card_pick", "lord_boost", "carry", "swap", "auction", "use_card", "route"]
# KINDS 之外的新题型：全局 one-hot 放在 v1 全局向量的空位上（31..39 已被 KINDS 占满、40 起是规则表），旧题型的全局向量逐位不变
EXTRA_KIND_GV = {"eco": 62}
FACTIONS = ["河洛", "逐鹿", "三分", "大河", "日落海"]
EDGE_TYPES = None          # 从图谱读（21 种）
# 额外通道（放在 2*len(EDGE_TYPES) 之后）
EXTRA_CH = ["same_card", "option_ref", "route_member", "subject_target", "lineup_cooc", "lineup_carry_with", "lineup_carry_of"]
# 后三个是连续值通道（阿离官方阵容里的共现比例 / 以 a 为主C 的阵容里有 b 的比例），不是 0/1

NUM_DIM = 28
GLOB_DIM = 64


def opt_type(kind: str, key: str) -> str:
    k = re.sub(r"\d+", "", key)
    return {"buy": "buy", "refresh": "talent_refresh" if kind == "talent" else "refresh", "levelup": "levelup",
            "stop": "stop", "lock": "lock", "opt": "pick", "use": "use", "hold": "hold", "keep": "keep",
            "c": "switch", "s": "sell", "bench": "bench", "lot": "lot", "pass": "pass", "r": "route",
            "flex": "flex", "roll": "refresh", "save": "stop"}[k]     # eco：roll = 停级刷新、save = 不花


# ---------------------------------------------------------------- 图谱

class Graph:
    def __init__(self):
        g = load_graph()
        global EDGE_TYPES
        EDGE_TYPES = list(g["edge_types"].keys())
        self.nodes = g["nodes"]
        self.ids = [n["id"] for n in self.nodes]
        self.idx = {n: i for i, n in enumerate(self.ids)}
        self.node = {n["id"]: n for n in self.nodes}
        self.alias = {k: v for k, v in (g.get("aliases_skipped") or {}).items()}
        heroes = [n for n in self.nodes if n["type"] == "hero"]
        # scope 展开：派系 / 关键词 / 品阶 / 全体
        scope_members: dict[str, set] = {}
        for n in self.nodes:
            if n["type"] != "scope":
                continue
            sid = n["id"]
            mem = set()
            if sid.startswith("@f:"):
                f = sid[3:]
                mem = {h["id"] for h in heroes if (h.get("faction") or "") == f}
            elif sid.startswith("@kw:"):
                kw = sid[4:]
                mem = {h["id"] for h in heroes if kw in (h.get("keywords") or []) or kw in (h.get("text") or "")}
            elif sid.startswith("@tier:"):
                t = sid[6:]
                lo, hi = (t.split("-") + [t])[:2] if "-" in t else (t, t)
                mem = {h["id"] for h in heroes if int(lo) <= int(h.get("quality") or 0) <= int(hi)}
            elif sid in ("*", "@hero_any", "@on_board"):
                mem = {h["id"] for h in heroes}
            elif sid == "@g:近战":
                mem = {h["id"] for h in heroes if h.get("melee")}
            if mem:
                scope_members[sid] = mem
        E = len(EDGE_TYPES)
        pair: dict[tuple, int] = collections.defaultdict(int)
        for e in g["edges"]:
            t = EDGE_TYPES.index(e["type"])
            a, b = e["from"], e["to"]
            srcs = scope_members.get(a, {a})
            dsts = scope_members.get(b, {b})
            for x in srcs:
                for y in dsts:
                    if x == y:
                        continue
                    pair[(x, y)] |= 1 << t              # 正向
                    pair[(y, x)] |= 1 << (E + t)        # 反向
        self.pair = dict(pair)
        # 阿离官方阵容（data/lineups.json，按 log(1+使用人数) 加权）：英雄共现 / 主C-队友
        from agent.decision import routes as R
        wsum = collections.Counter()
        csum = collections.Counter()
        co = collections.Counter()
        cw = collections.Counter()
        tot = 0.0
        for lu in R.ali_lineups():
            team = [h for h in R.final_team(lu) if h in self.node]
            w = float(np.log1p(lu.get("use_num") or 0))
            if w <= 0 or not team:
                continue
            tot += w
            carry_ = lu.get("carry") or team[0]
            csum[carry_] += w
            for a_ in team:
                wsum[a_] += w
                for b_ in team:
                    if a_ != b_:
                        co[(a_, b_)] += w
                        if a_ == carry_:
                            cw[(a_, b_)] += w
        self.cooc = {k: v / wsum[k[0]] for k, v in co.items() if v / wsum[k[0]] >= 0.05}
        self.carry_with = {k: v / csum[k[0]] for k, v in cw.items() if csum[k[0]] and v / csum[k[0]] >= 0.05}
        self.lu_freq = {k: v / tot for k, v in wsum.items()}
        self.lu_carry = {k: v / tot for k, v in csum.items()}
        # 手法节点：与哪些卡相连（enables / fits_role 任一方向）
        self.tech_of: dict[str, set] = collections.defaultdict(set)
        for e in g["edges"]:
            for a, b in ((e["from"], e["to"]), (e["to"], e["from"])):
                if b.startswith("tech:"):
                    for x in scope_members.get(a, {a}):
                        self.tech_of[x].add(b)

    def resolve(self, name: str) -> str | None:
        if not name:
            return None
        n = re.sub(r"\s+", "", str(name))
        if n in self.node:
            return n
        n2 = re.sub(r"[（(].*?[)）]$", "", n)
        for c in (self.alias.get(n), n2, self.alias.get(n2)):
            if c and c in self.node:
                return c
        return None

    def node_feat(self) -> np.ndarray:
        tags = sorted({t for n in self.nodes for t in (n.get("func_tags") or [])})
        roles = sorted({t for n in self.nodes for t in (n.get("combat_role") or [])})
        types = ["hero", "effect", "equip", "talent", "lord_skill", "summon", "scope", "technique"]
        D = len(types) + 6 + len(roles) + len(tags) + len(FACTIONS) + 2
        out = np.zeros((len(self.nodes) + 2, D), np.float32)          # 0 = pad，1 = 图谱外
        for i, n in enumerate(self.nodes):
            v = out[i + 2]
            v[types.index(n["type"])] = 1
            o = len(types)
            try:
                q = float(n.get("quality") or 0)
            except ValueError:
                q = 0.0
            v[o] = q / 5
            v[o + 1] = float(n.get("carry_value") or 0) / 10
            v[o + 2] = float(n.get("functional_value") or 0) / 10
            v[o + 3] = 1.0 if n.get("melee") else 0.0
            v[o + 4] = 1.0 if n.get("transient") else 0.0
            v[o + 5] = 1.0 if n.get("target_rule") in ("allow_list", "deny_list") else 0.0
            o += 6
            for r in n.get("combat_role") or []:
                v[o + roles.index(r)] = 1
            o += len(roles)
            for t in n.get("func_tags") or []:
                v[o + tags.index(t)] = 1
            o += len(tags)
            f = n.get("faction") or ""
            for j, ff in enumerate(FACTIONS):
                if ff in f:
                    v[o + j] = 1
            o += len(FACTIONS)
            v[o] = min(1.0, 3 * self.lu_freq.get(n["id"], 0.0))       # 在阿离官方阵容里出现的（加权）比例
            v[o + 1] = min(1.0, 5 * self.lu_carry.get(n["id"], 0.0))  # 当主C 的比例
        return out


# ---------------------------------------------------------------- 日志回放：已得天赋 / 场上事实

def replay_run(run: str) -> list[tuple[float, list, dict | None]]:
    """按时间顺序返回 (ts, 已得天赋名列表, 最近一次场上事实 board) 快照。"""
    f = PROJECT_DIR / "data" / "logs" / run / "states.jsonl"
    snaps = []
    if not f.exists():
        return snaps
    owned: list = []
    board = None
    last_round = 0
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        s = x.get("state") or {}
        if x.get("board") is not None:
            b = x["board"]
            if isinstance(b, str):
                try:
                    b = ast.literal_eval(b)
                except (ValueError, SyntaxError):
                    b = {}
            if isinstance(b, dict):
                # 与 build_dataset.sticky_board 同义：同一格同一人沿用上次读到的等级/觉醒/装备
                if board:
                    for k, v in b.items():
                        old = board.get(k)
                        if old and old.get("cell") == v.get("cell"):
                            for kk in ("lv", "awk_left", "eq", "pool_left"):
                                if v.get(kk) is None and old.get(kk) is not None:
                                    v[kk] = old[kk]
                board = b
        snaps.append((float(x.get("ts") or 0), list(owned), board))
        if x.get("executed") and x.get("actions"):
            a = x["actions"][0]
            if a.get("type") == "talent_pick" and s.get("phase") == "talent":
                nm = a.get("name")
                if not nm:
                    tl = s.get("talents") or []
                    i = a.get("index", -1)
                    nm = tl[i]["name"] if 0 <= i < len(tl) else None
                nm = re.sub(r"\s+", "", nm or "")
                if nm and nm not in owned:          # 同一个弹窗重复点选（执行失败重试）只记一次
                    owned.append(nm)
        rn = s.get("round_no") or 0
        if s.get("phase") in ("lobby", "hero_pick") or (rn and rn < last_round - 1):
            owned, board = [], None             # 新的一局
        if rn:
            last_round = rn
    return snaps


def snapshot(snaps, ts: float):
    if not snaps:
        return [], None
    lo, hi = 0, len(snaps)
    while lo < hi:
        m = (lo + hi) // 2
        if snaps[m][0] <= ts + 1e-6:
            lo = m + 1
        else:
            hi = m
    if lo == 0:
        return [], None
    return snaps[lo - 1][1], snaps[lo - 1][2]


# ---------------------------------------------------------------- 单题转换

def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def cell_row(cell) -> float | None:
    try:
        y = float(str(cell).split(",")[1])
    except (IndexError, ValueError):
        return None
    return 0.0 if y < 400 else 1 / 3 if y < 465 else 2 / 3 if y < 535 else 1.0


class Builder:
    def __init__(self, G: Graph, enc: Encoder | None):
        self.G = G
        self.extra_texts: list[str] = []       # 图谱外卡牌的文字（名 + OCR 描述）
        self.extra_idx: dict[str, int] = {}
        self.name_hits = collections.Counter()
        self.name_miss = collections.Counter()

    def text_id(self, name: str, desc: str = "") -> int:
        """返回 (文本表下标)：图谱节点 -> 节点下标 + 1；图谱外 -> 额外文本下标 + 1 + len(nodes)。0 = 无。"""
        nid = self.G.resolve(name)
        if nid:
            return self.G.idx[nid] + 1
        t = f"{name}\n{desc}".strip()
        if t not in self.extra_idx:
            self.extra_idx[t] = len(self.extra_texts)
            self.extra_texts.append(t)
        return len(self.G.nodes) + 1 + self.extra_idx[t]

    def convert(self, p: dict, lab: dict, owned_talents: list, log_board: dict | None) -> dict | None:
        G = self.G
        kind = p["kind"]
        ctx = p.get("ctx") or {}
        raw = p.get("raw") or {}
        board = p.get("board")
        if isinstance(board, str):
            try:
                board = ast.literal_eval(board)
            except (ValueError, SyntaxError):
                board = None
        field = p.get("field") or []
        if isinstance(field, dict):
            field = list(field.keys())
        board_src = "point"
        if not board and log_board and p.get("source") != "aug":
            # 题目没存场上事实：日志里同一时刻的场上事实（只取仍在 field 里的人）
            board = {k: v for k, v in log_board.items() if not field or k in field}
            board_src = "log"
        if not board:
            board = {n: {} for n in field}
            board_src = "names"
        carry = ctx.get("carry") or ""
        maple = ctx.get("maple")
        route_key = ctx.get("route") or ""
        route_mem: set = set()
        if route_key:
            from agent.decision import routes as R
            lu = R.by_key(route_key)
            route_mem = R.members(lu) if lu else set()
        banned = str(ctx.get("banned") or "")
        hand = [re.sub(r"\s+", "", h) for h in (ctx.get("hand") or [])]
        hand_awk = set(ctx.get("hand_awk") or [])
        names_all = list(board.keys()) + hand
        copies = collections.Counter(names_all)
        gold = _num(ctx.get("gold"))
        lvl = _num(ctx.get("level"))
        lvl = int(lvl) if lvl and 1 <= lvl <= 6 else None
        from agent.decision import meta as M

        objs: list[dict] = []

        def add(tt: str, name: str, desc: str = "", num: dict | None = None, ot: str = "none", ref: int = -1) -> int:
            nid = G.resolve(name) if name else None
            if name:
                (self.name_hits if nid else self.name_miss)[name] += 1
            v = np.zeros(NUM_DIM, np.float32)
            nd = G.node.get(nid) or {}
            fac = str(nd.get("faction") or "")
            v[19] = 1.0 if fac and any(f in fac and f in banned for f in FACTIONS) else 0.0
            v[20] = 1.0 if name and name in route_mem else 0.0
            v[21] = min(copies.get(name, 0), 6) / 3 if name else 0.0
            v[10] = 1.0 if name and name == carry else 0.0
            v[11] = 1.0 if name and maple and name == maple else 0.0
            v[24] = 1.0 if name and name == (ctx.get("carry_prev") or "") else 0.0
            try:
                q = int(nd.get("quality") or 0)
            except ValueError:
                q = 0
            if nd.get("type") == "hero" and q and lvl:
                v[14] = M.tier_odds(q, lvl)
                v[15] = M.tier_odds(q, min(lvl + 1, 6))
                v[16] = 1.0 if M.can_appear(q, lvl) else 0.0
            for k, val in (num or {}).items():
                v[k] = val
            objs.append({"tt": TOK_TYPES.index(tt), "ot": OPT_TYPES.index(ot),
                         "node": (G.idx[nid] + 2) if nid else (1 if name else 0),
                         "text": self.text_id(name, desc) if name else 0, "num": v, "name": name or "",
                         "nid": nid, "ref": ref})
            return len(objs) - 1

        def unit_num(info: dict) -> dict:
            d = {}
            lv = _num(info.get("lv"))
            if lv is not None:
                d[0], d[1] = float(np.log1p(max(lv, 0)) / 5), 1.0
            aw = _num(info.get("awk_left"))
            if aw is not None:
                d[2], d[3] = (3 - aw) / 3, 1.0
            if info.get("keep") is not None:
                d[4] = float(bool(info.get("keep")))
            if info.get("late") is not None:
                d[5] = float(bool(info.get("late")))
            eq = _num(info.get("eq"))
            if eq is not None:
                d[6], d[7] = eq / 3, 1.0
            pl = _num(info.get("pool_left"))
            if pl is not None:
                d[8], d[9] = min(pl, 20) / 10, 1.0
            r = cell_row(info.get("cell"))
            if r is not None:
                d[22], d[23] = r, 1.0
            return d

        board_idx: dict[str, int] = {}
        for nm, info in board.items():
            board_idx.setdefault(nm, add("board", nm, num=unit_num(info or {})))
        hand_idx: dict[str, int] = {}
        for nm in hand:
            i = add("hand", nm, num={12: float(nm in hand_awk)})
            hand_idx.setdefault(nm, i)
        shop_idx: dict[int, int] = {}
        for ent in raw.get("shop") or []:
            slot, nm, cost = ent[0], ent[1], ent[2] if len(ent) > 2 else None
            d = {}
            if _num(cost) is not None:
                d[13] = cost / 5
                if gold is not None:
                    d[26] = float(gold >= cost)
            shop_idx[slot] = add("shop", nm, num=d)
        for nm in owned_talents:
            add("talent_owned", nm)
        for nm in ctx.get("lord_boosts") or []:
            add("boost_owned", nm)
        subj = -1
        if kind == "use_card" and raw.get("card"):
            subj = add("subject", raw["card"])
        if kind == "swap" and raw.get("incoming"):
            subj = add("subject", raw["incoming"], num={12: float(raw["incoming"] in hand_awk)})

        # eco（2026-10-09）：缺的配套英雄作为对象（tt="wanted"）；对象数值 17/18/25 = 在 L / L+1 / L+2 级刷一次见到它的概率
        # （v1 对象数值里没用到的三个维；只有 eco 题会写，旧题型逐位不变）
        eco_hits = None
        if kind == "eco":
            eco_hits = {0: 0.0, 1: 0.0, 2: 0.0}
            for ent in raw.get("missing") or []:
                nm, q = ent[0], int(ent[1] or 0) if len(ent) > 1 else 0
                d = {}
                for k_, dl in ((17, 0), (18, 1), (25, 2)):
                    L_ = (lvl + dl) if lvl else None
                    hp_ = M.shop_hit_prob(q, L_) if (L_ and L_ <= 6 and q and M.can_appear(q, L_)) else 0.0
                    d[k_] = hp_
                    eco_hits[dl] += hp_
                add("wanted", nm, num=d)

        # 选项对象
        keys = list(p["options"])
        opt_pos = []
        opt_named: dict[int, str] = {}
        for k in keys:
            ot = opt_type(kind, k)
            nm, desc, ref = "", "", -1
            num: dict = {}
            m = re.search(r"\d+", k)
            n_ = int(m.group()) if m else -1
            if kind == "shop" and ot == "buy":
                ref = shop_idx.get(n_, -1)
                ent = next((e for e in raw.get("shop") or [] if e[0] == n_), None)
                nm = ent[1] if ent else ""
            elif kind in ("talent", "lord_boost") and ot == "pick":
                lst = raw.get("talents" if kind == "talent" else "boosts") or []
                if 0 <= n_ < len(lst):
                    nm, desc = lst[n_][0], lst[n_][1] if len(lst[n_]) > 1 else ""
            elif kind == "card_pick" and ot == "pick":
                lst = raw.get("cards") or []
                nm = lst[n_] if 0 <= n_ < len(lst) else ""
            elif kind == "use_card" and ot == "use":
                lst = raw.get("targets") or []
                nm = lst[n_] if 0 <= n_ < len(lst) else ""
            elif kind in ("carry", "swap", "auction") and ot in ("keep", "switch", "sell", "lot"):
                nm = (raw.get("names") or {}).get(k, "")
            elif kind == "route" and ot == "route":
                from agent.decision import routes as R
                lu = R.by_key((raw.get("keys") or {}).get(k))
                nm = (lu or {}).get("carry") or (R.final_team(lu)[0] if lu and R.final_team(lu) else "")
            if ref < 0 and nm:
                ref = board_idx.get(nm, hand_idx.get(nm, -1))
            base = dict(zip(range(NUM_DIM), objs[ref]["num"])) if ref >= 0 else {}
            base.update(num)
            lc = None
            if ot == "levelup":
                lc = _num(ctx.get("level_cost"))
                if lc is not None and lc > 0 and gold is not None:
                    base[26] = float(gold >= lc)
            if ot == "keep":
                base[27] = 1.0
            if kind == "eco":
                # eco 选项：17 = 选这一项后每刷一次期望见到的缺件数（levelup 按 L+1、roll/save 按 L）、18 = 人口 +1（levelup）、
                # 25 = 能刷几次 / 10（roll）或能量 / 40（save）、26 = 付得起（levelup 已在上面按升级费写）
                if ot == "levelup":
                    base[17] = eco_hits[1]
                    base[18] = 1.0 if lvl and M.level_raises_pop(lvl) else 0.0
                    if lc is None or lc <= 0:
                        x_, xn_ = _num(ctx.get("xp")), _num(ctx.get("xp_needed"))
                        if x_ is not None and xn_ and gold is not None:
                            base[26] = float(gold >= xn_ - x_)
                elif ot == "refresh":
                    base[17] = eco_hits[0]
                    n_ref = (int(gold) // 2 if gold else 0) + int(_num(ctx.get("free_refresh")) or 0)
                    base[25] = min(n_ref, 20) / 10
                    base[26] = float(n_ref > 0)
                elif ot == "stop":
                    base[17] = eco_hits[0]
                    base[25] = (gold or 0) / 40
            i = add("option", nm, desc=desc, num=base, ot=ot, ref=ref)
            opt_pos.append(i)
            if kind == "route" and ot == "route":
                from agent.decision import routes as R
                lu = R.by_key((raw.get("keys") or {}).get(k))
                opt_named[i] = "route:" + str((raw.get("keys") or {}).get(k))
                objs[i]["route_mem"] = R.members(lu) if lu else set()

        # 手法节点：与场上出现的卡有 enables / fits_role 边
        present = {o["nid"] for o in objs if o["nid"]}
        techs = sorted({t for n in present for t in G.tech_of.get(n, ())})
        for t in techs:
            add("technique", t)

        # 关系（稀疏三元组 i, j, 通道）
        E2 = 2 * len(EDGE_TYPES)
        rel = []
        relv = []
        C0 = E2 + EXTRA_CH.index("lineup_cooc")
        N = len(objs)
        for i in range(N):
            a = objs[i]["nid"]
            for j in range(N):
                if i == j:
                    continue
                b = objs[j]["nid"]
                if a and b:
                    if a == b:
                        rel.append((i, j, E2 + 0))
                    m = G.pair.get((a, b), 0)
                    c = 0
                    while m:
                        if m & 1:
                            rel.append((i, j, c))
                        m >>= 1
                        c += 1
                if a and b and a != b:
                    for c_, dct, key in ((C0, G.cooc, (a, b)), (C0 + 1, G.carry_with, (a, b)), (C0 + 2, G.carry_with, (b, a))):
                        val = dct.get(key)
                        if val:
                            relv.append((i, j, c_, val))
                if objs[i]["ref"] == j or objs[j]["ref"] == i:
                    rel.append((i, j, E2 + 1))
                rm = objs[i].get("route_mem")
                if rm and objs[j]["name"] in rm and objs[j]["tt"] in (1, 2):
                    rel.append((i, j, E2 + 2))
                    rel.append((j, i, E2 + 2))
                if subj >= 0 and i == subj and objs[j]["tt"] == TOK_TYPES.index("option"):
                    rel.append((i, j, E2 + 3))
                    rel.append((j, i, E2 + 3))
        rel = sorted(set(rel))
        rv = [1.0] * len(rel) + [x[3] for x in relv]
        rel = np.array(rel + [x[:3] for x in relv], np.int16).reshape(-1, 3)
        rv = np.array(rv, np.float16)

        # 全局特征
        gv = np.zeros(GLOB_DIM, np.float32)

        def put(i, x, scale, known_i=None):
            x = _num(x)
            if x is not None:
                gv[i] = x / scale
                if known_i is not None:
                    gv[known_i] = 1.0

        put(0, ctx.get("round") or p.get("round"), 15)
        put(1, ctx.get("gold"), 40, 2)
        put(3, ctx.get("level"), 6)
        lc = _num(ctx.get("level_cost"))
        if lc is not None and lc > 0:
            gv[4], gv[5] = lc / 20, 1.0
        put(6, ctx.get("pop"), 8, 7)
        put(8, ctx.get("pop_cap"), 8, 9)
        if _num(ctx.get("pop")) is not None and _num(ctx.get("pop_cap")) is not None:
            gv[10] = (ctx["pop_cap"] - ctx["pop"]) / 4
        if _num(ctx.get("xp_needed")):
            gv[11] = (ctx.get("xp") or 0) / ctx["xp_needed"]
        put(12, ctx.get("free_refresh"), 3)
        put(13, ctx.get("auction_in"), 4)
        gv[14] = float(bool(ctx.get("auction_done")))
        put(15, ctx.get("shop_max_tier"), 5)
        gv[16] = 1.0 if maple else 0.0                          # 枫叶舞绑定了某人
        gv[17] = 1.0 if maple == "" else 0.0                    # 明确未绑
        gv[18] = 1.0 if maple is None else 0.0                  # 未知
        gv[19] = len(ctx.get("lord_boosts") or []) / 3
        gv[20] = 1.0 if route_key else 0.0
        put(21, ctx.get("carry_since"), 15)
        put(22, ctx.get("carry_switches"), 5)
        gv[23] = 1.0 if carry else 0.0
        gv[24] = len(hand) / 10
        gv[25] = len(board) / 8
        for j, f in enumerate(FACTIONS):
            gv[26 + j] = 1.0 if f in banned else 0.0
        if kind in KINDS:
            gv[31 + KINDS.index(kind)] = 1.0                       # 31..39
        else:
            gv[EXTRA_KIND_GV[kind]] = 1.0                          # eco → 62
        # 40..: 游戏规则表（agent/decision/meta.py，实测/攻略表，不是规则引擎的判断）
        rnd = _num(ctx.get("round") or p.get("round"))
        if rnd:
            tgt_lv = M.level_target(int(rnd))
            gv[40] = tgt_lv / 6
            if lvl:
                gv[41] = (lvl - tgt_lv) / 3
            gv[42] = M.energy_income(int(rnd) + 1) / 15
            gv[43] = M.rounds_to_auction(int(rnd)) / 4
        if lvl:
            for q in range(1, 6):
                gv[43 + q] = M.tier_odds(q, lvl)                 # 44..48
                gv[48 + q] = M.tier_odds(q, min(lvl + 1, 6))     # 49..53
            gv[54] = 1.0 if M.level_raises_pop(lvl) else 0.0
            gv[55] = 1.0
        ul = [x for x in (ctx.get("unit_levels") or []) if _num(x) is not None]
        for j, x in enumerate(sorted(ul, reverse=True)[:3]):
            gv[56 + j] = float(np.log1p(x) / 5)                  # 56..58
        put(59, ctx.get("units_seen"), 8)
        gv[60] = len(ctx.get("hand_awk") or []) / 2
        gv[61] = min(len(owned_talents), 12) / 6

        tgt = np.array([float(lab["dist"].get(k, 0.0)) for k in keys], np.float32)
        tgt = tgt / tgt.sum() if tgt.sum() > 0 else np.full(len(keys), 1 / len(keys), np.float32)
        return {"id": p["id"], "kind": kind, "split": p["split"], "run": p.get("run"), "source": p.get("source", ""),
                "keys": keys, "t_best": lab["best"], "target": tgt, "rule_choice": p.get("rule_choice"),
                "tt": np.array([o["tt"] for o in objs], np.int8), "ot": np.array([o["ot"] for o in objs], np.int8),
                "node": np.array([o["node"] for o in objs], np.int16), "text": np.array([o["text"] for o in objs], np.int32),
                "num": np.stack([o["num"] for o in objs]), "names": [o["name"] for o in objs],
                "opt_pos": np.array(opt_pos, np.int16), "rel": rel, "relv": rv, "glob": gv, "board_src": board_src,
                "round": ctx.get("round") or p.get("round")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="r9m")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--encs", default="bge,laya")
    a = ap.parse_args()
    labels_path = Path(a.labels) if a.labels else PROJECT_DIR / "data" / "laya" / "labels" / "r6" / f"labels_all_{a.ds}.jsonl"
    labs = {}
    for line in labels_path.open(encoding="utf-8"):
        d = json.loads(line)
        labs[d["id"]] = d
    G = Graph()
    B = Builder(G, None)
    out = []
    snaps_cache: dict = {}
    skipped = collections.Counter()
    for line in (PROJECT_DIR / "data" / "laya" / "datasets" / a.ds / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        lab = labs.get(p["id"])
        if lab is None:
            skipped["no_label"] += 1
            continue
        run = p.get("run") or ""
        if run not in snaps_cache:
            snaps_cache[run] = replay_run(run)
        owned, lb = snapshot(snaps_cache[run], float(p.get("ts") or 0))
        s = B.convert(p, lab, owned, lb)
        if s is None:
            skipped["convert"] += 1
            continue
        out.append(s)
    od = OUT_ROOT / a.ds
    od.mkdir(parents=True, exist_ok=True)
    with (od / "samples.pkl").open("wb") as f:
        pickle.dump(out, f)
    np.save(od / "node_feat.npy", G.node_feat())
    (od / "text_keys.json").write_text(json.dumps(B.extra_texts, ensure_ascii=False), encoding="utf-8")
    for enc in a.encs.split(","):
        e = Encoder(enc)
        ne = np.load(OUT_ROOT / f"text_emb_{enc}.npy")
        ex = e.encode(B.extra_texts) if B.extra_texts else np.zeros((0, ne.shape[1]), np.float32)
        table = np.concatenate([np.zeros((1, ne.shape[1]), np.float32), ne, ex]).astype(np.float32)
        np.save(od / f"text_table_{enc}.npy", table)
    by = collections.Counter((s["kind"], s["split"]) for s in out)
    nobj = [len(s["tt"]) for s in out]
    hits, miss = sum(B.name_hits.values()), sum(B.name_miss.values())
    meta = {"ds": a.ds, "labels": str(labels_path.relative_to(PROJECT_DIR)), "n": len(out), "skipped": dict(skipped),
            "by_kind_split": {f"{k}:{s}": v for (k, s), v in sorted(by.items())},
            "tok_types": TOK_TYPES, "opt_types": OPT_TYPES, "kinds": KINDS, "edge_types": EDGE_TYPES,
            "extra_channels": EXTRA_CH, "n_channels": 2 * len(EDGE_TYPES) + len(EXTRA_CH),
            "num_dim": NUM_DIM, "glob_dim": GLOB_DIM, "n_nodes": len(G.nodes),
            "objects_per_sample": {"mean": float(np.mean(nobj)), "max": int(np.max(nobj))},
            "rel_per_sample_mean": float(np.mean([len(s["rel"]) for s in out])),
            "name_hit_rate": hits / max(1, hits + miss), "name_miss_top": B.name_miss.most_common(40),
            "board_src": dict(collections.Counter((s["kind"], s["board_src"]) for s in out).most_common()),
            "board_src_str": {f"{k}:{v}": c for (k, v), c in collections.Counter((s["kind"], s["board_src"]) for s in out).items()},
            "n_extra_texts": len(B.extra_texts),
            "n_techniques_mean": float(np.mean([int((s["tt"] == TOK_TYPES.index("technique")).sum()) for s in out]))}
    meta.pop("board_src")
    (od / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("name_miss_top",)}, ensure_ascii=False, indent=1))
    print("miss top:", B.name_miss.most_common(25))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
