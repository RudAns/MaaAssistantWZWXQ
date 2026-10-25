"""cgnet v2 数据：在 v1 对象表之上加三类信息（都只由结构化状态 + 卡面 / 图谱 / 规则表数据算出，不用规则引擎的判断、不用 facts.py 的结论）。

1. 汇总事实（全局 64..111 维 + 对象数值 28..35 维）
   手里等上场的英雄数 vs 空位、升级费 vs 能量、刷不刷得起、商店里同名 / 差一张觉醒 / 路线成员 / 引擎缺件 / 买得起的张数、
   "想要的牌"刷新一次至少出一张的概率（meta.shop_hit_prob，当前等级与升一级）、场上事实缺失位、引擎完成度、主C 概况。
2. 手法动态角色绑定：对每个手法节点，按图谱里 roles 的状态谓词在当前局面判每个对象能不能填每个槽
   （能判的判，判不了的算"未知"：全真 = 1，有未知无假 = 0.5，任一假 = 0）→ 每个槽一个关系通道（对象↔手法，连续值）；
   手法对象数值（36..43）：启用牌在手 / 前置条件满足 / 各槽有人可填的比例 / 按当前人算的预期收益 / 遗志命中率。
   手法对象只在"启用牌在手 / 已得"或"每个槽都有候选"时加入；v1 的手法对象（只按边相连加入）先去掉再按新规则加。
3. 条件边（10 个通道）：牌 → 英雄 的关系按目标状态分开：觉醒进度+1 给未觉醒 / 给已觉醒、变身牌给普通 / 给主C·已觉醒·高等级、
   移除类给普通 / 给主C·已觉醒、加等级给主C / 给别人、装备给已满 3 件的人、品阶限制不符。

用法：
  python tools/cgnet/build_data_v2.py --ds r9m --out r9m_v2                       # v2
  python tools/cgnet/build_data_v2.py --ds r9m --out r9m_v1g2 --v1only            # "v1 配置 + 当前图谱"（分离图谱更新的影响）
  --graph 指定图谱文件（默认 data/card_graph.json）
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
import build_data as BD  # noqa: E402
from textenc import Encoder, node_text  # noqa: E402
from agent.decision import meta as M  # noqa: E402
from agent.decision import engines as EN  # noqa: E402
MIN_ENGINE_FEAT = False    # 2026-10-11 路线最低配套特征（num 42/43、glob 94~96；data/route_min_engine.json）。部署包 manifest "min_engine_feat" 为真时线上打开
_MIN_ENG = None


def _min_engine(route_key) -> dict:
    global _MIN_ENG
    if _MIN_ENG is None:
        try:
            _MIN_ENG = json.loads((PROJECT_DIR / "data" / "route_min_engine.json").read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            _MIN_ENG = {}
    return (_MIN_ENG or {}).get(str(route_key or "")) or {}


def _min_groups(me: dict) -> list:
    """与 Laya 题面同一份解析（tools/laya/facts.min_engine_groups：只认真实英雄名、跳过"（牌）"条件）。"""
    import importlib
    if str(PROJECT_DIR / "tools" / "laya") not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))
    return importlib.import_module("facts").min_engine_groups(me)


FLOW_FEAT = False          # 2026-10-14 流派特征（glob 103~108；agent/sim/flows.flow_rank，和 Laya 题面 f9lp 的 FLOW OPTIONS 同一份）。部署包 manifest "flow_feat" 为真时线上打开
ECO_CORE_FEAT = False      # 2026-10-13 eco 题 f9ln「CORE pieces … best at lvK」（glob 97~102，只在 eco 题写）。部署包 manifest "eco_core_feat" 为真时线上打开
                           # ⚠ 线上 agent/decision/cgnet_policy.py 目前只按 manifest 切 MIN_ENGINE_FEAT，要让它也切这个开关；旧部署包输入不变
ECO_CORE_SRC = ("MIN ENGINE core", "ROUTE CARRY not owned")   # 与 tools/laya/facts.eco_state_extra 的 core 一致


def _eco_hit(q: int, L: int):
    """与 tools/laya/facts._eco_hit 同口径（不 import facts，避免切版本）：刷一次见到某张 q 阶英雄的概率；没实测 → None。"""
    if not q or not L or L > M.MAX_LORD_LEVEL:
        return None
    if not M.can_appear(q, L):
        return 0.0
    if M.tier_odds(q, L) <= 0:
        return None
    return M.shop_hit_prob(q, L)


def eco_core_glob(missing: list, lvl) -> dict:
    """{glob 下标: 值}：97~99 = 核心缺件在 L/L+1/L+2 每刷期望见到几张；100 = (最好刷的等级 − L)/2；101 = 核心缺件数/4；102 = 有核心缺件。"""
    core = [(m[0], int(m[1] or 0)) for m in (missing or []) if len(m) > 2 and m[2] in ECO_CORE_SRC]
    if not core or not lvl:
        return {}
    lvls = list(range(lvl, min(lvl + 2, M.MAX_LORD_LEVEL) + 1))
    sums = {lv: sum(x for x in (_eco_hit(q, lv) for _, q in core) if x is not None) for lv in lvls}
    best = max(lvls, key=lambda lv: (round(sums[lv], 3), -lv))
    out = {97 + j: sums[lv] for j, lv in enumerate(lvls)}
    out[100] = (best - lvl) / 2
    out[101] = len(core) / 4
    out[102] = 1.0
    return out


ENGINE_CORRECTED = False   # 2026-10-11 引擎目录修正（太乙不是放大器）：已部署的 cgnet 用未修正版训练；新一轮训练要用修正版时改 True 并重建数据
from agent.decision import routes as R  # noqa: E402

OUT_ROOT = PROJECT_DIR / "data" / "cgnet"
NUM_DIM2 = 44              # ENGINE_FEAT 打开时置 48（num 44~47 = 推演器对这个选项的 ΔV / 10%分位 / 90%分位 / p_best）
ENGINE_FEAT = False         # 2026-10-14 推演器特征（tools/engine/annotate_points.py）。部署包 manifest "engine_feat" 为真时线上打开
ENGINE_ANNOT: dict = {}     # 题 id -> 推演器打分（离线建数据集时读 data/engine/annot_*.jsonl；线上由 cgnet_policy 现算后放进 p["_engine"]）


def load_engine_annot(path) -> int:
    n = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("id") and r.get("opts"):
            ENGINE_ANNOT[r["id"]] = r
            n += 1
    return n
W_FEAT = False              # 2026-10-16 f9lw（=2：再加 num 49 场上杂牌 filler_on_board，NUM_DIM2 = 50）（裁决 ㊱~㊴）：num 44~48 = 纯过渡件 / 已回本 / 别家主C / 枫叶舞在临时主C 上 / 等位的路线缺件。
                            # 打开时 NUM_DIM2 = 49（与 ENGINE_FEAT 互斥）；部署包 manifest "w_feat" 为真时线上打开
GLOB_DIM2 = 112


def w_unit_facts(p: dict) -> dict:
    """f9lw 的 5 个单位级事实（和 Laya 题面 f9lw 同一套 facts.Board 判定，train / serve 一致）。返回 {名字: [5 个 0/1]}。"""
    import sys as _s
    _s.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))
    import facts as X
    ctx = dict(p.get("ctx") or {})
    if isinstance(ctx.get("field"), dict) and ctx["field"]:
        field = ctx["field"]
    else:
        board = p.get("board") or {}
        field = {n: dict(board.get(n) or {}) for n in (p.get("field") or [])}
    ctx["field"] = field
    if "lordxp_since" not in ctx and p.get("run"):
        try:
            import rerender as RR
            ctx["lordxp_since"] = RR.carry_hist_at(p.get("run"), p.get("ts")).get("lordxp_since") or {}
        except Exception:  # noqa: BLE001
            ctx["lordxp_since"] = {}
    old = X.FV
    X.FV = "f9lz" if W_FEAT >= 3 else ("f9lx" if W_FEAT >= 2 else "f9lw")   # W_FEAT=3（cs_r9zp_x 起）：f9lz —— 路线外"带牺牲"不算喂主C     # W_FEAT=2（cs_r9zo_w 起）按 f9lx 判：自定义路线主C 也算别家主C（第 665 局马可波罗）
    try:
        B = X.Board(ctx)
        out = {}
        maple = ctx.get("maple") or ""
        incoming = (p.get("raw") or {}).get("incoming") or ""
        names = set(B.heroes) | set(B.hand) | ({incoming} if incoming else set())
        route_final = set(X._routes().final_team(B.route)) if (B.route and not B.route_stale) else set()
        for n in names:
            if not n:
                continue
            filler = 1.0 if X.lord_xp_unit(n) else 0.0
            paid = 1.0 if (filler and n in B.heroes and (B.filler_paid(n) or 0) >= 1) else 0.0
            olc = 1.0 if B.other_lineup_carry(n) else 0.0
            mt = 1.0 if (n == maple and n in B.heroes and B.maple_on_temp(n)) else 0.0
            wait = 1.0 if (n == incoming and (n in route_final or B.route_member(n) or X.min_engine_tag(n, B, False))) else 0.0
            row = [filler, paid, olc, mt, wait]
            if W_FEAT >= 2:
                # 2026-10-16 第 662 局：曜（1 级、多余输出、不在路线）五维全 0，换人头看不出它是可卖的杂牌（程咬金等位却 bench）
                fob = 1.0 if (n in B.heroes and n not in (B.fc, B.cur_carry, B.main_tank) and X.is_filler(n, B)
                              and not B.feeds_or_part(n)) else 0.0
                row.append(fob)
            if W_FEAT >= 3:
                # 2026-10-16 晚第 677 局：盾山（路线前期优先件）等位两回合都 bench —— 场上廉颇 / 鬼谷子 25 级、不在路线，
                # 但"场上杂牌"只标 1~3 级，网络看不出它们是路线外的人（标签 157 题 bench 平均 0.07，网络仍给 0.4~0.5）。
                # num 50 = 场上不在路线、不喂主C 的单位（不看等级）
                offr = 1.0 if (n in B.heroes and route_final and n not in route_final and not B.route_member(n)
                               and n not in (B.fc, B.cur_carry) and not B.feeds_or_part(n)) else 0.0
                row.append(offr)
            if W_FEAT >= 4:
                # 2026-10-17 第 684 局：锁定"伞亚连"后 carry 网络 R7~R9 一直"留着苍"（苍 lv20~30 路线外，亚连 lv7~15 是路线主C）——
                # 网络只看得到"路线缺件"（num 30），看不到"这就是锁定路线的主C"。num 51 = 锁定路线的主C；num 52 = 现任主C 不在路线里
                rlock = bool(B.route and not B.route_stale)
                rcar = 1.0 if (rlock and n == (B.route.get("carry") or "")) else 0.0
                coff = 1.0 if (rlock and n == B.cur_carry and n != (B.route.get("carry") or "") and n not in route_final
                               and not B.route_member(n)) else 0.0
                row += [rcar, coff]
            out[n] = row
        return out
    finally:
        X.FV = old

V1_NUM, V1_GLOB = 28, 64
COND_CH = ["awaken_ok", "awaken_wasted", "transform_ok", "transform_bad", "remove_ok", "remove_bad",
           "level_to_carry", "level_to_other", "equip_to_full", "tier_illegal"]
FACT_GLOB = (64, 112)          # 消融 no_facts 置零的全局维
FACT_NUM = (28, 36)            # 消融 no_facts 置零的对象维
TECH_NUM = (36, 44)            # 消融 no_techdyn 置零的对象维


def _cards() -> dict:
    return json.loads((PROJECT_DIR / "data" / "cards.json").read_text(encoding="utf-8")).get("cards") or {}


class V2:
    def __init__(self, G: BD.Graph):
        self.G = G
        self.cards = _cards()
        self.techs = [n for n in G.nodes if n["type"] == "technique"]
        self.slots = [(t["id"], r) for t in self.techs for r in (t.get("roles") or {})]
        self.enablers = collections.defaultdict(set)          # 手法 -> 启用它的卡
        g = json.loads(Path(G.path).read_text(encoding="utf-8"))
        for e in g["edges"]:
            if e["type"] == "enables" and e["to"].startswith("tech:"):
                self.enablers[e["to"]].add(e["from"])
        self.gives_level = {e["from"] for e in g["edges"] if e["type"] == "gives_level"}
        self.hero_ids = {n["id"] for n in G.nodes if n["type"] == "hero"}
        # 名字解析结果缓存（纯函数，结果不变，只为提速）
        if not getattr(G, "_resolve_cached", False):
            raw_resolve = G.resolve
            cache: dict = {}

            def resolve(name, _r=raw_resolve, _c=cache):
                if name not in _c:
                    _c[name] = _r(name)
                return _c[name]
            G.resolve = resolve
            G._resolve_cached = True

    # ------------------------------------------------------------ 卡面文字
    def text(self, name: str) -> str:
        nid = self.G.resolve(name)
        if nid:
            return self.G.node[nid].get("text") or ""
        c = self.cards.get(name) or {}
        return (c.get("skill") or "") + (c.get("awaken") or "")

    def quality(self, name: str) -> int:
        nid = self.G.resolve(name)
        q = (self.G.node.get(nid) or {}).get("quality") if nid else (self.cards.get(name) or {}).get("quality")
        try:
            return int(q or 0)
        except ValueError:
            return 0

    # ------------------------------------------------------------ 对象状态（从 v1 数值列反解）
    @staticmethod
    def ostate(s: dict, i: int, carry: str, board_names: set, hand_names: set) -> dict:
        v = s["num"][i]
        nm = s["names"][i]
        return {"name": nm, "tt": int(s["tt"][i]),
                "lv": float(np.expm1(v[0] * 5)) if v[1] > 0 else None,
                "awk_left": (3 - round(v[2] * 3)) if v[3] > 0 else None,
                "eq": round(v[6] * 3) if v[7] > 0 else None,
                "row": v[22] if v[23] > 0 else None,
                "is_carry": bool(carry) and nm == carry, "maple": v[11] > 0, "hand_awk": v[12] > 0,
                "on_board": nm in board_names, "in_hand": nm in hand_names}

    # ------------------------------------------------------------ 谓词
    def pred(self, p: str, o: dict, ctx: dict) -> bool | None:
        """谓词求值：解析一次、编译成闭包（结果与 _pred_slow 逐个相同；只为线上提速）。"""
        comp = self.__dict__.setdefault("_pc", {})
        f = comp.get(p)
        if f is None:
            f = comp[p] = self._compile(p)
        return f(o, ctx)

    def _compile(self, p: str):
        p = p.split("#")[0].strip()
        if not p:
            return lambda o, ctx: None
        if " or " in p:
            fs = [self._compile(x) for x in p.split(" or ")]

            def f_or(o, ctx):
                rs = [g(o, ctx) for g in fs]
                if any(r is True for r in rs):
                    return True
                return None if any(r is None for r in rs) else False
            return f_or
        if p.startswith("not "):
            g = self._compile(p[4:])

            def f_not(o, ctx):
                r = g(o, ctx)
                return None if r is None else (not r)
            return f_not
        G = self.G
        hero_ids = self.hero_ids

        def nd_of(o):
            nid = G.resolve(o["name"])
            return (G.node.get(nid) or {}), nid in hero_ids

        m = re.match(r"real_level\s*(>=|>)\s*(\d+)", p)
        if m:
            op, k = m.group(1), int(m.group(2))
            return lambda o, ctx: None if o["lv"] is None else (o["lv"] >= k if op == ">=" else o["lv"] > k)
        simple = {
            "is_maple_holder": lambda o, ctx: None if ctx.get("maple") is None else o["maple"],
            "on_board": lambda o, ctx: o["on_board"],
            "in_hand": lambda o, ctx: o["in_hand"],
            "is_carry": lambda o, ctx: o["is_carry"],
            "carry_eligible": lambda o, ctx: (lambda nd, h: h and float(nd.get("carry_value") or 0) >= 5)(*nd_of(o)),
            "awakened": lambda o, ctx: True if (o["hand_awk"] or o["awk_left"] == 0) else (None if o["awk_left"] is None else False),
            "front_row": lambda o, ctx: None if o["row"] is None else o["row"] <= 1 / 3 + 1e-6,
        }
        if p in simple:
            return simple[p]
        m = re.match(r"equip_count\s*(>=|<|==)\s*(\d+)", p)
        if m:
            op, k = m.group(1), int(m.group(2))
            return lambda o, ctx: None if o["eq"] is None else {">=": o["eq"] >= k, "<": o["eq"] < k, "==": o["eq"] == k}[op]
        m = re.match(r"has_keyword\((.+?)\)", p)
        if m:
            kw = m.group(1)
            return lambda o, ctx: (lambda nd, h: h and (kw in (nd.get("keywords") or []) or f"{kw}：" in (nd.get("text") or "")))(*nd_of(o))
        m = re.match(r"name\s*==\s*(\S+)", p)
        if m:
            nm = m.group(1)
            return lambda o, ctx: o["name"] == nm
        m = re.match(r"name in \[(.+?)\]", p)
        if m:
            lst = [x.strip() for x in m.group(1).split(",")]
            return lambda o, ctx: o["name"] in lst
        m = re.match(r"tier in \[(\d+),\s*(\d+)\]", p)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            return lambda o, ctx: (lambda q: lo <= q <= hi if q else None)(self.quality(o["name"]))
        m = re.match(r"tier\s*>=\s*(\d+)", p)
        if m:
            k = int(m.group(1))
            return lambda o, ctx: (lambda q: q >= k if q else None)(self.quality(o["name"]))
        m = re.match(r"faction\s*==\s*(\S+)", p)
        if m:
            fa = m.group(1)
            return lambda o, ctx: (lambda nd, h: (nd.get("faction") or "") == fa if nd else None)(*nd_of(o))
        m = re.match(r"carry_value\s*<=\s*(\d+)", p)
        if m:
            k = int(m.group(1))
            return lambda o, ctx: (lambda nd, h: float(nd.get("carry_value") or 0) <= k if h else None)(*nd_of(o))
        return lambda o, ctx: None

    def _pred_slow(self, p: str, o: dict, ctx: dict) -> bool | None:
        """原始实现（保留作对照：tools/cgnet/build_data_v2.py 的编译版必须与它逐个相同）。"""
        p = p.split("#")[0].strip()
        if not p:
            return None
        if " or " in p:
            rs = [self._pred_slow(x, o, ctx) for x in p.split(" or ")]
            if any(r is True for r in rs):
                return True
            return None if any(r is None for r in rs) else False
        if p.startswith("not "):
            r = self._pred_slow(p[4:], o, ctx)
            return None if r is None else (not r)
        nid = self.G.resolve(o["name"])
        nd = self.G.node.get(nid) or {}
        is_hero = nid in self.hero_ids
        lv = o["lv"]
        m = re.match(r"real_level\s*(>=|>)\s*(\d+)", p)
        if m:
            if lv is None:
                return None
            return lv >= int(m.group(2)) if m.group(1) == ">=" else lv > int(m.group(2))
        if p == "is_maple_holder":
            return None if ctx.get("maple") is None else o["maple"]
        if p == "on_board":
            return o["on_board"]
        if p == "in_hand":
            return o["in_hand"]
        if p == "is_carry":
            return o["is_carry"]
        if p == "carry_eligible":
            return is_hero and float(nd.get("carry_value") or 0) >= 5
        if p == "awakened":
            if o["hand_awk"] or o["awk_left"] == 0:
                return True
            return None if o["awk_left"] is None else False
        m = re.match(r"equip_count\s*(>=|<|==)\s*(\d+)", p)
        if m:
            if o["eq"] is None:
                return None
            k = int(m.group(2))
            return {">=": o["eq"] >= k, "<": o["eq"] < k, "==": o["eq"] == k}[m.group(1)]
        m = re.match(r"has_keyword\((.+?)\)", p)
        if m:
            kw = m.group(1)
            return is_hero and (kw in (nd.get("keywords") or []) or f"{kw}：" in (nd.get("text") or ""))
        if p == "front_row":
            return None if o["row"] is None else o["row"] <= 1 / 3 + 1e-6
        m = re.match(r"name\s*==\s*(\S+)", p)
        if m:
            return o["name"] == m.group(1)
        m = re.match(r"name in \[(.+?)\]", p)
        if m:
            return o["name"] in [x.strip() for x in m.group(1).split(",")]
        m = re.match(r"tier in \[(\d+),\s*(\d+)\]", p)
        if m:
            q = self.quality(o["name"])
            return int(m.group(1)) <= q <= int(m.group(2)) if q else None
        m = re.match(r"tier\s*>=\s*(\d+)", p)
        if m:
            q = self.quality(o["name"])
            return q >= int(m.group(1)) if q else None
        m = re.match(r"faction\s*==\s*(\S+)", p)
        if m:
            return (nd.get("faction") or "") == m.group(1) if nd else None
        m = re.match(r"carry_value\s*<=\s*(\d+)", p)
        if m:
            return float(nd.get("carry_value") or 0) <= int(m.group(1)) if is_hero else None
        return None                     # 判不了（相邻、随机、低价值、是否刚被卖…）

    def fill(self, preds: list, o: dict, ctx: dict) -> float:
        # 记忆化：谓词结果只取决于谓词本身 + 对象状态 + 枫叶舞是否已知（结果与不缓存时逐位相同，只为提速）
        rs = [self.pred(p, o, ctx) for p in preds]      # 谓词已编译成闭包，直接求值比查缓存快
        if any(r is False for r in rs):
            return 0.0
        if all(r is True for r in rs):
            return 1.0
        return 0.5 if any(r is True for r in rs) else 0.0

    # ------------------------------------------------------------ 主入口
    def augment(self, s: dict, p: dict, meta: dict) -> dict:
        s = dict(s)
        G = self.G
        ctx = p.get("ctx") or {}
        carry = ctx.get("carry") or ""
        tt_tech = meta["tok_types"].index("technique")
        tt_opt = meta["tok_types"].index("option")
        E2 = meta["n_channels"]                         # v1 通道数（49）
        # 去掉 v1 的手法对象（都在最后）
        keep = int((s["tt"] != tt_tech).sum())
        for k in ("tt", "ot", "node", "text"):
            s[k] = s[k][:keep]
        s["names"] = list(s["names"][:keep])
        r = s["rel"]
        km = (r[:, 0] < keep) & (r[:, 1] < keep) if len(r) else np.zeros(0, bool)
        rel = [tuple(x) for x in r[km]] if len(r) else []
        relv = list(s["relv"][km].astype(np.float32)) if len(r) else []
        num = np.zeros((keep, NUM_DIM2), np.float32)
        num[:, :V1_NUM] = s["num"][:keep]
        glob = np.zeros(GLOB_DIM2, np.float32)
        glob[:V1_GLOB] = s["glob"]
        s["num"] = num
        if ENGINE_FEAT:
            ann = p.get("_engine") or ENGINE_ANNOT.get(p.get("id")) or {}
            opos = list(s.get("opt_pos") if s.get("opt_pos") is not None else [])
            for i, k in enumerate(p.get("options") or []):
                o = (ann.get("opts") or {}).get(k)
                if o is None or i >= len(opos) or int(opos[i]) >= keep:
                    continue
                j = int(opos[i])
                num[j, 44] = float(np.tanh(o.get("dV", 0.0) / 10.0))
                num[j, 45] = float(np.tanh(o.get("lo", 0.0) / 10.0))
                num[j, 46] = float(np.tanh(o.get("hi", 0.0) / 10.0))
                num[j, 47] = float(o.get("p_best", 0.0))
        board_names = {n for n, t in zip(s["names"], s["tt"]) if t == 1}
        hand_names = {n for n, t in zip(s["names"], s["tt"]) if t == 2}
        if W_FEAT:
            assert NUM_DIM2 >= 44 + {1: 5, True: 5, 2: 6, 3: 7, 4: 9}.get(W_FEAT, 5) and not ENGINE_FEAT, "W_FEAT 要 NUM_DIM2=49/50/51、不能和 ENGINE_FEAT 同开"
            wf = w_unit_facts(p)
            for i in range(keep):
                v = wf.get(s["names"][i])
                if v and s["tt"][i] in (1, 2, 3, tt_opt, 6):
                    num[i, 44:44 + len(v)] = v
        states = [self.ostate(s, i, carry, board_names, hand_names) for i in range(keep)]
        lvl = ctx.get("level") if isinstance(ctx.get("level"), int) and 1 <= ctx.get("level") <= 6 else None
        gold = ctx.get("gold") if isinstance(ctx.get("gold"), (int, float)) else None
        owned = collections.Counter([s["names"][i] for i in range(keep) if s["tt"][i] in (1, 2) and s["names"][i]])
        route_mem: set = set()
        if ctx.get("route"):
            lu = R.by_key(ctx["route"])
            route_mem = R.members(lu) if lu else set()
        _old, EN.CORRECTED = EN.CORRECTED, ENGINE_CORRECTED     # 线上 cs_r9t / cs_r9x 用未修正的目录训练 → 保持一致
        try:
            eng = EN.best(set(owned))
            missing = set(EN.missing_parts(eng["key"], set(owned))) if eng else set()
            eng_parts = EN.parts(eng["key"]) if eng else set()
        finally:
            EN.CORRECTED = _old
        board_awk = {st["name"]: st["awk_left"] for st in states if st["tt"] == 1}
        if MIN_ENGINE_FEAT and ctx.get("route"):
            # 最低配套（2026-10-11：先凑最低配套，再凑其他英雄；太乙之类前期不急）：
            # num[42] = 这个英雄属于最低配套的某一槽；num[43] = 后期件且在第 8 回合以前；glob[94..96] = 场上已齐 / 缺几槽 / 第 8 回合以前
            me = _min_engine(ctx["route"])
            groups = _min_groups(me)
            rnd = ctx.get("round") or 0
            early = 0 < rnd <= 8
            late = set(me.get("late_fill") or []) - {x for g in groups for x in g}
            for i, st in enumerate(states):
                nm = st["name"]
                if not nm:
                    continue
                num[i, 42] = 1.0 if any(nm in g for g in groups) else 0.0
                num[i, 43] = 1.0 if (early and nm in late) else 0.0
            if groups:
                on = sum(any(x in board_names for x in g) for g in groups)
                glob[94] = on / len(groups)
                glob[95] = (len(groups) - on) / 5
                glob[96] = 1.0 if early else 0.0
        if FLOW_FEAT:
            # 流派离成型还差什么（如核心配件场上有一个、商店里有两个，买了就凑齐）：
            # glob[103] 最接近成型的流派已有比例；[104] 现在买就凑齐；[105] 缺的里商店有几件 /5；[106] 缺几件 /5；
            # [107] 已有一半以上的流派个数 /5；[108] 现任主C 就是这个流派的最终主C
            from agent.sim.flows import flow_rank
            shop_names = [n for n, t in zip(s["names"], s["tt"]) if t == 3 and n]
            fr = flow_rank(board_names | hand_names, shop_names)
            if fr:
                b = fr[0]
                glob[103] = len(b["have"]) / len(b["pieces"])
                glob[104] = 1.0 if b["buy_completes"] else 0.0
                glob[105] = len(b["in_shop"]) / 5
                glob[106] = len(b["missing"]) / 5
                glob[107] = sum(1 for r in fr if len(r["have"]) * 2 >= len(r["pieces"])) / 5
                glob[108] = 1.0 if ctx.get("carry") and ctx.get("carry") == b["final"] else 0.0
        if ECO_CORE_FEAT and p.get("kind") == "eco":
            for k_, v_ in eco_core_glob((p.get("raw") or {}).get("missing"), lvl).items():
                glob[k_] = v_

        def hp(name, level):
            q = self.quality(name)
            return M.shop_hit_prob(q, level) if (q and level and name in self.hero_ids) else 0.0

        # ---- 1. 对象汇总事实
        for i, st in enumerate(states):
            nm = st["name"]
            if not nm:
                continue
            is_hero = G.resolve(nm) in self.hero_ids
            tt = st["tt"]
            if tt in (3, tt_opt, 6):                              # 商店 / 选项 / 主体：是否已有同名、差一张觉醒
                num[i, 28] = 1.0 if owned.get(nm, 0) >= 1 else 0.0
                num[i, 29] = 1.0 if board_awk.get(nm) == 1 else 0.0
            num[i, 30] = 1.0 if (nm in route_mem and owned.get(nm, 0) == 0) else 0.0
            num[i, 31] = 1.0 if nm in missing else 0.0
            num[i, 32] = 1.0 if (nm in eng_parts and owned.get(nm, 0) > 0) else 0.0
            if is_hero and lvl:
                num[i, 33] = hp(nm, lvl)
            num[i, 34] = 1.0 if (tt == 1 and s.get("board_src") == "names") else 0.0
            num[i, 35] = 1.0 if (tt == 2 and is_hero and nm not in board_names) else 0.0

        # ---- 1. 全局汇总事实
        heroes_hand = [n for n in hand_names if G.resolve(n) in self.hero_ids and n not in board_names]
        n_wait = sum(1 for i in range(keep) if s["tt"][i] == 2 and s["names"][i] in heroes_hand)
        pop, cap = ctx.get("pop"), ctx.get("pop_cap")
        glob[64] = n_wait / 4
        if isinstance(pop, int) and isinstance(cap, int):
            glob[65] = (cap - pop) / 4
            glob[66] = (n_wait - (cap - pop)) / 4
            glob[67] = 1.0
            glob[68] = 1.0 if n_wait > cap - pop else 0.0
        lc = ctx.get("level_cost")
        if isinstance(lc, int) and lc > 0 and gold is not None:
            glob[69] = (gold - lc) / 20
            glob[70] = 1.0 if gold >= lc else 0.0
        fr = ctx.get("free_refresh") or 0
        glob[71] = 1.0 if (fr > 0 or (gold is not None and gold >= 2)) else 0.0
        shop = [(i, st) for i, st in enumerate(states) if st["tt"] == 3]
        glob[72] = len(shop) / 5
        glob[73] = sum(owned.get(st["name"], 0) >= 1 for _, st in shop) / 4
        glob[74] = sum(board_awk.get(st["name"]) == 1 for _, st in shop) / 4
        glob[75] = sum(st["name"] in route_mem for _, st in shop) / 4
        glob[76] = sum(st["name"] in missing for _, st in shop) / 4
        glob[77] = sum(num[i, 26] > 0 for i, _ in shop) / 4
        glob[78] = sum(owned.get(st["name"], 0) == 0 and G.resolve(st["name"]) in self.hero_ids for _, st in shop) / 4
        wanted = {st["name"] for st in states if st["tt"] == 1 and (st["awk_left"] is None or st["awk_left"] > 0)}
        wanted |= {n for n in route_mem if owned.get(n, 0) == 0}
        wanted |= missing
        wanted = {w for w in wanted if w in self.hero_ids}
        if lvl:
            glob[79] = 1 - float(np.prod([1 - hp(w, lvl) for w in wanted])) if wanted else 0.0
            nxt = min(lvl + 1, 6)
            glob[80] = 1 - float(np.prod([1 - hp(w, nxt) for w in wanted])) if wanted else 0.0
        glob[81] = len(wanted) / 8
        bstates = [st for st in states if st["tt"] == 1]
        glob[82] = 1.0 if s.get("board_src") == "names" else 0.0
        glob[83] = (sum(st["lv"] is not None for st in bstates) / len(bstates)) if bstates else 0.0
        if eng:
            glob[84] = min(eng["score"], 10) / 10
            glob[85] = float(eng["started"])
            glob[86] = float(eng["complete"])
        glob[87] = len(missing) / 5
        cst = next((st for st in bstates if st["is_carry"]), None)
        if cst:
            glob[88] = 1.0
            if cst["lv"] is not None:
                glob[89] = float(np.log1p(cst["lv"]) / 5)
            glob[90] = 1.0 if cst["awk_left"] == 0 else 0.0
        glob[91] = owned.get(carry, 0) / 3 if carry else 0.0
        glob[92] = 1.0 if carry and carry in hand_names and carry not in board_names else 0.0
        mx = max([st["lv"] for st in bstates if st["lv"] is not None and not st["is_carry"]] or [0])
        glob[93] = float(np.log1p(mx) / 5)

        # ---- 3. 条件边：牌 -> 英雄
        C_COND = E2
        # eco（2026-10-09）：缺的配套（tt="wanted"）也参与条件边 / 手法槽位（停级搜牌的 support 槽）；旧题型没有 wanted 对象，结果不变
        tt_want = meta["tok_types"].index("wanted") if "wanted" in meta["tok_types"] else -99
        hero_objs = [i for i, st in enumerate(states) if st["name"] and G.resolve(st["name"]) in self.hero_ids
                     and st["tt"] in (1, 2, tt_opt, tt_want)]
        card_objs = [i for i, st in enumerate(states) if st["name"] and G.resolve(st["name"]) not in self.hero_ids
                     and st["tt"] in (2, 3, 6, tt_opt)]
        for ci in card_objs:
            nm = states[ci]["name"]
            txt = self.text(nm)
            nid = G.resolve(nm)
            ntype = (G.node.get(nid) or {}).get("type") or ((self.cards.get(nm) or {}).get("category") or "")
            awaken_card = "觉醒进度+1" in txt
            transform = ("变为" in txt and "英雄" in txt and ("指定" in txt or "解冻" in txt)) or nm == "霜寒进化"
            remove = "移除" in txt and "指定" in txt
            lvl_card = (nid in self.gives_level) or bool(re.search(r"等级\+\d", txt))
            is_equip = ntype in ("equip", "装备牌")
            tier_m = re.search(r"(\d)\s*[-~～]\s*(\d)\s*阶", txt)
            for hi in hero_objs:
                h = states[hi]
                awk = True if (h["hand_awk"] or h["awk_left"] == 0) else (None if h["awk_left"] is None else False)
                chans = []
                if awaken_card:
                    chans.append("awaken_wasted" if awk else "awaken_ok" if awk is False else None)
                if transform:
                    bad = h["is_carry"] or awk or (h["lv"] is not None and h["lv"] >= 20)
                    chans.append("transform_bad" if bad else "transform_ok")
                if remove:
                    chans.append("remove_bad" if (h["is_carry"] or awk) else "remove_ok")
                if lvl_card:
                    chans.append("level_to_carry" if h["is_carry"] else "level_to_other")
                if is_equip and h["eq"] is not None and h["eq"] >= 3:
                    chans.append("equip_to_full")
                if tier_m:
                    q = self.quality(h["name"])
                    if q and not (int(tier_m.group(1)) <= q <= int(tier_m.group(2))):
                        chans.append("tier_illegal")
                for c in chans:
                    if c:
                        k = C_COND + COND_CH.index(c)
                        rel += [(ci, hi, k), (hi, ci, k)]
                        relv += [1.0, 1.0]

        # ---- 2. 手法动态绑定
        C_SLOT = E2 + len(COND_CH)
        present_cards = {st["name"] for st in states if st["tt"] in (2, 4, 5, 6, tt_opt)} | set(ctx.get("lord_boosts") or [])
        tech_rows = []
        extra_names = []
        fmemo: dict = {}
        okeys = {hi: tuple(states[hi][k] for k in ("name", "lv", "awk_left", "eq", "row", "is_carry", "maple", "hand_awk",
                                                    "on_board", "in_hand")) for hi in hero_objs}
        for t in self.techs:
            tid = t["id"]
            roles = t.get("roles") or {}
            fills = {}
            for rname, preds in roles.items():
                row = []
                for hi in hero_objs:                          # 同一状态的对象（选项复制自场上单位）只算一次
                    k = (tid, rname, okeys[hi])
                    if k not in fmemo:
                        fmemo[k] = self.fill(preds, states[hi], ctx)
                    row.append((hi, fmemo[k]))
                fills[rname] = row
            enabler = any(self.G.resolve(c) in self.enablers[tid] or c in self.enablers[tid] for c in present_cards)
            cover = [any(f > 0 for _, f in fl) for fl in fills.values()]
            if not (enabler or (cover and all(cover))):
                continue
            pre = t.get("precondition") or []
            pre_ok = 1.0
            for pc in pre:
                m = re.match(r"has_maple_boost\((.+?)\)", pc)
                if m:
                    pre_ok = 1.0 if m.group(1) in (ctx.get("lord_boosts") or []) else 0.0
            nb = [st for st in bstates]
            lvs_nc = [st["lv"] for st in nb if st["lv"] is not None and not st["is_carry"]]
            benefit, hitp = 0.0, 0.0
            if tid == "tech:maple_level_carryover":
                mh = next((st for st in nb if st["maple"] and st["lv"] is not None), None)
                benefit = 0.5 * mh["lv"] * pre_ok if mh else 0.0
            elif tid == "tech:yizhi_transfer":
                benefit = max(lvs_nc or [0])
                hitp = 1.0 / (len(nb) - 1) if len(nb) > 1 else 0.0
            elif tid == "tech:chuancheng_spread":
                benefit = max([v for v in lvs_nc if v > 10] or [0])
            elif tid == "tech:tier_evolve":
                cand = [st["lv"] for st in nb if st["lv"] is not None and not st["is_carry"] and 1 <= self.quality(st["name"]) <= 2]
                benefit = (max(cand) + 10) if cand else 0.0
            elif tid == "tech:rift_devour":
                benefit = 2 * max(lvs_nc or [0])
            elif tid == "tech:steal_merge":
                benefit = min(10, max(lvs_nc or [0]))
            elif tid == "tech:level_swap":
                sf = [st["lv"] for st in nb if st["lv"] is not None and (G.node.get(G.resolve(st["name"])) or {}).get("faction") == "三分之地"]
                benefit = max(sf or [0])
            tv = np.zeros(NUM_DIM2, np.float32)
            tv[36] = 1.0 if enabler else 0.0
            tv[37] = pre_ok
            tv[38] = float(np.mean(cover)) if cover else 0.0
            tv[39] = float(np.log1p(benefit) / 5)
            tv[40] = hitp
            tv[41] = sum(1 for fl in fills.values() for _, f in fl if f >= 1) / 8
            tech_rows.append((tid, tv, fills))
            extra_names.append(tid)
        # 追加手法对象
        base_n = keep
        tt_l, ot_l, node_l, text_l = list(s["tt"]), list(s["ot"]), list(s["node"]), list(s["text"])
        num_rows = [num]
        for j, (tid, tv, fills) in enumerate(tech_rows):
            ti = base_n + j
            tt_l.append(tt_tech)
            ot_l.append(0)
            node_l.append(G.idx[tid] + 2)
            text_l.append(G.idx[tid] + 1)
            s["names"].append(tid)
            num_rows.append(tv[None, :])
            for rname, fl in fills.items():
                k = C_SLOT + self.slots.index((tid, rname))
                for hi, f in fl:
                    if f > 0:
                        rel += [(hi, ti, k), (ti, hi, k)]
                        relv += [f, f]
            # v1 的图谱边（手法 <-> 卡）也补上：enables / fits_role
            for oi in range(base_n):
                a = G.resolve(states[oi]["name"]) if states[oi]["name"] else None
                if not a:
                    continue
                for (x, y) in ((oi, ti), (ti, oi)):
                    na = a if x == oi else tid
                    nb_ = tid if x == oi else a
                    m = G.pair.get((na, nb_), 0)
                    c = 0
                    while m:
                        if m & 1:
                            rel.append((x, y, c))
                            relv.append(1.0)
                        m >>= 1
                        c += 1
        s["tt"] = np.array(tt_l, np.int8)
        s["ot"] = np.array(ot_l, np.int8)
        s["node"] = np.array(node_l, np.int16)
        s["text"] = np.array(text_l, np.int32)
        s["num"] = np.concatenate(num_rows).astype(np.float32)
        # 关系去重（同一 (i,j,c) 取最大值）
        d: dict = {}
        for (i, j, c), v in zip(rel, relv):
            key = (int(i), int(j), int(c))
            d[key] = max(d.get(key, 0.0), float(v))
        keys = sorted(d)
        s["rel"] = np.array(keys, np.int16).reshape(-1, 3)
        s["relv"] = np.array([d[k] for k in keys], np.float16)
        s["glob"] = glob
        s["n_tech"] = len(tech_rows)
        return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="r9m")
    ap.add_argument("--out", default="r9m_v2")
    ap.add_argument("--graph", default=str(PROJECT_DIR / "data" / "card_graph.json"))
    ap.add_argument("--v1only", action="store_true", help="只做 v1 的构造（用于分离图谱更新的影响）")
    a = ap.parse_args()
    gpath = Path(a.graph)
    BD.load_graph = lambda: json.loads(gpath.read_text(encoding="utf-8"))          # 本脚本内换图谱，不改 build_data.py
    labels_path = PROJECT_DIR / "data" / "laya" / "labels" / "r6" / f"labels_all_{a.ds}.jsonl"
    labs = {json.loads(l)["id"]: json.loads(l) for l in labels_path.open(encoding="utf-8")}
    G = BD.Graph()
    G.path = str(gpath)
    B = BD.Builder(G, None)
    v2 = None if a.v1only else V2(G)
    meta1 = {"tok_types": BD.TOK_TYPES, "n_channels": 2 * len(BD.EDGE_TYPES) + len(BD.EXTRA_CH)}
    out, snaps_cache = [], {}
    miss = collections.defaultdict(lambda: collections.Counter())
    for line in (PROJECT_DIR / "data" / "laya" / "datasets" / a.ds / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        lab = labs.get(p["id"])
        if lab is None:
            continue
        run = p.get("run") or ""
        if run not in snaps_cache:
            snaps_cache[run] = BD.replay_run(run)
        owned, lb = BD.snapshot(snaps_cache[run], float(p.get("ts") or 0))
        s = B.convert(p, lab, owned, lb)
        # 缺失率统计（场上对象）
        for k in (p["kind"], f'{p["kind"]}:{p["split"]}'):
          for i in np.where(s["tt"] == 1)[0]:
            v = s["num"][i]
            miss[k]["units"] += 1
            miss[k]["lv_known"] += int(v[1] > 0)
            miss[k]["awk_known"] += int(v[3] > 0)
            miss[k]["eq_known"] += int(v[7] > 0)
            miss[k]["cell_known"] += int(v[23] > 0)
          miss[k]["samples"] += 1
          miss[k]["board_names_only"] += int(s["board_src"] == "names")
          miss[k]["unit_levels_nonempty"] += int(bool((p.get("ctx") or {}).get("unit_levels")))
          miss[k]["maple_known"] += int((p.get("ctx") or {}).get("maple") is not None)
        if v2 is not None:
            s = v2.augment(s, p, meta1)
        out.append(s)
    od = OUT_ROOT / a.out
    od.mkdir(parents=True, exist_ok=True)
    with (od / "samples.pkl").open("wb") as f:
        pickle.dump(out, f)
    np.save(od / "node_feat.npy", G.node_feat())
    (od / "text_keys.json").write_text(json.dumps(B.extra_texts, ensure_ascii=False), encoding="utf-8")
    g = json.loads(gpath.read_text(encoding="utf-8"))
    enc = Encoder("bge")
    ne = enc.encode([node_text(n) for n in g["nodes"]])
    ex = enc.encode(B.extra_texts) if B.extra_texts else np.zeros((0, ne.shape[1]), np.float32)
    np.save(od / "text_table_bge.npy", np.concatenate([np.zeros((1, ne.shape[1]), np.float32), ne, ex]).astype(np.float32))
    n_ch1 = 2 * len(BD.EDGE_TYPES) + len(BD.EXTRA_CH)
    meta = json.loads((OUT_ROOT / a.ds / "meta.json").read_text(encoding="utf-8"))
    meta.update({"ds": a.out, "points_ds": a.ds, "graph": str(gpath.relative_to(PROJECT_DIR)),
                 "graph_md5": hashlib.md5(gpath.read_bytes()).hexdigest(), "n": len(out),
                 "objects_per_sample": {"mean": float(np.mean([len(s["tt"]) for s in out])),
                                        "max": int(max(len(s["tt"]) for s in out))},
                 "rel_per_sample_mean": float(np.mean([len(s["rel"]) for s in out])),
                 "missing_stats": {k: dict(v) for k, v in miss.items()}})
    if v2 is not None:
        meta.update({"version": "v2", "num_dim": NUM_DIM2, "glob_dim": GLOB_DIM2,
                     "n_channels": n_ch1 + len(COND_CH) + len(v2.slots),
                     "cond_channels": COND_CH, "slot_channels": [f"{t}/{r}" for t, r in v2.slots],
                     "channel_ranges": {"v1": [0, n_ch1], "cond": [n_ch1, n_ch1 + len(COND_CH)],
                                        "slot": [n_ch1 + len(COND_CH), n_ch1 + len(COND_CH) + len(v2.slots)]},
                     "fact_glob": FACT_GLOB, "fact_num": FACT_NUM, "tech_num": TECH_NUM,
                     "n_tech_mean": float(np.mean([s["n_tech"] for s in out])),
                     "cond_rel_mean": float(np.mean([int(((s["rel"][:, 2] >= n_ch1) & (s["rel"][:, 2] < n_ch1 + len(COND_CH))).sum()) for s in out])),
                     "slot_rel_mean": float(np.mean([int((s["rel"][:, 2] >= n_ch1 + len(COND_CH)).sum()) for s in out]))})
    else:
        meta.update({"version": "v1only"})
    (od / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in meta if k in ("ds", "graph_md5", "n", "objects_per_sample", "rel_per_sample_mean",
                                                        "n_channels", "n_tech_mean", "cond_rel_mean", "slot_rel_mean")},
                     ensure_ascii=False))
    for k, v in miss.items():
        u = max(1, v["units"])
        print(f"{k:10s} n={v['samples']:5d} names_only={v['board_names_only'] / v['samples']:.2f} "
              f"lv_known={v['lv_known'] / u:.2f} awk_known={v['awk_known'] / u:.2f} eq_known={v['eq_known'] / u:.2f} "
              f"cell_known={v['cell_known'] / u:.2f} maple_known={v['maple_known'] / v['samples']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
