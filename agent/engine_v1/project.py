"""按回合推演（期望值、确定性）。

一步 = 当前回合准备阶段结束 → 战斗 → 结算 → 收入/经验/自动升级 → 下一回合 round_start + 整备 结算完。
（回放对比就是拿"第 R 回合最后一个准备帧"推一步，和"第 R+1 回合第一批准备帧"的读数比。）

战斗内顺序（写清假设，参数都在 PARAMS）：
  1. 开团(open) 效果，逐单位结算；每次结算发出 open 事件 → react:open 反应器（熊灵 +20% 攻速、韩信 +2 …）
     开团里"摧毁图腾"之类 remove 在这里生效（所以刘邦摧毁图腾会让熊灵的单位数差变小 = 反协同能算出来）
  2. 战斗开始时(battle_start) 效果（含阿科米亚核心按关键词规则分临时等级：日落海英雄平分，单人≤核心 50%）
  3. 战力 → 胜率 → 期望击杀 / 每个单位阵亡概率（简单模型，见 battle_model）
  4. 普攻计数 attack_n：每个单位普攻次数 = 基础攻速 ×(1+本场额外攻速) × 战斗时长 × 存活比例（动态，不查静态表）
     技能 skill_cast：次数 = (初始法力 + 普攻次数×每次回蓝) / 法力上限
  5. kill（按击杀分配）/ death、sacrifice（按阵亡概率）/ react:sacrifice / win（×胜率）/ lose（×败率）
  6. 临时转永久（明世隐类，目标=最近友军，额度=卡面值 + 登场累加的 conv_cap），然后清临时等级、本场攻速
回合间：收入（meta.energy_income）、经验（meta.lord_xp_gain）、经验满自动升级（7/13/17/23/27），升级事件 → "棋手升级时" 类效果
下一回合：round_start（装备 / 天赋"每回合开始"）→ 整备 prepare（每个单位，觉醒"触发 2 次"按 repeat）→ react:prepare
触发器：trigger 效果按 trigger_kind 让目标的 prepare/deploy/open/sacrifice 条目额外结算一次（曹操 整备→登场、
  露娜 登场→整备、雅典娜 开团→整备 都是这条路径），递归深度 ≤3、同一单位同一事件每回合 ≤6 次，防止互相触发死循环。
古币：coin 效果 = 对目标使用 N 张古币（或精致古币），每张 = 古币卡面等级 + 场上/天赋的 coin_amp 放大，并发 react:coin。
"""
from __future__ import annotations

import re

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import json

from . import board as B
from .effects import db, Effect
from .state import Position, Unit, make_unit

ROOT = Path(__file__).resolve().parents[2]

PARAMS = {
    "copy_power": 0.5,           # 分身镜类复制体给输出位的战力加成（承伤 350%，活得短；估计值）
    "mana_ratio_cap": 2.0,       # 法力上限降低带来的施法倍数上限（估计值）
    "team_gamma": 1.8,           # 全队战力聚合指数（>1 偏向最强单位；估计值，范围见 PARAM_RANGES）
    "yizhi_p3": 0.386,           # 一个 3 阶效果锦囊开出遗志的概率（17/44 实测）
    "tactic_k": 3.0,             # 嬴政"每次使用战术牌会提升该加成"每张提升多少法攻（卡面没写数，估计值）
    "skill_share": 0.7,          # 输出位伤害里技能占的比例（估计值；普攻部分不受施法次数影响）
    "cast_ratio_cap": 3.0,       # 施法次数相对强度表的倍数上限
    "ctrl_rate": 0.12,           # 引导技能每秒被控制打断的概率（估计值；有"免疫控制"装备时为 0）
    "carry_tau": 14.0,           # 主C / 输出位平均能活多久（秒，指数分布均值；估计值）
    "cast_ref_s": 4.0,           # 强度表默认的"典型出手时间"（秒）
    "skill_alpha": 0.7,          # 技能强度（dps 相对中位数）折进战力的指数（估计值）
    "bind_power": 0.2,           # 棋手技能绑定（枫叶舞护盾+转移）给被绑的人的战力加成（估计值）
    "invest_frac": 0.5,          # 日常投入 = 实测主C 涨级 × 这个比例（实测里含卡面效果，推演已算的部分要扣掉；估计值）
    "fill_level_frac": 0.5,      # 补位替补的等级 = 场上非主C 英雄等级中位数 × 这个比例（估计值）
    "death_cost": 0.5,           # 阵亡单位平均少打的那部分输出（中途阵亡；估计值）
    "win_iter": 2,               # 胜率 ↔ 阵亡 迭代次数
    "revive_power": 0.35,        # 复生：战力 ×1.35（多一条命，估计值）
    "revive_death_mult": 0.35,   # 复生：最终阵亡概率 ×0.35（估计值）
    "buys_per_round": 3.0,       # 每回合在商店买几张牌（期望，待按日志校准）
    "battle_seconds": 25.0,      # 一场战斗平均时长（秒）
    "aps_default": 0.7,          # 卡库没有攻速时的每秒普攻次数
    "mana_per_attack": 10.0,     # 每次普攻回蓝（2026-10-14 确认：法力上限最低 20 = 两下普攻放一次技能）
    "mana_floor": 20.0,          # 法力值上限再怎么降也不低于 20（2026-10-14）
    "mana_hit": 20.0,            # 一场战斗里被打 / 其他来源回的蓝（估计值）
    "win_k": 2.0,                # 胜率 = P^k / (P^k + E^k)
    "kill_base": 0.3,            # 输了也能杀掉的敌方比例
    "death_lose": 0.85,          # 输时我方单位阵亡概率（再乘前后排系数）
    "death_win": 0.25,           # 赢时
    "front_mult": 1.4, "back_mult": 0.6,
    "output_kill_w": 2.0, "carry_kill_w": 2.5,   # 击杀分配权重（输出位 / 主C 额外）
    "merge_gain": 1.0,           # 合成一张同名 +级（商店牌等级按 1 算）
    "awaken_need": 3,            # 本体 + 3 张同名 = 觉醒
    "enemy_summons_from": 6,     # 敌方从第几回合起平均多 1 个召唤单位
    "trigger_depth": 3, "trigger_cap": 6,
    "game_end_round": 16,        # "到对局结束"视野的终点
}
# 估计值参数的合理范围（敏感性分析：每道题在范围内抽样多次，看结论稳不稳）。
# 范围是"我说不准的程度"，不是拟合结果；实测校准后收窄。
PARAM_RANGES = {
    "copy_power": (0.25, 0.8),
    "mana_ratio_cap": (1.4, 2.6),
    "revive_power": (0.15, 0.6),
    "revive_death_mult": (0.15, 0.6),
    "death_cost": (0.3, 0.7),
    "fill_level_frac": (0.3, 0.8),
    "invest_frac": (0.3, 0.8),
    "bind_power": (0.05, 0.4),
    "skill_alpha": (0.4, 1.0),
    "carry_tau": (8.0, 22.0),
    "ctrl_rate": (0.03, 0.25),
    "skill_share": (0.5, 0.85),
    "tactic_k": (1.0, 7.0),
    "yizhi_p3": (0.25, 0.54),
    "team_gamma": (1.3, 2.5),
    "battle_seconds": (18.0, 32.0),
    "mana_per_attack": (8.0, 12.0),
    "mana_hit": (0.0, 40.0),
    "win_k": (1.5, 3.0),
    "kill_base": (0.15, 0.45),
    "death_lose": (0.65, 0.95),
    "death_win": (0.12, 0.4),
    "front_mult": (1.15, 1.7),
    "output_kill_w": (1.5, 3.0),
    "carry_kill_w": (1.8, 3.5),
    "buys_per_round": (2.0, 4.0),
}
# 主C 每回合的实际涨级（第 548~627 局，同一主C 相邻两回合准备阶段读数之差的中位数）。
# 推演只模拟卡面效果；买同名合成、古币、加等级牌、枫叶舞这些"日常投入"不在卡面里，
# 但它们**跟着主C 走** —— 这正是换主C 的意义：以后的投入落在谁身上、每一级值多少战力。
CARRY_INVEST = {1: 0, 2: 1, 3: 3, 4: 4, 5: 6, 6: 6, 7: 7, 8: 9, 9: 9, 10: 10, 11: 16, 12: 16, 13: 21, 14: 25, 15: 25, 16: 25}
_EQUIP_NAME = __import__("re").compile(r"装备([^，。、\s]+)")
_RAND_N = __import__("re").compile(r"随机[^0-9]{0,4}(\d+)名")
QUALITY_POWER = {0: 0.6, 1: 1.0, 2: 1.25, 3: 1.55, 4: 1.95, 5: 2.4}
OUTPUT_ROLES = {"carry_mag", "carry_phys", "assassin", "bruiser"}


# ------------------------------------------------------------------ 静态数据
@lru_cache(maxsize=1)
def _roles() -> dict:
    try:
        return json.loads((ROOT / "data" / "hero_roles.json").read_text(encoding="utf-8")).get("heroes") or {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def enemy_level_curve() -> dict:
    """敌方每回合"全队等级和"中位数（data/enemy_levels.json 全部对局，读数为 0 的帧不算）。读不到用默认表。"""
    default = {1: 3, 2: 6, 3: 10, 4: 16, 5: 24, 6: 34, 7: 50, 8: 65, 9: 85, 10: 105, 11: 130, 12: 160}
    try:
        d = json.loads((ROOT / "data" / "enemy_levels.json").read_text(encoding="utf-8"))
    except Exception:
        return default
    acc: dict = {}
    for run in d.values():
        for r, v in (run or {}).items():
            if isinstance(v, dict) and (v.get("enemy_sum") or 0) > 0:
                acc.setdefault(int(r), []).append(v["enemy_sum"])
    out = dict(default)
    for r, xs in acc.items():
        if len(xs) >= 5:
            xs.sort()
            out[r] = xs[len(xs) // 2]
    # 单调化
    last = 0
    for r in sorted(out):
        out[r] = max(out[r], last)
        last = out[r]
    return out


def is_output(u: Unit) -> bool:
    if u.kind == "summon":
        return u.can_carry
    roles = set((_roles().get(u.name) or {}).get("combat_role") or [])
    return bool(roles & OUTPUT_ROLES) if roles else True


def _props(name: str) -> dict:
    return (db().raw(name) or {}).get("properties") or {}


def base_aps(u: Unit) -> float:
    a = _props(u.name).get("attackSpeed")
    return (a / 10000.0) if a else PARAMS["aps_default"]


@lru_cache(maxsize=1)
def _dps_table() -> tuple:
    """data/hero_strength.json：按技能原文算的各等级 dps（15/40/70/100/150 级）+ 同等级全英雄中位数。"""
    try:
        H = json.loads((ROOT / "data" / "hero_strength.json").read_text(encoding="utf-8")).get("heroes") or {}
    except Exception:
        return {}, {}
    knots = ("15", "40", "70", "100", "150")
    tab = {n: [h["levels"][k]["dps"] for k in knots] for n, h in H.items()
           if isinstance(h, dict) and all(k in (h.get("levels") or {}) for k in knots)}
    import statistics
    med = [statistics.median(v[i] for v in tab.values()) for i in range(len(knots))] if tab else []
    return tab, med


def skill_ratio(name: str, L: float) -> float | None:
    """技能强度：这个英雄在等级 L 的 dps / 同等级全英雄中位数（对数插值）。没有数据返回 None。"""
    tab, med = _dps_table()
    v = tab.get(name)
    if not v:
        return None
    xs = (15.0, 40.0, 70.0, 100.0, 150.0)
    L = min(max(L, xs[0]), xs[-1])
    for i in range(len(xs) - 1):
        if L <= xs[i + 1]:
            t = (math.log(L) - math.log(xs[i])) / (math.log(xs[i + 1]) - math.log(xs[i]))
            a = v[i] / med[i]
            b = v[i + 1] / med[i + 1]
            return a + (b - a) * t
    return v[-1] / med[-1]


_CHANNEL_RE = re.compile(r"在(\d+(?:\.\d+)?)秒内")
_REGEN_RE = re.compile(r"每秒(?:恢复|回复)(\d+)(?:点)?法力")
_INIT_RE = re.compile(r"(?:战斗开始|开战)[^。]{0,12}(?:(?:获得|恢复)(\d+)点?法力|法力值?\+(\d+))")
_CAP_RE = re.compile(r"法力值上限-(\d+)")
_TACTIC_RE = re.compile(r"获得\d*张([^，。]{0,6}战术)")


def first_cast_time(u: Unit) -> float:
    return cast_timing(u)[0]


def cast_timing(u: Unit, base: bool = False) -> tuple:
    """(第一次放完技能的时间, 之后每次的周期) 秒。base=True：按卡面原始法力、不算任何降法力 / 回蓝 / 攻速加成
    （强度表 hero_strength 就是这么算的）。"""
    pr = _props(u.name)
    mana = pr.get("energy") or 0
    if not mana:
        return 0.0, 1e9
    r = db().raw(u.name) or {}
    sk = " ".join(x.get("desc", "") for x in r.get("skills") or [])
    m = _CHANNEL_RE.search(sk)
    ch = float(m.group(1)) if m else 0.0
    if base:
        rate = base_aps(u) * PARAMS["mana_per_attack"]
        return max(0.0, mana - (pr.get("initEnergy") or 0)) / max(0.1, rate) + ch, mana / max(0.1, rate) + ch
    t = _first_cast(u, ch)
    return t


def _first_cast(u: Unit, ch: float) -> tuple:
    """第一次技能放完的时间（秒）：(法力上限 − 降低 − 初始 − 开战回蓝) / (每秒普攻×每下回蓝 + 每秒回蓝) + 技能持续时间。
    法力上限最低 20（2026-10-14）。"""
    pr = _props(u.name)
    mana = pr.get("energy") or 0
    if not mana:
        return 0.0
    init = pr.get("initEnergy") or 0
    cut = 0.5 * u.mana_red + u.mana_cut       # 战斗中阵亡降法力（平均半场）+ 本回合推演战术类
    regen = 0.0
    for it in u.equips:
        r = db().raw(it) or {}
        txt = (r.get("skill") or "") + (r.get("desc") or "")
        if "仅" in txt and "穿戴时生效" in txt and u.name not in txt:
            continue
        m = _REGEN_RE.search(txt)
        if m:
            regen += float(m.group(1))
        m = _INIT_RE.search(txt)
        if m:
            init += float(m.group(1) or m.group(2))
        m = _CAP_RE.search(txt)
        if m and "回合开始" not in txt:
            cut += float(m.group(1))
    cap = max(PARAMS["mana_floor"], mana - cut) if mana > PARAMS["mana_floor"] else mana
    need = max(0.0, cap - init)
    rate = base_aps(u) * (1 + max(0.0, u.aspd)) * PARAMS["mana_per_attack"] + regen + PARAMS["mana_hit"] / max(1.0, PARAMS["battle_seconds"])
    return need / max(0.1, rate) + ch, cap / max(0.1, rate) + ch


def expected_casts(t1: float, cyc: float, survive: bool, ctrl_p: float = 0.0) -> float:
    """一场战斗（battle_seconds）里期望放出几次技能：第 k 次在 t1+(k-1)·cyc 放完，要活到那时（存活按指数分布，均值 carry_tau）。"""
    T, tau = PARAMS["battle_seconds"], PARAMS["carry_tau"]
    n, t = 0.0, t1
    while t <= T and cyc > 0:
        n += (math.exp(-t / tau) if survive else 1.0) * (1 - ctrl_p)
        t += cyc
    return n


def cast_factor(u: Unit) -> float:
    """技能次数系数 = 实际期望施法次数 / 强度表假设的施法次数（卡面原始法力、能活满整场）。
    同时体现三件事：① 活不到放技能（出手慢的嬴政吃亏）；② 降法力 / 回蓝 / 攻速 → 一场放更多次（战术牌循环凑齐后嬴政远超强度表）；
    ③ 引导技能被控制打断（有"免疫控制"装备则不打断）。按 skill_share 折进战力（普攻部分不受影响）。"""
    t1, cyc = cast_timing(u)
    b1, bcyc = cast_timing(u, base=True)
    base = expected_casts(b1, bcyc, survive=False)
    if base <= 0:
        base = 0.5                                  # 强度表里一次都放不出来的，也按半次算，避免除零
    ctrl_p = 0.0
    r = db().raw(u.name) or {}
    m = _CHANNEL_RE.search(" ".join(x.get("desc", "") for x in r.get("skills") or []))
    if m and not any("免疫控制" in ((db().raw(it) or {}).get("skill") or (db().raw(it) or {}).get("desc") or "") for it in u.equips):
        ctrl_p = 1 - math.exp(-float(m.group(1)) * PARAMS["ctrl_rate"])
    ratio = expected_casts(t1, cyc, survive=True, ctrl_p=ctrl_p) / base
    sh = PARAMS["skill_share"]
    return (1 - sh) + sh * min(PARAMS["cast_ratio_cap"], ratio)


def _cast_factor_old(u: Unit) -> float:
    """能不能活到把技能放出来：存活时间按指数分布（均值 carry_tau 秒），P(活到第一次放完) 相对"典型英雄 4 秒出手"的比值。
    强度表（hero_strength）假设英雄能活满 20 秒，对出手慢的（嬴政 8 秒攒蓝 + 5 秒引导）高估；回蓝装备 / 降法力 / 攻速对他们价值大。"""
    t = first_cast_time(u)
    tau = PARAMS["carry_tau"]
    f = math.exp(-t / tau) / math.exp(-PARAMS["cast_ref_s"] / tau)
    # 引导型技能（"在 N 秒内…"）会被控制打断（2026-10-14：嬴政大招很难放出来；攻略：必带炽热支配者"免疫控制"）
    r = db().raw(u.name) or {}
    m = _CHANNEL_RE.search(" ".join(x.get("desc", "") for x in r.get("skills") or []))
    if m and not any("免疫控制" in ((db().raw(it) or {}).get("skill") or (db().raw(it) or {}).get("desc") or "")
                     for it in u.equips):
        f *= math.exp(-float(m.group(1)) * PARAMS["ctrl_rate"])
    return f


def agg(powers) -> float:
    """全队战力 = (Σ p^γ)^(1/γ)，γ>1：最强的一两个人决定胜负（单核打法；2026-10-14 推演发现按相加算时，
    "把投入摊给一个 2 级新人"会比"堆在 84 级主C 身上"更划算 —— 和游戏实际相反）。γ=1 即相加。"""
    g = PARAMS["team_gamma"]
    ps = [max(0.0, x) for x in powers]
    return (sum(x ** g for x in ps) ** (1.0 / g)) if ps else 1e-6


def unit_power(u: Unit, lvl: float | None = None) -> float:
    """单位战力（相对量）：品阶系数 ×(1+等级/8)× 10/40/100 级技能质变 × 装备 × 觉醒 × 攻速（输出位）。
    输出位有技能 dps 数据时：品阶系数和 10/40/100 级质变换成"技能强度"（同等级 dps / 全英雄中位数）^skill_alpha × 1.5
    —— 嬴政、马可波罗这类技能本身强的英雄每一级值得更多（2026-10-14 确认嬴政的技能非常强）。"""
    L = (u.level + u.temp) if lvl is None else lvl
    q = u.quality if u.kind == "hero" or u.can_carry else 0
    sr = skill_ratio(u.name, L) if (u.kind == "hero" and is_output(u)) else None
    if sr is not None:
        sr *= cast_factor(u)
        p = 1.5 * max(0.3, sr) ** PARAMS["skill_alpha"] * (1 + max(0.0, L) / 8.0)
    else:
        p = QUALITY_POWER.get(q, 1.0) * (1 + max(0.0, L) / 8.0)
        for th, m in ((10, 1.15), (40, 1.25), (100, 1.3)):
            if L >= th:
                p *= m
    p *= 1 + 0.15 * sum(1 for it in u.equips[:B.MAX_EQUIPS] if equip_fits(it, u))
    if is_output(u) and any(copy_equip(it) for it in u.equips[:B.MAX_EQUIPS]):
        p *= 1 + PARAMS.get("copy_power", 0.5)   # 分身镜类：战斗开始复制一个自己（承伤高、活得短）
    if u.atk and is_output(u):
        p *= 1 + u.atk / 150.0
    if u.is_awakened:
        p *= 1.5
    if is_output(u):
        p *= 1 + max(0.0, u.aspd) * 0.8
    if False and is_output(u) and u.mana_red > 0:   # 已并进 cast_factor（cast_timing 里的 cut），保留代码作对照
        mana = _props(u.name).get("energy") or 0
        if mana:
            red = 0.5 * u.mana_red               # 阵亡发生在战斗中途：平均只有一半时间享受降低
            ratio = min(PARAMS["mana_ratio_cap"], mana / max(PARAMS["mana_floor"], mana - red)) if mana > PARAMS["mana_floor"] else 1.0
            p *= 1 + 0.5 * (ratio - 1)
    return p


GRANT_RE = re.compile(r"(自身及)?(左右|周围|相邻)(\d)格(?:内)?的(?:己方)?(?:英雄|单位)获得(牺牲|整备|开团|登场)")


def granted_keywords(p: Position) -> dict:
    """卡面"自身及左右 1 格的英雄获得牺牲"这类：按站位算出谁额外拥有哪个关键词。左右 = 同一行。"""
    out: dict = {}
    for g in p.on_board():
        r = db().raw(g.name) or {}
        txt = (r.get("skill") or "") + (r.get("desc") or "")
        for m in GRANT_RE.finditer(txt):
            n, kw = int(m.group(3)), m.group(4)
            for u in p.on_board():
                if u is g:
                    if m.group(1):
                        out.setdefault(u.name, set()).add(kw)
                    continue
                if not (u.cell and g.cell):
                    continue
                if m.group(2) == "左右" and B.row_of(u.cell) != B.row_of(g.cell):
                    continue
                if B.hex_dist(u.cell, g.cell) <= n:
                    out.setdefault(u.name, set()).add(kw)
    return out


def hero_dmg(name: str) -> str:
    c = db().unit(name) or db().get(name)
    r = db().records.get(c.key) if c else None
    d = str((r or {}).get("dmg") or "")
    if not d:                                   # 效果表没写：按卡面技能文本 / 面板判（hero_traits.dmg_type，只读）
        try:
            from agent.decision.hero_traits import dmg_type
            d = dmg_type(name)
        except Exception:                       # noqa: BLE001
            d = ""
        d = {"hybrid": ""}.get(d, d)
    return d


def copy_equip(item: str) -> bool:
    """装备卡面"召唤…复制体"：开战复制穿戴者。"""
    c = db().get(item) if item and item != "?" else None
    r = db().raw(item) or {}
    txt = (r.get("skill") or "") + (r.get("desc") or "") + " ".join(e.raw or "" for e in (c.effects if c else []))
    return bool(c and c.category == "装备牌" and re.search(r"召唤.{0,12}复制体", txt))


def equip_fits(item: str, u: Unit) -> bool:
    """卡面伤害类型硬错配（物理装给法术英雄 / 反之）不算战力；未知按适配。"""
    if item == "?":
        return True
    c = db().get(item)
    d = str((c.equip or {}).get("dmg") or "") if c else ""
    h = hero_dmg(u.name)
    return not (d in ("phys", "mag") and h in ("phys", "mag") and d != h)


def enemy_power(round_no: int) -> tuple[float, int]:
    round_no = max(1, min(int(round_no or 1), 20))      # 读数异常的回合（实测有 1313）钳住
    lv_t = B.meta.level_target(round_no)
    n = B.pop_cap(lv_t) + (1 if round_no >= PARAMS["enemy_summons_from"] else 0)
    curve = enemy_level_curve()
    tot = curve.get(round_no) or curve[max(curve)] * (1.25 ** (round_no - max(curve)))
    avg_lv = tot / max(1, n)
    q = min(5, max(1, lv_t - 1))
    one = QUALITY_POWER[q] * (1 + avg_lv / 8.0)
    for th, m in ((10, 1.15), (40, 1.25), (100, 1.3)):
        if avg_lv >= th:
            one *= m
    return agg([one * 1.15] * n), n   # 1.15 ≈ 敌方平均装备；和我方一样按 agg 聚合


# ------------------------------------------------------------------ 推演记录
@dataclass
class RoundRecord:
    round: int
    win_p: float
    power: float
    enemy: float
    kills: float
    levels: dict            # 名字 → 永久等级（下一回合整备之后）
    energy: float
    lord_level: int
    unmodeled: int


@dataclass
class Projection:
    start: Position
    end: Position
    rounds: list = field(default_factory=list)

    def level_of(self, name: str, k: int = -1) -> float:
        if not self.rounds:
            u = self.start.unit(name)
            return u.level if u else 0.0
        return self.rounds[k].levels.get(name, 0.0)


# ------------------------------------------------------------------ 推演器
class Sim:
    """对一个 Position 做原地推演（调用方先 clone）。"""

    def __init__(self, pos: Position, params: dict | None = None):
        self.p = pos
        self.P = dict(PARAMS, **(params or {}))
        self.D = db()
        self.fires: dict = {}            # (单位, 事件) → 本回合已结算次数
        self.removed_totems = 0
        self.mana_acc: dict = {}         # 本场"法力值上限 -N"累计
        self.revived: set = set()        # 本场拥有复生的单位
        self.granted: dict = {}          # 站位授予的关键词
        self.battle = {}                 # 本场战斗的中间量
        pos.meta.setdefault("unmodeled", 0)

    # ---------------------------------------------- 基础查询
    def carry(self) -> Unit | None:
        return pick_carry(self.p)

    def card_fx(self, u: Unit):
        c = u.fx
        if not c:
            return []
        out = list(c.fx(u.is_awakened))
        return out

    def _double(self, u: Unit) -> float:
        """觉醒且卡面没有单独的觉醒文本 → 词条翻倍（关键词规则）。"""
        c = u.fx
        return 2.0 if (u.is_awakened and c and not c.awaken_effects and c.unit_kind == "hero") else 1.0

    def ctx(self, src: Unit | None, extra: dict | None = None) -> dict:
        p = self.p
        board = p.on_board()
        heroes = p.heroes()
        totems = sum(1 for u in board if "图腾" in u.name) - self.removed_totems
        c = {
            "round": p.round, "lord_level": p.lord_level, "energy": p.energy, "core_level": p.core,
            "ally_units": max(0, len(board) - self.removed_totems), "enemy_units": self.battle.get("enemy_n", 0),
            "totems_on_board": max(0, totems), "totems_count": max(0, totems),
            "factions_on_board": len({u.faction for u in heroes if u.faction and u.faction != "无阵营"}),
            "awakened_heroes_on_board": sum(1 for u in heroes if u.is_awakened),
            "kills_last_round": p.meta.get("kills_last_round", 0.0),
            "level_up_1": max(0, B.XP_NEED.get(p.lord_level, 0) - p.xp),
            "flash_kills_last_round": p.meta.get("flash_kills_last_round", 0.0),
            "tactic_cards_used": p.meta.get("tactic_total", 0), "k": self.P["tactic_k"],
            "faction_count": lambda f: sum(1 for u in heroes if u.faction == _nf(f)),
            "sum_perm_level": lambda f: sum(u.level for u in heroes if u.faction == _nf(f)),
        }
        if src is not None:
            c.update({"level": src.level, "self_level": src.level, "equips_worn": len(src.equips),
                      "awaken_progress": src.awaken, "tier": src.quality,
                      "attacks": self.battle.get("attacks", {}).get(id(src), 0.0)})
        if extra:
            c.update(extra)
        return c

    # ---------------------------------------------- 目标解析
    def targets(self, e: Effect, src: Unit | None, chosen: Unit | None = None) -> list:
        p = self.p
        who = e.who
        fac = _nf(e.tfaction)
        pool = [u for u in p.heroes() if (not fac or fac in ("无阵营",) or u.faction == fac)]
        n = e.tcount
        if who == "self":
            return [(src, 1.0)] if src is not None else []
        if who == "carry":
            c = self.carry()
            return [(c, 1.0)] if c else []
        if who == "chosen":
            t = chosen or self.default_chosen(e, pool)
            return [(t, 1.0)] if t else []
        if who in ("nearest_ally", "nearest_faction"):
            return self._nearest(src, pool, n)
        if who == "random_ally":
            pool2 = [u for u in pool if u is not src] if "其他" in e.raw else pool
            return [(u, min(1.0, n / len(pool2))) for u in pool2] if pool2 else []
        if who in ("all_allies", "faction"):
            m = _RAND_N.search(e.raw or "")
            if m and pool:                      # "随机对3名不同【河洛】英雄" → 每人期望 min(1, 3/人数)
                k = int(m.group(1))
                return [(u, min(1.0, k / len(pool))) for u in pool]
            return [(u, 1.0) for u in pool]
        if who == "keyword_holders":
            kw = str(e.target.get("keyword") or "")
            if not kw:
                for k in ("闪现", "整备", "开团", "牺牲", "登场"):
                    if k in e.raw:
                        kw = k
                        break
            hs = [u for u in pool if u.fx and u.fx.has_keyword(kw)]
            if "随机" in e.raw and hs:
                return [(u, min(1.0, n / len(hs))) for u in hs]
            return [(u, 1.0) for u in hs]
        if who in ("lowest_level", "highest_level", "highest_tier"):
            key = (lambda u: u.level) if who != "highest_tier" else (lambda u: u.quality)
            cand = [u for u in pool if u is not src or "其他" not in e.raw]
            cand.sort(key=key, reverse=(who != "lowest_level"))
            return [(u, 1.0) for u in cand[:n]]
        if who == "adjacent":
            if not src or not src.cell:
                return []
            nb = set(B.neighbors(src.cell))
            return [(u, 1.0) for u in p.heroes() if u.cell in nb]
        if who == "most_equipped":
            if not pool:
                return []
            m = max(len(u.equips) for u in pool)
            top = [u for u in pool if len(u.equips) == m]
            c = self.carry()
            if c in top:            # 并列时如何选卡面未写：假设落在主C（待实测）
                return [(c, 1.0)]
            return [(u, 1.0 / len(top)) for u in top]
        if who == "named":
            names = e.target.get("names") or []
            return [(u, 1.0) for u in p.units if u.name in names]
        return []

    def _nearest(self, src: Unit | None, pool: list, n: int) -> list:
        if not src or not src.cell:
            return []
        cand = [u for u in pool if u is not src and u.cell]
        if not cand:
            return []
        cand.sort(key=lambda u: B.hex_dist(src.cell, u.cell))
        out, i = [], 0
        while i < len(cand) and len(out) < n:
            d = B.hex_dist(src.cell, cand[i].cell)
            tie = [u for u in cand if B.hex_dist(src.cell, u.cell) == d and u not in [o for o, _ in out]]
            slots = n - len(out)
            w = min(1.0, slots / len(tie))
            out += [(u, w) for u in tie]
            i += len(tie)
            if w < 1.0:
                break
        return out

    def default_chosen(self, e: Effect, pool: list) -> Unit | None:
        """'指定1名'的牌在推演里自动用时：主C 合规就给主C，否则给合规的最高等级。"""
        c = self.carry()
        if c and c in pool:
            return c
        return max(pool, key=lambda u: u.level, default=None)

    # ---------------------------------------------- 应用效果
    def apply(self, e: Effect, src: Unit | None, src_name: str, mult: float = 1.0, chosen=None, depth: int = 0,
              extra: dict | None = None) -> float:
        """返回给出的永久等级总量（用于统计）。"""
        if mult <= 0:
            return 0.0
        eff = e.effect
        if eff in ("other", "hero_card", "transform", "remove", "shield", "hp") \
                and not (eff == "remove" and "图腾" in e.raw):
            if eff in ("other",):
                self.p.meta["unmodeled"] += 1
            return 0.0
        if not self.cond_ok(e):
            return 0.0
        amt = e.value(self.ctx(src, extra))
        if amt is None:
            self.p.meta["unmodeled"] += 1
            self.p.meta.setdefault("unmodeled_raw", set()).add(e.raw[:30])
            return 0.0
        rep = float(e.extra.get("repeat") or 1)
        amt *= mult * rep
        p = self.p
        if eff == "energy":
            p.energy += amt
            return 0.0
        if eff == "lord_xp":
            p.xp += amt
            return 0.0
        if eff == "pop":
            p.pop_cap += int(round(amt))
            return 0.0
        if eff == "refresh":
            p.free_refresh += amt
            return 0.0
        if eff == "level" and e.who == "lord":     # 棋手等级 +N（扶摇直上类）
            p.lord_level = min(B.MAX_LORD_LEVEL, p.lord_level + int(round(amt)))
            p.pop_cap = max(p.pop_cap, B.pop_cap(p.lord_level))
            return 0.0
        if eff == "core_level":
            p.core += amt
            return 0.0
        if eff == "coin_amp":
            return 0.0                       # 由 coin_bonus() 汇总
        if eff == "conv_cap":
            if src is not None:
                src.conv_cap += amt
            return 0.0
        if eff == "card":
            m = _TACTIC_RE.search(e.raw or "") or _TACTIC_RE.search(str(e.extra.get("card_name") or "获得1张" + str(e.extra.get("card_name") or "")))
            if m or "战术" in str(e.extra.get("card_name") or ""):
                name = (m.group(1) if m else str(e.extra.get("card_name")))
                for _ in range(int(round(amt))):
                    self.play_tactic(name, depth)
            return 0.0
        if eff == "remove":                  # 摧毁图腾（开团）
            self.removed_totems += sum(1 for u in p.on_board() if "图腾" in u.name)
            return 0.0
        gave = 0.0
        for t, w in self.targets(e, src, chosen):
            if t is None:
                continue
            a = amt * w
            if eff == "level":
                t.level += a
                t.src[src_name] = t.src.get(src_name, 0.0) + a
                gave += a
            elif eff == "temp_level":
                t.temp += a
            elif eff == "aspd":
                t.aspd += a
            elif eff == "revive":            # 本场获得复生（太乙真人开团给最近的人）
                self.revived.add(t.name)
            elif eff == "mana":
                if a < 0:                    # 法力值上限降低 → 本场多放技能（下一场按它估施法次数）
                    self.mana_acc[t.name] = self.mana_acc.get(t.name, 0.0) - a
            elif eff == "atk":
                if e.permanent or e.event in ("equip", "kill"):
                    t.atk += a
            elif eff == "coin":
                gave += self.use_coins(t, a, src_name, fine="精致" in (e.raw + str(e.extra.get("coin_type") or "")),
                                       depth=depth)
            elif eff == "trigger":
                for kind in str(e.trigger_kind or "prepare").split("|"):
                    gave += self.fire(t, kind, w * mult, depth + 1, via=src_name)
            elif eff == "steal_level":
                gave += self.steal(t if e.who != "adjacent" else src, src_name)
                break
            elif eff == "equip":
                item = str(e.extra.get("card_name") or e.extra.get("item") or e.extra.get("card") or "")
                if not item:
                    m = _EQUIP_NAME.search(e.raw or "")
                    item = m.group(1) if m else "?"
                if e.event in ("passive", "deploy") and e.who == "self":       # 出生自带（干将莫邪）→ 绑定
                    if item not in t.equips and len(t.equips) < B.MAX_EQUIPS:
                        t.equips.append(item)
                        t.bound.append(item)
                elif len(t.equips) < B.MAX_EQUIPS and e.event not in ("open", "battle_start"):
                    t.equips.append(item)
            elif eff == "temp_to_perm":
                pass                          # 战斗结束统一处理
            elif eff == "summon":
                pass                          # 上场时处理（deploy_summons）
        return gave

    def cond_ok(self, e: Effect) -> bool:
        """条件只认几种能机械判定的写法，其余（自由文本）一律当成立 —— 计入文档"已知不足"。"""
        c = e.condition
        txt = (c if isinstance(c, str) else json.dumps(c, ensure_ascii=False) if c else "") + ""
        if "满级" in txt and "未满级" not in txt and self.p.lord_level < B.MAX_LORD_LEVEL:
            return False
        if "未满级" in txt and self.p.lord_level >= B.MAX_LORD_LEVEL:
            return False
        if ("如果已满级" in e.raw or "若已满级" in e.raw) and self.p.lord_level < B.MAX_LORD_LEVEL and e.effect == "energy":
            return False
        return True

    def coin_bonus(self) -> float:
        b = 0.0
        for u in self.p.units:
            for e in self.card_fx(u):
                if e.effect == "coin_amp":
                    b += (e.value({}) or 0)
        for n in self.p.talents + self.p.lord_cards:
            c = self.D.get(n)
            for e in (c.effects if c else []):
                if e.effect == "coin_amp":
                    b += e.value({}) or 0
        return b

    def use_coins(self, t: Unit, n: float, src_name: str, fine: bool = False, depth: int = 0) -> float:
        c = self.D.get("精致古币" if fine else "古币")
        per = 0.0
        for e in (c.effects if c else []):
            if e.effect == "level":
                per += e.value({}) or 0
        per = (per or (4 if fine else 2)) + self.coin_bonus()
        a = per * n
        t.level += a
        t.src[src_name] = t.src.get(src_name, 0.0) + a
        self.p.meta["coins_used"] = self.p.meta.get("coins_used", 0) + n
        return a + self.react("react:coin", n, depth)

    def steal(self, thief: Unit | None, src_name: str) -> float:
        """夺取：相邻随机 1 名己方英雄的永久等级（≤10）→ 夺取者（期望值：相邻者平分）。"""
        if not thief or not thief.cell:
            return 0.0
        nb = [u for u in self.p.heroes() if u.cell in set(B.neighbors(thief.cell)) and u is not thief]
        if not nb:
            return 0.0
        tot = 0.0
        for u in nb:
            a = min(10.0, max(0.0, u.level - 1)) / len(nb)
            u.level -= a
            u.src["被夺取"] = u.src.get("被夺取", 0.0) - a
            tot += a
        thief.level += tot
        thief.src[src_name] = thief.src.get(src_name, 0.0) + tot
        return 0.0

    def fire(self, u: Unit, event: str, mult: float = 1.0, depth: int = 0, via: str = "") -> float:
        """让单位 u 的 event 类条目结算一次（含觉醒翻倍、repeat），并发反应器事件。"""
        if depth > self.P["trigger_depth"]:
            return 0.0
        k = (id(u), event)
        if self.fires.get(k, 0) >= self.P["trigger_cap"]:
            return 0.0
        self.fires[k] = self.fires.get(k, 0) + 1
        gave, fired = 0.0, False
        for e in self.card_fx(u):
            if e.event == event:
                fired = True
                gave += self.apply(e, u, u.name if not via else f"{u.name}({via})", mult * self._double(u),
                                   depth=depth)
        for name in u.equips:
            c = self.D.get(name)
            for e in (c.effects if c else []):
                if e.event == event and event in ("prepare", "open", "round_start", "battle_start"):
                    fired = True
                    gave += self.apply(e, u, name, mult, depth=depth)
        if fired and event in ("prepare", "open", "deploy"):
            n = mult * max(float(e.extra.get("repeat") or 1) for e in self.card_fx(u) if e.event == event) \
                if any(e.event == event for e in self.card_fx(u)) else mult
            gave += self.react(f"react:{event}", n, depth)
        return gave

    def react(self, ev: str, n: float, depth: int = 0) -> float:
        """反应器：场上单位 / 天赋 / 棋手卡里 event 匹配 ev 的条目，每次触发结算 n 次。"""
        if n <= 0:
            return 0.0
        key = ev.split(":", 1)[1]
        gave = 0.0
        for u in list(self.p.units):
            for e in self.card_fx(u):
                if e.event.startswith("react:") and key in e.event.split(":", 1)[1].split("|"):
                    m = n
                    cap = e.extra.get("per_round_cap")
                    if cap:                                   # "每回合首次…"：每回合最多 cap 次
                        ck = ("cap", u.name, e.raw)
                        m = min(n, max(0, cap - self.fires.get(ck, 0)))
                        self.fires[ck] = self.fires.get(ck, 0) + m
                        if m <= 0:
                            continue
                    gave += self.apply(e, u, u.name, m * self._double(u), depth=depth + 1)
        for nm in self.p.talents + self.p.lord_cards:
            c = self.D.get(nm)
            for e in (c.effects if c else []):
                if e.event.startswith("react:") and key in e.event.split(":", 1)[1].split("|"):
                    gave += self.apply(e, None, nm, n, depth=depth + 1)
        return gave

    def global_fx(self, event: str):
        """天赋 / 棋手卡 / 装备里的 event 条目（不属于某个单位卡面的）。"""
        for nm in self.p.talents + self.p.lord_cards:
            c = self.D.get(nm)
            for e in (c.effects if c else []):
                if e.event == event:
                    holder = None
                    if e.who in ("chosen", "self") and nm in self.p.lord_cards and self.p.lord_bound:
                        holder = self.p.unit(self.p.lord_bound)
                    self.apply(e, holder, nm, chosen=holder)

    # ---------------------------------------------- 上场 / 合成
    def deploy(self, name: str, cell=None, level: float = 1.0) -> Unit | None:
        p = self.p
        u = make_unit(name, level, cell)
        if u.cell is None:
            u.cell = best_cell(p, u)
        if u.cell is None:
            return None
        p.units.append(u)
        self.fire(u, "deploy")
        for e in self.card_fx(u):
            if e.effect == "equip" and e.event == "passive" and e.who == "self":
                self.apply(e, u, u.name)
        self.deploy_summons(u)
        for e in self.card_fx(u):
            if e.effect == "steal_level" and e.event in ("deploy", "merge"):
                self.steal(u, u.name)
                break
        return u

    def merge(self, u: Unit, gain: float | None = None) -> None:
        """合成同名：+等级、觉醒进度 +1；使用英雄牌 = 一次登场（"合成登场英雄的同名 = 一次登场"）。"""
        u.level += self.P["merge_gain"] if gain is None else gain
        u.src["合成"] = u.src.get("合成", 0.0) + (self.P["merge_gain"] if gain is None else gain)
        u.awaken += 1
        if u.awaken >= self.P["awaken_need"] and not u.awakened:
            u.awakened = True                  # 觉醒回手、重新打出 → 再触发一次登场
            self.fire(u, "deploy")
        self.fire(u, "deploy")
        for e in self.card_fx(u):
            if e.effect == "steal_level" and e.event in ("deploy", "merge"):
                self.steal(u, u.name)
                break
        self.react("react:merge", 1)
        self.jinnang_tick()

    def jinnang_tick(self) -> None:
        """每合成 3 次送 1 个效果锦囊，品阶 = min(棋手等级, 5)（game-rules）。遗志只从 3 阶锦囊开出：
        第 548~627 局 44 道"纯 3 阶效果牌 N 选 1"里 17 道有遗志（38.6%，95% 区间约 25%~54%），4/5 阶锦囊 0/144。
        记"至今仍没开出遗志的概率" p.meta['yizhi_miss']，推演器据此估遗志路线的期望（概率从日志自行演算）。"""
        p = self.p
        p.meta["merges"] = p.meta.get("merges", 0) + 1
        if p.meta["merges"] % 3:
            return
        tier = min(p.lord_level, 5)
        p.meta.setdefault("jinnang", {})
        p.meta["jinnang"][tier] = p.meta["jinnang"].get(tier, 0) + 1
        if tier == 3:
            p.meta["yizhi_miss"] = p.meta.get("yizhi_miss", 1.0) * (1 - self.P["yizhi_p3"])

    def deploy_summons(self, u: Unit) -> None:
        for e in self.card_fx(u):
            if e.effect == "summon" and e.event in ("passive", "deploy"):
                s = str(e.extra.get("summon") or "") or summon_name_in(e.raw)
                if not s:
                    continue
                if any(x.name == s for x in self.p.units):
                    continue
                su = make_unit(s, u.level if "等级与自身相同" in e.raw else 1)
                su.cell = best_cell(self.p, su, near=u.cell)
                if su.presence == "prep" and su.cell:
                    self.p.units.append(su)

    # ---------------------------------------------- 一场战斗
    def run_battle(self) -> RoundRecord:
        p, P = self.p, self.P
        self.removed_totems = 0
        self.mana_acc = {}
        self.revived = {u.name for u in p.on_board() if u.fx and u.fx.has_keyword("复生")}
        self.granted = granted_keywords(p)
        self.fires = {}
        E, en = enemy_power(p.round)
        self.battle = {"enemy_n": en, "attacks": {}}
        for u in p.units:
            u.aspd = 0.0
        # 1. 开团
        for u in list(p.on_board()):
            self.fire(u, "open")
        self.global_fx("open")
        # 2. 战斗开始时
        for u in list(p.on_board()):
            self.fire(u, "battle_start")
        self.global_fx("battle_start")
        if p.core > 0:                                   # 关键词「阿科米亚核心」
            sun = [u for u in p.heroes() if u.faction == "日落海"]
            for u in sun:
                u.temp += min(p.core * 0.5, p.core / len(sun))
        # 3. 战力 / 胜率
        team = [u for u in p.on_board()]
        bind = p.lord_bound if any(self.D.get(x) and self.D.get(x).category == "棋手技能牌" for x in p.lord_cards) else ""
        # 复生 = 多一条命；棋手技能绑定的人（枫叶舞：受重伤时护盾并转移）= 更难死
        rv = lambda u: (1 + P["revive_power"] if u.name in self.revived else 1.0) * \
            (1 + P["bind_power"] if bind and u.name == bind else 1.0)
        k = P["win_k"]
        pw0 = {id(u): unit_power(u) * rv(u) for u in team}
        Pw = agg(pw0.values())
        win = 1.0 / (1.0 + min(1e12, (E / max(1e-9, Pw)) ** k))
        # 阵亡的人只打了一部分时间：战力按存活时间折算（中途阵亡平均损失 death_cost 的输出），
        # 胜率 → 阵亡概率 → 有效战力 → 胜率，迭代到稳定（牺牲件阵亡给守约降法力的好处在下一场生效，这里只扣它本场少打的输出）
        for _it in range(P["win_iter"]):
            pd0 = {}
            for u in team:
                rowm = P["front_mult"] if B.row_of(u.cell) <= 1 else P["back_mult"]
                d = min(0.98, ((1 - win) * P["death_lose"] + win * P["death_win"]) * rowm)
                pd0[id(u)] = d * (P["revive_death_mult"] if u.name in self.revived else 1.0)
            Pw = agg(pw0[id(u)] * (1 - P["death_cost"] * pd0[id(u)]) for u in team)
            win = 1.0 / (1.0 + min(1e12, (E / max(1e-9, Pw)) ** k))
        kills = en * (P["kill_base"] + (1 - P["kill_base"]) * win)
        car = self.carry()
        wts = {}
        for u in team:
            w = unit_power(u) * rv(u) * (P["output_kill_w"] if is_output(u) else 0.4)
            if u is car:
                w *= P["carry_kill_w"]
            wts[id(u)] = w
        W = sum(wts.values()) or 1.0
        pdie = {}
        for u in team:
            rowm = P["front_mult"] if B.row_of(u.cell) <= 1 else P["back_mult"]
            pdie[id(u)] = min(0.98, ((1 - win) * P["death_lose"] + win * P["death_win"]) * rowm)
            if u.name in self.revived:
                pdie[id(u)] *= P["revive_death_mult"]
        # 4. 普攻 / 技能计数（动态攻速）
        tot_att = 0.0
        for u in team:
            alive = 1 - 0.5 * pdie[id(u)]
            a = base_aps(u) * (1 + u.aspd) * P["battle_seconds"] * alive
            self.battle["attacks"][id(u)] = a
            tot_att += a
        for u in team:
            for e in self.card_fx(u):
                if e.event == "attack_n" and e.when_n:
                    n_att = tot_att if ("合计" in e.raw or "你的单位" in e.raw) else self.battle["attacks"][id(u)]
                    self.apply(e, u, u.name, n_att / e.when_n)
            pr = _props(u.name)
            mana_max = pr.get("energy") or 0
            if mana_max:
                cap = max(P["mana_floor"], mana_max - 0.5 * u.mana_red) if mana_max > P["mana_floor"] else mana_max
                casts = (pr.get("initEnergy", 0) + P["mana_hit"] + self.battle["attacks"][id(u)] * P["mana_per_attack"]) / cap
                for e in self.card_fx(u):
                    if e.event == "skill_cast" and e.effect in ("level",) and e.permanent:
                        self.apply(e, u, u.name, casts)
        # 5. 击杀 / 阵亡 / 胜负
        flash_k = 0.0
        for u in team:
            ku = kills * wts[id(u)] / W
            if u.fx and u.fx.has_keyword("闪现"):
                flash_k += ku
            for e in self.card_fx(u):
                if e.event == "kill" and e.effect in ("level", "temp_level", "coin", "energy") and e.who == "self":
                    self.apply(e, u, u.name, ku * self._double(u))
            for nm in u.equips:
                c = self.D.get(nm)
                effs = c.effects if c else []
                dbl = any("胜利翻倍" in x.raw for x in effs)
                for e in effs:
                    if e.event == "kill" and e.effect in ("level", "atk") and "胜利翻倍" not in e.raw:
                        self.apply(e, u, nm, ku * ((1 + win) if dbl else 1))
            pd = pdie[id(u)]
            for e in self.card_fx(u):
                if e.event in ("sacrifice", "death"):
                    self.apply(e, u, u.name, pd * self._double(u))
            if (u.fx and u.fx.has_keyword("牺牲") and any(e.event == "sacrifice" for e in self.card_fx(u))) \
                    or "牺牲" in self.granted.get(u.name, ()):       # 钟馗类：左右 1 格的英雄获得牺牲
                self.react("react:sacrifice", pd)
        for u in team:
            for e in self.card_fx(u):
                if e.event == "win":
                    self.apply(e, u, u.name, win * self._double(u))
                elif e.event == "lose":
                    self.apply(e, u, u.name, (1 - win) * self._double(u))
        for nm in p.talents + p.lord_cards:
            c = self.D.get(nm)
            for e in (c.effects if c else []):
                if e.event == "win":
                    self.apply(e, None, nm, win)
                elif e.event == "lose":
                    self.apply(e, None, nm, 1 - win)
        # 6. 临时转永久 → 清临时
        for u in team:
            for e in self.card_fx(u):
                if e.effect == "temp_to_perm" and e.event == "passive":
                    cap = (e.value({}) or 0) * self._double(u) + u.conv_cap
                    for t, w in self.targets(e, u):
                        if t is None:
                            continue
                        a = min(max(0.0, t.temp), cap) * w
                        t.level += a
                        t.src[f"{u.name}(临时转永久)"] = t.src.get(f"{u.name}(临时转永久)", 0.0) + a
        for u in p.units:
            u.temp = 0.0
            u.aspd = 0.0
            u.mana_red = self.mana_acc.get(u.name, 0.0)
        p.meta["kills_last_round"] = kills
        p.meta["flash_kills_last_round"] = flash_k
        return RoundRecord(p.round, win, Pw, E, kills, {}, p.energy, p.lord_level, 0)

    # ---------------------------------------------- 回合间
    def round_end(self) -> None:
        p = self.p
        r = p.round
        p.xp += B.xp_gain(r)
        p.round = r + 1
        p.energy += B.income(p.round)
        self.level_ups()

    def level_ups(self) -> None:
        p = self.p
        while p.lord_level < B.MAX_LORD_LEVEL and p.xp >= B.XP_NEED.get(p.lord_level, 999):
            p.xp -= B.XP_NEED[p.lord_level]
            p.lord_level += 1
            p.pop_cap = max(p.pop_cap, B.pop_cap(p.lord_level))
            p.meta["talent_picks"] = p.meta.get("talent_picks", 0) + 1
            for u in p.units:                       # "棋手升级时" 类条目
                for e in self.card_fx(u):
                    if e.effect == "lord_xp" and "升级时" in e.raw:
                        p.xp += (e.value({}) or 0) * self._double(u)

    def invest(self, car, lv0: float) -> None:
        """日常投入：主C 每回合 + 实测涨级中位数 × invest_frac（实测里有一部分本来就来自卡面引擎，所以比例 <1）。"""
        if car is None or car not in self.p.units:
            return
        want = CARRY_INVEST.get(min(16, self.p.round), 25) * self.P["invest_frac"]
        add = want          # 固定加：卡面引擎（芈月、战术牌…）给的是"额外"的，不能被它吸收（2026-10-14：原来补到中位数为止，芈月 +1 全被抵掉）
        if add:
            car.level += add
            car.src["日常投入"] = car.src.get("日常投入", 0.0) + add

    def play_tactic(self, name: str, depth: int = 0) -> None:
        """战术牌（转瞬，拿到就打）：按卡面结算，并发"使用效果牌"事件（芈月 +1 级给装备最多的逐鹿英雄、白起每回合首次再给 1 张援护战术）。
        推演战术：指定 1 名逐鹿英雄本回合法力上限 -5×已用张数（上限 20；高阶 -10×，上限 40）—— 嬴政攒蓝的关键。"""
        if depth > self.P["trigger_depth"]:
            return
        p = self.p
        cnt = p.meta.setdefault("tactics", {})
        cnt[name] = cnt.get(name, 0) + 1
        p.meta["tactic_total"] = p.meta.get("tactic_total", 0) + 1
        car = self.carry()
        zl = [u for u in p.heroes() if u.faction == "逐鹿"]
        if "推演" in name:
            hi = "高阶" in name
            t = car if (car in zl) else (max(zl, key=lambda u: unit_power(u)) if zl else None)
            if t is not None:
                t.mana_cut = max(t.mana_cut, min(40 if hi else 20, (10 if hi else 5) * cnt[name]))
        elif "特训" in name and car is not None:
            car.level += 8 if "高阶" in name else 4
            car.src[name] = car.src.get(name, 0.0) + (8 if "高阶" in name else 4)
        elif "演练" in name:
            for u in p.heroes():
                u.level += 2 if "高阶" in name else 1
        self.react("react:effect_card", 1, depth + 1)

    def fill_slots(self) -> None:
        """空着的人口会被补上（真实玩家每回合都会买人补位）：补一个没有卡面效果的"替补"，
        品阶按商店当前能出的、等级按场上非主C 英雄的中位数 × fill_level_frac。
        否则推演里"卖掉一个人"等于永久少一个人，任何卖人 / 换绑都被高估成灾难。"""
        p = self.p
        car = self.carry()
        lv = sorted(u.level for u in p.heroes() if u is not car and not u.q_fill) or [1.0]
        while p.pop_used() < p.pop_cap:
            cell = p.free_cells()
            if not cell:
                break
            n = sum(1 for u in p.units if u.q_fill)
            u = make_unit(f"替补{n + 1}", max(1.0, lv[len(lv) // 2] * self.P["fill_level_frac"]), None)
            u.q_fill = max(1, min(5, p.lord_level - 1))
            u.cell = best_cell(p, u)
            if u.cell is None:
                break
            p.units.append(u)

    def round_start(self) -> None:
        self.fires = {}
        p = self.p
        for u in p.units:
            u.mana_cut = 0.0
        for u in list(p.on_board()):
            self.fire(u, "round_start")
        self.global_fx("round_start")
        for u in list(p.on_board()):
            self.fire(u, "prepare")
        self.global_fx("prepare")
        self.shop_buy_fx()
        self.level_ups()

    def shop_buy_fx(self) -> None:
        """"在商店每买 N 张牌…"（危险数字等）：按每回合期望买牌数折算。"""
        buys = self.P["buys_per_round"]
        for nm in self.p.talents:
            c = self.D.get(nm)
            for e in (c.effects if c else []):
                m = re.search(r"每(?:在商店)?(?:购)?买(\d)张", e.raw or "")
                if m and e.effect == "energy":
                    self.p.energy += (e.value({}) or 0) * buys / int(m.group(1))

    def step(self) -> RoundRecord:
        rec = self.run_battle()
        self.round_end()
        self.fill_slots()
        car = self.carry()
        lv0 = car.level if car else 0.0
        self.round_start()
        self.invest(car, lv0)
        rec.levels = {u.name: u.level for u in self.p.units}
        rec.energy = self.p.energy
        rec.lord_level = self.p.lord_level
        rec.unmodeled = self.p.meta.get("unmodeled", 0)
        return rec


@lru_cache(maxsize=None)
def summon_name_in(raw: str) -> str:
    """raw 里点名的召唤单位（summon_units + 召唤物卡名，长名优先）。"""
    from .state import SUMMON_ATTRS
    names = set(SUMMON_ATTRS) | {c.name for c in db().by_key.values() if c.category == "召唤物" and c.unit_kind != "hero_copy"}
    for n in sorted(names, key=len, reverse=True):
        if n and n in (raw or ""):
            return n
    return ""


def _nf(f: str) -> str:
    from .normalize import norm_faction
    return norm_faction(f) or ""


# ------------------------------------------------------------------ 主C / 站位
def pick_carry(p: Position) -> Unit | None:
    """主C = 外部设定（在场）；没有则取 can_carry 且是输出位里战力最高的（召唤单位 can_carry 也算）。"""
    if p.carry:
        u = p.unit(p.carry)
        if u:
            return u
    cand = [u for u in p.units if u.can_carry and u.presence == "prep" and is_output(u)]
    return max(cand, key=unit_power, default=None)


def best_cell(p: Position, u: Unit, near=None):
    free = p.free_cells()
    if not free:
        return None
    if near:
        return min(free, key=lambda c: B.hex_dist(near, c))
    roles = set((_roles().get(u.name) or {}).get("combat_role") or [])
    front = bool(roles & {"tank", "bruiser"})
    # 坦克前排（row 0），其余后排（row 3）；同行取中间
    return min(free, key=lambda c: (abs(B.row_of(c) - (0 if front else 3)), abs(c[0] - 650)))


def project(pos: Position, rounds: int = 3, params: dict | None = None, inplace: bool = False) -> Projection:
    p = pos if inplace else pos.clone()
    start = pos.clone() if not inplace else None
    sim = Sim(p, params)
    out = Projection(start=start or pos, end=p)
    for _ in range(max(0, rounds)):
        out.rounds.append(sim.step())
    return out
