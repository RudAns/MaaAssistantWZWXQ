"""r9zzi 数据集（2026-10-25，f9zb / g9zze = 规范 54.38；R53 用；复制自 build_r9zzh.py）：
r9zzh 全部题原样沿用（题面版本不变，不重渲染）+ 第 897 局起线上题（897~1040 训练、1041~1070 测试 test_r9zzi_new）。

- 第 897 局起每局抽 10 道重标；r9zzh 抽中但没标的题（teacher_*_ids 里没有标签的）也并进本轮清单。
- 新 tag（规范 54.38）：E_noeff_buy（买 / 拿对主C 无效的牌）、E_swap_route_member（换人卖路线成员）、E_talent_new（897 起天赋题）、
  E_basic_def（N 选 1 里有全防御基础装备）。r9zzh 的旧 E_* tag 只在新局题里照打（封顶小）。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zzi.py [--check]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_r9s as BR  # noqa: E402
import build_r9za as BA  # noqa: E402
import build_r9zm as BM  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

VER = "f9zb"
PREV_VER = "f9zb"
BASE = "r9zzh"
NAME = "r9zzi"
MATCH_RANGE = (897, 1070)
TEST_MATCHES = set(range(1041, 1071))
RESPLIT_TRAIN: set = set()
PER_MATCH_FROM = 897                   # 每局抽 10 道重标从这局起
PER_MATCH_N = 10
TEST_SPLIT = "test_r9zzi_new"
OFFLINE_FROM = 600                     # 离线 use_timing / sell_hand 题从这局起（states.jsonl 有棋手等级与场上读数）
LAYA_KINDS = BA.LAYA_KINDS + ("use_timing", "sell_hand")     # sell_filler 停用（g9zj 13）
STUDENT_KEYS = BR.STUDENT_KEYS + ("model",)
NOTE_PREFIXES = ("R29", "R31", "R32", "R34", "R35", "R36", "R37", "R38", "R39", "R40", "R41", "R42", "R43", "R44", "R45", "R46", "R47",
                 "R48", "R49", "R50", "R51", "R52", "R53")

# ---------------------------------------------------------------- tag 识别（a = f9zb 渲染；t = state + question 的 json 文本）
N1_KINDS = ("card_pick", "auction", "talent", "lord_boost")           # N 选 1 类
BUY_KINDS = ("shop",) + N1_KINDS
# 54.31-1：f9za_level_gain 写的"给主C 涨多少级"行
LG_RE = re.compile(r"(carry [^\s:]+|carry \(none yet\)|nearest hero \(no carry yet\)) (≈ )?\+\d+ (once|total|lv|per|with|if)"
                   r"|converts \+\d+ more temp lv|LOWEST-level unit, never the carry|\+\d+ to (every unit|one 【|all t\d)")
LG_MSY_RE = re.compile(r"converts \+\d+ more temp lv|lv if 明世隐 is deployed")
COUNT_TALENTS = ("同心协力", "四重奏", "王牌部队", "小小的助力", "随机成长", "随机进化", "随机强化")
COUNT_RE = re.compile(r"placeable on the final board|factions in the final lineup|in the final lineup|carry is the top-tier unit of the final lineup")
# 54.29-14/16/19/20/23、54.30-7/10/11/16：点名的牌
NAMED_RULE = ("阵营祝福", "阿科米亚之子", "星源星徽", "扶倾济弱", "扫拂", "分身镜", "红叶染锋", "制裁之剑")
OPT_NAME_RE = re.compile(r"^(?:buy \([^)]*\) |bid for |sell from hand: |use on )?([^:：;(]+?)\s*[:：]")


def _review_ineffective() -> dict:
    """data/review/*_carry.json（54.29-17）：牌名 → {对其无效的主C}（把握 ≥0.6 / high / medium）。"""
    out: dict = {}
    for f in ("effect_cards_carry_a", "effect_cards_carry_b", "talents_carry", "equips_carry", "special_equips_talents_v2"):
        p = PROJECT_DIR / "data/review" / f"{f}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, v in d.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            cf = v.get("confidence")
            ok = (isinstance(cf, (int, float)) and cf >= 0.6) or cf in ("high", "medium") or cf is None
            if not ok:
                continue
            cs = {x.get("carry") if isinstance(x, dict) else x for x in (v.get("ineffective_for") or [])}
            cs.discard(None)
            if cs:
                out.setdefault(k, set()).update(cs)
    return out


INEFF = _review_ineffective()
REVIEW_CARRIES = ("百里守约", "沈梦溪", "马可波罗", "狄仁杰", "虞姬", "公孙离", "亚连", "韩信", "诸葛亮", "孙尚香", "艾琳", "蒙犽", "李信", "敖隐", "海月",
                  "刘邦", "百里玄策")
CARRY_RE = re.compile(r"(?:carry|target) (" + "|".join(REVIEW_CARRIES) + r")\b|(" + "|".join(REVIEW_CARRIES) + r") \(carry\)")


def _crit(a: dict) -> dict:
    return a["question"].get("criteria") or {}


def _opt_names(a: dict) -> list:
    out = []
    for v in _crit(a).values():
        m = OPT_NAME_RE.match(str(v))
        if m:
            out.append(m.group(1).strip())
    return out


def _carry_of(t: str) -> str:
    c = collections.Counter(m.group(1) or m.group(2) for m in CARRY_RE.finditer(t))
    return c.most_common(1)[0][0] if c else ""


def t_same_route_swap(a, t):
    """54.36-5（第 884 局钟馗 ↔ 蒙恬）：换人题同路线前排平级互换（f9zb_swap_lines "same-route swap, no clear gain"）。"""
    return a["kind"] == "swap" and "same-route swap, no clear gain" in t


def t_msy_stack(a, t):
    """54.34-3 / 54.36-3：明世隐叠层（"entry stacks lost if sold"）/ 后期新明世隐只转 +2（"new 明世隐 ≈ +2 ×"）；或换人题明世隐与太乙同题。"""
    if a["kind"] != "swap":
        return False
    return ("entry stacks lost if sold" in t or "new 明世隐 ≈ +2 ×" in t
            or ("明世隐" in json.dumps(a["question"], ensure_ascii=False) and "太乙真人" in json.dumps(a["question"], ensure_ascii=False)))


def t_lyf_taiyi(a, t):
    """54.36-7（第 886 局）：后期太乙等位、李元芳没觉醒 + 明世隐已觉醒（"李元芳 +2 lv/round (1 coin) < 太乙… revive"）。"""
    return a["kind"] == "swap" and "李元芳 +2 lv/round (1 coin) <" in t


def t_taiyi_offroute(a, t):
    """54.36-1/2（第 881 局）：第 12 回合前太乙在等、场上有路线外的人（"off-route: OK to sell for 太乙" / LATE-FILL "only replaces off-route"）。"""
    return a["kind"] in ("swap",) + BUY_KINDS and ("off-route: OK to sell for" in t or "only replaces off-route" in t)


REFRESH_SPLIT_RE = re.compile(r"refresh \([^)]*\): (missing \d+%|no missing card)(, dup \d+%)?")


def t_refresh_levelup(a, t):
    """54.36-4（第 883 局）：路线缺的引擎核心件当前等级全刷不出 → 刷新行写 "level up first"。"""
    return a["kind"] == "shop" and "unreachable at lv" in t and "level up first" in t


def t_refresh_dup(a, t):
    """54.36-4 / 54.34-1：刷新行把缺件和已有件同名（dup）分开写，且两部分都有（missing P%, dup Q%）。"""
    if a["kind"] != "shop":
        return False
    return any(x.group(2) and x.group(1).startswith("missing") for x in REFRESH_SPLIT_RE.finditer(t))


def t_refresh_miss(a, t):
    """54.36-4：刷新行只写缺件命中率（missing N%，无 dup）。"""
    if a["kind"] != "shop":
        return False
    m = [x for x in REFRESH_SPLIT_RE.finditer(t)]
    return bool(m) and not any(x.group(2) for x in m) and any(x.group(1).startswith("missing") for x in m)


def t_carry_dup(a, t):
    """54.32-1 / 54.33-1/9：主C 同名在店（买的英雄 = ctx.carry）—— 规范：主C 同名照买。"""
    if a["kind"] != "shop":
        return False
    c = (a.get("ctx") or {}).get("carry") or ""
    if not c:
        return False
    for k, v in _crit(a).items():
        if not str(k).startswith("buy"):
            continue
        m = OPT_NAME_RE.match(str(v))
        if m and m.group(1).strip() == c:
            return True
    return False


def t_awaken(a, t):
    """54.31-1 / 54.36-6：唤醒 / 集体唤醒 —— 商店 / N 选 1 的选项里有，或用牌题用的就是这张。"""
    if a["kind"] in BUY_KINDS:
        return any("唤醒" in n for n in _opt_names(a))
    if a["kind"] in ("use_card", "use_timing"):
        return "唤醒" in str(a["question"].get("instructions") or "") or USE_NOW_RE in t
    return False


USE_NOW_RE = "use now: no better timing left"


def t_entry_all_now(a, t):
    """54.36-6（第 886 局）：集体唤醒阵容定型后 use now（f9zb USE_NOW_F9ZB）。"""
    return USE_NOW_RE in t


USE_CARD_RE = re.compile(r"use the card (\S+?) on ")


def t_noncarry_equip(a, t):
    """54.31-5 / R51 非主C 装备 −0.067：装备给非主C（主C 满装 / 穿戴对象不是主C）：辅助性（控制 / 回血 / 护盾）等级行；
    或装备牌的用牌题（选项里有非主C 的穿戴对象）。"""
    if "for non-carry " in t or re.search(r"carry \S+ full: for others = ", t) is not None:
        return True
    if a["kind"] != "use_card":
        return False
    import facts as FA
    m = USE_CARD_RE.search(str(a["question"].get("instructions") or ""))
    if not m or FA.cat(m.group(1)) != "装备牌":
        return False
    c = (a.get("ctx") or {}).get("carry") or ""
    return any(str(k).startswith("use") and c and not str(v).startswith(f"use on {c}:") for k, v in _crit(a).items())


def t_ranged_only(a, t):
    """f9zb（10-21 第 884 局）：自然法杖只给远程（"ranged heroes only, X melee"）。"""
    return "ranged heroes only" in t


def t_carry_item_mismatch(a, t):
    """54.24 / 普攻叠层装不给技能型主C："for carry X: NO USE (basic-attack stacking item…)" 等主C 装备不配。"""
    return a["kind"] in BUY_KINDS + ("use_card",) and re.search(r"for carry \S+: NO USE", t) is not None


def t_full_buy_none(a, t):
    """54.31-2 / 54.32-1：商店满员买新英雄、却没有路线外的人可卖（"none → don't buy"；路线核心 / 主C 同名例外）。"""
    return a["kind"] == "shop" and ("buying means selling one of: none" in t or "board full, core piece: buy and hold" in t)


def t_full_buy(a, t):
    """54.31-2（f9za full_buy）：商店满员买新英雄 = 要卖掉一个路线外的人（有人可卖）。"""
    return a["kind"] == "shop" and "board full: buying means selling one of: " in t and "selling one of: none" not in t


def t_sac_fuel(a, t):
    """54.35-4（修正 54.31-3）：牺牲燃料降级 —— ROUTE fuel / off-route sacrifice fuel 出现在换人 / 商店 / N 选 1 / 卖手牌题里（R51 用旧口径）。"""
    return a["kind"] in ("swap", "sell_hand") + BUY_KINDS and ("ROUTE fuel: has 牺牲" in t or "off-route sacrifice fuel" in t)


def t_xuance_engine(a, t):
    """54.36-5（第 884 局）：百里玄策线引擎是河洛·古币（不是河洛·牺牲）—— 玄策线的换人 / 商店 / N 选 1 题。"""
    if a["kind"] not in ("swap", "carry") + BUY_KINDS:
        return False
    rt = (a.get("ctx") or {}).get("route")
    return "百里玄策" in json.dumps(rt, ensure_ascii=False) if rt else bool(re.search(r"ROUTE[^\n]{0,80}百里玄策|百里玄策 \(carry\)", t))


def t_listed_carry(a, t):
    """54.32-12 / R52 待补"关掉的主C"：现任是名单外的人、开着的主C 在场 / 等位（"listed carry (should take over)"）。"""
    return "listed carry (should take over)" in t


ROUTE_OWNED_RE = re.compile(r"carry (\S+) (?:lv[\d?]+ on board|in hand)")
ENGINE_LINE_RE = re.compile(r"level engine: [^\n]*")


def t_route_engine(a, t):
    """54.36-8（第 889 局 R4：李元芳产古币、河洛·古币 ASSEMBLED、狄仁杰在手，Laya 选守约 0.61）：选路线题有 ≥2 条路线的主C 在手 / 在场，
    且场上已转起来的等级引擎（level engine 行 STARTED / ASSEMBLED）的 payoff 是其中一条路线的主C。"""
    if a["kind"] != "route" or (a.get("round") or 0) < 4:
        return False
    owned = {m.group(1) for v in _crit(a).values() for m in ROUTE_OWNED_RE.finditer(str(v))}
    if len(owned) < 2:
        return False
    st = a.get("state") if isinstance(a.get("state"), str) else ""
    m = ENGINE_LINE_RE.search(st)
    if not m or not re.search(r"STARTED|ASSEMBLED", m.group(0)):
        return False
    q = re.search(r"payoff ([^;]+);", m.group(0))
    pay = {x.strip() for x in q.group(1).split(",")} if q else set()
    return bool(pay & owned)


def t_noeff_buy(a, t):
    """54.38-1：商店 / N 选 1 / 拍卖的选项写着对现任主C 无效。"""
    c = (a.get("ctx") or {}).get("carry") or ""
    return a["kind"] in BUY_KINDS and bool(c) and f"no effect on {c}" in t


def t_swap_route_member(a, t):
    """54.38-3：换人题有卖路线成员的选项。"""
    return a["kind"] == "swap" and any(str(k).startswith("s") and "ROUTE member" in str(v) for k, v in _crit(a).items())


def t_talent_new(a, t):
    """54.38-5：897 局起的天赋题。"""
    return a["kind"] == "talent" and (a.get("match_no") or 0) >= MATCH_RANGE[0]


def t_basic_def(a, t):
    """54.38-2：N 选 1 里有全防御基础装备（力量腰带 / 守护者之铠）。"""
    return a["kind"] == "card_pick" and any(n in ("力量腰带", "守护者之铠") for n in _opt_names(a))


TAGS = (
    ("E_noeff_buy", t_noeff_buy),
    ("E_swap_route_member", t_swap_route_member),
    ("E_talent_new", t_talent_new),
    ("E_basic_def", t_basic_def),
    ("E_same_route_swap", t_same_route_swap),
    ("E_msy_stack", t_msy_stack),
    ("E_lyf_taiyi", t_lyf_taiyi),
    ("E_taiyi_offroute", t_taiyi_offroute),
    ("E_refresh_levelup", t_refresh_levelup),
    ("E_refresh_dup", t_refresh_dup),
    ("E_refresh_miss", t_refresh_miss),
    ("E_carry_dup", t_carry_dup),
    ("E_awaken", t_awaken),
    ("E_entry_all_now", t_entry_all_now),
    ("E_noncarry_equip", t_noncarry_equip),
    ("E_ranged_only", t_ranged_only),
    ("E_carry_item_mismatch", t_carry_item_mismatch),
    ("E_full_buy_none", t_full_buy_none),
    ("E_full_buy", t_full_buy),
    ("E_sac_fuel", t_sac_fuel),
    ("E_xuance_engine", t_xuance_engine),
    ("E_listed_carry", t_listed_carry),
    ("E_route_engine", t_route_engine),
)
# E_carry_off（关掉的主C：carry 题且 f9za→f9zb 文本变了，或写了 listed carry）在 main 里打
CAPS = {"E_noeff_buy": (120, 30), "E_swap_route_member": (120, 30), "E_talent_new": (150, 40), "E_basic_def": (50, 12),
        "E_same_route_swap": (15, 4), "E_msy_stack": (20, 5), "E_lyf_taiyi": (8, 2), "E_taiyi_offroute": (15, 4),
        "E_refresh_levelup": (15, 4), "E_refresh_dup": (15, 4), "E_refresh_miss": (15, 4), "E_carry_dup": (20, 5),
        "E_awaken": (20, 5), "E_entry_all_now": (8, 2), "E_noncarry_equip": (25, 6), "E_ranged_only": (8, 2),
        "E_carry_item_mismatch": (20, 5), "E_full_buy_none": (15, 4), "E_full_buy": (15, 4), "E_sac_fuel": (15, 4),
        "E_xuance_engine": (15, 4), "E_listed_carry": (15, 4), "E_route_engine": (15, 4), "E_carry_off": (10, 3)}


def render(p: dict, ver: str, keys) -> dict | None:
    return BA.render(p, ver, keys)


def tags(a: dict) -> list[str]:
    t = json.dumps([a.get("state"), a["question"]], ensure_ascii=False)
    return [name for name, f in TAGS if f(a, t)]


# ---------------------------------------------------------------- 局号 ↔ 运行目录
def match_runs() -> dict:
    """局号 → [live_ 目录…]（batch2_matches 的 run + agent_one_<局号>*.log 里提到的 live_ 目录；同 check_components.matches()）。"""
    logs = PROJECT_DIR / "data" / "logs"
    out = {}
    for line in (PROJECT_DIR / "data/laya/iterations/batch2_matches.jsonl").open(encoding="utf-8"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        mn = m.get("match") or 0
        if not mn or not m.get("run"):
            continue
        segs = []
        pre = f"agent_one_{mn}"
        for f in sorted(logs.glob(pre + "*.log")):
            if not (f.stem == pre or f.stem[len(pre):].isalpha()):
                continue
            for d in re.findall(r"live_\d{6,}", f.read_text(encoding="utf-8", errors="ignore")):
                if d not in segs and (logs / d).is_dir():
                    segs.append(d)
        if m["run"] not in segs:
            segs.append(m["run"])
        out[mn] = (sorted(segs), m)
    return out


def _bought_by_match(runs: list) -> list:
    """同一局各段 laya_points 里 Laya 主动买的新英雄：[(ts, 回合, 英雄)]（同线上 laya_policy._laya_bought 口径）。"""
    import facts as FA
    out = []
    for d in runs:
        f = PROJECT_DIR / "data/logs" / d / "laya_points.jsonl"
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") != "shop" or not str(r.get("used") or "").startswith("laya") or not str(r.get("laya_choice") or "").startswith("buy"):
                continue
            q = r.get("question") or {}
            v = str((q.get("criteria") or {}).get(r["laya_choice"]) or "")
            m = OPT_NAME_RE.match(v)
            if not m:
                continue
            n = m.group(1).strip()
            ctx = r.get("ctx") or {}
            if not FA.is_hero(n) or n in (r.get("field") or []) or n in (ctx.get("hand") or []):
                continue
            out.append((float(r.get("ts") or 0), int(r.get("round") or ctx.get("round") or 0), n))
    return sorted(out)


def online_new(st) -> list[dict]:
    """第 840 局起的线上题（多段合并 + 回填 ctx.laya_bought）。"""
    mr = match_runs()
    BM.MATCH_RANGE, BM.TEST_MATCHES, BM.TEST_SPLIT = MATCH_RANGE, TEST_MATCHES, TEST_SPLIT
    BM.NOTE_PREFIXES = NOTE_PREFIXES
    BM.EXTRA_RUNS = {mn: segs for mn, (segs, m) in mr.items() if MATCH_RANGE[0] <= mn <= MATCH_RANGE[1] and len(segs) > 1
                     and (not m.get("note") or str(m.get("note")).startswith(NOTE_PREFIXES))}
    for mn, segs in BM.EXTRA_RUNS.items():
        st[f"multi_run:{mn}"] = len(segs)
    out = BM.new_points(st)
    bought = {mn: _bought_by_match(segs) for mn, (segs, _m) in mr.items() if MATCH_RANGE[0] <= mn <= MATCH_RANGE[1]}
    for p in out:
        p["match"] = p["match"].replace("r9zm_", NAME + "_")
        ctx = p.get("ctx")
        if isinstance(ctx, dict) and "laya_bought" not in ctx:
            ts = float(p.get("ts") or 0)
            lb = {}
            for t0, rnd, n in bought.get(p["match_no"]) or []:
                if t0 < ts:
                    lb[n] = rnd
            if lb:
                ctx["laya_bought"] = lb
                st["laya_bought_backfilled"] += 1
    return out


def offline_points(st) -> list[dict]:
    """点兵点将 use_timing / sell_hand：各局 states.jsonl 还原（多段的局每段都还原）。"""
    import build_dataset as BD
    BD.SELL_FILLER_QUESTIONS = True      # 点兵点将 use_timing 仍从这条开关出（sell_filler 题 f9lx 线上停用，照出但不进重标）
    BD.SELL_HAND_QUESTIONS = True
    runs = {}
    for mn, (segs, m) in match_runs().items():
        if mn < OFFLINE_FROM:
            continue
        for d in (segs if mn >= 840 else [m["run"]]):       # 840 起多段都还原（同 r9zzg）；旧局只用 batch2 里那一段（id 不变）
            runs[d] = mn
    cfgs = BR.run_render_cfgs()
    out = []
    for d, mn in sorted(runs.items(), key=lambda x: x[1]):
        rd = PROJECT_DIR / "data/logs" / d
        if not (rd / "states.jsonl").exists():
            st["offline_no_states"] += 1
            continue
        rows = []
        f = rd / "laya_points.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        F.ACTIVE = BM.OLD_VER
        RR.ALLOW_NEW_KEYS = True
        cfg, _hit = BR.pick_cfg(rows, {k: v for k, v in (cfgs.get(d) or {}).items() if k != "log"} or None)
        BR.apply_cfg(cfg)
        F.ACTIVE = VER
        F._facts()
        try:
            pts = BD.extract_run(rd)
        except Exception as e:  # noqa: BLE001
            st["offline_err"] += 1
            print("extract_run 出错", d, type(e).__name__, e)
            continue
        for p in pts:
            if p["kind"] not in ("sell_filler", "use_timing", "sell_hand"):
                continue
            p["run"] = d
            p["id"] = BR.pid(d, p.get("ts"), p["kind"])
            p["source"] = "offline"
            p["render_cfg"] = cfg
            p["match"] = f"{NAME}_m{mn:03d}"
            p["match_no"] = mn
            p["split"] = TEST_SPLIT if mn in TEST_MATCHES else "train"
            p["feature_version"] = VER
            st[f"offline:{p['split']}:{p['kind']}"] += 1
            out.append(p)
    BR.apply_cfg(None)
    return out


def _strat(ids, rows_by, cap, rng):
    return BM._stratified(ids, rows_by, cap, rng)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data/laya/datasets"
    st = collections.Counter()
    t0 = time.time()
    base = [json.loads(l) for l in (ds / BASE / "points.jsonl").open(encoding="utf-8")]
    base_ids = {p["id"] for p in base}
    rows, changed = [], set()
    for p in sorted(base, key=lambda p: (p.get("run") or "", p["id"])):
        a = p                                    # 题面版本没变（f9zb）：r9zzh 的题原样沿用，不重渲染
        if (a.get("match_no") or 0) in RESPLIT_TRAIN and a.get("split") != "train":
            a["split"] = "train"                 # 856~869：r9zzg 的 test_r9zzg_new → train
            st["base_resplit_train"] += 1
        rows.append(a)
    st["base_kept"], st["base_changed_text"] = len(rows), len(changed)
    print("base 渲染完", len(rows), round(time.time() - t0), "s", flush=True)
    seen = set()
    for p in sorted(online_new(st), key=lambda p: (p["match"], p.get("ts") or 0)):
        if p["id"] in base_ids:
            continue
        keys = list(p["question"]["criteria"])
        a = render(p, VER, keys)
        if a is None and p["kind"] == "talent":
            # 线上天赋题被临时规则删到只剩 1 个选项（C 级去掉）→ 限不回去：保留完整重渲染给老师标，让模型自己学会避开 C（撤临时规则的前提）
            a = render(p, VER, None)
            if a is not None and len(a["question"]["criteria"]) >= 2:
                st["new_talent_full_options"] += 1
            else:
                a = None
        if a is None:
            st["new_dropped:" + p["kind"]] += 1
            continue
        sig = (p["match"], p.get("round"), p["kind"], json.dumps(a["question"]["criteria"], ensure_ascii=False, sort_keys=True))
        if sig in seen:
            continue
        seen.add(sig)
        st[f"new_kept:{a['split']}:{a['kind']}"] += 1
        rows.append(a)
    have = {r["id"] for r in rows}
    for p in offline_points(st):
        if p["id"] in have:
            continue
        have.add(p["id"])
        rows.append(p)
    BR.apply_cfg(None)
    F.ACTIVE = VER
    new_ids = [r["id"] for r in rows if r["id"] not in base_ids]
    new_set = set(new_ids)
    rows_by = {r["id"]: r for r in rows}
    assert len(rows_by) == len(rows)

    # ---- 重标清单
    rng = random.Random(args.seed)
    tg = {}
    for r in rows:
        if not (r["kind"] in LAYA_KINDS or r["kind"] == "swap") or r["id"] not in new_set:
            continue
        t = tags(r)
        if r["id"] in changed:
            t.append("Z_changed")
        if r["kind"] == "carry" and (r["id"] in changed or "E_listed_carry" in t):
            t.append("E_carry_off")
        tg[r["id"]] = t
    for i, t in tg.items():
        for x in t:
            st[f"tag:{rows_by[i]['split'] == 'train' and 'train' or 'test'}:{x}"] += 1
    tr, te = set(), set()
    per_match = collections.defaultdict(list)   # 856 起各局线上题每局抽 35 道（含 base 里已有的 856~869）
    for i in sorted(rows_by):
        r = rows_by[i]
        if r.get("source") == "offline" or r["kind"] not in LAYA_KINDS + ("swap",) or (r.get("match_no") or 0) < PER_MATCH_FROM:
            continue
        per_match[r["match"]].append(i)
    for m, ids in sorted(per_match.items()):
        rng.shuffle(ids)
        for i in ids[:PER_MATCH_N]:
            (te if rows_by[i]["split"] == TEST_SPLIT else tr).add(i)
    st["per_match_matches"] = len(per_match)
    _lab = set()
    for line in (PROJECT_DIR / "data/laya/labels/r9zzh_all/labels_all.jsonl").open(encoding="utf-8"):
        if line.strip():
            _lab.add(json.loads(line)["id"])
    for fn, dst in (("teacher_train_ids.json", tr), ("teacher_test_ids.json", te)):
        for i in json.loads((ds / BASE / fn).read_text(encoding="utf-8")):
            if i in rows_by and i not in _lab and rows_by[i]["kind"] != "sell_filler":
                dst.add(i)
                st[f"carry_over_unlabeled:{fn[8:-9]}"] += 1
    for name, (ctr, cte) in CAPS.items():
        idx_tr = [i for i, t in tg.items() if name in t and rows_by[i]["split"] == "train" and i not in tr]
        idx_te = [i for i, t in tg.items() if name in t and rows_by[i]["split"] != "train" and i not in te]
        tr |= set(_strat(idx_tr, rows_by, ctr, rng))
        te |= set(_strat(idx_te, rows_by, cte, rng))
    sel = collections.Counter()
    for i in tr | te:
        sel[f"{'train' if i in tr else 'test'}:{rows_by[i]['kind']}"] += 1
    sel_tag = collections.Counter()
    for i in tr | te:
        for x in tg.get(i) or []:
            sel_tag[f"{'train' if i in tr else 'test'}:{x}"] += 1
    up = sorted(i for i, t in tg.items() if any(x.startswith("E_") for x in t) and rows_by[i]["split"] == "train")

    h = hashlib.md5()
    for r in rows:
        h.update(json.dumps([r["id"], r["state"], r["question"]], ensure_ascii=False, sort_keys=True).encode())
    summary = {"points": len(rows), "new": len(new_ids), "md5": h.hexdigest(), "secs": round(time.time() - t0),
               "relabel_train": len(tr), "relabel_test": len(te), "selected": dict(sorted(sel.items())),
               "selected_tags": dict(sorted(sel_tag.items())),
               "upsample_candidates": len(up), "stats": {k: v for k, v in sorted(st.items()) if not k.startswith("cfg_hit:")},
               "by_split": dict(collections.Counter(r["split"] for r in rows)),
               "new_by_split_kind": dict(sorted(collections.Counter(f"{rows_by[i]['split']}:{rows_by[i]['kind']}" for i in new_ids).items()))}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    od = ds / NAME
    od.mkdir(parents=True, exist_ok=True)
    with (od / "points.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (od / "info.json").write_text(json.dumps(dict(summary, base=BASE, feature_version=VER, built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                  matches=list(MATCH_RANGE), test_matches=sorted(TEST_MATCHES)),
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    (od / "new_ids.json").write_text(json.dumps(new_ids), encoding="utf-8")
    (od / "tags.json").write_text(json.dumps({i: t for i, t in tg.items() if t}, ensure_ascii=False), encoding="utf-8")
    (od / "changed_vs_prev.json").write_text(json.dumps(sorted(changed)), encoding="utf-8")
    (od / "teacher_train_ids.json").write_text(json.dumps(sorted(tr)), encoding="utf-8")
    (od / "teacher_test_ids.json").write_text(json.dumps(sorted(te)), encoding="utf-8")
    (od / "upsample_w_ids.json").write_text(json.dumps(up), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
