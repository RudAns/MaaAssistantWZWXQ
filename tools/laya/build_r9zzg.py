"""r9zzg 数据集（2026-10-21，f9za / g9zzc = 规范 54.29~54.31；R51 用；复制自 build_r9zzf.py）：
r9zzf 全部题按 f9za 重渲染 + 第 840 局起线上题（840~855 训练、856 起测试；849 局人工接手，不在 batch2_matches 里，自动跳过）。

- r9zzf 只收到第 839 局（10-20 下午建的），840~844（R49）此前没进任何数据集 → 这次一起进训练（只按 tag 抽重标，不做每局 35 题）；
  第 845 局起（R49 最后一局 + R50）每局抽 35 道重标。
- 一局中途重启有多个 live_ 目录（851 / 852）：按 agent_one_<局号>*.log 里提到的 live_ 目录合并（同 tools/check_components.matches()）。
- 回填 ctx.laya_bought（54.31-2）：线上从 10-21 才开始记"Laya 主动买的新英雄 → 回合"，之前的题没有这个字段 →
  按同一局 laya_points 里更早的商店题（used=laya*，laya_choice=buy*，买的是不在场 / 不在手的英雄）还原，f9za 换人题才会写 "bought this round"。
- 定向重标 tag（E_*，识别条件见 TAGS 注释；按类封顶 CAPS）+ Z_changed（f9z→f9za 文本变了的题）。r9zzf 的旧 E_* tag（54.28）R50 已重标过，不再沿用。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zzg.py [--check]
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

VER = "f9za"
PREV_VER = "f9z"
BASE = "r9zzf"
NAME = "r9zzg"
MATCH_RANGE = (840, 900)
TEST_MATCHES = set(range(856, 901))
PER_MATCH_FROM = 845                   # 每局抽 35 道重标从这局起（840~844 只按 tag 抽）
PER_MATCH_N = 35
TEST_SPLIT = "test_r9zzg_new"
OFFLINE_FROM = 600                     # 离线 use_timing / sell_hand 题从这局起（states.jsonl 有棋手等级与场上读数）
LAYA_KINDS = BA.LAYA_KINDS + ("use_timing", "sell_hand")     # sell_filler 停用（g9zj 13）
STUDENT_KEYS = BR.STUDENT_KEYS + ("model",)
NOTE_PREFIXES = ("R29", "R31", "R32", "R34", "R35", "R36", "R37", "R38", "R39", "R40", "R41", "R42", "R43", "R44", "R45", "R46", "R47",
                 "R48", "R49", "R50", "R51")

# ---------------------------------------------------------------- tag 识别（a = f9za 渲染；t = state + question 的 json 文本）
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


def t_level_gain(a, t):
    """54.31-1：N 选 1 / 拍卖 / 天赋里有选项写了"给主C 涨多少级"（f9za_level_gain）；商店题要 ≥2 个这样的选项；
    任何一题有明世隐登场转化（唤醒 / 集体唤醒 / 锦囊）的选项都算。"""
    if a["kind"] not in BUY_KINDS:
        return False
    vs = [str(v) for v in _crit(a).values()]
    n = sum(1 for v in vs if LG_RE.search(v))
    return n >= (2 if a["kind"] == "shop" else 1) or any(LG_MSY_RE.search(v) for v in vs)


def t_level_use(a, t):
    """54.31-1：用牌题里写了明世隐转化 / 同品阶 +k 的主C 收益（f9za_use_gain）。"""
    return a["kind"] == "use_card" and bool(re.search(r"converts \+\d+ more temp lv|\+\d+ to all t\d \(\d+ units\); carry ", t))


def t_bought_swap(a, t):
    """54.31-2：换人题等位的是本回合 / 上回合 Laya 主动买的新英雄（ctx.laya_bought，840 起的线上题回填）。"""
    return a["kind"] == "swap" and ("bought this round (our own buy)" in t or "bought last round (our own buy)" in t)


def t_full_buy_none(a, t):
    """54.31-2：商店满员买新英雄、却没有路线外的人可卖（"none → don't buy"）。"""
    return a["kind"] == "shop" and ("buying means selling one of: none" in t or "board full, core piece: buy and hold" in t)


def t_full_buy(a, t):
    """54.31-2：商店满员买新英雄 = 要卖掉一个路线外的人（有人可卖）。"""
    return a["kind"] == "shop" and "board full: buying means selling one of: " in t and "selling one of: none" not in t


def t_sac_fuel(a, t):
    """54.31-3：守约线带牺牲的英雄（ROUTE fuel 行）出现在换人 / 商店 / N 选 1 / 卖手牌题里。"""
    return a["kind"] in ("swap", "sell_hand") + BUY_KINDS and ("ROUTE fuel: has 牺牲" in t or "off-route sacrifice fuel" in t)


def t_count_talent(a, t):
    """54.31-4：阵营 / 品阶计数类天赋（同心协力 / 四重奏 / 王牌部队 / 小小的助力 / 随机成长·进化·强化，或 f9za 写了按最终可摆阵容计数的条件）。"""
    if a["kind"] != "talent":
        return False
    names = _opt_names(a)
    return any(n.startswith(COUNT_TALENTS) or any(c in n for c in COUNT_TALENTS) for n in names) or bool(COUNT_RE.search(t))


def t_noncarry_equip(a, t):
    """54.31-5：装备给非主C（主C 满装 / 穿戴对象不是主C）：f9za 写了辅助性等级行。"""
    return "for non-carry " in t or re.search(r"carry \S+ full: for others = ", t) is not None


def t_route(a, t):
    """54.29-24 / 54.30-1~2 / 54.30-6：路线题（关掉的主C 的路线被排除；第 4 回合拍卖后定）。"""
    return a["kind"] == "route" and (a.get("round") or 0) >= 4      # 10-21：第 4 回合前线上不出路线题


def t_ineffective(a, t):
    """54.29-17/18/22、54.30-12：N 选 1 / 拍卖 / 天赋里有按主C 复审表（data/review/*_carry.json）对当前主C 无效的牌（开团号令对亚连、守护者之铠给守约…）。"""
    if a["kind"] not in N1_KINDS:
        return False
    C = _carry_of(t)
    if not C:
        return False
    return any(C in INEFF.get(n, ()) for n in _opt_names(a))


def t_talent_equip(a, t):
    """54.29-17 / f9za：天赋三选一里直接给装备的选项（末世…），题面写了对主C 的 FITS / NO USE。"""
    return a["kind"] == "talent" and re.search(r"for carry \S+: (FITS|NO USE)", t) is not None


def t_named_rule(a, t):
    """54.29-14/16/19/23、54.30-7/11/16：规范点名的牌（阵营祝福 / 阿科米亚之子 / 星源星徽 / 扶倾济弱 / 扫拂 / 分身镜 / 红叶染锋 / 制裁之剑）出现在选项里。"""
    if a["kind"] not in BUY_KINDS + ("use_card",):
        return False
    return any(any(x in n for x in NAMED_RULE) for n in _opt_names(a))


TAGS = (
    ("E_level_gain", t_level_gain),
    ("E_level_use", t_level_use),
    ("E_bought_swap", t_bought_swap),
    ("E_full_buy_none", t_full_buy_none),
    ("E_full_buy", t_full_buy),
    ("E_sac_fuel", t_sac_fuel),
    ("E_count_talent", t_count_talent),
    ("E_noncarry_equip", t_noncarry_equip),
    ("E_route", t_route),
    ("E_ineffective", t_ineffective),
    ("E_talent_equip", t_talent_equip),
    ("E_named_rule", t_named_rule),
)
# E_carry_off（54.29-24 / 54.30-2：换主C 题候选按 carry_pool 去掉关掉的主C）= carry 题且 f9z→f9za 文本变了，在 main 里打
CAPS = {"E_level_gain": (70, 18), "E_level_use": (25, 8), "E_bought_swap": (40, 10), "E_full_buy_none": (40, 10), "E_full_buy": (40, 10),
        "E_sac_fuel": (60, 15), "E_count_talent": (80, 20), "E_noncarry_equip": (50, 12), "E_route": (30, 8), "E_carry_off": (25, 6),
        "E_ineffective": (80, 20), "E_talent_equip": (40, 10), "E_named_rule": (50, 12), "Z_changed": (150, 40)}


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
        for d in (segs if mn >= MATCH_RANGE[0] else [m["run"]]):       # 旧局照 r9zzf 只用 batch2 里那一段（id 不变）
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
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data/laya/datasets"
    st = collections.Counter()
    t0 = time.time()
    base = [json.loads(l) for l in (ds / BASE / "points.jsonl").open(encoding="utf-8")]
    base_ids = {p["id"] for p in base}
    rows, changed = [], set()
    for p in sorted(base, key=lambda p: (p.get("run") or "", p["id"])):
        keys = list(p["options"])
        a = render(p, VER, keys)
        if a is None:
            st["base_dropped:" + p["kind"]] += 1
            continue
        c = render(p, PREV_VER, keys)
        if c is None or json.dumps([a.get("state"), a["question"]], ensure_ascii=False, sort_keys=True) != \
                json.dumps([c.get("state"), c["question"]], ensure_ascii=False, sort_keys=True):
            changed.add(a["id"])
        rows.append(a)
    st["base_kept"], st["base_changed_text"] = len(rows), len(changed)
    print("base 渲染完", len(rows), round(time.time() - t0), "s", flush=True)
    seen = set()
    for p in sorted(online_new(st), key=lambda p: (p["match"], p.get("ts") or 0)):
        if p["id"] in base_ids:
            continue
        keys = list(p["question"]["criteria"])
        a = render(p, VER, keys)
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
        if not (r["kind"] in LAYA_KINDS or r["kind"] == "swap"):
            continue
        t = tags(r)
        if r["id"] in changed:
            t.append("Z_changed")
            if r["kind"] == "carry":
                t.append("E_carry_off")
        tg[r["id"]] = t
    for i, t in tg.items():
        for x in t:
            st[f"tag:{rows_by[i]['split'] == 'train' and 'train' or 'test'}:{x}"] += 1
    tr, te = set(), set()
    per_match = collections.defaultdict(list)   # 845 起新局线上题每局抽 35 道
    for i in sorted(new_set):
        r = rows_by[i]
        if r.get("source") == "offline" or r["kind"] not in LAYA_KINDS + ("swap",) or (r.get("match_no") or 0) < PER_MATCH_FROM:
            continue
        per_match[r["match"]].append(i)
    for m, ids in sorted(per_match.items()):
        rng.shuffle(ids)
        for i in ids[:PER_MATCH_N]:
            (te if rows_by[i]["split"] == TEST_SPLIT else tr).add(i)
    st["per_match_matches"] = len(per_match)
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
