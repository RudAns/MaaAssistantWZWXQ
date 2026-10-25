"""每局（也可以对局中）自动找"一眼就能看出来的蠢操作"（2026-10-13：这类明显错误要能从每局日志里自动发现，
不靠人工逐局翻看）。

和 audit_match.py 的区别：那个看资源曲线 / 错过机会；这个逐条抓**具体的错误动作**，每条带回合、证据原文、该归谁（规则 / 模型 / 题面）：
  B1 模型选了"自己题面写着不该选"的选项（unused → sold / NO ROOM / CANNOT AFFORD / MISMATCH / does NOT fit / WORSE / no effect …）
  B2 买了（或 N 选 1 / 拍卖拿到）又在 2 回合内卖掉 —— 白花能量
  B3 手里压着同名、场上没有、满员，还在买这个人的同名
  B4 装备给错：防御装给主C、伤害类型对不上、适配分 <3 而场上有 ≥5 分的人
  B5 卖掉路线最低配套件 / 路线核心件 / 觉醒过的人
  B11 为新拿到的人腾位卖掉 10 级以上的单位
  B6 枫叶舞绑在非主C 身上超过 1 个回合
  B7 准备阶段结束时：有空位、手里有英雄却没上场
  B8 天赋选了装备类、伤害类型和主C / 路线主C 都对不上

  .venv/Scripts/python.exe -X utf8 tools/blunder_check.py 615            # 按局号（读 data/logs/agent_one_615.log 找 live 目录）
  .venv/Scripts/python.exe -X utf8 tools/blunder_check.py live_1791889657
  .venv/Scripts/python.exe -X utf8 tools/blunder_check.py 615 --follow   # 对局中：每 20 秒重查，只打印新发现（给 Monitor 用）
只读，不改任何文件。
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BAD_MARK = re.compile(r"unused → sold|NO ROOM|CANNOT AFFORD|MISMATCH|does NOT fit|\(WORSE\)|no effect on|NOT enough to level"
                      r"|selling breaks the level engine|MIN ENGINE piece on board|can't afford a hero")


def jl(f: Path) -> list:
    out = []
    if f.exists():
        for line in f.open(encoding="utf-8", errors="replace"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def run_dir(arg: str) -> Path:
    if arg.startswith("live_"):
        return ROOT / "data" / "logs" / arg
    log = ROOT / "data" / "logs" / f"agent_one_{arg}.log"
    m = re.search(r"live_\d+", log.read_text(encoding="utf-8", errors="replace")[:200000]) if log.exists() else None
    if not m:
        raise SystemExit(f"找不到第 {arg} 局的 live 目录")
    return ROOT / "data" / "logs" / m.group(0)


_C = None


def cards() -> dict:
    global _C
    if _C is None:
        c = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))
        c = c if isinstance(c, list) else (c.get("cards") if isinstance(c.get("cards"), (list, dict)) else c)
        c = c if isinstance(c, list) else list(c.values())
        _C = {}
        for x in c:
            if isinstance(x, dict) and x.get("name") and (x["name"] not in _C or x.get("category") == "英雄牌"):
                _C[x["name"]] = x
    return _C


def is_hero(n: str) -> bool:
    return (cards().get(n) or {}).get("category") == "英雄牌"


def min_engine_all() -> set:
    try:
        d = json.loads((ROOT / "data" / "route_min_engine.json").read_text(encoding="utf-8"))
    except Exception:
        return set()
    out = set()
    for v in d.values():
        if isinstance(v, dict):
            out |= set(v.get("early_priority") or [])
    return out


def check(d: Path) -> list[tuple[int, str, str, str]]:
    """[(回合, 类别, 归属, 说明)]"""
    from agent.decision import hero_traits as H
    H.USE_HERO_ITEM_PREF = True
    pts, led, sts = jl(d / "laya_points.jsonl"), jl(d / "ledger.jsonl"), jl(d / "states.jsonl")
    out: list = []

    # ---- 每回合的主C / 枫叶舞 / 路线主C（取自题目 ctx：线上真实值）
    carry_at, maple_at = {}, {}
    for p in pts:
        c, r = p.get("ctx") or {}, p.get("round") or 0
        if c.get("carry"):
            carry_at[r] = c["carry"]
        if c.get("maple"):
            maple_at[r] = c["maple"]

    # B1 模型选了题面自己写着不该选的
    for p in pts:
        if p.get("used") not in ("laya",):
            continue
        ch = p.get("laya_choice") or ""
        txt = (p.get("question") or {}).get("criteria", {}).get(ch, "")
        if p.get("kind") == "swap" and ch.startswith("s"):
            continue                      # 换人题"卖 X"选项天然带"卖了有损失"的事实，不算选错
        m = BAD_MARK.search(txt)
        if m and re.search(r"CORE of route|ROUTE unit|MIN ENGINE piece missing", txt) and m.group(0) not in ("CANNOT AFFORD", "MISMATCH"):
            m = None        # 核心件 / 路线成员：顺带写的"对现任主C 没效果 / 满员"不算选错（第 625 局拍铠）
        if m and ch not in ("stop", "hold", "bench", "keep"):
            out.append((p.get("round") or 0, "B1 选了题面写着不该选的", f"模型[{p.get('kind')}]",
                        f"{ch}（{p.get('laya_conf', 0):.2f}）：{txt[:150]}"))

    # B2 买了又卖（2 回合内）
    got = collections.defaultdict(list)
    for e in led:
        r = e.get("round") or 0
        if e.get("event") == "买牌" and is_hero(e.get("牌", "")):
            got[e["牌"]].append((r, "买"))
        elif e.get("event") == "N选1" and is_hero(e.get("选择", "")):
            got[e["选择"]].append((r, "N选1"))
        elif e.get("event") in ("卖手牌", "卖出"):
            n = e.get("牌") or e.get("英雄") or ""
            hits = [(rr, how) for rr, how in got.get(n, []) if 0 <= r - rr <= 2]
            if hits:
                out.append((r, "B2 买了又卖", "规则/模型", f"{n}：{'、'.join(f'R{rr}{how}' for rr, how in hits)} → R{r} {e['event']}（{e.get('原因', '')[:50]}）"))
                got[n] = [x for x in got[n] if x not in hits[:1]]

    # B3 满员 + 手里压着同名 + 场上没有 → 还买
    _b3_seen: set = set()
    _carryish = set(carry_at.values()) | {((p.get("ctx") or {}).get("route_carry") or "") for p in pts}
    try:
        from agent.decision import routes as _Rb
        _carryish |= {(_Rb.by_key((p.get("ctx") or {}).get("route")) or {}).get("carry") for p in pts if (p.get("ctx") or {}).get("route")}
    except Exception:
        pass
    last_state_by_ts = [(x.get("ts") or 0, x.get("state") or {}) for x in sts]
    for x in sts:
        for a in (x.get("actions") if x.get("executed") else None) or []:
            if not isinstance(a, dict) or a.get("type") != "buy_piece":
                continue
            s = x.get("state") or {}
            n = a.get("piece") or ""
            hand = [h for h in (s.get("hand") or []) if isinstance(h, str)]
            pop, cap = s.get("population"), s.get("population_cap")
            board = set((x.get("board") or {}).keys()) if isinstance(x.get("board"), dict) else set()
            if (is_hero(n) and n in hand and n not in board and isinstance(pop, int) and isinstance(cap, int) and pop >= cap > 0
                    and (s.get("round_no"), n) not in _b3_seen
                    and n not in _carryish):                   # 主C / 路线主C 的同名是合成材料（第 625 局守约）
                _b3_seen.add((s.get("round_no"), n))       # states 里同一次买入会记好几帧，按 (回合, 牌) 去重
                out.append((s.get("round_no") or 0, "B3 满员手里压着同名还买", "模型[shop]", f"{n}（人口 {pop}/{cap}，手里已有）"))

    # B4 装备给错
    for e in led:
        if e.get("event") != "用牌" or e.get("类别") != "装备牌":
            continue
        r, n, t = e.get("round") or 0, e.get("牌", ""), e.get("目标", "")
        if not t or not is_hero(t):
            continue
        carry = carry_at.get(r) or carry_at.get(r - 1)
        cls = (H.EQUIP_CLASS.get(n) or ("",))[0]
        f = H.item_fit_score(n, t)
        why = []
        _late_def = n in ("力量腰带", "霸者重铠", "护命灵珠", "翼苍穹盾", "苍穹王权", "磐石之心") and (H.is_melee(t) or r >= 12 or n == "磐石之心")
        if t == carry and (cls == "tank" or H.base_kind(n) == "defense") and not _late_def:
            why.append("防御装给了主C")           # 近战主C / 后期远程主C 的一件不降攻速防御装是允许的（临时规则 carry_late_defense，10-21）
        if _late_def and t == carry:
            continue
        if H.item_dmg(n) and H.item_mismatch(n, [t]):
            why.append(f"伤害类型不对（{H.item_dmg(n)} 装给 {H.dmg_type(t)}）")
        if isinstance(f, (int, float)) and f < 3:
            why.append(f"适配分 {f:.1f}")
        if why:
            out.append((r, "B4 装备给错", "规则/模型[use_card]", f"{n} → {t}：{'；'.join(why)}"))

    # B5 卖掉最低配套件 / 觉醒过的
    me = min_engine_all()
    try:
        _me_d = json.loads((ROOT / "data" / "route_min_engine.json").read_text(encoding="utf-8"))
    except Exception:
        _me_d = {}
    route_at = {}
    for p in pts:
        if (p.get("ctx") or {}).get("route"):
            route_at[p.get("round") or 0] = p["ctx"]["route"]

    try:            # 2026-10-20（b5-analysis-805-824：70 条里 56 条误报）：同时读 f9v 补表（含 f9w_override），口径同线上 rules.min_engine
        for _k, _v in json.loads((ROOT / "data" / "route_min_engine_f9v.json").read_text(encoding="utf-8")).items():
            if isinstance(_v, dict) and not _k.startswith("_"):
                _me_d[_k] = dict(_v, **(_v.get("f9w_override") or {}))
    except Exception:
        pass

    def _route_me(r):
        # 没锁路线 / 路线没有引擎表 → 不报（原来退回到所有路线 early_priority 的并集 28 人，卖谁都报）
        k = route_at.get(r) or route_at.get(r - 1) or ""
        v = _me_d.get(k) if isinstance(_me_d.get(k), dict) else None
        if not v:
            return set()
        core = {u.get("unit") for u in v.get("core_set") or [] if isinstance(u, dict)}
        return (set(v.get("early_priority") or []) | core) - set(v.get("not_wanted") or [])
    awk = set()
    for x in sts:
        for c, v in ((x.get("state") or {}).get("awaken_progress") or {}).items():
            pass
    for e in led:
        if e.get("event") in ("卖出", "卖手牌"):
            n = e.get("英雄") or e.get("牌") or ""
            if n in _route_me(e.get("round") or 0):
                _why = e.get("原因", "") or ""
                _tag = "B5 卖掉最低配套件" if "枫叶舞换绑" not in _why else "B5' 枫叶舞换绑卖了配套件"
                out.append((e.get("round") or 0, _tag, "规则/模型", f"{n}（{_why[:60]}）"))

    # B11 为新拿到的人腾位卖掉 10 级以上的单位（10-21 第 879 局 R17：为 1 级朵莉亚卖了高等级明世隐，而这个明世隐已攒了很高的等级）
    _got_r = {}
    for e in led:
        if e.get("event") in ("买牌", "获得卡牌", "N选1"):
            _got_r.setdefault(e.get("牌") or e.get("选择") or "", []).append(e.get("round") or 0)
    for p in pts:
        if p.get("kind") != "swap" or p.get("used") == "shadow":
            continue
        ch = p.get("laya_choice") or ""
        names = (p.get("raw") or {}).get("names") or {}
        vic, inc = names.get(ch, ""), (p.get("raw") or {}).get("incoming", "")
        if not vic or not ch.startswith("s") or ch == "bench":
            continue
        txt = json.dumps(p.get("question") or {}, ensure_ascii=False)
        mline = re.search(r"sell " + re.escape(vic) + r":[^\"]*", txt)
        line = mline.group(0) if mline else ""
        mlv = re.search(r"lv ?(\d+)", line)
        lv = int(mlv.group(1)) if mlv else 0
        rd = p.get("round") or 0
        tags = [t for t in ("AWAKENED", "ROUTE member") if t in line]
        if (lv >= 10 or tags) and any(rd - g <= 1 for g in _got_r.get(inc, [])):
            out.append((rd, "B11 为新拿到的人卖高等级 / 觉醒 / 路线成员", "模型/规则[swap]", f"卖 {vic} lv{lv} {'/'.join(tags)} 给刚拿到的 {inc} 腾位"))

    # B6 枫叶舞绑在非主C 身上超过 1 回合
    bad = [r for r in sorted(maple_at) if carry_at.get(r) and maple_at[r] != carry_at[r]]
    runs, cur = [], []
    for r in bad:
        if cur and r == cur[-1] + 1:
            cur.append(r)
        else:
            if cur:
                runs.append(cur)
            cur = [r]
    if cur:
        runs.append(cur)
    for rr in runs:
        if len(rr) >= 2:
            out.append((rr[0], "B6 枫叶舞不在主C 身上", "规则", f"R{rr[0]}~R{rr[-1]}：枫叶舞在 {maple_at[rr[0]]}，主C {carry_at[rr[0]]}"))

    # B7 准备阶段结束时有空位、手里有英雄却没上场（每回合最后一个准备帧，商店关着、人口读数有效）
    lastp = {}
    for x in sts:
        s = x.get("state") or {}
        if s.get("phase") == "prepare" and s.get("population_valid") and not s.get("shop_open"):
            lastp[s.get("round_no") or 0] = s
    for r, s in sorted(lastp.items()):
        pop, cap = s.get("population"), s.get("population_cap")
        heroes = [h for h in (s.get("hand") or []) if isinstance(h, str) and is_hero(h)]
        deployed = {e.get("英雄") for e in led if e.get("event") == "上场" and (e.get("round") or 0) == r}
        heroes = [h for h in heroes if h not in deployed]     # 最后几秒才拖上去的不算（账本"上场"是拖拽回执确认的）
        _late = sum(1 for e in led if e.get("event") == "上场" and (e.get("round") or 0) == r
                    and (e.get("英雄") or "") not in (s.get("hand") or []) )
        if isinstance(pop, int) and isinstance(cap, int) and pop + _late < cap and heroes and (s.get("countdown") or 0) <= 8:
            out.append((r, "B7 有空位手里有英雄没上", "规则", f"人口 {pop}/{cap}，手里 {heroes[:4]}，倒计时 {s.get('countdown')}"))

    # B8 天赋选了装备、伤害类型对不上主C 和路线主C
    for p in pts:
        if p.get("kind") != "talent" or not str(p.get("laya_choice") or "").startswith("opt"):
            continue
        i = int(p["laya_choice"][3:])
        tl = (p.get("raw") or {}).get("talents") or []
        if i >= len(tl):
            continue
        name = tl[i][0]
        c = p.get("ctx") or {}
        heroes = [h for h in (c.get("carry"),) if h]
        if heroes and H.item_dmg(name) and H.item_mismatch(name, heroes):
            out.append((p.get("round") or 0, "B8 天赋装备伤害类型不对", "模型[talent]", f"{name}（{H.item_dmg(name)}）给 {heroes}"))
    # B9 拿 / 买了"给场上等级最低的人加等级"的牌（扶倾济弱 / 扫拂 / 传承）：单主C 阵容给不到主C
    #   2026-10-20 第 844 局：扶倾济弱是给最低等级英雄加等级的，对养主C 的阵容无效（这类要能自动发现）
    _low_re = re.compile(r"(等级最低|最低等级)[^。]*(提升|获得|等级\s*[+＋]|分配)")

    def _feeds_lowest(n: str) -> bool:
        c = cards().get(n) or {}
        return c.get("category") not in ("英雄牌", "召唤物", None) and bool(_low_re.search((c.get("skill") or "").replace("敌方等级最低", "")))
    for e in led:
        ev = e.get("event")
        n = e.get("选择") if ev == "N选1" else (e.get("牌") if ev == "买牌" else None)
        if n and _feeds_lowest(n):
            out.append((e.get("round") or e.get("回合") or 0, "B9 拿了给最低等级英雄的牌", "模型/规则[" + ev + "]",
                        f"{n}：{(cards().get(n) or {}).get('skill', '')[:40]}（主C 不会是最低等级）"))
    # B10 拿 / 买 / 打给主C / 选的牌，在按主C 复审表里标着"对这个主C 无效"（data/review/{effect_cards_carry_a,b,talents_carry,equips_carry}.json，2026-10-20 晚）
    rv = review_tables()
    def _bad_for(n: str, carry: str) -> str:
        v = rv.get(n) or rv.get(n + "(天赋牌)") or rv.get(n + "(装备牌)")
        if not v or not carry:
            return ""
        try:
            if float(v.get("confidence") if v.get("confidence") not in (None, "") else 1) < 0.6:
                return ""
        except (TypeError, ValueError):
            pass                       # 有的表把把握写成"高 / 中"文字：不按把握过滤
        for x in v.get("ineffective_for") or []:
            if isinstance(x, dict) and x.get("carry") == carry:
                return x.get("why") or "无效"
        return ""
    def _gets_core_tier(n: str, carry: str) -> bool:
        """X阶英雄兑换券 / X阶英雄：X 阶里有这个主C 的核心件 → 是给核心件拿的，不算对主C 无效。"""
        import re as _re
        m = _re.search(r"([1-5])阶英雄", n)
        if not m:
            return False
        try:
            pool = json.loads((ROOT / "data" / "lords" / "阿离" / "carry_pool.json").read_text(encoding="utf-8"))
            core = (pool.get("carries", {}).get(carry) or {}).get("core_lineup") or []
            cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
        except (OSError, ValueError, KeyError):
            return False
        return any(int((cards.get(h) or {}).get("quality") or 0) == int(m.group(1)) for h in core if h != carry)
    def _carry(r: int) -> str:
        ks = [k for k in carry_at if k <= r]
        return carry_at[max(ks)] if ks else ""
    for e in led:
        ev, r = e.get("event"), e.get("round") or 0
        n, C = None, _carry(r)
        if ev == "用牌" and e.get("目标") and e.get("目标") == C:
            n = e.get("牌")
        elif ev == "N选1":
            n = e.get("选择")
        elif ev == "买牌" and not is_hero(e.get("牌") or ""):
            n = e.get("牌")
        w = _bad_for(n or "", C)
        if n and w and n in ("唤醒", "集体唤醒", "锦囊") and any("明世隐" in (x.get("英雄"), x.get("牌")) for x in led):
            w = ""                     # 10-21 第 864 局：触发登场的牌给明世隐（转临时等级给主C，规范 54.31-1），复审表只按主C 本人判
        if n and w and ev in ("N选1", "买牌") and _gets_core_tier(n, C):
            w = ""                     # 第 852 局：3阶兑换券对 2 阶守约"拿不到同名"，但能拿 3 阶核心件铠 / 明世隐（复审表只按主C 本人判）
        if n and w and ev == "用牌" and n in ("力量腰带", "霸者重铠", "护命灵珠", "翼苍穹盾", "苍穹王权", "磐石之心") and (H.is_melee(C) or r >= 12 or n == "磐石之心"):
            w = ""                     # 10-21：近战主C / 后期远程主C 的一件不降攻速防御装（临时规则 carry_late_defense）
        if n and w and not _feeds_lowest(n):
            out.append((r, "B10 对主C 无效的牌", "模型/规则[" + ev + "]", f"{n} ⇢ 主C {C}：{w[:60]}"))
    for p in pts:
        if p.get("kind") != "talent" or not str(p.get("laya_choice") or p.get("choice") or "").startswith("opt"):
            continue
        ch = p.get("laya_choice") or p.get("choice")
        tl = (p.get("raw") or {}).get("talents") or []
        i = int(ch[3:])
        if i < len(tl):
            C = (p.get("ctx") or {}).get("carry") or _carry(p.get("round") or 0)
            w = _bad_for(tl[i][0], C)
            if w:
                out.append((p.get("round") or 0, "B10 对主C 无效的牌", "模型[talent]", f"{tl[i][0]} ⇢ 主C {C}：{w[:60]}"))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


_RV = None


def review_tables() -> dict:
    """按主C 复审表合并（2026-10-20 晚 4 个子代理：效果牌两半 / 天赋 / 装备）。"""
    global _RV
    if _RV is None:
        _RV = {}
        for f in ("effect_cards_carry_a", "effect_cards_carry_b", "talents_carry", "equips_carry", "positional_carry", "special_equips_talents_v2"):   # 站位承伤复审最后读、覆盖前面
            try:
                d = json.loads((ROOT / "data" / "review" / f"{f}.json").read_text(encoding="utf-8"))
                _RV.update({k: v for k, v in d.items() if isinstance(v, dict) and not k.startswith("_")})
            except (OSError, ValueError):
                pass
    return _RV


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    follow = "--follow" in sys.argv
    if not args:
        print(__doc__)
        return 1
    d = run_dir(args[0])
    seen: set = set()
    while True:
        found = check(d)
        new = [f for f in found if f not in seen]
        if not follow:
            cnt = collections.Counter(f[1] for f in found)
            print(f"## {d.name}：{len(found)} 条（" + "，".join(f"{k} {v}" for k, v in sorted(cnt.items())) + "）")
        for f in new:
            print(f"R{f[0]:<2} {f[1]}｜{f[2]}｜{f[3]}", flush=True)
            seen.add(f)
        if not follow:
            return 0
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
