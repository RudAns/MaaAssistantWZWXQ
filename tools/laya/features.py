"""把局面和候选项转成 Laya 的输入（英文语义描述）。**离线建数据集和在线影子推理共用这一个模块**，
保证训练和推理看到的是同一种文本（train/serve 一致）。

设计：
- 卡名对模型是专有名词噪声，每张卡用 data/card_features.json 的英文描述（品阶/派系/定位/触发/效果/成长性）；
- 选项文本很短（Laya 所有选项共用 head_max_len=256 token），只放"动作 + 最关键的几个事实"；
  详细描述放进局面文本（最多 max_len=1024 token）；
- 给规则引擎算出的**事实**（已拥有几张、是不是主C、买不买得起…），**不给规则的最终打分**。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURE_VERSION = "f9k"         # f9k（2026-09-29）：官方路线进题面（ROUTE 行/标签、不参选路线不列）+ route 题 + 引擎落点优先、NO ROOM 置前、LOSES 查现任主C。f9j（2026-09-26）：等级引擎（状态行 / 零件标签 / 换人换主C 选项）。f9i（2026-09-26）：swap 题写"JUST deployed this round"。f9h（2026-09-26）：过了前期 1~2 阶核心不压"多余输出"、换主C 写被替者现况、有福同享写同品阶受益人。f9g（2026-09-26）：换主C 题写出现任主C 的来历（since rX / JUST switched / replaced Y / N switches）。f9f（2026-09-26）：1~2 阶主C 过了前期写进 keep、锁店逐张放、建议卖谁不含现任主C。f9e（2026-09-26）：阵容骨架（EXTRA OUTPUT、outputs on board、current carry）+ 锁店写明品阶定位。f9d（2026-09-26）：和主C 的化学反应（with carry）+ 商店 lock 选项。f9c： f9c（2026-09-25 深夜）：没空位时写"场上最弱的是谁"（不再笼统写 only core heroes）。f9b： f9b（2026-09-25 晚）：swap 题"升一级腾位"独立成 levelup 选项（买不起写 CANNOT AFFORD）。f9： f9（2026-09-25 晚）：swap 同名计数、解绑/装备前移、升级 +1 人口、tN 进商店等级、keep 代价、开团数、前排坦克保护。 f8（2026-09-25）：carry/swap 题面修正（代价前移、枫叶舞强化、手里更强候选、觉醒不算后期低价值）。 文本格式一改就升版本号，数据集和 checkpoint 里都记着
# f9l（2026-09-30 夜，r9n 测试集老师 g9k 反馈，FEATURE_VERSION 仍是 f9k —— 线上 R15 与离线默认不变，f9l 只在显式 ACTIVE="f9l" 时生效）：
#   失效路线（第 5 回合起路线主C 及同体系主C 都不在我方）写 ROUTE stale 并去掉全部路线标签，第 5 回合前写 not owned yet；
#   主C 装备满 3/3 时装备改指下一个对口的人（hero_traits.item_fit_score ≥5）；2nd carry candidate × 引擎零件合成一条（零件 / 第二主C 落点）；
#   "获得 N 张 X"写合计效果；满员商店的刷新写"刷完还够不够下回合升级"；swap 的 levelup 买不起写明"什么也不做"。
# f9le（2026-10-09）：f9l + eco 题（见文件末尾 q_eco）。
# f9lf（2026-10-10；"f9le" < "f9lf" < "f9m"，f9m 已预留）：f9le + 三类事实（都在 facts.py 末尾）——
#   商店刷新选项写"刷完买不买得起英雄（3 能量）"；发英雄牌写卡池 / 阵容要的人 / 期望拿到几个（card_graph_v3 grants_pool）；
#   阿离 6 级专属天赋写卡面原文。只在显式 ACTIVE >= "f9lf" 时生效（R17 = f9l、eco = f9le 一字不变）。
# f9zb（2026-10-21 晚；R52 用，ACTIVE = "f9zb" 才生效）：facts.py 末尾 f9zb 段 —— 选项预算超长行写短 + pack_f9zb（关键结论置前、兜底缩写）、
#   刷新行缺件 / 同名分开 + 引擎缺件刷不出写 level up first、换人题明世隐叠层 / 新明世隐 ×剩余回合 / 李元芳 vs 太乙 / 同路线前排互换、
#   集体唤醒定型后 use now、自然法杖只给远程（规范 54.34~54.36）。
# f9lg ~ f9ll：见 facts.py 末尾各版本常量（F9LG…F9LL）。线上 R25b = f9lk，R26 = f9ll（未上线）。
# f9lm（2026-10-13，目标打法；"f9ll" < "f9lm" < "f9m"）：临时主C（current carry X is TEMPORARY / copy of TEMPORARY carry /
#   target is TEMPORARY carry）、候选路线核心卡（candidate route cores 行 + CORE of route R (own k)）、FINAL carry settled、
#   天赋点名派系时 our 【X】 heroes: board a, hand b, candidate-route cores c、卡面带赠送牌效果（given_skill）、
#   空位事实（商店/eco/swap 状态行 empty slots now…、升级选项 after level-up: …can buy n hero(es) to fill k empty slot(s)、
#   英雄选项 fills an empty slot / board full: needs a swap to deploy；来自第 5~6 名复盘）、
#   选项打包放不下的那条跳过（后面短的照放）。全部挂 facts.FV >= F9LM，f9lk / f9ll 渲染逐字不变。
# **实战时生效的版本跟着 checkpoint 走**（laya_policy 按 checkpoint 训练时的版本设置）：
# 线上学生是 f4 训的，就给它 f4 的文本；离线建新数据集用 FEATURE_VERSION。
ACTIVE = FEATURE_VERSION
# f6（2026-09-24）：**整套题面重设计**，由 tools/laya/facts.py 生成 ——
#   "老师花 10 分钟推出来的结论"算好写进题面（合成后觉醒到几/3、效果在我方局面里空不空转、给谁用加到几级、
#   未来主C / 主T / 过渡件身份、以主C 为锚的阵容队友与装备（阿离 ×1.0、其他 ×0.8）、刷新命中、升级收益…）。
#   只用卡面/机制库/阵容/实测规则表 + 感知读数，不用规则引擎的判断。选项结论前置、按真实 token 数打包（≤44）。
#   f6 起 max_len 2048 / head_max_len 512（训练与推理配置同步）。f4/f5 路径一字不动。
# f7（2026-09-25 凌晨，老师 g5 标注反馈 + 第四轮第 1 局）：场上每人伤害类型；装备 fits/does NOT fit 主C、基础装备给主C 铸成什么；
#   闲置英雄手牌；弦音/妙音只列未觉醒候选；变身牌后果与合适对象；可卖对象含"后期低价值"；枫叶舞机制；"高 N 阶"解析；
#   触发整备/登场效果的牌列出对应英雄；拍卖回合写明"战斗后进行、尚未开始"。全部挂在 facts.FV >= "f7" 下。


def _v(ver: str) -> bool:
    return ACTIVE >= ver


# f9lo（2026-10-14 夜）：每个选项最前面写推演器结论（tools/engine：按游戏规则 + 卡面推演 3 回合 / 换主C 到对局结束，
# 估计参数抽样 8 次）。只写"相对"结论：在本题选项里排第几、比最好的差多少、结论稳不稳 —— 绝对值随局面量级变化，没有可比性。
ENGINE_ANNOT: dict = {}          # 题 id -> 推演器打分（建数据集时读 data/engine/annot_*.jsonl；线上 laya_policy 现算放进 p["_engine"]）


def engine_text_on() -> bool:
    """选项前写推演结论：版本号带 "+sim"（在某一版题面上只加这一项，如 R28 的 "f9lm+sim"），或 f9lo 及以后。"""
    return "+sim" in str(ACTIVE) or _v("f9lo")


def load_engine_annot(path) -> int:
    n = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("id") and r.get("opts"):
            ENGINE_ANNOT[r["id"]] = r
            n += 1
    return n


def engine_prefix(criteria: dict, ann: dict | None) -> dict:
    """选项文本前加 "SIM #k[/n] (-gap) sure|close; "。没有推演结果的选项不加。"""
    opts = (ann or {}).get("opts") or {}
    if not opts or not engine_text_on():
        return criteria
    ranked = sorted(opts, key=lambda k: -opts[k]["dV"])
    best = opts[ranked[0]]["dV"]
    out = {}
    for k, txt in criteria.items():
        o = opts.get(k)
        if o is None:
            out[k] = txt
            continue
        r = ranked.index(k) + 1
        gap = best - o["dV"]
        conf = "sure" if o.get("p_best", 0) >= 0.8 else ("likely" if o.get("p_best", 0) >= 0.5 else "")
        tag = f"SIM #{r}" + ("" if r == 1 else f" (-{gap:.0f})") + (f" {conf}" if conf else "")
        out[k] = f"{tag}; {txt}"
    return out
# f3（2026-09-24）：按老师（Claude 标注员）在 36 个批次里一致指出的"缺判断依据"补字段 ——
#   名次/各家血量、棋手经验、免费刷新次数、本回合拍卖是否已进行、画面上看到的单位数与等级、手牌张数、
#   读数可信度（血量>60/等级与商店品阶矛盾 → 标出来而不是当真）；商店里同名牌只保留一个选项（等价选项）。
# f4（2026-09-24）：① 选项文本带上效果要点（f3 的 N 选 1 选项只有 "effect t5" 这种标签，效果描述只在局面文本里，
#   学生要自己对号 —— N 选 1 是 f3 学得最差的一类）；② 卡名→"自身"只对英雄牌做（「5阶英雄」的效果
#   "获得2张5阶英雄牌"曾被渲染成"获得2张自身牌"）；③ 等级显示原始读数 + "商店最高品阶"，不再替读者做修正
#   （老师反映修正后的等级与人口上限矛盾）。
# f5（2026-09-24：尽可能多的信息交给 Laya，它的判断和老师的判断才能拟合；
#   购买和升级策略要按玩法规则和商店刷新概率判断，而不是定死；目标是 Laya 而不是完美的规则引擎）：
#   ① 场上每人：觉醒进度 k/3（等级徽章外圈，视觉）与还差几张、是否留到最后（keep）、是否推荐阵容的后期角色（late）；
#   ② 当前等级 / 升一级后的各品阶刷新概率（meta.TIER_ODDS，实测表）；
#   ③ 刷新选项：刷一次看到"还差张数的留场英雄"同名牌的概率（meta.shop_hit_prob，按公共牌池结构算）；
#   ④ 升级选项：升级后哪些品阶概率变了、人口上限涨不涨。
#   场上信息实战与离线同源：rules.board_facts() 每帧写进 states.jsonl 的 "board"，离线直接读它。
#   ⑤ 去掉血量和按血量排的名次（玩家血量不参与判断：输了就是输了）。

FACTION_EN = {"逐鹿": "zhulu", "河洛": "heluo", "三分": "sanfen", "三分之地": "sanfen", "大河": "dahe", "大河流域": "dahe",
              "日落海": "riluohai", "无阵营": "neutral", "": "none"}


@lru_cache(maxsize=1)
def _feats() -> dict:
    return json.loads((PROJECT_DIR / "data" / "card_features.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _cards() -> dict:
    return json.loads((PROJECT_DIR / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]


def card_text_zh(name: str, fallback: str = "", limit: int = 90) -> str:
    """卡面效果原文（中文，**不带卡名**，卡名里出现的自身名字换成"自身"）。
    f2 起加入：只有英文标签时，N 选 1 的 "effect t3"、棋手特性的三个 "maple dance" 根本分不开（2026-09-23 抽样发现）。
    多语言编码器读得懂中文效果描述；"中文是噪声"指的是卡名这种专有名词。"""
    c = _cards().get(name) or {}
    txt = c.get("skill") or fallback or ""
    if name and c.get("category") == "英雄牌":          # f4：只有英雄牌的效果里会用自己的名字指代自己
        txt = txt.replace(name, "自身")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:limit]


def card_en(name: str, fallback: str = "", short: bool = False) -> str:
    """一张牌的英文描述。卡库里没有就用画面上的中文描述（多语言编码器读得懂中文）。"""
    f = _feats().get(name)
    if not f:
        return ("unknown card: " + fallback[:40]) if fallback else "unknown card"
    en = f.get("en", "")
    if short:
        # "hero t4 neutral frontline melee | grow 5 via ... | ..." -> 只留第一段 + 成长分
        head = en.split("|")[0].strip()
        if f.get("grow_score") is not None:
            head += f" grow{f['grow_score']}"
        return head
    return en


def card_tier(name: str) -> int:
    return int((_feats().get(name) or {}).get("tier") or 0)


def card_category(name: str) -> str:
    return (_feats().get(name) or {}).get("category", "")


@lru_cache(maxsize=1)
def _carry_scores() -> dict:
    """主C 适合度评分（0~9，data/hero_carry_reviewed.json，主 agent 逐个读技能写的）。
    f5 起放进学生输入：老师标注规范第 13 节让老师按它判断，学生看不到就违反信息对齐（r3f5/test_g4 老师反馈）。"""
    p = PROJECT_DIR / "data" / "hero_carry_reviewed.json"
    try:
        return {k: v.get("carry_score") for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    except Exception:                                   # noqa: BLE001
        return {}


def carry_fit(name: str) -> str:
    cs = _carry_scores().get(name)
    return f" carry-fit {cs}/9" if (cs is not None and _v("f5")) else ""


def _ban(b: str) -> str:
    """f9k：禁用派系读数规范成库名（「大河口」原来渲染成 other）。旧版本照旧，保证重渲染逐字一致。"""
    if not _v("f9k"):
        return b
    import sys as _sys
    if str(PROJECT_DIR) not in _sys.path:
        _sys.path.insert(0, str(PROJECT_DIR))
    from agent.decision.routes import norm_banned
    return norm_banned(b)


def _faction(s: str) -> str:
    return FACTION_EN.get(s or "", "other")


def _meta():
    """规则常量（品阶概率表 / 牌池结构）。任何入口（build_dataset / label_batches / laya_policy）都要拿得到 ——
    拿不到时**不能静默返回空**（label_batches 曾因此导出了空的概率行），所以先保证项目根目录在 sys.path 里。"""
    import sys as _sys
    if str(PROJECT_DIR) not in _sys.path:
        _sys.path.insert(0, str(PROJECT_DIR))
    from agent.decision import meta
    return meta


def _level(ctx: dict) -> int:
    """算概率用的等级：**用修正后的等级**（商店里出现了 N 阶牌 → 至少 N 级，比读数可信）。
    老师在 r3f5 标注里指出：修正后等级 4、概率行却按读数 3 算，自相矛盾。"""
    lv = ctx.get("level")
    return lv if isinstance(lv, int) and 1 <= lv <= 6 else 0


def _odds_str(level: int) -> str:
    m = _meta()
    if not m or not level:
        return ""
    od = m.TIER_ODDS.get(level) or {}
    return ", ".join(f"t{t} {round(p * 100)}%" for t, p in sorted(od.items()) if p > 0)


def _board_unit(n: str, v: dict) -> str:
    s = card_en(n, short=True)
    if _v("f5"):                       # f4 训练数据的场上只有名字（账本还原），f4 学生不能看到等级 —— 否则训练/实战不一致
        s += carry_fit(n)
        s += f" lv{v.get('lv')}" if v.get("lv") else ""
        left = v.get("awk_left")
        if left is not None:
            s += " awakened" if left == 0 else f" awaken {3 - left}/3 (needs {left} more copies)"
        if v.get("pool_left") is not None:
            s += f" (pool {v['pool_left']} left)"
        tags = [t for t, k in (("keep to end", "keep"), ("late-game lineup role", "late")) if v.get(k)]
        if tags:
            s += " [" + ", ".join(tags) + "]"
    return s


def wanted_copies(ctx: dict) -> list[tuple[str, int | None]]:
    """f5：要留到最后、还没觉醒的场上英雄 -> [(卡名, 还差几张 或 None=读不到)]。"""
    return [(n, v.get("awk_left")) for n, v in (ctx.get("field") or {}).items()
            if v.get("keep") and v.get("awk_left") != 0 and card_category(n) == "hero"][:6]


def _facts():
    import sys as _sys
    here = str(Path(__file__).resolve().parent)
    if here not in _sys.path:
        _sys.path.insert(0, here)
    import facts
    facts.FV = ACTIVE          # 事实引擎按当前生效的特征版本渲染（f7 起的新事实对 f6 学生不可见）
    try:
        facts._ht().LEGACY_DMG = ACTIVE < "f9o"     # f9o：人工确认的伤害类型（亚连 = 物理）只对 f9o 起的题面生效
    except Exception:  # noqa: BLE001
        pass
    return facts


def state_text(ctx: dict, extra_cards: list[str] | None = None, kind: str = "") -> str:
    """局面描述。ctx 字段（都可缺省）：
    round, hp, gold, level, level_cost, pop, pop_cap, banned, hand[list], field{name: {"lv":..}},
    carry, maple, unit_levels(list of int), auction_in
    extra_cards：候选项里出现的牌，**详细描述**放在这里（选项本身只写短描述）。
    """
    if _v("f6"):
        return _facts().state_block(ctx, list(extra_cards or []), kind)
    lines = []
    # f5 起不给血量、也不给按血量排的名次（2026-09-24：玩家血量不参与判断，输了就是输了）
    hp_part = "" if _v("f5") else f"hp {ctx.get('hp') or '?'}/60; "
    q = lambda k: "?" if ctx.get(k) is None else ctx.get(k)          # noqa: E731  None 与缺失一样显示 ?
    lines.append(f"round {ctx.get('round') or '?'}; {hp_part}energy {q('gold')}; "
                 f"lord level {ctx['level_raw'] if ctx.get('level_raw') is not None else q('level')}"
                 + (f" (shop shows tier-{ctx['shop_max_tier']} heroes)" if ctx.get("level_fixed") else "")
                 + (f" (level-up cost {ctx['level_cost']})" if (ctx.get('level_cost') or -1) > 0 else "")
                 + (f"; lord xp {ctx['xp']}/{ctx['xp_needed']}" if (ctx.get('xp_needed') or 0) > 0 else "")
                 + f"; population {q('pop')}/{q('pop_cap')}"
                 + (f"; banned faction {_faction(_ban(ctx.get('banned', '')))}" if ctx.get("banned") else ""))
    if _v("f5") and _level(ctx):
        lines.append(f"shop tier odds now: {_odds_str(_level(ctx))}"
                     + (f"; after level-up: {_odds_str(_level(ctx) + 1)}" if _level(ctx) < 6 else ""))
    extra = []
    if ctx.get("rank") and not _v("f5"):
        extra.append(f"standing {ctx['rank']}/{ctx.get('alive', 6)} by hp")
    if ctx.get("free_refresh"):
        extra.append(f"free refreshes {ctx['free_refresh']}")
    if ctx.get("hand") is not None:
        extra.append(f"hand size {len(ctx.get('hand') or [])}")
    if ctx.get("units_seen") is not None:
        extra.append(f"units seen on board {ctx['units_seen']}"
                     + (f" (levels {', '.join(str(x) for x in ctx['unit_levels'])})" if ctx.get("unit_levels") else ""))
    if extra:
        lines.append("; ".join(extra))
    if ctx.get("auction_in") is not None:
        if ctx["auction_in"] == 0:
            lines.append("auction round: " + ("auction already done this round" if ctx.get("auction_done")
                                              else "auction not yet held this round"))
        else:
            lines.append(f"next auction in {ctx['auction_in']} rounds")
    carry = ctx.get("carry")
    if carry:
        lv = (ctx.get("field") or {}).get(carry, {}).get("lv") if _v("f5") else None
        lines.append(f"main carry: {card_en(carry, short=True)}" + (f" level {lv}" if lv else ""))
    else:
        lines.append("main carry: not decided")
    field = ctx.get("field") or {}
    if field:
        parts = []
        for n, v in field.items():
            parts.append(_board_unit(n, v))
        lines.append("board: " + "; ".join(parts))

    hand = ctx.get("hand") or []
    if hand:
        cnt: dict = {}
        for n in hand:
            cnt[n] = cnt.get(n, 0) + 1
        awk = set(ctx.get("hand_awk") or []) if _v("f5") else set()
        lines.append("hand: " + "; ".join(card_en(n, short=True) + (f" x{c}" if c > 1 else "")
                                          + (" [AWAKENED, deploy first]" if n in awk else "")
                                          for n, c in cnt.items()))
    if extra_cards:
        lines.append("candidate details:")
        for i, item in enumerate(extra_cards):
            n, desc = (item if isinstance(item, tuple) else (item, ""))
            zh = card_text_zh(n, fallback=desc)
            lines.append(f"  [{i}] {card_en(n, fallback=desc)}" + (f" || {zh}" if zh else ""))
    return "\n".join(lines)


def owned_copies(name: str, ctx: dict) -> int:
    return (ctx.get("hand") or []).count(name) + (1 if name in (ctx.get("field") or {}) else 0)


def card_gist(name: str, fallback: str = "", limit: int = 70) -> str:
    """选项里用的效果要点（f4）：英文描述里"|"之后的效果标签，截短。英雄的成长方式也在这里。"""
    f = _feats().get(name)
    if not f:
        return ("unknown: " + fallback[:30]) if fallback else ""
    parts = [x.strip() for x in f.get("en", "").split("|")[1:] if x.strip()]
    return "; ".join(parts)[:limit].rstrip("; ")


def option_card(name: str, ctx: dict, prefix: str = "") -> str:
    """一个"选这张牌"的选项：短描述 + 效果要点 + 事实（已有几张 / 是否主C 同名）。"""
    facts = []
    c = owned_copies(name, ctx)
    if c:
        facts.append(f"owned {c}")
    if name and name == ctx.get("carry"):
        facts.append("same as carry")
    v = (ctx.get("field") or {}).get(name)
    if _v("f5") and v is not None:
        left = v.get("awk_left")
        facts.append("on board" + ("" if left is None else " already awakened" if left == 0
                                   else f", awaken needs {left} more"))
        if v.get("keep"):
            facts.append("keep to end")
        if v.get("late"):
            facts.append("late-game lineup role")
    gist = card_gist(name) if FEATURE_VERSION >= "f4" else ""
    return (prefix + card_en(name, short=True) + carry_fit(name) + (" | " + gist if gist else "")
            + (" | " + ", ".join(facts) if facts else "")).strip()


# ---------------- 各决策类型的"问题"（Laya typed question）----------------
# 选项键是**稳定的英文标识**（opt0/opt1/...），选项描述放在 criteria 的值里 —— Laya 渲染成 "opt0: <描述>"。

F6_INS = {
    "card_pick": "Auto-chess (lord Ali): pick ONE free card that most improves final placement. "
                 "Each option starts with what it does on THIS board; details in the state.",
    "talent": "Auto-chess (lord Ali): the lord levelled up; choose ONE talent that most improves final placement "
              "(or refresh the offer once).",
    "lord_boost": "Auto-chess (lord Ali): choose ONE upgrade for the lord skill maple dance.",
    "shop": "Auto-chess (lord Ali) shop turn: choose the next action that most improves final placement.",
}


def q_card_pick(options: list[str], ctx: dict) -> dict:
    if _v("f6"):
        X = _facts()
        if _v("f9w") and options and all(X.cat(n) == "装备牌" for n in options):
            ctx = dict(ctx, _equip_pick=True)        # f9w（54.25-5）：N 选 1 全是装备（珍品宝箱等）→ 每件写主C 件数 + 输出 / 防御
        B = X.Board(ctx)
        return {"type": "choice", "instructions": F6_INS["card_pick"],
                "criteria": {f"opt{i}": X.option_text(n, ctx, B=B) for i, n in enumerate(options)}}
    return {"type": "choice",
            "instructions": "Auto-chess: pick exactly one of the offered free cards. Prefer what makes the main carry "
                            "stronger or grows levels; consider the board and hand.",
            "criteria": {f"opt{i}": option_card(n, ctx) for i, n in enumerate(options)}}


def q_use_card(name: str, ctx: dict) -> dict | None:
    """特殊效果牌用在谁 / 先留着（f7 起；影子模式，2026-09-25）。没有合法目标时仍出题（hold 之外无选项则返回 None）。"""
    X = _facts()
    B = X.Board(ctx)
    tg = X.use_card_targets(name, B)
    if not tg:
        return None
    crit = {"hold": X.pack(X.use_card_hold(name, B), [])}
    for i, t in enumerate(tg):
        crit[f"use{i}"] = X.pack(f"use on {t}: ", X.use_on_facts(name, t, B))
    return {"type": "choice",
            "instructions": f"Auto-chess (lord Ali): use the card {name} on which hero now, or keep it?",
            "criteria": crit, "_targets": tg}


def q_use_timing(name: str, ctx: dict) -> dict | None:
    """f9lt：不选目标、只问时机的牌（契约瓶 / 众星闪耀 / 险中求富…）现在用还是先留着（影子题，2026-10-15 起：替换写死的规则）。"""
    if not _v("f9lt"):
        return None
    X = _facts()
    B = X.Board(ctx)
    now, hold = X.timing_parts(name, B)
    return {"type": "choice",
            "instructions": f"Auto-chess (lord Ali): use the card {name} now, or keep holding it?",
            "criteria": {"now": X.pack(f"use {name} now: ", now), "hold": X.pack("keep holding: ", hold)}}


def q_carry(ctx: dict) -> dict | None:
    """谁当主C（影子题，2026-09-25 起：出足够多的题，让 Laya 决定是否换主C）。选项 keep（当前主C）+ c1.. 其他候选。"""
    X = _facts()
    B = X.Board(ctx)
    cur = ctx.get("carry") or ""
    maple = ctx.get("maple") or ""
    cands = X.carry_candidates(B, cur)
    if len(cands) < 2:
        return None
    crit, names = {}, {}
    for i, n in enumerate(cands):
        k = "keep" if n == cur else f"c{i}"
        names[k] = n
        crit[k] = X.pack(("keep " if n == cur else "switch to ") + n + ": ", X.carry_option(n, B, cur, maple))
    return {"type": "choice",
            "instructions": "Auto-chess (lord Ali): who should be the CARRY from now on (gets maple dance, items and "
                            "level-up cards)? " + X.carry_state_extra(B),
            "criteria": crit, "_names": names}


def q_route(ctx: dict) -> dict | None:
    """f9k：选哪条官方路线（还没锁定时每回合问一次）。选项 r1..r3（routes.candidates 的前 3 条）+ flex（先不定）；
    _keys 记每个 rN 对应的阵容 key。"""
    X = _facts()
    B = X.Board(ctx)
    if B.route:
        return None
    cands = X.route_candidates(B, banned=ctx.get("banned") or "")
    if not cands:
        return None
    crit, keys = {}, {}
    for i, lu in enumerate(cands, 1):
        k = f"r{i}"
        keys[k] = str(lu.get("key"))
        crit[k] = X.pack(f"{lu.get('name')}: ", X.route_option(lu, B))
    crit["flex"] = "stay flexible: no route yet, keep buying the best units and ask again next round"
    return {"type": "choice",
            "instructions": "Auto-chess (lord Ali): commit to one official lineup ROUTE now, or stay flexible? "
                            + X.route_state_extra(B),
            "criteria": crit, "_keys": keys}


def q_swap(ctx: dict) -> dict | None:
    """人口满、手里有英雄等着上场：卖场上谁给它腾位，还是先不换（bench）。选项 bench + sN；_names 记每个 sN 对应的英雄。"""
    X = _facts()
    B = X.Board(ctx)
    if B.free is None or B.free > 0:
        return None
    x = X.swap_incoming(B)
    cur = ctx.get("carry") or ""
    cands = X.swap_candidates(B, cur)
    if not x or not cands:
        return None
    maple = ctx.get("maple") or ""
    bench = [f"{x} stays in hand (hand {len(B.hand)}/10)", "sell or deploy it later"]
    if _v("f9ln"):
        # f9ln（2026-10-13 第 615 局：程咬金是沈梦溪路线的最低配套件，"不换"这一项只写了"留在手里"，网络两回合都选不换，规则随后把 3 张都卖了）：
        # 写出等位置的人在路线里的角色（事实），和 sN 那边"卖了会断引擎"对称
        role = X.min_engine_tag(x, B, False)
        if not role and x in ((getattr(B, "min_eng", None) or {}).get("early_priority") or []):
            role = "route early-priority piece (level engine, deploy early)"
        if not role and B.route and not B.route_stale and x in X._routes().final_team(B.route):
            role = "ROUTE member"
        _skip_eng = _v("f9m") and ((B.route and not B.route_stale and x not in X._routes().members(B.route)
                                    and "sacrifice" in str(B.engine.get("key") if B.engine else ""))
                                   or B.destroys_own_totems(x))     # f9m：路线外牺牲件 / 拆本方图腾的不算引擎件（r9zq 老师 000/013/017/026）
        if not role and _v("f9lz") and not _skip_eng and B.engine and B.engine_role(x) in ("producer", "amplifier"):
            # f9lz（第 681 局 R14：雅典娜 = 日落海·核心 放大器在等位，bench 行什么都没写，网络 bench 0.53；正确做法是卖廉颇换雅典娜）
            role = f"ENGINE {B.engine_role(x)} of the carry's level engine"
            if _v("f9p") and B.engine_role(x) == "producer":
                # f9p（r9zs 老师多批：吕布 / 嬴政卡面不给等级，马可波罗 / 亚瑟只有登场一次，莱西奥要卖 2 人才 +1）
                pk = X.producer_kind(x)
                role = ("" if not pk else role if pk == "per-round"
                        else f"ENGINE producer ({'one-time 登场 only' if pk == 'one-time' else 'conditional'}) of the carry's level engine")
        if role:
            bench.insert(0, f"waiting {x}: {role}")
    lvl_opt = None
    if _v("f9b"):
        # f9b（版本号不能叫 f10：字符串比较 "f10" < "f8"）：升一级腾位是**独立选项**（老师 r9：bench 混了"升级后上场"和"真的不换"，而线上 bench 什么也不做）
        m = X.meta()
        cost, gold = ctx.get("level_cost"), ctx.get("gold")
        if B.level and B.level < 6 and m.level_raises_pop(B.level) and isinstance(cost, int) and cost > 0:
            afford = isinstance(gold, int) and gold >= cost
            if _v("f9l") and not afford:
                # f9l（老师：只剩 "levelup: CANNOT AFFORD" 时含义模糊）：买不起 = 这一项什么也不做，和 keep the board 一样
                lvl_opt = X.pack(f"level up ({cost}e): ", [f"CANNOT AFFORD (have {gold}e) → does nothing, same as keep the board"])
            else:
                lvl_opt = X.pack(f"level up ({cost}e): ", ["+1 slot, deploy it without selling" if afford else "CANNOT AFFORD",
                                                           f"energy {gold}→{gold - cost}" if afford else f"energy {gold}"])
    elif _v("f9"):
        # f9：升一级就多一个位置时写出来（老师 aug：好几题"升级 1e 就能免费上场"，题面没说）
        m = X.meta()
        cost = ctx.get("level_cost")
        if B.level and B.level < 6 and m.level_raises_pop(B.level):
            bench = [f"level-up ({cost}e) gives +1 slot" if isinstance(cost, int) and cost > 0 else "level-up gives +1 slot"] + bench
    crit = {"bench": X.pack("keep the board: ", bench)}
    names = {"bench": ""}
    if lvl_opt:
        crit["levelup"], names["levelup"] = lvl_opt, ""
    if _v("f9lv"):
        B.ctx["_swap_in"] = x          # f9lv：swap_option 要知道等上场的是谁（前排数会不会因换人而减少）
    for i, n in enumerate(cands):
        crit[f"s{i}"] = X.pack(f"sell {n}: ", X.swap_option(n, B, maple))
        names[f"s{i}"] = n
    return {"type": "choice",
            "instructions": "Auto-chess (lord Ali): population FULL. Hero waiting in hand: " + X.swap_head(x, B)
                            + ". Sell one unit on board to deploy it now, or keep the board?"
                            + (f" (current carry {cur} is not an option)" if cur else ""),
            "criteria": crit, "_names": names, "_incoming": x}


def q_sell_filler(ctx: dict) -> dict | None:
    """f9lw（2026-10-16：弈星升一次棋手等级就回本，可以直接卖掉腾位子）：没人等位时，场上回本的弈星 / 过了阶段的开局件 /
    别家主C 卖不卖。选项 keep + sN；_names 记每个 sN 的英雄。执行 = sell_unit（规则不判值不值）。"""
    if not _v("f9lw"):
        return None
    X = _facts()
    B = X.Board(ctx)
    cur = ctx.get("carry") or ""
    if X.swap_incoming(B):
        return None          # 老师 r9zn/012：手里有人等位时是 swap 题，不是 sell_filler
    cands = X.sell_filler_candidates(B, cur)
    if not cands:
        return None
    maple = ctx.get("maple") or ""
    free = B.free if isinstance(B.free, int) else "?"
    crit = {"keep": X.pack("keep the board: ", [f"empty slots now {free}", f"energy {ctx.get('gold')}"])}
    names = {"keep": ""}
    for i, n in enumerate(cands):
        crit[f"s{i}"] = X.pack(f"sell {n}: ", X.swap_option(n, B, maple))
        names[f"s{i}"] = n
    return {"type": "choice",
            "instructions": "Auto-chess (lord Ali): nobody is waiting for a slot. Sell one of these units now to free the slot "
                            "(and get energy), or keep the board?" + (f" (current carry {cur} is not an option)" if cur else ""),
            "criteria": crit, "_names": names}


def q_sell_hand(ctx: dict) -> dict | None:
    """f9lx sell_hand：手里的英雄牌卖不卖（keep + sN）。每张写商店同口径的事实（别家主C / 不在路线 / 路线件 / 喂主C …）。
    2026-10-16：已经换不过去的别家主C（如海月）不该留；路线不需要的英雄要及时卖（此前多次出现）。"""
    if not _v("f9lx"):
        return None
    X = _facts()
    B = X.Board(ctx)
    cur = ctx.get("carry") or ""
    cands = X.sell_hand_candidates(B, cur)
    if not cands:
        return None
    free = B.free if isinstance(B.free, int) else "?"
    crit = {"keep": X.pack("keep them in hand: ", [f"hand {len(B.hand)}/10", f"empty slots now {free}"])}
    names = {"keep": ""}
    team, stage = set(), set()
    if _v("f9ly") and B.route and not B.route_stale:
        team = set(X._routes().final_team(B.route))
        stage = set(X._routes().members(B.route)) - team
    for i, n in enumerate(cands):
        t = X.option_text(n, ctx, B=B, prefix="sell from hand: ")
        if _v("f9ly") and B.route and not B.route_stale:
            # f9ly（第 671 局：花木兰 ×2、武则天 ×2 压在手里不卖 —— "not in ROUTE" 在详情里被截掉，选项只看到"同一套打法"）：
            # 路线身份紧跟卡名写（和别家主C 标签同一位置）
            if _v("f9x") and B.route_not_wanted(n):
                tag = "not wanted on this route"          # f9x（54.26-2）：狄仁杰线钟馗
            elif n in team:
                tag = "ROUTE final team"
            elif _v("f9x") and n in stage and B.route_sub_only(n):
                tag = X.SUB_ONLY_F9X                     # f9x（54.26-12）：计划外的可选替补
            elif n in stage:
                tag = "ROUTE stage unit"
            elif _v("f9z") and B.route_core_engine(n):
                tag = "ROUTE core engine piece"            # f9z（e91219caffea：马可波罗线明世隐写 not in ROUTE but feeds the carry）：引擎件先于"喂主C"
            elif B.feeds_or_part(n):
                tag = "not in ROUTE but feeds the carry"
            elif _v("f9y") and B.route_core_engine(n):
                tag = "ROUTE core engine piece"            # f9y：引擎表 early_priority（日落海雅典娜）
            elif _v("f9y") and (B.anch["mates"].get(n) or 0) >= 0.8 and n != B.anchor and not B.other_lineup_carry(n):
                # f9y（r9zzd 43a5ce16a24c：露娜在马可波罗阵容 100% 却写 no slot, never fed）：同 f9o 的 identity 口径
                tag = f"in {round(B.anch['mates'][n] * 100)}% of {B.anchor} lineups (not in ROUTE list)"
            else:
                tag = "not in ROUTE: no slot, never fed"
            head = f"sell from hand: {n}: "
            if t.startswith(head) and tag not in t:
                t = head + tag + "; " + t[len(head):]
        crit[f"s{i}"] = t
        names[f"s{i}"] = n
    return {"type": "choice",
            "instructions": "Auto-chess (lord Ali): hero cards are sitting in your hand. Sell one of them now for energy, or keep them?",
            "criteria": crit, "_names": names}


def q_auction(lots: list[tuple[int, str]], ctx: dict, bid_cap: int = 5) -> dict | None:
    """拍卖（2026-09-25 起交给 Laya）：争哪一件，或不拍。lots = [(拍品序号, 卡名)]。
    出价机械（追价 / 同价先到先得 / 能量底线）仍由规则做；学生只选目标。"""
    X = _facts()
    B = X.Board(ctx)
    lots = [(i, n) for i, n in lots if n]
    if len(lots) < 2:
        return None
    m = X.meta()
    g = ctx.get("gold")
    inc = m.energy_income((ctx.get("round") or 0) + 1) if ctx.get("round") else None
    crit = {f"lot{i}": X.option_text(n, ctx, prefix="bid for ", B=B) for i, n in lots}
    crit["pass"] = X.pack("pass: ", [f"keep energy for next round's shop ({g}e now" + (f" +{inc}" if inc else "") + ")"])
    return {"type": "choice",
            "instructions": ("Auto-chess (lord Ali): AUCTION — each player wins at most ONE lot, highest bid wins; "
                             f"our bids go up to ~{bid_cap}e. Which lot should we fight for (or pass)?"),
            "criteria": crit}


def q_talent(options: list[tuple[str, str]], ctx: dict, can_refresh: bool) -> dict:
    if _v("f6"):
        X = _facts()
        if _v("f9lu"):
            ctx = dict(ctx, _talent_q=True)          # f9lu：装备类天赋（魂祭绝刃）卡库里是装备牌，也要写"现在值多少"
        B = X.Board(ctx)
        crit = {f"opt{i}": X.option_text(n, ctx, desc=d, B=B) for i, (n, d) in enumerate(options)}
        if can_refresh:
            crit["refresh"] = "refresh: see three other talents (once per level)"
            if _v("f9za"):
                # f9za（r9zzf 老师 / 54.28-25）：天赋刷新整局只有 1 次；第 8 回合前（青铜天赋）不刷
                _sp = X.f9za_special_all_bad([n for n, _d in options], B)
                crit["refresh"] = ("refresh: see three other talents (once per match)"
                                   + (f"; {_sp}: allowed any round" if _sp
                                      else ("; NOT allowed before round 8" if (ctx.get("round") or 0) < 8 else "")))
        return {"type": "choice", "instructions": F6_INS["talent"], "criteria": crit}
    crit = {f"opt{i}": (card_en(n, fallback=d, short=False)[:120]) for i, (n, d) in enumerate(options)}
    if can_refresh:
        crit["refresh"] = "refresh the three talents once"
    return {"type": "choice",
            "instructions": "Auto-chess: the lord levelled up; choose one talent (or refresh the offer).",
            "criteria": crit}


def q_lord_boost(options: list[tuple[str, str]], ctx: dict) -> dict:
    if _v("f6"):
        X = _facts()
        B = X.Board(ctx)
        return {"type": "choice", "instructions": F6_INS["lord_boost"],
                "criteria": {f"opt{i}": X.option_text(n, ctx, desc=d, kind="lord_boost", B=B)
                             for i, (n, d) in enumerate(options)}}
    return {"type": "choice",
            "instructions": "Auto-chess: choose one upgrade for the lord skill (maple dance).",
            "criteria": {f"opt{i}": (card_en(n, fallback=d)[:100]) for i, (n, d) in enumerate(options)}}


def q_shop(shop: list[tuple[int, str, int]], ctx: dict, can_refresh: bool, can_level: bool) -> dict:
    """商店行动：买某张 / 刷新 / 升级 / 不买了。shop = [(槽位, 卡名, 价格)]（只含买得起的未售出牌）。"""
    if _v("f6"):
        X = _facts()
        ctx["_shop_all"] = [n for _, n, _ in shop]      # 同名汇总用（state_text 随后用同一个 ctx，看到的一样）
        if _v("f9i"):
            ctx["_shop_q"] = True                        # f9i：商店题的买牌选项也写 NO ROOM（option_text 不带 kind）
        B = X.Board(ctx)
        g = ctx.get("gold")
        crit = {f"buy{slot}": X.option_text(name, ctx, B=B, prefix=f"buy ({cost}e" + (
                    f", {g - cost}e left) " if isinstance(g, int) and isinstance(cost, int) else ") "))
                for slot, name, cost in dedupe_shop(shop)}
        if can_refresh:
            crit["refresh"] = X.pack(*X.refresh_parts(B))
        if can_level:
            crit["levelup"] = X.pack(*X.levelup_parts(B))
        crit["stop"] = X.pack(X.stop_facts(B), [])
        if _v("f9d") and ctx.get("_shop_full") is not None and ctx.get("_lock_ok", True):
            crit["lock"] = X.pack(*X.lock_parts(B, ctx["_shop_full"]))     # f9d：锁店也交给学生
        return {"type": "choice", "instructions": F6_INS["shop"], "criteria": crit}
    crit = {}
    for slot, name, cost in dedupe_shop(shop):
        crit[f"buy{slot}"] = option_card(name, ctx, prefix=f"buy ({cost}e) ")
    m, lv = _meta(), _level(ctx)
    if can_refresh:
        crit["refresh"] = "refresh the shop" + (" (free)" if ctx.get("free_refresh") else "")
        if _v("f5") and m and lv:
            want = wanted_copies(ctx)
            fld = ctx.get("field") or {}
            ps = [(n, m.shop_hit_prob(card_tier(n), lv, remaining=(fld.get(n) or {}).get("pool_left")))
                  for n, _ in want if card_tier(n)]
            if ps:
                anyp = 1.0
                for _, p in ps:
                    anyp *= (1 - p)
                crit["refresh"] += (" | chance per refresh to see a copy of a keeper still short of awakening: "
                                    + ", ".join(f"t{card_tier(n)} {round(p * 100)}%" for n, p in ps[:4])
                                    + f"; any {round((1 - anyp) * 100)}%")
    if can_level:
        lc = ctx.get("level_cost") or -1
        crit["levelup"] = f"level up the lord ({lc}e)" if lc > 0 else "level up the lord (cost unread)"
        if _v("f5") and m and 0 < lv < 6:
            a, b = m.TIER_ODDS.get(lv) or {}, m.TIER_ODDS.get(lv + 1) or {}
            ts = sorted((t for t in set(a) | set(b) if abs(a.get(t, 0) - b.get(t, 0)) >= 0.01),
                        key=lambda t: -abs(a.get(t, 0) - b.get(t, 0)))[:4]      # 变化最大的几档（新开的品阶排前面）
            crit["levelup"] += " | odds " + ", ".join(
                f"t{t} {round(a.get(t, 0) * 100)}->{round(b.get(t, 0) * 100)}%" for t in sorted(ts))
            if m.level_raises_pop(lv):
                crit["levelup"] += "; +1 population cap"
    crit["stop"] = "stop shopping (close shop, keep energy)"
    return {"type": "choice",
            "instructions": "Auto-chess shop turn: choose the next shop action.",
            "criteria": crit}


def dedupe_shop(shop):
    """同名牌只留第一个槽位：买哪一张是同一个决策（f3 起；f2 数据里老师只能在等价项之间随手挑）。"""
    seen, out = set(), []
    for slot, name, cost in shop:
        if name in seen:
            continue
        seen.add(name)
        out.append((slot, name, cost))
    return out


def to_internal(qdef: dict) -> dict:
    """Laya 公开 API 的问题格式 -> laya.common.build_sequence 用的内部格式（同 laya.agent.Agent._to_internal）。"""
    crit = qdef.get("criteria")
    if qdef["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    return {"t": qdef["type"], "ins": qdef["instructions"], "crit": crit}


def clean_name(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


# ---------------- eco 题（经济节奏，2026-10-09，挂 f9le）----------------
# 选项 levelup / roll / save；说明里写缺的配套逐个在 L / L+1 / L+2 级的命中率（facts.eco_*）。
# f9le = f9l 全部 + eco（字符串比较 "f9l" < "f9le" < "f9m"）。eco 是新题型，已有题型的渲染不受影响。
ECO_VERSION = "f9le"


def q_eco(ctx: dict) -> dict | None:
    """每回合准备阶段开头一题：升级 / 停在这一级刷牌 / 攒钱。ACTIVE < f9le 时不出题（返回 None）。"""
    if not _v(ECO_VERSION):
        return None
    X = _facts()
    B = X.Board(ctx)
    crit = {}
    lv = X.eco_levelup_parts(B)
    if lv:
        crit["levelup"] = X.pack(*lv)
    ro = X.eco_roll_parts(B)
    if ro:
        crit["roll"] = X.pack(*ro)
    crit["save"] = X.pack(*X.eco_save_parts(B))
    if len(crit) < 2:
        return None
    return {"type": "choice",
            "instructions": "Auto-chess (lord Ali) economy for this round: level up the lord, stay at this level and spend "
                            "on refreshes to find the MISSING support heroes, or save energy? " + X.eco_state_extra(B),
            "criteria": crit}
