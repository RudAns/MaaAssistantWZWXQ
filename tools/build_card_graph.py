# -*- coding: utf-8 -*-
"""全卡种异构关系图谱 data/card_graph.json 的生成脚本（可重跑）。

用法：
    .venv/Scripts/python.exe -X utf8 tools/build_card_graph.py            # 生成 + 自检
    .venv/Scripts/python.exe -X utf8 tools/build_card_graph.py --sample 20 # 另外打印抽样边
    .venv/Scripts/python.exe -X utf8 tools/build_card_graph.py --v3        # 图谱 v3（2026-10-09）：写 data/card_graph_v3.json，
        # 在默认输出之上加 手法 tech:slow_roll_lineup / tech:level_open_pool、grants_pool 边（发英雄牌 → 卡池里的每个英雄）、
        # 英雄节点 shop_hit（各棋手等级刷一次商店见到它的概率）。**不带 --v3 时输出与之前逐字相同**（data/card_graph.json 不动）。
    --out PATH   改输出路径（默认：不带 --v3 = data/card_graph.json，带 --v3 = data/card_graph_v3.json）

原则：定位与关系一律读卡面原文（skill + awaken + 技能/强化描述），不靠 cards.json 的 tags；
每条边带 quote，quote 必须是 from 或 to 节点文本的**逐字子串**（hero_graph 原样并入的边若含“…”，按段校验）。
不合格的边**直接丢弃**并在输出 meta.dropped 里列出。

输入：data/cards.json、hero_graph.json、hero_roles.json、equip_profiles.json、equip_recipes.json、
      carry_playbooks.json、keywords.json、card_understanding.json（逐卡“理解”，人工按原文推理）。
可调用（只读）：agent.decision.rules.RuleEngine.card_target_ok（合法性）、agent.decision.hero_traits（伤害类型）。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = ROOT / "data"

# 图谱 v3 开关（2026-10-09）。模块级代码在 argparse 之前就跑完了，所以这里先读 argv。
V3 = "--v3" in sys.argv[1:]
_out_arg = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[1:-1], 1) if x == "--out"), None)
OUT_PATH = Path(_out_arg) if _out_arg else DATA / ("card_graph_v3.json" if V3 else "card_graph.json")
# --understanding PATH（2026-10-26）：改读暂存的理解文件（如 data/card_understanding.next.json），不动线上那份
_und_arg = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[1:-1], 1) if x == "--understanding"), None)


def _load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


CARDS = _load("cards.json")["cards"]
HG = _load("hero_graph.json")
HROLES = _load("hero_roles.json")["heroes"]
EPROF = _load("equip_profiles.json")
RECIPES = _load("equip_recipes.json")["recipes"]
PLAYBOOK = _load("carry_playbooks.json")
KEYWORDS = _load("keywords.json")["keywords"]
try:
    UNDERSTAND = (json.loads(Path(_und_arg).read_text(encoding="utf-8")) if _und_arg
                  else _load("card_understanding.json"))
except FileNotFoundError:
    UNDERSTAND = {}

FACTIONS = ["河洛", "逐鹿", "日落海", "三分之地", "大河流域", "无阵营"]
FAC_NORM = {"三分": "三分之地", "大河": "大河流域", "河洛": "河洛", "逐鹿": "逐鹿", "日落海": "日落海",
            "三分之地": "三分之地", "大河流域": "大河流域", "无阵营": "无阵营"}
KW_LIST = ["整备", "登场", "牺牲", "开团", "闪现", "夺取", "复生", "凯旋", "退场", "败阵"]

# ---------------------------------------------------------------- 边类型（≤21 种）
EDGE_TYPES = {
    "can_target":     "合法性（允许名单）：该牌/装备**可以**作用于该英雄。只对“有使用对象限制”的牌生成；"
                      "节点 target_rule=allow_list 时，没有 can_target 边的英雄 = 不可用。由 RuleEngine.card_target_ok 程序化生成。",
    "cannot_target":  "合法性（禁止名单）：该牌/装备**不能**作用于该英雄。节点 target_rule=deny_list 时使用（被禁的比允许的少）。",
    "gives_level":    "供给等级：from 使 to 的等级提升。attrs.mode = real(真实等级)/perm(战斗中永久)/temp(临时，战斗结束清除)；"
                      "attrs.pick = chosen(玩家指定)/all/random:N/lowest/highest/nearest…；to=阿科米亚核心 表示给核心。",
    "gives_mana":     "法力：降低法力值上限或回复法力（加快放技能）。",
    "gives_buff":     "战斗增益：attrs.effect ∈ shield/revive/heal/hp/defense/damage/attack_speed/crit/lifesteal/cast/cc_immune/untargetable/stat。",
    "gives_resource": "棋手资源：attrs.res ∈ energy/xp/refresh/pop/player_hp/player_level；to=player。",
    "gives_equip":    "给装备（含战斗内临时装备、装备分享）。attrs.item 为具体装备（若有）。",
    "gives_awaken":   "觉醒进度 +N。",
    "grants_keyword": "赋予关键词：to 获得 attrs.keyword（如 玄微之种→穿戴者获得牺牲）。",
    "triggers":       "主动触发关键词效果：from 触发 to 的 attrs.keyword 效果（to 已展开到带该关键词的具体英雄）。",
    "produces":       "产出卡牌/单位：from 会给你（或召唤出）to（具体卡/召唤物，或一类卡的集合节点）。attrs.auto_use=true 表示产出后立即自动使用。",
    "feeds":          "喂养：使用/触发 from（通常是一类牌）会推进 to 的被动计数（如 效果牌→镜“每使用2张效果牌”）。",
    "amplifies":      "放大：from 让 to 这张牌/机制的收益变大（如 武则天→古币“古币牌提升的等级+1”、枫叶舞强化→枫叶舞）。",
    "steals_level":   "夺取/扣减：from 让 to 失去等级（负向关系，如 众星拱月夺取其他英雄各10级）。",
    "equip_fits":     "装备适配：该装备适合该英雄（来自 equip_profiles 的 heroes_best / 英雄 top_items，attrs.rank 为该英雄的优先序）。",
    "equip_avoid":    "装备不适配：equip_profiles 的 heroes_avoid，或伤害类型相反（hero_traits.item_mismatch）。",
    "exclusive_for":  "专属生效：装备/牌的特殊效果只对 to 生效（月影之力→露娜、【日落海】专属装备→日落海）。",
    "pairs_with":     "成对：两者同在时产生额外效果（双人天赋的两名英雄、怒莲之刺↔银莲庇佑）。对称，只存一条。",
    "talent_boosts":  "天赋→路线主C：该天赋在以 to 为主C 的路线里评分 ≥8（carry_playbooks.json）。",
    "enables":        "卡牌→手法：这张牌是某个多步手法（technique）的必要/可选部件。",
    "fits_role":      "英雄→手法：卡面原文明确点名该英雄是手法的一环，attrs.role 为角色槽名。",
}

# ---------------------------------------------------------------- 功能标签（节点 func_tags，正则按原文抽取）
FUNC_TAGS = [
    ("level_up", r"等级\+|等级永久\+|提升\d+级|提升5-20|等级提升|获得.{0,12}的等级|额外\+\d+等?级|等级数.{0,6}临时等级"),
    ("temp_level", r"临时等级"),
    ("perm_level", r"永久\+|等级永久|永久等级"),
    ("core_level", r"阿科米亚核心"),
    ("mana", r"法力值"),
    ("shield", r"护盾"),
    ("heal", r"恢复.{0,12}生命|生命恢复|治疗"),
    ("revive", r"复生"),
    ("cc", r"眩晕|击飞|冻结|沉默|嘲讽|压制|变成小猪|变为小猪|变为猪猪|控制4秒"),
    ("damage_amp", r"伤害加成|伤害\+|伤害提高|伤害提升|真实伤害"),
    ("attack_speed", r"攻速|攻击速度"),
    ("crit", r"暴击"),
    ("lifesteal", r"吸血"),
    ("hp", r"生命值上限|最大生命值\+|基础生命值"),
    ("defense", r"防御|伤害减免|伤害-|格挡"),
    ("energy", r"\d+能量|能量[，。]"),
    ("xp", r"棋手经验"),
    ("refresh", r"刷新"),
    ("pop", r"可上阵人数"),
    ("hero_gen", r"英雄牌|兑换券"),
    ("card_gen", r"获得.{0,6}张|选1|选2|随机获得"),
    ("equip_gen", r"装备牌|装备箱|获得.{0,4}件|基础装备|普通装备双剑"),
    ("awaken", r"觉醒进度"),
    ("trigger", r"触发.{0,8}(整备|登场|牺牲|开团)效果"),
    ("keyword_grant", r"获得牺牲|获得闪现"),
    ("steal", r"夺取|偷取"),
    ("summon", r"召唤"),
    ("transient", r"转瞬"),
    ("sell_hook", r"出售|退场"),
    ("merge_hook", r"合成"),
    ("coin", r"古币"),
    ("tactic", r"战术牌|战术"),
    ("totem", r"图腾"),
    ("maple", r"枫叶舞|转移"),
    ("player_hp", r"棋手生命值|棋手每次受到的伤害"),
    ("remove_unit", r"移除|消除|吞噬"),
    ("flash", r"闪现"),
    ("unique", r"唯一"),
]

# ---------------------------------------------------------------- 节点收集
SKIP_ALIAS = {}          # 重复条目 name -> 本名
NODES: dict[str, dict] = {}


def card_text(c: dict) -> str:
    parts = [c.get("skill") or "", c.get("awaken") or ""]
    for s in c.get("skills") or []:
        parts.append(s.get("desc") or "")
        for e in s.get("enhance") or []:
            parts.append(e.get("desc") or "")
    return "\n".join(p for p in parts if p)


TYPE_OF = {"英雄牌": "hero", "效果牌": "effect", "装备牌": "equip", "天赋牌": "talent",
           "棋手技能牌": "lord_skill", "召唤物": "summon"}


def keywords_held(c: dict) -> list[str]:
    """卡面自带的关键词（行首 / 逗号后紧跟“：”或换行 / 单独成词），不算“拥有牺牲效果的英雄”这类引用。"""
    txt = (c.get("skill") or "") + "\n" + (c.get("awaken") or "")
    out = []
    for k in KW_LIST:
        if re.search(rf"(?:^|\n|，|、|/ ){k}(?:：|\n|，|$)", txt):
            out.append(k)
    return out


def func_tags(txt: str) -> list[str]:
    return [t for t, rx in FUNC_TAGS if re.search(rx, txt)]


for name, c in CARDS.items():
    cat = c.get("category")
    if cat == "棋手牌":
        continue
    if cat in ("效果牌", "棋手技能牌") and c.get("lord") and c.get("lord") != "阿离":
        continue            # 其他棋手的专属牌：本项目只打阿离
    m = re.match(r"^(.+)\((效果牌|装备牌|召唤物)\)$", name)
    if m and m.group(1) in CARDS:
        SKIP_ALIAS[name] = m.group(1)        # 数据里的重复条目（文本相同或是英雄的召唤复制体）
        continue
    # 2026-10-25 重抓卡库新增"X(赠送)"条目：卡面与本体逐字相同的（天赋送的装备）只是重复条目，并到本体；
    # 文本不同的（望梅止渴(赠送) = 天赋发出来的那张牌本身）保留为独立节点
    m = re.match(r"^(.+)\(赠送\)$", name)
    if m and m.group(1) in CARDS and (CARDS[m.group(1)].get("skill") or "") == (c.get("skill") or ""):
        SKIP_ALIAS[name] = m.group(1)
        continue
    t = TYPE_OF.get(cat)
    if not t:
        continue
    txt = card_text(c)
    node = {"id": name, "type": t, "category": cat, "faction": c.get("faction") or "",
            "quality": c.get("quality"), "text": txt, "func_tags": func_tags(txt)}
    if t == "hero":
        node["keywords"] = keywords_held(c)
        hr = HROLES.get(name) or {}
        node["combat_role"] = hr.get("combat_role") or []
        node["carry_value"] = hr.get("carry_value")
        node["functional_value"] = hr.get("functional_value")
        node["melee"] = ((c.get("properties") or {}).get("attackDistance") or 0) <= 1
    if t == "summon":
        node["keywords"] = keywords_held(c)
    if c.get("lord"):
        node["lord"] = c["lord"]
        node["lord_group"] = c.get("talent_group")
    if t == "talent":
        node["talent_group"] = c.get("talent_group")
    if t == "equip":
        node["equip_class"] = ("base" if name in RECIPES else
                               "normal" if any(name in v for v in RECIPES.values()) else
                               "sunset" if "【日落海】专属装备" in txt else "special")
        if "转瞬" in txt:
            node["transient"] = True
    if "转瞬" in (c.get("skill") or ""):
        node["transient"] = True
    NODES[name] = node

HEROES = sorted(n for n, v in NODES.items() if v["type"] == "hero")
TEXT = {n: v["text"] for n, v in NODES.items()}

# ---------------------------------------------------------------- 集合节点（scope）
ALI_BOOSTS = [n for n, v in NODES.items() if v.get("lord") == "阿离" and v.get("lord_group") == "关联"
              and not n.startswith("翩跹舞步") and n != "即刻起舞"]
PIANXIAN = [n for n in NODES if n.startswith("翩跹舞步")]
TACTICS = [n for n in NODES if NODES[n]["type"] == "effect" and n.endswith("战术")]


def members_kw(k):
    return sorted(h for h in HEROES if k in NODES[h]["keywords"])


SCOPES: dict[str, dict] = {
    "*": {"kind": "all", "def": "你场上的全部英雄（按边的 attrs.pick 取其中一部分）"},
    "carry": {"kind": "role", "def": "主C（运行时由规则/Laya 决定）"},
    "@chosen": {"kind": "chosen", "def": "玩家指定的那名英雄；合法集合见该牌的 can_target/cannot_target 边"},
    "player": {"kind": "player", "def": "棋手本人（能量/经验/刷新/人口/棋手生命值/棋手等级）"},
    "阿科米亚核心": {"kind": "core", "def": "日落海的阿科米亚核心：开战时按比例把等级以临时等级分给日落海英雄"},
    "@maple_holder": {"kind": "role", "def": "当前被枫叶舞指定的英雄（阿离）"},
    "@maple_boost": {"kind": "card_set", "def": "枫叶舞强化牌（阿离关联效果牌）", "members": sorted(ALI_BOOSTS)},
    "@pianxian": {"kind": "card_set", "def": "秘技牌翩跹舞步一族", "members": sorted(PIANXIAN)},
    "@tactic": {"kind": "card_set", "def": "战术牌（逐鹿效果牌，名字以“战术”结尾）", "members": sorted(TACTICS)},
    "@effect": {"kind": "card_set", "def": "任意效果牌（使用时触发“使用效果牌”类被动）"},
    "@effect_bag": {"kind": "card_set", "def": "效果锦囊（N阶效果锦囊）",
                    "members": sorted(n for n in NODES if n.endswith("效果锦囊"))},
    "@equip:base": {"kind": "card_set", "def": "基础装备（穿上后四选一铸成普通装备）", "members": sorted(RECIPES)},
    "@equip:normal": {"kind": "card_set", "def": "普通装备（基础装备铸成的 32 件成品）",
                      "members": sorted({x for v in RECIPES.values() for x in v})},
    "@equip:sunset": {"kind": "card_set", "def": "【日落海】专属装备",
                      "members": sorted(n for n, v in NODES.items() if v.get("equip_class") == "sunset")},
    "@equip:special": {"kind": "card_set", "def": "特殊装备 / 天赋装备（非基础、非普通、非日落海专属）",
                       "members": sorted(n for n, v in NODES.items() if v.get("equip_class") == "special")},
    "@equip:any": {"kind": "card_set", "def": "任意装备"},
    "@hero_any": {"kind": "card_set", "def": "英雄牌（不限定派系；attrs.tier 给品阶范围）"},
    "@on_board": {"kind": "card_set", "def": "你场上已有英雄的同名英雄牌"},
    "@copy_of_chosen": {"kind": "card_set", "def": "被指定英雄的同名英雄牌"},
    "@same_faction_as_chosen": {"kind": "card_set", "def": "与被指定英雄同阵营的英雄牌"},
    "@same_tier_as_chosen": {"kind": "relative", "def": "与被指定英雄同品阶的（场上英雄 / 英雄牌，见边说明）"},
    "@highest_tier": {"kind": "relative", "def": "你场上品阶最高的 1 名英雄"},
    "@highest_level": {"kind": "relative", "def": "你场上等级最高的英雄"},
    "@lowest_level": {"kind": "relative", "def": "你场上等级最低的英雄"},
    "@awakened": {"kind": "state", "def": "已觉醒的英雄"},
    "@wearer": {"kind": "relative", "def": "该装备的穿戴者"},
    "@adjacent": {"kind": "relative", "def": "穿戴者/该英雄周围 1 格的己方单位"},
    "@back_rows": {"kind": "position", "def": "你后两排的英雄"},
    "@front_row": {"kind": "position", "def": "你第一排的英雄"},
    "@equipped2": {"kind": "state", "def": "至少穿戴 2 件装备的英雄"},
    "@summons": {"kind": "unit_set", "def": "你的召唤单位（召唤物、图腾、士兵…）"},
    "@totem": {"kind": "unit_set", "def": "图腾（先知/因缘/往生…）",
               "members": sorted(n for n in NODES if "图腾" in n and NODES[n]["type"] == "summon")},
    "@played": {"kind": "relative", "def": "本次打出的那张英雄牌"},
    "@tier:4-5": {"kind": "tier", "def": "4~5 阶英雄", "members": sorted(h for h in HEROES if NODES[h]["quality"] in (4, 5))},
    "@tier:1-3": {"kind": "tier", "def": "1~3 阶英雄", "members": sorted(h for h in HEROES if NODES[h]["quality"] in (1, 2, 3))},
}
for q in range(1, 6):
    SCOPES[f"@tier:{q}"] = {"kind": "tier", "def": f"{q} 阶英雄", "members": sorted(h for h in HEROES if NODES[h]["quality"] == q)}
for f in FACTIONS:
    SCOPES[f"@f:{f}"] = {"kind": "faction", "def": f"【{f}】英雄", "members": sorted(h for h in HEROES if NODES[h]["faction"] == f)}
for k in KW_LIST:
    SCOPES[f"@kw:{k}"] = {"kind": "keyword", "def": f"卡面自带「{k}」的英雄（{KEYWORDS.get(k, '')}）", "members": members_kw(k)}
# hero_graph 的非派系/非关键词分组
HG_GROUP_MAP = {}
for g, v in HG["groups"].items():
    if v["type"] == "faction":
        HG_GROUP_MAP["@" + g] = "@f:" + FAC_NORM.get(g, g)
    elif g in KW_LIST:
        HG_GROUP_MAP["@" + g] = "@kw:" + g
    else:
        HG_GROUP_MAP["@" + g] = "@g:" + g
        SCOPES["@g:" + g] = {"kind": "hero_group", "def": v.get("why", ""), "members": v["members"], "source": "hero_graph"}

# ---------------------------------------------------------------- 手法（technique）：超边 + 角色槽
# 谓词词汇：on_board / in_hand / is_maple_holder / real_level>=N / tier in [a,b] /
# faction==X / has_keyword(K) / carry_eligible / is_carry / is_engine_part / is_keeper / equip_count>=N /
# awaken_progress==N / adjacent(a,b) / nearest_ally_of(a)==b / carry_value<=N / functional_value>=N / name in [...]
TECHNIQUES = {
    "tech:maple_level_carryover": {
        "name": "红叶染锋：卖旧枫叶舞载体、把 50% 真实等级带给新主C",
        "steps": ["sell(old_holder)", "use_card(枫叶舞 -> new_carry)"],
        "repeatable": False, "once_note": "限一次（原文“限一次”）",
        "precondition": ["has_maple_boost(红叶染锋)"],
        "roles": {
            "old_holder": ["is_maple_holder", "on_board", "real_level >= 20  # 低于此收益可忽略（见 value）"],
            "new_carry": ["carry_eligible", "on_board or in_hand", "not is_maple_holder"],
        },
        "cost": ["失去 old_holder 这个单位", "old_holder 身上装备退回手牌（可转给 new_carry）", "卖出获得能量（卖价）"],
        "benefit": "new_carry.real_level += floor(0.5 * old_holder.real_level)",
        "timing": "中后期（旧载体已攒下大量真实等级时）；前期过渡主C 等级低，红叶几乎给不了什么",
        "value": "≈ 0.5 * old_holder.real_level；前期（旧载体 <20 级）≈0 —— 想立刻换主C 时根本不需要红叶，"
                 "天赋/强化三选一也不该为了“马上换主C”拿它（用户 2026-09-30）",
        "status": "confirmed",
        "evidence": "红叶染锋原文 + 枫叶舞“卖掉被绑的英雄才能重绑”（用户 2026-09-22 更正）",
    },
    "tech:yizhi_transfer": {
        "name": "遗志：移除一名英雄，把其真实等级转给 1 名随机英雄",
        "steps": ["use_card(遗志 -> level_source)"],
        "repeatable": True,
        "roles": {
            "level_source": ["on_board", "real_level >= 10", "not is_carry"],
            "receiver": ["on_board", "random: 场上除 level_source 外任意 1 名英雄"],
            "intended_receiver": ["is_carry or carry_eligible"],
        },
        "cost": ["level_source 被移除（不是出售，不给能量）", "临时等级不转移",
                 "要确定命中 intended_receiver 必须先清场到只剩它（= 伞海月式“卖光全场搬等级”，用户已排除：操作难）"],
        "benefit": "random_receiver.real_level += level_source.real_level",
        "success_prob": "P(命中 intended_receiver) = 1 / (n_board - 1)，n_board 含 level_source；只剩两人时 = 1",
        "timing": "场上人数少（刚卖完人 / 前期）时命中率高；满编 7 人时只有 1/6",
        "status": "confirmed",
        "evidence": "遗志原文“给1名随机英雄”",
    },
    "tech:chuancheng_spread": {
        "name": "传承：把“等级高但等级对它没用”的功能件的等级摊给 3 名队友",
        "steps": ["use_card(传承 -> level_source)"],
        "repeatable": True,
        "roles": {
            "level_source": ["on_board", "real_level > 10", "not is_carry",
                             "carry_value <= 3 or level_payoff_done  # 等级对它价值低：功能来自卡面而非战斗力，"
                             "或技能强化档位(10/40/100)已拿到下一档还很远",
                             "passively_leveled  # 等级是被动涨的：吃了全队加等级/牺牲自涨/枫叶舞外溢/古币"],
            "receivers": ["on_board", "random 3 名其他英雄（不重复）"],
        },
        "cost": ["level_source 的等级清零（原文“将其等级随机分配”，推断 source 失去这些等级，uncertain）"],
        "benefit": "3 名随机其他英雄合计 += level_source.real_level（每人约 1/3，分配方式随机，uncertain）",
        "success_prob": "主C 被选中的概率 = min(1, 3/(n_board-1))；不是给主C 定向加等级的手段",
        "timing": "功能件被动涨到很高等级之后（中后期）",
        "status": "confirmed_with_uncertain_parts",
        "uncertain": "“随机分配”是均分还是随机份额、source 是否被清零，卡面没写清",
        "evidence": "传承原文 + 用户 2026-09-30 用法说明",
    },
    "tech:maple_rebind": {
        "name": "卖人解绑枫叶舞，改绑新主C",
        "steps": ["sell(old_holder)", "use_card(枫叶舞 -> new_carry)"],
        "repeatable": True,
        "roles": {
            "old_holder": ["is_maple_holder", "on_board", "not is_carry or carry_changed"],
            "new_carry": ["is_carry", "on_board"],
        },
        "cost": ["失去 old_holder", "old_holder 的装备退回手牌"],
        "benefit": "枫叶舞（护盾+转移至纸伞处，及已选强化）作用到新主C；配合 tech:maple_level_carryover 可带走 50% 等级",
        "timing": "新主C 已上场且当满一回合后（换主C 的收尾一步）",
        "status": "confirmed",
        "evidence": "用户 2026-09-22：绑定后必须把被绑的英雄卖掉才能重绑；卡面亮=可绑、灰=已绑",
    },
    "tech:equip_return_transfer": {
        "name": "卖人退装备，转穿给新主C",
        "steps": ["sell(old_wearer)", "equip(item -> new_carry) × n"],
        "repeatable": True,
        "roles": {
            "old_wearer": ["on_board", "equip_count >= 1", "not is_keeper"],
            "new_carry": ["is_carry", "equip_count < 3"],
            "item": ["returned_to_hand", "not (唯一 and new_carry already wears same)",
                     "not 无法拆卸  # 盛华/百锻刀/聚灵法杖 无法拆卸，推断卖人即损失，uncertain"],
        },
        "cost": ["失去 old_wearer", "基础装备铸成的成品退回后是否仍是成品：只确认过“装备会退回手牌”，没区分，uncertain"],
        "benefit": "装备集中到主C",
        "status": "confirmed",
        "evidence": "实战确认：卖掉带装备的英雄，装备会退回手牌（曾卖掉蒙恬，制裁之剑/破晓退回）",
    },
    "tech:awaken_redeploy": {
        "name": "觉醒回手重上",
        "steps": ["(觉醒进度满 → 英雄回到手牌)", "deploy(awakened_unit -> cell)"],
        "repeatable": True,
        "roles": {
            "awakened_unit": ["in_hand", "awakened", "was_on_board"],
            "cell": ["empty or freed_by_sell", "row fits role"],
        },
        "cost": ["人口满时需先卖人腾位"],
        "benefit": "觉醒卡面效果（通常数值翻倍）+ 觉醒后强度",
        "timing": "立即（最高优先级：曾因觉醒后没及时回手重上输掉一局）",
        "status": "confirmed",
        "evidence": "实战确认的觉醒回手规则；觉醒进度来源见 gives_awaken 边",
    },
    "tech:sacrifice_engine": {
        "name": "牺牲引擎：带牺牲的前排每回合阵亡，喂牺牲收益方",
        "steps": ["deploy(fodder -> front_row)", "(战斗中 fodder 阵亡触发牺牲)"],
        "repeatable": True,
        "roles": {
            "fodder": ["has_keyword(牺牲) or granted(牺牲)", "front_row"],
            "payoff": ["name in [百里守约, 李信, 大司命] or self_growth_on_sacrifice"],
            "granter": ["name == 钟馗 (左右1格获得牺牲)  or 持有赋予牺牲的天赋/装备"],
        },
        "cost": ["需要前排真的阵亡：阵容太强赢得太轻松时触发少"],
        "benefit": "百里守约本回合法力上限-20/次；李信临时等级+7/次；大司命往生图腾永久+2/次；苏烈自身永久+2",
        "status": "confirmed",
        "evidence": "各英雄卡面原文",
    },
    "tech:tier_evolve": {
        "name": "品阶进化：把低阶英雄连同等级换成高阶英雄",
        "variants": {
            "限制性进化": "移除 1~2 阶 → 得高 2 阶随机英雄牌，上阵时等级 = 原等级 + 10",
            "化铁为金": "1~3 阶 → 随机变为不在场的 5 阶英雄，觉醒并保留当前等级（含临时等级）",
        },
        "steps": ["use_card(进化牌 -> source)", "deploy(new_hero)  # 限制性进化需要重新上场；化铁为金原地变身"],
        "repeatable": True,
        "roles": {
            "source": ["on_board", "tier in [1,2] (限制性进化) / tier in [1,3] (化铁为金)",
                       "not is_carry or carry_is_transitional", "real_level 高越赚"],
        },
        "cost": ["随机性：得到的英雄不可控，可能不在路线里", "化铁为金会把已觉醒主C 变成随机英雄（与霜寒进化把主C 变掉是同类风险）"],
        "benefit": "限制性进化：+2 品阶 +10 级；化铁为金：变 5 阶 + 觉醒 + 保留等级",
        "timing": "越早用越赚（用户 2026-09-24）",
        "status": "confirmed",
        "evidence": "两张牌原文",
    },
    "tech:yao_first_equip_share": {
        "name": "瑶首件普通装备分享",
        "steps": ["deploy(瑶)", "equip(first_normal_item -> 瑶)"],
        "repeatable": False,
        "roles": {
            "sharer": ["name == 瑶", "equip_count == 0 before"],
            "item": ["is_normal_equip  # 基础装备铸成的普通装备；特殊装备不算", "best: 破茧之衣（开团眩晕）"],
        },
        "cost": ["瑶的第一件装备位"],
        "benefit": "开团时 3 名随机友军获得同一件普通装备（未觉醒持续 15 秒，觉醒后全场）",
        "status": "confirmed",
        "evidence": "瑶原文",
    },
    "tech:nearest_ally_buddy": {
        "name": "贴主C 站位：作用于“最近友军”的辅助站到主C 旁边",
        "steps": ["move_unit/deploy(buddy -> cell adjacent to carry, carry 是 buddy 唯一最近的友军)"],
        "repeatable": True,
        "roles": {
            "buddy": ["name in [太乙真人, 瑶, 明世隐, 李元芳]"],
            "carry": ["is_carry", "李元芳时 faction == 河洛"],
        },
        "cost": ["占用主C 相邻格"],
        "benefit": "太乙：主C 获得复生；瑶：附身护盾+双攻双防；明世隐：临时等级转永久+连线治疗；李元芳：整备古币给主C",
        "status": "confirmed",
        "evidence": "各英雄原文“距离最近的/最近的友方英雄”",
    },
    "tech:steal_merge": {
        "name": "夺取站位合成：把高等级的非主C 放到夺取者相邻格，再合成/上场夺取者",
        "steps": ["move_unit(fodder -> adjacent(stealer))", "deploy or merge(stealer 同名牌)"],
        "repeatable": True,
        "roles": {
            "stealer": ["has_keyword(夺取)"],
            "fodder": ["adjacent(stealer)", "real_level >= 10", "not is_carry", "唯一的相邻友军（否则随机夺到别人）"],
        },
        "cost": ["fodder 失去最多 10 级", "相邻格只能站 fodder，否则可能夺到主C"],
        "benefit": "stealer.real_level += min(10, fodder.real_level)（不含临时等级）",
        "status": "confirmed",
        "excluded_by_user": "公孙离当主C 的夺取流（2026-09-30 用户：要站位合成，操作不了）",
        "evidence": "关键词“夺取”的官方定义；公孙离/猪八戒/孙悟空卡面“夺取”",
    },
    "tech:clear_board_swap": {
        "name": "清场换人：移除/卖掉低价值单位，给新主C 腾位并回收资源",
        "steps": ["use_card(清场牌) or sell(removed)×n", "deploy(new_carry)", "(可选) use_card(遗志 -> 旧主C) 搬等级"],
        "repeatable": True,
        "roles": {
            "removed": ["on_board", "not is_carry", "not is_engine_part", "low value"],
            "new_carry": ["tier >= 4", "carry_eligible"],
        },
        "cost": ["失去单位（战斗力下降一回合）", "4~5 阶主C 从 0 级起步，需要搬等级（遗志随机、传承随机）才追得上"],
        "benefit": "听天由命：每消除 1 名 +3 能量 + 同阶兑换券 + 简易协作；体面离场：按觉醒进度 3/5/7/9 能量",
        "timing": "用户 2026-09-29：4~5 阶主C 要遗志、要清场来换，很难 —— 默认不走",
        "status": "uncertain",
        "uncertain": "“清场换人”是用户口述的套路名，具体哪些牌组合最常用未经实测",
        "evidence": "听天由命 / 体面离场 / 遗志 原文",
    },
    "tech:sell_hooks": {
        "name": "出售换收益：卖场上英雄触发出售类被动",
        "steps": ["sell(unit)"],
        "repeatable": True,
        "roles": {"sold": ["on_board", "not is_keeper"], "hook_holder": ["on_board or talent held"]},
        "cost": ["失去单位"],
        "benefit": "银/金算盘随机 1 人 +2/+3；莱西奥核心+1/2 人；姬小满临时刷新；能量汲取核心+卖出者等级；丝绸之路古币；梦奇退场兑换券",
        "status": "confirmed",
        "evidence": "各卡原文“出售场上英雄时 / 退场：”",
    },
    "tech:level_swap": {
        "name": "三分换等级：让指定【三分】英雄与最近的【三分】英雄交换/复制等级",
        "variants": {"换帅": "交换等级", "挂帅": "使其等级等同于最近的【三分】英雄（复制，不扣对方）"},
        "steps": ["move_unit(high_level_partner -> 与 new_carry 最近)", "use_card(换帅/挂帅 -> new_carry)"],
        "repeatable": True,
        "roles": {
            "new_carry": ["faction == 三分之地"],
            "partner": ["faction == 三分之地", "nearest_ally_of(new_carry) among 三分", "real_level 高"],
        },
        "cost": ["换帅：partner 得到 new_carry 的低等级"],
        "benefit": "new_carry.real_level = partner.real_level",
        "status": "confirmed",
        "evidence": "换帅 / 挂帅上阵 原文",
    },
    "tech:rift_devour": {
        "name": "混沌裂隙吞噬：把单位放进裂隙格，开战前被吞噬，转化成满装备召唤英雄",
        "steps": ["use_card(混沌裂隙 -> cell)", "deploy/move_unit(victim -> cell)"],
        "repeatable": False,
        "roles": {"victim": ["on cell", "real_level 高", "not is_carry"]},
        "cost": ["victim 被吞噬（永久失去）"],
        "benefit": "裂隙永久获得 victim 双倍等级；开战召唤 1 名品阶≥商店等级的英雄（无卡牌效果）拿裂隙等级与适配装备",
        "status": "uncertain",
        "uncertain": "每回合是否都吞噬、裂隙等级是否跨回合累积，卡面未写清；从未实战验证",
        "evidence": "混沌裂隙原文",
    },
}

# ---------------------------------------------------------------- 人工整理的边（逐条读原文）
# 行格式：from | type | to | quote | k=v;k=v     （quote 必须是 from 或 to 文本的逐字子串）
MANUAL = r"""
# ===== 效果牌
古币|gives_level|@chosen|指定1名【河洛】英雄等级+2|amt=2;mode=real;pick=chosen
援护战术|gives_buff|@f:逐鹿|全体【逐鹿】英雄本回合获得50|effect=shield
演练战术|gives_level|*|所有英雄等级+1|amt=1;mode=real;pick=all
推演战术|gives_mana|@chosen|使其本回合法力值上限|pick=chosen
特训战术|gives_level|@chosen|指定1名英雄等级+4|amt=4;mode=real;pick=chosen
精致古币|gives_level|@chosen|指定1名【河洛】英雄等级+4|amt=4;mode=real;pick=chosen
弦音|gives_awaken|@chosen|使其觉醒进度+1|amt=1;pick=chosen
跳级|gives_level|@chosen|使1名英雄夺取阿科米亚核心至多5级|amt=5;mode=real;pick=chosen;pool=core
跳级|steals_level|阿科米亚核心|夺取阿科米亚核心至多5级|amt=5
图腾仪式|produces|@g:图腾|与图腾相关的【大河流域】英雄牌|
临阵磨枪|triggers|@kw:整备|触发其整备效果|pick=chosen
唤醒|triggers|@kw:登场|触发其登场效果|pick=chosen
简易协作|gives_level|@chosen|指定1名英雄等级+3|amt=3;mode=real;pick=chosen
简易协作|gives_level|*|随机1名英雄等级+2|amt=2;mode=real;pick=random:1
精密协作|gives_level|@chosen|指定1名英雄等级+4|amt=4;mode=real;pick=chosen
精密协作|gives_level|*|随机1名英雄等级+3|amt=3;mode=real;pick=random:1
八方来援|gives_level|@chosen|指定1名英雄等级+2。场上每有1种阵营，额外触发1次此效果|amt=2;mode=real;pick=chosen;scale=n_factions_on_board
武装之力|gives_level|*|所有英雄获得等同于自身装备数量的等级|mode=real;pick=all;scale=equip_count
遗志|enables|tech:yizhi_transfer|转移其等级（临时等级除外）给1名随机英雄|
遗志|enables|tech:clear_board_swap|指定场上1名英雄将其移除|role=level_move
呼朋唤友|produces|@same_faction_as_chosen|随机获得1张与其同阵营的英雄牌|
换帅|enables|tech:level_swap|与其最近的【三分】英雄交换等级|
照影|produces|@on_board|随机获得1张场上英雄牌|
助力核心|gives_level|阿科米亚核心|阿科米亚核心等级+2，场上每有1名【日落海】英雄重复1次|amt=2;scale=n_faction_日落海
夺取之术|gives_level|@chosen|夺取最近1名己方英雄的等级（最多10级|amt<=10;mode=real;pick=chosen;pool=nearest_ally
夺取之术|steals_level|*|夺取最近1名己方英雄的等级|pick=nearest_of_chosen;amt<=10
夺取之术|enables|tech:steal_merge|夺取最近1名己方英雄的等级|role=card_variant
高阶演练战术|gives_level|*|所有英雄等级+2|amt=2;mode=real;pick=all
武装重构|produces|@equip:base|随机获得对应数量+1的基础装备|
阵营祝福|gives_level|*|各阵营中随机1名英雄等级+4|amt=4;mode=real;pick=random_per_faction
高阶招募战术|produces|@f:逐鹿|3选1【逐鹿】英雄牌|
高阶特训战术|gives_level|@chosen|指定1名英雄等级+8|amt=8;mode=real;pick=chosen
高阶援护战术|gives_buff|@f:逐鹿|全体【逐鹿】英雄本回合获得100|effect=shield
扶倾济弱|gives_level|@lowest_level|使场上等级最低（临时等级除外）的英雄提升9级|amt=9;mode=real;pick=lowest
妙音|gives_awaken|@chosen|使其觉醒进度+1|amt=1;pick=chosen
乱世|produces|@kw:登场|卡牌3选1（登场英雄牌）|
战前宣誓|gives_level|@chosen|获得自身品阶×3的临时等级|mode=temp;pick=chosen;scale=tier*3
逍遥自在|gives_level|@f:无阵营|所有无阵营英雄获得2×场上无阵营英雄数量的等级|mode=real;pick=all;scale=2*n_faction_无阵营
远洋宝藏|produces|@equip:sunset|从5件【日落海】专属装备中随机获得1件|
因材施教|produces|@effect|随机获得1张你场上主要阵营相关的效果牌|
传承|enables|tech:chuancheng_spread|指定场上1名10级以上的英雄，将其等级随机分配给其他英雄中等级最低的3名|
高阶推演战术|gives_mana|@chosen|使其本回合法力值上限|pick=chosen
度命之护|produces|贤者的庇护|获得1张特殊装备贤者的庇护|
武装分解|gives_resource|player|摧毁1件获得4能量|res=energy;amt=4
武装分解|produces|妙音|2件获得1张妙音|
武装分解|produces|珍品宝箱|3件获得1张珍品宝箱|
珍品宝箱|produces|@equip:special|卡牌3选1（特殊装备或天赋装备）|
久经沙场|produces|@copy_of_chosen|获得1张该英雄牌|
机关秘典|produces|机关匣|获得1件特殊装备机关匣|
图腾祝福|gives_level|@f:大河流域|随机1名【大河流域】英雄等级+5。场上每有1个图腾，额外触发1次此效果|amt=5;mode=real;pick=random:1;scale=n_totems
移花接木|gives_level|*|随机1名英雄获得敌方当前等级最高英雄的一半等级（包含临时等级）|pick=random:1;mode=battle;scale=enemy_max_level/2
有福同享|gives_level|@same_tier_as_chosen|场上和其品阶相同的英雄等级+4|amt=4;mode=real
求贤|produces|@same_faction_as_chosen|发现与其同阵营的3张英雄牌，从中选择1张|
奇谋|gives_buff|@chosen|下场战斗开始时会立即释放技能|effect=cast
开局狂飙|gives_buff|@chosen|本回合开战后前15秒内，可以无限释放技能|effect=cast
集体唤醒|triggers|@kw:登场|立即触发所有英雄的登场效果|pick=all
全军动员|triggers|@kw:整备|立即触发所有英雄的整备效果|pick=all
精选武装|produces|@equip:base|卡牌4选1（基础装备牌）|
众星闪耀|gives_level|*|全体己方英雄等级+10。每持有一回合，此效果增强5级|amt=10;mode=real;pick=all;scale=+5_per_round_held
展翅腾飞|gives_resource|player|本回合刷新商店不消耗能量|res=refresh;amt=inf_this_round
军械库|produces|@equip:base|随机获得3张基础装备牌|
众星拱月|gives_level|@chosen|夺取其他英雄各10级|pick=chosen;scale=10*(n_board-1)
众星拱月|steals_level|*|夺取其他英雄各10级|pick=all_others;amt=10
呼风唤雨|produces|@effect|随机获得4张5阶效果牌|tier=5
破境之证|gives_level|@chosen|指定1名英雄立即提升20级，之后每回合提升10级|amt=20;mode=real;pick=chosen;per_round=10
绝处逢生|produces|@effect|随机获得10张效果牌|
化铁为金|enables|tech:tier_evolve|随机变为不在场的5阶英雄，觉醒并保留其当前等级|role=variant_化铁为金
超级复印机|produces|@on_board|随机获得3张相同的场上英雄牌|
扶摇直上|gives_resource|player|棋手等级提升1|res=player_level;amt=1
四人同行|produces|@hero_any|将选中的卡牌一分为四|note=5选1任意卡
沧海拾遗|produces|@effect|随机获得1张其他6阶效果牌|tier=6
1阶效果锦囊|produces|@effect|选择1张1阶效果牌|tier=1
2阶效果锦囊|produces|@effect|选择1张2阶效果牌|tier=2
3阶效果锦囊|produces|@effect|选择1张3阶效果牌|tier=3
4阶效果锦囊|produces|@effect|选择1张4阶效果牌|tier=4
5阶效果锦囊|produces|@effect|选择1张5阶效果牌|tier=5
装备箱|produces|@equip:any|挑选1件装备|
吸纳人才|produces|@hero_any|卡牌4选1（英雄牌）|
广纳良才|produces|@hero_any|卡牌5选2（英雄牌）|
高级人才|produces|@hero_any|随机获得1张特定品阶（棋手等级+1）的英雄牌|tier=lord_level+1
# ===== 阿离：枫叶舞与强化
枫叶舞|gives_buff|@chosen|其受到最大生命值20%的伤害时将获得护盾并转移至纸伞处|effect=shield
枫叶舞|enables|tech:maple_rebind|使用技能，指向1名英雄|
枫叶舞|enables|tech:maple_level_carryover|使用技能，指向1名英雄|
即刻起舞|produces|@maple_boost|从3个枫叶舞强化中选择1项|
有叶新生|amplifies|枫叶舞|转移后恢复20%最大生命值|effect=heal
枫佑永驻|amplifies|枫叶舞|护盾量增加且持续时间变为无限|effect=shield
拈花一指|amplifies|枫叶舞|转移后眩晕附近的敌人2秒|effect=cc
红枫疾舞|amplifies|枫叶舞|转移后的4次普通攻击以最大速度进行|effect=attack_speed
枫灵涌动|amplifies|枫叶舞|转移后每秒获得15点法力值|effect=mana
红叶染锋|amplifies|枫叶舞|其真实等级的50%会保留给下一个枫叶舞英雄，限一次|effect=level_carryover
红叶染锋|enables|tech:maple_level_carryover|枫叶舞英雄离场后，其真实等级的50%会保留给下一个枫叶舞英雄|value=0.5*old_holder.real_level;early_game≈0
枫华之刃|amplifies|枫叶舞|转移后获得25%伤害加成|effect=damage
枫舞扶摇|gives_level|@maple_holder|枫叶舞英雄本局对战每击败1名英雄，此等级额外+1|mode=temp;amt=5;scale=+1_per_kill
枫舞扶摇|amplifies|枫叶舞|每回合首次转移后获得|effect=temp_level
枫影无拘|amplifies|枫叶舞|转移后5秒内免疫受到的控制|effect=cc_immune
斗转星移|amplifies|枫叶舞|获得15%物理与法术吸血|effect=lifesteal
青叶绽蕊|amplifies|枫叶舞|指定的英雄在拥有护盾时，每击败1名英雄则恢复10%最大生命值|effect=heal
层林遮影|amplifies|枫叶舞|转移后7秒内不被敌人优先攻击|effect=untargetable
一叶双生|amplifies|枫叶舞|转移时在原地创造1个具有30%最大生命值|effect=decoy
瞬步无影|amplifies|枫叶舞|开战时获得1次转移次数|effect=extra_transfer
踏叶摘星|amplifies|枫叶舞|转移所需要的击败数减1|effect=extra_transfer
翩跹舞步·余韵|produces|@maple_boost|为枫叶舞挑选1张强化牌|
翩跹舞步·进化|produces|@maple_boost|为枫叶舞挑选1张强化牌|
翩跹舞步·进化|gives_buff|@maple_holder|枫叶舞英雄获得本局击败敌方英雄数量×4点物理和法术攻击力|effect=stat
# ----- 2026-10-08 平衡调整后新增（阿离 6 级专属强化 / 新天赋 / 新装备；只连卡面写明的关系）
翩跹馈赠|amplifies|枫叶舞|枫叶舞英雄累积普攻15次时也会获得1次转移次数|effect=extra_transfer
枫之洪流|amplifies|枫叶舞|枫叶舞英雄在首次转移后的3秒内，可以无限释放技能|effect=cast
落英汲取|amplifies|枫叶舞|对1格内每个敌人造成自身最大生命值20%的法术伤害|effect=damage
神鹰锻匠|gives_equip|@f:大河流域|云中君会为距离最近的1个【大河】英雄添加1件神兵装备|pick=nearest;via=云中君
透支|gives_resource|player|能量可用至-15|res=energy;amt=15;note=borrow_next_round
狂铁·强化|gives_buff|狂铁|狂铁触发卡牌效果时，自身+10%攻速|effect=attack_speed
唤鹰骨笛|produces|云中君|召唤一名等级为自身一半的云中君|note=battle_summon;level=0.5*wearer.level
翩跹舞步·新生|produces|@maple_boost|为枫叶舞挑选1张强化牌|
翩跹舞步·新生|gives_level|@maple_holder|枫叶舞英雄每回合等级+2|amt=2;mode=real;per_round=2
翩跹舞步·流动|produces|@maple_boost|为枫叶舞挑选1张强化牌|
翩跹舞步·流动|amplifies|枫叶舞|获得额外的1次转移次数|effect=extra_transfer
翩跹舞步·蓄势|produces|@maple_boost|为枫叶舞挑选1张强化牌|
翩跹舞步·蓄势|gives_level|@maple_holder|枫叶舞英雄击败敌方英雄时永久等级+1|amt=1;mode=real;scale=kills
惊鸿游龙|produces|@pianxian|获得1张秘技牌翩跹舞步|
惊鸿游龙|amplifies|枫叶舞|为枫叶舞解锁1项特性（共4项）|
# ===== 装备
双剑·雄|exclusive_for|干将莫邪|若穿戴者为干将莫邪，其法力值上限降低至10|
双剑·雄|gives_mana|干将莫邪|其法力值上限降低至10|
玄微之种|grants_keyword|@wearer|穿戴者获得牺牲|keyword=牺牲
玄微之种|enables|tech:sacrifice_engine|穿戴者获得牺牲|role=granter
星源装甲|exclusive_for|@f:日落海|【日落海】专属装备|
星源狩枪|exclusive_for|@f:日落海|【日落海】专属装备|
星源星徽|exclusive_for|@f:日落海|【日落海】专属装备|
星源星徽|steals_level|@wearer|该英雄不会被阿科米亚核心分配等级|note=阻止核心分配
星源合金盾|exclusive_for|@f:日落海|【日落海】专属装备|
月影之力|exclusive_for|露娜|仅露娜穿戴时生效|
机关匣|produces|故障机器人|召唤2个拥有你等级最高英雄一半等级的“故障机器人”|
灵泉项链|gives_mana|@adjacent|周围1格的单位每秒回复4点法力值|
近卫荣耀|gives_buff|@adjacent|周围1格己方单位获得10点物理和法术防御|effect=defense
生命魔典|gives_buff|@adjacent|为自身和周围1格的己方单位恢复5%最大生命值|effect=heal
鼓舞之戒|gives_buff|*|与穿戴者同阵营的英雄，提升30%最大生命值|effect=hp;pick=same_faction_as_wearer
三分旗帜|gives_buff|@f:三分之地|平摊给所有三分英雄|effect=heal
法潮灵杖|steals_level|@adjacent|获得左右1格的英雄的100%法术攻击力|note=只复制不扣减，uncertain;kind=stat_copy
不屈意志|gives_level|@wearer|获得场上友军穿戴不屈意志数量x2的临时等级|mode=temp;scale=2*n_wearers
不屈意志|amplifies|不屈意志|将自身等级的30%平均分配给穿戴不屈意志的友军|note=多人同穿互相放大
怒莲之刺|pairs_with|银莲庇佑|受到的全部伤害转移给穿戴银莲庇佑的单位|
学识宝石|gives_level|@wearer|每回合开始时等级+8%（至少+3级）|mode=real;per_round=max(3,0.08*level)
学识宝石|gives_mana|@wearer|减少2点法力值上限|per_round=2
驱雷|exclusive_for|司空震|若命中司空震则改为使其快速攻击4次|
阿科米亚之子|grants_keyword|@wearer|参与阿科米亚核心的等级分配|keyword=核心分配
阿科米亚之子|gives_mana|@adjacent|穿戴者及周围1格单位每秒恢复4法力值|
灵泉法杖|grants_keyword|@wearer|获得闪现|keyword=闪现
捕能手套|gives_resource|player|每击败1名英雄，获得1能量|res=energy;scale=kills
# ===== 天赋
复印机|produces|@copy_of_chosen|指定1名1-3阶英雄，获得该英雄牌|
终极复印机|produces|@copy_of_chosen|指定1名英雄，获得该英雄牌|
二象性|produces|二象坍缩|获得1张天赋牌二象坍缩|
二象坍缩|produces|@same_tier_as_chosen|获得3张同阶英雄牌|note=指英雄牌
二象坍缩|produces|@same_faction_as_chosen|2张同阵营的当前最高阶英雄牌|
预言家|produces|胜利预言|分别获得1张天赋牌胜利预言与失败预言|
预言家|produces|失败预言|胜利预言与失败预言|
好聚好散|produces|体面离场|获得1张天赋牌体面离场|
体面离场|gives_resource|player|此英雄0/1/2/3觉醒进度时，获得3/6/9/12能量|res=energy;scale=3+3*awaken_progress
体面离场|enables|tech:clear_board_swap|指定移除一名英雄，获得能量|role=remover
瓶中人|produces|契约瓶|获得1张天赋牌契约瓶|
契约瓶|produces|@hero_any|打出后获得累积的英雄牌|scale=3*wins
随机成长|gives_level|@highest_tier|你品阶最高的1名英雄等级永久+4|amt=4;mode=real;per_battle=4
随机进化|gives_level|@highest_tier|你品阶最高的1名英雄等级永久+6|amt=6;mode=real;per_battle=6
随机强化|gives_level|@highest_tier|你品阶最高的1名英雄等级永久+9|amt=9;mode=real;per_battle=9
战争头牌|gives_level|@played|打出相同品阶的英雄牌时，额外+4等级|amt=4;mode=real
加速成长|gives_level|@chosen|指定1名英雄等级+5|amt=5;mode=real;pick=chosen
微弱优势|gives_buff|@awakened|打出觉醒英雄牌时，其获得22%暴击率和11%物理及法术吸血|effect=crit
双人游|produces|@hero_any|随机获得2张2-3阶英雄牌|tier=2-3
刻舟减蓝|gives_mana|*|英雄20级后，法力值上限-10|cond=level>=20
专注射击|gives_buff|@back_rows|你后两排英雄每次普攻获得3%攻速和3点物理攻击力|effect=attack_speed
超专注射击|gives_buff|@back_rows|你后两排英雄每次普攻获得5%攻速和5点物理攻击力|effect=attack_speed
银算盘|gives_level|*|出售场上英雄时，随机1名英雄等级+2|amt=2;mode=real;pick=random:1
银算盘|enables|tech:sell_hooks|出售场上英雄时|
金算盘|gives_level|*|出售场上英雄时，随机1名英雄等级+3|amt=3;mode=real;pick=random:1
金算盘|enables|tech:sell_hooks|出售场上英雄时|
班叔的遗留物|produces|2级电磁弹簧|获得2级电磁弹簧|
五福临门|produces|@hero_any|随机获得5张英雄牌|
兵行诡招|gives_level|*|己方每名英雄+2临时等级|amt=2;mode=temp;pick=all
一个响指|produces|听天由命|获得1张天赋牌听天由命|
听天由命|produces|简易协作|1张简易协作|scale=n_removed
听天由命|gives_resource|player|每消除1名，获得3能量|res=energy;scale=3*n_removed
听天由命|enables|tech:clear_board_swap|消除场上4名等级最低的英雄|role=remover
小小的助力|gives_buff|@tier:4-5|则4-5阶英雄获得30%伤害加成和60%攻速|effect=damage;cond=n_tier1_3>=4
补给线：开团|produces|@kw:开团|获得3张开团效果英雄牌|
中坚力量|gives_level|@tier:3|打出3阶英雄牌，额外+4级|amt=4;mode=real
中流砥柱|gives_level|@tier:3|打出3阶英雄牌，额外+6级|amt=6;mode=real
消费返利|produces|加速成长|获得1张加速成长|scale=energy_spent/15
人才市场|produces|人才搜寻|获得3张天赋牌人才搜寻|
人才搜寻|produces|@hero_any|卡牌3选1（商店可购的最高品阶英雄）|
人才济济|produces|人才搜寻|每次升级时获得1张天赋牌人才搜寻|
人才济济|gives_resource|player|升级所需棋手经验-2|res=xp
战前叫阵|gives_buff|@highest_level|我方胜利，其英雄获得1次完美复生|effect=revive
重装上阵|gives_buff|@equipped2|至少穿戴2件装备的英雄获得15%伤害加成和15%伤害减免|effect=damage
前线堡垒|gives_buff|@front_row|你第一排的英雄获得20点物理及法术防御|effect=defense
限制性进化|enables|tech:tier_evolve|指定移除1名1~2阶英雄，获得高2阶的随机英雄牌，且上阵时为原等级+10|role=variant_限制性进化
援军|produces|@same_faction_as_chosen|随机获得4张同阵营的英雄牌|
群英荟萃|gives_level|*|全体英雄等级+3|amt=3;mode=real;pick=all;cond=3_distinct_tiers_played
自适应武装|produces|@equip:normal|随机获得2张与其适配的普通装备牌|
虚构装备|gives_equip|*|随机3名己方英雄临时获得1件普通装备|pick=random:3;mode=battle
虚构武装|gives_equip|*|随机3名己方英雄临时获得2件普通装备|pick=random:3;mode=battle
强者自强|gives_buff|@awakened|已觉醒英雄获得40点物理攻击力和40点法术攻击力|effect=stat
强者恒强|gives_buff|@awakened|你的已觉醒英雄获得50点物理攻击力和50点法术攻击力|effect=stat
破败之力|gives_buff|*|你的英雄普攻附带敌人最大生命值2%的法术伤害|effect=damage
破败神威|gives_buff|*|你的英雄普攻附带敌人最大生命值3%的法术伤害|effect=damage
抢攻|produces|@equip:normal|获得1张普通装备牌|
厉兵秣马|produces|@equip:normal|随机获得1张普通装备牌|
擂台猛虎|produces|猛虎擂台|获得1个猛虎擂台|
擂台猛虎|gives_buff|@f:三分之地|擂台中的【三分】英雄获得30点物攻和法攻|effect=stat
拍卖专家|produces|拍卖重现：一|获得1张天赋牌拍卖重现|
超速成长|gives_level|@chosen|指定1名英雄等级+15|amt=15;mode=real;pick=chosen
王牌认证|produces|@effect_bag|获得1张2-3阶效果锦囊|
祝你好运|produces|碰碰运气|获得1张天赋牌碰碰运气|
碰碰运气|gives_level|@chosen|指定英雄使其随机提升5-20级|amt=5~20;mode=real;pick=chosen
临时增幅|gives_level|*|将20临时等级随机分配给己方英雄|amt=20;mode=temp;pick=random_split;per_round=+5
紧急行动|produces|装备箱|1张装备箱|
富贵险中求|produces|一刀穷一刀富|获得1张天赋牌一刀穷一刀富|
一刀穷一刀富|produces|@equip:base|获得1张基础装备牌|
一刀穷一刀富|produces|险中求富|1张险中求富|
险中求富|produces|@equip:base|获得基础装备牌|scale=wins
前浪的荣光|produces|@same_faction_as_chosen|获得1张高1-2阶的同阵营英雄牌|
全军列阵|produces|@same_faction_as_chosen|召唤其同阵营英雄上场直到阵容人数上限|note=直接上场
逆袭之路|gives_buff|*|你的所有单位伤害+10%，最大生命值+20%|effect=damage;cond=rank>2
绝境逆袭|gives_buff|*|你的所有单位伤害+20%，最大生命值+25%|effect=damage;cond=rank>2
小身体大能量|gives_buff|*|攻速+60%，暴击率+40%，生命值上限-20%|effect=attack_speed
一分钱一分货|produces|等价交换|获得1张天赋牌等价交换|
等价交换|produces|@hero_any|＜2，获得2张英雄牌|
等价交换|produces|@equip:base|2-4，获得1张基础装备牌|
等价交换|produces|@effect_bag|2张效果锦囊牌|
能量改造|triggers|@kw:整备|随机触发场上1名英雄的1次整备或登场效果|pick=random:1;scale=core_level/30
能量改造|triggers|@kw:登场|随机触发场上1名英雄的1次整备或登场效果|pick=random:1;scale=core_level/30
能量改造|steals_level|阿科米亚核心|阿科米亚核心不再分配等级|note=核心不再分等级
光棱塔|steals_level|阿科米亚核心|阿科米亚核心不再分配等级|note=核心改为造成法术伤害
瞬影连斩|gives_buff|@kw:闪现|闪现英雄击败敌方单位时，会立刻闪现至最近的敌人处，并无敌0.5秒|effect=untargetable
场外供能|gives_level|阿科米亚核心|阿科米亚核心获得棋手等级×3的等级|scale=3*lord_level;per_round=1
积极整备|produces|@equip:sunset|随机获得1件日落海专属装备|scale=n_prep_triggers/5
胜之不武|gives_level|*|下场战斗你的英雄+20临时等级|amt=20;mode=temp;pick=all
崛起|produces|马可波罗|获得1张马可波罗和1张百里守约|
崛起|produces|百里守约|1张百里守约|
崛起|pairs_with|马可波罗|共战时，获得3临时等级|with=百里守约
协同一致|gives_level|@f:无阵营|无阵营英雄获得等于当前棋手等级的等级|scale=lord_level;per_round=1;mode=real
荆棘牢笼|amplifies|先知图腾|先知图腾获得额外效果|
因缘际会|produces|因缘图腾|召唤1个1级的因缘图腾|
动能灌注|gives_level|阿科米亚核心|阿科米亚核心等级+10|amt=10
智能装甲|produces|@equip:sunset|随机获得1件【日落海专属装备】|
智能装甲|triggers|@kw:整备|其整备效果会额外触发1次|scale=n_sunset_equips
多彩古币|amplifies|古币|古币只能对非【河洛】英雄主动使用|note=改变古币用法：非河洛变身+7
坚不可摧|gives_buff|@totem|我方先知图腾、因缘图腾和往生图腾获得回合数x100的护盾|effect=shield
图腾启灵|produces|@totem|每回合随机召唤1个图腾，直到回合结束|
传家宝|amplifies|古币|古币不再附带转瞬。每回合开始时，手牌中古币等级+1|
古币翻新|produces|古币|获得1张古币|
古币翻新|amplifies|古币|古币牌会额外提供2级|amt=2
生命绽放|produces|鬼谷子|获得1张鬼谷子|
生命绽放|amplifies|玄微之种|玄微之种获得额外效果|
献祭之赐|produces|李信|获得1张李信|
献祭之赐|amplifies|李信|李信卡牌效果提供的临时等级额外+3|
贤士登门|produces|@f:逐鹿|随机获得1张【逐鹿】英雄牌|per_round=1
零钱兑换|produces|破损古币|自动兑换为2张无法提供等级的破损古币|
零钱兑换|amplifies|古币|你获得的古币会自动兑换为2张无法提供等级的破损古币|sign=negative
乘胜追击|produces|高阶推演战术|获得1张高阶推演战术|
最终守卫|gives_buff|@f:河洛|你最后1名【河洛】英雄受到致命伤害时进入无敌状态|effect=untargetable
古币契约|produces|古币|每触发3次牺牲效果，获得1张古币|scale=n_sacrifice/3
古币契约|enables|tech:sacrifice_engine|你的英雄每触发3次牺牲效果，获得1张古币|role=payoff
鲸吞|produces|庄周|获得1张庄周|
鲸吞|amplifies|庄周|庄周会吞噬敌方等级最低的1名英雄|
强力小弟|gives_level|@summons|你的召唤单位等级+6。每回合开始时，再+6|amt=6;per_round=6
幸运硬币|amplifies|古币|打出古币时有概率额外触发1-3次|
红豆之托|produces|王维|获得1张【河洛】英雄王维|
膨化生命|gives_buff|@f:三分之地|你的【三分】英雄额外获得100%生命值上限|effect=hp
乐善好施|gives_level|@f:三分之地|随机2名【三分】英雄等级+2|amt=2;mode=real;pick=random:2;per=登场
锦囊妙计|produces|锦囊|获得1张锦囊|
锦囊|triggers|@kw:登场|本回合打出的登场英雄牌，会额外触发1次登场效果|pick=played
一身是胆|produces|赵云|获得1张赵云|
一身是胆|gives_buff|赵云|赵云在释放技能前，会额外释放1个技能惊雷之龙|effect=cast
卧龙威势|produces|诸葛亮|获得1张诸葛亮|
卧龙威势|gives_mana|@f:三分之地|诸葛亮周围1格的其他【三分】英雄获得10点法力值|pick=adjacent_诸葛亮
怒涛|grants_keyword|@f:三分之地|【三分】英雄获得牺牲|keyword=牺牲
怒涛|enables|tech:sacrifice_engine|【三分】英雄获得牺牲|role=granter
控制感应|gives_buff|@f:三分之地|你的【三分】英雄获得20点法术攻击力|effect=stat;cond=施加控制
搏战|gives_buff|@f:三分之地|每秒恢复5%生命值，并提升50%攻速|effect=heal;pick=front_two_rows
三三得三|gives_level|@f:三分之地|他们分别获得33临时等级|amt=33;mode=temp;cond=n_三分%3==0
挂帅上阵|enables|tech:level_swap|使其等级等同于1名最近的【三分】英雄|role=variant_挂帅
四重奏|gives_level|*|全体英雄获得5×棋手等级的临时等级|mode=temp;scale=5*lord_level;cond=n_factions>=3
三分领袖|produces|三分旗帜|获得1张装备牌三分旗帜|
鬼神降临|produces|吕布|召唤吕布|scale=sum_dead_三分_levels/2
意外之财|produces|@effect_bag|获得1张随机1-5阶效果锦囊牌|scale=n_merge_逐鹿/4
令行如流|produces|@tactic|获得1张战术牌|per_round=1
穷兵黩武|produces|玄雍士兵|召唤1个玄雍士兵|
复现|amplifies|@effect|每回合首次对【逐鹿】英雄使用的效果牌会返回手牌并附加转瞬|
演武|amplifies|@tactic|你获得的战术牌都会变为高阶战术|cond=lord_level>=5
罡气护体|gives_buff|@f:逐鹿|每1秒获得自身最大生命值5%的护盾|effect=shield
钱生战力|gives_level|@f:河洛|随机4名【河洛】英雄获得累计使用古币数（选择此天赋后）的临时等级|mode=temp;pick=random:4;scale=n_coins_used
千金换酒|produces|李白|获得1张李白|
千金换酒|amplifies|李白|李白每获得12次古币，释放技能时会永久额外释放1次|
丝绸之路|produces|古币|每出售4名场上英雄获得1张古币|
丝绸之路|produces|@f:河洛|每出售1张古币获得1张【河洛】英雄牌|
丝绸之路|enables|tech:sell_hooks|每出售4名场上英雄获得1张古币|
春风化雨|produces|@on_board|获得1张场上【逐鹿】英雄的同名牌|pick=逐鹿
四海来援|grants_keyword|@f:无阵营|无阵营英雄获得牺牲|keyword=牺牲
四海来援|enables|tech:sacrifice_engine|无阵营英雄获得牺牲|role=granter
自我奉献|grants_keyword|@f:河洛|【河洛】英雄获得牺牲|keyword=牺牲
自我奉献|gives_level|*|所有友方单位+1临时等级，每次触发此牺牲效果时，临时等级额外+1|mode=temp;pick=all;scale=n_sacrifice
自我奉献|enables|tech:sacrifice_engine|【河洛】英雄获得牺牲|role=granter
熊魂附甲|grants_keyword|@f:大河流域|【大河】英雄阵亡获得牺牲|keyword=牺牲;cond=equip_count>=2
一起发财|gives_level|*|其他每个阵营中各1名随机英雄等级+3|amt=3;mode=real;pick=random_per_faction;per=古币
古币机|produces|@f:河洛|随机5张河洛英雄牌|
古币机|produces|贤者的庇护|1件贤者的庇护|
古币机|produces|超速成长|1张超速成长|
升值|produces|古币|获得2张古币|
升值|amplifies|古币|战斗胜利时，古币牌提供的等级永久+1|
灵祭升华|gives_level|@f:大河流域|等级最高的【大河】英雄获得图腾一半等级的临时等级|mode=temp;pick=highest
核心裂痕|produces|盗火|获得1张盗火|per_round=1
盗火|gives_level|@chosen|至多偷取阿科米亚核心20级|amt<=20;mode=real;pool=core
盗火|steals_level|阿科米亚核心|至多偷取阿科米亚核心20级|amt=20
能源置换|steals_level|阿科米亚核心|阿科米亚核心失去20级|amt=20
千里相聚|gives_level|*|每打出3张无阵营英雄牌时，全体英雄等级+4|amt=4;mode=real;pick=all
身外身|produces|孙悟空|获得1张孙悟空|
身外身|amplifies|孙悟空|孙悟空释放技能时，呼出2个猴毛化身|
兄妹|produces|铠|获得1张铠和1张露娜|
兄妹|produces|露娜|1张露娜|
兄妹|pairs_with|铠|共战时，铠首次释放技能，露娜可以无限释放技能|with=露娜
兽觉|produces|裴擒虎|获得1张裴擒虎和1张云中君|
兽觉|produces|云中君|1张云中君|
兽觉|pairs_with|裴擒虎|共战时，每当敌人被击败，二人获得兽觉|with=云中君
巨像|produces|米莱狄|获得1张米莱狄和1张蒙恬|
巨像|produces|蒙恬|1张蒙恬|
巨像|produces|蔚蓝神兽|召唤与米莱狄等级相同的蔚蓝神兽|
巨像|produces|猩红神兽|召唤与蒙恬等级相同的猩红神兽|
舞力全开|pairs_with|貂蝉|貂蝉、婉儿共战时二人共享法力值|with=上官婉儿
起舞|produces|貂蝉|获得1张貂蝉和1张上官婉儿|
起舞|produces|上官婉儿|1张上官婉儿|
起舞|pairs_with|貂蝉|共战时，参与击败敌方英雄后，二人均+10点法术攻击力并+1级|with=上官婉儿
箭雨|produces|虞姬|获得1张虞姬和1张公孙离|
箭雨|produces|公孙离|1张公孙离|
箭雨|pairs_with|虞姬|共战时，每合计攻击10次，将往敌方人群最密集处释放箭雨|with=公孙离
双生突袭|pairs_with|马可波罗|马可波罗、百里守约共战，释放技能时创造分身朝敌人释放技能|with=百里守约
双重分配|amplifies|阿科米亚核心|阿科米亚核心每分配1等级，还将随机分配1点属性|
毒奶|produces|蔡文姬|获得1张蔡文姬和1张扁鹊|
毒奶|produces|扁鹊|1张扁鹊|
毒奶|pairs_with|蔡文姬|共战时，扁鹊固定对蔡文姬释放技能|with=扁鹊
逐星|produces|李白|获得1张李白和1张曜|
逐星|produces|曜|1张曜|
逐星|pairs_with|李白|共战时，李白释放技能后，曜同步模仿李白|with=曜
宝盒|produces|露娜|获得1张露娜和1张孙悟空|
宝盒|produces|孙悟空|1张孙悟空|
宝盒|pairs_with|露娜|共战时二者任意一方受到致死伤害时会暂时离场|with=孙悟空
旋刃|produces|李元芳|获得1张李元芳和1张公孙离|
旋刃|produces|公孙离|1张公孙离|
旋刃|pairs_with|李元芳|共战时二人共享法力值（不会变更），技能变为融合技|with=公孙离
旋刃|gives_level|李元芳|旋刃每造成2次伤害，李元芳等级永久+1|mode=real;scale=hits/2
慕名而来|produces|@hero_any|每触发4次登场效果，随机获得1张英雄牌|scale=n_deploy_triggers/4
积少成多|gives_level|阿科米亚核心|【日落海】英雄每提升10级，阿科米亚核心获得4等级|scale=0.4*sunset_level_gain
超级蓄能|gives_level|@kw:整备|你的整备效果英雄会提升场上整备效果英雄数x2的等级|mode=real;per_round=1;scale=2*n_kw_整备
镜像迷离|gives_level|*|你的英雄偷取敌方同名英雄33%的等级|mode=battle;cond=enemy_same_name
摄人心魄|gives_level|@kw:夺取|你的夺取效果英雄击败敌方英雄时，将永久夺取对方3级|amt=3;mode=real;scale=kills
控魂|gives_buff|@kw:夺取|夺取效果英雄击败敌方英雄时，将其灵魂复活|effect=summon
能量汲取|gives_level|阿科米亚核心|出售场上英雄时，阿科米亚核心会获得其等级|scale=sold.level
能量汲取|enables|tech:sell_hooks|出售场上英雄时，阿科米亚核心会获得其等级|
双生相|gives_buff|@kw:闪现|第一排闪现英雄不再闪现，获得30%最大生命值上限以及10点双抗|effect=hp
生生不息|gives_buff|@summons|4/5阶的召唤单位获得完美复生|effect=revive
定向传输|amplifies|阿科米亚核心|阿科米亚核心的等级会优先分配给穿戴装备最多的【日落海】英雄，最多分配200级|
点兵点将|produces|@on_board|获得场上英雄牌各1张|
天兵|produces|@tier:5|获得1张5阶英雄牌。回合开始时，再获得1张|per_round=1
群星璀璨|gives_level|@tier:4-5|所有4-5阶英雄获得20临时等级|amt=20;mode=temp;cond=n_tier4_5>=5
无影手|gives_resource|player|回合开始时，获得3次仅本回合可用的免费商店刷新|res=refresh;amt=3;per_round=1
学霸|produces|留级证明|失败则获得留级证明|
买橘子去|produces|站这别动|获得1张天赋牌站这别动|
站这别动|produces|@on_board|每过2回合，获得1张此格上的英雄牌|
战略中枢|produces|有福同享|同时获得2张有福同享|
延迟满足|gives_level|@chosen|指定1名英雄等级+10。3回合后，再+25|amt=10+25;mode=real;pick=chosen
全副武装|gives_buff|@equipped2|至少穿戴2件装备的英雄获得15%伤害加成和减免|effect=damage
全副武装|produces|@equip:base|获得1张基础装备牌|
同心协力|gives_buff|*|若你至少有6名同阵营英雄，其获得50点物攻和法攻以及50%攻速|effect=stat;cond=n_same_faction>=6
三人游|produces|@tier:4-5|随机获得3张4-5阶英雄牌|
补给线：整备|produces|@kw:整备|获得1张整备效果英雄牌|per_round=1
补给线：牺牲|produces|@kw:牺牲|获得1张牺牲效果英雄牌|per_round=1
补给线：登场|produces|@kw:登场|获得1张登场效果英雄牌|per_round=1
混沌次元|produces|混沌裂隙|获得混沌裂隙|
混沌裂隙|enables|tech:rift_devour|开战前吞噬该格的英雄，永久获得其双倍等级|
小咖秀|produces|@on_board|随机获得1张场上的英雄牌|cond=only_tier1_3
永不独行|produces|双人同行·2阶|获得1张双人同行|
永不独行|produces|双人同行·3阶|获得1张双人同行|
双人同行·2阶|produces|@tier:2|随机获得2张相同的2阶英雄牌|
双人同行·3阶|produces|@tier:3|随机获得2张相同的3阶英雄牌|
大明星|produces|@tier:5|随机上阵1名不在场的等级为棋手等级x7的5阶英雄|note=直接上场
大明星|gives_awaken|@played|其觉醒进度为1，此后每经过2回合则额外+1|per_2_rounds=1
有舍有得|gives_awaken|*|随机1名英雄觉醒进度+1|pick=random:1;per_round=1
有舍有得|amplifies|曜|英雄合成提升的等级-1|sign=negative;note=与曜（合成+1）互相抵消
资深猎头|produces|@tier:4|4阶卡牌2选1，并将选中的卡牌一分为四|
控能达人|produces|@hero_any|随机获得1张3-4阶英雄牌|tier=3-4
王牌部队|produces|@effect_bag|获得1张4-5阶效果锦囊|
重大里程碑|produces|@effect|1张高阶效果牌|
等价交换|produces|5阶英雄兑换券|1张5阶兑换券|note=原文“5阶兑换券”，DB 名为 5阶英雄兑换券
紧急救援|produces|5阶英雄兑换券|1张5阶兑换券|note=原文简写
保持进步|produces|3阶英雄兑换券|获得1张3阶兑换券|note=原文简写
无限进步|produces|3阶英雄兑换券|获得1张3阶兑换券|note=原文简写
无阵营征召令|produces|@f:无阵营|获得2张无阵营英雄牌|
同路人|produces|@hero_any|随机获得1张该阵营英雄牌|pick=same_faction_as_played
成年礼|produces|@hero_any|随机获得2张其阵营的英雄牌|cond=hero_level>=30
凯旋征召|produces|@on_board|获得1张你等级最高的英雄牌|pick=highest_level;cond=win
如法炮制|produces|@hero_any|获得1张同阶的英雄牌|pick=same_tier_as_played
拍卖专家|produces|拍卖重现：二|获得1张天赋牌拍卖重现|
拍卖专家|produces|拍卖重现：三|获得1张天赋牌拍卖重现|
破格|produces|@hero_any|会出现1个高1阶英雄牌|note=出现在商店，仍需购买
节能商店|gives_resource|player|总有1张商店卡牌所需能量-1|res=energy;note=shop_discount
升格|amplifies|@summons|变为其召唤者（无卡牌效果）并保留等级|
守护灵|produces|@tier:5|随机召唤1名满装备的5阶英雄（无卡牌效果）出战|note=battle_only
命运图腾|produces|@totem|随机在其周围1格召唤命运图腾|note=命运图腾不在卡库
虚妄转生|produces|@totem|召唤1个虚妄图腾|note=虚妄图腾不在卡库
惊雷之龙|exclusive_for|赵云|赵云向敌人冲锋|
常胜将军|gives_resource|player|4棋手生命值|res=player_hp;amt=4
全域共鸣|amplifies|@equip:sunset|星源装备的【日落海】专属效果对穿戴它的任意英雄生效|note=星源装备=星源合金盾/星源装甲/星源狩枪/星源星徽
往生图腾|produces|往生图腾（山鬼形态）|往生图腾变身为往生图腾（山鬼形态）|
弦音|enables|tech:awaken_redeploy|使其觉醒进度+1|role=awaken_source
妙音|enables|tech:awaken_redeploy|使其觉醒进度+1|role=awaken_source
有舍有得|enables|tech:awaken_redeploy|随机1名英雄觉醒进度+1|role=awaken_source
# ===== 召唤物
先知图腾|gives_equip|@on_board|距离最近的2名己方英雄获得装备玄微之种|item=玄微之种;pick=nearest:2
先知图腾|produces|玄微之种|获得装备玄微之种|
因缘图腾|gives_level|*|所有己方英雄等级永久+1|amt=1;mode=real;pick=all;scale=attacks/25
机关傀儡|gives_level|机关傀儡|获得等同于你场上【逐鹿】英雄或召唤单位中最高等级一半的临时等级|mode=temp
猛虎擂台|gives_buff|@f:三分之地|擂台中的【三分】英雄获得30点物理和法术攻击力|effect=stat
猩红神兽|gives_buff|@summons|我方召唤物入场时使之获得猩红神兽之力|effect=stat
# ===== 英雄（hero_graph 之外补的：产出 / 喂养 / 放大 / 专属 / 手法角色）
盾山|produces|古币|牺牲：获得1张古币|
盾山|produces|精致古币|牺牲：获得1张精致古币|awaken=1
程咬金|produces|古币|整备：获得1张古币，对自身使用1张古币|auto_use=1
李元芳|produces|古币|整备：对距离最近的【河洛】英雄使用1张古币|auto_use=1;target=nearest_河洛
李元芳|produces|古币|整备：对距离最近的【河洛】英雄使用2张古币|auto_use=1;awaken=1;amt=2;target=nearest_河洛
上官婉儿|produces|古币|整备：随机对3名不同【河洛】英雄使用1张古币|auto_use=1;pick=random:3
武则天|produces|精致古币|登场：获得1张精致古币|
武则天|amplifies|古币|古币牌提升的等级+1|amt=1
武则天|amplifies|精致古币|古币牌提升的等级+1|amt=1
李白|amplifies|古币|古币牌会额外提供4临时等级|amt=4;mode=temp
古币|feeds|狄仁杰|对狄仁杰使用古币时，其攻击速度永久+5%|
精致古币|feeds|狄仁杰|对狄仁杰使用古币时|note=原文只写“古币”，精致古币是否计入 uncertain
古币|feeds|花木兰|本回合每使用5张古币牌，获得1次复生|
古币|feeds|钱生战力|累计使用古币数|
古币|feeds|一起发财|使用古币时，其他每个阵营中各1名随机英雄等级+3|
古币|feeds|古币机|每对其使用4张古币牌|
古币|feeds|千金换酒|李白每获得12次古币|
@effect|feeds|芈月|每当你使用效果牌时，己方穿戴装备最多的1名【逐鹿】英雄等级+1|
@effect|feeds|镜|每使用2张效果牌，等级+1|
@effect|feeds|沈梦溪|每对沈梦溪使用1张效果牌，本回合法力值上限-15|
@effect|feeds|白起|每回合首次使用效果牌时，获得1张援护战术|
@tactic|feeds|廉颇|每次使用战术牌会提升该护盾值|
@tactic|feeds|嬴政|每次使用战术牌会提升该加成|
@tactic|feeds|援护战术|+25×战术牌打出次数|
@tactic|feeds|高阶援护战术|+50×战术牌打出次数|
推演战术|amplifies|推演战术|每次使用此牌都会提升效果，上限20点|
白起|produces|援护战术|每回合首次使用效果牌时，获得1张援护战术|
白起|produces|高阶援护战术|获得1张高阶援护战术|awaken=1
西施|produces|推演战术|整备：获得1张推演战术|
西施|produces|高阶推演战术|整备：获得1张高阶推演战术|awaken=1
老夫子|produces|招募战术|每回合首次合成时，获得1张招募战术|
老夫子|produces|高阶招募战术|每回合首次合成时，获得1张高阶招募战术|awaken=1
庄周|produces|招募战术|登场：自身等级+2并获得1张招募战术|
庄周|produces|高阶招募战术|获得1张高阶招募战术|awaken=1
蒙犽|produces|特训战术|登场：获得1张特训战术|
蒙犽|produces|高阶特训战术|登场：获得1张高阶特训战术|awaken=1
鲁班七号|produces|@tactic|你的英雄每合成3次，随机获得1张战术牌|
杨玉环|produces|弦音|每过3个回合获得1张弦音|
杨玉环|produces|妙音|每过3个回合获得1张妙音|awaken=1
梦奇|produces|1阶英雄兑换券|退场：获得1张 1阶英雄兑换券|
梦奇|enables|tech:sell_hooks|退场：获得1张 1阶英雄兑换券|
干将莫邪|produces|双剑·雄|干将莫邪在场时，获得普通装备双剑·雄|
甄姬|produces|@kw:登场|登场：获得1张其他的登场英雄牌|
蒙恬|produces|玄雍士兵|牺牲：召唤2名与自身等级相同的玄雍士兵|
鬼谷子|produces|先知图腾|在场时，召唤先知图腾|
少司缘|produces|因缘图腾|在场时，召唤因缘图腾|
大司命|produces|往生图腾|在场时，召唤往生图腾|
鲁班大师|produces|机关傀儡|召唤1个等级与自身相同的机关傀儡|
项羽|gives_level|@totem|牺牲：所有图腾等级永久+3|amt=3;mode=real
大司命|gives_level|往生图腾|往生图腾等级永久+2并获得50点法力值|amt=2;mode=real;per=牺牲
姬小满|enables|tech:sell_hooks|每当你出售4名场上英雄后，获得1次临时刷新|
莱西奥|enables|tech:sell_hooks|每当你出售2名场上英雄后，阿科米亚核心等级+1|
# ===== 手法角色（原文点名）
百里守约|fits_role|tech:sacrifice_engine|每当你拥有牺牲效果的英雄阵亡时，本回合百里守约的法力值上限-20|role=payoff
李信|fits_role|tech:sacrifice_engine|每当你拥有牺牲效果的英雄阵亡时，李信临时等级+7|role=payoff
大司命|fits_role|tech:sacrifice_engine|每当你触发牺牲效果时：往生图腾等级永久+2|role=payoff
苏烈|fits_role|tech:sacrifice_engine|牺牲：苏烈的等级永久+2|role=fodder
盾山|fits_role|tech:sacrifice_engine|牺牲：获得1张古币|role=fodder
铠|fits_role|tech:sacrifice_engine|牺牲：你的【河洛】英雄获得5临时等级|role=fodder
蒙恬|fits_role|tech:sacrifice_engine|牺牲：召唤2名与自身等级相同的玄雍士兵|role=fodder
花木兰|fits_role|tech:sacrifice_engine|牺牲：临时等级+10|role=fodder
钟馗|fits_role|tech:sacrifice_engine|自身及左右1格的英雄获得牺牲|role=granter
瑶|fits_role|tech:yao_first_equip_share|开团：自身的首件普通装备分享给随机3名友方英雄|role=sharer
瑶|fits_role|tech:nearest_ally_buddy|选择最近的友方英雄作为保护目标|role=buddy
太乙真人|fits_role|tech:nearest_ally_buddy|开团：使距离最近的1名英雄获得复生|role=buddy
明世隐|fits_role|tech:nearest_ally_buddy|距离最近的英雄，临时等级中的 2级可转化为永久等级|role=buddy
李元芳|fits_role|tech:nearest_ally_buddy|对距离最近的【河洛】英雄使用1张古币|role=buddy
公孙离|fits_role|tech:steal_merge|夺取|role=stealer;excluded_by_user=1
猪八戒|fits_role|tech:steal_merge|夺取|role=stealer
孙悟空|fits_role|tech:steal_merge|闪现，夺取|role=stealer
# ===== 2026-10-26 卡库重抓新增的卡（发牌 / 被发的牌本身）
王维|produces|空山人语|获得空山人语和水穷云起各1张|
王维|produces|水穷云起|获得空山人语和水穷云起各1张|
空山人语|gives_level|@chosen|指定1名英雄等级+1|amt=1;mode=real;pick=chosen;scale=1+n_sacrifice
水穷云起|gives_level|*|全体英雄的等级+1|amt=1;mode=real;pick=all;scale=1+n_prep/2
捕能手套|produces|能量转换|单回合击败不少于5名英雄时，获得能量转换|
能量转换|produces|@effect_bag|获得与棋手等级相对应的效果锦囊牌|cost_energy=3
干将·强化|gives_equip|干将莫邪|干将莫邪获得1件【法源之杯】|item=法源之杯
干将·强化|produces|法源之杯|获得1件【法源之杯】|
干将·强化|exclusive_for|干将莫邪|干将莫邪获得1件【法源之杯】|
望梅止渴|produces|望梅止渴(赠送)|获得1张望梅止渴|
望梅止渴(赠送)|gives_level|@chosen|指定1名英雄临时等级+3|amt=3;mode=temp;pick=chosen;perm_if_win=1;scale=3+n_equip
点兵点将|produces|点兵点将(赠送)|获得1张天赋牌点兵点将|
点兵点将(赠送)|produces|@on_board|获得场上所有英雄牌各1张|
始祖熊灵|produces|始祖熊灵(赠送)|获得1张始祖熊灵|
空手道|gives_level|@chosen|该英雄获得等同于其真实等级的临时等级|mode=temp;pick=chosen;scale=real_level
挂帅|gives_level|@chosen|使其等级等同于1名最近的【三分】英雄|mode=real;pick=chosen;scale=nearest_三分.level-level
热血拳套|gives_buff|@wearer|攻击速度+30%|effect=attack_speed
套娃枪|produces|@wearer|每次普攻30%概率产生1个无法释放技能的分身|note=battle_only
"""


def parse_attrs(s):
    out = {}
    for kv in (s or "").split(";"):
        kv = kv.strip()
        if not kv:
            continue
        if "=" in kv:
            k, v = kv.split("=", 1)
        elif "<=" in kv:
            k, v = kv.split("<=", 1)
            v = "<=" + v
        else:
            k, v = kv, True
        if isinstance(v, str) and re.fullmatch(r"-?\d+", v):
            v = int(v)
        out[k.strip()] = v
    return out


EDGES: list[dict] = []
DROPPED: list[dict] = []


def add(frm, typ, to, quote, source, **attrs):
    EDGES.append({"from": frm, "to": to, "type": typ, "quote": quote, "source": source,
                  **({"attrs": attrs} if attrs else {})})


for line in MANUAL.strip().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = [p.strip() for p in line.split("|")]
    while len(parts) < 5:
        parts.append("")
    frm, typ, to, quote, attrs = parts[:5]
    add(frm, typ, to, quote, "manual", **parse_attrs(attrs))

# ---------------------------------------------------------------- 自动：棋手资源（非英雄卡）
RES_RX = [
    ("energy", r"(?:获得|改为获得|额外获得|再获得)(\d+(?:-\d+)?)能量"),
    ("xp", r"获得(\d+)棋手经验|棋手经验\+(\d+)"),
    ("refresh", r"获得(\d+)次免费刷新|获得(\d+)次临时刷新|获得1次临时刷新"),
    ("pop", r"可上阵人数\+1"),
    ("player_hp", r"(?:获得|恢复)(\d+)(?:点)?棋手生命值"),
]
MANUAL_RES = {(e["from"], e["attrs"].get("res")) for e in EDGES if e["type"] == "gives_resource" and e.get("attrs")}
for n, v in NODES.items():
    if v["type"] in ("hero", "summon"):
        continue
    raw = CARDS[n].get("skill") or ""
    for res, rx in RES_RX:
        if (n, res) in MANUAL_RES:
            continue
        m = re.search(rx, raw)
        if m:
            amt = next((g for g in m.groups() if g), None) if m.groups() else 1
            add(n, "gives_resource", "player", m.group(0), "auto_text", res=res, amt=amt if amt else 1)

# ---------------------------------------------------------------- 自动：产出（非英雄卡）
NAME_SET = sorted(NODES, key=len, reverse=True)
MANUAL_PROD = {(e["from"], e["to"]) for e in EDGES if e["type"] == "produces"}
MANUAL_PROD_FROM = {e["from"] for e in EDGES if e["type"] == "produces"}


AUTO_PROD_SKIP = {("古币机", "古币"),            # “每对其使用4张古币牌”是消耗，不是产出
                  ("红豆之托", "@f:河洛"),        # 给的是召唤物 王维（人工边已连）
                  ("虚构装备", "@equip:normal"), ("虚构武装", "@equip:normal")}   # 战斗内临时装备，走 gives_equip


def _prod(n, to, q, **a):
    if (n, to) in MANUAL_PROD or (n, to) in AUTO_PROD_SKIP:
        return
    MANUAL_PROD.add((n, to))
    add(n, "produces", to, q, "auto_text", **a)


for n, v in NODES.items():
    if v["type"] in ("hero", "summon"):
        continue
    raw = CARDS[n].get("skill") or ""
    # 1) 具体卡名：“N张/件 [天赋牌|秘技牌|特殊装备|装备牌] 卡名”
    for m in re.finditer(r"(\d+)\s*(?:张|件)(?:天赋牌|秘技牌|特殊装备|装备牌)?\s*", raw):
        rest = raw[m.end():]
        hit = next((nm for nm in NAME_SET if len(nm) >= 2 and rest.startswith(nm) and nm != n), None)
        if hit and not rest[len(hit):].startswith("牌"):      # “1张5阶英雄牌” ≠ 效果牌「5阶英雄」
            _prod(n, hit, raw[m.start():m.end() + len(hit)])
    # 2) “获得X”（天赋牌版装备：“怒神项链(天赋牌)”→ 怒神项链）
    mm = re.match(r"^(.+)\(天赋牌\)$", n)
    if mm and mm.group(1) in NODES:
        q = "获得" + mm.group(1)
        if q in raw:
            _prod(n, mm.group(1), q)
    # 3) 派系英雄牌
    for m in re.finditer(r"(?:张|选1（?|选1)((?:\d(?:-\d)?阶|4阶或5阶的)?)【([^】]+)】英雄", raw):
        fac = FAC_NORM.get(m.group(2))
        if fac:
            _prod(n, f"@f:{fac}", m.group(0), **({"tier": m.group(1).rstrip("的")} if m.group(1) else {}))
    # 4) 品阶英雄牌（不带派系）
    for m in re.finditer(r"(?:张|选1（)(\d)阶英雄牌", raw):
        _prod(n, f"@tier:{m.group(1)}", m.group(0))
    # 5) 关键词英雄牌
    for m in re.finditer(r"具有(整备|登场|牺牲|闪现|开团)效果的英雄牌|(整备|登场|牺牲|开团)效果英雄牌", raw):
        k = m.group(1) or m.group(2)
        _prod(n, f"@kw:{k}", m.group(0))
    # 6) 装备 / 战术 / 锦囊 / 效果 / 普通英雄牌 / 场上
    for rx, to in [(r"\d张基础装备牌", "@equip:base"), (r"\d张普通装备牌|\d件普通装备", "@equip:normal"),
                   (r"\d张战术牌", "@tactic"), (r"\d张[^，。]{0,6}效果锦囊", "@effect_bag"),
                   (r"\d张装备箱", "装备箱"), (r"获得\d张英雄牌|随机获得\d张英雄牌", "@hero_any")]:
        m = re.search(rx, raw)
        if m and to in NODES or (m and to.startswith("@")):
            _prod(n, to, m.group(0))

# ---------------------------------------------------------------- 自动：触发 → 展开到带关键词的英雄
for e in list(EDGES):
    if e["type"] == "triggers" and e["to"].startswith("@kw:"):
        k = e["to"][4:]
        EDGES.remove(e)
        for h in SCOPES[e["to"]]["members"]:
            add(e["from"], "triggers", h, e["quote"], e["source"], keyword=k, via=e["to"], **(e.get("attrs") or {}))

# ---------------------------------------------------------------- 自动：合法性 can_target / cannot_target（RuleEngine.card_target_ok）
from agent.decision.rules import RuleEngine          # noqa: E402
from agent.decision import hero_traits as HT          # noqa: E402

RE = RuleEngine(DATA / "chess_pieces.json", DATA / "cards.json")
TARGET_RULE = {}
# card_target_ok 没覆盖到、但原文明确的限制（补丁，source=manual_text）
# card_target_ok 的“同阵营”规则对这两张不成立/不确定 → 不出合法性边（宁缺毋滥），原因写进节点
# 2026-09-30 首次构建时审计出 card_target_ok 的三处错误（体面离场被“觉醒进度”误拦 / “1~2阶”没识别 / “同阵营”对二象坍缩·鼓舞之戒过严），
# 已在 rules.py 修掉，这里的补丁与豁免随之清空（保留变量，以后再发现问题可以临时补）。
NO_LIMIT_OVERRIDE: dict = {}
EXTRA_LIMIT: dict = {}
for n, v in NODES.items():
    if v["type"] not in ("effect", "equip", "talent", "lord_skill"):
        continue
    raw = CARDS[n].get("skill") or ""
    targeted = v["type"] == "equip" or "指定" in raw or raw.startswith("使1名英雄")
    if not targeted:
        continue
    ok, bad, reasons = [], [], {}
    for h in HEROES:
        r, why = RE.card_target_ok(n, h)
        if r and n in EXTRA_LIMIT and not EXTRA_LIMIT[n][1](h):
            r, why = False, f"{n} 只能给 {EXTRA_LIMIT[n][0]} 英雄"
        (ok if r else bad).append(h)
        if not r:
            reasons[h] = why
    state_req = []
    if re.search(r"指定\d*名已觉醒", raw):
        state_req.append("awakened")
    if re.search(r"觉醒进度\+", raw):
        state_req.append("not_awakened")     # 注意：card_target_ok 用的是“觉醒进度”四个字，会误伤 体面离场（见文档·数据问题）
    if state_req:
        v["target_state_req"] = state_req
    if n in NO_LIMIT_OVERRIDE:
        v["target_rule"] = "any"
        v["target_rule_note"] = NO_LIMIT_OVERRIDE[n]
        continue
    if not bad:
        v["target_rule"] = "any"
        continue
    # 原文里的限制短语作 quote
    q = None
    for rx in [r"触发其(整备|登场|牺牲|退场|开团)效果", r"指定\d*名(?:已觉醒的)?【[^】]+】英雄", r"可穿戴给【[^】]+】英雄",
               r"仅三分英雄穿戴", r"仅1-3阶英雄可穿戴", r"指定1名1-3阶的?英雄", r"限定远程英雄", r"限定近战英雄",
               r"仅[^\s，。]{1,5}穿戴时生效", r"指定移除1名1~2阶英雄", r"同阵营"]:
        m = re.search(rx, raw)
        if m:
            q = m.group(0)
            break
    if not q:
        q = raw[:30]
    src = "manual_text" if n in EXTRA_LIMIT else "rules.card_target_ok"
    v["target_limit"] = q
    if len(ok) <= len(bad):
        v["target_rule"] = "allow_list"
        for h in ok:
            add(n, "can_target", h, q, src, why=f"满足卡面限制「{q}」")
    else:
        v["target_rule"] = "deny_list"
        for h in bad:
            add(n, "cannot_target", h, q, src, why=reasons[h])

# ---------------------------------------------------------------- 自动：基础装备 → 四选一成品
for base, prods in RECIPES.items():
    q = re.search(r"可以铸造成1件[^。]+", CARDS[base]["skill"]).group(0)
    for i, p in enumerate(prods):
        add(base, "produces", p, q, "equip_recipes", slot=i)

# ---------------------------------------------------------------- 自动：装备适配 equip_fits / equip_avoid
def first_clause(txt):
    """装备/天赋原文的第一句“实际效果”（跳过“唯一”“唯一，无法拆卸”“唯一，限定远程英雄”这类纯标签句）。"""
    parts = [p.strip() for p in re.split(r"[。\n]", txt) if p.strip()]
    tag = r"(唯一|无法拆卸|限定(远程|近战)英雄|【日落海】专属装备|唯一被动：?)([，、](唯一|无法拆卸|限定(远程|近战)英雄))*"
    parts = [p for p in parts if not re.fullmatch(tag, p)]
    return parts[0] if parts else txt.split("\n")[0]


fit_seen = set()
for item, prof in EPROF["items"].items():
    if item not in NODES:
        continue
    q = first_clause(CARDS[item]["skill"])
    for h in prof.get("heroes_best") or []:
        if h in NODES:
            fit_seen.add((item, h))
            add(item, "equip_fits", h, q, "equip_profiles", why=prof.get("why", "")[:80], official=prof.get("official_count"))
    for h in prof.get("heroes_avoid") or []:
        if h in NODES:
            add(item, "equip_avoid", h, q, "equip_profiles", why=prof.get("why", "")[:80])
for h, hp in EPROF["heroes"].items():
    for rank, item in enumerate(hp.get("top_items") or []):
        if item in NODES and h in NODES and (item, h) not in fit_seen:
            fit_seen.add((item, h))
            add(item, "equip_fits", h, first_clause(CARDS[item]["skill"]), "equip_profiles.heroes", rank=rank)
        elif (item, h) in fit_seen:
            for e in EDGES:
                if e["type"] == "equip_fits" and e["from"] == item and e["to"] == h:
                    e.setdefault("attrs", {})["rank"] = rank
# 伤害类型相反（只对“主C 型”英雄生成，否则边数爆炸且没有意义）
avoid_seen = {(e["from"], e["to"]) for e in EDGES if e["type"] == "equip_avoid"}
carry_like = [h for h in HEROES if {"carry_phys", "carry_mag", "assassin"} & set(NODES[h]["combat_role"])]
for item, v in NODES.items():
    if v["type"] != "equip" or item in RECIPES:
        continue
    d = HT.item_dmg(item)
    if not d:
        continue
    ptype = (EPROF["items"].get(item) or {}).get("type", "")
    cls = (HT.EQUIP_CLASS.get(item) or ("",))[0]
    if not (ptype.startswith("输出-") or cls in ("dps_phys", "dps_mag")):
        continue                      # 只对“纯输出装”按伤害类型判不适配；功能/保命装（静谧之铃等）另有价值
    for h in carry_like:
        if (item, h) not in avoid_seen and (item, h) not in fit_seen and HT.item_mismatch(item, [h]):
            add(item, "equip_avoid", h, first_clause(CARDS[item]["skill"]), "hero_traits.item_mismatch",
                why=f"装备偏{d}，英雄伤害类型相反")

# ---------------------------------------------------------------- 自动：天赋 → 路线主C（carry_playbooks，评分 ≥8）
for key, r in PLAYBOOK["routes"].items():
    carry = r.get("carry")
    for t, info in (r.get("talents") or {}).items():
        if t in NODES and carry in NODES and (info.get("score") or 0) >= 8:
            add(t, "talent_boosts", carry, CARDS[t]["skill"].split(chr(10))[0], "carry_playbooks",
                score=info["score"], route=r.get("name"), why=(info.get("why") or "")[:80],
                official=bool(info.get("official")))
# 红叶染锋这类“收益依赖时机”的天赋/强化：附条件
for e in EDGES:
    if e["from"] == "红叶染锋" and e["type"] in ("enables", "talent_boosts"):
        e.setdefault("attrs", {})["cond"] = "old_holder.real_level >= 20（前期≈0 收益）"

# ---------------------------------------------------------------- hero_graph 原样并入
HG_KIND_MAP = {"level": ("gives_level", "real"), "perm_level": ("gives_level", "perm"), "temp_level": ("gives_level", "temp"),
               "core_level": ("gives_level", "real"), "mana": ("gives_mana", None), "buff": ("gives_buff", None),
               "shield": ("gives_buff", None), "revive": ("gives_buff", None), "heal": ("gives_buff", None),
               "steal": ("steals_level", None), "trigger": ("triggers", None), "equip": ("gives_equip", None),
               "awaken": ("gives_awaken", None), "position": ("grants_keyword", None)}


def hg_economy_type(q):
    if "能量" in q:
        return "gives_resource", {"res": "energy"}
    if "刷新" in q:
        return "gives_resource", {"res": "refresh"}
    if "经验" in q:
        return "gives_resource", {"res": "xp"}
    return "produces", {}


for e in HG["edges"]:
    to = HG_GROUP_MAP.get(e["to"], e["to"])
    if e["kind"] == "core_level":
        to = "阿科米亚核心" if e["to"] == "@日落海核心" else to
    if e["kind"] == "economy":
        typ, extra = hg_economy_type(e["quote"])
        to2 = "player" if typ == "gives_resource" else to
    else:
        typ, mode = HG_KIND_MAP[e["kind"]]
        extra = {"mode": mode} if mode else {}
        if e["kind"] in ("shield", "revive", "heal"):
            extra["effect"] = e["kind"]
        if e["kind"] == "position":
            extra["keyword"] = "牺牲"
        to2 = to
    EDGES.append({"from": e["from"], "to": to2, "type": typ, "quote": e["quote"], "source": "hero_graph",
                  "attrs": {**extra, "hg_kind": e["kind"], "hg_to": e["to"], "condition": e["condition"],
                            "strength": e["strength"], "needs_from_is_carry": e["needs_from_is_carry"]}})

# ---------------------------------------------------------------- 叠加链（事件 / 反应器 / 核心泵等手法 / 按等级成长的装备）
import card_graph_chains as CHAINS                     # noqa: E402  (tools/card_graph_chains.py)

_CTX = {"add": add, "NODES": NODES, "SCOPES": SCOPES, "TECHNIQUES": TECHNIQUES, "EDGE_TYPES": EDGE_TYPES,
        "HROLES": HROLES, "HT": HT, "CARDS": CARDS}
CHAINS.extend(_CTX)
# 按等级成长的装备：与已有 equip_avoid 冲突时 —— 学识宝石以“复利给主C”为准删 avoid；其余以 profiles 的 avoid 为准删新 fit；
# 与 profiles 已有的 fit 重复时去掉新加的那条
_ls = _CTX["LEVEL_SCALING_ITEMS"]
_avoid = {(e["from"], e["to"]) for e in EDGES if e["type"] == "equip_avoid" and e["from"] in _ls}
_fit_old = {(e["from"], e["to"]) for e in EDGES if e["type"] == "equip_fits" and e["from"] in _ls and e["source"] != "level_scaling"}
EDGES[:] = [e for e in EDGES if not (
    (e["type"] == "equip_avoid" and e["from"] == "学识宝石") or
    (e["source"] == "level_scaling" and e["from"] != "学识宝石" and (e["from"], e["to"]) in _avoid) or
    (e["source"] == "level_scaling" and (e["from"], e["to"]) in _fit_old))]

# ---------------------------------------------------------------- 图谱 v3（2026-10-09，只在 --v3 时生效）
# 图谱 v3：手法（超边 + 角色槽）与发英雄牌的卡池。
if V3:
    from agent.decision import meta as _META                 # noqa: E402  实测品阶概率 / 公共牌池（不是规则引擎的判断）

    EDGE_TYPES["grants_pool"] = (
        "发英雄牌的卡池：from（效果牌/天赋/技能牌）会给你英雄牌，to 是**卡池里的一个英雄**（按卡面限定语自动展开：关键词效果条目 / 阵营 / 品阶 / 图腾相关）。"
        "attrs：n=给几张、pick=random(随机)/choose:K选N(发现)/coupon(兑换券=3选1)、pool_size=卡池人数、p_each=该英雄进入可选范围的概率（按全池均匀估）、"
        "tier_rule=lord_level+1（品阶随棋手等级变，该边只在 tier==棋手等级+1 时成立）、via=keyword/entry_colon/filter（关键词行 / 只在“X：”条目里出现 / 阵营品阶过滤）。"
        "相对卡池（场上 / 同阵营 / 同阶 / 指定英雄）不展开，只在牌节点 hero_grant 里记 pool=relative。")

    # ---- 手法：停级搜牌 vs 先升级开池（同一决策的两面，eco 题的 roll / levelup 对应它们）
    TECHNIQUES["tech:slow_roll_lineup"] = {
        "name": "停级搜牌：主C 到手、缺的配套集中在低品阶时，停在配套刷得最多的等级刷牌凑齐，再一口气补等级",
        "steps": ["hold_level", "refresh × k", "buy(missing_support)", "(配套齐了) level_up × n"],
        "repeatable": True,
        "roles": {
            "carry": ["is_carry", "on_board or in_hand"],
            "support": ["not on_board", "not in_hand", "tier in [1, 3]",
                        "route_member or carry_support  # 运行时：路线当前阶段成员 / 以主C 为锚的官方阵容高频队友"],
        },
        "state_predicates": [
            "hit_sum(level) > hit_sum(level + 1)   # 缺的人在当前等级更好刷（hit_sum = Σ 缺的人 shop_hit(品阶, 等级)）",
            "gold >= 2                            # 刷得起（或有免费刷新）",
            "carry is set                         # 主C 已定（没主C 时谈不上配套）",
        ],
        "cost": ["人口不涨（只有 2→3、4→5 才 +1）", "高品阶池晚开", "等级落后进度表时，血量压力",
                 "低品阶牌在高等级会变稀（5 级 1 阶只有 1%）—— 先升级再回头刷 2~3 阶更难"],
        "benefit": "配套成型提前 N 回合（主C 配套效果 / 牺牲前排 / 连接辅助都依赖它）；每刷一次期望看到的缺件数 ≈ Σ shop_hit(q, L)",
        "counter_technique": "tech:level_open_pool",
        "timing": "主C 到手后、第 4~9 回合；缺的配套是 1~3 阶且 hit_sum(L) ≥ hit_sum(L+1)",
        "value": "每 2 能量 ≈ hit_sum(L) 个缺件出现；对比升一级：hit_sum(L+1) 与人口 +1",
        "status": "proposed",
        "evidence": "第 479 局：守约路线缺 盾山(2)/明世隐·钟馗·程咬金·铠(3)，3 级 hit_sum 0.378 > 4 级 0.352，按进度表升到 5 级后一直没刷到；"
                    "对应 TFT 的 slow roll / reroll comp（用户 2026-10-09：这也是一种手法）",
    }
    TECHNIQUES["tech:level_open_pool"] = {
        "name": "先升级开池：主C / 缺的配套是 4~5 阶（或下一级 +1 人口且手里有人能上）时，先升级开高品阶池，再刷",
        "steps": ["level_up × n", "refresh × k", "buy(missing_support)"],
        "repeatable": True,
        "roles": {
            "carry": ["is_carry or carry_eligible"],
            "support": ["not on_board", "not in_hand", "tier >= 4"],
        },
        "state_predicates": [
            "hit_sum(level + 1) > hit_sum(level)   # 缺的人在高一级更好刷（4~5 阶在低等级刷不出 / 很少）",
            "gold >= level_cost",
            "or: level_raises_pop(level) and heroes waiting in hand   # 2→3 / 4→5 多一个上阵位，手里有人能立刻上",
            "or: level + 1 == 6                    # 6 级解锁阿离专属天赋（落英汲取 / 枫之洪流 / 翩跹馈赠）",
        ],
        "cost": ["升级费（能量）", "低品阶牌变稀（2~3 阶配套更难刷）"],
        "benefit": "开高品阶池 / +1 人口 / 天赋三选一（5 级必出金、6 级专属）",
        "counter_technique": "tech:slow_roll_lineup",
        "timing": "4~5 阶主C 路线（如大河开团韩信）、或 2~3 阶配套已齐、或血量危险需要立刻变强（多一个人）",
        "value": "Δ人口 + Δhit_sum(高阶缺件) + 天赋",
        "status": "proposed",
        "evidence": "meta.TIER_ODDS（2026-09-24 实测）；docs/eco-roll-technique-2026-10-09.md 第 1 节",
    }

    # ---- 启用牌：免费刷新 → 停级搜牌；给棋手经验 → 先升级开池（quote = 卡面原文片段）
    _ENABLE_RX = (("tech:slow_roll_lineup", r"获得\d+次免费刷新|本回合刷新商店不消耗能量|获得\d+次仅本回合可用的免费商店刷新|每次手动刷新商店后[^。]*"),
                  ("tech:level_open_pool", r"获得\d+棋手经验|升级所需棋手经验-\d+"))
    for _n, _v in NODES.items():
        if _v["type"] not in ("effect", "talent", "lord_skill", "equip"):
            continue
        for _tid, _rx in _ENABLE_RX:
            _m = re.search(_rx, _v["text"])
            if _m:
                add(_n, "enables", _tid, _m.group(0), "auto_text_v3")

    # ---- 英雄节点：各棋手等级刷一次商店至少见到它一张的概率（满池近似，meta.shop_hit_prob）+ 最早进池等级
    for _h in HEROES:
        _q = NODES[_h].get("quality")
        if isinstance(_q, int) and 1 <= _q <= 5:
            NODES[_h]["shop_hit"] = {str(_L): round(_META.shop_hit_prob(_q, _L), 4) for _L in range(1, _META.MAX_LORD_LEVEL + 1)}
            NODES[_h]["min_level"] = _META.min_level_for(_q)

    # ---- grants_pool：发英雄牌的牌 → 卡池里的每个英雄（不写死卡名，按卡面限定语展开）
    _NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5}
    _GRANT_RX = (
        (re.compile(r"卡牌(\d)选(\d)（([^）]*?)英雄牌）"), "choose"),
        (re.compile(r"(\d)选(1)([^。；，（]*?)英雄牌"), "choose"),
        (re.compile(r"(?:随机)?(?:获得|抽取)(\d|[一两二三四五])张([^。；，：]*?)英雄牌(?!兑换)"), "get"),
        (re.compile(r"(?:获得|抽取)(\d|[一两二三四五])张([^。；，：]*?)(?:英雄)?兑换券"), "coupon"),
    )
    _RELATIVE = re.compile(r"场上|同阵营|该|其阵营|同阶|高\d|你等级最高|此格|转瞬|神秘|手牌|上一张|累积|同一|指定|当前最高")
    _TOTEM = set((SCOPES.get("@g:图腾") or {}).get("members") or [])

    def _kw_members(k):
        """卡池里“具有 K 效果的英雄”：卡面自带关键词（keywords_held —— 含“闪现”这种**不带冒号**的关键词行，
        只按“闪现：”匹配会得到 0 人）∪ 卡面里出现“K：”效果条目的（如钟馗“左右1格的英雄获得牺牲：…”）。"""
        held = {h for h in HEROES if k in NODES[h]["keywords"]}
        colon = {h for h in HEROES if f"{k}：" in NODES[h]["text"]}
        return held | colon, colon - held

    def _pool(qual):
        q = qual
        same = False
        m = re.fullmatch(r"相同的(\d)阶", q)
        if m:                              # 双人同行·N阶：2 张**相同的** N 阶英雄 —— 卡池 = N 阶，抽中一人就是两张
            q, same = f"{m.group(1)}阶", True
        elif _RELATIVE.search(q) or "相同" in q:
            return None, {"relative": q}
        kws = [k for k in KW_LIST if re.search(rf"具有{k}|{k}效果|^（?{k}$", q)]
        facs = sorted({FAC_NORM[x] for x in re.findall(r"【(.+?)】", q) if x in FAC_NORM} | ({"无阵营"} if "无阵营" in q else set()))
        tiers = None
        m = re.search(r"(\d)\s*[-~～]\s*(\d)阶", q)
        if m:
            tiers = set(range(int(m.group(1)), int(m.group(2)) + 1))
        elif re.search(r"(\d)阶及以下", q):
            tiers = set(range(1, int(re.search(r"(\d)阶及以下", q).group(1)) + 1))
        elif re.search(r"(\d)阶", q):
            tiers = {int(re.search(r"(\d)阶", q).group(1))}
        tier_rule = "lord_level+1" if "棋手等级+1" in q else ""
        pool, via_colon = set(HEROES), set()
        for k in kws:
            mem, col = _kw_members(k)
            pool &= mem
            via_colon |= col
        if facs:
            pool = {h for h in pool if NODES[h]["faction"] in facs}
        if tiers:
            pool = {h for h in pool if NODES[h]["quality"] in tiers}
        if "图腾" in q:
            pool &= _TOTEM
        spec = {"kw": kws, "faction": facs, "tiers": sorted(tiers) if tiers else None, "tier_rule": tier_rule,
                "totem": "图腾" in q, "same": same}
        return sorted(pool), dict(spec, via_colon=sorted(via_colon & pool))

    for _n, _v in NODES.items():
        if _v["type"] not in ("effect", "talent", "lord_skill", "equip"):
            continue
        _spans, _grants = [], []
        for _rx, _typ in _GRANT_RX:
            for _m in _rx.finditer(_v["text"]):
                if any(not (_m.end() <= a or _m.start() >= b) for a, b in _spans):
                    continue                                 # 同一句被前一个更具体的正则吃过了
                _spans.append((_m.start(), _m.end()))
                if _typ == "choose":
                    _cnt, _qual, _pick = int(_m.group(2)), _m.group(3), f"choose:{_m.group(1)}选{_m.group(2)}"
                else:
                    _cnt, _qual = int(_NUM.get(_m.group(1), _m.group(1))), _m.group(2)
                    _pick = "coupon" if _typ == "coupon" else "random"
                _pool_l, _spec = _pool(_qual)
                _g = {"quote": _m.group(0), "n": _cnt, "pick": _pick, **{k: v for k, v in _spec.items() if k != "via_colon"}}
                if _pool_l is None:
                    _g["pool"] = "relative"
                else:
                    _g["pool_size"] = len(_pool_l)
                    _g["via_colon"] = _spec["via_colon"]
                    # p_each：该英雄进入“可选范围”的概率（随机 n 张 = n/池；发现 K 选 N / 兑换券 3 选 1 = 看到 K 张的概率 ×（选得到））
                    _seen = (_cnt if _pick == "random" else
                             int(re.search(r"(\d)选", _pick).group(1)) if _pick.startswith("choose") else 3 * _cnt)
                    for _h in _pool_l:
                        _a = {"n": _cnt, "pick": _pick, "pool_size": len(_pool_l),
                              "p_each": round(min(1.0, _seen / (sum(1 for x in _pool_l if NODES[x]["quality"] == NODES[_h]["quality"])
                                                               if _spec["tier_rule"] else len(_pool_l))), 4),
                              "via": "entry_colon" if _h in _spec["via_colon"] else "keyword" if _spec["kw"] else "filter"}
                        if _spec["tier_rule"]:
                            _a["tier_rule"] = _spec["tier_rule"]
                            _a["tier"] = NODES[_h]["quality"]
                        if _spec["same"]:
                            _a["same"] = True
                        add(_n, "grants_pool", _h, _m.group(0), "auto_text_v3", **_a)
                _grants.append(_g)
        if _grants:
            _v["hero_grant"] = _grants

# ---------------------------------------------------------------- 节点：scope / technique / 理解
for sid, s in SCOPES.items():
    NODES[sid] = {"id": sid, "type": "scope", **s}
# 传承的 level_source 静态候选：hero_roles 里 carry_value<=3（等级对它价值低）且卡面自带被动涨级/功能价值高的
TECHNIQUES["tech:chuancheng_spread"]["static_candidates"] = {
    "rule": "hero_roles.carry_value <= 3 且 (functional_value >= 6 或 functions 含 self_growth)；运行时还要 real_level > 10",
    "heroes": sorted(h for h, r in HROLES.items() if (r.get("carry_value") or 0) <= 3
                     and ((r.get("functional_value") or 0) >= 6 or "self_growth" in (r.get("functions") or {}))),
}
for tid, t in TECHNIQUES.items():
    NODES[tid] = {"id": tid, "type": "technique", **t}

# ---------------------------------------------------------------- 校验
def seg_ok(quote, text):
    segs = [s for s in re.split(r"…+|\.\.\.", quote) if s.strip()]
    return bool(segs) and all(s in text for s in segs)


kept = []
for e in EDGES:
    for end in ("from", "to"):
        if e[end] not in NODES:
            DROPPED.append({**e, "why": f"{end} 节点不存在: {e[end]}"})
            break
    else:
        if e["type"] not in EDGE_TYPES:
            DROPPED.append({**e, "why": "未定义的边类型"})
            continue
        q = e["quote"]
        texts = [(e["from"], TEXT.get(e["from"], "")), (e["to"], TEXT.get(e["to"], ""))]
        where = next((nid for nid, t in texts if t and q and q in t), None)
        if where is None and e["source"] == "hero_graph":
            where = next((nid for nid, t in texts if t and q and seg_ok(q, t)), None)
            if where:
                e["quote_check"] = "segments"        # hero_graph 原样并入：含“…”，按段逐字校验
        if where is None:
            if e["source"] == "hero_graph":
                e["quote_check"] = "FAILED"          # 原样保留，但标出来
                kept.append(e)
            else:
                DROPPED.append({**e, "why": "quote 不是 from/to 文本的逐字子串"})
            continue
        e["quote_from"] = where
        e.setdefault("quote_check", "exact")
        kept.append(e)
# 去重（同 from/type/to/quote）
seen, EDGES_OUT = set(), []
for e in kept:
    k = (e["from"], e["type"], e["to"], e["quote"])
    if k in seen:
        continue
    seen.add(k)
    EDGES_OUT.append(e)

# --order-like PATH（2026-10-26）：节点顺序与参照图谱一致（参照里有的按参照顺序，新节点追加在末尾），
# 让已有节点的下标不变（cgnet 用节点下标查嵌入表）。不带此参数时顺序与以前相同。
_ord_arg = next((sys.argv[i + 1] for i, x in enumerate(sys.argv[1:-1], 1) if x == "--order-like"), None)
if _ord_arg:
    _ref = [n["id"] for n in json.loads(Path(_ord_arg).read_text(encoding="utf-8"))["nodes"]]
    _pos = {n: i for i, n in enumerate(_ref)}
    _keys = sorted(NODES, key=lambda n: (_pos.get(n, len(_ref)), 0))     # sorted 稳定：新节点保持原相对顺序
    NODES = {k: NODES[k] for k in _keys}

# 理解字段并入
missing_under = []
for n, v in NODES.items():
    if v["type"] in ("effect", "talent", "equip", "lord_skill"):
        u = UNDERSTAND.get(n)
        if u:
            v["understanding"] = u
        else:
            missing_under.append(n)
unknown_under = sorted(k for k in UNDERSTAND if k not in NODES and not k.startswith("_"))

# ---------------------------------------------------------------- 统计 / 输出
deg = Counter()
for e in EDGES_OUT:
    deg[e["from"]] += 1
    deg[e["to"]] += 1
for n in [n for n, v in NODES.items() if v["type"] == "scope" and deg[n] == 0]:
    del NODES[n]                      # 没有边引用的集合节点不输出
isolated = sorted(n for n, v in NODES.items() if deg[n] == 0)
iso_by_type = Counter(NODES[n]["type"] for n in isolated)

out = {
    "version": "2026-10-09-v3" if V3 else "2026-10-08",
    "doc": "docs/card-graph.md",
    "note": "全卡种异构关系图谱。节点 type ∈ hero/effect/equip/talent/lord_skill/summon/scope/technique；"
            "边 type 见 edge_types；quote 为 quote_from 节点文本的逐字子串（hero_graph 并入的含省略号边按段校验）。",
    "edge_types": EDGE_TYPES,
    "func_tag_vocab": [t for t, _ in FUNC_TAGS],
    "aliases_skipped": SKIP_ALIAS,
    "nodes": [{k: v for k, v in n.items() if k != "text"} | ({"text": n["text"]} if n["type"] not in ("scope", "technique") else {})
              for n in NODES.values()],
    "edges": EDGES_OUT,
    "meta": {
        "node_counts": dict(Counter(v["type"] for v in NODES.values())),
        "edge_counts": dict(Counter(e["type"] for e in EDGES_OUT)),
        "edge_sources": dict(Counter(e["source"] for e in EDGES_OUT)),
        "quote_check": dict(Counter(e["quote_check"] for e in EDGES_OUT)),
        "isolated_by_type": dict(iso_by_type),
        "isolated": isolated,
        "dropped": [{k: d[k] for k in ("from", "type", "to", "quote", "source", "why")} for d in DROPPED],
        "understanding_missing": missing_under,
        "understanding_unknown_keys": unknown_under,
    },
}
OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--show-dropped", action="store_true")
    ap.add_argument("--v3", action="store_true", help="图谱 v3（见文件头；已在模块顶部读过 argv）")
    ap.add_argument("--out", default=None, help="输出路径（已在模块顶部读过 argv）")
    ap.add_argument("--understanding", default=None, help="理解文件路径（已在模块顶部读过 argv）")
    ap.add_argument("--order-like", default=None, help="按参照图谱的节点顺序输出（已在模块里读过 argv）")
    a = ap.parse_args()
    m = out["meta"]
    print("节点:", m["node_counts"], "合计", len(NODES))
    print("边:", m["edge_counts"], "合计", len(EDGES_OUT))
    print("来源:", m["edge_sources"])
    print("quote 校验:", m["quote_check"])
    print("孤立节点:", m["isolated_by_type"])
    print("丢弃:", len(DROPPED))
    print("缺 understanding:", len(missing_under), " 多余键:", unknown_under[:10])
    if a.show_dropped:
        for d in DROPPED:
            print("  DROP", d["from"], d["type"], d["to"], "|", d["quote"][:40], "|", d["why"])
    if a.sample:
        rnd = random.Random(a.seed)
        for e in rnd.sample(EDGES_OUT, a.sample):
            print(json.dumps({k: e[k] for k in ("from", "type", "to", "quote", "source")} | {"attrs": e.get("attrs", {})},
                             ensure_ascii=False))
