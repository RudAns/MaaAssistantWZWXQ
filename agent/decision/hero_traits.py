"""英雄的伤害类型 + 装备分类（装备配置用；规则引擎与 Laya 题面 tools/laya/facts.py 共用）。

2026-09-25 发现：主C 到最后只穿上 2 件装备；装备分配不能强制补护甲，要按英雄是物理伤害还是法术伤害来定。
第四轮第 1 局：主C 孙悟空（物理近战，卡库只有「前排」标签）被当成坦克，四选一按坦克偏好铸出了御灵之纱、反伤重甲。

依据只用卡库原始数据：英雄 properties.phyAttack / magAttack + 技能描述里"物理伤害/物理攻击力"与"法术伤害/法术攻击力"出现次数；
装备分类是主 agent 逐件读卡面定的（下表，理由写在旁边），8 种基础装备的 32 件成品全覆盖。
"""
from __future__ import annotations

from typing import Optional

import json
import re
from functools import lru_cache
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _cards() -> dict:
    return json.loads((PROJECT_DIR / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]


# 人工确认的伤害类型（卡面文字推不准的）：2026-10-17 确认亚连是物理伤害（卡面推成 hybrid）
DMG_OVERRIDE = {"亚连": "phys"}
LEGACY_DMG = False          # 题面渲染旧版本（< f9o）时由 features 置 True，保持旧题面逐字不变；规则层调用显式传 legacy=False


def dmg_type(name: str, legacy: bool | None = None) -> str:
    """"phys" / "mag" / "hybrid"；查不到返回 ""。人工确认表优先（legacy=True：旧题面版本不用确认表，保持逐字不变），
    其次技能文本里的伤害类型，数值面板最后。"""
    if legacy is None:
        legacy = LEGACY_DMG
    if not legacy and name in DMG_OVERRIDE:
        return DMG_OVERRIDE[name]
    c = _cards().get(name) or {}
    if c.get("category") != "英雄牌":
        return ""
    p = (c.get("properties") or {}).get("phyAttack") or 0
    m = (c.get("properties") or {}).get("magAttack") or 0
    txt = (c.get("skill") or "") + " ".join(s.get("desc", "") for s in c.get("skills") or [])
    cp = len(re.findall(r"物理伤害|物理攻击力", txt))
    cm = len(re.findall(r"法术伤害|法术攻击力", txt))
    if cp and not cm:
        return "phys"
    if cm and not cp:
        return "mag"
    if cp and cm:
        return "hybrid" if abs(cp - cm) <= 1 else ("phys" if cp > cm else "mag")
    if p >= m * 1.2:
        return "phys"
    if m >= p * 1.2:
        return "mag"
    return "hybrid"


def is_melee(name: str) -> bool:
    return (((_cards().get(name) or {}).get("properties") or {}).get("attackDistance") or 0) <= 1


# 成品装备分类：dps_phys 物理输出 / dps_mag 法术输出 / caster 靠放技能的都吃 / dps_any 任意输出 /
# survive 主C 保命 / tank 坦克防御 / util 功能。limit：只适合近战 / 远程。
EQUIP_CLASS = {
    # 风暴巨剑（输出型）
    "无尽巨剑": ("dps_phys", "", "暴击率+30%、暴击效果+60%：普攻暴击"),
    "铸梦·逐风": ("dps_phys", "", "每 8 秒快速 4 次普攻"),
    "制裁之剑": ("dps_any", "", "重伤：降低目标回复与护盾"),
    "纯净苍穹": ("dps_mag", "", "每 2 秒 +10 法术攻击力、清负面"),
    # 雷鸣刃（输出型）
    "冥王": ("dps_phys", "melee", "周围每有 1 个敌人 +15% 基础物攻：贴脸的近战"),
    "名刀·司命": ("survive", "", "首次致命伤害无敌 1.5 秒"),
    "红魔": ("dps_any", "", "独处/受伤叠增伤"),
    "魔律之刃": ("caster", "", "放技能 35% 重复释放"),
    # 速击之枪（输出型）
    "凝渊": ("dps_phys", "", "暴击率 +20% 且受击叠暴击"),
    "日渊": ("dps_phys", "ranged", "限定远程：普攻额外飞箭物理伤害"),
    "影牙": ("dps_any", "", "攻速 +50%：普攻型都吃"),
    "破晓": ("dps_phys", "", "普攻降物防、破防后附加物理伤害"),
    # 大棒（输出型）
    "虚无法杖": ("dps_mag", "", "法术穿透 +15"),
    "寒霜侵袭": ("dps_any", "", "普攻减速、叠满增伤 20%：普攻型"),
    "金色圣剑": ("dps_mag", "", "攻速 + 普攻附带法术伤害：普攻型法系"),
    "自然法杖": ("dps_mag", "", "法术攻击力 +20%"),
    # 圣者法典（法术型）
    "贤者之书": ("dps_mag", "", "放技能叠法术攻击力"),
    "炽热支配者": ("survive", "", "首次被控给护盾并免控 15 秒"),
    "生命魔典": ("tank", "", "受伤时给自身和周围回血"),
    "御灵之纱": ("tank", "", "按等级给护盾、破盾眩晕"),
    # 破碎圣杯（功能型）
    "痛苦面具": ("dps_mag", "", "法术大伤害点燃：真实伤害"),
    "法源之杯": ("caster", "", "每秒回 5 法力"),
    "破茧之衣": ("caster", "", "开场 +15 法力并眩晕敌方高等级英雄"),
    "聚灵水晶": ("tank", "", "受击回蓝，满了回血"),
    # 守护者之铠 / 力量腰带（防御型）
    "不祥征兆": ("tank", "", "格挡普攻"),
    "极寒风暴": ("tank", "", "受普攻降攻速/冻结"),
    "近卫荣耀": ("tank", "", "给周围友军加双抗"),
    "反伤重甲": ("tank", "", "反伤 30%"),
    "霸者重铠": ("tank", "", "高血量时 40% 减伤"),
    "不死鸟之眼": ("tank", "", "+15% 生命、回复与护盾 +30%"),
    "魔女斗篷": ("tank", "", "法抗 + 法术护盾"),
    "红莲炎甲": ("tank", "", "周身灼烧、降回复"),
    # 10-25 卡库重爬新增（天赋给的 3~5 阶唯一装备）
    "热血拳套": ("dps_any", "", "攻速 +30%、攻速上限 5.0、法力换攻速：普攻型"),
    "套娃枪": ("dps_phys", "ranged", "普攻 30% 出分身（承伤 500%）：远程普攻型，近战分身会被打死"),
    "根源之力": ("dps_mag", "", "每 4 秒打最低血敌人 600% 法攻"),
    "强敌克星": ("dps_any", "", "15% 伤害转真伤、普攻 4 次眩晕"),
}


def damage_source(hero: str) -> str:
    """卡面分析的伤害来源（data/hero_roles.json damage_shape.source）：skill / auto / mixed / ''。"""
    try:
        import json as _json
        from pathlib import Path as _P
        global _ROLES
        if _ROLES is None:
            r = _json.loads((_P(__file__).resolve().parents[2] / "data" / "hero_roles.json").read_text(encoding="utf-8"))
            _ROLES = r.get("heroes", r)
    except (OSError, ValueError):
        return ""
    return ((_ROLES.get(hero) or {}).get("damage_shape") or {}).get("source") or ""


_ROLES = None
# 靠普攻叠层 / 普攻触发的输出装（卡面：寒霜侵袭"普攻减速…叠满 5 层增伤"、金色圣剑"普攻附带法术伤害"）。
# 2026-10-19 第 791 局：守约不适合用寒霜侵袭（没法多次普攻其他英雄）—— 技能型主C 触发不了，卡面硬条件不满足。
BASIC_ATTACK_ITEMS = ("寒霜侵袭", "金色圣剑")
# 10-20（review-data-consistency 第 5 条）：同样只靠普攻生效的 4 件；线上 onhit=2 用，题面 f9z 起（f9x/f9y 传 1 只用上面两件，渲染不变）
BASIC_ATTACK_ITEMS_2 = ("影牙", "破晓", "日渊", "铸梦·逐风", "套娃枪")


def onhit_mismatch(item: str, hero: str, level: int = 2) -> bool:
    return (item in BASIC_ATTACK_ITEMS or (level >= 2 and item in BASIC_ATTACK_ITEMS_2)) and damage_source(hero) == "skill" and hero not in FAST_ATTACK_HEROES


def fits(item: str, hero: str, onhit: int = 2) -> tuple[bool, str]:
    """这件成品装备适不适合这个英雄当**输出装**（给主C / 输出位时用）。返回 (合适, 理由)。"""
    if onhit and onhit_mismatch(item, hero, int(onhit)):     # onhit=False：题面旧版本（f9x 前）照旧渲染
        return False, "basic-attack stacking item, hero deals SKILL damage"
    rf = rel_fit(item, hero)
    if rf is not None:
        if rf.get("score", 0) < 0 or rf.get("hard_mismatch"):
            return False, "equip_relations: " + str(rf.get("why", ""))[:60]
        return rf.get("score", 0) >= 5, "equip_relations: " + str(rf.get("why", ""))[:60]
    cls = EQUIP_CLASS.get(item)
    if not cls:
        return False, "unclassified"
    kind, limit, _ = cls
    ps = profile_says(item, hero)
    if ps == "avoid":
        return False, "equip_profiles: avoid"
    if ps == "best":
        return True, "equip_profiles: best"
    if USE_HERO_ITEM_PREF and LIVE_ITEM_LIMITS.get(item) == "assassin" and not is_assassin(hero):
        return False, "assassin-only effect"
    if limit == "melee" and not is_melee(hero):
        return False, "melee-only effect"
    if limit == "ranged" and is_melee(hero):
        return False, "ranged heroes only"
    dt = dmg_type(hero)
    if kind in ("dps_any", "survive", "caster"):
        return True, kind
    if kind == "dps_phys":
        return dt in ("phys", "hybrid"), f"physical item, hero is {dt or '?'}"
    if kind == "dps_mag":
        return dt in ("mag", "hybrid"), f"magic item, hero is {dt or '?'}"
    return False, kind           # tank / util：不当输出装


# 同一类里的强弱（主 agent 按卡面数值判断；越靠前越强）。只在同类之间排序时用。
STRENGTH = ["无尽巨剑", "破晓", "凝渊", "冥王", "日渊", "铸梦·逐风",          # 物理
            "自然法杖", "贤者之书", "痛苦面具", "纯净苍穹", "金色圣剑", "虚无法杖",  # 法术
            "红魔", "影牙", "寒霜侵袭", "制裁之剑",                          # 通用输出
            "魔律之刃", "法源之杯", "破茧之衣",                              # 施法
            "名刀·司命", "炽热支配者"]                                       # 保命


# 攻略点名的主C 装备（2026-09-29 的 T0 守约攻略："首选无尽战刃、炽热支配者、名刀"）—— 排最前，四选一也不跟 👍
HERO_ITEM_PREF = {"百里守约": ("无尽巨剑", "炽热支配者", "名刀·司命")}
# 默认关：离线重渲染 / 数据集要和训练时逐字一致（题面里有"会铸出什么"的预览）；实战由 live_match_agent 打开
USE_HERO_ITEM_PREF = False
# 实战才生效的装备限制（同上，默认不影响题面）。
# 红魔「周围 1 格没有友军时 +25% 伤害；每承受 3% 最大生命值伤害 +1%」—— 适合切后排、挨打的刺客；
# 后排射手/法师主C 旁边站着辅助（瑶/太乙/明世隐贴主C 放）、又不怎么挨打，吃不到（2026-09-30）
LIVE_ITEM_LIMITS = {"红魔": "assassin"}


_RELATIONS = None


def relations() -> dict:
    """装备 × 英雄关系表（data/equip_relations.json，2026-10-01 逐件读装备 / 英雄原文推导；构建 tools/build_equip_relations.py）。
    起因：马超的物理 / 法术归属判错，需要逐件分析装备描述、建立关系图。"""
    global _RELATIONS
    if _RELATIONS is None:
        try:
            import json as _json
            from pathlib import Path as _P
            _RELATIONS = _json.loads((_P(__file__).resolve().parents[2] / "data" / "equip_relations.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _RELATIONS = {}
    return _RELATIONS


def rel_fit(item: str, hero: str) -> Optional[dict]:
    """新关系表里 (英雄, 装备) 的条目 {score, why, hard_mismatch?, best_forge?}；只在实战开关打开时用（离线题面不变）。"""
    if not USE_HERO_ITEM_PREF or not item or not hero:
        return None
    return ((relations().get("fit") or {}).get(hero) or {}).get(item)


_PROFILES = None


def profiles() -> dict:
    """装备分类（data/equip_profiles.json，2026-09-30：所有装备分类，标出适合的职业 / 英雄及优先级）。"""
    global _PROFILES
    if _PROFILES is None:
        try:
            import json as _json
            from pathlib import Path as _P
            _PROFILES = _json.loads((_P(__file__).resolve().parents[2] / "data" / "equip_profiles.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _PROFILES = {}
    return _PROFILES


def profile_says(item: str, hero: str) -> str:
    """"best"：这件是该英雄前 6 件 / 该英雄在这件的 heroes_best；"avoid"：在 heroes_avoid；否则 ""。只在实战开关打开时生效。"""
    if not USE_HERO_ITEM_PREF or not hero:
        return ""
    rf = rel_fit(item, hero)
    if rf is not None:          # 新关系表优先（卡面硬条件：伤害类型 / 远近战 / 穿戴限制；官方次数只是 ±1 先验）
        sc = rf.get("score", 0)
        if sc < 0 or sc <= 2 or rf.get("hard_mismatch"):
            return "avoid"
        return "best" if sc >= 8 else ""
    p = profiles()
    it = (p.get("items") or {}).get(item) or {}
    if hero in (it.get("heroes_avoid") or []):
        return "avoid"
    if hero in (it.get("heroes_best") or []) or item in ((p.get("heroes") or {}).get(hero) or {}).get("top_items", []):
        return "best"
    return ""


def profile_rank(item: str, hero: str) -> float:
    """越小越优先：英雄前 6 件按序 0~5；其余按该英雄类型的优先级（10−分）+ 10；avoid 放最后。"""
    p = profiles()
    h = (p.get("heroes") or {}).get(hero) or {}
    top = h.get("top_items") or []
    if item in top:
        return top.index(item)
    it = (p.get("items") or {}).get(item) or {}
    if hero in (it.get("heroes_avoid") or []):
        return 100
    kind = h.get("kind") or ""
    pr = (it.get("carry_priority") or {}).get(kind) if kind not in ("tank", "support") else it.get("tank_priority")
    return 10 + (10 - (pr if isinstance(pr, (int, float)) else 3))


def is_assassin(hero: str) -> bool:
    try:
        import json as _json
        from pathlib import Path as _P
        global _CARD_TAGS
        if "_CARD_TAGS" not in globals():
            cards = _json.loads((_P(__file__).resolve().parents[2] / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
            _CARD_TAGS = {n: (v.get("tags") or []) for n, v in cards.items()}
        return "刺客" in _CARD_TAGS.get(hero, [])
    except (OSError, ValueError, KeyError):
        return False


def carry_order(recipes: list[str], hero: str, onhit: int = 2) -> list[str]:
    """基础装备铸造时给**输出位**的偏好：攻略点名的 > 匹配伤害类型的输出装 > 通用输出 > 施法 > 保命 > 其余。
    2026-10-21 第 884 局（大棒给物理主C 百里玄策选了自然法杖）：线上适配 live_fits 判不配的一律排到后面（和题面 / 穿装合法性同口径）。"""
    out = _carry_order_raw(recipes, hero, onhit)
    try:
        return [it for it in out if live_fits(it, hero)[0]] + [it for it in out if not live_fits(it, hero)[0]]
    except Exception:                                   # noqa: BLE001
        return out


def _carry_order_raw(recipes: list[str], hero: str, onhit: int = 2) -> list[str]:
    pref = HERO_ITEM_PREF.get(hero, ()) if USE_HERO_ITEM_PREF else ()
    base = _carry_order_base(recipes, hero, onhit)
    rs = {it: rel_fit(it, hero) for it in recipes}
    if all(v is not None for v in rs.values()) and rs:
        # 新关系表：按适配分从高到低（同分按 STRENGTH）；攻略点名的仍排最前，但硬错配的不排前
        base = sorted(recipes, key=lambda it: (-(rs[it].get("score") or 0),
                                               STRENGTH.index(it) if it in STRENGTH else 99, recipes.index(it)))
        if onhit:
            base = [it for it in base if not onhit_mismatch(it, hero, int(onhit))] + [it for it in base if onhit_mismatch(it, hero, int(onhit))]   # 普攻叠层装不给技能型（10-19）
        pref = tuple(it for it in pref if not (rs.get(it) or {}).get("hard_mismatch"))
        return [it for it in pref if it in recipes] + [it for it in base if it not in pref]
    if USE_HERO_ITEM_PREF and hero in (profiles().get("heroes") or {}):
        base = sorted(base, key=lambda it: (profile_rank(it, hero), base.index(it)))   # 装备分类的英雄优先级
    return [it for it in pref if it in recipes] + [it for it in base if it not in pref]


def _carry_order_base(recipes: list[str], hero: str, onhit: int = 2) -> list[str]:
    rank = {"dps_phys": 0, "dps_mag": 0, "dps_any": 1, "caster": 2, "survive": 3, "tank": 5, "util": 4}
    scored = []
    for i, it in enumerate(recipes):
        ok, _ = fits(it, hero, onhit)
        kind = (EQUIP_CLASS.get(it) or ("util",))[0]
        r = rank.get(kind, 4) if ok else 6 + rank.get(kind, 4)
        scored.append((r, STRENGTH.index(it) if it in STRENGTH else 99, i, it))
    return [it for *_, it in sorted(scored)]


def item_dmg(item: str) -> str:
    """装备偏向的伤害类型："mag" / "phys" / ""（通用或不认识）。
    四选一成品按 EQUIP_CLASS；其余成品（装备箱 / 拍卖 / N 选 1 拿到的，如 噬神之书）按卡面文字判。
    基础装备（圣者法典等，会弹四选一）不判 —— 铸成什么由四选一决定。"""
    cls = EQUIP_CLASS.get(item)
    if cls:
        return {"dps_mag": "mag", "dps_phys": "phys"}.get(cls[0], "")
    try:
        import json as _json
        from pathlib import Path as _P
        rec = _json.loads((_P(__file__).resolve().parents[2] / "data" / "equip_recipes.json").read_text(encoding="utf-8"))
        if item in rec:
            return ""
    except (OSError, ValueError):
        pass
    c = _cards().get(item) or {}
    if c.get("category") != "装备牌":
        return ""
    # **先看装备关系表**（data/equip_relations.json，逐件读卡面推的）：dmg_req = 数值部分要的伤害类型，
    # dmg_share = 价值里有多少靠这部分数值。价值基本不靠法攻 / 物攻的（保命 / 功能装）不分物理法术。
    # 2026-10-09 卡牌审查第 6 条：苍穹王权（致命免死 + 按已损失生命打法术伤害）、驱雷（按最大生命打法术伤害）
    # 卡面写着"法术伤害"就被判成法术装，物理主C 和坦克全穿不上；关系表里它们 dmg_share=0
    it = (relations().get("items") or {}).get(item)
    if isinstance(it, dict) and "dmg_req" in it:
        if (it.get("dmg_share") or 0) < 0.2:
            return ""
        return it.get("dmg_req") if it.get("dmg_req") in ("mag", "phys") else ""
    t = (c.get("skill") or "") + " ".join(x.get("desc", "") for x in c.get("skills") or [])
    # 「物理和法术吸血」（魂祭绝刃）两种都吃 = 通用；原来只匹配到"法术吸血"判成法术装，
    # 物理主C 守约整局拿不到它（2026-10-05 第 266 局）
    t = re.sub(r"物理[和与及]法术|法术[和与及]物理", "", t)
    # 只认**吃属性**的写法（法术攻击力 / 法术吸血 / 法术穿透 / 法术强度）；"造成…法术伤害"只说伤害类型，
    # 不代表要法攻（按生命值百分比 / 固定值打的根本不吃法攻）
    mag = any(k in t for k in ("法术攻击", "法术吸血", "法术穿透", "法术强度"))
    phys = any(k in t for k in ("物理攻击", "物理穿透", "物理吸血", "暴击"))
    return "mag" if mag and not phys else ("phys" if phys and not mag else "")


def item_mismatch(item: str, heroes) -> bool:
    """2026-09-26：噬神之书是法术道具，主C 与未来主C 都是物理系时不能选。
    heroes 全部读得出伤害类型、且都和装备相反（法术装 vs 物理 / 物理装 vs 法术；混合不算）时为 True。"""
    it = item_dmg(item)
    hs = [h for h in heroes if h]
    if not it or not hs:
        return False
    other = "phys" if it == "mag" else "mag"
    return all(dmg_type(h) == other for h in hs)


def base_kind(base: str) -> str:
    """基础装备的大类："output"（四个成品都是输出/施法/保命装：风暴巨剑/雷鸣刃/速击之枪/大棒）、
    "defense"（成品多为坦克装：守护者之铠/力量腰带）、"mixed"（圣者法典/破碎圣杯）、""（不是基础装备）。
    2026-09-26 发现给坦克铺了输出装备 —— 云中君铺装备时把雷鸣刃给了盾山（铸成冥王）。"""
    try:
        import json as _json
        from pathlib import Path as _P
        rec = _json.loads((_P(__file__).resolve().parents[2] / "data" / "equip_recipes.json").read_text(encoding="utf-8"))
        prods = (rec.get("recipes") or rec).get(base)
    except (OSError, ValueError, AttributeError):
        prods = None
    if not prods:
        return ""
    kinds = [(EQUIP_CLASS.get(x) or ("util",))[0] for x in prods]
    tank = sum(k == "tank" for k in kinds)
    if tank == 0:
        return "output"
    if tank >= 3:
        return "defense"
    return "mixed"


ROLE_TO_KIND = {"carry_phys": "phys_marksman", "carry_mag": "mage", "assassin": "assassin", "bruiser": "fighter",
                "tank": "tank", "support": "support", "utility": "support"}


def hero_kind(hero: str, combat_roles=None) -> str:
    """装备画像里的英雄类型：先查 equip_profiles.heroes（43 人），没有就按 hero_roles 的 combat_role 映射。"""
    h = (profiles().get("heroes") or {}).get(hero) or {}
    if h.get("kind"):
        return h["kind"]
    roles = list(combat_roles or [])
    if not roles:
        return ""
    if roles[0] == "controller":
        return "tank" if "tank" in roles else "support"
    return ROLE_TO_KIND.get(roles[0], "")


def item_fit_score(item: str, hero: str, combat_roles=None) -> Optional[float]:
    """这件装备对这个英雄的适配分 0~10（-1 = 不要给；None = 画像里没这件，不评）。
    依据 data/equip_profiles.json（=关系图谱 equip_fits/avoid 的来源）：英雄前 6 件 / heroes_best → 10，heroes_avoid → -1，
    否则按英雄类型取 carry_priority[kind] 或 tank_priority。**基础装备取它四个成品里最好的那件**（四选一会按人挑）。
    2026-09-30 改为按装备关系图判断谁适合什么装备（原来按"输出/防御 × 职业"一刀切，把守护者之铠给了孙膑）。"""
    rf = rel_fit(item, hero)
    if rf is not None:
        return float(rf.get("score", 0))
    p = profiles()
    items = p.get("items") or {}
    it = items.get(item)
    if not it:
        return None
    kind = hero_kind(hero, combat_roles)

    def one(name):
        d = items.get(name) or {}
        if hero in (d.get("heroes_avoid") or []):
            return -1.0
        if hero in (d.get("heroes_best") or []) or name in (((p.get("heroes") or {}).get(hero) or {}).get("top_items") or []):
            return 10.0
        if not kind:
            return None
        pr = d.get("tank_priority") if kind in ("tank", "support") else (d.get("carry_priority") or {}).get(kind)
        return float(pr) if isinstance(pr, (int, float)) else None

    forges = it.get("forges") or []
    scores = [x for x in ([one(item)] + [one(f) for f in forges]) if x is not None]
    if not scores:
        return None
    if forges:
        return max(scores)          # 基础装备：四选一会挑最适合的那件
    return scores[0]


# ---- 高攻速主C 配吸血（2026-10-19：高攻速英雄最好配一件带吸血的装备）----
_LIFESTEAL = None


def lifesteal_items() -> set:
    """equip_relations 里属性带"吸血"的成品（征服之剑 / 噬神之书 / 魂祭绝刃 / 星源装甲 / 掠魂爪…）。"""
    global _LIFESTEAL
    if _LIFESTEAL is None:
        try:
            import json as _j
            from pathlib import Path as _P
            d = _j.loads((_P(__file__).resolve().parents[2] / "data" / "equip_relations.json").read_text(encoding="utf-8"))
            items = d.get("items", d)
            _LIFESTEAL = {k for k, v in items.items() if isinstance(v, dict) and any("吸血" in s for s in (v.get("stats") or {}))}
        except Exception:                                   # noqa: BLE001
            _LIFESTEAL = set()
    return _LIFESTEAL


FAST_ATTACK_HEROES = {"虞姬", "公孙离", "马可波罗"}   # 马可波罗：2026-10-19 晚确认算攻速型（定位表标 skill）
ATTACK_SPEED_ADDED_F9X = {"马可波罗"}    # 题面 f9x 起才按攻速型写（旧版本渲染不变）     # 定位表标 mixed，但打法是攻速流（规范 42.6 快攻主C：狄仁杰 / 虞姬；阿离攻速流 = 公孙离）


LIVE_FAST_ADDED = {"韩信"}     # 2026-10-21 确认：韩信算攻速型，但也属于近战——只在线上执行层算（题面旧版本渲染不变；f9za 起题面再加）


def attack_speed_carry_live(hero: str) -> bool:
    """线上执行层用的攻速判定 = attack_speed_carry + LIVE_FAST_ADDED。"""
    return bool(hero) and (hero in LIVE_FAST_ADDED or attack_speed_carry(hero))


def attack_speed_carry(hero: str) -> bool:
    """高攻速英雄：伤害来源是普攻（hero_roles damage_shape.source），或卡面靠攻速成长（狄仁杰"攻击速度永久+5%"）。"""
    if not hero:
        return False
    if hero in FAST_ATTACK_HEROES:
        return True
    try:
        import json as _j
        from pathlib import Path as _P
        roles = _j.loads((_P(__file__).resolve().parents[2] / "data" / "hero_roles.json").read_text(encoding="utf-8"))
        roles = roles.get("heroes", roles)
        src = (((roles.get(hero) or {}).get("damage_shape") or {}).get("source") or "")
    except Exception:                                   # noqa: BLE001
        src = ""
    if src in ("auto", "attack"):
        return True
    from agent.decision import rules as _r  # noqa: F401  (只为卡库路径一致)
    try:
        import json as _j
        from pathlib import Path as _P
        cards = _j.loads((_P(__file__).resolve().parents[2] / "data" / "cards.json").read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return False

    def _find(o):
        if isinstance(o, dict):
            if o.get("name") == hero and o.get("category") == "英雄牌":
                return o.get("skill") or ""
            for v in o.values():
                r = _find(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = _find(v)
                if r is not None:
                    return r
        return None
    sk = _find(cards) or ""
    return "攻击速度" in sk or "攻速" in sk


# ---- 主C 必带免控（2026-10-19：主C 必带一件免疫控制的装备，或带"转移后几秒内免疫控制"的效果）----
CC_IMMUNE_ITEMS = {"炽热支配者", "星源装甲"}     # 机器头也免负面，但锁攻速、只给花木兰 / 嬴政（另有规则）


# ---- 线上按主C 判断的吸血 / 免控（2026-10-20 晚装备复审 data/review/equips_carry.json；题面仍用上面的旧表，f9za 再改）----
SUNSET_ONLY_ITEMS = {"星源装甲", "星源狩枪", "星源合金盾"}          # 专属效果只对【日落海】英雄生效（C41）


GLOBAL_RESONANCE = False          # 选了天赋「全域共鸣」：星源装备的日落海专属效果对任意英雄生效（rules 选天赋时置位、新局清零）


def _is_sunset(hero: str) -> bool:
    if GLOBAL_RESONANCE:
        return True
    try:
        from agent.decision import routes as _R
        return "日落海" in "".join((_R.pieces().get(hero) or {}).get("synergies") or [])
    except Exception:                                       # noqa: BLE001
        return False


def carry_lifesteal_items(hero: str) -> set:
    """对这个主C 真正算"吸血装"的：物理主C 不算噬神之书（只吸法术伤害）、算末世（普攻吸取等量生命）；星源系只算日落海。"""
    ls = set(lifesteal_items())
    if dmg_type(hero) == "phys":
        ls = (ls - {"噬神之书"}) | {"末世"}
    if not _is_sunset(hero):
        ls -= SUNSET_ONLY_ITEMS
    return ls


def carry_cc_items(hero: str) -> set:
    """对这个主C 真正免控的装备：星源装甲只对日落海生效（非日落海穿上不算有免控，照样去追炽热支配者）。"""
    return set(CC_IMMUNE_ITEMS) - (set() if _is_sunset(hero) else SUNSET_ONLY_ITEMS)
CC_IMMUNE_MAPLE = "枫影无拘"                     # 枫叶舞强化：转移后 5 秒免疫控制


# ---- 线上装备适配（2026-10-20 晚确认；题面仍用 fits，f9za 再改）----
# 2026-10-22 起这几张名单是**阿离打法偏好**，数据在 data/lords/<棋手>/lord_rules.json（写死的打法偏好一律绑定到棋手）：
#   模块级名字 = 导入时当前棋手的值（题面 facts.py 等照旧引用）；线上函数用 _lr() 每次按当前棋手取。下面的字面值是兜底（配置里没有时用）。
#   RANGED_ONLY（自然法杖，卡面远/近战法力不同）、SUNSET_ONLY_ITEMS（卡面"日落海专属"）是游戏事实，不挪。
_LR_DEFAULTS = {
    "no_onhit_carries": {"百里守约", "沈梦溪", "诸葛亮"},
    "fast_or_melee_good": {"魂祭绝刃"},
    "ranged_fast_only": {"分身镜"},
    "live_fit_any": {"不屈意志", "磐石之心"},
    "live_fit_low_tier": {"破阵战芒"},
}


def _lr(key: str) -> set:
    """当前棋手 lord_rules.json 的一张名单（取不到 → 上面的兜底值）。"""
    try:
        from agent.decision import lord_config as _LCh
        return _LCh.rule(key, _LR_DEFAULTS[key])
    except Exception:                                       # noqa: BLE001
        return _LR_DEFAULTS[key]


NO_ONHIT_CARRIES = _lr("no_onhit_carries")   # "技能型主C 不给攻速 / 普攻叠层装"只针对这三人（亚连 / 孙尚香 / 李信不限）
FAST_OR_MELEE_GOOD = _lr("fast_or_melee_good")                        # 攻速型非常有用、近战也能带；开战扣血不触发枫叶舞转移（2026-10-20 晚）
RANGED_ONLY = {"自然法杖"}                              # 2026-10-21 第 884 局确认只能远程用（卡面：远程每 40 法力变猪、近战 80）
RANGED_FAST_ONLY = _lr("ranged_fast_only")                           # 只适合远程高攻速英雄（2026-10-20 晚）
LIVE_FIT_ANY = _lr("live_fit_any")     # 磐石之心：2026-10-22 确认任何主C 都能带（很强的防秒杀装备）
LIVE_FIT_LOW_TIER = _lr("live_fit_low_tier")                       # 1-3 阶主C 都配：等级×1% 伤害加成 + 攻速（10-21 确认：守约 / 沈梦溪也能穿，有伤害加成）                            # 每 3 秒 +2×(穿戴数) 临时等级，算自己（规则确认第 10 条）→ 对谁都配


def live_onhit(hero: str) -> int:
    return 2 if hero in _lr("no_onhit_carries") else 0


def live_fits(item: str, hero: str) -> tuple:
    if item in _lr("live_fit_any"):
        return True, "live: fits any carry"
    if item in _lr("live_fit_low_tier"):
        q = int((_cards().get(hero) or {}).get("quality") or 9)
        return q <= 3, "live: tier 1-3 carries only"
    if item in SUNSET_ONLY_ITEMS:
        # 10-21 审查：日落海主C（亚连）原来落到 fits() 判 unclassified=False，星源装甲永远选不到（规范 54.30-12：这三件只给亚连）
        return _is_sunset(hero), ("live: sunset unique item" if _is_sunset(hero) else "live: sunset-only unique passive (rest too weak)")
    if item in _lr("fast_or_melee_good") and hero not in _lr("no_onhit_carries"):
        return (attack_speed_carry_live(hero) or is_melee(hero)), "live: attack-speed or melee carries"
    if item in RANGED_ONLY and is_melee(hero):
        return False, "live: ranged heroes only"
    if item in _lr("ranged_fast_only"):
        ok = (not is_melee(hero)) and attack_speed_carry_live(hero)
        return ok, "live: ranged fast-attack heroes only"
    return fits(item, hero, onhit=live_onhit(hero))



# ---- 非主C 装备：辅助性优先（2026-10-21 第 853 局：主C 之外的英雄多选辅助性装备，如眩晕、变猪、回血）
import re as _re_u
_U_CTRL = _re_u.compile(r"眩晕|变成小猪|变猪|冻结|击飞|沉默|禁锢|嘲讽|定身")                 # 控制敌人
_U_SUP = _re_u.compile(r"(恢复|回复)[^。]{0,8}生命|己方单位(获得|恢复)|友军获得")              # 回血 / 给队友加成
_U_DEF = _re_u.compile(r"获得[^。]{0,8}(防御|护盾)(?!穿透)|防御力\+|伤害减免|减伤|格挡|回敬|无敌|基础生命值|降低目标\d+%攻击速度|获得\d+点物理防御")   # 自保 / 抗伤


def utility_rank(item: str) -> int:
    """读卡面原文：0 控制敌人 / 1 回血·给队友加成 / 2 自保·抗伤（护盾 / 减伤 / 反伤）/ 3 其他（输出）。"""
    v = _cards().get(item) or {}
    t = str(v.get("skill") or v.get("desc") or "")
    if _U_CTRL.search(t):
        return 0
    if _U_SUP.search(t):
        return 1
    if _U_DEF.search(t):
        return 2
    return 3
