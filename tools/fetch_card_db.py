"""官网卡牌数据库爬虫：从王者万象棋官网拉取结构化卡牌数据。

数据源（官网百科页前端直接加载的 JSON，纯 HTTP，秒级完成）：
  https://wxq.qq.com/cp/a20260707sfgw/index.html 的 js/cards.js / js/role.js 引用
  - oscard_new_1.js  → heroCards   英雄牌 85
  - oscard_new_2.js  → effectCards 效果牌 98
  - oscard_new_4.js  → equipCards  装备牌 73
  - oscard_new_8.js  → talentCards 天赋牌 255
  - oscard_new_16.js → lords       棋手牌 19

游戏更新后重跑本工具即可重建数据库，全程不依赖 AI、不需要开游戏。

用法:
  python tools/fetch_card_db.py                # 全量更新（数据 + 头像）
  python tools/fetch_card_db.py --no-images    # 只更新数据，不下载头像
  python tools/fetch_card_db.py --offline      # 不联网，用 data/web/ 里已下载的原始 JSON 重新转换
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "data" / "web"
CARDS_DIR = ROOT / "data" / "cards"
PORTRAIT_DIR = ROOT / "assets" / "image" / "pieces"

SOURCES = {
    "英雄牌": ("https://game.gtimg.cn/images/amside/ide_timer/589094_oscard_new_1.js", "heroCards"),
    "效果牌": ("https://game.gtimg.cn/images/amside/ide_timer/589094_oscard_new_2.js", "effectCards"),
    "装备牌": ("https://game.gtimg.cn/images/amside/ide_timer/589094_oscard_new_4.js", "equipCards"),
    "天赋牌": ("https://game.gtimg.cn/images/amside/ide_timer/589094_oscard_new_8.js", "talentCards"),
    "棋手牌": ("https://game.gtimg.cn/images/amside/ide_timer/589094_oscard_new_16.js", "lords"),
}

_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def strip_html(text: str) -> str:
    """去掉官网 desc 里的 <a>/<color>/<b> 标记，保留纯文本。"""
    return _HTML_TAG.sub("", text or "").strip()


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def derive_position(hero: dict) -> str:
    """由攻击距离推导前排/后排：近战(1格)为前排。"""
    props = (hero.get("heroCard") or {}).get("properties") or {}
    dist = props.get("attackDistance", 0)
    return "前排" if dist <= 1 else "后排"


def load_gui_tags() -> dict[str, list[str]]:
    """读 GUI 爬虫已收录的职业标签（坦克/法师/射手/刺客/辅助/战士），官网数据没有这一维。"""
    roles = {"坦克", "法师", "射手", "刺客", "辅助", "战士"}
    out: dict[str, list[str]] = {}
    p = CARDS_DIR / "英雄牌.json"
    if p.exists():
        try:
            db = json.loads(p.read_text(encoding="utf-8"))
            for name, card in db.get("cards", {}).items():
                role_tags = [t for t in card.get("tags", []) if t in roles]
                if role_tags:
                    out[name] = role_tags
        except Exception:
            pass
    return out


# 卡牌类型号 -> 类别名。previewCards 里的卡自带 `type`，靠它归类。
# **这张表要从主卡列表反推，别拿 URL 后缀号当 type**（我就填错过：
# URL 是 .._4.js/.._8.js，但里面主卡的 type 分别是 5 和 6）。
# 实测各主卡列表的 type 分布：英雄牌=1、效果牌=2、装备牌=5、天赋牌=6。
# 归错类后果很实在：「体面离场」被归成效果牌，就会掉进"拖给主力"的通用分支
# —— 而它的效果是**移除指定英雄**，等于自己把主力删了。
CARD_TYPE_NAME = {1: "英雄牌", 2: "效果牌", 5: "装备牌", 6: "天赋牌"}


def iter_preview_cards(raw: dict, default_category: str):
    """递归吐出 `previewCards` 里的卡（含它们自己的 previewCards）。

    **官网页面上这类卡是"翻页"才看得到的**（卡牌介绍里
    如果这张天赋会送额外的牌，右下角有翻页点可以切过去看）。
    爬虫原来只取第一页，于是这些牌**整层都没进库**。
    实测被丢掉的有 294 条，其中「体面离场」（天赋「好聚好散」赠送）
    害得决策层整局把它当不存在（日志刷了 33 次 "卡名未收录于 DB"）。
    """
    for sub in (raw.get("previewCards") or []):
        if not sub.get("name"):
            continue
        cat = CARD_TYPE_NAME.get(sub.get("type"), default_category)
        # **previewCards 里 type=1 的不能算「英雄牌」**：主列表那 85 位才是完整名册
        # （攻略的 9/20/24/18/14 和 WXHub 的"85英雄"两个独立来源都对得上，
        #  可以拿这个分布当校验）。这里的 41 张是**召唤物**（先知图腾、因缘图腾、
        # 往生图腾、玄雍士兵、机关傀儡、故障机器人…）和同名英雄的变体。
        # 但**也不能直接丢**：召唤物是**系统自动放上场**的（图腾类英雄一上场就带出图腾），
        # 还能被玩家移动 —— 场上见到它们时得查得到"这是什么"。
        # 所以单独归一类 `召唤物`：数据留着，又不会让 decide_buy 把图腾当成可买的英雄。
        # 召唤物**不占人口**（实测），位置靠**徽章观测**认，记账认不出来（不是我们放的）。
        _pdesc = re.sub(r"<[^>]+>", "", str(raw.get("desc") or raw.get("skill") or ""))     # 原文带 <color> 标签，"英雄<color>王维" 连不上
        if cat == "英雄牌" and f"英雄{sub.get('name')}" not in _pdesc:
            # 例外（2026-10-12）：天赋卡面写明"获得1张【河洛】英雄王维"的，是真英雄（红豆之托 → 王维），不是召唤物
            cat = "召唤物"
        yield sub, cat
        yield from iter_preview_cards(sub, default_category)


def convert_card(raw: dict, category: str, gui_tags: dict[str, list[str]]) -> dict:
    hero = raw.get("heroCard") or {}
    skills = []
    for s in hero.get("skillList") or []:
        skills.append({
            "name": s.get("name", ""),
            "desc": strip_html(s.get("desc", "")),
            "enhance": [{"level": p.get("level"), "desc": strip_html(p.get("desc", ""))}
                        for p in (s.get("enhanceSkill") or {}).get("params") or []],
        })
    awaken = hero.get("awakeingCard") or {}
    tags: list[str] = []
    if category == "英雄牌":
        pos = derive_position(raw)
        tags.append(pos)
        tags.extend(gui_tags.get(raw["name"], []))

    return {
        "id": raw.get("id"),
        "name": raw["name"],
        "category": category,
        "faction": raw.get("relationName", ""),
        "cost": (raw.get("shopBuyCost") or {}).get("Count", 0),
        "quality": raw.get("quality", 0),
        "tags": tags,
        "skill": strip_html(raw.get("desc", "")),
        "awaken": strip_html(awaken.get("desc", "")),
        "skills": skills,
        "properties": hero.get("properties") or {},
        "keywords": hero.get("keywordDescription") or [],
        "get_desc": raw.get("cardGetDesc", ""),
        "talent_group": (raw.get("talentCard") or {}).get("grpName", ""),
        "portrait": f"image/pieces/{category}/{raw['name']}.png",
        "thumb_url": raw.get("thumb") or raw.get("image") or "",
        "source": "web",
    }


# 官网未收录、但游戏里存在的卡牌（棋手技能牌等），人工维护补录。
#  effect_source 标注效果文本来源：manual=人工录入（待游戏内核对）。
# 合并 data/cards/*.json 时的**同名优先级**。
#
# 为什么需要：cards.json 是按**卡名**做 key 的字典，而游戏里确实存在同名不同类的卡，
# 实测 16 组。最要命的是「弈星」和「镜」——它们**既是棋手牌、又是英雄牌**：
#   弈星(棋手牌 id=36)  选棋手阶段的角色，skill 为空
#   弈星(英雄牌 id=1971) HERO NO.73，"棋手升级时，棋手经验+2" ← 对局里真正会买到的
# 原来的 `merged.update(cards)` 是后读的覆盖先读的，棋手牌排在英雄牌后面，
# 于是英雄版被整张吃掉，决策层查「弈星」只能查到一张 skill 为空的棋手牌，
# 当然也就不会去买它（2026-09-22 弈星一直不被使用，根因在这儿）。
#
# 另外 14 组是「装备牌 vs 天赋牌」同名（断念/末世/怒神项链…）：天赋版是"获得这件装备"
# 的天赋选项，装备版才是手牌里那件实物，决策层要的是后者。
#
# 数字越小越优先。输掉的那张不丢，挪到 "名字(类别)" 这个 key 下留档。
CATEGORY_PRIORITY = {
    "英雄牌": 0,       # 对局中最常打交道，且带 cost/faction/tags/skill，信息最全
    "装备牌": 1,
    "效果牌": 2,
    "棋手技能牌": 3,
    "天赋牌": 4,
    "棋手牌": 5,       # 一局只选一次，不会出现在商店/手牌里，优先级最低
}


def merge_cards(merged: dict, cards: dict) -> None:
    """把一个分库并进总库，同名按 CATEGORY_PRIORITY 决胜，输家改存到 "名字(类别)"。"""
    for name, rec in cards.items():
        old = merged.get(name)
        if old is None:
            merged[name] = rec
            continue
        new_p = CATEGORY_PRIORITY.get(rec.get("category"), 9)
        old_p = CATEGORY_PRIORITY.get(old.get("category"), 9)
        win, lose = (rec, old) if new_p < old_p else (old, rec)
        merged[name] = win
        merged[f"{name}({lose.get('category')})"] = lose
        print(f"  同名冲突 {name}: 取 {win.get('category')}(id={win.get('id')})，"
              f"{lose.get('category')}(id={lose.get('id')}) 存为 {name}({lose.get('category')})",
              flush=True)


MANUAL_CARDS: list[dict] = [
    # 注：「体面离场」曾在这里人工补录，**现已由爬虫的 previewCards 正常抓到**
    # （官网页面上它是「好聚好散」卡的第二页），补录已撤 —— 留着会反过来覆盖真数据。
    # 「N阶效果锦囊」（1~5 阶）：官网卡库没收录，但对局里经常拿到。
    # 机制（2026-09-22 确认）：
    #   来源 —— **主要是"每合成几次英雄，系统白送一张到你的卡牌里"**，也有少数
    #           效果牌会给；所以它不是买来的，是白来的，留着没有任何意义。
    #   效果 —— 打出后「**选择对应阶级的效果牌**」：要么弹一个 N 选 1 界面，
    #           要么直接把牌塞进手里。
    # **打出它会弹 N 选 1，而那个界面占着商店的位置、不选就一直卡着**，
    # 所以"打锦囊"和"处理 card_pick"必须当成一条链路（见 rules.SELF_CAST_NAME_RE
    # 和 decide_all 里 card_pick 的最高优先级分支）。
    # 不补录的话决策层查不到 category，`decide_use_cards` 直接跳过，它就一直烂在手里
    # 占手牌上限（曾出现 1 阶效果锦囊一直不用）。
    *[
        {
            "id": 900000 + tier,
            "name": f"{tier}阶效果锦囊",
            "category": "效果牌",
            "faction": "",
            "cost": 0,
            "quality": tier,
            "tags": [],
            "skill": f"选择1张{tier}阶效果牌（弹N选1界面或直接入手）。",
            "source": "manual",
        }
        for tier in range(1, 6)
    ],
    # 「空山人语」「水穷云起」：王维（天赋 红豆之托 送的河洛英雄）"开团：获得空山人语和水穷云起各1张"。
    # 官网王维的 previewCardIDs 是 311004 / 311005 但内容为空（第二层没展开），游戏内百科也只到王维这一层 → 查不到卡面。
    # 卡面 2026-10-15 人工补录（原文）。每回合开团都送、转瞬（不用就没了）；和牺牲 / 整备英雄配合好，比古币还好用。
    {
        "id": 311004, "name": "空山人语", "category": "效果牌", "faction": "河洛", "cost": 0, "quality": 2, "tags": [],
        "skill": "转瞬\n指定1名英雄等级+1，每触发1次牺牲效果，额外+1。",
        "source": "manual",
    },
    {
        "id": 311005, "name": "水穷云起", "category": "效果牌", "faction": "河洛", "cost": 0, "quality": 2, "tags": [],
        "skill": "转瞬\n全体英雄的等级+1，每触发2次整备效果，额外+1。",
        "source": "manual",
    },
    # 「装备箱」：天赋 里程碑 / 重大里程碑 / 紧急行动 / 紧急救援 送的，官网卡库没有它自己的卡面。
    # 2026-09-26 确认：装备箱直接使用即可，就是挑选一个装备 → 和效果锦囊一样是"打出 → N 选 1"。
    {
        "id": 900010,
        "name": "装备箱",
        "category": "效果牌",
        "faction": "",
        "cost": 0,
        "quality": 2,
        "tags": [],
        "skill": "使用：挑选1件装备（弹出装备N选1）。",
        "source": "manual",
    },
    # ---- 2026-10-09 卡牌覆盖审查补录 ----
    # 只收**有原文**的：卡面文字逐字取自送出它的那张牌（装备 / 天赋）的描述，不编。
    # 「能量转换」：捕能手套（装备，id 641005）原文"单回合击败不少于5名英雄时，获得能量转换。
    #   能量转换：消耗3能量，获得与棋手等级相对应的效果锦囊牌（不超过5阶）。"（第 449 / 478 局日志 162 次"未收录"）
    {
        "id": 900020,
        "name": "能量转换",
        "category": "效果牌",
        "faction": "",
        "cost": 0,
        "quality": 0,
        "tags": [],
        "skill": "消耗3能量，获得与棋手等级相对应的效果锦囊牌（不超过5阶）。",
        "get_desc": "来源: 由「捕能手套」赠送（单回合击败不少于5名英雄时）。",
        "source": "manual",
    },
    # 「挂帅」：天赋「挂帅上阵」（id 636316）原文"获得1张挂帅：指定1名【三分】英雄，使其等级等同于1名最近的【三分】英雄。"
    {
        "id": 900021,
        "name": "挂帅",
        "category": "天赋牌",
        "faction": "",
        "cost": 0,
        "quality": 3,
        "tags": [],
        "skill": "指定1名【三分】英雄，使其等级等同于1名最近的【三分】英雄。",
        "get_desc": "来源: 由「挂帅上阵」赠送。",
        "source": "manual",
    },
    # 「神秘英雄牌」：天赋「神秘英雄」（id 681002）原文"获得2张神秘英雄牌。手动刷新时，随机变为不低于当前棋手对应品阶的英雄牌。
    #   每回合有15%概率自动销毁并获得3能量。" —— 它本身打不出去（等手动刷新变身），所以**不能当英雄牌**（会搅乱 9/20/24/18/14 的英雄分组），
    #   归天赋牌；规则层不用、也不卖（rules._never_sell_hand）。
    {
        "id": 900022,
        "name": "神秘英雄牌",
        "category": "天赋牌",
        "faction": "",
        "cost": 0,
        "quality": 0,
        "tags": [],
        "skill": "手动刷新时，随机变为不低于当前棋手对应品阶的英雄牌。每回合有15%概率自动销毁并获得3能量。",
        "get_desc": "来源: 由「神秘英雄」赠送。",
        "source": "manual",
    },
    # 没补的（没有牌面原文，只有送出它的天赋的一句概述 —— 不编，待补截图）：
    #   根源之力 / 强敌克星（天赋送的**装备**，与天赋同名）、始祖熊灵（天赋送的单位牌，与天赋同名）、
    #   进攻型武装 / 均衡型武装 / 防御型武装 / 一莲托生（天赋三选一选项，卡库完全没有）。
]


def convert_lord_cards(lord: dict) -> dict:
    """把一位棋手的关联卡（relatedCards + talent[].cards）转成标准卡牌记录。

    每位棋手约 21 张专属强化/效果牌，全库 198 张。官网 lords 数据里本来就带着，
    但早期版本只提取了棋手本体和天赋，导致这些牌在决策层**完全不可见**
    ——2026-09-22 实战发现阿离的"翩跹舞步·新生"在手牌里被静默忽略才查出来。

    注意 type 字段无法区分类别（实测 type 7 同时对应棋手技能牌/效果牌/天赋牌），
    所以按来源分组判定：talent 里 name=="技能" 的算棋手技能牌，其余按效果牌处理。
    """
    lord_name = lord.get("name", "")
    groups = [("关联", lord.get("relatedCards") or [])]
    for t in (lord.get("talent") or []):
        groups.append((t.get("name", "技能"), t.get("cards") or []))

    out: dict[str, dict] = {}
    for group, items in groups:
        for c in items:
            name = c.get("name")
            if not name:
                continue
            out[name] = {
                "id": c.get("id", 0),
                "name": name,
                "category": "棋手技能牌" if group == "技能" else "效果牌",
                "faction": c.get("relationName", "") or "",
                "cost": (c.get("shopBuyCost") or {}).get("Count", 0),
                "quality": c.get("quality", 0),
                "tags": [],
                "skill": strip_html(c.get("desc") or ""),
                "awaken": "", "skills": [], "properties": {}, "keywords": [],
                "get_desc": c.get("cardGetDesc", ""),
                "talent_group": group,
                "lord": lord_name,
                "portrait": "",
                "thumb_url": c.get("thumb", ""),
                "source": "lord_related",
            }
    return out


def convert_lord(raw: dict) -> dict:
    talents = []
    for t in raw.get("talent") or []:
        talents.append({
            "name": t.get("name", ""),
            "level": t.get("level", 0),
            "cards": [{"id": c.get("id"), "name": c.get("name", ""),
                       "desc": strip_html(c.get("desc", ""))}
                      for c in t.get("cards") or []],
        })
    return {
        "id": raw.get("lordId"),
        "name": raw["name"],
        "category": "棋手牌",
        "faction": "",
        "cost": 0,
        "quality": 0,
        "tags": [],
        "skill": "",
        "talents": talents,
        "portrait": f"image/pieces/棋手牌/{raw['name']}.png",
        "thumb_url": raw.get("icon") or raw.get("portrait") or "",
        "source": "web",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-images", action="store_true", help="不下载头像")
    parser.add_argument("--offline", action="store_true", help="用 data/web/ 已有原始数据重新转换")
    args = parser.parse_args()

    gui_tags = load_gui_tags()
    if gui_tags:
        print(f"GUI 爬虫职业标签覆盖层: {len(gui_tags)} 个英雄")

    merged: dict[str, dict] = {}
    for category, (url, key) in SOURCES.items():
        raw_path = WEB_DIR / f"{category}.raw.json"
        if not args.offline:
            print(f"下载 {category}: {url}", flush=True)
            download(url, raw_path)
        d = json.loads(raw_path.read_text(encoding="utf-8"))
        rows = d.get(key) or []

        cards: dict[str, dict] = {}
        for raw in rows:
            if not raw.get("name"):
                continue
            rec = convert_lord(raw) if category == "棋手牌" else convert_card(raw, category, gui_tags)
            cards[rec["name"]] = rec
            # **翻页才看得到的赠送牌**：天赋/效果卡的 previewCards 里挂着它们送出的牌。
            # 不收进来的话，这些牌在对局里拿到手，决策层却查不到、只能静默跳过。
            for sub, sub_cat in iter_preview_cards(raw, category):
                sub_rec = convert_card(sub, sub_cat, gui_tags)
                sub_rec["get_desc"] = sub_rec.get("get_desc") or f"来源: 由「{rec['name']}」赠送。"
                old = cards.get(sub_rec["name"])
                if old is None:
                    cards[sub_rec["name"]] = sub_rec
                    continue
                # **同名不同效果不能静默丢**（2026-10-13 发现天赋送的同名牌被漏掉）：
                # 天赋「望梅止渴」送的牌也叫「望梅止渴」（临时等级 +3、每件装备 +1、赢了永久），原来 setdefault 把它丢了。
                # 效果一样就跳过；不一样 → 另存 "名字(赠送)"，并在同名的那张上记 given_skill（规则读卡面时一起读）
                new_txt, old_txt = (sub_rec.get("skill") or "").strip(), (old.get("skill") or "")
                if not new_txt or new_txt in old_txt:
                    continue
                key = f"{sub_rec['name']}(赠送)"
                if key not in cards:
                    cards[key] = sub_rec
                    old["given_skill"] = new_txt
                    print(f"  同名赠送牌 {sub_rec['name']}(id={sub_rec.get('id')})：效果和已有的不同，存为 {key}", flush=True)

        CARDS_DIR.mkdir(parents=True, exist_ok=True)
        out = CARDS_DIR / f"{category}.json"
        out.write_text(json.dumps(
            {"version": time.strftime("%Y-%m-%d"), "expected": len(rows),
             "source_url": url, "cards": cards},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {category}: {len(cards)} 张 -> {out}", flush=True)
        merge_cards(merged, cards)

        if not args.no_images:
            ok = skip = fail = 0
            for rec in cards.values():
                dest = PORTRAIT_DIR / category / f"{rec['name']}.png"
                if dest.exists():
                    skip += 1
                    continue
                try:
                    download(rec["thumb_url"], dest)
                    ok += 1
                except Exception as e:
                    print(f"    头像下载失败 {rec['name']}: {e}", flush=True)
                    fail += 1
            print(f"  头像: 新下载 {ok}，已有 {skip}，失败 {fail}", flush=True)

    # 人工补录卡（官网未收录）
    # 棋手关联卡（每位棋手 ~21 张专属强化/效果牌，全库 198 张）：
    # 官网 lords 数据里就带着 relatedCards / talent[].cards，但早期版本没有导入，
    # 导致这些牌在决策层完全不可见（2026-09-22 实战发现阿离的"翩跹舞步·新生"被忽略）。
    lords_raw = WEB_DIR / "棋手牌.raw.json"
    if lords_raw.exists():
        lords = json.loads(lords_raw.read_text(encoding="utf-8"))
        n_new = 0
        for lord in (lords.get("lords") if isinstance(lords, dict) else lords) or []:
            for name, rec in convert_lord_cards(lord).items():
                # 不覆盖官网正式卡条目，只填补缺失的 / 补上空描述
                if name not in merged:
                    merged[name] = rec
                    n_new += 1
                elif rec["skill"] and not merged[name].get("skill"):
                    merged[name]["skill"] = rec["skill"]
                    merged[name]["lord"] = rec["lord"]
        print(f"  棋手关联卡: 新增 {n_new} 张", flush=True)

    for rec in MANUAL_CARDS:
        merged[rec["name"]] = rec
        print(f"  人工补录: {rec['name']} ({rec['category']})", flush=True)

    # 聚合关键词库：每张卡内嵌了它用到的词条解释（keywordDescription），去重合并
    # 关键词库**只增不删**：先读回已有的，再用本次抓到的补充。
    # 2026-09-22 教训：离线重建（--offline）时原始数据里没有「转瞬」，
    # 直接重写文件就把它弄丢了——而"转瞬=回合结束自动销毁、出售不给能量"
    # 正是决策层判断结算时机的依据之一。
    kw_path = ROOT / "data" / "keywords.json"
    keywords: dict[str, str] = {}
    if kw_path.exists():
        try:
            keywords = json.loads(kw_path.read_text(encoding="utf-8")).get("keywords") or {}
        except Exception as e:
            print(f"  已有关键词库读取失败（将重建）: {e}", flush=True)
    for category, (_, key) in SOURCES.items():
        raw_path = WEB_DIR / f"{category}.raw.json"
        if not raw_path.exists():
            continue
        d = json.loads(raw_path.read_text(encoding="utf-8"))
        for card in d.get(key) or []:
            for sub in ("heroCard", "effectCard", "equipCard", "talentCard", "awakeingCard"):
                sub_obj = card.get(sub) or (card.get("heroCard") or {}).get(sub) or {}
                for kw in sub_obj.get("keywordDescription") or []:
                    if kw.get("name"):
                        keywords.setdefault(kw["name"], strip_html(kw.get("desc", "")))
    (ROOT / "data" / "keywords.json").write_text(json.dumps(
        {"version": time.strftime("%Y-%m-%d"), "keywords": keywords},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"关键词库: {len(keywords)} 条 -> data/keywords.json")

    # 合并版（兼容 rules.py 的单文件读取）
    (ROOT / "data" / "cards.json").write_text(json.dumps(
        {"version": time.strftime("%Y-%m-%d"), "cards": merged},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全部完成: 共 {len(merged)} 张，分库在 data/cards/，合并版 data/cards.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
