# -*- coding: utf-8 -*-
"""触发器 × 被触发效果 等级引擎表（2026-10-14 只读研究产物）。

生成 data/trigger_engines.json，说明见 docs/trigger-engines-2026-10-14.md。
- 卡面原文一律从 data/cards.json 现取（quote 与 quote_awaken），不手抄；卡库更新后重跑。
- 结构化字段（触发对象、次数、单次给谁多少级）是人读卡面后写的，写在本文件里，便于审。
- 每回合增益按 SCENARIO 的默认假设现算（公式写在 formula 字段里），改假设重跑即可。
- shop_freq：第 1~613 局 states.jsonl 里"内容不同的商店"中出现该牌的比例（按棋手等级分），
  2026-10-14 扫描结果直接写在 SHOP_FREQ 里（扫描脚本见报告附录）。

不改任何已有数据文件，不被 agent 引用。用法：
    .venv/Scripts/python.exe -X utf8 tools/build_trigger_engines.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARDS = json.load(open(os.path.join(ROOT, "data", "cards.json"), encoding="utf-8"))["cards"]
OUT = os.path.join(ROOT, "data", "trigger_engines.json")

# ---------------------------------------------------------------- 默认假设（中期，约第 8 回合）
SCENARIO = {
    "n": 7,            # 场上英雄数（随机 N 人的期望 = N/n）
    "carry_equips": 3, # 主C 穿 3 件，且是"穿戴装备最多的英雄"
    "K": 2,            # 闪现英雄上回合合计击杀数（玄策整备用）
    "H": 3,            # 场上河洛英雄数（上官婉儿随机 3 名河洛）
    "T": 2,            # 场上图腾数（虞姬开团）
    "D": 4,            # 一场战斗里"拥有牺牲效果的英雄"阵亡次数（李信/守约/大司命）
    "hp_lost": 30,     # 棋手已掉血（阿轲：每 10 点重复 1 次）
    "coin": 2,         # 1 张古币 = +2（武则天 +1、古币翻新 +2 另算）
}
S = SCENARIO

# ---------------------------------------------------------------- 商店出现率（按棋手等级 1~6）
SHOP_FREQ = {
    "临阵磨枪": [0, .034, .032, .022, .015, .006], "唤醒": [0, .034, .042, .023, .015, .008],
    "全军动员": [0, 0, 0, 0, .005, .006], "集体唤醒": [0, 0, 0, 0, .003, .006],
    "乱世": [0, 0, 0, .012, .015, .021], "图腾祝福": [0, 0, 0, 0, .004, .007],
    "孙悟空": [0, 0, .002, .042, .063, .078], "曹操": [.001, .079, .069, .040, .031, .017],
    "露娜": [0, 0, 0, .001, .017, .030], "雅典娜": [0, 0, .001, .030, .049, .057],
    "韩信": [0, 0, 0, .034, .057, .067], "张良": [0, 0, .041, .043, .058, .051],
    "朵莉亚": [0, 0, .036, .049, .045, .047], "小乔": [0, 0, .001, .031, .048, .059],
    "蒙犽": [0, .001, .002, .032, .049, .060], "周瑜": [0, 0, 0, .001, .019, .034],
    "明世隐": [.002, 0, .049, .052, .067, .063], "武则天": [0, 0, 0, 0, .020, .035],
    "百里玄策": [.004, .075, .075, .047, .030, .016], "安琪拉": [0, .001, .036, .044, .050, .044],
    "马可波罗": [.002, .067, .078, .049, .037, .015], "虞姬": [.003, .069, .075, .045, .029, .015],
    "李信": [0, 0, .001, .031, .050, .060], "百里守约": [.001, .065, .075, .053, .031, .015],
    "沈梦溪": [.001, .069, .075, .051, .039, .014], "狄仁杰": [0, 0, .035, .047, .054, .055],
    "云中君": [0, .001, .035, .043, .041, .051], "钟馗": [0, .001, .035, .047, .055, .051],
    "铠": [0, .001, .035, .044, .047, .048], "李元芳": [.244, .186, .081, .066, .007, .006],
    "程咬金": [0, .002, .035, .043, .046, .056], "上官婉儿": [0, 0, 0, .032, .051, .066],
}
# 历史执行次数（第 1~613 局，executed=true 的动作）
HISTORY = {
    "临阵磨枪": {"buy": 99, "card_pick": 92, "used": 228,
              "targets": {"亚连": 86, "安琪拉": 37, "百里玄策": 32, "百里守约": 20, "米莱狄": 14,
                          "李元芳": 11, "程咬金": 11, "狄仁杰": 9, "孙悟空": 2, "朵莉亚": 2},
              "note": "守约/狄仁杰/朵莉亚 没有『整备：』条目 = 白用（31 次，都在 2026-09-30 card_target_ok 修好之前）；孙悟空只用过 2 次"},
    "唤醒": {"buy": 6, "card_pick": 15, "used": 73,
           "targets": {"安琪拉": 32, "马超": 16, "明世隐": 9, "干将莫邪": 3, "曜": 3, "貂蝉": 3,
                       "马可波罗": 2, "露娜": 1, "武则天": 1, "阿轲": 1},
           "note": "安琪拉/马超/干将/曜 没有『登场：』= 白用 54 次（旧版）；近期给过 孙尚香（只换 1 次免费刷新）；蒙犽 一次没给过"},
    "全军动员": {"buy": 14, "card_pick": 42, "used": 36},
    "集体唤醒": {"buy": 1, "card_pick": 9, "used": 1},
    "开团号令": {"card_pick": 65, "used": 50}, "牺牲品": {"card_pick": 84, "used": 89},
    "整备高手": {"card_pick": 18, "used": 14}, "登场人物": {"card_pick": 14, "used": 17},
    "talent_picks": {"补给线：整备": 12, "超级蓄能": 12, "古币契约": 12, "自我奉献": 10, "补给线：登场": 7,
                     "补给线：开团": 5, "献祭之赐": 3, "智能装甲": 2, "乐善好施": 1, "锦囊妙计": 0, "能量改造": 0},
}

FLASH = [n for n, c in CARDS.items() if c.get("category") == "英雄牌" and "闪现" in (c.get("skill") or "")]
KW = ("整备", "登场", "开团", "牺牲")
HEROES_BY_KW = {k: sorted(n for n, c in CARDS.items() if c.get("category") == "英雄牌"
                         and re.search(k + "：", c.get("skill") or "")) for k in KW}


def card(n):
    c = CARDS.get(n)
    if not c:
        return {"name": n, "missing_in_cards_json": True}
    d = {"name": n, "category": c.get("category"), "quality": c.get("quality"),
         "faction": c.get("faction") or "", "quote": (c.get("skill") or "").replace("\n", " ").strip()}
    if c.get("awaken"):
        d["quote_awaken"] = c["awaken"].replace("\n", " ").strip()
    if c.get("given_skill"):
        d["given_skill"] = c["given_skill"]
    if c.get("talent_group"):
        d["talent_group"] = c["talent_group"]
    d["get_desc"] = (c.get("get_desc") or "").replace("来源: ", "")
    if n in SHOP_FREQ:
        d["shop_freq_by_level"] = dict(zip(range(1, 7), SHOP_FREQ[n]))
    return d


# ================================================================ 1. 触发器
# kind: retrigger=让某关键词效果额外结算 / converter=把 A 关键词变成 B 关键词的触发 /
#       breadth=增加持有该关键词的单位（触发次数随之变多）/ keyword_grant=给别人加上牺牲条目 /
#       revive=复生=多死一次=多一次牺牲 / self_double=觉醒后自己的整备触发 2 次
TRIGGERS = [
    # ---- 直接再触发
    dict(name="临阵磨枪", kind="retrigger", keyword="整备", target="指定 1 名英雄（必须自己有『整备：』条目）",
         times="每张 1 次", obtain="商店（棋手 2 级起，约 3%/次商店）/ 效果锦囊 / N选1 / 拍卖", ali_ok=True,
         effect_card=True, note="不是转瞬，可以攒到关键回合；本身算效果牌（镜/芈月/白起 也吃）"),
    dict(name="全军动员", kind="retrigger", keyword="整备", target="全体（所有有整备的英雄各 1 次）",
         times="每张 1 次", obtain="商店（5 级起，约 0.5%）/ N选1（历史 42 次）/ 效果锦囊 / 拍卖", ali_ok=True, effect_card=True),
    dict(name="唤醒", kind="retrigger", keyword="登场", target="指定 1 名英雄（必须有『登场：』条目）",
         times="每张 1 次", obtain="商店（2 级起，约 3~4%）/ 效果锦囊 / N选1 / 拍卖", ali_ok=True, effect_card=True),
    dict(name="集体唤醒", kind="retrigger", keyword="登场", target="全体", times="每张 1 次",
         obtain="商店（5 级起，约 0.3~0.6%）/ N选1 / 效果锦囊 / 拍卖", ali_ok=True, effect_card=True),
    dict(name="锦囊妙计", kind="retrigger", keyword="登场", target="本回合打出的登场英雄牌（含合成）",
         times="送 1 张锦囊；每张锦囊作用 1 回合，每张英雄牌多 1 次登场", obtain="天赋（天赋包四）", ali_ok=True,
         note="送的『锦囊』本身也是天赋牌，单独出现在初始天赋池"),
    dict(name="锦囊", kind="retrigger", keyword="登场", target="本回合打出的登场英雄牌", times="使用当回合每张 +1 次",
         obtain="天赋（初始天赋）", ali_ok=True),
    dict(name="智能装甲", kind="retrigger", keyword="整备", target="穿戴【日落海专属装备】的英雄",
         times="每件专属装备 +1 次/回合（常驻）", obtain="天赋（初始天赋），自带 1 件随机日落海专属装备", ali_ok=True),
    dict(name="能量改造", kind="retrigger", keyword="整备/登场", target="随机场上 1 名英雄",
         times="回合开始，核心每 30 级 1 次（上限 100）", obtain="天赋（天赋包四）", ali_ok=True,
         note="代价：核心不再给日落海英雄分配等级 —— 对靠核心分等级的主C（马可/安琪拉/亚连/艾琳）是负收益；对韩信（数触发）是纯收益"),
    # ---- 关键词转换器（本身是英雄，自带的效果就是"触发别人的效果"）
    dict(name="曹操", kind="converter", keyword="整备→登场", target="随机 1 名【三分】英雄（觉醒 2 名）",
         times="每回合 1 次（被临阵磨枪/全军动员再触发时再 1 次）", obtain="商店 2 阶（2~3 级约 7~8%）", ali_ok=True),
    dict(name="露娜", kind="converter", keyword="登场→整备", target="随机 2 名英雄（觉醒 4 名）",
         times="打出/合成/唤醒 时", obtain="商店 5 阶（5 级起 1.7~3%）", ali_ok=True,
         note="本身也是一次登场（小乔/乐善好施/大乔 吃），且是闪现英雄（孙悟空 吃）"),
    dict(name="雅典娜", kind="converter", keyword="开团→整备", target="随机 2 名【日落海】英雄（觉醒 4 名）",
         times="每场战斗 1 次", obtain="商店 4 阶（4 级起 3~6%）/ 天赋 日落海来客（送 1 张）", ali_ok=True,
         note="一人 = 1 次开团 + 2 次整备 = 韩信 +6/回合；场上日落海整备英雄不足 2 个时实际触发数待实测"),
    # ---- 觉醒自倍增
    dict(name="狂铁/亚连/米莱狄/安琪拉（觉醒）", kind="self_double", keyword="整备", target="自己",
         times="觉醒后整备『触发2次』", obtain="合成觉醒", ali_ok=True,
         note="『触发2次』对 韩信/朵莉亚/积极整备 是否计 2 次未实测（按卡面字义应计 2 次）"),
    # ---- 广度：送带关键词的英雄牌（触发次数 = 场上持有该关键词的单位数）
    dict(name="补给线：整备", kind="breadth", keyword="整备", target="—", times="立得 1 张整备英雄牌，之后每回合 +1 张",
         obtain="天赋（初始天赋，历史选过 12 次）", ali_ok=True),
    dict(name="补给线：登场", kind="breadth", keyword="登场", target="—", times="立得 1 张 + 每回合 1 张", obtain="天赋（初始天赋）", ali_ok=True),
    dict(name="补给线：开团", kind="breadth", keyword="开团", target="—", times="一次性 3 张开团英雄牌", obtain="天赋（初始天赋）", ali_ok=True),
    dict(name="整备高手", kind="breadth", keyword="整备", target="—", times="一次性 2 张", obtain="效果锦囊 / N选1", ali_ok=True),
    dict(name="登场人物", kind="breadth", keyword="登场", target="—", times="一次性 2 张", obtain="效果锦囊 / N选1", ali_ok=True),
    dict(name="开团号令", kind="breadth", keyword="开团", target="—", times="一次性 2 张", obtain="效果锦囊 / N选1（历史 65 次）", ali_ok=True),
    dict(name="牺牲品", kind="breadth", keyword="牺牲", target="—", times="一次性 2 张", obtain="效果锦囊 / N选1（历史 84 次）", ali_ok=True),
    dict(name="乱世", kind="breadth", keyword="登场", target="—", times="3 选 1 登场英雄牌", obtain="商店 4 阶效果牌（4 级起 1~2%）", ali_ok=True),
    dict(name="甄姬", kind="breadth", keyword="登场", target="—", times="登场：拿 1 张别的登场英雄牌（觉醒 3 选 1）", obtain="商店 4 阶", ali_ok=True,
         note="被唤醒/合成时再拿一张 → 登场链"),
    dict(name="闪现突击者", kind="breadth", keyword="闪现", target="—", times="一次性 2 张闪现英雄牌", obtain="效果锦囊", ali_ok=True,
         note="闪现不是触发型关键词，但孙悟空按『拥有闪现的英雄数』给等级"),
    dict(name="灵泉法杖", kind="keyword_grant", keyword="闪现", target="穿戴者", times="常驻", obtain="装备（商店/锦囊/拍卖）/ 天赋 灵泉法杖", ali_ok=True,
         note="让任意主C 变成闪现英雄 → 吃孙悟空整备 +2；与孙悟空搭配是跨派系的唯一桥"),
    dict(name="合成同名（登场英雄）", kind="retrigger", keyword="登场", target="被合成的那名英雄",
         times="每合成 1 张 1 次", obtain="商店买同名", ali_ok=True,
         note="data/keywords.json：合成 = 使用场上已存在的英雄牌；登场 = 使用此卡牌后触发 → 合成就是一次登场。孙悟空合成会再次『夺取』"),
    # ---- 牺牲的触发器 = 让更多单位"带着牺牲"去死 / 多死几次
    dict(name="钟馗", kind="keyword_grant", keyword="牺牲", target="自身及左右 1 格", times="常驻（每场战斗该效果限 1 次）",
         obtain="商店 3 阶", ali_ok=True, note="被赋予牺牲的人阵亡 = 李信 +7 临时、守约 -20 法力"),
    dict(name="自我奉献", kind="keyword_grant", keyword="牺牲", target="所有【河洛】英雄", times="常驻", obtain="天赋（天赋包一，历史选过 10 次）", ali_ok=True,
         note="自带：所有友方 +1 临时，每触发 1 次再 +1"),
    dict(name="先知图腾（鬼谷子）", kind="keyword_grant", keyword="牺牲", target="距离最近的 2 名【大河】英雄（玄微之种）",
         times="每场开团时", obtain="鬼谷子 2 阶（召唤）", ali_ok=True, note="先知图腾本身也是一次『开团』（张良吃）；玄微之种 = 穿戴者获得牺牲"),
    dict(name="四海来援", kind="keyword_grant", keyword="牺牲", target="无阵营英雄", times="常驻", obtain="天赋（天赋包?）", ali_ok=True),
    dict(name="熊魂附甲", kind="keyword_grant", keyword="牺牲", target="穿 ≥2 件装备的【大河】英雄", times="常驻", obtain="天赋", ali_ok=True),
    dict(name="怒涛", kind="keyword_grant", keyword="牺牲", target="【三分】英雄", times="常驻", obtain="天赋", ali_ok=True),
    dict(name="铠", kind="revive", keyword="牺牲", target="自己（复生 = 一场死 2 次）", times="每场", obtain="商店 3 阶", ali_ok=True,
         note="牺牲：河洛 +5 临时；复生让这条结算 2 次"),
    dict(name="太乙真人", kind="revive", keyword="牺牲", target="距离最近的 1 名英雄获得复生", times="每场开团", obtain="商店 4 阶", ali_ok=True,
         note="贴在牺牲件旁边 = 多一次牺牲；本身也是开团单位（张良/韩信）"),
    dict(name="献祭之赐", kind="retrigger", keyword="牺牲", target="李信", times="常驻",
         obtain="天赋（天赋包一，送 1 张李信）", ali_ok=True, note="李信临时等级 +3/次；李信觉醒后敌方的牺牲也算"),
    # ---- 其他棋手专属 / 初始牌（阿离拿不到，列出避免误判）
    dict(name="魅力无限", kind="retrigger", keyword="效果牌", target="每回合第 1 张效果牌触发 2 次", times="—",
         obtain="其他棋手专属天赋（talent_group=专属）", ali_ok=False),
    dict(name="时之回环 / 永恒时之回环", kind="retrigger", keyword="英雄牌/效果牌", target="下回合重放 8/12 张", times="—",
         obtain="其他棋手初始牌", ali_ok=False),
    dict(name="时光倒流 / 霓裳曲 / 轮指", kind="retrigger", keyword="—", target="—", times="—", obtain="其他棋手技能牌", ali_ok=False),
]
for t in TRIGGERS:
    base = t["name"].split("（")[0].split("/")[0].strip()
    c = CARDS.get(base)
    if c:
        t.update({k: v for k, v in card(base).items() if k not in ("name",)})
    if t["name"] in HISTORY:
        t["history"] = HISTORY[t["name"]]

# ================================================================ 2. 被触发后给等级的效果
# carry: 单次触发给主C 的等级（数字或公式字符串）；team: 给全队；perm: 永久/临时/核心
def P(name, kw, to, carry, carry_awk, team, perm, deps="", note=""):
    c = CARDS[name]
    return {"hero": name, "faction": c.get("faction"), "quality": c.get("quality"), "keyword": kw,
            "quote": (c.get("skill") or "").replace("\n", " "), "quote_awaken": (c.get("awaken") or "").replace("\n", " "),
            "gives_to": to, "carry_per_trigger": carry, "carry_per_trigger_awaken": carry_awk,
            "team_per_trigger": team, "level_type": perm, "depends_on": deps, "note": note}

n, K, H, T, D = S["n"], S["K"], S["H"], S["T"], S["D"]
PAYLOADS = [
    # 整备
    P("孙悟空", "整备", "所有闪现英雄各 +2", "+2（主C 是闪现英雄时）", "+4", "2×F（F=场上闪现英雄数，含他自己）", "永久",
      "主C 是否闪现（玄策/韩信/李白/云中君/露娜/阿轲/镜/裴擒虎，或穿灵泉法杖）",
      "上场/合成时『夺取』相邻 1 格随机友军最多 10 级 —— 不能挨着主C 放"),
    P("亚连", "整备", "穿戴装备最多的 1 名英雄 +2；核心 +1", 2, "2×2（触发2次）", "核心 +1", "永久 + 核心",
      "主C 必须是装备最多的人（并列时谁拿到待实测）", "跨派系：任何主C 只要装备最多就吃"),
    P("安琪拉", "整备", "自身 +2；核心 +2", "+2（安琪拉是主C 时）", "+4", "核心 +2", "永久 + 核心"),
    P("李元芳", "整备", "距离最近的【河洛】英雄 +1 张古币（+2）", "+2（主C 是最近的河洛）", "+4（精致古币）", "—", "永久",
      "站位：主C 必须是离他最近的河洛（程咬金/盾山 更近就被抢）", "古币算效果牌：沈梦溪 -15 法力、狄仁杰 +5% 攻速、武则天 +1"),
    P("程咬金", "整备", "自身 +2（古币）；手里 +1 张古币", "+2（手里那张打给河洛主C）", "+4（精致）", "程咬金自身 +2", "永久"),
    P("上官婉儿", "整备", "随机 3 名不同【河洛】英雄各 1 张古币", f"+2×min(1,3/H) = {2*min(1,3/H):.1f}（H={H}）", "+2（6 名）", "+6", "永久", "场上河洛人数 H"),
    P("米莱狄", "整备", "召唤单位临时 +3；核心 +3", "0（直接）/ 核心 +3 → 日落海主C 开战分到临时", "核心 +3×2", "核心 +3", "核心/临时"),
    P("狂铁", "整备", "核心 +2", "0（直接）/ 核心 +2", "核心 +2×2", "核心 +2", "核心", "", "对韩信：每次 = 1 次整备触发 → 韩信 +2"),
    P("百里玄策", "整备", "闪现英雄上回合每击败 1 名敌方英雄 → 随机 1 名闪现英雄 +1", f"K/F（K={K} 击杀，F=闪现数）", "2K/F", "K", "永久",
      "击杀数 K、闪现英雄数 F（F 越多玄策自己分得越少，但孙悟空越赚）"),
    P("曹操", "整备", "随机触发 1 名【三分】英雄的登场", "= 被抽到的那名三分英雄的登场收益", "×2", "—", "—", "场上三分登场英雄"),
    P("裴擒虎", "整备", "随机 1 名其他英雄等级一半的临时等级（不可叠加）", 0, 0, "—", "临时（不叠加 → 再触发无收益）",
      "", "再触发它没有意义；但对韩信/朵莉亚仍是 1 次整备"),
    P("西施", "整备", "+1 张推演战术（法力）", 0, 0, "—", "—", "", "不给等级；对韩信/朵莉亚仍是 1 次整备"),
    # 登场
    P("蒙犽", "登场", "+1 张特训战术（指定 1 名英雄 +4）", 4, "8（高阶特训）", "—", "永久", "", "跨派系、指定主C：唤醒/合成蒙犽 = 主C +4，最通用的登场收益"),
    P("武则天", "登场", "+1 张精致古币（河洛 +4，武则天在场 +1）", "+5（河洛主C）", "+10~12（2 张）", "—", "永久", "主C 河洛", "觉醒后古币每张 +2"),
    P("明世隐", "登场", "最近英雄的『临时→永久』可转化额度 +2", "+2/回合（之后每回合，前提是主C 有 ≥ 额度的临时等级）", "+4/回合", "—", "永久（复利）",
      "主C 每回合有临时等级（铠阵亡 +5、李信、李白古币、自我奉献…）且是离明世隐最近的人",
      "按卡面字义额度永久增加 → 每次唤醒/合成明世隐 = 之后每回合多 2 级；是否永久累加待实测"),
    P("马可波罗", "登场", "自身 +2；核心 +2", "+2（马可是主C）", "+4", "核心 +2", "永久 + 核心"),
    P("周瑜", "登场", "全体 +1", 1, 2, "+n", "永久"),
    P("赵云", "登场", "自身 +1，随机 2 名其他 +1", f"{2/(n-1):.2f}", f"{4/(n-1):.2f}", "+3", "永久"),
    P("貂蝉", "登场", "【三分】英雄 +1", "+1（主C 三分）", "+2", "+三分人数", "永久"),
    P("阿轲", "登场", "自身及随机 1 名英雄 +1，棋手每掉 10 血重复 1 次", f"{(1+S['hp_lost']//10)/(n-1):.2f}（掉 {S['hp_lost']} 血 → {1+S['hp_lost']//10} 次）",
      "×2", f"+{2*(1+S['hp_lost']//10)}", "永久", "棋手已掉血"),
    P("庄周", "登场", "自身 +2；+1 张招募战术", 0, 0, "+2", "永久"),
    P("露娜", "登场", "核心 +2；随机触发 2 名英雄的整备", "= 2 次随机整备的收益", "核心 +4、4 次", "核心 +2", "核心", "场上整备英雄"),
    P("甄姬", "登场", "+1 张别的登场英雄牌", 0, 0, "—", "—"),
    P("孙尚香", "登场", "+1 次免费刷新", 0, 0, "—", "经济", "", "历史上唤醒给过她：只换来 1 次免费刷新"),
    # 开团（每场 1 次，不存在『再触发开团』的牌）
    P("云中君", "开团", "每名英雄按自身装备件数 +1/件", f"+{S['carry_equips']}", f"+{2*S['carry_equips']}", "+全队装备件数", "永久",
      "主C 装备件数", "最稳的跨派系每回合永久等级；本身也是开团单位（张良/韩信）"),
    P("虞姬", "开团", "每个图腾：自身及随机 2 名不同【大河】+1", f"+T={T}（虞姬主C）", f"+T", f"+3T", "临时?（卡面未写永久）", "图腾数 T",
      "卡面没写『永久』，按关键词默认理解为本场有效 —— 待实测"),
    P("刘邦", "开团", "摧毁所有图腾，每个 +5（永久）", "0（刘邦不当主C）", 0, "+5T（自己）", "永久", "", "和虞姬/张良图腾体系冲突"),
    P("东皇太一", "开团", "获得每个图腾等级 15% 的临时等级", 0, 0, "—", "临时"),
    # 牺牲
    P("苏烈", "牺牲", "自身 +2（永久）", 0, 0, "+2", "永久"),
    P("铠", "牺牲", "【河洛】英雄 +5 临时", "+5 临时（河洛主C）", "+10 临时", "+5×河洛数（临时）", "临时 → 明世隐可转永久"),
    P("盾山", "牺牲", "+1 张古币", "+2（打给河洛主C）", "+4（精致）", "—", "永久"),
    P("项羽", "牺牲", "所有图腾 +3（永久）", 0, 0, "图腾 +3", "永久"),
    P("花木兰", "牺牲", "自身临时 +10", 0, 0, "+10 临时", "临时"),
]

# 被动放大器（每次触发 → 等级）
REACTORS = [
    dict(hero="韩信", listens="整备 + 开团（你的英雄）", per_trigger="韩信 +2（觉醒 +4），永久", carry_if="韩信是主C"),
    dict(hero="朵莉亚", listens="整备", per_trigger=f"随机 3 人 +1（主C 期望 {3/n:.2f}）+ 核心 +1；觉醒：全员 +1", carry_if="任意主C"),
    dict(hero="张良", listens="开团（你的单位，含图腾/召唤物）", per_trigger=f"随机 3 人永久 +1（主C 期望 {3/n:.2f}）；觉醒：全员 +1", carry_if="任意主C"),
    dict(hero="小乔", listens="登场", per_trigger=f"随机 3 人 +1（主C 期望 {3/n:.2f}）；觉醒：全员 +1", carry_if="任意主C"),
    dict(hero="乐善好施（天赋）", listens="登场", per_trigger="随机 2 名【三分】+2", carry_if="三分主C"),
    dict(hero="超级蓄能（天赋）", listens="回合开始（不是触发，按人数）", per_trigger="每名整备英雄 +2×N（N=场上整备英雄数）", carry_if="主C 自己有整备（安琪拉/玄策/孙悟空/亚连）"),
    dict(hero="李信", listens="拥有牺牲效果的英雄阵亡", per_trigger="李信 +7 临时（觉醒 +14；献祭之赐再 +3）", carry_if="李信是主C"),
    dict(hero="百里守约", listens="拥有牺牲效果的英雄阵亡", per_trigger="本回合法力上限 -20（不给等级）", carry_if="守约是主C"),
    dict(hero="大司命", listens="牺牲", per_trigger="往生图腾 +2（永久）", carry_if="—"),
    dict(hero="古币契约（天赋）", listens="牺牲 ×3", per_trigger="+1 张古币", carry_if="河洛主C"),
    dict(hero="海诺", listens="核心等级提升", per_trigger="核心额外 +1（觉醒 +2）", carry_if="日落海主C（核心分配）"),
    dict(hero="积极整备（天赋）", listens="整备 ×5", per_trigger="+1 件日落海专属装备（上限 3）", carry_if="—"),
    dict(hero="慕名而来（天赋）", listens="登场 ×4", per_trigger="+1 张英雄牌", carry_if="—"),
    dict(hero="大乔", listens="登场 ×2", per_trigger="大乔法力上限 -5（不给等级）", carry_if="—"),
    dict(hero="镜 / 芈月 / 白起", listens="使用效果牌（临阵磨枪/唤醒/全军动员 本身就是效果牌）", per_trigger="镜 每 2 张 +1；芈月 装备最多的逐鹿 +1；白起 每回合首张 +援护战术", carry_if="—"),
]
for r in REACTORS:
    base = r["hero"].split("（")[0].split(" ")[0]
    if base in CARDS:
        r["quote"] = (CARDS[base].get("skill") or "").replace("\n", " ")

# ================================================================ 3. 配对算账
# per: per_round（每回合自然发生）/ per_card（每张触发牌一次性）
# ease: 1~5，5 = 2~3 阶、商店常见、件数少；1 = 5 阶效果牌/要凑 4 件以上
F2 = 2  # 玄策 + 孙悟空
PAIRS = [
    # ---------------- 整备 × 闪现（玄策/韩信 漏掉的那条）
    dict(id="sunwukong_flash", per="per_round", trigger="孙悟空自身整备", payload=["孙悟空"], carry=["百里玄策", "韩信", "李白", "云中君", "孙悟空"],
         carry_gain=2, carry_gain_awaken=4, team_gain=f"2×F（F={F2} 时 +4）", ease=3, compounding=False,
         formula="闪现主C +2/回合；韩信主C 再 +2（孙悟空的整备也是 1 次整备触发）→ 韩信 +4/回合",
         useful_rounds=[7, 15], note="4 阶，4 级起 4%、5~6 级 6~8%；夺取：别挨着主C 放"),
    dict(id="linzhen_x_sunwukong", per="per_card", trigger="临阵磨枪", payload=["孙悟空"], carry=["百里玄策", "韩信", "李白", "孙悟空"],
         carry_gain=2, carry_gain_awaken=4, team_gain="2×F", ease=3, compounding=False,
         formula="每张：闪现主C +2（+韩信 +2）；历史上临阵磨枪 228 次只给过孙悟空 2 次",
         useful_rounds=[7, 15]),
    dict(id="linzhen_x_yalian", per="per_card", trigger="临阵磨枪", payload=["亚连"], carry=["任意（装备最多的人）"],
         carry_gain=2, carry_gain_awaken=4, team_gain="核心 +1", ease=4, compounding=False,
         formula="每张：装备最多的主C +2（觉醒亚连 +4）；韩信主C 再 +2", useful_rounds=[2, 15],
         note="跨派系通用；亚连 2 阶好凑"),
    dict(id="linzhen_x_angela", per="per_card", trigger="临阵磨枪", payload=["安琪拉"], carry=["安琪拉"],
         carry_gain=2 + 3 / n, carry_gain_awaken=4 + 3 / n, team_gain="核心 +2", ease=4, compounding=False,
         formula="安琪拉主C：自身 +2（+朵莉亚 3/n）", useful_rounds=[4, 15]),
    dict(id="linzhen_x_coin", per="per_card", trigger="临阵磨枪", payload=["程咬金", "李元芳", "上官婉儿"], carry=["百里守约", "沈梦溪", "狄仁杰", "百里玄策"],
         carry_gain=2, carry_gain_awaken=4, team_gain="程咬金 +2 / 婉儿 +6", ease=4, compounding=False,
         formula="= 1 张古币（+2）；给上官婉儿 = 3 张古币（主C 期望 +2，全队 +6）；沈梦溪主C 同时 -15 法力",
         useful_rounds=[2, 15], note="和直接拿 1 张古币等价 —— 有更好的整备目标时（孙悟空/亚连/上官婉儿）别给李元芳"),
    dict(id="quanjun_x_board", per="per_card", trigger="全军动员", payload=HEROES_BY_KW["整备"], carry=["任意"],
         carry_gain="Σ 场上整备英雄单次收益（+韩信 2N、朵莉亚 3N/n）", carry_gain_awaken=None, team_gain="—", ease=1, compounding=False,
         formula="玄策线（玄策+元芳+程咬金+孙悟空）：+2+2+2+K/F≈+7；韩信日落海线（狂铁+亚连+米莱狄+孙悟空）：韩信 2×4+亚连 2+孙悟空 2=+12",
         useful_rounds=[9, 15], note="5 阶，5 级起才 0.5%；主要来源是 N选1（历史 42 次）"),
    dict(id="superstore_x_prep", per="per_round", trigger="超级蓄能（天赋）", payload=HEROES_BY_KW["整备"], carry=["安琪拉", "百里玄策", "孙悟空", "亚连"],
         carry_gain="2N（N=4 → +8）", carry_gain_awaken=None, team_gain="每名整备英雄 +2N → 共 2N²（N=4 → +32）", ease=3, compounding="平方：多 1 个整备英雄，每人多 +2",
         formula="主C 本身有整备时 +2N/回合；李元芳/狂铁 这类 1 阶整备件在这时不该卖（每多 1 个 = 全体整备英雄各 +2）",
         useful_rounds=[3, 15], note="初始天赋，历史选过 12 次；当前规则没有把它和『整备主C』关联"),
    dict(id="hanxin_x_prep_open", per="per_round", trigger="韩信（被动）", payload=["狂铁", "亚连", "米莱狄", "雅典娜", "云中君", "孙悟空", "李元芳"],
         carry=["韩信"], carry_gain="2×(整备次数+开团英雄数)", carry_gain_awaken="4×(…)", team_gain="—", ease=3, compounding=False,
         formula="日落海线：狂铁1+亚连1+雅典娜(开团1+整备2)+云中君1 = 6 → +12；加孙悟空 +4（整备 1 次 + 闪现 +2）；"
                 "再加 亚连 +2（韩信装备最多）+ 云中君 +3 = +21/回合",
         useful_rounds=[7, 15]),
    dict(id="athena_x_hanxin", per="per_round", trigger="雅典娜（开团→整备）", payload=["狂铁", "亚连", "米莱狄", "安琪拉"], carry=["韩信", "安琪拉"],
         carry_gain="韩信 +6 / 安琪拉 2×(2/日落海整备数)", carry_gain_awaken="韩信 +10", team_gain="2 次整备的收益", ease=3, compounding=False,
         formula="韩信：1 次开团 + 2 次整备 = 3 次触发 × 2", useful_rounds=[7, 15]),
    dict(id="dolia_x_prep", per="per_round", trigger="朵莉亚（被动）", payload=HEROES_BY_KW["整备"], carry=["任意"],
         carry_gain=f"3N/n（N=4 整备 → {12/n:.1f}）", carry_gain_awaken="N（全员 +1/次 → +4）", team_gain="3N + 核心 +N", ease=4, compounding=False,
         formula="每次整备随机 3 人 +1；觉醒后全员 +1/次；全军动员/临阵磨枪/雅典娜/露娜 每多 1 次整备再 +1", useful_rounds=[4, 15]),
    dict(id="xuance_own", per="per_round", trigger="百里玄策自身整备", payload=["百里玄策"], carry=["百里玄策"],
         carry_gain=f"K/F（K={K}, F=1 → +{K}；F=2 → +{K/2:.0f}）", carry_gain_awaken="2K/F", team_gain="K", ease=5, compounding=False,
         formula="靠击杀，前期输了就是 0；临阵磨枪给玄策只值 K/F", useful_rounds=[2, 15]),
    # ---------------- 登场
    dict(id="huanxing_x_mengya", per="per_card", trigger="唤醒", payload=["蒙犽"], carry=["任意"],
         carry_gain=4, carry_gain_awaken=8, team_gain="—", ease=3, compounding=False,
         formula="每张唤醒 = 1 张特训战术 = 指定主C +4（觉醒蒙犽 高阶 +8）；合成蒙犽同理", useful_rounds=[7, 15],
         note="跨派系最通用的登场收益；历史上唤醒一次没给过蒙犽"),
    dict(id="huanxing_x_wuzetian", per="per_card", trigger="唤醒", payload=["武则天"], carry=["百里守约", "沈梦溪", "狄仁杰", "百里玄策", "李白"],
         carry_gain=5, carry_gain_awaken=12, team_gain="—", ease=2, compounding=False,
         formula="精致古币 +4 + 武则天 +1；觉醒：2 张、每张 +2 → 2×(4+2)=12", useful_rounds=[9, 15]),
    dict(id="huanxing_x_mingshiyin", per="per_card", trigger="唤醒 / 合成明世隐", payload=["明世隐"], carry=["百里守约", "沈梦溪", "李信", "狄仁杰", "百里玄策"],
         carry_gain="+2/回合 × 剩余回合（第 6 回合用 ≈ +16）", carry_gain_awaken="+4/回合", team_gain="—", ease=3,
         compounding="是：一次触发 = 之后每回合多转化 2 级（前提：主C 每回合有临时等级可转）",
         formula="可转化额度 +2 永久 → 剩余 R 回合共 +2R；需要临时等级来源（铠阵亡 +5/次、李信、李白古币、自我奉献）",
         useful_rounds=[4, 12], note="『额度永久累加』是按卡面字义推断，未实测 —— 是本报告最需要实测的一条"),
    dict(id="huanxing_x_marco", per="per_card", trigger="唤醒 / 合成马可波罗 / 锦囊", payload=["马可波罗"], carry=["马可波罗"],
         carry_gain=2, carry_gain_awaken=4, team_gain="核心 +2（有海诺 +3）", ease=4, compounding=False,
         formula="马可主C：每次登场自身 +2、核心 +2", useful_rounds=[2, 15], note="马可路线 effect_cards 目前为空，唤醒/锦囊都没列"),
    dict(id="huanxing_x_luna", per="per_card", trigger="唤醒 / 集体唤醒", payload=["露娜"], carry=["韩信", "安琪拉", "马可波罗"],
         carry_gain="2 次随机整备（韩信 +4；安琪拉 期望 4/k）", carry_gain_awaken="4 次", team_gain="核心 +2", ease=1, compounding=False,
         formula="登场→整备链；露娜 5 阶", useful_rounds=[10, 15]),
    dict(id="caocao_x_xiaoqiao", per="per_round", trigger="曹操（整备→登场）", payload=["周瑜", "赵云", "貂蝉", "甄姬", "孙尚香"], carry=["三分主C"],
         carry_gain=f"周瑜 +1 / 小乔 {3/n:.2f}", carry_gain_awaken="×2", team_gain="周瑜 +n、小乔 +3", ease=3, compounding=False,
         formula="阿离没有三分路线，这条只作参考（engines.json 已有 sanfen_deploy）", useful_rounds=[4, 15]),
    dict(id="merge_x_entry", per="per_card", trigger="合成同名", payload=["蒙犽", "马可波罗", "明世隐", "武则天", "周瑜"], carry=["任意"],
         carry_gain="同对应的唤醒", carry_gain_awaken=None, team_gain="—", ease=4, compounding=False,
         formula="合成 = 登场：多买登场英雄的同名就是免费的『唤醒』（每张 3 能量）", useful_rounds=[2, 15]),
    dict(id="jinnang_x_entry", per="per_card", trigger="锦囊（天赋）", payload=["马可波罗", "蒙犽", "明世隐", "周瑜"], carry=["任意"],
         carry_gain="本回合每张登场英雄牌的收益 ×2", carry_gain_awaken=None, team_gain="—", ease=3, compounding=False,
         formula="锦囊当回合集中合成登场英雄（例如马可 2 张 = +4 → +8）", useful_rounds=[4, 15]),
    # ---------------- 开团
    dict(id="zhangliang_x_open", per="per_round", trigger="张良（被动）", payload=HEROES_BY_KW["开团"] + ["先知图腾"], carry=["任意"],
         carry_gain=f"3U/n（U=开团单位数；U=5 → {15/n:.1f}）", carry_gain_awaken="U（全员 +1/次 → +5）", team_gain="3U", ease=4, compounding=False,
         formula="开团没有再触发牌，只能靠『更多开团单位』：鬼谷子(先知图腾) 2 阶最便宜、云中君/瑶/太乙/雅典娜/虞姬", useful_rounds=[4, 15]),
    dict(id="yunzhongjun_equips", per="per_round", trigger="云中君自身开团", payload=["云中君"], carry=["任意（3 件装备）"],
         carry_gain=S["carry_equips"], carry_gain_awaken=2 * S["carry_equips"], team_gain="全队装备件数", ease=4, compounding=False,
         formula="主C 3 件 → +3/回合永久；同时算 1 个开团单位（张良 3/n、韩信 +2）", useful_rounds=[4, 15],
         note="虞姬/韩信大河线里排在 late_fill，按收益应前移"),
    dict(id="yuji_totem", per="per_round", trigger="虞姬自身开团 × 图腾数", payload=["虞姬"], carry=["虞姬"],
         carry_gain=f"T={T}", carry_gain_awaken="T（觉醒多 2 名大河）", team_gain="2T", ease=4, compounding=False,
         formula="图腾来源：少司缘(因缘)、鬼谷子(先知)、大司命(往生)、天赋 图腾启灵/因缘际会", useful_rounds=[2, 15]),
    # ---------------- 牺牲
    dict(id="lixin_x_sacrifice", per="per_round", trigger="钟馗 / 自我奉献 / 铠复生 / 太乙复生 / 先知图腾", payload=["铠", "苏烈", "盾山", "钟馗"],
         carry=["李信"], carry_gain=f"李信 +7×D 临时（D={D} → +{7*D}，战斗内有效）", carry_gain_awaken=f"+{14*D}", team_gain="—", ease=4, compounding=False,
         formula="永久部分只有 明世隐 每回合转化 2（唤醒/合成明世隐 +2）—— 李信线缺的是『临时→永久』", useful_rounds=[4, 15]),
    dict(id="mingshiyin_x_kai", per="per_round", trigger="铠阵亡（+5 临时）→ 明世隐转化", payload=["铠", "明世隐"], carry=["百里守约", "沈梦溪", "李信"],
         carry_gain="+2/回合（每次唤醒/合成明世隐再 +2/回合）", carry_gain_awaken="+4/回合", team_gain="—", ease=4, compounding="见 huanxing_x_mingshiyin",
         formula="守约/沈梦溪线已有；李信线 core_set 缺明世隐", useful_rounds=[4, 15]),
    # ---------------- 古币底座（作对照）
    dict(id="coin_base", per="per_round", trigger="李元芳 + 程咬金（+上官婉儿）自然整备", payload=["李元芳", "程咬金", "上官婉儿"],
         carry=["百里守约", "沈梦溪", "狄仁杰", "百里玄策"], carry_gain=f"+4（元芳 2 + 程咬金手里 1 张 2）；+婉儿 {2*min(1,3/H):.0f} → +6",
         carry_gain_awaken="+8~12", team_gain="程咬金 +2", ease=5, compounding=False,
         formula="李元芳只值 +2/回合：前期有用，第 8 回合后（程咬金/婉儿到位）可卖 —— 除非拿了超级蓄能（每个整备英雄 = 全体 +2）或主C 是韩信",
         useful_rounds=[1, 8]),
]


# 主C 每回合"永久等级"折算（人工按 formula 定，便于排序；临时等级另列 est_temp）
EST = {"sunwukong_flash": 2, "linzhen_x_sunwukong": 2, "linzhen_x_yalian": 2, "linzhen_x_angela": 2 + 3 / n,
       "linzhen_x_coin": 2, "quanjun_x_board": 9, "superstore_x_prep": 8, "hanxin_x_prep_open": 12,
       "athena_x_hanxin": 6, "dolia_x_prep": 12 / n, "xuance_own": 2, "huanxing_x_mengya": 4,
       "huanxing_x_wuzetian": 5, "huanxing_x_mingshiyin": 2, "huanxing_x_marco": 2, "huanxing_x_luna": 4,
       "caocao_x_xiaoqiao": 1, "merge_x_entry": 3, "jinnang_x_entry": 4, "zhangliang_x_open": 15 / n,
       "yunzhongjun_equips": 3, "yuji_totem": 2, "lixin_x_sacrifice": 0, "mingshiyin_x_kai": 2, "coin_base": 4}
EST_TEMP = {"lixin_x_sacrifice": 7 * D, "yuji_totem": T}
# 频率系数：per_round=1；per_card=0.5（平均每 2 回合拿到 1 张）；per_card_rate=每张让之后每回合 +est，按"整局拿 1.5 张"折 1.5
FREQ = {"per_round": 1.0, "per_card": 0.5, "per_card_rate": 1.5}
for p in PAIRS:
    if p["id"] == "huanxing_x_mingshiyin":
        p["per"] = "per_card_rate"
    p["carry_gain_est"] = round(EST[p["id"]], 2)
    if p["id"] in EST_TEMP:
        p["carry_gain_temp_in_battle"] = EST_TEMP[p["id"]]
    p["score"] = round(p["carry_gain_est"] * FREQ[p["per"]] * p["ease"] / 5, 2)
    p["score_doc"] = "score = carry_gain_est(永久级/回合或/张) × 频率系数(per_round 1 / per_card 0.5 / per_card_rate 1.5) × ease/5"
PAIRS.sort(key=lambda p: -p["score"])

# ================================================================ 4. 各路线
ROUTES = {
    "122215359354172436": dict(name="阿离Ace 百里玄策", carry="百里玄策",
        producers=["李元芳（前期）", "程咬金", "孙悟空（缺）", "上官婉儿（后期，缺）"],
        amplifiers=["孙悟空：闪现英雄 +2/回合（缺）", "武则天：古币 +1（缺）", "超级蓄能（天赋，缺）"],
        triggers=["临阵磨枪 → 孙悟空 > 上官婉儿 > 程咬金 > 玄策（K/F）> 李元芳", "全军动员", "唤醒 → 武则天"],
        missing=["孙悟空", "临阵磨枪（目标优先级）", "超级蓄能", "上官婉儿", "武则天"],
        per_round_now="≈ +6（元芳 +2、程咬金 +2、玄策自身 K=2/F=1 → +2）",
        per_round_full="≈ +15（+孙悟空 2、玄策自身降为 K/F=1、+婉儿 2；拿到超级蓄能 N=4 再 +8）",
        engine_parts=[
            dict(unit="百里玄策", role="落点", useful_rounds=[2, 15]),
            dict(unit="李元芳", role="古币产出 +2/回合", useful_rounds=[1, 8], late_keep_if="超级蓄能 / 没有程咬金"),
            dict(unit="程咬金", role="古币产出 +2/回合 + 自身 +2", useful_rounds=[4, 15]),
            dict(unit="孙悟空", role="闪现放大：玄策 +2/回合（觉醒 +4），临阵磨枪首选目标", useful_rounds=[7, 15],
                 caveat="夺取：放在不挨着玄策的位置"),
            dict(unit="上官婉儿", role="3 张古币/回合（河洛 ≤3 人时玄策必得 1 张）", useful_rounds=[8, 15]),
            dict(unit="武则天", role="古币 +1，唤醒 = +5", useful_rounds=[9, 15]),
        ]),
    "50720714303640586": dict(name="阿离颗秒万暴 百里守约", carry="百里守约",
        producers=["李元芳（前期）", "程咬金", "盾山（阵亡 1 古币）", "铠（阵亡 +5 临时）"],
        amplifiers=["明世隐（临时→永久，可被唤醒放大）", "武则天", "上官婉儿", "钟馗/自我奉献（牺牲件 → 守约法力）"],
        triggers=["唤醒 → 明世隐（≈+2/回合 永久累加）> 武则天（+5）", "临阵磨枪 → 上官婉儿 > 程咬金 > 李元芳", "合成明世隐", "全军动员"],
        missing=["唤醒（route 里没列）", "临阵磨枪", "全军动员"],
        per_round_now="≈ +6~8（元芳 2、程咬金 2、盾山阵亡 ~2、明世隐转化 2）",
        per_round_full="≈ +12（+婉儿 2、明世隐被唤醒 1 次 +2、武则天每张古币 +1）",
        engine_parts=[
            dict(unit="百里守约", role="落点", useful_rounds=[2, 15]),
            dict(unit="铠", role="临时等级 + 牺牲次数（复生 → 2 次）", useful_rounds=[4, 15]),
            dict(unit="明世隐", role="临时→永久；唤醒/合成 = 额度 +2", useful_rounds=[4, 15]),
            dict(unit="李元芳", role="古币 +2/回合", useful_rounds=[1, 8]),
            dict(unit="盾山", role="牺牲件 + 古币", useful_rounds=[2, 10]),
            dict(unit="程咬金", role="古币", useful_rounds=[5, 15]),
            dict(unit="钟馗", role="左右 2 人带牺牲 → 守约法力", useful_rounds=[8, 15]),
        ]),
    "81682962224371742": dict(name="无限技能炸弹猫 沈梦溪", carry="沈梦溪",
        producers=["李元芳（自动古币 = 对沈梦溪用效果牌）", "程咬金", "铠"],
        amplifiers=["明世隐", "武则天"],
        triggers=["临阵磨枪 → 李元芳（古币自动落在沈梦溪：+2 级且 -15 法力）> 程咬金", "唤醒 → 明世隐 / 武则天"],
        missing=["临阵磨枪", "唤醒"],
        per_round_now="≈ +6（元芳 2、程咬金 2、明世隐 2）", per_round_full="≈ +10",
        engine_parts=[
            dict(unit="沈梦溪", role="落点", useful_rounds=[2, 15]),
            dict(unit="铠", role="临时等级", useful_rounds=[4, 15]),
            dict(unit="明世隐", role="临时→永久", useful_rounds=[4, 15]),
            dict(unit="李元芳", role="每回合 1 张古币自动打在沈梦溪身上（法力 + 等级两用）", useful_rounds=[1, 15],
                 note="沈梦溪线李元芳后期也有用：每回合一张『对他使用的效果牌』"),
            dict(unit="程咬金", role="古币", useful_rounds=[4, 15]),
        ]),
    "32706315349545018": dict(name="阿离狄仁杰速秒流", carry="狄仁杰",
        producers=["李元芳", "程咬金", "盾山", "上官婉儿"], amplifiers=["武则天"],
        triggers=["临阵磨枪 → 上官婉儿 > 程咬金 > 李元芳", "唤醒 → 武则天", "全军动员"],
        missing=["临阵磨枪", "唤醒→武则天"],
        per_round_now="≈ +4~6", per_round_full="≈ +10（+婉儿 +武则天）",
        engine_parts=[dict(unit="狄仁杰", role="落点", useful_rounds=[4, 15]),
                      dict(unit="李元芳", role="古币", useful_rounds=[1, 10]),
                      dict(unit="程咬金", role="古币", useful_rounds=[4, 15]),
                      dict(unit="上官婉儿", role="3 古币/回合", useful_rounds=[8, 15]),
                      dict(unit="武则天", role="+1/古币", useful_rounds=[9, 15])]),
    "user_hanxin_sunset": dict(name="日落海整备韩信", carry="韩信",
        producers=["狂铁", "亚连", "米莱狄", "雅典娜", "云中君", "孙悟空（缺）", "李元芳（缺，1 阶便宜触发）"],
        amplifiers=["韩信本人", "朵莉亚（缺：每次整备 3/n 给全队）"],
        triggers=["全军动员（≈+12）", "临阵磨枪 → 孙悟空（韩信 +4）= 亚连（韩信 +2 触发 +2 装备最多）", "露娜（登场→2 整备）", "能量改造（不影响韩信，可选）"],
        missing=["孙悟空（闪现 +2 且算 1 次整备 = 韩信 +4/回合）", "全军动员/临阵磨枪 的目标顺序", "朵莉亚"],
        per_round_now="≈ +17（触发 6 次 +12、亚连 +2、云中君 +3）",
        per_round_full="≈ +21（+孙悟空 4），有一张临阵磨枪再 +4",
        engine_parts=[dict(unit="韩信", role="落点", useful_rounds=[7, 15]),
                      dict(unit="雅典娜", role="1 开团 + 2 整备", useful_rounds=[7, 15]),
                      dict(unit="亚连", role="整备 + 装备最多 +2", useful_rounds=[2, 15]),
                      dict(unit="狂铁", role="1 阶整备件", useful_rounds=[1, 10]),
                      dict(unit="云中君", role="开团 + 装备 +3", useful_rounds=[4, 15]),
                      dict(unit="孙悟空", role="整备触发 + 闪现 +2", useful_rounds=[7, 15], caveat="夺取：别挨着韩信")]),
    "user_hanxin_open": dict(name="大河开团韩信", carry="韩信",
        producers=["虞姬", "云中君", "瑶", "太乙真人"], amplifiers=["张良", "韩信本人"],
        triggers=["开团号令 / 补给线：开团（多开团单位）", "跨派系整备件：孙悟空（+4/回合）"],
        missing=["孙悟空", "开团号令/补给线：开团 在题面里的价值"],
        per_round_now="≈ +13（4 个开团英雄 +8、张良 ~2、云中君 +3）", per_round_full="≈ +17（+孙悟空）",
        engine_parts=[dict(unit="韩信", role="落点", useful_rounds=[8, 15]),
                      dict(unit="张良", role="开团放大", useful_rounds=[4, 15]),
                      dict(unit="云中君", role="开团 + 装备", useful_rounds=[4, 15]),
                      dict(unit="虞姬", role="前期过渡 + 开团", useful_rounds=[2, 10]),
                      dict(unit="鬼谷子", role="先知图腾（张良吃，韩信不吃）", useful_rounds=[2, 10])]),
    "user_compound_river_totem": dict(name="大河图腾虞姬", carry="虞姬",
        producers=["少司缘（因缘图腾）", "鬼谷子（先知图腾）", "大司命（往生图腾）", "云中君"], amplifiers=["张良"],
        triggers=["图腾启灵 / 因缘际会（多图腾）", "开团号令 / 补给线：开团", "图腾祝福（5 阶，+5×(1+T)）"],
        missing=["云中君前移到 early（+3/回合、开团单位）"],
        per_round_now="≈ +4（T=2 → +2，张良 ~1.3，因缘图腾 ~+1）", per_round_full="≈ +8（+云中君 3、张良觉醒）",
        engine_parts=[dict(unit="虞姬", role="落点", useful_rounds=[2, 15]),
                      dict(unit="少司缘", role="因缘图腾", useful_rounds=[3, 15]),
                      dict(unit="鬼谷子", role="先知图腾（开团）", useful_rounds=[2, 12]),
                      dict(unit="张良", role="开团放大", useful_rounds=[4, 15]),
                      dict(unit="云中君", role="开团 + 装备 +3", useful_rounds=[4, 15])]),
    "user_marco_sunset": dict(name="日落海核心马可波罗", carry="马可波罗",
        producers=["米莱狄", "狂铁", "安琪拉", "亚连（缺：唯一给马可本人永久 +2/回合的整备）"], amplifiers=["朵莉亚", "海诺"],
        triggers=["合成马可同名（登场 +2/+2）", "唤醒 → 马可", "锦囊 / 锦囊妙计", "临阵磨枪 → 亚连（马可装备最多 +2）/ 米莱狄", "露娜"],
        missing=["亚连进 core/early", "唤醒 / 锦囊 / 临阵磨枪（effect_cards 为空）"],
        per_round_now="马可本人 ≈ +1.7（朵莉亚 3×N/n）+ 核心临时", per_round_full="≈ +4（+亚连 2）+ 每张唤醒 +2",
        engine_parts=[dict(unit="马可波罗", role="落点（登场）", useful_rounds=[2, 15]),
                      dict(unit="米莱狄", role="核心 +3", useful_rounds=[2, 15]),
                      dict(unit="朵莉亚", role="整备放大", useful_rounds=[4, 15]),
                      dict(unit="亚连", role="装备最多 +2/回合", useful_rounds=[2, 15])]),
    "user_angela_sunset": dict(name="日落海整备安琪拉", carry="安琪拉",
        producers=["安琪拉本人", "米莱狄", "狂铁", "亚连"], amplifiers=["朵莉亚", "超级蓄能（缺）", "智能装甲（缺）"],
        triggers=["临阵磨枪 → 安琪拉（+2+3/n）", "全军动员", "超级蓄能 N=4 → +8/回合", "雅典娜/露娜（随机整备）"],
        missing=["超级蓄能", "临阵磨枪目标=安琪拉本人"],
        per_round_now="≈ +4（自身 2 + 朵莉亚 ~1.7）", per_round_full="≈ +12（超级蓄能 +8）",
        engine_parts=[dict(unit="安琪拉", role="落点 + 整备件", useful_rounds=[3, 15]),
                      dict(unit="朵莉亚", role="整备放大", useful_rounds=[4, 15]),
                      dict(unit="米莱狄", role="整备件", useful_rounds=[2, 15]),
                      dict(unit="狂铁", role="1 阶整备件（超级蓄能下不卖）", useful_rounds=[1, 15])]),
    "user_lixin_sacrifice": dict(name="河洛牺牲李信", carry="李信",
        producers=["铠", "苏烈", "盾山", "钟馗（给左右牺牲）"], amplifiers=["明世隐（缺：唯一的临时→永久）", "自我奉献", "献祭之赐"],
        triggers=["太乙复生（多死一次）", "先知图腾/玄微之种", "唤醒 → 明世隐"],
        missing=["明世隐（core）", "唤醒→明世隐"],
        per_round_now=f"战斗内 +{7*D} 临时；永久 ≈ 0", per_round_full="永久 +2~4/回合（明世隐），战斗内同上",
        engine_parts=[dict(unit="李信", role="落点", useful_rounds=[7, 15]),
                      dict(unit="钟馗", role="牺牲赋予", useful_rounds=[4, 15]),
                      dict(unit="铠", role="复生牺牲件", useful_rounds=[4, 15]),
                      dict(unit="明世隐", role="临时→永久", useful_rounds=[4, 15])]),
}

OUT_DATA = {
    "version": "2026-10-14",
    "_doc": "触发器 × 被触发效果 × 放大器 的等级引擎表（只读研究，未接入规则/题面）。报告 docs/trigger-engines-2026-10-14.md；"
            "生成 tools/build_trigger_engines.py。数字是按 scenario 的假设算的期望，不是实测；标了『待实测』的是按卡面字义的推断。",
    "scenario": SCENARIO,
    "keyword_heroes": HEROES_BY_KW,
    "flash_heroes": FLASH,
    "triggers": TRIGGERS,
    "payloads": PAYLOADS,
    "reactors": REACTORS,
    "pairs": PAIRS,
    "routes": ROUTES,
    "history": HISTORY,
    "open_questions": [
        "觉醒狂铁/亚连/米莱狄/安琪拉『触发2次』对 韩信/朵莉亚/超级蓄能 算 1 次还是 2 次",
        "明世隐『登场：可转化的等级+2』是否永久累加（唤醒/合成多次 → 额度 2→4→6…）",
        "亚连『穿戴装备最多』并列时选谁",
        "雅典娜场上日落海整备英雄不足 2 个时实际触发几次（会不会落到无整备的日落海英雄上白算）",
        "虞姬开团加的等级是不是永久（卡面没写）",
        "唤醒/临阵磨枪 指定的目标没有对应条目时游戏是否拒收（历史 31+54 次白用发生在 2026-09-30 之前）",
    ],
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(OUT_DATA, f, ensure_ascii=False, indent=1)
print("wrote", OUT, "triggers", len(TRIGGERS), "payloads", len(PAYLOADS), "pairs", len(PAIRS))
for p in PAIRS[:15]:
    print(f"{p['score']:6.2f}  {p['id']:24s} gain={p['carry_gain_est']} ease={p['ease']} {p['per']}")
