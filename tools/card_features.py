"""卡牌语义特征标注器：data/cards.json -> data/card_features.json

用途：
  (a) 给 Laya 微调用的**英文语义特征**（卡名对模型是噪声，特征才有意义）；
  (b) 给规则引擎判断"哪个英雄好养 / 哪些牌能叠加"。

原则：**每个标签都来自下面一条显式正则**，可以逐条审计。
没有网络、没有模型、没有随机数 —— 同一份 cards.json 永远产出同一份结果。

输出 data/card_features.json：
  {card_name: {
      "en":      一行英文描述（给 Laya 的特征串，由标签 + 关键数字拼成）,
      "category": 英文类别, "tier": 品阶(quality), "faction": 拼音阵营, "role": 定位,
      "row": front/back, "range": melee/ranged/None,
      "tags":    [英文 snake_case 标签...],
      "nums":    {标签: 从原文抠出来的关键数字},
      "grow":    (仅英雄/召唤物英雄) {"self_perm_per_trigger", "trigger", "needs", "rules", "awaken_mult", ...},
      "grow_score": 0~10（这个英雄当主C时有多容易把等级堆起来）}}

用法:
  .venv/Scripts/python.exe -X utf8 tools/card_features.py              # 生成 + 打印覆盖率
  .venv/Scripts/python.exe -X utf8 tools/card_features.py --untagged   # 列出 0 标签的卡
  .venv/Scripts/python.exe -X utf8 tools/card_features.py --synergy    # 打印"产出方 -> 受益方"叠加关系 + 用到它的阵容
  .venv/Scripts/python.exe -X utf8 tools/card_features.py --card 镜    # 看某张卡的标签和命中的原文片段
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "data" / "cards.json"
LINEUPS = ROOT / "data" / "lineups.json"
OUT = ROOT / "data" / "card_features.json"

# ---------------------------------------------------------------- 基础映射

CATEGORY_EN = {
    "英雄牌": "hero", "效果牌": "effect", "天赋牌": "talent", "装备牌": "equipment",
    "召唤物": "summon", "棋手牌": "lord", "棋手技能牌": "lord_skill",
}
FACTION_EN = {
    "河洛": "heluo", "逐鹿": "zhulu", "三分之地": "sanfen", "三分": "sanfen",
    "大河流域": "dahe", "大河": "dahe", "日落海": "riluohai",
    "无阵营": "neutral", "通用": "any", "": None, None: None,
}
# 卡面里出现的阵营写法 -> 拼音（【三分】/【大河】是简称）
FACTION_ALT = "河洛|逐鹿|三分之地|三分|大河流域|大河|日落海"
ROLE_EN = {"坦克": "tank", "战士": "fighter", "刺客": "assassin",
           "射手": "marksman", "法师": "mage", "辅助": "support"}

# 卡名后缀 "(效果牌)" "(召唤物)" 等是爬虫为避免同名覆盖加的，认"自己"时要去掉
_SUFFIX = re.compile(r"[（(](?:效果牌|装备牌|天赋牌|召唤物|棋手牌)[)）]$")

# ---------------------------------------------------------------- 标签规则
# (tag, regex)。文本是 skill + awaken（英雄的自称已替换成"自身"，见 card_text）。
# 规则只看文字，不看类别 —— 类别相关的标签（role/range/faction）单独算。

def R(p: str) -> re.Pattern:
    """所有规则都按多行模式：^ = 行首（卡面常见"闪现<换行>整备：…"）。"""
    return re.compile(p, re.M)


TAG_RULES: list[tuple[str, re.Pattern]] = [
    # ---- 触发时机：这张卡"自己"带的触发器
    ("on_deploy",      R(r"登场[：:]")),
    ("on_prepare",     R(r"整备[：:]")),
    ("on_sacrifice",   R(r"牺牲[：:]")),
    ("on_exit",        R(r"退场[：:]")),
    ("on_engage",      R(r"开团[：:]")),
    ("on_victory",     R(r"凯旋[：:]|战斗胜利时|获胜时|若上场战斗胜利|战斗胜利翻倍")),
    ("on_defeat",      R(r"败阵[：:]|战斗失败时|输掉\d场战斗|战斗失败\d次|战败")),
    ("on_battle_start", R(r"战斗开始(?:时|后|，|及)|开战(?:时|后)|战斗中")),
    ("on_round_start", R(r"回合开始(?:时|商店)|每回合(?:开始时|获得|额外|等级|首次)|每过\d个回合|每经过\d个回合|下回合开始时|每\d个回合|第[\d/]+回合")),
    ("on_lord_level_up", R(r"棋手升级时|升级时[，,]|棋手(?:到达|等级达到|升到)\d|升到\d级|到达5/6级")),
    ("blink",          R(r"^闪现|获得闪现|闪现[，,]夺取")),
    ("steal_level",    R(r"^夺取|[，,]夺取|夺取[^。\n]{0,16}(?:等级|\d+级)|偷取[^。\n]{0,16}(?:等级|\d+级)")),
    ("revive",         R(r"复生")),
    ("fleeting",       R(r"^转瞬$")),
    ("link",           R(r"共战|二人(?:在场|均|共享)|二者")),  # 共战/双人组合 = 游戏里的"联动"
    ("each_n_times",   R(r"每(?!回合限|场战斗限|名英雄仅|有)[^，。\n（]{0,14}?(?:[2-9]|1\d)(?:次|张|名)|每(?:消耗|累积造成|提供)\d+(?:能量|点|级)")),

    # ---- 受益方（payoff）：任何单位触发 X 时我获益
    ("payoff_deploy",    R(r"触发(?:\d次)?登场效果(?:时|，)|每触发\d次登场效果|每当你的英雄触发登场")),
    ("payoff_prepare",   R(r"触发整备(?:或开团)?效果时|每触发\d次整备效果|整备效果英雄会提升")),
    ("payoff_engage",    R(r"触发(?:整备或)?开团效果时")),
    ("payoff_sacrifice", R(r"拥有牺牲效果的英雄阵亡|触发牺牲效果|每触发\d次牺牲效果|触发过\d次牺牲|敌方英雄触发牺牲")),
    ("payoff_merge",     R(r"合成提升的等级|合成时|每回合首次合成|每合成\d次|每进行\d次英雄合成|合成英雄时|每合成4次")),
    ("payoff_effect_card", R(r"使用(?:\d张)?效果牌|使用1张效果牌|首次使用效果牌|打出的第1张效果牌|使用的效果牌|打出了不少于\d张英雄及效果牌|每打出\d张英雄或效果牌")),
    ("payoff_tactic",    R(r"战术牌(?:打出次数|会提升)|使用战术牌|战术牌打出")),
    ("payoff_skill_cast", R(r"释放(?:\d次)?技能(?:时|后)|每释放\d次技能|施放技能|技能释放期间")),
    ("payoff_hero_play", R(r"(?:使用|打出)[^，。\n]{0,16}英雄牌(?:时|，)|每打出\d张[^，。\n]{0,10}英雄牌|打出1-3阶英雄牌|打出3阶英雄牌|每买\d张牌|每购买\d张牌")),
    ("payoff_coin",      R(r"古币牌(?:会|提升|提供)|对[^，。\n]{0,6}使用古币|使用古币时|每获得\d+次古币|打出古币|累计使用古币|每使用\d张古币|每对其使用\d张古币|古币等级\+|古币不再")),
    ("payoff_summon",    R(r"召唤(?:单位|物)(?:获得|阵亡|等级|上场|入场|存活)|每次召唤|召唤物入场|召唤物上场")),
    ("payoff_totem",     R(r"(?:每有1个|每个|所有)图腾|图腾(?:等级|被摧毁|周围|不会)|摧毁你所有的图腾|先知图腾获得|因缘图腾和往生图腾")),
    ("payoff_awakened",  R(r"已觉醒(?:的)?(?:英雄|【)|觉醒英雄牌时|已觉醒英雄数量|所有己方英雄已觉醒")),
    ("payoff_equipment", R(r"每穿戴1件装备|穿戴装备最多|装备数量|至少穿戴\d件装备|首件普通装备|每穿戴1件【")),
    ("payoff_temp_level", R(r"获得临时等级时|临时等级中的")),
    ("payoff_shield_break", R(r"护盾被(?:击破|破)")),
    ("payoff_control",   R(r"施加控制效果时")),
    ("payoff_kill",      R(r"击败(?:\d名)?(?:敌方)?(?:英雄|敌人|单位)(?:时|，|数|的|则)|每击败|击败敌方英雄时|击败[\d/]+名|击败次数|击败过|每当敌人于擂台范围内被击败|每当敌人被击败")),
    ("payoff_sell",      R(r"出售(?:\d名)?场上(?:的)?(?:1-3阶)?英雄|出售英雄|出售1张古币|每出售")),
    ("payoff_blink",     R(r"闪现(?:效果的)?英雄|闪现英雄击败")),
    ("payoff_lord_hp_loss", R(r"棋手生命值每降低|棋手生命值≤|排名不是第1")),

    # ---- 让别的牌再触发一次（retrigger）
    ("retrigger_deploy",  R(r"(?:随机触发\d名[^，。\n]{0,10}的?|触发其|触发所有英雄的|额外触发1次|整备或)登场效果")),
    ("retrigger_prepare", R(r"(?:随机触发\d名[^，。\n]{0,10}的?|触发其|触发所有英雄的|触发场上1名英雄的1次)整备(?:或登场)?效果|整备效果会额外触发")),
    ("retrigger_cards",   R(r"再次触发这些卡牌|随机触发其中1张|会触发2次|第1张效果牌会触发2次|下个指法触发2次|额外触发1次登场效果")),
    ("grant_sacrifice",   R(r"获得牺牲[：:]")),
    ("grant_blink",       R(r"获得闪现")),

    # ---- 拿牌：拿到带某种触发的英雄
    ("fetch_deploy",     R(r"登场(?:效果)?英雄牌|具有登场效果的英雄牌")),
    ("fetch_prepare",    R(r"整备效果(?:的)?英雄牌|具有整备效果")),
    ("fetch_sacrifice",  R(r"牺牲效果(?:的)?英雄牌")),
    ("fetch_engage",     R(r"开团效果(?:的)?英雄牌")),
    ("fetch_blink",      R(r"闪现效果的英雄牌")),
    ("fetch_totem_hero", R(r"与图腾相关")),

    # ---- 效果：等级
    ("level_up_self_perm", R(r"自身(?:的)?(?:及|和)?[^。\n临]{0,18}?等级(?:永久)?\+\d+|(?:^|[，\n])等级\+\d+|对自身使用1张(?:精致)?古币|自身等级永久\+|永久获得其双倍等级|永久增加\d级|等级永久增加")),
    ("level_up_self_temp", R(r"自身[^。\n]{0,20}?\d+临时等级|(?:^|[：])临时等级\+\d+|开团：获得[^。\n]{0,16}临时等级|随机获得1名其他英雄等级数[^。\n]{0,6}临时等级|自身获得(?:一半|全部)效果|自身临时等级\+")),
    ("level_up_temp",     R(r"\d+临时等级|临时等级\+\d|的临时等级|临时等级(?:的形式|均分)|以临时等级")),
    ("level_up_per_hero", R(r"该英雄(?:等级永久|获得\d+临时等级|等级\+)")),
    ("level_up_target",   R(r"枫叶舞英雄[^。\n]{0,14}等级\+|解冻后等级永久\+|每次等级永久提升时|使1名英雄夺取|指定(?:1名|场上1名|英雄)?[^。\n]{0,18}?(?:英雄)?[^。\n]{0,6}?(?:等级\+|提升\d+级|立即提升|随机提升|等级等同|交换等级|获得[^。\n]{0,12}等级|觉醒进度)")),
    ("level_up_group",    R(r"【(?:河洛|逐鹿|三分之地|三分|大河流域|大河|日落海)】英雄(?:等级\+|获得\d+临时等级)|闪现(?:效果的)?英雄等级\+|无阵营英雄获得|整备效果英雄会提升|同阶的英雄等级\+|品阶相同的英雄等级\+|1-3阶英雄牌时其等级\+|品阶最高的1名英雄等级(?:永久)?\+|等级最低[^。\n]{0,16}提升\d+级|等级最低的\d名英雄获得|打出3阶英雄牌，额外\+")),
    ("level_up_random",   R(r"随机(?:\d+名|1名)?[^。\n]{0,20}?(?:英雄|单位)[^。\n]{0,10}?等级(?:永久)?\+|场上随机英雄等级\+|随机助力1名英雄|等级随机分配|给1名随机英雄|临时等级随机分配|随机英雄等级\+|随机\d名英雄各|随机提升|随机1名英雄获得[^。\n]{0,24}等级")),
    ("level_up_all",      R(r"(?:所有|全体|全员|场上所有|己方所有|全体己方)(?:己方|友方|无阵营)?(?:英雄|单位)[^。\n]{0,14}?(?:等级(?:永久)?\+|获得[^。\n]{0,24}等级|\+\d临时等级)|场上友方英雄获得|己方每名英雄\+|全队|你的英雄获得\s*\d+临时等级|所有友方单位\+|他们分别获得")),
    ("level_up_merge",    R(r"合成提升的等级|合成时等级额外|与其同阶的英雄等级")),
    ("temp_to_perm",      R(r"临时等级中的\s*\d+级可转化为永久|保留\d+级|至多保留[^。\n]*临时等级")),
    ("level_transfer",    R(r"转移其等级|等级随机分配给|真实等级的\d+%会保留给|交换等级|等级等同于|获得其双倍等级|将自身等级的\d+%平均分配|将自身等级以临时等级均分|继承等级|保留其当前等级|保留等级|变为其召唤者")),
    ("level_scaling",     R(r"自身等级×|等级数\d+%|等级与自身相同|与自身等级相同|与其等级相同|等级×\d+点|每有7级|每5级|有1级|等级之和|每有\d+级|每\d+级|等级\s*[x×]")),
    ("level_up_on_count", R(r"场上每有1(?:种|个|名)|每有\d+(?:名|个|棋手)|已觉醒英雄数量×|场上品阶总数|主要阵营人数|场上\w{0,4}英雄数量")),

    # ---- 效果：资源 / 卡牌
    ("gain_card",        R(r"(?:获得|抽取|发现|累积)[^，。\n]{0,6}?\d+(?:-\d+)?张|卡牌\d选\d|\d选1|从\d(?:张|个|种)[^。\n]{0,16}中选择|获得1张|获得该英雄牌|复制[^。\n]{0,10}(?:英雄|卡牌|手牌)|一分为四|各1张")),
    ("gain_hero",        R(r"(?:获得|抽取|发现|复制|累积|选择)[^。\n，]{0,20}?英雄牌|兑换券|选\d（[^）]*英雄|\d张(?:HERO_NAMES)(?!兑换)|张【[^】]+】英雄|同名牌|场上英雄牌|场上1名英雄的复制|手牌中1张非觉醒英雄牌的复制|召唤其同阵营英雄上场|大明星")),
    ("gain_equipment",   R(r"(?:获得|随机获得|分享给|卡牌\d选1（|立即穿戴|获得装备)[^。\n]{0,16}?(?:装备|玄微之种|名刀·司命|贤者的庇护|机关匣|三分旗帜)|装备箱|获得普通装备|\d件[^。\n]{0,10}装备中随机获得")),
    # 只认"效果牌"类别的牌名（EFFECT_NAMES 运行时由 cards.json 里 category=效果牌 的卡名生成）。
    # 天赋送的牌（加速成长/复印机…）不算效果牌，算 gain_talent —— 镜/芈月"使用效果牌"是否认它们未经实测。
    ("gain_effect_card", R(r"(?:获得|随机获得|抽取|选择)(?:[^。\n，（%]{0,6}张)?[^。\n，（%]{0,14}?(?:效果牌|效果锦囊|战术牌?|EFFECT_NAMES)(?!牌)")),
    ("gain_talent",      R(r"天赋牌|可购买天赋|天赋3选1|天赋装备")),
    ("free_refresh",     R(r"免费刷新|临时刷新|刷新商店不消耗|额外获得\d次刷新|\d次刷新")),
    ("shop_mod",         R(r"商店")),
    ("energy_gain",      R(r"获得[^。\n，]{0,10}?\d+(?:-\d+)?(?:点)?能量|获得能量|能量改为|改为获得\d+能量|再获得\d+能量|额外获得\d+能量|额外获得1能量|获得0能量")),
    ("energy_interest",  R(r"每有7能量")),
    ("energy_cost",      R(r"花费\d+能量|消耗\d+(?:-\d+)?能量|扣除(?:当前)?所有能量|额外花费\d能量|购买消耗-|能量-\d|能量可用至")),
    ("lord_xp",          R(r"\d*棋手经验|升级所需")),
    ("lord_level",       R(r"棋手等级提升")),
    ("lord_hp",          R(r"棋手生命值|棋手每次受到的伤害|恢复\d+棋手生命值|棋手伤害")),
    ("pop_cap",          R(r"(?:可)?上阵人数\+|不占人口")),
    ("awaken_progress",  R(r"觉醒进度")),
    ("tier_upgrade",     R(r"变为(?:高\d阶|5阶|不在场的5阶|同阵营高1阶)|提升英雄兑换券的品阶|高1-2阶|高2阶的随机英雄|高1阶英雄牌")),
    ("copy_card",        R(r"复制|同名牌|获得该英雄牌|获得1张该英雄牌|一分为四|获得\d张该英雄牌|获得\d张相同|2张相同")),
    ("discover",         R(r"\d选\d|从\d(?:个|张|种)[^。\n]{0,16}中(?:选择|挑选)|选择1(?:个|张|项)|挑选1(?:张|个)|发现")),

    # ---- 效果：战斗
    ("summon",           R(r"召唤(?!单位|物)|创造1个|创造分身|生成1个")),
    ("totem",            R(r"图腾")),
    ("core_level",       R(r"阿科米亚核心")),
    ("coin",             R(r"古币")),
    ("core_up",          R(r"阿科米亚核心(?:的)?等级(?:额外)?\+|核心等级(?:额外)?\+|核心(?:会)?获得|核心等级\+")),
    ("payoff_core",      R(r"阿科米亚核心等级提升时|夺取阿科米亚核心|偷取阿科米亚核心|核心每拥有|核心失去|核心不再分配|优先分配给|参与阿科米亚核心的等级分配|核心的等级会以临时等级|核心每分配")),
    ("lord_mechanic",    R(r"小鹿|香香|狙击弹|炮弹|激光|子弹|琵琶|指法|惊鸿调|解冻|冻结的英雄|将其冻结|小菟|镜像|宝藏牌|卦象|至高决策|战略决策|特权决策|梦幻|纸伞|秘技牌|明先生|嬴律|小妲己|庄小鱼|孙小宾|白歌|马可每|玉环|弈星随机")),
    ("apply_coin",       R(r"使用\d张(?:精致)?古币|对[^，。\n]{0,10}使用1张(?:精致)?古币")),
    ("tactic",           R(r"战术")),
    ("mana",             R(r"法力值")),
    ("mana_reduce",      R(r"法力值上限(?:\{[^}]*\}|-|降低)|最大法力值-|无限释放技能|立即释放技能|法力值上限-")),
    ("shield",           R(r"护盾")),
    ("heal",             R(r"恢复[^。\n]{0,12}生命值|生命恢复|治疗|吸血|回复[^。\n]{0,6}生命")),
    ("damage",           R(r"以真实伤害|造成[^。\n]{0,40}?伤害|伤害(?:提升|加成|\+)|附带[^。\n]{0,20}伤害|重击|直接击败|消灭")),
    ("stat_offense",     R(r"攻击力|攻击速度|攻速|暴击|伤害加成|伤害提升|法术防御穿透|快速进行\d次普通攻击|以最大速度|普攻附带|快速攻击")),
    ("stat_defense",     R(r"防御|生命值上限|最大生命值|伤害减免|格挡|无敌|免疫|不可选中|不被敌人优先攻击|不会被优先攻击|承担伤害|伤害的30%将转移|受到的伤害-|每次受到伤害不会超过|隐形")),
    ("control",          R(r"眩晕|嘲讽|沉默|小猪|猪猪|击飞|缴械|禁锢|困入|(?<!免疫)控制\d秒|冻结1秒|减少目标\d+%移速|弹至")),
    ("remove_hero",      R(r"(?:将其)?移除|消除(?:场上|等级最低)|吞噬[^。\n]{0,6}(?:英雄|该格)|摧毁你所有的图腾|自动销毁并获得")),
    ("destroy_equipment", R(r"摧毁其所有装备")),
    ("sell_related",     R(r"出售")),
    ("maple",            R(r"枫|纸伞|转移(?:后|时|次数|所需|至纸伞)|翩跹舞步|惊鸿游龙|瞬步无影")),
    ("clone_enemy",      R(r"复制对方|复制1名敌方|将敌方[^。\n]{0,12}复制|偷取敌方同名|复制进你的手牌")),
    ("trap_device",      R(r"陷阱|弹簧|屏障|机关|擂台|穹顶|裂隙")),

    # ---- 基础装备：拖给英雄后四选一成品（见 data/equip_recipes.json）
    ("base_equipment",   R(r"可以铸造成1件")),
    ("forge_offense",    R(r"铸造成1件输出型")),
    ("forge_magic",      R(r"铸造成1件法术型")),
    ("forge_defense",    R(r"铸造成1件防御型")),
    ("forge_utility",    R(r"铸造成1件功能型")),

    # ---- 目标
    ("target_self",        R(r"自身|(?:^|[，\n])等级\+\d+")),
    ("target_wearer",      R(r"穿戴者|穿戴给英雄|穿戴它的")),
    ("target_nearest",     R(r"距离最近|最近(?:的)?1名|最近的【|最近1名|最近的敌人")),
    ("target_adjacent",    R(r"(?:左右|周围)1格(?:的)?(?:英雄|己方|友方|友军|单位|其他|英雄的)|周围1格单位")),
    ("target_ranged_only", R(r"限定远程")),
    ("target_melee_only",  R(r"限定近战")),
    ("melee_bonus",        R(r"近战英雄时，效果翻倍")),
    ("target_random",      R(r"随机(?:\d+名|1名|\d名|英雄|分配|对\d名|在其周围|召唤|传送|1个|变|触发)|场上随机")),
    ("target_all_allies",  R(r"(?:所有|全体|全员)(?:己方|友方)?(?:英雄|单位)|场上(?:所有|友方)(?:友方)?英雄|己方(?:所有|每名)|你的英雄(?:获得|释放|施加|普攻|合成)|该英雄等级永久|全体【|友方单位|友军")),
    ("target_chosen",      R(r"指定(?:1名|场上1名|1个|英雄)|对狄仁杰使用|若目标为|对沈梦溪使用|指向1名英雄")),
    ("target_highest_level", R(r"等级最高")),
    ("target_lowest_level",  R(r"等级最低|生命值百分比最低|生命值最低")),
    ("target_most_equipped", R(r"穿戴装备最多")),
    ("target_tier_limit",    R(r"\d-\d阶(?:的)?(?:英雄|召唤)|\d阶及以下|仅1-3阶|4/5阶")),
    ("target_same_tier",     R(r"同阶|品阶相同|同品阶|其品阶相同")),
    ("target_same_faction",  R(r"同阵营|其阵营|同一阵营")),
    ("target_enemy",         R(r"敌方|敌人|对手|击败者")),
]

# 少数卡的文字里没有可认的字，只能按卡名归类（同样是显式正则，可审计）
NAME_RULES: list[tuple[str, re.Pattern]] = [
    ("maple", R(r"枫|翩跹舞步")),          # 枫叶舞强化牌：如「枫佑永驻」文字只写"护盾量增加"
]

# 阵营相关的标签带参数，单独用捕获组算
FACTION_TARGET = R(r"【(" + FACTION_ALT + r")】(?:英雄(?!牌)|单位)|仅(三分)英雄穿戴|(" + FACTION_ALT + r")英雄(?!牌)")
FACTION_FETCH = R(r"【(" + FACTION_ALT + r")】(?:英雄牌|专属装备)|(" + FACTION_ALT + r")英雄牌")

# 标签 -> 英文短语（拼 "en" 描述用）
PHRASE = {
    "on_deploy": "deploy trigger", "on_prepare": "prepare trigger (every round start)",
    "on_sacrifice": "sacrifice trigger (on death)", "on_exit": "exit trigger (on sell)",
    "on_engage": "engage trigger (battle start)", "on_victory": "on win", "on_defeat": "on loss",
    "blink": "blink", "steal_level": "steals levels", "revive": "revive", "fleeting": "fleeting (vanishes at round end)",
    "link": "duo link bonus", "each_n_times": "counter every N",
    "payoff_deploy": "benefits when deploy effects fire", "payoff_prepare": "benefits when prepare effects fire",
    "payoff_engage": "benefits when engage effects fire", "payoff_sacrifice": "benefits when sacrifice units die",
    "payoff_merge": "boosts merges", "payoff_effect_card": "benefits when effect cards are played",
    "payoff_tactic": "scales with tactic cards", "payoff_skill_cast": "benefits on skill casts",
    "payoff_hero_play": "benefits when hero cards are played", "payoff_coin": "boosts ancient coins",
    "payoff_summon": "boosts summons", "payoff_totem": "scales with totems",
    "payoff_awakened": "scales with awakened heroes", "payoff_equipment": "scales with equipment",
    "payoff_temp_level": "uses temp levels", "payoff_shield_break": "on shield break",
    "payoff_control": "on crowd control", "payoff_kill": "on kills", "payoff_sell": "on selling heroes",
    "retrigger_deploy": "re-triggers deploy effects", "retrigger_prepare": "re-triggers prepare effects",
    "retrigger_cards": "replays cards", "grant_sacrifice": "grants sacrifice", "grant_blink": "grants blink",
    "fetch_deploy": "gets deploy heroes", "fetch_prepare": "gets prepare heroes",
    "fetch_sacrifice": "gets sacrifice heroes", "fetch_engage": "gets engage heroes", "fetch_blink": "gets blink heroes",
    "level_up_self_perm": "levels itself permanently", "level_up_self_temp": "gains temp levels itself",
    "level_up_target": "levels a chosen hero", "level_up_random": "levels random heroes",
    "level_up_all": "levels all heroes", "level_up_merge": "extra levels on merge",
    "level_up_per_hero": "levels each triggering hero",
    "temp_to_perm": "converts temp levels to permanent", "level_transfer": "moves levels between heroes",
    "level_scaling": "scales with level", "gain_hero": "gives hero cards", "gain_equipment": "gives equipment",
    "gain_effect_card": "gives effect cards", "free_refresh": "free shop refresh", "energy_gain": "gives energy",
    "lord_xp": "lord xp", "pop_cap": "+board slot", "summon": "summons units", "totem": "totem",
    "core_level": "arcomia core levels", "mana_reduce": "casts faster", "shield": "shield", "heal": "heal/lifesteal",
    "control": "crowd control", "remove_hero": "removes a hero", "maple": "maple dance (Ali)",
    "awaken_progress": "awaken progress", "tier_upgrade": "tier upgrade", "copy_card": "copies cards",
    "target_nearest": "targets nearest hero", "target_adjacent": "affects adjacent cells",
    "target_ranged_only": "ranged only", "target_melee_only": "melee only", "target_chosen": "you pick the target",
    "target_most_equipped": "targets most-equipped hero", "apply_coin": "applies ancient coins",
    "level_up_group": "levels a hero group", "steal_level": "steals levels", "payoff_blink": "boosts blink heroes",
    "core_up": "raises arcomia core", "payoff_core": "uses arcomia core", "gain_talent": "gives talent cards",
    "discover": "choose one of N", "gain_card": "gives cards", "stat_offense": "offense stats",
    "stat_defense": "defense stats", "damage": "deals damage", "on_battle_start": "at battle start",
    "on_round_start": "at round start", "on_lord_level_up": "on lord level up", "lord_hp": "lord hp",
    "energy_cost": "costs energy", "lord_level": "lord level +1", "destroy_equipment": "destroys equipment",
    "clone_enemy": "copies enemy units", "trap_device": "board device", "lord_mechanic": "lord-specific mechanic",
    "melee_bonus": "stronger on melee", "target_highest_level": "targets highest-level hero",
    "target_lowest_level": "targets lowest-level hero", "target_tier_limit": "tier-restricted",
    "payoff_lord_hp_loss": "stronger when losing", "revive": "revive", "level_up_on_count": "scales with board count",
    "base_equipment": "base item (pick 1 of 4 on equip)", "forge_offense": "forges attack item",
    "forge_magic": "forges magic item", "forge_defense": "forges defense item", "forge_utility": "forges utility item",
    "energy_interest": "energy interest",
}

# 描述里优先出现的标签顺序（越靠前越能区分一张卡）
PHRASE_ORDER = list(PHRASE)

# ---------------------------------------------------------------- 成长规则（英雄当主C好不好养）
# (id, regex, trigger, kind, needs, weight)
#   kind: perm = 永久等级；temp = 临时等级（战斗结束清空，除非有明世隐/短期企划）；
#         share = 从别处"分"等级；scale = 不涨等级但吃等级
#   weight: 对 grow_score 的贡献。按"触发频率 × 是否可控"定：
#     每回合自动 3~4；玩家可主动刷的事件（打效果牌/别人整备开团/放技能）4~5；
#     阵亡才触发 2；登场一次性 1；临时等级 2（要配明世隐才落袋）。
#   第一个捕获组（如有）是每次触发的等级数。
GROW_RULES = [
    ("perm_per_team_trigger", R(r"每当你的英雄触发(整备或开团|整备|开团|登场)效果时，自身等级永久\+(\d+)"),
     "any_prepare_or_engage", "perm", "prepare_engage_triggers", 5),
    ("perm_per_effect_cards", R(r"每使用(\d+)张效果牌，(?:自身)?等级\+(\d+)"),
     "effect_cards", "perm", "effect_cards", 5),
    ("perm_per_hero_play", R(r"每次使用[^。\n]{0,20}英雄牌时，自身[^。\n]{0,16}?等级\+(\d+)"),
     "hero_cards_played", "perm", "hero_cards_off_faction", 4),
    ("perm_on_prepare_self", R(r"整备：[^。\n]{0,6}自身[^。\n]{0,10}?等级\+(\d+)"),
     "prepare", "perm", "none_auto_each_round", 4),
    ("coin_self_on_prepare", R(r"整备：[^。\n]*?对自身使用1张(精致)?古币"),
     "prepare", "perm", "none_auto_each_round", 4),
    ("blink_team_on_prepare", R(r"整备：拥有闪现效果的英雄等级\+(\d+)"),
     "prepare", "perm", "self_has_blink", 4),
    ("perm_per_totem_engage", R(r"开团：[^。\n]*?(?:每摧毁1个|每有1个图腾)，?[^。\n]{0,6}自身[^。\n]{0,20}?等级(?:永久)?\+(\d+)"),
     "engage", "perm", "totems", 4),
    ("per_hero_skill_casts", R(r"你的英雄每释放(\d+)次技能，该英雄等级永久\+(\d+)"),
     "skill_casts", "perm", "skill_casts", 3),
    ("per_hero_equipment", R(r"开团：你的英雄每穿戴1件装备，该英雄等级永久\+(\d+)"),
     "engage", "perm", "equipment", 3),
    ("perm_on_self_sacrifice", R(r"牺牲：自身的?等级永久\+(\d+)"),
     "self_death", "perm", "dies_each_battle", 2),
    ("perm_on_deploy", R(r"登场：[^。\n]{0,6}自身[^。\n]{0,16}?等级\+(\d+)"),
     "deploy", "perm", "retrigger_deploy", 1),
    ("perm_most_equipped_faction", R(r"穿戴装备最多的1名【(?:日落海|逐鹿)】英雄等级\+(\d+)"),
     "prepare_or_effect_cards", "perm", "most_equipment", 3),
    ("perm_random_blink_kills", R(r"随机1名闪现英雄等级\+(\d+)"),
     "prepare", "perm", "kills", 1),
    ("perm_random_team", R(r"(?:场上随机3名己方英雄|随机3名英雄(?:及阿科米亚核心)?|随机2名[^。\n]{0,12}英雄)等级(?:永久)?\+(\d+)"),
     "various", "perm", "team_triggers", 1),
    ("perm_all_team", R(r"(?:全体|所有)英雄等级(?:永久)?\+(\d+)"),
     "various", "perm", "team_triggers", 1),
    ("temp_share_all", R(r"其他英雄获得临时等级时，自身获得(一半|全部)效果"),
     "any_temp_level", "temp", "temp_level_sources", 4),
    ("temp_per_sacrifice_death", R(r"拥有牺牲效果的英雄阵亡时，自身临时等级\+(\d+)"),
     "sacrifice_deaths", "temp", "sacrifice_units", 3),
    ("temp_per_team_skill", R(r"其他英雄释放技能时，自身获得\d+点法力值以及(\d+)临时等级"),
     "skill_casts", "temp", "skill_casts", 3),
    ("temp_per_totem", R(r"每个图腾每\d秒为自身提供[^。\n]{0,12}?(\d+)临时等级"),
     "battle_time", "temp", "totems", 3),
    ("temp_from_totem_levels", R(r"开团：获得每个图腾等级数(\d+)%的临时等级"),
     "engage", "temp", "totems", 2),
    ("temp_copy_half_random", R(r"整备：随机获得1名其他英雄等级数(一半)?的临时等级"),
     "prepare", "temp", "high_level_teammate", 2),
    ("temp_on_skill_team", R(r"你的英雄释放技能时，该英雄获得(\d+)临时等级"),
     "skill_casts", "temp", "skill_casts", 2),
    ("temp_on_sacrifice_self", R(r"牺牲：临时等级\+(\d+)"),
     "self_death", "temp", "dies_each_battle", 1),
    ("temp_coin_bonus_self", R(r"古币牌会额外提供(\d+)临时等级。若目标为自身"),
     "coins", "temp", "coins", 2),
    ("merge_bonus_all", R(r"你的英雄合成提升的等级\+(\d+)"),
     "merge", "perm", "copies_of_self", 2),
    ("perm_same_tier_merge", R(r"每当3阶及以下英雄合成时，与其同阶的英雄等级\+(\d+)"),
     "merge", "perm", "same_tier_merges", 2),
    ("deploy_repeat_on_hp_loss", R(r"棋手生命值每降低\d+点，重复1次"),
     "deploy", "perm", "low_lord_hp", 1),
    ("steal_on_deploy", R(r"^夺取|闪现，夺取"),
     "deploy", "share", "adjacent_teammate_levels", 1),
    ("scale_with_level", R(r"自身等级×|等级与自身相同|与自身等级相同|等级数\d+%"),
     "passive", "scale", "levels", 1),
]
# 阵营自带的"外部喂等级"通道：古币只能给【河洛】（且能指定目标），核心只分给【日落海】
FACTION_FEED = {"heluo": ("coins_targetable", 1.0), "riluohai": ("arcomia_core_share", 0.5)}


# ---------------------------------------------------------------- 工具

def base_name(name: str) -> str:
    return _SUFFIX.sub("", name)


def card_text(name: str, card: dict) -> tuple[str, str]:
    """返回 (skill, awaken)，把英雄/召唤物对自己的称呼统一替换成"自身"。"""
    skill = card.get("skill") or ""
    awaken = card.get("awaken") or ""
    if card.get("category") == "棋手牌":   # 棋手牌的文字在 talents[].cards[].desc 里
        parts = []
        for t in card.get("talents") or []:
            for c in t.get("cards") or []:
                parts.append(f"{c.get('name')}：{c.get('desc') or ''}")
        skill = "\n".join(parts)
    if card.get("category") in ("英雄牌", "召唤物"):
        me = base_name(name)
        if len(me) >= 1:
            skill = skill.replace(me, "自身")
            awaken = awaken.replace(me, "自身")
    return skill.strip(), awaken.strip()


def role_of(card: dict) -> tuple[str | None, str | None]:
    tags = card.get("tags") or []
    row = "front" if "前排" in tags else ("back" if "后排" in tags else None)
    for zh, en in ROLE_EN.items():
        if zh in tags:
            return en, row
    return (row and f"{row}line"), row


def range_of(card: dict) -> str | None:
    d = (card.get("properties") or {}).get("attackDistance")
    if not d:   # None 或 0（图腾/擂台不攻击）
        return None
    return "melee" if d == 1 else "ranged"


def first_int(s: str | None) -> float | None:
    if s is None:
        return None
    if s in ("一半",):
        return 0.5
    if s in ("全部",):
        return 1.0
    if s == "精致":
        return 4
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------- 标注

def tag_card(name: str, card: dict, hero_alt: str, effect_alt: str) -> dict:
    skill, awaken = card_text(name, card)
    text = skill + ("\n" + awaken if awaken else "")
    tags: list[str] = []
    hits: dict[str, str] = {}
    for tag, rx in TAG_RULES:
        pat = rx
        if "HERO_NAMES" in rx.pattern or "EFFECT_NAMES" in rx.pattern:
            pat = re.compile(rx.pattern.replace("HERO_NAMES", hero_alt).replace("EFFECT_NAMES", effect_alt), re.M)
        m = pat.search(text)
        if m:
            tags.append(tag)
            hits[tag] = text[m.start():m.end() + 8]   # 多带几个字，方便抠后面的数字
    for tag, rx in NAME_RULES:
        if tag not in tags and rx.search(name):
            tags.append(tag)
            hits[tag] = "name:" + name
    for m in FACTION_TARGET.finditer(text):
        f = FACTION_EN.get(next(g for g in m.groups() if g))
        t = f"target_faction:{f}"
        if f and t not in tags:
            tags.append(t)
            hits[t] = m.group(0)
    for m in FACTION_FETCH.finditer(text):
        f = FACTION_EN.get(next(g for g in m.groups() if g))
        t = f"fetch_faction:{f}"
        if f and t not in tags:
            tags.append(t)
            hits[t] = m.group(0)

    cat = card.get("category")
    role, row = role_of(card) if cat in ("英雄牌", "召唤物") else (None, None)
    feat = {
        "category": CATEGORY_EN.get(cat, cat),
        "tier": card.get("quality"),
        "cost": card.get("cost"),
        "faction": FACTION_EN.get(card.get("faction"), card.get("faction")),
        "role": role,
        "row": row,
        "range": range_of(card) if cat in ("英雄牌", "召唤物") else None,
        "tags": tags,
        "nums": {},
        "_hits": hits,       # 调试用：每个标签命中的原文片段（写文件时去掉）
        "_text": text,
    }
    # 关键数字：命中片段里的第一个 +N / N临时等级 / N能量
    for t, frag in hits.items():
        if t == "energy_interest" or not t.startswith(("level_up", "energy", "lord_xp", "free_refresh", "temp_to_perm")):
            continue
        m = re.search(r"\+\s*(\d+)|(\d+)临时等级|(\d+)(?:点)?能量|(\d+)棋手经验|提升(\d+)级", frag)
        if m:
            feat["nums"][t] = int(next(g for g in m.groups() if g))
    if cat in ("英雄牌", "召唤物") and role:
        feat["grow"], feat["grow_score"] = grow_of(skill, awaken, feat)
    return feat


def grow_of(skill: str, awaken: str, feat: dict) -> tuple[dict, int]:
    """按 GROW_RULES 算"这个英雄当主C时自己涨等级有多快"。只看普通版文字，
    觉醒版通常是数值翻倍，记在 awaken_mult 里。"""
    matched = []
    spans: list[tuple[int, int]] = []
    # 权重高的规则先认；同一段原文只算一次（虞姬"自身及随机2名…+1"不能既算自身成长又算随机成长）
    for rid, rx, trig, kind, needs, w in sorted(GROW_RULES, key=lambda r: -r[5]):
        m = rx.search(skill)
        if not m:
            continue
        if any(m.start() < e and s < m.end() for s, e in spans):
            continue
        spans.append((m.start(), m.end()))
        groups = [g for g in m.groups() if g is not None]
        n = None
        if rid == "perm_per_effect_cards" and len(groups) == 2:
            n = float(groups[1]) / float(groups[0])          # 每张效果牌折合几级
        elif rid == "per_hero_skill_casts" and len(groups) == 2:
            n = float(groups[1]) / float(groups[0])
        elif rid == "coin_self_on_prepare":
            n = 4.0 if groups and groups[-1] == "精致" else 2.0
        elif rid == "perm_per_team_trigger" and groups:
            n = first_int(groups[-1])
        elif groups:
            n = first_int(groups[-1]) if rid != "temp_share_all" else first_int(groups[0])
        matched.append({"rule": rid, "trigger": trig, "kind": kind, "needs": needs,
                        "per_trigger": n, "weight": w, "text": m.group(0)})
    score = sum(x["weight"] for x in matched)
    fac = FACTION_FEED.get(feat.get("faction"))
    if fac:
        score += fac[1]
    if "level_scaling" in feat["tags"] and not any(x["rule"] == "scale_with_level" for x in matched):
        score += 0.5
    # 觉醒倍率：普通/觉醒两版里第一个 "+N" 的比值
    mult = None
    a = re.search(r"\+\s*(\d+)|(\d+)临时等级", awaken or "")
    s = re.search(r"\+\s*(\d+)|(\d+)临时等级", skill or "")
    if a and s:
        av = int(next(g for g in a.groups() if g))
        sv = int(next(g for g in s.groups() if g))
        if sv:
            mult = round(av / sv, 2)
    primary = max(matched, key=lambda x: x["weight"]) if matched else None
    grow = {
        "self_perm_per_trigger": (primary["per_trigger"] if primary and primary["kind"] == "perm" else 0) or 0,
        "self_temp_per_trigger": (primary["per_trigger"] if primary and primary["kind"] == "temp" else 0) or 0,
        "trigger": primary["trigger"] if primary else "none",
        "needs": primary["needs"] if primary else "external_level_cards",
        "kind": primary["kind"] if primary else None,
        "faction_feed": fac[0] if fac else None,
        "awaken_mult": mult,
        "rules": [{k: v for k, v in x.items()} for x in matched],
    }
    return grow, int(min(10, round(score)))


def make_en(name: str, f: dict) -> str:
    """给 Laya 的一行英文特征。不含卡名（卡名对模型是噪声）。"""
    head = [f["category"]]
    if f.get("tier") is not None:
        head.append(f"t{f['tier']}")
    if f.get("faction") and f["faction"] not in ("any", "neutral"):
        head.append(f["faction"])
    elif f.get("faction") == "neutral" and f["category"] == "hero":
        head.append("neutral")
    if f.get("role"):
        head.append(f["role"])
    if f.get("range"):
        head.append(f["range"])
    phrases = []
    for t in PHRASE_ORDER:
        if t in f["tags"]:
            p = PHRASE[t]
            n = f["nums"].get(t)
            phrases.append(f"{p} +{n}" if n else p)
    facs = [t.split(":")[1] for t in f["tags"] if t.startswith("target_faction:")]
    if facs:
        phrases.append("only " + "/".join(facs) + " heroes")
    s = " ".join(head)
    if "grow_score" in f:
        s += f" | grow {f['grow_score']}"
        if f["grow"]["needs"] not in ("external_level_cards",):
            s += f" via {f['grow']['needs']}"
    if phrases:
        s += " | " + "; ".join(phrases[:8])
    return s


def build(cards: dict) -> dict:
    heroes = sorted({base_name(n) for n, c in cards.items() if c.get("category") == "英雄牌"}, key=len, reverse=True)
    hero_alt = "|".join(map(re.escape, heroes))
    effects = sorted({base_name(n) for n, c in cards.items() if c.get("category") == "效果牌"}, key=len, reverse=True)
    effect_alt = "|".join(map(re.escape, effects))
    # 阵容数据：这个英雄在官网 455 套阵容里当主C / 出现的次数（数据统计，不是标签）
    carry_n, use_n = Counter(), Counter()
    if LINEUPS.exists():
        for l in json.loads(LINEUPS.read_text(encoding="utf-8"))["lineups"]:
            if l.get("carry"):
                carry_n[l["carry"]] += 1
            for h in {h["name"] for h in l.get("heroes") or []}:
                use_n[h] += 1
    out = {}
    for name, card in cards.items():
        f = tag_card(name, card, hero_alt, effect_alt)
        if f["category"] == "hero":
            f["lineup_carry"] = carry_n.get(name, 0)
            f["lineup_use"] = use_n.get(name, 0)
        f["en"] = make_en(name, f)
        out[name] = f
    return out


# ---------------------------------------------------------------- 叠加关系（产出方 -> 受益方）

# 每种"事件"：谁产出它、谁吃它。都由上面的标签组合而来。
SYNERGY_AXES = {
    "deploy":      (["on_deploy", "retrigger_deploy", "fetch_deploy"], ["payoff_deploy"]),
    "prepare":     (["on_prepare", "retrigger_prepare", "fetch_prepare"], ["payoff_prepare"]),
    "engage":      (["on_engage", "fetch_engage"], ["payoff_engage"]),
    "sacrifice":   (["on_sacrifice", "grant_sacrifice", "fetch_sacrifice"], ["payoff_sacrifice"]),
    "effect_card": (["gain_effect_card"], ["payoff_effect_card"]),
    "tactic":      (["tactic"], ["payoff_tactic"]),
    "coin":        (["apply_coin", "coin"], ["payoff_coin"]),
    "temp_level":  (["level_up_temp", "level_up_self_temp"], ["payoff_temp_level", "temp_to_perm"]),
    "merge":       (["copy_card"], ["payoff_merge"]),
    "skill_cast":  (["mana_reduce"], ["payoff_skill_cast"]),
    "summon":      (["summon"], ["payoff_summon"]),
    "totem":       (["totem"], ["payoff_totem"]),
    "equipment":   (["gain_equipment"], ["payoff_equipment"]),
    "blink":       (["blink", "grant_blink", "fetch_blink"], ["payoff_blink"]),
    "sell":        (["sell_related"], ["payoff_sell"]),
    "hero_play":   (["gain_hero"], ["payoff_hero_play"]),
    "awakened":    (["awaken_progress"], ["payoff_awakened"]),
    "shield_break": (["shield"], ["payoff_shield_break"]),
    "control":     (["control"], ["payoff_control"]),
    "core":        (["core_up"], ["payoff_core"]),
}


def synergy(feats: dict, lineups: list) -> dict:
    res = {}
    for axis, (prod_tags, cons_tags) in SYNERGY_AXES.items():
        def has(f, ts):
            return any(t in f["tags"] for t in ts)
        prods = [n for n, f in feats.items() if has(f, prod_tags) and not n.endswith(")")]
        cons = [n for n, f in feats.items() if has(f, cons_tags) and not n.endswith(")")]
        # 阵容里同时出现产出方英雄和受益方英雄的次数（按使用人数加权）
        combos = Counter()
        lu_names = defaultdict(list)
        for l in lineups:
            names = {h["name"] for h in l.get("heroes") or []}
            for st in l.get("stages") or []:
                names |= set(st.get("main") or []) | set(st.get("sub") or [])
            names |= set(l.get("effect_cards") or []) | set(l.get("talent_cards") or [])
            ps = [p for p in prods if p in names]
            cs = [c for c in cons if c in names]
            for c in cs:
                for p in ps:
                    if p != c:
                        combos[(p, c)] += 1
                        lu_names[(p, c)].append(l["name"])
        res[axis] = {"producers": prods, "consumers": cons,
                     "lineup_pairs": [(p, c, k, lu_names[(p, c)][:3]) for (p, c), k in combos.most_common(12)]}
    return res


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--untagged", action="store_true", help="列出 0 标签的卡")
    ap.add_argument("--synergy", action="store_true", help="打印叠加关系")
    ap.add_argument("--card", help="查看某张卡的标注细节")
    ap.add_argument("--top", type=int, default=25, help="打印 grow_score 前 N 名英雄")
    args = ap.parse_args()

    cards = json.loads(CARDS.read_text(encoding="utf-8"))["cards"]
    feats = build(cards)

    if args.card:
        f = feats.get(args.card)
        if not f:
            print("no such card"); return 1
        print(f["_text"]); print(json.dumps({k: v for k, v in f.items() if k != "_text"}, ensure_ascii=False, indent=1))
        return 0

    # 写文件（去掉调试字段）
    clean = {n: {k: v for k, v in f.items() if not k.startswith("_")} for n, f in feats.items()}
    for f in clean.values():
        if "grow" in f:
            for r in f["grow"]["rules"]:
                r.pop("text", None)
    OUT.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")

    # 覆盖率
    print(f"wrote {OUT.relative_to(ROOT)}  cards={len(feats)}")
    print(f"{'category':<10}{'n':>5}{'0 tags':>8}{'<=1 tag':>9}{'avg tags':>10}")
    by = defaultdict(list)
    for n, f in feats.items():
        by[f["category"]].append((n, f))
    untagged = []
    for cat, items in sorted(by.items(), key=lambda x: -len(x[1])):
        zero = [n for n, f in items if not f["tags"]]
        le1 = [n for n, f in items if len(f["tags"]) <= 1]
        avg = sum(len(f["tags"]) for _, f in items) / len(items)
        print(f"{cat:<10}{len(items):>5}{len(zero):>8}{len(le1):>9}{avg:>10.1f}")
        untagged += [(cat, n) for n in zero]
    tc = Counter(t for f in feats.values() for t in f["tags"])
    print(f"distinct tags: {len(tc)}")
    print("rarest tags:", ", ".join(f"{t}={c}" for t, c in sorted(tc.items(), key=lambda x: x[1])[:12]))
    if untagged:
        print(f"untagged ({len(untagged)}):", ", ".join(f"{n}[{c}]" for c, n in untagged))

    # grow 排名
    heroes = [(n, f) for n, f in feats.items() if f["category"] == "hero"]
    heroes.sort(key=lambda x: (-x[1]["grow_score"], x[0]))
    print(f"\ntop {args.top} heroes by grow_score:")
    for n, f in heroes[:args.top]:
        g = f["grow"]
        print(f"  {f['grow_score']:>2}  {n:<6} t{f['tier']} {f['faction']:<9} {g['trigger']:<22} needs={g['needs']:<24} "
              f"rules={[r['rule'] for r in g['rules']]}")
    print("grow_score histogram:", dict(sorted(Counter(f["grow_score"] for _, f in heroes).items())))

    if args.untagged:
        for c, n in untagged:
            print(f"--- {n} [{c}]\n{feats[n]['_text']}")

    if args.synergy:
        lineups = json.loads(LINEUPS.read_text(encoding="utf-8"))["lineups"]
        syn = synergy(feats, lineups)
        for axis, d in syn.items():
            print(f"\n== {axis}: producers {len(d['producers'])}, consumers {len(d['consumers'])}")
            hp = [p for p in d["producers"] if feats[p]["category"] == "hero"]
            hc = [c for c in d["consumers"] if feats[c]["category"] in ("hero",)]
            print("  hero producers:", " ".join(hp))
            print("  other producers:", " ".join(p for p in d["producers"] if p not in hp)[:600])
            print("  hero consumers:", " ".join(hc))
            print("  other consumers:", " ".join(c for c in d["consumers"] if c not in hc)[:600])
            for p, c, k, ls in d["lineup_pairs"]:
                print(f"    {p} -> {c}: {k} lineups  e.g. {ls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
