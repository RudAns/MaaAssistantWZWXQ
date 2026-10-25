# -*- coding: utf-8 -*-
"""重建「装备 × 英雄」关系图 -> data/equip_relations.json + 说明文档

原则（2026-10-01 起因：给马超穿了远程才能带的装备；需逐件分析装备描述、制定关系图）：
  卡面硬错配（伤害类型 / 远近战条件 / 穿戴限制）优先；官方阵容次数只是 ±1 的弱先验，不能把硬错配翻成推荐。
用法：.venv/Scripts/python.exe -X utf8 build_equip_relations.py <项目根目录>
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))
CARDS = json.loads((ROOT / "data/cards.json").read_text(encoding="utf-8"))["cards"]
RECIPES = json.loads((ROOT / "data/equip_recipes.json").read_text(encoding="utf-8"))["recipes"]
ROLES = json.loads((ROOT / "data/hero_roles.json").read_text(encoding="utf-8"))["heroes"]
LINEUPS = json.loads((ROOT / "data/lineups.json").read_text(encoding="utf-8"))["lineups"]

EQUIPS = {}
for c in CARDS.values():
    if c.get("category") == "装备牌":
        EQUIPS.setdefault(c["name"], c)
HEROES = {c["name"]: c for c in CARDS.values() if c.get("category") == "英雄牌"}
assert len(HEROES) == 85, len(HEROES)

# ---------------------------------------------------------------------------------------------
# 1. 装备属性（逐件读原文后手工结构化；每条 trigger/condition/effect 的引文都会校验是否真在原文里）
#    dmg   : 数值部分要求的伤害类型 phys/mag/None
#    share : 价值里有多少来自"伤害类型相关的数值"（错配时这部分作废），其余是与类型无关的效果
#    carry/tank/sup : 给"理想匹配的输出位 / 前排坦克 / 辅助"时的基准分（0~10）
#    trig  : auto=普攻触发  skill=放技能触发  mana=回蓝/法力  crit=暴击  ""=无
#    rng   : ranged_only / melee_only / ranged_better(近战效果减半) / melee_better(要贴脸) / isolated(身边无友军)
#    restrict : ("ranged",) ("melee",) ("faction",X) ("tier",3)
# ---------------------------------------------------------------------------------------------
I = {}


def item(name, cat, *, dmg=None, share=0.0, carry=0, tank=0, sup=0, trig="", rng=None, restrict=None,
         stats=None, triggers=(), conds=(), effects=(), keeper=False, ctl=False, note=""):
    I[name] = dict(cat=cat, dmg=dmg, share=share, carry=carry, tank=tank, sup=sup, trig=trig, rng=rng,
                   restrict=restrict, stats=stats or {}, triggers=list(triggers), conds=list(conds),
                   effects=list(effects), keeper=keeper, ctl=ctl, note=note)


# ---- 风暴巨剑 ----
item("无尽巨剑", "暴击", dmg="phys", share=0.9, carry=9, tank=1, sup=0, trig="crit",
     stats={"暴击率": "+30%", "暴击效果": "+60%"}, effects=[("暴击增伤", "暴击率+30%，暴击效果+60%")],
     conds=[("暴击效果不叠加", "暴击效果不可叠加")],
     note="法术伤害默认不能暴击（亚连被动 / 怒神项链除外），法系穿等于白板")
item("制裁之剑", "功能", carry=6, tank=3, sup=6, trig="",
     triggers=[("造成伤害", "造成伤害时降低目标30%生命恢复及护盾效果"), ("普攻加强", "若为普通攻击，效果提升至70%")],
     effects=[("重伤", "降低目标30%生命恢复及护盾效果")], conds=[("唯一", "唯一")])
item("纯净苍穹", "输出-法术", dmg="mag", share=0.7, carry=7, tank=3, sup=3,
     stats={"法术攻击力": "每2秒+10（成长）"}, triggers=[("每2秒", "每2秒获得10点法术攻击力")],
     effects=[("净化", "清除自身负面状态"), ("战斗内成长", "每2秒获得10点法术攻击力")], conds=[("唯一", "唯一")])
item("铸梦·逐风", "攻速", dmg="phys", share=0.6, carry=7, tank=1, sup=0, trig="auto",
     triggers=[("开战+每8秒", "战斗开始及之后每过8秒"), ("普攻", "快速进行4次普通攻击")],
     effects=[("额外普攻", "快速进行4次普通攻击")], conds=[("唯一", "唯一")],
     note="额外的是普攻：法系英雄普攻吃物攻(24)，几乎没用")
# ---- 雷鸣刃 ----
item("冥王", "输出-物理", dmg="phys", share=0.6, carry=7, tank=5, sup=0, rng="melee_better",
     stats={"物理防御": "+10/每个相邻敌人", "基础物理攻击力": "+15%/每个相邻敌人"},
     conds=[("要贴脸", "穿戴者周围1格每有1个敌方单位"), ("唯一", "唯一")],
     effects=[("被围增益", "获得10点物理防御和15%基础物理攻击力")])
item("名刀·司命", "保命", carry=6, tank=4, sup=4,
     triggers=[("致命伤害", "穿戴者首次受到致命伤害时")], effects=[("无敌", "进入无敌和不可选中状态，持续1.5秒")],
     conds=[("唯一", "唯一")])
item("红魔", "输出-通用", carry=7, tank=3, sup=0, rng="isolated",
     stats={"伤害加成": "+25%（身边无友军）；每承受3%最大生命+1%，上限100%"},
     conds=[("身边无友军", "周围1格没有友军时"), ("要挨打", "每承受3%最大生命值的伤害"), ("唯一", "唯一")],
     effects=[("增伤", "获得25%伤害加成")])
item("魔律之刃", "施法", carry=8, tank=3, sup=5, trig="skill",
     triggers=[("放技能", "释放技能时，35%概率重复释放一次")], effects=[("技能重复", "重复释放一次")],
     conds=[("按等级成长", "每有7级，额外提升1%概率"), ("唯一", "唯一")])
# ---- 速击之枪 ----
item("凝渊", "暴击", dmg="phys", share=0.8, carry=7, tank=2, sup=0, trig="crit", rng=None,
     stats={"暴击率": "+20%（受击+1%）"}, triggers=[("受击", "每次受击时提升1%暴击率")],
     effects=[("暴击", "暴击率+20%"), ("满暴加速", "暴击率达到100%时提升100%移动速度")], conds=[("唯一", "唯一")])
item("日渊", "输出-物理", dmg="phys", share=0.8, carry=8, tank=0, sup=0, trig="auto", restrict=("ranged",),
     triggers=[("普攻", "普通攻击会释放飞箭")], effects=[("追加物理伤害", "对生命值最低的2名敌人造成45%物理攻击力的物理伤害")],
     conds=[("限定远程", "限定远程英雄"), ("唯一", "唯一")])
item("影牙", "攻速", carry=7, tank=1, sup=0, trig="auto", stats={"攻击速度": "+50%"}, effects=[("攻速", "攻击速度+50%")])
item("破晓", "输出-物理", dmg="phys", share=0.6, carry=8, tank=1, sup=0, trig="auto",
     triggers=[("普攻", "每次普通攻击降低目标3点物防")],
     effects=[("破甲", "降低目标3点物防"), ("追加物理伤害", "附加1次目标15%最大生命值的物理伤害")])
# ---- 大棒 ----
item("虚无法杖", "输出-法术", dmg="mag", share=1.0, carry=6, tank=0, sup=1, stats={"法术穿透": "+15"},
     effects=[("法穿", "获得15点法术防御穿透")], conds=[("唯一", "唯一")])
item("寒霜侵袭", "输出-通用", carry=7, tank=2, sup=2, trig="auto",
     triggers=[("普攻", "普通攻击减少目标15%移速")],
     effects=[("减速", "减少目标15%移速"), ("易伤", "叠满后目标受到的伤害提高20%")], conds=[("唯一", "唯一")])
item("金色圣剑", "输出-法术", dmg="mag", share=0.5, carry=8, tank=1, sup=0, trig="auto",
     stats={"攻击速度": "+30%（+15%法术攻击力）", "普攻附带法术伤害": "20（+30%法术攻击力）"},
     triggers=[("普攻", "每次普通攻击附带20（+30%法术攻击力）点法术伤害")],
     effects=[("攻速", "获得30%（+15%法术攻击力）攻击速度"), ("法术追伤", "附带20（+30%法术攻击力）点法术伤害")],
     conds=[("唯一", "唯一")], note="攻速那半与伤害类型无关；追伤和攻速加成都按法攻缩放")
item("自然法杖", "输出-法术", dmg="mag", share=0.6, carry=7, tank=1, sup=3, trig="mana", rng="ranged_better", ctl=True,
     stats={"法术攻击力": "+20%"}, triggers=[("耗蓝", "远程穿戴者每消耗40点法力值（近战80点法力值）")],
     effects=[("法攻", "法术攻击力+20%"), ("控制·变猪", "会将随机1名敌人变成小猪，持续2秒")],
     conds=[("近战减半", "（近战80点法力值）"), ("唯一", "唯一")],
     note="2026-10-01 马超事故：法攻+20%对物理英雄是 0，变猪近战频率减半")
# ---- 圣者法典 ----
item("贤者之书", "施法", dmg="mag", share=0.75, carry=8, tank=0, sup=2, trig="skill",
     stats={"法力值上限": "-10", "基础法术攻击力": "+20%/层（放技能叠，最多6层）"},
     triggers=[("放技能", "释放技能后获得20%基础法术攻击力")], effects=[("法攻叠层", "最多叠加6层"), ("减蓝耗", "法力值上限-10")],
     conds=[("唯一", "唯一")])
item("炽热支配者", "保命", carry=7, tank=5, sup=4,
     triggers=[("首次被控", "首次受到控制效果时")], effects=[("护盾", "获得300点护盾"), ("免控", "免疫控制效果（压制除外）")],
     conds=[("唯一", "唯一")])
item("生命魔典", "坦克", carry=2, tank=7, sup=5,
     triggers=[("受伤", "受到伤害时会为自身和周围1格的己方单位恢复5%最大生命值")],
     effects=[("群体治疗", "恢复5%最大生命值")], conds=[("冷却2秒", "冷却2秒"), ("唯一", "唯一")])
item("御灵之纱", "坦克", carry=3, tank=7, sup=6, ctl=True,
     triggers=[("开战", "战斗开始时，获得等级×15点护盾"), ("破盾", "护盾被击破时")],
     effects=[("护盾", "获得等级×15点护盾"), ("控制·眩晕", "对周围1格敌人造成1秒眩晕")], conds=[("唯一", "唯一")])
# ---- 破碎圣杯 ----
item("痛苦面具", "输出-法术", dmg="mag", share=1.0, carry=7, tank=0, sup=1,
     triggers=[("单次法伤>8%", "单次魔法伤害超过目标最大生命值的8%时")],
     effects=[("点燃·真伤", "每秒造成75%法术攻击力的真实伤害")], conds=[("要法术爆发", "单次魔法伤害超过目标最大生命值的8%"), ("唯一", "唯一")])
item("法源之杯", "施法", carry=7, tank=4, sup=7, trig="mana", stats={"法力回复": "+5/秒"},
     effects=[("回蓝", "每秒回复5点法力值")], conds=[("唯一", "唯一")])
item("破茧之衣", "功能", carry=4, tank=6, sup=8, trig="mana", ctl=True, stats={"开战法力": "+15"},
     triggers=[("开战", "战斗开始时，法力值+15")],
     effects=[("控制·眩晕", "眩晕敌方等级最高3名英雄中的随机1名"), ("按目标等级延长", "目标每有10级，眩晕时间增加0.5秒")],
     conds=[("唯一", "唯一")])
item("聚灵水晶", "坦克", carry=3, tank=7, sup=4, trig="mana",
     triggers=[("受击", "受击时额外获得10点法力值")], effects=[("回蓝", "额外获得10点法力值"), ("回血", "恢复20%最大生命值")],
     conds=[("唯一", "唯一")])
# ---- 守护者之铠 ----
item("不祥征兆", "坦克", carry=2, tank=7, sup=4, triggers=[("开战15秒", "战斗前15秒")],
     effects=[("格挡普攻", "格挡50%普通攻击伤害")], conds=[("只前15秒", "战斗前15秒"), ("唯一", "唯一")])
item("极寒风暴", "坦克", carry=1, tank=8, sup=4, ctl=True,
     triggers=[("被普攻", "受到普攻时，降低目标20%攻击速度")],
     effects=[("减攻速", "降低目标20%攻击速度"), ("控制·冻结", "远程目标将被冻结1秒")], conds=[("近战目标减半", "近战目标减半"), ("唯一", "唯一")])
item("近卫荣耀", "坦克", carry=2, tank=7, sup=8, triggers=[("开战", "战斗开始时")],
     effects=[("光环双抗", "周围1格己方单位获得10点物理和法术防御"), ("自身双倍", "穿戴者获得双倍效果")],
     conds=[("不可叠加", "不可叠加"), ("唯一", "唯一")])
item("反伤重甲", "坦克", carry=2, tank=8, sup=2, triggers=[("受伤", "将受到伤害的30%")],
     effects=[("反伤·真伤", "以真实伤害回敬给目标")], conds=[("唯一", "唯一")])
# ---- 力量腰带 ----
item("霸者重铠", "坦克", carry=3, tank=8, sup=5, effects=[("减伤", "生命值高于70%时获得40%伤害减免")],
     conds=[("高血量才生效", "生命值高于70%时"), ("唯一", "唯一")])
item("不死鸟之眼", "坦克", carry=3, tank=8, sup=5, stats={"基础生命值": "+15%", "生命恢复与护盾效果": "+30%"},
     effects=[("加血", "增加15%基础生命值"), ("回复护盾增强", "生命恢复和护盾效果提升30%")], conds=[("唯一", "唯一")])
item("魔女斗篷", "坦克", carry=3, tank=6, sup=5, stats={"法术防御力": "+30"},
     triggers=[("开战", "战斗开始时")], effects=[("法抗", "法术防御力+30"), ("法术护盾", "获得等级x5点法术护盾")])
item("红莲炎甲", "坦克", carry=2, tank=8, sup=1, rng="melee_better",
     triggers=[("每2秒", "每2秒对周围1格敌人造成")],
     effects=[("灼烧", "20（+5%穿戴者最大生命值）点法术伤害"), ("重伤", "降低目标50%生命恢复和吸血效果")],
     conds=[("要贴脸", "周围1格敌人"), ("唯一", "唯一")])
# ---- 其他（非四选一）----
item("双剑·雄", "专属", carry=5, tank=1, sup=0, trig="auto", stats={"攻击速度": "+30%"},
     effects=[("攻速", "攻击速度+30%"), ("干将专属", "若穿戴者为干将莫邪，其法力值上限降低至10")],
     note="干将莫邪在场时自动获得")
item("玄微之种", "功能", carry=1, tank=6, sup=5, triggers=[("牺牲", "穿戴者获得牺牲")],
     effects=[("赋予牺牲", "穿戴者获得牺牲"), ("阵亡治疗", "生命值百分比最低的己方英雄恢复200（+20%最大生命值）点生命值")])
item("学识宝石", "成长", carry=8, tank=4, sup=4, triggers=[("每回合开始", "每回合开始时等级+8%")],
     effects=[("永久升级", "等级+8%（至少+3级）"), ("减蓝耗", "减少2点法力值上限")], conds=[("唯一", "唯一")])
item("百锻刀", "成长", dmg="phys", share=1.0, carry=8, tank=1, sup=0, keeper=True,
     stats={"物理攻击力": "+30（击杀永久+5）"}, triggers=[("击杀", "每击败1名英雄")],
     effects=[("永久成长", "永久获得5点物理攻击力")], conds=[("无法拆卸", "无法拆卸"), ("唯一", "唯一")])
item("聚灵法杖", "成长", dmg="mag", share=1.0, carry=8, tank=1, sup=1, trig="auto", keeper=True,
     stats={"法术攻击力": "+30（击杀永久+5）"}, triggers=[("普攻", "普攻附带自身70%法术攻击力"), ("击杀", "每击败1名英雄")],
     effects=[("永久成长", "永久获得5点法术攻击力")], conds=[("无法拆卸", "无法拆卸"), ("唯一", "唯一")])
item("分身镜", "功能", carry=6, tank=4, sup=3, triggers=[("开战", "战斗开始时")],
     effects=[("复制体", "召唤1个承受350%伤害的复制体")], conds=[("复制体无卡牌效果", "（无卡牌效果）"), ("唯一", "唯一")])
item("捕能手套", "功能", carry=5, tank=1, sup=1, triggers=[("击杀", "每击败1名英雄，获得1能量")],
     effects=[("经济", "获得1能量"), ("能量转换", "获得与棋手等级相对应的效果锦囊牌")], conds=[("唯一", "唯一")])
item("鼓舞之戒", "功能", carry=3, tank=5, sup=6, triggers=[("开战", "战斗开始时")],
     effects=[("同阵营加血", "与穿戴者同阵营的英雄，提升30%最大生命值")], conds=[("看同阵营人数", "与穿戴者同阵营的英雄"), ("唯一", "唯一")])
item("小鹿的庇护", "保命", carry=5, tank=4, sup=3, effects=[("转移伤害", "穿戴者所受伤害的30%将转移到小鹿身上")],
     conds=[("唯一", "唯一")])
item("静谧之铃", "功能", dmg="mag", share=0.3, carry=5, tank=3, sup=6,
     triggers=[("敌方放技能", "敌方等级最高的两名英雄释放技能后")],
     effects=[("断蓝", "5秒内无法恢复法力值"), ("法术伤害", "每0.5秒受到穿戴者20%法术攻击力的法术伤害")], conds=[("唯一", "唯一")])
item("盛华", "成长", carry=3, tank=8, sup=5, keeper=True, triggers=[("战斗结束", "每次战斗结束后永久增加200点生命值")],
     effects=[("永久加血", "永久增加200点生命值")], conds=[("无法拆卸", "无法拆卸"), ("唯一", "唯一")])
item("贤者的庇护", "保命", carry=7, tank=5, sup=4, restrict=("faction", "河洛"),
     triggers=[("首次阵亡", "首次被击败的2秒后复生")], effects=[("复生", "复生，仅恢复30%生命值")],
     conds=[("限河洛", "可穿戴给【河洛】英雄")])
item("青莲剑谱", "暴击", dmg="phys", share=0.6, carry=6, tank=2, sup=0, trig="crit", restrict=("melee",),
     stats={"暴击率": "+40%"}, effects=[("暴击", "暴击率+40%"), ("替换技能", "穿戴者释放的技能替换为青莲剑歌")],
     conds=[("限定近战", "限定近战英雄"), ("唯一", "唯一")], note="技能被替换：自身技能是核心的英雄不要")
item("不屈意志", "成长", carry=4, tank=4, sup=4, triggers=[("开战+每3秒", "战斗开始及之后每3秒"), ("阵亡", "阵亡时")],
     effects=[("临时等级", "获得场上友军穿戴不屈意志数量x2的临时等级"), ("阵亡传等级", "将自身等级的30%平均分配给穿戴不屈意志的友军")],
     conds=[("要多人穿", "穿戴不屈意志数量"), ("唯一", "唯一")])
item("征服之剑", "输出-物理", dmg="phys", share=1.0, carry=7, tank=2, sup=0, stats={"物理吸血": "+30%（<50%血翻倍）"},
     effects=[("物理吸血", "物理吸血+30%")], conds=[("低血翻倍", "生命值低于50%时")])
item("噬神之书", "输出-法术", dmg="mag", share=1.0, carry=9, tank=0, sup=2, stats={"法术攻击力": "+100", "法术吸血": "+60%"},
     effects=[("法攻", "法术攻击力+100"), ("法术吸血", "法术吸血+60%")])
item("梦魇之牙", "输出-法术", dmg="mag", share=1.0, carry=8, tank=0, sup=1, stats={"法术攻击力": "+35%"},
     triggers=[("造成法伤", "造成法术伤害时添加梦魇印记")], effects=[("法攻", "法术攻击力+35%"), ("亡语爆炸", "穿戴者法术攻击力×2的法术伤害")],
     conds=[("唯一", "唯一")])
item("灵泉法杖", "输出-通用", carry=7, tank=2, sup=0, trig="auto", rng="melee_better", ctl=True,
     triggers=[("闪现后普攻", "每次闪现后普攻附带一次重击")],
     effects=[("赋予闪现", "获得闪现"), ("重击", "造成300%伤害"), ("控制·眩晕", "眩晕敌方1秒")], conds=[("唯一", "唯一")])
item("破妄之刃", "输出-通用", carry=8, tank=2, sup=0, stats={"移动速度": "+60%"},
     effects=[("真伤转化", "造成伤害的50%会转化为真实伤害"), ("免减速", "免疫减速效果")], conds=[("唯一", "唯一")])
item("破阵战芒", "攻速", carry=8, tank=2, sup=0, restrict=("tier", 3),
     triggers=[("开战", "战斗开始时")], effects=[("按等级增伤+攻速", "获得等级×1%的伤害加成和攻速")],
     conds=[("限1-3阶", "仅1-3阶英雄可穿戴"), ("上限50%", "上限50%"), ("唯一", "唯一")])
item("断念", "输出-通用", carry=7, tank=4, sup=5, triggers=[("造成伤害", "造成伤害时")],
     effects=[("削最大生命", "降低敌方单位20%最大生命值上限")], conds=[("不可叠加", "不可叠加"), ("唯一", "唯一")])
item("法潮灵杖", "输出-法术", dmg="mag", share=1.0, carry=8, tank=0, sup=1, stats={"法术攻击力": "+25%"},
     triggers=[("开战", "战斗开始")], effects=[("吸邻居法攻", "会获得左右1格的英雄的100%法术攻击力")],
     conds=[("看左右邻居", "左右1格的英雄"), ("唯一", "唯一")])
item("护命灵珠", "保命", carry=6, tank=7, sup=4, triggers=[("每4秒", "每4秒获得基础生命值25%的护盾")],
     effects=[("护盾", "获得基础生命值25%的护盾"), ("有盾增伤", "拥有护盾时伤害提升25%")], conds=[("唯一", "唯一")])
item("光辉圣铠", "坦克", carry=3, tank=8, sup=4, effects=[("回血", "每秒恢复50+4%最大生命值")], conds=[("唯一", "唯一")])
item("魂祭绝刃", "攻速", carry=7, tank=0, sup=0, trig="auto", stats={"攻击速度": "+50%", "物理和法术吸血": "+30%"},
     effects=[("攻速", "攻击速度+50%"), ("吸血", "物理和法术吸血+30%"), ("开战扣血", "扣除50%最大生命值")],
     conds=[("开战扣半血", "战斗开始时，扣除50%最大生命值"), ("唯一", "唯一")])
item("灵泉项链", "功能", carry=5, tank=4, sup=7, trig="mana", triggers=[("开战", "战斗开始时")],
     effects=[("光环回蓝", "周围1格的单位每秒回复4点法力值"), ("减蓝耗", "穿戴者法力值上限-10")], conds=[("唯一", "唯一")])
item("魂灭灵冠", "坦克", carry=2, tank=7, sup=4, rng="melee_better", triggers=[("阵亡", "阵亡时")],
     effects=[("亡语真伤", "对周围1格敌人造成目标35%最大生命值的真实伤害")], conds=[("要贴脸", "周围1格敌人"), ("唯一", "唯一")])
item("怒神项链", "暴击", carry=8, tank=1, sup=1, trig="crit", stats={"暴击率": "+50%"},
     effects=[("暴击", "暴击率+50%"), ("法伤可暴击", "法术伤害可以暴击")], conds=[("唯一", "唯一")],
     note="唯一让法系吃暴击的装备，不受伤害类型限制")
item("驱雷", "功能", carry=3, tank=3, sup=3, triggers=[("每2秒", "每2秒召唤2道雷电")],
     effects=[("敌我不分", "轰击敌我双方任意2名单位"), ("司空震联动", "若命中司空震则改为使其快速攻击4次")],
     conds=[("会打友军", "命中友军不触发吸血"), ("唯一", "唯一")], note="雷电会打自己人；司空震在场时给别人穿")
item("怒莲之刺", "攻速", carry=8, tank=0, sup=0, trig="auto", stats={"攻击速度": "+100%"},
     effects=[("攻速", "提升100%攻速"), ("转移伤害", "受到的全部伤害转移给穿戴银莲庇佑的单位")],
     conds=[("要配银莲庇佑", "穿戴银莲庇佑的单位"), ("唯一", "唯一")])
item("银莲庇佑", "坦克", carry=1, tank=7, sup=3, stats={"物理及法术防御力": "+10"},
     effects=[("双抗", "提升10点物理及法术防御力"), ("替人承伤", "替穿戴怒莲之刺的英雄承担伤害")], conds=[("要配怒莲之刺", "穿戴怒莲之刺的英雄"), ("唯一", "唯一")])
item("机器头", "输出-通用", carry=7, tank=2, sup=2, trig="anti_auto", stats={"攻击速度": "固定0.4", "伤害加成": "+60%"},
     triggers=[("开战", "战斗开始")], effects=[("增伤", "获得60%伤害加成"), ("免负面", "免疫负面效果")],
     conds=[("攻速锁0.4", "英雄固定0.4的攻击速度"), ("唯一", "唯一")], note="攻速被锁死：靠普攻的英雄是负收益")
item("阿科米亚之子", "功能", carry=5, tank=6, sup=7, trig="mana",
     effects=[("参与核心分配", "参与阿科米亚核心的等级分配"), ("光环回蓝回血", "每秒恢复4法力值与2%最大生命值")], conds=[("唯一", "唯一")])
item("磐石之心", "保命", carry=5, tank=7, sup=5, effects=[("单次伤害封顶", "每次受到伤害不会超过15%最大生命值")], conds=[("唯一", "唯一")])
item("机关匣", "功能", carry=4, tank=6, sup=6, restrict=("faction", "逐鹿"), triggers=[("开战", "战斗开始时")],
     effects=[("召唤", "召唤2个拥有你等级最高英雄一半等级的“故障机器人”")], conds=[("限逐鹿", "可穿戴给【逐鹿】英雄"), ("唯一", "唯一")])
item("星源装甲", "专属", carry=6, tank=5, sup=3, restrict=("faction", "日落海"), stats={"物理和法术吸血": "+20%"},
     triggers=[("放技能", "释放技能后的2秒内免疫控制")], effects=[("吸血", "物理和法术吸血+20%"), ("短暂免控", "免疫控制")],
     conds=[("日落海专属", "【日落海】专属装备")])
item("星源狩枪", "专属", carry=9, tank=1, sup=2, trig="skill", restrict=("faction", "日落海"),
     stats={"物理和法术攻击力": "+20%", "法力值上限": "-20"}, effects=[("双攻", "物理和法术攻击力+20%"), ("技能斩杀", "技能会直接击败生命值10%以下的敌人")],
     conds=[("日落海专属", "【日落海】专属装备")])
item("星源星徽", "专属", carry=0, tank=3, sup=3, restrict=("faction", "日落海"),
     effects=[("不吃核心等级", "该英雄不会被阿科米亚核心分配等级")], conds=[("日落海专属", "【日落海】专属装备")],
     note="作用是把核心等级让给别人：主C 绝不能穿")
item("星源合金盾", "专属", carry=4, tank=9, sup=5, restrict=("faction", "日落海"), stats={"物理和法术防御力": "+20", "受到伤害": "-20%"},
     effects=[("双抗", "物理和法术防御力+20"), ("减伤", "受到的伤害-20%")], conds=[("日落海专属", "【日落海】专属装备")])
item("月影之力", "专属", carry=2, tank=4, sup=2, restrict=("faction", "日落海"), stats={"生命值上限": "+15%"},
     effects=[("加血", "生命值上限+15%"), ("露娜专属", "将释放1次【月下无限连】")], conds=[("仅露娜生效", "仅露娜穿戴时生效"), ("日落海专属", "【日落海】专属装备")])
item("三分旗帜", "功能", carry=6, tank=3, sup=3, restrict=("faction", "三分之地"),
     effects=[("伤害转治疗", "将造成伤害的30%转化为治疗量，平摊给所有三分英雄")], conds=[("限三分", "仅三分英雄穿戴")])
# ---- 2026-10-08 版本新增（10-09 补录漏掉的装备牌）----
item("掠魂爪", "输出-物理", dmg="phys", share=0.8, carry=8, tank=2, sup=0, trig="auto",
     stats={"物理和法术吸血": "+50%"}, effects=[("普攻追伤", "普通攻击额外造成目标8%当前生命值的物理伤害")],
     conds=[("唯一", "唯一")], note="追伤挂在普攻上：高攻速物理主C 最赚")
item("永冻霜羽", "输出-法术", dmg="mag", share=0.8, carry=8, tank=2, sup=3, ctl=True,
     stats={"法术攻击力": "+100"}, effects=[("冻结", "冻结目标2秒")], conds=[("唯一", "唯一")])
item("流云之刺", "攻速", dmg="phys", share=0.7, carry=8, tank=1, sup=0, trig="auto",
     stats={"攻击速度": "+30%"}, triggers=[("每3次普攻", "每对目标进行3次普通攻击")],
     effects=[("召唤灵体", "召唤云中君灵体")], conds=[("唯一", "唯一")])
item("翼苍穹盾", "防御", carry=2, tank=8, sup=3,
     stats={"物理和法术防御力": "+10，敌方每阵亡1个单位再+10（上限100）"},
     effects=[("战斗内成长", "敌方每阵亡1个单位")], conds=[("唯一", "唯一")])
item("苍穹王权", "保命", dmg="mag", carry=6, tank=6, sup=3,
     triggers=[("受到致命伤害", "受到致命伤害时")],
     effects=[("免死", "免疫此次伤害，持续1秒"), ("反击", "对全部敌人造成100+其已损失30%生命值的法术伤害")],
     conds=[("唯一", "唯一")])
item("唤鹰骨笛", "召唤", carry=6, tank=3, sup=4, trig="auto",
     triggers=[("开战后每8秒", "战斗开始后每8秒")],
     effects=[("召唤云中君", "召唤一名等级为自身一半的云中君")],
     note="召唤物等级 = 穿戴者一半：给高等级的人穿收益最大")
item("末世", "输出-物理", carry=7, tank=3, sup=0, trig="auto", rng="melee_better",
     triggers=[("普攻", "普通攻击会减少目标2%最大生命值并恢复等量生命值")], effects=[("百分比削血+吸血", "减少目标2%最大生命值并恢复等量生命值")],
     conds=[("近战翻倍", "穿戴者为近战英雄时，效果翻倍"), ("唯一", "唯一")])

BASE = list(RECIPES)  # 8 件基础装备
for b in BASE:
    item(b, "基础", note="拖到英雄身上弹四选一：%s" % "/".join(RECIPES[b]))

missing = set(EQUIPS) - set(I)
extra = set(I) - set(EQUIPS)
assert not missing and not extra, (missing, extra)
for n, d in I.items():   # 引文必须真在原文里
    txt = EQUIPS[n]["skill"]
    for k in ("triggers", "conds", "effects"):
        for tag, q in d[k]:
            assert q in txt, (n, tag, q)

# ---------------------------------------------------------------------------------------------
# 2. 英雄属性
# ---------------------------------------------------------------------------------------------

def htext(c):
    s = (c.get("skill") or "") + "\n" + (c.get("awaken") or "")
    for sk in c.get("skills") or []:
        s += "\n" + sk.get("desc", "")
        for e in sk.get("enhance") or []:
            s += "\n" + e.get("desc", "")
    return s


DMG_OVERRIDE = {   # (取值, 理由)
    "亚连": ("mixed", "被动：赤影=法术、守心=物理、合击=真伤，且普攻和技能都能暴击"),
    "艾琳": ("mixed", "普攻吃物攻(32)、技能是法术伤害；40级起普攻附带法伤"),
    "露娜": ("mag", "突击全是法术伤害（物攻28只是普攻）"),
    "大司命": ("phys", "普攻连接与技能都是物理伤害，主要靠物理普攻"),
    "曹操": ("phys", "提升基础攻击力，普攻剑气；物攻40/法攻0"),
    "海月": ("mag", "法球按法攻；牺牲按生命值打真伤"),
    "刘邦": ("mag", "印记与横扫追伤是法术，'每拥有15点法术攻击力提升20%攻速'"),
}
AUTO_OVERRIDE = {  # 普攻依赖度 0~1（1=几乎全靠普攻）
    "狄仁杰": 0.9, "艾琳": 0.8, "公孙离": 0.7, "虞姬": 0.7, "苍": 0.7, "莱西奥": 0.7, "大司命": 0.7, "司空震": 0.6,
    "敖隐": 0.6, "韩信": 0.6, "裴擒虎": 0.6, "百里玄策": 0.5, "云中君": 0.5, "吕布": 0.5, "李元芳": 0.5,
    "孙尚香": 0.4, "阿轲": 0.4, "花木兰": 0.35, "马可波罗": 0.3, "镜": 0.3, "马超": 0.25, "李白": 0.25,
    "孙悟空": 0.2, "百里守约": 0.15, "曹操": 0.6, "刘邦": 0.5, "铠": 0.5,
}
ROLE_OVERRIDE = {
    "海月": ("tankcarry", "卡面后排坦克：护盾转生命、牺牲按生命打真伤；官方装备全是肉装"),
    "鲁班大师": ("tank", "前排拉人控制，输出极低"),
    "莱西奥": ("sub_carry", "射手，烟雾是团队增益但本体靠普攻输出"),
    "大司命": ("sub_carry", "前排吸血型输出（标签写坦克）"),
    "周瑜": ("sub_carry", "法术范围输出 + 自回复，站前排"),
    "杨玉环": ("support", "治疗为主，按法攻缩放"),
    "李元芳": ("sub_carry", "1 阶射手，古币产出方"),
}
SPLIT = re.compile(r"[。；\n]")


def pieces(c):
    """(来源, 等级门槛, 文本) 列表"""
    out = [("卡面", 0, c.get("skill") or "")]
    if c.get("awaken"):
        out.append(("觉醒卡面", 0, c["awaken"]))
    for sk in c.get("skills") or []:
        out.append(("技能", 0, sk.get("desc", "")))
        for e in sk.get("enhance") or []:
            out.append(("强化%d级" % e["level"], e["level"], e["desc"]))
    return out


HOOK_RULES = [   # (tag, 正则, 在哪段文字里找) —— 英雄身上和装备有关的钩子，全部从卡面推出
    ("ally_control_feeds", r"己方施加控制[^。]*(法力|暴击)", "all"),
    ("crit_any", r"技能伤害可以暴击|伤害可以暴击", "all"),
    ("mag_to_as", r"每拥有\d+点法术攻击力提升\d+%攻速", "all"),
    ("shield_to_hp", r"护盾[^。]{0,12}转化[^。]{0,12}最大生命值", "all"),
    ("heal_reduced", r"生命恢复效果大幅降低", "all"),
    ("hp_scaling", r"(自身|NAME)\d*%?[^，。]{0,4}最大生命值的(物理|法术)伤害|永久提升\d+点最大生命值", "skill"),
    ("as_scaling", r"每\d+次普攻|普攻时有|每次普攻为|继承[^，。]{0,3}攻击速度|提供\d+%攻击速度|基础攻击速度|攻击速度\+\d|获得自身等级×1%的攻击速度|普通攻击会持续连接|普攻能产生", "all"),
    ("no_mana_after_cast", r"无法恢复法力值(?![^，。]*敌)|期间不恢复法力值", "skill"),
    ("lifesteal", r"(获得|提升)\d+%(物理|法术)?吸血|额外获得\d+%物理吸血", "skill"),
    ("share_first_equip", r"首件普通装备分享", "all"),
    ("heal_scales_mag", r"恢复[^。]{0,24}法术攻击力的生命值", "skill"),
]


def derive_hooks(name, c):
    out = []
    for tag, pat, where in HOOK_RULES:
        for src, lv, txt in pieces(c):
            if where == "skill" and src.startswith("卡面") or where == "skill" and src.startswith("觉醒"):
                continue
            m = None
            for s in SPLIT.split(txt):
                if re.search(pat.replace("NAME", name), s):
                    m = s.strip()
                    break
            if m:
                if tag == "hp_scaling" and re.search(r"造成其|目标[^，。]{0,4}最大生命", m):
                    continue
                out.append((tag, m))
                break
    p = c["properties"]
    if (p.get("energy") or 0) == 0:
        out.append(("no_active_skill", "法力值上限 %s（没有主动技能）" % p.get("energy")))
    return out


def dmg_of(name, c):
    if name in DMG_OVERRIDE:
        return DMG_OVERRIDE[name]
    t = htext(c)
    cp = len(re.findall(r"物理伤害|物理攻击力", t))
    cm = len(re.findall(r"法术伤害|法术攻击力", t))
    p = c["properties"].get("phyAttack") or 0
    m = c["properties"].get("magAttack") or 0
    if cp and not cm:
        return "phys", f"技能只写物理（{cp} 处），物攻{p}/法攻{m}"
    if cm and not cp:
        return "mag", f"技能只写法术（{cm} 处），物攻{p}/法攻{m}"
    if cp and cm:
        if abs(cp - cm) <= 1:
            return "mixed", f"技能物理 {cp} 处 / 法术 {cm} 处"
        return ("phys" if cp > cm else "mag"), f"技能物理 {cp} 处 / 法术 {cm} 处"
    if p > m:
        return "phys", f"技能无伤害字样，物攻{p}>法攻{m}"
    return "mag", f"技能无伤害字样，法攻{m}≥物攻{p}"


def auto_rel(name, c):
    if name in AUTO_OVERRIDE:
        return AUTO_OVERRIDE[name]
    sp = c["properties"].get("attackSpeed") or 6000
    base = {10000: 0.5, 8000: 0.35}.get(sp, 0.15)
    t = htext(c)
    base += min(0.3, 0.15 * len(re.findall(r"普攻|普通攻击|攻击速度|攻速", t)))
    src = (ROLES.get(name) or {}).get("damage_shape", {}).get("source")
    base += {"auto": 0.3, "mixed": 0.15}.get(src, 0)
    return round(min(0.9, base), 2)


def role_of(name):
    if name in ROLE_OVERRIDE:
        return ROLE_OVERRIDE[name]
    r = ROLES[name]
    roles = r["combat_role"]
    cv = r["carry_value"]
    prim = roles[0]
    if cv >= 6 and prim not in ("tank", "support"):
        return "carry", f"hero_roles carry_value={cv}，{prim}"
    if prim == "tank" or ("tank" in roles and cv <= 3):
        return "tank", f"hero_roles {roles}，carry_value={cv}"
    if prim == "support" or (prim in ("controller", "utility") and cv <= 3):
        return "support", f"hero_roles {roles}，carry_value={cv}"
    return "sub_carry", f"hero_roles {roles}，carry_value={cv}"


HERO = {}
for n, c in sorted(HEROES.items(), key=lambda x: (x[1]["quality"], x[1]["id"])):
    p = c["properties"]
    dm, dm_why = dmg_of(n, c)
    ar = auto_rel(n, c)
    role, role_why = role_of(n)
    melee = (p.get("attackDistance") or 0) <= 1
    energy = p.get("energy") or 0
    relies = []
    relies.append("普攻" if ar >= 0.5 else ("普攻+技能" if ar >= 0.3 else "技能"))
    if energy == 0:
        relies.append("无主动技能（法力上限0）")
    elif energy <= 45:
        relies.append(f"法力循环快（上限{energy}）")
    elif energy >= 70:
        relies.append(f"法力循环慢（上限{energy}）")
    hooks = derive_hooks(n, c)
    if role in ("tank", "support") and energy >= 90 and (p.get("initEnergy") or 0) < energy:
        hooks.append(("slow_mana_tank", "法力上限 %d、开局 %s：控制型前排放技能慢" % (energy, p.get("initEnergy"))))
    crit_ok = dm in ("phys",) or any(h[0] == "crit_any" for h in hooks)
    HERO[n] = {
        "dmg": dm, "dmg_why": dm_why,
        "range": "melee" if melee else "ranged", "attackDistance": p.get("attackDistance"),
        "auto_reliance": ar, "relies_on": relies,
        "mana": {"init": p.get("initEnergy"), "max": energy},
        "crit": "吃暴击" if crit_ok else ("法术伤害默认不能暴击" if dm == "mag" else "部分（混合）"),
        "base_crit_rate": (p.get("criticalRate") or 0) / 100,
        "passive_hooks": [{"tag": a, "quote": b} for a, b in hooks],
        "role": role, "role_why": role_why,
        "faction": c["faction"], "quality": c["quality"],
        "props": {k: p.get(k) for k in ("phyAttack", "magAttack", "HP", "attackSpeed", "attackDistance")},
    }

# ---------------------------------------------------------------------------------------------
# 2b. 队友依赖 / 能提供的触发 / 靠什么变强 / 能否当单C —— 全部从卡面文字用通用规则推出，规则里不出现英雄名
# ---------------------------------------------------------------------------------------------

# 队友触发的识别（触发句式 → 类型, 需要场上几个提供者, 是否任何阵容都自然满足）
DEP_PATTERNS = [
    (r"己方施加控制", "control", "施加控制", 2, False),
    (r"其他英雄释放技能|你的英雄(每)?释放.{0,3}技能", "ally_cast", "队友释放技能", 1, True),
    (r"拥有牺牲效果的英雄阵亡|触发牺牲效果", "sacrifice", "牺牲英雄阵亡", 2, False),
    (r"每个图腾|每有1个图腾|所有图腾", "totem", "图腾", 2, False),
    (r"已觉醒英雄", "awakened", "已觉醒英雄", 2, False),
    (r"触发.{0,3}登场效果", "entry", "登场效果", 2, False),
    (r"触发整备", "prepare", "整备效果", 1, False),
    (r"触发.{0,3}开团效果|开团效果时", "open", "开团效果", 1, False),
    (r"其他英雄获得临时等级", "temp_level", "临时等级来源", 1, False),
    (r"友方单位护盾被击破|友军护盾被击破", "shield_break", "护盾", 1, False),
    (r"闪现英雄.{0,12}击败", "flash", "其他闪现英雄", 1, False),
    (r"召唤单位阵亡", "summon_death", "召唤物", 1, False),
    (r"其他英雄等级数", "ally_level", "高等级队友", 1, False),
]
PLAYER_PATTERNS = [(r"使用.{0,6}(效果牌|古币|战术牌|英雄牌)|对.{0,6}使用", "玩家用牌"), (r"出售", "出售"), (r"合成", "合成")]
OUTPUT_WORDS = r"法力|暴击|攻击速度|攻速|伤害|攻击力"
OTHER_BENEFICIARY = r"随机|其他|己方英雄|友方|友军|该单位|该英雄|核心|图腾|全体|等级最低|等级最高|距离最近|同阶"

# 能提供的触发（给队友的 team_dependency 当"供给"）
PROVIDE_PATTERNS = [
    ("control", "施加控制", r"眩晕|击飞|冻结|变成小猪|嘲讽|压制|沉默|浮空|击退|拽离|拉到身边|困入|吹起"),
    ("shield", "护盾", r"(获得|提供|产生)[^，。]{0,16}护盾|护盾值"),
    ("heal", "治疗", r"恢复[^，。]{0,24}生命值"),
    ("sacrifice", "牺牲", r"牺牲："),
    ("totem", "图腾", r"召唤[^，。]{0,4}图腾"),
    ("summon", "召唤物", r"召唤"),
    ("temp_level", "临时等级", r"临时等级[+＋]|获得[^，。]{0,10}临时等级"),
    ("level", "永久等级", r"等级(永久)?[+＋]\d"),
    ("entry", "登场效果", r"登场："),
    ("prepare", "整备效果", r"整备："),
    ("open", "开团效果", r"开团："),
    ("flash", "闪现", r"^闪现"),
    ("effect_card", "产出卡牌", r"获得\d+张"),
    ("mana_ally", "给队友回蓝", r"(友军|单位)[^，。]{0,8}(法力值|回复\d+点法力)"),
]
CONTROL_EXCLUDE = r"免控|免疫控制|施加控制|控制效果时"


def team_info(name, c):
    deps, player = [], []
    card_sents = [s for s in SPLIT.split((c.get("skill") or "")) if s.strip()]
    card_dep_sents = 0
    for src, lv, txt in pieces(c):
        for s in SPLIT.split(txt):
            s = s.strip()
            if not s:
                continue
            hit = False
            for pat, kind, label, need, universal in DEP_PATTERNS:
                m = re.search(pat, s)
                if not m:
                    continue
                effect = s[m.end():]
                self_ben = (name in effect) or ("自身" in effect) or (re.search("场上每有|每个图腾|每有1个", s) and not re.search(OTHER_BENEFICIARY, effect)) \
                    or not re.search(OTHER_BENEFICIARY + r"|英雄", effect)
                if not self_ben:
                    continue
                feeds = "output" if re.search(OUTPUT_WORDS, effect) else ("growth" if "等级" in effect else "other")
                deps.append({"trigger": label, "kind": kind, "text": s, "source": src, "feeds": feeds,
                             "needs": "任何阵容都满足（每个英雄都会放技能）" if universal else "场上提供「%s」的单位 ≥%d" % (label, need),
                             "need_count": 0 if universal else need, "universal": universal})
                hit = True
                break
            if not hit:
                for pat, label in PLAYER_PATTERNS:
                    if src.startswith("卡面") and re.search(pat, s) and re.search(name + "|自身|该英雄", s):
                        player.append({"by": label, "text": s, "source": src})
                        break
            if hit and src == "卡面":
                card_dep_sents += 1
    keyword_only = [s for s in card_sents if re.fullmatch(r"\s*(闪现|夺取|复生)[，, ]*(闪现|夺取|复生)?\s*", s)]
    card_all_dep = bool(card_sents) and card_dep_sents > 0 and card_dep_sents >= len(card_sents) - len(keyword_only)
    return deps, player, card_all_dep


def provides_of(name, c):
    out = []
    for tag, label, pat in PROVIDE_PATTERNS:
        for src, lv, txt in pieces(c):
            found = None
            for s in SPLIT.split(txt):
                s2 = s.strip()
                if not s2:
                    continue
                if tag == "control" and re.search(CONTROL_EXCLUDE, s2) and not re.search(r"眩晕|击飞|冻结|嘲讽|压制|沉默|浮空|击退", s2):
                    continue
                if tag == "summon" and "图腾" in s2:
                    continue
                if tag == "shield" and "被击破" in s2 and not re.search(r"获得[^，。]{0,16}护盾", s2):
                    continue
                if re.search(pat, s2):
                    found = s2
                    break
            if found:
                t2, l2 = tag, label
                if tag in ("heal", "shield") and not re.search(r"友军|友方|己方|队友|周围1格的己方|全体|其他英雄|生命值最低", found):
                    t2, l2 = "self_" + tag, "自身" + label     # 只给自己的，不算给队伍提供
                out.append({"tag": t2, "label": l2, "text": found, "source": src, "min_level": lv})
                break
    return out


COEF = re.compile(r"(\d+)%(?:的)?(物理|法术)攻击力")


def scales_of(name, c, deps, hero):
    sk = " ".join(s.get("desc", "") for s in c.get("skills") or [])
    coef = sum(int(x) for x, _ in COEF.findall(sk))
    dep_stats = set()
    for d in deps:
        if d["feeds"] == "output" and not d["universal"]:
            for w in ("暴击", "法力", "攻击速度", "攻速"):
                if w in d["text"]:
                    dep_stats.add(w)
    base_crit = (c["properties"].get("criticalRate") or 0) > 0
    enh = []
    for s in c.get("skills") or []:
        for e in s.get("enhance") or []:
            t = e["desc"]
            out = bool(re.search(r"伤害|数量|次数|段|攻击速度|攻速|子弹|飞剑|导弹|范围|连续|弹射|狼魂|仆从|飞叶|炸弹|冲击波|剑气", t))
            cond = None
            if "暴击" in t and not base_crit and ("暴击" in dep_stats or not re.search("必定暴击", t)):
                cond = "要暴击（自身基础暴击率 0%%%s）" % ("，暴击只来自队友触发" if "暴击" in dep_stats else "")
            enh.append({"level": e["level"], "text": t, "output": out, "conditional_on": cond})
    stack = [s for s in SPLIT.split(sk) if re.search(r"叠加|最多\d+层|层数|永久|每击败|每拥有", s)]
    self_scale = sum(1 for e in enh if e["output"] and not e["conditional_on"]) + (1 if coef >= 300 else 0) + (1 if stack else 0)
    return {
        "level_enhance": enh,
        "attack_coef_pct": coef,
        "equipment": "技能按攻击力系数合计 %d%%：%s" % (coef, "攻击类装备收益高" if coef >= 300 else ("一般" if coef >= 150 else "低（伤害多为固定值/百分比生命）")),
        "self_stacking": stack,
        "team_triggers": [d["text"] for d in deps],
        "self_scale": self_scale,
    }


TEAM_RULES = [
    "S1 定位不是输出（坦克/辅助）→ solo_carry_ok=false（不当主C，与队友依赖无关）。",
    "S2 team_dependency 只收'触发来自队友、收益落在自己身上'的句子（受益方含本人名字/自身，或句子里没有别的受益方）；'该英雄/随机3名'这种给别人的是他 provides 的引擎，不算依赖。",
    "S3 队友放技能（任何阵容都有）记为 universal，不计入依赖权重。",
    "S4 依赖权重 dep_weight = Σ(喂输出[法力/暴击/攻速/伤害]的非 universal 依赖 ×1，喂成长[等级]的 ×0.5) + (卡面句子全部是队友依赖 +1) + (某个等级强化要靠依赖喂的属性才生效，如要暴击而暴击只来自队友 +1)。",
    "S5 自身成长 self_scale = 不附带条件的输出型等级强化条数 + (技能攻击力系数合计 ≥300% +1) + (技能自带叠层/永久成长 +1)。阿离给主C 的是等级和装备，这两项正对应 self_scale。",
    "S6 solo_carry_ok=false 当且仅当 S1，或 dep_weight≥2 且 self_scale≤2，或 dep_weight≥1 且 self_scale≤1；否则 true。",
    "S7 needs = 场上提供该触发的单位估计下限（控制/牺牲/图腾/觉醒/登场 ≥2，其余 ≥1）；provides 给每个英雄列出能提供的触发，主线程可用 provides 数当前阵容能否喂饱某条依赖。",
]

for n, c in HEROES.items():
    h = HERO[n]
    deps, player, card_all_dep = team_info(n, c)
    prov = provides_of(n, c)
    sc = scales_of(n, c, deps, h)
    w = 0.0
    parts = []
    seen_kind = set()
    for d in deps:
        if d["kind"] in seen_kind:      # 卡面和觉醒卡面是同一条依赖的两个强度，只算一次
            continue
        seen_kind.add(d["kind"])
        if d["universal"]:
            continue
        if d["feeds"] == "output":
            w += 1
            parts.append("队友「%s」喂输出 +1" % d["trigger"])
        elif d["feeds"] == "growth":
            w += 0.5
            parts.append("队友「%s」喂成长 +0.5" % d["trigger"])
    if card_all_dep and deps and not all(d["universal"] for d in deps):
        w += 1
        parts.append("卡面效果全靠队友触发 +1")
    if any(e["conditional_on"] and "队友" in e["conditional_on"] for e in sc["level_enhance"]):
        w += 1
        parts.append("等级强化要暴击、暴击只来自队友 +1")
    ss = sc["self_scale"]
    if h["role"] in ("tank", "support"):
        ok, why = False, "S1 定位是%s，不当主C" % {"tank": "坦克", "support": "辅助"}[h["role"]]
    elif (w >= 2 and ss <= 2) or (w >= 1 and ss <= 1):
        ok, why = False, "S6 依赖权重 %.1f（%s），自身成长 %d → 主要增益来自队友触发，单C 资源（等级/装备）喂不起来" % (w, "；".join(parts), ss)
    else:
        ok, why = True, "S6 依赖权重 %.1f%s，自身成长 %d → 吃等级和装备就能变强" % (w, "（" + "；".join(parts) + "）" if parts else "", ss)
    h["team_dependency"] = deps
    h["player_actions"] = player
    h["provides"] = prov
    h["scales_with"] = sc
    h["dep_weight"] = w
    h["solo_carry_ok"] = ok
    h["solo_carry_why"] = why

# ---------------------------------------------------------------------------------------------
# 3. 官方先验：data/lineups.json 里每个 (英雄, 装备) 被推荐的次数
# ---------------------------------------------------------------------------------------------
OFF = Counter((h["name"], e) for l in LINEUPS for h in l["heroes"] for e in h.get("equips", []))


def prior(c):
    return 1.0 if c >= 10 else (0.5 if c >= 3 else 0.0)


# ---------------------------------------------------------------------------------------------
# 4. 适配分
# ---------------------------------------------------------------------------------------------
W = {  # (输出, 前排, 辅助) 权重
    "carry": (1.0, 0.35, 0.15), "sub_carry": (0.8, 0.55, 0.35), "tank": (0.25, 1.0, 0.45),
    "support": (0.3, 0.55, 1.0), "tankcarry": (0.6, 1.0, 0.3),
}
CRIT_ITEMS = {"无尽巨剑", "凝渊", "青莲剑谱", "怒神项链"}
AS_ITEMS = {"影牙", "铸梦·逐风", "金色圣剑", "魂祭绝刃", "怒莲之刺", "破阵战芒", "双剑·雄"}
MANA_ITEMS = {"法源之杯", "破茧之衣", "聚灵水晶", "灵泉项链", "阿科米亚之子", "贤者之书", "学识宝石", "星源狩枪"}
HP_ITEMS = {"不死鸟之眼", "盛华", "生命魔典", "光辉圣铠", "月影之力", "鼓舞之戒"}
SHIELD_ITEMS = {"御灵之纱", "护命灵珠", "魔女斗篷", "不死鸟之眼", "炽热支配者"}
HEAL_ITEMS = {"光辉圣铠", "生命魔典", "聚灵水晶", "征服之剑", "魂祭绝刃", "星源装甲"}
LIFESTEAL_ITEMS = {"征服之剑", "魂祭绝刃", "噬神之书", "星源装甲", "末世"}
EXCLUSIVE_FOR, LINKED_TARGET = {}, {}
for _n, _c in EQUIPS.items():
    _m = re.search(r"若穿戴者为(\S+?)，[^。]*|[^。（]*（仅(\S+?)穿戴时生效[^）]*）", _c["skill"])
    if _m:
        EXCLUSIVE_FOR[_n] = ((_m.group(1) or _m.group(2)), _m.group(0))
    _m = re.search(r"若命中(\S+?)则[^。]*", _c["skill"])
    if _m:
        LINKED_TARGET[_n] = (_m.group(1), _m.group(0))


def restricted(hn, it):
    h, r = HERO[hn], I[it]["restrict"]
    if not r:
        return None
    if r[0] == "ranged" and h["range"] == "melee":
        return "限定远程英雄，%s 是近战（attackDistance=%s）" % (hn, h["attackDistance"])
    if r[0] == "melee" and h["range"] == "ranged":
        return "限定近战英雄，%s 是远程（attackDistance=%s）" % (hn, h["attackDistance"])
    if r[0] == "faction" and HEROES[hn]["faction"] != r[1]:
        return "限【%s】，%s 是【%s】" % (r[1], hn, HEROES[hn]["faction"])
    if r[0] == "tier" and HEROES[hn]["quality"] > r[1]:
        return "仅1-%d阶可穿，%s 是 %d 阶" % (r[1], hn, HEROES[hn]["quality"])
    return None


def score_one(hn, it):
    """返回 (score, why_list, hard)；不含基础装备。"""
    d, h = I[it], HERO[hn]
    why = []
    rs = restricted(hn, it)
    if rs:
        return -1.0, [rs], True
    ow, dw, sw = W[h["role"]]
    if h["role"] in ("carry", "sub_carry") and h["range"] == "melee":
        dw = min(1.0, dw + 0.15)          # 近战输出要进场挨打，防装价值更高
    hooks = {x["tag"] for x in h["passive_hooks"]}
    # 专属效果
    ex = EXCLUSIVE_FOR.get(it)
    if ex and ex[0] == hn:
        return 10.0, ["装备原文点名该英雄：'%s'" % ex[1]], False
    if it == "破茧之衣" and "share_first_equip" in hooks:
        return 10.0, ["瑶开团把首件普通装备分给3名友军：开场眩晕+15法力 ×4"], False
    # ---- 输出分量 ----
    carry = float(d["carry"])
    hard = False
    m = 1.0
    if d["dmg"]:
        if h["dmg"] == d["dmg"]:
            m = 1.0
        elif h["dmg"] == "mixed":
            m = 0.5
        else:
            m = 0.0 if d["dmg"] == "mag" else 0.15   # 法系英雄的普攻吃物攻(24)，物理装还剩一点点
        if m < 1:
            lost = d["share"] * (1 - m)
            why.append("%s装（%d%%价值靠%s数值）× %s英雄 → 输出价值 ×%.2f" % (
                {"phys": "物理", "mag": "法术"}[d["dmg"]], d["share"] * 100, {"phys": "物攻", "mag": "法攻"}[d["dmg"]],
                {"phys": "物理", "mag": "法术", "mixed": "混合"}[h["dmg"]], 1 - lost))
            carry *= (1 - lost)
            if m == 0 and d["share"] >= 0.5:
                hard = True
    if d["trig"] == "crit" and it != "怒神项链":
        if h["dmg"] == "mag" and "crit_any" not in hooks:
            carry *= 0.1
            hard = True
            why.append("暴击装：法术伤害不能暴击")
    ar = h["auto_reliance"]
    if d["trig"] == "auto":
        f = 0.4 + 0.6 * min(1.0, ar / 0.6)
        if f < 0.99:
            why.append("普攻触发，%s 普攻依赖 %.2f → ×%.2f" % (hn, ar, f))
        carry *= f
    elif d["trig"] == "anti_auto":
        f = 1.0 - 0.8 * min(1.0, ar / 0.6)
        if f < 0.99:
            why.append("攻速锁 0.4，%s 普攻依赖 %.2f → ×%.2f" % (hn, ar, f))
        carry *= f
    elif d["trig"] in ("skill", "mana"):
        if "no_active_skill" in hooks:
            carry *= 0.0
            why.append("没有主动技能（法力上限 0），技能/法力类无效")
        else:
            sr = 1 - ar
            f = 0.4 + 0.6 * min(1.0, sr / 0.6)
            if (h["mana"]["max"] or 0) >= 70:
                f = min(1.1, f + 0.1)
            if "no_mana_after_cast" in hooks and d["trig"] == "mana":
                f *= 0.5
                why.append("神巫状态无法回蓝 → 法力装减半")
            if f < 0.99:
                why.append("技能/法力类，%s 技能依赖 %.2f → ×%.2f" % (hn, sr, f))
            carry *= f
    rng = d["rng"]
    eff_share = 1 - (d["share"] if d["dmg"] else 0)   # 与伤害类型无关的效果部分
    if rng == "ranged_better" and h["range"] == "melee":
        carry -= d["carry"] * eff_share * 0.5
        why.append("远程条件：近战穿效果减半")
    if rng == "melee_better" and h["range"] == "ranged":
        carry *= 0.5
        why.append("要贴脸/近战加成：远程只拿一半")
    if rng == "isolated" and h["range"] == "ranged":
        carry *= 0.5
        why.append("身边无友军才增伤：后排身边常站辅助")
    _rp = re.search(r"技能替换为(\S+?)。", EQUIPS[it]["skill"])
    _own = [x.get("name") for x in HEROES[hn].get("skills") or []]
    if _rp and _rp.group(1) in _own:
        why.append("替换成的技能「%s」就是自己的技能，不吃亏" % _rp.group(1))
    elif _rp and (h["scales_with"]["attack_coef_pct"] >= 300 or h["role"] == "carry"):
        carry *= 0.4
        why.append("技能被替换为青莲剑歌，%s 自身技能是核心" % hn)
    carry = max(0.0, carry)
    # ---- 前排/辅助分量 ----
    tank = float(d["tank"])
    sup = float(d["sup"])
    if rng == "melee_better" and h["range"] == "ranged":
        tank *= 0.5
    if d["trig"] in ("skill", "mana") and "no_active_skill" in hooks:
        tank *= 0.3
        sup *= 0.3
    if "heal_scales_mag" in hooks and d["dmg"] == "mag":
        s2 = d["carry"] * d["share"] * 0.7
        if s2 > sup:
            sup = s2
            why.append("治疗/召唤按法攻缩放 → 法攻装对辅助有用")
    if "share_first_equip" in hooks and it not in BASE:
        gen = d["carry"] if not d["dmg"] else d["carry"] * (1 - d["share"])
        s2 = max(sup, 0.75 * max(gen, d["tank"]))
        if s2 > sup:
            sup = s2
            why.append("瑶分享首件普通装备给3名友军")
    if "slow_mana_tank" in hooks and it in MANA_ITEMS:
        tank += 1.5
        why.append("法力上限高的控制前排：回蓝装 +1.5")
    comp = {"输出": ow * carry, "前排": dw * tank, "辅助": sw * sup}
    best = max(comp, key=comp.get)
    s = comp[best]
    why.insert(0, "%s位 → 取%s分量 %.1f（输出%.1f×%.2f / 前排%.1f×%.2f / 辅助%.1f×%.2f）" % (
        {"carry": "主C", "sub_carry": "副C", "tank": "坦克", "support": "辅助", "tankcarry": "坦克型输出"}[h["role"]],
        best, s, carry, ow, tank, dw, sup, sw))
    # ---- 钩子 ----
    bonus = 0.0
    if "ally_control_feeds" in hooks:
        if d["ctl"]:
            bonus += 1.0
            why.append("马超被动：己方施加控制→+15法力+5%暴击 (+1)")
        if it in CRIT_ITEMS and not hard:
            bonus += 0.5
            why.append("马超被动叠暴击 (+0.5)")
    if "as_scaling" in hooks and it in AS_ITEMS:
        bonus += 1.0
        why.append("被动吃攻速 (+1)")
    if "mag_to_as" in hooks and d["dmg"] == "mag":
        bonus += 2.0
        why.append("刘邦：每15法攻+20%攻速 (+2)")
    if "shield_to_hp" in hooks and it in SHIELD_ITEMS:
        bonus += 1.0
        why.append("海月：护盾转最大生命 (+1)")
    if "heal_reduced" in hooks and it in HEAL_ITEMS:
        bonus -= 2.0
        why.append("海月：生命恢复大幅降低 (-2)")
    if "hp_scaling" in hooks and it in HP_ITEMS:
        bonus += 1.0
        why.append("技能按最大生命缩放 (+1)")
    if "lifesteal" in hooks and it in LIFESTEAL_ITEMS:
        bonus += 1.0
        why.append("自带吸血/续航被动 (+1)")
    if d["keeper"] and h["role"] != "carry" and HEROES[hn]["quality"] <= 2:
        pen = 4.0 if HEROES[hn]["quality"] <= 1 else 1.5
        bonus -= pen
        why.append("无法拆卸的成长装：%d 阶非主C 多半中途被卖/换下 (-%.1f)" % (HEROES[hn]["quality"], pen))
    if any(q == "不吃核心等级" for q, _ in d["effects"]) and h["role"] in ("carry", "sub_carry", "tankcarry"):
        bonus -= 3
        why.append("星徽让出核心等级：输出位绝不穿")
    lk = LINKED_TARGET.get(it)
    if lk and lk[0] == hn:
        bonus -= 2
        why.append("装备联动对象是穿戴者本人：'%s'，而效果'穿戴者除外' → 联动失效" % lk[1])
    s += bonus
    # ---- 官方先验 ----
    c = OFF.get((hn, it), 0)
    pr = prior(c)
    if pr:
        s += pr
        why.append("官方阵容 %d 次 (+%.1f 先验)" % (c, pr))
    # ---- 硬错配封顶 ----
    if hard:
        cap = 4.0
        if rng == "ranged_better" and h["range"] == "melee":
            cap = 3.5
        if s > cap:
            why.append("伤害类型硬错配 → 封顶 %.1f（先验不能翻盘）" % cap)
            s = cap
    s = max(0.0, min(10.0, s))
    return round(s, 1), why, hard


FIT = {}
for hn in HERO:
    row = {}
    for it in I:
        if it in BASE:
            continue
        sc, why, hard = score_one(hn, it)
        row[it] = {"score": sc, "why": "；".join(why)}
        if hard:
            row[it]["hard_mismatch"] = True
    for b in BASE:
        cands = sorted(((row[f]["score"], f) for f in RECIPES[b]), reverse=True)
        bs, bf = cands[0]
        c = OFF.get((hn, b), 0)
        row[b] = {"score": bs, "best_forge": bf,
                  "forges": {f: row[f]["score"] for f in RECIPES[b]},
                  "why": "四选一按人挑：最佳成品「%s」%.1f（%s）" % (bf, bs, " / ".join("%s %.1f" % (f, row[f]["score"]) for f in RECIPES[b]))
                  + ("；官方直接带基础装 %d 次" % c if c else "")}
    FIT[hn] = row

# ---------------------------------------------------------------------------------------------
# 5. 输出
# ---------------------------------------------------------------------------------------------
RULES = [
    "R1 穿戴限制（限定远程/近战、限阵营【X】、日落海专属、仅1-3阶）不满足 → -1（不可穿）。",
    "R2 伤害类型：装备数值部分只对同类伤害有用。share=数值部分占比；物理英雄×法术装、法术英雄×物理装时这部分作废（法系的普攻吃物攻，物理装留 15%），混合英雄留一半；亚连（普攻和技能都能暴击）不受限。",
    "R3 暴击装（无尽巨剑/凝渊/青莲剑谱）对纯法术英雄 ×0.1：法术伤害不能暴击；怒神项链例外（法伤可暴击）。",
    "R4 触发机制：普攻触发 ×(0.4+0.6·min(1,普攻依赖/0.6))；技能/法力触发 ×(0.4+0.6·min(1,技能依赖/0.6))，法力上限≥70 再+0.1；机器头锁攻速 ×(1-0.8·min(1,普攻依赖/0.6))；法力上限 0（蔡文姬）技能/法力类归零。",
    "R5 远近条件：自然法杖'近战80点法力'→近战的效果部分减半；冥王/红莲炎甲/魂灭灵冠/末世/灵泉法杖要贴脸 → 远程 ×0.5；红魔要身边无友军 → 后排 ×0.5；青莲剑谱替换技能 → 技能是核心的英雄 ×0.4。",
    "R6 定位：每件装备有 输出/前排/辅助 三个基准分，按英雄定位权重（主C 1/0.35/0.15，副C 0.8/0.55/0.35，坦克 0.25/1/0.45，辅助 0.3/0.55/1，坦克型输出 0.6/1/0.3；近战输出前排权重 +0.15）取最大的分量 —— 坦克装给坦克、输出装给输出。",
    "R7 被动钩子（全部由 HOOK_RULES 正则从卡面推出，规则里不写英雄名）：卡面写'己方施加控制…法力/暴击'→带控制效果的装备 +1、暴击装 +0.5；被动吃攻速（每N次普攻/继承攻速/攻速+N…）→攻速装 +1；'每拥有N点法术攻击力提升攻速'→法攻装 +2；护盾转最大生命→护盾装 +1；'生命恢复效果大幅降低'→回血装 -2；'自身最大生命值的伤害/永久提升最大生命值'→加血装 +1；技能自带吸血→吸血装 +1；'恢复…法术攻击力的生命值'→辅助拿法攻装按 0.7 折算；'首件普通装备分享'→普通装备按团队价值 0.75 折算、破茧之衣 10；装备原文点名'若穿戴者为X/仅X穿戴时生效'→X=10；'若命中X则…'且穿戴者除外→X 自己穿 -2；'技能替换为Y'→Y 不是自己的技能且自己是主C ×0.4。",
    "R8 官方先验：lineups.json 中该英雄带该装备 ≥10 次 +1，≥3 次 +0.5，否则 0；只加不减。",
    "R9 硬错配封顶：伤害类型完全错配且数值占比≥50%（或纯法术英雄拿暴击装）→ 最终分 ≤4（近战再叠远程条件 ≤3.5），先验和钩子都不能越过。",
    "R10 基础装备 = 它四个成品里对该英雄分数最高的那件（四选一按人挑），并记 best_forge。",
]

items_out = {}
for n, d in I.items():
    c = EQUIPS[n]
    items_out[n] = {
        "category": d["cat"], "dmg_req": d["dmg"], "dmg_share": d["share"],
        "base": {"carry": d["carry"], "tank": d["tank"], "support": d["sup"]},
        "stats": d["stats"],
        "triggers": [{"tag": a, "quote": b} for a, b in d["triggers"]] + ([{"tag": "trigger_class", "quote": d["trig"]}] if d["trig"] else []),
        "conditions": [{"tag": a, "quote": b} for a, b in d["conds"]]
        + ([{"tag": "restrict", "quote": "/".join(map(str, d["restrict"]))}] if d["restrict"] else [])
        + ([{"tag": "range_cond", "quote": d["rng"]}] if d["rng"] else []),
        "effects": [{"tag": a, "quote": b} for a, b in d["effects"]] + ([{"tag": "control", "quote": "施加控制"}] if d["ctl"] else []),
        "non_removable": d["keeper"], "note": d["note"], "text": c["skill"],
    }
    if n in RECIPES:
        items_out[n]["forges"] = RECIPES[n]
out = {"version": "2026-10-01",
       "doc": "docs/equip-relations.md；构建脚本 build_equip_relations.py（见文档第 1 节）",
       "rules": RULES, "team_rules": TEAM_RULES, "items": items_out, "heroes": HERO, "fit": FIT}
(ROOT / "data/equip_relations.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------------------------------------
# 6. 与旧画像对比 + 文档
# ---------------------------------------------------------------------------------------------
from agent.decision import hero_traits as HT  # noqa: E402

diffs = []
for hn in HERO:
    cr = ROLES[hn]["combat_role"]
    for it in I:
        old = HT.item_fit_score(it, hn, cr)
        if old is None:
            continue
        new = FIT[hn][it]["score"]
        diffs.append((abs(new - old), hn, it, old, new, FIT[hn][it]["why"]))
diffs.sort(key=lambda x: (-x[0], x[1], x[2]))


def who(it, top=True, k=6):
    arr = sorted(((FIT[h][it]["score"], h) for h in HERO), reverse=top)
    if top:
        return "、".join("%s %.0f" % (h, s) for s, h in arr[:k] if s >= 6) or "—"
    no = [h for s, h in sorted(arr) if s == -1]
    low = [h for s, h in sorted(arr) if 0 <= s <= 2 and HERO[h]["role"] in ("carry", "sub_carry")]
    parts = []
    if no:
        parts.append("不可穿：" + ("、".join(no[:6]) + ("…等%d人" % len(no) if len(no) > 6 else "")))
    if low:
        parts.append("≤2：" + "、".join(low[:6]) + ("…" if len(low) > 6 else ""))
    return "；".join(parts) or "—"


L = []
L.append("# 装备 × 英雄关系图（2026-10-01 重建）\n")
L.append("> 起因：2026-10-01 给马超穿上了远程才能带的装备，说明原装备关系网不够清楚，需要按全部装备的卡面描述重建关系图。")
L.append("> 事故：基础装备「大棒」四选一按游戏 👍 和旧画像（`equip_profiles.json` 自然法杖 heroes_best 含马超，依据是官方阵容马超带它 19 次）给物理近战马超铸出「自然法杖」。")
L.append("> 教训：**官方次数只是弱先验；卡面硬错配（伤害类型、远/近战条件、穿戴限制）优先**。关系 > 标签。\n")
L.append("机器可读：`data/equip_relations.json`（items 82 件结构化属性 + 原文引用；heroes 85 人；fit 85×82 全覆盖，-1=不可穿）。自检：`tools/check_equip_relations.py`。")
L.append("构建脚本没放进仓库（任务限定只写这三个文件），在会话 scratchpad 的 `build_equip_relations.py`；主线程如需可原样拷到 `tools/`（用法 `python build_equip_relations.py <项目根>`）。\n")
L.append("## 1. 方法\n")
L.append("1. **装备**：逐件读 `cards.json` 的 `skill` 原文，手工结构化为 数值 / 触发 / 条件 / 效果，每条附原文引文（脚本校验引文确实在原文里）；"
         "另给三个基准分：给理想匹配的输出位 / 前排 / 辅助各值几分，以及 `dmg_req`（数值部分需要的伤害类型）和 `dmg_share`（价值里多少来自这部分数值）。")
L.append("2. **英雄**：伤害类型按技能原文里'物理伤害/物理攻击力'与'法术伤害/法术攻击力'出现次数，数值面板兜底，7 个特殊英雄手工覆盖（亚连/艾琳/露娜/大司命/曹操/海月/刘邦，理由写在 json）；"
         "射程按 `attackDistance`（1=近战）；普攻依赖度按攻速档位+技能里'普攻/攻速'字样+`hero_roles.damage_shape.source`，27 人按技能原文手工校正（数值校正表，不是规则）；法力循环按 `energy`；定位按 `hero_roles` 的 combat_role + carry_value，7 人覆盖；被动钩子用 HOOK_RULES 正则从卡面推出、每条附引文；队友依赖/提供/成长/单C 判定见第 6 节。")
L.append("3. **适配分**：按下表规则逐条计算，`why` 字段写明每一步。\n")
L.append("## 2. 规则表\n")
for r in RULES:
    L.append("- " + r)
L.append("\n## 3. 每件装备一行摘要\n")
L.append("| 装备 | 类别 | 数值 / 关键效果 | 条件 | 适合谁（≥6 前 6） | 不给谁 |")
L.append("|---|---|---|---|---|---|")
for n, d in I.items():
    st = "，".join("%s %s" % kv for kv in d["stats"].items())
    ef = "，".join(a for a, _ in d["effects"])
    cond = "，".join(a for a, _ in d["conds"] if a != "唯一")
    if d["rng"]:
        cond = (cond + "，" if cond else "") + d["rng"]
    if n in RECIPES:
        st, ef = "四选一：" + "/".join(RECIPES[n]), ""
    L.append("| %s | %s | %s | %s | %s | %s |" % (n, d["cat"], "；".join(x for x in (st, ef) if x) or "—", cond or "—", who(n), who(n, False)))
restr = [x for x in diffs if x[4] == -1 and x[3] > 0]
rc = Counter(x[2] for x in restr)
L.append("\n## 4. 与旧画像（`hero_traits.item_fit_score` / `equip_profiles.json`）的差异\n")
L.append("对比口径：旧分 = 现在 `hero_traits.item_fit_score(装备, 英雄, hero_roles.combat_role)` 的返回值（None 的不比；注意旧画像 2026-10-01 已手工把马超加进自然法杖 heroes_avoid，但大棒的 heroes_best 里仍有马超）。共比较 %d 对，|差|≥3 的 %d 对。\n" % (len(diffs), sum(x[0] >= 3 for x in diffs)))
L.append("### 4a. 穿戴限制：旧画像给了正分、新表判不可穿（-1）的 %d 对\n" % len(restr))
L.append("旧画像按'英雄类型 × 优先级'打分，完全不看穿戴限制。按装备汇总：" + "、".join("%s %d 人" % kv for kv in rc.most_common()) + "。例：" + "；".join("%s×%s 旧 %g" % (x[1], x[2], x[3]) for x in restr[:6]) + "。\n")
L.append("### 4b. 其余差异最大的 30 处\n")
L.append("| # | 英雄 | 装备 | 旧分 | 新分 | 原因 |")
L.append("|---|---|---|---|---|---|")
rest = [x for x in diffs if not (x[4] == -1 and x[3] > 0)]
for i, (dd, hn, it, old, new, why) in enumerate(rest[:30], 1):
    L.append("| %d | %s | %s | %g | %g | %s |" % (i, hn, it, old, new, why.replace("|", "/")))
L.append("\n## 5. 马超 × 自然法杖 / 大棒 核对\n")
mc = FIT["马超"]
L.append("- 马超卡面：物攻 40 / 法攻 0、技能全是物理伤害、attackDistance=1（近战）；被动'己方施加控制效果时，马超获得15点法力值及5%暴击率'。")
L.append("- 自然法杖：'法术攻击力+20%' 对法攻 0 的马超 = 0；'远程穿戴者每消耗40点法力值（近战80点法力值）'变猪 → 近战频率减半。")
L.append("- 新分：自然法杖 **%g**（%s）" % (mc["自然法杖"]["score"], mc["自然法杖"]["why"]))
L.append("- 大棒四选一：%s" % mc["大棒"]["why"])
L.append("- 马超前 8：" + "、".join("%s %g" % (k, v["score"]) for k, v in sorted(mc.items(), key=lambda x: -x[1]["score"])[:8] if k not in RECIPES))
L.append("\n## 6. 不能当单C 的英雄（solo_carry_ok=false）\n")
L.append("背景：阿离的打法是**单C** —— 枫叶舞只绑 1 人，红叶染锋/遗志/传承把别人的等级搬给主C，古币/加等级牌/装备都喂给主C。"
         "所以主C 必须是'吃等级和装备就变强'的人；阿离常用阵容（河洛牺牲、日落海、大河…）以等级引擎和前排为主，控制来源少。\n")
L.append("判定规则（通用，规则里不出现任何英雄名，全部从卡面文字推出）：\n")
for r in TEAM_RULES:
    L.append("- " + r)
L.append("\n**推理链示例 —— 马超为什么不适合阿离**：\n")
mh = HERO["马超"]
L.append("1. team_dependency：卡面'%s' → 触发来自'己方施加控制'（需要场上提供控制的单位 ≥2），收益是**法力+暴击** = 喂输出（+1）。" % mh["team_dependency"][0]["text"])
L.append("2. 卡面只有这一句，整张卡的效果都靠队友触发（+1）。")
e40 = [e for e in mh["scales_with"]["level_enhance"] if e["conditional_on"]]
if e40:
    L.append("3. 等级强化 %d 级'%s'要暴击，而马超基础暴击率 0%%、暴击只来自上面那条队友依赖（+1）→ 阿离搬来的等级在这一档兑现不了。" % (e40[0]["level"], e40[0]["text"]))
L.append("4. 自身成长 self_scale=%d：技能攻击力系数合计 %d%%（<300%%），无自带叠层，只有 %d 条不附带条件的输出强化 → 装备的攻击属性收益也一般。" % (
    mh["scales_with"]["self_scale"], mh["scales_with"]["attack_coef_pct"], sum(1 for e in mh["scales_with"]["level_enhance"] if e["output"] and not e["conditional_on"])))
L.append("5. 依赖权重 %.1f ≥2 且 self_scale ≤2 → false。阿离给的是等级和装备，马超要的是队友控制；阿离阵容控制少，喂不饱。马超自己也不 provides 控制（provides=%s）。" % (
    mh["dep_weight"], "、".join(p["label"] for p in mh["provides"]) or "无"))
L.append("\n| 英雄 | 定位 | 依赖权重 | 自身成长 | 队友依赖（原文 / 需要） | 理由 |")
L.append("|---|---|---|---|---|---|")
for n, x in HERO.items():
    if x["solo_carry_ok"] or x["role"] in ("tank", "support"):
        continue
    deps = "；".join(sorted({"%s（%s）" % (t["text"], t["needs"]) for t in x["team_dependency"] if not t["universal"]}))
    L.append("| %s | %s | %.1f | %d | %s | %s |" % (n, x["role"], x["dep_weight"], x["scales_with"]["self_scale"], deps or "—", x["solo_carry_why"]))
nt = [n for n, x in HERO.items() if x["role"] in ("tank", "support")]
L.append("\n另有 %d 个坦克/辅助按 S1 一律 false（不当主C）：%s。\n" % (len(nt), "、".join(nt)))
near = [(n, x) for n, x in HERO.items() if x["solo_carry_ok"] and x["dep_weight"] >= 1]
if near:
    L.append("**能当单C、但有队友依赖要留意的**（依赖权重 ≥1，自身成长够强所以判 true）：" + "；".join(
        "%s（权重 %.1f / 成长 %d：%s）" % (n, x["dep_weight"], x["scales_with"]["self_scale"], "、".join(sorted({t["trigger"] for t in x["team_dependency"] if not t["universal"]}))) for n, x in near) + "。\n")
L.append("\n## 7. 已知局限\n")
L.append("- 普攻是否回蓝游戏没写明；按'朵莉亚普攻额外回 5 法力'推断普攻本身也回蓝，所以攻速装对技能型只按触发规则降权，不归零。")
L.append("- 「不死鸟之眼」'生命恢复和护盾效果提升30%' 没写是施放还是承受，按承受处理（给坦克）。")
L.append("- 「小鹿的庇护」的小鹿来源没查到（疑似棋手召唤物），按通用保命估分。")
L.append("- 阵营/配对类（鼓舞之戒看同阵营人数、怒莲之刺要配银莲庇佑、法潮灵杖看左右邻居、驱雷看司空震在不在场、不屈意志要多人穿）是**局面相关**的，表里只给静态分，接入时要再乘局面因子。")
(ROOT / "docs/equip-relations.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("items", len(I), "heroes", len(HERO), "pairs", sum(len(v) for v in FIT.values()))
print("restricted diffs", len(restr))
for dd, hn, it, old, new, why in rest[:15]:
    print(hn, it, old, "->", new, "|", why[:160])
print("solo_carry_ok=false 的输出位:", [h for h, v in HERO.items() if not v["solo_carry_ok"] and v["role"] not in ("tank", "support")])
