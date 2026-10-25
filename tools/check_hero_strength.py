"""英雄「当阿离主C」的输出强度模型（2026-10-01）+ 自检。

用法：
  python -X utf8 tools/check_hero_strength.py --build   # 重算并写 data/hero_strength.json 与强度报告，然后自检
  python -X utf8 tools/check_hero_strength.py           # 只自检现有的 data/hero_strength.json
  python -X utf8 tools/check_hero_strength.py --sens    # 敏感性：换全局假设（等级成长率/战斗时长/回蓝）看排名稳不稳

方法：
  1. 技能抽取表 SPEC：每个英雄按 1/10/40/100 级强化写成"每次施放的伤害分量 (固定值, 系数, 按什么缩放, 段数, 目标数)"
     + 普攻倍率 + 自保。只抽数字，不写英雄特判；每条的原文依据由脚本从 cards.json 自动附上，解释写在 notes。
  2. 统一公式模拟一场 20 秒战斗：法力循环 → 施放次数；普攻 + 技能 + 召唤物 + 持续伤害；
     承伤模型 → 存活时间（打不出来的伤害不算）。主C 等级 15/40/70/100/150、身上 0~3 件输出装各算一次
     （敌方单位固定 25 级，敌方主C 与我方同级）。
  3. 评分 = 五个等级下"有效伤害"相对第 90 百分位、按实测主C 等级分布加权平均 × 等级可兑现度修正，线性映射到 0~10。
     可兑现度 = 100→150 级（没有新强化）伤害涨幅 / 理想涨幅：固定值、按敌方生命百分比的伤害不随主C 涨。
  4. 校准：历史实战（主C → 名次）与官网阵容热度，只用来检验，不回灌进分数。
"""
import json, math, re, sys, glob, collections, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "data" / "hero_strength.json"
OUT_MD = ROOT / "docs" / "hero-strength.md"

# ───────────────────────── 全局假设（改这里，敏感性用 --sens 看） ─────────────────────────
G = dict(
    g=0.02,          # 英雄每级属性（攻击/生命）成长率：attr(lv) = attr0 × (1 + g×lv)。卡牌库没给，假设对所有英雄相同
    T=20.0,          # 一场战斗的参考时长（秒）
    mana_per_aa=10,  # 每次普攻回的法力（卡牌库没给，按常见自走棋设定）
    enemy_lv=25,     # 参考敌方单位等级（我方打的对象；data/enemy_levels.json：R10 前后对手非主C 单位多在 20~30 级）
    inc_follow=1.0,  # 承伤随"敌方主C 等级"走，敌方主C 等级 = 我方主C 等级 × inc_follow（enemy_levels.json：对手最高等级与我方主C 同量级甚至更高）
    enemy_hp0=280,   # 参考敌方单位基础生命（85 名英雄 HP 中位数附近）
    inc0=25.0,       # 承伤基准（自洽推算）：本模型 85 名英雄 1 级 DPS 中位数 24/s × 敌方 6 人 ÷ 我方暴露当量 4.35（3 前排×1 + 3 后排×0.45）× 边打边减员 0.75 ≈ 25 HP/s
    N=6,             # 参考敌方单位数（中后期人口 6~7）
    team_casts=0.6,  # 队友合计每秒施放次数（只用于"队友施法给我回蓝"的卡面，如诸葛亮）
    item_atk=0.20,   # 阿离喂给主C 的一件典型输出装：+20% 攻击力（只放大攻击系数部分，固定伤害不涨）
    item_as=0.10,    #                         +10% 攻速
)
ITEMS_BY_LV = ((100, 3), (70, 2), (40, 1), (0, 0))   # 主C 等级 → 身上输出装件数（装备 3 件上限）
LEVELS = (15, 40, 70, 100, 150)
# 权重 = 实测"第 8 回合起我方主C 处在该等级段的回合占比"（147 局 754 个回合：<30 级 35.1% / 30~55 24.4% /
#   55~85 17.8% / 85~125 13.7% / ≥125 9.0%；各回合中位数 R8 21 级、R10 40、R12 74、R14 97）。
#   第 8 回合起：路线已锁、主C 已定，名次在这之后拉开；第 5~7 回合主C 常是占位的过渡件。
W_LV = {15: 0.35, 40: 0.24, 70: 0.18, 100: 0.14, 150: 0.09}
R_W = 0.25   # 等级可兑现度的权重：raw × (1 − R_W + R_W × min(1, 兑现度))。阿离喂的等级和装备都只放大"攻击系数"那部分
TIER_CUT = [(8.5, "S"), (7.0, "A"), (5.5, "B"), (4.0, "C"), (-1, "D")]

# 目标数常量（参考 6 个敌人、抱团站位）
T1 = 1.0      # 单体
NEAR = 1.0    # "附近/相邻敌人"（不含主目标）的期望人数
ADJ1 = 2.0    # 目标及附近 1 格
SURR = 2.5    # 自身周围 1 格（近战被围时）
ADJ2 = 3.0    # 周围 2 格 / 半径扩大
PATH = 2.0    # 沿途 / 路径 / 穿透一条线
ROW = 2.5     # 一整排
DENSE = 2.5   # "敌人密集处"小范围
ALL = 6.0     # 全场


def C(B=0, k=0.0, kind="main", hits=1, tgt=T1, crit_gate=False):
    """一个伤害分量：每次 (B + k×属性)，打 hits 段，每段命中 tgt 个目标。
    kind: main=英雄主攻击（物攻或法攻，取大的）/ pa / ma / hp=自身最大生命 / ehp=敌方最大生命（B 忽略）"""
    return (B, k, kind, hits, tgt, crit_gate)


# ───────────────────────── 技能抽取表 ─────────────────────────
# 每个英雄：t1/t10/t40/t100 逐档覆盖（后一档只写变化的字段；surv 按键合并）。
# 字段：sk 每次施放的伤害分量；aa 普攻倍率；aa_t 普攻目标数；aa_ehp 每次普攻附带敌方最大生命比例伤害；
#      aspd 常驻攻速加成；buf_as/buf_atk=(加成, 持续秒) 施放后的攻速/攻击力增益；atk_flat 固定攻击力加成（不随等级涨）；
#      chan 引导时长（期间不普攻不回蓝）；lock 施放后不回蓝的秒数（普攻照打）；once 一场只放一次；
#      cyc 法力需求倍率（如击杀回蓝）；team_mana 队友施法给的法力；pet 召唤物；pas 每秒持续伤害；open 开战一次性伤害；
#      skill_crit 技能可暴击；crit_add 额外暴击率；aspd_per_lv 每级攻速（卡面自带）；ma_flat 卡面固定法攻；
#      surv: sh/shp/sha 护盾(固定/最大生命比例/攻击系数) hl/hlp/hla 回复 dr 平均减伤 rev 复生比例 hide 不可选中时间占比 hp_add 额外生命
#      exp 暴露倍率（突进进后排 >1，远离敌人 <1，嘲讽 >1）；E 法力上限覆盖；init_full 开战满蓝
SPEC = {
 "苏烈": dict(t1=dict(sk=[C(80, .8, tgt=SURR)]), t10=dict(sk=[C(80, 1.2, tgt=SURR)]),
            t40=dict(surv=dict(rev=.5)), t100=dict(surv=dict(rev=1.0)),
            notes=["10 级减防是团队收益，不计", "100 级复生击飞的 10% 最大生命伤害（谁的生命不明）不计"]),
 "曜": dict(t1=dict(sk=[C(70, 1.2)]), t10=dict(surv=dict(sh=100, sha=1.0)),
           t100=dict(sk=[C(70, 1.2), C(130, 1.3)], surv=dict(sh=100, sha=1.0, hl=200, hla=.75)),
           notes=["40 级每 3 次击飞是控制，不计伤害", "100 级强化攻击按每次施法多一段 130+130% 计"]),
 "孙膑": dict(t1=dict(sk=[C(50, 1.0, tgt=ADJ1)]), t10=dict(sk=[C(50, 1.5, tgt=ADJ1)]),
            t40=dict(sk=[C(50, 1.5, tgt=(2 * ADJ1 + 3 * ADJ1) / 3)]),
            notes=["40 级每第 3 次对 3 人投弹 → 平均每次施法 (2×2+6)/3 个目标当量", "卡面『每放 3 次技能永久 +1 级』是等级引擎，不计输出"]),
 "狂铁": dict(t1=dict(sk=[C(50, 1.0, hits=2, tgt=1.5)]), t10=dict(surv=dict(sh=100, shp=.1)),
            t40=dict(surv=dict(sh=400, shp=.1)), t100=dict(surv=dict(sh=400, shp=.1, hlp=.1)),
            notes=["『身前的敌人』按 1.5 个", "40 级每命中 1 人护盾 +100：2 段×1.5 人 ≈ +300"]),
 "孙尚香": dict(t1=dict(sk=[C(50, 1.0)]), t10=dict(sk=[C(50, 1.5)]),
             t40=dict(sk=[C(50, 1.5), C(30, .9, tgt=NEAR)]),
             t100=dict(sk=[C(50, 1.5), C(30, .9, tgt=NEAR), C(75, 2.25), C(45, 1.35, tgt=NEAR)]),
             notes=["降 50% 物防对自己和队友都有收益，未计（偏保守）", "40 级爆炸 = 60% 技能伤害给附近敌人", "100 级额外 1 发 1.5 倍"]),
 "弈星": dict(t1=dict(sk=[C(100, .5, tgt=ROW)]), t10=dict(sk=[C(100, 1.0, tgt=ROW)]),
            notes=["40/100 级是困住敌人，不计伤害", "卡面是棋手经验件"]),
 "干将莫邪": dict(t1=dict(sk=[C(100, .5)]), t10=dict(sk=[C(150, 1.0)]), t40=dict(sk=[C(150, 1.0, hits=1.5)]),
              t100=dict(sk=[C(150, 2.0, hits=1.5)]),
              notes=["『1 对飞剑』按单体一次计", "40 级每第 2 次多 1 对 → 平均 1.5 对", "卡面送装备『双剑·雄』未计"]),
 "李元芳": dict(t1=dict(sk=[C(0, 1.5), C(0, 1.0 / 3)]), t10=dict(sk=[C(0, 1.5), C(0, 1.5 / 3)]),
             t40=dict(sk=[C(0, 1.5), C(0, 1.5 / 3)], aa=1.5), t100=dict(sk=[C(0, 1.5), C(0, 1.5 / 3)], aa=1.5, aspd=.75),
             notes=["印记 3 层引爆 → 平均每次命中 +1/3 次引爆", "40 级普攻也叠印记：普攻平均 1+1.5/3", "100 级引爆后叠攻速（最多 5 层×25%），整场平均按 +75%"]),
 "牛魔": dict(t1=dict(sk=[C(30, 1.0)], surv=dict(sh=80)), t10=dict(sk=[C(30, 1.2)]),
            notes=["40/100 级击飞眩晕是控制"]),
 "亚连": dict(t1=dict(sk=[C(0, .65, "ma"), C(0, .65, "pa", hits=2)], skill_crit=True, surv=dict(dr=.15)),
            t10=dict(sk=[C(0, .65, "ma", hits=2), C(0, .65, "pa", hits=3)]),
            t40=dict(sk=[C(0, .65, "ma", hits=3), C(0, 1.15, "pa", hits=4)], surv=dict(dr=.3)),
            t100=dict(sk=[C(0, .65, "ma", hits=3, tgt=ADJ2), C(0, 1.15, "pa", hits=4, tgt=ADJ2)]),
            notes=["普攻和技能可暴击（原文）", "怒气减伤 2%/层，10 层上限、40 级 20 层；整场平均按一半", "5 段/7 段的法术/物理拆分原文没写，按物理多一段分",
                   "双剑合击真实伤害未单列"]),
 "米莱狄": dict(t1=dict(sk=[], pet=dict(n=1, cap=2, B=50, k=.3, kind="ma", aps=.7)),
             t10=dict(pet=dict(n=2, cap=2, B=50, k=.6, kind="ma", aps=.7)),
             t100=dict(pet=dict(n=2, cap=4, B=50, k=.6, kind="ma", aps=.7)),
             notes=["仆从默认上限原文没写，按 2（100 级提升至 4）", "仆从攻速按 0.7/秒", "满上限后施法加仆从攻速 15%，未计"]),
 "项羽": dict(t1=dict(sk=[], open=[C(100, .05, "hp", tgt=PATH)], surv=dict(hl=150, hlp=.05)),
            t10=dict(surv=dict(hl=300, hlp=.08)), t100=dict(surv=dict(hl=300, hlp=.08, dr=.3)),
            notes=["开战冲撞 100+5% 最大生命一次", "主动只回血", "100 级减伤叠层按平均 30%"]),
 "鬼谷子": dict(t1=dict(sk=[]), notes=["技能只给友军临时等级，没有伤害"]),
 "梦奇": dict(t1=dict(sk=[C(50, .7, tgt=SURR)], surv=dict(hlp=.08)), t10=dict(sk=[C(50, 1.0, tgt=SURR)], surv=dict(hl=100, hlp=.1)),
            t100=dict(surv=dict(hl=100, hlp=.2))),
 "芈月": dict(t1=dict(sk=[], pet=dict(n=1, cap=9, B=0, k=.8, kind="ma", aps=.7, dur=10)),
            t10=dict(pet=dict(n=1, cap=9, B=0, k=1.0, kind="ma", aps=.7, dur=10)),
            t40=dict(pet=dict(n=5, cap=9, B=0, k=1.0, kind="ma", aps=.7, dur=10)),
            notes=["仆从给等级最高的近战友军，伤害仍按芈月法攻", "仆从攻速按 0.7/秒，持续 10 秒", "窃取双防是团队收益"]),
 "沈梦溪": dict(t1=dict(sk=[C(100, 1.0, tgt=DENSE)]), t10=dict(sk=[C(100, 1.5, tgt=DENSE)]),
             t40=dict(sk=[C(100, 1.5, tgt=3.5)]),
             notes=["40 级半径 +1 格 → 目标数 3.5", "卡面『每对他用 1 张效果牌本回合法力上限 -15』要靠阿离喂牌，基础分不计（见路线加成）"]),
 "盾山": dict(t1=dict(sk=[C(150, 0, "fix", tgt=1.5)], surv=dict(dr=.1)), t10=dict(sk=[C(400, 0, "fix", tgt=1.5)], surv=dict(dr=.3)),
            t100=dict(surv=dict(dr=.35)),
            notes=["每秒 50 点固定伤害 × 持续 3/8 秒", "减伤按持续时间折成平均"]),
 "曹操": dict(t1=dict(sk=[], buf_atk=(.2, 5), surv=dict(hl=100, hlp=.15)), t10=dict(buf_atk=(.5, 5)),
            t40=dict(buf_atk=(.5, 10), aa_t=1.5), t100=dict(surv=dict(hl=100, hlp=.15, hla=1.5)),
            notes=["技能本身没有伤害，提高基础攻击力", "40 级剑气可穿透 → 普攻 1.5 个目标", "100 级 30% 吸血按每次施法回 1.5 倍攻击力"]),
 "虞姬": dict(t1=dict(sk=[C(0, 1.2), C(50, .8, hits=4)], aa=1.3), t10=dict(sk=[C(0, 1.2), C(50, 1.2, hits=4)]),
            t40=dict(buf_as=(.5, 3)), notes=["被动 30% 追加普攻 → 普攻 ×1.3", "100 级眩晕是控制"]),
 "马可波罗": dict(t1=dict(sk=[C(25, .15, hits=8, tgt=1.3)]), t10=dict(sk=[C(40, .25, hits=8, tgt=1.3)]),
              t40=dict(sk=[C(40, .25, hits=16, tgt=1.3)]),
              notes=["朝当前方向连射 → 每颗按 1.3 个目标", "伤害以固定值为主（40 vs 25%攻击），等级很难兑现", "击杀加子弹、100 级友军攻速未计"]),
 "赵云": dict(t1=dict(sk=[C(60, .6, hits=4)], surv=dict(hl=160)), t10=dict(sk=[C(60, 1.0, hits=4)], surv=dict(hl=320)),
            t40=dict(sk=[C(60, 1.0, hits=5.5)], surv=dict(hl=440)),
            t100=dict(sk=[C(60, 1.0, hits=5.5 * 1.15)], surv=dict(hl=500, shp=.3 / 4)),
            notes=["40 级半血以下 +3 次 → 平均 5.5 次", "100 级 30% 血以下 1.5 倍按平均 +15%；首次护盾一次性按 4 次施法摊"]),
 "大乔": dict(t1=dict(sk=[C(100, 1.0, tgt=PATH)]), t10=dict(sk=[C(100, 1.5, tgt=PATH)]),
            t40=dict(sk=[C(100, (1.5 * 2 + 2.0) / 3, tgt=PATH)]), t100=dict(sk=[C(100, 2.0, tgt=PATH)]),
            notes=["法攻只有 20（辅助）", "40 级每第 3 次巨浪 → 平均系数"]),
 "公孙离": dict(t1=dict(sk=[C(0, 1.0, tgt=PATH), C(0, 1.0, hits=2, tgt=1.5)]), t10=dict(buf_as=(.3, 4)),
             t40=dict(buf_as=(.5, 4)),
             notes=["纸伞首次命中 + 悬停每秒 100%（悬停按 2 秒、1.5 人）", "伞不在手时攻速增益按每次施法 4 秒", "卡面『每个已觉醒英雄攻速 +15%』要靠队友，未计；夺取是等级手段"]),
 "百里守约": dict(t1=dict(sk=[C(50, 7.0)]), t10=dict(sk=[C(200, 8.0)]), t40=dict(sk=[C(200, 8.0, tgt=1.3)]),
              t100=dict(sk=[C(200, 8.0, tgt=1.6)]),
              notes=["40 级击杀后子弹继续（衰减）→ 1.3 个目标；100 级不衰减 → 1.6", "卡面『牺牲英雄阵亡时法力上限 -20』要靠牺牲队友，基础分不计（见路线加成）"]),
 "墨子": dict(t1=dict(sk=[C(30, .5, hits=4, tgt=1.5)]), t10=dict(sk=[C(50, .75, hits=4, tgt=1.5)]),
            t100=dict(sk=[C(50, .75, hits=10, tgt=1.5)]), notes=["每 0.5 秒一跳，屏障 2 秒（100 级 5 秒）", "法攻只有 20"]),
 "百里玄策": dict(t1=dict(sk=[C(50, 1.0), C(50, 1.0)]), t10=dict(sk=[C(50, 1.3), C(50, 1.0)]),
              t40=dict(sk=[C(50, 1.3), C(50, 1.0), C(100, 2.0, tgt=PATH)], aa=1.2, exp=1.15),
              t100=dict(aspd=.5),
              notes=["40 级钩住后突进、普攻被钩目标 120%", "100 级击杀后狂暴攻速 +150% 持续 5 秒，按整场平均 +50%", "卡面闪现英雄互相喂等级，要队友"]),
 "廉颇": dict(t1=dict(sk=[C(50, .8)], surv=dict(sh=100)), t10=dict(sk=[C(50, 1.2)], surv=dict(sh=250)),
            t40=dict(surv=dict(sh=330)), t100=dict(surv=dict(sh=450, hlp=.1))),
 "蒙恬": dict(t1=dict(sk=[], pet=dict(n=1, cap=2, B=20, k=.3, kind="pa", aps=.7, dur=20), surv=dict(dr=.05)),
            t10=dict(surv=dict(dr=.12)), t40=dict(pet=dict(n=2, cap=4, B=20, k=.3, kind="pa", aps=.7, dur=20), surv=dict(dr=.24)),
            t100=dict(pet=dict(n=4, cap=4, B=20, k=.3, kind="pa", aps=.7, dur=20)),
            notes=["玄雍士兵的攻击力原文没写，按 20+30% 物攻、0.7 次/秒估", "分担伤害折成减伤"]),
 "蔡文姬": dict(t1=dict(sk=[]), notes=["被动回血加双防，没有伤害"]),
 "云中君": dict(t1=dict(sk=[C(50, .8, tgt=SURR), C(0, .2 * .7, "ehp", tgt=SURR)]),
             t10=dict(sk=[C(50, 1.2, tgt=SURR), C(0, .2 * .7, "ehp", tgt=SURR)]),
             t40=dict(aa_ehp=.1 * .7), t100=dict(sk=[C(50, 1.2, tgt=4), C(0, .2 * .7, "ehp", tgt=4)]),
             notes=["撕裂：每层每秒 2% 当前生命 × 5 秒，2 层 = 20% 当前生命（按 70% 折算成最大生命）",
                    "撕裂按敌方生命算 → 主C 自己涨等级它不涨（等级难兑现）", "卡面开团：穿装备永久涨级，是全队等级引擎"]),
 "姬小满": dict(t1=dict(sk=[C(150, 1.0)]), t10=dict(sk=[C(150, 1.5)]), t40=dict(buf_atk=(.4, 5)),
             t100=dict(sk=[C(150, 1.5), C(0, 1.5, tgt=ADJ1)]),
             notes=["40 级印记 +40% 伤害按 5 秒攻击力增益近似", "100 级重击范围伤害；跃起 70% 减伤时间太短未计"]),
 "苍": dict(t1=dict(sk=[C(50, 1.2)], aa=1.4, lock=8, buf_atk=(.2, 99), pet=dict(n=1, cap=3, B=0, k=.8, kind="pa", aps=1.0)),
           t10=dict(pet=dict(n=1, cap=3, B=0, k=1.2, kind="pa", aps=1.0)), t40=dict(buf_atk=(.4, 99)),
           t100=dict(sk=[C(50, 1.2, hits=3)], pet=dict(n=3, cap=3, B=0, k=1.2, kind="pa", aps=1.0)),
           notes=["二连普攻 60%+80% → 普攻 1.4；状态 8 秒内不回蓝", "狼魂基础攻击倍率原文只写 10 级 120%，基础按 80%",
                  "狼魂叠攻击力上限 30%/60%，整场平均按 20%/40%"]),
 "铠": dict(t1=dict(sk=[], surv=dict(dr=.1)), t10=dict(surv=dict(dr=.15, sh=200)), t40=dict(aa=1 + 1 / 3),
           notes=["技能只是减伤武装；40 级每第 3 次普攻 +100%"]),
 "白起": dict(t1=dict(sk=[], surv=dict(hlp=.1), exp=1.3), t10=dict(surv=dict(hlp=.1, shp=.2)),
            t40=dict(surv=dict(hlp=.1, sh=200, shp=.3)), notes=["嘲讽：暴露 ×1.3；没有伤害"]),
 "明世隐": dict(t1=dict(sk=[], surv=dict(hla=.6, hl=100)), notes=["治疗辅助，没有伤害"]),
 "阿轲": dict(t1=dict(sk=[C(30, 1.0, hits=2)]), t10=dict(sk=[C(30, 1.3, hits=2)]), t40=dict(surv=dict(hide=.7)),
            t100=dict(skill_crit=True, crit_add=1.0, surv=dict(hide=.85)),
            notes=["40 级开战 20 秒不被优先攻击 ≈ 大部分时间不挨打（按 70%）", "100 级技能必定暴击（只对技能）"]),
 "马超": dict(t1=dict(sk=[C(50, 1.0)], exp=1.15, atk_flat=3.5, surv=dict(hl=100 + 70, hla=.75 * .7)),
            t10=dict(sk=[C(50, 1.5)]),
            t40=dict(sk=[C(50, 1.5), C(0, .5, tgt=PATH, crit_gate=True)]),
            t100=dict(sk=[C(50, 1.5, hits=1 + 5 / 6), C(0, .5, tgt=PATH, crit_gate=True)]),
            notes=["突袭 3 格内最远的敌人 → 暴露 ×1.15", "拾枪回血 100+75% 与 +5 物攻按每次施法 0.7 把",
                   "40 级只有『暴击时』才回收并打路径 → 按暴击率兑现，而马超基础暴击率 0%（只能靠队友控制喂 5%/次）",
                   "100 级每回收 3 把下一次连续突袭 6 次：回收主要靠走路拾取，按每 6 次施法触发 1 次估"]),
 "裴擒虎": dict(t1=dict(sk=[C(0, 1.2)], atk_flat=20 * .8), t10=dict(sk=[C(0, 1.5)]),
             t40=dict(atk_flat=20, aspd=.3), t100=dict(sk=[C(0, 2.0)], atk_flat=40, aspd=.6),
             notes=["化身猛虎 +20 物攻是固定值（不随等级涨），40 级前第一次施法才变身 → 按 80% 时间", "100 级属性翻倍"]),
 "老夫子": dict(t1=dict(sk=[], surv=dict(dr=.1)), t10=dict(surv=dict(dr=.15)), t40=dict(surv=dict(dr=.15, hl=160)),
             t100=dict(surv=dict(dr=.15, hl=400)), notes=["技能只有减伤和回血"]),
 "西施": dict(t1=dict(sk=[C(50, 1.0, tgt=1.5), C(100, 1.0, tgt=1.5)]), t10=dict(sk=[C(100, 1.2, tgt=1.5), C(100, 1.5, tgt=1.5)]),
            t40=dict(sk=[C(100, 1.2, tgt=1.5), C(100, 1.5, tgt=1.5), C(100, 0, "fix", tgt=1.5)]),
            notes=["能量球位置的单位按 1.5 个", "40 级牵引爆炸 100 点固定伤害 + 眩晕"]),
 "刘禅": dict(t1=dict(sk=[C(50, 1.0)], surv=dict(sh=100)), t10=dict(surv=dict(sh=300)), t40=dict(surv=dict(sh=400, dr=.2)),
            t100=dict(surv=dict(sh=600, shp=.05, dr=.2))),
 "猪八戒": dict(t1=dict(sk=[C(0, .07, "hp", tgt=SURR)], surv=dict(hp_add=45 * 15)), t10=dict(sk=[C(0, .10, "hp", tgt=SURR)]),
             t40=dict(surv=dict(hp_add=45 * 30)), t100=dict(sk=[C(0, .10, "hp", tgt=DENSE)], surv=dict(hp_add=45 * 50)),
             notes=["技能按自身最大生命 → 随自身生命成长", "叠层每 600 伤害 +45 生命，整场平均按上限的一半"]),
 "程咬金": dict(t1=dict(sk=[], surv=dict(hl=35, hlp=.1)), t10=dict(surv=dict(hl=60, hlp=.1)), t40=dict(surv=dict(hl=60, hlp=.1, dr=.1)),
             t100=dict(surv=dict(hl=60, hlp=.2, dr=.1))),
 "狄仁杰": dict(t1=dict(sk=[], buf_as=(.4, 5)), t10=dict(buf_as=(.9, 5)), t40=dict(aa=(2 + 1.5) / 3), t100=dict(aa=1.5),
             notes=["技能不造成伤害，加 40%/90% 基础攻速 5 秒 → 输出几乎全靠普攻",
                    "40 级每 3 次普攻抽 1 张蓝牌（按替换该次普攻计）；100 级全部普攻换蓝牌",
                    "卡面『古币 → 攻速永久 +5%』靠河洛古币路线，基础分不计（见路线加成）"]),
 "张良": dict(t1=dict(sk=[C(100, 0, "fix", hits=3)], chan=1.5), t10=dict(sk=[C(100, .5, hits=3)]),
            t100=dict(sk=[C(100, .5, hits=6)], chan=3), notes=["每 0.5 秒一跳；1 级伤害是纯固定值"]),
 "少司缘": dict(t1=dict(sk=[], surv=dict(hl=100)), t100=dict(sk=[C(150, .5, tgt=PATH)]), notes=["治疗辅助，100 级才有伤害；法攻 20"]),
 "亚瑟": dict(t1=dict(sk=[C(0, .5, hits=5, tgt=1.5)], surv=dict(dr=.1)), t10=dict(sk=[C(0, .8, hits=5, tgt=1.5)], surv=dict(dr=.15)),
            t40=dict(surv=dict(dr=.25)), t100=dict(sk=[C(0, .8, hits=5, tgt=1.5), C(0, .15, "hp", tgt=DENSE)])),
 "钟馗": dict(t1=dict(sk=[C(50, 1.0)]), t10=dict(init_full=True), t40=dict(sk=[C(50, 1.5)]), t100=dict(E=50),
            notes=["法攻只有 20；钩人是控制"]),
 "貂蝉": dict(t1=dict(sk=[C(50, .4, hits=5)]), t10=dict(sk=[C(50, .6, hits=5)]), t40=dict(sk=[C(50, .6, hits=9)]),
            t100=dict(exp=.75), notes=["花球随机打周围 2 格的敌人，每朵按 1 个目标", "伤害以固定 50 为主 → 等级难兑现", "100 级施法远离敌人 → 暴露 ×0.75"]),
 "安琪拉": dict(t1=dict(sk=[C(0, .1, hits=30, tgt=PATH)], chan=3), t10=dict(sk=[C(0, .17, hits=30, tgt=PATH)]),
             t40=dict(sk=[C(0, .17, hits=30, tgt=4.5)]), t100=dict(sk=[C(0, .17 * 1.3, hits=30, tgt=4.5)]),
             notes=["3 秒引导，每 0.1 秒一跳 → 30 跳", "40 级 3 道光辉 → 目标数 4.5", "100 级击杀延长 +15% 伤害按 ×1.3", "纯攻击系数 → 等级兑现度高"]),
 "瑶": dict(t1=dict(sk=[], pas=[C(50, .5, "ma", hits=.5, tgt=2)], surv=dict(shp=0)),
           notes=["附身在队友身上，每 2 秒打至多 2 人 50+50%；法攻 20", "护盾给的是队友"]),
 "镜": dict(t1=dict(sk=[C(60, .8)], exp=1.0), t10=dict(sk=[C(80, 1.2)]), t40=dict(sk=[C(80, 1.5, hits=2)]),
           t100=dict(sk=[C(80, 1.5, hits=3)], surv=dict(hl=450)),
           notes=["单体；镜像按『再打一次技能』计（偏宽松）", "闪现 → 开战跳到敌方后排镜像位，HP 240 近战", "卡面『每 2 张效果牌 +1 级』只涨等级，不涨伤害"]),
 "朵莉亚": dict(t1=dict(sk=[], aa=1 + .5 * 20 / 24, aa_k="ma", surv=dict(hla=1.0, hl=75)), notes=["治疗辅助；普攻附带 50% 法攻（法攻 20）"]),
 "莱西奥": dict(t1=dict(sk=[], aa=1 + 2.0 / 8, buf_as=(.3, 6)), t10=dict(buf_as=(.5, 6)), t40=dict(aa=1 + 2.0 / 4),
             t100=dict(sk=[C(0, 1.5, hits=5), C(0, .5, hits=5, tgt=NEAR)]),
             notes=["每 8 次普攻 1 枚强化飞弹 150%+周围 50%", "烟雾攻速增益按自身在烟雾里", "40 级友军伤害也充能（依赖队友）→ 按飞弹频率翻倍"]),
 "上官婉儿": dict(t1=dict(sk=[C(100, 1.0, tgt=PATH)]), t10=dict(sk=[C(100, 1.5, tgt=PATH)]), t40=dict(sk=[C(100, 1.5, tgt=2.5)]),
              t100=dict(sk=[C(100, 2.4, tgt=2.5)]), notes=["40 级每命中 1 人伤害 +50% → 按 2.5 个目标当量"]),
 "艾琳": dict(t1=dict(sk=[C(0, 1.0, "ma")], aa_k="pa", aspd_per_lv=.01), t10=dict(sk=[C(0, 1.5, "ma", hits=2.5)]),
            t40=dict(aa=1.0, aa_extra=[C(0, .1, "ma")], ma_flat_ramp=2 * 15),
            t100=dict(aa_t=1.5),
            notes=["卡面『开团：攻速 + 自身等级×1%』是英雄自带的等级→输出转换，计入", "10/20 层进化后每次多片飞叶，平均按 2.5 片",
                   "40 级 30 层后每次普攻 +2 法攻（整场平均按 +30）并附带 10% 法攻", "100 级普攻穿透 → 1.5 个目标"]),
 "扁鹊": dict(t1=dict(sk=[C(20, .6, hits=5, tgt=1.5)]), t10=dict(sk=[C(20, .9, hits=5, tgt=1.5)]),
            t40=dict(sk=[C(20, .9, hits=5, tgt=2.5)]), notes=["药瓶给血量比例最低的 3 个单位（敌我都算）→ 敌人按 1.5 个", "法攻 20"]),
 "嬴政": dict(t1=dict(sk=[C(10, .1, hits=80, tgt=1.2)], chan=5, ma_flat=7), t10=dict(sk=[C(15, .15, hits=80, tgt=1.2)]),
            t40=dict(sk=[C(15, .15, hits=100, tgt=1.2)]), t100=dict(sk=[C(15, .15, hits=120, tgt=1.5)]),
            notes=["5 秒内 80 把飞剑按引导 5 秒", "卡面开团 +7 法攻（用战术牌还会涨，未计）"]),
 "李信": dict(t1=dict(sk=[C(50, 1.8, tgt=PATH)]), t10=dict(sk=[C(100, 2.2, tgt=PATH)]), t40=dict(sk=[C(100, 2.2, hits=3, tgt=1.4)]),
            t100=dict(surv=dict(rev=1.0)), notes=["40 级 3 道剑气按扇形，每道 1.4 人", "100 级阵亡后狂暴复生按复生 100%", "卡面牺牲队友给临时等级（依赖队友）"]),
 "李白": dict(t1=dict(sk=[C(0, .8 * 6)], buf_atk=(.2, 99)), t10=dict(sk=[C(0, 1.2 * 6)]), t40=dict(sk=[C(0, 1.5 * 6)]),
            t100=dict(surv=dict(hide=.3)),
            notes=["6 次斩击伤害由范围内敌人平均分摊 → 总量不随人数涨，按单体总量计", "剑气每层 +10% 基础攻击、最多 4 层，整场平均 +20%", "100 级斩击不可选中按 30% 时间"]),
 "雅典娜": dict(t1=dict(sk=[C(30, 0, "fix", tgt=SURR)], surv=dict(shp=.15)), t10=dict(surv=dict(shp=.35))),
 "太乙真人": dict(t1=dict(sk=[], exp=1.3), t10=dict(surv=dict(sh=400)), t40=dict(sk=[C(400, 0, "fix", tgt=SURR)]), notes=["拉人嘲讽，伤害只有 40 级护盾破碎 400 固定值"]),
 "孙悟空": dict(t1=dict(sk=[C(100, 4.0, tgt=1.5)]), t10=dict(sk=[C(100, 5.0, tgt=1.5)]), t40=dict(sk=[C(100, 7.0, tgt=ADJ2)]),
             t100=dict(sk=[C(100, 7.0, tgt=4.5)]), notes=["砸向前方按 1.5 人；40 级 2 格范围；100 级 4 格"]),
 "甄姬": dict(t1=dict(sk=[C(30, .4, hits=4)]), t10=dict(sk=[C(30, .75, hits=4)]), t40=dict(sk=[C(30, .75, hits=4), C(50, 1.0)]),
            t100=dict(sk=[C(30, .75, hits=8), C(50, 1.0, hits=2)]), notes=["弹射每次 1 个目标", "40 级冻结结束伤害按每次施法冻住 1 人（100 级 2 人）"]),
 "小乔": dict(t1=dict(sk=[C(0, .1, hits=15)], chan=3), t10=dict(sk=[C(0, .15, hits=15)]), t40=dict(sk=[C(0, .15, hits=15, tgt=2)]),
            t100=dict(sk=[C(0, .15, hits=15, tgt=3)]), notes=["引导 3 秒，每 0.2 秒一跳 → 15 跳"]),
 "大司命": dict(t1=dict(sk=[C(150, 2.0, tgt=ADJ2)], aa=1.4, once=True, surv=dict(hl=100, hlp=.05 + .1)),
             t10=dict(surv=dict(hl=250, hlp=.1 + .1)), t40=dict(aa_t=2.5),
             t100=dict(pet=dict(n=3, cap=3, B=0, k=.5, kind="pa", aps=.6)),
             notes=["普攻连接 3 秒 5 次×50% = 每秒 0.83 倍攻击，折成普攻倍率 1.4（攻速 0.6）", "施放后进入神巫状态不再回蓝 → 一场只放 1 次", "神巫状态普攻回血按 10% 生命计入"]),
 "庄周": dict(t1=dict(sk=[], surv=dict(sh=200)), t10=dict(surv=dict(sh=300)), notes=["只给护盾和驱散"]),
 "刘邦": dict(t1=dict(sk=[C(50, .3, "ma", hits=2)], aa_t=1.5, surv=dict(sh=200)), t10=dict(surv=dict(sh=320), aspd=.2),
            t100=dict(sk=[C(50, .3, "ma", hits=2), C(100, .5, "ma", tgt=SURR)]), notes=["印记由友军引爆，按每次施法 2 次", "法攻 20"]),
 "诸葛亮": dict(t1=dict(sk=[C(150, 1.0), C(60, .4, tgt=NEAR)], team_mana=15), t10=dict(sk=[C(150, 2.0), C(60, .8, tgt=NEAR)]),
             t40=dict(cyc=.7), t100=dict(sk=[C(250, 3.0), C(100, 1.2, tgt=NEAR), C(62.5, .75, hits=4)]),
             notes=["卡面『其他英雄放技能 → 诸葛亮 +15 法力』任何阵容都成立，按队友每秒 0.6 次施法计入",
                    "40 级击杀回 70% 法力 → 法力需求 ×0.7", "100 级 4 发小元气弹 25%"]),
 "韩信": dict(t1=dict(sk=[C(100, 0, "fix")], aa=(2 + 1.6) / 3), t10=dict(sk=[C(100, 1.5)]),
            t40=dict(sk=[C(100, 1.5), C(100, 2.0, hits=1.5)], exp=1.15),
            t100=dict(sk=[C(100, 2.0), C(200, 0, "fix", tgt=NEAR + 1), C(100, 2.0, hits=.5)]),
            notes=["每第 3 次普攻 160% 击飞 → 普攻平均 1.2", "40 级击杀后冲锋下一个按 0.5 次", "击杀加攻速未计"]),
 "蒙犽": dict(t1=dict(sk=[C(30, 1.0, hits=8 * .8)]), t10=dict(sk=[C(30, 1.5, hits=8 * .8)]), t40=dict(sk=[C(30, 1.5, hits=8 * .95)]),
            t100=dict(sk=[C(30, 1.5, hits=14 * .95)]), notes=["导弹随机打，重复命中衰减 40%（40 级 10%）→ 有效 80%/95%", "100 级 11 枚、每次施法 +1 至多 20，整场平均 14"]),
 "东皇太一": dict(t1=dict(sk=[], pas=[C(0, .03, "ehp", hits=1, tgt=1.5)], surv=dict(hlp=.03 * 4)),
              t10=dict(pas=[C(0, .05, "ehp", hits=1, tgt=1.5)], surv=dict(hlp=.05 * 4)),
              t40=dict(pas=[C(0, .05, "ehp", hits=1.5, tgt=1.5)]), notes=["能量球按每秒碰 1 次、1.5 人；伤害按敌方最大生命 → 自己涨等级不涨伤害"]),
 "敖隐": dict(t1=dict(sk=[C(100, .75, tgt=DENSE)]), t10=dict(aa_t=1.5, aa_extra=[C(0, .05)], surv=dict(hl=50)),
            t40=dict(aa_extra=[C(0, .075)], surv=dict(hl=175)), t100=dict(aa_extra=[C(0, .15)], surv=dict(hl=350)),
            notes=["10 级随机两种武器之一（真伤或回血），各按一半", "卡面『图腾给攻速和临时等级』要大河图腾，未计"]),
 "鲁班大师": dict(t1=dict(sk=[C(50, .6, "ma")]), t10=dict(sk=[C(50, .9, "ma")], surv=dict(sh=250)), t40=dict(sk=[C(50, .9, "ma", tgt=3)]),
              t100=dict(sk=[C(50, .9, "ma", tgt=6)]), notes=["法攻 20；以控制为主"]),
 "花木兰": dict(t1=dict(sk=[C(0, .8, hits=7 / 2), C(80, .6, tgt=SURR / 2)]), t10=dict(sk=[C(0, 1.05, hits=7 / 2), C(80, .6, tgt=SURR / 2)]),
             t40=dict(sk=[C(0, 1.05, hits=10 / 2), C(80, .6, tgt=SURR / 2)], surv=dict(dr=.1)),
             t100=dict(sk=[C(0, 1.05, hits=10 / 2), C(80, .6, tgt=ALL / 2)]),
             notes=["轻剑/重剑两种形态各按一半时间", "40 级轻剑斩击持续至目标阵亡 → 段数按 10", "100 级重剑全场"]),
 "司空震": dict(t1=dict(sk=[C(50, .65, tgt=ALL)]), t10=dict(sk=[C(50, 1.0, tgt=ALL)]),
             t40=dict(aa_extra=[C(75, .5, hits=2.31)], aa_extra_uptime=8, aspd=.15 * .5, surv=dict(sh=150)),
             t100=dict(aa_extra_uptime=99, aspd=.15, open=[C(50, 1.0, tgt=ALL)], surv=dict(sh=300)),
             notes=["雷鼓轰击全场", "雷霆之王：普攻闪电链传导 3 次、逐次 -25% → 2.31 段，40 级起施放后 8 秒，100 级全场",
                    "卡面『别人拿临时等级我拿一半』是等级引擎"]),
 "姜子牙": dict(t1=dict(sk=[C(80, .8, tgt=PATH)], chan=2), t10=dict(sk=[C(140, 1.4, tgt=PATH)]),
             t40=dict(sk=[C(140, 1.4, hits=3 * .9, tgt=PATH)]), t100=dict(sk=[C(140, 1.4, hits=8 * .8, tgt=1.2)]),
             notes=["汇聚 2 秒按引导", "多道冲击波命中同一单位衰减 15%", "100 级 8 方向散射，每道 1.2 人"]),
 "武则天": dict(t1=dict(sk=[C(100, .75, hits=10.7 * .9)], chan=8), t10=dict(sk=[C(100, 1.0, hits=10.7 * .9)]),
             t40=dict(sk=[C(100, 1.0, hits=10.7 * .9), C(100, 1.25, tgt=ALL)]),
             t100=dict(sk=[C(100, 1.0, hits=10.7 * .9), C(100, 1.25, hits=2.1, tgt=ALL)], chan=8, E=1),
             notes=["吟唱 8 秒、每 0.75 秒一次 → 10.7 次，重复命中衰减按 90%", "40 级吟唱结束巨型法阵打全场",
                    "100 级吟唱到回合结束且每 5 次伤害一个全场法阵 → 按连续吟唱建模（法力上限视作 1）"]),
 "吕布": dict(t1=dict(sk=[C(120, 1.2, tgt=ADJ2)], aa_t=1.5, surv=dict(hlp=.15, hla=.5)), t10=dict(sk=[C(120, 2.0, tgt=ADJ2)]),
            t40=dict(buf_as=(1.0, 5)), t100=dict(sk=[C(120, 2.0, tgt=ALL)], surv=dict(hlp=.3, hla=.8)),
            notes=["普攻对身前范围 → 1.5 人", "每命中 1 人回 5% 生命 + 20% 吸血", "100 级整个战场"]),
 "海月": dict(t1=dict(sk=[C(100, 1.5), C(50, .75, hits=2)]), t10=dict(sk=[C(100, 2.5), C(50, 1.25, hits=2)]),
            t40=dict(surv=dict(sh=50, shp=.1)), t100=dict(pas=[C(0, .05 / 5, "ehp", tgt=ALL)], surv=dict(sh=50 + 100, shp=.1)),
            notes=["法攻只有 20 → 伤害以固定值为主", "100 级水月幻境每 5 秒 5% 最大生命真伤（按全体敌人）", "卡面『牺牲：真伤 = 海月生命』是遗志流的玩法"]),
 "鲁班七号": dict(t1=dict(sk=[C(50, .8, hits=4, tgt=1.2), C(0, .8, hits=8, tgt=.8)]),
              t10=dict(sk=[C(50, 1.0, hits=4, tgt=1.2), C(0, 1.0, hits=8, tgt=.8)]),
              t40=dict(sk=[C(50, 1.0, hits=8, tgt=1.2), C(0, 1.0, hits=16, tgt=.8)]),
              t100=dict(sk=[C(50, 1.0, hits=8, tgt=1.2), C(0, 1.0, hits=16, tgt=.8), C(200, 3.0, tgt=ALL)]),
              notes=["弹坑每秒 80%/100%，按每个弹坑有效 2 秒、0.8 人", "100 级每次施法飞艇坠毁打全场"]),
 "海诺": dict(t1=dict(sk=[C(100, 1.5), C(50, 1.0, hits=2)]), t10=dict(sk=[C(150, 2.5), C(100, 1.5, hits=2)]),
            t40=dict(sk=[C(150, 2.5, hits=2), C(100, 1.5, hits=4)]), t100=dict(sk=[C(150, 2.5, hits=2), C(100, 1.5, hits=2 * (ALL - 1))]),
            notes=["40 级连续施放 2 次", "100 级子杖打至多 14 人 → 全部敌人"]),
 "露娜": dict(t1=dict(sk=[C(100, 1.0, "ma", tgt=PATH), C(0, .15, "ehp")], exp=1.15, surv=dict(hide=.1)),
            t10=dict(sk=[C(100, 2.0, "ma", tgt=PATH), C(0, .15, "ehp")]), t40=dict(cyc=.85),
            t100=dict(sk=[C(100, 2.0, "ma", tgt=PATH, hits=2), C(0, .15, "ehp", hits=2)]),
            notes=["突击最远敌人（突进进后排）", "标记 15% 最大生命伤害按敌方生命算", "40 级击杀回 15 法力 → 需求 ×0.85"]),
 "杨玉环": dict(t1=dict(sk=[C(100, .8, hits=.5, tgt=3)]), t10=dict(sk=[C(100, .8, tgt=(3 + ALL) / 3)]),
             t40=dict(sk=[C(100, 1.25, tgt=(3 + ALL) / 3)]), t100=dict(sk=[C(100, 1.25, tgt=ALL)]),
             notes=["清平/破阵循环 → 一半施法打 3 人", "10 级每第 3 次两种都放且全场 → 3 次里 1 次 3 人、1 次全场", "100 级每次全场"]),
 "周瑜": dict(t1=dict(sk=[C(100, .5, tgt=ADJ2)], pas=[C(0, .01, "ehp", tgt=ADJ2)], surv=dict(hl=100, hla=.5)),
            t10=dict(sk=[C(100, 1.2, tgt=ADJ2)], surv=dict(hl=100, hla=1.2)),
            t40=dict(sk=[C(100, 1.2, tgt=ADJ2), C(100, 1.5, tgt=ADJ2)]), t100=dict(pas=[C(0, .01, "ehp", tgt=ALL)]),
            notes=["法攻 20", "焦土每秒 1% 最大生命（按敌方生命）"]),
}

TIERS = (1, 10, 40, 100)


def load_cards():
    d = json.load(open(ROOT / "data" / "cards.json", encoding="utf-8"))["cards"]
    return {c["name"]: c for c in d.values() if c.get("category") == "英雄牌"}


def merged(spec, lv):
    m = {}
    for t in TIERS:
        if lv < t:
            break
        part = spec.get(f"t{t}", {})
        for k, v in part.items():
            if k == "surv":
                m["surv"] = {**m.get("surv", {}), **v}
            else:
                m[k] = v
    return m


def evidence(card):
    ev = []
    for sk in card.get("skills", []):
        ev.append(f"技能「{sk['name']}」：{sk['desc']}")
        for e in sk.get("enhance", []):
            ev.append(f"{e['level']} 级：{e['desc']}")
    if card.get("skill"):
        ev.append("卡面：" + card["skill"].replace("\n", " | "))
    return ev


def exposure(card, m):
    p = card["properties"]
    rng = p.get("attackDistance", 1)
    e = 0.45 if rng >= 4 else (0.7 if rng >= 2 else 1.0)
    if "闪现" in (card.get("skill") or ""):
        e *= 1.1
    return e * m.get("exp", 1.0)


def simulate(card, spec, lv, g=None):
    """一场参考战斗。返回 dict：dmg(总有效伤害)/dps/alive/surv/casts。"""
    G_ = {**G, **(g or {})}
    m = merged(spec, lv)
    p = card["properties"]
    s = 1 + G_["g"] * lv
    se = 1 + G_["g"] * G_["enemy_lv"]
    n_items = next(n for l0, n in ITEMS_BY_LV if lv >= l0)
    eq = 1 + G_["item_atk"] * n_items
    pa = p.get("phyAttack", 0) * s * eq
    ma = p.get("magAttack", 0) * s * eq + m.get("ma_flat", 0) + m.get("ma_flat_ramp", 0)
    main_k = "pa" if p.get("phyAttack", 0) >= p.get("magAttack", 0) else "ma"
    if m.get("atk_flat"):
        if main_k == "pa":
            pa += m["atk_flat"]
        else:
            ma += m["atk_flat"]
    surv = m.get("surv", {})
    hp = (p.get("HP", 240) + surv.get("hp_add", 0)) * s
    ehp_e = G_["enemy_hp0"] * se
    crit = min(1.0, p.get("criticalRate", 0) / 10000 + m.get("crit_add", 0))
    cm = 1 + crit * p.get("criticalEffect", 4000) / 10000

    def val(c, atk_mult=1.0):
        B, k, kind, hits, tgt = c[:5]
        kind = main_k if kind == "main" else kind
        if kind == "pa":
            x = B + k * pa * atk_mult
        elif kind == "ma":
            x = B + k * ma * atk_mult
        elif kind == "hp":
            x = B + k * hp
        elif kind == "ehp":
            x = k * ehp_e
        else:
            x = B
        x = x * hits * min(tgt, G_["N"])
        return x * crit if c[5] else x   # 原文"暴击时才触发"的分量按暴击率兑现

    aa_k = m.get("aa_k", main_k)
    base_aps = p.get("attackSpeed", 6000) / 10000
    aspd_static = m.get("aspd", 0) + m.get("aspd_per_lv", 0) * lv + G_["item_as"] * n_items
    E = m.get("E", p.get("energy", 0))
    init = E if m.get("init_full") else min(p.get("initEnergy", 0), E)
    chan = m.get("chan", 0)
    lock = m.get("lock", 0)
    W = G_["T"]

    def run(window):
        # 两轮迭代：增益覆盖率依赖施法周期，周期又依赖攻速
        aps = base_aps * (1 + aspd_static)
        cycle = t0 = None
        for _ in range(3):
            rate = G_["mana_per_aa"] * aps + m.get("team_mana", 0) * G_["team_casts"]
            if E <= 0 or rate <= 0:
                cycle = t0 = None
                break
            need = E * m.get("cyc", 1.0)
            cycle = need / rate + chan + lock
            t0 = max(0.0, (E - init) / rate)
            ba = m.get("buf_as")
            up = min(1.0, ba[1] / cycle) if ba else 0
            aps = base_aps * (1 + aspd_static + (ba[0] * up if ba else 0))
        casts = []
        if cycle is not None:
            t = t0
            while t <= window + 1e-9:
                casts.append(t)
                if m.get("once"):
                    break
                t += cycle
        ncast = len(casts)
        bt = m.get("buf_atk")
        atk_mult = 1 + (bt[0] * min(1.0, bt[1] / cycle) if (bt and cycle) else (bt[0] if bt else 0))
        # 普攻
        aa_time = max(0.0, window - ncast * chan)
        aa_hit = (m.get("aa", 1.0) * (pa if aa_k == "pa" else ma) * atk_mult) * cm
        aa_hit += sum(val(c, atk_mult) for c in m.get("aa_extra", [])) * (min(1.0, m.get("aa_extra_uptime", 99) / cycle) if cycle else 1)
        aa_hit = aa_hit * min(m.get("aa_t", 1.0), G_["N"]) + m.get("aa_ehp", 0) * ehp_e * m.get("aa_t", 1.0)
        dmg = aa_time * aps * aa_hit
        # 技能
        per_cast = 0.0
        for c in m.get("sk", []):
            per_cast += val(c, atk_mult)
        if m.get("skill_crit"):
            per_cast *= cm
        dmg += per_cast * ncast
        # 召唤物
        pet = m.get("pet")
        if pet and casts:
            n_alive = 0.0
            spawns = []
            for t in casts:
                spawns.append((t, pet["n"]))
            # 每个时刻存活数 = min(cap, 期间生成且未到期的数量)
            steps = 200
            dt = window / steps
            pv = (pet.get("B", 0) + pet["k"] * (pa if pet["kind"] == "pa" else ma)) * pet["aps"] * min(pet.get("tgt", 1), G_["N"])
            for i in range(steps):
                tt = (i + .5) * dt
                alive = sum(nn for (t, nn) in spawns if t <= tt < t + pet.get("dur", 999))
                dmg += min(alive, pet["cap"]) * pv * dt
        for c in m.get("pas", []):
            dmg += val(c) * window
        for c in m.get("open", []):
            dmg += val(c)
        return dmg, ncast, aps, cycle

    # 存活：EHP / 承伤
    exp_ = exposure(card, m) * (1 - surv.get("hide", 0))
    s_inc = 1 + G_["g"] * max(G_["enemy_lv"], lv * G_["inc_follow"])
    inc = G_["inc0"] * s_inc * exp_ * (1 - surv.get("dr", 0))
    per_cast_s = (surv.get("sh", 0) + surv.get("hl", 0)) * s ** 0 + (surv.get("shp", 0) + surv.get("hlp", 0)) * hp \
        + (surv.get("sha", 0) + surv.get("hla", 0)) * (pa if main_k == "pa" else ma)
    alive = W
    for _ in range(4):
        _, nc, _, _ = run(alive)
        ehp_tot = hp * (1 + surv.get("rev", 0)) + nc * per_cast_s
        alive = min(W, ehp_tot / inc) if inc > 0 else W
    dmg, nc, aps, cycle = run(alive)
    return dict(dmg=round(dmg, 1), dps=round(dmg / W, 2), alive=round(alive, 2), surv=round(alive / W, 3),
                casts=nc, aps=round(aps, 3), cycle=round(cycle, 2) if cycle else None)


def factors(card, spec):
    p = card["properties"]
    m100 = merged(spec, 100)
    kinds = collections.Counter()
    for c in m100.get("sk", []):
        kinds[c[2]] += 1
    tg = max([c[4] for c in m100.get("sk", [])] + [m100.get("aa_t", 1)])
    dmg_type = "物理" if p.get("phyAttack", 0) > p.get("magAttack", 0) else ("法术" if p.get("magAttack", 0) > p.get("phyAttack", 0) else "混合")
    if card["name"] in ("亚连", "露娜", "艾琳"):
        dmg_type = "混合"
    return dict(dmg_type=dmg_type, range=p.get("attackDistance"), HP=p.get("HP"), mana=f"{p.get('initEnergy')}/{p.get('energy')}",
                attack_speed=p.get("attackSpeed", 0) / 10000, crit=p.get("criticalRate", 0) / 10000,
                flash="闪现" in (card.get("skill") or ""), max_targets_lv100=round(min(tg, G["N"]), 2),
                single_target=bool(m100.get("sk")) and all(c[4] <= 1.0 for c in m100.get("sk", [])) and m100.get("aa_t", 1) <= 1,
                exposure=round(exposure(card, m100), 3), skill_fix_only=all(c[2] == "fix" for c in m100.get("sk", [])) and bool(m100.get("sk")))


# ───────────────────────── 历史校准 ─────────────────────────
def history():
    """每局：所有引用到的 live_* 目录合并，取准备阶段每回合出现最多的 carry；
    口径 = 第 5 回合起"当主C 回合数最多"的那位（并列取更晚的），前 4 回合常是占位主C 不算。"""
    games = {}
    for f in glob.glob(str(ROOT / "data/logs/*.log")):
        t = Path(f).read_text(encoding="utf-8", errors="replace")
        rk = re.findall(r"本局名次：第 (\d) 名", t)
        if not rk:
            continue
        runs = tuple(dict.fromkeys(re.findall(r"live_\d{10}", t)))
        if runs:
            games[runs] = int(rk[-1])
    for f in glob.glob(str(ROOT / "data/laya/iterations/*_matches.jsonl")):
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get("run") and x.get("rank"):
                rr = x["run"] if isinstance(x["run"], list) else [x["run"]]
                if not any(r in g for g in games for r in rr):
                    games[tuple(rr)] = int(x["rank"])
    out = []
    for runs, rank in games.items():
        per = collections.defaultdict(collections.Counter)
        for r in runs:
            sp = ROOT / "data/logs" / r / "states.jsonl"
            if not sp.exists():
                continue
            for line in sp.open(encoding="utf-8", errors="replace"):
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                st = x.get("state") or {}
                if st.get("phase") != "prepare":
                    continue
                c, rd = x.get("carry"), st.get("round_no") or 0
                if c and rd:
                    per[rd][c] += 1
        if not per:
            continue
        rc = {rd: cn.most_common(1)[0][0] for rd, cn in per.items()}
        late = [c for rd, c in rc.items() if rd >= 5] or list(rc.values())
        held = collections.Counter(late)
        top = max(held.items(), key=lambda kv: (kv[1], max(rd for rd, c in rc.items() if c == kv[0])))[0]
        out.append(dict(runs=list(runs), rank=rank, carry=top, rounds=max(rc)))
    agg = {}
    for gm in out:
        a = agg.setdefault(gm["carry"], dict(games=0, ranks=[]))
        a["games"] += 1
        a["ranks"].append(gm["rank"])
    res = {k: dict(games=v["games"], avg_rank=round(sum(v["ranks"]) / len(v["ranks"]), 2),
                   top3=sum(1 for r in v["ranks"] if r <= 3)) for k, v in agg.items()}
    return res, out


def lineup_pop():
    d = json.load(open(ROOT / "data/lineups.json", encoding="utf-8"))["lineups"]
    cnt, use = collections.Counter(), collections.Counter()
    for x in d:
        cnt[x["carry"]] += 1
        use[x["carry"]] += int(x.get("use_num") or 0)
    return cnt, use


def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    ma_, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma_) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma_) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def compute(cards, g=None):
    sims = {}
    for name, spec in SPEC.items():
        sims[name] = {lv: simulate(cards[name], spec, lv, g) for lv in LEVELS}
    ref = {}
    for lv in LEVELS:
        vals = sorted(v[lv]["dmg"] for v in sims.values())
        ref[lv] = vals[int(0.9 * (len(vals) - 1))]
    raws = {}
    for name, v in sims.items():
        raw = sum(W_LV[lv] * v[lv]["dmg"] / ref[lv] for lv in LEVELS)
        L = max(0.0, min(1.0, realize(v, g)))
        raws[name] = raw * (1 - R_W + R_W * L)
    xs = sorted(raws.values())
    x90 = xs[int(0.9 * (len(xs) - 1))]
    scores = {n: (round(min(10.0, 8.5 * x / x90), 2), round(x, 4)) for n, x in raws.items()}
    return sims, scores, ref


def tier_of(s):
    for cut, t in TIER_CUT:
        if s >= cut:
            return t
    return "D"


def realize(v, g=None):
    """等级可兑现度：主C 从 100 级喂到 200 级（没有新强化），有效伤害涨了多少，相对"全部伤害都随属性涨"的理想值。"""
    gg = (g or {}).get("g", G["g"])
    s100, s150 = 1 + gg * 100, 1 + gg * 150
    if v[100]["dmg"] <= 0:
        return 0.0
    return round(math.log(v[150]["dmg"] / v[100]["dmg"]) / math.log(s150 / s100), 3)


FORMULA = (
    "主C 等级 lv ∈ {15,40,70,100,150}；atk(lv)=atk0×(1+g·lv)×(1+0.2×装备件数)，HP(lv)=HP0×(1+g·lv)，g=0.02；敌方单位固定 25 级。"
    "一场 T=20 秒：法力每次普攻 +10 → 首次施放 t0=(E−init)/(10·aps)，周期=E/(10·aps)+引导+锁蓝；"
    "伤害 = 普攻(倍率×攻击×暴击期望×目标数)×可普攻时间 + Σ施放×Σ分量[(固定+系数×属性)×段数×min(目标数,6)] + 召唤物 + 持续伤害；"
    "存活时间 = min(T, (HP(lv)×(1+复生)+施放次数×每次护盾/回复) / (25×(1+g·max(25,lv))×暴露×(1−减伤)×(1−不可选中)))，"
    "承伤随敌方主C 等级走（取与我方主C 同级），暴露：射程≥4→0.45、2~3→0.7、近战→1.0，闪现×1.1，技能突进/嘲讽/远离另乘系数；"
    "只累计存活时间内打出的伤害 = 有效伤害 D(lv)。"
    "装备件数：40 级 1 件、70 级 2 件、≥100 级 3 件，每件另加 10% 攻速；"
    "raw = Σ w(lv)·D(lv)/P90_D(lv)，w = {15:0.35, 40:0.24, 70:0.18, 100:0.14, 150:0.09}（第 8 回合起我方主C 处在各等级段的实测回合占比）；"
    "兑现度 L = ln(D(150)/D(100)) / ln(atk(150)/atk(100))（100→150 没有新强化，固定值与按敌方生命百分比的伤害不随主C 涨）；"
    "x = raw×(0.75+0.25·min(1,L))；score = min(10, 8.5·x/P90_x)。"
    "分档：S≥8.5 A≥7 B≥5.5 C≥4 D<4。"
)


def build():
    cards = load_cards()
    missing = [n for n in cards if n not in SPEC]
    assert not missing, f"SPEC 缺英雄：{missing}"
    sims, scores, ref = compute(cards)
    hist, games = history()
    cnt, use = lineup_pop()
    rev = json.load(open(ROOT / "data/hero_carry_reviewed.json", encoding="utf-8"))
    heroes = {}
    for name in SPEC:
        v = sims[name]
        sc = scores[name][0]
        h = hist.get(name, dict(games=0, avg_rank=None, top3=0))
        fac = factors(cards[name], SPEC[name])
        fac["level_realization"] = realize(v)
        fac["raw"] = scores[name][1]
        fac["survival_lv100"] = v[100]["surv"]
        heroes[name] = dict(score=sc, tier=tier_of(sc), quality=cards[name].get("quality"),
                            lv40=v[40], lv100=v[100], levels={str(k): v[k] for k in LEVELS}, factors=fac,
                            evidence=evidence(cards[name]), notes=SPEC[name].get("notes", []),
                            history=h, lineups=dict(as_carry=cnt.get(name, 0), use_num_wan=round(use.get(name, 0) / 1e4)),
                            reviewed_carry_score=rev.get(name, {}).get("carry_score"))
    doc = dict(version="2026-10-01", formula=FORMULA, assumptions=G, level_weights=W_LV, p90_reference=ref,
               tier_cut={t: c for c, t in TIER_CUT}, history_rule="第 5 回合起当主C 回合数最多的英雄（并列取更晚的）；"
               "局来源：data/logs/*.log 的「本局名次」+ data/laya/iterations/*_matches.jsonl，按 live_* 目录去重",
               history_games=len(games), heroes=heroes)
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(doc, cards, games)
    print(f"写入 {OUT_JSON} 与 {OUT_MD}")


def sens():
    cards = load_cards()
    _, base, _ = compute(cards)
    names = list(base)
    b = [base[n][0] for n in names]
    out = []
    for lab, g in [("g=0.01", dict(g=0.01)), ("g=0.05", dict(g=0.05)), ("T=15s", dict(T=15)), ("T=30s", dict(T=30)),
                   ("回蓝 7/普攻", dict(mana_per_aa=7)), ("回蓝 15/普攻", dict(mana_per_aa=15)), ("承伤 ×1.5", dict(inc0=56)),
                   ("敌方 50 级", dict(enemy_lv=50))]:
        _, sc, _ = compute(cards, g)
        x = [sc[n][0] for n in names]
        low = [n for n in ("镜", "马超") if tier_of(sc[n][0]) in "ABS"]
        out.append((lab, round(spearman(b, x), 3), sc["镜"][0], sc["马超"][0], sc["百里守约"][0], low))
    return out


def write_md(doc, cards, games):
    H = doc["heroes"]
    order = sorted(H, key=lambda n: -H[n]["score"])
    L = []
    L.append("# 英雄当阿离主C 的输出强度（2026-10-01）\n")
    L.append("> 生成：`python -X utf8 tools/check_hero_strength.py --build`（同时写 `data/hero_strength.json`）。自检：不带参数运行。")
    L.append("> 起因：镜、马超这类主C 输出偏弱，主C 强度需要按卡面推演审查，而不是写死结论。\n")
    L.append("## 1. 阿离的主C 要什么\n")
    L.append("阿离是单C 打法：枫叶舞只能绑 1 人；红叶染锋/遗志/传承把别人的等级搬给主C；古币、加等级牌、装备都喂给主C。"
             "所以主C 的价值 = **吃到等级和装备后能打出多少伤害、能不能活着打出来**。"
             "队友触发的增益（牺牲减法力、古币攻速、图腾攻速、控制回蓝）在单C 打法里不稳定，基础分**一律不计**，"
             "只在第 7 节作为\"路线加成\"单独列出。\n")
    L.append("## 2. 方法\n")
    L.append("1. **技能抽取**：每个英雄按 1/10/40/100 级强化写成\"每次施放的伤害分量（固定值, 攻击系数, 按什么缩放, 段数, 目标数）\""
             "+ 普攻倍率 + 自保（护盾/回复/减伤/复生/不可选中）。抽取表在 `tools/check_hero_strength.py` 的 `SPEC`，"
             "**只抽数字，公式里没有任何英雄名特判**；原文依据由脚本从 `data/cards.json` 自动附进 JSON 的 `evidence`，"
             "我的解读（镜像怎么算、平均覆盖率等）写在 `notes`。")
    L.append("2. **统一模拟**一场 20 秒战斗，主C 分别为 15 / 40 / 70 / 100 / 150 级，身上 0~3 件输出装（敌方单位固定 25 级，敌方主C 与我方同级）。")
    L.append("3. **只算活着打出来的伤害**（有效伤害 D）。")
    L.append("4. **评分**：五个等级下 D 相对全体第 90 百分位、按实测等级分布加权平均，再乘等级可兑现度修正，线性映射到 0~10（第 90 百分位 = 8.5）。\n")
    L.append("### 公式\n")
    L.append("```")
    for seg in doc["formula"].split("；"):
        L.append(seg)
    L.append("```\n")
    L.append("目标数常量（6 个敌人、抱团）：单体 1、附近(不含主目标) 1、目标及附近 1 格 2、自身周围 1 格 2.5、周围 2 格 3、沿途/穿透 2、一整排 2.5、密集处 2.5、全场 6。\n")
    L.append("### 假设（卡牌库没给的，都对所有英雄相同）\n")
    L.append("| 假设 | 值 | 依据 |\n|---|---|---|")
    L.append("| 每级属性成长 g | 0.02（100 级 = 3 倍基础） | 卡牌库没有成长表；对所有英雄相同，只影响\"固定值 vs 攻击系数\"的相对权重 |")
    L.append("| 每次普攻回蓝 | 10 | 常见自走棋设定；守约 80 法力、镜 20 法力的差距靠它体现 |")
    L.append("| 承伤基准 | 25 HP/s（暴露 1.0、同级） | 自洽推算：本模型 85 名英雄 1 级 DPS 中位数 24/s × 敌方 6 人 ÷ 我方暴露当量 4.35 × 边打边减员 0.75 |")
    L.append("| 承伤随谁的等级涨 | 敌方主C 与我方主C 同级 | `data/enemy_levels.json`：对手最高等级与我方主C 同量级甚至更高 → 存活时间基本不随等级变，只有按攻击/生命比例的自保会涨 |")
    L.append("| 装备 | 40 级 1 件、70 级 2 件、≥100 级 3 件；每件 +20% 攻击、+10% 攻速 | 阿离把装备都喂给主C；固定伤害吃不到攻击加成 |")
    L.append("| 战斗时长 | 20 秒 | 经验值；`--sens` 看 15/30 秒 |")
    L.append("| 敌方单位等级 | 25 | `data/enemy_levels.json`：第 10 回合前后对手非主C 单位多在 20~30 级 |")
    L.append("| 暴击 | 期望 = 1 + 暴击率 × criticalEffect/10000（=0.4） | 只对普攻；技能写明可暴击才算 |\n")
    L.append("敏感性（`--sens`，与默认排名的 Spearman 相关；以及镜/马超/守约在各假设下的分数）：\n")
    L.append("| 假设变化 | Spearman | 镜 | 马超 | 百里守约 | 镜/马超 跳出 C/D |\n|---|---|---|---|---|---|")
    for lab, r, a, b, c, low in sens():
        L.append(f"| {lab} | {r} | {a} | {b} | {c} | {'、'.join(low) or '无'} |")
    L.append("")
    L.append("## 3. 输出位排名（全部 85 名英雄）\n")
    L.append("各等级列 = 一场 20 秒里活着打出的有效伤害（括号里是存活秒数）；兑现度 = 从 100 级喂到 150 级（没有新强化）伤害涨幅 / 理想涨幅（1 = 伤害全部随属性涨；"
             "固定伤害、按敌方生命百分比的伤害、召唤物固定值会拉低它）。历史 = 当主C 的局数 / 平均名次 / 前三次数。\n")
    L.append("| # | 英雄 | 阶 | 分 | 档 | lv15 | lv40 | lv70 | lv100 | lv150 | 兑现度 | 射程 | 历史 | 官网当主C 套数/万人 | 现评分 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, n in enumerate(order, 1):
        h = H[n]
        hi = h["history"]
        hs = f"{hi['games']}局/{hi['avg_rank']}/{hi['top3']}" if hi["games"] else "—"
        lvs = " | ".join(f"{h['levels'][str(l)]['dmg']:.0f}（{h['levels'][str(l)]['alive']:.0f}s）" for l in LEVELS)
        L.append(f"| {i} | {n} | {h['quality']} | {h['score']} | {h['tier']} | {lvs} | {h['factors']['level_realization']} | "
                 f"{h['factors']['range']} | {hs} | {h['lineups']['as_carry']}/{h['lineups']['use_num_wan']} | "
                 f"{h['reviewed_carry_score']} |")
    L.append("")
    # 历史校准
    L.append("## 4. 历史校准\n")
    L.append(f"口径：{doc['history_rule']}。共 {doc['history_games']} 局有名次且有逐帧记录。"
             "**注意混杂**：主C 是谁由路线规则决定，路线又和禁用派系、版本代码相关（早期局 bug 多、名次普遍差），"
             "所以历史只用来检验方向，不回灌进分数。\n")
    allr = [g["rank"] for g in games]
    L.append(f"全体平均名次 {sum(allr) / len(allr):.2f}，前三率 {sum(1 for r in allr if r <= 3) / len(allr):.0%}。\n")
    L.append("| 主C | 局数 | 平均名次 | 前三 | 前三率 | 模型分 | 档 |\n|---|---|---|---|---|---|---|")
    hs = sorted([(n, H[n]["history"]) for n in H if H[n]["history"]["games"]], key=lambda kv: (kv[1]["avg_rank"], -kv[1]["games"]))
    for n, hi in hs:
        L.append(f"| {n} | {hi['games']} | {hi['avg_rank']} | {hi['top3']} | {hi['top3'] / hi['games']:.0%} | {H[n]['score']} | {H[n]['tier']} |")
    big = [(n, hi) for n, hi in hs if hi["games"] >= 3]
    if len(big) >= 3:
        r = spearman([H[n]["score"] for n, _ in big], [-hi["avg_rank"] for _, hi in big])
        L.append(f"\n局数 ≥3 的 {len(big)} 名主C：模型分 与 (−平均名次) 的 Spearman = **{r:.2f}**。")
    cnt_pairs = [(n, H[n]["lineups"]["use_num_wan"]) for n in H if H[n]["lineups"]["as_carry"] > 0]
    r2 = spearman([H[n]["score"] for n, _ in cnt_pairs], [u for _, u in cnt_pairs])
    L.append(f"官网阵容里当过主C 的 {len(cnt_pairs)} 名英雄：模型分 与 当主C 使用人数 的 Spearman = **{r2:.2f}**"
             "（热度混着棋手专属玩法，比如司空震/艾琳/李信是别的棋手的核心，只作弱参照）。\n")
    L.append(HAND_MD_CALIB)
    # 与现评分差异
    L.append("## 5. 与 `data/hero_carry_reviewed.json` 现有评分差异大的条目\n")
    L.append("对照方法：模型分（0~10）直接和现评分比，差 ≥3 的逐条列出；\"建议\"一列是我结合模型分、历史与原文给的人工建议值"
             "（不是模型分四舍五入 —— 现评分还承担\"能不能当主C\"的资格语义）。\n")
    L.append("| 英雄 | 现评分 | 模型分 | 建议 | 理由 |\n|---|---|---|---|---|")
    for n in sorted(H, key=lambda n: -(abs(H[n]["score"] - (H[n]["reviewed_carry_score"] or 0)))):
        rv = H[n]["reviewed_carry_score"]
        if rv is None or (abs(H[n]["score"] - rv) < 3 and n not in ALWAYS_SHOW):
            continue
        sug, why = SUGGEST.get(n, (None, ""))
        L.append(f"| {n} | {rv} | {H[n]['score']} | {sug if sug is not None else '—'} | {why or auto_why(H[n])} |")
    L.append("")
    roles = json.load(open(ROOT / "data/hero_roles.json", encoding="utf-8"))["heroes"]
    rows = []
    for n in sorted(H, key=lambda n: H[n]["score"]):
        if set(roles.get(n, {}).get("combat_role", [])) & {"tank", "support"}:
            continue
        h = H[n]
        f = h["factors"]
        jump = h["levels"]["100"]["dmg"] > 4.5 * h["levels"]["40"]["dmg"]   # 纯属性+装备成长约 ×2.2~2.6
        if h["score"] < 4.5 or (h["score"] < 5.5 and (f["level_realization"] < .8 or f["survival_lv100"] < .5 or jump)):
            ex = NOT_REC_EXCEPT.get(n)
            rows.append(f"| {n} | {h['score']} | {not_rec_why(h)}{'。**例外**：' + ex if ex else ''} |")
    L.append(HAND_MD_TAIL.replace("{NOT_RECOMMENDED}", "\n".join(rows)))
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def not_rec_why(h):
    f = h["factors"]
    bits = []
    if f["flash"] and f["range"] == 1 and h["levels"]["40"]["alive"] < 12:
        bits.append(f"闪现近战 {f['HP']} 血（40 级只活 {h['levels']['40']['alive']:.0f} 秒）")
    elif f["survival_lv100"] < .5:
        bits.append(f"100 级只活 {f['survival_lv100'] * 20:.0f} 秒")
    if f["level_realization"] < .8:
        bits.append(f"兑现度 {f['level_realization']}（固定值或按敌方生命的伤害多）")
    d40, d100 = h["levels"]["40"]["dmg"], h["levels"]["100"]["dmg"]
    if d100 > 4.5 * d40:
        bits.append(f"伤害集中在 100 级门槛（40 级 {d40:.0f} → 100 级 {d100:.0f}）")
    if f["single_target"]:
        bits.append("单体")
    bits.append(f"40 级有效伤害 {d40:.0f}")
    if h["notes"]:
        bits.append(h["notes"][0])
    return "；".join(bits)


def auto_why(h):
    f = h["factors"]
    bits = []
    if f["level_realization"] < .6:
        bits.append(f"兑现度 {f['level_realization']}（固定值/百分比伤害多）")
    if f["survival_lv100"] < .5:
        bits.append(f"100 级存活 {f['survival_lv100']}")
    bits.append(f"lv100 有效伤害 {h['lv100']['dmg']:.0f}")
    return "；".join(bits)


# 人工结论（写进文档；数字会随 --build 更新，结论段落要人工复核）
SUGGEST = {
    "露娜": (5, "闪现近战 240 血，法力 70 一场只够突 1 次就阵亡；标记 15% 最大生命按**敌方**生命算，喂等级不涨。官网热度来自别的棋手（露娜不在阿离阵容里），阿离单C 下撑不起 8"),
    "韩信": (4, "1 级主动是 100 点固定伤害，法力 65、闪现近战 240 血 → 平均只放 1 次技能；40/100 级伤害靠击杀连锁，单C 喂不出来"),
    "赵云": (4, "模型高分来自技能每段**固定回血**（40/80 点）在 20 秒内几乎不死 + 每段 60 点固定伤害，前期很强；但实战判定为过渡主C（曜/赵云/姬小满同组），且只有 2 阶、单体。**保持 4**，模型在这里偏乐观（固定回血不受敌方治疗削减等机制约束），等实战数据再议"),
    "海月": (4, "作为普通主C 只有 3.3：法攻 20，伤害以固定值为主；现评分 7 是**遗志流**里的身份（牺牲真伤 = 自身生命），而遗志流已被排除（伞海月难操作）"),
    "猪八戒": (1, "保持 1：输出来自自身最大生命百分比，模型分不低只是因为坦克 20 秒一直活着在打；它是坦克 + 夺取件，不是主C"),
    "虞姬": (7, "后排射程 5、冲刺 120% + 4 支弩矢 50+120%、普攻 30% 追加，伤害几乎全是攻击系数（兑现度 0.94）；2 阶好凑。官网只把她当大河开团件用，阿离单C 值得试"),
    "苍": (7, "后排射手 + 常驻狼魂（召唤物也吃主C 攻击力），兑现度 0.99；狼魂基础攻击倍率原文只给了 10 级 120%，模型按 80% 估，有不确定性 → 先给 7 不给 9"),
    "云中君": (3, "闪现近战 240 血；撕裂按**敌方**当前生命算，阿离喂的等级、装备都不涨它（兑现度 0.79）。它真正的价值是卡面『开团：每件装备 +1 永久等级』的全队引擎，不是主C"),
    "李白": (5, "6 次斩击伤害**由范围内敌人平均分摊**（总量不随人数涨），闪现近战 240 血、法力 55 → 平均 1~2 次；强度来自河洛古币临时等级（自身 +7），离开古币体系就是中等"),
    "杨玉环": (7, "法力 20、射程 5、100 级每次施法全场伤害 + 全场治疗；10 级前一半施法只奶不打。固定 100 点占比大（兑现度 0.78），所以给 7 不给 9"),
    "瑶": (1, "保持 1：附身在队友身上、每 2 秒打 2 人 50+50% 法攻（法攻 20），伤害是附带的，不能当主C"),
    "马可波罗": (6, "模型满分来自每颗子弹 40 点**固定伤害** × 16 发（前期很凶），但兑现度 0.71 属于最低一档：阿离喂的等级、攻击装只涨 25% 系数那部分；官网 455 套阵容几乎不用他当主C，实战 1 局第 6 → 下调到 6"),
    "马超": (3, "实战判定：单C 阵容下马超偏弱。模型 3.95（D）：近战突袭 3 格内**最远**敌人（暴露 ×1.15）；40 级的回收与路径伤害**只在暴击时**触发，而基础暴击率 0%、暴击只能靠队友施加控制喂（5%/次）→ 单C 打法下这一档基本兑现不了"),
    "亚连": (7, "保持 7：模型 4.2（C），输出系数偏低（每段 65%）；但实战 27 局平均 3.78、前三 10 次 —— 他的强在**等级来源**（日落海核心 + 自身卡面整备，第 41 局养到 621 级）和 20 层怒气 40% 减伤，不在每级输出。属于『能拿到的等级 × 每级输出』里前一项特别大的类型"),
    "镜": (3, "保持 3：实战判定输出偏弱。模型 4.5（C），见第 6 节逐级推演"),
}
ALWAYS_SHOW = ("马超", "亚连", "镜")
NOT_REC_EXCEPT = {
    "亚连": "路线主C 照常用：实战 27 局前三 37%，强在等级来源（日落海核心）与 40% 减伤，见第 5 节",
    "弈星": "卡面是棋手经验件，本来就不当主C",
    "姜子牙": "5 阶，输出本身不差（B 边缘），扣分来自 140 点固定伤害占比高",
}

HAND_MD_CALIB = """### 4.1 读法与结论

- **方向一致**：实战局数 ≥3 的主C 里，模型分最高的 百里守约（8.3，38 局平均 3.53、前三 19 次）、孙悟空（6.6，3 局 2 次前三）、
  安琪拉（7.3）实战也在前列；模型分最低的 曜（4.2）、百里玄策（4.6）、孙尚香（5.1）、西施（5.1）实战也偏后。
- **最大的反例是 亚连**：模型 4.2，实战 27 局平均 3.78、前三 10 次（37%）。原因不是模型算错了他的输出，
  而是他的强在**拿到多少等级**（日落海核心 + 自身卡面整备喂等级，第 41 局 621 级），这一项本模型不评（见第 8 节"主C 价值 = 等级来源 × 每级输出"）。
- **狄仁杰** 模型 6.0、实战 7 局 0 次前三：他几乎全靠普攻（技能只加攻速），基础分不计河洛古币的永久攻速；实战里古币引擎没转起来（第 45~47 局复盘）。
- 局数 1~2 的（艾琳 2 局 2 次前三、嬴政 1 局第 2、镜 1 局第 6、马可波罗 1 局第 6）样本太少，只作旁证。
- 与官网热度相关性弱（0.12）：热门主C 里 司空震/嬴政 在模型里是 A~S、艾琳/李信 是 B，但 露娜/李白/韩信（闪现近战）模型很低 ——
  官网阵容按全部棋手统计，这几位是别的棋手（以及配套的复生/免伤装备、古币临时等级）撑起来的，阿离单C 没有这些。

### 4.2 与参照冲突的英雄

| 英雄 | 参照 | 模型 | 解释 |
|---|---|---|---|
| 亚连 | 实战常用、前三 37% | 4.2 C | 强在等级来源（核心引擎）与 40% 减伤，不在每级输出系数 |
| 露娜、李白、韩信 | 官网热门主C | 2.2~3.8 D | 闪现近战 240 血，开战独自跳进敌阵；模型没算"先切死对方后排、对面火力下降"的收益，可能偏低 —— 但对阿离单C 来说这正是"能不能活着打出来"的风险 |
| 马可波罗 | 官网几乎不用、实战 1 局第 6 | 10 S | 每颗子弹 40 固定伤害 × 16 发，前期很凶；兑现度 0.71。若对方有按固定值减伤的防御机制，会是最吃亏的一类（未建模） |
| 赵云 | 实战判定过渡主C | 8.4 A | 每段固定回血让他 20 秒内几乎不死；模型对固定回血可能偏乐观 |
| 狄仁杰 | 实战 7 局 0 前三 | 6.0 B | 实战古币引擎没转起来；且他的输出全靠普攻、易被控 |
"""

HAND_MD_TAIL = """## 6. 镜和马超：为什么阿离不适合（逐级推演）

> 为什么阿离不适合用他们当主C：下面的数字全部来自 `data/hero_strength.json`（一场 20 秒、主C 处在该等级时活着打出的伤害）。

| 主C 等级 | 15 | 40 | 70 | 100 | 150 |
|---|---|---|---|---|---|
| 百里守约 | 2073（16s） | 5250（19s） | 7984（19s） | 12992（19s） | 17003（19s） |
| 镜 | 767（8s） | 2393（9s） | 3469（9s） | 15353（20s） | 19671（20s） |
| 马超 | 1288（20s） | 2137（20s） | 3418（20s） | 5938（20s） | 7388（18s） |
| 亚连（对照） | 671（12s） | 2376（17s） | 3843（17s） | 11625（17s） | 15500（17s） |

**镜**：
1. **单体、系数小**：每次「裂空」只打当前目标 60+80% → 80+120% → 80+150%（40 级 +1 镜像），对比守约一枪 200+800%。
   她的输出全靠法力上限 20 → 每 2~3 秒放一次，而这点优势被下一条吃掉。
2. **活不长**：卡面「闪现」→ 开战独自跳到敌方后排镜像位，近战 240 血，模型里 15~70 级只活 8~9 秒（守约 16~19 秒）。
3. **等级只在 100 级兑现**：15→70 级伤害只从 767 涨到 3469，到 100 级才因为「2 名镜像 + 每次施法回 450 血」突然变成 15353。
   而第 8 回合起我方主C **只有 23% 的回合在 85 级以上**（147 局实测）—— 大部分时间她处在"喂了等级也没变强"的区间。
   这正是"伤害只在门槛变化 → 等级换不成输出"。
4. **被动只涨等级不涨伤害**：「每使用 2 张效果牌等级 +1」让她涨级很快，但第 3 条说明这些等级在 100 级前几乎不变成伤害。

**马超**：
1. **核心强化被暴击卡住**：40 级「突袭伤害**暴击时**会回收场上的冷晖枪，并对路径上的敌人造成 50% 物理攻击力」—— 马超基础暴击率 0%，
   唯一的暴击来源是卡面「己方施加控制效果时，马超获得 15 点法力值及 5% 暴击率」，**要队友提供控制**。单C 打法下这一档兑现率≈0。
2. **100 级的 6 连突要先回收 3 把枪**，回收靠走路拾取或第 1 条的暴击 → 平均每 6 次施法才触发一次。
3. **输出本体很小**：突袭 50+150% 单体，每 7 秒一次；拾枪 +5 物攻是固定值（不随等级涨），兑现度 0.76。
4. **站位危险**：突袭到 3 格内**最远**的敌人身边（暴露 ×1.15）。
所以马超是"队友喂控制 → 他回蓝暴击"的**多核阵容件**，不是单C。

**共同规律（写成判据，不写名字）**：阿离主C 要同时满足
① 伤害主要是**攻击系数**（兑现度 ≥0.8）；② 15~70 级就有像样的输出（不是只在 100 级跳变）；③ 关键强化**不依赖队友喂的属性**（暴击、控制、牺牲）；④ 能活着打（后排，或近战但有按比例的自保）。
镜缺 ②④，马超缺 ③（以及输出量），亚连缺的是输出系数但靠等级来源补回来。

## 7. 不建议当阿离主C 的英雄

规则：输出位（hero_roles 不含 tank/support）里模型分 <4.5，或模型分 <5.5 且（兑现度 <0.8 或 100 级存活 <50% 或 100 级伤害 >4.5 倍 40 级 = 门槛型；纯属性+装备成长只有 ×2.2~2.6）。理由按原文：

| 英雄 | 分 | 主要原因 |
|---|---|---|
""" + "{NOT_RECOMMENDED}" + """

坦克、辅助（苏烈、牛魔、廉颇、白起、明世隐、鬼谷子、蔡文姬、庄周、太乙真人…）分数都在 D，本来就不该当主C，不逐个列。
**例外**：弈星（卡面是棋手经验件）、云中君（全队装备等级引擎）、李元芳（古币产出方）的价值在卡面，不在输出，留在场上但不当主C。

### 路线加成（基础分不计、单独列出）

这些英雄有"要队友 / 要路线"才成立的输出增益，路线成型时可以在基础分上加：
- 百里守约：牺牲英雄阵亡 → 法力上限 −20（觉醒 −40）；按模型：每回合死 1 个牺牲英雄（法力 80→60）有效输出 ×1.2~1.4，死 2 个（→40）×1.6~1.9 → 路线成型后是 S。
- 狄仁杰：对他用古币 → 攻速永久 +5%（河洛古币路线每回合 3~5 张）。
- 沈梦溪：每对他用 1 张效果牌 → 本回合法力上限 −15（阿离每回合喂牌，法力 40 → 很快降到 0~10，等于持续施法）。
  这条对阿离特别契合：沈梦溪基础分 5.4（C），按每回合喂 2 张算法力 40→10，模型有效输出 ×3.2~3.7 → 相当于 S。
  **前提是每回合真的有 2 张效果牌打在他身上**（第 45~47 局复盘：河洛路线的牌没喂起来时他只是 C）。
- 公孙离：每个已觉醒英雄攻速 +15%。马超：队友控制 → 法力 + 暴击（多核件）。

## 8. 建议主线程如何接入

1. **主C 资格门槛**（`rules._carry_eligible` / `_carry_score`）：输出位模型分 ≥5.5（B 及以上）才可当主C；
   **例外**：路线指定、且历史前三率 ≥30%（局数 ≥5）的主C（目前只有 亚连）照常；
   4.0~5.5 只在"路线指定 + 路线加成成立"（如沈梦溪有效果牌喂）时可当；<4.0 不当。镜（4.5）、马超（3.95）因此出局，不需要写名字。
   `_carry_score` 里的人工 carry_score 可以换成 `hero_strength.score`（或两者取平均过渡一段时间）。
2. **主C 价值 = 等级来源 × 每级输出**：本模型只评后一项。路线排序建议 `路线分 = 历史前三率（局数 ≥5）优先，其次 模型分 × 等级引擎完成度`。
   按现有数据：守约路线（8.3 / 前三 50%）> 亚连路线（4.2 但前三 37%，等级来源强）> 安琪拉（7.3）> 沈梦溪（5.4，配喂牌加成）> 狄仁杰（6.0，实战 0 前三，排后）> 玄策（4.6）。
3. **2~3 阶定调**（`routes.CARRY_TIERS`）里模型分高、实战还没试过的：苍（3 阶 9.3）、虞姬（2 阶 8.5）、貂蝉（3 阶 6.8）、米莱狄（2 阶 6.7）。
   苍/虞姬 官网没有阿离路线，要上需要先在 `data/custom_lineups.json` 配路线再实测（**不要直接改规则**）。
4. **`data/hero_carry_reviewed.json` 要改的**：见第 5 节"建议"列 —— 主要是 马超 6→3、海月 7→4、露娜 8→5、韩信 7→4、云中君 6→3、李白 7→5、马可波罗 7→6、虞姬 5→7、苍 6→7、杨玉环 6→7；
   赵云/亚连/镜/猪八戒/瑶 保持。改之前跑一遍冒烟测试（`_carry_eligible` 依赖它）。
5. **给 Laya / cgnet 的特征**：`score`、`factors.level_realization`、`levels.*.alive` 可直接作为 carry 题的英雄特征（挂新题面版本号，别塞进现有版本）。

## 9. 局限

- 卡牌库没有属性成长表、法力获取规则、伤害公式（防御怎么算）—— 用了统一假设，`--sens` 显示排名对这些假设很稳（Spearman ≥0.93），但绝对值不可信。
- 没有建模：控制（眩晕/击飞/嘲讽）、减防/易伤的团队收益、击杀后的连锁（守约穿透、韩信冲锋按平均处理）、对方切后排、觉醒卡面加成。
- 召唤物（苍的狼魂、米莱狄的仆从、蒙恬的士兵）攻击倍率原文没写全，用了估计值，见 JSON `notes`。
- 历史校准有路线混杂：谁当主C 是规则决定的，不是随机分配。
"""


# ───────────────────────── 自检 ─────────────────────────
HOT_CARRIES = ("百里守约",)   # 官网阿离阵容热度第一 + 实战局数最多、前三最多


def check():
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print("✗", msg)

    doc = json.load(open(OUT_JSON, encoding="utf-8"))
    cards = load_cards()
    H = doc["heroes"]
    miss = [n for n in cards if n not in H]
    if miss:
        fail(f"覆盖不全：{miss}")
    extra = [n for n in H if n not in cards]
    if extra:
        fail(f"多出非英雄：{extra}")
    for n, h in H.items():
        if not (0 <= h["score"] <= 10):
            fail(f"{n} 分数越界 {h['score']}")
        if h["tier"] not in "SABCD":
            fail(f"{n} 档位异常 {h['tier']}")
        if h["tier"] != tier_of(h["score"]):
            fail(f"{n} 档位与分数不符")
        if not h["evidence"]:
            fail(f"{n} 没有原文依据")
        for k in ("lv40", "lv100", "levels", "factors", "history"):
            if k not in h:
                fail(f"{n} 缺 {k}")
    for n in ("镜", "马超"):
        if H[n]["tier"] not in ("C", "D"):
            fail(f"实战判定弱的 {n} 不在低分段：{H[n]['score']} {H[n]['tier']}")
    for n in HOT_CARRIES:
        if H[n]["tier"] not in ("S", "A"):
            fail(f"热门主C {n} 不在高分段：{H[n]['score']} {H[n]['tier']}")
    # 实战多次进前三（≥3 局且前三率 ≥40%）的主C 应在 B 档以上
    for n, h in H.items():
        hi = h["history"]
        if hi["games"] >= 3 and hi["top3"] / hi["games"] >= .4 and h["tier"] not in ("S", "A", "B"):
            fail(f"实战常进前三的 {n}（{hi}）落在 {h['tier']}")
    # 公式一致性：重算与文件一致
    sims, scores, _ = compute(cards)
    diff = [n for n in H if abs(scores[n][0] - H[n]["score"]) > 1e-6]
    if diff:
        fail(f"JSON 与当前 SPEC/公式不一致（需 --build）：{diff[:8]}")
    # 纯辅助/坦克不应进 A 以上
    for n in ("鬼谷子", "蔡文姬", "明世隐", "庄周", "白起"):
        if H[n]["tier"] in ("S", "A", "B"):
            fail(f"无输出的 {n} 分数过高 {H[n]['score']}")
    print("自检", "通过" if ok else "失败", f"（{len(H)} 名英雄）")
    return ok


if __name__ == "__main__":
    if "--sens" in sys.argv:
        for row in sens():
            print(row)
        sys.exit(0)
    if "--build" in sys.argv:
        build()
    sys.exit(0 if check() else 1)
