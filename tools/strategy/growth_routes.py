"""**养成引擎表**：主 agent 逐张读卡牌库里 281 张"和等级有关"的卡后写的产出估计（2026-09-24：
能让英雄升级的卡非常多，其他英雄卡、效果卡的效果都能成为阵容养成的套路）。

  python tools/strategy/growth_routes.py      # 生成 data/strategy/growth_routes.json 并按套路打印汇总

每张卡给出**在"标准中期场面"下**的产出：
  **永久等级**会逐回合累积；**临时等级**（战斗中临时加的等级）只在当场战斗有效、
  打完清空 —— 不累积，但当场战斗力按全额算（`temp`）；明世隐能把其中一部分转成永久。
  once   = 一次性：给主C 多少级 / 全场合计多少级
  per    = 每回合：给主C 多少级 / 全场合计多少级（持续产出，越早拿越值钱）
  temp   = 每场战斗的临时等级：主C / 全场（只加当场战斗力；战斗中逐步叠的按平均值）
  core / core_once = 每回合 / 一次性给**阿科米亚核心**加多少级（日落海套路，2026-09-24 确认：核心一直在场边、
           只加不减，**每场开战把累积的等级分给场上的日落海英雄**；定向传输 = 优先集中给装备最多的日落海英雄，至多 200）
  xp / energy = 棋手经验 / 能量
  need   = 生效条件（阵营/标签/别的卡），模型里按"满足的概率"打折
  route  = 归到哪个养成套路

标准中期场面（估计，写死在这里，可改）：人口 6、主C 每场放 6 次技能、主C 穿 2 件装备、全场 6 件装备、
每回合打出 2 张英雄牌（2 次登场）、场上 2 个整备英雄、1 个开团英雄、每场全队约 90 次普攻、主要阵营 4 人、
场上 3 个阵营、场上品阶合计 18、棋手 4 级、血量 45。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POP, CASTS, EQ_C, EQ_ALL, PLAYS, PREP, OPEN, AUTOS, MAIN_FAC, N_FAC, TIER_SUM, LORD, HP = \
    6, 6, 2, 6, 2, 2, 1, 90, 4, 3, 18, 4, 45
HALF = 0.5          # 战斗中逐步叠加的临时等级，按一场里的平均值算

# 名字: dict(route=, once=(主C, 全场), per=(主C, 全场), xp=, energy=, need=, why=)
R: dict[str, dict] = {
    # ================= 效果牌（效果锦囊 N 选 1 / 商店）=================
    "古币": dict(route="河洛古币", once=(2, 2), need="河洛", why="转瞬，指定河洛 +2"),
    "精致古币": dict(route="河洛古币", once=(4, 4), need="河洛", why="转瞬，指定河洛 +4"),
    "演练战术": dict(route="战术牌", once=(1, POP), why="转瞬，所有英雄 +1"),
    "特训战术": dict(route="战术牌", once=(4, 4), why="转瞬，指定 +4"),
    "高阶演练战术": dict(route="战术牌", once=(2, 2 * POP), why="所有 +2"),
    "高阶特训战术": dict(route="战术牌", once=(8, 8), why="指定 +8"),
    "简易协作": dict(route="单点加级", once=(3, 5), why="指定 +3，随机 +2"),
    "精密协作": dict(route="单点加级", once=(4, 7), why="指定 +4，随机 +3"),
    "红笺锦鲤符": dict(route="单点加级", once=(4, 4), why="指定 +4"),
    "八方来援": dict(route="单点加级", once=(2 * (1 + N_FAC), 2 * (1 + N_FAC)), why="+2，每种阵营再触发 1 次"),
    "神树彩陶": dict(route="全体加级", once=(2 * 4 / POP, 8), why="随机 4 人 +2"),
    "绞弦": dict(route="全体加级", once=(1 / POP, 1), why="随机 1 人 +1"),
    "扫拂": dict(route="全体加级", once=(0, 2 * LORD), why="最低 2 人 + 棋手等级（给不到主C）"),
    "武装之力": dict(route="装备流", once=(EQ_C, EQ_ALL), why="所有英雄 + 自身装备数"),
    "阵营祝福": dict(route="全体加级", once=(4 / 2, 4 * N_FAC), why="各阵营随机 1 人 +4"),
    "扶倾济弱": dict(route="全体加级", once=(0, 9), why="最低的 +9"),
    "逍遥自在": dict(route="无阵营", once=(4, 8), need="无阵营", why="无阵营 +2×无阵营人数（按 2 人）"),
    "战前宣誓": dict(route="临时等级", temp=(3 * 4 / 5, 3 * 4 / 5), turns=1, why="一次性：主C 品阶×3 临时等级（一场）"),
    "有福同享": dict(route="全体加级", once=(0, 6), why="同品阶的其他英雄 +3（主C 本人不加）"),
    "潮汐的交响": dict(route="全体加级", once=(8 * 5 / POP, 40), why="随机 5 人 +8"),
    "马可的魅力飞吻": dict(route="全体加级", once=(10, 10 * POP), why="全体 +10"),
    "众星闪耀": dict(route="全体加级", once=(15, 15 * POP), why="全体 +10，持有每回合 +5（按留 1 回合）"),
    "命运馈赠·阳": dict(route="每回合全体", per=(5, 5 * POP), turns=3, why="回合开始全体 +5，持续 3 回合"),
    "低压运营": dict(route="每回合全体", per=(HP // 15, HP // 15 * POP), why="每 15 血全体 +1"),
    "冰心领域": dict(route="每回合全体", per=(TIER_SUM * .3, TIER_SUM * .3 * POP), why="全体 + 场上品阶总数 30%"),
    "寒潮共振": dict(route="觉醒流", per=(1.5, 1.5 * POP), why="全体 + 觉醒数×1.5（按 1 个觉醒）"),
    "寒雨凝结": dict(route="阵营堆叠", per=(MAIN_FAC, MAIN_FAC * POP), why="全体 + 主要阵营人数"),
    "协同增效": dict(route="合成流", per=(4, 8), why="合成额外 +4（按每回合合成 2 次，主C 1 次）"),
    "基层提拔": dict(route="低阶流", per=(0, 3 * PLAYS), why="打出 1-3 阶英雄牌 +3"),
    "破境之证": dict(route="单核", once=(20, 20), per=(10, 10), why="立即 +20，之后每回合 +10"),
    "封神一瞬": dict(route="单核", per=(4, 4), why="主C 每次永久升级额外 +1（按每回合升 4 次）"),
    "短期企划": dict(route="临时等级", per=(5 * .7, 5 * POP * .5), temp=(5, 5 * POP), why="战斗开始 10 临时，活下来留 5"),
    "图腾祝福": dict(route="大河图腾", once=(0, 10), need="大河流域", why="随机大河 +5 × (1+图腾数)"),
    "助力核心": dict(route="日落海核心", core_once=2 * 4, need="日落海", why="核心 +2 × 日落海人数（按 4 人）"),
    "跳级": dict(route="日落海核心", once=(5, 5), need="日落海", why="夺取核心至多 5 级（永久）"),
    "研读会": dict(route="棋手经验", xp=3), "经典研读会": dict(route="棋手经验", xp=4),
    "管理进修": dict(route="棋手经验", xp=6, why="+3，升级时再 +3"),
    "战略性收缩": dict(route="棋手经验", xp=9 * .5, why="输 2 场给 9 经验（按一半概率）"),
    "投资增值": dict(route="经济", energy=4, why="升级时 +2 能量"),
    "沉淀收益": dict(route="经济", per_energy=2.5, why="停在当前等级每回合多 1，至多 4"),
    "扶摇直上": dict(route="棋手经验", xp=20, why="棋手等级 +1（满级 10 能量）"),
    "人才储备": dict(route="经济", per_energy=2, why="4 级起每回合一张人才搜寻"),
    "学识宝石": dict(route="单核", per=(4, 4), why="装备：每回合 +8%（至少 3）—— 复利；中期约 +4"),
    # 搬等级：不产生新等级，价值在"换主C 不亏"—— 模型里记 0，另算
    "遗志": dict(route="搬等级", why="移除一人，等级随机给一人"), "传承": dict(route="搬等级"),
    "夺取之术": dict(route="搬等级", once=(8, 0), why="主C 夺取邻位至多 10 级（全场总量不变）"),
    "众星拱月": dict(route="搬等级", once=(40, 0), why="主C 夺取其他人各 10 级"),
    "红叶染锋": dict(route="搬等级", why="枫叶舞换绑保留 50%"),
    "枫舞扶摇": dict(route="枫叶舞", temp=(5 + 2.5, 5 + 2.5), why="转移后 5 临时等级，本局每击杀 +1"),
    "翩跹舞步·新生": dict(route="枫叶舞", per=(2, 2), why="枫叶舞英雄每回合 +2"),
    "翩跹舞步·蓄势": dict(route="枫叶舞", per=(2.5, 2.5), why="枫叶舞英雄每击杀永久 +1（按每回合 2.5 杀）"),
    "化铁为金": dict(route="觉醒流", why="1-3 阶变 5 阶觉醒、保留等级 —— 强度跳阶，不加级"),
    # ================= 天赋牌（升级弹出，同档三选一）=================
    "加速成长": dict(route="单点加级", once=(10, 10), why="2 张 +5"),
    "超速成长": dict(route="单点加级", once=(15, 15)),
    "延迟满足": dict(route="单点加级", once=(35, 35), why="+10，3 回合后再 +25"),
    "碰碰运气": dict(route="单点加级", once=(12.5, 12.5), energy=-3),
    "祝你好运": dict(route="单点加级", once=(12.5 * 3, 12.5 * 3), energy=-9, why="之后每次升级一张碰碰运气（按 3 次）"),
    "消费返利": dict(route="单点加级", per=(5 * 12 / 15, 5 * 12 / 15), why="每花 15 能量一张 +5（按每回合花 12）"),
    "随机成长": dict(route="开团永久", per=(4, 4), why="品阶最高的 1 人每场 +4 永久"),
    "随机进化": dict(route="开团永久", per=(6, 6)),
    "随机强化": dict(route="开团永久", per=(9, 9)),
    "临时增幅": dict(route="临时等级", temp=(30 / POP, 30), why="20 临时等级随机分，每回合 +5（按平均 30）"),
    "兵行诡招": dict(route="临时等级", temp=(2.5, 2.5 * POP), why="全员 +2 临时（胜 +1），代价 15 血"),
    "胜之不武": dict(route="临时等级", temp=(20, 20 * POP), turns=3, why="3 张：下场全员 +20 临时，伤害 -10"),
    "四重奏": dict(route="临时等级", temp=(5 * LORD, 5 * LORD * POP), need="3 阵营"),
    "三三得三": dict(route="临时等级", temp=(33, 99), need="三分"),
    "中坚力量": dict(route="低阶流", per=(0, 4), why="打出 3 阶英雄牌 +4"),
    "中流砥柱": dict(route="低阶流", per=(0, 6)),
    "战争头牌": dict(route="低阶流", per=(0, 4)),
    "群英荟萃": dict(route="全体加级", per=(1, 3 * POP / 3), why="每打出 3 张不同品阶英雄 全体 +3"),
    "千里相聚": dict(route="无阵营", per=(1.3, 4 * POP / 3), need="无阵营"),
    "协同一致": dict(route="无阵营", per=(LORD, LORD * 2), need="无阵营"),
    "银算盘": dict(route="卖人", per=(0, 2)), "金算盘": dict(route="卖人", per=(0, 3)),
    "超级蓄能": dict(route="整备流", per=(0, PREP * 2 * PREP)),
    "一起发财": dict(route="河洛古币", per=(0, 3 * (N_FAC - 1)), need="河洛"),
    "升值": dict(route="河洛古币", once=(4, 4), per=(1, 1), need="河洛"),
    "古币翻新": dict(route="河洛古币", once=(4, 4), need="河洛"),
    "传家宝": dict(route="河洛古币", per=(1, 2), need="河洛"),
    "钱生战力": dict(route="河洛古币", temp=(6, 24), need="河洛", why="随机 4 名河洛 + 累计用过的古币数（按 6）"),
    "自我奉献": dict(route="河洛牺牲", temp=(2 * HALF * 3, 2 * HALF * 3 * POP), need="河洛", why="每次牺牲全员 +1 临时且递增（按一场 3 次）"),
    "因缘际会": dict(route="大河图腾", per=(AUTOS / 25, AUTOS / 25 * POP), why="因缘图腾：每 25 次普攻全体永久 +1"),
    "场外供能": dict(route="日落海核心", core=LORD * 3, why="回合开始核心 + 棋手等级×3"),
    "动能灌注": dict(route="日落海核心", core_once=10, core=AUTOS / 10, why="核心 +10；全队每攻击 10 次核心 +1"),
    "积少成多": dict(route="日落海核心", core=4, need="日落海", why="日落海英雄每升 10 级核心 +4"),
    "强力小弟": dict(route="召唤", per=(0, 5)),
    "人才济济": dict(route="棋手经验", xp=2 * 3, why="每级少 2 经验（按剩 3 次升级）"),
    "保持进步": dict(route="棋手经验", xp=1 * 8, why="每回合 +1 经验（按 8 回合）"),
    "无限进步": dict(route="棋手经验", xp=2 * 8),
    "泉水": dict(route="棋手经验", xp=4 * 3),
    "三十而立": dict(route="棋手经验", xp=6), "三十而达": dict(route="棋手经验", xp=9),
    "先苦后甜": dict(route="棋手经验", xp=6), "卧薪尝胆": dict(route="棋手经验", xp=9),
    "学霸": dict(route="经济", energy=5),
    "里程碑": dict(route="经济", energy=6), "重大里程碑": dict(route="经济", energy=8),
    "大明星": dict(route="觉醒流", once=(0, 7 * LORD), why="上阵一个等级=棋手等级×7 的 5 阶觉醒英雄"),
    "限制性进化": dict(route="低阶流", once=(0, 20)),
    "镜像迷离": dict(route="临时等级"),
    "混沌裂隙": dict(route="搬等级"), "混沌次元": dict(route="搬等级"),
    # ================= 英雄牌自带的引擎（在场时持续产出；靠商店买到）=================
    "孙膑": dict(route="施法流", hero=True, per=(CASTS / 3, CASTS / 3 * 3), why="每放 3 次技能永久 +1（觉醒 +2）"),
    "弈星": dict(route="棋手经验", hero=True, xp=2 * 2, why="升级时经验 +2"),
    "曜": dict(route="合成流", hero=True, per=(1, 2), why="合成额外 +1"),
    "苏烈": dict(route="河洛牺牲", hero=True, per=(0, 2)),
    "赵云": dict(route="登场流", hero=True, per=(0, 3 * PLAYS / 2)),
    "马可波罗": dict(route="日落海核心", hero=True, once=(0, 2), core_once=2, why="登场：自身和核心 +2"),
    "芈月": dict(route="逐鹿效果牌", hero=True, per=(1, 1), need="逐鹿"),
    "蔡文姬": dict(route="三分登场", hero=True, per=(0, 3), need="三分"),
    "云中君": dict(route="装备流", hero=True, per=(EQ_C, EQ_ALL), why="每件装备每场永久 +1"),
    "安琪拉": dict(route="日落海核心", hero=True, per=(0, 2), core=2, why="整备：自身和核心 +2（自身若是主C 由模型另算）"),
    "张良": dict(route="开团流", hero=True, per=(OPEN * 3 / POP, OPEN * 3)),
    "明世隐": dict(route="临时等级", hero=True, per=(2, 2), why="主C 临时等级 2 级转永久（登场再 +2）"),
    "猪八戒": dict(route="合成流", hero=True, per=(0, 2)),
    "镜": dict(route="逐鹿效果牌", hero=True, per=(1, 1)),
    "阿轲": dict(route="登场流", hero=True, once=(0, 2 * (60 - HP) // 10)),
    "孙悟空": dict(route="闪现", hero=True, per=(2, 2 * 3), why="整备：闪现英雄 +2（按 3 个闪现）"),
    "小乔": dict(route="登场流", hero=True, per=(PLAYS * 3 / POP, PLAYS * 3)),
    "庄周": dict(route="登场流", hero=True, once=(0, 2)),
    "韩信": dict(route="整备流", hero=True, per=(0, PREP + OPEN)),
    "刘禅": dict(route="施法流", hero=True, temp=(2 * CASTS * HALF, 2 * CASTS * HALF * POP), why="英雄放技能时该英雄 +2 临时"),
    "诸葛亮": dict(route="施法流", hero=True, temp=(0, 4 * CASTS * (POP - 1) * HALF), why="其他英雄每放技能 诸葛亮 +4 临时"),
    "周瑜": dict(route="登场流", hero=True, once=(1, POP)),
    "姜子牙": dict(route="无阵营", hero=True, per=(0, 4)),
    "武则天": dict(route="河洛古币", hero=True, once=(5, 5), per=(1, 2), need="河洛"),
    "海诺": dict(route="日落海核心", hero=True, core_mult=1.5, why="核心每次升级额外 +1（约 ×1.5）"),
    "刘邦": dict(route="大河图腾", hero=True, per=(5, 5), need="大河流域"),
    "狂铁": dict(route="日落海核心", hero=True, core=2, why="整备：核心 +2"),
    "亚连": dict(route="日落海核心", hero=True, per=(0, 2), core=1, why="整备：装备最多的日落海 +2、核心 +1"),
    "米莱狄": dict(route="日落海核心", hero=True, core=3, why="整备：核心 +3（召唤物临时等级另算）"),
    "亚瑟": dict(route="日落海核心", hero=True, core=3 * 1, why="每打出日落海英雄牌核心 +3（按每回合 1 张）"),
    "朵莉亚": dict(route="整备流", hero=True, per=(PREP * 3 / POP, PREP * 3), core=PREP, why="每次整备：随机 3 人和核心 +1"),
    "莱西奥": dict(route="日落海核心", hero=True, core=.5, why="每卖 2 个场上英雄核心 +1"),
    "露娜": dict(route="日落海核心", hero=True, core_once=2, why="登场：核心 +2，并触发 2 个整备"),
    "定向传输": dict(route="日落海核心", core_focus=True, why="核心优先给装备最多的日落海英雄，至多 200"),
    "能量汲取": dict(route="日落海核心", core=6, why="卖场上英雄时核心获得其等级（按每回合 6）"),
    "核心裂痕": dict(route="日落海核心", per=(20, 20), why="每回合一张盗火：从核心偷至多 20 级（永久）"),
    "盗火": dict(route="日落海核心", once=(20, 20)),
    "双重分配": dict(route="日落海核心", why="核心分配时附带属性（不加级）"),
    "李信": dict(route="河洛牺牲", hero=True, temp=(0, 7 * 3 * HALF), why="牺牲英雄阵亡时 李信 +7 临时（按一场 3 次）"),
    "铠": dict(route="河洛牺牲", hero=True, temp=(5 * HALF, 5 * HALF * 3), need="河洛", why="牺牲：河洛英雄 +5 临时"),
    "裴擒虎": dict(route="闪现", hero=True, temp=(0, 15), why="整备：拿随机一人一半等级的临时等级"),
    "司空震": dict(route="临时等级", hero=True, temp=(0, 10), why="其他英雄拿临时等级时分一半"),
    "敖隐": dict(route="大河图腾", hero=True, temp=(0, 4 * 5 * 1 * HALF), need="大河流域", why="每个图腾每 4 秒 +4 临时"),
    "东皇太一": dict(route="大河图腾", hero=True, temp=(0, 10), need="大河流域"),
    "李白": dict(route="河洛古币", hero=True, temp=(4 * 3, 7 * 3), need="河洛", why="古币额外 4 临时（给李白 +7）"),
    "花木兰": dict(route="河洛牺牲", hero=True, temp=(0, 10 * HALF)),
}


def value(e: dict, rounds_left: int) -> dict:
    """剩余 rounds_left 回合里，这张卡一共给主C / 全场多少级（持续产出 × 剩余回合，封顶 turns）。"""
    turns = min(rounds_left, e.get("turns", rounds_left))
    oc, ob = e.get("once", (0, 0))
    pc, pb = e.get("per", (0, 0))
    tc, tb = e.get("temp", (0, 0))
    return {"carry": oc + pc * turns, "board": ob + pb * turns,
            "temp_carry": tc, "temp_board": tb, "temp_turns": turns,
            "core": e.get("core_once", 0) + e.get("core", 0) * turns,
            "xp": e.get("xp", 0), "energy": e.get("energy", 0) + e.get("per_energy", 0) * turns}


def main() -> int:
    cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    miss = [n for n in R if n not in cards]
    if miss:
        print("卡牌库里没有：", miss)
        return 1
    out = {n: dict(e, category=cards[n]["category"], quality=cards[n].get("quality")) for n, e in R.items()}
    (ROOT / "data" / "strategy").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "strategy" / "growth_routes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                     encoding="utf-8")
    by = {}
    for n, e in out.items():
        v = value(e, 6)
        by.setdefault(e["route"], []).append((v["carry"], v["board"], n, e["category"], e["quality"]))
    print(f"{len(out)} 张 -> data/strategy/growth_routes.json；按套路（剩 6 回合时的总产出：主C / 全场）：")
    for rt, xs in sorted(by.items(), key=lambda kv: -max(x[1] for x in kv[1])):
        xs.sort(reverse=True, key=lambda x: x[1])
        print(f"  {rt:8s}: " + "、".join(f"{n}({c}{q}阶 {a:.0f}/{b:.0f})" for a, b, n, c, q in xs[:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
