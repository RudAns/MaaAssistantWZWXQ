"""**临时规则登记表**（2026-10-13 建立：价值判断不写死在规则里；写死的冲突规则容易在训练之后忘记撤掉，统一在这里登记）。

凡是规则层替模型做的**价值判断**（不是合法性 / 感知 / 执行），一律在这里登记一个开关：
  - where：代码位置（grep 关键字）
  - why：起因
  - teach：要教给哪个模型、写进哪里（题面 / 规范 / 模拟器）
  - remove_when：什么时候撤（通常 = 学会它的模型上线后，实战验证不退步）

训练 / 部署脚本结束时会打印 `active()`；模型学会后把对应开关改成 False，跑一局确认，再删代码。
新加价值类规则必须在这里登记，否则复盘时会被当成"规则挡住模型"。
"""
from __future__ import annotations

RULES: dict[str, dict] = {
    "talent_sunset_core_drain": {
        "on": True,    # 2026-10-17 开
        "where": "laya_policy._talent_decide：主C / 路线主C 是日落海时，去掉'阿科米亚核心失去 N 级'的天赋（能源置换）",
        "why": "第 682 局伞亚连选了能源置换（每回合核心 −20 级换 10 能量）；日落海路线核心被扣级后无法成型，这个天赋不可选",
        "teach": "f9lz 题面写 AVOID on a 日落海 route: drains 阿科米亚核心 levels（R37 训练）",
        "remove_when": "f9lz 学生上线、天赋定向题学会避开",
    },
    "dead_equip_route": {
        "on": False,    # 2026-10-16 晚开   # 10-20 关（review-interim-conflicts：已无代码调用，装备不进卖手牌）
        "where": "rules._dead_hand_card / _equip_nobody：场上和锁定路线最终阵容里都没人能穿的装备，放满一回合后卖",
        "why": "第 676 局：阿科米亚之子（只能非日落海英雄穿）、月影之力（只有露娜生效）压在日落海马可波罗路线的手里；两件装备都没人能穿",
        "teach": "Laya sell_hand 题扩到手里的装备牌（f9lz：写'场上 / 路线阵容里谁能穿'）",
        "remove_when": "sell_hand 覆盖装备牌的学生上线后",
    },
    "dbdj_core_use": {
        "on": False,   # 2026-10-16 晚开；同晚 R36 上线后关：点兵点将定向测试 tmass +0.232 [+0.171,+0.297]（Y_dbdj，n=102）—— 下一局验证
        "where": "laya_policy._shadow_timing：点兵点将 use_timing 题判'留着'时，题面写 CORE ≥5 或 awakens NOW 含现任主C → 现在用",
        "why": "第 670 局 R9~R11 核心 4~5/6、点兵点将能让守约当场觉醒，R35 四次都 0.55~0.59 判留着（use_timing 训练题只有十几道）",
        "teach": "Laya use_timing：离线多出点兵点将时机题（核心多 / 主C 当场觉醒）+ g9zj 规范 + R36 过采样",
        "remove_when": "R36（或之后）在点兵点将定向测试上学会'核心 ≥4/6 或主C 当场觉醒就用'，实战一局不再留着",
    },
    "haiyue_plan": {
        "on": True,
        "where": "rules.haiyue_phase / decide_haiyue / haiyue_hold_level / haiyue_save_equip（只在伞海月路线 user_haiyue_umbrella 上生效）",
        "why": "伞海月的设计节奏：1~2 级养干将、觉醒后同名先捏着；3 级停下集中合成刷 3 阶锦囊开遗志；拿到遗志冲 5 级刷海月；海月到手清场后对干将打遗志（或红叶染锋卖干将换绑）；海月到手前装备存着",
        "teach": "推演器（锦囊 / 遗志 / 换主C 代价已建模）+ 局面网络 eco 题 + Laya 题面写出伞海月阶段",
        "remove_when": "推演器 / 模型能自己算出这套节奏后撤",
    },
    "equip_hold_for_real_carry": {
        "on": True,    # 2026-10-19 开（临时主C 先不穿装备，否则真主C 到了可能缺装备）
        "where": "rules._equip_target：现任主C 是临时主C（1 阶、非路线主C）时装备先留在手里，手牌 ≥8 张时照常穿",
        "why": "临时主C 很快卖掉（换绑），装备回手但成品 / 四选一已经按临时主C 选了（第 751 局虚无法杖）；真主C 来了缺装备",
        "teach": "use_card 题面写 TEMPORARY carry: equip goes back to hand when sold → 规范 + 重标",
        "remove_when": "Laya 用牌题在临时主C 期间自己选 hold",
    },
    "eco_behind_schedule": {
        "on": True,    # 2026-10-19 开（起因：第 755 局升级落后）
        "where": "laya_policy._shadow_eco：等级低于 meta.level_target（第 7 回合 4 级 / 第 10 回合 5 级）时，升得起 → levelup，升不起但攒一回合够 → save；10-19 加的一回合预判当天撤（分析结论：R4~R8 看店太少才是核心件慢的主因，预判让攒钱更多）",
        "why": "第 755 局 3 级卡到第 11 回合：R5~R8 升不起却每回合 roll 刷光钱，R9/R10 升得起仍 roll（0.51 / 0.48），6 人口打 7~8 人口，血 57→11",
        "teach": "规范 54.22（落后进度时停级多刷不适用）+ 重标 r9zx_eco 落后进度题 → 重训经济网络",
        "remove_when": "经济网络在落后进度时自己选 levelup / save",
    },
    "route_carry_takes_over": {
        "on": True,    # 2026-10-19 开（最终目标是凑齐阵容推荐里的最终阵容；路线定好后不换）
        "where": "laya_policy 换主C 题之前：路线已锁、路线主C 在场且合格、现任主C 不是他 → 路线主C 接任（原来只对 1 阶占位主C 生效）",
        "why": "第 752 局 R6 主C 空缺那一刻只有虞姬合格 → 虞姬接任换绑枫叶舞；R7 锁守约路线后局面网络一直 keep 虞姬，路线和主C 对不上到结束（第 5 名）",
        "teach": "cgnet carry 标签：路线已锁、路线主C 在场时换给他 ≥0.9 → 重训",
        "remove_when": "cgnet carry 在'路线主C 在场、现任不是他'时自己换",
    },
    "forge_avoid_low_use": {
        "on": True,    # 2026-10-18 开（合成装备时不选可用性太低的成品，如虚无法杖）
        "where": "tools/live_match_agent 装备四选一：FORGE_AVOID 里的成品不跟 👍、选中了也换同基础装备的另一件",
        "why": "第 751 局 R1 跟游戏 👍 给临时主C 干将选了虚无法杖（只有 15 法穿），后来路线主C 守约物理 → R10 / R13 卖掉",
        "teach": "执行层四选一目前不走模型；以后接 Laya 题时把可用性写进题面",
        "remove_when": "四选一交给模型、且模型不再选低可用性成品",
    },
    "talent_unplayable": {
        "on": True,    # 2026-10-18 开（战争头牌：执行不了的天赋）
        "where": "laya_policy._talent_decide：卡面要按'每回合品阶数字'出牌的天赋（战争头牌）去掉——感知层读不到这个数字",
        "why": "第 751 局 R2 Laya 0.51 选了战争头牌；talent_techniques.json 标 handled=no、感知读不到",
        "teach": "题面 f9t 写 AVOID（同混沌裂隙）→ R44；若以后感知能读到品阶数字再接执行",
        "remove_when": "R44 对战争头牌 ≤0.05，或执行层接上品阶数字",
    },
    "duo_needs_carry": {
        "on": True,    # 2026-10-18 开（兄妹只有其中一人是主C 时才有意义）
        "where": "laya_policy._talent_decide：共战 / 组合天赋（卡面写共战，或点名 ≥2 名英雄：崛起 兄妹 兽觉 巨像 舞力全开 起舞 箭雨 双生突袭 毒奶 逐星 宝盒 旋刃）里没有主C / 路线主C → 去掉（双人天赋里没有主C 时收益很低）",
        # 2026-10-19 晚收紧（只有两个人都有用时才选）：路线已锁时两人都要在最终阵容 / 核心引擎 / 主C 里；送的英雄不单独算
        "why": "第 748 局 R9 主C 守约、兄妹给铠 + 露娜，Laya 0.52 选了；第 717 局同类",
        "teach": "规范 54.20-3 + 题面 f9t（DUO 写 no value unless carry）→ R44",
        "remove_when": "R44 在主C 不是共战之一时给共战天赋 ≤0.05",
    },
    "route_avoid_weak": {
        "on": True,    # 2026-10-19 开（段位卡在星耀 2）
        "where": "laya_policy._route：路线题问 Laya 前去掉阿离日落海安琪拉 / 伞亚连 / 双剑出鞘伞亚连 / 双刃亚连的选项；rules 里'主C 定了路线跟着主C 锁'也跳过这几条（第 800 局安琪拉漏网）；第 804 局起主C 是安琪拉的路线不论名字一律屏蔽（'日落海整备安琪拉（阿离版）'漏网）；第 809 局后：路线没锁时换主C 题也去掉安琪拉（806 / 809 主C 成了安琪拉、路线永远锁不上、两局第 6）；第 823 局后：现任主C（≥2 阶）就是屏蔽路线的主C 时，路线题不去掉他的路线、规则跟随锁定也放行（54.28-12）；安琪拉：换主C 题有别的人选时去掉；只剩她时照常当主C 并锁她的路线（第 827 局全局挡导致主C 空着，10-20 改）；马可波罗第 827 局后撤回（屏蔽后没主C 可选）（还剩别的路线或 flex 时）；伞亚连 10-19 起：前期杀不了人、无法升级 → 先屏蔽",
        # 10-22 补：开局首选组（没禁河洛 = 河洛组）还能走时，第 6 回合前选路线题只留首选组路线（906 / 915 局禁逐鹿却锁了伞亚连）
        "why": "740~776 局：伞亚连 5 局均名次 5.4、前三 0/5；日落海安琪拉 2 局均 5.5、0/2；守约系 9 局均 2.1、前三 8/9（有选择偏差：主C 早到才走）",
        "teach": "f9v 路线题写本系统战绩（ours avg#X nN，data/laya/route_stats_f9v.json）+ 规范 54.24-5 → R46",
        "remove_when": "R46 在路线题里给这两条 ≤0.2（主C 未到手时），或这两条在新数据里战绩回到均名次 ≤3.5",
    },
    "engine_tier_eco": {
        "on": True,    # 2026-10-19 开（增长引擎刷出太慢，伞亚连的雅典娜来得太晚；10-18：路线定好后不换，要在对应棋手等级多刷几次）
        "where": "laya_policy._shadow_eco：路线已锁、核心引擎（min_engine early_priority）缺 t 阶件（10-20 晚加过'主C 没觉醒也算缺'，10-21 撤回：停级刷主C 到觉醒的打法取消） → 棋手 < t 级：升得起升、升不起攒；≥ t 级：save 改 roll",
        "why": "第 779 局雅典娜（4 阶）R6~R8 停 3 级刷不到、R4~R8 只刷 3 次，R11 才出现；759~777 引擎启动中位 R10",
        "teach": "经济题面 / 局面网络特征写缺的引擎件品阶与各级命中（eco_core_feat 已有，伞亚连等 5 条路线 10-19 才补引擎表）+ 规范 54.24-6/9 → 重训 eco",
        "remove_when": "重训的经济网络在'引擎缺 t 阶、棋手 < t 级'时自己选 levelup / save，≥ t 级时选 roll",
    },
    "lock_for_carry_copy": {
        "on": True,    # 2026-10-19 开（没钱时商店里有狄仁杰却没锁店）
        "where": "laya_policy._shop_action：Laya 选 stop、钱 <3、没有免费刷新、店里有主C 同名或缺的核心引擎件（min_engine early_priority 未拥有）→ 锁店",
        "why": "第 781 局 R9：剩 2e，店里第二张狄仁杰（主C 合成），题面 lock 写 keeps 狄仁杰 (merge)，Laya stop 0.52",
        "teach": "规范 54.25：没钱时店里有主C 同名 / 缺的引擎件 → lock ≥0.7 → 重标 R47",
        "remove_when": "新学生在这类题 lock ≥0.6",
    },
    "taiyi_shield_early": {
        "on": True,   # 2026-10-21 重新打开（太乙前期先贴盾山、后期贴主C 的做法保留）
           # 2026-10-20 晚开（前期拿到太乙可以先挂到盾山旁边，让盾山多复活一次、多触发一次牺牲拿古币）
           # 2026-10-21 晚改（第 881 局：庄周没有作用，太乙挂到铠或盾山旁边，后期再挂到主C 旁边；太乙可以多次触发牺牲英雄的牺牲效果）
        "where": "rules.buddy_anchor：路线锁了、牺牲线（路线主C 卡面吃「牺牲效果的英雄阵亡」，或河洛）、第 12 回合前且明世隐已觉醒（转化够大，10-21 第 900 局）→ 太乙贴铠（优先，复生再死再给临时等级）/ 盾山；否则 / 第 12 回合起贴主C",
        "why": "盾山「牺牲：获得 1 张古币」，太乙让他复活一次 = 多一次牺牲多一张古币（河洛古币 / 牺牲线前期等级）",
        "teach": "规范 54.30 + 题面写太乙早期贴盾山",
        "remove_when": "站位交给模型（目前站位不是模型题）——长期保留，登记在案",
    },
    "maple_temp_host": {
        "on": False,   # 2026-10-21 关（恢复原来的节奏，只保留阿离的主C 配置与屏蔽）；代码留着，要用再开
           # 2026-10-20 晚开（枫叶舞挂在不是该主C 核心配件的英雄身上；4 阶主C 用干将 / 孙膑当临时主C）
        "where": "rules._skill_target：路线已锁、主C 没到时，枫叶舞绑临时宿主：4 阶以上路线优先干将莫邪 / 孙膑（lord_config.temp_carry），否则场上能打的、非路线核心件里估值最高的；主C 到了按老规矩卖宿主换绑",
        "why": "第 849 局主C 空缺、枫叶舞整局没绑（10-15 关掉 maple_fallback 后没主C 就不绑）",
        "teach": "规范 54.30 + 题面写'no carry yet: bind maple to a non-core temp host'",
        "remove_when": "模型在没主C 时自己出绑临时宿主的动作",
    },
    "core_first_no_copies": {
        "on": True,    # 2026-10-21 开（每个核心阵容英雄先只买一张，先凑完整核心阵容，再补觉醒）
        "where": "laya_policy._shop_decide / _card_pick_answer：路线已锁、核心还缺人（rules.route_core_missing）→ 已有的核心件不买 / 不拿第二张；**主C 本人除外**（同名照买）、明世隐除外（stack_copies_buy）",
        # 10-21：超核心件没觉醒前豁免；第 12 回合起不生效（后期认命）
        # 10-21 第 879 局：引擎核心件（min_engine core_set，如雅典娜）同名不受限（雅典娜是绝对核心）
        # 10-21 审查：缺件判定改用 route_core_missing_now（第 12 回合前不算太乙、当前等级刷不出的不算），原来太乙不急买导致铠 / 盾山同名整段不买
        "why": "先把核心阵容每人凑 1 张，再补同名觉醒",
        "teach": "规范 54.30-15 + 题面写'core incomplete: missing X; Y already owned'",
        "remove_when": "新学生在核心缺人时对已有核心件的同名 ≤0.1",
    },
    "no_core_for_offroute": {
        "on": True,    # 2026-10-21 开（第 851 局 R5：Laya 换人题卖亚连线核心件米莱狄给马可波罗（关掉的主C）腾位）
        "where": "laya_policy._swap：路线已锁、等位的人不在路线里 → 换人题去掉'卖路线成员 / 主C'的选项（还剩别的时）",
        "why": "路线定了不换（设计约定），为路线外的人卖核心件等于拆自己的阵容",
        "teach": "规范 54.30-14 + 换人题题面写'incoming X not in ROUTE'",
        "remove_when": "新学生在这类题上卖路线成员 ≤0.05",
    },
    "stack_copies_buy": {
        "on": True,    # 2026-10-21 开（明世隐在核心阵容里时有多少买多少）
        "where": "laya_policy._shop_decide / _card_pick_answer：路线已锁、明世隐（STACK_COPIES）是路线成员 → 店里有且买得起就只留买他；N 选 1 有他就拿",
        # 10-21：只凑到觉醒（明世隐觉醒后不再只留买他）
        # 10-21 审查：成员只认核心阵容（route_core）；主C 同名 / 现在缺的核心件（route_core_missing_now）也留着，不再"只留明世隐"
        "why": "明世隐登场「可转化的等级 +」随同名叠加，每张都涨主C 永久等级（卡面答复 C23）",
        "teach": "规范 54.30-10 + 题面写'明世隐 in core: every copy adds conversion'",
        "remove_when": "新学生在路线有明世隐时买他 ≥0.9",
    },
    "core_roll_first": {
        "on": True,    # 2026-10-22 开（根因：核心阵容没有在合适的棋手等级刷出；先上临时规则）
        "where": "rules.eco_says：路线已锁、第 4~11 回合、棋手等级已到本回合进度线（meta.level_target，948 局 3 级停级教训）、路线核心件还缺且当前棋手等级刷得出（route_core_missing_now）→ 经济决定改成 roll（钱先刷核心件）；"
                 "钱够升级 + 8 时仍照经济网络升级；同条件下 laya_policy 商店题去掉路线外的英雄（947 局 R11 东皇太一买了又不上场）；10-24 起路线没锁、河洛没禁、开局组一个没到时第 3~8 回合也刷组员（group_wait_missing，1002~1030 局等不到组员的 9 局均 5.4）",
        "why": "945 局铠（3 阶）R4~R8 一直缺，棋手 3~4 级刷得出，但每回合只刷 0~2 次，钱花在升级和路线外的人上；942 局铠 / 明世隐同理",
        "teach": "eco 网络标签：路线核心件缺且刷得出时 roll ≥0.8（除非钱很多）→ 重训 eco",
        "remove_when": "新 eco 网络在这类局面自己选刷新",
    },
    "engine_tier_rush": {
        "on": True,    # 2026-10-22 开（亚连路线可以走，但要先把亚连养起来、早点拿到雅典娜，按棋手等级推算怎么拿）
        "where": "rules 商店开着的经济路径：第 4~10 回合（10-22 第 951 局：R4 还 2 级、铠刷不出，从 5 提前到 4，和进度线 R4=3 级对齐）、缺的最低配套引擎件当前棋手等级刷不出（engine_missing_unreachable）、钱够升级、本回合还没升 → 先升级",
        "why": "雅典娜 4 阶（18 个 4 阶英雄）：3 级刷不出、4 级每次刷新约 4.4%、5 级约 6.5%；亚连局 906~924 实测第 8~9 回合才到 4 级（进度第 7 回合）",
        "teach": "eco 网络标签：缺的引擎件刷不出时 levelup ≥0.8 → 重训 eco",
        "remove_when": "新 eco 网络在这类局面自己选升级",
    },
    "no_offlist_carry_buy": {
        "on": True,    # 2026-10-22 开（第 902 局 R3 买了马可波罗：他既不是主C 也不是任何阵容的配件，没必要优先买）
        "where": "laya_policy._shop_decide：名单外（carry_pool 没开）的 2 阶以上能当主C 的英雄、不在锁定路线阵容里（没锁 = 都不在）、不是现任主C → 商店题去掉买他",
        "why": "f9za 题面候选路线清单没按主C 名单过滤，把马可波罗写成 CORE of route，Laya 0.46 买了",
        "teach": "题面 f9zb flex_route_keys 按名单过滤（已改）→ R52 / R53",
        "remove_when": "R52（f9zb）上线后名单外主C 不再被写成 CORE、商店题不再选买",
    },
    "carry_late_defense": {
        "on": True,    # 2026-10-21 开（第 891 局：主C 前两件都是输出时，后期缺的最后一件可以选防御装备）
        "where": "rules._equip_target：近战主C 任何回合（已有 ≥1 件输出、还没防御装）/ 远程主C 第 12 回合起身上正好 2 件输出装、第三格空 → 护命灵珠 / 翼苍穹盾 / 苍穹王权 / 霸者重铠（力量腰带铸霸者重铠）可以给主C；live_match_agent 四选一力量腰带给主C 只铸霸者重铠",
        "why": "只限不降攻速的特殊防御装和霸者重铠（生命 >70% 减伤 40%）；其余时候第三格仍等输出 / 免控装",
        "teach": "规范 54.36-9 + 题面 f9zb 装备适配写 'late 3rd slot: OK (non-slowing defense)' → R52",
        "remove_when": "R52 装备题学会这条，且用牌 / 穿装交给模型后",
    },
    "carry_copy_buy": {
        "on": True,    # 2026-10-21 开（第 887 局 R9：刷新出了守约却没买，之后一直刷新、目标不明）
        "where": "laya_policy._shop_decide：路线已锁、路线主C 在场、店里有他的同名且买得起、手牌没满 → 商店题只留买他（缺的核心件也留着）；在 eco_level_budget 之后执行，主C 同名本来就在额度豁免里",
        "why": "主C 同名合成 = 主C 涨等级（觉醒后也照涨）；887 局 R9 Laya 0.34 买 vs 0.35 刷新，刷的是苏烈 / 盾山 / 铠这些已有件的同名",
        "teach": "题面 f9zb 写'merge into carry: +N lv'、刷新行分开缺件 / 同名 + 规范（主C 同名照买）",
        "remove_when": "R52 主C 同名在店时选买 ≥0.9",
    },
    "final_team_first_liyuanfang_note": {
        "on": True,    # 说明条目（代码挂在 final_team_first 里，关 final_team_first 一并关）：2026-10-21 第 886 局
        "where": "laya_policy._swap final_team_first 后期件分支：第 12 回合起太乙等位、明世隐已觉醒、李元芳在场没觉醒 → 李元芳算可卖（有可卖的人时'不换'去掉 = 直接换）",
        "why": "后期李元芳还没觉醒、明世隐已能提供很高等级时，用太乙换李元芳更划算：少一回合两级的差距不大，太乙复生的性价比更高",
        "teach": "规范 54.36-7 + 题面 f9zb 写李元芳 +2 lv/round vs 太乙复活",
        "remove_when": "R52 后期这类换人题选卖李元芳 ≥0.8",
    },
    "roll_engine_unreachable": {
        "on": True,    # 2026-10-21 开（第 883 局 R5~R7 棋手 3 级连刷：凑核心阵容要结合缺的英雄的品阶，缺的都是 4 阶英雄）
        "where": "laya_policy._shop_decide：第 12 回合前、没有免费刷新、缺的最低配套引擎件（min_engine early_priority，替代件算有，后期件不算）当前棋手等级全都刷不出 → 商店题去掉 refresh；eco_roll_follow 同条件不加码刷",
        "why": "883 局 3 级缺雅典娜（t4），题面'28% 刷到想要的牌'全是已有件同名，Laya 照刷、落后进度；钱应该留着升 4 级开 t4",
        "teach": "题面 f9zb：刷新行把'缺的核心件'和'已有件同名'分开写，缺件刷不出时写'missing core unreachable at lv N: level up first' + 规范 54.36-4；经济网络同类题重标",
        "remove_when": "R52 / 新 eco 网络上线后，引擎件刷不出时商店题不选花钱刷新 ≥0.8",
    },
    "msy_taiyi_no_swap": {
        "on": True,    # 2026-10-21 开（第 882 局 R13 卖太乙给新到的明世隐：非常后期没必要用明世隐换太乙；明世隐叠到觉醒后也不用太乙换明世隐）
        "where": "laya_policy._swap：第 12 回合起明世隐等位 → 不卖后期件（太乙）给他、也不触发 late_fill 的'后期件让位'；后期件（太乙）等位而场上明世隐已觉醒 → 不卖明世隐（任何回合）",
        "why": "刚上场的明世隐只转 +2，后期剩 2~3 回合不值一个太乙；觉醒的明世隐叠了登场层，卖掉清零（mingshiyin-entry-stacks）",
        "teach": "题面 f9zb 换人题写'新明世隐 ≈ +2 × 剩余回合 vs 太乙'、'明世隐登场已叠 N 层，卖掉清零' + 规范 54.36-3 → R52",
        "remove_when": "R52（f9zb）上线、后期太乙 / 明世隐互换题不再选卖",
    },
    "entry_trigger_msy": {
        "on": True,    # 2026-10-21 开（第 881 局仍不买唤醒；第 879 局出现唤醒没买；第 853 局：有明世隐时唤醒收益更高）
        "where": "laya_policy._shop_decide / _card_pick_answer：场上有明世隐、手里没有没用的触发登场牌 → 店里 / N 选 1 的唤醒 / 集体唤醒只留拿它（主C 同名 / 缺的核心件也留着）",
        "why": "明世隐觉醒后每次登场转化 +4 临时等级 → 主C 永久等级；881 局 R12 题面收益行（守约 ≈ +16 lv）超出 44 token 预算被 pack 丢掉，Laya 0.33 vs 刷新 0.35",
        "teach": "题面 f9zb：明世隐收益行写短（放得进选项预算）+ 规范 54.31-1 → R52 重标训练",
        "remove_when": "R52（f9zb）上线、场上有明世隐时商店题选买唤醒 ≥0.8",
    },
    "core_before_level_cards": {
        "on": True,    # 2026-10-21 开（凑到至少 1 个核心英雄，优先级高于升 2 级的效果牌）
        "where": "laya_policy._shop_decide / _card_pick_answer：路线已锁、选项里有一张都还没有的路线核心件（rules.route_core_missing）→ 这道题去掉分摊升级牌（rules.SPREAD_LEVEL_CARDS：简易协作 / 精密协作 / 有福同享 / 阵营祝福 / 演练战术…；古币等集中给 1 人的不算，10-21 更正），下一题照常",
        "why": "核心件只靠商店整套要 45~50 次刷新（按刷新概率推算），加等级牌随时有",
        "teach": "规范 54.30-9 + 题面写'missing core X in shop: take it before level cards'",
        "remove_when": "新学生在有缺的核心件时对加等级牌 ≤0.1",
    },
    "no_red_leaf": {
        "on": True,    # 2026-10-21 开（红叶染锋不选）
        "where": "laya_policy.pick_lord_boost：枫叶舞强化三选一去掉红叶染锋（还剩别的选项时）",
        "why": "换绑只带走 50% 等级，实测临时宿主第 6~8 回合才 10~15 级（带走 5~8 级），还占掉枫影无拘那一格",
        "teach": "规范 54.30-7 + 重标枫叶舞强化题",
        "remove_when": "新学生对红叶染锋 ≤0.05",
    },
    "preferred_route_per_carry": {
        "on": True,    # 2026-10-23 开
        "where": "laya_policy._route：路线题里同一主C 有多条路线、且首选那条在候选里 → 去掉其他（首选表 data/lords/阿离/lord_rules.json preferred_route_by_carry）",
        "why": "879~997 局：守约 一枪清屏 15 局均 2.3 / 枪有点饿了 16 局 3.9 / 无限狙 6 局 4.7；沈梦溪 伞猫无cd 17 局 3.5 / 无限炸弹猫 5 局 4.4",
        "teach": "路线题标签：同主C 多路线时按实战战绩",
        "remove_when": "试 15 局：锁首选路线的局均名不优于原来同主C 的整体均名就关",
    },
    "no_yalian_unless_heluo_ban": {
        "on": False,   # 2026-10-23 开（试行）；10-25 第 1093 局后关：1024~1093 组外主C 被挡到 R9 才锁路线 8 局均 5.25（其余 62 局 3.95），中途换回组员的 1061 / 1062 也是 6 名 —— 比规则前早锁亚连的 4.3 还差
        "where": "laya_policy._route 路线题 + rules 路线未锁跟主C 锁路线：开局配置是河洛三C 组（河洛没禁）时，不锁亚连 / 狄仁杰（拿了幸运硬币除外）的路线（名单 data/lords/阿离/lord_rules.json no_lock_outside_group；10-23 第 1005 局扩到狄仁杰）",
        "why": "929~989 局没禁河洛时退到亚连 6 局 5/1/5/6/4/5 均 4.3；同期守约 3.3、沈梦溪 3.6",
        "teach": "路线题标签：河洛没禁时亚连路线 ≤0.2",
        "remove_when": "试 10 局对比：不退亚连的局均名不优于 4.3 就关掉（10-24 1021/1023 整局不锁 6/5 名 → 改为只挡前 8 回合）",
    },
    "route_lock_single_carry": {
        "on": True,    # 2026-10-22 开（route_lock_opening 10-21 已关，这是窄版）
        "where": "laya_policy._route：开局配置是单一主C（禁河洛 → 亚连）时，第 4 回合起路线题里有这条路线就只留它（不选 flex）",
        "why": "941 / 947 局禁河洛、亚连没到手，路线题只有伞亚连 / flex，Laya 一直 flex → 整局没路线，core_roll_first 不触发、主C 在名单外的人之间换（947：马可→海诺）；"
               "10-22 复盘结论：先把亚连养起来、早点拿到雅典娜；根因是核心阵容没有在合适的棋手等级刷出",
        "teach": "路线题标签：单一开局主C 的路线在候选里时选它（主C 没到手也锁）",
        "remove_when": "新学生在这类题上选开局路线 ≥90%",
    },
    "route_lock_opening": {
        "on": False,   # 2026-10-21 关（恢复原来的节奏，只保留阿离的主C 配置与屏蔽）；代码留着，要用再开
           # 2026-10-20 晚开（先定要玩的路线，开局锁定的就是最终路线）
        "where": "laya_policy._route：第一个准备阶段按 data/lords/阿离/carry_pool.json 的 opening_order 锁路线（开着、阵营没被禁的第一个组 / 主C）；主C 组谁先在场路线换到他的阵容一次（rules.group_sync）",
        "why": "第 849 局到第 8 回合路线还是 flex，升级 / 刷牌 / 买谁都没有目标",
        "teach": "规范 54.30 + 题面 f9za 写'opening route = final route'；路线题改成只在开局问一次",
        "remove_when": "新学生在开局路线题上和 opening_order 一致 ≥90%",
    },
    "single_copy": {
        "on": True,    # 2026-10-20 晚开（第 846 局：太乙真人只买 1 张，不凑觉醒）
        "where": "laya_policy._shop_decide / _card_pick_answer：rules.SINGLE_COPY（太乙真人）场上或手里已有 → 商店 / N 选 1 去掉同名",
        "why": "太乙只要开团复生给主C，第二张合成觉醒没收益，还占钱和手牌",
        "teach": "规范 54.29-20 + 题面 f9za 给太乙同名写 'only 1 copy needed, do not awaken'",
        "remove_when": "新学生对已有太乙时的同名 ≤0.05",
    },
    "no_lowest_level_cards": {
        "on": True,    # 2026-10-20 晚开（第 844 局：扶倾济弱给最低等级英雄加等级，对养主C 的阵容无效）
        "where": "laya_policy._card_pick_answer / _shop_decide：rules.never_pick = 卡面把等级给场上等级最低的人的牌（扶倾济弱、扫拂、传承）+ NEVER_PICK（星源星徽 对单主C 阵容没有收益；阿科米亚之子 / 驱雷 / 月影之力 别家棋手专属、阿离用不上；阵营祝福 随到主C 概率太低）从 N 选 1 / 商店题 / 拍卖题（10-21 起）去掉（还有别的选项时）",
        "why": "单主C 阵容主C 永远不是等级最低的，这类牌只喂杂牌",
        "teach": "规范 54.29-16 + 题面 f9za 给这类牌写 'goes to the LOWEST-level unit (X lvN), never the carry'；复盘 blunder_check B9 自动发现",
        "remove_when": "新学生在这类选项上 ≤0.1",
    },
    "eco_level_budget": {
        "on": True,    # 2026-10-21 eco876 分析：等级平均 R4 2.3 / R7 3.3 / R10 4.1（进度线 3/4/5）；进度点前一回合（R3/R6/R9）eco 判 roll 后实际花超额度 58/89 题，57 题守住额度就能按时升
        "where": "rules.level_spare（同题面 f9ll 'can spend Xe now and still level up next round' 算法）；laya_policy._shop_decide：下回合是进度点、攒着升得起 → 去掉单次花费 > 额度的买 / 刷（主C 同名、真缺核心件除外）；eco_roll_follow 同样不超额度",
        "why": "执行经济网络自己的计划（题面写给它的额度）；商店题 / eco_roll_follow 不看额度把升级钱花掉",
        "teach": "规范 54.33（进度点前一回合守额度）+ 54.32-47 修正；eco876_e 重标后补训 eco / 商店题",
        "remove_when": "新学生 / eco 网络在进度点前一回合实际花费 ≤ 额度（复盘统计 0 次花超后下回合升不起）",
    },
    "buy_listed_carry": {
        "on": True,    # 2026-10-21 第 868 局：整局没开着的主C（亚连 / 韩信在店里两次、蒙犽在拍卖里，Laya 都没拿）→ 关掉的马可波罗顶到结束，第 6
        "where": "laya_policy._shop_decide / _auction：场上 + 手里没有棋手配置开着的 2 阶以上主C → 商店题只留买开着的主C（买得起时）、拍卖题只拍开着的主C；_shadow_carry：现任主C 在名单外、场上有开着的 2 阶以上主C → 换主C 题去掉'维持'只留开着的（第 869 局貂蝉）",
        "why": "执行棋手配置的主C 名单（data/lords/阿离/carry_pool.json）；线上 f9z 题面没有'名单内主C 应接任'这一行，R50 学生不会主动拿",
        "teach": "f9za 题面 listed carry (should take over) / switched off in Ali's carry list + 规范 54.32-6/12（R51）",
        "remove_when": "R51（f9za）上线后，学生在'没有开着的主C、店里 / 拍卖有'时买 / 拍 ≥0.8",
    },
    "bought_must_deploy": {
        "on": False,   # 10-21 第 879 局 R17 关闭：强制换人只剩"卖明世隐"（伞亚连引擎表没算他），随后 Laya 又卖刚上场的朵莉亚换回明世隐；R51 在这类题已 +0.061（E_bought_swap），撤销条件达成；此前 873 / 876 局也出过副作用
        # 原 "on": True,    # 2026-10-21 第 853 局 R6：Laya 0.35 买蒙恬、8 秒后换人题 0.39 选"不换"，3 能量白花
        "where": "laya_policy._swap：等位的人是 Laya 本回合 / 上回合商店题主动买的、或拍卖题拍下的（10-21 第 873 局露娜）、N 选 1 选中的（第 864 / 872 局鲁班大师 / 花木兰）新英雄（_laya_bought，不含合成同名、第 12 回合前的太乙）→ 有路线外的人可卖时：换人题去掉'不换'、只留卖外人（卖哪个外人由 Laya 定）；只能卖路线核心 / 最终阵容 / 引擎件 / 主C 时不强制（10-21 第 859 局：买云中君后被迫卖太乙）",
        "why": "执行模型自己已经做出的买牌决定（买英雄 = 要上场）；两道题各问各的、换人题不知道'这是我刚买的'",
        "teach": "规范 54.31-2 + 题面 f9za：商店题满员时写'买英雄 = 要卖一个人'并列出可卖的人；换人题写'本回合刚买的'",
        "remove_when": "新学生满员时买了新英雄、换人题自己不选'不换'（复盘 B2 买了又卖 / 买了不上 0 次）",
    },
    "buy_before_levelup": {
        "on": True,    # 2026-10-21 审查（规则 / 模型冲突第 10 条）：经济网络的"升级"在 Laya 商店题之前执行，升完主C 同名 / 核心件买不起
        "where": "rules.preempt_shop（模型接管分支）：经济网络判 levelup、路线已锁、店里有主C 同名或现在缺的核心件（route_core_missing_now）、钱 ≥ 牌价 + 升级费 → 先买这张，下一帧再升级",
        "why": "规范 54.29-3：钱够两样时先买牌（升级不改变商店）；执行顺序问题，升级本身仍由经济网络定",
        "teach": "54.29-3 已在规范里；商店题和经济题合并或经济题带上商店内容后撤",
        "remove_when": "经济网络 / 商店题在'钱够两样'时自己先买（复盘里 0 次升完买不起主C 同名）",
    },
    "awakened_no_copy": {
        "on": True,    # 2026-10-20 晚开（辅助觉醒后不再买同名，等棋手满级后再补，优先升主C 和棋手等级；盾山、李元芳觉醒后仍被买过同名）
        "where": "laya_policy._shop：棋手 < 满级（meta.MAX_LORD_LEVEL）时，商店题去掉场上已觉醒的非主C / 非路线主C 的同名（rules.awakened_no_copy；明世隐这类登场可叠加的除外）",
        # 10-21：明世隐不再例外（超核心件只凑到觉醒）
        "why": "觉醒后同名只加等级，非主C 的等级几乎不改战力；钱先给主C 同名 / 加等级牌 / 升棋手",
        "teach": "规范 54.29-13 + 题面 f9za 给已觉醒非主C 的同名标 'awakened: copy only adds levels; buy after lord max level'",
        "remove_when": "新学生在满级前对已觉醒非主C 同名 ≤0.1",
    },
    "talent_low_value": {
        "on": True,    # 2026-10-19 开（节能商店没什么用，−1 能量不一定落在需要的那张上）
        "where": "laya_policy._talent_decide：LOW_VALUE_TALENTS（节能商店、限制性进化、随机成长、随机强化、随机进化、成年礼、群星璀璨、扣自己血的兵行诡招 / 极限操作 / 搏命、拾荒者；主C 不是攻速型时还有因缘际会；2026-10-20 起再加 data/talent_conditions.json 判 C / 硬条件不满足的）从天赋三选一里去掉（还剩别的选项或刷新时）；2026-10-20 第 839 局起管刷新：第 8 回合起三个全是 C 才刷新（只留刷新），其余去掉刷新（前期青铜天赋刷新收益低，只有全是 C 时刷新）；特殊高级装备三选一（非基础 / 四选一成品）三件都配不上主C → 不分回合刷新（10-20 第 844 局：静谧之铃这一档要支持刷新，普通装备不刷）",
        "why": "第 782 局 R2 选了节能商店（降价落在随机 1 张卡上）；限制性进化（单主C 阵容退场的都是不需要的英雄、等级也不高；第 762 局 R5 选过）；随机成长（加给最高品阶，主C 一般不是）、成年礼（送的多是别的阵营，核心套路同阵营）—— 10-19",
        "teach": "题面 f9w 写'discount lands on a random card'+ 规范 54.25 → 重标",
        "remove_when": "新学生在有其他像样选项时给节能商店 ≤0.1",
    },
    "sunset_equip_buy": {
        "on": True,    # 2026-10-19 开（确认：日落海英雄当主C、装备没满时，商店出日落海装备可以买）
        "where": "laya_policy 商店题前：主C 日落海、穿戴账本 <3 件、钱够 → 买卡面含『【日落海】专属装备』的牌（远洋宝藏）",
        "why": "第 706 局 R13 主C 亚连 2/3 件，Laya 买 0.31 < 刷新 0.33 没买",
        "teach": "规范 54.25：日落海主C 未满装时日落海专属装备 ≥0.7 → 重标",
        "remove_when": "新学生在这类题 buy ≥0.6",
    },
    "maple_cc_immune": {
        "on": True,    # 2026-10-19 开（主C 要有免控，免控也可以用枫叶舞补；装备仍优先吸血装）
        "where": "laya_policy.pick_lord_boost：主C 没有免控装（炽热支配者 / 星源装甲 / 机器头）且没选过枫影无拘 → 三选一里有枫影无拘就选",
        "why": "主C 被控就打不出输出；装备位留给吸血装（高攻速防反伤甲），免控由枫叶舞强化补",
        "teach": "题面 f9w lord_boost 写'carry has no CC immunity'+ 规范 54.25 → 重标",
        "remove_when": "新学生在这类题给枫影无拘 ≥0.6",
    },
    "special_equip_output": {
        "on": True,    # 2026-10-19 开（主C 没满装时特殊装备优先选输出）
        "where": "laya_policy._card_pick：N 选 1 全是装备、主C <3 件 → 选对主C 适配分最高的输出装；高攻速主C 没吸血装时吸血优先",
        "why": "珍品宝箱等特殊装备 N 选 1 原来交 Laya / 规则按通用分选，会拿到防御 / 辅助装",
        "teach": "题面 f9w card_pick 装备选项写 carry n/3 + 输出/吸血 + 规范 54.25 → 重标",
        "remove_when": "新学生在主C 未满装时选输出装 ≥0.6",
    },
    "talent_offroute_gift": {
        "on": True,    # 2026-10-19 开（起因：选了没用的献祭之赐）
        "where": "laya_policy._talent_decide：路线已锁，data/talent_hero_ties.json 里 gives / duo / about 天赋点名的英雄全不在路线最终阵容 / 核心引擎 / 主C → 去掉（还剩别的选项时；gives_plus 如 X来客 不去）",
        "why": "第 787 局 R9 主C 沈梦溪，献祭之赐送李信，Laya 0.58 选、紧接着 sell_hand 0.77 卖掉",
        "teach": "题面 f9w 写 gives X: not in route + 规范 54.25 → 重标",
        "remove_when": "新学生在这类题 ≤0.1",
    },
    "not_wanted_no_buy": {
        "on": False,   # 2026-10-19 晚开、第 797 局后关（撤掉该临时规则）——交给 f9x 题面 + 训练
        "where": "laya_policy._shop：路线已锁，引擎表 not_wanted 的人（狄仁杰线钟馗）从商店题买选项去掉",
        "why": "第 796 局锁伞狄速射（当时没引擎表），R47（f9w）题面写钟馗 ROUTE final team，连买两张",
        "teach": "题面 f9x：not_wanted 不进 ROUTE final team / missing / CORE 计数 + 规范 54.26-2 → 重标",
        "remove_when": "f9x 学生在 not_wanted 的人上 buy ≤0.05",
    },
    "lock_needs_wanted": {
        "on": True,    # 2026-10-19 开（几次锁店都没有道理）
        "where": "laya_policy._shop_action：Laya 选 lock 但店里没有主C 同名 / 路线成员 / 核心引擎件 → 不执行（按概率次序换下一个动作）",
        "why": "第 788 局 R7 锁 芈月/刘禅/蒙恬/苏烈（Laya 0.53）、R8 锁 西施/苏烈/钟馗/换帅（0.63），没有要的牌，白丢下回合新商店",
        "teach": "规范 54.25：lock 只为留要的牌 → 重标",
        "remove_when": "新学生在店里没有要的牌时 lock ≤0.1",
    },
    "coin_machine_after_carry": {
        "on": False,   # 2026-10-19 R45 上线后关（87 道古币机题 R45 只 1 题首选、与老师一致；中位 0.112）；原开：    # 2026-10-19 开（古币机选得太多，它偏团队收益）
        "where": "laya_policy._talent_decide：主C（现任 / 路线主C）已定时，天赋三选一去掉古币机",
        "why": "第 736 局 R9（0.42）、第 758 局 R11（0.46）选了古币机；题面被装备正则误伤只剩 'carry X n/3 items'，规范 627/705 行还把它写成'比 +2 级值钱''路线配套优先'；古币机在场后古币全给它，河洛主C 每张少 +2 永久级",
        "teach": "题面 f9u 写清「分走主C 古币」+ 去掉装备正则误伤；规范 54.23-7 → R45",
        "remove_when": "新学生在主C 已定时给古币机 ≤0.1",
    },
    "final_team_first": {
        "on": True,    # 2026-10-18 开（同时复查其他坦克英雄，包括苏烈）
        "where": "laya_policy._swap：路线已锁、第 5 回合起、等位的是最终阵容件（不含太乙类后期件）→ 换人选项只留最终阵容外的人（去掉最终阵容件和'不换'）；10-19 起核心引擎件（min_engine early_priority）等位同样处理（第 782 局李元芳）；10-19 晚第 792 局：第 12 回合起太乙等位也只卖外人（没外人不换），'太乙让位'只在场上没有外人时执行（李元芳 / 太乙来回卖）",
        "why": "第 747 局牛魔开局件卖不掉；第 724~747 局第 7 回合后苏烈 72/77、廉颇 24/29、牛魔 21/28 次被留着（规范 39/41'别卖主坦'+题面'唯一前排'）",
        "teach": "规范 54.20 + 题面 f9t（无回合路线开局阶段到第 6 回合）→ R44",
        "remove_when": "R44 换人题在最终阵容件等位时自己卖最终阵容外的人",
    },
    "route_lock_needs_carry": {
        "on": True,    # 2026-10-18 开（"定好了不能换路线"的前提：路线真的定好 = 路线主C 到手）
        "where": "laya_policy._route：Laya 选的路线主C 不在场上 / 手里 → 不锁，下回合再问",
        "why": "第 739 局 R4 以 0.36 锁守约路线，守约整局没来；实际阵容是日落海，9/9 海诺被当路线外 bench 两次",
        "teach": "规范 54.18-9（route 题路线主C 没到手 ≤0.3）→ R43",
        "remove_when": "R43 上线后路线题不再锁主C 没到手的路线",
    },
    "eco_roll_follow": {
        "on": True,    # 2026-10-18 开（路线定好后不换，要在对应棋手等级多刷几次）
        "where": "laya_policy._shop_action：经济网络本回合判 roll、路线已锁且核心还缺（不含后期件）、商店里没有缺的件、刷完还买得起 → 商店选 stop 改为继续刷（每回合 ≤ ECO_ROLL_MAX=6）",
        # 10-21：第 12 回合起不替商店追刷缺件（后期认命，凑主C 等级）
        "why": "第 735 局 R9：eco roll、缺明世隐，商店题写 'only merges help'，免费刷完就停、留 10e；明世隐 R11 才到",
        "teach": "题面 f9s `FULL: missing core X would replace Y` + 规范 54.18-8 → R43",
        "remove_when": "R43 上线后商店题在 eco roll + 缺核心件时自己选付费刷新",
    },
    "late_fill_before_r12": {
        "on": True,    # 2026-10-18 开（第 734 局：太乙上得太早占了位置；太乙的复活到后期才明显，第 12 回合前降低优先级，没位置可以不上）
        "where": "laya_policy._swap：第 12 回合前，等位的后期件（min_engine late_fill：太乙这类）不出换人题、不为它卖人（有空位照常上）——10-21 第 881 局起例外：路线锁了且场上有路线外的人（不在路线 / 核心 / 引擎件 / 主C、没觉醒，如庄周）→ 太乙也出换人题，但只许卖这些外人；核心 / 最终阵容件等位而场上有后期件 → 只能卖后期件（任何回合，太乙都要让位）；rules.MIN_ENGINE_UNTIL 8→11",
        "why": "第 734 局 R9 卖狂铁上太乙，明世隐到手没位置、卖刘禅；题面 f9q 写的是 'add after round 8'",
        "teach": "题面 f9s（LATE-FILL 第 12 回合前低优先、之后按最终阵容其余件到齐）+ 规范 54.18-3 → R43 训练",
        "remove_when": "R43（f9s）上线、换人题对第 12 回合前的太乙不再选卖人",
    },
    "route_carry_locked": {
        "on": True,    # 2026-10-18 开（路线定好后不换，要在对应棋手等级多刷几次）
        "where": "laya_policy 换主C 题：路线已锁、现任主C 就是路线主C 且在场 → 不出换主C 题（维持）",
        "why": "最终目标永远是凑推荐最终阵容；726 / 728 局锁路线后局面网络把亚连换成干将 / 安琪拉，路线实际被换掉",
        "teach": "cgnet carry 标签：路线主C 在场时 keep ≥0.9（规范 54.18）后重训",
        "remove_when": "cgnet carry 在'路线主C 在场'的题上不再选换人",
    },
    "temp_carry_no_swap": {
        "on": False,   # 2026-10-15 关（A1 一部分）
        "where": "laya_policy._shadow_carry：临时主C 不换成另一个临时主C（temps → allowed）",
        "why": "前期 1~5 回合临时主C 只需要 1 个（第 599 / 603 局孙尚香 → 孙膑）",
        "teach": "carry 模拟器加'临时主C 很快被顶掉'的成本；cgnet carry 重训",
        "remove_when": "cgnet carry 在临时主C 互换题上不再选互换",
    },
    "temp_carry_no_copies": {
        "on": False,   # 2026-10-13 R27（f9lm / g9y）上线后关：学生已学到（test_r9zb_new top1 0.706→0.770，tmass +0.039）
        "where": "laya_policy._shop_decide（临时主C 不买同名凑觉醒）+ rules._merge_buy",
        "why": "临时主C 不值得多买同名凑觉醒；干将买了 4 张属于浪费",
        "teach": "Laya 题面写'temporary carry'，规范：钱留给最终主C 配套",
        "remove_when": "Laya 新学生上线、复盘不再出现给临时主C 买同名",
    },
    "flex_buy": {
        "on": False,   # 2026-10-15 第 641 局关：把别的路线的主C 也当核心卡买（马可波罗 + 沈梦溪 → 双主C）；规则层修补会在新模型上再次出错
                       # —— 不打补丁，整条关掉，交给 Laya（R30b 起题面有 CORE of route / FULL + N waiting）    # 2026-10-13 晚重新打开：第 620 / 622 局 Laya 没买同体系 / 其他路线核心（沈梦溪 / 狄仁杰在店不买，守约不来只能用米莱狄）；
                       # 但多买其他套路的公共核心英雄这一打法仍要保留 —— 等 4 点那轮新学生学会再关
        "where": "rules._flex_buy / flex_core_cards（先于 Laya）+ laya_policy 商店题'只买一张'",
        "why": "设计打法：多个路线的配套核心卡见到就买、只买一张；定好最终主C 后疯狂买配套",
        "teach": "Laya 题面标出各候选路线核心卡 / 已有张数 / 是否已定最终主C；规范写这套打法",
        "remove_when": "Laya 新学生在这类商店题上自己这么买",
    },
    "flex_buy_settled": {
        "on": False,   # 2026-10-15 第 632 局关：主C 定了之后规则抢先"买配套"（孙膑 R4、钟馗 ×2 R6、满员时 R15~R18 连买），手牌堆到 13/15、
                       # 认牌失效整回合空转。买哪个配套是价值判断，交给 Laya 商店题（g9y 50.2.3 / g9za 已教"主C 定了买配套"）
        "where": "rules._flex_buy（settled 分支）",
        "why": "定好最终主C 之后集中买配套",
        "teach": "Laya 题面：配套 / 喂主C / 满员要换人才能上、手里已有几个在等；规范 50.2.3 + 53",
        "remove_when": "已关；Laya 商店题在主C 定了之后能自己买配套、满员时不乱买",
    },
    "talent_dead_faction": {
        "on": False,   # 2026-10-13 R27（f9lm / g9y）上线后关：学生已学到（test_r9zb_new top1 0.706→0.770，tmass +0.039）
        "where": "laya_policy._drop_dead_faction_talents",
        "why": "第 605 局擂台猛虎用不上；选天赋必须现在和未来用得上",
        "teach": "Laya 天赋题面写'我们场上 / 候选路线有几个【X】英雄'",
        "remove_when": "Laya 新学生不再选用不上的派系天赋",
    },
    "talent_dmg_mismatch": {
        "on": False,   # 2026-10-13：把问题描述清楚、让模型选择，而不是靠删选项修 → 不删选项，改写进题面 f9ln
        "where": "laya_policy._drop_mismatch_talents（装备类天赋与主C / 路线主C 伤害类型相反 → 去掉）",
        "why": "第 614 局：噬神之书是法术装，孙尚香 / 虞姬都是物理，没法给主C 用",
        "teach": "Laya 天赋题面写出装备伤害类型 vs 目标伤害类型（mag item, target phys → MISMATCH）",
        "remove_when": "题面带伤害类型的 Laya 新学生上线，复盘不再选错配装备天赋",
    },
    "eco_roll_no_stop": {
        "on": False,   # 2026-10-13 审计：每局删 stop/lock 14~51 次，逼学生满员买杂牌再被规则卖掉（606/611/613/614）；改为描述问题、不删选项
        "where": "laya_policy 商店题：经济网络说 roll → 去掉 stop / lock",
        "why": "2026-10-11 第 564 局 eco 说 roll，Laya 刷 1 次就停、囤钱",
        "teach": "f9ln 商店题面写 'economy plan this round: roll'；规范 g9z 教'roll = 为缺的件刷，满员不买杂牌'",
        "remove_when": "f9ln 学生上线后删代码",
    },
    "swap_spare_for_route": {
        "on": False,   # 2026-10-15 关（A14）
        "where": "rules.spare_swap（decide_all 末尾：满员、路线阵容里的人在手里等、场上有路线外的备用卡 → 卖最弱的备用卡）",
        "why": "第 627 局：备用卡可以先上场（总比空着好），但拿到配件后应卖掉备用卡（换人网络判 bench 0.54，铠 / 程咬金上不去）",
        "teach": "swap 题面 / cgnet 特征按核心阵容三档（最终阵容 / 阶段件 / 补位件）+ 等位置者角色；重标「满员、配件在等、场上有补位件」的题",
        "remove_when": "新的局面网络在这类 swap 题上自己选卖补位件",
    },
    "carry_value_filter": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "rules._carry_eligible(hard_only=True)：模型选主C 时按 1 阶 / 审定评分 / 队友喂不饱 / 辅助坦克 / 阵容频率删候选（A1，5 局 44 次，日志伪装成'合法性'）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "carry（局面网络）+ 题面 carry_why_not",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "placeholder_carry": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "rules._decide_deploy 末尾：主C 空缺时第一个合格的人上场就设成主C（A2）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "carry（局面网络；只剩一个候选时 _yield_single 直接接任）",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "maple_fallback": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "rules._skill_target：没有主C 时规则按估值'凑合绑'枫叶舞（A2）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "carry 定了再绑",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "placeholder_yield": {
        "on": True,   # 2026-10-21 重新打开（第 851 局：这时应把亚连换成主C——1 阶孙尚香占位、亚连在场，局面网络一直 keep）；接班人限主C 名单开着的
        #   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "laya_policy._shadow_carry：1 阶占位有接班人时删换主C 题的'维持'（A3）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "carry",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "sell_level_engine": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "rules.decide_sell_unit 弈星分支：升过级就卖（A4）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "swap",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "use_card_rule_targets": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "laya_policy._use_card：不在 use_card_question 范围的牌（众星拱月 / 求贤 / 跳级…）目标由规则定（A5）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "use_card",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "trigger_no_hold": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "laya_policy._use_card：触发类牌有合规目标时删'先留着'（A6）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "use_card",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "swap_engine_parts": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "laya_policy._swap：引擎锁定后删引擎零件候选（A7）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "swap",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "transfer_guards": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "rules._transfer_victim：模型选的新主C 收尾时，引擎零件 / 最低配套件 / keep_as_support 不卖（A8）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "carry / swap",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "transform_route_carry": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "laya_policy._use_card：变身 / 移除牌删路线主C 目标（A9；现任主C 仍删 = 安全约束）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "use_card 题面",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "card_pick_dmg": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "laya_policy._card_pick_answer：伤害类型相反改选（A10）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "card_pick 题面",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "lord_boost_veto": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "rules.decide_lord_boost：'阵亡'类 / 红叶染锋否决（A11）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "lord_boost",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "shop_no_room": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "laya_policy._shop_action：满员且手里 ≥2 人在等不买新英雄（A12）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "shop 题面 FULL + N waiting",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "shop_refresh_floor": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "laya_policy._shop_action：刷完剩 <3 能量不刷（A12）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "shop 题面 can't afford a hero after",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "duo_no_sell": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "rules.decide_all + laya_policy._swap：共战两人不卖（A13）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "swap 题面",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "awakened_sell_pick": {
        "on": False,   # 2026-10-15 关（规则审计：与模型冲突的规则一律关掉）
        "where": "rules.decide_sell_unit for_awakened：觉醒卡腾位由规则挑卖谁（A15；现在由换人网络挑，题里去掉 bench）",
        "why": "规则替模型做价值判断（A 类：和现有模型题型重叠）",
        "teach": "swap",
        "remove_when": "R31 及之后实战复盘没有因此出现的错误 → 删代码",
    },
    "card_usage_when": {
        "on": True,
        "where": "rules._usage_when_ok（data/card_usage_reviewed.json 的 when：round>= / gold / field / hand_room / hp / contract…，119 张牌）",
        "why": "契约瓶写死第 9 回合起才用导致长期不用，且属于写死的价值规则（现改为按输赢推算张数的兜底 contract_ready）",
        "teach": "Laya 新题型 use_timing（现在用 / 先留着，f9lt 起影子题）：题面写卡面 + 算好的事实（契约瓶攒了几张、近几场胜率、兑现阈值）；老师标注后训练",
        "remove_when": "Laya use_timing 学生上线、复盘里这类牌的使用时机由模型决定且不退步；只保留合法性条件（tag_on_field / faction_on_field：场上没有作用对象）",
    },
}


def _lord_override(name: str):
    """按棋手覆盖开关（data/lords/<当前棋手>/interim_overrides.json，可选：{规则名: true/false}）；没写 → None。2026-10-22"""
    try:
        from agent.decision import lord_config as _LCi
        return _LCi.load().interim_override(name)
    except Exception:                                       # noqa: BLE001
        return None


def on(name: str) -> bool:
    ov = _lord_override(name)
    if ov is not None:
        return bool(ov)
    return bool((RULES.get(name) or {}).get("on"))


def active() -> list[str]:
    return [f"{k}：{v['where']}（撤的条件：{v['remove_when']}）" for k, v in RULES.items() if on(k)]


if __name__ == "__main__":
    print("当前还开着的临时规则（模型学会后要撤）：")
    for line in active():
        print(" -", line)
