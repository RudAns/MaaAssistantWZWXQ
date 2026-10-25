"""标注流程工具：把决策点导出成"标注视图"（中文、完整信息、**隐藏规则引擎的选择和名次**），回收并校验标签。

  export  按类型/数量抽样，切成批次，写 data/laya/labels/<ver>/batches/batch_XXX.md（+ .ids.json）
          --dup 0.1：再抽 10% 做第二遍独立标注（衡量标注一致性），写 dup_XXX.md
  merge   读 data/laya/labels/<ver>/out/*.jsonl（标注员写的），校验后合并成 labels.jsonl / labels_dup.jsonl
  status  看进度

标签格式见标注规范第 0 节。
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))
import features as F  # noqa: E402

GUIDE_VERSION = "g9zza"  # g9zza（2026-10-20 下午）：第 54.28 节（r9zze 老师冲突 21 条 + 打法裁决：屏蔽安琪拉、狄仁杰线带铠、特殊阵位天赋不选）。 g9zz（2026-10-20）：第 54.27 节（r9zzd 老师冲突 + 10-20 卡面机制确认：天赋硬条件表、幸运硬币、屏蔽路线主C、雅典娜 4 阶、刚上场不卖）。 g9zy（2026-10-19 夜）：第 54.26 节（r9zzc 老师冲突 + 晚间打法裁决：临时宿主不选红叶染锋、低价值天赋审查、共战两人都要有用、少司缘补位、上不了场不买）。 g9zx（2026-10-19 傍晚）：第 54.25 节（下午打法裁决 + r9zzb 老师冲突裁决）。 g9zw（2026-10-19 午）：第 54.24 节（机器头只给木兰嬴政、辅助不给装备、路线战绩、先启动引擎、明世隐同名、刷新 1 能量）。 g9zv（2026-10-19）：第 54.23 节（共战送牌、未来主C 阵营、LATE-FILL 只认复生、雅典娜核心、古币机团队天赋、法术不暴击、机器头锁攻速）。 g9zt（2026-10-19）：第 54.20~54.22 节（最终阵容件等位只卖阵容外 / 组合天赋要主C / 落后进度先升级攒钱 / 雅典娜再触发整备 / 唤醒给明世隐 / 卡面机制确认）。 g9zq（2026-10-18 下午）：g9zq（2026-10-18 下午）：第 54.18 节（打法裁决：最终阵容唯一目标、路线定了不换、缺件对应等级多刷、LATE-FILL 最后补；拍卖满员只拍路线件；不卖刚上场的人）。 g9zp（2026-10-18 下午）：g9zp（2026-10-18 下午）：第 54.17 节（r9zv 老师冲突：PAID-OFF 弈星可卖给任何能上场的人、主C 未定时 OTHER LINEUP 不触发 bench、双重身份、满级刷新找合成、前期无买人选项先升级）。 g9zo（2026-10-18）：g9zo（2026-10-18）：第 54.16 节（r9zt 老师 25 条：卖人统一顺序、过阶段件例外、满员封顶条件、1 阶临时主C 不算已定、前期补位、坦克不是别家主C）。 g9zn（2026-10-17 晚）：第 54.15 节（r9zs 老师 66 条冲突裁决：前期件最先卖 / 产出方分档 / 喂主C 看阵营 / 前期商店统一 / 满员路线件 / 已定主C 统一）。 g9zm（2026-10-17 晚）：第 54.14 节（10-17 下午打法裁决：核心件见到就买 / 雅典娜 > 太乙 / 狂铁类引擎产出方 > 路线外条件喂养 / 前浪的荣光尽早用 / 杨玉环退场只在场上卖 / 远洋宝藏 / 五阶锦囊同回合用 / 刘邦最先卖 + r9zr 老师 29 条）。 g9zl（2026-10-17）：第 54.13 节（r9zq 老师冲突裁决：补位 / 免费刷新 / 升级先后、路线外等位、多人等位、双标签、f9n 写法、路线题）。 g9zk（2026-10-17）：第 54.12 节（天赋阵营 DEAD / 前期补空位 / 点兵点将中间档 / 换人卖谁先后 / 三档 SECONDARY / 遗志 / 武装分解 / 珍品宝箱特殊装备）。 g9zj（2026-10-16 晚）：第 54.11 节（别家主C 低于核心牌、免费刷新优先、满员路线外不买、换人顶掉小额喂等级的路线外单位、点兵点将核心≥4/6 用、裂隙放等级最高非主C、sell_hand 路线身份、卡面事实）。 g9zi（2026-10-16 上午）：第 54.10 节（15 条打法裁决：换绑不看宿主多高 / 开局只绑可卖临时主C / 弈星等补位再卖 / sell_hand / 路线不定死 / 裂隙放主T / 点兵点将 / 前浪的荣光 / 武装重构 + r9zn 老师反馈裁决）。 g9zh（2026-10-16）：第 54.9 节（打法裁决：阿离只养一个主C、别家主C 没位置、第二输出位只放装备；枫叶舞换绑给接班主C；弈星纯过渡回本就卖、sell_filler 题；路线缺件等位先卖占位件；点兵点将看核心占比；混沌裂隙放等级第二高的非辅助、天赋混沌次元避开）。 g9zg（2026-10-15 夜）：第 54.8 节（f9lv：引擎零件只标卖了就断的、冗余零件可卖、开局件过阶段、弈星剩余价值、前排换人、条件天赋）。 g9zf（2026-10-15）：第 54.7 节（天赋按现在值多少判、商店配合经济计划、换人别卖引擎 / 燃料、普通免费刷新留到下回合）。 g9ze（2026-10-15）：第 54.5 / 54.6 节（满级或攒钱无用时刷新找合成、路线配套件能换杂牌就买、f9lt 新读数）。 g9zd（2026-10-15）：第 54.4 节（f9ls：燃料行只算能用在主C 身上的效果牌、变身 / 移除类用牌题不写燃料行、燃料件在手不算稳定、锁店列非英雄牌）。 g9zc（2026-10-15）：第 54 节（f9lr：反应器 idle、主C 燃料、满员还有人在等；前期升级 vs 补空位、换人先卖谁等冲突裁决）。 g9zb（2026-10-15）：第 53 节（f9lq：人口补全、条件天赋触发率、换人题路线成员在等 vs 场上低等级多余输出、roll 看主C 觉醒）。 g9za（2026-10-14）：第 52 节（f9lp：现在买就凑齐 COMPLETES、流派选项 FLOW OPTIONS；推荐阵容是终点、中间往不同流派靠、能养起来就强）。 g9y（2026-10-13）：第 50 节（f9lm：临时主C 不养、候选路线核心卡见到就买一张、最终主C 定后疯狂买配套、天赋派系要用得上、攒钱要有下回合用途）。 g9x（2026-10-12 晚）：第 49 节（f9ll：主C 用不了的 5 阶神装、CAN WEAR 标签）。 g9w（2026-10-12）：第 48 节（f9lk：主C 资格、同体系备选 BACKUP ROUTE / CARRY CANDIDATE、人口按等级表、引擎按场上）。 g9v（2026-10-11）：第 47 节（f9lj：路线最低配套 MIN ENGINE / LATE-FILL；先凑最低配套，再凑其他英雄）。 g9u（2026-10-11）：第 46 节（f9li：已选天赋 → 结论、满员时 lineup 行、C-POSITION 标签）。 g9t（2026-10-11）：第 45 节（f9lh：主C 觉醒差几张 + 各级见同名概率、满员、攒一回合够不够升级；攒钱升棋手收益更高）。 g9s（2026-10-11）：第 44 节（eco 题不再按 42.2/42.7 的顺序表套条款，改为理解取舍自己判断）。 g9r（2026-10-10）：第 43 节（f9lf 三类新事实：刷完买不起英雄的刷新 ≤0.05；发英雄牌按 expect 定价；专属天赋按 42.6）。 g9q（2026-10-09）：第 42 节（eco 经济节奏题：停级搜牌 vs 升级开池 vs 攒钱；1 阶临时主C 不互换；swap 别卖枫叶舞宿主；发英雄牌按期望拿到阵容要的人定价；6 级专属天赋效果）。 g9p：第 41 节（阿离单主C 总纲落到 carry / swap：已投入的主C 不换给低等级的人、不要第二个 C、阵容 = 主C + 喂主C + 功能辅助 + 一个主坦、现任转正；41.4 红叶染锋只在之后还要换主C 时有用）。 g9o：第 39 节（养主坦不养副核）、第 40 节（主C 定了不拿第二主C，优先喂主C 的英雄）。 g9n：第 38 节。 g9m：第 37 节（前三 vs 5~6 名复盘：守约优先、日落海核心、天赋刷新）。  # g9l：第 36 节（f9l 题面：失效路线 stale、引擎零件 vs 第二主C、装备满 3/3、N 张合计、枫叶舞强化价值表）。 g9k：第 35 节（只改规范）。 g9j：第 34 节（官方路线 ROUTE + route 题）、第 30 节更正。 g9i：第 33 节（等级引擎）。 g9h：第 32 节（第 30 > 31 节、被替者现况、有福同享受益人）。 g9d：化学反应 + 锁店（第 27 节）。g9c： g9c：没空位时写场上最弱的单位（第 26 节）。g9b： g9b：swap 的 levelup 选项（第 25 节）。g9： g9（2026-09-25 晚）：f9 题面（第 24 节）。g8： g8（2026-09-25）：f8 题面（第 23 节）。g7：第 21 节 carry 题、第 22 节 swap 题。g6： g6（2026-09-25 夜）：f7 题面；规范第 20 节（拍卖时序更正、关键词、卡库没有的卡）。g5：f6 题面、第 17、18 节
# g4（2026-09-24，和学生 f5 输入对齐 —— 老师只依据学生同样能看到的信息）：
#   去掉血量与名次（玩家血量不参与判断：输了就是输了）；场上每人写等级 / 觉醒进度 / 是否留场 / 是否后期角色；
#   写出当前与升一级后的品阶刷新概率；刷新、升级选项写出与学生输入相同的概率数字（同一套 features 函数算）。


def _cards():
    return F._cards()


def hero_brief(name: str) -> str:
    c = _cards().get(name) or {}
    if not c:
        return f"{name}（卡库无此牌）"
    tags = "/".join(c.get("tags") or [])
    cs = F._carry_scores().get(name) if c.get("category") == "英雄牌" else None
    return (f"{name}（{c.get('category', '')} {c.get('quality', '?')}阶 {c.get('faction') or '无阵营'} {tags}"
            + (f"｜主C评分 {cs}/9" if cs is not None else "") + "）")


def card_full(name: str, desc: str = "") -> str:
    c = _cards().get(name) or {}
    skill = (c.get("skill") or desc or "").replace("\n", " ")
    head = hero_brief(name) if c else f"{name or '未识别'}（卡库无此牌）"
    return f"{head} 效果：{skill[:160]}" if skill else head


def render_point(p: dict) -> str:
    if GUIDE_VERSION >= "g5" and (p.get("feature_version") or "") >= "f6":
        # g5（question-design-v2.md）：老师看的**就是学生的输入原文**（f6 facts.py 生成），逐字同源 —— 信息对齐最严格。
        crit = p["question"]["criteria"]
        return "\n".join([f"### id={p['id']}  类型={p['kind']}", "题目：" + p["question"]["instructions"],
                          "```", p["state"], "```", "选项（键: 学生看到的原文）："]
                         + [f"- `{k}`: {v}" for k, v in crit.items()]
                         + [f"（可选键：{', '.join(crit.keys())}）"])
    ctx = p.get("ctx") or {}
    lines = [f"### id={p['id']}  类型={p['kind']}"]
    lines.append(f"回合 {ctx.get('round')}"
                 + f"｜能量 {ctx.get('gold')}｜棋手等级 {ctx.get('level')}"
                 + ("（读数与商店品阶矛盾，已按品阶修正）" if ctx.get("level_fixed") else "")
                 + ("（满级，不能再升级）" if (ctx.get("level") or 0) >= 6 else "")
                 + (f"（升级费 {ctx.get('level_cost')}）" if (ctx.get('level_cost') or -1) > 0 else "")
                 + (f"（经验 {ctx['xp']}/{ctx['xp_needed']}）" if (ctx.get('xp_needed') or 0) > 0 else "")
                 + f"｜人口 {ctx.get('pop')}/{ctx.get('pop_cap')}｜禁用派系 {ctx.get('banned') or '-'}")
    extra = []
    if ctx.get("auction_in") is not None:
        extra.append(("本回合是拍卖回合，" + ("拍卖已经结束" if ctx.get("auction_done") else "拍卖还没开始"))
                     if ctx["auction_in"] == 0 else f"距下次拍卖 {ctx['auction_in']} 回合")
    if "free_refresh" in ctx:
        extra.append(f"免费刷新 {ctx.get('free_refresh') or 0} 次")
    if ctx.get("units_seen") is not None:
        extra.append(f"画面上看到 {ctx['units_seen']} 个单位"
                     + (f"（读到的等级 {ctx['unit_levels']}）" if ctx.get("unit_levels") else ""))
    if "hand" in ctx:
        extra.append(f"手牌 {len(ctx.get('hand') or [])} 张")
    lv = F._level(ctx)
    if lv:
        extra.append("商店品阶概率 " + F._odds_str(lv) + (f"；升一级后 {F._odds_str(lv + 1)}" if lv < 6 else ""))
        tgt = F._meta().level_target(ctx.get("round") or 0)
        if tgt and lv < tgt:
            extra.append(f"棋手等级落后于进度（第 {ctx.get('round')} 回合应到 {tgt} 级）")
    if p.get("board") or ctx.get("field"):
        names = list((p.get("board") or ctx.get("field") or {}).keys())
        extra.append(f"场上前排坦克 {sum(1 for n in names if ' tank' in F.card_en(n, short=True) or 'frontline' in F.card_en(n, short=True))} 个")
    if extra:
        lines.append("｜".join(extra))
    carry = ctx.get("carry")
    lines.append("主C：" + (card_full(carry) if carry else "未定"))
    field = p.get("field") or []
    board = p.get("board") or {}

    def unit_line(n):
        b = board.get(n) or {}
        s = hero_brief(n) + (f" {b['lv']}级" if b.get("lv") else "")
        left = b.get("awk_left")
        if n in board:
            s += ("｜觉醒进度读不到" if left is None else "｜已觉醒" if left == 0 else f"｜觉醒 {3 - left}/3（还差 {left} 张同名）")
            s += f"｜牌池剩 {b['pool_left']} 张" if b.get("pool_left") is not None else ""
            s += "｜留到最后" if b.get("keep") else ""
            s += "｜推荐阵容后期角色" if b.get("late") else ""
        return s
    lines.append(("场上：" if board else "场上（账本近似）：")
                 + ("；".join(unit_line(n) for n in field) if field else "未知/空"))
    hand = ctx.get("hand") or []
    if hand:
        cnt = collections.Counter(hand)
        awk = set(ctx.get("hand_awk") or [])
        lines.append("手牌：" + "、".join(f"{hero_brief(n)}" + (f"×{k}" if k > 1 else "")
                                        + ("【已觉醒，等待上场】" if n in awk else "") for n, k in cnt.items()))
    lines.append("选项：")
    raw = p.get("raw") or {}
    crit = p["question"]["criteria"]
    if p["kind"] == "card_pick":
        for i, n in enumerate(raw.get("cards", [])):
            k = sum(1 for h in hand if h == n)
            lines.append(f"- opt{i}: {card_full(n)}" + (f"（手里已有 {k} 张）" if k else ""))
    elif p["kind"] in ("talent", "lord_boost"):
        items = raw.get("talents") or raw.get("boosts") or []
        for i, (n, d) in enumerate(items):
            lines.append(f"- opt{i}: {card_full(n, d)}")
        if "refresh" in crit:
            lines.append("- refresh: 刷新这三个天赋（本级只能刷一次）")
    elif p["kind"] == "shop":
        for slot, n, cost in raw.get("shop", []):
            if f"buy{slot}" not in p["options"]:
                continue                    # 同名牌只保留一个选项（f3）
            k = sum(1 for h in hand if h == n) + (1 if n in field else 0)
            lines.append(f"- buy{slot}: 买 {card_full(n)}｜价格 {cost}" + (f"｜已拥有 {k} 张" if k else ""))
        # 刷新 / 升级：和学生输入逐字同源的概率说明（F.q_shop 里算好的英文附注，翻成中文标签）
        extra_ref = crit.get("refresh", "").split(" | ", 1)[1] if " | " in crit.get("refresh", "") else ""
        extra_lv = crit.get("levelup", "").split(" | ", 1)[1] if " | " in crit.get("levelup", "") else ""
        if "refresh" in crit:
            lines.append("- refresh: 刷新商店（" + ("免费" if ctx.get("free_refresh") else "2 能量") + "）"
                         + (f"｜{extra_ref}" if extra_ref else ""))
        if "levelup" in crit:
            lc = ctx.get("level_cost") or -1
            lines.append((f"- levelup: 升级棋手（{lc} 能量）" if lc > 0 else "- levelup: 升级棋手（升级费没读出来）")
                         + (f"｜{extra_lv}" if extra_lv else ""))
        lines.append("- stop: 不买了，关店留钱")
    lines.append(f"（可选键：{', '.join(p['options'])}）")
    return "\n".join(lines)


def load_points(ver: str) -> list[dict]:
    f = PROJECT_DIR / "data" / "laya" / "datasets" / ver / "points.jsonl"
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]


def cmd_export(args) -> int:
    pts = load_points(args.ver)
    if args.only_new:
        # 迭代轮次：只标本轮新加的题（build_round.py 写的 new_ids.json），旧题沿用上一版标签
        keep = set(json.loads((PROJECT_DIR / "data" / "laya" / "datasets" / args.ver / "new_ids.json")
                              .read_text(encoding="utf-8")))
        pts = [p for p in pts if p["id"] in keep]
    if getattr(args, "ids_file", ""):
        # 指定题号清单（例：r9m 第一波标完后题面又改了，剩余 + 变了的题导到新目录）
        keep = set(json.loads((PROJECT_DIR / args.ids_file).read_text(encoding="utf-8")))
        pts = [p for p in pts if p["id"] in keep]
    if args.split:
        # 按划分导出（例：用新规范重标固定测试集，让每一轮都在同一版老师标签上评估）
        pts = [p for p in pts if p.get("split") == args.split]
    rng = random.Random(args.seed)
    quota = {"card_pick": 10 ** 6, "lord_boost": 10 ** 6, "talent": 10 ** 6, "carry": 10 ** 6, "swap": 10 ** 6, "use_card": 10 ** 6, "auction": 10 ** 6, "route": 10 ** 6, "eco": 10 ** 6, "sell_filler": 10 ** 6, "use_timing": 10 ** 6, "sell_hand": 10 ** 6, "shop": args.shop}
    chosen = []
    for kind in ("card_pick", "lord_boost", "talent", "carry", "swap", "use_card", "auction", "route", "eco", "sell_filler", "use_timing", "sell_hand", "shop"):
        rows = [p for p in pts if p["kind"] == kind]
        rng.shuffle(rows)
        chosen += rows[:quota[kind]]
    rng.shuffle(chosen)
    root = PROJECT_DIR / "data" / "laya" / "labels" / (args.out_ver or args.ver)
    (root / "batches").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)

    def write(prefix, rows):
        n = 0
        for b in range(0, len(rows), args.batch):
            part = rows[b:b + args.batch]
            name = f"{prefix}_{b // args.batch:03d}"
            md = [f"# 标注批次 {name}（{len(part)} 题，规范 {GUIDE_VERSION}）\n"] + [render_point(p) + "\n" for p in part]
            (root / "batches" / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
            (root / "batches" / f"{name}.ids.json").write_text(json.dumps(
                {p["id"]: p["options"] for p in part}, ensure_ascii=False), encoding="utf-8")
            hs = {p["id"]: p["hindsight"] for p in part if p.get("hindsight")}
            if hs:
                # g9q：eco 题的事后信息（最终名次、后两回合等级、缺件拿到没有）**不进标注视图**（信息对齐，第 15 节），
                # 单独存一份，标完后核对用（tools/laya/label_batches.py 之外的分析脚本读它）。
                (root / "batches" / f"{name}.hindsight.json").write_text(json.dumps(hs, ensure_ascii=False, indent=1),
                                                                         encoding="utf-8")
            n += 1
        return n
    nb = write("batch", chosen)
    dup = rng.sample(chosen, max(1, int(len(chosen) * args.dup))) if args.dup > 0 else []
    nd = write("dup", dup) if dup else 0
    manifest = {"ver": args.ver, "guide": GUIDE_VERSION, "seed": args.seed, "n": len(chosen), "batches": nb,
                "dup_n": len(dup), "dup_batches": nd,
                "by_kind": dict(collections.Counter(p["kind"] for p in chosen))}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def _validate(rec: dict, opts: list[str]) -> str | None:
    d = rec.get("dist")
    if not isinstance(d, dict) or set(d) != set(opts):
        return f"dist 键不对：{sorted(d or {})} vs {opts}"
    s = sum(float(v) for v in d.values())
    if not 0.98 <= s <= 1.02:
        return f"概率和 {s:.3f}"
    if rec.get("best") not in opts:
        return "best 不在选项里"
    return None


def cmd_merge(args) -> int:
    root = PROJECT_DIR / "data" / "laya" / "labels" / args.ver
    ids = {}
    for f in glob.glob(str(root / "batches" / "*.ids.json")):
        name = Path(f).name.replace(".ids.json", "")
        ids[name] = json.loads(Path(f).read_text(encoding="utf-8"))
    good = {"batch": {}, "dup": {}}
    bad = []
    for f in sorted(glob.glob(str(root / "out" / "*.jsonl"))):
        name = Path(f).stem
        kind = "dup" if name.startswith("dup") else "batch"
        want = ids.get(name, {})
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad.append((name, "JSON 解析失败", line[:60]))
                continue
            opts = want.get(rec.get("id"))
            if opts is None:
                bad.append((name, "id 不在批次里", rec.get("id")))
                continue
            err = _validate(rec, opts)
            if err:
                bad.append((name, err, rec.get("id")))
                continue
            s = sum(float(v) for v in rec["dist"].values())
            rec["dist"] = {k: float(v) / s for k, v in rec["dist"].items()}
            rec["guide"] = args.guide
            rec["batch"] = name
            good[kind][rec["id"]] = rec
    for kind, fname in (("batch", "labels.jsonl"), ("dup", "labels_dup.jsonl")):
        with (root / fname).open("w", encoding="utf-8") as fh:
            for r in good[kind].values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"合并：主标注 {len(good['batch'])}，重复标注 {len(good['dup'])}，不合格 {len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    return 0


def cmd_status(args) -> int:
    root = PROJECT_DIR / "data" / "laya" / "labels" / args.ver
    batches = sorted(Path(f).stem for f in glob.glob(str(root / "batches" / "*.md")))
    done = sorted(Path(f).stem for f in glob.glob(str(root / "out" / "*.jsonl")))
    todo = [b for b in batches if b not in done]
    print(f"批次 {len(batches)}，已完成 {len(done)}，待标 {len(todo)}：{todo[:40]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ver", default="v2")
    e.add_argument("--shop", type=int, default=800)
    e.add_argument("--batch", type=int, default=40)
    e.add_argument("--dup", type=float, default=0.1)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--only-new", action="store_true", help="只导出 new_ids.json 里的新题（迭代轮次）")
    e.add_argument("--split", default="", help="只导出这个划分的题（train/val/test）")
    e.add_argument("--ids-file", default="", help="只导出这个 json 清单里的题号（相对项目根目录）")
    e.add_argument("--out-ver", default="", help="标签目录名（默认同 --ver）；重标测试集时用独立目录，不和训练标签混在一起")
    m = sub.add_parser("merge")
    m.add_argument("--ver", default="v2")
    m.add_argument("--guide", default=GUIDE_VERSION, help="这批标注用的规范版本（v2 标注是 g1）")
    s = sub.add_parser("status")
    s.add_argument("--ver", default="v2")
    args = ap.parse_args()
    return {"export": cmd_export, "merge": cmd_merge, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
