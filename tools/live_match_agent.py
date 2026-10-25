"""实时对局干跑（dry-run）：连接 MuMu ADB，每 2 秒感知一帧并给出规则引擎决策，
但默认不执行任何点击（--execute 才真实操作）。用于真局验证感知/决策链路。

用法:
  python -X utf8 tools/live_match_agent.py            # 干跑观察
  python -X utf8 tools/live_match_agent.py --execute  # 真实执行
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools"))

from test_perception_offline import FrameContext  # noqa: E402

from agent.adb_client import connect_adb_controller  # noqa: E402
from agent.actions.executor import ActionExecutor, board_drop as board_drop_point  # noqa: E402
from agent.actions import tap_gate  # noqa: E402  点击 gate + 点击日志（SAFE_TAPS 表在 agent/actions/tap_gate.py）
from agent.decision.rules import RuleEngine  # noqa: E402
from agent.perception.extractors import (PERF, ROI_DESIGNATE, StateExtractor,  # noqa: E402
                                          _hit, _ocr_boxes, cell_key,
                                          detect_equip_counts, equip_popup_present,
                                          find_equip_slots, find_equip_thumb,
                                          pick_equip_index)
from agent.win32_client import create_tasker, screencap  # noqa: E402

# 装备四选一评分由 rules.score_equipment_option 提供（按数值词条×目标定位权重）


def _equip_count_at(controller, center) -> int:
    """那个格子上的英雄身上现在挂着几件装备（像素观测，不是记账）。"""
    try:
        return detect_equip_counts(screencap(controller)).get(cell_key(center), 0)
    except Exception:
        logger.exception("读装备件数出错")
        return -1


def _equip_forged(ctx, controller, extractor, center, base_equip, before: int,
                  wait: float = 1.2) -> bool:
    """这一下到底把装备铸造上去了没有。

    判据来回改过三版，把理由留在这里免得再绕：
      v1「我点了」—— 只证明发出过一次点击。2026-09-23 画面上
         装备四选一 3 个英雄都没有成功，而同一局日志里三条全是 ✅。
      v2「英雄身上装备件数变多了」—— 听起来最硬，**但观测覆盖不全**，
         按它判失败会多做一次棋盘上的拖拽，等于把棋子拖到别处。
      v3（现在）**以"弹窗还在不在"为主**，件数只写日志作加强证据。
    2026-09-23 起"弹窗在不在"换成 `equip_popup_present`（数环绕目标的彩色亮圆）。
    旧判据数的是"随便什么圆"，把等级徽章也算进去，**弹窗压根没出现时也会说它在/不在**，
    这一版 v3 实际上一直是在拿噪声做判断。
    """
    time.sleep(wait)
    try:
        shot = screencap(controller)
        still = equip_popup_present(shot, center)
        after = detect_equip_counts(shot).get(cell_key(center), -1)
    except Exception:
        logger.exception("装备四选一复查出错，当作没生效")
        return False
    logger.info("装备四选一复查：弹窗%s，%s 那一格装备件数 %s -> %s",
                "还在" if still else "没了", center, before, after)
    return not still


_LAST_FORGED = ""        # detect_equip_popup 最后一次选中的成品名



def _util_ctrl_names(extractor) -> set:
    """四选一成品里带控制效果的（卡面读：眩晕 / 冻结 / 击飞…），给 util 角色用。"""
    global _UTIL_CTRL
    if _UTIL_CTRL is None:
        from agent.perception.extractors import UTIL_CTRL_RE
        cards = json.loads((PROJECT_DIR / "data" / "cards.json").read_text(encoding="utf-8")).get("cards", {})
        _UTIL_CTRL = {n for ps in (extractor.equip_recipes or {}).values() for n in ps
                      if UTIL_CTRL_RE.search((cards.get(n) or {}).get("skill", "") or "")}
    return _UTIL_CTRL


_UTIL_CTRL = None

FORGE_AVOID = ("虚无法杖",)      # 合成四选一不选的低可用性成品（2026-10-18；临时规则 forge_avoid_low_use）


_CC_NEED = False      # 主C 缺免控（调用方每次设置；10-20 第 836 局起）


def detect_equip_popup(ctx, image, extractor: StateExtractor,
                       center: tuple[int, int] | None = None,
                       role: str = "dps", run_dir=None,
                       base_equip: str | None = None, worn=()):
    """装备四选一：弹窗在就返回要点的那个点，不在返回 None。**不读任何文字。**

    2026-09-23 起的做法：弹窗里 4 个选项的次序是固定的、和爬虫的 recipes 次序一致，
    所以只要①认出"弹窗在"②知道这件基础装备的 recipes ③按偏好挑一个下标，
    就能直接点对应位置。旧实现要 OCR 效果概括再去匹配卡库，一直 3/4、4/4 地摇摆，
    而且"选哪个"的打分对大多数装备都是 0 分 —— 其实从来没在比好坏。
    """
    global _LAST_FORGED
    _LAST_FORGED = ""                     # 选了哪件成品（不知道就是空）—— 给"同名装备只能穿一件"记账
    if center is None:
        return None                       # 按位置选必须知道目标英雄在哪一格
    slots = find_equip_slots(image, center)
    if slots[0] is None:
        return None
    pick = pick_equip_index(extractor.image_dir.parent.parent / "data",
                            base_equip or "", role)
    thumb = find_equip_thumb(image, slots)
    if pick is None:
        # **不知道拖上去的是哪件基础装备，弹窗照样要选**（2026-09-23 第七局：给悟空穿装备
        # 4选1 又失败了）：决策以为拖的是成品 红莲炎甲，实际手牌已经挪位、抓起来的是
        # 力量腰带 —— 弹窗出来了，我们却"不认识就不点"，弹窗过期、装备白扔。
        # 选项次序不知道，但**游戏的 👍 推荐不依赖我们知道**；没有 👍 就拿第一个，
        # 总比让弹窗过期强（过期 = 什么都没穿上）。
        idx = thumb if thumb is not None else 0
        logger.warning("装备四选一：弹窗在，但不知道是哪件基础装备（决策以为是 %r）—— "
                       "按%s选第 %d 个", base_equip, "游戏推荐👍" if thumb is not None else "位置",
                       idx)
        point = slots[idx]
        if run_dir is not None:
            import cv2
            _imwrite(run_dir / f"equip_popup_unknown_{idx}_{int(time.time())}.png",
                     cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return point
    idx, name = pick
    why = f"偏好表({role})"
    from agent.decision import hero_traits as _htr
    _hip = _htr.HERO_ITEM_PREF if _htr.USE_HERO_ITEM_PREF else {}
    _thumb_name = (extractor.equip_recipes.get(base_equip or "", []) + [None] * 4)[thumb] if thumb is not None else None
    _tgt_hero = (role.split(":", 1)[1] if ":" in role else "").split("|")[0]
    _util_ctrl = role.startswith("util:") and name in _util_ctrl_names(extractor)     # 挑中的是控制装：不跟 👍（非主C 要辅助性质的）
    _thumb_bad = ((_htr.USE_HERO_ITEM_PREF and _htr.LIVE_ITEM_LIMITS.get(_thumb_name) == "assassin"
                   and not _htr.is_assassin(_tgt_hero))                      # 👍 推荐红魔、穿的人不是刺客 → 不跟
                  or (bool(_thumb_name) and _htr.profile_says(_thumb_name, _tgt_hero) == "avoid"))   # 装备分类说不要给
    from agent.decision import interim_rules as _IRf
    _avoid = set(FORGE_AVOID) if _IRf.on("forge_avoid_low_use") else set()
    if _tgt_hero not in ("花木兰", "嬴政"):
        _avoid.add("机器头")           # 2026-10-19：机器头锁攻速（法力靠普攻回），只有花木兰 / 嬴政当主C 才选（👍 / 偏好表都避开）
    if _thumb_name in _avoid:
        _thumb_bad = True              # 2026-10-18：虚无法杖可用性太低，合成别选（第 751 局 R1 跟 👍 给临时主C 干将选了它，后来只能卖）
    if not _thumb_bad and _thumb_name and _tgt_hero:
        # 新关系表（2026-10-01，按所有装备的描述整理的关系图）：👍 那件比这件基础装备对这个人最好的成品低 2 分以上 → 不跟
        _rb = _htr.rel_fit(base_equip or "", _tgt_hero) or {}
        _rt = _htr.rel_fit(_thumb_name, _tgt_hero) or {}
        if _rb.get("score") is not None and _rt.get("score") is not None and _rt["score"] < _rb["score"] - 2:
            _thumb_bad = True
            logger.info("装备四选一：👍 推荐「%s」给 %s 只有 %.1f 分（最佳成品「%s」%.1f）—— 不跟 👍",
                        _thumb_name, _tgt_hero, _rt["score"], _rb.get("best_forge"), _rb["score"])
    if thumb is not None and not _thumb_bad and not _util_ctrl and not (role.startswith("share:") and name == "破茧之衣")             and not (role.startswith("carry:") and name in _hip.get(role[6:], ())):
        recipes = extractor.equip_recipes.get(base_equip or "", [])
        # 👍 是游戏自己的推荐（瑶的破茧之衣除外：分享给 3 名友军的价值游戏推荐不知道）：它知道英雄是物理还是法术，我们的偏好表不知道
        idx = thumb
        name = recipes[idx] if idx < len(recipes) else "?"
        why = "游戏推荐👍"
    worn = set(worn or ())
    if name in worn:
        # **同一个英雄不能带两件同名装备**（2026-10-03：4 选 1 时避开这个英雄身上已经有的装备）：
        # 选中的成品他身上已经有了 → 按同样的偏好顺序换下一件他还没有的
        recipes = extractor.equip_recipes.get(base_equip or "", [])
        if role.startswith("carry:"):
            from agent.decision.hero_traits import carry_order
            order = list(carry_order(recipes, role[6:]))
        else:
            import json as _json
            _pref = _json.loads((extractor.image_dir.parent.parent / "data" / "equip_recipes.json")
                                .read_text(encoding="utf-8")).get("prefer", {}).get(base_equip or "", {})
            # 角色名是 util:苏烈|tank 这种，偏好表的键是 tank / dps（第 853 局 R12：查不到落到 dps，给苏烈选了贤者之书）
            order = list(_pref.get(role) or _pref.get(role.split("|")[-1]) or _pref.get("tank" if "|tank" in role else "dps") or [])
        order += [r for r in recipes if r not in order]
        alt = next((r for r in order if r in recipes and r not in worn), None)
        if alt is not None:
            logger.info("装备四选一：%s 身上已经有「%s」（同名只能穿一件）→ 改选「%s」", role, name, alt)
            idx, name, why = recipes.index(alt), alt, why + "；避开已穿的同名"
    _LSI = _htr.carry_lifesteal_items(_tgt_hero) if _tgt_hero else _htr.lifesteal_items()      # 按主C 算（噬神之书不给物理主C 算吸血，10-20 晚）
    _CCI = _htr.carry_cc_items(_tgt_hero) if _tgt_hero else _htr.CC_IMMUNE_ITEMS                 # 星源装甲只对日落海算免控
    if _tgt_hero and _htr.attack_speed_carry_live(_tgt_hero) and not (set(worn or ()) & _LSI) and name not in _LSI:
        # 2026-10-19：高攻速英雄最好配一件带吸血的装备：身上还没吸血装 → 四个成品里有对口的吸血装就选它
        _rcp = extractor.equip_recipes.get(base_equip or "", [])
        _ls = [r for r in _rcp if r in _LSI and r not in _avoid and r not in set(worn or ())
               and _htr.profile_says(r, _tgt_hero) != "avoid" and not _htr.item_mismatch(r, [_tgt_hero])]
        if _ls:
            logger.info("装备四选一：%s 高攻速、身上没有吸血装 → 选「%s」（原选「%s」）", _tgt_hero, _ls[0], name)
            idx, name, why = _rcp.index(_ls[0]), _ls[0], why + "；高攻速配吸血"
    if _CC_NEED and name not in _CCI:
        # 2026-10-20 第 836 局：守约没穿免控装备、也没有枫叶舞天赋，一直被控、开不出技能：
        # 主C 身上没有免控装、也没拿到枫影无拘 → 四个成品里有免控装（炽热支配者 / 星源装甲）就选它（吸血优先的攻速主C 已在上面处理）
        _rcc = extractor.equip_recipes.get(base_equip or "", [])
        _cc = [r for r in _rcc if r in _CCI and r not in set(worn or ())]
        if _cc and not (_htr.attack_speed_carry_live(_tgt_hero) and name in _LSI):
            logger.info("装备四选一：主C %s 没有免控（装备 / 枫影无拘都没有）→ 选「%s」（原选「%s」）", _tgt_hero, _cc[0], name)
            idx, name, why = _rcc.index(_cc[0]), _cc[0], why + "；主C 缺免控"
    if role.startswith("carry:") and (base_equip or "") in RuleEngine.CARRY_LATE_DEF_BASE:
        _rcl = extractor.equip_recipes.get(base_equip or "", [])
        _want = RuleEngine.CARRY_LATE_DEF_BASE[base_equip]
        if _want in _rcl and _want not in set(worn or ()):
            # carry_late_defense：后期力量腰带给主C 只铸霸者重铠（血量 >70% 减伤 40%，不降攻速）
            logger.info("装备四选一：力量腰带给主C %s → 铸「%s」（原选「%s」；临时规则 carry_late_defense）", _tgt_hero, _want, name)
            idx, name, why = _rcl.index(_want), _want, why + "；主C 后期防御装"
    if role.startswith("util:"):
        # 非主C：辅助性优先（控制 > 回血 / 护盾 / 团队 > 防御 > 输出；读卡面原文，10-21）
        _rcu = extractor.equip_recipes.get(base_equip or "", [])
        _mel = bool(_tgt_hero) and _htr.is_melee(_tgt_hero)
        _cu = [r for r in _rcu if r not in set(worn or ()) and r not in _avoid
               and not (r in _htr.RANGED_ONLY and _mel)          # 自然法杖只给远程（10-21 第 884 局）
               and not (_mel and "限定远程" in str((_htr._cards().get(r) or {}).get("skill") or ""))]   # 卡面限定远程（日渊）不给近战（铠是近战，10-22）
        if _cu:
            _bu = min(_cu, key=lambda r: (_htr.utility_rank(r), _rcu.index(r)))
            if name in set(worn or ()) or name not in _cu or _htr.utility_rank(_bu) < _htr.utility_rank(name):
                logger.info("装备四选一：%s 不是主C → 选辅助性的「%s」（原选「%s」）", _tgt_hero, _bu, name)
                idx, name, why = _rcu.index(_bu), _bu, why + "；非主C 选辅助性"
    if name in _avoid:
        recipes = extractor.equip_recipes.get(base_equip or "", [])
        alt = next((r for r in recipes if r not in _avoid and r not in worn), None)
        if alt is not None:
            logger.info("装备四选一：「%s」可用性太低（临时规则 forge_avoid_low_use）→ 改选「%s」", name, alt)
            idx, name, why = recipes.index(alt), alt, why + "；避开低可用性成品"
    if slots[idx][0] < EQUIP_OPT_MIN_X and any(sl and sl[0] >= EQUIP_OPT_MIN_X for sl in slots):
        # **最左边那块点不到**（2026-09-24 第二轮第 3 局证据帧 equip_before_tap_风暴巨剑）：单位站在最左下角时，
        # 四选一的左侧选项落在 x<300 —— 那里常开着英雄详情面板（点击被面板吃掉），不开时站着棋手阿离（点了弹棋手面板）。
        # 单击、拖拽都点不到，弹窗过期、只穿上基础装备。改选点得到的里面离原选项最近的一个。
        recipes = extractor.equip_recipes.get(base_equip or "", [])
        # 点得到的里面**挑对这个人合适的**，不是"离原选项最近"（2026-10-06 第 322 局：首选点不到，改点旁边的金色圣剑 ——
        # 法术装给了物理主C 虞姬）。排序：没穿过同名 > 不是 avoid / 伤害类型不错配 > 适配分高 > 离原选项近。
        def _reach_key(i):
            nm = recipes[i] if i < len(recipes) else ""
            bad = bool(nm) and bool(_tgt_hero) and (_htr.profile_says(nm, _tgt_hero) == "avoid"
                                                     or _htr.item_mismatch(nm, [_tgt_hero]))
            fit = (_htr.item_fit_score(nm, _tgt_hero) or 0) if (nm and _tgt_hero) else 0
            return (nm in worn, bad, -fit, abs(i - idx))
        alt = min((i for i, sl in enumerate(slots) if sl and sl[0] >= EQUIP_OPT_MIN_X), key=_reach_key)
        logger.warning("装备四选一：第 %d 个选项在 x=%d（最左边点不到），改选第 %d 个「%s」",
                       idx, slots[idx][0], alt, recipes[alt] if alt < len(recipes) else "?")
        idx, name = alt, (recipes[alt] if alt < len(recipes) else "?")
    point = slots[idx]
    _LAST_FORGED = name if name and name != "?" else ""
    logger.info("装备四选一：%s 穿给 %s 位 -> 选第 %d 个「%s」（依据：%s），点 %s（从左到右四个点 %s）",
                base_equip, role, idx, name, why, point, slots)
    if run_dir is not None:
        # **认出弹窗的这一帧**才存（以前存的是第一次探测那帧，那时弹窗多半还没出来）。
        # 每张都是一个新样本：拿图标旁的效果文字对一下 recipes，就能核对这件基础装备的次序
        # —— 目前只在力量腰带上核对过。
        import cv2
        _imwrite(run_dir / f"equip_popup_{base_equip}_{idx}_{int(time.time())}.png",
                 cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return point


# ==================== 点开手牌看放大详情（认觉醒卡） ====================
# 2026-09-23：记牌问题很大，改为逐张点开牌看是什么；
#                  金色的卡是觉醒卡，要优先上场。
# 手牌条上的 AWAKENED 只有 8~9px 高，OCR 读不出来；卡面色调还会在金色/冰蓝之间变。
# 点开之后是一张大卡：顶部（觉醒卡印着 AWAKENED）、中间卡名、右侧「说明 / 后排·法师」。
# 坐标取自**实战帧** data/logs/fixtures/inspect_big_card_normal.png（2026-09-23 第一次点开，孙膑）：
#   大卡 x520~762 / y110~458，卡名 (627,311)，「说明」(828,127)，「后排 · 法师」(832,181)。
# ⚠ 第一版是按一张外部截图换算的，那张截图不是 16:9（1367x875），换算全偏了 ——
#   检测区从 x=840 起、把「说明」切掉了，于是明明打开了却判"没打开"。
ROI_BIG_CARD_TOP = (560, 110, 205, 45)    # 大卡顶部（✦ 图标右边）—— AWAKENED 那行
ROI_BIG_CARD_NAME = (525, 290, 235, 45)   # 大卡中间的卡名（孙膑 在 (627,311)）
HAND_CENTER_X = 645                       # 手牌整排的中心（实测 638~659）
ROI_BIG_CARD_SIDE = (770, 100, 340, 110)  # 「说明」「后排·法师」—— 放大图在不在的判据
# 关放大图点哪：**不能用 TAP_BLANK(170,100)** —— 准备阶段那里是左侧**玩家列表**第一行，
# 点下去可能打开别人的阵容。(400,60) 是场馆背景（绿色球），画面上没有任何可点的东西。
TAP_CLOSE_BIG_CARD = (400, 60)
AWAKEN_RE = re.compile(r"AWAK|WAKE|AKEN|觉醒", re.I)
INSPECT_PER_ROUND = 6                     # 每回合最多点开几张
INSPECT_PER_FRAME = 4                     # 每帧最多连着看几张
_INSPECT = {"flags": {}, "round": -1, "count": 0, "gaps": set(), "alias": {}}
# 提前点「准备 / 开战」（2026-10-15：本回合操作做完就可以点准备或开战；对面准备好后准备按钮会变成开战）：
# 准备阶段约 76 秒，原来每回合都等倒计时走完。现在商店已经用过（或没钱）、连续 READY_IDLE 帧没有任何动作 → 点一次，每回合一次
_READY = {"round": -1, "idle": 0, "done": False, "t0": 0.0}
READY_MIN_S = 20.0          # 准备阶段开始至少 20 秒后才考虑（防止误判导致什么都不做就进了战斗）
READY_IDLE = 12
THUMB_SAME = 7.0      # 露出那一截缩到 14x28 后的平均像素差（3000 张真实裁图核对：<7 同牌命中 ~83%、异牌误中 0；<9 误中 0.2%）


def _big_card_open(ctx, image, extractor=None) -> bool:
    """放大详情在不在。**英雄牌**右侧有「说明 / 后排·法师」一栏；**效果牌/装备牌没有这一栏**
    （2026-09-23 实战：点开「1阶效果锦囊」明明打开了，却因为没有右侧栏被判成"没打开"）——
    所以再看大卡中间能不能读出一个卡库里有的卡名。"""
    texts = [t for t, _b in _ocr_boxes(ctx, image, ROI_BIG_CARD_SIDE)]
    if any("说明" in t or "后排" in t or "前排" in t for t in texts):
        return True
    if extractor is None:
        return False
    return _big_card_name(ctx, image, extractor) is not None


def _big_card_name(ctx, image, extractor):
    """大卡中间的卡名。**要求很严**：必须是卡库里真有的卡名、且原文长度和它差不超过 1
    —— 放大图没开时这块区域常是商店卡面的描述文字，`resolve_name` 会把
    「你的英雄每触发2次」原样返回、把「登场」补全成「登场人物」（2026-09-23 测试抓到的误判）。
    容许的是「曜业」-> 曜 这种多读一个字的情况。"""
    for t, _b in _ocr_boxes(ctx, image, ROI_BIG_CARD_NAME):
        t = (t or "").strip()
        r = extractor.resolve_name(t) if t else None
        if r and r in extractor._card_cats and abs(len(t) - len(r)) <= 1:
            return r
    return None


def _open_big_card(ctx, controller, executor, x, y, extractor=None):
    """点一张手牌，等放大详情出来（每 0.15s 看一次，最多 ~1s）。出来了返回那一帧，没出来返回 None。"""
    executor._tap(x, y)
    for _ in range(7):
        time.sleep(0.15)
        shot = screencap(controller)
        if _big_card_open(ctx, shot, extractor):
            return shot
    return None


def _close_big_card(ctx, controller, executor, extractor=None) -> bool:
    """关放大详情并**确认关掉了**（每 0.15s 看一次）。关不掉返回 False。"""
    for _attempt in (1, 2):
        # 放大图是 OCR（_big_card_open）确认过的 → gate 只影子记录
        _fixed_tap(executor, "TAP_CLOSE_BIG_CARD", *TAP_CLOSE_BIG_CARD, verified=True, why="看牌关放大图")
        for _ in range(5):
            time.sleep(0.15)
            if not _big_card_open(ctx, screencap(controller), extractor):
                return True
    return False


def _hand_gaps(slots, wide_x: int | None = None) -> list[int]:
    """按"手牌等间距、整排居中在 x≈640"推断**漏读的牌**在哪儿（返回推断的 x）。

    实测（2026-09-23 多局）：手牌中心稳定在 638~659；读漏一张时相邻两张的间距正好是
    正常牌距的两倍（例：684 -> 928，牌距 ~122，中间那张弈星没读出来）。
    """
    xs = sorted(sl["x"] for sl in slots)
    if len(xs) < 2:
        return []
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    small = min(gaps)
    pitch = sorted(g for g in gaps if g <= small * 1.4)[0] if gaps else 0
    if pitch < 70:
        return []
    out = []
    for a, g in zip(xs, gaps):
        if wide_x is not None and (a == wide_x or a + g == wide_x):
            # 枫叶舞（棋手技能牌）那张卡比普通手牌宽，它两边的间距本来就大 ——
            # 实测 267 -> 410 = 143，而牌距才 86，被推断成"中间漏了一张"，白点一次空位
            continue
        k = round(g / pitch) - 1
        if g > pitch * 1.6 and k >= 1:
            out += [int(a + pitch * j) for j in range(1, k + 1)]
    center = (xs[0] + xs[-1]) / 2
    if center < HAND_CENTER_X - pitch * 0.35 and xs[-1] + pitch < 1270:
        out.append(int(xs[-1] + pitch))            # 右端少了一张
    elif center > HAND_CENTER_X + pitch * 0.35 and xs[0] - pitch > 210:
        out.append(int(xs[0] - pitch))             # 左端少了一张
    return out


SETTLE_ROW_Y = (104, 186, 268, 352, 435, 518)   # 结算页 1~6 名那六行的中线（1280x720）


def read_my_rank(image) -> int:
    """结算页上**我们的名次**：自己的名字是绿色的（其他人是白色）。认不出返回 0。

    第十一局结算页实测：绿色像素占比 自己那行 0.119、其他行 0.000。名次写进 decisions.jsonl，
    是赛后评判和按名次回传（TD）的锚点。
    """
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2HSV)
    best, best_r = 0.0, 0
    for i, y in enumerate(SETTLE_ROW_Y):
        band = hsv[y - 12:y + 12, 185:320]
        g = float(((band[..., 0] >= 35) & (band[..., 0] <= 85) & (band[..., 1] >= 80)
                   & (band[..., 2] >= 120)).mean())
        if g > best:
            best, best_r = g, i + 1
    return best_r if best >= 0.03 else 0


def inspect_hand(ctx, controller, executor, extractor, rules, state, image=None) -> None:
    """**点开手牌看放大详情**，把 OCR 在手牌条上读不准的牌认出来（2026-09-23：
    牌太多时文本识别不了，应逐张点开看，但查看速度要快）。

    点开看的三种情况：
      ① 卡名读不全 / 卡库里查不到（「翩跹舞步」后缀被切掉 -> 类别"未知"，整局没用上）
      ② 按"等间距、居中"推断出**漏读**的位置（挨着的同名牌、压在亮底上的卡名）
      ③ 分不清是合成材料还是觉醒卡（两张同名英雄 / 场上记账里也有）
    要快：点开后每 0.15s 看一次、出来就读；一帧里连着看完几张（点牌不会动手牌），
    不必每看一张就重新感知一遍。每回合最多 INSPECT_PER_ROUND 张，每帧最多 INSPECT_PER_FRAME 张。
    结果按 (回合, 卡名, 同名张数) 缓存 —— 手牌挪位置不会让同一张牌被反复点
    （第一版按位置缓存，同一张莱西奥 15 秒内被点了 3 次）。
    """
    slots = state.hand_slots or []
    if state.round_no < _INSPECT["round"] and state.round_no <= 1:
        _INSPECT["thumbs"] = []                         # 新的一局：按图记住的牌清掉
    if state.round_no != _INSPECT["round"]:
        _INSPECT.update({"round": state.round_no, "count": 0, "flags": {}, "gaps": set(), "at": {},
                         "sus_done": set(), "tint_at": {}, "tint_gap": set(), "gap_fail": {}, "gap_at": {}})
    # **点开看过一次的误读，记住它对应的真名**（「兼颜」= 廉颇）。不记的话，
    # 下一帧手牌条上还是读成「兼颜」，对不上缓存又点一次 —— 实测廉颇 25 秒内被点开 3 次。
    for sl in slots:
        real = _INSPECT["alias"].get(sl["name"])
        if real:
            sl["name"] = real
    names = [sl["name"] for sl in slots]
    state.hand = list(names)

    def key(name):
        return (name, names.count(name))

    def is_hero(n):
        return extractor._card_cats.get(n) == "英雄牌"

    # 已经看出是觉醒卡的：标在同名里最右那张上（state 每帧都是新的，所以每帧都要标）
    for name in set(names):
        if _INSPECT["flags"].get(key(name)) is True:
            max((sl for sl in slots if sl["name"] == name), key=lambda s: s["x"])["awakened"] = True

    todo = []                                           # (原因, x, y, 对应的 slot 或 None)
    for sl in slots:
        if sl["name"] not in extractor._card_cats and key(sl["name"]) not in _INSPECT["flags"]:
            todo.append(("读不全", sl["x"], sl["y"], sl))
    wide = next((sl["x"] for sl in slots if extractor._card_cats.get(sl["name"]) == "棋手技能牌"), None)
    # 按图标布局读的手牌：张数和位置是确定的，认不出的那几张**就在 hand_unread 里**，直接点它们；
    # 旧路径（整条 OCR）才需要按"等间距"去推断哪里漏了一张。
    if state.hand_by_layout:
        gap_xy = [(u[0], u[1], u[2] if len(u) > 2 else "") for u in (state.hand_unread or [])]
        icon_of = {u[0]: u[3] for u in (state.hand_unread or []) if len(u) > 3}
    else:
        gap_xy = [(gx, slots[0]["y"] if slots else 663, "") for gx in _hand_gaps(slots, wide)]
        icon_of = {}
    # 点开看过的位置：**手牌张数没变**就直接沿用上次看到的（露出来的字认不出，但牌没动）
    n_cards = len(slots) + len(state.hand_unread or [])
    known_at = _INSPECT.setdefault("at", {})
    unread_txt: dict = {}
    # **按图记住看过的牌**（2026-10-11 第 573 局：芈月点开认出 22 次 —— 露出来的字读不出（「芈」是生僻字），
    # 而"沿用"要求露出的字一致且不为空，于是每帧都当新牌重点。现象是点开看了很多次、都变大图了还是没有识别）。
    # 露出的那一截（图标 + 立绘）对同一张牌是固定的：点开看过就把它的小图记下，之后按图比对，位置挪了也认得。
    thumbs = _INSPECT.setdefault("thumbs", [])

    def _thumb(ix):
        if image is None or ix is None:
            return None
        x0 = int(ix) - 8
        if x0 < 0 or x0 + 56 > image.shape[1]:
            return None
        import cv2 as _cv2
        return _cv2.resize(image[599:711, x0:x0 + 56], (14, 28), interpolation=_cv2.INTER_AREA).astype("int16")

    def _thumb_name(th):
        if th is None:
            return None
        best = min(((float(abs(th - t).mean()), n) for t, n in thumbs), default=None)
        return best[1] if best and best[0] < THUMB_SAME else None

    for gx, gy, gtxt in gap_xy:
        seen = _thumb_name(_thumb(icon_of.get(gx)))
        if seen:
            slots.append({"name": seen, "x": gx, "y": gy, "name_y": gy + 25})
            state.hand_unread = [u for u in (state.hand_unread or []) if u[0] != gx]
            logger.info("手牌 x=%d 按图认出：%s（之前点开看过）", gx, seen)
            continue
        hit = known_at.get(gx // 20)
        # 沿用的条件：张数没变 **且** 那一截露出来的字和当时一样（且不是空的）。
        # 只看张数不够：2026-09-23 锦囊用掉、又买进一张曜，张数还是 2，
        # 同一位置被沿用成"1阶效果锦囊"，拖之前校验读到「曜」，反复失败。
        if state.hand_by_layout and hit and hit[1] == n_cards and gtxt and hit[2] == gtxt:
            slots.append({"name": hit[0], "x": gx, "y": gy, "name_y": gy + 25})
            continue
        if gx // 20 not in _INSPECT["gaps"] and time.time() - _INSPECT.setdefault("gap_at", {}).get(gx // 20, 0) >= 0.8:
            todo.append(("漏读", gx, gy, None))
            unread_txt[gx] = gtxt
    slots.sort(key=lambda s: s["x"])
    # **颜色像觉醒卡的，最先看**（2026-09-23：先确认颜色，再打开牌确认是哪个英雄觉醒了）。
    # 原来只在"两张同名 / 场上也有"时才去看 —— 觉醒卡回手时往往没有同名，要等别的判据兜底，
    # 第八局孙膑从觉醒到重新上场花了 9 秒。颜色判据纯像素、每帧都跑，**点开确认**才算数。
    if image is not None:
        from agent.perception.extractors import awaken_tint_spans
        tinted = []
        for x0, x1, ice, dark in awaken_tint_spans(image):
            sl = next((s for s in slots if x0 <= s["x"] <= x1), None)
            already = sl is not None and sl["name"] in rules._awakened and sl["name"] not in rules._field
            # **颜色证据优先于"看过、不是觉醒"的缓存**（2026-09-24 match13）：缓存按 (卡名, 同名张数) 记，
            # 买来的普通西施看过一次记成"不是"，之后场上西施觉醒回手 —— 手里还是 1 张西施、键不变，
            # 连颜色像觉醒卡也被跳过，16 秒后才认出来，空位早被明世隐占了。同一张每 5 秒最多重看一次。
            # 但**同一 (卡名, 张数) 被颜色触发、看过 2 次都不是觉醒 → 不再按颜色重看**：立绘本身偏金/蓝的卡
            # （2026-09-25 第四轮第 3 局：5 阶武则天 2 分钟里被点开 9 次，每次 1 秒）。张数一变（新买一张 / 觉醒回手）键就变，照常看。
            tint_false = _INSPECT.setdefault("tint_false", {})
            recheck = (sl is not None and _INSPECT["flags"].get(key(sl["name"])) is False
                       and tint_false.get(key(sl["name"]), 0) < 2
                       and time.time() - _INSPECT.setdefault("tint_at", {}).get(key(sl["name"]), 0) > 5)
            if recheck:
                tint_false[key(sl["name"])] = tint_false.get(key(sl["name"]), 0) + 1
                _INSPECT["tint_at"][key(sl["name"])] = time.time()
                _INSPECT["flags"].pop(key(sl["name"]), None)
            if sl is not None and is_hero(sl["name"]) and key(sl["name"]) not in _INSPECT["flags"]                     and not already and not any(t[3] is sl for t in todo):
                tinted.append(("觉醒色", sl["x"], sl["y"], sl))
                logger.info("手牌 %s 颜色像觉醒卡（冰蓝 %.2f / 深色 %.2f），点开确认", sl["name"], ice, dark)
            elif sl is None:
                # **颜色像觉醒卡、但那个位置手牌条上没读出牌**（2026-09-24 match14：芈月觉醒回手了，又没上场）：
                # 合成凑满觉醒 → 芈月回手，手牌条漏读了这张 → 原来颜色证据在这里被直接丢掉，账本还当她在场上，
                # 第 9 回合少她一个人开战。现在也点开确认（按位置记，同一位置每回合只看一次）。
                cx = (x0 + x1) // 2
                seen_gap = _INSPECT.setdefault("tint_gap", set())
                if cx // 20 not in seen_gap and not any(abs(t[1] - cx) < 30 for t in todo):
                    seen_gap.add(cx // 20)
                    tinted.append(("觉醒色", cx, slots[0]["y"] if slots else 663, None))
                    logger.info("手牌 x=%d 颜色像觉醒卡（冰蓝 %.2f / 深色 %.2f）但没读出是哪张，点开确认", cx, ice, dark)
        todo = tinted + todo
    # **觉醒回手嫌疑**（人口比账本少 + 手里有场上某人的同名牌）：清掉缓存结论，最先看（同名组最右那张 = 最新回手的）
    for n in rules.awaken_suspects(state):
        sl = max((s for s in slots if s["name"] == n), key=lambda s: s["x"], default=None)
        done = _INSPECT.setdefault("sus_done", set())
        if sl is not None and (n, names.count(n)) not in done and not any(t[3] is sl for t in todo):
            done.add((n, names.count(n)))              # 每回合每个 (卡名, 张数) 只查一次
            _INSPECT["flags"].pop(key(n), None)
            todo.insert(0, ("觉醒嫌疑", sl["x"], sl["y"], sl))
            logger.info("人口少了一个、手里有场上的 %s —— 疑似觉醒回手，点开确认", n)
    # **有手牌觉醒模型时，不再按"两张同名 / 场上也有"去点开看是不是觉醒**（2026-09-25：重复打开查看手牌的问题，要验证）：
    # 第四轮第 5 局中段 5 回合看牌 18 次，其中这条"觉醒?"触发 9 次（韩信一回合 3 次）；
    # 而觉醒与否模型已经直接判了（测试集 11/0/0，感知层概率 ≥0.99 直接标）—— 这条只剩在模型不可用时兜底。
    import numpy as _np
    from agent.perception.extractors import _hand_awaken_by_model
    model_on = _hand_awaken_by_model(_np.zeros((720, 1280, 3), _np.uint8), []) is not None
    for sl in reversed(slots):                          # 同名组里最右那张先看
        n = sl["name"]
        if model_on:
            break
        if is_hero(n) and key(n) not in _INSPECT["flags"] and (names.count(n) >= 2 or n in rules._field):
            if not any(t[3] is not None and t[3]["name"] == n for t in todo):
                todo.append(("觉醒?", sl["x"], sl["y"], sl))
    if not todo:
        state.hand = [s["name"] for s in slots]
        return
    budget = min(INSPECT_PER_FRAME, INSPECT_PER_ROUND - _INSPECT["count"])
    if todo[0][0] in ("觉醒色", "觉醒嫌疑"):
        budget = max(budget, 1)          # 觉醒候选不受每回合看牌次数限制（它是最该优先的）
    from agent.perception.extractors import hand_layout

    def _layout_sig():
        lay = hand_layout(screencap(controller))
        return (lay["n"], tuple(int(v) // 10 for v in lay["xs"])) if lay else None

    # **看牌期间手牌可能变**（登场/拍卖/回合奖励随时塞新牌，整排重新居中）。
    # 2026-09-23：点开 3 张的 3 秒里进来一张锦囊，第三次看到的"锦囊"被套到了原来的位置上，
    # 那个位置读到的「曜」又被记成"锦囊的误读"，之后整局的曜都被当成锦囊。
    # 所以每看一张都核对布局没变，变了就丢掉这次结果、下一帧重新来。
    sig0 = _layout_sig() if state.hand_by_layout else None
    for reason, x, y, sl in todo[:max(0, budget)]:
        # **点之前核对那个位置现在还有牌**（2026-10-09：第 473~479 局看牌失败 78/218 次，证据帧里手牌只剩 1 张在 x≈660，
        # 却按几帧前的布局点了 x=792 —— 空点一次 ~1 秒，反应变慢）。当前布局里附近没有牌就跳过，不计次数
        try:
            lay_now = hand_layout(screencap(controller))
        except Exception:                                           # noqa: BLE001
            lay_now = None
        # 手牌横跨 [第一张图标 − 20, 最后一张图标 − 20 + 牌宽]（与 extractors 读手牌的切法一致）；x 不在这个范围 = 那里没牌
        from agent.perception.extractors import HAND_ICON_TO_LEFT, HAND_PITCH_MAX
        if lay_now and lay_now.get("xs") and not (
                min(lay_now["xs"]) - HAND_ICON_TO_LEFT - 10 <= x <= max(lay_now["xs"]) - HAND_ICON_TO_LEFT + HAND_PITCH_MAX + 10):
            logger.info("【看牌】x=%d（%s）现在那里没有牌（手牌已变），跳过", x, reason)
            continue
        if reason == "漏读" and lay_now and lay_now.get("by") != "模型":
            # **漏读的空档只在手牌条模型有把握时点**（2026-10-18：模型没把握时退回图标推算，选中抬起的古币牌面圆币被当成图标、
            # 末尾多排一格（3 张牌给出 xs=[457,599,741,883]），点 x=934 是空地，每次白耗 ~6 秒；718~726 局 17 次）。下一帧模型有把握再看，不计次数
            logger.info("【看牌】x=%d（漏读）这一帧手牌位置是图标推算的（%s），等模型有把握再点", x, lay_now.get("by"))
            continue
        _INSPECT["count"] += 1
        t0 = time.time()
        shot = _open_big_card(ctx, controller, executor, x, (sl or {}).get("name_y", y + 25) - 55,
                              extractor)
        if shot is None:
            logger.warning("【看牌】点了 x=%d（%s）没看到放大详情", x, reason)
            _save_evidence(screencap(controller), "inspect_no_view")
            if sl is not None:
                _INSPECT["flags"][key(sl["name"])] = False
            else:
                # **认不出的牌点开失败，不要一次就放弃**（2026-09-25 第四轮第 4 局：N 选 1 刚拿到的亚瑟 OCR 读成「业热」，
                # 点开时牌还在飞入动画里没弹出详情，这个位置当即被记成"看过"，整回合再没试 —— 人口 3/5 却没上场，获得的亚瑟没上场）。
                # 同一位置每回合最多试 3 次、两次之间至少隔 0.8 秒（等动画走完）。
                gf = _INSPECT.setdefault("gap_fail", {})
                gf[x // 20] = gf.get(x // 20, 0) + 1
                _INSPECT.setdefault("gap_at", {})[x // 20] = time.time()
                if gf[x // 20] >= 3:
                    _INSPECT["gaps"].add(x // 20)
            _close_big_card(ctx, controller, executor, extractor)   # 可能其实开着，关一下无害（点的是场馆背景）
            continue
        texts = [t for t, _b in _ocr_boxes(ctx, shot, ROI_BIG_CARD_NAME)]
        name = _big_card_name(ctx, shot, extractor)
        top = [t for t, _b in _ocr_boxes(ctx, shot, ROI_BIG_CARD_TOP)]
        awake = any(AWAKEN_RE.search(t) for t in top)
        closed = _close_big_card(ctx, controller, executor, extractor)
        logger.info("【看牌】x=%d（%s）：%s%s，用时 %.1fs", x, reason, name or f"没认出{texts}",
                    "（**觉醒卡**）" if awake else "", time.time() - t0)
        if sig0 is not None and closed and _layout_sig() != sig0:
            logger.warning("【看牌】看牌期间手牌变了（张数/位置不同），这次结果作废，下一帧重新看")
            return
        _save_evidence(shot, f"inspect_{'awake' if awake else 'card'}")
        if name and image is not None:
            _save_hand_label(image, x, name, awake, (lay_now or {}).get("xs"), (sl or {}).get("name", ""))
        if name:
            if sl is not None:
                if sl["name"] != name and sl["name"] in extractor._card_cats:
                    # 读到的本身就是一张**完整的真卡名**（「曜」），和点开看到的不一样 ——
                    # 那是位置对错了（手牌在动），不是字认错了。**绝不能记成别名**：
                    # 记了就会把以后每一张真正的曜都当成别的牌（2026-09-23 实战）。
                    logger.warning("【看牌】x=%d 读到的是 %s、点开却是 %s —— 位置对不上，不记别名、这次作废",
                                   x, sl["name"], name)
                    if not closed:
                        _save_evidence(screencap(controller), "inspect_cannot_close")
                        logger.error("【看牌】放大详情关不掉，本局不再点开手牌（证据帧已存）")
                        _INSPECT["count"] = 10 ** 6
                        return
                    continue
                if sl["name"] != name:
                    logger.info("【看牌】手牌条上读成 %r，其实是 %s（记住了，之后不再点开）",
                                sl["name"], name)
                    _INSPECT["alias"][sl["name"]] = name
                sl["name"] = name
            else:
                slots.append({"name": name, "x": x, "y": y, "name_y": y + 25})
                slots.sort(key=lambda s: s["x"])
                _INSPECT["gaps"].add(x // 20)
                known_at[x // 20] = (name, n_cards, unread_txt.get(x, ""))
                th = _thumb(icon_of.get(x))
                if th is not None and not awake:
                    thumbs.append((th, name))
            target = sl if sl is not None else next(s for s in slots if s["x"] == x)
            if awake:
                target["awakened"] = True
            names = [s["name"] for s in slots]
            _INSPECT["flags"][key(name)] = awake
            if reason == "觉醒嫌疑" and not awake:
                rules._awk_cleared.add((rules._round_seen, name))   # 只是合成材料：解除"暂停别的英雄上场"
            # 记进"最近见过"：叠放时下一帧只露出一两个字，靠它按前缀认出来，不用再点第二次
            extractor._hand_seen[name] = time.time()
        elif sl is None:
            _INSPECT["gaps"].add(x // 20)
        if not closed:
            _save_evidence(screencap(controller), "inspect_cannot_close")
            logger.error("【看牌】放大详情关不掉，本局不再点开手牌（证据帧已存）")
            _INSPECT["count"] = 10 ** 6
            return
    state.hand = [s["name"] for s in slots]


# ==================== 点开场上单位，看是谁、站哪格 ====================
# 2026-09-23：场上的英雄也可以点开看是谁；数量对不上时点开确认位置。
# 账本和画面对不上是老问题（"账本 4 个、画面 6 个"），中途重启更是把账本清空 ——
# 之后决策层不知道场上谁是谁：该合成的同名卡当成闲置牌去卖、换人时找不到人。
# 点棋盘上的英雄会弹「英雄详情面板」（fixture hero_panel_blocks_levelup.png）：
# 左上角一张卡，卡名在 (172,213)；左下角「+1 出售 / 剩余数量」—— 用后者认面板在不在。
ROI_UNIT_PANEL_NAME = (68, 188, 218, 52)
BOARD_INSPECT_PER_FRAME = 3
BOARD_INSPECT_PER_ROUND = 8
# 点单位要点在**身上**，不是头顶的等级徽章：徽章是 UI 元素，点它常常不弹详情
# （2026-09-23 那局 60 次看场，"点了没弹出英雄详情"占一半）。身体在徽章下方约 40px。
UNIT_BODY_DY = 40
_BOARD_SEEN = {"round": -1, "cells": {}, "count": 0}


def _hero_panel_up(ctx, image) -> bool:
    from agent.perception.extractors import ROI_HERO_PANEL
    return bool(_hit(ctx, image, ROI_HERO_PANEL, "出售|剩余数量"))


def _tap_read_unit(ctx, controller, executor, extractor, x: int, y: int):
    """点 (x, y) 上的单位，读英雄详情面板上的名字，再把面板关掉。返回 (名字或 None, 面板关掉了没)。"""
    from agent.perception.extractors import TAP_SIDE_GROUND
    executor._tap(x, y)
    shot = None
    for _ in range(7):
        time.sleep(0.15)
        s2 = screencap(controller)
        if _hero_panel_up(ctx, s2):
            shot = s2
            break
    if shot is None:
        # **面板有时弹得慢**（最下面一行的单位，2026-09-26 第四轮第 6 局）：1 秒内没等到就当"看不清"返回，
        # 面板随后才弹出、没人关 —— 挡住了接下来的装备四选一（单击、拖拽都没铸造上去）。再等两次，出来了照样读名字并关掉。
        for _ in range(2):
            time.sleep(0.3)
            s2 = screencap(controller)
            if _hero_panel_up(ctx, s2):
                shot = s2
                logger.info("【看场】详情面板弹得慢（%d,%d），补读并关掉", x, y)
                break
    if shot is None:
        return None, True
    texts = [t for t, _b in _ocr_boxes(ctx, shot, ROI_UNIT_PANEL_NAME)]
    name = next((extractor.resolve_name(t) for t in texts if extractor.resolve_name(t)), None)
    if name:
        # 顺带读「剩余数量 a/b」= 这个英雄在公共牌池里还剩几张（Laya 的刷新命中概率用它，不再按满池近似）
        try:
            pool = parse_pool_left("".join(t for t, _b in _ocr_boxes(ctx, shot, ROI_UNIT_PANEL_POOL)),
                                   extractor.pieces_quality(name) if hasattr(extractor, "pieces_quality") else None, name)
            if pool is not None:
                POOL_LEFT[name] = pool
        except Exception:                                           # noqa: BLE001
            logger.exception("读剩余数量出错（忽略）")
    closed = False
    for _attempt in (1, 2):
        # 英雄详情面板是 OCR（「出售/剩余数量」）确认过的 → gate 只影子记录
        _fixed_tap(executor, "TAP_SIDE_GROUND", *TAP_SIDE_GROUND, image=shot if _attempt == 1 else None,
                            verified=True, why="核对单位后关英雄详情")
        for _ in range(5):
            time.sleep(0.15)
            if not _hero_panel_up(ctx, screencap(controller)):
                closed = True
                break
        if closed:
            break
    _save_evidence(shot, "merge_check")
    return name, closed


SOLO_JOIN_DY = 177          # 「独自战斗」标题中心 (945,330) → 同列「加入」按钮中心 (953,507)


def find_solo_button(ctx, image):
    """组队弹窗里选「独自」→ (文字, x, y)；没有返回 None。

    2026-09-24 第三轮第 4 局：弹窗是三列（挑战者队伍 / 挑战者队伍 / **独自战斗**），每列底下一个按钮
    （加入队伍 / 加入队伍 / **加入**）。「独自战斗」只是**标题**，点它没反应 —— 点了两次、倒计时快到才手动点掉
    （要点「独自战斗」那一列下面的「加入」）。证据帧 data/logs/fixtures/team_choice_solo_join.png。
    现在：认出「独自…」后，在**同一列下方**找「加入」（不能是「加入队伍」）；这一帧没认出按钮就按标题往下
    SOLO_JOIN_DY 点。老式弹窗若「独自…」本身就是按钮（旁边没有「加入」、下方也没内容）同样落在按钮区附近。
    只认短文字（≤6 字），不点长句里的片段（同 unknown 兜底的约束）。"""
    boxes = [(t.strip(), b) for t, b in _ocr_boxes(ctx, image, (150, 120, 980, 560))]
    for t, box in boxes:
        if len(t) <= 6 and t.startswith("独自"):
            cx, cy = box[0] + box[2] // 2, box[1] + box[3] // 2
            for t2, b2 in boxes:
                bx, by = b2[0] + b2[2] // 2, b2[1] + b2[3] // 2
                if t2 == "加入" and abs(bx - cx) <= 140 and by > cy + 60:
                    return f"{t}→加入", bx, by
            if t == "独自战斗":
                return f"{t}→加入(按位置)", cx + 8, cy + SOLO_JOIN_DY
            return t, cx, cy
    return None


_MERGE_SEEN = {"round": -1, "cells": {}}
ROI_UNIT_PANEL_POOL = (60, 530, 200, 34)      # 英雄详情面板左下「剩余数量：a/b」
POOL_LEFT: dict = {}                          # 英雄 -> 公共牌池剩余张数（点开看时读到的最新值），主循环同步给规则引擎
WHEEL_SHOTS = [0]           # 本局已存的装备轮盘证据帧数
_RUN_DIR: list = [None]     # 本局日志目录（给不传 run_dir 的函数存证据帧）
PREP_SEEN: dict = {}        # 回合 -> 第一次看到准备阶段的时刻（推演器回放用，2026-10-14）
LEVEL_ACTED: set = set()
# 准备阶段时长（秒，第 589~627 局实测"第一帧准备 → 第一帧战斗"的中位数，略往下取）。用来估"本回合还剩多少秒"，
# 时间不够时跳过可有可无的点开查看（看场 / 看牌，每次约 1 秒）。2026-10-14：经常在规定时间内操作不完
PREP_LEN = {1: 25, 2: 33, 3: 38, 4: 46, 5: 52, 6: 57, 7: 63, 8: 66, 9: 72}
PREP_LEN_LATE = 78
PREP_SPARE_S = 12           # 剩余少于这么多秒：不再点开看场 / 看牌


def prep_left(round_no: int) -> float:
    t0 = PREP_SEEN.get(round_no)
    if t0 is None:
        return 99.0
    return PREP_LEN.get(round_no, PREP_LEN_LATE) - (time.time() - t0)    # 本局已经做过"会改等级"操作的回合
LEVEL_ACTION_TYPES = {"drag_card", "use_card", "deploy", "sell", "sell_card", "equip", "sequence", "move_unit"}


def parse_pool_left(text: str, quality, name: str = ""):
    """「剩余数量：8/14」→ 8。OCR 常把"/"读成"1"（"4/7"→"417"、"6/7"→"617"）：
    **分母一定等于该英雄所在品阶的牌池张数**（meta.POOL_COPIES：1 阶 14 / 2 阶 11 / 3 阶 9 / 4 阶 7 / 5 阶 6），
    按已知分母从末尾切开还原。读数不合法（剩余 > 总数）返回 None。"""
    from agent.decision.meta import POOL_COPIES
    import re as _re
    t = text.split("：")[-1].split(":")[-1]
    b_known = POOL_COPIES.get(quality) if quality else None
    m = _re.search(r"(\d+)\s*/\s*(\d+)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
    else:
        s = _re.sub(r"\D", "", t)
        if not s or not b_known or not s.endswith(str(b_known)):
            return None
        head = s[:-len(str(b_known))]
        if len(head) >= 2 and head.endswith("1"):
            head = head[:-1]                     # 被读成 1 的斜杠
        if not head:
            return None
        a, b = int(head), b_known
    if b_known and b != b_known:
        return None
    return a if 0 <= a <= b else None


RECHECK_CELLS: set = set()     # 被动作改变过、需要看场复查的格子（"x,y"）；看过一次就移除
BOARD_UNREADABLE: set = set()  # 本局点开看过却认不出名字的格子（换局 / 那一格换了人时清）
BOARD_MISS: dict = {}          # 格子 -> 本局连续"点开没弹出详情"的回合数（≥2 进 BOARD_UNREADABLE；换局 / 那一格换了人时清）


IDENTITY_CHANGE_RE = re.compile(r"变为|变成|移除|消除|进化|觉醒进度|离场|回到手牌")
_CARD_SKILL: dict = {}


def _changes_identity(piece) -> bool:
    """这张牌打在英雄身上，会不会让那一格**换人或空出来**（变身 / 移除 / 进化 / 觉醒进度凑满回手）。
    加等级、护盾、加攻速这类不会 —— 不必点开复查（2026-09-25 第四轮第 3 局：给百里守约用了几张加成牌，
    那一格就被反复点开复查，看场 27 次里 16 次看到的还是同一个人；查看仍然非常频繁，图像识别没能用上）。
    查不到卡面时保守地当"会"。"""
    if not piece:
        return True
    if not _CARD_SKILL:
        try:
            cards = json.loads((PROJECT_DIR / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
            _CARD_SKILL.update({n: (c.get("skill") or "") for n, c in cards.items()})
        except Exception:                                   # noqa: BLE001
            return True
    sk = _CARD_SKILL.get(piece)
    return True if sk is None else bool(IDENTITY_CHANGE_RE.search(sk))


def forget_seen_cells(act: dict) -> None:
    """**某一格换了人，"这一格站的是谁"的看过结果就作废**。

    2026-09-24 数据局：第 6 回合核对 (276,501) 看到李元芳并缓存 → 卖掉李元芳、第一张曜上到这一格 →
    第二张曜要合成时沿用缓存"那儿是李元芳"，没合成，还把场上的曜（主C）从账本里撤了，耽误一整回合
    （现象：买了两张曜，第一个上场了，第二张没用来升级）。原来只在换回合时清。
    卖人（src 空出来）、上场/合成（dst 来了人）、挪位（两头都变）之后都清掉对应格子。
    """
    t = act.get("type")
    cells = []
    if t in ("sell_unit", "sell") and act.get("src"):
        cells.append(act["src"])
    if t in ("drag_card", "deploy") and act.get("dst") and act.get("kind", "英雄牌") == "英雄牌":
        cells.append(act["dst"])
    if t == "move_unit":
        cells += [c for c in (act.get("src"), act.get("dst")) if c]
    if (t == "drag_card" and act.get("kind") in ("效果牌", "天赋牌") and act.get("target") and act.get("dst")
            and _changes_identity(act.get("piece"))):
        cells.append(act["dst"])          # 对英雄用了牌（移除 / 转化 / 觉醒进度…）：那一格可能换人或空了
    if t == "sequence":
        for sub in act.get("steps") or act.get("actions") or []:
            forget_seen_cells(sub)
    for c in cells:
        c = tuple(c)
        _MERGE_SEEN["cells"].pop(c, None)
        _BOARD_SEEN["cells"].pop(cell_key(c), None)
        BOARD_UNREADABLE.discard(cell_key(c))   # 那一格换了人，重新有机会认
        BOARD_MISS.pop(cell_key(c), None)
        # 只有"挪位 / 对英雄用了牌"才需要点开复查：上场、合成有拖拽回执，卖人、觉醒回手有视觉（rules.vision_absence）兜底
        if (t == "move_unit" and act.get("piece") != "古币机") or (t == "drag_card" and act.get("kind") in ("效果牌", "天赋牌")
                                and _changes_identity(act.get("piece"))):
            RECHECK_CELLS.add(cell_key(c))


def settle_card_pick(ctx, controller, extractor, act) -> dict:
    """**等 N 选 1 的几张牌铺开、停稳再点**（和装备四选一同一个道理）。

    2026-09-23：效果牌打出后弹三选一，三张牌还在飞入、叠在一起时就点了 (538,230)，
    这一下没选中任何一张，三选一被收了起来（之后要从能量圆盘重新打开）。
    做法：连续两次探测到要点的那张牌位置一致（±6px）才点，最多等 1.5 秒。
    """
    from agent.perception.state import GameState as _GS
    want = act.get("piece")
    prev = None
    deadline = time.time() + 1.5
    while time.time() < deadline:
        probe = _GS()
        extractor._detect_card_pick(ctx, screencap(controller), probe)
        opt = next((o for o in probe.card_options if o["name"] == want), None)
        if opt is None:
            time.sleep(0.15)
            continue
        cur = (opt["x"], opt["y"])
        if prev and abs(cur[0] - prev[0]) <= 6 and abs(cur[1] - prev[1]) <= 6:
            if cur != (act.get("x"), act.get("y")):
                logger.info("【N选1】%s 停稳在 %s（决策时读到 %s）", want, cur, (act.get("x"), act.get("y")))
            act = dict(act, x=cur[0], y=cur[1])
            break
        prev = cur
        time.sleep(0.15)
    time.sleep(0.2)
    return act


def confirm_merge_target(ctx, controller, executor, extractor, rules, state, act) -> bool:
    """**合成之前先确认目标格子上站的就是同名英雄**（点开看，约 1 秒；同一格每回合只看一次）。

    2026-09-23："试三次"不是正确的判断逻辑。
    那局账本以为 (1027,572) 站着弈星，其实是别人 —— 于是往那儿"合成"了十几次（被游戏拒收、牌弹回手里），
    "试 3 次就停"只是止损，停下来之后下回合还会再撞。**正确的做法是先弄清楚那一格是谁**：
      - 是它：照常合成；
      - 不是：把账本改成实际看到的英雄，这次不合成 —— 这张牌回到正常的上场/换人/卖掉流程。
    看不清（没弹出详情，比如点空了）就照拖，交给拖拽回执。
    """
    # 合成：那一格该站的是这张牌本身；装备/效果牌：该站的是 `target`
    piece = (act.get("piece") or "") if act.get("merge") else (act.get("target") or "")
    cell = tuple(act.get("dst") or ())
    if not cell or not piece:
        return True
    if piece == "古币机" and tuple(rules.coin_machine_cell or ()) == cell:
        return True          # 古币机由模板匹配连续两帧确认；点开它不弹英雄详情，核对只会误判"没人"
    if state.round_no != _MERGE_SEEN["round"]:
        _MERGE_SEEN.update({"round": state.round_no, "cells": {}})
    to_hero = not act.get("merge")          # 装备牌/效果牌：必须落在英雄身上
    _un0 = rules.trusted_unit_names(state).get(cell_key(cell))
    if act.get("_fresh"):
        # **卖场上英雄：不用本回合缓存的核对结果**（第 851 局 R5：账本把米莱狄 / 孙尚香的位置记反，卖"米莱狄"那一格其实是孙尚香，
        # 那一格本回合早先核对过 → 直接放行，卖错人、枫叶舞宿主没了也没换绑）。卖人收不回来：每次都按这一帧重新认
        _MERGE_SEEN["cells"].pop(cell, None)
        if _un0 and _un0[0] != piece:
            logger.warning("【卖前核对】认人模型说 %s 站的是 %s，不是要卖的 %s —— 不卖，改账本", cell, _un0[0], piece)
            rules.note_src_missing(piece)
            return False
    if _un0 and _un0[0] == piece:
        # 认人模型这一帧确认那格就是目标 → 直接放行（2026-10-11 第 573 局：认人模型确认守约在 (300,569)，
        # D-FINE 那一帧漏检判"没人"，结论被缓存一整回合 → 古币抓了 3 次都不拖、被搁置，转瞬牌回合末消失）
        _MERGE_SEEN["cells"][cell] = piece
        return True
    if cell in _MERGE_SEEN["cells"]:
        seen = _MERGE_SEEN["cells"][cell]
        if seen == piece or (seen is None and not to_hero):
            return True
        if seen is None:
            rules.note_src_missing(act.get("piece"))
            return False
    else:
        vu = state.vision_units or {}
        if vu and cell_key(cell) not in vu and not state.shop_open:
            # **视觉说这一格没人，就别点开"确认"**：点 (x, y-30) 会点中上一格那个人的身体，
            # 详情面板报的是它 → 把错的位置"确认"成真（2026-09-28 第 27 局 百里守约，见上场校正处）。
            # 只算这一帧（不缓存成一整回合的"没人"）：D-FINE 单帧漏检很常见（2026-10-11 第 573 局）
            logger.warning("【拖前核对】视觉看 %s 没人（目标 %s）—— 不点开、不往那儿拖", cell, piece)
            rules.note_empty_cell(cell)
            rules.note_src_missing(act.get("piece"))
            return False
        anchor = (state.unit_anchors or {}).get(cell_key(cell))
        x, y = ((anchor[0], anchor[1] + UNIT_BODY_DY) if anchor else (cell[0], cell[1] - 30))
        t0 = time.time()
        un = rules.trusted_unit_names(state).get(cell_key(cell))
        if un:
            # 认人模型已经认出这一格是谁（门槛 0.5 采纳准确率 99.94%）→ 不点开（点开会点歪到隔壁，2026-09-30 连出三次）
            seen, closed = un[0], True
            logger.info("【拖前核对】认人模型：%s 是 %s（%.2f），不点开", cell, un[0], un[1])
        else:
            seen, closed = _tap_read_unit(ctx, controller, executor, extractor, int(x), int(y))
        _MERGE_SEEN["cells"][cell] = seen
        logger.info("【拖前核对】%s 那一格站的是 %s（目标是 %s），用时 %.1fs",
                    cell, seen or "看不清", piece, time.time() - t0)
        if not closed:
            logger.error("【合成前核对】英雄详情面板关不掉（证据帧已存）")
        if seen == piece:
            rules.confirm_unit(piece, cell)     # 直接证据记下来：之后"试够次数"的止损不会再撤销它
            return True
        if seen is None:
            if not to_hero:
                return True
            # **点开没弹出详情，先问视觉**（D-FINE）：最下面一行常被手牌区挡住，点下去弹不出详情 ≠ 那格没人。
            # 2026-09-25 第四轮第 5 局：复印机目标孙尚香在 (300,569)，核对点开没弹详情就判"空地"，
            # 整个第 5 回合每帧被拦，第 6 回合往同一格拖才成功（现象：复印机这张卡一直没用出去）。
            if (state.vision_units or {}).get(cell_key(cell)) == "hero":
                logger.info("【拖前核对】%s 点开没弹详情，但视觉看到那格有英雄 —— 照拖 %s", cell, act.get("piece"))
                return True
            # **点开没看到英雄 = 那一格多半是空的**（点中单位一定会弹详情）：装备/效果牌不往空地上拖。
            # 2026-09-24 第二轮第 4 局：账本把墨子记在 (555,569)/(783,438)，反伤重甲一直往空地上拖、举起来补点也没人接
            # （现象：反甲一直往空地上拖）。原来"看不清就照拖"，于是每回合都撞同一个空格。
            logger.warning("【拖前核对】%s 点开没看到英雄 —— %s 不往空地上拖，这一格记一次失败", cell, act.get("piece"))
            rules.note_empty_cell(cell)
            rules.note_src_missing(act.get("piece"))
            return False
    other = rules._field.get(seen) if seen else None
    if other and tuple(other) != tuple(cell) and abs(other[0] - cell[0]) + abs(other[1] - cell[1]) <= 180:   # 130→180（第 27 局：李元芳在 (390,502)、目标弈星在 (300,569)，差 157）
        # **看到的人账本里记在隔壁格 → 多半是点歪了、看到的是隔壁**：这次核对不算数，不改账本，这一帧也不动手。
        # 2026-09-30 第 23 局：武则天在 (300,569)、孙尚香在隔壁 (276,501)；核对点在 (300,539) 看到孙尚香，
        # 账本把武则天删了、孙尚香改记到 (300,569)，随后"卖孙尚香"按账本卖掉的是武则天。
        logger.warning("【拖前核对】%s 那格看到的是 %s，但 %s 记在隔壁 %s —— 多半点歪了，不改账本，这一帧不拖",
                       cell, seen, seen, tuple(other))
        rules.note("核对点歪", 目标=piece, 格=f"{cell[0]},{cell[1]}", 看到=seen, 看到的人记在=f"{other[0]},{other[1]}")
        rules.note_unclear_cell(cell)           # 本回合别再瞄这格，牌改给别的合规目标（不搁置整张牌）
        return False
    logger.warning("【拖前核对】%s 不在 %s（那儿是 %s）—— 改正账本，%s 不往那儿拖",
                   piece, cell, seen, act.get("piece"))
    if extractor._card_cats.get(seen) == "英雄牌":
        rules.confirm_unit(seen, cell)          # 同时会把记错在这一格的 piece 撤掉（一格一人）
    if rules._field.get(piece) and tuple(rules._field[piece]) == cell:
        rules._field.pop(piece, None)
    rules.note("合成前核对", 牌=piece, 格=f"{cell[0]},{cell[1]}", 实际=seen, 处理="改账本、不合成")
    # **拒绝也要计数**（"试够次数就停"又一个漏掉的入口）：核对不过 = 没拖 = 不算失败，
    # 下一帧决策层给出一模一样的动作，每 0.3 秒一次、整回合卡死（2026-09-23 第八局 推演战术）。
    rules.note_src_missing(act.get("piece"))
    return False


_SC_STATS = {"n": 0, "agree": 0, "diff": 0, "shots": 0, "last": 0.0, "pairs": {}}
_SC_LENIENT = {("battle", "unknown"), ("unknown", "battle")}


def _screen_cls_shadow(state, image, run_dir) -> None:
    """界面分类模型**影子核对**：和 OCR 判出的阶段比，置信 ≥ 门槛且不一致 → 记日志 + 存帧（每局 ≤60 张、每 10 秒 ≤1 张）。
    跑几局看分歧分布，再按类放权（建议先放拍卖 / 奖励展示 / 英雄面板 / N 选 1）。"""
    if not state.screen_cls:
        return
    try:
        from agent.perception import screen_cls as _SC
        label, conf = state.screen_cls
        if conf < _SC.threshold():
            return
        mp = _SC.to_phase(label)
        ocr = state.phase
        _SC_STATS["n"] += 1
        same = (mp == ocr or (mp is None and ocr == "unknown")
                or (label in ("board_idle", "battle") and ocr in ("battle", "unknown"))
                or (label == "prepare_shop" and ocr == "prepare")
                or (label in ("hero_panel", "equip_popup") and ocr in ("prepare", "unknown")))
        if same:
            _SC_STATS["agree"] += 1
            return
        _SC_STATS["diff"] += 1
        key = f"{ocr}->{label}"
        _SC_STATS["pairs"][key] = _SC_STATS["pairs"].get(key, 0) + 1
        now = time.time()
        if now - _SC_STATS["last"] > 10 and _SC_STATS["shots"] < 60:
            _SC_STATS["last"] = now
            _SC_STATS["shots"] += 1
            logger.info("【界面模型分歧】OCR=%s 模型=%s(%.2f) —— 存帧（累计 一致 %d / 分歧 %d）",
                        ocr, label, conf, _SC_STATS["agree"], _SC_STATS["diff"])
            _save_evidence(image, f"screen_diff_{ocr}_{label}")
    except Exception:                                               # noqa: BLE001
        logger.exception("界面模型影子核对出错（忽略）")


SC_GRANT = {"auction", "hero_pick", "match_found", "settle", "settle_final",   # 第一批放权（第 45 局分歧截图核对：这几类模型都对）
            "battle"}   # 2026-10-09：有的地图右上角图标没有「统计」字样，战斗画面 OCR 认不出 → unknown → 每回合在战斗中盲点一次棋盘
                        # （第 468~479 局 184 次）。模型判这些帧 battle 0.999


def _screen_cls_grant(state, image, ctx, extractor):
    """**OCR 判成 unknown、模型置信 ≥ 门槛且属于已放权的类** → 按模型的阶段重新提取这一帧（2026-09-30 第 45 局后放权）。
    OCR 已给出明确阶段的不改；N 选 1 要先定位选项，只改阶段没用，暂不放。"""
    if state.phase == "auction" and state.screen_cls:
        # **拍卖结束后的"获得分红"展示**顶部也有小字「拍卖」，OCR 判成拍卖、读出空拍品（2026-10-11 识别审查）；
        # 界面模型在这些帧判 reward_popup 0.88~0.97 → 模型高置信判成奖励展示时，否决文字判据
        try:
            from agent.perception import screen_cls as _SC0
            _lb, _cf = state.screen_cls
            if _lb == "reward_popup" and _cf >= _SC0.threshold():
                if state.auction and state.auction.lots:
                    logger.info("【界面模型】拍卖后的分红展示（%s %.2f）：清掉读出的拍品 %s，不出题、不出价",
                                _lb, _cf, [l.piece for l in state.auction.lots])
                    state.auction.lots = []
                return state
        except Exception:                                           # noqa: BLE001
            logger.exception("界面模型否决出错（沿用 OCR 结果）")
    if state.phase != "unknown" or not state.screen_cls:
        return state
    try:
        from agent.perception import screen_cls as _SC
        label, conf = state.screen_cls
        mp = _SC.to_phase(label)
        if conf < _SC.threshold() or mp not in SC_GRANT:
            return state
        st2 = extractor.extract(ctx, image, phase=mp)
        st2.screen_cls = state.screen_cls
        logger.info("【界面模型接管】OCR 认不出，模型判 %s(%.2f) → 按 %s 处理", label, conf, mp)
        return st2
    except Exception:                                               # noqa: BLE001
        logger.exception("界面模型接管出错（沿用 OCR 结果）")
        return state


def find_coin_machine(ctx, controller, executor, extractor, rules, state, image=None) -> None:
    """找「古币机」站在哪一格（天赋召唤的物件；D-FINE 检测不到、点开也不弹英雄详情）。
    2026-10-12 改成**图像模板匹配**（extractors.find_coin_machine，~130ms）：原来逐格点开找，拿过古币机的局一次都没找到，
    古币全打给了英雄（古币要优先给古币机）。连续两帧同一格才采信；找到后每 4 个准备帧复查一次（挪位后跟上新位置）。
    没拿天赋（coin_machine_wanted=False）时每回合也看一眼 —— 天赋没认出来也能接住。"""
    if image is None or state.shop_open:
        return
    rnd = state.round_no
    P = _COIN_PROBE
    P["frame"] = P.get("frame", 0) + 1
    if not rules.coin_machine_wanted and P.get("idle_round") == rnd:
        return
    if rules.coin_machine_cell and P.get("frame", 0) % 4:
        return
    P["idle_round"] = rnd
    from agent.perception.extractors import find_coin_machine as _find
    r = _find(image)
    if not r:
        P["last"] = None
        return
    cell, score = r
    if P.get("last") == cell:
        if tuple(rules.coin_machine_cell or ()) != tuple(cell):
            rules.coin_machine_wanted = True
            rules.note_object(tuple(cell), "古币机")
            logger.info("【古币机】在 %s（模板匹配 %.2f，连续两帧），之后古币都打给它", cell, score)
            _save_evidence(image, "coin_machine")
    P["last"] = cell


_COIN_PROBE: dict = {}
_CUR_ROUND = [0]          # 当前回合号（点击 gate 的证据帧按回合限流）


def _fixed_tap(executor, name: str, x: int, y: int, image=None, verified: bool = False, why: str = "") -> bool:
    """写死坐标的兜底点击：先过界面 gate（agent/actions/tap_gate.SAFE_TAPS）。返回是否真的点了。
    executor 没有 _tap_fixed（离线测试的假执行器）时自己过 gate 再调 _tap。"""
    f = getattr(executor, "_tap_fixed", None)
    if f is not None:
        return f(name, x, y, image=image, verified=verified, why=why)
    if tap_gate.gate(name, (x, y), image=image, verified=verified, why=why):
        executor._tap(x, y)
        return True
    return False


def inspect_board(ctx, controller, executor, extractor, rules, state) -> None:
    """画面上看得到有人、账本里却没记的格子：**点开看是谁**，记进账本。"""
    from agent.perception.extractors import TAP_SIDE_GROUND
    if state.round_no != _BOARD_SEEN["round"]:
        _BOARD_SEEN.update({"round": state.round_no, "cells": {}, "count": 0})
    anchors = state.unit_anchors or {}
    known = {f"{c[0]},{c[1]}" for c in rules._field.values()}
    # **账本里记了的格子也要复查**（2026-09-23：多读场上信息来修正账本）。
    # 原来只点"账本没记的"，记错了（把别人记成弈星、记着的人早就卖了）就一直错下去，
    # 后果是往错的人身上合成/贴装备、人口其实满了还去上场。
    # 每回合每个看得见的单位点一次：**没记的先看**（最可能错），再核对记了的。
    # **多用视觉、少点开**（2026-09-24 match14：查看场上英雄过于频繁，视觉本来就能看出哪个位置有英雄，
    # 而且很快；那局点开看场 103 次、每次约 1 秒）。只点两类：
    #   ① 视觉看到有人、账本不知道是谁的格子（招募/召唤/抓错牌送上场的）；
    #   ② 被动作改变过的格子（上场、合成、卖人、挪位、对英雄用了牌 —— RECHECK_CELLS）。
    # 账本记着、视觉也看到、这回合没人动过的格子不再例行复查；人不在了由 rules.vision_absence 用视觉判断。
    # 另外跳过：视觉标成**召唤物**的格子（617,496 一局被点空 5 次）、本局点开**认不出名字**的格子（墨子读成"福/黑"，每回合重点一遍）
    vis = state.vision_units or {}
    # **只点连续两次都看到的格子**（2026-10-18：617,496 这类幽灵格只在单位拖动 / 挪位的过渡帧里出现，
    # 上千帧里十几次、坐标每次不同；保存的 33 张准备帧徽章 / D-FINE 都不给它 —— 点开必然点空，每次 5~6 秒）
    _prev_t, _prev_keys = _BOARD_SEEN.get("prev_anchors") or (0.0, set())
    _stable = set(anchors) & _prev_keys if time.time() - _prev_t < 4.0 else set()
    _BOARD_SEEN["prev_anchors"] = (time.time(), set(anchors))
    todo = sorted(((k, a) for k, a in anchors.items() if k not in _BOARD_SEEN["cells"]
                   and (k in _stable or k in RECHECK_CELLS)
                   and (k not in known or k in RECHECK_CELLS)
                   and vis.get(k) != "summon" and k not in BOARD_UNREADABLE),
                  key=lambda ka: ka[0] in known)
    # **要挪的图腾身份没确认时，点开视觉标成召唤物的格子读名字**（2026-10-13 第 605 局：按"离召唤者最近"猜身份把因缘图腾挪到了前排）。
    # 召唤物没有等级徽章，点击点取格心上方 40px（锚点按徽章算：格心 -80，再 +UNIT_BODY_DY）
    for (cx, cy) in rules.totem_inspect_cells(state):
        k = f"{cx},{cy}"
        if k not in _BOARD_SEEN["cells"] and k not in BOARD_UNREADABLE and all(k != kk for kk, _ in todo):
            todo.insert(0, (k, (cx, cy - 80)))
    if not todo:
        return
    budget = min(BOARD_INSPECT_PER_FRAME, BOARD_INSPECT_PER_ROUND - _BOARD_SEEN["count"])
    if budget <= 0:
        return
    logger.info("【看场】账本记了 %d 个、画面上看到 %d 个，点开 %s 看是谁（含复查）",
                len(rules._field), len(anchors), [k for k, _ in todo[:budget]])
    for key, (ax, ay) in todo[:budget]:
        un = rules.trusted_unit_names(state).get(key)
        if un:
            # 认人模型认出来了就不点开（2026-09-30）
            rules.confirm_unit(un[0], tuple(int(v) for v in key.split(",")))
            _BOARD_SEEN["cells"][key] = un[0]
            RECHECK_CELLS.discard(key)
            logger.info("【看场】认人模型：%s 是 %s（%.2f），不点开", key, un[0], un[1])
            continue
        _BOARD_SEEN["count"] += 1
        RECHECK_CELLS.discard(key)
        t0 = time.time()
        executor._tap(ax, ay + UNIT_BODY_DY)
        shot = None
        for _ in range(7):
            time.sleep(0.15)
            s2 = screencap(controller)
            if _hero_panel_up(ctx, s2):
                shot = s2
                break
        if shot is None:
            # **召唤物的详情面板没有「出售」**（2026-10-12 第 583 局：鲁班大师召唤的「机关傀儡」被点开 2 次都判"没弹出"，
            # 面板留在屏幕左侧没人关）。读面板卡名：认出是召唤物 / 非英雄 → 记成占格物件、关面板、本局不再点这一格。
            s3 = screencap(controller)
            texts = [t for t, _b in _ocr_boxes(ctx, s3, ROI_UNIT_PANEL_NAME)]
            # 只认卡库里查得到的非英雄牌（keep_raw=False）：原来保留原文，左侧玩家列表的昵称（RudAns / 冰酸梅…）
            # 被当成"召唤物"，第 584~587 局封掉 34 格、其中一格站着苏烈（2026-10-12 识别审查）
            other = next((n for n in (extractor.resolve_name(t, keep_raw=False) for t in texts) if n), None)
            if other and other in extractor._card_cats and extractor._card_cats.get(other) != "英雄牌":
                try:
                    rules.note_object(tuple(int(v) for v in key.split(",")), other)
                except ValueError:
                    pass
                BOARD_UNREADABLE.add(key)
                _fixed_tap(executor, "TAP_SIDE_GROUND", *TAP_SIDE_GROUND, image=s3, verified=True, why="看场后关召唤物详情")
                time.sleep(0.4)
                logger.info("【看场】%s 是召唤物「%s」（%s），记成占格物件、关面板，本局不再点它",
                            key, other, extractor._card_cats.get(other) or "?")
                _BOARD_SEEN["cells"][key] = None
                continue
            logger.info("【看场】点了 %s 没弹出英雄详情（可能是召唤物/图腾或点空了）", key)
            _BOARD_SEEN["cells"][key] = None
            # 跨回合连续 2 次点空 → 本局不再点（第 722 / 723 局 617,496 各点空 9 次，每次 5~6 秒；那一格换了人时 forget_seen_cells 会清）
            BOARD_MISS[key] = BOARD_MISS.get(key, 0) + 1
            if BOARD_MISS[key] >= 2:
                BOARD_UNREADABLE.add(key)
                logger.info("【看场】%s 连续 %d 回合点开都没弹出详情 → 本局不再点（换了人会重新看）", key, BOARD_MISS[key])
            continue
        texts = [t for t, _b in _ocr_boxes(ctx, shot, ROI_UNIT_PANEL_NAME)]
        name = next((extractor.resolve_name(t) for t in texts if extractor.resolve_name(t)), None)
        # 装备轮盘证据帧（2026-10-14 确认：点带装备的英雄会展开装备轮盘，拖出去可转给别人）：
        # 每局最多存 8 张，供做轮盘槽位识别；只存图，不做任何拖拽
        if (state.equip_counts or {}).get(key, 0) > 0 and WHEEL_SHOTS[0] < 8:
            WHEEL_SHOTS[0] += 1
            try:
                import cv2
                _imwrite(_RUN_DIR[0] / f"wheel_{key.replace(',', '_')}_{int(time.time())}.png",
                         cv2.cvtColor(shot, cv2.COLOR_RGB2BGR))
                logger.info("【装备轮盘】存证据帧（%s，%d 件装备）", key, state.equip_counts.get(key))
            except Exception:                    # noqa: BLE001
                logger.exception("存装备轮盘证据帧失败")
        # 关面板：点棋盘右侧空地（主循环关这个面板用的就是这个点），并确认关掉
        for _attempt in (1, 2):
            _fixed_tap(executor, "TAP_SIDE_GROUND", *TAP_SIDE_GROUND, image=shot if _attempt == 1 else None,
                                verified=True, why="看场后关英雄详情")
            closed = False
            for _ in range(5):
                time.sleep(0.15)
                if not _hero_panel_up(ctx, screencap(controller)):
                    closed = True
                    break
            if closed:
                break
        _BOARD_SEEN["cells"][key] = name
        if not name and texts:
            BOARD_UNREADABLE.add(key)          # 面板弹出了、名字认不出：本局别再点这一格
        cell = tuple(int(v) for v in key.split(","))
        cat = extractor._card_cats.get(name) if name else None
        logger.info("【看场】%s 站的是 %s%s，用时 %.1fs", key, name or f"没认出{texts}",
                    f"（{cat}）" if cat and cat != "英雄牌" else "", time.time() - t0)
        _save_evidence(shot, "board_inspect")
        if name and cat and cat != "英雄牌":
            rules.note_object(cell, name)      # 重剑 / 古币机 / 图腾…：占格物件，不是英雄
        if name and cat == "英雄牌":
            old = rules._field.get(name)
            vis = state.vision_units or {}
            # 视觉（D-FINE）说账本记的旧格**没人** → 真挪了位，改账；旧格也有人 → 点开看到的可能是隔壁的人，不改账。
            # （第 27 局试过"两格都有人就改账"：左下两格离得太近，点哪格都看到干将，第 29 局账本来回改了 31 次。
            #   两格都有人时到底谁是谁，交给认人模型）
            old_seen = vis.get(f"{old[0]},{old[1]}") == "hero" if (old and vis) else True
            # D-FINE 说账本那一格**没人** → 不是徽章映射到相邻两格，是真挪了位（重连 / 被挤开），改记账
            # （2026-09-30 第 25 局：卡死重连后苏烈在 (617,496)，账本还记 (672,430)，"同一个苏烈"不改账，
            #  换绑枫叶舞要卖苏烈一直被"那格没人"拦下）
            if old and tuple(old) != cell and f"{old[0]},{old[1]}" in anchors and old_seen:
                # **同一个单位被徽章映射到了相邻两格**（徽章在头顶、纵向容差放得宽）——
                # 实测米莱狄在 (955,502) 和 (917,570) 之间来回：搬过去，另一格又成了"没记的"，
                # 再点一次、再搬回来，每次 1.1 秒。那一格此刻也看得到人，就别动账了。
                logger.info("【看场】%s 和账本里的 %s 是同一个 %s（徽章映射到相邻两格），不改记账",
                            key, old, name)
            else:
                rules.confirm_unit(name, cell)
        if not closed:
            _save_evidence(screencap(controller), "board_inspect_cannot_close")
            logger.error("【看场】英雄详情面板关不掉，本局不再点场上单位")
            _BOARD_SEEN["count"] = 10 ** 6
            return


# ==================== 拖牌前确认"抓的是不是这张" ====================
# 2026-09-23：买卡牌、或者卖英雄导致手牌里多出卖掉英雄的装备，都会让手牌位置改变。
# 日志完全对得上：卖掉带装备的莱西奥 -> 3 秒后按**上一帧**的位置 (805,663) 去抓扁鹊 ->
# 回执"拖完手牌反而多了(5->6)"（装备退回、整排手牌重新居中）-> 再按旧位置抓一次，没生效。
# 手牌是居中排列的：多一张、少一张，所有牌都会挪位置；新牌还是带动画滑进来的。
HAND_SRC_MIN_Y = 600          # src 在这以下 = 从手牌里拖


def _same_card(extractor, text: str, piece: str) -> bool:
    if not text or not piece:
        return False
    if extractor.resolve_name(text) == piece:
        return True
    # 叠放时只露出前一截（「孙」「干卷」）：用和手牌识别同一套前缀规则
    if extractor._resolve_hand_fragment(text) == piece:
        return True
    # 名字读不全（「翩跹舞步」这一族常被切掉后缀）：前缀对得上也算
    return len(text) >= 2 and (text.startswith(piece) or piece.startswith(text))


def try_reforge(ctx, controller, executor, extractor, rules, run_dir, state) -> None:
    """**补选没铸成的装备**（2026-09-24：白色光环有两种：一种是莲花那种激活中，另一种是没有选择的；
    点击一次确认，并记账，防止以后无限点击）。

    只对**账本里记着**"这一格有一件没铸成的基础装备"的格子做（不靠光环去猜 —— 莲花类激活装备也有光环）。
    做法：找到那一格的等级徽章 → 徽章下方那排装备图标里**白色外圈最明显**的那个 → 按住点一下 →
    弹出四选一就按正常流程选成品；没弹就存证据帧。**每件只点一次**（rules.next_reforge 取出即记为试过）。
    """
    import cv2
    import numpy as np
    from agent.perception.extractors import _detect_badges, _hough
    job = rules.next_reforge()
    if not job:
        return
    key, base = job
    shot = screencap(controller)
    badges = _detect_badges(shot)
    if key not in badges:
        logger.info("【补选装备】%s 这一帧没看到徽章，这次不点（已记为试过）", key)
        return
    bx, by = badges[key]
    x0, y0 = max(0, bx - 45), by + 10
    sub = shot[y0:by + 42, x0:bx + 45]
    icons = _hough(cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY), 13, 6, 12, 14)
    if not icons:
        logger.info("【补选装备】%s 徽章下面没找到装备图标，这次不点", key)
        _save_evidence(shot, f"reforge_noicon_{base}")
        return
    hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)
    hh, ww = hsv.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww]

    def ring(ix, iy, r):          # 外圈（r+2~r+6）里"白"（低饱和、高亮度）像素的占比
        m = ((xx - ix) ** 2 + (yy - iy) ** 2 >= (r + 2) ** 2) & ((xx - ix) ** 2 + (yy - iy) ** 2 <= (r + 6) ** 2)
        if not m.any():
            return 0.0
        return float(((hsv[..., 1] < 50) & (hsv[..., 2] > 200))[m].mean())

    ix, iy, r = max(icons, key=lambda c: ring(*c))
    tx, ty = x0 + int(ix), y0 + int(iy)
    _save_evidence(shot, f"reforge_before_{base}")
    logger.info("【补选装备】%s 身上没铸成的 %s：点图标 (%d,%d)（白圈占比 %.2f）", key, base, tx, ty, ring(ix, iy, r))
    rules.note("补选装备", 格=key, 装备=base, 点=f"{tx},{ty}")
    executor._tap_hold(tx, ty) if hasattr(executor, "_tap_hold") else executor._tap(tx, ty)
    cx, cy = (int(v) for v in key.split(","))
    wearer = rules._cell_owner.get((cx, cy)) or ""
    ok = grab_equip_popup(ctx, controller, executor, extractor, rules, run_dir, (cx, cy), base, wearer,
                          rules.equip_role_for(wearer),
                          (state.equip_counts or {}).get(key, 0), budget=1.5, expect=False)
    if ok:
        logger.info("【补选装备】✅ 点一下弹出了四选一，已补选（%s）", base)
        rules.note("补选装备成功", 格=key, 装备=base)
    else:
        logger.info("【补选装备】点了没弹四选一 —— 记账，这件不再点（证据帧已存）")
        _save_evidence(screencap(controller), f"reforge_after_{base}")


HAND_STRIP = (560, 720, 200, 1100)      # 手牌条 y0,y1,x0,x1（判断"停稳"用）
HAND_STILL_DIFF = 4.0                   # 两帧平均像素差低于这个 = 停稳了
HAND_SETTLE_MAX = 1.2                   # 最多等多久


EQUIP_OPT_MIN_X = 170    # 2026-10-13 复盘：x=176~226 的选项截图上完全可见，280 → 170。 装备四选一的选项落在这条线左边就点不到（英雄详情面板 / 棋手阿离）。2026-10-13 第 588 局：300 → 280，最左格单位的右上选项在 x≈289~298（历史 113 次），截图上无遮挡（equip_popup_风暴巨剑_3_1791832757.png），原来被当成点不到、法师沈梦溪改铸了物理装
_GRAB_CTX: dict = {}     # 拖前核对用的那一帧（"停稳"的手牌），半空确认时和它比对
GRAB_SHADOW_MAX = 60     # 半空确认影子模式每个进程最多存几组帧


def _hand_settled_shot(controller):
    """**等手牌停稳再截图**（2026-09-24 第二轮第 1 局，证据帧 wrong_grab_蒙犽_李信）：
    上一张牌刚拖出去，剩下的牌还在往中间滑、重新排位 —— 核对那一刻抓取点下面是「蒙犽」，
    按下去的那一瞬滑过来的是「李信」。隔 0.2 秒连截两张，手牌条基本没变才算停稳。"""
    import numpy as np
    y0, y1, x0, x1 = HAND_STRIP
    shot = screencap(controller)
    deadline = time.time() + HAND_SETTLE_MAX
    while time.time() < deadline:
        time.sleep(0.2)
        nxt = screencap(controller)
        try:
            diff = float(np.mean(np.abs(nxt[y0:y1, x0:x1].astype(np.int16) - shot[y0:y1, x0:x1].astype(np.int16))))
        except (TypeError, ValueError, IndexError):
            return nxt
        shot = nxt
        if diff < HAND_STILL_DIFF:
            break
    else:
        logger.info("【抓牌校验】等了 %.1fs 手牌还在动，按最后一帧核对", HAND_SETTLE_MAX)
    _GRAB_CTX["pre_shot"], _GRAB_CTX["t"] = shot, time.time()
    return shot


def confirm_hand_src(ctx, controller, extractor, act) -> bool:
    """拖之前**当场**看一眼抓取点下面是不是这张牌。对不上就重读手牌、按新位置拖；
    找不到就返回 False（这一帧别拖）。画面没变时 OCR 走缓存，几乎不花时间。"""
    from agent.perception.extractors import HAND_GRIP_DY
    from agent.perception.state import GameState
    piece = act.get("piece") or ""
    sx, sy = act["src"]
    shot = _hand_settled_shot(controller)
    if act.get("awk"):
        # **觉醒卡按颜色核对抓取点**（2026-09-23 第十三局：卖掉孙膑后装备退回、手牌重排，按旧坐标抓起来的是
        # 别的英雄 —— 它上了场、人口 +1，回执于是报"沈梦溪上场成功"，而金色的觉醒沈梦溪还在手牌最右边）。
        # 卡名 OCR 在叠放时会读到隔壁露出来的碎片，颜色不会。
        from agent.perception.extractors import awaken_tint_spans
        spans = awaken_tint_spans(shot)          # screencap 给的就是主循环/感知层用的同一种图（RGB）
        if spans and not any(x0 <= sx <= x1 for x0, x1, _i, _d in spans):
            x0, x1, _i, _d = spans[-1]                 # 最右边那张觉醒色（新进手的牌排在最右）
            logger.warning("【抓牌校验】觉醒卡 %s 不在 %s（觉醒色的牌在 x=%d~%d），改到那儿抓",
                           piece, (sx, sy), x0, x1)
            act["src"] = ((x0 + x1) // 2, sy)
            return True
        if spans:
            return True
    name_y = sy - HAND_GRIP_DY
    boxes = _ocr_boxes(ctx, shot, (sx - 62, name_y - 17, 124, 34))
    texts = [t for t, _b in boxes]
    hits = [b for t, b in boxes if _same_card(extractor, t, piece)]
    if hits:
        # **按认出来的卡名框中心抓，不按布局推算的 x 抓**（2026-09-24 第二轮第 1 局，证据帧 wrong_grab_*）：
        # 叠放时核对窗口 124px 宽，**隔壁那张的卡名也会落进窗口** —— 核对"通过"了，按下去的却是另一张
        # （想拖孙尚香抓到速击之枪、想拖西施上场的是孙膑；第一轮第 5 局的干将莫邪→牛魔/风暴巨剑同理）。
        # 卡名文字露在外面 = 那一处是这张牌的可见部分，按它的中心 x 抓才一定抓到它。
        b = min(hits, key=lambda b: abs(b[0] + b[2] / 2 - sx))
        cx = int(b[0] + b[2] / 2)
        if abs(cx - sx) > 10:
            logger.info("【抓牌校验】%s 的卡名在 x=%d（布局推算 %d），按卡名位置抓", piece, cx, sx)
            act["src"] = (cx, sy)
        return True
    tmp = GameState(phase="prepare")
    extractor._extract_hand(ctx, shot, tmp)
    cands = [sl for sl in tmp.hand_slots if _same_card(extractor, sl["name"], piece)]
    if not cands and tmp.hand_by_layout:
        # 这张牌是**点开看才认出来的**（露出来的字认不出）：抓取点那个位置
        # 仍是一张"认不出的牌"、且那儿点开看到的就是它，就照拖
        seen = _INSPECT.get("at", {}).get(sx // 20)
        if seen and seen[0] == piece and any(abs(u[0] - sx) <= 8 for u in tmp.hand_unread):
            return True
    if not cands:
        logger.warning("【抓牌校验】%s 不在 %s 了（抓取点读到 %s），手牌里也找不到 —— 这一帧不拖",
                       piece, (sx, sy), texts)
        return False
    best = min(cands, key=lambda sl: abs(sl["x"] - sx))
    if abs(best["x"] - sx) > 8:
        logger.warning("【抓牌校验】%s 从 %s 挪到了 %s（手牌重排了），按新位置拖",
                       piece, (sx, sy), (best["x"], best["y"]))
    act["src"] = (best["x"], best["y"])
    return True


def verify_sell(controller, rules, act) -> bool:
    """**卖人要有回执**：卖完用视觉检测（D-FINE）复查那一格，两次都还站着英雄 = 没卖掉 → rules.undo_sell()。

    2026-09-28 第 26 局：第 7 回合"卖孙膑"没生效，账本删了他、枫叶舞记成"已解绑"，
    枫叶舞整局灰着、主C 没绑上（见 rules.undo_sell）。视觉没启用时判不了，当成功（与原来一致）。
    """
    from agent.perception import vision_units
    if not vision_units.enabled():
        return True
    cell = tuple(act["src"])
    for wait in (1.0, 1.0):
        time.sleep(wait)
        try:
            from agent.perception.extractors import OWN_BOARD_ROWS
            found = vision_units.cell_units(screencap(controller),
                                            [tuple(c) for row in OWN_BOARD_ROWS for c in row])
        except Exception:  # noqa: BLE001
            logger.exception("卖出复查的视觉检测失败（当成功）")
            return True
        if found.get(cell, ("",))[0] != "hero":
            return True
    rules.undo_sell()
    return False


def wait_equip_refund(ctx, controller, extractor, hand_before: int, n_equips: int,
                      budget: float = 6.0) -> None:
    """卖掉带装备的英雄后，**等装备退回手牌**再继续。实测要 2~6 秒才到，
    而一次拖拽要 ~1.7 秒 —— 不等的话，装备正好在下一次拖拽途中滑进来，把手牌挤乱。"""
    from agent.perception.state import GameState
    deadline = time.time() + budget
    while time.time() < deadline:
        time.sleep(0.5)
        tmp = GameState(phase="prepare")
        extractor._extract_hand(ctx, screencap(controller), tmp)
        if len(tmp.hand_slots) >= hand_before + n_equips:
            logger.info("卖人退回的 %d 件装备已经进手牌（手牌 %d -> %d）",
                        n_equips, hand_before, len(tmp.hand_slots))
            return
    logger.info("等了 %.0f 秒，退回的装备还没全部进手牌，先继续", budget)


# 会改变手牌/棋盘的动作：**每帧最多做一个**，做完立刻重新感知。
# 理由是实测出来的：手牌是居中排列的，拿走一张，剩下的全部重新居中。
# 2026-09-22 实战里一帧连发 3 个拖拽，第 2、3 个用的还是拖之前算出来的槽位坐标，
# 于是抓错牌（把牛魔当成李元芳拖上场），后面整局的账都跟着错。
EQUIP_MIN_LEFT_S = 5           # 剩不到这么多秒不拖基础装备：拖拽 + 等四选一弹窗 + 点选 ≈ 3 秒（第 878 局：剩 2 秒拖圣者法典，弹窗刚出来就开战，选没选上全看运气）
AWAKEN_MIN_LEFT_S = 10          # 准备阶段剩不到这么多秒就不触发觉醒（回手 + 认牌 + 拖回 ≈ 5~6 秒，留余量；第 843 局）
MUTATING_ACTIONS = {"drag_card", "move_unit", "deploy", "sell_unit", "sell_card",
                    "sell", "use_card", "equip", "sequence"}

# 按阶段变速：拍卖/天赋/特性是**抢时间**的（拍品几秒就被别人拍走、模态不选就一直挡着），
# 战斗阶段什么都做不了、纯等，慢点轮询省 OCR。其余用 --interval。
# unknown 持续多久才启动兜底。**不能太短**：每回合"战斗→准备"的过场本来就有
# 8~12 秒既不是 battle 也不是 prepare（结算动画/镜头切换），6 秒门槛会让兜底
# 在每个回合间隙都按一次返回键——正常过场按返回键是有风险的（没模态时会弹
# 「是否退出游戏？」）。15 秒足够跨过过场，又能救真正卡死的局面。
UNKNOWN_RESCUE_SEC = 15
# 上场之后"按观测校正落点"能接受的最大位移。棋盘相邻格约 100px(x) / 60-70px(y)，
# 归位最多挪一两格，所以 250 已经很宽松。超过就认为是认错了徽章。
RELOCATE_MAX_DIST = 250
BLANK_ESCAPE_MAX = 3        # 每局最多点几次空白处去关"该关没关"的模态

PHASE_INTERVAL = {
    # **prepare 必须显式给值**：漏了它就会退回 --interval 默认的 2.0 秒，
    # 而准备阶段只有 ~76 秒倒计时、又是所有动作发生的地方 ——
    # 等于一半时间在睡觉（2026-09-22：判断太慢，实测每帧 4.1s）。
    # 感知本身约 0.4s、一次拖拽约 1.7s，0.3s 的间隔已经够让画面稳定下来。
    "prepare": 0.3,
    "auction": 0.4,
    "talent": 0.5,
    "lord_boost": 0.5,
    "card_pick": 0.5,
    # unknown 多半是过场（N 选 1 的卡正在飞入、天赋弹窗展开），原来按默认 2 秒一帧，
    # 界面出来了还要干等 2 秒才看到（2026-09-23 第八局）。
    "unknown": 0.5,
    "battle": 3.0,
    "loading": 3.0,
    # 单场模式结束后就停在这一页等人，慢慢轮询就行（原来 2.4s 一帧刷满屏日志，
    # 看上去像死循环 —— 2026-09-23 看起来像卡住，其实只是在等指示）。
    "settle_final": 6.0,
}


_EVIDENCE_DIR = None          # 由 main() 设成本局的 run_dir
_EVIDENCE_SEEN: set = set()


def _imwrite(path, bgr) -> None:
    """存图，**路径里可以有中文**。cv2.imwrite 在 Windows 上写不了中文路径，
    不报错、而是写出一个乱码文件名（2026-09-23 `equip_popup_椋庢毚宸ㄥ墤_*.png`）。"""
    import cv2
    import threading

    def _w(img=bgr.copy()):             # 编码 + 写盘放后台线程：主循环不再为存证据帧等 ~0.3 秒（2026-10-14 提速）
        try:
            ok, buf = cv2.imencode(Path(str(path)).suffix or ".png", img)
            if ok:
                buf.tofile(str(path))
        except Exception:                # noqa: BLE001
            logger.exception("存图失败 %s", path)
    threading.Thread(target=_w, daemon=True).start()


def _save_hand_label(image, x: int, name: str, awake: bool, xs, read_as: str) -> None:
    """看牌确认了牌名 → 存这张手牌的截图 + 真名，攒着重训手牌立绘模型（2026-10-17：每局 40~150 次"认不出"）。
    存宽一点的条（y 580~720、x−80~x+140），线下按 xs（当时的图标位置）再切成 64x120；每局最多 300 张。"""
    if _EVIDENCE_DIR is None:
        return
    try:
        import cv2
        import numpy as np
        d = _EVIDENCE_DIR / "hand_labels"
        d.mkdir(exist_ok=True)
        meta = d / "labels.jsonl"
        if meta.exists() and sum(1 for _ in meta.open(encoding="utf-8")) >= 300:
            return
        x0, x1 = max(0, int(x) - 80), min(image.shape[1], int(x) + 140)
        fn = f"{int(time.time() * 1000)}_{int(x)}.png"
        _imwrite(d / fn, cv2.cvtColor(np.ascontiguousarray(image[580:720, x0:x1]), cv2.COLOR_RGB2BGR))
        with meta.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"file": fn, "x": int(x), "x0": x0, "y0": 580, "name": name, "awake": bool(awake),
                                "read_as": read_as, "xs": [float(v) for v in (xs or [])]}, ensure_ascii=False) + "\n")
    except Exception:                                           # noqa: BLE001
        logger.exception("存手牌标注图失败")


def _save_evidence(image, tag: str) -> None:
    """留一帧证据。**同一个 tag 每局只存一次**，不然会刷爆磁盘。"""
    if _EVIDENCE_DIR is None or tag in _EVIDENCE_SEEN:
        return
    _EVIDENCE_SEEN.add(tag)
    try:
        import cv2
        fp = _EVIDENCE_DIR / f"{tag}_{int(time.time())}.png"
        _imwrite(fp, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.info("证据帧已存: %s", fp.name)
    except Exception:
        logger.exception("存证据帧失败")


# 拖拽回执的第三种结果：**牌已经举起来了，游戏在等我们点目标**。
# 它既不是成功也不是失败，而且**必须马上补点那一下**，否则牌就卡在半空：
# 装备四选一不会弹、牌也不消耗（2026-09-23 实战：雷鸣刃举起后没人补点，
# 4 秒后报"超时未完成，这件装备没有铸造"）。
# 原来这一下依赖**下一帧还能认出顶部横幅**，而横幅 OCR 漏一帧就永远补不上了。
LIFTED = object()
WRONG_GRAB = object()     # 抓错牌：想拖的那张还在，少的是另一张（act["_wrong"] = 那张的名字）


def _wrong_grab(hand_before, hand_after, piece):
    """拖完手牌恰好少了一张、想拖的那张一张没少、少的是另外**唯一**一张 -> 返回那张的名字。"""
    import collections
    if not hand_after or not hand_before or len(hand_after) != len(hand_before) - 1:
        return None
    b, a = collections.Counter(hand_before), collections.Counter(hand_after)
    if a[piece] < b[piece]:
        return None
    gone = [n for n in b if n != piece and a[n] < b[n]]
    return gone[0] if len(gone) == 1 else None


def verify_drag(ctx, controller, extractor, act, hand_before, pop_before=None):
    """拖完重新读一眼手牌，判断这张牌到底出去了没有。返回 None = 无法判断。

    最省的闭环反馈：一次截图 + 一次 OCR（手牌条），约 100ms。
    判据：手牌里这张牌的**数量减少了** = 真的打出去了。
    拖英雄上场/合成、贴装备/技能牌，全都符合"成功则手牌少一张"。
    """
    from agent.perception.state import GameState
    piece = act.get("piece")
    if not piece or act.get("type") not in ("drag_card",):
        return None                      # move_unit 是场上挪位，手牌不变，判不了
    if act.get("kind") == "棋手技能牌":
        # **棋手技能牌用完仍然留在手牌里**（这条早有记录，之前被忽略了）。
        # 所以"牌还在手里"对它不代表失败 —— 这个判据对它**永远报失败**，
        # 于是每回合都误判一次、还把落点拉黑（2026-09-22 第十局：枫叶舞明明贴到
        # 弈星身上生效了，日志却一直报"没生效"，看画面才发现）。
        # 我们没有别的办法验证它，所以返回"判不了"，让决策层不要据此改状态。
        return None
    probe = GameState()
    try:
        shot = screencap(controller)
        # **「请指定英雄使用」挂着时一律判不了**：这类牌是两段式的 ——
        # 拖出去只是"把牌举起来"，牌**仍然留在手上**，要等补点目标那一下才真正消耗。
        # 立刻按"牌还在手里"判定，就会**每次都误报失败**，
        # 进而拉黑落点格、搁置这张牌 —— 等于我们自己把能用的格子和牌一个个毒死。
        # 2026-09-22 实战：拖拽成功率只有 4/9，推演战术连着两次被判失败并拉黑 (617,372)，
        # 而那一格站着牛魔、本来是合法目标。
        # **只有两段式的牌才需要补点目标**（装备牌/效果牌：拖出去只是举起来，
        # 顶部挂「请指定英雄使用」，要再点一下目标才真正打出）。
        # 英雄牌是直接落格的 —— 对它补点一下反而会把它**重新拿起来**，
        # 或者点出英雄详情面板（2026-09-23 实战：百里玄策 上场后被补点了一下）。
        # 横幅可能是上一张牌残留的、也可能是 OCR 在别处认出了"指定"二字，
        # 所以这里以**牌的类别**为准，不以横幅为准。
        if act.get("kind") in ("装备牌", "效果牌") and _hit(ctx, shot, ROI_DESIGNATE, "指定"):
            logger.info("拖拽校验: %s 已举起、等待指定目标 —— 立刻补点目标", piece)
            return LIFTED
        if not act.get("kind") and _hit(ctx, shot, ROI_DESIGNATE, "指定"):
            # 拖的是英雄牌，顶部却挂出「请指定英雄使用」= 举起来的是一张装备/效果牌（第一轮第 5 局第 2 回合：
            # 想拖干将莫邪，举起的是风暴巨剑）。不是格子的问题，别拉黑。
            act["_wrong"], act["_wrong_pop"] = "(两段式的装备/效果牌)", False
            _save_evidence(shot, f"wrong_grab_{piece}_lifted")
            logger.warning("拖拽校验: 想拖英雄 %s，举起来的却是一张要指定目标的牌（证据帧已存）", piece)
            return WRONG_GRAB
        extractor._extract_hand(ctx, shot, probe)
        other = _wrong_grab(hand_before, probe.hand, piece)
        if other and not act.get("kind"):
            act["_wrong"] = other
            act["_wrong_pop"] = False
            if pop_before and pop_before[2]:
                extractor._extract_population(ctx, shot, probe)
                act["_wrong_pop"] = bool(probe.population_valid and probe.population == pop_before[0] + 1)
            _save_evidence(shot, f"wrong_grab_{piece}_{other}")
            logger.warning("拖拽校验: 想拖 %s，手里少的却是 %s（证据帧已存）", piece, other)
            return WRONG_GRAB
        # **英雄上场的直接证据是人口 +1**，比"手里这张牌少了一张"硬得多：
        # 2026-09-23 第 8 回合，阿轲 上场成功（人口 5/6 -> 6/6），回执却报"没生效" ——
        # 于是账本没记它，手里第二张阿轲被当成新英雄又去"上场"（人口满了还想拖到场上），
        # 本该是合成。手牌读数在叠放、塞新牌时都会乱，人口数字是单独一块、很稳。
        if (pop_before and pop_before[2] and act.get("type") == "drag_card"
                and not act.get("kind") and not act.get("merge")):
            extractor._extract_population(ctx, shot, probe)
            if probe.population_valid and probe.population_cap == pop_before[1]:
                if probe.population == pop_before[0] + 1:
                    return True
                if probe.population == pop_before[0]:
                    time.sleep(0.4)           # 人口数字可能还没刷新，再看一眼
                    probe2 = GameState()
                    extractor._extract_population(ctx, screencap(controller), probe2)
                    if probe2.population_valid and probe2.population == pop_before[0] + 1:
                        return True
    except Exception:
        logger.exception("拖拽校验失败（读手牌出错），本次不给回执")
        return None
    before = hand_before.count(piece)
    if not probe.hand and before:
        # 一张都没读到，多半是画面在动（拖拽动画/卡牌回弹）。**再看一眼**：
        # 原来这里直接返回"判不了"，而它在实战里会连续命中 ——
        # 2026-09-23 回合6，白起 合成连发十几次，一次回执都没拿到，
        # 于是所有"失败够次数就停"的止损全部失效，整个回合作废。
        # 判不出成败比判错还贵：宁可多花 0.4 秒重读一次。
        time.sleep(0.4)
        try:
            probe = GameState()
            shot = screencap(controller)
            # **重读时要把横幅也再认一次**。「请指定英雄使用」是一闪一闪的，
            # 第一次截图没认出来不代表它不在 —— 2026-09-23 实战：雷鸣刃拖完
            # 两次都读成空手牌、横幅也没认出来，于是没人去补点目标，
            # 装备就烂在半空（装备四选一 3 个英雄都没有成功）。
            if (act.get("kind") in ("装备牌", "效果牌")
                    and _hit(ctx, shot, ROI_DESIGNATE, "指定")):
                logger.info("拖拽校验: %s 已举起（复查时才认出横幅）—— 立刻补点目标", piece)
                return LIFTED
            extractor._extract_hand(ctx, shot, probe)
        except Exception:
            logger.exception("拖拽校验重读手牌出错，本次不给回执")
            return None
        if not probe.hand:
            # **手牌整条读成空本身就很可疑**：正常情况下手里至少有枫叶舞。
            # 留一帧证据，下次就知道这个状态在画面上长什么样（是牌举在半空？
            # 是画面在做动画？还是手牌条被什么盖住了）。
            _save_evidence(shot, f"hand_empty_{piece}")
            logger.info("拖拽校验: 手牌读成空（重读两次都是），%s 本次不给回执"
                        "（证据帧已存）", piece)
            return None
    after = probe.hand.count(piece)
    if after < before and act.get("merge"):
        # **合成被拒时牌会弹回手里**，而上面那一眼可能正好在牌飞回去的路上（少了一张）。
        # 2026-09-23：弈星往 (1027,572) "合成成功" 15 次、手里 3 张一张没少。
        # 多等一会儿再看一次：数量回到原来 = 弹回来了 = 失败。
        time.sleep(0.8)
        try:
            again = GameState()
            extractor._extract_hand(ctx, screencap(controller), again)
            if again.hand and again.hand.count(piece) >= before                     and len(again.hand) <= len(hand_before):
                logger.info("拖拽校验: %s 合成后又弹回手里了（%d 张没少）—— 判失败", piece, before)
                return False
        except Exception:
            logger.exception("合成复查出错，按第一眼的结果算")
    if after < before:
        return True
    # **手牌变多了就判不了**。判据是"这张牌少了一张"，而牌是会**凭空多出来的**
    # （登场效果/拍卖/回合奖励随时塞新牌进来）。2026-09-23 实战抓到：
    # 干将莫邪 拖上场那一瞬手牌从 4 张涨到 7 张，其中又来了一张同名的 ——
    # 数量没减少，于是**明明成功了却报失败**（人口同时从 3 涨到 4，是硬证据）。
    # 后果很贵：拉黑一个好格子、撤销正确的记录、再重试，
    # 今晚大量"拖拽失败"很可能都是这么来的。
    # 判不了比判错好：交给"试够次数就停"那条止损兜底。
    if len(probe.hand) > len(hand_before):
        logger.info("拖拽校验: 拖完手牌反而多了(%d->%d，多半是登场/拍卖又塞了牌)，"
                    "%s 这次判不了", len(hand_before), len(probe.hand), piece)
        return None
    _pos = lambda h: [i for i, n in enumerate(h or []) if n == piece]          # noqa: E731
    if len(probe.hand) == len(hand_before or []) and _pos(probe.hand) != _pos(hand_before):
        # **同名张数没少、但手牌排列 / 别的牌变了 = 有牌进出**（10-21 审查，第 850 局：古币打守约后场上效果又送来一张古币，
        # 张数不变被判"没生效"两次 → 守约被拉黑、之后 6 张古币全给了苏烈 / 铠）。判不了，交给止损兜底
        logger.info("拖拽校验: %s 张数没变但手牌变了（%s → %s，多半是用完又得到新牌），这次判不了",
                    piece, list(hand_before or [])[:8], list(probe.hand)[:8])
        return None
    return False


def shop_is_open(ctx, controller) -> bool:
    """当场看一眼商店面板在不在（和感知层用的是同一条判据）。"""
    from agent.perception.extractors import ROI_FREE_REFRESH, ROI_SHOP_PROB
    # 面板有开合动画，**查早了会看到旧画面**：2026-09-23 实测点完"开店"立刻查，
    # 报的是"还关着"，而下一帧商店已经开了。executor 已经等了 0.9s，这里再补一点。
    time.sleep(0.5)
    shot = screencap(controller)
    return bool(_hit(ctx, shot, ROI_SHOP_PROB, "当前概率")
                or _hit(ctx, shot, ROI_FREE_REFRESH, "刷新"))


UNKNOWN_SHOT_EVERY = 8.0   # unknown 挂着不动时每隔几秒存一帧证据
UNKNOWN_SHOTS_MAX = 40     # 每局最多存几张（第 43 局 12 张不够：真正卡住时已经存满、没留下证据）
EQUIP_SETTLE_MAX = 0.6   # 等弹窗展开动画停下（连续两次探测位置一致）最多等多久
EQUIP_TAP_DELAY = 0.25   # 停下后再等一拍才点


def grab_equip_popup(ctx, controller, executor, extractor, rules, run_dir,
                     center, base_equip, target_piece, role,
                     before_count: int, budget: float = 3.0, expect: bool = True) -> bool:
    """**拖完立刻抢弹窗**。装备四选一只在画面上停一两秒，抢不到就白拖。

    这是 2026-09-23 靠两张证据帧才定位到的：
      `hand_empty_*.png` 那一帧弹窗**明明开着**（4 个选项围着英雄），
      2 秒后的 `equip_no_popup_*.png` 里只剩下英雄详情面板，弹窗没了。
    而我们在拖完之后**先去做"读手牌回执"**（截图 + OCR + sleep 0.4s + 再截图），
    等回头进主循环找弹窗时早就过期了 —— 于是一路"本帧没认出弹窗"直到超时。
    回执什么时候做都行，弹窗**过期不候**，所以顺序必须倒过来。

    顺带解释了另一个谜：拖装备之后手牌总是"读成空"——
    因为这一下同时**打开了英雄详情面板**，面板把手牌条盖住了。

    `expect=False`：拖的是**成品装备**，本来不该有弹窗 —— 只是短暂看一眼防"拖错牌"
    （手牌挪位后抓起来的其实是基础装备）。这时要求**连续两次探测位置一致**才点
    （成品装备没弹窗是常态，误把棋盘上的徽章认成选项、点下去就是乱点），
    等不到也不报错、不存证据帧。
    """
    deadline = time.time() + budget
    tries = 0
    probe_shot = None
    designated = False
    while time.time() < deadline:
        tries += 1
        shot = screencap(controller)
        if tries == 3:
            # 抓到弹窗的历次都在第 2~3 次探测（0.2~0.4s）。等不到时**存这一帧**：
            # 原来只在 3 秒超时后存图，那时弹窗早关了，看不出它当时长什么样
            # （2026-09-23 大棒贴到棋盘右边缘那次，就是这么查不下去的）。
            probe_shot = shot
        if tries == 2 and not designated and _hit(ctx, shot, ROI_DESIGNATE, "指定"):
            # **两段式：牌举起来了，游戏在等"点一下目标英雄"**（2026-09-23 第十四局证据帧：
            # 顶部「请指定英雄使用」—— 落点格子上没人（账本把曜的位置记错了一格），牌一直举着到超时）。
            # 点离落点最近的那个单位（按等级徽章），然后照常等四选一弹窗。
            from agent.perception.extractors import detect_unit_anchors
            anchors = detect_unit_anchors(shot)
            designated = True
            if anchors:
                cell = min((tuple(int(v) for v in k.split(",")) for k in anchors),
                           key=lambda c: (c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2)
                logger.warning("【装备四选一】牌举着在等指定目标，点离落点最近的单位 %s（原落点 %s）",
                               cell, center)
                executor._tap(*board_drop_point(cell))
                center = cell
                time.sleep(0.3)
                continue
        global _CC_NEED
        from agent.decision import hero_traits as _htc
        _CC_NEED = bool(role.startswith("carry:") and target_piece == rules._carry_piece
                        and not (set(rules._worn.get(target_piece) or ()) & (_htc.carry_cc_items(target_piece) | {"机器头"}))
                        and _htc.CC_IMMUNE_MAPLE not in (getattr(rules, "_lord_boosts", set()) or set()))
        point = detect_equip_popup(ctx, shot, extractor, center, role,
                                   run_dir, base_equip, worn=rules._worn.get(target_piece) or ())
        if not point:
            # **等不到弹窗时，先看看是不是有别的模态盖住了棋盘**。
            # 2026-09-23 实战证据帧：拖装备那会儿屏幕上挂着一个**卡牌三选一**
            # （明世隐/赵云/貂蝉三张大卡盖住上半屏），棋盘根本点不到，
            # 于是白白空转 3 秒、探测 15 次。
            # 认出来就立刻退出，让主循环去把那个模态处理掉，装备下回合再贴
            # （它还在手里，不会丢）。
            if tries % 5 == 0:
                from agent.perception.state import GameState as _GS
                probe = _GS()
                extractor._detect_card_pick(ctx, shot, probe)
                if probe.card_pick:
                    logger.warning("【装备四选一】棋盘被「卡牌N选1」盖住了（%s）—— "
                                   "先不等弹窗，交给主循环处理模态",
                                   [o["name"] for o in probe.card_options])
                    return False
            time.sleep(0.15)
            continue
        # **别抓到就点**：弹窗有展开动画，选项还在飞的时候点下去，这一下可能只是把弹窗关掉、
        # 没算作选择 —— 基础装备照样穿上、件数照样 +1，复查分辨不出来
        # （2026-09-23 亚连：日志 ✅，画面上是失败的；可能是点击太快了）。
        # 所以等**连续两次探测位置一致**（动画停了），再多等一拍，并且**按住**再抬起
        # （这游戏对输入时长敏感，能量圆盘就不吃瞬时点击）。
        settle_until = time.time() + EQUIP_SETTLE_MAX
        confirmed = False
        while time.time() < settle_until:
            time.sleep(0.12)
            again = detect_equip_popup(ctx, screencap(controller), extractor, center, role,
                                       None, base_equip, worn=rules._worn.get(target_piece) or ())     # run_dir=None：复核这几帧不存证据图
            if not again:
                break
            if abs(again[0] - point[0]) <= 4 and abs(again[1] - point[1]) <= 4:
                point = again
                confirmed = True
                break
            point = again
        if not expect and not confirmed:
            continue                      # 成品装备：没复核上就当没看见，绝不乱点
        if not expect:
            logger.warning("【装备四选一】拖的是成品 %s，却弹出了四选一 —— "
                           "抓起来的其实是一件基础装备（手牌挪位了），照样选掉", base_equip)
        time.sleep(EQUIP_TAP_DELAY)
        logger.info("【装备四选一】抢到弹窗（第 %d 次探测，用时 %.1fs）：按住点击 %s，"
                    "基础装备 %s 穿给 %s(%s)", tries, budget - (deadline - time.time()),
                    point, base_equip, target_piece or f"{center} 那一格", role)
        if f"equip_before_tap_{base_equip}" not in _EVIDENCE_SEEN:
            _save_evidence(screencap(controller), f"equip_before_tap_{base_equip}")
        executor._tap_hold(*point, hold=0.15)
        forged_name = _LAST_FORGED
        ok = _equip_forged(ctx, controller, extractor, center, base_equip, before_count)
        if not ok:
            logger.warning("【装备四选一】单击没铸造出来 —— 改用拖拽：从 %s 拖到 %s",
                           center, point)
            executor._drag(center, point)
            ok = _equip_forged(ctx, controller, extractor, center, base_equip,
                               before_count)
            if ok:
                logger.warning("【装备四选一】✅ **拖拽才生效**（单击无效）")
        else:
            logger.info("【装备四选一】✅ 单击生效（弹窗已消失，%s）",
                        target_piece or f"{center} 那一格")
        if ok:
            rules.note_worn(target_piece, forged_name)
            if rules._hand_book[base_equip] > 0:
                rules._hand_book[base_equip] -= 1      # 手牌账：基础装备用掉了（这条路径不走 confirm_drag）
        # 点完 1 秒再存一张：看单位身上挂的是成品还是基础装备（复查分不出来，留给人/赛后核对）
        # 同一件基础装备每局只存一次 —— 已经存过就不再白等 0.6 秒（2026-10-14 提速：每件装备省 ~1 秒）
        if f"equip_after_{base_equip}" not in _EVIDENCE_SEEN:
            time.sleep(0.6)
            _save_evidence(screencap(controller), f"equip_after_{base_equip}")
        if not ok:
            _save_evidence(screencap(controller), f"equip_stuck_{base_equip}")
            logger.error("【装备四选一】❌ 单击和拖拽都没把装备铸造上去，证据帧已存")
        # 弹窗确实出现过 —— 牌已经离手了，不该再按"拖失败"去拉黑落点
        return True
    if not expect:
        return False
    logger.warning("【装备四选一】❌ %.1fs 内（探测 %d 次）没等到弹窗 —— "
                   "基础装备 %s 穿给 %s，这件装备**没有铸造**",
                   budget, tries, base_equip, target_piece)
    _save_evidence(probe_shot if probe_shot is not None else screencap(controller),
                   f"equip_no_popup_{base_equip}")
    return False


def locate_bid_button(ctx, controller, run_dir, tag: str):
    """点中拍品之后，OCR 找"出价"按钮，返回其坐标；没找到返回 None。

    出价按钮只有在拍品被选中后才出现，所以"找不到"本身就是有用信息：
    要么没点中拍品，要么这个 UI 版本按钮位置变了。写死坐标已被证明会点空。
    """
    import cv2
    # 按钮是**圆形按钮**，就在右下能量圆盘左边（实测 (1075,645)，圆心附近写"出价"、
    # 下面一行是本次出价数字）。它带淡入动画，选中后立刻截图经常还没画出来——
    # 所以要重试几次；只截一张就放弃，整场拍卖一次都出不了价
    # （2026-09-22 实战："点了拍品但没找到'出价'按钮"）。
    image = None
    for attempt in range(3):
        image = screencap(controller)
        # **不要用全屏兜底**：2026-09-22 第二局实测，全屏搜到屏幕顶部 (639,18) 的
        # 另一处"出价"字样（出价榜/提示条），点下去是空点，还白白让出了一次出价机会。
        # 真按钮固定在右下角能量圆盘左边（实测 (1075,636)/(1076,636)），
        # 只在底部那一条带里找，找不到就认账放弃，比点错强。
        for roi in ((950, 580, 330, 140), (600, 540, 680, 180)):
            for text, box in _ocr_boxes(ctx, image, roi):
                if "出价" in text:
                    bx, by = box[0] + box[2] // 2, box[1] + box[3] // 2
                    # **把实际点击点画在证据帧上**：2026-09-22 出价按钮
                    # 位置点击有问题，但光看截图分不清是"没找到按钮"还是"找到了
                    # 但点偏了"——文本框中心不一定等于圆形按钮的中心。
                    # 画出来下次一眼就能判定，不用靠猜。
                    shot = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(shot, (box[0], box[1]),
                                  (box[0] + box[2], box[1] + box[3]), (0, 255, 255), 1)
                    cv2.circle(shot, (bx, by), 10, (0, 0, 255), 2)
                    cv2.drawMarker(shot, (bx, by), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                    cv2.imwrite(str(run_dir / f"auction_{tag}_ok.png"), shot)
                    logger.info("出价按钮：文本框 %s -> 点击 (%d,%d)", box, bx, by)
                    return (bx, by)
        time.sleep(0.25)
    if image is not None:      # 三次都没有 → 存证据帧，下次照着它调 ROI
        cv2.imwrite(str(run_dir / f"auction_{tag}.png"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return None


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live")

LOG_DIR = PROJECT_DIR / "data" / "logs"


def _limit_cpu_threads() -> None:
    """2026-10-12（CPU 占用时低时高，会突然冲到 70%）：OpenCV / torch 默认按核数开线程（24 核），
    一帧里几个模型同时跑就把整机冲满。agent 本身的并发已经够用，这里统一限住。"""
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(4)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
    except Exception:                                   # noqa: BLE001
        pass


def main() -> None:
    _limit_cpu_threads()
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="真实执行决策动作")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--perf", action="store_true",
                    help="逐帧打印耗时分解（截图/阶段识别/各项提取 + OCR 次数），用于定位循环瓶颈")
    ap.add_argument("--enter-match", action="store_true",
                    help="允许从大厅/模式页导航进入**一局**对局（结算后仍然停，不连开）")
    ap.add_argument("--force-route", default="",
                    help="开局就锁这条路线（lineups.json / custom_lineups.json 的 key，如 user_ganjiang = 用户指定的干将养成）")
    ap.add_argument("--eco-takeover", action="store_true",
                    help="经济节奏（升级 / 停级刷牌 / 攒钱）交给局面网络 eco 题执行（默认只做影子题）")
    ap.add_argument("--carry-tiers", default="2,3",
                    help="只走这些品阶主C 的官方路线、主C 评分向它们倾斜（默认 2~3 阶主C）；空串 = 不限")
    ap.add_argument("--auto-chain", action="store_true",
                    help="允许自动连开下一局（默认关闭：一局结束即停，等人工）")
    ap.add_argument("--no-laya", action="store_true",
                    help="强制禁用 Laya（默认：本地有权重才启用）")
    ap.add_argument("--laya-ckpt", default=None,
                    help="蒸馏后的 Laya checkpoint 目录（data/laya/runs/<run>/checkpoint）。给了就让学生上场做决定")
    ap.add_argument("--laya-kinds", default="card_pick,talent,lord_boost,shop",
                    help="学生接管哪些决策（逗号分隔）")
    # 默认 0：商店一律按 Laya 执行（只受硬约束和防攒钱安全阀限制）。2026-09-24：这种不应该退回规则，
    # 多让 Laya 发挥才能收集数据；规则的选择 Laya 只给 10%，比 Laya 的最优解低太多（百里守约 27% vs 规则曜 10%，因 0.27<0.3 被退回）
    ap.add_argument("--laya-shop-conf", type=float, default=0.0,
                    help="商店决策：学生最高概率低于它就回退规则")
    ap.add_argument("--laya-download", action="store_true",
                    help="本地没权重时允许现场下载（约 1.7G，会堵住启动）")
    args = ap.parse_args()

    PERF.enabled = args.perf
    controller = connect_adb_controller()
    tasker = create_tasker(controller, PROJECT_DIR / "assets")
    ctx = FrameContext(tasker)
    extractor = StateExtractor(PROJECT_DIR / "assets" / "image")
    # OCR 引擎在进对局前加载好：TensorRT 读缓存 ~1.5s，**第一次要建引擎 ~8 分钟**，
    # 懒加载会让第一帧卡死。放在 Laya 之前 —— 先让 onnxruntime 登记好它的 CUDA/cuDNN DLL。
    from agent.perception import onnx_ocr
    if onnx_ocr.enabled():
        _t = time.time()
        _ocr = onnx_ocr.OnnxOcr.get()
        import numpy as _np
        _ocr.boxes(_np.zeros((720, 1280, 3), _np.uint8), (0, 0, 1280, 720))   # 预热
        logger.info("OCR 引擎：%s，加载 %.1fs", _ocr.describe(), time.time() - _t)
    else:
        logger.info("OCR 引擎：MaaFramework 自带（WZWXQ_OCR=maa 或没装 onnxruntime）")
    # 视觉单位检测（D-FINE，TensorRT FP16）同样先预热：第一次建引擎要几分钟，缓存在 .trt_cache/
    try:
        from agent.perception import vision_units as _vis
        _t = time.time()
        logger.info("视觉单位检测（D-FINE）：%s，加载 %.1fs", _vis.warmup() or "未启用", time.time() - _t)
    except Exception:  # noqa: BLE001
        logger.exception("视觉单位检测预热失败（忽略，只用徽章）")
    rules = RuleEngine(PROJECT_DIR / "data" / "chess_pieces.json",
                       cards_db_path=PROJECT_DIR / "data" / "cards.json")
    if args.execute:
        # **实战一开始就是"有拖拽回执"模式**（2026-09-24 match14）：`_verified` 原来要等第一次拖拽回执才置位，
        # 进程重启后的那段时间里，给干跑准备的开环猜测（"同一个合成决策出了 3 次 = 合成失败 = 觉醒回手"）生效，
        # 把"手牌里找不到那张芈月、这一帧不拖"连着 3 帧当成合成失败，删掉了场上确认过的芈月，下个回合才又合成上去。
        rules._verified = True
    # 官网推荐阵容库（tools/fetch_lineups.py 产出）：有就用它替换手工 comps
    rules.load_lineups(PROJECT_DIR / "data" / "lineups.json", lord="阿离")
    from agent.decision import hero_traits as _ht
    _ht.USE_HERO_ITEM_PREF = True          # 攻略点名的主C 装备（守约：无尽/炽热/名刀），只在实战打开
    if args.eco_takeover:
        RuleEngine.ECO_TAKEOVER = True
        logger.info("经济节奏交给局面网络（eco 题接管）")
    if args.carry_tiers:
        from agent.decision import routes as _routes
        _routes.CARRY_TIERS = tuple(int(x) for x in args.carry_tiers.split(",") if x.strip())
        _routes.EXCLUDE_CARRIES = frozenset({"公孙离"})     # 要"夺取"合成站位，操作不了（2026-09-30）
        logger.info("主C 品阶定调：只走 %s 阶主C 的路线", _routes.CARRY_TIERS)
    if args.force_route:
        _fr = [k.strip() for k in args.force_route.split(",") if k.strip()]
        rules.forced_route, rules.forced_backups = _fr[0], tuple(_fr[1:])
        args.force_route = _fr[0]
        logger.info("用户指定路线：%s（开局锁定，主C 不换）", args.force_route)
    # 本地 Laya 可选：装了就用（购买类决策），没装自动回退纯规则
    from agent.decision.laya_engine import HybridEngine, LayaEngine
    laya_engine = LayaEngine(confidence_threshold=0.85,
                             enabled=not args.no_laya and not args.laya_ckpt,   # 有学生就不再加载零样本 Laya
                             allow_download=args.laya_download)
    # 冷启动（加载 checkpoint 到 GPU）要 15~30 秒。懒加载会让第一个 prepare 帧
    # 卡死整个主循环——实战实测堵了 15.4 秒，那期间 agent 完全瞎。在进对局前付掉。
    if laya_engine.enabled:
        _t = time.time()
        laya_engine._get_router()
        logger.info("Laya 预热完成 %.1fs（advisory_only=%s）",
                    time.time() - _t, laya_engine.advisory_only)
    policy = None
    if args.laya_ckpt and not args.no_laya:
        from agent.decision.laya_policy import LayaPolicy
        policy = LayaPolicy(args.laya_ckpt, rules, None,       # run_dir 下面建好后再设
                            shop_min_conf=args.laya_shop_conf, kinds=tuple(args.laya_kinds.split(",")))
    hybrid = HybridEngine(rules, laya_engine, policy=policy)
    executor = ActionExecutor(controller) if args.execute else None
    if executor is not None:
        from agent.perception.extractors import pop_color as _pop_color

        def _pop_red() -> bool:
            import cv2
            shot = screencap(controller)
            red = _pop_color(shot) == "red"
            if red and _EVIDENCE_DIR is not None:
                _imwrite(_EVIDENCE_DIR / f"pop_red_{int(time.time())}.png", cv2.cvtColor(shot, cv2.COLOR_RGB2BGR))
            return red
        executor.drop_guard = _pop_red

        def _grab_wrong(action) -> bool:
            """牌举在半空时确认抓的是不是这张：想拖的那张还在手里、少的是另一张 = 抓错了（拖回去）。"""
            import cv2
            from agent.perception.state import GameState as _GS
            pre = _GRAB_CTX.pop("pre_shot", None)
            # 只用**这一次**拖前核对留下的那一帧（5 秒内）；旧帧和现在的手牌对不上会误报
            if pre is None or not action.get("piece") or time.time() - _GRAB_CTX.get("t", 0) > 5:
                return False
            before = _GS(phase="prepare")
            extractor._extract_hand(ctx, pre, before)
            mid_shot = screencap(controller)
            mid = _GS(phase="prepare")
            extractor._extract_hand(ctx, mid_shot, mid)
            other = _wrong_grab(before.hand, mid.hand, action["piece"])
            # **影子模式：只记录、不拦截**（2026-09-24 第二轮第 3 局实测误判）：拖拽中被拖的那张牌**仍留在手牌条里**
            # （抬高一截、放大显示描述），会盖住隔壁那张 —— "手牌差异"在半空时判不准，牛魔连着被错拦 3 次。
            # 先每次存"按下前/半空"两帧，拖完之后的抓错牌检测给出真值，攒够样本再离线调"找出被抬起的那张"，准了再拦截。
            ts = int(time.time() * 1000)
            if _EVIDENCE_DIR is not None and _GRAB_CTX.get("saved", 0) < GRAB_SHADOW_MAX:
                _GRAB_CTX["saved"] = _GRAB_CTX.get("saved", 0) + 1
                tag = f"{action['piece']}_{'差异' + other if other else '无差异'}_{ts}"
                _imwrite(_EVIDENCE_DIR / f"grab_shadow_pre_{tag}.png", cv2.cvtColor(pre, cv2.COLOR_RGB2BGR))
                _imwrite(_EVIDENCE_DIR / f"grab_shadow_mid_{tag}.png", cv2.cvtColor(mid_shot, cv2.COLOR_RGB2BGR))
                _GRAB_CTX["last_tag"] = tag
            return False
        executor.grab_guard = _grab_wrong

        def _sell_zone_up(piece: str = "") -> bool:
            """两侧「+N 出售」紫色竖条（BGR≈75,52,64）占比：没拖 0~0.04、拿起能卖的牌 ≈0.69（839 局前 600 帧离线统计）。
            拿起的是不能卖的牌（装备 / 觉醒合成）时显示灰色「禁止出售」→ 也判 False，正好不卖。"""
            import cv2
            import numpy as np
            shot = cv2.cvtColor(screencap(controller), cv2.COLOR_RGB2BGR)
            h, w = shot.shape[:2]
            sy, sx = h / 720.0, w / 1280.0
            ref = np.array([75, 52, 64], np.int16)
            fr = []
            for x0, x1 in ((4, 68), (1212, 1276)):
                r = shot[int(480 * sy):int(700 * sy), int(x0 * sx):int(x1 * sx)].astype(np.int16)
                fr.append(float((np.abs(r - ref).max(axis=2) < 22).mean()))
            ok = max(fr) >= 0.25
            nm = ""
            if ok and piece:
                # 放大详情（左上角）写着被拿起的是谁：读得出且对不上 = 拿起的不是这张（多半是棋盘上的人）
                try:
                    from agent.perception import extractors as _ex
                    raw = _ex._ocr(ctx, cv2.resize(shot, (1280, 720))[:, :, ::-1].copy(), (40, 215, 220, 40))
                    nm = extractor.resolve_name(raw, keep_raw=False) or ""
                except Exception:                                   # noqa: BLE001
                    nm = ""
                if nm and nm != piece and piece not in nm and nm not in piece:
                    logger.warning("卖牌：拿起的是 %s，不是 %s → 放回", nm, piece)
                    ok = False
            if not ok and _EVIDENCE_DIR is not None:
                _imwrite(_EVIDENCE_DIR / f"sell_no_zone_{piece}_{nm}_{int(time.time() * 1000)}.png", shot)
            return ok
        executor.sell_grab_ok = _sell_zone_up
        def _ui_settled() -> bool:
            # 选完 N 选 1 / 天赋后是不是已经回到正常画面：「准备/开战」按钮看得见、N 选 1 不在了
            from agent.perception.extractors import ROI_READY_BTN
            from agent.perception.state import GameState as _GS
            shot = screencap(controller)
            if not _hit(ctx, shot, ROI_READY_BTN, "准备|开战"):
                return False
            probe = _GS()
            extractor._detect_card_pick(ctx, shot, probe)
            return not probe.card_pick
        executor.ui_settled = _ui_settled

    # **接着上一个进程的记忆**：最近的 rules_state.json 是 3 分钟内写的，就当是同一局
    # （真开了新局，进大厅/选棋手时 reset_match 会清空，不会串局）
    try:
        prev = max(LOG_DIR.glob("live_*/rules_state.json"), key=lambda f: f.stat().st_mtime, default=None)
        # 60 分钟（原 3 分钟）：2026-09-30 游戏卡死重连回原局超过 3 分钟，没读回"枫叶舞绑在苏烈"，换绑做不了。
        # 不会串局：新开一局进大厅/选棋手时 reset_match 会清空，指定路线也只在同一首选时恢复（load_state）
        if prev is not None and time.time() - prev.stat().st_mtime < 3600:
            _prev_d = json.loads(prev.read_text(encoding="utf-8"))
            rules.load_state(_prev_d)
            _hpm = _prev_d.get("_perc_hp")
            if _hpm and 0 < int(_hpm[0] or 0) <= 60:
                # 血量记忆也接上（10-21 审查：851 / 852 局重启后首读 2 / 4 血被当"本局首读"直接采信，规则按残血打）
                extractor._last_hp, extractor._hp_before, extractor._hp_round = int(_hpm[0]), int(_hpm[1] or _hpm[0]), int(_hpm[2] or 0)
                extractor._hp_seen = True
                logger.info("接上上个进程的血量记忆：%s（第 %s 回合认定）", _hpm[0], _hpm[2])
    except Exception:                                               # noqa: BLE001
        logger.exception("读上一个进程的记忆失败（忽略，从零开始）")
    hand_miss_saved: dict = {}
    run_dir = LOG_DIR / f"live_{int(time.time())}"
    _RUN_DIR[0] = run_dir
    global _EVIDENCE_DIR
    _EVIDENCE_DIR = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if policy is not None:
        policy.set_run_dir(run_dir)
    log_fp = open(run_dir / "states.jsonl", "a", encoding="utf-8")
    # 点击日志 run_dir/clicks.jsonl + 兜底点位 gate 现截帧用的 probe
    tap_gate.configure(run_dir, probe=lambda: screencap(controller), round_getter=lambda: _CUR_ROUND[0])
    import atexit
    atexit.register(tap_gate.flush)
    logger.info("模式=%s 输出=%s pid=%d", "EXECUTE" if args.execute else "DRY-RUN",
                run_dir, __import__("os").getpid())

    last_phase = ""
    unknown_shot_ts = time.time()
    unknown_shots = 0
    ledger_len = 0
    decisions_len = 0
    unknown_since: float | None = None
    blank_escapes = 0           # 本局已用掉几次点空白逃生
    known_phase = ""            # 最近一次认出来的（非 unknown）阶段
    prep_escapes: dict = {}     # 回合 -> 准备阶段里点空地逃生的次数
    close_big_escapes: dict = {}   # 回合 -> unknown 兜底里点场馆背景 (400,60) 关放大详情的次数（2026-10-10 第 524 局）
    pending_target: tuple | None = None   # 上一次拖拽的落点（补点用）
    equip_popup_until: float = 0
    last_action_sig: str = ""
    last_action_ts: float = 0
    equip_target: tuple[int, int] | None = None
    equip_base: str | None = None
    equip_done: bool = False
    equip_target_piece: str | None = None   # 这件装备要穿给谁（打分按他的职业，不是主C）
    equip_before_count: int = 0             # 拖装备**之前**那个格子上有几件装备
    hero_selected: bool = False
    match_over_logged: bool = False
    hero_panel_tries = [0, 0]        # [回合, 本回合关英雄详情面板的次数]
    played_a_match: bool = False     # 这个进程有没有真的进过一局（见 settle_final 分支）
    shop_fail_round: int = -1        # 已经为"关店失败"存过证据帧的回合
    last_prepare_state = None        # 最后一帧准备阶段的状态（结算页上手牌读不到了）
    # **N 选 K（K≥2）连续选**（2026-09-26：有些效果牌是 5 选 2 或 5 选 3，读卡面决定，例如广纳良才）：
    # 打出卡面写着"N选K"的牌 → 记下要选 K 张；每选一张就从后续候选里去掉；没选够时界面认不出就留证据帧、找确认按钮。
    multi_pick = {"k": 0, "picked": [], "ts": 0.0, "card": "", "shots": 0}
    prep_clock: dict = {}            # 准备阶段倒计时：{r: 回合, v: 秒, t: 读到的时刻}（觉醒时机判断用）
    while True:
        try:
            _t_cap = time.perf_counter()
            image = screencap(controller)
            cap_ms = (time.perf_counter() - _t_cap) * 1000
            _t_ext = time.perf_counter()
            extractor.hand_hint = rules.hand_book_names()
            extractor.pop_bonus = rules.pop_bonus        # 天赋加的人口上限（感知层据此放宽合法上限）
            try:                                          # 手牌账里有、上一帧没认出来的牌（立绘闭集核对，第 678 局捕能手套）
                import collections as _co
                _seen_h = _co.Counter(sl.get("name") for sl in (getattr(state, "hand_slots", None) or []) if sl.get("name"))
                extractor.hand_book_hint = {n for n, c in (rules._hand_book - _seen_h).items() if c > 0
                                            and (rules.pieces.get(n) or {}).get("category") != "棋手技能牌"}
            except Exception:                             # noqa: BLE001
                extractor.hand_book_hint = set()
            state = extractor.extract(ctx, image)
            if extractor.pop_bonus_revoked:      # 白色满员证伪了画面认的额外人口（第 644 局）→ 退回记账值
                rules.pop_bonus = rules.__dict__.get("pop_bonus_ledger", 0)
                extractor.pop_bonus_revoked = False
            rules.pop_bonus = max(rules.pop_bonus, extractor.pop_bonus_seen)   # 画面认出的额外人口（记账漏了的兜底）
            _screen_cls_shadow(state, image, run_dir)
            state = _screen_cls_grant(state, image, ctx, extractor)
            _CUR_ROUND[0] = state.round_no or 0
            tap_gate.note_frame(*(state.screen_cls[:2] if state.screen_cls else (None, None)))   # 补写点击后的界面
            rules.pool_left = dict(POOL_LEFT)            # 点开看读到的公共牌池剩余张数
            rules.sync_hand_book(state)
            rules.fill_hand_from_book(state)
            if (state.phase == "prepare" and state.hand_count and not state.shop_open
                    and state.hand_count[0] - len(state.hand_slots or []) >= 3):
                # 2026-10-15 第 632 局 R15：手牌 13/15 只认出 1 张、整回合空转。存证据帧（给手牌条模型补多卡样本），每局最多 20 张
                _hm = hand_miss_saved.setdefault("n", 0)
                if _hm < 20 and time.time() - hand_miss_saved.get("ts", 0) > 10:
                    hand_miss_saved.update(n=_hm + 1, ts=time.time())
                    logger.warning("手牌认漏：左下角 %d/%d，只认出 %d 张 → 存证据帧",
                                   state.hand_count[0], state.hand_count[1], len(state.hand_slots or []))
                    import cv2 as _cv2
                    _imwrite(run_dir / f"hand_miss_{state.hand_count[0]}_{int(time.time())}.png",
                             _cv2.cvtColor(image, _cv2.COLOR_RGB2BGR))
            ext_ms = (time.perf_counter() - _t_ext) * 1000
            if multi_pick["k"] and time.time() - multi_pick["ts"] > 25:
                multi_pick.update(k=0, picked=[])                  # 超时：这一轮多选结束了
            if (state.card_pick and len(state.card_pick_quota) == 2 and state.card_pick_quota[1] >= 2
                    and not multi_pick["k"]):
                # 界面自己写着「请选择1/2张卡牌」—— 不管是哪张牌/天赋触发的，都按多选处理
                # （2026-09-30 第 37 局：回合开始弹的英雄四选二，只选了一张就停）
                multi_pick.update(k=state.card_pick_quota[1], picked=[], ts=time.time(),
                                  card="界面横幅", shots=0)
                logger.info("N选多：界面写着要选 %d 张", state.card_pick_quota[1])
            if multi_pick["k"] and len(multi_pick["picked"]) < multi_pick["k"]:
                extractor.card_pick_hint_until = time.time() + 3
                if state.card_pick and state.card_options:
                    left = [o for o in state.card_options if o["name"] not in multi_pick["picked"]]
                    if left and len(left) < len(state.card_options):
                        logger.info("【%s 选 %d 张】已选 %s，剩下候选 %s", multi_pick["card"], multi_pick["k"],
                                    multi_pick["picked"], [o["name"] for o in left])
                        state.card_options = left
                elif state.phase == "unknown" and executor:
                    if multi_pick["shots"] < 6:
                        multi_pick["shots"] += 1
                        import cv2
                        _imwrite(run_dir / f"multi_pick_{len(multi_pick['picked'])}of{multi_pick['k']}_{int(time.time())}.png",
                                 cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    btn = next(((t, b) for t, b in _ocr_boxes(ctx, image, (300, 420, 700, 300))
                                if len(t) <= 4 and re.search(r"确定|确认|完成|选好", t)), None)
                    if btn:
                        bx, by = btn[1][0] + btn[1][2] // 2, btn[1][1] + btn[1][3] // 2
                        logger.info("【%s 选 %d 张】已选 %s，界面上有「%s」按钮 —— 点 (%d,%d)",
                                    multi_pick["card"], multi_pick["k"], multi_pick["picked"], btn[0], bx, by)
                        executor._tap(bx, by)
                        time.sleep(0.5)

            if state.phase != last_phase:
                logger.info("=== 阶段切换: %s -> %s (回合%d) ===",
                            last_phase, state.phase, state.round_no)
                # 阶段切换时留证据帧，便于事后校准
                import cv2
                cv2.imwrite(str(run_dir / f"phase_{state.phase}_{int(time.time())}.png"),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                last_phase = state.phase
                unknown_shot_ts = time.time()
            elif (state.phase == "unknown" and unknown_shots < UNKNOWN_SHOTS_MAX
                    and time.time() - unknown_shot_ts >= UNKNOWN_SHOT_EVERY):
                # **认不出的界面要多留几帧**：阶段切换那一帧往往是过场动画（卡牌还在飞），
                # 真正挂着不动的那个界面一张都没存 —— 第七局英雄三选一卡死，
                # 翻遍证据帧都找不到它，只能去历史对局里捞同类帧（2026-09-23）。
                import cv2
                _imwrite(run_dir / f"unknown_{int(time.time())}.png",
                         cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                unknown_shots += 1
                unknown_shot_ts = time.time()
            if state.phase == "prepare":
                last_prepare_state = state   # 自评报告要用它看"还剩几张没打出去"
            if state.phase in ("hero_pick", "prepare", "battle", "auction"):
                # 真的进到对局里了。settle_final 分支靠这个标记区分
                # "启动时就在结算页"（该点再来一局）和"这一局刚打完"（该停手）。
                played_a_match = True
                match_over_logged = False        # 连开模式：下一局的名次要重新记

            # 装备四选一弹窗期间**不要信棋盘占用观测**：弹窗是 4 个深色圆环绕英雄，
            # 几何上和等级徽章分不开。这里是唯一确定知道"弹窗在不在"的地方
            # （感知层只看得到圆，猜不出来），所以在这里把观测作废。
            if equip_popup_until and time.time() < equip_popup_until:
                state.board_seen = False
                state.board_occupied = []

            if state.phase == "prepare":
                PREP_SEEN.setdefault(state.round_no, time.time())
                rules._prep_left = prep_left(state.round_no)
            short_on_time = state.phase == "prepare" and prep_left(state.round_no) < PREP_SPARE_S
            # 账本和画面对不上：点开场上单位看是谁（2026-09-23 起）
            if (state.phase == "prepare" and not state.shop_open and executor is not None and not short_on_time
                    and not state.hero_panel_open and _BOARD_SEEN["count"] < 10 ** 6):
                try:
                    # 枫叶舞宿主不在账本里（卡面灰）：把它最后的格子和"认不出是谁"的格子加进复查（第 214 局马超）
                    _ms = rules.maple_search_cells(state)
                    if _ms and _BOARD_SEEN.get("maple_round") != state.round_no:
                        _BOARD_SEEN["maple_round"] = state.round_no
                        RECHECK_CELLS.update(_ms)
                        logger.info("【看场】枫叶舞宿主 %s 不在账本里，去这些格子找：%s", rules._maple_bound, _ms)
                    inspect_board(ctx, controller, executor, extractor, rules, state)
                    find_coin_machine(ctx, controller, executor, extractor, rules, state, image)
                except Exception:                                   # noqa: BLE001
                    logger.exception("看场出错（跳过，不影响本帧决策）")
            # 分不清是合成材料还是觉醒卡时，点开看一眼（只在准备阶段、商店关着时）
            if (state.phase == "prepare" and not state.shop_open and executor is not None and not short_on_time
                    and _INSPECT["count"] < 10 ** 6):
                try:
                    inspect_hand(ctx, controller, executor, extractor, rules, state, image)
                except Exception:                                   # noqa: BLE001
                    logger.exception("看牌出错（跳过，不影响本帧决策）")
            # 没铸成的装备：点一次试试能不能补选四选一（每件只点一次）
            if (state.phase == "prepare" and not state.shop_open and executor is not None
                    and not state.hero_panel_open and rules._unforged):
                try:
                    try_reforge(ctx, controller, executor, extractor, rules, run_dir, state)
                except Exception:                                   # noqa: BLE001
                    logger.exception("补选装备出错（跳过，不影响本帧决策）")

            _t_dec = time.perf_counter()
            actions, _diag = hybrid.decide(state)
            if state.phase == "prepare" and any(a.get("type") == "buy_xp" for a in actions or []):
                # 手里有"棋手 +1 级"的牌（扶摇直上）就先用它，不花钱升级（2026-10-14 第 625 局：手牌里有扶摇直上，
                # 就应该先用这张牌，而不是花 20 钱去升棋手）。执行层替换：效果一样、更便宜
                _fl = rules.free_level_card(state)
                if _fl:
                    actions = [_fl if a.get("type") == "buy_xp" else a for a in actions]
                    logger.info("手里有 %s：先用它升级，不花能量", _fl.get("piece"))
            if state.phase == "prepare" and executor is not None:
                if _READY["round"] != state.round_no:
                    _READY.update(round=state.round_no, idle=0, done=False, t0=time.time())
                _hand = [sl.get("name") for sl in state.hand_slots or [] if sl.get("name")]
                _cats = {n: (rules.pieces.get(n) or {}).get("category") for n in _hand}
                # 有空位且手里有英雄 / 手里有转瞬牌（不用就没）→ 还有事没做完，绝不提前开战
                _ph, _pc, _ = rules.pop_now(state)          # 统一口径（2026-10-19）：读不到时 = 上次读数 + 账本确定变化
                _free = _pc > 0 and _ph < _pc
                _hero_waiting = _free and any(c == "英雄牌" for c in _cats.values())
                _fleeting = any("转瞬" in ((rules.pieces.get(n) or {}).get("skill") or "") for n in _hand)
                if (actions or state.shop_open or state.hero_panel_open or state.hand_unread
                        or _hero_waiting or _fleeting):
                    _READY["idle"] = 0
                else:
                    _READY["idle"] += 1
                    shop_used = (state.gold < 3 or rules._shop_opens >= 1 or rules._stop_shop_round == state.round_no)
                    if (_READY["idle"] >= READY_IDLE and shop_used and not _READY["done"] and (state.round_no or 0) >= 1
                            and time.time() - _READY["t0"] >= READY_MIN_S):
                        _READY["done"] = True
                        actions = [{"type": "ready", "why": "本回合操作做完，提前开战"}]
                        logger.info("【提前开战】连续 %d 帧无事可做（准备阶段已 %.0f 秒，能量 %d，人口 %s/%s，本回合开过店 %d 次，手牌 %s），点准备 / 开战",
                                    _READY["idle"], time.time() - _READY["t0"], state.gold, state.population,
                                    state.population_cap, rules._shop_opens, _hand)
            if state.phase == "prepare":
                try:
                    (run_dir / "rules_state.json").write_text(
                        json.dumps(dict(rules.dump_state(), _perc_hp=[getattr(extractor, "_last_hp", 0), getattr(extractor, "_hp_before", 0),
                                                                      getattr(extractor, "_hp_round", 0)] if getattr(extractor, "_hp_seen", False) else None),
                                   ensure_ascii=False, default=str), encoding="utf-8")
                except Exception:                                   # noqa: BLE001
                    logger.exception("存规则引擎记忆失败（忽略）")
            dec_ms = (time.perf_counter() - _t_dec) * 1000
            if args.perf:
                logger.info("[perf] 一帧 %.0fms = 截图 %.0f + 感知 %.0f + 决策 %.0f || %s",
                            cap_ms + ext_ms + dec_ms, cap_ms, ext_ms, dec_ms, PERF.summary())
            # 规则引擎不管的机械操作，直接在这里兜底
            if state.phase == "hero_pick":
                # 点选是 toggle：只点阿离一次（再点会取消选择），找不到就不点
                if not hero_selected:
                    from agent.perception.extractors import ROI_HERO_NAMES
                    pick = None
                    for text, box in _ocr_boxes(ctx, image, ROI_HERO_NAMES):
                        if "阿离" in text:
                            pick = (box[0] + box[2] // 2, 655)
                            break
                    # **等头像条停稳再点**（2026-10-08 第 430 局：新版本选棋手界面头像条有滑入动画，第一帧读到「阿离」在 x=785，
                    # 点下去时条已经滑到位，785 变成了「白歌」，确认后整局用错棋手）。连续两帧位置一致（±6px）才点
                    prev = getattr(main, "_pick_prev", None)
                    if pick and prev and abs(prev[0] - pick[0]) <= 6:
                        actions = [{"type": "hero_pick", "x": pick[0], "y": pick[1]}]
                        hero_selected = True
                        _save_evidence(image, "hero_pick_ali")
                    else:
                        actions = []
                    main._pick_prev = pick
                else:
                    actions = []
            elif state.phase == "settle":
                # 出局弹窗（立即结算）与最终名次页（现在退出）按钮位置不同，按文本区分。
                # 坐标用 OCR 实测框，不用写死常量：实测「现在退出」的框是 y587-627，
                # 而 BTN_EXIT_MATCH 写的 y=627 正压在下边缘，容易点空
                #（项目既有规矩：按钮一律运行时定位）。
                from agent.perception.extractors import ROI_SETTLE_BTNS
                btns = {t: b for t, b in _ocr_boxes(ctx, image, ROI_SETTLE_BTNS)}
                hit = next((t for t in btns if "现在退出" in t), None)
                if hit:
                    b = btns[hit]
                    actions = [{"type": "exit_match",
                                "x": b[0] + b[2] // 2, "y": b[1] + b[3] // 2}]
                else:
                    hit = next((t for t in btns if "立即结算" in t), None)
                    if hit:
                        b = btns[hit]
                        actions = [{"type": "settle_now",
                                    "x": b[0] + b[2] // 2, "y": b[1] + b[3] // 2}]
                    else:
                        actions = [{"type": "settle_now"}]
            elif state.phase == "settle_final":
                # 一局结束：默认停在这里等人工，不自动开下一局。
                # **例外：进程刚起来、还没打过这一局**——那说明我们是接在上一局的
                # 结算页上启动的，而 `--enter-match` 的本意就是"从现在这个画面进一局"。
                # 它原来只认大厅，于是一启动就打印"对局已结束"然后停住
                # （2026-09-23 实测：重开一把直接空转，看起来像卡住了）。
                # 只放行**一次**（打过之后 played_a_match 为真），所以不等于 --auto-chain。
                if args.auto_chain or (args.enter_match and not played_a_match):
                    actions = [{"type": "play_again"}]
                    if not args.auto_chain:
                        logger.info("启动时就停在结算页：点「再来一局」进入本次要打的那一局")
                    elif not match_over_logged:
                        # 连开模式也要留下名次和总结（2026-09-26 十连打：前两局名次没记）
                        match_over_logged = True
                        rank = read_my_rank(image)
                        logger.info("本局名次：%s", f"第 {rank} 名" if rank else "没认出来")
                        rules.note("对局结束", round=state.round_no, 名次=rank)
                        rules.decisions.append({"kind": "match_result", "rank": rank,
                                                "round": state.round_no,
                                                "state": rules._snapshot(state)})
                        for line in rules.match_summary(last_prepare_state or state).splitlines():
                            logger.info("%s", line)
                else:
                    actions = []
                    if not match_over_logged:
                        match_over_logged = True
                        rank = read_my_rank(image)
                        logger.info("本局名次：%s", f"第 {rank} 名" if rank else "没认出来")
                        rules.note("对局结束", round=state.round_no, 名次=rank)
                        rules.decisions.append({"kind": "match_result", "rank": rank,
                                                "round": state.round_no,
                                                "state": rules._snapshot(state)})
                        for line in rules.match_summary(
                                last_prepare_state or state).splitlines():
                            logger.info("%s", line)
                        logger.info("对局已结束（单场模式）：停止操作，等待人工指示")
            elif state.phase in ("lobby", "mode_lobby", "match_found"):
                # 局外导航默认不做（单场模式）；--enter-match 进一局，--auto-chain 连开
                # --enter-match 只进**一局**：打过之后回到大厅就不再点开始（2026-09-30 第 36 局
                # 打完后结算页自己退回了大厅，agent 又点了两下「开始游戏」差点开第二局）。
                navigate = args.auto_chain or (args.enter_match and not played_a_match)
                solo = find_solo_button(ctx, image) if navigate and executor else None
                if solo:
                    # **组队相关的弹窗直接点「独自」**（2026-09-24：中间会有组队相关的弹窗，直接点独自）
                    logger.info("组队弹窗：点「%s」(%d,%d)", solo[0], solo[1], solo[2])
                    _save_evidence(image, "solo_button")
                    executor._tap(solo[1], solo[2])
                    actions = []
                elif navigate and state.phase == "mode_lobby" and "钻石" in " ".join(
                        t for t, _b in _ocr_boxes(ctx, image, (1040, 660, 240, 60))):
                    # 2026-10-01：新模式「钻石狂潮」（消耗钻石买初始阵容）。「开始游戏」下面写的模式是它就不点，停手等人工
                    logger.error("模式选择停在「钻石狂潮」上 —— 不开始（只打万象大赛），请人工切回")
                    _save_evidence(image, "mode_diamond")
                    actions = []
                elif navigate:
                    actions = [{"type": {"lobby": "start_match",
                                         "mode_lobby": "start_game",
                                         "match_found": "confirm_match"}[state.phase]}]
                else:
                    actions = []
            elif state.phase == "lord_panel":
                logger.info("棋手详情面板挡住了画面（多半是误点了棋盘上的棋手），点面板外关掉")
                actions = [{"type": "close_lord_panel"}]
            elif state.phase == "settings":
                # 设置面板盖住了整个画面（多半是误触返回键弹出来的）。
                # 点左上角 ✕ 关掉 —— **绝不能碰右下角的「退出对局」**。
                actions = [{"type": "close_settings"}]
            elif state.phase == "unknown":
                # 卡在未识别界面：先 OCR 找常见按钮文本直接点；找不到再点空白
                if unknown_since is None:
                    unknown_since = time.time()
                elif time.time() - unknown_since > (5 if known_phase == "prepare" else UNKNOWN_RESCUE_SEC):
                    # 准备阶段里冒出来的 unknown 多半是某个详情面板（召唤物「重剑」的面板没有「出售」，认不成英雄详情），
                    # 准备阶段只有七十来秒 —— 5 秒就动手（2026-09-30 第 43 局：卡在重剑面板上，翩跹舞步没用、回合白丢）
                    clicked = False
                    if executor:
                        # 先处理"是否退出游戏？"——只点取消，绝不点确定
                        quit_box = [(t, b) for t, b in _ocr_boxes(ctx, image, (300, 250, 700, 250))
                                    if "退出游戏" in t]
                        if quit_box:
                            cancel = next(((t, b) for t, b in
                                           _ocr_boxes(ctx, image, (300, 380, 700, 160))
                                           if re.search(r"取消|否|继续游戏", t)), None)
                            if cancel:
                                bx = cancel[1][0] + cancel[1][2] // 2
                                by = cancel[1][1] + cancel[1][3] // 2
                                logger.warning("出现退出游戏弹窗，点「%s」取消 (%d,%d)",
                                               cancel[0], bx, by)
                                executor._tap(bx, by)
                            else:
                                logger.error("出现退出游戏弹窗但找不到取消按钮，停手等人工")
                            unknown_since = time.time()
                            continue
                        # **底部那条要一直罩到屏幕右缘**：名次结算页的「下一步」在 x≈1100~1200，
                        # 原来写的 (200,550,900,160) 只到 x=1100，**按钮刚好在外面** ——
                        # 打完一局就卡在结算页出不来（2026-09-22 拿了第 2 名卡住）。
                        # **局外的账号弹窗**（2026-09-26 第 10 局开局卡了 8 分钟）：成长之路升级 →「前往」→ 成就达成 / 棋手解锁
                        # 「点击空白退出/继续」→ 获得棋手「继续」→ 成长之路页（左上角返回）。只在局外（回合 0）处理，
                        # 不点「一键领取」（账号奖励留给玩家自己）。
                        # 局外界面兜底：回合号为 0，或已经 unknown 超过 60 秒（2026-10-09 第 448 局：客户端冻结重启后回合号沿用 3，
                        # 「公告」面板认得出却被这个条件挡住，干等）。下面每条都要求认出特定文字，放宽门控不会盲点
                        if not clicked and executor and (not state.round_no or time.time() - unknown_since > 60):
                            full = " ".join(t for t, _b in _ocr_boxes(ctx, image, (0, 0, 1280, 720)))
                            spot = None
                            if "新模式解锁" in full:
                                # 2026-10-01 第 50 局：新模式「钻石狂潮」介绍页（消耗钻石买初始阵容）。
                                # **不点「下一步」、不点钻石价格按钮**，点右上角 ✕（证据帧 scratchpad s1）。
                                spot, why = (1009, 111), "新模式解锁介绍 → 右上角 ✕"
                            elif "今日宜上分" in full or ("试试这套阵容" in full and "收藏阵容" in full):
                                # 2026-10-03 第 172 局：开局推荐阵容弹窗「今日宜上分，试试这套阵容~」，按钮只有「查看详情」「收藏阵容」，
                                # 关闭是右上角 ✕ 图标（认不出文字），卡了 12 分钟（证据帧 live_1791036401/stuck_1791037139.png）
                                spot, why = (1188, 105), "今日推荐阵容弹窗 → 右上角 ✕"
                            elif "前往获取" in full and ("今日不再弹出" in full or "新棋手上线" in full or "CONTESTANT" in full):
                                # 2026-10-05 第 274 局：游戏冻结后重启客户端，进大厅先弹活动广告（「安琪拉 新棋手上线」），
                                # 按钮只有「前往获取」（不点），关闭是右上角 ✕（1920 截图 (1707,234) → 1280 (1138,156)）
                                spot, why = (1138, 156), "活动广告弹窗 → 右上角 ✕"
                            elif "点击进入游戏" in full:
                                # 2026-10-08 第 398 局：游戏不停机更新（v1.5.1）后退回标题页，卡 12 分钟
                                spot, why = (640, 630), "标题页「点击进入游戏」"
                            elif "解锁" in full and "点击查看" in full:
                                # 同上的前一屏「钻石狂潮解锁 / 点击查看」：唯一按钮，点了进介绍页再由上一条关掉
                                spot, why = (640, 590), "新模式解锁提示「点击查看」"
                            elif "点击空白" in full:
                                spot, why = (640, 690), "点击空白"
                            elif "成长之路等级" in full and "前往" in full:
                                spot = next(((b[0] + b[2] // 2, b[1] + b[3] // 2) for t, b in
                                             _ocr_boxes(ctx, image, (400, 520, 480, 120)) if "前往" in t), None)
                                why = "成长之路升级弹窗「前往」"
                            elif (re.fullmatch(r"\s*棋手\s*", " ".join(t for t, _b in _ocr_boxes(ctx, image, (0, 0, 400, 80))))
                                  and re.search(r"局内预览|总览|限时活动获取", full)):
                                # 2026-10-07 第 377 局：重启游戏后的广告没关掉，兜底点"棋盘右侧空地"进了「棋手」皮肤页，卡 12 分钟
                                # （证据帧 live_1791381091/stuck_1791381160.png）。左上角「< 棋手」返回
                                spot, why = (35, 40), "棋手页 → 左上角返回"
                            elif re.search(r"更新公告", " ".join(t for t, _b in _ocr_boxes(ctx, image, (0, 0, 400, 80)))):
                                # 2026-10-08 第 425 局：闪退重进后弹「卡牌更新公告」（各英雄增强 / 削弱的卡片列表）→ 左上角返回
                                spot, why = (35, 40), "卡牌更新公告 → 左上角返回"
                            elif re.search(r"商城", " ".join(t for t, _b in _ocr_boxes(ctx, image, (0, 0, 400, 80)))):
                                # 同一次：从棋手页返回后落在「商城」页（皮肤商店，有钻石价格）—— 什么都不点，只点左上角返回
                                spot, why = (35, 40), "商城页 → 左上角返回"
                            elif "成长之路" in " ".join(t for t, _b in _ocr_boxes(ctx, image, (0, 0, 400, 80))):
                                spot, why = (35, 40), "成长之路页 → 左上角返回"
                            elif re.fullmatch(r"\s*公告\s*", " ".join(t for t, _b in _ocr_boxes(ctx, image, (60, 15, 160, 55)))):
                                # 游戏更新后的「公告」面板（2026-09-30 v1.4.1：客户端重启后弹出，✕ 是图标不是文字，
                                # 认不出按钮就一直干等，卡了 4 个小时）。左上角标题「公告」认出来才点右上角 ✕（证据帧 unknown_1790720842）
                                spot, why = (1187, 47), "更新公告 → 右上角 ✕"
                            if spot and tap_gate.gate("LOBBY_SPOT", spot, image=image, why=why):
                                # ✕ 的位置是写死的：只在局外 / 过场类界面上点（tap_gate.SAFE_TAPS），对局内界面上不点
                                logger.info("unknown 兜底：局外弹窗（%s），点 %s", why, spot)
                                _save_evidence(image, "lobby_popup")
                                executor._tap(*spot, src=f"const:LOBBY_SPOT:{why}")
                                clicked = True
                        for roi in ((200, 550, 1080, 165), (300, 200, 700, 400)):
                            if clicked:
                                break
                            for text, box in _ocr_boxes(ctx, image, roi):
                                # 两条约束缺一不可：
                                # 1) **不要匹配"退出"**：会命中"是否退出游戏？"这类弹窗标题，
                                #    把标题当按钮点下去等于确认退出（2026-09-22 险些发生）。
                                #    退出弹窗由下面的专门分支处理。
                                # 2) **按钮文字一定很短**，且关键词要是完整词而不是长句里的片段。
                                #    2026-09-22 第二局：棋手详情面板被误点打开后，兜底把面板里
                                #    「始祖熊灵」的描述 '口。战斗开始时' 当成按钮（"开始"命中句中），
                                #    点在棋盘上越点越出不去。所以限定长度 ≤6，并把"开始/再来/立即"
                                #    这种会出现在句子里的片段换成完整按钮名。
                                if len(text) > 6:
                                    continue
                                # 「重连有分同享」：组队时自己出局后被重启 / 断线，重连观战等队伍结算才能共享队伍最高名次（2026-10-08 第 431 局）
                                if re.search(r"重连有分同享|独自|确定|确认|关闭|知道了|跳过|再来一局|立即结算|领取|下一步|继续|前往大厅",
                                             text):
                                    bx, by = box[0] + box[2] // 2, box[1] + box[3] // 2
                                    if text.startswith("独自"):
                                        # 「独自战斗」是标题，按钮是它下面的「加入」（第三轮第 4 局点了两次标题没反应）
                                        solo = find_solo_button(ctx, image)
                                        if solo:
                                            text, bx, by = solo
                                            _save_evidence(image, "solo_button")
                                    logger.info("unknown 兜底：识别到按钮 %r，点击 (%d,%d)", text, bx, by)
                                    executor._tap(bx, by)
                                    clicked = True
                                    break
                            if clicked:
                                break
                        prep_ok = known_phase == "prepare" and prep_escapes.get(state.round_no, 0) < 3
                        # 局外（回合 0 / 结算后 / 大厅）不点"棋盘右侧空地"：那是按棋盘标定的点，
                        # 大厅里会点进棋手页或广告（2026-10-07 第 377 局）
                        in_lobby = not state.round_no or known_phase in ("settle_final", "lobby", "mode_lobby", "match_found")
                        # 点之前问 screen_cls（2026-10-09）：历史 prep_unknown_escape 帧抽 150 张，124 张模型判 battle(≥0.9)、
                        # 其余低置信 —— 这条逃生绝大多数是在**战斗画面**上盲点棋盘。不在允许界面（英雄面板 / 放大详情）就不点
                        from agent.perception.extractors import TAP_SIDE_GROUND
                        if (not clicked and not in_lobby and (blank_escapes < BLANK_ESCAPE_MAX or prep_ok) and executor
                                and tap_gate.gate("TAP_SIDE_GROUND", TAP_SIDE_GROUND, image=image,
                                                  why=f"unknown 兜底逃生（上一个阶段 {known_phase}）")):
                            if known_phase == "prepare":
                                prep_escapes[state.round_no] = prep_escapes.get(state.round_no, 0) + 1
                                _save_evidence(image, "prep_unknown_escape")
                            # **有界逃生**：点一次 TAP_BLANK(170,100)。
                            # 它和下面禁掉的那些"盲操作"不是一回事 —— 那是空景，
                            # 实测不会碰到棋子/按钮；禁的是点棋盘、点"空格"、按返回键。
                            # 为什么需要它：2026-09-22 第十一局，天赋三选一**选对了**，
                            # 卡在"选完弹出的放大详情"上 3 分钟，直到被下一个拍卖界面盖掉
                            # —— 那个面板不会自己关，而 executor 里那一下 0.6s 的空白点
                            # 多半点早了。整局就这么废了。
                            # 上限 3 次/局：能救回"该关没关的模态"，又不会退化成无脑乱点。
                            blank_escapes += 1
                            logger.warning("unknown 兜底：认不出按钮，点一次棋盘右侧空地试图关闭模态"
                                           "（第 %d/%d 次）", blank_escapes, BLANK_ESCAPE_MAX)
                            executor._tap(*TAP_SIDE_GROUND, src="const:TAP_SIDE_GROUND:unknown_escape")
                            clicked = True
                        if (not clicked and not in_lobby and executor and known_phase in ("talent", "card_pick", "lord_boost")
                                and close_big_escapes.get(state.round_no, 0) < 3
                                and tap_gate.gate("TAP_CLOSE_BIG_CARD", (400, 60), image=image,
                                                  why=f"unknown 兜底：关放大详情（上一个阶段 {known_phase}）")):
                            # **右侧空地被 gate 拦了，换场馆背景**（2026-10-10 第 524 局：一直点到棋手，导致后面的操作都无法进行）：
                            # 选完天赋的放大详情 screen_cls 判 talent，TAP_SIDE_GROUND 在 talent 上不允许 → 什么都没点，卡 35 秒。
                            # (400,60) 在 talent / 放大详情上允许，第 3 回合同样的画面就是它关掉的。每回合最多 3 次。
                            close_big_escapes[state.round_no] = close_big_escapes.get(state.round_no, 0) + 1
                            logger.warning("unknown 兜底：上一个阶段 %s，点场馆背景 (400,60) 关放大详情（本回合第 %d 次）",
                                           known_phase, close_big_escapes[state.round_no])
                            executor._tap(400, 60, src="const:TAP_CLOSE_BIG_CARD:unknown_escape")
                            clicked = True
                        if not clicked:
                            # **什么都不做**，只留证据帧。所有"盲操作"的路都试过且都有害：
                            #  - 点棋盘 → 点中棋子 → 弹英雄详情面板（左下角是「出售」）
                            #  - 点"空格" → `_field` 记账不全，那格往往站着人，同上
                            #  - 按返回键 → **这个游戏的返回键是打开设置面板**
                            #    （2026-09-22 实测复现：设置页右下角还有「退出对局」）
                            # 认不出来就等，比乱点强；这张帧就是下次补阶段判据的依据。
                            import cv2
                            shot = run_dir / f"stuck_{int(time.time())}.png"
                            cv2.imwrite(str(shot),
                                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                            logger.warning("unknown 已持续 %ds 且认不出任何按钮 —— 不做盲操作，"
                                           "证据帧存至 %s", UNKNOWN_RESCUE_SEC, shot.name)
                    unknown_since = time.time()
            if state.phase != "unknown":
                unknown_since = None
                known_phase = state.phase
            if state.phase in ("mode_lobby", "lobby", "hero_pick"):
                blank_escapes = 0          # 新的一局，逃生次数重新计
                POOL_LEFT.clear()
                PREP_SEEN.clear()
                LEVEL_ACTED.clear()
                BOARD_UNREADABLE.clear()
                BOARD_MISS.clear()
                RECHECK_CELLS.clear()
            record = {"ts": time.time(), "state": state.to_dict(),
                      "actions": actions, "executed": bool(args.execute)}
            if state.phase == "prepare":
                try:
                    record["board"] = rules.board_facts(state)    # Laya f5 的场上事实（离线建数据集直接读它）
                    record["carry"] = rules._carry_piece
                    record["maple"] = rules._maple_bound or ""     # 枫叶舞绑定对象（carry/swap 题的换人代价；"" = 没绑）
                    record["route"] = rules.route_ctx()             # f9k：锁定的官方路线 {route: key, route_since: 回合}；没锁 = {}
                    record["lord_boosts"] = sorted(rules._lord_boosts)   # 已选的枫叶舞强化（红叶染锋 = 换主C 保留一半等级）
                except Exception:                                   # noqa: BLE001
                    logger.exception("board_facts 出错（跳过）")
                # 推演器回放真值（2026-10-14）：本回合准备阶段第几秒、我们是否已经做过会改等级的操作。
                # 离线取"pre_action 且 prep_t ≥ 几秒"的帧 = 整备结算完、我们还没动手时的全场等级。
                PREP_SEEN.setdefault(state.round_no, time.time())
                record["prep_t"] = round(time.time() - PREP_SEEN[state.round_no], 2)
                record["pre_action"] = state.round_no not in LEVEL_ACTED
            log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_fp.flush()
            if state.phase == "prepare" and args.execute and any(
                    isinstance(a, dict) and a.get("type") in LEVEL_ACTION_TYPES for a in (actions or [])):
                LEVEL_ACTED.add(state.round_no)

            # 账本落盘：决策层记录的完整事件流水，事后可与日志/证据帧对照排查
            if len(rules.decisions) != decisions_len:
                # 决策点记录（Laya 训练数据），追加写
                with open(run_dir / "decisions.jsonl", "a", encoding="utf-8") as fh:
                    for rec in rules.decisions[decisions_len:]:
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + chr(10))
                decisions_len = len(rules.decisions)
            if len(rules.ledger) != ledger_len:
                (run_dir / 'ledger.jsonl').write_text(
                    chr(10).join(json.dumps(e, ensure_ascii=False)
                                 for e in rules.ledger),
                    encoding='utf-8')
                ledger_len = len(rules.ledger)

            summary = (f"[{state.phase}] 回合{state.round_no} 能量{state.gold} "
                       f"人口{state.population}/{state.population_cap or '?'} "
                       f"禁:{state.banned_faction or '-'} "
                       f"商店:{[s.piece for s in state.shop]} 手牌:{len(state.hand)}")
            if state.phase == "auction" and state.auction:
                summary += f" 拍卖:{[(l.piece, l.top_bid) for l in state.auction.lots]} cd={state.auction.countdown}"
            if state.phase == "talent":
                summary += f" 天赋:{[t.name for t in state.talents]}"
            if actions:
                summary += f" => {actions}"
            logger.info(summary)

            # 装备四选一弹窗：装备拖到英雄身上后 4 秒内检测并点击选项
            # ↓ **这一段现在走不到了**（基础装备的四选一改成拖完当场抢，见 grab_equip_popup）。
            # equip_popup_until 再也不会被置成未来时间，保留只是为了万一要回退。
            if equip_popup_until and time.time() < equip_popup_until and executor:
                # 用**这件装备实际穿在谁身上**的职业来打分，不是主C 的职业。
                # 装备未必给主C（decide_use_cards 会挑场上目标），拿主C 的定位去
                # 评"这件装备好不好"是错配：2026-09-22 第五局装备穿在孙膑（辅助）身上，
                # 却按主C 的 dps 定位选，选出的是魔女斗篷（法抗）。
                equip_role = rules.equip_role_for(equip_target_piece or rules._carry_piece)
                role = equip_role
                point = detect_equip_popup(ctx, image, extractor, equip_target, role,
                                           run_dir, equip_base,
                                           worn=rules._worn.get(equip_target_piece or rules._carry_piece) or ())
                if point:
                    logger.info("【装备四选一】点击 %s —— 基础装备 %s 穿给 %s(%s)，"
                                "弹窗中心 %s", point, equip_base, equip_target_piece,
                                role, equip_target)
                    # **点完必须验一下**。原来点完就记 ✅，而 ✅ 只代表"我点了"，
                    # 不代表"铸造成功" —— 2026-09-23 画面上装备四选一
                    # 3 个英雄都没有成功，而同一局日志里三条全是 ✅。
                    # 判据是**那个英雄身上的装备件数有没有变多**（像素观测），
                    # 不是"弹窗消失了"——关掉弹窗和选中一项都会让它消失，分不出来。
                    # 证据帧确认过点击坐标正好落在选项图标正中，所以坐标不是问题；
                    # 剩下的未知是"这个界面吃不吃单击"——点了没用就改试拖拽，
                    # 两种都试完还没铸造出来，就如实报失败并留证据帧，别再假装成功。
                    executor._tap(*point)
                    equip_ok = _equip_forged(ctx, controller, extractor,
                                             equip_target, equip_base,
                                             equip_before_count)
                    if not equip_ok:
                        logger.warning("【装备四选一】单击没铸造出来 —— 改用拖拽："
                                       "从 %s 拖到 %s", equip_target, point)
                        executor._drag(equip_target, point)
                        equip_ok = _equip_forged(ctx, controller, extractor,
                                                 equip_target, equip_base,
                                                 equip_before_count)
                        if equip_ok:
                            logger.warning("【装备四选一】✅ **拖拽才生效**（单击无效）"
                                           " —— 执行层应改为拖拽")
                    else:
                        logger.info("【装备四选一】✅ 单击生效（弹窗已消失，%s）",
                                    equip_target_piece)
                    if not equip_ok:
                        import cv2
                        shot = screencap(controller)
                        fp = run_dir / f"equip_stuck_{int(time.time())}.png"
                        cv2.imwrite(str(fp), cv2.cvtColor(shot, cv2.COLOR_RGB2BGR))
                        logger.error("【装备四选一】❌ 单击和拖拽都没把装备铸造上去 —— "
                                     "机制还没搞对，证据帧 %s", fp.name)
                    equip_popup_until = 0
                    equip_done = True
                else:
                    # 没认出弹窗有两种完全不同的原因，光看日志分不出来：
                    #   a) 牌还举在半空（两段式没补点目标）—— 弹窗压根没开；
                    #   b) 弹窗开了但霍夫圆没检出来。
                    # 留一帧现场（每局一次），下一轮直接看图，别再靠猜。
                    _save_evidence(image, f"equip_no_popup_{equip_base}")
                    logger.info("【装备四选一】…本帧没认出弹窗（剩余 %.1fs）—— "
                                "基础装备 %s / 目标 %s / 中心 %s / 顶部横幅=%s",
                                equip_popup_until - time.time(), equip_base,
                                equip_target_piece, equip_target,
                                "在" if state.awaiting_target else "不在")
            elif equip_popup_until and time.time() >= equip_popup_until and executor:
                # **超时了要明确记一笔**：2026-09-22 出现装备没点击，
                # 而原来超时是静默的 —— 日志里只看得到"拖了装备"，看不到"选没选成"，
                # 于是每次都要翻证据帧才知道结果。
                logger.warning("【装备四选一】❌ 超时未完成 —— 基础装备 %s 穿给 %s，"
                               "这件装备**没有铸造**，还停在未成品状态",
                               equip_base, equip_target_piece)
                equip_popup_until = 0

            if state.hero_panel_open and executor and state.phase == "prepare":
                # 英雄详情面板盖住了升级按钮（左下角变成「出售」）。
                # 点棋盘右侧空地关掉它 —— 这里是棋盘画面，TAP_SIDE_GROUND 有效
                # （拍卖界面上不能用，见 _dismiss_enlarged）。
                # **要有次数上限**（2026-10-01 第 63 局：每帧都"点空地关掉 + continue"，关不掉就整帧什么都不做，
                # 决策突然变得很慢；面板盖住手牌，古币连续 3 次找不到被搁置）。
                # 第 1~2 次点右侧空地，第 3~4 次点场馆背景 (400,60)，再不行存证据帧、本回合不再为它停下。
                if hero_panel_tries[0] != state.round_no:
                    hero_panel_tries[:] = [state.round_no, 0]
                hero_panel_tries[1] += 1
                k = hero_panel_tries[1]
                if k <= 4:
                    from agent.perception.extractors import TAP_SIDE_GROUND as _TSG
                    spot, spot_name = (_TSG, "TAP_SIDE_GROUND") if k <= 2 else (TAP_CLOSE_BIG_CARD, "TAP_CLOSE_BIG_CARD")
                    if k == 1:
                        _save_evidence(image, "hero_panel")
                    logger.info("英雄详情面板挡住了升级按钮，第 %d 次关：点 %s", k, spot)
                    # 面板判据是 OCR「出售/剩余数量」，偶有误判（hero_panel_stuck）；模型高置信说不是面板就不点、照常决策
                    if _fixed_tap(executor, spot_name, *spot, image=image, why="关英雄详情面板"):
                        time.sleep(0.5)
                        continue
                if k == 5:
                    _save_evidence(image, "hero_panel_stuck")
                    logger.warning("英雄详情面板点了 4 次还在（或是误判）—— 本回合不再为它停下，其余决策照常")

            # 关店点了两次还开着 —— 这坑已经出现三次了，但**报警时什么都没存**，
            # 事后只能对着坐标猜。存一帧证据（每回合一张），下次就能直接看画面。
            if (state.shop_open and rules._shop_closes >= 2
                    and shop_fail_round != state.round_no):
                shop_fail_round = state.round_no
                import cv2
                path = run_dir / f"shop_close_fail_r{state.round_no}_{int(time.time())}.png"
                cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                logger.warning("关店失败证据帧已存：%s", path.name)

            if state.awaiting_target and executor:
                # **两段式交互的第二段**：顶部挂着「请指定英雄使用」，说明牌已经举起来了、
                # 游戏在等我们再点一下目标英雄。不补这一下，牌就一直卡着，
                # 装备四选一压根不会弹（2026-09-22 实战 5/5 帧都停在这个状态）。
                spot = pending_target or rules._carry_pos(state) or None
                if spot:
                    # 补点也要避开手牌区（y≤545）：最下面一行格子中心被手牌盖着，点原坐标会点到手牌上
                    # （2026-09-24 第二轮第 4 局：反伤重甲 → 墨子(555,569) 连补点 3 次都没穿上，被搁置）
                    logger.info("检测到「请指定英雄使用」，补点目标 %s", board_drop_point(spot))
                    executor._tap(*board_drop_point(spot))
                    pending_target = None
                    time.sleep(0.6)
                    continue          # 立刻重新感知：这一下多半让四选一弹出来了
                logger.warning("挂着「请指定英雄使用」但不知道该点谁 —— 不乱点")

            replan = False
            if executor and actions:
                # 准备阶段剩余秒数：画面读得到就用（顶部"第N回合 | Ms"），读不到按上次读数减去流逝时间估（第 843 局）
                # **只信前后一致的读数**（别因为识别不了就不觉醒）：这一帧读数要和上一帧按流逝时间推出来的值差 ≤2 秒才算"可信"，
                # 不可信 / 读不到 → _left=None → 照常觉醒（宁可偶尔晚觉醒，也不因错读整回合不觉醒）。最多只拖到下回合（下回合准备阶段 ≥30 秒）。
                if state.phase == "prepare":
                    _now = time.time()
                    if isinstance(state.countdown, int) and state.countdown >= 0:
                        _pv = prep_clock if prep_clock.get("r") == state.round_no else {}
                        _ok = bool(_pv) and abs(state.countdown - (_pv["v"] - (_now - _pv["t"]))) <= 2
                        prep_clock.update(r=state.round_no, v=state.countdown, t=_now, ok=_ok)
                    _left = ((prep_clock["v"] - (_now - prep_clock["t"]))
                             if prep_clock.get("r") == state.round_no and prep_clock.get("ok") else None)
                else:
                    _left = None
                for act in actions:
                    _awk = rules.would_awaken(act) if state.phase == "prepare" else ""
                    if _awk and _left is not None and _left < AWAKEN_MIN_LEFT_S:
                        # **准备阶段快结束时不触发觉醒**（2026-10-20 第 843 局：最后 5 秒觉醒了主C，主C 回到了手牌，要避免）：
                        # 觉醒回手 → 认出来 → 拖回场上要 5~6 秒，来不及就整场战斗少一个人（R12 守约）。留到下回合再做。
                        logger.info("准备阶段只剩约 %.0f 秒：%s 会让 %s 觉醒回手、来不及拖回场上 → 本回合先不做", _left, act.get("piece"), _awk)
                        continue
                    if (state.phase == "prepare" and _left is not None and _left < EQUIP_MIN_LEFT_S
                            and act.get("type") == "drag_card" and act.get("kind") == "装备牌"
                            and (act.get("piece") or "") in (extractor.equip_recipes or {})):
                        # 基础装备要等四选一弹窗再点选；准备阶段快结束时弹窗会被开战关掉（2026-10-21：普通装备放到人身上之后 4 选 1 失败，时有发生）→ 留到下回合
                        logger.info("准备阶段只剩约 %.0f 秒：%s 是基础装备（要四选一），来不及选 → 本回合先不穿", _left, act.get("piece"))
                        continue
                    if replan:
                        break        # 上一个动作已改变手牌/棋盘，后面的坐标全过期
                    # **回合号进签名**：去重窗口原本会跨回合。实战后果：
                    # 回合0 点过一次开店 → 回合1 游戏自动关店 → 回合1 再想开店，
                    # 两次决策都落在那 6 秒窗口里被丢掉，而决策层的"每回合最多开 2 次"
                    # 计数照加 —— 预算被**从没真正发出去的点击**耗光，
                    # 于是整局商店没打开过、一张英雄都没买（2026-09-22）。
                    extra = {}
                    if act.get("type") == "refresh_shop":
                        # 刷新之后商店就换了一批：**商店变了就允许立刻再刷**（搜牌），
                        # 只有商店没变（上一下还没生效）才按"6 秒内不重发"去重。
                        # 原来签名不含商店内容，每刷一次都要干等 6 秒（2026-09-23 第 6 回合）。
                        extra["_shop"] = [sl.piece for sl in (state.shop or [])]
                    if act.get("type") == "buy_piece":
                        # **买成了（能量 / 手牌变了）就允许立刻再买同一格**（2026-10-20 第 844 局：商店买了两次沈梦溪，中间停顿有点久：
                        # 第一次买成后商店那一格仍是沈梦溪（9→7 能量、手牌 3→4），签名只含动作 → 干等 6 秒去重才买第二张）。
                        # 能量和手牌都没变 = 上一下还没生效，照样按 6 秒去重，不会连点
                        extra["_gold"] = state.gold
                        extra["_hand_n"] = len(state.hand_slots or [])
                    if act.get("type") in ("drag_card", "sell_card"):
                        # 同理：**手牌变了就允许立刻再用同名牌**（2026-10-18 第 748 局：手里有多个古币时经常卡住操作——
                        # 两张古币拖给同一个主C，签名一模一样，第二张干等 6 秒）。画面手牌还没更新（上一张没生效）时签名不变，照样去重
                        extra["_hand"] = [sl.get("name") for sl in (state.hand_slots or [])]
                    sig = json.dumps({**act, **extra, "_r": state.round_no},
                                     sort_keys=True, ensure_ascii=False)
                    if sig == last_action_sig and time.time() - last_action_ts < 6:
                        continue  # 同一动作 6 秒内不重发（避免重复点击买牌/选棋手）
                    if act.get("type") == "card_pick":
                        act = settle_card_pick(ctx, controller, extractor, act)
                    if act.get("type") == "auction_bid":
                        # 两步：先点中拍品（用感知层实测坐标），再 OCR 定位"出价"按钮。
                        # 定位不到 = 没选中，跳过本次，别空点。
                        lot_x, lot_y = act.get("x", -1), act.get("y", -1)
                        if lot_x >= 0:
                            executor._tap(lot_x, lot_y - 60)
                            time.sleep(0.6)
                        if act.get("select_only"):
                            # 预览期：拍品已选中，出价按钮还没出来，等下一帧
                            logger.info("拍卖预览期：已选中 %s，等正式开始再出价",
                                        act.get("piece"))
                            last_action_sig, last_action_ts = sig, time.time()
                            continue
                        btn = locate_bid_button(ctx, controller, run_dir,
                                                f"{int(time.time())}_after_select")
                        if btn is None:
                            logger.warning("拍卖：点了拍品 %s 但没找到'出价'按钮，"
                                           "本次放弃（证据帧已存）", act.get("piece"))
                            continue
                        logger.info("拍卖：'出价'按钮定位到 %s，出价 %s", btn, act.get("bid"))
                        executor._tap(*btn)
                        last_action_sig, last_action_ts = sig, time.time()
                        continue
                    if (act.get("type") in ("drag_card", "sell_card") and act.get("src")
                            and act["src"][1] >= HAND_SRC_MIN_Y and state.phase == "prepare"):
                        # 卖牌也要核对：2026-09-23 卖艾琳那一刻正好塞进来两张新牌，
                        # 抓取点过时，连卖 6 次都没卖掉
                        if not confirm_hand_src(ctx, controller, extractor, act):
                            rules.note_src_missing(act.get("piece") or "")
                            continue
                    # **往某个英雄身上送东西之前，先确认那一格站的就是他**（合成 / 装备牌 / 效果牌）。
                    # 账本里的位置会过期：2026-09-23 日渊往"曜的格子"拖，那格其实没人，牌举在空地上
                    # （日渊拖到空地失败）。
                    if (act.get("type") == "drag_card" and state.phase == "prepare"
                            and (act.get("merge")
                                 or (act.get("kind") in ("装备牌", "效果牌") and act.get("target")
                                     and not act.get("self_cast")))
                            and not confirm_merge_target(ctx, controller, executor, extractor,
                                                         rules, state, act)):
                        replan = True               # 账本改过了，重新感知、重新决策
                        continue
                    # **卖场上英雄之前也先点开核对**：卖错人收不回来（可能把主力卖了）。
                    # 2026-09-23 账本把苏烈记成猪八戒，"卖猪八戒"那一下抓的是苏烈（结果把苏烈移动到了右下角）。
                    if (act.get("type") == "sell_unit" and act.get("src") and state.phase == "prepare"
                            and not confirm_merge_target(ctx, controller, executor, extractor, rules, state,
                                                         {"merge": True, "piece": act.get("piece"), "_fresh": True,
                                                          "dst": tuple(act["src"])})):
                        replan = True
                        continue
                    sold_equips = 0
                    if act.get("type") == "sell_unit" and act.get("src"):
                        sold_equips = (state.equip_counts or {}).get(cell_key(tuple(act["src"])), 0)
                        # 卡面"在场时，获得…装备"的自带装备（干将莫邪的双剑·雄）卖掉就消失、不退回手牌（2026-10-18：718~727 局卖干将 5 次都干等 6 秒）
                        if sold_equips > 0 and re.search(r"在场时，获得[^。]{0,8}装备", (rules.pieces.get(act.get("piece") or "", {}) or {}).get("skill") or ""):
                            sold_equips -= 1
                    act["_phase"] = state.phase   # 关放大详情时要知道现在是哪个界面
                    if (act.get("type") == "card_pick" and multi_pick["k"]
                            and len(multi_pick["picked"]) + 1 < multi_pick["k"]):
                        # **N 选多还没选完：选完这张别点空地**（2026-10-13：5 选 2 的效果牌经常只选 1 张就关闭界面）：
                        # 执行层"选完关放大详情"那一下点 (1120,500) 正好把多选界面收起来
                        act["multi_more"] = True
                    executor.execute(act)
                    rules.note_executed(act)      # 预算按**实际执行**计数，不按决策计数
                    if act.get("type") == "drag_card" and act.get("kind") in ("效果牌", "天赋牌"):
                        m_k = re.search(r"(\d)选([2-9])", rules._card_text(act.get("piece") or "")
                                        or rules.pieces.get(act.get("piece") or "", {}).get("skill", ""))
                        if m_k:
                            multi_pick.update(k=int(m_k.group(2)), picked=[], ts=time.time(),
                                              card=act.get("piece"), shots=0)
                            logger.info("【%s】卡面是 %s选%s —— 要连续选 %s 张", act.get("piece"),
                                        m_k.group(1), m_k.group(2), m_k.group(2))
                    elif act.get("type") == "card_pick" and multi_pick["k"]:
                        multi_pick["picked"].append(act.get("piece"))
                        multi_pick["ts"] = time.time()
                        if len(multi_pick["picked"]) >= multi_pick["k"]:
                            logger.info("【%s 选 %d 张】选完：%s", multi_pick["card"], multi_pick["k"], multi_pick["picked"])
                            multi_pick.update(k=0)
                    forget_seen_cells(act)
                    if (act.get("type") in ("use_card", "sequence")
                            or (act.get("type") == "drag_card"
                                and act.get("kind") in ("效果牌", "天赋牌"))):
                        # **只有打出效果牌/天赋牌才会弹「卡牌N选1」**（锦囊、兑换券、翩跹舞步…）。
                        # 原来买牌/上场/卖人之后也查，而几乎每帧都紧跟一个动作 —— 等于每帧都在跑
                        # 这个 11~15 次 OCR 的检测（`--perf` 实测 shop_open 这一项 500~700ms）。
                        extractor.card_pick_hint_until = time.time() + 6
                    if (act.get("type") == "sell_unit" and act.get("src") and state.phase == "prepare"
                            and not verify_sell(controller, rules, act)):
                        sold_equips = 0             # 没卖掉 = 装备也不会退回，别干等 6 秒
                    if sold_equips > 0:
                        wait_equip_refund(ctx, controller, extractor,
                                          len(state.hand_slots or []), sold_equips)
                    if act.get("type") == "toggle_shop":
                        # **当场验一下开关到底生没生效**。"点过 2 次关店但商店仍开着"
                        # 这条告警出现过很多次，但一直分不清是
                        #   a) 点了没反应（坐标/控件问题），还是
                        #   b) 关上了又被我们自己开回来（开关条件不对称）。
                        # 证据帧证明点击点正落在能量圆盘正中，所以更可能是 b。
                        # 记一条事实进日志，下一局直接看，别再靠猜。
                        still = shop_is_open(ctx, controller)
                        want = act.get("intent")
                        ok_toggle = (want == "open") == still
                        logger.info("商店开关：意图=%s，点完商店%s%s", want,
                                    "开着" if still else "关着",
                                    "" if ok_toggle else "  ← 没按意图变")
                        if not ok_toggle and want == "open":
                            # **圆盘打开的可能不是商店**：被收起的「卡牌N选1」/天赋三选一，入口也是这个圆盘
                            # （2026-09-23 确认：入口跟商店是同一个）。这时再点一次就把它关上了 ——
                            # 那局就是这样：开、认不出、重试关掉、再开……整回合"一直关闭商店"，三选一没选。
                            from agent.perception.state import GameState as _GS
                            probe = _GS()
                            shot = screencap(controller)
                            extractor._detect_card_pick(ctx, shot, probe)
                            _ph = "" if probe.card_pick else extractor.detect_phase(ctx, shot)
                            # 枫舞强化三选一（lord_boost）同样能被收起、入口也是圆盘（2026-10-13 审计：第 609 局 R13 一直判成
                            # 商店没开、重试把刚打开的面板又关上，整回合 17 能量没用；证据帧 shop_open_fail_1791882189.png）
                            if probe.card_pick or _ph in ("talent", "lord_boost"):
                                logger.warning("点圆盘打开的是%s，不是商店 —— 不重试，下一帧去选",
                                               "卡牌N选1" if probe.card_pick else
                                               ("枫舞强化三选一" if _ph == "lord_boost" else "天赋三选一"))
                                extractor.card_pick_hint_until = time.time() + 6
                                ok_toggle = True
                        if not ok_toggle:
                            # **当场再点一次**。实测同一局 4 次开关里会失败 1 次
                            # （多半是面板动画没走完就吃了下一下）。
                            # 不重试的代价很实在：商店开着时**前两排格子被面板盖住**，
                            # 后面所有拖拽都会拖到面板上（2026-09-23 第 1 回合实测，
                            # 曜 上场和贴装备全废在 (617,372)）。
                            if want == "open":
                                # 第 37 局 R7~R10 开店连败 12 次、一张证据帧都没留（疑似圆盘打开的是认不出的 N 选多）
                                _save_evidence(shot, "shop_open_fail")
                            executor.execute(dict(act))
                            still2 = shop_is_open(ctx, controller)
                            logger.info("商店开关重试：点完商店%s%s",
                                        "开着" if still2 else "关着",
                                        "" if (want == "open") == still2
                                        else "  ← 两次都没变")
                    last_action_sig, last_action_ts = sig, time.time()
                    if act.get("type") == "drag_card" and act.get("dst"):
                        pending_target = tuple(act["dst"])   # 可能还要"再点一下目标"
                    base_eq = (act.get("piece")
                               if (act.get("type") == "drag_card"
                                   and act.get("kind") == "装备牌") else None)
                    if getattr(executor, "last_drag_aborted", False) and act.get("_grab_aborted"):
                        # 半空确认发现抓错牌、已经拖回原位：不是格子的问题，也不是人口满（别记 pop_over）
                        rules.note("抓错牌已拖回", 想拖=act.get("piece"), 实际=act.get("_wrong"))
                        executor.last_drag_aborted = False
                        _GRAB_CTX.pop("pre_shot", None)
                    elif getattr(executor, "last_drag_aborted", False):
                        # 拖到落点时人口数字变红 = 放下就超上限（2026-09-24 确认）→ 已拖回手牌。
                        # 这不是落点的问题，**不拉黑格子**；本回合按人口已满处理（改走卖弱换强）。
                        logger.warning("拖 %s 时人口数字变红（放下就超上限）→ 已拖回手牌，本回合按人口已满处理",
                                       act.get("piece"))
                        rules.confirm_drag(act, False, blame_cell=False)
                        rules.note_pop_over(state.round_no)
                        executor.last_drag_aborted = False
                    elif base_eq and base_eq in extractor.equip_recipes:
                        # **顺序很重要**：弹窗一两秒就没，回执什么时候做都行。
                        tgt = tuple(act.get("dst", (660, 505)))
                        eq_role = rules.equip_role_for(act.get("target") or rules._carry_piece)
                        forged = grab_equip_popup(
                            ctx, controller, executor, extractor, rules, run_dir,
                            tgt, base_eq, act.get("target"), eq_role,
                            (state.equip_counts or {}).get(cell_key(tgt), 0))
                        # **抢不到弹窗 = 这一拖没贴到人身上**，要当失败回执交回决策层，
                        # 否则那套"失败就拉黑落点/搁置这张牌"的机制**整条失效**：
                        # 2026-09-23 实战 风暴巨剑 → 牛魔 连着三回合重试同一格，
                        # 因为基础装备这条新路径把 verify_drag 跳过了、没人给回执。
                        # 弹窗出来过就不给失败回执（那时牌已经消耗掉了，成败另说）。
                        if not forged:
                            rules.confirm_drag(act, False)
                            # 没弹四选一，最常见的原因是**那人已经带满 3 件**（游戏直接拒收）。
                            # 2026-09-23：马可波罗满 3 件，速击之枪每回合都往他身上拖两次。
                            rules.note_equip_rejected(tgt)
                        equip_popup_until = 0      # 已经处理完，别再让主循环等
                        replan = True
                    elif (base_eq and act.get("dst") and grab_equip_popup(
                            ctx, controller, executor, extractor, rules, run_dir,
                            tuple(act["dst"]), base_eq, act.get("target"),
                            rules.equip_role_for(act.get("target") or rules._carry_piece),
                            (state.equip_counts or {}).get(cell_key(tuple(act["dst"])), 0),
                            budget=1.2, expect=False)):
                        # 成品装备拖完竟然弹了四选一 = 抓起来的是别的牌（基础装备）。
                        # 这张成品还在手里，**不给回执**（给成功会记错、给失败会拉黑一个好格子），
                        # 下一帧重新读手牌再说。
                        equip_popup_until = 0
                        replan = True
                    elif act.get("type") in MUTATING_ACTIONS:
                        ok = verify_drag(ctx, controller, extractor, act, state.hand,
                                         (state.population, state.population_cap,
                                          state.population_valid))
                        if ok is LIFTED:
                            # 两段式交互的第二段，就地做掉，不等下一帧去认横幅
                            dst = act.get("dst")
                            if dst:
                                logger.info("补点目标 %s（%s 已举起）", tuple(dst),
                                            act.get("piece"))
                                executor._tap(*board_drop_point(tuple(dst)))
                                time.sleep(0.6)
                                pending_target = None
                            else:
                                logger.warning("%s 已举起但不知道该点谁 —— 不乱点",
                                               act.get("piece"))
                            ok = None
                        if ok is WRONG_GRAB:
                            rules.note_wrong_grab(act, act.get("_wrong"), act.get("_wrong_pop", False))
                            other = act.get("_wrong")
                            if other in extractor.equip_recipes and act.get("dst"):
                                # **抓错的是基础装备、又正好落在一个英雄身上 → 四选一已经弹出来了**，马上接住。
                                # 2026-09-24 第二轮第 1 局：想拖孙尚香去合成，抓起来的是速击之枪、落在孙尚香身上，
                                # 我们当成"合成"没去选，弹窗过期，只穿上了基础装备（截图显示装备又选择失败了）。
                                tgt = tuple(act["dst"])
                                wearer = rules._cell_owner.get(tgt) or act.get("piece")
                                if not grab_equip_popup(
                                        ctx, controller, executor, extractor, rules, run_dir,
                                        tgt, other, wearer, rules.equip_role_for(wearer),
                                        (state.equip_counts or {}).get(cell_key(tgt), 0),
                                        budget=1.5, expect=False):
                                    rules.note_unforged(tgt, other)     # 四选一没接住：身上是一件没铸成的基础装备
                            ok = None
                        if ok is not None:
                            logger.info("拖拽校验: %s -> %s %s", act.get("piece"),
                                        act.get("dst"), "成功" if ok else "**没生效**")
                            rules.confirm_drag(act, ok)
                            if ok and act.get("piece"):
                                # **牌出手了，这个卡名"点开看过"的结论作废**（2026-09-23 第十三局：觉醒沈梦溪上场后，
                                # 手里剩下的普通沈梦溪张数恰好还是 1，沿用了"觉醒"标记 → 场上记录被撤销、装备扣着不发）
                                _INSPECT["flags"] = {k: v for k, v in _INSPECT.get("flags", {}).items()
                                                     if k[0] != act["piece"]}
                            if (ok and act.get("type") == "drag_card"
                                    and not act.get("kind") and not act.get("merge")):
                                # **牌离手了 ≠ 落在我拖的那一格**。游戏会把棋子归位到
                                # 最近的合法格；我们记错格子，后面给它贴装备就全拖到空地上
                                # （2026-09-23 实战：孙膑 记在 (300,569)、人在 (411,372)，
                                #  装备件数一直是 0）。用**新增的等级徽章锚点**校正一次。
                                try:
                                    from agent.perception.extractors import                                         detect_unit_anchors
                                    from agent.perception import vision_units
                                    if vision_units.enabled() and state.vision_units is not None:
                                        # **优先用视觉（D-FINE）**：徽章检测只有 ~66% 准，新单位常常认不出，校正一直不生效。
                                        # 2026-09-28 第 27 局：百里守约 拖向 (300,569)（落点压到 y=545），实际站进上一格 (276,501)；
                                        # 账本一直记 (300,569)，拖前核对点 (300,539) 又点中上一格的它 → "确认"了错的位置，
                                        # 复印机/简易协作 往 (300,569) 拖了 4 次全失败（视觉在 (276,501) "补认"了 1383 次）。
                                        from agent.perception.extractors import OWN_BOARD_ROWS
                                        time.sleep(0.4)
                                        before = {k for k, v in (state.vision_units or {}).items() if v == "hero"}
                                        found = vision_units.cell_units(
                                            screencap(controller), [tuple(c) for row in OWN_BOARD_ROWS for c in row])
                                        after = {cell_key(c) for c, v in found.items() if v[0] == "hero"}
                                    else:
                                        before = set(state.unit_anchors or {})
                                        after = set(detect_unit_anchors(screencap(controller)))
                                    fresh = [c for c in after - before]
                                    want = cell_key(tuple(act["dst"]))
                                    if want not in after and len(fresh) == 1:
                                        cx, cy = (int(v) for v in fresh[0].split(","))
                                        dx = cx - act["dst"][0]
                                        dy = cy - act["dst"][1]
                                        # **只接受"挪到附近一格"**。游戏归位是往**相邻的
                                        # 合法格**挪，不可能从棋盘这头跳到那头。
                                        # 2026-09-23 实测：不加这条时，曜 被"校正"到
                                        # (617,372) -> (1027,572)，横跨整个棋盘 ——
                                        # 那多半是另一个单位的徽章刚好这一帧才检出来。
                                        # 校正记账是把双刃剑：改对了省一次白拖，
                                        # 改错了直接把主力记丢。宁可不改。
                                        if dx * dx + dy * dy <= RELOCATE_MAX_DIST ** 2:
                                            rules.relocate_unit(act["piece"], (cx, cy))
                                        else:
                                            logger.info("观测到的新锚点 (%d,%d) 离落点 %s "
                                                        "太远，不当作归位（多半是别的单位"
                                                        "刚被检出来）", cx, cy,
                                                        tuple(act["dst"]))
                                except Exception:
                                    logger.exception("按观测校正落点时出错，忽略")
                        replan = True      # 手牌/棋盘已变，坐标全过期 → 立刻重新感知
                    if act.get("type") == "drag_card" and act.get("kind") == "装备牌":
                        # **只有基础装备**（8 种，见 data/equip_recipes.json）拖上去才弹四选一；
                        # 普通装备是成品，直接穿上没有弹窗。
                        # 之前不区分，给安琪拉穿普通装备后空等 4 秒找不存在的弹窗
                        # （2026-09-22）。
                        base = act.get("piece")
                        if base not in extractor.equip_recipes:
                            logger.info("%s 是成品普通装备，直接穿上，不等四选一", base)
                        # 基础装备的四选一已经在上面 grab_equip_popup 里当场处理完了

            # 刚做完一个会改变画面的动作 → 不等满一个 interval，马上重新感知
            # （既避免用过期坐标，也把"一帧一个动作"的代价压到最小）
            time.sleep(0.2 if replan else PHASE_INTERVAL.get(state.phase, args.interval))
        except KeyboardInterrupt:
            break
        except Exception:
            logger.exception("循环异常")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
