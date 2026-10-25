"""蒸馏后的 Laya 在实战里做决定（学生上场）。

分工（2026-09-24 定）：
- **N 选 1 / 天赋三选一 / 枫叶舞强化**：Laya 直接决定（每局只出现几次、风险低，界面会停着等你选）。
- **商店**（买哪张 / 刷新 / 升级 / 不买了）：Laya 决定，**规则只管硬约束**（买不起、买回来没位置、本回合买够了、
  按钮不在画面上）；Laya 最高概率低于 `shop_min_conf` 时回退规则的选择 —— **默认 0（不因低置信回退）**：
  让 Laya 多发挥才能收集数据；5 选 1 的题里 0.27 已远高于随机，而规则选项在 Laya 眼里常只有 0.1。
- 其余（上场、合成、装备、卖人、拍卖……）仍由规则引擎负责。

**输入与训练逐字一致**：题目由 tools/laya/build_dataset.py 里的 `card_pick_point / talent_point /
lord_boost_point / shop_point` 构造（离线建数据集也用它们），局面上下文用同一个 `ctx_from_state`。
唯一差别：场上阵容和主C 取规则引擎的实时记账（离线是账本回放），信息只会更准。

每个决策点写一行到 `run_dir/laya_points.jsonl`：题目本身（与 points.jsonl 同格式，可直接导出给老师标注）+
Laya 的分布与置信度 + 规则当时的选择 + 最终执行的是谁（laya / rule_fallback）+ 回退原因。
"""
from __future__ import annotations

from agent.decision import interim_rules as _IR   # 临时规则登记表（价值规则别写死、训练后要撤）
from agent.decision import meta as _meta   # awakened_no_copy 用（MAX_LORD_LEVEL）；函数里另有局部同名导入
import hashlib
import json
import re
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR / "tools" / "laya") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))

SHOP_TYPES = ("buy_piece", "refresh_shop", "buy_xp")


ENGINE_KINDS = ("shop", "card_pick", "talent", "auction", "use_card", "carry", "swap")
SIM_PREFIX_LIVE = False      # 见 _ask：训练集没有推演器前缀，线上也不加（2026-10-17）

class LayaPolicy:
    PAID_REFRESH_CAP = 3          # Laya 决定的付费刷新每回合最多几次（防止把钱刷光）
    STOP_SPARE_MAX = 12           # 扣掉拍卖预留后还剩这么多能量时，学生的"不买了"不执行（防攒钱）

    def __init__(self, ckpt: str, rules, run_dir: Path, shop_min_conf: float = 0.0,
                 kinds=("card_pick", "talent", "lord_boost", "shop")):
        import torch  # noqa: F401  —— 先 import torch 再让 onnxruntime 载 GPU 库
        import laya
        import build_dataset as B
        import features as F
        self.B, self.F = B, F
        self.rules = rules
        self.kinds = set(kinds)
        rules.shop_by_laya = "shop" in self.kinds          # 规则别再按张数卡 Laya 的商店（见 rules._buy_cap）
        rules.auction_by_laya = "auction" in self.kinds    # 拍哪件由 Laya 定，规则只做出价机械（2026-10-11）
        self.shop_min_conf = shop_min_conf
        t0 = time.time()
        self.agent = laya.load(str(ckpt))            # fp32 权重，bf16 autocast 推理
        self.ckpt = str(ckpt)
        info = (json.loads((Path(ckpt) / "rl_agent_config.json").read_text(encoding="utf-8"))
                .get("training", {}).get("wzwxq_distill", {}))
        self.feature_version = info.get("feature_version")
        if self.feature_version:
            F.ACTIVE = self.feature_version      # 给学生它训练时的那一版文本（f5 的新信息只在 f5 学生上生效）
        if self.feature_version and self.feature_version != F.FEATURE_VERSION:
            logger.info("Laya checkpoint 用 %s 特征训练（代码最新 %s）—— 按 %s 渲染输入，与训练一致",
                        self.feature_version, F.FEATURE_VERSION, self.feature_version)
        # 预热：第一次前向有 CUDA 初始化开销，别让它落在对局里
        self.agent.system_one("warmup", {"q": {"type": "choice", "instructions": "warmup",
                                               "criteria": {"a": "x", "b": "y"}}})
        logger.info("Laya 学生已加载 %.1fs：%s（特征 %s，接管 %s，商店置信门槛 %.2f）",
                    time.time() - t0, ckpt, self.feature_version, sorted(self.kinds), shop_min_conf)
        self.out: Optional[Path] = None
        if run_dir is not None:
            self.set_run_dir(run_dir)
        self.ctx: dict = {}
        self.auction_round = -1
        self._seen: set = set()
        self._cache: dict = {}
        self._paid_refresh: dict = {}
        rules._laya_pick = self.pick_lord_boost if "lord_boost" in self.kinds else None
        rules.carry_by_laya = "carry" in self.kinds      # 谁当主C 由 carry 题决定（规则只在主C 没了时补位）
        rules.swap_by_laya = "swap" in self.kinds        # 人口满时卖谁换人由 swap 题决定
        rules.route_by_laya = "route" in self.kinds      # f9k：锁哪条官方路线由 route 题决定（规则不再自动锁）
        rules.hand_by_laya = "sell_hand" in self.kinds   # f9lx：手里的英雄卖不卖由 sell_hand 题决定（规则的"闲置手牌"价值判断让出）
        rules.lock_by_laya = "shop" in self.kinds and self.F.ACTIVE >= "f9d"   # 锁店由商店题的 lock 选项决定
        self._swap_todo: dict = {}                       # (回合, 等着上场的英雄) -> 要卖的人（"" = 不换）
        # **定主C / 换主C（carry）与买谁换人（swap）交给局面判断网络 cgnet**（2026-09-30 定分工），
        # 其余题型仍由 Laya 答。加载失败 = None → 这两题也退回 Laya。
        self.cg = None
        try:
            from agent.decision.cgnet_policy import CGNetPolicy
            self.cg = CGNetPolicy.load(run_dir=run_dir)
            if self.cg:
                logger.info("局面判断网络已加载：接管 %s", sorted(self.cg.kinds))
        except Exception:                                           # noqa: BLE001
            logger.exception("局面判断网络加载失败 —— carry/swap 退回 Laya")
            self.cg = None
        self._last_model = "laya"

    def set_run_dir(self, run_dir: Path) -> None:
        self.out = Path(run_dir) / "laya_points.jsonl"
        if getattr(self, "cg", None):
            self.cg.set_run_dir(run_dir)

    # ------------------------------------------------------------ 上下文
    def observe(self, state) -> None:
        if state.phase in ("lobby", "mode_lobby", "match_found", "hero_pick"):
            self.ctx, self.auction_round, self._seen = {}, -1, set()
            # **按回合记"问过了"的缓存也要清**（2026-09-26 十连打第 8 局：连开模式下 _carry_seen 带着上一局的回合号，
            # 整局一道换主C 题都没出 —— 曜（1 阶过渡件）绑着枫叶舞当了 12 回合主C，场上 5 阶 fit 9 的海诺没人理）
            for k in ("_carry_seen", "_route_seen", "_use_seen", "_use_done", "_carry_hist", "_carry_prev_choice", "_board_mem", "_laya_bought"):
                self.__dict__.pop(k, None)
            self._swap_todo = {}
            return
        if state.phase == "prepare":
            self.ctx = self.B.ctx_from_state(state.to_dict(), self.ctx)
            # **每个准备帧**都更新"沿用读数"的记忆（离线 extract_run 就是逐帧沿用的）。原来只在出题时更新 ——
            # 出题都在商店开着时，棋盘被盖住读不到等级，记忆里就永远没有等级：2026-09-25 第四轮第 2 局，
            # 感知读到了 18/24/30 级，学生题面里却全是 `lv ?`（训练-推理不一致）。
            try:
                self._board_mem = self.B.sticky_board(getattr(self, "_board_mem", None), self.rules.board_facts(state))
            except Exception:                                       # noqa: BLE001
                pass
        elif state.phase == "auction":
            self.auction_round = state.round_no or self.ctx.get("round") or 0

    def _c(self, state) -> dict:
        rnd = state.round_no or self.ctx.get("round") or 0
        try:
            field = self.rules.board_facts(state)
        except Exception:                                           # noqa: BLE001
            field = {n: {} for n in (self.rules._field or {})}
        field = self.B.sticky_board(getattr(self, "_board_mem", None), field)   # 商店开着读不到的沿用上次读数
        self._board_mem = field
        ch = self.__dict__.setdefault("_carry_hist", self.B.CarryHistory())   # f9g：主C 的来历（与离线 extract_run 同一套）
        ch.update(self.rules._carry_piece or "", rnd)
        ch.see_board(list(field), state.level if state else None)        # f9lw ㊳：弈星在场期间棋手升过几级
        return dict(self.ctx, **ch.ctx(), field=field, carry=self.rules._carry_piece or "", round=rnd,
                    auction_done=(self.auction_round == rnd),
                    maple=getattr(self.rules, "_maple_bound", None),   # f6：枫叶舞绑定对象（记账事实）
                    eco_plan=self.rules.eco_says(state) if state else None,   # f9lu：本回合经济网络的决定（商店题要看得到）
                    **{"route": "", "route_since": 0, **self.rules.route_ctx()},   # f9k：锁定的官方路线
                    lord_boosts=sorted(getattr(self.rules, "_lord_boosts", set())),   # f8：已选的枫叶舞强化
                    talents=sorted(getattr(self.rules, "_talents_taken", set())),     # f9li：本局已选的天赋
                    worn={h: sorted(v) for h, v in (getattr(self.rules, "_worn", {}) or {}).items() if v},   # f9ly：穿戴账本（拆装备写装备名）
                    laya_bought=dict(self.__dict__.get("_laya_bought") or {}),        # f9za：Laya 主动买的新英雄 → 回合（换人题写 bought this round）
                    eco=self.rules.eco_says(state))                                    # f9ln：经济网络本回合的决定

    def _meta(self, state) -> dict:
        ts = time.time()
        return {"run": self.out.parent.name if self.out else "", "ts": ts, "round": state.round_no or self.ctx.get("round") or 0}

    # ------------------------------------------------------------ 推理
    def _ask(self, p: dict) -> tuple[str, dict, float]:
        self._last_model = "laya"
        if getattr(self, "cg", None) and p.get("kind") in self.cg.kinds:
            try:
                r = self.cg.ask(p)
            except Exception:                                       # noqa: BLE001
                logger.exception("局面判断网络答题失败 —— 这题退回 Laya")
                r = None
            if r:
                self._last_model = "cgnet"
                return r
        # 2026-10-17 关掉：r9zq / r9zr 训练数据一道带 "SIM #" 前缀的题都没有（rerender 只在有预算打分 ENGINE_ANNOT 时才加），
        # 线上却每题现算推演器加前缀 → 训练 / 线上题面不一致；10-16 起推演器只辅助出题、不进线上。
        # 以后要重新接推演器结论：训练集同样加前缀后，把 SIM_PREFIX_LIVE 打开。
        if SIM_PREFIX_LIVE and self.F.engine_text_on() and p.get("kind") in ENGINE_KINDS:
            try:
                ann = p.get("_engine")
                if ann is None:
                    from agent.decision.cgnet_policy import engine_annot
                    ann = engine_annot(p)
                p = dict(p, question=dict(p["question"], criteria=self.F.engine_prefix(p["question"]["criteria"], ann)))
            except Exception:                                       # noqa: BLE001
                logger.exception("f9lo 推演器结论失败（这题不加）")
        sig = json.dumps([p["state"], p["question"]], ensure_ascii=False, sort_keys=True)
        if sig in self._cache:
            return self._cache[sig]
        t0 = time.perf_counter()
        ans = self.agent.system_one(p["state"], {"q": p["question"]})["answers"]["q"]
        probs = ans["probabilities"]
        choice = max(probs, key=probs.get)
        res = (choice, probs, float(probs[choice]), round((time.perf_counter() - t0) * 1000, 1))
        self._cache[sig] = res
        if len(self._cache) > 256:
            self._cache.pop(next(iter(self._cache)))
        return res

    def _log(self, p: dict, choice: str, probs: dict, conf: float, ms: float, rule_key: Optional[str],
             used: str, why: str = "") -> None:
        key = (p["round"], p["kind"], json.dumps(p["question"]["criteria"], ensure_ascii=False, sort_keys=True))
        if key in self._seen:
            return
        self._seen.add(key)
        rec = dict(p, id=hashlib.md5(f"{p['run']}|{p['ts']}|{p['kind']}".encode()).hexdigest()[:12],
                   feature_version=self.F.ACTIVE, laya=probs, laya_choice=choice,
                   laya_conf=round(conf, 4), laya_ms=ms, rule_choice=rule_key, used=used, why=why,
                   ckpt=self.ckpt, model=getattr(self, "_last_model", "laya"))
        try:
            if self.out is None:
                return
            with self.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:                                           # noqa: BLE001
            logger.exception("写 laya_points.jsonl 失败")
        self._patch_decision(p, choice, used)
        logger.info("%s[%s] 选 %s（%.2f，%.0fms）规则会选 %s → 执行 %s%s",
                    "局面网络" if getattr(self, "_last_model", "laya") == "cgnet" else "Laya", p["kind"], choice, conf, ms,
                    rule_key, used, f"（{why}）" if why else "")

    def _patch_decision(self, p: dict, choice: str, used: str) -> None:
        """decisions.jsonl 记的是规则那一刻的选择（rules.record_decision）；模型接管时改成实际执行的那个（10-21 审查：
        847 局 4 次天赋全记成规则的选择，R2 刷新没记）。原规则选择留在 rule_chosen。"""
        kind = {"talent": "talent", "card_pick": "card_pick"}.get(p.get("kind"))
        if not kind or used not in ("laya", "cgnet") or not str(choice or "").startswith(("opt", "refresh")):
            return
        decs = getattr(self.rules, "decisions", None) or []
        rec = next((d for d in reversed(decs) if d.get("kind") == kind
                    and (d.get("state") or {}).get("round") == p.get("round")), None)
        if rec is None or rec.get("executed_by"):
            return
        if choice == "refresh":
            name = "refresh"
        else:
            i = int(choice[3:]) if choice[3:].isdigit() else -1
            opts = rec.get("options") or []
            name = (opts[i] or {}).get("name") if 0 <= i < len(opts) and isinstance(opts[i], dict) else None
        if name and name != rec.get("chosen"):
            rec["rule_chosen"] = rec.get("chosen")
            rec["chosen"] = name
        rec["executed_by"] = used

    # ------------------------------------------------------------ 各类决策
    _TIES = None

    @classmethod
    def _talent_ties(cls) -> dict:
        """data/talent_hero_ties.json：天赋 → 送 / 点名的英雄（10-19 起打标记）。"""
        if cls._TIES is None:
            try:
                cls._TIES = json.loads((Path(__file__).resolve().parents[2] / "data" / "talent_hero_ties.json").read_text(encoding="utf-8"))
            except Exception:                                   # noqa: BLE001
                cls._TIES = {}
        return cls._TIES

    def pick_lord_boost(self, state, options):
        """rules.decide_lord_boost 的钩子：在规则已排除结构性废牌的候选里选。"""
        try:
            opts = [(self.F.clean_name(o.name), o.desc or "") for o in options]
            if _IR.on("maple_cc_immune"):
                # 临时规则 maple_cc_immune（2026-10-19：主C 必须有免控装备，免控也可以用枫叶舞补；
                # 装备仍优先吸血装）：主C 身上没有免控装（炽热支配者 / 星源装甲 / 机器头）、还没选过枫影无拘 → 有枫影无拘就选
                from agent.decision import hero_traits as _ht
                r = self.rules
                _worn = set(r._worn.get(r._carry_piece) or ())
                if (r._carry_piece and not (_worn & (_ht.carry_cc_items(r._carry_piece) | {"机器头"}))
                        and _ht.CC_IMMUNE_MAPLE not in (getattr(r, "_lord_boosts", set()) or set())):
                    for i, (nm, _d) in enumerate(opts):
                        if nm == _ht.CC_IMMUNE_MAPLE:
                            logger.info("主C %s 没有免控装 → 枫叶舞强化选 %s（临时规则 maple_cc_immune）", r._carry_piece, nm)
                            return options[i]
            if _IR.on("maple_temp_host"):
                # 4 阶主C 路线（韩信 / 刘邦 / 蒙犽）：枫叶舞挂在刻意养的临时主C（干将莫邪 / 孙膑）上时选红叶染锋，换绑保留 50% 等级
                # （2026-10-20 晚：4 阶主C 路线都可以先用干将或孙膑当主C，枫叶天赋选红叶染锋保留 50% 等级）
                from agent.decision import lord_config as _LCb
                r = self.rules
                _cb = _LCb.load()
                _rc = r.route_carry()
                _tc = _cb.temp_carry_for(_cb.tier(_rc)) if (_cb.present and _rc) else {}
                if _tc and r._maple_bound in (_tc.get("heroes") or []) and _tc.get("maple_boost") not in (getattr(r, "_lord_boosts", set()) or set()):
                    for i, (nm, _d) in enumerate(opts):
                        if nm == _tc.get("maple_boost"):
                            logger.info("4 阶路线（%s）临时主C %s 挂枫叶舞 → 强化选 %s（临时规则 maple_temp_host）", _rc, r._maple_bound, nm)
                            return options[i]
            p = self.B.lord_boost_point(opts, self._c(state), self._meta(state))
            if not p:
                return None
            if _IR.on("no_red_leaf"):
                # 临时规则 no_red_leaf（2026-10-21 定：红叶染锋不选）：枫叶舞强化三选一去掉红叶染锋（还剩别的选项时）
                _drop = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(opts)
                         and opts[int(k[3:])][0] == "红叶染锋"]
                if _drop and len(_drop) < len([k for k in p["question"]["criteria"] if k.startswith("opt")]):
                    logger.info("枫叶舞强化去掉红叶染锋（临时规则 no_red_leaf）")
                    self._restrict(p, _drop)
            choice, probs, conf, ms = self._ask(p)
            self._log(p, choice, probs, conf, ms, None, "laya")
            return options[int(choice[3:])]
        except Exception:                                           # noqa: BLE001
            logger.exception("Laya 枫叶舞强化选择失败，回退规则")
            return None

    def _shadow_use_card(self, state) -> None:
        """**特殊效果牌的影子题**（2026-09-25：特殊效果牌最终要交给 Laya 决定）：
        手里有需要"用在谁"判断的牌（变身/移除/觉醒进度/限制性进化/遗志…），每回合每张出一道 use_card 题，
        学生的答案只记进 laya_points（used="shadow"），**不执行** —— 规则照常用牌。攒够题、老师标注、学生学会后再接管。"""
        if state.phase != "prepare" or state.shop_open:
            return
        seen = self.__dict__.setdefault("_use_seen", set())
        X = self.F._facts()
        c = None
        for slot in state.hand_slots or []:
            name = self.F.clean_name(slot.get("name", ""))
            if not name or (state.round_no, name) in seen or not X.special_card(name):
                continue
            seen.add((state.round_no, name))
            c = c or self._c(state)
            p = self.B.use_card_point(name, c, self._meta(state))
            if not p:
                continue
            choice, probs, conf, ms = self._ask(p)
            self._log(p, choice, probs, conf, ms, None, "shadow", "影子题：只记录，不执行")

    def _real_carry(self, state=None) -> str:
        """真主C（10-20 review-interim-conflicts 第 8 条）：现任是 1 阶临时主C 时不算 → 路线已锁取路线主C，否则空。"""
        r = self.rules
        cur = r._carry_piece or ""
        if cur and int((r.pieces.get(cur) or {}).get("quality") or 0) >= 2:
            return cur
        return (r.route_carry() or "") if r.route_locked() else ""

    @staticmethod
    def _restrict(p: dict, drop: list) -> None:
        """从一道题里去掉几个选项（题面其它部分不动；学生只在剩下的选项里选）。"""
        for k in drop:
            p["question"]["criteria"].pop(k, None)
            (p.get("raw") or {}).get("names", {}).pop(k, None)
        p["options"] = list(p["question"]["criteria"])
        p["n_opt"] = len(p["options"])

    ROUTE_ROUNDS = (4, 8)    # 2026-10-21：等第一次拍卖后再决定主C（第一次拍卖出狄仁杰的概率不低）→ 第 4 回合（拍卖在回合开头）准备阶段才开始问；原 (3, 8)

    LOW_VALUE_TALENTS = ("节能商店", "限制性进化", "随机成长", "随机强化", "随机进化", "成年礼", "群星璀璨", "兵行诡招", "极限操作", "搏命", "拾荒者")     # talent_low_value：10-19 判为低价值（限制性进化：移除的是不需要的低等级英雄，换随机高 2 阶牌，不练主C；群星璀璨：要场上 ≥5 名 4-5 阶，单核阵容凑不出，与单核玩法不适配）
    STACK_COPIES = ("明世隐",)     # 登场效果随同名叠加、核心阵容里有多少买多少（2026-10-21）

    @property
    def WEAK_CARRIES(self) -> tuple:
        """关掉的主C：读棋手配置（data/lords/阿离/carry_pool.json）；没有配置时退回旧名单。
        路线题 / 换主C 题都按它去掉，但删空时不删（主C 位永不留空，第 849 局教训）。"""
        from agent.decision import lord_config as _LC
        _c = _LC.load()
        # 10-21 审查：1 阶的（干将 / 孙膑 / 曜 / 赵云）是临时主C，不按"关掉的主C"删（852a 局换主C 题删孙膑 17 次，临时主C 只剩孙尚香）
        return tuple(n for n in _c.disabled() if _c.tier(n) != 1) if _c.present else ("安琪拉", "马可波罗")
    WEAK_ROUTES = ("阿离日落海安琪拉", "双刃亚连")    # 10-20 晚第 849 局后：伞亚连 / 双剑出鞘伞亚连解除屏蔽    # route_avoid_weak：日落海安琪拉 2 局均 5.5；伞亚连 10-19 下午曾改为先屏蔽（前期杀不了人、枫叶舞升不了级）

    def _route(self, state) -> None:
        """**选哪条官方路线**（f9k，2026-09-28 起交给 Laya）：还没锁定时第 3~8 回合每回合问一次
        （时机同 carry 题：准备阶段、商店关着、人口读得到）。kinds 带 route 时照答案锁定（rules.set_route，只查合法性），
        否则只记录（影子题）。"""
        if state.phase != "prepare" or state.shop_open or not state.population_valid:
            return
        if _IR.on("route_lock_opening"):
            # 开局定路线（2026-10-20 晚：先定要玩的路线，再看这个主C 是几阶；开局锁定的就是最终路线）：
            # 第一个准备阶段按棋手配置的 opening_order 锁（开着、阵营没被禁的第一个组 / 主C），之后不再出路线题
            from agent.decision import lord_config as _LC
            _cfg = _LC.load()
            if not self.rules.route_locked() and _cfg.present:
                ch = _cfg.opening_choice(state.banned_faction or "")
                if ch and self.rules.set_route(ch["key"], state.round_no or 0,
                                               why=f"开局定路线：{ch['name']}（禁 {state.banned_faction or '无'}；临时规则 route_lock_opening）"):
                    self.rules._route_group = ch["members"] if ch["kind"] == "group" else ()
            self.rules.group_sync(state)
            if self.rules.route_locked():
                return
        if not (self.ROUTE_ROUNDS[0] <= (state.round_no or 0) <= self.ROUTE_ROUNDS[1]) or self.rules.route_locked():
            return
        seen = self.__dict__.setdefault("_route_seen", set())
        if state.round_no in seen:
            return
        seen.add(state.round_no)
        if _IR.on("route_lock_single_carry") and (state.round_no or 0) >= 4:
            # 10-23 第 971 局：禁河洛 R4~R7 路线题没出（候选 0 条），R8 亚连到手才锁 → 单一开局主C 直接按配置锁，不靠路线题
            try:
                from agent.decision import lord_config as _LCd
                _och = _LCd.load().opening_choice(state.banned_faction or "")
                if _och and _och.get("kind") == "carry" and self.rules.set_route(
                        _och["key"], state.round_no or 0,
                        why=f"开局主C {_och['name']}（禁 {state.banned_faction or '无'}；临时规则 route_lock_single_carry）"):
                    logger.info("开局主C %s → 直接锁定路线 %s（临时规则 route_lock_single_carry）", _och["name"], _och["key"])
                    return
            except Exception:                                   # noqa: BLE001
                logger.exception("route_lock_single_carry 直接锁定失败")
        p = self.B.route_point(self._c(state), self._meta(state))
        if not p:
            # 第 722 局：整局没出路线题（离线用同一回合的题面能出），记下原因再查
            try:
                _c = self._c(state)
                _B = self.F._facts().Board(_c)
                logger.info("路线题没出：第 %s 回合，题面 route=%r，候选 %d 条", state.round_no, _c.get("route"),
                            len(self.F._facts().route_candidates(_B, banned=_c.get("banned") or "")))
            except Exception:                                   # noqa: BLE001
                logger.exception("路线题没出（记录原因失败）")
            return
        if _IR.on("route_lock_single_carry") and (state.round_no or 0) >= 4:
            # 10-22（941 / 947 局禁河洛整局 flex）：开局配置是单一主C（亚连）→ 路线题里有他的路线就锁，不等他到手
            try:
                from agent.decision import lord_config as _LCs
                _och = _LCs.load().opening_choice(state.banned_faction or "")
                if _och and _och.get("kind") == "carry":
                    _crs = p["question"]["criteria"]
                    _mine = [k for k, v in _crs.items() if k.startswith("r") and f": carry {_och['name']} " in str(v)]
                    if _mine and len(_mine) < len(_crs):
                        self._restrict(p, [k for k in _crs if k not in _mine])
                        logger.info("开局主C %s（禁 %s）→ 路线题只留他的路线 %s（临时规则 route_lock_single_carry）",
                                    _och["name"], state.banned_faction or "无", [str(_crs[k]).split(":")[0] for k in _mine])
            except Exception:                                   # noqa: BLE001
                logger.exception("route_lock_single_carry 失败")
        if _IR.on("route_avoid_weak"):
            # 临时规则 route_avoid_weak（2026-10-19：段位卡在星耀 2；740~776 局伞亚连 5 局均 5.4、前三 0/5，
            # 日落海安琪拉 2 局均 5.5；守约系 9 局均 2.1）：这两条路线的选项先去掉（还剩别的路线或 flex 时）
            crit = p["question"]["criteria"]
            # 2026-10-19 夜第 804 局："日落海整备安琪拉（阿离版）"名字不在名单里漏网 → 主C 是安琪拉的路线一律去掉
            drop = [k for k, v in crit.items() if k.startswith("r") and (str(v).startswith(self.WEAK_ROUTES)
                                                                          or any(f": carry {c} " in str(v) for c in self.WEAK_CARRIES))]
            # 54.28-12（第 806 / 809 / 823 局：主C 已经是亚连 / 安琪拉，路线却全被屏蔽 → 整局 flex）：屏蔽只管"不新选他当主C"，
            # 现任主C（非 1 阶占位）就是这条路线的主C 时不去掉
            _cur = self.rules._carry_piece or ""
            if _cur and int((self.rules.pieces.get(_cur) or {}).get("quality") or 0) >= 2:      # 只剩安琪拉时她也会当主C → 照常锁她的路线
                drop = [k for k in drop if f": carry {_cur} " not in str(crit[k])]
            # 2026-10-22（906 / 915 局：禁的是逐鹿，河洛组能走，R4 手里有亚连就锁了伞亚连，两局 5 / 6 名；
            # 亚连线 865 局起 12 局 0 次前三，河洛线同期均 ~2.5）：开局首选组还能走时，第 6 回合前只留首选组的路线（伞亚连只作禁河洛的退路）
            try:
                from agent.decision import lord_config as _LCr, routes as _Rr
                _oc = _LCr.load().opening_choice(state.banned_faction or "")
                _mem = set((_oc or {}).get("members") or ())
                if _oc and _oc.get("kind") == "group" and (state.round_no or 0) <= 8 and _cur not in _mem:   # 10-23：6 → 8（非禁河洛退到亚连 5 局均 4.2，守约 3.27 / 沈梦溪 3.64）
                    # 10-22 第 935 局（禁大河）：R4 亚连当上现任主C → 原来"现任主C 的路线不去掉"把伞亚连留下并锁了（5 名）；
                    # 首选组还能走时，现任主C 不在组里也先不锁他的路线（留 flex，等组里的主C）
                    _rk = (p.get("raw") or {}).get("keys", {})
                    _off = [k for k in crit if k.startswith("r") and k not in drop
                            and ((_Rr.by_key(_rk.get(k)) or {}).get("carry") or "") not in _mem]
                    if _off:
                        logger.info("开局首选组 %s 还能走 → 第 8 回合前先不锁组外路线 %s", (_oc or {}).get("name"), [str(crit[k]).split(":")[0] for k in _off])
                        drop = drop + _off
                if _oc and _oc.get("kind") == "group" and _cur in _mem:
                    # 10-23 第 975 局：玄策（河洛三C 组员）R5 已是主C，R7 路线题 0.37 锁了狄仁杰、卖玄策 →
                    # 10-20 定"主C 组谁先出用谁"：现任主C 是组员 → 路线题只留他的路线（不限回合）
                    _rk2 = (p.get("raw") or {}).get("keys", {})
                    _his = [k for k in crit if k.startswith("r") and ((_Rr.by_key(_rk2.get(k)) or {}).get("carry") or "") == _cur]
                    _oth = [k for k in crit if k not in _his and k not in drop] if _his else []
                    if _oth:
                        logger.info("现任主C %s 是开局组 %s 的组员 → 路线题只留他的路线（去掉 %s）", _cur, _oc.get("name"),
                                    [str(crit[k]).split(":")[0] for k in _oth])
                        drop = drop + _oth
                if _oc and _oc.get("kind") == "group" and _IR.on("no_yalian_unless_heluo_ban"):
                    # 10-23 试行：没禁河洛时退到亚连 6 局均 4.3（守约 3.3 / 沈梦溪 3.6）→ 没禁河洛的局不锁亚连路线，等河洛三C / 狄仁杰
                    _rk3 = (p.get("raw") or {}).get("keys", {})
                    _nlo = self.rules.no_lock_outside_group(state)
                    _ya = [k for k in crit if k.startswith("r") and k not in drop
                           and ((_Rr.by_key(_rk3.get(k)) or {}).get("carry") or "") in _nlo]
                    if _ya:
                        logger.info("没禁河洛 → 路线题去掉亚连路线 %s（临时规则 no_yalian_unless_heluo_ban）", [str(crit[k]).split(":")[0] for k in _ya])
                        drop = drop + _ya
                if _IR.on("preferred_route_per_carry"):
                    # 10-23 第 997 局：同一主C 的几条路线战绩差很多（守约 一枪清屏 2.3 vs 无限狙 4.7）→ 只留该主C 的首选路线（棋手配置）
                    _pref = _LCr.rule("preferred_route_by_carry", {}) or {}
                    _rk4 = (p.get("raw") or {}).get("keys", {})
                    _cof = {k: ((_Rr.by_key(_rk4.get(k)) or {}).get("carry") or "") for k in crit if k.startswith("r")}
                    _alt = [k for k, c in _cof.items() if k not in drop and c in _pref and str(_rk4.get(k)) != str(_pref[c])
                            and any(str(_rk4.get(k2)) == str(_pref[c]) for k2 in crit if k2 not in drop)]
                    if _alt:
                        logger.info("同主C 只留首选路线 → 去掉 %s（临时规则 preferred_route_per_carry）", [str(crit[k]).split(":")[0] for k in _alt])
                        drop = drop + _alt
            except Exception:                               # noqa: BLE001
                logger.exception("首选组路线过滤失败")
            if drop and len(drop) < len(crit):
                logger.info("路线题去掉战绩差的路线 %s（临时规则 route_avoid_weak）", [str(crit[k]).split(":")[0] for k in drop])
                self._restrict(p, drop)
        if _IR.on("route_avoid_weak"):
            # 绑定主C 的天赋（data/talent_conditions.json 的 commit_route_carry；2026-10-20：幸运硬币是狄仁杰路线的神天赋，
            # 选了这个天赋就一定要玩狄仁杰，不能再选其他主C）：已选了它 → 路线题只留该主C 的路线（有的话）
            from agent.decision import talent_conditions as _tcr
            _commit = {(_tcr.table().get(t) or {}).get("commit_route_carry") for t in self.rules._talents_taken} - {None, ""}
            _cfac = {(_tcr.table().get(t) or {}).get("commit_route_faction") for t in self.rules._talents_taken} - {None, ""}
            if _cfac and not _commit:
                # 10-22 第 906 局（幸运硬币）：吃古币的天赋已选 → 路线题优先留该阵营主C 的路线（有的话）
                crit = p["question"]["criteria"]
                _rk = (p.get("raw") or {}).get("keys", {})
                from agent.decision import routes as _Rcf
                keep = [k for k in crit if k.startswith("r") and self.rules._faction(((_Rcf.by_key(_rk.get(k)) or {}).get("carry") or "")) in _cfac]
                if keep and len(keep) < len(crit):
                    logger.info("已选绑阵营的天赋（%s）→ 路线题只留该阵营的路线", sorted(_cfac))
                    self._restrict(p, [k for k in crit if k not in keep])
                elif not keep and "flex" in crit and (state.round_no or 0) <= 8:
                    # 10-22 第 906 局：这局走狄仁杰路线本可以赢——R4 题里只有伞亚连（狄仁杰还没到手），锁了亚连、幸运硬币白拿
                    logger.info("已选绑阵营的天赋（%s）但题里没有该阵营路线 → 第 8 回合前先不定路线，等主C 到手", sorted(_cfac))
                    self._restrict(p, [k for k in crit if k != "flex"])
            if _commit:
                crit = p["question"]["criteria"]
                keep = [k for k, v in crit.items() if k.startswith("r") and any(f": carry {c} " in str(v) for c in _commit)]
                if keep:
                    drop = [k for k in crit if k not in keep]
                    if drop:
                        logger.info("已选绑定主C 的天赋（%s）→ 路线题只留 %s 的路线", sorted(_commit), [str(crit[k]).split(':')[0] for k in keep])
                        self._restrict(p, drop)
        choice, probs, conf, ms = self._ask(p)
        key = (p.get("raw") or {}).get("keys", {}).get(choice, "")
        if "route" not in self.kinds:
            self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "shadow", "影子题：只记录，不执行")
            return
        if choice == "flex" or not key:
            self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya", "先不定路线")
            return
        if _IR.on("route_lock_needs_carry"):
            # 临时规则 route_lock_needs_carry（第 739 局：R4 以 0.36 锁守约路线，守约整局没来，实际阵容走成日落海，9/9 的海诺被当路线外 bench 两次）：
            # "定好路线" = 路线主C 已到手（场上 / 手里）；没到手先不锁，下回合再问
            try:
                from agent.decision import routes as _R
                _rc = (_R.by_key(key) or {}).get("carry") or ""
            except Exception:                                   # noqa: BLE001
                _rc = ""
            _own = set(self.rules._field) | {sl.get("name") for sl in state.hand_slots or []}
            _single = False
            if _IR.on("route_lock_single_carry"):
                # 10-22 第 964 局：单一开局主C（禁河洛 → 亚连）的路线不等他到手就锁（route_lock_single_carry），这里不再挡
                try:
                    from agent.decision import lord_config as _LCn
                    _och = _LCn.load().opening_choice(state.banned_faction or "")
                    _single = bool(_och and _och.get("kind") == "carry" and _och.get("name") == _rc)
                except Exception:                               # noqa: BLE001
                    _single = False
            if _rc and _rc not in _own and not _single:
                self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "rule", f"路线主C {_rc} 还没到手，先不锁（临时规则 route_lock_needs_carry）")
                logger.info("路线题选了 %s（主C %s），但主C 还没到手 → 先不锁", key, _rc)
                return
        if _IR.on("preferred_route_per_carry"):
            # 10-23 第 998 局：候选里没有首选路线（伞猫无cd），锁了同主C 的无限炸弹猫 → 选中的路线主C 有首选路线就换成首选
            try:
                from agent.decision import lord_config as _LCp, routes as _Rp
                _pc = ((_Rp.by_key(key) or {}).get("carry") or "")
                _pk = (_LCp.rule("preferred_route_by_carry", {}) or {}).get(_pc)
                if _pk and str(_pk) != str(key) and _Rp.by_key(str(_pk)):
                    logger.info("路线主C %s 有首选路线 %s → 锁首选，不锁 %s（临时规则 preferred_route_per_carry）",
                                _pc, (_Rp.by_key(str(_pk)) or {}).get("name"), (_Rp.by_key(key) or {}).get("name"))
                    key = str(_pk)
            except Exception:                                   # noqa: BLE001
                logger.exception("首选路线替换失败")
        ok = self.rules.set_route(key, state.round_no, why=f"Laya {conf:.2f}")
        self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya" if ok else "rule",
                  f"锁定路线 {key}" if ok else f"路线 {key} 不合法（不在阵容库 / 在排除名单）")

    def _shadow_eco(self, state) -> None:
        """**经济节奏（eco）影子题**（2026-10-09）：每回合准备阶段第一帧商店开着时问一次
        局面判断网络"升级 / 停级刷牌 / 攒钱"，只记进 laya_points（used=shadow），**不执行**。只在部署包里有 eco 模型时问
        （Laya 没学过这道题，不拿它答）。跑几局看分歧再决定接管。"""
        if state.phase != "prepare" or not state.shop_open or not (getattr(self, "cg", None) and "eco" in self.cg.kinds):
            return
        seen = self.__dict__.setdefault("_eco_seen", set())
        if state.round_no in seen:
            return
        seen.add(state.round_no)
        # eco 题挂在题面版本 f9le 下，而线上 Laya 学生（R17）用的是旧版本 —— 出这道题时临时切到 eco 版本，出完切回
        # （2026-10-10 第 488 局：版本不够，q_eco 一直返回 None，影子题一次没出）
        old_v = self.F.ACTIVE
        try:
            self.F.ACTIVE = max(old_v, getattr(self.F, "ECO_VERSION", old_v))
            _em = getattr(self.cg, "models", {}).get("eco") if getattr(self, "cg", None) else None
            if _em is not None and (getattr(_em, "manifest", None) or {}).get("eco_core_feat"):
                # 2026-10-13 cs_r9zc 是按 f9ln 的缺件名单（[MIN ENGINE core] 在前、自定义路线不混官方队友）训练的 →
                # 出 eco 题时题面版本至少 f9ln，否则 raw.missing 和训练不一致（Laya 学生仍是 f9lm 时也要切）
                self.F.ACTIVE = max(self.F.ACTIVE, "f9ln")
            p = self.B.eco_point(self._c(state), self._meta(state))
        finally:
            self.F.ACTIVE = old_v
            self.F._facts()                 # facts.FV 跟回学生版本
        if not p:
            return
        choice, probs, conf, ms = self._ask(p)
        if _IR.on("eco_behind_schedule") and choice in ("roll", "save", "levelup") and state.level:
            # 临时规则（2026-10-19 第 755 局：3 级卡到第 11 回合 —— R5~R8 升不起却每回合选 roll 把钱刷光，R9/R10 升得起仍 0.51/0.48 选 roll；
            # 这局打得很差）：落后进度线（meta.level_target）时，升得起 → 升；升不起但攒一回合够 → 攒
            from agent.decision import meta as _meta
            tgt = _meta.level_target(state.round_no or 0)
            cost, gold = state.level_cost or 0, state.gold or 0
            if state.level < tgt and cost > 0:
                nxt = max(0, cost - _meta.lord_xp_gain(state.round_no or 0))
                want = "levelup" if gold >= cost else ("save" if gold + _meta.energy_income((state.round_no or 0) + 1) >= nxt else choice)
                if want != choice:
                    logger.info("落后进度（%s 级 < 进度线 %s 级）：经济网络选 %s → 改 %s（能量 %s / 升级费 %s，临时规则 eco_behind_schedule）",
                                state.level, tgt, choice, want, gold, cost)
                    choice = want
        if _IR.on("engine_tier_eco") and choice in ("roll", "save", "levelup") and state.level and self.rules.route_locked():
            # 临时规则 engine_tier_eco（2026-10-19：增长引擎刷出太慢，伞亚连的雅典娜来得太晚；第 779 局：雅典娜 4 阶，
            # R6~R8 停 3 级（4 阶 0%）、R4~R8 只刷 3 次，R11 才首次出现）：路线已锁、核心引擎缺 t 阶件 →
            # 棋手等级 < t：升得起就升、升不起就攒；等级 ≥ t：经济网络选 save 时改 roll（就在这一级多刷）
            from agent.decision import meta as _meta
            r = self.rules
            _own = set(r._field) | {sl.get("name") for sl in state.hand_slots or []}
            _eng = [n for n in ((r.min_engine().get("early_priority_full") or r.min_engine().get("early_priority")) or []) if n not in _own and n != r._carry_piece]
            _tiers = [int((r.pieces.get(n) or {}).get("quality") or 0) for n in _eng]
            _tiers = [t for t in _tiers if t > 0]
            if _tiers:
                t = max(_tiers)           # 按缺的最高品阶（伞亚连缺米莱狄 2 阶 + 雅典娜 4 阶 → 先到 4 级；4 级照样刷得到 2 阶）
                cost, gold = state.level_cost or 0, state.gold or 0
                want = choice
                if state.level < min(t, 5) and cost > 0:
                    nxt = max(0, cost - _meta.lord_xp_gain(state.round_no or 0))
                    want = "levelup" if gold >= cost else ("save" if gold + _meta.energy_income((state.round_no or 0) + 1) >= nxt else choice)
                elif state.level >= t and choice == "save" and not (
                        hasattr(_meta, "level_target") and state.level < (_meta.level_target(state.round_no or 0) or 0)):
                    # 落后进度线时不把"攒钱升级"翻成"刷牌"（review-interim-conflicts 第 2 条：和 eco_behind_schedule 打架，钱刷光升不了级）
                    want = "roll"
                if want != choice:
                    logger.info("引擎还缺 %s（最高 %d 阶，棋手 %d 级）：经济网络选 %s → 改 %s（临时规则 engine_tier_eco）",
                                _eng, t, state.level, choice, want)
                    choice = want
        if self.rules.ECO_TAKEOVER and choice in ("levelup", "roll", "save"):
            self.rules.set_eco(state.round_no, choice)
            self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "cgnet", f"经济节奏接管：本回合 {choice}")
            logger.info("局面网络[eco] 接管：第 %s 回合 %s（%.2f）%s", state.round_no, choice, conf,
                        {k: round(v, 2) for k, v in probs.items()})
            return
        self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "shadow", "影子题：只记录，不执行")
        logger.info("局面网络[eco] 影子：%s（%.2f）%s", choice, conf, {k: round(v, 2) for k, v in probs.items()})

    def _shadow_timing(self, state) -> None:
        """**"现在用 / 先留着"影子题**（2026-10-15：把写死的规则改成题）：审定用牌表的 when 条件没满足、规则把牌扣着时，
        每张牌每回合问一次 Laya，只记进 laya_points（used=shadow），攒训练数据；规则的 when 仍是兜底（interim card_usage_when）。"""
        if state.phase != "prepare" or state.shop_open:
            return
        held = self.rules.__dict__.get("_timing_held") or {}
        asked = self.__dict__.setdefault("_timing_asked", set())
        for (rnd, name), why in list(held.items()):
            if rnd != state.round_no or (rnd, name) in asked:
                continue
            asked.add((rnd, name))
            c = self._c(state)
            if "累积" in (self.F._facts().skill(name) or ""):
                cnt, pw, nb = self.rules.contract_estimate(state, name)
                c["contract"] = {"count": cnt, "p": round(pw, 3), "battles": nb}
            old_v = self.F.ACTIVE
            try:
                self.F.ACTIVE = max(old_v, "f9lt")
                self.F._facts()
                p = self.B.use_timing_point(name, c, self._meta(state))
                if not p:
                    continue
                choice, probs, conf, ms = self._ask(p)
                if why.startswith("时机交给 Laya"):
                    # f9lw ㊵：when=laya_timing 的牌（点兵点将）由这道题真的决定：now → 规则这回合照用牌表的目标用掉
                    if choice != "now" and name == "点兵点将" and _IR.on("dbdj_core_use"):
                        # 临时规则（2026-10-16 晚开）：第 670 局核心 5/6、能让守约当场觉醒，Laya 四回合都 0.55~0.59 判"留着"
                        import re as _re
                        _t = (p["question"]["criteria"].get("now") or "")
                        _m = _re.search(r"CORE (\d+)/(\d+)", _t)
                        _carry = self.rules._carry_piece or ""
                        _awk = bool(_carry) and _re.search(r"awakens NOW:[^;]*" + _re.escape(_carry), _t) is not None
                        if (_m and int(_m.group(1)) >= 5) or _awk:
                            _why = f"能让主C {_carry} 当场觉醒" if _awk else f"场上核心 {_m.group(1)}/{_m.group(2)}"
                            logger.info("临时规则 dbdj_core_use：点兵点将%s → 现在用（Laya 判留着 %.2f）", _why, conf)
                            choice = "now"
                    if choice == "now":
                        self.rules.__dict__.setdefault("_timing_go", set()).add((rnd, name))
                    logger.info("Laya[use_timing] %s：%s（%.2f）", name, "现在用" if choice == "now" else "先留着", conf)
                    self._log(p, choice, probs, conf, ms, "hold", "laya", f"时机交给 Laya（{name}）")
                    continue
                self._log(p, choice, probs, conf, ms, "hold", "shadow", f"影子题：规则按审定表扣着（{why}）")
            finally:
                self.F.ACTIVE = old_v
                self.F._facts()

    def _yield_single(self, p, state, must_yield) -> None:
        """必须让位、去掉"维持"后只剩一个合法接班人：没得选，直接换给他（合法性，不是价值判断）。
        2026-10-13 第 588 局：1 阶李元芳占位、唯一接班人沈梦溪 → 题只剩 1 个选项被丢弃，主C 一直换不下来。"""
        # 第 601 局：主C 空缺（1 阶干将过了前期被清掉）时同样只剩 1 个候选 → 题被丢弃，第 6~10 回合没有主C
        if not (must_yield or not self.rules._carry_piece) or "carry" not in self.kinds:
            return
        names = (p.get("raw") or {}).get("names", {})
        left = [names.get(k) for k in p["question"]["criteria"] if k != "keep" and names.get(k)]
        # 第 684 局 R6：Laya 刚锁"伞亚连"（主C 亚连，人在手牌），孙尚香过了前期必须让位，场上唯一合格的苍被自动接任 →
        # 整局养路线外的苍、亚连当第二个 C。路线是模型定的：必须让位时接班人就是路线主C（执行模型的决定，不是价值判断）；
        # 路线主C 这一帧在手牌里 → 先不接任，等他上场再问。
        try:
            rc = self.rules.route_carry() if self.rules.route_locked() else ""
        except Exception:  # noqa: BLE001
            rc = ""
        if rc and rc != self.rules._carry_piece:
            if rc in left and self.rules.apply_carry(rc, state, why="laya-only 让位给锁定路线的主C"):
                logger.info("必须让位：锁定路线的主C %s 在场 → 接任", rc)
                return
            if rc not in self.rules._field and rc in {sl.get("name") for sl in state.hand_slots or []}:
                logger.info("必须让位：锁定路线的主C %s 在手牌里 → 等他上场，不让 %s 接任", rc, left)
                return
            if rc in self.rules._field and rc not in left:
                # 第 858 局 R4：守约刚拖上场、回执没确认（出题时按不合法去掉）→ 只剩 1 阶孙膑"唯一接班人"，卖干将换绑给孙膑，下回合又让给守约
                logger.info("必须让位：锁定路线的主C %s 刚上场还没确认 → 等确认后再接任，不让 %s 接任", rc, left)
                return
        from agent.decision import lord_config as _LCy1
        _cy1 = _LCy1.load()
        if _cy1.present and len(left) == 1 and not _cy1.enabled(left[0]):
            # 第 878 局 R6：孙膑必须让位时亚连还在手里（刚买）→ 名单外的西施"唯一接班人"接任、枫叶舞也绑过去，下回合又换。
            # 名单内主C 已在手里 / 正在上场 → 先等他，不让名单外的人接任
            _new1 = {e.get("牌") for e in (self.rules.ledger or []) if e.get("event") in ("买牌", "获得卡牌")
                     and e.get("round") == state.round_no}                    # 只等本回合刚拿到的
            import time as _t11
            _soon1 = ({sl.get("name") for sl in state.hand_slots or []} & _new1) | {
                n for n, t0 in (self.rules.__dict__.get("_unconfirmed_deploy") or {}).items() if _t11.time() - t0 < 30}
            _lst = [n for n in _soon1 if n and _cy1.enabled(n) and _cy1.tier(n) >= 2]
            if _lst:
                logger.info("名单内主C %s 在手里 / 正在上场 → 不让名单外的 %s 接任，等他", _lst, left[0])
                return
        _q1 = lambda n: int((self.rules.pieces.get(n) or {}).get("quality") or 0)          # noqa: E731
        if len(left) == 1 and self.rules._carry_piece and _q1(self.rules._carry_piece) == 1 and _q1(left[0]) <= 1:
            logger.info("1 阶占位 %s 不让给另一个 1 阶的 %s（换了还要卖人换绑枫叶舞，白亏）", self.rules._carry_piece, left[0])
            return
        if len(left) == 1 and self.rules.apply_carry(left[0], state, why="laya-only 唯一接班人（1 阶占位让位 / 主C 空缺）"):
            logger.info("1 阶占位让位 / 主C 空缺，唯一合格候选 %s → 直接接任", left[0])

    def _shadow_carry(self, state) -> None:
        """**谁当主C 的影子题**（2026-09-25：出足够多的题，让 Laya 学会判断是否换主C）：每回合准备阶段（商店关着、
        人口读得到）出一题，学生答案只记进 laya_points（used="shadow"），**不执行** —— 主C 仍由规则 choose_carry 定。
        攒够题、老师标注、学生学会后再接管。"""
        if (state.phase == "prepare" and not state.shop_open and not state.population_valid and "carry" in self.kinds
                and not self.rules._carry_piece and self.rules._field):
            # **第 1 回合人口常读不到**（第 723 局：R1 孙膑上场，主C 空缺这步因为人口读不到整回合没跑，枫叶舞拖到 R2 才绑；
            # 第一回合没给枫叶舞一个临时主C）。"只有一个合格的人 → 由他接任"不需要人口读数（合法性，不是价值判断）
            ok = [n for n in self.rules._field if self.rules._carry_eligible(n, hard_only=self.rules.model_decides())]
            if len(ok) == 1 and self.rules.apply_carry(ok[0], state, why="laya-only 唯一合格（人口未读到）"):
                logger.info("主C 空缺（人口没读到），场上只有 %s 合格 → 由他接任", ok[0])
            return
        if state.phase != "prepare" or state.shop_open or not state.population_valid:
            return
        # 2026-10-12（第 584 局）：1 阶占位主C 必须让位，不然枫叶舞滚不起来 —— 让位是硬约束；
        # 让给谁交给局面网络（审查 10-12 B2：原来由规则按 路线主C > 同体系 > 评分 挑接班人）→ 出题时去掉"维持"
        # **主C 在手里等着重新上场（觉醒回手）时不出题**（2026-10-13 第 604 局：觉醒守约吃了 8 张古币，回手那一帧出了换主C 题，
        # 题里没有"维持"（他不在场上），局面网络只能从场上挑 → 换成李信，随后为换绑枫叶舞把重新上场的守约卖了）
        _cur = self.rules._carry_piece
        # 只认这一帧手牌里真读到他（不用"觉醒过"的记录：主C 真的没了时它不会自己清，会让换主C 题永远不出）
        if _cur and _cur not in self.rules._field and _cur in {sl.get("name") for sl in state.hand_slots or []}:
            return
        _succ = self.rules.placeholder_yield(state)
        from agent.decision import lord_config as _LCy
        _cy = _LCy.load()
        # 接班人必须是主C 名单里开着的（10-21：孙尚香 1 阶占位、亚连在场却一直不换）；关着的不强制换（避免又出现无主C）
        must_yield = bool(_succ) and (not _cy.present or _cy.enabled(_succ))
        if "carry" in self.kinds and _cur and _cur in self.rules._field                 and (self.rules.pieces.get(_cur, {}).get("quality") or 0) == 1:
            # **1 阶占位主C 让位给 Laya 锁定路线的主C**（第 717 局：R3 Laya 锁"枪有点饿了"（主C 百里守约），R4 守约上场，
            # 局面网络 4 次以 0.50~0.58 选"维持"孙尚香，枫叶舞一直在 1 阶身上；前期没有及时换主C）。
            # 10-12 定的硬约束"1 阶占位主C 必须让位" + 执行模型已做出的路线决定（锁路线 = 定了主C 是谁），不是价值判断
            try:
                rc = self.rules.route_carry() if self.rules.route_locked() else ""
            except Exception:  # noqa: BLE001
                rc = ""
            if (rc and rc != _cur and rc in self.rules._field and self.rules._carry_eligible(rc, hard_only=True)
                    and self.rules.apply_carry(rc, state, why="laya-only 1 阶占位让位给锁定路线的主C")):
                logger.info("1 阶占位主C %s 让位：Laya 锁定路线的主C %s 已在场 → 接任", _cur, rc)
                return
        if ("carry" in self.kinds and _IR.on("route_carry_takes_over") and self.rules.route_locked() and _cur
                and (not self.rules._committed_carry_offlist(_cur)       # 10-21 审查 1：认了的名单外主C 不被路线主C 顶掉
                     or self.rules.__dict__.get("_talent_route_carry"))):   # 10-22：天赋定死的路线（幸运硬币 → 狄仁杰）到了就换
            # 2026-10-18 定（最终目标是凑齐阵容推荐里的最终阵容；路线定好后不换）；第 752 局：R6 1 阶干将过期、主C 空缺那一刻
            # 只有虞姬合格 → 虞姬接任并换绑枫叶舞；同回合守约上场，R7 Laya 锁了守约路线，局面网络因枫叶舞在虞姬身上一直 keep，路线和主C 对不上到结束
            try:
                rc = self.rules.route_carry()
            except Exception:  # noqa: BLE001
                rc = ""
            # **主C 组谁先出用谁**（10-20 晚：沈梦溪 / 守约 / 玄策一组；第 850 局 R4 锁守约阵容、主C 已是沈梦溪，R5 守约到场接任、
            # 卖了沈梦溪和他身上的装备）：现任是同组的人、2 阶以上在场 → 路线跟现任走（换到他的阵容，只换一次），不让位
            from agent.decision import lord_config as _LCg
            _cg = _LCg.load()
            _g, _mem = _cg.group_of(_cur) if (_cg.present and _cur) else ("", ())
            if (rc and rc != _cur and _mem and rc in _mem and _cur in self.rules._field
                    and (self.rules.pieces.get(_cur, {}).get("quality") or 0) >= 2
                    and not self.rules.__dict__.get("_group_followed")):
                self.rules._group_followed = True
                if self.rules.set_route(_cg.lineup_key(_cur), state.round_no or 0, why=f"主C 组谁先出用谁：{_cur} 先到 → 路线跟他走"):
                    logger.info("路线主C %s 到了，但现任 %s 是同组（%s）先到的 → 路线换成 %s 的阵容，不让位", rc, _cur, _g, _cur)
                    return
            if (rc and rc != _cur and rc in self.rules._field and self.rules._carry_eligible(rc, hard_only=True)
                    and not (_mem and rc in _mem and _cur in self.rules._field
                             and (self.rules.pieces.get(_cur, {}).get("quality") or 0) >= 2)
                    and self.rules.apply_carry(rc, state, why="laya-only 锁定路线的主C 在场 → 接任（临时规则 route_carry_takes_over）")):
                logger.info("路线已锁、路线主C %s 在场，现任 %s 让位（临时规则 route_carry_takes_over）", rc, _cur)
                return
        seen = self.__dict__.setdefault("_carry_seen", set())
        # 场上的人变了（新英雄上场 / 有人卖掉）就再问一次，不等下一回合（2026-10-08：原来每次都等到下个回合才换）；
        # 主C 变了 / 空缺了也再问（审查 10-12 B1）
        # 第 723 局 R1：孙膑上场那一帧唯一合格 → apply_carry 因"拖拽回执还没确认"暂缓；2 秒后回执成功，但 key 没变、不再重试，
        # 整回合没主C、枫叶舞没绑。把"还没确认的上场"放进 key：回执一到 key 就变，下一帧重试
        key = (state.round_no, frozenset(self.rules._field), self.rules._carry_piece,
               frozenset((self.rules.__dict__.get("_unconfirmed_deploy") or {}).keys()))
        if key in seen:
            return
        seen.add(key)
        c = self._c(state)
        p = self.B.carry_point(c, self._meta(state))
        _cur0 = self.rules._carry_piece
        if p and _cur0 and _IR.on("route_avoid_weak"):
            # 10-22 第 947 局 R8（一回合换了两次主C）：禁河洛、亚连没到手，马可波罗（名单外）→ 海诺（名单外，还是亚连线的引擎件）；
            # 候选全是名单外的人时原来的"去掉名单外"因为会删空而不生效 → 现任主C 已有人时这题不问，等名单内主C
            from agent.decision import lord_config as _LCx
            _cx = _LCx.load()
            _nmx = (p.get("raw") or {}).get("names") or {}
            _cands = [n for k, n in _nmx.items() if k != "keep" and n]
            if _cx.present and _cands and not any(_cx.enabled(n) for n in _cands):
                logger.info("换主C 候选 %s 全在主C 名单外 → 不换，维持 %s（临时规则 route_avoid_weak）", _cands, _cur0)
                return
        if (p and _cur0 and "keep" in p["question"]["criteria"] and len(p["question"]["criteria"]) > 1
                and not self.rules._carry_eligible(_cur0, hard_only=True)):
            # 现任主C 违反硬约束（弈星 / 辅助坦克 / 后期 1 阶）→ 合法性：不能"维持"（第 642 局：开局弈星被定成主C）
            self._restrict(p, ["keep"])
            logger.info("现任主C %s 不能当主C（硬约束）→ 换主C 题去掉\"维持\"", _cur0)
        # 10-21 定"尽量不给，给了就认"：撤掉"名单外现任主C 让位给名单内主C"（第 869 / 876 局加的）——名单外 2 阶以上的人一旦当上主C 就认他、锁他的路线
        if p and _cur0 and int((self.rules.pieces.get(_cur0) or {}).get("quality") or 0) <= 1:
            # 2026-10-22 第 928 局 R4（卖了临时主C 后把主C 让给了孙尚香，场上明明有沈梦溪）：1 阶临时主C 只让给 2 阶以上的人，
            # 不在两个 1 阶之间互换（原来"临时主C 不互换"只记日志没执行）；锁定路线的主C 在手里 / 刚上场没确认 → 这题先不换，等他
            _nm = (p.get("raw") or {}).get("names") or {}
            _t1 = [k for k, n in _nm.items() if k != "keep" and n and int((self.rules.pieces.get(n) or {}).get("quality") or 0) <= 1]
            if _t1 and len(_t1) < len(p["question"]["criteria"]):
                _t1n = [_nm.get(k) for k in _t1]            # 先取名字再删（_restrict 会把 names 里的也删掉）
                self._restrict(p, _t1)
                logger.info("1 阶临时主C 不让给 1 阶的 %s", _t1n)
            _rc = self.rules.route_carry() if self.rules.route_locked() else ""
            _coming = _rc and _rc != _cur0 and (_rc in {sl.get("name") for sl in state.hand_slots or []}
                                                 or _rc in (self.rules.__dict__.get("_unconfirmed_deploy") or {}))
            if _coming and "keep" in p["question"]["criteria"]:
                logger.info("锁定路线的主C %s 在手里 / 刚上场没确认 → 这题维持 %s，等他接任", _rc, _cur0)
                self._restrict(p, [k for k in p["question"]["criteria"] if k != "keep"])
                must_yield = False
        if p and must_yield and _IR.on("placeholder_yield") and "keep" in p["question"]["criteria"] and len(p["question"]["criteria"]) > 1:
            self._restrict(p, ["keep"])
            logger.info("现任主C 是 1 阶占位、场上已有合格接班人 → 换主C 题去掉\"维持\"（必须让位，让给谁由局面网络定）")
        if p and _IR.on("route_avoid_weak"):
            # 屏蔽路线的另一半（第 806 / 809 局都是第 6 名）：路线屏蔽了安琪拉，换主C 题却把主C 换成安琪拉 → 主C 是他、路线永远锁不上，整局 flex。
            # 路线没锁时，换主C 候选去掉 WEAK_CARRIES（还剩"维持"或别的候选时）
            from agent.decision import lord_config as _LCw2
            _cw2 = _LCw2.load()
            # 10-21 定"尽量不给"：名单外（carry_pool 没开着，含没登记的孙悟空 / 西施 / 貂蝉）的 2 阶以上英雄都不进换主C 候选（还剩别的时）
            _wk = [k for k, n in ((p.get("raw") or {}).get("names") or {}).items()
                   if k != "keep" and n and (n in self.WEAK_CARRIES or (_cw2.present and not _cw2.enabled(n)
                                                                    and int((self.rules.pieces.get(n) or {}).get("quality") or 0) >= 2))]
            # 主C 空着时"维持"= 继续没主C，不算剩下的选项（10-20 晚第 849 局：孙尚香卖掉后只剩马可 / 安琪拉，去掉后只剩"维持"→ R6 起整局无主C、枫叶舞没绑）
            _nm = (p.get("raw") or {}).get("names") or {}
            _others = [k for k in p["question"]["criteria"] if k not in _wk and (
                (k == "keep" and self.rules._carry_piece)
                or (k != "keep" and _nm.get(k) and self.rules._carry_eligible(_nm[k], hard_only=self.rules.model_decides())))]
            if _wk and _others:
                _wkn = [p["raw"]["names"][k] for k in _wk]      # 先取名字再删（第 827 局：删完再取 → KeyError，题作废，R6~R11 没主C）
                self._restrict(p, _wk)
                logger.info("换主C 题去掉屏蔽路线的主C %s（临时规则 route_avoid_weak）", _wkn)
        if not p:
            if not self.rules._carry_piece and "carry" in self.kinds:
                # 主C 空缺、候选不到 2 个出不了题：场上只有一个合格的人 → 用他（合法性，不是价值判断）
                ok = [n for n in self.rules._field if self.rules._carry_eligible(n, hard_only=self.rules.model_decides())]
                if len(ok) > 1:
                    # 第 714 局 R3~R6：题面候选（输出位、强度 ≥5）只有亚连 → 出不了题；规则资格还多算了蒙恬 → 这里也是 2 个 → 主C 空了 4 回合、
                    # 枫叶舞一直没用（人工手动绑的）。两边口径不一致时，以出题的候选为准：题面只剩 1 个 = 没得选
                    try:
                        X = self.F._facts()
                        qc = X.carry_candidates(X.Board(c), "")
                        ok = [n for n in ok if n in qc] or ok
                    except Exception:                                   # noqa: BLE001
                        logger.exception("主C 空缺：取题面候选失败")
                if len(ok) == 1 and self.rules.apply_carry(ok[0], state, why="laya-only 唯一合格"):
                    logger.info("主C 空缺，场上只有 %s 合格 → 由他接任", ok[0])
                elif len(ok) != 1:
                    logger.info("主C 空缺，出不了题、合格的人 %s 不止一个 / 没有 → 本帧不定", ok)
            return
        levels = {n: (v or {}).get("lv") for n, v in (c.get("field") or {}).items()}
        # **候选不再按引擎 / 路线硬删，交给局面网络判断**（2026-10-08：多用局面判断网络，不加硬性限制；
        # 第 429 局路线锁定时守约被删出候选，主C 一直是 1 阶李元芳）。合法性（主C 资格）仍由 apply_carry 查
        allowed = None
        # 2026-10-12 清查（规则挡住模型的问题）：模型接管时，下面三条价值判断都只记日志，候选照常摆给局面网络
        _md = self.rules.model_decides()
        _value_drop: set = set()
        if (not _md and getattr(self.rules, "forced_route", "") and not getattr(self.rules, "_forced_released", False)
                and self.rules._carry_piece == self.rules.route_carry()):
            return                              # 指定了路线且主C 已是路线主C：不出换主C 题（不换）
        if (_IR.on("route_carry_locked") and self.rules.route_locked() and self.rules._carry_piece
                and self.rules._carry_piece == self.rules.route_carry() and self.rules._carry_piece in self.rules._field):
            # 2026-10-18 定：路线定好后不换 —— 路线主C 已在场当主C：不再出换主C 题（临时规则 route_carry_locked）
            if self.__dict__.setdefault("_route_lock_logged", set()).isdisjoint({state.round_no}):
                self._route_lock_logged.add(state.round_no)
                logger.info("路线已锁、主C %s 就是路线主C → 不出换主C 题（设计约定：定好了不换路线）", self.rules._carry_piece)
            return
        if "carry" in self.kinds and state.round_no >= 10:
            # 后期不换成比现任低 20 级以上的人（第 17 局：第 12 回合 0.41 把 25 级百里守约换成 1 级鲁班七号，血 4）
            cur = self.rules._carry_piece
            lc = levels.get(cur) if cur else None
            if isinstance(lc, int):
                low = {n for n, lv in levels.items() if isinstance(lv, int) and lv < lc - 20}
                if low and _md:
                    _value_drop |= low
                elif low:
                    allowed = (set(allowed) if allowed is not None else set(levels) | {cur}) - low
        cur = self.rules._carry_piece
        _temp = lambda n: (bool(n) and (self.rules.pieces.get(n, {}).get("quality") or 0) == 1  # noqa: E731
                           and n not in self.rules._ever_awakened)
        if _temp(cur):
            # **临时主C 不换成另一个临时主C**（2026-10-09：李元芳换成孙尚香没有意义，两人都弱，
            # 不用一个临时主C 替换另一个临时主C；438/452/462/467/470/484 局都是这种 0.5 上下的互换）
            _rc = self.rules.route_carry() if self.rules.route_locked() else ""
            temps = {n for n in levels if n != cur and n != _rc and _temp(n)}     # 路线指定的 1 阶主C（干将路线）照常可换（审查 10-09）
            # 2026-10-13 再次确认（第 599 / 603 局孙尚香 → 孙膑）：前期 1~5 回合临时主C 只需要 1 个 —— 硬约束，
            # 模型接管时也执行（模拟器的强度曲线判孙膑更好，与既定打法冲突，以既定打法为准）
            if temps and _IR.on("temp_carry_no_swap"):          # 临时规则（interim_rules），模型学会后撤
                allowed = (set(allowed) if allowed is not None else set(levels) | {cur}) - temps
            elif temps:
                _value_drop |= temps
        if _value_drop:
            logger.info("（仅记录）旧规则会去掉换主C 候选：%s（临时主C 不互换 / 后期不换低 20 级以上的人）", sorted(_value_drop))
        # **合法性先过滤**：不能当主C 的候选（apply_carry 一定拒绝）不摆给网络选（2026-10-11 第 558 局：局面网络 4 次选甄姬，
        # 全被 apply_carry 判不合法、退回规则 —— 出题时就该去掉）。这是合法性，不是价值判断
        _illegal = {n for n in levels if n != self.rules._carry_piece
                    and (n not in self.rules._field or not self.rules._carry_eligible(n, hard_only=_md))}
        # 10-21 审查：apply_carry 的另外两条合法性（拖拽回执 30 秒内没确认 / 本回合已换过）也前移（846 / 848 / 850 局 5 次选中后退回规则）
        import time as _tm
        _illegal |= {n for n, t0 in (self.rules.__dict__.get("_unconfirmed_deploy") or {}).items()
                     if n in levels and n != self.rules._carry_piece and _tm.time() - t0 < 30}
        if self.rules._carry_switch_round == state.round_no:
            _illegal |= {n for n in levels if n != self.rules._carry_piece}
        _q = lambda n: (self.rules.pieces.get(n, {}).get("quality") or 0)            # noqa: E731
        _cur_q = _q(self.rules._carry_piece) if self.rules._carry_piece in self.rules._field else 0
        try:
            _rc1 = self.rules.route_carry() if self.rules.route_locked() else ""
        except Exception:  # noqa: BLE001
            _rc1 = ""
        if _cur_q >= 2:
            # **1 阶候选不替换 2 阶以上的现任主C**（第 726 局 R5：锁定"伞亚连"、主C 亚连 t2，局面网络 0.63 换成 1 阶干将并卖掉亚连换绑枫叶舞；
            # R6 干将过前期不合格 → 又换沈梦溪、卖干将。10-02 定：1 阶只做开局占位、不能用 1 级的主C 玩到最后，规范 54.13-14）。
            # 下一回合 1 阶就不合格（_carry_eligible），换过去必然再换 → 合法性。路线自己指定的 1 阶主C 除外
            _illegal |= {n for n in levels if n != self.rules._carry_piece and _q(n) == 1 and n != _rc1}
        if _illegal:
            allowed = (set(allowed) if allowed is not None else set(levels) | {self.rules._carry_piece}) - _illegal
        if allowed is not None:
            # **主C 跟引擎**（2026-09-26）：引擎转起来后，只把引擎落点（+现任）摆给学生选，不否决它的答案
            names = p["raw"]["names"]
            drop = [k for k, n in names.items() if k != "keep" and n not in allowed]
            if drop:
                dropped = [names[k] for k in drop]      # 先取名字再删（_restrict 会把 names 里的键一起删掉）
                self._restrict(p, drop)
                logger.info("换主C 候选去掉：%s（%s）", dropped,
                            "不能当主C（合法性）" if _md else "临时主C 不互换 / 后期不换低 20 级以上的人 / 不能当主C")
            if len(p["question"]["criteria"]) < 2:
                self._yield_single(p, state, must_yield)
                return
        if (must_yield or not self.rules._carry_piece) and len(p["question"]["criteria"]) < 2:
            self._yield_single(p, state, must_yield)
            return
        choice, probs, conf, ms = self._ask(p)
        if "carry" not in self.kinds:
            self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "shadow", "影子题：只记录，不执行")
            return
        name = (p.get("raw") or {}).get("names", {}).get(choice, "")
        if choice == "keep" or not name:
            self._carry_prev_choice = None
            if (self.rules.model_decides() and self.rules._carry_piece
                    and not str(self.rules._carry_switch_by).lower().startswith("laya")):
                # 2026-10-12 第 578 局：安琪拉是规则临时补的主C，局面网络一直答"维持" —— 等于认可了她，
                # 但收尾只认"模型换来的主C"，枫叶舞一直留在干将莫邪。维持现任也记成模型选定
                self.rules._carry_switch_by = f"Laya keep {conf:.2f}"
            self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya", "维持现任")
            return
        # 2026-09-26：影响 Laya 的保护都不需要，训练得当后保护和规则可以逐步去除
        # → 去掉换主C 的门槛 / 冷却 / "被全面压制" 检查，只查合法性（apply_carry：在场、输出位、本回合没换过）。
        ok = self.rules.apply_carry(name, state, why=f"Laya {conf:.2f}")
        self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya" if ok else "rule",
                  f"换成 {name}" if ok else f"换成 {name} 不合法（不在场/不是输出位/本回合已换过）")

    def _auction(self, state) -> None:
        """**拍卖争哪一件**（2026-09-26 起交给 Laya）：每场拍卖问一次；目标被别人拍走 / 抬过上限时再问。
        只设目标（rules.auction_target），出价机械（追价、同价先到先得、能量底线、上限）仍由规则做。"""
        if state.phase != "auction" or not state.auction or "auction" not in self.kinds:
            return
        auc = state.auction
        r = self.rules
        by = {lot.piece: lot for lot in auc.lots if lot.piece}
        tgt = r.auction_target if r.auction_target and r.auction_target[0] == state.round_no else None
        _asked = (self.__dict__.get("_auction_lots") or {}).get(state.round_no)
        _new_lots = bool(_asked is not None and set(by) - _asked)
        _bid_yet = bool(getattr(r, "_bids", None)) and getattr(r, "_bids_round", None) == state.round_no
        if _new_lots and not _bid_yet:
            # 出题时拍品还没读全（2026-10-11 第 564 局：先读到 5 件选了武则天，太乙真人第 6 件才出现、没再问）→ 还没出过价就重新问
            logger.info("拍品读全了（新增 %s），还没出过价 → 重新出拍卖题", sorted(set(by) - _asked))
            tgt = None
        if tgt is not None and (tgt[1] == "" or (tgt[1] in by and not by[tgt[1]].sold
                                                  and by[tgt[1]].top_bid < r.auction_max_bid)):
            return                          # 目标还有效
        # **拍品读不全先不出题**（2026-10-11 第 566 局：那一帧只读到 [?, 姬小满, 猪八戒]，少司缘 / 镜 / 瑶 / 百里守约没读到，
        # Laya 只能在两件里挑，拍了不属于任何阵容的猪八戒；之后几帧拍品列表是空的，没机会重问）。
        # 张数看感知层推算的 auc.expected（拍品卡等距居中 → 总张数 = 剩余玩家数；先看有多少张牌再识别）。
        # 有没认出名字的拍品、或认出的少于推算张数，且倒计时还够（>3 秒或未知）→ 等下一帧；本场最多等 12 帧
        _named = [l for l in auc.lots if l.piece and not l.sold]
        _unk = any(not l.piece for l in auc.lots)
        _exp = getattr(auc, "expected", -1) or -1
        _short = (len([l for l in auc.lots if l.piece]) < _exp) if _exp > 0 else (len(_named) < 2)
        _wait = self.__dict__.setdefault("_auction_wait", {})
        if ((_unk or _short) and (auc.countdown is None or auc.countdown < 0 or auc.countdown > 3)
                and _wait.get(state.round_no, 0) < 12 and tgt is None):
            _wait[state.round_no] = _wait.get(state.round_no, 0) + 1
            return
        s = state.to_dict()
        c = dict(self._c(state), round=state.round_no)
        s["auction"]["lots"] = [l for l in s["auction"]["lots"] if not l.get("sold")
                                and not (l.get("piece") and l.get("top_bid", 0) >= r.auction_max_bid)
                                # 上不了场的 1 阶英雄不拍（第 6 回合后 _obsolete，场上有同名可合成的除外；同商店题，第 726 局）
                                and not (l.get("piece") and l["piece"] not in r._field and not r._is_carry(l["piece"])
                                         and r._obsolete(l["piece"], state.round_no))]
        _listed_only = False
        if _IR.on("buy_listed_carry"):
            # 同上（第 868 局 R7：拍品里有蒙犽，Laya 0.25 去拍雅典娜）：没有开着的主C → 拍品里有开着的主C 就只拍他
            from agent.decision import lord_config as _LCa
            _ca = _LCa.load()
            _own_a = set(r._field) | {sl.get("name") for sl in state.hand_slots or []}
            _cq_a = int((r.pieces.get(r._carry_piece or "") or {}).get("quality") or 0)
            if _ca.present and _cq_a <= 1 and not any(_ca.enabled(n) and _ca.tier(n) >= 2 for n in _own_a):
                _rc_a = r.route_carry() if r.route_locked() else ""
                _grp_a = ({_rc_a} | set(_ca.group_of(_rc_a)[1])) if _rc_a else set()
                _la = [l for l in s["auction"]["lots"] if l.get("piece") and _ca.enabled(l["piece"]) and _ca.tier(l["piece"]) >= 2
                       and (not _grp_a or l["piece"] in _grp_a)]
                # 1054 局 R4：只剩组外的狄仁杰也被直接拍下 —— 和 no_lock_outside_group（第 8 回合前不走组外主C）冲突 → 组外的不算
                _nlo = set(r.no_lock_outside_group(state) or ())
                _la = [l for l in _la if l["piece"] not in _nlo]
                if _la and len(_la) < len(s["auction"]["lots"]):
                    logger.info("没有开着的主C，拍品里有 %s → 只拍他（临时规则 buy_listed_carry）", [l["piece"] for l in _la])
                    s["auction"]["lots"] = _la
                    _listed_only = len(_la) == 1
        if _IR.on("no_lowest_level_cards"):
            # 10-21 审查：NEVER_PICK（星源星徽 / 阿科米亚之子 / 驱雷 / 月影之力 / 阵营祝福）和"给等级最低的人"的牌拍卖也不拿（还剩别的拍品时）
            _np = [l for l in s["auction"]["lots"] if not (l.get("piece") and r.never_pick(l["piece"]))]
            if _np and len(_np) < len(s["auction"]["lots"]):
                logger.info("拍卖去掉不拿的牌 %s（临时规则 no_lowest_level_cards）",
                            [l.get("piece") for l in s["auction"]["lots"] if l not in _np])
                s["auction"]["lots"] = _np
        # 2026-10-11 撤掉"拍品里有路线要的牌 → 直接定它、不问 Laya"（规则不应影响拍卖）。拍什么全由 Laya 判断。
        _left = [l for l in s["auction"]["lots"] if l.get("piece")]
        if _listed_only and len(_left) == 1:
            # 只在 buy_listed_carry 真的筛到一件时（1042 局 R7：拍品没读全、只读到 1 件「强者自强」也被直接定了目标）
            # 第 1037 局 R4：buy_listed_carry 把拍品筛到只剩沈梦溪 → 拍卖题至少 2 件才出题（q_auction 返回 None）→ 整场一价没出，
            # 拿了姬小满，开局组主C 没到、整局走了组外亚连。临时规则筛到只剩一件 = 规则已经定了目标，直接执行
            name = _left[0]["piece"]
            r.auction_target = (state.round_no, name)
            self.__dict__.setdefault("_auction_lots", {})[state.round_no] = set(by)
            if (r.pieces.get(name) or {}).get("category") == "英雄牌" and name not in r._field:
                self.__dict__.setdefault("_laya_bought", {})[name] = state.round_no
            logger.info("拍品筛到只剩 %s（临时规则）→ 直接定为拍卖目标", name)
            return
        p = self.B.auction_point(s, c, self._meta(state))
        if not p:
            return
        self.__dict__.setdefault("_auction_lots", {})[state.round_no] = set(by)
        choice, probs, conf, ms = self._ask(p)
        names = (p.get("raw") or {}).get("names", {})
        name = names.get(choice, "")

        def _bad(n):
            # 第二主C，或**拍回来也会被当闲置卖掉的英雄**（2026-10-08 第 425 局：花 5 块钱拍了杨玉环，上场就被卖掉 ——
            # 拍卖护栏把鲁班七号改成概率次高的杨玉环，下一回合杨玉环以"不在阵容规划里"被卖掉）
            if r.unwanted_second_carry(n):
                return True
            if r.other_c(n):
                return True     # 主C 定了：拍卖不拍别的 C 位英雄（2026-10-08：确定主C 之后拍卖行不拍其他 C 位英雄）
            return (r.route_locked() and r.pieces.get(n, {}).get("category") == "英雄牌"
                    and not r._planned_hero(n))
        if (r.carry_settled() or r.route_locked()) and _bad(name) and not r.model_decides():
            for k in sorted(probs, key=lambda k: -probs[k]):
                if not _bad(names.get(k, "")):
                    logger.info("Laya 拍卖选的 %s 是第二主C / 拍回来也会被卖掉，改为 %s", name, names.get(k, "") or "这场不拍")
                    choice, name = k, names.get(k, "")
                    break
        elif (r.carry_settled() or r.route_locked()) and _bad(name):
            # 2026-10-11 撤掉改选（第 538 局把亚连日落海阵容的配套米莱狄判成"第二主C"、改成不拍；导致连续几场拍卖都没出价）。
            # 只记日志，留作拍卖题重标的对照；Laya 选错（敖隐 / 花木兰）靠训练改。
            logger.info("（仅记录）旧拍卖护栏会认为 %s 是第二主C / 拍回来会被卖掉，不再改选", name)
        r.auction_target = (state.round_no, name)
        if name and (r.pieces.get(name) or {}).get("category") == "英雄牌" and name not in r._field                 and name not in {sl.get("name") for sl in state.hand_slots or []}:
            # 第 873 局 R8：拍到露娜 → 换人题"不换" → 卖手牌把露娜卖了（应卖掉牛魔、上露娜）。
            # 拍卖拍下的新英雄和商店买的一样记进 _laya_bought（bought_must_deploy；拍不到就不会等位，不影响）
            self.__dict__.setdefault("_laya_bought", {})[name] = state.round_no
        self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya",
                  f"拍卖争 {name}" if name else "这场不拍")

    def _use_card(self, state, a: dict):
        """**指定英雄的效果牌给谁**（f9d；2026-09-26：主C 本身弱时，效果牌全给主C 会让其他英雄跟不上）。
        规则要把牌拖给某人时先问学生：换人 → 改落点；先留着 → 本回合不再用这张（搁置）。返回 (动作, 是否接管)。"""
        X = self.F._facts()
        name = self.F.clean_name(a.get("piece") or "")
        _tgt = a.get("target")
        _rule_tgt = bool(_tgt) and _tgt in self.rules._field and not _IR.on("use_card_rule_targets")
        if not name or not (X.use_card_question(name) or _rule_tgt):
            # A5（2026-10-15）：规则要把牌给某个英雄、而这张牌不在用牌题范围里（众星拱月 / 求贤 / 跳级…）→ 也问 Laya 给谁
            return a, False
        if a.get("rule_only") or self.rules.card_rule_mod(name):
            # rule_only：换帅 / 挂帅这类方向是算出来的（目标只升不降），换人可能把等级换低；
            # card_rule_mod：天赋改写了合法目标（多彩古币：古币改为只给非河洛），题面的候选还是按原卡面列的 → 都不交给学生改目标
            return a, False
        if a.get("target") == "古币机":
            return a, False          # 古币机在场：古币按规则打给它（第 43 局人工手动这样做；Laya 的候选里没有古币机）
        key = (state.round_no, name)
        done = self.__dict__.setdefault("_use_done", {})
        if key in done:
            c = done[key]
            if c is None:
                return a, True
            t0 = c.get("target")
            # 2026-10-19 第 793 局 R12（妙音一直想给已经觉醒的英雄用）：3 张妙音，前 2 张给盾山（1→3 觉醒回手），
            # 第 3 张沿用缓存目标盾山 → 已觉醒、拖不上，连失败。觉醒进度牌：目标已觉醒 / 刚觉醒待回手 → 不沿用，重新问（题面会去掉已觉醒的人）
            _awk_done = (t0 in self.rules._awakened or t0 in self.rules._ever_awakened
                         or t0 in (self.rules.awaken_expected(state) or []))
            if "觉醒进度" in (X.skill(name) or "") and _awk_done:
                done.pop(key, None)
            elif (t0 in self.rules._field and not (name in self.rules._no_repeat
                                                  and (state.round_no, name, t0) in self.rules._used_on)):
                # 2026-10-19 第 779 局 R5：同名第二张（战略中枢送 2 张有福同享）沿用了上一张的 src (649,663)，那里已经没牌 →
                # 慢拖变成长按棋盘、弹出表情轮盘，牌没用出去（截图确认）。src 用规则这一帧算的位置，只沿用目标
                return dict(c, src=a.get("src") or c.get("src"), dst=self.rules._field[t0]), True
            done.pop(key, None)                      # 目标走了 / 同名第二张不能再给同一人 → 重新问
        p = self.B.use_card_point(name, self._c(state), self._meta(state))
        if not p:
            return a, False
        tg = p["raw"]["targets"]
        p["rule_choice"] = f"use{tg.index(a['target'])}" if a.get("target") in tg else None
        r = self.rules
        sk = X.skill(name)
        if (X._is_transform(sk) or X.REMOVE_RE.search(sk) or ("移除" in sk and r.pieces.get(name, {}).get("category") != "装备牌")
                or "摧毁其所有装备" in sk):      # 武装分解 / 武装重构：不拆主C 的装备（2026-10-15 进用牌题）
            # 移除 / 变身牌不许指向现任主C / 路线主C（第 19 局：限制性进化被用在路线主C 百里守约身上，路线核心断掉；
            # 题面只标了"未来主C 会失去身份"，没标现任主C —— 题面修正在 f9k）
            protect = ({r._carry_piece, r.route_carry()} if _IR.on("transform_route_carry") else {r._carry_piece}) - {"", None}   # A9：路线主C 交给题面
            drop = [f"use{i}" for i, t in enumerate(tg) if t in protect and f"use{i}" in p["question"]["criteria"]]
            if drop and len(drop) < len(p["question"]["criteria"]):
                self._restrict(p, drop)
                logger.info("%s 不指向主C：去掉 %s", name, [tg[int(k[3:])] for k in drop])
        # **卡面限定的使用者**（2026-09-30：临阵磨枪 = 触发整备效果，只能给有整备的人）：
        # 不合规的目标从选项里删掉（只查合法性）；一个合规的都没有 → 先留着
        def _illegal(t):
            if not r.card_target_ok(name, t)[0]:
                return True
            if name in r._no_repeat and (state.round_no, name, t) in r._used_on:
                return True                          # 本回合已经对他用过、这张牌不能重复叠
            pos = r._field.get(t)
            return bool(pos) and r._cell_dead(pos)   # 那一格已判死（拖不进去）
        bad = [f"use{i}" for i, t in enumerate(tg) if f"use{i}" in p["question"]["criteria"] and _illegal(t)]
        if bad:
            ok_left = [k for k in p["question"]["criteria"] if k not in bad and k != "hold"]
            if not ok_left:
                logger.info("%s 场上没有合规的使用者（%s）→ 先留着", name, r.card_target_ok(name, tg[int(bad[0][3:])])[1])
                r.park_card(name)
                done[key] = None
                return None, True
            self._restrict(p, bad)
            logger.info("%s 去掉不合规的目标 %s（%s）", name, [tg[int(k[3:])] for k in bad],
                        r.card_target_ok(name, tg[int(bad[0][3:])])[1])
        if (_IR.on("trigger_no_hold") and r._TRIGGER_RE.search(sk or "") and "hold" in p["question"]["criteria"]
                and any(k != "hold" for k in p["question"]["criteria"])):
            # **触发类的牌（临阵磨枪 = 触发整备、唤醒 = 触发登场）有合规目标就用，不许"先留着"**
            # （2026-09-30 第 13 局：学生两次选 hold（0.39 / 0.46），人工手动打出去的）
            self._restrict(p, ["hold"])
            logger.info("%s 是触发类的牌、场上有合规目标 → 用牌题去掉\"先留着\"", name)
        choice, probs, conf, ms = self._ask(p)
        if choice == "hold":
            r.park_card(name)
            self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya", f"{name} 先留着")
            done[key] = None
            return None, True
        t = tg[int(choice[3:])]
        if t not in r._field:
            self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "rule", f"{name} 想给 {t}，但它不在场上 → 按规则")
            return a, False
        act = dict(a, dst=r._field[t], target=t, why=f"Laya use_card：{name} 给 {t}")
        self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya", f"{name} 给 {t}")
        done[key] = act                              # 本回合同一张牌只问一次，别每帧重问、答案来回变（2026-09-30 审查）
        return act, True

    def _held_special(self, state) -> Optional[dict]:
        """**规则按审定表"先留着"的特殊效果牌，交给 Laya 决定用不用、给谁**（2026-10-15 第 632 局：化铁为金在手里压了 6 回合，
        Laya 用牌题四次判"给蒙恬"（0.45~0.53），可规则按审定目标 lowest_tier_high_level 判"场上没有合适对象"不出动作，
        学生的答案只当影子题记录 —— 规则挡住了模型的价值判断）。规则这一帧没有别的事可做时才问；合法性（主C / 路线主C 不变身、
        卡面限定、格子判死、本回合重复）仍由 _use_card 查。"""
        if state.phase != "prepare" or state.shop_open:
            return None
        X = self.F._facts()
        r = self.rules
        for slot in state.hand_slots or []:
            name = self.F.clean_name(slot.get("name", ""))
            cat = (r.pieces.get(name) or {}).get("category")
            if not name or cat not in ("效果牌", "天赋牌") or not X.special_card(name) or not X.use_card_question(name):
                continue
            if r._parked(name) or (state.round_no, name) in (self.__dict__.get("_use_done") or {}):
                continue
            a = {"type": "drag_card", "piece": name, "src": (slot["x"], slot["y"]), "kind": cat, "target": None}
            act, took = self._use_card(state, a)
            if took and act and act.get("target"):
                return act
        return None

    def _swap_levelup(self, state) -> Optional[dict]:
        """swap 题选了 levelup：商店开着、看得到升级按钮、钱够 → 点升级（只点一次）。"""
        if getattr(self, "_swap_lvl_round", None) != state.round_no or state.phase != "prepare" or not state.shop_open:
            return None
        self._swap_lvl_round = None
        if state.level_btn and 0 < (state.level_cost or -1) <= (state.gold or 0):
            return {"type": "buy_xp", "x": state.level_btn[0], "y": state.level_btn[1], "why": "Laya swap：升级腾位"}
        logger.info("Laya swap 想升级腾位，但商店里没看到升级按钮或钱不够 —— 放弃")
        return None

    def _swap(self, state, actions: list) -> Optional[dict]:
        """**人口满时卖谁换人（swap 题）**：每 (回合, 等着上场的英雄) 问一次。影子模式只记录；
        接管模式（kinds 带 swap）把答案记下，之后在"这一帧没有更要紧的动作"时执行卖人（只查合法性：
        被卖的人还在场上、不是主C、人口仍满、那个英雄还在手里）。"""
        if state.phase != "prepare" or state.shop_open or not state.population_valid:
            return None
        r = self.rules
        _here, _capn, _ = r.pop_now(state)          # 唯一口径（rules.pop_now）：本帧读数钳在上限内
        if not (_capn > 0 and _here >= _capn):
            return None
        if r.awaken_expected(state):
            # 觉醒进度牌 / 合成凑满刚打成功，他马上回手、自己空出一格（rules.awaken_expected，确定事件）→ 不出换人题，等他回手直接上
            return None
        _cap0 = state.population_cap or 0
        if _cap0 and len(r._field) < _cap0 and any(n not in r._field and any(sl.get("name") == n for sl in state.hand_slots or [])
                                                   for n in r._awakened):
            # 2026-10-18 第 751 局 R9：英雄觉醒回到手牌时要先把他上场——守约觉醒回手，人口读数还是旧的 6/6，
            # 换人题先卖了太乙、9 秒后才上守约。觉醒回手那人自己空出一格（账本已移除）→ 不出换人题，交给 decide_deploy 直接上（和 rules 同一判据）
            return None
        _lf_me = r.min_engine() or {}
        _lf_fin = (set(r._route_team()) | set(r.route_core()) | set(_lf_me.get("early_priority_full") or _lf_me.get("early_priority") or [])
                   | {r._carry_piece, r.route_carry()}) - {"", None}
        # 10-21 第 881 局：庄周没有作用，不如把太乙挂到铠或盾山旁边——路线锁了且场上有路线外的人 → 太乙第 12 回合前也能等位（只换外人）
        _lf_out = {n for n in r._field if n not in _lf_fin and n not in r._awakened} if r.route_locked() else set()
        waiting = tuple(sorted({sl["name"] for sl in state.hand_slots or []
                                if sl.get("name") not in r._field
                                and r.pieces.get(sl.get("name"), {}).get("category") == "英雄牌"
                                # 上不了场的不算等位（第 726 局 R7/R8：1 阶李元芳第 6 回合后按 _obsolete 永不上场，换人题两次卖狂铁 / 廉颇"给他腾位"，
                                # 空格一直空到 R9）—— 合法性，与 decide_deploy 同一判据
                                and not (r._obsolete(sl.get("name"), state.round_no) and not r._is_carry(sl.get("name")))
                                # 临时规则 late_fill_before_r12（10-18）：第 12 回合前不为太乙这类后期件卖人，有空位照常上
                                and not (_IR.on("late_fill_before_r12") and (state.round_no or 0) < 12 and not _lf_out
                                         and sl.get("name") in r.late_pieces())}))
        if not waiting:
            return None
        # 2026-10-16 第 668 局：每回合只问一个等位英雄、判"不换"后就不再问 → 刘禅连问 4 回合、路线核心盾山 / 程咬金一直没轮到。
        # 本回合已经判过"不换"的人记进 _swap_skip，下一次换下一个等位的人来问（流程，不是价值判断；判不判换仍归换人网络）
        skip = tuple(sorted(x for (rn, x), v in self._swap_todo.items() if rn == state.round_no and v == "" and x in waiting))
        sig = (state.round_no, waiting, skip)
        if sig == getattr(self, "_swap_sig", None) and not any(
                v for (rn, _), v in self._swap_todo.items() if rn == state.round_no):
            return None          # 同一回合、同一批等着上场的英雄已经问过，且没有待执行的卖人
        self._swap_sig = sig
        c = self._c(state)
        if _IR.on("late_fill_before_r12") and (state.round_no or 0) < 12 and (not _lf_out or (set(waiting) - r.late_pieces())):
            # 第 747 局 R10：米莱狄在等位时出题代码自己挑了太乙当等位者，卖露娜给太乙腾位 —— 出题这一步也要跳过后期件
            # （10-21：别的件也在等时先问别的；只有太乙在等、场上有路线外的人时才问太乙）
            skip = tuple(sorted(set(skip) | {sl.get("name") for sl in state.hand_slots or [] if sl.get("name") in r.late_pieces()}))
        elif _IR.on("late_fill_before_r12") and ((set(waiting) - set(skip) - r.late_pieces()) & (r.route_core() | r._route_team())):
            # 第 12 回合起：核心件和太乙同时在等 → 先问核心件（第 851 局 R12：先卖马可让太乙上，7 秒后明世隐等位只能卖太乙）
            skip = tuple(sorted(set(skip) | {n for n in waiting if n in r.late_pieces()}))
        if skip:
            c["_swap_skip"] = list(skip)
        if self.rules.route_locked():
            c["_swap_pref"] = sorted(self.rules.route_stage_members(state.round_no))   # 先问路线这阶段要的人
        p = self.B.swap_point(c, self._meta(state))
        if not p:
            return None
        x = p["raw"]["incoming"]
        key = (state.round_no, x)
        if (x in self.rules._ever_awakened or x in self.rules._awakened) and "bench" in p["question"]["criteria"]                 and any(k.startswith("s") and k != "bench" for k in p["question"]["criteria"]) and not _IR.on("awakened_sell_pick"):
            # A15（2026-10-15）：觉醒卡在等 → 卖谁由换人网络定；"觉醒卡最先上场"是硬约束，去掉"不换"
            self._restrict(p, ["bench"])
            logger.info("觉醒卡 %s 在等上场：换人题去掉\"不换\"（硬约束），卖谁由换人网络定", x)
        _bought = (self.__dict__.get("_laya_bought") or {}).get(x)
        if (_IR.on("bought_must_deploy") and _bought is not None and (state.round_no or 0) - _bought <= 1
                and self.rules.route_locked()            # 第 876 局：路线没锁时人人都是"路线外"，逼着买来的人轮流上场互卖
                and "bench" in p["question"]["criteria"] and any(k.startswith("s") and k != "bench" for k in p["question"]["criteria"])
                and not (x in self.rules.late_pieces() and (state.round_no or 0) < 12)):
            # 临时规则 bought_must_deploy（2026-10-21 第 853 局：买了就说明判断过要用，买到手后却没上场）：
            # Laya 本回合 / 上回合主动买的新英雄在等位 → 执行它自己的买牌决定：换人题去掉"不换"，卖谁仍由 Laya 定
            # 10-21 第 859 局：只在有路线外的人可卖时才强制上场，而且只卖外人（R13~R15 买云中君后被迫卖太乙）；
            # 腾位只能卖路线核心 / 最终阵容 / 引擎件 / 主C → 不强制（买错了靠训练改，不拿核心件陪葬）
            _r = self.rules
            _me = _r.min_engine() or {}
            _fin = (set(_r._route_team()) | set(_r.route_core()) | set(_me.get("early_priority_full") or _me.get("early_priority") or [])
                    | {_r._carry_piece, _r.route_carry()}) - {"", None}
            _out = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and k != "bench" and n not in _fin]
            if _out:
                _drop = [k for k in p["question"]["criteria"] if k not in _out]
                self._restrict(p, _drop)
                logger.info("%s 是 Laya 第 %d 回合主动买的 → 换人题去掉\"不换\"、只卖路线外的人 %s（临时规则 bought_must_deploy）",
                            x, _bought, [p["raw"]["names"].get(k) for k in _out])
            else:
                logger.info("%s 是 Laya 第 %d 回合主动买的，但腾位只能卖路线核心件 → 不强制上场（bought_must_deploy）", x, _bought)
        if _IR.on("late_fill_before_r12") and (state.round_no or 0) < 12 and x in r.late_pieces():
            _o12 = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and k != "bench" and n in _lf_out]
            self._restrict(p, [k for k in p["question"]["criteria"] if k.startswith("s") and k != "bench" and k not in _o12])
            logger.info("第 12 回合前太乙等位：只许卖路线外的人 %s（late_fill_before_r12，10-21）", [p["raw"]["names"].get(k) for k in _o12])
        if _IR.on("no_core_for_offroute") and self.rules.route_locked():
            # 临时规则 no_core_for_offroute（第 851 局 R5：卖亚连线核心件米莱狄给马可波罗（关掉的主C）腾位）：
            # 等位的人不是路线成员 → 不卖路线成员 / 主C 给他腾位（还剩"不换"或别的卖法时）
            _team = set(self.rules.route_members()) | set(self.rules._route_team()) | {self.rules._carry_piece, self.rules.route_carry()}
            if x not in _team:
                _bad = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and k != "bench" and n in _team]
                if _bad and len(_bad) < len(p["question"]["criteria"]):
                    logger.info("等位的 %s 不在路线里 → 不卖路线成员 %s 给他腾位（临时规则 no_core_for_offroute）", x,
                                sorted({p["raw"]["names"][k] for k in _bad}))
                    self._restrict(p, _bad)
        # **不占人口的单位（天赋召唤的始祖熊灵等）不进换人候选**：换人是为了腾人口，卖它腾不出位置，还白丢它攒的永久等级（合法性，2026-10-14）
        _np = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and (self.rules.pieces.get(n) or {}).get("no_pop")]
        if _np:
            _npn = sorted({p["raw"]["names"][k] for k in _np})
            self._restrict(p, _np)
            logger.info("换人候选去掉不占人口的单位（卖了不腾位，合法性）：%s", _npn)
        _msy_late = False
        if _IR.on("msy_taiyi_no_swap"):
            # 临时规则 msy_taiyi_no_swap（2026-10-21 第 882 局 R13 卖太乙给新到的明世隐）：到了非常后期，没必要用明世隐换太乙，
            # 刚上场的明世隐只能提供 2 级没啥意义；反过来明世隐叠到觉醒了，就不要用太乙换明世隐
            _lfm = self.rules.late_pieces()
            _dropm = []
            if x == "明世隐" and (state.round_no or 0) >= 12:
                _msy_late = True
                _dropm = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and k != "bench" and n in _lfm]
            elif x in _lfm and "明世隐" in self.rules._field and self.rules.is_awakened("明世隐"):
                _dropm = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and k != "bench" and n == "明世隐"]
            _dropm = [k for k in _dropm if k in p["question"]["criteria"]]
            if _dropm and len(_dropm) < len(p["question"]["criteria"]):
                _nm = [p["raw"]["names"].get(k) for k in _dropm]
                self._restrict(p, _dropm)
                logger.info("%s 等位：不拿 %s 换（临时规则 msy_taiyi_no_swap）", x, _nm)
        if _IR.on("late_fill_before_r12") and not _msy_late:
            # 2026-10-18：有核心件进来、没有位置时，太乙要让位 —— 等位的是路线核心 / 最终阵容件、场上有后期件（太乙这类）→ 只能卖后期件
            _lf = self.rules.late_pieces()
            _core = (self.rules.route_core() | self.rules._route_team()) - _lf
            _lfk = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and n in _lf]
            # 2026-10-19 第 792 局 R12~R13（后期反复卖英雄、买英雄上场又卖，尤其李元芳和太乙）：场上还有最终阵容外的人（钟馗 / 武则天）时
            # 这里只留太乙 → 卖太乙给李元芳；下回合太乙回来又卖李元芳。改：场上有最终阵容 / 核心引擎之外的人 → 不强制后期件让位（交给 final_team_first 卖外人）
            _fin0 = _core | set((self.rules.min_engine().get("early_priority_full") or self.rules.min_engine().get("early_priority")) or []) | _lf | {self.rules._carry_piece}
            if any(k.startswith("s") and n not in _fin0 for k, n in p["raw"]["names"].items()):
                _lfk = []
            if x in _core and _lfk:
                _drop = [k for k in p["question"]["criteria"] if k not in _lfk]
                if _drop:
                    self._restrict(p, _drop)
                    logger.info("核心件 %s 等位、场上有后期件 → 后期件让位，只留：%s", x, _lfk)
        if _IR.on("final_team_first") and self.rules.route_locked() and (state.round_no or 0) >= 5:
            # 2026-10-18：最终目标是凑齐阵容推荐里的最终阵容；第 747 局牛魔、近 24 局苏烈 72/77 次、廉颇 / 项羽多次被保着
            # （"别卖主坦 / 唯一前排"），最终阵容件压在手里 → 最终阵容件等位时，只在最终阵容外的人里卖（不卖最终阵容件、不选不换）
            _fin = self.rules._route_team() | ({self.rules.route_carry()} if self.rules.route_carry() else set())
            # 2026-10-19 第 782 局 R7（这时应该上李元芳）：瞬步4.0 狄仁杰的最终阵容没有李元芳，但他是核心引擎（古币英雄）
            # → 核心引擎件（min_engine early_priority）等位也按最终阵容件处理，换人只在两者之外卖
            _fin = _fin | set((self.rules.min_engine().get("early_priority_full") or self.rules.min_engine().get("early_priority")) or [])
            _late = self.rules.late_pieces()
            if (state.round_no or 0) >= 12 and x in _late:
                # 第 12 回合起后期件（太乙）也是最终阵容件（late_fill_before_r12 只管第 12 回合前）：它等位时只卖最终阵容 / 核心引擎之外的人，
                # 没有外人就不换（第 792 局 R13 卖李元芳给太乙腾位、R12 又反过来）
                _out = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and n not in _fin and n != self.rules._carry_piece]
                if ("明世隐" in self.rules._field and self.rules.is_awakened("明世隐")
                        and "李元芳" in self.rules._field and not self.rules.is_awakened("李元芳")):
                    # 2026-10-21 第 886 局：到最后李元芳还没觉醒、明世隐已能提供很高等级时，可以用太乙换李元芳
                    # —— 明世隐觉醒后转化量够大，没觉醒的李元芳（一回合 1 张古币 = +2 级）不如太乙复活（这时不差那一回合两级，重生性价比更高）；_out 非空时"不换"也去掉 = 直接换
                    _out += [k for k, n in p["raw"]["names"].items() if k.startswith("s") and n == "李元芳" and k not in _out]
                _keep = _out or (["bench"] if "bench" in p["question"]["criteria"] else [])
                _drop = [k for k in p["question"]["criteria"] if k not in _keep]
                if _keep and _drop:
                    self._restrict(p, _drop)
                    logger.info("后期件 %s 等位（第 12 回合起算最终阵容）→ 只卖最终阵容外的人：%s（临时规则 final_team_first）",
                                x, [p["raw"]["names"].get(k, k) for k in _keep])
            elif x in _fin and x not in _late:
                # 太乙这类后期件（late_pieces）给核心 / 最终阵容件让位（10-18；10-19 加刘禅）
                _out = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and n not in _fin and n != self.rules._carry_piece]
                if not _out:      # 没有外人时，后期件才让位（第 792 局：外人钟馗在场却卖了太乙）
                    _out = [k for k, n in p["raw"]["names"].items() if k.startswith("s") and n in _late and n != self.rules._carry_piece]
                if _out:
                    _drop = [k for k in p["question"]["criteria"] if k not in _out]
                    if _drop:
                        self._restrict(p, _drop)
                        logger.info("最终阵容件 %s 等位 → 只在最终阵容外的人里卖：%s（临时规则 final_team_first）",
                                    x, [p["raw"]["names"][k] for k in _out])
        duo = self.rules.duo_members() if _IR.on("duo_no_sell") else set()
        if duo and self.rules._carry_piece not in duo:
            duo = set()       # 2026-10-14 第 626 局：主C 是守约（不在共战两人里）时还保护貂蝉 / 上官婉儿，满员 2 格被占、铠 / 钟馗 / 太乙压在手里。
                              # 10-12 的"不能卖"前提是主C 就是共战之一；主C 不是他们时这两人和别的补位件一样，卖不卖交给换人网络
        if duo:
            # 共战天赋的两人不能卖（2026-10-12：必须同时上场，不能卖）—— 硬约束，换人候选里去掉
            drop = [k for k, n in p["raw"]["names"].items() if n in duo and k.startswith("s")]
            if drop:
                _names = sorted({p["raw"]["names"][k] for k in drop})   # 先取名字再删（_restrict 会把 names 里的键一起删掉）
                self._restrict(p, drop)
                logger.info("换人候选去掉共战天赋的英雄：%s", _names)
            if not any(k.startswith("s") and k != "bench" for k in p["question"]["criteria"]):
                self._swap_todo[key] = ""
                return None
        if key not in self._swap_todo and "swap" in self.kinds and self.rules.route_locked():
            # 路线锁定：不卖路线**当前阶段**要的人（2026-09-28）。原来保护的是路线全部成员，连开局过渡的牛魔也算 ——
            # 2026-09-30 轮换第 12 局：后期 1 级牛魔卖不掉，手里两张雅典娜（当前阶段要的）一直上不去。
            members = self.rules.route_stage_members(state.round_no)
            drop = [k for k, n in p["raw"]["names"].items() if n and n in members]
            if drop and self.rules.model_decides():
                # 2026-10-12 清查：模型接管时只记录，卖不卖路线成员交给换人网络（题面有 ROUTE / MIN ENGINE 标签）
                logger.info("（仅记录）旧规则会从换人候选去掉路线这阶段要的人：%s", [p["raw"]["names"][k] for k in drop])
                drop = []
            if drop:
                self._restrict(p, drop)
            # 2026-10-11 撤掉"等着上场的是路线这阶段要的人 → 去掉'不换'"：第 537 局阵容已成型，为路线写死的铠卖了明世隐
            # （这样做没有道理）。卖不卖、卖谁交给换人网络。
            if not any(k.startswith("s") and k != "bench" for k in p["question"]["criteria"]):
                self._swap_todo[key] = ""
                logger.info("场上能卖的都是路线成员，%s 先留在手里", x)
                return None
        if (key not in self._swap_todo and "swap" in self.kinds and self.rules.route_locked()
                and self.rules.unwanted_second_carry(x) and self.rules.model_decides()):
            # 2026-10-12 清查（原规则："艾琳是第二主C，不为它卖人"）：模型接管时照常出题，题面有 C-POSITION 标签
            logger.info("（仅记录）旧规则会因 %s 是第二主C 不出换人题", x)
        elif (key not in self._swap_todo and "swap" in self.kinds and self.rules.route_locked()
                and self.rules.unwanted_second_carry(x)):
            # 等着上场的是第二主C（不给主C 涨等级、不是路线成员）→ 不卖场上的人给它腾位（2026-10-08 第 401 局：
            # 卖了产古币的李元芳给花木兰腾位，而场上的花木兰没有任何作用）。临时护栏，同 N 选 1 / 拍卖
            self._swap_todo[key] = ""
            logger.info("%s 是第二主C（不喂主C、不是路线成员），不为它卖人", x)
            return None
        mb = getattr(self.rules, "_maple_bound", "")
        if key not in self._swap_todo and "swap" in self.kinds and mb and self.rules.model_decides():
            # 2026-10-12：宿主不是主C 时卖掉他正好让枫叶舞换绑给主C；题面标着 MAPLE HOST / maple unbinds，交给换人网络
            if any(n == mb and k.startswith("s") for k, n in p["raw"]["names"].items()):
                logger.info("（仅记录）旧规则会从换人候选去掉枫叶舞宿主：%s", mb)
        elif key not in self._swap_todo and "swap" in self.kinds and mb:
            # 绑着枫叶舞的人卖不卖、什么时候卖，归"换主C 收尾"管（含等路线主C 追上一半的推迟），换人题不卖他
            # （2026-10-09 第 442 局：收尾刚判"129 级临时主C 李元芳先不卖"，换人题 0.41 就把他卖了给瑶腾位）
            drop = [k for k, n in p["raw"]["names"].items() if n == mb and k.startswith("s")]
            if drop:
                self._restrict(p, drop)
                logger.info("换人候选去掉枫叶舞宿主：%s", mb)
            if not any(k.startswith("s") and k != "bench" for k in p["question"]["criteria"]):
                self._swap_todo[key] = ""
                logger.info("场上能卖的只有枫叶舞宿主，%s 先留在手里", x)
                return None
        if key not in self._swap_todo and "swap" in self.kinds and _IR.on("swap_engine_parts") and self.rules.engine_lock():
            # 引擎转起来后，卖谁腾位的候选**去掉引擎零件**（第 14 局 swap 卖了产出方鬼谷子）
            drop = [k for k, n in p["raw"]["names"].items() if n and self.rules._is_engine_part(n)]
            if drop:
                self._restrict(p, drop)
                logger.info("换人候选去掉引擎零件：%s", drop)
            if not any(k.startswith("s") and k != "bench" for k in p["question"]["criteria"]):
                self._swap_todo[key] = ""
                logger.info("场上能卖的都是引擎零件，%s 先留在手里", x)
                return None
        if key not in self._swap_todo:
            choice, probs, conf, ms = self._ask(p)
            name = p["raw"]["names"].get(choice, "")
            if "swap" not in self.kinds:
                self._swap_todo[key] = None
                self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "shadow", "影子题：只记录，不执行")
                return None
            if choice == "levelup":
                # f10：升一级腾位（只查合法：升级按钮在、钱够）；升完人口 +1，规则的上场会把它放上去
                self._swap_todo[key] = ""
                ok = 0 < (state.level_cost or -1) <= (state.gold or 0) or (state.level_cost or -1) <= 0
                self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya" if ok else "rule",
                          f"升一级给 {x} 腾位（先开店找升级按钮）" if ok else "想升级腾位，但钱不够")
                if ok:
                    # 升级按钮基本只在商店开着时可见：记下待办，先开店，下一帧在 _swap_levelup 里点
                    self._swap_lvl_round = state.round_no
                    return {"type": "toggle_shop", "intent": "open", "why": f"Laya swap：开店升级给 {x} 腾位"}
                return None
            # 2026-09-26：去掉 swap 的门槛（SWAP_MARGIN）和"刚换上场不卖"（影响 Laya 的保护都不需要）
            self._swap_todo[key] = name
            r.__dict__.setdefault("_swap_asked", {})[x] = state.round_no      # 本回合换人网络已经对 x 表过态
            if name:
                (r.__dict__.get("_swap_benched") or {}).pop(x, None)          # 这回合要给 x 腾位了：撤掉上回合的"不换"
            if not name:
                # 局面网络判"不换，先留在手里"：记下来，过了这一回合还上不去就当闲置手牌卖（rules._idle_hand_hero）
                r.__dict__.setdefault("_swap_benched", {}).setdefault(x, state.round_no)   # 记第一次判"不换"的回合
            self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya",
                      f"卖 {name} 给 {x} 腾位" if name else f"不换，{x} 先留在手里")
        name = self._swap_todo.get(key)
        if not name:
            return None
        if name not in r._field or name == r._carry_piece or r.sell_blocked(name):
            self._swap_todo[key] = ""
            return None
        if actions and actions[0].get("type") not in ("sell_unit", "deploy", "drag_card", "move_unit"):
            return None          # 这一帧有别的事（关店/用牌/升级…），下一帧再卖
        self._swap_todo[key] = ""                        # 只卖一次；没上成场也不再为它卖第二个
        r._swapped_for.add((state.round_no, x))
        r.deploy_first = (state.round_no, x)             # 腾出的格子先给它（别让"吃登场效果"的杂牌抢走）
        return {"type": "sell_unit", "piece": name, "src": r._field[name], "why": f"Laya swap：给 {x} 腾位"}

    def _sell_filler(self, state) -> Optional[dict]:
        """f9lw sell_filler 题（2026-10-16：弈星升一次棋手等级就回本，可以直接卖掉腾位子）：没人等位时，
        回本的弈星 / 过了阶段的开局件 / 别家主C 卖不卖由 Laya 定。每回合问一次；答 sN 就卖（只查合法性：还在场、不是主C、能卖）。"""
        if "sell_filler" not in self.kinds or state.phase != "prepare" or state.shop_open:
            return None
        if getattr(self, "_filler_round", None) == state.round_no:
            return None
        self._filler_round = state.round_no
        c = self._c(state)
        p = self.B.sell_filler_point(c, self._meta(state))
        if not p:
            return None
        r = self.rules
        bad = [k for k, n in p["raw"]["names"].items() if k.startswith("s")
               and (n not in r._field or n == r._carry_piece or r.sell_blocked(n) or (r.pieces.get(n) or {}).get("no_pop"))]
        if bad:
            self._restrict(p, bad)
        if not any(k.startswith("s") for k in p["question"]["criteria"]):
            return None
        choice, probs, conf, ms = self._ask(p)
        name = p["raw"]["names"].get(choice) or ""
        self._log(p, choice, probs, conf, ms, "keep", "laya", f"卖占位件 {name}" if name else "占位件先留着")
        logger.info("Laya[sell_filler] %s（%.2f）%s", f"卖 {name}" if name else "不卖", conf, {k: round(v, 2) for k, v in probs.items()})
        if not name:
            return None
        return {"type": "sell_unit", "piece": name, "src": r._field[name], "why": f"Laya sell_filler：卖占位件 {name}"}

    def _sell_hand(self, state) -> Optional[dict]:
        """f9lx sell_hand 题（2026-10-16：别家主C（如海月）不必留着，路线不需要的英雄要及时卖掉）：
        手里的英雄牌卖不卖由 Laya 定。每回合问一次；答 sN 就卖那张手牌（只查合法性：还在手里、不是觉醒卡 / 转瞬）。"""
        if "sell_hand" not in self.kinds or state.phase != "prepare" or state.shop_open:
            return None
        # 2026-10-16 第 671 局：每回合只问一次、只卖一张 → 花木兰 ×2、武则天 ×2 压了好几回合。
        # 改成：同一回合、手里英雄没变就不再问；卖掉一张后手牌变了，接着再问（Laya 答"留着"就停）—— 流程，卖不卖仍由 Laya 定
        sig = (state.round_no, tuple(sorted(sl.get("name") or "" for sl in state.hand_slots or [])))
        if getattr(self, "_hand_q_sig", None) == sig:
            return None
        cnt = self.__dict__.setdefault("_hand_q_count", {})
        if cnt.get(state.round_no, 0) >= 6:
            return None                     # 手牌读数抖动时别每帧出题：每回合最多 6 问
        c = self._c(state)
        p = self.B.sell_hand_point(c, self._meta(state))
        if not p:
            return None
        self._hand_q_sig = sig
        cnt[state.round_no] = cnt.get(state.round_no, 0) + 1
        slots = {sl.get("name"): sl for sl in state.hand_slots or []}
        # 本回合刚买进来的不问卖不卖（第 720 局 R13：云中君 买→卖手牌 来回 3 次、716 局杨玉环 4 次 —— 商店题刚做的决定，执行它；下回合照常问）
        _bought = {e.get("牌") for e in (self.rules.ledger or []) if e.get("event") == "买牌" and e.get("round") == state.round_no}
        # 拍卖拍下的同样算（第 873 局：拍到露娜 → 同回合卖手牌卖了）
        _bought |= {n for n, rd in (self.__dict__.get("_laya_bought") or {}).items() if rd == state.round_no}
        from agent.decision import lord_config as _LCh
        _chh = _LCh.load()
        if _chh.present and _IR.on("buy_listed_carry") and not any(_chh.enabled(n) and _chh.tier(n) >= 2 for n in self.rules._field):
            # 10-21 审查 3：场上没有名单内主C → 手里的名单内主C 不问卖不卖（否则逼买 → 卖手牌 → 再逼买）
            _bought |= {sl.get("name") for sl in state.hand_slots or []
                        if _chh.enabled(sl.get("name") or "") and _chh.tier(sl.get("name") or "") >= 2}
        bad = [k for k, n in p["raw"]["names"].items() if k.startswith("s")
               and (n not in slots or slots[n].get("awakened") or n in _bought)]
        if bad:
            self._restrict(p, bad)
        if not any(k.startswith("s") for k in p["question"]["criteria"]):
            return None
        choice, probs, conf, ms = self._ask(p)
        name = p["raw"]["names"].get(choice) or ""
        self._log(p, choice, probs, conf, ms, "keep", "laya", f"卖手牌 {name}" if name else "手牌英雄先留着")
        logger.info("Laya[sell_hand] %s（%.2f）%s", f"卖 {name}" if name else "不卖", conf, {k: round(v, 2) for k, v in probs.items()})
        if not name:
            return None
        sl = slots[name]
        self.rules.note("卖手牌", round=state.round_no, 牌=name, 原因=f"Laya sell_hand {conf:.2f}")
        return {"type": "sell_card", "piece": name, "src": (sl["x"], sl["y"]), "why": f"Laya sell_hand：卖手牌 {name}"}

    def override(self, state, actions: list) -> list:
        try:
            self._shadow_use_card(state)
        except Exception:                                           # noqa: BLE001
            logger.exception("use_card 影子题出错（不影响决策）")
        try:
            self._route(state)                  # 先定路线，同一帧的 carry 题就能看到它
        except Exception:                                           # noqa: BLE001
            logger.exception("route 题出错（不影响决策）")
        try:
            self._shadow_carry(state)
        except Exception:                                           # noqa: BLE001
            logger.exception("carry 影子题出错（不影响决策）")
        try:
            self._shadow_eco(state)
        except Exception:                                           # noqa: BLE001
            logger.exception("eco 影子题出错（不影响决策）")
        # 只在搜牌窗口真的开得了时撤（第 500 局：1 级时 eco 说 roll，但缺的守约 1 级刷不出 → 窗口没开，却照样撤掉升级，
        # 第 3 回合还是 1 级、前 7 回合只刷了 1 次）
        if (self.rules.eco_says(state) == "roll" and actions and actions[0].get("type") == "buy_xp"
                and self.rules._in_roll_window(state) and not self.rules.hold_level_for_carry(state)):
            logger.info("局面网络[eco] 本回合停级刷牌 → 撤掉规则这一帧的升级")
            actions = []
        try:
            self._auction(state)
        except Exception:                                           # noqa: BLE001
            logger.exception("拍卖题出错（不影响决策，交给规则）")
        _tgt = self.rules.auction_target
        if _tgt and _tgt[0] == state.round_no and any(a.get("type") == "auction_bid" for a in actions or []):
            # 规则这一帧已经给出了出价动作，Laya 后定的目标要覆盖它（2026-10-11 第 536 局：Laya 改判"这场不拍"花木兰，
            # 规则同一帧的"给花木兰出价 1"照样执行了）
            _keep = [a for a in actions if not (a.get("type") == "auction_bid" and a.get("piece") != _tgt[1])]
            if len(_keep) != len(actions):
                logger.info("拍卖：Laya 目标是 %s，撤掉规则这一帧对别的拍品的出价", _tgt[1] or "这场不拍")
                actions = _keep
        try:
            lv = self._swap_levelup(state)
            if lv:
                return [lv]
            sw = self._swap(state, actions)
            if sw:
                return [sw]
        except Exception:                                           # noqa: BLE001
            logger.exception("swap 题出错（不影响决策）")
        if not actions and self._shop_event_ok(state, None):
            act = self._shop(state, None)                # 商店开着、规则没动作：问 Laya（局面没变沿用上次答案）
            if act is not None:
                return [act]
        if not actions:
            try:
                act = self._sell_hand(state)
                if act:
                    return [act]
            except Exception:                                       # noqa: BLE001
                logger.exception("sell_hand 题出错（不影响决策）")
        if not actions:
            try:
                act = self._sell_filler(state)
                if act:
                    return [act]
            except Exception:                                       # noqa: BLE001
                logger.exception("sell_filler 题出错（不影响决策）")
        if not actions:
            try:
                self._shadow_timing(state)
            except Exception:                                       # noqa: BLE001
                logger.exception("use_timing 影子题出错（不影响决策）")
        if not actions and "use_card" in self.kinds:
            try:
                act = self._held_special(state)
                if act:
                    return [act]
            except Exception:                                       # noqa: BLE001
                logger.exception("留在手里的特殊效果牌交给 Laya 出错（不影响决策）")
        if not actions:
            # **规则本帧给不出动作时也查一次"路线主C / 路线要的牌在商店"**（2026-10-01 第 84 局：路线锁定守约，第 4 回合守约在商店、
            # 能量 4 价 3，可规则因人口满 / 等级低刷不出主力品阶给了空决策 → preempt_shop 只挂在"有商店动作"的分支上，从没被调用，
            # 3 帧后商店被关，第 6 回合以"没拿到守约"换了路线，第 6 名）
            if state.phase == "prepare" and state.shop_open and not self.rules.model_decides():
                pre = self.rules.preempt_shop(state)
                if pre and pre.get("why") in ("路线主C", "路线要的牌"):
                    return [pre]
            return actions
        a = actions[0]
        if ("use_card" in self.kinds and a.get("type") == "drag_card" and a.get("target")
                and a.get("kind") in ("效果牌", "天赋牌") and state.phase == "prepare"):
            try:
                act, took = self._use_card(state, a)
                if took:
                    return ([act] if act else []) + actions[1:]
            except Exception:                                       # noqa: BLE001
                logger.exception("use_card 题出错（本帧按规则）")
        try:
            if a.get("type") == "card_pick" and "card_pick" in self.kinds and state.card_options:
                return [self._card_pick(state, a)] + actions[1:]
            if a.get("type") in ("talent_pick", "talent_refresh") and "talent" in self.kinds and state.talents:
                return [self._talent(state, a)] + actions[1:]
            if (state.phase == "prepare" and state.shop_open and "shop" in self.kinds
                    and a.get("type") in SHOP_TYPES):
                act = self._shop(state, a)
                return ([act] if act else []) + actions[1:]      # None = 学生选"不买了"但店关不掉：这一帧什么也不做
            if self._shop_event_ok(state, a):
                # **商店开着、规则想关店（买够了 / 去处理棋盘）：先问 Laya 还买不买**（2026-10-09：买不买应按局面判断）。
                # Laya 选"停手"才关店；局面没变就沿用上次的答案（_shop 里按签名缓存），不会每帧重问
                act = self._shop(state, None)
                if act is not None:
                    return [act] + actions[1:]
        except Exception:                                           # noqa: BLE001
            logger.exception("Laya 决策出错，本帧用规则")
        return actions

    def _card_pick(self, state, a):
        res = self._card_pick_impl(state, a)
        try:
            n = (res or {}).get("piece") or ""
            r = self.rules
            if (res or {}).get("type") == "card_pick" and (r.pieces.get(n) or {}).get("category") == "英雄牌"                     and n not in r._field and n not in {sl.get("name") for sl in state.hand_slots or []}:
                # N 选 1 主动选的新英雄也算"自己拿的"（第 864 / 872 局：选了鲁班大师 / 花木兰，同回合卖手牌又卖了）：
                # 同回合不问卖不卖、满员有外人可卖时上场（bought_must_deploy）
                self.__dict__.setdefault("_laya_bought", {})[n] = state.round_no
        except Exception:                                           # noqa: BLE001
            logger.exception("记 N 选 1 拿到的英雄出错（忽略）")
        return res

    def _card_pick_impl(self, state, a):
        if _IR.on("special_equip_output"):
            # 临时规则 special_equip_output（2026-10-19：主C 没满装时特殊装备优先选输出；
            # 装备仍优先吸血装）：N 选 1 全是装备、主C 没满 3 件 → 选对主C 适配分最高的输出装（高攻速主C 没吸血装时吸血优先）
            from agent.decision import hero_traits as _ht
            r = self.rules
            C = self._real_carry(state)      # 1 阶临时主C 不算（10-20）
            _opts = state.card_options or []
            if (C and _opts and len(r._worn.get(C) or ()) < 3
                    and all((r.pieces.get(o.get("name") or "") or {}).get("category") == "装备牌" for o in _opts)):
                def _out(n):
                    cls = ((_ht.EQUIP_CLASS.get(n) or ("",))[0] or "")
                    return _ht.base_kind(n) == "output" or cls.startswith("dps") or bool(_ht.item_dmg(n))
                # 2026-10-20 第 807 局 R9（主C 装备没满，装备却给了别人）：四件全是基础装备（大棒 / 力量腰带 / 圣者法典 / 破碎圣杯），
                # 按"输出装"挑了大棒——大棒四件成品对物理技能型守约全不对口，执行层只好穿给李元芳。
                # 改：基础装备按"铸成后对主C 最好的那件"算（守约 → 破碎圣杯铸法源之杯，官方阵容就带它）；铸不出对口成品的不选
                import json as _js
                from pathlib import Path as _Pp
                try:
                    _rec = _js.loads((_Pp(__file__).resolve().parents[2] / "data" / "equip_recipes.json").read_text(encoding="utf-8")).get("recipes", {})
                except (OSError, ValueError):
                    _rec = {}

                def _forged(n):
                    rs = _rec.get(n)
                    return _ht.carry_order(rs, C, onhit=_ht.live_onhit(C))[0] if rs else n
                ok = [o for o in _opts if (_out(o.get("name") or "") or (o.get("name") in _rec))
                      and r._equip_allowed(_forged(o.get("name") or ""), C)
                      and _ht.live_fits(_forged(o.get("name") or ""), C)[0]
                      and _ht.profile_says(_forged(o.get("name") or ""), C) != "avoid"]
                if ok:
                    _ls = set() if (set(r._worn.get(C) or ()) & _ht.carry_lifesteal_items(C)) or not _ht.attack_speed_carry_live(C) \
                        else _ht.carry_lifesteal_items(C)
                    # 10-20 第 836 局（守约没穿免控装备、也没选枫叶舞免控天赋，一直被控放不出技能）：主C 缺免控 → 免控装优先（吸血之后）
                    _ccn = (not (set(r._worn.get(C) or ()) & (_ht.carry_cc_items(C) | {"机器头"}))
                            and _ht.CC_IMMUNE_MAPLE not in (getattr(r, "_lord_boosts", set()) or set()))
                    best = max(ok, key=lambda o: ((_forged(o.get("name") or "") in _ls),
                                                  _ccn and _forged(o.get("name") or "") in _ht.carry_cc_items(C),
                                                  _ht.item_fit_score(_forged(o.get("name") or ""), C) or 0))
                    logger.info("N 选 1 全是装备、主C %s %d/3 件 → 选输出装 %s（临时规则 special_equip_output）",
                                C, len(r._worn.get(C) or ()), best.get("name"))
                    return dict(a, piece=best["name"], x=best["x"], y=best["y"])
        want = self.rules.route_wanted_cards()
        hit = (None if self.rules.model_decides()        # 2026-10-11：模型接管时不由规则抢先拿"路线要的牌"
               else next((o for o in state.card_options or [] if o.get("name") in want), None))
        if hit is not None:
            logger.info("N 选 1 里有路线要的牌 %s（路线推荐 / 引擎用牌）→ 拿它", hit["name"])
            self.rules.note("路线要的牌", round=state.round_no, 牌=hit["name"], 来源="N选1")
            return dict(a, piece=hit["name"], x=hit["x"], y=hit["y"])
        # **重名选项合并成一个再出题**（2026-10-11 第 539 局：[遗志, 逐鹿征召令, 大河流域征召令 ×2] → 出题函数遇重名放弃，
        # 退回规则的价值表选了遗志，选得没有明确目标）。选中后点第一张同名的
        _uniq, _seen = [], set()
        for o in state.card_options or []:
            _k = self.B.F.clean_name(o.get("name") or "") if hasattr(self.B, "F") else o.get("name")
            if _k not in _seen:                     # 和出题函数同一口径（clean_name）去重（审查 2026-10-11）
                _seen.add(_k)
                _uniq.append(o)
        _sd = state.to_dict()
        _sd["card_options"] = [dict(o) for o in _uniq]
        p = self.B.card_pick_point(_sd, self._c(state), self._meta(state), chosen=a.get("piece"))
        if not p:
            logger.info("N 选 1 出不了题（选项 %s），按规则", [o.get("name") for o in state.card_options or []])
            return a
        _opts_all = state.card_options
        state.card_options = _uniq
        try:
            return self._card_pick_answer(state, a, p)
        finally:
            state.card_options = _opts_all

    def _card_pick_answer(self, state, a, p):
        # 2026-10-21 第 891 局 R11（珍品宝箱三选一：驱雷 / 盛华 / 青莲剑谱，三件对主C 守约都没用）：Laya 0.66 选了限定近战的青莲剑谱，
        # 拿到后"场上没有合适的对象，先留着"白占手牌；盛华本可给主坦程咬金。装备选项当下场上没人能穿（审定目标 / 穿戴限制）→ 去掉（合法性：拿了用不上）
        _eq_dead = []
        for k in p["question"]["criteria"]:
            if not k.startswith("opt") or int(k[3:]) >= len(state.card_options or []):
                continue
            nm = (state.card_options[int(k[3:])] or {}).get("name") or ""
            if (self.rules.pieces.get(nm) or {}).get("category") != "装备牌":
                continue
            rv = self.rules.usage_reviewed.get(nm) or {}
            try:
                if rv.get("target") and rv.get("target") != "none":
                    dead = self.rules._reviewed_target(state, nm, rv) == "skip"
                else:
                    dead = not any(self.rules._equip_allowed(nm, n) and not self.rules._no_gear(n)
                                   and n not in self.rules.NO_EQUIP_UNITS for n in self.rules._field)
            except Exception:                                   # noqa: BLE001
                dead = False
            if dead:
                _eq_dead.append(k)
        if _eq_dead and len(_eq_dead) < len(p["question"]["criteria"]):
            logger.info("N 选 1：装备 %s 当下场上没人能穿 → 不选", [p["raw"]["names"].get(k) if isinstance(p.get("raw"), dict) and p["raw"].get("names") else k for k in _eq_dead])
            self._restrict(p, _eq_dead)
        if (_IR.on("core_first_no_copies") and self.rules.route_locked() and (state.round_no or 0) < 12
                and self.rules.route_core_missing_now(state)):      # 第 12 回合起认命（10-21）
            _own = set(self.rules._field) | {sl.get("name") for sl in state.hand_slots or []} | set(self.rules.hand_book_names())
            _core = set(self.rules.route_core()) - {self.rules.route_carry(), self.rules._carry_piece}   # 主C 同名照买（10-21：守约出现了也没买）
            # 引擎核心件（引擎表 core_set：伞亚连的雅典娜 / 米莱狄…）同名照买、先凑到觉醒（10-21 第 879 局：雅典娜是绝对核心，R7/R11/R14 三次被这条挡掉）
            _core -= {n for n in self.rules.super_core() if not self.rules.is_awakened(n)}     # 超核心件没觉醒前不受限（10-21：只凑到觉醒）
            drop = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.card_options or [])
                    and (lambda n: n and n in _core and n in _own)(
                        (state.card_options[int(k[3:])] or {}).get("name") or "")]
            if drop and len(drop) < len(p["question"]["criteria"]):
                logger.info("N 选 1：核心还缺人 → 已有的核心件先不拿同名（临时规则 core_first_no_copies）")
                self._restrict(p, drop)
        if _IR.on("stack_copies_buy") and self.rules.route_locked():
            # 10-21 审查：成员只认核心阵容（条件是"明世隐在核心阵容里"；route_members 含前 8 人支援，几乎总有他）；
            # 主C 同名 / 现在缺的核心件也留着（原来"只留明世隐"压过了它们）
            _rm = set(self.rules.route_core())
            _also = ({self.rules.route_carry(), self.rules._carry_piece} | self.rules.route_core_missing_now(state)) - {"", None}
            _keep = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.card_options or [])
                     and (state.card_options[int(k[3:])] or {}).get("name") in self.STACK_COPIES
                     and (state.card_options[int(k[3:])] or {}).get("name") in _rm
                     and not self.rules.is_awakened((state.card_options[int(k[3:])] or {}).get("name") or "")]   # 只凑到觉醒（10-21）
            if _keep:
                _keep += [k for k in p["question"]["criteria"] if k.startswith("opt") and k not in _keep
                          and int(k[3:]) < len(state.card_options or [])
                          and (state.card_options[int(k[3:])] or {}).get("name") in _also]
            if _keep and len(_keep) < len(p["question"]["criteria"]):
                logger.info("N 选 1 有核心阵容里的明世隐 → 拿（临时规则 stack_copies_buy）")
                self._restrict(p, [k for k in p["question"]["criteria"] if k not in _keep])
        if _IR.on("entry_trigger_msy") and "明世隐" in self.rules._field:
            _keep = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.card_options or [])
                     and self._entry_trigger((state.card_options[int(k[3:])] or {}).get("name") or "")]
            if _keep and len(_keep) < len(p["question"]["criteria"]):
                logger.info("N 选 1 有触发登场的牌、场上有明世隐 → 拿（临时规则 entry_trigger_msy）")
                self._restrict(p, [k for k in p["question"]["criteria"] if k not in _keep])
        if _IR.on("core_before_level_cards") and self.rules.route_locked():
            _miss = self.rules.route_core_missing(state)
            _nm = {k: (state.card_options[int(k[3:])] or {}).get("name") or "" for k in p["question"]["criteria"]
                   if k.startswith("opt") and int(k[3:]) < len(state.card_options or [])}
            if any(n in _miss for n in _nm.values()):
                drop = [k for k, n in _nm.items() if self.rules.is_level_card(n)]
                if drop and len(drop) < len(p["question"]["criteria"]):
                    logger.info("N 选 1 有缺的核心件 → 去掉加等级牌 %s（临时规则 core_before_level_cards）", [_nm[k] for k in drop])
                    self._restrict(p, drop)
        if _IR.on("single_copy"):
            drop = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.card_options or [])
                    and self.rules.single_copy_owned((state.card_options[int(k[3:])] or {}).get("name") or "", state)]
            if drop and len(drop) < len(p["question"]["criteria"]):
                logger.info("N 选 1 去掉只要 1 张、已经有了的人（临时规则 single_copy）：%s", [state.card_options[int(k[3:])].get("name") for k in drop])
                self._restrict(p, drop)
        if _IR.on("no_lowest_level_cards"):
            # 临时规则 no_lowest_level_cards（2026-10-20 第 844 局：N 选 1 选了扶倾济弱 —— 给最低等级英雄加等级，对养主C 的阵容无效）
            drop = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.card_options or [])
                    and self.rules.never_pick((state.card_options[int(k[3:])] or {}).get("name") or "")]
            if drop and len(drop) < len(p["question"]["criteria"]):
                logger.info("N 选 1 去掉给最低等级英雄的牌（临时规则 no_lowest_level_cards）：%s",
                            [state.card_options[int(k[3:])].get("name") for k in drop])
                self._restrict(p, drop)
        choice, probs, conf, ms = self._ask(p)
        i = int(choice[3:])
        o = state.card_options[i]
        if _IR.on("card_pick_dmg") and self.rules.item_wrong_type(o["name"]):
            # 2026-09-26 定的硬规则：主C 与未来主C 都是物理系时不选法术装（反之亦然）→ 按学生概率取下一个
            for k in sorted(probs, key=lambda k: -probs[k]):
                j = int(k[3:]) if k.startswith("opt") else -1
                if 0 <= j < len(state.card_options) and not self.rules.item_wrong_type(state.card_options[j]["name"]):
                    logger.info("Laya 选的 %s 和主C 伤害类型相反，改选 %s", o["name"], state.card_options[j]["name"])
                    choice, o = k, state.card_options[j]
                    break
        r = self.rules
        if r.model_decides() and (((r.carry_settled() or r.route_locked()) and r.unwanted_second_carry(o["name"])) or r.other_c(o["name"])):
            # 2026-10-11：R23（f9li C-POSITION / g9u）接管时只记录（第 547 局把 Laya 选的海月改成了鲁班大师）
            logger.info("（仅记录）旧 N 选 1 护栏会认为 %s 是第二主C / 别的 C 位", o["name"])
        elif ((r.carry_settled() or r.route_locked()) and r.unwanted_second_carry(o["name"])) or r.other_c(o["name"]):
            # **主C 定了不拿第二主C，有给主C 涨等级的就拿它**（2026-10-06 第 332 局：多拿的花木兰完全没有作用，
            # 不如能给主C 提供等级的角色）。临时护栏：Laya 还没学会，等下一轮标注写进规范后撤掉。
            for k in sorted(probs, key=lambda k: -probs[k]):
                j = int(k[3:]) if k.startswith("opt") else -1
                if not (0 <= j < len(state.card_options)):
                    continue
                n = state.card_options[j]["name"]
                if not r.unwanted_second_carry(n) and not r.other_c(n):
                    logger.info("Laya 选的 %s 是第二主C / 别的 C 位（不给主C 涨等级），改选 %s", o["name"], n)
                    choice, o = k, state.card_options[j]
                    break
        self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya")
        return dict(a, piece=o["name"], x=o["x"], y=o["y"])

    def _talent(self, state, a):
        # **选完之后的放大详情不是新的三选一**：第一张还是刚选的那个天赋、另外两张换成了它送的牌 → 关掉，别再选
        # （2026-09-29 第 12 局：选完「积极整备」后画面是 [积极整备, 星源装甲, 星源合金盾]，被当成三选一连点 7 次、卡 60 秒）
        last = self.__dict__.get("_talent_last")
        names = [t.name for t in state.talents or []]
        if (last and names and names[0] == last[0] and time.time() - last[1] < 30 and set(names) != last[2]):
            logger.info("天赋放大详情（%s + 送的牌 %s）—— 关掉，不再选", names[0], names[1:])
            return {"type": "dismiss_detail"}
        act = self._talent_decide(state, a)
        if act.get("type") == "talent_pick":
            self._talent_last = (act.get("name"), time.time(), set(names))
        return act

    def _official_says(self, carry: str, phrase: str) -> bool:
        """官方推荐阵容里，主C 是 carry 的那几套，阶段说明原文有没有提到 phrase（只读数据，不写死英雄）。"""
        cache = self.__dict__.setdefault("_official_txt", {})
        if carry not in cache:
            import json as _j
            from pathlib import Path as _P
            txt = ""
            try:
                L = _j.loads((_P(__file__).resolve().parents[2] / "data" / "lineups.json").read_text(encoding="utf-8"))
                L = L if isinstance(L, list) else L.get("lineups", [])
                for x in L:
                    st = x.get("stages") or []
                    if st and carry in (st[-1].get("main") or []) and "阿离" in _j.dumps(x, ensure_ascii=False):
                        txt += "".join(s.get("desc") or "" for s in st) + (x.get("desc") or "")
            except Exception:
                pass
            cache[carry] = txt
        return phrase in cache[carry]

    def _drop_dead_faction_talents(self, state, p) -> None:
        """**只对某个派系生效、而我们哪条路都用不上这个派系的天赋，从选项里去掉**（2026-10-13，第 605 局「擂台猛虎」：
        根本用不上；10-12 定：选天赋必须现在和未来用得上）。
        "用得上的派系" = 场上 + 手里 + 各候选路线核心（flex_core_cards）+ 当前路线最终阵容 的英雄派系。
        天赋卡面点名的【X】派系一个都不在里面 → 去掉；全去掉时留着"刷新"，连刷新都没有就不动。"""
        R = self.rules
        from agent.decision import routes as _R
        heroes = set(R._field) | {sl.get("name") for sl in state.hand_slots or []} | set(R.flex_core_cards(state))
        if R._route_key:
            heroes |= set(_R.final_team(_R.by_key(R._route_key) or {}))
        mine = {str(R.pieces.get(h, {}).get("faction") or "")[:2] for h in heroes} - {"", "无阵"}
        crit = p["question"]["criteria"]
        drop = []
        for k in list(crit):
            if not k.startswith("opt"):
                continue
            j = int(k[3:])
            if j >= len(state.talents or []):
                continue
            t = state.talents[j]
            txt = R._card_text(t.name) or getattr(t, "desc", "") or ""
            facs = {f[:2] for f in re.findall(r"【([^】]+)】", txt)} & {"河洛", "逐鹿", "日落", "三分", "大河"}
            if facs and not (facs & mine):
                drop.append(k)
        if not drop:
            return
        left = [k for k in crit if k not in drop]
        if not any(k.startswith("opt") for k in left) and "refresh" not in left:
            return
        names = [state.talents[int(k[3:])].name for k in drop]
        self._restrict(p, drop)
        logger.info("天赋只对没用的派系生效 → 去掉 %s（我们用得上的派系 %s）", names, sorted(mine))

    _OUTPUT_STAT_RE = re.compile(r"(攻击力|法术攻击力|法术强度|暴击率|攻击速度|吸血|法术吸血)\s*[+＋]\s*\d|[+＋]\s*\d+%?\s*(点)?(物理|法术)?(攻击力|吸血)")

    def _special_equip_fits(self, item: str, carry: str) -> bool:
        """特殊装备对主C 配不配：穿得上（卡面限制 / 伤害类型），且是对口的输出装（hero_traits 判对口，或卡面直接加攻击 / 法强 / 暴击 / 攻速 / 吸血）。
        静谧之铃（只有控蓝 + 按穿戴者法攻的小额灼烧，不加属性）对沈梦溪算不配（2026-10-20）。"""
        from agent.decision import hero_traits as _ht
        if not self.rules._equip_allowed(item, carry):
            return False
        if _ht.live_fits(item, carry)[0]:
            return True
        d = _ht.item_dmg(item)
        return bool(self._OUTPUT_STAT_RE.search(self.rules._card_text(item) or self.rules.pieces.get(item, {}).get("skill") or ""))             and (not d or d == _ht.dmg_type(carry))

    def _drop_mismatch_talents(self, state, p) -> None:
        """**装备类天赋和主C / 未来主C 伤害类型相反 → 去掉**（2026-09-26：噬神之书是法术道具，主C 与未来主C 都是物理系时
        不能选；N 选 1 / 穿戴早就拦了，天赋这条路漏了 —— 2026-10-13 第 614 局孙尚香 / 虞姬都是物理，Laya 0.47 选了噬神之书）。"""
        from agent.decision import hero_traits as _HT
        R = self.rules
        heroes = [h for h in (R._carry_piece, R.route_carry()) if h]
        crit = p["question"]["criteria"]
        drop = [k for k in crit if k.startswith("opt") and int(k[3:]) < len(state.talents or [])
                and _HT.item_mismatch(state.talents[int(k[3:])].name, heroes)]
        left = [k for k in crit if k not in drop]
        if not drop or not (any(k.startswith("opt") for k in left) or "refresh" in left):
            return
        names = [state.talents[int(k[3:])].name for k in drop]
        self._restrict(p, drop)
        logger.info("天赋装备和主C 伤害类型相反 → 去掉 %s（主C / 路线主C %s）", names, heroes)

    def _hero_names(self) -> list:
        hs = self.__dict__.get("_hero_names_cache")
        if hs is None:
            hs = self._hero_names_cache = sorted({n for n, c in self.rules.pieces.items()
                                                  if c.get("category") == "英雄牌" and "(" not in n and not c.get("no_pop")},
                                                 key=len, reverse=True)
        return hs

    def _talent_decide(self, state, a):
        # 规则若决定刷新，它已经把"这一级刷过了"记上 —— 那刷新这个选项当时是可用的
        # 刷没刷过只认画面（「全局N次」/「已用完」）+ 自己真执行过的刷新（2026-10-20 第 839 局：R2 规则层预判"要刷新"就记了
        # _talent_refreshed_level，实际执行的是 Laya 选的天赋，后面 R4 / R8 的题里都没了刷新选项 —— 整局 1 次的刷新白白丢了）
        can_ref = bool(state.talent_can_refresh) and self.__dict__.get("_talent_ref_round") != state.round_no
        # **共战天赋优先**（2026-10-12：双生突袭「马可波罗、百里守约共战，释放技能时创造分身朝敌人释放技能」——
        # 效果很强，可以提高优先级，凑上了基本都是第一）。卡面写着"获得1张A和1张B"（崛起）直接拿；
        # 只要求共战的（双生突袭）在已经有其中一人（场上 / 手里）或已选过同组天赋时拿。（2026-10-12 更正：交给 Laya 判断，这里只记录）
        R = self.rules
        owned = set(R._field) | {sl.get("name") for sl in state.hand_slots or []} | R.duo_members()
        heroes = [n for n, v in R.pieces.items() if v.get("category") == "英雄牌"]
        for t in state.talents or []:
            txt = (R.pieces.get(t.name, {}).get("skill") or "") + R._card_text(t.name)
            if "共战" not in txt:
                continue
            names = [h for h in heroes if h in txt]
            if len(names) >= 2 and (re.search(r"获得1张", txt) or owned & set(names)):
                # 2026-10-12 更正：仍交给 Laya 判断，这个打法 Laya 也要学会 → 只记录，照常出题（规范 49.10 教老师）
                logger.info("（仅记录）共战天赋出现：%s（%s；已有 %s）—— 交给 Laya", t.name, "+".join(names), sorted(owned & set(names)))
                break
        if self.rules.route_locked() and a.get("type") != "talent_refresh":
            # **路线推荐的天赋出现就拿**（2026-09-29 等级来源分析：守约路线推荐的「幸运硬币/升值/古币翻新」15 局一次没拿，
            # 古币这条等级线因此没放大；天赋是路线的等级倍率来源）。推荐表来自阵容库的 talent_cards。
            from agent.decision import routes as _R
            lu = _R.by_key(self.rules._route_key) or {}
            want = [t for t in (lu.get("talent_cards") or []) if t] + list(self.rules.route_addons())   # + 顺带养 T 的补件天赋
            # 日落海主C 的路线：阿科米亚核心天赋出现就拿（2026-10-02 前三 vs 5~6 名分析：日落海路线进前三的 7 局里 5 局拿到
            # 场外供能 / 核心裂痕，第 4~6 名 9 局一次没出现，p≈0.005；这两张一出现就被选了，差别在有没有刷出来）
            _rc = lu.get("carry") or ""
            if (self.rules.pieces.get(_rc, {}).get("faction") or "") == "日落海":
                # **按复利强弱排优先级**（2026-10-03：核心裂痕 = 每回合一张盗火把核心抽给主C；
                # 动能灌注 = 普攻次数 → 核心；场外供能 = 线性；积少成多 = 日落海英雄涨级回流核心）。
                # 能量改造（核心不再分配、按核心等级触发整备）要核心够大才赚，读不到核心等级，先不主动拿。
                want = ["核心裂痕", "动能灌注", "场外供能", "积少成多"] + want
            if _rc and self._official_says(_rc, "无限释放"):
                # 官方阵容（同主C）原文写着"转移无限释放技能天赋直接起飞"→ 满级专属天赋拿 枫之洪流
                # （2026-10-09 守约 4 局：6 级三选一 落英汲取 / 枫之洪流 / 翩跹馈赠，Laya 0.36~0.50 选了 3 次翩跹馈赠、1 次落英汲取，
                # 题面里这几张写的是 card not in database，学生看不到效果）
                want = ["枫之洪流"] + want
            elif _rc and self._official_says(_rc, "普攻一定次数转移"):
                # 官方 "瞬步4.0狄仁杰"：6 级"选择普攻一定次数转移天赋" = 翩跹馈赠（累积普攻 15 次再获得 1 次转移）
                want = ["翩跹馈赠"] + want
            if self.rules.attack_roll_summoners():
                want.append("因缘际会")       # 高攻速主C：多一个按普攻次数给全队涨级的因缘图腾
            order = {n: i for i, n in reversed(list(enumerate(want)))}
            hits = [t for t in state.talents or [] if t.name in order]
            hit = min(hits, key=lambda t: order[t.name]) if hits else None    # 按推荐顺序，不按选项顺序
            if hit is not None and self.rules.model_decides():
                # 2026-10-12：天赋题 Laya 一直在答，规则抢先等于不让它学 → 只记录
                logger.info("（仅记录）旧规则会按路线推荐直接拿天赋：%s（路线 %s 推荐 %s）", hit.name, lu.get("name"), want)
            elif hit is not None:
                logger.info("路线推荐天赋：%s（路线 %s 推荐 %s）", hit.name, lu.get("name"), want)
                self.rules.note("路线推荐天赋", round=state.round_no, 天赋=hit.name, 路线=lu.get("name"))
                return {"type": "talent_pick", "index": hit.index, "name": hit.name}
        p = self.B.talent_point(state.to_dict(), self._c(state), self._meta(state),
                                refreshed=a.get("type") == "talent_refresh",
                                index=a.get("index"), can_refresh=can_ref)
        if not p:
            return a
        if _IR.on("talent_dead_faction"):
            self._drop_dead_faction_talents(state, p)
        if _IR.on("talent_dmg_mismatch"):
            self._drop_mismatch_talents(state, p)
        if _IR.on("talent_sunset_core_drain"):
            # 临时规则（2026-10-17 第 682 局：日落海路线不能选能源置换，根本没法玩——每回合核心 −20 级）
            r = self.rules
            carries = [c for c in (r.route_carry(), r._carry_piece) if c]
            if any("日落海" in ((r.pieces.get(c) or {}).get("synergies") or []) for c in carries):
                drop = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.talents)
                        and re.search(r"阿科米亚核心失去\s*\d+\s*级", r._card_text(state.talents[int(k[3:])].name) or "")]
                if drop and len(drop) < len([k for k in p["question"]["criteria"] if k.startswith("opt")]):
                    logger.info("日落海路线去掉扣核心等级的天赋（临时规则 talent_sunset_core_drain）：%s",
                                [state.talents[int(k[3:])].name for k in drop])
                    self._restrict(p, drop)
        if _IR.on("talent_low_value"):
            # 临时规则 talent_low_value（2026-10-19 第 782 局 R2：节能商店没什么用，−1 能量不一定落在需要的那张上）：
            # 判为低价值的天赋去掉（还剩别的选项或刷新时）
            # 因缘际会（召唤因缘图腾，同少司缘）：只有攻速型主C 吃得到（2026-10-19 晚第 803 局：主C 不是攻速型时因缘图腾没什么用）
            from agent.decision import hero_traits as _htl
            # 2026-10-20（r9zze_test 老师：1 阶临时主C 孙膑按逐鹿判，把河洛新援误判成低价值；54.23-2 主C 未定按未来主C）：
            # 路线已锁且路线主C 已到手 → 路线主C；现任主C 是 1 阶临时主C → 不按主C 判（传空，主C 类条件不判）
            _rcl = self.rules.route_carry() if self.rules.route_locked() else ""
            _own = set(self.rules._field) | {sl.get("name") for sl in state.hand_slots or []}
            try:            # 10-22 第 904 局（神鹰锻匠）：绑阵营的天赋在路线没锁时按开局配置预计的路线主C 判
                from agent.decision import lord_config as _LCo2, talent_conditions as _tco2
                _oc = _LCo2.load().opening_choice(state.banned_faction or "") if not self.rules.route_locked() else None
                _tco2.EXPECTED_ROUTE_CARRY = ((_oc or {}).get("members") or ("",))[0]
                _tco2.BANNED_FACTION = _LCo2._norm_faction(state.banned_faction or "")
                _sc = self.rules._carry_piece or ""
                # 10-22 第 929 局：一个响指——场上等级最低的 4 人是不是全是路线外占位（路线没锁判不了）
                if len(self.rules._field) >= 1:
                    _me4 = self.rules.min_engine() or {}
                    _fin4 = set(self.rules.super_core()) | set(self.rules.ENGINE_ANCHORS_LOW4) | {self.rules._carry_piece}
                    if self.rules.route_locked():
                        _fin4 |= (set(self.rules._route_team()) | set(self.rules.route_core()) | {self.rules.route_carry()}
                                  | set(_me4.get("early_priority_full") or _me4.get("early_priority") or []))
                    _fin4 -= {"", None}
                    _low4 = sorted(self.rules._field, key=lambda n: self.rules._unit_lv.get(n) or 0)[:4]
                    _bad4 = [n for n in _low4 if n in _fin4 or self.rules.is_awakened(n)]
                    _tco2.LOWEST_OFFROUTE = (not _bad4, _bad4)
                else:
                    _tco2.LOWEST_OFFROUTE = None
                _tco2.SETTLED_CARRY = _sc if (self.rules.final_carry_settled() and _sc in self.rules._field and (
                    (self.rules._unit_lv.get(_sc) or 0) >= 10 or self.rules._maple_bound == _sc)) else ""
            except Exception:                               # noqa: BLE001
                pass
            _cur = self.rules._carry_piece or ""
            _temp = bool(_cur) and int((self.rules.pieces.get(_cur) or {}).get("quality") or 0) <= 1
            _Cl = (_rcl if _rcl and _rcl in _own else ("" if _temp else _cur)) or ""
            if not _Cl:
                # 10-20 第 838 局 R4（选了中坚力量，但主C 不是 3 阶）：主C 未定（1 阶临时孙膑、路线没锁）→
                # 按未来主C 判（54.23-2）：锁定 / 首选路线的主C
                _Cl = self.rules.route_carry() or ""
            _low = set(self.LOW_VALUE_TALENTS) | ({"因缘际会"} if _Cl and not _htl.attack_speed_carry_live(_Cl) else set())
            # 2026-10-20 硬条件表（data/talent_conditions.json，逐张复审）：
            # 按我方阵容（路线最终阵容 + 场上 + 手里）和主C 求值，等级 C 或硬条件不满足的去掉（战术牌没逐鹿 / 共战两人不全在 / 阵营不对 …）
            from agent.decision import talent_conditions as _tcd
            _team = set(self.rules._route_team()) | set(self.rules._field) | {sl.get("name") for sl in state.hand_slots or [] if sl.get("name")}
            _board = self.rules.final_board(state)
            for _k in p["question"]["criteria"]:
                if _k.startswith("opt") and int(_k[3:]) < len(state.talents):
                    _tn = state.talents[int(_k[3:])].name
                    _g, _why = _tcd.check(_tn, _team, _Cl or "", _rcl if _rcl in _own else "", unknown_carry_fails=True,      # 路线主C 没到手 = 路线没定（54.28-2）
                                          board=_board)                    # 10-21：按最终可摆阵容计数（同心协力等）
                    if _g == "?" and _Cl and (self.rules.pieces.get(_tn) or {}).get("category") == "装备牌":
                        # 三选一里直接给装备（末世 / 特殊装备）：不在天赋表里 → 按主C 能不能用判（第 855 局 R11：沈梦溪法术技能型选了末世，没人能穿）
                        _okq, _whyq = _htl.live_fits(_tn, _Cl)
                        if _whyq == "unclassified":
                            # 分类表没收录（末世）→ 读卡面原文：靠普攻生效的只给攻速型主C
                            _txq = str((self.rules.pieces.get(_tn) or {}).get("skill") or "")
                            _okq = not ("普通攻击" in _txq and _Cl in _htl._lr("no_onhit_carries"))     # 规范 54.29-18：只禁守约 / 沈梦溪 / 诸葛亮
                            _whyq = "卡面靠普攻、主C 是技能型（守约 / 沈梦溪 / 诸葛亮）" if not _okq else ""
                        if not _okq:
                            _low.add(_tn)
                            logger.info("天赋三选一的装备 %s 对主C %s 不适用（%s）", _tn, _Cl, _whyq)
                    if _g != "C" and _tn in _low and _tn in ("随机成长", "随机进化", "随机强化"):
                        _low.discard(_tn)            # 主C 是最终阵容最高品阶（4 阶韩信 / 蒙犽）时这三张有用（规范 54.30-12，10-21 定并列随机给）
                    if _g == "C":
                        _low.add(_tn)
                        logger.info("天赋 %s：硬条件表判 C（%s）", _tn, _why or (_tcd.table().get(_tn) or {}).get("why", "")[:60])
            if _Cl and _htl._is_sunset(_Cl) and any(t.name == "学识宝石" for t in state.talents)                     and any(t.name not in _low and t.name != "学识宝石" for t in state.talents):
                # 2026-10-21 第 892 局（学识宝石 vs 磐石之心，主C 亚连）：磐石之心更好，会让主C 非常硬；日落海主要靠核心等级，
                # 主C 基础等级不会很高，学识宝石（每回合 +8%）到最后也提供不了多少等级 —— 日落海主C 时学识宝石让位
                _low.add("学识宝石")
                logger.info("主C %s 是日落海（等级靠核心）→ 学识宝石按比例涨的基础等级很少，不选", _Cl)
            drop = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.talents)
                    and (state.talents[int(k[3:])].name in _low or self.rules.never_pick(state.talents[int(k[3:])].name))]
            _first = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.talents)
                      and state.talents[int(k[3:])].name in self.rules.CARRY_ANY_DEF
                      and state.talents[int(k[3:])].name not in set(self.rules._worn.get(self.rules._carry_piece or "", ()))]
            if _first:
                # 2026-10-22：磐石之心优先，装给主C——三选一里有它（主C 还没穿）就选它
                logger.info("天赋三选一有 %s → 优先选它给主C", [state.talents[int(k[3:])].name for k in _first])
                drop = [k for k in p["question"]["criteria"] if k not in _first]
            # 刷新（2026-10-20 第 839 局：从第 8 回合开始；之前是青铜天赋，刷新也刷不出好的；只有全是 C 的时候刷新）：
            # 第 8 回合起三个全是 C → 只留刷新；其余情况去掉刷新
            _opts = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.talents)]
            # **特殊高级装备三选一**（静谧之铃 / 青莲剑谱 / 征服之剑 这一档，不是 8 件基础装备 + 32 件四选一成品）三件都配不上主C → 不分回合刷新
            # （2026-10-20 第 844 局 R5：静谧之铃这一档的特殊高级装备要支持刷新；普通装备三选一不刷新）
            _spec_bad = False
            if _Cl and _opts:
                _names = [state.talents[int(k[3:])].name for k in _opts]
                _cards = [self.rules.pieces.get(n, {}) for n in _names]
                if all(c.get("category") == "装备牌" and not self.rules._is_normal_equip(n) for n, c in zip(_names, _cards)):
                    _spec_bad = not any(self._special_equip_fits(n, _Cl) for n in _names)
            if "refresh" in p["question"]["criteria"] and _spec_bad:
                logger.info("特殊装备三选一 %s 都配不上主C %s → 刷新（不分回合）", [state.talents[int(k[3:])].name for k in _opts], _Cl)
                drop = list(_opts)
            elif "refresh" not in p["question"]["criteria"] and _opts and len(drop) == len(_opts):
                # 2026-10-22 第 926 局 R7（随机进化 / 全军列阵 / 法潮灵杖，全是 C、刷新已用）：Laya 选了法潮灵杖——法术装、物理主C 玄策用不了、
                # 也没人能穿，拿了白占手牌。全是 C 又不能刷新时，先去掉没人能穿的装备，在剩下的 C 里选
                _dead = [k for k in _opts if (self.rules.pieces.get(state.talents[int(k[3:])].name) or {}).get("category") == "装备牌"
                         and not any(self.rules._equip_allowed(state.talents[int(k[3:])].name, n) and not self.rules._no_gear(n)
                                     and n not in self.rules.NO_EQUIP_UNITS
                                     and (n != _Cl or _htl.live_fits(state.talents[int(k[3:])].name, n)[0])
                                     for n in self.rules._field)]
                drop = _dead if len(_dead) < len(_opts) else []
            elif "refresh" in p["question"]["criteria"]:
                if _opts and len(drop) == len(_opts):
                    # 2026-10-21 第 900 局 R2（多彩古币 / 随机进化 / 协同一致）：三个都不好，不如刷新——全是 C 就刷新，不再限第 8 回合起
                    logger.info("第 %s 回合天赋三个全是 C %s → 刷新", state.round_no, [state.talents[int(k[3:])].name for k in drop if k.startswith("opt")])
                else:
                    if len(drop) == len(_opts):
                        drop = []          # （全是 C 已在上面刷新；这里不会走到）
                    drop = drop + ["refresh"]
            if drop and len(drop) < len([k for k in p["question"]["criteria"] if k.startswith("opt") or k == "refresh"]):
                logger.info("去掉登记的低价值天赋 / 刷新（临时规则 talent_low_value）：%s",
                            [state.talents[int(k[3:])].name if k.startswith("opt") else k for k in drop])
                self._restrict(p, drop)
        if _IR.on("talent_offroute_gift") and self.rules.route_locked():
            # 临时规则 talent_offroute_gift（2026-10-19 第 787 局 R9：献祭之赐送李信、主C 沈梦溪，选了又卖；
            # 送英雄或与特定英雄有关的天赋全部打标记 → data/talent_hero_ties.json）：
            # 点名英雄全不在路线最终阵容 / 核心引擎 / 主C 的 gives / duo / about 天赋 → 去掉（gives_plus 有额外收益，不去）
            r = self.rules
            _team = set(r._route_team()) | set((r.min_engine().get("early_priority_full") or r.min_engine().get("early_priority")) or []) | {r._carry_piece}
            _ties = self._talent_ties()
            drop = []
            for k in p["question"]["criteria"]:
                if not (k.startswith("opt") and int(k[3:]) < len(state.talents)):
                    continue
                t = _ties.get(self.F.clean_name(state.talents[int(k[3:])].name) or "") or {}
                if t.get("kind") in ("gives", "duo", "about") and t.get("heroes") and not (set(t["heroes"]) & _team):
                    drop.append(k)
            if drop and len(drop) < len([k for k in p["question"]["criteria"] if k.startswith("opt") or k == "refresh"]):
                logger.info("天赋点名的英雄不在路线里（临时规则 talent_offroute_gift）：%s", [state.talents[int(k[3:])].name for k in drop])
                self._restrict(p, drop)
        if _IR.on("talent_unplayable"):
            # 2026-10-18（第 751 局 R2 战争头牌）：我们执行不了的天赋（感知读不到每回合品阶数字）去掉；混沌裂隙同理（54.11 / f9lw）
            drop = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.talents)
                    and re.search(r"品阶数字", self.rules._card_text(state.talents[int(k[3:])].name) or "")]
            if drop and len(drop) < len([k for k in p["question"]["criteria"] if k.startswith("opt") or k == "refresh"]):
                logger.info("去掉执行不了的天赋（临时规则 talent_unplayable）：%s", [state.talents[int(k[3:])].name for k in drop])
                self._restrict(p, drop)
        if _IR.on("duo_needs_carry"):
            # 2026-10-18（第 748 局 R9 选了兄妹 0.52，主C 守约不是铠 / 露娜）：兄妹只有在其中一个是主C 时才有意义
            r = self.rules
            C = self._real_carry(state) or r.route_carry()
            drop = []
            for k in p["question"]["criteria"]:
                if not (k.startswith("opt") and int(k[3:]) < len(state.talents)):
                    continue
                tx = r._card_text(state.talents[int(k[3:])].name) or ""
                # 2026-10-19：双人组合类天赋不涉及主C 英雄就根本没用——不只"共战"：卡面点名具体英雄
                # （兄妹 / 舞力全开 / 身外身 / 巨像 / 双生突袭 / X·强化 / 技能突破：X …）而主C 不在其中 → 去掉
                _named = [h for h in self._hero_names() if h in tx]
                # 送的英雄牌里有路线主C / 最终阵容件时不删（r9zy 老师：崛起送路线主C 守约、兄妹送缺的最低配套件铠 —— 组合效果没用，送的牌有用）
                # 2026-10-19 晚：只有两个人都有用时才选——送的英雄不单独算（撤掉上面 r9zy 的"送路线件不删"例外）；
                # 路线已锁时另一个人也要在最终阵容 / 核心引擎里
                _rt = ((r._route_team() | set((r.min_engine().get("early_priority_full") or r.min_engine().get("early_priority")) or []) | {C}) - {"", None}) if r.route_locked() else set()
                _other_bad = bool(_rt) and len(_named) >= 2 and any(h not in _rt for h in _named)
                if C and (("共战" in tx and C not in tx) or (len(_named) >= 2 and C not in _named) or _other_bad):
                    drop.append(k)
            if drop and len(drop) < len([k for k in p["question"]["criteria"] if k.startswith("opt") or k == "refresh"]):
                logger.info("共战天赋的两人里没有主C %s → 去掉（临时规则 duo_needs_carry）：%s", C, [state.talents[int(k[3:])].name for k in drop])
                self._restrict(p, drop)
        if _IR.on("coin_machine_after_carry"):
            # 2026-10-19（736 R9 / 758 R11 选了古币机 0.42 / 0.46）：古币机偏团队收益，要比较同组天赋哪个对主C 效果更强
            # 古币机在场后古币全打给它（rules 6873）：河洛主C 每张少 +2 永久级；奖励是随机河洛英雄牌 / 杂项 → 主C 已定时去掉
            r = self.rules
            C = r._carry_piece or r.route_carry()
            drop = [k for k in p["question"]["criteria"] if k.startswith("opt") and int(k[3:]) < len(state.talents)
                    and state.talents[int(k[3:])].name == "古币机"]
            if C and drop and len(drop) < len([k for k in p["question"]["criteria"] if k.startswith("opt") or k == "refresh"]):
                logger.info("主C %s 已定 → 去掉古币机（分走主C 古币的团队天赋；临时规则 coin_machine_after_carry）", C)
                self._restrict(p, drop)
        choice, probs, conf, ms = self._ask(p)
        self._log(p, choice, probs, conf, ms, p.get("rule_choice"), "laya")
        if choice == "refresh":
            self.rules._talent_refreshed_level = state.level     # 这一级的刷新机会用掉了
            self._talent_ref_round = state.round_no              # 同一回合画面没刷新前不再出刷新
            return {"type": "talent_refresh"}
        i = int(choice[3:])
        t = state.talents[i]
        why = self.rules.talent_unsupported(t.name, getattr(t, "desc", ""))
        if why:
            # **agent 做不出来的天赋不选**（扩编名额 / 扩编高峰：选了还要花 8 能量"购买"，而购买入口还没实机确认，
            # 17 次选了 0 次买 —— 卡牌审查第 1 条）。这是"能不能执行"，不是价值判断：按学生概率换下一个能执行的
            for k in sorted(probs, key=lambda k: -probs[k]):
                j = int(k[3:]) if k.startswith("opt") else -1
                if 0 <= j < len(state.talents) and not self.rules.talent_unsupported(
                        state.talents[j].name, getattr(state.talents[j], "desc", "")):
                    logger.info("Laya 选的天赋 %s 做不出来（%s），改选 %s", t.name, why, state.talents[j].name)
                    t = state.talents[j]
                    break
        return {"type": "talent_pick", "index": t.index, "name": t.name}

    def _shop_event_ok(self, state, a) -> bool:
        """规则没提商店动作时，要不要主动问 Laya 商店题：准备阶段、店开着、本回合 Laya 没说停手、
        规则给的是空决策或"关店"（不是空场必须先上人的那种）。"""
        if not (state.phase == "prepare" and state.shop_open and "shop" in self.kinds):
            return False
        if self.rules._stop_shop_round == state.round_no or not state.shop:
            return False
        if any("转瞬" in (self.rules.pieces.get(sl.get("name"), {}).get("skill") or "")
               and self.rules.pieces.get(sl.get("name"), {}).get("category") == "效果牌"
               and self.rules.group_card_ok(sl.get("name"))[0]          # 打不出去的转瞬牌不挡商店题（审查 10-09）
               for sl in state.hand_slots or []):
            # 手里有「转瞬」牌（古币 / 精致古币…，开战即删）：先关店把它打掉，别为商店题把店一直开着
            # （2026-10-09 第 477 局：手里有好几张古币时用得很慢，总漏用 1~2 张：R7/R11/R16 回合结束时店开着、古币没打）
            return False
        if a is not None and not (a.get("type") == "toggle_shop" and a.get("intent") == "close"):
            return False
        if not self.rules._field and self.rules._board_work_pending(state):
            return False                  # 场上空着、手里有人：先关店上人（空场必输）
        return True

    def _shop(self, state, a):
        # **商店卡名一张都没读出来（刚开店的动画帧）就先不出题**（2026-10-13 第 595 局：第 1、2 回合商店读成 [None]*4，
        # 买牌选项全空，经济网络又删了停手 / 锁店 → 只剩"刷新"1.00，白花 2 能量）。等下一帧读出来再问。
        # 读不出的格子也标成"已售"（✗），和真的买空分不开 → 按持续时间区分：整排没卡名先等 1.5 秒，
        # 还读不出才当真的卖空、照常出题（第 598 局：第一版只认"没卡名且没售"，被 ✗ 绕过去又白刷了一次）
        import time as _t
        sh = state.shop or []
        if sh and all(not getattr(sl, "piece", None) for sl in sh):
            now = _t.time()
            last = self.__dict__.get("_blank_shop_last", 0.0)
            if now - last > 3.0 or "_blank_shop_since" not in self.__dict__:   # 中间关过店 / 隔了很久 / 中间有过非空帧：重新计时
                self._blank_shop_since = now
            self._blank_shop_last = now
            if now - self._blank_shop_since < 1.5:
                return None
        else:
            self.__dict__.pop("_blank_shop_since", None)
        # **局面没变就不重问**（2026-10-09：每帧都问太耗性能）：签名 = 回合 / 能量 / 商店内容 / 手牌数 / 场上的人。
        # 刚开店、刚刷新、刚买完一张、钱或人变了，签名都会变
        sig = (state.round_no, state.gold, tuple((sl.piece, sl.sold) for sl in state.shop or []),
               len(state.hand_slots or []), frozenset(self.rules._field), a.get("type") if a else None,
               bool(state.level_btn), state.level_cost)      # 升级按钮这帧才出现 → 重新问（审查 10-09）
        if _IR.on("sunset_equip_buy"):
            # 临时规则 sunset_equip_buy（2026-10-19 确认：日落海英雄当主C、装备没满时，商店出日落海装备可以买）：
            # 主C 是日落海英雄、没满 3 件、钱够 → 直接买卡面"【日落海】专属装备"的牌（远洋宝藏）
            r = self.rules
            C = self._real_carry(state)      # 1 阶临时主C 不算（10-20）
            # 10-20（review-interim-conflicts 第 10 条）：手里已经攒着的专属装备也算进件数、每回合最多买一次（原来会重复买远洋宝藏）
            _held = sum(1 for sl in state.hand_slots or [] if "【日落海】专属装备" in (r._card_text(sl.get("name") or "") or ""))
            if (C and "日落海" in ((r.pieces.get(C) or {}).get("synergies") or []) and len(r._worn.get(C) or ()) + _held < 3
                    and self.__dict__.get("_sunset_buy_round") != state.round_no):
                for sl in state.shop or []:
                    if (sl.piece and not sl.sold and "【日落海】专属装备" in (r._card_text(sl.piece) or "")
                            and (state.gold or 0) >= (sl.cost or 0) > 0):
                        self._sunset_buy_round = state.round_no
                        logger.info("主C %s 是日落海、%d/3 件 → 买 %s（临时规则 sunset_equip_buy）", C, len(r._worn.get(C) or ()), sl.piece)
                        return {"type": "buy_piece", "slot": sl.index, "piece": sl.piece, "cost": sl.cost,
                                "cost_est": False, "round": state.round_no, "why": "sunset_equip_buy"}
        last = self.__dict__.get("_shop_last")
        if last and last[0] == sig and isinstance(last[1], dict) and last[1].get("type") == "lock_shop":
            last = None     # 锁店是开关：重放一次就是解锁（2026-10-11 第 565 局第 12 回合连点 4 次 = 没锁），不走缓存
        if last and last[0] == sig:
            if last[2] < 3:
                self._shop_last = (sig, last[1], last[2] + 1)
                return last[1]
            # 同一个答案连续返回 3 次局面都没变 = 这个动作点了没生效（审查 10-09）：作废，这帧交还规则
            if self.rules.model_decides():
                # 2026-10-12 第 576 局：这里把规则自己的"买狂铁 / 买虞姬"放行了，模型没被问到。
                # 模型接管时本帧不动、清掉缓存，下一帧重新出题
                self._shop_last = None
                return None
            return None if a is None else a
        act = self._shop_decide(state, a)
        self._shop_last = (sig, act, 0)
        return act

    def _entry_trigger(self, name: str) -> bool:
        """指定英雄 / 所有英雄再触发一次登场（唤醒 / 集体唤醒）——按卡面原文认，同 rules 第 7609 行口径。"""
        t = self.rules._card_text(name) or ""
        return bool(name) and ("触发其登场效果" in t or "触发所有英雄的登场效果" in t)

    def _stack_copy_lock(self, state):
        """2026-10-23：明世隐在场且还要继续买时，商店里有明世隐但钱不够也要锁店。
        与 stack_copies_buy 同一判据（路线核心里的明世隐、在场、还要叠）：店里有他、钱不够买 → 锁店（每回合一次）。"""
        r = self.rules
        if not (_IR.on("stack_copies_buy") and r.route_locked() and state.shop_open) or r._locked_round == state.round_no:
            return None
        _rm = set(r.route_core())
        hit = [sl for sl in state.shop if sl.piece in self.STACK_COPIES and sl.piece in _rm and not sl.sold
               and sl.piece in r._field and not r.is_awakened(sl.piece)
               and (state.gold or 0) < (sl.cost or 3)]
        if not hit:
            return None
        r._locked_round = state.round_no
        logger.info("场上的 %s 还要叠、店里有他但钱不够（%s 能量）→ 锁店留到下回合", hit[0].piece, state.gold)
        return {"type": "lock_shop", "pieces": [sl.piece for sl in state.shop if sl.piece and not sl.sold]}

    def _shop_decide(self, state, a):
        pre = self.rules.preempt_shop(state)      # 进度表升级 / 引擎零件（2026-09-26 定：这两条先于 Laya）
        if pre:
            return pre
        _lk = self._stack_copy_lock(state)
        if _lk:
            return _lk
        s = state.to_dict()
        p = self.B.shop_point(s, dict(self._c(state), _lock_ok=self.rules._locked_round != state.round_no),
                              self._meta(state), act=a)
        if not p:
            if self.rules.model_decides() and a is not None and a.get("type") in ("buy_piece", "refresh_shop", "lock_shop"):
                logger.info("（仅记录）商店题没出成，规则想 %s —— 模型接管时不放行", a.get("type"))
                return None
            return a
        _obs = [k for k in p["question"]["criteria"] if k.startswith("buy")
                for nm in [next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or ""]
                if nm and nm not in self.rules._field and not self.rules._is_carry(nm)
                and self.rules._obsolete(nm, state.round_no)]
        if _obs and len(_obs) < len(p["question"]["criteria"]):
            # 上不了场的不买（第 726 局 R7：花 3e 买 1 阶李元芳，第 6 回合后 _obsolete 永不上场；场上有同名可合成的除外）—— 合法性，与 decide_deploy 同一判据
            self._restrict(p, _obs)
            logger.info("商店题去掉第 6 回合后上不了场的 1 阶英雄：%s", _obs)
        if self.rules.core_roll_active(state) and self.rules.route_locked():
            # 10-22 第 947 局 R11（东皇太一买了又不上场）：雅典娜（核心）还缺时先花 3e 买路线外的东皇填空位，随后刷到雅典娜占了位 →
            # 核心件先刷期间，只买路线里的人 / 核心 / 场上同名（合成），路线外的填位英雄不买
            r = self.rules
            from agent.decision import meta as _mte
            _cape = int(_mte.POP_CAP_BY_LEVEL.get(state.level or 0, 0) or 0) + int(r.pop_bonus or 0)
            _hh = len({sl.get("name") for sl in state.hand_slots or [] if (r.pieces.get(sl.get("name") or "") or {}).get("category") == "英雄牌"
                       and sl.get("name") not in r._field})
            _empty2 = bool(_cape and _cape - (len(r._field) + _hh) >= 2)      # 空位 ≥2：先补人，不按路线挑（10-23）
            _keep_n = (set(r._route_team()) | set(r.route_core()) | set(r.super_core()) | set(r._field)
                       | {r.route_carry(), r._carry_piece}) - {"", None}
            _off = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    for nm in [next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or ""]
                    if nm and (r.pieces.get(nm) or {}).get("category") == "英雄牌" and nm not in _keep_n]
            if _off and not _empty2 and len(_off) < len(p["question"]["criteria"]):
                self._restrict(p, _off)
                logger.info("核心件 %s 还缺、钱先刷 → 商店题去掉路线外的英雄 %s（临时规则 core_roll_first）",
                            sorted(r.route_core_missing_now(state)), _off)
        if _IR.on("not_wanted_no_buy") and self.rules.route_locked():
            # 临时规则 not_wanted_no_buy（第 796 局：锁伞狄速射，R47 题面（f9w）把钟馗写成 ROUTE final team，连买两张；
            # 狄仁杰阵容里不该有钟馗）：引擎表 not_wanted 的人不买
            _nw = set(self.rules.min_engine().get("not_wanted") or [])
            _nwk = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    and (next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "") in _nw]
            if _nwk and len(_nwk) < len(p["question"]["criteria"]):
                self._restrict(p, _nwk)
                logger.info("商店题去掉本路线不要的人 %s（临时规则 not_wanted_no_buy）", _nwk)
        cap = state.population_cap or 0
        here = self.rules.pop_now(state)[0]
        board_full = cap > 0 and here >= cap
        if self.rules.route_locked() and board_full:
            # 路线锁定且**满员**时：商店只买路线成员 / 已有英雄的同名 / 非英雄牌（2026-09-28）。
            # 有空位时不过滤 —— 先把场面铺满（第 18 局：路线成员刷不到、过滤又不让买别人，整局只买 11 张）
            owned = [sl.get("name") for sl in state.hand_slots or []]
            drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    and not self.rules.route_allows_hero(
                        next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "", owned)]
            if drop and len(drop) < len(p["question"]["criteria"]):
                if self.rules.model_decides():
                    # 2026-10-12 清查：模型接管时只记录（题面有 ROUTE / MIN ENGINE / lineup 行，买谁交给学生）
                    logger.info("（仅记录）旧规则会在满员时只留路线成员，去掉 %s", drop)
                else:
                    self._restrict(p, drop)
        _md = self.rules.model_decides()        # 2026-10-11：模型接管时攒不攒钱由经济网络定（经"升级预算"落到合法性），不再按进度表删选项
        if not _md and self.rules.saving_for_level(state) and "refresh" in p["question"]["criteria"]                 and len(p["question"]["criteria"]) > 1:
            # 攒钱升级：差升级费不到 8、等级落后进度表 → 不付费刷新（2026-09-30：盲目刷新，一直升不到 6 级）
            self._restrict(p, ["refresh"])
            logger.info("攒钱升级（能量 %s / 升级费 %s，落后进度表）→ 商店题去掉付费刷新", state.gold, state.level_cost)
        if self.rules.carry_settled() or self.rules.route_locked():   # 路线主C 刚到手也要挡（other_c 自己判"定了没有"）
            # **主C 定了：不买"第二个主C"**（路线外、评分 ≥7 的输出型；已有同名的合成照买）——空位留给功能性英雄
            owned0 = set(self.rules._field) | {sl.get("name") for sl in state.hand_slots or []}
            drop = []
            for k in p["question"]["criteria"]:
                if not k.startswith("buy"):
                    continue
                pc = next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or ""
                if pc and pc not in owned0 and (self.rules.second_carry(pc) or self.rules.other_c(pc)):
                    drop.append(k)
            if drop and len(drop) < len(p["question"]["criteria"]):
                if _md:
                    # 2026-10-11：R23 题面（f9li）自带 C-POSITION 事实、按 g9u 标签学过 → 模型接管时只记录不删（这条的判定比"纯第二个 C"宽，会删掉喂主C 的人）
                    logger.info("（仅记录）旧规则会去掉路线外的输出型 %s", drop)
                else:
                    self._restrict(p, drop)
                    logger.info("主C 已定 → 商店题去掉路线外的输出型 %s", drop)
        # **已觉醒的辅助不再买同名**（2026-10-08：辅助觉醒后再买同名升级没什么作用，还拖慢棋手升级）
        drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                and self.rules.awakened_support(next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "")]
        if drop:
            # 2026-10-14 确认：觉醒后买同名英雄仍可升级，升级级数与之前相同 → 不是"白买"，是值不值的判断，交给学生（只记录）
            logger.info("（仅记录）旧规则会去掉已觉醒辅助的同名 %s", drop)
        if _IR.on("roll_engine_unreachable") and (state.round_no or 0) < 12 and "refresh" in p["question"]["criteria"]                 and not (state.free_refresh or 0):
            # 临时规则 roll_engine_unreachable（2026-10-21 第 883 局：凑核心阵容要结合缺的英雄的品阶；3 级商店里一直刷新，
            # 但缺的都是 4 阶英雄）：缺的引擎核心件当前等级全都刷不出 → 去掉花钱刷新（免费刷新照用），钱留着升级
            _un = self.rules.engine_missing_unreachable(state)
            if _un and len(p["question"]["criteria"]) > 1:
                self._restrict(p, ["refresh"])
                logger.info("缺的引擎件 %s 在 %s 级商店刷不出 → 去掉花钱刷新（临时规则 roll_engine_unreachable）", _un, state.level)
        if _IR.on("eco_level_budget"):
            # 临时规则 eco_level_budget（2026-10-21 eco876 分析：进度点前一回合花超"花完下回合还能升级"的额度 58/89 题 → 下回合升不起）：
            # 下回合就是进度点、攒着升得起 → 这道商店题去掉单次花费超过额度的买 / 刷（主C 同名、现在真缺的核心件除外；停手 / 锁店不动）
            _sp = self.rules.level_spare(state)
            if _sp is not None:
                _exempt = ({self.rules.route_carry(), self.rules._carry_piece} | self.rules.route_core_missing_now(state)) - {"", None}
                # 10-21 r9zzh 老师 batch_033：没觉醒的超核心件（雅典娜 / 明世隐）定为"见到就买到觉醒"，额度不该把它删掉
                _exempt |= {n for n in self.rules.super_core() if not self.rules.is_awakened(n)}
                from agent.decision import lord_config as _LCe
                _ce = _LCe.load()
                if _ce.present:                 # 10-21 审查 7：同组主C、（没有名单内主C 时）名单内主C 也豁免
                    _exempt |= set(_ce.group_of(self.rules.route_carry() or "")[1])
                    if not any(_ce.enabled(n) and _ce.tier(n) >= 2 for n in self.rules._field):
                        _exempt |= {sl.piece for sl in state.shop if sl.piece and _ce.enabled(sl.piece) and _ce.tier(sl.piece) >= 2}
                if _IR.on("entry_trigger_msy") and "明世隐" in self.rules._field:     # 唤醒（2e）也豁免（entry_trigger_msy）
                    _exempt |= {sl.piece for sl in state.shop if self._entry_trigger(sl.piece or "")}
                # 10-23 第 986 局 R3：场上只有 1 个人，额度把买人全删了、只剩刷新 → 前 4 回合 60→42 血。
                # 空位 ≥2（场上 + 手里英雄牌仍比人口上限少 2 个以上）→ 买英雄不受额度限制
                from agent.decision import meta as _mtc
                _cap = int(_mtc.POP_CAP_BY_LEVEL.get(state.level or 0, 0) or state.population_cap or 0) + int(self.rules.pop_bonus or 0)
                _hand_h = len({sl.get("name") for sl in state.hand_slots or []
                               if (self.rules.pieces.get(sl.get("name") or "") or {}).get("category") == "英雄牌"
                               and sl.get("name") not in self.rules._field})     # 10-23 第 1002 局：场上同名（合成）不占新位置，同名多张也只算 1 个人
                if _cap and _cap - (len(self.rules._field) + _hand_h) >= 2:
                    _exempt |= {sl.piece for sl in state.shop if sl.piece and (self.rules.pieces.get(sl.piece) or {}).get("category") == "英雄牌"}
                _rcb = state.refresh_cost if (state.refresh_cost or 0) > 0 else 1      # 10-23 第 984 局：费用读成 0 → 按 1 算（免费刷新另由 free_refresh 放行）
                _over = []
                for k in p["question"]["criteria"]:
                    if k == "refresh" and not (state.free_refresh or 0) and _rcb > _sp:
                        _over.append(k)
                    elif k.startswith("buy"):
                        sl = next((x for x in state.shop if x.index == int(k[3:])), None)
                        if sl is not None and sl.piece not in _exempt and (sl.cost or 3) > _sp:
                            _over.append(k)
                if _over and len(_over) < len(p["question"]["criteria"]):
                    logger.info("下回合是进度点、升级额度只剩 %de → 去掉超额的 %s（临时规则 eco_level_budget）", _sp, _over)
                    self._restrict(p, _over)
        if _IR.on("buy_listed_carry"):
            # 临时规则 buy_listed_carry（第 868 局：整局没有开着的主C，亚连 / 韩信在店里两次 Laya 都刷新了 → 主C 位被关掉的马可波罗顶到结束）：
            # 场上 / 手里没有棋手配置开着的主C（2 阶以上）→ 店里有、买得起 → 只留买他（执行棋手配置的主C 名单）
            from agent.decision import lord_config as _LCb
            _cb = _LCb.load()
            _own_b = set(self.rules._field) | {sl.get("name") for sl in state.hand_slots or []}
            _cq_b = int((self.rules.pieces.get(self.rules._carry_piece or "") or {}).get("quality") or 0)
            if _cb.present and _cq_b <= 1 and not any(_cb.enabled(n) and _cb.tier(n) >= 2 for n in _own_b):     # 现任是名单外 2 阶以上 = 已认，不再买名单内主C 来换
                _rc_b = self.rules.route_carry() if self.rules.route_locked() else ""
                _grp_b = ({_rc_b} | set(_cb.group_of(_rc_b)[1])) if _rc_b else set()        # 10-21 审查 2：路线锁了只认路线主C / 同组
                _keep_b = [k for k in p["question"]["criteria"] if k.startswith("buy")
                           and (lambda sl: sl is not None and not sl.sold and _cb.enabled(sl.piece) and _cb.tier(sl.piece) >= 2
                                and (not _grp_b or sl.piece in _grp_b)
                                and (state.gold or 0) >= (sl.cost or 3))(next((sl for sl in state.shop if sl.index == int(k[3:])), None))]
                if _keep_b:
                    _drop_b = [k for k in p["question"]["criteria"] if k not in _keep_b]
                    if _drop_b:
                        logger.info("没有开着的主C，店里有 %s → 买（临时规则 buy_listed_carry）",
                                    [next(sl.piece for sl in state.shop if sl.index == int(k[3:])) for k in _keep_b])
                        self._restrict(p, _drop_b)
            # 1058 局 R6 / R7：现任主C 狄仁杰在开局组外（no_lock_outside_group 挡着不锁路线、等组主C），店里来了沈梦溪 Laya 两次都刷新 →
            # "等组主C"的规则没有接上"组主C 到店就买"。现任在组外名单里、还没锁路线 → 店里的开局组员买得起就只留买他
            _nlo_b = set(self.rules.no_lock_outside_group(state) or ())
            if _cb.present and self.rules._carry_piece in _nlo_b and not self.rules.route_locked():
                _oc_b = _cb.opening_choice(state.banned_faction or "") or {}
                # 1061 局：只排除了已有的人 → 一回合把玄策 / 沈梦溪 / 守约三个组员全买了。手里 / 场上已有任一组员就不再触发
                _got_b = _own_b | {n for n, rd in (self.__dict__.get("_laya_bought") or {}).items() if rd == state.round_no}   # 刚买、手牌还没认出来
                _mem_b = set() if (set(_oc_b.get("members") or ()) & _got_b) else set(_oc_b.get("members") or ())
                _keep_g = [k for k in p["question"]["criteria"] if k.startswith("buy")
                           and (lambda sl: sl is not None and not sl.sold and sl.piece in _mem_b
                                and (state.gold or 0) >= (sl.cost or 3))(next((sl for sl in state.shop if sl.index == int(k[3:])), None))]
                _drop_g = [k for k in p["question"]["criteria"] if k not in _keep_g]
                if _keep_g and _drop_g:
                    logger.info("现任主C %s 在开局组外，店里有组员 %s → 买（临时规则 buy_listed_carry）", self.rules._carry_piece,
                                [next(sl.piece for sl in state.shop if sl.index == int(k[3:])) for k in _keep_g])
                    self._restrict(p, _drop_g)
        if _IR.on("stack_copies_buy") and self.rules.route_locked():
            # 临时规则 stack_copies_buy（2026-10-21：明世隐在核心阵容里时，有多少买多少）：
            # 路线成员里登场效果随同名叠加的人（STACK_COPIES：明世隐）在店里、买得起 → 这道题只留买他（还有别的选项时）
            _rm = set(self.rules.route_core())          # 10-21 审查：只认核心阵容里的明世隐（同上）
            _also = ({self.rules.route_carry(), self.rules._carry_piece} | self.rules.route_core_missing_now(state)) - {"", None}
            _keep = [k for k in p["question"]["criteria"] if k.startswith("buy")
                     and (next((sl for sl in state.shop if sl.index == int(k[3:])), None) is not None)
                     and (lambda sl: sl.piece in self.STACK_COPIES and sl.piece in _rm and not sl.sold
                          and not self.rules.is_awakened(sl.piece)                       # 只凑到觉醒（10-21）
                          and (state.gold or 0) >= (sl.cost or 3))(next(sl for sl in state.shop if sl.index == int(k[3:])))]
            if _keep:
                # 主C 同名 / 现在缺的核心件也留着（这一帧买了它，下一帧照样只留明世隐）
                _keep += [k for k in p["question"]["criteria"] if k.startswith("buy") and k not in _keep
                          and (lambda sl: sl is not None and sl.piece in _also and not sl.sold)(
                              next((sl for sl in state.shop if sl.index == int(k[3:])), None))]
                drop = [k for k in p["question"]["criteria"] if k not in _keep]
                if drop:
                    logger.info("核心阵容里的明世隐在店里 → 买（临时规则 stack_copies_buy）")
                    self._restrict(p, drop)
        if _IR.on("no_offlist_carry_buy"):
            # 临时规则 no_offlist_carry_buy（2026-10-22 第 902 局 R3：马可波罗既不是主C 也不是任何阵容的配件，没必要优先买）：
            # 名单外（carry_pool 没开）的 2 阶以上主C，不在锁定路线阵容里（没锁 = 都不在）→ 商店题不买（f9za 题面还把他写成 CORE of route）
            from agent.decision import lord_config as _LCx
            _cx = _LCx.load()
            if _cx.present:
                _team = set(self.rules._route_team()) if self.rules.route_locked() else set()
                _cur = self.rules._carry_piece or ""
                _dx = [k for k in p["question"]["criteria"] if k.startswith("buy")
                       and (lambda sl: sl is not None and sl.piece and not _cx.enabled(sl.piece) and sl.piece not in _team
                            and sl.piece != _cur and self.rules._carry_eligible(sl.piece)
                            and int((self.rules.pieces.get(sl.piece) or {}).get("quality") or 0) >= 2)(
                           next((sl for sl in state.shop if sl.index == int(k[3:])), None))]
                if _dx and len(_dx) < len(p["question"]["criteria"]):
                    logger.info("名单外主C %s 不在路线阵容里 → 不买（临时规则 no_offlist_carry_buy）",
                                [next(sl.piece for sl in state.shop if sl.index == int(k[3:])) for k in _dx])
                    self._restrict(p, _dx)
        if not self.rules.route_locked() and len(state.hand_slots or []) < self.rules.HAND_LIMIT:
            # 10-22 第 906 局：已选绑主C 的天赋（幸运硬币 → 狄仁杰）、路线还没锁 → 店里有这个主C 就买（他才是这把的路线）
            from agent.decision import talent_conditions as _tcb
            _want_c = {(_tcb.table().get(t) or {}).get("commit_carry_buy") for t in self.rules._talents_taken} - {None, ""}
            _kc = [k for k in p["question"]["criteria"] if k.startswith("buy")
                   and (lambda sl: sl is not None and not sl.sold and sl.piece in _want_c and (state.gold or 0) >= (sl.cost or 3))(
                       next((sl for sl in state.shop if sl.index == int(k[3:])), None))]
            if _kc and len(_kc) < len(p["question"]["criteria"]):
                logger.info("已选绑主C 的天赋 → 店里的 %s 买下（路线等他）", sorted(_want_c))
                self._restrict(p, [k for k in p["question"]["criteria"] if k not in _kc])
        if _IR.on("carry_copy_buy") and self.rules.route_locked() and len(state.hand_slots or []) < self.rules.HAND_LIMIT:
            # 临时规则 carry_copy_buy（2026-10-21 第 887 局 R9：刷新出了守约却没买，之后一直刷新、目标不明）：
            # 路线主C 本人在场、店里有他的同名、买得起 → 只留买他（缺的核心件也留着）。Laya 0.34 vs 刷新 0.35，题面没写主C 同名合成 = 主C 涨等级
            _rcc = self.rules.route_carry() or ""
            _alsoc = self.rules.route_core_missing_now(state) - {"", None}
            _keep = [k for k in p["question"]["criteria"] if k.startswith("buy") and _rcc and _rcc in self.rules._field
                     and (lambda sl: sl is not None and not sl.sold and sl.piece == _rcc and (state.gold or 0) >= (sl.cost or 3))(
                         next((sl for sl in state.shop if sl.index == int(k[3:])), None))]
            if _keep:
                _keep += [k for k in p["question"]["criteria"] if k.startswith("buy") and k not in _keep
                          and (lambda sl: sl is not None and sl.piece in _alsoc and not sl.sold)(
                              next((sl for sl in state.shop if sl.index == int(k[3:])), None))]
                drop = [k for k in p["question"]["criteria"] if k not in _keep]
                if drop:
                    logger.info("路线主C %s 的同名在店里 → 买（临时规则 carry_copy_buy）", _rcc)
                    self._restrict(p, drop)
        if _IR.on("entry_trigger_msy") and "明世隐" in self.rules._field and len(state.hand_slots or []) < self.rules.HAND_LIMIT:
            # 临时规则 entry_trigger_msy（2026-10-21 第 881 局：仍然不买唤醒；第 879 局起第 3 次）：场上有明世隐 →
            # 店里"指定英雄触发登场"的牌（唤醒 / 集体唤醒）买得起就只留买它（主C 同名 / 缺的核心件也留着）。
            # 原因是题面 f9za 的收益行（守约 ≈ +16 lv）超出选项预算被丢掉，模型看不到 → f9zb 写短后重训撤
            _held = any(self._entry_trigger(sl.get("name") or "") for sl in state.hand_slots or [])
            _alsoe = ({self.rules.route_carry(), self.rules._carry_piece} | self.rules.route_core_missing_now(state)) - {"", None}
            _keep = [k for k in p["question"]["criteria"] if k.startswith("buy") and not _held
                     and (lambda sl: sl is not None and not sl.sold and self._entry_trigger(sl.piece or "")
                          and (state.gold or 0) >= (sl.cost or 2))(next((sl for sl in state.shop if sl.index == int(k[3:])), None))]
            if _keep:
                _keep += [k for k in p["question"]["criteria"] if k.startswith("buy") and k not in _keep
                          and (lambda sl: sl is not None and sl.piece in _alsoe and not sl.sold)(
                              next((sl for sl in state.shop if sl.index == int(k[3:])), None))]
                drop = [k for k in p["question"]["criteria"] if k not in _keep]
                if drop:
                    logger.info("场上有明世隐、店里有触发登场的牌 → 买（临时规则 entry_trigger_msy）")
                    self._restrict(p, drop)
        if (_IR.on("core_first_no_copies") and self.rules.route_locked() and (state.round_no or 0) < 12
                and self.rules.route_core_missing_now(state)):      # 第 12 回合起认命（10-21）
            # 临时规则 core_first_no_copies（2026-10-21：每个核心阵容英雄先只买一张，先凑完整核心阵容，再补觉醒）：
            # 核心还缺人时，已经有了的核心件不买第二张（明世隐除外：有多少买多少）
            _own = set(self.rules._field) | {sl.get("name") for sl in state.hand_slots or []} | set(self.rules.hand_book_names())
            _core = set(self.rules.route_core()) - {self.rules.route_carry(), self.rules._carry_piece}   # 主C 同名照买（10-21：守约出现了也没买）
            # 引擎核心件（引擎表 core_set：伞亚连的雅典娜 / 米莱狄…）同名照买、先凑到觉醒（10-21 第 879 局：雅典娜是绝对核心，R7/R11/R14 三次被这条挡掉）
            _core -= {n for n in self.rules.super_core() if not self.rules.is_awakened(n)}     # 超核心件没觉醒前不受限（10-21：只凑到觉醒）
            drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    and (lambda n: n and n in _core and n in _own)(
                        next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "")]
            if drop and len(drop) < len(p["question"]["criteria"]):
                logger.info("核心还缺 %s → 已有的核心件先不买同名 %s（临时规则 core_first_no_copies）",
                            sorted(self.rules.route_core_missing_now(state)), drop)
                self._restrict(p, drop)
        if _IR.on("core_before_level_cards") and self.rules.route_locked():
            # 临时规则 core_before_level_cards（2026-10-21：凑到至少 1 个核心英雄，优先级高于升 2 级的效果牌）：
            # 店里有一张都还没有的路线核心件 → 这道题先去掉"给英雄加等级"的效果牌（下一题照常可买）
            _miss = self.rules.route_core_missing(state)
            _shop = {int(k[3:]): (next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "")
                     for k in p["question"]["criteria"] if k.startswith("buy")}
            if any(n in _miss for n in _shop.values()):
                drop = [f"buy{i}" for i, n in _shop.items() if n and self.rules.is_level_card(n)]
                if drop and len(drop) < len(p["question"]["criteria"]):
                    logger.info("店里有缺的核心件 %s → 先不买加等级牌 %s（临时规则 core_before_level_cards）",
                                sorted(n for n in _shop.values() if n in _miss), [_shop[int(k[3:])] for k in drop])
                    self._restrict(p, drop)
        if _IR.on("single_copy"):
            drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    and self.rules.single_copy_owned(next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "", state)]
            if drop and len(drop) < len(p["question"]["criteria"]):
                logger.info("只要 1 张的人已经有了（临时规则 single_copy）：商店题去掉 %s", drop)
                self._restrict(p, drop)
        if _IR.on("no_lowest_level_cards"):
            drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    and self.rules.never_pick(next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "")]
            if drop and len(drop) < len(p["question"]["criteria"]):
                logger.info("商店题去掉给最低等级英雄的牌（临时规则 no_lowest_level_cards）：%s", drop)
                self._restrict(p, drop)
        if _IR.on("awakened_no_copy") and isinstance(state.level, int) and state.level < _meta.MAX_LORD_LEVEL:
            # 临时规则 awakened_no_copy（2026-10-20 晚：觉醒了的辅助 / 功能件（盾山、李元芳）满级前不买同名，钱先给主C 和棋手等级）
            drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    and self.rules.awakened_no_copy(next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "")]
            if drop and len(drop) < len(p["question"]["criteria"]):
                logger.info("已觉醒的非主C 同名，棋手满级前不买（临时规则 awakened_no_copy）：%s",
                            [next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") for k in drop])
                self._restrict(p, drop)
        # **主C 定了以后不买新的 C 位英雄**（2026-10-10 第 515 局：敖隐在这套阵容里也是 C 位英雄，等级低、后面上场没有用）
        drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                and self.rules.late_c_position(next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "")]
        if drop and len(drop) < len(p["question"]["criteria"]):
            if _md:
                logger.info("（仅记录）旧规则会去掉新的 C 位英雄 %s（R23 题面有 C-POSITION）", drop)
            else:
                self._restrict(p, drop)
                logger.info("主C 已定，新的 C 位英雄不买 → 商店题去掉 %s", drop)
        if ((self.rules.hold_level_for_carry(state) or self.rules.haiyue_hold_level(state)
             or (self.rules.eco_says(state) == "roll" and (self.rules._in_roll_window(state)
                                                            or self.rules.core_roll_active(state))))
                and "levelup" in p["question"]["criteria"] and len(p["question"]["criteria"]) > 1):
            self._restrict(p, ["levelup"])       # 3 级还没有主C：先找主C 再升级（2026-09-30）；局面网络说停级刷牌（ECO_TAKEOVER）
        if board_full and self.rules.lineup_complete():
            # 阵容已成型（满员、场上都是规划里的人）：新英雄买来上不了场、马上会被当闲置卖掉 → 只买场上同名（合成）和路线主C
            rc = self.rules.route_carry()
            drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    and (lambda pc: pc and self.rules.pieces.get(pc, {}).get("category") == "英雄牌"
                         and pc not in self.rules._field and pc != rc)(
                        next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "")]
            if drop and len(drop) < len(p["question"]["criteria"]):
                if _md:
                    logger.info("（仅记录）旧规则会因阵容已成型去掉新英雄 %s（题面有 lineup 行）", drop)
                else:
                    self._restrict(p, drop)
                    logger.info("阵容已成型 → 商店题去掉新英雄 %s", drop)
        # **不为临时主C 买同名凑觉醒**（2026-10-13：临时主C 不值得多买卡觉醒；干将买了 4 张纯属浪费，
        # 不如多买中期最终 C 位的配套卡）：现任主C 是 1 阶（含觉醒）、不是路线主C → 商店题去掉买他的同名
        _cc = self.rules._carry_piece
        if (_IR.on("temp_carry_no_copies") and _cc and self.rules.is_temp_carry(_cc)):     # 1 阶（含觉醒）、不是路线主C（10-13 与题面同口径）
            drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    and (next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "") == _cc]
            if drop and len(drop) < len(p["question"]["criteria"]):
                self._restrict(p, drop)
                logger.info("临时主C %s 不买同名凑觉醒 → 商店题去掉 %s", _cc, drop)
        # **多路线核心卡只买一张**（2026-10-13 定的理想打法第 3 条）：已经有一张的不再买，现任主C（不是临时的）照常合成
        if _IR.on("flex_buy") and not self.rules.final_carry_settled():   # 主C 定了就不限张数（2026-10-13）
            _core = self.rules.flex_core_cards(state)
            _own = set(self.rules._field) | {sl.get("name") for sl in state.hand_slots or []}
            drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                    and (lambda n: n in _core and n in _own and n != _cc)(
                        next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "")]
            if drop and len(drop) < len(p["question"]["criteria"]):
                self._restrict(p, drop)
                logger.info("多路线核心卡只买一张 → 商店题去掉 %s", drop)
        # **刚卖掉的英雄不买回来**（2026-09-30：先卖掉一个英雄、开店又把他买回来，纯属浪费）
        drop = [k for k in p["question"]["criteria"] if k.startswith("buy")
                and self.rules.recently_sold(next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or "")]
        if drop:
            # 2026-10-14 硬规则清查：价值判断（12 局 46 次），交给学生；题面待补"2 回合内刚卖掉"事实（只记录）
            logger.info("（仅记录）旧规则会去掉刚卖掉的英雄 %s", drop)
        if not _md and self.rules.saving_for_level(state):
            # 也不买路线外的新人（2026-09-30 第 24 局：20/22 只差 2，学生花 9 买了 3 张，6 级晚了两个回合）；
            # 路线要的人 / 路线要的牌 / 已有同名（合成）照买
            keep = self.rules.route_stage_members(state.round_no) | self.rules.route_wanted_cards()
            owned = set(self.rules._field) | {sl.get("name") for sl in state.hand_slots or []}
            drop = []
            for k in p["question"]["criteria"]:
                if not k.startswith("buy"):
                    continue
                pc = next((sl.piece for sl in state.shop if sl.index == int(k[3:])), "") or ""
                if pc and pc not in keep and pc not in owned:
                    drop.append(k)
            if drop and len(drop) < len(p["question"]["criteria"]):
                self._restrict(p, drop)
                logger.info("攒钱升级 → 商店题去掉路线外的新人 %s", drop)
        # **经济网络说本回合刷（roll）→ 刷得起就不停手**（2026-10-11 第 564 局：eco 第 6/7 回合 roll 0.68/0.51，Laya 刷 1 次就 stop，
        # 14 能量囤到下回合；第 8 回合又被 eco 拿去升级，明世隐 / 钟馗买不起）。这是执行经济网络的决定：买什么、刷几次仍由 Laya 选；
        # 付费刷新每回合最多 ECO_ROLL_MAX 次（安全阀），刷完剩不到一张英雄的钱就允许停
        if _md and self.rules.eco_says(state) == "roll" and _IR.on("eco_roll_no_stop"):   # 临时规则（已关，见 interim_rules）
            _cost = state.refresh_cost if (state.refresh_cost or 0) > 0 else 1
            _n = (getattr(self, "_paid_refresh", None) or {}).get(state.round_no, 0)
            _can = (state.free_refresh or 0) > 0 or (isinstance(state.gold, int) and state.gold - _cost >= 3)
            _stops = [k for k in ("stop", "lock") if k in p["question"]["criteria"]]
            if _can and _n < self.ECO_ROLL_MAX and "refresh" in p["question"]["criteria"] and _stops                     and len(p["question"]["criteria"]) > len(_stops):
                self._restrict(p, _stops)
                logger.info("经济网络说本回合刷牌（能量 %s）→ 商店题去掉 %s", state.gold, _stops)
        reserve = self.rules.level_budget(state)
        if reserve > 0 and isinstance(state.gold, int):
            # **升级预算**：买 / 付费刷新之后剩的钱不能低于升级要留的数（按局面算，见 rules.level_budget）。
            # 路线主C 的同名照买（主C 优先于升级）
            spare = state.gold - reserve
            rc = self.rules.route_carry() if self.rules.route_locked() else ""
            if _md:
                rc = self.rules._carry_piece or ""     # 审查 10-12（A6）：豁免现任主C 的同名，不豁免路线写死的主C（第 587 局只剩"买守约 / 停手"）
            drop = []
            for k in p["question"]["criteria"]:
                if k.startswith("buy"):
                    sl = next((x for x in state.shop if x.index == int(k[3:])), None)
                    if sl is None or sl.piece == rc:
                        continue
                    _me = self.rules.min_engine()
                    if (sl.piece in self.rules.route_core() or sl.piece in ((_me.get("early_priority_full") or _me.get("early_priority")) or [])
                            or sl.piece in set(self.rules._route_team()) - set(_me.get("late_fill") or [])):
                        continue      # 2026-10-13 核心阵容审计：升级预算删掉核心件 10 次、配套 13 次 —— 核心件不删，买不买交给学生
                                      # 2026-10-20（f9y 第 5 条，r9zzd 3fbd0a1b588e / d2e80b16c226）：最终阵容件（明世隐）也不删，后期件除外
                    _cp = self.rules._carry_piece or ""
                    if (_cp and re.search(r"装备", self.rules.pieces.get(sl.piece, {}).get("skill") or "")
                            and self.rules.pieces.get(sl.piece, {}).get("category") == "效果牌"
                            and len(self.rules._worn.get(_cp) or ()) < 3):
                        continue      # 专属装备箱（远洋宝藏）主C 没满 3 件不删（10-17：亚连还差一个装备时，商店出了日落海宝藏可以买；第 714 局 R9 被预算删掉）
                    price, _e = self.rules.est_price(sl.piece, sl.cost)
                    if price > spare:
                        drop.append(k)
                elif k == "refresh" and not state.free_refresh and spare < (state.refresh_cost if (state.refresh_cost or 0) > 0 else 1):   # 刷新实际 1 能量（10-19）
                    drop.append(k)
            if drop and len(drop) < len(p["question"]["criteria"]):
                self._restrict(p, drop)
                logger.info("升级预算：能量 %s 要留 %s 给升级 → 商店题去掉 %s", state.gold, reserve, drop)
        choice, probs, conf, ms = self._ask(p)
        rule_key = p.get("rule_choice")
        self._rule_shop_act = a if (a or {}).get("type") == "buy_piece" else None    # eco_roll_follow 改刷新前先买规则想买的（第 779 局 R9 明世隐被刷掉）
        # **不退回规则**（2026-09-25：Laya 控制的题不再退回规则；实例：第四轮第 1 局第 13 回合学生要买孙膑，
        # 规则以"买回来没位置"拦下、换成它自己的"买鲁班大师"）。按学生概率从高到低，执行第一个**能做**的动作；
        # 只有全部选项都做不了（违规）才交给规则。规则的价值判断不再否决学生。
        skipped = []
        for k in sorted(probs, key=lambda x: -probs[x]):
            if k == rule_key:
                act = a
            else:
                act, why = self._shop_action(state, k)
                if act is None:
                    skipped.append(f"{k}:{why}")
                    continue
                if act == "HOLD":
                    self._log(p, choice, probs, conf, ms, rule_key, "laya" if k == choice else "laya_next",
                              "不买了（店关不掉，本帧不动）")
                    return None
            used = "laya" if k == choice else "laya_next"
            if k == rule_key:
                used = "laya(=rule)" if k == choice else "laya_next(=rule)"
            self._log(p, choice, probs, conf, ms, rule_key, used,
                      ("首选做不了 → " + "；".join(skipped)) if skipped else "")
            return act
        self._log(p, choice, probs, conf, ms, rule_key, "rule_fallback", "所有选项都做不了：" + "；".join(skipped))
        if self.rules.model_decides() and a is not None and a.get("type") in ("buy_piece", "refresh_shop", "lock_shop"):
            return None
        return a

    ECO_ROLL_MAX = 8

    ECO_ROLL_MAX = 6                # eco_roll_follow：每回合最多替商店多刷几次（经济网络题面 "up to N" 一般 4~9）

    def _shop_action(self, state, choice: str):
        """Laya 的商店选择 -> executor 动作；硬约束不过就返回 (None, 原因)。"""
        r = self.rules
        if choice.startswith("buy"):
            slot = next((sl for sl in state.shop if sl.index == int(choice[3:])), None)
            if slot is None or not slot.piece or slot.sold:
                return None, "槽位已空"
            cost, est = slot.cost, slot.cost is None
            # 只查**违规**（不退回规则）：钱不够。"买回来有没有位置、值不值"交给学生。
            price, _e = r.est_price(slot.piece, cost)          # 价格读不到也要查（用卡牌库标价估）
            if (state.gold if state.gold is not None else 0) >= 0 and (state.gold or 0) < price:
                return None, "能量不够" + ("（价格按卡库估）" if cost is None else "")
            # 手牌满了也能买（2026-10-14 确认：能买，超出的部分下个回合自动销毁）—— 不再拦，值不值交给学生
            # **放不下**（物理约束，不是价值判断）：场上满员、手里已经压着 ≥2 个等着上场的英雄，
            # 再买一个**新**英雄就只是占手牌。同名（场上/手里已有 → 合成、推进觉醒）照常放行。
            # 2026-09-25 第四轮第 2 局：第 9 回合手里压了 7 个英雄（人口 6/6、手牌 8/10），学生仍在买新人。
            if r.pieces.get(slot.piece, {}).get("category") == "英雄牌":
                hand_heroes = [sl["name"] for sl in state.hand_slots or []
                               if r.pieces.get(sl.get("name"), {}).get("category") == "英雄牌"]
                waiting = [n for n in hand_heroes if n not in r._field]
                # 满员判定与题面 "free slots" **同一口径**：人口读数（商店开着时是上次读到的值）；读不到才用账本人数。
                # 原来用账本人数 —— 第四轮第 3 局账本 7 人、人口读数 6/7，题面写"1 free slot"，学生选的 8/9 主C 安琪拉被判"放不下"。
                cap = state.population_cap or 0
                here = r.pop_now(state)[0]
                full = cap > 0 and here >= cap
                # 路线成员、场上有路线外补位件可换 → 放得下（和题面 f9ln "ROUTE unit: can replace filler" 同口径；
                # 2026-10-13 核心阵容审计：这条拦下了学生选的核心 / 配套件 18 次，如第 595 局 R6 学生 0.57 选铠）
                _team = r._route_team() | set(r.route_members()) if r.route_locked() else set()
                _filler = [n for n in r._field if n not in _team and n != r._carry_piece]
                if (_IR.on("shop_no_room") and full and len(waiting) >= 2 and slot.piece not in r._field and slot.piece not in hand_heroes
                        and not (slot.piece in _team and _filler)):
                    return None, f"放不下：场上满员、手里已有 {len(waiting)} 个英雄在等"
            if (r.pieces.get(slot.piece) or {}).get("category") == "英雄牌" and slot.piece not in r._field and slot.piece not in hand_heroes:
                # 记下"Laya 主动买的新英雄"：满员时换人题不给"不换"（bought_must_deploy，第 853 局 R6 买蒙恬又不上场）
                self.__dict__.setdefault("_laya_bought", {})[slot.piece] = state.round_no
            return {"type": "buy_piece", "slot": slot.index, "piece": slot.piece,
                    "cost": cost if cost is not None else 3, "cost_est": est, "round": state.round_no}, ""
        if choice == "refresh":
            free = (state.free_refresh or 0) > 0
            if not free:
                rc = state.refresh_cost if (state.refresh_cost or -1) >= 0 else 1
                if state.gold < rc:
                    return None, "能量不够刷新"
                if _IR.on("shop_refresh_floor") and state.gold - rc < 3:
                    # **刷完买不起英雄就别刷**（2026-10-09：3 块钱以内不刷新商店，刷出来也买不起）。
                    # 这是算术约束（刷新费 + 英雄 3 能量），不是价值判断；规则自己的刷新分支早就这么查了，学生这条漏了
                    return None, f"刷完只剩 {state.gold - rc} 能量，买不起英雄"
                # 付费刷新上限、等级落后闸已去掉（2026-09-25）
                self._paid_refresh[state.round_no] = self._paid_refresh.get(state.round_no, 0) + 1
            return {"type": "refresh_shop"}, ""
        if choice == "levelup":
            if not state.level_btn:
                return None, "画面上没找到升级按钮"
            if state.level_cost and state.level_cost > 0 and state.gold < state.level_cost:
                return None, "能量不够升级"
            return {"type": "buy_xp", "x": state.level_btn[0], "y": state.level_btn[1]}, ""
        if choice == "lock":
            if r._locked_round == state.round_no:
                return None, "本回合已锁过店"
            if _IR.on("lock_needs_wanted"):
                # 临时规则 lock_needs_wanted（2026-10-19 第 788 局：几次锁店都没有道理——R7 锁 芈月/刘禅/蒙恬/苏烈、
                # R8 锁 西施/苏烈/钟馗/换帅，没有一张要的牌，白丢下回合新商店）：店里没有主C 同名 / 路线成员 / 核心引擎件 → 不锁（交回停手）
                _want = ({r._carry_piece} | set(r._route_team()) | set((r.min_engine().get("early_priority_full") or r.min_engine().get("early_priority")) or [])) - {"", None}
                if not any(sl.piece in _want for sl in state.shop if sl.piece and not sl.sold):
                    logger.info("Laya 要锁店但店里没有要的牌 %s → 不锁（临时规则 lock_needs_wanted）",
                                [sl.piece for sl in state.shop if sl.piece and not sl.sold])
                    return None, "店里没有要的牌，不锁（lock_needs_wanted）"
            r._locked_round = state.round_no
            r._stop_shop_round = state.round_no
            return {"type": "lock_shop", "pieces": [sl.piece for sl in state.shop if sl.piece and not sl.sold]}, ""
        from agent.decision import meta as _meta
        _behind = bool(state.level) and state.level < _meta.level_target(state.round_no or 0)
        _gw = r.group_wait_missing(state)          # 10-24：路线没锁、等开局组主C 时也跟（缺的是组员）
        if (choice == "stop" and _IR.on("eco_roll_follow") and r.eco_says(state) == "roll" and (r.route_locked() or _gw) and not _behind
                and not (_IR.on("roll_engine_unreachable") and r.engine_missing_unreachable(state))
                and (state.round_no or 0) < 12):        # 第 12 回合起不替商店追刷缺件（10-21：后期刷不到就认命，主要凑主C 等级）
            # （2026-10-19 第 755 局：3 级卡到第 11 回合、进度线第 7 回合 4 级 / 第 10 回合 5 级，这条在后面加码刷光了升级钱 → 落后进度时不触发）
            # 临时规则 eco_roll_follow（2026-10-18：路线定好后不换，要在对应棋手等级多刷几次；第 735 局 R9：
            # 经济网络判"停级刷牌"、缺明世隐，商店 Laya 免费刷完就停、留 10e，明世隐拖到 R11）：执行经济网络已经做出的"刷牌"决定 ——
            # 路线核心还缺（不算太乙这类后期件）、刷完还买得起一个人、本回合付费刷新 < 上限 → 继续刷
            _lf = r.late_pieces()
            _miss = {x for x in r.route_core_missing(state) if x not in _lf} | set(_gw)
            # 10-20（review-interim-conflicts 第 9 条）：当前棋手等级根本刷不出的件不算（缺 4 阶雅典娜、3 级时连刷 6 次 = 白刷）
            from agent.decision import meta as _mt
            _miss = {x for x in _miss if _mt.can_appear(int((r.pieces.get(x) or {}).get("quality") or 0), state.level or 1)}
            _rc = 0 if (state.free_refresh or 0) > 0 else (state.refresh_cost if (state.refresh_cost or 0) > 0 else 1)   # 10-23 第 984 局：费用读成 0 也按 1 算
            _done = self._paid_refresh.get(state.round_no, 0)
            _in_shop = {sl.piece for sl in state.shop if sl.piece and not sl.sold}
            _ra = self.__dict__.get("_rule_shop_act")
            # 只先买主C 同名 / 路线成员 / 核心引擎件（第 785 局 R11 把用不上的效果牌逍遥自在也买了 —— 场上 0 个无阵营英雄）
            _keep_names = ({r._carry_piece} | set(r._route_team()) | set((r.min_engine().get("early_priority_full") or r.min_engine().get("early_priority")) or [])) - {""}
            if _ra and _ra.get("piece") not in _keep_names:
                _ra = None
            if _miss and _ra and _ra.get("piece") in _in_shop and (state.gold or 0) >= (_ra.get("cost") or 3):
                # 2026-10-19 第 779 局 R9（中间刷出过明世隐却没买）：Laya 停手 → 这条规则改成刷新，把规则要买的明世隐刷掉了。
                # 改刷新之前先买规则这一帧要买的那张（下一帧再按原条件继续刷）
                logger.info("改刷新前先买规则要买的 %s（临时规则 eco_roll_follow）", _ra.get("piece"))
                return dict(_ra), ""
            _sp = r.level_spare(state) if _IR.on("eco_level_budget") else None
            if _sp is not None and _rc > _sp:
                logger.info("经济网络判刷，但下回合是进度点、再刷 %de 会超出升级额度 %de → 不替商店继续刷（临时规则 eco_level_budget）", _rc, _sp)
                _miss = set()
            if _miss and not (_miss & _in_shop) and (state.gold or 0) - _rc >= 3 and _done < self.ECO_ROLL_MAX:
                self._paid_refresh[state.round_no] = _done + 1
                logger.info("经济网络本回合判刷牌、路线还缺 %s → 商店选停手改为继续刷（第 %d 次，临时规则 eco_roll_follow）",
                            sorted(_miss), _done + 1)
                return {"type": "refresh_shop"}, ""
        if choice == "stop" and _IR.on("lock_for_carry_copy") and r._locked_round != state.round_no:
            # 临时规则 lock_for_carry_copy（2026-10-19 第 781 局 R9：没钱时商店里有狄仁杰，却没锁店；
            # Laya 选 stop 0.52、题面 lock 写着 keeps 狄仁杰 (merge)）：钱不够再买一张英雄、店里有主C 同名或缺的核心引擎件 → 锁店（不花钱）
            _rc0 = state.refresh_cost if (state.refresh_cost or 0) > 0 else 1
            _want = {self._real_carry(state)} | {n for n in ((r.min_engine().get("early_priority_full") or r.min_engine().get("early_priority")) or [])
                                        if n not in r._field and n not in {sl.get("name") for sl in state.hand_slots or []}}
            _keep = [sl.piece for sl in state.shop if sl.piece and not sl.sold and sl.piece in _want]
            if _keep and (state.gold or 0) < 3 and not state.free_refresh:
                r._locked_round = state.round_no
                r._stop_shop_round = state.round_no
                logger.info("没钱了、店里有 %s → 停手改锁店（临时规则 lock_for_carry_copy）", _keep)
                return {"type": "lock_shop", "pieces": [sl.piece for sl in state.shop if sl.piece and not sl.sold]}, ""
        if choice == "stop":
            if (self.rules.level_budget(state) > 0 and not state.level_btn and state.level_cost
                    and 0 < state.level_cost <= (state.gold or 0)):
                k = ("wait_lvbtn", state.round_no)
                n = self.__dict__.setdefault("_wait_lvbtn", {}).get(k, 0)
                if n < 6:
                    self._wait_lvbtn[k] = n + 1
                    return "HOLD", ""        # 升级按钮这帧没读到（约 4 成商店帧读不到）：别关店，等它出现再升（审查 10-09）
            # 学生说"不买了" → **关店**，并标记本回合不再开店（规则别再把店打开，否则来回开关）。
            # 2026-09-25 第四轮第 5 局：原来"本回合已关过 2 次店"就改成"本帧不动"(HOLD)，结果店一直开着、盖住棋盘，
            # 手里 3 个英雄整回合没上场。关店次数上限放宽到 4，防真关不掉时无限点。
            if r._shop_closes >= 4:
                self.rules._stop_shop_round = state.round_no
                return None, "本回合已关过 4 次店（关不掉）"
            self.rules._stop_shop_round = state.round_no     # 本回合不买了：规则别再把店打开（否则来回开关）
            return {"type": "toggle_shop", "intent": "close"}, ""
        return None, f"未知选项 {choice}"
