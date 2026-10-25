"""Laya 决策引擎：Router 封装 + 置信度门控 + 规则兜底。

注意：Laya 基础模型在决策任务上零样本能力有限（官方实测接近随机），
本模块的定位是"辅助意见 + 数据采集"，经微调后再逐步替代规则引擎。
默认启用置信度门控：低于阈值一律回退规则引擎。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from ..perception.state import GameState
from .questions import build_lord_boost_question, build_questions

logger = logging.getLogger(__name__)

# 模型权重缓存默认放项目目录（C 盘空间紧张；单个 checkpoint 的 safetensors 约 1.7G）。
# 外部已显式设置 HF_HOME 时不覆盖。
HF_CACHE_DIR = Path(__file__).resolve().parents[2] / ".hf_cache"

# laya.Router 的模型仓库：BUNDLE_REPO="convaiinnovations/laya"，每个 checkpoint 是
# 一个子目录（multilingual / typed-decisions），english 在仓库根。
# HF 缓存布局：<HF_HOME>/hub/models--convaiinnovations--laya/snapshots/<rev>/<子目录>/
HF_REPO_DIRNAME = "models--convaiinnovations--laya"


def weights_root(model: str = "multilingual") -> Optional[Path]:
    """返回**已经存着**这个 checkpoint 权重的 HF 缓存根目录，没有就 None。

    两件事靠它：
    1. 新机器上 `laya.Router().preload()` 找不到权重会**直接开始下载**（约 1.7G）。
       那发生在主循环里 = agent 瞎好几分钟，而 Laya 当前是 advisory_only
       （只记录不产出动作），下载纯属白等。setup_env.bat 第 5 步是**可选**的，
       所以"装了 laya 但没下权重"是常态，必须先查。
    2. 别把 HF_HOME 无脑指向项目 .hf_cache：用户级默认缓存
       (~/.cache/huggingface) 里可能**已经有**这份权重，硬改 HF_HOME 会让
       huggingface_hub 认为缓存是空的，**再下一份**（2026-09-22 实测又下了 647M）。
       查到在哪就用哪，不搬家。
    """
    roots: list[Path] = []
    env_home = os.environ.get("HF_HOME")
    if env_home:
        roots.append(Path(env_home))
    roots.append(HF_CACHE_DIR)
    roots.append(Path.home() / ".cache" / "huggingface")
    for root in roots:
        snap_dir = root / "hub" / HF_REPO_DIRNAME / "snapshots"
        if not snap_dir.is_dir():
            continue
        for snap in snap_dir.iterdir():
            sub = snap / model if model != "english" else snap
            if any(sub.glob("*.safetensors")) or any(sub.glob("*.bin")):
                return root
    return None


def weights_cached(model: str = "multilingual") -> bool:
    return weights_root(model) is not None


class LayaEngine:
    """Laya 决策引擎：预置词表 + 每回合一次批量预测（结果按回合缓存）。

    用法要点（对局倒计时内必须出结果）：
    - 词表预定义：每个决策点的选项键是固定词表（买0/买1/.../不买），运行时只填当前数值
    - 批量：一次 predict 带全部问题，每回合只问一次，结果缓存到下一回合
    - 置信度门控：低于阈值回退规则引擎

    2026-09-21 实测（3090）：
    - 延迟不是问题：全套 4 题批量 47.6ms（每题 11.9ms），单题 57ms。可以同步放主循环。
    - 质量是问题：buy_which（5 选 1，随机基线 0.200）三个局面平均只有 0.112，**低于随机**；
      而 should_refresh / should_level_up 在三个差异极大的局面上恒定给 ~0.99，
      是"无脑说 yes"而不是高置信判断——却正好能穿过 0.85 门槛去覆盖规则引擎的购买决策。
    因此默认 advisory_only=True：照常预测并记录（攒微调数据），但**不产出动作**。
    微调之后再把 advisory_only 关掉。
    """

    # 实测 typed-decisions 在中文 state 上全面更差（buy_which 平均 0.034），
    # 且自动路由在 OCR 大面积失败时会回退到读不懂中文的 english checkpoint，故写死。
    MODEL = "multilingual"

    def __init__(self, confidence_threshold: float = 0.85, enabled: bool = True,
                 advisory_only: bool = True, allow_download: bool = False):
        self.threshold = confidence_threshold
        self.enabled = enabled
        if enabled and not allow_download and not weights_cached(self.MODEL):
            logger.warning("本地没有 Laya 权重（%s），本次禁用 Laya 以免主循环里现下 1.7G。"
                           "要用的话先跑 setup_env.bat 第 5 步，或加 --laya-download",
                           self.MODEL)
            self.enabled = False
        self.advisory_only = advisory_only
        self._router = None
        self._cache_round: int = -1
        self._cache: dict = {"decisions": [], "raw": {}}

    def _get_router(self):
        if self._router is None:
            # 权重已经在某个缓存里就用那个，别搬家（搬了等于重下，见 weights_root）
            root = weights_root(self.MODEL)
            if root is not None:
                os.environ["HF_HOME"] = str(root)
            else:
                os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
            try:
                import laya
            except ImportError:
                logger.warning("未安装 Laya（pip install laya），回退纯规则决策。"
                               "接入方式见文档：本地安装 Laya 后自动启用")
                self.enabled = False
                return None
            # 只预热真正要用的那个 checkpoint：preload=True 会连带加载另外两个
            # （每个 safetensors 约 1.7G，english 本地没下，会触发下载）。
            # 冷启动实测 26~63 秒，进程起来时先付掉，别拖到对局里。
            self._router = laya.Router()      # device=None：自动选 cuda
            self._router.preload([self.MODEL])
            logger.info("Laya Router 已加载（checkpoint=%s）", self.MODEL)
        return self._router

    def predict(self, state: GameState) -> dict[str, Any]:
        """返回 {decisions: [...], raw: {...}}。低置信度决策不会出现在 decisions 里。

        只在准备阶段且商店有牌时运行；每回合只预测一次，结果缓存复用
        （对局倒计时内必须出结果，不能每帧现想）。
        """
        if not self.enabled:
            return {"decisions": [], "raw": {}}
        if state.phase != "prepare" or not state.shop:
            return {"decisions": [], "raw": {}}
        if state.round_no == self._cache_round:
            return self._cache

        router = self._get_router()
        if router is None:
            return {"decisions": [], "raw": {}}

        questions = build_questions(state)
        result = router.predict(state.to_laya_state(), questions, model=self.MODEL)
        answers = result["answers"]

        decisions: list[dict[str, Any]] = []
        if self.advisory_only:
            # 只记录不决策：raw 答案仍写进日志，作为微调语料
            logger.info("Laya 建议（仅记录，不执行）: %s",
                        {k: (v.get("choice") or v.get("noul") or v.get("score"),
                             round(v.get("confidence", -1), 3)) for k, v in answers.items()})
            self._cache_round = state.round_no
            self._cache = {"decisions": [], "raw": answers}
            return self._cache

        buy = answers.get("buy_which")
        if buy and buy["confidence"] >= self.threshold:
            m = re.match(r"槽位(\d)_", buy["choice"])
            if m:
                slot = state.shop[int(m.group(1))]
                decisions.append({"type": "buy_piece", "slot": slot.index, "piece": slot.piece})

        refresh = answers.get("should_refresh")
        if refresh and refresh["confidence"] >= self.threshold and refresh["noul"] >= 0.5:
            decisions.append({"type": "refresh_shop"})

        level = answers.get("should_level_up")
        if level and level["confidence"] >= self.threshold and level["noul"] >= 0.5:
            decisions.append({"type": "buy_xp"})

        logger.info(
            "Laya 决策: %s (routing=%s)", decisions, result.get("routing", {})
        )
        self._cache_round = state.round_no
        self._cache = {"decisions": decisions, "raw": answers}
        return self._cache


    def pick_lord_boost(self, state: GameState, options):
        """棋手特性三选一：让 Laya 在**规则已过滤**的候选里挑一个。

        这是指定由 Laya 负责的决策点（"选哪个"属于价值判断）。
        与购买类不同，这里**不受 advisory_only 限制**：模态不选会一直挂着挡住
        后续所有操作，总得选一个；而且一局只触发约 4 次，选错的代价有限。
        低于置信度阈值仍然回退规则启发式（返回 None 即可）。
        """
        if not self.enabled or not options:
            return None
        router = self._get_router()
        if router is None:
            return None
        try:
            qs = build_lord_boost_question(state, options)
            ans = router.predict(state.to_laya_state(), qs, model=self.MODEL)["answers"]
            a = ans.get("lord_boost") or {}
            conf, choice = a.get("confidence", 0), a.get("choice")
            logger.info("Laya 特性选择: %r 置信度 %.3f（阈值 %.2f）",
                        choice, conf, self.threshold)
            if conf >= self.threshold:
                return next((o for o in options if o.name == choice), None)
        except Exception:
            logger.exception("Laya 特性选择失败，回退规则")
        return None


class HybridEngine:
    """Laya 优先、规则兜底的混合引擎。"""

    def __init__(self, rule_engine, laya_engine: Optional[LayaEngine] = None, policy=None):
        self.rules = rule_engine
        self.laya = laya_engine
        self.policy = policy          # 蒸馏后的学生（agent/decision/laya_policy.py）；有它就不用旧的零样本 Laya
        if laya_engine is not None and policy is None:
            # 把"特性选哪个"的决策权注入规则引擎（规则只负责否决结构性废牌）
            rule_engine._laya_pick = laya_engine.pick_lord_boost

    def decide(self, state: GameState) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """返回 (最终操作列表, 诊断信息)。"""
        if self.policy is not None:
            self.policy.observe(state)                 # 先更新上下文（枫叶舞强化的钩子在 decide_all 里就会用到）
            rule_actions = self.rules.decide_all(state)
            return self.policy.override(state, rule_actions), {"source": "laya_policy"}
        rule_actions = self.rules.decide_all(state)
        if self.laya is None or not self.laya.enabled:
            return rule_actions, {"source": "rules"}

        laya_result = self.laya.predict(state)
        if laya_result["decisions"]:
            logger.info("采用 Laya 购买决策（高置信度）")
            # 合并策略：Laya 只覆盖购买类（买哪张/刷新/升级），
            # 上场/升级/用牌等非购买决策保留规则结果（不重复执行购买类）
            purchase_types = {"buy_piece", "refresh_shop", "buy_xp"}
            merged = [a for a in rule_actions if a["type"] not in purchase_types]
            merged = laya_result["decisions"] + merged
            return merged, {
                "source": "hybrid",
                "laya_raw": laya_result["raw"],
                "rule_fallback": rule_actions,
            }
        return rule_actions, {"source": "rules", "laya_raw": laya_result["raw"]}
