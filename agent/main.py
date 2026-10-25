"""AgentServer 入口：注册自定义识别器与自定义动作。

由 interface.json 的 agent.child_exec 启动；通用 UI 执行到 Pipeline 中的
Custom 节点时会回调到这里注册的模块。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wzwxq")

from maa.agent.agent_server import AgentServer  # noqa: E402
from maa.context import Context  # noqa: E402
from maa.custom_action import CustomAction  # noqa: E402
from maa.custom_recognition import CustomRecognition  # noqa: E402

from agent.actions.executor import ActionExecutor  # noqa: E402
from agent.decision.laya_engine import HybridEngine, LayaEngine  # noqa: E402
from agent.decision.rules import RuleEngine  # noqa: E402
from agent.perception.extractors import StateExtractor  # noqa: E402
from agent.recorder import GameRecorder  # noqa: E402

# ---- 全局单例（AgentServer 进程生命周期内复用，避免重复加载模型） ----
extractor = StateExtractor(PROJECT_DIR / "assets" / "image")
rule_engine = RuleEngine(
    PROJECT_DIR / "data" / "chess_pieces.json",
    cards_db_path=PROJECT_DIR / "data" / "cards.json",
)
recorder = GameRecorder(PROJECT_DIR / "data" / "logs")

# Laya 默认关闭：确认感知链路稳定后，改为 enabled=True 再启用。
# 首次启用会自动从 HuggingFace 下载模型权重（约 1-2GB）。
laya_engine = LayaEngine(confidence_threshold=0.85, enabled=False)
hybrid = HybridEngine(rule_engine, laya_engine)


@AgentServer.custom_recognition("WzwxqStateReco")
class WzwxqStateReco(CustomRecognition):
    """自定义识别器：整帧截图 → GameState，结果写入 detail 供后续节点/动作使用。"""

    def analyze(self, context: Context, argv) -> CustomRecognition.AnalyzeResult:
        import numpy as np

        image = context.tasker.controller.cached_image
        if image is None:
            return CustomRecognition.AnalyzeResult(box=None, detail="no image")
        state = extractor.extract(context, np.array(image))
        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 1, 1),
            detail=state.to_dict(),
        )


@AgentServer.custom_action("WzwxqDecide")
class WzwxqDecide(CustomAction):
    """自定义动作：提取状态 → 混合引擎决策 → 执行 → 记录。

    在准备/拍卖/天赋阶段由 Pipeline 对应节点循环调用。
    """

    def run(self, context: Context, argv) -> bool:
        import numpy as np

        image = context.tasker.controller.cached_image
        if image is None:
            logger.warning("无缓存截图，跳过本轮决策")
            return False

        state = extractor.extract(context, np.array(image))
        if state.phase not in ("prepare", "auction", "talent"):
            return True

        actions, diag = hybrid.decide(state)
        recorder.record_decision(state.to_dict(), actions, diag)

        executor = ActionExecutor(context.tasker.controller)
        executor.execute_all(actions)
        return True


@AgentServer.custom_action("WzwxqHeroPick")
class WzwxqHeroPick(CustomAction):
    """选棋手：默认选阿离（底部候选 index 2），点候选后点确认。"""

    def run(self, context: Context, argv) -> bool:
        executor = ActionExecutor(context.tasker.controller)
        executor.execute({"type": "hero_pick", "index": 2})
        return True


def main() -> None:
    sock_id = sys.argv[1] if len(sys.argv) > 1 else None
    AgentServer.start_up(sock_id)
    logger.info("AgentServer 已启动 (sock_id=%s)", sock_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
