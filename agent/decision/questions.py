"""Laya typed questions 定义：把自走棋决策点映射为 choice / score / noul 问题。"""

from __future__ import annotations

from typing import Any

from ..perception.state import GameState


def build_lord_boost_question(state: GameState, options) -> dict[str, Any]:
    """棋手特性三选一。规则层已剔掉结构性废牌，这里只问"剩下的哪个最值"。

    喂给模型的是**效果文本 + 当前局势**，不是卡名——卡名对模型是纯噪声
    （基准测试：中文卡名输入下 buy_which 置信度低于随机基线）。
    """
    return {
        "lord_boost": {
            "type": "choice",
            "instructions": (
                f"我是单核阵容，主力英雄是{state.__dict__.get('_carry', '主C')}，"
                f"现在第{state.round_no}回合，我的血量{state.hp}。"
                "棋手技能会给指定英雄挂护盾并在受伤时转移伤害，"
                "技能任务按该英雄的击杀数推进。下面哪个强化对我这局帮助最大？"),
            "criteria": {o.name: (o.desc or o.name) for o in options},
        }
    }


def build_questions(state: GameState) -> dict[str, Any]:
    """基于当前状态动态构建 Laya 问题集。"""
    questions: dict[str, Any] = {}

    shop_options = {
        f"槽位{s.index}_{s.piece}": f"{s.piece}，{s.cost or '?'}费棋子"
        for s in state.shop
        if s.piece
    }
    if shop_options:
        shop_options["都不买"] = "放弃本次购买，存钱吃利息"
        questions["buy_which"] = {
            "type": "choice",
            "instructions": "根据当前金币、血量、阵容羁绊，商店里最应该购买哪个棋子？",
            "criteria": shop_options,
        }

    questions["should_refresh"] = {
        "type": "noul",
        "instructions": "当前是否应该花费2金币刷新商店寻找关键棋子？",
    }
    questions["should_level_up"] = {
        "type": "noul",
        "instructions": "当前是否应该花费4金币购买经验以提升人口上限？",
    }
    questions["urgency"] = {
        "type": "score",
        "instructions": "当前局势的紧急程度如何（决定存钱发育还是花钱提战力）？",
        "criteria": ["局势安全，安心存钱发育", "需要适度投入提升战力", "血量危险，必须全力花钱提战力"],
    }

    if len(state.bench) > 1:
        bench_options = {
            f"备战席{c.col}_{c.piece}": f"{c.piece}，{c.star}星" for c in state.bench
        }
        bench_options["都不卖"] = "所有备战席棋子都保留"
        questions["sell_which"] = {
            "type": "choice",
            "instructions": "如果必须卖一个备战席棋子腾位置或换金币，卖哪个损失最小？",
            "criteria": bench_options,
        }

    return questions
