#!/usr/bin/env bash
# 手牌条识别小模型 hand_strip_v1 —— 晚上跑（白天机器有人用，别跑）。
# 前提：没有对局在跑（训练会抢显卡：对局进行时不要在同一块显卡上训练）。
#   ps -W | grep -i MaaAssistantWZWXQ   # 先确认没有 live_match_agent
set -e
cd "$(dirname "$0")/../.."
PY=.venv-dfine/Scripts/python.exe          # 有 torch(cu128) + onnx；主 .venv 没装 onnx，导出会失败

# 1) 数据（已建好可跳过；重建约 10 分钟、4 进程、CPU）
[ -f data/vision/hand_strip_v1/labels.jsonl ] || \
  .venv/Scripts/python.exe -X utf8 tools/vision/build_hand_strip.py --workers 4

# 2) 训练 + 验证集选模型 + 测试集评一次 + 导出 ONNX + CPU 延迟（RTX 3060 预计 15~25 分钟）
$PY -X utf8 tools/vision/train_hand_strip.py --device cuda --epochs 40 --seed 0 \
    --out data/vision/runs/hand_strip_v1 \
    --export-to assets/model/vision/hand_strip_v1.onnx 2>&1 | tee data/vision/runs/hand_strip_v1.log

# 3) 看结果：data/vision/runs/hand_strip_v1/metrics.json
#    合格线（上线前）：test A+B 张数准确率 ≥ 0.99、空位误报帧率 ≤ 0.5%、抓取中心误差 p90 ≤ 12px、
#    N 档（已知空位）model_says_card_at_empty_x 明显低于 hand_layout 基线、CPU p90 ≤ 20ms。
