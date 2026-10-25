"""通用局面判断引擎 v1 —— 第 2 层推演器。

只依据游戏规则（data/keywords.json、agent/decision/meta.py）+ 卡面 + 技能；
不读推荐阵容，不写死棋手 / 路线 / 英雄名字。

  effects.py  第 1 层效果表加载 + Effect 数据类 + amount 公式求值
  standin.py  effects_v1.json 未生成时的替身（从 cards.json 卡面正则抽取，同一 Effect 格式）
  board.py    棋盘格子、步数距离、规则常量（经验表 / 人口 / 收入，转引 meta）
  state.py    通用局面 Position / Unit；from_live / from_point
  project.py  按回合推演（整备 → 开团 → 战斗估计 → 击杀/牺牲/阵亡 → 收入/经验/升级）
  odds.py     商店命中、刷新、凑齐一组英雄的回合分布
  options.py  枚举可行动作 → 推演 → 价值 V 与分项、ΔV
"""
