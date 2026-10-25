"""棋盘几何 + 规则常量（引擎内部只从这里取，方便以后换棋手 / 改版本时一处改）。

格子坐标照抄 agent/perception/extractors.OWN_BOARD_ROWS（不 import extractors：它会拉起 cv2 / onnx）。
test_engine.py 会核对两边一致。相邻关系与 agent/decision/rules._cell_neighbors 相同（蜂窝错位）。
"""
from __future__ import annotations

from functools import lru_cache

from agent.decision import meta

# 由近敌(row0) 到近己(row3)，每行 7 格
BOARD_ROWS: list[list[tuple[int, int]]] = [
    [(319, 375), (411, 372), (529, 372), (617, 372), (718, 371), (818, 374), (918, 368)],
    [(346, 430), (451, 433), (556, 430), (672, 430), (783, 438), (885, 437), (980, 430)],
    [(276, 501), (390, 502), (499, 500), (617, 496), (720, 500), (839, 502), (955, 502)],
    [(300, 569), (433, 569), (555, 569), (681, 571), (786, 569), (917, 570), (1027, 572)],
]
CELLS: list[tuple[int, int]] = [c for r in BOARD_ROWS for c in r]
_RC = {c: (r, i) for r, row in enumerate(BOARD_ROWS) for i, c in enumerate(row)}

# 【网查】升级所需经验 7/13/17/23/27（1→2 … 5→6）
XP_NEED = {1: 7, 2: 13, 3: 17, 4: 23, 5: 27}
MAX_LORD_LEVEL = meta.MAX_LORD_LEVEL
MAX_EQUIPS = meta.MAX_EQUIPS
HAND_LIMIT = 10                       # 【实测】手牌上限 10
HERO_COST = 3                         # 【实测】英雄牌统一 3 能量
REFRESH_COST = 2                      # 【实测】手动刷新约 2 能量


def cell_of(v) -> tuple[int, int] | None:
    """'x,y' / [x,y] / (x,y) → 最近的标准格（像素误差内吸附），读不出返回 None。"""
    if v is None:
        return None
    try:
        if isinstance(v, str):
            x, y = (int(float(t)) for t in v.split(",")[:2])
        else:
            x, y = int(v[0]), int(v[1])
    except Exception:
        return None
    if (x, y) in _RC:
        return (x, y)
    best = min(CELLS, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)
    return best if (best[0] - x) ** 2 + (best[1] - y) ** 2 <= 45 ** 2 else None


def row_of(cell) -> int:
    rc = _RC.get(tuple(cell)) if cell else None
    return rc[0] if rc else 2


@lru_cache(maxsize=None)
def neighbors(cell: tuple[int, int]) -> tuple:
    """蜂窝相邻：同行左右；偶数行连上下行 i-1、i，奇数行连 i、i+1（同 rules._cell_neighbors）。"""
    rc = _RC.get(tuple(cell))
    if not rc:
        return ()
    r, i = rc
    row = BOARD_ROWS[r]
    out = [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
    off = (i - 1, i) if r % 2 == 0 else (i, i + 1)
    for rr in (r - 1, r + 1):
        if 0 <= rr < len(BOARD_ROWS):
            out += [BOARD_ROWS[rr][j] for j in off if 0 <= j < len(BOARD_ROWS[rr])]
    return tuple(out)


@lru_cache(maxsize=None)
def hex_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    """格子步数（BFS），同 rules._hex_dist。不在棋盘上的返回 99。"""
    if a is None or b is None:
        return 99
    a, b = tuple(a), tuple(b)
    if a == b:
        return 0
    seen, frontier, d = {a}, [a], 0
    while frontier and d < 12:
        d += 1
        nxt = []
        for c in frontier:
            for n in neighbors(c):
                if n == b:
                    return d
                if n not in seen:
                    seen.add(n)
                    nxt.append(n)
        frontier = nxt
    return 99


def pop_cap(level: int) -> int:
    return meta.pop_cap_at(level)


def income(round_no: int) -> int:
    return meta.energy_income(round_no)


def xp_gain(round_no: int) -> int:
    """第 round_no 回合结束时自动得到的棋手经验（拍卖回合合计 3）。"""
    return meta.lord_xp_gain(round_no)


def level_cost(level: int, xp: int) -> int:
    """升级按钮上的数字 = 还差多少经验（1 能量 = 1 经验）。满级返回 0。"""
    if level >= MAX_LORD_LEVEL:
        return 0
    return max(0, XP_NEED.get(level, 99) - max(0, xp))
