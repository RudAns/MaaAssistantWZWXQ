"""离线检查：写死坐标兜底点位的界面 gate（agent/actions/tap_gate.py 的 SAFE_TAPS）。

不要模拟器、不用 GPU（screen_cls 跑 CPU）。用历史证据帧验证：
  1. 战斗画面上点「棋盘右侧空地」TAP_SIDE_GROUND 会被拦
     —— prep_unknown_escape_*（准备阶段 unknown 逃生时存的帧，实际几乎全是战斗画面）、模型判 battle 的 stuck_*；
  2. 选完天赋的放大详情点 (400,60) TAP_CLOSE_BIG_CARD 会放行
     —— fixtures/talent_enlarged_detail.png + 日志里发出 dismiss_detail 那一刻存的 phase_talent 帧；
  3. 英雄详情面板 / 看牌放大图 / 棋手面板 上对应的关闭点放行；拍卖界面上任何写死点都不放；
  4. 低置信度 → uncertain（保持原行为，不拦）。
注意：磁盘上的图是 BGR，喂 screen_cls 前要转 RGB。

用法：.venv/Scripts/python.exe -X utf8 tools/test_safe_taps.py [--n 60]
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.actions import tap_gate  # noqa: E402
from agent.perception import screen_cls as SC  # noqa: E402

LOGS = ROOT / "data" / "logs"
FIX = LOGS / "fixtures"


def load_rgb(fp) -> np.ndarray | None:
    im = cv2.imdecode(np.fromfile(str(fp), np.uint8), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)     # 磁盘 BGR -> 模型要 RGB


def sample(pattern: str, n: int) -> list[str]:
    fs = sorted(glob.glob(str(LOGS / pattern)), key=os.path.getmtime)[-n * 3:]
    random.Random(0).shuffle(fs)
    return fs[:n]


FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def run(n: int) -> int:
    if SC.classify(np.zeros((720, 1280, 3), np.uint8)) is None:
        print("screen_cls 加载失败 —— gate 线上会全部退回 uncertain（不拦），本检查无法进行")
        return 1

    # ---- 1. 战斗画面点棋盘右侧空地：拦 ----
    print("[1] 战斗画面 x TAP_SIDE_GROUND")
    res = {"block": 0, "allow": 0, "uncertain": 0}
    labels = {}
    for fp in sample("live_*/prep_unknown_escape_*.png", n) + sample("live_*/stuck_*.png", n):
        im = load_rgb(fp)
        if im is None:
            continue
        r, lb, cf = tap_gate.decide("TAP_SIDE_GROUND", image=im)
        if lb in ("battle", "board_idle") and r != "uncertain":
            res[r] += 1
        elif r == "allow":
            labels[os.path.basename(fp)] = (lb, cf)
    print("   模型高置信判战斗的帧：", res)
    check(res["block"] >= 20, f"至少 20 张高置信战斗帧被拦（实际 {res['block']}）")
    check(res["allow"] == 0, "战斗帧没有一张放行")
    if labels:
        print("   （非战斗、被放行的帧，供核对）", list(labels.items())[:5])

    # ---- 2. 选完天赋的放大详情点 (400,60)：放行 ----
    print("[2] 天赋放大详情 x TAP_CLOSE_BIG_CARD")
    talent_frames = [FIX / "talent_enlarged_detail.png"] + [
        Path(p) for p in glob.glob(str(LOGS / "live_1791319214" / "phase_talent_179132045[38].png"))
        + glob.glob(str(LOGS / "live_1791319214" / "phase_talent_1791320462.png"))]
    talent_frames = [p for p in talent_frames if p.exists()]
    check(len(talent_frames) >= 1, f"找到天赋放大详情帧 {len(talent_frames)} 张")
    for fp in talent_frames:
        r, lb, cf = tap_gate.decide("TAP_CLOSE_BIG_CARD", image=load_rgb(fp))
        check(r in ("allow", "uncertain"), f"{fp.name}: {lb}({cf:.2f}) -> {r}")
        r2, _, _ = tap_gate.decide("TAP_SIDE_GROUND", image=load_rgb(fp))
        if lb == "talent" and r != "uncertain":
            check(r2 == "block", f"{fp.name}: 天赋界面上不点棋盘右侧空地 -> {r2}")

    # ---- 3. 面板类：对应关闭点放行 ----
    print("[3] 面板 x 关闭点")
    cases = [("hero_panel_blocks_levelup.png", "TAP_SIDE_GROUND"),
             ("inspect_big_card_normal.png", "TAP_CLOSE_BIG_CARD"),
             ("inspect_big_card_awake.png", "TAP_CLOSE_BIG_CARD"),
             ("lord_panel.png", "BTN_LORD_PANEL_CLOSE")]
    for name, pt in cases:
        fp = FIX / name
        if not fp.exists():
            print(f"  skip {name}（没有这张 fixture）")
            continue
        r, lb, cf = tap_gate.decide(pt, image=load_rgb(fp))
        check(r in ("allow", "uncertain"), f"{name} x {pt}: {lb}({cf:.2f}) -> {r}")
    # 棋手面板内部点棋盘右侧空地 = 点了等于没点（09-22 死循环）→ 要拦
    fp = FIX / "lord_panel.png"
    if fp.exists():
        r, lb, cf = tap_gate.decide("TAP_SIDE_GROUND", image=load_rgb(fp))
        check(r in ("block", "uncertain"), f"lord_panel x TAP_SIDE_GROUND: {lb}({cf:.2f}) -> {r}（应拦）")
    hp = sample("live_*/merge_check_*.png", min(n, 40))
    st = {"allow": 0, "block": 0, "uncertain": 0}
    for f in hp:
        im = load_rgb(f)
        if im is not None:
            st[tap_gate.decide("TAP_SIDE_GROUND", image=im)[0]] += 1
    print("   merge_check（核对单位弹出的英雄详情）x TAP_SIDE_GROUND：", st)
    check(st["block"] <= max(1, len(hp) // 20), "英雄详情面板上关面板的点基本不被拦（≤5%）")

    # ---- 4. 拍卖：任何写死点都不放 ----
    # phase_auction_* 是"阶段切到拍卖"那一帧，少数还是入场过场动画（模型判 transition，局外 ✕ 在 transition 上是放行的，
    # 而局外 ✕ 本身还要 OCR 认出特定弹窗文字才会点）—— 这里只统计模型判成 auction / reward_popup 的帧
    print("[4] 拍卖 x 写死点")
    st = {"allow": 0, "block": 0, "uncertain": 0}
    other = {}
    for f in sample("live_*/phase_auction_*.png", n):
        im = load_rgb(f)
        if im is None:
            continue
        for pt in ("TAP_SIDE_GROUND", "TAP_CLOSE_BIG_CARD", "LOBBY_SPOT"):
            r, lb, _cf = tap_gate.decide(pt, image=im)
            if r == "uncertain" or lb in ("auction", "reward_popup"):
                st[r] += 1
            else:
                other[lb] = other.get(lb, 0) + 1
    print("   模型判拍卖 / 低置信：", st, "  其余细类（点位次数）：", other)
    check(st["allow"] == 0, "拍卖帧上写死点一次都不放行")

    # ---- 5. 低置信 -> uncertain；局外点位只在局外放 ----
    print("[5] 门槛与局外点位")
    check(tap_gate.decide("TAP_SIDE_GROUND", label="battle", conf=0.5)[0] == "uncertain", "置信 0.5 -> uncertain（不拦）")
    check(tap_gate.decide("LOBBY_SPOT", label="prepare", conf=0.99)[0] == "block", "局外 ✕ 在准备画面上 -> 拦")
    check(tap_gate.decide("LOBBY_SPOT", label="lobby", conf=0.99)[0] == "allow", "局外 ✕ 在大厅 -> 放行")
    check(tap_gate.decide("TAP_SIDE_GROUND", label="auction", conf=0.99)[0] == "block", "拍卖 x 空地 -> 拦")
    check(tap_gate.gate("TAP_SIDE_GROUND", (1120, 500), image=None) is True, "没有画面 / probe → 保持原行为（点）")

    print()
    if FAILS:
        print(f"失败 {len(FAILS)} 项：")
        for m in FAILS:
            print("  -", m)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    sys.exit(run(ap.parse_args().n))
