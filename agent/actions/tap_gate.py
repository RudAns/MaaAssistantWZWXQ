"""点击 gate + 点击日志（2026-10-09，通用性路线 N1/N2）。

一、点击日志（clicks.jsonl，每局写在 run_dir 下）
    executor 的 _tap / _tap_hold / _drag 每次真正发出触控都记一条：
      ts、动作类型(tap/tap_hold/drag)、坐标、坐标来源 src（调用方传；没传就记调用栈函数名）、
      点击前最近一帧的 screen_cls 细类 + 置信度 + 那一帧离现在几秒、
      点击后约 1 秒的 screen_cls 细类（主循环下一帧补写，不额外截图、不拖慢主循环）。
    gate 拦下的点击也记一条（blocked=true），screen_ui 的"安全空白区"弱标签全靠这份日志。
    **任何异常都吞掉**，日志坏了不能影响操作。

二、写死坐标兜底点位按界面放行（SAFE_TAPS）
    "通用兜底点位害人"在开发记录里至少出现了 7 次，**每次都是按某个界面量的点被用在了另一个界面上**
    （拍卖点空地取消拍品、棋手面板内点空地等于没点、战斗中每回合盲点棋盘 184 次、局外点进棋手页卡 12 分钟…）。
    现在这些点在点之前先问 screen_cls 当前是什么界面：
      - 细类在该点位的允许集合里 → 点；
      - 不在 → **不点**，记日志 + 存证据帧（同一 (界面, 点位) 每回合最多 1 张）；
      - screen_cls 不可用 / 置信度 < threshold → **保持原行为**（点），日志记一笔 "uncertain"。
    verified=True（调用前 OCR 已经认出了要关的那个面板，如看场关英雄详情、看牌关放大图、设置面板）
    → 只做影子记录（会拦就记 "shadow_block"），不拦：OCR 已经给了直接证据，模型在这些类上样本少/不稳。
    **OCR 认出按钮文字再点**的那类点击是运行时定位，不走 gate。

SAFE_TAPS 的来源：路线图第 1 节 T 段每个点位"是给哪个界面量的" + 历史证据帧上 screen_cls 的实测分布
（2026-10-09 抽样：prep_unknown_escape 150 张 → battle 124 / 低置信 26，**全是战斗画面**；
 hero_panel / merge_check / board_inspect → hero_panel；inspect_card / inspect_awake → enlarged_detail；
 phase_lord_panel → lord_panel；dismiss_detail 发出时 → talent；lobby_popup → 低置信 transition）。
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------- 点位名 -> 坐标（只用于日志 / 文档；真正点哪里由调用方给） ----------
# 局外弹窗那几个 ✕ / 返回 / 标题页点位统一叫 LOBBY_SPOT（文字判据是 OCR，位置是写死的）
POINTS = {
    "TAP_SIDE_GROUND": (1120, 500),      # 棋盘右侧空地：关英雄详情面板 / 放大详情、准备阶段 unknown 逃生
    "TAP_CLOSE_BIG_CARD": (400, 60),     # 场馆背景：关放大图、关天赋放大详情、英雄面板第 3~4 次
    "BTN_LORD_PANEL_CLOSE": (1240, 300), # 棋手详情面板外侧
    "BTN_SETTINGS_CLOSE": (76, 30),      # 设置面板左上 ✕
    "LOBBY_SPOT": None,                  # 局外弹窗：(1009,111)/(1188,105)/(1138,156)/(640,630)/(640,590)/(640,690)/(35,40)/(1187,47)
}

# 局外 / 结算 / 过场类细类（局外弹窗的 ✕ 只允许在这些界面上点）
_OUT_OF_MATCH = {"lobby", "mode_lobby", "match_found", "loading", "settle", "settle_final", "post_match",
                 "transition", "notice", "solo_popup", "other"}

# 细类 -> 允许的点位名。**不在表里的细类 = 什么写死点都不点**（battle / board_idle / auction / reward_popup /
# card_pick / equip_popup / prepare_shop / hero_pick / lord_boost …）
SAFE_TAPS: dict[str, set[str]] = {
    # 英雄详情面板（也盖住召唤物面板）：面板外的棋盘右侧空地能关；第 3~4 次换场馆背景
    "hero_panel": {"TAP_SIDE_GROUND", "TAP_CLOSE_BIG_CARD"},
    # 放大详情（看牌、选完 N 选 1 / 棋手特性后的大卡）：两个点历史上都关得掉
    "enlarged_detail": {"TAP_SIDE_GROUND", "TAP_CLOSE_BIG_CARD"},
    # 天赋：只允许场馆背景（dismiss_detail —— 选完天赋的放大详情 screen_cls 判成 talent，6/7 次 ≥0.9）。
    # **不允许 TAP_SIDE_GROUND**：天赋界面那儿是等级进度条 / 卡面，点了可能把三选一收起
    "talent": {"TAP_CLOSE_BIG_CARD"},
    "lord_panel": {"BTN_LORD_PANEL_CLOSE"},
    "settings": {"BTN_SETTINGS_CLOSE"},
}
for _lb in _OUT_OF_MATCH:
    SAFE_TAPS.setdefault(_lb, set()).add("LOBBY_SPOT")

# ---------- 运行期配置（主循环注入） ----------
_CFG: dict = {"run_dir": None, "probe": None, "round": lambda: -1}
_LAST_FRAME: dict = {"label": None, "conf": None, "ts": 0.0}
_PENDING: list = []              # 等补写"点击后 1 秒界面"的记录
_LOCK = threading.Lock()
_EVID_SEEN: set = set()          # (回合, 细类, 点位名)：证据帧每回合每类一张
STATS = {"allow": 0, "block": 0, "uncertain": 0, "shadow_block": 0, "logged": 0}
AFTER_MIN_S = 0.8                # 点击后至少过这么久的帧才算"点击后的界面"
AFTER_MAX_S = 15.0               # 超过这么久还没有新帧 → after 留空写出


def configure(run_dir=None, probe: Optional[Callable] = None, round_getter: Optional[Callable] = None) -> None:
    """probe(): 无参，返回一帧整屏 RGB（用于 gate 现截一帧）；round_getter(): 当前回合号（证据帧限流用）。"""
    if run_dir is not None:
        _CFG["run_dir"] = Path(run_dir)
    if probe is not None:
        _CFG["probe"] = probe
    if round_getter is not None:
        _CFG["round"] = round_getter


def _classify(image):
    try:
        from agent.perception import screen_cls as _SC
        r = _SC.classify(image)
        if r is None:
            return None, None, None
        return r[0], float(r[1]), _SC.threshold()
    except Exception:                                           # noqa: BLE001
        return None, None, None


def _caller(depth: int = 3) -> str:
    """没传 src 时记调用栈里的函数名（跳过本模块和 executor 内部的包装层）。"""
    try:
        names = []
        f = sys._getframe(1)
        while f is not None and len(names) < depth:
            fn = f.f_code.co_name
            mod = Path(f.f_code.co_filename).stem
            if mod not in ("tap_gate",) and fn not in ("_tap", "_tap_hold", "_drag", "_log_touch"):
                names.append(fn)
            f = f.f_back
        return "<".join(names) or "?"
    except Exception:                                           # noqa: BLE001
        return "?"


def _write(rec: dict) -> None:
    rd = _CFG.get("run_dir")
    if rd is None:
        return
    try:
        with open(Path(rd) / "clicks.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        STATS["logged"] += 1
    except Exception:                                           # noqa: BLE001
        pass


def log_touch(kind: str, pts, src: Optional[str] = None, extra: Optional[dict] = None) -> None:
    """executor 每次发触控调用。pts = [(x,y)] 或拖拽 [(x0,y0),(x1,y1)]。"""
    try:
        now = time.time()
        lf = dict(_LAST_FRAME)
        rec = {"ts": round(now, 3), "kind": kind, "pts": [[int(p[0]), int(p[1])] for p in pts],
               "src": src or _caller(),
               "pre": [lf["label"], lf["conf"], round(now - lf["ts"], 2) if lf["ts"] else None],
               "after": None, "round": _CFG["round"]()}
        if extra:
            rec.update(extra)
        with _LOCK:
            _PENDING.append(rec)
    except Exception:                                           # noqa: BLE001
        pass


def note_frame(label, conf, ts: Optional[float] = None) -> None:
    """主循环每帧调用（以及 gate 现截的帧）：更新"最近一帧界面"，并给 ≥0.8 秒前的点击补写"点击后界面"。"""
    try:
        ts = ts or time.time()
        if label is not None:
            _LAST_FRAME.update(label=label, conf=round(float(conf), 3), ts=ts)
        done = []
        with _LOCK:
            keep = []
            for rec in _PENDING:
                dt = ts - rec["ts"]
                if label is not None and dt >= AFTER_MIN_S:
                    rec["after"] = [label, round(float(conf), 3), round(dt, 2)]
                    done.append(rec)
                elif dt > AFTER_MAX_S:
                    done.append(rec)
                else:
                    keep.append(rec)
            _PENDING[:] = keep
        for rec in done:
            _write(rec)
    except Exception:                                           # noqa: BLE001
        pass


def flush() -> None:
    with _LOCK:
        rest = list(_PENDING)
        _PENDING.clear()
    for rec in rest:
        _write(rec)


def allowed(label: str, name: str) -> bool:
    return name in SAFE_TAPS.get(label, ())


def decide(name: str, image=None, label=None, conf=None) -> tuple[str, Optional[str], Optional[float]]:
    """纯判断（离线测试用）：-> (结果, 细类, 置信)。结果 ∈ allow / block / uncertain。"""
    thr = None
    if label is None and image is not None:
        label, conf, thr = _classify(image)
    if thr is None:
        try:
            from agent.perception import screen_cls as _SC
            thr = _SC.threshold()
        except Exception:                                       # noqa: BLE001
            thr = 0.9
    if label is None or conf is None or conf < thr:
        return "uncertain", label, conf
    return ("allow" if allowed(label, name) else "block"), label, conf


def _save_evidence(image, label: str, name: str) -> None:
    try:
        rd = _CFG.get("run_dir")
        if rd is None or image is None:
            return
        key = (_CFG["round"](), label, name)
        if key in _EVID_SEEN:
            return
        _EVID_SEEN.add(key)
        import cv2
        import numpy as np
        ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        if ok:
            buf.tofile(str(Path(rd) / f"tap_blocked_{name}_{label}_{int(time.time())}.png"))
    except Exception:                                           # noqa: BLE001
        pass


def gate(name: str, point, image=None, verified: bool = False, why: str = "") -> bool:
    """写死坐标的兜底点击之前调用。返回 True = 照点，False = 不点。
    image: 当前帧整屏 RGB（没有就用 probe 现截一帧）。任何异常 → True（保持原行为）。"""
    try:
        if image is None and _CFG.get("probe") is not None:
            try:
                image = _CFG["probe"]()
            except Exception:                                   # noqa: BLE001
                image = None
        res, label, conf = decide(name, image=image) if image is not None else ("uncertain", None, None)
        if image is not None and label is not None:
            note_frame(label, conf)
        pt = [int(point[0]), int(point[1])] if point else None
        if res == "block" and verified:
            res = "shadow_block"
        STATS[res] = STATS.get(res, 0) + 1
        if res == "allow":
            return True
        if res in ("uncertain", "shadow_block"):
            logger.info("【点击 gate】%s %s：界面=%s(%s) —— %s，照点（%s）", name, pt, label,
                        f"{conf:.2f}" if conf is not None else "-",
                        "模型不确定" if res == "uncertain" else "OCR 已确认面板，只影子记录", why)
            return True
        logger.warning("【点击 gate】拦下 %s %s：当前界面 %s(%.2f) 不在允许集合 %s —— 不点（%s）",
                       name, pt, label, conf, sorted(k for k, v in SAFE_TAPS.items() if name in v), why)
        _write({"ts": round(time.time(), 3), "kind": "blocked", "pts": [pt] if pt else [], "src": f"gate:{name}",
                "pre": [label, round(conf, 3), 0.0], "after": None, "round": _CFG["round"](), "blocked": True,
                "why": why})
        _save_evidence(image, label, name)
        return False
    except Exception:                                           # noqa: BLE001
        return True
