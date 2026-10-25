"""手牌条识别小模型的数据集：历史准备帧 + 运行时状态 → 弱标签（每张手牌的左边缘 / 露出右边缘 / 枫叶舞槽 / 觉醒）。

  .venv/Scripts/python.exe -X utf8 tools/vision/build_hand_strip.py [--workers 4] [--test-runs 60] [--val-runs 60]

起因（2026-10-09：经常拖动空牌位、误触出表情发送轮盘）：手牌"有几张、每张在哪"靠
extractors.hand_layout（霍夫圆找左上角图标 + 枫叶舞红色角标 + 网格打分），会推错张数
（被点起的牌错位、翩跹舞步这一族牌没有图标、场地背景上的圆被当成图标）。改用"固定区域裁图 + 专用小 CNN"。

**标签从哪来（都是弱标签，靠多条独立证据互相印证来筛）**
  1. 帧：data/logs/live_*/phase_prepare_<ts>.png（阶段切换那一帧，和 states.jsonl 里同一时刻的状态是同一张截图）。
  2. 运行时状态：states.jsonl 里 timestamp ∈ [ts-4, ts+1] 的最后一条 prepare 记录 → hand_slots（x = 露出那一截的中心）。
  3. 现在的 hand_layout 在这张图上重算一遍 → 图标 x（= 卡牌左边缘 + 20）。
  可信（tier=A）要同时满足：
    a. 运行时 hand_by_layout=True、hand_unread 为空（每个位置都读出了一个真卡名；闭集解码有"空白"选项，空位读不出名字）；
    b. 运行时的张数 / 每张中心 与 现在重算的布局一致（±8px）；
    c. 图标核对：网格上每个点都有检测到的图标（没图标的翩跹舞步一族除外），网格范围内没有多出来的图标；
    d. 时间稳定：前后 6 秒内有另一条 prepare 记录读到完全一样的手牌序列（= 手牌账"连续两帧一致才校正"那条规则）；
    e. 最后一张牌右边那一截是场地背景（_slot_is_ground），排除"少数一张"。
  只满足 a+b+d（缺图标 / 最后一张右边像卡底）= tier=B。抽 20 帧看图全对（右边常是暗色场地），
  所以 B 也进 val/test，评估时 A、A+B 分开报。立绘里的圆被当"多出来的图标"不降档。
  其余丢弃（统计里记原因）。
  **手牌账**（rules._hand_book）没有逐帧落盘，ledger.jsonl 只有回合号没有时间戳，无法逐帧对齐；
  d 条就是手牌账校正自己用的判据，用它代替。
  3'. 拖前核对帧 grab_shadow_pre_<piece>_<..>_<ms>.png（同样按 a~e 核对，状态窗口 [t-6, t+0.5]），
     以及同一次拖拽的半空帧 grab_shadow_mid（手牌条整体下沉 ~28px、被拖那张抬起或离开牌条）：
     pre 是 A 档才收，借用 pre 的标签，被拖那张标 held（训练时它那几列不监督）= tier=M。
  3''. 0 张手牌（只有枫叶舞槽）：运行时布局要求 ≥1 张所以这种帧 hand_by_layout=False，单独认（tier=Z）：
     本帧和前后 6 秒另一帧都只读到枫叶舞、手牌行一个图标都没检测到。
  4. 空位样本：inspect_no_view_<ts>.png（点开看牌没弹出详情时存的帧）。点的 x 从 agent 日志
     「【看牌】点了 x=… 没看到放大详情」+ 紧跟的「证据帧已存: inspect_no_view_<ts>.png」取。
     那一列像素确实是场地背景才收（tier=N，只监督 x±16px 那几列 = 背景）；牌还在飞入动画里没弹出详情的不收。
  5. 觉醒：data/vision/hand_awaken_v1/labels.jsonl（人工核对过的）按 (帧, 图标 x) 并上；没有的不监督。

输出 data/vision/hand_strip_v1/：
  crops/<id>.jpg     裁图 BGR（y 528~720, x 160~1184 → 缩到 96x512，q=92）
  labels.jsonl       id, frame, run, split, tier, n, cards=[{l, r, icon, name, awk}], skill=[l, r]|None, empty_x, shop_open
  stats.json         样本数、张数分布、叠放比例、各条筛选的淘汰数
划分按对局（运行目录）时间：最新 --test-runs 局 = test，再往前 --val-runs 局 = val，其余 train（不泄漏、测试集代表最新 UI）。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "vision" / "hand_strip_v1"
LOGS = ROOT / "data" / "logs"

# 裁图区域（1280x720 坐标）：卡牌顶边 ≈565，被点起的牌会再往上抬，留到 528；
# 左边罩住叠放时 x≈190 的枫叶舞槽，右边到最后一张牌右边缘（≈1125）再留一点，不进能量圆盘（≈1200）
X0, X1, Y0, Y1 = 160, 1184, 528, 720
SCALE = 2                      # 存盘缩小 2 倍：96x512
CARD_W = 142                   # 一张牌的宽度（= 不叠放时的牌距 HAND_PITCH_MAX）
SKILL_HALF = 75                # 枫叶舞槽半宽（卡宽 ~151）
MATCH_TOL = 8
STABLE_S = 6.0


def _imread(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def iconless(name: str) -> bool:
    return name.startswith("翩跹")


def load_states(run: Path) -> list[dict]:
    """只留 prepare 记录的必要字段。"""
    out = []
    f = run / "states.jsonl"
    if not f.exists():
        return out
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"phase": "prepare"' not in line[:200]:
                continue
            try:
                r = json.loads(line)
            except Exception:                                       # noqa: BLE001
                continue
            s = r.get("state") or {}
            if s.get("phase") != "prepare":
                continue
            out.append({"t": float(s.get("timestamp") or r.get("ts") or 0),
                        "slots": s.get("hand_slots") or [], "unread": s.get("hand_unread") or [],
                        "by_layout": bool(s.get("hand_by_layout")), "shop_open": bool(s.get("shop_open")),
                        "round": s.get("round_no", 0)})
    out.sort(key=lambda r: r["t"])
    return out


def no_view_clicks() -> dict:
    """{(run, ts): x}：从 agent 日志里取"点开看牌没看到放大详情"时点的 x。"""
    res = {}
    for lf in sorted(LOGS.glob("*.log")):
        run, last_x = None, None
        try:
            fh = open(lf, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if "输出=" in line:
                    m = re.search(r"(live_\d+)", line)
                    run = m.group(1) if m else run
                elif "没看到放大详情" in line:
                    m = re.search(r"点了 x=(\d+)", line)
                    last_x = int(m.group(1)) if m else None
                elif "证据帧已存: inspect_no_view_" in line and run and last_x is not None:
                    m = re.search(r"inspect_no_view_(\d+)\.png", line)
                    if m:
                        res[(run, m.group(1))] = last_x
                    last_x = None
    return res


def awaken_labels() -> dict:
    """{(帧相对路径, 图标x//4): 0/1}，来自人工核对过的 hand_awaken_v1。"""
    d = ROOT / "data" / "vision" / "hand_awaken_v1"
    if not (d / "labels.jsonl").exists():
        return {}
    pool = {}
    for line in (d / "pool.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        pool[r["id"]] = (r["frame"].replace("\\", "/"), r["x"])
    out = {}
    for line in (d / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["id"] in pool and r.get("label") in (0, 1):
            f, x = pool[r["id"]]
            out[(f, int(x) // 4)] = int(r["label"])
    return out


def save_crop(bgr: np.ndarray, cid: str) -> None:
    c = bgr[Y0:Y1, X0:X1]
    c = cv2.resize(c, ((X1 - X0) // SCALE, (Y1 - Y0) // SCALE), interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(OUT / "crops" / f"{cid}.jpg"))


def cards_from_icons(xs: list[float]) -> list[dict]:
    cards = []
    for k, ix in enumerate(xs):
        left = ix - 20
        right = (xs[k + 1] - 20) if k + 1 < len(xs) else left + CARD_W
        cards.append({"l": round(left, 1), "r": round(right, 1), "icon": round(ix, 1)})
    return cards


def zero_frame(E, rgb, states, i, why):
    """0 张手牌（只有枫叶舞槽）：运行时这种帧不走图标布局（布局要求 ≥1 张），单独认。
    条件：本帧和前后 6 秒内另一帧都只读到枫叶舞、没有认不出的位置；手牌那一行一个卡牌图标都没检测到。"""
    st = states[i]
    if st["unread"] or not st["slots"] or any(s["name"] != "枫叶舞" for s in st["slots"]):
        return None
    ok = any(j != i and abs(states[j]["t"] - st["t"]) <= STABLE_S and not states[j]["unread"]
             and [s["name"] for s in states[j]["slots"]] == ["枫叶舞"]
             for j in range(max(0, i - 6), min(len(states), i + 7)))
    if not ok:
        why["Z:前后帧不一致"] += 1
        return None
    icons, _b, _y = E._hand_circles(rgb)
    if icons:
        why["Z:检测到图标"] += 1
        return None
    why["tierZ"] += 1
    x = st["slots"][0]["x"]
    return [x - SKILL_HALF, x + SKILL_HALF]


def label_frame(E, rgb, states, i, why) -> tuple:
    """按 a~e 条核对一帧，返回 (tier, cards, skill, names, lay) 或 (None, ...)。"""
    st = states[i]
    if not st["by_layout"]:
        why["运行时没走图标布局"] += 1
        return (None,) * 5
    if st["unread"]:
        why["有认不出的位置"] += 1
        return (None,) * 5
    lay = E.hand_layout(rgb)
    if not lay:
        why["重算布局失败"] += 1
        return (None,) * 5
    xs = lay["xs"]
    skill = [s for s in st["slots"] if s["name"] == "枫叶舞" or s.get("ready") is not None]
    hand = sorted((s for s in st["slots"] if s not in skill), key=lambda s: s["x"])
    if len(hand) != lay["n"]:
        why["张数与重算不一致"] += 1
        return (None,) * 5
    cards = cards_from_icons(xs)
    if any(abs(s["x"] - (c["l"] + c["r"]) / 2) > MATCH_TOL for s, c in zip(hand, cards)):
        why["位置与重算不一致"] += 1
        return (None,) * 5
    names = [s["name"] for s in hand]
    seq = tuple(names)
    stable = False
    for j in range(max(0, i - 6), min(len(states), i + 7)):
        if j != i and abs(states[j]["t"] - st["t"]) <= STABLE_S and states[j]["by_layout"]                 and not states[j]["unread"]:
            o = tuple(s["name"] for s in sorted(states[j]["slots"], key=lambda s: s["x"])
                      if not (s["name"] == "枫叶舞" or s.get("ready") is not None))
            if o == seq:
                stable = True
                break
    if not stable:
        why["前后帧不一致"] += 1
        return (None,) * 5
    icons, _badge, _y = E._hand_circles(rgb)
    near = lambda a, b: abs(a - b) <= E.HAND_ICON_TOL   # noqa: E731
    miss = [k for k, g in enumerate(xs) if not any(near(g, x) for x in icons) and not iconless(names[k])]
    extra = [x for x in icons if xs and xs[0] - 10 <= x <= xs[-1] + 10 and not any(near(g, x) for g in xs)]
    # 最后一张右边紧挨着的那一截不能是深色卡底（卡牌下半截均值 30~100；场地 >105，起伏大小不论）
    if xs:
        rx = int(xs[-1] - 20 + CARD_W)
        patch = rgb[655:700, rx + 6:rx + 40]
        right_ok = rx + 40 > 1180 or (patch.size > 0 and float(patch.mean()) > 105)
    else:
        right_ok = True
    # 立绘里的圆会被当成"多出来的图标"（盾山头盔、古币），不作为降档理由，只记数
    tier = "A" if (not miss and right_ok) else "B"
    why[f"tier{tier}"] += 1
    if tier == "B":
        why["B:缺图标" if miss else "B:右侧像卡底"] += 1
    for c, nm, hs in zip(cards, names, hand):
        c["name"] = nm
        c["awk"] = 1 if hs.get("awakened") else None
    sk = [skill[0]["x"] - SKILL_HALF, skill[0]["x"] + SKILL_HALF] if skill else None
    return tier, cards, sk, names, lay


def _read(f: Path):
    bgr = _imread(f)
    if bgr is None:
        return None, None
    if bgr.shape[:2] != (720, 1280):
        bgr = cv2.resize(bgr, (1280, 720))
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # 磁盘是 BGR，extractors 吃 RGB


def process_run(args) -> tuple[list[dict], collections.Counter]:
    run_dir, nv_map, awk = args
    import bisect
    from agent.perception import extractors as E
    run = Path(run_dir)
    rows, why = [], collections.Counter()
    states = load_states(run)
    ts_list = [s["t"] for s in states]

    def add(f, bgr, cid, tier, cards, sk, lay, st, src, **extra):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        for c in cards:
            if c.get("awk") is None:
                k4 = int(c["icon"]) // 4
                c["awk"] = awk.get((rel, k4), awk.get((rel, k4 + 1), awk.get((rel, k4 - 1))))
        save_crop(bgr, cid)
        rows.append({"id": cid, "frame": rel, "run": run.name, "src": src, "tier": tier, "n": len(cards),
                     "cards": cards, "skill": sk, "shop_open": st["shop_open"], "round": st["round"],
                     "by": lay["by"] if lay else None, "stacked": len(cards) >= 6, "empty_x": None, **extra})

    # 1) 阶段切换帧：截图时刻 ≈ 文件名秒数（存图在提取之后），对应 [ts-4, ts+1] 里最后一条 prepare 记录
    for f in sorted(run.glob("phase_prepare_*.png")):
        why["frames_phase"] += 1
        t = int(f.stem.split("_")[-1])
        i = bisect.bisect_right(ts_list, t + 1) - 1
        if i < 0 or states[i]["t"] < t - 4:
            why["无对应状态"] += 1
            continue
        bgr, rgb = _read(f)
        if bgr is None:
            continue
        if not states[i]["by_layout"]:
            sk = zero_frame(E, rgb, states, i, why)
            if sk:
                add(f, bgr, f"{run.name[5:]}_{t}", "Z", [], sk, None, states[i], "phase")
                continue
        tier, cards, sk, names, lay = label_frame(E, rgb, states, i, why)
        if tier:
            add(f, bgr, f"{run.name[5:]}_{t}", tier, cards, sk, lay, states[i], "phase")
    # 2) 拖前核对帧（grab_shadow_pre）+ 同一次拖拽的半空帧（grab_shadow_mid：手牌条整体下沉 ~28px，
    #    被拖的那张抬起 / 已离开牌条）。pre 按状态核对；mid 借用 pre 的标签，被拖那张的列不监督（held）。
    for f in sorted(run.glob("grab_shadow_pre_*.png")):
        why["frames_grab"] += 1
        m = re.match(r"grab_shadow_pre_(.+?)_(无差异|差异.*?)_(\d+)$", f.stem)
        if not m:
            continue
        piece, tms = m.group(1), int(m.group(3))
        t = tms / 1000.0
        i = bisect.bisect_right(ts_list, t + 0.5) - 1
        if i < 0 or states[i]["t"] < t - 6:
            why["无对应状态"] += 1
            continue
        bgr, rgb = _read(f)
        if bgr is None:
            continue
        tier, cards, sk, names, lay = label_frame(E, rgb, states, i, why)
        if not tier:
            continue
        add(f, bgr, f"{run.name[5:]}_g{tms}", tier, cards, sk, lay, states[i], "grab_pre")
        fm = f.with_name(f.name.replace("grab_shadow_pre_", "grab_shadow_mid_"))
        held = [k for k, nm in enumerate(names) if nm == piece]
        if tier == "A" and fm.exists() and len(held) <= 1 and piece != "枫叶舞":
            bm, _ = _read(fm)
            if bm is not None:
                cm = [dict(c, awk=None) for c in cards]
                for k in held:
                    cm[k]["held"] = True
                why["tierM"] += 1
                add(fm, bm, f"{run.name[5:]}_m{tms}", "M", cm, sk, None, states[i], "grab_mid",
                    held_name=piece)
    # 空位样本
    for f in sorted(run.glob("inspect_no_view_*.png")):
        t = f.stem.split("_")[-1]
        x = nv_map.get((run.name, t))
        why["no_view帧"] += 1
        if x is None:
            why["no_view日志里找不到x"] += 1
            continue
        bgr = _imread(f)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        col = rgb[600:700, max(0, x - 8):x + 8]
        # 场地背景：亮、起伏小；卡牌下半截是深色底（均值 30~100）
        if not (float(col.mean()) > 115 and float(col.std()) < 50):
            why["no_view那一列不是背景(牌在动画里?)"] += 1
            continue
        why["tierN"] += 1
        cid = f"{run.name[5:]}_nv{t}"
        save_crop(bgr, cid)
        rows.append({"id": cid, "frame": str(f.relative_to(ROOT)).replace("\\", "/"), "run": run.name,
                     "src": "no_view", "tier": "N", "n": None, "cards": None, "skill": None, "shop_open": None, "round": None,
                     "by": None, "stacked": None, "empty_x": x})
    return rows, why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test-runs", type=int, default=60)
    ap.add_argument("--val-runs", type=int, default=60)
    ap.add_argument("--limit-runs", type=int, default=0, help="调试用：只处理最新 N 局")
    args = ap.parse_args()
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    runs = sorted((p for p in LOGS.glob("live_*") if p.is_dir()), key=lambda p: int(p.name[5:]))
    if args.limit_runs:
        runs = runs[-args.limit_runs:]
    nv = no_view_clicks()
    awk = awaken_labels()
    print(f"对局 {len(runs)}，no_view 点位 {len(nv)}，觉醒人工标签 {len(awk)}")
    nv_by = collections.defaultdict(dict)
    for (r, t), x in nv.items():
        nv_by[r][(r, t)] = x
    rows, why = [], collections.Counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (rr, w) in enumerate(ex.map(process_run, [(str(r), nv_by.get(r.name, {}), awk) for r in runs],
                                           chunksize=4)):
            rows += rr
            why.update(w)
            if k % 50 == 0:
                print(f"  {k}/{len(runs)} 样本 {len(rows)}", flush=True)
    # 按对局时间划分：只看产出了样本的对局
    used = sorted({r["run"] for r in rows}, key=lambda s: int(s[5:]))
    test = set(used[-args.test_runs:])
    val = set(used[-args.test_runs - args.val_runs:-args.test_runs])
    for r in rows:
        r["split"] = "test" if r["run"] in test else "val" if r["run"] in val else "train"
    (OUT / "labels.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                      encoding="utf-8")
    st = {"filter": dict(why), "splits": {}}
    for sp in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == sp]
        lab = [r for r in rs if r["tier"] in ("A", "B", "M", "Z")]
        st["splits"][sp] = {
            "runs": len({r["run"] for r in rs}), "samples": len(rs),
            "tier": dict(collections.Counter(r["tier"] for r in rs)),
            "n_dist": dict(sorted(collections.Counter(r["n"] for r in lab).items())),
            "stacked": sum(1 for r in lab if r["stacked"]), "not_stacked": sum(1 for r in lab if not r["stacked"]),
            "shop_open": sum(1 for r in lab if r["shop_open"]),
            "with_skill": sum(1 for r in lab if r["skill"]),
            "awk_pos": sum(1 for r in lab for c in r["cards"] if c.get("awk") == 1),
            "awk_neg": sum(1 for r in lab for c in r["cards"] if c.get("awk") == 0),
        }
    st["test_runs"] = sorted(test, key=lambda s: int(s[5:]))
    (OUT / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(st["filter"], ensure_ascii=False))
    for sp, v in st["splits"].items():
        print(sp, json.dumps(v, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
