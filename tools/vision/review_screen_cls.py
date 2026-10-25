"""界面分类数据集的人工复核流程（val/test 逐张看图定标签）。

  # 1. 生成复核拼板：按初始模型预测类别排序，每张拼板 25 张缩略图，左上角 = 序号 + 预测类别缩写
  .venv/Scripts/python.exe -X utf8 tools/vision/review_screen_cls.py make --set test --preds data/vision/runs/screen_cls_s1/preds.jsonl
  # 2. 人看拼板，把"不是预测类别"的格子写进 review_notes/<set>.txt，格式（每张拼板一行，没写的格子 = 认可预测类别）：
  #      <拼板号>: <格子号>=<真实类别> <格子号>=<真实类别> ...       例：  12: 3=other 17=board_idle
  #      <拼板号>: all=<类别>                                          整板改成同一类
  # 3. 应用：写入 labels_review.jsonl（id, label, split, by="manual", pred_shown）
  .venv/Scripts/python.exe -X utf8 tools/vision/review_screen_cls.py apply --set test

复核集：test 全部；val 每个 (目录, 弱标签) 抽一部分（大类 3~6 张、小类全要，见 VAL_CAP），控制工作量；
没进复核集的 val 样本不参加训练也不参加评测。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_cls_sheet import sheet  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "vision" / "screen_cls_v1"
NOTES = DS / "review_notes"
PER = 25
VAL_CAP = {"unknown": 4, "prepare": 3, "battle": 3, "prepare_shop": 3, "auction": 3, "talent": 4, "card_pick": 6}
ABBR = {"prepare": "PRE", "prepare_shop": "SHOP", "board_idle": "IDLE", "battle": "BAT", "auction": "AUC",
        "talent": "TAL", "card_pick": "PICK", "lord_boost": "BOOST", "hero_pick": "HPICK", "loading": "LOAD",
        "settle": "SET", "settle_final": "FIN", "post_match": "POST", "lobby": "LOBBY", "mode_lobby": "MODE", "match_found": "MATCH",
        "settings": "SETTINGS", "lord_panel": "LPANEL", "hero_panel": "HPANEL", "enlarged_detail": "ENL",
        "notice": "NOTICE", "equip_popup": "EQUIP", "reward_popup": "REWARD", "transition": "TRANS", "solo_popup": "SOLO", "other": "OTHER"}


def samples():
    return [json.loads(l) for l in (DS / "samples.jsonl").read_text(encoding="utf-8").splitlines()]


def review_set(ss, which):
    if which == "test":
        return [s for s in ss if s["split"] == "test"]
    if which == "val":
        rng = random.Random(0)
        g = defaultdict(list)
        for s in ss:
            if s["split"] == "val":
                g[(s["run"], s["weak"])].append(s)
        out = []
        for (run, weak), v in sorted(g.items()):
            cap = VAL_CAP.get(weak, 10 ** 6)
            out += v if len(v) <= cap else rng.sample(v, cap)
        return out
    raise SystemExit(f"未知集合 {which}")


def cmd_make(a):
    ss = review_set(samples(), a.set)
    preds = {r["id"]: r for r in map(json.loads, Path(a.preds).read_text(encoding="utf-8").splitlines())}
    order = list(ABBR)
    ss.sort(key=lambda s: (order.index(preds[s["id"]]["pred"]), s["weak"], -preds[s["id"]]["conf"], s["id"]))
    outdir = NOTES / f"sheets_{a.set}"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k in range(0, len(ss), PER):
        chunk = ss[k:k + PER]
        shown = [preds[s["id"]]["pred"] for s in chunk]
        im = sheet([ROOT / s["path"] for s in chunk], [ABBR[p] for p in shown], 5, 240)
        cv2.imencode(".png", im)[1].tofile(str(outdir / f"{k // PER:03d}.png"))
        manifest.append({"sheet": k // PER, "ids": [s["id"] for s in chunk], "shown": shown})
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"{a.set}: {len(ss)} 张 -> {len(manifest)} 张拼板，{outdir}")


def cmd_apply(a):
    man = json.loads((NOTES / f"sheets_{a.set}" / "manifest.json").read_text(encoding="utf-8"))
    notes = {}
    for line in (NOTES / f"{a.set}.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        k, rest = line.split(":", 1)
        notes[int(k)] = rest.split()
    split = {s["id"]: s["split"] for s in samples()}
    rows, changed, missing = [], 0, []
    for m in man:
        if m["sheet"] not in notes:
            missing.append(m["sheet"])
            continue
        labels = list(m["shown"])
        for tok in notes[m["sheet"]]:
            if tok == "ok":
                continue
            i, lab = tok.split("=")
            assert lab in ABBR or lab == "drop", f"拼板 {m['sheet']} 未知类别 {lab}"
            if i == "all":
                labels = [lab] * len(labels)
            else:
                labels[int(i)] = lab
        for sid, lab, sh in zip(m["ids"], labels, m["shown"]):
            rows.append({"id": sid, "label": lab, "split": split[sid], "by": "manual", "pred_shown": sh})
            changed += lab != sh
    # 按 id 的补充更正（二次核对时发现的，见 review_notes/fixes_<set>.json：{id: 类别}）
    fx = NOTES / f"fixes_{a.set}.json"
    if fx.exists():
        fixes = json.loads(fx.read_text(encoding="utf-8"))
        for r in rows:
            if r["id"] in fixes:
                r["label"], r["by"] = fixes[r["id"]], "manual_fix"
    changed = sum(r["label"] != r["pred_shown"] for r in rows)
    if missing:
        print(f"⚠ 还没复核的拼板: {missing}")
    p = DS / "labels_review.jsonl"
    old = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    keep = [r for r in old if r["split"] != a.set]
    with open(p, "w", encoding="utf-8") as f:
        for r in keep + rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.set}: 写入 {len(rows)} 条（改预测 {changed} 条），labels_review.jsonl 共 {len(keep) + len(rows)} 条")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--set", required=True); m.add_argument("--preds", required=True)
    p = sub.add_parser("apply"); p.add_argument("--set", required=True)
    a = ap.parse_args()
    {"make": cmd_make, "apply": cmd_apply}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
