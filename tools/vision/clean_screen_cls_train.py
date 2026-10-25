"""界面分类训练集清洗（第二轮）：用中间模型 + 固定 UI 标记给训练集改标，写 labels_train_auto.jsonl。

  .venv/Scripts/python.exe -X utf8 tools/vision/clean_screen_cls_train.py --preds data/vision/runs/screen_cls_s2/preds.jsonl

规则（在 val/test 人工复核 + 训练集分组抽查之后定的）：
  1. 棋盘类（弱标签或模型预测落在 prepare / battle / board_idle / unknown 里的棋盘帧）按 UI 标记定子类：
       「战斗统计」模板 ≥0.5 -> battle；「准备/开战」模板 ≥0.5 -> prepare；
       两者都没有：模型 ≥0.9 且预测 board_idle / battle -> 用模型；否则丢弃（不进训练）。
     （UI 标记在复核集上：有「准备/开战」的 = prepare 几乎无例外；「战斗统计」被别的面板挡住时认不出，所以没标记不等于不是 battle）
  2. 其余样本：模型置信 ≥0.9 且与弱标签不同 -> 采用模型预测。
     例外：弱标签 unknown、模型说 card_pick 的一组，抽查 5 张全是棋盘帧（选完卡后的残影），不采用、丢弃。
  3. 弱标签 unknown 且模型置信 <0.9 -> 丢弃（不进训练）。
  4. 最后一步（--feats 给了才做）：在中间模型的特征空间里找 7 个最近的**人工复核样本**（只用 val 的 fit 半，
     不碰 val-sel / test），7 个标签全一致且与当前标签不同 -> 改成它（包括救回第 1~3 步丢掉的样本）。
     这一步在 val-sel 上验证过：7/7 一致的覆盖 88%，准确 98.0%（模型本身 90%）。
     专治弱标签整组系统性错误：拍卖回合的棋盘帧被 OCR 判成 auction、拍卖奖励弹窗被判成 battle、
     点开看完已关闭的 inspect_* 帧还挂着 enlarged_detail。
抽查记录：每个 (弱标签 -> 模型预测) 组合（≥3 张的）随机看 5 张，结论写在文档里。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "vision" / "screen_cls_v1"
BOARD = {"prepare", "battle", "board_idle"}
HI = 0.9
REJECT = {("unknown", "card_pick")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--feats", default="", help="中间模型 feats.npy（与 samples.jsonl 同序）")
    a = ap.parse_args()
    S = {json.loads(l)["id"]: json.loads(l) for l in (DS / "samples.jsonl").read_text(encoding="utf-8").splitlines()}
    P = {json.loads(l)["id"]: json.loads(l) for l in Path(a.preds).read_text(encoding="utf-8").splitlines()}
    M = {json.loads(l)["id"]: json.loads(l) for l in (DS / "ui_marks.jsonl").read_text(encoding="utf-8").splitlines()}
    out, stats, cur = [], Counter(), {}
    for i, s in S.items():
        if s["split"] != "train":
            continue
        p = P[i]
        weak = "equip_popup" if s["src"] == "equip_popup" else s["weak"]
        pred, conf = p["pred"], p["conf"]
        m = M.get(i) or {}
        lab = None
        if (weak in BOARD or weak == "unknown") and (pred in BOARD or weak in BOARD and conf < HI):
            if m.get("stat", 0) >= 0.5:
                lab = "battle"
            elif m.get("ready", 0) >= 0.5 and weak != "unknown" or m.get("ready", 0) >= 0.5 and pred in BOARD:
                lab = "prepare"
            elif conf >= HI and pred in ("board_idle", "battle"):
                lab = pred
            elif weak in BOARD and pred == weak and conf >= HI:
                lab = weak
            else:
                lab = "drop"
            stats[f"棋盘:{weak}->{lab}"] += 1
        elif conf >= HI and pred != weak:
            lab = "drop" if (weak, pred) in REJECT else pred
            stats[f"改标:{weak}->{lab}"] += 1
        elif weak == "unknown":
            lab = "drop"
            stats["unknown低置信丢弃"] += 1
        cur[i] = lab if lab is not None else weak
    if a.feats:
        import sys
        import numpy as np
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_screen_cls import val_half
        order = list(S)
        F = np.load(a.feats).astype(np.float32)
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-6
        R = {json.loads(l)["id"]: json.loads(l)["label"] for l in (DS / "labels_review.jsonl").read_text(encoding="utf-8").splitlines()}
        ref = [k for k, i in enumerate(order) if S[i]["split"] == "val" and R.get(i, "drop") != "drop"
               and val_half(S[i]["match"]) == "fit"]
        RF, RL = F[ref], np.array([R[order[k]] for k in ref])
        tr = [k for k, i in enumerate(order) if S[i]["split"] == "train"]
        for b in range(0, len(tr), 2048):
            idx = tr[b:b + 2048]
            nn_ = np.argsort(-(F[idx] @ RF.T), 1)[:, :7]
            for k, oo in zip(idx, nn_):
                labs = set(RL[oo])
                if len(labs) == 1:
                    L = labs.pop()
                    i = order[k]
                    if cur[i] != L:
                        stats[f"近邻改标:{cur[i]}->{L}"] += 1
                        cur[i] = L
    for i, lab in cur.items():
        s = S[i]
        weak = "equip_popup" if s["src"] == "equip_popup" else s["weak"]
        if lab != weak:
            out.append({"id": i, "label": lab, "weak": weak, "pred": P[i]["pred"], "conf": P[i]["conf"]})
    with open(DS / "labels_train_auto.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {len(out)} 条 -> labels_train_auto.jsonl")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
