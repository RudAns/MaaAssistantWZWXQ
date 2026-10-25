"""Laya 蒸馏训练 + 评估（老师 = Claude 标注的软标签，学生 = Laya multilingual）。

用法：
  # 零样本基线（不训练，只评估）
  python tools/laya/train.py --run S0_zeroshot --epochs 0
  # 只训决策头（编码器冻结）
  python tools/laya/train.py --run S1_head --stage head --epochs 8 --lr 3e-4
  # 解冻编码器上层
  python tools/laya/train.py --run S2_top4 --stage top4 --epochs 4 --lr 2e-5 --init data/laya/runs/S1_head/checkpoint

每个 run 写到 data/laya/runs/<run>/：
  config.json     全部超参 + 数据版本 + 标注版本 + 随机种子
  curves.jsonl    每个 epoch：train loss / val 指标
  metrics.json    最终 val/test 指标（总体 + 按决策类型）
  preds_test.jsonl  每道测试题：老师分布、模型分布、规则选择（仅参考）
  checkpoint/     laya 格式，`laya.load(path)` 直接能用

指标（全部以**老师（Claude）标签**为准；规则引擎一致率只作参考，不作为验证依据）：
  top1     模型 argmax == 老师 best
  tmass    老师分布在模型所选项上的概率质量（选了"次优但也合理"的项能拿部分分）
  ce       交叉熵 H(老师, 模型)，越低越好；kl = ce - H(老师)
  ece      模型置信度（max p）相对 top1 的校准误差
  rule_top1  模型 argmax == 规则引擎当时的选择（参考）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))
import features as F  # noqa: E402

from laya.common import QTYPES, build_model, build_sequence, collate_items, ece_score, render_options  # noqa: E402

BASE_CKPT = Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots/" \
    "1c5edc17a7acd8701df6fc341c0d179f1c62c982/multilingual"
KINDS = ("shop", "talent", "card_pick", "lord_boost")


# ---------------------------------------------------------------- 数据

def load_data(ver: str, labels_path: Path):
    pts = {}
    for line in (PROJECT_DIR / "data" / "laya" / "datasets" / ver / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    rows = []
    for line in labels_path.open(encoding="utf-8"):
        lab = json.loads(line)
        p = pts.get(lab["id"])
        if p is None:
            continue
        rows.append({**p, "teacher": lab["dist"], "t_best": lab["best"], "t_conf": lab.get("confidence", 1.0)})
    return rows


SEQ_CACHE = None      # --pretok 时设为 seqcache.SeqCache：同一道题只分词一次（与 build_sequence 逐 token 等价）


def make_item(tok, p: dict, cfg: dict, order: list[int] | None = None) -> dict:
    q = F.to_internal(p["question"])
    keys = list(q["crit"].keys())
    if SEQ_CACHE is not None:
        ids, markers = SEQ_CACHE.build(p["id"], p["state"], q, option_order=order)
    else:
        ids, markers = build_sequence(tok, p["state"], q, cfg["max_len"], cfg["head_max_len"], option_order=order)
    if len(markers) != len(keys):
        raise ValueError(f"{p['id']}: markers {len(markers)} != options {len(keys)}")
    order = order or list(range(len(keys)))
    tgt = np.array([float(p["teacher"].get(keys[i], 0.0)) for i in order])
    tgt = tgt / tgt.sum() if tgt.sum() > 0 else np.full(len(keys), 1 / len(keys))
    return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "target": tgt.tolist(),
            "id": p["id"], "kind": p["kind"], "keys": [keys[i] for i in order]}


# ---------------------------------------------------------------- 模型

def load_model(ckpt: Path, device):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"))
    model = build_model(cfg, encoder_dir=str(ckpt / "encoder"))
    model.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    return model.to(device), tok, cfg


def save_checkpoint(model, tok, cfg: dict, src: Path, dst: Path, train_info: dict):
    from safetensors.torch import save_file
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("tokenizer", "encoder"):
        if (dst / sub).exists():
            shutil.rmtree(dst / sub)
        shutil.copytree(src / sub, dst / sub)
    sd = {k: v.detach().to("cpu").to(torch.bfloat16 if v.is_floating_point() else v.dtype).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, str(dst / "model.safetensors"))
    c = dict(cfg)
    c["training"] = {**(cfg.get("training") or {}), "wzwxq_distill": train_info}
    (dst / "rl_agent_config.json").write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")


def set_trainable(model, stage: str):
    """head：只训决策头（type_emb/head/scorer）；topN：再解冻编码器最后 N 层 + final_norm；full：除词嵌入外全部。"""
    for p in model.parameters():
        p.requires_grad = False
    for mod in (model.type_emb, model.head, model.scorer):
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
    if stage.startswith("top") or stage == "full":
        layers = model.encoder.layers
        n = len(layers) if stage == "full" else int(stage[3:])
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for name, p in model.encoder.named_parameters():
            if "final_norm" in name:
                p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- 评估

@torch.no_grad()
def predict(model, tok, cfg, rows, device, bs=8, temp=1.0):
    model.eval()
    out = []
    for b in range(0, len(rows), bs):
        items = [make_item(tok, p, cfg) for p in rows[b:b + bs]]
        batch = collate_items([items], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                              batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                              batch["qtype"].to(device))
        logits = logits.float().cpu().numpy()
        for r, it in enumerate(items):
            k = len(it["markers"])
            z = logits[r, :k] / temp
            pr = np.exp(z - z.max())
            pr /= pr.sum()
            out.append({"id": it["id"], "kind": it["kind"], "keys": it["keys"], "p": pr.tolist(),
                        "logits": logits[r, :k].tolist()})
    return out


def metrics(rows, preds) -> dict:
    by_id = {p["id"]: p for p in rows}
    agg: dict = {}
    for pr in preds:
        p = by_id[pr["id"]]
        keys, q = pr["keys"], np.array(pr["p"])
        t = np.array([p["teacher"].get(k, 0.0) for k in keys])
        t = t / t.sum()
        a = int(q.argmax())
        # 等价选项：选项文本完全相同（商店里两张同名牌 = 同一个决策）。老师只能在它们之间随手挑一个当 best，
        # 所以另报"等价 top1 / 等价 tmass"：选中与老师 best 同文本的任一项都算对，tmass 合并等价项的质量。
        crit = p["question"]["criteria"]
        same = [i for i, k in enumerate(keys) if crit.get(k) == crit.get(keys[a])]
        rec = {"top1": float(keys[a] == p["t_best"]), "tmass": float(t[a]),
               "top1_eq": float(crit.get(keys[a]) == crit.get(p["t_best"])), "tmass_eq": float(t[same].sum()),
               "ce": float(-(t * np.log(np.clip(q, 1e-12, 1))).sum()),
               "h_t": float(-(t * np.log(np.clip(t, 1e-12, 1))).sum()),
               "conf": float(q.max()), "rule": float(keys[a] == p.get("rule_choice")),
               "rule_vs_teacher": float(p.get("rule_choice") == p["t_best"]),
               "rand_tmass": float(t.mean()), "n": 1}
        for g in ("all", p["kind"]):
            d = agg.setdefault(g, {"_conf": [], "_top1": []})
            for k, v in rec.items():
                d[k] = d.get(k, 0.0) + v
            d["_conf"].append(rec["conf"])
            d["_top1"].append(rec["top1"])
    res = {}
    for g, d in agg.items():
        n = d["n"]
        res[g] = {"n": int(n), "top1": d["top1"] / n, "tmass": d["tmass"] / n, "ce": d["ce"] / n,
                  "top1_eq": d["top1_eq"] / n, "tmass_eq": d["tmass_eq"] / n,
                  "kl": (d["ce"] - d["h_t"]) / n, "ece": ece_score(np.array(d["_conf"]), np.array(d["_top1"])),
                  "rule_top1_ref": d["rule"] / n, "rule_vs_teacher_ref": d["rule_vs_teacher"] / n,
                  "random_tmass": d["rand_tmass"] / n}
    return res


def fit_temperature(preds, rows) -> float:
    """在 val 上拟合单一温度（最小化对老师分布的交叉熵），只用于校准，不改 argmax。"""
    by_id = {p["id"]: p for p in rows}
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(math.log(0.5), math.log(5.0), 40)):
        ce = 0.0
        for pr in preds:
            t = np.array([by_id[pr["id"]]["teacher"].get(k, 0.0) for k in pr["keys"]])
            t /= t.sum()
            z = np.array(pr["logits"]) / T
            q = np.exp(z - z.max())
            q /= q.sum()
            ce -= (t * np.log(np.clip(q, 1e-12, 1))).sum()
        best = min(best, (ce, float(T)))
    return best[1]


# ---------------------------------------------------------------- 训练

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ver", default="v2")
    ap.add_argument("--labels", default="data/laya/labels/v2/labels.jsonl")
    ap.add_argument("--init", default=str(BASE_CKPT))
    ap.add_argument("--stage", default="head", help="head | top2 | top4 | top8 | full")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4, help="编码器（解冻部分）的学习率；只训头时即头的学习率")
    ap.add_argument("--head-lr", type=float, default=None, help="决策头学习率（默认同 --lr）")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--shuffle-opts", type=float, default=0.5, help="每条样本打乱选项顺序的概率（去位置偏置）")
    ap.add_argument("--w-sph", type=float, default=0.5, help="proper_reward 里球面分的权重（0=纯交叉熵）")
    ap.add_argument("--conf-weight", action="store_true", help="按老师 confidence 加权样本")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--train-splits", default="train")
    ap.add_argument("--cv", type=int, default=0, help="K 折交叉验证（按对局分折）；0 = 用数据集自带的 split")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--no-save", action="store_true", help="不写 checkpoint（交叉验证时省磁盘）")
    ap.add_argument("--pretok", action="store_true", help="预分词缓存（结果不变，只省时间；见 seqcache.py --check）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=0, help="序列上限（0 = f6 数据集用 2048，否则沿用 checkpoint 配置）")
    ap.add_argument("--head-max-len", type=int, default=0, help="题目+选项上限（0 = f6 用 512，否则沿用配置）")
    ap.add_argument("--notes", default="")
    ap.add_argument("--upsample-ids", default="", help="json 题号清单：这些训练题重复 --upsample 遍（本轮重标的题只占 ~5%，不加权学不进去，2026-10-15 R30）")
    ap.add_argument("--upsample", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = PROJECT_DIR / "data" / "laya" / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    kinds = set(args.kinds.split(","))
    rows = [r for r in load_data(args.ver, PROJECT_DIR / args.labels) if r["kind"] in kinds]
    if args.cv > 0:
        # 按对局做 K 折：test = 第 fold 折，val = 下一折，其余训练。同一局的决策点不会跨折（防泄漏）。
        # 早期数据的对局编号是 int，新数据（build_round 并入的实战局）是 str：数字在前按数值、字符串在后按字母，
        # 全是 int 的旧数据集排序与原来完全一致（旧实验的折划分可复现）
        ms = sorted({r["match"] for r in rows},
                    key=lambda m: (0, m, "") if isinstance(m, (int, float)) else (1, 0, str(m)))
        random.Random(1234).shuffle(ms)
        fold_of = {m: i % args.cv for i, m in enumerate(ms)}
        for r in rows:
            f = fold_of[r["match"]]
            r["split"] = "test" if f == args.fold else "val" if f == (args.fold + 1) % args.cv else "train"
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    train_rows = [r for r in rows if r["split"] in args.train_splits.split(",")]
    if args.upsample_ids and args.upsample > 1:
        _up = set(json.loads(Path(args.upsample_ids).read_text(encoding="utf-8")))
        _extra = [r for r in train_rows if r["id"] in _up]
        train_rows = train_rows + _extra * (args.upsample - 1)
        print(f"upsample: {len(_extra)} 题 × {args.upsample}（训练集 {len(train_rows)} 条）")
    init = Path(args.init)
    model, tok, cfg = load_model(init, device)
    # f6（question-design-v2.md）：局面中位 ~1000、最长 ~1700 token，原配置 1024/256 会截掉一半商店题。
    # 编码器 mmBERT 位置上限 8192；这两个数写进保存的 rl_agent_config.json，推理端（laya.Agent）读同一份 → 训练/推理一致。
    fv = rows[0].get("feature_version") if rows else None
    cfg["max_len"] = args.max_len or (2048 if (fv or "") >= "f6" else cfg["max_len"])
    cfg["head_max_len"] = args.head_max_len or (512 if (fv or "") >= "f6" else cfg["head_max_len"])
    print(f"seq limits: max_len={cfg['max_len']} head_max_len={cfg['head_max_len']} (features {fv})")
    if args.pretok:
        global SEQ_CACHE
        from seqcache import SeqCache
        SEQ_CACHE = SeqCache(tok, cfg["max_len"], cfg["head_max_len"])
    n_train = set_trainable(model, args.stage) if args.epochs > 0 else 0

    config = {**vars(args), "device": str(device), "n_trainable": n_train,
              "n": {s: len(v) for s, v in splits.items()}, "n_train_rows": len(train_rows),
              "feature_version": (rows[0].get("feature_version") if rows else None), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: config[k] for k in ("run", "stage", "n", "n_trainable")}, ensure_ascii=False))

    curves = (run_dir / "curves.jsonl").open("w", encoding="utf-8")
    val0 = metrics(splits["val"], predict(model, tok, cfg, splits["val"], device))
    curves.write(json.dumps({"epoch": 0, "val": val0["all"]}) + "\n")
    curves.flush()
    print(f"epoch 0 val top1={val0['all']['top1']:.3f} tmass={val0['all']['tmass']:.3f} ce={val0['all']['ce']:.3f}")
    best = (val0["all"]["ce"], 0)
    best_state = None

    if args.epochs > 0:
        params = [p for p in model.parameters() if p.requires_grad]
        enc_ids = {id(p) for p in model.encoder.parameters()}
        groups = [{"params": [p for p in params if id(p) not in enc_ids], "lr": args.head_lr or args.lr},
                  {"params": [p for p in params if id(p) in enc_ids], "lr": args.lr}]
        opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=args.wd)
        steps = math.ceil(len(train_rows) / args.bs / args.accum) * args.epochs
        warm = max(1, int(0.06 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
        detach = args.stage == "head"
        for ep in range(1, args.epochs + 1):
            model.train()
            if detach:
                model.encoder.eval()     # 冻结编码器时关掉它的 dropout，特征稳定
            random.shuffle(train_rows)
            t0, tot, nb = time.time(), 0.0, 0
            opt.zero_grad()
            for b in range(0, len(train_rows), args.bs):
                items = []
                for p in train_rows[b:b + args.bs]:
                    k = len(p["options"])
                    order = random.sample(range(k), k) if random.random() < args.shuffle_opts else None
                    items.append(make_item(tok, p, cfg, order))
                batch = collate_items([items], tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                      batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                                      batch["qtype"].to(device), detach_encoder=detach)
                mask = batch["marker_mask"].to(device)
                tgt = batch["target"].to(device)
                logq = torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)
                q = logq.exp() * mask
                ce = -(tgt * logq.masked_fill(~mask, 0)).sum(-1)
                sph = (tgt * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
                loss_i = ce - args.w_sph * sph
                if args.conf_weight:
                    w = torch.tensor([rw["t_conf"] for rw in train_rows[b:b + args.bs]], device=device)
                    loss = (loss_i * w).sum() / w.sum()
                else:
                    loss = loss_i.mean()
                (loss / args.accum).backward()
                nb += 1
                tot += loss.item()
                if nb % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
            if nb % args.accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            vm = metrics(splits["val"], predict(model, tok, cfg, splits["val"], device))
            rec = {"epoch": ep, "train_loss": tot / max(1, nb), "secs": round(time.time() - t0, 1),
                   "lr": sched.get_last_lr()[0], "val": vm["all"], "val_by_kind": {k: vm[k] for k in vm if k != "all"}}
            curves.write(json.dumps(rec) + "\n")
            curves.flush()
            print(f"epoch {ep} loss={rec['train_loss']:.4f} val top1={vm['all']['top1']:.3f} "
                  f"tmass={vm['all']['tmass']:.3f} ce={vm['all']['ce']:.3f} ({rec['secs']}s)", flush=True)
            if vm["all"]["ce"] < best[0]:
                best = (vm["all"]["ce"], ep)
                best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()
                              if not k.startswith("encoder.") or not detach}
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
    curves.close()

    # 最终：val 拟合温度，test 报告（温度前/后）
    vpred = predict(model, tok, cfg, splits["val"], device)
    T = fit_temperature(vpred, splits["val"])
    res = {"best_epoch": best[1], "temperature": T,
           "val": metrics(splits["val"], vpred),
           "test": metrics(splits["test"], predict(model, tok, cfg, splits["test"], device)),
           "test_T": metrics(splits["test"], tpred := predict(model, tok, cfg, splits["test"], device, temp=T))}
    (run_dir / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    by_id = {p["id"]: p for p in splits["test"]}
    with (run_dir / "preds_test.jsonl").open("w", encoding="utf-8") as fh:
        for pr in tpred:
            p = by_id[pr["id"]]
            fh.write(json.dumps({"id": pr["id"], "kind": pr["kind"], "keys": pr["keys"],
                                 "model": [round(x, 4) for x in pr["p"]],
                                 "teacher": [round(p["teacher"].get(k, 0), 4) for k in pr["keys"]],
                                 "t_best": p["t_best"], "rule_ref": p.get("rule_choice")}, ensure_ascii=False) + "\n")
    t = res["test"].get("all")
    if not t:      # 2026-10-15 云端试训：子集没有测试题时这里 KeyError，checkpoint 也就没存
        print("TEST: 没有测试题（跳过）")
    else:
      print(f"TEST top1={t['top1']:.3f} tmass={t['tmass']:.3f} ce={t['ce']:.3f} ece={t['ece']:.3f} "
          f"(random tmass={t['random_tmass']:.3f}, rule-vs-teacher={t['rule_vs_teacher_ref']:.3f}) T={T:.2f}")
    if args.epochs > 0 and not args.no_save:
        cfg2 = dict(cfg)
        cfg2["temperature"] = [T, T, T]
        save_checkpoint(model, tok, cfg2, init, run_dir / "checkpoint",
                        {"run": args.run, "stage": args.stage, "best_epoch": best[1], "labels": args.labels,
                         "dataset": args.ver, "feature_version": config["feature_version"], "test": res["test"].get("all")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
