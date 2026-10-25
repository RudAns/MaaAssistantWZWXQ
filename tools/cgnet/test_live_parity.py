"""cgnet 线上 / 训练 一致性测试（carry / swap 题）。

对 r9m 里每道**真实**（非增强）的 carry / swap 题：
  1. 线上构造路径：用 tools/laya/build_dataset.extract_run 从原始日志重新出题（与 LayaPolicy 线上调用的是同一个
     carry_point / swap_point），按 (题型, ts) 对上原题，比对 options / raw / ctx / board / field；
     "已得天赋"用 cgnet_policy.TalentTail 逐帧回放到出题时刻；再走 CGNetPolicy.features() → collate_np()。
  2. 训练构造路径：data/cgnet/<ds>/samples.pkl 里同一 id 的样本 → train.collate()。
  3. 比对：输入张量逐元素相等（列出不一致的字段）；输出概率 = 线上 ONNX（多种子平均）vs 训练路径 PyTorch（同样平均），最大绝对差。

  python tools/cgnet/test_live_parity.py            # 默认读 data/cgnet/deploy/manifest.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))     # build_dataset（线上出题模块）；cgnet 放在它前面（两边都有 train.py）
sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
from agent.decision.cgnet_policy import CGNetModel, TalentTail, collate_np  # noqa: E402
from train import collate  # noqa: E402
from export_deploy import load_model  # noqa: E402

ROOT = PROJECT_DIR / "data" / "cgnet"
FIELDS = ["tt", "ot", "node", "text", "num", "glob", "opt_pos", "rel", "relv"]
PT_FIELDS = ("options", "raw", "ctx", "board", "field")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default=str(ROOT / "deploy"))
    ap.add_argument("--points_ds", default="r9m")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "parity_report.json"))
    a = ap.parse_args()
    pol = CGNetModel(Path(a.deploy))
    man = pol.manifest
    train_s = pickle.loads((ROOT / man["ds"] / "samples.pkl").read_bytes())
    if (man.get("v2ablate") or "none") != "none":
        from train_v2 import apply                  # noqa: E402  —— 训练路径的消融实现（与线上 apply_ablation 独立，交叉校验）
        train_s = apply(train_s, json.loads((ROOT / man["ds"] / "meta.json").read_text(encoding="utf-8")), man["v2ablate"])
    train_s = {s["id"]: s for s in train_s}
    models = [load_model(man["run"], sd)[0] for sd in man["seeds"]]
    import build_dataset as B                      # noqa: E402  —— 线上 LayaPolicy 用的同一个模块
    pts = [json.loads(l) for l in (PROJECT_DIR / "data/laya/datasets" / a.points_ds / "points.jsonl").open(encoding="utf-8")]
    pts = [p for p in pts if p["kind"] in pol.kinds and p.get("source") != "aug"]
    by_run = collections.defaultdict(list)
    for p in pts:
        by_run[p["run"]].append(p)
    stats = collections.Counter()
    mism = collections.Counter()
    pt_mism = collections.Counter()
    examples = []
    maxdiff, argmax_agree, n_cmp = 0.0, 0, 0
    lat = []
    done = 0
    for run, plist in sorted(by_run.items()):
        run_dir = PROJECT_DIR / "data" / "logs" / run
        try:
            regen = B.extract_run(run_dir)
        except Exception as e:                                     # noqa: BLE001
            stats["extract_error"] += len(plist)
            examples.append({"run": run, "error": repr(e)[:200]})
            continue
        idx = {(q["kind"], round(float(q.get("ts") or 0), 4)): q for q in regen}
        frames = [json.loads(l) for l in (run_dir / "states.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for p in plist:
            q = idx.get((p["kind"], round(float(p["ts"]), 4)))
            if q is None:
                # 当前代码在这一帧不再出这道题（出题条件后来改过 / 老数据来源不同）→ 只能检验"特征构造"这一段：
                # 用存档题走线上 features()，与训练样本比
                stats["not_regenerated"] += 1
                tail = TalentTail(None)
                for x in frames:
                    if float(x.get("ts") or 0) >= float(p["ts"]):
                        break
                    tail.feed(x)
                s_on = pol.features(dict(p), owned_talents=list(tail.owned))
                s_tr = train_s.get(p["id"])
                ok = s_tr is not None and all(np.asarray(s_on.get(f)).shape == np.asarray(s_tr.get(f)).shape and
                                              np.array_equal(np.asarray(s_on.get(f)), np.asarray(s_tr.get(f))) for f in FIELDS)
                stats["stored_point_features_exact" if ok else "stored_point_features_mismatch"] += 1
                continue
            stats["matched"] += 1
            for k in PT_FIELDS:
                if json.dumps(q.get(k), ensure_ascii=False, sort_keys=True, default=str) != \
                        json.dumps(p.get(k), ensure_ascii=False, sort_keys=True, default=str):
                    pt_mism[k] += 1
            # 线上路径用**重新出的题**；若选项对不上原题（老数据被重渲染过），标签没法对齐 → 改用原题的结构化字段
            use = q if q.get("options") == p.get("options") else p
            if use is p:
                stats["used_stored_point"] += 1
            tail = TalentTail(None)
            for x in frames:
                if float(x.get("ts") or 0) >= float(p["ts"]):
                    break
                tail.feed(x)
            import time as _t
            _t0 = _t.perf_counter()
            _pr = pol.probs(dict(use, id=p["id"], split=p["split"]), owned_talents=list(tail.owned))   # 线上一次完整推理的耗时
            lat.append((_t.perf_counter() - _t0) * 1000)
            s_on = pol.features(dict(use, id=p["id"], split=p["split"]), owned_talents=list(tail.owned))
            # B：把"原题存档时还没有的 ctx 字段"（主C 来历 carry_since/carry_prev/carry_switches 等）对齐成原题的值，
            # 检查除这些已知差异外是否逐字节一致
            ctx_b = dict(use.get("ctx") or {})
            for k in set(ctx_b) | set(p.get("ctx") or {}):
                if ctx_b.get(k) != (p.get("ctx") or {}).get(k):
                    ctx_b[k] = (p.get("ctx") or {}).get(k)
            s_on_b = pol.features(dict(use, ctx=ctx_b, id=p["id"], split=p["split"]), owned_talents=list(tail.owned))
            s_tr = train_s.get(p["id"])
            if s_tr is None:
                stats["no_train_sample"] += 1
                continue
            bad = []
            for f in FIELDS:
                x1, x2 = np.asarray(s_on.get(f)), np.asarray(s_tr.get(f))
                if x1.shape != x2.shape or not np.array_equal(x1, x2):
                    bad.append(f)
                    mism[f] += 1
            if list(s_on["keys"]) != list(s_tr["keys"]):
                bad.append("keys")
                mism["keys"] += 1
            stats["exact" if not bad else "input_mismatch"] += 1
            bad_b = [f for f in FIELDS if s_tr is not None and (np.asarray(s_on_b.get(f)).shape != np.asarray(s_tr.get(f)).shape
                                                                 or not np.array_equal(np.asarray(s_on_b.get(f)), np.asarray(s_tr.get(f))))]
            stats["B_exact_after_ctx_align" if not bad_b else "B_still_mismatch"] += 1
            for f in bad_b:
                mism["B:" + f] += 1
            if bad and len(examples) < 15:
                examples.append({"id": p["id"], "kind": p["kind"], "run": run, "fields": bad,
                                 "names_on": s_on["names"][:40], "names_tr": s_tr["names"][:40]})
            # 概率
            def onnx_p(ss):
                xo = collate_np(ss, pol.n_ch)
                out = []
                for sess in pol.sessions:
                    z = sess.run(None, xo)[0][0].astype(np.float64)
                    e = np.exp(z - z.max())
                    out.append(e / e.sum())
                return np.mean(out, 0)
            p_on = onnx_p(s_on)
            p_on_b = onnx_p(s_on_b)
            xt, _ = collate([s_tr], pol.n_ch, torch.device("cpu"))
            with torch.no_grad():
                ps_tr = [torch.softmax(m(**xt)[0].double(), -1).numpy()[:len(s_tr["keys"])] for m in models]
            p_tr = np.mean(ps_tr, 0)
            if p_on.shape == p_tr.shape:
                n_cmp += 1
                maxdiff = max(maxdiff, float(np.abs(p_on - p_tr).max()))
                argmax_agree += int(p_on.argmax() == p_tr.argmax())
                stats["B_prob_max_abs_diff_x1e6"] = max(stats.get("B_prob_max_abs_diff_x1e6", 0), int(1e6 * float(np.abs(p_on_b - p_tr).max())))
                stats["B_argmax_agree"] += int(p_on_b.argmax() == p_tr.argmax())
            done += 1
            if a.limit and done >= a.limit:
                break
        if a.limit and done >= a.limit:
            break
    # C：全部真实题（含对不上号的）用存档题 + 训练时的"已得天赋"快照（build_data.replay_run/snapshot）走线上 features()，
    #    检验特征构造本身是否与训练逐元素一致（排除出题代码变化与天赋回放时刻两个已知差异）
    import build_data as BDm                     # noqa: E402
    snaps: dict = {}
    for p in pts:
        if p["run"] not in snaps:
            snaps[p["run"]] = BDm.replay_run(p["run"])
        ow, _ = BDm.snapshot(snaps[p["run"]], float(p["ts"]))
        s_c = pol.features(dict(p), owned_talents=list(ow))
        s_t = train_s.get(p["id"])
        ok = s_t is not None and all(np.asarray(s_c.get(f)).shape == np.asarray(s_t.get(f)).shape and
                                     np.array_equal(np.asarray(s_c.get(f)), np.asarray(s_t.get(f))) for f in FIELDS)
        stats["C_stored_point_train_talents_exact" if ok else "C_mismatch"] += 1
    rep = {"deploy": man, "stats": dict(stats), "input_field_mismatch": dict(mism),
           "question_field_mismatch_vs_stored": dict(pt_mism),
           "online_probs_latency_ms": ({"p50": float(np.percentile(lat, 50)), "p90": float(np.percentile(lat, 90)),
                                        "max": float(np.max(lat)), "n": len(lat)} if lat else None),
           "prob_compared": n_cmp, "prob_max_abs_diff": maxdiff, "argmax_agree": argmax_agree, "examples": examples}
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("examples", "deploy")}, ensure_ascii=False, indent=1))
    for e in examples[:5]:
        print(json.dumps(e, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
