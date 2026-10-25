"""在任意划分上给已训练的学生打分（和 train.py 同一套指标；温度用 checkpoint 里存的）。
  python tools/laya/eval_split.py --ver r2 --labels data/laya/labels/r2/labels_all.jsonl --split exam_test \
      --ckpt data/laya/runs/DEPLOY_r1_top8/checkpoint data/laya/runs/DEPLOY_r2_top8/checkpoint
输出：总体 + 按决策类型 + 按考题类型（exam_type），外加规则选择的对照。结果追加到 data/laya/runs/eval_split.jsonl。
"""
import argparse, collections, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "laya"))
import torch  # noqa: E402
import train as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ver", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--split", default="exam_test")
ap.add_argument("--ckpt", nargs="+", required=True)
args = ap.parse_args()
rows = [r for r in T.load_data(args.ver, ROOT / args.labels) if r["split"] == args.split]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{args.split}: {len(rows)} 题  类型 {dict(collections.Counter(r['kind'] for r in rows))}  "
      f"考题类型 {dict(collections.Counter(r.get('exam_type', '学生题') for r in rows))}")
rule_ok = [r["teacher"].get(r.get("rule_choice") or "", 0.0) for r in rows if r.get("rule_choice")]
print(f"  规则对照：tmass {sum(rule_ok)/max(1,len(rule_ok)):.3f}（{len(rule_ok)} 题有规则选择）")
out = (ROOT / "data" / "laya" / "runs" / "eval_split.jsonl").open("a", encoding="utf-8")
for ck in args.ckpt:
    model, tok, cfg = T.load_model(Path(ck), dev)
    temp = (cfg.get("temperature") or [1.0])[0]
    preds = T.predict(model, tok, cfg, rows, dev, temp=temp)
    m = T.metrics(rows, preds)
    by_id = {p["id"]: p for p in preds}
    et = collections.defaultdict(list)
    for r in rows:
        et[r.get("exam_type", "学生题")].append(r)
    et_m = {k: T.metrics(v, [by_id[x["id"]] for x in v])["all"] for k, v in et.items()}
    a = m["all"]
    print(f"  {Path(ck).parent.name}: top1 {a['top1']:.3f} tmass {a['tmass']:.3f} ce {a['ce']:.3f} (T={temp:.2f})")
    for k, v in m.items():
        if k != "all":
            print(f"      {k:10s} n={v['n']:3d} top1 {v['top1']:.3f} tmass {v['tmass']:.3f}")
    for k, v in et_m.items():
        print(f"      [{k}] n={v['n']:3d} top1 {v['top1']:.3f} tmass {v['tmass']:.3f}")
    out.write(json.dumps({"t": time.strftime("%m-%d %H:%M"), "ckpt": ck, "ver": args.ver, "split": args.split,
                          "all": a, "by_kind": {k: v for k, v in m.items() if k != "all"}, "by_exam_type": et_m},
                         ensure_ascii=False) + "\n")
    del model
    torch.cuda.empty_cache()
