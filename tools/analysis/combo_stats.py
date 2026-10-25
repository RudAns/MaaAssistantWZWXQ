"""钻石段（第 100 局起）按"Laya 学生 + 局面网络"组合统计前三率（2026-10-05：实验完成后用前三率最高的组合来打，主要看钻石阶段）。
组合从每局日志里读：Laya checkpoint（--laya-ckpt）与 cgnet 加载行。带 note 的局（GPU 抢占等）不计。"""
import json, re, collections, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(l) for f in ("batch_matches.jsonl", "batch2_matches.jsonl")
        for l in (ROOT / "data/laya/iterations" / f).open(encoding="utf-8")]
lo = int(sys.argv[1]) if len(sys.argv) > 1 else 100
g = collections.defaultdict(list)
for x in rows:
    if x["match"] < lo or not x.get("rank") or x.get("note"):
        continue
    p = ROOT / f"data/logs/agent_one_{x['match']}.log"
    if not p.exists():
        continue
    t = p.read_text(encoding="utf-8", errors="ignore")[:200000]
    laya = (re.findall(r"(R\d+)_s1_top8", t) or ["?"])[0]
    cg = re.search(r"cgnet 已加载[^{]*\{'carry': '([^(']+)", t)
    g[(laya, cg.group(1) if cg else "?")].append((x["match"], x["rank"]))
for k, v in sorted(g.items(), key=lambda kv: -sum(r <= 3 for _, r in kv[1]) / len(kv[1])):
    rk = [r for _, r in v]
    print(f"{k[0]:>5} + {k[1]:<6} 局数 {len(rk):3d}  前三 {sum(r<=3 for r in rk):3d} ({sum(r<=3 for r in rk)/len(rk):.0%})  平均 {sum(rk)/len(rk):.2f}  第 {v[0][0]}~{v[-1][0]} 局")
