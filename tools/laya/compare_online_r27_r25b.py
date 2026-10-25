"""第 606~612 局线上 Laya 题：分别渲染 f9lk(R25b) / f9lm(R27)，两个学生推理，输出逐题结果。"""
import collections, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "laya"))
sys.path.insert(0, str(ROOT))
import build_r9s as BR  # noqa
import build_r9za as BA  # noqa
import features as F  # noqa
import rerender as RR  # noqa

KINDS = ("shop", "card_pick", "talent", "auction", "use_card", "lord_boost")
OUT = Path(sys.argv[1])
matches = {}
for line in (ROOT / "data/laya/iterations/batch2_matches.jsonl").open(encoding="utf-8"):
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if 606 <= (m.get("match") or 0) <= 612:
        matches[m["run"]] = m
RR.ALLOW_NEW_KEYS = True
cfgs = BR.run_render_cfgs()
st = collections.Counter()
pairs = []
for run, m in sorted(matches.items()):
    rows = [dict(json.loads(l), run=run) for l in (ROOT / "data/logs" / run / "laya_points.jsonl").open(encoding="utf-8")]
    log_cfg = {k: v for k, v in (cfgs.get(run) or {}).items() if k != "log"} or None
    F.ACTIVE = "f9lm"
    cfg, hit = BR.pick_cfg(rows, log_cfg)
    st[f"cfg_hit:{run}"] = hit
    for p in rows:
        if p.get("kind") not in KINDS or not p.get("question"):
            continue
        online = {k: p.get(k) for k in ("laya", "laya_choice", "laya_conf", "used", "why", "ckpt", "model", "rule_choice")}
        online_state = p.get("state")
        for k in BA.STUDENT_KEYS:
            p.pop(k, None)
        p["id"] = BR.pid(run, p.get("ts"), p["kind"])
        p["render_cfg"] = cfg
        keys = list(p["question"]["criteria"])
        a = BA.render(p, "f9lm", keys)
        b = BA.render(p, "f9lk", keys)
        if a is None or b is None:
            st["drop:" + p["kind"]] += 1
            continue
        st["kept:" + p["kind"]] += 1
        st["f9lm_state_same_as_online"] += int(a["state"] == online_state)
        pairs.append({"id": p["id"], "kind": p["kind"], "match": m["match"], "rank": m["rank"], "round": p.get("round"),
                      "ctx": p.get("ctx"), "raw": p.get("raw"), "online": online, "m": a, "k": b})
BR.apply_cfg(None)
print(json.dumps(dict(st), ensure_ascii=False, indent=0))

import torch  # noqa
import train as T  # noqa
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = {}
for name, ck, side in (("R25b", "data/laya/runs/R25b_s1_top8/checkpoint", "k"),
                       ("R27", "data/laya/runs/R27_s1_top8/checkpoint", "m")):
    F.ACTIVE = "f9lk" if side == "k" else "f9lm"
    model, tok, cfg = T.load_model(ROOT / ck, dev)
    temp = (cfg.get("temperature") or [1.0])[0]
    rows = [dict(x[side], teacher={}) for x in pairs]
    t0 = time.time()
    preds = T.predict(model, tok, cfg, rows, dev, temp=temp)
    print(name, "temp", temp, "secs", round(time.time() - t0, 1))
    res[name] = {p["id"]: dict(zip(p["keys"], p["p"])) for p in preds}
    del model
    torch.cuda.empty_cache()
with OUT.open("w", encoding="utf-8") as f:
    for x in pairs:
        f.write(json.dumps({"id": x["id"], "kind": x["kind"], "match": x["match"], "rank": x["rank"], "round": x["round"],
                            "ctx": x["ctx"], "raw": x["raw"], "online": x["online"],
                            "state_m": x["m"]["state"], "state_k": x["k"]["state"],
                            "crit_m": x["m"]["question"]["criteria"], "crit_k": x["k"]["question"]["criteria"],
                            "p25": res["R25b"][x["id"]], "p27": res["R27"][x["id"]]}, ensure_ascii=False) + "\n")
print("wrote", OUT, len(pairs))
