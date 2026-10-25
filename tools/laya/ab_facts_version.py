"""同一个 Laya 学生、同一批真实题，换题面版本（如 f9lm → f9ln）答案变了哪些（2026-10-13：Laya 训练耗时长，先看题面改动影响哪些答案）。
不训练：只看题面新写的事实能不能直接让现有学生选对。

  .venv/Scripts/python.exe -X utf8 tools/laya/ab_facts_version.py --ckpt data/laya/runs/R27_s1_top8/checkpoint \
      --old f9lm --new f9ln --matches 606-617 [--out data/laya/ab_f9ln.jsonl]

输出：每类题答案改变的比例；改变的题按"新题面多了什么"分组，各打印几道（旧答案 / 新答案 / 新增事实）。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "laya"))
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

KINDS = ("shop", "card_pick", "talent", "use_card", "auction", "lord_boost")
TAGS = {"MISMATCH": r"MISMATCH", "FULL": r"equips 3/3 FULL", "carry_equips": r"carry \S+ equips \d/3",
        "ROUTE unit": r"ROUTE unit: can replace", "eco plan": r"economy plan this round", "waiting": r"waiting \S+:"}


def runs(spec: str) -> list[str]:
    a, b = (int(x) for x in spec.split("-"))
    out = []
    for n in range(a, b + 1):
        log = ROOT / "data" / "logs" / f"agent_one_{n}.log"
        if log.exists():
            m = re.search(r"live_\d+", log.read_text(encoding="utf-8", errors="replace")[:200000])
            if m:
                out.append(m.group(0))
    return out


def render(p: dict, ver: str):
    F.ACTIVE = ver
    F._facts()
    try:
        return RR.rerender(dict(p))
    except Exception:            # noqa: BLE001
        return None


def text(r: dict) -> str:
    return json.dumps(r.get("question", {}).get("criteria", {}), ensure_ascii=False) + r.get("state", "") + \
        (r.get("question", {}).get("instructions") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--old", default="f9lm")
    ap.add_argument("--new", default="f9ln")
    ap.add_argument("--matches", default="606-617")
    ap.add_argument("--out", default="data/laya/ab_f9ln.jsonl")
    ap.add_argument("--show", type=int, default=4)
    a = ap.parse_args()
    pts = []
    for run in runs(a.matches):
        f = ROOT / "data" / "logs" / run / "laya_points.jsonl"
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            try:
                p = json.loads(line)
            except ValueError:
                continue
            if p.get("kind") in KINDS:
                p.setdefault("run", run)
                p["id"] = p.get("id") or f"{run}:{len(pts)}"
                pts.append(p)
    pairs = []
    for p in pts:
        ro, rn = render(p, a.old), render(p, a.new)
        if ro and rn and list(ro["question"]["criteria"]) == list(rn["question"]["criteria"]):
            if text(ro) != text(rn):
                ro.update(id=p["id"], teacher={}); rn.update(id=p["id"], teacher={})   # predict 要 teacher 字段（占位）
                pairs.append((p, ro, rn))
    print(f"{len(pts)} 道 Laya 题，题面有变化的 {len(pairs)} 道（只对这些比答案）")
    import torch  # noqa: E402
    import train as T  # noqa: E402
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = T.load_model(Path(a.ckpt), dev)
    temp = (cfg.get("temperature") or [1.0])[0]
    po = {x["id"]: x for x in T.predict(model, tok, cfg, [ro for _, ro, _ in pairs], dev, temp=temp)}
    pn = {x["id"]: x for x in T.predict(model, tok, cfg, [rn for _, _, rn in pairs], dev, temp=temp)}
    out = (ROOT / a.out).open("w", encoding="utf-8")
    by_kind, by_tag, shown = collections.Counter(), collections.defaultdict(list), collections.Counter()
    for p, ro, rn in pairs:
        o, n = po[ro["id"]], pn[rn["id"]]
        ko, kn = o["keys"][max(range(len(o["p"])), key=o["p"].__getitem__)], n["keys"][max(range(len(n["p"])), key=n["p"].__getitem__)]
        tags = [t for t, rx in TAGS.items() if re.search(rx, text(rn)) and not re.search(rx, text(ro))]
        by_kind[(p["kind"], ko != kn)] += 1
        rec = {"id": p["id"], "kind": p["kind"], "round": p.get("round"), "old": ko, "new": kn, "tags": tags,
               "p_old": dict(zip(o["keys"], [round(x, 2) for x in o["p"]])),
               "p_new": dict(zip(n["keys"], [round(x, 2) for x in n["p"]])),
               "crit_old": ro["question"]["criteria"], "crit_new": rn["question"]["criteria"]}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if ko != kn:
            for t in tags or ["(其它措辞)"]:
                by_tag[t].append(rec)
    print("按题型：改变 / 题面有变化")
    for k in KINDS:
        ch, same = by_kind[(k, True)], by_kind[(k, False)]
        if ch + same:
            print(f"  {k:10s} {ch}/{ch + same}")
    for t, recs in sorted(by_tag.items(), key=lambda x: -len(x[1])):
        print(f"\n## 新增「{t}」后答案变了 {len(recs)} 道")
        for r in recs[:a.show]:
            print(f"- {r['id']} R{r['round']} {r['kind']}：{r['old']}({r['p_old'].get(r['old'])}) → {r['new']}({r['p_new'].get(r['new'])})")
            print(f"    旧选：{str(r['crit_old'].get(r['old']))[:140]}")
            print(f"    新选：{str(r['crit_new'].get(r['new']))[:140]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
