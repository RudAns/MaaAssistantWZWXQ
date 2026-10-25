"""用推演器给训练题逐题打分（2026-10-14，局面网络 v4 的特征 / 标签来源）。

每道 carry / swap / eco 题：把题面选项映射成推演器动作，在 PARAM_RANGES 里抽 n 组参数各推演一遍，
输出每个选项的 平均 ΔV、10%/90% 分位、"是最优"的比例（p_best）。结论稳（某选项 p_best ≥ 0.8）的题可直接当标签，
其余交给老师 / 网络学权衡。纯 CPU，多进程。

  .venv/Scripts/python.exe -X utf8 tools/engine/annotate_points.py --ds data/laya/datasets/r9zc_cg/points.jsonl \
      --out data/engine/annot_r9zc.jsonl [--n 8] [--limit 100] [--procs 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


KINDS = ("carry", "swap", "eco", "shop", "card_pick", "talent", "auction", "use_card")


ENGINE_PKG = ["agent.engine"]     # 线上给已部署模型算特征时用冻结版（agent.engine_v1 = cs_r9zd / R28 训练时的推演器）


def _mods():
    import importlib
    return (importlib.import_module(ENGINE_PKG[0] + ".options"), importlib.import_module(ENGINE_PKG[0] + ".state"))


def build(p_point: dict):
    O, S = _mods()
    Action, from_point = O.Action, S.from_point
    raw = p_point.get("raw") or {}
    pos = from_point(p_point)
    kind = p_point["kind"]
    acts = {}
    names = raw.get("names") or {}
    for k in p_point.get("options") or []:
        if kind == "carry":
            if k == "keep":
                acts[k] = [Action("stop", "维持")]
            else:
                nm = names.get(k)
                if nm and pos.unit(nm):
                    acts[k] = [Action("carry", f"换主C → {nm}", {"name": nm}, 10),
                               Action("carry", f"换主C → {nm}（绑定不动）", {"name": nm, "keep_bind": True}, 10)]
        elif kind == "swap":
            inc = raw.get("incoming") or ""
            if inc and inc not in pos.hand:
                pos.hand.append(inc)
            if k == "bench":
                acts[k] = [Action("stop", "不换")]
            elif k == "levelup":
                acts[k] = [Action("levelup", "升级")]
            else:
                nm = names.get(k)
                if nm:
                    acts[k] = [Action("swap", f"卖 {nm} 换上 {inc}", {"sell": nm, "name": inc})]
        elif kind == "shop":
            if k.startswith("buy"):
                it = next((x for x in raw.get("shop") or [] if f"buy{x[0]}" == k), None)
                if it:
                    if it[1] not in pos.shop:
                        pos.shop.append(it[1])
                    acts[k] = [Action("buy", f"买 {it[1]}", {"name": it[1], "cost": it[2]})]
            elif k == "refresh":
                acts[k] = [Action("refresh", "刷新")]
            elif k == "levelup":
                acts[k] = [Action("levelup", "升级")]
            else:
                acts[k] = [Action("stop", k)]
        elif kind in ("card_pick", "talent"):
            lst = raw.get("cards") if kind == "card_pick" else [t[0] if isinstance(t, (list, tuple)) else t
                                                                   for t in raw.get("talents") or []]
            if k.startswith("opt"):
                i = int(k[3:])
                if i < len(lst or []):
                    nm = lst[i]
                    acts[k] = [Action("talent" if kind == "talent" else "gain", f"拿 {nm}", {"name": nm})]
            else:
                acts[k] = [Action("stop", k)]
        elif kind == "auction":
            nm = names.get(k)
            acts[k] = [Action("stop", "不拍")] if k == "pass" or not nm else [Action("gain", f"拍 {nm}", {"name": nm})]
        elif kind == "use_card":
            card = raw.get("card") or ""
            tg = raw.get("targets") or []
            if card and card not in pos.hand:
                pos.hand.append(card)
            if k == "hold":
                acts[k] = [Action("stop", "先留着")]
            elif k.startswith("use"):
                i = int(k[3:])
                if i < len(tg):
                    acts[k] = [Action("use", f"{card} → {tg[i]}", {"name": card, "target": tg[i]})]
        elif kind == "eco":
            acts[k] = [{"levelup": Action("levelup", "升级"), "roll": Action("refresh", "刷新"),
                        "save": Action("stop", "攒钱")}.get(k, Action("stop", k))]
    return pos, acts


def annotate(args):
    line, n = args[0], args[1]
    if len(args) > 2:
        ENGINE_PKG[0] = args[2]
    robust_scores = _mods()[0].robust_scores
    try:
        pt = json.loads(line)
        if pt.get("kind") not in KINDS:
            return None
        pos, acts = build(pt)
        flat = [a for v in acts.values() for a in v]
        if len(acts) < 2:
            return {"id": pt.get("id"), "kind": pt["kind"], "skip": "选项映射不到"}
        r = robust_scores(pos, n=n, actions=flat)
        out = {}
        for k, alts in acts.items():
            best = None
            for a in alts:
                v = r.get(a.label.split(" @")[0])
                if v and (best is None or v["dV_mean"] > best["dV_mean"]):
                    best = v
            if best:
                out[k] = {"dV": round(best["dV_mean"], 3), "lo": round(best["dV_lo"], 3), "hi": round(best["dV_hi"], 3),
                          "p_best": round(best["p_best"], 3)}
        # p_best 按选项合并（同一选项的两种做法加起来）
        for k, alts in acts.items():
            if k in out:
                out[k]["p_best"] = round(sum((r.get(a.label.split(" @")[0]) or {}).get("p_best", 0) for a in alts), 3)
        return {"id": pt.get("id"), "kind": pt["kind"], "split": pt.get("split"), "round": pt.get("round"), "opts": out}
    except Exception as e:                       # noqa: BLE001
        return {"id": None, "err": repr(e)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--pkg", default="agent.engine", help="推演器包（agent.engine = 开发版；agent.engine_v1 = 冻结版）")
    a = ap.parse_args()
    lines = [l for l in open(a.ds, encoding="utf-8")]
    if a.limit:
        lines = lines[:a.limit]
    t0 = time.time()
    ok = err = skip = 0
    with open(a.out, "w", encoding="utf-8") as fo, Pool(a.procs) as pool:
        for i, res in enumerate(pool.imap(annotate, [(l, a.n, a.pkg) for l in lines], chunksize=8)):
            if res is None:
                continue
            if res.get("err"):
                err += 1
            elif res.get("skip"):
                skip += 1
            else:
                ok += 1
            fo.write(json.dumps(res, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(lines)}  {time.time() - t0:.0f}s  ok {ok} skip {skip} err {err}", flush=True)
    print(f"完成 {len(lines)} 道，用时 {time.time() - t0:.0f}s：ok {ok} / 映射不到 {skip} / 出错 {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
