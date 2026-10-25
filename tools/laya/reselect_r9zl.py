"""r9zl 重标清单（2026-10-15，第 642 局复盘：天赋 / 商店配合经济计划 / 杂牌判定 / 换人别卖引擎燃料）。
标签：T_talent（全部天赋题按新事实重标）、E_roll（商店题局面写着经济计划 ROLL）、Z_filler（can replace filler 文本变了）、
Z_f9lu（其余 f9lu 文本变化）、S_engine（换人题：引擎零件 / 稳定燃料 / 路线成员在等 → 给局面网络），T_new（第 642 局起全部题）。"""
import collections, json, random
from pathlib import Path
P = Path(__file__).resolve().parents[2] / "data" / "laya" / "datasets"
rows = {json.loads(l)["id"]: json.loads(l) for l in (P / "r9zl" / "points.jsonl").open(encoding="utf-8")}
prev = {json.loads(l)["id"]: json.loads(l) for l in (P / "r9zk" / "points.jsonl").open(encoding="utf-8")}
new_set = set(json.loads((P / "r9zl" / "new_ids.json").read_text(encoding="utf-8")))
fine = {}
for i, r in rows.items():
    k = r["kind"]
    t = json.dumps([r.get("state"), r["question"]], ensure_ascii=False)
    o = prev.get(i)
    to = json.dumps([o.get("state"), o["question"]], ensure_ascii=False) if o else ""
    tags = []
    if i in new_set:
        tags.append("T_new")
    if k == "talent":
        tags.append("T_talent")
    if k == "shop" and "ECONOMY PLAN this round (economy network): ROLL" in t:
        tags.append("E_roll")
    if o is not None and k != "swap" and ("can replace filler" in t) != ("can replace filler" in to) or (o is not None and k != "swap" and t.count("can replace filler") != to.count("can replace filler")):
        tags.append("Z_filler")
    if o is not None and t != to and not tags:
        tags.append("Z_f9lu")
    if k == "swap" and ("ENGINE PART" in t or "steady carry FUEL" in t or "ROUTE member" in t):
        tags.append("S_engine")
    if tags:
        fine[i] = tags
rng = random.Random(20261016)
CAP_TR = {"T_talent": 700, "E_roll": 300, "Z_filler": 200, "Z_f9lu": 80}
CAP_TE = {"T_talent": 120, "E_roll": 60, "Z_filler": 40}
tr, te, sw_tr, sw_te = set(), set(), set(), set()
by_tr, by_te = collections.defaultdict(list), collections.defaultdict(list)
for i, t in sorted(fine.items()):
    r = rows[i]
    if r["kind"] in ("carry", "swap", "eco"):
        if "S_engine" in t:
            (sw_tr if r["split"] == "train" else sw_te).add(i)
        continue
    if "T_new" in t:
        te.add(i)
        continue
    for x in t:
        (by_tr if r["split"] == "train" else by_te)[x].append(i)
for caps, by, dst in ((CAP_TR, by_tr, tr), (CAP_TE, by_te, te)):
    for x, cap in caps.items():
        ids = [i for i in by.get(x, []) if i not in dst]
        # 越新的局越优先（按 match 号倒序抽一半，另一半随机）
        ids.sort(key=lambda i: str(rows[i].get("match") or ""), reverse=True)
        head, rest = ids[: cap // 2], ids[cap // 2:]
        rng.shuffle(rest)
        dst |= set(head + rest[: cap - len(head)])
sw_tr = set(sorted(sw_tr)[:0]) | set(rng.sample(sorted(sw_tr), min(500, len(sw_tr))))
sw_te = set(rng.sample(sorted(sw_te), min(120, len(sw_te))))
(P / "r9zl" / "relabel_fine.json").write_text(json.dumps(fine, ensure_ascii=False), encoding="utf-8")
for name, ids in (("teacher_train_ids", tr), ("teacher_test_ids", te), ("teacher_swap_train_ids", sw_tr), ("teacher_swap_test_ids", sw_te)):
    (P / "r9zl" / f"{name}.json").write_text(json.dumps(sorted(ids)), encoding="utf-8")
c = collections.Counter(x for t in fine.values() for x in t)
print("候选", dict(c))
print("train", len(tr), collections.Counter(rows[i]["kind"] for i in tr))
print("test", len(te), collections.Counter(rows[i]["kind"] for i in te))
print("swap", len(sw_tr), len(sw_te))
