"""r9zk 重标清单（2026-10-15）：f9lt 的 6 级实测概率让 1.3 万道题的数字变了，旧标签基本仍有效 → 不重标、不丢。
只重标结构性变化 + g9ze 点到的题：
  U_targets   用牌题的目标列表变了（选项键同名、指的人变了 → 旧标签必须作废）
  U_auto      "消除最低 N 名"这类，选项写 target choice ignored
  Z_free      本回合限定免费刷新（THIS ROUND ONLY）
  E_endgame / E_engine_filler   规范 54.5 / 54.6
  S_second    主C 已定（FINAL carry settled）而选项里有第二个 C 位候选（2nd carry candidate / EXTRA OUTPUT）
写 teacher_*_ids.json、relabel_fine.json（assemble 只丢 U_targets 的旧标签；其余没重标的保留旧标签）。"""
import collections, json, random
from pathlib import Path
P = Path(__file__).resolve().parents[2] / "data" / "laya" / "datasets"
rows = {json.loads(l)["id"]: json.loads(l) for l in (P / "r9zk" / "points.jsonl").open(encoding="utf-8")}
prev = {json.loads(l)["id"]: json.loads(l) for l in (P / "r9zj" / "points.jsonl").open(encoding="utf-8")}
new_set = set(json.loads((P / "r9zk" / "new_ids.json").read_text(encoding="utf-8")))
fine = {}
for i, r in rows.items():
    k = r["kind"]
    if k in ("carry", "swap", "eco"):
        continue
    t = json.dumps([r.get("state"), r["question"]], ensure_ascii=False)
    tags = []
    o = prev.get(i)
    if k == "use_card" and o is not None:
        a = [v.split(":")[0] for kk, v in sorted(r["question"]["criteria"].items())]
        b = [v.split(":")[0] for kk, v in sorted(o["question"]["criteria"].items())]
        if a != b:
            tags.append("U_targets")
    if "target choice ignored" in t:
        tags.append("U_auto")
    if "THIS ROUND ONLY" in t:
        tags.append("Z_free")
    if k == "shop" and ("only merges help" in t or "finds would only wait" in t):
        tags.append("E_endgame")
    if k == "shop" and "can replace filler" in t and ("MIN ENGINE piece missing" in t or "ROUTE stage unit" in t):
        tags.append("E_engine_filler")
    if k in ("shop", "auction", "card_pick") and "FINAL carry settled" in t and ("2nd carry candidate" in t or "EXTRA OUTPUT" in t):
        tags.append("S_second")
    if i in new_set:
        tags.append("T_new")
    if tags:
        fine[i] = tags
CAPS = {"U_targets": 10**6, "U_auto": 10**6, "Z_free": 150, "E_endgame": 350, "E_engine_filler": 250, "S_second": 300}
rng = random.Random(20261015)
tr, te = set(), set()
TCAP = {"S_second": 120, "E_engine_filler": 60, "E_endgame": 60, "Z_free": 20}
tby = collections.defaultdict(list)
for i, t in sorted(fine.items()):
    if rows[i]["split"] == "train":
        continue
    if "T_new" in t or "U_targets" in t or "U_auto" in t:
        te.add(i)
    for x in t:
        tby[x].append(i)
for x, cap in TCAP.items():
    ids = [i for i in tby.get(x, []) if i not in te]
    rng.shuffle(ids)
    te |= set(ids[:cap])
by = collections.defaultdict(list)
for i, t in sorted(fine.items()):
    if rows[i]["split"] == "train":
        for x in t:
            by[x].append(i)
for x, cap in CAPS.items():
    ids = [i for i in by.get(x, []) if i not in tr]
    rng.shuffle(ids)
    tr |= set(ids[:cap])
    print(x, len(by.get(x, [])), "->", min(cap, len(ids)))
(P / "r9zk" / "relabel_fine.json").write_text(json.dumps(fine, ensure_ascii=False), encoding="utf-8")
(P / "r9zk" / "teacher_train_ids.json").write_text(json.dumps(sorted(tr)), encoding="utf-8")
(P / "r9zk" / "teacher_test_ids.json").write_text(json.dumps(sorted(te)), encoding="utf-8")
print("train", len(tr), collections.Counter(rows[i]["kind"] for i in tr))
print("test", len(te), collections.Counter(rows[i]["kind"] for i in te))
