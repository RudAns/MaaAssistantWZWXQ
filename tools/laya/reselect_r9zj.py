"""r9zj 重标清单重选（2026-10-15）：build_r9zj 的 Z_f9ls 配额被商店题占满（商店题多半只是锁店选项多列了非英雄牌），
用 r9zh（f9lr 文本）细分：状态行变了 = Z_fuel（燃料行改了，必须重标），只是锁店选项变了 = Z_lock（旧标签基本仍有效）。
选：非商店 Z_fuel 全收（carry / swap 除外，局面网络读）、商店 Z_fuel ≤400、Z_lock ≤80，新局全收；测试：新局全收 + 旧测试集 Z_fuel 全收。
写 teacher_*_ids.json 和 relabel_fine.json（assemble_r9zj 只丢 Z_fuel 的旧标签）。"""
import collections, json, random
from pathlib import Path
P = Path(__file__).resolve().parents[2] / "data" / "laya" / "datasets"
rows = {json.loads(l)["id"]: json.loads(l) for l in (P / "r9zj" / "points.jsonl").open(encoding="utf-8")}
prev = {}
for l in (P / "r9zh" / "points.jsonl").open(encoding="utf-8"):
    r = json.loads(l)
    prev[r["id"]] = r["state"]
rl = json.loads((P / "r9zj" / "relabel.json").read_text(encoding="utf-8"))
fine = {}
for i, t in rl.items():
    if "Z_f9ls" in t:
        fine[i] = ["Z_fuel" if prev.get(i) != rows[i]["state"] else "Z_lock"]
    else:
        fine[i] = t
rng = random.Random(20261015)
tr, te = [], []
by = collections.defaultdict(list)
for i, t in sorted(fine.items()):
    k = rows[i]["kind"]
    if k in ("carry", "swap"):
        continue
    sp = rows[i]["split"]
    if "S_new_match" in t or "T_latest_test_all" in t:
        (tr if sp == "train" else te).append(i)
    elif sp != "train":
        if "Z_fuel" in t:
            te.append(i)
    else:
        by[(k == "shop", t[0])].append(i)
for (shop, tag), ids in sorted(by.items()):
    rng.shuffle(ids)
    cap = (400 if tag == "Z_fuel" else 80) if shop else (10**6 if tag == "Z_fuel" else 40)
    tr += ids[:cap]
    print("shop" if shop else "other", tag, len(ids), "->", min(cap, len(ids)))
(P / "r9zj" / "relabel_fine.json").write_text(json.dumps(fine, ensure_ascii=False), encoding="utf-8")
(P / "r9zj" / "teacher_train_ids.json").write_text(json.dumps(sorted(tr)), encoding="utf-8")
(P / "r9zj" / "teacher_test_ids.json").write_text(json.dumps(sorted(te)), encoding="utf-8")
c = collections.Counter((rows[i]["kind"], fine[i][0]) for i in tr)
print("train", len(tr), sorted(c.items()))
c = collections.Counter((rows[i]["kind"], fine[i][0]) for i in te)
print("test", len(te), sorted(c.items()))
