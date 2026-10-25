"""r9zm 重标清单（2026-10-15 夜，f9lv：引擎零件收窄 / 开局件过阶段 / 弈星剩余价值 / 满员可替换杂牌 / 前排换人 / 条件天赋）。
标签：T_new（第 643 局起全部题，进测试）、Z_f9lv（文本和 f9lu 不同的 Laya 题）、S_f9lv（文本变了的换人题 → 局面网络）。"""
import collections, json, random
from pathlib import Path
P = Path(__file__).resolve().parents[2] / "data" / "laya" / "datasets"
rows = {json.loads(l)["id"]: json.loads(l) for l in (P / "r9zm" / "points.jsonl").open(encoding="utf-8")}
prev = {json.loads(l)["id"]: json.loads(l) for l in (P / "r9zl" / "points.jsonl").open(encoding="utf-8")}
new_set = set(json.loads((P / "r9zm" / "new_ids.json").read_text(encoding="utf-8")))
fine = {}
for i, r in rows.items():
    k = r["kind"]
    o = prev.get(i)
    changed = o is not None and json.dumps([r.get("state"), r["question"]], ensure_ascii=False, sort_keys=True) != \
        json.dumps([o.get("state"), o["question"]], ensure_ascii=False, sort_keys=True)
    tags = []
    if i in new_set:
        tags.append("T_new")
    if changed:
        tags.append("S_f9lv" if k == "swap" else "Z_f9lv")
    if tags:
        fine[i] = tags
rng = random.Random(20261017)
CAP_TR, CAP_TE = 700, 150
SW_TR, SW_TE = 600, 150
tr, te, sw_tr, sw_te = [], [], [], []
for i, t in sorted(fine.items()):
    r = rows[i]
    if r["kind"] in ("carry", "eco"):
        continue
    if r["kind"] == "swap":
        if "S_f9lv" in t or "T_new" in t:
            (sw_tr if r["split"] == "train" else sw_te).append(i)
        continue
    if "T_new" in t:
        te.append(i)
    elif "Z_f9lv" in t:
        (tr if r["split"] == "train" else te).append(i)
def cap(ids, n):
    ids = sorted(ids)
    new = [i for i in ids if i in new_set]
    old = [i for i in ids if i not in new_set]
    rng.shuffle(old)
    return sorted(new + old[: max(0, n - len(new))])
tr, te = cap(tr, CAP_TR), cap(te, CAP_TE + len([i for i in te if i in new_set]))
sw_tr, sw_te = cap(sw_tr, SW_TR), cap(sw_te, SW_TE + len([i for i in sw_te if i in new_set]))
(P / "r9zm" / "relabel_fine.json").write_text(json.dumps(fine, ensure_ascii=False), encoding="utf-8")
for name, ids in (("teacher_train_ids", tr), ("teacher_test_ids", te), ("teacher_swap_train_ids", sw_tr), ("teacher_swap_test_ids", sw_te)):
    (P / "r9zm" / f"{name}.json").write_text(json.dumps(ids), encoding="utf-8")
c = collections.Counter(f"{rows[i]['kind']}:{x}" for i, t in fine.items() for x in t)
print(json.dumps({"train": len(tr), "test": len(te), "swap_train": len(sw_tr), "swap_test": len(sw_te),
                  "by_kind": dict(sorted(c.items()))}, ensure_ascii=False, indent=1))
