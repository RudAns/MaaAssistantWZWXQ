"""r9u 定向重标清单收窄（2026-10-10）：build_r9u.py 按"题面出现新事实"挑出的候选里，只留**新事实会改变旧标签判断**的题。

  .venv/Scripts/python.exe -X utf8 tools/laya/select_r9u_relabel.py

读 datasets/r9u/relabel.json（候选，build_r9u 写）+ labels/r9s_all（旧标签，g9q 及以前），改写
datasets/r9u/relabel_train_ids.json、relabel_test_ids.json，选择理由写 relabel_selected.json（候选原样保存在 relabel_candidates.json）。

- 新题（第 470~487 局，没有旧标签）：带 A/B/C 标记的全留；test_r9u_new 全部题都留（最新测试集）。
- A 刷新后买不起英雄：旧标签给 refresh >0.05 的留（g9r 43.1 要压到 ≤0.05；≤0.05 的旧标签已经对了）。
- B 发英雄牌：某个发英雄牌选项 ① expect <0.3、旧标签给它 >0.1、且满员（`(0 free slots)`）或手牌 ≥8/10（43.2 第 1 条要压到 ≤0.1）；
  或 ② expect ≥0.7、旧标签给它 <0.3（43.2 第 4 条：被低估）。
- C 专属天赋：全留（旧题面写 card not in database / 空白，旧标签是猜的）。
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DS = PROJECT_DIR / "data" / "laya" / "datasets" / "r9u"
EXPECT = re.compile(r"expect ([\d.]+) useful")


def main() -> int:
    pts = {}
    for line in (DS / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        pts[p["id"]] = p
    old = {}
    for line in (PROJECT_DIR / "data" / "laya" / "labels" / "r9s_all" / "labels_all.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            old[r["id"]] = r
    cand_f = DS / "relabel_candidates.json"
    if not cand_f.exists():
        cand_f.write_text((DS / "relabel.json").read_text(encoding="utf-8"), encoding="utf-8")
    cand = json.loads(cand_f.read_text(encoding="utf-8"))
    sel: dict = {}
    for i, tags in sorted(cand.items()):
        p, o = pts[i], old.get(i)
        why = []
        if p["split"] == "test_r9u_new":
            why.append("T_latest_test_all")
        for t in tags:
            if t == "T_latest_test_all":
                continue
            if o is None:
                why.append(t + ":new")
            elif t == "A_refresh_cant_afford" and o["dist"].get("refresh", 0) > 0.05:
                why.append(t + ":old_refresh>0.05")
            elif t == "B_grant_pool":
                st = p["state"]
                full = bool(re.search(r"\(0 free slots?\)", st))
                hm = re.search(r"hand (\d+)/10", st)
                hfull = bool(hm and int(hm.group(1)) >= 8)
                for k, v in p["question"]["criteria"].items():
                    m = EXPECT.search(v)
                    if not m:
                        continue
                    e, ms = float(m.group(1)), o["dist"].get(k, 0.0)
                    if e < 0.3 and ms > 0.1 and (full or hfull):
                        why.append(f"{t}:low_expect_full({k})")
                    if e >= 0.7 and ms < 0.3:
                        why.append(f"{t}:high_expect_undervalued({k})")
            elif t == "C_exclusive_talent":
                why.append(t)
        if why:
            sel[i] = why
    tr = sorted(i for i in sel if pts[i]["split"] == "train")
    te = sorted(i for i in sel if pts[i]["split"] != "train")
    (DS / "relabel_train_ids.json").write_text(json.dumps(tr), encoding="utf-8")
    (DS / "relabel_test_ids.json").write_text(json.dumps(te), encoding="utf-8")
    (DS / "relabel_selected.json").write_text(json.dumps(sel, ensure_ascii=False, indent=0), encoding="utf-8")
    # assemble / eval 读 relabel.json：只保留选中的（标记去掉理由后缀）
    (DS / "relabel.json").write_text(json.dumps({i: sorted({w.split(":")[0] for w in v}) for i, v in sel.items()},
                                                ensure_ascii=False, indent=0), encoding="utf-8")
    c = collections.Counter()
    for i, v in sel.items():
        sp = "train" if pts[i]["split"] == "train" else "test"
        for w in sorted({x.split("(")[0] for x in v}):
            c[f"{sp}:{pts[i]['kind']}:{w}"] += 1
    print(json.dumps({"candidates": len(cand), "selected_train": len(tr), "selected_test": len(te),
                      "by_split": dict(collections.Counter(pts[i]["split"] for i in sel)),
                      "by_kind": dict(collections.Counter(pts[i]["kind"] for i in sel)),
                      "reasons": dict(sorted(c.items()))}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
