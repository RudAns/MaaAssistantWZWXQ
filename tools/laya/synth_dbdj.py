"""点兵点将 use_timing 反事实题（2026-10-16 晚；R35 在这类题上只有十几道训练题，670 局核心 5/6、能让守约当场觉醒仍判"留着"）。

点兵点将"现在用值多少"只取决于当时场上有谁（几张核心、谁会因此觉醒），和手里是否真有这张牌无关。
所以从数据集里取真实的准备阶段局面（每局每回合一个，第 6~14 回合），在手牌里加一张点兵点将，用同一个题面生成器
（features.q_use_timing，经 build_r9za.render）出"现在用 / 先留着"题。划分沿用原局；source=synthetic、标 Y_dbdj。

  .venv/Scripts/python.exe -X utf8 tools/laya/synth_dbdj.py --ds r9zp --n-train 300 --n-test 80
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_r9za as BA  # noqa: E402
import rerender as RR  # noqa: E402

CARD = "点兵点将"
SRC_KINDS = ("shop", "swap", "use_card", "card_pick")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="r9zp")
    ap.add_argument("--ver", default="f9ly")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261016)
    a = ap.parse_args()
    RR.ALLOW_NEW_KEYS = True
    ds = PROJECT_DIR / "data/laya/datasets" / a.ds
    rows = [json.loads(l) for l in (ds / "points.jsonl").open(encoding="utf-8")]
    have = {r["id"] for r in rows}
    pick = {}
    for p in rows:                                   # 每局每回合取一个局面（第 6~14 回合、场上 ≥4 人）
        if p["kind"] not in SRC_KINDS or not (6 <= (p.get("round") or 0) <= 14) or len(p.get("field") or []) < 4:
            continue
        if CARD in ((p.get("ctx") or {}).get("hand") or []):
            continue                                 # 手里真有这张的局面线上已经出过题
        pick.setdefault((p.get("run"), p.get("round")), p)
    rng = random.Random(a.seed)
    cands = list(pick.values())
    rng.shuffle(cands)
    out, st = [], collections.Counter()
    want = {"train": a.n_train, "test": a.n_test}
    for p in cands:
        grp = "train" if p["split"] == "train" else "test"
        if st[grp] >= want[grp]:
            continue
        q = json.loads(json.dumps(p))
        q["kind"] = "use_timing"
        q["raw"] = {"card": CARD}
        q["ctx"] = dict(q.get("ctx") or {}, hand=list((q.get("ctx") or {}).get("hand") or []) + [CARD])
        q["question"] = {"criteria": {"now": "", "hold": ""}}
        q["options"] = ["now", "hold"]
        r = BA.render(q, a.ver, ["now", "hold"])
        if r is None or set(r["question"]["criteria"]) != {"now", "hold"}:
            st["render_fail"] += 1
            continue
        sid = hashlib.md5(f"{p['id']}|dbdj_syn".encode()).hexdigest()[:12]
        if sid in have:
            continue
        r.update(id=sid, source="synthetic", synth_from=p["id"], split=p["split"], rule_choice="hold")
        out.append(r)
        st[grp] += 1
        core = "CORE" in json.dumps(r["question"], ensure_ascii=False)
        st[f"{grp}:core_line"] += int(core)
    with (ds / "points.jsonl").open("a", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tr = json.loads((ds / "teacher_train_ids.json").read_text(encoding="utf-8"))
    te = json.loads((ds / "teacher_test_ids.json").read_text(encoding="utf-8"))
    tr += [r["id"] for r in out if r["split"] == "train"]
    te += [r["id"] for r in out if r["split"] != "train"]
    (ds / "teacher_train_ids.json").write_text(json.dumps(sorted(set(tr))), encoding="utf-8")
    (ds / "teacher_test_ids.json").write_text(json.dumps(sorted(set(te))), encoding="utf-8")
    tags = json.loads((ds / "tags.json").read_text(encoding="utf-8")) if (ds / "tags.json").exists() else {}
    for r in out:
        tags[r["id"]] = sorted(set(tags.get(r["id"], [])) | {"Y_dbdj"})
    (ds / "tags.json").write_text(json.dumps(tags, ensure_ascii=False), encoding="utf-8")
    (ds / "synth_dbdj_ids.json").write_text(json.dumps([r["id"] for r in out]), encoding="utf-8")
    print(json.dumps({"candidates": len(cands), **st}, ensure_ascii=False))
    if out:
        print("例：", json.dumps(out[0]["question"]["criteria"], ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
