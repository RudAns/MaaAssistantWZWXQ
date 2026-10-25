"""r9zn 增量重标（2026-10-16）：标注期间题面按老师反馈修了几轮，重建 r9zn 后找出
已标、但文本变了 / 题没了的（旧标签作废），以及还没有有效标签的重标题 → 写 teacher_{train,test}_ids_c.json（导出到 r9zn_c / r9zn_test_c）。
每个标签目录配一份"标注时老师看到的版本"快照：
  .venv/Scripts/python.exe -X utf8 tools/laya/r9zn_delta.py --snap r9zn=<v1 points> --snap r9zn_b=<v2 points> --snap r9zn_test=<v2 points>
"""
import argparse
import json
from pathlib import Path

P = Path(__file__).resolve().parents[2]
DS = P / "data/laya/datasets/r9zn"


def sig(p):
    return json.dumps([p.get("state"), p["question"]], ensure_ascii=False, sort_keys=True)


def load_sig(f):
    out = {}
    for l in open(f, encoding="utf-8"):
        p = json.loads(l)
        out[p["id"]] = sig(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", action="append", required=True, help="标签目录=快照 points.jsonl")
    a = ap.parse_args()
    now = load_sig(DS / "points.jsonl")
    valid, stale, st, per_dir = set(), set(), {}, {}
    for item in a.snap:
        d, f = item.split("=", 1)
        snap = load_sig(f)
        done = set()
        for g in (P / "data/laya/labels" / d / "out").glob("*.jsonl"):
            done |= {json.loads(l)["id"] for l in g.open(encoding="utf-8") if l.strip()}
        ok = {i for i in done if i in now and snap.get(i) == now[i]}
        bad = done - ok
        valid |= ok
        stale |= bad
        per_dir[d] = sorted(bad)
        st[d] = {"labeled": len(done), "still_valid": len(ok), "stale": len(bad)}
    stale -= valid                      # 同一题在后一轮已按新文本标过 → 不算作废
    tr = set(json.loads((DS / "teacher_train_ids.json").read_text(encoding="utf-8")))
    te = set(json.loads((DS / "teacher_test_ids.json").read_text(encoding="utf-8")))
    todo_tr, todo_te = sorted(tr - valid), sorted(te - valid)
    (DS / "teacher_train_ids_c.json").write_text(json.dumps(todo_tr), encoding="utf-8")
    (DS / "teacher_test_ids_c.json").write_text(json.dumps(todo_te), encoding="utf-8")
    (DS / "r9zn_stale_ids.json").write_text(json.dumps(per_dir), encoding="utf-8")      # {标签目录: 作废的题号}
    print(json.dumps({"by_dir": st, "valid_total": len(valid), "stale_total": len(stale),
                      "teacher_train": len(tr), "teacher_test": len(te),
                      "to_label_train": len(todo_tr), "to_label_test": len(todo_te)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
