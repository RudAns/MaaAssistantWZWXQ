"""2026-10-13：eco 题 f9ln 数据准备（只做数据，不训练、不标注）。纯 CPU。

  .venv/Scripts/python.exe -X utf8 tools/laya/prep_eco_f9ln.py render --src r9v --out r9v_f9ln
  .venv/Scripts/python.exe -X utf8 tools/laya/prep_eco_f9ln.py render --src r9w_eco_new_f9le --out r9w_eco_new --split-name test_r9we_new
  .venv/Scripts/python.exe -X utf8 tools/laya/prep_eco_f9ln.py select --ds r9v_f9ln --out data/laya/datasets/r9v_f9ln_relabel_ids.json

render：和 r9v → r9v_f9lh 同一做法（rerender.rerender，不 backfill），只是特征版本切到 f9ln；
        同时用 f9lm 再渲染一遍（独立进程），把两版的 eco_missing 段 / CORE 行写进 _cmp（选题用，只存在 cmp.jsonl，不进题面）。
select：挑重标题 —— A. 出现 CORE pieces 行，且 best 等级 ≠ 当前等级、或 core 在当前等级每刷期望 < LOW_HIT；
        B. f9lm → f9ln 的 MISSING 名单因为去掉官方队友（自定义路线 user_*）明显变了（少了 ≥2 人）。
        按"影响大"排序见 cmd_select 的说明（用旧老师标签判断新事实和旧答案是否方向相反）。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import features as F  # noqa: E402
import rerender as RR  # noqa: E402

DS = PROJECT_DIR / "data" / "laya" / "datasets"
LOW_HIT = 0.10          # core 在当前等级每刷新一次期望见到的核心件数 < 0.10 = 很难刷到
CORE_RE = re.compile(r"CORE pieces \((.*?)\) seen per refresh: (.*?) → best at lv(\d)")
MISS_RE = re.compile(r"MISSING support \(\d+; per-refresh chance to see each at lv [\d/]+\): (.*?); missing seen per refresh")


def set_ver(v: str) -> None:
    F.ACTIVE = F.FEATURE_VERSION = v
    F._facts()                     # facts.FV 跟着 ACTIVE


def miss_names(instr: str) -> list:
    m = MISS_RE.search(instr or "")
    if not m:
        return []
    return [re.split(r" t\d ", x.strip(), 1)[0] for x in re.split(r", (?=[^,\[]+ t\d \[)", m.group(1))]


def cmd_render(a) -> int:
    t0 = time.time()
    src = [json.loads(l) for l in (DS / a.src / "points.jsonl").open(encoding="utf-8")]
    # f9lm 在独立进程里渲染（facts 里有 21 个 lru_cache，同一进程切版本不保险）
    import subprocess
    tmp = DS / a.out / "_instr_f9lm.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-X", "utf8", __file__, "instr", "--src", a.src, "--ver", "f9lm", "--out", str(tmp)], check=True)
    old = json.loads(tmp.read_text(encoding="utf-8"))
    tmp.unlink()
    set_ver("f9ln")
    out, cmp_rows, dropped = [], [], collections.Counter()
    for p in src:
        r = RR.rerender(p)
        if r is None:
            dropped[p["kind"]] += 1
            continue
        if a.split_name and r.get("split") != "train":
            r["split"] = a.split_name
        out.append(r)
        ins = r["question"]["instructions"]
        m = CORE_RE.search(ins)
        cmp_rows.append({"id": r["id"], "split": r["split"], "match": r.get("match"), "round": r.get("round"),
                         "level": (r.get("ctx") or {}).get("level"), "route": (r.get("ctx") or {}).get("route") or "",
                         "miss_old": miss_names(old.get(r["id"], "")), "miss_new": miss_names(ins),
                         "core": m.group(1) if m else None, "core_hits": m.group(2) if m else None,
                         "core_best": int(m.group(3)) if m else None})
    od = DS / a.out
    od.mkdir(parents=True, exist_ok=True)
    with (od / "points.jsonl").open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (od / "cmp.jsonl").open("w", encoding="utf-8") as fh:
        for r in cmp_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for f in ("info.json", "matches.json"):
        if (DS / a.src / f).exists():
            t = (DS / a.src / f).read_text(encoding="utf-8")
            if a.split_name:
                t = t.replace("test_r9t_new", a.split_name)
            (od / f).write_text(t, encoding="utf-8")
    info = {"src": a.src, "feature_version": "f9ln", "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "points": len(out), "dropped": dict(dropped), "with_board": sum(1 for r in out if r.get("board")),
            "by_split": dict(collections.Counter(r.get("split") for r in out)),
            "with_core_line": sum(1 for c in cmp_rows if c["core"]),
            "miss_changed": sum(1 for c in cmp_rows if c["miss_old"] != c["miss_new"]),
            "seconds": round(time.time() - t0, 1)}
    (od / "rerender.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))
    return 0


def cmd_instr(a) -> int:
    """某个版本下每道题的 instructions（给 render 对比 f9lm / f9ln 用）。"""
    set_ver(a.ver)
    out = {}
    for line in (DS / a.src / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        r = RR.rerender(p)
        if r is not None:
            out[p["id"]] = r["question"]["instructions"]
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


def core_hits(c: dict) -> dict:
    if not c["core_hits"]:
        return {}
    return {int(p.split(" ")[0][2:]): float(p.split(" ")[1]) for p in c["core_hits"].split(", ")}


def old_labels(names: list) -> dict:
    """旧老师标签：后面的覆盖前面的（r9v：g9s 全量 → g9t 重标的 1069 道）。"""
    out = {}
    for n in names:
        f = PROJECT_DIR / "data" / "laya" / "labels" / n / "labels.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                out[r["id"]] = r
    return out


def cmd_select(a) -> int:
    """候选 = A（CORE 行且 best≠当前等级，或当前等级 core 每刷期望 < LOW_HIT）∪ B（自定义路线、MISSING 少了 ≥2 个官方队友）。
    影响排序（旧标签 = 老师当时没看到 CORE / 名单变化时的判断）：
      P0 B（名单本身变了，旧判断的依据没了）
      P1 A best>L、旧标签首选 roll/save、升到 best 的核心命中增益 ≥ GAIN_BIG（新事实与旧答案方向相反且差得多）
      P2 A 当前等级 core 命中低、旧标签首选 roll（在刷不到核心的等级刷）
      P3 A best>L、旧标签首选 roll/save、增益 < GAIN_BIG
      P4 其余候选（旧答案方向和新事实一致），按增益从大到小
    --max 截断时按 P0→P4、同档内增益大的优先。"""
    rows = [json.loads(l) for l in (DS / a.ds / "cmp.jsonl").open(encoding="utf-8")]
    lab = old_labels(a.old_labels.split(",")) if a.old_labels else {}
    picked = []
    for c in rows:
        h = core_hits(c)
        L = c["level"] if isinstance(c["level"], int) else None
        hl = h.get(L) if L is not None else None
        a_best = bool(h) and L is not None and c["core_best"] != L
        a_low = bool(h) and hl is not None and hl < LOW_HIT
        lost = [n for n in c["miss_old"] if n not in c["miss_new"]]
        b = c["route"].startswith("user_") and len(lost) >= 2
        if not (a_best or a_low or b):
            continue
        gain = (h[c["core_best"]] - (hl or 0)) if a_best else 0.0
        ob = (lab.get(c["id"]) or {}).get("best")
        if b:
            pri = 0
        elif a_best and ob in ("roll", "save") and gain >= a.gain_big:
            pri = 1
        elif a_low and not a_best and ob == "roll":
            pri = 2
        elif a_best and ob in ("roll", "save"):
            pri = 3
        else:
            pri = 4
        picked.append({"id": c["id"], "split": c["split"], "pri": pri, "gain": round(gain, 3), "old_best": ob,
                       "lost": len(lost), "a_best": a_best, "a_low": a_low, "b": b})
    picked.sort(key=lambda x: (x["pri"], -x["lost"], -x["gain"], x["id"]))
    n_all = len(picked)
    if a.max and len(picked) > a.max:
        picked = picked[:a.max]
    Path(a.out).write_text(json.dumps([x["id"] for x in picked], ensure_ascii=False), encoding="utf-8")
    Path(a.out).with_suffix(".detail.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in picked), encoding="utf-8")
    print(json.dumps({"candidates": n_all, "written": len(picked),
                      "by_pri": dict(sorted(collections.Counter(x["pri"] for x in picked).items())),
                      "by_split": dict(collections.Counter(x["split"] for x in picked)),
                      "by_old_best": dict(collections.Counter(str(x["old_best"]) for x in picked))}, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--src", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--split-name", default="", help="把非 train 划分改成这个名字（新补充集别和 r9v 的 test_r9t_new 混名）")
    s = sub.add_parser("select")
    s.add_argument("--ds", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max", type=int, default=1200)
    s.add_argument("--old-labels", default="r9v_g9s,r9v_g9t", help="旧标签目录（逗号分隔，后者覆盖前者）")
    s.add_argument("--gain-big", type=float, default=0.05, help="升到 best 等级后核心件每刷期望至少多这么多才算'差得多'")
    i = sub.add_parser("instr")
    i.add_argument("--src", required=True)
    i.add_argument("--ver", required=True)
    i.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"render": cmd_render, "select": cmd_select, "instr": cmd_instr}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
