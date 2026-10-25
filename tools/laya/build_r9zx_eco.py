"""r9zx_eco 数据集（2026-10-18，经济题重标 / 经济网络重训用）：第 700 局起**线上** eco 题（laya_points.jsonl 里 kind=="eco"）按最新题面版本重渲染。

为什么：线上经济网络 cs_r9zh_nf 是 10-15 用 r9zh_eco（g9zc，第 330~615 局离线 eco 题 f9ln）训的，之后没更新；
10-18 定了 54.18-2 / 54.18-8（路线定好不换、缺最终阵容件就在对应棋手等级多刷）——第 746 局 R7 缺铠、明世隐（t3），eco 判 save 0.46 vs roll 0.45。
第 616 局以后的 eco 题从没进过训练集（r9zq_cg 的 eco 题全部是 offline_eco、第 330~615 局）。

来源与口径：
- 局：batch2_matches.jsonl 里 match >= --min-match、有 run 的局；中途重启的后续目录按 build_r9s.group_runs 并进同一局；
  EXTRA_RUNS 显式补：第 721 局实际在 live_1792237989 里打（batch2 记的 live_1792238001 只有第 1 回合、和它时间重叠 —— 当时两个 agent 同时在跑）。
- 进行中的局（最新目录且 states.jsonl 最近 --busy-min 分钟内还在写）排除。
- 每局每回合只取一题（同一回合重启后重复的丢掉）。
- 渲染：rerender.rerender 按 --ver（默认 = facts 里最大的题面版本）出 question / state；raw.missing 按同一版本重算（局面网络读 raw.missing，
  不重算就和题面对不上）。路线配置（carry_tiers / 公孙离排除）按 build_r9s.pick_cfg 用线上原文反推（同 build_r9zm）。
- online_eco：线上经济网络当时的答案（choice / probs / model / 线上题面版本），只做分析，不进题面、不给老师看（label_batches.render_point 不读它）。
- hindsight（只给老师核对，不进题面）+ rule_choice（本回合实际做了什么）：build_dataset.extract_eco 从 states.jsonl 还原，按回合对上。
- 划分：按局号，最新 --test-matches 局 → test_r9zx_new，其余 train。
- 标签 tag（统计 / 定向测试用）：E_core_missing（缺 MIN ENGINE core / 路线主C 未到手）、E_route_locked（路线已锁且主C 在场）、
  E_full（满员）、E_online_save_core（线上判 save 但缺核心件 —— 第 746 局那类）。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zx_eco.py --check     # 只打印统计
  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9zx_eco.py             # 写 data/laya/datasets/r9zx_eco/
输出：points.jsonl、info.json、matches.json、tags.json、targets/<tag>.json（测试集里带该 tag 的题号）。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # 纯 CPU（对局可能在跑）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_dataset as BD  # noqa: E402
import build_r9s as BR  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

NAME = "r9zx_eco"
TEST_SPLIT = "test_r9zx_new"
EXTRA_RUNS = {721: ["live_1792237989"]}
STUDENT_KEYS = BR.STUDENT_KEYS + ("model",)
CORE_SRC = ("MIN ENGINE core", "ROUTE CARRY not owned")


def latest_version() -> str:
    """facts 里登记的最大题面版本（F9* 常量，按字符串比大小，同 features._v 的口径）。"""
    X = F._facts()
    vs = [v for k, v in vars(X).items() if k.startswith("F9") and isinstance(v, str) and v.startswith("f9")]
    return max(vs + [F.FEATURE_VERSION, F.ECO_VERSION])


def set_ver(v: str) -> None:
    F.ACTIVE = v
    F._facts()                      # facts.FV 跟着 ACTIVE


def eco_ctx(p: dict) -> dict:
    """和 rerender.rerender 对 eco 题同一套 ctx 补全（raw.missing 要在同一个 ctx 上重算）。"""
    ctx = dict(p.get("ctx") or {})
    if F._v("f9k") and "route" not in ctx:
        ctx.update(RR.route_at(p.get("run"), p.get("ts"), ctx.get("round")))
    if F._v("f9g") and "carry_since" not in ctx:
        ctx.update(RR.carry_hist_at(p.get("run"), p.get("ts")))
    if F._v("f9lw") and "lordxp_since" not in ctx:
        ctx["lordxp_since"] = RR.carry_hist_at(p.get("run"), p.get("ts")).get("lordxp_since") or {}
    if F._v("f9li") and "talents" not in ctx:
        ctx["talents"] = RR.talents_at(p.get("run") or "", float(p.get("ts") or 0))
    if F._v("f9ly") and "worn" not in ctx:
        ctx["worn"] = RR.worn_at(p.get("run") or "", ctx.get("round") or p.get("round") or 0)
    board = p.get("board") or {}
    ctx["field"] = {n: dict(board.get(n) or {}) for n in (p.get("field") or [])}
    return ctx


def eco_raw(ctx: dict) -> dict:
    X = F._facts()
    B = X.Board(ctx)
    miss = X.eco_missing(B)
    return {"missing": [[n, q, src] for n, q, src in miss], "carry": X.eco_carry(B), "level_cost_used": X.eco_level_cost(ctx)}


def tags_of(p: dict) -> list[str]:
    miss = (p.get("raw") or {}).get("missing") or []
    core = any(len(m) > 2 and m[2] in CORE_SRC for m in miss)
    ins = p["question"]["instructions"]
    out = []
    if core:
        out.append("E_core_missing")
    ctx = p.get("ctx") or {}
    carry = (p.get("raw") or {}).get("carry") or ""
    if ctx.get("route") and carry and carry in (p.get("field") or []):
        out.append("E_route_locked")
    if "board FULL" in ins:
        out.append("E_full")
    if core and (p.get("online_eco") or {}).get("choice") == "save":
        out.append("E_online_save_core")
    return out


def match_runs(min_match: int, busy_min: float, st: collections.Counter) -> tuple[dict, dict]:
    rows = {}
    for line in (PROJECT_DIR / "data/laya/iterations/batch2_matches.jsonl").open(encoding="utf-8"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if (m.get("match") or 0) >= min_match and m.get("run"):
            rows[m["match"]] = m
    logs = PROJECT_DIR / "data/logs"
    first = min(r["run"] for r in rows.values())
    now = time.time()
    spans, busy = [], []
    newest = max(p for p in logs.glob("live_*") if p.is_dir())
    for d in sorted(logs.glob("live_*")):
        if not d.is_dir() or d.name < first or not (d / "states.jsonl").exists():
            continue
        if d == newest and now - (d / "states.jsonl").stat().st_mtime < busy_min * 60:   # 只排除最新目录（上一局 agent 停在结算页也会写）
            busy.append(d.name)
            continue
        spans.append((d.name, BR.run_span(d)))
    run2match = {r["run"]: k for k, r in rows.items()}
    for k, v in EXTRA_RUNS.items():
        for d in v:
            run2match.setdefault(d, k)
    by_match: dict = {}
    unmapped = []
    for g in BR.group_runs(spans):
        ms = [run2match[x] for x in g if x in run2match]
        if ms:
            by_match.setdefault(ms[0], [])
            by_match[ms[0]] += [x for x in g if x not in by_match[ms[0]]]
        else:
            unmapped += g
    for k in by_match:
        by_match[k].sort()
    st["busy_runs_skipped"] = len(busy)
    return by_match, {"rows": rows, "busy": busy, "unmapped": unmapped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只打印统计，不写文件")
    ap.add_argument("--min-match", type=int, default=700)
    ap.add_argument("--test-matches", type=int, default=10, help="最新的几局做测试集")
    ap.add_argument("--ver", default="", help="题面版本（默认 facts 里最大的版本）")
    ap.add_argument("--busy-min", type=float, default=10.0)
    ap.add_argument("--no-hindsight", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    st = collections.Counter()
    ver = args.ver or latest_version()
    RR.ALLOW_NEW_KEYS = True
    by_match, meta = match_runs(args.min_match, args.busy_min, st)
    rows = meta["rows"]
    cfgs = BR.run_render_cfgs()
    logs = PROJECT_DIR / "data/logs"
    out = []
    unmapped_eco = {}
    for d in meta["unmapped"]:
        f = logs / d / "laya_points.jsonl"
        if f.exists():
            n = sum(1 for l in f.open(encoding="utf-8") if '"kind": "eco"' in l)
            if n:
                unmapped_eco[d] = n
    for mn in sorted(by_match):
        seen_rounds = set()
        for d in by_match[mn]:
            f = logs / d / "laya_points.jsonl"
            if not f.exists():
                st["run_without_points"] += 1
                continue
            online = []
            for line in f.open(encoding="utf-8"):
                try:
                    online.append(dict(json.loads(line), run=d))
                except ValueError:
                    st["bad_json"] += 1
            ecos = [p for p in online if p.get("kind") == "eco" and p.get("question")]
            if not ecos:
                continue
            # 路线配置：用线上原文反推（按这一目录线上题面的多数版本渲染比对）
            vers = collections.Counter(p.get("feature_version") for p in online if p.get("kind") != "eco" and p.get("feature_version"))
            set_ver(vers.most_common(1)[0][0] if vers else ver)
            log_cfg = {k: v for k, v in (cfgs.get(d) or {}).items() if k != "log"} or None
            cfg, hit = BR.pick_cfg(online, log_cfg)
            st["cfg_hit_sum"] += max(hit, 0)
            BR.apply_cfg(cfg)
            set_ver(ver)
            hs = {}
            if not args.no_hindsight:
                try:
                    hs = {q["round"]: q for q in BD.extract_eco(logs / d)}
                except Exception as e:  # noqa: BLE001
                    st["hindsight_err"] += 1
                    print("extract_eco 出错", d, type(e).__name__, e)
            for p in sorted(ecos, key=lambda x: x.get("ts") or 0):
                rnd = p.get("round") or (p.get("ctx") or {}).get("round")
                if rnd in seen_rounds:
                    st["dup_round"] += 1
                    continue
                seen_rounds.add(rnd)
                onl = {"choice": p.get("laya_choice"), "probs": p.get("laya"), "model": p.get("model"),
                       "used": p.get("used"), "feature_version": p.get("feature_version"), "id": p.get("id")}
                old_q, old_raw, old_state = p["question"], p.get("raw") or {}, p.get("state")
                for k in STUDENT_KEYS:
                    p.pop(k, None)
                a = RR.rerender(dict(p))
                if a is None:
                    st["dropped_render_none"] += 1
                    continue
                ctx = eco_ctx(p)
                q2 = F.q_eco(ctx)
                if q2 != a["question"]:
                    st["warn_ctx_mismatch"] += 1          # eco_ctx 与 rerender 不同口径 → raw 与题面可能对不上
                a["raw"] = eco_raw(ctx)
                a["online_eco"] = onl
                a["id"] = BR.pid(d, p.get("ts"), "eco")
                a["source"] = "online_eco"
                a["render_cfg"] = cfg
                a["match"] = mn
                a["rank"] = rows.get(mn, {}).get("rank")
                a["feature_version"] = ver
                h = hs.get(rnd)
                if h:
                    a["rule_choice"] = h.get("rule_choice")
                    a["hindsight"] = dict(h.get("hindsight") or {}, final_rank=a["rank"], ban=rows.get(mn, {}).get("ban"))
                else:
                    st["no_hindsight"] += 1
                if list(a["question"]["criteria"]) != list(old_q.get("criteria") or {}):
                    st["keys_changed_vs_online"] += 1
                if a["question"] != old_q:
                    st["text_changed_vs_online"] += 1
                if a["state"] != old_state:
                    st["state_changed_vs_online"] += 1
                if [m[0] for m in a["raw"]["missing"]] != [m[0] for m in (old_raw.get("missing") or [])]:
                    st["missing_changed_vs_online"] += 1
                out.append(a)
        BR.apply_cfg(None)
    ms = sorted({p["match"] for p in out})
    test = set(ms[-args.test_matches:]) if args.test_matches else set()
    for p in out:
        p["split"] = TEST_SPLIT if p["match"] in test else "train"
    ids = [p["id"] for p in out]
    assert len(ids) == len(set(ids)), "题号重复"
    tg = {p["id"]: tags_of(p) for p in out}
    tagc = collections.Counter(f"{p['split']}:{t}" for p in out for t in tg[p["id"]])
    h = hashlib.md5()
    for p in out:
        h.update(json.dumps([p["id"], p["state"], p["question"], p["raw"]], ensure_ascii=False, sort_keys=True).encode())
    info = {
        "name": NAME, "feature_version": ver, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "min_match": args.min_match, "matches": len(ms), "match_range": [ms[0], ms[-1]] if ms else None,
        "test_matches": sorted(test), "points": len(out),
        "by_split": dict(collections.Counter(p["split"] for p in out)),
        "options_hist": dict(collections.Counter("/".join(p["options"]) for p in out)),
        "online_choice": dict(collections.Counter(str((p.get("online_eco") or {}).get("choice")) for p in out)),
        "online_version": dict(collections.Counter(str((p.get("online_eco") or {}).get("feature_version")) for p in out)),
        "rule_choice": dict(collections.Counter(str(p.get("rule_choice")) for p in out)),
        "by_round": dict(sorted(collections.Counter(p.get("round") for p in out).items(), key=lambda x: x[0] or 0)),
        "tags": dict(sorted(tagc.items())),
        "unmapped_runs_with_eco": unmapped_eco, "busy_runs": meta["busy"],
        "stats": dict(sorted(st.items())), "md5": h.hexdigest(), "secs": round(time.time() - t0, 1),
    }
    print(json.dumps(info, ensure_ascii=False, indent=1))
    if args.check:
        return 0
    od = PROJECT_DIR / "data/laya/datasets" / NAME
    (od / "targets").mkdir(parents=True, exist_ok=True)
    with (od / "points.jsonl").open("w", encoding="utf-8") as fh:
        for p in out:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (od / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (od / "matches.json").write_text(json.dumps({str(k): {"runs": by_match[k], "rank": rows.get(k, {}).get("rank"),
                                                        "split": TEST_SPLIT if k in test else "train"}
                                                 for k in sorted(by_match) if k in set(ms)}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    (od / "tags.json").write_text(json.dumps({i: t for i, t in tg.items() if t}, ensure_ascii=False), encoding="utf-8")
    by_t = collections.defaultdict(list)
    for p in out:
        if p["split"] != "train":
            for t in tg[p["id"]]:
                by_t[t].append(p["id"])
    for t, v in by_t.items():
        (od / "targets" / f"{t}.json").write_text(json.dumps(sorted(v)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
