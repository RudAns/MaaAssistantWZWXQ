"""r9s 数据集（2026-10-08 夜；复制自 build_r9r）：r9r 原封不动 + 2026-10-08 平衡调整后的对局里的 **carry / swap** 题（只给局面网络 cgnet 用）。

  .venv/Scripts/python.exe -X utf8 tools/laya/build_r9s.py

- 对局来源：data/laya/iterations/batch2_matches.jsonl 里 match >= 426、没有 note 的局（note = 重启重连 / 不计统计的局）。
- 出题：线上题（laya_points.jsonl 的 carry / swap，剥掉学生 / cgnet 答案，按 f9l 重渲染 —— 线上 R17 是 f9l）
  + 离线 carry / swap 题（build_dataset.extract_run → carry_point / swap_point，只补线上这一回合没出该题型的回合）。
- swap 训练题若线上被删掉了 bench，补回 bench（与 r9r_add_bench 同口径）。
- 划分：整局洗牌，--test-frac 的局进 test_r9s_new。
- 输出：datasets/r9s/points.jsonl（r9r 全部 + 新题）、new_ids.json、matches.json、info.json。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_DIR))

import build_dataset as BD  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402
from agent.decision import routes as ROUTES  # noqa: E402

STUDENT_KEYS = ("laya", "laya_choice", "laya_conf", "laya_ms", "used", "why", "ckpt")
TEST_SPLIT = "test_r9s_new"
KINDS = ("carry", "swap")
MIN_MATCH = 426


def pid(run, ts, kind) -> str:
    return hashlib.md5(f"{run}|{ts}|{kind}".encode()).hexdigest()[:12]


def run_span(d: Path) -> dict:
    t0 = t1 = None
    rounds = []
    for line in (d / "states.jsonl").open(encoding="utf-8"):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        ts = x.get("ts")
        if ts:
            t0 = t0 or ts
            t1 = ts
        s = x.get("state") or {}
        r = s.get("round_no")
        if r and s.get("phase") == "prepare":
            rounds.append(r)
    return {"t0": t0, "t1": t1, "r_first": rounds[0] if rounds else None, "r_last": rounds[-1] if rounds else None}


def group_runs(spans: list[tuple[str, dict]]) -> list[list[str]]:
    groups, prev = [], None
    for name, sp in spans:
        cont = False
        if prev is not None and sp["t0"] and prev["t1"]:
            gap = sp["t0"] - prev["t1"]
            if gap < 1800 and sp["r_first"] and prev["r_last"]:
                if sp["r_first"] > 1 and sp["r_first"] >= prev["r_last"] - 1:
                    cont = True                     # 中途重启：接着上一目录的回合打
                elif sp["r_first"] == 1 and prev["r_last"] == 1 and gap < 120:
                    cont = True                     # 第 1 回合就重启
        if cont:
            groups[-1].append(name)
        else:
            groups.append([name])
        if sp["t1"]:
            prev = sp
    return groups


def run_render_cfgs() -> dict:
    """每个运行目录当时的路线配置（live_match_agent --carry-tiers）：从 data/logs/agent_*.log 里认
    「主C 品阶定调：只走 (2, 3) 阶主C 的路线」；认到的 run 在实战里 routes.CARRY_TIERS=(2,3)、EXCLUDE_CARRIES={公孙离}。
    题面里的阵容列表 / 路线候选依赖这两个全局量 —— 不按当时的配置渲染，题面就和线上学生看到的不一样。"""
    import re
    out = {}
    for f in sorted((PROJECT_DIR / "data" / "logs").glob("agent_*.log")):
        b = f.read_bytes().decode("utf-8", "replace")
        m = re.search(r"主C 品阶定调：只走 \(([^)]*)\)", b)
        if not m:
            continue
        tiers = [int(x) for x in m.group(1).split(",") if x.strip()]
        for r in set(re.findall(r"live_17\d{8}", b)):
            out[r] = {"carry_tiers": tiers, "exclude_carries": ["公孙离"], "log": f.name}
    return out


def apply_cfg(cfg: dict | None) -> None:
    ROUTES.CARRY_TIERS = tuple(cfg["carry_tiers"]) if cfg else None
    ROUTES.EXCLUDE_CARRIES = frozenset(cfg["exclude_carries"]) if cfg else frozenset()


CFG_CANDIDATES = (None, {"carry_tiers": [2, 3], "exclude_carries": []},
                  {"carry_tiers": [2, 3], "exclude_carries": ["公孙离"]})


def pick_cfg(online_rows: list, log_cfg: dict | None) -> tuple:
    """按"哪套配置重渲染出的题面和线上原文逐字一致最多"选这一局的配置（抽前 40 道非 route 题）。
    日志只记了品阶定调，没记公孙离排除（09-30 下午才加）—— 所以用线上原文反推；平手时取日志推断的那套。"""
    rows = [p for p in online_rows if p.get("kind") != "route" and p.get("question")][:40]
    if not rows:
        return log_cfg, -1
    best, best_n = None, -1
    order = sorted(CFG_CANDIDATES, key=lambda c: 0 if (c or {}).get("carry_tiers") == (log_cfg or {}).get("carry_tiers")
                   and (c is None) == (log_cfg is None) and (c or {}).get("exclude_carries") == (log_cfg or {}).get("exclude_carries") else 1)
    for c in order:
        apply_cfg(c)
        n = 0
        for p in rows:
            r = RR.rerender(dict(p, run=p.get("run")))
            if r is not None and r["state"] == p.get("state"):
                n += 1
        if n > best_n:
            best, best_n = c, n
    apply_cfg(None)
    return best, best_n


def restrict_to(r: dict, keys: list) -> bool:
    """把重渲染的题限回线上学生当时看到的选项集合（laya_policy._restrict 删掉的选项不再加回来）。
    只在"线上选项 ⊂ 重渲染选项"时限；否则保留完整重渲染（返回 False）。共有选项文本是否一致另在统计里记。"""
    crit = r["question"]["criteria"]
    if not set(keys) <= set(crit) or len(keys) < 2:
        return False
    r["question"] = dict(r["question"], criteria={k: crit[k] for k in crit if k in set(keys)})
    r["options"], r["n_opt"] = list(r["question"]["criteria"]), len(r["question"]["criteria"])
    return True


def shop_key(p: dict) -> tuple:
    names = sorted(str(x[1]) for x in (p.get("raw") or {}).get("shop", []))
    return tuple(names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r9r")
    ap.add_argument("--out", default="r9s")
    ap.add_argument("--shop", type=int, default=900)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--busy-min", type=float, default=10.0, help="states.jsonl 最近多少分钟内还在写 = 进行中的对局，排除")
    args = ap.parse_args()
    RR.ALLOW_NEW_KEYS = True
    F.ACTIVE = "f9l"                     # r9o：新题按 f9l 渲染（线上 R16 就是 f9l）
    ds = PROJECT_DIR / "data" / "laya" / "datasets"
    base = [json.loads(l) for l in (ds / args.base / "points.jsonl").open(encoding="utf-8")]
    base_runs = {p["run"] for p in base}
    base_ids = {p["id"] for p in base}
    base_sig = {hashlib.md5((p["kind"] + "\x00" + p["state"] + "\x00" + json.dumps(p["question"]["criteria"], ensure_ascii=False,
                                                                                    sort_keys=True)).encode()).hexdigest()
                for p in base}
    last_base = max(base_runs)
    now = time.time()
    spans, skipped = [], {}
    newest = max(p for p in (PROJECT_DIR / "data" / "logs").glob("live_*") if p.is_dir())
    wanted, match_no = set(), {}
    for line in (PROJECT_DIR / "data" / "laya" / "iterations" / "batch2_matches.jsonl").open(encoding="utf-8"):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if (m.get("match") or 0) >= MIN_MATCH and not m.get("note") and m.get("run"):
            wanted.add(m["run"])
            match_no[m["run"]] = m["match"]
    for d in sorted((PROJECT_DIR / "data" / "logs").glob("live_*")):
        if not d.is_dir() or d.name not in wanted or d.name in base_runs:
            continue
        st = d / "states.jsonl"
        if not st.exists():
            continue
        if d == newest and now - st.stat().st_mtime < args.busy_min * 60:
            skipped[d.name] = "进行中（最新目录，states.jsonl 仍在写）"
            continue
        spans.append((d.name, run_span(d)))
    groups = group_runs(spans)
    cfgs = run_render_cfgs()
    cfg_used = {}

    # ---- 出题 ----
    online, offline_route, offline_carry = [], [], []
    stats = collections.Counter()
    for d, _ in spans:
        rd = PROJECT_DIR / "data" / "logs" / d
        f = rd / "laya_points.jsonl"
        rows = []
        if f.exists():
            for line in f.open(encoding="utf-8"):
                try:
                    rows.append(dict(json.loads(line), run=d))
                except ValueError:
                    stats["online_bad_json"] += 1
        log_cfg = {k: v for k, v in (cfgs.get(d) or {}).items() if k != "log"} or None
        cfg, hit = pick_cfg(rows, log_cfg)
        cfg_used[d] = {"cfg": cfg, "log_cfg": log_cfg, "sample_hit": hit}
        apply_cfg(cfg)
        cfg_rec = cfg
        if rows:
            for p in rows:
                if not p.get("question"):
                    continue
                stats["online_raw:" + p["kind"]] += 1
                if p.get("kind") not in KINDS:
                    continue                            # r9s 只要 carry / swap
                for k in STUDENT_KEYS + ("model",):
                    p.pop(k, None)
                p["run"] = d
                p["id"] = pid(d, p.get("ts"), p["kind"])
                p["source"] = "online"
                online_state, online_crit = p.get("state"), p["question"]["criteria"]
                r = RR.rerender(p)
                if r is None:
                    stats["online_dropped:" + p["kind"]] += 1
                    continue
                if list(r["question"]["criteria"]) != list(online_crit):
                    stats["online_keys_changed:" + p["kind"]] += 1
                    if restrict_to(r, list(online_crit)):
                        r["restricted_to_online"] = True
                        stats["online_restricted:" + p["kind"]] += 1
                same = r["state"] == online_state and r["question"]["criteria"] == online_crit
                stats["online_same_text" if same else "online_text_changed:" + p["kind"]] += 1
                r["render_cfg"] = cfg_rec
                online.append(r)
        try:
            pts = BD.extract_run(rd)
        except Exception as e:  # noqa: BLE001
            print("extract 失败", d, e)
            stats["extract_fail"] += 1
            continue
        for p in pts:
            if p["kind"] not in KINDS:
                continue
            p["id"] = pid(p["run"], p.get("ts"), p["kind"])
            p["feature_version"] = F.ACTIVE
            p["source"] = "offline_" + p["kind"]
            p["render_cfg"] = cfg_rec
            (offline_route if p["kind"] == "swap" else offline_carry).append(p)    # offline_route 这里装离线 swap
    apply_cfg(None)

    match_of = {r: i for i, g in enumerate(groups) for r in g}
    # 离线 carry 只补线上这一回合没出 carry 题的回合
    online_carry_rounds = {(match_of[p["run"]], p.get("round")) for p in online if p["kind"] == "carry"}
    offline_carry = [p for p in offline_carry if (match_of[p["run"]], p.get("round")) not in online_carry_rounds]
    online_swap_rounds = {(match_of[p["run"]], p.get("round")) for p in online if p["kind"] == "swap"}
    offline_route = [p for p in offline_route if (match_of[p["run"]], p.get("round")) not in online_swap_rounds]

    allnew = online + offline_route + offline_carry
    for p in allnew:
        p["match"] = f"r9s_m{match_no[p['run']]:03d}"
    # ---- 去重 ----
    allnew.sort(key=lambda p: (p["match"], p.get("ts") or 0))
    seen, seen_shop, uniq = set(), set(), []
    dup = collections.Counter()
    for p in allnew:
        if p["id"] in base_ids:
            dup["id_in_base:" + p["kind"]] += 1
            continue
        sig = hashlib.md5((p["kind"] + "\x00" + p["state"] + "\x00" + json.dumps(p["question"]["criteria"], ensure_ascii=False,
                                                                                 sort_keys=True)).encode()).hexdigest()
        if sig in base_sig:
            dup["text_in_base:" + p["kind"]] += 1
            continue
        key = (p["match"], p.get("round"), p["kind"], json.dumps(p["question"]["criteria"], ensure_ascii=False, sort_keys=True))
        if key in seen:
            dup["same_round_same_options:" + p["kind"]] += 1
            continue
        if p["kind"] == "shop":
            sk = (p["match"], p.get("round"), shop_key(p))
            if sk in seen_shop:
                dup["same_round_same_shop"] += 1
                continue
            seen_shop.add(sk)
        seen.add(key)
        uniq.append(p)
    ids = collections.Counter(p["id"] for p in uniq)
    assert max(ids.values()) == 1, [i for i, k in ids.items() if k > 1][:5]

    # ---- 划分（整局）----
    mids = sorted({p["match"] for p in uniq})
    rng = random.Random(args.seed)
    order = mids[:]
    rng.shuffle(order)
    n_test = max(1, round(len(order) * args.test_frac))
    test_m = set(order[:n_test])
    for p in uniq:
        p["split"] = TEST_SPLIT if p["match"] in test_m else "train"

    # ---- 抽样：shop 按局按回合轮转 ----
    shop = [p for p in uniq if p["kind"] == "shop"]
    buckets = collections.defaultdict(list)
    for p in shop:
        buckets[(p["match"], p.get("round"))].append(p)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    rng.shuffle(keys)
    chosen_shop = []
    while len(chosen_shop) < args.shop and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(chosen_shop) < args.shop:
                chosen_shop.append(buckets[k].pop())
    chosen_ids = {p["id"] for p in chosen_shop}
    unsampled = [p for p in shop if p["id"] not in chosen_ids]
    keep = [p for p in uniq if p["kind"] != "shop" or p["id"] in chosen_ids]
    # swap 训练题补回 bench（线上 _restrict 删过；与 r9r_add_bench 同口径）
    for i, p in enumerate(keep):
        if p["kind"] != "swap" or p["split"] != "train" or "bench" in p["question"]["criteria"]:
            continue
        apply_cfg(p.get("render_cfg"))
        r = RR.rerender(dict(p))
        if r is None or "bench" not in r["question"]["criteria"]:
            stats["bench_add_fail"] += 1
            continue
        ks = [k for k in r["question"]["criteria"] if k in set(p["options"]) | {"bench"}]
        if not restrict_to(r, ks):
            stats["bench_add_fail"] += 1
            continue
        for k in ("id", "match", "split", "source", "render_cfg"):
            r[k] = p[k]
        r["feature_version"] = F.ACTIVE
        r["bench_added"] = True
        keep[i] = r
        stats["bench_added"] += 1
    apply_cfg(None)

    out = base + keep
    od = ds / args.out
    od.mkdir(parents=True, exist_ok=True)
    with (od / "points.jsonl").open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (od / "unsampled_shop.jsonl").open("w", encoding="utf-8") as fh:
        for r in unsampled:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    need = sorted(p["id"] for p in keep)
    carryx = sorted(p["id"] for p in keep if p["source"] == "offline_carry")
    (od / "new_ids.json").write_text(json.dumps(need), encoding="utf-8")
    (od / "carryx_ids.json").write_text(json.dumps(carryx), encoding="utf-8")
    matches = [{"match": f"r9s_m{match_no[g[0]]:03d}", "runs": g, "split": TEST_SPLIT if f"r9s_m{match_no[g[0]]:03d}" in test_m else "train",
                "points": sum(1 for p in keep if p["match"] == f"r9s_m{match_no[g[0]]:03d}")} for g in groups]
    (od / "matches.json").write_text(json.dumps(matches, ensure_ascii=False, indent=0), encoding="utf-8")

    def by(rows, extra=None):
        c = collections.Counter((p["kind"], p["split"]) for p in rows)
        return {f"{k}:{s}": v for (k, s), v in sorted(c.items())}
    info = {"base": args.base, "feature_version": F.ACTIVE, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed, "base_points": len(base), "new_runs": len(spans), "skipped_runs": skipped,
            "matches": len(groups), "matches_with_points": len(mids), "test_matches": len(test_m),
            "restart_merged": sum(1 for g in groups if len(g) > 1),
            "runs_with_carry_tiers": sum(1 for d, _ in spans if d in cfgs),
            "render_cfg_by_run": {d: v["cfg"] for d, v in cfg_used.items()},
            "render_cfg_log_disagree": {d: v for d, v in cfg_used.items() if v["cfg"] != v["log_cfg"]},
            "stats": dict(stats), "dedupe": dict(dup),
            "uniq_by_kind": dict(collections.Counter(p["kind"] for p in uniq)),
            "shop_uniq": len(shop), "shop_sampled": len(chosen_shop), "shop_unsampled": len(unsampled),
            "kept_by_kind_split": by(keep),
            "need_label": len(need), "carryx_pool": len(carryx), "points": len(out),
            "by_split": dict(collections.Counter(p.get("split") for p in out))}
    (od / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
