"""r9q 数据集（2026-10-04，r9p + 第 117~214 局）。复制自 build_r9p：r9p 数据集（f9l，2026-10-02）：r9o_g9m 原封不动 + 之后新打的对局（第 60~116 局，主要是钻石段）里的决策点。由 build_r9o.py 复制改名。

  python tools/laya/build_r9n.py --base r9m --out r9n

- r9m 的 8705 题原样保留（id / 划分 / 题面都不动；r9m 测试集 299 题不变，便于和 R15/R14/cgnet 对比）。
- 新对局 = data/logs/live_* 里不在 r9m、晚于 r9m 最后一局、且有决策点的运行目录；正在写的目录（states.jsonl 最近
  --busy-min 分钟内还在变）视为进行中的对局，整局排除。
- 同一局中途重启的多个目录合并成一局：下一个目录第一回合 >1 且不比上一目录最后回合小 2 以上、间隔 <30 分钟，
  或上一目录只打到第 1 回合、下一目录又从第 1 回合开始且间隔 <2 分钟。
- 出题方式同 r9m（build_r9m.py）：线上题（laya_points.jsonl，剥掉学生答案，按 f9k 重渲染）+ 离线 route 题（build_dataset.extract_run）。
  额外：离线 carry 题（extract_run，每回合一题）只补"线上这一回合没出 carry 题"的回合，单独成池（source=offline_carry），
  单独切批（carryx_*），由调用方决定标不标 —— 线上 carry 题路线锁定后常被限到 <2 个候选而不出（laya_policy._shadow_carry）。
- 去重：同一局同一回合同一题型同一组选项文本只留一题（build_dataset 的口径）；shop 另按"同一局同一回合同一商店内容（卡名集合）"只留一题。
- 划分：新对局按整局洗牌（seed 固定），约 15% 进新测试集 split="test_r9n_new"，其余 "train"。
- 抽样：非 shop 全量；shop 按局按回合均匀抽约 --shop 道（轮转：每局每回合各抽一题，直到配额用完）。
- 输出：datasets/<out>/points.jsonl（r9m 全部 + 新抽中的题）、new_ids.json（需要标的新题，不含 carryx 池）、
  carryx_ids.json（离线 carry 补充池）、unsampled_shop.jsonl（没抽中的 shop，留作后用）、matches.json、info.json。
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
TEST_SPLIT = "test_r9q_new"


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
    ap.add_argument("--base", default="r9p")
    ap.add_argument("--out", default="r9q")
    ap.add_argument("--shop", type=int, default=900)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260930)
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
    for d in sorted((PROJECT_DIR / "data" / "logs").glob("live_*")):
        if not d.is_dir() or d.name in base_runs or d.name <= last_base:
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
                if p.get("kind") == "route":
                    continue                            # route 题用离线出的（同 r9m）
                for k in STUDENT_KEYS:
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
            if p["kind"] not in ("route", "carry"):
                continue
            p["id"] = pid(p["run"], p.get("ts"), p["kind"])
            p["feature_version"] = F.ACTIVE
            p["source"] = "offline_" + p["kind"]
            p["render_cfg"] = cfg_rec
            (offline_route if p["kind"] == "route" else offline_carry).append(p)
    apply_cfg(None)

    match_of = {r: i for i, g in enumerate(groups) for r in g}
    # 离线 carry 只补线上这一回合没出 carry 题的回合
    online_carry_rounds = {(match_of[p["run"]], p.get("round")) for p in online if p["kind"] == "carry"}
    offline_carry = [p for p in offline_carry if (match_of[p["run"]], p.get("round")) not in online_carry_rounds]

    allnew = online + offline_route + offline_carry
    for p in allnew:
        p["match"] = f"r9q_m{match_of[p['run']]:03d}"
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

    out = base + keep
    od = ds / args.out
    od.mkdir(parents=True, exist_ok=True)
    with (od / "points.jsonl").open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (od / "unsampled_shop.jsonl").open("w", encoding="utf-8") as fh:
        for r in unsampled:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    need = sorted(p["id"] for p in keep if p["source"] != "offline_carry")
    carryx = sorted(p["id"] for p in keep if p["source"] == "offline_carry")
    (od / "new_ids.json").write_text(json.dumps(need), encoding="utf-8")
    (od / "carryx_ids.json").write_text(json.dumps(carryx), encoding="utf-8")
    matches = [{"match": f"r9q_m{i:03d}", "runs": g, "split": TEST_SPLIT if f"r9q_m{i:03d}" in test_m else "train",
                "points": sum(1 for p in keep if p["match"] == f"r9q_m{i:03d}")} for i, g in enumerate(groups)]
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
