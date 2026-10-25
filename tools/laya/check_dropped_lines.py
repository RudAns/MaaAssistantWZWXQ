"""选项行被 token 预算丢掉的检查（2026-10-21 第 881 局：文字太长超出 token 预算，用工具检查）。

  python tools/laya/check_dropped_lines.py --live 881            # 某局线上题（data/logs/agent_one_<局号> 对应的 live 目录）
  python tools/laya/check_dropped_lines.py --live-dir data/logs/live_xxx
  python tools/laya/check_dropped_lines.py --ds r9zzg --limit 2000

用 rerender 按题面原版本重渲染每道题，拦截 facts.pack：记下"算出来了、但放不进选项预算（OPT_TOKENS）被跳过 / 截断"的行，
按行的模板（数字、中文名归一）汇总：被丢次数、涉及题型、示例。重点看收益行 / 适配行 / AVOID 这类结论被丢的。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import facts as FA  # noqa: E402
import features as F  # noqa: E402
import rerender as RR  # noqa: E402

_DROPS: list = []


def _spy_pack(orig):
    def pack(head, items, budget=FA.OPT_TOKENS):
        out = orig(head, items, budget)
        for it in items:
            if it and it not in out:
                _DROPS.append((head, it, FA.est_tokens(head + it)))
        return out
    return pack


def _tmpl(s: str) -> str:
    s = re.sub(r"[+\-−]?\d+(\.\d+)?", "N", s)
    s = re.sub(r"[一-鿿]+", "X", s)
    return s[:70]


def _live_dir(match: str) -> Path:
    log = ROOT / "data" / "logs" / f"agent_one_{match}.log"
    txt = log.read_bytes().decode("utf-8", "ignore")
    m = re.findall(r"(data[\\/]logs[\\/]live_\d+)", txt)
    if m:
        return ROOT / m[0].replace("\\", "/")
    # 退回：日志修改时间之前最新的 live 目录
    cands = sorted((ROOT / "data" / "logs").glob("live_*"), key=lambda p: p.stat().st_mtime)
    return [c for c in cands if c.stat().st_mtime <= log.stat().st_mtime + 60][-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", help="局号")
    g.add_argument("--live-dir")
    g.add_argument("--ds")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--show", type=int, default=2, help="每个模板打印几个示例")
    ap.add_argument("--ver", default="", help="强制按这个题面版本渲染（默认按每道题的 feature_version）")
    ap.add_argument("--no-fallback", action="store_true", help="f9zb：关掉 pack 的缩写 / 截断兜底，只看源头写短了没有")
    args = ap.parse_args()
    if args.ds:
        src = ROOT / "data" / "laya" / "datasets" / args.ds / "points.jsonl"
    else:
        src = (Path(args.live_dir) if args.live_dir else _live_dir(args.live)) / "laya_points.jsonl"
    pts = []
    for l in src.open(encoding="utf-8", errors="ignore"):
        try:
            pts.append(json.loads(l.replace("\x00", "")))
        except Exception:
            pass
    if args.limit and args.limit < len(pts):
        import random
        pts = random.Random(0).sample(pts, args.limit)
    FA.pack = _spy_pack(FA.pack)
    if args.no_fallback and hasattr(FA, "ZB_FALLBACK"):
        FA.ZB_FALLBACK = False
    nfb = collections.Counter()
    agg = collections.defaultdict(lambda: {"n": 0, "kinds": collections.Counter(), "ex": []})
    nq = nd = 0
    for p in pts:
        ver = args.ver or p.get("feature_version") or F.ACTIVE
        F.ACTIVE = ver
        _DROPS.clear()
        try:
            r = RR.rerender(p)
        except Exception as e:      # 渲染失败的题跳过，计数
            agg["<rerender error> " + type(e).__name__]["n"] += 1
            continue
        if r is None:
            continue
        nq += 1
        seen = set()
        finals = [v for v in (r["question"].get("criteria") or {}).values() if isinstance(v, str)]
        for head, it, tk in _DROPS:
            if (head, it) in seen:
                continue
            # pack 会被试算多次：只算最终选项文本里（同一开头）确实没有的那条
            mine = [c for c in finals if c.startswith(head)]
            if not mine or any(it in c for c in mine):
                continue
            if FA.FV >= getattr(FA, "F9ZB", "~") and getattr(FA, "ZB_FALLBACK", False):
                sh = FA.zb_short(head, it)
                if sh != it and any(sh in c for c in mine):
                    nfb["截断" if sh.endswith("…") else "缩写/去括号"] += 1      # f9zb 兜底写短后放进去了（不算丢）
                    continue
            seen.add((head, it))
            nd += 1
            a = agg[_tmpl(it)]
            a["n"] += 1
            a["kinds"][p.get("kind")] += 1
            if FA.FV >= getattr(FA, "F9ZB", "~") and getattr(FA, "ZB_FALLBACK", False):
                tk = FA.est_tokens(head + FA.zb_short(head, it))       # f9zb：pack 兜底会先写短，按写短后的长度判"永远放不进"
            a["over"] = a.get("over", 0) + (tk > FA.OPT_TOKENS)      # 单条就超预算：永远放不进去
            if len(a["ex"]) < args.show:
                a["ex"].append(f"r{p.get('round')} {head.strip()} | 丢掉：{it}（≈{tk} tok）")
    print(f"来源 {src}｜重渲染 {nq} 题｜被预算丢掉的行 {nd} 条（预算 {FA.OPT_TOKENS} token/选项）"
          + (f"｜f9zb 兜底写短后放进 {dict(nfb)}" if nfb else ""))
    nev = [(t, a) for t, a in agg.items() if a.get("over")]
    print(f"\n== 单条就超预算、永远放不进选项的模板 {len(nev)} 个（必须改短）==")
    for t, a in sorted(nev, key=lambda kv: -kv[1]["over"]):
        print(f"  [{a['over']}] {t}   题型 {dict(a['kinds'])}")
        for e in a["ex"][:1]:
            print("      " + e[:260])
    print("\n== 被挤掉最多的模板 ==")
    for t, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"])[:args.top]:
        print(f"\n[{a['n']}] {t}   题型 {dict(a['kinds'])}")
        for e in a["ex"]:
            print("    " + e[:260])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
