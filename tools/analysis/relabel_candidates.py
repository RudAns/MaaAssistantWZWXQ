"""按 docs/analysis-top3-vs-bottom.md 第 37 节草案，统计历史题里哪些该重标（只计数 + 导出 id，不改数据）。

  python tools/analysis/relabel_candidates.py

数据源：data/laya/datasets/r9o/points.jsonl（训练集现状） + 第 49~85 局的 live_*/laya_points.jsonl（r9o 之后的新局）。
输出：各条件命中题数；id 清单写 data/analysis/relabel_candidates.json。
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUNSET_CORE_TALENTS = {"场外供能", "核心裂痕", "动能灌注", "积少成多", "定向传输", "能量汲取", "双重分配",
                       "能量改造", "光棱塔", "能源置换", "智能装甲"}
SUNSET_ENGINE_HEROES = {"海诺", "露娜", "朵莉亚", "米莱狄", "安琪拉", "亚瑟", "狂铁", "马可波罗"}
SUNSET_CARRIES = {"亚连", "安琪拉"}


def lit(v):
    if isinstance(v, (dict, list)):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:   # noqa: BLE001
        return {}


def load():
    pts = []
    seen = set()
    p = ROOT / "data" / "laya" / "datasets" / "r9o" / "points.jsonl"
    for line in p.open(encoding="utf-8"):
        x = json.loads(line)
        x["_src"] = "r9o"
        seen.add(x.get("id"))
        pts.append(x)
    for f in (ROOT / "data" / "analysis" / "match_features.jsonl").open(encoding="utf-8"):
        m = json.loads(f)
        lp = ROOT / "data" / "logs" / m["run"] / "laya_points.jsonl"
        if not lp.exists():
            continue
        for line in lp.open(encoding="utf-8"):
            x = json.loads(line)
            if x.get("id") in seen:
                continue
            x["_src"] = "live49-85"
            pts.append(x)
    return pts


def main():
    pts = load()
    kinds = Counter((x["_src"], x.get("kind")) for x in pts)
    print("题量：", dict(kinds))
    hits = defaultdict(list)
    for x in pts:
        k = x.get("kind")
        q = lit(x.get("question"))
        crit = q.get("criteria", {}) if isinstance(q, dict) else {}
        raw = lit(x.get("raw"))
        ctx = lit(x.get("ctx"))
        st = str(x.get("state") or "")
        rnd = int(x.get("round") or 0)
        carry = (ctx or {}).get("carry") or ""
        ban = str((ctx or {}).get("banned") or "")
        txt = json.dumps(crit, ensure_ascii=False)
        if k == "talent":
            names = [t[0] for t in (raw or {}).get("talents", [])] if isinstance(raw, dict) else []
            if "refresh" in crit:
                hits["T1 天赋题有 refresh 选项"].append(x["id"])
            if carry in SUNSET_CARRIES or "亚连" in st[:600] and "future carry" in st:
                if any(n in SUNSET_CORE_TALENTS for n in names):
                    hits["T2 日落海主C 且选项含核心类天赋"].append(x["id"])
        elif k == "auction":
            lots = re.findall(r"[一-龥]{2,5}", txt)
            if carry in SUNSET_CARRIES and any(n in SUNSET_ENGINE_HEROES for n in lots):
                hits["A1 日落海主C 且拍品含引擎件"].append(x["id"])
            if "百里守约" in txt and carry != "百里守约" and "河洛" not in ban and rnd <= 4:
                hits["A2 第一次拍卖有守约、主C 不是守约、河洛没禁"].append(x["id"])
        elif k == "shop":
            shop = [s[1] for s in (raw or {}).get("shop", [])] if isinstance(raw, dict) else []
            if "百里守约" in shop and carry != "百里守约" and "河洛" not in ban and rnd <= 6:
                hits["S1 第 3~6 回合商店有守约、主C 不是守约、河洛没禁"].append(x["id"])
            if carry in SUNSET_CARRIES and any(n in SUNSET_ENGINE_HEROES - {carry} for n in shop) and 3 <= rnd <= 8:
                hits["S2 日落海主C、第 3~8 回合商店有引擎件"].append(x["id"])
            if 9 <= rnd <= 10 and "refresh" in crit:
                hits["S3 第 9~10 回合商店题（中期掉血窗口）"].append(x["id"])
        elif k == "use_card":
            if 8 <= rnd <= 10 and carry and carry not in txt:
                hits["U1 第 8~10 回合用牌题、选项里没有主C"].append(x["id"])
            elif 8 <= rnd <= 10 and carry:
                hits["U2 第 8~10 回合用牌题、主C 在选项里"].append(x["id"])
        elif k in ("carry", "swap"):
            if "百里守约" in txt and carry != "百里守约" and rnd <= 6:
                hits[f"C1 {k} 题候选有守约、现任不是守约、≤R6"].append(x["id"])
            if carry in SUNSET_CARRIES and "河洛" in ban:
                hits[f"C2 {k} 题：河洛被禁、现任是日落海主C"].append(x["id"])
    out = {}
    for name in sorted(hits):
        ids = hits[name]
        src = Counter(next(p["_src"] for p in pts if p["id"] == i) for i in ids[:5000])
        print(f"{name}: {len(ids)} 题 {dict(src)}")
        out[name] = ids
    (ROOT / "data" / "analysis" / "relabel_candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
