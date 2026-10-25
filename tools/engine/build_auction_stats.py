"""拍卖的实测分布（2026-10-14）→ data/engine/auction_stats.json，推演器按它结算拍卖。

规则（官方新手指引 + 腾讯新闻拍卖详解）：拍品数 = 剩余人数，起拍 1、每次 +1，价高者得、同价先出者得，每人只能拍到 1 张；
拍不到不扣钱；有效出价进分红池，结束后分给所有人（血越少分得越多）；等待期间每隔几秒掉落（能量每次 1 点、英雄牌只掉 1~2 阶）；
拍卖结束随机给 1 件基础装备（推演器在 EQUIP_DROP_ROUNDS 里给，这里不重复）。

统计（只用画面读数，不含我们的选择）：
  contested    拍品被人出过价（最高出价 ≥1）的比例
  price        有人出价的拍品，最终最高出价的分布
  energy_delta 拍卖期间我方能量的变化（首读 → 末读，去掉含下回合收入到账的读数），= 分红 + 掉落 − 自己的成交价，只作量级
  .venv/Scripts/python.exe -X utf8 tools/engine/build_auction_stats.py
"""
import collections
import glob
import json
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    per = collections.defaultdict(lambda: {"gold": [], "tops": {}, "won": False})
    for path in sorted(glob.glob(str(ROOT / "data/logs/live_*/states.jsonl")))[-400:]:
        for l in open(path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            st = r.get("state") or {}
            if st.get("phase") != "auction":
                continue
            k = (path, st.get("round_no"))
            a = st.get("auction") or {}
            if isinstance(a.get("my_gold"), (int, float)) and 0 <= a["my_gold"] < 100:
                per[k]["gold"].append(a["my_gold"])
            for lot in a.get("lots") or []:
                p, tb = lot.get("piece"), lot.get("top_bid")
                if p and isinstance(tb, (int, float)) and 0 <= tb < 30:
                    per[k]["tops"][p] = max(per[k]["tops"].get(p, 0), tb)
    tops = [t for v in per.values() for t in v["tops"].values()]
    contested = [t for t in tops if t >= 1]
    deltas = [v["gold"][-1] - v["gold"][0] for v in per.values() if len(v["gold"]) >= 2]
    deltas = [d for d in deltas if -6 <= d <= 8]          # 大幅负值 / 正值多半是下回合收入到账或读错
    out = {
        "n_auctions": len(per), "n_lots": len(tops),
        "contested": round(len(contested) / max(1, len(tops)), 3),
        "price": dict(sorted(collections.Counter(contested).items())),
        "energy_delta": {"median": S.median(deltas) if deltas else 0, "mean": round(S.mean(deltas), 2) if deltas else 0,
                         "dist": dict(collections.Counter(deltas).most_common())},
    }
    (ROOT / "data/engine/auction_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
