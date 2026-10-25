"""官网推荐阵容爬虫：把 teamlist 页的阵容数据结构化成 data/lineups.json。

数据源（teamlist.js 前端直接加载的 JS 数组，纯 HTTP）：
  https://wxq.qq.com/cp/a20260707sfgw/teamlist.html
  - 598252_oslineupbyrecommend_pro_<页>.js  综合推荐
  - 600264_oslineupbyhot_pro_<页>.js        热门推荐
  - 600269_oslineupbybeginner_pro_<页>.js   新手推荐（默认不抓：强度低，明确排除）

每套阵容含：棋手、英雄阵容（带棋盘坐标/装备/是否觉醒）、推荐装备、天赋/效果牌取舍、运营思路。

用法:
  python tools/fetch_lineups.py                 # 抓综合+热门
  python tools/fetch_lineups.py --lord 阿离     # 只保留能用该棋手的阵容
  python tools/fetch_lineups.py --include-beginner
  python tools/fetch_lineups.py --offline       # 用 data/web/ 里已下载的原始数据重新转换
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "data" / "web"
OUT = ROOT / "data" / "lineups.json"

SOURCES = {
    "综合推荐": "https://game.gtimg.cn/images/amside/ide_timer/598252_oslineupbyrecommend_pro_{}.js",
    "热门推荐": "https://game.gtimg.cn/images/amside/ide_timer/600264_oslineupbyhot_pro_{}.js",
    "新手推荐": "https://game.gtimg.cn/images/amside/ide_timer/600269_oslineupbybeginner_pro_{}.js",
}
MAX_PAGES = 30
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def strip_html(text: str) -> str:
    return _HTML_TAG.sub("", text or "").strip()


def parse_js_array(raw: str) -> list:
    """这些 .js 是 `xxx = [...]` 或裸 JSON 数组。"""
    m = re.search(r"=\s*(\[.*\])\s*;?\s*$", raw, re.S)
    return json.loads(m.group(1) if m else raw)


def fetch_source(name: str, url_tpl: str, offline: bool) -> list:
    cache = WEB_DIR / f"lineup_{name}.raw.json"
    if offline:
        if not cache.exists():
            print(f"  {name}: 无缓存，跳过")
            return []
        return json.loads(cache.read_text(encoding="utf-8"))

    rows: list = []
    for page in range(1, MAX_PAGES + 1):
        req = urllib.request.Request(url_tpl.format(page),
                                     headers={"User-Agent": "Mozilla/5.0"})
        try:
            raw = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "ignore")
        except Exception:
            break            # 翻到不存在的页就结束
        try:
            data = parse_js_array(raw)
        except Exception as e:
            print(f"  {name} 第 {page} 页解析失败: {e}")
            break
        if not data:
            break
        rows.extend(data)
        print(f"  {name} 第 {page} 页: {len(data)} 套", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def _infer_carry(heroes: list[dict], title: str) -> str | None:
    """推断主C。官方数据没有显式字段，用装备数 + 标题命中来判。"""
    if not heroes:
        return None
    max_eq = max(len(h["equips"]) for h in heroes)
    if max_eq > 0:
        top = [h for h in heroes if len(h["equips"]) == max_eq]
        if len(top) == 1:
            return top[0]["name"]
        for h in top:                       # 并列时看标题
            if h["name"] and h["name"] in title:
                return h["name"]
        return top[0]["name"]
    for h in heroes:
        if h["name"] and h["name"] in title:
            return h["name"]
    return heroes[0]["name"]


def convert(item: dict, source: str) -> dict:
    """一套阵容 -> 结构化记录。

    heroList 里的 positionX/positionZ 是**官方给的棋盘站位**，
    equipList 是该英雄身上的推荐装备——这两样是决策层最值钱的部分。
    """
    heroes = []
    for h in item.get("heroList") or []:
        heroes.append({
            "name": h.get("name"),
            "level": h.get("level"),
            "quality": h.get("cardQuality"),
            "is_awakened": bool(h.get("isEvo")),
            "pos": [h.get("positionX"), h.get("positionZ")],
            "equips": [e.get("name") for e in (h.get("equipList") or []) if e.get("name")],
            "desc": strip_html(h.get("desc") or ""),
        })
    lords = [{"id": l.get("id"), "name": l.get("name")} for l in item.get("lordList") or []]
    return {
        "key": item.get("key"),
        "name": item.get("name"),
        "source": source,
        "use_num": item.get("useNum", 0),
        "lords": lords,
        "lord_names": [l["name"] for l in lords],
        "heroes": heroes,
        # 主C推断：官方数据没有显式标记。列表首位**不可靠**
        # （"一枪清屏百里守约"的首位是明世隐），改用三级判据：
        #   1. 装备最多的英雄（推荐装备都堆给主C）
        #   2. 名字出现在阵容标题里的英雄
        #   3. 兜底取首位
        "carry": _infer_carry(heroes, item.get("name") or ""),
        "tags": item.get("tags") or ([item["tag"]] if item.get("tag") else []),
        "brief": strip_html(item.get("brief") or ""),
        "position_desc": strip_html(item.get("positionDesc") or ""),
        "equip_desc": strip_html(item.get("equipDesc") or ""),
        "talent_desc": strip_html(item.get("talentCardDesc") or ""),
        "effect_desc": strip_html(item.get("effectCardDesc") or ""),
        "talent_cards": [c.get("name") for c in (item.get("talentCardList") or []) if c.get("name")],
        "effect_cards": [c.get("name") for c in (item.get("effectCardList") or []) if c.get("name")],
        # operationDetails 是**分回合运营路线**：每阶段该上哪些英雄 + 文字说明。
        # 例："回合1-3 弈星卖血，二回合升二级找守约" / "回合8-14 李元芳卖掉换太乙"
        "stages": [{
            "from_round": o.get("beginRound"),
            "to_round": o.get("endRound"),
            "main": [h.get("name") for h in (o.get("mainHeroList") or []) if h.get("name")],
            "sub": [h.get("name") for h in (o.get("subHeroList") or []) if h.get("name")],
            "desc": strip_html(o.get("desc") or ""),
        } for o in (item.get("operationDetails") or [])],
        "author": (item.get("author") or {}).get("name") if isinstance(item.get("author"), dict)
                  else item.get("author"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lord", help="只保留能用该棋手的阵容（如 阿离）")
    ap.add_argument("--include-beginner", action="store_true",
                    help="连新手推荐一起抓（默认不抓：强度低）")
    ap.add_argument("--offline", action="store_true", help="用 data/web/ 缓存重新转换")
    args = ap.parse_args()

    wanted = ["综合推荐", "热门推荐"] + (["新手推荐"] if args.include_beginner else [])
    rows: list[dict] = []
    seen: set = set()
    for name in wanted:
        for item in fetch_source(name, SOURCES[name], args.offline):
            rec = convert(item, name)
            if not rec["key"] or rec["key"] in seen:
                continue        # 同一套可能同时出现在综合和热门里
            seen.add(rec["key"])
            rows.append(rec)

    if args.lord:
        before = len(rows)
        rows = [r for r in rows if args.lord in r["lord_names"]]
        print(f"按棋手「{args.lord}」筛选: {before} -> {len(rows)} 套")

    rows.sort(key=lambda r: -(r["use_num"] or 0))
    OUT.write_text(json.dumps({
        "_comment": "官网 teamlist 推荐阵容。use_num 降序（热度）。"
                    "carry=英雄列表首位（官方按核心排序）；heroes[].pos 是官方站位；"
                    "heroes[].equips 是该英雄的推荐装备。重跑 tools/fetch_lineups.py 更新。",
        "sources": wanted,
        "count": len(rows),
        "lineups": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共 {len(rows)} 套 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
