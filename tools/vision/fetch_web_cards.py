"""下载官网完整卡面图（cardImage，游戏内卡面原样）→ data/vision/web_cards/<卡名>[_awk].png，给手牌立绘模型合成训练样本。

  .venv/Scripts/python.exe -X utf8 tools/vision/fetch_web_cards.py

来源：data/web/{英雄牌,效果牌,装备牌,天赋牌,棋手牌}.raw.json 的 cardImage；英雄另有 heroCard.awakeingCard.cardImage（觉醒版）。
previewCards（赠送牌）里的也收。已下载的跳过。清单写 data/vision/web_cards/index.json（卡名、类别、是否觉醒、url）。
"""
from __future__ import annotations

import ast
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "vision" / "web_cards"


def _obj(v):
    if isinstance(v, (dict, list)) or v is None:
        return v
    try:
        return ast.literal_eval(v)
    except Exception:                                       # noqa: BLE001
        try:
            return json.loads(v)
        except Exception:                                   # noqa: BLE001
            return None


def collect() -> list[dict]:
    out, seen = [], set()

    def add(name, url, cat, awk):
        if not name or not url or not str(url).startswith("http"):
            return
        key = (name, awk)
        if key in seen:
            return
        seen.add(key)
        out.append({"name": name, "url": url, "cat": cat, "awk": awk})

    def walk(it, cat):
        add(it.get("name"), it.get("cardImage"), cat, False)
        hc = _obj(it.get("heroCard"))
        if isinstance(hc, dict):
            ak = hc.get("awakeingCard")
            if isinstance(ak, dict):
                add(it.get("name"), ak.get("cardImage"), cat, True)
        pv = _obj(it.get("previewCards"))
        if isinstance(pv, list):
            for p in pv:
                if isinstance(p, dict):
                    walk(p, cat + "/赠送")

    for cat in ("英雄牌", "效果牌", "装备牌", "天赋牌", "棋手牌"):
        d = json.loads((ROOT / "data" / "web" / f"{cat}.raw.json").read_text(encoding="utf-8"))
        for v in d.values():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        walk(it, cat)
    return out


def fname(e) -> Path:
    safe = e["name"].replace("/", "_").replace("\\", "_").replace(":", "_")
    return OUT / f"{safe}{'_awk' if e['awk'] else ''}.png"


def get(e) -> str:
    p = fname(e)
    if p.exists() and p.stat().st_size > 1000:
        return "skip"
    for k in range(3):
        try:
            req = urllib.request.Request(e["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                p.write_bytes(r.read())
            return "ok"
        except Exception as ex:                             # noqa: BLE001
            err = str(ex)
            time.sleep(1 + k)
    return "fail " + err


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    es = collect()
    print("URL", len(es), "其中觉醒版", sum(e["awk"] for e in es), flush=True)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(get, es))
    for e, r in zip(es, res):
        e["file"] = fname(e).name
        e["status"] = r
    (OUT / "index.json").write_text(json.dumps(es, ensure_ascii=False, indent=0), encoding="utf-8")
    from collections import Counter
    print(Counter(r.split()[0] for r in res))
    for e, r in zip(es, res):
        if r.startswith("fail"):
            print(e["name"], r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
