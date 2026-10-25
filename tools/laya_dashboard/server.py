"""Laya 决策展示界面的本地服务（2026-09-25：展示 Laya 的各个选项及判断百分比，配卡面图标）。

  .venv/Scripts/python.exe tools/laya_dashboard/server.py --port 8765     # 然后浏览器开 http://localhost:8765

数据：每局 `data/logs/live_*/laya_points.jsonl`（agent/decision/laya_policy.py 写的：每个决策点的全部选项概率、
Laya 选择、规则选择、最终采用谁、原因、局面摘要）。对局进行中会不断追加，页面"实时"模式每 2 秒拉一次。
图片：本地 `assets/image/pieces/<类别>/<名>.png`（官网头像）；没有的用爬虫数据里的官方卡面图地址（302 跳转）；都没有给占位 SVG。
只读，不改任何对局数据。
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOGS = ROOT / "data" / "logs"


def _load_cards() -> dict:
    cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    kb = {}
    p = ROOT / "data" / "card_kb.json"
    if p.exists():
        kb = json.loads(p.read_text(encoding="utf-8"))["entries"]
    out = {}
    for n, c in cards.items():
        e = kb.get(n, {})
        out[n] = {"name": n, "category": c.get("category"), "quality": c.get("quality"), "faction": c.get("faction", ""),
                  "skill": (c.get("skill") or "").replace("\n", " "), "tags": c.get("tags"),
                  "routes": e.get("routes", []), "note": e.get("note", ""), "scope": e.get("scope", "")}
    return out


def _load_images() -> tuple[dict, dict]:
    local = {}
    for p in glob.glob(str(ROOT / "assets" / "image" / "pieces" / "*" / "*.png")):
        local.setdefault(Path(p).stem, p)
    remote = {}
    web = ROOT / "data" / "web"
    for f in ("英雄牌", "效果牌", "装备牌", "天赋牌", "棋手牌"):
        p = web / f"{f}.raw.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))

        def walk(o):
            if isinstance(o, dict):
                n = o.get("name")
                url = o.get("image") or o.get("thumb") or o.get("cardImage")
                if isinstance(n, str) and isinstance(url, str) and url.startswith("http"):
                    remote.setdefault(n, url)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
    return local, remote


CARDS = _load_cards()
LOCAL_IMG, REMOTE_IMG = _load_images()


def _parse(v):
    if isinstance(v, str) and v[:1] in "[{(":
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _clean_name(n: str) -> str:
    """OCR 读出的卡名常带前后杂字（'☆TA落英汲取'）：去掉非中文前缀后在卡牌库/图片表里找。"""
    if n in CARDS or n in LOCAL_IMG or n in REMOTE_IMG:
        return n
    m = re.sub(r"^[^一-鿿]+|[^一-鿿·]+$", "", n or "")
    return m or n


def _option(kind: str, key: str, raw: dict) -> dict:
    """把 buy0 / opt1 / refresh 这类选项键翻译成可展示的东西。"""
    special = {"refresh": ("刷新", "花能量刷新（天赋：换一批选项）"), "levelup": ("升级", "买经验升棋手等级"),
               "stop": ("结束", "本回合不再花钱"),
               "hold": ("先留着", "这张牌先不用（留在手里）")}
    if key in special:
        t, d = special[key]
        return {"key": key, "title": t, "desc": d, "special": key}
    if kind in ("carry", "swap"):
        # carry：keep / cN = 维持现任 / 换成某人；swap：bench / sN = 不换 / 卖掉某人（raw["names"] 记着每个键对应的英雄）
        who = (raw.get("names") or {}).get(key, "")
        if key == "bench":
            return {"key": key, "title": "不换", "desc": f"「{raw.get('incoming', '')}」先留在手里", "special": key}
        title = ("维持主C：" if key == "keep" else "换主C：") if kind == "carry" else "卖掉："
        cn = _clean_name(who)
        card = CARDS.get(cn, {})
        return {"key": key, "title": title + who, "card": cn,
                "desc": (f"给「{raw.get('incoming', '')}」腾位" if kind == "swap" else card.get("skill", "")),
                "quality": card.get("quality"), "category": card.get("category"), "faction": card.get("faction", ""),
                "img": f"/img/{quote(cn)}" if (cn in LOCAL_IMG or cn in REMOTE_IMG) else None}
    m = re.match(r"(buy|opt|use)(\d+)$", key)
    i = int(m.group(2)) if m else -1
    name, desc, cost = key, "", None
    if kind == "shop":
        for s in raw.get("shop", []):
            if s[0] == i:
                name, cost = s[1], s[2]
    elif kind in ("talent", "lord_boost"):
        lst = raw.get("talents") or raw.get("boosts") or []
        if 0 <= i < len(lst):
            name, desc = lst[i][0], lst[i][1]
    elif kind == "use_card":
        lst = raw.get("targets", [])
        if 0 <= i < len(lst):
            name, desc = lst[i], f"把「{raw.get('card', '')}」用在它身上"
    elif kind == "card_pick":
        lst = raw.get("cards", [])
        if 0 <= i < len(lst):
            name = lst[i]
    cn = _clean_name(name)
    card = CARDS.get(cn, {})
    return {"key": key, "title": name, "card": cn, "desc": desc or card.get("skill", ""), "cost": cost,
            "quality": card.get("quality"), "category": card.get("category"), "faction": card.get("faction", ""),
            "note": card.get("note", ""), "routes": card.get("routes", []),
            "img": f"/img/{quote(cn)}" if (cn in LOCAL_IMG or cn in REMOTE_IMG) else None}


def _runs() -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(LOGS / "live_*" / "laya_points.jsonl")), key=os.path.getmtime, reverse=True):
        n = sum(1 for _ in open(p, encoding="utf-8"))
        out.append({"run": Path(p).parent.name, "points": n, "mtime": os.path.getmtime(p)})
    return out


_CACHE: dict[str, tuple[float, int, list]] = {}


def _points(run: str) -> list[dict]:
    """页面每 2 秒拉一次：文件没变（修改时间和大小都相同）就直接返回上次解析的结果。"""
    if run == "latest":
        # 按目录名里的时间戳取**最新开的一局**（新局在第一个 Laya 决策之前还没有记录文件 → 返回空，页面显示"等待"）
        dirs = sorted((d for d in LOGS.glob("live_*") if d.is_dir() and re.fullmatch(r"live_\d+", d.name)),
                      key=lambda d: int(d.name[5:]))       # live_agent.log 之类的文件不算
        if not dirs:
            return []
        run = dirs[-1].name
    p = LOGS / run / "laya_points.jsonl"
    if not p.exists():
        return []
    st = p.stat()
    hit = _CACHE.get(run)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    res = _parse_points(p, run)
    _CACHE[run] = (st.st_mtime, st.st_size, res)
    if len(_CACHE) > 8:                      # 只留最近看过的几局
        _CACHE.pop(next(iter(_CACHE)))
    return res


_FACTS = None


def _identities(ctx: dict, board: dict, field: list) -> dict:
    """每个场上单位的 f6 身份标签（和 Laya 输入同源：tools/laya/facts.py）。
    面板原来显示规则引擎的 keep/late（"场上估值前 4 都算留"，几乎人人都标，2026-09-25 发现不对），现改为 f6 身份。"""
    global _FACTS
    try:
        if _FACTS is None:
            sys.path.insert(0, str(ROOT / "tools" / "laya"))
            import facts as _F
            _FACTS = _F
        c = dict(ctx or {})
        c["field"] = {n: dict((board or {}).get(n) or {}) for n in (field or list((board or {}).keys()))}
        B = _FACTS.Board(c)
        out = {}
        for n in B.heroes:
            tags = []
            for t in B.identity(n):
                t0 = t.split(" (")[0]
                short = {"FUTURE CARRY": "未来主C", "MAIN TANK": "主T", "OFF-TANK": "副T", "lord-xp unit": "经验件"}.get(t0)
                if short is None:
                    if t0.startswith("2nd carry"):
                        short = "主C候选"
                    elif t0.startswith("transitional"):
                        short = "过渡"
                    elif t0.startswith("buffs NEAREST"):
                        short = "贴主C" + ("✓" if "(adjacent now)" in t else "✗" if "NOT adjacent" in t else "")
                    elif t0.startswith("stealable"):
                        short = "可夺取"
                    elif t0.startswith("low value late"):
                        short = "后期低价值"
                    else:
                        short = t0[:8]
                tags.append(short)
            out[n] = {"tags": tags, "fit": _FACTS.fit(n)}
        return out
    except Exception as e:                                  # noqa: BLE001
        return {"_error": str(e)[:80]}


_CARD_CACHE: dict = {}


def _card_detail(name: str) -> dict:
    """浮框用的卡牌详情（点击英雄 / 装备弹出详细信息小浮框）。
    只读原始数据：卡面、机制库、审定用法、主C 评分、伤害类型、装备分类、推荐阵容统计。"""
    name = _clean_name(name)
    if name in _CARD_CACHE:
        return _CARD_CACHE[name]
    cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))["cards"]
    c = cards.get(name)
    if not c:
        return {"name": name, "missing": True}
    out = {"name": name, "category": c.get("category"), "quality": c.get("quality"), "faction": c.get("faction", ""),
           "tags": c.get("tags") or [], "skill": (c.get("skill") or "").strip(), "awaken": (c.get("awaken") or "").strip(),
           "img": f"/img/{quote(name)}" if (name in LOCAL_IMG or name in REMOTE_IMG) else None,
           "keywords": [k.get("name") for k in c.get("keywords") or [] if k.get("name")]}
    try:
        kb = json.loads((ROOT / "data" / "card_kb.json").read_text(encoding="utf-8"))["entries"].get(name) or {}
        if kb:
            out["kb"] = {"trig": kb.get("trig"), "note": kb.get("note"), "scope": kb.get("scope"), "routes": kb.get("routes"),
                         "gives": [" ".join(str(x) for x in (g.get("res"), g.get("n"), ("→ " + g["to"]) if g.get("to") else "",
                                                               ("/" + g["per"]) if g.get("per") else "") if x not in (None, ""))
                                   for g in kb.get("gives") or []]}
        u = json.loads((ROOT / "data" / "card_usage_reviewed.json").read_text(encoding="utf-8")).get(name)
        if u:
            out["usage"] = {"target": u.get("target"), "reason": u.get("reason")}
    except Exception:                                       # noqa: BLE001
        pass
    if c.get("category") == "英雄牌":
        pr = c.get("properties") or {}
        out["stats"] = {"生命": pr.get("HP"), "物攻": pr.get("phyAttack"), "法攻": pr.get("magAttack"),
                        "攻击距离": pr.get("attackDistance"), "法力": pr.get("energy")}
        out["skills"] = [{"name": sk.get("name"), "desc": sk.get("desc"),
                          "enhance": [f"{e.get('level')} 级：{e.get('desc')}" for e in sk.get("enhance") or []]}
                         for sk in c.get("skills") or []]
        try:
            sys.path.insert(0, str(ROOT))
            from agent.decision import hero_traits as HT
            out["dmg"] = {"phys": "物理", "mag": "法术", "hybrid": "混合"}.get(HT.dmg_type(name), "?")
            hc = json.loads((ROOT / "data" / "hero_carry_reviewed.json").read_text(encoding="utf-8")).get(name) or {}
            if hc:
                out["carry"] = {"score": hc.get("carry_score"), "why": hc.get("why")}
            global _FACTS
            if _FACTS is None:
                sys.path.insert(0, str(ROOT / "tools" / "laya"))
                import facts as _F
                _FACTS = _F
            a = _FACTS.anchored(name)
            if a.get("n"):
                out["lineups"] = {"n": a["n"], "ali": a["ali"],
                                  "mates": [f"{k} {round(v * 100)}%" for k, v in list(a["mates"].items())[:6]],
                                  "equips": [f"{k} {round(v * 100)}%" for k, v in list(a["equips"].items())[:4]]}
        except Exception as e:                              # noqa: BLE001
            out["err"] = str(e)[:80]
    if c.get("category") == "装备牌":
        try:
            sys.path.insert(0, str(ROOT))
            from agent.decision import hero_traits as HT
            names = {"dps_phys": "物理输出", "dps_mag": "法术输出", "dps_any": "通用输出", "caster": "施法类",
                     "survive": "主C 保命", "tank": "坦克防御", "util": "功能"}
            if name in HT.EQUIP_CLASS:
                k, lim, why = HT.EQUIP_CLASS[name]
                out["equip"] = {"class": names.get(k, k), "limit": {"melee": "只适合近战", "ranged": "只限远程"}.get(lim, ""),
                                "why": why}
            rec = json.loads((ROOT / "data" / "equip_recipes.json").read_text(encoding="utf-8")).get("recipes", {}).get(name)
            if rec:
                out["forge"] = [{"name": r, "class": names.get((HT.EQUIP_CLASS.get(r) or ("",))[0], ""),
                                 "skill": (cards.get(r, {}).get("skill") or "")[:60]} for r in rec]
        except Exception as e:                              # noqa: BLE001
            out["err"] = str(e)[:80]
    _CARD_CACHE[name] = out
    return out


def _parse_points(p: Path, run: str) -> list[dict]:
    res = []
    for l in open(p, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        raw = _parse(r.get("raw")) or {}
        ctx = _parse(r.get("ctx")) or {}
        probs = _parse(r.get("laya")) or {}
        opts = _parse(r.get("options")) or list(probs)
        res.append({
            "id": r.get("id"), "run": r.get("run", run), "kind": r.get("kind"), "round": r.get("round") or ctx.get("round"),
            "ts": float(r.get("ts") or 0), "ctx": ctx, "field": _parse(r.get("field")) or [], "board": _parse(r.get("board")) or {},
            "ident": _identities(ctx, _parse(r.get("board")) or {}, _parse(r.get("field")) or []),
            "options": [dict(_option(r.get("kind"), k, raw if isinstance(raw, dict) else {}), p=float(probs.get(k, 0))) for k in opts],
            "laya_choice": r.get("laya_choice"), "rule_choice": r.get("rule_choice"), "used": r.get("used"),
            "why": r.get("why") or "", "conf": float(r.get("laya_conf") or 0), "ms": float(r.get("laya_ms") or 0),
            "ckpt": r.get("ckpt", ""), "state_text": r.get("state", ""),
        })
    return res


class H(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/runs":
            return self._json(_runs())
        if u.path == "/api/points":
            return self._json(_points(q.get("run", ["latest"])[0]))
        if u.path == "/api/card":
            return self._json(_card_detail(q.get("name", [""])[0]))
        if u.path.startswith("/img/"):
            n = unquote(u.path[5:])
            if n in LOCAL_IMG:
                b = Path(LOCAL_IMG[n]).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            if n in REMOTE_IMG:
                self.send_response(302)
                self.send_header("Location", REMOTE_IMG[n])
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()
            return
        if u.path in ("/", "/index.html"):
            b = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        self.send_response(404)
        self.end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"Laya 决策展示：http://localhost:{args.port}  （卡片 {len(CARDS)}，本地图 {len(LOCAL_IMG)}，远程图 {len(REMOTE_IMG)}）")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
