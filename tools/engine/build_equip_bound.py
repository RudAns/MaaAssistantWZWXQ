"""从官网卡库读出"不能换人"的装备（2026-10-14：绑定装备在装备数据库里都有写明，直接读）。
判据（只读卡面原文）：
  locked   装备卡面写"无法拆卸"
  innate   英雄卡面写"X在场时，获得(普通)装备Y"——随英雄在场存在，跟人走
输出 data/engine/equip_bound_v1.json。卡库更新后重跑。
"""
import glob, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
strip = lambda s: re.sub(r"<[^>]+>", "", s or "")
out = {}
def walk(o, owner=None):
    if isinstance(o, dict):
        name, desc = o.get("name"), strip(o.get("desc"))
        if name and o.get("type") == 5 and "无法拆卸" in desc:
            out.setdefault(name, {"kind": "locked", "quality": o.get("quality"), "raw": desc[:80]})
        if name and o.get("type") in (1, 9):
            for m in re.finditer(r"在场时，获得(?:普通|特殊)?装备([^，。及]+)(?:及([^，。]+))?", desc):
                for it in filter(None, m.groups()):
                    out.setdefault(it.strip(), {"kind": "innate", "owner": name, "raw": desc[:80]})
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
for f in glob.glob(str(ROOT / "data/web/*.raw.json")):
    walk(json.load(open(f, encoding="utf-8")))
p = ROOT / "data/engine/equip_bound_v1.json"
p.write_text(json.dumps({"_doc": __doc__.strip().splitlines()[0], "bound": out}, ensure_ascii=False, indent=2), encoding="utf-8")
for k, v in out.items(): print(v["kind"], k, v.get("owner", ""), "|", v["raw"][:50])
