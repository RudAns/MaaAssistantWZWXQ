"""把 data/card_graph.json 生成成可直接双击打开的图谱页面 docs/card_graph.html（2026-09-30，便于随时查看）。

数据整份内联进页面（file:// 下不能 fetch），d3 从 cdnjs 加载（需要联网；断网时"手法 / 卡牌理解 / 边类型"三页照常能看）。
图谱数据更新后重跑：.venv/Scripts/python.exe -X utf8 tools/build_card_graph_html.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "card_graph.json"
TPL = Path(__file__).with_name("card_graph_template.html")
OUT = ROOT / "docs" / "card_graph.html"


def main() -> None:
    # 可选：--src PATH 改读暂存图谱（如 data/card_graph.next.json，2026-10-26）
    src = Path(sys.argv[sys.argv.index("--src") + 1]) if "--src" in sys.argv else SRC
    g = json.loads(src.read_text(encoding="utf-8"))
    keep = {k: g[k] for k in ("version", "edge_types", "func_tag_vocab", "nodes", "edges", "meta")}
    data = json.dumps(keep, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TPL.read_text(encoding="utf-8").replace("/*__DATA__*/null", data, 1)
    OUT.write_text(html, encoding="utf-8")
    print(f"写出 {OUT}（{OUT.stat().st_size/1024:.0f} KB，{len(g['nodes'])} 节点 / {len(g['edges'])} 边）")


if __name__ == "__main__":
    main()
