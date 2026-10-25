"""r9zk 的测试题按 f9lr 渲染（R30b 线上看到的文本），供 compare_students 对比 R30b。只渲染非 train 划分。"""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import build_r9za as BA  # noqa: E402
import rerender as RR  # noqa: E402
RR.ALLOW_NEW_KEYS = True
ds = HERE.parents[1] / "data" / "laya" / "datasets"
out = ds / "r9zk_f9lr"
out.mkdir(exist_ok=True)
t0 = time.time(); n = d = 0
with (out / "points.jsonl").open("w", encoding="utf-8") as fo:
    for l in (ds / "r9zk" / "points.jsonl").open(encoding="utf-8"):
        p = json.loads(l)
        if p["split"] == "train":
            continue
        r = BA.render(p, "f9lr", list(p["options"]))
        if r is None:
            d += 1
            continue
        fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
(out / "info.json").write_text(json.dumps({"feature_version": "f9lr", "points": n, "dropped": d}), encoding="utf-8")
print("rendered", n, "dropped", d, round(time.time() - t0), "s")
