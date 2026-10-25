"""equip_cnt 标签修正（2026-10-15，第 637 局廉颇）：原标签 = 区间内规则读数的众数，规则读法会把英雄模型的护臂 / 盾 / 身体部件认成装备图标
（抽查：廉颇 / 太乙真人 / 苏烈 / 蒙恬 账本 0 件、标签 1，裁图上徽章下面都没有装备图标），错误进了标签，模型学成同样的误认。
修法：认得出名字、穿戴账本 = 0（从上场 / 上次卖出起没被拖过任何装备）而标签 ≥1 → 改成 0、记 strong；
卡面"在场时获得…装备"的英雄（干将莫邪 = 双剑·雄）除外。原文件留在 labels_v1_orig.jsonl。"""
import json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "laya"))
import facts  # noqa: E402
DS = ROOT / "data" / "vision_v2" / "equip_cnt"
INTRINSIC = re.compile(r"在场时[^。]{0,10}获得[^。]{0,10}装备")
rows = [json.loads(l) for l in (DS / "labels_v1_orig.jsonl").open(encoding="utf-8")]
st = Counter()
for r in rows:
    n = r.get("name")
    if n and r.get("ledger") == 0 and r["label"] > 0:
        if INTRINSIC.search(facts.skill(n) or ""):
            st["keep_intrinsic"] += 1
            continue
        r["label_orig"] = r["label"]
        r["label"] = 0
        r["strong"] = True
        r["ledger_fix"] = True
        st[f"fixed:{r['split']}"] += 1
with (DS / "labels.jsonl").open("w", encoding="utf-8") as fo:
    for r in rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(dict(st))
