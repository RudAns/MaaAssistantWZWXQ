"""线上题回放冒烟（2026-09-30 第 42 局教训）：拿对局里真实写下的 carry/swap 题（laya_points.jsonl，去掉线上没有的字段）
逐题喂给部署的 cgnet，任何一题返回 None（= 线上会退回 Laya）就失败。
离线一致性测试用的是数据集题（带 id/split），抓不到"线上题缺字段 / 图谱版本变了"这类错。
用法：.venv/Scripts/python.exe -X utf8 tools/cgnet/test_live_points.py [live_目录 ...]（默认最近 5 局）"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.decision.cgnet_policy import CGNetPolicy  # noqa: E402

LIVE_ONLY_DROP = ("id", "split", "laya", "laya_choice", "laya_conf", "laya_ms", "rule_choice", "used", "why",
                  "ckpt", "feature_version", "model")


def main(argv):
    runs = [ROOT / "data" / "logs" / a for a in argv] or sorted(
        (p.parent for p in (ROOT / "data" / "logs").glob("live_*/laya_points.jsonl")), key=lambda p: p.stat().st_mtime)[-5:]
    pol = CGNetPolicy.load(run_dir=runs[-1])
    if pol is None:
        print("FAIL：cgnet 加载失败")
        return 1
    ok = bad = 0
    for run in runs:
        f = run / "laya_points.jsonl"
        if not f.exists():
            continue
        pol.set_run_dir(run)
        for line in f.open(encoding="utf-8"):
            d = json.loads(line)
            if d.get("kind") not in pol.kinds:
                continue
            for k in LIVE_ONLY_DROP:
                d.pop(k, None)
            if pol.ask(d):
                ok += 1
            else:
                bad += 1
    print(f"线上题回放：成功 {ok}，失败 {bad}")
    if bad or not ok:
        print("FAIL")
        return 1
    print("通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
