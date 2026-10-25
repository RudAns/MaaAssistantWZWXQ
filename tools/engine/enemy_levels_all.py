"""对 data/logs 里所有战斗截图跑对手等级识别（tools/analyze_enemy_levels.analyze），结果写 data/engine/enemy_levels_all.json，
不覆盖 data/enemy_levels.json（冻结版推演器 agent/engine_v1 读它，改了线上特征就和训练对不上）。只用 CPU，多进程。
  .venv/Scripts/python.exe -X utf8 tools/engine/enemy_levels_all.py [最近 N 局，默认 400] [进程数，默认 8]
"""
import glob, json, os, sys
from multiprocessing import Pool
os.environ["WZWXQ_OCR_EP"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
_OCR = None


def work(d):
    global _OCR
    import analyze_enemy_levels as A
    if _OCR is None:
        from agent.perception.onnx_ocr import OnnxOcr
        _OCR = OnnxOcr.get()
    try:
        return os.path.basename(d), A.analyze(d, _OCR)
    except Exception as e:  # noqa: BLE001
        return os.path.basename(d), {"_err": repr(e)[:100]}


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    runs = sorted(glob.glob(os.path.join(ROOT, "data", "logs", "live_*")), key=os.path.getmtime)[-N:]
    dst = os.path.join(ROOT, "data", "engine", "enemy_levels_all.json")
    out = json.load(open(dst, encoding="utf-8")) if os.path.exists(dst) else {}
    todo = [d for d in runs if os.path.basename(d) not in out]
    with Pool(P) as pool:
        for i, (k, per) in enumerate(pool.imap_unordered(work, todo)):
            if per and "_err" not in per:
                out[k] = per
            if i % 10 == 0:
                json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
                print(i, len(todo), len(out), flush=True)
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    print("完成", len(out))
