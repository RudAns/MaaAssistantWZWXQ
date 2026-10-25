"""把 train_hand_art_v2.py 的 best.pt 导出成 ONNX（主 .venv 没装 onnx，用 .venv-dfine 跑，CPU）。

  CUDA_VISIBLE_DEVICES= .venv-dfine/Scripts/python.exe -X utf8 tools/vision/export_hand_art.py data/vision/runs/hand_art_v2 [--thresh 0.xx]
输入 x [N,3,112,56] RGB/255，输出 logits；同目录写 <名>.onnx 与 <名>.json（classes, thresh）。
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hand_art import Net  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--thresh", type=float, default=None)
a = ap.parse_args()
run = Path(a.run)
m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
net = Net(len(m["classes"]))
net.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
net.eval()
torch.onnx.export(net, torch.zeros(1, 3, 112, 56), str(run / f"{run.name}.onnx"), input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
thr = a.thresh if a.thresh is not None else m["thresh"]
(run / f"{run.name}.json").write_text(json.dumps({"classes": m["classes"], "thresh": thr}, ensure_ascii=False), encoding="utf-8")
print("ok", len(m["classes"]), thr)
