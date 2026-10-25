"""D-FINE-M 冒烟测试 + 3060 延迟（PyTorch fp16，未上 TensorRT）。
  .venv-dfine/Scripts/python.exe tools/vision_bench_dfine.py --size 640 640 --size 1024 576
"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DF = ROOT / "third_party" / "D-FINE"
sys.path.insert(0, str(DF))
import torch
from src.core import YAMLConfig

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default=str(DF / "configs/dfine/objects365/dfine_hgnetv2_m_obj2coco.yml"))
ap.add_argument("--ckpt", default=str(ROOT / "third_party/pretrained/dfine_m_obj2coco.pth"))
ap.add_argument("--size", nargs=2, type=int, action="append")
ap.add_argument("--export", action="store_true")
args = ap.parse_args()
st = torch.load(args.ckpt, map_location="cpu")
st = st["ema"]["module"] if "ema" in st else st["model"]


def build(h, w):
    # 编码器/解码器的位置编码按 eval_spatial_size 预先生成，换输入尺寸要按那个尺寸重建
    cfg = YAMLConfig(args.cfg, eval_spatial_size=[h, w])
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    m = cfg.model
    m.load_state_dict({k: v for k, v in st.items() if "anchors" not in k and "valid_mask" not in k}, strict=False)
    return m.deploy().cuda().eval(), cfg.postprocessor.deploy().cuda().eval()


for w, h in ([] if args.export else (args.size or [[640, 640]])):
    model, post = build(h, w)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand(1, 3, h, w, device="cuda")
    sz = torch.tensor([[w, h]], device="cuda")
    for dtype in (torch.float32, torch.float16):
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.float16):
            for _ in range(10):
                post(model(x), sz)
            torch.cuda.synchronize(); t = time.perf_counter(); n = 50
            for _ in range(n):
                post(model(x), sz)
            torch.cuda.synchronize()
        print(f"  D-FINE-M {nparam:.1f}M 输入 {w}x{h} {str(dtype)[6:]}: {1000*(time.perf_counter()-t)/n:.1f} ms/帧, "
              f"峰值显存 {torch.cuda.max_memory_allocated()/2**20:.0f} MB")
    del model, post


def export_onnx(h, w, out):
    """导出 ONNX（固定尺寸，含后处理），给 onnxruntime-TensorRT 用。"""
    model, post = build(h, w)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.post = model.cpu(), post.cpu()

        def forward(self, images, orig_target_sizes):
            return self.post(self.model(images), orig_target_sizes)

    m = M().eval()
    x = torch.rand(1, 3, h, w)
    sz = torch.tensor([[w, h]])
    torch.onnx.export(m, (x, sz), out, input_names=["images", "orig_target_sizes"],
                      output_names=["labels", "boxes", "scores"], opset_version=17, dynamo=False)
    print("导出", out)


if args.export:
    out_dir = ROOT / "assets" / "model" / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    for w, h in (args.size or [[640, 640]]):
        export_onnx(h, w, str(out_dir / f"dfine_m_coco_{w}x{h}.onnx"))
