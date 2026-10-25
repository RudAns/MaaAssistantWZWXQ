"""把微调好的 D-FINE-M（己方单位/格子占用）导出为 ONNX（固定 640x640，含后处理），给线上 onnxruntime-TensorRT 用。

  .venv-dfine/Scripts/python.exe tools/vision/export_board.py --ckpt data/vision/runs/board_m_v1/best_stg1.pth
  -> assets/model/vision/board_m_v1_640.onnx（输入 images(1,3,640,640) RGB/255、orig_target_sizes(1,2)=[[W,H]]；
     输出 labels / boxes(原图坐标 x0y0x1y1) / scores）

权重挑选与 tools/vision/eval_cells.py 一致（有 EMA 用 EMA，否则 model），保证线上用的就是评测过的那一版。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DF = ROOT / "third_party" / "D-FINE"
sys.path.insert(0, str(DF))
from src.core import YAMLConfig  # noqa: E402


def grid_sample_gather(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    """与 F.grid_sample(mode=bilinear, padding_mode=zeros, align_corners=False) 数学等价，只用 gather + 逐元素运算。

    为什么（2026-09-24）：带 grid_sample 的 D-FINE 在 TensorRT 下**编译成功但结果错**——测试集格子召回
    CPU/CUDA 100% → TensorRT FP32 72.8% / FP16 84.5%。公开资料同样记录了这个问题（dfine-cpp：COCO AP
    0.5509 → 0.4455，改成 gather-bilinear 后恢复）。这里是按同一思路自己写的实现，只在导出时替换。
    """
    assert mode == "bilinear" and padding_mode == "zeros" and not align_corners
    n, c, h, w = v.shape
    hg, wg = grid.shape[1], grid.shape[2]
    x = (grid[..., 0] + 1) * (w / 2) - 0.5          # align_corners=False 的像素坐标
    y = (grid[..., 1] + 1) * (h / 2) - 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    x1, y1 = x0 + 1, y0 + 1
    vf = v.reshape(n, c, h * w)

    def corner(xi, yi, wt):
        valid = ((xi >= 0) & (xi <= w - 1) & (yi >= 0) & (yi <= h - 1)).to(v.dtype)
        idx = (yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)).to(torch.int64).reshape(n, 1, hg * wg).expand(n, c, hg * wg)
        return torch.gather(vf, 2, idx).reshape(n, c, hg, wg) * (wt * valid).unsqueeze(1)

    return (corner(x0, y0, (x1 - x) * (y1 - y)) + corner(x0, y1, (x1 - x) * (y - y0))
            + corner(x1, y0, (x - x0) * (y1 - y)) + corner(x1, y1, (x - x0) * (y - y0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default=str(DF / "configs/dfine/custom/wxq_board_m.yml"))
    ap.add_argument("--out", default=str(ROOT / "assets/model/vision/board_m_v1_640.onnx"))
    ap.add_argument("--gather", action="store_true", help="导出时把 grid_sample 换成等价的 gather 双线性（TensorRT 才算得对）")
    ap.add_argument("--raw", action="store_true",
                    help="不带后处理，输出 pred_logits(1,300,C) / pred_boxes(1,300,4 cxcywh 归一化)；"
                         "带后处理的版本在 TensorRT 下算错，后处理改在 numpy 里做")
    args = ap.parse_args()
    cfg = YAMLConfig(args.cfg, resume=args.ckpt)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    st = torch.load(args.ckpt, map_location="cpu")
    st = st["ema"]["module"] if "ema" in st and st.get("ema") else st["model"]
    cfg.model.load_state_dict(st)
    model, post = cfg.model.deploy().eval(), cfg.postprocessor.deploy().eval()

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.post = model, post

        def forward(self, images, orig_target_sizes):
            return self.post(self.model(images), orig_target_sizes)

    x = torch.rand(1, 3, 640, 640)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.gather:
        # 先在 PyTorch 里核对替换前后输出一致，再导出
        with torch.no_grad():
            ref = model(x)
            orig = torch.nn.functional.grid_sample
            torch.nn.functional.grid_sample = grid_sample_gather
            new = model(x)
        dl = (ref["pred_logits"] - new["pred_logits"]).abs().max().item()
        db = (ref["pred_boxes"] - new["pred_boxes"]).abs().max().item()
        print(f"grid_sample → gather 双线性：logits 最大差 {dl:.2e}，boxes 最大差 {db:.2e}")
        assert dl < 1e-3 and db < 1e-4, "替换后输出不一致，不导出"
        args.out = args.out.replace(".onnx", "_gather.onnx")
        _ = orig
    if args.raw:
        class Raw(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, images):
                o = self.model(images)
                return o["pred_logits"], o["pred_boxes"]
        out = args.out.replace(".onnx", "_raw.onnx")
        torch.onnx.export(Raw().eval(), (x,), out, input_names=["images"], output_names=["pred_logits", "pred_boxes"],
                          opset_version=17, dynamo=False)
        print("导出", out)
        return 0
    m = M().eval()
    sz = torch.tensor([[1280, 720]])
    torch.onnx.export(m, (x, sz), args.out, input_names=["images", "orig_target_sizes"],
                      output_names=["labels", "boxes", "scores"], opset_version=17, dynamo=False)
    print("导出", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
