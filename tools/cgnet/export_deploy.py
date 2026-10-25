"""把选定的 cgnet 配置导出成线上部署包 data/cgnet/deploy/（agent/decision/cgnet_policy.py 读它）。

  .venv-dfine/Scripts/python.exe -X utf8 tools/cgnet/export_deploy.py --run v1b --seeds 1,2,3
（主 .venv 没有 onnx 包；.venv-dfine 有，不装新库。）

产物：seed<s>.onnx（每个种子一个，线上取概率平均）、text_keys.json（训练时图谱外文本的下标表）、manifest.json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "tools" / "cgnet"))
from model import CGNet  # noqa: E402
from train import collate  # noqa: E402

ROOT = PROJECT_DIR / "data" / "cgnet"
NAMES = ["tt", "ot", "node", "text", "num", "glob", "mask", "rel", "opt", "optm"]


def load_model(run: str, seed: int):
    d = ROOT / "runs" / run / f"seed{seed}"
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    meta = json.loads((ROOT / cfg["ds"] / "meta.json").read_text(encoding="utf-8"))
    tt = np.load(ROOT / cfg["ds"] / f"text_table_{cfg['enc']}.npy")
    nf = np.load(ROOT / cfg["ds"] / "node_feat.npy")
    m = CGNet(torch.from_numpy(tt), torch.from_numpy(nf), len(meta["tok_types"]), len(meta["opt_types"]),
              meta["num_dim"], meta["glob_dim"], meta["n_channels"], d=cfg["d"], heads=cfg["heads"],
              layers=cfg["layers"], ff=cfg["ff"], drop=0.0, use_text=cfg["ablate"] != "no_text",
              use_rel=cfg["ablate"] != "no_rel", use_node_id=cfg["ablate"] != "no_nodeid")
    m.load_state_dict(ck["state"])
    m.eval()
    return m, cfg, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--kinds", default="carry,swap")
    ap.add_argument("--out", default=str(ROOT / "deploy"))
    ap.add_argument("--why", default="")
    ap.add_argument("--min-engine-feat", action="store_true", help="训练数据打开了 build_data_v2.MIN_ENGINE_FEAT（线上按它开关）")
    ap.add_argument("--engine-feat", action="store_true", help="训练数据打开了 build_data_v2.ENGINE_FEAT（2026-10-14 推演器特征 num 44~47）")
    ap.add_argument("--w-feat", type=int, nargs="?", const=1, default=0, help="训练数据打开了 build_data_v2.W_FEAT（2026-10-16 f9lw 单位事实 num 44~48）")
    ap.add_argument("--flow-feat", action="store_true", help="训练数据打开了 build_data_v2.FLOW_FEAT（2026-10-14 流派特征 glob 103~108）")
    ap.add_argument("--eco-core-feat", action="store_true", help="训练数据打开了 build_data_v2.ECO_CORE_FEAT（2026-10-13；线上要 cgnet_policy 按 manifest 切换）")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    cfg = meta = None
    for sd in [int(x) for x in a.seeds.split(",")]:
        model, cfg, meta = load_model(a.run, sd)
        s0 = next(s for s in pickle.loads((ROOT / cfg["ds"] / "samples.pkl").read_bytes()) if s["kind"] == "swap")
        x0, _ = collate([s0], meta["n_channels"], torch.device("cpu"))
        dyn = {"tt": {1: "N"}, "ot": {1: "N"}, "node": {1: "N"}, "text": {1: "N"}, "num": {1: "N"}, "mask": {1: "N"},
               "rel": {1: "N", 2: "N2"}, "opt": {1: "K"}, "optm": {1: "K"}, "logits": {1: "K"}}
        f = out / f"seed{sd}.onnx"
        torch.onnx.export(model, tuple(x0[k] for k in NAMES), str(f), input_names=NAMES, output_names=["logits"],
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
        files.append(f.name)
    shutil.copy(ROOT / cfg["ds"] / "text_keys.json", out / "text_keys.json")
    graph = meta.get("graph") or "data/card_graph.json"
    man = {"run": a.run, "seeds": [int(x) for x in a.seeds.split(",")], "ds": cfg["ds"],
           "version": meta.get("version", "v1"), "graph": graph,
           "graph_md5": hashlib.md5((PROJECT_DIR / graph).read_bytes()).hexdigest(),
           "n_channels": meta["n_channels"], "num_dim": meta["num_dim"], "glob_dim": meta["glob_dim"],
           "kinds": a.kinds.split(","), "onnx": files, "why": a.why,
           "v2ablate": cfg.get("v2ablate", "none"), "train_kinds": cfg.get("kinds", ""),
           "channel_ranges": meta.get("channel_ranges"), "fact_glob": meta.get("fact_glob"),
           "fact_num": meta.get("fact_num"), "tech_num": meta.get("tech_num"),
           "tt_technique": meta["tok_types"].index("technique"), "min_engine_feat": bool(a.min_engine_feat),
           "eco_core_feat": bool(a.eco_core_feat), "engine_feat": bool(a.engine_feat), "flow_feat": bool(a.flow_feat), "w_feat": int(a.w_feat)}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
