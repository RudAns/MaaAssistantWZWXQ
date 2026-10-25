"""cgnet 文本编码：把卡面文字编码成向量（离线预计算，线上只查表）。

两种编码器（对照）：
  bge   BAAI/bge-small-zh-v1.5（512 维，CLS 池化 + L2 归一化，官方推荐用法）
  laya  线上 Laya 学生（R15）的 ModernBERT 编码器（768 维，均值池化 + L2 归一化）

缓存：data/cgnet/text_cache_<enc>.npz（key = 文本 sha1）—— 同一段文本只编码一次，可反复复跑。
用法：
  python tools/cgnet/textenc.py            # 编码图谱全部节点，写 data/cgnet/text_emb_<enc>.npy + node_index.json
  python tools/cgnet/textenc.py --enc laya
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT = PROJECT_DIR / "data" / "cgnet"
BGE_ID = "BAAI/bge-small-zh-v1.5"
LAYA_CKPT = PROJECT_DIR / "data" / "laya" / "runs" / "R15_s1_top8" / "checkpoint"


def _sha(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def node_text(n: dict) -> str:
    """节点文本 = 卡面原文 + understanding 的 effect / depends_on（手法节点：名称 + 收益 + 价值 + 时机）。"""
    parts = [str(n.get("id", ""))]
    if n.get("type") == "technique":
        for k in ("name", "benefit", "value", "timing"):
            if n.get(k):
                parts.append(str(n[k]))
    elif n.get("type") == "scope":
        parts.append(str(n.get("def", "")))
    else:
        if n.get("text"):
            parts.append(str(n["text"]))
        u = n.get("understanding") or {}
        if u.get("effect"):
            parts.append("作用：" + str(u["effect"]))
        if u.get("depends_on"):
            parts.append("依赖：" + "；".join(map(str, u["depends_on"])))
    return "\n".join(p for p in parts if p)


class Encoder:
    def __init__(self, enc: str = "bge", device: str | None = None):
        import torch                    # 延迟导入：线上 cgnet_policy 只用 load_graph / node_text，不需要 torch
        self.enc = enc
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_path = OUT / f"text_cache_{enc}.npz"
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            z = np.load(self.cache_path)
            self.cache = {k: z[k] for k in z.files}
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        if self.enc == "bge":
            from huggingface_hub import snapshot_download
            p = snapshot_download(BGE_ID, allow_patterns=["*.json", "*.txt", "model.safetensors"])
            self.tok = AutoTokenizer.from_pretrained(p)
            self._model = AutoModel.from_pretrained(p).to(self.device).eval()
            self.max_len = 512
        elif self.enc == "laya":
            sys.path.insert(0, str(PROJECT_DIR / "tools" / "laya"))
            from train import load_model  # noqa: E402
            model, tok, _ = load_model(LAYA_CKPT, self.device)
            self.tok = tok
            self._model = model.encoder.eval()
            self.max_len = 512
        else:
            raise ValueError(self.enc)

    def _run(self, texts: list[str]) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self._run_nograd(texts)

    def _run_nograd(self, texts: list[str]) -> np.ndarray:
        import torch
        self._load()
        out = []
        for b in range(0, len(texts), 32):
            batch = self.tok(texts[b:b + 32], padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt").to(self.device)
            h = self._model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state.float()
            if self.enc == "bge":
                v = h[:, 0]
            else:
                m = batch["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            v = torch.nn.functional.normalize(v, dim=-1)
            out.append(v.cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        todo = sorted({t for t in texts if _sha(t) not in self.cache})
        if todo:
            vec = self._run(todo)
            for t, v in zip(todo, vec):
                self.cache[_sha(t)] = v
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, **self.cache)
        return np.stack([self.cache[_sha(t)] for t in texts]) if texts else np.zeros((0, 1), np.float32)


def load_graph() -> dict:
    # WZWXQ_CGNET_GRAPH：指定图谱文件（v3 起用部署冻结版 data/cgnet/deploy/card_graph_frozen.json，与线上一致）
    import os
    path = os.environ.get("WZWXQ_CGNET_GRAPH") or str(PROJECT_DIR / "data" / "card_graph.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="bge", choices=["bge", "laya"])
    a = ap.parse_args()
    g = load_graph()
    nodes = g["nodes"]
    texts = [node_text(n) for n in nodes]
    enc = Encoder(a.enc)
    emb = enc.encode(texts)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"text_emb_{a.enc}.npy", emb)
    idx = {"nodes": [n["id"] for n in nodes], "types": [n.get("type") for n in nodes], "encoder": a.enc,
           "dim": int(emb.shape[1]), "graph_version": g.get("version"),
           "text_chars_median": int(np.median([len(t) for t in texts]))}
    (OUT / f"node_index_{a.enc}.json").write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    # 简单自检：同类最近邻
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1)
    for name in ("古币", "苏烈", "红叶染锋", "tech:maple_level_carryover"):
        if name in idx["nodes"]:
            i = idx["nodes"].index(name)
            nn = np.argsort(-sims[i])[:5]
            print(a.enc, name, "->", [(idx["nodes"][j], round(float(sims[i, j]), 3)) for j in nn])
    print(f"{a.enc}: {emb.shape} -> {OUT / f'text_emb_{a.enc}.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
