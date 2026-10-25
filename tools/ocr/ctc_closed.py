"""闭集 CTC 解码：不让认字模型自由拼字，而是给卡库里每个卡名（及其前缀）打分，取最高的。

2026-09-25：原始轻量 OCR 准确度偏低，需要更好的模型或算法减少误判。
自由解码会拼出「业热」这种不存在的名字（亚瑟），之后的模糊匹配也救不回来；
闭集解码的输出**只可能是真卡名**：只要模型给"亚""瑟"留了一点概率，亚瑟就会赢过所有别的名字。

用法：
  dec = ClosedDecoder(chars, names)            # chars：认字模型的字表（含 <blank>）；names：候选卡名
  dec.rank(probs, prefix=True) -> [(name_or_prefix, logp_per_char, is_full), ...]（按分数降序）
probs：单张图的 (T, C) softmax 概率。
"""
from __future__ import annotations

import numpy as np

NEG = -1e30


def ctc_logp(logp: np.ndarray, seq: list[int], prefix: bool = False) -> float:
    """CTC 前向算法（向量化到状态维）：log P(seq | x)。logp：(T, C) 的 log 概率；seq：字下标（不含空白 0）。"""
    L = len(seq)
    if L == 0:
        return float(logp[:, 0].sum())
    ext = np.zeros(2 * L + 1, dtype=np.int64)
    ext[1::2] = seq
    S = len(ext)
    skip = np.zeros(S, dtype=bool)                  # 允许从 s-2 跳过来：非空白、且与 s-2 不同
    skip[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    em = logp[:, ext]                               # (T, S)
    a = np.full(S, NEG)
    a[0], a[1] = em[0, 0], em[0, 1]
    for t in range(1, logp.shape[0]):
        s1 = np.concatenate(([NEG], a[:-1]))
        s2 = np.where(skip, np.concatenate(([NEG, NEG], a[:-2])), NEG)
        a = np.logaddexp(np.logaddexp(a, s1), s2) + em[t]
    return float(np.logaddexp(a[-1], a[-2]))


class ClosedDecoder:
    def __init__(self, chars: list[str], names: list[str]):
        idx = {c: i for i, c in enumerate(chars)}
        self.items = []                         # (name, [char idx])
        for n in dict.fromkeys(names):
            ids = [idx.get(ch) for ch in n]
            if n and all(i is not None for i in ids):
                self.items.append((n, ids))

    def rank(self, probs: np.ndarray, prefix: bool = True, top: int = 5):
        """候选 = 全部卡名（+ prefix 时所有前缀，**当作"图上恰好就这几个字"**：叠放时只露出前 k 个字）。
        每个候选算完整 CTC log P（不用"以此开头"的概率——那样单字前缀总赢）。返回 [(串, 总 logp, 是否完整卡名)]。"""
        lp = np.log(np.clip(probs, 1e-12, 1.0))
        cand: dict = {}
        for n, ids in self.items:
            cand[n] = (ids, True)
            if prefix:
                for k in range(1, len(ids)):
                    cand.setdefault(n[:k], (ids[:k], False))
        out = [(c, ctc_logp(lp, ids), full) for c, (ids, full) in cand.items()]
        out.sort(key=lambda r: -r[1])
        return out[:top]
