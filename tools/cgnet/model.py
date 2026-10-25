"""cgnet v1：对象集合 Transformer + 关系类型注意力偏置（Graphormer 式）+ 选项指针头。

输入（一个 batch，N = 对象数上限，C = 关系通道数，K = 选项数上限）：
  tt   (B,N) long   对象类型（board/hand/shop/.../option/technique）
  ot   (B,N) long   选项动作类型（buy/refresh/sell/...，非选项 = 0）
  node (B,N) long   图谱节点下标 +2（0 = 空位，1 = 图谱外卡牌）
  text (B,N) long   文本表下标（0 = 无文本）
  num  (B,N,Dn)     数值特征（等级/觉醒/装备/是否主C/…）
  glob (B,Dg)       全局特征（回合/能量/等级/人口/…/题型 one-hot）
  mask (B,N) bool   有效对象
  rel  (B,N,N,C)    关系多热（图谱边类型 × 方向 + 同卡/选项引用/路线成员/主体-选项）
  opt  (B,K) long   选项在对象序列中的位置；optm (B,K) bool
输出：logits (B,K)（无效选项 = -1e4）

注意力：score_h(i,j) = q_i·k_j/√d + Σ_c rel[i,j,c]·w[layer,c,h]；全局 token 放在第 0 位（无关系偏置）。
不含位置编码 → 对象与选项的顺序不影响输出（集合等变）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelAttnLayer(nn.Module):
    def __init__(self, d: int, h: int, ff: int, n_ch: int, drop: float):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.rel_w = nn.Parameter(torch.zeros(n_ch, h))       # 每层每头每种关系一个标量偏置
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, key_mask, rel, use_rel: bool = True):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)                 # (B,H,N,N)
        if use_rel:
            s = s + torch.einsum("bijc,ch->bhij", rel, self.rel_w)
        s = s.masked_fill(~key_mask[:, None, None, :], -1e4)
        a = self.drop(torch.softmax(s, dim=-1))
        y = (a @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.o(y))
        return x + self.drop(self.ff(self.n2(x)))


class CGNet(nn.Module):
    def __init__(self, text_table: torch.Tensor, node_feat: torch.Tensor, n_tt: int, n_ot: int, num_dim: int,
                 glob_dim: int, n_ch: int, d: int = 128, heads: int = 4, layers: int = 3, ff: int = 256,
                 node_dim: int = 32, text_dim: int = 64, drop: float = 0.1,
                 use_text: bool = True, use_rel: bool = True, use_node_id: bool = True):
        super().__init__()
        self.register_buffer("text_table", text_table.float(), persistent=True)
        self.register_buffer("node_feat", node_feat.float(), persistent=True)
        self.use_text, self.use_rel, self.use_node_id = use_text, use_rel, use_node_id
        self.tt = nn.Embedding(n_tt, d)
        self.ot = nn.Embedding(n_ot, d)
        self.node = nn.Embedding(node_feat.shape[0], node_dim, padding_idx=0)
        self.text_proj = nn.Sequential(nn.Linear(text_table.shape[1], text_dim), nn.GELU())
        in_dim = node_dim + text_dim + node_feat.shape[1] + num_dim
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(glob_dim, d), nn.GELU(), nn.Linear(d, d))
        self.g_tok = nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList([RelAttnLayer(d, heads, ff, n_ch, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.emb_drop = nn.Dropout(drop)

    def forward(self, tt, ot, node, text, num, glob, mask, rel, opt, optm):
        B, N = tt.shape
        te = self.text_table[text]
        if not self.use_text:
            te = torch.zeros_like(te)
        ne = self.node(node)
        if not self.use_node_id:
            ne = torch.zeros_like(ne)
        feat = torch.cat([ne, self.text_proj(te), self.node_feat[node], num], dim=-1)
        x = self.inp(feat) + self.tt(tt) + self.ot(ot)
        g = (self.glob(glob) + self.g_tok)[:, None, :]
        x = self.emb_drop(torch.cat([g, x], dim=1))                            # (B,1+N,d)
        km = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        relp = F.pad(rel, (0, 0, 1, 0, 1, 0))                                  # 全局 token 无关系
        for L in self.layers:
            x = L(x, km, relp, self.use_rel)
        x = self.norm(x)
        h_opt = torch.gather(x, 1, (opt + 1).clamp(min=0)[..., None].expand(-1, -1, x.shape[-1]))
        h_g = x[:, :1].expand(-1, opt.shape[1], -1)
        logits = self.head(torch.cat([h_opt, h_g], dim=-1)).squeeze(-1)
        return logits.masked_fill(~optm, -1e4)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
