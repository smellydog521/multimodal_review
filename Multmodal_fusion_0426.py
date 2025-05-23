#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合脚本，支持灵活开启/关闭 Pre-gating、Attention、CAG，其中
DynamicPCAGModule 和 PCAGProbabilisticModule 均可复用相同核心逻辑。
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# —— 核心算法模块 —— #
class PreGate(nn.Module):
    """Pre-gating：为 Q·K 点乘前生成位置级 gate_pq, gate_pk"""

    def __init__(self, dim, use_tanh=False):
        super().__init__()
        self.use_tanh = use_tanh
        self.pg_q = nn.Linear(dim, 1)
        self.pg_k = nn.Linear(dim, 1)

    def forward(self, Q, K):
        # Q: [B, Lq, d], K: [B, Lk, d]
        p_q = self.pg_q(Q)  # [B, Lq, 1]
        p_k = self.pg_k(K)  # [B, Lk, 1]
        if self.use_tanh:
            p_q = (torch.tanh(p_q) + 1) * 0.5
            p_k = (torch.tanh(p_k) + 1) * 0.5
        else:
            p_q = torch.sigmoid(p_q)
            p_k = torch.sigmoid(p_k)
        return p_q, p_k


class DotProductAttention(nn.Module):
    """Q-K-V 点乘注意力，支持可选传入 pre-gate"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, Q, K, V, p_q=None, p_k=None):
        # Q: [B, Lq, d], K/V: [B, Lk, d]
        A = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5)  # [B, Lq, Lk]
        if p_q is not None and p_k is not None:
            A = A * p_q * p_k.permute(0, 2, 1)
        W = F.softmax(A, dim=-1)  # [B, Lq, Lk]
        return torch.matmul(W, V)  # [B, Lq, d]


class ContextualAttentionGate(nn.Module):
    """CAG：两路线性 + LayerNorm，再逐元素相乘"""

    def __init__(self, dim):
        super().__init__()
        self.W_E = nn.Linear(dim, dim)
        self.W_h = nn.Linear(dim, dim)
        self.W_q = nn.Linear(dim, dim)
        self.ln_E = nn.LayerNorm(dim)
        self.ln_G = nn.LayerNorm(dim)

    def forward(self, attended, Q):
        # attended/Q: [B, L, d]
        E = F.relu(self.W_E(attended))
        G = F.relu(self.W_h(attended) + self.W_q(Q))
        return self.ln_E(E) * self.ln_G(G)


# —— PCAGModule 基类 —— #
class PCAGModule(nn.Module):
    def __init__(self,
                 modal_dims,
                 hidden_dim,
                 use_pregate=True,
                 use_cag=True,
                 attn_dim=None,
                 use_tanh=False):
        super().__init__()
        self.M = len(modal_dims)
        self.use_pregate = use_pregate
        self.use_cag = use_cag
        self.attn_dim = attn_dim or modal_dims[0]

        # 1) Q/K/V 投影
        self.to_q = nn.ModuleList([nn.Linear(d, self.attn_dim) for d in modal_dims])
        self.to_k = nn.ModuleList([nn.Linear(d, self.attn_dim) for d in modal_dims])
        self.to_v = nn.ModuleList([nn.Linear(d, self.attn_dim) for d in modal_dims])

        # 2) Pre-gate
        if use_pregate:
            self.pregates = nn.ModuleList([
                PreGate(self.attn_dim, use_tanh) for _ in modal_dims
            ])

        # 3) 注意力层
        self.attn = DotProductAttention(self.attn_dim)

        # 4) CAG
        if use_cag:
            self.cags = nn.ModuleList([
                ContextualAttentionGate(self.attn_dim) for _ in modal_dims
            ])

        # 5) 输出融合
        self.out_fc = nn.Linear(self.attn_dim * self.M, hidden_dim)

    def forward(self, features):
        # features: list of tensors, each [B, L_i, dim_i] or [B, dim_i]
        feats = [f.unsqueeze(1) if f.dim() == 2 else f for f in features]
        Qs = [self.to_q[i](feats[i]) for i in range(self.M)]
        Ks = [self.to_k[i](feats[i]) for i in range(self.M)]
        Vs = [self.to_v[i](feats[i]) for i in range(self.M)]

        outs = []
        for i in range(self.M):
            Qi = Qs[i]
            agg = torch.zeros_like(Qi)
            for j in range(self.M):
                if i == j:
                    continue
                # Pre-gate
                if self.use_pregate:
                    p_q, p_k = self.pregates[i](Qi, Ks[j])
                else:
                    p_q = p_k = None
                # Attention
                agg = agg + self.attn(Qi, Ks[j], Vs[j], p_q, p_k)
            # CAG
            Ci = agg
            if self.use_cag:
                Ci = self.cags[i](Ci, Qi)
            outs.append(Ci)

        fused = torch.cat(outs, dim=-1)  # [B, Lmax, M*d]
        vec = fused[:, 0, :]  # 取第一时间步
        return self.out_fc(vec)  # [B, hidden_dim]


# —— DynamicPCAGModule —— #
class MultmodalaAttention(PCAGModule):
    def __init__(self,
                 modal_dims,
                 hidden_dim,
                 use_pregate=True,
                 use_cag=True,
                 attn_dim=None,
                 use_tanh=False,
                 dynamic_scale=1.0):
        super().__init__(modal_dims, hidden_dim,
                         use_pregate, use_cag,
                         attn_dim, use_tanh)
        self.dynamic_scale = dynamic_scale
        self.dynamic_gates = nn.ModuleList([
            nn.Linear(self.attn_dim, 1) for _ in modal_dims
        ])

    def forward(self, features):
        feats = [f.unsqueeze(1) if f.dim() == 2 else f for f in features]
        Qs = [self.to_q[i](feats[i]) for i in range(self.M)]
        Ks = [self.to_k[i](feats[i]) for i in range(self.M)]
        Vs = [self.to_v[i](feats[i]) for i in range(self.M)]

        outs = []
        for i in range(self.M):
            Qi = Qs[i]
            agg = torch.zeros_like(Qi)
            for j in range(self.M):
                if i == j:
                    continue

                # —— 1. 预门控（可选） —— #
                if self.use_pregate:
                    p_q, p_k = self.pregates[i](Qi, Ks[j])
                else:
                    p_q = p_k = None

                # —— 2. 动态门控（始终生效） —— #
                d_q = torch.sigmoid(self.dynamic_gates[i](Qi)) * self.dynamic_scale
                d_k = torch.sigmoid(self.dynamic_gates[j](Ks[j])) * self.dynamic_scale

                if self.use_pregate:
                    # 在 pregate 输出上做放缩
                    p_q = p_q * (1 + d_q)
                    p_k = p_k * (1 + d_k)
                    Q_mod, K_mod = Qi, Ks[j]
                else:
                    # 直接在 Q/K 特征上做缩放
                    Q_mod = Qi * (1 + d_q)
                    K_mod = Ks[j] * (1 + d_k)

                # 注意力计算：把 Q_mod/K_mod/Vj 送进去，p_q/p_k 可传 None
                agg = agg + self.attn(Q_mod, K_mod, Vs[j], p_q, p_k)

            Ci = agg
            if self.use_cag:
                Ci = self.cags[i](Ci, Qi)
            outs.append(Ci)

        fused = torch.cat(outs, dim=-1)
        vec = fused[:, 0, :]
        return self.out_fc(vec)


# —— PCAGProbabilisticModule —— #
class PCAGProbabilisticModule(PCAGModule):
    def __init__(self,
                 modal_dims,
                 hidden_dim,
                 use_pregate=True,
                 use_cag=True,
                 attn_dim=None,
                 use_tanh=False):
        super().__init__(modal_dims, hidden_dim,
                         use_pregate, use_cag,
                         attn_dim, use_tanh)
        if use_pregate:
            self.mu_layers = nn.ModuleList([nn.Linear(self.attn_dim, 1) for _ in modal_dims])
            self.logvar_layers = nn.ModuleList([nn.Linear(self.attn_dim, 1) for _ in modal_dims])
            self.dynamic_scale = nn.Parameter(torch.ones(1))

    def forward(self, features):
        feats = [f.unsqueeze(1) if f.dim() == 2 else f for f in features]
        Qs = [self.to_q[i](feats[i]) for i in range(self.M)]
        Ks = [self.to_k[i](feats[i]) for i in range(self.M)]
        Vs = [self.to_v[i](feats[i]) for i in range(self.M)]

        kl_loss = 0.0
        outs = []
        for i in range(self.M):
            Qi = Qs[i]
            agg = torch.zeros_like(Qi)
            for j in range(self.M):
                if i == j:
                    continue
                if self.use_pregate:
                    mu = self.mu_layers[i](Qi)
                    logvar = self.logvar_layers[i](Qi)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    d_gate = mu + self.dynamic_scale * std * eps
                    kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    p_q, p_k = self.pregates[i](Qi, Ks[j])
                    p_q = p_q * torch.sigmoid(d_gate)
                    p_k = p_k * torch.sigmoid(d_gate)
                else:
                    p_q = p_k = None
                agg = agg + self.attn(Qi, Ks[j], Vs[j], p_q, p_k)
            Ci = agg
            if self.use_cag:
                Ci = self.cags[i](Ci, Qi)
            outs.append(Ci)

        fused = torch.cat(outs, dim=-1)
        vec = fused[:, 0, :]
        out, kl = self.out_fc(vec), kl_loss
        return out, kl


# —— 主流程示例 —— #
if __name__ == '__main__':
    # —— 1) 配置 —— #
    xlsx_path    = '../PCAG_fusion_file/0501/all_comment_0501.xlsx'
    # [文本, 图片1, 图片2, 视频] 的列宽
    modal_dims   = [768, 2048, 2048, 2048]
    batch_size   = 128
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'


    # —— 2) 读取并按段切分特征 —— #
    df       = pd.read_excel(xlsx_path)
    features = []
    start    = 0
    for d in modal_dims:
        end = start + d
        arr = torch.tensor(
            df.iloc[:, start:end].values,
            dtype=torch.float32
        ).to(device)
        features.append(arr)
        start = end

    # 最后一列为 label
    labels = torch.tensor(
        df.iloc[:, -1].values,
        dtype=torch.long
    ).to(device)

    # —— 3) 构建 DataLoader —— #
    dataset = TensorDataset(*features, labels)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # —— 4) 模型配置 & 实例化 —— #
    use_pregate   = True
    use_cag       = True
    attn_dim      = 256
    use_tanh      = True
    dynamic_scale = 1.5
    hidden_dim    = 1280
    floder = 'Dynamic_all'

    # 4) 选择并实例化模型（取消其余两行注释）
    # model = PCAGModule(
    #     modal_dims=modal_dims,
    #     hidden_dim=hidden_dim,
    #     use_pregate=use_pregate,
    #     use_cag=use_cag,
    #     attn_dim=attn_dim,
    #     use_tanh=use_tanh
    # )
    model = MultmodalaAttention(
        modal_dims=modal_dims,
        hidden_dim=hidden_dim,
        use_pregate=use_pregate,
        use_cag=use_cag,
        attn_dim=attn_dim,
        use_tanh=use_tanh,
        dynamic_scale=dynamic_scale
    )


    model.to(device).eval()

    # —— 5) 批量推理 —— #
    fused_list, label_list = [], []
    with torch.no_grad():
        for batch in loader:
            *bs, lb = batch
            # bs 是一个 list：[text_tensor, img1_tensor, img2_tensor, vid_tensor]
            out = model(bs)
            if isinstance(out, tuple):  # Probabilistic 模块返回 (out, kl)
                out, _ = out
            fused_list.append(out.cpu())
            label_list.append(lb.cpu())

    fused = torch.cat(fused_list, dim=0).numpy()
    labs  = torch.cat(label_list, dim=0).numpy()

    # —— 6) 保存到 Excel —— #
    df_out     = pd.DataFrame(fused)
    df_out['label'] = labs
    output_dir = f'../PCAG_fusion_file/0501/{floder}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'all_comments_fusion_{hidden_dim}.xlsx')
    df_out.to_excel(output_path, index=False)

    print(f"Saved fused features and labels to {output_path}")
