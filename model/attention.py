
import torch
import torch.nn as nn
import math

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model // n_heads)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv_proj(x)  # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, -1).transpose(1, 2)  # (B, H, N, Dh)
        k = k.view(B, N, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, N, self.n_heads, -1).transpose(1, 2)

        # ProbSparse sampling
        score = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        k_sparse = max(1, int(score.size(-1) * 0.1))
        sparsity_threshold = torch.topk(score, k=k_sparse, dim=-1)[0][..., -1, None]
        mask = score >= sparsity_threshold
        score = score.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # (B, H, N, Dh)
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(output)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
