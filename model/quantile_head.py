
import torch
import torch.nn as nn

class QuantileRegressionHead(nn.Module):
    def __init__(self, d_model, pred_len, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.heads = nn.ModuleList([nn.Linear(d_model, pred_len) for _ in quantiles])

    def forward(self, x):
        # x: (B, N, D) => only take the last patch representation
        x = x[:, -1, :]  # (B, D)
        return torch.stack([head(x) for head in self.heads], dim=-1)  # (B, pred_len, num_quantiles)
