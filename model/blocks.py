
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, input_len, patch_len, stride, in_channels, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.proj(x)  # (B, d_model, num_patches)
        x = x.permute(0, 2, 1)  # (B, num_patches, d_model)
        return x

class ChannelMixer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        # x: (B, N, D)
        return self.fc(x)

class DecompositionHead(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        # x: (B, T, C)
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal
