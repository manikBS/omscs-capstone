import torch
import torch.nn as nn
from model.blocks import PatchEmbedding, ChannelMixer, DecompositionHead
from model.attention import TransformerEncoder, ProbSparseAttention
from model.quantile_head import QuantileRegressionHead


class PatchTST_SOTA(nn.Module):
    def __init__(self, input_len, pred_len, num_features, d_model=128, patch_len=16, stride=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.decompose = DecompositionHead()
        self.patch_embed = PatchEmbedding(input_len, patch_len, stride, num_features, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, dropout)
        self.channel_mixer = ChannelMixer(d_model)
        self.head = QuantileRegressionHead(d_model, pred_len, quantiles=[0.1, 0.5, 0.9])

    def forward(self, x):
        x = self.decompose(x)
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.channel_mixer(x)
        out = self.head(x)
        return out

    def predict(self, input_window):
        """
        input_window: torch.Tensor of shape (1, input_len, num_features)
        Returns: torch.Tensor of shape (1, pred_len, num_quantiles)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_window)
        return output