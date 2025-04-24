from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class GaussianMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first)
        self.linear = nn.Linear(embed_dim*num_heads, embed_dim)
        self.guass_attn_weights_for_plot = None
        self.raw_attn_weights_for_plot = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        attn_output, attn_weights = super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            need_weights=True, average_attn_weights=False
        )

        self.raw_attn_weights_for_plot = attn_weights.detach().cpu()

        B, H, Tq, Tk = attn_weights.shape
        E = query.shape[-1]
        V_start = 2 * E
        V_end = 3 * E
        D = E // H

        v_proj = F.linear(value, self.in_proj_weight[V_start:V_end], self.in_proj_bias[V_start:V_end])
        v_proj = v_proj.view(B, Tk, H, D).transpose(1, 2)

        device = attn_weights.device

        pos_q = torch.arange(Tq, device=device)
        pos_k = torch.arange(Tk, device=device)
        dist = pos_q.unsqueeze(1) - pos_k.unsqueeze(0)
        gaussian_bias = torch.exp(- (dist.float() ** 2) / (2 * (Tq / 4) ** 2))

        gaussian_bias = gaussian_bias.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights * gaussian_bias

        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-5)

        self.guass_attn_weights_for_plot = attn_weights.detach().cpu()

        attn_output = torch.matmul(attn_weights, v_proj)
        attn_output = attn_output.transpose(1, 2).reshape(B, Tq, -1)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.self_attn = GaussianMultiheadAttention(
            embed_dim=self.self_attn.embed_dim,
            num_heads=self.self_attn.num_heads,
            dropout=self.self_attn.dropout,
            batch_first=True
        )

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Self-attention (decoder input attends to itself)
        self.self_attn = GaussianMultiheadAttention(
            embed_dim=self.self_attn.embed_dim,
            num_heads=self.self_attn.num_heads,
            dropout=self.self_attn.dropout,
            batch_first=True
        )

        # Cross-attention (decoder attends to encoder output)
        self.multihead_attn = GaussianMultiheadAttention(
            embed_dim=self.multihead_attn.embed_dim,
            num_heads=self.multihead_attn.num_heads,
            dropout=self.multihead_attn.dropout,
            batch_first=True
        )

class CustomTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = CustomTransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.pos_encoder(src)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

class CustomTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_encoder_layers=3, dim_feedforward=1024):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, src_key_padding_mask=None):
        src = self.pos_encoder(src)
        output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        #mean_pooled = output.mean(dim=1)
        #output = self.output_layer(mean_pooled)
        last_step = output[:, -1, :]
        output = self.output_layer(last_step)
        return output

