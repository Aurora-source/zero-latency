from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

__all__ = ["SocialTransformerLayer", "SocialTransformer"]


class SocialTransformerLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor, attention_bias: Optional[Tensor] = None) -> Tensor:

        x, _ = self.self_attention(
            inputs,
            inputs,
            inputs,
            attn_mask=attention_bias,
            need_weights=False,
        )

        inputs = self.attention_norm(inputs + self.attention_dropout(x))

        x = self.feedforward(inputs)
        return self.feedforward_norm(
            inputs + self.feedforward_dropout(x)
        )


class SocialTransformer(nn.Module):

    def __init__(
        self,
        num_layers: int = 6,
        num_heads: int = 8,
        embed_dim: int = 896,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        distance_sigma: float = 10.0,
        num_types: int = 3,
        type_embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if distance_sigma <= 0.0:
            raise ValueError("distance_sigma must be positive")
        if num_types <= 0:
            raise ValueError("num_types must be positive")
        if type_embedding_dim <= 0:
            raise ValueError("type_embedding_dim must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_p = dropout
        self.num_types = num_types
        self.type_embedding_dim = type_embedding_dim

        self.log_distance_sigma = nn.Parameter(
            torch.tensor(math.log(distance_sigma), dtype=torch.float32)
        )
        self.type_embedding = nn.Embedding(num_types, type_embedding_dim)
        self.type_projection = nn.Linear(embed_dim + type_embedding_dim, embed_dim)

        self.layers = nn.ModuleList(
            [
                SocialTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs: Tensor,
        positions: Tensor,
        agent_mask: Optional[Tensor] = None,
        agent_types: Optional[Tensor] = None,
    ) -> Tensor:

        if inputs.ndim != 4:
            raise ValueError(f"inputs shape wrong: {tuple(inputs.shape)}")
        if positions.ndim != 4:
            raise ValueError(f"positions shape wrong: {tuple(positions.shape)}")

        b, t, a, e = inputs.shape

        if e != self.embed_dim:
            raise ValueError(f"embed dim mismatch: {e}")
        if positions.shape[:3] != inputs.shape[:3] or positions.size(-1) != 2:
            raise ValueError("positions mismatch")

        dtype = self.layers[0].self_attention.in_proj_weight.dtype
        out = inputs.to(dtype=dtype)

        pos = positions.to(device=inputs.device, dtype=torch.float32).reshape(
            b * t, a, 2
        )

        types = self._normalize_agent_types(
            agent_types, b, t, a, inputs.device
        )

        if types is not None:
            type_feat = self.type_embedding(types)
            out = self.type_projection(torch.cat((out, type_feat), dim=-1))

        out = out.reshape(b * t, a, e)

        mask = self._normalize_agent_mask(
            agent_mask, b, t, a, inputs.device
        )

        flat_mask = None
        if mask is not None:
            flat_mask = mask.reshape(b * t, a)
            out = out.masked_fill(flat_mask.unsqueeze(-1), 0.0)

        bias = self._build_attention_bias(pos, flat_mask, out.dtype)

        for layer in self.layers:
            out = layer(out, attention_bias=bias)
            if flat_mask is not None:
                out = out.masked_fill(flat_mask.unsqueeze(-1), 0.0)

        out = out.reshape(b, t, a, e)

        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)

        return out