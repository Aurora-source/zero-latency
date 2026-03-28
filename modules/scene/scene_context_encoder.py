from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ["SceneContextLayer", "SceneContextEncoder"]


class SceneContextLayer(nn.Module):

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

        self.cross_attention = nn.MultiheadAttention(
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

    def forward(
        self,
        query_tokens: Tensor,
        map_tokens: Tensor,
        attention_mask: Optional[Tensor] = None,
        zero_attention_queries: Optional[Tensor] = None,
    ) -> Tensor:

        x, _ = self.cross_attention(
            query=query_tokens,
            key=map_tokens,
            value=map_tokens,
            attn_mask=attention_mask,
            need_weights=False,
        )

        if zero_attention_queries is not None:
            x = x.masked_fill(zero_attention_queries.unsqueeze(-1), 0.0)

        query_tokens = self.attention_norm(
            query_tokens + self.attention_dropout(x)
        )

        x = self.feedforward(query_tokens)
        return self.feedforward_norm(
            query_tokens + self.feedforward_dropout(x)
        )


class SceneContextEncoder(nn.Module):

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        embed_dim: int = 896,
        map_dim: int = 256,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_distance: float = 50.0,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if map_dim <= 0:
            raise ValueError("map_dim must be positive")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.map_dim = map_dim
        self.ff_dim = ff_dim
        self.dropout_p = dropout
        self.max_distance = max_distance

        self.map_projection = nn.Linear(map_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                SceneContextLayer(
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
        agent_embeddings: Tensor,
        map_features: Tensor,
        map_padding_mask: Optional[Tensor] = None,
        agent_positions: Optional[Tensor] = None,
        map_positions: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
    ) -> Tensor:

        if agent_embeddings.ndim != 4:
            raise ValueError(f"agent_embeddings shape wrong: {tuple(agent_embeddings.shape)}")
        if map_features.ndim != 3:
            raise ValueError(f"map_features shape wrong: {tuple(map_features.shape)}")

        b, t, a, e = agent_embeddings.shape
        mb, m, md = map_features.shape

        if mb != b:
            raise ValueError("batch mismatch")
        if e != self.embed_dim:
            raise ValueError(f"embed dim mismatch: {e}")
        if md != self.map_dim:
            raise ValueError(f"map dim mismatch: {md}")

        self._validate_position_inputs(
            agent_positions, map_positions, b, t, a, m
        )

        map_mask = self._normalize_map_padding_mask(
            map_padding_mask, b, m, agent_embeddings.device
        )
        agent_mask = self._normalize_agent_mask(
            agent_mask, b, t, a, agent_embeddings.device
        )

        qn = t * a
        dtype = self.layers[0].cross_attention.in_proj_weight.dtype

        q = agent_embeddings.to(dtype=dtype).reshape(b, qn, e)
        map_proj = self.map_projection(map_features.to(dtype=dtype))

        flat_mask = None
        if agent_mask is not None:
            flat_mask = agent_mask.reshape(b, qn)
            q = q.masked_fill(flat_mask.unsqueeze(-1), 0.0)

        attn_mask, zero_q = self._build_attention_mask(
            b, t, a, m, map_mask,
            agent_positions, map_positions,
            agent_embeddings.device,
        )

        if flat_mask is not None:
            zero_q = flat_mask if zero_q is None else (zero_q | flat_mask)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            ).reshape(
                b * self.num_heads, qn, m
            )

        for layer in self.layers:
            q = layer(q, map_proj, attn_mask, zero_q)
            if flat_mask is not None:
                q = q.masked_fill(flat_mask.unsqueeze(-1), 0.0)

        out = q.reshape(b, t, a, e)

        if agent_mask is not None:
            out = out.masked_fill(agent_mask.unsqueeze(-1), 0.0)

        return out