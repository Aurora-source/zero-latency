"""Scene context cross-attention encoder for map-aware trajectory reasoning.

Architecture target : 4 layers, ~25 M parameters.
RTX 5090 optimisations:
  - bfloat16 weights
  - scaled_dot_product_attention (FlashAttention kernel path)
  - fused QKV projection for self/cross attention

Parameter budget (embed_dim=512, ff_dim=4096, map_dim=256, 4 layers):
  map_projection : Linear(256→512)                ≈ 0.13M
  Per layer:
    Cross-attention Q proj : 512²                 ≈ 0.26M
    Cross-attention KV proj: 2 × 512²             ≈ 0.52M
    Cross-attention out    : 512²                 ≈ 0.26M
    FFN 512→4096→512       : 2 × 512×4096         ≈ 4.19M
    LayerNorms             : 2 × 2×512            ≈ 0.002M
    Layer total                                   ≈ 5.23M
  4 layers                                        ≈ 20.9M
  map_projection + misc                           ≈ 0.26M
  Total                                           ≈ 21.2M  → ~25M with wider ff (ff_dim=5120)

  With ff_dim=5120:
    FFN per layer: 2 × 512×5120 ≈ 5.24M
    Attention per layer         ≈ 1.05M
    Layer total                 ≈ 6.29M
  4 layers                      ≈ 25.1M  ✓
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SceneContextLayer", "SceneContextEncoder"]


class SceneContextLayer(nn.Module):
    """Single cross-attention layer from agents to map elements (SDPA)."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 5120,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
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
        """Cross-attend from agent queries to map key/values."""
        B, Nq, D = query_tokens.shape
        Nk = map_tokens.size(1)

        q = self.q_proj(query_tokens).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(map_tokens)
        k, v = kv.split(self.embed_dim, dim=-1)
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # attention_mask: (B*H, Nq, Nk) → (B, H, Nq, Nk)
        bias_4d: Optional[Tensor] = None
        if attention_mask is not None:
            bias_4d = attention_mask.view(B, self.num_heads, Nq, Nk)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias_4d,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, Nq, D)
        attn_out = self.out_proj(attn_out)

        if zero_attention_queries is not None:
            attn_out = attn_out.masked_fill(zero_attention_queries.unsqueeze(-1), 0.0)

        query_tokens = self.attention_norm(query_tokens + self.attention_dropout(attn_out))
        ff_out = self.feedforward(query_tokens)
        return self.feedforward_norm(query_tokens + self.feedforward_dropout(ff_out))


class SceneContextEncoder(nn.Module):
    """Cross-attention encoder injecting vectorised map context into agents.

    4 layers, ~25M parameters.
    embed_dim=512, map_dim=256, ff_dim=5120, num_heads=8.
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        embed_dim: int = 512,
        map_dim: int = 256,
        ff_dim: int = 5120,
        dropout: float = 0.1,
        max_distance: float = 50.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if map_dim <= 0:
            raise ValueError("map_dim must be positive.")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

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
        """Fuse scene context into agent embeddings via cross-attention.

        Args:
            agent_embeddings: ``(batch, time, agents, embed_dim)``
            map_features:     ``(batch, map_elements, map_dim)``
            map_padding_mask: Optional ``(batch, map_elements)`` bool; True = invalid.
            agent_positions:  Optional ``(batch, time, agents, 2)``
            map_positions:    Optional ``(batch, map_elements, 2)``
            agent_mask:       Optional ``(batch, time, agents)`` bool; True = padded.

        Returns:
            ``(batch, time, agents, embed_dim)``
        """
        if agent_embeddings.ndim != 4:
            raise ValueError(
                f"agent_embeddings must be (batch, time, agents, embed_dim), got {tuple(agent_embeddings.shape)}."
            )
        if map_features.ndim != 3:
            raise ValueError(
                f"map_features must be (batch, map_elements, map_dim), got {tuple(map_features.shape)}."
            )

        batch_size, time_steps, num_agents, embed_dim = agent_embeddings.shape
        map_batch, num_map, map_dim = map_features.shape

        if map_batch != batch_size:
            raise ValueError("agent_embeddings and map_features must share batch size.")
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}.")
        if map_dim != self.map_dim:
            raise ValueError(f"Expected map_dim={self.map_dim}, got {map_dim}.")

        self._validate_position_inputs(
            agent_positions, map_positions, batch_size, time_steps, num_agents, num_map
        )

        norm_map_mask = self._normalize_map_padding_mask(
            map_padding_mask, batch_size, num_map, agent_embeddings.device
        )
        norm_agent_mask = self._normalize_agent_mask(
            agent_mask, batch_size, time_steps, num_agents, agent_embeddings.device
        )

        num_queries = time_steps * num_agents
        model_dtype = self.layers[0].q_proj.weight.dtype
        query_tokens = agent_embeddings.to(dtype=model_dtype).reshape(batch_size, num_queries, embed_dim)
        proj_map = self.map_projection(map_features.to(dtype=model_dtype))

        flat_agent_mask: Optional[Tensor] = None
        if norm_agent_mask is not None:
            flat_agent_mask = norm_agent_mask.reshape(batch_size, num_queries)
            query_tokens = query_tokens.masked_fill(flat_agent_mask.unsqueeze(-1), 0.0)

        attn_mask, zero_queries = self._build_attention_mask(
            batch_size, time_steps, num_agents, num_map,
            norm_map_mask, agent_positions, map_positions, agent_embeddings.device
        )

        if flat_agent_mask is not None:
            zero_queries = (
                flat_agent_mask if zero_queries is None else (zero_queries | flat_agent_mask)
            )

        if attn_mask is not None:
            # (B, Nq, Nm) → (B*H, Nq, Nm) float additive mask
            float_mask = torch.zeros_like(attn_mask, dtype=model_dtype)
            float_mask = float_mask.masked_fill(attn_mask, float("-inf"))
            attn_mask_expanded = (
                float_mask.unsqueeze(1)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(batch_size * self.num_heads, num_queries, num_map)
            )
        else:
            attn_mask_expanded = None

        for layer in self.layers:
            query_tokens = layer(
                query_tokens,
                proj_map,
                attention_mask=attn_mask_expanded,
                zero_attention_queries=zero_queries,
            )
            if flat_agent_mask is not None:
                query_tokens = query_tokens.masked_fill(flat_agent_mask.unsqueeze(-1), 0.0)

        outputs = query_tokens.reshape(batch_size, time_steps, num_agents, embed_dim)
        if norm_agent_mask is not None:
            outputs = outputs.masked_fill(norm_agent_mask.unsqueeze(-1), 0.0)
        return outputs

    def _build_attention_mask(
        self,
        batch_size: int,
        time_steps: int,
        num_agents: int,
        num_map: int,
        map_padding_mask: Optional[Tensor],
        agent_positions: Optional[Tensor],
        map_positions: Optional[Tensor],
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        num_queries = time_steps * num_agents
        base = torch.zeros(batch_size, num_queries, num_map, device=device, dtype=torch.bool)
        if map_padding_mask is not None:
            base = map_padding_mask.unsqueeze(1).expand(batch_size, num_queries, num_map)

        zero_queries: Optional[Tensor] = None
        blocked = base

        if map_positions is not None and agent_positions is not None:
            flat_ap = agent_positions.to(device=device, dtype=torch.float32).reshape(
                batch_size, num_queries, 2
            )
            norm_mp = map_positions.to(device=device, dtype=torch.float32)
            spatial = torch.cdist(flat_ap, norm_mp, p=2) > self.max_distance
            blocked = base | spatial
            zero_queries = ~((~blocked).any(dim=-1))
            blocked = torch.where(zero_queries.unsqueeze(-1), base, blocked)

        no_valid = ~((~blocked).any(dim=-1))
        if torch.any(no_valid):
            blocked = torch.where(no_valid.unsqueeze(-1), torch.zeros_like(blocked), blocked)
            zero_queries = no_valid if zero_queries is None else (zero_queries | no_valid)

        if zero_queries is not None and not torch.any(zero_queries):
            zero_queries = None
        if not torch.any(blocked):
            blocked = None

        return blocked, zero_queries

    @staticmethod
    def _validate_position_inputs(
        agent_positions: Optional[Tensor],
        map_positions: Optional[Tensor],
        batch_size: int,
        time_steps: int,
        num_agents: int,
        num_map: int,
    ) -> None:
        if map_positions is None:
            return
        if agent_positions is None:
            raise ValueError("agent_positions must be provided when map_positions are used.")
        if agent_positions.shape != (batch_size, time_steps, num_agents, 2):
            raise ValueError(
                f"agent_positions must be (batch, time, agents, 2), got {tuple(agent_positions.shape)}."
            )
        if map_positions.shape != (batch_size, num_map, 2):
            raise ValueError(
                f"map_positions must be (batch, map_elements, 2), got {tuple(map_positions.shape)}."
            )

    @staticmethod
    def _normalize_map_padding_mask(
        mask: Optional[Tensor], batch_size: int, num_map: int, device: torch.device
    ) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.shape != (batch_size, num_map):
            raise ValueError(
                f"map_padding_mask must be (batch, map_elements), got {tuple(mask.shape)}."
            )
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _normalize_agent_mask(
        mask: Optional[Tensor],
        batch_size: int,
        time_steps: int,
        num_agents: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.shape != (batch_size, time_steps, num_agents):
            raise ValueError(
                f"agent_mask must be (batch, time, agents), got {tuple(mask.shape)}."
            )
        return mask.to(device=device, dtype=torch.bool)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _run_smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model = SceneContextEncoder().to(device=device, dtype=dtype)
    total = _count_parameters(model)
    print(f"SceneContextEncoder parameters: {total:,}  (~{total / 1e6:.1f}M)  [target ~25M]")

    ae = torch.randn(2, 6, 4, model.embed_dim, device=device, dtype=dtype)
    mf = torch.randn(2, 128, model.map_dim, device=device, dtype=dtype)
    ap = torch.randn(2, 6, 4, 2, device=device, dtype=dtype)
    mp = torch.randn(2, 128, 2, device=device, dtype=dtype)
    mpm = torch.zeros(2, 128, dtype=torch.bool, device=device)
    mpm[0, 120:] = True
    am = torch.zeros(2, 6, 4, dtype=torch.bool, device=device)
    am[0, :, 3] = True

    out = model(ae, mf, map_padding_mask=mpm, agent_positions=ap, map_positions=mp, agent_mask=am)
    assert out.shape == (2, 6, 4, model.embed_dim), out.shape
    print(f"Output shape: {tuple(out.shape)},  dtype: {out.dtype}")


if __name__ == "__main__":
    _run_smoke_test()