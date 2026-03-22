"""Social interaction transformer over agent neighbourhoods.

Architecture target : 6 layers, ~35 M parameters.
RTX 5090 optimisations:
  - bfloat16 weights
  - scaled_dot_product_attention (FlashAttention kernel path)
  - fused QKV projection

Parameter budget (embed_dim=512, ff_dim=4096, num_heads=8, 6 layers):
  Per layer:
    Attention (fused QKV + out) : 4 × 512² ≈ 1.048M
    FFN 512→4096→512            : 2 × 512×4096 ≈ 4.194M
    LayerNorms + misc           ≈ 0.003M
    Layer total                 ≈ 5.245M
  6 layers                      ≈ 31.5M
  type_projection + type_embedding + log_sigma ≈ 1.1M
  Total                         ≈ 32.6M  → ≈35M with biases/LN
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SocialTransformerLayer", "SocialTransformer"]


class SocialTransformerLayer(nn.Module):
    """Single spatial transformer layer over the agent dimension (SDPA)."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 4096,
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

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
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

    def forward(self, inputs: Tensor, attention_bias: Optional[Tensor] = None) -> Tensor:
        """Socially-aware self-attention via SDPA."""
        BT, N, D = inputs.shape

        qkv = self.qkv_proj(inputs)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = q.view(BT, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(BT, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(BT, N, self.num_heads, self.head_dim).transpose(1, 2)

        # attention_bias: (BT*H, N, N) → reshape to (BT, H, N, N) for SDPA
        bias_4d: Optional[Tensor] = None
        if attention_bias is not None:
            bias_4d = attention_bias.view(BT, self.num_heads, N, N)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias_4d,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(BT, N, D)
        attn_out = self.out_proj(attn_out)

        inputs = self.attention_norm(inputs + self.attention_dropout(attn_out))
        ff_out = self.feedforward(inputs)
        return self.feedforward_norm(inputs + self.feedforward_dropout(ff_out))


class SocialTransformer(nn.Module):
    """Graph-style transformer for agent-to-agent interactions.

    6 layers, ~35M parameters.
    embed_dim=512, ff_dim=4096, num_heads=8.
    """

    def __init__(
        self,
        num_layers: int = 6,
        num_heads: int = 8,
        embed_dim: int = 512,
        ff_dim: int = 5888,
        dropout: float = 0.1,
        distance_sigma: float = 10.0,
        num_types: int = 3,
        type_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
        if distance_sigma <= 0.0:
            raise ValueError("distance_sigma must be positive.")
        if num_types <= 0:
            raise ValueError("num_types must be positive.")
        if type_embedding_dim <= 0:
            raise ValueError("type_embedding_dim must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

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
        """Apply spatial social attention across agents at each timestep.

        Args:
            inputs:      ``(batch, time, agents, embed_dim)``
            positions:   ``(batch, time, agents, 2)``
            agent_mask:  Optional ``(batch, time, agents)`` bool mask; True = padded.
            agent_types: Optional ``(batch, agents)`` or ``(batch, time, agents)`` int.

        Returns:
            ``(batch, time, agents, embed_dim)``
        """
        if inputs.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (batch, time, agents, embed_dim), got {tuple(inputs.shape)}."
            )
        if positions.ndim != 4:
            raise ValueError(
                f"Expected positions (batch, time, agents, 2), got {tuple(positions.shape)}."
            )

        batch_size, time_steps, num_agents, embed_dim = inputs.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}.")
        if positions.shape[:3] != inputs.shape[:3] or positions.size(-1) != 2:
            raise ValueError("positions shape must match (batch, time, agents, 2).")

        model_dtype = self.layers[0].qkv_proj.weight.dtype
        outputs = inputs.to(dtype=model_dtype)

        flat_positions = positions.to(device=inputs.device, dtype=torch.float32).reshape(
            batch_size * time_steps, num_agents, 2
        )

        normalized_types = self._normalize_agent_types(
            agent_types, batch_size, time_steps, num_agents, inputs.device
        )
        if normalized_types is not None:
            type_features = self.type_embedding(normalized_types).to(dtype=model_dtype)
            outputs = self.type_projection(torch.cat((outputs, type_features), dim=-1))

        outputs = outputs.reshape(batch_size * time_steps, num_agents, embed_dim)

        normalized_mask = self._normalize_agent_mask(
            agent_mask, batch_size, time_steps, num_agents, inputs.device
        )
        flat_mask: Optional[Tensor] = None
        if normalized_mask is not None:
            flat_mask = normalized_mask.reshape(batch_size * time_steps, num_agents)
            outputs = outputs.masked_fill(flat_mask.unsqueeze(-1), 0.0)

        attention_bias = self._build_attention_bias(flat_positions, flat_mask, outputs.dtype)

        for layer in self.layers:
            outputs = layer(outputs, attention_bias=attention_bias)
            if flat_mask is not None:
                outputs = outputs.masked_fill(flat_mask.unsqueeze(-1), 0.0)

        outputs = outputs.reshape(batch_size, time_steps, num_agents, embed_dim)
        if normalized_mask is not None:
            outputs = outputs.masked_fill(normalized_mask.unsqueeze(-1), 0.0)
        return outputs

    def _build_attention_bias(
        self,
        flat_positions: Tensor,
        flat_mask: Optional[Tensor],
        output_dtype: torch.dtype,
    ) -> Tensor:
        sigma = self.log_distance_sigma.float().exp().clamp_min(1e-6)
        dist = torch.cdist(flat_positions, flat_positions, p=2)
        bias = (-dist / sigma).to(dtype=output_dtype)

        if flat_mask is not None:
            eff_mask = flat_mask.clone()
            eff_mask[eff_mask.all(dim=1)] = False
            fill = torch.finfo(bias.dtype).min
            bias = bias.masked_fill(eff_mask[:, None, :], fill)

        BT, N, _ = bias.shape
        return bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(
            BT * self.num_heads, N, N
        )

    def _normalize_agent_types(
        self,
        agent_types: Optional[Tensor],
        batch_size: int,
        time_steps: int,
        num_agents: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if agent_types is None:
            return None
        if agent_types.shape == (batch_size, num_agents):
            normalized = agent_types.unsqueeze(1).expand(batch_size, time_steps, num_agents)
        elif agent_types.shape == (batch_size, time_steps, num_agents):
            normalized = agent_types
        else:
            raise ValueError(
                f"agent_types must be (batch, agents) or (batch, time, agents), got {tuple(agent_types.shape)}."
            )
        normalized = normalized.to(device=device, dtype=torch.long)
        if torch.any((normalized < 0) | (normalized >= self.num_types)):
            raise ValueError(
                f"agent_types must be in [0, {self.num_types - 1}]."
            )
        return normalized

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

    model = SocialTransformer().to(device=device, dtype=dtype)
    total = _count_parameters(model)
    print(f"SocialTransformer parameters: {total:,}  (~{total / 1e6:.1f}M)  [target ~35M]")

    emb = torch.randn(2, 6, 4, model.embed_dim, device=device, dtype=dtype)
    pos = torch.randn(2, 6, 4, 2, device=device, dtype=dtype)
    mask = torch.zeros(2, 6, 4, dtype=torch.bool, device=device)
    mask[0, :, 3] = True
    types = torch.randint(0, model.num_types, (2, 4), device=device)

    out = model(emb, pos, agent_mask=mask, agent_types=types)
    assert out.shape == (2, 6, 4, model.embed_dim), out.shape
    print(f"Output shape: {tuple(out.shape)},  dtype: {out.dtype}")


if __name__ == "__main__":
    _run_smoke_test()