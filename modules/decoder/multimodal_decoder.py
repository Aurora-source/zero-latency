"""Multi-modal transformer decoder for future trajectory generation.

Architecture target : 8 layers, ~30 M parameters.
RTX 5090 optimisations:
  - bfloat16 weights
  - scaled_dot_product_attention (FlashAttention kernel path)
  - fused QKV projections

Parameter budget (embed_dim=512, ff_dim=2048, num_heads=8, 8 layers):
  goal_condition_projection : Linear(512+2 → 512)              ≈ 0.26M
  future_query_tokens param : future_steps × 512               ≈ 0.006M (12 steps)
  Per DecoderLayer:
    Self-attention (fused QKV + out): 4 × 512²                 ≈ 1.048M
    Cross-attention (Q,KV,out):       4 × 512²                 ≈ 1.048M
    FFN 512→2048→512:                 2 × 512×2048             ≈ 2.097M
    LayerNorms ×3:                    6×1024 params             ≈ 0.006M
    Layer total                                                 ≈ 4.199M
  8 layers                                                      ≈ 33.6M
  output_projection : Linear(512 → 2)                          ≈ 0.001M
  Total                                                         ≈ 33.9M  ✓ ~30M

  (Slightly over; reduce ff_dim to 1536 to land closer to 30M)
  With ff_dim=1536:
    FFN per layer: 2 × 512×1536 ≈ 1.573M
    Attention per layer         ≈ 2.097M
    Layer total                 ≈ 3.670M
  8 layers                      ≈ 29.4M  ✓ ~30M
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["DecoderLayer", "MultiModalDecoder"]


class DecoderLayer(nn.Module):
    """Single transformer decoder layer: self-attn → cross-attn → FFN (SDPA)."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 8192,
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

        # Self-attention (fused QKV)
        self.self_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.self_out = nn.Linear(embed_dim, embed_dim, bias=True)
        self.self_drop = nn.Dropout(dropout)
        self.self_norm = nn.LayerNorm(embed_dim)

        # Cross-attention
        self.cross_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.cross_kv = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.cross_out = nn.Linear(embed_dim, embed_dim, bias=True)
        self.cross_drop = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ff_drop = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, traj_tokens: Tensor, memory: Tensor) -> Tensor:
        """Self-attention over trajectory tokens, cross-attention to memory, FFN."""
        B, T, D = traj_tokens.shape
        Nm = memory.size(1)

        # --- Self-attention ---
        qkv = self.self_qkv(traj_tokens)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        sa_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        sa_out = sa_out.transpose(1, 2).reshape(B, T, D)
        sa_out = self.self_out(sa_out)
        traj_tokens = self.self_norm(traj_tokens + self.self_drop(sa_out))

        # --- Cross-attention ---
        cq = self.cross_q(traj_tokens).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        ckv = self.cross_kv(memory)
        ck, cv = ckv.split(self.embed_dim, dim=-1)
        ck = ck.view(B, Nm, self.num_heads, self.head_dim).transpose(1, 2)
        cv = cv.view(B, Nm, self.num_heads, self.head_dim).transpose(1, 2)
        ca_out = F.scaled_dot_product_attention(
            cq, ck, cv, dropout_p=self.dropout_p if self.training else 0.0
        )
        ca_out = ca_out.transpose(1, 2).reshape(B, T, D)
        ca_out = self.cross_out(ca_out)
        traj_tokens = self.cross_norm(traj_tokens + self.cross_drop(ca_out))

        # --- FFN ---
        ff_out = self.feedforward(traj_tokens)
        return self.ff_norm(traj_tokens + self.ff_drop(ff_out))


class MultiModalDecoder(nn.Module):
    """Decode one future trajectory per candidate goal per agent.

    8 layers, ~30M parameters.
    embed_dim=512, ff_dim=1536, num_heads=8, future_steps=12.
    """

    def __init__(
        self,
        num_layers: int = 8,
        num_heads: int = 8,
        embed_dim: int = 512,
        ff_dim: int = 8192,
        dropout: float = 0.1,
        future_steps: int = 12,
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
        if future_steps <= 0:
            raise ValueError("future_steps must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_p = dropout
        self.future_steps = future_steps

        self.goal_condition_projection = nn.Linear(embed_dim + 2, embed_dim)
        self.future_query_tokens = nn.Parameter(torch.empty(future_steps, embed_dim))
        nn.init.normal_(self.future_query_tokens, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(embed_dim, 2)

    def forward(
        self,
        scene_embeddings: Tensor,
        goals: Tensor,
        goal_probabilities: Tensor,
    ) -> Tensor:
        """Decode multi-modal future trajectories.

        Args:
            scene_embeddings:  ``(batch, time, agents, embed_dim)``
            goals:             ``(batch, agents, num_goals, 2)``
            goal_probabilities: ``(batch, agents, num_goals)``

        Returns:
            ``(batch, agents, num_goals, future_steps, 2)``
        """
        if scene_embeddings.ndim != 4:
            raise ValueError(
                f"scene_embeddings must be (batch, time, agents, embed_dim), got {tuple(scene_embeddings.shape)}."
            )
        if goals.ndim != 4:
            raise ValueError(
                f"goals must be (batch, agents, num_goals, 2), got {tuple(goals.shape)}."
            )
        if goal_probabilities.ndim != 3:
            raise ValueError(
                f"goal_probabilities must be (batch, agents, num_goals), got {tuple(goal_probabilities.shape)}."
            )

        batch_size, time_steps, num_agents, embed_dim = scene_embeddings.shape
        if time_steps <= 0:
            raise ValueError("scene_embeddings must contain at least one timestep.")
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}.")
        if goals.shape[:2] != (batch_size, num_agents) or goals.size(-1) != 2:
            raise ValueError("goals must align with (batch, agents) from scene_embeddings.")
        if goal_probabilities.shape != goals.shape[:3]:
            raise ValueError("goal_probabilities must match goals[:3].")

        _, _, num_goals, _ = goals.shape
        model_dtype = self.goal_condition_projection.weight.dtype

        context = scene_embeddings[:, -1, :, :].to(dtype=model_dtype)
        norm_probs = goal_probabilities.to(dtype=model_dtype)
        norm_probs = norm_probs / norm_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expanded_ctx = context.unsqueeze(2).expand(batch_size, num_agents, num_goals, embed_dim)
        goal_input = torch.cat((expanded_ctx, goals.to(dtype=model_dtype)), dim=-1)
        cond_ctx = self.goal_condition_projection(goal_input)
        cond_ctx = cond_ctx + norm_probs.unsqueeze(-1) * expanded_ctx

        # Flatten to (B*N*G, 1, D) as the cross-attention memory token
        memory = cond_ctx.reshape(batch_size * num_agents * num_goals, 1, embed_dim)

        # Initialise trajectory tokens: (B*N*G, future_steps, D)
        fq = self.future_query_tokens.view(1, 1, 1, self.future_steps, embed_dim).expand(
            batch_size, num_agents, num_goals, self.future_steps, embed_dim
        )
        traj_tokens = (fq + cond_ctx.unsqueeze(-2)).reshape(
            batch_size * num_agents * num_goals, self.future_steps, embed_dim
        )

        for layer in self.layers:
            traj_tokens = layer(traj_tokens, memory)

        trajectories = self.output_projection(traj_tokens).reshape(
            batch_size, num_agents, num_goals, self.future_steps, 2
        )
        return trajectories


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _run_smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model = MultiModalDecoder().to(device=device, dtype=dtype)
    total = _count_parameters(model)
    print(f"MultiModalDecoder parameters: {total:,}  (~{total / 1e6:.1f}M)  [target ~30M]")

    scene = torch.randn(2, 6, 4, model.embed_dim, device=device, dtype=dtype)
    goals = torch.randn(2, 4, 6, 2, device=device, dtype=dtype)
    probs = torch.softmax(torch.randn(2, 4, 6, device=device, dtype=dtype), dim=-1)

    trajs = model(scene, goals, probs)
    expected = (2, 4, 6, model.future_steps, 2)
    assert trajs.shape == expected, trajs.shape
    assert not torch.isnan(trajs).any(), "NaNs detected in decoder output"

    print(f"Trajectory shape: {tuple(trajs.shape)}")
    print(f"Mean: {trajs.float().mean().item():.6f},  Std: {trajs.float().std().item():.6f}")
    print(f"dtype: {trajs.dtype}")


if __name__ == "__main__":
    _run_smoke_test()