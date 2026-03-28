from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

__all__ = ["DecoderLayer", "MultiModalDecoder"]


class DecoderLayer(nn.Module):

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
        self.self_attention_dropout = nn.Dropout(dropout)
        self.self_attention_norm = nn.LayerNorm(embed_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.cross_attention_norm = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(embed_dim)

    def forward(self, trajectory_tokens: Tensor, memory: Tensor) -> Tensor:

        x, _ = self.self_attention(
            trajectory_tokens,
            trajectory_tokens,
            trajectory_tokens,
            need_weights=False,
        )
        trajectory_tokens = self.self_attention_norm(
            trajectory_tokens + self.self_attention_dropout(x)
        )

        x, _ = self.cross_attention(
            query=trajectory_tokens,
            key=memory,
            value=memory,
            need_weights=False,
        )
        trajectory_tokens = self.cross_attention_norm(
            trajectory_tokens + self.cross_attention_dropout(x)
        )

        x = self.feedforward(trajectory_tokens)
        return self.feedforward_norm(
            trajectory_tokens + self.feedforward_dropout(x)
        )


class MultiModalDecoder(nn.Module):

    def __init__(
        self,
        num_layers: int = 8,
        num_heads: int = 8,
        embed_dim: int = 896,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        future_steps: int = 12,
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
        if future_steps <= 0:
            raise ValueError("future_steps must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

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

        if scene_embeddings.ndim != 4:
            raise ValueError(f"scene_embeddings shape wrong: {tuple(scene_embeddings.shape)}")
        if goals.ndim != 4:
            raise ValueError(f"goals shape wrong: {tuple(goals.shape)}")
        if goal_probabilities.ndim != 3:
            raise ValueError(f"goal_probs shape wrong: {tuple(goal_probabilities.shape)}")

        b, t, a, e = scene_embeddings.shape

        if t <= 0:
            raise ValueError("no timesteps")
        if e != self.embed_dim:
            raise ValueError(f"embed dim mismatch: {e}")

        if goals.shape[:2] != (b, a) or goals.size(-1) != 2:
            raise ValueError("goals shape mismatch")
        if goal_probabilities.shape != goals.shape[:3]:
            raise ValueError("goal probs mismatch")

        _, _, k, _ = goals.shape

        dtype = self.goal_condition_projection.weight.dtype

        ctx = scene_embeddings[:, -1, :, :].to(dtype=dtype)

        prob = goal_probabilities.to(dtype=dtype)
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        ctx_exp = ctx.unsqueeze(2).expand(b, a, k, e)
        goal_in = torch.cat((ctx_exp, goals.to(dtype=dtype)), dim=-1)

        cond = self.goal_condition_projection(goal_in)
        cond = cond + prob.unsqueeze(-1) * ctx_exp

        memory = cond.reshape(b * a * k, 1, e)

        future_tokens = self.future_query_tokens.view(1, 1, 1, self.future_steps, e).expand(
            b, a, k, self.future_steps, e
        )

        traj = future_tokens + cond.unsqueeze(-2)
        traj = traj.reshape(b * a * k, self.future_steps, e)

        for layer in self.layers:
            traj = layer(traj, memory)

        out = self.output_projection(traj).reshape(
            b, a, k, self.future_steps, 2
        )

        return out


def _run_smoke_test() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalDecoder().to(device)

    x = torch.randn(2, 6, 4, model.embed_dim, device=device)
    goals = torch.randn(2, 4, 6, 2, device=device)
    probs = torch.softmax(torch.randn(2, 4, 6, device=device), dim=-1)

    traj = model(x, goals, probs)

    exp = (2, 4, 6, model.future_steps, 2)
    has_nan = torch.isnan(traj).any().item()

    assert traj.shape == exp, f"shape wrong: {tuple(traj.shape)}"
    assert not has_nan

    print(traj.shape)
    print(traj.mean().item())
    print(traj.std().item())
    print(has_nan)


if __name__ == "__main__":
    _run_smoke_test()