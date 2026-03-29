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
            raise ValueError("embed_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

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
        self_attention_output, _ = self.self_attention(
            trajectory_tokens,
            trajectory_tokens,
            trajectory_tokens,
            need_weights=False,
        )
        trajectory_tokens = self.self_attention_norm(
            trajectory_tokens + self.self_attention_dropout(self_attention_output)
        )

        cross_attention_output, _ = self.cross_attention(
            query=trajectory_tokens,
            key=memory,
            value=memory,
            need_weights=False,
        )
        trajectory_tokens = self.cross_attention_norm(
            trajectory_tokens + self.cross_attention_dropout(cross_attention_output)
        )

        feedforward_output = self.feedforward(trajectory_tokens)
        return self.feedforward_norm(
            trajectory_tokens + self.feedforward_dropout(feedforward_output)
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
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
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

        if scene_embeddings.ndim != 4:
            raise ValueError(
                "scene_embeddings must have shape (batch, time, agents, embed_dim), "
                f"but received {tuple(scene_embeddings.shape)}."
            )
        if goals.ndim != 4:
            raise ValueError(
                f"goals must have shape (batch, agents, num_goals, 2), but received {tuple(goals.shape)}."
            )
        if goal_probabilities.ndim != 3:
            raise ValueError(
                "goal_probabilities must have shape (batch, agents, num_goals), "
                f"but received {tuple(goal_probabilities.shape)}."
            )

        batch_size, time_steps, num_agents, embed_dim = scene_embeddings.shape
        if time_steps <= 0:
            raise ValueError("scene_embeddings must contain at least one timestep.")
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, but received {embed_dim}."
            )
        if goals.shape[:2] != (batch_size, num_agents) or goals.size(-1) != 2:
            raise ValueError(
                "goals must have shape (batch, agents, num_goals, 2) aligned with scene_embeddings."
            )
        if goal_probabilities.shape != goals.shape[:3]:
            raise ValueError(
                "goal_probabilities must have shape (batch, agents, num_goals) aligned with goals."
            )

        _, _, num_goals, _ = goals.shape
        model_dtype = self.goal_condition_projection.weight.dtype

        context = scene_embeddings[:, -1, :, :].to(dtype=model_dtype)
        normalized_goal_probabilities = goal_probabilities.to(dtype=model_dtype)
        normalized_goal_probabilities = normalized_goal_probabilities / normalized_goal_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        expanded_context = context.unsqueeze(2).expand(batch_size, num_agents, num_goals, embed_dim)
        goal_inputs = torch.cat((expanded_context, goals.to(dtype=model_dtype)), dim=-1)
        conditioned_context = self.goal_condition_projection(goal_inputs)
        conditioned_context = conditioned_context + normalized_goal_probabilities.unsqueeze(-1) * expanded_context

        memory = conditioned_context.reshape(batch_size * num_agents * num_goals, 1, embed_dim)
        future_tokens = self.future_query_tokens.view(1, 1, 1, self.future_steps, embed_dim).expand(
            batch_size,
            num_agents,
            num_goals,
            self.future_steps,
            embed_dim,
        )
        trajectory_tokens = future_tokens + conditioned_context.unsqueeze(-2)
        trajectory_tokens = trajectory_tokens.reshape(
            batch_size * num_agents * num_goals,
            self.future_steps,
            embed_dim,
        )

        for layer in self.layers:
            trajectory_tokens = layer(trajectory_tokens, memory)

        trajectories = self.output_projection(trajectory_tokens).reshape(
            batch_size,
            num_agents,
            num_goals,
            self.future_steps,
            2,
        )
        return trajectories


def _run_smoke_test() -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalDecoder().to(device)

    scene_embeddings = torch.randn(2, 6, 4, model.embed_dim, device=device)
    goals = torch.randn(2, 4, 6, 2, device=device)
    goal_probabilities = torch.softmax(torch.randn(2, 4, 6, device=device), dim=-1)

    trajectories = model(scene_embeddings, goals, goal_probabilities)
    expected_shape = (2, 4, 6, model.future_steps, 2)
    has_nans = torch.isnan(trajectories).any().item()

    assert trajectories.shape == expected_shape, (
        f"Expected trajectory shape {expected_shape}, got {tuple(trajectories.shape)}."
    )
    assert not has_nans, "Decoder output contains NaNs."

    print(f"Trajectory shape: {tuple(trajectories.shape)}")
    print(f"Mean: {trajectories.mean().item():.6f}")
    print(f"Std: {trajectories.std().item():.6f}")
    print(f"Has NaNs: {has_nans}")


if __name__ == "__main__":
    _run_smoke_test()
