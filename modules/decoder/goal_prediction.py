"""Goal prediction network for multimodal future intent estimation."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

__all__ = ["GoalPredictionNetwork"]


class GoalPredictionNetwork(nn.Module):
    """Predict multiple future goal candidates and their probabilities per agent.

    The module consumes scene-aware agent embeddings of shape
    ``(batch, time, agents, embed_dim)`` and uses the final observed timestep as
    the temporal summary for multimodal goal prediction.
    """

    def __init__(
        self,
        embed_dim: int = 896,
        num_goals: int = 6,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if num_goals <= 0:
            raise ValueError("num_goals must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.embed_dim = embed_dim
        self.num_goals = num_goals
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.dropout_p = dropout

        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.goal_position_head = nn.Linear(bottleneck_dim, num_goals * 2)
        self.goal_probability_head = nn.Linear(bottleneck_dim, num_goals)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict goal positions and probabilities for each agent.

        Args:
            inputs: Tensor of shape ``(batch, time, agents, embed_dim)``.

        Returns:
            A tuple ``(goal_positions, goal_probabilities)`` with shapes
            ``(batch, agents, num_goals, 2)`` and ``(batch, agents, num_goals)``.
        """

        if inputs.ndim != 4:
            raise ValueError(
                f"Expected inputs with 4 dimensions (batch, time, agents, embed_dim), "
                f"but received shape {tuple(inputs.shape)}."
            )

        batch_size, time_steps, num_agents, embed_dim = inputs.shape
        if time_steps <= 0:
            raise ValueError("inputs must contain at least one timestep.")
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, but received {embed_dim}."
            )

        model_dtype = self.backbone[0].weight.dtype
        aggregated = inputs[:, -1, :, :].to(dtype=model_dtype)
        hidden = self.backbone(aggregated)

        goal_positions = self.goal_position_head(hidden).reshape(
            batch_size,
            num_agents,
            self.num_goals,
            2,
        )
        goal_logits = self.goal_probability_head(hidden)
        goal_probabilities = torch.softmax(goal_logits, dim=-1)
        return goal_positions, goal_probabilities


def _run_smoke_test() -> None:
    """Run a minimal shape and probability-normalization test."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoalPredictionNetwork().to(device)

    dummy_inputs = torch.randn(2, 6, 4, model.embed_dim, device=device)
    goals, probabilities = model(dummy_inputs)

    expected_goal_shape = (2, 4, model.num_goals, 2)
    expected_prob_shape = (2, 4, model.num_goals)
    probability_sums = probabilities.sum(dim=-1)

    assert goals.shape == expected_goal_shape, (
        f"Expected goal shape {expected_goal_shape}, got {tuple(goals.shape)}."
    )
    assert probabilities.shape == expected_prob_shape, (
        f"Expected probability shape {expected_prob_shape}, got {tuple(probabilities.shape)}."
    )
    assert torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-6,
    ), "Goal probabilities must sum to 1."

    print(f"Goal shape: {tuple(goals.shape)}")
    print(f"Probability shape: {tuple(probabilities.shape)}")
    print(f"Probability sums: {probability_sums}")


if __name__ == "__main__":
    _run_smoke_test()
