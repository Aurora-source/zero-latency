from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

__all__ = ["GoalPredictionNetwork"]


class GoalPredictionNetwork(nn.Module):

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
            raise ValueError("embed_dim must be positive")
        if num_goals <= 0:
            raise ValueError("num_goals must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

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

        if inputs.ndim != 4:
            raise ValueError(f"expected 4D input, got {tuple(inputs.shape)}")

        b, t, a, e = inputs.shape

        if t <= 0:
            raise ValueError("no timesteps")
        if e != self.embed_dim:
            raise ValueError(f"embed dim mismatch: {e}")

        dtype = self.backbone[0].weight.dtype
        x = inputs[:, -1, :, :].to(dtype=dtype)
        h = self.backbone(x)

        goal_pos = self.goal_position_head(h).reshape(
            b,
            a,
            self.num_goals,
            2,
        )

        logits = self.goal_probability_head(h)
        goal_prob = torch.softmax(logits, dim=-1)

        return goal_pos, goal_prob


def _run_smoke_test() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoalPredictionNetwork().to(device)

    x = torch.randn(2, 6, 4, model.embed_dim, device=device)
    goals, probs = model(x)

    exp_goal = (2, 4, model.num_goals, 2)
    exp_prob = (2, 4, model.num_goals)

    s = probs.sum(dim=-1)

    assert goals.shape == exp_goal, f"goal shape wrong: {tuple(goals.shape)}"
    assert probs.shape == exp_prob, f"prob shape wrong: {tuple(probs.shape)}"
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6)

    print(goals.shape)
    print(probs.shape)
    print(s)


if __name__ == "__main__":
    _run_smoke_test()