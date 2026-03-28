"""Loss functions for multi-modal trajectory prediction."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import torch.nn as nn



__all__ = [
    "compute_ade",
    "compute_fde",
    "best_of_k_loss",
    "goal_classification_loss",
]


def compute_ade(pred: Tensor, gt: Tensor) -> Tensor:
    """Compute average displacement error over the trajectory horizon.

    Args:
        pred: Predicted trajectories with shape ``(..., future_steps, 2)``.
        gt: Ground-truth trajectories broadcastable to ``pred``.

    Returns:
        Tensor of shape ``(...)`` containing ADE values.
    """

    if pred.shape[-1] != 2 or gt.shape[-1] != 2:
        raise ValueError("pred and gt must end with coordinate dimension 2.")
    if pred.shape[-2] != gt.shape[-2]:
        raise ValueError("pred and gt must have the same future_steps dimension.")

    distances = torch.linalg.vector_norm(pred - gt, dim=-1)
    return distances.mean(dim=-1)


def compute_fde(pred: Tensor, gt: Tensor) -> Tensor:
    """Compute final displacement error at the final timestep."""

    if pred.shape[-1] != 2 or gt.shape[-1] != 2:
        raise ValueError("pred and gt must end with coordinate dimension 2.")
    if pred.shape[-2] != gt.shape[-2]:
        raise ValueError("pred and gt must have the same future_steps dimension.")

    final_displacement = pred[..., -1, :] - gt[..., -1, :]
    return torch.linalg.vector_norm(final_displacement, dim=-1)


class AutoTunedBestOfKLoss(nn.Module):
    """Compute best-of-K trajectory loss with auto-tuned ADE/FDE and smoothness weights."""
    
    def __init__(self, ade_weight: float = 1.0, fde_weight: float = 1.0):
        super().__init__()
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        
        # Initialize learnable log-variances (s) for uncertainty weighting.
        # Initializing at 0.0 gives a starting effective weight of 1.0.
        self.s_tracking = nn.Parameter(torch.zeros(1))
        self.s_smoothness = nn.Parameter(torch.zeros(1))

    def forward(
        self, 
        pred_traj: Tensor, 
        gt_traj: Tensor, 
        return_metrics: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        
        # 1. Shape validations (Same as your original code)
        if pred_traj.ndim != 5:
            raise ValueError(f"pred_traj must have shape (batch, agents, modes, future_steps, 2)")
        if gt_traj.ndim != 4:
            raise ValueError(f"gt_traj must have shape (batch, agents, future_steps, 2)")

        # 2. Compute ADE/FDE metrics for the Best-of-K selection
        expanded_gt = gt_traj.unsqueeze(2)
        ade_per_mode = compute_ade(pred_traj, expanded_gt)
        best_mode_indices = ade_per_mode.argmin(dim=-1)

        best_ade = ade_per_mode.gather(dim=-1, index=best_mode_indices.unsqueeze(-1)).squeeze(-1)
        fde_per_mode = compute_fde(pred_traj, expanded_gt)
        best_fde = fde_per_mode.gather(dim=-1, index=best_mode_indices.unsqueeze(-1)).squeeze(-1)

        # 3. Calculate the RAW losses
        raw_tracking_loss = (self.ade_weight * best_ade.mean()) + (self.fde_weight * best_fde.mean())
        raw_smoothness_loss = compute_smoothness_loss(pred_traj)

        # 4. Apply Auto-Tuning (Homoscedastic Uncertainty Weighting)
        # L = exp(-s) * raw_loss + s
        weighted_tracking = torch.exp(-self.s_tracking) * raw_tracking_loss + self.s_tracking
        weighted_smoothness = torch.exp(-self.s_smoothness) * raw_smoothness_loss + self.s_smoothness

        # Combine for final loss
        total_loss = weighted_tracking + weighted_smoothness

        if return_metrics:
            return total_loss, best_ade.mean(), best_fde.mean()
        return total_loss


def goal_classification_loss(goal_probs: Tensor, goals: Tensor, gt_traj: Tensor) -> Tensor:
    """Encourage the highest-probability goal to match the GT final position.

    Args:
        goal_probs: Goal probabilities of shape ``(batch, agents, modes)``.
        goals: Goal coordinates of shape ``(batch, agents, modes, 2)``.
        gt_traj: Ground-truth trajectories of shape ``(batch, agents, future_steps, 2)``.

    Returns:
        Scalar goal classification loss.
    """

    if goal_probs.ndim != 3:
        raise ValueError(
            f"goal_probs must have shape (batch, agents, modes), got {tuple(goal_probs.shape)}."
        )
    if goals.ndim != 4 or goals.shape[-1] != 2:
        raise ValueError(
            f"goals must have shape (batch, agents, modes, 2), got {tuple(goals.shape)}."
        )
    if gt_traj.ndim != 4 or gt_traj.shape[-1] != 2:
        raise ValueError(
            f"gt_traj must have shape (batch, agents, future_steps, 2), got {tuple(gt_traj.shape)}."
        )
    if goal_probs.shape != goals.shape[:3] or goal_probs.shape[:2] != gt_traj.shape[:2]:
        raise ValueError("goal_probs, goals, and gt_traj must align on batch and agents.")

    gt_final_positions = gt_traj[..., -1, :].unsqueeze(2)
    goal_distances = torch.linalg.vector_norm(goals - gt_final_positions, dim=-1)
    target_indices = goal_distances.argmin(dim=-1)

    log_goal_probs = goal_probs.clamp_min(1e-8).log()
    return F.nll_loss(
        log_goal_probs.reshape(-1, goal_probs.size(-1)),
        target_indices.reshape(-1),
        reduction="mean",
    )



def compute_smoothness_loss(pred_traj: Tensor) -> Tensor:
    """Compute the smoothness regularization loss (second-order difference).
    
    This penalizes high acceleration/jerk in the predicted trajectories to 
    ensure physically feasible paths.

    Args:
        pred_traj: Predicted trajectories of shape ``(batch, agents, modes, future_steps, 2)``.

    Returns:
        Scalar tensor representing the mean smoothness penalty.
    """
    if pred_traj.ndim != 5:
        raise ValueError(
            f"pred_traj must have shape (batch, agents, modes, future_steps, 2), "
            f"but received {tuple(pred_traj.shape)}."
        )

    # Calculate velocity: delta_x = x_{t} - x_{t-1}
    # Shape becomes (batch, agents, modes, future_steps - 1, 2)
    velocity = pred_traj[..., 1:, :] - pred_traj[..., :-1, :]

    # Calculate acceleration (curvature): delta_v = v_{t} - v_{t-1}
    # Shape becomes (batch, agents, modes, future_steps - 2, 2)
    acceleration = velocity[..., 1:, :] - velocity[..., :-1, :]

    # We use the L2 norm squared to penalize large jumps more heavily
    # Mean over all dimensions to return a scalar loss
    return torch.mean(acceleration**2)