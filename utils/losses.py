#changed
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "compute_ade",
    "compute_fde",
    "best_of_k_loss",
    "goal_classification_loss",
]


def compute_ade(pred: Tensor, gt: Tensor) -> Tensor:
    if pred.shape[-1] != 2 or gt.shape[-1] != 2:
        raise ValueError("last dim must be 2")
    if pred.shape[-2] != gt.shape[-2]:
        raise ValueError("future steps mismatch")

    d = torch.linalg.vector_norm(pred - gt, dim=-1)
    return d.mean(dim=-1)


def compute_fde(pred: Tensor, gt: Tensor) -> Tensor:
    if pred.shape[-1] != 2 or gt.shape[-1] != 2:
        raise ValueError("last dim must be 2")
    if pred.shape[-2] != gt.shape[-2]:
        raise ValueError("future steps mismatch")

    d = pred[..., -1, :] - gt[..., -1, :]
    return torch.linalg.vector_norm(d, dim=-1)


def best_of_k_loss(
    pred_traj: Tensor,
    gt_traj: Tensor,
    ade_weight: float = 1.0,
    fde_weight: float = 1.0,
    return_metrics: bool = False,
) -> Tensor | Tuple[Tensor, Tensor, Tensor]:

    if pred_traj.ndim != 5:
        raise ValueError(f"expected 5D, got {tuple(pred_traj.shape)}")
    if gt_traj.ndim != 4:
        raise ValueError(f"expected 4D, got {tuple(gt_traj.shape)}")
    if pred_traj.shape[:2] != gt_traj.shape[:2] or pred_traj.shape[-2:] != gt_traj.shape[-2:]:
        raise ValueError("shape mismatch")

    g_exp = gt_traj.unsqueeze(2)

    ade_modes = compute_ade(pred_traj, g_exp)
    idx = ade_modes.argmin(dim=-1)

    best_ade = ade_modes.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    fde_modes = compute_fde(pred_traj, g_exp)
    best_fde = fde_modes.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    loss = ade_weight * best_ade.mean() + fde_weight * best_fde.mean()

    if return_metrics:
        return loss, best_ade.mean(), best_fde.mean()
    return loss


def goal_classification_loss(goal_probs: Tensor, goals: Tensor, gt_traj: Tensor) -> Tensor:

    if goal_probs.ndim != 3:
        raise ValueError(f"prob shape wrong: {tuple(goal_probs.shape)}")
    if goals.ndim != 4 or goals.shape[-1] != 2:
        raise ValueError(f"goals shape wrong: {tuple(goals.shape)}")
    if gt_traj.ndim != 4 or gt_traj.shape[-1] != 2:
        raise ValueError(f"gt shape wrong: {tuple(gt_traj.shape)}")
    if goal_probs.shape != goals.shape[:3] or goal_probs.shape[:2] != gt_traj.shape[:2]:
        raise ValueError("shape mismatch")

    g_final = gt_traj[..., -1, :].unsqueeze(2)
    d = torch.linalg.vector_norm(goals - g_final, dim=-1)
    idx = d.argmin(dim=-1)

    log_p = goal_probs.clamp_min(1e-8).log()

    return F.nll_loss(
        log_p.reshape(-1, goal_probs.size(-1)),
        idx.reshape(-1),
        reduction="mean",
    )