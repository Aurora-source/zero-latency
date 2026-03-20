"""Minimal end-to-end training script for the trajectory prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from modules.decoder.goal_prediction import GoalPredictionNetwork
from modules.decoder.multimodal_decoder import MultiModalDecoder
from modules.input_embedding import InputEmbedding
from modules.scene.scene_context_encoder import SceneContextEncoder
from modules.social.social_transformer import SocialTransformer
from modules.temporal_transformer import TemporalTransformer
from utils.losses import best_of_k_loss, goal_classification_loss


@dataclass(frozen=True)
class TrainConfig:
    """Training and synthetic data configuration."""

    dataset_size: int = 32
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    goal_classification_weight: float = 0.1
    past_steps: int = 6
    future_steps: int = 12
    num_agents: int = 4
    num_map_elements: int = 128
    map_dim: int = 256
    num_goals: int = 6
    input_feature_dim: int = 9
    seed: int = 7


class DummyTrajectoryDataset(Dataset[Dict[str, Tensor]]):
    """Small deterministic synthetic dataset for pipeline and overfitting checks."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(config.seed)

        dataset_size = config.dataset_size
        num_agents = config.num_agents
        past_steps = config.past_steps
        future_steps = config.future_steps
        dt = 0.5

        initial_positions = torch.randn(dataset_size, num_agents, 2, generator=generator) * 10.0
        initial_velocities = torch.randn(dataset_size, num_agents, 2, generator=generator)
        accelerations = torch.randn(dataset_size, num_agents, 2, generator=generator) * 0.1
        agent_types = torch.randint(
            low=0,
            high=3,
            size=(dataset_size, num_agents, 1),
            generator=generator,
        )

        past_time = torch.arange(past_steps, dtype=torch.float32).view(1, past_steps, 1, 1) * dt
        future_time = torch.arange(1, future_steps + 1, dtype=torch.float32).view(1, 1, future_steps, 1) * dt

        past_positions = (
            initial_positions.unsqueeze(1)
            + initial_velocities.unsqueeze(1) * past_time
            + 0.5 * accelerations.unsqueeze(1) * past_time.square()
        )
        past_velocities = initial_velocities.unsqueeze(1) + accelerations.unsqueeze(1) * past_time
        past_accelerations = accelerations.unsqueeze(1).expand(dataset_size, past_steps, num_agents, 2)
        headings = torch.atan2(past_velocities[..., 1], past_velocities[..., 0]).unsqueeze(-1)
        speed_magnitude = torch.linalg.vector_norm(past_velocities, dim=-1, keepdim=True)
        type_feature = agent_types.unsqueeze(1).expand(dataset_size, past_steps, num_agents, 1).to(torch.float32)

        self.agent_features = torch.cat(
            (
                past_positions,
                past_velocities,
                past_accelerations,
                headings,
                speed_magnitude,
                type_feature,
            ),
            dim=-1,
        )
        self.positions = past_positions

        last_positions = past_positions[:, -1, :, :]
        last_velocities = past_velocities[:, -1, :, :]
        future_trajectories = (
            last_positions.unsqueeze(2)
            + last_velocities.unsqueeze(2) * future_time
            + 0.5 * accelerations.unsqueeze(2) * future_time.square()
        )
        noise = torch.randn(future_trajectories.shape, generator=generator) * 0.01
        self.gt_trajectories = future_trajectories + noise

        self.map_features = torch.randn(
            dataset_size,
            config.num_map_elements,
            config.map_dim,
            generator=generator,
        ) * 0.1

        if self.agent_features.size(-1) != config.input_feature_dim:
            raise ValueError(
                f"Expected input feature dimension {config.input_feature_dim}, "
                f"got {self.agent_features.size(-1)}."
            )

    def __len__(self) -> int:
        return self.agent_features.size(0)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "agent_features": self.agent_features[index],
            "positions": self.positions[index],
            "map_features": self.map_features[index],
            "gt_trajectories": self.gt_trajectories[index],
        }


class TrajectoryPredictionModel(nn.Module):
    """Convenience wrapper around the full trajectory prediction stack."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.embedding = InputEmbedding()
        self.temporal = TemporalTransformer()
        self.social = SocialTransformer(num_layers=6, num_heads=8, embed_dim=896, ff_dim=2048, dropout=0.1)
        self.scene = SceneContextEncoder(
            num_layers=4,
            num_heads=8,
            embed_dim=896,
            map_dim=config.map_dim,
            ff_dim=2048,
            dropout=0.1,
            max_distance=50.0,
        )
        self.goal = GoalPredictionNetwork(num_goals=config.num_goals)
        self.decoder = MultiModalDecoder(
            num_layers=8,
            num_heads=8,
            embed_dim=896,
            ff_dim=2048,
            dropout=0.1,
            future_steps=config.future_steps,
        )

    def forward(
        self,
        agent_features: Tensor,
        positions: Tensor,
        map_features: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run the complete forward pass from raw inputs to trajectories."""

        embeddings = self.embedding(agent_features)
        temporal_context = self.temporal(embeddings)
        social_context = self.social(temporal_context, positions)
        scene_context = self.scene(social_context, map_features)
        goals, goal_probabilities = self.goal(scene_context)
        trajectories = self.decoder(scene_context, goals, goal_probabilities)
        return trajectories, goals, goal_probabilities


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible dummy training runs."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(config: TrainConfig) -> None:
    """Train the full model on a small synthetic dataset."""

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DummyTrajectoryDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TrajectoryPredictionModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    print(f"Training on device: {device}")
    print(f"Dataset size: {len(dataset)} | Batch size: {config.batch_size} | Epochs: {config.epochs}")

    for epoch in range(config.epochs):
        model.train()
        for step, batch in enumerate(dataloader, start=1):
            agent_features = batch["agent_features"].to(device)
            positions = batch["positions"].to(device)
            map_features = batch["map_features"].to(device)
            gt_trajectories = batch["gt_trajectories"].to(device)

            trajectories, goals, goal_probabilities = model(
                agent_features,
                positions,
                map_features,
            )

            trajectory_loss, ade, fde = best_of_k_loss(
                trajectories,
                gt_trajectories,
                return_metrics=True,
            )
            goal_loss = goal_classification_loss(goal_probabilities, goals, gt_trajectories)
            loss = trajectory_loss + config.goal_classification_weight * goal_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()

            print(
                f"epoch={epoch + 1}/{config.epochs} "
                f"step={step}/{len(dataloader)} "
                f"loss={loss.item():.4f} "
                f"ade={ade.item():.4f} "
                f"fde={fde.item():.4f} "
                f"goal_loss={goal_loss.item():.4f} "
                f"grad_norm={float(grad_norm):.4f}"
            )

        scheduler.step()
        print(f"epoch={epoch + 1} lr={scheduler.get_last_lr()[0]:.6f}")


def main() -> None:
    """Script entrypoint."""

    run_training(TrainConfig())


if __name__ == "__main__":
    main()
