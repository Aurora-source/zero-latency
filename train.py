"""Minimal training entrypoint for the nuScenes mini trajectory prediction pipeline."""

from __future__ import annotations

from itertools import chain

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from dataset import NuScenesDataset
from modules.decoder.goal_prediction import GoalPredictionNetwork
from modules.decoder.multimodal_decoder import MultiModalDecoder
from modules.input_embedding import InputEmbedding
from modules.scene.scene_context_encoder import SceneContextEncoder
from modules.social.social_transformer import SocialTransformer
from modules.temporal_transformer import TemporalTransformer
from utils.losses import best_of_k_loss


def train() -> None:
    """Run a minimal end-to-end training loop on nuScenes mini."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NuScenesDataset(
        dataroot="data/raw/nuscenes",
        version="v1.0-mini",
    )
    dataset = Subset(dataset, range(min(200, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    embedding = InputEmbedding().to(device)
    temporal = TemporalTransformer().to(device)
    social = SocialTransformer().to(device)
    scene = SceneContextEncoder(map_dim=256).to(device)
    goal = GoalPredictionNetwork().to(device)
    decoder = MultiModalDecoder(future_steps=12).to(device)

    optimizer = torch.optim.Adam(
        chain(
            embedding.parameters(),
            temporal.parameters(),
            social.parameters(),
            scene.parameters(),
            goal.parameters(),
            decoder.parameters(),
        ),
        lr=1e-4,
    )

    print(f"device={device}")
    print(f"dataset_size={len(dataset)}")

    for epoch in range(1):
        embedding.train()
        temporal.train()
        social.train()
        scene.train()
        goal.train()
        decoder.train()

        for step, batch in enumerate(loader):
            x = batch["x"].to(device)
            positions = batch["positions"].to(device)
            gt = batch["future"].to(device)
            map_features = batch["map"].to(device)

            optimizer.zero_grad(set_to_none=True)

            emb = embedding(x)
            temp = temporal(emb)
            soc = social(temp, positions)
            scene_out = scene(
                soc,
                map_features,
                agent_positions=positions,
                map_positions=None,
            )
            goals, probs = goal(scene_out)
            traj = decoder(scene_out, goals, probs)

            loss = best_of_k_loss(traj, gt)

            if torch.isnan(loss):
                print("NaN detected, skipping batch")
                continue

            loss.backward()
            clip_grad_norm_(
                chain(
                    embedding.parameters(),
                    temporal.parameters(),
                    social.parameters(),
                    scene.parameters(),
                    goal.parameters(),
                    decoder.parameters(),
                ),
                max_norm=1.0,
            )
            optimizer.step()

            print(f"epoch={epoch} step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    train()
