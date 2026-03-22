"""Evaluate trained model on nuScenes mini — reports ADE and FDE."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from dataset import NuScenesDataset
from modules.decoder.goal_prediction import GoalPredictionNetwork
from modules.decoder.multimodal_decoder import MultiModalDecoder
from modules.input_embedding import InputEmbedding
from modules.scene.scene_context_encoder import SceneContextEncoder
from modules.social.social_transformer import SocialTransformer
from modules.temporal_transformer import TemporalTransformer
from utils.losses import best_of_k_loss

CHECKPOINT_PATH = "checkpoints/best_1.pt"
DATAROOT        = "data/raw/nuscenes"
VERSION         = "v1.0-mini"
BATCH_SIZE      = 2
EMBED           = 640    # must match train.py
NUM_GOALS       = 12     # must match train.py


class TrajectoryPredictor(nn.Module):
    """Must match TrajectoryPredictor in train.py exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = InputEmbedding(embedding_dim=EMBED, continuous_hidden_dim=1280)
        self.temporal  = TemporalTransformer(num_layers=8,  embed_dim=EMBED, num_heads=8, ff_dim=1280)
        self.social    = SocialTransformer(num_layers=4,    embed_dim=EMBED, num_heads=8, ff_dim=1280)
        self.scene     = SceneContextEncoder(num_layers=3,  embed_dim=EMBED, num_heads=8, ff_dim=1280, map_dim=256)
        self.goal      = GoalPredictionNetwork(embed_dim=EMBED, hidden_dim=1280, bottleneck_dim=640, num_goals=NUM_GOALS)
        self.decoder   = MultiModalDecoder(num_layers=4,    embed_dim=EMBED, num_heads=8, ff_dim=1280, future_steps=12)

    def forward(self, x: Tensor, positions: Tensor, map_features: Tensor):
        emb       = self.embedding(x)
        temp      = self.temporal(emb)
        soc       = self.social(temp, positions)
        scene_out = self.scene(soc, map_features, agent_positions=positions, map_positions=None)
        goals, probs = self.goal(scene_out)
        traj      = self.decoder(scene_out, goals, probs)
        return traj, goals, probs


def load_model(checkpoint_path: str, device: torch.device) -> TrajectoryPredictor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TrajectoryPredictor().to(device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint (epoch={checkpoint.get('epoch', '?')})")
    else:
        raise KeyError("Unsupported checkpoint format — expected 'model' key.")
    model.eval()
    return model


@torch.inference_mode()
def evaluate(checkpoint_path: str = CHECKPOINT_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model_params={total_params:,} ({total_params/1e6:.1f}M)  embed={EMBED}  goals={NUM_GOALS}")

    dataset = NuScenesDataset(dataroot=DATAROOT, version=VERSION)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(dataset)} samples...")

    total_ade = total_fde = total_loss = 0.0
    steps = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp   = device.type == "cuda"

    for batch in loader:
        x            = batch["x"].to(device)
        positions    = batch["positions"].to(device)
        gt           = batch["future"].to(device)
        map_features = batch["map"].to(device)

        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                traj, goals, probs = model(x, positions, map_features)
                loss, ade, fde = best_of_k_loss(traj, gt, return_metrics=True)
        except torch.OutOfMemoryError:
            print("OOM on eval batch, skipping")
            torch.cuda.empty_cache()
            continue

        if not torch.isfinite(loss):
            continue

        total_loss += float(loss.item())
        total_ade  += float(ade.item())
        total_fde  += float(fde.item())
        steps      += 1

    if steps == 0:
        print("No valid batches found!")
        return

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Samples evaluated : {steps * BATCH_SIZE}")
    print(f"Loss              : {total_loss/steps:.4f}")
    print(f"minADE            : {total_ade/steps:.4f}  m  (lower is better)")
    print(f"minFDE            : {total_fde/steps:.4f}  m  (lower is better)")
    print("="*40)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Dataset    : {VERSION} ({DATAROOT})")


if __name__ == "__main__":
    evaluate()
