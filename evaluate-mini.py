from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import NuScenesDataset
from modules.decoder.goal_prediction import GoalPredictionNetwork
from modules.decoder.multimodal_decoder import MultiModalDecoder
from modules.input_embedding import InputEmbedding
from modules.scene.scene_context_encoder import SceneContextEncoder
from modules.social.social_transformer import SocialTransformer
from modules.temporal_transformer import TemporalTransformer
from utils.losses import best_of_k_loss

CHECKPOINT_PATH = "models/model_fp32.pt"
DATAROOT        = "data/raw/nuscenes"
VERSION         = "v1.0-mini"
BATCH_SIZE      = 4


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = torch.nn.ModuleDict({
        "embedding": InputEmbedding(),
        "temporal":  TemporalTransformer(),
        "social":    SocialTransformer(),
        "scene":     SceneContextEncoder(map_dim=256),
        "goal":      GoalPredictionNetwork(),
        "decoder":   MultiModalDecoder(future_steps=12),
    }).to(device)

    if "model" in checkpoint:
        from modules.decoder.goal_prediction import GoalPredictionNetwork as G
        from modules.decoder.multimodal_decoder import MultiModalDecoder as D
        from modules.input_embedding import InputEmbedding as E
        from modules.scene.scene_context_encoder import SceneContextEncoder as SC
        from modules.social.social_transformer import SocialTransformer as ST
        from modules.temporal_transformer import TemporalTransformer as TT

        class TrajectoryPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = E()
                self.temporal  = TT()
                self.social    = ST()
                self.scene     = SC(map_dim=256)
                self.goal      = G()
                self.decoder   = D(future_steps=12)

            def forward(self, x, positions, map_features):
                emb       = self.embedding(x)
                temp      = self.temporal(emb)
                soc       = self.social(temp, positions)
                scene_out = self.scene(soc, map_features, agent_positions=positions, map_positions=None)
                goals, probs = self.goal(scene_out)
                traj      = self.decoder(scene_out, goals, probs)
                return traj, goals, probs

        full_model = TrajectoryPredictor().to(device)
        full_model.load_state_dict(checkpoint["model"])
        full_model.eval()
        return full_model

    raise KeyError("Unsupported checkpoint format.")


@torch.inference_mode()
def evaluate(checkpoint_path: str = CHECKPOINT_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, device)

    dataset = NuScenesDataset(dataroot=DATAROOT, version=VERSION)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(dataset)} samples...")

    total_ade   = 0.0
    total_fde   = 0.0
    total_loss  = 0.0
    steps       = 0

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp   = device.type == "cuda"

    for batch in loader:
        x            = batch["x"].to(device)
        positions    = batch["positions"].to(device)
        gt           = batch["future"].to(device)
        map_features = batch["map"].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            traj, goals, probs = model(x, positions, map_features)
            loss, ade, fde = best_of_k_loss(traj, gt, return_metrics=True)

        if not torch.isfinite(loss):
            continue

        total_loss += float(loss.item())
        total_ade  += float(ade.item())
        total_fde  += float(fde.item())
        steps      += 1

    if steps == 0:
        print("No valid batches found!")
        return

    avg_loss = total_loss / steps
    avg_ade  = total_ade  / steps
    avg_fde  = total_fde  / steps

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Samples evaluated : {steps * BATCH_SIZE}")
    print(f"Loss              : {avg_loss:.4f}")
    print(f"ADE               : {avg_ade:.4f}  ← Mean displacement error")
    print(f"FDE               : {avg_fde:.4f}  ← Final displacement error")
    print("="*40)
    print(f"\nCheckpoint : {checkpoint_path}")
    print(f"Dataset    : {VERSION} ({DATAROOT})")


if __name__ == "__main__":
    evaluate()
