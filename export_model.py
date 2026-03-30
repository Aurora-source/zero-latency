import torch
import argparse
from pathlib import Path

from modules.decoder.goal_prediction import GoalPredictionNetwork
from modules.decoder.multimodal_decoder import MultiModalDecoder
from modules.input_embedding import InputEmbedding
from modules.scene.scene_context_encoder import SceneContextEncoder
from modules.social.social_transformer import SocialTransformer
from modules.temporal_transformer import TemporalTransformer


# ------------------------------------------------
# Model (identical to training)
# ------------------------------------------------

class TrajectoryPredictor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = InputEmbedding()
        self.temporal  = TemporalTransformer()
        self.social    = SocialTransformer()
        self.scene     = SceneContextEncoder(map_dim=256)
        self.goal      = GoalPredictionNetwork()
        self.decoder   = MultiModalDecoder(future_steps=12)

    def forward(self, x, positions, map_features):
        emb       = self.embedding(x)
        temp      = self.temporal(emb)
        soc       = self.social(temp, positions)
        scene_ctx = self.scene(soc, map_features,
                               agent_positions=positions, map_positions=None)
        goals, probs = self.goal(scene_ctx)
        traj         = self.decoder(scene_ctx, goals, probs)
        return traj, goals, probs


# ------------------------------------------------
# Dummy inputs
# ------------------------------------------------

def make_dummy_inputs():
    return (
        torch.zeros(1, 6, 10, 8),
        torch.zeros(1, 6, 10, 2),
        torch.zeros(1, 1, 256),
    )


# ------------------------------------------------
# Load model
# ------------------------------------------------

def load_model(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model = TrajectoryPredictor()
    model.load_state_dict(ckpt["model"])
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
    return model


# ------------------------------------------------
# Export FP32
# ------------------------------------------------

def export_fp32(model, output_dir):
    path = output_dir / "model_fp32.pt"
    torch.save({"model": model.state_dict()}, path)
    print(f"Saved FP32  → {path}")
    return path


# ------------------------------------------------
# Export FP16
# ------------------------------------------------

def export_fp16(model, output_dir):
    model_fp16 = model.half()
    path       = output_dir / "model_fp16.pt"
    torch.save({"model": model_fp16.state_dict()}, path)
    print(f"Saved FP16  → {path}")
    return path


# ------------------------------------------------
# Verify
# ------------------------------------------------

def verify(model):
    print("\nVerifying model...")
    dummy = make_dummy_inputs()
    with torch.inference_mode():
        traj, goals, probs = model(*dummy)
    print(f"  traj  : {tuple(traj.shape)}")
    print(f"  goals : {tuple(goals.shape)}")
    print(f"  probs : {tuple(probs.shape)}")
    print("Verification passed.")


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Export TrajectoryPredictor checkpoint to FP32, FP16, and ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_1.pt",
        help="Path to raw training checkpoint (default: checkpoints/best_1.pt)",
    )
    parser.add_argument(
        "--output",
        default="models",          # ← writes to models/ (not exports/)
        help="Output folder for exported weights (default: models)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    model = load_model(args.checkpoint)

    export_fp32(model, output_dir)
    export_fp16(model, output_dir)

    verify(model)

    print("\nExport complete.")
    print(f"  models/ contains: model_fp32.pt  model_fp16.pt")
