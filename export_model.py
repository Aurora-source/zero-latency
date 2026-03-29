import torch
import argparse
from pathlib import Path

# Import model modules (same as train.py)

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

        emb = self.embedding(x)

        temp = self.temporal(emb)

        soc = self.social(temp, positions)

        scene_ctx = self.scene(
            soc,
            map_features,
            agent_positions=positions,
            map_positions=None
        )

        goals, probs = self.goal(scene_ctx)

        traj = self.decoder(
            scene_ctx,
            goals,
            probs
        )

        return traj, goals, probs


# ------------------------------------------------
# Dummy Inputs
# ------------------------------------------------

def make_dummy_inputs():

    return (
        torch.zeros(1, 6, 10, 8),
        torch.zeros(1, 6, 10, 2),
        torch.zeros(1, 1, 256),
    )


# ------------------------------------------------
# Load Model
# ------------------------------------------------

def load_model(checkpoint_path):

    print(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

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

    torch.save(
        {"model": model.state_dict()},
        path
    )

    print(f"Saved FP32 → {path}")

    return path


# ------------------------------------------------
# Export FP16
# ------------------------------------------------

def export_fp16(model, output_dir):

    model_fp16 = model.half()

    path = output_dir / "model_fp16.pt"

    torch.save(
        {"model": model_fp16.state_dict()},
        path
    )

    print(f"Saved FP16 → {path}")

    return path


# ------------------------------------------------
# Export ONNX
# ------------------------------------------------

def export_onnx(model, output_dir):

    dummy = make_dummy_inputs()

    path = output_dir / "model.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(path),

        input_names=[
            "x",
            "positions",
            "map_features"
        ],

        output_names=[
            "trajectories",
            "goals",
            "probabilities"
        ],

        dynamic_axes={
            "x": {0: "batch"},
            "positions": {0: "batch"},
            "map_features": {0: "batch"},
            "trajectories": {0: "batch"},
            "goals": {0: "batch"},
            "probabilities": {0: "batch"},
        },

        opset_version=17
    )

    print(f"Saved ONNX → {path}")

    return path


# ------------------------------------------------
# Verify
# ------------------------------------------------

def verify(model):

    print("\nVerifying model...")

    dummy = make_dummy_inputs()

    with torch.inference_mode():

        traj, goals, probs = model(*dummy)

    print("Output Shapes:")
    print("traj:", traj.shape)
    print("goals:", goals.shape)
    print("probs:", probs.shape)

    print("Verification Passed")


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_1.pt"
    )

    parser.add_argument(
        "--output",
        default="exports"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    model = load_model(args.checkpoint)

    export_fp32(model, output_dir)

    export_fp16(model, output_dir)

    export_onnx(model, output_dir)

    verify(model)

    print("\nExport Complete.")
