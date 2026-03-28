from __future__ import annotations

import torch
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
DATAROOT = "data/raw/nuscenes"
VERSION = "v1.0-mini"
BATCH_SIZE = 4


def load_model(path: str, device: torch.device):

    ckpt = torch.load(path, map_location=device)

    if "model" not in ckpt:
        raise KeyError("invalid checkpoint")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = InputEmbedding()
            self.temporal = TemporalTransformer()
            self.social = SocialTransformer()
            self.scene = SceneContextEncoder(map_dim=256)
            self.goal = GoalPredictionNetwork()
            self.decoder = MultiModalDecoder(future_steps=12)

        def forward(self, x, pos, m):
            e = self.embedding(x)
            t = self.temporal(e)
            s = self.social(t, pos)
            c = self.scene(s, m, agent_positions=pos, map_positions=None)
            g, p = self.goal(c)
            tr = self.decoder(c, g, p)
            return tr, g, p

    model = Model().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model


@torch.inference_mode()
def evaluate(path: str = CHECKPOINT_PATH):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    print(f"checkpoint: {path}")

    model = load_model(path, device)

    ds = NuScenesDataset(dataroot=DATAROOT, version=VERSION)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"samples: {len(ds)}")

    tot_loss = 0.0
    tot_ade = 0.0
    tot_fde = 0.0
    steps = 0

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda"

    for batch in dl:

        x = batch["x"].to(device)
        pos = batch["positions"].to(device)
        gt = batch["future"].to(device)
        m = batch["map"].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            traj, g, p = model(x, pos, m)
            loss, ade, fde = best_of_k_loss(traj, gt, return_metrics=True)

        if not torch.isfinite(loss):
            continue

        tot_loss += float(loss.item())
        tot_ade += float(ade.item())
        tot_fde += float(fde.item())
        steps += 1

    if steps == 0:
        print("no valid batches")
        return

    print("\n==== RESULTS ====")
    print(f"loss: {tot_loss / steps:.4f}")
    print(f"ADE : {tot_ade / steps:.4f}")
    print(f"FDE : {tot_fde / steps:.4f}")


if __name__ == "__main__":
    evaluate()