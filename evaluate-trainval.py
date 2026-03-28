from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
VERSION = "v1.0-trainval"
BATCH_SIZE = 4
OUTPUT_DIR = "evaluation_results"


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e = InputEmbedding()
        self.t = TemporalTransformer()
        self.s = SocialTransformer()
        self.c = SceneContextEncoder(map_dim=256)
        self.g = GoalPredictionNetwork()
        self.d = MultiModalDecoder(future_steps=12)

    def forward(self, x, pos, m):
        x = self.e(x)
        x = self.t(x)
        x = self.s(x, pos)
        x = self.c(x, m, agent_positions=pos, map_positions=None)
        g, p = self.g(x)
        tr = self.d(x, g, p)
        return tr, g, p


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    m = Model().to(device)

    if "model" in ckpt:
        m.load_state_dict(ckpt["model"])
    else:
        raise KeyError("bad checkpoint")

    m.eval()
    return m


def compute_metrics(pred, gt):
    diff = (pred - gt).norm(dim=-1)
    per_step = diff.mean(dim=(0, 1)).cpu().numpy()
    ade = diff.mean(dim=(1, 2)).cpu().numpy()
    fde = diff[:, :, -1].mean(dim=1).cpu().numpy()
    return per_step, ade, fde


def best_mode(traj, gt):
    if traj.shape[1] == gt.shape[1]:
        traj = traj.permute(0, 2, 1, 3, 4)

    B, K, N, T, _ = traj.shape
    gt_exp = gt.unsqueeze(1).expand(-1, K, -1, -1, -1)

    err = (traj - gt_exp).norm(dim=-1).mean(dim=(2, 3))
    idx = err.argmin(dim=1)

    return traj[torch.arange(B, device=traj.device), idx]


def plot(metrics, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["ade_t"])
    plt.title("ADE over time")
    plt.xlabel("t")
    plt.ylabel("error")

    p = out_dir / "plot.png"
    plt.savefig(p)
    plt.close()

    print("saved:", p)


@torch.inference_mode()
def evaluate(path=CHECKPOINT_PATH, dataroot=DATAROOT, version=VERSION, bs=BATCH_SIZE, out=OUTPUT_DIR):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = load_model(path, device)

    ds = NuScenesDataset(dataroot=dataroot, version=version)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    all_ade = []
    all_fde = []
    all_step = []
    tot_loss = 0.0
    steps = 0

    for b in dl:
        x = b["x"].to(device)
        pos = b["positions"].to(device)
        gt = b["future"].to(device)
        m = b["map"].to(device)

        traj, g, p = model(x, pos, m)
        loss, _, _ = best_of_k_loss(traj, gt, return_metrics=True)

        if not torch.isfinite(loss):
            continue

        tr = best_mode(traj, gt) if traj.dim() == 5 else traj
        step, ade, fde = compute_metrics(tr, gt)

        all_step.append(step)
        all_ade.extend(ade.tolist())
        all_fde.extend(fde.tolist())

        tot_loss += float(loss.item())
        steps += 1

    if steps == 0:
        print("no data")
        return

    ade_arr = np.array(all_ade)
    fde_arr = np.array(all_fde)

    metrics = {
        "ade": float(ade_arr.mean()),
        "fde": float(fde_arr.mean()),
        "ade_t": np.stack(all_step).mean(axis=0),
    }

    print("\nRESULTS")
    print("loss:", tot_loss / steps)
    print("ADE :", metrics["ade"])
    print("FDE :", metrics["fde"])

    plot(metrics, Path(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--dataroot", default=DATAROOT)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)

    a = parser.parse_args()

    evaluate(
        path=a.checkpoint,
        dataroot=a.dataroot,
        version=a.version,
        bs=a.batch_size,
        out=a.output_dir,
    )