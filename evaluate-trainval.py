from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
DATAROOT        = "data/raw/nuscenes"        
VERSION         = "v1.0-trainval"
BATCH_SIZE      = 4
OUTPUT_DIR      = "evaluation_results"

class TrajectoryPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = InputEmbedding()
        self.temporal  = TemporalTransformer()
        self.social    = SocialTransformer()
        self.scene     = SceneContextEncoder(map_dim=256)
        self.goal      = GoalPredictionNetwork()
        self.decoder   = MultiModalDecoder(future_steps=12)

    def forward(self, x, positions, map_features):
        emb          = self.embedding(x)
        temp         = self.temporal(emb)
        soc          = self.social(temp, positions)
        scene_out    = self.scene(soc, map_features, agent_positions=positions, map_positions=None)
        goals, probs = self.goal(scene_out)
        traj         = self.decoder(scene_out, goals, probs)
        return traj, goals, probs


def load_model(checkpoint_path: str, device: torch.device) -> TrajectoryPredictor:
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TrajectoryPredictor().to(device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        legacy = ("embedding", "temporal", "social", "scene", "goal", "decoder")
        if all(name in checkpoint for name in legacy):
            for name in legacy:
                getattr(model, name).load_state_dict(checkpoint[name])
        else:
            raise KeyError("Unsupported checkpoint format.")

    model.eval()
    epoch    = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_metrics", {}).get("loss", "?")
    print(f"  Checkpoint epoch={epoch}  saved_val_loss={val_loss}")
    return model
    
def compute_ade_fde_per_timestep(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff      = (pred - gt).norm(dim=-1)
    per_step  = diff.mean(dim=(0, 1)).cpu().numpy()
    ade_per_b = diff.mean(dim=(1, 2)).cpu().numpy()
    fde_per_b = diff[:, :, -1].mean(dim=1).cpu().numpy()
    return per_step, ade_per_b, fde_per_b


def select_best_mode(
    traj: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:


    if traj.shape[1] == gt.shape[1]:
        traj = traj.permute(0, 2, 1, 3, 4)

    B, K, N, T, _ = traj.shape

    gt_exp = gt.unsqueeze(1).expand(-1, K, -1, -1, -1)

    ade_k = (
        (traj - gt_exp)
        .norm(dim=-1)
        .mean(dim=(2, 3))
    )

    best_k = ade_k.argmin(dim=1)

    best_traj = traj[
        torch.arange(B, device=traj.device),
        best_k
    ]

    return best_traj
    
def plot_results(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12), facecolor="#0d0d0d")
    fig.suptitle("Trajectory Prediction — Evaluation Results",
                 fontsize=18, color="white", fontweight="bold", y=0.98)

    gs      = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    accent  = "#00e5ff"
    accent2 = "#ff6b6b"
    accent3 = "#a8ff78"
    bg      = "#1a1a2e"

    ax1 = fig.add_subplot(gs[0, 0])
    T   = len(metrics["ade_per_timestep"])
    t   = np.arange(1, T + 1) * 0.5
    ax1.plot(t, metrics["ade_per_timestep"], color=accent, linewidth=2.5, marker="o", markersize=4)
    ax1.fill_between(t, metrics["ade_per_timestep"], alpha=0.15, color=accent)
    ax1.set_title("Displacement Error per Timestep", color="white", fontsize=11)
    ax1.set_xlabel("Time (s)", color="gray"); ax1.set_ylabel("ADE (m)", color="gray")
    ax1.tick_params(colors="gray"); ax1.set_facecolor(bg); ax1.spines[:].set_color("#333")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(metrics["ade_per_sample"], bins=40, color=accent, alpha=0.85, edgecolor="#0d0d0d")
    ax2.axvline(metrics["mean_ade"], color=accent2, linewidth=2, linestyle="--",
                label=f"Mean: {metrics['mean_ade']:.3f}m")
    ax2.axvline(np.median(metrics["ade_per_sample"]), color=accent3, linewidth=2, linestyle=":",
                label=f"Median: {np.median(metrics['ade_per_sample']):.3f}m")
    ax2.set_title("ADE Distribution", color="white", fontsize=11)
    ax2.set_xlabel("ADE (m)", color="gray"); ax2.set_ylabel("Count", color="gray")
    ax2.tick_params(colors="gray"); ax2.set_facecolor(bg); ax2.spines[:].set_color("#333")
    ax2.legend(fontsize=9, labelcolor="white", facecolor=bg)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(metrics["fde_per_sample"], bins=40, color=accent2, alpha=0.85, edgecolor="#0d0d0d")
    ax3.axvline(metrics["mean_fde"], color=accent, linewidth=2, linestyle="--",
                label=f"Mean: {metrics['mean_fde']:.3f}m")
    ax3.axvline(np.median(metrics["fde_per_sample"]), color=accent3, linewidth=2, linestyle=":",
                label=f"Median: {np.median(metrics['fde_per_sample']):.3f}m")
    ax3.set_title("FDE Distribution", color="white", fontsize=11)
    ax3.set_xlabel("FDE (m)", color="gray"); ax3.set_ylabel("Count", color="gray")
    ax3.tick_params(colors="gray"); ax3.set_facecolor(bg); ax3.spines[:].set_color("#333")
    ax3.legend(fontsize=9, labelcolor="white", facecolor=bg)

    ax4 = fig.add_subplot(gs[1, 0])
    sc = ax4.scatter(metrics["ade_per_sample"], metrics["fde_per_sample"],
                     alpha=0.3, s=8, c=metrics["fde_per_sample"],
                     cmap="plasma", linewidths=0)
    plt.colorbar(sc, ax=ax4, label="FDE (m)")
    ax4.set_title("ADE vs FDE per Sample", color="white", fontsize=11)
    ax4.set_xlabel("ADE (m)", color="gray"); ax4.set_ylabel("FDE (m)", color="gray")
    ax4.tick_params(colors="gray"); ax4.set_facecolor(bg); ax4.spines[:].set_color("#333")

    ax5 = fig.add_subplot(gs[1, 1])
    ade_s = np.sort(metrics["ade_per_sample"])
    fde_s = np.sort(metrics["fde_per_sample"])
    cdf   = np.arange(1, len(ade_s) + 1) / len(ade_s)
    ax5.plot(ade_s, cdf, color=accent,  linewidth=2.5, label="ADE")
    ax5.plot(fde_s, cdf, color=accent2, linewidth=2.5, label="FDE")
    ax5.axhline(0.5, color="gray", linewidth=1, linestyle=":")
    ax5.axhline(0.9, color="gray", linewidth=1, linestyle=":")
    ax5.set_title("Cumulative Error Distribution", color="white", fontsize=11)
    ax5.set_xlabel("Error (m)", color="gray"); ax5.set_ylabel("CDF", color="gray")
    ax5.tick_params(colors="gray"); ax5.set_facecolor(bg); ax5.spines[:].set_color("#333")
    ax5.legend(fontsize=9, labelcolor="white", facecolor=bg)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off"); ax6.set_facecolor(bg)
    rows = [
        ["Metric",             "Value"],
        ["Mean ADE",           f"{metrics['mean_ade']:.4f} m"],
        ["Mean FDE",           f"{metrics['mean_fde']:.4f} m"],
        ["Median ADE",         f"{np.median(metrics['ade_per_sample']):.4f} m"],
        ["Median FDE",         f"{np.median(metrics['fde_per_sample']):.4f} m"],
        ["ADE p90",            f"{np.percentile(metrics['ade_per_sample'], 90):.4f} m"],
        ["FDE p90",            f"{np.percentile(metrics['fde_per_sample'], 90):.4f} m"],
        ["Miss Rate FDE>2m",   f"{metrics['miss_rate_2m']:.1%}"],
        ["Miss Rate FDE>4m",   f"{metrics['miss_rate_4m']:.1%}"],
        ["Samples evaluated",  str(metrics["total_samples"])],
    ]
    table = ax6.table(cellText=rows[1:], colLabels=rows[0],
                      cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#1a1a2e" if r % 2 == 0 else "#12122a")
        cell.set_edgecolor("#333")
        cell.set_text_props(color="white" if r > 0 else accent)
    ax6.set_title("Summary Metrics", color="white", fontsize=11, pad=10)

    save_path = output_dir / "evaluation_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"\nPlot saved → {save_path}")

@torch.inference_mode()
def evaluate(
    checkpoint_path: str = CHECKPOINT_PATH,
    dataroot: str        = DATAROOT,
    version: str         = VERSION,
    batch_size: int      = BATCH_SIZE,
    output_dir: str      = OUTPUT_DIR,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"dataset={version}  dataroot={dataroot!r}")

    model   = load_model(checkpoint_path, device)
    dataset = NuScenesDataset(dataroot=dataroot, version=version)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(dataset)} samples...")

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp   = device.type == "cuda"

    all_ade_per_step: list[np.ndarray] = []
    all_ade: list[float] = []
    all_fde: list[float] = []
    total_loss = 0.0
    steps      = 0

    for batch in loader:
        x            = batch["x"].to(device)
        positions    = batch["positions"].to(device)
        gt           = batch["future"].to(device)
        map_features = batch["map"].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            traj, goals, probs = model(x, positions, map_features)
            loss, _, _ = best_of_k_loss(traj, gt, return_metrics=True)

        if not torch.isfinite(loss):
            continue

        best = select_best_mode(traj, gt) if traj.dim() == 5 else traj
        per_step, ade_b, fde_b = compute_ade_fde_per_timestep(best, gt)
        all_ade_per_step.append(per_step)
        all_ade.extend(ade_b.tolist())
        all_fde.extend(fde_b.tolist())
        total_loss += float(loss.item())
        steps      += 1

    if steps == 0:
        print("No valid batches found!")
        return

    ade_arr = np.array(all_ade)
    fde_arr = np.array(all_fde)

    metrics = {
        "mean_ade":         float(ade_arr.mean()),
        "mean_fde":         float(fde_arr.mean()),
        "ade_per_timestep": np.stack(all_ade_per_step).mean(axis=0),
        "ade_per_sample":   ade_arr,
        "fde_per_sample":   fde_arr,
        "miss_rate_2m":     float((fde_arr > 2.0).mean()),
        "miss_rate_4m":     float((fde_arr > 4.0).mean()),
        "total_samples":    len(ade_arr),
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Samples evaluated    : {metrics['total_samples']}")
    print(f"Loss                 : {total_loss / steps:.4f}")
    print(f"Mean ADE             : {metrics['mean_ade']:.4f} m")
    print(f"Mean FDE             : {metrics['mean_fde']:.4f} m")
    print(f"Median ADE           : {np.median(ade_arr):.4f} m")
    print(f"Median FDE           : {np.median(fde_arr):.4f} m")
    print(f"ADE p90              : {np.percentile(ade_arr, 90):.4f} m")
    print(f"FDE p90              : {np.percentile(fde_arr, 90):.4f} m")
    print(f"Miss Rate (FDE>2m)   : {metrics['miss_rate_2m']:.1%}")
    print(f"Miss Rate (FDE>4m)   : {metrics['miss_rate_4m']:.1%}")
    print("=" * 50)
    print(f"\nCheckpoint : {checkpoint_path}")
    print(f"Dataset    : {version} ({dataroot})")

    plot_results(metrics, Path(output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory prediction model")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--dataroot",   default=DATAROOT)
    parser.add_argument("--version",    default=VERSION)
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        dataroot=args.dataroot,
        version=args.version,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
