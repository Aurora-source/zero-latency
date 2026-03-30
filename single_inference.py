import argparse
import random
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Resolve repo root — this file lives at zero-latency/infer_single.py
# All other paths are relative to the same directory.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))   # make sure local modules are importable

from dataset import NuScenesDataset
from modules.decoder.goal_prediction import GoalPredictionNetwork
from modules.decoder.multimodal_decoder import MultiModalDecoder
from modules.input_embedding import InputEmbedding
from modules.scene.scene_context_encoder import SceneContextEncoder
from modules.social.social_transformer import SocialTransformer
from modules.temporal_transformer import TemporalTransformer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Single scene inference with visualisation")
parser.add_argument("--trainval",  action="store_true", help="Use v1.0-trainval instead of v1.0-mini")
parser.add_argument("--seed",      type=int, default=None, help="Random seed for scene selection")
parser.add_argument("--no_anim",   action="store_true",    help="Skip MP4 animation, save PNG only")
parser.add_argument("--min_move",  type=float, default=5.0, help="Min agent movement in metres (default 5.0)")
parser.add_argument("--max_search",type=int,   default=200,  help="Max samples to scan (default 200)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Paths — all relative to REPO_ROOT
# ---------------------------------------------------------------------------

MODEL_PATH   = REPO_ROOT / "models"    / "model_fp32.pt"

if args.trainval:
    DATAROOT = REPO_ROOT / "nuscenes"          # zero-latency/nuscenes/
    VERSION  = "v1.0-trainval"
else:
    DATAROOT = REPO_ROOT / "data" / "raw" / "nuscenes"   # zero-latency/data/raw/nuscenes/
    VERSION  = "v1.0-mini"

OUTPUT_DIR   = REPO_ROOT / "visualisations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG   = OUTPUT_DIR / "multi_agent_prediction.png"
OUTPUT_MP4   = OUTPUT_DIR / "multi_agent_prediction.mp4"

MIN_TOTAL_MOVEMENT = args.min_move
MAX_SEARCH         = args.max_search
SEED               = args.seed

# Animation settings
ANIM_FPS          = 20
ANIM_SPEED_MULT   = 4
ANIM_DPI          = 120
ANIM_BITRATE      = 1800
ANIM_END_HOLD_SEC = 3

print("=" * 55)
print("Zero-Latency — Single Scene Inference")
print("=" * 55)
print(f"  repo root  : {REPO_ROOT}")
print(f"  model      : {MODEL_PATH}")
print(f"  dataset    : {VERSION}  ({DATAROOT})")
print(f"  outputs    : {OUTPUT_DIR}")
print(f"  seed       : {SEED if SEED is not None else 'random'}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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
        emb          = self.embedding(x)
        temp         = self.temporal(emb)
        soc          = self.social(temp, positions)
        scene_out    = self.scene(soc, map_features, agent_positions=positions, map_positions=None)
        goals, probs = self.goal(scene_out)
        traj         = self.decoder(scene_out, goals, probs)
        return traj, goals, probs


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\ndevice={device}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}\n"
        f"Place model_fp32.pt in:  {REPO_ROOT / 'models'}/"
    )

print(f"Loading model: {MODEL_PATH}")
ckpt  = torch.load(str(MODEL_PATH), map_location=device)
model = TrajectoryPredictor().to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print("Model loaded.")


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

if not DATAROOT.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATAROOT}\n"
        f"{'Place v1.0-trainval in:  ' + str(REPO_ROOT) + '/nuscenes/' if args.trainval else 'Place v1.0-mini in:  ' + str(REPO_ROOT) + '/data/raw/nuscenes/'}"
    )

print(f"\nLoading dataset: {VERSION}")
dataset = NuScenesDataset(dataroot=str(DATAROOT), version=VERSION)
print(f"Total samples: {len(dataset)}")


# ---------------------------------------------------------------------------
# Random moving scene selection
# ---------------------------------------------------------------------------

def scene_has_movement(sample, min_movement):
    future = sample["future"]
    for agent_id in range(future.shape[0]):
        if np.linalg.norm(future[agent_id].numpy()[-1] - future[agent_id].numpy()[0]) >= min_movement:
            return True
    return False


def find_random_moving_sample(dataset, min_movement, max_search, seed):
    rng     = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for i, idx in enumerate(indices[:max_search]):
        s = dataset[idx]
        if scene_has_movement(s, min_movement):
            print(f"Found moving scene at index {idx} (checked {i+1} samples)")
            return s, idx
    raise RuntimeError(
        f"No moving scene found in {max_search} samples. "
        f"Try lowering --min_move (currently {min_movement}m)."
    )


print(f"\nSearching for a random moving scene (min_movement={MIN_TOTAL_MOVEMENT}m)...")
sample, sample_idx = find_random_moving_sample(
    dataset, MIN_TOTAL_MOVEMENT, MAX_SEARCH, SEED
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

x            = sample["x"].unsqueeze(0).to(device)
positions    = sample["positions"].unsqueeze(0).to(device)
map_features = sample["map"].unsqueeze(0).to(device)

gt_future      = sample["future"]        # (N, T_future, 2)
past_positions = sample["positions"]     # (T_past, N, 2)
last_pos       = past_positions[-1]      # (N, 2)
num_agents     = gt_future.shape[0]

TYPE_NAMES   = {0: "vehicle", 1: "pedestrian", 2: "cyclist"}
TYPE_ICONS   = {0: "[V]", 1: "[P]", 2: "[C]"}
TYPE_MARKERS = {0: "o",   1: "^",   2: "D"}

raw_types       = sample["x"][-1, :, 7].round().long()
agent_types_np  = raw_types[:num_agents].numpy().astype(int)


# ---------------------------------------------------------------------------
# Ground truth movement
# ---------------------------------------------------------------------------

print("\n=== GROUND TRUTH MOVEMENT ===")
moving_agents = []
for agent_id in range(num_agents):
    gt       = gt_future[agent_id].numpy()
    movement = float(np.linalg.norm(gt[-1] - gt[0]))
    t        = int(agent_types_np[agent_id])
    status   = "moving" if movement >= MIN_TOTAL_MOVEMENT else "static"
    print(f"  Agent {agent_id:>2} {TYPE_ICONS.get(t,'?')} [{TYPE_NAMES.get(t,'?')}]: {movement:>6.2f}m  [{status}]")
    if movement >= MIN_TOTAL_MOVEMENT:
        moving_agents.append(agent_id)
print(f"\n  Moving agents: {moving_agents}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

print("\nRunning inference...")
with torch.inference_mode():
    traj, goals, probs = model(x, positions, map_features)

print(f"  traj  : {tuple(traj.shape)}")
print(f"  goals : {tuple(goals.shape)}")
print(f"  probs : {tuple(probs.shape)}")

# Normalise to (N, K, T, 2)
if traj.shape[1] == num_agents:
    traj_np = traj[0].cpu().numpy()
else:
    traj_np = traj[0].permute(1, 0, 2, 3).cpu().numpy()

K = traj_np.shape[1]

if probs.dim() == 2:
    mode_probs = np.tile(torch.softmax(probs[0], dim=-1).cpu().numpy(), (num_agents, 1))
else:
    mode_probs = torch.softmax(probs[0], dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# minADE mode selection — identical to best_of_k_loss in evaluate-mini.py
# ---------------------------------------------------------------------------

print("\n=== PER-AGENT METRICS (minADE — matches evaluate-mini.py) ===")
all_ade    = []
all_fde    = []
best_modes = []

for agent_id in range(num_agents):
    gt     = gt_future[agent_id].numpy()
    best_k = min(range(K),
                 key=lambda k: float(np.linalg.norm(traj_np[agent_id, k] - gt, axis=-1).mean()))
    best_modes.append(best_k)
    pred   = traj_np[agent_id, best_k]
    ade    = float(np.linalg.norm(pred - gt, axis=-1).mean())
    fde    = float(np.linalg.norm(pred[-1] - gt[-1]))
    all_ade.append(ade)
    all_fde.append(fde)
    t = int(agent_types_np[agent_id])
    print(
        f"  Agent {agent_id:>2} {TYPE_ICONS.get(t,'?')} [{TYPE_NAMES.get(t,'?')}]"
        f" | mode={best_k} p={mode_probs[agent_id][best_k]:.3f}"
        f" | ADE={ade:.3f}m  FDE={fde:.3f}m"
    )

mean_ade = float(np.mean(all_ade))
mean_fde = float(np.mean(all_fde))
print(f"\n  Mean ADE : {mean_ade:.4f} m")
print(f"  Mean FDE : {mean_fde:.4f} m")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

TABLEAU10 = [
    "#4e79a7","#f28e2b","#e15759","#76b7b2",
    "#59a14f","#edc948","#b07aa1","#ff9da7",
    "#9c755f","#bab0ac",
]
colors = TABLEAU10

LEGEND_W = 2.8

# Axis limits from past + GT + best pred (percentile clipped)
_ax_pts, _ay_pts = [], []
for a in range(num_agents):
    for arr in [past_positions[:, a].numpy(),
                gt_future[a].numpy(),
                traj_np[a, best_modes[a]]]:
        _ax_pts.extend(arr[:, 0].tolist())
        _ay_pts.extend(arr[:, 1].tolist())

_pad  = 3.0
x_min = float(np.percentile(_ax_pts,  2)) - _pad
x_max = float(np.percentile(_ax_pts, 98)) + _pad
y_min = float(np.percentile(_ay_pts,  2)) - _pad
y_max = float(np.percentile(_ay_pts, 98)) + _pad

_dw   = max(x_max - x_min, 1e-3)
_dh   = max(y_max - y_min, 1e-3)
_pw   = min(11.0, max(7.0, _dw / 10.0))
_ph   = min(8.0,  max(4.0, _pw * (_dh / _dw)))
fig_w = _pw + LEGEND_W
fig_h = _ph
_lf   = LEGEND_W / fig_w


def _apply_axes(ax):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _make_title(extra=""):
    return (
        f"Multi-Agent Trajectory Prediction  |  idx={sample_idx}  |  {VERSION}\n"
        f"Mean ADE={mean_ade:.3f} m   Mean FDE={mean_fde:.3f} m"
        f"  (minADE — matches evaluate-mini.py){extra}"
    )


def _build_legend(ax):
    handles = []
    for agent_id in range(num_agents):
        t = int(agent_types_np[agent_id])
        handles.append(mpatches.Patch(
            color=colors[agent_id % len(colors)],
            label=(
                f"{TYPE_ICONS.get(t,'?')} {TYPE_NAMES.get(t,'?').capitalize()} {agent_id}"
                f"  ADE={all_ade[agent_id]:.2f}m"
                f"  FDE={all_fde[agent_id]:.2f}m"
            )
        ))
    handles += [
        mpatches.Patch(color="none", label=""),
        plt.Line2D([0],[0], color="#555", linewidth=2.0, label="Predicted (minADE mode)"),
        plt.Line2D([0],[0], color="#555", linewidth=1.5, linestyle="--", label="Ground truth"),
        plt.Line2D([0],[0], color="#555", linewidth=1.0, linestyle=":",  label="Past trajectory"),
        plt.Line2D([0],[0], color="#555", linewidth=0.8, alpha=0.35,     label="All modes (faint)"),
        mpatches.Patch(color="none", label=""),
        plt.Line2D([0],[0], marker="o", color="#555", markersize=7, linewidth=0, label="[V] Vehicle"),
        plt.Line2D([0],[0], marker="^", color="#555", markersize=7, linewidth=0, label="[P] Pedestrian"),
        plt.Line2D([0],[0], marker="D", color="#555", markersize=7, linewidth=0, label="[C] Cyclist"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0, fontsize=7.5,
        framealpha=0.95, edgecolor="#ccc",
        fancybox=False,
        title="Agents & styles", title_fontsize=8,
    )
    leg.get_title().set_fontweight("bold")
    return leg


def _draw_all(ax):
    for agent_id in range(num_agents):
        color  = colors[agent_id % len(colors)]
        best_k = best_modes[agent_id]
        pred   = traj_np[agent_id, best_k]
        gt     = gt_future[agent_id].numpy()
        past   = past_positions[:, agent_id].numpy()
        cur    = last_pos[agent_id].numpy()
        mk     = TYPE_MARKERS.get(int(agent_types_np[agent_id]), "o")

        for k in range(K):
            ax.plot(traj_np[agent_id, k, :, 0], traj_np[agent_id, k, :, 1],
                    color=color, alpha=0.10, linewidth=0.7, zorder=1)
        ax.plot(past[:, 0], past[:, 1],
                color=color, linewidth=1.1, linestyle=":", alpha=0.55, zorder=2)
        ax.plot(pred[:, 0], pred[:, 1],
                color=color, marker=mk, markersize=4,
                linewidth=2.0, alpha=0.95, zorder=3)
        ax.plot(gt[:, 0], gt[:, 1],
                color=color, marker="x", markersize=5,
                linewidth=1.6, linestyle="--", alpha=0.85, zorder=3)
        ax.scatter(cur[0], cur[1], color=color, s=80, zorder=6,
                   marker=mk, edgecolors="white", linewidths=1.2)
        ax.annotate(f"A{agent_id}", xy=(pred[-1, 0], pred[-1, 1]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=6.5, color=color, alpha=0.85, zorder=7)


# ---------------------------------------------------------------------------
# Static PNG
# ---------------------------------------------------------------------------

fig_s, ax_s = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
fig_s.subplots_adjust(left=0.08, right=1.0 - _lf - 0.01, top=0.88, bottom=0.10)
_apply_axes(ax_s)
_draw_all(ax_s)
_build_legend(ax_s)
ax_s.set_title(_make_title(), fontsize=9, loc="left", pad=8)
fig_s.text(0.01, 0.005,
           "● = current pos   ── best predicted   - - ground truth   ··· past   faint = all modes",
           fontsize=7, color="#666", va="bottom")

fig_s.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches="tight")
plt.close(fig_s)
print(f"\nStatic PNG saved → {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Animated MP4
# ---------------------------------------------------------------------------

if not args.no_anim:
    print(f"Building animation → {OUTPUT_MP4}")

    T_past        = past_positions.shape[0]
    T_future      = gt_future.shape[1]
    HOLD_FRAMES   = ANIM_FPS * ANIM_END_HOLD_SEC
    logical_steps = T_past + T_future
    total_frames  = logical_steps * ANIM_SPEED_MULT + HOLD_FRAMES

    fig_a, ax_a = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
    fig_a.subplots_adjust(left=0.08, right=1.0 - _lf - 0.01, top=0.88, bottom=0.10)
    _apply_axes(ax_a)
    _build_legend(ax_a)

    agent_arts = []
    for agent_id in range(num_agents):
        color = colors[agent_id % len(colors)]
        mk    = TYPE_MARKERS.get(int(agent_types_np[agent_id]), "o")
        agent_arts.append(dict(
            mode_lines=[ax_a.plot([], [], color=color, alpha=0.10, linewidth=0.7, zorder=1)[0] for _ in range(K)],
            pred_line=ax_a.plot([], [], color=color, marker=mk, markersize=4, linewidth=2.0, alpha=0.95, zorder=3)[0],
            gt_line=ax_a.plot([], [], color=color, marker="x", markersize=5, linewidth=1.6, linestyle="--", alpha=0.85, zorder=3)[0],
            past_line=ax_a.plot([], [], color=color, linewidth=1.1, linestyle=":", alpha=0.55, zorder=2)[0],
            cur_dot=ax_a.scatter([], [], color=color, s=80, zorder=6, marker=mk, edgecolors="white", linewidths=1.2),
            label_txt=ax_a.text(0, 0, f"A{agent_id}", fontsize=6.5, color=color, alpha=0.0, zorder=7),
        ))

    title_obj  = ax_a.set_title("", fontsize=9, loc="left", pad=8)
    footer_obj = fig_a.text(0.01, 0.005,
                            "● = current pos   ── best predicted   - - ground truth   ··· past",
                            fontsize=7, color="#666", va="bottom")

    def _update(frame):
        logical  = min(frame // ANIM_SPEED_MULT, logical_steps - 1)
        in_past  = logical < T_past
        fut_step = max(0, logical - T_past + 1)

        phase = (f"\nPast step {logical+1}/{T_past}" if in_past
                 else f"\nFuture step {min(fut_step, T_future)}/{T_future}"
                 + (" (complete)" if fut_step >= T_future else ""))
        title_obj.set_text(_make_title(phase))

        for agent_id, arts in enumerate(agent_arts):
            best_k = best_modes[agent_id]
            pred   = traj_np[agent_id, best_k]
            gt     = gt_future[agent_id].numpy()
            past   = past_positions[:, agent_id].numpy()
            cur    = last_pos[agent_id].numpy()

            p_sl = past[:logical+1] if in_past else past
            arts["past_line"].set_data(p_sl[:, 0], p_sl[:, 1]) if len(p_sl) else arts["past_line"].set_data([], [])
            arts["cur_dot"].set_offsets(cur.reshape(1, 2) if not in_past else np.empty((0, 2)))

            if not in_past and fut_step > 0:
                for k, ml in enumerate(arts["mode_lines"]):
                    ml.set_data(traj_np[agent_id, k, :, 0], traj_np[agent_id, k, :, 1])
                arts["pred_line"].set_data(pred[:fut_step, 0], pred[:fut_step, 1])
                arts["gt_line"].set_data(gt[:fut_step, 0], gt[:fut_step, 1])
                arts["label_txt"].set_position((pred[min(fut_step-1, len(pred)-1), 0]+0.3,
                                                pred[min(fut_step-1, len(pred)-1), 1]+0.3))
                arts["label_txt"].set_alpha(0.85)
            else:
                for ml in arts["mode_lines"]: ml.set_data([], [])
                arts["pred_line"].set_data([], [])
                arts["gt_line"].set_data([], [])
                arts["label_txt"].set_alpha(0.0)

        updated = [title_obj, footer_obj]
        for arts in agent_arts:
            updated += arts["mode_lines"]
            updated += [arts["pred_line"], arts["gt_line"], arts["past_line"],
                        arts["cur_dot"], arts["label_txt"]]
        return updated

    ani = animation.FuncAnimation(
        fig_a, _update, frames=total_frames,
        interval=1000 // ANIM_FPS, blit=False,
    )

    try:
        writer = animation.FFMpegWriter(
            fps=ANIM_FPS, bitrate=ANIM_BITRATE,
            metadata=dict(title="Multi-Agent Trajectory Prediction"),
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        ani.save(str(OUTPUT_MP4), writer=writer, dpi=ANIM_DPI)
        print(f"Animation saved → {OUTPUT_MP4}")
    except FileNotFoundError:
        print(
            "\n  FFmpeg not found — animation skipped.\n"
            "  Install FFmpeg: https://ffmpeg.org/download.html\n"
            "  Re-run without --no_anim once FFmpeg is on PATH."
        )

    plt.close(fig_a)

print(f"\nTo reproduce this scene: python infer_single.py --seed {sample_idx}")
print("Done.")
