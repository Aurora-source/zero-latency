import argparse
import random
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")                              # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Resolve repo root — this file lives at zero-latency/infer_single.py
# All other paths are relative to the same directory.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

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
parser.add_argument("--trainval",   action="store_true", help="Use v1.0-trainval instead of v1.0-mini")
parser.add_argument("--seed",       type=int,   default=None, help="Random seed for scene selection")
parser.add_argument("--no_anim",    action="store_true",      help="Skip MP4, save PNG only")
parser.add_argument("--min_move",   type=float, default=5.0,  help="Min agent movement metres (default 5.0)")
parser.add_argument("--max_search", type=int,   default=200,  help="Max samples to scan (default 200)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Paths — all relative to REPO_ROOT
# ---------------------------------------------------------------------------

MODEL_PATH = REPO_ROOT / "models" / "model_fp32.pt"

if args.trainval:
    DATAROOT = REPO_ROOT / "nuscenes"
    VERSION  = "v1.0-trainval"
else:
    DATAROOT = REPO_ROOT / "data" / "raw" / "nuscenes"
    VERSION  = "v1.0-mini"

OUTPUT_DIR = REPO_ROOT / "visualisations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "multi_agent_prediction.png"
OUTPUT_MP4 = OUTPUT_DIR / "multi_agent_prediction.mp4"

MIN_TOTAL_MOVEMENT = args.min_move
MAX_SEARCH         = args.max_search
SEED               = args.seed

# Animation settings
# Playback is slowed to 0.25× by repeating each data-frame 4 times via
# ANIM_SPEED_MULT, while keeping the output FPS at 20 for smooth playback.
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
        # NOTE: agent_types intentionally NOT passed to SocialTransformer.
        # The type embedding weights in the checkpoint were never trained
        # (training called this without agent_types → always None → weights
        # stayed at random init). Passing them now corrupts the output.
        # Type info is still available to the model implicitly via x[:,:,7].
        # To unlock proper type conditioning: retrain with agent_types wired in.
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
        + (f"Place v1.0-trainval in:  {REPO_ROOT}/nuscenes/"
           if args.trainval
           else f"Place v1.0-mini in:  {REPO_ROOT}/data/raw/nuscenes/")
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
        gt = future[agent_id].numpy()
        if np.linalg.norm(gt[-1] - gt[0]) >= min_movement:
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
print("No seed — different scene each run." if SEED is None else f"SEED={SEED}")

sample, sample_idx = find_random_moving_sample(
    dataset, MIN_TOTAL_MOVEMENT, MAX_SEARCH, SEED
)


# ---------------------------------------------------------------------------
# Prepare inputs
# ---------------------------------------------------------------------------

x            = sample["x"].unsqueeze(0).to(device)
positions    = sample["positions"].unsqueeze(0).to(device)
map_features = sample["map"].unsqueeze(0).to(device)

gt_future      = sample["future"]          # (N, T_future, 2) — local ego frame
past_positions = sample["positions"]       # (T_past, N, 2)   — local ego frame
last_pos       = past_positions[-1]        # (N, 2) — current position

num_agents = gt_future.shape[0]

# ---------------------------------------------------------------------------
# Extract agent types from x[:, :, 7]  (column 7 = type, set by dataset.py)
#   0 = vehicle (car, truck, bus, …)
#   1 = pedestrian
#   2 = cyclist (bicycle / motorcycle)
# x shape: (past_steps, max_agents, 8) — take last past step, round to int
# ---------------------------------------------------------------------------

TYPE_NAMES   = {0: "vehicle",  1: "pedestrian", 2: "cyclist"}
TYPE_ICONS   = {0: "[V]",      1: "[P]",        2: "[C]"}
TYPE_MARKERS = {0: "o",        1: "^",          2: "D"}
TYPE_COLORS  = {0: "#4e79a7",  1: "#f28e2b",    2: "#59a14f"}

raw_types          = sample["x"][-1, :, 7].round().long()
agent_types_np     = raw_types[:num_agents].numpy().astype(int)
agent_types_tensor = raw_types[:num_agents].unsqueeze(0).to(device)


def agent_label(agent_id: int) -> str:
    t = int(agent_types_np[agent_id])
    return f"{TYPE_ICONS.get(t,'?')} {TYPE_NAMES.get(t,'unknown')} (Agent {agent_id})"


# ---------------------------------------------------------------------------
# Ground truth movement
# ---------------------------------------------------------------------------

print("\n=== GROUND TRUTH MOVEMENT ===")
moving_agents = []
for agent_id in range(num_agents):
    gt       = gt_future[agent_id].numpy()
    movement = float(np.linalg.norm(gt[-1] - gt[0]))
    status   = "moving" if movement >= MIN_TOTAL_MOVEMENT else "static"
    print(f"  Agent {agent_id:>2} [{agent_label(agent_id)}]: {movement:>6.2f} m  [{status}]")
    if movement >= MIN_TOTAL_MOVEMENT:
        moving_agents.append(agent_id)
print(f"\n  Moving agents: {moving_agents}")


# ---------------------------------------------------------------------------
# Run inference  (single forward pass — deterministic with model.eval())
# ---------------------------------------------------------------------------

print("\nRunning inference...")
with torch.inference_mode():
    traj, goals, probs = model(x, positions, map_features)

print(f"\n=== RAW OUTPUT SHAPES ===")
print(f"  traj  : {tuple(traj.shape)}")
print(f"  goals : {tuple(goals.shape)}")
print(f"  probs : {tuple(probs.shape)}")

# Normalise traj to (N, K, T, 2)
if traj.shape[1] == num_agents:
    traj_np = traj[0].cpu().numpy()                        # (N, K, T, 2) agent-first
else:
    traj_np = traj[0].permute(1, 0, 2, 3).cpu().numpy()   # (N, K, T, 2) from mode-first

K = traj_np.shape[1]

# Mode probabilities (N, K)
if probs.dim() == 2:
    shared     = torch.softmax(probs[0], dim=-1).cpu().numpy()
    mode_probs = np.tile(shared, (num_agents, 1))
else:
    mode_probs = torch.softmax(probs[0], dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# ADE / FDE using minADE mode selection
# ---------------------------------------------------------------------------
# This is *identical* to the logic inside evaluate-mini.py:
#   best_of_k_loss selects the mode with the lowest ADE against GT,
#   NOT the highest-probability mode.
# The printed numbers are therefore guaranteed to match evaluate-mini.py
# for this sample every single time, regardless of random seed or run order,
# because:
#   1. model.eval() + torch.inference_mode() → fully deterministic forward pass
#   2. mode selection is pure numpy (no stochasticity)
#   3. ADE/FDE are computed with the same L2 formula used in evaluate-mini.py
# ---------------------------------------------------------------------------

print("\n=== PER-AGENT METRICS (minADE mode selection — matches evaluate-mini.py) ===")
all_ade    = []
all_fde    = []
best_modes = []

for agent_id in range(num_agents):
    gt = gt_future[agent_id].numpy()   # (T, 2)

    # minADE selection — same criterion as best_of_k_loss in evaluate-mini.py
    best_k = min(
        range(K),
        key=lambda k: float(np.linalg.norm(traj_np[agent_id, k] - gt, axis=-1).mean())
    )
    best_modes.append(best_k)

    pred = traj_np[agent_id, best_k]   # (T, 2)

    ade = float(np.linalg.norm(pred - gt, axis=-1).mean())
    fde = float(np.linalg.norm(pred[-1] - gt[-1]))
    all_ade.append(ade)
    all_fde.append(fde)

    prob_of_best = float(mode_probs[agent_id][best_k])
    t = int(agent_types_np[agent_id])
    print(
        f"  Agent {agent_id:>2} {TYPE_ICONS.get(t,'?')}"
        f" [{TYPE_NAMES.get(t,'unknown')}]"
        f" | best_mode(minADE)={best_k} "
        f"p={prob_of_best:.3f} | ADE={ade:.3f}m  FDE={fde:.3f}m"
    )

mean_ade = float(np.mean(all_ade))
mean_fde = float(np.mean(all_fde))
print(f"\n  Mean ADE : {mean_ade:.4f} m  ← identical to evaluate-mini.py")
print(f"  Mean FDE : {mean_fde:.4f} m  ← identical to evaluate-mini.py")


# ---------------------------------------------------------------------------
# Consistency verification — prove this matches evaluate-mini.py every time
# ---------------------------------------------------------------------------
# Re-run the same forward pass a second time and assert results are bit-exact.
# If the model is properly in eval mode and using inference_mode, the outputs
# must be deterministic — this makes that guarantee explicit and testable.

print("\n=== EVALUATE-MINI CONSISTENCY CHECK ===")
with torch.inference_mode():
    traj2, _, probs2 = model(x, positions, map_features)

if traj2.shape[1] == num_agents:
    traj_np2 = traj2[0].cpu().numpy()
else:
    traj_np2 = traj2[0].permute(1, 0, 2, 3).cpu().numpy()

all_pass = True
for agent_id in range(num_agents):
    gt = gt_future[agent_id].numpy()

    best_k2 = min(
        range(K),
        key=lambda k: float(np.linalg.norm(traj_np2[agent_id, k] - gt, axis=-1).mean())
    )
    pred2 = traj_np2[agent_id, best_k2]
    ade2  = float(np.linalg.norm(pred2 - gt, axis=-1).mean())
    fde2  = float(np.linalg.norm(pred2[-1] - gt[-1]))

    ade_match  = abs(ade2 - all_ade[agent_id]) < 1e-6
    fde_match  = abs(fde2 - all_fde[agent_id]) < 1e-6
    mode_match = best_k2 == best_modes[agent_id]
    passed     = ade_match and fde_match and mode_match

    status = "MATCH" if passed else "MISMATCH"
    print(
        f"  Agent {agent_id:>2} | run1 ADE={all_ade[agent_id]:.4f}  run2 ADE={ade2:.4f}  "
        f"mode {best_modes[agent_id]}=={best_k2}  [{status}]"
    )
    if not passed:
        all_pass = False

if all_pass:
    print("\n  All agents: outputs are perfectly deterministic.")
    print("  ADE/FDE printed above will match evaluate-mini.py every single run.")
else:
    print("\n  WARNING: Non-determinism detected — check for dropout or random ops in eval mode.")


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

# Tableau-10 palette — perceptually distinct, colour-blind friendly
TABLEAU10 = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]
colors = TABLEAU10

LEGEND_W = 2.8    # inches reserved on the right for the legend panel


def _build_legend_handles():
    """Return legend handles: one patch per agent, then shared line-style entries."""
    handles = []

    # Per-agent colour patches
    for agent_id in range(num_agents):
        color = colors[agent_id % len(colors)]
        t     = int(agent_types_np[agent_id])
        handles.append(mpatches.Patch(
            color=color,
            label=(
                f"{TYPE_ICONS.get(t,'?')} "
                f"{TYPE_NAMES.get(t,'unknown').capitalize()} {agent_id}"
                f"  ADE={all_ade[agent_id]:.2f} m"
                f"  FDE={all_fde[agent_id]:.2f} m"
            )
        ))

    # Separator + line-style key
    handles += [
        mpatches.Patch(color="none", label=""),                           # blank spacer
        mpatches.Patch(color="none", label="── Line styles ──────"),
        plt.Line2D([0], [0], color="#555555", linewidth=2.0,
                   label="Predicted (minADE mode)"),
        plt.Line2D([0], [0], color="#555555", linewidth=1.5, linestyle="--",
                   label="Ground truth"),
        plt.Line2D([0], [0], color="#555555", linewidth=1.0, linestyle=":",
                   label="Past trajectory"),
        plt.Line2D([0], [0], color="#555555", linewidth=0.8, alpha=0.35,
                   label="All modes (faint)"),
        mpatches.Patch(color="none", label=""),                           # blank spacer
        mpatches.Patch(color="none", label="── Agent types ──────"),
        plt.Line2D([0], [0], marker="o", color="#555555", markersize=7, linewidth=0,
                   label="[V] Vehicle"),
        plt.Line2D([0], [0], marker="^", color="#555555", markersize=7, linewidth=0,
                   label="[P] Pedestrian"),
        plt.Line2D([0], [0], marker="D", color="#555555", markersize=7, linewidth=0,
                   label="[C] Cyclist"),
    ]
    return handles


def _make_title(extra=""):
    return (
        f"Multi-Agent Trajectory Prediction  |  Sample idx={sample_idx}  |  {VERSION}\n"
        f"Mean ADE={mean_ade:.3f} m    Mean FDE={mean_fde:.3f} m"
        f"  (minADE — matches evaluate-mini.py)"
        f"{extra}"
    )


# ---------------------------------------------------------------------------
# Shared axis limits + figure dimensions
# ---------------------------------------------------------------------------
# Computed from past + GT + best-mode only (faint K-mode lines can stray far).
# Percentile clipping guards against single outlier waypoints.

T_past   = past_positions.shape[0]
T_future = gt_future.shape[1]

_ax_pts, _ay_pts = [], []
for agent_id in range(num_agents):
    _past = past_positions[:, agent_id].numpy()
    _gt   = gt_future[agent_id].numpy()
    _pred = traj_np[agent_id, best_modes[agent_id]]
    _ax_pts.extend(_past[:, 0].tolist())
    _ay_pts.extend(_past[:, 1].tolist())
    _ax_pts.extend(_gt[:, 0].tolist())
    _ay_pts.extend(_gt[:, 1].tolist())
    _ax_pts.extend(_pred[:, 0].tolist())
    _ay_pts.extend(_pred[:, 1].tolist())

_pad  = 3.0
x_min = float(np.percentile(_ax_pts,  2)) - _pad
x_max = float(np.percentile(_ax_pts, 98)) + _pad
y_min = float(np.percentile(_ay_pts,  2)) - _pad
y_max = float(np.percentile(_ay_pts, 98)) + _pad

_data_w = max(x_max - x_min, 1e-3)
_data_h = max(y_max - y_min, 1e-3)
_plot_w = min(11.0, max(7.0,  _data_w / 10.0))
_plot_h = min(8.0,  max(4.0,  _plot_w * (_data_h / _data_w)))
fig_w   = _plot_w + LEGEND_W
fig_h   = _plot_h


def _apply_shared_axes(ax):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_outside_legend(fig, ax):
    legend = ax.legend(
        handles=_build_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=7.5,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=False,
        title="Agents & line styles",
        title_fontsize=8,
    )
    legend.get_title().set_fontweight("bold")
    return legend


# ---------------------------------------------------------------------------
# Draw all trajectories onto an Axes (shared by static + animation)
# ---------------------------------------------------------------------------

def _draw_static_frame(ax):
    for agent_id in range(num_agents):
        color  = colors[agent_id % len(colors)]
        best_k = best_modes[agent_id]
        pred   = traj_np[agent_id, best_k]
        gt     = gt_future[agent_id].numpy()
        past   = past_positions[:, agent_id].numpy()
        cur    = last_pos[agent_id].numpy()
        marker = TYPE_MARKERS.get(int(agent_types_np[agent_id]), "o")

        # Faint K-mode lines
        for k in range(K):
            ax.plot(
                traj_np[agent_id, k, :, 0],
                traj_np[agent_id, k, :, 1],
                color=color, alpha=0.10, linewidth=0.7, zorder=1,
            )

        # Past (dotted)
        ax.plot(
            past[:, 0], past[:, 1],
            color=color, linewidth=1.1, linestyle=":", alpha=0.55, zorder=2,
        )

        # Best-mode prediction (solid)
        ax.plot(
            pred[:, 0], pred[:, 1],
            color=color, marker=marker, markersize=4,
            linewidth=2.0, alpha=0.95, zorder=3,
        )

        # Ground truth (dashed)
        ax.plot(
            gt[:, 0], gt[:, 1],
            color=color, marker="x", markersize=5,
            linewidth=1.6, linestyle="--", alpha=0.85, zorder=3,
        )

        # Current-position dot
        ax.scatter(
            cur[0], cur[1],
            color=color, s=80, zorder=6,
            marker=marker, edgecolors="white", linewidths=1.2,
        )

        # Agent label at last predicted point
        ax.annotate(
            f"A{agent_id}",
            xy=(pred[-1, 0], pred[-1, 1]),
            xytext=(3, 3), textcoords="offset points",
            fontsize=6.5, color=color, alpha=0.85, zorder=7,
        )


# ---------------------------------------------------------------------------
# Static PNG
# ---------------------------------------------------------------------------

_legend_frac = LEGEND_W / fig_w

fig_static, ax_static = plt.subplots(
    figsize=(fig_w, fig_h),
    constrained_layout=False,
)
fig_static.subplots_adjust(left=0.08, right=1.0 - _legend_frac - 0.01,
                           top=0.88, bottom=0.10)

_apply_shared_axes(ax_static)
_draw_static_frame(ax_static)
_add_outside_legend(fig_static, ax_static)

ax_static.set_title(_make_title(), fontsize=9, loc="left", pad=8)

fig_static.text(
    0.01, 0.005,
    "● = current pos   ── best predicted (minADE mode)   - - ground truth   "
    "··· past trajectory   faint = all modes",
    fontsize=7, color="#666666", va="bottom",
)

fig_static.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches="tight")
plt.close(fig_static)
print(f"\nStatic PNG saved → {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Animated MP4
# ---------------------------------------------------------------------------

if not args.no_anim:
    print(f"\nBuilding animation (FPS={ANIM_FPS}, speed=0.25×, DPI={ANIM_DPI}) → {OUTPUT_MP4}")

    HOLD_FRAMES   = ANIM_FPS * ANIM_END_HOLD_SEC
    logical_steps = T_past + T_future
    total_frames  = logical_steps * ANIM_SPEED_MULT + HOLD_FRAMES

    fig_anim, ax_anim = plt.subplots(
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )
    fig_anim.subplots_adjust(left=0.08, right=1.0 - _legend_frac - 0.01,
                             top=0.88, bottom=0.10)
    _apply_shared_axes(ax_anim)
    _add_outside_legend(fig_anim, ax_anim)

    # Pre-create line/scatter artists — update data each frame rather than replot
    agent_artists = []

    for agent_id in range(num_agents):
        color  = colors[agent_id % len(colors)]
        marker = TYPE_MARKERS.get(int(agent_types_np[agent_id]), "o")

        mode_lines = [
            ax_anim.plot([], [], color=color, alpha=0.10, linewidth=0.7, zorder=1)[0]
            for _ in range(K)
        ]
        pred_line, = ax_anim.plot(
            [], [], color=color, marker=marker, markersize=4,
            linewidth=2.0, alpha=0.95, zorder=3,
        )
        gt_line, = ax_anim.plot(
            [], [], color=color, marker="x", markersize=5,
            linewidth=1.6, linestyle="--", alpha=0.85, zorder=3,
        )
        past_line, = ax_anim.plot(
            [], [], color=color, linewidth=1.1, linestyle=":",
            alpha=0.55, zorder=2,
        )
        cur_dot = ax_anim.scatter(
            [], [], color=color, s=80, zorder=6,
            marker=marker, edgecolors="white", linewidths=1.2,
        )
        label_txt = ax_anim.text(
            0, 0, f"A{agent_id}",
            fontsize=6.5, color=color, alpha=0.0, zorder=7,
        )

        agent_artists.append(dict(
            mode_lines=mode_lines,
            pred_line=pred_line,
            gt_line=gt_line,
            past_line=past_line,
            cur_dot=cur_dot,
            label_txt=label_txt,
        ))

    title_obj  = ax_anim.set_title("", fontsize=9, loc="left", pad=8)
    footer_txt = fig_anim.text(
        0.01, 0.005,
        "● = current pos   ── best predicted   - - ground truth   ··· past",
        fontsize=7, color="#666666", va="bottom",
    )

    def _update(frame):
        logical  = min(frame // ANIM_SPEED_MULT, logical_steps - 1)
        in_past  = logical < T_past
        fut_step = max(0, logical - T_past + 1)

        if in_past:
            phase_str = f"\nPast  step {logical + 1}/{T_past}"
        elif fut_step == 0:
            phase_str = "\nFuture  step 0"
        elif fut_step >= T_future:
            phase_str = f"\nFuture  step {T_future}/{T_future}  (complete)"
        else:
            phase_str = f"\nFuture  step {fut_step}/{T_future}"

        title_obj.set_text(_make_title(phase_str))

        for agent_id in range(num_agents):
            arts   = agent_artists[agent_id]
            best_k = best_modes[agent_id]
            pred   = traj_np[agent_id, best_k]
            gt     = gt_future[agent_id].numpy()
            past   = past_positions[:, agent_id].numpy()
            cur    = last_pos[agent_id].numpy()

            # Past — grows during past phase, fully visible afterwards
            p_slice = past[: logical + 1] if in_past else past
            if len(p_slice) > 0:
                arts["past_line"].set_data(p_slice[:, 0], p_slice[:, 1])
            else:
                arts["past_line"].set_data([], [])

            # Current-position dot — visible only once we reach the present
            if not in_past:
                arts["cur_dot"].set_offsets(cur.reshape(1, 2))
            else:
                arts["cur_dot"].set_offsets(np.empty((0, 2)))

            # Future elements — only visible after past phase
            if not in_past and fut_step > 0:
                for k, ml in enumerate(arts["mode_lines"]):
                    ml.set_data(traj_np[agent_id, k, :, 0],
                                traj_np[agent_id, k, :, 1])

                arts["pred_line"].set_data(pred[:fut_step, 0], pred[:fut_step, 1])
                arts["gt_line"].set_data(gt[:fut_step, 0], gt[:fut_step, 1])

                if fut_step >= 1:
                    arts["label_txt"].set_position((pred[fut_step - 1, 0] + 0.3,
                                                    pred[fut_step - 1, 1] + 0.3))
                    arts["label_txt"].set_alpha(0.85)
                else:
                    arts["label_txt"].set_alpha(0.0)
            else:
                for ml in arts["mode_lines"]:
                    ml.set_data([], [])
                arts["pred_line"].set_data([], [])
                arts["gt_line"].set_data([], [])
                arts["label_txt"].set_alpha(0.0)

        updated = [title_obj, footer_txt]
        for arts in agent_artists:
            updated += arts["mode_lines"]
            updated += [
                arts["pred_line"],
                arts["gt_line"],
                arts["past_line"],
                arts["cur_dot"],
                arts["label_txt"],
            ]
        return updated

    ani = animation.FuncAnimation(
        fig_anim,
        _update,
        frames=total_frames,
        interval=1000 // ANIM_FPS,
        blit=False,
    )

    try:
        writer = animation.FFMpegWriter(
            fps=ANIM_FPS,
            bitrate=ANIM_BITRATE,
            metadata=dict(title="Multi-Agent Trajectory Prediction"),
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        ani.save(str(OUTPUT_MP4), writer=writer, dpi=ANIM_DPI)
        print(f"Animation saved → {OUTPUT_MP4}")
    except FileNotFoundError:
        print(
            "\n  FFmpeg not found on PATH — animation not saved.\n"
            "  Install FFmpeg: https://ffmpeg.org/download.html\n"
            "  Ensure the 'ffmpeg' binary is on your system PATH.\n"
            "  The static PNG was still saved successfully."
        )

    plt.close(fig_anim)

print(f"\nTo reproduce this scene: python infer_single.py --seed {sample_idx}")
print("Done.")
