"""Training entrypoint optimised for Windows + 8 GB VRAM.

Architecture dims (matched to rebuilt modules):
  embed_dim = 256   — scaled-down from 512 to fit 8 GB VRAM
  num_goals = 6
  future_steps = 12

8 GB VRAM strategy:
  - Smaller embed_dim (256) reduces activation memory ~4× vs 512
  - micro batch_size=4, grad_accum=8  →  effective batch=32
  - num_workers=0  (Windows spawn is unreliable with multiprocessing)
  - torch.compile OFF  (no Triton on Windows)
  - fp16 AMP with GradScaler  (8 GB cards often lack bf16 support)
  - ReduceLROnPlateau scheduler  (robust to noisy mini-dataset loss curves)
"""

from __future__ import annotations

import inspect
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore", message="expandable_segments not supported")
warnings.filterwarnings("ignore", message="Online softmax is disabled")

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from dataset import NuScenesDataset
from modules.decoder.goal_prediction import GoalPredictionNetwork
from modules.decoder.multimodal_decoder import MultiModalDecoder
from modules.input_embedding import InputEmbedding
from modules.scene.scene_context_encoder import SceneContextEncoder
from modules.social.social_transformer import SocialTransformer
from modules.temporal_transformer import TemporalTransformer
from utils.losses import best_of_k_loss, goal_classification_loss

# ── Architecture constants — scaled for 8 GB VRAM ────────────────────────────
EMBED        = 256   # 512→256: cuts activation memory ~4×
NUM_GOALS    = 6
FUTURE_STEPS = 12
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration for Windows + 8 GB VRAM GPU."""

    dataset_root: str   = os.getenv("NUSCENES_ROOT", "data/raw/nuscenes")
    version: str        = "v1.0-mini"
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    dataset_limit: int  = int(os.getenv("DATASET_LIMIT", "404"))
    run_epochs: int     = int(os.getenv("RUN_EPOCHS", "40"))

    # --- 8 GB VRAM: small micro-batch, accumulate to effective=32 ---
    batch_size: int        = int(os.getenv("BATCH_SIZE", "4"))
    eval_batch_size: int   = int(os.getenv("EVAL_BATCH_SIZE", "4"))
    grad_accum_steps: int  = int(os.getenv("GRAD_ACCUM_STEPS", "8"))

    # --- Optimiser ---
    learning_rate: float      = float(os.getenv("LR", "5e-5"))
    min_learning_rate: float  = float(os.getenv("MIN_LR", "1e-6"))
    weight_decay: float       = float(os.getenv("WEIGHT_DECAY", "1e-2"))
    goal_loss_weight: float   = float(os.getenv("GOAL_LOSS_WEIGHT", "0.1"))
    gradient_clip_norm: float = float(os.getenv("GRAD_CLIP", "1.0"))

    # --- Dataset ---
    train_repeat_factor: int = int(os.getenv("TRAIN_REPEAT_FACTOR", "4"))
    validation_ratio: float  = float(os.getenv("VAL_RATIO", "0.1"))
    log_interval: int        = int(os.getenv("LOG_INTERVAL", "10"))

    # --- Windows: num_workers=0 avoids spawn MemoryError on 16 GB RAM ---
    num_workers: int     = int(os.getenv("NUM_WORKERS", "0"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "2"))

    # --- Misc ---
    seed: int                = int(os.getenv("SEED", "7"))
    resume: bool             = os.getenv("RESUME", "1") == "1"
    use_compile: bool        = os.getenv("TORCH_COMPILE", "0") == "1"   # OFF — no Triton on Windows
    compile_mode: str        = os.getenv("TORCH_COMPILE_MODE", "reduce-overhead")
    empty_cache_on_oom: bool = os.getenv("EMPTY_CACHE_ON_OOM", "1") == "1"


class CachedTrajectoryDataset(Dataset[Dict[str, Tensor]]):
    """Cache dataset fully in RAM — fits comfortably in 16 GB for mini split."""

    def __init__(self, dataset: Dataset[Dict[str, Tensor]]) -> None:
        super().__init__()
        samples = [dataset[i] for i in range(len(dataset))]  # type: ignore[arg-type]
        if not samples:
            raise RuntimeError("CachedTrajectoryDataset received an empty dataset.")

        x            = torch.stack([s["x"].to(torch.float32) for s in samples])
        positions    = torch.stack([s["positions"].to(torch.float32) for s in samples])
        future       = torch.stack([s["future"].to(torch.float32) for s in samples])
        map_features = torch.stack([s["map"].to(torch.float32) for s in samples])

        valid = future.abs().sum(dim=(1, 2, 3)) > 0.0
        if not torch.any(valid):
            raise RuntimeError("No valid samples with non-zero future trajectories.")
        x = x[valid]; positions = positions[valid]
        future = future[valid]; map_features = map_features[valid]

        self.x            = x.contiguous()
        self.positions    = positions.contiguous()
        self.future       = future.contiguous()
        self.map_features = map_features.contiguous()

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "x":         self.x[index],
            "positions": self.positions[index],
            "future":    self.future[index],
            "map":       self.map_features[index],
        }


class TrajectoryPredictor(nn.Module):
    """Trajectory prediction stack scaled for 8 GB VRAM — embed_dim=256.

    Approximate parameter counts at embed_dim=256:
      InputEmbedding         ~1.5 M  (continuous_hidden=512, type_emb=512)
      TemporalTransformer    ~14 M   (16 layers, ff=1536)
      SocialTransformer       ~4 M   (6 layers, ff=1024)
      SceneContextEncoder     ~3 M   (4 layers, ff=1280)
      GoalPredictionNetwork   ~2 M   (hidden=1536, bottleneck=512)
      MultiModalDecoder        ~4 M  (8 layers, ff=512)
      ─────────────────────────────────────────────────────
      Total                  ~28 M
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding = InputEmbedding(
            continuous_dim=8,
            embedding_dim=EMBED,
            continuous_hidden_dim=512,
            type_embedding_dim=512,
            num_types=3,
            dropout=0.1,
        )
        self.temporal = TemporalTransformer(
            num_layers=16,
            num_heads=8,
            embed_dim=EMBED,
            ff_dim=1536,
            dropout=0.1,
        )
        self.social = SocialTransformer(
            num_layers=6,
            num_heads=8,
            embed_dim=EMBED,
            ff_dim=1024,
            dropout=0.1,
        )
        self.scene = SceneContextEncoder(
            num_layers=4,
            num_heads=8,
            embed_dim=EMBED,
            map_dim=256,
            ff_dim=1280,
            dropout=0.1,
        )
        self.goal = GoalPredictionNetwork(
            embed_dim=EMBED,
            num_goals=NUM_GOALS,
            hidden_dim=1536,
            bottleneck_dim=512,
            dropout=0.1,
        )
        self.decoder = MultiModalDecoder(
            num_layers=8,
            num_heads=8,
            embed_dim=EMBED,
            ff_dim=512,
            dropout=0.1,
            future_steps=FUTURE_STEPS,
        )

    def forward(
        self, x: Tensor, positions: Tensor, map_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        emb       = self.embedding(x)
        temp      = self.temporal(emb)
        soc       = self.social(temp, positions)
        scene_out = self.scene(soc, map_features, agent_positions=positions, map_positions=None)
        goals, probs = self.goal(scene_out)
        traj      = self.decoder(scene_out, goals, probs)
        return traj, goals, probs


# ── Utility functions ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.backends.cudnn.deterministic    = False


def supports_fused_adamw() -> bool:
    return "fused" in inspect.signature(torch.optim.AdamW).parameters


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def make_dataloader(
    dataset: Dataset[Dict[str, Tensor]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader[Dict[str, Tensor]]:
    kwargs: dict = {
        "batch_size":  batch_size,
        "shuffle":     shuffle,
        "num_workers": num_workers,
        "pin_memory":  pin_memory,
        "drop_last":   drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"]    = prefetch_factor
        # NOTE: Windows uses spawn — no fork context here
    return DataLoader(dataset, **kwargs)


def build_datasets(
    config: TrainConfig,
) -> tuple[Dataset[Dict[str, Tensor]], Dataset[Dict[str, Tensor]]]:
    raw     = NuScenesDataset(dataroot=config.dataset_root, version=config.version)
    limited = Subset(raw, range(min(config.dataset_limit, len(raw))))
    cached  = CachedTrajectoryDataset(limited)

    if len(cached) < 2:
        raise RuntimeError("Need at least 2 cached samples.")

    val_size   = max(1, int(round(len(cached) * config.validation_ratio)))
    train_size = len(cached) - val_size

    train_sub, val_sub = random_split(
        cached, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    if config.train_repeat_factor > 1:
        train_sub = ConcatDataset([train_sub] * config.train_repeat_factor)

    return train_sub, val_sub


def build_optimizer(
    model: nn.Module, config: TrainConfig, device: torch.device
) -> torch.optim.Optimizer:
    kwargs: dict = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if device.type == "cuda" and supports_fused_adamw():
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def create_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    global_step: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> Dict[str, object]:
    return {
        "epoch":         epoch,
        "global_step":   global_step,
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "scaler":        scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
    }


def save_checkpoint(checkpoint: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def extract_checkpoint_loss(checkpoint: Dict[str, object]) -> float:
    for key in ("val_metrics", "train_metrics"):
        m = checkpoint.get(key)
        if isinstance(m, dict) and "loss" in m:
            return float(m["loss"])
    v = checkpoint.get("avg_loss")
    return float(v) if v is not None else float("inf")


def load_model_state(model: nn.Module, checkpoint: Dict[str, object]) -> None:
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        return
    legacy = ("embedding", "temporal", "social", "scene", "goal", "decoder")
    if all(n in checkpoint for n in legacy):
        for n in legacy:
            getattr(model, n).load_state_dict(checkpoint[n])
        return
    raise KeyError("Checkpoint does not contain a supported model state.")


def update_best_checkpoints(
    checkpoint: Dict[str, object], checkpoint_dir: Path
) -> None:
    candidates = [checkpoint]
    for rank in range(1, 3):
        path = checkpoint_dir / f"best_{rank}.pt"
        if path.exists():
            candidates.append(torch.load(path, map_location="cpu"))
    candidates.sort(key=extract_checkpoint_loss)
    kept = candidates[:2]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for rank, item in enumerate(kept, start=1):
        save_checkpoint(item, checkpoint_dir / f"best_{rank}.pt")
    for rank in range(len(kept) + 1, 3):
        path = checkpoint_dir / f"best_{rank}.pt"
        if path.exists():
            path.unlink()


def load_resume_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: Optional[torch.amp.GradScaler],
    checkpoint_dir: Path,
    device: torch.device,
    config: TrainConfig,
) -> tuple[int, int, float]:
    for resume_path in (
        checkpoint_dir / "best_1.pt",
        checkpoint_dir / "best.pt",
        checkpoint_dir / "latest.pt",
    ):
        if not resume_path.exists():
            continue
        try:
            ckpt = torch.load(resume_path, map_location=device)
            load_model_state(model, ckpt)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                move_optimizer_state(optimizer, device)
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and scaler.is_enabled() and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])

            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            best_loss   = extract_checkpoint_loss(ckpt)

            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < config.min_learning_rate:
                for pg in optimizer.param_groups:
                    pg["lr"] = config.learning_rate
                print(f"  LR was zero — reset to {config.learning_rate:.2e}")
            else:
                print(f"  LR continuing at {current_lr:.2e}")

            print(f"Resumed from {resume_path}  epoch={start_epoch}  best_val={best_loss:.4f}")
            return start_epoch, global_step, best_loss
        except Exception as exc:
            print(f"Failed to resume from {resume_path}: {exc}")

    print("No checkpoint found — starting from scratch.")
    return 0, 0, float("inf")


def maybe_compile_model(
    model: nn.Module, config: TrainConfig, device: torch.device
) -> nn.Module:
    if not config.use_compile or device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    try:
        print(f"Compiling model (mode={config.compile_mode!r})...")
        compiled = torch.compile(model, mode=config.compile_mode)
        print("Compilation complete.")
        return compiled
    except Exception as exc:
        print(f"torch.compile failed — continuing without: {exc}")
        return model


def maybe_empty_cuda_cache(device: torch.device, reason: str) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"  cleared_cuda_cache  reason={reason}")


def train_one_epoch(
    model_forward: nn.Module,
    model: nn.Module,
    loader: DataLoader[Dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    amp_dtype: torch.dtype,
    config: TrainConfig,
    epoch: int,
    total_epochs: int,
) -> tuple[Dict[str, float], int]:
    model.train()
    running = {"loss": 0.0, "traj_loss": 0.0, "goal_loss": 0.0, "ade": 0.0, "fde": 0.0}
    steps = optimizer_steps = 0
    start_time   = time.time()
    use_autocast = device.type == "cuda"
    use_scaler   = scaler is not None and scaler.is_enabled()
    accum_steps  = max(1, config.grad_accum_steps)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        x            = batch["x"].to(device, non_blocking=True)
        positions    = batch["positions"].to(device, non_blocking=True)
        gt           = batch["future"].to(device, non_blocking=True)
        map_features = batch["map"].to(device, non_blocking=True)

        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                traj, goals, probs = model_forward(x, positions, map_features)
                traj_loss, ade, fde = best_of_k_loss(traj, gt, return_metrics=True)
                goal_loss  = goal_classification_loss(probs, goals, gt)
                loss       = traj_loss + config.goal_loss_weight * goal_loss
                scaled     = loss / accum_steps

            if not torch.isfinite(loss):
                print(f"  epoch={epoch+1}/{total_epochs} step={step} non-finite loss — skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

        except torch.OutOfMemoryError:
            print(f"  epoch={epoch+1}/{total_epochs} step={step} OOM — skipping batch")
            optimizer.zero_grad(set_to_none=True)
            if config.empty_cache_on_oom:
                maybe_empty_cuda_cache(device, "oom")
            continue

        running["loss"]      += float(loss.item())
        running["traj_loss"] += float(traj_loss.item())
        running["goal_loss"] += float(goal_loss.item())
        running["ade"]       += float(ade.item())
        running["fde"]       += float(fde.item())
        steps += 1

        if (step % accum_steps == 0) or (step == len(loader)):
            if use_scaler:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if step % config.log_interval == 0 or step == 1 or step == len(loader):
            elapsed = time.time() - start_time
            sps     = (step * x.size(0)) / max(elapsed, 1e-6)
            mem_gb  = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            print(
                f"  epoch={epoch+1}/{total_epochs} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} traj={traj_loss.item():.4f} goal={goal_loss.item():.4f} "
                f"ade={ade.item():.4f} fde={fde.item():.4f} "
                f"sps={sps:.1f} mem={mem_gb:.2f}GB opt_steps={optimizer_steps}"
            )

    if steps == 0:
        raise RuntimeError("All training batches were skipped.")
    return {k: v / steps for k, v in running.items()}, optimizer_steps


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Dict[str, Tensor]],
    device: torch.device,
    amp_dtype: torch.dtype,
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "traj_loss": 0.0, "goal_loss": 0.0, "ade": 0.0, "fde": 0.0}
    steps = 0

    for step, batch in enumerate(loader, start=1):
        x            = batch["x"].to(device, non_blocking=True)
        positions    = batch["positions"].to(device, non_blocking=True)
        gt           = batch["future"].to(device, non_blocking=True)
        map_features = batch["map"].to(device, non_blocking=True)

        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                traj, goals, probs = model(x, positions, map_features)
                traj_loss, ade, fde = best_of_k_loss(traj, gt, return_metrics=True)
                goal_loss = goal_classification_loss(probs, goals, gt)
                loss      = traj_loss + config.goal_loss_weight * goal_loss
        except torch.OutOfMemoryError:
            print(f"  eval step={step} OOM — skipping")
            if config.empty_cache_on_oom:
                maybe_empty_cuda_cache(device, "eval_oom")
            continue

        if not torch.isfinite(loss):
            continue

        running["loss"]      += float(loss.item())
        running["traj_loss"] += float(traj_loss.item())
        running["goal_loss"] += float(goal_loss.item())
        running["ade"]       += float(ade.item())
        running["fde"]       += float(fde.item())
        steps += 1

    if steps == 0:
        raise RuntimeError("Validation produced no finite batches.")
    return {k: v / steps for k, v in running.items()}


def train() -> None:
    """Train on nuScenes mini — Windows + 8 GB VRAM + embed_dim=256."""

    config = TrainConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(device)

    print("=" * 60)
    print("nuScenes mini  ·  Windows  ·  8 GB VRAM  ·  embed_dim=256")
    print("=" * 60)
    print(f"device={device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name}  vram={props.total_memory/1024**3:.1f}GB")
    print(f"dataset={config.version}  limit={config.dataset_limit}")
    print(f"batch={config.batch_size}  accum={config.grad_accum_steps}  effective={config.batch_size*config.grad_accum_steps}")
    print(f"num_workers={config.num_workers}  (0 = safe on Windows)")

    train_dataset, val_dataset = build_datasets(config)
    pin_memory = device.type == "cuda"

    train_loader = make_dataloader(
        train_dataset, config.batch_size, shuffle=True,
        num_workers=config.num_workers, prefetch_factor=config.prefetch_factor,
        pin_memory=pin_memory, drop_last=True,
    )
    val_loader = make_dataloader(
        val_dataset, config.eval_batch_size, shuffle=False,
        num_workers=0, prefetch_factor=config.prefetch_factor,
        pin_memory=pin_memory, drop_last=False,
    )

    model   = TrajectoryPredictor().to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\ntotal_params={total_p:,}  (~{total_p/1e6:.1f}M)  embed={EMBED}  goals={NUM_GOALS}")

    optimizer = build_optimizer(model, config, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=config.min_learning_rate,
    )

    # fp16 AMP + GradScaler for cards that may not support bf16
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler: Optional[torch.amp.GradScaler] = (
        torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        if device.type == "cuda" else None
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    start_epoch = global_step = 0
    best_val_loss = float("inf")

    if config.resume:
        start_epoch, global_step, best_val_loss = load_resume_checkpoint(
            model, optimizer, scheduler, scaler, checkpoint_dir, device, config,
        )

    compiled_model = maybe_compile_model(model, config, device)
    total_epochs   = start_epoch + config.run_epochs

    print(f"\ntrain_samples={len(train_dataset)}  val_samples={len(val_dataset)}")
    print(f"steps/epoch={len(train_loader)}  eval_steps={len(val_loader)}")
    print(f"lr={optimizer.param_groups[0]['lr']:.2e}  amp={amp_dtype}")
    print(f"training epochs {start_epoch+1}→{total_epochs}  checkpoints→{checkpoint_dir}\n")

    total_start = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics, step_inc = train_one_epoch(
            compiled_model, model, train_loader, optimizer, scaler,
            device, amp_dtype, config, epoch, total_epochs,
        )
        global_step += step_inc

        val_metrics = evaluate(model, val_loader, device, amp_dtype, config)
        scheduler.step(val_metrics["loss"])

        ckpt = create_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch=epoch + 1, global_step=global_step,
            train_metrics=train_metrics, val_metrics=val_metrics,
        )
        save_checkpoint(ckpt, checkpoint_dir / "latest.pt")
        update_best_checkpoints(ckpt, checkpoint_dir)
        best_val_loss = min(best_val_loss, val_metrics["loss"])

        epoch_time = time.time() - epoch_start
        peak_mem   = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        print(
            f"[epoch={epoch+1}/{total_epochs}]  "
            f"train loss={train_metrics['loss']:.4f}  ade={train_metrics['ade']:.4f}  |  "
            f"val loss={val_metrics['loss']:.4f}  ade={val_metrics['ade']:.4f}  fde={val_metrics['fde']:.4f}  |  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  time={epoch_time:.1f}s  "
            f"best={best_val_loss:.4f}  peak_mem={peak_mem:.2f}GB"
        )

    total_time = time.time() - total_start
    print(f"\ntraining_complete  minutes={total_time/60:.1f}")
    print(f"best_val_loss={best_val_loss:.4f}")
    print(f"checkpoints → {checkpoint_dir}/")


if __name__ == "__main__":
    train()