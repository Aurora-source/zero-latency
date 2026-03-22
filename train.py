"""High-throughput training entrypoint for the nuScenes mini trajectory pipeline."""

from __future__ import annotations

import inspect
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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

# ── Model dims (change here to scale up/down) ─────────────────────────────────
EMBED      = 640   # was 512 — more capacity, still fits in 4GB
NUM_GOALS  = 12
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration with safe defaults for RTX 2050 (4GB VRAM)."""

    dataset_root: str  = os.getenv("NUSCENES_ROOT", "data/raw/nuscenes")
    version: str       = "v1.0-mini"
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    dataset_limit: int = int(os.getenv("DATASET_LIMIT", "404"))
    run_epochs: int    = int(os.getenv("RUN_EPOCHS", "15"))        # was 30
    batch_size: int    = int(os.getenv("BATCH_SIZE", "2"))
    eval_batch_size: int = int(os.getenv("EVAL_BATCH_SIZE", "2"))
    grad_accum_steps: int = int(os.getenv("GRAD_ACCUM_STEPS", "8"))
    learning_rate: float  = float(os.getenv("LR", "1e-4"))
    min_learning_rate: float = float(os.getenv("MIN_LR", "1e-6"))
    weight_decay: float   = float(os.getenv("WEIGHT_DECAY", "1e-2"))
    goal_loss_weight: float = float(os.getenv("GOAL_LOSS_WEIGHT", "0.1"))
    gradient_clip_norm: float = float(os.getenv("GRAD_CLIP", "1.0"))
    train_repeat_factor: int = int(os.getenv("TRAIN_REPEAT_FACTOR", "4"))
    validation_ratio: float  = float(os.getenv("VAL_RATIO", "0.1"))
    log_interval: int  = int(os.getenv("LOG_INTERVAL", "5"))
    num_workers: int   = int(os.getenv("NUM_WORKERS", "2"))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "2"))
    warmup_ratio: float  = float(os.getenv("WARMUP_RATIO", "0.05"))
    seed: int          = int(os.getenv("SEED", "7"))
    resume: bool       = os.getenv("RESUME", "1") == "1"
    use_compile: bool  = os.getenv("TORCH_COMPILE", "0") == "1"
    compile_mode: str  = os.getenv("TORCH_COMPILE_MODE", "reduce-overhead")
    empty_cache_on_oom: bool = os.getenv("EMPTY_CACHE_ON_OOM", "1") == "1"
    augment: bool      = os.getenv("AUGMENT", "1") == "1"         # data augmentation


class CachedTrajectoryDataset(Dataset[Dict[str, Tensor]]):
    """Cache a small dataset fully in memory to avoid devkit I/O during training."""

    def __init__(self, dataset: Dataset[Dict[str, Tensor]], augment: bool = False) -> None:
        super().__init__()
        self.augment = augment
        samples = [dataset[index] for index in range(len(dataset))]
        if not samples:
            raise RuntimeError("CachedTrajectoryDataset received an empty dataset.")

        x            = torch.stack([s["x"].to(dtype=torch.float32) for s in samples])
        positions    = torch.stack([s["positions"].to(dtype=torch.float32) for s in samples])
        future       = torch.stack([s["future"].to(dtype=torch.float32) for s in samples])
        map_features = torch.stack([s["map"].to(dtype=torch.float32) for s in samples])

        valid_mask = future.abs().sum(dim=(1, 2, 3)) > 0.0
        if torch.any(valid_mask):
            x            = x[valid_mask]
            positions    = positions[valid_mask]
            future       = future[valid_mask]
            map_features = map_features[valid_mask]
        else:
            raise RuntimeError("No valid samples with non-zero future trajectories were found.")

        self.x            = x.contiguous()
        self.positions    = positions.contiguous()
        self.future       = future.contiguous()
        self.map_features = map_features.contiguous()

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        x         = self.x[index].clone()
        positions = self.positions[index].clone()
        future    = self.future[index].clone()

        if self.augment and torch.rand(1).item() > 0.5:
            # random horizontal flip — negate x coordinate and vx, ax
            x[..., 0] *= -1   # x position
            x[..., 2] *= -1   # vx velocity
            x[..., 4] *= -1   # ax acceleration
            positions[..., 0] *= -1
            future[..., 0]    *= -1

        if self.augment and torch.rand(1).item() > 0.5:
            # random gaussian noise on positions (±0.1m)
            x[..., :2]        += torch.randn_like(x[..., :2]) * 0.1
            positions         += torch.randn_like(positions) * 0.1

        return {
            "x":         x,
            "positions": positions,
            "future":    future,
            "map":       self.map_features[index],
        }


class TrajectoryPredictor(nn.Module):
    """Trajectory prediction stack — EMBED=640, num_goals=12 for RTX 2050."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = InputEmbedding(
            embedding_dim=EMBED,
            continuous_hidden_dim=1280,
        )
        self.temporal = TemporalTransformer(
            num_layers=8,
            embed_dim=EMBED,
            num_heads=8,
            ff_dim=1280,
        )
        self.social = SocialTransformer(
            num_layers=4,
            embed_dim=EMBED,
            num_heads=8,
            ff_dim=1280,
        )
        self.scene = SceneContextEncoder(
            num_layers=3,
            embed_dim=EMBED,
            num_heads=8,
            ff_dim=1280,
            map_dim=256,
        )
        self.goal = GoalPredictionNetwork(
            embed_dim=EMBED,
            hidden_dim=1280,
            bottleneck_dim=640,
            num_goals=NUM_GOALS,
        )
        self.decoder = MultiModalDecoder(
            num_layers=4,
            embed_dim=EMBED,
            num_heads=8,
            ff_dim=1280,
            future_steps=12,
        )

    def forward(self, x: Tensor, positions: Tensor, map_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        emb       = self.embedding(x)
        temp      = self.temporal(emb)
        soc       = self.social(temp, positions)
        scene_out = self.scene(soc, map_features, agent_positions=positions, map_positions=None)
        goals, goal_probabilities = self.goal(scene_out)
        trajectories = self.decoder(scene_out, goals, goal_probabilities)
        return trajectories, goals, goal_probabilities


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


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
    loader_kwargs: Dict = {
        "batch_size":  batch_size,
        "shuffle":     shuffle,
        "num_workers": num_workers,
        "pin_memory":  pin_memory,
        "drop_last":   drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"]    = prefetch_factor
        if os.name != "nt":
            loader_kwargs["multiprocessing_context"] = "fork"
    return DataLoader(dataset, **loader_kwargs)


def build_datasets(config: TrainConfig) -> tuple[Dataset[Dict[str, Tensor]], Dataset[Dict[str, Tensor]]]:
    raw_dataset     = NuScenesDataset(dataroot=config.dataset_root, version=config.version)
    limited_dataset = Subset(raw_dataset, range(min(config.dataset_limit, len(raw_dataset))))

    # train gets augmentation, val does not
    train_cache = CachedTrajectoryDataset(limited_dataset, augment=config.augment)
    val_cache   = CachedTrajectoryDataset(limited_dataset, augment=False)

    if len(train_cache) < 2:
        raise RuntimeError("Need at least two cached samples to build train/validation splits.")

    val_size   = max(1, int(round(len(train_cache) * config.validation_ratio)))
    train_size = len(train_cache) - val_size
    if train_size <= 0:
        train_size, val_size = len(train_cache) - 1, 1

    train_indices, val_indices = random_split(
        range(len(train_cache)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    from torch.utils.data import Subset as TSubset
    train_subset = TSubset(train_cache, list(train_indices))
    val_subset   = TSubset(val_cache,   list(val_indices))

    if config.train_repeat_factor > 1:
        repeated: Dataset[Dict[str, Tensor]] = ConcatDataset([train_subset] * config.train_repeat_factor)
    else:
        repeated = train_subset

    return repeated, val_subset


def build_optimizer(model: nn.Module, config: TrainConfig, device: torch.device) -> torch.optim.Optimizer:
    kwargs: Dict = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if device.type == "cuda" and supports_fused_adamw():
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""
    total_steps  = config.run_epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    base_lr      = config.learning_rate
    min_lr       = config.min_learning_rate

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
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
    avg = checkpoint.get("avg_loss")
    return float(avg) if avg is not None else float("inf")


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


def update_best_checkpoints(checkpoint: Dict[str, object], checkpoint_dir: Path) -> None:
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
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.amp.GradScaler],
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[int, int, float]:
    for resume_path in (checkpoint_dir / "best_1.pt", checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt"):
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
            print(f"Resumed from {resume_path} epoch={start_epoch} step={global_step} best_val={best_loss:.4f}")
            return start_epoch, global_step, best_loss
        except Exception as exc:
            print(f"Failed to resume from {resume_path}: {exc}")
    print("No compatible checkpoint found, starting from scratch.")
    return 0, 0, float("inf")


def maybe_compile_model(model: nn.Module, config: TrainConfig, device: torch.device) -> nn.Module:
    if not config.use_compile or device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    try:
        print(f"Compiling model with torch.compile(mode={config.compile_mode!r})...")
        compiled = torch.compile(model, mode=config.compile_mode)
        print("Model compilation complete.")
        return compiled
    except Exception as exc:
        print(f"torch.compile failed, continuing without: {exc}")
        return model


def maybe_empty_cuda_cache(device: torch.device, reason: str) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"cleared_cuda_cache reason={reason}")


def train_one_epoch(
    model_forward: nn.Module,
    model: nn.Module,
    loader: DataLoader[Dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
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
                goal_loss = goal_classification_loss(probs, goals, gt)
                loss      = traj_loss + config.goal_loss_weight * goal_loss
                scaled    = loss / accum_steps

            if not torch.isfinite(loss):
                print(f"epoch={epoch+1}/{total_epochs} step={step}/{len(loader)} invalid loss, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

        except torch.OutOfMemoryError:
            print(f"epoch={epoch+1}/{total_epochs} step={step}/{len(loader)} OOM, skipping")
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
            scheduler.step()
            optimizer_steps += 1

        if step % config.log_interval == 0 or step == 1 or step == len(loader):
            elapsed = time.time() - start_time
            sps     = (step * x.size(0)) / max(elapsed, 1e-6)
            mem_gb  = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            cur_lr  = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch+1}/{total_epochs} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} traj={traj_loss.item():.4f} goal={goal_loss.item():.4f} "
                f"ade={ade.item():.4f} fde={fde.item():.4f} "
                f"lr={cur_lr:.2e} sps={sps:.1f} mem={mem_gb:.2f}GB"
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
    steps        = 0
    use_autocast = device.type == "cuda"

    for batch in loader:
        x            = batch["x"].to(device, non_blocking=True)
        positions    = batch["positions"].to(device, non_blocking=True)
        gt           = batch["future"].to(device, non_blocking=True)
        map_features = batch["map"].to(device, non_blocking=True)

        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                traj, goals, probs = model(x, positions, map_features)
                traj_loss, ade, fde = best_of_k_loss(traj, gt, return_metrics=True)
                goal_loss = goal_classification_loss(probs, goals, gt)
                loss      = traj_loss + config.goal_loss_weight * goal_loss
        except torch.OutOfMemoryError:
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
    """Train the trajectory model — RTX 2050 safe, cosine LR, EMBED=640, augmentation."""

    config = TrainConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(device)

    print(f"device={device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name}  vram={props.total_memory/1024**3:.1f}GB")

    train_dataset, val_dataset = build_datasets(config)
    pin_memory   = device.type == "cuda"
    train_loader = make_dataloader(train_dataset, config.batch_size, True,  config.num_workers, config.prefetch_factor, pin_memory, True)
    val_loader   = make_dataloader(val_dataset,   config.eval_batch_size, False, max(0, config.num_workers//2), config.prefetch_factor, pin_memory, False)

    model = TrajectoryPredictor().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total_params={total_params:,} ({total_params/1e6:.1f}M)  embed={EMBED}  goals={NUM_GOALS}  augment={config.augment}")

    optimizer = build_optimizer(model, config, device)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader) // max(1, config.grad_accum_steps))

    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler: Optional[torch.amp.GradScaler] = (
        torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16)) if device.type == "cuda" else None
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    start_epoch = global_step = 0
    best_val_loss = float("inf")

    if config.resume:
        start_epoch, global_step, best_val_loss = load_resume_checkpoint(
            model, optimizer, scheduler, scaler, checkpoint_dir, device,
        )

    compiled_model = maybe_compile_model(model, config, device)

    total_epochs    = start_epoch + config.run_epochs
    effective_batch = config.batch_size * max(1, config.grad_accum_steps)
    print(f"train={len(train_dataset)}  val={len(val_dataset)}  steps/epoch={len(train_loader)}")
    print(f"micro_batch={config.batch_size}  effective_batch={effective_batch}  accum={config.grad_accum_steps}")
    print(f"lr={config.learning_rate:.2e}  min_lr={config.min_learning_rate:.2e}  warmup={config.warmup_ratio*100:.0f}%  amp={amp_dtype}")
    print(f"epochs={start_epoch+1}→{total_epochs}  checkpoints={checkpoint_dir}")

    total_start = time.time()
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics, step_inc = train_one_epoch(
            compiled_model, model, train_loader, optimizer, scheduler,
            scaler, device, amp_dtype, config, epoch, total_epochs,
        )
        global_step += step_inc

        val_metrics = evaluate(model, val_loader, device, amp_dtype, config)

        ckpt = create_checkpoint(model, optimizer, scheduler, scaler,
                                  epoch=epoch+1, global_step=global_step,
                                  train_metrics=train_metrics, val_metrics=val_metrics)
        save_checkpoint(ckpt, checkpoint_dir / "latest.pt")
        update_best_checkpoints(ckpt, checkpoint_dir)
        best_val_loss = min(best_val_loss, val_metrics["loss"])

        peak_mem = torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.0
        cur_lr   = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch+1}/{total_epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_ade={train_metrics['ade']:.4f} train_fde={train_metrics['fde']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_ade={val_metrics['ade']:.4f} val_fde={val_metrics['fde']:.4f} "
            f"lr={cur_lr:.2e} time={time.time()-epoch_start:.1f}s best_val={best_val_loss:.4f} peak_mem={peak_mem:.2f}GB"
        )

    print(f"training_complete={( time.time()-total_start)/60:.1f}min  best_val={best_val_loss:.4f}  saved={checkpoint_dir}")


if __name__ == "__main__":
    train()
