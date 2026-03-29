
"""High-throughput training entrypoint for the nuScenes mini trajectory pipeline."""

from __future__ import annotations

import inspect
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from utils.losses import AutoTunedBestOfKLoss, goal_classification_loss


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration with safer memory defaults for a high-end CUDA GPU."""

    dataset_root: str = os.getenv("NUSCENES_ROOT", "data/raw/nuscenes")
    version: str = "v1.0-mini"
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    dataset_limit: int = int(os.getenv("DATASET_LIMIT", "404"))       # full mini dataset
    run_epochs: int = int(os.getenv("RUN_EPOCHS", "40"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    eval_batch_size: int = int(os.getenv("EVAL_BATCH_SIZE", "8"))
    grad_accum_steps: int = int(os.getenv("GRAD_ACCUM_STEPS", "4"))   # effective batch = 64
    learning_rate: float = float(os.getenv("LR", "2e-4"))
    min_learning_rate: float = float(os.getenv("MIN_LR", "1e-5"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "1e-2"))
    goal_loss_weight: float = float(os.getenv("GOAL_LOSS_WEIGHT", "0.1"))
    gradient_clip_norm: float = float(os.getenv("GRAD_CLIP", "1.0"))
    train_repeat_factor: int = int(os.getenv("TRAIN_REPEAT_FACTOR", "4"))  # 404 x 4 = 1616
    validation_ratio: float = float(os.getenv("VAL_RATIO", "0.1"))
    log_interval: int = int(os.getenv("LOG_INTERVAL", "10"))
    num_workers: int = int(os.getenv("NUM_WORKERS", str(min(os.cpu_count() or 8, 12))))
    prefetch_factor: int = int(os.getenv("PREFETCH_FACTOR", "4"))
    seed: int = int(os.getenv("SEED", "7"))
    resume: bool = os.getenv("RESUME", "1") == "1"
    use_compile: bool = os.getenv("TORCH_COMPILE", "0") == "1"
    compile_mode: str = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
    empty_cache_on_oom: bool = os.getenv("EMPTY_CACHE_ON_OOM", "1") == "1"


class CachedTrajectoryDataset(Dataset[Dict[str, Tensor]]):
    """Cache a small dataset fully in memory to avoid devkit I/O during training."""

    def __init__(self, dataset: Dataset[Dict[str, Tensor]]) -> None:
        super().__init__()
        samples = [dataset[index] for index in range(len(dataset))]
        if not samples:
            raise RuntimeError("CachedTrajectoryDataset received an empty dataset.")

        x = torch.stack([sample["x"].to(dtype=torch.float32) for sample in samples], dim=0)
        positions = torch.stack(
            [sample["positions"].to(dtype=torch.float32) for sample in samples],
            dim=0,
        )
        future = torch.stack(
            [sample["future"].to(dtype=torch.float32) for sample in samples],
            dim=0,
        )
        map_features = torch.stack(
            [sample["map"].to(dtype=torch.float32) for sample in samples],
            dim=0,
        )

        valid_mask = future.abs().sum(dim=(1, 2, 3)) > 0.0
        if torch.any(valid_mask):
            x = x[valid_mask]
            positions = positions[valid_mask]
            future = future[valid_mask]
            map_features = map_features[valid_mask]
        else:
            raise RuntimeError("No valid samples with non-zero future trajectories were found.")

        self.x = x.contiguous()
        self.positions = positions.contiguous()
        self.future = future.contiguous()
        self.map_features = map_features.contiguous()

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "x": self.x[index],
            "positions": self.positions[index],
            "future": self.future[index],
            "map": self.map_features[index],
        }


class TrajectoryPredictor(nn.Module):
    """Wrapper around the full trajectory prediction stack."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = InputEmbedding()
        self.temporal = TemporalTransformer()
        self.social = SocialTransformer()
        self.scene = SceneContextEncoder(map_dim=256)
        self.goal = GoalPredictionNetwork()
        self.decoder = MultiModalDecoder(future_steps=12)

    def forward(self, x: Tensor, positions: Tensor, map_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        embeddings = self.embedding(x)
        temporal_context = self.temporal(embeddings)
        social_context = self.social(temporal_context, positions)
        scene_context = self.scene(
            social_context,
            map_features,
            agent_positions=positions,
            map_positions=None,
        )
        goals, goal_probabilities = self.goal(scene_context)
        trajectories = self.decoder(scene_context, goals, goal_probabilities)
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
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
        if os.name != "nt":
            loader_kwargs["multiprocessing_context"] = "fork"
    return DataLoader(dataset, **loader_kwargs)


def build_datasets(config: TrainConfig) -> tuple[Dataset[Dict[str, Tensor]], Dataset[Dict[str, Tensor]]]:
    raw_dataset = NuScenesDataset(
        dataroot=config.dataset_root,
        version=config.version,
    )
    
    actual_size = min(config.dataset_limit, len(raw_dataset))
    limited_dataset = Subset(raw_dataset, range(actual_size))
    
    #
    print(f"Loading {actual_size} samples completely into System RAM for max throughput...")
    cached_dataset = CachedTrajectoryDataset(limited_dataset)

    if len(cached_dataset) < 2:
        raise RuntimeError("Need at least two cached samples to build train/validation splits.")

   
    val_size = max(1, int(round(len(cached_dataset) * config.validation_ratio)))
    train_size = len(cached_dataset) - val_size

    train_subset, val_subset = random_split(
        cached_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    # 4. Apply repetition factor if needed
    repeated_train_dataset: Dataset[Dict[str, Tensor]]
    if config.train_repeat_factor > 1:
        repeated_train_dataset = ConcatDataset([train_subset] * config.train_repeat_factor)
    else:
        repeated_train_dataset = train_subset

    return repeated_train_dataset, val_subset


def build_optimizer(
    model: nn.Module, 
    criterion: nn.Module, 
    config: TrainConfig, 
    device: torch.device
) -> torch.optim.Optimizer:
    optimizer_kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if device.type == "cuda" and supports_fused_adamw():
        optimizer_kwargs["fused"] = True
        
    params = [
        {"params": model.parameters()},
        {"params": criterion.parameters()}
    ]
    return torch.optim.AdamW(params, **optimizer_kwargs)


def create_checkpoint(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    global_step: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> Dict[str, object]:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
    }


def save_checkpoint(checkpoint: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def extract_checkpoint_loss(checkpoint: Dict[str, object]) -> float:
    val_metrics = checkpoint.get("val_metrics")
    if isinstance(val_metrics, dict) and "loss" in val_metrics:
        return float(val_metrics["loss"])
    train_metrics = checkpoint.get("train_metrics")
    if isinstance(train_metrics, dict) and "loss" in train_metrics:
        return float(train_metrics["loss"])
    avg_loss = checkpoint.get("avg_loss")
    if avg_loss is not None:
        return float(avg_loss)
    return float("inf")


def load_model_state(model: nn.Module, checkpoint: Dict[str, object]) -> None:
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        return
    legacy_module_names = ("embedding", "temporal", "social", "scene", "goal", "decoder")
    if all(name in checkpoint for name in legacy_module_names):
        for name in legacy_module_names:
            getattr(model, name).load_state_dict(checkpoint[name])
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
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: Optional[torch.amp.GradScaler],
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[int, int, float]:
    candidate_paths = (
        checkpoint_dir / "best_1.pt",
        checkpoint_dir / "best.pt",
        checkpoint_dir / "latest.pt",
    )
    for resume_path in candidate_paths:
        if not resume_path.exists():
            continue
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 1. Load model weights safely (This is the critical part!)
            load_model_state(model, checkpoint)
            
            # 2. Load loss weights safely (strict=False ignores the missing Frenet parameters)
            if "criterion" in checkpoint:
                criterion.load_state_dict(checkpoint["criterion"], strict=False)
                
            # 3. Try to load optimizer, but DON'T crash if the parameters mismatch
            if "optimizer" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    move_optimizer_state(optimizer, device)
                except ValueError as e:
                    print(f"⚠️ Optimizer mismatch: {e}. Keeping pre-trained model weights but resetting optimizer.")
                    
            # 4. Safely load scheduler
            if "scheduler" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                except Exception:
                    pass
                    
            if scaler is not None and scaler.is_enabled() and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])
                
            start_epoch = int(checkpoint.get("epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            best_loss = extract_checkpoint_loss(checkpoint)
            
            print(
                f"Resumed from {resume_path} at epoch={start_epoch} "
                f"global_step={global_step} best_val_loss={best_loss:.4f}"
            )
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
        compiled_model = torch.compile(model, mode=config.compile_mode)
        print("Model compilation complete.")
        return compiled_model
    except Exception as exc:
        print(f"torch.compile failed, continuing without compilation: {exc}")
        return model


def maybe_empty_cuda_cache(device: torch.device, reason: str) -> None:
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    print(f"cleared_cuda_cache reason={reason}")


def train_one_epoch(
    model_forward: nn.Module,
    model: nn.Module,
    criterion: nn.Module,
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
    steps = 0
    optimizer_steps = 0
    start_time = time.time()

    use_autocast = device.type == "cuda"
    use_scaler = scaler is not None and scaler.is_enabled()
    accum_steps = max(1, config.grad_accum_steps)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        x            = batch["x"].to(device, non_blocking=True)
        positions    = batch["positions"].to(device, non_blocking=True)
        gt           = batch["future"].to(device, non_blocking=True)
        map_features = batch["map"].to(device, non_blocking=True)

        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                traj, goals, probs = model_forward(x, positions, map_features)
                traj_loss, ade, fde = criterion(traj, gt, return_metrics=True)
                goal_loss = goal_classification_loss(probs, goals, gt)
                loss = traj_loss + config.goal_loss_weight * goal_loss
                scaled_loss = loss / accum_steps

            if not torch.isfinite(loss):
                print(f"epoch={epoch + 1}/{total_epochs} step={step}/{len(loader)} invalid loss, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        except torch.OutOfMemoryError:
            print(f"epoch={epoch + 1}/{total_epochs} step={step}/{len(loader)} cuda_oom, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            if config.empty_cache_on_oom:
                maybe_empty_cuda_cache(device, reason="oom")
            continue

        running["loss"]      += float(loss.item())
        running["traj_loss"] += float(traj_loss.item())
        running["goal_loss"] += float(goal_loss.item())
        running["ade"]       += float(ade.item())
        running["fde"]       += float(fde.item())
        steps += 1

        should_step = (step % accum_steps == 0) or (step == len(loader))
        if should_step:
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
            samples_per_second = (step * x.size(0)) / max(elapsed, 1e-6)
            gpu_mem_gb = torch.cuda.max_memory_allocated(device=device) / 1024**3 if device.type == "cuda" else 0.0
            print(
                f"epoch={epoch + 1}/{total_epochs} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} traj={traj_loss.item():.4f} goal={goal_loss.item():.4f} "
                f"ade={ade.item():.4f} fde={fde.item():.4f} samples_per_sec={samples_per_second:.1f} "
                f"opt_steps={optimizer_steps} max_mem_gb={gpu_mem_gb:.2f}"
            )

    if steps == 0:
        raise RuntimeError("All training batches were skipped due to invalid losses or OOM.")

    metrics = {key: value / steps for key, value in running.items()}
    return metrics, optimizer_steps


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader[Dict[str, Tensor]],
    device: torch.device,
    amp_dtype: torch.dtype,
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    running = {"loss": 0.0, "traj_loss": 0.0, "goal_loss": 0.0, "ade": 0.0, "fde": 0.0}
    steps = 0
    use_autocast = device.type == "cuda"

    for step, batch in enumerate(loader, start=1):
        x            = batch["x"].to(device, non_blocking=True)
        positions    = batch["positions"].to(device, non_blocking=True)
        gt           = batch["future"].to(device, non_blocking=True)
        map_features = batch["map"].to(device, non_blocking=True)

        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                traj, goals, probs = model(x, positions, map_features)
                traj_loss, ade, fde = criterion(traj, gt, return_metrics=True)
                goal_loss = goal_classification_loss(probs, goals, gt)
                loss = traj_loss + config.goal_loss_weight * goal_loss
        except torch.OutOfMemoryError:
            print(f"eval_step={step}/{len(loader)} cuda_oom, skipping batch")
            if config.empty_cache_on_oom:
                maybe_empty_cuda_cache(device, reason="eval_oom")
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

    return {key: value / steps for key, value in running.items()}


def train() -> None:
    """Train the trajectory model with strong defaults for RTX 5090 on Linux."""

    config = TrainConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(device)

    print(f"device={device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name}")
        print(f"vram_gb={props.total_memory / 1024**3:.1f}")
        print(f"allocator_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}")

    train_dataset, val_dataset = build_datasets(config)
    pin_memory = device.type == "cuda"
    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=max(0, config.num_workers // 2),
        prefetch_factor=config.prefetch_factor,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = TrajectoryPredictor().to(device)
    criterion = AutoTunedBestOfKLoss().to(device)
    
    optimizer = build_optimizer(model, criterion, config, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=config.min_learning_rate,
    )

    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler: Optional[torch.amp.GradScaler]
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    else:
        scaler = None

    checkpoint_dir = Path(config.checkpoint_dir)
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if config.resume:
        start_epoch, global_step, best_val_loss = load_resume_checkpoint(
            model, criterion, optimizer, scheduler, scaler, checkpoint_dir, device,
        )

    compiled_model = maybe_compile_model(model, config, device)

    total_epochs = start_epoch + config.run_epochs
    effective_batch = config.batch_size * max(1, config.grad_accum_steps)
    print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    print(f"steps_per_epoch={len(train_loader)} eval_steps={len(val_loader)}")
    print(
        f"micro_batch={config.batch_size} effective_batch={effective_batch} "
        f"eval_batch={config.eval_batch_size} grad_accum={config.grad_accum_steps}"
    )
    print(f"starting_lr={optimizer.param_groups[0]['lr']:.6f} amp_dtype={amp_dtype}")
    print(f"training from epoch {start_epoch + 1} → {total_epochs}")
    print(f"checkpoint_dir={checkpoint_dir}")

    total_start = time.time()
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics, step_increment = train_one_epoch(
            compiled_model, model, criterion, train_loader, optimizer, scaler,
            device, amp_dtype, config, epoch, total_epochs,
        )
        global_step += step_increment

        val_metrics = evaluate(model, criterion, val_loader, device, amp_dtype, config)
        scheduler.step(val_metrics["loss"])

        checkpoint = create_checkpoint(
            model, criterion, optimizer, scheduler, scaler,
            epoch=epoch + 1,
            global_step=global_step,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        save_checkpoint(checkpoint, checkpoint_dir / "latest.pt")
        update_best_checkpoints(checkpoint, checkpoint_dir)
        best_val_loss = min(best_val_loss, val_metrics["loss"])

        epoch_time = time.time() - epoch_start
        peak_mem_gb = torch.cuda.max_memory_allocated(device=device) / 1024**3 if device.type == "cuda" else 0.0
        print(
            f"epoch={epoch + 1}/{total_epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_ade={train_metrics['ade']:.4f} train_fde={train_metrics['fde']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_ade={val_metrics['ade']:.4f} val_fde={val_metrics['fde']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} time={epoch_time:.1f}s best_val={best_val_loss:.4f} peak_mem_gb={peak_mem_gb:.2f}"
        )

    total_time = time.time() - total_start
    print(f"training_complete_minutes={total_time / 60.0:.1f}")
    print(f"best_val_loss={best_val_loss:.4f}")
    print(f"checkpoints_saved_to={checkpoint_dir}")


if __name__ == "__main__":
    train()
