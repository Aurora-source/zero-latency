from __future__ import annotations

import inspect
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import warnings
warnings.filterwarnings("ignore", message="expandable_segments not supported")

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
from utils.losses import best_of_k_loss, goal_classification_loss


@dataclass(frozen=True)
class TrainConfig:

    dataset_root: str   = os.getenv("NUSCENES_ROOT", "nuscenes")
    version: str        = os.getenv("NUSCENES_VERSION", "v1.0-trainval")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    dataset_limit: int  = int(os.getenv("DATASET_LIMIT", "6000"))
    run_epochs: int     = int(os.getenv("RUN_EPOCHS", "40"))

    batch_size: int        = int(os.getenv("BATCH_SIZE", "128"))
    eval_batch_size: int   = int(os.getenv("EVAL_BATCH_SIZE", "64"))
    grad_accum_steps: int  = int(os.getenv("GRAD_ACCUM_STEPS", "1"))

    learning_rate: float   = float(os.getenv("LR", "5e-5"))
    min_learning_rate: float = float(os.getenv("MIN_LR", "1e-6"))
    weight_decay: float    = float(os.getenv("WEIGHT_DECAY", "1e-2"))
    goal_loss_weight: float = float(os.getenv("GOAL_LOSS_WEIGHT", "0.1"))
    gradient_clip_norm: float = float(os.getenv("GRAD_CLIP", "1.0"))

    train_repeat_factor: int = int(os.getenv("TRAIN_REPEAT_FACTOR", "1"))
    validation_ratio: float  = float(os.getenv("VAL_RATIO", "0.1"))
    log_interval: int        = int(os.getenv("LOG_INTERVAL", "10"))
    num_workers: int         = int(os.getenv("NUM_WORKERS", "32"))
    prefetch_factor: int     = int(os.getenv("PREFETCH_FACTOR", "6"))

    seed: int            = int(os.getenv("SEED", "7"))
    resume: bool         = os.getenv("RESUME", "1") == "1"
    use_compile: bool    = os.getenv("TORCH_COMPILE", "1") == "1"
    compile_mode: str    = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
    empty_cache_on_oom: bool = os.getenv("EMPTY_CACHE_ON_OOM", "1") == "1"
    pin_memory_device: str   = os.getenv("PIN_MEMORY_DEVICE", "")


class CachedTrajectoryDataset(Dataset[Dict[str, Tensor]]):

    def __init__(self, dataset: Dataset[Dict[str, Tensor]]) -> None:
        super().__init__()
        print(f"  Caching {len(dataset)} samples into RAM...")
        samples = [dataset[i] for i in range(len(dataset))]
        if not samples:
            raise RuntimeError("CachedTrajectoryDataset received an empty dataset.")

        x            = torch.stack([s["x"].to(torch.float32) for s in samples])
        positions    = torch.stack([s["positions"].to(torch.float32) for s in samples])
        future       = torch.stack([s["future"].to(torch.float32) for s in samples])
        map_features = torch.stack([s["map"].to(torch.float32) for s in samples])

        valid_mask = future.abs().sum(dim=(1, 2, 3)) > 0.0
        if torch.any(valid_mask):
            x = x[valid_mask]; positions = positions[valid_mask]
            future = future[valid_mask]; map_features = map_features[valid_mask]
        else:
            raise RuntimeError("No valid samples with non-zero future trajectories.")

        self.x            = x.contiguous()
        self.positions    = positions.contiguous()
        self.future       = future.contiguous()
        self.map_features = map_features.contiguous()
        print(f"  Cached {len(self.x)} valid samples.")

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

    def __init__(self) -> None:
        super().__init__()
        self.embedding = InputEmbedding()
        self.temporal  = TemporalTransformer()
        self.social    = SocialTransformer()
        self.scene     = SceneContextEncoder(map_dim=256)
        self.goal      = GoalPredictionNetwork()
        self.decoder   = MultiModalDecoder(future_steps=12)

    def forward(self, x: Tensor, positions: Tensor, map_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        emb          = self.embedding(x)
        temp         = self.temporal(emb)
        soc          = self.social(temp, positions)
        scene_ctx    = self.scene(soc, map_features, agent_positions=positions, map_positions=None)
        goals, probs = self.goal(scene_ctx)
        traj         = self.decoder(scene_ctx, goals, probs)
        return traj, goals, probs

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(32)
    torch.set_num_interop_threads(8)

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


def build_datasets(config: TrainConfig):
    print(f"Loading {config.version} dataset from {config.dataset_root!r}")

    raw_dataset = NuScenesDataset(
        dataroot=config.dataset_root,
        version=config.version,
    )

    print(f"Total samples: {len(raw_dataset)}")

    limited = Subset(raw_dataset, range(min(config.dataset_limit, len(raw_dataset))))
    cached = CachedTrajectoryDataset(limited)

    if len(cached) < 2:
        raise RuntimeError("too few samples")

    val_size = max(1, int(round(len(cached) * config.validation_ratio)))
    train_size = len(cached) - val_size

    train_subset, val_subset = random_split(
        cached,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    if config.train_repeat_factor > 1:
        train_subset = ConcatDataset([train_subset] * config.train_repeat_factor)

    return train_subset, val_subset


def build_optimizer(model: nn.Module, config: TrainConfig, device: torch.device):
    kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }

    if device.type == "cuda" and supports_fused_adamw():
        kwargs["fused"] = True

    return torch.optim.AdamW(model.parameters(), **kwargs)


def create_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    train_metrics,
    val_metrics,
):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
    }


def save_checkpoint(checkpoint, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def extract_checkpoint_loss(checkpoint):
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


def load_model_state(model, checkpoint):
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        return

    legacy = ("embedding", "temporal", "social", "scene", "goal", "decoder")

    if all(name in checkpoint for name in legacy):
        for name in legacy:
            getattr(model, name).load_state_dict(checkpoint[name])
        return

    raise KeyError("invalid checkpoint")


def update_best_checkpoints(checkpoint, checkpoint_dir: Path):
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
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint_dir: Path,
    device,
    config: TrainConfig,
):
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

            load_model_state(model, checkpoint)

            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                move_optimizer_state(optimizer, device)

            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])

            if scaler is not None and scaler.is_enabled() and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])

            start_epoch = int(checkpoint.get("epoch", 0))
            global_step = int(checkpoint.get("global_step", 0))
            best_loss = extract_checkpoint_loss(checkpoint)

            current_lr = optimizer.param_groups[0]["lr"]

            if current_lr < config.min_learning_rate:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.learning_rate
                print(f"LR reset to {config.learning_rate:.2e}")
            else:
                print(f"LR={current_lr:.2e}")

            return start_epoch, global_step, best_loss

        except Exception as exc:
            print(f"resume failed: {exc}")

    print("no checkpoint, starting fresh")
    return 0, 0, float("inf")


def maybe_compile_model(model, config: TrainConfig, device):
    if not config.use_compile or device.type != "cuda" or not hasattr(torch, "compile"):
        return model

    try:
        print(f"compile mode={config.compile_mode}")
        compiled = torch.compile(model, mode=config.compile_mode)
        print("compiled")
        return compiled
    except Exception as exc:
        print(f"compile failed: {exc}")
        return model


def maybe_empty_cuda_cache(device, reason: str):
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    print(f"cache cleared ({reason})")


def train_one_epoch(
    model_forward,
    model,
    loader,
    optimizer,
    scaler,
    device,
    amp_dtype,
    config: TrainConfig,
    epoch,
    total_epochs,
):
    model.train()

    running = {
        "loss": 0.0,
        "traj_loss": 0.0,
        "goal_loss": 0.0,
        "ade": 0.0,
        "fde": 0.0,
    }

    steps = 0
    optimizer_steps = 0
    start_time = time.time()

    use_autocast = device.type == "cuda"
    use_scaler = scaler is not None and scaler.is_enabled()
    accum_steps = max(1, config.grad_accum_steps)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):

        x = batch["x"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        gt = batch["future"].to(device, non_blocking=True)
        map_features = batch["map"].to(device, non_blocking=True)

        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):

                traj, goals, probs = model_forward(x, positions, map_features)

                traj_loss, ade, fde = best_of_k_loss(traj, gt, return_metrics=True)
                goal_loss = goal_classification_loss(probs, goals, gt)

                loss = traj_loss + config.goal_loss_weight * goal_loss
                scaled_loss = loss / accum_steps

            if not torch.isfinite(loss):
                print(f"epoch={epoch+1} step={step} bad loss")
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        except torch.OutOfMemoryError:
            print(f"epoch={epoch+1} step={step} OOM")
            optimizer.zero_grad(set_to_none=True)
            if config.empty_cache_on_oom:
                maybe_empty_cuda_cache(device, "oom")
            continue

        running["loss"] += float(loss.item())
        running["traj_loss"] += float(traj_loss.item())
        running["goal_loss"] += float(goal_loss.item())
        running["ade"] += float(ade.item())
        running["fde"] += float(fde.item())
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
            sps = (step * x.size(0)) / max(elapsed, 1e-6)

            mem_gb = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            )

            print(
                f"epoch={epoch+1}/{total_epochs} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} ade={ade.item():.4f} fde={fde.item():.4f} "
                f"sps={sps:.1f} mem={mem_gb:.2f}GB"
            )

    if steps == 0:
        raise RuntimeError("no training batches")

    return {k: v / steps for k, v in running.items()}, optimizer_steps


@torch.inference_mode()
def evaluate(model, loader, device, amp_dtype, config: TrainConfig):

    model.eval()

    running = {
        "loss": 0.0,
        "traj_loss": 0.0,
        "goal_loss": 0.0,
        "ade": 0.0,
        "fde": 0.0,
    }

    steps = 0
    use_autocast = device.type == "cuda"

    for step, batch in enumerate(loader, start=1):

        x = batch["x"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        gt = batch["future"].to(device, non_blocking=True)
        map_features = batch["map"].to(device, non_blocking=True)

        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):

                traj, goals, probs = model(x, positions, map_features)

                traj_loss, ade, fde = best_of_k_loss(traj, gt, return_metrics=True)
                goal_loss = goal_classification_loss(probs, goals, gt)

                loss = traj_loss + config.goal_loss_weight * goal_loss

        except torch.OutOfMemoryError:
            print(f"eval step={step} OOM")
            if config.empty_cache_on_oom:
                maybe_empty_cuda_cache(device, "eval_oom")
            continue

        if not torch.isfinite(loss):
            continue

        running["loss"] += float(loss.item())
        running["traj_loss"] += float(traj_loss.item())
        running["goal_loss"] += float(goal_loss.item())
        running["ade"] += float(ade.item())
        running["fde"] += float(fde.item())
        steps += 1

    if steps == 0:
        raise RuntimeError("no valid eval batches")

    return {k: v / steps for k, v in running.items()}


def train():

    config = TrainConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(device)

    print("=" * 50)
    print("training start")
    print("=" * 50)

    print(f"device={device}")

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
    optimizer = build_optimizer(model, config, device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=config.min_learning_rate,
    )

    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    scaler = (
        torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        if device.type == "cuda"
        else None
    )

    checkpoint_dir = Path(config.checkpoint_dir)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if config.resume:
        start_epoch, global_step, best_val_loss = load_resume_checkpoint(
            model, optimizer, scheduler, scaler, checkpoint_dir, device, config
        )

    compiled_model = maybe_compile_model(model, config, device)

    total_epochs = start_epoch + config.run_epochs

    for epoch in range(start_epoch, total_epochs):

        train_metrics, step_increment = train_one_epoch(
            compiled_model,
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_dtype,
            config,
            epoch,
            total_epochs,
        )

        global_step += step_increment

        val_metrics = evaluate(model, val_loader, device, amp_dtype, config)

        scheduler.step(val_metrics["loss"])

        checkpoint = create_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch + 1,
            global_step=global_step,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        save_checkpoint(checkpoint, checkpoint_dir / "latest.pt")
        update_best_checkpoints(checkpoint, checkpoint_dir)

        best_val_loss = min(best_val_loss, val_metrics["loss"])

        print(
            f"[epoch={epoch+1}/{total_epochs}] "
            f"train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} "
            f"best={best_val_loss:.4f}"
        )

    print("\ntraining done")
    print(f"best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    train()