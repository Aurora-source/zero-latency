"""Minimal nuScenes mini dataset loader for trajectory prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from nuscenes import NuScenes
    from nuscenes.eval.prediction.splits import get_prediction_challenge_split
    from nuscenes.prediction import PredictHelper
except ImportError as exc:  # pragma: no cover - import is validated at runtime.
    NuScenes = None  # type: ignore[assignment]
    PredictHelper = None  # type: ignore[assignment]
    get_prediction_challenge_split = None  # type: ignore[assignment]
    _NUSCENES_IMPORT_ERROR = exc
else:
    _NUSCENES_IMPORT_ERROR = None

from pyquaternion import Quaternion

__all__ = ["NuScenesDataset"]


@dataclass(frozen=True)
class SceneEntry:
    """Index entry for a single training sample."""

    sample_token: str
    agent_annotation_tokens: Sequence[str]


class NuScenesDataset(Dataset[Dict[str, Tensor]]):
    """Minimal scene-level nuScenes mini dataset for trajectory prediction.

    Each dataset item corresponds to one nuScenes sample token from the prediction
    split. Up to ``max_agents`` annotated agents with sufficient history and future
    context are selected, sorted by current distance to the ego vehicle, and padded
    with zeros when fewer agents are available.

    Returned tensors use a fixed shape contract:

    - ``x``: ``(past_steps, max_agents, 8)`` with features
      ``[x, y, vx, vy, ax, ay, heading, type]``.
    - ``positions``: ``(past_steps, max_agents, 2)``.
    - ``future``: ``(max_agents, future_steps, 2)``.
    - ``map``: dummy zero tensor for downstream compatibility.

    Coordinates are normalized into the current ego frame of the sample.
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "mini_train",
        past_steps: int = 6,
        future_steps: int = 12,
        max_agents: int = 10,
        dt: float = 0.5,
        map_dim: int = 256,
        dummy_map_elements: int = 1,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if _NUSCENES_IMPORT_ERROR is not None:
            raise ImportError(
                "nuscenes-devkit is required to use NuScenesDataset. "
                "Install it with `pip install nuscenes-devkit`."
            ) from _NUSCENES_IMPORT_ERROR
        if past_steps < 2:
            raise ValueError("past_steps must be at least 2.")
        if future_steps <= 0:
            raise ValueError("future_steps must be positive.")
        if max_agents <= 0:
            raise ValueError("max_agents must be positive.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if map_dim <= 0 or dummy_map_elements <= 0:
            raise ValueError("Dummy map dimensions must be positive.")

        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.max_agents = max_agents
        self.dt = dt
        self.map_dim = map_dim
        self.dummy_map_elements = dummy_map_elements

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.helper = PredictHelper(self.nusc)
        self.dummy_map = torch.zeros(dummy_map_elements, map_dim, dtype=torch.float32)
        self.entries = self._build_index(max_samples=max_samples)

        if not self.entries:
            raise RuntimeError(
                "No valid samples were found. Check that the nuScenes mini dataset "
                "exists at the provided dataroot and contains agents with enough "
                "past/future context."
            )

    def __len__(self) -> int:
        """Return the number of indexed scene samples."""

        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Load one scene sample and return padded agent tensors."""

        entry = self.entries[index]
        sample = self.nusc.get("sample", entry.sample_token)
        ego_xy, ego_yaw = self._get_ego_pose(sample)

        x = torch.zeros(self.past_steps, self.max_agents, 8, dtype=torch.float32)
        positions = torch.zeros(self.past_steps, self.max_agents, 2, dtype=torch.float32)
        future = torch.zeros(self.max_agents, self.future_steps, 2, dtype=torch.float32)

        for agent_index, annotation_token in enumerate(entry.agent_annotation_tokens):
            annotation = self.nusc.get("sample_annotation", annotation_token)
            past_records = self._collect_past_records(annotation)
            future_records = self._collect_future_records(annotation)

            past_xy_global = self._records_to_xy(past_records)
            future_xy_global = self._records_to_xy(future_records)

            past_xy_local = self._global_to_local(past_xy_global, ego_xy, ego_yaw)
            future_xy_local = self._global_to_local(future_xy_global, ego_xy, ego_yaw)
            headings = self._records_to_heading(past_records, ego_yaw)
            velocities = self._differentiate(past_xy_local)
            accelerations = self._differentiate(velocities)

            agent_type = self._category_to_type(annotation["category_name"])
            type_column = torch.full(
                (self.past_steps, 1),
                float(agent_type),
                dtype=torch.float32,
            )

            features = torch.cat(
                (past_xy_local, velocities, accelerations, headings.unsqueeze(-1), type_column),
                dim=-1,
            )

            x[:, agent_index] = features
            positions[:, agent_index] = past_xy_local
            future[agent_index] = future_xy_local

        return {
            "x": x,
            "positions": positions,
            "future": future,
            "map": self.dummy_map.clone(),
        }

    def _build_index(self, max_samples: Optional[int]) -> List[SceneEntry]:
        """Index scene samples with at least one valid agent trajectory."""

        split_tokens = get_prediction_challenge_split(self.split, dataroot=self.dataroot)
        sample_tokens: List[str] = []
        seen_sample_tokens = set()

        for token_pair in split_tokens:
            _, sample_token = token_pair.split("_", maxsplit=1)
            if sample_token in seen_sample_tokens:
                continue
            seen_sample_tokens.add(sample_token)
            sample_tokens.append(sample_token)

        entries: List[SceneEntry] = []
        for sample_token in sample_tokens:
            sample = self.nusc.get("sample", sample_token)
            ego_xy, _ = self._get_ego_pose(sample)
            annotations = self.helper.get_annotations_for_sample(sample_token)

            eligible_annotations = []
            for annotation in annotations:
                category_name = annotation["category_name"]
                if self._category_to_type(category_name, raise_on_unknown=False) is None:
                    continue
                if not self._has_required_context(annotation):
                    continue

                translation = torch.tensor(annotation["translation"][:2], dtype=torch.float32)
                distance = torch.linalg.vector_norm(translation - ego_xy).item()
                eligible_annotations.append((distance, annotation["token"]))

            if not eligible_annotations:
                continue

            eligible_annotations.sort(key=lambda item: item[0])
            selected_tokens = [token for _, token in eligible_annotations[: self.max_agents]]
            entries.append(
                SceneEntry(
                    sample_token=sample_token,
                    agent_annotation_tokens=tuple(selected_tokens),
                )
            )

            if max_samples is not None and len(entries) >= max_samples:
                break

        return entries

    def _has_required_context(self, annotation: Dict[str, object]) -> bool:
        """Check whether an annotation has enough past and future steps."""

        prev_token = annotation["prev"]
        for _ in range(self.past_steps - 1):
            if not prev_token:
                return False
            prev_annotation = self.nusc.get("sample_annotation", prev_token)
            prev_token = prev_annotation["prev"]

        next_token = annotation["next"]
        for _ in range(self.future_steps):
            if not next_token:
                return False
            next_annotation = self.nusc.get("sample_annotation", next_token)
            next_token = next_annotation["next"]

        return True

    def _collect_past_records(self, annotation: Dict[str, object]) -> List[Dict[str, object]]:
        """Collect ``past_steps`` records ending at the current annotation."""

        records = [annotation]
        prev_token = annotation["prev"]
        for _ in range(self.past_steps - 1):
            if not prev_token:
                raise RuntimeError("Annotation is missing required past context.")
            prev_annotation = self.nusc.get("sample_annotation", prev_token)
            records.append(prev_annotation)
            prev_token = prev_annotation["prev"]
        records.reverse()
        return records

    def _collect_future_records(self, annotation: Dict[str, object]) -> List[Dict[str, object]]:
        """Collect the next ``future_steps`` records after the current annotation."""

        records: List[Dict[str, object]] = []
        next_token = annotation["next"]
        for _ in range(self.future_steps):
            if not next_token:
                raise RuntimeError("Annotation is missing required future context.")
            next_annotation = self.nusc.get("sample_annotation", next_token)
            records.append(next_annotation)
            next_token = next_annotation["next"]
        return records

    def _get_ego_pose(self, sample: Dict[str, object]) -> tuple[Tensor, float]:
        """Return the current ego position and yaw from the LIDAR_TOP pose."""

        lidar_token = sample["data"]["LIDAR_TOP"]
        sample_data = self.nusc.get("sample_data", lidar_token)
        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

        ego_xy = torch.tensor(ego_pose["translation"][:2], dtype=torch.float32)
        ego_yaw = self._quaternion_to_yaw(ego_pose["rotation"])
        return ego_xy, ego_yaw

    @staticmethod
    def _records_to_xy(records: Sequence[Dict[str, object]]) -> Tensor:
        """Convert annotation records to an ``(T, 2)`` tensor of xy positions."""

        return torch.tensor(
            [record["translation"][:2] for record in records],
            dtype=torch.float32,
        )

    def _records_to_heading(self, records: Sequence[Dict[str, object]], ego_yaw: float) -> Tensor:
        """Convert annotation rotations to local-frame heading angles."""

        headings = [self._wrap_angle(self._quaternion_to_yaw(record["rotation"]) - ego_yaw) for record in records]
        return torch.tensor(headings, dtype=torch.float32)

    @staticmethod
    def _quaternion_to_yaw(rotation: Sequence[float]) -> float:
        """Extract yaw from a quaternion rotation record."""

        return float(Quaternion(rotation).yaw_pitch_roll[0])

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to ``[-pi, pi]``."""

        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _global_to_local(points_xy: Tensor, ego_xy: Tensor, ego_yaw: float) -> Tensor:
        """Transform global xy coordinates into the current ego frame."""

        relative = points_xy - ego_xy.unsqueeze(0)
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)
        rotation = torch.tensor(
            [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]],
            dtype=points_xy.dtype,
        )
        return relative @ rotation.T

    def _differentiate(self, values: Tensor) -> Tensor:
        """Estimate first-order derivatives with finite differences."""

        derivative = torch.zeros_like(values)
        derivative[0] = (values[1] - values[0]) / self.dt
        derivative[-1] = (values[-1] - values[-2]) / self.dt
        if values.size(0) > 2:
            derivative[1:-1] = (values[2:] - values[:-2]) / (2.0 * self.dt)
        return derivative

    @staticmethod
    def _category_to_type(
        category_name: str,
        raise_on_unknown: bool = True,
    ) -> Optional[int]:
        """Map nuScenes category names to coarse trajectory agent types."""

        if category_name.startswith("human.pedestrian"):
            return 1
        if category_name in {"vehicle.bicycle", "vehicle.motorcycle"}:
            return 2
        if category_name.startswith("vehicle."):
            return 0
        if raise_on_unknown:
            raise ValueError(f"Unsupported category for trajectory prediction: {category_name}")
        return None


if __name__ == "__main__":
    dataset = NuScenesDataset(
        dataroot="data/nuscenes",
        split="mini_train",
        max_samples=2,
        verbose=True,
    )
    sample = dataset[0]
    print({key: tuple(value.shape) for key, value in sample.items()})
