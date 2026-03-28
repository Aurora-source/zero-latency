from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from nuscenes import NuScenes
    from nuscenes.prediction import PredictHelper
except ImportError as exc:
    NuScenes = None
    PredictHelper = None
    _NUSCENES_IMPORT_ERROR = exc
else:
    _NUSCENES_IMPORT_ERROR = None

from pyquaternion import Quaternion

__all__ = ["NuScenesDataset"]


class NuScenesDataset(Dataset[Dict[str, Tensor]]):

    SUPPORTED_VERSIONS = {"v1.0-mini", "v1.0-trainval", "v1.0-test"}

    def __init__(
        self,
        dataroot: str = "data/raw/nuscenes",
        version: str = "v1.0-mini",
        past_steps: int = 6,
        future_steps: int = 12,
        max_agents: int = 10,
        dt: float = 0.5,
        map_dim: int = 256,
        dummy_map_elements: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if _NUSCENES_IMPORT_ERROR is not None:
            raise ImportError("nuscenes-devkit required") from _NUSCENES_IMPORT_ERROR

        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"bad version: {version}")
        if past_steps < 2:
            raise ValueError("past_steps >= 2")
        if future_steps <= 0:
            raise ValueError("future_steps > 0")
        if max_agents <= 0:
            raise ValueError("max_agents > 0")
        if dt <= 0.0:
            raise ValueError("dt > 0")
        if map_dim <= 0 or dummy_map_elements <= 0:
            raise ValueError("map dims invalid")

        self.dataroot = dataroot
        self.version = version
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.max_agents = max_agents
        self.dt = dt
        self.map_dim = map_dim
        self.dummy_map_elements = dummy_map_elements

        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.dataroot,
            verbose=verbose,
        )

        self.helper = PredictHelper(self.nusc)
        self.dummy_map = torch.zeros(dummy_map_elements, map_dim, dtype=torch.float32)
        self.entries = self._build_index()

        if not self.entries:
            raise RuntimeError("no samples found")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:

        sample_token = self.entries[index]
        sample = self.nusc.get("sample", sample_token)

        ego_xy, ego_yaw = self._get_ego_pose(sample)
        agent_tokens = self._select_agent_tokens(sample_token, ego_xy)

        x = torch.zeros(self.past_steps, self.max_agents, 8, dtype=torch.float32)
        positions = torch.zeros(self.past_steps, self.max_agents, 2, dtype=torch.float32)
        future = torch.zeros(self.max_agents, self.future_steps, 2, dtype=torch.float32)

        for agent_index, annotation_token in enumerate(agent_tokens):

            ann = self.nusc.get("sample_annotation", annotation_token)

            past_records = self._collect_past_records(ann)
            future_records = self._collect_future_records(ann)

            past_xy = self._global_to_local(
                self._records_to_xy(past_records), ego_xy, ego_yaw
            )
            future_xy = self._global_to_local(
                self._records_to_xy(future_records), ego_xy, ego_yaw
            )

            headings = self._records_to_heading(past_records, ego_yaw)
            vel = self._differentiate(past_xy)
            acc = self._differentiate(vel)

            t = self._category_to_type(ann["category_name"])
            tcol = torch.full((self.past_steps, 1), float(t), dtype=torch.float32)

            feat = torch.cat(
                (past_xy, vel, acc, headings.unsqueeze(-1), tcol),
                dim=-1,
            )

            x[:, agent_index] = feat
            positions[:, agent_index] = past_xy
            future[agent_index] = future_xy

        return {
            "x": x,
            "positions": positions,
            "future": future,
            "map": self.dummy_map.clone(),
        }

    def _build_index(self) -> List[str]:

        entries: List[str] = []

        for scene in self.nusc.scene:
            token = scene["first_sample_token"]

            while token != "":
                sample = self.nusc.get("sample", token)
                entries.append(token)
                token = sample["next"]

        return entries

    def _select_agent_tokens(self, sample_token: str, ego_xy: Tensor) -> List[str]:

        anns = self.helper.get_annotations_for_sample(sample_token)
        valid = []

        for ann in anns:
            cat = ann["category_name"]

            if self._category_to_type(cat, False) is None:
                continue
            if not self._has_required_context(ann):
                continue

            pos = torch.tensor(ann["translation"][:2], dtype=torch.float32)
            dist = torch.linalg.vector_norm(pos - ego_xy).item()

            valid.append((dist, ann["token"]))

        valid.sort(key=lambda x: x[0])

        return [t for _, t in valid[: self.max_agents]]

    def _has_required_context(self, ann: Dict[str, object]) -> bool:

        prev = ann["prev"]
        for _ in range(self.past_steps - 1):
            if not prev:
                return False
            prev = self.nusc.get("sample_annotation", prev)["prev"]

        nxt = ann["next"]
        for _ in range(self.future_steps):
            if not nxt:
                return False
            nxt = self.nusc.get("sample_annotation", nxt)["next"]

        return True

    def _collect_past_records(self, ann):
        rec = [ann]
        prev = ann["prev"]

        for _ in range(self.past_steps - 1):
            if not prev:
                raise RuntimeError("missing past")
            prev_ann = self.nusc.get("sample_annotation", prev)
            rec.append(prev_ann)
            prev = prev_ann["prev"]

        rec.reverse()
        return rec

    def _collect_future_records(self, ann):
        rec = []
        nxt = ann["next"]

        for _ in range(self.future_steps):
            if not nxt:
                raise RuntimeError("missing future")
            nxt_ann = self.nusc.get("sample_annotation", nxt)
            rec.append(nxt_ann)
            nxt = nxt_ann["next"]

        return rec

    def _get_ego_pose(self, sample):

        lidar = sample["data"]["LIDAR_TOP"]
        sd = self.nusc.get("sample_data", lidar)
        ego = self.nusc.get("ego_pose", sd["ego_pose_token"])

        xy = torch.tensor(ego["translation"][:2], dtype=torch.float32)
        yaw = float(Quaternion(ego["rotation"]).yaw_pitch_roll[0])

        return xy, yaw

    @staticmethod
    def _records_to_xy(records):
        return torch.tensor([r["translation"][:2] for r in records], dtype=torch.float32)

    def _records_to_heading(self, records, ego_yaw):
        return torch.tensor(
            [math.atan2(math.sin(self._quaternion_to_yaw(r["rotation"]) - ego_yaw),
                        math.cos(self._quaternion_to_yaw(r["rotation"]) - ego_yaw))
             for r in records],
            dtype=torch.float32,
        )

    @staticmethod
    def _quaternion_to_yaw(rot):
        return float(Quaternion(rot).yaw_pitch_roll[0])

    def _global_to_local(self, pts, ego_xy, ego_yaw):
        rel = pts - ego_xy.unsqueeze(0)
        c, s = math.cos(ego_yaw), math.sin(ego_yaw)
        R = torch.tensor([[c, s], [-s, c]], dtype=pts.dtype)
        return rel @ R.T

    def _differentiate(self, v):
        d = torch.zeros_like(v)
        d[0] = (v[1] - v[0]) / self.dt
        d[-1] = (v[-1] - v[-2]) / self.dt
        if v.size(0) > 2:
            d[1:-1] = (v[2:] - v[:-2]) / (2.0 * self.dt)
        return d

    @staticmethod
    def _category_to_type(name, raise_on_unknown=True):

        if name.startswith("human.pedestrian"):
            return 1
        if name in {"vehicle.bicycle", "vehicle.motorcycle"}:
            return 2
        if name.startswith("vehicle."):
            return 0

        if raise_on_unknown:
            raise ValueError(f"unsupported category: {name}")

        return None