"""Input embedding module for transformer-based trajectory prediction."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = ["InputEmbedding"]


class InputEmbedding(nn.Module):
    """Embed raw per-agent trajectory features into transformer-ready tokens.

    The module expects a 4D tensor with shape ``(batch, time, agents, features)``.
    By default, the first eight feature slots are treated as continuous inputs for
    the motion encoder, the heading angle is read from ``heading_index`` and
    converted to a sine/cosine representation, and the final feature slot is used
    as the categorical agent type for an embedding lookup.

    Args:
        continuous_dim: Number of continuous features passed to the linear encoder.
        embedding_dim: Output embedding size.
        continuous_hidden_dim: Hidden size of the continuous feature encoder.
        type_embedding_dim: Embedding size for categorical agent types.
        num_types: Number of supported agent categories.
        dropout: Dropout probability applied after the fusion projection.
        heading_index: Index of the raw heading feature in the last dimension.
        type_index: Index of the categorical type feature in the last dimension.
        continuous_indices: Explicit feature indices for the continuous encoder.
            When ``None``, the first ``continuous_dim`` features are used.
    """

    def __init__(
        self,
        continuous_dim: int = 8,
        embedding_dim: int = 896,
        continuous_hidden_dim: int = 512,
        type_embedding_dim: int = 64,
        num_types: int = 3,
        dropout: float = 0.1,
        heading_index: int = 6,
        type_index: int = -1,
        continuous_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        if continuous_dim <= 0:
            raise ValueError("continuous_dim must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if continuous_hidden_dim <= 0:
            raise ValueError("continuous_hidden_dim must be positive.")
        if type_embedding_dim <= 0:
            raise ValueError("type_embedding_dim must be positive.")
        if num_types <= 0:
            raise ValueError("num_types must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        if continuous_indices is None:
            resolved_continuous_indices = tuple(range(continuous_dim))
        else:
            if len(continuous_indices) != continuous_dim:
                raise ValueError(
                    "continuous_indices must contain exactly continuous_dim entries."
                )
            resolved_continuous_indices = tuple(int(index) for index in continuous_indices)

        self.continuous_dim = continuous_dim
        self.embedding_dim = embedding_dim
        self.continuous_hidden_dim = continuous_hidden_dim
        self.type_embedding_dim = type_embedding_dim
        self.num_types = num_types
        self.dropout_p = dropout
        self.heading_index = heading_index
        self.type_index = type_index
        self.continuous_indices: Tuple[int, ...] = resolved_continuous_indices
        self._use_slice_for_continuous = self.continuous_indices == tuple(
            range(self.continuous_dim)
        )

        self.continuous_encoder = nn.Sequential(
            nn.Linear(self.continuous_dim, self.continuous_hidden_dim),
            nn.LayerNorm(self.continuous_hidden_dim),
            nn.GELU(),
        )
        self.type_embedding = nn.Embedding(self.num_types, self.type_embedding_dim)
        self.fusion_projection = nn.Linear(
            self.continuous_hidden_dim + 2 + self.type_embedding_dim,
            self.embedding_dim,
        )
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, inputs: Tensor) -> Tensor:
        """Embed raw trajectory inputs.

        Args:
            inputs: Tensor of shape ``(batch, time, agents, features)``.

        Returns:
            Tensor of shape ``(batch, time, agents, embedding_dim)``.
        """

        if inputs.ndim != 4:
            raise ValueError(
                f"Expected inputs with 4 dimensions (batch, time, agents, features), "
                f"but received shape {tuple(inputs.shape)}."
            )

        feature_dim = inputs.size(-1)
        heading_index = self._resolve_index(self.heading_index, feature_dim)
        type_index = self._resolve_index(self.type_index, feature_dim)

        continuous_inputs = self._select_continuous_features(inputs, feature_dim)
        projection_dtype = self.continuous_encoder[0].weight.dtype
        continuous_inputs = continuous_inputs.to(dtype=projection_dtype)

        heading = inputs[..., heading_index].to(dtype=projection_dtype)
        heading_encoding = self._encode_heading(heading)

        type_ids = inputs[..., type_index].round().to(dtype=torch.long)
        if torch.any((type_ids < 0) | (type_ids >= self.num_types)):
            raise ValueError(
                f"Agent type indices must be in [0, {self.num_types - 1}], "
                f"but received values in [{type_ids.min().item()}, {type_ids.max().item()}]."
            )

        continuous_embedding = self.continuous_encoder(continuous_inputs)
        type_embedding = self.type_embedding(type_ids)

        fused = torch.cat(
            (continuous_embedding, heading_encoding, type_embedding),
            dim=-1,
        )
        return self.dropout(self.fusion_projection(fused))

    @staticmethod
    def _encode_heading(heading: Tensor) -> Tensor:
        """Encode a heading angle in radians as sine and cosine features."""

        return torch.stack((torch.sin(heading), torch.cos(heading)), dim=-1)

    def _select_continuous_features(self, inputs: Tensor, feature_dim: int) -> Tensor:
        """Gather continuous features without Python loops over batch dimensions."""

        resolved_indices = tuple(
            self._resolve_index(index, feature_dim) for index in self.continuous_indices
        )

        if self._use_slice_for_continuous and resolved_indices == tuple(
            range(self.continuous_dim)
        ):
            return inputs[..., : self.continuous_dim]

        index_tensor = torch.tensor(
            resolved_indices,
            device=inputs.device,
            dtype=torch.long,
        )
        return torch.index_select(inputs, dim=-1, index=index_tensor)

    @staticmethod
    def _resolve_index(index: int, feature_dim: int) -> int:
        """Resolve possibly negative feature indices against the runtime feature size."""

        resolved_index = index if index >= 0 else feature_dim + index
        if resolved_index < 0 or resolved_index >= feature_dim:
            raise ValueError(
                f"Feature index {index} is out of bounds for feature dimension {feature_dim}."
            )
        return resolved_index


def _run_smoke_test() -> None:
    """Run a minimal shape test with dummy nuScenes-like inputs."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, time_steps, num_agents = 2, 6, 4

    model = InputEmbedding().to(device)
    feature_dim = model.continuous_dim + 1
    heading_slot = model._resolve_index(model.heading_index, feature_dim)
    type_slot = model._resolve_index(model.type_index, feature_dim)

    dummy_inputs = torch.randn(
        batch_size,
        time_steps,
        num_agents,
        feature_dim,
        device=device,
    )
    dummy_inputs[..., heading_slot] = torch.empty(
        batch_size,
        time_steps,
        num_agents,
        device=device,
    ).uniform_(-math.pi, math.pi)
    dummy_inputs[..., type_slot] = torch.randint(
        low=0,
        high=model.num_types,
        size=(batch_size, time_steps, num_agents),
        device=device,
        dtype=torch.long,
    ).to(dtype=dummy_inputs.dtype)

    outputs = model(dummy_inputs)
    expected_shape = (batch_size, time_steps, num_agents, model.embedding_dim)

    assert outputs.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(outputs.shape)}."
    )
    print(f"Smoke test passed on {device.type}: output shape {tuple(outputs.shape)}")


if __name__ == "__main__":
    _run_smoke_test()
