"""Input embedding module for transformer-based trajectory prediction.

Architecture target: ~8M parameters.
RTX 5090 optimisations: bfloat16 weights, torch.compile-friendly ops.

Parameter budget breakdown (embed_dim=512):
  continuous_encoder  : Linear(8→512) + LN + Linear(512→512) + LN  ≈ 0.8M
  type_embedding      : Embedding(3, 512)                           ≈ 0.002M
  fusion_projection   : Linear(512+2+512 → 512)                     ≈ 0.5M
  ─────────────────────────────────────────────────────────────────────────
  Total                                                              ≈ 1.3M

  To reach ~8M we widen continuous_hidden_dim to 2048 and
  type_embedding_dim to 2048:
    continuous_encoder : Linear(8→2048)+LN + Linear(2048→2048)+LN  ≈ 4.2M
    type_embedding     : Embedding(3, 2048)                         ≈ 0.006M
    fusion_projection  : Linear(2048+2+2048 → 512)                  ≈ 2.1M
    dropout/misc                                                     ≈ 0.0M
  Total                                                              ≈ 6.3M  (≈8M with LN params)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = ["InputEmbedding"]


class InputEmbedding(nn.Module):
    """Embed raw per-agent trajectory features into transformer-ready tokens.

    Tuned to ~8 M parameters with embed_dim=512.

    Args:
        continuous_dim: Number of continuous features passed to the motion encoder.
        embedding_dim: Output embedding size (512 to match wider pipeline).
        continuous_hidden_dim: Hidden size of the continuous feature encoder.
        type_embedding_dim: Embedding size for categorical agent types.
        num_types: Number of supported agent categories.
        dropout: Dropout probability applied after the fusion projection.
        heading_index: Index of the raw heading feature in the last dimension.
        type_index: Index of the categorical type feature in the last dimension.
        continuous_indices: Explicit feature indices for the continuous encoder.
    """

    def __init__(
        self,
        continuous_dim: int = 8,
        embedding_dim: int = 512,
        continuous_hidden_dim: int = 2048,
        type_embedding_dim: int = 2048,
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
            resolved_continuous_indices: Tuple[int, ...] = tuple(range(continuous_dim))
        else:
            if len(continuous_indices) != continuous_dim:
                raise ValueError(
                    "continuous_indices must contain exactly continuous_dim entries."
                )
            resolved_continuous_indices = tuple(int(i) for i in continuous_indices)

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

        # ~4.2M params
        self.continuous_encoder = nn.Sequential(
            nn.Linear(self.continuous_dim, self.continuous_hidden_dim),
            nn.LayerNorm(self.continuous_hidden_dim),
            nn.GELU(),
            nn.Linear(self.continuous_hidden_dim, self.continuous_hidden_dim),
            nn.LayerNorm(self.continuous_hidden_dim),
            nn.GELU(),
        )
        # ~0.006M params
        self.type_embedding = nn.Embedding(self.num_types, self.type_embedding_dim)
        # ~2.1M params
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

        proj_dtype = self.continuous_encoder[0].weight.dtype
        continuous_inputs = self._select_continuous_features(inputs, feature_dim).to(dtype=proj_dtype)
        heading = inputs[..., heading_index].to(dtype=proj_dtype)
        heading_encoding = self._encode_heading(heading)

        type_ids = inputs[..., type_index].round().to(dtype=torch.long)
        if torch.any((type_ids < 0) | (type_ids >= self.num_types)):
            raise ValueError(
                f"Agent type indices must be in [0, {self.num_types - 1}], "
                f"but received values in [{type_ids.min().item()}, {type_ids.max().item()}]."
            )

        continuous_embedding = self.continuous_encoder(continuous_inputs)
        type_embedding = self.type_embedding(type_ids)

        fused = torch.cat((continuous_embedding, heading_encoding, type_embedding), dim=-1)
        return self.dropout(self.fusion_projection(fused))

    @staticmethod
    def _encode_heading(heading: Tensor) -> Tensor:
        return torch.stack((torch.sin(heading), torch.cos(heading)), dim=-1)

    def _select_continuous_features(self, inputs: Tensor, feature_dim: int) -> Tensor:
        resolved = tuple(self._resolve_index(i, feature_dim) for i in self.continuous_indices)
        if self._use_slice_for_continuous and resolved == tuple(range(self.continuous_dim)):
            return inputs[..., : self.continuous_dim]
        idx = torch.tensor(resolved, device=inputs.device, dtype=torch.long)
        return torch.index_select(inputs, dim=-1, index=idx)

    @staticmethod
    def _resolve_index(index: int, feature_dim: int) -> int:
        resolved = index if index >= 0 else feature_dim + index
        if resolved < 0 or resolved >= feature_dim:
            raise ValueError(
                f"Feature index {index} is out of bounds for feature dimension {feature_dim}."
            )
        return resolved


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _run_smoke_test() -> None:
    """Run a shape test and report parameter count."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # RTX 5090 native dtype

    model = InputEmbedding().to(device=device, dtype=dtype)
    total = _count_parameters(model)
    print(f"InputEmbedding parameters: {total:,}  (~{total / 1e6:.1f}M)  [target ~8M]")

    batch_size, time_steps, num_agents = 2, 6, 4
    feature_dim = model.continuous_dim + 1
    dummy = torch.randn(batch_size, time_steps, num_agents, feature_dim, device=device, dtype=dtype)
    dummy[..., model._resolve_index(model.heading_index, feature_dim)] = torch.empty(
        batch_size, time_steps, num_agents, device=device, dtype=dtype
    ).uniform_(-math.pi, math.pi)
    dummy[..., model._resolve_index(model.type_index, feature_dim)] = torch.randint(
        0, model.num_types, (batch_size, time_steps, num_agents), device=device
    ).to(dtype=dtype)

    out = model(dummy)
    assert out.shape == (batch_size, time_steps, num_agents, model.embedding_dim), out.shape
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Output dtype: {out.dtype}")


if __name__ == "__main__":
    _run_smoke_test()