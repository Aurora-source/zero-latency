from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = ["InputEmbedding"]


class InputEmbedding(nn.Module):

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
            raise ValueError("continuous_dim must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if continuous_hidden_dim <= 0:
            raise ValueError("continuous_hidden_dim must be positive")
        if type_embedding_dim <= 0:
            raise ValueError("type_embedding_dim must be positive")
        if num_types <= 0:
            raise ValueError("num_types must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

        if continuous_indices is None:
            idx = tuple(range(continuous_dim))
        else:
            if len(continuous_indices) != continuous_dim:
                raise ValueError("continuous_indices size mismatch")
            idx = tuple(int(i) for i in continuous_indices)

        self.continuous_dim = continuous_dim
        self.embedding_dim = embedding_dim
        self.continuous_hidden_dim = continuous_hidden_dim
        self.type_embedding_dim = type_embedding_dim
        self.num_types = num_types
        self.dropout_p = dropout
        self.heading_index = heading_index
        self.type_index = type_index
        self.continuous_indices: Tuple[int, ...] = idx
        self._use_slice_for_continuous = idx == tuple(range(self.continuous_dim))

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

        if inputs.ndim != 4:
            raise ValueError(f"input shape wrong: {tuple(inputs.shape)}")

        fdim = inputs.size(-1)

        h_idx = self._resolve_index(self.heading_index, fdim)
        t_idx = self._resolve_index(self.type_index, fdim)

        cont = self._select_continuous_features(inputs, fdim)

        dtype = self.continuous_encoder[0].weight.dtype
        cont = cont.to(dtype=dtype)

        heading = inputs[..., h_idx].to(dtype=dtype)
        heading_enc = self._encode_heading(heading)

        type_ids = inputs[..., t_idx].round().to(dtype=torch.long)

        if torch.any((type_ids < 0) | (type_ids >= self.num_types)):
            raise ValueError("type index out of range")

        cont_emb = self.continuous_encoder(cont)
        type_emb = self.type_embedding(type_ids)

        fused = torch.cat((cont_emb, heading_enc, type_emb), dim=-1)

        return self.dropout(self.fusion_projection(fused))

    @staticmethod
    def _encode_heading(heading: Tensor) -> Tensor:
        return torch.stack((torch.sin(heading), torch.cos(heading)), dim=-1)

    def _select_continuous_features(self, inputs: Tensor, fdim: int) -> Tensor:

        idx = tuple(self._resolve_index(i, fdim) for i in self.continuous_indices)

        if self._use_slice_for_continuous and idx == tuple(range(self.continuous_dim)):
            return inputs[..., : self.continuous_dim]

        idx_tensor = torch.tensor(idx, device=inputs.device, dtype=torch.long)

        return torch.index_select(inputs, dim=-1, index=idx_tensor)

    @staticmethod
    def _resolve_index(index: int, fdim: int) -> int:

        i = index if index >= 0 else fdim + index

        if i < 0 or i >= fdim:
            raise ValueError(f"index {index} out of bounds")

        return i