"""Social interaction transformer over agent neighborhoods."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

__all__ = ["SocialTransformerLayer", "SocialTransformer"]


class SocialTransformerLayer(nn.Module):
    """Single spatial transformer layer over the agent dimension."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor, attention_bias: Optional[Tensor] = None) -> Tensor:
        """Apply socially-aware self-attention followed by a feedforward block."""

        attention_output, _ = self.self_attention(
            inputs,
            inputs,
            inputs,
            attn_mask=attention_bias,
            need_weights=False,
        )
        inputs = self.attention_norm(inputs + self.attention_dropout(attention_output))

        feedforward_output = self.feedforward(inputs)
        return self.feedforward_norm(
            inputs + self.feedforward_dropout(feedforward_output)
        )


class SocialTransformer(nn.Module):
    """Graph-style transformer that models agent-to-agent interactions per timestep.

    The module expects embeddings from the temporal encoder and performs attention
    over the agent dimension at each timestep after reshaping ``(B, T, N, D)`` to
    ``(B * T, N, D)``.
    """

    def __init__(
        self,
        num_layers: int = 6,
        num_heads: int = 8,
        embed_dim: int = 896,
        ff_dim: int = 1536,
        dropout: float = 0.1,
        distance_sigma: float = 10.0,
        num_types: int = 3,
        type_embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if distance_sigma <= 0.0:
            raise ValueError("distance_sigma must be positive.")
        if num_types <= 0:
            raise ValueError("num_types must be positive.")
        if type_embedding_dim <= 0:
            raise ValueError("type_embedding_dim must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_p = dropout
        self.num_types = num_types
        self.type_embedding_dim = type_embedding_dim

        self.log_distance_sigma = nn.Parameter(
            torch.tensor(math.log(distance_sigma), dtype=torch.float32)
        )
        self.type_embedding = nn.Embedding(num_types, type_embedding_dim)
        self.type_projection = nn.Linear(embed_dim + type_embedding_dim, embed_dim)

        self.layers = nn.ModuleList(
            [
                SocialTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs: Tensor,
        positions: Tensor,
        agent_mask: Optional[Tensor] = None,
        agent_types: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply spatial social attention across agents at each timestep.

        Args:
            inputs: Tensor of shape ``(batch, time, agents, embed_dim)``.
            positions: Tensor of shape ``(batch, time, agents, 2)``.
            agent_mask: Optional boolean-compatible mask of shape ``(batch, time, agents)``.
                ``True`` values mark padded agents that should be ignored.
            agent_types: Optional integer tensor of shape ``(batch, agents)`` or
                ``(batch, time, agents)``. When omitted, the module relies on the
                type information already encoded by ``InputEmbedding``.

        Returns:
            Tensor of shape ``(batch, time, agents, embed_dim)``.
        """

        if inputs.ndim != 4:
            raise ValueError(
                f"Expected inputs with 4 dimensions (batch, time, agents, embed_dim), "
                f"but received shape {tuple(inputs.shape)}."
            )
        if positions.ndim != 4:
            raise ValueError(
                f"Expected positions with 4 dimensions (batch, time, agents, 2), "
                f"but received shape {tuple(positions.shape)}."
            )

        batch_size, time_steps, num_agents, embed_dim = inputs.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, but received {embed_dim}."
            )
        if positions.shape[:3] != inputs.shape[:3] or positions.size(-1) != 2:
            raise ValueError(
                "positions must have shape (batch, time, agents, 2) that matches inputs."
            )

        model_dtype = self.layers[0].self_attention.in_proj_weight.dtype
        outputs = inputs.to(dtype=model_dtype)
        flat_positions = positions.to(device=inputs.device, dtype=torch.float32).reshape(
            batch_size * time_steps,
            num_agents,
            2,
        )

        normalized_types = self._normalize_agent_types(
            agent_types=agent_types,
            batch_size=batch_size,
            time_steps=time_steps,
            num_agents=num_agents,
            device=inputs.device,
        )
        if normalized_types is not None:
            type_features = self.type_embedding(normalized_types)
            outputs = self.type_projection(torch.cat((outputs, type_features), dim=-1))

        outputs = outputs.reshape(batch_size * time_steps, num_agents, embed_dim)

        normalized_mask = self._normalize_agent_mask(
            agent_mask=agent_mask,
            batch_size=batch_size,
            time_steps=time_steps,
            num_agents=num_agents,
            device=inputs.device,
        )
        flat_agent_mask = None
        if normalized_mask is not None:
            flat_agent_mask = normalized_mask.reshape(batch_size * time_steps, num_agents)
            outputs = outputs.masked_fill(flat_agent_mask.unsqueeze(-1), 0.0)

        attention_bias = self._build_attention_bias(
            flat_positions=flat_positions,
            flat_agent_mask=flat_agent_mask,
            output_dtype=outputs.dtype,
        )

        for layer in self.layers:
            outputs = layer(outputs, attention_bias=attention_bias)
            if flat_agent_mask is not None:
                outputs = outputs.masked_fill(flat_agent_mask.unsqueeze(-1), 0.0)

        outputs = outputs.reshape(batch_size, time_steps, num_agents, embed_dim)
        if normalized_mask is not None:
            outputs = outputs.masked_fill(normalized_mask.unsqueeze(-1), 0.0)

        return outputs

    def _build_attention_bias(
        self,
        flat_positions: Tensor,
        flat_agent_mask: Optional[Tensor],
        output_dtype: torch.dtype,
    ) -> Tensor:
        """Construct additive spatial attention bias from pairwise distances."""

        sigma = self.log_distance_sigma.float().exp().clamp_min(1e-6)
        pairwise_distances = torch.cdist(flat_positions, flat_positions, p=2)
        attention_bias = (-pairwise_distances / sigma).to(dtype=output_dtype)

        if flat_agent_mask is not None:
            effective_agent_mask = flat_agent_mask
            fully_padded_rows = effective_agent_mask.all(dim=1)
            if torch.any(fully_padded_rows):
                effective_agent_mask = effective_agent_mask.clone()
                effective_agent_mask[fully_padded_rows] = False

            mask_fill_value = torch.finfo(attention_bias.dtype).min
            attention_bias = attention_bias.masked_fill(
                effective_agent_mask[:, None, :],
                mask_fill_value,
            )

        batch_time, num_agents, _ = attention_bias.shape
        return attention_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(
            batch_time * self.num_heads,
            num_agents,
            num_agents,
        )

    def _normalize_agent_types(
        self,
        agent_types: Optional[Tensor],
        batch_size: int,
        time_steps: int,
        num_agents: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Validate and broadcast agent types when provided."""

        if agent_types is None:
            return None

        if agent_types.shape == (batch_size, num_agents):
            normalized_types = agent_types.unsqueeze(1).expand(batch_size, time_steps, num_agents)
        elif agent_types.shape == (batch_size, time_steps, num_agents):
            normalized_types = agent_types
        else:
            raise ValueError(
                "agent_types must have shape (batch, agents) or (batch, time, agents), "
                f"but received {tuple(agent_types.shape)}."
            )

        normalized_types = normalized_types.to(device=device, dtype=torch.long)
        if torch.any((normalized_types < 0) | (normalized_types >= self.num_types)):
            raise ValueError(
                f"agent_types must be in [0, {self.num_types - 1}], "
                f"but received values in [{normalized_types.min().item()}, {normalized_types.max().item()}]."
            )
        return normalized_types

    @staticmethod
    def _normalize_agent_mask(
        agent_mask: Optional[Tensor],
        batch_size: int,
        time_steps: int,
        num_agents: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Validate and normalize an agent padding mask to boolean form."""

        if agent_mask is None:
            return None
        if agent_mask.shape != (batch_size, time_steps, num_agents):
            raise ValueError(
                "agent_mask must have shape (batch, time, agents), "
                f"but received {tuple(agent_mask.shape)}."
            )
        return agent_mask.to(device=device, dtype=torch.bool)


def _run_smoke_test() -> None:
    """Run a minimal shape test with dummy embeddings and positions."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SocialTransformer().to(device)

    embeddings = torch.randn(2, 6, 4, model.embed_dim, device=device)
    positions = torch.randn(2, 6, 4, 2, device=device)
    agent_mask = torch.zeros(2, 6, 4, dtype=torch.bool, device=device)
    agent_mask[0, :, 3] = True
    agent_mask[1, 5, 2:] = True
    agent_types = torch.randint(0, model.num_types, (2, 4), device=device)

    outputs = model(
        embeddings,
        positions,
        agent_mask=agent_mask,
        agent_types=agent_types,
    )
    expected_shape = (2, 6, 4, model.embed_dim)
    assert outputs.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(outputs.shape)}."
    )
    print(f"Output shape: {tuple(outputs.shape)}")


if __name__ == "__main__":
    _run_smoke_test()
