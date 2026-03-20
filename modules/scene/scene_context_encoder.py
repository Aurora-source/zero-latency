"""Scene context cross-attention encoder for map-aware trajectory reasoning."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ["SceneContextLayer", "SceneContextEncoder"]


class SceneContextLayer(nn.Module):
    """Single cross-attention layer from agents to map elements."""

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

        self.cross_attention = nn.MultiheadAttention(
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

    def forward(
        self,
        query_tokens: Tensor,
        map_tokens: Tensor,
        attention_mask: Optional[Tensor] = None,
        zero_attention_queries: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply map cross-attention followed by a feedforward block."""

        attention_output, _ = self.cross_attention(
            query=query_tokens,
            key=map_tokens,
            value=map_tokens,
            attn_mask=attention_mask,
            need_weights=False,
        )
        if zero_attention_queries is not None:
            attention_output = attention_output.masked_fill(
                zero_attention_queries.unsqueeze(-1),
                0.0,
            )

        query_tokens = self.attention_norm(
            query_tokens + self.attention_dropout(attention_output)
        )
        feedforward_output = self.feedforward(query_tokens)
        return self.feedforward_norm(
            query_tokens + self.feedforward_dropout(feedforward_output)
        )


class SceneContextEncoder(nn.Module):
    """Cross-attention encoder that injects vectorized map context into agents."""

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        embed_dim: int = 896,
        map_dim: int = 256,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_distance: float = 50.0,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if map_dim <= 0:
            raise ValueError("map_dim must be positive.")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.map_dim = map_dim
        self.ff_dim = ff_dim
        self.dropout_p = dropout
        self.max_distance = max_distance

        self.map_projection = nn.Linear(map_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                SceneContextLayer(
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
        agent_embeddings: Tensor,
        map_features: Tensor,
        map_padding_mask: Optional[Tensor] = None,
        agent_positions: Optional[Tensor] = None,
        map_positions: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse scene context into agent embeddings via cross-attention.

        Args:
            agent_embeddings: Tensor of shape ``(batch, time, agents, embed_dim)``.
            map_features: Tensor of shape ``(batch, map_elements, map_dim)``.
            map_padding_mask: Optional boolean-compatible mask of shape
                ``(batch, map_elements)`` where ``True`` marks invalid map elements.
            agent_positions: Optional tensor of shape ``(batch, time, agents, 2)``.
            map_positions: Optional tensor of shape ``(batch, map_elements, 2)``.
            agent_mask: Optional boolean-compatible mask of shape
                ``(batch, time, agents)`` to preserve padded agents through this stage.

        Returns:
            Tensor of shape ``(batch, time, agents, embed_dim)``.
        """

        if agent_embeddings.ndim != 4:
            raise ValueError(
                "agent_embeddings must have shape (batch, time, agents, embed_dim), "
                f"but received {tuple(agent_embeddings.shape)}."
            )
        if map_features.ndim != 3:
            raise ValueError(
                "map_features must have shape (batch, map_elements, map_dim), "
                f"but received {tuple(map_features.shape)}."
            )

        batch_size, time_steps, num_agents, embed_dim = agent_embeddings.shape
        map_batch_size, num_map_elements, map_dim = map_features.shape
        if map_batch_size != batch_size:
            raise ValueError("agent_embeddings and map_features must share the same batch size.")
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected agent embed_dim={self.embed_dim}, but received {embed_dim}."
            )
        if map_dim != self.map_dim:
            raise ValueError(
                f"Expected map_dim={self.map_dim}, but received {map_dim}."
            )

        self._validate_position_inputs(
            agent_positions=agent_positions,
            map_positions=map_positions,
            batch_size=batch_size,
            time_steps=time_steps,
            num_agents=num_agents,
            num_map_elements=num_map_elements,
        )

        normalized_map_padding_mask = self._normalize_map_padding_mask(
            map_padding_mask=map_padding_mask,
            batch_size=batch_size,
            num_map_elements=num_map_elements,
            device=agent_embeddings.device,
        )
        normalized_agent_mask = self._normalize_agent_mask(
            agent_mask=agent_mask,
            batch_size=batch_size,
            time_steps=time_steps,
            num_agents=num_agents,
            device=agent_embeddings.device,
        )

        num_queries = time_steps * num_agents
        model_dtype = self.layers[0].cross_attention.in_proj_weight.dtype
        query_tokens = agent_embeddings.to(dtype=model_dtype).reshape(
            batch_size,
            num_queries,
            embed_dim,
        )
        projected_map_features = self.map_projection(map_features.to(dtype=model_dtype))

        flat_agent_mask = None
        if normalized_agent_mask is not None:
            flat_agent_mask = normalized_agent_mask.reshape(batch_size, num_queries)
            query_tokens = query_tokens.masked_fill(flat_agent_mask.unsqueeze(-1), 0.0)

        attention_mask, zero_attention_queries = self._build_attention_mask(
            batch_size=batch_size,
            time_steps=time_steps,
            num_agents=num_agents,
            num_map_elements=num_map_elements,
            map_padding_mask=normalized_map_padding_mask,
            agent_positions=agent_positions,
            map_positions=map_positions,
            device=agent_embeddings.device,
        )

        if flat_agent_mask is not None:
            zero_attention_queries = (
                flat_agent_mask
                if zero_attention_queries is None
                else (zero_attention_queries | flat_agent_mask)
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(
                -1,
                self.num_heads,
                -1,
                -1,
            ).reshape(
                batch_size * self.num_heads,
                num_queries,
                num_map_elements,
            )

        for layer in self.layers:
            query_tokens = layer(
                query_tokens,
                projected_map_features,
                attention_mask=attention_mask,
                zero_attention_queries=zero_attention_queries,
            )
            if flat_agent_mask is not None:
                query_tokens = query_tokens.masked_fill(flat_agent_mask.unsqueeze(-1), 0.0)

        outputs = query_tokens.reshape(batch_size, time_steps, num_agents, embed_dim)
        if normalized_agent_mask is not None:
            outputs = outputs.masked_fill(normalized_agent_mask.unsqueeze(-1), 0.0)
        return outputs

    def _build_attention_mask(
        self,
        batch_size: int,
        time_steps: int,
        num_agents: int,
        num_map_elements: int,
        map_padding_mask: Optional[Tensor],
        agent_positions: Optional[Tensor],
        map_positions: Optional[Tensor],
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Build a per-query map mask from padding and optional spatial filtering."""

        num_queries = time_steps * num_agents
        base_mask = torch.zeros(
            batch_size,
            num_queries,
            num_map_elements,
            device=device,
            dtype=torch.bool,
        )
        if map_padding_mask is not None:
            base_mask = map_padding_mask.unsqueeze(1).expand(
                batch_size,
                num_queries,
                num_map_elements,
            )

        zero_attention_queries: Optional[Tensor] = None
        blocked = base_mask

        if agent_positions is not None and map_positions is not None:
            flat_agent_positions = agent_positions.to(device=device, dtype=torch.float32).reshape(
                batch_size,
                num_queries,
                2,
            )
            normalized_map_positions = map_positions.to(device=device, dtype=torch.float32)
            spatial_mask = torch.cdist(flat_agent_positions, normalized_map_positions, p=2) > self.max_distance
            blocked = base_mask | spatial_mask
            zero_attention_queries = ~((~blocked).any(dim=-1))
            blocked = torch.where(zero_attention_queries.unsqueeze(-1), base_mask, blocked)

        no_valid_after_fallback = ~((~blocked).any(dim=-1))
        if torch.any(no_valid_after_fallback):
            blocked = torch.where(
                no_valid_after_fallback.unsqueeze(-1),
                torch.zeros_like(blocked),
                blocked,
            )
            zero_attention_queries = (
                no_valid_after_fallback
                if zero_attention_queries is None
                else (zero_attention_queries | no_valid_after_fallback)
            )

        if zero_attention_queries is not None and not torch.any(zero_attention_queries):
            zero_attention_queries = None
        if not torch.any(blocked):
            blocked = None

        return blocked, zero_attention_queries

    @staticmethod
    def _validate_position_inputs(
        agent_positions: Optional[Tensor],
        map_positions: Optional[Tensor],
        batch_size: int,
        time_steps: int,
        num_agents: int,
        num_map_elements: int,
    ) -> None:
        """Validate optional spatial-filtering position tensors."""

        if (agent_positions is None) != (map_positions is None):
            raise ValueError(
                "agent_positions and map_positions must both be provided for spatial filtering."
            )
        if agent_positions is None and map_positions is None:
            return
        if agent_positions is None or map_positions is None:
            raise ValueError("Both agent_positions and map_positions are required.")
        if agent_positions.shape != (batch_size, time_steps, num_agents, 2):
            raise ValueError(
                "agent_positions must have shape (batch, time, agents, 2), "
                f"but received {tuple(agent_positions.shape)}."
            )
        if map_positions.shape != (batch_size, num_map_elements, 2):
            raise ValueError(
                "map_positions must have shape (batch, map_elements, 2), "
                f"but received {tuple(map_positions.shape)}."
            )

    @staticmethod
    def _normalize_map_padding_mask(
        map_padding_mask: Optional[Tensor],
        batch_size: int,
        num_map_elements: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Validate and normalize a map padding mask to boolean form."""

        if map_padding_mask is None:
            return None
        if map_padding_mask.shape != (batch_size, num_map_elements):
            raise ValueError(
                "map_padding_mask must have shape (batch, map_elements), "
                f"but received {tuple(map_padding_mask.shape)}."
            )
        return map_padding_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _normalize_agent_mask(
        agent_mask: Optional[Tensor],
        batch_size: int,
        time_steps: int,
        num_agents: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Validate and normalize an optional agent padding mask to boolean form."""

        if agent_mask is None:
            return None
        if agent_mask.shape != (batch_size, time_steps, num_agents):
            raise ValueError(
                "agent_mask must have shape (batch, time, agents), "
                f"but received {tuple(agent_mask.shape)}."
            )
        return agent_mask.to(device=device, dtype=torch.bool)


def _run_smoke_test() -> None:
    """Run a minimal shape test with dummy agents and map features."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SceneContextEncoder().to(device)

    agent_embeddings = torch.randn(2, 6, 4, model.embed_dim, device=device)
    map_features = torch.randn(2, 128, model.map_dim, device=device)
    agent_positions = torch.randn(2, 6, 4, 2, device=device)
    map_positions = torch.randn(2, 128, 2, device=device)
    map_padding_mask = torch.zeros(2, 128, dtype=torch.bool, device=device)
    map_padding_mask[0, 120:] = True
    map_padding_mask[1, 110:] = True
    agent_mask = torch.zeros(2, 6, 4, dtype=torch.bool, device=device)
    agent_mask[0, :, 3] = True

    outputs = model(
        agent_embeddings,
        map_features,
        map_padding_mask=map_padding_mask,
        agent_positions=agent_positions,
        map_positions=map_positions,
        agent_mask=agent_mask,
    )
    expected_shape = (2, 6, 4, model.embed_dim)
    assert outputs.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(outputs.shape)}."
    )
    print(f"Output shape: {tuple(outputs.shape)}")


if __name__ == "__main__":
    _run_smoke_test()
