from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

__all__ = ["PositionalEncoding", "TransformerLayer", "TemporalTransformer"]


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "positional_encoding",
            self._build_encoding(max_len, embed_dim),
            persistent=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim not in (3, 4):
            raise ValueError(
                "PositionalEncoding expects a tensor with 3 or 4 dimensions, "
                f"but received shape {tuple(inputs.shape)}."
            )

        sequence_length = inputs.size(1)
        encoding = self._get_encoding(
            sequence_length=sequence_length,
            device=inputs.device,
            dtype=inputs.dtype,
        )

        if inputs.ndim == 3:
            return self.dropout(inputs + encoding.unsqueeze(0))
        return self.dropout(inputs + encoding.view(1, sequence_length, 1, self.embed_dim))

    def _get_encoding(
        self,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if sequence_length > self.positional_encoding.size(0):
            self.positional_encoding = self._build_encoding(
                sequence_length,
                self.embed_dim,
                device=self.positional_encoding.device,
            )

        return self.positional_encoding[:sequence_length].to(device=device, dtype=dtype)

    @staticmethod
    def _build_encoding(
        sequence_length: int,
        embed_dim: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        position = torch.arange(sequence_length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )

        encoding = torch.zeros(sequence_length, embed_dim, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        return encoding


class TransformerLayer(nn.Module):
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

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attention_output, _ = self.self_attention(
            inputs,
            inputs,
            inputs,
            key_padding_mask=padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        inputs = self.attention_norm(inputs + self.attention_dropout(attention_output))

        feedforward_output = self.feedforward(inputs)
        return self.feedforward_norm(
            inputs + self.feedforward_dropout(feedforward_output)
        )


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int = 16,
        num_heads: int = 8,
        embed_dim: int = 896,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
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

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_p = dropout

        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            dropout=dropout,
            max_len=max_len,
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
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
        padding_mask: Optional[Tensor] = None,
        use_causal_mask: bool = False,
    ) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                f"Expected inputs with 4 dimensions (batch, time, agents, embed_dim), "
                f"but received shape {tuple(inputs.shape)}."
            )

        batch_size, time_steps, num_agents, embed_dim = inputs.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, but received {embed_dim}."
            )

        model_dtype = self.layers[0].self_attention.in_proj_weight.dtype
        outputs = inputs.to(dtype=model_dtype)
        outputs = self.positional_encoding(outputs)
        outputs = outputs.permute(0, 2, 1, 3).reshape(
            batch_size * num_agents,
            time_steps,
            embed_dim,
        )

        base_padding_mask = self._normalize_padding_mask(
            padding_mask=padding_mask,
            batch_size=batch_size,
            time_steps=time_steps,
            device=inputs.device,
        )
        expanded_padding_mask = self._expand_padding_mask(base_padding_mask, num_agents)

        if expanded_padding_mask is not None:
            outputs = outputs.masked_fill(expanded_padding_mask.unsqueeze(-1), 0.0)
            effective_padding_mask = expanded_padding_mask
            fully_padded_rows = effective_padding_mask.all(dim=1)
            if torch.any(fully_padded_rows):
                effective_padding_mask = effective_padding_mask.clone()
                effective_padding_mask[fully_padded_rows] = False
        else:
            effective_padding_mask = None

        attention_mask = None
        if use_causal_mask:
            attention_mask = self._build_causal_mask(time_steps, device=inputs.device)

        for layer in self.layers:
            outputs = layer(
                outputs,
                padding_mask=effective_padding_mask,
                attention_mask=attention_mask,
            )
            if expanded_padding_mask is not None:
                outputs = outputs.masked_fill(expanded_padding_mask.unsqueeze(-1), 0.0)

        outputs = outputs.reshape(batch_size, num_agents, time_steps, embed_dim)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()

        if base_padding_mask is not None:
            outputs = outputs.masked_fill(base_padding_mask[:, :, None, None], 0.0)

        return outputs

    @staticmethod
    def _normalize_padding_mask(
        padding_mask: Optional[Tensor],
        batch_size: int,
        time_steps: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if padding_mask is None:
            return None
        if padding_mask.shape != (batch_size, time_steps):
            raise ValueError(
                "padding_mask must have shape (batch, time), "
                f"but received {tuple(padding_mask.shape)}."
            )
        return padding_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _expand_padding_mask(
        padding_mask: Optional[Tensor],
        num_agents: int,
    ) -> Optional[Tensor]:

        if padding_mask is None:
            return None

        batch_size, time_steps = padding_mask.shape
        return padding_mask.unsqueeze(1).expand(batch_size, num_agents, time_steps).reshape(
            batch_size * num_agents,
            time_steps,
        )

    @staticmethod
    def _build_causal_mask(time_steps: int, device: torch.device) -> Tensor:
        
        return torch.triu(
            torch.ones(time_steps, time_steps, device=device, dtype=torch.bool),
            diagonal=1,
        )


def _run_smoke_test() -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformer().to(device)

    dummy_inputs = torch.randn(2, 10, 5, model.embed_dim, device=device)
    padding_mask = torch.zeros(2, 10, dtype=torch.bool, device=device)
    padding_mask[0, 8:] = True
    padding_mask[1, 9:] = True

    outputs = model(dummy_inputs, padding_mask=padding_mask, use_causal_mask=True)
    expected_shape = (2, 10, 5, model.embed_dim)
    assert outputs.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(outputs.shape)}."
    )
    print(f"Output shape: {tuple(outputs.shape)}")


if __name__ == "__main__":
    _run_smoke_test()
