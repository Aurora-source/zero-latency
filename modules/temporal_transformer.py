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
            raise ValueError("embed_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "positional_encoding",
            self._build_encoding(max_len, embed_dim),
            persistent=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:

        if inputs.ndim not in (3, 4):
            raise ValueError(f"input shape wrong: {tuple(inputs.shape)}")

        L = inputs.size(1)

        enc = self._get_encoding(
            sequence_length=L,
            device=inputs.device,
            dtype=inputs.dtype,
        )

        if inputs.ndim == 3:
            return self.dropout(inputs + enc.unsqueeze(0))

        return self.dropout(inputs + enc.view(1, L, 1, self.embed_dim))

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

        pos = torch.arange(sequence_length, device=device, dtype=torch.float32).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, embed_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )

        enc = torch.zeros(sequence_length, embed_dim, device=device, dtype=torch.float32)

        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div[: enc[:, 1::2].shape[1]])

        return enc


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
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

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

        x, _ = self.self_attention(
            inputs,
            inputs,
            inputs,
            key_padding_mask=padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )

        inputs = self.attention_norm(inputs + self.attention_dropout(x))

        x = self.feedforward(inputs)

        return self.feedforward_norm(inputs + self.feedforward_dropout(x))


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
            raise ValueError("num_layers must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

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
            raise ValueError(f"input shape wrong: {tuple(inputs.shape)}")

        b, t, a, e = inputs.shape

        if e != self.embed_dim:
            raise ValueError(f"embed dim mismatch: {e}")

        dtype = self.layers[0].self_attention.in_proj_weight.dtype

        out = inputs.to(dtype=dtype)
        out = self.positional_encoding(out)

        out = out.permute(0, 2, 1, 3).reshape(b * a, t, e)

        base_mask = self._normalize_padding_mask(
            padding_mask, b, t, inputs.device
        )

        exp_mask = self._expand_padding_mask(base_mask, a)

        if exp_mask is not None:
            out = out.masked_fill(exp_mask.unsqueeze(-1), 0.0)

            eff_mask = exp_mask
            full = eff_mask.all(dim=1)

            if torch.any(full):
                eff_mask = eff_mask.clone()
                eff_mask[full] = False
        else:
            eff_mask = None

        attn_mask = None
        if use_causal_mask:
            attn_mask = self._build_causal_mask(t, inputs.device)

        for layer in self.layers:
            out = layer(out, padding_mask=eff_mask, attention_mask=attn_mask)

            if exp_mask is not None:
                out = out.masked_fill(exp_mask.unsqueeze(-1), 0.0)

        out = out.reshape(b, a, t, e).permute(0, 2, 1, 3).contiguous()

        if base_mask is not None:
            out = out.masked_fill(base_mask[:, :, None, None], 0.0)

        return out

    @staticmethod
    def _normalize_padding_mask(
        padding_mask: Optional[Tensor],
        b: int,
        t: int,
        device: torch.device,
    ) -> Optional[Tensor]:

        if padding_mask is None:
            return None

        if padding_mask.shape != (b, t):
            raise ValueError(f"padding mask wrong: {tuple(padding_mask.shape)}")

        return padding_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _expand_padding_mask(
        padding_mask: Optional[Tensor],
        a: int,
    ) -> Optional[Tensor]:

        if padding_mask is None:
            return None

        b, t = padding_mask.shape

        return padding_mask.unsqueeze(1).expand(b, a, t).reshape(b * a, t)

    @staticmethod
    def _build_causal_mask(t: int, device: torch.device) -> Tensor:

        return torch.triu(
            torch.ones(t, t, device=device, dtype=torch.bool),
            diagonal=1,
        )