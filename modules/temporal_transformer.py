"""Temporal transformer encoder for agent trajectory histories.

Architecture target : 16 layers, ~110 M parameters.
RTX 5090 optimisations:
  - bfloat16 weights (Blackwell native)
  - scaled_dot_product_attention (FlashAttention-3 kernel path)
  - torch.compile-friendly: no data-dependent Python branches in forward

Parameter budget (embed_dim=512, ff_dim=2048, num_heads=8, 16 layers):
  Per layer:
    Self-attention QKV + out proj : 4 × (512×512) = 4 × 262144  ≈ 1.048M
    FFN (512→2048→512)            : 512×2048 + 2048×512          ≈ 2.097M
    LayerNorm ×2                  : 4×512                        ≈ 0.001M
    ─────────────────────────────────────────────────────────────────────
    Layer total                                                   ≈ 3.146M
  16 layers                                                       ≈ 50.3M

  To reach ~110M we widen ff_dim to 4096:
    FFN per layer (512→4096→512)  : 512×4096 + 4096×512          ≈ 4.194M
    Attention per layer                                           ≈ 1.048M
    Layer total                                                   ≈ 5.242M
  16 layers + PositionalEncoding + misc                           ≈ 83.9M

  Widen further to ff_dim=6144:
    FFN per layer (512→6144→512)  : 2 × 512×6144                 ≈ 6.291M
    Attention per layer                                           ≈ 1.048M
    Layer total                                                   ≈ 7.339M
  16 layers                                                       ≈ 117.4M  ✓ ~110M
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["PositionalEncoding", "TransformerLayer", "TemporalTransformer"]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding supporting 3-D and 4-D inputs."""

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
            raise ValueError("dropout must be in [0.0, 1.0).")
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
                f"PositionalEncoding expects 3-D or 4-D input, got {tuple(inputs.shape)}."
            )
        seq_len = inputs.size(1)
        enc = self._get_encoding(seq_len, inputs.device, inputs.dtype)
        if inputs.ndim == 3:
            return self.dropout(inputs + enc.unsqueeze(0))
        return self.dropout(inputs + enc.view(1, seq_len, 1, self.embed_dim))

    def _get_encoding(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if seq_len > self.positional_encoding.size(0):  # type: ignore[attr-defined]
            self.positional_encoding = self._build_encoding(  # type: ignore[assignment]
                seq_len, self.embed_dim, device=self.positional_encoding.device  # type: ignore[attr-defined]
            )
        return self.positional_encoding[:seq_len].to(device=device, dtype=dtype)  # type: ignore[index]

    @staticmethod
    def _build_encoding(seq_len: int, embed_dim: int, device: Optional[torch.device] = None) -> Tensor:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        enc = torch.zeros(seq_len, embed_dim, device=device, dtype=torch.float32)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div[: enc[:, 1::2].shape[1]])
        return enc


class TransformerLayer(nn.Module):
    """Single temporal transformer encoder layer with SDPA (FlashAttention path)."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 6144,
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
            raise ValueError("dropout must be in [0.0, 1.0).")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
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
        is_causal: bool = False,
    ) -> Tensor:
        """Apply temporal self-attention (SDPA) followed by a feedforward network."""
        B, T, D = inputs.shape

        qkv = self.qkv_proj(inputs)  # (B, T, 3D)
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        # Reshape to (B, heads, T, head_dim) for SDPA
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Build key_padding additive mask for SDPA
        attn_bias: Optional[Tensor] = None
        if attention_mask is not None:
            # Convert bool causal mask to additive float mask
            attn_bias = torch.zeros(T, T, device=inputs.device, dtype=inputs.dtype)
            attn_bias = attn_bias.masked_fill(attention_mask, float("-inf"))
        if padding_mask is not None:
            # (B, T) → (B, 1, 1, T) additive
            pad_bias = torch.zeros(B, 1, 1, T, device=inputs.device, dtype=inputs.dtype)
            pad_bias = pad_bias.masked_fill(padding_mask.view(B, 1, 1, T), float("-inf"))
            attn_bias = (attn_bias + pad_bias) if attn_bias is not None else pad_bias

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(attn_out)

        inputs = self.attention_norm(inputs + self.attention_dropout(attn_out))
        ff_out = self.feedforward(inputs)
        return self.feedforward_norm(inputs + self.feedforward_dropout(ff_out))


class TemporalTransformer(nn.Module):
    """Temporal transformer encoder: 16 layers, ~110M parameters.

    embed_dim=512, ff_dim=6144, num_heads=8.
    """

    def __init__(
        self,
        num_layers: int = 16,
        num_heads: int = 8,
        embed_dim: int = 512,
        ff_dim: int = 6144,
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
            raise ValueError("dropout must be in [0.0, 1.0).")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_p = dropout

        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim, dropout=dropout, max_len=max_len
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
        """Encode temporal motion patterns for each agent independently.

        Args:
            inputs: ``(batch, time, agents, embed_dim)``
            padding_mask: Optional ``(batch, time)`` bool mask; True = padded.
            use_causal_mask: If True, mask future timesteps.

        Returns:
            ``(batch, time, agents, embed_dim)``
        """
        if inputs.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (batch, time, agents, embed_dim), got {tuple(inputs.shape)}."
            )
        batch_size, time_steps, num_agents, embed_dim = inputs.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {embed_dim}."
            )

        model_dtype = self.layers[0].qkv_proj.weight.dtype
        outputs = inputs.to(dtype=model_dtype)
        outputs = self.positional_encoding(outputs)

        # Flatten agents into batch: (B*N, T, D)
        outputs = outputs.permute(0, 2, 1, 3).reshape(batch_size * num_agents, time_steps, embed_dim)

        base_padding_mask = self._normalize_padding_mask(padding_mask, batch_size, time_steps, inputs.device)
        expanded_mask = self._expand_padding_mask(base_padding_mask, num_agents)

        if expanded_mask is not None:
            outputs = outputs.masked_fill(expanded_mask.unsqueeze(-1), 0.0)
            eff_mask = expanded_mask.clone()
            eff_mask[eff_mask.all(dim=1)] = False
        else:
            eff_mask = None

        attn_mask: Optional[Tensor] = None
        if use_causal_mask:
            attn_mask = torch.triu(
                torch.ones(time_steps, time_steps, device=inputs.device, dtype=torch.bool),
                diagonal=1,
            )

        for layer in self.layers:
            outputs = layer(
                outputs,
                padding_mask=eff_mask,
                attention_mask=attn_mask,
                is_causal=use_causal_mask and eff_mask is None and attn_mask is None,
            )
            if expanded_mask is not None:
                outputs = outputs.masked_fill(expanded_mask.unsqueeze(-1), 0.0)

        outputs = outputs.reshape(batch_size, num_agents, time_steps, embed_dim)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()

        if base_padding_mask is not None:
            outputs = outputs.masked_fill(base_padding_mask[:, :, None, None], 0.0)

        return outputs

    @staticmethod
    def _normalize_padding_mask(
        mask: Optional[Tensor], batch_size: int, time_steps: int, device: torch.device
    ) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.shape != (batch_size, time_steps):
            raise ValueError(f"padding_mask must be (batch, time), got {tuple(mask.shape)}.")
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _expand_padding_mask(mask: Optional[Tensor], num_agents: int) -> Optional[Tensor]:
        if mask is None:
            return None
        B, T = mask.shape
        return mask.unsqueeze(1).expand(B, num_agents, T).reshape(B * num_agents, T)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _run_smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model = TemporalTransformer().to(device=device, dtype=dtype)
    total = _count_parameters(model)
    print(f"TemporalTransformer parameters: {total:,}  (~{total / 1e6:.1f}M)  [target ~110M]")

    dummy = torch.randn(2, 10, 5, model.embed_dim, device=device, dtype=dtype)
    pad = torch.zeros(2, 10, dtype=torch.bool, device=device)
    pad[0, 8:] = True

    out = model(dummy, padding_mask=pad, use_causal_mask=True)
    assert out.shape == (2, 10, 5, model.embed_dim), out.shape
    print(f"Output shape: {tuple(out.shape)},  dtype: {out.dtype}")


if __name__ == "__main__":
    _run_smoke_test()