"""Building blocks of the CORTEX-0 Transformer.

Implemented from scratch: RMSNorm, rotary position embeddings, grouped-query
causal self-attention and a SwiGLU feed-forward network.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin tables of shape (max_seq_len, head_dim // 2)."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to ``x`` of shape (B, H, T, D).

    ``cos``/``sin`` have shape (T, D // 2) and are broadcast over batch/heads.
    """
    # Duplicate each frequency so cos/sin line up with the interleaved halves.
    cos = torch.cat((cos, cos), dim=-1)[None, None, :, :]
    sin = torch.cat((sin, sin), dim=-1)[None, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


class Attention(nn.Module):
    """Causal grouped-query attention with optional incremental KV cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        rope_theta: float = 10_000.0,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads or n_heads % n_kv_heads:
            raise ValueError("invalid head configuration")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)
        self.o_proj._cortex_residual = True  # type: ignore[attr-defined]

        cos, sin = precompute_rope(self.head_dim, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _shape(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, t, _ = x.shape
        past_len = 0 if past_kv is None else past_kv[0].shape[2]

        q = self._shape(self.q_proj(x), self.n_heads)
        k = self._shape(self.k_proj(x), self.n_kv_heads)
        v = self._shape(self.v_proj(x), self.n_kv_heads)

        cos = self.rope_cos[past_len : past_len + t]
        sin = self.rope_sin[past_len : past_len + t]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        present = (k, v) if use_cache else None

        # Expand KV heads to query heads (grouped-query attention).
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn_mask = None
        is_causal = past_kv is None
        if past_kv is not None and t > 1:
            kv_len = past_len + t
            mask = torch.ones(t, kv_len, dtype=torch.bool, device=x.device).tril(diagonal=past_len)
            attn_mask = mask

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out), present


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=bias)
        self.down_proj._cortex_residual = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def scaled_init(module: nn.Module, std: float, residual_depth: int | None = None) -> None:
    """Normal-initialise linear weights, optionally shrinking residual outputs."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.normal_(child.weight, mean=0.0, std=std)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
    if residual_depth:
        # GPT-2 style scaling of output projections that write into residuals.
        scale = 1.0 / math.sqrt(2 * residual_depth)
        for name in ("o_proj", "down_proj"):
            for child in module.modules():
                if isinstance(child, nn.Linear) and getattr(child, "_cortex_residual", False):
                    with torch.no_grad():
                        child.weight.mul_(scale)
