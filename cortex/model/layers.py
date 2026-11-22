"""Building blocks: RMSNorm, SwiGLU feed-forward, rotary position embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square layer normalisation (Zhang & Sennrich, 2019).

    The normalisation is computed in float32 for numerical stability, then cast
    back to the input dtype, matching the reference Llama implementation.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class SwiGLU(nn.Module):
    """Gated linear unit with SiLU activation, as used by Llama."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu") -> None:
        super().__init__()
        if hidden_act not in {"silu", "gelu"}:
            raise ValueError(f"unsupported hidden_act: {hidden_act}")
        self.hidden_act = hidden_act
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) if self.hidden_act == "silu" else F.gelu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self._act(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to a ``[batch, heads, seq, head_dim]`` tensor.

    ``cos``/``sin`` have shape ``[batch, 1, seq, head_dim]``.  The computation is
    done in float32 and cast back, which keeps long-context training stable.
    """
    dtype = tensor.dtype
    t = tensor.float()
    return (t * cos + rotate_half(t) * sin).to(dtype)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables with dynamic extension beyond max length."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("rotary dimension must be even")
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, inv_freq.device, torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_pos = int(position_ids.max().item()) + 1 if position_ids.numel() else 0
        if max_pos > self.max_seq_len_cached or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(
                max(max_pos, self.max_position_embeddings), x.device, self.cos_cached.dtype
            )
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        return cos, sin
