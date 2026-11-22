"""Autoregressive sampling: temperature, top-k, top-p and greedy decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.0
    seed: int | None = None
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row from ``logits`` shaped (batch, vocab)."""
    if logits.dim() != 2:
        raise ValueError("logits must be 2-D (batch, vocab)")
    logits = logits.float()

    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # Keep the first token that crosses top_p, then mask the rest.
        remove = cumulative - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    if generator is not None and generator.device != probs.device:
        generator = torch.Generator(device=probs.device)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def generate_text(
    model,
    tokenizer,
    prompt_ids: Sequence[int] | torch.Tensor,
    config: GenerationConfig,
    stop_strings: Sequence[str] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, list[int]]:
    """Generate text and return ``(decoded_text, all_token_ids)``.

    ``on_token`` receives incremental decoded text (best-effort; byte-level BPE
    can split multi-byte characters, so this is for UI streaming only).
    """
    device = next(model.parameters()).device
    if isinstance(prompt_ids, torch.Tensor):
        ids = prompt_ids.to(device)
    else:
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    out = model.generate(
        ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        eos_token_id=getattr(tokenizer, "end_id", None),
        use_cache=config.use_cache,
        seed=config.seed,
    )
    prompt_len = ids.shape[1]
    all_ids = out[0].tolist()
    text = tokenizer.decode(all_ids[prompt_len:])
    if "<|end|>" in text:
        text = text.split("<|end|>", 1)[0]
    if stop_strings:
        for stop in stop_strings:
            if stop in text:
                text = text.split(stop, 1)[0]
    text = text.strip()
    if on_token is not None and text:
        on_token(text)
    return text, all_ids
