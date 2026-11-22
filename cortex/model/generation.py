"""Autoregressive sampling: temperature, top-k, top-p, repetition penalty."""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F

from .transformer import CortexForCausalLM


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Divide positive logits / multiply negative logits of seen tokens."""
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for batch_idx in range(generated.shape[0]):
        seen = torch.unique(generated[batch_idx])
        seen_logits = logits[batch_idx, seen]
        logits[batch_idx, seen] = torch.where(
            seen_logits > 0, seen_logits / penalty, seen_logits * penalty
        )
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set with cumulative prob >= p."""
    if p >= 1.0:
        return logits
    if p <= 0.0:
        raise ValueError("top_p must be in (0, 1]")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens once the cumulative probability exceeds p, always keeping
    # the first (highest probability) token.
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


@torch.no_grad()
def _iter_tokens(
    model: CortexForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: float,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    attention_mask: Optional[torch.Tensor],
    do_sample: bool,
    generator: Optional[torch.Generator],
    stop_token_ids: Optional[Sequence[int]],
) -> Iterator[torch.Tensor]:
    device = input_ids.device
    batch_size = input_ids.shape[0]
    max_context = model.config.max_position_embeddings
    if input_ids.shape[1] + max_new_tokens > max_context:
        raise ValueError(
            f"prompt ({input_ids.shape[1]} tokens) + max_new_tokens ({max_new_tokens}) "
            f"exceeds context length {max_context}"
        )
    stop_ids: Set[int] = set(stop_token_ids or [])
    if eos_token_id is not None:
        stop_ids.add(eos_token_id)

    sequences = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(sequences)
    past_key_values = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        model_input = sequences if past_key_values is None else sequences[:, -1:]
        outputs = model(
            model_input,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs["past_key_values"]
        logits = outputs["logits"][:, -1, :].float()

        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, sequences, repetition_penalty)

        greedy = (not do_sample) or temperature <= 0.0
        if greedy:
            next_tokens = torch.argmax(logits, dim=-1)
        else:
            logits = logits / temperature
            if top_k is not None:
                logits = top_k_filter(logits, top_k)
            if top_p is not None:
                logits = top_p_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

        if pad_token_id is not None:
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, pad_token_id), next_tokens
            )
        sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device),
            ],
            dim=-1,
        )
        for idx, token in enumerate(next_tokens.tolist()):
            if token in stop_ids:
                finished[idx] = True
        yield next_tokens
        if bool(finished.all()):
            break


@torch.no_grad()
def generate(
    model: CortexForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    do_sample: bool = True,
    generator: Optional[torch.Generator] = None,
    stop_token_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Generate tokens, returning the full sequence (prompt + new tokens).

    ``temperature <= 0`` (or ``do_sample=False``) selects greedy decoding.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    was_training = model.training
    model.eval()
    sequences = input_ids
    for next_tokens in _iter_tokens(
        model,
        input_ids,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        eos_token_id,
        pad_token_id,
        attention_mask,
        do_sample,
        generator,
        stop_token_ids,
    ):
        sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)
    if was_training:
        model.train()
    return sequences


@torch.no_grad()
def generate_stream(
    model: CortexForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    do_sample: bool = True,
    generator: Optional[torch.Generator] = None,
    stop_token_ids: Optional[Sequence[int]] = None,
) -> Iterator[torch.Tensor]:
    """Yield one ``[batch]`` tensor of new token ids per decoding step."""
    was_training = model.training
    model.eval()
    try:
        yield from _iter_tokens(
            model,
            input_ids,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            eos_token_id,
            pad_token_id,
            attention_mask,
            do_sample,
            generator,
            stop_token_ids,
        )
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def generate_text(
    model: CortexForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    **kwargs,
) -> str:
    """Convenience wrapper returning only the newly generated text."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.bos_token_id]
    input_ids = torch.tensor([ids], dtype=torch.long)
    out = generate(model, input_ids, max_new_tokens=max_new_tokens, **kwargs)
    new_ids = out[0, len(ids) :].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def list_stop_ids(tokenizer, names: Sequence[str] = ("<|end|>", "<|eos|>")) -> List[int]:
    ids = []
    for name in names:
        idx = tokenizer.token_to_id(name)
        if idx is not None:
            ids.append(idx)
    return ids
