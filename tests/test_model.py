from __future__ import annotations

import pytest
import torch

from cortex.model.generation import generate, generate_stream, top_k_filter, top_p_filter


def test_forward_shapes(tiny_model):
    ids = torch.randint(0, tiny_model.config.vocab_size, (2, 8))
    out = tiny_model(input_ids=ids)
    assert out["logits"].shape == (2, 8, tiny_model.config.vocab_size)
    assert out["loss"] is None


def test_causality(tiny_model):
    torch.manual_seed(1)
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 12))
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % tiny_model.config.vocab_size
    with torch.no_grad():
        first = tiny_model(input_ids=ids)["logits"]
        second = tiny_model(input_ids=changed)["logits"]
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)
    assert not torch.allclose(first[:, -1], second[:, -1])


def test_kv_cache_equivalence(tiny_model):
    torch.manual_seed(2)
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 10))
    with torch.no_grad():
        full = tiny_model(input_ids=ids)["logits"]
        past = None
        pieces = []
        for index in range(ids.shape[1]):
            out = tiny_model(
                input_ids=ids[:, index : index + 1],
                past_key_values=past,
                use_cache=True,
            )
            past = out["past_key_values"]
            pieces.append(out["logits"])
        incremental = torch.cat(pieces, dim=1)
    assert torch.max(torch.abs(full - incremental)).item() < 1e-5


def test_attention_mask_ignores_padding(tiny_model):
    torch.manual_seed(3)
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 6))
    padded = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, 6, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)], dim=1)
    with torch.no_grad():
        clean = tiny_model(input_ids=ids)["logits"]
        masked = tiny_model(input_ids=padded, attention_mask=mask)["logits"]
    assert torch.allclose(clean, masked[:, :6], atol=1e-6)


def test_context_length_guard(tiny_model):
    ids = torch.zeros(1, tiny_model.config.max_position_embeddings + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        tiny_model(input_ids=ids)


def test_generate_is_deterministic_when_greedy(tiny_model):
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 5))
    first = generate(tiny_model, ids, max_new_tokens=6, do_sample=False)
    second = generate(tiny_model, ids, max_new_tokens=6, do_sample=False)
    assert first.shape == (1, 11)
    assert torch.equal(first, second)


def test_stream_matches_generate(tiny_model):
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 4))
    full = generate(tiny_model, ids, max_new_tokens=5, do_sample=False)
    pieces = list(generate_stream(tiny_model, ids, max_new_tokens=5, do_sample=False))
    assert torch.equal(full[:, 4:], torch.stack(pieces, dim=1))


def test_generation_budget_guard(tiny_model):
    ids = torch.zeros(1, tiny_model.config.max_position_embeddings, dtype=torch.long)
    with pytest.raises(ValueError):
        generate(tiny_model, ids, max_new_tokens=1)


def test_top_k_filter_keeps_k_tokens():
    logits = torch.randn(2, 20)
    filtered = top_k_filter(logits, 5)
    assert (filtered != float("-inf")).sum(dim=-1).tolist() == [5, 5]


def test_top_p_filter_keeps_nucleus():
    torch.manual_seed(4)
    logits = torch.randn(1, 50)
    filtered = top_p_filter(logits, 0.8)
    finite = torch.isfinite(filtered)
    assert finite.sum().item() >= 1
    kept_mass = torch.softmax(filtered, dim=-1).sum().item()
    assert kept_mass == pytest.approx(1.0, abs=1e-5)
    assert filtered[~finite].numel() == 50 - finite.sum().item()


def test_sampling_with_top_k_and_top_p(tiny_model):
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 3))
    generator = torch.Generator().manual_seed(7)
    out = generate(
        tiny_model,
        ids,
        max_new_tokens=4,
        temperature=0.9,
        top_k=10,
        top_p=0.9,
        generator=generator,
    )
    assert out.shape == (1, 7)
    assert out[:, 3:].max().item() < tiny_model.config.vocab_size
