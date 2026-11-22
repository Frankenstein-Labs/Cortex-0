from __future__ import annotations

import torch

from cortex.generation.sampler import GenerationConfig, generate_text, sample_next_token


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2, -1.0]])
    assert int(sample_next_token(logits, temperature=0.0)) == 1


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, 0.0, -10.0]])
    gen = torch.Generator().manual_seed(0)
    seen = {int(sample_next_token(logits, temperature=1.0, top_k=2, generator=gen)) for _ in range(100)}
    assert seen <= {0, 1}


def test_top_p_restricts_support():
    logits = torch.tensor([[10.0, 9.0, -10.0, -10.0]])
    gen = torch.Generator().manual_seed(1)
    seen = {int(sample_next_token(logits, temperature=1.0, top_p=0.5, generator=gen)) for _ in range(100)}
    assert seen == {0}


def test_seeded_sampling_reproducible():
    logits = torch.randn(2, 50)
    a = sample_next_token(logits, temperature=0.9, top_k=10, generator=torch.Generator().manual_seed(7))
    b = sample_next_token(logits, temperature=0.9, top_k=10, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_generation_config_validation():
    import pytest

    with pytest.raises(ValueError):
        GenerationConfig(top_p=0)
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-1)
    with pytest.raises(ValueError):
        GenerationConfig(top_k=0)


def test_model_generate_and_cache_equivalence(tiny_model):
    tiny_model.eval()
    ids = torch.randint(0, tiny_model.cfg.vocab_size, (1, 5))
    a = tiny_model.generate(ids, max_new_tokens=8, temperature=0.0, use_cache=True)
    b = tiny_model.generate(ids, max_new_tokens=8, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)
    assert a.shape[1] == 13


def test_generate_text_strips_prompt(tiny_model, tiny_tokenizer):
    ids = tiny_tokenizer.encode("Once upon a time")
    text, all_ids = generate_text(
        tiny_model,
        tiny_tokenizer,
        ids,
        GenerationConfig(max_new_tokens=6, temperature=0.0, seed=1),
    )
    assert isinstance(text, str)
    assert len(all_ids) >= len(ids)


def test_repetition_penalty_applied(tiny_model):
    from cortex.model.lm import apply_repetition_penalty

    logits = torch.tensor([[5.0, 4.0, 3.0]])
    ids = torch.tensor([[0, 0, 0]])
    out = apply_repetition_penalty(logits, ids, 2.0)
    assert out[0, 0] < logits[0, 0]
    assert out[0, 1] == logits[0, 1]
