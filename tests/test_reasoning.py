from __future__ import annotations

import torch

from cortex.evaluation.reasoning import extract_answer, normalize_answer
from cortex.reasoning.engine import ReasoningConfig, ReasoningEngine
from cortex.reasoning.grpo import group_advantages, grpo_loss, sequence_logprobs
from cortex.reasoning.verifier import ArithmeticVerifier, check_equations, safe_arithmetic


def test_safe_arithmetic_and_equations():
    assert safe_arithmetic("2 + 3 * 4") == 14
    assert safe_arithmetic("(10 - 2) / 4") == 2
    checks = check_equations("2 + 2 = 4 et 3 * 3 = 10")
    assert checks[0][1] is True
    assert checks[1][1] is False


def test_arithmetic_verifier():
    verifier = ArithmeticVerifier()
    assert verifier.verify("Réponse : 42", expected="42").verified
    assert not verifier.verify("Réponse : 41", expected="42").verified
    assert not verifier.verify("aucune réponse", expected="42").verified


def test_answer_normalization():
    assert normalize_answer("42.0") == "42"
    assert normalize_answer("oui") == "oui"
    assert extract_answer("Raisonnement... Réponse : 17") == "17"


def test_group_advantages_are_normalised():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    advantages = group_advantages(rewards, group_size=2)
    assert advantages.shape == rewards.shape
    assert abs(float(advantages[:2].mean())) < 1e-5
    assert float(advantages[0]) > 0 > float(advantages[1])


def test_grpo_loss_with_identical_policy():
    logprobs = torch.zeros(4, 5)
    old = torch.zeros(4, 5)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(4, 5)
    result = grpo_loss(logprobs, old, rewards, group_size=2, mask=mask)
    assert torch.isfinite(result["loss"])
    assert float(result["mean_ratio"]) == 1.0
    assert float(result["clip_fraction"]) == 0.0


def test_sequence_logprobs_masks_prompt(tiny_model):
    ids = torch.randint(0, tiny_model.config.vocab_size, (2, 8))
    mask = torch.zeros_like(ids)
    mask[:, 4:] = 1
    logprobs = sequence_logprobs(tiny_model, ids, mask)
    assert logprobs.shape == (2, 7)
    assert torch.all(logprobs[:, :3] == 0)
    assert torch.isfinite(logprobs[:, 3:]).all()


def test_reasoning_engine_records_runs(tiny_model, tokenizer, store):
    engine = ReasoningEngine(
        tiny_model,
        tokenizer,
        ReasoningConfig(num_samples=2, max_new_tokens=8, temperature=0.8, seed=0),
        store=store,
    )
    result = engine.solve("Combien font 2 + 2 ?", expected="4")
    assert len(result.candidates) >= 2
    assert isinstance(result.verified, bool)
    assert isinstance(result.answer, str)
    runs = store.reasoning_runs(limit=5)
    assert len(runs) == 1
    assert runs[0]["question"].startswith("Combien")
