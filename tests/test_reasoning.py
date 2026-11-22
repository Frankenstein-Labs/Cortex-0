from __future__ import annotations

import pytest

from cortex.evaluation.reasoning_eval import make_problems, score_answer
from cortex.generation.sampler import GenerationConfig
from cortex.reasoning.engine import ReasoningEngine
from cortex.reasoning.verifiers import (
    ArithmeticVerifier,
    SelfConsistencyVerifier,
    extract_answer,
    majority_answer,
    normalize_answer,
    safe_eval,
)


def test_safe_eval():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(2 + 3) * 4") == 20
    assert safe_eval("2 ^ 5") == 32
    assert safe_eval("__import__('os')") is None
    assert safe_eval("1 / 0") is None


def test_extract_answer():
    assert extract_answer("so the answer is 42") == "42"
    assert extract_answer("Reasoning...\nAnswer: 17") == "17"
    assert extract_answer("Réponse : 9") == "9"
    assert extract_answer("no number here") is None


def test_majority_and_normalisation():
    assert normalize_answer(" 42.0 ") == "42"
    assert majority_answer(["7", "7.0", "8", None]) == "7"


def test_arithmetic_verifier():
    v = ArithmeticVerifier()
    assert v.check("What is 12 + 7?", "12 + 7 = 19\nAnswer: 19").ok is True
    assert v.check("What is 12 + 7?", "12 + 7 = 20\nAnswer: 20").ok is False
    assert v.check("What is 12 + 7?", "I do not know").ok is None
    assert v.check("Tell me a story", "Answer: 3").ok is None


def test_self_consistency_verifier():
    v = SelfConsistencyVerifier(["5", "5", "6"])
    assert v.check("q", "Answer: 5").ok is True
    assert v.check("q", "Answer: 6").ok is False


def test_reasoning_engine_structure(tiny_model, tiny_tokenizer):
    engine = ReasoningEngine(tiny_model, tiny_tokenizer)
    gen = GenerationConfig(max_new_tokens=8, temperature=0.0, seed=1)
    result = engine.solve("What is 2 + 3?", strategy="direct", gen=gen)
    assert result.strategy == "direct"
    assert len(result.candidates) == 1
    payload = result.as_dict()
    assert payload["problem"] == "What is 2 + 3?"


def test_reasoning_engine_multi_sample(tiny_model, tiny_tokenizer):
    engine = ReasoningEngine(tiny_model, tiny_tokenizer)
    gen = GenerationConfig(max_new_tokens=6, temperature=0.0, seed=1)
    result = engine.solve("What is 4 + 5?", strategy="self_consistency", n_samples=2, gen=gen, reflect=False)
    assert len(result.candidates) == 2
    assert 0.0 <= result.agreement <= 1.0


def test_reasoning_engine_rejects_unknown_strategy(tiny_model, tiny_tokenizer):
    engine = ReasoningEngine(tiny_model, tiny_tokenizer)
    with pytest.raises(ValueError):
        engine.solve("2+2", strategy="magic")


def test_problem_generation_and_scoring():
    problems = make_problems(10, seed=1)
    assert len(problems) == 10
    for p in problems:
        assert p.answer.lstrip("-").isdigit()
    assert score_answer("19", "19")
    assert score_answer("The answer is 19.", "19")
    assert not score_answer("20", "19")
    assert not score_answer(None, "19")
