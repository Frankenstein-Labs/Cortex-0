"""Procedurally generated arithmetic benchmark for the reasoning engine.

The benchmark is deterministic and its ground truth is computed, not invented.
Results must be reported with the exact configuration (model, samples, steps).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from cortex.generation.sampler import GenerationConfig
from cortex.reasoning.verifiers import normalize_answer


@dataclass
class Problem:
    text: str
    answer: str


def make_problems(n: int = 20, seed: int = 7, max_value: int = 20) -> list[Problem]:
    rng = random.Random(seed)
    problems: list[Problem] = []
    for _ in range(n):
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            a, b = rng.randint(2, 9), rng.randint(2, 9)
        else:
            a, b = rng.randint(1, max_value), rng.randint(1, max_value)
            if op == "-" and b > a:
                a, b = b, a
        value = {"+": a + b, "-": a - b, "*": a * b}[op]
        problems.append(Problem(f"What is {a} {op} {b}?", str(value)))
    return problems


def score_answer(predicted: str | None, expected: str) -> bool:
    if predicted is None:
        return False
    pred = normalize_answer(predicted)
    exp = normalize_answer(expected)
    if pred == exp:
        return True
    numbers = [normalize_answer(n) for n in re.findall(r"-?\d+(?:\.\d+)?", pred)]
    return exp in numbers


def evaluate_reasoning(
    engine,
    problems: list[Problem],
    strategies: tuple[str, ...] = ("direct", "cot", "self_consistency", "verified"),
    n_samples: int = 3,
    gen: GenerationConfig | None = None,
) -> dict:
    gen = gen or GenerationConfig(max_new_tokens=64, temperature=0.8, top_k=50, top_p=0.95, seed=1234)
    report: dict = {"n_problems": len(problems), "n_samples": n_samples, "strategies": {}}
    for strategy in strategies:
        correct = 0
        details = []
        for problem in problems:
            result = engine.solve(problem.text, strategy=strategy, n_samples=n_samples, gen=gen)
            ok = score_answer(result.answer, problem.answer)
            correct += int(ok)
            details.append(
                {
                    "problem": problem.text,
                    "expected": problem.answer,
                    "predicted": result.answer,
                    "correct": ok,
                    "agreement": round(result.agreement, 3),
                }
            )
        report["strategies"][strategy] = {
            "correct": correct,
            "accuracy": round(correct / max(1, len(problems)), 4),
            "details": details,
        }
    return report
