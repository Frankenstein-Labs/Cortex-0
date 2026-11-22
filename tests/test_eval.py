from __future__ import annotations

from dataclasses import dataclass

from cortex.evaluation.reasoning_eval import evaluate_reasoning, make_problems, score_answer
from cortex.evaluation.suite import run_suite


def test_suite_passes_mechanism_checks(tmp_path):
    report = run_suite(tmp_path)
    failed = [c for c in report["checks"] if not c["ok"]]
    assert not failed, failed
    assert report["passed"] == report["total"]
    assert all(c["tag"] == "mechanism" for c in report["checks"])


@dataclass
class _Result:
    answer: str | None
    agreement: float = 1.0


class _StubEngine:
    def __init__(self, answers):
        self.answers = iter(answers)

    def solve(self, problem, strategy, n_samples, gen):
        return _Result(next(self.answers))


def test_reasoning_eval_scores_against_ground_truth():
    problems = make_problems(3, seed=2)
    engine = _StubEngine([p.answer for p in problems])
    report = evaluate_reasoning(engine, problems, strategies=("direct",), n_samples=1)
    assert report["strategies"]["direct"]["accuracy"] == 1.0

    engine = _StubEngine(["999"] * len(problems))
    report = evaluate_reasoning(engine, problems, strategies=("direct",), n_samples=1)
    assert report["strategies"]["direct"]["accuracy"] == 0.0


def test_score_answer_tolerance():
    assert score_answer("42", "42")
    assert score_answer("Answer: 42.", "42")
    assert not score_answer("41", "42")
