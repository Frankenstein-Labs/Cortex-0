"""Deterministic verifiers used by the reasoning engine.

Verifiers are intentionally not model-based: they give the engine a trustworthy
signal for a narrow class of problems (arithmetic) without pretending the LM is
a general critic.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Protocol

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}

_EQUATION_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷^])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def safe_eval(expr: str) -> float | None:
    """Evaluate a small arithmetic expression without ``eval``."""
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
    except SyntaxError:
        return None

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](visit(node.operand))
        raise ValueError("unsupported expression")

    try:
        value = visit(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


@dataclass
class Verification:
    ok: bool | None  # None = verifier cannot decide
    detail: str

    def as_dict(self) -> dict:
        return {"ok": self.ok, "detail": self.detail}


class Verifier(Protocol):
    name: str

    def check(self, problem: str, candidate: str) -> Verification: ...


class ArithmeticVerifier:
    """Verify arithmetic statements found in a candidate answer.

    It checks two things:
    1. every explicit ``a op b = c`` equation in the trace is correct;
    2. the final answer equals the value of the expression asked in the problem.
    """

    name = "arithmetic"

    def _expected(self, problem: str) -> float | None:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷^])\s*(-?\d+(?:\.\d+)?)", problem)
        if not match:
            return None
        a, op, b = match.groups()
        op = {"x": "*", "×": "*", "÷": "/"}.get(op, op)
        return safe_eval(f"{a} {op} {b}")

    def check(self, problem: str, candidate: str) -> Verification:
        expected = self._expected(problem)
        if expected is None:
            return Verification(None, "no arithmetic expression in problem")

        for a, op, b, c in _EQUATION_RE.findall(candidate):
            op = {"x": "*", "×": "*", "÷": "/"}.get(op, op)
            value = safe_eval(f"{a} {op} {b}")
            if value is None:
                continue
            if abs(value - float(c)) > 1e-6:
                return Verification(False, f"equation {a}{op}{b}={c} is false (expected {value:g})")

        answer = extract_answer(candidate)
        if answer is None:
            return Verification(None, "no final answer found")
        value = safe_eval(answer)
        if value is None:
            return Verification(None, "final answer is not numeric")
        if abs(value - expected) <= 1e-6:
            return Verification(True, f"matches expected value {expected:g}")
        return Verification(False, f"final answer {value:g} != expected {expected:g}")


class SelfConsistencyVerifier:
    """Agreement-based signal: a candidate is supported when it matches the mode."""

    name = "self_consistency"

    def __init__(self, answers: list[str | None]) -> None:
        self.mode = majority_answer(answers)

    def check(self, problem: str, candidate: str) -> Verification:
        if self.mode is None:
            return Verification(None, "no majority answer")
        answer = extract_answer(candidate)
        if answer is None:
            return Verification(None, "no final answer")
        ok = normalize_answer(answer) == self.mode
        return Verification(ok, f"mode={self.mode}")


def normalize_answer(answer: str) -> str:
    answer = answer.strip().lower()
    answer = re.sub(r"[\s,]+", "", answer)
    answer = re.sub(r"\.0+$", "", answer)
    return answer


def extract_answer(text: str) -> str | None:
    """Extract the final answer from a trace.

    Priority: explicit ``Answer:`` / ``=`` markers, otherwise the last number.
    """
    if not text:
        return None
    lower = text.lower()
    for marker in ("answer:", "final answer:", "réponse :", "réponse:", "result:", "resultat:", "résultat :"):
        idx = lower.rfind(marker)
        if idx != -1:
            tail = text[idx + len(marker) :].strip()
            tail = tail.splitlines()[0] if tail else tail
            tail = tail.strip().strip(".")
            if tail:
                return tail
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def majority_answer(answers: list[str | None]) -> str | None:
    from collections import Counter

    normalised = [normalize_answer(a) for a in answers if a is not None]
    if not normalised:
        return None
    return Counter(normalised).most_common(1)[0][0]
