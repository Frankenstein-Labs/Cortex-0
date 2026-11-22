"""Programmatic verifiers for reasoning traces.

A verifier is deliberately *not* another language model: it parses the trace
and checks arithmetic with a sandboxed AST evaluator, or compares the final
answer with a known ground truth.  This is what makes self-verification
measurable instead of decorative.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..evaluation.reasoning import extract_answer, normalize_answer

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _eval_node(node: ast.AST, depth: int = 0) -> float:
    if depth > 20:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval_node(node.operand, depth + 1)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def safe_arithmetic(expression: str) -> float:
    """Evaluate an arithmetic expression with an allow-listed AST."""
    expression = expression.replace("×", "*").replace("x", "*").replace("÷", "/")
    return _eval_node(ast.parse(expression, mode="eval"))


EQUATION_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def check_equations(text: str, tolerance: float = 1e-6) -> List[Tuple[str, bool]]:
    results = []
    for left, op, right, claimed in EQUATION_PATTERN.findall(text):
        expression = f"{left} {op} {right}"
        try:
            expected = safe_arithmetic(expression)
            ok = abs(expected - float(claimed)) <= tolerance
        except Exception:
            ok = False
        results.append((f"{expression} = {claimed}", ok))
    return results


@dataclass
class VerificationResult:
    verified: bool
    reason: str
    parsed: Optional[str] = None


class ArithmeticVerifier:
    name = "arithmetic"

    def verify(self, text: str, expected: Optional[str] = None) -> VerificationResult:
        answer = extract_answer(text)
        if not answer:
            return VerificationResult(False, "no answer marker found")
        try:
            value = float(normalize_answer(answer))
        except ValueError:
            return VerificationResult(False, f"answer is not numeric: {answer!r}", answer)
        if expected is not None:
            try:
                target = float(normalize_answer(str(expected)))
            except ValueError:
                return VerificationResult(False, f"invalid expected value: {expected!r}", answer)
            if abs(value - target) <= 1e-6:
                return VerificationResult(True, "matches ground truth", answer)
            return VerificationResult(False, f"expected {target}, got {value}", answer)
        equations = check_equations(text)
        if equations and not all(ok for _, ok in equations):
            bad = [expr for expr, ok in equations if not ok][:2]
            return VerificationResult(False, f"inconsistent equations: {bad}", answer)
        return VerificationResult(
            True,
            f"answer present; {len(equations)} equations checked",
            answer,
        )


class ExactMatchVerifier:
    name = "exact-match"

    def verify(self, text: str, expected: Optional[str] = None) -> VerificationResult:
        answer = extract_answer(text)
        if not answer:
            return VerificationResult(False, "no answer marker found")
        if expected is None:
            return VerificationResult(True, "answer present", answer)
        if normalize_answer(answer) == normalize_answer(str(expected)):
            return VerificationResult(True, "matches ground truth", answer)
        return VerificationResult(False, f"expected {expected}, got {answer}", answer)


def default_verifier(question: str):
    if re.search(r"\d", question):
        return ArithmeticVerifier()
    return ExactMatchVerifier()
