"""Reasoning engine: multi-sample generation, voting, verification, revision.

This is a *software* reasoning layer around the language model. It can improve
answers when the base model produces useful traces; it cannot create knowledge
the model does not have. Every measured result must be reported as measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from cortex.generation.sampler import GenerationConfig, generate_text
from cortex.reasoning.verifiers import (
    ArithmeticVerifier,
    Verification,
    extract_answer,
    majority_answer,
    normalize_answer,
)

STRATEGIES = ("direct", "cot", "self_consistency", "verified")


@dataclass
class Candidate:
    text: str
    answer: str | None
    verification: Verification | None = None
    score: float = 0.0

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "answer": self.answer,
            "verification": self.verification.as_dict() if self.verification else None,
            "score": round(self.score, 4),
        }


@dataclass
class ReasoningResult:
    problem: str
    strategy: str
    answer: str | None
    candidates: list[Candidate] = field(default_factory=list)
    revised: Candidate | None = None
    agreement: float = 0.0

    def as_dict(self) -> dict:
        return {
            "problem": self.problem,
            "strategy": self.strategy,
            "answer": self.answer,
            "agreement": round(self.agreement, 4),
            "candidates": [c.as_dict() for c in self.candidates],
            "revised": self.revised.as_dict() if self.revised else None,
        }


COT_INSTRUCTION = (
    "Solve the problem step by step. Write each step on its own line. "
    "End with a final line that starts with 'Answer:'."
)


class ReasoningEngine:
    def __init__(self, model, tokenizer, verifiers: Sequence | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verifiers = list(verifiers) if verifiers is not None else [ArithmeticVerifier()]

    def _prompt(self, problem: str, instruction: str | None) -> str:
        content = f"{instruction}\n\nProblem: {problem}" if instruction else f"Problem: {problem}"
        return self.tokenizer.chat_prompt(
            [{"role": "user", "content": content}], add_generation_prompt=True
        )

    def _sample(self, problem: str, instruction: str | None, gen: GenerationConfig, seed_offset: int) -> Candidate:
        prompt = self._prompt(problem, instruction)
        ids = self.tokenizer.encode(prompt)
        if gen.seed is not None:
            gen = GenerationConfig(**{**gen.__dict__, "seed": gen.seed + seed_offset})
        text, _ = generate_text(self.model, self.tokenizer, ids, gen)
        return Candidate(text=text, answer=extract_answer(text))

    def _verify(self, problem: str, candidate: Candidate) -> None:
        best: Verification | None = None
        for verifier in self.verifiers:
            result = verifier.check(problem, candidate.text)
            if result.ok is True:
                best = result
                break
            if best is None or (best.ok is None and result.ok is not None):
                best = result
        candidate.verification = best
        if best and best.ok is True:
            candidate.score = 1.0
        elif best and best.ok is False:
            candidate.score = -1.0
        else:
            candidate.score = 0.0

    def solve(
        self,
        problem: str,
        strategy: str = "self_consistency",
        n_samples: int = 5,
        gen: GenerationConfig | None = None,
        reflect: bool = True,
    ) -> ReasoningResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
        gen = gen or GenerationConfig(max_new_tokens=96, temperature=0.8, top_k=50, top_p=0.95, seed=1234)

        if strategy == "direct":
            candidate = self._sample(problem, None, gen, 0)
            self._verify(problem, candidate)
            return ReasoningResult(problem, strategy, candidate.answer, [candidate], agreement=1.0)

        instruction = COT_INSTRUCTION
        samples = [self._sample(problem, instruction, gen, i) for i in range(max(1, n_samples))]
        for candidate in samples:
            self._verify(problem, candidate)

        mode = majority_answer([c.answer for c in samples])
        agreement = (
            sum(1 for c in samples if c.answer and normalize_answer(c.answer) == mode) / len(samples)
            if mode
            else 0.0
        )

        chosen: Candidate | None
        if strategy == "cot":
            chosen = samples[0]
        elif strategy == "self_consistency":
            chosen = next((c for c in samples if c.answer and normalize_answer(c.answer) == mode), samples[0])
        else:  # verified
            verified = [c for c in samples if c.verification and c.verification.ok is True]
            if verified:
                chosen = verified[0]
            else:
                chosen = next((c for c in samples if c.answer and normalize_answer(c.answer) == mode), samples[0])

        revised: Candidate | None = None
        if reflect and strategy in ("self_consistency", "verified"):
            revised = self._reflect(problem, chosen, gen)

        answer = revised.answer if revised and revised.answer else chosen.answer if chosen else None
        if revised and chosen and revised.verification and chosen.verification:
            if revised.verification.ok is True and chosen.verification.ok is not True:
                answer = revised.answer
            elif chosen.verification.ok is True and revised.verification.ok is not True:
                answer = chosen.answer
                revised = None
        return ReasoningResult(problem, strategy, answer, samples, revised, agreement)

    def _reflect(self, problem: str, candidate: Candidate, gen: GenerationConfig) -> Candidate | None:
        critique_prompt = (
            "Check the proposed solution. Find arithmetic or logical mistakes. "
            "If it is wrong, correct it. End with 'Answer:'.\n\n"
            f"Problem: {problem}\nProposed solution: {candidate.text}"
        )
        ids = self.tokenizer.encode(
            self.tokenizer.chat_prompt([{"role": "user", "content": critique_prompt}], add_generation_prompt=True)
        )
        text, _ = generate_text(self.model, self.tokenizer, ids, gen)
        revised = Candidate(text=text, answer=extract_answer(text))
        self._verify(problem, revised)
        return revised
