"""Multi-sample reasoning with verification, hypothesis comparison, revision."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..evaluation.reasoning import extract_answer, normalize_answer
from ..model.generation import generate
from ..model.transformer import CortexForCausalLM
from ..tokenizer.bpe import CortexTokenizer
from .verifier import ArithmeticVerifier, VerificationResult, default_verifier


@dataclass
class ReasoningConfig:
    num_samples: int = 4
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 192
    max_revisions: int = 1
    seed: int = 0
    prompt_prefix: str = "Question: "


@dataclass
class Candidate:
    text: str
    answer: str
    verified: bool
    reason: str
    mean_logprob: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "verified": self.verified,
            "reason": self.reason,
            "mean_logprob": round(self.mean_logprob, 5),
            "text": self.text[:1200],
        }


@dataclass
class ReasoningResult:
    question: str
    answer: str
    verified: bool
    confidence: float
    chosen: Candidate
    candidates: List[Candidate] = field(default_factory=list)
    revisions: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "verified": self.verified,
            "confidence": round(self.confidence, 4),
            "revisions": self.revisions,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "chosen": self.chosen.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
        }


@torch.no_grad()
def sequence_logprob(
    model: CortexForCausalLM,
    tokenizer: CortexTokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    max_length: int = 512,
) -> float:
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    if not completion_ids:
        return -100.0
    max_length = min(max_length, model.config.max_position_embeddings)
    ids = (prompt_ids + completion_ids)[-max_length:]
    prompt_len = min(len(prompt_ids), len(ids))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids)["logits"].float()
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]
    token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    mask = torch.arange(token_logprobs.numel(), device=device) >= max(0, prompt_len - 1)
    if not bool(mask.any()):
        return -100.0
    return float(token_logprobs[mask].mean())


class ReasoningEngine:
    """Best-of-N + verifier + optional revision.

    The engine never claims correctness on its own: ``verified`` is only true
    when a programmatic verifier accepts the trace.
    """

    def __init__(
        self,
        model: CortexForCausalLM,
        tokenizer: CortexTokenizer,
        config: Optional[ReasoningConfig] = None,
        device: str | torch.device = "cpu",
        store=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ReasoningConfig()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.store = store

    def _complete(
        self,
        prompt: str,
        temperature: float,
        generator: torch.Generator,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        ids = self.tokenizer.encode(prompt)
        limit = self.model.config.max_position_embeddings - max_new_tokens
        if limit <= 0:
            raise ValueError("max_new_tokens exceeds the model context length")
        ids = ids[-limit:]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = generate(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            do_sample=temperature > 0,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_token_ids=[self.tokenizer.end_token_id],
            generator=generator,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out[0, len(ids) :].tolist(), skip_special_tokens=True)

    def solve(
        self,
        question: str,
        expected: Optional[str] = None,
        config: Optional[ReasoningConfig] = None,
    ) -> ReasoningResult:
        cfg = config or self.config
        started = time.time()
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        prompt = f"{cfg.prompt_prefix}{question}\n"
        verifier = ArithmeticVerifier() if expected is not None and any(c.isdigit() for c in str(expected)) else default_verifier(question)
        if expected is not None and not any(c.isdigit() for c in str(expected)):
            from .verifier import ExactMatchVerifier

            verifier = ExactMatchVerifier()

        candidates: List[Candidate] = []
        for index in range(max(1, cfg.num_samples)):
            text = self._complete(
                prompt,
                temperature=cfg.temperature if cfg.num_samples > 1 else 0.0,
                generator=generator,
            )
            verification: VerificationResult = verifier.verify(text, expected)
            logprob = sequence_logprob(
                self.model, self.tokenizer, prompt, text, self.device
            )
            candidates.append(
                Candidate(
                    text=text,
                    answer=verification.parsed or extract_answer(text),
                    verified=verification.verified,
                    reason=verification.reason,
                    mean_logprob=logprob,
                )
            )

        chosen, revisions = self._choose(candidates)
        if not chosen.verified and cfg.max_revisions > 0:
            revision_prompt = (
                f"{prompt}{chosen.text}\n"
                f"Vérification: {chosen.reason}.\n"
                "Corrige le raisonnement et termine par « Réponse: X ».\n"
            )
            revised_text = self._complete(
                revision_prompt, temperature=0.3, generator=generator
            )
            verification = verifier.verify(revised_text, expected)
            revised = Candidate(
                text=revised_text,
                answer=verification.parsed or extract_answer(revised_text),
                verified=verification.verified,
                reason=verification.reason,
                mean_logprob=sequence_logprob(
                    self.model, self.tokenizer, revision_prompt, revised_text, self.device
                ),
            )
            candidates.append(revised)
            revisions = 1
            if revised.verified or not chosen.verified:
                chosen = revised

        confidence = self._confidence(candidates, chosen)
        result = ReasoningResult(
            question=question,
            answer=chosen.answer,
            verified=chosen.verified,
            confidence=confidence,
            chosen=chosen,
            candidates=candidates,
            revisions=revisions,
            elapsed_seconds=time.time() - started,
        )
        if self.store is not None and hasattr(self.store, "record_reasoning"):
            self.store.record_reasoning(result.to_dict())
        return result

    @staticmethod
    def _choose(candidates: List[Candidate]) -> tuple[Candidate, int]:
        verified = [c for c in candidates if c.verified]
        if verified:
            return max(verified, key=lambda c: c.mean_logprob), 0
        counts: Dict[str, int] = {}
        for candidate in candidates:
            key = normalize_answer(candidate.answer)
            counts[key] = counts.get(key, 0) + 1
        if counts:
            best_key = max(counts, key=counts.get)
            if counts[best_key] >= 2:
                pool = [c for c in candidates if normalize_answer(c.answer) == best_key]
                return max(pool, key=lambda c: c.mean_logprob), 0
        return max(candidates, key=lambda c: c.mean_logprob), 0

    @staticmethod
    def _confidence(candidates: List[Candidate], chosen: Candidate) -> float:
        if chosen.verified:
            return 1.0
        if not candidates:
            return 0.0
        agreement = sum(
            1 for c in candidates if normalize_answer(c.answer) == normalize_answer(chosen.answer)
        ) / len(candidates)
        return max(0.0, min(0.99, agreement * torch.exp(torch.tensor(chosen.mean_logprob)).item()))
