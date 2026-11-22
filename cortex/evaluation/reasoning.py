"""Deterministic reasoning benchmarks with computed ground truth.

These suites are small and honest: they measure whether the model can follow
the arithmetic/logic format it was trained on.  They are *not* MMLU or GSM8K
and the results must never be reported as such.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from ..model.generation import generate
from ..model.transformer import CortexForCausalLM
from ..tokenizer.bpe import CortexTokenizer
from ..training.synthetic import arithmetic_examples, logic_examples

ANSWER_PATTERN = re.compile(r"(?:R[ée]ponse|Answer)\s*:\s*\**\s*([^\n*]+)", re.I)


def extract_answer(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip().strip(".,;:")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    lowered = text.strip().lower()
    for token in ("oui", "non", "yes", "no"):
        if re.search(rf"\b{token}\b", lowered):
            return token
    return ""


def normalize_answer(answer: str, lang: str = "fr") -> str:
    answer = answer.strip().lower().strip(".,;:!?\"'")
    answer = answer.replace(",", ".")
    mapping = {"yes": "oui", "no": "non", "true": "oui", "false": "non"}
    answer = mapping.get(answer, answer)
    try:
        value = float(answer)
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    except ValueError:
        return answer


def answers_match(predicted: str, expected: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(expected)


@torch.no_grad()
def _generate_completion(
    model: CortexForCausalLM,
    tokenizer: CortexTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    generator: Optional[torch.Generator],
    device: torch.device,
) -> str:
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
        stop_token_ids=[tokenizer.end_token_id],
        generator=generator,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0, len(ids) :].tolist(), skip_special_tokens=True)


def _evaluate_examples(
    model: CortexForCausalLM,
    tokenizer: CortexTokenizer,
    examples: Sequence[Dict[str, object]],
    lang: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    seed: int,
    device: torch.device,
    self_consistency: int = 1,
) -> Dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    correct = 0
    records: List[Dict[str, object]] = []
    started = time.time()
    for example in examples:
        prompt = f"Question: {example['question']}\n"
        samples: List[str] = []
        answers: List[str] = []
        for _ in range(max(1, self_consistency)):
            completion = _generate_completion(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if self_consistency > 1 else 0.0,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
                device=device,
            )
            samples.append(completion)
            answers.append(extract_answer(completion))
        if self_consistency > 1:
            counts: Dict[str, int] = {}
            for answer in answers:
                counts[normalize_answer(answer)] = counts.get(normalize_answer(answer), 0) + 1
            predicted = max(counts, key=counts.get) if counts else ""
        else:
            predicted = answers[0]
        is_correct = answers_match(predicted, str(example["answer"]))
        correct += int(is_correct)
        records.append(
            {
                "question": example["question"],
                "expected": example["answer"],
                "predicted": predicted,
                "correct": is_correct,
                "completion": samples[0][:600],
            }
        )
    elapsed = time.time() - started
    expected_counter: Dict[str, int] = {}
    for example in examples:
        key = normalize_answer(str(example["answer"]))
        expected_counter[key] = expected_counter.get(key, 0) + 1
    predicted_counter: Dict[str, int] = {}
    for record in records:
        key = normalize_answer(str(record["predicted"]))
        predicted_counter[key] = predicted_counter.get(key, 0) + 1
    accuracy = correct / max(1, len(examples))
    majority = max(expected_counter.values()) / max(1, len(examples))
    return {
        "n": len(examples),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "majority_baseline": round(majority, 4),
        "above_majority": round(accuracy - majority, 4),
        "expected_distribution": expected_counter,
        "predicted_distribution": predicted_counter,
        "lang": lang,
        "self_consistency": self_consistency,
        "seconds": round(elapsed, 2),
        "examples": records[:20],
    }


@torch.no_grad()
def evaluate_arithmetic(
    model: CortexForCausalLM,
    tokenizer: CortexTokenizer,
    n: int = 40,
    seed: int = 0,
    lang: str = "fr",
    max_new_tokens: int = 192,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    self_consistency: int = 1,
    device: str | torch.device = "cpu",
) -> Dict[str, object]:
    examples = arithmetic_examples(n, seed=seed, lang=lang)
    return _evaluate_examples(
        model, tokenizer, examples, lang, max_new_tokens, temperature, top_p, top_k,
        seed, torch.device(device), self_consistency
    )


@torch.no_grad()
def evaluate_logic(
    model: CortexForCausalLM,
    tokenizer: CortexTokenizer,
    n: int = 40,
    seed: int = 0,
    lang: str = "fr",
    max_new_tokens: int = 192,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    self_consistency: int = 1,
    device: str | torch.device = "cpu",
) -> Dict[str, object]:
    examples = logic_examples(n, seed=seed, lang=lang)
    return _evaluate_examples(
        model, tokenizer, examples, lang, max_new_tokens, temperature, top_p, top_k,
        seed, torch.device(device), self_consistency
    )


@torch.no_grad()
def generation_stats(
    model: CortexForCausalLM,
    tokenizer: CortexTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """Distinct-n and repetition rate over a small prompt set."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    all_tokens: List[int] = []
    outputs: List[str] = []
    for prompt in prompts:
        text = _generate_completion(
            model, tokenizer, prompt, max_new_tokens, temperature, top_p, None,
            generator, torch.device(device),
        )
        outputs.append(text)
        all_tokens.extend(tokenizer.encode(text))
    if not all_tokens:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_rate": 0.0}
    distinct_1 = len(set(all_tokens)) / len(all_tokens)
    bigrams = list(zip(all_tokens, all_tokens[1:]))
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams))
    repeats = sum(1 for a, b in bigrams if a == b)
    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "repetition_rate": round(repeats / max(1, len(bigrams)), 4),
        "samples": outputs[:5],
    }


def run_suite(
    model: CortexForCausalLM,
    tokenizer: CortexTokenizer,
    output_path: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    n: int = 40,
    seed: int = 0,
    max_new_tokens: int = 192,
) -> Dict[str, object]:
    results: Dict[str, object] = {
        "model_parameters": model.num_parameters(),
        "suite": "cortex-internal-v1",
        "max_new_tokens": max_new_tokens,
        "arithmetic_fr": evaluate_arithmetic(model, tokenizer, n=n, seed=seed, lang="fr", device=device, max_new_tokens=max_new_tokens),
        "arithmetic_en": evaluate_arithmetic(model, tokenizer, n=n, seed=seed + 1, lang="en", device=device, max_new_tokens=max_new_tokens),
        "logic_fr": evaluate_logic(model, tokenizer, n=n, seed=seed + 2, lang="fr", device=device, max_new_tokens=max_new_tokens),
        "logic_en": evaluate_logic(model, tokenizer, n=n, seed=seed + 3, lang="en", device=device, max_new_tokens=max_new_tokens),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results
