"""French capability evaluation with fixed, versioned items.

The items are deliberately simple and fixed so that runs are comparable across
checkpoints. Scores here are *measurements of a small model*, not claims that
CORTEX-0 understands French; see docs/EVALUATION.md.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

from ..model.generation import generate

# (question, substrings that must all appear in a correct answer)
IDENTITY_QUESTIONS: List[Tuple[str, List[str]]] = [
    ("Quel est ton nom ?", ["CORTEX-0"]),
    ("Qui t'a créé ?", ["Abdoulaye Coumbassa"]),
    ("Quelle organisation te développe ?", ["Frankenstein-Labs"]),
    ("D'où vient le projet ?", ["Guinée"]),
    ("Quelle est ta première langue ?", ["français"]),
    ("Quel est ton deuxième domaine ?", ["informatique"]),
    ("As-tu une voix ?", ["non"]),
    ("As-tu accès à Internet ?", ["non"]),
]

# (prefix, the word that a French speaker expects next)
GRAMMAR_COMPLETIONS: List[Tuple[str, str]] = [
    ("Le chat mange une", "souris"),
    ("Les enfants jouent dans le", "jardin"),
    ("Je vais à l'école pour", "apprendre"),
    ("Il fait beau, le ciel est", "bleu"),
    ("Nous avons besoin d'eau pour", "boire"),
    ("Le soleil se lève à l'", "est"),
    ("Elle écrit une lettre à son", "ami"),
    ("Les mathématiques demandent de la", "précision"),
    ("Un ordinateur exécute des", "programmes"),
    ("La mémoire conserve les", "informations"),
    ("Le train arrive à la", "gare"),
    ("Il lit un livre dans la", "bibliothèque"),
]

PROMPTS_FOR_HEALTH = [
    "Bonjour, je m'appelle",
    "Le français est une langue",
    "Pour écrire un programme Python, il faut",
    "Explique-moi ce qu'est un algorithme",
    "La capitale de la Guinée est",
]

FRENCH_LETTERS = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 40,
    seed: int = 1337,
) -> str:
    ids = tokenizer.encode(prompt)
    max_context = model.config.max_position_embeddings
    max_new = max(1, min(max_new_tokens, max_context - 1))
    ids = ids[-(max_context - max_new):]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    # multinomial requires the generator to live on the model's device.
    generator = torch.Generator(device=device).manual_seed(seed)
    output = generate(
        model,
        input_ids,
        max_new_tokens=max_new,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        stop_token_ids=[tokenizer.end_token_id],
        generator=generator,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True).strip()


def evaluate_identity(model, tokenizer, device: torch.device) -> Dict[str, object]:
    details = []
    correct = 0
    for question, required in IDENTITY_QUESTIONS:
        answer = _generate_text(model, tokenizer, question, device, max_new_tokens=40)
        lowered = answer.lower()
        hit = all(token.lower() in lowered for token in required)
        correct += int(hit)
        details.append({"question": question, "answer": answer, "correct": hit})
    return {
        "correct": correct,
        "total": len(IDENTITY_QUESTIONS),
        "accuracy": round(correct / max(1, len(IDENTITY_QUESTIONS)), 4),
        "details": details,
    }


def evaluate_completions(model, tokenizer, device: torch.device) -> Dict[str, object]:
    details = []
    correct = 0
    for prefix, expected in GRAMMAR_COMPLETIONS:
        continuation = _generate_text(model, tokenizer, prefix, device, max_new_tokens=8)
        first_words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", continuation)[:3]
        hit = expected.lower() in [w.lower() for w in first_words]
        correct += int(hit)
        details.append({"prefix": prefix, "expected": expected,
                        "continuation": continuation, "correct": hit})
    return {
        "correct": correct,
        "total": len(GRAMMAR_COMPLETIONS),
        "accuracy": round(correct / max(1, len(GRAMMAR_COMPLETIONS)), 4),
        "details": details,
    }


def generation_health(model, tokenizer, device: torch.device) -> Dict[str, object]:
    """Coarse signals: repetition, French character share, empty outputs."""
    samples = []
    repeated = 0
    french_share = []
    for prompt in PROMPTS_FOR_HEALTH:
        text = _generate_text(model, tokenizer, prompt, device, max_new_tokens=48)
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text)
        if len(words) >= 6:
            trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
            if len(set(trigrams)) / max(1, len(trigrams)) < 0.6:
                repeated += 1
        if text:
            letters = FRENCH_LETTERS.findall(text)
            french_share.append(len(letters) / max(1, len(text)))
        samples.append({"prompt": prompt, "output": text})
    return {
        "prompts": len(PROMPTS_FOR_HEALTH),
        "repetitive_outputs": repeated,
        "mean_letter_share": round(sum(french_share) / max(1, len(french_share)), 4),
        "samples": samples,
    }


def run_french_suite(model, tokenizer, device: torch.device) -> Dict[str, object]:
    model.eval()
    return {
        "identity": evaluate_identity(model, tokenizer, device),
        "grammar_completion": evaluate_completions(model, tokenizer, device),
        "generation_health": generation_health(model, tokenizer, device),
    }
