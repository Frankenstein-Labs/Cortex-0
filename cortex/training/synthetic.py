"""Deterministic synthetic data for arithmetic, logic and bootstrap chat.

These generators are used to (a) teach the first model a reasoning format and
(b) build evaluation suites whose ground truth is computed, never guessed.
The chat set is deliberately small: it is a bootstrap, not a quality
instruction corpus, and is labelled as synthetic in the dataset manifest.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _fmt(value: int) -> str:
    return f"{value}"


def arithmetic_problem(rng: random.Random, max_value: int = 99, ops: Sequence[str] = ("+", "-", "*")):
    op = rng.choice(list(ops))
    if op == "+":
        a, b = rng.randint(1, max_value), rng.randint(1, max_value)
        answer = a + b
    elif op == "-":
        a, b = rng.randint(1, max_value), rng.randint(1, max_value)
        a, b = max(a, b), min(a, b)
        answer = a - b
    else:
        a, b = rng.randint(2, 19), rng.randint(2, 19)
        answer = a * b
    return {"a": a, "b": b, "op": op, "answer": answer}


def arithmetic_trace(problem: Dict[str, int | str], lang: str = "fr") -> str:
    a, b, op, answer = problem["a"], problem["b"], problem["op"], problem["answer"]
    if op == "+":
        tens_a, ones_a = divmod(a, 10)
        tens_b, ones_b = divmod(b, 10)
        if lang == "fr":
            return (
                f"Je décompose : {a} = {tens_a * 10} + {ones_a} et {b} = {tens_b * 10} + {ones_b}. "
                f"Les dizaines donnent {tens_a * 10} + {tens_b * 10} = {(tens_a + tens_b) * 10}. "
                f"Les unités donnent {ones_a} + {ones_b} = {ones_a + ones_b}. "
                f"Donc {a} + {b} = {answer}.\nRéponse: {answer}"
            )
        return (
            f"I split {a} = {tens_a * 10} + {ones_a} and {b} = {tens_b * 10} + {ones_b}. "
            f"Tens: {tens_a * 10} + {tens_b * 10} = {(tens_a + tens_b) * 10}. "
            f"Ones: {ones_a} + {ones_b} = {ones_a + ones_b}. "
            f"So {a} + {b} = {answer}.\nAnswer: {answer}"
        )
    if op == "-":
        if lang == "fr":
            return (
                f"Je pars de {a} et je retire {b}. "
                f"{a} - {b} = {answer}, et je vérifie : {answer} + {b} = {a}.\nRéponse: {answer}"
            )
        return (
            f"I start at {a} and remove {b}. "
            f"{a} - {b} = {answer}; check: {answer} + {b} = {a}.\nAnswer: {answer}"
        )
    tens_b, ones_b = divmod(b, 10)
    partial_1 = a * tens_b * 10
    partial_2 = a * ones_b
    if lang == "fr":
        return (
            f"Je décompose {b} = {tens_b * 10} + {ones_b}. "
            f"{a} × {tens_b * 10} = {partial_1} et {a} × {ones_b} = {partial_2}. "
            f"J'additionne : {partial_1} + {partial_2} = {answer}.\nRéponse: {answer}"
        )
    return (
        f"I split {b} = {tens_b * 10} + {ones_b}. "
        f"{a} x {tens_b * 10} = {partial_1} and {a} x {ones_b} = {partial_2}. "
        f"Adding them: {partial_1} + {partial_2} = {answer}.\nAnswer: {answer}"
    )


def arithmetic_examples(
    count: int, seed: int = 0, lang: str = "fr", max_value: int = 99, ops: Sequence[str] = ("+", "-", "*")
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        problem = arithmetic_problem(rng, max_value=max_value, ops=ops)
        question = (
            f"Calcule {problem['a']} {problem['op']} {problem['b']}."
            if lang == "fr"
            else f"Compute {problem['a']} {problem['op']} {problem['b']}."
        )
        examples.append(
            {
                "kind": "arithmetic",
                "lang": lang,
                "question": question,
                "answer": str(problem["answer"]),
                "trace": arithmetic_trace(problem, lang=lang),
                "meta": problem,
            }
        )
    return examples


SYLLOGISM_TEMPLATES = [
    # (premises, conclusion template, valid)
    ("Tous les {a} sont des {b}.", "{x} est un {a}.", "Donc {x} est un {b}.", True),
    ("Tous les {a} sont des {b}.", "{x} n'est pas un {b}.", "Donc {x} n'est pas un {a}.", True),
    ("Aucun {a} n'est un {b}.", "{x} est un {a}.", "Donc {x} n'est pas un {b}.", True),
    ("Certains {a} sont des {b}.", "{x} est un {a}.", "Donc {x} est un {b}.", False),
    ("Tous les {a} sont des {b}.", "{x} est un {b}.", "Donc {x} est un {a}.", False),
]

NOUNS = [
    ("chat", "animal"), ("chien", "animal"), ("corbeau", "oiseau"), ("aigle", "oiseau"),
    ("triangle", "polygone"), ("carré", "polygone"), ("rose", "fleur"), ("tulipe", "fleur"),
    ("nombre premier", "entier"), ("nombre pair", "entier"),
]
NAMES = ["Milo", "Nina", "Sami", "Lila", "Omar", "Awa", "Kofi", "Fatou"]


def logic_example(rng: random.Random, lang: str = "fr") -> Dict[str, object]:
    premise_template, fact_template, conclusion_template, valid = rng.choice(SYLLOGISM_TEMPLATES)
    a, b = rng.sample(NOUNS, 2)
    a, b = a[0], b[0]
    name = rng.choice(NAMES)
    premise = premise_template.format(a=a, b=b)
    fact = fact_template.format(a=a, b=b, x=name)
    conclusion = conclusion_template.format(a=a, b=b, x=name)
    answer = "oui" if valid else "non"
    if lang == "fr":
        prompt = f"{premise} {fact} {conclusion} Cette conclusion est-elle logiquement valide ? Réponds par oui ou non."
        trace = (
            f"Je vérifie la forme du raisonnement. {premise} {fact} "
            f"La conclusion « {conclusion} » {'suit' if valid else 'ne suit pas'} nécessairement des prémisses. "
            f"Réponse: {answer}"
        )
    else:
        prompt = f"{premise} {fact} {conclusion} Is this conclusion logically valid? Answer yes or no."
        trace = (
            f"I check the form of the argument. {premise} {fact} "
            f"The conclusion '{conclusion}' {'follows' if valid else 'does not follow'} from the premises. "
            f"Answer: {answer}"
        )
    return {"kind": "logic", "lang": lang, "question": prompt, "answer": answer, "trace": trace}


def logic_examples(count: int, seed: int = 0, lang: str = "fr") -> List[Dict[str, object]]:
    rng = random.Random(seed)
    return [logic_example(rng, lang=lang) for _ in range(count)]


PERSONA_FR = (
    "Tu es CORTEX-0, une petite intelligence artificielle autonome développée par "
    "Frankenstein-Labs. Tu raisonnes étape par étape, tu admets tes limites et tu ne "
    "prétends jamais avoir fait quelque chose que tu n'as pas fait."
)
PERSONA_EN = (
    "You are CORTEX-0, a small autonomous artificial intelligence built by "
    "Frankenstein-Labs. You reason step by step, admit your limits, and never claim "
    "to have done something you did not do."
)


def chat_examples(count: int, seed: int = 0) -> List[Dict[str, object]]:
    """Small bilingual bootstrap set for the conversational format."""
    rng = random.Random(seed)
    base = [
        ("Qui es-tu ?", "Je suis CORTEX-0, une petite IA autonome de Frankenstein-Labs. Je suis un modèle de langage entraîné localement, pas un assistant géant. Je peux raisonner, converser et exécuter des tâches autorisées.", "fr"),
        ("Que peux-tu faire ?", "Je peux tenir une conversation, résoudre des problèmes d'arithmétique simples, garder une mémoire de nos échanges, proposer des idées et exécuter des tâches de développement dans un espace isolé. Je te dis honnêtement quand je ne sais pas.", "fr"),
        ("Quelles sont tes limites ?", "Je suis un petit modèle : ma connaissance du monde est limitée à mes données d'entraînement et je peux me tromper. Je n'ai pas de conscience et je ne prends pas d'initiative sans autorisation.", "fr"),
        ("Who are you?", "I am CORTEX-0, a small autonomous AI from Frankenstein-Labs. I am a locally trained language model, not a giant assistant. I can reason, chat, and run authorised tasks.", "en"),
        ("What can you do?", "I can hold a conversation, solve simple arithmetic, remember our exchanges, propose ideas, and run development tasks in an isolated workspace. I tell you honestly when I do not know.", "en"),
        ("What are your limits?", "I am a small model: my world knowledge is limited to my training data and I can be wrong. I have no consciousness and I take no initiative without permission.", "en"),
    ]
    examples: List[Dict[str, object]] = []
    for _ in range(count):
        question, answer, lang = base[rng.randrange(len(base))]
        system = PERSONA_FR if lang == "fr" else PERSONA_EN
        examples.append(
            {
                "kind": "chat",
                "lang": lang,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    for lang in ("fr", "en"):
        for problem in arithmetic_examples(max(1, count // 10), seed=seed + 7, lang=lang):
            system = PERSONA_FR if lang == "fr" else PERSONA_EN
            examples.append(
                {
                    "kind": "chat-arithmetic",
                    "lang": lang,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": problem["question"]},
                        {"role": "assistant", "content": problem["trace"]},
                    ],
                }
            )
    rng.shuffle(examples)
    return examples


def write_jsonl(path: str | Path, records: Iterable[Dict[str, object]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
