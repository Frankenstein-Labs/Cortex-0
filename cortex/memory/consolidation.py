"""Extractive consolidation of episodic memory into semantic memory.

This is intentionally extractive: the small model is not trusted to invent
facts about the user.  A future version can use the LLM to summarise, but the
current method is deterministic and auditable.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Tuple

from .retrieval import STOPWORDS, tokenize

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

PREFERENCE_PATTERNS = [
    re.compile(r"(?:je préfère|j'aime mieux|j'aime|je veux)\s+([^.!?\n]{3,160})", re.I),
    re.compile(r"(?:rappelle-toi que|souviens-toi que|retiens que)\s+([^.!?\n]{3,200})", re.I),
    re.compile(r"(?:I prefer|I like|I want)\s+([^.!?\n]{3,160})", re.I),
    re.compile(r"(?:remember that|keep in mind that)\s+([^.!?\n]{3,200})", re.I),
]


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) >= 12]


def extractive_summary(texts: Iterable[str], max_sentences: int = 6, max_chars: int = 900) -> str:
    sentences: List[str] = []
    for text in texts:
        sentences.extend(split_sentences(text))
    if not sentences:
        return ""
    frequency = Counter()
    for sentence in sentences:
        for token in tokenize(sentence):
            if token not in STOPWORDS and len(token) > 2:
                frequency[token] += 1
    scored = []
    for index, sentence in enumerate(sentences):
        tokens = [t for t in tokenize(sentence) if t in frequency]
        if not tokens:
            continue
        score = sum(frequency[t] for t in tokens) / (len(tokens) ** 0.5)
        scored.append((score, index, sentence))
    scored.sort(reverse=True)
    chosen = sorted(scored[:max_sentences], key=lambda item: item[1])
    summary = " ".join(sentence for _, _, sentence in chosen)
    return summary[:max_chars].strip()


def extract_preferences(texts: Iterable[str]) -> List[str]:
    preferences: List[str] = []
    seen = set()
    for text in texts:
        for pattern in PREFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                statement = match.group(0).strip()
                key = statement.lower()
                if key not in seen:
                    seen.add(key)
                    preferences.append(statement)
    return preferences


def summarize_turns(turns: Iterable[Tuple[str, str]], max_chars: int = 700) -> str:
    lines = [f"{role}: {content}" for role, content in turns if content.strip()]
    return extractive_summary(lines, max_sentences=8, max_chars=max_chars)
