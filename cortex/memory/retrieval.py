"""Lexical and hashed-embedding retrieval utilities (no external model)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List

import numpy as np

WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à", "au", "aux",
    "en", "dans", "pour", "par", "sur", "avec", "sans", "que", "qui", "quoi", "dont",
    "est", "sont", "être", "avoir", "je", "tu", "il", "elle", "nous", "vous", "ils",
    "elles", "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son",
    "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs", "se", "ne", "pas",
    "plus", "moins", "très", "tout", "tous", "toute", "toutes", "comme", "mais",
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "without",
    "that", "which", "who", "what", "is", "are", "be", "have", "has", "i", "you",
    "he", "she", "we", "they", "it", "this", "these", "those", "my", "your", "his",
    "her", "our", "their", "not", "no", "very", "all", "as", "but", "at", "by",
    "from", "so", "if", "then", "than", "too", "can", "will", "would", "should",
}


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def content_tokens(text: str) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2]


class HashingEmbedder:
    """Deterministic signed feature hashing; stable across processes and runs."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    @staticmethod
    def to_bytes(vector: np.ndarray) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def from_bytes(payload: bytes, dim: int = 256) -> np.ndarray:
        if not payload:
            return np.zeros(dim, dtype=np.float32)
        return np.frombuffer(payload, dtype=np.float32).copy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def recency_score(age_seconds: float, half_life_seconds: float) -> float:
    if age_seconds <= 0:
        return 1.0
    return math.exp(-age_seconds * math.log(2) / max(1.0, half_life_seconds))


def lexical_overlap(query: str, text: str) -> float:
    query_tokens = set(content_tokens(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(content_tokens(text))
    return len(query_tokens & text_tokens) / len(query_tokens)


def jaccard(a: str, b: str) -> float:
    left, right = set(content_tokens(a)), set(content_tokens(b))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def dedupe_texts(texts: Iterable[str], threshold: float = 0.85) -> List[str]:
    kept: List[str] = []
    for text in texts:
        if all(jaccard(text, existing) < threshold for existing in kept):
            kept.append(text)
    return kept
