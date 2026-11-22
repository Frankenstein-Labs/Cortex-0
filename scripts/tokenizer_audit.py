#!/usr/bin/env python3
"""Measure how well a CORTEX-0 tokenizer handles French and source code.

Produces a JSON report with real measurements so that a vocabulary change can be
justified by evidence rather than intuition. Nothing here modifies the
tokenizer.

Usage:
  python scripts/tokenizer_audit.py \
      --tokenizer-dir datasets/processed/tokenizer \
      --corpus datasets/french/corpus.jsonl \
      --out experiments/tokenizer-audit-4096.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.tokenizer import CortexTokenizer  # noqa: E402

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?")
ACCENT_RE = re.compile(r"[àâäéèêëîïôöùûüçœæÀÂÄÉÈÊËÎÏÔÖÙÛÜÇŒÆ]")


def sample_texts(corpus: Path, per_category: int, limit_chars: int) -> Dict[str, List[str]]:
    """Take a deterministic sample per category from the corpus JSONL."""
    buckets: Dict[str, List[str]] = {}
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            category = record["category"]
            bucket = buckets.setdefault(category, [])
            if len(bucket) < per_category:
                bucket.append(record["text"][:limit_chars])
            if all(len(v) >= per_category for v in buckets.values()) and len(buckets) >= 7:
                break
    return buckets


def measure(tokenizer: CortexTokenizer, texts: List[str]) -> Dict[str, float]:
    words: Counter = Counter()
    split_words = 0
    total_words = 0
    tokens_total = 0
    chars_total = 0
    accented_words = 0
    accented_words_split = 0

    for text in texts:
        ids = tokenizer.encode(text)
        tokens_total += len(ids)
        chars_total += len(text)
        for word in WORD_RE.findall(text):
            total_words += 1
            words[word] += 1
            pieces = tokenizer.encode(word)
            if len(pieces) > 1:
                split_words += 1
                if ACCENT_RE.search(word):
                    accented_words_split += 1
            if ACCENT_RE.search(word):
                accented_words += 1

    rare = [w for w, c in words.items() if c == 1]
    rare_split = sum(1 for w in rare if len(tokenizer.encode(w)) > 1)

    return {
        "documents": len(texts),
        "chars": chars_total,
        "tokens": tokens_total,
        "chars_per_token": round(chars_total / max(1, tokens_total), 3),
        "words": total_words,
        "tokens_per_word": round(tokens_total / max(1, total_words), 3),
        "split_word_rate": round(split_words / max(1, total_words), 4),
        "accented_words": accented_words,
        "accented_split_rate": round(accented_words_split / max(1, accented_words), 4),
        "rare_words": len(rare),
        "rare_split_rate": round(rare_split / max(1, len(rare)), 4),
        "unique_words": len(words),
    }


def code_measure(tokenizer: CortexTokenizer, paths: List[Path]) -> Dict[str, float]:
    tokens = chars = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens += len(tokenizer.encode(text))
        chars += len(text)
    return {
        "files": len(paths),
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": round(chars / max(1, tokens), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", default="datasets/processed/tokenizer")
    parser.add_argument("--corpus", default="datasets/french/corpus.jsonl")
    parser.add_argument("--out", default="experiments/tokenizer-audit-4096.json")
    parser.add_argument("--per-category", type=int, default=40)
    parser.add_argument("--limit-chars", type=int, default=4000)
    args = parser.parse_args()

    tokenizer = CortexTokenizer.from_pretrained(ROOT / args.tokenizer_dir)
    corpus = ROOT / args.corpus
    report: Dict[str, object] = {
        "tokenizer_dir": args.tokenizer_dir,
        "vocab_size": tokenizer.vocab_size,
        "per_category": {},
    }

    if corpus.exists():
        buckets = sample_texts(corpus, args.per_category, args.limit_chars)
        for category, texts in sorted(buckets.items()):
            report["per_category"][category] = measure(tokenizer, texts)  # type: ignore[index]
        all_texts = [t for texts in buckets.values() for t in texts]
        report["overall_french"] = measure(tokenizer, all_texts)
    else:
        report["corpus_missing"] = str(corpus)

    code_paths = sorted((ROOT / "cortex").rglob("*.py"))
    report["code_python"] = code_measure(tokenizer, code_paths)

    # Character-level coverage: which frequent French characters are single tokens?
    alphabet = "abcdefghijklmnopqrstuvwxyzàâäéèêëîïôöùûüçœæ'’-\n"
    single = {}
    for char in alphabet:
        pieces = tokenizer.encode(char)
        single[char] = len(pieces) == 1
    report["single_char_tokens"] = {
        "covered": sum(1 for v in single.values() if v),
        "total": len(single),
        "missing": [c for c, ok in single.items() if not ok],
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[audit] wrote {out}")


if __name__ == "__main__":
    main()
