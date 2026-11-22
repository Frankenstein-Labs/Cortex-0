#!/usr/bin/env python3
"""Build the CORTEX-0 corpus, tokenizer and token files.

Sources are public-domain books (Project Gutenberg mirror), Tiny Shakespeare,
French Wikipedia abstracts (Hugging Face datasets server), the project's own
source code, and deterministic synthetic reasoning data.  Every source is
recorded with a SHA-256 in ``datasets/processed/manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.tokenizer.bpe import CortexTokenizer  # noqa: E402
from cortex.training.data import tokenize_documents  # noqa: E402
from cortex.training.synthetic import (  # noqa: E402
    arithmetic_examples,
    chat_examples,
    logic_examples,
    write_jsonl,
)

BOOKS = [
    ("les_miserables_1", "fr", "https://gutenberg.pglaf.org/cache/epub/17489/pg17489.txt"),
    ("vingt_mille_lieues", "fr", "https://gutenberg.pglaf.org/cache/epub/5097/pg5097.txt"),
    ("monte_cristo_1", "fr", "https://gutenberg.pglaf.org/cache/epub/17989/pg17989.txt"),
    ("frankenstein", "en", "https://gutenberg.pglaf.org/cache/epub/84/pg84.txt"),
]
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
USER_AGENT = "CORTEX-0-dataset-builder/0.1 (Frankenstein-Labs)"


def log(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def fetch(url: str, destination: Path, retries: int = 3) -> Path:
    if destination.exists() and destination.stat().st_size > 1000:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if len(data) < 1000:
                raise ValueError(f"suspiciously small download ({len(data)} bytes)")
            destination.write_bytes(data)
            return destination
        except Exception as exc:  # network errors are expected and retried
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def decode_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I)
    end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I)
    if start:
        text = text[start.end() :]
    if end:
        text = text[: end.start()]
    return text


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = 4000) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for paragraph in paragraphs:
        if length + len(paragraph) + 2 > size and current:
            chunks.append("\n\n".join(current))
            current, length = [], 0
        current.append(paragraph)
        length += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def wikipedia_french(rows: int = 2200, batch: int = 100) -> List[str]:
    docs: List[str] = []
    for offset in range(0, rows, batch):
        query = urllib.parse.urlencode(
            {
                "dataset": "wikimedia/wikipedia",
                "config": "20231101.fr",
                "split": "train",
                "offset": offset,
                "length": batch,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            log(f"wikipedia fetch stopped at offset {offset}: {exc}")
            break
        for row in payload.get("rows", []):
            text = normalize(row.get("row", {}).get("text", ""))
            if len(text) >= 500:
                docs.append(text[:8000])
        if offset % 500 == 0:
            log(f"wikipedia rows: {offset + batch}, docs kept: {len(docs)}")
        time.sleep(0.15)
    return docs


def code_documents() -> List[str]:
    docs = []
    for pattern in ("cortex/**/*.py", "scripts/*.py", "tests/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            if path.stat().st_size > 60_000:
                continue
            docs.append(f"# file: {path.relative_to(ROOT)}\n" + path.read_text(encoding="utf-8"))
    return docs


def synthetic_documents(seed: int) -> Tuple[List[str], List[dict]]:
    math_fr = arithmetic_examples(1200, seed=seed, lang="fr")
    math_en = arithmetic_examples(600, seed=seed + 1, lang="en")
    logic_fr = logic_examples(500, seed=seed + 2, lang="fr")
    logic_en = logic_examples(250, seed=seed + 3, lang="en")
    docs: List[str] = []
    for item in math_fr + math_en + logic_fr + logic_en:
        docs.append(f"Question: {item['question']}\n{item['trace']}")
    chat = chat_examples(900, seed=seed + 4)
    return docs, chat


def document_fingerprint(doc: str) -> str:
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="datasets/processed")
    parser.add_argument("--raw-dir", default="datasets/raw")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--skip-network", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    raw_dir = ROOT / args.raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    documents: List[str] = []
    manifest: Dict[str, object] = {"sources": [], "synthetic": True}
    rng = random.Random(args.seed)

    if not args.skip_network:
        for name, lang, url in BOOKS:
            try:
                path = fetch(url, raw_dir / f"{name}.txt")
                text = normalize(strip_gutenberg(decode_text(path)))
                chunks = chunk_text(text)
                documents.extend(chunks)
                manifest["sources"].append(
                    {
                        "name": name,
                        "language": lang,
                        "url": url,
                        "chars": len(text),
                        "documents": len(chunks),
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "license": "public domain",
                    }
                )
                log(f"{name}: {len(text):,} chars, {len(chunks)} documents")
            except Exception as exc:
                log(f"{name} unavailable: {exc}")

        try:
            path = fetch(SHAKESPEARE_URL, raw_dir / "tinyshakespeare.txt")
            text = normalize(decode_text(path))
            chunks = chunk_text(text, size=3000)
            documents.extend(chunks)
            manifest["sources"].append(
                {
                    "name": "tinyshakespeare",
                    "language": "en",
                    "url": SHAKESPEARE_URL,
                    "chars": len(text),
                    "documents": len(chunks),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "license": "public domain",
                }
            )
            log(f"tinyshakespeare: {len(text):,} chars, {len(chunks)} documents")
        except Exception as exc:
            log(f"tinyshakespeare unavailable: {exc}")

        wiki = wikipedia_french()
        if wiki:
            documents.extend(wiki)
            manifest["sources"].append(
                {
                    "name": "wikipedia_fr",
                    "language": "fr",
                    "url": "https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia&config=20231101.fr",
                    "documents": len(wiki),
                    "chars": sum(len(d) for d in wiki),
                    "license": "CC BY-SA 4.0",
                }
            )
            log(f"wikipedia_fr: {len(wiki)} articles")
    else:
        log("network skipped; using synthetic + local code only")

    code = code_documents()
    documents.extend(code)
    manifest["sources"].append({"name": "project_code", "documents": len(code), "license": "Apache-2.0"})

    synthetic_docs, chat = synthetic_documents(args.seed)
    documents.extend(synthetic_docs)
    manifest["sources"].append(
        {
            "name": "synthetic_reasoning",
            "documents": len(synthetic_docs),
            "license": "Apache-2.0",
            "note": "deterministically generated arithmetic and logic traces",
        }
    )
    log(f"synthetic reasoning documents: {len(synthetic_docs)}")

    if not documents:
        raise SystemExit("no documents collected")

    # Deduplicate, then split deterministically by fingerprint so reruns are stable.
    unique: Dict[str, str] = {}
    for doc in documents:
        unique.setdefault(document_fingerprint(doc), doc)
    documents = list(unique.values())
    rng.shuffle(documents)
    train_docs, val_docs = [], []
    for doc in documents:
        digest = int(document_fingerprint(doc)[:8], 16)
        (val_docs if digest % int(1 / args.val_ratio) == 0 else train_docs).append(doc)

    # The tokenizer sees training data only, never validation text.
    tokenizer_path = out_dir / "tokenizer"
    corpus_file = out_dir / "_tokenizer_corpus.txt"
    corpus_file.write_text("\n\n".join(train_docs), encoding="utf-8")
    tokenizer = CortexTokenizer.train([corpus_file], vocab_size=args.vocab_size)
    tokenizer.save(tokenizer_path)
    corpus_file.unlink(missing_ok=True)
    log(f"tokenizer trained: vocab={tokenizer.vocab_size}")

    train_stats = tokenize_documents(tokenizer, train_docs, out_dir / "train.bin")
    val_stats = tokenize_documents(tokenizer, val_docs, out_dir / "val.bin")
    log(f"train: {train_stats}, val: {val_stats}")

    sft_path = out_dir / "sft.jsonl"
    sft_count = write_jsonl(sft_path, chat)
    log(f"SFT bootstrap examples: {sft_count}")

    manifest.update(
        {
            "seed": args.seed,
            "vocab_size": tokenizer.vocab_size,
            "documents": {"train": len(train_docs), "val": len(val_docs)},
            "tokens": {"train": train_stats["tokens"], "val": val_stats["tokens"]},
            "val_ratio": args.val_ratio,
            "train_sha256": hashlib.sha256(
                (out_dir / "train.bin").read_bytes()
            ).hexdigest(),
            "val_sha256": hashlib.sha256((out_dir / "val.bin").read_bytes()).hexdigest(),
        }
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(
            {
                "block_dtype": "uint16",
                "vocab_size": tokenizer.vocab_size,
                "train_tokens": train_stats["tokens"],
                "val_tokens": val_stats["tokens"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log("done")


if __name__ == "__main__":
    main()
