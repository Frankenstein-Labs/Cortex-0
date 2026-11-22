#!/usr/bin/env python3
"""Build a French-first training corpus for CORTEX-0.

Sources are open and their licences are recorded in the manifest. Nothing is
invented: every document carries its source name and licence, and a source that
fails to load is reported as failed instead of being silently skipped.

Categories (see docs/CORPUS.md):
  A fondamental      grammar/spelling reference text
  B conversationnel  natural dialogue and everyday prose
  C professionnel    administrative, professional and encyclopedic prose
  D scientifique     science and mathematics prose
  E technique        programming and computer-science material
  F pedagogique      explanatory and instructional prose
  identity           controlled CORTEX-0 identity facts (synthetic, labelled)

Usage:
  python scripts/build_french_corpus.py --out-dir datasets/french \
      --wikipedia-docs 20000 --target-tokens 30000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

USER_AGENT = "CORTEX-0-corpus-builder/1.0 (Frankenstein-Labs)"

# Licence per source, kept in one place so the manifest cannot drift.
LICENCES = {
    "wikipedia_fr": ("CC BY-SA 4.0", "https://dumps.wikimedia.org/frwiki/"),
    "gutenberg_fr": ("public domain", "https://www.gutenberg.org/"),
    "project_code": ("Apache-2.0", "https://github.com/Frankenstein-Labs/Cortex-0"),
    "synthetic_instruction": ("Apache-2.0 (generated)", "generated deterministically"),
    "identity": ("Apache-2.0 (controlled)", "docs/IDENTITY.md"),
}

FRENCH_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "que", "qui",
    "dans", "pour", "pas", "sur", "au", "aux", "ce", "cette", "il", "elle", "nous",
    "vous", "ils", "elles", "avec", "sont", "être", "avoir", "plus", "mais", "ou",
    "comme", "tout", "fait", "aussi", "leur", "ses", "son", "sa", "par", "en",
}


def log(message: str) -> None:
    print(f"[corpus] {message}", flush=True)


# --------------------------------------------------------------------- cleaning
def normalize(text: str) -> str:
    """NFC normalise and collapse whitespace without destroying French accents."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def french_score(text: str) -> float:
    """Heuristic share of French function words; used only as a coarse filter."""
    words = re.findall(r"[a-zàâäéèêëîïôöùûüç']+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in FRENCH_STOPWORDS)
    return hits / len(words)


def is_low_quality(text: str, min_chars: int = 200, max_chars: int = 40000) -> Optional[str]:
    """Return a rejection reason, or None if the document is acceptable."""
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if french_score(text) < 0.08:
        return "not_french"
    words = re.findall(r"\w+", text)
    if len(words) < 40:
        return "too_few_words"
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    if unique_ratio < 0.20:
        return "repetitive"
    if text.count("<") > len(text) / 50:
        return "markup_heavy"
    return None


def shingles(text: str, k: int = 5) -> frozenset:
    words = re.findall(r"\w+", text.lower())
    return frozenset(" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1)))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def split_for(doc_id: str, val_ratio: float, test_ratio: float) -> str:
    """Assign a split by hash so the same document always lands in one split."""
    bucket = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


# ---------------------------------------------------------------------- sources
def fetch_url(url: str, retries: int = 3, timeout: int = 60) -> str:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def source_wikipedia(limit: int) -> Iterator[Tuple[str, str]]:
    """Stream French Wikipedia articles. Licence: CC BY-SA 4.0."""
    try:
        from datasets import load_dataset  # type: ignore

        stream = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
        for index, row in enumerate(stream):
            if index >= limit:
                break
            text = row.get("text") or ""
            if text:
                yield f"wikipedia_fr:{row.get('id', index)}", text
        return
    except Exception as exc:  # noqa: BLE001
        log(f"datasets streaming unavailable ({exc}); falling back to datasets-server HTTP")

    # Fallback: page the datasets-server rows API, 100 rows per request.
    base = "https://datasets-server.huggingface.co/rows"
    offset, produced = 0, 0
    while produced < limit:
        params = urllib.parse.urlencode({
            "dataset": "wikimedia/wikipedia",
            "config": "20231101.fr",
            "split": "train",
            "offset": offset,
            "length": min(100, limit - produced),
        })
        payload = json.loads(fetch_url(f"{base}?{params}"))
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row", {})
            text = row.get("text") or ""
            if text:
                produced += 1
                yield f"wikipedia_fr:{row.get('id', offset)}", text
        offset += len(rows)


GUTENBERG_FR = [
    ("les_miserables", "https://gutenberg.pglaf.org/cache/epub/17489/pg17489.txt"),
    ("vingt_mille_lieues", "https://gutenberg.pglaf.org/cache/epub/5097/pg5097.txt"),
    ("monte_cristo", "https://gutenberg.pglaf.org/cache/epub/17989/pg17989.txt"),
]


def source_gutenberg() -> Iterator[Tuple[str, str]]:
    """Public-domain French books. Licence: public domain."""
    for name, url in GUTENBERG_FR:
        try:
            yield f"gutenberg_fr:{name}", fetch_url(url)
        except Exception as exc:  # noqa: BLE001
            log(f"gutenberg source failed for {name}: {exc}")


def source_project_code() -> Iterator[Tuple[str, str]]:
    """The project's own Python sources. Licence: Apache-2.0."""
    for path in sorted((ROOT / "cortex").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield f"project_code:{path.relative_to(ROOT)}", path.read_text(encoding="utf-8")


def source_synthetic_instruction() -> Iterator[Tuple[str, str]]:
    """Deterministic French instruction/explanation templates.

    This is generated text, not scraped data, and is labelled as such in the
    manifest so it is never mistaken for a real corpus. ``chat_examples``
    returns ``{"messages": [...]}`` records, so they are flattened here.
    """
    from cortex.training.synthetic import chat_examples

    for index, example in enumerate(chat_examples(600, seed=1337)):
        messages = example.get("messages") or []
        if len(messages) < 3:
            continue
        prompt = messages[1].get("content", "")
        answer = messages[2].get("content", "")
        if prompt and answer:
            yield f"synthetic_instruction:{index}", f"Question : {prompt}\nRéponse : {answer}"


def source_identity() -> Iterator[Tuple[str, str]]:
    """Controlled identity facts from docs/IDENTITY.md."""
    facts = [
        ("Quel est ton nom ?", "Je m'appelle CORTEX-0."),
        ("Qui t'a créé ?", "J'ai été créé par Abdoulaye Coumbassa."),
        ("Quelle organisation te développe ?", "Je suis développé par Frankenstein-Labs."),
        ("D'où vient le projet ?", "Le projet CORTEX-0 est originaire de Guinée."),
        ("Quelle est ta première langue ?", "Ma première langue fondamentale est le français."),
        ("Quel est ton deuxième domaine ?", "Mon deuxième domaine prioritaire est l'informatique."),
        ("Quelle est ta mission ?",
         "Ma mission est d'être une intelligence artificielle utile, accessible et éducative, "
         "capable d'aider les étudiants, les développeurs, les chercheurs et les créateurs."),
        ("As-tu une voix ?",
         "Non. Je ne dispose ni de synthèse vocale ni de reconnaissance vocale."),
        ("As-tu accès à Internet ?", "Non, je n'ai pas d'accès à Internet."),
    ]
    for index, (question, answer) in enumerate(facts):
        yield f"identity:{index}", f"Question : {question}\nRéponse : {answer}"


# Pretraining sources only. Identity and synthetic instruction data are short
# by nature and are emitted as SFT records by build_sft() instead.
SOURCES = {
    "wikipedia_fr": source_wikipedia,
    "gutenberg_fr": source_gutenberg,
    "project_code": source_project_code,
}


def build_sft(out_dir: Path) -> Dict[str, int]:
    """Write a conversational SFT file from identity and synthetic sources.

    These records are ``{"messages": [...]}`` chat examples, the format
    ``ChatSFTDataset`` and ``scripts/train_sft.py`` expect.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    identity_path = ROOT / "datasets" / "identity" / "cortex0_identity.jsonl"
    if identity_path.exists():
        for line in identity_path.open(encoding="utf-8"):
            if line.strip():
                records.append(json.loads(line))

    from cortex.training.synthetic import chat_examples

    for example in chat_examples(600, seed=1337):
        if example.get("lang") == "fr" and len(example.get("messages") or []) >= 3:
            records.append({"messages": example["messages"]})

    path = out_dir / "train.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    log(f"wrote {len(records)} SFT records to {path}")
    return {"records": len(records)}


# ------------------------------------------------------------------------ build
def assign_category(source: str, text: str) -> str:
    """Coarse topical routing; the manifest records that this is heuristic."""
    lowered = text.lower()
    code_markers = ("def ", "import ", "function ", "class ", "const ", "return ")
    if source in {"project_code"}:
        return "E_technique"
    if source in {"identity", "synthetic_instruction"}:
        return "F_pedagogique"
    if sum(lowered.count(m) for m in code_markers) > 5:
        return "E_technique"
    if any(k in lowered for k in ("théorème", "équation", "molécule", "atome", "cellule")):
        return "D_scientifique"
    if any(k in lowered for k in ("monsieur", "madame", "courriel", "rapport", "société")):
        return "C_professionnel"
    if any(k in lowered for k in ("bonjour", "salut", "merci", "comment vas-tu")):
        return "B_conversationnel"
    return "A_fondamental"


def tokenize_splits(out_dir: Path, tokenizer_dir: Path) -> Dict[str, Dict[str, int]]:
    """Tokenize each JSONL split into a flat .bin file consumable by train.py."""
    from cortex.tokenizer import CortexTokenizer
    from cortex.training.data import tokenize_documents

    tokenizer = CortexTokenizer.from_pretrained(tokenizer_dir)
    summary: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        documents = [json.loads(line)["text"] for line in path.open(encoding="utf-8")]
        summary[split] = tokenize_documents(tokenizer, documents, out_dir / f"{split}.bin")
        log(f"tokenized {split}: {summary[split]}")
    return summary


def build(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    seen_hashes: Dict[str, str] = {}
    kept_shingles: List[frozenset] = []
    rejections: Counter = Counter()
    per_source: Dict[str, int] = Counter()
    per_category: Dict[str, int] = Counter()
    per_split: Dict[str, int] = Counter()
    failures: Dict[str, str] = {}
    records: List[Dict[str, object]] = []

    for name, loader in SOURCES.items():
        if name in args.skip_source:
            continue
        limit = args.wikipedia_docs if name == "wikipedia_fr" else 10 ** 9
        produced = 0
        try:
            for doc_id, raw in loader(limit) if name == "wikipedia_fr" else loader():
                text = normalize(raw)
                reason = is_low_quality(text)
                if reason:
                    rejections[reason] += 1
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen_hashes:
                    rejections["duplicate"] += 1
                    continue
                sh = shingles(text)
                if any(jaccard(sh, other) > args.dedup_threshold for other in kept_shingles[-2000:]):
                    rejections["near_duplicate"] += 1
                    continue
                seen_hashes[digest] = doc_id
                kept_shingles.append(sh)
                licence, url = LICENCES[name]
                category = assign_category(name, text)
                split = split_for(doc_id, args.val_ratio, args.test_ratio)
                records.append({
                    "id": doc_id,
                    "category": category,
                    "source": name,
                    "licence": licence,
                    "source_url": url,
                    "split": split,
                    "chars": len(text),
                    "text": text,
                })
                per_source[name] += 1
                per_category[category] += 1
                per_split[split] += 1
                produced += 1
                if produced % 500 == 0:
                    log(f"{name}: {produced} kept")
        except Exception as exc:  # noqa: BLE001
            failures[name] = f"{type(exc).__name__}: {exc}"
            log(f"source {name} failed: {exc}")

    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                if record["split"] == split:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "documents": dict(per_source),
        "categories": dict(per_category),
        "splits": dict(per_split),
        "chars_total": sum(int(r["chars"]) for r in records),
        "rejections": dict(rejections),
        "failed_sources": failures,
        "licences": {k: {"licence": v[0], "url": v[1]} for k, v in LICENCES.items()},
        "params": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "dedup_threshold": args.dedup_threshold,
            "wikipedia_docs": args.wikipedia_docs,
        },
        "notes": [
            "Categories are assigned by a documented keyword heuristic, not by a classifier.",
            "french_score and is_low_quality are heuristics, not validated language ID.",
            "synthetic_instruction and identity documents are generated, not scraped.",
            "Splits are assigned by document hash, so a document never spans two splits.",
        ],
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    if args.tokenize:
        tokenizer_dir = Path(args.tokenizer_dir)
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = ROOT / tokenizer_dir
        manifest["tokenized"] = tokenize_splits(out_dir, tokenizer_dir)
        manifest["tokenizer_dir"] = str(args.tokenizer_dir)
    if args.sft_dir:
        manifest["sft"] = build_sft(Path(args.sft_dir))
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"wrote {len(records)} documents to {corpus_path}")
    log(f"splits={dict(per_split)} categories={dict(per_category)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="datasets/french")
    parser.add_argument("--wikipedia-docs", type=int, default=20000)
    parser.add_argument("--target-tokens", type=int, default=0,
                        help="advisory only; recorded in the manifest")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-source", action="append", default=[])
    parser.add_argument("--tokenize", action="store_true",
                        help="also write train.bin/val.bin/test.bin with --tokenizer-dir")
    parser.add_argument("--tokenizer-dir", default="datasets/processed/tokenizer")
    parser.add_argument("--sft-dir", default="datasets/sft",
                        help="write conversational SFT records here; empty string disables")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
