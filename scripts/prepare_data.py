#!/usr/bin/env python3
"""Prepare a reproducible token corpus from TinyStories parquet files.

Writes:
  <out>/train.bin, <out>/val.bin   flat uint16 token streams
  <out>/meta.json                  measured counts and provenance
  <out>/tokenizer.json             trained byte-level BPE tokenizer
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from cortex.tokenizer.bpe import CortexTokenizer
from cortex.training.data import write_token_file


def read_stories(path: Path, limit: int, min_chars: int, max_chars: int, seed: int) -> list[str]:
    table = pq.read_table(path, columns=["text"])
    texts = [t for t in table.column("text").to_pylist() if t and min_chars <= len(t) <= max_chars]
    rng = random.Random(seed)
    if limit and len(texts) > limit:
        rng.shuffle(texts)
        texts = texts[:limit]
    return texts


def encode_stories(tok: CortexTokenizer, texts: list[str], batch_size: int = 512):
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for ids in tok.encode_batch(batch):
            yield [tok.bos_id] + ids + [tok.eos_id]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", default="data/raw/tinystories_train.parquet")
    ap.add_argument("--val-parquet", default="data/raw/tinystories_valid.parquet")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--max-train-stories", type=int, default=120_000)
    ap.add_argument("--max-val-stories", type=int, default=2_000)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--tokenizer-sample-stories", type=int, default=20_000)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"reading {args.train_parquet} ...", flush=True)
    train_texts = read_stories(
        Path(args.train_parquet), args.max_train_stories, args.min_chars, args.max_chars, args.seed
    )
    print(f"reading {args.val_parquet} ...", flush=True)
    val_texts = read_stories(
        Path(args.val_parquet), args.max_val_stories, args.min_chars, args.max_chars, args.seed + 1
    )
    if not train_texts or not val_texts:
        raise SystemExit("no stories loaded; check the parquet paths")

    sample = train_texts[: args.tokenizer_sample_stories]
    print(f"training BPE tokenizer (vocab={args.vocab_size}) on {len(sample)} stories ...", flush=True)
    tok = CortexTokenizer.train(sample, vocab_size=args.vocab_size, save_path=out / "tokenizer.json")

    print(f"tokenizing {len(train_texts)} train / {len(val_texts)} val stories ...", flush=True)
    train_tokens = write_token_file(
        out / "train.bin", encode_stories(tok, train_texts), vocab_size=tok.vocab_size
    )
    val_tokens = write_token_file(out / "val.bin", encode_stories(tok, val_texts), vocab_size=tok.vocab_size)

    train_chars = sum(len(t) for t in train_texts)
    val_chars = sum(len(t) for t in val_texts)
    meta = {
        "source": "roneneldan/TinyStories (Hugging Face)",
        "license": "CDLA-Sharing-1.0",
        "train_parquet": str(args.train_parquet),
        "val_parquet": str(args.val_parquet),
        "train_stories": len(train_texts),
        "val_stories": len(val_texts),
        "train_chars": train_chars,
        "val_chars": val_chars,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "chars_per_token": round((train_chars + val_chars) / max(1, train_tokens + val_tokens), 4),
        "vocab_size": tok.vocab_size,
        "dtype": "uint16",
        "seed": args.seed,
        "prepared_seconds": round(time.time() - t0, 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tok.save_with_meta(out / "tokenizer.json", meta)
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
