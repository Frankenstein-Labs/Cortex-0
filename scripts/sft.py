#!/usr/bin/env python3
"""Supervised fine-tuning of a pretrained CORTEX-0 checkpoint on chat examples."""

from __future__ import annotations

import argparse
import json
import random

import pyarrow.parquet as pq

from cortex.model.lm import load_model
from cortex.tokenizer.bpe import CortexTokenizer
from cortex.training.sft import SFTConfig, build_examples, run_sft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cortex-0-nano/best.safetensors")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--parquet", default="data/raw/tinystories_train.parquet")
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-name", default="cortex-0-nano-sft")
    args = ap.parse_args()

    table = pq.read_table(args.parquet, columns=["text"])
    texts = [t for t in table.column("text").to_pylist() if t]
    random.Random(1337).shuffle(texts)
    examples = build_examples(texts[: args.examples * 2], max_examples=args.examples)
    print(f"built {len(examples)} chat examples")

    model = load_model(args.checkpoint)
    tok = CortexTokenizer.load(args.tokenizer)
    cfg = SFTConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        device=args.device,
        run_name=args.run_name,
    )
    summary = run_sft(model, tok, examples, cfg, output_dir="checkpoints")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
