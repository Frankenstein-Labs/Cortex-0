#!/usr/bin/env python3
"""Generate text from a CORTEX-0 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cortex.generation.sampler import GenerationConfig, generate_text
from cortex.model.lm import load_model
from cortex.tokenizer.bpe import CortexTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cortex-0-nano/best.safetensors")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=1)
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    tok = CortexTokenizer.load(args.tokenizer)
    ids = tok.encode(args.prompt)
    for i in range(args.samples):
        text, _ = generate_text(
            model,
            tok,
            ids,
            GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=None if args.seed is None else args.seed + i,
            ),
        )
        print(json.dumps({"sample": i, "prompt": args.prompt, "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
