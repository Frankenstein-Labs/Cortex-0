#!/usr/bin/env python3
"""Sample from a trained CORTEX-0 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cortex.model.generation import generate
from cortex.training.trainer import load_pretrained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    parser.add_argument("--prompt", default="CORTEX est")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chat", action="store_true", help="wrap the prompt in the CORTEX chat template")
    parser.add_argument("--system", default="Tu es CORTEX-0, une petite intelligence artificielle autonome.")
    args = parser.parse_args()

    model, tokenizer, config = load_pretrained(args.model_dir)
    model.to(args.device)
    prompt = args.prompt
    if args.chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": args.system}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
    ids = tokenizer.encode(prompt)
    limit = config.max_position_embeddings - args.max_new_tokens
    if len(ids) > limit:
        ids = ids[-limit:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    for sample in range(args.samples):
        generator = torch.Generator(device="cpu").manual_seed(args.seed + sample)
        output = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            stop_token_ids=[tokenizer.end_token_id],
            generator=generator,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True)
        print(f"--- sample {sample + 1} ---")
        print(text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
