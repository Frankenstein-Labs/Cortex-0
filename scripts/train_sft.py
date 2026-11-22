#!/usr/bin/env python3
"""Supervised fine-tuning of a pretrained CORTEX-0 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.training.data import ChatSFTDataset
from cortex.training.sft import SFTConfig, SFTTrainer, split_dataset
from cortex.training.trainer import load_pretrained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    parser.add_argument("--data", default="datasets/processed/sft.jsonl")
    parser.add_argument("--out", default="checkpoints/tiny-cpu-sft")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    model, tokenizer, _ = load_pretrained(args.model_dir)
    dataset = ChatSFTDataset(args.data, tokenizer, max_length=args.max_length)
    train_data, val_data = split_dataset(dataset, val_ratio=0.1)
    config = SFTConfig(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.steps,
        learning_rate=args.lr,
        max_length=args.max_length,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        device=args.device,
        out_dir=args.out,
    )
    trainer = SFTTrainer(model, tokenizer, train_data, config, val_dataset=val_data)
    metrics = trainer.train()
    print(json.dumps({"status": "ok", **metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
