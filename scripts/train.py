#!/usr/bin/env python3
"""Train CORTEX-0 from a prepared token corpus.

Example (CPU, ~10M parameters):
  python scripts/train.py --config configs/cortex-0-mini.json \
      --data data/processed --run-name cortex-0-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cortex.config import ExperimentConfig
from cortex.model.lm import CortexLM
from cortex.training.data import TokenDataMeta, TokenDataset
from cortex.training.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--resume", default=None, help="path to a *.train.pt state file")
    ap.add_argument("--steps", type=int, default=None, help="override max_steps")
    ap.add_argument("--device", default=None, help="override device (cpu/cuda/mps/auto)")
    args = ap.parse_args()

    exp = ExperimentConfig.load(args.config)
    if args.run_name:
        exp.train.run_name = args.run_name
    if args.steps:
        exp.train.max_steps = args.steps
    if args.device:
        exp.train.device = args.device

    data_dir = Path(args.data)
    meta = TokenDataMeta.load(data_dir / "meta.json")
    if exp.model.vocab_size != meta.vocab_size:
        print(
            f"[warn] config vocab_size={exp.model.vocab_size} != tokenizer vocab={meta.vocab_size}; "
            "using tokenizer size",
            file=sys.stderr,
        )
        exp.model.vocab_size = meta.vocab_size

    train_ds = TokenDataset(data_dir / "train.bin", exp.train.seq_len, meta.vocab_size, meta.dtype)
    val_ds = TokenDataset(data_dir / "val.bin", exp.train.seq_len, meta.vocab_size, meta.dtype)

    model = CortexLM(exp.model)
    print(
        json.dumps(
            {
                "run_name": exp.train.run_name,
                "parameters": model.num_parameters(),
                "non_embedding_parameters": model.num_parameters(non_embedding=True),
                "train_tokens": meta.train_tokens,
                "val_tokens": meta.val_tokens,
                "seq_len": exp.train.seq_len,
                "batch_size": exp.train.batch_size,
                "max_steps": exp.train.max_steps,
                "tokens_per_step": exp.train.batch_size * exp.train.grad_accum * exp.train.seq_len,
            },
            indent=2,
        ),
        flush=True,
    )

    trainer = Trainer(
        model,
        exp.train,
        train_ds,
        val_ds,
        output_dir=args.checkpoint_dir,
        tokenizer_meta={"path": str(data_dir / "tokenizer.json"), **meta.__dict__},
    )
    if args.resume:
        trainer.resume(args.resume)
        print(f"resumed from {args.resume} at step {trainer.step}", flush=True)
    summary = trainer.train()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
