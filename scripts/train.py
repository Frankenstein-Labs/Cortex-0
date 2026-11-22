#!/usr/bin/env python3
"""Train CORTEX-0 from a YAML experiment configuration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.config import ExperimentConfig, get_preset  # noqa: E402
from cortex.model import CortexForCausalLM  # noqa: E402
from cortex.tokenizer import CortexTokenizer  # noqa: E402
from cortex.training.trainer import Trainer, load_dataset_pair, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny-cpu.yaml")
    parser.add_argument("--preset", choices=["tiny-cpu", "cortex-50m", "cortex-150m"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    experiment = ExperimentConfig.load(ROOT / args.config)
    if args.preset:
        experiment.model = get_preset(args.preset)
    training = experiment.training
    if args.max_steps:
        training = replace(training, max_steps=args.max_steps)
    if args.device:
        training = replace(training, device=args.device)
    if args.out_dir:
        training = replace(training, out_dir=args.out_dir)
    if args.no_resume:
        training = replace(training, resume=False)

    tokenizer = CortexTokenizer.from_pretrained(ROOT / experiment.tokenizer_dir)
    model_config = experiment.model
    if model_config.vocab_size != tokenizer.vocab_size:
        # A preset may declare a larger vocabulary than the tokenizer actually
        # provides; the embedding must match the tokenizer, so say so loudly.
        print(
            f"[train] WARNING: config vocab_size={model_config.vocab_size} "
            f"does not match tokenizer vocab_size={tokenizer.vocab_size}; "
            f"using {tokenizer.vocab_size}. Parameter count will differ from the preset.",
            file=sys.stderr,
            flush=True,
        )
        model_config = replace(model_config, vocab_size=tokenizer.vocab_size)

    train_dataset, val_dataset = load_dataset_pair(
        ROOT / experiment.train_data,
        ROOT / experiment.val_data,
        block_size=model_config.max_position_embeddings,
    )
    model = CortexForCausalLM(model_config)
    set_seed(training.seed)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training,
        out_dir=ROOT / training.out_dir,
    )
    print(
        json.dumps(
            {
                "parameters": model.num_parameters(),
                "parameters_non_embedding": model.num_parameters(non_embedding=True),
                "train_tokens": int(len(train_dataset.data)),
                "val_tokens": int(len(val_dataset.data)) if val_dataset else 0,
                "block_size": model_config.max_position_embeddings,
                "device": str(trainer.device),
                "dtype": str(trainer.dtype),
            },
            indent=2,
        ),
        flush=True,
    )
    result = trainer.train()
    print(json.dumps({"result": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
