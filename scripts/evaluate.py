#!/usr/bin/env python3
"""Evaluate perplexity and the internal reasoning suite, with provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.evaluation.perplexity import evaluate_perplexity
from cortex.evaluation.reasoning import run_suite
from cortex.training.data import TokenDataset
from cortex.training.trainer import load_pretrained


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    parser.add_argument("--val-data", default="datasets/processed/val.bin")
    parser.add_argument("--output", default="experiments/eval-tiny-cpu.json")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--skip-suite", action="store_true")
    args = parser.parse_args()

    started = time.time()
    model, tokenizer, config = load_pretrained(args.model_dir)
    model.to(args.device)
    results = {
        "model_dir": args.model_dir,
        "parameters": model.num_parameters(),
        "parameters_non_embedding": model.num_parameters(non_embedding=True),
        "config": config.to_dict(),
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit(),
            "device": args.device,
            "suite": "cortex-internal-v1",
            "note": "Internal arithmetic/logic suite; not MMLU, GSM8K or any external benchmark.",
        },
    }
    if not args.skip_perplexity:
        dataset = TokenDataset(
            args.val_data,
            block_size=config.max_position_embeddings,
            stride=config.max_position_embeddings,
        )
        results["perplexity"] = evaluate_perplexity(
            model,
            dataset,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            device=args.device,
        )
    if not args.skip_suite:
        suite = run_suite(
            model,
            tokenizer,
            device=args.device,
            n=args.n,
            max_new_tokens=args.max_new_tokens,
        )
        for key in ("arithmetic_fr", "arithmetic_en", "logic_fr", "logic_en"):
            if key in suite and isinstance(suite[key], dict):
                suite[key].pop("examples", None)
        results["reasoning"] = suite
    results["elapsed_seconds"] = round(time.time() - started, 1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
