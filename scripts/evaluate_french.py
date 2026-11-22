#!/usr/bin/env python3
"""Run the French evaluation suite and record a reproducible artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cortex.evaluation.french import run_french_suite  # noqa: E402
from cortex.evaluation.perplexity import evaluate_perplexity  # noqa: E402
from cortex.training.data import TokenDataset  # noqa: E402
from cortex.training.trainer import load_pretrained  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--val-data", default="datasets/french/val.bin")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, tokenizer, config = load_pretrained(ROOT / args.model_dir)
    model.to(device)

    report: dict = {
        "model_dir": args.model_dir,
        "git_commit": git_commit(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "parameters": model.num_parameters(),
        "context": config.max_position_embeddings,
        "vocab_size": tokenizer.vocab_size,
    }

    val_path = ROOT / args.val_data
    if val_path.exists():
        dataset = TokenDataset(val_path, block_size=config.max_position_embeddings)
        metrics = evaluate_perplexity(
            model, dataset, batch_size=args.batch_size,
            device=device, max_batches=args.max_batches,
        )
        report["perplexity"] = metrics
    else:
        report["perplexity"] = None
        report["perplexity_note"] = f"validation data not found: {val_path}"

    report["french_suite"] = run_french_suite(model, tokenizer, device)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    suite = report["french_suite"]
    print(json.dumps({
        "output": str(out),
        "perplexity": (report.get("perplexity") or {}).get("perplexity"),
        "identity_accuracy": suite["identity"]["accuracy"],
        "completion_accuracy": suite["grammar_completion"]["accuracy"],
        "repetitive_outputs": suite["generation_health"]["repetitive_outputs"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
