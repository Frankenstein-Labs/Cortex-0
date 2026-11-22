#!/usr/bin/env python3
"""Measure the reasoning engine on a procedural arithmetic benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cortex.evaluation.reasoning_eval import evaluate_reasoning, make_problems
from cortex.generation.sampler import GenerationConfig
from cortex.model.lm import load_model
from cortex.reasoning.engine import ReasoningEngine
from cortex.tokenizer.bpe import CortexTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cortex-0-nano/best.safetensors")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--problems", type=int, default=20)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="artifacts/reasoning_eval.json")
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    tok = CortexTokenizer.load(args.tokenizer)
    engine = ReasoningEngine(model, tok)
    problems = make_problems(args.problems, seed=7)
    gen = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.8, top_k=50, top_p=0.95, seed=1234)
    report = evaluate_reasoning(engine, problems, n_samples=args.samples, gen=gen)
    report["checkpoint"] = args.checkpoint
    report["parameters"] = model.num_parameters()
    report["note"] = (
        "Arithmetic problems are generated procedurally; ground truth is computed. "
        "This measures the current model, not a general reasoning claim."
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        name: {"correct": s["correct"], "accuracy": s["accuracy"]}
        for name, s in report["strategies"].items()
    }
    print(json.dumps({"n_problems": args.problems, "results": summary}, indent=2))


if __name__ == "__main__":
    main()
