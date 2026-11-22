#!/usr/bin/env python3
"""Run best-of-N reasoning with the programmatic verifier."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.memory.store import MemoryStore
from cortex.reasoning.engine import ReasoningConfig, ReasoningEngine
from cortex.training.trainer import load_pretrained


def main() -> int:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "checkpoints" / "tiny-cpu")
    if not (model_dir / "model.safetensors").exists():
        print(f"Aucun checkpoint dans {model_dir}.")
        return 1
    model, tokenizer, _ = load_pretrained(model_dir)
    store = MemoryStore(Path("reasoning-demo.sqlite"))
    engine = ReasoningEngine(
        model,
        tokenizer,
        ReasoningConfig(num_samples=3, max_new_tokens=48, temperature=0.8, seed=0),
        store=store,
    )
    problems = [
        ("Combien font 12 + 9 ?", "21"),
        ("Combien font 7 * 6 ?", "42"),
        ("Combien font 100 - 37 ?", "63"),
    ]
    for question, expected in problems:
        result = engine.solve(question, expected=expected)
        print(f"Q: {question}")
        print(f"   réponse proposée : {result.answer!r}")
        print(f"   vérifiée         : {result.verified}")
        print(f"   confiance        : {result.confidence:.4f}")
        print(f"   candidats        : {len(result.candidates)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
