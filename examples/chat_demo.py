#!/usr/bin/env python3
"""Talk to a trained CORTEX-0 checkpoint from Python."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.conversation.companion import Companion
from cortex.memory.store import MemoryStore
from cortex.training.trainer import load_pretrained


def main() -> int:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "checkpoints" / "tiny-cpu")
    if not (model_dir / "model.safetensors").exists():
        print(f"Aucun checkpoint dans {model_dir}. Lancez d'abord scripts/train.py.")
        return 1
    model, tokenizer, _ = load_pretrained(model_dir)
    with tempfile.TemporaryDirectory() as directory:
        store = MemoryStore(Path(directory) / "demo.sqlite")
        companion = Companion(store, model=model, tokenizer=tokenizer, max_new_tokens=48)
        for message in ["Bonjour CORTEX, qui es-tu ?", "Que sais-tu faire ?"]:
            reply = companion.chat("demo", message)
            print(f"Vous> {message}")
            print(f"CORTEX> {reply.text.strip()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
