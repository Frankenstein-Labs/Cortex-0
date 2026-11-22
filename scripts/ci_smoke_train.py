#!/usr/bin/env python3
"""End-to-end smoke training used by CI (synthetic data, no download)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from cortex.config import CortexConfig, TrainConfig
from cortex.model.lm import CortexLM, load_model
from cortex.training.data import TokenDataset, write_token_file
from cortex.training.trainer import Trainer


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cortex-ci-"))
    vocab = 64
    train = np.array([(i * 7) % vocab for i in range(40_000)], dtype=np.uint16)
    val = np.array([(i * 7) % vocab for i in range(4_000)], dtype=np.uint16)
    write_token_file(tmp / "train.bin", [train], vocab)
    write_token_file(tmp / "val.bin", [val], vocab)

    model = CortexLM(CortexConfig(vocab_size=vocab, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32))
    cfg = TrainConfig(
        seq_len=32,
        batch_size=16,
        max_steps=30,
        warmup_steps=5,
        learning_rate=3e-3,
        log_interval=10,
        eval_interval=15,
        eval_batches=2,
        checkpoint_dir=str(tmp / "ckpt"),
        run_name="ci",
    )
    trainer = Trainer(
        model,
        cfg,
        TokenDataset(tmp / "train.bin", 32, vocab),
        TokenDataset(tmp / "val.bin", 32, vocab),
        output_dir=tmp / "ckpt",
    )
    summary = trainer.train()
    checkpoint = tmp / "ckpt" / "ci" / "best.safetensors"
    assert checkpoint.exists(), "checkpoint was not written"
    loaded = load_model(checkpoint)
    assert loaded.num_parameters() == model.num_parameters()
    print(json.dumps({"ok": True, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
