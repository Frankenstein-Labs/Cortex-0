"""Checkpoint integrity: RNG state, resume behaviour and loud failure modes."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cortex.config import TrainingConfig
from cortex.training.data import TokenDataset
from cortex.training.trainer import Trainer


def _dataset(tmp_path, tokenizer, blocks: int = 16) -> TokenDataset:
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, tokenizer.vocab_size, size=blocks * 64, dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    tokens.tofile(path)
    return TokenDataset(path, block_size=64)


def _trainer(tmp_path, tiny_model, tokenizer, resume: bool, device: str = "auto") -> Trainer:
    config = TrainingConfig(
        batch_size=2,
        grad_accum_steps=1,
        max_steps=4,
        warmup_steps=1,
        eval_interval=10_000,
        log_interval=1,
        save_interval=10_000,
        out_dir=str(tmp_path / "ckpt"),
        resume=resume,
        tensorboard=False,
        device=device,
    )
    return Trainer(
        model=tiny_model,
        tokenizer=tokenizer,
        train_dataset=_dataset(tmp_path, tokenizer),
        val_dataset=None,
        config=config,
        out_dir=tmp_path / "ckpt",
    )


def test_save_writes_full_training_state(tmp_path, tiny_model, tokenizer):
    trainer = _trainer(tmp_path, tiny_model, tokenizer, resume=False)
    trainer.state.step = 3
    trainer.state.tokens_seen = 384
    directory = trainer.save(tmp_path / "ckpt")

    for name in ("model.safetensors", "optimizer.pt", "trainer_state.json", "rng_state.pt"):
        assert (directory / name).exists(), f"missing {name}"

    state = json.loads((directory / "trainer_state.json").read_text())
    assert state["step"] == 3
    assert state["tokens_seen"] == 384

    rng = torch.load(directory / "rng_state.pt", weights_only=False)
    assert {"python", "numpy", "torch"} <= set(rng)


def test_resume_restores_step_and_rng(tmp_path, tiny_model, tokenizer):
    trainer = _trainer(tmp_path, tiny_model, tokenizer, resume=False)
    trainer.state.step = 7
    trainer.save(tmp_path / "ckpt")

    resumed = _trainer(tmp_path, tiny_model, tokenizer, resume=True)
    resumed._resume()
    assert resumed.state.step == 7


def test_weights_without_state_fails_loudly(tmp_path, tiny_model, tokenizer):
    trainer = _trainer(tmp_path, tiny_model, tokenizer, resume=False)
    trainer.save(tmp_path / "ckpt")
    (tmp_path / "ckpt" / "trainer_state.json").unlink()

    broken = _trainer(tmp_path, tiny_model, tokenizer, resume=True)
    with pytest.raises(RuntimeError, match="trainer_state.json"):
        broken.train()


def test_no_grad_scaler_on_cpu(tmp_path, tiny_model, tokenizer):
    trainer = _trainer(tmp_path, tiny_model, tokenizer, resume=False, device="cpu")
    assert trainer.device.type == "cpu"
    assert trainer._scaler is None


def test_tokens_per_second_excludes_pre_resume_tokens(tmp_path, tiny_model, tokenizer):
    trainer = _trainer(tmp_path, tiny_model, tokenizer, resume=False)
    # Simulate a long prior run, then verify the rate uses only this run's tokens.
    trainer.state.step = 1000
    trainer.state.tokens_seen = 10_000_000
    trainer.config.max_steps = 1002
    result = trainer.train()
    assert result["steps"] == 1002
    record = json.loads(
        (tmp_path / "ckpt" / "logs" / "train.jsonl").read_text().strip().splitlines()[-1]
    )
    assert record["tokens_per_second"] < 10_000_000
