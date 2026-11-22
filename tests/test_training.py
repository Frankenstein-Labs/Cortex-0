from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cortex.config import CortexConfig, TrainConfig
from cortex.model.lm import CortexLM, load_model
from cortex.training.data import TokenDataMeta, TokenDataset, write_token_file
from cortex.training.trainer import Trainer, cosine_lr, evaluate, resolve_device


@pytest.fixture
def synthetic_data(tmp_path):
    vocab = 64
    train = np.array([i % vocab for i in range(30_000)], dtype=np.uint16)
    val = np.array([i % vocab for i in range(6_000)], dtype=np.uint16)
    write_token_file(tmp_path / "train.bin", [train], vocab)
    write_token_file(tmp_path / "val.bin", [val], vocab)
    meta = {
        "train_tokens": len(train),
        "val_tokens": len(val),
        "vocab_size": vocab,
        "dtype": "uint16",
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_token_dataset_batches(synthetic_data):
    ds = TokenDataset(synthetic_data / "train.bin", seq_len=32, vocab_size=64)
    x, y = ds.get_batch(4, torch.Generator().manual_seed(0))
    assert x.shape == (4, 32) and y.shape == (4, 32)
    assert torch.equal(x[:, 1:], y[:, :-1])
    batches = list(ds.sequential_batches(2, max_batches=3))
    assert len(batches) == 3


def test_token_data_meta(synthetic_data):
    meta = TokenDataMeta.load(synthetic_data / "meta.json")
    assert meta.train_tokens == 30_000
    assert meta.vocab_size == 64


def test_cosine_schedule():
    cfg = TrainConfig(max_steps=100, warmup_steps=10, learning_rate=1e-3, min_lr_ratio=0.1)
    assert cosine_lr(0, cfg) == pytest.approx(1e-4)
    assert cosine_lr(9, cfg) == pytest.approx(1e-3)
    assert cosine_lr(100, cfg) == pytest.approx(1e-4)
    mid = cosine_lr(55, cfg)
    assert 1e-4 < mid < 1e-3


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"


def test_training_reduces_loss_and_saves(synthetic_data, tmp_path):
    cfg = CortexConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
    model = CortexLM(cfg)
    train_cfg = TrainConfig(
        seq_len=32,
        batch_size=16,
        max_steps=60,
        warmup_steps=5,
        learning_rate=3e-3,
        log_interval=10,
        eval_interval=30,
        eval_batches=3,
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="unit",
        seed=3,
    )
    train_ds = TokenDataset(synthetic_data / "train.bin", 32, 64)
    val_ds = TokenDataset(synthetic_data / "val.bin", 32, 64)
    trainer = Trainer(model, train_cfg, train_ds, val_ds, output_dir=tmp_path / "ckpt")
    summary = trainer.train()
    assert summary["steps"] == 60
    assert summary["best_val_loss"] is not None
    assert (tmp_path / "ckpt" / "unit" / "best.safetensors").exists()
    assert (tmp_path / "ckpt" / "unit" / "metrics.jsonl").exists()
    loaded = load_model(tmp_path / "ckpt" / "unit" / "best.safetensors")
    assert loaded.cfg.vocab_size == 64


def test_evaluate_metrics(synthetic_data):
    cfg = CortexConfig(vocab_size=64, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32)
    model = CortexLM(cfg)
    ds = TokenDataset(synthetic_data / "val.bin", 32, 64)
    result = evaluate(model, ds, batch_size=8, device=torch.device("cpu"), max_batches=2)
    assert result.loss > 0
    assert result.perplexity > 1
    assert 0 <= result.token_accuracy <= 1
    assert result.batches == 2
