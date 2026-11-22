from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cortex.training.data import ChatSFTDataset, TokenDataset, collate_sft
from cortex.training.sft import SFTConfig, SFTTrainer, split_dataset
from cortex.training.trainer import cosine_lr, load_pretrained, save_pretrained


def test_token_dataset_windows(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(200, dtype=np.uint16).tofile(path)
    dataset = TokenDataset(path, block_size=16)
    x, y = dataset[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert torch.equal(y[:-1], x[1:])
    assert y[-1].item() == x[0].item() + 16


def test_collate_sft_pads_and_masks():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [1, 2], "labels": [1, 2]},
    ]
    out = collate_sft(batch, pad_token_id=0)
    assert out["input_ids"].tolist() == [[1, 2, 3], [1, 2, 0]]
    # Labels are shifted left: position t predicts the target at t + 1.
    assert out["labels"].tolist() == [[2, 3, -100], [2, -100, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_cosine_lr_schedule():
    lrs = [cosine_lr(step, 100, 10, 1e-3, 0.1) for step in (0, 5, 10, 50, 99)]
    assert lrs[0] < lrs[1] < lrs[2]
    assert lrs[2] >= lrs[3] >= lrs[4]
    assert lrs[-1] == pytest.approx(1e-4, rel=0.05)


def test_one_optimizer_step_changes_weights(tiny_model, tokenizer):
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    before = tiny_model.model.embed_tokens.weight.detach().clone()
    ids = torch.randint(0, tiny_model.config.vocab_size, (2, 8))
    loss = tiny_model(input_ids=ids, labels=ids)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    after = tiny_model.model.embed_tokens.weight.detach()
    assert not torch.allclose(before, after)


def test_save_and_load_roundtrip(tmp_path, tiny_model, tokenizer):
    save_pretrained(tiny_model, tokenizer, tiny_model.config, tmp_path / "ckpt")
    loaded, loaded_tokenizer, config = load_pretrained(tmp_path / "ckpt")
    ids = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        first = tiny_model(input_ids=ids)["logits"]
        second = loaded(input_ids=ids)["logits"]
    assert torch.allclose(first, second, atol=1e-6)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size


def test_sft_trainer_runs(tmp_path, tiny_model, tokenizer):
    path = tmp_path / "sft.jsonl"
    records = [
        {"messages": [
            {"role": "system", "content": "Tu es CORTEX"},
            {"role": "user", "content": f"Question {index}"},
            {"role": "assistant", "content": f"Réponse {index}"},
        ]}
        for index in range(8)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    dataset = ChatSFTDataset(path, tokenizer, max_length=64)
    train_data, val_data = split_dataset(dataset, 0.25)
    config = SFTConfig(
        batch_size=2,
        max_steps=3,
        learning_rate=1e-3,
        warmup_steps=1,
        eval_interval=2,
        save_interval=100,
        max_length=64,
        out_dir=str(tmp_path / "sft-out"),
    )
    trainer = SFTTrainer(tiny_model, tokenizer, train_data, config, val_dataset=val_data)
    metrics = trainer.train()
    assert metrics["steps"] == 3
    assert metrics["final_loss"] == metrics["final_loss"]  # not NaN
    assert (tmp_path / "sft-out" / "model.safetensors").exists()
