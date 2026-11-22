"""Checkpoint evaluation: validation loss, perplexity and samples."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cortex.model.lm import load_model
from cortex.training.data import TokenDataMeta, TokenDataset
from cortex.training.trainer import evaluate


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str | Path,
    data_dir: str | Path = "data/processed",
    batch_size: int = 16,
    max_batches: int = 40,
    device: str = "cpu",
) -> dict:
    model = load_model(checkpoint, device=device)
    meta = TokenDataMeta.load(Path(data_dir) / "meta.json")
    val = TokenDataset(Path(data_dir) / "val.bin", min(model.cfg.max_seq_len, 256), meta.vocab_size, meta.dtype)
    result = evaluate(model, val, batch_size, torch.device(device), max_batches=max_batches)
    sidecar = json.loads(Path(checkpoint).with_suffix(".json").read_text(encoding="utf-8"))
    return {
        "checkpoint": str(checkpoint),
        "parameters": model.num_parameters(),
        "model_config": model.cfg.to_dict(),
        "training": sidecar.get("extra", {}),
        "val": result.to_dict(),
    }


def sample(
    checkpoint: str | Path,
    prompt: str = "Once upon a time",
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: int | None = 42,
    tokenizer_path: str | Path = "data/processed/tokenizer.json",
) -> str:
    from cortex.generation.sampler import GenerationConfig, generate_text
    from cortex.tokenizer.bpe import CortexTokenizer

    model = load_model(checkpoint)
    tok = CortexTokenizer.load(tokenizer_path)
    ids = tok.encode(prompt)
    text, _ = generate_text(
        model,
        tok,
        ids,
        GenerationConfig(
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed
        ),
    )
    return text
