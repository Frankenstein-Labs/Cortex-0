"""Language-modelling metrics: loss, perplexity, bits per token."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ..model.transformer import CortexForCausalLM
from ..training.data import TokenDataset, make_dataloader


@torch.no_grad()
def evaluate_perplexity(
    model: CortexForCausalLM,
    dataset: TokenDataset,
    batch_size: int = 8,
    max_batches: Optional[int] = None,
    device: Optional[torch.device | str] = None,
) -> Dict[str, float]:
    device = torch.device(device or "cpu")
    was_training = model.training
    model.eval()
    model.to(device)
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_loss = 0.0
    total_tokens = 0
    for idx, (x, y) in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        loss = model(input_ids=x, labels=y)["loss"]
        total_tokens += y.numel()
        total_loss += float(loss.detach()) * y.numel()
    if total_tokens == 0:
        raise ValueError("no tokens evaluated")
    loss = total_loss / total_tokens
    if was_training:
        model.train()
    return {
        "loss": round(loss, 5),
        "perplexity": round(math.exp(loss), 3),
        "bits_per_token": round(loss / math.log(2), 5),
        "tokens": total_tokens,
    }
