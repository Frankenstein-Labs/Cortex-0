"""Supervised fine-tuning on chat-formatted examples.

Only assistant tokens contribute to the loss. Examples are built from
TinyStories with deterministic templates, so the run is reproducible and does
not depend on an external instruction dataset.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from cortex.model.lm import save_checkpoint
from cortex.training.trainer import cosine_lr, resolve_device, resolve_dtype


@dataclass
class SFTConfig:
    max_steps: int = 300
    batch_size: int = 8
    seq_len: int = 256
    learning_rate: float = 2e-4
    warmup_steps: int = 30
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    log_interval: int = 20
    seed: int = 1337
    device: str = "cpu"
    dtype: str = "float32"
    run_name: str = "cortex-0-nano-sft"


def build_examples(texts: list[str], max_examples: int | None = None, seed: int = 1337) -> list[dict[str, str]]:
    """Turn stories into instruction/response pairs with simple templates."""
    rng = random.Random(seed)
    examples: list[dict[str, str]] = []
    for text in texts:
        text = " ".join(text.split())
        if len(text) < 120:
            continue
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        if len(sentences) < 3:
            continue
        opening = sentences[0][:120]
        templates = [
            ("Continue the story that starts like this:", text),
            ("Write a short children's story. Start with:", text),
            ("Tell me a story about a character who appears in this opening:", text),
        ]
        prompt, response = rng.choice(templates)
        examples.append({"prompt": f"{prompt}\n{opening}", "response": text})
        if max_examples and len(examples) >= max_examples:
            break
    return examples


class SFTDataset:
    def __init__(self, tokenizer, examples: list[dict[str, str]], seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples: list[tuple[list[int], list[int]]] = []
        for ex in examples:
            prompt_ids = tokenizer.encode(
                tokenizer.chat_prompt([{"role": "user", "content": ex["prompt"]}], add_generation_prompt=True)
            )
            response_ids = tokenizer.encode(ex["response"]) + [tokenizer.end_id]
            input_ids = (prompt_ids + response_ids)[: seq_len + 1]
            if len(input_ids) < 8:
                continue
            labels = ([-100] * len(prompt_ids) + response_ids)[: len(input_ids)]
            self.samples.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def batch(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.samples), (batch_size,), generator=generator).tolist()
        max_len = max(len(self.samples[i][0]) for i in idx)
        pad = self.tokenizer.pad_id
        xs, ys = [], []
        for i in idx:
            ids, labels = self.samples[i]
            pad_n = max_len - len(ids)
            xs.append(ids + [pad] * pad_n)
            ys.append(labels + [-100] * pad_n)
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def run_sft(
    model,
    tokenizer,
    examples: list[dict[str, str]],
    cfg: SFTConfig,
    output_dir: str | Path = "checkpoints",
) -> dict:
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    model.to(device)
    model.train()
    dataset = SFTDataset(tokenizer, examples, cfg.seq_len)
    if len(dataset) < cfg.batch_size:
        raise ValueError("not enough SFT examples")
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(cfg.seed)
    out = Path(output_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    losses = []
    for step in range(cfg.max_steps):
        lr = cosine_lr(step, _as_train_cfg(cfg))
        for group in optim.param_groups:
            group["lr"] = lr
        x, y = dataset.batch(cfg.batch_size, generator)
        x, y = x.to(device), y.to(device)
        out_dict = model(x, targets=y)
        optim.zero_grad()
        out_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        losses.append(float(out_dict["loss"].detach()))
        if (step + 1) % cfg.log_interval == 0 or step == 0:
            record = {
                "step": step + 1,
                "loss": round(sum(losses[-cfg.log_interval :]) / len(losses[-cfg.log_interval :]), 6),
                "lr": round(lr, 8),
                "elapsed_sec": round(time.time() - started, 2),
            }
            print(json.dumps(record), flush=True)
    save_checkpoint(
        model,
        out / "sft.safetensors",
        extra={"sft_steps": cfg.max_steps, "sft_examples": len(dataset), "instruction_tuned": True},
    )
    summary = {
        "run_name": cfg.run_name,
        "steps": cfg.max_steps,
        "examples": len(dataset),
        "first_loss": round(losses[0], 4),
        "last_loss": round(sum(losses[-10:]) / min(10, len(losses)), 4),
        "wall_time_sec": round(time.time() - started, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _as_train_cfg(cfg: SFTConfig):
    from cortex.config import TrainConfig

    return TrainConfig(
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        min_lr_ratio=cfg.min_lr_ratio,
    )
