"""Supervised fine-tuning on the CORTEX chat format."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..model.transformer import CortexForCausalLM
from ..tokenizer.bpe import CortexTokenizer
from .data import ChatSFTDataset, collate_sft
from .trainer import cosine_lr, save_pretrained, set_seed


@dataclass
class SFTConfig:
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 300
    learning_rate: float = 1e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 20
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 50
    save_interval: int = 100
    max_length: int = 256
    val_ratio: float = 0.1
    seed: int = 1337
    device: str = "auto"
    out_dir: str = "checkpoints/tiny-cpu-sft"

    def to_dict(self) -> dict:
        return asdict(self)


class SFTTrainer:
    def __init__(
        self,
        model: CortexForCausalLM,
        tokenizer: CortexTokenizer,
        dataset: ChatSFTDataset,
        config: SFTConfig,
        val_dataset: Optional[ChatSFTDataset] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device(
            "cuda" if config.device == "auto" and torch.cuda.is_available() else
            ("cpu" if config.device == "auto" else config.device)
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "sft_log.jsonl"

    def _loader(self, dataset: ChatSFTDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            collate_fn=lambda batch: collate_sft(batch, self.tokenizer.pad_token_id),
        )

    def train(self) -> Dict[str, float]:
        set_seed(self.config.seed)
        loader = self._loader(self.dataset, shuffle=True)
        iterator = iter(loader)
        self.model.train()
        started = time.time()
        last_loss = float("nan")
        for step in range(self.config.max_steps):
            lr = cosine_lr(
                step,
                self.config.max_steps,
                self.config.warmup_steps,
                self.config.learning_rate,
                self.config.min_lr_ratio,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(self.config.grad_accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
                (loss / self.config.grad_accum_steps).backward()
                loss_value += float(loss.detach()) / self.config.grad_accum_steps
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            last_loss = loss_value
            if (step + 1) % 10 == 0 or step == 0:
                record = {"step": step + 1, "loss": round(loss_value, 5), "lr": round(lr, 8)}
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if self.val_dataset is not None and (step + 1) % self.config.eval_interval == 0:
                metrics = self.evaluate()
                print(json.dumps({"step": step + 1, **metrics}), flush=True)
                self.model.train()
            if (step + 1) % self.config.save_interval == 0:
                save_pretrained(self.model, self.tokenizer, self.model.config, self.out_dir / f"checkpoint-{step + 1}")
        save_pretrained(self.model, self.tokenizer, self.model.config, self.out_dir)
        return {
            "steps": self.config.max_steps,
            "final_loss": last_loss,
            "elapsed_seconds": round(time.time() - started, 1),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_dataset is None:
            return {"val_loss": float("nan")}
        self.model.eval()
        loader = self._loader(self.val_dataset, shuffle=False)
        total, count = 0.0, 0
        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            loss = self.model(**batch)["loss"]
            total += float(loss.detach())
            count += 1
        loss = total / max(1, count)
        return {"val_loss": round(loss, 5), "val_perplexity": round(math.exp(loss), 3)}


def split_dataset(dataset: ChatSFTDataset, val_ratio: float) -> tuple[ChatSFTDataset, ChatSFTDataset]:
    import random

    indices = list(range(len(dataset)))
    random.Random(1337).shuffle(indices)
    split = max(1, int(len(indices) * val_ratio))
    val_indices = set(indices[:split])

    class Subset(ChatSFTDataset):
        def __init__(self, base, chosen):
            self.tokenizer = base.tokenizer
            self.max_length = base.max_length
            self.records = [base.records[index] for index in chosen]

    return (
        Subset(dataset, [index for index in indices if index not in val_indices]),
        Subset(dataset, sorted(val_indices)),
    )
