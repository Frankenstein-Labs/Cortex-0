"""Reproducible pretraining loop with checkpointing and resume support."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..config import CortexConfig, TrainingConfig
from ..model.transformer import CortexForCausalLM
from ..tokenizer.bpe import CortexTokenizer
from .data import TokenDataset, make_dataloader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def cosine_lr(step: int, max_steps: int, warmup: int, base_lr: float, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return base_lr * min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * coeff)


def save_pretrained(
    model: CortexForCausalLM,
    tokenizer: CortexTokenizer,
    config: CortexConfig,
    directory: str | Path,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config.save_json(directory / "config.json")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    save_file(state, str(directory / "model.safetensors"))
    tokenizer.save(directory / "tokenizer")
    (directory / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "max_new_tokens": 128,
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 40,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


def load_pretrained(directory: str | Path) -> tuple[CortexForCausalLM, CortexTokenizer, CortexConfig]:
    directory = Path(directory)
    config = CortexConfig.load_json(directory / "config.json")
    model = CortexForCausalLM(config)
    state = load_file(str(directory / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()
    tokenizer = CortexTokenizer.from_pretrained(directory / "tokenizer")
    return model, tokenizer, config


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    history: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Trainer:
    def __init__(
        self,
        model: CortexForCausalLM,
        tokenizer: CortexTokenizer,
        train_dataset: TokenDataset,
        val_dataset: Optional[TokenDataset],
        config: TrainingConfig,
        out_dir: Optional[str | Path] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.out_dir = Path(out_dir or config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        self.model.to(self.device)
        self.state = TrainState()
        self.optimizer = self._build_optimizer()
        # fp16 on GPU needs loss scaling or gradients underflow; bf16 does not.
        self._scaler = (
            torch.amp.GradScaler("cuda")
            if self.device.type == "cuda" and self.dtype == torch.float16
            else None
        )
        self._log_path = self.out_dir / "logs" / "train.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.out_dir / "tensorboard"))
            except ImportError:
                pass

    # ------------------------------------------------------------------ setup
    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)
        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype != torch.float32:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    def _infinite_loader(self) -> Iterator[dict]:
        loader = make_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        while True:
            for batch in loader:
                yield {
                    "input_ids": batch[0].to(self.device),
                    "labels": batch[1].to(self.device),
                }

    # ------------------------------------------------------------------ train
    def train(self) -> Dict[str, float]:
        cfg = self.config
        set_seed(cfg.seed)
        if cfg.resume:
            self._resume()

        data_iter = self._infinite_loader()
        self.model.train()
        optimizer = self.optimizer
        started = time.time()
        tokens_per_step = cfg.batch_size * self.train_dataset.block_size * cfg.grad_accum_steps
        last_loss = float("nan")

        # Count only this run's tokens so that resuming does not inflate the rate.
        tokens_this_run = 0

        for step in range(self.state.step, cfg.max_steps):
            lr = cosine_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.learning_rate, cfg.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(cfg.grad_accum_steps):
                batch = next(data_iter)
                with self._autocast():
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                scaled = loss / cfg.grad_accum_steps
                if self._scaler is not None:
                    self._scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                loss_value += float(loss.detach()) / cfg.grad_accum_steps

            if self._scaler is not None:
                self._scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if self._scaler is not None:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()
            self.state.step = step + 1
            self.state.tokens_seen += tokens_per_step
            tokens_this_run += tokens_per_step
            last_loss = loss_value

            if (step + 1) % cfg.log_interval == 0 or step == 0:
                elapsed = max(1e-6, time.time() - started)
                record = {
                    "step": step + 1,
                    "loss": round(loss_value, 5),
                    "lr": round(lr, 8),
                    "grad_norm": round(float(grad_norm), 4),
                    "tokens_per_second": round(tokens_this_run / elapsed, 1),
                }
                self._log(record)

            if self.val_dataset is not None and (step + 1) % cfg.eval_interval == 0:
                metrics = self.evaluate()
                self._log({"step": step + 1, **metrics})
                if metrics["val_loss"] < self.state.best_val_loss:
                    self.state.best_val_loss = metrics["val_loss"]
                    self.save(self.out_dir / "best")
                self.model.train()

            if (step + 1) % cfg.save_interval == 0:
                self.save(self.out_dir / f"checkpoint-{step + 1}")

        self.save(self.out_dir)
        if self._writer is not None:
            self._writer.close()
        elapsed = time.time() - started
        return {
            "steps": self.state.step,
            "final_loss": last_loss,
            "best_val_loss": self.state.best_val_loss,
            "tokens_seen": self.state.tokens_seen,
            "elapsed_seconds": round(elapsed, 1),
        }

    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        if self.val_dataset is None:
            return {"val_loss": float("nan"), "val_perplexity": float("nan")}
        self.model.eval()
        loader = make_dataloader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        total_loss = 0.0
        total_tokens = 0
        for idx, (x, y) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            with self._autocast():
                outputs = self.model(input_ids=x, labels=y)
            n_tokens = y.numel()
            total_loss += float(outputs["loss"].detach()) * n_tokens
            total_tokens += n_tokens
        loss = total_loss / max(1, total_tokens)
        return {"val_loss": round(loss, 5), "val_perplexity": round(math.exp(loss), 3)}

    # ------------------------------------------------------------- checkpoint
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_pretrained(self.model, self.tokenizer, self.model.config, directory)
        torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "scaler": self._scaler.state_dict() if self._scaler else None,
            },
            directory / "rng_state.pt",
        )
        (directory / "trainer_state.json").write_text(
            json.dumps(self.state.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return directory

    def _resume(self) -> None:
        state_path = self.out_dir / "trainer_state.json"
        weights = self.out_dir / "model.safetensors"
        if not weights.exists():
            return
        # Weights without state would silently restart from step 0, losing the
        # run; fail loudly instead.
        if not state_path.exists():
            raise RuntimeError(
                f"checkpoint at {self.out_dir} has weights but no trainer_state.json; "
                "refusing to restart silently would lose progress"
            )
        self.state = TrainState(**json.loads(state_path.read_text(encoding="utf-8")))
        loaded = load_file(str(weights))
        missing, unexpected = self.model.load_state_dict(loaded, strict=False)
        if self.model.config.tie_word_embeddings:
            self.model.lm_head.weight = self.model.model.embed_tokens.weight
        optimizer_path = self.out_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        rng_path = self.out_dir / "rng_state.pt"
        if rng_path.exists():
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            if self._scaler is not None and rng.get("scaler") is not None:
                self._scaler.load_state_dict(rng["scaler"])
        self._log({"resumed_from_step": self.state.step, "missing": list(missing)})

    def _log(self, record: Dict[str, float]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        if self._writer is not None:
            step = int(record.get("step", 0))
            for key, value in record.items():
                if key != "step" and isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, step)


def load_dataset_pair(
    train_path: str | Path,
    val_path: str | Path,
    block_size: int,
    dtype: str = "uint16",
) -> tuple[TokenDataset, Optional[TokenDataset]]:
    train = TokenDataset(train_path, block_size=block_size, dtype=dtype)
    val = TokenDataset(val_path, block_size=block_size, dtype=dtype) if Path(val_path).exists() else None
    return train, val
