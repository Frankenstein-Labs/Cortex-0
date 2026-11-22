"""Reproducible pretraining loop with checkpointing and honest metrics."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cortex.config import CortexConfig, TrainConfig
from cortex.model.lm import CortexLM, save_checkpoint
from cortex.training.data import TokenDataset


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[requested]


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.learning_rate * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.learning_rate * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * coeff)


@dataclass
class EvalResult:
    loss: float
    perplexity: float
    bits_per_token: float
    token_accuracy: float
    batches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": round(self.loss, 6),
            "perplexity": round(self.perplexity, 4),
            "bits_per_token": round(self.bits_per_token, 6),
            "token_accuracy": round(self.token_accuracy, 6),
            "batches": self.batches,
        }


@torch.no_grad()
def evaluate(
    model: CortexLM,
    dataset: TokenDataset,
    batch_size: int,
    device: torch.device,
    max_batches: int = 20,
    autocast_dtype: torch.dtype | None = None,
) -> EvalResult:
    was_training = model.training
    model.eval()
    losses, accs, batches = [], [], 0
    for x, y in dataset.sequential_batches(batch_size, max_batches=max_batches):
        x, y = x.to(device), y.to(device)
        ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype else _null()
        with ctx:
            out = model(x, targets=y)
        losses.append(float(out["loss"]))
        accs.append(float(out["token_accuracy"]))
        batches += 1
    if was_training:
        model.train()
    if not losses:
        raise RuntimeError("evaluation dataset produced no batches")
    mean_loss = sum(losses) / len(losses)
    return EvalResult(
        loss=mean_loss,
        perplexity=math.exp(min(mean_loss, 50)),
        bits_per_token=mean_loss / math.log(2),
        token_accuracy=sum(accs) / len(accs),
        batches=batches,
    )


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class Trainer:
    def __init__(
        self,
        model: CortexLM,
        train_cfg: TrainConfig,
        train_data: TokenDataset,
        val_data: TokenDataset | None = None,
        output_dir: str | Path = "checkpoints",
        tokenizer_meta: dict | None = None,
    ) -> None:
        self.model = model
        self.cfg = train_cfg
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = Path(output_dir) / train_cfg.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_meta = tokenizer_meta or {}

        self.device = resolve_device(train_cfg.device)
        self.dtype = resolve_dtype(train_cfg.dtype, self.device)
        self.model.to(self.device)
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self._param_groups(),
            lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2),
            weight_decay=train_cfg.weight_decay,
            fused=False,
        )
        self.step = 0
        self.tokens_seen = 0
        self.best_val: float | None = None
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.metrics_path.touch(exist_ok=True)

    def _param_groups(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2 and "emb" not in name and "norm" not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        return [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _log(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def save(self, name: str, val: EvalResult | None = None) -> Path:
        ckpt = self.output_dir / f"{name}.safetensors"
        extra = {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "train_config": self.cfg.to_dict(),
            "tokenizer": self.tokenizer_meta,
            "val": val.to_dict() if val else None,
        }
        save_checkpoint(self.model, ckpt, extra=extra)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "tokens_seen": self.tokens_seen,
                "best_val": self.best_val,
            },
            self.output_dir / f"{name}.train.pt",
        )
        (self.output_dir / "latest.json").write_text(
            json.dumps({"checkpoint": str(ckpt), "step": self.step, **extra}, indent=2) + "\n",
            encoding="utf-8",
        )
        return ckpt

    def resume(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state["step"])
        self.tokens_seen = int(state.get("tokens_seen", 0))
        self.best_val = state.get("best_val")

    def train(self, on_step=None) -> dict[str, Any]:
        cfg = self.cfg
        generator = torch.Generator().manual_seed(cfg.seed)
        autocast_dtype = self.dtype if self.dtype != torch.float32 else None
        tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.seq_len
        started = time.time()
        last_log = started
        running_loss = 0.0
        running_acc = 0.0
        log_count = 0

        self.model.train()
        while self.step < cfg.max_steps:
            lr = cosine_lr(self.step, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_acc = 0.0
            for _ in range(cfg.grad_accum):
                x, y = self.train_data.get_batch(cfg.batch_size, generator)
                x, y = x.to(self.device), y.to(self.device)
                ctx = (
                    torch.autocast(device_type=self.device.type, dtype=autocast_dtype)
                    if autocast_dtype
                    else _null()
                )
                with ctx:
                    out = self.model(x, targets=y)
                    loss = out["loss"] / cfg.grad_accum
                loss.backward()
                step_loss += float(out["loss"].detach())
                step_acc += float(out["token_accuracy"].detach())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            self.step += 1
            self.tokens_seen += tokens_per_step
            step_loss /= cfg.grad_accum
            step_acc /= cfg.grad_accum
            running_loss += step_loss
            running_acc += step_acc
            log_count += 1

            if self.step % cfg.log_interval == 0 or self.step == 1:
                now = time.time()
                elapsed = max(1e-6, now - last_log)
                record = {
                    "step": self.step,
                    "lr": round(lr, 8),
                    "train_loss": round(running_loss / log_count, 6),
                    "train_token_accuracy": round(running_acc / log_count, 6),
                    "tokens_seen": self.tokens_seen,
                    "tokens_per_sec": round((cfg.log_interval * tokens_per_step) / elapsed, 2)
                    if self.step > 1
                    else None,
                    "elapsed_sec": round(now - started, 2),
                }
                self._log(record)
                print(json.dumps(record), flush=True)
                if on_step:
                    on_step(record)
                running_loss = running_acc = 0.0
                log_count = 0
                last_log = now

            if self.val_data is not None and (self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps):
                val = evaluate(
                    self.model,
                    self.val_data,
                    cfg.batch_size,
                    self.device,
                    max_batches=cfg.eval_batches,
                    autocast_dtype=autocast_dtype,
                )
                record = {"step": self.step, "val": val.to_dict(), "elapsed_sec": round(time.time() - started, 2)}
                self._log(record)
                print(json.dumps(record), flush=True)
                improved = self.best_val is None or val.loss < self.best_val
                if improved:
                    self.best_val = val.loss
                    self.save("best", val)
                self.save("latest", val)
                self.model.train()

        if self.val_data is None:
            self.save("final")
        summary = {
            "run_name": cfg.run_name,
            "steps": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val,
            "wall_time_sec": round(time.time() - started, 2),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "parameters": self.model.num_parameters(),
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary


def build_model(cfg: CortexConfig) -> CortexLM:
    return CortexLM(cfg)
