"""Model and training configuration for CORTEX-0."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CortexConfig:
    """Configuration of a decoder-only Transformer.

    Defaults describe the ``mini`` (~10M parameter) model that is small enough
    to train on CPU. Larger presets live in ``configs/``.
    """

    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_mult: float = 2.6875  # SwiGLU hidden size, rounded to a multiple of 64
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0
    init_std: float = 0.02
    bias: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        raw = int(self.ffn_mult * self.d_model)
        return max(64, 64 * round(raw / 64))

    def parameter_estimate(self) -> int:
        """Analytic parameter count (ignores norms and biases)."""
        emb = self.vocab_size * self.d_model
        qkv = self.d_model * self.head_dim * (self.n_heads + 2 * self.n_kv_heads)
        out = self.d_model * self.d_model
        mlp = 3 * self.d_model * self.ffn_hidden
        per_layer = qkv + out + mlp
        head = 0 if self.tie_embeddings else emb
        return emb + self.n_layers * per_layer + head + 2 * self.d_model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CortexConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "CortexConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "model" in data and isinstance(data["model"], dict):
            data = data["model"]
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass
class TrainConfig:
    seq_len: int = 256
    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 2000
    warmup_steps: int = 100
    learning_rate: float = 3e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    log_interval: int = 25
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"  # auto | float32 | bfloat16 | float16
    compile: bool = False
    checkpoint_dir: str = "checkpoints"
    run_name: str = "cortex-0-mini"
    tensorboard: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ExperimentConfig:
    model: CortexConfig = field(default_factory=CortexConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model=CortexConfig.from_dict(data.get("model", {})),
            train=TrainConfig.from_dict(data.get("train", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model.to_dict(), "train": self.train.to_dict()}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
