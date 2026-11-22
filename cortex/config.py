"""Configuration objects for CORTEX-0.

Every experiment is fully described by a JSON/YAML file so that training runs
are reproducible.  ``CortexConfig`` mirrors the Llama architecture so that a
trained checkpoint can be exported to Hugging Face ``transformers`` without
changing tensor shapes or names (see ``scripts/export_hf.py``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class CortexConfig:
    """Architecture hyper-parameters of a CORTEX decoder-only Transformer."""

    vocab_size: int = 4096
    hidden_size: int = 256
    intermediate_size: int = 688
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    hidden_act: str = "silu"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.validate()

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_act not in {"silu", "gelu"}:
            raise ValueError(f"unsupported hidden_act: {self.hidden_act}")
        for name in ("vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_parameters(self) -> int:
        """Parameter count, counting tied embeddings once."""
        embed = self.vocab_size * self.hidden_size
        q = self.hidden_size * self.num_attention_heads * self.head_dim
        kv = self.hidden_size * self.num_key_value_heads * self.head_dim
        o = self.num_attention_heads * self.head_dim * self.hidden_size
        mlp = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size
        per_layer = q + 2 * kv + o + mlp + norms
        total = embed + self.num_hidden_layers * per_layer + self.hidden_size
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.hidden_size
        return total

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CortexConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown CortexConfig fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load_json(cls, path: str | Path) -> "CortexConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class TrainingConfig:
    """Hyper-parameters for the pretraining loop."""

    batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 1000
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_steps: int = 20
    log_interval: int = 10
    save_interval: int = 250
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"  # auto | float32 | bfloat16 | float16
    out_dir: str = "checkpoints/tiny-cpu"
    tensorboard: bool = False
    compile: bool = False
    resume: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown TrainingConfig fields: {sorted(unknown)}")
        return cls(**data)


@dataclass
class ExperimentConfig:
    model: CortexConfig = field(default_factory=CortexConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    train_data: str = "datasets/processed/train.bin"
    val_data: str = "datasets/processed/val.bin"
    tokenizer_dir: str = "datasets/processed/tokenizer"

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
        return cls(
            model=CortexConfig.from_dict(data["model"]),
            training=TrainingConfig.from_dict(data.get("training", {})),
            train_data=data.get("train_data", "datasets/processed/train.bin"),
            val_data=data.get("val_data", "datasets/processed/val.bin"),
            tokenizer_dir=data.get("tokenizer_dir", "datasets/processed/tokenizer"),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "train_data": self.train_data,
            "val_data": self.val_data,
            "tokenizer_dir": self.tokenizer_dir,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# Architecture presets.  ``tiny-cpu`` is the one trained in this repository's
# smoke run; the larger presets are ready for GPU fine-tuning.
PRESETS: Dict[str, CortexConfig] = {
    "tiny-cpu": CortexConfig(
        vocab_size=4096,
        hidden_size=192,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=6,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    ),
    "cortex-50m": CortexConfig(
        vocab_size=16384,
        hidden_size=640,
        intermediate_size=1728,
        num_hidden_layers=10,
        num_attention_heads=10,
        num_key_value_heads=5,
        max_position_embeddings=1024,
    ),
    "cortex-150m": CortexConfig(
        vocab_size=32768,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    ),
}


def get_preset(name: str) -> CortexConfig:
    try:
        return CortexConfig.from_dict(PRESETS[name].to_dict())
    except KeyError as exc:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}") from exc
