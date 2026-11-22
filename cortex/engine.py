"""Runtime container wiring model, tokenizer, memory, tasks and autonomy."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cortex.conversation.companion import Companion
from cortex.initiative.engine import ConsoleNotifier, InitiativeConfig, InitiativeEngine
from cortex.memory.store import MemoryStore
from cortex.planning.tasks import TaskStore
from cortex.tools.registry import default_registry


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    checkpoint: str = field(
        default_factory=lambda: os.environ.get("CORTEX_CHECKPOINT", "checkpoints/cortex-0-nano/best.safetensors")
    )
    tokenizer: str = field(
        default_factory=lambda: os.environ.get("CORTEX_TOKENIZER", "data/processed/tokenizer.json")
    )
    memory_db: str = field(default_factory=lambda: os.environ.get("CORTEX_MEMORY_DB", "artifacts/cortex_memory.db"))
    tasks_db: str = field(default_factory=lambda: os.environ.get("CORTEX_TASKS_DB", "artifacts/cortex_tasks.db"))
    task_checkpoints: str = field(
        default_factory=lambda: os.environ.get("CORTEX_TASK_CHECKPOINTS", "artifacts/tasks")
    )
    initiative_enabled: bool = field(default_factory=lambda: _env_bool("CORTEX_INITIATIVE_ENABLED", True))
    instruction_tuned: bool = field(default_factory=lambda: _env_bool("CORTEX_INSTRUCTION_TUNED", False))
    device: str = field(default_factory=lambda: os.environ.get("CORTEX_DEVICE", "cpu"))
    load_model: bool = True


class CortexEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.model = None
        self.tokenizer = None
        self.model_meta: dict[str, Any] = {}
        self.load_error: str | None = None
        self.memory = MemoryStore(self.settings.memory_db)
        self.tasks = TaskStore(self.settings.tasks_db, checkpoint_dir=self.settings.task_checkpoints)
        self.initiative = InitiativeEngine(
            self.memory,
            tasks=self.tasks,
            config=InitiativeConfig(enabled=self.settings.initiative_enabled),
            notifier=ConsoleNotifier(),
        )
        self.companion: Companion | None = None
        self.tools = default_registry(self.memory)
        if self.settings.load_model:
            self.load()

    def load(self) -> None:
        checkpoint = Path(self.settings.checkpoint)
        tokenizer = Path(self.settings.tokenizer)
        if not checkpoint.exists():
            self.load_error = f"checkpoint not found: {checkpoint}"
            return
        if not tokenizer.exists():
            self.load_error = f"tokenizer not found: {tokenizer}"
            return
        from cortex.model.lm import load_model
        from cortex.tokenizer.bpe import CortexTokenizer

        self.model = load_model(checkpoint, device=self.settings.device)
        self.tokenizer = CortexTokenizer.load(tokenizer)
        sidecar = checkpoint.with_suffix(".json")
        if sidecar.exists():
            self.model_meta = json.loads(sidecar.read_text(encoding="utf-8"))
        self.companion = Companion(
            self.model,
            self.tokenizer,
            self.memory,
            instruction_tuned=self.settings.instruction_tuned,
        )
        self.initiative.model = self.model
        self.initiative.tokenizer = self.tokenizer

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def status(self) -> dict:
        info: dict[str, Any] = {
            "name": "CORTEX-0",
            "loaded": self.loaded,
            "load_error": self.load_error,
            "checkpoint": self.settings.checkpoint,
            "tokenizer": self.settings.tokenizer,
            "instruction_tuned": self.settings.instruction_tuned,
            "memory": self.memory.stats(),
            "open_tasks": len(self.tasks.open_tasks()),
            "initiative": self.initiative.config.as_dict(),
            "device": self.settings.device,
        }
        if self.loaded:
            info["parameters"] = self.model.num_parameters()
            info["model_config"] = self.model.cfg.to_dict()
            info["training"] = self.model_meta.get("extra", {})
            info["vocab_size"] = self.tokenizer.vocab_size
        return info

    def _require_model(self) -> None:
        if not self.loaded:
            raise RuntimeError(self.load_error or "model is not loaded")

    def generate(self, prompt: str, **kwargs) -> dict:
        from cortex.generation.sampler import GenerationConfig, generate_text

        self._require_model()
        gen = GenerationConfig(**{k: v for k, v in kwargs.items() if v is not None})
        ids = self.tokenizer.encode(prompt)
        text, all_ids = generate_text(self.model, self.tokenizer, ids, gen)
        return {
            "prompt": prompt,
            "text": text,
            "prompt_tokens": len(ids),
            "generated_tokens": len(all_ids) - len(ids),
            "config": gen.__dict__,
        }

    def chat(self, message: str, session_id: str = "default", **kwargs) -> dict:
        self._require_model()
        from cortex.generation.sampler import GenerationConfig

        assert self.companion is not None
        gen = GenerationConfig(**{k: v for k, v in kwargs.items() if v is not None})
        return self.companion.reply(session_id, message, gen).as_dict()

    def tick(self, session_id: str = "default") -> dict:
        decision = self.initiative.tick(session_id=session_id)
        if decision.message and self.companion:
            self.companion.deliver_proactive(session_id, decision.message.text)
        return decision.as_dict()

    def close(self) -> None:
        self.memory.close()
        self.tasks.close()
