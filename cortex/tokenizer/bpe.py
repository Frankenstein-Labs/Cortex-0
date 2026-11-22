"""Byte-level BPE tokenizer for CORTEX-0, built with Hugging Face Tokenizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|end|>"]


class CortexTokenizer:
    """Thin wrapper around a byte-level BPE model with a chat template."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self.pad_id = self._tok.token_to_id("<|pad|>")
        self.bos_id = self._tok.token_to_id("<|bos|>")
        self.eos_id = self._tok.token_to_id("<|eos|>")
        self.user_id = self._tok.token_to_id("<|user|>")
        self.assistant_id = self._tok.token_to_id("<|assistant|>")
        self.end_id = self._tok.token_to_id("<|end|>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8192,
        save_path: str | Path | None = None,
        min_frequency: int = 2,
        sample_chars: int | None = None,
    ) -> "CortexTokenizer":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )

        def stream() -> Iterable[str]:
            used = 0
            for text in texts:
                if sample_chars is not None and used >= sample_chars:
                    return
                used += len(text)
                yield text

        tok.train_from_iterator(stream(), trainer=trainer)
        wrapper = cls(tok)
        if save_path is not None:
            wrapper.save(save_path)
        return wrapper

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "CortexTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: Sequence[str], add_special_tokens: bool = False) -> list[list[int]]:
        encodings = self._tok.encode_batch(list(texts), add_special_tokens=add_special_tokens)
        return [e.ids for e in encodings]

    def chat_prompt(self, messages: Sequence[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Render a simple, explicit chat format the model can be fine-tuned on."""
        parts: list[str] = ["<|bos|>"]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            tag = "<|user|>" if role == "user" else "<|assistant|>"
            parts.append(f"{tag}{content}<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)

    def encode_chat(self, messages: Sequence[dict[str, str]], add_generation_prompt: bool = True) -> list[int]:
        return self.encode(self.chat_prompt(messages, add_generation_prompt))

    def info(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": {t: self._tok.token_to_id(t) for t in SPECIAL_TOKENS},
        }

    @classmethod
    def from_meta(cls, path: str | Path) -> tuple["CortexTokenizer", dict]:
        path = Path(path)
        tok = cls.load(path)
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return tok, meta

    def save_with_meta(self, path: str | Path, meta: dict) -> None:
        self.save(path)
        Path(path).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
