"""Byte-level BPE tokenizer and CORTEX chat template.

The tokenizer is trained locally with Hugging Face ``tokenizers`` (Rust BPE).
A byte-level alphabet guarantees that any UTF-8 input - French accents,
Malinke, Bambara, Soussou, code, emoji - is representable without an UNK token.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS: List[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|tool|>",
    "<|observation|>",
]

SPECIAL_TOKEN_IDS: Dict[str, int] = {name: idx for idx, name in enumerate(SPECIAL_TOKENS)}

CHAT_ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "observation": "<|observation|>",
}


class CortexTokenizer:
    """Thin, explicit wrapper around a byte-level BPE model."""

    def __init__(self, backend: Tokenizer) -> None:
        self._tok = backend

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        files: Sequence[str | Path],
        vocab_size: int = 4096,
        min_frequency: int = 2,
    ) -> "CortexTokenizer":
        if vocab_size < 512:
            raise ValueError("vocab_size must be >= 512 for byte-level BPE")
        backend = Tokenizer(BPE(unk_token=None))
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            min_frequency=min_frequency,
            show_progress=False,
        )
        backend.train([str(f) for f in files], trainer)
        return cls(backend)

    # ------------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return SPECIAL_TOKEN_IDS["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        return SPECIAL_TOKEN_IDS["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        return SPECIAL_TOKEN_IDS["<|eos|>"]

    @property
    def end_token_id(self) -> int:
        return SPECIAL_TOKEN_IDS["<|end|>"]

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    # ---------------------------------------------------------------- encode
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def encode_batch(self, texts: Iterable[str], add_eos: bool = False) -> List[List[int]]:
        encodings = self._tok.encode_batch(list(texts), add_special_tokens=False)
        out = [enc.ids for enc in encodings]
        if add_eos:
            out = [ids + [self.eos_token_id] for ids in out]
        return out

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def encode_supervised(
        self, messages: Sequence[Dict[str, str]], add_bos: bool = True
    ) -> tuple[List[int], List[int]]:
        """Encode a conversation and mask every token except assistant answers.

        Labels are ``-100`` on system/user turns and on the assistant role
        marker, so the loss only trains the model on what it must generate.
        """
        ids: List[int] = []
        labels: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
            labels.append(-100)
        for message in messages:
            role = message.get("role")
            if role not in CHAT_ROLE_TOKENS:
                raise ValueError(f"unknown chat role: {role!r}")
            prefix = self.encode(CHAT_ROLE_TOKENS[role])
            content = self.encode(message.get("content", ""))
            end = self.encode("<|end|>")
            ids.extend(prefix + content + end)
            if role == "assistant":
                labels.extend([-100] * len(prefix) + content + end)
            else:
                labels.extend([-100] * (len(prefix) + len(content) + len(end)))
        return ids, labels

    # ----------------------------------------------------------- chat format
    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = True,
        add_bos: bool = True,
    ) -> str:
        """Render a conversation in the CORTEX chat format.

        ``messages`` is a list of ``{"role": ..., "content": ...}`` dictionaries.
        Unknown roles raise, which prevents silent prompt corruption.
        """
        parts: List[str] = ["<|bos|>"] if add_bos else []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role not in CHAT_ROLE_TOKENS:
                raise ValueError(f"unknown chat role: {role!r}")
            parts.append(f"{CHAT_ROLE_TOKENS[role]}{content}<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)

    def encode_chat(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        return self.encode(
            self.apply_chat_template(messages, add_generation_prompt=add_generation_prompt),
            add_special_tokens=False,
        )

    # ----------------------------------------------------------------- io
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(directory / "tokenizer.json"))
        meta = {
            "type": "cortex-byte-level-bpe",
            "special_tokens": SPECIAL_TOKENS,
            "vocab_size": self.vocab_size,
        }
        (directory / "cortex_tokenizer.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "CortexTokenizer":
        path = Path(directory) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"no tokenizer.json in {directory}")
        return cls(Tokenizer.from_file(str(path)))
