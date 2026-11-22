"""Memory-mapped token datasets used for pretraining and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TokenDataMeta:
    train_tokens: int
    val_tokens: int
    vocab_size: int
    dtype: str = "uint16"

    @classmethod
    def load(cls, path: str | Path) -> "TokenDataMeta":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            train_tokens=int(data["train_tokens"]),
            val_tokens=int(data["val_tokens"]),
            vocab_size=int(data["vocab_size"]),
            dtype=data.get("dtype", "uint16"),
        )


class TokenDataset:
    """Random fixed-length crops from a flat token file."""

    def __init__(self, bin_path: str | Path, seq_len: int, vocab_size: int, dtype: str = "uint16") -> None:
        self.path = Path(bin_path)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = np.memmap(self.path, dtype=np.dtype(dtype), mode="r")
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{self.path} has too few tokens ({len(self.data)}) for seq_len={seq_len}")

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.seq_len - 1
        if generator is None:
            starts = np.random.randint(0, max_start, size=batch_size)
        else:
            starts = torch.randint(0, max_start, (batch_size,), generator=generator).numpy()
        x = np.stack([self.data[s : s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1 : s + 1 + self.seq_len] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def sequential_batches(self, batch_size: int, max_batches: int | None = None):
        """Yield deterministic non-overlapping windows, useful for evaluation."""
        usable = (len(self.data) - 1) // self.seq_len * self.seq_len
        n_windows = usable // self.seq_len
        produced = 0
        for start in range(0, n_windows - batch_size + 1, batch_size):
            xs, ys = [], []
            for i in range(batch_size):
                s = (start + i) * self.seq_len
                xs.append(self.data[s : s + self.seq_len])
                ys.append(self.data[s + 1 : s + 1 + self.seq_len])
            yield torch.from_numpy(np.stack(xs).astype(np.int64)), torch.from_numpy(np.stack(ys).astype(np.int64))
            produced += 1
            if max_batches is not None and produced >= max_batches:
                return


def write_token_file(path: str | Path, token_chunks, vocab_size: int) -> int:
    """Write concatenated token arrays to a flat binary file. Returns token count."""
    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("wb") as fh:
        for chunk in token_chunks:
            arr = np.asarray(chunk, dtype=dtype)
            fh.write(arr.tobytes())
            total += len(arr)
    return total
