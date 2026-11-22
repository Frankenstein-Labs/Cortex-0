"""Datasets for pretraining and supervised fine-tuning."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..tokenizer.bpe import CortexTokenizer


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size <= 65535 else np.dtype(np.uint32)


def tokenize_documents(
    tokenizer: CortexTokenizer,
    documents: Iterable[str],
    out_path: str | Path,
    add_eos: bool = True,
    batch_docs: int = 128,
    dtype: Optional[np.dtype] = None,
) -> Dict[str, int]:
    """Tokenize documents into a flat binary token file.

    Writing incrementally keeps memory bounded even for multi-gigabyte corpora.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = dtype or dtype_for_vocab(tokenizer.vocab_size)
    total_tokens = 0
    total_docs = 0
    buffer: List[str] = []

    def flush() -> None:
        nonlocal total_tokens, total_docs
        if not buffer:
            return
        encoded = tokenizer.encode_batch(buffer, add_eos=add_eos)
        with out_path.open("ab") as handle:
            for ids in encoded:
                np.asarray(ids, dtype=dtype).tofile(handle)
                total_tokens += len(ids)
        total_docs += len(buffer)
        buffer.clear()

    out_path.write_bytes(b"")
    for doc in documents:
        if not doc:
            continue
        buffer.append(doc)
        if len(buffer) >= batch_docs:
            flush()
    flush()
    return {"documents": total_docs, "tokens": total_tokens, "dtype": str(dtype)}


class TokenDataset(Dataset):
    """Fixed-length next-token-prediction windows over a memmapped token file."""

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        stride: Optional[int] = None,
        dtype: str = "uint16",
    ) -> None:
        self.path = str(path)
        self.block_size = int(block_size)
        self.stride = int(stride or block_size)
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        if len(self.data) < self.block_size + 1:
            raise ValueError(
                f"{path} has {len(self.data)} tokens, need at least {self.block_size + 1}"
            )
        self.num_windows = (len(self.data) - 1 - self.block_size) // self.stride + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.num_windows:
            raise IndexError(index)
        start = index * self.stride
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


class ChatSFTDataset(Dataset):
    """Supervised fine-tuning dataset over ``{"messages": [...]}`` JSONL."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: CortexTokenizer,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: List[List[Dict[str, str]]] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                messages = payload.get("messages", payload)
                if not isinstance(messages, list):
                    raise ValueError(f"invalid SFT record in {path}")
                self.records.append(messages)
        if not self.records:
            raise ValueError(f"empty SFT dataset: {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        ids, labels = self.tokenizer.encode_supervised(self.records[index])
        ids = ids[: self.max_length]
        labels = labels[: self.max_length]
        return {"input_ids": ids, "labels": labels}


def collate_sft(batch: Sequence[Dict[str, List[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        # The model computes an unshifted loss, so shift targets left here:
        # position t must predict the label attached to position t + 1.
        if length > 1:
            labels[row, : length - 1] = torch.tensor(item["labels"][1:], dtype=torch.long)
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
    )


def dataset_summary(path: str | Path, block_size: int, dtype: str = "uint16") -> Dict[str, float]:
    ds = TokenDataset(path, block_size=block_size, dtype=dtype)
    return {
        "tokens": int(len(ds.data)),
        "windows": int(len(ds)),
        "approx_epochs_per_1k_steps": math.ceil(1000 / max(1, len(ds))),
    }
