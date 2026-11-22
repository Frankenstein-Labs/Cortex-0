from pathlib import Path

import torch
from torch.utils.data import Dataset

from cortex.tokenizer import ByteTokenizer


class CausalTextDataset(Dataset):
    """Slices a text corpus into fixed-length input/target pairs."""
    def __init__(self, path: str | Path, context_size: int, tokenizer: ByteTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or ByteTokenizer()
        text = Path(path).read_text(encoding="utf-8")
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(tokens) < context_size + 1:
            raise ValueError("corpus must contain at least context_size + 1 tokens")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_size = context_size

    def __len__(self) -> int:
        return self.tokens.numel() - self.context_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.tokens[index:index + self.context_size + 1]
        return window[:-1], window[1:]
