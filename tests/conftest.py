from __future__ import annotations

import pytest

from cortex.config import CortexConfig
from cortex.model.lm import CortexLM
from cortex.tokenizer.bpe import CortexTokenizer

SAMPLE_TEXTS = [
    "Once upon a time, there was a little girl named Lily. She found a shiny red ball in the garden.",
    "Tom the cat liked to sleep in the sun. One day he met a friendly dog named Max near the river.",
    "The little robot Cortex wanted to learn new things. It read a book about stars and planets.",
    "A small bird lived in a tall tree. Every morning she sang a happy song for the forest.",
    "Ben and Anna built a sandcastle on the beach. A wave came and washed it away, so they laughed.",
    "The moon was bright that night. A fox walked through the snow and looked at the sky.",
    "Sara had a blue kite. The wind lifted it high above the green hills and the old house.",
    "A turtle walked slowly along the path. He met a rabbit who was in a hurry to reach the lake.",
] * 12


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory) -> CortexTokenizer:
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    return CortexTokenizer.train(SAMPLE_TEXTS, vocab_size=320, save_path=path)


@pytest.fixture
def tiny_cfg(tiny_tokenizer) -> CortexConfig:
    return CortexConfig(
        vocab_size=tiny_tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_cfg) -> CortexLM:
    return CortexLM(tiny_cfg)
