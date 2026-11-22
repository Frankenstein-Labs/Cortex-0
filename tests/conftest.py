from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortex.config import CortexConfig
from cortex.memory.store import MemoryStore
from cortex.model.transformer import CortexForCausalLM
from cortex.tokenizer.bpe import CortexTokenizer

CORPUS = [
    "CORTEX est une petite intelligence artificielle autonome.",
    "Frankenstein-Labs construit des systèmes capables de raisonner.",
    "Le chat mange une souris dans le jardin de la maison.",
    "La mémoire à long terme conserve les décisions importantes.",
    "Un objectif clair guide la planification et l'action.",
    "Les mathématiques demandent de la précision et de la vérification.",
    "The quick brown fox jumps over the lazy dog.",
    "A language model predicts the next token from previous tokens.",
    "Testing every component makes the system reliable.",
    "La curiosité pousse à poser de nouvelles questions.",
] * 20


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory) -> CortexTokenizer:
    directory = tmp_path_factory.mktemp("tokenizer")
    corpus_path = directory / "corpus.txt"
    corpus_path.write_text("\n".join(CORPUS), encoding="utf-8")
    tok = CortexTokenizer.train([corpus_path], vocab_size=512, min_frequency=1)
    tok.save(directory / "tokenizer")
    return CortexTokenizer.from_pretrained(directory / "tokenizer")


@pytest.fixture()
def tiny_config(tokenizer) -> CortexConfig:
    return CortexConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


@pytest.fixture()
def tiny_model(tiny_config) -> CortexForCausalLM:
    torch.manual_seed(0)
    model = CortexForCausalLM(tiny_config)
    model.eval()
    return model


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.sqlite")
