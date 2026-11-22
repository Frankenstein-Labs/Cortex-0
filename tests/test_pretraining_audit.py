"""Regression tests for the pretraining/SFT audit findings."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F

from cortex.config import CortexConfig
from cortex.model.transformer import CortexForCausalLM
from cortex.tokenizer.bpe import CortexTokenizer
from cortex.training.data import PlainSFTDataset, collate_sft


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "datasets/cortex-v2/tokenizer"
SFT = ROOT / "datasets/cortex-v2/sft-v6.jsonl"
EVAL = ROOT / "datasets/eval/cortex-100-fr.jsonl"


def tiny_model(vocab_size: int = 128) -> CortexForCausalLM:
    config = CortexConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
    )
    return CortexForCausalLM(config).eval()


def test_causal_mask_has_no_future_leakage():
    torch.manual_seed(0)
    model = tiny_model()
    ids = torch.randint(3, 100, (1, 8))
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 100
    with torch.no_grad():
        before = model(input_ids=ids)["logits"]
        after = model(input_ids=changed)["logits"]
    assert torch.equal(before[:, :-1], after[:, :-1])


def test_padding_does_not_change_valid_logits():
    torch.manual_seed(1)
    model = tiny_model()
    ids = torch.randint(3, 100, (1, 6))
    padded = torch.cat([ids, torch.zeros((1, 4), dtype=torch.long)], dim=1)
    mask = torch.tensor([[1] * 6 + [0] * 4])
    with torch.no_grad():
        short = model(input_ids=ids)["logits"]
        long = model(input_ids=padded, attention_mask=mask)["logits"]
    assert torch.allclose(short, long[:, :6], atol=1e-5)


def test_pretraining_labels_are_next_token_targets():
    torch.manual_seed(2)
    model = tiny_model()
    ids = torch.randint(3, 100, (1, 9))
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        logits = model(input_ids=x)["logits"]
        loss = model(input_ids=x, labels=y)["loss"]
    manual = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    assert torch.allclose(loss, manual)


def test_tokenizer_roundtrip_and_special_ids():
    tokenizer = CortexTokenizer.from_pretrained(TOKENIZER)
    samples = [
        "L'été à Dakar est chaud et humide.",
        "def carre(x):\n    return x * x",
        "soussou: i nu xa siga?",
        "bambara: ne bɛ taa sugu la.",
    ]
    for text in samples:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2


def test_sft_labels_mask_the_question_and_keep_the_answer():
    tokenizer = CortexTokenizer.from_pretrained(TOKENIZER)
    dataset = PlainSFTDataset(SFT, tokenizer, max_length=256)
    item = dataset[0]
    batch = collate_sft([item], tokenizer.pad_token_id)
    labels = batch["labels"][0]
    supervised = [int(x) for x in labels if int(x) != -100]
    decoded = tokenizer.decode(supervised)
    answer = " ".join(
        m["content"] for m in dataset.records[0] if m["role"] == "assistant"
    )
    assert decoded.strip() == answer.strip()
    assert int((labels != -100).sum()) < len(item["input_ids"])


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return " ".join(text.split())


def test_sft_v6_is_balanced_and_uncontaminated():
    rows = [json.loads(line) for line in SFT.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [
        next(m["content"] for m in row["messages"] if m["role"] == "user")
        for row in rows
    ]
    assert len(prompts) == len(set(prompts))
    refusals = sum(row["kind"] == "refusal" for row in rows)
    assert refusals / len(rows) <= 0.05
    eval_prompts = {
        _normalize(json.loads(line)["prompt"])
        for line in EVAL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert not (set(map(_normalize, prompts)) & eval_prompts)


def test_sft_v6_code_examples_execute():
    rows = [json.loads(line) for line in SFT.read_text(encoding="utf-8").splitlines() if line.strip()]
    code_rows = [row for row in rows if row["kind"] == "code"]
    assert len(code_rows) >= 20
    for row in code_rows[:5]:
        code = row["messages"][-1]["content"]
        namespace: dict = {}
        exec(compile(code, "<sft>", "exec"), namespace)
        assert any(callable(value) for value in namespace.values())
