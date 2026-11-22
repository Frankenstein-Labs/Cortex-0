"""The identity dataset must stay consistent with docs/IDENTITY.md."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "identity" / "cortex0_identity.jsonl"
DOC = ROOT / "docs" / "IDENTITY.md"

FACTS = {
    "CORTEX-0": "name",
    "Frankenstein-Labs": "organisation",
    "Abdoulaye Coumbassa": "creator",
    "Guinée": "origin",
    "français": "first language",
    "informatique": "second domain",
}


def load_records() -> list[dict]:
    with DATASET.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_dataset_exists_and_is_non_empty():
    assert DATASET.exists(), "identity dataset is missing"
    assert len(load_records()) >= 10


def test_every_record_has_the_three_roles():
    for record in load_records():
        roles = [m["role"] for m in record["messages"]]
        assert roles == ["system", "user", "assistant"], roles
        assert all(m["content"].strip() for m in record["messages"])


def test_all_identity_facts_are_taught():
    answers = " ".join(
        m["content"] for r in load_records() for m in r["messages"] if m["role"] == "assistant"
    )
    for fact, label in FACTS.items():
        assert fact in answers, f"dataset never states the {label}: {fact}"


def test_document_and_dataset_agree():
    doc = DOC.read_text(encoding="utf-8")
    for fact, label in FACTS.items():
        assert fact in doc, f"docs/IDENTITY.md is missing the {label}: {fact}"


def test_no_capability_is_claimed_that_does_not_exist():
    for record in load_records():
        user = record["messages"][1]["content"].lower()
        answer = record["messages"][2]["content"]
        if "voix" in user or "internet" in user:
            assert answer.strip().lower().startswith("non"), answer


@pytest.mark.parametrize("forbidden", ["huggingface.co", "http://", "https://"])
def test_no_links_or_secrets_in_identity_data(forbidden):
    text = DATASET.read_text(encoding="utf-8")
    assert forbidden not in text
