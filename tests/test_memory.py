from __future__ import annotations

import time

import pytest

from cortex.memory.retrieval import cosine, jaccard
from cortex.memory.store import KINDS, MemoryStore


def test_add_get_and_search(store):
    memory_id = store.add("Le projet CORTEX-0 utilise un Transformer décodeur.", kind="project")
    hit = store.get(memory_id)
    assert hit is not None
    assert hit.kind == "project"
    assert hit.summary
    results = store.search("Transformer décodeur CORTEX", limit=5)
    assert results
    assert results[0].id == memory_id
    assert results[0].score > 0


def test_search_kind_filter(store):
    store.add("La mémoire sémantique stocke des faits.", kind="semantic")
    store.add("La mission est de construire une LLM.", kind="project")
    semantic = store.search("mémoire", kinds=["semantic"])
    assert semantic
    assert all(hit.kind == "semantic" for hit in semantic)


def test_importance_defaults_by_kind(store):
    preference_id = store.add("L'utilisateur préfère les réponses courtes.", kind="preference")
    working_id = store.add("Question en cours", kind="working")
    assert store.get(preference_id).importance > store.get(working_id).importance


def test_search_is_injection_safe(store):
    store.add("Une information importante sur le projet.", kind="project")
    results = store.search("'; DROP TABLE memories; --", limit=5)
    assert isinstance(results, list)
    assert store.stats()["memories"] == 1


def test_preferences_and_kv(store):
    store.set_preference("langue", "français")
    assert store.get_preference("langue") == "français"
    assert store.preferences()["langue"] == "français"
    store.kv_set("cursor", "42")
    assert store.kv_get("cursor") == "42"
    assert store.kv_get("missing", "default") == "default"


def test_events_and_working_memory(store):
    store.add_event("s1", "user", "Bonjour")
    store.add_event("s1", "assistant", "Bonjour, je suis CORTEX")
    events = store.recent_events("s1", limit=10)
    assert [event["role"] for event in events] == ["user", "assistant"]
    store.add("note temporaire", kind="working", session_id="s1")
    assert store.working_memory("s1")
    sessions = store.sessions()
    assert sessions[0]["session_id"] == "s1"


def test_consolidate_creates_summary(store):
    for index in range(8):
        store.add_event("s2", "user", f"Message numéro {index} sur le projet CORTEX.")
        store.add_event("s2", "assistant", f"Réponse {index} concernant le raisonnement.")
    result = store.consolidate(session_id="s2", min_events=6)
    assert result["events"] >= 8
    assert result["consolidated"] >= 1
    assert store.recent(kind="semantic", limit=5)


def test_retrieval_metrics():
    assert jaccard("bonjour le monde", "bonjour le monde") == 1.0
    assert jaccard("bonjour", "soir") == 0.0
    import numpy as np

    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_stats_counts_all_kinds(store):
    for kind in KINDS:
        store.add(f"Contenu de type {kind}", kind=kind)
    stats = store.stats()
    assert stats["memories"] == len(KINDS)
    assert set(stats["by_kind"]) == set(KINDS)
