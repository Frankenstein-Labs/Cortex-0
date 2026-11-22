from __future__ import annotations

from cortex.memory.store import MemoryStore, WorkingMemory


def test_episodes_and_retrieval(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.add_episode("s1", "user", "We should implement a reasoning engine for arithmetic.")
    store.add_episode("s1", "assistant", "The garden is full of red flowers.")
    store.add_episode("s2", "user", "The deployment needs a database migration.")
    hits = store.search_episodes("arithmetic reasoning", limit=1)
    assert hits and "arithmetic" in hits[0].content
    assert len(store.recent_episodes("s1")) == 2
    assert store.stats()["episodes"] == 3


def test_facts_and_preferences(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.add_fact("cortex", "uses RoPE", source="docs")
    store.add_fact("cortex", "uses RoPE", source="docs")  # unique constraint
    assert len(store.get_facts("cortex")) == 1
    store.set_preference("initiative.enabled", "0")
    store.set_preference("initiative.enabled", "1")
    assert store.get_preference("initiative.enabled") == "1"
    assert store.all_preferences() == {"initiative.enabled": "1"}


def test_working_memory_budget():
    wm = WorkingMemory(max_tokens=10)
    for i in range(20):
        wm.add(f"item {i} " + "x" * 40)
    assert wm.tokens <= 10
    assert len(wm) <= 2


def test_session_summary_and_consolidation(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    for i in range(4):
        store.add_episode("s1", "user", "We need a reliable evaluation suite for the project.")
        store.add_episode("s1", "assistant", f"Detail number {i} about implementation choices.")
    summary = store.summarize_session("s1")
    assert "evaluation" in summary
    promoted = store.consolidate(min_mentions=3)
    assert promoted
    assert any("evaluation" in f.content for f in promoted)


def test_projects(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.upsert_project("CORTEX-0", goal="build a small LM")
    store.upsert_project("CORTEX-0", goal="build a small LM with reasoning")
    projects = store.projects()
    assert len(projects) == 1
    assert "reasoning" in projects[0]["goal"]
