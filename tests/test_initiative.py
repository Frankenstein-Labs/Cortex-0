from __future__ import annotations

from datetime import datetime

from cortex.initiative.engine import (
    InitiativeConfig,
    InitiativeEngine,
    RecordingNotifier,
    WebhookNotifier,
)
from cortex.memory.store import MemoryStore
from cortex.planning.tasks import TaskStore


def _noon() -> float:
    return datetime(2026, 1, 15, 12, 0, 0).timestamp()


def _engine(tmp_path, **cfg):
    store = MemoryStore(tmp_path / "m.db")
    store.upsert_project("CORTEX-0", goal="build a reasoning engine with tests")
    tasks = TaskStore(tmp_path / "t.db", checkpoint_dir=tmp_path / "ckpt")
    tasks.create_task("Reasoning engine", goal="implement arithmetic", steps=["write tests", "implement verifier"])
    notifier = RecordingNotifier()
    params = dict(min_interval_sec=0, min_score=0.0, min_novelty=0.0, quiet_hours_start=0, quiet_hours_end=0)
    params.update(cfg)
    config = InitiativeConfig(**params)
    engine = InitiativeEngine(store, tasks=tasks, config=config, notifier=notifier)
    return engine, store, notifier


def test_disabled_engine_is_silent(tmp_path):
    engine, _, notifier = _engine(tmp_path, enabled=False)
    decision = engine.tick(now=_noon())
    assert decision.message is None
    assert decision.reason == "initiative disabled"
    assert notifier.messages == []


def test_quiet_hours_suppress(tmp_path):
    engine, _, notifier = _engine(tmp_path)
    engine.config.quiet_hours_start = 0
    engine.config.quiet_hours_end = 23
    decision = engine.tick(now=_noon())
    assert decision.message is None
    assert decision.reason == "quiet hours"
    assert notifier.messages == []


def test_delivers_once_then_cooldown(tmp_path):
    engine, store, notifier = _engine(tmp_path, min_interval_sec=900)
    first = engine.tick(now=_noon())
    assert first.message is not None, first.reason
    assert first.message.delivered
    assert len(notifier.messages) == 1
    assert store.recent_initiatives(limit=5)
    second = engine.tick(now=_noon() + 100)
    assert second.message is None
    assert second.reason == "cooldown"


def test_daily_cap(tmp_path):
    engine, _, notifier = _engine(tmp_path, max_per_day=1)
    assert engine.tick(now=_noon()).message is not None
    assert engine.tick(now=_noon() + 3600).message is None
    assert engine.tick(now=_noon() + 3600).reason == "daily cap reached"
    assert len(notifier.messages) == 1


def test_novelty_gate(tmp_path):
    engine, store, notifier = _engine(tmp_path, min_novelty=0.99)
    candidates = engine._candidates("default")
    assert candidates
    store.add_initiative(candidates[0].kind, candidates[0].text, 1.0, delivered=True, ts=_noon() - 10)
    decision = engine.tick(now=_noon())
    assert decision.message is None
    assert "novel" in decision.reason


def test_no_candidates(tmp_path):
    store = MemoryStore(tmp_path / "empty.db")
    engine = InitiativeEngine(
        store,
        config=InitiativeConfig(min_interval_sec=0, quiet_hours_start=0, quiet_hours_end=0),
        notifier=RecordingNotifier(),
    )
    decision = engine.tick(now=_noon())
    assert decision.message is None
    assert decision.reason == "no candidate"


def test_webhook_validation():
    import pytest

    with pytest.raises(ValueError):
        WebhookNotifier("ftp://example.com")


def test_history_persisted(tmp_path):
    engine, _, _ = _engine(tmp_path)
    engine.tick(now=_noon())
    history = engine.history(limit=5)
    assert len(history) == 1
    assert history[0]["content"]
