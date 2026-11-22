from __future__ import annotations

import time

from cortex.initiative.engine import InitiativeEngine
from cortex.initiative.policy import InitiativePolicy
from cortex.initiative.scheduler import InitiativeScheduler


def _ts(hour: int) -> float:
    return time.mktime((2026, 1, 1, hour, 0, 0, 0, 0, -1))


def test_quiet_hours():
    policy = InitiativePolicy(quiet_hours=(22, 7))
    assert policy.in_quiet_hours(_ts(23))
    assert policy.in_quiet_hours(_ts(3))
    assert not policy.in_quiet_hours(_ts(12))
    assert not InitiativePolicy(quiet_hours=None).in_quiet_hours(_ts(23))


def test_policy_rate_limits():
    policy = InitiativePolicy(
        quiet_hours=None, min_interval_seconds=60, daily_cap=2, max_pending=1
    )
    assert policy.check(1000, None, 0, 0, None) == (True, "allowed")
    assert policy.check(1000, 990, 0, 0, None) == (False, "min_interval")
    assert policy.check(1000, None, 2, 0, None) == (False, "daily_cap")
    assert policy.check(1000, None, 0, 1, None) == (False, "too_many_pending")
    assert policy.check(1000, None, 0, 0, 999)[0] is False


def _engine(store) -> InitiativeEngine:
    policy = InitiativePolicy(
        quiet_hours=None,
        min_interval_seconds=0,
        daily_cap=10,
        idle_min_seconds=0,
        min_score=0.3,
        max_pending=3,
    )
    return InitiativeEngine(store, policy=policy, seed=0)


def test_engine_proposes_then_deduplicates(store):
    store.add("Le projet CORTEX-0 développe un moteur de raisonnement.", kind="project")
    engine = _engine(store)
    now = time.time()
    first = engine.tick(now=now)
    assert len(first) == 1
    assert first[0].message
    assert first[0].status == "pending"
    second = engine.tick(now=now + 1)
    assert second == []


def test_daily_cap_and_dismiss(store):
    store.add("Une mémoire importante sur le raisonnement.", kind="project")
    engine = _engine(store)
    engine.policy.daily_cap = 1
    first = engine.tick(now=time.time())
    assert len(first) == 1
    engine.mark_sent(first[0].id)
    assert engine.tick(now=time.time() + 1) == []
    assert engine.recent(status="sent")[0].status == "sent"
    assert engine.dismiss(first[0].id).status == "dismissed"


def test_scheduler_respects_interval(store):
    store.add("Le projet CORTEX-0 explore la planification.", kind="project")
    engine = _engine(store)
    delivered = []
    scheduler = InitiativeScheduler(
        engine,
        store,
        interval_seconds=60,
        notifier=delivered.append,
        clock=lambda: 1000.0,
    )
    first = scheduler.run_once(now=1000.0)
    assert len(first) == 1
    assert len(delivered) == 1
    assert scheduler.run_once(now=1001.0) == []
    assert engine.recent(status="sent")


def test_disabled_policy_never_proposes(store):
    store.add("Une mémoire quelconque sur un sujet.", kind="project")
    engine = _engine(store)
    engine.policy.enabled = False
    assert engine.tick(now=time.time()) == []
