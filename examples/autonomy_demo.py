#!/usr/bin/env python3
"""Show the initiative engine proposing a bounded, deduplicated idea."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.initiative.engine import InitiativeEngine
from cortex.initiative.policy import InitiativePolicy
from cortex.memory.store import MemoryStore


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        store = MemoryStore(Path(directory) / "autonomy.sqlite")
        store.add(
            "Le projet CORTEX-0 doit améliorer le raisonnement mathématique.",
            kind="project",
            importance=0.9,
        )
        store.add(
            "Décision : utiliser un vérificateur programmatique plutôt qu'un second LLM.",
            kind="project",
            importance=0.85,
        )
        policy = InitiativePolicy(
            quiet_hours=None,
            min_interval_seconds=0,
            daily_cap=3,
            idle_min_seconds=0,
            min_score=0.3,
            max_pending=3,
        )
        engine = InitiativeEngine(store, policy=policy, seed=0)
        first = engine.tick()
        second = engine.tick()
        print(f"Propositions au premier tick : {len(first)}")
        for initiative in first:
            print(f"  [{initiative.kind}] score={initiative.score:.2f}")
            print(f"  {initiative.message}")
        print(f"Propositions au tick suivant : {len(second)} (sources différentes, pas de doublon exact)")
        if first:
            engine.mark_sent(first[0].id)
            print(f"Statut après envoi : {engine.recent(status='sent')[0].status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
