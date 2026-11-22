"""Combinatorial idea generation over remembered projects and goals.

This is honest ideation: it recombines known project elements with explicit
templates.  The LLM can elaborate a proposal later; the generator never
claims that an idea is novel or validated.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..memory.retrieval import dedupe_texts
from ..memory.store import MemoryStore
from ..planning.goals import GoalStore


class IdeaGenerator:
    def __init__(self, store: MemoryStore, seed: int = 0) -> None:
        self.store = store
        self.goals = GoalStore(store)
        self.rng = random.Random(seed)

    def ideas(self, limit: int = 5, now: Optional[float] = None) -> List[Dict[str, object]]:
        memories = self.store.recent(kind="project", limit=15)
        statements = dedupe_texts(
            [memory.summary or memory.content for memory in memories if memory.content]
        )
        goal_titles = [goal.title for goal in self.goals.list(status="active", limit=10)]
        pool = statements + goal_titles
        if len(pool) < 2:
            return []
        ideas: List[Dict[str, object]] = []
        seen = set()
        for _ in range(limit * 3):
            if len(ideas) >= limit:
                break
            left, right = self.rng.sample(pool, 2)
            left, right = left.strip().rstrip("."), right.strip().rstrip(".")
            key = tuple(sorted((left, right)))
            if key in seen:
                continue
            seen.add(key)
            ideas.append(
                {
                    "kind": "idea",
                    "message": f"Proposition : explorer « {left[:80]} » avec « {right[:80]} ».",
                    "importance": 0.55,
                    "source": "combinatorial",
                }
            )
        return ideas
