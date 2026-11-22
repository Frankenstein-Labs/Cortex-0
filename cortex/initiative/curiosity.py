"""Curiosity: turn remembered material into concrete open questions."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..memory.retrieval import dedupe_texts
from ..memory.store import MemoryStore


class CuriosityEngine:
    def __init__(self, store: MemoryStore, max_items: int = 20) -> None:
        self.store = store
        self.max_items = max_items

    def questions(self, limit: int = 5, now: Optional[float] = None) -> List[Dict[str, object]]:
        memories = self.store.recent(kind="semantic", limit=self.max_items) + self.store.recent(
            kind="project", limit=self.max_items
        )
        statements = dedupe_texts([memory.summary or memory.content for memory in memories])
        questions = []
        for statement in statements[:limit]:
            statement = statement.strip().rstrip(".")
            if not statement:
                continue
            questions.append(
                {
                    "kind": "curiosity",
                    "message": f"Question : {statement}. On peut explorer ce point ?",
                    "importance": 0.6,
                    "source": "semantic_memory",
                }
            )
        return questions

    def unanswered_questions(self, session_id: str, limit: int = 5) -> List[Dict[str, object]]:
        events = self.store.recent_events(session_id, limit=40)
        questions = []
        for index, event in enumerate(events):
            if event["role"] != "user":
                continue
            if not event["content"].strip().endswith("?"):
                continue
            answered = any(
                later["role"] == "assistant" for later in events[index + 1 :]
            )
            if not answered:
                questions.append(
                    {
                        "kind": "open_question",
                        "message": f"Question : {event['content'].strip()}",
                        "importance": 0.7,
                        "source": "conversation",
                    }
                )
        return questions[-limit:]
