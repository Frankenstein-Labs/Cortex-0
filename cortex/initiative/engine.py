"""Autonomous initiative engine.

It proposes *one* high-value message at a time, only when policy allows it and
only if the proposal is sufficiently novel.  Every proposal is persisted with
its score and source so that proactivity is measurable and auditable.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..memory.retrieval import cosine, jaccard
from ..memory.store import MemoryStore
from ..planning.goals import GoalStore
from .curiosity import CuriosityEngine
from .ideas import IdeaGenerator
from .policy import InitiativePolicy
from .self_evaluation import SelfEvaluator


@dataclass
class Initiative:
    id: int
    session_id: Optional[str]
    kind: str
    message: str
    score: float
    status: str
    meta: dict
    created_at: float
    sent_at: Optional[float]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "kind": self.kind,
            "message": self.message,
            "score": round(self.score, 4),
            "status": self.status,
            "meta": self.meta,
            "created_at": self.created_at,
            "sent_at": self.sent_at,
        }


def _row_to_initiative(row) -> Initiative:
    return Initiative(
        id=int(row["id"]),
        session_id=row["session_id"],
        kind=row["kind"],
        message=row["message"],
        score=float(row["score"]),
        status=row["status"],
        meta=json.loads(row["meta"] or "{}"),
        created_at=float(row["created_at"]),
        sent_at=row["sent_at"],
    )


class InitiativeEngine:
    def __init__(
        self,
        store: MemoryStore,
        policy: Optional[InitiativePolicy] = None,
        seed: int = 0,
    ) -> None:
        self.store = store
        self.policy = policy or InitiativePolicy()
        self.goals = GoalStore(store)
        self.curiosity = CuriosityEngine(store)
        self.ideas = IdeaGenerator(store, seed=seed)
        self.self_evaluator = SelfEvaluator(store)

    # ------------------------------------------------------------------ public
    def tick(
        self,
        session_id: Optional[str] = None,
        now: Optional[float] = None,
        user_last_active: Optional[float] = None,
    ) -> List[Initiative]:
        now = now or time.time()
        last_sent, sent_today, pending = self._stats(now)
        allowed, reason = self.policy.check(
            now, last_sent, sent_today, pending, user_last_active
        )
        if not allowed:
            return []

        candidates = self._candidates(session_id, now)
        if not candidates:
            return []

        recent = self.recent(limit=30)
        recent_messages = [item.message for item in recent]
        selected = None
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            similarity = self._max_similarity(candidate["message"], recent_messages)
            if similarity >= self.policy.similarity_threshold:
                continue
            score = candidate["score"] * (0.6 + 0.4 * (1.0 - similarity))
            if score < self.policy.min_score:
                continue
            selected = {**candidate, "score": min(1.0, score)}
            break

        if selected is None:
            return []
        initiative_id = self._insert(session_id, selected, now)
        return [self.get(initiative_id)]

    def get(self, initiative_id: int) -> Optional[Initiative]:
        with self.store.connection() as conn:
            row = conn.execute("SELECT * FROM initiatives WHERE id=?", (initiative_id,)).fetchone()
        return _row_to_initiative(row) if row else None

    def pending(self, session_id: Optional[str] = None) -> List[Initiative]:
        return self._query(status="pending", session_id=session_id)

    def recent(self, limit: int = 30, status: Optional[str] = None) -> List[Initiative]:
        return self._query(limit=limit, status=status)

    def mark_sent(self, initiative_id: int, now: Optional[float] = None) -> Initiative:
        now = now or time.time()
        with self.store.connection() as conn:
            conn.execute(
                "UPDATE initiatives SET status='sent', sent_at=? WHERE id=?",
                (now, initiative_id),
            )
        return self.get(initiative_id)

    def dismiss(self, initiative_id: int) -> Initiative:
        with self.store.connection() as conn:
            conn.execute("UPDATE initiatives SET status='dismissed' WHERE id=?", (initiative_id,))
        return self.get(initiative_id)

    # --------------------------------------------------------------- candidate
    def _candidates(self, session_id: Optional[str], now: float) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        for goal in self.goals.stalled(
            older_than_seconds=self.policy.stalled_goal_seconds, now=now
        ):
            staleness = min(1.0, (now - goal.updated_at) / (7 * 86400))
            candidates.append(
                {
                    "kind": "goal_followup",
                    "message": (
                        f"Objectif : {goal.title}. Progression : {goal.progress:.0%}. "
                        "On continue ?"
                    ),
                    "score": 0.55 + 0.25 * goal.priority + 0.1 * staleness,
                    "meta": {"goal_id": goal.id},
                }
            )

        for question in self.curiosity.questions(limit=4, now=now):
            candidates.append(
                {
                    "kind": question["kind"],
                    "message": question["message"],
                    "score": 0.48 + 0.2 * float(question["importance"]),
                    "meta": {"source": question["source"]},
                }
            )

        if session_id:
            for question in self.curiosity.unanswered_questions(session_id, limit=3):
                candidates.append(
                    {
                        "kind": question["kind"],
                        "message": question["message"],
                        "score": 0.6,
                        "meta": {"source": question["source"]},
                    }
                )

        for idea in self.ideas.ideas(limit=3, now=now):
            candidates.append(
                {
                    "kind": idea["kind"],
                    "message": idea["message"],
                    "score": 0.46 + 0.15 * float(idea["importance"]),
                    "meta": {"source": idea["source"]},
                }
            )

        evaluation = self.self_evaluator.initiative(now=now)
        if evaluation:
            candidates.append(
                {
                    "kind": evaluation["kind"],
                    "message": evaluation["message"],
                    "score": float(evaluation["importance"]) + 0.1,
                    "meta": {"source": evaluation["source"]},
                }
            )
        return candidates

    # ------------------------------------------------------------------ stats
    def _stats(self, now: float) -> tuple[Optional[float], int, int]:
        local_now = _dt.datetime.fromtimestamp(now)
        midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        midnight_ts = midnight.timestamp()
        with self.store.connection() as conn:
            last_sent = conn.execute(
                "SELECT MAX(sent_at) AS value FROM initiatives WHERE status='sent'"
            ).fetchone()["value"]
            sent_today = conn.execute(
                "SELECT COUNT(*) AS c FROM initiatives WHERE status='sent' AND sent_at>=?",
                (midnight_ts,),
            ).fetchone()["c"]
            pending = conn.execute(
                "SELECT COUNT(*) AS c FROM initiatives WHERE status='pending'"
            ).fetchone()["c"]
        return last_sent, int(sent_today), int(pending)

    def _max_similarity(self, message: str, recent_messages: List[str]) -> float:
        if not recent_messages:
            return 0.0
        embedding = self.store.embedder.embed(message)
        best = 0.0
        for previous in recent_messages:
            lexical = jaccard(message, previous)
            semantic = max(0.0, cosine(embedding, self.store.embedder.embed(previous)))
            best = max(best, lexical, semantic)
        return best

    def _insert(self, session_id: Optional[str], candidate: Dict[str, object], now: float) -> int:
        with self.store.connection() as conn:
            cursor = conn.execute(
                """INSERT INTO initiatives(session_id, kind, message, score, status, meta, created_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    session_id,
                    candidate["kind"],
                    candidate["message"],
                    float(candidate["score"]),
                    json.dumps(candidate.get("meta", {}), ensure_ascii=False),
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def _query(
        self,
        status: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Initiative]:
        sql = "SELECT * FROM initiatives WHERE 1=1"
        params: List[object] = []
        if status:
            sql += " AND status=?"
            params.append(status)
        if session_id:
            sql += " AND session_id=?"
            params.append(session_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.store.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_initiative(row) for row in rows]
