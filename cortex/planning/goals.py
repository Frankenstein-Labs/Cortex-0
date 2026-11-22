"""Persistent goals with progress tracking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List, Optional

from ..memory.store import MemoryStore


@dataclass
class Goal:
    id: int
    title: str
    description: str
    status: str
    priority: float
    progress: float
    meta: dict
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": round(self.priority, 3),
            "progress": round(self.progress, 3),
            "meta": self.meta,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _row_to_goal(row) -> Goal:
    return Goal(
        id=int(row["id"]),
        title=row["title"],
        description=row["description"],
        status=row["status"],
        priority=float(row["priority"]),
        progress=float(row["progress"]),
        meta=json.loads(row["meta"] or "{}"),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


class GoalStore:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def create(
        self, title: str, description: str = "", priority: float = 0.5, meta: Optional[dict] = None
    ) -> Goal:
        if not title.strip():
            raise ValueError("goal title cannot be empty")
        now = time.time()
        with self.store.connection() as conn:
            cursor = conn.execute(
                """INSERT INTO goals(title, description, status, priority, progress, meta, created_at, updated_at)
                   VALUES (?, ?, 'active', ?, 0.0, ?, ?, ?)""",
                (
                    title.strip(),
                    description,
                    max(0.0, min(1.0, priority)),
                    json.dumps(meta or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            goal_id = int(cursor.lastrowid)
        return self.get(goal_id)

    def get(self, goal_id: int) -> Optional[Goal]:
        with self.store.connection() as conn:
            row = conn.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
        return _row_to_goal(row) if row else None

    def list(self, status: Optional[str] = None, limit: int = 50) -> List[Goal]:
        sql = "SELECT * FROM goals"
        params: List[object] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY priority DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        with self.store.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_goal(row) for row in rows]

    def update(
        self,
        goal_id: int,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        priority: Optional[float] = None,
    ) -> Goal:
        goal = self.get(goal_id)
        if goal is None:
            raise KeyError(goal_id)
        status = status or goal.status
        progress = goal.progress if progress is None else max(0.0, min(1.0, progress))
        priority = goal.priority if priority is None else max(0.0, min(1.0, priority))
        with self.store.connection() as conn:
            conn.execute(
                "UPDATE goals SET status=?, progress=?, priority=?, updated_at=? WHERE id=?",
                (status, progress, priority, time.time(), goal_id),
            )
        return self.get(goal_id)

    def complete(self, goal_id: int) -> Goal:
        return self.update(goal_id, status="completed", progress=1.0)

    def stalled(self, older_than_seconds: float = 86400, now: Optional[float] = None) -> List[Goal]:
        now = now or time.time()
        return [
            goal
            for goal in self.list(status="active")
            if goal.progress < 1.0 and now - goal.updated_at >= older_than_seconds
        ]
