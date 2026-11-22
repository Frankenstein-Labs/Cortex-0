"""Durable SQLite task queue with visibility timeout, retries and workers."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..memory.store import MemoryStore

TASK_STATUSES = {"queued", "running", "done", "failed", "cancelled"}


@dataclass
class Task:
    id: int
    kind: str
    payload: dict
    status: str
    attempts: int
    max_attempts: int
    priority: int
    run_at: float
    locked_by: Optional[str]
    locked_at: Optional[float]
    error: Optional[str]
    result: Optional[str]
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "payload": self.payload,
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "priority": self.priority,
            "run_at": self.run_at,
            "locked_by": self.locked_by,
            "error": self.error,
            "result": json.loads(self.result) if self.result else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _row_to_task(row) -> Task:
    return Task(
        id=int(row["id"]),
        kind=row["kind"],
        payload=json.loads(row["payload"] or "{}"),
        status=row["status"],
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        priority=int(row["priority"]),
        run_at=float(row["run_at"]),
        locked_by=row["locked_by"],
        locked_at=row["locked_at"],
        error=row["error"],
        result=row["result"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


class TaskQueue:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def enqueue(
        self,
        kind: str,
        payload: Optional[dict] = None,
        run_at: Optional[float] = None,
        priority: int = 0,
        max_attempts: int = 3,
        idempotency_key: Optional[str] = None,
    ) -> Task:
        now = time.time()
        run_at = now if run_at is None else run_at
        with self.store.connection() as conn:
            cursor = conn.execute(
                """INSERT INTO tasks(kind, payload, status, attempts, max_attempts, priority,
                                     run_at, created_at, updated_at, idempotency_key)
                   VALUES (?, ?, 'queued', 0, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(idempotency_key) DO NOTHING""",
                (
                    kind,
                    json.dumps(payload or {}, ensure_ascii=False),
                    max_attempts,
                    priority,
                    run_at,
                    now,
                    now,
                    idempotency_key,
                ),
            )
            if cursor.rowcount == 0 and idempotency_key:
                row = conn.execute(
                    "SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                return _row_to_task(row)
            task_id = int(cursor.lastrowid)
        return self.get(task_id)

    def get(self, task_id: int) -> Optional[Task]:
        with self.store.connection() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None

    def dequeue(
        self,
        worker_id: str,
        now: Optional[float] = None,
        visibility_timeout: float = 300.0,
    ) -> Optional[Task]:
        now = now or time.time()
        stale_before = now - visibility_timeout
        with self.store.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """UPDATE tasks SET status='queued', locked_by=NULL, locked_at=NULL, updated_at=?
                   WHERE status='running' AND locked_at < ?""",
                (now, stale_before),
            )
            row = conn.execute(
                """SELECT * FROM tasks WHERE status='queued' AND run_at <= ?
                   ORDER BY priority DESC, id ASC LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """UPDATE tasks SET status='running', locked_by=?, locked_at=?,
                                    attempts=attempts+1, updated_at=?
                   WHERE id=? AND status='queued'""",
                (worker_id, now, now, row["id"]),
            )
            if cursor.rowcount == 0:
                return None
        return self.get(int(row["id"]))

    def ack(self, task_id: int, result: Optional[dict] = None) -> Task:
        with self.store.connection() as conn:
            conn.execute(
                """UPDATE tasks SET status='done', result=?, error=NULL, locked_by=NULL,
                                    locked_at=NULL, updated_at=? WHERE id=?""",
                (json.dumps(result or {}, ensure_ascii=False), time.time(), task_id),
            )
        return self.get(task_id)

    def fail(self, task_id: int, error: str, retry_delay: float = 5.0) -> Task:
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        now = time.time()
        if task.attempts >= task.max_attempts:
            status, run_at = "failed", now
        else:
            status = "queued"
            run_at = now + retry_delay * (2 ** max(0, task.attempts - 1))
        with self.store.connection() as conn:
            conn.execute(
                """UPDATE tasks SET status=?, error=?, run_at=?, locked_by=NULL,
                                    locked_at=NULL, updated_at=? WHERE id=?""",
                (status, error[:2000], run_at, now, task_id),
            )
        return self.get(task_id)

    def cancel(self, task_id: int) -> Task:
        with self.store.connection() as conn:
            conn.execute(
                "UPDATE tasks SET status='cancelled', updated_at=? WHERE id=?",
                (time.time(), task_id),
            )
        return self.get(task_id)

    def list(self, status: Optional[str] = None, limit: int = 50) -> List[Task]:
        sql = "SELECT * FROM tasks"
        params: List[object] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self.store.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_task(row) for row in rows]

    def stats(self) -> Dict[str, int]:
        with self.store.connection() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS c FROM tasks GROUP BY status").fetchall()
        counts = {status: 0 for status in TASK_STATUSES}
        for row in rows:
            counts[row["status"]] = int(row["c"])
        counts["total"] = sum(counts.values())
        return counts


class Worker:
    """Sequential worker; can be embedded in a thread or run in-process."""

    def __init__(
        self,
        queue: TaskQueue,
        handlers: Dict[str, Callable[[dict], dict]],
        worker_id: Optional[str] = None,
        poll_interval: float = 1.0,
    ) -> None:
        self.queue = queue
        self.handlers = handlers
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self._stop = threading.Event()

    def run_once(self, now: Optional[float] = None) -> Optional[dict]:
        task = self.queue.dequeue(self.worker_id, now=now)
        if task is None:
            return None
        handler = self.handlers.get(task.kind)
        if handler is None:
            self.queue.fail(task.id, f"no handler for kind {task.kind!r}")
            return {"task": task.id, "status": "failed", "error": "missing handler"}
        try:
            result = handler(task.payload)
            self.queue.ack(task.id, result)
            return {"task": task.id, "status": "done", "result": result}
        except Exception as exc:  # a handler failure must not kill the worker
            updated = self.queue.fail(task.id, f"{type(exc).__name__}: {exc}")
            return {"task": task.id, "status": updated.status, "error": str(exc)}

    def run_forever(
        self,
        stop_event: Optional[threading.Event] = None,
        max_iterations: Optional[int] = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> int:
        stop_event = stop_event or self._stop
        processed = 0
        iterations = 0
        while not stop_event.is_set():
            result = self.run_once()
            if result is not None:
                processed += 1
            else:
                sleeper(self.poll_interval)
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
        return processed

    def stop(self) -> None:
        self._stop.set()
