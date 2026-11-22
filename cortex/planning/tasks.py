"""Persistent task engine with resumable plans and checkpoints.

Designed for long-running missions: a task, its plan and its progress live in
SQLite, so a new process can resume where the previous one stopped. New user
instructions are appended to the existing plan instead of restarting the work.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_TASK_STATUS = {"open", "in_progress", "done", "blocked", "cancelled"}
VALID_STEP_STATUS = {"pending", "in_progress", "done", "failed", "skipped"}


@dataclass
class Step:
    id: int
    task_id: int
    position: int
    description: str
    status: str = "pending"
    result: str = ""
    updated_at: float = 0.0

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "position": self.position,
            "description": self.description,
            "status": self.status,
            "result": self.result,
            "updated_at": self.updated_at,
        }


@dataclass
class Task:
    id: int
    title: str
    goal: str
    status: str
    revision: int
    created_at: float
    updated_at: float
    context: dict = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "goal": self.goal,
            "status": self.status,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "context": self.context,
            "steps": [s.as_dict() for s in self.steps],
        }

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.status in ("done", "skipped")) / len(self.steps)

    def next_step(self) -> Step | None:
        return next((s for s in self.steps if s.status in ("pending", "in_progress")), None)


class TaskStore:
    def __init__(self, path: str | Path = "cortex_tasks.db", checkpoint_dir: str | Path = "artifacts/tasks") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                goal TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                revision INTEGER NOT NULL DEFAULT 0,
                context TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_steps_task ON steps(task_id, position);
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                label TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ creation
    def create_task(self, title: str, goal: str = "", steps: list[str] | None = None, context: dict | None = None) -> Task:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO tasks(title, goal, status, revision, context, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (title, goal, "open", 0, json.dumps(context or {}), now, now),
        )
        task_id = int(cur.lastrowid)
        for i, description in enumerate(steps or []):
            self._conn.execute(
                "INSERT INTO steps(task_id, position, description, status, updated_at) VALUES (?,?,?,?,?)",
                (task_id, i, description, "pending", now),
            )
        self._conn.commit()
        return self.get_task(task_id)

    # ------------------------------------------------------------------- reading
    def _load_steps(self, task_id: int) -> list[Step]:
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE task_id=? ORDER BY position", (task_id,)
        ).fetchall()
        return [Step(**dict(r)) for r in rows]

    def get_task(self, task_id: int) -> Task:
        row = self._conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"task {task_id} not found")
        data = dict(row)
        data["context"] = json.loads(data["context"])
        return Task(**data, steps=self._load_steps(task_id))

    def list_tasks(self, status: str | None = None, limit: int = 100) -> list[Task]:
        if status:
            rows = self._conn.execute(
                "SELECT id FROM tasks WHERE status=? ORDER BY updated_at DESC LIMIT ?", (status, limit)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT id FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.get_task(r["id"]) for r in rows]

    def open_tasks(self) -> list[Task]:
        return [t for t in self.list_tasks() if t.status in ("open", "in_progress", "blocked")]

    # ------------------------------------------------------------------ mutation
    def _touch(self, task_id: int, status: str | None = None) -> None:
        if status is not None:
            self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE id=?", (status, time.time(), task_id)
            )
        else:
            self._conn.execute("UPDATE tasks SET updated_at=? WHERE id=?", (time.time(), task_id))
        self._conn.commit()

    def set_status(self, task_id: int, status: str) -> Task:
        if status not in VALID_TASK_STATUS:
            raise ValueError(f"invalid status {status!r}")
        self._touch(task_id, status)
        return self.get_task(task_id)

    def add_instruction(self, task_id: int, instruction: str, steps: list[str] | None = None) -> Task:
        """Append a new instruction to a running mission without losing progress."""
        task = self.get_task(task_id)
        position = max((s.position for s in task.steps), default=-1) + 1
        now = time.time()
        new_steps = steps or [f"Apply new instruction: {instruction}"]
        for i, description in enumerate(new_steps):
            self._conn.execute(
                "INSERT INTO steps(task_id, position, description, status, updated_at) VALUES (?,?,?,?,?)",
                (task_id, position + i, description, "pending", now),
            )
        self._conn.execute(
            "UPDATE tasks SET revision=revision+1, status=?, updated_at=? WHERE id=?",
            ("in_progress" if task.status == "done" else task.status, now, task_id),
        )
        self._conn.commit()
        self.checkpoint(task_id, f"instruction-r{task.revision + 1}", {"instruction": instruction})
        return self.get_task(task_id)

    def start_next_step(self, task_id: int) -> Step | None:
        task = self.get_task(task_id)
        step = task.next_step()
        if step is None:
            if all(s.status in ("done", "skipped") for s in task.steps) and task.steps:
                self.set_status(task_id, "done")
            return None
        self._conn.execute(
            "UPDATE steps SET status=?, updated_at=? WHERE id=?", ("in_progress", time.time(), step.id)
        )
        self._conn.execute("UPDATE tasks SET status=?, updated_at=? WHERE id=?", ("in_progress", time.time(), task_id))
        self._conn.commit()
        return next(s for s in self._load_steps(task_id) if s.id == step.id)

    def complete_step(self, task_id: int, step_id: int, result: str = "", status: str = "done") -> Task:
        if status not in VALID_STEP_STATUS:
            raise ValueError(f"invalid step status {status!r}")
        self._conn.execute(
            "UPDATE steps SET status=?, result=?, updated_at=? WHERE id=? AND task_id=?",
            (status, result, time.time(), step_id, task_id),
        )
        self._conn.commit()
        task = self.get_task(task_id)
        if task.steps and all(s.status in ("done", "skipped") for s in task.steps):
            self.set_status(task_id, "done")
        else:
            self._touch(task_id)
        return self.get_task(task_id)

    # --------------------------------------------------------------- checkpoints
    def checkpoint(self, task_id: int, label: str, data: dict[str, Any]) -> Path:
        path = self.checkpoint_dir / f"task-{task_id}" / f"{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"task_id": task_id, "label": label, "created_at": time.time(), "data": data}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._conn.execute(
            "INSERT INTO checkpoints(task_id, label, path, created_at) VALUES (?,?,?,?)",
            (task_id, label, str(path), time.time()),
        )
        self._conn.commit()
        return path

    def list_checkpoints(self, task_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM checkpoints WHERE task_id=? ORDER BY created_at DESC", (task_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def resume(self, task_id: int) -> dict:
        """Return everything a worker needs to continue a mission."""
        task = self.get_task(task_id)
        next_step = task.next_step()
        checkpoints = self.list_checkpoints(task_id)
        latest_data = None
        if checkpoints:
            latest_data = json.loads(Path(checkpoints[0]["path"]).read_text(encoding="utf-8"))
        return {
            "task": task.as_dict(),
            "next_step": next_step.as_dict() if next_step else None,
            "progress": round(task.progress, 4),
            "revision": task.revision,
            "latest_checkpoint": latest_data,
        }

    def delete_task(self, task_id: int) -> None:
        self._conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
