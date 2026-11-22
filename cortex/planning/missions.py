"""Persistent missions: plan, step state, checkpoints, resumption.

A mission survives restarts and new user instructions.  Completed steps are
never replayed when a new instruction arrives; only pending work is revised.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..memory.store import MemoryStore

STEP_STATUSES = {"pending", "in_progress", "done", "failed", "skipped"}
MISSION_STATUSES = {"active", "paused", "completed", "cancelled", "blocked"}


@dataclass
class MissionStep:
    id: int
    title: str
    status: str = "pending"
    tool: Optional[str] = None
    result: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "tool": self.tool,
            "result": self.result,
            "notes": self.notes,
        }


@dataclass
class Mission:
    id: str
    title: str
    instruction: str
    status: str
    plan: List[dict]
    instructions: List[dict]
    checkpoints: List[dict]
    created_at: float
    updated_at: float

    @property
    def steps(self) -> List[MissionStep]:
        return [MissionStep(**step) for step in self.plan]

    def next_step(self) -> Optional[MissionStep]:
        for step in self.steps:
            if step.status in {"pending", "in_progress", "failed"}:
                return step
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "instruction": self.instruction,
            "status": self.status,
            "plan": self.plan,
            "instructions": self.instructions,
            "checkpoints": self.checkpoints[-5:],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _row_to_mission(row) -> Mission:
    return Mission(
        id=row["id"],
        title=row["title"],
        instruction=row["instruction"],
        status=row["status"],
        plan=json.loads(row["plan"] or "[]"),
        instructions=json.loads(row["instructions"] or "[]"),
        checkpoints=json.loads(row["checkpoints"] or "[]"),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def default_plan(instruction: str) -> List[dict]:
    """Conservative template plan used until a stronger planner is trained."""
    steps = [
        ("Inspecter le dépôt et l'état actuel", "workspace"),
        ("Identifier les fichiers et dépendances concernés", "workspace"),
        (f"Implémenter : {instruction[:120]}", "edit"),
        ("Exécuter les tests et corriger les erreurs", "test"),
        ("Préparer un diff et un message de commit", "git"),
    ]
    return [
        {"id": index + 1, "title": title, "status": "pending", "tool": tool, "result": None, "notes": ""}
        for index, (title, tool) in enumerate(steps)
    ]


class MissionStore:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def create(
        self,
        title: str,
        instruction: str,
        plan: Optional[List[dict]] = None,
        mission_id: Optional[str] = None,
    ) -> Mission:
        mission_id = mission_id or uuid.uuid4().hex[:12]
        now = time.time()
        plan = plan or default_plan(instruction)
        with self.store.connection() as conn:
            conn.execute(
                """INSERT INTO missions(id, title, instruction, status, plan, instructions, checkpoints, created_at, updated_at)
                   VALUES (?, ?, ?, 'active', ?, ?, '[]', ?, ?)""",
                (
                    mission_id,
                    title,
                    instruction,
                    json.dumps(plan, ensure_ascii=False),
                    json.dumps([{"text": instruction, "at": now}], ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return self.get(mission_id)

    def get(self, mission_id: str) -> Optional[Mission]:
        with self.store.connection() as conn:
            row = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        return _row_to_mission(row) if row else None

    def list(self, status: Optional[str] = None, limit: int = 50) -> List[Mission]:
        sql = "SELECT * FROM missions"
        params: List[object] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self.store.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_mission(row) for row in rows]

    def update_step(
        self,
        mission_id: str,
        step_id: int,
        status: str,
        result: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Mission:
        if status not in STEP_STATUSES:
            raise ValueError(f"invalid step status: {status}")
        mission = self.get(mission_id)
        if mission is None:
            raise KeyError(mission_id)
        found = False
        for step in mission.plan:
            if int(step["id"]) == int(step_id):
                step["status"] = status
                if result is not None:
                    step["result"] = result
                if notes is not None:
                    step["notes"] = notes
                found = True
                break
        if not found:
            raise KeyError(f"step {step_id} not in mission {mission_id}")
        self._write_plan(mission_id, mission.plan, status=mission.status)
        return self.get(mission_id)

    def update_plan(self, mission_id: str, plan: List[dict]) -> Mission:
        self._write_plan(mission_id, plan)
        return self.get(mission_id)

    def add_instruction(self, mission_id: str, text: str) -> Mission:
        """Append an instruction and revise pending work without replaying done steps."""
        mission = self.get(mission_id)
        if mission is None:
            raise KeyError(mission_id)
        now = time.time()
        instructions = mission.instructions + [{"text": text, "at": now}]
        plan = mission.plan
        for step in plan:
            if step["status"] == "in_progress":
                step["status"] = "pending"
        next_id = max([int(step["id"]) for step in plan], default=0) + 1
        plan.append(
            {
                "id": next_id,
                "title": f"Intégrer la nouvelle instruction : {text[:100]}",
                "status": "pending",
                "tool": "edit",
                "result": None,
                "notes": "ajoutée après reprise de mission",
            }
        )
        with self.store.connection() as conn:
            conn.execute(
                "UPDATE missions SET instructions=?, plan=?, updated_at=? WHERE id=?",
                (
                    json.dumps(instructions, ensure_ascii=False),
                    json.dumps(plan, ensure_ascii=False),
                    now,
                    mission_id,
                ),
            )
        self.checkpoint(mission_id, "instruction-added", {"instruction": text})
        return self.get(mission_id)

    def checkpoint(self, mission_id: str, label: str, state: Optional[dict] = None) -> Mission:
        mission = self.get(mission_id)
        if mission is None:
            raise KeyError(mission_id)
        checkpoints = mission.checkpoints + [
            {"label": label, "at": time.time(), "state": state or {}}
        ]
        with self.store.connection() as conn:
            conn.execute(
                "UPDATE missions SET checkpoints=?, updated_at=? WHERE id=?",
                (json.dumps(checkpoints[-20:], ensure_ascii=False), time.time(), mission_id),
            )
        return self.get(mission_id)

    def set_status(self, mission_id: str, status: str) -> Mission:
        if status not in MISSION_STATUSES:
            raise ValueError(f"invalid mission status: {status}")
        with self.store.connection() as conn:
            conn.execute(
                "UPDATE missions SET status=?, updated_at=? WHERE id=?",
                (status, time.time(), mission_id),
            )
        return self.get(mission_id)

    def resume(self, mission_id: str) -> Dict[str, object]:
        mission = self.get(mission_id)
        if mission is None:
            raise KeyError(mission_id)
        return {
            "mission": mission.to_dict(),
            "next_step": mission.next_step().to_dict() if mission.next_step() else None,
            "last_checkpoint": mission.checkpoints[-1] if mission.checkpoints else None,
        }

    def _write_plan(self, mission_id: str, plan: List[dict], status: Optional[str] = None) -> None:
        mission = self.get(mission_id)
        if mission is None:
            raise KeyError(mission_id)
        with self.store.connection() as conn:
            conn.execute(
                "UPDATE missions SET plan=?, status=?, updated_at=? WHERE id=?",
                (
                    json.dumps(plan, ensure_ascii=False),
                    status or mission.status,
                    time.time(),
                    mission_id,
                ),
            )
