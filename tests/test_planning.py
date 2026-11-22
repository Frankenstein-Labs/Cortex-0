from __future__ import annotations

import time

import pytest

from cortex.planning.goals import GoalStore
from cortex.planning.missions import MissionStore, default_plan


def test_goal_lifecycle(store):
    goals = GoalStore(store)
    goal = goals.create("Construire CORTEX-0", "Une petite LLM autonome", priority=0.8)
    assert goal.status == "active"
    assert goals.get(goal.id).title == "Construire CORTEX-0"
    updated = goals.update(goal.id, progress=0.4, priority=0.9)
    assert updated.progress == pytest.approx(0.4)
    assert updated.priority == pytest.approx(0.9)
    assert [item.id for item in goals.list(status="active")] == [goal.id]
    completed = goals.complete(goal.id)
    assert completed.status == "completed"
    assert goals.list(status="active") == []


def test_stalled_goals(store):
    goals = GoalStore(store)
    goal = goals.create("Objectif ancien", priority=0.5)
    stalled = goals.stalled(older_than_seconds=0, now=time.time() + 10)
    assert [item.id for item in stalled] == [goal.id]
    assert goals.stalled(older_than_seconds=10**9, now=time.time() + 10) == []


def test_default_plan_has_five_steps():
    plan = default_plan("Ajouter l'authentification")
    assert len(plan) == 5
    assert [step["id"] for step in plan] == [1, 2, 3, 4, 5]
    assert all(step["status"] == "pending" for step in plan)


def test_mission_progress_and_resume(store):
    missions = MissionStore(store)
    mission = missions.create("Auth", "Ajouter l'authentification")
    first = mission.next_step()
    assert first.id == 1
    mission = missions.update_step(mission.id, 1, "done", result='{"ok": true}')
    assert mission.next_step().id == 2
    missions.update_step(mission.id, 2, "in_progress")
    resumed = missions.resume(mission.id)
    assert resumed["mission"]["id"] == mission.id
    assert resumed["next_step"]["id"] == 2
    assert resumed["last_checkpoint"] is None


def test_new_instruction_preserves_completed_work(store):
    missions = MissionStore(store)
    mission = missions.create("Auth", "Ajouter l'authentification")
    missions.update_step(mission.id, 1, "done", result="inspection terminée")
    missions.update_step(mission.id, 2, "in_progress")
    updated = missions.add_instruction(mission.id, "Utiliser OAuth")
    assert updated.instructions[-1]["text"] == "Utiliser OAuth"
    assert updated.plan[0]["status"] == "done"
    assert updated.plan[1]["status"] == "pending"  # in-progress work is re-queued, not lost
    assert updated.plan[-1]["title"].startswith("Intégrer la nouvelle instruction")
    assert any(checkpoint["label"] == "instruction-added" for checkpoint in updated.checkpoints)


def test_mission_status_validation(store):
    missions = MissionStore(store)
    mission = missions.create("Test", "Faire quelque chose")
    with pytest.raises(ValueError):
        missions.set_status(mission.id, "invented")
    assert missions.set_status(mission.id, "completed").status == "completed"


def test_mission_checkpoint_limit(store):
    missions = MissionStore(store)
    mission = missions.create("Test", "Faire quelque chose")
    for index in range(30):
        missions.checkpoint(mission.id, f"step-{index}", {"index": index})
    assert len(missions.get(mission.id).checkpoints) == 20
