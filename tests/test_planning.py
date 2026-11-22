from __future__ import annotations

import pytest

from cortex.planning.tasks import TaskStore


def test_create_and_progress(tmp_path):
    store = TaskStore(tmp_path / "tasks.db", checkpoint_dir=tmp_path / "ckpt")
    task = store.create_task("Build auth", goal="secure login", steps=["read repo", "write code", "run tests"])
    assert task.status == "open"
    assert len(task.steps) == 3
    step = store.start_next_step(task.id)
    assert step.description == "read repo" and step.status == "in_progress"
    task = store.complete_step(task.id, step.id, "done")
    assert task.progress == pytest.approx(1 / 3)
    assert task.next_step().description == "write code"
    task = store.complete_step(task.id, task.next_step().id)
    task = store.complete_step(task.id, task.next_step().id)
    assert task.status == "done"


def test_add_instruction_preserves_progress(tmp_path):
    store = TaskStore(tmp_path / "tasks.db", checkpoint_dir=tmp_path / "ckpt")
    task = store.create_task("Build auth", steps=["read repo", "write code"])
    step = store.start_next_step(task.id)
    store.complete_step(task.id, step.id, "read")
    task = store.add_instruction(task.id, "also add logout")
    assert task.revision == 1
    assert len(task.steps) == 3
    assert task.steps[0].status == "done"
    assert task.steps[2].status == "pending"


def test_resume_across_processes(tmp_path):
    db = tmp_path / "tasks.db"
    store = TaskStore(db, checkpoint_dir=tmp_path / "ckpt")
    task = store.create_task("Mission", steps=["a", "b"])
    step = store.start_next_step(task.id)
    store.complete_step(task.id, step.id, "ok")
    store.checkpoint(task.id, "after-a", {"files": ["x.py"]})
    store.close()

    reopened = TaskStore(db, checkpoint_dir=tmp_path / "ckpt")
    resumed = reopened.resume(task.id)
    assert resumed["next_step"]["description"] == "b"
    assert resumed["progress"] == pytest.approx(0.5)
    assert resumed["latest_checkpoint"]["data"]["files"] == ["x.py"]
    assert reopened.get_task(task.id).status == "in_progress"
    reopened.close()


def test_invalid_status(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    task = store.create_task("x")
    with pytest.raises(ValueError):
        store.set_status(task.id, "exploded")
    with pytest.raises(KeyError):
        store.get_task(999)


def test_open_tasks(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    a = store.create_task("a")
    b = store.create_task("b")
    store.set_status(b.id, "done")
    assert [t.id for t in store.open_tasks()] == [a.id]
