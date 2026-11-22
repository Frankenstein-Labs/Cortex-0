from __future__ import annotations

import threading

import pytest

from cortex.execution.queue import TaskQueue, Worker


def test_enqueue_and_priority_order(store):
    queue = TaskQueue(store)
    low = queue.enqueue("low", priority=0)
    high = queue.enqueue("high", priority=10)
    first = queue.dequeue("worker")
    assert first.id == high.id
    queue.ack(first.id)
    second = queue.dequeue("worker")
    assert second.id == low.id


def test_idempotency_key(store):
    queue = TaskQueue(store)
    first = queue.enqueue("job", {"x": 1}, idempotency_key="same")
    second = queue.enqueue("job", {"x": 2}, idempotency_key="same")
    assert first.id == second.id
    assert second.payload == {"x": 1}


def test_ack_and_failure_retry(store):
    queue = TaskQueue(store)
    task = queue.enqueue("job", max_attempts=2)
    running = queue.dequeue("w1")
    assert running.status == "running"
    assert running.attempts == 1
    done = queue.ack(running.id, {"value": 42})
    assert done.status == "done"
    assert done.to_dict()["result"] == {"value": 42}

    retry = queue.enqueue("job", max_attempts=2)
    queue.dequeue("w1")
    queued = queue.fail(retry.id, "boom", retry_delay=0)
    assert queued.status == "queued"
    queue.dequeue("w1")
    failed = queue.fail(retry.id, "boom", retry_delay=0)
    assert failed.status == "failed"
    assert failed.error == "boom"


def test_visibility_timeout_reclaims_task(store):
    queue = TaskQueue(store)
    task = queue.enqueue("job", run_at=0)
    first = queue.dequeue("dead-worker", now=100.0, visibility_timeout=10)
    assert first.id == task.id
    reclaimed = queue.dequeue("live-worker", now=200.0, visibility_timeout=10)
    assert reclaimed.id == task.id
    assert reclaimed.locked_by == "live-worker"


def test_cancel_and_stats(store):
    queue = TaskQueue(store)
    task = queue.enqueue("job")
    assert queue.cancel(task.id).status == "cancelled"
    stats = queue.stats()
    assert stats["cancelled"] == 1
    assert stats["total"] == 1


def test_worker_handles_success_and_failure(store):
    queue = TaskQueue(store)
    worker = Worker(queue, {"echo": lambda payload: {"echo": payload["value"]}})
    task = queue.enqueue("echo", {"value": "hello"})
    result = worker.run_once()
    assert result["status"] == "done"
    assert result["result"] == {"echo": "hello"}
    assert queue.get(task.id).status == "done"

    bad = queue.enqueue("missing")
    result = worker.run_once()
    assert result["status"] == "failed" or result["status"] == "queued"
    assert queue.get(bad.id).status in {"queued", "failed"}


def test_worker_run_forever_is_bounded(store):
    queue = TaskQueue(store)
    for index in range(3):
        queue.enqueue("echo", {"value": index})
    worker = Worker(queue, {"echo": lambda payload: payload}, poll_interval=0)
    processed = worker.run_forever(max_iterations=3, sleeper=lambda _: None)
    assert processed == 3
    assert queue.stats()["done"] == 3
