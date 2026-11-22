"""Small multi-agent orchestrator with dependency-aware execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence


@dataclass
class AgentTask:
    name: str
    handler: Callable[[dict], dict]
    depends_on: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.depends_on = tuple(self.depends_on)


class Orchestrator:
    def __init__(self, max_workers: int = 2) -> None:
        self.max_workers = max_workers

    def _validate(self, tasks: Sequence[AgentTask]) -> None:
        names = [task.name for task in tasks]
        if len(names) != len(set(names)):
            raise ValueError("duplicate task names")
        known = set(names)
        for task in tasks:
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(f"task {task.name!r} depends on unknown tasks: {sorted(missing)}")

    def _order(self, tasks: Sequence[AgentTask]) -> List[str]:
        pending = {task.name: set(task.depends_on) for task in tasks}
        order: List[str] = []
        while pending:
            ready = sorted(name for name, deps in pending.items() if not deps)
            if not ready:
                raise ValueError("dependency cycle detected")
            for name in ready:
                order.append(name)
                pending.pop(name)
            for deps in pending.values():
                deps.difference_update(ready)
        return order

    def run(self, tasks: Sequence[AgentTask], parallel: bool = False) -> dict:
        self._validate(tasks)
        by_name = {task.name: task for task in tasks}
        order = self._order(tasks)
        results: Dict[str, dict] = {}
        errors: Dict[str, str] = {}
        skipped: List[str] = []

        def execute(name: str) -> None:
            task = by_name[name]
            failed_deps = [dep for dep in task.depends_on if dep in errors or dep in skipped]
            if failed_deps:
                skipped.append(name)
                return
            payload = {dep: results[dep] for dep in task.depends_on}
            try:
                results[name] = task.handler(payload)
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"

        if not parallel:
            for name in order:
                execute(name)
        else:
            completed: set[str] = set()
            while len(completed) < len(order):
                ready = [
                    name
                    for name in order
                    if name not in completed
                    and all(dep in completed or dep in skipped for dep in by_name[name].depends_on)
                ]
                if not ready:
                    break
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    list(pool.map(execute, ready))
                completed.update(ready)
        return {
            "order": order,
            "results": results,
            "errors": errors,
            "skipped": skipped,
            "ok": not errors and not skipped,
        }
