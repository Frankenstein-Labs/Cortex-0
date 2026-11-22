"""A confined software-engineering agent.

The agent reads a workspace, builds a plan, edits files, runs allowlisted
commands and tests, keeps checkpoints, and reports what it actually did.
Sensitive operations (delete, dependency install, network, deploy) require an
explicit approval.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from cortex.execution.sandbox import CommandResult, Sandbox

SENSITIVE_ACTIONS = {"delete", "install", "network", "deploy", "overwrite_outside_plan"}
_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", "checkpoints"}


class ApprovalRequired(RuntimeError):
    def __init__(self, action: str, detail: str) -> None:
        super().__init__(f"approval required for {action}: {detail}")
        self.action = action
        self.detail = detail


@dataclass
class ApprovalPolicy:
    auto_approve: set[str] = field(default_factory=set)

    def require(self, action: str, detail: str) -> None:
        if action in SENSITIVE_ACTIONS and action not in self.auto_approve:
            raise ApprovalRequired(action, detail)


@dataclass
class MissionReport:
    goal: str
    steps: list[dict] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    commands: list[dict] = field(default_factory=list)
    tests: dict | None = None
    status: str = "completed"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def as_dict(self) -> dict:
        return {
            "goal": self.goal,
            "status": self.status,
            "steps": self.steps,
            "files_changed": self.files_changed,
            "commands": self.commands,
            "tests": self.tests,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": round((self.finished_at or time.time()) - self.started_at, 3),
        }


class CodingAgent:
    def __init__(
        self,
        workspace: str | Path,
        sandbox: Sandbox | None = None,
        approvals: ApprovalPolicy | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox or Sandbox(self.workspace)
        self.approvals = approvals or ApprovalPolicy()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.workspace / ".cortex" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ workspace
    def list_files(self, max_files: int = 500) -> list[str]:
        out: list[str] = []
        for path in sorted(self.workspace.rglob("*")):
            if any(part in _IGNORED_DIRS for part in path.parts):
                continue
            if path.is_file():
                out.append(str(path.relative_to(self.workspace)))
            if len(out) >= max_files:
                break
        return out

    def read_file(self, relative_path: str | Path, max_chars: int = 200_000) -> str:
        path = self.sandbox.resolve(relative_path)
        if not path.exists():
            raise FileNotFoundError(str(relative_path))
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def search(self, needle: str, max_results: int = 50) -> list[dict]:
        results = []
        for rel in self.list_files():
            try:
                text = (self.workspace / rel).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if needle in line:
                    results.append({"file": rel, "line": lineno, "text": line.strip()[:300]})
                    if len(results) >= max_results:
                        return results
        return results

    def write_file(self, relative_path: str | Path, content: str) -> str:
        path = self.sandbox.resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.relative_to(self.workspace))

    def delete_file(self, relative_path: str | Path) -> None:
        self.approvals.require("delete", str(relative_path))
        path = self.sandbox.resolve(relative_path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    # ------------------------------------------------------------------ commands
    def run_command(self, argv: Sequence[str], timeout_sec: float | None = None) -> CommandResult:
        result = self.sandbox.run(argv, timeout_sec=timeout_sec)
        if any(x in " ".join(argv) for x in ("pip install", "apt-get", "npm install")):
            # Installations are refused after the allowlist check; keep the gate explicit.
            self.approvals.require("install", " ".join(argv))
        return result

    def run_tests(self, target: str = "tests", timeout_sec: float = 300.0) -> CommandResult:
        import sys

        return self.sandbox.run([sys.executable, "-m", "pytest", target, "-q"], timeout_sec=timeout_sec)

    # --------------------------------------------------------------- checkpoints
    def snapshot(self, label: str) -> Path:
        dest = self.checkpoint_dir / label
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for path in self.workspace.rglob("*"):
            if any(part in _IGNORED_DIRS for part in path.relative_to(self.workspace).parts):
                continue
            if path.is_file() and self.checkpoint_dir not in path.parents:
                target = dest / path.relative_to(self.workspace)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        return dest

    def restore(self, label: str) -> None:
        src = self.checkpoint_dir / label
        if not src.exists():
            raise FileNotFoundError(f"checkpoint {label!r} not found")
        for path in self.workspace.iterdir():
            if path.name in _IGNORED_DIRS or path.name == ".cortex":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        for path in src.rglob("*"):
            if path.is_file():
                target = self.workspace / path.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    # ------------------------------------------------------------------- missions
    def plan(self, goal: str, files: list[str] | None = None) -> list[dict[str, Any]]:
        """Build a deterministic, inspectable plan (no model required)."""
        files = files if files is not None else self.list_files(max_files=40)
        steps: list[dict[str, Any]] = [{"kind": "inspect", "description": f"Inspect workspace ({len(files)} files)"}]
        lowered = goal.lower()
        if any(word in lowered for word in ("test", "pytest")):
            steps.append({"kind": "test", "description": "Run the test suite"})
        if any(word in lowered for word in ("read", "analyse", "analyze", "understand", "explique")):
            steps.append({"kind": "read", "description": "Read the most relevant files"})
        if any(word in lowered for word in ("fix", "implement", "add", "build", "create", "corrige", "ajoute")):
            steps.append({"kind": "edit", "description": "Apply the requested code change"})
            steps.append({"kind": "test", "description": "Run tests after the change"})
        if any(word in lowered for word in ("commit", "diff")):
            steps.append({"kind": "diff", "description": "Show the resulting diff"})
        return steps

    def execute_mission(self, goal: str, actions: list[dict[str, Any]] | None = None) -> MissionReport:
        """Execute explicit actions: write_file / run / test / delete.

        This is intentionally not an unbounded autonomous loop: every action is
        declared, logged and gated.
        """
        report = MissionReport(goal=goal)
        plan = self.plan(goal)
        report.steps.extend({"kind": s["kind"], "description": s["description"], "status": "planned"} for s in plan)
        try:
            for action in actions or []:
                kind = action.get("kind")
                if kind == "write_file":
                    rel = self.write_file(action["path"], action["content"])
                    report.files_changed.append(rel)
                    report.steps.append({"kind": kind, "path": rel, "status": "done"})
                elif kind == "run":
                    result = self.run_command(action["argv"], timeout_sec=action.get("timeout_sec"))
                    report.commands.append(result.as_dict())
                    report.steps.append({"kind": kind, "argv": action["argv"], "status": "done" if result.ok else "failed"})
                    if not result.ok:
                        report.status = "failed"
                        break
                elif kind == "test":
                    result = self.run_tests(action.get("target", "tests"), action.get("timeout_sec", 300.0))
                    report.tests = result.as_dict()
                    report.steps.append({"kind": kind, "status": "done" if result.ok else "failed"})
                    if not result.ok:
                        report.status = "failed"
                        break
                elif kind == "delete":
                    self.delete_file(action["path"])
                    report.files_changed.append(action["path"])
                    report.steps.append({"kind": kind, "path": action["path"], "status": "done"})
                else:
                    report.steps.append({"kind": kind, "status": "skipped", "reason": "unknown action"})
        except ApprovalRequired as exc:
            report.status = "approval_required"
            report.steps.append({"kind": "approval", "action": exc.action, "detail": exc.detail, "status": "blocked"})
        report.finished_at = time.time()
        return report

    def save_report(self, report: MissionReport, path: str | Path = "artifacts/coding_report.json") -> Path:
        target = Path(path)
        if not target.is_absolute():
            target = self.workspace / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
        return target
