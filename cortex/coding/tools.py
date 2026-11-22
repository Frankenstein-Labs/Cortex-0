"""Concrete workspace tools exposed to the coding agent."""

from __future__ import annotations

from typing import Dict

from ..agents.workspace import Workspace
from ..tools.registry import ToolRegistry


def build_workspace_registry(workspace: Workspace) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(
        "list_files",
        "List files in the workspace, optionally filtered by a glob pattern.",
        {"pattern": "glob pattern, default **/*", "limit": "maximum number of files"},
    )
    def list_files(pattern: str = "**/*", limit: int = 200) -> Dict:
        return {"files": [info.to_dict() for info in workspace.list_files(pattern, int(limit))]}

    @registry.register(
        "read_file",
        "Read a UTF-8 text file inside the workspace.",
        {"path": "relative path"},
    )
    def read_file(path: str) -> Dict:
        return {"path": path, "content": workspace.read(path)}

    @registry.register(
        "write_file",
        "Create or overwrite a text file inside the workspace. Requires approval.",
        {"path": "relative path", "content": "full file content"},
        dangerous=True,
    )
    def write_file(path: str, content: str) -> Dict:
        resolved = workspace.write(path, content)
        return {"path": str(resolved.relative_to(workspace.root)), "bytes": len(content.encode("utf-8"))}

    @registry.register(
        "run_command",
        "Run an allow-listed command inside the sandbox.",
        {"command": "command string", "timeout": "seconds"},
        dangerous=True,
    )
    def run_command(command: str, timeout: float = 60.0) -> Dict:
        return workspace.sandbox.run(command, timeout=float(timeout)).to_dict()

    @registry.register(
        "run_tests",
        "Run pytest inside the sandbox.",
        {"path": "test path, default tests"},
        dangerous=True,
    )
    def run_tests(path: str = "tests") -> Dict:
        return workspace.sandbox.run_tests(path).to_dict()

    @registry.register("git_status", "Show the git working tree status.", {})
    def git_status() -> Dict:
        return workspace.git_status().to_dict()

    @registry.register("git_diff", "Show the current git diff.", {"staged": "show staged diff"})
    def git_diff(staged: bool = False) -> Dict:
        return workspace.git_diff(bool(staged)).to_dict()

    @registry.register(
        "git_commit",
        "Stage all changes and create a commit. Requires approval; never pushes.",
        {"message": "commit message"},
        dangerous=True,
    )
    def git_commit(message: str) -> Dict:
        return workspace.git_commit(message).to_dict()

    return registry
