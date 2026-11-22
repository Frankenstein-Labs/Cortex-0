"""Workspace file and git operations constrained by the sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .sandbox import CommandResult, Sandbox

IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".cortex", ".venv"}


@dataclass
class FileInfo:
    path: str
    size: int

    def to_dict(self) -> dict:
        return {"path": self.path, "size": self.size}


class Workspace:
    def __init__(self, root: str | Path, sandbox: Optional[Sandbox] = None) -> None:
        self.sandbox = sandbox or Sandbox(root)
        self.root = self.sandbox.workspace
        self.max_read_bytes = 200_000
        self.max_write_bytes = 1_000_000

    def list_files(self, pattern: str = "**/*", limit: int = 500) -> List[FileInfo]:
        files: List[FileInfo] = []
        for path in sorted(self.root.glob(pattern)):
            if not path.is_file():
                continue
            if any(part in IGNORED_PARTS for part in path.parts):
                continue
            files.append(FileInfo(str(path.relative_to(self.root)), path.stat().st_size))
            if len(files) >= limit:
                break
        return files

    def read(self, path: str | Path) -> str:
        resolved = self.sandbox.resolve(path, must_exist=True)
        if resolved.stat().st_size > self.max_read_bytes:
            raise ValueError(f"file too large to read: {resolved}")
        return resolved.read_text(encoding="utf-8", errors="replace")

    def write(self, path: str | Path, content: str) -> Path:
        resolved = self.sandbox.resolve(path)
        if len(content.encode("utf-8")) > self.max_write_bytes:
            raise ValueError("content exceeds maximum write size")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return resolved

    def exists(self, path: str | Path) -> bool:
        try:
            return self.sandbox.resolve(path).exists()
        except Exception:
            return False

    def git_status(self) -> CommandResult:
        return self.sandbox.run(["git", "status", "--short"])

    def git_diff(self, staged: bool = False) -> CommandResult:
        command = ["git", "diff"]
        if staged:
            command.append("--staged")
        return self.sandbox.run(command)

    def git_commit(self, message: str, add_all: bool = True) -> CommandResult:
        if add_all:
            added = self.sandbox.run(["git", "add", "-A"])
            if not added.ok:
                return added
        return self.sandbox.run(["git", "commit", "-m", message])
