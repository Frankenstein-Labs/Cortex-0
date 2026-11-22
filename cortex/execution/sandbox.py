"""Process-level execution sandbox for the coding agent.

Scope of the guarantee: commands run with a confined working directory, a
cleared environment, an executable allowlist and a timeout. This is *not* a
kernel-level security boundary. Run untrusted code inside a container/VM.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

DEFAULT_ALLOWLIST = {
    "python", "python3", "pytest", "ls", "cat", "head", "tail", "wc", "grep", "find",
    "git", "echo", "pwd", "mkdir", "touch", "cp", "mv", "diff", "sort", "uniq",
}

# Arguments that are never acceptable even for allowlisted executables.
_BLOCKED_ARG_PATTERNS = (
    "rm -rf /", "sudo ", "mkfs", "shutdown", "reboot", ":(){", "curl ", "wget ",
    "ssh ", "scp ", "nc ", "ncat ", "chmod 777 /", "> /dev/sd", "dd if=",
)


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False
    blocked_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.blocked_reason is None

    def as_dict(self) -> dict:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_sec": round(self.duration_sec, 3),
            "timed_out": self.timed_out,
            "blocked_reason": self.blocked_reason,
            "ok": self.ok,
        }


class SandboxViolation(RuntimeError):
    pass


@dataclass
class Sandbox:
    root: str | Path
    allowlist: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWLIST))
    timeout_sec: float = 30.0
    max_output_chars: int = 20_000
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise SandboxViolation(f"path escapes workspace: {path}")
        return resolved

    def _check_command(self, argv: Sequence[str]) -> str | None:
        if not argv:
            return "empty command"
        program = Path(argv[0]).name
        if program not in self.allowlist:
            return f"executable {program!r} is not in the allowlist"
        joined = " ".join(argv)
        for pattern in _BLOCKED_ARG_PATTERNS:
            if pattern in joined:
                return f"blocked pattern {pattern!r}"
        return None

    def run(
        self,
        argv: Sequence[str],
        timeout_sec: float | None = None,
        cwd: str | Path | None = None,
        stdin: str | None = None,
    ) -> CommandResult:
        argv = [str(a) for a in argv]
        reason = self._check_command(argv)
        workdir = self.resolve(cwd or ".")
        if reason:
            return CommandResult(argv, 126, "", reason, 0.0, blocked_reason=reason)

        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": str(workdir),
               "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        env.update(self.extra_env)
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(workdir),
                env=env,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout_sec or self.timeout_sec,
            )
            duration = time.time() - started
            return CommandResult(
                argv,
                proc.returncode,
                proc.stdout[-self.max_output_chars :],
                proc.stderr[-self.max_output_chars :],
                duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - started
            return CommandResult(
                argv,
                124,
                (exc.stdout or "")[-self.max_output_chars :] if isinstance(exc.stdout, str) else "",
                (exc.stderr or "")[-self.max_output_chars :] if isinstance(exc.stderr, str) else "",
                duration,
                timed_out=True,
            )

    def run_python_file(self, relative_path: str | Path, args: Sequence[str] = (), timeout_sec: float | None = None):
        target = self.resolve(relative_path)
        return self.run([sys.executable, str(target), *args], timeout_sec=timeout_sec)

    def shell_hint(self, command: str) -> list[str]:
        """Parse a command string into argv without invoking a shell."""
        return shlex.split(command)
