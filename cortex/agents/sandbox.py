"""Isolated command execution with a path jail, allow-list and resource limits.

This is a process-level sandbox for a single-user research machine, not a
container.  Production deployments must additionally use a container or VM;
that limitation is documented in ``docs/SECURITY.md``.
"""

from __future__ import annotations

import os
import re
import resource
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_ALLOWED = {
    "python",
    "python3",
    "pytest",
    "git",
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "rg",
    "find",
    "echo",
    "mkdir",
    "touch",
    "diff",
    "sed",
    "pwd",
}
GIT_ALLOWED_SUBCOMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "add",
    "commit",
    "rev-parse",
    "init",
    "branch",
    "checkout",
    "restore",
    "stash",
    "ls-files",
}
GIT_FORBIDDEN_ARGS = {"--exec", "--upload-pack", "--receive-pack", "--config"}
SAFE_ENV_KEYS = {
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "USER",
    "SHELL",
    "TERM",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "VIRTUAL_ENV",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
}
SECRET_PATTERN = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_KEY|PRIVATE)", re.I)


class SandboxError(RuntimeError):
    pass


@dataclass
class CommandResult:
    command: List[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": round(self.duration_seconds, 3),
            "timed_out": self.timed_out,
            "ok": self.ok,
        }


class Sandbox:
    def __init__(
        self,
        workspace: str | Path,
        allowed_executables: Optional[Sequence[str]] = None,
        timeout: float = 60.0,
        max_output_chars: int = 20_000,
        max_memory_bytes: int = 2 * 1024**3,
        max_file_bytes: int = 50 * 1024**2,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        if not self.workspace.exists():
            raise FileNotFoundError(self.workspace)
        self.allowed = set(allowed_executables or DEFAULT_ALLOWED)
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.max_memory_bytes = max_memory_bytes
        self.max_file_bytes = max_file_bytes

    # ------------------------------------------------------------------ path
    def resolve(self, path: str | Path, must_exist: bool = False) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if resolved != self.workspace and self.workspace not in resolved.parents:
            raise SandboxError(f"path escapes workspace: {path}")
        if must_exist and not resolved.exists():
            raise SandboxError(f"path does not exist: {path}")
        return resolved

    # --------------------------------------------------------------- command
    def _normalize(self, command: Sequence[str] | str) -> List[str]:
        if isinstance(command, str):
            parts = shlex.split(command)
        else:
            parts = [str(part) for part in command]
        if not parts:
            raise SandboxError("empty command")
        executable = Path(parts[0]).name
        if executable not in self.allowed:
            raise SandboxError(f"executable not allowed: {executable!r}")
        if executable == "git":
            self._validate_git(parts[1:])
        for part in parts:
            if "\x00" in part:
                raise SandboxError("NUL byte in command")
        return parts

    @staticmethod
    def _validate_git(args: Sequence[str]) -> None:
        if any(arg in GIT_FORBIDDEN_ARGS for arg in args):
            raise SandboxError("forbidden git argument")
        subcommand = next((arg for arg in args if not arg.startswith("-")), None)
        if subcommand is None:
            raise SandboxError("git requires a subcommand")
        if subcommand not in GIT_ALLOWED_SUBCOMMANDS:
            raise SandboxError(f"git subcommand not allowed: {subcommand!r}")

    def _environment(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env["PYTHONIOENCODING"] = "utf-8"
        for key, value in (extra or {}).items():
            if SECRET_PATTERN.search(key):
                raise SandboxError(f"refusing to pass secret-like env var: {key}")
            env[str(key)] = str(value)
        return env

    def _limits(self) -> None:
        cpu = int(self.timeout) + 2
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (self.max_memory_bytes, self.max_memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (self.max_file_bytes, self.max_file_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))

    def run(
        self,
        command: Sequence[str] | str,
        cwd: Optional[str | Path] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        parts = self._normalize(command)
        working_dir = self.resolve(cwd or self.workspace, must_exist=True)
        if not working_dir.is_dir():
            raise SandboxError(f"cwd is not a directory: {working_dir}")
        timeout = float(timeout or self.timeout)
        started = time.time()
        process = subprocess.Popen(
            parts,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=self._environment(env),
            preexec_fn=self._limits,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        duration = time.time() - started
        return CommandResult(
            command=parts,
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=self._truncate(stdout),
            stderr=self._truncate(stderr),
            duration_seconds=duration,
            timed_out=timed_out,
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        half = self.max_output_chars // 2
        return text[:half] + f"\n...[{len(text) - self.max_output_chars} chars truncated]...\n" + text[-half:]

    # ---------------------------------------------------------------- helpers
    def run_python(self, code: str, timeout: Optional[float] = None) -> CommandResult:
        tmp_dir = self.resolve(".cortex/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        script = tmp_dir / f"snippet-{int(time.time() * 1000)}.py"
        script.write_text(code, encoding="utf-8")
        try:
            return self.run(["python3", str(script)], timeout=timeout)
        finally:
            script.unlink(missing_ok=True)

    def run_tests(self, path: str = "tests", timeout: Optional[float] = None) -> CommandResult:
        return self.run(["python3", "-m", "pytest", path, "-q"], timeout=timeout)
