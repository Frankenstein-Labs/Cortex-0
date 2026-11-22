from __future__ import annotations

import time
from pathlib import Path

import pytest

from cortex.agents.orchestrator import AgentTask, Orchestrator
from cortex.agents.sandbox import Sandbox, SandboxError
from cortex.agents.workspace import Workspace
from cortex.coding.agent import CodingAction, CodingAgent, ScriptedPolicy
from cortex.planning.missions import MissionStore


def test_sandbox_allows_safe_command(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run("echo bonjour")
    assert result.ok
    assert "bonjour" in result.stdout


def test_sandbox_rejects_unknown_executable(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.run("bash -c 'echo hi'")
    with pytest.raises(SandboxError):
        sandbox.run("curl http://example.com")


def test_sandbox_rejects_git_push(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.run("git push origin main")
    with pytest.raises(SandboxError):
        sandbox.run("git remote add origin https://example.com/repo.git")


def test_sandbox_path_jail(tmp_path):
    sandbox = Sandbox(tmp_path)
    assert sandbox.resolve("inside.txt") == (tmp_path / "inside.txt").resolve()
    with pytest.raises(SandboxError):
        sandbox.resolve("../outside.txt")
    with pytest.raises(SandboxError):
        sandbox.resolve("/etc/passwd")


def test_sandbox_rejects_secret_environment(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.run(["echo", "hi"], env={"GITHUB_TOKEN": "x"})


def test_sandbox_timeout(tmp_path):
    sandbox = Sandbox(tmp_path, timeout=0.3)
    result = sandbox.run(["python3", "-c", "import time; time.sleep(2)"], timeout=0.3)
    assert result.timed_out
    assert result.duration_seconds < 2


def test_sandbox_run_python_and_tests(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_python("print(6 * 7)")
    assert result.ok
    assert "42" in result.stdout
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    assert sandbox.run_tests("tests").ok


def test_workspace_read_write_list(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.write("src/app.py", "print('hello')\n")
    assert workspace.read("src/app.py") == "print('hello')\n"
    files = [info.path for info in workspace.list_files()]
    assert "src/app.py" in files
    with pytest.raises(SandboxError):
        workspace.write("../escape.txt", "no")


def test_coding_agent_requires_approval_for_writes(tmp_path, store):
    workspace = Workspace(tmp_path)
    (tmp_path / "README.md").write_text("hello\n")
    missions = MissionStore(store)
    policy = ScriptedPolicy([
        CodingAction("list_files", {}),
        CodingAction("write_file", {"path": "new.txt", "content": "bonjour"}),
    ])
    agent = CodingAgent(workspace, missions, policy, auto_approve=False)
    mission = agent.start_mission("Créer un fichier", "Créer new.txt")
    first = agent.run_step(mission.id)
    assert first["status"] == "done"
    second = agent.run_step(mission.id)
    assert second["status"] == "awaiting_approval"
    assert not (tmp_path / "new.txt").exists()
    approved = agent.approve(mission.id)
    assert approved["status"] == "done"
    assert (tmp_path / "new.txt").read_text() == "bonjour"


def test_pending_approval_survives_agent_restart(tmp_path, store):
    workspace = Workspace(tmp_path)
    missions = MissionStore(store)
    policy = ScriptedPolicy([CodingAction("write_file", {"path": "x.txt", "content": "x"})])
    agent = CodingAgent(workspace, missions, policy, auto_approve=False)
    mission = agent.start_mission("Écrire", "Écrire x.txt")
    assert agent.run_step(mission.id)["status"] == "awaiting_approval"
    fresh = CodingAgent(workspace, missions, ScriptedPolicy([]), auto_approve=False)
    result = fresh.approve(mission.id)
    assert result["status"] == "done"
    assert (tmp_path / "x.txt").exists()
    assert fresh.approve(mission.id)["status"] == "no_pending_action"


def test_coding_agent_records_failure(tmp_path, store):
    workspace = Workspace(tmp_path)
    missions = MissionStore(store)
    agent = CodingAgent(
        workspace,
        missions,
        ScriptedPolicy([CodingAction("read_file", {"path": "missing.txt"})]),
        auto_approve=True,
    )
    mission = agent.start_mission("Lire", "Lire un fichier absent")
    result = agent.run_step(mission.id)
    assert result["status"] == "failed"
    assert "missing.txt" in str(result["result"])


def test_orchestrator_order_and_dependencies():
    calls = []
    tasks = [
        AgentTask("b", lambda deps: calls.append("b") or {"value": deps["a"]["value"] + 1}, ["a"]),
        AgentTask("a", lambda deps: calls.append("a") or {"value": 1}),
    ]
    result = Orchestrator().run(tasks)
    assert result["ok"]
    assert result["order"] == ["a", "b"]
    assert calls == ["a", "b"]
    assert result["results"]["b"]["value"] == 2


def test_orchestrator_skips_dependents_on_failure():
    def boom(deps):
        raise RuntimeError("boom")

    tasks = [AgentTask("a", boom), AgentTask("b", lambda deps: {"ok": True}, ["a"])]
    result = Orchestrator().run(tasks)
    assert not result["ok"]
    assert "a" in result["errors"]
    assert "b" in result["skipped"]


def test_orchestrator_detects_cycle():
    tasks = [AgentTask("a", lambda deps: {}, ["b"]), AgentTask("b", lambda deps: {}, ["a"])]
    with pytest.raises(ValueError):
        Orchestrator().run(tasks)
