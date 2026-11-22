from __future__ import annotations

import pytest

from cortex.coding.agent import ApprovalPolicy, ApprovalRequired, CodingAgent
from cortex.execution.sandbox import Sandbox, SandboxViolation


def test_sandbox_confines_paths(tmp_path):
    sandbox = Sandbox(tmp_path)
    (tmp_path / "inside.txt").write_text("hello")
    assert sandbox.resolve("inside.txt").exists()
    with pytest.raises(SandboxViolation):
        sandbox.resolve("../outside.txt")


def test_sandbox_allowlist_and_timeout(tmp_path):
    sandbox = Sandbox(tmp_path)
    denied = sandbox.run(["curl", "https://example.com"])
    assert not denied.ok
    assert "allowlist" in (denied.blocked_reason or "")
    slow = sandbox.run(["python3", "-c", "import time; time.sleep(3)"], timeout_sec=0.2)
    assert slow.timed_out


def test_agent_read_write_search(tmp_path):
    agent = CodingAgent(tmp_path)
    agent.write_file("pkg/module.py", "def add(a, b):\n    return a + b\n")
    assert "def add" in agent.read_file("pkg/module.py")
    assert "pkg/module.py" in agent.list_files()
    results = agent.search("return a + b")
    assert results and results[0]["line"] == 2


def test_agent_blocks_escape(tmp_path):
    agent = CodingAgent(tmp_path / "ws")
    with pytest.raises(SandboxViolation):
        agent.write_file("../escape.py", "bad")


def test_agent_delete_requires_approval(tmp_path):
    agent = CodingAgent(tmp_path)
    agent.write_file("x.txt", "data")
    with pytest.raises(ApprovalRequired):
        agent.delete_file("x.txt")
    assert (tmp_path / "x.txt").exists()

    approved = CodingAgent(tmp_path, approvals=ApprovalPolicy(auto_approve={"delete"}))
    approved.delete_file("x.txt")
    assert not (tmp_path / "x.txt").exists()


def test_agent_snapshot_restore(tmp_path):
    agent = CodingAgent(tmp_path)
    agent.write_file("a.txt", "original")
    agent.snapshot("before")
    agent.write_file("a.txt", "changed")
    agent.write_file("b.txt", "new")
    agent.restore("before")
    assert (tmp_path / "a.txt").read_text() == "original"
    assert not (tmp_path / "b.txt").exists()


def test_agent_runs_python_file(tmp_path):
    agent = CodingAgent(tmp_path)
    agent.write_file("hello.py", "print('hello from cortex')\n")
    result = agent.sandbox.run_python_file("hello.py")
    assert result.ok
    assert "hello from cortex" in result.stdout


def test_mission_report(tmp_path):
    agent = CodingAgent(tmp_path)
    report = agent.execute_mission(
        "create and run a script",
        actions=[
            {"kind": "write_file", "path": "run.py", "content": "print(6 * 7)\n"},
            {"kind": "run", "argv": ["python3", "run.py"]},
        ],
    )
    assert report.status == "completed"
    assert report.files_changed == ["run.py"]
    assert report.commands[0]["stdout"].strip() == "42"
    assert any(step["kind"] == "inspect" for step in report.steps)


def test_mission_blocks_sensitive_action(tmp_path):
    agent = CodingAgent(tmp_path)
    agent.write_file("secret.txt", "x")
    report = agent.execute_mission("delete a file", actions=[{"kind": "delete", "path": "secret.txt"}])
    assert report.status == "approval_required"
    assert (tmp_path / "secret.txt").exists()


def test_plan_is_inspectable(tmp_path):
    agent = CodingAgent(tmp_path)
    agent.write_file("main.py", "print('x')")
    plan = agent.plan("fix the failing tests")
    kinds = {step["kind"] for step in plan}
    assert "inspect" in kinds and "test" in kinds and "edit" in kinds
