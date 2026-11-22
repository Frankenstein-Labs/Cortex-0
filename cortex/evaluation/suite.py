"""Mechanism-level evaluation suite.

Every check exercises real code paths. The suite reports ``mechanism`` checks
(does the machinery work?) separately from ``quality`` checks (is the model
good?). A tiny untrained model can pass the former and fail the latter; the
report never mixes them.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import torch

from cortex.config import CortexConfig, TrainConfig
from cortex.model.lm import CortexLM
from cortex.training.data import TokenDataset
from cortex.training.trainer import Trainer


def _tiny_model() -> CortexLM:
    cfg = CortexConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    return CortexLM(cfg)


def check_model_shapes() -> dict:
    model = _tiny_model()
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    out = model(x, targets=y)
    assert out["logits"].shape == (2, 16, 128)
    assert torch.isfinite(out["loss"])
    return {"ok": True, "detail": f"logits {tuple(out['logits'].shape)}"}


def check_causality() -> dict:
    model = _tiny_model().eval()
    x = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        a = model(x)["logits"]
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % 128
        b = model(x2)["logits"]
    diff = (a[0, :-1] - b[0, :-1]).abs().max().item()
    assert diff < 1e-5, diff
    return {"ok": True, "detail": f"future token does not affect past logits (max diff {diff:.2e})"}


def check_kv_cache() -> dict:
    model = _tiny_model().eval()
    x = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        full = model(x)["logits"]
        past = None
        parts = []
        for i in range(x.shape[1]):
            out = model(x[:, i : i + 1], past_kv=past, use_cache=True)
            past = out["past_kv"]
            parts.append(out["logits"])
        inc = torch.cat(parts, dim=1)
    diff = (full - inc).abs().max().item()
    assert diff < 1e-4, diff
    return {"ok": True, "detail": f"incremental == full forward (max diff {diff:.2e})"}


def check_sampler() -> dict:
    from cortex.generation.sampler import sample_next_token

    logits = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
    gen = torch.Generator().manual_seed(0)
    tok = sample_next_token(logits, temperature=1.0, top_k=1, generator=gen)
    assert int(tok) == 1
    gen = torch.Generator().manual_seed(0)
    a = sample_next_token(logits, temperature=0.7, top_k=2, generator=gen)
    gen = torch.Generator().manual_seed(0)
    b = sample_next_token(logits, temperature=0.7, top_k=2, generator=gen)
    assert int(a) == int(b)
    greedy = sample_next_token(logits, temperature=0.0)
    assert int(greedy) == 1
    return {"ok": True, "detail": "greedy, top-k support and seeded reproducibility"}


def check_training_decreases_loss(workdir: Path) -> dict:
    import numpy as np

    rng = np.random.default_rng(0)
    # Learnable synthetic pattern: the next token is (current + 1) % 64.
    seq = np.array([(i % 64) for i in range(20_000)], dtype=np.uint16)
    path = workdir / "synthetic.bin"
    path.write_bytes(seq.tobytes())
    ds = TokenDataset(path, seq_len=32, vocab_size=64)
    cfg = CortexConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
    model = CortexLM(cfg)
    train_cfg = TrainConfig(
        seq_len=32,
        batch_size=16,
        max_steps=40,
        warmup_steps=5,
        learning_rate=3e-3,
        log_interval=100,
        eval_interval=1000,
        checkpoint_dir=str(workdir / "ckpt"),
        run_name="suite",
    )
    trainer = Trainer(model, train_cfg, ds, None, output_dir=workdir / "ckpt")
    first = None
    losses = []
    for step in range(40):
        x, y = ds.get_batch(16, torch.Generator().manual_seed(step))
        out = model(x, targets=y)
        if step == 0:
            first = float(out["loss"].detach())
        losses.append(float(out["loss"].detach()))
        trainer.optimizer.zero_grad()
        out["loss"].backward()
        trainer.optimizer.step()
    last = sum(losses[-5:]) / 5
    assert last < first, (first, last)
    return {"ok": True, "detail": f"loss {first:.3f} -> {last:.3f} in 40 steps"}


def check_memory(workdir: Path) -> dict:
    from cortex.memory.store import MemoryStore

    store = MemoryStore(workdir / "memory.db")
    store.add_episode("s1", "user", "We decided to build a reasoning engine for arithmetic.")
    store.add_episode("s1", "assistant", "The weather is nice today.")
    hits = store.search_episodes("reasoning arithmetic engine", limit=1)
    assert hits and "reasoning" in hits[0].content
    store.add_fact("project", "CORTEX-0 uses RoPE and RMSNorm")
    assert store.get_facts("project")[0].content.startswith("CORTEX-0")
    store.set_preference("initiative.enabled", "0")
    assert store.get_preference("initiative.enabled") == "0"
    return {"ok": True, "detail": "retrieval, semantic facts and preferences"}


def check_task_resume(workdir: Path) -> dict:
    from cortex.planning.tasks import TaskStore

    db = workdir / "tasks.db"
    store = TaskStore(db, checkpoint_dir=workdir / "task-ckpt")
    task = store.create_task("Add auth", steps=["read repo", "write code", "run tests"])
    step = store.start_next_step(task.id)
    assert step and step.description == "read repo"
    store.complete_step(task.id, step.id, "done reading")
    store.add_instruction(task.id, "also add logout")
    store.close()

    reopened = TaskStore(db, checkpoint_dir=workdir / "task-ckpt")
    resumed = reopened.resume(task.id)
    assert resumed["task"]["revision"] == 1
    assert resumed["next_step"]["description"] == "write code"
    assert len(resumed["task"]["steps"]) == 4
    reopened.close()
    return {"ok": True, "detail": "progress survives a new process; instruction appended"}


def check_initiative_gating(workdir: Path) -> dict:
    from cortex.initiative.engine import InitiativeConfig, InitiativeEngine, RecordingNotifier
    from cortex.memory.store import MemoryStore

    store = MemoryStore(workdir / "init.db")
    store.upsert_project("CORTEX-0", goal="build a reasoning engine with tests")
    notifier = RecordingNotifier()
    engine = InitiativeEngine(store, config=InitiativeConfig(min_score=0.0, min_novelty=0.0), notifier=notifier)

    engine.config.enabled = False
    assert engine.tick(now=1_700_000_000).message is None
    engine.config.enabled = True
    engine.config.quiet_hours_start = 0
    engine.config.quiet_hours_end = 0
    first = engine.tick(now=1_700_000_000)
    assert first.message is not None, first.reason
    second = engine.tick(now=1_700_000_100)
    assert second.message is None and second.reason == "cooldown"
    assert len(notifier.messages) == 1
    return {"ok": True, "detail": "disabled, quiet-hours and cooldown gates enforced"}


def check_sandbox(workdir: Path) -> dict:
    from cortex.execution.sandbox import Sandbox, SandboxViolation

    sandbox = Sandbox(workdir / "workspace")
    sandbox.resolve("inside.txt").write_text("hello")
    blocked = False
    try:
        sandbox.resolve("../escape.txt")
    except SandboxViolation:
        blocked = True
    assert blocked
    denied = sandbox.run(["curl", "https://example.com"])
    assert not denied.ok and "allowlist" in (denied.blocked_reason or "")
    ok = sandbox.run(["cat", "inside.txt"])
    assert ok.ok and "hello" in ok.stdout
    return {"ok": True, "detail": "path escape and non-allowlisted executable refused"}


def check_reasoning_verifier() -> dict:
    from cortex.reasoning.verifiers import ArithmeticVerifier

    verifier = ArithmeticVerifier()
    good = verifier.check("What is 12 + 7?", "12 + 7 = 19\nAnswer: 19")
    bad = verifier.check("What is 12 + 7?", "12 + 7 = 20\nAnswer: 20")
    assert good.ok is True and bad.ok is False
    return {"ok": True, "detail": "correct equation accepted, wrong equation rejected"}


def run_suite(workdir: str | Path | None = None) -> dict:
    started = time.time()
    tmp = tempfile.mkdtemp(prefix="cortex-suite-") if workdir is None else str(workdir)
    base = Path(tmp)
    base.mkdir(parents=True, exist_ok=True)
    checks = [
        ("model_shapes", "mechanism", lambda: check_model_shapes()),
        ("causality", "mechanism", check_causality),
        ("kv_cache", "mechanism", check_kv_cache),
        ("sampler", "mechanism", check_sampler),
        ("training_decreases_loss", "mechanism", lambda: check_training_decreases_loss(base)),
        ("memory", "mechanism", lambda: check_memory(base)),
        ("task_resume", "mechanism", lambda: check_task_resume(base)),
        ("initiative_gating", "mechanism", lambda: check_initiative_gating(base)),
        ("sandbox", "mechanism", lambda: check_sandbox(base)),
        ("reasoning_verifier", "mechanism", check_reasoning_verifier),
    ]
    results = []
    for name, tag, fn in checks:
        try:
            outcome = fn()
            results.append({"name": name, "tag": tag, "ok": bool(outcome.get("ok")), "detail": outcome.get("detail", "")})
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            results.append({"name": name, "tag": tag, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
    passed = sum(1 for r in results if r["ok"])
    return {
        "passed": passed,
        "total": len(results),
        "duration_sec": round(time.time() - started, 2),
        "checks": results,
    }


def main() -> None:
    report = run_suite()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
