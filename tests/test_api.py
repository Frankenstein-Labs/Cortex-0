from __future__ import annotations

from fastapi.testclient import TestClient

from cortex.api.app import create_app
from cortex.engine import CortexEngine, Settings
from cortex.model.lm import save_checkpoint


def _client(tmp_path, tiny_model, tiny_tokenizer, load_model=True):
    tokenizer_path = tmp_path / "tokenizer.json"
    tiny_tokenizer.save(tokenizer_path)
    checkpoint = tmp_path / "model.safetensors"
    save_checkpoint(tiny_model, checkpoint, extra={"step": 1})
    settings = Settings(
        checkpoint=str(checkpoint),
        tokenizer=str(tokenizer_path),
        memory_db=str(tmp_path / "memory.db"),
        tasks_db=str(tmp_path / "tasks.db"),
        task_checkpoints=str(tmp_path / "task-ckpt"),
        initiative_enabled=False,
        load_model=load_model,
    )
    engine = CortexEngine(settings)
    return TestClient(create_app(engine)), engine


def test_health_and_model_info(tmp_path, tiny_model, tiny_tokenizer):
    client, _ = _client(tmp_path, tiny_model, tiny_tokenizer)
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["loaded"] is True
    assert health["parameters"] == tiny_model.num_parameters()
    info = client.get("/model/info").json()
    assert info["vocab_size"] == tiny_tokenizer.vocab_size


def test_generate_and_chat(tmp_path, tiny_model, tiny_tokenizer):
    client, _ = _client(tmp_path, tiny_model, tiny_tokenizer)
    resp = client.post("/generate", json={"prompt": "Once", "max_new_tokens": 4, "temperature": 0.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prompt"] == "Once"
    assert body["generated_tokens"] >= 1

    chat = client.post("/chat", json={"message": "Hello", "max_new_tokens": 4, "temperature": 0.0})
    assert chat.status_code == 200
    assert chat.json()["session_id"] == "default"

    memory = client.get("/memory/search", params={"q": "Hello"}).json()
    assert memory["results"]


def test_unloaded_model_returns_503(tmp_path, tiny_model, tiny_tokenizer):
    client, _ = _client(tmp_path, tiny_model, tiny_tokenizer, load_model=False)
    assert client.get("/health").json()["status"] == "model_not_loaded"
    assert client.post("/generate", json={"prompt": "x"}).status_code == 503


def test_tasks_api(tmp_path, tiny_model, tiny_tokenizer):
    client, _ = _client(tmp_path, tiny_model, tiny_tokenizer)
    created = client.post("/tasks", json={"title": "Build", "steps": ["a", "b"]}).json()
    task_id = created["id"]
    assert client.get(f"/tasks/{task_id}").json()["title"] == "Build"
    updated = client.post(f"/tasks/{task_id}/instructions", json={"instruction": "also c"}).json()
    assert updated["revision"] == 1
    resumed = client.get(f"/tasks/{task_id}/resume").json()
    assert resumed["next_step"]["description"] == "a"
    assert len(client.get("/tasks").json()["tasks"]) == 1
    assert client.get("/tasks/9999").status_code == 404


def test_initiative_and_tools(tmp_path, tiny_model, tiny_tokenizer):
    client, _ = _client(tmp_path, tiny_model, tiny_tokenizer)
    decision = client.post("/initiative/tick").json()
    assert "message" in decision and "reason" in decision
    assert decision["message"] is None  # disabled in test settings
    tools = client.get("/tools").json()["tools"]
    assert any(t["name"] == "calculator" for t in tools)
