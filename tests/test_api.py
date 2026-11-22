from __future__ import annotations

from fastapi.testclient import TestClient

from cortex.api.app import Settings, create_app


def make_client(tmp_path, api_key=None, model=None, tokenizer=None):
    settings = Settings(
        db_path=str(tmp_path / "api.sqlite"),
        workspace=str(tmp_path),
        api_key=api_key,
        device="cpu",
    )
    app = create_app(settings, model=model, tokenizer=tokenizer)
    return TestClient(app)


def test_health_is_public_and_v1_requires_key(tmp_path):
    with make_client(tmp_path, api_key="secret") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/model/info").status_code == 401
        assert client.get("/v1/model/info", headers={"x-api-key": "wrong"}).status_code == 401
        response = client.get("/v1/model/info", headers={"x-api-key": "secret"})
        assert response.status_code == 200
        assert response.json()["loaded"] is False


def test_memory_goals_and_tasks(tmp_path):
    with make_client(tmp_path) as client:
        added = client.post(
            "/v1/memory",
            json={"content": "CORTEX-0 est développé par Frankenstein-Labs", "kind": "project"},
        )
        assert added.status_code == 200
        found = client.get("/v1/memory/search", params={"q": "CORTEX Frankenstein"})
        assert found.status_code == 200
        assert found.json()["hits"]

        goal = client.post("/v1/goals", json={"title": "Entraîner le modèle", "priority": 0.9})
        assert goal.status_code == 200
        goal_id = goal.json()["id"]
        patched = client.patch(f"/v1/goals/{goal_id}", json={"progress": 0.5})
        assert patched.json()["progress"] == 0.5

        task = client.post("/v1/tasks", json={"kind": "echo", "payload": {"x": 1}})
        assert task.status_code == 200
        assert client.get("/v1/tasks/stats").json()["queued"] == 1
        assert client.post(f"/v1/tasks/{task.json()['id']}/cancel").json()["status"] == "cancelled"


def test_chat_fallback_without_model(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/chat", json={"session_id": "s1", "message": "Bonjour CORTEX"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["used_model"] is False
        assert payload["text"]
        assert payload["session_id"] == "s1"
        events = client.get("/v1/sessions").json()["sessions"]
        assert events[0]["session_id"] == "s1"


def test_chat_stream_sse(tmp_path):
    with make_client(tmp_path) as client:
        with client.stream(
            "POST", "/v1/chat/stream", json={"session_id": "s2", "message": "Bonjour"}
        ) as response:
            body = "".join(response.iter_text())
        assert response.status_code == 200
        assert "data:" in body
        assert '"done": true' in body


def test_generate_requires_model(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/generate", json={"prompt": "Bonjour", "max_new_tokens": 4})
        assert response.status_code == 503


def test_generate_and_reason_with_model(tmp_path, tiny_model, tokenizer):
    with make_client(tmp_path, model=tiny_model, tokenizer=tokenizer) as client:
        generated = client.post(
            "/v1/generate",
            json={"prompt": "Bonjour", "max_new_tokens": 4, "temperature": 0.0},
        )
        assert generated.status_code == 200
        payload = generated.json()
        assert payload["model"]["loaded"] is True
        assert isinstance(payload["text"], str)
        assert payload["generated_tokens"] <= 4

        chat = client.post("/v1/chat", json={"session_id": "m1", "message": "Bonjour"})
        assert chat.status_code == 200
        assert chat.json()["used_model"] is True

        reasoned = client.post(
            "/v1/reason",
            json={"question": "Combien font 2 + 2 ?", "expected": "4", "num_samples": 1, "max_new_tokens": 8},
        )
        assert reasoned.status_code == 200
        assert "candidates" in reasoned.json()


def test_mission_requires_model(tmp_path):
    with make_client(tmp_path) as client:
        mission = client.post("/v1/missions", json={"title": "Test", "instruction": "Créer un fichier"})
        assert mission.status_code == 200
        run = client.post(f"/v1/missions/{mission.json()['id']}/run", json={"max_steps": 1})
        assert run.status_code == 503


def test_memory_rejects_unknown_kind(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/v1/memory", json={"content": "x", "kind": "invented"})
        assert response.status_code == 400


def test_tools_endpoint(tmp_path):
    with make_client(tmp_path) as client:
        tools = client.get("/v1/tools").json()["tools"]
        names = {tool["name"] for tool in tools}
        assert {"read_file", "write_file", "run_tests", "git_status"} <= names
