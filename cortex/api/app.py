"""FastAPI service for CORTEX-0."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from cortex.engine import CortexEngine


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int | None = Field(default=50, ge=1)
    top_p: float | None = Field(default=0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=5.0)
    seed: int | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = "default"
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int | None = 50
    top_p: float | None = 0.95
    seed: int | None = None


class TaskCreateRequest(BaseModel):
    title: str = Field(min_length=1)
    goal: str = ""
    steps: list[str] = Field(default_factory=list)


class InstructionRequest(BaseModel):
    instruction: str = Field(min_length=1)
    steps: list[str] | None = None


def create_app(engine: CortexEngine | None = None) -> FastAPI:
    app = FastAPI(title="CORTEX-0", version="0.1.0", description="Small autonomous language model by Frankenstein-Labs")
    state = {"engine": engine or CortexEngine()}

    def get_engine() -> CortexEngine:
        return state["engine"]

    @app.get("/health")
    def health() -> dict:
        eng = get_engine()
        return {"status": "ok" if eng.loaded else "model_not_loaded", **eng.status()}

    @app.get("/model/info")
    def model_info() -> dict:
        return get_engine().status()

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict:
        try:
            return get_engine().generate(req.prompt, **req.model_dump(exclude={"prompt"}))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        try:
            return get_engine().chat(req.message, req.session_id, **req.model_dump(exclude={"message", "session_id"}))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/memory/stats")
    def memory_stats() -> dict:
        return get_engine().memory.stats()

    @app.get("/memory/search")
    def memory_search(q: str = Query(min_length=1), limit: int = Query(default=5, ge=1, le=50)) -> dict:
        return {"query": q, "results": [e.as_dict() for e in get_engine().memory.search_episodes(q, limit)]}

    @app.get("/tasks")
    def list_tasks(status: str | None = None) -> dict:
        return {"tasks": [t.as_dict() for t in get_engine().tasks.list_tasks(status=status)]}

    @app.post("/tasks")
    def create_task(req: TaskCreateRequest) -> dict:
        return get_engine().tasks.create_task(req.title, req.goal, req.steps).as_dict()

    @app.get("/tasks/{task_id}")
    def get_task(task_id: int) -> dict:
        try:
            return get_engine().tasks.get_task(task_id).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/tasks/{task_id}/instructions")
    def add_instruction(task_id: int, req: InstructionRequest) -> dict:
        try:
            return get_engine().tasks.add_instruction(task_id, req.instruction, req.steps).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/tasks/{task_id}/resume")
    def resume_task(task_id: int) -> dict:
        try:
            return get_engine().tasks.resume(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/initiative/tick")
    def initiative_tick(session_id: str = "default") -> dict:
        return get_engine().tick(session_id)

    @app.get("/initiative/history")
    def initiative_history(session_id: str = "default", limit: int = 20) -> dict:
        return {"messages": get_engine().initiative.history(session_id, limit)}

    @app.get("/tools")
    def tools() -> dict[str, Any]:
        return {"tools": get_engine().tools.list()}

    return app
