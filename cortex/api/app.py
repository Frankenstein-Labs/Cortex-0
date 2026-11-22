"""FastAPI service for CORTEX-0: chat, generation, memory, missions, autonomy."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from .. import __version__
from ..agents.workspace import Workspace
from ..coding.agent import CodingAgent, ModelPolicy
from ..coding.tools import build_workspace_registry
from ..conversation.companion import Companion
from ..execution.queue import TaskQueue
from ..initiative.engine import InitiativeEngine
from ..initiative.scheduler import InitiativeScheduler
from ..memory.store import KINDS, MemoryStore
from ..model.generation import generate, generate_stream
from ..planning.goals import GoalStore
from ..planning.missions import MissionStore
from ..reasoning.engine import ReasoningConfig, ReasoningEngine
from ..training.trainer import load_pretrained
from .schemas import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    GoalCreateRequest,
    GoalUpdateRequest,
    InitiativeTickRequest,
    MemoryAddRequest,
    MemoryAddResponse,
    MissionCreateRequest,
    MissionInstructionRequest,
    MissionRunRequest,
    ReasonRequest,
    TaskCreateRequest,
)

log = logging.getLogger("cortex.api")


@dataclass
class Settings:
    model_dir: Optional[str] = None
    db_path: str = "cortex.sqlite"
    workspace: str = "."
    api_key: Optional[str] = None
    device: str = "cpu"
    enable_scheduler: bool = False
    scheduler_interval: float = 60.0
    default_session: str = "default"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_dir=os.environ.get("CORTEX_MODEL_DIR"),
            db_path=os.environ.get("CORTEX_DB_PATH", "cortex.sqlite"),
            workspace=os.environ.get("CORTEX_WORKSPACE", "."),
            api_key=os.environ.get("CORTEX_API_KEY"),
            device=os.environ.get("CORTEX_DEVICE", "cpu"),
            enable_scheduler=os.environ.get("CORTEX_ENABLE_SCHEDULER", "0") == "1",
            scheduler_interval=float(os.environ.get("CORTEX_SCHEDULER_INTERVAL", "60")),
            default_session=os.environ.get("CORTEX_DEFAULT_SESSION", "default"),
        )


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _model_info(app: FastAPI) -> Dict[str, Any]:
    model = app.state.model
    info: Dict[str, Any] = {
        "loaded": model is not None,
        "model_dir": app.state.settings.model_dir,
        "device": app.state.settings.device,
        "version": __version__,
        "error": app.state.model_error,
    }
    if model is not None:
        config = model.config
        info.update(
            {
                "architecture": "cortex-decoder-llama-compatible",
                "parameters": model.num_parameters(),
                "parameters_non_embedding": model.num_parameters(non_embedding=True),
                "context_length": config.max_position_embeddings,
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "layers": config.num_hidden_layers,
                "heads": config.num_attention_heads,
                "kv_heads": config.num_key_value_heads,
                "tie_word_embeddings": config.tie_word_embeddings,
            }
        )
    return info


def _encode_prompt(app: FastAPI, prompt: str, max_new_tokens: int) -> List[int]:
    tokenizer = app.state.tokenizer
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.bos_token_id]
    limit = app.state.model.config.max_position_embeddings - max_new_tokens
    return ids[-limit:] if limit > 0 else ids[-1:]


def _token_deltas(app: FastAPI, req: GenerateRequest):
    ids = _encode_prompt(app, req.prompt, req.max_new_tokens)
    input_ids = torch.tensor([ids], dtype=torch.long, device=app.state.settings.device)
    generator = torch.Generator(device="cpu").manual_seed(req.seed)
    produced: List[int] = []
    emitted = 0
    for token in generate_stream(
        app.state.model,
        input_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        eos_token_id=app.state.tokenizer.eos_token_id,
        stop_token_ids=[app.state.tokenizer.end_token_id],
        generator=generator,
        pad_token_id=app.state.tokenizer.pad_token_id,
    ):
        produced.append(int(token[0]))
        text = app.state.tokenizer.decode(produced, skip_special_tokens=True)
        if "\ufffd" in text:
            continue
        if len(text) > emitted:
            yield text[emitted:]
            emitted = len(text)


def create_app(
    settings: Optional[Settings] = None,
    store: Optional[MemoryStore] = None,
    model=None,
    tokenizer=None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or MemoryStore(settings.db_path)
    companion = Companion(
        store,
        model=model,
        tokenizer=tokenizer,
        device=settings.device,
    )
    app = FastAPI(title="CORTEX-0", version=__version__)
    app.state.settings = settings
    app.state.store = store
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.model_error = None
    app.state.companion = companion
    app.state.goals = GoalStore(store)
    app.state.missions = MissionStore(store)
    app.state.tasks = TaskQueue(store)
    app.state.initiatives = InitiativeEngine(store)
    app.state.started_at = time.time()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.model is None and settings.model_dir:
            try:
                loaded_model, loaded_tokenizer, _ = load_pretrained(Path(settings.model_dir))
                application.state.model = loaded_model
                application.state.tokenizer = loaded_tokenizer
                application.state.companion.model = loaded_model
                application.state.companion.tokenizer = loaded_tokenizer
                log.info("loaded model from %s", settings.model_dir)
            except Exception as exc:  # a broken checkpoint must not kill the API
                application.state.model_error = f"{type(exc).__name__}: {exc}"
                log.exception("model load failed")
        scheduler = None
        if settings.enable_scheduler:
            scheduler = InitiativeScheduler(
                application.state.initiatives,
                store,
                interval_seconds=settings.scheduler_interval,
                session_id=settings.default_session,
            )
            scheduler.start()
        yield
        if scheduler is not None:
            scheduler.stop()

    app.router.lifespan_context = lifespan

    def auth(x_api_key: Optional[str] = Header(default=None, alias="x-api-key")) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    router = APIRouter(prefix="/v1", dependencies=[Depends(auth)])

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {"name": "CORTEX-0", "version": __version__, "docs": "/docs"}

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "model_loaded": app.state.model is not None,
            "uptime_seconds": round(time.time() - app.state.started_at, 1),
        }

    @router.get("/model/info")
    def model_info() -> Dict[str, Any]:
        return _model_info(app)

    @router.post("/generate", response_model=GenerateResponse)
    def generate_endpoint(req: GenerateRequest) -> Dict[str, Any]:
        if app.state.model is None:
            raise HTTPException(503, app.state.model_error or "no model loaded")
        ids = _encode_prompt(app, req.prompt, req.max_new_tokens)
        input_ids = torch.tensor([ids], dtype=torch.long, device=app.state.settings.device)
        generator = torch.Generator(device="cpu").manual_seed(req.seed)
        output = generate(
            app.state.model,
            input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            eos_token_id=app.state.tokenizer.eos_token_id,
            stop_token_ids=[app.state.tokenizer.end_token_id],
            generator=generator,
            pad_token_id=app.state.tokenizer.pad_token_id,
        )
        new_ids = output[0, len(ids):].tolist()
        return {
            "text": app.state.tokenizer.decode(new_ids, skip_special_tokens=True),
            "prompt_tokens": len(ids),
            "generated_tokens": len(new_ids),
            "model": _model_info(app),
        }

    @router.post("/generate/stream")
    def generate_stream_endpoint(req: GenerateRequest) -> StreamingResponse:
        if app.state.model is None:
            raise HTTPException(503, app.state.model_error or "no model loaded")

        def event_stream():
            for delta in _token_deltas(app, req):
                yield _sse({"delta": delta})
            yield _sse({"done": True})

        return StreamingResponse(iterate_in_threadpool(event_stream()), media_type="text/event-stream")

    @router.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> Dict[str, Any]:
        return app.state.companion.chat(req.session_id, req.message).to_dict()

    @router.post("/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        def event_stream():
            for delta in app.state.companion.stream_chat(req.session_id, req.message):
                yield _sse({"delta": delta})
            yield _sse({"done": True})

        return StreamingResponse(iterate_in_threadpool(event_stream()), media_type="text/event-stream")

    @router.post("/reason")
    def reason(req: ReasonRequest) -> Dict[str, Any]:
        if app.state.model is None:
            raise HTTPException(503, app.state.model_error or "no model loaded")
        engine = ReasoningEngine(
            app.state.model,
            app.state.tokenizer,
            ReasoningConfig(
                num_samples=req.num_samples,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                seed=req.seed,
            ),
            device=app.state.settings.device,
            store=store,
        )
        return engine.solve(req.question, expected=req.expected).to_dict()

    @router.get("/memory/stats")
    def memory_stats() -> Dict[str, Any]:
        return store.stats()

    @router.post("/memory", response_model=MemoryAddResponse)
    def memory_add(req: MemoryAddRequest) -> Dict[str, Any]:
        if req.kind not in KINDS:
            raise HTTPException(400, f"unknown kind: {req.kind}")
        memory_id = store.add(
            req.content,
            kind=req.kind,
            importance=req.importance,
            source=req.source,
            session_id=req.session_id,
        )
        hit = store.get(memory_id)
        return {"id": hit.id, "kind": hit.kind, "summary": hit.summary}

    @router.get("/memory/search")
    def memory_search(
        q: str = Query(min_length=1),
        limit: int = Query(default=8, ge=1, le=50),
        kinds: Optional[str] = None,
    ) -> Dict[str, Any]:
        kind_list = [kind.strip() for kind in kinds.split(",")] if kinds else None
        hits = store.search(q, limit=limit, kinds=kind_list)
        return {"query": q, "hits": [hit.to_dict() for hit in hits]}

    @router.get("/memory/recent")
    def memory_recent(
        kind: Optional[str] = None, limit: int = Query(default=10, ge=1, le=100)
    ) -> Dict[str, Any]:
        return {"memories": [hit.to_dict() for hit in store.recent(kind=kind, limit=limit)]}

    @router.post("/memory/consolidate")
    def memory_consolidate(session_id: Optional[str] = None, min_events: int = 6) -> Dict[str, Any]:
        return store.consolidate(session_id=session_id, min_events=min_events)

    @router.get("/sessions")
    def sessions(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
        return {"sessions": store.sessions(limit=limit)}

    @router.get("/goals")
    def list_goals(status: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200)):
        return {"goals": [goal.to_dict() for goal in app.state.goals.list(status=status, limit=limit)]}

    @router.post("/goals")
    def create_goal(req: GoalCreateRequest) -> Dict[str, Any]:
        return app.state.goals.create(req.title, req.description, req.priority).to_dict()

    @router.patch("/goals/{goal_id}")
    def update_goal(goal_id: int, req: GoalUpdateRequest) -> Dict[str, Any]:
        try:
            goal = app.state.goals.update(
                goal_id, status=req.status, progress=req.progress, priority=req.priority
            )
        except KeyError:
            raise HTTPException(404, "goal not found")
        return goal.to_dict()

    def mission_agent(auto_approve: bool) -> CodingAgent:
        if app.state.model is None:
            raise HTTPException(503, app.state.model_error or "no model loaded")
        workspace_path = Path(settings.workspace)
        if not workspace_path.exists():
            raise HTTPException(400, f"workspace does not exist: {workspace_path}")
        workspace = Workspace(workspace_path)
        registry = build_workspace_registry(workspace)
        policy = ModelPolicy(
            app.state.model, app.state.tokenizer, registry, device=settings.device
        )
        return CodingAgent(
            workspace, app.state.missions, policy, registry, auto_approve=auto_approve
        )

    @router.get("/missions")
    def list_missions(status: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200)):
        return {"missions": [mission.to_dict() for mission in app.state.missions.list(status=status, limit=limit)]}

    @router.post("/missions")
    def create_mission(req: MissionCreateRequest) -> Dict[str, Any]:
        return app.state.missions.create(req.title, req.instruction).to_dict()

    @router.get("/missions/{mission_id}")
    def get_mission(mission_id: str) -> Dict[str, Any]:
        mission = app.state.missions.get(mission_id)
        if mission is None:
            raise HTTPException(404, "mission not found")
        return mission.to_dict()

    @router.post("/missions/{mission_id}/instruction")
    def add_instruction(mission_id: str, req: MissionInstructionRequest) -> Dict[str, Any]:
        try:
            mission = app.state.missions.add_instruction(mission_id, req.instruction)
        except KeyError:
            raise HTTPException(404, "mission not found")
        return mission.to_dict()

    @router.post("/missions/{mission_id}/run")
    def run_mission(mission_id: str, req: MissionRunRequest) -> Dict[str, Any]:
        agent = mission_agent(req.auto_approve)
        try:
            return agent.run(mission_id, max_steps=req.max_steps)
        except KeyError:
            raise HTTPException(404, "mission not found")

    @router.post("/missions/{mission_id}/approve")
    def approve_mission(mission_id: str) -> Dict[str, Any]:
        agent = mission_agent(auto_approve=True)
        return agent.approve(mission_id)

    @router.post("/missions/{mission_id}/reject")
    def reject_mission(mission_id: str, reason: str = "rejected by user") -> Dict[str, Any]:
        agent = mission_agent(auto_approve=True)
        return agent.reject(mission_id, reason=reason)

    @router.get("/tasks/stats")
    def task_stats() -> Dict[str, int]:
        return app.state.tasks.stats()

    @router.get("/tasks")
    def list_tasks(status: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200)):
        return {"tasks": [task.to_dict() for task in app.state.tasks.list(status=status, limit=limit)]}

    @router.post("/tasks")
    def create_task(req: TaskCreateRequest) -> Dict[str, Any]:
        task = app.state.tasks.enqueue(
            req.kind,
            req.payload,
            run_at=req.run_at,
            priority=req.priority,
            max_attempts=req.max_attempts,
            idempotency_key=req.idempotency_key,
        )
        return task.to_dict()

    @router.get("/tasks/{task_id}")
    def get_task(task_id: int) -> Dict[str, Any]:
        task = app.state.tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "task not found")
        return task.to_dict()

    @router.post("/tasks/{task_id}/cancel")
    def cancel_task(task_id: int) -> Dict[str, Any]:
        return app.state.tasks.cancel(task_id).to_dict()

    @router.get("/initiatives")
    def list_initiatives(
        status: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200)
    ):
        return {
            "initiatives": [
                initiative.to_dict()
                for initiative in app.state.initiatives.recent(limit=limit, status=status)
            ]
        }

    @router.post("/initiatives/tick")
    def tick_initiatives(req: InitiativeTickRequest) -> Dict[str, Any]:
        created = app.state.initiatives.tick(
            session_id=req.session_id, user_last_active=req.user_last_active
        )
        return {"initiatives": [initiative.to_dict() for initiative in created]}

    @router.post("/initiatives/{initiative_id}/dismiss")
    def dismiss_initiative(initiative_id: int) -> Dict[str, Any]:
        return app.state.initiatives.dismiss(initiative_id).to_dict()

    @router.post("/initiatives/{initiative_id}/send")
    def send_initiative(initiative_id: int) -> Dict[str, Any]:
        return app.state.initiatives.mark_sent(initiative_id).to_dict()

    @router.get("/events")
    def events() -> StreamingResponse:
        async def event_stream():
            seen = set()
            for _ in range(30):
                for initiative in app.state.initiatives.pending():
                    if initiative.id not in seen:
                        seen.add(initiative.id)
                        yield _sse({"initiative": initiative.to_dict()})
                await asyncio.sleep(2)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.get("/tools")
    def tools() -> Dict[str, Any]:
        workspace = Workspace(Path(settings.workspace))
        return {"tools": build_workspace_registry(workspace).list()}

    app.include_router(router)
    return app


app = create_app()
