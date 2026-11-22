"""Pydantic request/response models for the CORTEX-0 API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    text: str
    session_id: str
    used_model: bool
    memory_hits: List[Dict[str, Any]] = []
    initiative: Optional[Dict[str, Any]] = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=1, le=4096)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    seed: int = 0


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    model: Dict[str, Any]


class MemoryAddRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    kind: str = "long_term"
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    session_id: Optional[str] = None
    source: str = "api"


class MemoryAddResponse(BaseModel):
    id: int
    kind: str
    summary: str


class GoalCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    description: str = ""
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class GoalUpdateRequest(BaseModel):
    status: Optional[str] = None
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    priority: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MissionCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=4000)


class MissionInstructionRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=4000)


class MissionRunRequest(BaseModel):
    max_steps: int = Field(default=4, ge=1, le=20)
    auto_approve: bool = False


class TaskCreateRequest(BaseModel):
    kind: str = Field(min_length=1, max_length=100)
    payload: Dict[str, Any] = {}
    run_at: Optional[float] = None
    priority: int = 0
    max_attempts: int = Field(default=3, ge=1, le=10)
    idempotency_key: Optional[str] = None


class InitiativeTickRequest(BaseModel):
    session_id: Optional[str] = None
    user_last_active: Optional[float] = None


class ReasonRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    expected: Optional[str] = None
    num_samples: int = Field(default=4, ge=1, le=8)
    max_new_tokens: int = Field(default=192, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    seed: int = 0
