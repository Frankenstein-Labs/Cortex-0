"""Mission-driven coding agent with approval gates and persistent state.

The agent separates *policy* (what to do next) from *tools* (how to do it).
``ScriptedPolicy`` is deterministic and used in tests; ``ModelPolicy`` asks the
local language model for one JSON action at a time.  The tiny model is not a
reliable coder yet, so the default plan is conservative and every dangerous
action requires explicit approval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import torch

from ..agents.workspace import Workspace
from ..model.generation import generate
from ..model.transformer import CortexForCausalLM
from ..planning.missions import Mission, MissionStore
from ..tokenizer.bpe import CortexTokenizer
from ..tools.registry import ToolError, ToolRegistry
from .tools import build_workspace_registry

JSON_PATTERN = re.compile(r"\{.*\}", re.S)


@dataclass
class CodingAction:
    tool: str
    args: Dict[str, object]
    rationale: str = ""

    def to_dict(self) -> dict:
        return {"tool": self.tool, "args": self.args, "rationale": self.rationale}


class Policy(Protocol):
    def next_action(self, mission: Mission, last_result: Optional[dict]) -> Optional[CodingAction]:
        ...


class ScriptedPolicy:
    """Deterministic policy for tests and dry runs."""

    def __init__(self, actions: List[CodingAction]) -> None:
        self.actions = list(actions)
        self.index = 0

    def next_action(self, mission: Mission, last_result: Optional[dict]) -> Optional[CodingAction]:
        if self.index >= len(self.actions):
            return None
        action = self.actions[self.index]
        self.index += 1
        return action


class ModelPolicy:
    def __init__(
        self,
        model: CortexForCausalLM,
        tokenizer: CortexTokenizer,
        registry: ToolRegistry,
        device: str | torch.device = "cpu",
        max_new_tokens: int = 192,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.device = torch.device(device)
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.last_error: Optional[str] = None

    def _prompt(self, mission: Mission, last_result: Optional[dict]) -> str:
        tools = "\n".join(
            f"- {tool['name']}({', '.join(tool['parameters'])}) : {tool['description']}"
            for tool in self.registry.list()
        )
        plan = "\n".join(
            f"{step['id']}. [{step['status']}] {step['title']}" for step in mission.plan
        )
        return (
            "Tu es un agent de développement. Choisis UNE action.\n"
            f"Mission : {mission.instruction}\nPlan :\n{plan}\n"
            f"Dernier résultat : {json.dumps(last_result or {}, ensure_ascii=False)[:800]}\n"
            f"Outils :\n{tools}\n"
            'Réponds uniquement par un objet JSON : {"tool": "...", "args": {...}, "rationale": "..."}\n'
        )

    def next_action(self, mission: Mission, last_result: Optional[dict]) -> Optional[CodingAction]:
        prompt = self._prompt(mission, last_result)
        ids = self.tokenizer.encode(prompt)
        max_context = self.model.config.max_position_embeddings
        ids = ids[-(max_context - self.max_new_tokens) :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        output = generate(
            self.model,
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_token_ids=[self.tokenizer.end_token_id],
            generator=generator,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(output[0, len(ids) :].tolist(), skip_special_tokens=True)
        return self.parse(text)

    def parse(self, text: str) -> Optional[CodingAction]:
        match = JSON_PATTERN.search(text)
        if not match:
            self.last_error = "no JSON object in model output"
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            self.last_error = f"invalid JSON: {exc}"
            return None
        tool = payload.get("tool") or payload.get("action")
        args = payload.get("args") or payload.get("arguments") or {}
        if not isinstance(tool, str) or not isinstance(args, dict):
            self.last_error = "tool must be a string and args an object"
            return None
        try:
            self.registry.get(tool)
        except ToolError:
            self.last_error = f"unknown tool: {tool}"
            return None
        return CodingAction(tool=tool, args=args, rationale=str(payload.get("rationale", "")))


class CodingAgent:
    def __init__(
        self,
        workspace: Workspace,
        missions: MissionStore,
        policy: Policy,
        registry: Optional[ToolRegistry] = None,
        auto_approve: bool = False,
    ) -> None:
        self.workspace = workspace
        self.missions = missions
        self.policy = policy
        self.registry = registry or build_workspace_registry(workspace)
        self.auto_approve = auto_approve
        self._pending: Dict[str, CodingAction] = {}
        self._last_result: Dict[str, dict] = {}

    def start_mission(self, title: str, instruction: str) -> Mission:
        return self.missions.create(title, instruction)

    def run_step(self, mission_id: str) -> dict:
        mission = self.missions.get(mission_id)
        if mission is None:
            raise KeyError(mission_id)
        if mission.status != "active":
            return {"status": mission.status, "mission_id": mission_id}

        step = mission.next_step()
        if step is None:
            self.missions.set_status(mission_id, "completed")
            return {"status": "completed", "mission_id": mission_id}

        if step.status != "in_progress":
            mission = self.missions.update_step(mission_id, step.id, "in_progress")
        last_result = self._last_result.get(mission_id) or self._last_step_result(mission)
        action = self.policy.next_action(mission, last_result)
        if action is None:
            error = getattr(self.policy, "last_error", None)
            self.missions.update_step(
                mission_id,
                step.id,
                "done",
                result=json.dumps({"note": "no action proposed", "error": error}, ensure_ascii=False),
            )
            return {
                "status": "done",
                "mission_id": mission_id,
                "step": step.to_dict(),
                "note": "no action proposed",
                "error": error,
            }

        tool = self.registry.get(action.tool)
        if tool.dangerous and not self.auto_approve:
            self._pending[mission_id] = action
            self.missions.checkpoint(
                mission_id, "pending-action", {"action": action.to_dict()}
            )
            return {
                "status": "awaiting_approval",
                "mission_id": mission_id,
                "step": step.to_dict(),
                "action": action.to_dict(),
            }
        return self._execute(mission_id, step.id, action)

    def approve(self, mission_id: str) -> dict:
        action = self._pending.pop(mission_id, None) or self._load_pending(mission_id)
        if action is None:
            return {"status": "no_pending_action", "mission_id": mission_id}
        self.missions.checkpoint(mission_id, "pending-action-cleared", {})
        mission = self.missions.get(mission_id)
        step = mission.next_step()
        if step is None:
            return {"status": "completed", "mission_id": mission_id}
        return self._execute(mission_id, step.id, action)

    def reject(self, mission_id: str, reason: str = "rejected by user") -> dict:
        action = self._pending.pop(mission_id, None)
        mission = self.missions.get(mission_id)
        step = mission.next_step()
        if step is not None:
            self.missions.update_step(mission_id, step.id, "skipped", notes=reason)
        return {
            "status": "rejected",
            "mission_id": mission_id,
            "action": action.to_dict() if action else None,
            "reason": reason,
        }

    def run(self, mission_id: str, max_steps: int = 8) -> dict:
        history = []
        for _ in range(max_steps):
            result = self.run_step(mission_id)
            history.append(result)
            if result["status"] in {"awaiting_approval", "completed", "cancelled", "failed", "blocked"}:
                break
        return {"mission_id": mission_id, "status": history[-1]["status"] if history else "noop", "history": history}

    def add_instruction(self, mission_id: str, text: str) -> Mission:
        return self.missions.add_instruction(mission_id, text)

    @staticmethod
    def _last_step_result(mission: Mission) -> Optional[dict]:
        for step in reversed(mission.plan):
            if step.get("result"):
                try:
                    return json.loads(step["result"])
                except (TypeError, json.JSONDecodeError):
                    return {"raw": step["result"]}
        return None

    def _load_pending(self, mission_id: str) -> Optional[CodingAction]:
        mission = self.missions.get(mission_id)
        if mission is None:
            return None
        for checkpoint in reversed(mission.checkpoints):
            if checkpoint.get("label") == "pending-action-cleared":
                return None
            if checkpoint.get("label") == "pending-action":
                payload = checkpoint.get("state", {}).get("action")
                if payload:
                    return CodingAction(
                        tool=payload["tool"],
                        args=payload.get("args", {}),
                        rationale=payload.get("rationale", ""),
                    )
        return None

    def _execute(self, mission_id: str, step_id: int, action: CodingAction) -> dict:
        try:
            result = self.registry.execute(action.tool, action.args)
            ok = bool(result.get("ok", True)) if isinstance(result, dict) else True
            status = "done" if ok else "failed"
        except ToolError as exc:
            result = {"error": str(exc)}
            status = "failed"
        self._last_result[mission_id] = result if isinstance(result, dict) else {"value": result}
        self.missions.update_step(
            mission_id,
            step_id,
            status,
            result=json.dumps(result, ensure_ascii=False)[:3000],
        )
        self.missions.checkpoint(
            mission_id, f"step-{step_id}-{status}", {"action": action.to_dict(), "result": result}
        )
        return {
            "status": status,
            "mission_id": mission_id,
            "step_id": step_id,
            "action": action.to_dict(),
            "result": result,
        }
