"""Small, explicit tool registry used by agents and the API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from cortex.reasoning.verifiers import safe_eval


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    func: Callable[..., Any]
    requires_approval: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}")
        return self._tools[name]

    def list(self) -> list[dict]:
        return [t.as_dict() for t in self._tools.values()]

    def call(self, name: str, **kwargs: Any) -> Any:
        return self.get(name).func(**kwargs)


def default_registry(memory=None) -> ToolRegistry:
    reg = ToolRegistry()

    def calculator(expression: str) -> dict:
        value = safe_eval(expression)
        if value is None:
            return {"ok": False, "error": "invalid or unsafe expression"}
        return {"ok": True, "value": value}

    reg.register(
        Tool(
            "calculator",
            "Evaluate a basic arithmetic expression (+, -, *, /, ^).",
            {"expression": "string"},
            calculator,
        )
    )

    if memory is not None:

        def memory_search(query: str, limit: int = 5) -> dict:
            eps = memory.search_episodes(query, limit=limit)
            return {"ok": True, "results": [e.as_dict() for e in eps]}

        def memory_write(subject: str, content: str) -> dict:
            fact_id = memory.add_fact(subject, content, source="tool")
            return {"ok": True, "fact_id": fact_id}

        reg.register(
            Tool(
                "memory_search",
                "Search episodic memory.",
                {"query": "string", "limit": "integer"},
                memory_search,
            )
        )
        reg.register(
            Tool(
                "memory_write",
                "Store a semantic fact.",
                {"subject": "string", "content": "string"},
                memory_write,
            )
        )

    return reg
