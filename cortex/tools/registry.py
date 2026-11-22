"""Typed tool registry used by the coding agent and the API."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ToolError(RuntimeError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: Dict[str, str] = field(default_factory=dict)
    dangerous: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "dangerous": self.dangerous,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, str]] = None,
        dangerous: bool = False,
    ):
        def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                raise ValueError(f"tool already registered: {name}")
            self._tools[name] = Tool(
                name=name,
                description=description,
                handler=handler,
                parameters=parameters or {},
                dangerous=dangerous,
            )
            return handler

        return decorator

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name!r}")
        return self._tools[name]

    def list(self) -> List[dict]:
        return [tool.to_dict() for tool in self._tools.values()]

    def execute(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        tool = self.get(name)
        arguments = arguments or {}
        signature = inspect.signature(tool.handler)
        accepted = set(signature.parameters)
        unknown = set(arguments) - accepted
        if unknown:
            raise ToolError(f"unexpected arguments for {name}: {sorted(unknown)}")
        try:
            return tool.handler(**arguments)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{name} failed: {type(exc).__name__}: {exc}") from exc
