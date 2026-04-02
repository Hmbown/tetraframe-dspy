"""ModelTool protocol — the standard interface any model provider implements.

Auth is the tool's problem. TetraFrame never sees API keys, OAuth tokens,
or CLI session state. It just calls tool.complete() and gets text back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolInfo:
    """Identity card for a ModelTool."""

    name: str
    provider: str
    model: str
    kind: str  # "api", "cli", "mcp"
    priority: int = 50  # lower = preferred (0-100)
    cost_tier: str = "unknown"  # "free", "cheap", "standard", "expensive"
    tags: tuple[str, ...] = ()


@dataclass
class CompletionResult:
    """What a tool returns from complete()."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None
    model: str = ""


@runtime_checkable
class ModelTool(Protocol):
    """Standard interface any model provider implements."""

    @property
    def info(self) -> ToolInfo: ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> list[CompletionResult]: ...

    def is_available(self) -> bool: ...
