"""Backend protocol and shared types for model invocation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BackendCapabilities:
    """Declares what a backend actually supports.

    Any capability set to False means TetraFrame will not attempt that
    feature and will emit an explicit warning when the caller requests it.
    """

    streaming: bool = False
    max_tokens: bool = True
    temperature: bool = True
    structured_json: bool = False
    tool_use: bool = False


@dataclass
class BackendMetadata:
    """Identity and capability snapshot for traces and health endpoints."""

    name: str
    kind: str  # "api" or "cli"
    provider: str
    model: str
    capabilities: BackendCapabilities
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "provider": self.provider,
            "model": self.model,
            "capabilities": {
                "streaming": self.capabilities.streaming,
                "max_tokens": self.capabilities.max_tokens,
                "temperature": self.capabilities.temperature,
                "structured_json": self.capabilities.structured_json,
                "tool_use": self.capabilities.tool_use,
            },
            "warnings": self.warnings,
        }


@runtime_checkable
class Backend(Protocol):
    """Minimal contract for a model invocation target.

    Implemented by API backends (OpenAI, Anthropic, etc.) and CLI backends
    (Claude Code, Codex, OpenCode).
    """

    @property
    def metadata(self) -> BackendMetadata: ...

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Send messages and return the assistant's text."""
        ...

    def chat_with_usage(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        """Send messages and return (text, usage_dict)."""
        ...

    def is_available(self) -> bool: ...

    def list_models(self) -> list[str]: ...
