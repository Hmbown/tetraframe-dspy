"""CLI tool — wraps an existing CLIBackendBase as a ModelTool."""
from __future__ import annotations

from typing import Any

from tetraframe.backends.cli_base import CLIBackendBase
from tetraframe.tools.protocol import CompletionResult, ToolInfo


class CLITool:
    """Wraps a CLIBackendBase (claude, codex, opencode) as a ModelTool."""

    def __init__(self, backend: CLIBackendBase, *, priority: int = 70) -> None:
        self._backend = backend
        self._priority = priority

    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            name=f"{self._backend.provider_name}-cli",
            provider=self._backend.provider_name,
            model=self._backend._model or "(default)",
            kind="cli",
            priority=self._priority,
            tags=("cli",),
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> list[CompletionResult]:
        results = []
        for _ in range(n):
            text, usage = self._backend.chat_with_usage(messages)
            results.append(CompletionResult(
                text=text,
                usage=usage,
                model=self._backend._model or "(default)",
            ))
        return results

    def is_available(self) -> bool:
        return self._backend.is_available()
