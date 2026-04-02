"""DSPy-compatible language model wrapper for CLI backends.

Makes CLI backends (Claude Code, Codex, OpenCode) usable as the ``lm``
argument to ``dspy.configure(lm=...)``.  This lets the full TetraFrame
pipeline run against CLI backends without requiring the proxy.

The adapter translates DSPy's internal message-based calling convention
into prompt strings, invokes the CLI backend, and returns responses in
the format DSPy expects.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from tetraframe.backends.cli_base import CLIBackendBase

try:
    from dspy import BaseLM as _BaseLM
except ImportError:
    _BaseLM = None  # type: ignore[assignment,misc]


class _DictNamespace(SimpleNamespace):
    """SimpleNamespace that also supports ``dict(obj)`` iteration."""

    def __iter__(self):
        return iter(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()


def _make_mock_response(text: str) -> SimpleNamespace:
    """Build an OpenAI-compatible response object from plain text."""
    message = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    message.provider_specific_fields = {}
    choice = SimpleNamespace(message=message, logprobs=None)
    usage = _DictNamespace(
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model="cli")


# Resolve base class: use dspy.BaseLM when available, else plain object
_Base: type = _BaseLM if _BaseLM is not None else object


class CLILanguageModel(_Base):  # type: ignore[misc]
    """A DSPy-LM-compatible wrapper around a CLIBackendBase.

    Extends ``dspy.BaseLM`` (when DSPy >= 3.x is installed) so that
    ``dspy.configure(lm=...)`` type-checks pass.  The ``forward`` method
    returns an OpenAI-compatible mock response that ``BaseLM.__call__``
    knows how to process.
    """

    def __init__(self, backend: CLIBackendBase, **defaults: Any) -> None:
        model_name = f"cli/{backend.provider_name}/{backend._model or 'default'}"
        if _BaseLM is not None:
            super().__init__(
                model=model_name,
                model_type="chat",
                temperature=0.0,
                max_tokens=4096,
                cache=False,
            )
        else:
            self.model = model_name
            self.kwargs = {}
            self.history = []
        self._backend = backend
        self._defaults = defaults

    # -- dspy.BaseLM interface ---------------------------------------------------

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        merged = {**self._defaults, **kwargs}
        if messages:
            effective_messages = messages
        elif prompt:
            effective_messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("CLILanguageModel requires either prompt or messages")

        n = merged.pop("n", 1)
        choices = []
        for _ in range(n):
            text = self._backend.chat(effective_messages, **merged)
            message = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
            message.provider_specific_fields = {}
            choices.append(SimpleNamespace(message=message, logprobs=None))

        usage = _DictNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return SimpleNamespace(choices=choices, usage=usage, model="cli")

    # -- legacy interface (kept for direct callers) ------------------------------

    def inspect_history(self, n: int = 1) -> list[dict[str, Any]]:
        return self.history[-n:]

    def __repr__(self) -> str:
        return f"CLILanguageModel(backend={self._backend.provider_name}, model={self._backend._model})"
