"""Base class for CLI backend adapters.

Shared subprocess invocation, timeout handling, and output parsing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Any

from tetraframe.backends.base import Backend, BackendCapabilities, BackendMetadata


def _assemble_prompt(messages: list[dict[str, Any]]) -> str:
    """Convert OpenAI-style messages into a single prompt string for CLI tools."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        if role == "system":
            parts.append(f"<system>\n{content}\n</system>")
        elif role == "assistant":
            parts.append(f"<assistant>\n{content}\n</assistant>")
        else:
            parts.append(content)
    return "\n\n".join(parts)


class CLIBackendBase(ABC):
    """Abstract base for subprocess-based CLI backends.

    Subclasses implement ``_build_command``, ``_parse_output``, and
    ``_parse_output_with_usage`` for their specific CLI tool.
    """

    def __init__(
        self,
        *,
        binary: str | None = None,
        model: str | None = None,
        timeout: float = 180.0,
        cli_args: list[str] | None = None,
        env_passthrough: list[str] | None = None,
    ) -> None:
        resolved = binary or self._detect_binary()
        if not resolved:
            raise RuntimeError(
                f"{self.provider_name} CLI not found on PATH. "
                f"Expected binary: {self.default_binary_name}"
            )
        self._binary = resolved
        self._model = model or ""
        self.timeout = timeout
        self._cli_args = list(cli_args or [])
        self._env_passthrough = list(env_passthrough or [])

    # -- subclass hooks -------------------------------------------------------

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def default_binary_name(self) -> str: ...

    @abstractmethod
    def _build_command(self, prompt: str, model: str, **kwargs: Any) -> list[str]: ...

    @abstractmethod
    def _parse_output(self, stdout: str) -> str: ...

    @abstractmethod
    def _parse_output_with_usage(self, stdout: str) -> tuple[str, dict[str, Any]]: ...

    @abstractmethod
    def _known_models(self) -> list[str]: ...

    def _make_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=False,
            max_tokens=False,
            temperature=False,
            structured_json=False,
            tool_use=False,
        )

    # -- detection ------------------------------------------------------------

    def _detect_binary(self) -> str | None:
        return shutil.which(self.default_binary_name)

    # -- Backend protocol -----------------------------------------------------

    @property
    def metadata(self) -> BackendMetadata:
        warnings: list[str] = []
        caps = self._make_capabilities()
        if not caps.max_tokens:
            warnings.append("max_tokens is not enforced by this CLI backend")
        if not caps.temperature:
            warnings.append("temperature is not enforced by this CLI backend")
        return BackendMetadata(
            name=f"{self.provider_name}-cli",
            kind="cli",
            provider=self.provider_name,
            model=self._model or "(default)",
            capabilities=caps,
            warnings=warnings,
        )

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        prompt = _assemble_prompt(messages)
        model = kwargs.get("model", self._model)
        return self._invoke(prompt, model, **kwargs)

    def chat_with_usage(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        prompt = _assemble_prompt(messages)
        model = kwargs.get("model", self._model)
        return self._invoke_with_usage(prompt, model, **kwargs)

    def is_available(self) -> bool:
        return self._binary is not None

    def list_models(self) -> list[str]:
        return self._known_models()

    # -- invocation -----------------------------------------------------------

    def _invoke(self, prompt: str, model: str, **kwargs: Any) -> str:
        cmd = self._build_command(prompt, model, **kwargs)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{self.provider_name} CLI failed (exit {result.returncode}): "
                f"{result.stderr.strip()[:500]}"
            )
        return self._parse_output(result.stdout)

    def _invoke_with_usage(
        self, prompt: str, model: str, **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        cmd = self._build_command(prompt, model, **kwargs)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{self.provider_name} CLI failed (exit {result.returncode}): "
                f"{result.stderr.strip()[:500]}"
            )
        return self._parse_output_with_usage(result.stdout)
