"""Codex CLI backend adapter.

Codex (OpenAI's CLI agent) can be invoked via ``codex exec`` with a prompt.
This adapter wraps that interface.  If the Codex CLI is not installed or its
interface changes, the adapter fails loudly at construction time.

Assumptions documented here:
- ``codex exec -p "<prompt>"`` sends a one-shot prompt and prints the response
  to stdout.
- ``--model`` selects the model (e.g. ``o4-mini``, ``gpt-4.1``).
- Exit code 0 means success.
- No structured JSON output mode exists; we capture raw stdout.
"""
from __future__ import annotations

from typing import Any

from tetraframe.backends.base import BackendCapabilities
from tetraframe.backends.cli_base import CLIBackendBase


_KNOWN_MODELS = [
    "o4-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]


class CodexBackend(CLIBackendBase):
    """Invoke Codex CLI via ``codex exec -p <prompt>``."""

    @property
    def provider_name(self) -> str:
        return "codex"

    @property
    def default_binary_name(self) -> str:
        return "codex"

    def _make_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=False,
            max_tokens=False,
            temperature=False,
            structured_json=False,
            tool_use=False,
        )

    def _known_models(self) -> list[str]:
        return list(_KNOWN_MODELS)

    def _build_command(self, prompt: str, model: str, **kwargs: Any) -> list[str]:
        cmd: list[str] = [self._binary, "exec", "-p", prompt]
        if model:
            cmd += ["--model", model]
        cmd.extend(self._cli_args)
        return cmd

    def _parse_output(self, stdout: str) -> str:
        # Codex exec prints raw text to stdout.
        return stdout.strip()

    def _parse_output_with_usage(self, stdout: str) -> tuple[str, dict[str, Any]]:
        # No usage reporting from Codex CLI.
        return stdout.strip(), {}
