"""OpenCode CLI backend adapter.

OpenCode is a Go-based AI coding assistant.  This adapter wraps its
non-interactive prompt mode.

Assumptions documented here:
- ``opencode ask "<prompt>"`` sends a one-shot prompt and prints the response.
- ``--model`` selects the model.
- Exit code 0 means success.
- No structured JSON output mode; raw stdout is captured.

If these assumptions are wrong for your version of OpenCode, adjust the
``_build_command`` and ``_parse_output`` methods or configure ``cli_args``
in the backend config.
"""
from __future__ import annotations

import json
from typing import Any

from tetraframe.backends.base import BackendCapabilities
from tetraframe.backends.cli_base import CLIBackendBase


class OpenCodeBackend(CLIBackendBase):
    """Invoke OpenCode CLI via ``opencode ask <prompt>``."""

    @property
    def provider_name(self) -> str:
        return "opencode"

    @property
    def default_binary_name(self) -> str:
        return "opencode"

    def _make_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=False,
            max_tokens=False,
            temperature=False,
            structured_json=False,
            tool_use=False,
        )

    def _known_models(self) -> list[str]:
        # OpenCode model list depends on its own config; we don't enumerate.
        return []

    def _build_command(self, prompt: str, model: str, **kwargs: Any) -> list[str]:
        cmd: list[str] = [self._binary, "run", prompt, "--format", "json"]
        if model:
            cmd += ["-m", model]
        cmd.extend(self._cli_args)
        return cmd

    def _parse_output(self, stdout: str) -> str:
        """Extract text from opencode's JSONL output."""
        text_parts: list[str] = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "text":
                part = event.get("part", {})
                txt = part.get("text", "")
                if txt:
                    text_parts.append(txt)
        if not text_parts:
            # fallback: return raw stdout if no JSON text events found
            return stdout.strip()
        return "\n".join(text_parts)

    def _parse_output_with_usage(self, stdout: str) -> tuple[str, dict[str, Any]]:
        text = self._parse_output(stdout)
        usage: dict[str, Any] = {}
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "step_finish":
                tokens = event.get("part", {}).get("tokens", {})
                cost = event.get("part", {}).get("cost", 0)
                usage = {
                    "prompt_tokens": tokens.get("input", 0),
                    "completion_tokens": tokens.get("output", 0),
                    "total_tokens": tokens.get("total", 0),
                    "cost": cost,
                }
                break
        return text, usage
