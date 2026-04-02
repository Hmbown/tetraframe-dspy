"""Claude Code CLI backend adapter.

Invokes ``claude -p <prompt> --output-format json`` and parses the
structured JSON response.
"""
from __future__ import annotations

import json
from typing import Any

from tetraframe.backends.base import BackendCapabilities
from tetraframe.backends.cli_base import CLIBackendBase


# Claude Code bare aliases — don't pass --model for these
_BARE_ALIASES = frozenset({"claude-sonnet", "claude-opus", "claude-haiku", ""})

_KNOWN_MODELS = [
    "claude-sonnet",
    "claude-opus",
    "claude-haiku",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20250414",
]


class ClaudeCodeBackend(CLIBackendBase):
    """Invoke Claude via ``claude -p --output-format json``."""

    @property
    def provider_name(self) -> str:
        return "claude-code"

    @property
    def default_binary_name(self) -> str:
        return "claude"

    def _make_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=False,
            max_tokens=False,   # CLI does not enforce max_tokens
            temperature=False,  # CLI does not accept temperature
            structured_json=False,
            tool_use=False,
        )

    def _known_models(self) -> list[str]:
        return list(_KNOWN_MODELS)

    def _build_command(self, prompt: str, model: str, **kwargs: Any) -> list[str]:
        cmd: list[str] = [self._binary, "-p", prompt, "--output-format", "json"]
        if model and model not in _BARE_ALIASES:
            cmd += ["--model", model]
        cmd.extend(self._cli_args)
        return cmd

    # -- parsing --------------------------------------------------------------

    def _parse_output(self, stdout: str) -> str:
        return _parse_claude_stdout(stdout)

    def _parse_output_with_usage(self, stdout: str) -> tuple[str, dict[str, Any]]:
        return _parse_claude_stdout_with_usage(stdout)


# ---------------------------------------------------------------------------
# Output parsers (extracted from the original proxy/client.py)
# ---------------------------------------------------------------------------

def _parse_claude_stdout(raw: str) -> str:
    lines = raw.strip().splitlines()
    text_parts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                if event.get("type") == "error":
                    err = event.get("error", {})
                    msg = str(err.get("data", {}).get("message", err.get("name", "unknown")))
                    raise RuntimeError(f"ClaudeCLI API error: {msg}")
                if "result" in event:
                    return str(event["result"]).strip()
                part = event.get("part", {})
                if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                    text_parts.append(str(part["text"]))
                elif "content" in event:
                    text_parts.append(str(event["content"]))
                elif "text" in event:
                    text_parts.append(str(event["text"]))
        except json.JSONDecodeError:
            text_parts.append(line)
    if text_parts:
        return "\n".join(text_parts).strip()
    return raw.strip()


def _parse_claude_stdout_with_usage(raw: str) -> tuple[str, dict[str, Any]]:
    lines = raw.strip().splitlines()
    text_parts: list[str] = []
    usage: dict[str, Any] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error":
                err = event.get("error", {})
                msg = str(err.get("data", {}).get("message", err.get("name", "unknown")))
                raise RuntimeError(f"ClaudeCLI API error: {msg}")
            if "result" in event:
                text = str(event["result"]).strip()
                usage = _extract_usage(event)
                return text, usage
            part = event.get("part", {})
            if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                text_parts.append(str(part["text"]))
            elif "content" in event:
                text_parts.append(str(event["content"]))
            elif "text" in event:
                text_parts.append(str(event["text"]))
            if "modelUsage" in event:
                usage = _best_usage(event["modelUsage"])
        except json.JSONDecodeError:
            text_parts.append(line)

    text = "\n".join(text_parts).strip() if text_parts else raw.strip()
    return text, usage


def _extract_usage(event: dict[str, Any]) -> dict[str, Any]:
    if "modelUsage" in event:
        return _best_usage(event["modelUsage"])
    if "usage" in event:
        u = event["usage"]
        if isinstance(u, dict):
            return {
                "prompt_tokens": u.get("input_tokens", 0),
                "completion_tokens": u.get("output_tokens", 0),
                "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
            }
    return {}


def _best_usage(model_usage: dict[str, Any]) -> dict[str, Any]:
    for _name, info in model_usage.items():
        if not isinstance(info, dict):
            continue
        return {
            "prompt_tokens": info.get("inputTokens", 0),
            "completion_tokens": info.get("outputTokens", 0),
            "total_tokens": info.get("inputTokens", 0) + info.get("outputTokens", 0),
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
