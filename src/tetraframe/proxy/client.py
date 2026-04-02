"""Claude CLI backend adapter for TetraFrame.

This module is a backward-compatible shim.  The real implementation lives
in ``tetraframe.backends.cli_claude``.  Import paths used by existing code
and tests continue to work.
"""

from __future__ import annotations

from tetraframe.backends.cli_claude import ClaudeCodeBackend, _KNOWN_MODELS


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

def detect_claude_cli() -> str | None:
    """Return the path to the Claude CLI if on PATH."""
    import shutil
    return shutil.which("claude")


def list_claude_models() -> list[str]:
    return list(_KNOWN_MODELS)


class ClaudeCLIBackend:
    """Backward-compatible wrapper around ClaudeCodeBackend.

    Preserves the old ``invoke`` / ``invoke_with_usage`` API used by the
    proxy server and existing tests.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 180.0,
        claude_bin: str | None = None,
    ) -> None:
        self._inner = ClaudeCodeBackend(
            binary=claude_bin,
            model=model,
            timeout=timeout,
        )
        self._model_passthrough = model or ""

    def invoke(self, prompt: str, *, config: dict | None = None) -> str:
        cfg = dict(config or {})
        model = cfg.get("model", self._model_passthrough)
        messages = [{"role": "user", "content": prompt}]
        return self._inner.chat(messages, model=model)

    def invoke_with_usage(
        self, prompt: str, *, config: dict | None = None
    ) -> tuple[str, dict]:
        cfg = dict(config or {})
        model = cfg.get("model", self._model_passthrough)
        messages = [{"role": "user", "content": prompt}]
        return self._inner.chat_with_usage(messages, model=model)

    def is_available(self) -> bool:
        return self._inner.is_available()

    @property
    def _claude_bin(self) -> str | None:
        return self._inner._binary
