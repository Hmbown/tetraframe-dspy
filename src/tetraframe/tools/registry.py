"""Tool registry — discovers and manages available ModelTools."""
from __future__ import annotations

import os
import shutil
from typing import Any

from tetraframe.tools.protocol import ModelTool, ToolInfo


class ToolRegistry:
    """Discovers and manages available ModelTools."""

    def __init__(self) -> None:
        self._tools: dict[str, ModelTool] = {}

    def register(self, tool: ModelTool) -> None:
        self._tools[tool.info.name] = tool

    def get(self, name: str) -> ModelTool | None:
        return self._tools.get(name)

    def best_available(self, *, tags: tuple[str, ...] = ()) -> ModelTool | None:
        candidates = [t for t in self._tools.values() if t.is_available()]
        if tags:
            candidates = [t for t in candidates if set(tags) <= set(t.info.tags)]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t.info.priority)

    def all_available(self) -> list[ModelTool]:
        return sorted(
            [t for t in self._tools.values() if t.is_available()],
            key=lambda t: t.info.priority,
        )

    def summary(self) -> list[dict[str, Any]]:
        entries = []
        for tool in sorted(self._tools.values(), key=lambda t: t.info.priority):
            info = tool.info
            entries.append({
                "name": info.name,
                "provider": info.provider,
                "model": info.model,
                "kind": info.kind,
                "priority": info.priority,
                "cost_tier": info.cost_tier,
                "available": tool.is_available(),
            })
        return entries


# ---------------------------------------------------------------------------
# Auto-discovery probes
# ---------------------------------------------------------------------------

# Known base URLs — providers without one can't be auto-discovered.
_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

# Default API probes — checked when no config specifies a backend.
# Convention: {PROVIDER}_API_KEY for the env var.
# Add more providers by dropping a YAML config in configs/ — no code changes needed.
_API_PROBES: list[dict[str, Any]] = [
    {
        "name": "openai-api",
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
        "priority": 30,
        "cost_tier": "standard",
    },
    {
        "name": "anthropic-api",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "priority": 30,
        "cost_tier": "standard",
    },
]

_CLI_PROBES: list[dict[str, Any]] = [
    {"provider": "claude-code", "binary": "claude", "priority": 60},
    {"provider": "opencode", "binary": "opencode", "priority": 65},
    {"provider": "codex", "binary": "codex", "priority": 70},
]


def auto_discover() -> ToolRegistry:
    """Probe the environment and return a registry with all found tools.

    Discovery order:
    1. Hermes agent credential pools (highest priority, richest auth)
    2. Direct API key env vars
    3. CLI tools on PATH
    """
    from tetraframe.tools.api_tool import DirectAPITool

    registry = ToolRegistry()

    # 1. Hermes integration (credential pools, OAuth, multi-strategy rotation)
    try:
        from tetraframe.tools.hermes_tool import discover_hermes_tools
        for tool in discover_hermes_tools():
            registry.register(tool)
    except Exception:
        pass  # Hermes not available or broken

    # 2. Direct API probes (fallback for non-Hermes environments)
    for probe in _API_PROBES:
        tool = DirectAPITool(
            name=probe["name"],
            provider=probe["provider"],
            model=probe["model"],
            base_url=probe.get("base_url", _PROVIDER_BASE_URLS.get(probe["provider"], "")),
            api_key_env=probe["api_key_env"],
            priority=probe["priority"],
            cost_tier=probe.get("cost_tier", "standard"),
        )
        registry.register(tool)

    # 3. CLI probes
    for probe in _CLI_PROBES:
        binary = shutil.which(probe["binary"])
        if binary:
            try:
                from tetraframe.tools.cli_tool import CLITool
                from tetraframe.backends.factory import _build_cli_backend
                from tetraframe.config import BackendConfig

                cfg = BackendConfig(provider=probe["provider"], binary=binary)
                backend = _build_cli_backend(cfg)
                tool = CLITool(backend, priority=probe["priority"])
                registry.register(tool)
            except Exception:
                pass  # CLI not functional, skip

    return registry
