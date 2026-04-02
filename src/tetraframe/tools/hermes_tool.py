"""Hermes agent integration — uses Hermes credential pools and provider resolution.

If ~/.hermes/ exists, this tool leverages Hermes's multi-credential pooling
(round_robin, fill_first, least_used), OAuth token refresh, and model metadata
catalog instead of raw env vars.

This is the preferred tool when running under Hermes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tetraframe.tools.protocol import CompletionResult, ToolInfo
from tetraframe.tools.api_tool import _openai_chat

HERMES_DIR = Path.home() / ".hermes"
HERMES_AUTH = HERMES_DIR / "auth.json"
HERMES_ENV = HERMES_DIR / ".env"
HERMES_CONFIG = HERMES_DIR / "config.yaml"

# Known base URLs for providers that need them (no discovery possible without them).
_PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
}


def _load_env_file(path: Path) -> dict[str, str]:
    """Load key=value pairs from a .env-style file."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _pick_credential(pool: list[dict[str, Any]], strategy: str) -> dict[str, Any] | None:
    available = [c for c in pool if c.get("last_status") != "exhausted"] or pool
    if not available:
        return None
    if strategy == "fill_first":
        return min(available, key=lambda c: c.get("priority", 99))
    elif strategy == "round_robin":
        return min(available, key=lambda c: c.get("request_count", 0))
    elif strategy == "least_used":
        return min(available, key=lambda c: c.get("request_count", 0))
    elif strategy == "random":
        import random as _random
        return _random.choice(available)
    return available[0]


class HermesTool:
    """Model tool backed by Hermes credential pools and provider config.

    Reads auth.json, .env, and config.yaml to resolve credentials.
    Uses raw httpx for API calls — no litellm.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str | None = None,
        api_mode: str = "chat_completions",
        priority: int = 10,
        cost_tier: str = "standard",
    ) -> None:
        self._provider = provider
        self._model = model
        self._base_url = base_url
        self._api_mode = api_mode
        self._priority = priority
        self._cost_tier = cost_tier
        self._hermes_env = _load_env_file(HERMES_ENV)
        self._auth = _load_json(HERMES_AUTH)
        self._config = _load_yaml(HERMES_CONFIG)

    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            name=f"hermes-{self._provider}",
            provider=self._provider,
            model=self._model,
            kind="api",
            priority=self._priority,
            cost_tier=self._cost_tier,
            tags=("hermes",),
        )

    def _resolve_credential(self) -> str | None:
        """Resolve the best credential. Priority:
        1. Credential pool (with strategy)
        2. auth.json OAuth tokens
        3. Hermes .env or os.environ ({PROVIDER}_API_KEY)
        4. ANTHROPIC_TOKEN special case
        """
        # 1. Credential pool
        pool = self._auth.get("credential_pool", {}).get(self._provider, [])
        strategies = self._config.get("credential_pool_strategies", {})
        strategy = strategies.get(self._provider, "fill_first")
        if pool:
            cred = _pick_credential(pool, strategy)
            if cred and cred.get("token"):
                return cred["token"]

        # 2. auth.json provider tokens
        provider_auth = self._auth.get("providers", {}).get(self._provider, {})
        if provider_auth:
            oauth = provider_auth.get("tokens", {})
            if oauth.get("access_token"):
                return oauth["access_token"]
            if provider_auth.get("access_token"):
                return provider_auth["access_token"]

        # 3. .env / environ — convention: {PROVIDER}_API_KEY
        env_key = f"{self._provider.upper().replace('-', '_')}_API_KEY"
        val = self._hermes_env.get(env_key) or os.environ.get(env_key)
        if val:
            return val

        # 4. Anthropic OAuth token special case
        if self._provider == "anthropic":
            token = self._hermes_env.get("ANTHROPIC_TOKEN") or os.environ.get("ANTHROPIC_TOKEN")
            if token:
                return token

        return None

    def _resolve_base_url(self) -> str | None:
        if self._base_url:
            return self._base_url
        # Check credential pool
        pool = self._auth.get("credential_pool", {}).get(self._provider, [])
        for cred in pool:
            if cred.get("base_url"):
                return cred["base_url"]
        # Check auth.json
        provider_auth = self._auth.get("providers", {}).get(self._provider, {})
        if provider_auth.get("inference_base_url"):
            return provider_auth["inference_base_url"]
        # Known defaults
        return _PROVIDER_BASE_URLS.get(self._provider)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> list[CompletionResult]:
        api_key = self._resolve_credential()
        if not api_key:
            raise RuntimeError(
                f"No credential found for provider '{self._provider}' "
                f"in Hermes pools, auth.json, or environment"
            )

        base_url = self._resolve_base_url()
        if not base_url:
            raise RuntimeError(
                f"No base URL found for provider '{self._provider}'. "
                f"Set it in config or credential pool."
            )

        data = _openai_chat(
            base_url=base_url,
            model=self._model,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )

        results = []
        for choice in data.get("choices", []):
            usage = data.get("usage") or {}
            results.append(CompletionResult(
                text=choice.get("message", {}).get("content", ""),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                raw=data,
                model=data.get("model", self._model),
            ))
        return results

    def is_available(self) -> bool:
        return self._resolve_credential() is not None


def discover_hermes_tools() -> list[HermesTool]:
    """Discover available tools from Hermes configuration."""
    if not HERMES_DIR.exists():
        return []

    tools: list[HermesTool] = []

    # Default model from config
    default_model = _load_yaml(HERMES_CONFIG).get("model", {})
    if default_model.get("default") and default_model.get("provider"):
        tools.append(HermesTool(
            provider=default_model["provider"],
            model=default_model["default"],
            base_url=default_model.get("base_url"),
            priority=5,
        ))

    # Credential pool providers
    auth = _load_json(HERMES_AUTH)
    pools = auth.get("credential_pool", {})
    for provider, creds in pools.items():
        if creds:
            # Pick a sensible default model per provider
            model = _default_model_for_provider(provider)
            tools.append(HermesTool(
                provider=provider,
                model=model,
                priority=15,
            ))

    return tools


def _default_model_for_provider(provider: str) -> str:
    """Best-guess default model for a known provider."""
    defaults = {
        "openai": "gpt-4.1-mini",
        "anthropic": "claude-sonnet-4-6",
        "openrouter": "anthropic/claude-sonnet-4.6",
    }
    return defaults.get(provider, "default")
