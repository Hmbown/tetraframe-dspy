"""API backend for OpenAI-compatible HTTP providers.

This backend validates configuration and constructs the parameters needed
for ``dspy.LM()``.  The ``chat_with_usage()`` method uses raw httpx
for direct HTTP calls — no litellm.
"""
from __future__ import annotations

import os
from typing import Any

from tetraframe.backends.base import Backend, BackendCapabilities, BackendMetadata
from tetraframe.config import BackendConfig


# Default capabilities for any OpenAI-compatible API.
_DEFAULT_API_CAPS = BackendCapabilities(
    streaming=True, max_tokens=True, temperature=True,
    structured_json=False, tool_use=False,
)


class APIBackend:
    """Backend wrapping an OpenAI-compatible API provider.

    Primary use: construct a properly configured ``dspy.LM`` via
    ``build_dspy_lm_kwargs()``.  Also usable directly through
    ``chat()`` for proxy forwarding (requires ``httpx``).
    """

    def __init__(self, cfg: BackendConfig) -> None:
        self._cfg = cfg
        self._validate()

    def _validate(self) -> None:
        if not self._cfg.model and not self._cfg.dspy_model_string():
            raise ValueError(
                f"API backend for provider '{self._cfg.provider}' requires "
                f"a model name in backend config"
            )
        if self._cfg.api_key_env:
            key = os.environ.get(self._cfg.api_key_env, "")
            if not key:
                raise ValueError(
                    f"API backend for provider '{self._cfg.provider}' requires "
                    f"env var {self._cfg.api_key_env} to be set"
                )

    @property
    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name=f"{self._cfg.provider}-api",
            kind="api",
            provider=self._cfg.provider,
            model=self._cfg.model,
            capabilities=_DEFAULT_API_CAPS,
        )

    def build_dspy_lm_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``dspy.LM(model, **kwargs)``."""
        model_str = self._cfg.dspy_model_string()
        kwargs: dict[str, Any] = {}
        if self._cfg.temperature is not None:
            kwargs["temperature"] = self._cfg.temperature
        if self._cfg.max_tokens is not None:
            kwargs["max_tokens"] = self._cfg.max_tokens
        if self._cfg.base_url:
            kwargs["api_base"] = self._cfg.base_url
        if self._cfg.api_key_env:
            key = os.environ.get(self._cfg.api_key_env, "")
            if key:
                kwargs["api_key"] = key
        return {"model": model_str, **kwargs}

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Direct HTTP call via httpx (for proxy use)."""
        text, _ = self.chat_with_usage(messages, **kwargs)
        return text

    def chat_with_usage(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        """Direct HTTP call returning (text, usage_dict)."""
        import httpx

        base_url = self._cfg.base_url
        if not base_url:
            raise RuntimeError("API backend requires base_url for direct calls")

        api_key = os.environ.get(self._cfg.api_key_env, "") if self._cfg.api_key_env else ""
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
        }
        temperature = kwargs.get("temperature", self._cfg.temperature)
        if temperature is not None:
            body["temperature"] = temperature
        max_tokens = kwargs.get("max_tokens", self._cfg.max_tokens)
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        resp = httpx.post(url, json=body, headers=headers, timeout=self._cfg.timeout)
        resp.raise_for_status()
        data = resp.json()

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        return text, {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def is_available(self) -> bool:
        if self._cfg.api_key_env:
            return bool(os.environ.get(self._cfg.api_key_env))
        return True

    def list_models(self) -> list[str]:
        return [self._cfg.model] if self._cfg.model else []
