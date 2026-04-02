"""API backend for OpenAI-compatible HTTP providers.

This backend validates configuration and constructs the parameters needed
for ``dspy.LM()``.  It does not duplicate HTTP logic — litellm (inside
DSPy) handles the actual API calls.  The backend also implements the
``Backend`` protocol so the proxy can use it for direct HTTP forwarding.
"""
from __future__ import annotations

import os
from typing import Any

from tetraframe.backends.base import Backend, BackendCapabilities, BackendMetadata
from tetraframe.config import BackendConfig


# Provider → capability map
_PROVIDER_CAPS: dict[str, BackendCapabilities] = {
    "openai": BackendCapabilities(
        streaming=True, max_tokens=True, temperature=True,
        structured_json=True, tool_use=True,
    ),
    "anthropic": BackendCapabilities(
        streaming=True, max_tokens=True, temperature=True,
        structured_json=True, tool_use=True,
    ),
    "openrouter": BackendCapabilities(
        streaming=True, max_tokens=True, temperature=True,
        structured_json=False, tool_use=False,
    ),
    "openai-compatible": BackendCapabilities(
        streaming=True, max_tokens=True, temperature=True,
        structured_json=False, tool_use=False,
    ),
}


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
        caps = _PROVIDER_CAPS.get(self._cfg.provider, BackendCapabilities())
        return BackendMetadata(
            name=f"{self._cfg.provider}-api",
            kind="api",
            provider=self._cfg.provider,
            model=self._cfg.model,
            capabilities=caps,
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
        """Direct HTTP call returning (text, usage_dict).

        Uses litellm.completion for consistency with DSPy's provider support.
        """
        try:
            import litellm  # type: ignore
        except ImportError:
            raise RuntimeError(
                "litellm is required for direct API backend calls. "
                "Install dspy (which includes litellm) or install litellm directly."
            )
        model_str = self._cfg.dspy_model_string()
        call_kwargs: dict[str, Any] = {
            "model": model_str,
            "messages": messages,
        }
        if self._cfg.base_url:
            call_kwargs["api_base"] = self._cfg.base_url
        if self._cfg.api_key_env:
            key = os.environ.get(self._cfg.api_key_env, "")
            if key:
                call_kwargs["api_key"] = key
        temperature = kwargs.get("temperature", self._cfg.temperature)
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        max_tokens = kwargs.get("max_tokens", self._cfg.max_tokens)
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        response = litellm.completion(**call_kwargs)
        text = response.choices[0].message.content or ""
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return text, usage

    def is_available(self) -> bool:
        if self._cfg.api_key_env:
            return bool(os.environ.get(self._cfg.api_key_env))
        return True

    def list_models(self) -> list[str]:
        return [self._cfg.model] if self._cfg.model else []
