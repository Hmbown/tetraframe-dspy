"""Direct API tool — wraps any OpenAI-compatible endpoint via httpx."""
from __future__ import annotations

import os
from typing import Any

from tetraframe.tools.protocol import CompletionResult, ModelTool, ToolInfo


def _openai_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    n: int = 1,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Raw POST to an OpenAI-compatible /v1/chat/completions endpoint."""
    import httpx

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body: dict[str, Any] = {"model": model, "messages": messages, "n": n}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class DirectAPITool:
    """Direct API client over httpx — no litellm, no openai library.

    Wraps any OpenAI-compatible /v1 endpoint.
    Auth is the tool's problem — TetraFrame never sees API keys.
    """

    def __init__(
        self,
        *,
        name: str,
        provider: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str | None = None,
        priority: int = 50,
        cost_tier: str = "standard",
        tags: tuple[str, ...] = (),
        timeout: float = 120.0,
    ) -> None:
        self._name = name
        self._provider = provider
        self._model = model
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._priority = priority
        self._cost_tier = cost_tier
        self._tags = tags
        self._timeout = timeout

    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            name=self._name,
            provider=self._provider,
            model=self._model,
            kind="api",
            priority=self._priority,
            cost_tier=self._cost_tier,
            tags=self._tags,
        )

    def _api_key(self) -> str | None:
        if self._api_key_env:
            return os.environ.get(self._api_key_env)
        return None

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> list[CompletionResult]:
        data = _openai_chat(
            base_url=self._base_url,
            model=self._model,
            messages=messages,
            api_key=self._api_key(),
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            timeout=self._timeout,
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
        if self._api_key_env:
            return bool(os.environ.get(self._api_key_env))
        return True
