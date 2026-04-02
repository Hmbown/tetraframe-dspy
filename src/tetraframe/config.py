from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

# Providers that use the OpenAI-compatible API surface (through litellm / dspy)
API_PROVIDERS = frozenset({
    "openai",
    "anthropic",
    "openrouter",
    "openai-compatible",
})

# Providers that invoke a local CLI binary via subprocess
CLI_PROVIDERS = frozenset({
    "claude-code",
    "codex",
    "opencode",
})

# Map provider names to the env var that typically holds the API key.
DEFAULT_API_KEY_ENVS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class BackendConfig(BaseModel):
    """Describes how to reach a model.

    ``kind`` is inferred from ``provider`` when omitted.
    """

    kind: Literal["api", "cli"] | None = None
    provider: str = "openai"
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float = 120.0
    max_tokens: int | None = None
    temperature: float | None = None
    # CLI-specific
    binary: str | None = None
    cli_args: list[str] = Field(default_factory=list)
    env_passthrough: list[str] = Field(default_factory=list)
    # Retry
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _infer_kind_and_defaults(self) -> "BackendConfig":
        # Infer kind from provider when not explicitly set
        if self.kind is None:
            if self.provider in CLI_PROVIDERS:
                self.kind = "cli"
            else:
                self.kind = "api"
        # Infer api_key_env for known API providers
        if self.kind == "api" and self.api_key_env is None:
            self.api_key_env = DEFAULT_API_KEY_ENVS.get(self.provider)
        return self

    def resolved_api_key(self) -> str | None:
        """Return the API key from the configured env var, or None."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def dspy_model_string(self) -> str:
        """Return the litellm-compatible model string for dspy.LM().

        For provider 'openai' with model 'gpt-4.1-mini' -> 'openai/gpt-4.1-mini'.
        For provider 'openai-compatible' -> 'openai/<model>' (litellm uses openai/).
        For provider 'openrouter' with model 'anthropic/claude-sonnet-4-6' ->
            'openrouter/anthropic/claude-sonnet-4-6'.
        """
        model = self.model
        if not model:
            return ""
        provider = self.provider
        if provider == "openai-compatible":
            # litellm routes through OpenAI client when prefix is 'openai/'
            if not model.startswith("openai/"):
                return f"openai/{model}"
            return model
        # For routing providers like openrouter, always prefix with provider
        if provider == "openrouter":
            if model.startswith("openrouter/"):
                return model
            return f"openrouter/{model}"
        if "/" not in model:
            return f"{provider}/{model}"
        return model


class ProxyConfig(BaseModel):
    """Proxy server settings."""

    host: str = "127.0.0.1"
    port: int = 8765
    backend: str = "default"


# ---------------------------------------------------------------------------
# Existing configs (preserved)
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    runtime_model: str = "openai/gpt-4.1-mini"
    reflection_model: str = "openai/gpt-4.1"
    backend: BackendConfig = Field(default_factory=BackendConfig)
    reflection_backend: BackendConfig | None = None


class ProgramConfig(BaseModel):
    parallel_corners: bool = True
    randomize_sequential_fallback_order: bool = True
    max_corner_generation_attempts: int = Field(default=2, ge=1, le=8)
    corner_temperatures: dict[str, float] = Field(
        default_factory=lambda: {"P": 0.7, "not-P": 0.85, "both": 0.9, "neither": 0.9}
    )
    trace_dir: str = "runs/traces"


class CompileConfig(BaseModel):
    train_path: str = "examples/benchmark_cases_train.jsonl"
    dev_path: str = "examples/benchmark_cases_dev.jsonl"
    test_path: str = "examples/benchmark_cases_test.jsonl"


class BenchmarkConfig(BaseModel):
    pass_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class RootConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    program: ProgramConfig = Field(default_factory=ProgramConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)


def load_config(path: str | Path) -> RootConfig:
    path = Path(path)
    if not path.exists():
        return RootConfig()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RootConfig.model_validate(payload)
