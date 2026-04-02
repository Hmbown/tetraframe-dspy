from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

# Providers that invoke a local CLI binary via subprocess.
# Everything else is treated as a generic OpenAI-compatible API.
CLI_PROVIDERS = frozenset({
    "claude-code",
    "codex",
    "opencode",
})


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
            self.kind = "cli" if self.provider in CLI_PROVIDERS else "api"
        # Convention: {PROVIDER}_API_KEY for known providers
        if self.kind == "api" and self.api_key_env is None and self.provider != "openai-compatible":
            self.api_key_env = f"{self.provider.upper()}_API_KEY"
        return self

    def resolved_api_key(self) -> str | None:
        """Return the API key from the configured env var, or None."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def dspy_model_string(self) -> str:
        """Return the provider/model string for dspy.LM().

        Rules:
        - If model already contains '/', return as-is.
        - 'openai-compatible' prefix becomes 'openai/'.
        - Everything else: '{provider}/{model}'.
        """
        model = self.model
        if not model:
            return ""
        if "/" in model:
            return model
        if self.provider == "openai-compatible":
            return f"openai/{model}"
        return f"{self.provider}/{model}"


class ToolConfig(BaseModel):
    """Plugin tool configuration — the new simplified path."""

    auto_discover: bool = True
    preferred_tool: str | None = None
    fallback_chain: list[str] = Field(default_factory=list)


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
    tools: ToolConfig = Field(default_factory=ToolConfig)
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
