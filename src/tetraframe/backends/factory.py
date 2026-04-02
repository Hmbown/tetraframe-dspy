"""Backend and DSPy LM construction from config.

Central factory used by CLI commands, the proxy, and the compile/benchmark
paths so they all construct backends identically.
"""
from __future__ import annotations

from typing import Any

from tetraframe.backends.base import Backend, BackendMetadata
from tetraframe.config import BackendConfig, RootConfig


def build_backend(cfg: BackendConfig) -> Backend:
    """Construct the appropriate Backend from a BackendConfig.

    Raises RuntimeError if the backend cannot be initialised (e.g. missing
    binary for CLI, missing API key for API).
    """
    kind = cfg.kind or ("cli" if cfg.provider in ("claude-code", "codex", "opencode") else "api")

    if kind == "cli":
        return _build_cli_backend(cfg)
    return _build_api_backend(cfg)


def build_dspy_lm(cfg: RootConfig) -> Any:
    """Construct a DSPy-compatible LM from the root config.

    For API backends: returns ``dspy.LM(model, **kwargs)``.
    For CLI backends: returns ``CLILanguageModel(backend)``.
    """
    backend_cfg = cfg.model.backend

    # If no explicit backend config, fall back to legacy model string path
    if not backend_cfg.model and cfg.model.runtime_model:
        return _build_legacy_dspy_lm(cfg.model.runtime_model, backend_cfg)

    kind = backend_cfg.kind or ("cli" if backend_cfg.provider in ("claude-code", "codex", "opencode") else "api")

    if kind == "cli":
        from tetraframe.backends.dspy_adapter import CLILanguageModel
        backend = _build_cli_backend(backend_cfg)
        return CLILanguageModel(backend)

    # API backend — construct dspy.LM
    from tetraframe.backends.api import APIBackend
    api = APIBackend(backend_cfg)
    lm_kwargs = api.build_dspy_lm_kwargs()
    model_str = lm_kwargs.pop("model")
    return _make_dspy_lm(model_str, **lm_kwargs)


def build_reflection_lm(cfg: RootConfig) -> Any:
    """Construct the reflection/teacher LM for compilation.

    Uses ``model.reflection_backend`` if set, otherwise falls back to
    building from ``model.reflection_model``.
    """
    if cfg.model.reflection_backend:
        # Construct from explicit reflection backend config
        ref_cfg = cfg.model.reflection_backend
        kind = ref_cfg.kind or ("cli" if ref_cfg.provider in ("claude-code", "codex", "opencode") else "api")
        if kind == "cli":
            from tetraframe.backends.dspy_adapter import CLILanguageModel
            backend = _build_cli_backend(ref_cfg)
            return CLILanguageModel(backend)
        from tetraframe.backends.api import APIBackend
        api = APIBackend(ref_cfg)
        lm_kwargs = api.build_dspy_lm_kwargs()
        model_str = lm_kwargs.pop("model")
        return _make_dspy_lm(model_str, temperature=1.0, max_tokens=32000, **lm_kwargs)

    # Legacy path: use reflection_model string
    return _make_dspy_lm(cfg.model.reflection_model, temperature=1.0, max_tokens=32000)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_cli_backend(cfg: BackendConfig) -> Any:
    provider = cfg.provider
    kwargs = {
        "binary": cfg.binary,
        "model": cfg.model or None,
        "timeout": cfg.timeout,
        "cli_args": cfg.cli_args,
        "env_passthrough": cfg.env_passthrough,
    }
    if provider == "claude-code":
        from tetraframe.backends.cli_claude import ClaudeCodeBackend
        return ClaudeCodeBackend(**kwargs)
    if provider == "codex":
        from tetraframe.backends.cli_codex import CodexBackend
        return CodexBackend(**kwargs)
    if provider == "opencode":
        from tetraframe.backends.cli_opencode import OpenCodeBackend
        return OpenCodeBackend(**kwargs)
    raise ValueError(f"Unknown CLI provider: {provider}")


def _build_api_backend(cfg: BackendConfig) -> Any:
    from tetraframe.backends.api import APIBackend
    return APIBackend(cfg)


def _build_legacy_dspy_lm(model_string: str, backend_cfg: BackendConfig) -> Any:
    """Construct dspy.LM from a plain litellm model string (legacy path)."""
    kwargs: dict[str, Any] = {}
    if backend_cfg.base_url:
        kwargs["api_base"] = backend_cfg.base_url
    if backend_cfg.api_key_env:
        import os
        key = os.environ.get(backend_cfg.api_key_env, "")
        if key:
            kwargs["api_key"] = key
    if backend_cfg.temperature is not None:
        kwargs["temperature"] = backend_cfg.temperature
    if backend_cfg.max_tokens is not None:
        kwargs["max_tokens"] = backend_cfg.max_tokens
    return _make_dspy_lm(model_string, **kwargs)


def _make_dspy_lm(model: str, **kwargs: Any) -> Any:
    """Construct a dspy.LM, using the real library or the shim."""
    import tetraframe.dspy_compat as dspy
    return dspy.LM(model, **kwargs)


def get_backend_metadata(cfg: RootConfig) -> BackendMetadata:
    """Return metadata for the configured runtime backend without invoking it."""
    backend_cfg = cfg.model.backend
    kind = backend_cfg.kind or ("cli" if backend_cfg.provider in ("claude-code", "codex", "opencode") else "api")
    model = backend_cfg.model or cfg.model.runtime_model
    provider = backend_cfg.provider

    if kind == "cli":
        from tetraframe.backends.cli_base import CLIBackendBase
        # Return metadata without constructing the full backend (which checks binary)
        from tetraframe.backends.base import BackendCapabilities, BackendMetadata
        caps = BackendCapabilities(streaming=False, max_tokens=False, temperature=False)
        return BackendMetadata(
            name=f"{provider}-cli",
            kind="cli",
            provider=provider,
            model=model or "(default)",
            capabilities=caps,
            warnings=["max_tokens not enforced", "temperature not enforced"],
        )

    from tetraframe.backends.base import BackendCapabilities, BackendMetadata
    from tetraframe.backends.api import _PROVIDER_CAPS
    caps = _PROVIDER_CAPS.get(provider, BackendCapabilities())
    return BackendMetadata(
        name=f"{provider}-api",
        kind="api",
        provider=provider,
        model=model,
        capabilities=caps,
    )
