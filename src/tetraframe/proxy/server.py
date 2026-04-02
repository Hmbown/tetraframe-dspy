"""Multi-backend OpenAI-compatible proxy server for TetraFrame.

Exposes:
- ``/health``                   — backend status and capabilities
- ``/v1/models``                — model listing per backend
- ``/v1/chat/completions``      — chat completions with backend selection
- ``/v1/chat/completions`` (stream) — proper SSE streaming

Supports:
- CLI backends (Claude Code, Codex, OpenCode) via subprocess adapters
- API backends (OpenAI, Anthropic, OpenRouter, generic) via httpx

Backend selection:
- ``model`` field: if it contains a ``/`` prefix matching a provider, routes
  to that backend (e.g. ``claude-code/claude-sonnet-4-6``).
- ``X-Backend`` header: explicit backend selection by provider name.
- Default: uses the configured default backend.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tetraframe.backends.base import Backend, BackendMetadata
from tetraframe.backends.factory import build_backend
from tetraframe.config import BackendConfig, load_config

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_backends: dict[str, Backend] = {}
_default_backend_name: str = ""


def register_backend(name: str, backend: Backend) -> None:
    _backends[name] = backend


def get_backend(name: str | None = None) -> Backend:
    key = name or _default_backend_name
    if key not in _backends:
        available = list(_backends.keys())
        raise RuntimeError(
            f"Backend '{key}' not registered. Available: {available}"
        )
    return _backends[key]


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Auto-register from env vars for backward compat with old proxy CLI
    if not _backends:
        _try_register_from_env()
    yield


def _try_register_from_env() -> None:
    """Register a Claude CLI backend from env vars (backward compat)."""
    global _default_backend_name
    cbin = os.environ.get("TETRAFRAME_PROXY_BIN", "")
    dmodel = os.environ.get("TETRAFRAME_PROXY_MODEL", "")
    try:
        cfg = BackendConfig(
            kind="cli",
            provider="claude-code",
            model=dmodel or "",
            binary=cbin or None,
        )
        backend = build_backend(cfg)
        register_backend("claude-code", backend)
        _default_backend_name = "claude-code"
    except Exception:
        pass


app = FastAPI(title="TetraFrame Multi-Backend Proxy", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health(request: Request) -> Any:
    if not _backends:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "backends": [],
                "error": "No backends registered",
            },
        )
    backend_reports = []
    overall_ok = False
    for name, backend in _backends.items():
        available = False
        try:
            available = backend.is_available()
        except Exception:
            pass
        meta = backend.metadata
        report = {
            "name": name,
            "available": available,
            "default": name == _default_backend_name,
            **meta.to_dict(),
        }
        backend_reports.append(report)
        if available:
            overall_ok = True

    status_code = 200 if overall_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if overall_ok else "degraded",
            "backends": backend_reports,
        },
    )


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for name, backend in _backends.items():
        try:
            for model_id in backend.list_models():
                models.append({
                    "id": model_id,
                    "object": "model",
                    "owned_by": backend.metadata.provider,
                    "backend": name,
                })
        except Exception:
            pass
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    req: dict[str, Any] = await request.json()
    messages = req.get("messages", [])
    model = req.get("model")
    stream = req.get("stream", False)
    max_tokens = req.get("max_tokens")
    temperature = req.get("temperature")

    # Validate messages
    if not isinstance(messages, list) or not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "`messages` must be a non-empty list.", "type": "invalid_request_error"}},
        )
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "`max_tokens` must be a positive integer when provided.", "type": "invalid_request_error"}},
        )

    # Select backend
    backend_name = request.headers.get("X-Backend")
    if not backend_name and model and "/" in model:
        # Try to parse provider prefix from model
        prefix, _, rest = model.partition("/")
        if prefix in _backends:
            backend_name = prefix
            model = rest

    try:
        backend = get_backend(backend_name)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": str(exc), "type": "proxy_error", "code": "no_backend"}},
        )

    # Collect capability warnings
    proxy_warnings: list[str] = []
    caps = backend.metadata.capabilities
    if max_tokens is not None and not caps.max_tokens:
        proxy_warnings.append(
            f"`max_tokens` is not enforced by the {backend.metadata.name} backend; "
            f"outputs may exceed the requested limit."
        )
    if temperature is not None and not caps.temperature:
        proxy_warnings.append(
            f"`temperature` is not enforced by the {backend.metadata.name} backend."
        )

    # Invoke
    kwargs: dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        text, usage = backend.chat_with_usage(messages, **kwargs)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "proxy_error", "code": "backend_error"}},
        )
    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Backend timed out", "type": "proxy_error", "code": "timeout"}},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Internal server error: {exc}", "type": "internal_error"}},
        )

    response_model = model or backend.metadata.model or "unknown"
    result = _build_completion_response(response_model, text, usage, proxy_warnings)

    if stream:
        return StreamingResponse(
            _stream_sse(result),
            media_type="text/event-stream",
        )
    return result


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

import subprocess  # noqa: E402 — needed for TimeoutExpired above


def _build_completion_response(
    model: str,
    text: str,
    usage: dict[str, Any],
    proxy_warnings: list[str],
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "proxy_warnings": proxy_warnings,
    }


async def _stream_sse(result: dict[str, Any]):
    """Yield SSE events for a streaming response.

    For CLI backends that don't support real streaming, we emit the full
    response as a single chunk followed by [DONE].  For API backends
    that do support streaming, this path still works correctly — the
    difference is documented as a capability limitation.
    """
    text = result["choices"][0]["message"]["content"]

    # Emit chunk event
    chunk = {
        "id": result["id"],
        "object": "chat.completion.chunk",
        "created": result["created"],
        "model": result["model"],
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Emit finish event
    finish = {
        "id": result["id"],
        "object": "chat.completion.chunk",
        "created": result["created"],
        "model": result["model"],
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
        "usage": result.get("usage", {}),
    }
    yield f"data: {json.dumps(finish)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="TetraFrame Multi-Backend Proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default="", help="Path to TetraFrame config yaml")
    # Backward-compat flags
    parser.add_argument("--model", default="")
    parser.add_argument("--claude-bin", default="")
    args = parser.parse_args()

    global _default_backend_name

    if args.config:
        # Load from config file — may register multiple backends
        cfg = load_config(args.config)
        proxy_cfg = cfg.proxy
        backend_cfg = cfg.model.backend
        try:
            backend = build_backend(backend_cfg)
            name = backend.metadata.provider
            register_backend(name, backend)
            _default_backend_name = name
            print(f"Registered backend: {name} ({backend.metadata.kind})")
        except Exception as exc:
            print(f"WARNING: Failed to register backend: {exc}", file=sys.stderr)
    else:
        # Backward compat: register Claude CLI from flags/env
        cbin = args.claude_bin or os.environ.get("TETRAFRAME_PROXY_BIN", "")
        dmodel = args.model or os.environ.get("TETRAFRAME_PROXY_MODEL", "")
        try:
            cfg_obj = BackendConfig(
                kind="cli",
                provider="claude-code",
                model=dmodel or "",
                binary=cbin or None,
            )
            backend = build_backend(cfg_obj)
            register_backend("claude-code", backend)
            _default_backend_name = "claude-code"
            print(f"Registered backend: claude-code (cli)")
            print(f"  Model: {dmodel or 'default'}  Binary: {backend.metadata.model}")
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    host = args.host
    port = args.port
    print(f"Proxy running on http://{host}:{port}")
    print(f"  Default backend: {_default_backend_name}")
    print(f"  Registered backends: {list(_backends.keys())}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
