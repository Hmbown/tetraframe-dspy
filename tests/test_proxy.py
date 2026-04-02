from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tetraframe.backends.base import Backend, BackendCapabilities, BackendMetadata
from tetraframe.proxy import server


class _DummyBackend:
    """Fake backend for proxy tests."""

    @property
    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="dummy-cli",
            kind="cli",
            provider="dummy",
            model="dummy-model",
            capabilities=BackendCapabilities(
                streaming=False, max_tokens=False, temperature=False,
            ),
            warnings=["max_tokens not enforced"],
        )

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return "proxy reply"

    def chat_with_usage(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        return "proxy reply", {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["dummy-model"]


def _setup_dummy(monkeypatch):
    """Register a dummy backend and set it as default."""
    server._backends.clear()
    server.register_backend("dummy", _DummyBackend())
    monkeypatch.setattr(server, "_default_backend_name", "dummy")


def _clear_backends(monkeypatch):
    """Clear all backends to simulate no-backend state."""
    server._backends.clear()
    monkeypatch.setattr(server, "_default_backend_name", "")


def test_proxy_health_reports_degraded_when_no_backends(monkeypatch):
    _clear_backends(monkeypatch)
    client = TestClient(server.app)

    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_proxy_health_reports_ok_with_backend(monkeypatch):
    _setup_dummy(monkeypatch)
    client = TestClient(server.app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert len(data["backends"]) == 1
    assert data["backends"][0]["available"] is True


def test_proxy_chat_returns_warning_for_unenforced_max_tokens(monkeypatch):
    _setup_dummy(monkeypatch)
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "proxy reply"
    assert payload["proxy_warnings"]
    assert "max_tokens" in payload["proxy_warnings"][0]


def test_proxy_chat_validates_messages(monkeypatch):
    _setup_dummy(monkeypatch)
    client = TestClient(server.app)

    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400


def test_proxy_models_lists_backend_models(monkeypatch):
    _setup_dummy(monkeypatch)
    client = TestClient(server.app)

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "dummy-model"


def test_proxy_chat_returns_503_when_no_backends(monkeypatch):
    _clear_backends(monkeypatch)
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 503


def test_proxy_backend_selection_via_header(monkeypatch):
    _setup_dummy(monkeypatch)
    # Register a second backend
    second = _DummyBackend()
    server.register_backend("other", second)

    client = TestClient(server.app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"X-Backend": "other"},
    )
    assert response.status_code == 200


def test_proxy_stream_returns_sse_events(monkeypatch):
    _setup_dummy(monkeypatch)
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "data: " in body
    assert "[DONE]" in body


def test_proxy_temperature_warning_for_cli_backend(monkeypatch):
    _setup_dummy(monkeypatch)
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "temperature": 0.5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert any("temperature" in w for w in payload["proxy_warnings"])
