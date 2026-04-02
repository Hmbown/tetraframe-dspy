"""Tests for the ModelTool plugin system."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tetraframe.tools.protocol import CompletionResult, ModelTool, ToolInfo
from tetraframe.tools.registry import ToolRegistry, auto_discover
from tetraframe.tools.dspy_adapter import ModelToolLM
from tetraframe.tools.api_tool import DirectAPITool
from tetraframe.config import RootConfig


# =========================================================================
# Fake tool for testing
# =========================================================================

class FakeTool:
    """A ModelTool that returns canned responses."""

    def __init__(
        self,
        name: str = "fake",
        provider: str = "test",
        model: str = "fake-model",
        available: bool = True,
        priority: int = 50,
        response: str = "fake response",
    ):
        self._info = ToolInfo(
            name=name, provider=provider, model=model,
            kind="api", priority=priority,
        )
        self._available = available
        self._response = response

    @property
    def info(self) -> ToolInfo:
        return self._info

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> list[CompletionResult]:
        return [
            CompletionResult(text=self._response, model=self._info.model)
            for _ in range(n)
        ]

    def is_available(self) -> bool:
        return self._available


# =========================================================================
# Protocol compliance
# =========================================================================

class TestProtocol:
    def test_fake_tool_is_model_tool(self):
        assert isinstance(FakeTool(), ModelTool)

    def test_direct_api_tool_is_model_tool(self):
        tool = DirectAPITool(name="t", provider="p", model="m")
        assert isinstance(tool, ModelTool)

    def test_tool_info_fields(self):
        info = ToolInfo(name="a", provider="b", model="c", kind="api")
        assert info.name == "a"
        assert info.priority == 50

    def test_completion_result_defaults(self):
        r = CompletionResult(text="hi")
        assert r.usage == {}
        assert r.raw is None
        assert r.model == ""


# =========================================================================
# Registry
# =========================================================================

class TestRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = FakeTool(name="my-tool")
        reg.register(tool)
        assert reg.get("my-tool") is tool
        assert reg.get("nonexistent") is None

    def test_best_available_by_priority(self):
        reg = ToolRegistry()
        reg.register(FakeTool(name="low", priority=10))
        reg.register(FakeTool(name="high", priority=90))
        best = reg.best_available()
        assert best is not None
        assert best.info.name == "low"

    def test_best_available_skips_unavailable(self):
        reg = ToolRegistry()
        reg.register(FakeTool(name="down", priority=1, available=False))
        reg.register(FakeTool(name="up", priority=50))
        best = reg.best_available()
        assert best is not None
        assert best.info.name == "up"

    def test_best_available_none_when_empty(self):
        reg = ToolRegistry()
        assert reg.best_available() is None

    def test_best_available_none_when_all_down(self):
        reg = ToolRegistry()
        reg.register(FakeTool(name="a", available=False))
        reg.register(FakeTool(name="b", available=False))
        assert reg.best_available() is None

    def test_all_available_sorted(self):
        reg = ToolRegistry()
        reg.register(FakeTool(name="c", priority=30))
        reg.register(FakeTool(name="a", priority=10))
        reg.register(FakeTool(name="b", priority=20, available=False))
        avail = reg.all_available()
        assert len(avail) == 2
        assert avail[0].info.name == "a"
        assert avail[1].info.name == "c"

    def test_summary_includes_all(self):
        reg = ToolRegistry()
        reg.register(FakeTool(name="x", available=True))
        reg.register(FakeTool(name="y", available=False))
        summary = reg.summary()
        assert len(summary) == 2
        names = {s["name"] for s in summary}
        assert names == {"x", "y"}

    def test_best_available_with_tag_filter(self):
        reg = ToolRegistry()
        t1 = FakeTool(name="a", priority=1)
        t1._info = ToolInfo(name="a", provider="p", model="m", kind="api", priority=1, tags=("fast",))
        t2 = FakeTool(name="b", priority=2)
        t2._info = ToolInfo(name="b", provider="p", model="m", kind="api", priority=2, tags=("reasoning",))
        reg.register(t1)
        reg.register(t2)
        best = reg.best_available(tags=("reasoning",))
        assert best is not None
        assert best.info.name == "b"


# =========================================================================
# DSPy adapter
# =========================================================================

class TestModelToolLM:
    def test_forward_returns_expected_structure(self):
        tool = FakeTool(response="hello world")
        lm = ModelToolLM(tool)
        result = lm.forward(messages=[{"role": "user", "content": "hi"}])
        assert hasattr(result, "choices")
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "hello world"
        assert hasattr(result, "usage")

    def test_forward_with_prompt(self):
        tool = FakeTool(response="from prompt")
        lm = ModelToolLM(tool)
        result = lm.forward(prompt="test prompt")
        assert result.choices[0].message.content == "from prompt"

    def test_forward_n_greater_than_1(self):
        tool = FakeTool(response="multi")
        lm = ModelToolLM(tool)
        result = lm.forward(messages=[{"role": "user", "content": "hi"}], n=3)
        assert len(result.choices) == 3

    def test_forward_raises_without_input(self):
        tool = FakeTool()
        lm = ModelToolLM(tool)
        with pytest.raises(ValueError, match="requires either prompt or messages"):
            lm.forward()

    def test_model_name_format(self):
        tool = FakeTool(provider="deepseek", model="deepseek-chat")
        lm = ModelToolLM(tool)
        assert lm.model == "tool/deepseek/deepseek-chat"

    def test_repr(self):
        tool = FakeTool(name="my-tool", model="m")
        lm = ModelToolLM(tool)
        assert "my-tool" in repr(lm)

    def test_tool_attribute_exposed(self):
        tool = FakeTool()
        lm = ModelToolLM(tool)
        assert lm._tool is tool


# =========================================================================
# DirectAPITool
# =========================================================================

class TestDirectAPITool:
    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "sk-123")
        tool = DirectAPITool(name="t", provider="p", model="m", api_key_env="TEST_KEY")
        assert tool.is_available()

    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("TEST_KEY", raising=False)
        tool = DirectAPITool(name="t", provider="p", model="m", api_key_env="TEST_KEY")
        assert not tool.is_available()

    def test_always_available_without_key_env(self):
        tool = DirectAPITool(name="t", provider="p", model="m")
        assert tool.is_available()

    def test_info(self):
        tool = DirectAPITool(
            name="deepseek", provider="deepseek", model="deepseek-chat",
            priority=20, cost_tier="cheap",
        )
        assert tool.info.name == "deepseek"
        assert tool.info.priority == 20
        assert tool.info.cost_tier == "cheap"
        assert tool.info.kind == "api"


# =========================================================================
# Factory integration
# =========================================================================

class TestFactoryToolIntegration:
    def test_explicit_config_bypasses_tools(self):
        """Configs with explicit models should NOT be overridden by auto-discover."""
        from tetraframe.backends.factory import build_dspy_lm
        cfg = RootConfig.model_validate({
            "model": {"runtime_model": "openai/gpt-4.1-mini"},
        })
        # auto_discover is True by default, but explicit model should win
        lm = build_dspy_lm(cfg)
        assert lm.model == "openai/gpt-4.1-mini"

    def test_preferred_tool_overrides_config(self, monkeypatch):
        """preferred_tool should use tool path even with explicit model."""
        from tetraframe.backends.factory import build_dspy_lm

        # Mock auto_discover to return a known tool
        fake_tool = FakeTool(name="test-tool", provider="test", model="test-m")
        mock_registry = ToolRegistry()
        mock_registry.register(fake_tool)

        monkeypatch.setattr(
            "tetraframe.tools.registry.auto_discover",
            lambda: mock_registry,
        )

        cfg = RootConfig.model_validate({
            "model": {"runtime_model": "openai/gpt-4.1-mini"},
            "tools": {"preferred_tool": "test-tool"},
        })
        lm = build_dspy_lm(cfg)
        assert hasattr(lm, "_tool")
        assert lm._tool.info.name == "test-tool"

    def test_zero_config_uses_tools(self, monkeypatch):
        """Config with no model should auto-discover and use tools."""
        from tetraframe.backends.factory import build_dspy_lm

        fake_tool = FakeTool(name="auto-found", priority=1)
        mock_registry = ToolRegistry()
        mock_registry.register(fake_tool)

        monkeypatch.setattr(
            "tetraframe.tools.registry.auto_discover",
            lambda: mock_registry,
        )

        # Clear default model so auto-discover kicks in
        cfg = RootConfig.model_validate({
            "model": {"runtime_model": "", "backend": {"model": ""}},
        })
        lm = build_dspy_lm(cfg)
        assert hasattr(lm, "_tool")
        assert lm._tool.info.name == "auto-found"


# =========================================================================
# Auto-discover
# =========================================================================

class TestAutoDiscover:
    def test_discovers_env_tools(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Suppress Hermes and CLI discovery
        monkeypatch.setattr("tetraframe.tools.registry.shutil.which", lambda x: None)
        monkeypatch.setattr(
            "tetraframe.tools.hermes_tool.HERMES_DIR",
            type("P", (), {"exists": lambda self: False})(),
        )

        registry = auto_discover()
        avail = registry.all_available()
        names = {t.info.name for t in avail}
        assert "openai-api" in names

    def test_hermes_tools_have_highest_priority(self, monkeypatch):
        """Hermes tools should come before direct API tools."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("tetraframe.tools.registry.shutil.which", lambda x: None)

        registry = auto_discover()
        avail = registry.all_available()
        if len(avail) >= 2:
            hermes = [t for t in avail if "hermes" in t.info.name]
            direct = [t for t in avail if "hermes" not in t.info.name and t.info.kind == "api"]
            if hermes and direct:
                assert hermes[0].info.priority < direct[0].info.priority
