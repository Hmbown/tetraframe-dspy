"""Tests for backend config, factory, capabilities, and CLI adapters.

All tests are deterministic — no live API calls or real subprocess invocations.
"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from tetraframe.backends.base import Backend, BackendCapabilities, BackendMetadata
from tetraframe.backends.cli_base import CLIBackendBase, _assemble_prompt
from tetraframe.backends.cli_claude import ClaudeCodeBackend
from tetraframe.backends.cli_codex import CodexBackend
from tetraframe.backends.cli_opencode import OpenCodeBackend
from tetraframe.backends.dspy_adapter import CLILanguageModel
from tetraframe.backends.factory import build_backend, build_dspy_lm, get_backend_metadata
from tetraframe.config import BackendConfig, RootConfig


# =========================================================================
# Config validation
# =========================================================================


class TestBackendConfig:
    def test_default_config_is_api_openai(self):
        cfg = BackendConfig()
        assert cfg.kind == "api"
        assert cfg.provider == "openai"
        assert cfg.api_key_env == "OPENAI_API_KEY"

    def test_cli_provider_infers_cli_kind(self):
        cfg = BackendConfig(provider="claude-code")
        assert cfg.kind == "cli"

    def test_codex_provider_infers_cli_kind(self):
        cfg = BackendConfig(provider="codex")
        assert cfg.kind == "cli"

    def test_opencode_provider_infers_cli_kind(self):
        cfg = BackendConfig(provider="opencode")
        assert cfg.kind == "cli"

    def test_anthropic_provider_infers_api_key_env(self):
        cfg = BackendConfig(provider="anthropic")
        assert cfg.api_key_env == "ANTHROPIC_API_KEY"

    def test_openrouter_provider_infers_api_key_env(self):
        cfg = BackendConfig(provider="openrouter")
        assert cfg.api_key_env == "OPENROUTER_API_KEY"

    def test_openai_compatible_gets_no_default_key_env(self):
        cfg = BackendConfig(provider="openai-compatible")
        assert cfg.api_key_env is None

    def test_explicit_kind_overrides_inference(self):
        cfg = BackendConfig(kind="api", provider="claude-code")
        assert cfg.kind == "api"

    def test_dspy_model_string_openai(self):
        cfg = BackendConfig(provider="openai", model="gpt-4.1-mini")
        assert cfg.dspy_model_string() == "openai/gpt-4.1-mini"

    def test_dspy_model_string_already_prefixed(self):
        cfg = BackendConfig(provider="openai", model="openai/gpt-4.1-mini")
        assert cfg.dspy_model_string() == "openai/gpt-4.1-mini"

    def test_dspy_model_string_openai_compatible(self):
        cfg = BackendConfig(provider="openai-compatible", model="my-model")
        assert cfg.dspy_model_string() == "openai/my-model"

    def test_dspy_model_string_anthropic(self):
        cfg = BackendConfig(provider="anthropic", model="claude-sonnet-4-6")
        assert cfg.dspy_model_string() == "anthropic/claude-sonnet-4-6"

    def test_resolved_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "sk-test")
        cfg = BackendConfig(provider="openai-compatible", api_key_env="MY_KEY")
        assert cfg.resolved_api_key() == "sk-test"

    def test_resolved_api_key_missing_env(self):
        cfg = BackendConfig(provider="openai-compatible", api_key_env="NONEXISTENT_KEY_XYZ")
        assert cfg.resolved_api_key() is None

    def test_max_retries_bounds(self):
        with pytest.raises(Exception):
            BackendConfig(max_retries=-1)
        with pytest.raises(Exception):
            BackendConfig(max_retries=11)


class TestRootConfigWithBackend:
    def test_default_root_config_has_backend(self):
        cfg = RootConfig()
        assert cfg.model.backend.kind == "api"

    def test_root_config_loads_backend_from_dict(self):
        cfg = RootConfig.model_validate({
            "model": {
                "runtime_model": "anthropic/claude-sonnet-4-6",
                "backend": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "timeout": 60.0,
                }
            }
        })
        assert cfg.model.backend.provider == "anthropic"
        assert cfg.model.backend.timeout == 60.0

    def test_root_config_loads_cli_backend(self):
        cfg = RootConfig.model_validate({
            "model": {
                "backend": {
                    "provider": "claude-code",
                    "model": "claude-sonnet-4-6",
                    "binary": "/usr/local/bin/claude",
                }
            }
        })
        assert cfg.model.backend.kind == "cli"
        assert cfg.model.backend.binary == "/usr/local/bin/claude"

    def test_root_config_has_proxy_section(self):
        cfg = RootConfig()
        assert cfg.proxy.host == "127.0.0.1"
        assert cfg.proxy.port == 8765


# =========================================================================
# Factory
# =========================================================================


class TestFactory:
    def test_build_backend_creates_claude_code(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        cfg = BackendConfig(provider="claude-code", model="claude-sonnet-4-6")
        backend = build_backend(cfg)
        assert backend.metadata.kind == "cli"
        assert backend.metadata.provider == "claude-code"

    def test_build_backend_creates_codex(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/codex")
        cfg = BackendConfig(provider="codex", model="o4-mini")
        backend = build_backend(cfg)
        assert backend.metadata.provider == "codex"

    def test_build_backend_creates_opencode(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/opencode")
        cfg = BackendConfig(provider="opencode")
        backend = build_backend(cfg)
        assert backend.metadata.provider == "opencode"

    def test_build_backend_api_requires_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = BackendConfig(provider="openai", model="")
        with pytest.raises(ValueError, match="requires a model name"):
            build_backend(cfg)

    def test_build_backend_api_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = BackendConfig(provider="openai", model="gpt-4.1-mini")
        with pytest.raises(ValueError, match="env var"):
            build_backend(cfg)

    def test_build_backend_api_succeeds_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = BackendConfig(provider="openai", model="gpt-4.1-mini")
        backend = build_backend(cfg)
        assert backend.metadata.kind == "api"

    def test_build_backend_unknown_cli_raises(self):
        cfg = BackendConfig(kind="cli", provider="unknown-cli")
        with pytest.raises(ValueError, match="Unknown CLI provider"):
            build_backend(cfg)

    def test_get_backend_metadata_api(self):
        cfg = RootConfig.model_validate({
            "model": {
                "runtime_model": "openai/gpt-4.1-mini",
                "backend": {"provider": "openai", "model": "gpt-4.1-mini"},
            }
        })
        meta = get_backend_metadata(cfg)
        assert meta.kind == "api"
        assert meta.provider == "openai"

    def test_get_backend_metadata_cli(self):
        cfg = RootConfig.model_validate({
            "model": {
                "backend": {"provider": "claude-code", "model": "claude-opus-4-6"},
            }
        })
        meta = get_backend_metadata(cfg)
        assert meta.kind == "cli"
        assert "max_tokens" in meta.warnings[0]

    def test_build_dspy_lm_legacy_path(self, monkeypatch):
        """When no explicit backend model, use runtime_model string."""
        cfg = RootConfig.model_validate({
            "model": {"runtime_model": "openai/gpt-4.1-mini"},
        })
        lm = build_dspy_lm(cfg)
        assert lm.model == "openai/gpt-4.1-mini"

    def test_build_dspy_lm_cli_backend(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        cfg = RootConfig.model_validate({
            "model": {
                "backend": {
                    "provider": "claude-code",
                    "model": "claude-sonnet-4-6",
                }
            }
        })
        lm = build_dspy_lm(cfg)
        assert isinstance(lm, CLILanguageModel)
        assert "claude-code" in lm.model


# =========================================================================
# CLI Backend adapters — mocked subprocess
# =========================================================================


def _mock_subprocess_run(stdout: str, returncode: int = 0, stderr: str = ""):
    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode, stdout=stdout, stderr=stderr,
        )
    return _run


class TestClaudeCodeBackend:
    def test_chat_parses_json_result(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        backend = ClaudeCodeBackend(binary="/usr/bin/claude")
        payload = json.dumps({"result": "Hello from Claude", "modelUsage": {}})
        monkeypatch.setattr(subprocess, "run", _mock_subprocess_run(payload))
        text = backend.chat([{"role": "user", "content": "hi"}])
        assert text == "Hello from Claude"

    def test_chat_with_usage_extracts_tokens(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        backend = ClaudeCodeBackend(binary="/usr/bin/claude")
        payload = json.dumps({
            "result": "response",
            "modelUsage": {"claude-sonnet-4-6": {"inputTokens": 10, "outputTokens": 20}},
        })
        monkeypatch.setattr(subprocess, "run", _mock_subprocess_run(payload))
        text, usage = backend.chat_with_usage([{"role": "user", "content": "hi"}])
        assert text == "response"
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_chat_raises_on_nonzero_exit(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        backend = ClaudeCodeBackend(binary="/usr/bin/claude")
        monkeypatch.setattr(subprocess, "run", _mock_subprocess_run("", 1, "error msg"))
        with pytest.raises(RuntimeError, match="claude-code CLI failed"):
            backend.chat([{"role": "user", "content": "hi"}])

    def test_chat_raises_on_api_error_event(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        backend = ClaudeCodeBackend(binary="/usr/bin/claude")
        payload = json.dumps({"type": "error", "error": {"name": "rate_limit"}})
        monkeypatch.setattr(subprocess, "run", _mock_subprocess_run(payload))
        with pytest.raises(RuntimeError, match="ClaudeCLI API error"):
            backend.chat([{"role": "user", "content": "hi"}])

    def test_metadata_reports_no_max_tokens(self):
        with patch("shutil.which", return_value="/usr/bin/claude"):
            backend = ClaudeCodeBackend(binary="/usr/bin/claude")
        assert backend.metadata.capabilities.max_tokens is False
        assert any("max_tokens" in w for w in backend.metadata.warnings)

    def test_list_models(self):
        with patch("shutil.which", return_value="/usr/bin/claude"):
            backend = ClaudeCodeBackend(binary="/usr/bin/claude")
        models = backend.list_models()
        assert "claude-sonnet-4-6" in models
        assert "claude-opus-4-6" in models

    def test_command_includes_model_flag_for_specific_model(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        backend = ClaudeCodeBackend(binary="/usr/bin/claude", model="claude-opus-4-6")
        cmd = backend._build_command("test prompt", "claude-opus-4-6")
        assert "--model" in cmd
        assert "claude-opus-4-6" in cmd

    def test_command_omits_model_flag_for_bare_alias(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        backend = ClaudeCodeBackend(binary="/usr/bin/claude")
        cmd = backend._build_command("test prompt", "claude-sonnet")
        assert "--model" not in cmd


class TestCodexBackend:
    def test_chat_captures_raw_stdout(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/codex")
        backend = CodexBackend(binary="/usr/bin/codex")
        monkeypatch.setattr(subprocess, "run", _mock_subprocess_run("Codex response\n"))
        text = backend.chat([{"role": "user", "content": "hi"}])
        assert text == "Codex response"

    def test_command_structure(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/codex")
        backend = CodexBackend(binary="/usr/bin/codex", model="o4-mini")
        cmd = backend._build_command("prompt text", "o4-mini")
        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        assert "--model" in cmd


class TestOpenCodeBackend:
    def test_chat_captures_raw_stdout(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/opencode")
        backend = OpenCodeBackend(binary="/usr/bin/opencode")
        monkeypatch.setattr(subprocess, "run", _mock_subprocess_run("OpenCode response\n"))
        text = backend.chat([{"role": "user", "content": "hi"}])
        assert text == "OpenCode response"

    def test_command_structure(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/opencode")
        backend = OpenCodeBackend(binary="/usr/bin/opencode")
        cmd = backend._build_command("prompt text", "my-model")
        assert "run" in cmd
        assert "-m" in cmd


# =========================================================================
# DSPy adapter
# =========================================================================


class _FakeCLIBackend(CLIBackendBase):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def default_binary_name(self) -> str:
        return "fake"

    def _build_command(self, prompt, model, **kwargs):
        return ["fake", prompt]

    def _parse_output(self, stdout):
        return stdout.strip()

    def _parse_output_with_usage(self, stdout):
        return stdout.strip(), {}

    def _known_models(self):
        return ["fake-model"]


class TestCLILanguageModel:
    def test_call_returns_list_of_strings(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/fake")
        backend = _FakeCLIBackend(binary="/usr/bin/fake")
        monkeypatch.setattr(subprocess, "run", _mock_subprocess_run("LM response"))
        lm = CLILanguageModel(backend)
        result = lm(prompt="hello")
        assert result == ["LM response"]

    def test_call_with_messages(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/fake")
        backend = _FakeCLIBackend(binary="/usr/bin/fake")
        monkeypatch.setattr(subprocess, "run", _mock_subprocess_run("msg response"))
        lm = CLILanguageModel(backend)
        result = lm(messages=[{"role": "user", "content": "hi"}])
        assert result == ["msg response"]

    def test_n_parameter_repeats_calls(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/fake")
        backend = _FakeCLIBackend(binary="/usr/bin/fake")
        call_count = 0
        def counting_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=f"response {call_count}", stderr="")
        monkeypatch.setattr(subprocess, "run", counting_run)
        lm = CLILanguageModel(backend)
        result = lm(prompt="hello", n=3)
        assert len(result) == 3

    def test_history_is_recorded(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/fake")
        backend = _FakeCLIBackend(binary="/usr/bin/fake")
        monkeypatch.setattr(subprocess, "run", _mock_subprocess_run("response"))
        lm = CLILanguageModel(backend)
        lm(prompt="hello")
        assert len(lm.history) == 1
        assert lm.history[0]["outputs"] == ["response"]


# =========================================================================
# Prompt assembly
# =========================================================================


class TestAssemblePrompt:
    def test_system_message_wrapped(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        prompt = _assemble_prompt(messages)
        assert "<system>" in prompt
        assert "You are helpful." in prompt

    def test_assistant_message_wrapped(self):
        messages = [{"role": "assistant", "content": "I said this."}]
        prompt = _assemble_prompt(messages)
        assert "<assistant>" in prompt

    def test_user_message_is_plain(self):
        messages = [{"role": "user", "content": "hello"}]
        prompt = _assemble_prompt(messages)
        assert prompt == "hello"

    def test_multipart_content_extracted(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}]
        prompt = _assemble_prompt(messages)
        assert "part1" in prompt
        assert "part2" in prompt


# =========================================================================
# Tracing with backend metadata
# =========================================================================


class TestTracingBackendMetadata:
    def test_trace_logger_propagates_backend_info(self, tmp_path):
        from tetraframe.tracing import TraceLogger
        logger = TraceLogger(str(tmp_path / "traces"))
        logger.set_backend_info(
            name="openai-api",
            kind="api",
            model="gpt-4.1-mini",
            execution_mode="direct",
            capability_warnings=["none"],
        )
        ctx = logger.stage(
            run_id="run_test",
            stage_name="stage0",
            module_name="TestModule",
            signature_name="TestSig",
            attempt=1,
            visible_inputs={"x": 1},
            blocked_input_fields=[],
        )
        trace = ctx.close({"output": "y"})
        assert trace.backend_name == "openai-api"
        assert trace.backend_kind == "api"
        assert trace.backend_model == "gpt-4.1-mini"
        assert trace.execution_mode == "direct"

    def test_trace_logger_defaults_to_empty_backend_info(self, tmp_path):
        from tetraframe.tracing import TraceLogger
        logger = TraceLogger(str(tmp_path / "traces"))
        ctx = logger.stage(
            run_id="run_test",
            stage_name="stage0",
            module_name="TestModule",
            signature_name=None,
            attempt=1,
            visible_inputs={},
            blocked_input_fields=[],
        )
        trace = ctx.close({})
        assert trace.backend_name == ""
        assert trace.capability_warnings == []


# =========================================================================
# API backend (unit tests without live calls)
# =========================================================================


class TestAPIBackend:
    def test_build_dspy_lm_kwargs(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from tetraframe.backends.api import APIBackend
        cfg = BackendConfig(provider="openai", model="gpt-4.1-mini", temperature=0.5)
        api = APIBackend(cfg)
        kwargs = api.build_dspy_lm_kwargs()
        assert kwargs["model"] == "openai/gpt-4.1-mini"
        assert kwargs["temperature"] == 0.5
        assert kwargs["api_key"] == "sk-test"

    def test_build_dspy_lm_kwargs_with_base_url(self, monkeypatch):
        from tetraframe.backends.api import APIBackend
        cfg = BackendConfig(
            provider="openai-compatible",
            model="my-model",
            base_url="http://localhost:8080/v1",
        )
        api = APIBackend(cfg)
        kwargs = api.build_dspy_lm_kwargs()
        assert kwargs["model"] == "openai/my-model"
        assert kwargs["api_base"] == "http://localhost:8080/v1"

    def test_metadata_reports_capabilities(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from tetraframe.backends.api import APIBackend
        cfg = BackendConfig(provider="openai", model="gpt-4.1-mini")
        api = APIBackend(cfg)
        # All API backends report generic capabilities (we can't probe the actual endpoint)
        assert api.metadata.capabilities.streaming is True
        assert api.metadata.capabilities.max_tokens is True
        assert api.metadata.capabilities.temperature is True

    def test_is_available_checks_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from tetraframe.backends.api import APIBackend
        cfg = BackendConfig(provider="openai", model="gpt-4.1-mini")
        api = APIBackend(cfg)
        assert api.is_available() is True

    def test_list_models_returns_configured_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from tetraframe.backends.api import APIBackend
        cfg = BackendConfig(provider="openai", model="gpt-4.1-mini")
        api = APIBackend(cfg)
        assert api.list_models() == ["gpt-4.1-mini"]
