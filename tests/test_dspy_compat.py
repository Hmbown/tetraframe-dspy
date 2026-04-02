from __future__ import annotations

import asyncio
import builtins
import importlib.util
from pathlib import Path

import pytest


def test_dspy_shim_operates_without_real_dspy(tmp_path, monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "src" / "tetraframe" / "dspy_compat.py"
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dspy" or name.startswith("dspy."):
            raise ImportError("blocked for shim test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    spec = importlib.util.spec_from_file_location("tetraframe_dspy_compat_shim_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.__version__ == "shim"

    predictor = module.Predict()
    with pytest.raises(RuntimeError, match="DSPy is not installed"):
        predictor()

    example = module.Example(seed="x").with_inputs("seed")
    assert example["_inputs"] == ["seed"]

    saved = tmp_path / "shim-module.json"
    module.Module().save(str(saved), save_program=True)
    assert "\"save_program\": true" in saved.read_text(encoding="utf-8")

    async_runner = module.asyncify(lambda value: value + 1)
    assert asyncio.run(async_runner(2)) == 3
