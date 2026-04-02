"""TetraFrame model tools — plugin-based model access."""
from tetraframe.tools.protocol import CompletionResult, ModelTool, ToolInfo
from tetraframe.tools.registry import ToolRegistry, auto_discover
from tetraframe.tools.dspy_adapter import ModelToolLM

__all__ = [
    "CompletionResult",
    "ModelTool",
    "ModelToolLM",
    "ToolInfo",
    "ToolRegistry",
    "auto_discover",
]
