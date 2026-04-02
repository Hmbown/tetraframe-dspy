"""CLI-to-API proxy / backend adapter module.\n\nProvides `ClaudeCLIBackend` for invoking Claude via its CLI,\nand FastAPI routes to expose it as an OpenAI-compatible endpoint.\n"""

__all__ = ["ClaudeCLIBackend", "detect_claude_cli", "list_claude_models"]

from tetraframe.proxy.client import (
    ClaudeCLIBackend,
    detect_claude_cli,
    list_claude_models,
)
