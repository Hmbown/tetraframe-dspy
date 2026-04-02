"""DSPy adapter — wraps any ModelTool as a dspy.BaseLM."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tetraframe.tools.protocol import ModelTool

try:
    from dspy import BaseLM as _BaseLM
except ImportError:
    _BaseLM = None  # type: ignore[assignment,misc]


class _DictNamespace(SimpleNamespace):
    """SimpleNamespace that supports dict() iteration."""

    def __iter__(self):
        return iter(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()


_Base: type = _BaseLM if _BaseLM is not None else object


class ModelToolLM(_Base):  # type: ignore[misc]
    """DSPy-compatible LM backed by a ModelTool.

    Drop-in replacement for dspy.LM(). Used via:
        dspy.configure(lm=ModelToolLM(tool))
    """

    def __init__(self, tool: ModelTool, **defaults: Any) -> None:
        info = tool.info
        model_name = f"tool/{info.provider}/{info.model}"
        if _BaseLM is not None:
            super().__init__(
                model=model_name,
                model_type="chat",
                temperature=defaults.get("temperature", 0.0),
                max_tokens=defaults.get("max_tokens", 4096),
                cache=False,
            )
        else:
            self.model = model_name
            self.kwargs = {}
            self.history = []
        self._tool = tool
        self._defaults = defaults

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        merged = {**self._defaults, **kwargs}
        if messages:
            effective_messages = messages
        elif prompt:
            effective_messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("ModelToolLM requires either prompt or messages")

        n = merged.pop("n", 1)
        temperature = merged.pop("temperature", None)
        max_tokens = merged.pop("max_tokens", None)
        # Drop kwargs that the tool doesn't understand
        merged.pop("num_retries", None)
        merged.pop("cache", None)

        results = self._tool.complete(
            effective_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )

        choices = []
        for result in results:
            message = SimpleNamespace(
                content=result.text,
                tool_calls=None,
                reasoning_content=None,
            )
            message.provider_specific_fields = {}
            choices.append(SimpleNamespace(message=message, logprobs=None))

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for r in results:
            for k in total_usage:
                total_usage[k] += r.usage.get(k, 0)

        usage = _DictNamespace(**total_usage)
        model_name = results[0].model if results else "unknown"
        return SimpleNamespace(choices=choices, usage=usage, model=model_name)

    def inspect_history(self, n: int = 1) -> list[dict[str, Any]]:
        return getattr(self, "history", [])[-n:]

    def __repr__(self) -> str:
        info = self._tool.info
        return f"ModelToolLM(tool={info.name}, model={info.model})"
