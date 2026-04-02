from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

try:  # pragma: no cover - exercised when DSPy is installed
    import dspy as _real_dspy  # type: ignore
    from dspy.teleprompt import BootstrapFewShot, GEPA, MIPROv2  # type: ignore

    Signature = _real_dspy.Signature
    Module = _real_dspy.Module
    InputField = _real_dspy.InputField
    OutputField = _real_dspy.OutputField
    Predict = _real_dspy.Predict
    ChainOfThought = _real_dspy.ChainOfThought
    Refine = _real_dspy.Refine
    BestOfN = _real_dspy.BestOfN
    Example = _real_dspy.Example
    LM = _real_dspy.LM
    configure = _real_dspy.configure
    asyncify = _real_dspy.asyncify
    __version__ = getattr(_real_dspy, "__version__", "unknown")
except Exception:  # pragma: no cover - exercised in lightweight test envs
    __version__ = "shim"

    class Signature:
        pass

    def InputField(**kwargs: Any) -> Any:
        return None

    def OutputField(**kwargs: Any) -> Any:
        return None

    class Module:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

        def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - interface only
            raise NotImplementedError

        def deepcopy(self):
            return copy.deepcopy(self)

        def save(self, path: str, save_program: bool = False) -> None:
            payload = {"type": self.__class__.__name__, "save_program": save_program}
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    class _PredictorBase:
        def __init__(self, signature: Any = None, **kwargs: Any):
            self.signature = signature
            self.kwargs = kwargs

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime guard only
            raise RuntimeError(
                "DSPy is not installed in this environment. Install dspy to execute the pipeline, or use the shipped sample artifacts and tests."
            )

    class Predict(_PredictorBase):
        pass

    class ChainOfThought(_PredictorBase):
        pass

    class Refine:
        def __init__(self, module: Any, N: int = 2, reward_fn: Callable[..., float] | None = None, threshold: float = 0.0):
            self.module = module
            self.N = N
            self.reward_fn = reward_fn
            self.threshold = threshold

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.module(*args, **kwargs)

    class BestOfN:
        def __init__(self, module: Any, N: int = 3, reward_fn: Callable[..., float] | None = None, threshold: float = 0.0):
            self.module = module
            self.N = N
            self.reward_fn = reward_fn
            self.threshold = threshold

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.module(*args, **kwargs)

    class Example(dict):
        def with_inputs(self, *inputs: str):
            self["_inputs"] = list(inputs)
            return self

    class LM:
        def __init__(self, model: str, **kwargs: Any):
            self.model = model
            self.kwargs = kwargs

    def configure(**kwargs: Any) -> None:
        return None

    def asyncify(module: Any):
        async def _runner(*args: Any, **kwargs: Any) -> Any:
            return module(*args, **kwargs)

        return _runner

    class _Optimizer:
        def __init__(self, *args: Any, **kwargs: Any):
            self.args = args
            self.kwargs = kwargs

        def compile(self, student: Any, trainset: Any = None, valset: Any = None, **kwargs: Any) -> Any:
            return student

    class BootstrapFewShot(_Optimizer):
        pass

    class GEPA(_Optimizer):
        pass

    class MIPROv2(_Optimizer):
        pass

# Allow `import tetraframe.dspy_compat as dspy` patterns.
dspy = SimpleNamespace(
    Signature=Signature,
    Module=Module,
    InputField=InputField,
    OutputField=OutputField,
    Predict=Predict,
    ChainOfThought=ChainOfThought,
    Refine=Refine,
    BestOfN=BestOfN,
    Example=Example,
    LM=LM,
    configure=configure,
    asyncify=asyncify,
    __version__=__version__,
)
