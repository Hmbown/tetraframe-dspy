from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tetraframe.artifacts import StageTraceArtifact
from tetraframe.guards import stable_digest


class _StageTraceContext:
    def __init__(
        self,
        logger: "TraceLogger",
        run_id: str,
        stage_name: str,
        module_name: str,
        signature_name: str | None,
        attempt: int,
        visible_inputs: dict[str, Any],
        blocked_input_fields: list[str],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.logger = logger
        self.run_id = run_id
        self.stage_name = stage_name
        self.module_name = module_name
        self.signature_name = signature_name
        self.attempt = attempt
        self.visible_inputs = visible_inputs
        self.blocked_input_fields = list(blocked_input_fields)
        self.config = config or {}
        self.t0 = time.perf_counter()

    def close(
        self,
        output_payload: dict[str, Any],
        *,
        resolved_run_id: str | None = None,
        scores: dict[str, float] | None = None,
        warnings: list[str] | None = None,
        retry_reason: str = "",
    ) -> StageTraceArtifact:
        elapsed_ms = int((time.perf_counter() - self.t0) * 1000)
        run_id = resolved_run_id or self.run_id
        # Pull backend metadata from logger if available
        backend_info = self.logger._backend_info
        artifact = StageTraceArtifact(
            run_id=run_id,
            stage_name=self.stage_name,
            module_name=self.module_name,
            signature_name=self.signature_name,
            attempt=self.attempt,
            input_digest=stable_digest(self.visible_inputs),
            output_digest=stable_digest(output_payload),
            visible_input_fields=sorted(self.visible_inputs.keys()),
            blocked_input_fields=self.blocked_input_fields,
            config=self.config,
            latency_ms=elapsed_ms,
            warnings=warnings or [],
            scores=scores or {},
            retry_reason=retry_reason,
            backend_name=backend_info.get("name", ""),
            backend_kind=backend_info.get("kind", ""),
            backend_model=backend_info.get("model", ""),
            execution_mode=backend_info.get("execution_mode", ""),
            capability_warnings=backend_info.get("capability_warnings", []),
        )
        self.logger._append(artifact)
        return artifact


class TraceLogger:
    def __init__(self, trace_dir: str | Path, backend_info: dict[str, Any] | None = None):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._backend_info: dict[str, Any] = backend_info or {}

    def set_backend_info(
        self,
        *,
        name: str = "",
        kind: str = "",
        model: str = "",
        execution_mode: str = "",
        capability_warnings: list[str] | None = None,
    ) -> None:
        """Update backend metadata that will be recorded in all subsequent traces."""
        self._backend_info = {
            "name": name,
            "kind": kind,
            "model": model,
            "execution_mode": execution_mode,
            "capability_warnings": capability_warnings or [],
        }

    def stage(
        self,
        run_id: str,
        stage_name: str,
        module_name: str,
        signature_name: str | None,
        attempt: int,
        visible_inputs: dict[str, Any],
        blocked_input_fields: list[str],
        config: dict[str, Any] | None = None,
    ) -> _StageTraceContext:
        return _StageTraceContext(
            logger=self,
            run_id=run_id,
            stage_name=stage_name,
            module_name=module_name,
            signature_name=signature_name,
            attempt=attempt,
            visible_inputs=visible_inputs,
            blocked_input_fields=blocked_input_fields,
            config=config,
        )

    def _append(self, artifact: StageTraceArtifact) -> None:
        path = self.trace_dir / f"{artifact.run_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact.model_dump(), ensure_ascii=False) + "\n")
