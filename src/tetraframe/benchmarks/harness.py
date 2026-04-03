from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from tetraframe.artifacts import CornerMode, CornerArtifact, TetraFrameRunArtifact
from tetraframe.metrics import benchmark_score_breakdown, benchmark_success
from tetraframe.pipeline import TetraFrameProgram, build_runtime_runner


class BenchmarkExample(BaseModel):
    example_id: str
    seed: str
    expected_primary_predicate_contains: list[str] = Field(default_factory=list)
    allowed_both_basis: list[str] = Field(default_factory=list)
    expected_neither_failure_modes: list[str] = Field(default_factory=list)
    expected_transformed_predicate_contains: list[str] = Field(default_factory=list)
    banned_transformed_phrases: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    example_id: str
    aggregate_score: float
    verification_score: float
    transformed_predicate: str
    both_basis: str
    neither_failure_mode: str
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    failed_expectations: list[str] = Field(default_factory=list)
    passed: bool = False


def load_benchmark_examples(path: str | Path) -> list[BenchmarkExample]:
    rows: list[BenchmarkExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(BenchmarkExample.model_validate_json(line))
    return rows


class BenchmarkHarness:
    def __init__(self, program: TetraFrameProgram, *, pass_threshold: float = 0.75):
        self.program = program
        self.pass_threshold = pass_threshold

    def _execute(self, seed: str) -> TetraFrameRunArtifact:
        runner = build_runtime_runner(self.program)
        if hasattr(runner, "run"):
            return runner.run(seed)
        return runner(seed)

    def run(self, examples: Iterable[BenchmarkExample]) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for example in examples:
            artifact = self._execute(example.seed)
            results.append(self._score_example(example, artifact))
        return results

    def _score_example(self, gold: BenchmarkExample, run: TetraFrameRunArtifact) -> BenchmarkResult:
        score = benchmark_success(run, gold)
        breakdown, failures = benchmark_score_breakdown(run, gold)
        return BenchmarkResult(
            example_id=gold.example_id,
            aggregate_score=score,
            verification_score=run.verification.aggregate_score,
            transformed_predicate=run.transformed_frame.transformed_predicate,
            both_basis=run.corners[CornerMode.BOTH].validity_basis_label,
            neither_failure_mode=run.corners[CornerMode.NEITHER].validity_basis_label,
            score_breakdown=breakdown,
            failed_expectations=failures,
            passed=score >= self.pass_threshold,
        )

    @staticmethod
    def summarize(results: list[BenchmarkResult], *, pass_threshold: float = 0.75) -> dict[str, Any]:
        if not results:
            return {"mean_score": 0.0, "count": 0, "pass_threshold": pass_threshold, "pass_rate": 0.0}
        mean_score = sum(r.aggregate_score for r in results) / len(results)
        verification_mean = sum(r.verification_score for r in results) / len(results)
        return {
            "count": len(results),
            "mean_score": round(mean_score, 3),
            "mean_verification_score": round(verification_mean, 3),
            "pass_threshold": round(pass_threshold, 3),
            "pass_rate": round(sum(r.aggregate_score >= pass_threshold for r in results) / len(results), 3),
            "passed_examples": sum(r.passed for r in results),
        }

    def run_ablation(self, examples: Iterable[BenchmarkExample], mode: str) -> list[BenchmarkResult]:
        original = self.program
        ablated = self._make_ablation(mode)
        try:
            self.program = ablated
            return self.run(examples)
        finally:
            self.program = original

    def _make_ablation(self, mode: str) -> TetraFrameProgram:
        ablated = self.program.deepcopy()
        if mode == "no_both":
            ablated.corner_generators[CornerMode.BOTH] = ablated.corner_generators[CornerMode.P]
        elif mode == "no_neither":
            ablated.corner_generators[CornerMode.NEITHER] = ablated.corner_generators[CornerMode.NOT_P]
        elif mode == "shared_context_corners":
            shared = ablated.corner_generators[CornerMode.P]
            ablated.corner_generators = {
                CornerMode.P: shared,
                CornerMode.NOT_P: shared,
                CornerMode.BOTH: shared,
                CornerMode.NEITHER: shared,
            }
        elif mode == "sequential_baseline":
            ablated.cfg.program.parallel_corners = False
            ablated.cfg.program.randomize_sequential_fallback_order = False
        elif mode == "no_cartography":
            ablated.cartograph = lambda corners: self.program.cartograph(corners).model_copy(update={
                "contradiction_map": [],
                "complementarity_map": [],
                "paradox_map": [],
                "invariant_map": [],
            })  # type: ignore[assignment]
        return ablated


def save_benchmark_report(path: str | Path, results: list[BenchmarkResult], *, pass_threshold: float = 0.75) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": BenchmarkHarness.summarize(results, pass_threshold=pass_threshold),
        "results": [r.model_dump() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
