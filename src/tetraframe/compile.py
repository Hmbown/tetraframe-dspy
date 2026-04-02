from __future__ import annotations

from pathlib import Path
from typing import Iterable

import tetraframe.dspy_compat as dspy
from tetraframe.dspy_compat import BootstrapFewShot, GEPA, MIPROv2

from tetraframe.artifacts import CornerMode
from tetraframe.benchmarks.harness import BenchmarkExample, BenchmarkHarness, load_benchmark_examples
from tetraframe.config import RootConfig
from tetraframe.metrics import corner_metric, gepa_feedback_metric, predicate_metric, stage0_metric
from tetraframe.pipeline import TetraFrameProgram


def as_dspy_examples(examples: Iterable[BenchmarkExample]) -> list[dspy.Example]:
    rows = []
    for example in examples:
        rows.append(
            dspy.Example(
                raw_seed=example.seed,
                expected_primary_predicate_contains=example.expected_primary_predicate_contains,
                allowed_both_basis=example.allowed_both_basis,
                expected_neither_failure_modes=example.expected_neither_failure_modes,
                expected_transformed_predicate_contains=example.expected_transformed_predicate_contains,
                banned_transformed_phrases=example.banned_transformed_phrases,
            ).with_inputs("raw_seed")
        )
    return rows


def freeze(module: dspy.Module) -> None:
    module._compiled = True  # DSPy optimizers use this flag to avoid mutating frozen modules.


class Compiler:
    """Progressive compilation strategy.

    1. Compile seed distillation and predicate selection first.
    2. Compile corners individually.
    3. Compile transformer.
    4. Run GEPA over the whole program using verification traces and textual feedback.
    """

    def __init__(self, cfg: RootConfig):
        self.cfg = cfg

    def build_lms(self) -> tuple:
        from tetraframe.backends.factory import build_dspy_lm, build_reflection_lm
        runtime_lm = build_dspy_lm(self.cfg)
        reflection_lm = build_reflection_lm(self.cfg)
        return runtime_lm, reflection_lm

    def compile(self, program: TetraFrameProgram, train_path: str | Path, dev_path: str | Path) -> TetraFrameProgram:
        trainset = as_dspy_examples(load_benchmark_examples(train_path))
        devset = as_dspy_examples(load_benchmark_examples(dev_path))

        runtime_lm, reflection_lm = self.build_lms()
        dspy.configure(lm=runtime_lm)

        # Stage 0: SeedDistill
        seed_compiler = BootstrapFewShot(metric=stage0_metric, max_labeled_demos=4, max_bootstrapped_demos=2)
        program.seed_distill = seed_compiler.compile(student=program.seed_distill.deepcopy(), trainset=trainset)
        freeze(program.seed_distill)

        # Stage 1: PredicateSelect
        predicate_compiler = MIPROv2(metric=predicate_metric, auto="light")
        program.predicate_select = predicate_compiler.compile(
            student=program.predicate_select.deepcopy(),
            trainset=trainset,
            valset=devset,
            max_labeled_demos=4,
            max_bootstrapped_demos=2,
        )
        freeze(program.predicate_select)

        # Stage 2: Corner generators. Compile nuanced generators more aggressively.
        for mode, module in program.corner_generators.items():
            if mode in {CornerMode.BOTH, CornerMode.NEITHER}:
                optimizer = MIPROv2(metric=corner_metric, auto="medium")
                compiled = optimizer.compile(
                    student=module.deepcopy(),
                    trainset=trainset,
                    valset=devset,
                    max_labeled_demos=4,
                    max_bootstrapped_demos=3,
                )
            else:
                optimizer = BootstrapFewShot(metric=corner_metric, max_labeled_demos=4, max_bootstrapped_demos=2)
                compiled = optimizer.compile(student=module.deepcopy(), trainset=trainset)
            program.corner_generators[mode] = compiled
            freeze(program.corner_generators[mode])

        # Stage 6: Transformation. Use GEPA because it can consume textual feedback tied to traces.
        transformer_optimizer = GEPA(
            metric=gepa_feedback_metric,
            auto="light",
            reflection_lm=reflection_lm,
            track_stats=True,
        )
        program.transform = transformer_optimizer.compile(
            student=program.transform.deepcopy(),
            trainset=trainset,
            valset=devset,
        )
        freeze(program.transform)

        # Program-level GEPA. This tunes remaining prompts jointly while keeping early modules frozen.
        program_optimizer = GEPA(
            metric=gepa_feedback_metric,
            auto="light",
            reflection_lm=reflection_lm,
            track_stats=True,
        )
        compiled_program = program_optimizer.compile(student=program.deepcopy(), trainset=trainset, valset=devset)
        return compiled_program

    def evaluate(self, program: TetraFrameProgram, dataset_path: str | Path) -> dict[str, object]:
        examples = load_benchmark_examples(dataset_path)
        harness = BenchmarkHarness(program, pass_threshold=self.cfg.benchmark.pass_threshold)
        results = harness.run(examples)
        return {
            "dataset_path": str(dataset_path),
            "summary": harness.summarize(results, pass_threshold=self.cfg.benchmark.pass_threshold),
            "results": [result.model_dump() for result in results],
        }
