from tetraframe.benchmarks.harness import BenchmarkExample, BenchmarkHarness
from tests.fixtures import build_sample_run_artifact


class StubProgram:
    def __call__(self, seed: str):
        return build_sample_run_artifact()

    def deepcopy(self):
        return self


def test_benchmark_harness_scores_example():
    harness = BenchmarkHarness(StubProgram())
    example = BenchmarkExample(
        example_id="x",
        seed="seed",
        expected_primary_predicate_contains=["anchoring", "independent", "transformation"],
        expected_transformed_predicate_contains=["exploration", "synthesis"],
        allowed_both_basis=["role_split"],
        expected_neither_failure_modes=["overloaded_predicate"],
    )
    results = harness.run([example])
    assert len(results) == 1
    assert results[0].aggregate_score >= 0.75
    assert results[0].score_breakdown["primary_predicate"] == 1.0
    assert results[0].passed is True


def test_benchmark_harness_penalizes_banned_compromise_language():
    class BannedPhraseProgram(StubProgram):
        def __call__(self, seed: str):
            run = build_sample_run_artifact()
            run.transformed_frame = run.transformed_frame.model_copy(
                update={"transformed_predicate": "A balanced approach that splits the difference."}
            )
            return run

    harness = BenchmarkHarness(BannedPhraseProgram())
    example = BenchmarkExample(
        example_id="banned",
        seed="seed",
        banned_transformed_phrases=["balanced approach", "split the difference"],
    )
    result = harness.run([example])[0]
    assert result.score_breakdown["banned_phrase_avoidance"] == 0.0
    assert "banned compromise language" in " ".join(result.failed_expectations)
