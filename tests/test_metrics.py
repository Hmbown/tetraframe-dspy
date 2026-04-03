from tetraframe.artifacts import CornerMode
from tetraframe.metrics import (
    VerificationSuite,
    both_rigor_heuristic,
    corner_contamination_score,
    corner_divergence_score,
    non_averaging_transformation_score,
    slop_resistance,
)
from types import SimpleNamespace
from tests.fixtures import build_sample_run_artifact


def test_corner_divergence_is_reasonable_on_sample():
    run = build_sample_run_artifact()
    assert corner_divergence_score(run) >= 0.6


def test_contamination_score_is_high_on_sample():
    run = build_sample_run_artifact()
    assert corner_contamination_score(run) >= 0.9


def test_both_rigor_penalizes_compromise_language():
    run = build_sample_run_artifact()
    both = run.corners[CornerMode.BOTH].model_copy(
        update={"strongest_case": "This is a balanced approach and middle ground.", "validity_basis_label": "role_split"}
    )
    assert both_rigor_heuristic(both) < 0.75


def test_non_averaging_transformation_score_is_high_on_sample():
    run = build_sample_run_artifact()
    assert non_averaging_transformation_score(run.transformed_frame, run.corners) >= 0.8


def test_slop_resistance_is_high_on_sample():
    run = build_sample_run_artifact()
    assert slop_resistance(run) >= 0.75


def test_verification_suite_penalizes_compromise_and_slop():
    run = build_sample_run_artifact()
    corners = dict(run.corners)
    for mode, corner in corners.items():
        update = {
            "evidence_needs": ["consider various factors"],
            "tightened_language": "A nuanced and balanced approach helps consider multiple perspectives.",
        }
        if mode == CornerMode.BOTH:
            update["strongest_case"] = "This is a balanced approach and middle ground."
        corners[mode] = corner.model_copy(update=update)

    transformed = run.transformed_frame.model_copy(
        update={
            "transformed_predicate": "A balanced approach that splits the difference.",
            "transformed_frame": "On the one hand each corner matters, and on the other hand we should balance both.",
            "non_averaging_explanation": "This thoughtful and nuanced compromise keeps multiple views in play.",
            "operational_tests": [],
        }
    )
    degraded = run.model_copy(update={"corners": corners, "transformed_frame": transformed})

    suite = VerificationSuite()
    suite.corner_judge = lambda **kwargs: SimpleNamespace(score=0.2, rationale="compromise-like")  # type: ignore[assignment]
    suite.transform_judge = lambda **kwargs: SimpleNamespace(score=0.2, rationale="averaged frame")  # type: ignore[assignment]
    report = suite.verify(degraded)
    assert report.rigor_of_both.score < 0.78
    assert report.transformation_quality.score < 0.82
    assert report.slop_risk.score < 0.70
    assert any("both corner" in recommendation.lower() for recommendation in report.retry_recommendations)
    assert any("stage 4" in recommendation.lower() for recommendation in report.retry_recommendations)
