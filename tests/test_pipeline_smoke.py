from tests.fixtures import build_sample_run_artifact


def test_sample_run_fixture_contains_all_core_sections():
    run = build_sample_run_artifact()
    assert run.distilled_seed.normalized_project_seed
    assert run.predicate_selection.primary_predicate.text
    assert len(run.corner_drafts) == 4
    assert len(run.hardened_corners) == 4
    assert run.cartography.invariant_map
    assert run.arbiter.transformation
    assert run.transformed_frame.transformed_predicate
    assert run.coding.interfaces
    assert run.research.discriminating_experiments
    assert run.planning.decision_thresholds
    assert run.verification.aggregate_score >= 0.8
