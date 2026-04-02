from tetraframe.artifacts import CornerMode
from tetraframe.guards import (
    BLOCKED_CORNER_FIELDS,
    assert_corner_view_isolation,
    build_anti_collapse_hint,
    detect_near_duplicate_corners,
    make_corner_input_view,
)
from tests.fixtures import build_sample_run_artifact


def test_corner_view_contains_only_allowed_fields():
    run = build_sample_run_artifact()
    view = make_corner_input_view(run.distilled_seed, run.predicate_selection, CornerMode.P)
    assert_corner_view_isolation(view)
    payload = view.model_dump()
    for blocked in BLOCKED_CORNER_FIELDS:
        assert blocked not in payload
    assert payload["primary_predicate"] == run.predicate_selection.primary_predicate.text


def test_duplicate_detection_flags_collapsed_corners():
    run = build_sample_run_artifact()
    p = run.corner_drafts[CornerMode.P].model_copy(update={"core_claim": "same repeated claim"})
    n = run.corner_drafts[CornerMode.NOT_P].model_copy(update={"core_claim": "same repeated claim"})
    duplicates = detect_near_duplicate_corners(
        {
            CornerMode.P: p,
            CornerMode.NOT_P: n,
            CornerMode.BOTH: run.corner_drafts[CornerMode.BOTH],
            CornerMode.NEITHER: run.corner_drafts[CornerMode.NEITHER],
        },
        run.distilled_seed.normalized_project_seed,
        similarity_threshold=0.70,
    )
    assert duplicates
    hint = build_anti_collapse_hint(CornerMode.NOT_P, duplicates)
    assert "deeper failure" in hint.lower()
