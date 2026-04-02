from tetraframe.artifacts import CornerDraftArtifact, CornerMode


def test_neither_requires_replacement_predicate_or_frame():
    try:
        CornerDraftArtifact(
            corner_mode=CornerMode.NEITHER,
            core_claim="Badly framed.",
            assumptions=["a"],
            strongest_case="The predicate is overloaded.",
            scope_conditions=["x"],
            falsifiers=["y"],
            evidence_needs=["z"],
            uncertainty="low",
            unique_signal="overload",
            validity_basis_label="overloaded_predicate",
            validity_basis_explanation="The predicate bundles incompatible objectives.",
        )
    except ValueError as exc:
        assert "replacement predicate or frame" in str(exc)
    else:
        raise AssertionError("Expected validation error for missing replacement predicate/frame")


def test_both_requires_valid_basis_label():
    try:
        CornerDraftArtifact(
            corner_mode=CornerMode.BOTH,
            core_claim="Both hold.",
            assumptions=["a"],
            strongest_case="Both somehow matter.",
            scope_conditions=["x"],
            falsifiers=["y"],
            evidence_needs=["z"],
            uncertainty="low",
            unique_signal="split",
            validity_basis_label="compromise",
            validity_basis_explanation="It splits the difference.",
        )
    except ValueError as exc:
        assert "both corner requires" in str(exc)
    else:
        raise AssertionError("Expected validation error for invalid both basis")
