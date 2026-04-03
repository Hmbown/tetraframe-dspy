from __future__ import annotations

from typing import Literal

import tetraframe.dspy_compat as dspy


class SeedDistillSignature(dspy.Signature):
    """Distill a raw project seed into an operational project brief with candidate predicates.

    Extract stakes, constraints, unknowns, and hidden assumptions. Prefer operational predicates over slogans.
    Give a high frame risk score when the seed is overloaded, category-confused, or silently bundles multiple objectives.
    """

    raw_seed: str = dspy.InputField()
    normalized_project_seed: str = dspy.OutputField()
    stakes: list[str] = dspy.OutputField()
    constraints: list[str] = dspy.OutputField()
    unknowns: list[str] = dspy.OutputField()
    hidden_assumptions: list[str] = dspy.OutputField()
    candidate_predicates: list[str] = dspy.OutputField()
    frame_risk_score: float = dspy.OutputField()
    evaluation_criteria: list[str] = dspy.OutputField()
    novelty_criteria: list[str] = dspy.OutputField()


class SplitPredicateSignature(dspy.Signature):
    """Split candidate predicates into atomic operational predicates and reject malformed ones.

    Rewrite vague, overloaded, or malformed predicates into operational candidates when possible.
    Return rejected predicates with reasons and rewrite suggestions as JSON.
    """

    normalized_project_seed: str = dspy.InputField()
    candidate_predicates: list[str] = dspy.InputField()
    hidden_assumptions: list[str] = dspy.InputField()
    constraints: list[str] = dspy.InputField()
    atomic_predicates: list[str] = dspy.OutputField()
    rejected_predicates_json: str = dspy.OutputField(
        desc="JSON list of {text, reason, rewrite_suggestion}"
    )
    split_notes: list[str] = dspy.OutputField()


class ChoosePredicateSignature(dspy.Signature):
    """Choose the best primary predicate and optional sub-predicates for tetralemmatic reasoning.

    The primary predicate must be operational, falsifiable, and specific enough to support meaningful P, not-P, both,
    and neither corners.
    """

    normalized_project_seed: str = dspy.InputField()
    atomic_predicates: list[str] = dspy.InputField()
    constraints: list[str] = dspy.InputField()
    evaluation_criteria: list[str] = dspy.InputField()
    novelty_criteria: list[str] = dspy.InputField()
    primary_predicate: str = dspy.OutputField()
    primary_predicate_parts_json: str = dspy.OutputField(
        desc="JSON object with text, subject, relation, object, measurable_terms, scope, objective_type, operational_tests"
    )
    sub_predicates_json: str = dspy.OutputField(
        desc="JSON list of predicate spec objects for supporting sub-predicates"
    )
    rationale: str = dspy.OutputField()
    operationalization_notes: list[str] = dspy.OutputField()


# ---------------------------------------------------------------------------
# Corner generation signatures
#
# Each corner is generated AND hardened in a single pass: the model builds the
# position, attacks it internally, patches weaknesses, and reports confidence.
# ---------------------------------------------------------------------------

_CORNER_INPUT_FIELDS = dict(
    normalized_project_seed=dspy.InputField(),
    stakes=dspy.InputField(),
    constraints=dspy.InputField(),
    unknowns=dspy.InputField(),
    hidden_assumptions=dspy.InputField(),
    primary_predicate=dspy.InputField(),
    sub_predicates=dspy.InputField(),
    operationalization_notes=dspy.InputField(),
    evaluation_criteria=dspy.InputField(),
    novelty_criteria=dspy.InputField(),
    anti_collapse_hint=dspy.InputField(),
)

_CORNER_OUTPUT_FIELDS = dict(
    # Generation
    core_claim=dspy.OutputField(),
    assumptions=dspy.OutputField(),
    strongest_case=dspy.OutputField(),
    scope_conditions=dspy.OutputField(),
    falsifiers=dspy.OutputField(),
    evidence_needs=dspy.OutputField(),
    uncertainty=dspy.OutputField(),
    unique_signal=dspy.OutputField(),
    validity_basis_label=dspy.OutputField(),
    validity_basis_explanation=dspy.OutputField(),
    # Hardening (self-critique in the same pass)
    internal_attack=dspy.OutputField(),
    patched_claim=dspy.OutputField(),
    patched_assumptions=dspy.OutputField(),
    clarified_scope_conditions=dspy.OutputField(),
    confidence_boundaries=dspy.OutputField(),
    minimal_falsifiers=dspy.OutputField(),
    tightened_language=dspy.OutputField(),
    unresolved_weaknesses=dspy.OutputField(),
    confidence_score=dspy.OutputField(),
    still_valid_after_hardening=dspy.OutputField(),
    invalidity_reason=dspy.OutputField(),
)


class GenerateCornerPSignature(dspy.Signature):
    """Generate and harden the strongest clean affirmation of the current predicate.

    Do not mention other corners. Do not hedge into compromise. Make the case as if this corner must stand on its own.
    Then attack the position from within, patch the obvious weaknesses, tighten language, identify minimal falsifiers,
    and state confidence boundaries.
    The validity basis label must be exactly 'affirmation'.
    """

    normalized_project_seed: str = dspy.InputField()
    stakes: list[str] = dspy.InputField()
    constraints: list[str] = dspy.InputField()
    unknowns: list[str] = dspy.InputField()
    hidden_assumptions: list[str] = dspy.InputField()
    primary_predicate: str = dspy.InputField()
    sub_predicates: list[str] = dspy.InputField()
    operationalization_notes: list[str] = dspy.InputField()
    evaluation_criteria: list[str] = dspy.InputField()
    novelty_criteria: list[str] = dspy.InputField()
    anti_collapse_hint: str = dspy.InputField()
    core_claim: str = dspy.OutputField()
    assumptions: list[str] = dspy.OutputField()
    strongest_case: str = dspy.OutputField()
    scope_conditions: list[str] = dspy.OutputField()
    falsifiers: list[str] = dspy.OutputField()
    evidence_needs: list[str] = dspy.OutputField()
    uncertainty: str = dspy.OutputField()
    unique_signal: str = dspy.OutputField()
    validity_basis_label: str = dspy.OutputField()
    validity_basis_explanation: str = dspy.OutputField()
    internal_attack: list[str] = dspy.OutputField()
    patched_claim: str = dspy.OutputField()
    patched_assumptions: list[str] = dspy.OutputField()
    clarified_scope_conditions: list[str] = dspy.OutputField()
    confidence_boundaries: list[str] = dspy.OutputField()
    minimal_falsifiers: list[str] = dspy.OutputField()
    tightened_language: str = dspy.OutputField()
    unresolved_weaknesses: list[str] = dspy.OutputField()
    confidence_score: float = dspy.OutputField()
    still_valid_after_hardening: bool = dspy.OutputField()
    invalidity_reason: str = dspy.OutputField()


class GenerateCornerNotPSignature(dspy.Signature):
    """Generate and harden the strongest clean rejection, inversion, or dismantling of the current predicate.

    Do not merely negate surface wording. Attack the predicate's substance, causal logic, or usefulness.
    Then attack your own position from within, patch weaknesses, tighten language, and state confidence boundaries.
    The validity basis label must be exactly 'rejection'.
    """

    normalized_project_seed: str = dspy.InputField()
    stakes: list[str] = dspy.InputField()
    constraints: list[str] = dspy.InputField()
    unknowns: list[str] = dspy.InputField()
    hidden_assumptions: list[str] = dspy.InputField()
    primary_predicate: str = dspy.InputField()
    sub_predicates: list[str] = dspy.InputField()
    operationalization_notes: list[str] = dspy.InputField()
    evaluation_criteria: list[str] = dspy.InputField()
    novelty_criteria: list[str] = dspy.InputField()
    anti_collapse_hint: str = dspy.InputField()
    core_claim: str = dspy.OutputField()
    assumptions: list[str] = dspy.OutputField()
    strongest_case: str = dspy.OutputField()
    scope_conditions: list[str] = dspy.OutputField()
    falsifiers: list[str] = dspy.OutputField()
    evidence_needs: list[str] = dspy.OutputField()
    uncertainty: str = dspy.OutputField()
    unique_signal: str = dspy.OutputField()
    validity_basis_label: str = dspy.OutputField()
    validity_basis_explanation: str = dspy.OutputField()
    internal_attack: list[str] = dspy.OutputField()
    patched_claim: str = dspy.OutputField()
    patched_assumptions: list[str] = dspy.OutputField()
    clarified_scope_conditions: list[str] = dspy.OutputField()
    confidence_boundaries: list[str] = dspy.OutputField()
    minimal_falsifiers: list[str] = dspy.OutputField()
    tightened_language: str = dspy.OutputField()
    unresolved_weaknesses: list[str] = dspy.OutputField()
    confidence_score: float = dspy.OutputField()
    still_valid_after_hardening: bool = dspy.OutputField()
    invalidity_reason: str = dspy.OutputField()


class GenerateCornerBothSignature(dspy.Signature):
    """Generate and harden the strongest valid both corner.

    This is NOT compromise. It is only valid if P and not-P both hold due to one of:
    temporal_split, scale_split, role_split, ontology_split, context_split, layered_causality, admissible_paradox.
    You must explicitly choose one basis label from that list and show why both claims can co-hold.
    Then attack your own position from within, patch weaknesses, tighten language, and state confidence boundaries.
    """

    normalized_project_seed: str = dspy.InputField()
    stakes: list[str] = dspy.InputField()
    constraints: list[str] = dspy.InputField()
    unknowns: list[str] = dspy.InputField()
    hidden_assumptions: list[str] = dspy.InputField()
    primary_predicate: str = dspy.InputField()
    sub_predicates: list[str] = dspy.InputField()
    operationalization_notes: list[str] = dspy.InputField()
    evaluation_criteria: list[str] = dspy.InputField()
    novelty_criteria: list[str] = dspy.InputField()
    anti_collapse_hint: str = dspy.InputField()
    core_claim: str = dspy.OutputField()
    assumptions: list[str] = dspy.OutputField()
    strongest_case: str = dspy.OutputField()
    scope_conditions: list[str] = dspy.OutputField()
    falsifiers: list[str] = dspy.OutputField()
    evidence_needs: list[str] = dspy.OutputField()
    uncertainty: str = dspy.OutputField()
    unique_signal: str = dspy.OutputField()
    validity_basis_label: str = dspy.OutputField()
    validity_basis_explanation: str = dspy.OutputField()
    internal_attack: list[str] = dspy.OutputField()
    patched_claim: str = dspy.OutputField()
    patched_assumptions: list[str] = dspy.OutputField()
    clarified_scope_conditions: list[str] = dspy.OutputField()
    confidence_boundaries: list[str] = dspy.OutputField()
    minimal_falsifiers: list[str] = dspy.OutputField()
    tightened_language: str = dspy.OutputField()
    unresolved_weaknesses: list[str] = dspy.OutputField()
    confidence_score: float = dspy.OutputField()
    still_valid_after_hardening: bool = dspy.OutputField()
    invalidity_reason: str = dspy.OutputField()


class GenerateCornerNeitherSignature(dspy.Signature):
    """Generate and harden the strongest valid neither corner.

    This is NOT evasion. It is only valid if the original predicate is malformed, overloaded, category-confused,
    falsely binary, or otherwise misframed. You must choose one failure mode label from:
    category_error, false_binary, overloaded_predicate, missing_latent_variable, bad_ontology,
    ill_posed_objective, frame_collapse_under_scrutiny.
    You must also propose a replacement predicate or replacement frame.
    Then attack your own position from within, patch weaknesses, tighten language, and state confidence boundaries.
    """

    normalized_project_seed: str = dspy.InputField()
    stakes: list[str] = dspy.InputField()
    constraints: list[str] = dspy.InputField()
    unknowns: list[str] = dspy.InputField()
    hidden_assumptions: list[str] = dspy.InputField()
    primary_predicate: str = dspy.InputField()
    sub_predicates: list[str] = dspy.InputField()
    operationalization_notes: list[str] = dspy.InputField()
    evaluation_criteria: list[str] = dspy.InputField()
    novelty_criteria: list[str] = dspy.InputField()
    anti_collapse_hint: str = dspy.InputField()
    core_claim: str = dspy.OutputField()
    assumptions: list[str] = dspy.OutputField()
    strongest_case: str = dspy.OutputField()
    scope_conditions: list[str] = dspy.OutputField()
    falsifiers: list[str] = dspy.OutputField()
    evidence_needs: list[str] = dspy.OutputField()
    uncertainty: str = dspy.OutputField()
    unique_signal: str = dspy.OutputField()
    validity_basis_label: str = dspy.OutputField()
    validity_basis_explanation: str = dspy.OutputField()
    replacement_predicate: str = dspy.OutputField()
    replacement_frame: str = dspy.OutputField()
    internal_attack: list[str] = dspy.OutputField()
    patched_claim: str = dspy.OutputField()
    patched_assumptions: list[str] = dspy.OutputField()
    clarified_scope_conditions: list[str] = dspy.OutputField()
    confidence_boundaries: list[str] = dspy.OutputField()
    minimal_falsifiers: list[str] = dspy.OutputField()
    tightened_language: str = dspy.OutputField()
    unresolved_weaknesses: list[str] = dspy.OutputField()
    confidence_score: float = dspy.OutputField()
    still_valid_after_hardening: bool = dspy.OutputField()
    invalidity_reason: str = dspy.OutputField()


# Backward compat alias
HardenCornerSignature = None  # Removed — hardening is now part of corner generation


class PairwiseRelationSignature(dspy.Signature):
    """Compare two corners without inventing unsupported premises.

    Classify the relation as one of: opposition, contradiction, complementarity, paradox, dissolution,
    transformation, support, block. Explain what evidence would discriminate between the corners.
    """

    source_corner: str = dspy.InputField(desc="Serialized corner A")
    target_corner: str = dspy.InputField(desc="Serialized corner B")
    relation_type: str = dspy.OutputField()
    rationale: str = dspy.OutputField()
    evidence_discriminator: str = dspy.OutputField()
    reversible: bool = dspy.OutputField()
    invariant_tags: list[str] = dspy.OutputField()


class GlobalCartographySignature(dspy.Signature):
    """Map relationships between all four corners and produce arbiter judgments.

    Produce contradiction, complementarity, paradox, category-error, frame-validity, evidence-discriminator,
    invariant, reversible, irreversible, and structural-miss maps.
    Also reconstruct each corner sympathetically, distinguish dissolution from transformation,
    and write arbiter notes. Do not add unsupported premises.
    """

    corners_json: str = dspy.InputField()
    pairwise_relations_json: str = dspy.InputField()
    contradiction_map: list[str] = dspy.OutputField()
    complementarity_map: list[str] = dspy.OutputField()
    paradox_map: list[str] = dspy.OutputField()
    category_error_map: list[str] = dspy.OutputField()
    frame_validity_map: list[str] = dspy.OutputField()
    evidence_discriminator_map_json: str = dspy.OutputField(
        desc="JSON list of {discriminator, corner_favors, evidence_needed}"
    )
    invariant_map: list[str] = dspy.OutputField()
    reversible_implications: list[str] = dspy.OutputField()
    irreversible_implications: list[str] = dspy.OutputField()
    structural_miss_map_json: str = dspy.OutputField(desc="JSON object mapping corner to what it uniquely detects")
    # Arbiter outputs
    reconstructions_json: str = dspy.OutputField(
        desc="JSON list of {corner_mode, strongest_fair_restatement, unsupported_premises}"
    )
    dissolution: list[str] = dspy.OutputField()
    transformation: list[str] = dspy.OutputField()
    arbiter_notes: str = dspy.OutputField()


class TransformFrameSignature(dspy.Signature):
    """Produce a transformed framing P*.

    P* must not be an average or compromise. It must preserve what survives from P and not-P,
    reveal hidden structure from both, and dissolve false framing exposed by neither.
    """

    primary_predicate: str = dspy.InputField()
    corners_json: str = dspy.InputField()
    cartography_json: str = dspy.InputField()
    evaluation_criteria: list[str] = dspy.InputField()
    novelty_criteria: list[str] = dspy.InputField()
    transformed_predicate: str = dspy.OutputField()
    transformed_frame: str = dspy.OutputField()
    survivors_from_p: list[str] = dspy.OutputField()
    survivors_from_not_p: list[str] = dspy.OutputField()
    hidden_structure_from_both: list[str] = dspy.OutputField()
    dissolved_false_frame_from_neither: list[str] = dspy.OutputField()
    non_averaging_explanation: str = dspy.OutputField()
    operational_tests: list[str] = dspy.OutputField()
    boundary_conditions: list[str] = dspy.OutputField()
    failure_modes: list[str] = dspy.OutputField()
    confidence: float = dspy.OutputField()


class CornerRigorJudgeSignature(dspy.Signature):
    """Judge whether a both or neither corner is rigorous.

    Score 0.0 to 1.0. Penalize compromise masquerading as both and evasion masquerading as neither.
    """

    corner_mode: Literal["both", "neither"] = dspy.InputField()
    corner_json: str = dspy.InputField()
    score: float = dspy.OutputField()
    rationale: str = dspy.OutputField()


class TransformationJudgeSignature(dspy.Signature):
    """Judge whether a transformed frame is truly transformed rather than averaged.

    Score 0.0 to 1.0 and explain the main weakness.
    """

    transformed_frame_json: str = dspy.InputField()
    corners_json: str = dspy.InputField()
    score: float = dspy.OutputField()
    rationale: str = dspy.OutputField()
