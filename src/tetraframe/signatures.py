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


class GenerateCornerPSignature(dspy.Signature):
    """Generate the strongest clean affirmation of the current predicate.

    Do not mention other corners. Do not hedge into compromise. Make the case as if this corner must stand on its own.
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


class GenerateCornerNotPSignature(dspy.Signature):
    """Generate the strongest clean rejection, inversion, or dismantling of the current predicate.

    Do not merely negate surface wording. Attack the predicate's substance, causal logic, or usefulness.
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


class GenerateCornerBothSignature(dspy.Signature):
    """Generate the strongest valid both corner.

    This is NOT compromise. It is only valid if P and not-P both hold due to one of:
    temporal_split, scale_split, role_split, ontology_split, context_split, layered_causality, admissible_paradox.
    You must explicitly choose one basis label from that list and show why both claims can co-hold.
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


class GenerateCornerNeitherSignature(dspy.Signature):
    """Generate the strongest valid neither corner.

    This is NOT evasion. It is only valid if the original predicate is malformed, overloaded, category-confused,
    falsely binary, or otherwise misframed. You must choose one failure mode label from:
    category_error, false_binary, overloaded_predicate, missing_latent_variable, bad_ontology,
    ill_posed_objective, frame_collapse_under_scrutiny.
    You must also propose a replacement predicate or replacement frame.
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


class HardenCornerSignature(dspy.Signature):
    """Harden a corner in isolation.

    Attack the corner from within, patch the obvious weaknesses, tighten language, identify minimal falsifiers,
    and state confidence boundaries without referring to any other corner.
    """

    corner_mode: Literal["P", "not-P", "both", "neither"] = dspy.InputField()
    normalized_project_seed: str = dspy.InputField()
    primary_predicate: str = dspy.InputField()
    core_claim: str = dspy.InputField()
    assumptions: list[str] = dspy.InputField()
    strongest_case: str = dspy.InputField()
    scope_conditions: list[str] = dspy.InputField()
    falsifiers: list[str] = dspy.InputField()
    evidence_needs: list[str] = dspy.InputField()
    uncertainty: str = dspy.InputField()
    unique_signal: str = dspy.InputField()
    validity_basis_label: str = dspy.InputField()
    validity_basis_explanation: str = dspy.InputField()
    replacement_predicate: str = dspy.InputField()
    replacement_frame: str = dspy.InputField()
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


class PairwiseRelationSignature(dspy.Signature):
    """Compare two hardened corners without inventing unsupported premises.

    Classify the relation as one of: opposition, contradiction, complementarity, paradox, dissolution,
    transformation, support, block. Explain what evidence would discriminate between the corners.
    """

    source_corner: str = dspy.InputField(desc="Serialized hardened corner A")
    target_corner: str = dspy.InputField(desc="Serialized hardened corner B")
    relation_type: str = dspy.OutputField()
    rationale: str = dspy.OutputField()
    evidence_discriminator: str = dspy.OutputField()
    reversible: bool = dspy.OutputField()
    invariant_tags: list[str] = dspy.OutputField()


class GlobalCartographySignature(dspy.Signature):
    """Turn hardened corners and pairwise relations into explicit cartography.

    Produce contradiction, complementarity, paradox, category-error, frame-validity, evidence-discriminator,
    invariant, reversible, irreversible, and structural-miss maps.
    """

    hardened_corners_json: str = dspy.InputField()
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


class FourCornerArbiterSignature(dspy.Signature):
    """Reconstruct all four corners sympathetically from existing artifacts only.

    Distinguish opposition, contradiction, complementarity, paradox, dissolution, and transformation.
    Do not add unsupported premises.
    """

    hardened_corners_json: str = dspy.InputField()
    cartography_json: str = dspy.InputField()
    reconstructions_json: str = dspy.OutputField(
        desc="JSON list of {corner_mode, strongest_fair_restatement, unsupported_premises}"
    )
    opposition: list[str] = dspy.OutputField()
    contradiction: list[str] = dspy.OutputField()
    complementarity: list[str] = dspy.OutputField()
    paradox: list[str] = dspy.OutputField()
    dissolution: list[str] = dspy.OutputField()
    transformation: list[str] = dspy.OutputField()
    arbiter_notes: str = dspy.OutputField()


class TransformFrameSignature(dspy.Signature):
    """Produce a transformed framing P*.

    P* must not be an average or compromise. It must preserve what survives from P and not-P,
    reveal hidden structure from both, and dissolve false framing exposed by neither.
    """

    primary_predicate: str = dspy.InputField()
    hardened_corners_json: str = dspy.InputField()
    cartography_json: str = dspy.InputField()
    arbiter_json: str = dspy.InputField()
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


class AdaptWritingSignature(dspy.Signature):
    """Adapt a transformed frame into a writing artifact."""

    transformed_predicate: str = dspy.InputField()
    transformed_frame: str = dspy.InputField()
    cartography_summary: str = dspy.InputField()
    central_claim: str = dspy.OutputField()
    rival_readings: list[str] = dspy.OutputField()
    tension_map: list[str] = dspy.OutputField()
    outline: list[str] = dspy.OutputField()
    voice_options: list[str] = dspy.OutputField()
    stress_test: list[str] = dspy.OutputField()
    revision_plan: list[str] = dspy.OutputField()


class AdaptCodingSignature(dspy.Signature):
    """Adapt a transformed frame into a coding artifact."""

    transformed_predicate: str = dspy.InputField()
    transformed_frame: str = dspy.InputField()
    cartography_summary: str = dspy.InputField()
    architecture: str = dspy.OutputField()
    modules: list[str] = dspy.OutputField()
    interfaces: list[str] = dspy.OutputField()
    state_model: str = dspy.OutputField()
    verification_loop: list[str] = dspy.OutputField()
    tests: list[str] = dspy.OutputField()
    failure_modes: list[str] = dspy.OutputField()
    iteration_plan: list[str] = dspy.OutputField()


class AdaptResearchSignature(dspy.Signature):
    """Adapt a transformed frame into a research artifact."""

    transformed_predicate: str = dspy.InputField()
    transformed_frame: str = dspy.InputField()
    cartography_summary: str = dspy.InputField()
    competing_hypotheses: list[str] = dspy.OutputField()
    discriminating_experiments: list[str] = dspy.OutputField()
    evidence_agenda: list[str] = dspy.OutputField()
    confound_map: list[str] = dspy.OutputField()
    interpretation_grid: list[str] = dspy.OutputField()
    next_step_program: list[str] = dspy.OutputField()


class AdaptPlanningSignature(dspy.Signature):
    """Adapt a transformed frame into a planning artifact."""

    transformed_predicate: str = dspy.InputField()
    transformed_frame: str = dspy.InputField()
    cartography_summary: str = dspy.InputField()
    option_set: list[str] = dspy.OutputField()
    leverage_points: list[str] = dspy.OutputField()
    decision_thresholds: list[str] = dspy.OutputField()
    scenario_map: list[str] = dspy.OutputField()
    reversibility_map: list[str] = dspy.OutputField()
    execution_phases: list[str] = dspy.OutputField()
    monitoring_plan: list[str] = dspy.OutputField()


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
    hardened_corners_json: str = dspy.InputField()
    score: float = dspy.OutputField()
    rationale: str = dspy.OutputField()


class DomainUsefulnessJudgeSignature(dspy.Signature):
    """Judge whether a domain adaptation artifact is useful and specific.

    Score 0.0 to 1.0. Penalize generic prose, missing interfaces, weak experiments, and fuzzy thresholds.
    """

    domain: str = dspy.InputField()
    artifact_json: str = dspy.InputField()
    transformed_frame_json: str = dspy.InputField()
    score: float = dspy.OutputField()
    rationale: str = dspy.OutputField()
