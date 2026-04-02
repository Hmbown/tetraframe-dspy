from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import re

from pydantic import BaseModel, Field, field_validator, model_validator


class _CoercingBaseModel(BaseModel):
    """BaseModel that coerces scalar strings into single-item lists for list[str] fields.

    CLI backends (Claude Code, Codex, OpenCode) don't enforce structured JSON,
    so DSPy may parse list fields as plain strings.  This validator normalises
    them before Pydantic's strict type check runs.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_str_to_list(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        for field_name, field_info in cls.model_fields.items():
            annotation = field_info.annotation
            # Detect list[str] annotations (including Optional wrappers)
            origin = getattr(annotation, "__origin__", None)
            if origin is list and field_name in values:
                v = values[field_name]
                if isinstance(v, str):
                    values[field_name] = [v] if v.strip() else []
                elif isinstance(v, list):
                    # Coerce dict items to strings for list[str] fields
                    coerced = []
                    for item in v:
                        if isinstance(item, str):
                            coerced.append(item)
                        elif isinstance(item, dict):
                            # Extract a readable string from dict values
                            parts = []
                            for k, val in item.items():
                                if isinstance(val, str) and val.strip():
                                    parts.append(val)
                            coerced.append(": ".join(parts) if parts else str(item))
                        elif item is not None:
                            coerced.append(str(item))
                    values[field_name] = coerced
            # Coerce None -> "" for str fields
            if origin is None and field_name in values:
                if values[field_name] is None and (annotation is str or annotation == str):
                    values[field_name] = ""
        return values


def _fuzzy_enum_match(value: str, allowed: set[str]) -> str | None:
    """Try to match a free-form string to a valid enum value."""
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in allowed:
        return normalized
    for candidate in allowed:
        if candidate in normalized:
            return candidate
    return None


class CornerMode(str, Enum):
    P = "P"
    NOT_P = "not-P"
    BOTH = "both"
    NEITHER = "neither"


class BothBasis(str, Enum):
    TEMPORAL_SPLIT = "temporal_split"
    SCALE_SPLIT = "scale_split"
    ROLE_SPLIT = "role_split"
    ONTOLOGY_SPLIT = "ontology_split"
    CONTEXT_SPLIT = "context_split"
    LAYERED_CAUSALITY = "layered_causality"
    ADMISSIBLE_PARADOX = "admissible_paradox"


class NeitherFailureMode(str, Enum):
    CATEGORY_ERROR = "category_error"
    FALSE_BINARY = "false_binary"
    OVERLOADED_PREDICATE = "overloaded_predicate"
    MISSING_LATENT_VARIABLE = "missing_latent_variable"
    BAD_ONTOLOGY = "bad_ontology"
    ILL_POSED_OBJECTIVE = "ill_posed_objective"
    FRAME_COLLAPSE_UNDER_SCRUTINY = "frame_collapse_under_scrutiny"


class RelationType(str, Enum):
    OPPOSITION = "opposition"
    CONTRADICTION = "contradiction"
    COMPLEMENTARITY = "complementarity"
    PARADOX = "paradox"
    DISSOLUTION = "dissolution"
    TRANSFORMATION = "transformation"
    SUPPORT = "support"
    BLOCK = "block"


class PredicateSpec(_CoercingBaseModel):
    text: str
    subject: str = ""
    relation: str = ""
    object: str = ""
    measurable_terms: list[str] = Field(default_factory=list)
    scope: list[str] = Field(default_factory=list)
    objective_type: str = ""
    operational_tests: list[str] = Field(default_factory=list)


class RejectedPredicate(_CoercingBaseModel):
    text: str
    reason: str
    rewrite_suggestion: str = ""


class DistilledSeedArtifact(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    raw_seed: str
    normalized_project_seed: str
    stakes: list[str]
    constraints: list[str]
    unknowns: list[str]
    hidden_assumptions: list[str]
    candidate_predicates: list[str]
    frame_risk_score: float
    evaluation_criteria: list[str]
    novelty_criteria: list[str]


class PredicateSelectionArtifact(BaseModel):
    primary_predicate: PredicateSpec
    sub_predicates: list[PredicateSpec] = Field(default_factory=list)
    rejected_predicates: list[RejectedPredicate] = Field(default_factory=list)
    rationale: str
    operationalization_notes: list[str]


class CornerInputView(BaseModel):
    run_id: str
    normalized_project_seed: str
    stakes: list[str]
    constraints: list[str]
    unknowns: list[str]
    hidden_assumptions: list[str]
    primary_predicate: str
    sub_predicates: list[str]
    operationalization_notes: list[str]
    evaluation_criteria: list[str]
    novelty_criteria: list[str]
    corner_mode: CornerMode
    corner_contract: str
    anti_collapse_hint: str = ""


class CornerDraftArtifact(BaseModel):
    corner_mode: CornerMode
    core_claim: str
    assumptions: list[str]
    strongest_case: str
    scope_conditions: list[str]
    falsifiers: list[str]
    evidence_needs: list[str]
    uncertainty: str
    unique_signal: str
    validity_basis_label: str
    validity_basis_explanation: str
    replacement_predicate: str = ""
    replacement_frame: str = ""

    @model_validator(mode="after")
    def validate_corner_specifics(self) -> "CornerDraftArtifact":
        if self.corner_mode == CornerMode.BOTH:
            allowed = {x.value for x in BothBasis}
            if self.validity_basis_label not in allowed:
                # Try fuzzy match before raising
                matched = _fuzzy_enum_match(self.validity_basis_label, allowed)
                if matched:
                    self.validity_basis_label = matched
                else:
                    raise ValueError(f"both corner requires one of {sorted(allowed)}")
        if self.corner_mode == CornerMode.NEITHER:
            allowed = {x.value for x in NeitherFailureMode}
            if self.validity_basis_label not in allowed:
                matched = _fuzzy_enum_match(self.validity_basis_label, allowed)
                if matched:
                    self.validity_basis_label = matched
                else:
                    raise ValueError(f"neither corner requires one of {sorted(allowed)}")
            if not self.replacement_predicate.strip() and not self.replacement_frame.strip():
                raise ValueError("neither corner requires a replacement predicate or frame")
        return self


class HardenedCornerArtifact(CornerDraftArtifact):
    internal_attack: list[str]
    patched_claim: str
    patched_assumptions: list[str]
    clarified_scope_conditions: list[str]
    confidence_boundaries: list[str]
    minimal_falsifiers: list[str]
    tightened_language: str
    unresolved_weaknesses: list[str]
    confidence_score: float
    still_valid_after_hardening: bool = True
    invalidity_reason: str = ""


class PairwiseRelationArtifact(BaseModel):
    source_corner: CornerMode
    target_corner: CornerMode
    relation_type: RelationType
    rationale: str
    evidence_discriminator: str
    reversible: bool
    invariant_tags: list[str] = Field(default_factory=list)


class EvidenceDiscriminatorArtifact(_CoercingBaseModel):
    discriminator: str
    corner_favors: list[CornerMode]
    evidence_needed: list[str]

    @field_validator("corner_favors", mode="before")
    @classmethod
    def _parse_corner_favors(cls, v: Any) -> list[str]:
        """Extract valid CornerMode values from free-form LLM output."""
        valid = {m.value for m in CornerMode}
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        result: list[str] = []
        for item in v:
            s = str(item).strip()
            if s in valid:
                result.append(s)
            else:
                # Try to extract valid corner names from free-form text
                for mode_val in valid:
                    if mode_val in s:
                        result.append(mode_val)
        return result if result else ["P"]  # Fallback to P if nothing valid found


class CartographyArtifact(BaseModel):
    pairwise_relations: list[PairwiseRelationArtifact]
    contradiction_map: list[str]
    complementarity_map: list[str]
    paradox_map: list[str]
    category_error_map: list[str]
    frame_validity_map: list[str]
    evidence_discriminator_map: list[EvidenceDiscriminatorArtifact]
    invariant_map: list[str]
    reversible_implications: list[str]
    irreversible_implications: list[str]
    structural_miss_map: dict[str, str]


class CornerReconstructionArtifact(_CoercingBaseModel):
    corner_mode: CornerMode
    strongest_fair_restatement: str
    unsupported_premises: list[str] = Field(default_factory=list)

    @field_validator("corner_mode", mode="before")
    @classmethod
    def _parse_corner_mode(cls, v: Any) -> str:
        valid = {m.value for m in CornerMode}
        s = str(v).strip()
        if s in valid:
            return s
        for mode_val in valid:
            if mode_val in s:
                return mode_val
        return "P"


class ArbiterArtifact(BaseModel):
    reconstructions: list[CornerReconstructionArtifact]
    opposition: list[str]
    contradiction: list[str]
    complementarity: list[str]
    paradox: list[str]
    dissolution: list[str]
    transformation: list[str]
    arbiter_notes: str


class TransformedFrameArtifact(BaseModel):
    transformed_predicate: str
    transformed_frame: str
    survivors_from_p: list[str]
    survivors_from_not_p: list[str]
    hidden_structure_from_both: list[str]
    dissolved_false_frame_from_neither: list[str]
    non_averaging_explanation: str
    operational_tests: list[str]
    boundary_conditions: list[str]
    failure_modes: list[str]
    confidence: float


class WritingAdapterArtifact(BaseModel):
    central_claim: str
    rival_readings: list[str]
    tension_map: list[str]
    outline: list[str]
    voice_options: list[str]
    stress_test: list[str]
    revision_plan: list[str]


class CodingAdapterArtifact(BaseModel):
    architecture: str
    modules: list[str]
    interfaces: list[str]
    state_model: str
    verification_loop: list[str]
    tests: list[str]
    failure_modes: list[str]
    iteration_plan: list[str]


class ResearchAdapterArtifact(BaseModel):
    competing_hypotheses: list[str]
    discriminating_experiments: list[str]
    evidence_agenda: list[str]
    confound_map: list[str]
    interpretation_grid: list[str]
    next_step_program: list[str]


class PlanningAdapterArtifact(BaseModel):
    option_set: list[str]
    leverage_points: list[str]
    decision_thresholds: list[str]
    scenario_map: list[str]
    reversibility_map: list[str]
    execution_phases: list[str]
    monitoring_plan: list[str]


class VerificationMetricArtifact(BaseModel):
    score: float
    rationale: str
    passed: bool


class VerificationReportArtifact(BaseModel):
    branch_independence: VerificationMetricArtifact
    divergence_quality: VerificationMetricArtifact
    rigor_of_both: VerificationMetricArtifact
    rigor_of_neither: VerificationMetricArtifact
    contradiction_honesty: VerificationMetricArtifact
    transformation_quality: VerificationMetricArtifact
    actionability: VerificationMetricArtifact
    robustness: VerificationMetricArtifact
    fake_novelty_risk: VerificationMetricArtifact
    slop_risk: VerificationMetricArtifact
    domain_adapter_feedback: VerificationMetricArtifact = Field(
        default_factory=lambda: VerificationMetricArtifact(score=0.0, rationale="not computed", passed=False)
    )
    aggregate_score: float
    retry_recommendations: list[str] = Field(default_factory=list)


class StageTraceArtifact(BaseModel):
    run_id: str
    stage_name: str
    module_name: str
    signature_name: str | None = None
    attempt: int = 1
    input_digest: str = ""
    output_digest: str = ""
    visible_input_fields: list[str] = Field(default_factory=list)
    blocked_input_fields: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None
    warnings: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    retry_reason: str = ""
    # Backend metadata (populated when backend info is available)
    backend_name: str = ""
    backend_kind: str = ""
    backend_model: str = ""
    execution_mode: str = ""  # "direct", "proxy", "cli"
    capability_warnings: list[str] = Field(default_factory=list)


class TetraFrameRunArtifact(BaseModel):
    run_id: str
    distilled_seed: DistilledSeedArtifact
    predicate_selection: PredicateSelectionArtifact
    corner_inputs: dict[CornerMode, CornerInputView]
    corner_drafts: dict[CornerMode, CornerDraftArtifact]
    hardened_corners: dict[CornerMode, HardenedCornerArtifact]
    cartography: CartographyArtifact
    arbiter: ArbiterArtifact
    transformed_frame: TransformedFrameArtifact
    writing: WritingAdapterArtifact
    coding: CodingAdapterArtifact
    research: ResearchAdapterArtifact
    planning: PlanningAdapterArtifact
    verification: VerificationReportArtifact | None = None
    traces: list[StageTraceArtifact] = Field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
