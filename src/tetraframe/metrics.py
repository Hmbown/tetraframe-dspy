from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import tetraframe.dspy_compat as dspy

from tetraframe.artifacts import (
    CartographyArtifact,
    CornerMode,
    HardenedCornerArtifact,
    TetraFrameRunArtifact,
    TransformedFrameArtifact,
    VerificationMetricArtifact,
    VerificationReportArtifact,
)
from tetraframe.guards import incompatible_pairs, pairwise_similarity, residual_tokens
from tetraframe.signatures import (
    CornerRigorJudgeSignature,
    DomainUsefulnessJudgeSignature,
    TransformationJudgeSignature,
)


MUSH_WORDS = {
    "balanced",
    "nuanced",
    "important",
    "helpful",
    "complex",
    "thoughtful",
    "consider",
    "various",
    "multiple",
}

COMPROMISE_MARKERS = {
    "middle ground",
    "balance both",
    "balanced approach",
    "split the difference",
    "on the one hand",
    "on the other hand",
}

OBSERVABLE_MARKERS = {
    "experiment",
    "benchmark",
    "metric",
    "trace",
    "latency",
    "ab test",
    "counterexample",
    "failure rate",
    "user study",
    "log",
    "dataset",
    "measurement",
}


@dataclass
class ScoreWithReason:
    score: float
    rationale: str


class VerificationSuite:
    """Hybrid verifier: heuristics first, judge-model only where needed."""

    def __init__(self):
        self.corner_judge = dspy.Predict(CornerRigorJudgeSignature)
        self.transform_judge = dspy.Predict(TransformationJudgeSignature)
        self.domain_judge = dspy.Predict(DomainUsefulnessJudgeSignature)

    def verify(self, run: TetraFrameRunArtifact) -> VerificationReportArtifact:
        branch_independence = self.branch_independence(run)
        divergence_quality = self.divergence_quality(run)
        rigor_of_both = self.rigor_of_both(run.hardened_corners[CornerMode.BOTH])
        rigor_of_neither = self.rigor_of_neither(run.hardened_corners[CornerMode.NEITHER])
        contradiction_honesty = self.contradiction_honesty(run)
        transformation_quality = self.transformation_quality(run.transformed_frame, run.hardened_corners)
        actionability = self.actionability(run)
        robustness = self.robustness(run)
        fake_novelty_risk = self.fake_novelty_risk(run)
        slop_risk = self.slop_risk(run)
        adapter_feedback = self.domain_adapter_feedback(run)

        metrics = {
            "branch_independence": branch_independence,
            "divergence_quality": divergence_quality,
            "rigor_of_both": rigor_of_both,
            "rigor_of_neither": rigor_of_neither,
            "contradiction_honesty": contradiction_honesty,
            "transformation_quality": transformation_quality,
            "actionability": actionability,
            "robustness": robustness,
            "fake_novelty_risk": fake_novelty_risk,
            "slop_risk": slop_risk,
            "domain_adapter_feedback": adapter_feedback,
        }
        aggregate = round(sum(m.score for m in metrics.values()) / len(metrics), 3)

        retry_recommendations: list[str] = []
        if branch_independence.score < 0.90:
            retry_recommendations.append("Retry stage 2 with stronger anti-collapse hints and fresh rollout IDs.")
        if rigor_of_both.score < 0.78:
            retry_recommendations.append("Retry the both corner with basis-specific hardening.")
        if rigor_of_neither.score < 0.78:
            retry_recommendations.append("Retry the neither corner with a stricter frame-failure diagnosis.")
        if transformation_quality.score < 0.82:
            retry_recommendations.append("Retry stage 6 with non-averaging pressure and arbiter-derived invariants.")
        if actionability.score < 0.75:
            retry_recommendations.append("Retry domain adapters for missing interfaces, thresholds, experiments, or plans.")
        if adapter_feedback.score < 0.60:
            retry_recommendations.append("Domain adapters do not address P*'s operational tests — retry stage 7 with operational test references.")

        return VerificationReportArtifact(
            branch_independence=self._to_metric(branch_independence, threshold=0.90),
            divergence_quality=self._to_metric(divergence_quality, threshold=0.45),
            rigor_of_both=self._to_metric(rigor_of_both, threshold=0.78),
            rigor_of_neither=self._to_metric(rigor_of_neither, threshold=0.78),
            contradiction_honesty=self._to_metric(contradiction_honesty, threshold=0.75),
            transformation_quality=self._to_metric(transformation_quality, threshold=0.82),
            actionability=self._to_metric(actionability, threshold=0.75),
            robustness=self._to_metric(robustness, threshold=0.70),
            fake_novelty_risk=self._to_metric(fake_novelty_risk, threshold=0.70),
            slop_risk=self._to_metric(slop_risk, threshold=0.70),
            domain_adapter_feedback=self._to_metric(adapter_feedback, threshold=0.60),
            aggregate_score=aggregate,
            retry_recommendations=retry_recommendations,
        )

    @staticmethod
    def _to_metric(value: ScoreWithReason, threshold: float) -> VerificationMetricArtifact:
        return VerificationMetricArtifact(score=round(value.score, 3), rationale=value.rationale, passed=value.score >= threshold)

    def branch_independence(self, run: TetraFrameRunArtifact) -> ScoreWithReason:
        contamination = corner_contamination_score(run)
        rationale = (
            "Combines input-view isolation, absence of explicit cross-references, and residual overlap penalties. "
            f"Score={contamination:.3f}."
        )
        return ScoreWithReason(contamination, rationale)

    def divergence_quality(self, run: TetraFrameRunArtifact) -> ScoreWithReason:
        score = corner_divergence_score(run)
        rationale = (
            "Measures pairwise distinction among incompatible corners and rewards distinct unique signals and falsifiers. "
            f"Score={score:.3f}."
        )
        return ScoreWithReason(score, rationale)

    def rigor_of_both(self, corner: HardenedCornerArtifact) -> ScoreWithReason:
        heuristic = both_rigor_heuristic(corner)
        if heuristic >= 0.85:
            return ScoreWithReason(heuristic, "Heuristic pass: valid split basis, explicit co-holding conditions, no compromise markers.")
        pred = self.corner_judge(corner_mode="both", corner_json=corner.model_dump_json())
        return ScoreWithReason(float(pred.score), pred.rationale)

    def rigor_of_neither(self, corner: HardenedCornerArtifact) -> ScoreWithReason:
        heuristic = neither_rigor_heuristic(corner)
        if heuristic >= 0.85:
            return ScoreWithReason(heuristic, "Heuristic pass: explicit failure mode, frame diagnosis, and replacement predicate/frame present.")
        pred = self.corner_judge(corner_mode="neither", corner_json=corner.model_dump_json())
        return ScoreWithReason(float(pred.score), pred.rationale)

    def contradiction_honesty(self, run: TetraFrameRunArtifact) -> ScoreWithReason:
        score = contradiction_honesty_score(run.cartography)
        rationale = (
            "Penalizes softened maps that erase direct contradictions and rewards explicit contradiction/evidence links. "
            f"Score={score:.3f}."
        )
        return ScoreWithReason(score, rationale)

    def transformation_quality(
        self,
        frame: TransformedFrameArtifact,
        corners: dict[CornerMode, HardenedCornerArtifact],
    ) -> ScoreWithReason:
        heuristic = non_averaging_transformation_score(frame, corners)
        if heuristic >= 0.88:
            return ScoreWithReason(heuristic, "Heuristic pass: transformed predicate is not compromise text and includes survival/dissolution mappings.")
        pred = self.transform_judge(
            transformed_frame_json=frame.model_dump_json(),
            hardened_corners_json=json.dumps({k.value: v.model_dump() for k, v in corners.items()}, ensure_ascii=False),
        )
        return ScoreWithReason(float(pred.score), pred.rationale)

    def actionability(self, run: TetraFrameRunArtifact) -> ScoreWithReason:
        scores: list[float] = []
        reasons: list[str] = []
        for domain, artifact in {
            "writing": run.writing,
            "coding": run.coding,
            "research": run.research,
            "planning": run.planning,
        }.items():
            heuristic = domain_usefulness_heuristic(domain, artifact.model_dump())
            if heuristic < 0.75:
                pred = self.domain_judge(
                    domain=domain,
                    artifact_json=artifact.model_dump_json(),
                    transformed_frame_json=run.transformed_frame.model_dump_json(),
                )
                scores.append(float(pred.score))
                reasons.append(f"{domain}: {pred.rationale}")
            else:
                scores.append(heuristic)
                reasons.append(f"{domain}: heuristic pass")
        return ScoreWithReason(sum(scores) / len(scores), " | ".join(reasons))

    def robustness(self, run: TetraFrameRunArtifact) -> ScoreWithReason:
        score = robustness_heuristic(run)
        return ScoreWithReason(score, f"Uses confidence boundaries, reversible maps, and failure-mode coverage. Score={score:.3f}.")

    def fake_novelty_risk(self, run: TetraFrameRunArtifact) -> ScoreWithReason:
        score = fake_novelty_resistance(run)
        return ScoreWithReason(score, f"Penalizes unsupported new concepts in P*. Score={score:.3f}.")

    def slop_risk(self, run: TetraFrameRunArtifact) -> ScoreWithReason:
        score = slop_resistance(run)
        return ScoreWithReason(score, f"Penalizes mushy language and empty discriminators. Score={score:.3f}.")

    def domain_adapter_feedback(self, run: TetraFrameRunArtifact) -> ScoreWithReason:
        score = domain_adapter_feedback_score(run)
        return ScoreWithReason(score, f"Checks domain adapters reference P*'s operational tests and failure modes. Score={score:.3f}.")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return 0.0 if not values else sum(values) / len(values)


def evidence_specificity_score(items: list[str]) -> float:
    if not items:
        return 0.0
    hits = []
    for item in items:
        lowered = item.lower()
        marker_hit = any(marker in lowered for marker in OBSERVABLE_MARKERS)
        numeric_hit = bool(re.search(r"\b\d+\b", lowered))
        colon_hit = ":" in lowered or "->" in lowered
        hits.append(1.0 if (marker_hit or numeric_hit or colon_hit) else 0.25)
    return round(_mean(hits), 3)


def falsifier_quality_score(corner: HardenedCornerArtifact) -> float:
    if not corner.minimal_falsifiers:
        return 0.0
    directness = _mean(1.0 if len(f.split()) >= 5 else 0.5 for f in corner.minimal_falsifiers)
    specificity = evidence_specificity_score(corner.minimal_falsifiers)
    return round((directness + specificity) / 2.0, 3)


def internal_coherence_score(corner: HardenedCornerArtifact) -> float:
    components = [
        1.0 if corner.patched_claim.strip() else 0.0,
        1.0 if corner.clarified_scope_conditions else 0.0,
        1.0 if corner.confidence_boundaries else 0.0,
        falsifier_quality_score(corner),
    ]
    return round(_mean(components), 3)


def corner_divergence_score(run: TetraFrameRunArtifact) -> float:
    corners = run.hardened_corners
    seed_text = run.distilled_seed.normalized_project_seed
    incompatible = []
    for left, right in incompatible_pairs():
        a = corners[left].patched_claim
        b = corners[right].patched_claim
        sim = pairwise_similarity(
            " ".join(sorted(residual_tokens(a, seed_text))),
            " ".join(sorted(residual_tokens(b, seed_text))),
        )
        incompatible.append(1.0 - sim)
    unique_signal_diversity = []
    for left, right in incompatible_pairs():
        sim = pairwise_similarity(corners[left].unique_signal, corners[right].unique_signal)
        unique_signal_diversity.append(1.0 - sim)
    return round(0.7 * _mean(incompatible) + 0.3 * _mean(unique_signal_diversity), 3)


def explicit_cross_reference_count(text: str) -> float:
    lowered = (text or "").lower()
    markers = [
        "other corner",
        "another corner",
        "p says",
        "not-p says",
        "neither says",
        "both says",
        "as above",
        "the previous branch",
    ]
    return float(sum(marker in lowered for marker in markers))


def corner_contamination_score(run: TetraFrameRunArtifact) -> float:
    traces = [t for t in run.traces if t.stage_name.startswith("stage2")]
    if not traces:
        return 0.5
    view_penalty = 1.0 if any(t.blocked_input_fields and set(t.visible_input_fields) & set(t.blocked_input_fields) for t in traces) else 0.0
    cross_ref_penalty = _mean(
        explicit_cross_reference_count(c.core_claim + " " + c.strongest_case) / 2.0 for c in run.corner_drafts.values()
    )
    seed_text = run.distilled_seed.normalized_project_seed
    pair_penalties = []
    for left, right in incompatible_pairs():
        a = run.corner_drafts[left].core_claim
        b = run.corner_drafts[right].core_claim
        sim = pairwise_similarity(
            " ".join(sorted(residual_tokens(a, seed_text))),
            " ".join(sorted(residual_tokens(b, seed_text))),
        )
        pair_penalties.append(max(0.0, sim - 0.35))
    score = 1.0 - min(1.0, 0.5 * view_penalty + 0.2 * cross_ref_penalty + 0.6 * _mean(pair_penalties))
    return round(max(0.0, score), 3)


def both_rigor_heuristic(corner: HardenedCornerArtifact) -> float:
    allowed_basis = {
        "temporal_split",
        "scale_split",
        "role_split",
        "ontology_split",
        "context_split",
        "layered_causality",
        "admissible_paradox",
    }
    basis_ok = 1.0 if corner.validity_basis_label in allowed_basis else 0.0
    compromise_penalty = 0.0 if not any(x in corner.strongest_case.lower() for x in COMPROMISE_MARKERS) else 0.4
    simultaneous = 1.0 if any(word in corner.validity_basis_explanation.lower() for word in ["both", "co-hold", "simult", "split"]) else 0.5
    scope = 1.0 if corner.clarified_scope_conditions else 0.5
    score = max(0.0, _mean([basis_ok, simultaneous, scope, falsifier_quality_score(corner)]) - compromise_penalty)
    return round(score, 3)


def neither_rigor_heuristic(corner: HardenedCornerArtifact) -> float:
    allowed_failure_modes = {
        "category_error",
        "false_binary",
        "overloaded_predicate",
        "missing_latent_variable",
        "bad_ontology",
        "ill_posed_objective",
        "frame_collapse_under_scrutiny",
    }
    mode_ok = 1.0 if corner.validity_basis_label in allowed_failure_modes else 0.0
    replacement_ok = 1.0 if (corner.replacement_predicate.strip() or corner.replacement_frame.strip()) else 0.0
    diagnosis_ok = 1.0 if len(corner.validity_basis_explanation.split()) >= 8 else 0.5
    evasion_penalty = 0.3 if "it depends" in corner.strongest_case.lower() else 0.0
    score = max(0.0, _mean([mode_ok, replacement_ok, diagnosis_ok, falsifier_quality_score(corner)]) - evasion_penalty)
    return round(score, 3)


def contradiction_honesty_score(cartography: CartographyArtifact) -> float:
    contradiction_count = len(cartography.contradiction_map)
    complementarity_count = len(cartography.complementarity_map)
    evidence_count = len(cartography.evidence_discriminator_map)
    if contradiction_count == 0:
        return 0.35 if complementarity_count > 0 else 0.25
    return round(min(1.0, 0.4 + 0.1 * contradiction_count + 0.05 * evidence_count), 3)


def non_averaging_transformation_score(
    frame: TransformedFrameArtifact,
    corners: dict[CornerMode, HardenedCornerArtifact],
) -> float:
    compromise_hit = any(marker in frame.transformed_frame.lower() or marker in frame.transformed_predicate.lower() for marker in COMPROMISE_MARKERS)
    required_fields = [
        bool(frame.survivors_from_p),
        bool(frame.survivors_from_not_p),
        bool(frame.hidden_structure_from_both),
        bool(frame.dissolved_false_frame_from_neither),
        bool(frame.operational_tests),
    ]
    novelty_overlap = pairwise_similarity(
        frame.transformed_predicate,
        " ".join([
            corners[CornerMode.P].patched_claim,
            corners[CornerMode.NOT_P].patched_claim,
        ]),
    )
    score = _mean([1.0 if x else 0.0 for x in required_fields])
    score += 0.3 * (1.0 - min(1.0, novelty_overlap))
    if compromise_hit:
        score -= 0.4
    return round(max(0.0, min(1.0, score)), 3)


def domain_usefulness_heuristic(domain: str, artifact: dict[str, Any]) -> float:
    required_fields = {
        "writing": ["central_claim", "rival_readings", "tension_map", "outline", "voice_options", "stress_test", "revision_plan"],
        "coding": ["architecture", "modules", "interfaces", "state_model", "verification_loop", "tests", "failure_modes", "iteration_plan"],
        "research": ["competing_hypotheses", "discriminating_experiments", "evidence_agenda", "confound_map", "interpretation_grid", "next_step_program"],
        "planning": ["option_set", "leverage_points", "decision_thresholds", "scenario_map", "reversibility_map", "execution_phases", "monitoring_plan"],
    }[domain]
    present = []
    specificity = []
    for field in required_fields:
        value = artifact.get(field)
        present.append(1.0 if value else 0.0)
        if isinstance(value, list):
            texts = [str(item) for item in value if str(item).strip()]
        elif value:
            texts = [str(value)]
        else:
            texts = []
        if not texts:
            specificity.append(0.0)
            continue
        specificity.append(
            _mean(
                1.0
                if (
                    len(text.split()) >= 4
                    or any(marker in text.lower() for marker in OBSERVABLE_MARKERS)
                    or any(marker in text for marker in [":", "->", "_"])
                )
                else 0.35
                for text in texts
            )
        )
    return round(0.55 * _mean(present) + 0.45 * _mean(specificity), 3)


def robustness_heuristic(run: TetraFrameRunArtifact) -> float:
    frame = run.transformed_frame
    coverage = _mean(
        [
            1.0 if frame.boundary_conditions else 0.0,
            1.0 if frame.failure_modes else 0.0,
            1.0 if run.cartography.reversible_implications else 0.0,
            1.0 if run.cartography.irreversible_implications else 0.0,
        ]
    )
    confidence = _mean(c.confidence_score for c in run.hardened_corners.values())
    return round(0.6 * coverage + 0.4 * min(1.0, confidence), 3)


def domain_adapter_feedback_score(run: TetraFrameRunArtifact) -> float:
    """Check whether domain adapters are responsive to P*'s operational tests and failure modes.

    This closes the feedback loop the seed identified as missing: domain adapters
    produce typed artifacts but had no way to verify they actually address what P*
    specified.  The metric checks whether adapter outputs reference terms from
    the operational tests, boundary conditions, and failure modes defined by P*.
    """
    frame = run.transformed_frame
    # Build the set of distinctive terms from P*'s operational tests and failure modes
    reference_parts = (
        frame.operational_tests
        + frame.boundary_conditions
        + frame.failure_modes
    )
    if not reference_parts:
        return 0.5  # P* didn't specify operational tests — inconclusive

    reference_tokens = set()
    for part in reference_parts:
        reference_tokens.update(t.lower() for t in re.findall(r"[a-zA-Z_]+", part) if len(t) > 4)

    if not reference_tokens:
        return 0.5

    # Check each domain adapter for references to P*'s operational vocabulary
    domain_scores = []
    for adapter in [run.writing, run.coding, run.research, run.planning]:
        adapter_text = json.dumps(adapter.model_dump(), ensure_ascii=False).lower()
        adapter_tokens = set(re.findall(r"[a-zA-Z_]+", adapter_text))
        overlap = reference_tokens & adapter_tokens
        coverage = len(overlap) / len(reference_tokens) if reference_tokens else 0.0
        domain_scores.append(min(1.0, coverage * 1.5))  # Scale so 67% coverage = 1.0

    return round(_mean(domain_scores), 3)


def fake_novelty_resistance(run: TetraFrameRunArtifact) -> float:
    frame_tokens = set(re.findall(r"[a-zA-Z_]+", run.transformed_frame.transformed_predicate.lower()))
    # Include all reasoning artifacts as valid source material — P* synthesizes
    # from corners, arbiter notes, cartography, and the transformed frame's own
    # survival/dissolution mappings, not just the primary predicate.
    source_parts = [
        run.predicate_selection.primary_predicate.text,
        run.distilled_seed.normalized_project_seed,
        *(c.patched_claim for c in run.hardened_corners.values()),
        *(c.strongest_case for c in run.hardened_corners.values()),
        *(c.tightened_language for c in run.hardened_corners.values()),
        *(c.replacement_predicate for c in run.hardened_corners.values()),
        *(c.replacement_frame for c in run.hardened_corners.values()),
        *run.cartography.invariant_map,
        *run.cartography.category_error_map,
        *run.cartography.frame_validity_map,
        run.arbiter.arbiter_notes,
        *run.arbiter.transformation,
        *run.arbiter.dissolution,
        *run.transformed_frame.survivors_from_p,
        *run.transformed_frame.survivors_from_not_p,
        *run.transformed_frame.hidden_structure_from_both,
        *run.transformed_frame.dissolved_false_frame_from_neither,
    ]
    source_text = " ".join(source_parts).lower()
    source_tokens = set(re.findall(r"[a-zA-Z_]+", source_text))
    def _is_supported(token: str) -> bool:
        if token in source_tokens:
            return True
        # Check if the token is a compound of source tokens (e.g. "preregistered" = "pre" + "registered")
        for src in source_tokens:
            if len(src) > 3 and src in token:
                return True
        return False
    unsupported = [t for t in frame_tokens if len(t) > 4 and not _is_supported(t)]
    penalty = min(0.6, 0.08 * len(unsupported))
    return round(max(0.0, 1.0 - penalty), 3)


def slop_resistance(run: TetraFrameRunArtifact) -> float:
    texts = [
        run.transformed_frame.transformed_frame,
        run.transformed_frame.non_averaging_explanation,
        *(c.tightened_language for c in run.hardened_corners.values()),
    ]
    token_count = 0
    mush_count = 0
    for text in texts:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        token_count += len(tokens)
        mush_count += sum(token in MUSH_WORDS for token in tokens)
    mush_ratio = 0.0 if token_count == 0 else mush_count / token_count
    evidence = _mean(evidence_specificity_score(c.evidence_needs) for c in run.hardened_corners.values())
    score = max(0.0, 1.0 - 3.0 * mush_ratio)
    score = 0.5 * score + 0.5 * evidence
    return round(score, 3)


# ------------------------------
# compile / optimizer metrics
# ------------------------------

def stage0_metric(gold: Any, pred: Any) -> float:
    required = [
        bool(getattr(pred, "normalized_project_seed", "")),
        bool(getattr(pred, "candidate_predicates", [])),
        bool(getattr(pred, "evaluation_criteria", [])),
    ]
    return round(_mean(1.0 if x else 0.0 for x in required), 3)


def predicate_metric(gold: Any, pred: Any) -> float:
    text = getattr(pred, "primary_predicate", "")
    operational = 1.0 if len(text.split()) >= 6 and any(op in text.lower() for op in ["reduces", "improves", "maximizes", "minimizes", "separates", "increases"]) else 0.4
    notes = 1.0 if getattr(pred, "operationalization_notes", []) else 0.0
    return round((operational + notes) / 2.0, 3)


def corner_metric(gold: Any, pred: Any) -> float:
    required = [
        getattr(pred, "core_claim", ""),
        getattr(pred, "strongest_case", ""),
        getattr(pred, "scope_conditions", []),
        getattr(pred, "falsifiers", []),
        getattr(pred, "evidence_needs", []),
        getattr(pred, "unique_signal", ""),
        getattr(pred, "validity_basis_label", ""),
    ]
    base = _mean(1.0 if x else 0.0 for x in required)
    evidence = evidence_specificity_score(getattr(pred, "evidence_needs", []))
    return round(0.7 * base + 0.3 * evidence, 3)


def hardening_reward(args: dict[str, Any], pred: Any) -> float:
    base = _mean(
        [
            1.0 if getattr(pred, "patched_claim", "") else 0.0,
            1.0 if getattr(pred, "minimal_falsifiers", []) else 0.0,
            1.0 if getattr(pred, "confidence_boundaries", []) else 0.0,
            float(getattr(pred, "confidence_score", 0.0) or 0.0),
        ]
    )
    return round(min(1.0, base), 3)


def transform_reward(args: dict[str, Any], pred: Any) -> float:
    compromise_penalty = 0.4 if any(marker in getattr(pred, "transformed_predicate", "").lower() for marker in COMPROMISE_MARKERS) else 0.0
    components = _mean(
        [
            1.0 if getattr(pred, "survivors_from_p", []) else 0.0,
            1.0 if getattr(pred, "survivors_from_not_p", []) else 0.0,
            1.0 if getattr(pred, "hidden_structure_from_both", []) else 0.0,
            1.0 if getattr(pred, "dissolved_false_frame_from_neither", []) else 0.0,
        ]
    )
    return round(max(0.0, components - compromise_penalty), 3)


def _term_hit_score(text: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 0.0
    lowered = text.lower()
    return round(_mean(1.0 if term.lower() in lowered else 0.0 for term in expected_terms), 3)


def _domain_marker_score(run: TetraFrameRunArtifact, expected_domain_markers: dict[str, list[str]]) -> float:
    if not expected_domain_markers:
        return 0.0
    domain_payloads = {
        "writing": json.dumps(run.writing.model_dump(), ensure_ascii=False).lower(),
        "coding": json.dumps(run.coding.model_dump(), ensure_ascii=False).lower(),
        "research": json.dumps(run.research.model_dump(), ensure_ascii=False).lower(),
        "planning": json.dumps(run.planning.model_dump(), ensure_ascii=False).lower(),
    }
    per_domain_scores = []
    for domain, markers in expected_domain_markers.items():
        payload = domain_payloads.get(domain, "")
        if not markers:
            continue
        per_domain_scores.append(_term_hit_score(payload, markers))
    return round(_mean(per_domain_scores), 3)


def benchmark_score_breakdown(run: TetraFrameRunArtifact, gold: Any) -> tuple[dict[str, float], list[str]]:
    transformed_payload = " ".join(
        [
            run.transformed_frame.transformed_predicate,
            run.transformed_frame.transformed_frame,
        ]
    )
    breakdown: dict[str, float] = {}
    failures: list[str] = []

    expected_primary_terms = list(getattr(gold, "expected_primary_predicate_contains", []))
    if expected_primary_terms:
        score = _term_hit_score(run.predicate_selection.primary_predicate.text, expected_primary_terms)
        breakdown["primary_predicate"] = score
        if score < 1.0:
            failures.append("primary predicate missed expected terms")

    expected_transformed_terms = list(getattr(gold, "expected_transformed_predicate_contains", []))
    if expected_transformed_terms:
        score = _term_hit_score(transformed_payload, expected_transformed_terms)
        breakdown["transformed_predicate"] = score
        if score < 1.0:
            failures.append("transformed frame missed expected terms")

    allowed_both_basis = list(getattr(gold, "allowed_both_basis", []))
    if allowed_both_basis:
        score = 1.0 if run.hardened_corners[CornerMode.BOTH].validity_basis_label in allowed_both_basis else 0.0
        breakdown["both_basis"] = score
        if score < 1.0:
            failures.append("both corner used a disallowed basis")

    expected_neither_modes = list(getattr(gold, "expected_neither_failure_modes", []))
    if expected_neither_modes:
        score = 1.0 if run.hardened_corners[CornerMode.NEITHER].validity_basis_label in expected_neither_modes else 0.0
        breakdown["neither_failure_mode"] = score
        if score < 1.0:
            failures.append("neither corner used an unexpected failure mode")

    domain_markers = dict(getattr(gold, "expected_domain_markers", {}))
    if domain_markers:
        score = _domain_marker_score(run, domain_markers)
        breakdown["domain_markers"] = score
        if score < 1.0:
            failures.append("domain adapters missed expected markers")

    banned_phrases = list(getattr(gold, "banned_transformed_phrases", []))
    if banned_phrases:
        lowered = transformed_payload.lower()
        score = 1.0 if not any(term.lower() in lowered for term in banned_phrases) else 0.0
        breakdown["banned_phrase_avoidance"] = score
        if score < 1.0:
            failures.append("transformed frame used banned compromise language")

    verification_score = run.verification.aggregate_score if run.verification else 0.0
    breakdown["verification"] = round(verification_score, 3)
    return breakdown, failures


def benchmark_success(run: TetraFrameRunArtifact, gold: Any) -> float:
    breakdown, _ = benchmark_score_breakdown(run, gold)
    return round(_mean(breakdown.values()), 3)


def gepa_feedback_metric(gold: Any, pred: Any, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible metric that returns scalar score plus terse textual feedback."""
    if isinstance(pred, TetraFrameRunArtifact):
        score = pred.verification.aggregate_score
        deficits = []
        if pred.verification.branch_independence.score < 0.90:
            deficits.append("branch independence is weak")
        if pred.verification.rigor_of_both.score < 0.78:
            deficits.append("both is under-specified or compromise-like")
        if pred.verification.rigor_of_neither.score < 0.78:
            deficits.append("neither fails to diagnose the frame or replace it")
        if pred.verification.transformation_quality.score < 0.82:
            deficits.append("P* looks averaged rather than transformed")
        if pred.verification.actionability.score < 0.75:
            deficits.append("domain adapters are generic")
        feedback = " ; ".join(deficits) if deficits else "Trajectory is structurally strong. Preserve rigor and specificity."
        return {"score": score, "feedback": feedback}
    return {"score": 0.0, "feedback": "Prediction was not a TetraFrameRunArtifact."}
