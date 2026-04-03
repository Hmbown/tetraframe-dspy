from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Iterable

import tetraframe.dspy_compat as dspy

from tetraframe.artifacts import (
    CartographyArtifact,
    CornerArtifact,
    CornerInputView,
    CornerMode,
    CornerReconstructionArtifact,
    DistilledSeedArtifact,
    EvidenceDiscriminatorArtifact,
    PairwiseRelationArtifact,
    PredicateSelectionArtifact,
    PredicateSpec,
    RejectedPredicate,
    RelationType,
    TetraFrameRunArtifact,
    TransformedFrameArtifact,
)
from tetraframe.guards import parse_json_field
from tetraframe.metrics import VerificationSuite, transform_reward
from tetraframe.signatures import (
    ChoosePredicateSignature,
    GenerateCornerBothSignature,
    GenerateCornerNeitherSignature,
    GenerateCornerNotPSignature,
    GenerateCornerPSignature,
    GlobalCartographySignature,
    PairwiseRelationSignature,
    SeedDistillSignature,
    SplitPredicateSignature,
    TransformFrameSignature,
)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _safe_relation_type(value: str) -> RelationType:
    """Parse RelationType robustly from free-form LLM output."""
    value = str(value).strip().lower().replace(" ", "_")
    valid = {m.value: m for m in RelationType}
    if value in valid:
        return valid[value]
    for key, member in valid.items():
        if key in value:
            return member
    return RelationType.SUPPORT  # Safest fallback


class SeedDistillModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(SeedDistillSignature)

    def forward(self, raw_seed: str, **config: Any) -> DistilledSeedArtifact:
        pred = self.predict(raw_seed=raw_seed, **({"config": config} if config else {}))
        return DistilledSeedArtifact(
            raw_seed=raw_seed,
            normalized_project_seed=pred.normalized_project_seed,
            stakes=list(pred.stakes),
            constraints=list(pred.constraints),
            unknowns=list(pred.unknowns),
            hidden_assumptions=list(pred.hidden_assumptions),
            candidate_predicates=list(pred.candidate_predicates),
            frame_risk_score=float(pred.frame_risk_score),
            evaluation_criteria=list(pred.evaluation_criteria),
            novelty_criteria=list(pred.novelty_criteria),
        )


class PredicateSelectModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.splitter = dspy.ChainOfThought(SplitPredicateSignature)
        self.chooser = dspy.ChainOfThought(ChoosePredicateSignature)

    def forward(self, distilled: DistilledSeedArtifact, **config: Any) -> PredicateSelectionArtifact:
        kwargs = {"config": config} if config else {}
        split = self.splitter(
            normalized_project_seed=distilled.normalized_project_seed,
            candidate_predicates=distilled.candidate_predicates,
            hidden_assumptions=distilled.hidden_assumptions,
            constraints=distilled.constraints,
            **kwargs,
        )
        rejected = [RejectedPredicate.model_validate(x) for x in parse_json_field(split.rejected_predicates_json, [])]
        choice = self.chooser(
            normalized_project_seed=distilled.normalized_project_seed,
            atomic_predicates=list(split.atomic_predicates),
            constraints=distilled.constraints,
            evaluation_criteria=distilled.evaluation_criteria,
            novelty_criteria=distilled.novelty_criteria,
            **kwargs,
        )
        primary = PredicateSpec.model_validate(parse_json_field(choice.primary_predicate_parts_json, {"text": choice.primary_predicate}))
        sub_specs = [PredicateSpec.model_validate(x) for x in parse_json_field(choice.sub_predicates_json, [])]
        if not primary.text:
            primary.text = choice.primary_predicate
        return PredicateSelectionArtifact(
            primary_predicate=primary,
            sub_predicates=sub_specs,
            rejected_predicates=rejected,
            rationale=choice.rationale,
            operationalization_notes=list(choice.operationalization_notes),
        )


class CornerGeneratorBase(dspy.Module):
    """Generate and harden a corner position in a single pass."""
    signature_cls = None
    mode: CornerMode | None = None

    def __init__(self):
        super().__init__()
        if self.signature_cls is None or self.mode is None:
            raise ValueError("signature_cls and mode must be defined")
        self.predict = dspy.ChainOfThought(self.signature_cls)

    def forward(self, view: CornerInputView, **config: Any) -> CornerArtifact:
        payload = view.model_dump(exclude={"run_id", "corner_mode", "corner_contract"})
        pred = self.predict(**payload, **({"config": config} if config else {}))
        data = dict(
            corner_mode=self.mode,
            # Generation fields
            core_claim=pred.core_claim,
            assumptions=list(pred.assumptions),
            strongest_case=pred.strongest_case,
            scope_conditions=list(pred.scope_conditions),
            falsifiers=list(pred.falsifiers),
            evidence_needs=list(pred.evidence_needs),
            uncertainty=pred.uncertainty,
            unique_signal=pred.unique_signal,
            validity_basis_label=pred.validity_basis_label,
            validity_basis_explanation=pred.validity_basis_explanation,
            replacement_predicate=getattr(pred, "replacement_predicate", "") or "",
            replacement_frame=getattr(pred, "replacement_frame", "") or "",
            # Hardening fields
            internal_attack=list(getattr(pred, "internal_attack", []) or []),
            patched_claim=getattr(pred, "patched_claim", "") or "",
            patched_assumptions=list(getattr(pred, "patched_assumptions", []) or []),
            clarified_scope_conditions=list(getattr(pred, "clarified_scope_conditions", []) or []),
            confidence_boundaries=list(getattr(pred, "confidence_boundaries", []) or []),
            minimal_falsifiers=list(getattr(pred, "minimal_falsifiers", []) or []),
            tightened_language=getattr(pred, "tightened_language", "") or "",
            unresolved_weaknesses=list(getattr(pred, "unresolved_weaknesses", []) or []),
            confidence_score=float(getattr(pred, "confidence_score", 0.5) or 0.5),
            still_valid_after_hardening=bool(getattr(pred, "still_valid_after_hardening", True)),
            invalidity_reason=getattr(pred, "invalidity_reason", "") or "",
        )
        return CornerArtifact(**data)


class CornerPGenerator(CornerGeneratorBase):
    signature_cls = GenerateCornerPSignature
    mode = CornerMode.P


class CornerNotPGenerator(CornerGeneratorBase):
    signature_cls = GenerateCornerNotPSignature
    mode = CornerMode.NOT_P


class CornerBothGenerator(CornerGeneratorBase):
    signature_cls = GenerateCornerBothSignature
    mode = CornerMode.BOTH


class CornerNeitherGenerator(CornerGeneratorBase):
    signature_cls = GenerateCornerNeitherSignature
    mode = CornerMode.NEITHER


class PairwiseCornerRelator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(PairwiseRelationSignature)

    def forward(
        self,
        source: CornerArtifact,
        target: CornerArtifact,
        **config: Any,
    ) -> PairwiseRelationArtifact:
        pred = self.predict(
            source_corner=source.model_dump_json(),
            target_corner=target.model_dump_json(),
            **({"config": config} if config else {}),
        )
        return PairwiseRelationArtifact(
            source_corner=source.corner_mode,
            target_corner=target.corner_mode,
            relation_type=_safe_relation_type(pred.relation_type),
            rationale=pred.rationale,
            evidence_discriminator=pred.evidence_discriminator,
            reversible=bool(pred.reversible),
            invariant_tags=list(pred.invariant_tags),
        )


class CartographCornersModule(dspy.Module):
    """Map cross-corner relationships and produce arbiter judgments in a single pass."""

    def __init__(self):
        super().__init__()
        self.relator = PairwiseCornerRelator()
        self.global_map = dspy.ChainOfThought(GlobalCartographySignature)

    def forward(self, corners: dict[CornerMode, CornerArtifact], **config: Any) -> CartographyArtifact:
        relations = []
        ordered = [CornerMode.P, CornerMode.NOT_P, CornerMode.BOTH, CornerMode.NEITHER]
        for left, right in combinations(ordered, 2):
            relations.append(self.relator(corners[left], corners[right], **config))
        pred = self.global_map(
            corners_json=_json_dump({k.value: v.model_dump() for k, v in corners.items()}),
            pairwise_relations_json=_json_dump([r.model_dump() for r in relations]),
            **({"config": config} if config else {}),
        )
        evidence = [
            EvidenceDiscriminatorArtifact.model_validate(x)
            for x in parse_json_field(pred.evidence_discriminator_map_json, [])
        ]
        structural_miss = parse_json_field(pred.structural_miss_map_json, {})
        reconstructions = [
            CornerReconstructionArtifact.model_validate(x)
            for x in parse_json_field(pred.reconstructions_json, [])
        ]
        return CartographyArtifact(
            pairwise_relations=relations,
            contradiction_map=list(pred.contradiction_map),
            complementarity_map=list(pred.complementarity_map),
            paradox_map=list(pred.paradox_map),
            category_error_map=list(pred.category_error_map),
            frame_validity_map=list(pred.frame_validity_map),
            evidence_discriminator_map=evidence,
            invariant_map=list(pred.invariant_map),
            reversible_implications=list(pred.reversible_implications),
            irreversible_implications=list(pred.irreversible_implications),
            structural_miss_map=structural_miss,
            # Arbiter fields
            reconstructions=reconstructions,
            dissolution=list(pred.dissolution),
            transformation=list(pred.transformation),
            arbiter_notes=pred.arbiter_notes,
        )


class TransformFrameModule(dspy.Module):
    def __init__(self):
        super().__init__()
        base = dspy.ChainOfThought(TransformFrameSignature)
        self.best = dspy.BestOfN(module=base, N=3, reward_fn=transform_reward, threshold=0.84)

    def forward(
        self,
        distilled: DistilledSeedArtifact,
        selection: PredicateSelectionArtifact,
        corners: dict[CornerMode, CornerArtifact],
        cartography: CartographyArtifact,
        **config: Any,
    ) -> TransformedFrameArtifact:
        pred = self.best(
            primary_predicate=selection.primary_predicate.text,
            corners_json=_json_dump({k.value: v.model_dump() for k, v in corners.items()}),
            cartography_json=cartography.model_dump_json(),
            evaluation_criteria=distilled.evaluation_criteria,
            novelty_criteria=distilled.novelty_criteria,
            **({"config": config} if config else {}),
        )
        return TransformedFrameArtifact(
            transformed_predicate=pred.transformed_predicate,
            transformed_frame=pred.transformed_frame,
            survivors_from_p=list(pred.survivors_from_p),
            survivors_from_not_p=list(pred.survivors_from_not_p),
            hidden_structure_from_both=list(pred.hidden_structure_from_both),
            dissolved_false_frame_from_neither=list(pred.dissolved_false_frame_from_neither),
            non_averaging_explanation=pred.non_averaging_explanation,
            operational_tests=list(pred.operational_tests),
            boundary_conditions=list(pred.boundary_conditions),
            failure_modes=list(pred.failure_modes),
            confidence=float(pred.confidence),
        )


class VerifyRunModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.suite = VerificationSuite()

    def forward(self, run: TetraFrameRunArtifact):
        return self.suite.verify(run)
