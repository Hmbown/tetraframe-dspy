from __future__ import annotations

import hashlib
import json
import random
import re
from itertools import combinations
from typing import Any, Iterable

from tetraframe.artifacts import CartographyArtifact, CornerDraftArtifact, CornerInputView, CornerMode

CORNER_CONTRACTS = {
    CornerMode.P: "Strongest clean affirmation of the predicate. No compromise or mention of other corners.",
    CornerMode.NOT_P: "Strongest clean rejection, inversion, or dismantling of the predicate. No mere surface negation.",
    CornerMode.BOTH: "Valid only when P and not-P co-hold under a typed split or admissible paradox.",
    CornerMode.NEITHER: "Valid only when the original predicate is misframed and replaced by a better predicate or frame.",
}

BLOCKED_CORNER_FIELDS = [
    "raw_seed",
    "candidate_predicates",
    "rejected_predicates",
    "rationale",
    "other_corner_outputs",
    "corner_drafts",
    "hardened_corners",
    "cartography",
    "arbiter",
    "transformed_frame",
    "verification",
    "traces",
]

ALLOWED_CORNER_FIELDS = set(CornerInputView.model_fields.keys())
_TOKEN_RE = re.compile(r"[a-zA-Z_]+")


def parse_json_field(value: str | bytes | bytearray | None, fallback: Any) -> Any:
    if value in (None, "", b"", bytearray()):
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 2]


def residual_tokens(text: str, seed_text: str) -> list[str]:
    seed = set(_tokens(seed_text))
    return [t for t in _tokens(text) if t not in seed]


def pairwise_similarity(a: str, b: str) -> float:
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / len(ta | tb), 3)


def incompatible_pairs() -> list[tuple[CornerMode, CornerMode]]:
    return [
        (CornerMode.P, CornerMode.NOT_P),
        (CornerMode.P, CornerMode.NEITHER),
        (CornerMode.NOT_P, CornerMode.NEITHER),
    ]


def make_corner_input_view(distilled, selection, mode: CornerMode, anti_collapse_hint: str = "") -> CornerInputView:
    return CornerInputView(
        run_id=distilled.run_id,
        normalized_project_seed=distilled.normalized_project_seed,
        stakes=list(distilled.stakes),
        constraints=list(distilled.constraints),
        unknowns=list(distilled.unknowns),
        hidden_assumptions=list(distilled.hidden_assumptions),
        primary_predicate=selection.primary_predicate.text,
        sub_predicates=[p.text for p in selection.sub_predicates],
        operationalization_notes=list(selection.operationalization_notes),
        evaluation_criteria=list(distilled.evaluation_criteria),
        novelty_criteria=list(distilled.novelty_criteria),
        corner_mode=mode,
        corner_contract=CORNER_CONTRACTS[mode],
        anti_collapse_hint=anti_collapse_hint,
    )


def assert_corner_view_isolation(view: CornerInputView) -> None:
    payload = view.model_dump()
    fields = set(payload.keys())
    extra = fields - ALLOWED_CORNER_FIELDS
    blocked = fields & set(BLOCKED_CORNER_FIELDS)
    if extra:
        raise ValueError(f"Corner view leaked unexpected fields: {sorted(extra)}")
    if blocked:
        raise ValueError(f"Corner view leaked blocked fields: {sorted(blocked)}")


def detect_near_duplicate_corners(
    drafts: dict[CornerMode, CornerDraftArtifact],
    seed_text: str,
    similarity_threshold: float = 0.78,
) -> list[tuple[CornerMode, CornerMode, float]]:
    duplicates: list[tuple[CornerMode, CornerMode, float]] = []
    for left, right in combinations(drafts.keys(), 2):
        left_text = " ".join(residual_tokens(drafts[left].core_claim, seed_text))
        right_text = " ".join(residual_tokens(drafts[right].core_claim, seed_text))
        sim = pairwise_similarity(left_text, right_text)
        if sim >= similarity_threshold:
            duplicates.append((left, right, sim))
    return duplicates


def build_anti_collapse_hint(mode: CornerMode, duplicates: Iterable[tuple[CornerMode, CornerMode, float]] | None = None) -> str:
    if mode == CornerMode.P:
        return "State the strongest clean affirmation of the predicate itself. Do not hedge into synthesis or contrastive framing."
    if mode == CornerMode.NOT_P:
        return "Find a deeper failure, inversion, or dismantling of the predicate. Do not merely restate the seed with weaker confidence."
    if mode == CornerMode.BOTH:
        return "Choose one explicit co-holding basis such as role_split or temporal_split. Show why P and not-P both hold under that basis, not as compromise."
    return "Diagnose a concrete frame failure such as overloaded_predicate or false_binary and replace it with a better predicate or frame."


def reorder_for_unbiased_fallback(modes: list[CornerMode], run_id: str) -> list[CornerMode]:
    ordered = list(modes)
    rng = random.Random(run_id)
    rng.shuffle(ordered)
    return ordered


def cartography_summary(cartography: CartographyArtifact) -> str:
    payload = {
        "contradictions": cartography.contradiction_map[:3],
        "complementarities": cartography.complementarity_map[:3],
        "paradoxes": cartography.paradox_map[:2],
        "frame_validity": cartography.frame_validity_map[:4],
        "invariants": cartography.invariant_map[:4],
        "structural_miss": cartography.structural_miss_map,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def stable_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
