from __future__ import annotations

import asyncio
from typing import Any

import tetraframe.dspy_compat as dspy

from tetraframe.artifacts import (
    CornerDraftArtifact,
    CornerInputView,
    CornerMode,
    HardenedCornerArtifact,
    TetraFrameRunArtifact,
)
from tetraframe.config import RootConfig
from tetraframe.guards import (
    BLOCKED_CORNER_FIELDS,
    assert_corner_view_isolation,
    build_anti_collapse_hint,
    detect_near_duplicate_corners,
    make_corner_input_view,
    reorder_for_unbiased_fallback,
)
from tetraframe.modules import (
    CartographCornersModule,
    CornerBothGenerator,
    CornerNeitherGenerator,
    CornerNotPGenerator,
    CornerPGenerator,
    DomainAdaptModule,
    FourCornerArbiterModule,
    HardenCornerModule,
    PredicateSelectModule,
    SeedDistillModule,
    TransformFrameModule,
    VerifyRunModule,
)
from tetraframe.tracing import TraceLogger


class TetraFrameProgram(dspy.Module):
    """Compile-friendly sequential program with explicit branch isolation.

    Production uses AsyncTetraFrameRunner for actual parallel branch execution.
    This class is intentionally deterministic and optimizer-friendly.
    """

    def __init__(self, cfg: RootConfig):
        super().__init__()
        self.cfg = cfg
        self.seed_distill = SeedDistillModule()
        self.predicate_select = PredicateSelectModule()
        self.corner_generators = {
            CornerMode.P: CornerPGenerator(),
            CornerMode.NOT_P: CornerNotPGenerator(),
            CornerMode.BOTH: CornerBothGenerator(),
            CornerMode.NEITHER: CornerNeitherGenerator(),
        }
        self.harden_corner = HardenCornerModule()
        self.cartograph = CartographCornersModule()
        self.arbitrate = FourCornerArbiterModule()
        self.transform = TransformFrameModule()
        self.domain_adapt = DomainAdaptModule()
        self.verify = VerifyRunModule()
        self.trace_logger = TraceLogger(cfg.program.trace_dir)

    def forward(self, raw_seed: str) -> TetraFrameRunArtifact:
        distilled, traces = self._run_stage0(raw_seed)
        selection, trace = self._run_stage1(distilled)
        traces.append(trace)
        corner_inputs, corner_drafts, stage2_traces = self._run_stage2(distilled, selection)
        traces.extend(stage2_traces)
        hardened_corners, stage3_traces = self._run_stage3(distilled, selection, corner_drafts)
        traces.extend(stage3_traces)
        cartography, trace = self._run_stage4(distilled.run_id, hardened_corners)
        traces.append(trace)
        arbiter, trace = self._run_stage5(distilled.run_id, hardened_corners, cartography)
        traces.append(trace)
        transformed, trace = self._run_stage6(distilled, selection, hardened_corners, cartography, arbiter)
        traces.append(trace)
        (writing, coding, research, planning), stage7_traces = self._run_stage7(distilled.run_id, transformed, cartography)
        traces.extend(stage7_traces)
        provisional = TetraFrameRunArtifact(
            run_id=distilled.run_id,
            distilled_seed=distilled,
            predicate_selection=selection,
            corner_inputs=corner_inputs,
            corner_drafts=corner_drafts,
            hardened_corners=hardened_corners,
            cartography=cartography,
            arbiter=arbiter,
            transformed_frame=transformed,
            writing=writing,
            coding=coding,
            research=research,
            planning=planning,
            verification=None,  # type: ignore[arg-type]
            traces=traces,
        )
        verification, trace = self._run_stage8(provisional)
        traces.append(trace)
        provisional.verification = verification
        provisional.traces = traces
        return provisional

    def run(self, raw_seed: str) -> TetraFrameRunArtifact:
        return self(raw_seed)

    def _run_stage0(self, raw_seed: str):
        trace_ctx = self.trace_logger.stage(
            run_id="pending",
            stage_name="stage0.seed_distill",
            module_name="SeedDistillModule",
            signature_name="SeedDistillSignature",
            attempt=1,
            visible_inputs={"raw_seed": raw_seed},
            blocked_input_fields=[],
        )
        distilled = self.seed_distill(raw_seed)
        trace = trace_ctx.close(distilled.model_dump(), resolved_run_id=distilled.run_id)
        return distilled, [trace]

    def _ordered_corner_modes(self, run_id: str) -> list[CornerMode]:
        modes = list(self.corner_generators)
        if self.cfg.program.randomize_sequential_fallback_order:
            return reorder_for_unbiased_fallback(modes, run_id)
        return modes

    def _build_corner_views(self, distilled, selection) -> dict[CornerMode, CornerInputView]:
        views = {mode: make_corner_input_view(distilled, selection, mode) for mode in self.corner_generators}
        for view in views.values():
            assert_corner_view_isolation(view)
        return views

    @staticmethod
    def _apply_anti_collapse_hints(
        views: dict[CornerMode, CornerInputView],
        duplicates: list[tuple[CornerMode, CornerMode, float]],
    ) -> dict[CornerMode, CornerInputView]:
        updated = dict(views)
        affected_modes = {mode for pair in duplicates for mode in pair[:2]}
        for mode in affected_modes:
            updated[mode] = updated[mode].model_copy(update={"anti_collapse_hint": build_anti_collapse_hint(mode, duplicates)})
        return updated

    def _run_stage1(self, distilled):
        trace_ctx = self.trace_logger.stage(
            run_id=distilled.run_id,
            stage_name="stage1.predicate_select",
            module_name="PredicateSelectModule",
            signature_name="SplitPredicateSignature+ChoosePredicateSignature",
            attempt=1,
            visible_inputs={
                "normalized_project_seed": distilled.normalized_project_seed,
                "candidate_predicates": distilled.candidate_predicates,
                "constraints": distilled.constraints,
                "evaluation_criteria": distilled.evaluation_criteria,
            },
            blocked_input_fields=[],
        )
        selection = self.predicate_select(distilled)
        trace = trace_ctx.close(selection.model_dump())
        return selection, trace

    def _run_stage2(self, distilled, selection):
        modes = self._ordered_corner_modes(distilled.run_id)
        views = self._build_corner_views(distilled, selection)
        traces = []
        drafts: dict[CornerMode, CornerDraftArtifact] = {}
        for attempt in range(1, self.cfg.program.max_corner_generation_attempts + 1):
            contexts = []
            for mode in modes:
                view = views[mode]
                trace_ctx = self.trace_logger.stage(
                    run_id=distilled.run_id,
                    stage_name=f"stage2.generate.{mode.value}",
                    module_name=self.corner_generators[mode].__class__.__name__,
                    signature_name=self.corner_generators[mode].signature_cls.__name__,
                    attempt=attempt,
                    visible_inputs=view.model_dump(),
                    blocked_input_fields=BLOCKED_CORNER_FIELDS,
                    config={
                        "rollout_id": f"{distilled.run_id}:stage2:{mode.value}:{attempt}",
                        "temperature": self.cfg.program.corner_temperatures[mode.value],
                    },
                )
                contexts.append((mode, trace_ctx))
                draft = self.corner_generators[mode](
                    view,
                    rollout_id=f"{distilled.run_id}:stage2:{mode.value}:{attempt}",
                    temperature=self.cfg.program.corner_temperatures[mode.value],
                )
                drafts[mode] = draft
            duplicates = detect_near_duplicate_corners(drafts, distilled.normalized_project_seed)
            retry_reason = "near-duplicate corners detected" if duplicates else ""
            for mode, trace_ctx in contexts:
                traces.append(trace_ctx.close(drafts[mode].model_dump(), retry_reason=retry_reason))
            if not duplicates:
                break
            views = self._apply_anti_collapse_hints(views, duplicates)
        return views, drafts, traces

    def _run_stage3(self, distilled, selection, drafts):
        traces = []
        hardened: dict[CornerMode, HardenedCornerArtifact] = {}
        for mode, draft in drafts.items():
            trace_ctx = self.trace_logger.stage(
                run_id=distilled.run_id,
                stage_name=f"stage3.harden.{mode.value}",
                module_name="HardenCornerModule",
                signature_name="HardenCornerSignature",
                attempt=1,
                visible_inputs={
                    "mode": mode.value,
                    "primary_predicate": selection.primary_predicate.text,
                    "corner": draft.model_dump(),
                },
                blocked_input_fields=BLOCKED_CORNER_FIELDS,
            )
            hardened_corner = self.harden_corner(distilled, selection, draft)
            hardened[mode] = hardened_corner
            traces.append(trace_ctx.close(hardened_corner.model_dump()))
        return hardened, traces

    def _run_stage4(self, run_id, hardened):
        trace_ctx = self.trace_logger.stage(
            run_id=run_id,
            stage_name="stage4.cartograph",
            module_name="CartographCornersModule",
            signature_name="PairwiseRelationSignature+GlobalCartographySignature",
            attempt=1,
            visible_inputs={"corners": {k.value: v.model_dump() for k, v in hardened.items()}},
            blocked_input_fields=[],
        )
        cartography = self.cartograph(hardened)
        trace = trace_ctx.close(cartography.model_dump())
        return cartography, trace

    def _run_stage5(self, run_id, hardened, cartography):
        trace_ctx = self.trace_logger.stage(
            run_id=run_id,
            stage_name="stage5.arbiter",
            module_name="FourCornerArbiterModule",
            signature_name="FourCornerArbiterSignature",
            attempt=1,
            visible_inputs={
                "hardened_corners": {k.value: v.model_dump() for k, v in hardened.items()},
                "cartography": cartography.model_dump(),
            },
            blocked_input_fields=[],
        )
        arbiter = self.arbitrate(hardened, cartography)
        trace = trace_ctx.close(arbiter.model_dump())
        return arbiter, trace

    def _run_stage6(self, distilled, selection, hardened, cartography, arbiter):
        trace_ctx = self.trace_logger.stage(
            run_id=distilled.run_id,
            stage_name="stage6.transform",
            module_name="TransformFrameModule",
            signature_name="TransformFrameSignature",
            attempt=1,
            visible_inputs={
                "primary_predicate": selection.primary_predicate.text,
                "cartography": cartography.model_dump(),
                "arbiter": arbiter.model_dump(),
            },
            blocked_input_fields=[],
        )
        transformed = self.transform(distilled, selection, hardened, cartography, arbiter)
        trace = trace_ctx.close(transformed.model_dump())
        return transformed, trace

    def _run_stage7(self, run_id, transformed, cartography):
        outputs = self.domain_adapt(transformed, cartography)
        traces = []
        for name, artifact in zip(["writing", "coding", "research", "planning"], outputs, strict=True):
            trace_ctx = self.trace_logger.stage(
                run_id=run_id,
                stage_name=f"stage7.domain.{name}",
                module_name="DomainAdaptModule",
                signature_name=f"Adapt{name.title()}Signature",
                attempt=1,
                visible_inputs={
                    "transformed_predicate": transformed.transformed_predicate,
                    "transformed_frame": transformed.transformed_frame,
                },
                blocked_input_fields=[],
            )
            traces.append(trace_ctx.close(artifact.model_dump()))
        return outputs, traces

    def _run_stage8(self, run: TetraFrameRunArtifact):
        trace_ctx = self.trace_logger.stage(
            run_id=run.run_id,
            stage_name="stage8.verify",
            module_name="VerifyRunModule",
            signature_name=None,
            attempt=1,
            visible_inputs={
                "transformed_predicate": run.transformed_frame.transformed_predicate,
                "corner_modes": [k.value for k in run.hardened_corners],
            },
            blocked_input_fields=[],
        )
        report = self.verify(run)
        trace = trace_ctx.close(report.model_dump(), scores={"aggregate_score": report.aggregate_score})
        return report, trace


class AsyncTetraFrameRunner:
    """Production runner that preserves the same stage contracts but executes stages 2 and 3 in parallel.

    DSPy currently supports async wrapping via `dspy.asyncify`. This class keeps that wrapper at the orchestration layer,
    outside the module internals, which is the safest pattern for production parallelism.
    """

    def __init__(self, program: TetraFrameProgram):
        self.program = program

    async def _generate_corners_async(
        self,
        distilled,
        selection,
    ) -> tuple[dict[CornerMode, CornerInputView], dict[CornerMode, CornerDraftArtifact], list]:
        views = self.program._build_corner_views(distilled, selection)
        modes = self.program._ordered_corner_modes(distilled.run_id)
        traces = []
        drafts: dict[CornerMode, CornerDraftArtifact] = {}
        for attempt in range(1, self.program.cfg.program.max_corner_generation_attempts + 1):
            ordered_items = [(mode, self.program.corner_generators[mode]) for mode in modes]
            contexts = []
            tasks = []
            for mode, module in ordered_items:
                trace_ctx = self.program.trace_logger.stage(
                    run_id=distilled.run_id,
                    stage_name=f"stage2.generate.{mode.value}",
                    module_name=module.__class__.__name__,
                    signature_name=module.signature_cls.__name__,
                    attempt=attempt,
                    visible_inputs=views[mode].model_dump(),
                    blocked_input_fields=BLOCKED_CORNER_FIELDS,
                    config={
                        "rollout_id": f"{distilled.run_id}:stage2:{mode.value}:{attempt}",
                        "temperature": self.program.cfg.program.corner_temperatures[mode.value],
                    },
                )
                contexts.append((mode, trace_ctx))
                async_module = dspy.asyncify(module.deepcopy())
                tasks.append(
                    async_module(
                        views[mode],
                        rollout_id=f"{distilled.run_id}:stage2:{mode.value}:{attempt}",
                        temperature=self.program.cfg.program.corner_temperatures[mode.value],
                    )
                )
            results = await asyncio.gather(*tasks)
            drafts = {mode: result for (mode, _), result in zip(contexts, results, strict=True)}
            duplicates = detect_near_duplicate_corners(drafts, distilled.normalized_project_seed)
            retry_reason = "near-duplicate corners detected" if duplicates else ""
            for (mode, trace_ctx), result in zip(contexts, results, strict=True):
                traces.append(trace_ctx.close(result.model_dump(), retry_reason=retry_reason))
            if not duplicates:
                break
            views = self.program._apply_anti_collapse_hints(views, duplicates)
        return views, drafts, traces

    async def _harden_corners_async(self, distilled, selection, drafts):
        modes = list(drafts.keys())
        contexts = []
        tasks = []
        for mode in modes:
            trace_ctx = self.program.trace_logger.stage(
                run_id=distilled.run_id,
                stage_name=f"stage3.harden.{mode.value}",
                module_name="HardenCornerModule",
                signature_name="HardenCornerSignature",
                attempt=1,
                visible_inputs={
                    "mode": mode.value,
                    "primary_predicate": selection.primary_predicate.text,
                    "corner": drafts[mode].model_dump(),
                },
                blocked_input_fields=BLOCKED_CORNER_FIELDS,
            )
            contexts.append((mode, trace_ctx))
            async_module = dspy.asyncify(self.program.harden_corner.deepcopy())
            tasks.append(async_module(distilled, selection, drafts[mode]))
        results = await asyncio.gather(*tasks)
        hardened = {mode: result for (mode, _), result in zip(contexts, results, strict=True)}
        traces = [ctx.close(result.model_dump()) for (_, ctx), result in zip(contexts, results, strict=True)]
        return hardened, traces

    async def run_async(self, raw_seed: str) -> TetraFrameRunArtifact:
        distilled, traces = self.program._run_stage0(raw_seed)
        selection, trace = self.program._run_stage1(distilled)
        traces.append(trace)
        corner_inputs, drafts, stage2_traces = await self._generate_corners_async(distilled, selection)
        traces.extend(stage2_traces)
        hardened, stage3_traces = await self._harden_corners_async(distilled, selection, drafts)
        traces.extend(stage3_traces)
        cartography, trace = self.program._run_stage4(distilled.run_id, hardened)
        traces.append(trace)
        arbiter, trace = self.program._run_stage5(distilled.run_id, hardened, cartography)
        traces.append(trace)
        transformed, trace = self.program._run_stage6(distilled, selection, hardened, cartography, arbiter)
        traces.append(trace)
        (writing, coding, research, planning), stage7_traces = self.program._run_stage7(distilled.run_id, transformed, cartography)
        traces.extend(stage7_traces)
        provisional = TetraFrameRunArtifact(
            run_id=distilled.run_id,
            distilled_seed=distilled,
            predicate_selection=selection,
            corner_inputs=corner_inputs,
            corner_drafts=drafts,
            hardened_corners=hardened,
            cartography=cartography,
            arbiter=arbiter,
            transformed_frame=transformed,
            writing=writing,
            coding=coding,
            research=research,
            planning=planning,
            verification=None,
            traces=traces,
        )
        verification, trace = self.program._run_stage8(provisional)
        traces.append(trace)
        provisional.verification = verification
        provisional.traces = traces
        return provisional

    def run(self, raw_seed: str) -> TetraFrameRunArtifact:
        return asyncio.run(self.run_async(raw_seed))

    def __call__(self, raw_seed: str) -> TetraFrameRunArtifact:
        return self.run(raw_seed)


def build_runtime_runner(program: TetraFrameProgram | Any) -> TetraFrameProgram | AsyncTetraFrameRunner | Any:
    cfg = getattr(program, "cfg", None)
    if cfg is None:
        return program
    return AsyncTetraFrameRunner(program) if cfg.program.parallel_corners else program
