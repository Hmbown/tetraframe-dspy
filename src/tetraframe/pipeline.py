from __future__ import annotations

import asyncio
from typing import Any

import tetraframe.dspy_compat as dspy

from tetraframe.artifacts import (
    CornerArtifact,
    CornerInputView,
    CornerMode,
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
    PredicateSelectModule,
    SeedDistillModule,
    TransformFrameModule,
    VerifyRunModule,
)
from tetraframe.tracing import TraceLogger


class TetraFrameProgram(dspy.Module):
    """Six-stage tetralemmatic reasoning pipeline.

    Stages:
        0. Seed Distill
        1. Predicate Selection
        2. Four Corners (generate + harden in one pass)
        3. Cartography + Arbitration (merged)
        4. Transform
        5. Verify
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
        self.cartograph = CartographCornersModule()
        self.transform = TransformFrameModule()
        self.verify = VerifyRunModule()
        self.trace_logger = TraceLogger(cfg.program.trace_dir)

    def forward(self, raw_seed: str) -> TetraFrameRunArtifact:
        distilled, traces = self._run_stage0(raw_seed)
        selection, trace = self._run_stage1(distilled)
        traces.append(trace)
        corner_inputs, corners, stage2_traces = self._run_stage2(distilled, selection)
        traces.extend(stage2_traces)
        cartography, trace = self._run_stage3(distilled.run_id, corners)
        traces.append(trace)
        transformed, trace = self._run_stage4(distilled, selection, corners, cartography)
        traces.append(trace)
        provisional = TetraFrameRunArtifact(
            run_id=distilled.run_id,
            distilled_seed=distilled,
            predicate_selection=selection,
            corner_inputs=corner_inputs,
            corners=corners,
            cartography=cartography,
            transformed_frame=transformed,
            verification=None,  # type: ignore[arg-type]
            traces=traces,
        )
        verification, trace = self._run_stage5(provisional)
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
        corners: dict[CornerMode, CornerArtifact] = {}
        for attempt in range(1, self.cfg.program.max_corner_generation_attempts + 1):
            contexts = []
            for mode in modes:
                view = views[mode]
                trace_ctx = self.trace_logger.stage(
                    run_id=distilled.run_id,
                    stage_name=f"stage2.corner.{mode.value}",
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
                corner = self.corner_generators[mode](
                    view,
                    rollout_id=f"{distilled.run_id}:stage2:{mode.value}:{attempt}",
                    temperature=self.cfg.program.corner_temperatures[mode.value],
                )
                corners[mode] = corner
            duplicates = detect_near_duplicate_corners(corners, distilled.normalized_project_seed)
            retry_reason = "near-duplicate corners detected" if duplicates else ""
            for mode, trace_ctx in contexts:
                traces.append(trace_ctx.close(corners[mode].model_dump(), retry_reason=retry_reason))
            if not duplicates:
                break
            views = self._apply_anti_collapse_hints(views, duplicates)
        return views, corners, traces

    def _run_stage3(self, run_id, corners):
        trace_ctx = self.trace_logger.stage(
            run_id=run_id,
            stage_name="stage3.cartograph",
            module_name="CartographCornersModule",
            signature_name="PairwiseRelationSignature+GlobalCartographySignature",
            attempt=1,
            visible_inputs={"corners": {k.value: v.model_dump() for k, v in corners.items()}},
            blocked_input_fields=[],
        )
        cartography = self.cartograph(corners)
        trace = trace_ctx.close(cartography.model_dump())
        return cartography, trace

    def _run_stage4(self, distilled, selection, corners, cartography):
        trace_ctx = self.trace_logger.stage(
            run_id=distilled.run_id,
            stage_name="stage4.transform",
            module_name="TransformFrameModule",
            signature_name="TransformFrameSignature",
            attempt=1,
            visible_inputs={
                "primary_predicate": selection.primary_predicate.text,
                "cartography": cartography.model_dump(),
            },
            blocked_input_fields=[],
        )
        transformed = self.transform(distilled, selection, corners, cartography)
        trace = trace_ctx.close(transformed.model_dump())
        return transformed, trace

    def _run_stage5(self, run: TetraFrameRunArtifact):
        trace_ctx = self.trace_logger.stage(
            run_id=run.run_id,
            stage_name="stage5.verify",
            module_name="VerifyRunModule",
            signature_name=None,
            attempt=1,
            visible_inputs={
                "transformed_predicate": run.transformed_frame.transformed_predicate,
                "corner_modes": [k.value for k in run.corners],
            },
            blocked_input_fields=[],
        )
        report = self.verify(run)
        trace = trace_ctx.close(report.model_dump(), scores={"aggregate_score": report.aggregate_score})
        return report, trace


class AsyncTetraFrameRunner:
    """Production runner that executes corners in parallel."""

    def __init__(self, program: TetraFrameProgram):
        self.program = program

    async def _generate_corners_async(
        self,
        distilled,
        selection,
    ) -> tuple[dict[CornerMode, CornerInputView], dict[CornerMode, CornerArtifact], list]:
        views = self.program._build_corner_views(distilled, selection)
        modes = self.program._ordered_corner_modes(distilled.run_id)
        traces = []
        corners: dict[CornerMode, CornerArtifact] = {}
        for attempt in range(1, self.program.cfg.program.max_corner_generation_attempts + 1):
            ordered_items = [(mode, self.program.corner_generators[mode]) for mode in modes]
            contexts = []
            tasks = []
            for mode, module in ordered_items:
                trace_ctx = self.program.trace_logger.stage(
                    run_id=distilled.run_id,
                    stage_name=f"stage2.corner.{mode.value}",
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
            corners = {mode: result for (mode, _), result in zip(contexts, results, strict=True)}
            duplicates = detect_near_duplicate_corners(corners, distilled.normalized_project_seed)
            retry_reason = "near-duplicate corners detected" if duplicates else ""
            for (mode, trace_ctx), result in zip(contexts, results, strict=True):
                traces.append(trace_ctx.close(result.model_dump(), retry_reason=retry_reason))
            if not duplicates:
                break
            views = self.program._apply_anti_collapse_hints(views, duplicates)
        return views, corners, traces

    async def run_async(self, raw_seed: str) -> TetraFrameRunArtifact:
        distilled, traces = self.program._run_stage0(raw_seed)
        selection, trace = self.program._run_stage1(distilled)
        traces.append(trace)
        corner_inputs, corners, stage2_traces = await self._generate_corners_async(distilled, selection)
        traces.extend(stage2_traces)
        cartography, trace = self.program._run_stage3(distilled.run_id, corners)
        traces.append(trace)
        transformed, trace = self.program._run_stage4(distilled, selection, corners, cartography)
        traces.append(trace)
        provisional = TetraFrameRunArtifact(
            run_id=distilled.run_id,
            distilled_seed=distilled,
            predicate_selection=selection,
            corner_inputs=corner_inputs,
            corners=corners,
            cartography=cartography,
            transformed_frame=transformed,
            verification=None,
            traces=traces,
        )
        verification, trace = self.program._run_stage5(provisional)
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
