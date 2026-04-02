from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import tetraframe.pipeline as pipeline_module
from tetraframe.artifacts import CornerMode
from tetraframe.config import RootConfig
from tetraframe.pipeline import AsyncTetraFrameRunner, TetraFrameProgram
from tests.fixtures import build_sample_run_artifact


def _fake_asyncify(module):
    async def _runner(*args, **kwargs):
        return module(*args, **kwargs)

    return _runner


def _build_program(tmp_path: Path, monkeypatch, *, parallel_corners: bool):
    sample = build_sample_run_artifact().model_copy(deep=True)
    sample.run_id = "run_integration"
    sample.distilled_seed.run_id = "run_integration"

    cfg = RootConfig.model_validate(
        {
            "program": {
                "parallel_corners": parallel_corners,
                "randomize_sequential_fallback_order": True,
                "max_corner_generation_attempts": 2,
                "trace_dir": str(tmp_path / "traces"),
            }
        }
    )
    program = TetraFrameProgram(cfg)
    monkeypatch.setattr(pipeline_module.dspy, "asyncify", _fake_asyncify)

    call_counts: defaultdict[CornerMode, int] = defaultdict(int)

    program.seed_distill.forward = lambda raw_seed, **config: sample.distilled_seed.model_copy(  # type: ignore[method-assign]
        update={"run_id": "run_integration", "raw_seed": raw_seed}
    )
    program.predicate_select.forward = lambda distilled, **config: sample.predicate_selection.model_copy(deep=True)  # type: ignore[method-assign]

    def make_corner_forward(mode: CornerMode):
        def _forward(view, **config):
            call_counts[mode] += 1
            base = sample.corner_drafts[mode].model_copy(deep=True)
            if mode in {CornerMode.P, CornerMode.NOT_P} and call_counts[mode] == 1:
                return base.model_copy(
                    update={
                        "core_claim": "collapsed sentinel branch structure",
                        "strongest_case": "collapsed sentinel branch structure with no real divergence",
                        "unique_signal": "collapsed sentinel",
                    }
                )
            if view.anti_collapse_hint:
                return base.model_copy(update={"unique_signal": f"{base.unique_signal} [{view.anti_collapse_hint}]"})
            return base

        return _forward

    for mode, module in program.corner_generators.items():
        module.forward = make_corner_forward(mode)  # type: ignore[method-assign]
        module.deepcopy = lambda module=module: module  # type: ignore[method-assign]

    def harden_forward(distilled, selection, draft, **config):
        return sample.hardened_corners[draft.corner_mode].model_copy(deep=True, update=draft.model_dump())

    program.harden_corner.forward = harden_forward  # type: ignore[method-assign]
    program.harden_corner.deepcopy = lambda: program.harden_corner  # type: ignore[method-assign]
    program.cartograph.forward = lambda corners, **config: sample.cartography.model_copy(deep=True)  # type: ignore[method-assign]
    program.arbitrate.forward = lambda corners, cartography, **config: sample.arbiter.model_copy(deep=True)  # type: ignore[method-assign]
    program.transform.forward = (  # type: ignore[method-assign]
        lambda distilled, selection, corners, cartography, arbiter, **config: sample.transformed_frame.model_copy(deep=True)
    )
    program.domain_adapt.forward = lambda frame, cartography, **config: (  # type: ignore[method-assign]
        sample.writing.model_copy(deep=True),
        sample.coding.model_copy(deep=True),
        sample.research.model_copy(deep=True),
        sample.planning.model_copy(deep=True),
    )
    program.verify.forward = lambda run: sample.verification.model_copy(deep=True)  # type: ignore[method-assign]
    return program


def test_sequential_pipeline_retries_collapsed_corners_and_writes_single_trace_file(tmp_path, monkeypatch):
    program = _build_program(tmp_path, monkeypatch, parallel_corners=False)

    artifact = program.run("sequential seed")

    assert artifact.corner_inputs[CornerMode.P].anti_collapse_hint
    assert artifact.corner_inputs[CornerMode.NOT_P].anti_collapse_hint
    assert artifact.corner_inputs[CornerMode.BOTH].anti_collapse_hint == ""
    assert artifact.corner_inputs[CornerMode.NEITHER].anti_collapse_hint == ""

    stage2_attempts = [trace for trace in artifact.traces if trace.stage_name.startswith("stage2.generate")]
    assert any(trace.attempt == 1 and trace.retry_reason == "near-duplicate corners detected" for trace in stage2_attempts)
    assert any(trace.stage_name == "stage2.generate.P" and trace.attempt == 2 for trace in stage2_attempts)

    trace_path = tmp_path / "traces" / "run_integration.jsonl"
    assert trace_path.exists()
    assert not (tmp_path / "traces" / "pending.jsonl").exists()

    lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(artifact.traces)
    assert lines[0]["stage_name"] == "stage0.seed_distill"
    assert all(line["run_id"] == "run_integration" for line in lines)


def test_async_pipeline_uses_randomized_order_and_selective_retry_hints(tmp_path, monkeypatch):
    program = _build_program(tmp_path, monkeypatch, parallel_corners=True)
    forced_order = [CornerMode.NEITHER, CornerMode.BOTH, CornerMode.NOT_P, CornerMode.P]
    monkeypatch.setattr(pipeline_module, "reorder_for_unbiased_fallback", lambda modes, run_id: forced_order)

    artifact = AsyncTetraFrameRunner(program).run("async seed")

    stage2_attempt1 = [
        trace.stage_name
        for trace in artifact.traces
        if trace.stage_name.startswith("stage2.generate") and trace.attempt == 1
    ]
    assert stage2_attempt1 == [
        "stage2.generate.neither",
        "stage2.generate.both",
        "stage2.generate.not-P",
        "stage2.generate.P",
    ]
    assert artifact.corner_inputs[CornerMode.P].anti_collapse_hint
    assert artifact.corner_inputs[CornerMode.NOT_P].anti_collapse_hint
    assert artifact.corner_inputs[CornerMode.BOTH].anti_collapse_hint == ""
    assert artifact.corner_inputs[CornerMode.NEITHER].anti_collapse_hint == ""
