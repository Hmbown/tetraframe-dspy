from __future__ import annotations

import json
from pathlib import Path

import tetraframe.dspy_compat as dspy
import typer

from tetraframe.backends.factory import build_dspy_lm, get_backend_metadata
from tetraframe.benchmarks import BenchmarkHarness, load_benchmark_examples
from tetraframe.compile import Compiler
from tetraframe.config import load_config
from tetraframe.pipeline import TetraFrameProgram, build_runtime_runner

app = typer.Typer(add_completion=False)


def _configure_lm(cfg, *, echo: bool = True) -> None:
    """Build the DSPy LM from config and call dspy.configure().

    Returns backend metadata so callers can inject it into trace loggers.
    """
    lm = build_dspy_lm(cfg)
    dspy.configure(lm=lm)
    meta = get_backend_metadata(cfg)
    if echo:
        typer.echo(f"backend: {meta.name}  model: {meta.model}  kind: {meta.kind}")
        for w in meta.warnings:
            typer.echo(f"  warning: {w}")
    return meta


@app.command()
def run(
    seed: str = typer.Argument(..., help="Raw project seed."),
    config: Path = typer.Option(Path("configs/base.yaml"), exists=True),
    out: Path = typer.Option(Path("runs/latest/run.json")),
) -> None:
    cfg = load_config(config)
    meta = _configure_lm(cfg)
    program = TetraFrameProgram(cfg)
    program.trace_logger.set_backend_info(
        name=meta.name, kind=meta.kind, model=meta.model,
        execution_mode="direct", capability_warnings=meta.warnings,
    )
    runner = build_runtime_runner(program)
    artifact = runner.run(seed)
    artifact.to_json(out)
    typer.echo(f"saved run artifact -> {out}")


@app.command()
def compile(
    config: Path = typer.Option(Path("configs/base.yaml"), exists=True),
    out: Path = typer.Option(Path("runs/compiled_program.json")),
) -> None:
    cfg = load_config(config)
    compiler = Compiler(cfg)
    program = TetraFrameProgram(cfg)
    compiled = compiler.compile(program, cfg.compile.train_path, cfg.compile.dev_path)
    compiled.save(str(out), save_program=True)
    evaluation = compiler.evaluate(compiled, cfg.compile.test_path)
    eval_path = out.with_suffix(".eval.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    mean_score = evaluation["summary"]["mean_score"]  # type: ignore[index]
    threshold = evaluation["summary"]["pass_threshold"]  # type: ignore[index]
    typer.echo(f"saved compiled program -> {out}")
    typer.echo(f"saved compile evaluation -> {eval_path}")
    typer.echo(f"compile evaluation mean_score={mean_score} threshold={threshold}")


@app.command()
def benchmark(
    config: Path = typer.Option(Path("configs/base.yaml"), exists=True),
    dataset: Path = typer.Option(Path("examples/benchmark_cases_test.jsonl"), exists=True),
    out: Path = typer.Option(Path("runs/benchmarks/report.json")),
) -> None:
    cfg = load_config(config)
    meta = _configure_lm(cfg)
    program = TetraFrameProgram(cfg)
    program.trace_logger.set_backend_info(
        name=meta.name, kind=meta.kind, model=meta.model,
        execution_mode="direct", capability_warnings=meta.warnings,
    )
    harness = BenchmarkHarness(program, pass_threshold=cfg.benchmark.pass_threshold)
    examples = load_benchmark_examples(dataset)
    results = harness.run(examples)
    from tetraframe.benchmarks.harness import save_benchmark_report

    save_benchmark_report(out, results, pass_threshold=cfg.benchmark.pass_threshold)
    summary = harness.summarize(results, pass_threshold=cfg.benchmark.pass_threshold)
    typer.echo(f"saved benchmark report -> {out}")
    typer.echo(f"benchmark mean_score={summary['mean_score']} pass_rate={summary['pass_rate']} threshold={summary['pass_threshold']}")


if __name__ == "__main__":
    app()
