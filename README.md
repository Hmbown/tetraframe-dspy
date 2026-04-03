# TetraFrame

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A [DSPy](https://dspy.ai/) program for decisions that break ordinary pro/con reasoning. Generates four isolated positions on a predicate — P, not-P, both, neither — maps where they conflict or collapse, and synthesizes a result that preserves tension instead of flattening it.

```bash
pip install -e ".[dev]"
tetraframe run "Your question" --config configs/base.yaml
```

## Why four corners

Thesis/antithesis discards structure too early. TetraFrame uses a tetralemmatic frame (from Buddhist logic, Nāgārjuna ~2nd century CE) as a practical reasoning structure:

- **P** and **not-P** are generated independently. No corner sees any other.
- **Both** must show how P and not-P jointly hold under a clarified frame — not split the difference.
- **Neither** must reject the predicate and propose a better one.
- Corners stay isolated until relationship mapping.

## Pipeline

```
Seed ─▶ Predicate ─▶ Four Corners ─▶ Mapping ─▶ Transform ─▶ Verify
```

| # | Stage | |
|---|-------|-|
| 1 | Problem Distillation | Stakes, constraints, unknowns, hidden assumptions |
| 2 | Predicate Selection | Choose the predicate corners will reason over |
| 3 | Four Corners | P, not-P, both, neither — isolated, self-critiqued |
| 4 | Relationship Mapping | Contradiction, complementarity, collapse risk, reframing |
| 5 | Transform | Synthesis that preserves tension instead of averaging |
| 6 | Verify | Score for collapse, unsupported reframing, slop |

## Example

**Seed:** Is AI code review more effective on diffs or full files?

- **P:** Diffs are sufficient for localized changes at lower token cost.
- **not-P:** Diffs miss cross-module invariants, regression risks, and type contract violations.
- **Both:** Scale-dependent. Below ~30 LOC, diffs suffice. Above 50 LOC or across modules, full-file wins.
- **Neither:** Misframed. Prompt design and defect-type specificity matter more than input mode.

**Synthesis:** The question is not "diffs or full files" but "for which defect categories does input mode become the binding constraint?"

CLI runs are offline and may take tens of minutes.

## When not to use it

Overkill for factual QA, latency-sensitive chat, narrow bugs with clear ground truth, or tasks where decomposition adds cost but not signal.

## How DSPy fits

Each stage is a `dspy.Module` with typed `dspy.Signature` interfaces. DSPy handles prompt construction, parsing, and retries. The program compiles with `BootstrapFewShot`, `MIPROv2`, or `GEPA`, runs across any LM backend, and nests inside larger DSPy programs.

Corners use `ChainOfThought`. Transform uses `BestOfN` with a reward function that penalizes collapse. `TetraFrameProgram` is a `dspy.Module` with a `forward()` method.

## Install

```bash
git clone https://github.com/Hmbown/tetraframe-dspy.git && cd tetraframe-dspy
python -m venv venv && venv/bin/pip install -e ".[dev]"
venv/bin/python -m pytest -q
```

## Run

```bash
tetraframe run "Your seed question" --config configs/base.yaml --out runs/latest/run.json
```

Auto-discovery works without a config:

```bash
tetraframe discover                          # show available backends
tetraframe run "seed"                        # auto-select best available
tetraframe run "seed" --tool openai-api      # force a specific backend
```

## Python API

```python
from tetraframe.config import load_config
from tetraframe.backends.factory import build_dspy_lm
from tetraframe.pipeline import TetraFrameProgram, build_runtime_runner
import tetraframe.dspy_compat as dspy

cfg = load_config("configs/base.yaml")
lm = build_dspy_lm(cfg)
dspy.configure(lm=lm)

program = TetraFrameProgram(cfg)
runner = build_runtime_runner(program)
result = runner.run("Your reasoning seed")
result.to_json("runs/latest/run.json")
```

<details>
<summary><b>Backends</b></summary>

| Provider | Config | Env var |
|---|---|---|
| OpenAI | `configs/openai.yaml` | `OPENAI_API_KEY` |
| Anthropic | `configs/anthropic.yaml` | `ANTHROPIC_API_KEY` |
| OpenRouter | `configs/openrouter.yaml` | `OPENROUTER_API_KEY` |
| OpenAI-compatible | `configs/openai_compatible.yaml` | any |
| Claude Code | `configs/claude_code_cli.yaml` | — |
| Codex | `configs/codex_cli.yaml` | — |
| OpenCode | `configs/opencode_cli.yaml` | — |

Any OpenAI-compatible `/v1` endpoint works. CLI backends run as subprocesses (slower, no max_tokens/temperature enforcement).

</details>

<details>
<summary><b>Benchmark and compile</b></summary>

```bash
tetraframe benchmark --config configs/base.yaml --dataset examples/benchmark_cases_test.jsonl --out runs/benchmarks/report.json
tetraframe compile --config configs/compile.yaml --out runs/compiled_program.json
```

</details>

<details>
<summary><b>Config reference</b></summary>

```yaml
model:
  runtime_model: openai/gpt-5.4-mini
  reflection_model: openai/gpt-5.4
  backend:
    provider: openai
    model: gpt-5.4-mini
    base_url: null
    api_key_env: null
    timeout: 120.0
  reflection_backend: null
```

</details>

<details>
<summary><b>Project structure</b></summary>

```
src/tetraframe/
  pipeline.py          modules.py           signatures.py
  artifacts.py         guards.py            metrics.py
  tracing.py           config.py            cli.py
  compile.py
  backends/            tools/               proxy/
  benchmarks/
```

</details>

## License

MIT
