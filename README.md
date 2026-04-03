# TetraFrame

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A [DSPy](https://dspy.ai/) program for hard decisions that break ordinary pro/con reasoning.

TetraFrame generates four isolated positions on a predicate — P, not-P, both, and neither — maps where they conflict or collapse, and produces a synthesis that preserves real tension instead of flattening it into a generic middle answer.

Use it for strategic decisions, design tradeoffs, and reframing problems where the original question may itself be wrong.

```bash
pip install -e ".[dev]"
tetraframe run "Your question or problem statement" --config configs/base.yaml
```

## Why four corners instead of pro/con

Standard thesis/antithesis reasoning discards structure too early: state a position, react against it, compromise. The result inherits weaknesses from both sides.

TetraFrame uses a tetralemmatic frame — four corners instead of two. The idea has roots in Buddhist logic (Nāgārjuna, ~2nd century CE), but TetraFrame uses it as a practical reasoning structure, not a historical reconstruction.

TetraFrame enforces several constraints:

- **P** and **not-P** are generated independently — no corner can see any other.
- **Both** must articulate a coherent stance in which P and not-P jointly hold under a clarified frame. It may not split the difference.
- **Neither** must reject the original predicate and propose a better axis of analysis.
- No corner can inspect the others until relationship mapping.

## The pipeline

```
Seed ─▶ Predicate ─▶ Four Corners ─▶ Mapping ─▶ Transform ─▶ Verify
```

| # | Stage | Does |
|---|-------|------|
| 1 | Problem Distillation | Extract stakes, constraints, unknowns, and hidden assumptions from the seed |
| 2 | Predicate Selection | Choose the operative predicate the corners will reason over |
| 3 | Four Corners | Generate P, not-P, both, neither in isolation, with self-critique |
| 4 | Relationship Mapping | Identify contradiction, complementarity, collapse risk, and reframing opportunities across corners |
| 5 | Transform | Produce a synthesis that preserves useful tension instead of averaging it away |
| 6 | Verify | Score for collapse, unsupported reframing, contradiction handling, and tension preservation |

## Example outcome

**Seed:**
> Is code review by AI more effective when it operates on diffs or full files?

**Corners:**
- **P:** Diff-mode achieves comparable defect detection at lower token cost for localized changes — the diff contains nearly all the context needed for single-function bugs.
- **not-P:** Diff-mode systematically misses context-dependent defects — cross-module invariants, regression risks, and type contract violations require full-file visibility.
- **Both:** Both hold at different scales. Below ~30 LOC in a single module, diffs are sufficient. Above 50 LOC or across module boundaries, full-file context provides statistically significant recall gains.
- **Neither:** The original predicate is misframed. Input context mode is a secondary variable — prompt design, model capability, and defect-type specificity are likely stronger determinants of review quality.

**Synthesis:**
> Review quality is governed by context-budget-to-defect-dependence fit: diff-mode provides sufficient context for defects local to changed lines, but cross-boundary defects require full-file or richer context regardless of prompt optimization. The right question is not "diffs or full files" but "for which defect categories does input mode become the binding constraint?"

Current CLI-backed runs are intended for offline reasoning and may take tens of minutes on hard prompts.

## When not to use TetraFrame

TetraFrame is overkill for:
- routine factual QA
- latency-sensitive chat
- narrow coding bugs with clear ground truth
- tasks where decomposition adds cost but not signal

It is best used when the frame itself may be wrong, when multiple incompatible positions each contain real signal, or when ordinary pro/con reasoning collapses too early.

## How DSPy fits

TetraFrame is implemented as a DSPy program. Each stage is a `dspy.Module` with typed `dspy.Signature` interfaces — the pipeline is specified as structured inputs and outputs rather than hand-written prompt chains.

DSPy handles prompt construction, parsing, and retries. Because the program is modular and typed, it can be compiled against benchmark data using `BootstrapFewShot`, `MIPROv2`, or `GEPA` — saved and reloaded, nested inside larger DSPy programs, and run across multiple LM backends without changing the pipeline structure.

Corner generation uses `ChainOfThought`. The transform stage uses `BestOfN` to prefer candidates that preserve irreducible tension and avoid collapsing distinct positions into a generic averaged answer.

If you've used DSPy before: `TetraFrameProgram` is a `dspy.Module` with a `forward()` method.

## Install

```bash
git clone https://github.com/Hmbown/tetraframe-dspy.git && cd tetraframe-dspy
python -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/python -m pytest -q
```

## Run

```bash
venv/bin/tetraframe run \
  "We keep framing product expansion as enter or do not enter, but the real choice may be staged reversible commitments." \
  --config configs/base.yaml \
  --out runs/latest/run.json
```

Outputs:
- `runs/latest/run.json` — full run artifact
- `runs/traces/<run_id>.jsonl` — per-stage trace with timing and backend info

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

## Backends

TetraFrame can find backends automatically without a config file.

```bash
venv/bin/tetraframe discover        # show available tools
venv/bin/tetraframe run "seed"      # auto-select best available
venv/bin/tetraframe run "seed" --tool openai-api  # force a specific tool
```

Priority: Hermes pools > API backends with keys > CLI tools on PATH.

<details>
<summary><b>Backend details</b></summary>

### API backends

| Provider | Config | Env var |
|---|---|---|
| OpenAI | `configs/openai.yaml` | `OPENAI_API_KEY` |
| Anthropic | `configs/anthropic.yaml` | `ANTHROPIC_API_KEY` |
| OpenRouter | `configs/openrouter.yaml` | `OPENROUTER_API_KEY` |
| OpenAI-compatible | `configs/openai_compatible.yaml` | any |

Any provider with an OpenAI-compatible `/v1` endpoint works:

```yaml
model:
  runtime_model: openai/deepseek-chat
  backend:
    provider: openai-compatible
    model: deepseek-chat
    base_url: "https://api.deepseek.com/v1"
    api_key_env: DEEPSEEK_API_KEY
```

### CLI backends

| Tool | Config | Binary | Notes |
|---|---|---|---|
| Claude Code | `configs/claude_code_cli.yaml` | `claude` | No max_tokens, temperature, or streaming |
| Codex | `configs/codex_cli.yaml` | `codex` | Same limitations |
| OpenCode | `configs/opencode_cli.yaml` | `opencode` | Same limitations |

CLI backends run as subprocesses — slower, and the proxy warns when requested parameters can't be enforced.

### API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
```

</details>

## Benchmark and compile

```bash
venv/bin/tetraframe benchmark \
  --config configs/base.yaml \
  --dataset examples/benchmark_cases_test.jsonl \
  --out runs/benchmarks/report.json

venv/bin/tetraframe compile \
  --config configs/compile.yaml \
  --out runs/compiled_program.json
```

<details>
<summary><b>Proxy</b></summary>

`tetraframe-proxy` exposes any configured backend as an OpenAI-compatible HTTP API.

```bash
venv/bin/tetraframe-proxy --config configs/claude_code_cli.yaml
venv/bin/tetraframe-proxy --host 127.0.0.1 --port 8765
```

| Route | Method | Description |
|---|---|---|
| `/health` | GET | Backend status and capabilities |
| `/v1/models` | GET | Available models |
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |

Select backends per-request with `X-Backend`:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Backend: claude-code" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

</details>

<details>
<summary><b>Traces</b></summary>

Each run writes a JSONL trace to `runs/traces/<run_id>.jsonl`. Rows include stage name, latency, backend info, and input/output digests.

```python
import json
from pathlib import Path

run = json.loads(Path("runs/latest/run.json").read_text())
trace_path = Path("runs/traces") / f"{run['run_id']}.jsonl"
for line in trace_path.read_text().splitlines()[:5]:
    row = json.loads(line)
    print(row["stage_name"], row["backend_name"], row["latency_ms"])
```

</details>

<details>
<summary><b>Config reference</b></summary>

```yaml
model:
  runtime_model: openai/gpt-5.4-mini
  reflection_model: openai/gpt-5.4
  backend:
    provider: openai          # openai | anthropic | openrouter | openai-compatible | claude-code | codex | opencode
    model: gpt-5.4-mini
    base_url: null            # for openai-compatible endpoints
    api_key_env: null         # env var holding the API key
    timeout: 120.0
    max_tokens: null
    temperature: null
    binary: null              # CLI: path to binary
    cli_args: []
    env_passthrough: []
    max_retries: 2
    retry_delay: 1.0
  reflection_backend: null    # optional separate backend for compilation
```

</details>

<details>
<summary><b>Project structure</b></summary>

```
src/tetraframe/
  pipeline.py              # 6-stage tetralemmatic pipeline
  modules.py               # DSPy modules (one per stage)
  signatures.py            # DSPy signatures
  artifacts.py             # Typed run artifacts
  guards.py                # Isolation and anti-collapse guards
  metrics.py               # Verification and scoring
  tracing.py               # Per-run trace logger
  config.py                # Config model
  cli.py                   # CLI entry points
  compile.py               # Compilation strategy
  backends/                # Pluggable backend system
    base.py                # Backend protocol + capabilities
    api.py                 # API backends
    cli_base.py            # CLI backend base class
    cli_claude.py          # Claude Code adapter
    cli_codex.py           # Codex adapter
    cli_opencode.py        # OpenCode adapter
    factory.py             # build_backend(), build_dspy_lm()
  tools/                   # Tool plugin system
    protocol.py            # ModelTool protocol
    registry.py            # Auto-discovery registry
    api_tool.py            # Direct API tools
    cli_tool.py            # CLI-based tools
    hermes_tool.py         # Hermes credential pool
  proxy/
    server.py              # Multi-backend OpenAI-compatible proxy
    client.py              # CLI shim
  benchmarks/
    harness.py             # Benchmark runner
```

</details>

## License

MIT
