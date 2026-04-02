# TetraFrame

A DSPy reasoning package that generates four independent positions from a single question and transforms them into useful outputs.

## What is a tetralemma?

A tetralemma is a four-corner logical structure originating in classical Indian and Aristotelian logic. Where binary reasoning gives you two options —

```
A or not-A
```

— a tetralemma gives you four:

```
P · not-P · Both · Neither
```

The key insight: binary framing creates anchoring bias. You state a thesis, then react against it, and end up with a compromise that inherits the weaknesses of both sides. A tetralemma avoids this by making you explore all four corners independently before synthesizing anything.

TetraFrame enforces this structurally:

- **P** and **not-P** are generated in isolation from each other
- **Both** must be two distinct, basis-typed positions — not a mushy middle ground
- **Neither** must identify why the original framing is wrong and propose a replacement predicate
- No corner can see the other corners until the cartography stage

## The pipeline

```
Seed ─▶ Predicate ─▶ Four Corners ─▶ Harden ─▶ Cartograph ─▶ Arbitrate
                                                                    │
                                          ┌─────────────────────────┘
                                          ▼
                  Verify ◀── Domain Adapt ◀── Transform
```

TetraFrame runs 9 stages in sequence:

| # | Stage | Does |
|---|-------|------|
| 1 | Seed Distill | Extracts stakes, constraints, unknowns, and hidden assumptions from your input |
| 2 | Predicate Selection | Picks the operational predicate the four corners will reason about |
| 3 | Four-Corner Generation | Generates P, not-P, Both, Neither — each in complete isolation |
| 4 | Hardening | Strengthens each corner, flags "Both" that are actually compromises |
| 5 | Cartography | Maps overlaps, contradictions, and collapse signals between corners |
| 6 | Arbitration | Scores corners for rigor, detects near-duplicates and contamination |
| 7 | Transform | Synthesizes a non-averaging output from all four corners |
| 8 | Domain Adapt | Adapts the result into a domain-specific format (code, writing, plan, etc.) |
| 9 | Verify | Final quality checks against compromise language and fake novelty |

## Install

```bash
cd tetralemma
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

## Backends

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

CLI backends run as subprocesses. They're slower and the proxy returns warnings when requested parameters can't be enforced.

### Setting API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
```

### Auto-discovery

TetraFrame can find backends automatically without a config file:

```bash
venv/bin/tetraframe discover        # show available tools
venv/bin/tetraframe run "seed"      # auto-select best available
venv/bin/tetraframe run "seed" --tool openai-api  # force a specific tool
```

Priority: Hermes pools > API backends with keys > CLI tools on PATH.

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

## Proxy

`tetraframe-proxy` exposes any configured backend behind an OpenAI-compatible HTTP API:

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

The proxy returns `proxy_warnings` when a backend can't enforce `max_tokens`, `temperature`, or streaming.

## Benchmark and compile

```bash
# Benchmark against a dataset
venv/bin/tetraframe benchmark \
  --config configs/base.yaml \
  --dataset examples/benchmark_cases_test.jsonl \
  --out runs/benchmarks/report.json

# Compile an optimized program
venv/bin/tetraframe compile \
  --config configs/compile.yaml \
  --out runs/compiled_program.json
```

## Traces

Each run writes a JSONL trace at `runs/traces/<run_id>.jsonl`. Each row includes stage name, latency, backend info, and input/output digests.

```python
import json
from pathlib import Path

run = json.loads(Path("runs/latest/run.json").read_text())
trace_path = Path("runs/traces") / f"{run['run_id']}.jsonl"
for line in trace_path.read_text().splitlines()[:5]:
    row = json.loads(line)
    print(row["stage_name"], row["backend_name"], row["latency_ms"])
```

## Config reference

```yaml
model:
  runtime_model: openai/gpt-4.1-mini
  reflection_model: openai/gpt-4.1
  backend:
    provider: openai          # openai | anthropic | openrouter | openai-compatible | claude-code | codex | opencode
    model: gpt-4.1-mini
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

## Project structure

```
src/tetraframe/
  pipeline.py              # 9-stage tetralemmatic pipeline
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

## License

MIT
