# TetraFrame

**A structured reasoning engine that explores all four logical corners of a problem in parallel, then transforms them into useful outputs.**

## The Problem

When you reason about a hard question, you usually frame it as *A or not-A*. That framing anchors you — you explore one side, then react against it, and your "synthesis" ends up as a watered-down compromise.

**TetraFrame** rejects that. It forces you to explore all four corners simultaneously:

```
                         Predicate: "P"
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
         ┌─────────┐    ┌─────────┐    ┌─────────┐
         │    P    │    │ not-P   │    │  Both   │    Neither
         │         │    │         │    │(not a   │    (frame
         │ "enter  │    │"don't   │    │ compromise│   failure —
         │  the    │    │ enter"  │    │  — real  │   replace
         │ market" │    │         │    │  basis)" │   the
         └─────────┘    └─────────┘    └─────────┘   predicate)
```

| Corner | What it means | Example |
|--------|--------------|---------|
| **P** | The affirmative case | "We should enter the market" |
| **not-P** | The negation | "We should not enter the market" |
| **Both** | A genuine both, *not* a watered-down compromise | Two distinct commitments that are basis-typed and individually defensible |
| **Neither** | The frame is broken — the question itself is wrong | "Enter vs. not-enter is the wrong framing; the real choice is staged reversible commitments" |

Each corner is generated in isolation (no corner can see the others), then they are mapped, scored, and transformed into a final output that is explicitly tested against compromise language and fake novelty.

## The Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          9-STAGE PIPELINE                                │
│                                                                          │
│  ┌───────────┐    ┌───────────────┐    ┌───────────────────────────────┐ │
│  │  1. Seed  │───▶│  2. Predicate │───▶│  3. Four-Corner Generation    │ │
│  │ Distill   │    │   Selection   │    │  (P, not-P, Both, Neither     │ │
│  │           │    │               │    │   run in isolation)           │ │
│  └───────────┘    └───────────────┘    └───────────────────────────────┘ │
│        │                                                  │             │
│        ▼                                                  ▼             │
│  ┌───────────┐    ┌───────────────┐    ┌───────────────────────────────┐ │
│  │  4. Harden│───▶│ 5. Cartograph │───▶│  6. Arbitration               │ │
│  │  corners  │    │  (map and     │    │  (score corners, detect       │ │
│  │           │    │   compare)    │    │   collapse/contamination)     │ │
│  └───────────┘    └───────────────┘    └───────────────────────────────┘ │
│                                                │                         │
│                                                ▼                         │
│                  ┌───────────────┐    ┌───────────────────────────────┐  │
│                  │ 8. Domain     │◀───│  7. Transform                 │  │
│                  │  Adapt        │    │  (non-averaging synthesis     │  │
│                  │  (write, code,│    │   of all four corners)        │  │
│                  │   research…)  │    └───────────────────────────────┘  │
│                  └───────┬───────┘                                       │
│                          ▼                                               │
│                  ┌───────────────┐                                       │
│                  │ 9. Verify     │                                       │
│                  │  (rigor +     │                                       │
│                  │   quality)    │                                       │
│                  └───────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

**Stage details:**

| Stage | Name | What it does |
|-------|------|-------------|
| 1 | **Seed Distill** | Extracts the core reasoning seed: stakes, constraints, unknowns, hidden assumptions |
| 2 | **Predicate Selection** | Picks the operational predicate that the four corners will reason about |
| 3 | **Four-Corner Generation** | Generates P, not-P, Both, and Neither — each in complete isolation |
| 4 | **Hardening** | Strengthens each corner: checks for basis-typing, flags "Both" that are actually compromises |
| 5 | **Cartography** | Maps relationships between corners: overlaps, contradictions, collapse signals |
| 6 | **Arbitration** | Scores corners for rigor, detects near-duplicates and contamination |
| 7 | **Transform** | Synthesizes a non-averaging output from the hardened corners |
| 8 | **Domain Adapt** | Adapts the transformed frame into a domain-specific format (code, writing, plan, etc.) |
| 9 | **Verify** | Final quality checks: scores against compromise language, fake novelty, and rigor criteria |

## Quick Start

### Install

```bash
cd tetralemma
python -m venv venv
venv/bin/pip install -e ".[dev]"
```

Verify:

```bash
venv/bin/python -m pytest -q
```

### Run

```bash
venv/bin/tetraframe run \
  "We keep framing product expansion as enter or do not enter, but the real choice may be staged reversible commitments." \
  --config configs/base.yaml \
  --out runs/latest/run.json
```

That's it. The output is `runs/latest/run.json` — a structured artifact containing the full 9-stage trace.

### What you get

```
runs/
├── latest/
│   └── run.json                  # Full run artifact
└── traces/
    └── <run_id>.jsonl            # Per-stage trace with timing + backend info
```

## Supported Backends

TetraFrame can run against any of these. Use the config file that matches your provider.

### API Backends (recommended)

| Provider | Config | Env Var | Notes |
|---|---|---|---|
| **OpenAI** | `configs/openai.yaml` | `OPENAI_API_KEY` | Full feature support |
| **Anthropic** | `configs/anthropic.yaml` | `ANTHROPIC_API_KEY` | Full feature support |
| **OpenRouter** | `configs/openrouter.yaml` | `OPENROUTER_API_KEY` | Streaming, max_tokens, temperature |
| **Any OpenAI-compatible** | `configs/openai_compatible.yaml` | (custom) | DeepSeek, NVIDIA NIM, vLLM, Ollama, etc. |

### CLI Backends (slower, fewer controls)

| CLI Tool | Config | Requires |
|---|---|---|
| **Claude Code** | `configs/claude_code_cli.yaml` | `npm install -g @anthropic-ai/claude-code` |
| **Codex** | `configs/codex_cli.yaml` | `codex` on PATH |
| **OpenCode** | `configs/opencode_cli.yaml` | `opencode` on PATH |

CLI backends run as subprocesses. They're slower and can't enforce `max_tokens` or `temperature`.

### Capabilities at a glance

```
               streaming   max_tokens   temperature   JSON   tool use
OpenAI           ✅           ✅            ✅          ✅       ✅
Anthropic        ✅           ✅            ✅          ✅       ✅
OpenRouter       ✅           ✅            ✅          ❌       ❌
OpenAI-compat    ✅           ✅            ✅          ❌       ❌
Claude CLI       ❌           ❌            ❌          ❌       ❌
Codex CLI        ❌           ❌            ❌          ❌       ❌
OpenCode CLI     ❌           ❌            ❌          ❌       ❌
```

### Configure a backend

```bash
# OpenAI (default)
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# OpenAI-compatible (e.g. DeepSeek)
# No env var needed if key is in config, or:
export DEEPSEEK_API_KEY="sk-..."
```

Then point your run at the config:

```bash
venv/bin/tetraframe run "Your seed" --config configs/anthropic.yaml --out runs/latest/run.json
```

## Auto-Discovery (Tool Plugins)

TetraFrame can find available backends automatically — no config file needed.

```bash
# See what's available
venv/bin/tetraframe discover

# Run with zero config (auto-selects best available)
venv/bin/tetraframe run "Your reasoning seed"

# Force a specific tool
venv/bin/tetraframe run "Your reasoning seed" --tool openai-api
```

Priority: Hermes credential pools > API backends with keys > CLI tools on PATH.

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

## Proxy (OpenAI-compatible HTTP server)

The proxy wraps any configured backend behind an OpenAI-compatible API. Point DSPy, curl, or any HTTP client at it.

```bash
# With a config file
venv/bin/tetraframe-proxy --config configs/claude_code_cli.yaml

# Or with env vars
export TETRAFRAME_PROXY_BIN="/path/to/claude"
export TETRAFRAME_PROXY_MODEL="claude-sonnet-4-6"
venv/bin/tetraframe-proxy --host 127.0.0.1 --port 8765
```

**Endpoints:**

| Route | Method | Description |
|---|---|---|
| `/health` | GET | Backend status and capabilities |
| `/v1/models` | GET | Available models |
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |

Select backends per-request with the `X-Backend` header:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Backend: claude-code" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

The proxy returns `proxy_warnings` when a backend can't enforce `max_tokens`, `temperature`, or streaming — these are explicit, not hidden.

## Benchmarking

```bash
venv/bin/tetraframe benchmark \
  --config configs/base.yaml \
  --dataset examples/benchmark_cases_test.jsonl \
  --out runs/benchmarks/report.json
```

## Compilation

```bash
venv/bin/tetraframe compile \
  --config configs/compile.yaml \
  --out runs/compiled_program.json
```

Outputs:
- `runs/compiled_program.json` — the optimized program
- `runs/compiled_program.eval.json` — evaluation report

## Inspecting Traces

Each run produces a JSONL trace file at `runs/traces/<run_id>.jsonl`. Each line contains the stage name, latency, backend info, and input/output digests.

```bash
venv/bin/python - <<'PY'
import json
from pathlib import Path

run = json.loads(Path("runs/latest/run.json").read_text())
trace_path = Path("runs/traces") / f"{run['run_id']}.jsonl"
for line in trace_path.read_text().splitlines()[:5]:
    row = json.loads(line)
    print(row["stage_name"], row["backend_name"], row["latency_ms"])
PY
```

## Configuration Reference

Backend config lives under `model.backend`:

```yaml
model:
  runtime_model: openai/gpt-4.1-mini
  reflection_model: openai/gpt-4.1        # for compilation
  backend:
    provider: openai                       # openai | anthropic | openrouter | openai-compatible | claude-code | codex | opencode
    model: gpt-4.1-mini
    base_url: null                         # for openai-compatible endpoints
    api_key_env: null                      # env var name holding the API key
    timeout: 120.0
    max_tokens: null
    temperature: null
    binary: null                           # CLI: path to binary
    cli_args: []                           # CLI: extra args
    env_passthrough: []                    # CLI: env vars to forward
    max_retries: 2
    retry_delay: 1.0
  reflection_backend: null                 # optional separate backend for compilation
```

## Project Structure

```
src/tetraframe/
  pipeline.py            # 9-stage tetralemmatic pipeline
  modules.py             # DSPy modules (one per stage)
  signatures.py          # DSPy signatures
  artifacts.py           # Typed run artifacts
  guards.py              # Isolation and anti-collapse guards
  metrics.py             # Verification and scoring
  tracing.py             # Per-run trace logger
  config.py              # Config model
  cli.py                 # CLI entry points (tetraframe)
  compile.py             # Compilation strategy
  backends/              # Pluggable backend system
    base.py              # Backend protocol + capabilities
    api.py               # API backends (OpenAI, Anthropic, etc.)
    cli_base.py          # CLI backend base class
    cli_claude.py        # Claude Code adapter
    cli_codex.py         # Codex adapter
    cli_opencode.py      # OpenCode adapter
    factory.py           # build_backend(), build_dspy_lm()
  tools/                 # Tool plugin system
    protocol.py          # ModelTool protocol
    registry.py          # Auto-discovery registry
    api_tool.py          # Direct API tools
    cli_tool.py          # CLI-based tools
    hermes_tool.py       # Hermes credential pool integration
  proxy/
    server.py            # Multi-backend OpenAI-compatible proxy
    client.py            # Backward-compatible CLI shim
  benchmarks/
    harness.py           # Benchmark runner

configs/
  base.yaml              # Default (OpenAI)
  openai.yaml            # OpenAI direct
  anthropic.yaml         # Anthropic direct
  openrouter.yaml        # OpenRouter
  openai_compatible.yaml # Generic OpenAI-compatible endpoint
  claude_code_cli.yaml   # Claude Code CLI
  codex_cli.yaml         # Codex CLI
  opencode_cli.yaml      # OpenCode CLI
  compile.yaml           # Compilation settings
  benchmarks.yaml        # Benchmark settings
```

## Why "tetralemmatic"?

Standard binary reasoning (A vs. not-A) has a well-known problem: the "synthesis" tends to be a compromise that inherits the weaknesses of both sides. Tetralemmatic reasoning — from the Aristotelian *tetralemma* — explicitly models four positions:

1. **P** — the thing is true
2. **not-P** — the thing is false
3. **Both** — it's true in some respects and false in others (but as distinct basis-typed claims, not a mushy middle)
4. **Neither** — the question itself is malformed and needs to be reframed

TetFrame enforces this structurally: branches are isolated until cartography, "Both" must be basis-typed (not a compromise), and "Neither" must propose a replacement predicate. The transformation step is explicitly tested against compromise language and fake novelty.

## License

MIT
