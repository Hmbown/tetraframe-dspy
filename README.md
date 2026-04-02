# TetraFrame

TetraFrame is a DSPy-style tetralemmatic reasoning package built around a typed
9-stage pipeline:

1. seed distillation
2. predicate selection
3. isolated four-corner generation
4. hardening
5. cartography
6. arbitration
7. non-averaging transformation
8. domain adaptation
9. verification

The package is opinionated about tetralemmatic rigor:

- `both` must be basis-typed, not compromise.
- `neither` must diagnose frame failure and replace the predicate or frame.
- branch generation stays isolated until cartography.
- transformed frames are scored against compromise language and fake novelty.

## Supported Backends

TetraFrame runs against real model providers through a pluggable backend system.

### API Backends

| Provider | Config | Env Var | Notes |
|---|---|---|---|
| OpenAI | `configs/openai.yaml` | `OPENAI_API_KEY` | Full feature support |
| Anthropic | `configs/anthropic.yaml` | `ANTHROPIC_API_KEY` | Full feature support |
| OpenRouter | `configs/openrouter.yaml` | `OPENROUTER_API_KEY` | Streaming, max_tokens, temperature |
| OpenAI-compatible | `configs/openai_compatible.yaml` | (custom) | Any endpoint with OpenAI-compatible API |

### CLI Backends

| CLI Tool | Config | Binary | Limitations |
|---|---|---|---|
| Claude Code | `configs/claude_code_cli.yaml` | `claude` | No max_tokens, no temperature, no streaming |
| Codex | `configs/codex_cli.yaml` | `codex` | No max_tokens, no temperature, no streaming, no usage reporting |
| OpenCode | `configs/opencode_cli.yaml` | `opencode` | No max_tokens, no temperature, no streaming, no usage reporting |

CLI backends invoke the tool via subprocess. They are slower than API backends
and cannot enforce `max_tokens` or `temperature`. The proxy and traces report
explicit warnings when these limitations apply.

## Install

Use the repo venv. Do not rely on bare `python` or `pip`.

```bash
cd tetralemma
python -m venv venv
venv/bin/pip install -e ".[dev]"
```

Sanity check:

```bash
venv/bin/python -m pytest -q
```

## Configure A Backend

### OpenAI (default)

```bash
export OPENAI_API_KEY="sk-..."
```

Config: `configs/openai.yaml` or `configs/base.yaml` (same provider).

### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Config: `configs/anthropic.yaml`

### OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

Config: `configs/openrouter.yaml`

### Generic OpenAI-compatible endpoint

Edit `configs/openai_compatible.yaml`:

```yaml
model:
  backend:
    provider: openai-compatible
    model: my-local-model
    base_url: "http://localhost:8080/v1"
    # api_key_env: MY_API_KEY  # if your endpoint requires auth
```

### Claude Code CLI

Requires Claude Code installed: `npm install -g @anthropic-ai/claude-code`

```bash
which claude  # verify it's on PATH
```

Config: `configs/claude_code_cli.yaml`

To override the binary path:

```yaml
model:
  backend:
    provider: claude-code
    binary: /absolute/path/to/claude
```

### Codex CLI

Requires Codex installed and on PATH.

Config: `configs/codex_cli.yaml`

### OpenCode CLI

Requires OpenCode installed and on PATH.

Config: `configs/opencode_cli.yaml`

## Run The Pipeline

```bash
venv/bin/tetraframe run \
  "We keep framing product expansion as enter or do not enter, but the real choice may be staged reversible commitments." \
  --config configs/base.yaml \
  --out runs/latest/run.json
```

With Anthropic:

```bash
venv/bin/tetraframe run \
  "Your reasoning seed" \
  --config configs/anthropic.yaml \
  --out runs/latest/run.json
```

With Claude Code CLI:

```bash
venv/bin/tetraframe run \
  "Your reasoning seed" \
  --config configs/claude_code_cli.yaml \
  --out runs/latest/run.json
```

Python API:

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

Outputs:

- run artifact: `runs/latest/run.json`
- per-run trace log: `runs/traces/<run_id>.jsonl`

Trace files now include backend metadata: `backend_name`, `backend_kind`,
`backend_model`, `execution_mode`, and `capability_warnings` per stage.

## Inspect Traces

Each JSONL row is a typed stage trace with:

- `stage_name`
- `attempt`
- `visible_input_fields`
- `blocked_input_fields`
- `latency_ms`
- `retry_reason`
- `backend_name` / `backend_kind` / `backend_model`
- `execution_mode` (`direct`, `proxy`, `cli`)
- `capability_warnings`
- input and output digests

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

## Benchmark

```bash
venv/bin/tetraframe benchmark \
  --config configs/base.yaml \
  --dataset examples/benchmark_cases_test.jsonl \
  --out runs/benchmarks/report.json
```

With a different backend:

```bash
venv/bin/tetraframe benchmark \
  --config configs/anthropic.yaml \
  --dataset examples/benchmark_cases_test.jsonl \
  --out runs/benchmarks/report.json
```

## Compile

```bash
venv/bin/tetraframe compile \
  --config configs/compile.yaml \
  --out runs/compiled_program.json
```

Artifacts:

- compiled program: `runs/compiled_program.json`
- compile evaluation report: `runs/compiled_program.eval.json`

## Proxy

The proxy exposes an OpenAI-compatible HTTP surface in front of any configured
backend. External tools (DSPy clients, curl, other LLM libraries) can send
requests to it.

### Start the proxy

With a config file (recommended):

```bash
venv/bin/tetraframe-proxy --config configs/claude_code_cli.yaml
```

With env vars (backward compat):

```bash
export TETRAFRAME_PROXY_BIN="/path/to/claude"
export TETRAFRAME_PROXY_MODEL="claude-sonnet-4-6"
venv/bin/tetraframe-proxy --host 127.0.0.1 --port 8765
```

### Endpoints

| Route | Method | Description |
|---|---|---|
| `/health` | GET | Backend status, capabilities, warnings |
| `/v1/models` | GET | List models across all registered backends |
| `/v1/chat/completions` | POST | Chat completions (streaming and non-streaming) |

### Call the proxy with curl

```bash
curl http://127.0.0.1:8765/health | python -m json.tool

curl http://127.0.0.1:8765/v1/models | python -m json.tool

curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "What is tetralemmatic reasoning?"}]
  }'
```

Streaming:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Backend selection

Via `X-Backend` header:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Backend: claude-code" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### Point another client at the proxy

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="not-needed",
)
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

```python
import dspy

lm = dspy.LM("openai/claude-sonnet-4-6", api_base="http://127.0.0.1:8765/v1", api_key="not-needed")
dspy.configure(lm=lm)
```

### Proxy warnings

The proxy returns a `proxy_warnings` field in responses when:

- `max_tokens` is requested but the backend cannot enforce it
- `temperature` is requested but the backend cannot enforce it
- Streaming is requested but the backend delivers the full response as one chunk

These are explicit, not hidden.

## Backend Capabilities

| Capability | OpenAI | Anthropic | OpenRouter | OpenAI-compat | Claude CLI | Codex CLI | OpenCode CLI |
|---|---|---|---|---|---|---|---|
| streaming | yes | yes | yes | yes | no | no | no |
| max_tokens | yes | yes | yes | yes | no | no | no |
| temperature | yes | yes | yes | yes | no | no | no |
| structured JSON | yes | yes | no | no | no | no | no |
| tool use | yes | yes | no | no | no | no | no |

## Configuration Reference

Backend config lives under `model.backend`:

```yaml
model:
  runtime_model: openai/gpt-4.1-mini   # litellm model string (legacy compat)
  reflection_model: openai/gpt-4.1     # for compilation
  backend:
    kind: api                # "api" or "cli" (inferred from provider)
    provider: openai         # openai, anthropic, openrouter, openai-compatible,
                             # claude-code, codex, opencode
    model: gpt-4.1-mini
    base_url: null           # for openai-compatible endpoints
    api_key_env: null        # env var name holding the API key
    timeout: 120.0
    max_tokens: null         # default max_tokens for all calls
    temperature: null        # default temperature for all calls
    binary: null             # CLI: path to binary (auto-detected from PATH)
    cli_args: []             # CLI: extra args passed to the binary
    env_passthrough: []      # CLI: env vars to forward to subprocess
    max_retries: 2
    retry_delay: 1.0
  reflection_backend: null   # optional separate backend for compilation
```

## Project Structure

```
src/tetraframe/
  backends/
    __init__.py          # exports
    base.py              # Backend protocol, capabilities, metadata
    api.py               # API backend (wraps dspy.LM)
    cli_base.py          # CLI backend base class
    cli_claude.py        # Claude Code adapter
    cli_codex.py         # Codex adapter
    cli_opencode.py      # OpenCode adapter
    dspy_adapter.py      # CLILanguageModel (DSPy-compatible CLI wrapper)
    factory.py           # build_backend(), build_dspy_lm()
  proxy/
    server.py            # Multi-backend OpenAI-compatible proxy
    client.py            # Backward-compatible Claude CLI shim
  pipeline.py            # 9-stage tetralemmatic pipeline
  modules.py             # DSPy modules
  signatures.py          # DSPy signatures
  artifacts.py           # Typed artifacts
  guards.py              # Isolation and anti-collapse guards
  metrics.py             # Verification and scoring
  tracing.py             # Per-run trace logger with backend metadata
  config.py              # Config model with backend support
  cli.py                 # CLI entry points
  compile.py             # Compilation strategy
  benchmarks/harness.py  # Benchmark runner
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
```

## Verified Commands

```bash
venv/bin/pip install -e ".[dev]"
venv/bin/python -m pytest -q
venv/bin/tetraframe run "..." --config configs/base.yaml --out runs/latest/run.json
venv/bin/tetraframe benchmark --config configs/base.yaml --dataset examples/benchmark_cases_test.jsonl --out runs/benchmarks/report.json
venv/bin/tetraframe compile --config configs/compile.yaml --out runs/compiled_program.json
venv/bin/tetraframe-proxy --config configs/claude_code_cli.yaml
venv/bin/tetraframe-proxy --host 127.0.0.1 --port 8765
```
