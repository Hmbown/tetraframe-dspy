# TetraFrame

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A [DSPy](https://dspy.ai/) pipeline that splits a question into four independent positions (tetralemma), then synthesizes them into a single output. Each stage is a DSPy module with typed signatures — the whole pipeline is compilable with BootstrapFewShot, MIPROv2, and GEPA.

```bash
pip install -e ".[dev]"
tetraframe run "Your question or problem statement" --config configs/base.yaml
```

## What is a tetralemma?

A tetralemma (Sanskrit: catuṣkoṭi) is a four-cornered logical structure from Buddhist logic, developed most fully by Nāgārjuna (~2nd century CE). Where binary reasoning gives you two options —

```
A or not-A
```

— a tetralemma gives you four:

```
P · not-P · Both · Neither
```

Binary framing tends to collapse into compromise — state a thesis, react against it, split the difference. A tetralemma sidesteps this by exploring all four corners independently before any synthesis happens.

TetraFrame enforces this structurally:

- **P** and **not-P** are generated in isolation from each other
- **Both** must produce two distinct positions that hold P and not-P simultaneously — not a compromise
- **Neither** must reject the original framing and propose a replacement predicate
- No corner can see the other corners until the cartography stage

## The pipeline

TetraFrame takes a seed question, decomposes it into four corners, maps their relationships, and synthesizes a result that isn't a compromise between them.

```
Seed ─▶ Predicate ─▶ Four Corners ─▶ Cartography ─▶ Transform ─▶ Verify
```

Six stages, run in sequence:

| # | Stage | Does |
|---|-------|------|
| 1 | Seed Distill | Extracts stakes, constraints, unknowns, and hidden assumptions from the input |
| 2 | Predicate Selection | Picks the operational predicate the four corners will reason about |
| 3 | Four Corners | Generates and hardens P, not-P, Both, Neither — each in complete isolation, self-critiqued in one pass |
| 4 | Cartography | Maps contradictions, complementarities, and collapse signals; produces fair reconstructions and arbiter notes |
| 5 | Transform | Synthesizes an output from all four corners without averaging them |
| 6 | Verify | Final quality checks against compromise language and fake novelty |

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

CLI backends run as subprocesses — slower, and the proxy warns when requested parameters can't be enforced.

### Setting API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
```

### Auto-discovery

TetraFrame can find backends automatically without a config file.

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

## Example run

The following is from a real run where TetraFrame analyzed its own pipeline. The seed asked whether TetraFrame's multi-stage architecture produces genuine reasoning improvements or sophisticated restatement. Backend: Claude Sonnet 4 via CLI.

<details>
<summary><b>Stage 1 — Seed Distill</b> (61s)</summary>

Extracts structure from the raw seed. Output includes stakes, constraints, unknowns, hidden assumptions, and candidate predicates.

**Normalized seed:** TetraFrame is a multi-stage LLM call reasoning pipeline built on tetralemmatic decomposition. It faces three separable problems: (A) an unresolved architectural direction — whether to consolidate calls for cost/latency or deepen stage isolation for quality; (B) a verification gap — the current suite detects slop heuristically but cannot distinguish genuine dialectical transformation from sophisticated restatement; and (C) a closed-loop gap — downstream consumers have no signal path back to upstream stages.

**Example hidden assumption:** That the current stage count is the correct granularity — rather than an artifact of how the system was initially built.

</details>

<details>
<summary><b>Stage 2 — Predicate Selection</b> (152s)</summary>

Picks the operational predicate the four corners will reason about.

**Selected predicate:** TetraFrame's multi-stage pipeline produces outputs that are measurably distinguishable from sophisticated restatement of their inputs, as determined by a cross-validated binary classifier operating on embedding distance, structural novelty, and logical form change dimensions with precision and recall above 0.80 on held-out labeled examples.

**Rationale (excerpt):** The primary predicate was selected over the architectural consolidation question because it is the stated prerequisite: Problem B must be resolved before Problem A can be meaningfully addressed. A predicate about consolidation would produce a tetralemma whose corners all reduce to "we don't know yet" until a quality metric exists — that is a degenerate tetralemma.

</details>

<details>
<summary><b>Stage 3 — Four Corners</b> (P: 63s, not-P: 129s, Both: 218s, Neither: 306s)</summary>

Each corner is generated and self-critiqued in isolation — no corner can see the others.

**P:** TetraFrame's pipeline produces outputs measurably distinguishable from sophisticated restatement because tetralemmatic decomposition mechanically requires logical form change, structural novelty, and semantic distance as preconditions for stage completion — not as emergent side effects.

**not-P:** The predicate cannot be validated by the instrument it proposes, because the instrument requires as input the very judgment whose absence motivates the predicate. The three classifier dimensions measure visible text change, not dialectical transformation.

**Both:** TetraFrame's pipeline both is and is not measurably distinguishable from sophisticated restatement: it IS at the formal-structural measurement layer where a classifier will reliably exceed 0.80; it is NOT at the semantic-functional layer where dialectical transformation is defined to matter — because classifier success at the structural layer does not confirm that genuine tetralemmatic transformation occurred.

**Neither:** The transformation/restatement binary is a false discretization of a continuous, domain-relative, evaluator-relative gradient. The predicate's operationalization measures the detectability of a proxy, not the existence of the underlying construct.

</details>

<details>
<summary><b>Stage 4 — Cartography</b> (1034s)</summary>

First stage where all four corners are visible together. Maps pairwise relations, produces fair reconstructions, and writes arbiter notes.

**Example contradiction (P × not-P):** P's sufficiency claim — classifier > 0.80 on human-labeled examples constitutes validation of genuine transformation — directly contradicts not-P's structural circularity objection: a classifier trained on direct transformation/restatement judgments cannot validate the judgment it is trained to replace.

**Arbiter notes (excerpt):** The four-corner analysis is not primarily a debate between competing hypotheses about TetraFrame. It is a progressively fine-grained examination of what kind of evidence is meaningful at what ontological altitude. The only genuinely adversarial relation is P × not-P on the actionable empirical question.

</details>

<details>
<summary><b>Stage 5 — Transform</b> (379s)</summary>

Synthesizes a single output from all four corners without averaging them.

**Transformed predicate:** TetraFrame's multi-stage pipeline produces a per-stage measurable effect-size lift on a pre-registered continuous transformation rubric relative to a compute-matched single-call baseline, where: (1) the rubric dimensions are validated against at least one domain with formal transformation criteria before any pipeline scoring begins; (2) the lift is reported per-domain and per-stage to expose whether the corner-switching structure rather than token budget is the operative causal variable.

**Why it's not an average:** P* does not retain the binary frame with additional checks, does not soften P's sufficiency claim to a sufficiency claim with caveats, and does not accept not-P's rejection while proposing a weaker affirmation. The transformation is structural — replacing the binary classification frame entirely with a continuous, construct-validated, stage-decomposed attribution frame.

</details>

<details>
<summary><b>Stage 6 — Verify</b> (aggregate: 0.901)</summary>

Runs quality checks across 9 metrics. This run scored:

| Metric | Score | Pass |
|---|---|---|
| Branch independence | 1.000 | yes |
| Divergence quality | 0.904 | yes |
| Rigor of Both | 1.000 | yes |
| Rigor of Neither | 0.938 | yes |
| Contradiction honesty | 1.000 | yes |
| Transformation quality | 1.000 | yes |
| Robustness | 0.879 | yes |
| Fake novelty risk | 0.400 | **no** |
| Slop risk | 0.884 | yes |

The fake novelty flag fired on the transformed predicate — the verifier flagged unsupported new concepts in P*. This is a real check, not a false positive: the transformed predicate introduced terminology ("construct-validated rubric", "stage-decomposed attribution") that wasn't grounded in the corners' evidence base.

</details>

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

## License

MIT
