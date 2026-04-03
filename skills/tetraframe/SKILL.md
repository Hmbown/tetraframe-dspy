---
name: tetraframe
description: "Run the TetraFrame tetralemmatic reasoning pipeline on a question or problem. Use when the user wants to decompose a question into four independent positions (P, not-P, both, neither) and synthesize a non-averaging result. Triggers on /tetraframe, 'run tetraframe', 'tetralemma', or 'four corners analysis'."
metadata:
  author: Hunter Bown
  version: "0.1.0"
  style: instruction-only
  compatibility: Agent Skills compatible clients including Claude Code, OpenAI Codex, and OpenCode
---

# TetraFrame — Tetralemmatic Reasoning Pipeline

Run TetraFrame's 6-stage DSPy pipeline to decompose a question into four independent positions and synthesize a transformed output.

## What it does

Takes a seed question → extracts a predicate → generates four isolated corners (P, not-P, both, neither) → maps their contradictions and complementarities → synthesizes a non-averaging result → verifies quality.

## Quick start

The TetraFrame repo is at `/Volumes/VIXinSSD/tetralemma`.

### Run with auto-discovery (simplest)

```bash
cd /Volumes/VIXinSSD/tetralemma
venv/bin/tetraframe discover  # see available backends
venv/bin/tetraframe run "<seed>" --out runs/latest/run.json
```

### Run with a specific config

```bash
cd /Volumes/VIXinSSD/tetralemma
venv/bin/tetraframe run "<seed>" --config configs/openai.yaml --out runs/latest/run.json
```

### Available configs

| Config | Backend |
|---|---|
| `configs/openai.yaml` | OpenAI API (`OPENAI_API_KEY`) |
| `configs/anthropic.yaml` | Anthropic API (`ANTHROPIC_API_KEY`) |
| `configs/openrouter.yaml` | OpenRouter API (`OPENROUTER_API_KEY`) |
| `configs/claude_code_cli.yaml` | Claude Code CLI |
| `configs/codex_cli.yaml` | Codex CLI |
| `configs/opencode_cli.yaml` | OpenCode CLI |
| `configs/mimo_v2_pro.yaml` | Mimo v2 Pro via OpenCode |

## Reading results

After a run, read the output:

```python
import json
from pathlib import Path

run = json.loads(Path("/Volumes/VIXinSSD/tetralemma/runs/latest/run.json").read_text())

# Key sections
print(run["predicate_selection"]["primary_predicate"]["text"])
for mode in ["P", "not-P", "both", "neither"]:
    print(f"\n{mode}: {run['corners'][mode]['patched_claim']}")
print(f"\nP*: {run['transformed_frame']['transformed_predicate']}")
print(f"\nVerification: {run['verification']['aggregate_score']}")
```

## Writing good seeds

A seed should contain a genuine tension or decision. Good seeds:

- Frame a real tradeoff: "We keep framing X as A-or-B, but the real choice may be C."
- Surface a hidden assumption: "The team assumes X, but Y suggests otherwise."
- Identify a stuck decision: "We can't decide between X and Y because Z."

Bad seeds are vague ("How should we improve?") or already resolved ("Should we use React?").

## Pipeline stages

1. **Seed Distill** — extracts stakes, constraints, unknowns, hidden assumptions
2. **Predicate Selection** — picks the operational predicate for four-corner reasoning
3. **Four Corners** — generates P, not-P, both, neither in isolation (each self-critiqued)
4. **Cartography** — maps cross-corner relations, fair reconstructions, arbiter notes
5. **Transform** — synthesizes P* (non-averaging output from all four corners)
6. **Verify** — quality checks (branch independence, divergence, rigor, slop, fake novelty)

## Verification scores to watch

| Metric | What it catches |
|---|---|
| Branch independence | Corners leaked information to each other |
| Rigor of Both | "Both" is actually a compromise, not a genuine co-holding |
| Rigor of Neither | "Neither" is evasion, not a real frame diagnosis |
| Transformation quality | P* is just an average of the corners |
| Fake novelty risk | P* introduced unsupported terminology |
| Slop risk | Mushy language ("nuanced", "balanced", "consider") |
