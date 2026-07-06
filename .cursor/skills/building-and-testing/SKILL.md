---
name: building-and-testing
description: Use when you need to build, run, train, or evaluate any surface of this project — to satisfy a PR's build/test gate or to get a clean baseline.
---

# Building and Testing

Canonical commands for the De-Id SLM project. Treat this like a doc: the first time you run a command
and confirm it, mark it **verified**; if one is wrong, fix it here in the same change.

## Environment
One-time setup:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
> Status: NOT yet verified on this machine — resolve on first run (Day 1). Pin exact versions once the
> GPU environment is stood up.

## Commands (fill real paths as the pipeline is built)

| Goal | Command |
|---|---|
| Lint / format | `ruff check . && ruff format --check .` |
| Run all tests | `pytest -q` |
| Run behavioral-check tests only | `pytest tests/test_behavioral_checks.py -q` |
| Generate dataset | `python -m src.datagen.generate --config configs/datagen.yaml` |
| QLoRA train | `python -m src.train.qlora --config configs/train.yaml` |
| Eval (base vs. tuned, hard cases) | `python -m src.eval.run --split eval/hardcases --model <base|tuned>` |
| Inference demo | `python -m src.demo` |

> Status of the ML commands: NOT yet verified — these are the intended entry points; wire them up as
> the pipeline lands (Days 2–3) and mark verified once each runs green.

## The eval must exist before training
Per the hard ceilings (`AGENTS.md`): do not run the train command until the eval command and the
behavioral-check tests exist and pass. `docs/STATUS.md` should show the eval harness as Done first.

## Don't recompute from zero (if runs are slow)
Cache the frontier-teacher generations and the tokenized dataset so a rerun doesn't re-hit the API or
re-tokenize. Reviewers trust the builder's posted eval numbers and spot-check the ceiling items
(leakage / integrity) rather than re-running full training.

## Common mistakes
- Forgetting `enable_thinking=False` — Qwen3 defaults to thinking mode; the behavior needs non-thinking.
- Training on a split that overlaps the quarantined eval set (hard-ceiling violation — check overlap).
- Non-completion-only loss masking — mask the prompt so loss is only on the tagged output.
