# De-Id SLM — Context-Sensitive Personal-Name Judgment

Fine-tune **Qwen3-1.7B (QLoRA)** to do the one part of de-identification a prompt can't reliably do:
**context-sensitive personal-name judgment** — deciding, in context, whether a token is a person's
name or an identically-spelled non-person, every time, without drifting.

Pattern types (email / phone / ID) and output format are handled by regex + constrained decoding in
the surrounding pipeline. The **trained behavior, the dataset, and the eval are all about the
judgment core**.

## The behavior (the gate)

> Given a passage of educational text (student essay or tutoring-chat turn), the model returns it
> **unchanged except** that every span referring to a real person's name is wrapped in
> `⟨NAME⟩…⟨/NAME⟩`, and **no other span is tagged**.

- **PASS** = all and only personal-name spans tagged; text otherwise byte-identical to input.
- **FAIL** = any missed name, any non-name tagged, or any other change to the text.

The hard cases (where fine-tune beats prompting): person-vs-eponym (`Newton`), person-vs-place
(`Chelsea`), person-vs-common-word (`Grace`, `Bishop`), first-name-only, possessives, third parties,
and negative traps.

## Repo map

| Path | What |
|---|---|
| `docs/brainlift.md` | **Source of truth** — mandate, scope, facts (DOK 1–4), and spiky POVs (BrainLift v3). |
| `docs/plan.md` | The one-week build plan (Mon Jul 6 → Sun Jul 12). |
| `docs/STATUS.md` | Live "what's done / in-flight / next". Updated on every merge. |
| `docs/agent-workflow-starter-kit.md` | The multi-agent workflow reference this repo's process is built on. |
| `AGENTS.md` | Always-loaded agent rules: hard ceilings + skill routing. |
| `.cursor/skills/` | Project skills: `shipping-changes`, `building-and-testing` (mirrored to `.claude/skills/` for Claude Code). |
| `Makefile` / `.github/workflows/ci.yml` | Code-quality gate: `make check` (lint + format-check + tests), enforced on every push/PR. |

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Training backend (auto-selected by hardware)

- **Apple Silicon (Mac):** local LoRA via PyTorch + PEFT on the `mps` device — the default when no CUDA
  GPU is present. Train with `configs/train.mps.yaml`, prefixed by `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Colab / CUDA:** 4-bit QLoRA via Unsloth with `configs/train.yaml`. `unsloth`/`bitsandbytes` are
  Linux-gated in `requirements.txt`, so a Mac `pip install` skips them.

Exact build / run / train / eval commands live in `.cursor/skills/building-and-testing/SKILL.md`.

### Code quality (the green gate)

```bash
make check   # ruff check + ruff format --check + pytest — run before every PR
make fix     # auto-fix lint + apply formatting
```

`make check` mirrors the `code-quality` CI workflow, so a green run locally == a green tick on the PR.
It's the "verify what's verifiable" step the `shipping-changes` skill relies on.

See `docs/plan.md` for the day-by-day arc and `AGENTS.md` for how to ship changes here.
