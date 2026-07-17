# De-Id SLM — Context-Sensitive Personal-Name Judgment

Fine-tune **Qwen3-1.7B (QLoRA)** to do the one part of de-identification a prompt can't reliably do:
**context-sensitive personal-name judgment** — deciding, in context, whether a token is a person's
name or an identically-spelled non-person, every time, without drifting.

Pattern types (email / phone / ID) and output format are handled by regex + constrained decoding in
the surrounding pipeline. The **trained behavior, the dataset, and the eval are all about the
judgment core**.

> **Try it:** `python -m src.demo --adapter <sft-v3-gpt551 path>` runs the prompted base vs. the
> fine-tune side-by-side on the ambiguous cases. Submission steps (HF push, demo video):
> [`docs/submission-runbook.md`](docs/submission-runbook.md).

## The behavior (the gate)

> Given a passage of educational text (student essay or tutoring-chat turn), the model returns it
> **unchanged except** that every span referring to a real person's name is wrapped in
> `⟨NAME⟩…⟨/NAME⟩`, and **no other span is tagged**.

- **PASS** = all and only personal-name spans tagged; text otherwise byte-identical to input.
- **FAIL** = any missed name, any non-name tagged, or any other change to the text.

The hard cases (where fine-tune beats prompting): person-vs-eponym (`Newton`), person-vs-place
(`Chelsea`), person-vs-common-word (`Grace`, `Bishop`), first-name-only, possessives, third parties,
and negative traps.

## Results

Base is the prompted Qwen3-1.7B with no fine-tune; tuned is the QLoRA fine-tune (canonical
`gpt551` run). Aggregated over the quarantined hard-case eval sets — higher is better for
recall / pass / consistency, lower for the rest:

| Metric | Base (prompted) | Tuned (QLoRA) |
|---|--:|--:|
| Pass rate (spec-exact) | 0.35 | **0.82** |
| Name recall | 0.52 | **0.85** |
| Leakage (names missed) | 0.26 | **0.08** |
| Over-tagging (non-names tagged) | 0.55 | **0.16** |
| Text-integrity violations | 0.59 | **0.02** |
| Consistency across paraphrases | 0.38 | **0.56** |

Per-set pass rate climbs from **0.13–0.39** (base) to **0.78–0.87** (tuned) across five held-out
sets; on the naturally-worded sets the 1.7B fine-tune is competitive with `gpt-4.1` despite being
orders of magnitude smaller. Full tables with 95% confidence intervals:
[`docs/final-report.md`](docs/final-report.md), [`docs/results.md`](docs/results.md).

## Repo map

| Path | What |
|---|---|
| `docs/final-report.md` | Results write-up: base-vs-tuned, per-category breakdown, limitations. |
| `docs/results.md` | Every eval run with metric tables + confidence intervals. |
| `docs/model-card.md`, `docs/dataset-card-v3.md` | Model and dataset cards published to the HuggingFace Hub. |
| `docs/brainlift.md` | **Source of truth** — mandate, scope, facts (DOK 1–4), and spiky POVs (BrainLift v3). |
| `docs/plan.md` | The one-week build plan (Mon Jul 6 → Sun Jul 12). |
| `docs/STATUS.md` | Live "what's done / in-flight / next". Updated on every merge. |
| `docs/archive/` | Superseded version cards and daily process logs, kept for history. |

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

### Build / train / evaluate

```bash
make check                                                    # lint + format-check + tests (the PR gate)
python -m src.datagen.generate --config configs/datagen.yaml  # (re)generate the dataset
python -m src.train.qlora --config configs/train.yaml         # QLoRA train (Colab/CUDA)
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python -m src.train.qlora --config configs/train.mps.yaml   # LoRA train (Mac/MPS)
python -m src.eval.run --split eval/hardcases --model base    # eval the prompted base
python -m src.eval.run --split eval/hardcases --model tuned   # eval the fine-tune
python -m src.demo                                            # base-vs-tuned inference demo
```

See `docs/plan.md` for the day-by-day build arc.
