# Completion checklist — what's missing to ship, step by step

_Prepared 2026-07-11 · Companion to [`docs/final-report.md`](../final-report.md) and
[`docs/archive/next-steps-testing.md`](next-steps-testing.md). This is the honest "what's left" list mapped to
the plan's submission package (`docs/plan.md` → "Final submission package")._

## Headline

**The hard technical bet is won and documented** — the canonical live-teacher (gpt551) run beats the
prompted base on every privacy axis, ceilings held, leakage independently re-verified (0 overlap), plus a
frontier comparison the plan never even required. The suite is green (192 passed). **What's missing is
almost entirely the shipping/packaging layer** (Day 5–7): a real demo, HF Hub publication, the demo
video, the BrainLift verdict write-up — plus the one unbuilt planned feature (DPO). None of the remaining
blockers require touching training or risk an eval-leakage ceiling.

Two documentation-staleness items are also fixed as part of this pass (STATUS "In flight" was stale;
`report-nontechnical.md` predated gpt551).

---

## Submission package status

> **Update 2026-07-12 (branch `worktree-ship-submission`):** the code/writing layer for every
> remaining item is now built and green (202 passed) — real `src/demo.py`, consolidated
> `MODEL_CARD.md`, `scripts/push_to_hub.py` + `scripts/push_dataset.py` (both eval-leakage guarded),
> and the BrainLift verdict. What's left is **execution only**: run the three cells in
> [`submission-runbook.md`](../submission-runbook.md) on Colab (push model, push dataset, record demo).
> The table below is updated to reflect that.

| # | Checklist item | Status | What proves it / what's missing |
|---|---|---|---|
| 1 | Dataset published, quarantine documented | **READY TO PUSH** | In-repo + documented + leakage-guarded + human-reviewed. `scripts/push_dataset.py` publishes it (guard: 0 quarantine, 0 eval overlap, verified). **Left:** run the push cell. |
| 2 | Model on HF Hub + running demo | **READY TO PUSH** | Real `src/demo.py` (base-vs-tuned side-by-side, tested) + consolidated `MODEL_CARD.md` + `scripts/push_to_hub.py` (eval-leak guarded). **Left:** run the push cell (adapter is on Drive). |
| 3 | Eval harness + results table (base vs tuned, deltas) | **DONE** | `src/eval/*` + `docs/results.md` + 4-engine × 5-set matrix + this report. |
| 4 | BrainLift verdict (did data→behavior hold) | **DONE** | `docs/brainlift.md` → *Empirical verdict [v3]* resolves SPOV-7 with the gpt551 numbers. |
| 5 | 3–5 min demo video | **READY TO RECORD** | Depends only on running `python -m src.demo` (Step 1 of the runbook). |

---

## Step-by-step — blockers to complete the submission

Ordered so each step unblocks the next. Each is an upload/record/write task on top of finished artifacts.

### Step 1 — Build a real inference demo (`src/demo.py`)  ⟶ blocks Steps 2 & 5
- Load the tuned adapter (`sft-v3-gpt551`) over the 4-bit base; expose a `tag(passage) -> tagged` call.
- Ship a **base-vs-tuned side-by-side** on a canned set of ambiguous passages ("Newton was frustrated"
  [tag] vs "the Newton method" [don't]; "I visited Chelsea" [don't] vs "Chelsea helped me" [tag]).
- **Do not** use the `pipeline/cli.py --demo` stub for this — its README labels it a heuristic non-model
  tagger. It's fine for the *pattern-PII* demo, not the judgment demo.
- **Acceptance:** running it live shows the prompted base wobbling/over-tagging while the tune holds.
- _Dependency:_ needs GPU access to load the adapter (Colab or a 24GB card). Eval-only; no ceiling risk.

### Step 2 — Push the model + a consolidated model card to HF Hub  ⟶ checklist #2
- Upload the `sft-v3-gpt551` LoRA adapter (133 MB, on Drive) to the Hub.
- Write a single `MODEL_CARD.md` (base model, LoRA config, dataset hash, base-vs-tuned numbers from the
  final report, intended-use + limitations). The per-run cards (`model-card-gpt551.md` etc.) are the
  source material.
- **Acceptance:** a public model page with a loadable adapter + the headline numbers.
- _No ceiling conflict._ Keep the quarantined eval OUT of anything uploaded.

### Step 3 — Publish the dataset as a first-class artifact  ⟶ checklist #1
- Push the v3 live-teacher splits (818/90) to the HF Datasets Hub (or a clearly-linked release) with
  `dataset-card-v3.md` and the ambiguous-case breakdown.
- **Hard-ceiling constraint:** the quarantined eval sets stay physically separate and are **never**
  included — already enforced by `tests/test_no_eval_leakage.py`; preserve that on publish.
- **Acceptance:** external dataset page, eval sets demonstrably absent from it.

### Step 4 — Write the BrainLift empirical verdict  ⟶ checklist #4
- Add a verdict section to `docs/brainlift.md` (or a linked `brainlift-verdict.md`) resolving SPOV 7's
  falsifiable bet with the gpt551 numbers: *did fine-tune beat prompting on the hard cases?* (Yes —
  pass 0.13–0.39 → 0.61–0.97, leakage cut into the single/low-double digits, competitive with frontier
  on natural prose.)
- **Acceptance:** the source-of-truth doc states the outcome, win-or-lose, with a pointer to the report.
- _Pure writing task; evidence already exists._

### Step 5 — Record the 3–5 min demo video  ⟶ checklist #5
- Screen-record Step 1's demo on the ambiguous passages, narrating the base-vs-tuned contrast.
- **Acceptance:** a 3–5 min video linked from the README / submission.
- _Dependency:_ Step 1 must be working first.

---

## Nice-to-have (not blocking submission)

| Item | Status | Notes |
|---|---|---|
| **DPO (stretch rung 1)** | **UNBUILT** | The one genuinely-unbuilt planned feature. `data/splits/dpo.jsonl` + `configs/dpo.yaml` + `src/train/dpo.py`; report the delta over SFT. Training-method stretch, not an HP tweak. See `next-steps-testing.md` §4.1. |
| **`docs/error-analysis-final.md`** | **MISSING** | Day-6 S6.4 wanted a final "where it still fails" paragraph with quoted examples. Residuals already noted in `results.md` (possessive over-tag, person_vs_common flat recall); this is a consolidation write-up. |
| **Span-offset output as headline** | **PARTIALLY BUILT** | Offset-projection is merged in `pipeline/`; making the projected path the headline integrity number needs a Colab CRAPII run. See `next-steps-testing.md` §2.4. |
| **Single-token tag A/B (rung 4)** | **UNBUILT** | Experiment only; tokenization reality already pinned by a test. |
| **Composed ADDRESS/surrogate as trained behavior (rung 3)** | **UNBUILT** | Currently deterministic in `pipeline/`, not trained. |
| **Multi-seed / real-text benchmark / multi-teacher** | **UNBUILT** | Confidence-strengthening; see `next-steps-testing.md` Tiers 1 & 3. |

---

## Documentation staleness fixed in this pass

- **`docs/STATUS.md`** — its entire "In flight" block was stale: every item had already merged to `main`
  (PRs #5, #11, #16, #20, #22, #27, #36, #37). Rolled into "Done" and the live view corrected.
- **`docs/report-nontechnical.md`** — dated 2026-07-09, it predated the canonical gpt551 run and
  presented the *authored* tune as "the version you'd actually ship," with no gpt551 and no frontier
  comparison. Updated to make gpt551 the canonical line and add the frontier context.

---

## Day-by-day checkpoint status (for reference)

| Day | Checkpoint | Status |
|---|---|---|
| 1 | Base runs; behavior/tag locked | **MET** |
| 2 | Eval + checks exist before training; full loop on 50 junk | **MET** |
| 3 (midweek gate) | Base-vs-tuned numbers on quarantined hard cases | **MET** |
| 4 | One failure mode fixed in data, config unchanged | **MET** |
| 5 | Ship-ready model + running demo; tuned beats base; DPO delta | **PARTIAL** — tuned-beats-base done; no demo, no DPO |
| 6 | Robustness-under-attack numbers; full results table | **MET** (gap: no `error-analysis-final.md`) |
| 7 | Full submission package ready | **NOT MET** — HF push, demo, video, BrainLift verdict outstanding (Steps 1–5 above) |
