# Day 7 — Sun Jul 12: Ship & defend

**Goal.** Full submission package ready: dataset published, model on HF Hub with a running demo,
results table + eval harness public, BrainLift verdict written, demo video recorded.

## Specs (definition of done)
- **S7.1 Dataset published.** The dataset (train + quarantined eval, clearly separated) is published
  with a `dataset_card.md` documenting: schema, the ambiguous-case category breakdown + counts, the
  synthetic/real mix, the quality-gate, and the **quarantine** guarantee (eval never in train).
- **S7.2 Model on HF Hub.** `sft-final` (and DPO if it won) pushed with a complete `MODEL_CARD.md`:
  base model, LoRA config, training data hash, eval numbers vs. base, intended use + limitations
  (direct-identifiers-only; quasi-identifiers out of scope per SPOV 5).
- **S7.3 Running inference demo.** A hosted or one-command demo reproduces the ambiguous-name behavior.
- **S7.4 Eval harness + results table public.** The 4-dimension base-vs-tuned table (= prompt vs.
  fine-tune) with deltas is in the repo `README`/`docs/results.md`, reproducible via the eval README.
- **S7.5 BrainLift verdict.** `docs/brainlift.md` (or a `brainlift-verdict.md`) updated with the
  empirical resolution of SPOV 7's falsifiable bet: **did data→behavior hold? did fine-tune beat
  prompting on the hard cases?** — reported honestly, win or lose, with the numbers.
- **S7.6 Demo video (3–5 min).** Feeds ambiguous passages and shows the **prompted base
  wobbling/over-tagging while the tuned model holds**; links added to the README.

## Tasks
- [ ] Finalize + publish the dataset with its card; verify the eval quarantine is documented — S7.1.
- [ ] `huggingface-cli` push the adapter/merged model + `MODEL_CARD.md` — S7.2.
- [ ] Stand up the running demo (Space or one-command script) — S7.3.
- [ ] Confirm the results table + eval repro instructions are public — S7.4.
- [ ] Write the BrainLift empirical verdict tied to the numbers — S7.5.
- [ ] Record + link the 3–5 min demo video — S7.6.
- [ ] Tick the final-submission checklist in `docs/plan.md`.

## Deliverables
- Published dataset + card; HF model + card; running demo; public results table + eval harness;
  BrainLift verdict; demo video.

## Checkpoint (hard gate)
Full submission package ready — every box in `docs/plan.md` "Final submission package" checked.
