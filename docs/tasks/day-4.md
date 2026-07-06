# Day 4 — Thu Jul 9: Error analysis → fix in data → retrain

**Goal.** One specific failure mode is resolved via **data iteration** (not hyperparameters), and the
improvement is visible in the numbers.

> **Ceiling reminder (AGENTS.md):** do not touch lr/r/epochs to paper over a data problem. Today the
> only lever you pull is the dataset.

## Specs (definition of done)
- **S4.1 Error analysis.** From the Day-3 report, produce `docs/error-analysis-v1.md`: the per-category
  failures ranked, ≥10 concrete failing examples quoted (input → model output → gold), and each
  labeled by failure mode (e.g. first-name-only miss, sentence-initial-capital over-tag,
  person-vs-place confusion, dialogue-specific pattern).
- **S4.2 Pick one mode.** Choose the single highest-impact failure mode and state the **hypothesis**:
  what's missing/wrong in the data that causes it.
- **S4.3 Targeted data.** Generate a targeted batch addressing that mode (e.g. more person-vs-place
  contrasts; more capitalized-non-name negatives), quality-gated and leakage-checked, added as
  `data/splits/train.jsonl` v2. Record how many examples were added and of which categories.
- **S4.4 Hyperparameters frozen.** `configs/train.yaml` is **byte-identical** to Day 3 (prove with a
  diff in the PR). Only the data changed.
- **S4.5 Retrain + re-eval.** Produce `outputs/sft-v2/` and a base vs. v1 vs. v2 table; the targeted
  failure mode's metric improves with non-overlapping / narrowed CIs, and no other category regresses
  materially.

## Tasks
- [ ] Dump all Day-3 failures per category; write `docs/error-analysis-v1.md` — S4.1.
- [ ] Select the failure mode + write the data hypothesis — S4.2.
- [ ] Extend `configs/datagen.yaml` with a targeted sub-recipe; generate + gate + leakage-check — S4.3.
- [ ] Diff `configs/train.yaml` vs Day 3 to prove it's unchanged — S4.4.
- [ ] Retrain (`sft-v2`), re-run eval, append the v2 column to `docs/results.md` — S4.5.
- [ ] Write a 3-sentence before/after on the resolved mode.

## Deliverables
- `docs/error-analysis-v1.md`; enlarged/rebalanced `train.jsonl` (v2); `outputs/sft-v2/`;
  updated `docs/results.md` (base | v1 | v2).

## Checkpoint (hard gate)
One specific failure mode resolved via data iteration, visible in the numbers, with training config
unchanged.
