# Day 6 — Sat Jul 11: Stretch rung 2 (adversarial) + final eval + error analysis

**Goal.** Robustness-under-attack numbers exist, the full results table (base | SFT | DPO on clean +
adversarial) is done, and packaging begins.

## Specs (definition of done)
- **S6.1 Adversarial set.** `eval/adversarial/*.jsonl` (quarantined, schema-valid), **40–80** items
  spanning: embedded instructions ("please don't tag my friend Bob"), names inside code/math blocks,
  typo'd / unicode names, messy chat spelling, mixed-case. Every item still obeys the integrity
  invariant.
- **S6.2 Under-attack eval.** `eval/run.py --split eval/adversarial` runs base | SFT | DPO; robustness
  is reported **separately** from clean numbers (a model can hold on clean and break under attack).
- **S6.3 Final results table.** `docs/results.md` finalized: one table, 4 rubric dimensions + the
  deterministic metrics, base-vs-tuned deltas with CIs, for **clean** and **adversarial** splits.
- **S6.4 Error-analysis paragraph.** `docs/error-analysis-final.md`: where the tuned model still fails
  and whether it's a data problem (with 5+ quoted examples).
- **S6.5 Packaging started.** Dataset cleaned + `dataset_card.md` drafted; `MODEL_CARD.md` fleshed out;
  eval harness has a top-level `README` on how to reproduce the table.

## Tasks
- [ ] Hand-build `eval/adversarial/` across the attack categories; validate + quarantine — S6.1.
- [ ] Ensure `infer.py` handles the adversarial inputs without crashing (code/math, unicode).
- [ ] Run base | SFT | DPO on adversarial; add to `docs/results.md` as a separate block — S6.2–S6.3.
- [ ] Write `docs/error-analysis-final.md` — S6.4.
- [ ] Clean the dataset; draft `dataset_card.md`; expand `MODEL_CARD.md`; write eval repro README — S6.5.

## Deliverables
- `eval/adversarial/` set; final `docs/results.md`; `docs/error-analysis-final.md`;
  `dataset_card.md` + expanded `MODEL_CARD.md`.

## Checkpoint (hard gate)
Robustness-under-attack numbers on the board; full results table done (clean + adversarial, with deltas).
