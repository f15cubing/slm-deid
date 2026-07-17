# Day 3 — Wed Jul 8: v1 dataset + first real QLoRA run + first base-vs-tuned numbers

**Goal.** A real v1 dataset trains a QLoRA adapter, and the **first base-vs-tuned numbers on the
quarantined hard cases are on the board** (the midweek gate).

> **Ceiling reminders:** do not start training until Day 2's eval is green (it is). No eval leakage.
> Fix data, not hyperparameters (that's tomorrow's rule too).

## Specs (definition of done)
- **S3.1 Dataset size/mix.** `data/splits/train.jsonl` has **800–2,000** examples, weighted toward the
  hard ambiguous NAME categories, mixed as: distilled ambiguous passages (bulk) + Presidio-faker
  pattern-type **negatives** + entity-swap augmentation + a **small real slice** (CRAPII/TSCC). The
  mix ratios are recorded in `configs/datagen.yaml` and echoed into a `dataset_card` stub.
- **S3.2 Quality-gated.** 100% of `train.jsonl` passes `quality_gate.py` (integrity + tag
  well-formedness + schema) and the second-pass teacher verification; dropped counts are logged.
- **S3.3 No leakage.** `tests/test_no_eval_leakage.py` passes against the final train split.
- **S3.4 Training run.** `python -m src.train.qlora --config configs/train.yaml` with **r=32, α=32,
  lr=2e-4, seq_len=2048, 2–3 epochs, completion-only masking, non-thinking**; produces an adapter
  under `outputs/sft-v1/` and a training log (loss curve) without OOM on a 24GB card. *(Mac/MPS:
  `--config configs/train.mps.yaml` → adapter under `outputs/sft-v1-mps/`; same hyperparameters.)*
- **S3.5 First numbers.** `src/eval/run.py` produces a **base vs. SFT-v1** results table on
  `eval/hardcases` across all metrics (entity P/R/F5, leakage rate, over-tag rate, integrity,
  consistency) with per-category rows and **bootstrap CIs**; saved to `data/eval_reports/` and copied
  into `docs/results.md`.

## Tasks
- [ ] Finalize `configs/datagen.yaml` mix ratios; generate the full v1 set through the Day-2 pipeline.
- [ ] Run the quality gate; log kept/dropped counts and the drop reasons — S3.2.
- [ ] Build the train/val split (leakage-safe: a paraphrase group / faker template stays in one fold).
- [ ] Run `test_no_eval_leakage.py` on the final split — S3.3.
- [ ] Fill `configs/train.yaml` (Colab) / use `configs/train.mps.yaml` (Mac) — same hyperparameters (S3.4); run the first real train.
- [ ] Run `eval/run.py --model base` and `--model outputs/sft-v1` on the hard cases — S3.5.
- [ ] Write `docs/results.md` with the base-vs-SFT-v1 table + a 3-sentence read of the delta.

## Deliverables
- `data/splits/train.jsonl` (+ val), quality-gate log, `configs/{datagen,train}.yaml`.
- `outputs/sft-v1/` adapter + training log.
- `docs/results.md` with the first base-vs-tuned table.

## Checkpoint (hard gate — MIDWEEK GATE) — ✅ MET (2026-07-07)
Base-vs-tuned numbers exist on the quarantined hard cases. If tuned clearly beats prompting on the
ambiguous cases, the core thesis (SPOV 7) is validated early. **If it doesn't**, treat it as a real
signal: investigate **data quality first** (per the falsifiable-bet rule) before anything else.

**Result:** tuned beats base on the recall/F5 axis (F5 0.19→0.61, recall 0.19→0.63, leakage 0.41→0.20)
at a precision cost (over-tag 0.10→0.37) — SPOV-7 validated on recall/leakage; over-tagging is the Day-4
data-fix target (concentrated in `person_vs_place`/`person_vs_common`, which v1 under-covered). Full
table + read: [`docs/results.md`](../../results.md). Trained + evaluated locally on Apple MPS.
_(Note: S3.5 bootstrap CIs added in PR `agent/eval-ci-reporting` — report tables are now generated from the saved JSON reports with 95% percentile CIs (`python -m src.eval.report`), which also corrected a hand-transcribed `easy` row in `docs/results.md`. v1 was ~300 teacher examples time-boxed under 2h, not the full 800–2,000.)_
