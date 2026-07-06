# Day 5 — Fri Jul 10: Ship-ready core + Stretch rung 1 (DPO)

**Goal.** Confirm the win (tuned beats base on spec adherence + robustness), freeze a v1 model, stand
up the demo, and measure whether DPO sharpens spec adherence **beyond SFT alone**.

## Specs (definition of done)
- **S5.1 Win confirmed.** `docs/results.md` shows SFT (best of v1/v2) beating base on **spec
  adherence** (leakage rate ↓, over-tag rate ↓, integrity ok) **and robustness** (consistency across
  paraphrases ↑), with CIs. This is the assignment's definition of a win; state it explicitly.
- **S5.2 Frozen model.** The chosen adapter is copied to `outputs/sft-final/` and its exact config +
  dataset hash recorded in a `MODEL_CARD.md` stub (repro info).
- **S5.3 Demo.** `python -m src.demo` takes a passage (CLI arg or stdin) and prints the tagged output;
  ships with a canned ambiguous-passage showcase (e.g. "Newton was frustrated" [tag] vs "the Newton
  method" [don't]; "I visited Chelsea" [don't] vs "Chelsea helped me" [tag]) and prints base-vs-tuned
  side by side.
- **S5.4 DPO (stretch).** Build preference pairs (chosen = on-spec tagging; rejected = off-spec:
  over-tagged / missed / wrong boundary) into `data/splits/dpo.jsonl`; run
  `python -m src.train.dpo --config configs/dpo.yaml` on top of `sft-final`; produce `outputs/dpo-v1/`.
- **S5.5 DPO delta.** Add a base | SFT | DPO column to `docs/results.md`; report whether DPO improves
  spec adherence beyond SFT — honestly, even if the delta is ~0.

## Tasks
- [ ] Select best SFT adapter; write the explicit win statement + freeze to `outputs/sft-final/` — S5.1–S5.2.
- [ ] Write `MODEL_CARD.md` stub (base model, LoRA config, dataset hash, eval numbers) — S5.2.
- [ ] Implement `src/demo.py` with the canned showcase + base-vs-tuned side-by-side — S5.3.
- [ ] Generate DPO preference pairs from graded outputs; validate schema — S5.4.
- [ ] Add `configs/dpo.yaml`; run DPO on `sft-final`; produce `outputs/dpo-v1/` — S5.4.
- [ ] Re-eval DPO on hard cases; append column to `docs/results.md`; write the DPO-delta read — S5.5.

## Deliverables
- `outputs/sft-final/` + `MODEL_CARD.md`; working `src/demo.py`; `outputs/dpo-v1/`;
  `docs/results.md` with base | SFT | DPO.

## Checkpoint (hard gate)
Ship-ready model + running demo; tuned beats base on spec adherence & robustness; DPO delta measured.
