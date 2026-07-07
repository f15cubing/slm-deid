# Day 2 — Tue Jul 7: Behavior spec, eval harness, data-gen pipeline, smoke test

**Goal.** The full loop (generate → QLoRA train → eval) runs end-to-end on 50 junk examples, and the
**eval exists before any real training**. This is the hard rule — nothing on Day 3 starts until the
eval and behavioral checks are committed and green.

> **Ceiling reminders (AGENTS.md):** eval before train; the `eval/hardcases/` set is quarantined and
> never touched by data-gen; integrity (`unwrap(output)==input`) is a reject condition.

## Specs (definition of done)

### Behavior spec
- **S2.1** The locked behavior spec (from `docs/plan.md` "THE GATE") is committed and referenced;
  PASS/FAIL defined in code as: PASS = zero leakage ∧ zero over-tag ∧ integrity holds.

### Eval harness — 3 required pieces
- **S2.2 Behavioral checks** (`src/eval/behavioral_checks.py`, pure functions): `leakage(gold, out)`,
  `over_tag(gold, out)`, `integrity(input, out)` per the shared contract. Covered by
  `tests/test_behavioral_checks.py` with ≥1 leakage case, ≥1 over-tag case, ≥1 integrity-violation case.
- **S2.3 Metrics** (`src/eval/metrics.py`): entity-level precision/recall/**F5**, leakage rate,
  over-tag rate, integrity-violation rate, and **consistency** across `paraphrase_group`. Unit-tested
  on a tiny hand-made fixture with known expected numbers.
- **S2.4 LLM-as-judge** (`src/eval/judge.py`): scores each output 0–2 on the 4 rubric dimensions
  (spec adherence / robustness / task quality / consistency); returns structured JSON; deterministic
  temp=0; disagreement with behavioral checks on PASS/FAIL is logged, not silently overridden.
- **S2.5 Base-vs-tuned scaffold** (`src/eval/run.py`): `--model base|<adapter>` runs the same
  `eval/hardcases` split through `src/infer.py` and emits a per-category results table + overall row
  to `data/eval_reports/<model>-<timestamp>.json` and a markdown table.

### Hard-cases eval set (the crux)
- **S2.6** `eval/hardcases/*.jsonl` holds **80–150** ambiguous scenarios, hand-built + drawn from real
  CRAPII/TSCC, spanning every category in the shared contract, each with 2–3 paraphrases sharing a
  `paraphrase_group`. All rows `quarantine=true` and schema-valid.
- **S2.7** `tests/test_no_eval_leakage.py`: fails if any `input` (normalized) in `eval/hardcases`
  also appears in any `data/` training split — the mechanical leakage guard.

### Data-gen pipeline
- **S2.8** `src/datagen/teacher.py` generates educational passages (essay + dialogue registers) that
  **deliberately contain ambiguous name/non-name tokens**, with correct inline tagging; a second-pass
  teacher call verifies each and disagreements are dropped.
- **S2.9** `src/datagen/negatives.py` emits pattern-type **negatives** via Presidio Sentence Faker
  (emails/phones/IDs present but **untagged**) so the model learns those aren't its job.
- **S2.10** `src/datagen/quality_gate.py` rejects any generated item failing integrity, tag
  well-formedness, or schema validation; covered by `tests/test_quality_gate.py`.

### Smoke test
- **S2.11** `python -m src.datagen.generate --config configs/datagen.yaml` (50 junk examples) →
  `python -m src.train.qlora --config configs/train.yaml` (1 epoch, tiny) →
  `python -m src.eval.run` completes with no errors and writes a report. Numbers can be garbage; the
  loop must run.

## Tasks
- [x] TDD `behavioral_checks.py` (S2.2) — `src/eval/behavioral_checks.py`, 7 tests.
- [x] TDD `metrics.py` (S2.3) — `src/eval/metrics.py` (P/R/F5, rates, consistency, per-cat, md table), 7 tests.
- [x] Implement `infer.py` (`Tagger`/`HFTagger`/`FunctionTagger`, lazy torch) — used by eval + demo.
- [x] Implement `judge.py` (S2.4) — reference-based LLM-judge, provider-agnostic, JSON parse,
      behavioral cross-check; 4 tests (mocked).
- [x] Implement `eval/run.py` base-vs-tuned scaffold + markdown table (S2.5) — pure core, 4 tests; thin CLI.
- [x] Build `eval/hardcases/` (S2.6) — 51 validated ambiguous scenarios across all categories +
      paraphrase groups (`scripts/build_hardcases.py`). *(Grow toward 80–150 + mine real CRAPII/TSCC when access lands.)*
- [x] Add `tests/test_no_eval_leakage.py` overlap guard (S2.7).
- [x] Implement `teacher.py`, `negatives.py`, `quality_gate.py` + `configs/datagen.yaml` (S2.8–S2.10)
      + `generate.py` orchestrator; 16 tests.
- [x] Implement `train/dataset.py` (chat template, non-thinking, completion-only) + `train/qlora.py`
      + `configs/train.yaml`; pure parts tested.
- [ ] **(GPU / Colab)** Run the 50-example smoke test end-to-end — `notebooks/day2_smoke.ipynb`
      (generate junk → QLoRA 1 epoch → base-vs-tuned eval). Fix wiring until green (S2.11).

## Deliverables
- Eval harness (3 pieces) + passing tests (67 total); `eval/hardcases/` quarantined set (51); data-gen
  pipeline; `notebooks/day2_smoke.ipynb`. Smoke-test report under `data/eval_reports/` after the Colab run.

## Checkpoint (hard gate)
Eval + behavioral checks exist and are green **before** any real training — **DONE** (67 tests).
Full loop end-to-end on 50 junk examples — **run `notebooks/day2_smoke.ipynb` on Colab to confirm**.
