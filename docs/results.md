# Results — Base vs. Tuned

_Base-vs-tuned numbers on the quarantined 51 hard cases. Tables and 95% bootstrap CIs (spec S3.5) are
**generated from the saved reports** (`python -m src.eval.report`), not hand-transcribed._

- **[v3 (data rebalance)](#v3-data-rebalance--current) — current.** Rebalanced + scaled data (927/102),
  bf16. Fixes v2's recall/consistency regression: recall 0.44→0.93, consistency 0.13→0.75, leakage
  →0.04, with over-tag/integrity held. Model card: [`docs/model-card-v3.md`](model-card-v3.md).
- **[v2 (Day 4)](#v2-day-4--prior) — prior.** CRAPII-augmented data (242/26). Fixed over-tagging +
  integrity, but recall/consistency regressed. Model card: [`docs/model-card-v2.md`](model-card-v2.md).
- **[v1 (Day 3)](#v1-day-3--prior) — prior.** First numbers; high recall but 0.37 over-tag + integrity
  regression.

---

# v3 (data rebalance) — current

_`sft-v3-mps`: same LoRA recipe (r=32, α=32, lr=2e-4, seq 2048, 3 epochs, completion-only) over
`Qwen/Qwen3-1.7B`, **bfloat16** on Apple MPS, trained on the **v3 dataset (927 train / 102 val)** and
evaluated on the **quarantined 51 hard cases**. v3 targets v2's recall/consistency regression **in the
data**: the `person_vs_common` withhold bias (v2: 18 person / 38 withhold) is rebalanced to ~50/50, the
eval-disjoint vocab bank ~doubled (53→110: 43 common / 32 places / 35 eponyms) for generalization, and the set
scaled ~3.8×. Hyperparameters unchanged from v1/v2. `eval_leak = 0` (all three guards + independent
scan; positive control fires 50/51). Data authored in-session (teacher API unavailable) — see the model
card caveat. Reports: `outputs/eval_reports/{base,tuned}-20260708-202*.json`._

<!-- Regenerate (offline, no model / no network) with:
python -m src.eval.report \
  base=outputs/eval_reports/base-20260708-202439.json \
  tuned=outputs/eval_reports/tuned-20260708-202511.json -->

## Overall

| model | n | precision | recall | F5 | leakage_rate | over_tag_rate | integrity_violation_rate | pass_rate | consistency |
|---|---|---|---|---|---|---|---|---|---|
| base | 51 | 0.500 [0.18, 0.83] | 0.185 [0.04, 0.35] | 0.190 [0.04, 0.35] | 0.412 [0.27, 0.55] | 0.098 [0.02, 0.20] | 0.039 [0.00, 0.10] | 0.549 [0.41, 0.69] | 0.250 |
| tuned | 51 | 0.781 [0.63, 0.93] | 0.926 [0.81, 1.00] | 0.919 [0.81, 1.00] | 0.039 [0.00, 0.10] | 0.137 [0.04, 0.24] | 0.000 [0.00, 0.00] | 0.863 [0.76, 0.96] | 0.750 |
| Δ (tuned−base) |  | +0.281 | +0.741 | +0.730 | -0.373 | +0.039 | -0.039 | +0.314 | +0.500 |

## Per-category (base → tuned within each category)

| category | n | model | recall | over_tag_rate | integrity_violation_rate | pass_rate | consistency |
|---|---|---|---|---|---|---|---|
| person_vs_common | 16 | base | 0.125 [0.00, 0.43] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.562 [0.31, 0.81] | 0.125 |
| person_vs_common | 16 | tuned | 1.000 [1.00, 1.00] | 0.125 [0.00, 0.31] | 0.000 [0.00, 0.00] | 0.875 [0.69, 1.00] | 0.750 |
| person_vs_place | 10 | base | 0.400 [0.00, 0.86] | 0.100 [0.00, 0.30] | 0.000 [0.00, 0.00] | 0.600 [0.30, 0.90] | 0.250 |
| person_vs_place | 10 | tuned | 1.000 [1.00, 1.00] | 0.100 [0.00, 0.30] | 0.000 [0.00, 0.00] | 0.900 [0.70, 1.00] | 0.750 |
| person_vs_eponym | 8 | base | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.750 [0.38, 1.00] | 0.333 |
| person_vs_eponym | 8 | tuned | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | 1.000 |
| first_name_only | 3 | base | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | – |
| first_name_only | 3 | tuned | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | – |
| possessive | 3 | base | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 1.000 |
| possessive | 3 | tuned | 1.000 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.667 [0.00, 1.00] | 0.000 |
| third_party | 3 | base | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | – |
| third_party | 3 | tuned | 0.667 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.667 [0.00, 1.00] | – |
| negative_trap | 5 | base | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | – |
| negative_trap | 5 | tuned | 0.000 [0.00, 0.00] | 0.200 [0.00, 0.60] | 0.000 [0.00, 0.00] | 0.800 [0.40, 1.00] | – |
| easy | 3 | base | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | – |
| easy | 3 | tuned | 0.750 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.667 [0.00, 1.00] | – |

## The read (honest: the regression is fixed, with residual over-tag)

1. **The v2 recall/consistency regression is fixed — decisively, with separated CIs.** Recall
   0.444 → **0.926** (CI `[0.81, 1.00]` vs base `[0.04, 0.35]`), F5 → 0.919, and **consistency
   0.125 → 0.750** (6×). Leakage collapsed to **0.039**. This validates the diagnosis: v2 under-tagged
   because its `person_vs_common` data was 2:1 withhold-skewed; rebalancing to 50/50 + tripling the
   surface bank fixed it. `person_vs_common` recall went **0.125 → 1.000**, `person_vs_place` and
   `person_vs_eponym` likewise to 1.000.
2. **No trade-off this time — over-tag and integrity held.** over_tag 0.137 (unchanged vs v2, CI
   overlaps base) and **integrity 0.000** (perfect). So recall/consistency recovered *without* giving
   back the v2 gains — pass-rate rose to **0.863**.
3. **Residual failures are the eponymous-possessive + a stray pronoun tag.** The misses: "Newton's
   laws" (eponymous possessive over-tagged — `possessive` remains the weak spot, and its training
   contrast is still hard to author cleanly), "visited Chelsea" (place over-tagged), and one case where
   the model tagged the pronoun "She". `negative_trap` over_tag ticked to 0.200 (the pronoun case). These
   are the next targets, not regressions.

## Caveats (v3)

- **Small n, single seed.** n=51; per-category n=3 cells (possessive/third_party/easy) have CIs spanning
  most of [0,1]. The overall recall/F5/leakage/consistency gaps are the robust ones.
- **In-session authored data.** The teacher API was unavailable, so v3 passages were authored from
  templates (`src/datagen/author.py`) and routed through the same gate + leakage guards. There is no
  INDEPENDENT verifier pass (label trust rests on the deterministic gates). Template-authored text is
  less linguistically varied than a frontier teacher's — a real limitation to note when reading the size
  of the win. See the model card.
- **bf16 lineage** (same as v2); base reproduces the Day-3 fp16 base, so base-vs-tuned is fair.
- Reports: `outputs/eval_reports/{base,tuned}-20260708-202*.json`. Model: `outputs/sft-v3-mps/`.

---

# v2 (Day 4) — prior

_`sft-v2-mps`: LoRA (r=32, α=32, lr=2e-4, seq 2048, 3 epochs, completion-only) over `Qwen/Qwen3-1.7B`,
trained on the **v2 dataset (242 train / 26 val, CRAPII-augmented)** in **bfloat16** on Apple MPS, and
evaluated on the **quarantined `eval/hardcases` set (51 scenarios)**. Both models run through the same
`hf`/MPS backend. **F5** (β=5, recall-weighted) is the headline. `eval_leak = 0` (verified: all three
guards clean + independent raw scan). LoRA hyperparameters are byte-identical to v1 — only the data
changed (plus fp16→bf16 for numerical stability; see the model card). Reports:
`outputs/eval_reports/{base,tuned}-20260708-17*.json`._

<!-- Regenerate (offline, no model / no network) with:
python -m src.eval.report \
  base=outputs/eval_reports/base-20260708-173951.json \
  tuned=outputs/eval_reports/tuned-20260708-174019.json -->

## Overall

| model | n | precision | recall | F5 | leakage_rate | over_tag_rate | integrity_violation_rate | pass_rate | consistency |
|---|---|---|---|---|---|---|---|---|---|
| base | 51 | 0.500 [0.18, 0.83] | 0.185 [0.04, 0.35] | 0.190 [0.04, 0.35] | 0.412 [0.27, 0.55] | 0.098 [0.02, 0.20] | 0.039 [0.00, 0.10] | 0.549 [0.41, 0.69] | 0.250 |
| tuned | 51 | 0.632 [0.40, 0.85] | 0.444 [0.25, 0.64] | 0.450 [0.26, 0.64] | 0.275 [0.16, 0.39] | 0.137 [0.06, 0.24] | 0.020 [0.00, 0.06] | 0.627 [0.49, 0.76] | 0.125 |
| Δ (tuned−base) |  | +0.132 | +0.259 | +0.260 | -0.137 | +0.039 | -0.020 | +0.078 | -0.125 |

## Per-category (base → tuned within each category)

| category | n | model | recall | over_tag_rate | integrity_violation_rate | pass_rate | consistency |
|---|---|---|---|---|---|---|---|
| person_vs_common | 16 | base | 0.125 [0.00, 0.43] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.562 [0.31, 0.81] | 0.125 |
| person_vs_common | 16 | tuned | 0.125 [0.00, 0.43] | 0.062 [0.00, 0.19] | 0.000 [0.00, 0.00] | 0.500 [0.25, 0.75] | 0.000 |
| person_vs_place | 10 | base | 0.400 [0.00, 0.86] | 0.100 [0.00, 0.30] | 0.000 [0.00, 0.00] | 0.600 [0.30, 0.90] | 0.250 |
| person_vs_place | 10 | tuned | 0.400 [0.00, 0.83] | 0.200 [0.00, 0.50] | 0.000 [0.00, 0.00] | 0.500 [0.20, 0.80] | 0.250 |
| person_vs_eponym | 8 | base | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.750 [0.38, 1.00] | 0.333 |
| person_vs_eponym | 8 | tuned | 0.667 [0.00, 1.00] | 0.125 [0.00, 0.38] | 0.125 [0.00, 0.38] | 0.750 [0.38, 1.00] | 0.333 |
| first_name_only | 3 | base | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | – |
| first_name_only | 3 | tuned | 1.000 [1.00, 1.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | – |
| possessive | 3 | base | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 1.000 |
| possessive | 3 | tuned | 1.000 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.667 [0.00, 1.00] | 0.000 |
| third_party | 3 | base | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | – |
| third_party | 3 | tuned | 0.667 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.667 [0.00, 1.00] | – |
| negative_trap | 5 | base | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | – |
| negative_trap | 5 | tuned | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | – |
| easy | 3 | base | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | – |
| easy | 3 | tuned | 0.250 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | – |

## The read (honest: what improved, what regressed)

1. **The Day-4 goal is met — v1's two regressions are fixed.** Over-tagging fell **0.37 → 0.137** and
   integrity violations fell **0.118 → 0.020** (now below base's 0.039). Overall **pass-rate rose to
   0.627** (+0.078 vs base), where v1's pass-rate was flat. The fix came entirely from data (targeted
   person_vs_place / minimal pairs / CRAPII real text) — LoRA hyperparameters were untouched.
2. **Tuned still beats prompting on the safety axis.** Recall 0.185 → 0.444 and F5 0.190 → 0.450 with
   CIs that separate from base (recall base `[0.04, 0.35]` vs tuned `[0.25, 0.64]`); leakage 0.412 →
   0.275. `first_name_only` is the clearest category win (recall 0 → 1.0, pass 0 → 1.0).
3. **Regression to watch — consistency dropped 0.250 → 0.125.** The tuned model is *less* stable across
   paraphrases of the same name/non-name than base. Reliability across rewordings is the point of the
   behavior, so this is the headline weakness of this iteration and the next thing to chase.
4. **`person_vs_common` recall is flat at 0.125** — the largest eval category (16), and the model still
   barely tags common-word given-names. The next data-coverage target.
5. **Recall/F5 are lower than v1's tuned run** (0.44 vs 0.63 / 0.45 vs 0.61) — but v1's higher recall
   came *with* 0.37 over-tagging and 3× integrity violations. v2 trades a slice of recall for a model
   that actually holds the spec (higher pass-rate, integrity below base). This is the intended trade.

## Caveats (v2)

- **Small n, single seed.** Overall n=51; several per-category cells are n=3 with CIs spanning most of
  [0,1]. Only the overall recall/F5/leakage gaps clearly separate from base.
- **bf16 re-baseline.** v2 is a bf16 lineage (v1 was fp16) because fp16 NaN-diverged on the long CRAPII
  passages. The bf16 base reproduces the Day-3 fp16 base almost exactly, so base-vs-tuned is still fair;
  cross-version tuned deltas (v1→v2) should be read as directional. Details in the model card.
- Reports: `outputs/eval_reports/{base,tuned}-20260708-17*.json` (gitignored). Model: `outputs/sft-v2-mps/`.

---

# v1 (Day 3) — prior

_First base-vs-tuned numbers on the quarantined hard cases (Day 3 midweek gate). Kept as the historical
record and the baseline v2 improves on. Tables and 95% bootstrap CIs (spec S3.5) are **generated from
the saved reports**, not hand-transcribed._

**Setup.** Base = prompted `Qwen/Qwen3-1.7B` (non-thinking). Tuned = base + LoRA (r=32, α=32, lr=2e-4,
seq 2048, 3 epochs, completion-only), trained on the **v1 dataset (146 train / 16 val)** and evaluated
on the **quarantined `eval/hardcases` set (51 scenarios)**. Both models run through the **same `hf`/MPS
backend** (fair comparison). Metrics are entity-level; **F5** (β=5, recall-weighted) is the headline.
Trained + evaluated locally on Apple-Silicon MPS. `eval_leak = 0`.

Each metric shows its **point value** (copied verbatim from the saved report) followed by a **95%
percentile bootstrap CI** `[low, high]` over the per-item results (1,000 resamples, seed `20260707`).
`consistency` is a paraphrase-group statistic, so it is reported as a point value (no per-item CI).

<!-- Regenerate (offline, no model / no network) with:
python -m src.eval.report \
  base=outputs/eval_reports/base-20260707-204355.json \
  tuned=outputs/eval_reports/tuned-20260707-204436.json
The base/tuned reports predate the per-item tp/fp/fn field, so report.py recomputes those counts
offline via behavioral_checks.check() against eval/hardcases (join by id). Every number below is
derived from those two reports. -->

## Overall

| model | n | precision | recall | F5 | leakage_rate | over_tag_rate | integrity_violation_rate | pass_rate | consistency |
|---|---|---|---|---|---|---|---|---|---|
| base | 51 | 0.500 [0.18, 0.83] | 0.185 [0.04, 0.35] | 0.190 [0.04, 0.35] | 0.412 [0.27, 0.55] | 0.098 [0.02, 0.20] | 0.039 [0.00, 0.10] | 0.549 [0.41, 0.69] | 0.250 |
| tuned | 51 | 0.370 [0.20, 0.60] | 0.630 [0.43, 0.81] | 0.613 [0.42, 0.80] | 0.196 [0.10, 0.31] | 0.372 [0.25, 0.51] | 0.118 [0.04, 0.22] | 0.529 [0.39, 0.67] | 0.312 |
| Δ (tuned−base) |  | -0.130 | +0.444 | +0.423 | -0.216 | +0.274 | +0.078 | -0.020 | +0.062 |

## Per-category (base → tuned within each category)

| category | n | model | recall | over_tag_rate | integrity_violation_rate | pass_rate | consistency |
|---|---|---|---|---|---|---|---|
| person_vs_common | 16 | base | 0.125 [0.00, 0.43] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.562 [0.31, 0.81] | 0.125 |
| person_vs_common | 16 | tuned | 0.375 [0.00, 0.75] | 0.375 [0.12, 0.62] | 0.250 [0.06, 0.50] | 0.438 [0.19, 0.69] | 0.375 |
| person_vs_place | 10 | base | 0.400 [0.00, 0.86] | 0.100 [0.00, 0.30] | 0.000 [0.00, 0.00] | 0.600 [0.30, 0.90] | 0.250 |
| person_vs_place | 10 | tuned | 0.800 [0.40, 1.00] | 0.500 [0.20, 0.80] | 0.100 [0.00, 0.30] | 0.400 [0.10, 0.70] | 0.000 |
| person_vs_eponym | 8 | base | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.750 [0.38, 1.00] | 0.333 |
| person_vs_eponym | 8 | tuned | 0.667 [0.00, 1.00] | 0.250 [0.00, 0.62] | 0.000 [0.00, 0.00] | 0.625 [0.25, 0.88] | 0.667 |
| first_name_only | 3 | base | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | – |
| first_name_only | 3 | tuned | 0.667 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.667 [0.00, 1.00] | – |
| possessive | 3 | base | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 1.000 |
| possessive | 3 | tuned | 1.000 [0.00, 1.00] | 0.667 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.000 |
| third_party | 3 | base | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | – |
| third_party | 3 | tuned | 0.667 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.667 [0.00, 1.00] | – |
| negative_trap | 5 | base | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 1.000 [1.00, 1.00] | – |
| negative_trap | 5 | tuned | 0.000 [0.00, 0.00] | 0.200 [0.00, 0.60] | 0.000 [0.00, 0.00] | 0.800 [0.40, 1.00] | – |
| easy | 3 | base | 0.000 [0.00, 0.00] | 0.333 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | – |
| easy | 3 | tuned | 0.750 [0.00, 1.00] | 0.333 [0.00, 1.00] | 0.000 [0.00, 0.00] | 0.667 [0.00, 1.00] | – |

> **Doc-bug fix (why this table is now generated).** The previous hand-typed table listed base
> `easy` **pass 0.667** while base `easy` **recall was 0.0** — impossible, since every `easy` item
> contains a name, so zero recall means every item leaked and none can pass. The saved report always
> had it right (base `easy` recall 0.000, pass **0.000**); the doc was wrong because it was transcribed
> by hand. Tables are now emitted directly from the reports, and a regression test + a CLI guard reject
> any report where an all-named category shows `recall == 0` with `pass_rate > 0`.

## The read (honest: win + cost)

1. **Fine-tuning clearly beats prompting on the primary (safety) axis, and the CIs back it up.** Recall
   rose 0.185 → 0.630 and F5 0.190 → 0.613 with **non-overlapping** 95% CIs (recall base `[0.04, 0.35]`
   vs tuned `[0.43, 0.81]`; F5 base `[0.04, 0.35]` vs tuned `[0.42, 0.80]`), and leakage roughly halved
   (0.412 → 0.196, base `[0.27, 0.55]` vs tuned `[0.10, 0.31]`). The prompted base massively under-tags;
   the fine-tune fixes that. This validates the SPOV-7 bet on the recall/leakage axis.
2. **Cost #1 — precision / over-tagging (also CI-robust).** Over-tagging nearly quadrupled
   (0.098 → 0.372) with **non-overlapping** CIs (base `[0.02, 0.20]` vs tuned `[0.25, 0.51]`), precision
   fell 0.500 → 0.370, and overall **pass-rate is flat** (0.549 → 0.529; CIs `[0.41, 0.69]` vs
   `[0.39, 0.67]` overlap heavily). The tuned model learned to tag aggressively but not yet to *withhold*
   on identically-spelled non-persons.
3. **Cost #2 — integrity (a HARD CEILING) got worse.** `integrity_violation_rate` (output-minus-tags ==
   input) roughly **tripled**, 0.039 → 0.118. Altering the passage text is a hard ceiling per
   `AGENTS.md`, so this is a **safety regression**, not a mere quality dip — even though the CIs still
   overlap (base `[0.00, 0.10]` vs tuned `[0.04, 0.22]`), so the *size* is uncertain on n=51. It
   concentrates in `person_vs_common` (0.000 → 0.250) where the model emits malformed/garbled tag runs.
   **This must be driven back down** (targeted data + tighter decoding), and it must not be papered over.
4. **Consistency is still poor.** Paraphrase-group consistency barely moved, 0.250 → 0.312 — i.e. the
   model still flips its verdict across rewordings of the *same* name/non-name roughly two-thirds of the
   time. Reliability across paraphrases is the point of the behavior, so this remains a headline weakness,
   not a solved problem.
5. **The failures are a data-coverage gap, not hyperparameters.** Over-tagging concentrates exactly where
   v1 training data was thin or absent — **person_vs_place (0 train examples → over-tag 0.100 → 0.500)**,
   person_vs_common (0.000 → 0.375), possessive (0.333 → 0.667). The teacher's 54% quality-gate drop rate
   skewed v1 away from these discrimination cases. Fix in data (Day 4), per the ceiling.

## Day-4 target (fix in data, not hyperparameters)

Generate targeted **person_vs_place / person_vs_common / possessive / eponym-negative** examples that
teach the model to *not* tag non-person uses (cutting over-tagging), plus clean well-formed-tag examples
to cut the integrity regression, then retrain and re-measure over-tag / integrity / consistency on those
categories. Do **not** touch lr/r/epochs to mask this.

## Caveats

- **Directional; small n.** Overall n=51, and several per-category cells are **n=3**, so their CIs span
  most of `[0, 1]` (e.g. `possessive` tuned recall `1.000 [0.00, 1.00]`). Treat per-category numbers as
  suggestive; only the overall recall/F5/leakage/over-tag gaps have CIs that clearly separate.
- **Reproducible & offline.** Tables regenerate deterministically from the two saved reports via
  `python -m src.eval.report` (seeded bootstrap; no model, no network). Point values are the reports'
  own numbers; the harness compute was never in question — the earlier doc error was hand-transcription.
- **Mac baseline.** The Mac path is plain LoRA on an fp16 base (not 4-bit QLoRA), so these are the Mac
  baseline; a Colab 4-bit run would re-baseline separately.
- Reports: `outputs/eval_reports/{base,tuned}-*.json` (gitignored); dataset: `data/splits/`.
