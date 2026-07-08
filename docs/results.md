# Results — Base vs. Tuned (SFT v1)

_First base-vs-tuned numbers on the quarantined hard cases (Day 3 midweek gate)._

**Setup.** Base = prompted `Qwen/Qwen3-1.7B` (non-thinking). Tuned = base + LoRA (r=32, α=32, lr=2e-4,
seq 2048, 3 epochs, completion-only), trained on the **v1 dataset (146 train / 16 val)** and evaluated
on the **quarantined `eval/hardcases` set (51 scenarios)**. Both models run through the **same `hf`/MPS
backend** (fair comparison). Metrics are entity-level; **F5** (β=5, recall-weighted) is the headline.
Trained + evaluated locally on Apple-Silicon MPS. `eval_leak = 0`.

## Overall

| model | precision | recall | **F5** | leakage_rate | over_tag_rate | integrity_viol | pass_rate | consistency |
|---|---|---|---|---|---|---|---|---|
| base  | 0.500 | 0.185 | 0.190 | 0.412 | 0.098 | 0.039 | 0.549 | 0.250 |
| tuned | 0.370 | **0.630** | **0.613** | **0.196** | 0.373 | 0.118 | 0.529 | 0.313 |
| Δ | −0.130 | **+0.444** | **+0.423** | **−0.216** | +0.275 | +0.078 | −0.020 | +0.063 |

## Per-category (recall / over-tag / pass; base → tuned)

| category | n | recall b→t | over-tag b→t | pass b→t |
|---|---|---|---|---|
| person_vs_common | 16 | 0.125 → 0.375 | 0.000 → 0.375 | 0.563 → 0.438 |
| person_vs_place | 10 | 0.400 → 0.800 | 0.100 → 0.500 | 0.600 → 0.400 |
| person_vs_eponym | 8 | 0.333 → 0.667 | 0.000 → 0.250 | 0.750 → 0.625 |
| first_name_only | 3 | 0.000 → 0.667 | 0.333 → 0.333 | 0.000 → 0.667 |
| possessive | 3 | 0.000 → 1.000 | 0.333 → 0.667 | 0.333 → 0.333 |
| third_party | 3 | 0.333 → 0.667 | 0.333 → 0.333 | 0.333 → 0.667 |
| negative_trap | 5 | 0.000 → 0.000 | 0.000 → 0.200 | 1.000 → 0.800 |
| easy | 3 | 0.000 → 0.750 | 0.333 → 0.333 | 0.667 → 0.667 |

## The read (honest, win + cost)

1. **Fine-tuning clearly beats prompting on the primary axis:** F5 tripled (0.190 → 0.613), recall rose
   in **every** category, and leakage halved (0.412 → 0.196) — the prompted base massively under-tags
   (recall 0.185), and the fine-tune fixes that. This validates the SPOV-7 bet on the recall/leakage axis.
2. **The cost is precision:** over-tagging tripled (0.098 → 0.373), so overall pass-rate is flat
   (0.549 → 0.529) — the tuned model learned to tag aggressively but not yet to *withhold* on
   identically-spelled non-persons.
3. **The failure is a data-coverage gap, not hyperparameters:** over-tagging concentrates exactly where
   v1 training data was thin or absent — **person_vs_place (0 train examples → over-tag 0.1→0.5)**,
   person_vs_common (4 train → 0→0.375), possessive (0→0.67). The teacher's 54% quality-gate drop rate
   skewed v1 away from these discrimination cases.

## Day-4 target (fix in data, not hyperparameters)

Generate targeted **person_vs_place / person_vs_common / possessive / eponym-negative** examples that
teach the model to *not* tag non-person uses, then retrain and re-measure over-tag rate on those
categories. Do **not** touch lr/r/epochs to mask this.

## Caveats

- Small, skewed v1 set (146 train); numbers are directional. Bootstrap CIs (S3.5) not yet added.
- Mac path is plain LoRA on an fp16 base (not 4-bit QLoRA), so these numbers are the Mac baseline; a
  Colab 4-bit run would re-baseline separately.
- Reports: `outputs/eval_reports/{base,tuned}-*.json` (gitignored); dataset: `data/splits/`.
