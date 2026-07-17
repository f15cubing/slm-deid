# Model card — `sft-v2-mps` (De-Id name-judgment LoRA)

_Trained 2026-07-08 on Apple-Silicon MPS. Adapter: `outputs/sft-v2-mps/` (gitignored; provenance
bundle in `outputs/sft-v2-mps/provenance/`). This is the **Day-4 data iteration** — the retrain that
was supposed to fix v1's over-tagging (0.37) and integrity (0.118) regressions **in the data, not the
hyperparameters**._

## What it is
A **LoRA adapter** over `Qwen/Qwen3-1.7B` that performs the one thing this project fine-tunes for:
**context-sensitive personal-name judgment**. Given a short educational passage it returns the text
**byte-identical except** that every span referring to a real person's name is wrapped in
`⟨NAME⟩…⟨/NAME⟩`, and nothing else is tagged. Pattern types (email/phone/ID) and output format are
out of scope — handled by regex + constrained decoding in the surrounding pipeline.

## Training setup
| | |
|---|---|
| Base model | `Qwen/Qwen3-1.7B` (full-precision, non-thinking mode) |
| Method | LoRA (PEFT) via the `hf`/MPS backend |
| LoRA | r=32, α=32, dropout=0.0; targets q/k/v/o + gate/up/down proj |
| Optimization | lr=2e-4, 3 epochs, effective batch 16 (2×8 grad-accum), linear sched, warmup 0.05, wd 0.01, seed 0 |
| Precision | **bfloat16** — see "Why bf16" below |
| Masking | completion-only (loss on the tagged output, prompt masked) |
| Seq length | 2048 |
| Config | `configs/train.mps.bf16.yaml` |
| Result | `train_loss 0.0298`, `mean_token_accuracy 0.9993`, no NaN, ~56 min, 48 steps |

**LoRA hyperparameters are byte-identical to v1** (`configs/train.mps.yaml`). The only change from v1
was the **data** — this respects the hard ceiling *"never touch hyperparameters to paper over a data
problem."* The one non-data change (fp16→bf16) is a numerical-precision fix, not a capacity/learning
knob (see below).

### Why bf16 (methodology change to record)
v1 trained fine at the documented `dtype: float16`, but the **v2 data NaN-diverged at fp16**
(`grad_norm=nan` from the first logged step, loss→0.0). Root cause: fp16 overflow aggravated by the
long real CRAPII passages (38/242 train rows > 1024 tokens; max 521 words vs v1's short synthetic
passages). The data itself is clean (integrity holds, nothing exceeds `max_seq_len`). Fix: **bfloat16**
— fp32's exponent range (kills the overflow) at fp16-like speed (fp32 also works but ran ~2.5× slower
on MPS). Consequence: this adapter is a **bf16 lineage**, so its numbers re-baseline vs v1's fp16 run —
but the bf16 base reproduces the Day-3 fp16 base almost exactly, so base-vs-tuned stays a fair test.

## Training data summary (v2, CRAPII-augmented)
**242 train / 26 val.** Byte-identical-except-tags passages in essay + tutoring-dialogue registers.
Full dataset card (pre-CRAPII generation details, quality gates, drop funnel): `docs/archive/dataset-card-v2.md`.
> Note: the dataset card documents the 164/18 synthetic core; **this training used the CRAPII-spliced
> 242/26 splits** in `data/splits/` (commit `307b449`). The composition below is the as-trained data.

| | train | val |
|---|---|---|
| rows | 242 | 26 |
| tagged / untagged | 155 / 87 | 14 / 12 |
| name spans | 319 | 25 |
| words (min/median/mean/max) | 7 / 88 / 179 / 521 | 7 / 78 / 143 / 554 |
| registers | 187 essay / 55 dialogue | 20 essay / 6 dialogue |

**Sources (train):** `synthetic_teacher` 118 (gpt-4o minimal pairs + per-category passages),
`real_crapii` 89 (real student-essay slice), `presidio_faker` 35 (pattern-type negatives).

**Categories (train):** real 89, person_vs_common 56, person_vs_place 26, negative_trap 25,
person_vs_eponym 15, easy 15, first_name_only 9, third_party 6, **possessive 1**.

**Leakage — verified clean (hard ceiling).** The splits share **zero** surface with the 51 quarantined
hard cases: all three guards (exact-passage, intended-token, blocklist-surface) return 0 hits, the
guard fires on 50/51 eval inputs (live, not a no-op), and an independent raw scan for all 40 eval
surfaces across all 268 rows found none — including common words (`may`/`hope`/`rose`) that real CRAPII
text could contain, confirming the CRAPII slice was routed through the de-leak, not assumed clean.

## What this training achieved
Evaluated on the **51 quarantined hard cases** (`eval/hardcases/`), base = prompted `Qwen/Qwen3-1.7B`,
same MPS backend for both. Full tables + CIs: `docs/results.md`. Headlines:

**The Day-4 goal was met — the v1 regressions are fixed:**
- **over_tag_rate 0.37 (v1) → 0.137 (v2)** — the aggressive-tagging regression is largely undone.
- **integrity_violation 0.118 (v1) → 0.020 (v2)** — now *below* base (0.039); the hard-ceiling
  regression is gone.
- **pass_rate 0.549 → 0.627** (+0.078) — a real gain, where v1's pass-rate was flat.

**Tuned still beats prompting on the safety axis:** recall 0.185→0.444, F5 0.190→0.450 (CIs separate
from base), leakage 0.412→0.275. Standout per-category: first_name_only recall 0→1.0 (pass 0→1.0),
possessive 0→1.0, person_vs_eponym 0.33→0.67, third_party 0.33→0.67.

## Honest limitations & known failures
1. **Consistency regressed: 0.25 → 0.125.** The tuned model is *less* stable across paraphrases of the
   same name/non-name than base. This is the headline weakness of this iteration.
2. **`person_vs_common` recall is flat at 0.125** — the largest eval category (16 cases); the model
   still barely tags common-word given-names (Grace/Rose/…). The next data target.
3. **Lower recall/F5 than v1's tuned run** (0.44 vs 0.63 / 0.45 vs 0.61). But v1 bought that recall
   with 0.37 over-tagging and 3× integrity violations; v2 is the more balanced, spec-honest model
   (higher pass-rate). Trade acknowledged, not hidden.
4. **Possessive is `n=1` in training** — the disposition guard drops eponymous-possessive negatives the
   teacher mislabels; a prompt fix + top-up is the follow-up.
5. **Small eval (n=51), single seed.** Per-category CIs are wide (n=3 cells span most of [0,1]); only
   the overall recall/F5/leakage gaps clearly separate from base.

## Provenance & reproduction
- Adapter + bundle: `outputs/sft-v2-mps/` — weights, tokenizer, and `provenance/` (config, `train.log`,
  `eval-base.json`, `eval-tuned.json`, `train.jsonl`, `val.jsonl`, `adapter.sha256`).
- Adapter SHA-256: `40c9c823d98ef17660621f8a8b3baf7573aca21fa35989ccdda191929536639d`
- Trained at git commit `307b449` on branch `agent/datagen-v2-run`.
- Retrain: `PYTORCH_ENABLE_MPS_FALLBACK=1 python -u -m src.train.qlora --config configs/train.mps.bf16.yaml --output-dir outputs/sft-v2-mps`
- Re-eval: `python -m src.eval.run --split eval/hardcases --compare base outputs/sft-v2-mps --report-dir outputs/eval_reports`
- Regenerate tables (offline, no model/network): `python -m src.eval.report base=<report> tuned=<report>`
