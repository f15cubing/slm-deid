# Model card — `sft-v3-gpt551` (De-Id name-judgment QLoRA, canonical live-teacher)

_Trained 2026-07-10 on a Colab **A100-40GB**. Adapter: `outputs/sft-v3-gpt551/` (gitignored, 133 MB;
adapter + splits + reports also on Drive). This is the **canonical live-teacher run** — the follow-up
every prior card flagged as open. It replaces the authored-teacher runs as the credible reference line
because its labels come from a **live teacher + independent verifier**, not in-session templates._

## What it is
A **4-bit QLoRA adapter** over `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` for **context-sensitive
personal-name judgment**: given a short educational passage it returns the text byte-identical except
that spans naming a real person are wrapped in `⟨NAME⟩…⟨/NAME⟩`, and nothing else is tagged. Pattern
types (email/phone/ID) and output format are out of scope (handled by the surrounding pipeline).

## Why gpt551 exists
Every prior run (v1–v3, authored scale=2/10) trained on data whose labels rested on **in-session
template authoring with no independent verifier** — the single caveat that qualified every headline
number. This run closes it: a **live OpenAI-compatible teacher** (model id `gpt551`, via the TrueFoundry
LLM Gateway — `--provider openai` + `OPENAI_BASE_URL`/`TEACHER_MODEL`) writes the passages, and an
**independent verifier pass** rejects teacher/verifier disagreements. The result is the project's most
credible base-vs-tuned estimate, even though its absolute hard-case scores are not the highest (see
Honest limitations).

## Training setup
| | |
|---|---|
| Base model | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` (4-bit, non-thinking) |
| Method | QLoRA (unsloth + bitsandbytes 4-bit), **bfloat16** compute (A100/Ampere) |
| LoRA | r=32, α=32, dropout 0.0; q/k/v/o + gate/up/down proj |
| Optim | lr=2e-4, 3 epochs, eff. batch 16 (2×8), linear, warmup 0.05, wd 0.01, seed 0 |
| Masking / seq | completion-only; 2048 |
| Config | `configs/train.yaml` (**frozen since Day 4** — data-only iteration; no HP change) |
| Result | 156 steps / 3 epochs, loss 0.474 → ~2e-4, no NaN |
| Enabled by | the Colab EOS-token library-compat fix (PR #35) — Unsloth's `<EOS_TOKEN>` placeholder |

## Training data (v3, live teacher) — 818 train / 90 val
Full card: `docs/dataset-card-v3.md` (live-teacher section). Generation: matched minimal pairs
(person_vs_common / _place / _eponym / possessive) + single passages, live teacher + verifier.
Quality-gate + leakage drops: **134** verifier-disagreement, **98** eval-surface-leak, **48**
negative-trap-has-name, 3 possessive-not-possessive, 3 verifier-altered-text, 1 missing-ambiguous-token.
- **Leakage verified clean (hard ceiling):** all three generation guards `eval_leak = 0`, and
  **independently re-verified post-hoc** — 0 exact and 0 substring overlaps between the 818/90 splits and
  all **201** quarantined eval inputs (hardcases / adversarial / heldout / ood).

## What this training achieved
Evaluated on the 51 quarantined hard cases (base = prompted 4-bit `Qwen3-1.7B`, same backend). Full
tables + 95% bootstrap CIs: `docs/results.md` → gpt551. Headlines (base → tuned):

- **F5 0.507 → 0.847**, **recall 0.518 → 0.852** — CIs separate from base.
- **over_tag 0.549 → 0.157**, **integrity 0.588 → 0.020**, **leakage 0.255 → 0.078**.
- **pass_rate 0.353 → 0.824**, **consistency 0.375 → 0.562**.
- Per-category: `person_vs_eponym` recall 0.33 → 1.00 (pass 0.12 → 0.88), `first_name_only` and
  `third_party` 0 → catching names, `person_vs_place` recall → 1.00, `negative_trap` pass 0.40 → 1.00.
  `person_vs_common` recall is **flat at 0.75** — its gain is over_tag 0.56 → 0.12 (pass 0.38 → 0.81).

## Honest limitations & residual failures
1. **Lower than the authored run — reported, not hidden.** Tuned pass 0.82 vs the authored scale=2 run's
   0.96 (over_tag 0.16 vs 0.04, consistency 0.56 vs 0.94). Not underfitting (loss reached ~2e-4). Most
   likely **distributional**: the authored templates were written by the same author who designed the
   eval categories, so authored training text sits closer to the hard cases than a live teacher's more
   varied prose. On that reading the gpt551 number is the more credible one — we do **not** claim it beats
   the authored run.
2. **`possessive` is the unmoved category** — recall 1.0 but over_tag/integrity both **0.333** (n=3; it
   carries the single overall integrity violation). Possessive contrast has been the persistent weak spot
   across all runs; the clearest next data-iteration target.
3. **Small eval (n=51), single seed**; per-category n=3 cells are noisy (`[0.00, 1.00]` bands).
4. **Single teacher + single verifier pass** — better than authored templates, not a multi-teacher
   consensus. One teacher model, one verifier gate.

## Provenance & reproduction
- Base: `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`; adapter `sft-v3-gpt551` (r=32/α=32; checkpoint-156).
- Reports: `outputs/eval_reports_colab_gpt551/{base-20260710-210831,tuned-gpt551-20260710-210939}.json`
  (gitignored). Regenerate tables offline: `python -m src.eval.report base=… tuned=…` (see results.md).
- Reproduce (Colab, `notebooks/v3_colab_train_eval.ipynb`): live teacher via `--provider openai` with
  `OPENAI_BASE_URL`/`TEACHER_MODEL` set to the gateway → frozen `configs/train.yaml` QLoRA → base-vs-tuned
  on `eval/hardcases`. Data-only iteration; hyperparameters frozen (Day-4 rule).
- Related: [[slm-deid]] lineage — `docs/model-card-v3.md` (bf16 MPS), `docs/results.md` (all runs).
