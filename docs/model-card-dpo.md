# Model card — `dpo-gpt551` (De-Id name-judgment, DPO on top of SFT)

_Day-5 stretch rung 1. **Preference optimization (DPO)** applied on top of the canonical live-teacher
SFT adapter (`sft-v3-gpt551`, see [`docs/model-card-gpt551.md`](model-card-gpt551.md)). Built by
`notebooks/dpo_colab.ipynb`; the adapter + pairs + reports persist to `MyDrive/slm-deid-dpo-gpt551/`
(gitignored)._

> **STATUS: numbers pending the Colab run.** This card documents the *method and configuration* that
> ship in code (draft PR). The results section below is intentionally empty — per the project's hard
> ceiling (*never claim a base-vs-tuned win without the numbers*), it is filled **only** from a fresh
> eval run, reported honestly whether DPO helps, hurts, or is a wash.

## What it is
A **DPO adapter** continuing the gpt551 SFT LoRA over `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`. Same
behavior contract as SFT (wrap real-person-name spans in `⟨NAME⟩…⟨/NAME⟩`, change nothing else); DPO
does not change the task, only sharpens spec-adherence via a preference signal.

## Why DPO (the Day-5 question)
SFT (gpt551) already wins decisively on the hard cases, but its **weakest axes are precision-shaped**:
`over_tag 0.16`, `consistency 0.56`, `pass 0.82`. The stretch-rung question is whether **preference
optimization sharpens those axes beyond SFT alone** — i.e. does showing the model *on-spec vs off-spec*
pairs push down over-tagging and up consistency without giving back recall. gpt551 (not the near-ceiling
authored run) is the baseline precisely because it has the headroom to make the answer legible.

## Preference data (hybrid pairs) — `src/train/prefs.py`
Each pair is `{prompt, chosen, rejected}` where `chosen` is the gold tagging (the SFT row's `target`).
The `rejected` side is sourced two ways:

- **Stage A — on-policy.** The SFT model is sampled over the train inputs; rows where it **genuinely
  errs** (`behavioral_checks.check` → not passed) contribute the model's *own* wrong output as
  `rejected`. Highest-signal — DPO learns against real failure modes.
- **Stage B — deterministic backfill.** Where SFT was already correct, `rejected` is synthesized by
  perturbing the gold spans — **over-tag** (wrap an extra capitalized non-name, skipping function
  words), **missed-name** (drop one gold tag), **wrong-boundary** (swallow a possessive / adjacent
  word). Round-robined so all three failure modes and all categories are represented.

Every `rejected` preserves `unwrap(rejected) == input` and stays well-formed → a pure **judgment**
error, not a text-integrity error. So the DPO contrast is strictly about *which spans are names*.

- **Leakage (hard ceiling).** Pairs are drawn only from the eval-disjoint train split, and no
  perturbation edits the passage text, so no eval surface can enter. The notebook asserts
  `prefs.eval_leak_count(pairs, "eval") == 0` (reuses `vocab.BLOCKLIST` + eval-input matching)
  **before** any DPO step. Verified 0 on a local dry run of the committed split.

## Training setup — `configs/dpo.yaml` (new, frozen)
| | |
|---|---|
| Init (policy) | the gpt551 SFT adapter (`outputs/sft-gpt551/`) |
| Reference | **adapter-disabled base** (`ref_model=None` + PEFT policy) → regularizes toward base, not SFT |
| Method | QLoRA-DPO (unsloth 4-bit + `PatchDPOTrainer`), TRL `DPOTrainer` |
| β | 0.1 |
| Optim | lr **5e-6** (low — refining an already-good policy), 1 epoch, eff. batch 16 (4×4), linear, wd 0.01, seed 0 |
| Masking / seq | prompt masked by DPO; max_len 2048, max_prompt_len 1024 |
| Config | `configs/dpo.yaml` — **new knobs** (β/DPO-lr/epochs). The frozen SFT `configs/train.yaml` is untouched, so the Day-4 ceiling holds |

**On the reference-model choice.** `ref_model=None` with a PEFT policy makes TRL use the base
(adapter disabled) as the DPO reference — the standard, memory-cheap QLoRA-DPO path. It regularizes
toward the *base* model rather than toward SFT. A separate SFT reference (more correct, ~2× memory) is
a documented follow-up, not this run.

## Evaluation plan
`base` vs `sft-gpt551` vs `dpo`, each loaded once and scored on the same 4-bit runtime across the
quarantined sets (`hardcases`, `adversarial`, `heldout_names`, `ood_probe`) so the three columns are
directly comparable. **Win condition:** DPO improves `over_tag` / `consistency` / precision on the hard
cases without regressing recall. Reports written to `outputs/dpo_reports/`.

## Results
_Pending the Colab run. To be filled from `outputs/dpo_reports/` (base → sft → dpo), with 95% bootstrap
CIs via `python -m src.eval.report`, and cross-linked into `docs/results.md`. If DPO does **not** beat
SFT, that is reported as-is — a null result on the stretch rung is a legitimate finding._

## Honest limitations (method-level, pre-results)
1. **Reference = base, not SFT** (see above) — the regularization target is weaker than a true
   SFT-reference DPO.
2. **On-policy supply depends on SFT error rate.** gpt551 is already strong on train-like inputs, so
   the on-policy pair count may be modest; Stage B backfill guarantees coverage but its negatives are
   synthetic (off-distribution vs errors the model would actually make).
3. **QLoRA-DPO on a 4-bit base** — same quantization caveat as the SFT line; DPO numbers re-baseline
   against the gpt551 SFT column, not against any bf16 run.
