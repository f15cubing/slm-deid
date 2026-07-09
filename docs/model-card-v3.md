# Model card — `sft-v3-mps` (De-Id name-judgment LoRA)

_Trained 2026-07-08 on Apple-Silicon MPS. Adapter: `outputs/sft-v3-mps/` (gitignored; provenance bundle
in `outputs/sft-v3-mps/provenance/`). This is the **data-rebalance iteration** that fixes v2's
recall/consistency regression — **in the data, not the hyperparameters**._

## What it is
A **LoRA adapter** over `Qwen/Qwen3-1.7B` for **context-sensitive personal-name judgment**: given a short
educational passage it returns the text byte-identical except that spans naming a real person are wrapped
in `⟨NAME⟩…⟨/NAME⟩`, and nothing else is tagged. Pattern types (email/phone/ID) and format are out of
scope (regex + constrained decoding in the surrounding pipeline).

## Why v3 exists
v2 fixed over-tagging/integrity but **regressed recall (0.63→0.44) and consistency (0.25→0.13)**: it
learned to *withhold* on ambiguous common-word names (missed "Grace lent me notes", "May volunteered").
Root cause was measurable in the data — `person_vs_common` was **2:1 withhold-skewed** (18 person-use vs
38 non-person-use) with little context variety, so the model keyed on the surface, not the context. v3
fixes the data.

## Training setup
| | |
|---|---|
| Base model | `Qwen/Qwen3-1.7B` (full-precision, non-thinking) |
| Method | LoRA (PEFT) via the `hf`/MPS backend, **bfloat16** |
| LoRA | r=32, α=32, dropout 0.0; q/k/v/o + gate/up/down proj |
| Optim | lr=2e-4, 3 epochs, eff. batch 16 (2×8), linear, warmup 0.05, wd 0.01, seed 0 |
| Masking / seq | completion-only; 2048 |
| Config | `configs/train.mps.bf16.yaml` (**byte-identical to v1/v2** — data-only iteration) |
| Result | `train_loss 0.078`, `mean_token_accuracy 0.9999`, no NaN, 174 steps, ~113 min |

## Training data (v3) — 927 train / 102 val
Full card: `docs/dataset-card-v3.md`. What changed vs v2:
- **Rebalanced `person_vs_common` to ~50/50** person/non-person (v2 was 18/38) via matched minimal pairs,
  and **~doubled the eval-disjoint vocab bank** (53→110 tokens: 43 common / 32 places / 35 eponyms) at data-build time
  (post-merge, `src/datagen/vocab.py` is the larger consolidated bank — ~167 tokens — used for future runs)
  so the model learns the contrast over many surfaces (generalization), not a handful.
- **Scaled ~3.8×** (242 → 927 train). Sources: synthetic_teacher 777, real_crapii 109, presidio_faker 41.
  Registers 645 essay / 282 dialogue. Tagged 542 / untagged 385.
- **Leakage verified clean (hard ceiling):** all three guards 0 hits, positive control fires 50/51,
  independent raw scan of all 1,029 rows for the 40 eval surfaces = 0; the guards dropped 30 CRAPII
  passages that contained an eval surface.

### Methodology caveat — in-session authored teacher
The gpt-4o/Anthropic teacher was **unavailable** (OpenAI billing inactive; no Anthropic key). Per the
approved plan, v3 passages were **authored in-session** from context templates + the vocab bank
(`src/datagen/author.py`, `--provider authored`) and routed through the **same** quality gate,
disposition check, and eval-leakage guards. Consequence: **no independent second-pass verifier** (label
trust rests on the deterministic gates; every tag is placed by construction), and template text is **less
linguistically varied** than a frontier teacher's. Read the size of the win with this in mind; a
frontier-teacher regen is the natural follow-up.

## What this training achieved
Evaluated on the 51 quarantined hard cases (base = prompted `Qwen/Qwen3-1.7B`, same MPS backend). Full
tables + CIs: `docs/results.md` → v3. Headlines (base → tuned):

- **recall 0.185 → 0.926**, **F5 0.190 → 0.919** — CIs separated from base (`[0.81,1.00]` vs `[0.04,0.35]`).
- **consistency 0.250 → 0.750** — v2's regression (0.13) not just recovered but past base.
- **leakage 0.412 → 0.039**, **pass_rate 0.549 → 0.863**, **integrity → 0.000**.
- **over_tag held at 0.137** (≈ v2; CI overlaps base) — the recall gain did **not** cost precision.
- Per-category: `person_vs_common` recall **0.125 → 1.000**, `person_vs_place`/`person_vs_eponym`/
  `first_name_only` → 1.000. The model now correctly splits `Newton` (person) from the `Newton method`,
  `Grace`/`Rose`/`May`/`Bishop` the person from the flower/month/chess-piece.

## Honest limitations & residual failures
1. **Eponymous possessive** ("Newton's laws") is still over-tagged — `possessive` is the weak category
   (its clean contrast is hard to author). `negative_trap` over_tag ticked to 0.200 from one case where
   the model tagged the pronoun "She".
2. **Small eval (n=51), single seed**; per-category n=3 cells are noisy.
3. **Template-authored data** (see caveat) — the strong numbers partly reflect that the model generalizes
   from clean, balanced, if less-varied, contrast examples to the (disjoint) eval surfaces. A
   frontier-teacher rebuild would test robustness to messier language.
4. **bf16 lineage** (fp16 NaN-diverges on long passages, see [[slm-deid-mps-fp16-nan]] / v2 card).

## Provenance & reproduction
- Bundle: `outputs/sft-v3-mps/{MANIFEST.md, provenance/}` — config, datagen.yaml, `train.log`, both eval
  reports, exact `train.jsonl`/`val.jsonl`, `adapter.sha256`.
- Adapter SHA-256: `5a9f17da2069084424e0c8173dcb15cf0cbe127d6a4d9b09685a492b48fa0010`
- Rebuild data: `python -m src.datagen.generate --config configs/datagen.yaml --provider authored --seed 1 --out-dir data/_b1`
  then `python -m src.datagen.merge --out data --sources data/_b1/splits/train.jsonl data/_b1/splits/val.jsonl --crapii data/raw/cleaned_repository_pii_train.json --crapii-limit 150`
- Retrain: `PYTORCH_ENABLE_MPS_FALLBACK=1 python -u -m src.train.qlora --config configs/train.mps.bf16.yaml --output-dir outputs/sft-v3-mps`
- Re-eval: `python -m src.eval.run --split eval/hardcases --compare base outputs/sft-v3-mps --report-dir outputs/eval_reports`
