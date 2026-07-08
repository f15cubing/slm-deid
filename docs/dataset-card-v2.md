# Dataset card — v2 (SFT training data)

_The data the `sft-v2` LoRA will train on. Generated 2026-07-08 via the **hardened** data-gen
pipeline (`configs/datagen.yaml`, `scale: 1.0`, `seed: 0`, `gpt-4o` teacher). This is the Day-4
data iteration that targets the Day-3 over-tagging regression (base→tuned `over_tag` 0.10→0.37).
Reproduce with: `python -m src.datagen.generate --config configs/datagen.yaml --provider openai`._

## What it is
Short educational passages (student-essay + tutoring-dialogue registers), each returned **byte-identical
to its input except** that real personal names are wrapped in `⟨NAME⟩…⟨/NAME⟩`. The bulk is **matched
minimal pairs**: the *same* ambiguous surface used once as a person (tagged) and once as a non-person
(untagged) — the person-vs-non-person contrast the Day-3 data lacked. Ambiguous tokens are drawn from a
curated bank that is **disjoint from the quarantined eval set** (`src/datagen/vocab.py`), so the
hard-cases eval stays a clean generalization test.

## Size & split
| split | rows | tagged | untagged | name spans |
|---|---|---|---|---|
| train | **164** | 74 | 90 | 223 |
| val | 18 | 6 | 12 | — |

Passage length (train): min 7 / median 74 / mean 62.9 / max 116 words. Register: **110 essay / 54
dialogue** (33% dialogue). Source: 128 `synthetic_teacher` + 36 `presidio_faker` (pattern-type
negatives). 36 distinct ambiguous tokens.

> Smaller than the S3.1 target (800–2,000) — a deliberate consequence of the strict gate (64.5% drop).
> Quality/cleanliness was prioritized over raw size for this iteration.

## Category coverage (train) — v1 → v2
| category | v1 train | v2 tagged | v2 untagged | note |
|---|---|---|---|---|
| person_vs_place | 0 | 11 | 15 | **gap closed** (was the worst Day-3 over-tag) |
| person_vs_common | 4 | 20 | 42 | **gap closed**, strong contrast |
| person_vs_eponym | 40 | 9 | 8 | **now has negatives** (was 0) |
| possessive | 1 | 1 | 0 | **collapsed** — see Limitations |
| negative_trap | 55 (polluted) | 0 | 25 | **clean** (0 names) |
| first_name_only | 22 | 9 | 0 | adequate |
| third_party | 14 | 7 | 0 | adequate |
| easy | 10 | 17 | 0 | adequate |

**Minimal-pair contrast** (same token seen as both person and non-person): `person_vs_common` 13
true pairs, `person_vs_eponym` 4, `person_vs_place` 3, `possessive` 0.

## Quality gates (all enforced; only passing rows are kept)
- **Integrity** — `unwrap(target) == input` byte-for-byte.
- **Tag well-formedness** — balanced, non-nested `⟨NAME⟩` tags.
- **Schema** — offsets align; tagged spans == gold name spans; valid enums.
- **Teacher verification** — a second-pass re-tag must agree (drop on disagreement / altered text).
- **Category semantics** — `negative_trap` ⇒ 0 names; `person_vs_*` ⇒ the ambiguous token is present;
  `possessive` ⇒ a possessive form is present.
- **Disposition** (minimal pairs) — the person half must tag its token; the non-person half must tag
  **nothing**. Whole pair dropped otherwise.
- **Leakage (hard ceiling)** — intended-token guard → **passage-surface guard** (drops any passage
  containing *any* eval surface, e.g. an invented “Charles Darwin”) → exact-passage de-leak.

### Drop breakdown (raw 512 → 182 kept, 64.5% dropped)
| reason | count |
|---|---|
| pair_disposition | 176 |
| malformed_tags | 48 |
| verifier_disagreement | 41 |
| negative_trap_has_name | 31 |
| verifier_altered_text | 19 |
| eval_surface_leak | 15 |
| eval_token_leak | 0 |
| eval_leak (passage) | 0 |

## Verified clean (this build)
- **Eval-surface leakage in the final data: 0 rows** (the guard dropped 15 upstream, incl. the
  Darwin-tagged cases the first v2 run leaked).
- **Disposition: 100% correct** — person halves tag their token (44/44), non-person halves tag nothing
  (75/75).
- **`tests/test_no_eval_leakage.py` passes**; schema-invalid rows: 0.

## Known limitations
1. **`possessive` collapsed to 1 example.** The teacher repeatedly tags the eponym in “Kelvin’s law”
   (its surname *is* a person’s), and the disposition guard correctly drops those — clean eponymous-
   possessive **negatives** are hard to distill from `gpt-4o`. `possessive` is `n=3` in the eval, so
   the impact is small and measurable; a prompt fix + top-up is the follow-up if the eval shows
   possessive over-tagging persists.
2. **Below the 800–2,000 size target** (164 train), by design (strict gate). Clean-small over
   large-dirty for this iteration.
3. **Synthetic-distilled** (`gpt-4o`), not human-labeled; the CRAPII real slice is available
   (`src/datagen/real_data.py`) but not folded in here.
4. Register is now balanced (33% dialogue) but still essay-leaning.

## Provenance & reproduction
- Config: `configs/datagen.yaml` (`scale: 1.0`, `seed: 0`); vocab bank: `src/datagen/vocab.py`.
- Command: `python -m src.datagen.generate --config configs/datagen.yaml --provider openai`.
- Error analysis that motivated this iteration: [`docs/error-analysis-v1.md`](error-analysis-v1.md).
- Results (base vs. tuned) will be appended for v2 in [`docs/results.md`](results.md) after the
  `sft-v2` retrain + re-eval.
