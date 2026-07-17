---
license: apache-2.0
language:
  - en
task_categories:
  - token-classification
tags:
  - de-identification
  - pii
  - named-entity-recognition
  - privacy
  - education
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/val.jsonl
---

# Dataset card — v3 (SFT training data)

_The data `sft-v3-mps` trained on. Built 2026-07-08 to fix v2's recall/consistency regression by
**rebalancing toward person-use and scaling up**. Supersedes `docs/archive/dataset-card-v2.md`._

## What it is
Short educational passages (essay + tutoring-dialogue), each byte-identical to its input except that real
personal names are wrapped in `⟨NAME⟩…⟨/NAME⟩`. The bulk is **matched minimal pairs**: the same ambiguous
surface used once as a person (tagged) and once as a non-person (untagged) — e.g. `Ruby` the person vs
`ruby` the gem, `June` the person vs `June` the month, `Newton` the person vs the `Newton method`. Tokens
come from an **eval-disjoint** curated bank (`src/datagen/vocab.py`), so the hard-cases set stays a clean
generalization test.

## Size & split
| split | rows | tagged | untagged | name spans | words (min/med/max) |
|---|---|---|---|---|---|
| train | **927** | 542 | 385 | 576 | 6 / 11 / 554 |
| val | 102 | 54 | 48 | 62 | 6 / 11 / 513 |

Sources (train): **synthetic_teacher 777**, **real_crapii 109**, **presidio_faker 41**. Registers: 645
essay / 282 dialogue (30% dialogue).

## The v2 → v3 fix: rebalanced person-use
v2's `person_vs_common` was 2:1 withhold-skewed (18 person / 38 withhold), which taught the model to
under-tag ambiguous names. v3 restores balance and scale:

| category (train) | v2 person / withhold | v3 person / withhold |
|---|---|---|
| person_vs_common | 18 / 38 | **128 / 127** |
| person_vs_place | ~? | 101 / 93 |
| person_vs_eponym | ~? | 55 / 44 |
| possessive | ~1 total | 82 / 82 |

Category totals (train): person_vs_common 255, person_vs_place 194, possessive 164, real 109,
person_vs_eponym 99, negative_trap 39, easy 31, third_party 18, first_name_only 18.

**Vocab bank tripled** (eval-disjoint, verified by `tests/test_vocab.py`): COMMON_WORDS 19→43, PLACES
16→32, EPONYMS 18→35 — so the person/non-person contrast is taught over many surfaces (generalization),
not memorized on a few.

## How it was generated (methodology caveat)
The frontier teacher (gpt-4o) was **unavailable** (OpenAI billing inactive; no Anthropic key). Passages
were **authored in-session** from context templates + the vocab bank (`src/datagen/author.py`, invoked as
`--provider authored`) and run through the **same** pipeline as any teacher output: quality gate,
minimal-pair disposition check, and all eval-leakage guards. **No independent verifier second pass**
(label trust rests on the deterministic gates + tags placed by construction). Template text is **less
varied** than a frontier teacher's — the main limitation of this build.

## Live-teacher build (gpt551) — resolves the caveat above
On 2026-07-10 the **same v3 recipe** was re-run on Colab with a **live OpenAI-compatible teacher** (model
id `gpt551`, via the TrueFoundry LLM Gateway — `--provider openai` + `OPENAI_BASE_URL`/`TEACHER_MODEL`)
**plus an independent verifier pass**, producing the data the canonical **`sft-v3-gpt551`** model trained
on (see `docs/archive/model-card-gpt551.md`, `docs/results.md` → gpt551). This is the frontier-teacher regen the
authored build flagged as the follow-up, and it removes the "no independent verifier / low variety"
limitation for this line.

| split | rows | notes |
|---|---|---|
| train | **818** | live teacher + verifier; minimal pairs (~400 ×2 calls) + 166 singles, + co-occurrence/CRAPII folds |
| val | 90 | same generation |

**Drop funnel (live teacher):** `verifier_disagreement 134` (teacher vs independent verifier — the gate
the authored build lacked), `eval_surface_leak 98`, `negative_trap_has_name 48`,
`possessive_not_possessive 3`, `verifier_altered_text 3`, `missing_ambiguous_token 1`;
**`eval_token_leak 0`, `eval_leak 0`.**

**Leakage — independently re-verified (hard ceiling):** beyond the three in-pipeline guards returning 0, a
post-hoc scan found **0 exact and 0 substring overlaps** between the 818/90 splits and **all 201**
quarantined eval inputs (hardcases / adversarial / heldout / ood).

**Honest note:** the live-teacher model scores *below* the authored build on the 51 hard cases (pass 0.82
vs 0.96) — most likely because the authored templates sit closer to the eval distribution; see
`docs/results.md` → gpt551 for the full read. The splits/adapter/reports persist to Drive; reports are
mirrored at `outputs/eval_reports_colab_gpt551/` (the 133 MB adapter and the splits are not committed).

## Quality gates (all enforced; only passing rows kept)
Integrity (`unwrap(target)==input`), tag well-formedness, schema, category-semantics (negative_trap ⇒ 0
names; person_vs_* ⇒ intended token present; possessive ⇒ possessive form), minimal-pair disposition
(person half tags its token; non-person half tags nothing), and the three eval-leakage guards.

### Build funnel
- Generate batch (`--provider authored --seed 1`, scale 2.0): **908 train / 100 val**; drops
  `missing_ambiguous_token 7`, `eval_surface_leak 1`, disposition 0, eval_leak 0.
- Merge (+ CRAPII `--crapii-limit 150`): sources 1008 + crapii 150 → dedup −99 → **eval_surface_leak −30**
  (CRAPII passages containing eval surfaces) → **eval_token_leak 0, eval_leak 0** → 927 train / 102 val.

## Verified clean (hard ceiling)
All three eval-leakage guards return **0** on the final splits; positive control fires on **50/51** eval
inputs; an independent raw scan of all 1,029 rows for the 40 eval surfaces (incl. `may`/`hope`/`grace`)
finds **none**. `tests/test_no_eval_leakage.py` + `tests/test_vocab.py` green.

## Human review (seal of approval)
Labels are placed by construction and gated deterministically, but several sets have now also been
**reviewed item-by-item by a human** in `scripts/review_ui.py` (Approve/Deny per row; sealed to
`reviews/<split>.approved.jsonl`). Status as of 2026-07-09:

| set | rows | reviewed | approved | denied | status |
|---|---|---|---|---|---|
| val (`data/splits/val.jsonl`) | 102 | 102 | 102 | 0 | ✅ fully reviewed + sealed |
| hard-cases test set (`eval/hardcases/hardcases.jsonl`) | 51 | 51 | 50 | 1 | ✅ fully reviewed + sealed (1 flagged for follow-up) |
| co-occurrence contrast (`data/cooccur/cooccur.jsonl`) | 29 | 29 | 29 | 0 | ✅ fully reviewed + sealed |
| train (`data/splits/train.jsonl`) | 927 | partial | — | — | 🔶 human review in progress (large set; not expected to complete) |

So the **held-out test set and the validation split are fully human-approved**, not just
machine-gated. Sealed approved-only exports live under `reviews/` (git-ignored; regenerate any time
from the reviewer). Review decisions are advisory metadata — they do not alter the source splits.

## Known limitations
1. **Template-authored** (not frontier-distilled) — less linguistic variety. _Resolved for the canonical
   line:_ the **live-teacher gpt551 build** (see above) regenerates this recipe with a real teacher +
   verifier. _Labeling_ is also no longer fully unreviewed: val + the hard-cases test set are 100%
   human-approved (see **Human review** above); the 927-row authored train split is only partially reviewed.
2. **`possessive` contrast remains hard** — eponymous-possessive negatives ("Joule's law") are subtle;
   the eval still shows possessive over-tagging ("Newton's laws").
3. **CRAPII real slice (109)** carries the NAME_STUDENT under-tagging caveat (`src/datagen/real_data.py`).

## Provenance & reproduction
- Config: `configs/datagen.yaml` (scale 2.0, seed 0; CRAPII folded at merge). Bank: `src/datagen/vocab.py`.
- Commands: see `docs/archive/model-card-v3.md` → Reproduction. Results: `docs/results.md` → v3.
