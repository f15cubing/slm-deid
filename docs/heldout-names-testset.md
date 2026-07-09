# Held-out-names generalization test

**Question.** Does the model's context-sensitive name judgment transfer to names it has **never
seen** — not in training, not in the data-gen vocab bank, not in the existing `eval/hardcases`
set? (User ask: *"test the AI on names it's never seen before."*)

`eval/hardcases` is quarantined from training, but its ambiguous surfaces (Newton / Chelsea /
Grace / …) are exactly the tokens the data-gen `BLOCKLIST` reserves — so it proves the judgment
holds on the *canonical* traps, not that it generalizes to *arbitrary* new names. This probe fills
that gap.

## The set — `eval/heldout_names/heldout_names.jsonl`
74 hand-built examples, `quarantine=true`, built by
[`scripts/gen_heldout_names_testset.py`](../scripts/gen_heldout_names_testset.py). Same hard
categories as the plan, with 2–3 paraphrases per (name, sense) so consistency is measurable:

| category | n | what it tests |
|---|---|---|
| person_vs_place | 16 | place name as person (tag) vs as place (don't) |
| person_vs_common | 24 | common word as given name (tag) vs as common noun (don't) |
| person_vs_eponym | 16 | surname as person (tag) vs as unit/law (don't) |
| possessive | 6 | `Willow's essay` (tag) vs `Kepler's law` (don't) |
| first_name_only | 3 | buried first name in chat (tag) |
| third_party | 3 | author names a friend/teacher (tag) |
| negative_trap | 6 | no personal name present (tag nothing) |

**14 fresh ambiguous names**, all verified absent from training + bank + blocklist + existing eval:
`Aurora, Bragg, Jackson, Juniper, Kepler, Laurel, Marion, Meadow, Mercy, Nernst, Poppy, Snell,
Victoria, Willow`.

### Why it's a real generalization test (three disjointness guarantees)
Every ambiguous surface is verified disjoint from everything the model could have learned from —
enforced permanently by [`tests/test_heldout_names_disjoint.py`](../tests/test_heldout_names_disjoint.py):

1. **Training** — absent from `train/val/cooccur` passage text **and** `ambiguous_token` labels.
2. **Data-gen pool** — absent from the vocab bank + `BLOCKLIST` (so the teacher never seeded them).
3. **Existing eval** — absent from `eval/hardcases`, so this is a fresh, non-overlapping probe.

The generator's filter enforces the same union at build time (it dropped `Summer`/`Art` because
those words appear in the training essays).

> **Scope caveat (what this isolates).** The novel variable here is the **name**, not the phrasing.
> The context frames are deliberately the same *style* as training (educational essay / chat), and
> several rows are close paraphrases of `src/datagen/author.py`'s context templates with only the
> name swapped (e.g. "the documentary about **Marion** covered its rivers…" mirrors the training
> place-essay frame). So this measures *"does the judgment transfer to an unseen name in a
> familiar-style context?"* — which is the question asked — but part of the base→tuned delta may
> also reflect the tuned model's familiarity with those context frames, not name generalization
> alone. It is **not** a test of unfamiliar *phrasings*; the surfaces (names) are novel, the frames
> are held roughly constant. A stronger follow-up would vary the phrasing too.

> Leakage direction note: this only ever **reads** training/eval to prove the names are new; nothing
> here flows into data-gen or training. The hard-ceiling leakage guard (eval → training) is
> untouched.

## Result — base vs. v3-tuned (`outputs/sft-v3-mps`, MPS bf16→fp16 eval, greedy)

| model | n | precision | recall | leakage | over_tag | integrity_viol | pass | consistency |
|---|---|---|---|---|---|---|---|---|
| base (prompted) | 74 | 0.375 | 0.083 | 0.446 | 0.068 | 0.081 | 0.541 | 0.919 |
| **tuned (SFT v3)** | 74 | **0.947** | **1.000** | **0.000** | **0.027** | **0.000** | **0.973** | 0.946 |

**Read.** On names it has never seen, the **prompted base under-tags badly** — recall 0.083, i.e.
it misses ~92% of unseen personal names (leakage 0.446). The **fine-tuned model generalizes**: it
catches every unseen name (recall 1.00) with high precision (0.947) and zero leakage/integrity
violations. This is the SPOV-7 bet holding on out-of-vocabulary names, not just the canonical traps.

### Per-category (recall / pass, base → tuned)
| category | recall | pass |
|---|---|---|
| person_vs_eponym | 0.00 → 1.00 | 0.50 → 1.00 |
| person_vs_place | 0.00 → 1.00 | 0.44 → 0.88 |
| person_vs_common | 0.17 → 1.00 | 0.58 → 1.00 |
| possessive | 0.00 → 1.00 | 0.67 → 1.00 |
| first_name_only | 0.33 → 1.00 | 0.33 → 1.00 |
| third_party | 0.00 → 1.00 | 0.00 → 1.00 |
| negative_trap | — (no names) | 1.00 → 1.00 |

The base scores **recall 0.00** on unseen eponym/place/possessive/third-party names — it simply
doesn't recognize them as people — which is exactly the gap the fine-tune closes.

### Honest residual
The **only** 2 tuned failures (both `person_vs_place`, over-tagging) are place names heading a
clause with a person-like verb:

- *"…charted how **Jackson** expanded after the flood."* → tagged (should not)
- *"The article described how **Aurora** grew around the old rail depot."* → tagged (should not)

This is the same person-vs-place over-tag residual noted for v3 on `hardcases`, now confirmed to
persist on unseen names. It's a **data** fix (more place-as-subject negatives), consistent with the
Day-4 rule — not a hyperparameter change.

## Reproduce
```bash
python scripts/gen_heldout_names_testset.py            # rebuild the set (offline)
pytest tests/test_heldout_names_disjoint.py -q         # prove the names are unseen
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.eval.run \
  --split eval/heldout_names --compare base outputs/sft-v3-mps
```
