# Day-by-day tasks

Concrete, spec-level breakdown of `docs/plan.md`. Each day file has the same shape:

- **Goal** — the one-line outcome.
- **Specs** — testable acceptance criteria (the definition of done). Numbered `S<day>.<n>`.
- **Tasks** — concrete checkboxes, TDD-oriented, referencing real files.
- **Deliverables** — artifacts that must exist at end of day.
- **Checkpoint** — the hard gate from the plan; do not proceed past it unmet.

| Day | File | Focus |
|-----|------|-------|
| 1 | [day-1.md](day-1.md) | Environment + scope lock + tag syntax |
| 2 | [day-2.md](../archive/tasks/day-2.md) | Behavior spec + eval harness + data-gen + smoke test |
| 3 | [day-3.md](../archive/tasks/day-3.md) | v1 dataset + first QLoRA run + first base-vs-tuned numbers |
| 4 | [day-4.md](../archive/tasks/day-4.md) | Error analysis → fix-in-data → retrain |
| 5 | [day-5.md](../archive/tasks/day-5.md) | Ship-ready core + DPO (stretch 1) |
| 6 | [day-6.md](../archive/tasks/day-6.md) | Adversarial eval (stretch 2) + final results table |
| 7 | [day-7.md](../archive/tasks/day-7.md) | Ship & defend (publish dataset, HF model, demo, BrainLift verdict) |

Source of truth is `docs/brainlift.md`; hard ceilings are in `CLAUDE.md`. When a task conflicts with
either, they win.

---

## Shared contract (all days reference this)

Locking these on Day 1–2 makes every later task unambiguous. Anything marked *(decide Day 1)* is a
choice; the rest is fixed.

### Repo layout (target)

```
src/
  common/
    tags.py            # tag constants + wrap/unwrap; the ONE place the tag syntax lives
    schema.py          # Example / Span dataclasses + JSONL (de)serialization + validation
    device.py          # backend (unsloth/hf) + device (cuda/mps/cpu) auto-selection
    io.py              # load/save JSONL, split helpers
  datagen/
    generate.py        # entrypoint: python -m src.datagen.generate --config configs/datagen.yaml
    teacher.py         # frontier-teacher client (distillation) + 2nd-pass verification
    negatives.py       # Presidio Sentence Faker → pattern-type NEGATIVES (email/phone/ID = untagged)
    augment.py         # entity-swap augmentation over real/synthetic carriers
    quality_gate.py    # integrity parser + tag well-formedness + teacher-disagreement drop
  eval/
    behavioral_checks.py  # deterministic: leakage / over-tag / integrity  (pure, no model calls)
    metrics.py            # entity-level precision/recall/F5, leakage rate, consistency
    judge.py              # LLM-as-judge over the 4 rubric dims (0-2 each)
    run.py                # entrypoint: python -m src.eval.run --split eval/hardcases --model <id>
  train/
    dataset.py         # chat template (enable_thinking=False) + completion-only masking
    qlora.py           # entrypoint: python -m src.train.qlora --config configs/train.yaml
    dpo.py             # entrypoint: python -m src.train.dpo  --config configs/dpo.yaml   (Day 5)
  infer.py             # single-passage tag() used by eval + demo
  demo.py              # entrypoint: python -m src.demo  (ambiguous-passage showcase)
configs/
  datagen.yaml  train.yaml  train.mps.yaml  dpo.yaml
eval/
  hardcases/           # QUARANTINED held-out eval set (JSONL). NEVER fed to datagen/train.
  adversarial/         # QUARANTINED break-it set (Day 6).
data/                  # gitignored (see .gitignore): generated/ raw/ splits/
tests/
  test_schema.py  test_behavioral_checks.py  test_quality_gate.py
  test_tags.py    test_no_eval_leakage.py
```

### Tag syntax *(decide Day 1)*

- Primary: `⟨NAME⟩…⟨/NAME⟩` (U+27E8 / U+27E9). Fallback: `@@…##` (GPT-NER style) if the tokenizer
  splits the angle brackets badly. Whichever is chosen lives **only** in `src/common/tags.py`.

### Example schema (JSONL, one object per line)

```json
{
  "id": "essay-000123",
  "register": "essay",                 // essay | dialogue
  "category": "person_vs_place",       // see category list below; "easy" for control items
  "input": "I visited Chelsea last summer.",
  "target": "I visited Chelsea last summer.",     // input with ⟨NAME⟩ tags around person names only
  "spans": [                           // gold spans over the RAW input offsets
    {"start": 10, "end": 17, "text": "Chelsea", "is_name": false}
  ],
  "source": "synthetic_teacher",       // synthetic_teacher | presidio_faker | entity_swap | real_crapii | real_tscc | handbuilt
  "paraphrase_group": "pg-0007",       // shared id across 2-3 rewordings (consistency metric)
  "quarantine": false                  // true only for files under eval/
}
```

Invariant (checked by `schema.py` + `behavioral_checks.py`): **`unwrap(target) == input`** byte-for-byte.

### Ambiguous-case categories (the crux)

`person_vs_eponym` · `person_vs_place` · `person_vs_common` · `first_name_only` · `possessive` ·
`third_party` · `negative_trap` · `adversarial` (Day 6) · `easy` (control).

### The three behavioral checks (deterministic, pure functions)

- **leakage** = any gold `is_name` span left untagged in the model output → set of missed spans.
- **over_tag** = any tagged span that is not a gold `is_name` span → set of false tags.
- **integrity** = `unwrap(model_output) == input`; if false, the item is an automatic FAIL.

### Metrics reported everywhere

Entity-level precision / recall / **F5** (β=5) on NAME; **leakage rate** (fraction of items with ≥1
missed name); **over-tag rate**; **integrity-violation rate**; **consistency** (fraction of
paraphrase groups scored identically). Report base vs. tuned side-by-side with deltas + bootstrap CIs.
