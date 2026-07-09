# `pipeline/` — end-to-end de-identification

The **productionization layer** that wraps the fine-tuned name-judgment model into a full
de-identifier. Deliberately separate from `src/` (model, datagen, eval): **Colab runs stay purely
train/eval**, and this layer only *consumes* a trained `src.infer.Tagger`. It never feeds text back
into training, so it cannot leak the quarantined eval set.

## The division of labor (BrainLift / `docs/plan.md`)

The model spends its capacity only on the judgment a regex can't make — *is this token a person's
name, in context?* Everything else is deterministic:

```
text
 ├─ patterns.detect(text)              # regex: email / phone / SSN / credit-card / IP / URL / ID
 │                                     #   → spans on ORIGINAL offsets   (patterns.py)
 ├─ tagger.tag(text) ─▶ project(...)   # SLM name judgment, projected onto ORIGINAL text
 │                                     #   by char-offset alignment      (project.py)
 └─ merge + render(mode)               # tag | mask | surrogate          (deid.py)
```

## The offset-projection fix (`project.py`)

The model's output contract is "regenerate the passage verbatim, adding only `⟨NAME⟩` tags." On
messy **real** text that copy drifts — collapsed double-spaces, dropped zero-width chars,
normalized newlines, or a repeated token tail on long generations — so strict byte-identity
integrity failed ~100% on the held-out CRAPII probe *even though the name judgment was good*
(recall ~0.88).

`project()` never trusts the model's copy of the text. It takes only the model's **judgment**
(which spans are names), character-aligns the model's plain text to the original with
`difflib.SequenceMatcher`, maps each name span onto original offsets, and drops spans that don't
align (the hallucinated tail). It then re-inserts tags **into the original bytes**, so:

```
tags.unwrap(project_tags(original, any_model_output)) == original   # integrity by construction
```

This removes the integrity failure mode with **no retraining**.

## Render modes (`deid.py`)

| mode        | effect                                                        | integrity-preserving |
|-------------|---------------------------------------------------------------|----------------------|
| `tag`       | inline `⟨NAME⟩…⟨/NAME⟩` / `⟨EMAIL⟩…⟨/EMAIL⟩` …                 | yes (strips to original) |
| `mask`      | replace each span with `[NAME]` / `[EMAIL]` …                 | no (redaction)       |
| `surrogate` | replace with consistent, realistic fakes (`surrogate.py`)     | no (pseudonymization) |

On a name/pattern overlap the deterministic pattern wins (higher precision). Surrogates are seeded
and coreference-preserving: the same value maps to the same fake throughout a document.

## Usage

```bash
# offline demo — heuristic stub tagger (plumbing only, NOT real judgment)
echo "Marcus emailed sam@school.edu and called (415) 555-0132." \
  | python -m pipeline.cli --demo --mode surrogate

# real de-identification with the tuned adapter (once Colab produces it)
python -m pipeline.cli --adapter outputs/sft-v3-mps --mode tag < essay.txt
```

```python
from pipeline import Deidentifier
from src.infer import load_hf_tagger

deid = Deidentifier(load_hf_tagger(adapter="outputs/sft-v3-mps"))
print(deid.deidentify(text, mode="tag").text)
```

The tagger is injected, so the same pipeline serves the base model, the tuned adapter, or a stub —
and the tests run on CPU with no model loaded (`pipeline.stub.HeuristicNameStub`).

## Tests

`tests/test_pipeline_{project,patterns,surrogate,deid}.py` — projection drift / repetition-tail /
integrity invariant, pattern offsets, surrogate consistency, and all three render modes end-to-end.
Run with `make check` (hermetic; regex backend, no spaCy/model download).
