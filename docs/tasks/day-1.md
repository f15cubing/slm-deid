# Day 1 — Mon Jul 6: Setup, scope lock, tag syntax

**Goal.** Base model runs and responds locally; the behavior + tag syntax are locked in code; scope
matches the BrainLift. No training, no data-gen yet.

## Specs (definition of done)
- **S1.1** A single inference call against `Qwen/Qwen3-1.7B` (or `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`)
  returns text in **non-thinking mode** — the serialized prompt contains an empty `<think></think>`
  block (or none), `enable_thinking=False` is set, and the output has no visible chain-of-thought.
- **S1.2** `src/common/tags.py` exists and is the **single source** of the tag syntax: exposes
  `NAME_OPEN`, `NAME_CLOSE`, `wrap(text)`, and `unwrap(text)` such that
  `unwrap(f"{NAME_OPEN}Sarah{NAME_CLOSE}") == "Sarah"`.
- **S1.3** One serialized training-format example (system + user + assistant with inline tags) is
  written to `docs/tasks/artifacts/day1-serialized-example.txt` and eyeballed: prompt tokens vs.
  completion tokens are distinguishable (needed for completion-only masking on Day 3).
- **S1.4** Scope check: a 5-line note confirms the mandate = context-sensitive NAME judgment and that
  SPOV 7/8 in `docs/brainlift.md` match the locked behavior. ADDRESS/surrogates remain stretch-only.

## Tasks
- [ ] Stand up the GPU env (Colab / Modal / RunPod A100, or local 24GB). Record the choice + exact
      `pip freeze` of the training stack into `requirements.txt` (replace the loose pins).
- [ ] Load the base model + tokenizer; run one `tag this passage` prompt end-to-end (S1.1).
- [ ] Verify the chat template: apply with `enable_thinking=False`, print the raw serialized string,
      confirm no thinking block leaks into output (S1.1).
- [ ] Create `src/common/tags.py` with the decided syntax (default `⟨NAME⟩…⟨/NAME⟩`). Add
      `tests/test_tags.py` covering wrap/unwrap round-trip + nested/adjacent tags (S1.2).
- [ ] Decide angle-bracket vs `@@…##` by tokenizing a tagged sentence and checking the tags stay
      intact; record the decision + reason in this file (S1.2).
- [ ] Serialize one full example in the intended training format; save the artifact (S1.3).
- [ ] Write the scope-lock note (S1.4) at the bottom of this file.

## Deliverables
- `src/common/tags.py` + `tests/test_tags.py` (green).
- `docs/tasks/artifacts/day1-serialized-example.txt`.
- Pinned `requirements.txt`; env choice recorded.
- Tag-syntax decision + scope-lock note (below).

## Checkpoint (hard gate)
Base model runs and responds in non-thinking mode; tag syntax locked in `tags.py`; SPOVs match target.

---
### Decisions (fill in)
- Tag syntax chosen: `…`  — reason: `…`
- Env: `…`
### Scope-lock note (fill in)
`…`
