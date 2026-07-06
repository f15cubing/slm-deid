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
- [x] Create `src/common/tags.py` with the decided syntax (`⟨NAME⟩…⟨/NAME⟩`). Add `tests/test_tags.py`
      covering wrap/unwrap round-trip + well-formedness + span offsets (S1.2). **13/13 green locally.**
- [x] Write prompt + serializer code: `src/common/prompts.py` (`SYSTEM_PROMPT`, `build_messages`,
      `build_training_messages`, `serialize` with `enable_thinking=False`) (S1.1/S1.3 code).
- [x] Prep the GPU notebook `notebooks/day1_setup.ipynb` (load model → inference → showcase →
      serialize + tag tokenizer test → write artifact).
- [x] Write the scope-lock note (S1.4) — below.
- [x] **Confirmed `⟨NAME⟩` markers survive tokenization** — via the real `Qwen/Qwen3-1.7B` tokenizer
      (CPU): OPEN=3 tokens, CLOSE=4 tokens, both round-trip exactly; full tagged sentence round-trips
      and `unwrap(decoded)==raw`. Angle-bracket syntax **kept** (no `@@…##` fallback needed) (S1.2).
- [x] **Serialized one full example** through the real chat template with `enable_thinking=False`;
      committed `docs/tasks/artifacts/day1-serialized-example.txt`. It shows the **non-thinking empty
      `<think></think>` block**, confirming the template behaves as required (S1.3).
      (Run locally via `PYTHONPATH=. python scripts/day1_cpu_check.py --no-generate`.)
- [ ] **(GPU / Colab)** Run one `tag` generation end-to-end and confirm the model *responds* with no
      `<think>` leak (S1.1). Local CPU attempt blocked: the 3.4GB weight download stalls under the
      network sandbox — this belongs on Colab (`notebooks/day1_setup.ipynb`) where the template +
      tokenizer are already proven identical.
- [ ] **(GPU / Colab)** Record the exact `pip freeze` of the training stack into `requirements.txt`.

## Deliverables
- `src/common/tags.py` + `tests/test_tags.py` (green).
- `docs/tasks/artifacts/day1-serialized-example.txt`.
- Pinned `requirements.txt`; env choice recorded.
- Tag-syntax decision + scope-lock note (below).

## Checkpoint (hard gate)
Base model runs and responds in non-thinking mode; tag syntax locked in `tags.py`; SPOVs match target.

---
### Decisions
- **Tag syntax:** `⟨NAME⟩…⟨/NAME⟩` (U+27E8 / U+27E9). Reason: single non-ASCII codepoints, so they
  never collide with ASCII `<`/`>` a student might type in prose or code (`a < b`, `List<Name>`).
  Lives only in `src/common/tags.py`. **CONFIRMED** against the real Qwen3 tokenizer (OPEN=3 / CLOSE=4
  tokens, exact round-trip) — kept, no `@@…##` fallback needed.
- **Base model:** `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` (fastest QLoRA path, fits 24GB / Colab).
- **Env:** code + tokenizer on Mac (no CUDA); model load + inference via `notebooks/day1_setup.ipynb`
  on Colab/RunPod. `requirements.txt` to be pinned from the GPU env's `pip freeze`.
- **Teacher (Day 2):** provider configurable (Anthropic/OpenAI); no key wired yet.
- **Real data:** none yet — start synthetic-only + hand-built eval.

### Scope-lock note (S1.4)
Mandate = **context-sensitive personal-NAME judgment** in educational text: tag every span that
refers to a real person's name in `⟨NAME⟩…⟨/NAME⟩`, leave identically-spelled non-person uses
untagged, and change no other character (integrity: `unwrap(output) == input`). Pattern-type PII
(email/phone/ID/date/URL) and output format are **out of scope for the model** — owned by
regex/checksums + constrained decoding in the surrounding pipeline. **ADDRESS tagging and surrogate
generation are stretch-only** and deliberately excluded from the core. This matches `docs/brainlift.md`
SPOV 7 (fine-tune earns its keep exactly on the un-regexable judgment core, benchmarked vs. prompting)
and SPOV 8 (the model's job is judgment, not pattern-detection or formatting).
