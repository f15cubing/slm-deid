# STATUS — live project state
> The single "what's actually done / in-flight / next" view. Every merged PR updates its line here in
> the same merge (rule in `shipping-changes`). Keep this SKIMMABLE — roll old entries into a CHANGELOG,
> don't append forever.

_Last updated: 2026-07-07 — Day 3 midweek gate MET (first base-vs-tuned numbers: F5 0.19→0.61, leakage halved; over-tagging is the Day-4 data-fix target). Trained locally on MPS._

## Done
- Repo initialized and connected to `origin` (github.com/f15cubing/slm-deid, now public).
- Project scaffold: `README.md`, `.gitignore`, `requirements.txt`.
- Agent workflow: `AGENTS.md` (ceilings + routing), `shipping-changes` + `building-and-testing` skills.
- `docs/plan.md` (one-week build plan) + `docs/agent-workflow-starter-kit.md` (workflow reference).
- `docs/brainlift.md` (BrainLift v3 — source of truth: mandate, scope, DOK 1–4, spiky POVs).
- `docs/tasks/` — per-day specs (day-1..7) + shared repo/schema contract (`docs/tasks/README.md`).
- **[Day 1](tasks/day-1.md) — checkpoint MET.** `src/common/tags.py` (13 passing tests) with tag syntax
  `⟨NAME⟩…⟨/NAME⟩` confirmed against the real Qwen3 tokenizer; `src/common/prompts.py` (system prompt +
  non-thinking serializer); `notebooks/day1_setup.ipynb` + `scripts/day1_cpu_check.py`; `pyproject.toml`.
  Base model (`unsloth/Qwen3-1.7B-unsloth-bnb-4bit`) runs live on Colab T4 in non-thinking mode
  (`Newton…`→`⟨NAME⟩Newton⟨/NAME⟩…`, no `<think>` leak). Scope locked to NAME-only.
  (Optional follow-up: pin `requirements.txt` from the Colab `pip freeze`.)

- **[Day 2](tasks/day-2.md) — checkpoint MET.** Eval harness (behavioral checks + metrics + LLM-judge
  + base-vs-tuned scaffold), data-gen pipeline (teacher + Faker negatives + quality gate + orchestrator),
  training code (`train/{dataset,qlora}`), quarantined `eval/hardcases/` (51) + leakage guard; 67 tests.
  Full generate→train→eval loop verified end-to-end on Colab T4 (`notebooks/day2_smoke.ipynb`).
  Eval-before-training gate satisfied. Base already shows the target gap (over-tags on ambiguous cases).

- **Mac/MPS training backend added** (merged, PR #1). `src/common/device.py` auto-selects
  `unsloth` (CUDA) or `hf` (Apple-Silicon MPS: transformers + PEFT LoRA, `adamw_torch`); adds
  `configs/train.mps.yaml` (base `Qwen/Qwen3-1.7B`, fp16) and Linux-gates `unsloth`/`bitsandbytes` in
  `requirements.txt`. The Colab/Unsloth path is unchanged as the documented fallback. Trade-off: the Mac
  path is plain LoRA on an fp16 base (not 4-bit QLoRA), so its base-vs-tuned numbers re-baseline against
  the Colab 4-bit run rather than reusing them. Smoke-verified on Apple M-series (MPS): a 1-epoch LoRA
  step (finite loss, no NaN) → save adapter → base-vs-tuned eval runs end-to-end on `mps` via a tiny
  hand-built dataset (the `generate` step still needs a teacher API key). New unit tests: `test_device.py`
  (7), `test_qlora_backend.py` (4).

- **[Day 3](tasks/day-3.md) — MIDWEEK GATE MET.** v1 dataset (146 train/16 val; teacher 54% quality-gate
  drop, `eval_leak=0`) → LoRA on MPS (`train_loss 0.131`) → base-vs-tuned on the 51 quarantined hard cases.
  **F5 0.19→0.61, recall 0.19→0.63, leakage 0.41→0.20** (recall/F5 win — SPOV-7 validated), but
  **over-tag 0.10→0.37** (precision cost) so pass-rate is flat. Over-tagging concentrates where v1 data was
  thin/absent (`person_vs_place` had 0 train examples). Full table + honest read: `docs/results.md`. All
  local on Apple MPS.

- **Eval report bootstrap CIs (S3.5) + `docs/results.md` regenerated/corrected** (PR
  `agent/eval-ci-reporting`, high-risk lane). `src/eval/report.py` now renders the Overall + per-category
  tables **from the saved JSON reports** with 95% percentile bootstrap CIs (seeded, offline; older reports
  get `tp/fp/fn` recomputed via `behavioral_checks.check`); `run.py` saves per-item `tp/fp/fn`. Fixed the
  hand-transcribed impossible base `easy` row (recall 0.0 but pass 0.667 → pass **0.000**) and promoted
  integrity (0.039→0.118, ~3×, a hard-ceiling regression) + consistency (0.250→0.312, still poor) into the
  written read. Regression test + CLI guard reject any report where an all-named category has recall 0 with
  pass>0.

## In flight
- **[Day 4] PR `agent/datagen-minpairs-gate` (high-risk, in review)** — data-iteration machinery to fix
  the Day-3 over-tagging (over_tag 0.10→0.37). Adds: matched **minimal-pair** teacher generation
  (`teacher.generate_pair`) + an eval-**disjoint** vocab bank (`src/datagen/vocab.py`) whose hints no
  longer seed eval tokens; a **category-semantics** quality gate (negative_trap⇒0 names; person-vs-*⇒
  intended token present; possessive⇒possessive) so labels are trustworthy; a **token-level
  eval-leakage guard** (`drop_eval_token_overlap`) on top of the passage-level de-leak; a targeted
  `configs/datagen.yaml` recipe (scale knob; ~510 examples @1.0, ~50/50 person/non-person); and
  `docs/error-analysis-v1.md` (S4.1+S4.2). Code+config+analysis only — no generation run, no
  training-config change; leakage guard strengthened, not weakened. 109 tests green.

## Next  — per `docs/tasks/`
- **[Day 4](tasks/day-4.md) — fix in data, not hyperparameters:** generate targeted
  `person_vs_place` / `person_vs_common` / `possessive` / eponym-negative examples (and optionally fold in
  the already-built CRAPII real slice, `src/datagen/real_data.py`) to cut the tuned model's over-tagging
  (0.37) and the integrity regression (0.12), then retrain + re-measure on those categories. Scale v1
  toward 800–2,000. (Bootstrap CIs S3.5 — done; see Done.) Do NOT touch lr/r/epochs to mask the
  over-tagging.

_Note: the prompted base already handled the Day-1 sanity case ("Newton" the person). The real test is the Day-2 hard-cases set (the Newton method, Chelsea the place, first-name-only) — that's where the base is expected to wobble and the fine-tune to hold._
