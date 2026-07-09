# STATUS — live project state
> The single "what's actually done / in-flight / next" view. Every merged PR updates its line here in
> the same merge (rule in `shipping-changes`). Keep this SKIMMABLE — roll old entries into a CHANGELOG,
> don't append forever.

_Last updated: 2026-07-09 — merged v3 (authored-teacher data-rebalance) onto the consolidated `main`. v3 now has REAL numbers on MPS bf16: recall 0.44→0.93, consistency 0.13→0.75, leakage→0.04, pass→0.86 (over_tag/integrity held) — see `docs/results.md`→v3, `docs/model-card-v3.md`. The teacher-key blocker is resolved two ways: the in-session AUTHORED teacher (`--provider authored`, NO key), and now the **TrueFoundry LLM Gateway** (`--provider openai` + `OPENAI_BASE_URL`/`TEACHER_MODEL`). Pending: the canonical live-teacher 4-bit QLoRA run on Colab (re-baselines vs MPS bf16). Held-out CRAPII probe shows judgment generalizes (0.88 recall) but byte-identity fails on messy text → span-offset fix in backlog._

## Done
- **TrueFoundry (OpenAI-compatible) teacher gateway supported.** `build_openai_complete` now defaults
  its model from `TEACHER_MODEL` (else `gpt-4o`); the OpenAI SDK already reads `OPENAI_API_KEY` +
  `OPENAI_BASE_URL`, so pointing the teacher at the TrueFoundry LLM Gateway is pure env config
  (`--provider openai`). The v3 notebook's credentials cell documents the path. Unblocks the pending
  live-teacher run without an OpenAI/Anthropic key. Fast lane; teacher client only.
- **v3 training set up for Colab (4-bit QLoRA).** Consolidated `agent/datagen-v2-run` +
  `worktree-testset-review-ui` + `agent/infra-code-quality-loop` onto `main` (124 passed, 2 skipped).
  Added `notebooks/v3_colab_train_eval.ipynb`: clones `main` → generates the v3 dataset from the frozen
  `configs/datagen.yaml` recipe at `scale=3.0` (~1,530 raw → ~800–1k kept; a step up from v2's 268) +
  folds in the committed co-occurrence contrast set + CRAPII slice, all under the same eval-leakage
  guards → trains the **frozen** `configs/train.yaml` QLoRA → base-vs-tuned on the 51 hard cases →
  persists adapter/reports/splits to Drive. Data changes only; hyperparameters frozen (Day-4 rule).
  Blocker to actually generate: the teacher (`gpt-4o`) key returns `billing_not_active` — Colab needs a
  live OpenAI or Anthropic key, else run with `GENERATE=False` on the committed v2 splits.
- **Code-quality loop (`make check` + CI)** — `Makefile` gate (ruff `check` + `format --check` + `pytest`)
  and a GitHub Actions `code-quality` workflow running the same on every push/PR. Made the baseline green:
  `ruff format` on `src/`+`tests/` and excluded exploratory `notebooks/` from the linter (`pyproject.toml`).
  Project skills mirrored to `.claude/skills/` (symlinks into `.cursor/skills/`) so Claude Code discovers
  `shipping-changes` + `building-and-testing` too. CI installs only the CPU deps the hermetic suite uses
  (ruff/pytest/torch/faker). 124 passed, 2 skipped.
- Repo initialized and connected to `origin` (github.com/f15cubing/slm-deid, now public).
- Project scaffold: `README.md`, `.gitignore`, `requirements.txt`.
- Agent workflow: `CLAUDE.md` (ceilings + routing), `shipping-changes` + `building-and-testing` skills.
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

## Done (recent)
- **[v3] data-rebalance retrain + re-eval — MERGED to `main`** (independent-agent review passed; reconciled
  with main's parallel v3 consolidation). Fixes v2's recall/consistency regression **in the data**:
  rebalanced `person_vs_common` to ~50/50 (v2 was 18/38), consolidated eval-disjoint vocab bank (~167
  tokens), scaled 242→927 train; retrained bf16 → `outputs/sft-v3-mps`. **base→tuned: recall 0.185→0.926,
  F5 0.190→0.919, consistency 0.250→0.750, leakage 0.412→0.039, pass 0.549→0.863, integrity→0.000, over_tag
  held 0.137** (CIs separated on recall/F5/leakage). Residual: eponymous-possessive over-tag + one pronoun
  tag. **Caveat:** teacher API was down, so v3 data authored in-session (`src/datagen/author.py`,
  `--provider authored`, now the keyless default on Colab) — no independent verifier pass, template text
  less varied (a frontier-teacher/canonical 4-bit run is the follow-up). Leakage 0 (3 guards + scan). Cards:
  `docs/model-card-v3.md`, `docs/dataset-card-v3.md`.

## In flight
- **[Day 4] v2 retrain + re-eval (branch `agent/datagen-v2-run`, NOT merged)** — trained `sft-v2-mps`
  (LoRA on the CRAPII-augmented 242/26 v2 data, **bf16**; `train_loss 0.0298`, no NaN) and re-ran
  base-vs-tuned on the 51 hard cases. **Day-4 goal met: over_tag 0.37→0.137, integrity 0.118→0.020
  (below base), pass 0.549→0.627**; recall 0.185→0.444, F5 0.190→0.450, leakage 0.412→0.275. Honest
  regressions: **consistency 0.25→0.125**, `person_vs_common` recall flat at 0.125. Methodology note:
  fp16 (the documented MPS default) NaN-diverged on the long CRAPII passages, so v2 is a **bf16
  re-baseline** (bf16 base reproduces the Day-3 fp16 base, so the comparison holds). Model card:
  `docs/model-card-v2.md`; numbers: `docs/results.md` → v2; adapter + provenance: `outputs/sft-v2-mps/`.
  Not yet shipped (needs high-risk-lane review before merge).
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
- **Backlog — span-offset output (architecture fix, deferred; agreed to do later).** The held-out CRAPII
  probe (68 real essays, `scripts/eval_heldout.py`) showed the model's **name judgment generalizes well**
  (whitespace/case-tolerant recall 0.88 / precision 0.82; base 0.21) BUT **strict byte-identity integrity
  fails ~100% on messy real text** — 63/68 whitespace-only diffs (collapsed double-spaces / zero-width
  chars / newlines) + 5/68 long-generation token repetition. Root cause is the "regenerate the whole
  passage verbatim" output format, not the judgment. **Fix (later):** have the model emit span offsets (or
  tag-and-project onto the original text) instead of reproducing the passage, so tags are applied by
  offset and whitespace/length can't drift. This removes the integrity failure mode without retraining.
  Not blocking Colab; tracked here.
- **Backlog (v-next):** single-token tag scheme A/B — `⟨NAME⟩` tags fragment to 8 tokens/span on
  Qwen3 BPE (kept for collision-safety; now pinned by `tests/test_tag_tokenization.py`). Test
  registering the markers as *added* special tokens (1 token each) — see the `docs/plan.md` stretch ladder.

_Note: the prompted base already handled the Day-1 sanity case ("Newton" the person). The real test is the Day-2 hard-cases set (the Newton method, Chelsea the place, first-name-only) — that's where the base is expected to wobble and the fine-tune to hold._
