# STATUS — live project state
> The single "what's actually done / in-flight / next" view. Every merged PR updates its line here in
> the same merge (rule in `shipping-changes`). Keep this SKIMMABLE — roll old entries into a CHANGELOG,
> don't append forever.

_Last updated: 2026-07-10 — **canonical LIVE-teacher 4-bit QLoRA run landed (gpt551).** The last open follow-up is closed: a live OpenAI-compatible teacher (via the TrueFoundry gateway) + independent verifier generated the v3 data (818/90), and the frozen `configs/train.yaml` 4-bit QLoRA trained on an A100. base→tuned on the 51 hard cases: F5 0.51→0.85, over_tag 0.55→0.16, integrity 0.59→**0.02**, leakage 0.25→0.08, pass 0.35→**0.82**, consistency 0.38→0.56 (`eval_leak=0`, independently re-verified: 0 overlap vs 201 eval inputs). This removes the "authored-data / no-verifier" caveat that qualified every prior number — gpt551 is now the credible canonical line. Honest note: hard-case scores land BELOW the prior authored run (pass 0.82 vs 0.96), most likely because authored templates sit closer to the eval distribution. See `docs/results.md`→gpt551, `docs/model-card-gpt551.md`, `docs/dataset-card-v3.md`. Enabled by a Colab EOS-token library-compat fix (PR #35). Held-out CRAPII probe shows judgment generalizes (0.88 recall) but byte-identity fails on messy text → span-offset fix in backlog._

## Done
- **[gpt551] Canonical live-teacher 4-bit QLoRA run — DONE.** Live OpenAI-compatible teacher (TrueFoundry
  gateway) + independent verifier generated v3 data (818 train / 90 val; drops: 134 verifier-disagreement,
  98 eval-surface-leak, 48 negative-trap-has-name; `eval_leak=0`). Frozen `configs/train.yaml` 4-bit QLoRA
  on an A100 (156 steps/3 epochs, loss 0.47→~2e-4). base→tuned on the 51 hard cases: F5 0.51→**0.85**,
  over_tag 0.55→**0.16**, integrity 0.59→**0.02**, leakage 0.25→**0.08**, pass 0.35→**0.82**, consistency
  0.38→**0.56**; eval-leakage independently re-verified (0 exact + 0 substring overlaps vs all 201
  quarantined eval inputs). Removes the authored-data/no-verifier caveat → the credible canonical line.
  Scores land below the prior authored run (pass 0.82 vs 0.96) — honest read (likely eval-distribution
  proximity of the templates) in `docs/results.md`→gpt551. Cards: `docs/model-card-gpt551.md`,
  `docs/dataset-card-v3.md` (live-teacher section). Reports at `outputs/eval_reports_colab_gpt551/`
  (gitignored); adapter (133 MB) + splits on Drive, not committed. Run enabled by the EOS-token compat
  fix (PR #35).
- **Human review of the eval/val data (seal of approval).** Reviewed item-by-item in
  `scripts/review_ui.py` and sealed to `reviews/*.approved.jsonl`: val 102/102 approved, hard-cases test
  set 50/51 approved (1 denied), co-occurrence set 29/29 approved. The 927-row train split is under
  partial review (in progress). So the held-out test set + validation split are fully human-approved, not
  just machine-gated — recorded in `docs/dataset-card-v3.md` → Human review.
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
- **Held-out-NAMES generalization probe (branch `worktree-heldout-names-testset`, high-risk lane).**
  New quarantined `eval/heldout_names/heldout_names.jsonl` (74 hand-built, 14 fresh ambiguous names —
  Aurora/Bragg/Jackson/Kepler/Snell/Willow/… — verified absent from training + vocab bank + BLOCKLIST +
  existing `eval/hardcases`, guarded by `tests/test_heldout_names_disjoint.py`; built offline by
  `scripts/gen_heldout_names_testset.py`, fresh phrasings, not the author.py templates). Answers "does
  name judgment transfer to names never seen?" — **yes: base→tuned (`outputs/sft-v3-mps`, MPS) recall
  0.083→1.000, precision 0.375→0.947, leakage 0.446→0.000, integrity_viol 0.081→0.000, pass 0.541→0.973**.
  Prompted base misses ~92% of unseen names; tuned catches all. Only residual: 2 person-vs-place over-tags
  (place-as-subject: "Jackson expanded", "Aurora grew") — a data fix, not hyperparameters. Full note:
  `docs/heldout-names-testset.md`.
- **[v3-colab] canonical 4-bit QLoRA run on Colab T4 — DONE (numbers on the board).** Ran
  `notebooks/v3_colab_train_eval.ipynb` end-to-end on a Tesla T4: authored-teacher generation at
  `scale=2.0` → merge + co-occurrence → **924/102, eval_leak=0** (30 surface-overlap candidates dropped
  by the guard pre-training) → 4-bit QLoRA (`unsloth/Qwen3-1.7B-unsloth-bnb-4bit`, frozen
  `configs/train.yaml`, 174 steps/3 epochs, finite loss, no NaN) → `outputs/sft-v3`. **base→tuned:
  precision 0.36→0.93, recall 0.56→0.96, F5 0.54→0.96, leakage 0.24→0.02, over_tag 0.53→0.04, integrity
  0.55→0.00, pass 0.39→0.96, consistency 0.25→0.94** — every hard ceiling held; per-category
  person_vs_{eponym,common} F5→1.00 with over_tag→0. This is the **canonical 4-bit line** (re-baselines
  vs MPS bf16, not a carry-over). Adapter/reports/splits → `MyDrive/slm-deid-v3/`. Numbers:
  `docs/results.md` → v3-colab. Caveat: authored (not live-teacher) data — a live-teacher 4-bit pass is
  the follow-up. `outputs/` is gitignored so only point values are committed (CIs regenerate from the
  Drive reports).
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
- **Engine comparison + API benchmark (branch `worktree-eval-engine-comparisons`, DRAFT PR #36 — high-risk
  lane)** — adds (1) `scripts/eval_frontier.py`: score a frontier API model through the same metrics
  pipeline (scored on hand-built gold, no circularity); (2) `scripts/build_api_bench.py` +
  `benchmarks/api_bench/` (92 quarantined examples, live-teacher, gold-by-construction, `eval_leak=0`
  independently re-verified, guarded by `tests/test_api_bench.py`). **Frontier gpt-4.1** run on all 5 sets:
  on the 51 hardcases the 1.7B tunes are competitive (gpt551 pass 0.82, authored 0.96 vs frontier 0.88;
  tunes beat the frontier on recall — gpt-4.1 is precision-first/under-recalls). Numbers: `docs/eval-engine-comparison.md`.
  Reviewed by a separate agent (fence-handling fix applied). **Pending:** the small-model 3-way
  (base/authored/gpt551) on the 4 non-hardcases sets must run on Colab (4-bit/CUDA-only). Do not self-merge.
- **OOD generalization probe (branch `worktree-ood-probe`, DRAFT PR — high-risk lane)** — new quarantined
  `eval/ood_probe` (36 cases) built by `scripts/build_ood_probe.py`, with surfaces disjoint from **both**
  the eval set and the training banks (guard in-script + `test_vocab`/`test_no_eval_leakage`). Base-vs-tuned
  on `sft-v3-mps` (MPS bf16): **recall 0.05→0.89, pass 0.53→0.89, consistency 0.00→0.90, over-tag flat 0.11**
  — mirrors in-distribution → judgment generalized, not memorized. Perfect eponym/place/first-name on novel
  tokens. Two real residual over-tags (season-word, eponymous-possessive) + a new honorific-boundary quirk.
  See `docs/results.md` → OOD probe. Eval-only, never fed to training; leakage guard green.
- **[Day 6] Adversarial / break-it eval set (branch `worktree-agent-a90e544c94f5773f3`, DRAFT PR, NOT
  merged)** — 40 hand-built scenarios at `eval/adversarial/adversarial.jsonl` (built by
  `scripts/build_adversarial.py`, mirroring `build_hardcases.py`): 34 `adversarial` + 6 `negative_trap`,
  covering injection ("don't tag Bob" / over-tag traps), names-in-code/math (identifier vs person-in-
  comment), unicode/typo/run-together names, messy lowercase chat, negative traps under attack, and
  same-token person-vs-non-person adjacency. Quarantined (`quarantine=true`, `source=handbuilt`),
  physically separate from `eval/hardcases/`, shares no input with it, and is **never fed to the
  teacher/augmentation/training** — leakage guard + vocab-disjointness guard green. High-risk lane:
  needs independent review before merge. 131 tests green.
- **[Backlog→built] `pipeline/` end-to-end de-id pipeline (branch `worktree-deid-pipeline`, draft PR, fast lane)** —
  productionization layer that consumes a trained `src.infer.Tagger`: deterministic pattern PII
  (regex email/phone/SSN/credit-card/IP/URL/ID; Presidio optional) + **tag-by-offset projection**
  (the backlog architecture fix — `difflib`-aligns the model's drifted output onto the ORIGINAL
  bytes so `unwrap(project(...)) == original` **by construction**, killing the integrity failure
  mode with no retraining) + consistent Faker surrogates. Three render modes (`tag`/`mask`/
  `surrogate`) + CLI (`python -m pipeline.cli`). Kept separate from `src/` so Colab stays purely
  train/eval; the layer never feeds text back into training (no leakage path). 33 new tests; full
  `make check` green (166 passed, 4 skipped). Verified end-to-end via the CLI on all three modes.
  Independent review (high-risk) found + fixed a MAJOR projection bug: span projection is now
  per-contiguous-run (bridging only whitespace/zero-width gaps) so a drifted/garbage tag can't
  stretch across unrelated text or coincidentally project onto a real name span (recall-inflation);
  also tightened the ID regex (require a letter — no longer tags bare numbers) and IP octets, and
  made surrogate seeding instance-local.
- **[Eval] projection-based integrity metric (branch `agent/eval-projection-integrity`, stacked on the
  pipeline branch, draft PR, high-risk lane)** — quantifies the pipeline fix on the real eval path:
  `pipeline.project.ProjectingTagger` wraps any tagger so its output is projected onto the input
  before scoring, and `scripts/eval_heldout.py --project` adds a `tuned+proj` row. CPU test proves
  the effect without a model: on drifted output strict scoring gives `integrity_violation_rate=1.0`,
  `recall=0.0`; projection gives `0.0` / `1.0` (judgment recovered, repetition tail dropped). Colab
  run on the CRAPII held-out slice will turn this into published numbers. `make check` green (166
  passed, 4 skipped). Needs independent review (eval harness) before merge.
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
