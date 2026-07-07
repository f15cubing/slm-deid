# STATUS — live project state
> The single "what's actually done / in-flight / next" view. Every merged PR updates its line here in
> the same merge (rule in `shipping-changes`). Keep this SKIMMABLE — roll old entries into a CHANGELOG,
> don't append forever.

_Last updated: 2026-07-06 — Day 1 DONE (base model runs on Colab T4, no thinking leak)._

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

## In flight
- (none — ready to start Day 2)

## Next  — per `docs/tasks/`
- **[Day 2](tasks/day-2.md):** eval harness (behavioral checks + metrics + LLM-judge + base-vs-tuned scaffold) BEFORE training; quarantined hard-cases set; data-gen pipeline; 50-example smoke test.
- **[Day 3](tasks/day-3.md):** generate & filter v1 dataset (800–2,000); first QLoRA run; first base-vs-tuned numbers (midweek gate).

_Note: the prompted base already handled the Day-1 sanity case ("Newton" the person). The real test is the Day-2 hard-cases set (the Newton method, Chelsea the place, first-name-only) — that's where the base is expected to wobble and the fine-tune to hold._
