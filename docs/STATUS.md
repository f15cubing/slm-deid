# STATUS — live project state
> The single "what's actually done / in-flight / next" view. Every merged PR updates its line here in
> the same merge (rule in `shipping-changes`). Keep this SKIMMABLE — roll old entries into a CHANGELOG,
> don't append forever.

_Last updated: 2026-07-06 — split plan into per-day task specs under docs/tasks/._

## Done
- Repo initialized and connected to `origin` (github.com/f15cubing/slm-deid).
- Project scaffold: `README.md`, `.gitignore`, `requirements.txt`.
- Agent workflow: `AGENTS.md` (ceilings + routing), `shipping-changes` + `building-and-testing` skills.
- `docs/plan.md` (one-week build plan) + `docs/agent-workflow-starter-kit.md` (workflow reference).
- `docs/brainlift.md` (BrainLift v3 — source of truth: mandate, scope, DOK 1–4, spiky POVs).
- `docs/tasks/` — per-day specs (day-1..7) + shared repo/schema contract (`docs/tasks/README.md`).

## In flight
- (none yet)

## Next  — per `docs/tasks/`
- **[Day 1](tasks/day-1.md):** stand up Qwen3-1.7B; one inference call; verify non-thinking mode; lock tag syntax in `src/common/tags.py`.
- **[Day 2](tasks/day-2.md):** eval harness (behavioral checks + metrics + LLM-judge + base-vs-tuned scaffold) BEFORE training; quarantined hard-cases set; data-gen pipeline; 50-example smoke test.
- **[Day 3](tasks/day-3.md):** generate & filter v1 dataset (800–2,000); first QLoRA run; first base-vs-tuned numbers (midweek gate).
