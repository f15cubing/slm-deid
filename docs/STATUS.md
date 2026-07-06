# STATUS — live project state
> The single "what's actually done / in-flight / next" view. Every merged PR updates its line here in
> the same merge (rule in `shipping-changes`). Keep this SKIMMABLE — roll old entries into a CHANGELOG,
> don't append forever.

_Last updated: 2026-07-06 — repo initialized; agent workflow + plan scaffolded._

## Done
- Repo initialized and connected to `origin` (github.com/f15cubing/slm-deid).
- Project scaffold: `README.md`, `.gitignore`, `requirements.txt`.
- Agent workflow: `AGENTS.md` (ceilings + routing), `shipping-changes` + `building-and-testing` skills.
- `docs/plan.md` (one-week build plan) + `docs/agent-workflow-starter-kit.md` (workflow reference).

## In flight
- (none yet)

## Next  — per `docs/plan.md`
- **Day 1:** stand up Unsloth + Qwen3-1.7B; one inference call; verify non-thinking mode; lock tag syntax.
- **Day 2:** finalize Behavior Spec; build eval harness (LLM-judge + behavioral checks + base-vs-tuned scaffold); assemble quarantined hard-cases eval set; build data-gen pipeline; 50-example smoke test end-to-end.
- **Day 3:** generate & filter v1 dataset; first real QLoRA run; first base-vs-tuned numbers (midweek gate).
