---
name: shipping-changes
description: Use when making any code change that will merge, when about to commit/push or open a pull request, or when reviewing another agent's PR.
---

# Shipping Changes

Every change ships on a branch (never `main`) as a real pull request. Ceremony rides on the
change-risk tier.

## Which lane? (the high-risk test)
A PR is **high-risk** if it touches the **judgment core's integrity**:
- the eval harness (LLM-as-judge, behavioral checks, base-vs-tuned scaffold),
- the **quarantined hard-cases eval set** (any risk of leaking it into training),
- the data-gen **quality gates** (integrity parser, tag well-formedness, teacher verification),
- anything that could let eval text bleed into the training splits.

Everything else is **fast lane** (data-gen prompt tweaks, augmentation, training config within an
existing loop, the demo, docs). When in doubt → high-risk.

## Fast lane (most changes)
1. Branch `agent/<area>-<task>` off latest `main` (worktree optional).
2. Read `docs/plan.md` + relevant code; make the change, scoped to ONE task; TDD where it applies.
3. Update docs in the same change.
4. Verify what's verifiable (tests/lint for touched files) — use `verification-before-completion`.
5. Open a PR; the design gate is a 2–3 sentence intent note in the body.
6. Self-review against `pr-checklist.md`; merge; update `docs/STATUS.md` in the same merge.

## High-risk lane
Same as fast lane, plus: worktree required; the **extra gate** in `pr-checklist.md`; and a
**different agent** (a fresh review subagent in a solo context) reviews and merges. NEVER self-merge.
Trust the builder's posted green output and spot-check the ceiling items rather than rerunning blind.

## Compose (reference by name, don't duplicate)
`building-and-testing`, `using-git-worktrees`, `verification-before-completion`,
`requesting-code-review` / `receiving-code-review`, `finishing-a-development-branch`.

## Red flags — STOP
| Thought | Reality |
|---|---|
| "I'll commit straight to main." | No — branch + PR, always. |
| "I'll self-review my eval-harness change." | No — high-risk, different agent. |
| "I'll just feed a couple eval examples into training." | NEVER — that's a hard ceiling (leakage). |
| "I'll bump lr/epochs to fix this failure." | No — fix it in data (Day 4 rule). |
| "Fast lane, so skip the PR / skip docs." | No — still a real PR, scoped, docs + STATUS line in the same PR. |
