# Agent Workflow Starter Kit

> **What this is.** A portable guide for standing up the multi-agent development workflow used to
> build this project. It tells you what to build, in what order, why each piece exists, and —
> critically — **what to take as-is, what to adapt, and what to leave behind.**
>
> **How this repo applies it.** See §0 — this is a small, single-developer, greenfield ML project, so
> we take the **minimal spine** (§7) plus real hard ceilings drawn from the assignment's gates. We do
> **not** clone the whole thing (no codebase-docs, no submodules, no heavy subagent ledger).

---

## 0. Adapt this to your project first

| Question | This project's answer | Consequence |
|---|---|---|
| **A. Large or unfamiliar codebase?** | **No** — greenfield, small. | Skip `codebase-docs`; the README is enough. |
| **B. Multiple agents / parallel work?** | **Mostly no** — solo one-week build. | Single-branch + self-review is fine; worktrees available for the high-risk lane. |
| **C. A "blast radius" core that must never break?** | **Yes** — the eval harness + the quarantined eval set + data-gen quality gates. | Keep a high-risk lane with an extra gate + different-agent review for those. |

Most new projects answer at least one **no** — so **do not clone the whole thing.** Take the spine
(§7), add the parts your answers demand.

---

## 1. The mental model: three layers

```
Layer 3 — INSTRUCTION FILE (AGENTS.md, always loaded)
   Hard ceilings ("never do X"), product invariants, and ROUTES into the skills.
   Says WHAT MUST HOLD and WHERE TO LOOK — not how.
        │ routes to
        ▼
Layer 2 — PROJECT SKILLS (.cursor/skills/, committed to the repo)
   Project-shaped procedures every agent shares: how we ship a change, how we build & test.
   Compose Layer 1; don't duplicate it.
        │ composes / references by name
        ▼
Layer 1 — GENERIC WORKFLOW SKILLS ("superpowers", user-level)
   Reusable across all projects: brainstorming, writing-plans, TDD,
   verification-before-completion, worktrees, code review, subagent execution, writing-skills.
```

**The golden boundary rule:** project-specific conventions go in the instruction file;
broadly-reusable techniques go in skills; mechanical constraints get automated, not documented.

---

## 2. The core flow (what an agent actually does)

```
IDEA → brainstorming → SPEC → writing-plans → PLAN
     → subagent/executing-plans (per task: fresh subagent → TDD → self-review → task review)
     → CHANGE (branch + worktree) → shipping-changes (pick a tier → build & test → verify →
       real PR → review by tier → merge) → MERGED (+ update STATUS.md in the same merge)
```

Cross-cutting disciplines ride along: `test-driven-development`, `verification-before-completion`,
`systematic-debugging`, and (for large codebases only) `codebase-docs`. Ceremony scales with risk.

---

## 3. The verdict: take / adapt / drop (summary for this project)

**TAKE (ported here):** change-risk tiers; a tiny always-loaded `AGENTS.md` with checkable ceilings;
`STATUS.md` updated on every merge; `building-and-testing` as a skill with a verified/not-verified
marker; the brainstorm→plan→execute pipeline; TDD + verification-before-completion as iron laws;
green-or-revert / never self-merge a high-risk change; "trust posted green + spot-check" for
expensive verification (slow training/eval runs).

**ADAPT (shape kept, contents ours):** the hard ceilings are this project's (eval-before-train,
no eval leakage, fix-data-not-hyperparameters, no capability benchmarks, no unbacked win claims, no
text alteration). The PR checklist's extra gate lists our core's invariants, not Anki's.

**DROP (not inherited):** heavy uniform Day-1 ceremony; process-about-process specs (process is
frozen); `codebase-docs` + `INDEX.md` (small greenfield); `working-with-submodules` (no submodules);
the full subagent ledger/scripts (low task volume — a light dispatch→review→next loop is enough).

---

## 4. Build order (what was set up here)

**Layer 1 — generic skills:** confirmed installed at the user level (`~/.cursor/skills/`):
`using-superpowers`, `brainstorming`, `writing-plans`, `test-driven-development`,
`verification-before-completion`, `systematic-debugging`, `requesting-code-review`,
`receiving-code-review`, `subagent-driven-development`, `executing-plans`, `using-git-worktrees`,
`dispatching-parallel-agents`, `finishing-a-development-branch`, `writing-skills`. Installed once at
the user level — never copied into the repo or edited per project.

**Layer 2 — project skills (created in this repo):**
- `.cursor/skills/shipping-changes/` — `SKILL.md` + `pr-checklist.md` (two tiers; high-risk = judgment-core integrity).
- `.cursor/skills/building-and-testing/` — `SKILL.md` (real build/train/eval commands; verified markers).

**Wired together:** `AGENTS.md` (ceilings + routing) and `docs/STATUS.md` (Done / In-flight / Next,
updated on merge).

---

## 6. Lessons this workflow encodes (so we don't relearn them)

1. **Meta-work recursion is the #1 risk.** Front-load the tuned process, then freeze it. Build the
   product (dataset + model + eval), not the workflow.
2. **Uniform ceremony is a tax on the common case.** Start tiered — full armor on the eval/data-gen
   core, a light path for everything else.
3. **Expensive verification wants a trust protocol, not more reruns.** Trust posted green eval/training
   output; spot-check the ceiling items (leakage / integrity).
4. **"What's done?" must have one home.** `STATUS.md`, updated on merge.
5. **Docs rot toward length.** Keep `STATUS.md` skimmable; roll old entries into a CHANGELOG.
6. **Automate the mechanical, document the judgment.** A regex/CI check for eval-leakage and integrity
   beats "remember to check."

---

## 7. The minimal spine (what this project actually runs)

1. **`AGENTS.md`** — branch + PR always; never commit to `main`; keep the process lean; plus the real
   hard ceilings (eval-before-train, no eval leakage, fix-data-not-hyperparameters, no capability
   benchmarks, no unbacked win claims, no text alteration).
2. **`shipping-changes`** — branch → TDD the change → verify with real command output → PR →
   self-review (or different-agent review for the high-risk core) → merge → update `STATUS.md`.
3. **`docs/STATUS.md`** — Done / In-flight / Next, updated on merge.
4. **The two iron laws** — test-first, and no completion claim without fresh evidence.

Grow the process to meet the project — never the reverse.

---

*Provenance: distilled from the `speedrun/` workflow (`.cursor/skills/{shipping-changes,codebase-docs,
building-and-testing,working-with-submodules}`, the generic superpowers skills, `AGENTS.md`, the
workflow-tuning design specs, and the overnight-run logs), then trimmed to this project's §0 answers.*
