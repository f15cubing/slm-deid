# Working in this repo (agent rules)

This repo fine-tunes **Qwen3-1.7B (QLoRA)** for **context-sensitive personal-name judgment** in
educational text — the judgment core of de-identification. Read `docs/plan.md` (the one-week arc and
the locked behavior spec) and `docs/STATUS.md` (current state) before doing anything.

## To ship a change
Invoke the `shipping-changes` skill. Every change goes on a branch and ships as a real pull request —
never commit to `main`. Ceremony scales with the change-risk tier:
- **High-risk lane** (the eval harness, the quarantined eval set, the data-gen quality gates, anything
  that could leak eval data into training): worktree + extra gate + review by a DIFFERENT agent (a
  fresh review subagent is fine in a solo context). Never self-merge.
- **Fast lane** (everything else — data-gen tweaks, demo, docs, training config within an existing
  loop): self-review against the fast-lane checklist; design gate is a 2–3 sentence intent note in the
  PR body. When in doubt, use the high-risk lane.

## Build, test
Use the `building-and-testing` skill for all build / run / train / eval commands.

## Keep the process lean
- **Process is frozen.** Do not write new skills / process / working-rules specs unless something is
  actively blocking a feature. Build the product (dataset + model + eval), not the workflow.
- **Link, don't restate.** If a fact already lives in `docs/plan.md`, reference it — don't copy it.

## Hard ceilings (do not violate)
These come straight from the assignment's gates. Each is checkable.

- **Never train before the eval exists.** No training run happens until the hard-cases eval set +
  deterministic behavioral checks (leakage / over-tag / integrity) are committed and runnable.
  *Checked: `docs/STATUS.md` records the eval harness as Done before any training PR merges.*
- **Never leak the eval set into data generation or training.** The quarantined hard-cases eval is
  physically separate and never fed to the teacher, augmentation, or training splits.
  *Checked: eval files live in a quarantined path; a check flags overlap between eval and training text.*
- **Never touch hyperparameters to paper over a data problem.** Failure modes are fixed by generating
  targeted data, not by tuning lr/r/epochs to mask them (Day 4 is a data-iteration day).
- **Never chase capability benchmarks.** Only the target behavior is measured — spec adherence,
  leakage rate, precision/recall on ambiguous names, consistency across paraphrases. No trivia accuracy.
- **Never claim a base-vs-tuned win without the numbers.** Every "tuned beats base" statement is backed
  by fresh eval output on the held-out hard-cases set, reported honestly (win or lose).
- **Never alter the passage text.** The behavior is byte-identical input except for `⟨NAME⟩…⟨/NAME⟩`
  tags; integrity (output-minus-tags == input) is a reject condition in both data-gen and eval.
