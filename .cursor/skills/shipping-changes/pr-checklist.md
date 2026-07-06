# PR Checklist & Body Template

## Fast-lane self-review
- [ ] Not touching the judgment-core integrity (eval harness / quarantined eval / data-gen gates) and scoped to one task.
- [ ] On a branch + real PR (never committed to main).
- [ ] Verifiable things verified (tests/lint); result noted in the PR body.
- [ ] If it changes model output, integrity holds (output-minus-tags == input) — spot-checked.
- [ ] Relevant docs updated in THIS PR.
- [ ] On merge, `docs/STATUS.md` line updated.

## Extra gate (high-risk PRs, in addition)
- [ ] **No eval leakage:** touched code cannot feed the quarantined eval set into the teacher, augmentation, or training splits.
- [ ] **Eval-before-train invariant intact:** behavioral checks (leakage / over-tag / integrity) still run and gate.
- [ ] **Data-gen quality gate intact:** integrity parser + tag well-formedness + second-pass teacher verification still enforced.
- [ ] Behavioral-check unit tests pass (at least one leakage case, one over-tag case, one integrity-violation case).
- [ ] "Files touched + merge-difficulty" note in the body.

## PR body template
## What & why  <one paragraph; fast-lane = the 2–3 sentence intent note>
## Area(s) touched  <paths> — high-risk? yes/no
## Docs updated  <files>
## Test evidence  <commands run + results>
## Extra gate (if applicable)  <leakage/eval-before-train/quality-gate proof + test names — or N/A>
