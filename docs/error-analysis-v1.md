# Error Analysis — SFT v1 (Day 4, S4.1 + S4.2)

_Source of truth for the numbers: `outputs/eval_reports/base-20260707-204355.json` and
`outputs/eval_reports/tuned-20260707-204436.json` (per-item `pass/leaked/over_tagged/integrity_ok`
flags), scored on the 51 quarantined hard cases in `eval/hardcases/hardcases.jsonl`. Overall table
lives in [`results.md`](results.md)._

## What we're explaining

The Day-3 midweek gate was met on the primary axis — fine-tuning tripled F5 (0.190 → 0.613), lifted
recall in **every** category (0.185 → 0.630), and halved leakage (0.412 → 0.196). But it bought that
recall with **precision**: over-tagging **tripled** (over_tag_rate 0.098 → 0.373) and pass-rate went
flat (0.549 → 0.529). This document ranks where the tuned model fails, quotes the concrete cases, and
names the one mode to fix in data.

## Per-category ranking (by over-tagging damage, base → tuned)

Over-tag count = `over_tag_rate × n` (number of items with ≥1 false tag). Ranked by tuned over-tag
count, then rate. Training-data coverage is the count of that category in `data/splits/train.jsonl`
(v1) — the suspected cause.

| rank | category | n | over-tag b→t (count) | recall b→t | v1 train examples |
|---|---|---|---|---|---|
| 1 | **person_vs_common** | 16 | 0.00→0.375 (0→6) | 0.125→0.375 | **4** |
| 2 | **person_vs_place** | 10 | 0.10→0.50 (1→5) | 0.40→0.80 | **0** |
| 3 | **possessive** | 3 | 0.33→0.667 (1→2) | 0.00→1.00 | **1** |
| 4 | person_vs_eponym | 8 | 0.00→0.25 (0→2) | 0.33→0.67 | 40 |
| 5 | negative_trap | 5 | 0.00→0.20 (0→1) | — | 55 (but polluted, see below) |
| 5 | first_name_only | 3 | 0.33→0.33 (1→1) | 0.00→0.67 | 22 |
| 5 | third_party | 3 | 0.33→0.33 (1→1) | 0.33→0.67 | 14 |
| 5 | easy | 3 | 0.33→0.33 (1→1) | 0.00→0.75 | 10 |

**The pattern is unmistakable:** the three categories that regressed most on over-tagging
(person_vs_common, person_vs_place, possessive) are exactly the three with **near-zero v1 training
coverage** (4 / 0 / 1 examples). person_vs_eponym, with 40 training examples, over-tags far less
(0.25) despite being the "hardest" category conceptually. Coverage — not difficulty — predicts the
failure.

## Concrete failing cases (input → tuned output → gold), by failure mode

### Mode A — over-tag: non-person use of an ambiguous token tagged as a person (the dominant mode)

1. **person_vs_place** `place-darwin-city`
   - in:  `Our flight to Darwin was delayed by a tropical storm.`
   - out: `Our flight to ⟨NAME⟩Darwin⟨/NAME⟩ was delayed by a tropical storm.`
   - gold: *(no tags — Darwin is the city)*
2. **person_vs_place** `place-chelsea-place`
   - in:  `Last summer I visited Chelsea and walked along the river all afternoon.`
   - out: `Last summer I visited ⟨NAME⟩Chelsea⟨/NAME⟩ and walked along the river all afternoon.`
   - gold: *(no tags — Chelsea is the neighborhood)*
3. **person_vs_place** `place-jordan-country`
   - in:  `We studied the water scarcity crisis in Jordan for our geography unit.`
   - out: `We studied the water scarcity crisis in ⟨NAME⟩Jordan⟨/NAME⟩ for our geography unit.`
   - gold: *(no tags — Jordan is the country)*
4. **person_vs_place** `place-madison-city`
   - in:  `The conference was held in Madison over spring break.`
   - out: `The conference was held in ⟨NAME⟩Madison⟨/NAME⟩ over spring break.`
   - gold: *(no tags — Madison is the city)*
5. **person_vs_common** `common-grace-concept`
   - in:  `She handled the criticism with grace and kept improving her draft.`
   - out: `She handled the criticism with ⟨NAME⟩grace⟨/NAME⟩ and kept improving her draft.`
   - gold: *(no tags — "grace" is the abstract noun)*
6. **person_vs_common** `common-rose-flower`
   - in:  `The rose in the courtyard bloomed early this year.`
   - out: `The ⟨NAME⟩rose⟨/NAME⟩ in the courtyard bloomed early this year.`
   - gold: *(no tags — "rose" is the flower)*
7. **person_vs_common** `common-baker-job`
   - in:  `The baker down the street sponsored our fundraising bake sale.`
   - out: `The ⟨NAME⟩baker⟨/NAME⟩ down the street sponsored our fundraising bake sale.`
   - gold: *(no tags — "baker" is the occupation)*
8. **possessive (eponymous)** `poss-newton-laws`
   - in:  `Newton's laws describe the motion of classical objects.`
   - out: `⟨NAME⟩Newton⟨/NAME⟩'s laws describe the motion of classical objects.`
   - gold: *(no tags — "Newton's laws" is the eponymous law)*
9. **possessive (eponymous)** `poss-ohms-law`
   - in:  `We verified Ohm's law using three different resistors.`
   - out: `We verified ⟨NAME⟩Ohm's law⟨/NAME⟩ using three different resistors.`
   - gold: *(no tags — eponymous; note the wrong span boundary too)*
10. **person_vs_eponym** `eponym-newton-method-1`
    - in:  `We applied the Newton method to approximate the root of the equation.`
    - out: `We applied the ⟨NAME⟩Newton⟨/NAME⟩ method to approximate the root of the equation.`
    - gold: *(no tags — the Newton method)*
11. **person_vs_eponym (possessive form)** `eponym-newton-method-2`
    - in:  `Using Newton's method, the solver converged after four iterations.`
    - out: `Using ⟨NAME⟩Newton's⟨/NAME⟩ method, the solver converged after four iterations.`
    - gold: *(no tags)*
12. **negative_trap (org)** `neg-org`
    - in:  `The Red Cross visited our school to run a first-aid workshop.`
    - out: `The ⟨NAME⟩Red Cross⟨/NAME⟩ visited our school to run a first-aid workshop.`
    - gold: *(no tags — organization)*
13. **person_vs_place (spurious extra)** `place-florence-city`
    - in:  `The Renaissance art in Florence left a deep impression on the class.`
    - out: `The ⟨NAME⟩Renaissance⟨/NAME⟩ art in ⟨NAME⟩Florence⟨/NAME⟩ left a deep impression on the class.`
    - gold: *(no tags — Florence the city; "Renaissance" is not a name either)*

### Mode B — boundary over-tag: a real name tagged, but the span includes a title/possessive

14. **easy** `easy-dear-johnson`
    - in:  `Dear Mr. Johnson, thank you for reviewing my application.`
    - out: `Dear ⟨NAME⟩Mr. Johnson⟨/NAME⟩, thank you for reviewing my application.`
    - gold: `Dear Mr. ⟨NAME⟩Johnson⟨/NAME⟩, …` (tag `Johnson`, not the title)
15. **third_party** `third-teacher-rivera`
    - in:  `My teacher, Ms. Rivera, said the deadline moved to Monday.`
    - out: `My teacher, ⟨NAME⟩Ms. Rivera⟨/NAME⟩, said the deadline moved to Monday.`
    - gold: `My teacher, Ms. ⟨NAME⟩Rivera⟨/NAME⟩, …`

### Mode C — integrity break / format collapse (secondary; ~12% of items)

16. **person_vs_place** `place-chelsea-person-1`
    - in:  `Chelsea helped me revise my thesis statement after class.`
    - out: `<NAME> helped me revise my thesis statement after class.` (hallucinated literal `<NAME>`, dropped the token → integrity fail)
    - gold: `⟨NAME⟩Chelsea⟨/NAME⟩ helped me revise my thesis statement after class.`
17. **first_name_only** `firstname-liang`
    - in:  `ok Liang i'll send you the notes in a sec`
    - out: `⟨NAME⟩Liang⟨/NAME⟩ ⟨NAME⟩i⟨/NAME⟩ ⟨NAME⟩will⟨/NAME⟩ … ⟨NAME⟩sec⟨/NAME⟩` + a trailing prose "Note:" (tagged every word, then editorialized → integrity fail)
    - gold: `ok ⟨NAME⟩Liang⟨/NAME⟩ i'll send you the notes in a sec`

### Mode D — leakage: a real person's name left untagged (the base model's dominant mode; much reduced after tuning)

18. **person_vs_eponym** `eponym-turing-person`
    - in:  `Turing presented his proof to the seminar on Friday.`
    - out: `Turing presented his proof to the seminar on Friday.` *(missed)*
    - gold: `⟨NAME⟩Turing⟨/NAME⟩ presented his proof to the seminar on Friday.`
19. **person_vs_common** `common-rose-person`
    - in:  `Rose sat with me while I debugged the loop`
    - out: `Rose sat with me while I debugged the loop` *(missed)*
    - gold: `⟨NAME⟩Rose⟨/NAME⟩ sat with me while I debugged the loop`

## The single highest-impact failure mode (S4.2)

**Mode A — over-tagging non-person uses of ambiguous tokens — concentrated in
`person_vs_place`, `person_vs_common`, and `possessive` (eponymous).** It is the mode that (a)
regressed most from base→tuned (over_tag_rate 0.098 → 0.373 overall; +0.40/+0.28/+0.33 in those
three categories), (b) accounts for the largest share of tuned failures (≈16 of the 24 tuned
FAILs carry an over-tag), and (c) directly cancels the recall win, keeping pass-rate flat.

### Data hypothesis (why the model over-tags)

1. **No contrastive coverage.** The tuned model learned "these surfaces are usually names" because
   v1 training had **person_vs_place = 0, person_vs_common = 4, possessive = 1** examples, and — the
   deeper problem — **zero MINIMAL PAIRS**. The eval is built from minimal pairs (the same surface
   used once as a person [tagged] and once as a non-person [untagged]); the training set never once
   showed the model the *untagged* half of such a pair. With no negative-sense exemplar, the model
   generalizes "Rose/Grace/Darwin/Newton's ⇒ tag it."
2. **Untrustworthy labels made the gap invisible.** The `negative_trap` bucket (55 train rows) was
   polluted: 23 of them actually **tag real names** — Faker "Thanks {name}…" mixed rows plus teacher
   rows that tag **Sigmund Freud / Carl Jung / B.F. Skinner** under a "negative" label. So the model's
   "don't-tag" signal was both scarce AND corrupted.
3. **The eval wasn't a clean generalization test.** The teacher's category hints seeded the *exact*
   eval tokens (Newton/Chelsea/Grace/…), so some apparent learning was memorization of the surface,
   not the judgment.

### The fix this points to (implemented in this PR — data, not hyperparameters)

- **Minimal pairs** for person_vs_place / person_vs_common / possessive (+ eponym): every ambiguous
  surface appears once tagged (person) and once **untagged** (non-person) — the contrast the model
  never saw. (`src/datagen/teacher.py::generate_pair`, weighted in `configs/datagen.yaml`.)
- **Category-semantics gate** so labels are trustworthy: `negative_trap` must tag zero names, a
  person-vs-* row must contain its intended token, a `possessive` row must be possessive.
  (`src/datagen/quality_gate.py::check_category_semantics`.)
- **Eval-disjoint vocabulary bank** + a token-level leakage guard so training never re-uses an eval
  surface — restoring the eval as a clean generalization test. (`src/datagen/vocab.py`,
  `src/datagen/generate.py::drop_eval_token_overlap`.)

Retraining on this v2 data and re-measuring the over-tag rate on the same 51 hard cases is the next
step (Day-4 S4.5), with `configs/train.yaml` byte-identical to Day 3.
