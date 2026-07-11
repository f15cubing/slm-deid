# Project Report: A Small AI Model That Removes Names from Student Writing

*Prepared 2026-07-09 · Current version: v3 · Audience: mixed / semi-technical*

> **⚠️ Update (2026-07-11) — read this first.** This plain-language report was written on 2026-07-09 and
> is now partly superseded. Two things changed since:
> 1. **A more credible "canonical" model exists.** This report treats the *authored* (in-house template)
>    tune as the version to ship. We have since trained the same recipe on data from a **live teacher AI +
>    an independent verifier** (called **gpt551**), which is the more trustworthy model for real-world
>    text — its scores on hand-written test templates are a little *lower*, but that's because the
>    authored tune was flattered by test material that looked like its own training data. **gpt551 is now
>    the recommended line.**
> 2. **We now also compare against a frontier model** (a top-tier general model, gpt-4.1). Our tiny 1.7B
>    tune is *competitive with it* on natural text and actually *catches more real names* on the hardest
>    cases (the frontier model is cautious and misses ~1 in 5).
>
> For the current, complete picture — base vs. our two tunes vs. the frontier model, across five test
> sets — read **[`docs/final-report.md`](final-report.md)**. The plain-language walkthrough below is still
> accurate about *the problem, the method, and the iteration story*; just read its "which model ships"
> and results claims through the update above.

> A plain-language walkthrough of what we've built, how it was trained and tested, and how well it
> works — with just enough method and numbers to judge the results. For the full technical detail see
> `docs/final-report.md`, `docs/model-card-gpt551.md`, `docs/dataset-card-v3.md`, and `docs/results.md`.

## The one-sentence version

We fine-tuned a small, open-source AI model to read a piece of student writing and wrap every real
person's **name** in a marker tag — without changing anything else — and it now does this far more
reliably than simply *prompting* a general model to do the same job.

## What problem are we solving?

When schools or researchers want to share student essays and tutoring chats, they first have to strip
out personal information (names, emails, phone numbers, IDs) so no student can be identified. This is
**de-identification**.

Most of that work is easy for a computer: emails, phones, and ID numbers follow predictable patterns,
so simple rules ("find anything shaped like an email") handle them. We deliberately **do not** use the
AI model for those — deterministic rules do it better and cheaper.

The one part rules *can't* do is **judging names in context**:

- **"Newton"** — a person, or the "Newton method" in a math essay?
- **"Chelsea"** — a person, a London neighborhood, or a football club?
- **"Grace," "Rose," "May," "Bishop"** — a person, or a virtue, a flower, a month, a chess piece?

A rule can't tell these apart, and a general-purpose chatbot does it inconsistently — it over-flags,
misses some, and changes its answer when you reword the sentence. **Reliable name judgment is the
entire point of our model**, and it's where 100% of the effort goes.

## What we actually built

- **The model.** A compact, open-source base model — *Qwen3-1.7B* (1.7 billion parameters; small
  enough to run privately on a laptop, including an Apple Mac, with no data sent to the cloud). We then
  **fine-tuned** it for this one narrow task using **LoRA**, a lightweight technique that trains a small
  add-on ("adapter") instead of rewriting the whole model — fast and cheap.
- **The training dataset (the real deliverable).** ~930 example passages, most of them **matched
  pairs**: the same tricky word used once as a person (tagged) and once as a non-person (left alone) —
  e.g. "Ruby" the person vs. "ruby" the gem, "June" the person vs. "June" the month. Teaching the model
  on pairs like this forces it to learn *context* rather than memorize which words are "name-ish."
- **A held-out test set.** 51 deliberately hard scenarios the model **never sees during training**, so
  the scores measure whether it learned the skill rather than memorized answers. We verified there is
  **zero overlap** between training and test material (a "leakage" check), so the test is honest. A
  **human also reviewed all 51 cases one by one and signed off on them** (approving 50, flagging 1 for a
  second look) — so the scores rest on human-checked examples, not just machine-generated labels. The
  validation set (102 cases) was likewise fully human-approved.

## How we measure success

The task is graded strictly: a passage **passes** only if *all and only* the real names are tagged and
the rest of the text is byte-for-byte unchanged. The key metrics:

- **Recall** — of the names that should be tagged, how many were caught. (High recall = few names slip
  through. This is the safety-critical number.)
- **Leakage** — the share of names that were *missed*. (The direct privacy risk; lower is better.)
- **Over-tagging** — how often it tags something that *isn't* a name (lower is better).
- **Consistency** — does it give the same answer when the same case is reworded?
- **Integrity** — did it leave the rest of the text exactly as-is?
- **Pass rate** — fully-correct passages overall.

We compare **Base** (just *prompting* the general model) against **Tuned** (our fine-tuned model), always
on identical hardware so the comparison is fair.

## Results (v3, on the 51 hard cases)

We trained the finished v3 model on the **exact same dataset** in **two setups**, and tested both — so
the win can't be an artifact of one machine:

- **The 4-bit deployment version** — the model *compressed* to 4-bit so it's small and fast enough to
  run on a modest cloud GPU (or a laptop). This is the version you'd actually ship.
- **A full-precision version on an Apple Mac** — our reproducibility cross-check, run entirely offline
  on Apple silicon.

| What it measures | 4-bit deployment · base → **tuned** | Full-precision (Mac) · base → **tuned** |
|---|---|---|
| **Recall** (catches names it should) | 0.56 → **0.96** | 0.19 → **0.93** |
| **Leakage** (names missed — privacy risk) | 0.24 → **0.02** | 0.41 → **0.04** |
| **Consistency** (stable when reworded) | 0.25 → **0.94** | 0.25 → **0.75** |
| **Over-tagging** (flags non-names) | 0.53 → **0.04** | 0.10 → 0.14 |
| **Overall pass rate** | 0.39 → **0.96** | 0.55 → **0.86** |
| **Integrity** (never alters other text) | many errors → **0** | rare errors → **0** |

**In plain terms:** in *both* setups, plain prompting misses most of the tricky names — a serious
privacy failure — while our fine-tuned model catches almost all of them, stays consistent across
rewordings, and never garbles the surrounding text. It correctly splits "Newton" the person from the
"Newton method," and "Grace"/"May"/"Rose" the person from the everyday word.

**The 4-bit deployment version is actually the stronger one** — higher recall, consistency, and pass
rate, and its over-tagging *drops* to 0.04. So the version we'd ship is also the best-performing; we
treat its numbers as the headline and keep the Mac run for reproducibility. (The two "base" columns
differ because the 4-bit and full-precision versions start from slightly different copies of the general
model — so each column is read against its own base, not across the two.)

The recall, leakage, and consistency gains are **statistically robust** — their confidence intervals
don't overlap with the base model's, meaning the improvement is real and not measurement noise. The only
place the Mac run pays a small cost is over-tagging (0.10 → 0.14) — being occasionally over-cautious is
far safer than leaking a real name — and even that is gone in the 4-bit deployment version (→ 0.04).

### Why v3 exists (the iteration story)

This is our third iteration, and the path matters because it shows the method working:

- **v1** got high recall but *over-tagged* aggressively and sometimes altered the text.
- **v2** fixed the over-tagging — but overcorrected into being *too cautious* and started missing names
  (recall fell to 0.44, consistency to 0.13).
- **v3** diagnosed the cause **in the data**: the training examples for common-word names were
  2-to-1 skewed toward "don't tag." We rebalanced them to ~50/50, roughly tripled the vocabulary of
  tricky words, and scaled the dataset ~3.8× (to 927 training passages). That fixed the regression
  *without* giving back the earlier gains — all by changing the data, not the model's settings.

The deliberate principle here: **fix behavior in the training data, not by tweaking knobs** to mask
symptoms.

## Honest caveats

We've been careful not to oversell this:

1. **Small test set.** 51 hard cases, single run. The headline wins (recall, leakage, consistency) are
   solid; some fine-grained category scores rest on as few as 3 examples and are noisy.
2. **The training data was written in-house.** We normally generate examples by distilling a top-tier
   "teacher" AI, but that service was unavailable during this build, so we authored the examples from
   templates. They're clean and correctly labeled, but **less linguistically varied** than real messy
   text. Rebuilding with a frontier teacher is the clear next step to confirm the results hold. (Note:
   the *test* and *validation* sets have been fully human-reviewed; the larger training set is only
   partially reviewed so far.)
3. **A known weak spot remains:** possessive eponyms like *"Newton's laws"* still get over-tagged
   occasionally, and one pronoun ("She") was tagged by mistake. These are the next targets.
4. **Real-world text test.** On 68 genuine student essays, the model's *name judgment* generalized well
   (name recall ~0.88), but the raw output format — having it retype the whole passage — sometimes
   introduces tiny spacing differences. The fix is to mark name *positions* instead of retyping, which
   removes that issue without retraining — and this is **now built** into the end-to-end pipeline (see
   "Where the project stands"), so the shipped tool no longer has this failure mode.

## Where the project stands

- The **v3 model is trained and evaluated on both setups** (the 4-bit deployment version and the
  full-precision Mac cross-check), and it fixed the v2 regression. All of this work is now **reviewed and
  merged** into the main line.
- We've since added three independent **stress tests** to keep ourselves honest, all quarantined from
  training: an **out-of-distribution probe** (tricky words the model never saw as names — it still
  scored ~0.89), a **held-out-names probe** (brand-new people's names), and an **adversarial "break-it"
  set** (40 scenarios designed to trip it up, including prompt-injection traps).
- An **end-to-end de-identification pipeline** now wraps the model: it combines the rule-based handling
  of emails/phones/IDs with the model's name judgment, and applies the tags by *position* so the rest of
  the text is guaranteed untouched — the planned fix for the spacing issue noted above, now built.

## The bigger point

The takeaway from the whole effort: **for a narrow, safety-critical task, reliability engineered
through good training data beats raw capability accessed through clever prompting.** A large general
model asked nicely misses most of the hard cases; a small, focused, well-trained model gets them
right — and runs privately on your own hardware.
