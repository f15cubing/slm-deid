# Project Report: A Small AI Model That Removes Names from Student Writing

*Prepared 2026-07-09 · Current version: v3 · Audience: mixed / semi-technical*

> A plain-language walkthrough of what we've built, how it was trained and tested, and how well it
> works — with just enough method and numbers to judge the results. For the full technical detail see
> `docs/model-card-v3.md`, `docs/dataset-card-v3.md`, and `docs/results.md`.

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
  **zero overlap** between training and test material (a "leakage" check), so the test is honest.

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

We compare **Base** (just *prompting* the general model) against **Tuned** (our fine-tuned model), both
run on identical hardware for a fair comparison.

## Results (v3, on the 51 hard cases)

| What it measures | Plain prompting (base) | Our model (tuned v3) |
|---|---|---|
| **Recall** (catches names it should) | 0.19 | **0.93** |
| **Leakage** (names missed — privacy risk) | 0.41 | **0.04** |
| **Consistency** (stable when reworded) | 0.25 | **0.75** |
| **Over-tagging** (flags non-names) | 0.10 | 0.14 |
| **Overall pass rate** | 0.55 | **0.86** |
| **Integrity** (never alters other text) | rare errors | **0 errors** |

**In plain terms:** plain prompting *misses roughly 4 out of 5 tricky names* — a serious privacy
failure. Our model catches almost all of them, stays consistent across rewordings, and never garbles
the surrounding text. It correctly splits "Newton" the person from the "Newton method," and
"Grace"/"May"/"Rose" the person from the everyday word.

The recall, leakage, and consistency gains are **statistically robust** — their confidence intervals
don't overlap with the base model's, meaning the improvement is real and not measurement noise.

The one cost: over-tagging ticked up slightly (0.10 → 0.14). We consider this the right trade — being
occasionally over-cautious is far safer than leaking a real name.

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
   text. Rebuilding with a frontier teacher is the clear next step to confirm the results hold.
3. **A known weak spot remains:** possessive eponyms like *"Newton's laws"* still get over-tagged
   occasionally, and one pronoun ("She") was tagged by mistake. These are the next targets.
4. **Real-world text test.** On 68 genuine student essays, the model's *name judgment* generalized well
   (name recall ~0.88), but the current output format — having it retype the whole passage — sometimes
   introduces tiny spacing differences. The planned fix is to have it mark name *positions* instead of
   retyping, which removes that issue without retraining. Already scoped as future work.

## Where the project stands

- The **v3 model is trained and evaluated**, and it fixed the v2 regression.
- The work sits on a development branch and is **awaiting an independent review before it's finalized** —
  appropriate for a change this central to a privacy-safety tool.

## The bigger point

The takeaway from the whole effort: **for a narrow, safety-critical task, reliability engineered
through good training data beats raw capability accessed through clever prompting.** A large general
model asked nicely misses most of the hard cases; a small, focused, well-trained model gets them
right — and runs privately on your own hardware.
