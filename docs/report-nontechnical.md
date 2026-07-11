# Project Report: A Small AI Model That Removes Names from Student Writing

*Prepared 2026-07-11 · Report version: v2 · Model version: v3 (canonical run "gpt551") · Audience: mixed / semi-technical*

> A plain-language walkthrough of what we've built, how it was trained and tested, and how well it
> works — with just enough method and numbers to judge the results. For the full technical detail see
> **[`docs/final-report.md`](final-report.md)**, `docs/model-card-gpt551.md`, `docs/dataset-card-v3.md`,
> and `docs/results.md`.
>
> *What changed since report v1 (2026-07-09): we replaced the "canonical" model with a more credible one
> (trained on data from a live teacher AI + an independent checker, not in-house templates), and we added
> a comparison against a **frontier** model — a top-tier general AI far larger than ours. Both changes are
> reflected throughout below.*

## The one-sentence version

We fine-tuned a small, open-source AI model to read a piece of student writing and wrap every real
person's **name** in a marker tag — without changing anything else — and it now does this far more
reliably than simply *prompting* a general model to do the same job, and holds its own against a
frontier model many times its size.

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
- **The training dataset (the real deliverable).** ~900 example passages, most of them **matched
  pairs**: the same tricky word used once as a person (tagged) and once as a non-person (left alone) —
  e.g. "Ruby" the person vs. "ruby" the gem, "June" the person vs. "June" the month. Teaching the model
  on pairs like this forces it to learn *context* rather than memorize which words are "name-ish."
  - **How the examples were written (the big upgrade in the final model).** Our best model, called
    **gpt551**, was trained on passages written by a **live teacher AI** and then double-checked by a
    **separate "verifier" AI** that threw out any example the two disagreed on. That's more trustworthy
    than an earlier version ("authored") whose examples we wrote by hand from templates when the teacher
    service was briefly unavailable. Why this matters is under "Which model we recommend."
- **Five held-out test sets.** Deliberately hard scenarios the model **never sees during training**, so
  the scores measure whether it *learned the skill* rather than *memorized answers*. We verified there is
  **zero overlap** between training and test material — an independent check found **0 matches** (exact or
  partial) between our 293 test passages and 1,058 training passages. The five sets stress different
  things:
  1. **Hard cases (51)** — the core ambiguous words (Newton, Chelsea, Grace…).
  2. **Adversarial (40)** — trick attacks: "please don't tag my friend Bob," names hidden in code/math,
     typos, messy lowercase chat.
  3. **Held-out names (74)** — brand-new *people's* names the model never saw.
  4. **Out-of-distribution (36)** — tricky words disjoint from *both* the training and the other tests.
  5. **Natural-prose benchmark (92)** — realistically-worded passages from the live teacher.

  A **human also reviewed the core 51 hard cases and the 102 validation examples one by one** (approving
  50/51 and 102/102) — so the headline scores rest on human-checked examples, not just machine labels.

## How we measure success

The task is graded strictly: a passage **passes** only if *all and only* the real names are tagged and
the rest of the text is byte-for-byte unchanged. The key metrics:

- **Pass rate** — fully-correct passages overall (the headline number).
- **Recall** — of the names that should be tagged, how many were caught. (High recall = few names slip
  through. This is the safety-critical number.)
- **Leakage** — the share of names that were *missed*. (The direct privacy risk; lower is better.)
- **Over-tagging** — how often it tags something that *isn't* a name (lower is better).
- **Consistency** — does it give the same answer when the same case is reworded?
- **Integrity** — did it leave the rest of the text exactly as-is?

We compare four "engines" on the same tests, scored the same way:

- **Base** — just *prompting* the general Qwen3-1.7B model (no fine-tuning). This is the comparison the
  whole project hinges on: does fine-tuning beat prompting?
- **gpt551** — our recommended fine-tune (live-teacher data).
- **Authored** — the earlier fine-tune (hand-written template data).
- **Frontier** — a top-tier general model (gpt-4.1), *far* larger than ours, included as a "how high is
  the ceiling?" reference. It's scored on the same answer key (never its own), so the comparison is fair,
  but it is not an apples-to-apples contestant against a 1.7B model.

## Results

### The headline: fine-tuning beats prompting, everywhere

Pass rate (fully-correct passages) on each test set:

| Test set | Base (just prompting) | **gpt551 (ours)** | Authored (ours) | Frontier (gpt-4.1) |
|---|---|---|---|---|
| Hard cases | 0.39 | **0.82** | 0.90 | 0.88 |
| Adversarial | 0.15 | **0.78** | 0.88 | 0.95 |
| Held-out names | 0.34 | **0.87** | 0.97 | 0.92 |
| Out-of-distribution | 0.28 | **0.78** | 0.89 | 1.00 |
| Natural prose | 0.13 | **0.82** | 0.61 | 0.85 |

**In plain terms:** just *prompting* the base model passes only **13–39%** of these hard passages — it
misses most of the tricky names, which is a serious privacy failure. **Our fine-tuned models pass
61–97%.** The single biggest effect in the whole project is *fine-tuning vs. not* — far bigger than which
data or which hardware we used. The tuned model correctly splits "Newton" the person from the "Newton
method," and "Grace"/"May"/"Rose" the person from the everyday word.

### How close does our tiny model get to a frontier model?

Surprisingly close on the parts that matter:

- **On the hardest cases, our small models actually catch *more* real names than the frontier model.**
  Recall: gpt551 **0.85** and authored **0.96**, vs. the frontier's **0.78**. The big general model is
  *cautious* — when unsure, it withholds — so it quietly **misses about 1 in 5 real names** (the exact
  thing you don't want in de-identification). Our models err the other way, catching more.
- **On realistic natural-sounding text, gpt551 (0.82) is within a few points of the frontier (0.85).**
- The frontier model does clearly win where raw scale helps most — resisting trick "attacks"
  (adversarial) and handling never-seen vocabulary (out-of-distribution). We report that honestly.

The takeaway: a **1.7B model that runs on a laptop is competitive with a frontier model on this narrow,
safety-critical task**, and beats it on the privacy-critical measure (catching names) on the hardest set.

### Which model we recommend — and an honesty note

Notice the **Authored** column often scores *highest* on the tests. We still recommend **gpt551**, and
here's the honest reason: the authored examples were written by the same person who designed the test
categories, so its training text looks a lot like the test — which flatters its scores. The proof is the
last row: on **natural-prose** text that *doesn't* look like the hand-written templates, **authored
collapses to 0.61 while gpt551 holds at 0.82** (right next to the frontier). gpt551 was trained on
realistic live-teacher text, so it stays reliable on real-world writing. So we treat **gpt551 as the
credible model to ship**, even though its test numbers aren't the highest — the higher authored numbers
are partly an artifact of the test looking like its training data.

### What the numbers look like up close (gpt551, on the 51 hard cases)

For the single most-scrutinized test set, our recommended model vs. plain prompting:

| What it measures | Base (prompting) → **gpt551 (tuned)** |
|---|---|
| **Recall** (catches names it should) | 0.52 → **0.85** |
| **Leakage** (names missed — privacy risk) | 0.26 → **0.08** |
| **Over-tagging** (flags non-names) | 0.55 → **0.16** |
| **Consistency** (stable when reworded) | 0.38 → **0.56** |
| **Integrity** (never alters other text) | many errors → **near-zero** |
| **Overall pass rate** | 0.35 → **0.82** |

Every one of these moves in the right direction, and the recall gain is statistically robust (its
confidence interval doesn't overlap the base model's — the improvement is real, not measurement noise).

### Why v3 exists (the iteration story)

This is our third iteration, and the path matters because it shows the method working:

- **v1** got high recall but *over-tagged* aggressively and sometimes altered the text.
- **v2** fixed the over-tagging — but overcorrected into being *too cautious* and started missing names.
- **v3** diagnosed the cause **in the data**: the training examples for common-word names were skewed
  toward "don't tag." We rebalanced them to ~50/50, roughly tripled the vocabulary of tricky words, and
  scaled the dataset up — then, for the final **gpt551** run, regenerated it all with the live teacher +
  verifier. That fixed the regression *without* giving back the earlier gains — all by changing the data.

The deliberate principle here: **fix behavior in the training data, not by tweaking knobs** to mask
symptoms. (We never changed the model's training settings after the week's midpoint — every improvement
came from better data.)

## Honest caveats

We've been careful not to oversell this:

1. **Small test sets, single run.** 36–92 examples per set. The big wins (fine-tune beats base; gpt551
   stays reliable on natural text) are large and solid; some fine-grained category scores rest on as few
   as 3 examples and are noisy. Treat small gaps between the two tunes, or between a tune and the
   frontier, as roughly ties.
2. **One teacher, one verifier.** gpt551's data is much more trustworthy than hand-written templates, but
   it's still a single teacher AI checked by a single verifier — not a consensus of several. The core
   test and validation sets are fully human-reviewed; the larger training set is only partially reviewed.
3. **A known weak spot remains:** possessive eponyms like *"Newton's laws"* still get over-tagged
   occasionally. It's the clearest next target — and the fix is more training data for that case, not a
   settings change.
4. **Real-world messy text.** On 68 genuine student essays, the model's *name judgment* generalized well
   (name recall ~0.88), but the raw output format — having it retype the whole passage — sometimes
   introduces tiny spacing differences. The fix is to mark name *positions* instead of retyping, which
   removes the issue without retraining, and this is **now built** into the end-to-end pipeline (below).

## Where the project stands

- The **v3 model is trained, evaluated, and independently checked**: the full test suite passes (192
  checks), the training/test separation was re-verified (0 overlap), and the recommended **gpt551** model
  is documented as the canonical line.
- We built **five independent test sets** to keep ourselves honest — including trick "attacks," brand-new
  names, never-seen vocabulary, and realistic natural prose — all quarantined from training.
- We added a **frontier-model comparison** so the small model's performance has a meaningful ceiling to
  be judged against.
- An **end-to-end de-identification pipeline** now wraps the model: it combines rule-based handling of
  emails/phones/IDs with the model's name judgment, and applies the tags by *position* so the rest of the
  text is guaranteed untouched.

*What's still to do before final submission is mostly packaging — a live demo, publishing the model and
dataset, and a short demo video. See [`docs/completion-checklist.md`](completion-checklist.md).*

## The bigger point

The takeaway from the whole effort: **for a narrow, safety-critical task, reliability engineered
through good training data beats raw capability accessed through clever prompting.** A large general
model asked nicely misses most of the hard cases (and even a frontier model quietly under-catches names);
a small, focused, well-trained model gets them right — and runs privately on your own hardware.
