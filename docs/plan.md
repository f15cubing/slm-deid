# De-Id SLM — One-Week Build Plan (Mon Jul 6 → Sun Jul 12, 2026)

**Project.** Fine-tune Qwen3-1.7B (QLoRA) to reliably do the one part of de-identification a prompt
can't: **context-sensitive personal-name judgment** — deciding, in context, whether a token is a
person's name or an identically-spelled non-person, every time, without drifting. Pattern types
(email/phone/ID) and output format are handled by regex + constrained decoding in the surrounding
pipeline; the **trained behavior, the dataset, and the eval are all about the judgment core**
(BrainLift v3).

**The one reconciliation that makes this clean.** The assignment's litmus test is *fine-tune vs.
prompt*. In this project, the base model **is** the prompted model, so the assignment's required
**base-vs-tuned** comparison **is** your proof that fine-tuning beats prompting on the hard cases.
One eval, both jobs.

---

## THE GATE — Behavior Spec (write/lock this first)

> Given a passage of educational text (student essay or tutoring-chat turn), the model returns it
> **unchanged except** that every span referring to a real person's name is wrapped in
> `⟨NAME⟩…⟨/NAME⟩`, and **no other span is tagged**. It must (a) tag **every mention** of each
> personal name, (b) leave identically-spelled **non-personal** uses untagged — "Newton" the method,
> "Chelsea" the neighborhood, "Darwin" the city, "Bishop" the chess piece, "Grace" the concept — and
> (c) **add or alter no other text**.
>
> **PASS** = all and only personal-name spans tagged, text otherwise byte-identical to input.
> **FAIL** = any missed name, any non-name tagged, or any other change to the text.

This spec is simultaneously the data-gen rubric, the eval criterion, and the spiky POV (SPOV 8:
*the model's job is judgment, not pattern-detection or formatting*). One target, one context —
exactly what the assignment demands.

*(ADDRESS tagging and consistent surrogate replacement are held back as the "composed behavior"
stretch rung, so the core stays narrow.)*

**Tag syntax decision (Day 1):** primary = `⟨NAME⟩…⟨/NAME⟩` (matches the spec above). `@@…##`
(GPT-NER style) is the fallback if the tokenizer handles it more cleanly — decide after eyeballing a
serialized example. **Decided:** kept `⟨NAME⟩…⟨/NAME⟩` for collision-safety even though it fragments
on Qwen3's byte-level BPE (OPEN=3 / CLOSE=4 tokens → 8 per span, vs `@@…##`'s 3); the round-trip is
lossless so integrity holds. Now pinned by `tests/test_tag_tokenization.py`. A single-token variant
(markers as *added* special tokens) is a v-next A/B on the stretch ladder.

---

## THE CRUX — the hard-cases eval set (this is where the project is won or lost)

If the eval is full of easy names ("Dear Mr. Johnson…"), a prompted base model already nails it and
your fine-tune shows **no delta** — project looks pointless. The delta lives in the **ambiguous**
cases. Build the held-out eval around these categories:

- **Person vs. eponym/method:** Newton, Gauss, Pascal, Turing, Riemann (person? or unit/theorem/method?)
- **Person vs. place:** Chelsea, Darwin, Florence, Jordan, Madison, Devon
- **Person vs. common word:** Grace, Hope, Faith, Mark, Bill, Rose, May, Baker, Bishop
- **First-name-only / buried surname:** "thanks, Sam"; a surname mid-sentence with no title
- **Possessive / inflected:** "Newton's laws" (not a person here) vs. "Sarah's essay" (person)
- **Third parties:** a student naming a friend/teacher/parent (not the author)
- **Negative traps:** sentence-initial capitalized non-names, brands, course titles
- **(stretch/adversarial):** "please don't tag my friend Bob," names inside code/math, typo'd/unicode names

**Test each for:** correct tag, **all** mentions caught, and **consistency across paraphrases**
(run 2–3 rewordings — reliability is the point). Keep this set **hand-built + drawn from real
CRAPII/TSCC**, and **quarantined** from the data-gen pipeline (no leakage from the teacher that makes
training data).

> **Concrete day-by-day tasks with specs live in [`docs/tasks/`](tasks/README.md).** This file is the
> high-level arc; `docs/tasks/day-N.md` has the testable acceptance criteria and checkboxes, and
> `docs/tasks/README.md` holds the shared repo layout + data/eval schema contract.

---

## Arc at a glance

| Day | Date | Focus | Hard checkpoint |
|-----|------|-------|-----------------|
| 1 | Mon Jul 6 | Environment + scope lock + BrainLift confirm | Base model runs & responds; behavior locked |
| 2 | Tue Jul 7 | **Behavior Spec + eval harness + data-gen pipeline** + smoke test | Full loop (gen→train→eval) runs on 50 junk examples |
| 3 | Wed Jul 8 | v1 dataset + first real training + **first base-vs-tuned numbers** | **Midweek gate: base-vs-tuned on the board** |
| 4 | Thu Jul 9 | Error analysis → **fix in data** → retrain | One failure mode resolved via data iteration |
| 5 | Fri Jul 10 | Ship-ready core + stretch rung 1 (**DPO**) | Tuned beats base on spec-adherence & robustness; demo runs |
| 6 | Sat Jul 11 | Stretch rung 2 (**adversarial eval**) + final eval + error analysis | Robustness-under-attack numbers; results table done |
| 7 | Sun Jul 12 | **Ship & defend**: publish dataset, HF model, demo video, BrainLift verdict | Full submission package ready |

---

## Day 1 — Mon Jul 6: Setup, scope lock, BrainLift confirm
- **Environment:** stand up Unsloth + `Qwen/Qwen3-1.7B` (or `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`) on
  your GPU (Colab/Modal/RunPod A100, or a local 24GB card). Run one inference call; confirm it responds.
  *(Update: local Apple-Silicon Macs now train via the `hf`/MPS backend — `configs/train.mps.yaml`; the
  Unsloth/CUDA path stays the fallback. Backend auto-selects by hardware.)*
- **Chat template:** verify **non-thinking mode** (`enable_thinking=False`, empty-`<think>` template)
  and completion-only masking behavior. Serialize one example and eyeball it.
- **Scope lock:** confirm the mandate = context-sensitive NAME judgment (BrainLift v3 already encodes
  this; just check SPOVs match the behavior). Decide the tag syntax (`⟨NAME⟩…⟨/NAME⟩` or `@@…##`
  GPT-NER style).
- **Checkpoint:** base model runs and responds; behavior is locked; SPOVs match target.

## Day 2 — Tue Jul 7: Spec, eval harness, data-gen pipeline, smoke test
**Do the eval BEFORE any training — this is a hard rule.**
- **Finalize the Behavior Spec** (above) — commit it in the repo.
- **Build the eval harness (3 required pieces):**
  1. **LLM-as-judge** scoring each output on the 4 rubric dimensions (spec adherence / robustness /
     task quality / consistency), 0–2 each.
  2. **Behavioral checks** (deterministic): *leakage* = any true personal-name span left untagged;
     *over-tag* = any non-name span tagged; *integrity* = output-minus-tags == input (reject if not).
     Report entity-level recall/precision + leakage rate.
  3. **Base-vs-tuned scaffold** on the same held-out hard-cases scenarios.
- **Assemble the hard-cases eval set** (the crux, above): ~80–150 ambiguous scenarios, hand-built +
  real, quarantined.
- **Build the data-gen pipeline:** distill from a frontier teacher — generate educational passages
  (essay + dialogue registers) that **deliberately contain ambiguous name/non-name tokens**, with
  correct inline tagging. Quality gate: integrity parser + tag well-formedness + a second-pass teacher
  verification (drop disagreements).
- **Smoke test:** push 50 junk examples through generate → QLoRA train → eval, end to end.
- **Checkpoint:** the full loop runs.

## Day 3 — Wed Jul 8: v1 dataset + first real numbers
- **Generate & filter the v1 dataset** (~800–2,000 examples), mixed per the data strategy: distilled
  ambiguous-case passages (the bulk) + Presidio Sentence Faker for pattern-type **negatives** (so the
  model learns *not* to tag emails/phones — those aren't its job) + entity-swap augmentation + a
  **small real slice** (CRAPII/TSCC). Weight heavily toward the hard ambiguous name cases.
- **First real QLoRA run:** r=32/α=32, lr 2e-4, seq 2048, 2–3 epochs, completion-only, non-thinking.
  (Minutes on a 24GB CUDA card; ~1–2 hr locally on Apple-Silicon MPS via `configs/train.mps.yaml`.)
- **First base-vs-tuned eval** on the held-out hard-cases set. Put the numbers on the board.
- **Midweek gate:** base-vs-tuned numbers exist. If the tuned model already clearly beats prompting on
  the ambiguous cases, the core thesis is validated early.

## Day 4 — Thu Jul 9: v2 dataset (iteration) — fix in data, not hyperparameters
- **Error analysis:** where does the tuned model still fail? Likely suspects — first-name-only,
  over-tagging sentence-initial capitals, person-vs-place, dialogue-specific patterns.
- **Fix in DATA:** generate targeted examples for the specific failure mode (more person-vs-place, more
  capitalized-non-name negatives, etc.). **Do not touch hyperparameters to paper over a data problem.**
- **Retrain, re-eval, report the improvement** on the resolved failure mode.
- **Checkpoint:** one specific failure mode resolved via data iteration, visible in the numbers.

## Day 5 — Fri Jul 10: Ship-ready core + Stretch rung 1 (DPO)
- **Confirm the win:** tuned beats base on **spec adherence** and **robustness** (the assignment's
  definition of a win). Freeze a v1 model.
- **Stand up the inference demo** (the ambiguous-passage showcase).
- **DPO (stretch):** build preference pairs (on-spec tagging vs. off-spec: over-tagged / missed / wrong
  boundary), run DPO on top of the SFT model, measure whether spec-adherence sharpens **beyond SFT alone**.
- **Checkpoint:** ship-ready model + demo; DPO delta measured.

## Day 6 — Sat Jul 11: Stretch rung 2 (adversarial) + final eval + error analysis
- **Adversarial/robustness eval:** build the break-it set (embedded instructions "don't tag my friend
  Bob," names in code/math, typo'd/unicode names, messy chat spelling). Report robustness **under
  attack**, not just clean inputs.
- **Final eval:** base vs. SFT vs. DPO on clean + adversarial hard cases; full **results table** across
  the 4 dimensions with base-vs-tuned deltas.
- **Error-analysis paragraph:** where the tuned model still fails, and whether it's a data problem.
- **Begin packaging:** clean the dataset, prep the HF model card + eval harness.

## Day 7 — Sun Jul 12: Ship & defend
- **Publish the dataset** (your real artifact) — with the schema, the ambiguous-case breakdown, and the
  quarantined eval set documented.
- **Push the model to HF Hub** + running inference demo.
- **Finalize BrainLift** with the empirical verdict: *did data→behavior hold? Did the fine-tune beat
  prompting on the hard cases?* This is the resolution of SPOV 7's falsifiable bet — report it honestly,
  win or lose.
- **Record the 3–5 min demo:** feed ambiguous passages ("Newton was frustrated" [tag] vs. "the Newton
  method" [don't]; "I visited Chelsea" [don't] vs. "Chelsea helped me" [tag]) and show the **prompted
  base wobbling/over-tagging while the tuned model holds**.
- **Checkpoint:** full submission package ready.

---

## Gates & traps (the assignment's hard rules, applied here)
- **No training before the eval exists.** The hard-cases eval + behavioral checks are built Day 2,
  before the first real run.
- **Pick a behavior that fails the prompt test.** Ours does — context-sensitive name disambiguation is
  where prompting over-tags "the Newton method" and drifts. The eval is *designed* to expose that.
- **One target, one context.** Personal-name judgment in educational text. ADDRESS/surrogates are
  stretch only.
- **Fix data, not hyperparameters.** Day 4 is explicitly a data-iteration day.
- **Don't chase capability benchmarks.** Measure the target behavior (spec adherence, leakage,
  consistency on ambiguous names) — never trivia accuracy.
- **The falsifiable bet:** the whole re-centering assumes fine-tune > prompt on the hard cases. If Day
  3's numbers show it doesn't, that's a real signal — investigate (data quality first), and if it holds
  up, that's an honest finding worth reporting.

## Final submission package (checklist)
- [ ] **Dataset published** (the real artifact) — ambiguous-case-rich, documented, eval set quarantined
- [ ] **Model on HF Hub** + running inference demo
- [ ] **Eval harness + results table** — base vs. tuned (= prompt vs. fine-tune), 4 dimensions, with deltas
- [ ] **BrainLift** (v3+) — behavior thesis + whether data→behavior held, with evidence
- [ ] **3–5 min demo video** — tuned model doing the ambiguous-name thing the base/prompted model fails at

## Stretch ladder (if the core lands early)
1. **DPO** (Day 5) — preference pairs, sharpen spec adherence beyond SFT.
2. **Adversarial eval** (Day 6) — robustness under attack.
3. **Composed behavior** (if time) — add a second constraint (ADDRESS tagging, or consistent surrogate
   replacement) and show the model holds **both** without degrading name judgment.
4. **Single-token tags A/B** — register `⟨NAME⟩`/`⟨/NAME⟩` as *added* special tokens (1 token each vs
   8 per span on Qwen3 BPE) with `embed_tokens` + `lm_head` in `modules_to_save`; measure malformed-tag
   rate, integrity, and token cost against the current scheme. Tokenization reality is pinned by
   `tests/test_tag_tokenization.py`; current call is collision-safety over efficiency (see Day-1 decision).
