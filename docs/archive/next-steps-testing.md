# Next steps — testing & benchmarking

_Prepared 2026-07-11 · Companion to [`docs/final-report.md`](../final-report.md). This is the "what to test
and benchmark next" backlog: experiments that would strengthen or stress the current result, ordered by
value-per-effort. Everything here respects the hard ceilings in `CLAUDE.md` — in particular, **fixes go
in the data, never the frozen hyperparameters**, and **nothing here feeds the quarantined eval sets into
training.**_

## Where we are (so these have a baseline)

The canonical **gpt551** 4-bit QLoRA tune beats the prompted base decisively across all 5 quarantined
sets (pass 0.13–0.39 → 0.61–0.97), is competitive with frontier gpt-4.1 on the natural-prose sets, and
out-recalls it on hardcases. Suite green (192 passed), leakage independently re-verified (0 overlap).
The open questions below are about **tightening statistical confidence, closing residual failure modes,
and widening the evidence base** — not about whether the core thesis holds.

---

## Tier 1 — highest value, low effort (do these first)

### 1.1 Multi-seed / variance bars on the headline sets
**Why:** every set is n=36–92, single seed. Documented run-to-run variance (authored hardcases 0.96 vs
0.90) currently forces "treat gaps within a CI as ties." **Do:** re-run base + gpt551 + authored on
`hardcases`, `adversarial`, `api_bench` with ≥3 decoding seeds; report mean ± std alongside the bootstrap
CIs. **Payoff:** turns "likely a tie" into a defensible statement and confirms the api_bench
authored↓/gpt551↑ divergence is not a seed artifact. **Cost:** eval-only, hours on Colab.

### 1.2 Reconcile backend provenance in the matrix  ⟵ partially resolved 2026-07-11
**Status:** the `heldout_names` `authored` cell was **confirmed** (not just suspected) to be a bf16 MPS
carry-over: it is byte-identical to the `sft-v3-mps` run in `docs/heldout-names-testset.md`, and a
4-bit-Colab eval of `sft-v3-colab-authored` on `eval/heldout_names` is **absent from every saved report**
— it was never run. The docs are now corrected to label that one cell as bf16-MPS (dagger footnotes in
`eval-engine-comparison.md` and `final-report.md`); the value is no longer presented as a 4-bit result.
**Remaining (needs Colab/CUDA — 4-bit is not runnable on Mac/MPS):** produce the genuine 4-bit number and
replace the row. One eval pass:
```
python -m src.eval.run \
  --split eval/heldout_names \
  --compare base outputs/sft-v3-colab-authored \
  --backend unsloth \
  --report-dir outputs/eval_reports_heldout_authored_4bit
```
Then regenerate the table (`python -m src.eval.report base=… tuned=…`), drop the † footnotes, and update
STATUS. Also re-check the `ood`/`heldout` **base** rows, which differ across docs for the same MPS-vs-4bit
reason. **Payoff:** a single-provenance matrix with no footnotes.

### 1.3 LLM-judge as a reported dimension (not just code)
**Why:** `src/eval/judge.py` scores the 4 rubric dimensions (spec adherence / robustness / task quality /
consistency, 0–2 each) but the headline tables are the deterministic behavioral checks only. **Do:** run
the judge on base/gpt551/frontier outputs for hardcases + api_bench and publish the 4-dimension table
next to the behavioral metrics. **Payoff:** completes the "4 dimensions with deltas" the plan asked for.
**Cost:** low; needs a judge API key.

---

## Tier 2 — closes known residual failure modes (data iteration)

### 2.1 `possessive` contrast set (the persistent weak spot)
**Why:** `possessive` is the unmoved category across every run — recall 1.0 but over_tag/integrity both
0.333 (n=3), and it carries the lone integrity violation. "Newton's laws" (eponym possessive, don't tag)
vs "Sarah's essay" (person possessive, tag). **Do:** generate a targeted matched-pair possessive slice
(≥40 pairs) via the live teacher + verifier, retrain gpt551, re-measure. **Payoff:** the clearest
remaining data-fixable win. **Constraint:** data-only; keep the frozen config.

### 2.2 `person_vs_common` recall lift
**Why:** recall is flat at 0.75 on this category — the tune's gain there is over-tag reduction, not
recall. **Do:** add more person-side common-word examples (Grace/Hope/May/Rose as *people*, varied
registers) so the model stops under-tagging real people with common-word names. **Payoff:** lifts the
one category that didn't improve on recall.

### 2.3 Person-vs-place "place-as-subject" over-tags
**Why:** the held-out-names probe showed the only residual over-tags were places acting as sentence
subjects ("Jackson expanded", "Aurora grew"). **Do:** targeted negatives where a place name is the
grammatical subject. **Payoff:** removes a named, reproducible over-tag pattern.

### 2.4 Span-offset / projected-output as the headline path
**Why:** on 68 real CRAPII essays, judgment generalized (recall 0.88) but strict byte-identity failed
~100% from whitespace drift in the "retype the passage" format. The `pipeline/` offset-projection fix is
built (merged), but the *model's own* output path is still verbatim-regeneration. **Do:** either (a)
publish the projected path (`scripts/eval_heldout.py --project`, `ProjectingTagger`) as the headline
integrity number on a Colab CRAPII run, or (b) train a variant that emits span offsets directly. **Payoff:**
turns "integrity fails on messy text" into "integrity holds by construction on messy text." **Note:**
option (a) is eval/pipeline only; option (b) is a new output format (bigger change).

---

## Tier 3 — widen the evidence base (new benchmarks / harder tests)

### 3.1 A real held-out benchmark from CRAPII/TSCC (not synthetic)
**Why:** the literature's central warning (Enache: synthetic F1 0.99 in-domain → 0.33 on real) means the
honest test is *messy real text*. `api_bench` is live-teacher but still synthetic. **Do:** build a small
(~100) quarantined benchmark of hand-labeled real CRAPII essay excerpts / TSCC turns, guarded like the
others, and score all four engines on it. **Payoff:** the single most credibility-moving addition — a
real-text number. **Constraint:** DUA/credentialing for the source data; keep it quarantined and
human-labeled.

### 3.2 Multi-teacher / multi-verifier consensus data
**Why:** gpt551 is one teacher + one verifier. **Do:** generate a slice with a second teacher model and a
2-of-3 verifier vote; measure whether consensus labels shift the tune's numbers. **Payoff:** removes the
"single teacher" caveat; tests label robustness.

### 3.3 Frontier panel (not just gpt-4.1)
**Why:** the ceiling reference is a single frontier model. **Do:** add ≥1 more frontier model
(e.g. a Claude and a Gemini tier) through the same `scripts/eval_frontier.py` pipeline on the same gold.
**Payoff:** a more honest "how close is the 1.7B to *frontier models in general*" rather than to one.

### 3.4 Consistency-under-paraphrase at scale
**Why:** consistency is measured on the small paraphrase groups in `hardcases`. **Do:** generate 5–10
paraphrases per hard case (quarantined, eval-only) and report consistency as agreement-rate across the
group. **Payoff:** consistency is the reliability claim; measure it with more power.

---

## Tier 4 — stretch experiments (planned but unbuilt)

### 4.1 DPO (stretch rung 1 — genuinely unbuilt)
**Why:** the plan's rung 1. Build preference pairs (on-spec vs over-tagged / missed / wrong-boundary),
run DPO on top of the SFT model, and measure whether spec-adherence sharpens **beyond SFT alone**. **Do:**
`data/splits/dpo.jsonl` + `configs/dpo.yaml` + `src/train/dpo.py`; report the DPO delta honestly (win or
lose). **Constraint:** framed as a training-method stretch, **not** a hyperparameter tweak to mask a data
problem. **Payoff:** the one unbuilt planned capability; a clean "does preference-tuning add over SFT"
result.

### 4.2 Single-token tag scheme A/B (stretch rung 4)
**Why:** `⟨NAME⟩` fragments to 8 tokens/span on Qwen3 BPE. **Do:** register the markers as *added*
special tokens (1 token each, `embed_tokens`+`lm_head` in `modules_to_save`) and A/B malformed-tag rate,
integrity, and token cost vs the current scheme. **Payoff:** cheaper decoding + possibly fewer malformed
tags; tokenization reality is already pinned by `tests/test_tag_tokenization.py`.

### 4.3 Composed behavior (stretch rung 3)
**Why:** currently name-only; ADDRESS tagging + consistent surrogate replacement live in the
deterministic `pipeline/`, not as *trained* behavior. **Do:** add a second trained constraint (ADDRESS)
and show the model holds **both** without degrading name judgment. **Payoff:** demonstrates the method
scales to composed constraints.

---

## Guardrails for all of the above

- **Never train before the eval exists** for any new capability (new behavior ⇒ new quarantined eval
  first).
- **Never leak eval into data-gen/training** — every new benchmark gets a leakage + vocab-disjointness
  guard in CI before it is cited.
- **Fix failure modes in data, not hyperparameters** — `configs/train.yaml` stays frozen.
- **Report honestly** — every "beats X" claim is backed by fresh eval output with CIs, win or lose.
