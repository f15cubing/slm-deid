# Engine comparison — base vs authored vs gpt551 vs frontier

_Cross-engine comparison on the quarantined eval sets. The small models (base / authored / gpt551)
are 4-bit `Qwen3-1.7B` on Colab; the **frontier reference** is `openai-group/gpt-4.1` via the
TrueFoundry gateway, scored by the **same** `behavioral_checks` + metrics against the **same
hand-built gold** (never its own labels — no circularity). The frontier is a **ceiling reference**,
not an apples-to-apples contestant (it is orders of magnitude larger than the 1.7B target)._

## Frontier reference (gpt-4.1) across all four quarantined sets

Run 2026-07-10 via `scripts/eval_frontier.py` (pure API, on the Mac). Reports:
`outputs/eval_bench_frontier/` (gitignored).

| set | n | precision | recall | F5 | leakage | over_tag | integrity viol. | pass | consistency |
|---|---|---|---|---|---|---|---|---|---|
| hardcases | 51 | 1.000 | 0.778 | 0.784 | 0.118 | 0.000 | 0.020 | 0.882 | 0.625 |
| adversarial | 40 | 0.963 | 0.963 | 0.963 | 0.025 | 0.025 | 0.025 | 0.950 | 0.857 |
| heldout_names | 74 | 0.857 | 1.000 | 0.994 | 0.000 | 0.081 | 0.000 | 0.919 | 0.892 |
| ood_probe | 36 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

**Read.** gpt-4.1 is strong and precise — it essentially **never over-tags** (over_tag 0.00 on
hardcases/ood) — but it is **conservative on the hardest cases**: hardcases recall 0.778 / leakage
0.118 means it misses ~1 in 5 real names in the trickiest passages (e.g. first-name-only, ambiguous
common-word names). It is near-perfect on ood_probe and adversarial.

## Head-to-head on the 51 hardcases (all four engines)

The one set scored on every engine. Small-model numbers from the Colab 4-bit runs
(`docs/results.md`); frontier from the row above.

| engine | recall | leakage | over_tag | integrity viol. | pass | consistency |
|---|---|---|---|---|---|---|
| base (prompted 1.7B) | 0.518 | 0.255 | 0.549 | 0.588 | 0.353 | 0.375 |
| **gpt551** (live-teacher tune, 1.7B) | 0.852 | 0.078 | 0.157 | 0.020 | **0.824** | 0.562 |
| **authored** (template tune, 1.7B) | 0.963 | 0.020 | 0.039 | 0.000 | **0.961** | 0.938 |
| frontier (gpt-4.1) | 0.778 | 0.118 | 0.000 | 0.020 | **0.882** | 0.625 |

**Read (honest).**
1. **The tiny fine-tunes are competitive with the frontier on this narrow task.** gpt551 (pass 0.82)
   sits just below gpt-4.1 (0.88); the authored tune (0.96) is *above* it on pass/F5/recall. A 1.7B
   model within ~6 points of a frontier model — and beating it on recall — is the headline.
2. **Different failure shapes.** The frontier is **precision-first** (over_tag 0.00, precision 1.00)
   but **under-recalls** (0.778) — it withholds when unsure. The tunes are **recall-first**
   (0.85–0.96) at some precision cost. For de-id, recall/leakage is the privacy-critical axis, which
   favors the tunes here.
3. **Caveats.** (a) The authored tune's numbers are likely **eval-distribution-flattered** (its
   templates were authored by the same hand that designed the eval categories) — gpt551 (live
   teacher) is the more credible small-model line, and it is the fair one to compare against the
   frontier. (b) n=51, single seed — treat gaps within ~1 CI as ties (see `docs/results.md` for CIs).
   (c) The frontier is a reference ceiling, not a contestant; it also got a documented code-fence
   leniency the small models don't (`scripts/eval_frontier.py`).

## Pending: the 3-way on adversarial / heldout_names / ood_probe (Colab)

gpt551 and the 4-bit authored adapter are CUDA-only, so their numbers on the other three sets must be
produced on Colab (the frontier row above already covers all four). Run there:

```python
AUTHORED = "outputs/sft-v3-colab-authored"   # Drive path to the 4-bit authored adapter
GPT551   = "outputs/sft-v3-gpt551"
for split in ["adversarial", "heldout_names", "ood_probe"]:
    !python -m src.eval.run --split eval/{split} --compare base {AUTHORED} {GPT551} \
        --backend unsloth --report-dir outputs/eval_bench_{split}
```

Drop those numbers in here (or hand me the reports) to complete the 4-engine × 4-set matrix.
