# Submission runbook — run the GPU steps tonight (Colab)

_Companion to [`completion-checklist.md`](archive/completion-checklist.md). Everything code/writing here is
done and green (202 passed); this is the copy-paste sequence to turn the finished artifacts into the
submission package. Steps 1–3 need the Colab GPU + the Drive adapter; Step 4 is already committed._

> **Prefer a ready-made notebook?** Open [`notebooks/submission_publish.ipynb`](../notebooks/submission_publish.ipynb)
> in Colab (GPU runtime), edit the three paths in cell 0, and run top to bottom. The cells below are
> the same sequence in copy-paste form.

**Prereqs:** the `sft-v3-gpt551` adapter + v3 splits on Drive (from the canonical run), a HuggingFace
account + write token, and a screen recorder for the video.

---

## 0. Colab setup (one cell)

```python
!git clone -b worktree-ship-submission https://github.com/f15cubing/slm-deid.git
%cd slm-deid
!pip install -q -r requirements.txt
from google.colab import drive; drive.mount('/content/drive')

# Point these at your Drive copy of the canonical run:
ADAPTER = "/content/drive/MyDrive/slm-deid-gpt551/sft-v3"      # canonical gpt551 LoRA adapter dir
SPLITS  = "/content/drive/MyDrive/slm-deid-gpt551/splits"      # train.jsonl + val.jsonl
REPO_ID = "YOURNAME/slm-deid-name-judgment"                    # your Hub id
```

## 1. Run the demo → this is what you screen-record (checklist #5)

```python
!python -m src.demo --adapter "$ADAPTER"
```

Shows the prompted base vs. the fine-tune side-by-side on the ambiguous passages
("Newton was frustrated" → tag vs. "the Newton method" → don't; "I visited Chelsea" → don't vs.
"Chelsea helped me" → tag; first-name-only; …). **Narrate the disagreement rows** — that's the
whole thesis on screen. 3–5 min: read each passage, point out where the base over-tags / misses and
the tune holds, then cut to the base-vs-tuned numbers (below / `docs/final-report.md`).

## 2. Push the model + card to the Hub (checklist #2)

```python
# Log in — paste a WRITE token (huggingface.co/settings/tokens) into the box.
# (huggingface-cli is deprecated; this stores the token for the push scripts.)
from huggingface_hub import login; login()
# Dry-run first — verifies the eval-leakage guard passes, uploads nothing:
!python scripts/push_to_hub.py --adapter "$ADAPTER" --repo-id "$REPO_ID" --dry-run
# Real push:
!python scripts/push_to_hub.py --adapter "$ADAPTER" --repo-id "$REPO_ID"
```

Uploads the adapter weights + `MODEL_CARD.md` (as the repo README). Guard refuses if the adapter dir
contains anything that looks like eval data.

## 3. Push the dataset to the Hub (checklist #1) — hard-ceiling guarded

```python
# Dry-run first — asserts 0 quarantine rows + 0 overlap vs the 201 eval inputs:
!python scripts/push_dataset.py --splits-dir "$SPLITS" --repo-id "$REPO_ID" --dry-run
# Real push:
!python scripts/push_dataset.py --splits-dir "$SPLITS" --repo-id "$REPO_ID"
```

Publishes `train.jsonl` + `val.jsonl` + `docs/dataset-card-v3.md`. The script **exits non-zero and
uploads nothing** if any split row is `quarantine=true` or duplicates a quarantined eval input — the
`tests/test_no_eval_leakage.py` guarantee, re-checked at publish time.

> If your Drive splits are the authored v3 (924/102) rather than the gpt551 live-teacher splits
> (818/90), point `--splits-dir` at whichever you're publishing and say which in the dataset card.
> The in-repo `data/splits` also passes the guard if you'd rather publish those.

## 4. BrainLift verdict (checklist #4) — already done

No action. The empirical verdict resolving SPOV-7 ("did fine-tuning beat prompting on the hard
cases?" — yes) is committed in [`brainlift.md`](brainlift.md) → *Empirical verdict [v3]*.

---

## Submission links to collect

- [ ] HF **model** page: `https://huggingface.co/<REPO_ID>`
- [ ] HF **dataset** page: `https://huggingface.co/datasets/<REPO_ID>`
- [ ] Demo **video** (3–5 min) link
- [ ] Repo: `https://github.com/f15cubing/slm-deid` (eval harness, results, BrainLift verdict)

Drop these into the README header when you have them.
